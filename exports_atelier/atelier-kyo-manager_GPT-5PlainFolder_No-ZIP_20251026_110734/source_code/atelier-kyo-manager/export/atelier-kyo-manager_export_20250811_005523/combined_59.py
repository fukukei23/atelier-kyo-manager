
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\asyncio\session.py ===
# ext/asyncio/session.py
# Copyright (C) 2020-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

import asyncio
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import engine
from .base import ReversibleProxy
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import util
from ...orm import close_all_sessions as _sync_close_all_sessions
from ...orm import object_session
from ...orm import Session
from ...orm import SessionTransaction
from ...orm import state as _instance_state
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec


if TYPE_CHECKING:
    from .engine import AsyncConnection
    from .engine import AsyncEngine
    from ...engine import Connection
    from ...engine import CursorResult
    from ...engine import Engine
    from ...engine import Result
    from ...engine import Row
    from ...engine import RowMapping
    from ...engine import ScalarResult
    from ...engine.interfaces import _CoreAnyExecuteParams
    from ...engine.interfaces import CoreExecuteOptionsParameter
    from ...event import dispatcher
    from ...orm._typing import _IdentityKeyType
    from ...orm._typing import _O
    from ...orm._typing import OrmExecuteOptionsParameter
    from ...orm.identity import IdentityMap
    from ...orm.interfaces import ORMOption
    from ...orm.session import _BindArguments
    from ...orm.session import _EntityBindKey
    from ...orm.session import _PKIdentityArgument
    from ...orm.session import _SessionBind
    from ...orm.session import _SessionBindKey
    from ...sql._typing import _InfoType
    from ...sql.base import Executable
    from ...sql.dml import UpdateBase
    from ...sql.elements import ClauseElement
    from ...sql.selectable import ForUpdateParameter
    from ...sql.selectable import TypedReturnsRows

_AsyncSessionBind = Union["AsyncEngine", "AsyncConnection"]

_P = ParamSpec("_P")
_T = TypeVar("_T", bound=Any)


_EXECUTE_OPTIONS = util.immutabledict({"prebuffer_rows": True})
_STREAM_OPTIONS = util.immutabledict({"stream_results": True})


class AsyncAttrs:
    """Mixin class which provides an awaitable accessor for all attributes.

    E.g.::

        from __future__ import annotations

        from typing import List

        from sqlalchemy import ForeignKey
        from sqlalchemy import func
        from sqlalchemy.ext.asyncio import AsyncAttrs
        from sqlalchemy.orm import DeclarativeBase
        from sqlalchemy.orm import Mapped
        from sqlalchemy.orm import mapped_column
        from sqlalchemy.orm import relationship


        class Base(AsyncAttrs, DeclarativeBase):
            pass


        class A(Base):
            __tablename__ = "a"

            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str]
            bs: Mapped[List[B]] = relationship()


        class B(Base):
            __tablename__ = "b"
            id: Mapped[int] = mapped_column(primary_key=True)
            a_id: Mapped[int] = mapped_column(ForeignKey("a.id"))
            data: Mapped[str]

    In the above example, the :class:`_asyncio.AsyncAttrs` mixin is applied to
    the declarative ``Base`` class where it takes effect for all subclasses.
    This mixin adds a single new attribute
    :attr:`_asyncio.AsyncAttrs.awaitable_attrs` to all classes, which will
    yield the value of any attribute as an awaitable. This allows attributes
    which may be subject to lazy loading or deferred / unexpiry loading to be
    accessed such that IO can still be emitted::

        a1 = (await async_session.scalars(select(A).where(A.id == 5))).one()

        # use the lazy loader on ``a1.bs`` via the ``.awaitable_attrs``
        # interface, so that it may be awaited
        for b1 in await a1.awaitable_attrs.bs:
            print(b1)

    The :attr:`_asyncio.AsyncAttrs.awaitable_attrs` performs a call against the
    attribute that is approximately equivalent to using the
    :meth:`_asyncio.AsyncSession.run_sync` method, e.g.::

        for b1 in await async_session.run_sync(lambda sess: a1.bs):
            print(b1)

    .. versionadded:: 2.0.13

    .. seealso::

        :ref:`asyncio_orm_avoid_lazyloads`

    """

    class _AsyncAttrGetitem:
        __slots__ = "_instance"

        def __init__(self, _instance: Any):
            self._instance = _instance

        def __getattr__(self, name: str) -> Awaitable[Any]:
            return greenlet_spawn(getattr, self._instance, name)

    @property
    def awaitable_attrs(self) -> AsyncAttrs._AsyncAttrGetitem:
        """provide a namespace of all attributes on this object wrapped
        as awaitables.

        e.g.::


            a1 = (await async_session.scalars(select(A).where(A.id == 5))).one()

            some_attribute = await a1.awaitable_attrs.some_deferred_attribute
            some_collection = await a1.awaitable_attrs.some_collection

        """  # noqa: E501

        return AsyncAttrs._AsyncAttrGetitem(self)


@util.create_proxy_methods(
    Session,
    ":class:`_orm.Session`",
    ":class:`_asyncio.AsyncSession`",
    classmethods=["object_session", "identity_key"],
    methods=[
        "__contains__",
        "__iter__",
        "add",
        "add_all",
        "expire",
        "expire_all",
        "expunge",
        "expunge_all",
        "is_modified",
        "in_transaction",
        "in_nested_transaction",
    ],
    attributes=[
        "dirty",
        "deleted",
        "new",
        "identity_map",
        "is_active",
        "autoflush",
        "no_autoflush",
        "info",
    ],
)
class AsyncSession(ReversibleProxy[Session]):
    """Asyncio version of :class:`_orm.Session`.

    The :class:`_asyncio.AsyncSession` is a proxy for a traditional
    :class:`_orm.Session` instance.

    The :class:`_asyncio.AsyncSession` is **not safe for use in concurrent
    tasks.**.  See :ref:`session_faq_threadsafe` for background.

    .. versionadded:: 1.4

    To use an :class:`_asyncio.AsyncSession` with custom :class:`_orm.Session`
    implementations, see the
    :paramref:`_asyncio.AsyncSession.sync_session_class` parameter.


    """

    _is_asyncio = True

    dispatch: dispatcher[Session]

    def __init__(
        self,
        bind: Optional[_AsyncSessionBind] = None,
        *,
        binds: Optional[Dict[_SessionBindKey, _AsyncSessionBind]] = None,
        sync_session_class: Optional[Type[Session]] = None,
        **kw: Any,
    ):
        r"""Construct a new :class:`_asyncio.AsyncSession`.

        All parameters other than ``sync_session_class`` are passed to the
        ``sync_session_class`` callable directly to instantiate a new
        :class:`_orm.Session`. Refer to :meth:`_orm.Session.__init__` for
        parameter documentation.

        :param sync_session_class:
          A :class:`_orm.Session` subclass or other callable which will be used
          to construct the :class:`_orm.Session` which will be proxied. This
          parameter may be used to provide custom :class:`_orm.Session`
          subclasses. Defaults to the
          :attr:`_asyncio.AsyncSession.sync_session_class` class-level
          attribute.

          .. versionadded:: 1.4.24

        """
        sync_bind = sync_binds = None

        if bind:
            self.bind = bind
            sync_bind = engine._get_sync_engine_or_connection(bind)

        if binds:
            self.binds = binds
            sync_binds = {
                key: engine._get_sync_engine_or_connection(b)
                for key, b in binds.items()
            }

        if sync_session_class:
            self.sync_session_class = sync_session_class

        self.sync_session = self._proxied = self._assign_proxied(
            self.sync_session_class(bind=sync_bind, binds=sync_binds, **kw)
        )

    sync_session_class: Type[Session] = Session
    """The class or callable that provides the
    underlying :class:`_orm.Session` instance for a particular
    :class:`_asyncio.AsyncSession`.

    At the class level, this attribute is the default value for the
    :paramref:`_asyncio.AsyncSession.sync_session_class` parameter. Custom
    subclasses of :class:`_asyncio.AsyncSession` can override this.

    At the instance level, this attribute indicates the current class or
    callable that was used to provide the :class:`_orm.Session` instance for
    this :class:`_asyncio.AsyncSession` instance.

    .. versionadded:: 1.4.24

    """

    sync_session: Session
    """Reference to the underlying :class:`_orm.Session` this
    :class:`_asyncio.AsyncSession` proxies requests towards.

    This instance can be used as an event target.

    .. seealso::

        :ref:`asyncio_events`

    """

    @classmethod
    def _no_async_engine_events(cls) -> NoReturn:
        raise NotImplementedError(
            "asynchronous events are not implemented at this time.  Apply "
            "synchronous listeners to the AsyncSession.sync_session."
        )

    async def refresh(
        self,
        instance: object,
        attribute_names: Optional[Iterable[str]] = None,
        with_for_update: ForUpdateParameter = None,
    ) -> None:
        """Expire and refresh the attributes on the given instance.

        A query will be issued to the database and all attributes will be
        refreshed with their current database value.

        This is the async version of the :meth:`_orm.Session.refresh` method.
        See that method for a complete description of all options.

        .. seealso::

            :meth:`_orm.Session.refresh` - main documentation for refresh

        """

        await greenlet_spawn(
            self.sync_session.refresh,
            instance,
            attribute_names=attribute_names,
            with_for_update=with_for_update,
        )

    async def run_sync(
        self,
        fn: Callable[Concatenate[Session, _P], _T],
        *arg: _P.args,
        **kw: _P.kwargs,
    ) -> _T:
        '''Invoke the given synchronous (i.e. not async) callable,
        passing a synchronous-style :class:`_orm.Session` as the first
        argument.

        This method allows traditional synchronous SQLAlchemy functions to
        run within the context of an asyncio application.

        E.g.::

            def some_business_method(session: Session, param: str) -> str:
                """A synchronous function that does not require awaiting

                :param session: a SQLAlchemy Session, used synchronously

                :return: an optional return value is supported

                """
                session.add(MyObject(param=param))
                session.flush()
                return "success"


            async def do_something_async(async_engine: AsyncEngine) -> None:
                """an async function that uses awaiting"""

                with AsyncSession(async_engine) as async_session:
                    # run some_business_method() with a sync-style
                    # Session, proxied into an awaitable
                    return_code = await async_session.run_sync(
                        some_business_method, param="param1"
                    )
                    print(return_code)

        This method maintains the asyncio event loop all the way through
        to the database connection by running the given callable in a
        specially instrumented greenlet.

        .. tip::

            The provided callable is invoked inline within the asyncio event
            loop, and will block on traditional IO calls.  IO within this
            callable should only call into SQLAlchemy's asyncio database
            APIs which will be properly adapted to the greenlet context.

        .. seealso::

            :class:`.AsyncAttrs`  - a mixin for ORM mapped classes that provides
            a similar feature more succinctly on a per-attribute basis

            :meth:`.AsyncConnection.run_sync`

            :ref:`session_run_sync`
        '''  # noqa: E501

        return await greenlet_spawn(
            fn, self.sync_session, *arg, _require_await=False, **kw
        )

    @overload
    async def execute(
        self,
        statement: TypedReturnsRows[_T],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[_T]: ...

    @overload
    async def execute(
        self,
        statement: UpdateBase,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> CursorResult[Any]: ...

    @overload
    async def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[Any]: ...

    async def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Result[Any]:
        """Execute a statement and return a buffered
        :class:`_engine.Result` object.

        .. seealso::

            :meth:`_orm.Session.execute` - main documentation for execute

        """

        if execution_options:
            execution_options = util.immutabledict(execution_options).union(
                _EXECUTE_OPTIONS
            )
        else:
            execution_options = _EXECUTE_OPTIONS

        result = await greenlet_spawn(
            self.sync_session.execute,
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )
        return await _ensure_sync_result(result, self.execute)

    @overload
    async def scalar(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Optional[_T]: ...

    @overload
    async def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any: ...

    async def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any:
        """Execute a statement and return a scalar result.

        .. seealso::

            :meth:`_orm.Session.scalar` - main documentation for scalar

        """

        if execution_options:
            execution_options = util.immutabledict(execution_options).union(
                _EXECUTE_OPTIONS
            )
        else:
            execution_options = _EXECUTE_OPTIONS

        return await greenlet_spawn(
            self.sync_session.scalar,
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    @overload
    async def scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[_T]: ...

    @overload
    async def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]: ...

    async def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]:
        """Execute a statement and return scalar results.

        :return: a :class:`_result.ScalarResult` object

        .. versionadded:: 1.4.24 Added :meth:`_asyncio.AsyncSession.scalars`

        .. versionadded:: 1.4.26 Added
           :meth:`_asyncio.async_scoped_session.scalars`

        .. seealso::

            :meth:`_orm.Session.scalars` - main documentation for scalars

            :meth:`_asyncio.AsyncSession.stream_scalars` - streaming version

        """

        result = await self.execute(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )
        return result.scalars()

    async def get(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
    ) -> Union[_O, None]:
        """Return an instance based on the given primary key identifier,
        or ``None`` if not found.

        .. seealso::

            :meth:`_orm.Session.get` - main documentation for get


        """

        return await greenlet_spawn(
            cast("Callable[..., _O]", self.sync_session.get),
            entity,
            ident,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
        )

    async def get_one(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
    ) -> _O:
        """Return an instance based on the given primary key identifier,
        or raise an exception if not found.

        Raises :class:`_exc.NoResultFound` if the query selects no rows.

        ..versionadded: 2.0.22

        .. seealso::

            :meth:`_orm.Session.get_one` - main documentation for get_one

        """

        return await greenlet_spawn(
            cast("Callable[..., _O]", self.sync_session.get_one),
            entity,
            ident,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
        )

    @overload
    async def stream(
        self,
        statement: TypedReturnsRows[_T],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncResult[_T]: ...

    @overload
    async def stream(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncResult[Any]: ...

    async def stream(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncResult[Any]:
        """Execute a statement and return a streaming
        :class:`_asyncio.AsyncResult` object.

        """

        if execution_options:
            execution_options = util.immutabledict(execution_options).union(
                _STREAM_OPTIONS
            )
        else:
            execution_options = _STREAM_OPTIONS

        result = await greenlet_spawn(
            self.sync_session.execute,
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )
        return AsyncResult(result)

    @overload
    async def stream_scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncScalarResult[_T]: ...

    @overload
    async def stream_scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncScalarResult[Any]: ...

    async def stream_scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncScalarResult[Any]:
        """Execute a statement and return a stream of scalar results.

        :return: an :class:`_asyncio.AsyncScalarResult` object

        .. versionadded:: 1.4.24

        .. seealso::

            :meth:`_orm.Session.scalars` - main documentation for scalars

            :meth:`_asyncio.AsyncSession.scalars` - non streaming version

        """

        result = await self.stream(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )
        return result.scalars()

    async def delete(self, instance: object) -> None:
        """Mark an instance as deleted.

        The database delete operation occurs upon ``flush()``.

        As this operation may need to cascade along unloaded relationships,
        it is awaitable to allow for those queries to take place.

        .. seealso::

            :meth:`_orm.Session.delete` - main documentation for delete

        """
        await greenlet_spawn(self.sync_session.delete, instance)

    async def merge(
        self,
        instance: _O,
        *,
        load: bool = True,
        options: Optional[Sequence[ORMOption]] = None,
    ) -> _O:
        """Copy the state of a given instance into a corresponding instance
        within this :class:`_asyncio.AsyncSession`.

        .. seealso::

            :meth:`_orm.Session.merge` - main documentation for merge

        """
        return await greenlet_spawn(
            self.sync_session.merge, instance, load=load, options=options
        )

    async def flush(self, objects: Optional[Sequence[Any]] = None) -> None:
        """Flush all the object changes to the database.

        .. seealso::

            :meth:`_orm.Session.flush` - main documentation for flush

        """
        await greenlet_spawn(self.sync_session.flush, objects=objects)

    def get_transaction(self) -> Optional[AsyncSessionTransaction]:
        """Return the current root transaction in progress, if any.

        :return: an :class:`_asyncio.AsyncSessionTransaction` object, or
         ``None``.

        .. versionadded:: 1.4.18

        """
        trans = self.sync_session.get_transaction()
        if trans is not None:
            return AsyncSessionTransaction._retrieve_proxy_for_target(
                trans, async_session=self
            )
        else:
            return None

    def get_nested_transaction(self) -> Optional[AsyncSessionTransaction]:
        """Return the current nested transaction in progress, if any.

        :return: an :class:`_asyncio.AsyncSessionTransaction` object, or
         ``None``.

        .. versionadded:: 1.4.18

        """

        trans = self.sync_session.get_nested_transaction()
        if trans is not None:
            return AsyncSessionTransaction._retrieve_proxy_for_target(
                trans, async_session=self
            )
        else:
            return None

    def get_bind(
        self,
        mapper: Optional[_EntityBindKey[_O]] = None,
        clause: Optional[ClauseElement] = None,
        bind: Optional[_SessionBind] = None,
        **kw: Any,
    ) -> Union[Engine, Connection]:
        """Return a "bind" to which the synchronous proxied :class:`_orm.Session`
        is bound.

        Unlike the :meth:`_orm.Session.get_bind` method, this method is
        currently **not** used by this :class:`.AsyncSession` in any way
        in order to resolve engines for requests.

        .. note::

            This method proxies directly to the :meth:`_orm.Session.get_bind`
            method, however is currently **not** useful as an override target,
            in contrast to that of the :meth:`_orm.Session.get_bind` method.
            The example below illustrates how to implement custom
            :meth:`_orm.Session.get_bind` schemes that work with
            :class:`.AsyncSession` and :class:`.AsyncEngine`.

        The pattern introduced at :ref:`session_custom_partitioning`
        illustrates how to apply a custom bind-lookup scheme to a
        :class:`_orm.Session` given a set of :class:`_engine.Engine` objects.
        To apply a corresponding :meth:`_orm.Session.get_bind` implementation
        for use with a :class:`.AsyncSession` and :class:`.AsyncEngine`
        objects, continue to subclass :class:`_orm.Session` and apply it to
        :class:`.AsyncSession` using
        :paramref:`.AsyncSession.sync_session_class`. The inner method must
        continue to return :class:`_engine.Engine` instances, which can be
        acquired from a :class:`_asyncio.AsyncEngine` using the
        :attr:`_asyncio.AsyncEngine.sync_engine` attribute::

            # using example from "Custom Vertical Partitioning"


            import random

            from sqlalchemy.ext.asyncio import AsyncSession
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy.ext.asyncio import async_sessionmaker
            from sqlalchemy.orm import Session

            # construct async engines w/ async drivers
            engines = {
                "leader": create_async_engine("sqlite+aiosqlite:///leader.db"),
                "other": create_async_engine("sqlite+aiosqlite:///other.db"),
                "follower1": create_async_engine("sqlite+aiosqlite:///follower1.db"),
                "follower2": create_async_engine("sqlite+aiosqlite:///follower2.db"),
            }


            class RoutingSession(Session):
                def get_bind(self, mapper=None, clause=None, **kw):
                    # within get_bind(), return sync engines
                    if mapper and issubclass(mapper.class_, MyOtherClass):
                        return engines["other"].sync_engine
                    elif self._flushing or isinstance(clause, (Update, Delete)):
                        return engines["leader"].sync_engine
                    else:
                        return engines[
                            random.choice(["follower1", "follower2"])
                        ].sync_engine


            # apply to AsyncSession using sync_session_class
            AsyncSessionMaker = async_sessionmaker(sync_session_class=RoutingSession)

        The :meth:`_orm.Session.get_bind` method is called in a non-asyncio,
        implicitly non-blocking context in the same manner as ORM event hooks
        and functions that are invoked via :meth:`.AsyncSession.run_sync`, so
        routines that wish to run SQL commands inside of
        :meth:`_orm.Session.get_bind` can continue to do so using
        blocking-style code, which will be translated to implicitly async calls
        at the point of invoking IO on the database drivers.

        """  # noqa: E501

        return self.sync_session.get_bind(
            mapper=mapper, clause=clause, bind=bind, **kw
        )

    async def connection(
        self,
        bind_arguments: Optional[_BindArguments] = None,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
        **kw: Any,
    ) -> AsyncConnection:
        r"""Return a :class:`_asyncio.AsyncConnection` object corresponding to
        this :class:`.Session` object's transactional state.

        This method may also be used to establish execution options for the
        database connection used by the current transaction.

        .. versionadded:: 1.4.24  Added \**kw arguments which are passed
           through to the underlying :meth:`_orm.Session.connection` method.

        .. seealso::

            :meth:`_orm.Session.connection` - main documentation for
            "connection"

        """

        sync_connection = await greenlet_spawn(
            self.sync_session.connection,
            bind_arguments=bind_arguments,
            execution_options=execution_options,
            **kw,
        )
        return engine.AsyncConnection._retrieve_proxy_for_target(
            sync_connection
        )

    def begin(self) -> AsyncSessionTransaction:
        """Return an :class:`_asyncio.AsyncSessionTransaction` object.

        The underlying :class:`_orm.Session` will perform the
        "begin" action when the :class:`_asyncio.AsyncSessionTransaction`
        object is entered::

            async with async_session.begin():
                ...  # ORM transaction is begun

        Note that database IO will not normally occur when the session-level
        transaction is begun, as database transactions begin on an
        on-demand basis.  However, the begin block is async to accommodate
        for a :meth:`_orm.SessionEvents.after_transaction_create`
        event hook that may perform IO.

        For a general description of ORM begin, see
        :meth:`_orm.Session.begin`.

        """

        return AsyncSessionTransaction(self)

    def begin_nested(self) -> AsyncSessionTransaction:
        """Return an :class:`_asyncio.AsyncSessionTransaction` object
        which will begin a "nested" transaction, e.g. SAVEPOINT.

        Behavior is the same as that of :meth:`_asyncio.AsyncSession.begin`.

        For a general description of ORM begin nested, see
        :meth:`_orm.Session.begin_nested`.

        .. seealso::

            :ref:`aiosqlite_serializable` - special workarounds required
            with the SQLite asyncio driver in order for SAVEPOINT to work
            correctly.

        """

        return AsyncSessionTransaction(self, nested=True)

    async def rollback(self) -> None:
        """Rollback the current transaction in progress.

        .. seealso::

            :meth:`_orm.Session.rollback` - main documentation for
            "rollback"
        """
        await greenlet_spawn(self.sync_session.rollback)

    async def commit(self) -> None:
        """Commit the current transaction in progress.

        .. seealso::

            :meth:`_orm.Session.commit` - main documentation for
            "commit"
        """
        await greenlet_spawn(self.sync_session.commit)

    async def close(self) -> None:
        """Close out the transactional resources and ORM objects used by this
        :class:`_asyncio.AsyncSession`.

        .. seealso::

            :meth:`_orm.Session.close` - main documentation for
            "close"

            :ref:`session_closing` - detail on the semantics of
            :meth:`_asyncio.AsyncSession.close` and
            :meth:`_asyncio.AsyncSession.reset`.

        """
        await greenlet_spawn(self.sync_session.close)

    async def reset(self) -> None:
        """Close out the transactional resources and ORM objects used by this
        :class:`_orm.Session`, resetting the session to its initial state.

        .. versionadded:: 2.0.22

        .. seealso::

            :meth:`_orm.Session.reset` - main documentation for
            "reset"

            :ref:`session_closing` - detail on the semantics of
            :meth:`_asyncio.AsyncSession.close` and
            :meth:`_asyncio.AsyncSession.reset`.

        """
        await greenlet_spawn(self.sync_session.reset)

    async def aclose(self) -> None:
        """A synonym for :meth:`_asyncio.AsyncSession.close`.

        The :meth:`_asyncio.AsyncSession.aclose` name is specifically
        to support the Python standard library ``@contextlib.aclosing``
        context manager function.

        .. versionadded:: 2.0.20

        """
        await self.close()

    async def invalidate(self) -> None:
        """Close this Session, using connection invalidation.

        For a complete description, see :meth:`_orm.Session.invalidate`.
        """
        await greenlet_spawn(self.sync_session.invalidate)

    @classmethod
    @util.deprecated(
        "2.0",
        "The :meth:`.AsyncSession.close_all` method is deprecated and will be "
        "removed in a future release.  Please refer to "
        ":func:`_asyncio.close_all_sessions`.",
    )
    async def close_all(cls) -> None:
        """Close all :class:`_asyncio.AsyncSession` sessions."""
        await close_all_sessions()

    async def __aenter__(self: _AS) -> _AS:
        return self

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:
        task = asyncio.create_task(self.close())
        await asyncio.shield(task)

    def _maker_context_manager(self: _AS) -> _AsyncSessionContextManager[_AS]:
        return _AsyncSessionContextManager(self)

    # START PROXY METHODS AsyncSession

    # code within this block is **programmatically,
    # statically generated** by tools/generate_proxy_methods.py

    def __contains__(self, instance: object) -> bool:
        r"""Return True if the instance is associated with this session.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        The instance may be pending or persistent within the Session for a
        result of True.


        """  # noqa: E501

        return self._proxied.__contains__(instance)

    def __iter__(self) -> Iterator[object]:
        r"""Iterate over all pending or persistent instances within this
        Session.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.


        """  # noqa: E501

        return self._proxied.__iter__()

    def add(self, instance: object, _warn: bool = True) -> None:
        r"""Place an object into this :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        Objects that are in the :term:`transient` state when passed to the
        :meth:`_orm.Session.add` method will move to the
        :term:`pending` state, until the next flush, at which point they
        will move to the :term:`persistent` state.

        Objects that are in the :term:`detached` state when passed to the
        :meth:`_orm.Session.add` method will move to the :term:`persistent`
        state directly.

        If the transaction used by the :class:`_orm.Session` is rolled back,
        objects which were transient when they were passed to
        :meth:`_orm.Session.add` will be moved back to the
        :term:`transient` state, and will no longer be present within this
        :class:`_orm.Session`.

        .. seealso::

            :meth:`_orm.Session.add_all`

            :ref:`session_adding` - at :ref:`session_basics`


        """  # noqa: E501

        return self._proxied.add(instance, _warn=_warn)

    def add_all(self, instances: Iterable[object]) -> None:
        r"""Add the given collection of instances to this :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        See the documentation for :meth:`_orm.Session.add` for a general
        behavioral description.

        .. seealso::

            :meth:`_orm.Session.add`

            :ref:`session_adding` - at :ref:`session_basics`


        """  # noqa: E501

        return self._proxied.add_all(instances)

    def expire(
        self, instance: object, attribute_names: Optional[Iterable[str]] = None
    ) -> None:
        r"""Expire the attributes on an instance.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        Marks the attributes of an instance as out of date. When an expired
        attribute is next accessed, a query will be issued to the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire all objects in the :class:`.Session` simultaneously,
        use :meth:`Session.expire_all`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire` only makes sense for the specific
        case that a non-ORM SQL statement was emitted in the current
        transaction.

        :param instance: The instance to be refreshed.
        :param attribute_names: optional list of string attribute names
          indicating a subset of attributes to be expired.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`


        """  # noqa: E501

        return self._proxied.expire(instance, attribute_names=attribute_names)

    def expire_all(self) -> None:
        r"""Expires all persistent instances within this Session.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        When any attributes on a persistent instance is next accessed,
        a query will be issued using the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire individual objects and individual attributes
        on those objects, use :meth:`Session.expire`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire_all` is not usually needed,
        assuming the transaction is isolated.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`


        """  # noqa: E501

        return self._proxied.expire_all()

    def expunge(self, instance: object) -> None:
        r"""Remove the `instance` from this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This will free all internal references to the instance.  Cascading
        will be applied according to the *expunge* cascade rule.


        """  # noqa: E501

        return self._proxied.expunge(instance)

    def expunge_all(self) -> None:
        r"""Remove all object instances from this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This is equivalent to calling ``expunge(obj)`` on all objects in this
        ``Session``.


        """  # noqa: E501

        return self._proxied.expunge_all()

    def is_modified(
        self, instance: object, include_collections: bool = True
    ) -> bool:
        r"""Return ``True`` if the given instance has locally
        modified attributes.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This method retrieves the history for each instrumented
        attribute on the instance and performs a comparison of the current
        value to its previously flushed or committed value, if any.

        It is in effect a more expensive and accurate
        version of checking for the given instance in the
        :attr:`.Session.dirty` collection; a full test for
        each attribute's net "dirty" status is performed.

        E.g.::

            return session.is_modified(someobject)

        A few caveats to this method apply:

        * Instances present in the :attr:`.Session.dirty` collection may
          report ``False`` when tested with this method.  This is because
          the object may have received change events via attribute mutation,
          thus placing it in :attr:`.Session.dirty`, but ultimately the state
          is the same as that loaded from the database, resulting in no net
          change here.
        * Scalar attributes may not have recorded the previously set
          value when a new value was applied, if the attribute was not loaded,
          or was expired, at the time the new value was received - in these
          cases, the attribute is assumed to have a change, even if there is
          ultimately no net change against its database value. SQLAlchemy in
          most cases does not need the "old" value when a set event occurs, so
          it skips the expense of a SQL call if the old value isn't present,
          based on the assumption that an UPDATE of the scalar value is
          usually needed, and in those few cases where it isn't, is less
          expensive on average than issuing a defensive SELECT.

          The "old" value is fetched unconditionally upon set only if the
          attribute container has the ``active_history`` flag set to ``True``.
          This flag is set typically for primary key attributes and scalar
          object references that are not a simple many-to-one.  To set this
          flag for any arbitrary mapped column, use the ``active_history``
          argument with :func:`.column_property`.

        :param instance: mapped instance to be tested for pending changes.
        :param include_collections: Indicates if multivalued collections
         should be included in the operation.  Setting this to ``False`` is a
         way to detect only local-column based properties (i.e. scalar columns
         or many-to-one foreign keys) that would result in an UPDATE for this
         instance upon flush.


        """  # noqa: E501

        return self._proxied.is_modified(
            instance, include_collections=include_collections
        )

    def in_transaction(self) -> bool:
        r"""Return True if this :class:`_orm.Session` has begun a transaction.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_orm.Session.is_active`



        """  # noqa: E501

        return self._proxied.in_transaction()

    def in_nested_transaction(self) -> bool:
        r"""Return True if this :class:`_orm.Session` has begun a nested
        transaction, e.g. SAVEPOINT.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        .. versionadded:: 1.4


        """  # noqa: E501

        return self._proxied.in_nested_transaction()

    @property
    def dirty(self) -> Any:
        r"""The set of all persistent instances considered dirty.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        E.g.::

            some_mapped_object in session.dirty

        Instances are considered dirty when they were modified but not
        deleted.

        Note that this 'dirty' calculation is 'optimistic'; most
        attribute-setting or collection modification operations will
        mark an instance as 'dirty' and place it in this set, even if
        there is no net change to the attribute's value.  At flush
        time, the value of each attribute is compared to its
        previously saved value, and if there's no net change, no SQL
        operation will occur (this is a more expensive operation so
        it's only done at flush time).

        To check if an instance has actionable net changes to its
        attributes, use the :meth:`.Session.is_modified` method.


        """  # noqa: E501

        return self._proxied.dirty

    @property
    def deleted(self) -> Any:
        r"""The set of all instances marked as 'deleted' within this ``Session``

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        """  # noqa: E501

        return self._proxied.deleted

    @property
    def new(self) -> Any:
        r"""The set of all instances marked as 'new' within this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        """  # noqa: E501

        return self._proxied.new

    @property
    def identity_map(self) -> IdentityMap:
        r"""Proxy for the :attr:`_orm.Session.identity_map` attribute
        on behalf of the :class:`_asyncio.AsyncSession` class.

        """  # noqa: E501

        return self._proxied.identity_map

    @identity_map.setter
    def identity_map(self, attr: IdentityMap) -> None:
        self._proxied.identity_map = attr

    @property
    def is_active(self) -> Any:
        r"""True if this :class:`.Session` not in "partial rollback" state.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        .. versionchanged:: 1.4 The :class:`_orm.Session` no longer begins
           a new transaction immediately, so this attribute will be False
           when the :class:`_orm.Session` is first instantiated.

        "partial rollback" state typically indicates that the flush process
        of the :class:`_orm.Session` has failed, and that the
        :meth:`_orm.Session.rollback` method must be emitted in order to
        fully roll back the transaction.

        If this :class:`_orm.Session` is not in a transaction at all, the
        :class:`_orm.Session` will autobegin when it is first used, so in this
        case :attr:`_orm.Session.is_active` will return True.

        Otherwise, if this :class:`_orm.Session` is within a transaction,
        and that transaction has not been rolled back internally, the
        :attr:`_orm.Session.is_active` will also return True.

        .. seealso::

            :ref:`faq_session_rollback`

            :meth:`_orm.Session.in_transaction`


        """  # noqa: E501

        return self._proxied.is_active

    @property
    def autoflush(self) -> bool:
        r"""Proxy for the :attr:`_orm.Session.autoflush` attribute
        on behalf of the :class:`_asyncio.AsyncSession` class.

        """  # noqa: E501

        return self._proxied.autoflush

    @autoflush.setter
    def autoflush(self, attr: bool) -> None:
        self._proxied.autoflush = attr

    @property
    def no_autoflush(self) -> Any:
        r"""Return a context manager that disables autoflush.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        e.g.::

            with session.no_autoflush:

                some_object = SomeClass()
                session.add(some_object)
                # won't autoflush
                some_object.related_thing = session.query(SomeRelated).first()

        Operations that proceed within the ``with:`` block
        will not be subject to flushes occurring upon query
        access.  This is useful when initializing a series
        of objects which involve existing database queries,
        where the uncompleted object should not yet be flushed.


        """  # noqa: E501

        return self._proxied.no_autoflush

    @property
    def info(self) -> Any:
        r"""A user-modifiable dictionary.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        The initial value of this dictionary can be populated using the
        ``info`` argument to the :class:`.Session` constructor or
        :class:`.sessionmaker` constructor or factory methods.  The dictionary
        here is always local to this :class:`.Session` and can be modified
        independently of all other :class:`.Session` objects.


        """  # noqa: E501

        return self._proxied.info

    @classmethod
    def object_session(cls, instance: object) -> Optional[Session]:
        r"""Return the :class:`.Session` to which an object belongs.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This is an alias of :func:`.object_session`.


        """  # noqa: E501

        return Session.object_session(instance)

    @classmethod
    def identity_key(
        cls,
        class_: Optional[Type[Any]] = None,
        ident: Union[Any, Tuple[Any, ...]] = None,
        *,
        instance: Optional[Any] = None,
        row: Optional[Union[Row[Any], RowMapping]] = None,
        identity_token: Optional[Any] = None,
    ) -> _IdentityKeyType[Any]:
        r"""Return an identity key.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This is an alias of :func:`.util.identity_key`.


        """  # noqa: E501

        return Session.identity_key(
            class_=class_,
            ident=ident,
            instance=instance,
            row=row,
            identity_token=identity_token,
        )

    # END PROXY METHODS AsyncSession


_AS = TypeVar("_AS", bound="AsyncSession")


class async_sessionmaker(Generic[_AS]):
    """A configurable :class:`.AsyncSession` factory.

    The :class:`.async_sessionmaker` factory works in the same way as the
    :class:`.sessionmaker` factory, to generate new :class:`.AsyncSession`
    objects when called, creating them given
    the configurational arguments established here.

    e.g.::

        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.ext.asyncio import async_sessionmaker


        async def run_some_sql(
            async_session: async_sessionmaker[AsyncSession],
        ) -> None:
            async with async_session() as session:
                session.add(SomeObject(data="object"))
                session.add(SomeOtherObject(name="other object"))
                await session.commit()


        async def main() -> None:
            # an AsyncEngine, which the AsyncSession will use for connection
            # resources
            engine = create_async_engine(
                "postgresql+asyncpg://scott:tiger@localhost/"
            )

            # create a reusable factory for new AsyncSession instances
            async_session = async_sessionmaker(engine)

            await run_some_sql(async_session)

            await engine.dispose()

    The :class:`.async_sessionmaker` is useful so that different parts
    of a program can create new :class:`.AsyncSession` objects with a
    fixed configuration established up front.  Note that :class:`.AsyncSession`
    objects may also be instantiated directly when not using
    :class:`.async_sessionmaker`.

    .. versionadded:: 2.0  :class:`.async_sessionmaker` provides a
       :class:`.sessionmaker` class that's dedicated to the
       :class:`.AsyncSession` object, including pep-484 typing support.

    .. seealso::

        :ref:`asyncio_orm` - shows example use

        :class:`.sessionmaker`  - general overview of the
         :class:`.sessionmaker` architecture


        :ref:`session_getting` - introductory text on creating
        sessions using :class:`.sessionmaker`.

    """  # noqa E501

    class_: Type[_AS]

    @overload
    def __init__(
        self,
        bind: Optional[_AsyncSessionBind] = ...,
        *,
        class_: Type[_AS],
        autoflush: bool = ...,
        expire_on_commit: bool = ...,
        info: Optional[_InfoType] = ...,
        **kw: Any,
    ): ...

    @overload
    def __init__(
        self: "async_sessionmaker[AsyncSession]",
        bind: Optional[_AsyncSessionBind] = ...,
        *,
        autoflush: bool = ...,
        expire_on_commit: bool = ...,
        info: Optional[_InfoType] = ...,
        **kw: Any,
    ): ...

    def __init__(
        self,
        bind: Optional[_AsyncSessionBind] = None,
        *,
        class_: Type[_AS] = AsyncSession,  # type: ignore
        autoflush: bool = True,
        expire_on_commit: bool = True,
        info: Optional[_InfoType] = None,
        **kw: Any,
    ):
        r"""Construct a new :class:`.async_sessionmaker`.

        All arguments here except for ``class_`` correspond to arguments
        accepted by :class:`.Session` directly. See the
        :meth:`.AsyncSession.__init__` docstring for more details on
        parameters.


        """
        kw["bind"] = bind
        kw["autoflush"] = autoflush
        kw["expire_on_commit"] = expire_on_commit
        if info is not None:
            kw["info"] = info
        self.kw = kw
        self.class_ = class_

    def begin(self) -> _AsyncSessionContextManager[_AS]:
        """Produce a context manager that both provides a new
        :class:`_orm.AsyncSession` as well as a transaction that commits.


        e.g.::

            async def main():
                Session = async_sessionmaker(some_engine)

                async with Session.begin() as session:
                    session.add(some_object)

                # commits transaction, closes session

        """

        session = self()
        return session._maker_context_manager()

    def __call__(self, **local_kw: Any) -> _AS:
        """Produce a new :class:`.AsyncSession` object using the configuration
        established in this :class:`.async_sessionmaker`.

        In Python, the ``__call__`` method is invoked on an object when
        it is "called" in the same way as a function::

            AsyncSession = async_sessionmaker(async_engine, expire_on_commit=False)
            session = AsyncSession()  # invokes sessionmaker.__call__()

        """  # noqa E501
        for k, v in self.kw.items():
            if k == "info" and "info" in local_kw:
                d = v.copy()
                d.update(local_kw["info"])
                local_kw["info"] = d
            else:
                local_kw.setdefault(k, v)
        return self.class_(**local_kw)

    def configure(self, **new_kw: Any) -> None:
        """(Re)configure the arguments for this async_sessionmaker.

        e.g.::

            AsyncSession = async_sessionmaker(some_engine)

            AsyncSession.configure(bind=create_async_engine("sqlite+aiosqlite://"))
        """  # noqa E501

        self.kw.update(new_kw)

    def __repr__(self) -> str:
        return "%s(class_=%r, %s)" % (
            self.__class__.__name__,
            self.class_.__name__,
            ", ".join("%s=%r" % (k, v) for k, v in self.kw.items()),
        )


class _AsyncSessionContextManager(Generic[_AS]):
    __slots__ = ("async_session", "trans")

    async_session: _AS
    trans: AsyncSessionTransaction

    def __init__(self, async_session: _AS):
        self.async_session = async_session

    async def __aenter__(self) -> _AS:
        self.trans = self.async_session.begin()
        await self.trans.__aenter__()
        return self.async_session

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:
        async def go() -> None:
            await self.trans.__aexit__(type_, value, traceback)
            await self.async_session.__aexit__(type_, value, traceback)

        task = asyncio.create_task(go())
        await asyncio.shield(task)


class AsyncSessionTransaction(
    ReversibleProxy[SessionTransaction],
    StartableContext["AsyncSessionTransaction"],
):
    """A wrapper for the ORM :class:`_orm.SessionTransaction` object.

    This object is provided so that a transaction-holding object
    for the :meth:`_asyncio.AsyncSession.begin` may be returned.

    The object supports both explicit calls to
    :meth:`_asyncio.AsyncSessionTransaction.commit` and
    :meth:`_asyncio.AsyncSessionTransaction.rollback`, as well as use as an
    async context manager.


    .. versionadded:: 1.4

    """

    __slots__ = ("session", "sync_transaction", "nested")

    session: AsyncSession
    sync_transaction: Optional[SessionTransaction]

    def __init__(self, session: AsyncSession, nested: bool = False):
        self.session = session
        self.nested = nested
        self.sync_transaction = None

    @property
    def is_active(self) -> bool:
        return (
            self._sync_transaction() is not None
            and self._sync_transaction().is_active
        )

    def _sync_transaction(self) -> SessionTransaction:
        if not self.sync_transaction:
            self._raise_for_not_started()
        return self.sync_transaction

    async def rollback(self) -> None:
        """Roll back this :class:`_asyncio.AsyncTransaction`."""
        await greenlet_spawn(self._sync_transaction().rollback)

    async def commit(self) -> None:
        """Commit this :class:`_asyncio.AsyncTransaction`."""

        await greenlet_spawn(self._sync_transaction().commit)

    @classmethod
    def _regenerate_proxy_for_target(  # type: ignore[override]
        cls,
        target: SessionTransaction,
        async_session: AsyncSession,
        **additional_kw: Any,  # noqa: U100
    ) -> AsyncSessionTransaction:
        sync_transaction = target
        nested = target.nested
        obj = cls.__new__(cls)
        obj.session = async_session
        obj.sync_transaction = obj._assign_proxied(sync_transaction)
        obj.nested = nested
        return obj

    async def start(
        self, is_ctxmanager: bool = False
    ) -> AsyncSessionTransaction:
        self.sync_transaction = self._assign_proxied(
            await greenlet_spawn(
                self.session.sync_session.begin_nested
                if self.nested
                else self.session.sync_session.begin
            )
        )
        if is_ctxmanager:
            self.sync_transaction.__enter__()
        return self

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:
        await greenlet_spawn(
            self._sync_transaction().__exit__, type_, value, traceback
        )


def async_object_session(instance: object) -> Optional[AsyncSession]:
    """Return the :class:`_asyncio.AsyncSession` to which the given instance
    belongs.

    This function makes use of the sync-API function
    :class:`_orm.object_session` to retrieve the :class:`_orm.Session` which
    refers to the given instance, and from there links it to the original
    :class:`_asyncio.AsyncSession`.

    If the :class:`_asyncio.AsyncSession` has been garbage collected, the
    return value is ``None``.

    This functionality is also available from the
    :attr:`_orm.InstanceState.async_session` accessor.

    :param instance: an ORM mapped instance
    :return: an :class:`_asyncio.AsyncSession` object, or ``None``.

    .. versionadded:: 1.4.18

    """

    session = object_session(instance)
    if session is not None:
        return async_session(session)
    else:
        return None


def async_session(session: Session) -> Optional[AsyncSession]:
    """Return the :class:`_asyncio.AsyncSession` which is proxying the given
    :class:`_orm.Session` object, if any.

    :param session: a :class:`_orm.Session` instance.
    :return: a :class:`_asyncio.AsyncSession` instance, or ``None``.

    .. versionadded:: 1.4.18

    """
    return AsyncSession._retrieve_proxy_for_target(session, regenerate=False)


async def close_all_sessions() -> None:
    """Close all :class:`_asyncio.AsyncSession` sessions.

    .. versionadded:: 2.0.23

    .. seealso::

        :func:`.session.close_all_sessions`

    """
    await greenlet_spawn(_sync_close_all_sessions)


_instance_state._async_provider = async_session  # type: ignore

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\c_parser.py ===
#------------------------------------------------------------------------------
# pycparser: c_parser.py
#
# CParser class: Parser and AST builder for the C language
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#------------------------------------------------------------------------------
from .ply import yacc

from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers


@template
class CParser(PLYParser):
    def __init__(
            self,
            lex_optimize=True,
            lexer=CLexer,
            lextab='pycparser.lextab',
            yacc_optimize=True,
            yacctab='pycparser.yacctab',
            yacc_debug=False,
            taboutputdir=''):
        """ Create a new CParser.

            Some arguments for controlling the debug/optimization
            level of the parser are provided. The defaults are
            tuned for release/performance mode.
            The simple rules for using them are:
            *) When tweaking CParser/CLexer, set these to False
            *) When releasing a stable parser, set to True

            lex_optimize:
                Set to False when you're modifying the lexer.
                Otherwise, changes in the lexer won't be used, if
                some lextab.py file exists.
                When releasing with a stable lexer, set to True
                to save the re-generation of the lexer table on
                each run.

            lexer:
                Set this parameter to define the lexer to use if
                you're not using the default CLexer.

            lextab:
                Points to the lex table that's used for optimized
                mode. Only if you're modifying the lexer and want
                some tests to avoid re-generating the table, make
                this point to a local lex table file (that's been
                earlier generated with lex_optimize=True)

            yacc_optimize:
                Set to False when you're modifying the parser.
                Otherwise, changes in the parser won't be used, if
                some parsetab.py file exists.
                When releasing with a stable parser, set to True
                to save the re-generation of the parser table on
                each run.

            yacctab:
                Points to the yacc table that's used for optimized
                mode. Only if you're modifying the parser, make
                this point to a local yacc table file

            yacc_debug:
                Generate a parser.out file that explains how yacc
                built the parsing table from the grammar.

            taboutputdir:
                Set this parameter to control the location of generated
                lextab and yacctab files.
        """
        self.clex = lexer(
            error_func=self._lex_error_func,
            on_lbrace_func=self._lex_on_lbrace_func,
            on_rbrace_func=self._lex_on_rbrace_func,
            type_lookup_func=self._lex_type_lookup_func)

        self.clex.build(
            optimize=lex_optimize,
            lextab=lextab,
            outputdir=taboutputdir)
        self.tokens = self.clex.tokens

        rules_with_opt = [
            'abstract_declarator',
            'assignment_expression',
            'declaration_list',
            'declaration_specifiers_no_type',
            'designation',
            'expression',
            'identifier_list',
            'init_declarator_list',
            'id_init_declarator_list',
            'initializer_list',
            'parameter_type_list',
            'block_item_list',
            'type_qualifier_list',
            'struct_declarator_list'
        ]

        for rule in rules_with_opt:
            self._create_opt_rule(rule)

        self.cparser = yacc.yacc(
            module=self,
            start='translation_unit_or_empty',
            debug=yacc_debug,
            optimize=yacc_optimize,
            tabmodule=yacctab,
            outputdir=taboutputdir)

        # Stack of scopes for keeping track of symbols. _scope_stack[-1] is
        # the current (topmost) scope. Each scope is a dictionary that
        # specifies whether a name is a type. If _scope_stack[n][name] is
        # True, 'name' is currently a type in the scope. If it's False,
        # 'name' is used in the scope but not as a type (for instance, if we
        # saw: int name;
        # If 'name' is not a key in _scope_stack[n] then 'name' was not defined
        # in this scope at all.
        self._scope_stack = [dict()]

        # Keeps track of the last token given to yacc (the lookahead token)
        self._last_yielded_token = None

    def parse(self, text, filename='', debug=False):
        """ Parses C code and returns an AST.

            text:
                A string containing the C source code

            filename:
                Name of the file being parsed (for meaningful
                error messages)

            debug:
                Debug flag to YACC
        """
        self.clex.filename = filename
        self.clex.reset_lineno()
        self._scope_stack = [dict()]
        self._last_yielded_token = None
        return self.cparser.parse(
                input=text,
                lexer=self.clex,
                debug=debug)

    ######################--   PRIVATE   --######################

    def _push_scope(self):
        self._scope_stack.append(dict())

    def _pop_scope(self):
        assert len(self._scope_stack) > 1
        self._scope_stack.pop()

    def _add_typedef_name(self, name, coord):
        """ Add a new typedef name (ie a TYPEID) to the current scope
        """
        if not self._scope_stack[-1].get(name, True):
            self._parse_error(
                "Typedef %r previously declared as non-typedef "
                "in this scope" % name, coord)
        self._scope_stack[-1][name] = True

    def _add_identifier(self, name, coord):
        """ Add a new object, function, or enum member name (ie an ID) to the
            current scope
        """
        if self._scope_stack[-1].get(name, False):
            self._parse_error(
                "Non-typedef %r previously declared as typedef "
                "in this scope" % name, coord)
        self._scope_stack[-1][name] = False

    def _is_type_in_scope(self, name):
        """ Is *name* a typedef-name in the current scope?
        """
        for scope in reversed(self._scope_stack):
            # If name is an identifier in this scope it shadows typedefs in
            # higher scopes.
            in_scope = scope.get(name)
            if in_scope is not None: return in_scope
        return False

    def _lex_error_func(self, msg, line, column):
        self._parse_error(msg, self._coord(line, column))

    def _lex_on_lbrace_func(self):
        self._push_scope()

    def _lex_on_rbrace_func(self):
        self._pop_scope()

    def _lex_type_lookup_func(self, name):
        """ Looks up types that were previously defined with
            typedef.
            Passed to the lexer for recognizing identifiers that
            are types.
        """
        is_type = self._is_type_in_scope(name)
        return is_type

    def _get_yacc_lookahead_token(self):
        """ We need access to yacc's lookahead token in certain cases.
            This is the last token yacc requested from the lexer, so we
            ask the lexer.
        """
        return self.clex.last_token

    # To understand what's going on here, read sections A.8.5 and
    # A.8.6 of K&R2 very carefully.
    #
    # A C type consists of a basic type declaration, with a list
    # of modifiers. For example:
    #
    # int *c[5];
    #
    # The basic declaration here is 'int c', and the pointer and
    # the array are the modifiers.
    #
    # Basic declarations are represented by TypeDecl (from module c_ast) and the
    # modifiers are FuncDecl, PtrDecl and ArrayDecl.
    #
    # The standard states that whenever a new modifier is parsed, it should be
    # added to the end of the list of modifiers. For example:
    #
    # K&R2 A.8.6.2: Array Declarators
    #
    # In a declaration T D where D has the form
    #   D1 [constant-expression-opt]
    # and the type of the identifier in the declaration T D1 is
    # "type-modifier T", the type of the
    # identifier of D is "type-modifier array of T"
    #
    # This is what this method does. The declarator it receives
    # can be a list of declarators ending with TypeDecl. It
    # tacks the modifier to the end of this list, just before
    # the TypeDecl.
    #
    # Additionally, the modifier may be a list itself. This is
    # useful for pointers, that can come as a chain from the rule
    # p_pointer. In this case, the whole modifier list is spliced
    # into the new location.
    def _type_modify_decl(self, decl, modifier):
        """ Tacks a type modifier on a declarator, and returns
            the modified declarator.

            Note: the declarator and modifier may be modified
        """
        #~ print '****'
        #~ decl.show(offset=3)
        #~ modifier.show(offset=3)
        #~ print '****'

        modifier_head = modifier
        modifier_tail = modifier

        # The modifier may be a nested list. Reach its tail.
        while modifier_tail.type:
            modifier_tail = modifier_tail.type

        # If the decl is a basic type, just tack the modifier onto it.
        if isinstance(decl, c_ast.TypeDecl):
            modifier_tail.type = decl
            return modifier
        else:
            # Otherwise, the decl is a list of modifiers. Reach
            # its tail and splice the modifier onto the tail,
            # pointing to the underlying basic type.
            decl_tail = decl

            while not isinstance(decl_tail.type, c_ast.TypeDecl):
                decl_tail = decl_tail.type

            modifier_tail.type = decl_tail.type
            decl_tail.type = modifier_head
            return decl

    # Due to the order in which declarators are constructed,
    # they have to be fixed in order to look like a normal AST.
    #
    # When a declaration arrives from syntax construction, it has
    # these problems:
    # * The innermost TypeDecl has no type (because the basic
    #   type is only known at the uppermost declaration level)
    # * The declaration has no variable name, since that is saved
    #   in the innermost TypeDecl
    # * The typename of the declaration is a list of type
    #   specifiers, and not a node. Here, basic identifier types
    #   should be separated from more complex types like enums
    #   and structs.
    #
    # This method fixes these problems.
    def _fix_decl_name_type(self, decl, typename):
        """ Fixes a declaration. Modifies decl.
        """
        # Reach the underlying basic type
        #
        type = decl
        while not isinstance(type, c_ast.TypeDecl):
            type = type.type

        decl.name = type.declname
        type.quals = decl.quals[:]

        # The typename is a list of types. If any type in this
        # list isn't an IdentifierType, it must be the only
        # type in the list (it's illegal to declare "int enum ..")
        # If all the types are basic, they're collected in the
        # IdentifierType holder.
        for tn in typename:
            if not isinstance(tn, c_ast.IdentifierType):
                if len(typename) > 1:
                    self._parse_error(
                        "Invalid multiple types specified", tn.coord)
                else:
                    type.type = tn
                    return decl

        if not typename:
            # Functions default to returning int
            #
            if not isinstance(decl.type, c_ast.FuncDecl):
                self._parse_error(
                        "Missing type in declaration", decl.coord)
            type.type = c_ast.IdentifierType(
                    ['int'],
                    coord=decl.coord)
        else:
            # At this point, we know that typename is a list of IdentifierType
            # nodes. Concatenate all the names into a single list.
            #
            type.type = c_ast.IdentifierType(
                [name for id in typename for name in id.names],
                coord=typename[0].coord)
        return decl

    def _add_declaration_specifier(self, declspec, newspec, kind, append=False):
        """ Declaration specifiers are represented by a dictionary
            with the entries:
            * qual: a list of type qualifiers
            * storage: a list of storage type qualifiers
            * type: a list of type specifiers
            * function: a list of function specifiers
            * alignment: a list of alignment specifiers

            This method is given a declaration specifier, and a
            new specifier of a given kind.
            If `append` is True, the new specifier is added to the end of
            the specifiers list, otherwise it's added at the beginning.
            Returns the declaration specifier, with the new
            specifier incorporated.
        """
        spec = declspec or dict(qual=[], storage=[], type=[], function=[], alignment=[])

        if append:
            spec[kind].append(newspec)
        else:
            spec[kind].insert(0, newspec)

        return spec

    def _build_declarations(self, spec, decls, typedef_namespace=False):
        """ Builds a list of declarations all sharing the given specifiers.
            If typedef_namespace is true, each declared name is added
            to the "typedef namespace", which also includes objects,
            functions, and enum constants.
        """
        is_typedef = 'typedef' in spec['storage']
        declarations = []

        # Bit-fields are allowed to be unnamed.
        if decls[0].get('bitsize') is not None:
            pass

        # When redeclaring typedef names as identifiers in inner scopes, a
        # problem can occur where the identifier gets grouped into
        # spec['type'], leaving decl as None.  This can only occur for the
        # first declarator.
        elif decls[0]['decl'] is None:
            if len(spec['type']) < 2 or len(spec['type'][-1].names) != 1 or \
                    not self._is_type_in_scope(spec['type'][-1].names[0]):
                coord = '?'
                for t in spec['type']:
                    if hasattr(t, 'coord'):
                        coord = t.coord
                        break
                self._parse_error('Invalid declaration', coord)

            # Make this look as if it came from "direct_declarator:ID"
            decls[0]['decl'] = c_ast.TypeDecl(
                declname=spec['type'][-1].names[0],
                type=None,
                quals=None,
                align=spec['alignment'],
                coord=spec['type'][-1].coord)
            # Remove the "new" type's name from the end of spec['type']
            del spec['type'][-1]

        # A similar problem can occur where the declaration ends up looking
        # like an abstract declarator.  Give it a name if this is the case.
        elif not isinstance(decls[0]['decl'], (
                c_ast.Enum, c_ast.Struct, c_ast.Union, c_ast.IdentifierType)):
            decls_0_tail = decls[0]['decl']
            while not isinstance(decls_0_tail, c_ast.TypeDecl):
                decls_0_tail = decls_0_tail.type
            if decls_0_tail.declname is None:
                decls_0_tail.declname = spec['type'][-1].names[0]
                del spec['type'][-1]

        for decl in decls:
            assert decl['decl'] is not None
            if is_typedef:
                declaration = c_ast.Typedef(
                    name=None,
                    quals=spec['qual'],
                    storage=spec['storage'],
                    type=decl['decl'],
                    coord=decl['decl'].coord)
            else:
                declaration = c_ast.Decl(
                    name=None,
                    quals=spec['qual'],
                    align=spec['alignment'],
                    storage=spec['storage'],
                    funcspec=spec['function'],
                    type=decl['decl'],
                    init=decl.get('init'),
                    bitsize=decl.get('bitsize'),
                    coord=decl['decl'].coord)

            if isinstance(declaration.type, (
                    c_ast.Enum, c_ast.Struct, c_ast.Union,
                    c_ast.IdentifierType)):
                fixed_decl = declaration
            else:
                fixed_decl = self._fix_decl_name_type(declaration, spec['type'])

            # Add the type name defined by typedef to a
            # symbol table (for usage in the lexer)
            if typedef_namespace:
                if is_typedef:
                    self._add_typedef_name(fixed_decl.name, fixed_decl.coord)
                else:
                    self._add_identifier(fixed_decl.name, fixed_decl.coord)

            fixed_decl = fix_atomic_specifiers(fixed_decl)
            declarations.append(fixed_decl)

        return declarations

    def _build_function_definition(self, spec, decl, param_decls, body):
        """ Builds a function definition.
        """
        if 'typedef' in spec['storage']:
            self._parse_error("Invalid typedef", decl.coord)

        declaration = self._build_declarations(
            spec=spec,
            decls=[dict(decl=decl, init=None)],
            typedef_namespace=True)[0]

        return c_ast.FuncDef(
            decl=declaration,
            param_decls=param_decls,
            body=body,
            coord=decl.coord)

    def _select_struct_union_class(self, token):
        """ Given a token (either STRUCT or UNION), selects the
            appropriate AST class.
        """
        if token == 'struct':
            return c_ast.Struct
        else:
            return c_ast.Union

    ##
    ## Precedence and associativity of operators
    ##
    # If this changes, c_generator.CGenerator.precedence_map needs to change as
    # well
    precedence = (
        ('left', 'LOR'),
        ('left', 'LAND'),
        ('left', 'OR'),
        ('left', 'XOR'),
        ('left', 'AND'),
        ('left', 'EQ', 'NE'),
        ('left', 'GT', 'GE', 'LT', 'LE'),
        ('left', 'RSHIFT', 'LSHIFT'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE', 'MOD')
    )

    ##
    ## Grammar productions
    ## Implementation of the BNF defined in K&R2 A.13
    ##

    # Wrapper around a translation unit, to allow for empty input.
    # Not strictly part of the C99 Grammar, but useful in practice.
    def p_translation_unit_or_empty(self, p):
        """ translation_unit_or_empty   : translation_unit
                                        | empty
        """
        if p[1] is None:
            p[0] = c_ast.FileAST([])
        else:
            p[0] = c_ast.FileAST(p[1])

    def p_translation_unit_1(self, p):
        """ translation_unit    : external_declaration
        """
        # Note: external_declaration is already a list
        p[0] = p[1]

    def p_translation_unit_2(self, p):
        """ translation_unit    : translation_unit external_declaration
        """
        p[1].extend(p[2])
        p[0] = p[1]

    # Declarations always come as lists (because they can be
    # several in one line), so we wrap the function definition
    # into a list as well, to make the return value of
    # external_declaration homogeneous.
    def p_external_declaration_1(self, p):
        """ external_declaration    : function_definition
        """
        p[0] = [p[1]]

    def p_external_declaration_2(self, p):
        """ external_declaration    : declaration
        """
        p[0] = p[1]

    def p_external_declaration_3(self, p):
        """ external_declaration    : pp_directive
                                    | pppragma_directive
        """
        p[0] = [p[1]]

    def p_external_declaration_4(self, p):
        """ external_declaration    : SEMI
        """
        p[0] = []

    def p_external_declaration_5(self, p):
        """ external_declaration    : static_assert
        """
        p[0] = p[1]

    def p_static_assert_declaration(self, p):
        """ static_assert           : _STATIC_ASSERT LPAREN constant_expression COMMA unified_string_literal RPAREN
                                    | _STATIC_ASSERT LPAREN constant_expression RPAREN
        """
        if len(p) == 5:
            p[0] = [c_ast.StaticAssert(p[3], None, self._token_coord(p, 1))]
        else:
            p[0] = [c_ast.StaticAssert(p[3], p[5], self._token_coord(p, 1))]

    def p_pp_directive(self, p):
        """ pp_directive  : PPHASH
        """
        self._parse_error('Directives not supported yet',
                          self._token_coord(p, 1))

    # This encompasses two types of C99-compatible pragmas:
    # - The #pragma directive:
    #       # pragma character_sequence
    # - The _Pragma unary operator:
    #       _Pragma ( " string_literal " )
    def p_pppragma_directive(self, p):
        """ pppragma_directive      : PPPRAGMA
                                    | PPPRAGMA PPPRAGMASTR
                                    | _PRAGMA LPAREN unified_string_literal RPAREN
        """
        if len(p) == 5:
            p[0] = c_ast.Pragma(p[3], self._token_coord(p, 2))
        elif len(p) == 3:
            p[0] = c_ast.Pragma(p[2], self._token_coord(p, 2))
        else:
            p[0] = c_ast.Pragma("", self._token_coord(p, 1))

    def p_pppragma_directive_list(self, p):
        """ pppragma_directive_list : pppragma_directive
                                    | pppragma_directive_list pppragma_directive
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    # In function definitions, the declarator can be followed by
    # a declaration list, for old "K&R style" function definitios.
    def p_function_definition_1(self, p):
        """ function_definition : id_declarator declaration_list_opt compound_statement
        """
        # no declaration specifiers - 'int' becomes the default type
        spec = dict(
            qual=[],
            alignment=[],
            storage=[],
            type=[c_ast.IdentifierType(['int'],
                                       coord=self._token_coord(p, 1))],
            function=[])

        p[0] = self._build_function_definition(
            spec=spec,
            decl=p[1],
            param_decls=p[2],
            body=p[3])

    def p_function_definition_2(self, p):
        """ function_definition : declaration_specifiers id_declarator declaration_list_opt compound_statement
        """
        spec = p[1]

        p[0] = self._build_function_definition(
            spec=spec,
            decl=p[2],
            param_decls=p[3],
            body=p[4])

    # Note, according to C18 A.2.2 6.7.10 static_assert-declaration _Static_assert
    # is a declaration, not a statement. We additionally recognise it as a statement
    # to fix parsing of _Static_assert inside the functions.
    #
    def p_statement(self, p):
        """ statement   : labeled_statement
                        | expression_statement
                        | compound_statement
                        | selection_statement
                        | iteration_statement
                        | jump_statement
                        | pppragma_directive
                        | static_assert
        """
        p[0] = p[1]

    # A pragma is generally considered a decorator rather than an actual
    # statement. Still, for the purposes of analyzing an abstract syntax tree of
    # C code, pragma's should not be ignored and were previously treated as a
    # statement. This presents a problem for constructs that take a statement
    # such as labeled_statements, selection_statements, and
    # iteration_statements, causing a misleading structure in the AST. For
    # example, consider the following C code.
    #
    #   for (int i = 0; i < 3; i++)
    #       #pragma omp critical
    #       sum += 1;
    #
    # This code will compile and execute "sum += 1;" as the body of the for
    # loop. Previous implementations of PyCParser would render the AST for this
    # block of code as follows:
    #
    #   For:
    #     DeclList:
    #       Decl: i, [], [], []
    #         TypeDecl: i, []
    #           IdentifierType: ['int']
    #         Constant: int, 0
    #     BinaryOp: <
    #       ID: i
    #       Constant: int, 3
    #     UnaryOp: p++
    #       ID: i
    #     Pragma: omp critical
    #   Assignment: +=
    #     ID: sum
    #     Constant: int, 1
    #
    # This AST misleadingly takes the Pragma as the body of the loop and the
    # assignment then becomes a sibling of the loop.
    #
    # To solve edge cases like these, the pragmacomp_or_statement rule groups
    # a pragma and its following statement (which would otherwise be orphaned)
    # using a compound block, effectively turning the above code into:
    #
    #   for (int i = 0; i < 3; i++) {
    #       #pragma omp critical
    #       sum += 1;
    #   }
    def p_pragmacomp_or_statement(self, p):
        """ pragmacomp_or_statement     : pppragma_directive_list statement
                                        | statement
        """
        if len(p) == 3:
            p[0] = c_ast.Compound(
                block_items=p[1]+[p[2]],
                coord=self._token_coord(p, 1))
        else:
            p[0] = p[1]

    # In C, declarations can come several in a line:
    #   int x, *px, romulo = 5;
    #
    # However, for the AST, we will split them to separate Decl
    # nodes.
    #
    # This rule splits its declarations and always returns a list
    # of Decl nodes, even if it's one element long.
    #
    def p_decl_body(self, p):
        """ decl_body : declaration_specifiers init_declarator_list_opt
                      | declaration_specifiers_no_type id_init_declarator_list_opt
        """
        spec = p[1]

        # p[2] (init_declarator_list_opt) is either a list or None
        #
        if p[2] is None:
            # By the standard, you must have at least one declarator unless
            # declaring a structure tag, a union tag, or the members of an
            # enumeration.
            #
            ty = spec['type']
            s_u_or_e = (c_ast.Struct, c_ast.Union, c_ast.Enum)
            if len(ty) == 1 and isinstance(ty[0], s_u_or_e):
                decls = [c_ast.Decl(
                    name=None,
                    quals=spec['qual'],
                    align=spec['alignment'],
                    storage=spec['storage'],
                    funcspec=spec['function'],
                    type=ty[0],
                    init=None,
                    bitsize=None,
                    coord=ty[0].coord)]

            # However, this case can also occur on redeclared identifiers in
            # an inner scope.  The trouble is that the redeclared type's name
            # gets grouped into declaration_specifiers; _build_declarations
            # compensates for this.
            #
            else:
                decls = self._build_declarations(
                    spec=spec,
                    decls=[dict(decl=None, init=None)],
                    typedef_namespace=True)

        else:
            decls = self._build_declarations(
                spec=spec,
                decls=p[2],
                typedef_namespace=True)

        p[0] = decls

    # The declaration has been split to a decl_body sub-rule and
    # SEMI, because having them in a single rule created a problem
    # for defining typedefs.
    #
    # If a typedef line was directly followed by a line using the
    # type defined with the typedef, the type would not be
    # recognized. This is because to reduce the declaration rule,
    # the parser's lookahead asked for the token after SEMI, which
    # was the type from the next line, and the lexer had no chance
    # to see the updated type symbol table.
    #
    # Splitting solves this problem, because after seeing SEMI,
    # the parser reduces decl_body, which actually adds the new
    # type into the table to be seen by the lexer before the next
    # line is reached.
    def p_declaration(self, p):
        """ declaration : decl_body SEMI
        """
        p[0] = p[1]

    # Since each declaration is a list of declarations, this
    # rule will combine all the declarations and return a single
    # list
    #
    def p_declaration_list(self, p):
        """ declaration_list    : declaration
                                | declaration_list declaration
        """
        p[0] = p[1] if len(p) == 2 else p[1] + p[2]

    # To know when declaration-specifiers end and declarators begin,
    # we require declaration-specifiers to have at least one
    # type-specifier, and disallow typedef-names after we've seen any
    # type-specifier. These are both required by the spec.
    #
    def p_declaration_specifiers_no_type_1(self, p):
        """ declaration_specifiers_no_type  : type_qualifier declaration_specifiers_no_type_opt
        """
        p[0] = self._add_declaration_specifier(p[2], p[1], 'qual')

    def p_declaration_specifiers_no_type_2(self, p):
        """ declaration_specifiers_no_type  : storage_class_specifier declaration_specifiers_no_type_opt
        """
        p[0] = self._add_declaration_specifier(p[2], p[1], 'storage')

    def p_declaration_specifiers_no_type_3(self, p):
        """ declaration_specifiers_no_type  : function_specifier declaration_specifiers_no_type_opt
        """
        p[0] = self._add_declaration_specifier(p[2], p[1], 'function')

    # Without this, `typedef _Atomic(T) U` will parse incorrectly because the
    # _Atomic qualifier will match, instead of the specifier.
    def p_declaration_specifiers_no_type_4(self, p):
        """ declaration_specifiers_no_type  : atomic_specifier declaration_specifiers_no_type_opt
        """
        p[0] = self._add_declaration_specifier(p[2], p[1], 'type')

    def p_declaration_specifiers_no_type_5(self, p):
        """ declaration_specifiers_no_type  : alignment_specifier declaration_specifiers_no_type_opt
        """
        p[0] = self._add_declaration_specifier(p[2], p[1], 'alignment')

    def p_declaration_specifiers_1(self, p):
        """ declaration_specifiers  : declaration_specifiers type_qualifier
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'qual', append=True)

    def p_declaration_specifiers_2(self, p):
        """ declaration_specifiers  : declaration_specifiers storage_class_specifier
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'storage', append=True)

    def p_declaration_specifiers_3(self, p):
        """ declaration_specifiers  : declaration_specifiers function_specifier
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'function', append=True)

    def p_declaration_specifiers_4(self, p):
        """ declaration_specifiers  : declaration_specifiers type_specifier_no_typeid
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'type', append=True)

    def p_declaration_specifiers_5(self, p):
        """ declaration_specifiers  : type_specifier
        """
        p[0] = self._add_declaration_specifier(None, p[1], 'type')

    def p_declaration_specifiers_6(self, p):
        """ declaration_specifiers  : declaration_specifiers_no_type type_specifier
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'type', append=True)

    def p_declaration_specifiers_7(self, p):
        """ declaration_specifiers  : declaration_specifiers alignment_specifier
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'alignment', append=True)

    def p_storage_class_specifier(self, p):
        """ storage_class_specifier : AUTO
                                    | REGISTER
                                    | STATIC
                                    | EXTERN
                                    | TYPEDEF
                                    | _THREAD_LOCAL
        """
        p[0] = p[1]

    def p_function_specifier(self, p):
        """ function_specifier  : INLINE
                                | _NORETURN
        """
        p[0] = p[1]

    def p_type_specifier_no_typeid(self, p):
        """ type_specifier_no_typeid  : VOID
                                      | _BOOL
                                      | CHAR
                                      | SHORT
                                      | INT
                                      | LONG
                                      | FLOAT
                                      | DOUBLE
                                      | _COMPLEX
                                      | SIGNED
                                      | UNSIGNED
                                      | __INT128
        """
        p[0] = c_ast.IdentifierType([p[1]], coord=self._token_coord(p, 1))

    def p_type_specifier(self, p):
        """ type_specifier  : typedef_name
                            | enum_specifier
                            | struct_or_union_specifier
                            | type_specifier_no_typeid
                            | atomic_specifier
        """
        p[0] = p[1]

    # See section 6.7.2.4 of the C11 standard.
    def p_atomic_specifier(self, p):
        """ atomic_specifier  : _ATOMIC LPAREN type_name RPAREN
        """
        typ = p[3]
        typ.quals.append('_Atomic')
        p[0] = typ

    def p_type_qualifier(self, p):
        """ type_qualifier  : CONST
                            | RESTRICT
                            | VOLATILE
                            | _ATOMIC
        """
        p[0] = p[1]

    def p_init_declarator_list(self, p):
        """ init_declarator_list    : init_declarator
                                    | init_declarator_list COMMA init_declarator
        """
        p[0] = p[1] + [p[3]] if len(p) == 4 else [p[1]]

    # Returns a {decl=<declarator> : init=<initializer>} dictionary
    # If there's no initializer, uses None
    #
    def p_init_declarator(self, p):
        """ init_declarator : declarator
                            | declarator EQUALS initializer
        """
        p[0] = dict(decl=p[1], init=(p[3] if len(p) > 2 else None))

    def p_id_init_declarator_list(self, p):
        """ id_init_declarator_list    : id_init_declarator
                                       | id_init_declarator_list COMMA init_declarator
        """
        p[0] = p[1] + [p[3]] if len(p) == 4 else [p[1]]

    def p_id_init_declarator(self, p):
        """ id_init_declarator : id_declarator
                               | id_declarator EQUALS initializer
        """
        p[0] = dict(decl=p[1], init=(p[3] if len(p) > 2 else None))

    # Require at least one type specifier in a specifier-qualifier-list
    #
    def p_specifier_qualifier_list_1(self, p):
        """ specifier_qualifier_list    : specifier_qualifier_list type_specifier_no_typeid
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'type', append=True)

    def p_specifier_qualifier_list_2(self, p):
        """ specifier_qualifier_list    : specifier_qualifier_list type_qualifier
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'qual', append=True)

    def p_specifier_qualifier_list_3(self, p):
        """ specifier_qualifier_list  : type_specifier
        """
        p[0] = self._add_declaration_specifier(None, p[1], 'type')

    def p_specifier_qualifier_list_4(self, p):
        """ specifier_qualifier_list  : type_qualifier_list type_specifier
        """
        p[0] = dict(qual=p[1], alignment=[], storage=[], type=[p[2]], function=[])

    def p_specifier_qualifier_list_5(self, p):
        """ specifier_qualifier_list  : alignment_specifier
        """
        p[0] = dict(qual=[], alignment=[p[1]], storage=[], type=[], function=[])

    def p_specifier_qualifier_list_6(self, p):
        """ specifier_qualifier_list  : specifier_qualifier_list alignment_specifier
        """
        p[0] = self._add_declaration_specifier(p[1], p[2], 'alignment')

    # TYPEID is allowed here (and in other struct/enum related tag names), because
    # struct/enum tags reside in their own namespace and can be named the same as types
    #
    def p_struct_or_union_specifier_1(self, p):
        """ struct_or_union_specifier   : struct_or_union ID
                                        | struct_or_union TYPEID
        """
        klass = self._select_struct_union_class(p[1])
        # None means no list of members
        p[0] = klass(
            name=p[2],
            decls=None,
            coord=self._token_coord(p, 2))

    def p_struct_or_union_specifier_2(self, p):
        """ struct_or_union_specifier : struct_or_union brace_open struct_declaration_list brace_close
                                      | struct_or_union brace_open brace_close
        """
        klass = self._select_struct_union_class(p[1])
        if len(p) == 4:
            # Empty sequence means an empty list of members
            p[0] = klass(
                name=None,
                decls=[],
                coord=self._token_coord(p, 2))
        else:
            p[0] = klass(
                name=None,
                decls=p[3],
                coord=self._token_coord(p, 2))


    def p_struct_or_union_specifier_3(self, p):
        """ struct_or_union_specifier   : struct_or_union ID brace_open struct_declaration_list brace_close
                                        | struct_or_union ID brace_open brace_close
                                        | struct_or_union TYPEID brace_open struct_declaration_list brace_close
                                        | struct_or_union TYPEID brace_open brace_close
        """
        klass = self._select_struct_union_class(p[1])
        if len(p) == 5:
            # Empty sequence means an empty list of members
            p[0] = klass(
                name=p[2],
                decls=[],
                coord=self._token_coord(p, 2))
        else:
            p[0] = klass(
                name=p[2],
                decls=p[4],
                coord=self._token_coord(p, 2))

    def p_struct_or_union(self, p):
        """ struct_or_union : STRUCT
                            | UNION
        """
        p[0] = p[1]

    # Combine all declarations into a single list
    #
    def p_struct_declaration_list(self, p):
        """ struct_declaration_list     : struct_declaration
                                        | struct_declaration_list struct_declaration
        """
        if len(p) == 2:
            p[0] = p[1] or []
        else:
            p[0] = p[1] + (p[2] or [])

    def p_struct_declaration_1(self, p):
        """ struct_declaration : specifier_qualifier_list struct_declarator_list_opt SEMI
        """
        spec = p[1]
        assert 'typedef' not in spec['storage']

        if p[2] is not None:
            decls = self._build_declarations(
                spec=spec,
                decls=p[2])

        elif len(spec['type']) == 1:
            # Anonymous struct/union, gcc extension, C1x feature.
            # Although the standard only allows structs/unions here, I see no
            # reason to disallow other types since some compilers have typedefs
            # here, and pycparser isn't about rejecting all invalid code.
            #
            node = spec['type'][0]
            if isinstance(node, c_ast.Node):
                decl_type = node
            else:
                decl_type = c_ast.IdentifierType(node)

            decls = self._build_declarations(
                spec=spec,
                decls=[dict(decl=decl_type)])

        else:
            # Structure/union members can have the same names as typedefs.
            # The trouble is that the member's name gets grouped into
            # specifier_qualifier_list; _build_declarations compensates.
            #
            decls = self._build_declarations(
                spec=spec,
                decls=[dict(decl=None, init=None)])

        p[0] = decls

    def p_struct_declaration_2(self, p):
        """ struct_declaration : SEMI
        """
        p[0] = None

    def p_struct_declaration_3(self, p):
        """ struct_declaration : pppragma_directive
        """
        p[0] = [p[1]]

    def p_struct_declarator_list(self, p):
        """ struct_declarator_list  : struct_declarator
                                    | struct_declarator_list COMMA struct_declarator
        """
        p[0] = p[1] + [p[3]] if len(p) == 4 else [p[1]]

    # struct_declarator passes up a dict with the keys: decl (for
    # the underlying declarator) and bitsize (for the bitsize)
    #
    def p_struct_declarator_1(self, p):
        """ struct_declarator : declarator
        """
        p[0] = {'decl': p[1], 'bitsize': None}

    def p_struct_declarator_2(self, p):
        """ struct_declarator   : declarator COLON constant_expression
                                | COLON constant_expression
        """
        if len(p) > 3:
            p[0] = {'decl': p[1], 'bitsize': p[3]}
        else:
            p[0] = {'decl': c_ast.TypeDecl(None, None, None, None), 'bitsize': p[2]}

    def p_enum_specifier_1(self, p):
        """ enum_specifier  : ENUM ID
                            | ENUM TYPEID
        """
        p[0] = c_ast.Enum(p[2], None, self._token_coord(p, 1))

    def p_enum_specifier_2(self, p):
        """ enum_specifier  : ENUM brace_open enumerator_list brace_close
        """
        p[0] = c_ast.Enum(None, p[3], self._token_coord(p, 1))

    def p_enum_specifier_3(self, p):
        """ enum_specifier  : ENUM ID brace_open enumerator_list brace_close
                            | ENUM TYPEID brace_open enumerator_list brace_close
        """
        p[0] = c_ast.Enum(p[2], p[4], self._token_coord(p, 1))

    def p_enumerator_list(self, p):
        """ enumerator_list : enumerator
                            | enumerator_list COMMA
                            | enumerator_list COMMA enumerator
        """
        if len(p) == 2:
            p[0] = c_ast.EnumeratorList([p[1]], p[1].coord)
        elif len(p) == 3:
            p[0] = p[1]
        else:
            p[1].enumerators.append(p[3])
            p[0] = p[1]

    def p_alignment_specifier(self, p):
        """ alignment_specifier  : _ALIGNAS LPAREN type_name RPAREN
                                 | _ALIGNAS LPAREN constant_expression RPAREN
        """
        p[0] = c_ast.Alignas(p[3], self._token_coord(p, 1))

    def p_enumerator(self, p):
        """ enumerator  : ID
                        | ID EQUALS constant_expression
        """
        if len(p) == 2:
            enumerator = c_ast.Enumerator(
                        p[1], None,
                        self._token_coord(p, 1))
        else:
            enumerator = c_ast.Enumerator(
                        p[1], p[3],
                        self._token_coord(p, 1))
        self._add_identifier(enumerator.name, enumerator.coord)

        p[0] = enumerator

    def p_declarator(self, p):
        """ declarator  : id_declarator
                        | typeid_declarator
        """
        p[0] = p[1]

    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
    def p_xxx_declarator_1(self, p):
        """ xxx_declarator  : direct_xxx_declarator
        """
        p[0] = p[1]

    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
    def p_xxx_declarator_2(self, p):
        """ xxx_declarator  : pointer direct_xxx_declarator
        """
        p[0] = self._type_modify_decl(p[2], p[1])

    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
    def p_direct_xxx_declarator_1(self, p):
        """ direct_xxx_declarator   : yyy
        """
        p[0] = c_ast.TypeDecl(
            declname=p[1],
            type=None,
            quals=None,
            align=None,
            coord=self._token_coord(p, 1))

    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'))
    def p_direct_xxx_declarator_2(self, p):
        """ direct_xxx_declarator   : LPAREN xxx_declarator RPAREN
        """
        p[0] = p[2]

    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
    def p_direct_xxx_declarator_3(self, p):
        """ direct_xxx_declarator   : direct_xxx_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET
        """
        quals = (p[3] if len(p) > 5 else []) or []
        # Accept dimension qualifiers
        # Per C99 6.7.5.3 p7
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p[4] if len(p) > 5 else p[3],
            dim_quals=quals,
            coord=p[1].coord)

        p[0] = self._type_modify_decl(decl=p[1], modifier=arr)

    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
    def p_direct_xxx_declarator_4(self, p):
        """ direct_xxx_declarator   : direct_xxx_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET
                                    | direct_xxx_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET
        """
        # Using slice notation for PLY objects doesn't work in Python 3 for the
        # version of PLY embedded with pycparser; see PLY Google Code issue 30.
        # Work around that here by listing the two elements separately.
        listed_quals = [item if isinstance(item, list) else [item]
            for item in [p[3],p[4]]]
        dim_quals = [qual for sublist in listed_quals for qual in sublist
            if qual is not None]
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p[5],
            dim_quals=dim_quals,
            coord=p[1].coord)

        p[0] = self._type_modify_decl(decl=p[1], modifier=arr)

    # Special for VLAs
    #
    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
    def p_direct_xxx_declarator_5(self, p):
        """ direct_xxx_declarator   : direct_xxx_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET
        """
        arr = c_ast.ArrayDecl(
            type=None,
            dim=c_ast.ID(p[4], self._token_coord(p, 4)),
            dim_quals=p[3] if p[3] is not None else [],
            coord=p[1].coord)

        p[0] = self._type_modify_decl(decl=p[1], modifier=arr)

    @parameterized(('id', 'ID'), ('typeid', 'TYPEID'), ('typeid_noparen', 'TYPEID'))
    def p_direct_xxx_declarator_6(self, p):
        """ direct_xxx_declarator   : direct_xxx_declarator LPAREN parameter_type_list RPAREN
                                    | direct_xxx_declarator LPAREN identifier_list_opt RPAREN
        """
        func = c_ast.FuncDecl(
            args=p[3],
            type=None,
            coord=p[1].coord)

        # To see why _get_yacc_lookahead_token is needed, consider:
        #   typedef char TT;
        #   void foo(int TT) { TT = 10; }
        # Outside the function, TT is a typedef, but inside (starting and
        # ending with the braces) it's a parameter.  The trouble begins with
        # yacc's lookahead token.  We don't know if we're declaring or
        # defining a function until we see LBRACE, but if we wait for yacc to
        # trigger a rule on that token, then TT will have already been read
        # and incorrectly interpreted as TYPEID.  We need to add the
        # parameters to the scope the moment the lexer sees LBRACE.
        #
        if self._get_yacc_lookahead_token().type == "LBRACE":
            if func.args is not None:
                for param in func.args.params:
                    if isinstance(param, c_ast.EllipsisParam): break
                    self._add_identifier(param.name, param.coord)

        p[0] = self._type_modify_decl(decl=p[1], modifier=func)

    def p_pointer(self, p):
        """ pointer : TIMES type_qualifier_list_opt
                    | TIMES type_qualifier_list_opt pointer
        """
        coord = self._token_coord(p, 1)
        # Pointer decls nest from inside out. This is important when different
        # levels have different qualifiers. For example:
        #
        #  char * const * p;
        #
        # Means "pointer to const pointer to char"
        #
        # While:
        #
        #  char ** const p;
        #
        # Means "const pointer to pointer to char"
        #
        # So when we construct PtrDecl nestings, the leftmost pointer goes in
        # as the most nested type.
        nested_type = c_ast.PtrDecl(quals=p[2] or [], type=None, coord=coord)
        if len(p) > 3:
            tail_type = p[3]
            while tail_type.type is not None:
                tail_type = tail_type.type
            tail_type.type = nested_type
            p[0] = p[3]
        else:
            p[0] = nested_type

    def p_type_qualifier_list(self, p):
        """ type_qualifier_list : type_qualifier
                                | type_qualifier_list type_qualifier
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_parameter_type_list(self, p):
        """ parameter_type_list : parameter_list
                                | parameter_list COMMA ELLIPSIS
        """
        if len(p) > 2:
            p[1].params.append(c_ast.EllipsisParam(self._token_coord(p, 3)))

        p[0] = p[1]

    def p_parameter_list(self, p):
        """ parameter_list  : parameter_declaration
                            | parameter_list COMMA parameter_declaration
        """
        if len(p) == 2: # single parameter
            p[0] = c_ast.ParamList([p[1]], p[1].coord)
        else:
            p[1].params.append(p[3])
            p[0] = p[1]

    # From ISO/IEC 9899:TC2, 6.7.5.3.11:
    # "If, in a parameter declaration, an identifier can be treated either
    #  as a typedef name or as a parameter name, it shall be taken as a
    #  typedef name."
    #
    # Inside a parameter declaration, once we've reduced declaration specifiers,
    # if we shift in an LPAREN and see a TYPEID, it could be either an abstract
    # declarator or a declarator nested inside parens. This rule tells us to
    # always treat it as an abstract declarator. Therefore, we only accept
    # `id_declarator`s and `typeid_noparen_declarator`s.
    def p_parameter_declaration_1(self, p):
        """ parameter_declaration   : declaration_specifiers id_declarator
                                    | declaration_specifiers typeid_noparen_declarator
        """
        spec = p[1]
        if not spec['type']:
            spec['type'] = [c_ast.IdentifierType(['int'],
                coord=self._token_coord(p, 1))]
        p[0] = self._build_declarations(
            spec=spec,
            decls=[dict(decl=p[2])])[0]

    def p_parameter_declaration_2(self, p):
        """ parameter_declaration   : declaration_specifiers abstract_declarator_opt
        """
        spec = p[1]
        if not spec['type']:
            spec['type'] = [c_ast.IdentifierType(['int'],
                coord=self._token_coord(p, 1))]

        # Parameters can have the same names as typedefs.  The trouble is that
        # the parameter's name gets grouped into declaration_specifiers, making
        # it look like an old-style declaration; compensate.
        #
        if len(spec['type']) > 1 and len(spec['type'][-1].names) == 1 and \
                self._is_type_in_scope(spec['type'][-1].names[0]):
            decl = self._build_declarations(
                    spec=spec,
                    decls=[dict(decl=p[2], init=None)])[0]

        # This truly is an old-style parameter declaration
        #
        else:
            decl = c_ast.Typename(
                name='',
                quals=spec['qual'],
                align=None,
                type=p[2] or c_ast.TypeDecl(None, None, None, None),
                coord=self._token_coord(p, 2))
            typename = spec['type']
            decl = self._fix_decl_name_type(decl, typename)

        p[0] = decl

    def p_identifier_list(self, p):
        """ identifier_list : identifier
                            | identifier_list COMMA identifier
        """
        if len(p) == 2: # single parameter
            p[0] = c_ast.ParamList([p[1]], p[1].coord)
        else:
            p[1].params.append(p[3])
            p[0] = p[1]

    def p_initializer_1(self, p):
        """ initializer : assignment_expression
        """
        p[0] = p[1]

    def p_initializer_2(self, p):
        """ initializer : brace_open initializer_list_opt brace_close
                        | brace_open initializer_list COMMA brace_close
        """
        if p[2] is None:
            p[0] = c_ast.InitList([], self._token_coord(p, 1))
        else:
            p[0] = p[2]

    def p_initializer_list(self, p):
        """ initializer_list    : designation_opt initializer
                                | initializer_list COMMA designation_opt initializer
        """
        if len(p) == 3: # single initializer
            init = p[2] if p[1] is None else c_ast.NamedInitializer(p[1], p[2])
            p[0] = c_ast.InitList([init], p[2].coord)
        else:
            init = p[4] if p[3] is None else c_ast.NamedInitializer(p[3], p[4])
            p[1].exprs.append(init)
            p[0] = p[1]

    def p_designation(self, p):
        """ designation : designator_list EQUALS
        """
        p[0] = p[1]

    # Designators are represented as a list of nodes, in the order in which
    # they're written in the code.
    #
    def p_designator_list(self, p):
        """ designator_list : designator
                            | designator_list designator
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_designator(self, p):
        """ designator  : LBRACKET constant_expression RBRACKET
                        | PERIOD identifier
        """
        p[0] = p[2]

    def p_type_name(self, p):
        """ type_name   : specifier_qualifier_list abstract_declarator_opt
        """
        typename = c_ast.Typename(
            name='',
            quals=p[1]['qual'][:],
            align=None,
            type=p[2] or c_ast.TypeDecl(None, None, None, None),
            coord=self._token_coord(p, 2))

        p[0] = self._fix_decl_name_type(typename, p[1]['type'])

    def p_abstract_declarator_1(self, p):
        """ abstract_declarator     : pointer
        """
        dummytype = c_ast.TypeDecl(None, None, None, None)
        p[0] = self._type_modify_decl(
            decl=dummytype,
            modifier=p[1])

    def p_abstract_declarator_2(self, p):
        """ abstract_declarator     : pointer direct_abstract_declarator
        """
        p[0] = self._type_modify_decl(p[2], p[1])

    def p_abstract_declarator_3(self, p):
        """ abstract_declarator     : direct_abstract_declarator
        """
        p[0] = p[1]

    # Creating and using direct_abstract_declarator_opt here
    # instead of listing both direct_abstract_declarator and the
    # lack of it in the beginning of _1 and _2 caused two
    # shift/reduce errors.
    #
    def p_direct_abstract_declarator_1(self, p):
        """ direct_abstract_declarator  : LPAREN abstract_declarator RPAREN """
        p[0] = p[2]

    def p_direct_abstract_declarator_2(self, p):
        """ direct_abstract_declarator  : direct_abstract_declarator LBRACKET assignment_expression_opt RBRACKET
        """
        arr = c_ast.ArrayDecl(
            type=None,
            dim=p[3],
            dim_quals=[],
            coord=p[1].coord)

        p[0] = self._type_modify_decl(decl=p[1], modifier=arr)

    def p_direct_abstract_declarator_3(self, p):
        """ direct_abstract_declarator  : LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET
        """
        quals = (p[2] if len(p) > 4 else []) or []
        p[0] = c_ast.ArrayDecl(
            type=c_ast.TypeDecl(None, None, None, None),
            dim=p[3] if len(p) > 4 else p[2],
            dim_quals=quals,
            coord=self._token_coord(p, 1))

    def p_direct_abstract_declarator_4(self, p):
        """ direct_abstract_declarator  : direct_abstract_declarator LBRACKET TIMES RBRACKET
        """
        arr = c_ast.ArrayDecl(
            type=None,
            dim=c_ast.ID(p[3], self._token_coord(p, 3)),
            dim_quals=[],
            coord=p[1].coord)

        p[0] = self._type_modify_decl(decl=p[1], modifier=arr)

    def p_direct_abstract_declarator_5(self, p):
        """ direct_abstract_declarator  : LBRACKET TIMES RBRACKET
        """
        p[0] = c_ast.ArrayDecl(
            type=c_ast.TypeDecl(None, None, None, None),
            dim=c_ast.ID(p[3], self._token_coord(p, 3)),
            dim_quals=[],
            coord=self._token_coord(p, 1))

    def p_direct_abstract_declarator_6(self, p):
        """ direct_abstract_declarator  : direct_abstract_declarator LPAREN parameter_type_list_opt RPAREN
        """
        func = c_ast.FuncDecl(
            args=p[3],
            type=None,
            coord=p[1].coord)

        p[0] = self._type_modify_decl(decl=p[1], modifier=func)

    def p_direct_abstract_declarator_7(self, p):
        """ direct_abstract_declarator  : LPAREN parameter_type_list_opt RPAREN
        """
        p[0] = c_ast.FuncDecl(
            args=p[2],
            type=c_ast.TypeDecl(None, None, None, None),
            coord=self._token_coord(p, 1))

    # declaration is a list, statement isn't. To make it consistent, block_item
    # will always be a list
    #
    def p_block_item(self, p):
        """ block_item  : declaration
                        | statement
        """
        p[0] = p[1] if isinstance(p[1], list) else [p[1]]

    # Since we made block_item a list, this just combines lists
    #
    def p_block_item_list(self, p):
        """ block_item_list : block_item
                            | block_item_list block_item
        """
        # Empty block items (plain ';') produce [None], so ignore them
        p[0] = p[1] if (len(p) == 2 or p[2] == [None]) else p[1] + p[2]

    def p_compound_statement_1(self, p):
        """ compound_statement : brace_open block_item_list_opt brace_close """
        p[0] = c_ast.Compound(
            block_items=p[2],
            coord=self._token_coord(p, 1))

    def p_labeled_statement_1(self, p):
        """ labeled_statement : ID COLON pragmacomp_or_statement """
        p[0] = c_ast.Label(p[1], p[3], self._token_coord(p, 1))

    def p_labeled_statement_2(self, p):
        """ labeled_statement : CASE constant_expression COLON pragmacomp_or_statement """
        p[0] = c_ast.Case(p[2], [p[4]], self._token_coord(p, 1))

    def p_labeled_statement_3(self, p):
        """ labeled_statement : DEFAULT COLON pragmacomp_or_statement """
        p[0] = c_ast.Default([p[3]], self._token_coord(p, 1))

    def p_selection_statement_1(self, p):
        """ selection_statement : IF LPAREN expression RPAREN pragmacomp_or_statement """
        p[0] = c_ast.If(p[3], p[5], None, self._token_coord(p, 1))

    def p_selection_statement_2(self, p):
        """ selection_statement : IF LPAREN expression RPAREN statement ELSE pragmacomp_or_statement """
        p[0] = c_ast.If(p[3], p[5], p[7], self._token_coord(p, 1))

    def p_selection_statement_3(self, p):
        """ selection_statement : SWITCH LPAREN expression RPAREN pragmacomp_or_statement """
        p[0] = fix_switch_cases(
                c_ast.Switch(p[3], p[5], self._token_coord(p, 1)))

    def p_iteration_statement_1(self, p):
        """ iteration_statement : WHILE LPAREN expression RPAREN pragmacomp_or_statement """
        p[0] = c_ast.While(p[3], p[5], self._token_coord(p, 1))

    def p_iteration_statement_2(self, p):
        """ iteration_statement : DO pragmacomp_or_statement WHILE LPAREN expression RPAREN SEMI """
        p[0] = c_ast.DoWhile(p[5], p[2], self._token_coord(p, 1))

    def p_iteration_statement_3(self, p):
        """ iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement """
        p[0] = c_ast.For(p[3], p[5], p[7], p[9], self._token_coord(p, 1))

    def p_iteration_statement_4(self, p):
        """ iteration_statement : FOR LPAREN declaration expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement """
        p[0] = c_ast.For(c_ast.DeclList(p[3], self._token_coord(p, 1)),
                         p[4], p[6], p[8], self._token_coord(p, 1))

    def p_jump_statement_1(self, p):
        """ jump_statement  : GOTO ID SEMI """
        p[0] = c_ast.Goto(p[2], self._token_coord(p, 1))

    def p_jump_statement_2(self, p):
        """ jump_statement  : BREAK SEMI """
        p[0] = c_ast.Break(self._token_coord(p, 1))

    def p_jump_statement_3(self, p):
        """ jump_statement  : CONTINUE SEMI """
        p[0] = c_ast.Continue(self._token_coord(p, 1))

    def p_jump_statement_4(self, p):
        """ jump_statement  : RETURN expression SEMI
                            | RETURN SEMI
        """
        p[0] = c_ast.Return(p[2] if len(p) == 4 else None, self._token_coord(p, 1))

    def p_expression_statement(self, p):
        """ expression_statement : expression_opt SEMI """
        if p[1] is None:
            p[0] = c_ast.EmptyStatement(self._token_coord(p, 2))
        else:
            p[0] = p[1]

    def p_expression(self, p):
        """ expression  : assignment_expression
                        | expression COMMA assignment_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            if not isinstance(p[1], c_ast.ExprList):
                p[1] = c_ast.ExprList([p[1]], p[1].coord)

            p[1].exprs.append(p[3])
            p[0] = p[1]

    def p_parenthesized_compound_expression(self, p):
        """ assignment_expression : LPAREN compound_statement RPAREN """
        p[0] = p[2]

    def p_typedef_name(self, p):
        """ typedef_name : TYPEID """
        p[0] = c_ast.IdentifierType([p[1]], coord=self._token_coord(p, 1))

    def p_assignment_expression(self, p):
        """ assignment_expression   : conditional_expression
                                    | unary_expression assignment_operator assignment_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = c_ast.Assignment(p[2], p[1], p[3], p[1].coord)

    # K&R2 defines these as many separate rules, to encode
    # precedence and associativity. Why work hard ? I'll just use
    # the built in precedence/associativity specification feature
    # of PLY. (see precedence declaration above)
    #
    def p_assignment_operator(self, p):
        """ assignment_operator : EQUALS
                                | XOREQUAL
                                | TIMESEQUAL
                                | DIVEQUAL
                                | MODEQUAL
                                | PLUSEQUAL
                                | MINUSEQUAL
                                | LSHIFTEQUAL
                                | RSHIFTEQUAL
                                | ANDEQUAL
                                | OREQUAL
        """
        p[0] = p[1]

    def p_constant_expression(self, p):
        """ constant_expression : conditional_expression """
        p[0] = p[1]

    def p_conditional_expression(self, p):
        """ conditional_expression  : binary_expression
                                    | binary_expression CONDOP expression COLON conditional_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = c_ast.TernaryOp(p[1], p[3], p[5], p[1].coord)

    def p_binary_expression(self, p):
        """ binary_expression   : cast_expression
                                | binary_expression TIMES binary_expression
                                | binary_expression DIVIDE binary_expression
                                | binary_expression MOD binary_expression
                                | binary_expression PLUS binary_expression
                                | binary_expression MINUS binary_expression
                                | binary_expression RSHIFT binary_expression
                                | binary_expression LSHIFT binary_expression
                                | binary_expression LT binary_expression
                                | binary_expression LE binary_expression
                                | binary_expression GE binary_expression
                                | binary_expression GT binary_expression
                                | binary_expression EQ binary_expression
                                | binary_expression NE binary_expression
                                | binary_expression AND binary_expression
                                | binary_expression OR binary_expression
                                | binary_expression XOR binary_expression
                                | binary_expression LAND binary_expression
                                | binary_expression LOR binary_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = c_ast.BinaryOp(p[2], p[1], p[3], p[1].coord)

    def p_cast_expression_1(self, p):
        """ cast_expression : unary_expression """
        p[0] = p[1]

    def p_cast_expression_2(self, p):
        """ cast_expression : LPAREN type_name RPAREN cast_expression """
        p[0] = c_ast.Cast(p[2], p[4], self._token_coord(p, 1))

    def p_unary_expression_1(self, p):
        """ unary_expression    : postfix_expression """
        p[0] = p[1]

    def p_unary_expression_2(self, p):
        """ unary_expression    : PLUSPLUS unary_expression
                                | MINUSMINUS unary_expression
                                | unary_operator cast_expression
        """
        p[0] = c_ast.UnaryOp(p[1], p[2], p[2].coord)

    def p_unary_expression_3(self, p):
        """ unary_expression    : SIZEOF unary_expression
                                | SIZEOF LPAREN type_name RPAREN
                                | _ALIGNOF LPAREN type_name RPAREN
        """
        p[0] = c_ast.UnaryOp(
            p[1],
            p[2] if len(p) == 3 else p[3],
            self._token_coord(p, 1))

    def p_unary_operator(self, p):
        """ unary_operator  : AND
                            | TIMES
                            | PLUS
                            | MINUS
                            | NOT
                            | LNOT
        """
        p[0] = p[1]

    def p_postfix_expression_1(self, p):
        """ postfix_expression  : primary_expression """
        p[0] = p[1]

    def p_postfix_expression_2(self, p):
        """ postfix_expression  : postfix_expression LBRACKET expression RBRACKET """
        p[0] = c_ast.ArrayRef(p[1], p[3], p[1].coord)

    def p_postfix_expression_3(self, p):
        """ postfix_expression  : postfix_expression LPAREN argument_expression_list RPAREN
                                | postfix_expression LPAREN RPAREN
        """
        p[0] = c_ast.FuncCall(p[1], p[3] if len(p) == 5 else None, p[1].coord)

    def p_postfix_expression_4(self, p):
        """ postfix_expression  : postfix_expression PERIOD ID
                                | postfix_expression PERIOD TYPEID
                                | postfix_expression ARROW ID
                                | postfix_expression ARROW TYPEID
        """
        field = c_ast.ID(p[3], self._token_coord(p, 3))
        p[0] = c_ast.StructRef(p[1], p[2], field, p[1].coord)

    def p_postfix_expression_5(self, p):
        """ postfix_expression  : postfix_expression PLUSPLUS
                                | postfix_expression MINUSMINUS
        """
        p[0] = c_ast.UnaryOp('p' + p[2], p[1], p[1].coord)

    def p_postfix_expression_6(self, p):
        """ postfix_expression  : LPAREN type_name RPAREN brace_open initializer_list brace_close
                                | LPAREN type_name RPAREN brace_open initializer_list COMMA brace_close
        """
        p[0] = c_ast.CompoundLiteral(p[2], p[5])

    def p_primary_expression_1(self, p):
        """ primary_expression  : identifier """
        p[0] = p[1]

    def p_primary_expression_2(self, p):
        """ primary_expression  : constant """
        p[0] = p[1]

    def p_primary_expression_3(self, p):
        """ primary_expression  : unified_string_literal
                                | unified_wstring_literal
        """
        p[0] = p[1]

    def p_primary_expression_4(self, p):
        """ primary_expression  : LPAREN expression RPAREN """
        p[0] = p[2]

    def p_primary_expression_5(self, p):
        """ primary_expression  : OFFSETOF LPAREN type_name COMMA offsetof_member_designator RPAREN
        """
        coord = self._token_coord(p, 1)
        p[0] = c_ast.FuncCall(c_ast.ID(p[1], coord),
                              c_ast.ExprList([p[3], p[5]], coord),
                              coord)

    def p_offsetof_member_designator(self, p):
        """ offsetof_member_designator : identifier
                                         | offsetof_member_designator PERIOD identifier
                                         | offsetof_member_designator LBRACKET expression RBRACKET
        """
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 4:
            p[0] = c_ast.StructRef(p[1], p[2], p[3], p[1].coord)
        elif len(p) == 5:
            p[0] = c_ast.ArrayRef(p[1], p[3], p[1].coord)
        else:
            raise NotImplementedError("Unexpected parsing state. len(p): %u" % len(p))

    def p_argument_expression_list(self, p):
        """ argument_expression_list    : assignment_expression
                                        | argument_expression_list COMMA assignment_expression
        """
        if len(p) == 2: # single expr
            p[0] = c_ast.ExprList([p[1]], p[1].coord)
        else:
            p[1].exprs.append(p[3])
            p[0] = p[1]

    def p_identifier(self, p):
        """ identifier  : ID """
        p[0] = c_ast.ID(p[1], self._token_coord(p, 1))

    def p_constant_1(self, p):
        """ constant    : INT_CONST_DEC
                        | INT_CONST_OCT
                        | INT_CONST_HEX
                        | INT_CONST_BIN
                        | INT_CONST_CHAR
        """
        uCount = 0
        lCount = 0
        for x in p[1][-3:]:
            if x in ('l', 'L'):
                lCount += 1
            elif x in ('u', 'U'):
                uCount += 1
        t = ''
        if uCount > 1:
             raise ValueError('Constant cannot have more than one u/U suffix.')
        elif lCount > 2:
             raise ValueError('Constant cannot have more than two l/L suffix.')
        prefix = 'unsigned ' * uCount + 'long ' * lCount
        p[0] = c_ast.Constant(
            prefix + 'int', p[1], self._token_coord(p, 1))

    def p_constant_2(self, p):
        """ constant    : FLOAT_CONST
                        | HEX_FLOAT_CONST
        """
        if 'x' in p[1].lower():
            t = 'float'
        else:
            if p[1][-1] in ('f', 'F'):
                t = 'float'
            elif p[1][-1] in ('l', 'L'):
                t = 'long double'
            else:
                t = 'double'

        p[0] = c_ast.Constant(
            t, p[1], self._token_coord(p, 1))

    def p_constant_3(self, p):
        """ constant    : CHAR_CONST
                        | WCHAR_CONST
                        | U8CHAR_CONST
                        | U16CHAR_CONST
                        | U32CHAR_CONST
        """
        p[0] = c_ast.Constant(
            'char', p[1], self._token_coord(p, 1))

    # The "unified" string and wstring literal rules are for supporting
    # concatenation of adjacent string literals.
    # I.e. "hello " "world" is seen by the C compiler as a single string literal
    # with the value "hello world"
    #
    def p_unified_string_literal(self, p):
        """ unified_string_literal  : STRING_LITERAL
                                    | unified_string_literal STRING_LITERAL
        """
        if len(p) == 2: # single literal
            p[0] = c_ast.Constant(
                'string', p[1], self._token_coord(p, 1))
        else:
            p[1].value = p[1].value[:-1] + p[2][1:]
            p[0] = p[1]

    def p_unified_wstring_literal(self, p):
        """ unified_wstring_literal : WSTRING_LITERAL
                                    | U8STRING_LITERAL
                                    | U16STRING_LITERAL
                                    | U32STRING_LITERAL
                                    | unified_wstring_literal WSTRING_LITERAL
                                    | unified_wstring_literal U8STRING_LITERAL
                                    | unified_wstring_literal U16STRING_LITERAL
                                    | unified_wstring_literal U32STRING_LITERAL
        """
        if len(p) == 2: # single literal
            p[0] = c_ast.Constant(
                'string', p[1], self._token_coord(p, 1))
        else:
            p[1].value = p[1].value.rstrip()[:-1] + p[2][2:]
            p[0] = p[1]

    def p_brace_open(self, p):
        """ brace_open  :   LBRACE
        """
        p[0] = p[1]
        p.set_lineno(0, p.lineno(1))

    def p_brace_close(self, p):
        """ brace_close :   RBRACE
        """
        p[0] = p[1]
        p.set_lineno(0, p.lineno(1))

    def p_empty(self, p):
        'empty : '
        p[0] = None

    def p_error(self, p):
        # If error recovery is added here in the future, make sure
        # _get_yacc_lookahead_token still works!
        #
        if p:
            self._parse_error(
                'before: %s' % p.value,
                self._coord(lineno=p.lineno,
                            column=self.clex.find_tok_column(p)))
        else:
            self._parse_error('At end of input', self.clex.filename)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_core.py ===
from __future__ import annotations

import importlib
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

from pandas._config import get_option

from pandas.util._decorators import (
    Appender,
    Substitution,
)

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )
    import types

    from matplotlib.axes import Axes
    import numpy as np

    from pandas._typing import IndexLabel

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.groupby.generic import DataFrameGroupBy


def hist_series(
    self: Series,
    by=None,
    ax=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot: float | None = None,
    ylabelsize: int | None = None,
    yrot: float | None = None,
    figsize: tuple[int, int] | None = None,
    bins: int | Sequence[int] = 10,
    backend: str | None = None,
    legend: bool = False,
    **kwargs,
):
    """
    Draw histogram of the input series using matplotlib.

    Parameters
    ----------
    by : object, optional
        If passed, then used to form histograms for separate groups.
    ax : matplotlib axis object
        If not passed, uses gca().
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels.
    figsize : tuple, default None
        Figure size in inches by default.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    legend : bool, default False
        Whether to show the legend.

    **kwargs
        To be passed to the actual plotting function.

    Returns
    -------
    matplotlib.AxesSubplot
        A histogram plot.

    See Also
    --------
    matplotlib.axes.Axes.hist : Plot a histogram using matplotlib.

    Examples
    --------
    For Series:

    .. plot::
        :context: close-figs

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> hist = ser.hist()

    For Groupby:

    .. plot::
        :context: close-figs

        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> hist = ser.groupby(level=0).hist()
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(
        self,
        by=by,
        ax=ax,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        figsize=figsize,
        bins=bins,
        legend=legend,
        **kwargs,
    )


def hist_frame(
    data: DataFrame,
    column: IndexLabel | None = None,
    by=None,
    grid: bool = True,
    xlabelsize: int | None = None,
    xrot: float | None = None,
    ylabelsize: int | None = None,
    yrot: float | None = None,
    ax=None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[int, int] | None = None,
    layout: tuple[int, int] | None = None,
    bins: int | Sequence[int] = 10,
    backend: str | None = None,
    legend: bool = False,
    **kwargs,
):
    """
    Make a histogram of the DataFrame's columns.

    A `histogram`_ is a representation of the distribution of data.
    This function calls :meth:`matplotlib.pyplot.hist`, on each series in
    the DataFrame, resulting in one histogram per column.

    .. _histogram: https://en.wikipedia.org/wiki/Histogram

    Parameters
    ----------
    data : DataFrame
        The pandas object holding the data.
    column : str or sequence, optional
        If passed, will be used to limit data to a subset of columns.
    by : object, optional
        If passed, then used to form histograms for separate groups.
    grid : bool, default True
        Whether to show axis grid lines.
    xlabelsize : int, default None
        If specified changes the x-axis label size.
    xrot : float, default None
        Rotation of x axis labels. For example, a value of 90 displays the
        x labels rotated 90 degrees clockwise.
    ylabelsize : int, default None
        If specified changes the y-axis label size.
    yrot : float, default None
        Rotation of y axis labels. For example, a value of 90 displays the
        y labels rotated 90 degrees clockwise.
    ax : Matplotlib axes object, default None
        The axes to plot the histogram on.
    sharex : bool, default True if ax is None else False
        In case subplots=True, share x axis and set some x axis labels to
        invisible; defaults to True if ax is None otherwise False if an ax
        is passed in.
        Note that passing in both an ax and sharex=True will alter all x axis
        labels for all subplots in a figure.
    sharey : bool, default False
        In case subplots=True, share y axis and set some y axis labels to
        invisible.
    figsize : tuple, optional
        The size in inches of the figure to create. Uses the value in
        `matplotlib.rcParams` by default.
    layout : tuple, optional
        Tuple of (rows, columns) for the layout of the histograms.
    bins : int or sequence, default 10
        Number of histogram bins to be used. If an integer is given, bins + 1
        bin edges are calculated and returned. If bins is a sequence, gives
        bin edges, including left edge of first bin and right edge of last
        bin. In this case, bins is returned unmodified.

    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.

    legend : bool, default False
        Whether to show the legend.

    **kwargs
        All other plotting keyword arguments to be passed to
        :meth:`matplotlib.pyplot.hist`.

    Returns
    -------
    matplotlib.AxesSubplot or numpy.ndarray of them

    See Also
    --------
    matplotlib.pyplot.hist : Plot a histogram using matplotlib.

    Examples
    --------
    This example draws a histogram based on the length and width of
    some animals, displayed in three bins

    .. plot::
        :context: close-figs

        >>> data = {'length': [1.5, 0.5, 1.2, 0.9, 3],
        ...         'width': [0.7, 0.2, 0.15, 0.2, 1.1]}
        >>> index = ['pig', 'rabbit', 'duck', 'chicken', 'horse']
        >>> df = pd.DataFrame(data, index=index)
        >>> hist = df.hist(bins=3)
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(
        data,
        column=column,
        by=by,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        ax=ax,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
        legend=legend,
        bins=bins,
        **kwargs,
    )


_boxplot_doc = """
Make a box plot from DataFrame columns.

Make a box-and-whisker plot from DataFrame columns, optionally grouped
by some other columns. A box plot is a method for graphically depicting
groups of numerical data through their quartiles.
The box extends from the Q1 to Q3 quartile values of the data,
with a line at the median (Q2). The whiskers extend from the edges
of box to show the range of the data. By default, they extend no more than
`1.5 * IQR (IQR = Q3 - Q1)` from the edges of the box, ending at the farthest
data point within that interval. Outliers are plotted as separate dots.

For further details see
Wikipedia's entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`_.

Parameters
----------
%(data)s\
column : str or list of str, optional
    Column name or list of names, or vector.
    Can be any valid input to :meth:`pandas.DataFrame.groupby`.
by : str or array-like, optional
    Column in the DataFrame to :meth:`pandas.DataFrame.groupby`.
    One box-plot will be done per value of columns in `by`.
ax : object of class matplotlib.axes.Axes, optional
    The matplotlib axes to be used by boxplot.
fontsize : float or str
    Tick label font size in points or as a string (e.g., `large`).
rot : float, default 0
    The rotation angle of labels (in degrees)
    with respect to the screen coordinate system.
grid : bool, default True
    Setting this to True will show the grid.
figsize : A tuple (width, height) in inches
    The size of the figure to create in matplotlib.
layout : tuple (rows, columns), optional
    For example, (3, 5) will display the subplots
    using 3 rows and 5 columns, starting from the top-left.
return_type : {'axes', 'dict', 'both'} or None, default 'axes'
    The kind of object to return. The default is ``axes``.

    * 'axes' returns the matplotlib axes the boxplot is drawn on.
    * 'dict' returns a dictionary whose values are the matplotlib
      Lines of the boxplot.
    * 'both' returns a namedtuple with the axes and dict.
    * when grouping with ``by``, a Series mapping columns to
      ``return_type`` is returned.

      If ``return_type`` is `None`, a NumPy array
      of axes with the same shape as ``layout`` is returned.
%(backend)s\

**kwargs
    All other plotting keyword arguments to be passed to
    :func:`matplotlib.pyplot.boxplot`.

Returns
-------
result
    See Notes.

See Also
--------
pandas.Series.plot.hist: Make a histogram.
matplotlib.pyplot.boxplot : Matplotlib equivalent plot.

Notes
-----
The return type depends on the `return_type` parameter:

* 'axes' : object of class matplotlib.axes.Axes
* 'dict' : dict of matplotlib.lines.Line2D objects
* 'both' : a namedtuple with structure (ax, lines)

For data grouped with ``by``, return a Series of the above or a numpy
array:

* :class:`~pandas.Series`
* :class:`~numpy.array` (for ``return_type = None``)

Use ``return_type='dict'`` when you want to tweak the appearance
of the lines after plotting. In this case a dict containing the Lines
making up the boxes, caps, fliers, medians, and whiskers is returned.

Examples
--------

Boxplots can be created for every column in the dataframe
by ``df.boxplot()`` or indicating the columns to be used:

.. plot::
    :context: close-figs

    >>> np.random.seed(1234)
    >>> df = pd.DataFrame(np.random.randn(10, 4),
    ...                   columns=['Col1', 'Col2', 'Col3', 'Col4'])
    >>> boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])  # doctest: +SKIP

Boxplots of variables distributions grouped by the values of a third
variable can be created using the option ``by``. For instance:

.. plot::
    :context: close-figs

    >>> df = pd.DataFrame(np.random.randn(10, 2),
    ...                   columns=['Col1', 'Col2'])
    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
    ...                      'B', 'B', 'B', 'B', 'B'])
    >>> boxplot = df.boxplot(by='X')

A list of strings (i.e. ``['X', 'Y']``) can be passed to boxplot
in order to group the data by combination of the variables in the x-axis:

.. plot::
    :context: close-figs

    >>> df = pd.DataFrame(np.random.randn(10, 3),
    ...                   columns=['Col1', 'Col2', 'Col3'])
    >>> df['X'] = pd.Series(['A', 'A', 'A', 'A', 'A',
    ...                      'B', 'B', 'B', 'B', 'B'])
    >>> df['Y'] = pd.Series(['A', 'B', 'A', 'B', 'A',
    ...                      'B', 'A', 'B', 'A', 'B'])
    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])

The layout of boxplot can be adjusted giving a tuple to ``layout``:

.. plot::
    :context: close-figs

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      layout=(2, 1))

Additional formatting can be done to the boxplot, like suppressing the grid
(``grid=False``), rotating the labels in the x-axis (i.e. ``rot=45``)
or changing the fontsize (i.e. ``fontsize=15``):

.. plot::
    :context: close-figs

    >>> boxplot = df.boxplot(grid=False, rot=45, fontsize=15)  # doctest: +SKIP

The parameter ``return_type`` can be used to select the type of element
returned by `boxplot`.  When ``return_type='axes'`` is selected,
the matplotlib axes on which the boxplot is drawn are returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], return_type='axes')
    >>> type(boxplot)
    <class 'matplotlib.axes._axes.Axes'>

When grouping with ``by``, a Series mapping columns to ``return_type``
is returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      return_type='axes')
    >>> type(boxplot)
    <class 'pandas.core.series.Series'>

If ``return_type`` is `None`, a NumPy array of axes with the same shape
as ``layout`` is returned:

    >>> boxplot = df.boxplot(column=['Col1', 'Col2'], by='X',
    ...                      return_type=None)
    >>> type(boxplot)
    <class 'numpy.ndarray'>
"""

_backend_doc = """\
backend : str, default None
    Backend to use instead of the backend specified in the option
    ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
    specify the ``plotting.backend`` for the whole session, set
    ``pd.options.plotting.backend``.
"""


_bar_or_line_doc = """
        Parameters
        ----------
        x : label or position, optional
            Allows plotting of one column versus another. If not specified,
            the index of the DataFrame is used.
        y : label or position, optional
            Allows plotting of one column versus another. If not specified,
            all numerical columns are used.
        color : str, array-like, or dict, optional
            The color for each of the DataFrame's columns. Possible values are:

            - A single color string referred to by name, RGB or RGBA code,
                for instance 'red' or '#a98d19'.

            - A sequence of color strings referred to by name, RGB or RGBA
                code, which will be used for each column recursively. For
                instance ['green','yellow'] each column's %(kind)s will be filled in
                green or yellow, alternatively. If there is only a single column to
                be plotted, then only the first color from the color list will be
                used.

            - A dict of the form {column name : color}, so that each column will be
                colored accordingly. For example, if your columns are called `a` and
                `b`, then passing {'a': 'green', 'b': 'red'} will color %(kind)ss for
                column `a` in green and %(kind)ss for column `b` in red.

        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            An ndarray is returned with one :class:`matplotlib.axes.Axes`
            per column when ``subplots=True``.
"""


@Substitution(data="data : DataFrame\n    The data to visualize.\n", backend="")
@Appender(_boxplot_doc)
def boxplot(
    data: DataFrame,
    column: str | list[str] | None = None,
    by: str | list[str] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: str | None = None,
    **kwargs,
):
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.boxplot(
        data,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


@Substitution(data="", backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self: DataFrame,
    column=None,
    by=None,
    ax=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout=None,
    return_type=None,
    backend=None,
    **kwargs,
):
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame(
        self,
        column=column,
        by=by,
        ax=ax,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        figsize=figsize,
        layout=layout,
        return_type=return_type,
        **kwargs,
    )


def boxplot_frame_groupby(
    grouped: DataFrameGroupBy,
    subplots: bool = True,
    column=None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    ax=None,
    figsize: tuple[float, float] | None = None,
    layout=None,
    sharex: bool = False,
    sharey: bool = True,
    backend=None,
    **kwargs,
):
    """
    Make box plots from DataFrameGroupBy data.

    Parameters
    ----------
    grouped : Grouped DataFrame
    subplots : bool
        * ``False`` - no subplots will be used
        * ``True`` - create a subplot for each group.

    column : column name or list of names, or vector
        Can be any valid input to groupby.
    fontsize : float or str
    rot : label rotation angle
    grid : Setting this to True will show the grid
    ax : Matplotlib axis object, default None
    figsize : A tuple (width, height) in inches
    layout : tuple (optional)
        The layout of the plot: (rows, columns).
    sharex : bool, default False
        Whether x-axes will be shared among subplots.
    sharey : bool, default True
        Whether y-axes will be shared among subplots.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        All other plotting keyword arguments to be passed to
        matplotlib's boxplot function.

    Returns
    -------
    dict of key/value = group key/DataFrame.boxplot return value
    or DataFrame.boxplot return value in case subplots=figures=False

    Examples
    --------
    You can create boxplots for grouped data and show them as separate subplots:

    .. plot::
        :context: close-figs

        >>> import itertools
        >>> tuples = [t for t in itertools.product(range(1000), range(4))]
        >>> index = pd.MultiIndex.from_tuples(tuples, names=['lvl0', 'lvl1'])
        >>> data = np.random.randn(len(index), 4)
        >>> df = pd.DataFrame(data, columns=list('ABCD'), index=index)
        >>> grouped = df.groupby(level='lvl1')
        >>> grouped.boxplot(rot=45, fontsize=12, figsize=(8, 10))  # doctest: +SKIP

    The ``subplots=False`` option shows the boxplots in a single figure.

    .. plot::
        :context: close-figs

        >>> grouped.boxplot(subplots=False, rot=45, fontsize=12)  # doctest: +SKIP
    """
    plot_backend = _get_plot_backend(backend)
    return plot_backend.boxplot_frame_groupby(
        grouped,
        subplots=subplots,
        column=column,
        fontsize=fontsize,
        rot=rot,
        grid=grid,
        ax=ax,
        figsize=figsize,
        layout=layout,
        sharex=sharex,
        sharey=sharey,
        **kwargs,
    )


class PlotAccessor(PandasObject):
    """
    Make plots of Series or DataFrame.

    Uses the backend specified by the
    option ``plotting.backend``. By default, matplotlib is used.

    Parameters
    ----------
    data : Series or DataFrame
        The object for which the method is called.
    x : label or position, default None
        Only used if data is a DataFrame.
    y : label, position or list of label, positions, default None
        Allows plotting of one column versus another. Only used if data is a
        DataFrame.
    kind : str
        The kind of plot to produce:

        - 'line' : line plot (default)
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : boxplot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot (DataFrame only)
        - 'hexbin' : hexbin plot (DataFrame only)
    ax : matplotlib axes object, default None
        An axes of the current figure.
    subplots : bool or sequence of iterables, default False
        Whether to group columns into subplots:

        - ``False`` : No subplots will be used
        - ``True`` : Make separate subplots for each column.
        - sequence of iterables of column labels: Create a subplot for each
          group of columns. For example `[('a', 'c'), ('b', 'd')]` will
          create 2 subplots: one with columns 'a' and 'c', and one
          with columns 'b' and 'd'. Remaining columns that aren't specified
          will be plotted in additional subplots (one per column).

          .. versionadded:: 1.5.0

    sharex : bool, default True if ax is None else False
        In case ``subplots=True``, share x axis and set some x axis labels
        to invisible; defaults to True if ax is None otherwise False if
        an ax is passed in; Be aware, that passing in both an ax and
        ``sharex=True`` will alter all x axis labels for all axis in a figure.
    sharey : bool, default False
        In case ``subplots=True``, share y axis and set some y axis labels to invisible.
    layout : tuple, optional
        (rows, columns) for the layout of subplots.
    figsize : a tuple (width, height) in inches
        Size of a figure object.
    use_index : bool, default True
        Use index as ticks for x axis.
    title : str or list
        Title to use for the plot. If a string is passed, print the string
        at the top of the figure. If a list is passed and `subplots` is
        True, print each item in the list above the corresponding subplot.
    grid : bool, default None (matlab style default)
        Axis grid lines.
    legend : bool or {'reverse'}
        Place legend on axis subplots.
    style : list or dict
        The matplotlib line style per column.
    logx : bool or 'sym', default False
        Use log scaling or symlog scaling on x axis.

    logy : bool or 'sym' default False
        Use log scaling or symlog scaling on y axis.

    loglog : bool or 'sym', default False
        Use log scaling or symlog scaling on both x and y axes.

    xticks : sequence
        Values to use for the xticks.
    yticks : sequence
        Values to use for the yticks.
    xlim : 2-tuple/list
        Set the x limits of the current axes.
    ylim : 2-tuple/list
        Set the y limits of the current axes.
    xlabel : label, optional
        Name to use for the xlabel on x-axis. Default uses index name as xlabel, or the
        x-column name for planar plots.

        .. versionchanged:: 2.0.0

            Now applicable to histograms.

    ylabel : label, optional
        Name to use for the ylabel on y-axis. Default will show no ylabel, or the
        y-column name for planar plots.

        .. versionchanged:: 2.0.0

            Now applicable to histograms.

    rot : float, default None
        Rotation for ticks (xticks for vertical, yticks for horizontal
        plots).
    fontsize : float, default None
        Font size for xticks and yticks.
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that
        name from matplotlib.
    colorbar : bool, optional
        If True, plot colorbar (only relevant for 'scatter' and 'hexbin'
        plots).
    position : float
        Specify relative alignments for bar plot layout.
        From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
        (center).
    table : bool, Series or DataFrame, default False
        If True, draw a table using the data in the DataFrame and the data
        will be transposed to meet matplotlib's default layout.
        If a Series or DataFrame is passed, use passed data to draw a
        table.
    yerr : DataFrame, Series, array-like, dict and str
        See :ref:`Plotting with Error Bars <visualization.errorbars>` for
        detail.
    xerr : DataFrame, Series, array-like, dict and str
        Equivalent to yerr.
    stacked : bool, default False in line and bar plots, and True in area plot
        If True, create stacked plot.
    secondary_y : bool or sequence, default False
        Whether to plot on the secondary y-axis if a list/tuple, which
        columns to plot on secondary y-axis.
    mark_right : bool, default True
        When using a secondary_y axis, automatically mark the column
        labels with "(right)" in the legend.
    include_bool : bool, default is False
        If True, boolean values can be plotted.
    backend : str, default None
        Backend to use instead of the backend specified in the option
        ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
        specify the ``plotting.backend`` for the whole session, set
        ``pd.options.plotting.backend``.
    **kwargs
        Options to pass to matplotlib plotting method.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        If the backend is not the default matplotlib one, the return value
        will be the object returned by the backend.

    Notes
    -----
    - See matplotlib documentation online for more on this subject
    - If `kind` = 'bar' or 'barh', you can specify relative alignments
      for bar plot layout by `position` keyword.
      From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5
      (center)

    Examples
    --------
    For Series:

    .. plot::
        :context: close-figs

        >>> ser = pd.Series([1, 2, 3, 3])
        >>> plot = ser.plot(kind='hist', title="My plot")

    For DataFrame:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame({'length': [1.5, 0.5, 1.2, 0.9, 3],
        ...                   'width': [0.7, 0.2, 0.15, 0.2, 1.1]},
        ...                   index=['pig', 'rabbit', 'duck', 'chicken', 'horse'])
        >>> plot = df.plot(title="DataFrame Plot")

    For SeriesGroupBy:

    .. plot::
        :context: close-figs

        >>> lst = [-1, -2, -3, 1, 2, 3]
        >>> ser = pd.Series([1, 2, 2, 4, 6, 6], index=lst)
        >>> plot = ser.groupby(lambda x: x > 0).plot(title="SeriesGroupBy Plot")

    For DataFrameGroupBy:

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame({"col1" : [1, 2, 3, 4],
        ...                   "col2" : ["A", "B", "A", "B"]})
        >>> plot = df.groupby("col2").plot(kind="bar", title="DataFrameGroupBy Plot")
    """

    _common_kinds = ("line", "bar", "barh", "kde", "density", "area", "hist", "box")
    _series_kinds = ("pie",)
    _dataframe_kinds = ("scatter", "hexbin")
    _kind_aliases = {"density": "kde"}
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data: Series | DataFrame) -> None:
        self._parent = data

    @staticmethod
    def _get_call_args(backend_name: str, data: Series | DataFrame, args, kwargs):
        """
        This function makes calls to this accessor `__call__` method compatible
        with the previous `SeriesPlotMethods.__call__` and
        `DataFramePlotMethods.__call__`. Those had slightly different
        signatures, since `DataFramePlotMethods` accepted `x` and `y`
        parameters.
        """
        if isinstance(data, ABCSeries):
            arg_def = [
                ("kind", "line"),
                ("ax", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", False),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("label", None),
                ("secondary_y", False),
                ("xlabel", None),
                ("ylabel", None),
            ]
        elif isinstance(data, ABCDataFrame):
            arg_def = [
                ("x", None),
                ("y", None),
                ("kind", "line"),
                ("ax", None),
                ("subplots", False),
                ("sharex", None),
                ("sharey", False),
                ("layout", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", True),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("secondary_y", False),
                ("xlabel", None),
                ("ylabel", None),
            ]
        else:
            raise TypeError(
                f"Called plot accessor for type {type(data).__name__}, "
                "expected Series or DataFrame"
            )

        if args and isinstance(data, ABCSeries):
            positional_args = str(args)[1:-1]
            keyword_args = ", ".join(
                [f"{name}={repr(value)}" for (name, _), value in zip(arg_def, args)]
            )
            msg = (
                "`Series.plot()` should not be called with positional "
                "arguments, only keyword arguments. The order of "
                "positional arguments will change in the future. "
                f"Use `Series.plot({keyword_args})` instead of "
                f"`Series.plot({positional_args})`."
            )
            raise TypeError(msg)

        pos_args = {name: value for (name, _), value in zip(arg_def, args)}
        if backend_name == "pandas.plotting._matplotlib":
            kwargs = dict(arg_def, **pos_args, **kwargs)
        else:
            kwargs = dict(pos_args, **kwargs)

        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        kind = kwargs.pop("kind", "line")
        return x, y, kind, kwargs

    def __call__(self, *args, **kwargs):
        plot_backend = _get_plot_backend(kwargs.pop("backend", None))

        x, y, kind, kwargs = self._get_call_args(
            plot_backend.__name__, self._parent, args, kwargs
        )

        kind = self._kind_aliases.get(kind, kind)

        # when using another backend, get out of the way
        if plot_backend.__name__ != "pandas.plotting._matplotlib":
            return plot_backend.plot(self._parent, x=x, y=y, kind=kind, **kwargs)

        if kind not in self._all_kinds:
            raise ValueError(
                f"{kind} is not a valid plot kind "
                f"Valid plot kinds: {self._all_kinds}"
            )

        # The original data structured can be transformed before passed to the
        # backend. For example, for DataFrame is common to set the index as the
        # `x` parameter, and return a Series with the parameter `y` as values.
        data = self._parent.copy()

        if isinstance(data, ABCSeries):
            kwargs["reuse_plot"] = True

        if kind in self._dataframe_kinds:
            if isinstance(data, ABCDataFrame):
                return plot_backend.plot(data, x=x, y=y, kind=kind, **kwargs)
            else:
                raise ValueError(f"plot kind {kind} can only be used for data frames")
        elif kind in self._series_kinds:
            if isinstance(data, ABCDataFrame):
                if y is None and kwargs.get("subplots") is False:
                    raise ValueError(
                        f"{kind} requires either y column or 'subplots=True'"
                    )
                if y is not None:
                    if is_integer(y) and not data.columns._holds_integer():
                        y = data.columns[y]
                    # converted to series actually. copy to not modify
                    data = data[y].copy()
                    data.index.name = y
        elif isinstance(data, ABCDataFrame):
            data_cols = data.columns
            if x is not None:
                if is_integer(x) and not data.columns._holds_integer():
                    x = data_cols[x]
                elif not isinstance(data[x], ABCSeries):
                    raise ValueError("x must be a label or position")
                data = data.set_index(x)
            if y is not None:
                # check if we have y as int or list of ints
                int_ylist = is_list_like(y) and all(is_integer(c) for c in y)
                int_y_arg = is_integer(y) or int_ylist
                if int_y_arg and not data.columns._holds_integer():
                    y = data_cols[y]

                label_kw = kwargs["label"] if "label" in kwargs else False
                for kw in ["xerr", "yerr"]:
                    if kw in kwargs and (
                        isinstance(kwargs[kw], str) or is_integer(kwargs[kw])
                    ):
                        try:
                            kwargs[kw] = data[kwargs[kw]]
                        except (IndexError, KeyError, TypeError):
                            pass

                # don't overwrite
                data = data[y].copy()

                if isinstance(data, ABCSeries):
                    label_name = label_kw or y
                    data.name = label_name
                else:
                    match = is_list_like(label_kw) and len(label_kw) == len(y)
                    if label_kw and not match:
                        raise ValueError(
                            "label should be list-like and same length as y"
                        )
                    label_name = label_kw or data.columns
                    data.columns = label_name

        return plot_backend.plot(data, kind=kind, **kwargs)

    __call__.__doc__ = __doc__

    @Appender(
        """
        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Examples
        --------

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 3, 2])
            >>> s.plot.line()  # doctest: +SKIP

        .. plot::
            :context: close-figs

            The following example shows the populations for some animals
            over the years.

            >>> df = pd.DataFrame({
            ...    'pig': [20, 18, 489, 675, 1776],
            ...    'horse': [4, 25, 281, 600, 1900]
            ...    }, index=[1990, 1997, 2003, 2009, 2014])
            >>> lines = df.plot.line()

        .. plot::
           :context: close-figs

           An example with subplots, so an array of axes is returned.

           >>> axes = df.plot.line(subplots=True)
           >>> type(axes)
           <class 'numpy.ndarray'>

        .. plot::
           :context: close-figs

           Let's repeat the same example, but specifying colors for
           each column (in this case, for each animal).

           >>> axes = df.plot.line(
           ...     subplots=True, color={"pig": "pink", "horse": "#742802"}
           ... )

        .. plot::
            :context: close-figs

            The following example shows the relationship between both
            populations.

            >>> lines = df.plot.line(x='pig', y='horse')
        """
    )
    @Substitution(kind="line")
    @Appender(_bar_or_line_doc)
    def line(
        self, x: Hashable | None = None, y: Hashable | None = None, **kwargs
    ) -> PlotAccessor:
        """
        Plot Series or DataFrame as lines.

        This function is useful to plot lines using DataFrame's values
        as coordinates.
        """
        return self(kind="line", x=x, y=y, **kwargs)

    @Appender(
        """
        See Also
        --------
        DataFrame.plot.barh : Horizontal bar plot.
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.bar : Make a bar plot with matplotlib.

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
            >>> ax = df.plot.bar(x='lab', y='val', rot=0)

        Plot a whole dataframe to a bar plot. Each column is assigned a
        distinct color, and each row is nested in a group along the
        horizontal axis.

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.bar(rot=0)

        Plot stacked bar charts for the DataFrame

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(stacked=True)

        Instead of nesting, the figure can be split by column with
        ``subplots=True``. In this case, a :class:`numpy.ndarray` of
        :class:`matplotlib.axes.Axes` are returned.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(rot=0, subplots=True)
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        If you don't like the default colours, you can specify how you'd
        like each column to be colored.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(
            ...     rot=0, subplots=True, color={"speed": "red", "lifespan": "green"}
            ... )
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        Plot a single column.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(y='speed', rot=0)

        Plot only selected categories for the DataFrame.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(x='lifespan', rot=0)
    """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    def bar(  # pylint: disable=disallowed-name
        self, x: Hashable | None = None, y: Hashable | None = None, **kwargs
    ) -> PlotAccessor:
        """
        Vertical bar plot.

        A bar plot is a plot that presents categorical data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
        """
        return self(kind="bar", x=x, y=y, **kwargs)

    @Appender(
        """
        See Also
        --------
        DataFrame.plot.bar: Vertical bar plot.
        DataFrame.plot : Make plots of DataFrame using matplotlib.
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.

        Examples
        --------
        Basic example

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.barh(x='lab', y='val')

        Plot a whole DataFrame to a horizontal bar plot

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh()

        Plot stacked barh charts for the DataFrame

        .. plot::
            :context: close-figs

            >>> ax = df.plot.barh(stacked=True)

        We can specify colors for each column

        .. plot::
            :context: close-figs

            >>> ax = df.plot.barh(color={"speed": "red", "lifespan": "green"})

        Plot a column of the DataFrame to a horizontal bar plot

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(y='speed')

        Plot DataFrame versus the desired column

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(x='lifespan')
    """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    def barh(
        self, x: Hashable | None = None, y: Hashable | None = None, **kwargs
    ) -> PlotAccessor:
        """
        Make a horizontal bar plot.

        A horizontal bar plot is a plot that presents quantitative data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.
        """
        return self(kind="barh", x=x, y=y, **kwargs)

    def box(self, by: IndexLabel | None = None, **kwargs) -> PlotAccessor:
        r"""
        Make a box plot of the DataFrame columns.

        A box plot is a method for graphically depicting groups of numerical
        data through their quartiles.
        The box extends from the Q1 to Q3 quartile values of the data,
        with a line at the median (Q2). The whiskers extend from the edges
        of box to show the range of the data. The position of the whiskers
        is set by default to 1.5*IQR (IQR = Q3 - Q1) from the edges of the
        box. Outlier points are those past the end of the whiskers.

        For further details see Wikipedia's
        entry for `boxplot <https://en.wikipedia.org/wiki/Box_plot>`__.

        A consideration when using this chart is that the box and the whiskers
        can overlap, which is very common when plotting small sets of data.

        Parameters
        ----------
        by : str or sequence
            Column in the DataFrame to group by.

            .. versionchanged:: 1.4.0

               Previously, `by` is silently ignore and makes no groupings

        **kwargs
            Additional keywords are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        See Also
        --------
        DataFrame.boxplot: Another method to draw a box plot.
        Series.plot.box: Draw a box plot from a Series object.
        matplotlib.pyplot.boxplot: Draw a box plot in matplotlib.

        Examples
        --------
        Draw a box plot from a DataFrame with four columns of randomly
        generated data.

        .. plot::
            :context: close-figs

            >>> data = np.random.randn(25, 4)
            >>> df = pd.DataFrame(data, columns=list('ABCD'))
            >>> ax = df.plot.box()

        You can also generate groupings if you specify the `by` parameter (which
        can take a column name, or a list or tuple of column names):

        .. versionchanged:: 1.4.0

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.box(column="age", by="gender", figsize=(10, 8))
        """
        return self(kind="box", by=by, **kwargs)

    def hist(
        self, by: IndexLabel | None = None, bins: int = 10, **kwargs
    ) -> PlotAccessor:
        """
        Draw one histogram of the DataFrame's columns.

        A histogram is a representation of the distribution of data.
        This function groups the values of all given Series in the DataFrame
        into bins and draws all bins in one :class:`matplotlib.axes.Axes`.
        This is useful when the DataFrame's Series are in a similar scale.

        Parameters
        ----------
        by : str or sequence, optional
            Column in the DataFrame to group by.

            .. versionchanged:: 1.4.0

               Previously, `by` is silently ignore and makes no groupings

        bins : int, default 10
            Number of histogram bins to be used.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        class:`matplotlib.AxesSubplot`
            Return a histogram plot.

        See Also
        --------
        DataFrame.hist : Draw histograms per DataFrame's Series.
        Series.hist : Draw a histogram with Series' data.

        Examples
        --------
        When we roll a die 6000 times, we expect to get each value around 1000
        times. But when we roll two dice and sum the result, the distribution
        is going to be quite different. A histogram illustrates those
        distributions.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(np.random.randint(1, 7, 6000), columns=['one'])
            >>> df['two'] = df['one'] + np.random.randint(1, 7, 6000)
            >>> ax = df.plot.hist(bins=12, alpha=0.5)

        A grouped histogram can be generated by providing the parameter `by` (which
        can be a column name, or a list of column names):

        .. plot::
            :context: close-figs

            >>> age_list = [8, 10, 12, 14, 72, 74, 76, 78, 20, 25, 30, 35, 60, 85]
            >>> df = pd.DataFrame({"gender": list("MMMMMMMMFFFFFF"), "age": age_list})
            >>> ax = df.plot.hist(column=["age"], by="gender", figsize=(10, 8))
        """
        return self(kind="hist", by=by, bins=bins, **kwargs)

    def kde(
        self,
        bw_method: Literal["scott", "silverman"] | float | Callable | None = None,
        ind: np.ndarray | int | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Generate Kernel Density Estimate plot using Gaussian kernels.

        In statistics, `kernel density estimation`_ (KDE) is a non-parametric
        way to estimate the probability density function (PDF) of a random
        variable. This function uses Gaussian kernels and includes automatic
        bandwidth determination.

        .. _kernel density estimation:
            https://en.wikipedia.org/wiki/Kernel_density_estimation

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth. This can be
            'scott', 'silverman', a scalar constant or a callable.
            If None (default), 'scott' is used.
            See :class:`scipy.stats.gaussian_kde` for more information.
        ind : NumPy array or int, optional
            Evaluation points for the estimated PDF. If None (default),
            1000 equally spaced points are used. If `ind` is a NumPy array, the
            KDE is evaluated at the points passed. If `ind` is an integer,
            `ind` number of equally spaced points are used.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        See Also
        --------
        scipy.stats.gaussian_kde : Representation of a kernel-density
            estimate using Gaussian kernels. This is the function used
            internally to estimate the PDF.

        Examples
        --------
        Given a Series of points randomly sampled from an unknown
        distribution, estimate its PDF using KDE with automatic
        bandwidth determination and plot the results, evaluating them at
        1000 equally spaced points (default):

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 2, 2.5, 3, 3.5, 4, 5])
            >>> ax = s.plot.kde()

        A scalar bandwidth can be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(bw_method=3)

        Finally, the `ind` parameter determines the evaluation points for the
        plot of the estimated PDF:

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(ind=[1, 2, 3, 4, 5])

        For DataFrame, it works in the same way:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({
            ...     'x': [1, 2, 2.5, 3, 3.5, 4, 5],
            ...     'y': [4, 4, 4.5, 5, 5.5, 6, 6],
            ... })
            >>> ax = df.plot.kde()

        A scalar bandwidth can be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(bw_method=3)

        Finally, the `ind` parameter determines the evaluation points for the
        plot of the estimated PDF:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(ind=[1, 2, 3, 4, 5, 6])
        """
        return self(kind="kde", bw_method=bw_method, ind=ind, **kwargs)

    density = kde

    def area(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        stacked: bool = True,
        **kwargs,
    ) -> PlotAccessor:
        """
        Draw a stacked area plot.

        An area plot displays quantitative data visually.
        This function wraps the matplotlib area function.

        Parameters
        ----------
        x : label or position, optional
            Coordinates for the X axis. By default uses the index.
        y : label or position, optional
            Column to plot. By default uses all columns.
        stacked : bool, default True
            Area plots are stacked by default. Set to False to create a
            unstacked plot.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Area plot, or array of area plots if subplots is True.

        See Also
        --------
        DataFrame.plot : Make plots of DataFrame using matplotlib / pylab.

        Examples
        --------
        Draw an area plot based on basic business metrics:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({
            ...     'sales': [3, 2, 3, 9, 10, 6],
            ...     'signups': [5, 5, 6, 12, 14, 13],
            ...     'visits': [20, 42, 28, 62, 81, 50],
            ... }, index=pd.date_range(start='2018/01/01', end='2018/07/01',
            ...                        freq='ME'))
            >>> ax = df.plot.area()

        Area plots are stacked by default. To produce an unstacked plot,
        pass ``stacked=False``:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(stacked=False)

        Draw an area plot for a single column:

        .. plot::
            :context: close-figs

            >>> ax = df.plot.area(y='sales')

        Draw with a different `x`:

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({
            ...     'sales': [3, 2, 3],
            ...     'visits': [20, 42, 28],
            ...     'day': [1, 2, 3],
            ... })
            >>> ax = df.plot.area(x='day')
        """
        return self(kind="area", x=x, y=y, stacked=stacked, **kwargs)

    def pie(self, **kwargs) -> PlotAccessor:
        """
        Generate a pie plot.

        A pie plot is a proportional representation of the numerical data in a
        column. This function wraps :meth:`matplotlib.pyplot.pie` for the
        specified column. If no column reference is passed and
        ``subplots=True`` a pie plot is drawn for each numerical column
        independently.

        Parameters
        ----------
        y : int or label, optional
            Label or position of the column to plot.
            If not provided, ``subplots=True`` argument must be passed.
        **kwargs
            Keyword arguments to pass on to :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            A NumPy array is returned when `subplots` is True.

        See Also
        --------
        Series.plot.pie : Generate a pie plot for a Series.
        DataFrame.plot : Make plots of a DataFrame.

        Examples
        --------
        In the example below we have a DataFrame with the information about
        planet's mass and radius. We pass the 'mass' column to the
        pie function to get a pie plot.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame({'mass': [0.330, 4.87 , 5.97],
            ...                    'radius': [2439.7, 6051.8, 6378.1]},
            ...                   index=['Mercury', 'Venus', 'Earth'])
            >>> plot = df.plot.pie(y='mass', figsize=(5, 5))

        .. plot::
            :context: close-figs

            >>> plot = df.plot.pie(subplots=True, figsize=(11, 6))
        """
        if (
            isinstance(self._parent, ABCDataFrame)
            and kwargs.get("y", None) is None
            and not kwargs.get("subplots", False)
        ):
            raise ValueError("pie requires either y column or 'subplots=True'")
        return self(kind="pie", **kwargs)

    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[Hashable] | None = None,
        c: Hashable | Sequence[Hashable] | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Create a scatter plot with varying marker point size and color.

        The coordinates of each point are defined by two dataframe columns and
        filled circles are used to represent each point. This kind of plot is
        useful to see complex correlations between two variables. Points could
        be for instance natural 2D coordinates like longitude and latitude in
        a map or, in general, any pair of metrics that can be plotted against
        each other.

        Parameters
        ----------
        x : int or str
            The column name or column position to be used as horizontal
            coordinates for each point.
        y : int or str
            The column name or column position to be used as vertical
            coordinates for each point.
        s : str, scalar or array-like, optional
            The size of each point. Possible values are:

            - A string with the name of the column to be used for marker's size.

            - A single scalar so all points have the same size.

            - A sequence of scalars, which will be used for each point's size
              recursively. For instance, when passing [2,14] all points size
              will be either 2 or 14, alternatively.

        c : str, int or array-like, optional
            The color of each point. Possible values are:

            - A single color string referred to by name, RGB or RGBA code,
              for instance 'red' or '#a98d19'.

            - A sequence of color strings referred to by name, RGB or RGBA
              code, which will be used for each point's color recursively. For
              instance ['green','yellow'] all points will be filled in green or
              yellow, alternatively.

            - A column name or position whose values will be used to color the
              marker points according to a colormap.

        **kwargs
            Keyword arguments to pass on to :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        See Also
        --------
        matplotlib.pyplot.scatter : Scatter plot using multiple input data
            formats.

        Examples
        --------
        Let's see how to draw a scatter plot using coordinates from the values
        in a DataFrame's columns.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
            ...                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
            ...                   columns=['length', 'width', 'species'])
            >>> ax1 = df.plot.scatter(x='length',
            ...                       y='width',
            ...                       c='DarkBlue')

        And now with the color determined by a column as well.

        .. plot::
            :context: close-figs

            >>> ax2 = df.plot.scatter(x='length',
            ...                       y='width',
            ...                       c='species',
            ...                       colormap='viridis')
        """
        return self(kind="scatter", x=x, y=y, s=s, c=c, **kwargs)

    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = None,
        reduce_C_function: Callable | None = None,
        gridsize: int | tuple[int, int] | None = None,
        **kwargs,
    ) -> PlotAccessor:
        """
        Generate a hexagonal binning plot.

        Generate a hexagonal binning plot of `x` versus `y`. If `C` is `None`
        (the default), this is a histogram of the number of occurrences
        of the observations at ``(x[i], y[i])``.

        If `C` is specified, specifies values at given coordinates
        ``(x[i], y[i])``. These values are accumulated for each hexagonal
        bin and then reduced according to `reduce_C_function`,
        having as default the NumPy's mean function (:meth:`numpy.mean`).
        (If `C` is specified, it must also be a 1-D sequence
        of the same length as `x` and `y`, or a column label.)

        Parameters
        ----------
        x : int or str
            The column label or position for x points.
        y : int or str
            The column label or position for y points.
        C : int or str, optional
            The column label or position for the value of `(x, y)` point.
        reduce_C_function : callable, default `np.mean`
            Function of one argument that reduces all the values in a bin to
            a single number (e.g. `np.mean`, `np.max`, `np.sum`, `np.std`).
        gridsize : int or tuple of (int, int), default 100
            The number of hexagons in the x-direction.
            The corresponding number of hexagons in the y-direction is
            chosen in a way that the hexagons are approximately regular.
            Alternatively, gridsize can be a tuple with two elements
            specifying the number of hexagons in the x-direction and the
            y-direction.
        **kwargs
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.AxesSubplot
            The matplotlib ``Axes`` on which the hexbin is plotted.

        See Also
        --------
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.hexbin : Hexagonal binning plot using matplotlib,
            the matplotlib function that is used under the hood.

        Examples
        --------
        The following examples are generated with random data from
        a normal distribution.

        .. plot::
            :context: close-figs

            >>> n = 10000
            >>> df = pd.DataFrame({'x': np.random.randn(n),
            ...                    'y': np.random.randn(n)})
            >>> ax = df.plot.hexbin(x='x', y='y', gridsize=20)

        The next example uses `C` and `np.sum` as `reduce_C_function`.
        Note that `'observations'` values ranges from 1 to 5 but the result
        plot shows values up to more than 25. This is because of the
        `reduce_C_function`.

        .. plot::
            :context: close-figs

            >>> n = 500
            >>> df = pd.DataFrame({
            ...     'coord_x': np.random.uniform(-3, 3, size=n),
            ...     'coord_y': np.random.uniform(30, 50, size=n),
            ...     'observations': np.random.randint(1,5, size=n)
            ...     })
            >>> ax = df.plot.hexbin(x='coord_x',
            ...                     y='coord_y',
            ...                     C='observations',
            ...                     reduce_C_function=np.sum,
            ...                     gridsize=10,
            ...                     cmap="viridis")
        """
        if reduce_C_function is not None:
            kwargs["reduce_C_function"] = reduce_C_function
        if gridsize is not None:
            kwargs["gridsize"] = gridsize

        return self(kind="hexbin", x=x, y=y, C=C, **kwargs)


_backends: dict[str, types.ModuleType] = {}


def _load_backend(backend: str) -> types.ModuleType:
    """
    Load a pandas plotting backend.

    Parameters
    ----------
    backend : str
        The identifier for the backend. Either an entrypoint item registered
        with importlib.metadata, "matplotlib", or a module name.

    Returns
    -------
    types.ModuleType
        The imported backend.
    """
    from importlib.metadata import entry_points

    if backend == "matplotlib":
        # Because matplotlib is an optional dependency and first-party backend,
        # we need to attempt an import here to raise an ImportError if needed.
        try:
            module = importlib.import_module("pandas.plotting._matplotlib")
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting when the "
                'default backend "matplotlib" is selected.'
            ) from None
        return module

    found_backend = False

    eps = entry_points()
    key = "pandas_plotting_backends"
    # entry_points lost dict API ~ PY 3.10
    # https://github.com/python/importlib_metadata/issues/298
    if hasattr(eps, "select"):
        entry = eps.select(group=key)
    else:
        # Argument 2 to "get" of "dict" has incompatible type "Tuple[]";
        # expected "EntryPoints"  [arg-type]
        entry = eps.get(key, ())  # type: ignore[arg-type]
    for entry_point in entry:
        found_backend = entry_point.name == backend
        if found_backend:
            module = entry_point.load()
            break

    if not found_backend:
        # Fall back to unregistered, module name approach.
        try:
            module = importlib.import_module(backend)
            found_backend = True
        except ImportError:
            # We re-raise later on.
            pass

    if found_backend:
        if hasattr(module, "plot"):
            # Validate that the interface is implemented when the option is set,
            # rather than at plot time.
            return module

    raise ValueError(
        f"Could not find plotting backend '{backend}'. Ensure that you've "
        f"installed the package providing the '{backend}' entrypoint, or that "
        "the package has a top-level `.plot` method."
    )


def _get_plot_backend(backend: str | None = None):
    """
    Return the plotting backend to use (e.g. `pandas.plotting._matplotlib`).

    The plotting system of pandas uses matplotlib by default, but the idea here
    is that it can also work with other third-party backends. This function
    returns the module which provides a top-level `.plot` method that will
    actually do the plotting. The backend is specified from a string, which
    either comes from the keyword argument `backend`, or, if not specified, from
    the option `pandas.options.plotting.backend`. All the rest of the code in
    this file uses the backend specified there for the plotting.

    The backend is imported lazily, as matplotlib is a soft dependency, and
    pandas can be used without it being installed.

    Notes
    -----
    Modifies `_backends` with imported backend as a side effect.
    """
    backend_str: str = backend or get_option("plotting.backend")

    if backend_str in _backends:
        return _backends[backend_str]

    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\sparse\array.py ===
"""
SparseArray data structure
"""
from __future__ import annotations

from collections import abc
import numbers
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
    BlockIndex,
    IntIndex,
    SparseIndex,
)
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_insert_loc,
)

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    find_common_type,
    maybe_box_datetimelike,
)
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
)
from pandas.core.nanops import check_below_min_count

from pandas.io.formats import printing

# See https://github.com/python/typing/issues/684
if TYPE_CHECKING:
    from collections.abc import Sequence
    from enum import Enum

    class ellipsis(Enum):
        Ellipsis = "..."

    Ellipsis = ellipsis.Ellipsis

    from scipy.sparse import spmatrix

    from pandas._typing import (
        FillnaOptions,
        NumpySorter,
    )

    SparseIndexKind = Literal["integer", "block"]

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        Axis,
        AxisInt,
        Dtype,
        NpDtype,
        PositionalIndexer,
        Scalar,
        ScalarIndexer,
        Self,
        SequenceIndexer,
        npt,
    )

    from pandas import Series

else:
    ellipsis = type(Ellipsis)


# ----------------------------------------------------------------------------
# Array

_sparray_doc_kwargs = {"klass": "SparseArray"}


def _get_fill(arr: SparseArray) -> np.ndarray:
    """
    Create a 0-dim ndarray containing the fill value

    Parameters
    ----------
    arr : SparseArray

    Returns
    -------
    fill_value : ndarray
        0-dim ndarray with just the fill value.

    Notes
    -----
    coerce fill_value to arr dtype if possible
    int64 SparseArray can have NaN as fill_value if there is no missing
    """
    try:
        return np.asarray(arr.fill_value, dtype=arr.dtype.subtype)
    except ValueError:
        return np.asarray(arr.fill_value)


def _sparse_array_op(
    left: SparseArray, right: SparseArray, op: Callable, name: str
) -> SparseArray:
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : Union[SparseArray, ndarray]
    right : Union[SparseArray, ndarray]
    op : Callable
        The binary operation to perform
    name str
        Name of the callable.

    Returns
    -------
    SparseArray
    """
    if name.startswith("__"):
        # For lookups in _libs.sparse we need non-dunder op name
        name = name[2:-2]

    # dtype used to find corresponding sparse method
    ltype = left.dtype.subtype
    rtype = right.dtype.subtype

    if ltype != rtype:
        subtype = find_common_type([ltype, rtype])
        ltype = SparseDtype(subtype, left.fill_value)
        rtype = SparseDtype(subtype, right.fill_value)

        left = left.astype(ltype, copy=False)
        right = right.astype(rtype, copy=False)
        dtype = ltype.subtype
    else:
        dtype = ltype

    # dtype the result must have
    result_dtype = None

    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        with np.errstate(all="ignore"):
            result = op(left.to_dense(), right.to_dense())
            fill = op(_get_fill(left), _get_fill(right))

        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        with np.errstate(all="ignore"):
            result = op(left.sp_values, right.sp_values)
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        if name[0] == "r":
            left, right = right, left
            name = name[1:]

        if name in ("and", "or", "xor") and dtype == "bool":
            opname = f"sparse_{name}_uint8"
            # to make template simple, cast here
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            opname = f"sparse_{name}_{dtype}"
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values

        if (
            name in ["floordiv", "mod"]
            and (right == 0).any()
            and left.dtype.kind in "iu"
        ):
            # Match the non-Sparse Series behavior
            opname = f"sparse_{name}_float64"
            left_sp_values = left_sp_values.astype("float64")
            right_sp_values = right_sp_values.astype("float64")

        sparse_op = getattr(splib, opname)

        with np.errstate(all="ignore"):
            result, index, fill = sparse_op(
                left_sp_values,
                left.sp_index,
                left.fill_value,
                right_sp_values,
                right.sp_index,
                right.fill_value,
            )

    if name == "divmod":
        # result is a 2-tuple
        # error: Incompatible return value type (got "Tuple[SparseArray,
        # SparseArray]", expected "SparseArray")
        return (  # type: ignore[return-value]
            _wrap_result(name, result[0], index, fill[0], dtype=result_dtype),
            _wrap_result(name, result[1], index, fill[1], dtype=result_dtype),
        )

    if result_dtype is None:
        result_dtype = result.dtype

    return _wrap_result(name, result, index, fill, dtype=result_dtype)


def _wrap_result(
    name: str, data, sparse_index, fill_value, dtype: Dtype | None = None
) -> SparseArray:
    """
    wrap op result to have correct dtype
    """
    if name.startswith("__"):
        # e.g. __eq__ --> eq
        name = name[2:-2]

    if name in ("eq", "ne", "lt", "gt", "le", "ge"):
        dtype = bool

    fill_value = lib.item_from_zerodim(fill_value)

    if is_bool_dtype(dtype):
        # fill_value may be np.bool_
        fill_value = bool(fill_value)
    return SparseArray(
        data, sparse_index=sparse_index, fill_value=fill_value, dtype=dtype
    )


class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    """
    An ExtensionArray for storing sparse data.

    Parameters
    ----------
    data : array-like or scalar
        A dense array of values to store in the SparseArray. This may contain
        `fill_value`.
    sparse_index : SparseIndex, optional
    fill_value : scalar, optional
        Elements in data that are ``fill_value`` are not stored in the
        SparseArray. For memory savings, this should be the most common value
        in `data`. By default, `fill_value` depends on the dtype of `data`:

        =========== ==========
        data.dtype  na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        False
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The fill value is potentially specified in three ways. In order of
        precedence, these are

        1. The `fill_value` argument
        2. ``dtype.fill_value`` if `fill_value` is None and `dtype` is
           a ``SparseDtype``
        3. ``data.dtype.fill_value`` if `fill_value` is None and `dtype`
           is not a ``SparseDtype`` and `data` is a ``SparseArray``.

    kind : str
        Can be 'integer' or 'block', default is 'integer'.
        The type of storage for sparse locations.

        * 'block': Stores a `block` and `block_length` for each
          contiguous *span* of sparse values. This is best when
          sparse data tends to be clumped together, with large
          regions of ``fill-value`` values between sparse values.
        * 'integer': uses an integer to store the location of
          each sparse value.

    dtype : np.dtype or SparseDtype, optional
        The dtype to use for the SparseArray. For numpy dtypes, this
        determines the dtype of ``self.sp_values``. For SparseDtype,
        this determines ``self.sp_values`` and ``self.fill_value``.
    copy : bool, default False
        Whether to explicitly copy the incoming `data` array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> arr = SparseArray([0, 0, 1, 2])
    >>> arr
    [0, 0, 1, 2]
    Fill: 0
    IntIndex
    Indices: array([2, 3], dtype=int32)
    """

    _subtyp = "sparse_array"  # register ABCSparseArray
    _hidden_attrs = PandasObject._hidden_attrs | frozenset([])
    _sparse_index: SparseIndex
    _sparse_values: np.ndarray
    _dtype: SparseDtype

    def __init__(
        self,
        data,
        sparse_index=None,
        fill_value=None,
        kind: SparseIndexKind = "integer",
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> None:
        if fill_value is None and isinstance(dtype, SparseDtype):
            fill_value = dtype.fill_value

        if isinstance(data, type(self)):
            # disable normal inference on dtype, sparse_index, & fill_value
            if sparse_index is None:
                sparse_index = data.sp_index
            if fill_value is None:
                fill_value = data.fill_value
            if dtype is None:
                dtype = data.dtype
            # TODO: make kind=None, and use data.kind?
            data = data.sp_values

        # Handle use-provided dtype
        if isinstance(dtype, str):
            # Two options: dtype='int', regular numpy dtype
            # or dtype='Sparse[int]', a sparse dtype
            try:
                dtype = SparseDtype.construct_from_string(dtype)
            except TypeError:
                dtype = pandas_dtype(dtype)

        if isinstance(dtype, SparseDtype):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        if is_scalar(data):
            warnings.warn(
                f"Constructing {type(self).__name__} with scalar data is deprecated "
                "and will raise in a future version. Pass a sequence instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            if sparse_index is None:
                npoints = 1
            else:
                npoints = sparse_index.length

            data = construct_1d_arraylike_from_scalar(data, npoints, dtype=None)
            dtype = data.dtype

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        # TODO: disentangle the fill_value dtype inference from
        # dtype inference
        if data is None:
            # TODO: What should the empty dtype be? Object or float?

            # error: Argument "dtype" to "array" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "Union[dtype[Any],
            # None, type, _SupportsDType, str, Union[Tuple[Any, int], Tuple[Any,
            # Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]]"
            data = np.array([], dtype=dtype)  # type: ignore[arg-type]

        try:
            data = sanitize_array(data, index=None)
        except ValueError:
            # NumPy may raise a ValueError on data like [1, []]
            # we retry with object dtype here.
            if dtype is None:
                dtype = np.dtype(object)
                data = np.atleast_1d(np.asarray(data, dtype=dtype))
            else:
                raise

        if copy:
            # TODO: avoid double copy when dtype forces cast.
            data = data.copy()

        if fill_value is None:
            fill_value_dtype = data.dtype if dtype is None else dtype
            if fill_value_dtype is None:
                fill_value = np.nan
            else:
                fill_value = na_value_for_dtype(fill_value_dtype)

        if isinstance(data, type(self)) and sparse_index is None:
            sparse_index = data._sparse_index
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "None"
            sparse_values = np.asarray(
                data.sp_values, dtype=dtype  # type: ignore[arg-type]
            )
        elif sparse_index is None:
            data = extract_array(data, extract_numpy=True)
            if not isinstance(data, np.ndarray):
                # EA
                if isinstance(data.dtype, DatetimeTZDtype):
                    warnings.warn(
                        f"Creating SparseArray from {data.dtype} data "
                        "loses timezone information. Cast to object before "
                        "sparse to retain timezone information.",
                        UserWarning,
                        stacklevel=find_stack_level(),
                    )
                    data = np.asarray(data, dtype="datetime64[ns]")
                    if fill_value is NaT:
                        fill_value = np.datetime64("NaT", "ns")
                data = np.asarray(data)
            sparse_values, sparse_index, fill_value = _make_sparse(
                # error: Argument "dtype" to "_make_sparse" has incompatible type
                # "Union[ExtensionDtype, dtype[Any], None]"; expected
                # "Optional[dtype[Any]]"
                data,
                kind=kind,
                fill_value=fill_value,
                dtype=dtype,  # type: ignore[arg-type]
            )
        else:
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "None"
            sparse_values = np.asarray(data, dtype=dtype)  # type: ignore[arg-type]
            if len(sparse_values) != sparse_index.npoints:
                raise AssertionError(
                    f"Non array-like type {type(sparse_values)} must "
                    "have the same length as the index"
                )
        self._sparse_index = sparse_index
        self._sparse_values = sparse_values
        self._dtype = SparseDtype(sparse_values.dtype, fill_value)

    @classmethod
    def _simple_new(
        cls,
        sparse_array: np.ndarray,
        sparse_index: SparseIndex,
        dtype: SparseDtype,
    ) -> Self:
        new = object.__new__(cls)
        new._sparse_index = sparse_index
        new._sparse_values = sparse_array
        new._dtype = dtype
        return new

    @classmethod
    def from_spmatrix(cls, data: spmatrix) -> Self:
        """
        Create a SparseArray from a scipy.sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.sp_matrix
            This should be a SciPy sparse matrix where the size
            of the second dimension is 1. In other words, a
            sparse matrix with a single column.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.coo_matrix((4, 1))
        >>> pd.arrays.SparseArray.from_spmatrix(mat)
        [0.0, 0.0, 0.0, 0.0]
        Fill: 0.0
        IntIndex
        Indices: array([], dtype=int32)
        """
        length, ncol = data.shape

        if ncol != 1:
            raise ValueError(f"'data' must have a single column, not '{ncol}'")

        # our sparse index classes require that the positions be strictly
        # increasing. So we need to sort loc, and arr accordingly.
        data = data.tocsc()
        data.sort_indices()
        arr = data.data
        idx = data.indices

        zero = np.array(0, dtype=arr.dtype).item()
        dtype = SparseDtype(arr.dtype, zero)
        index = IntIndex(length, idx)

        return cls._simple_new(arr, index, dtype)

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if self.sp_index.ngaps == 0:
            # Compat for na dtype and int values.
            if copy is True:
                return np.array(self.sp_values)
            else:
                return self.sp_values

        if copy is False:
            warnings.warn(
                "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                "changed and passing 'copy=False' raises an error when returning "
                "a zero-copy NumPy array is not possible. pandas will follow "
                "this behavior starting with pandas 3.0.\nThis conversion to "
                "NumPy requires a copy, but 'copy=False' was passed. Consider "
                "using 'np.asarray(..)' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        fill_value = self.fill_value

        if dtype is None:
            # Can NumPy represent this type?
            # If not, `np.result_type` will raise. We catch that
            # and return object.
            if self.sp_values.dtype.kind == "M":
                # However, we *do* special-case the common case of
                # a datetime64 with pandas NaT.
                if fill_value is NaT:
                    # Can't put pd.NaT in a datetime64[ns]
                    fill_value = np.datetime64("NaT")
            try:
                dtype = np.result_type(self.sp_values.dtype, type(fill_value))
            except TypeError:
                dtype = object

        out = np.full(self.shape, fill_value, dtype=dtype)
        out[self.sp_index.indices] = self.sp_values
        return out

    def __setitem__(self, key, value) -> None:
        # I suppose we could allow setting of non-fill_value elements.
        # TODO(SparseArray.__setitem__): remove special cases in
        # ExtensionBlock.where
        msg = "SparseArray does not support item assignment via setitem"
        raise TypeError(msg)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    @property
    def sp_index(self) -> SparseIndex:
        """
        The SparseIndex containing the location of non- ``fill_value`` points.
        """
        return self._sparse_index

    @property
    def sp_values(self) -> np.ndarray:
        """
        An ndarray containing the non- ``fill_value`` values.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 0, 2], fill_value=0)
        >>> s.sp_values
        array([1, 2])
        """
        return self._sparse_values

    @property
    def dtype(self) -> SparseDtype:
        return self._dtype

    @property
    def fill_value(self):
        """
        Elements in `data` that are `fill_value` are not stored.

        For memory savings, this should be the most common value in the array.

        Examples
        --------
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
        >>> ser.sparse.fill_value
        0
        >>> spa_dtype = pd.SparseDtype(dtype=np.int32, fill_value=2)
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype=spa_dtype)
        >>> ser.sparse.fill_value
        2
        """
        return self.dtype.fill_value

    @fill_value.setter
    def fill_value(self, value) -> None:
        self._dtype = SparseDtype(self.dtype.subtype, value)

    @property
    def kind(self) -> SparseIndexKind:
        """
        The kind of sparse index for this array. One of {'integer', 'block'}.
        """
        if isinstance(self.sp_index, IntIndex):
            return "integer"
        else:
            return "block"

    @property
    def _valid_sp_values(self) -> np.ndarray:
        sp_vals = self.sp_values
        mask = notna(sp_vals)
        return sp_vals[mask]

    def __len__(self) -> int:
        return self.sp_index.length

    @property
    def _null_fill_value(self) -> bool:
        return self._dtype._is_na_fill_value

    def _fill_value_matches(self, fill_value) -> bool:
        if self._null_fill_value:
            return isna(fill_value)
        else:
            return self.fill_value == fill_value

    @property
    def nbytes(self) -> int:
        return self.sp_values.nbytes + self.sp_index.nbytes

    @property
    def density(self) -> float:
        """
        The percent of non- ``fill_value`` points, as decimal.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.density
        0.6
        """
        return self.sp_index.npoints / self.sp_index.length

    @property
    def npoints(self) -> int:
        """
        The number of non- ``fill_value`` points.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.npoints
        3
        """
        return self.sp_index.npoints

    # error: Return type "SparseArray" of "isna" incompatible with return type
    # "ndarray[Any, Any] | ExtensionArraySupportsAnyAll" in supertype "ExtensionArray"
    def isna(self) -> Self:  # type: ignore[override]
        # If null fill value, we want SparseDtype[bool, true]
        # to preserve the same memory usage.
        dtype = SparseDtype(bool, self._null_fill_value)
        if self._null_fill_value:
            return type(self)._simple_new(isna(self.sp_values), self.sp_index, dtype)
        mask = np.full(len(self), False, dtype=np.bool_)
        mask[self.sp_index.indices] = isna(self.sp_values)
        return type(self)(mask, fill_value=False, dtype=dtype)

    def _pad_or_backfill(  # pylint: disable=useless-parent-delegation
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # TODO(3.0): We can remove this method once deprecation for fillna method
        #  keyword is enforced.
        return super()._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )

    def fillna(
        self,
        value=None,
        method: FillnaOptions | None = None,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Fill missing values with `value`.

        Parameters
        ----------
        value : scalar, optional
        method : str, optional

            .. warning::

               Using 'method' will result in high memory use,
               as all `fill_value` methods will be converted to
               an in-memory ndarray

        limit : int, optional

        copy: bool, default True
            Ignored for SparseArray.

        Returns
        -------
        SparseArray

        Notes
        -----
        When `value` is specified, the result's ``fill_value`` depends on
        ``self.fill_value``. The goal is to maintain low-memory use.

        If ``self.fill_value`` is NA, the result dtype will be
        ``SparseDtype(self.dtype, fill_value=value)``. This will preserve
        amount of memory used before and after filling.

        When ``self.fill_value`` is not NA, the result dtype will be
        ``self.dtype``. Again, this preserves the amount of memory used.
        """
        if (method is None and value is None) or (
            method is not None and value is not None
        ):
            raise ValueError("Must specify one of 'method' or 'value'.")

        if method is not None:
            return super().fillna(method=method, limit=limit)

        else:
            new_values = np.where(isna(self.sp_values), value, self.sp_values)

            if self._null_fill_value:
                # This is essentially just updating the dtype.
                new_dtype = SparseDtype(self.dtype.subtype, fill_value=value)
            else:
                new_dtype = self.dtype

        return self._simple_new(new_values, self._sparse_index, new_dtype)

    def shift(self, periods: int = 1, fill_value=None) -> Self:
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        subtype = np.result_type(fill_value, self.dtype.subtype)

        if subtype != self.dtype.subtype:
            # just coerce up front
            arr = self.astype(SparseDtype(subtype, self.fill_value))
        else:
            arr = self

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=arr.dtype
        )

        if periods > 0:
            a = empty
            b = arr[:-periods]
        else:
            a = arr[abs(periods) :]
            b = empty
        return arr._concat_same_type([a, b])

    def _first_fill_value_loc(self):
        """
        Get the location of the first fill value.

        Returns
        -------
        int
        """
        if len(self) == 0 or self.sp_index.npoints == len(self):
            return -1

        indices = self.sp_index.indices
        if not len(indices) or indices[0] > 0:
            return 0

        # a number larger than 1 should be appended to
        # the last in case of fill value only appears
        # in the tail of array
        diff = np.r_[np.diff(indices), 2]
        return indices[(diff > 1).argmax()] + 1

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        values = np.asarray(self)
        mask = np.asarray(self.isna())
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        uniques = algos.unique(self.sp_values)
        if len(self.sp_values) != len(self):
            fill_loc = self._first_fill_value_loc()
            # Inorder to align the behavior of pd.unique or
            # pd.Series.unique, we should keep the original
            # order, here we use unique again to find the
            # insertion place. Since the length of sp_values
            # is not large, maybe minor performance hurt
            # is worthwhile to the correctness.
            insert_loc = len(algos.unique(self.sp_values[:fill_loc]))
            uniques = np.insert(uniques, insert_loc, self.fill_value)
        return type(self)._from_sequence(uniques, dtype=self.dtype)

    def _values_for_factorize(self):
        # Still override this for hash_pandas_object
        return np.asarray(self), self.fill_value

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, SparseArray]:
        # Currently, ExtensionArray.factorize -> Tuple[ndarray, EA]
        # The sparsity on this is backwards from what Sparse would want. Want
        # ExtensionArray.factorize -> Tuple[EA, EA]
        # Given that we have to return a dense array of codes, why bother
        # implementing an efficient factorize?
        codes, uniques = algos.factorize(
            np.asarray(self), use_na_sentinel=use_na_sentinel
        )
        uniques_sp = SparseArray(uniques, dtype=self.dtype)
        return codes, uniques_sp

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN, even if NaN is in sp_values.

        Returns
        -------
        counts : Series
        """
        from pandas import (
            Index,
            Series,
        )

        keys, counts, _ = algos.value_counts_arraylike(self.sp_values, dropna=dropna)
        fcounts = self.sp_index.ngaps
        if fcounts > 0 and (not self._null_fill_value or not dropna):
            mask = isna(keys) if self._null_fill_value else keys == self.fill_value
            if mask.any():
                counts[mask] += fcounts
            else:
                # error: Argument 1 to "insert" has incompatible type "Union[
                # ExtensionArray,ndarray[Any, Any]]"; expected "Union[
                # _SupportsArray[dtype[Any]], Sequence[_SupportsArray[dtype
                # [Any]]], Sequence[Sequence[_SupportsArray[dtype[Any]]]],
                # Sequence[Sequence[Sequence[_SupportsArray[dtype[Any]]]]], Sequence
                # [Sequence[Sequence[Sequence[_SupportsArray[dtype[Any]]]]]]]"
                keys = np.insert(keys, 0, self.fill_value)  # type: ignore[arg-type]
                counts = np.insert(counts, 0, fcounts)

        if not isinstance(keys, ABCIndex):
            index = Index(keys)
        else:
            index = keys
        return Series(counts, index=index, copy=False)

    # --------
    # Indexing
    # --------
    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any:
        ...

    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | tuple[int | ellipsis, ...],
    ) -> Self:
        ...

    def __getitem__(
        self,
        key: PositionalIndexer | tuple[int | ellipsis, ...],
    ) -> Self | Any:
        if isinstance(key, tuple):
            key = unpack_tuple_and_ellipses(key)
            if key is Ellipsis:
                raise ValueError("Cannot slice with Ellipsis")

        if is_integer(key):
            return self._get_val_at(key)
        elif isinstance(key, tuple):
            # error: Invalid index type "Tuple[Union[int, ellipsis], ...]"
            # for "ndarray[Any, Any]"; expected type
            # "Union[SupportsIndex, _SupportsArray[dtype[Union[bool_,
            # integer[Any]]]], _NestedSequence[_SupportsArray[dtype[
            # Union[bool_, integer[Any]]]]], _NestedSequence[Union[
            # bool, int]], Tuple[Union[SupportsIndex, _SupportsArray[
            # dtype[Union[bool_, integer[Any]]]], _NestedSequence[
            # _SupportsArray[dtype[Union[bool_, integer[Any]]]]],
            # _NestedSequence[Union[bool, int]]], ...]]"
            data_slice = self.to_dense()[key]  # type: ignore[index]
        elif isinstance(key, slice):
            # Avoid densifying when handling contiguous slices
            if key.step is None or key.step == 1:
                start = 0 if key.start is None else key.start
                if start < 0:
                    start += len(self)

                end = len(self) if key.stop is None else key.stop
                if end < 0:
                    end += len(self)

                indices = self.sp_index.indices
                keep_inds = np.flatnonzero((indices >= start) & (indices < end))
                sp_vals = self.sp_values[keep_inds]

                sp_index = indices[keep_inds].copy()

                # If we've sliced to not include the start of the array, all our indices
                # should be shifted. NB: here we are careful to also not shift by a
                # negative value for a case like [0, 1][-100:] where the start index
                # should be treated like 0
                if start > 0:
                    sp_index -= start

                # Length of our result should match applying this slice to a range
                # of the length of our original array
                new_len = len(range(len(self))[key])
                new_sp_index = make_sparse_index(new_len, sp_index, self.kind)
                return type(self)._simple_new(sp_vals, new_sp_index, self.dtype)
            else:
                indices = np.arange(len(self), dtype=np.int32)[key]
                return self.take(indices)

        elif not is_list_like(key):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )

        else:
            if isinstance(key, SparseArray):
                # NOTE: If we guarantee that SparseDType(bool)
                # has only fill_value - true, false or nan
                # (see GH PR 44955)
                # we can apply mask very fast:
                if is_bool_dtype(key):
                    if isna(key.fill_value):
                        return self.take(key.sp_index.indices[key.sp_values])
                    if not key.fill_value:
                        return self.take(key.sp_index.indices)
                    n = len(self)
                    mask = np.full(n, True, dtype=np.bool_)
                    mask[key.sp_index.indices] = False
                    return self.take(np.arange(n)[mask])
                else:
                    key = np.asarray(key)

            key = check_array_indexer(self, key)

            if com.is_bool_indexer(key):
                # mypy doesn't know we have an array here
                key = cast(np.ndarray, key)
                return self.take(np.arange(len(key), dtype=np.int32)[key])
            elif hasattr(key, "__len__"):
                return self.take(key)
            else:
                raise ValueError(f"Cannot slice with '{key}'")

        return type(self)(data_slice, kind=self.kind)

    def _get_val_at(self, loc):
        loc = validate_insert_loc(loc, len(self))

        sp_loc = self.sp_index.lookup(loc)
        if sp_loc == -1:
            return self.fill_value
        else:
            val = self.sp_values[sp_loc]
            val = maybe_box_datetimelike(val, self.sp_values.dtype)
            return val

    def take(self, indices, *, allow_fill: bool = False, fill_value=None) -> Self:
        if is_scalar(indices):
            raise ValueError(f"'indices' must be an array, not a scalar '{indices}'.")
        indices = np.asarray(indices, dtype=np.int32)

        dtype = None
        if indices.size == 0:
            result = np.array([], dtype="object")
            dtype = self.dtype
        elif allow_fill:
            result = self._take_with_fill(indices, fill_value=fill_value)
        else:
            return self._take_without_fill(indices)

        return type(self)(
            result, fill_value=self.fill_value, kind=self.kind, dtype=dtype
        )

    def _take_with_fill(self, indices, fill_value=None) -> np.ndarray:
        if fill_value is None:
            fill_value = self.dtype.na_value

        if indices.min() < -1:
            raise ValueError(
                "Invalid value in 'indices'. Must be between -1 "
                "and the length of the array."
            )

        if indices.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if len(self) == 0:
            # Empty... Allow taking only if all empty
            if (indices == -1).all():
                dtype = np.result_type(self.sp_values, type(fill_value))
                taken = np.empty_like(indices, dtype=dtype)
                taken.fill(fill_value)
                return taken
            else:
                raise IndexError("cannot do a non-empty take from an empty axes.")

        # sp_indexer may be -1 for two reasons
        # 1.) we took for an index of -1 (new)
        # 2.) we took a value that was self.fill_value (old)
        sp_indexer = self.sp_index.lookup_array(indices)
        new_fill_indices = indices == -1
        old_fill_indices = (sp_indexer == -1) & ~new_fill_indices

        if self.sp_index.npoints == 0 and old_fill_indices.all():
            # We've looked up all valid points on an all-sparse array.
            taken = np.full(
                sp_indexer.shape, fill_value=self.fill_value, dtype=self.dtype.subtype
            )

        elif self.sp_index.npoints == 0:
            # Use the old fill_value unless we took for an index of -1
            _dtype = np.result_type(self.dtype.subtype, type(fill_value))
            taken = np.full(sp_indexer.shape, fill_value=fill_value, dtype=_dtype)
            taken[old_fill_indices] = self.fill_value
        else:
            taken = self.sp_values.take(sp_indexer)

            # Fill in two steps.
            # Old fill values
            # New fill values
            # potentially coercing to a new dtype at each stage.

            m0 = sp_indexer[old_fill_indices] < 0
            m1 = sp_indexer[new_fill_indices] < 0

            result_type = taken.dtype

            if m0.any():
                result_type = np.result_type(result_type, type(self.fill_value))
                taken = taken.astype(result_type)
                taken[old_fill_indices] = self.fill_value

            if m1.any():
                result_type = np.result_type(result_type, type(fill_value))
                taken = taken.astype(result_type)
                taken[new_fill_indices] = fill_value

        return taken

    def _take_without_fill(self, indices) -> Self:
        to_shift = indices < 0

        n = len(self)

        if (indices.max() >= n) or (indices.min() < -n):
            if n == 0:
                raise IndexError("cannot do a non-empty take from an empty axes.")
            raise IndexError("out of bounds value in 'indices'.")

        if to_shift.any():
            indices = indices.copy()
            indices[to_shift] += n

        sp_indexer = self.sp_index.lookup_array(indices)
        value_mask = sp_indexer != -1
        new_sp_values = self.sp_values[sp_indexer[value_mask]]

        value_indices = np.flatnonzero(value_mask).astype(np.int32, copy=False)

        new_sp_index = make_sparse_index(len(indices), value_indices, kind=self.kind)
        return type(self)._simple_new(new_sp_values, new_sp_index, dtype=self.dtype)

    def searchsorted(
        self,
        v: ArrayLike | object,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        msg = "searchsorted requires high memory usage."
        warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())
        v = np.asarray(v)
        return np.asarray(self, dtype=self.dtype.subtype).searchsorted(v, side, sorter)

    def copy(self) -> Self:
        values = self.sp_values.copy()
        return self._simple_new(values, self.sp_index, self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        fill_value = to_concat[0].fill_value

        values = []
        length = 0

        if to_concat:
            sp_kind = to_concat[0].kind
        else:
            sp_kind = "integer"

        sp_index: SparseIndex
        if sp_kind == "integer":
            indices = []

            for arr in to_concat:
                int_idx = arr.sp_index.indices.copy()
                int_idx += length  # TODO: wraparound
                length += arr.sp_index.length

                values.append(arr.sp_values)
                indices.append(int_idx)

            data = np.concatenate(values)
            indices_arr = np.concatenate(indices)
            # error: Argument 2 to "IntIndex" has incompatible type
            # "ndarray[Any, dtype[signedinteger[_32Bit]]]";
            # expected "Sequence[int]"
            sp_index = IntIndex(length, indices_arr)  # type: ignore[arg-type]

        else:
            # when concatenating block indices, we don't claim that you'll
            # get an identical index as concatenating the values and then
            # creating a new index. We don't want to spend the time trying
            # to merge blocks across arrays in `to_concat`, so the resulting
            # BlockIndex may have more blocks.
            blengths = []
            blocs = []

            for arr in to_concat:
                block_idx = arr.sp_index.to_block_index()

                values.append(arr.sp_values)
                blocs.append(block_idx.blocs.copy() + length)
                blengths.append(block_idx.blengths)
                length += arr.sp_index.length

            data = np.concatenate(values)
            blocs_arr = np.concatenate(blocs)
            blengths_arr = np.concatenate(blengths)

            sp_index = BlockIndex(length, blocs_arr, blengths_arr)

        return cls(data, sparse_index=sp_index, fill_value=fill_value)

    def astype(self, dtype: AstypeArg | None = None, copy: bool = True):
        """
        Change the dtype of a SparseArray.

        The output will always be a SparseArray. To convert to a dense
        ndarray with a certain dtype, use :meth:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            For SparseDtype, this changes the dtype of
            ``self.sp_values`` and the ``self.fill_value``.

            For other dtypes, this only changes the dtype of
            ``self.sp_values``.

        copy : bool, default True
            Whether to ensure a copy is made, even if not necessary.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
        >>> arr
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        >>> arr.astype(SparseDtype(np.dtype('int32')))
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a NumPy dtype with a different kind (e.g. float) will coerce
        just ``self.sp_values``.

        >>> arr.astype(SparseDtype(np.dtype('float64')))
        ... # doctest: +NORMALIZE_WHITESPACE
        [nan, nan, 1.0, 2.0]
        Fill: nan
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a SparseDtype, you can also change the fill value as well.

        >>> arr.astype(SparseDtype("float64", fill_value=0.0))
        ... # doctest: +NORMALIZE_WHITESPACE
        [0.0, 0.0, 1.0, 2.0]
        Fill: 0.0
        IntIndex
        Indices: array([2, 3], dtype=int32)
        """
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()

        future_dtype = pandas_dtype(dtype)
        if not isinstance(future_dtype, SparseDtype):
            # GH#34457
            values = np.asarray(self)
            values = ensure_wrapped_if_datetimelike(values)
            return astype_array(values, dtype=future_dtype, copy=False)

        dtype = self.dtype.update_dtype(dtype)
        subtype = pandas_dtype(dtype._subtype_with_str)
        subtype = cast(np.dtype, subtype)  # ensured by update_dtype
        values = ensure_wrapped_if_datetimelike(self.sp_values)
        sp_values = astype_array(values, subtype, copy=copy)
        sp_values = np.asarray(sp_values)

        return self._simple_new(sp_values, self.sp_index, dtype)

    def map(self, mapper, na_action=None) -> Self:
        """
        Map categories using an input mapping or function.

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
        is_map = isinstance(mapper, (abc.Mapping, ABCSeries))

        fill_val = self.fill_value

        if na_action is None or notna(fill_val):
            fill_val = mapper.get(fill_val, fill_val) if is_map else mapper(fill_val)

        def func(sp_val):
            new_sp_val = mapper.get(sp_val, None) if is_map else mapper(sp_val)
            # check identity and equality because nans are not equal to each other
            if new_sp_val is fill_val or new_sp_val == fill_val:
                msg = "fill value in the sparse values not supported"
                raise ValueError(msg)
            return new_sp_val

        sp_values = [func(x) for x in self.sp_values]

        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_val)

    def to_dense(self) -> np.ndarray:
        """
        Convert SparseArray to a NumPy array.

        Returns
        -------
        arr : NumPy array
        """
        return np.asarray(self, dtype=self.sp_values.dtype)

    def _where(self, mask, value):
        # NB: may not preserve dtype, e.g. result may be Sparse[float64]
        #  while self is Sparse[int64]
        naive_implementation = np.where(mask, self, value)
        dtype = SparseDtype(naive_implementation.dtype, fill_value=self.fill_value)
        result = type(self)._from_sequence(naive_implementation, dtype=dtype)
        return result

    # ------------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------------
    def __setstate__(self, state) -> None:
        """Necessary for making this object picklable"""
        if isinstance(state, tuple):
            # Compat for pandas < 0.24.0
            nd_state, (fill_value, sp_index) = state
            sparse_values = np.array([])
            sparse_values.__setstate__(nd_state)

            self._sparse_values = sparse_values
            self._sparse_index = sp_index
            self._dtype = SparseDtype(sparse_values.dtype, fill_value)
        else:
            self.__dict__.update(state)

    def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
        if self.fill_value == 0:
            return (self.sp_index.indices,)
        else:
            return (self.sp_index.indices[self.sp_values != 0],)

    # ------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        method = getattr(self, name, None)

        if method is None:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            arr = self
        else:
            arr = self.dropna()

        result = getattr(arr, name)(**kwargs)

        if keepdims:
            return type(self)([result], dtype=self.dtype)
        else:
            return result

    def all(self, axis=None, *args, **kwargs):
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
        nv.validate_all(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and not np.all(self.fill_value):
            return False

        return values.all()

    def any(self, axis: AxisInt = 0, *args, **kwargs) -> bool:
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
        nv.validate_any(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and np.any(self.fill_value):
            return True

        return values.any().item()

    def sum(
        self,
        axis: AxisInt = 0,
        min_count: int = 0,
        skipna: bool = True,
        *args,
        **kwargs,
    ) -> Scalar:
        """
        Sum of non-NA/null values

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        min_count : int, default 0
            The required number of valid values to perform the summation. If fewer
            than ``min_count`` valid values are present, the result will be the missing
            value indicator for subarray type.
        *args, **kwargs
            Not Used. NumPy compatibility.

        Returns
        -------
        scalar
        """
        nv.validate_sum(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        has_na = self.sp_index.ngaps > 0 and not self._null_fill_value

        if has_na and not skipna:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

        if self._null_fill_value:
            if check_below_min_count(valid_vals.shape, None, min_count):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum
        else:
            nsparse = self.sp_index.ngaps
            if check_below_min_count(valid_vals.shape, None, min_count - nsparse):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum + self.fill_value * nsparse

    def cumsum(self, axis: AxisInt = 0, *args, **kwargs) -> SparseArray:
        """
        Cumulative sum of non-NA/null values.

        When performing the cumulative summation, any non-NA/null values will
        be skipped. The resulting SparseArray will preserve the locations of
        NaN values, but the fill value will be `np.nan` regardless.

        Parameters
        ----------
        axis : int or None
            Axis over which to perform the cumulative summation. If None,
            perform cumulative summation over flattened array.

        Returns
        -------
        cumsum : SparseArray
        """
        nv.validate_cumsum(args, kwargs)

        if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
            raise ValueError(f"axis(={axis}) out of bounds")

        if not self._null_fill_value:
            return SparseArray(self.to_dense()).cumsum()

        return SparseArray(
            self.sp_values.cumsum(),
            sparse_index=self.sp_index,
            fill_value=self.fill_value,
        )

    def mean(self, axis: Axis = 0, *args, **kwargs):
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
        nv.validate_mean(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        ct = len(valid_vals)

        if self._null_fill_value:
            return sp_sum / ct
        else:
            nsparse = self.sp_index.ngaps
            return (sp_sum + self.fill_value * nsparse) / (ct + nsparse)

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Max of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("max", skipna=skipna)

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Min of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("min", skipna=skipna)

    def _min_max(self, kind: Literal["min", "max"], skipna: bool) -> Scalar:
        """
        Min/max of non-NA/null values

        Parameters
        ----------
        kind : {"min", "max"}
        skipna : bool

        Returns
        -------
        scalar
        """
        valid_vals = self._valid_sp_values
        has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0

        if len(valid_vals) > 0:
            sp_min_max = getattr(valid_vals, kind)()

            # If a non-null fill value is currently present, it might be the min/max
            if has_nonnull_fill_vals:
                func = max if kind == "max" else min
                return func(sp_min_max, self.fill_value)
            elif skipna:
                return sp_min_max
            elif self.sp_index.ngaps == 0:
                # No NAs present
                return sp_min_max
            else:
                return na_value_for_dtype(self.dtype.subtype, compat=False)
        elif has_nonnull_fill_vals:
            return self.fill_value
        else:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

    def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
        values = self._sparse_values
        index = self._sparse_index.indices
        mask = np.asarray(isna(values))
        func = np.argmax if kind == "argmax" else np.argmin

        idx = np.arange(values.shape[0])
        non_nans = values[~mask]
        non_nan_idx = idx[~mask]

        _candidate = non_nan_idx[func(non_nans)]
        candidate = index[_candidate]

        if isna(self.fill_value):
            return candidate
        if kind == "argmin" and self[candidate] < self.fill_value:
            return candidate
        if kind == "argmax" and self[candidate] > self.fill_value:
            return candidate
        _loc = self._first_fill_value_loc()
        if _loc == -1:
            # fill_value doesn't exist
            return candidate
        else:
            return _loc

    def argmax(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return self._argmin_argmax("argmax")

    def argmin(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return self._argmin_argmax("argmin")

    # ------------------------------------------------------------------------
    # Ufuncs
    # ------------------------------------------------------------------------

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (SparseArray,)):
                return NotImplemented

        # for binary ops, use our custom dunder methods
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. tests.arrays.sparse.test_arithmetics.test_ndarray_inplace
            res = arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )
            return res

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                # e.g. tests.series.test_ufunc.TestNumpyReductions
                return result

        if len(inputs) == 1:
            # No alignment necessary.
            sp_values = getattr(ufunc, method)(self.sp_values, **kwargs)
            fill_value = getattr(ufunc, method)(self.fill_value, **kwargs)

            if ufunc.nout > 1:
                # multiple outputs. e.g. modf
                arrays = tuple(
                    self._simple_new(
                        sp_value, self.sp_index, SparseDtype(sp_value.dtype, fv)
                    )
                    for sp_value, fv in zip(sp_values, fill_value)
                )
                return arrays
            elif method == "reduce":
                # e.g. reductions
                return sp_values

            return self._simple_new(
                sp_values, self.sp_index, SparseDtype(sp_values.dtype, fill_value)
            )

        new_inputs = tuple(np.asarray(x) for x in inputs)
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        if out:
            if len(out) == 1:
                out = out[0]
            return out

        if ufunc.nout > 1:
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------------

    def _arith_method(self, other, op):
        op_name = op.__name__

        if isinstance(other, SparseArray):
            return _sparse_array_op(self, other, op, op_name)

        elif is_scalar(other):
            with np.errstate(all="ignore"):
                fill = op(_get_fill(self), np.asarray(other))
                result = op(self.sp_values, other)

            if op_name == "divmod":
                left, right = result
                lfill, rfill = fill
                return (
                    _wrap_result(op_name, left, self.sp_index, lfill),
                    _wrap_result(op_name, right, self.sp_index, rfill),
                )

            return _wrap_result(op_name, result, self.sp_index, fill)

        else:
            other = np.asarray(other)
            with np.errstate(all="ignore"):
                if len(self) != len(other):
                    raise AssertionError(
                        f"length mismatch: {len(self)} vs. {len(other)}"
                    )
                if not isinstance(other, SparseArray):
                    dtype = getattr(other, "dtype", None)
                    other = SparseArray(other, fill_value=self.fill_value, dtype=dtype)
                return _sparse_array_op(self, other, op, op_name)

    def _cmp_method(self, other, op) -> SparseArray:
        if not is_scalar(other) and not isinstance(other, type(self)):
            # convert list-like to ndarray
            other = np.asarray(other)

        if isinstance(other, np.ndarray):
            # TODO: make this more flexible than just ndarray...
            other = SparseArray(other, fill_value=self.fill_value)

        if isinstance(other, SparseArray):
            if len(self) != len(other):
                raise ValueError(
                    f"operands have mismatched length {len(self)} and {len(other)}"
                )

            op_name = op.__name__.strip("_")
            return _sparse_array_op(self, other, op, op_name)
        else:
            # scalar
            fill_value = op(self.fill_value, other)
            result = np.full(len(self), fill_value, dtype=np.bool_)
            result[self.sp_index.indices] = op(self.sp_values, other)

            return type(self)(
                result,
                fill_value=fill_value,
                dtype=np.bool_,
            )

    _logical_method = _cmp_method

    def _unary_method(self, op) -> SparseArray:
        fill_value = op(np.array(self.fill_value)).item()
        dtype = SparseDtype(self.dtype.subtype, fill_value)
        # NOTE: if fill_value doesn't change
        # we just have to apply op to sp_values
        if isna(self.fill_value) or fill_value == self.fill_value:
            values = op(self.sp_values)
            return type(self)._simple_new(values, self.sp_index, self.dtype)
        # In the other case we have to recalc indexes
        return type(self)(op(self.to_dense()), dtype=dtype)

    def __pos__(self) -> SparseArray:
        return self._unary_method(operator.pos)

    def __neg__(self) -> SparseArray:
        return self._unary_method(operator.neg)

    def __invert__(self) -> SparseArray:
        return self._unary_method(operator.invert)

    def __abs__(self) -> SparseArray:
        return self._unary_method(operator.abs)

    # ----------
    # Formatting
    # -----------
    def __repr__(self) -> str:
        pp_str = printing.pprint_thing(self)
        pp_fill = printing.pprint_thing(self.fill_value)
        pp_index = printing.pprint_thing(self.sp_index)
        return f"{pp_str}\nFill: {pp_fill}\n{pp_index}"

    def _formatter(self, boxed: bool = False):
        # Defer to the formatter from the GenericArrayFormatter calling us.
        # This will infer the correct formatter from the dtype of the values.
        return None


def _make_sparse(
    arr: np.ndarray,
    kind: SparseIndexKind = "block",
    fill_value=None,
    dtype: np.dtype | None = None,
):
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
    kind : {'block', 'integer'}
    fill_value : NaN or another value
    dtype : np.dtype, optional
    copy : bool, default False

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
    """
    assert isinstance(arr, np.ndarray)

    if arr.ndim > 1:
        raise TypeError("expected dimension <= 1 data")

    if fill_value is None:
        fill_value = na_value_for_dtype(arr.dtype)

    if isna(fill_value):
        mask = notna(arr)
    else:
        # cast to object comparison to be safe
        if is_string_dtype(arr.dtype):
            arr = arr.astype(object)

        if is_object_dtype(arr.dtype):
            # element-wise equality check method in numpy doesn't treat
            # each element type, eg. 0, 0.0, and False are treated as
            # same. So we have to check the both of its type and value.
            mask = splib.make_mask_object_ndarray(arr, fill_value)
        else:
            mask = arr != fill_value

    length = len(arr)
    if length != len(mask):
        # the arr is a SparseArray
        indices = mask.sp_index.indices
    else:
        indices = mask.nonzero()[0].astype(np.int32)

    index = make_sparse_index(length, indices, kind)
    sparsified_values = arr[mask]
    if dtype is not None:
        sparsified_values = ensure_wrapped_if_datetimelike(sparsified_values)
        sparsified_values = astype_array(sparsified_values, dtype=dtype)
        sparsified_values = np.asarray(sparsified_values)

    # TODO: copy
    return sparsified_values, index, fill_value


@overload
def make_sparse_index(length: int, indices, kind: Literal["block"]) -> BlockIndex:
    ...


@overload
def make_sparse_index(length: int, indices, kind: Literal["integer"]) -> IntIndex:
    ...


def make_sparse_index(length: int, indices, kind: SparseIndexKind) -> SparseIndex:
    index: SparseIndex
    if kind == "block":
        locs, lens = splib.get_blocks(indices)
        index = BlockIndex(length, locs, lens)
    elif kind == "integer":
        index = IntIndex(length, indices)
    else:  # pragma: no cover
        raise ValueError("must be block or integer type")
    return index

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\interval.py ===
from __future__ import annotations

import operator
from operator import (
    le,
    lt,
)
import textwrap
from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.interval import (
    VALID_CLOSED,
    Interval,
    IntervalMixin,
    intervals_to_interval_bounds,
)
from pandas._libs.missing import NA
from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    FillnaOptions,
    IntervalClosedType,
    NpDtype,
    PositionalIndexer,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    SortKind,
    TimeArrayLike,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    LossySetitemError,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIntervalIndex,
    ABCPeriodIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    notna,
)

from pandas.core.algorithms import (
    isin,
    take,
    unique,
    value_counts_internal as value_counts,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
    ExtensionArray,
    _extension_array_shared_docs,
)
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
    invalid_comparison,
    unpack_zerodim_and_defer,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from pandas import (
        Index,
        Series,
    )


IntervalSide = Union[TimeArrayLike, np.ndarray]
IntervalOrNA = Union[Interval, float]

_interval_shared_docs: dict[str, str] = {}

_shared_docs_kwargs = {
    "klass": "IntervalArray",
    "qualname": "arrays.IntervalArray",
    "name": "",
}


_interval_shared_docs[
    "class"
] = """
%(summary)s

Parameters
----------
data : array-like (1-dimensional)
    Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing
    Interval objects from which to build the %(klass)s.
closed : {'left', 'right', 'both', 'neither'}, default 'right'
    Whether the intervals are closed on the left-side, right-side, both or
    neither.
dtype : dtype or None, default None
    If None, dtype will be inferred.
copy : bool, default False
    Copy the input data.
%(name)s\
verify_integrity : bool, default True
    Verify that the %(klass)s is valid.

Attributes
----------
left
right
closed
mid
length
is_empty
is_non_overlapping_monotonic
%(extra_attributes)s\

Methods
-------
from_arrays
from_tuples
from_breaks
contains
overlaps
set_closed
to_tuples
%(extra_methods)s\

See Also
--------
Index : The base pandas Index type.
Interval : A bounded slice-like interval; the elements of an %(klass)s.
interval_range : Function to create a fixed frequency IntervalIndex.
cut : Bin values into discrete Intervals.
qcut : Bin values into equal-sized Intervals based on rank or sample quantiles.

Notes
-----
See the `user guide
<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__
for more.

%(examples)s\
"""


@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalArray",
        "summary": "Pandas array for interval data that are closed on the same side.",
        "name": "",
        "extra_attributes": "",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalArray`` can be constructed directly from an array-like of
    ``Interval`` objects:

    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
    <IntervalArray>
    [(0, 1], (1, 5]]
    Length: 2, dtype: interval[int64, right]

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalArray.from_arrays`,
    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.
    """
        ),
    }
)
class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na = True
    _na_value = _fill_value = np.nan

    @property
    def ndim(self) -> Literal[1]:
        return 1

    # To make mypy recognize the fields
    _left: IntervalSide
    _right: IntervalSide
    _dtype: IntervalDtype

    # ---------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        verify_integrity: bool = True,
    ):
        data = extract_array(data, extract_numpy=True)

        if isinstance(data, cls):
            left: IntervalSide = data._left
            right: IntervalSide = data._right
            closed = closed or data.closed
            dtype = IntervalDtype(left.dtype, closed=closed)
        else:
            # don't allow scalars
            if is_scalar(data):
                msg = (
                    f"{cls.__name__}(...) must be called with a collection "
                    f"of some kind, {data} was passed"
                )
                raise TypeError(msg)

            # might need to convert empty or purely na data
            data = _maybe_convert_platform_interval(data)
            left, right, infer_closed = intervals_to_interval_bounds(
                data, validate_closed=closed is None
            )
            if left.dtype == object:
                left = lib.maybe_convert_objects(left)
                right = lib.maybe_convert_objects(right)
            closed = closed or infer_closed

            left, right, dtype = cls._ensure_simple_new_inputs(
                left,
                right,
                closed=closed,
                copy=copy,
                dtype=dtype,
            )

        if verify_integrity:
            cls._validate(left, right, dtype=dtype)

        return cls._simple_new(
            left,
            right,
            dtype=dtype,
        )

    @classmethod
    def _simple_new(
        cls,
        left: IntervalSide,
        right: IntervalSide,
        dtype: IntervalDtype,
    ) -> Self:
        result = IntervalMixin.__new__(cls)
        result._left = left
        result._right = right
        result._dtype = dtype

        return result

    @classmethod
    def _ensure_simple_new_inputs(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> tuple[IntervalSide, IntervalSide, IntervalDtype]:
        """Ensure correctness of input parameters for cls._simple_new."""
        from pandas.core.indexes.base import ensure_index

        left = ensure_index(left, copy=copy)
        left = maybe_upcast_numeric_to_64bit(left)

        right = ensure_index(right, copy=copy)
        right = maybe_upcast_numeric_to_64bit(right)

        if closed is None and isinstance(dtype, IntervalDtype):
            closed = dtype.closed

        closed = closed or "right"

        if dtype is not None:
            # GH 19262: dtype must be an IntervalDtype to override inferred
            dtype = pandas_dtype(dtype)
            if isinstance(dtype, IntervalDtype):
                if dtype.subtype is not None:
                    left = left.astype(dtype.subtype)
                    right = right.astype(dtype.subtype)
            else:
                msg = f"dtype must be an IntervalDtype, got {dtype}"
                raise TypeError(msg)

            if dtype.closed is None:
                # possibly loading an old pickle
                dtype = IntervalDtype(dtype.subtype, closed)
            elif closed != dtype.closed:
                raise ValueError("closed keyword does not match dtype.closed")

        # coerce dtypes to match if needed
        if is_float_dtype(left.dtype) and is_integer_dtype(right.dtype):
            right = right.astype(left.dtype)
        elif is_float_dtype(right.dtype) and is_integer_dtype(left.dtype):
            left = left.astype(right.dtype)

        if type(left) != type(right):
            msg = (
                f"must not have differing left [{type(left).__name__}] and "
                f"right [{type(right).__name__}] types"
            )
            raise ValueError(msg)
        if isinstance(left.dtype, CategoricalDtype) or is_string_dtype(left.dtype):
            # GH 19016
            msg = (
                "category, object, and string subtypes are not supported "
                "for IntervalArray"
            )
            raise TypeError(msg)
        if isinstance(left, ABCPeriodIndex):
            msg = "Period dtypes are not supported, use a PeriodIndex instead"
            raise ValueError(msg)
        if isinstance(left, ABCDatetimeIndex) and str(left.tz) != str(right.tz):
            msg = (
                "left and right must have the same time zone, got "
                f"'{left.tz}' and '{right.tz}'"
            )
            raise ValueError(msg)
        elif needs_i8_conversion(left.dtype) and left.unit != right.unit:
            # e.g. m8[s] vs m8[ms], try to cast to a common dtype GH#55714
            left_arr, right_arr = left._data._ensure_matching_resos(right._data)
            left = ensure_index(left_arr)
            right = ensure_index(right_arr)

        # For dt64/td64 we want DatetimeArray/TimedeltaArray instead of ndarray
        left = ensure_wrapped_if_datetimelike(left)
        left = extract_array(left, extract_numpy=True)
        right = ensure_wrapped_if_datetimelike(right)
        right = extract_array(right, extract_numpy=True)

        if isinstance(left, ArrowExtensionArray) or isinstance(
            right, ArrowExtensionArray
        ):
            pass
        else:
            lbase = getattr(left, "_ndarray", left)
            lbase = getattr(lbase, "_data", lbase).base
            rbase = getattr(right, "_ndarray", right)
            rbase = getattr(rbase, "_data", rbase).base
            if lbase is not None and lbase is rbase:
                # If these share data, then setitem could corrupt our IA
                right = right.copy()

        dtype = IntervalDtype(left.dtype, closed=closed)

        return left, right, dtype

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: IntervalArray) -> Self:
        return cls._from_sequence(values, dtype=original.dtype)

    _interval_shared_docs["from_breaks"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct from a left and right array.
        %(klass)s.from_tuples : Construct from a sequence of tuples.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        breaks = _maybe_convert_platform_interval(breaks)

        return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)

    _interval_shared_docs["from_arrays"] = textwrap.dedent(
        """
        Construct from two arrays defining the left and right bounds.

        Parameters
        ----------
        left : array-like (1-dimensional)
            Left bounds for each interval.
        right : array-like (1-dimensional)
            Right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
            splits.
        %(klass)s.from_tuples : Construct an %(klass)s from an
            array-like of tuples.

        Notes
        -----
        Each element of `left` must be less than or equal to the `right`
        element at the same position. If an element is missing, it must be
        missing in both `left` and `right`. A TypeError is raised when
        using an unsupported type for `left` or `right`. At the moment,
        'category', 'object', and 'string' subtypes are not supported.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        left = _maybe_convert_platform_interval(left)
        right = _maybe_convert_platform_interval(right)

        left, right, dtype = cls._ensure_simple_new_inputs(
            left,
            right,
            closed=closed,
            copy=copy,
            dtype=dtype,
        )
        cls._validate(left, right, dtype=dtype)

        return cls._simple_new(left, right, dtype=dtype)

    _interval_shared_docs["from_tuples"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct an %(klass)s from a left and
                                    right array.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
                                    splits.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        if len(data):
            left, right = [], []
        else:
            # ensure that empty data keeps input dtype
            left = right = data

        for d in data:
            if not isinstance(d, tuple) and isna(d):
                lhs = rhs = np.nan
            else:
                name = cls.__name__
                try:
                    # need list of length 2 tuples, e.g. [(0, 1), (1, 2), ...]
                    lhs, rhs = d
                except ValueError as err:
                    msg = f"{name}.from_tuples requires tuples of length 2, got {d}"
                    raise ValueError(msg) from err
                except TypeError as err:
                    msg = f"{name}.from_tuples received an invalid item, {d}"
                    raise TypeError(msg) from err
            left.append(lhs)
            right.append(rhs)

        return cls.from_arrays(left, right, closed, copy=False, dtype=dtype)

    @classmethod
    def _validate(cls, left, right, dtype: IntervalDtype) -> None:
        """
        Verify that the IntervalArray is valid.

        Checks that

        * dtype is correct
        * left and right match lengths
        * left and right have the same missing values
        * left is always below right
        """
        if not isinstance(dtype, IntervalDtype):
            msg = f"invalid dtype: {dtype}"
            raise ValueError(msg)
        if len(left) != len(right):
            msg = "left and right must have the same length"
            raise ValueError(msg)
        left_mask = notna(left)
        right_mask = notna(right)
        if not (left_mask == right_mask).all():
            msg = (
                "missing values must be missing in the same "
                "location both left and right sides"
            )
            raise ValueError(msg)
        if not (left[left_mask] <= right[left_mask]).all():
            msg = "left side of interval must be <= right side"
            raise ValueError(msg)

    def _shallow_copy(self, left, right) -> Self:
        """
        Return a new IntervalArray with the replacement attributes

        Parameters
        ----------
        left : Index
            Values to be used for the left-side of the intervals.
        right : Index
            Values to be used for the right-side of the intervals.
        """
        dtype = IntervalDtype(left.dtype, closed=self.closed)
        left, right, dtype = self._ensure_simple_new_inputs(left, right, dtype=dtype)

        return self._simple_new(left, right, dtype=dtype)

    # ---------------------------------------------------------------------
    # Descriptive

    @property
    def dtype(self) -> IntervalDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self.left.nbytes + self.right.nbytes

    @property
    def size(self) -> int:
        # Avoid materializing self.values
        return self.left.size

    # ---------------------------------------------------------------------
    # EA Interface

    def __iter__(self) -> Iterator:
        return iter(np.asarray(self))

    def __len__(self) -> int:
        return len(self._left)

    @overload
    def __getitem__(self, key: ScalarIndexer) -> IntervalOrNA:
        ...

    @overload
    def __getitem__(self, key: SequenceIndexer) -> Self:
        ...

    def __getitem__(self, key: PositionalIndexer) -> Self | IntervalOrNA:
        key = check_array_indexer(self, key)
        left = self._left[key]
        right = self._right[key]

        if not isinstance(left, (np.ndarray, ExtensionArray)):
            # scalar
            if is_scalar(left) and isna(left):
                return self._fill_value
            return Interval(left, right, self.closed)
        if np.ndim(left) > 1:
            # GH#30588 multi-dimensional indexer disallowed
            raise ValueError("multi-dimensional indexing not allowed")
        # Argument 2 to "_simple_new" of "IntervalArray" has incompatible type
        # "Union[Period, Timestamp, Timedelta, NaTType, DatetimeArray, TimedeltaArray,
        # ndarray[Any, Any]]"; expected "Union[Union[DatetimeArray, TimedeltaArray],
        # ndarray[Any, Any]]"
        return self._simple_new(left, right, dtype=self.dtype)  # type: ignore[arg-type]

    def __setitem__(self, key, value) -> None:
        value_left, value_right = self._validate_setitem_value(value)
        key = check_array_indexer(self, key)

        self._left[key] = value_left
        self._right[key] = value_right

    def _cmp_method(self, other, op):
        # ensure pandas array for list-like and eliminate non-interval scalars
        if is_list_like(other):
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")
            other = pd_array(other)
        elif not isinstance(other, Interval):
            # non-interval scalar -> no matches
            if other is NA:
                # GH#31882
                from pandas.core.arrays import BooleanArray

                arr = np.empty(self.shape, dtype=bool)
                mask = np.ones(self.shape, dtype=bool)
                return BooleanArray(arr, mask)
            return invalid_comparison(self, other, op)

        # determine the dtype of the elements we want to compare
        if isinstance(other, Interval):
            other_dtype = pandas_dtype("interval")
        elif not isinstance(other.dtype, CategoricalDtype):
            other_dtype = other.dtype
        else:
            # for categorical defer to categories for dtype
            other_dtype = other.categories.dtype

            # extract intervals if we have interval categories with matching closed
            if isinstance(other_dtype, IntervalDtype):
                if self.closed != other.categories.closed:
                    return invalid_comparison(self, other, op)

                other = other.categories._values.take(
                    other.codes, allow_fill=True, fill_value=other.categories._na_value
                )

        # interval-like -> need same closed and matching endpoints
        if isinstance(other_dtype, IntervalDtype):
            if self.closed != other.closed:
                return invalid_comparison(self, other, op)
            elif not isinstance(other, Interval):
                other = type(self)(other)

            if op is operator.eq:
                return (self._left == other.left) & (self._right == other.right)
            elif op is operator.ne:
                return (self._left != other.left) | (self._right != other.right)
            elif op is operator.gt:
                return (self._left > other.left) | (
                    (self._left == other.left) & (self._right > other.right)
                )
            elif op is operator.ge:
                return (self == other) | (self > other)
            elif op is operator.lt:
                return (self._left < other.left) | (
                    (self._left == other.left) & (self._right < other.right)
                )
            else:
                # operator.lt
                return (self == other) | (self < other)

        # non-interval/non-object dtype -> no matches
        if not is_object_dtype(other_dtype):
            return invalid_comparison(self, other, op)

        # object dtype -> iteratively check for intervals
        result = np.zeros(len(self), dtype=bool)
        for i, obj in enumerate(other):
            try:
                result[i] = op(self[i], obj)
            except TypeError:
                if obj is NA:
                    # comparison with np.nan returns NA
                    # github.com/pandas-dev/pandas/pull/37124#discussion_r509095092
                    result = result.astype(object)
                    result[i] = NA
                else:
                    raise
        return result

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)

        if ascending and kind == "quicksort" and na_position == "last":
            # TODO: in an IntervalIndex we can reuse the cached
            #  IntervalTree.left_sorter
            return np.lexsort((self.right, self.left))

        # TODO: other cases we can use lexsort for?  much more performant.
        return super().argsort(
            ascending=ascending, kind=kind, na_position=na_position, **kwargs
        )

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        nv.validate_minmax_axis(axis, self.ndim)

        if not len(self):
            return self._na_value

        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self

        indexer = obj.argsort()[0]
        return obj[indexer]

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        nv.validate_minmax_axis(axis, self.ndim)

        if not len(self):
            return self._na_value

        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self

        indexer = obj.argsort()[-1]
        return obj[indexer]

    def _pad_or_backfill(  # pylint: disable=useless-parent-delegation
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # TODO(3.0): after EA.fillna 'method' deprecation is enforced, we can remove
        #  this method entirely.
        return super()._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )

    def fillna(
        self, value=None, method=None, limit: int | None = None, copy: bool = True
    ) -> Self:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, a Series or dict can be used to fill in different
            values for each index. The value should not be a list. The
            value(s) passed should be either Interval objects or NA/NaN.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            (Not implemented yet for IntervalArray)
            Method to use for filling holes in reindexed Series
        limit : int, default None
            (Not implemented yet for IntervalArray)
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        filled : IntervalArray with NA/NaN filled
        """
        if copy is False:
            raise NotImplementedError
        if method is not None:
            return super().fillna(value=value, method=method, limit=limit)

        value_left, value_right = self._validate_scalar(value)

        left = self.left.fillna(value=value_left)
        right = self.right.fillna(value=value_right)
        return self._shallow_copy(left, right)

    def astype(self, dtype, copy: bool = True):
        """
        Cast to an ExtensionArray or NumPy array with dtype 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ExtensionArray or ndarray
            ExtensionArray or NumPy ndarray with 'dtype' for its dtype.
        """
        from pandas import Index

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        if isinstance(dtype, IntervalDtype):
            if dtype == self.dtype:
                return self.copy() if copy else self

            if is_float_dtype(self.dtype.subtype) and needs_i8_conversion(
                dtype.subtype
            ):
                # This is allowed on the Index.astype but we disallow it here
                msg = (
                    f"Cannot convert {self.dtype} to {dtype}; subtypes are incompatible"
                )
                raise TypeError(msg)

            # need to cast to different subtype
            try:
                # We need to use Index rules for astype to prevent casting
                #  np.nan entries to int subtypes
                new_left = Index(self._left, copy=False).astype(dtype.subtype)
                new_right = Index(self._right, copy=False).astype(dtype.subtype)
            except IntCastingNaNError:
                # e.g test_subtype_integer
                raise
            except (TypeError, ValueError) as err:
                # e.g. test_subtype_integer_errors f8->u8 can be lossy
                #  and raises ValueError
                msg = (
                    f"Cannot convert {self.dtype} to {dtype}; subtypes are incompatible"
                )
                raise TypeError(msg) from err
            return self._shallow_copy(new_left, new_right)
        else:
            try:
                return super().astype(dtype, copy=copy)
            except (TypeError, ValueError) as err:
                msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
                raise TypeError(msg) from err

    def equals(self, other) -> bool:
        if type(self) != type(other):
            return False

        return bool(
            self.closed == other.closed
            and self.left.equals(other.left)
            and self.right.equals(other.right)
        )

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[IntervalArray]) -> Self:
        """
        Concatenate multiple IntervalArray

        Parameters
        ----------
        to_concat : sequence of IntervalArray

        Returns
        -------
        IntervalArray
        """
        closed_set = {interval.closed for interval in to_concat}
        if len(closed_set) != 1:
            raise ValueError("Intervals must all be closed on the same side.")
        closed = closed_set.pop()

        left: IntervalSide = np.concatenate([interval.left for interval in to_concat])
        right: IntervalSide = np.concatenate([interval.right for interval in to_concat])

        left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed)

        return cls._simple_new(left, right, dtype=dtype)

    def copy(self) -> Self:
        """
        Return a copy of the array.

        Returns
        -------
        IntervalArray
        """
        left = self._left.copy()
        right = self._right.copy()
        dtype = self.dtype
        return self._simple_new(left, right, dtype=dtype)

    def isna(self) -> np.ndarray:
        return isna(self._left)

    def shift(self, periods: int = 1, fill_value: object = None) -> IntervalArray:
        if not len(self) or periods == 0:
            return self.copy()

        self._validate_scalar(fill_value)

        # ExtensionArray.shift doesn't work for two reasons
        # 1. IntervalArray.dtype.na_value may not be correct for the dtype.
        # 2. IntervalArray._from_sequence only accepts NaN for missing values,
        #    not other values like NaT

        empty_len = min(abs(periods), len(self))
        if isna(fill_value):
            from pandas import Index

            fill_value = Index(self._left, copy=False)._na_value
            empty = IntervalArray.from_breaks([fill_value] * (empty_len + 1))
        else:
            empty = self._from_sequence([fill_value] * empty_len, dtype=self.dtype)

        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods) :]
            b = empty
        return self._concat_same_type([a, b])

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value=None,
        axis=None,
        **kwargs,
    ) -> Self:
        """
        Take elements from the IntervalArray.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.

        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : Interval or NA, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        axis : any, default None
            Present for compat with IntervalIndex; does nothing.

        Returns
        -------
        IntervalArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        """
        nv.validate_take((), kwargs)

        fill_left = fill_right = fill_value
        if allow_fill:
            fill_left, fill_right = self._validate_scalar(fill_value)

        left_take = take(
            self._left, indices, allow_fill=allow_fill, fill_value=fill_left
        )
        right_take = take(
            self._right, indices, allow_fill=allow_fill, fill_value=fill_right
        )

        return self._shallow_copy(left_take, right_take)

    def _validate_listlike(self, value):
        # list-like of intervals
        try:
            array = IntervalArray(value)
            self._check_closed_matches(array, name="value")
            value_left, value_right = array.left, array.right
        except TypeError as err:
            # wrong type: not interval or NA
            msg = f"'value' should be an interval type, got {type(value)} instead."
            raise TypeError(msg) from err

        try:
            self.left._validate_fill_value(value_left)
        except (LossySetitemError, TypeError) as err:
            msg = (
                "'value' should be a compatible interval type, "
                f"got {type(value)} instead."
            )
            raise TypeError(msg) from err

        return value_left, value_right

    def _validate_scalar(self, value):
        if isinstance(value, Interval):
            self._check_closed_matches(value, name="value")
            left, right = value.left, value.right
            # TODO: check subdtype match like _validate_setitem_value?
        elif is_valid_na_for_dtype(value, self.left.dtype):
            # GH#18295
            left = right = self.left._na_value
        else:
            raise TypeError(
                "can only insert Interval objects and NA into an IntervalArray"
            )
        return left, right

    def _validate_setitem_value(self, value):
        if is_valid_na_for_dtype(value, self.left.dtype):
            # na value: need special casing to set directly on numpy arrays
            value = self.left._na_value
            if is_integer_dtype(self.dtype.subtype):
                # can't set NaN on a numpy integer array
                # GH#45484 TypeError, not ValueError, matches what we get with
                #  non-NA un-holdable value.
                raise TypeError("Cannot set float NaN to integer-backed IntervalArray")
            value_left, value_right = value, value

        elif isinstance(value, Interval):
            # scalar interval
            self._check_closed_matches(value, name="value")
            value_left, value_right = value.left, value.right
            self.left._validate_fill_value(value_left)
            self.left._validate_fill_value(value_right)

        else:
            return self._validate_listlike(value)

        return value_left, value_right

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of each interval.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        # TODO: implement this is a non-naive way!
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "The behavior of value_counts with object-dtype is deprecated",
                category=FutureWarning,
            )
            result = value_counts(np.asarray(self), dropna=dropna)
            # Once the deprecation is enforced, we will need to do
            #  `result.index = result.index.astype(self.dtype)`
        return result

    # ---------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        # returning 'str' here causes us to render as e.g. "(0, 1]" instead of
        #  "Interval(0, 1, closed='right')"
        return str

    # ---------------------------------------------------------------------
    # Vectorized Interval Properties/Attributes

    @property
    def left(self) -> Index:
        """
        Return the left endpoints of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.left
        Index([0, 2], dtype='int64')
        """
        from pandas import Index

        return Index(self._left, copy=False)

    @property
    def right(self) -> Index:
        """
        Return the right endpoints of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.right
        Index([1, 5], dtype='int64')
        """
        from pandas import Index

        return Index(self._right, copy=False)

    @property
    def length(self) -> Index:
        """
        Return an Index with entries denoting the length of each Interval.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.length
        Index([1, 4], dtype='int64')
        """
        return self.right - self.left

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.mid
        Index([0.5, 3.0], dtype='float64')
        """
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * self.length

    _interval_shared_docs["overlaps"] = textwrap.dedent(
        """
        Check elementwise if an Interval overlaps the values in the %(klass)s.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : %(klass)s
            Interval to check against for an overlap.

        Returns
        -------
        ndarray
            Boolean array positionally indicating where an overlap occurs.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.

        Examples
        --------
        %(examples)s
        >>> intervals.overlaps(pd.Interval(0.5, 1.5))
        array([ True,  True, False])

        Intervals that share closed endpoints overlap:

        >>> intervals.overlaps(pd.Interval(1, 3, closed='left'))
        array([ True,  True, True])

        Intervals that only have an open endpoint in common do not overlap:

        >>> intervals.overlaps(pd.Interval(1, 2, closed='right'))
        array([False,  True, False])
        """
    )

    @Appender(
        _interval_shared_docs["overlaps"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """\
        >>> data = [(0, 1), (1, 3), (2, 4)]
        >>> intervals = pd.arrays.IntervalArray.from_tuples(data)
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def overlaps(self, other):
        if isinstance(other, (IntervalArray, ABCIntervalIndex)):
            raise NotImplementedError
        if not isinstance(other, Interval):
            msg = f"`other` must be Interval-like, got {type(other).__name__}"
            raise TypeError(msg)

        # equality is okay if both endpoints are closed (overlap at a point)
        op1 = le if (self.closed_left and other.closed_right) else lt
        op2 = le if (other.closed_left and self.closed_right) else lt

        # overlaps is equivalent negation of two interval being disjoint:
        # disjoint = (A.left > B.right) or (B.left > A.right)
        # (simplifying the negation allows this to be done in less operations)
        return op1(self.left, other.right) & op2(other.left, self.right)

    # ---------------------------------------------------------------------

    @property
    def closed(self) -> IntervalClosedType:
        """
        String describing the inclusive side the intervals.

        Either ``left``, ``right``, ``both`` or ``neither``.

        Examples
        --------

        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.closed
        'right'

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.closed
        'right'
        """
        return self.dtype.closed

    _interval_shared_docs["set_closed"] = textwrap.dedent(
        """
        Return an identical %(klass)s closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.

        Returns
        -------
        %(klass)s

        %(examples)s\
        """
    )

    @Appender(
        _interval_shared_docs["set_closed"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))
        >>> index
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        >>> index.set_closed('both')
        <IntervalArray>
        [[0, 1], [1, 2], [2, 3]]
        Length: 3, dtype: interval[int64, both]
        """
            ),
        }
    )
    def set_closed(self, closed: IntervalClosedType) -> Self:
        if closed not in VALID_CLOSED:
            msg = f"invalid option for 'closed': {closed}"
            raise ValueError(msg)

        left, right = self._left, self._right
        dtype = IntervalDtype(left.dtype, closed=closed)
        return self._simple_new(left, right, dtype=dtype)

    _interval_shared_docs[
        "is_non_overlapping_monotonic"
    ] = """
        Return a boolean whether the %(klass)s is non-overlapping and monotonic.

        Non-overlapping means (no Intervals share points), and monotonic means
        either monotonic increasing or monotonic decreasing.

        Examples
        --------
        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        True

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1),
        ...                                       pd.Interval(-1, 0.1)])
        >>> interv_arr
        <IntervalArray>
        [(0.0, 1.0], (-1.0, 0.1]]
        Length: 2, dtype: interval[float64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        False

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.is_non_overlapping_monotonic
        True

        >>> interv_idx = pd.interval_range(start=0, end=2, closed='both')
        >>> interv_idx
        IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')
        >>> interv_idx.is_non_overlapping_monotonic
        False
        """

    @property
    @Appender(
        _interval_shared_docs["is_non_overlapping_monotonic"] % _shared_docs_kwargs
    )
    def is_non_overlapping_monotonic(self) -> bool:
        # must be increasing  (e.g., [0, 1), [1, 2), [2, 3), ... )
        # or decreasing (e.g., [-1, 0), [-2, -1), [-3, -2), ...)
        # we already require left <= right

        # strict inequality for closed == 'both'; equality implies overlapping
        # at a point when both sides of intervals are included
        if self.closed == "both":
            return bool(
                (self._right[:-1] < self._left[1:]).all()
                or (self._left[:-1] > self._right[1:]).all()
            )

        # non-strict inequality when closed != 'both'; at least one side is
        # not included in the intervals, so equality does not imply overlapping
        return bool(
            (self._right[:-1] <= self._left[1:]).all()
            or (self._left[:-1] >= self._right[1:]).all()
        )

    # ---------------------------------------------------------------------
    # Conversion

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        Return the IntervalArray's data as a numpy array of Interval
        objects (with dtype='object')
        """
        if copy is False:
            warnings.warn(
                "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                "changed and passing 'copy=False' raises an error when returning "
                "a zero-copy NumPy array is not possible. pandas will follow "
                "this behavior starting with pandas 3.0.\nThis conversion to "
                "NumPy requires a copy, but 'copy=False' was passed. Consider "
                "using 'np.asarray(..)' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        left = self._left
        right = self._right
        mask = self.isna()
        closed = self.closed

        result = np.empty(len(left), dtype=object)
        for i, left_value in enumerate(left):
            if mask[i]:
                result[i] = np.nan
            else:
                result[i] = Interval(left_value, right[i], closed)
        return result

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow

        from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

        try:
            subtype = pyarrow.from_numpy_dtype(self.dtype.subtype)
        except TypeError as err:
            raise TypeError(
                f"Conversion to arrow with subtype '{self.dtype.subtype}' "
                "is not supported"
            ) from err
        interval_type = ArrowIntervalType(subtype, self.closed)
        storage_array = pyarrow.StructArray.from_arrays(
            [
                pyarrow.array(self._left, type=subtype, from_pandas=True),
                pyarrow.array(self._right, type=subtype, from_pandas=True),
            ],
            names=["left", "right"],
        )
        mask = self.isna()
        if mask.any():
            # if there are missing values, set validity bitmap also on the array level
            null_bitmap = pyarrow.array(~mask).buffers()[1]
            storage_array = pyarrow.StructArray.from_buffers(
                storage_array.type,
                len(storage_array),
                [null_bitmap],
                children=[storage_array.field(0), storage_array.field(1)],
            )

        if type is not None:
            if type.equals(interval_type.storage_type):
                return storage_array
            elif isinstance(type, ArrowIntervalType):
                # ensure we have the same subtype and closed attributes
                if not type.equals(interval_type):
                    raise TypeError(
                        "Not supported to convert IntervalArray to type with "
                        f"different 'subtype' ({self.dtype.subtype} vs {type.subtype}) "
                        f"and 'closed' ({self.closed} vs {type.closed}) attributes"
                    )
            else:
                raise TypeError(
                    f"Not supported to convert IntervalArray to '{type}' type"
                )

        return pyarrow.ExtensionArray.from_storage(interval_type, storage_array)

    _interval_shared_docs["to_tuples"] = textwrap.dedent(
        """
        Return an %(return_type)s of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.

        Returns
        -------
        tuples: %(return_type)s
        %(examples)s\
        """
    )

    @Appender(
        _interval_shared_docs["to_tuples"]
        % {
            "return_type": (
                "ndarray (if self is IntervalArray) or Index (if self is IntervalIndex)"
            ),
            "examples": textwrap.dedent(
                """\

         Examples
         --------
         For :class:`pandas.IntervalArray`:

         >>> idx = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
         >>> idx
         <IntervalArray>
         [(0, 1], (1, 2]]
         Length: 2, dtype: interval[int64, right]
         >>> idx.to_tuples()
         array([(0, 1), (1, 2)], dtype=object)

         For :class:`pandas.IntervalIndex`:

         >>> idx = pd.interval_range(start=0, end=2)
         >>> idx
         IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
         >>> idx.to_tuples()
         Index([(0, 1), (1, 2)], dtype='object')
         """
            ),
        }
    )
    def to_tuples(self, na_tuple: bool = True) -> np.ndarray:
        tuples = com.asarray_tuplesafe(zip(self._left, self._right))
        if not na_tuple:
            # GH 18756
            tuples = np.where(~self.isna(), tuples, np.nan)
        return tuples

    # ---------------------------------------------------------------------

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        value_left, value_right = self._validate_setitem_value(value)

        if isinstance(self._left, np.ndarray):
            np.putmask(self._left, mask, value_left)
            assert isinstance(self._right, np.ndarray)
            np.putmask(self._right, mask, value_right)
        else:
            self._left._putmask(mask, value_left)
            assert not isinstance(self._right, np.ndarray)
            self._right._putmask(mask, value_right)

    def insert(self, loc: int, item: Interval) -> Self:
        """
        Return a new IntervalArray inserting new item at location. Follows
        Python numpy.insert semantics for negative values.  Only Interval
        objects and NA can be inserted into an IntervalIndex

        Parameters
        ----------
        loc : int
        item : Interval

        Returns
        -------
        IntervalArray
        """
        left_insert, right_insert = self._validate_scalar(item)

        new_left = self.left.insert(loc, left_insert)
        new_right = self.right.insert(loc, right_insert)

        return self._shallow_copy(new_left, new_right)

    def delete(self, loc) -> Self:
        if isinstance(self._left, np.ndarray):
            new_left = np.delete(self._left, loc)
            assert isinstance(self._right, np.ndarray)
            new_right = np.delete(self._right, loc)
        else:
            new_left = self._left.delete(loc)
            assert not isinstance(self._right, np.ndarray)
            new_right = self._right.delete(loc)
        return self._shallow_copy(left=new_left, right=new_right)

    @Appender(_extension_array_shared_docs["repeat"] % _shared_docs_kwargs)
    def repeat(
        self,
        repeats: int | Sequence[int],
        axis: AxisInt | None = None,
    ) -> Self:
        nv.validate_repeat((), {"axis": axis})
        left_repeat = self.left.repeat(repeats)
        right_repeat = self.right.repeat(repeats)
        return self._shallow_copy(left=left_repeat, right=right_repeat)

    _interval_shared_docs["contains"] = textwrap.dedent(
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the %(klass)s.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        %(klass)s.overlaps : Check if an Interval overlaps the values in the
            %(klass)s.

        Examples
        --------
        %(examples)s
        >>> intervals.contains(0.5)
        array([ True, False, False])
    """
    )

    @Appender(
        _interval_shared_docs["contains"]
        % {
            "klass": "IntervalArray",
            "examples": textwrap.dedent(
                """\
        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def contains(self, other):
        if isinstance(other, Interval):
            raise NotImplementedError("contains not implemented for two intervals")

        return (self._left < other if self.open_left else self._left <= other) & (
            other < self._right if self.open_right else other <= self._right
        )

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        if isinstance(values, IntervalArray):
            if self.closed != values.closed:
                # not comparable -> no overlap
                return np.zeros(self.shape, dtype=bool)

            if self.dtype == values.dtype:
                # GH#38353 instead of casting to object, operating on a
                #  complex128 ndarray is much more performant.
                left = self._combined.view("complex128")
                right = values._combined.view("complex128")
                # error: Argument 1 to "isin" has incompatible type
                # "Union[ExtensionArray, ndarray[Any, Any],
                # ndarray[Any, dtype[Any]]]"; expected
                # "Union[_SupportsArray[dtype[Any]],
                # _NestedSequence[_SupportsArray[dtype[Any]]], bool,
                # int, float, complex, str, bytes, _NestedSequence[
                # Union[bool, int, float, complex, str, bytes]]]"
                return np.isin(left, right).ravel()  # type: ignore[arg-type]

            elif needs_i8_conversion(self.left.dtype) ^ needs_i8_conversion(
                values.left.dtype
            ):
                # not comparable -> no overlap
                return np.zeros(self.shape, dtype=bool)

        return isin(self.astype(object), values.astype(object))

    @property
    def _combined(self) -> IntervalSide:
        # error: Item "ExtensionArray" of "ExtensionArray | ndarray[Any, Any]"
        # has no attribute "reshape"  [union-attr]
        left = self.left._values.reshape(-1, 1)  # type: ignore[union-attr]
        right = self.right._values.reshape(-1, 1)  # type: ignore[union-attr]
        if needs_i8_conversion(left.dtype):
            # error: Item "ndarray[Any, Any]" of "Any | ndarray[Any, Any]" has
            # no attribute "_concat_same_type"
            comb = left._concat_same_type(  # type: ignore[union-attr]
                [left, right], axis=1
            )
        else:
            comb = np.concatenate([left, right], axis=1)
        return comb

    def _from_combined(self, combined: np.ndarray) -> IntervalArray:
        """
        Create a new IntervalArray with our dtype from a 1D complex128 ndarray.
        """
        nc = combined.view("i8").reshape(-1, 2)

        dtype = self._left.dtype
        if needs_i8_conversion(dtype):
            assert isinstance(self._left, (DatetimeArray, TimedeltaArray))
            new_left = type(self._left)._from_sequence(nc[:, 0], dtype=dtype)
            assert isinstance(self._right, (DatetimeArray, TimedeltaArray))
            new_right = type(self._right)._from_sequence(nc[:, 1], dtype=dtype)
        else:
            assert isinstance(dtype, np.dtype)
            new_left = nc[:, 0].view(dtype)
            new_right = nc[:, 1].view(dtype)
        return self._shallow_copy(left=new_left, right=new_right)

    def unique(self) -> IntervalArray:
        # No overload variant of "__getitem__" of "ExtensionArray" matches argument
        # type "Tuple[slice, int]"
        nc = unique(
            self._combined.view("complex128")[:, 0]  # type: ignore[call-overload]
        )
        nc = nc[:, None]
        return self._from_combined(nc)


def _maybe_convert_platform_interval(values) -> ArrayLike:
    """
    Try to do platform conversion, with special casing for IntervalArray.
    Wrapper around maybe_convert_platform that alters the default return
    dtype in certain cases to be compatible with IntervalArray.  For example,
    empty lists return with integer dtype instead of object dtype, which is
    prohibited for IntervalArray.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    array
    """
    if isinstance(values, (list, tuple)) and len(values) == 0:
        # GH 19016
        # empty lists/tuples get object dtype by default, but this is
        # prohibited for IntervalArray, so coerce to integer instead
        return np.array([], dtype=np.int64)
    elif not is_list_like(values) or isinstance(values, ABCDataFrame):
        # This will raise later, but we avoid passing to maybe_convert_platform
        return values
    elif isinstance(getattr(values, "dtype", None), CategoricalDtype):
        values = np.asarray(values)
    elif not hasattr(values, "dtype") and not isinstance(values, (list, tuple, range)):
        # TODO: should we just cast these to list?
        return values
    else:
        values = extract_array(values, extract_numpy=True)

    if not hasattr(values, "dtype"):
        values = np.asarray(values)
        if values.dtype.kind in "iu" and values.dtype != np.int64:
            values = values.astype(np.int64)
    return values

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\decl_api.py ===
# orm/decl_api.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Public API functions and helpers for declarative."""

from __future__ import annotations

import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import attributes
from . import clsregistry
from . import instrumentation
from . import interfaces
from . import mapperlib
from ._orm_constructors import composite
from ._orm_constructors import deferred
from ._orm_constructors import mapped_column
from ._orm_constructors import relationship
from ._orm_constructors import synonym
from .attributes import InstrumentedAttribute
from .base import _inspect_mapped_class
from .base import _is_mapped_class
from .base import Mapped
from .base import ORMDescriptor
from .decl_base import _add_attribute
from .decl_base import _as_declarative
from .decl_base import _ClassScanMapperConfig
from .decl_base import _declarative_constructor
from .decl_base import _DeferredMapperConfig
from .decl_base import _del_attribute
from .decl_base import _mapper
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .descriptor_props import Synonym as _orm_synonym
from .mapper import Mapper
from .properties import MappedColumn
from .relationships import RelationshipProperty
from .state import InstanceState
from .. import exc
from .. import inspection
from .. import util
from ..sql import sqltypes
from ..sql.base import _NoArg
from ..sql.elements import SQLCoreOperations
from ..sql.schema import MetaData
from ..sql.selectable import FromClause
from ..util import hybridmethod
from ..util import hybridproperty
from ..util import typing as compat_typing
from ..util import warn_deprecated
from ..util.typing import CallableReference
from ..util.typing import de_optionalize_union_types
from ..util.typing import flatten_newtype
from ..util.typing import is_generic
from ..util.typing import is_literal
from ..util.typing import is_newtype
from ..util.typing import is_pep695
from ..util.typing import Literal
from ..util.typing import LITERAL_TYPES
from ..util.typing import Self

if TYPE_CHECKING:
    from ._typing import _O
    from ._typing import _RegistryType
    from .decl_base import _DataclassArguments
    from .instrumentation import ClassManager
    from .interfaces import MapperProperty
    from .state import InstanceState  # noqa
    from ..sql._typing import _TypeEngineArgument
    from ..sql.type_api import _MatchedOnType

_T = TypeVar("_T", bound=Any)

_TT = TypeVar("_TT", bound=Any)

# it's not clear how to have Annotated, Union objects etc. as keys here
# from a typing perspective so just leave it open ended for now
_TypeAnnotationMapType = Mapping[Any, "_TypeEngineArgument[Any]"]
_MutableTypeAnnotationMapType = Dict[Any, "_TypeEngineArgument[Any]"]

_DeclaredAttrDecorated = Callable[
    ..., Union[Mapped[_T], ORMDescriptor[_T], SQLCoreOperations[_T]]
]


def has_inherited_table(cls: Type[_O]) -> bool:
    """Given a class, return True if any of the classes it inherits from has a
    mapped table, otherwise return False.

    This is used in declarative mixins to build attributes that behave
    differently for the base class vs. a subclass in an inheritance
    hierarchy.

    .. seealso::

        :ref:`decl_mixin_inheritance`

    """
    for class_ in cls.__mro__[1:]:
        if getattr(class_, "__table__", None) is not None:
            return True
    return False


class _DynamicAttributesType(type):
    def __setattr__(cls, key: str, value: Any) -> None:
        if "__mapper__" in cls.__dict__:
            _add_attribute(cls, key, value)
        else:
            type.__setattr__(cls, key, value)

    def __delattr__(cls, key: str) -> None:
        if "__mapper__" in cls.__dict__:
            _del_attribute(cls, key)
        else:
            type.__delattr__(cls, key)


class DeclarativeAttributeIntercept(
    _DynamicAttributesType,
    # Inspectable is used only by the mypy plugin
    inspection.Inspectable[Mapper[Any]],
):
    """Metaclass that may be used in conjunction with the
    :class:`_orm.DeclarativeBase` class to support addition of class
    attributes dynamically.

    """


@compat_typing.dataclass_transform(
    field_specifiers=(
        MappedColumn,
        RelationshipProperty,
        Composite,
        Synonym,
        mapped_column,
        relationship,
        composite,
        synonym,
        deferred,
    ),
)
class DCTransformDeclarative(DeclarativeAttributeIntercept):
    """metaclass that includes @dataclass_transforms"""


class DeclarativeMeta(DeclarativeAttributeIntercept):
    metadata: MetaData
    registry: RegistryType

    def __init__(
        cls, classname: Any, bases: Any, dict_: Any, **kw: Any
    ) -> None:
        # use cls.__dict__, which can be modified by an
        # __init_subclass__() method (#7900)
        dict_ = cls.__dict__

        # early-consume registry from the initial declarative base,
        # assign privately to not conflict with subclass attributes named
        # "registry"
        reg = getattr(cls, "_sa_registry", None)
        if reg is None:
            reg = dict_.get("registry", None)
            if not isinstance(reg, registry):
                raise exc.InvalidRequestError(
                    "Declarative base class has no 'registry' attribute, "
                    "or registry is not a sqlalchemy.orm.registry() object"
                )
            else:
                cls._sa_registry = reg

        if not cls.__dict__.get("__abstract__", False):
            _as_declarative(reg, cls, dict_)
        type.__init__(cls, classname, bases, dict_)


def synonym_for(
    name: str, map_column: bool = False
) -> Callable[[Callable[..., Any]], Synonym[Any]]:
    """Decorator that produces an :func:`_orm.synonym`
    attribute in conjunction with a Python descriptor.

    The function being decorated is passed to :func:`_orm.synonym` as the
    :paramref:`.orm.synonym.descriptor` parameter::

        class MyClass(Base):
            __tablename__ = "my_table"

            id = Column(Integer, primary_key=True)
            _job_status = Column("job_status", String(50))

            @synonym_for("job_status")
            @property
            def job_status(self):
                return "Status: %s" % self._job_status

    The :ref:`hybrid properties <mapper_hybrids>` feature of SQLAlchemy
    is typically preferred instead of synonyms, which is a more legacy
    feature.

    .. seealso::

        :ref:`synonyms` - Overview of synonyms

        :func:`_orm.synonym` - the mapper-level function

        :ref:`mapper_hybrids` - The Hybrid Attribute extension provides an
        updated approach to augmenting attribute behavior more flexibly than
        can be achieved with synonyms.

    """

    def decorate(fn: Callable[..., Any]) -> Synonym[Any]:
        return _orm_synonym(name, map_column=map_column, descriptor=fn)

    return decorate


class _declared_attr_common:
    def __init__(
        self,
        fn: Callable[..., Any],
        cascading: bool = False,
        quiet: bool = False,
    ):
        # suppport
        # @declared_attr
        # @classmethod
        # def foo(cls) -> Mapped[thing]:
        #    ...
        # which seems to help typing tools interpret the fn as a classmethod
        # for situations where needed
        if isinstance(fn, classmethod):
            fn = fn.__func__

        self.fget = fn
        self._cascading = cascading
        self._quiet = quiet
        self.__doc__ = fn.__doc__

    def _collect_return_annotation(self) -> Optional[Type[Any]]:
        return util.get_annotations(self.fget).get("return")

    def __get__(self, instance: Optional[object], owner: Any) -> Any:
        # the declared_attr needs to make use of a cache that exists
        # for the span of the declarative scan_attributes() phase.
        # to achieve this we look at the class manager that's configured.

        # note this method should not be called outside of the declarative
        # setup phase

        cls = owner
        manager = attributes.opt_manager_of_class(cls)
        if manager is None:
            if not re.match(r"^__.+__$", self.fget.__name__):
                # if there is no manager at all, then this class hasn't been
                # run through declarative or mapper() at all, emit a warning.
                util.warn(
                    "Unmanaged access of declarative attribute %s from "
                    "non-mapped class %s" % (self.fget.__name__, cls.__name__)
                )
            return self.fget(cls)
        elif manager.is_mapped:
            # the class is mapped, which means we're outside of the declarative
            # scan setup, just run the function.
            return self.fget(cls)

        # here, we are inside of the declarative scan.  use the registry
        # that is tracking the values of these attributes.
        declarative_scan = manager.declarative_scan()

        # assert that we are in fact in the declarative scan
        assert declarative_scan is not None

        reg = declarative_scan.declared_attr_reg

        if self in reg:
            return reg[self]
        else:
            reg[self] = obj = self.fget(cls)
            return obj


class _declared_directive(_declared_attr_common, Generic[_T]):
    # see mapping_api.rst for docstring

    if typing.TYPE_CHECKING:

        def __init__(
            self,
            fn: Callable[..., _T],
            cascading: bool = False,
        ): ...

        def __get__(self, instance: Optional[object], owner: Any) -> _T: ...

        def __set__(self, instance: Any, value: Any) -> None: ...

        def __delete__(self, instance: Any) -> None: ...

        def __call__(self, fn: Callable[..., _TT]) -> _declared_directive[_TT]:
            # extensive fooling of mypy underway...
            ...


class declared_attr(interfaces._MappedAttribute[_T], _declared_attr_common):
    """Mark a class-level method as representing the definition of
    a mapped property or Declarative directive.

    :class:`_orm.declared_attr` is typically applied as a decorator to a class
    level method, turning the attribute into a scalar-like property that can be
    invoked from the uninstantiated class. The Declarative mapping process
    looks for these :class:`_orm.declared_attr` callables as it scans classes,
    and assumes any attribute marked with :class:`_orm.declared_attr` will be a
    callable that will produce an object specific to the Declarative mapping or
    table configuration.

    :class:`_orm.declared_attr` is usually applicable to
    :ref:`mixins <orm_mixins_toplevel>`, to define relationships that are to be
    applied to different implementors of the class. It may also be used to
    define dynamically generated column expressions and other Declarative
    attributes.

    Example::

        class ProvidesUserMixin:
            "A mixin that adds a 'user' relationship to classes."

            user_id: Mapped[int] = mapped_column(ForeignKey("user_table.id"))

            @declared_attr
            def user(cls) -> Mapped["User"]:
                return relationship("User")

    When used with Declarative directives such as ``__tablename__``, the
    :meth:`_orm.declared_attr.directive` modifier may be used which indicates
    to :pep:`484` typing tools that the given method is not dealing with
    :class:`_orm.Mapped` attributes::

        class CreateTableName:
            @declared_attr.directive
            def __tablename__(cls) -> str:
                return cls.__name__.lower()

    :class:`_orm.declared_attr` can also be applied directly to mapped
    classes, to allow for attributes that dynamically configure themselves
    on subclasses when using mapped inheritance schemes.   Below
    illustrates :class:`_orm.declared_attr` to create a dynamic scheme
    for generating the :paramref:`_orm.Mapper.polymorphic_identity` parameter
    for subclasses::

        class Employee(Base):
            __tablename__ = "employee"

            id: Mapped[int] = mapped_column(primary_key=True)
            type: Mapped[str] = mapped_column(String(50))

            @declared_attr.directive
            def __mapper_args__(cls) -> Dict[str, Any]:
                if cls.__name__ == "Employee":
                    return {
                        "polymorphic_on": cls.type,
                        "polymorphic_identity": "Employee",
                    }
                else:
                    return {"polymorphic_identity": cls.__name__}


        class Engineer(Employee):
            pass

    :class:`_orm.declared_attr` supports decorating functions that are
    explicitly decorated with ``@classmethod``. This is never necessary from a
    runtime perspective, however may be needed in order to support :pep:`484`
    typing tools that don't otherwise recognize the decorated function as
    having class-level behaviors for the ``cls`` parameter::

        class SomethingMixin:
            x: Mapped[int]
            y: Mapped[int]

            @declared_attr
            @classmethod
            def x_plus_y(cls) -> Mapped[int]:
                return column_property(cls.x + cls.y)

    .. versionadded:: 2.0 - :class:`_orm.declared_attr` can accommodate a
       function decorated with ``@classmethod`` to help with :pep:`484`
       integration where needed.


    .. seealso::

        :ref:`orm_mixins_toplevel` - Declarative Mixin documentation with
        background on use patterns for :class:`_orm.declared_attr`.

    """  # noqa: E501

    if typing.TYPE_CHECKING:

        def __init__(
            self,
            fn: _DeclaredAttrDecorated[_T],
            cascading: bool = False,
        ): ...

        def __set__(self, instance: Any, value: Any) -> None: ...

        def __delete__(self, instance: Any) -> None: ...

        # this is the Mapped[] API where at class descriptor get time we want
        # the type checker to see InstrumentedAttribute[_T].   However the
        # callable function prior to mapping in fact calls the given
        # declarative function that does not return InstrumentedAttribute
        @overload
        def __get__(
            self, instance: None, owner: Any
        ) -> InstrumentedAttribute[_T]: ...

        @overload
        def __get__(self, instance: object, owner: Any) -> _T: ...

        def __get__(
            self, instance: Optional[object], owner: Any
        ) -> Union[InstrumentedAttribute[_T], _T]: ...

    @hybridmethod
    def _stateful(cls, **kw: Any) -> _stateful_declared_attr[_T]:
        return _stateful_declared_attr(**kw)

    @hybridproperty
    def directive(cls) -> _declared_directive[Any]:
        # see mapping_api.rst for docstring
        return _declared_directive  # type: ignore

    @hybridproperty
    def cascading(cls) -> _stateful_declared_attr[_T]:
        # see mapping_api.rst for docstring
        return cls._stateful(cascading=True)


class _stateful_declared_attr(declared_attr[_T]):
    kw: Dict[str, Any]

    def __init__(self, **kw: Any):
        self.kw = kw

    @hybridmethod
    def _stateful(self, **kw: Any) -> _stateful_declared_attr[_T]:
        new_kw = self.kw.copy()
        new_kw.update(kw)
        return _stateful_declared_attr(**new_kw)

    def __call__(self, fn: _DeclaredAttrDecorated[_T]) -> declared_attr[_T]:
        return declared_attr(fn, **self.kw)


def declarative_mixin(cls: Type[_T]) -> Type[_T]:
    """Mark a class as providing the feature of "declarative mixin".

    E.g.::

        from sqlalchemy.orm import declared_attr
        from sqlalchemy.orm import declarative_mixin


        @declarative_mixin
        class MyMixin:

            @declared_attr
            def __tablename__(cls):
                return cls.__name__.lower()

            __table_args__ = {"mysql_engine": "InnoDB"}
            __mapper_args__ = {"always_refresh": True}

            id = Column(Integer, primary_key=True)


        class MyModel(MyMixin, Base):
            name = Column(String(1000))

    The :func:`_orm.declarative_mixin` decorator currently does not modify
    the given class in any way; it's current purpose is strictly to assist
    the :ref:`Mypy plugin <mypy_toplevel>` in being able to identify
    SQLAlchemy declarative mixin classes when no other context is present.

    .. versionadded:: 1.4.6

    .. seealso::

        :ref:`orm_mixins_toplevel`

        :ref:`mypy_declarative_mixins` - in the
        :ref:`Mypy plugin documentation <mypy_toplevel>`

    """  # noqa: E501

    return cls


def _setup_declarative_base(cls: Type[Any]) -> None:
    if "metadata" in cls.__dict__:
        metadata = cls.__dict__["metadata"]
    else:
        metadata = None

    if "type_annotation_map" in cls.__dict__:
        type_annotation_map = cls.__dict__["type_annotation_map"]
    else:
        type_annotation_map = None

    reg = cls.__dict__.get("registry", None)
    if reg is not None:
        if not isinstance(reg, registry):
            raise exc.InvalidRequestError(
                "Declarative base class has a 'registry' attribute that is "
                "not an instance of sqlalchemy.orm.registry()"
            )
        elif type_annotation_map is not None:
            raise exc.InvalidRequestError(
                "Declarative base class has both a 'registry' attribute and a "
                "type_annotation_map entry.  Per-base type_annotation_maps "
                "are not supported.  Please apply the type_annotation_map "
                "to this registry directly."
            )

    else:
        reg = registry(
            metadata=metadata, type_annotation_map=type_annotation_map
        )
        cls.registry = reg

    cls._sa_registry = reg

    if "metadata" not in cls.__dict__:
        cls.metadata = cls.registry.metadata

    if getattr(cls, "__init__", object.__init__) is object.__init__:
        cls.__init__ = cls.registry.constructor


class MappedAsDataclass(metaclass=DCTransformDeclarative):
    """Mixin class to indicate when mapping this class, also convert it to be
    a dataclass.

    .. seealso::

        :ref:`orm_declarative_native_dataclasses` - complete background
        on SQLAlchemy native dataclass mapping

    .. versionadded:: 2.0

    """

    def __init_subclass__(
        cls,
        init: Union[_NoArg, bool] = _NoArg.NO_ARG,
        repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
        eq: Union[_NoArg, bool] = _NoArg.NO_ARG,
        order: Union[_NoArg, bool] = _NoArg.NO_ARG,
        unsafe_hash: Union[_NoArg, bool] = _NoArg.NO_ARG,
        match_args: Union[_NoArg, bool] = _NoArg.NO_ARG,
        kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
        dataclass_callable: Union[
            _NoArg, Callable[..., Type[Any]]
        ] = _NoArg.NO_ARG,
        **kw: Any,
    ) -> None:
        apply_dc_transforms: _DataclassArguments = {
            "init": init,
            "repr": repr,
            "eq": eq,
            "order": order,
            "unsafe_hash": unsafe_hash,
            "match_args": match_args,
            "kw_only": kw_only,
            "dataclass_callable": dataclass_callable,
        }

        current_transforms: _DataclassArguments

        if hasattr(cls, "_sa_apply_dc_transforms"):
            current = cls._sa_apply_dc_transforms

            _ClassScanMapperConfig._assert_dc_arguments(current)

            cls._sa_apply_dc_transforms = current_transforms = {  # type: ignore  # noqa: E501
                k: current.get(k, _NoArg.NO_ARG) if v is _NoArg.NO_ARG else v
                for k, v in apply_dc_transforms.items()
            }
        else:
            cls._sa_apply_dc_transforms = current_transforms = (
                apply_dc_transforms
            )

        super().__init_subclass__(**kw)

        if not _is_mapped_class(cls):
            new_anno = (
                _ClassScanMapperConfig._update_annotations_for_non_mapped_class
            )(cls)
            _ClassScanMapperConfig._apply_dataclasses_to_any_class(
                current_transforms, cls, new_anno
            )


class DeclarativeBase(
    # Inspectable is used only by the mypy plugin
    inspection.Inspectable[InstanceState[Any]],
    metaclass=DeclarativeAttributeIntercept,
):
    """Base class used for declarative class definitions.

    The :class:`_orm.DeclarativeBase` allows for the creation of new
    declarative bases in such a way that is compatible with type checkers::


        from sqlalchemy.orm import DeclarativeBase


        class Base(DeclarativeBase):
            pass

    The above ``Base`` class is now usable as the base for new declarative
    mappings.  The superclass makes use of the ``__init_subclass__()``
    method to set up new classes and metaclasses aren't used.

    When first used, the :class:`_orm.DeclarativeBase` class instantiates a new
    :class:`_orm.registry` to be used with the base, assuming one was not
    provided explicitly. The :class:`_orm.DeclarativeBase` class supports
    class-level attributes which act as parameters for the construction of this
    registry; such as to indicate a specific :class:`_schema.MetaData`
    collection as well as a specific value for
    :paramref:`_orm.registry.type_annotation_map`::

        from typing_extensions import Annotated

        from sqlalchemy import BigInteger
        from sqlalchemy import MetaData
        from sqlalchemy import String
        from sqlalchemy.orm import DeclarativeBase

        bigint = Annotated[int, "bigint"]
        my_metadata = MetaData()


        class Base(DeclarativeBase):
            metadata = my_metadata
            type_annotation_map = {
                str: String().with_variant(String(255), "mysql", "mariadb"),
                bigint: BigInteger(),
            }

    Class-level attributes which may be specified include:

    :param metadata: optional :class:`_schema.MetaData` collection.
     If a :class:`_orm.registry` is constructed automatically, this
     :class:`_schema.MetaData` collection will be used to construct it.
     Otherwise, the local :class:`_schema.MetaData` collection will supercede
     that used by an existing :class:`_orm.registry` passed using the
     :paramref:`_orm.DeclarativeBase.registry` parameter.
    :param type_annotation_map: optional type annotation map that will be
     passed to the :class:`_orm.registry` as
     :paramref:`_orm.registry.type_annotation_map`.
    :param registry: supply a pre-existing :class:`_orm.registry` directly.

    .. versionadded:: 2.0  Added :class:`.DeclarativeBase`, so that declarative
       base classes may be constructed in such a way that is also recognized
       by :pep:`484` type checkers.   As a result, :class:`.DeclarativeBase`
       and other subclassing-oriented APIs should be seen as
       superseding previous "class returned by a function" APIs, namely
       :func:`_orm.declarative_base` and :meth:`_orm.registry.generate_base`,
       where the base class returned cannot be recognized by type checkers
       without using plugins.

    **__init__ behavior**

    In a plain Python class, the base-most ``__init__()`` method in the class
    hierarchy is ``object.__init__()``, which accepts no arguments. However,
    when the :class:`_orm.DeclarativeBase` subclass is first declared, the
    class is given an ``__init__()`` method that links to the
    :paramref:`_orm.registry.constructor` constructor function, if no
    ``__init__()`` method is already present; this is the usual declarative
    constructor that will assign keyword arguments as attributes on the
    instance, assuming those attributes are established at the class level
    (i.e. are mapped, or are linked to a descriptor). This constructor is
    **never accessed by a mapped class without being called explicitly via
    super()**, as mapped classes are themselves given an ``__init__()`` method
    directly which calls :paramref:`_orm.registry.constructor`, so in the
    default case works independently of what the base-most ``__init__()``
    method does.

    .. versionchanged:: 2.0.1  :class:`_orm.DeclarativeBase` has a default
       constructor that links to :paramref:`_orm.registry.constructor` by
       default, so that calls to ``super().__init__()`` can access this
       constructor. Previously, due to an implementation mistake, this default
       constructor was missing, and calling ``super().__init__()`` would invoke
       ``object.__init__()``.

    The :class:`_orm.DeclarativeBase` subclass may also declare an explicit
    ``__init__()`` method which will replace the use of the
    :paramref:`_orm.registry.constructor` function at this level::

        class Base(DeclarativeBase):
            def __init__(self, id=None):
                self.id = id

    Mapped classes still will not invoke this constructor implicitly; it
    remains only accessible by calling ``super().__init__()``::

        class MyClass(Base):
            def __init__(self, id=None, name=None):
                self.name = name
                super().__init__(id=id)

    Note that this is a different behavior from what functions like the legacy
    :func:`_orm.declarative_base` would do; the base created by those functions
    would always install :paramref:`_orm.registry.constructor` for
    ``__init__()``.


    """

    if typing.TYPE_CHECKING:

        def _sa_inspect_type(self) -> Mapper[Self]: ...

        def _sa_inspect_instance(self) -> InstanceState[Self]: ...

        _sa_registry: ClassVar[_RegistryType]

        registry: ClassVar[_RegistryType]
        """Refers to the :class:`_orm.registry` in use where new
        :class:`_orm.Mapper` objects will be associated."""

        metadata: ClassVar[MetaData]
        """Refers to the :class:`_schema.MetaData` collection that will be used
        for new :class:`_schema.Table` objects.

        .. seealso::

            :ref:`orm_declarative_metadata`

        """

        __name__: ClassVar[str]

        # this ideally should be Mapper[Self], but mypy as of 1.4.1 does not
        # like it, and breaks the declared_attr_one test. Pyright/pylance is
        # ok with it.
        __mapper__: ClassVar[Mapper[Any]]
        """The :class:`_orm.Mapper` object to which a particular class is
        mapped.

        May also be acquired using :func:`_sa.inspect`, e.g.
        ``inspect(klass)``.

        """

        __table__: ClassVar[FromClause]
        """The :class:`_sql.FromClause` to which a particular subclass is
        mapped.

        This is usually an instance of :class:`_schema.Table` but may also
        refer to other kinds of :class:`_sql.FromClause` such as
        :class:`_sql.Subquery`, depending on how the class is mapped.

        .. seealso::

            :ref:`orm_declarative_metadata`

        """

        # pyright/pylance do not consider a classmethod a ClassVar so use Any
        # https://github.com/microsoft/pylance-release/issues/3484
        __tablename__: Any
        """String name to assign to the generated
        :class:`_schema.Table` object, if not specified directly via
        :attr:`_orm.DeclarativeBase.__table__`.

        .. seealso::

            :ref:`orm_declarative_table`

        """

        __mapper_args__: Any
        """Dictionary of arguments which will be passed to the
        :class:`_orm.Mapper` constructor.

        .. seealso::

            :ref:`orm_declarative_mapper_options`

        """

        __table_args__: Any
        """A dictionary or tuple of arguments that will be passed to the
        :class:`_schema.Table` constructor.  See
        :ref:`orm_declarative_table_configuration`
        for background on the specific structure of this collection.

        .. seealso::

            :ref:`orm_declarative_table_configuration`

        """

        def __init__(self, **kw: Any): ...

    def __init_subclass__(cls, **kw: Any) -> None:
        if DeclarativeBase in cls.__bases__:
            _check_not_declarative(cls, DeclarativeBase)
            _setup_declarative_base(cls)
        else:
            _as_declarative(cls._sa_registry, cls, cls.__dict__)
        super().__init_subclass__(**kw)


def _check_not_declarative(cls: Type[Any], base: Type[Any]) -> None:
    cls_dict = cls.__dict__
    if (
        "__table__" in cls_dict
        and not (
            callable(cls_dict["__table__"])
            or hasattr(cls_dict["__table__"], "__get__")
        )
    ) or isinstance(cls_dict.get("__tablename__", None), str):
        raise exc.InvalidRequestError(
            f"Cannot use {base.__name__!r} directly as a declarative base "
            "class. Create a Base by creating a subclass of it."
        )


class DeclarativeBaseNoMeta(
    # Inspectable is used only by the mypy plugin
    inspection.Inspectable[InstanceState[Any]]
):
    """Same as :class:`_orm.DeclarativeBase`, but does not use a metaclass
    to intercept new attributes.

    The :class:`_orm.DeclarativeBaseNoMeta` base may be used when use of
    custom metaclasses is desirable.

    .. versionadded:: 2.0


    """

    _sa_registry: ClassVar[_RegistryType]

    registry: ClassVar[_RegistryType]
    """Refers to the :class:`_orm.registry` in use where new
    :class:`_orm.Mapper` objects will be associated."""

    metadata: ClassVar[MetaData]
    """Refers to the :class:`_schema.MetaData` collection that will be used
    for new :class:`_schema.Table` objects.

    .. seealso::

        :ref:`orm_declarative_metadata`

    """

    # this ideally should be Mapper[Self], but mypy as of 1.4.1 does not
    # like it, and breaks the declared_attr_one test. Pyright/pylance is
    # ok with it.
    __mapper__: ClassVar[Mapper[Any]]
    """The :class:`_orm.Mapper` object to which a particular class is
    mapped.

    May also be acquired using :func:`_sa.inspect`, e.g.
    ``inspect(klass)``.

    """

    __table__: Optional[FromClause]
    """The :class:`_sql.FromClause` to which a particular subclass is
    mapped.

    This is usually an instance of :class:`_schema.Table` but may also
    refer to other kinds of :class:`_sql.FromClause` such as
    :class:`_sql.Subquery`, depending on how the class is mapped.

    .. seealso::

        :ref:`orm_declarative_metadata`

    """

    if typing.TYPE_CHECKING:

        def _sa_inspect_type(self) -> Mapper[Self]: ...

        def _sa_inspect_instance(self) -> InstanceState[Self]: ...

        __tablename__: Any
        """String name to assign to the generated
        :class:`_schema.Table` object, if not specified directly via
        :attr:`_orm.DeclarativeBase.__table__`.

        .. seealso::

            :ref:`orm_declarative_table`

        """

        __mapper_args__: Any
        """Dictionary of arguments which will be passed to the
        :class:`_orm.Mapper` constructor.

        .. seealso::

            :ref:`orm_declarative_mapper_options`

        """

        __table_args__: Any
        """A dictionary or tuple of arguments that will be passed to the
        :class:`_schema.Table` constructor.  See
        :ref:`orm_declarative_table_configuration`
        for background on the specific structure of this collection.

        .. seealso::

            :ref:`orm_declarative_table_configuration`

        """

        def __init__(self, **kw: Any): ...

    def __init_subclass__(cls, **kw: Any) -> None:
        if DeclarativeBaseNoMeta in cls.__bases__:
            _check_not_declarative(cls, DeclarativeBaseNoMeta)
            _setup_declarative_base(cls)
        else:
            _as_declarative(cls._sa_registry, cls, cls.__dict__)
        super().__init_subclass__(**kw)


def add_mapped_attribute(
    target: Type[_O], key: str, attr: MapperProperty[Any]
) -> None:
    """Add a new mapped attribute to an ORM mapped class.

    E.g.::

        add_mapped_attribute(User, "addresses", relationship(Address))

    This may be used for ORM mappings that aren't using a declarative
    metaclass that intercepts attribute set operations.

    .. versionadded:: 2.0


    """
    _add_attribute(target, key, attr)


def declarative_base(
    *,
    metadata: Optional[MetaData] = None,
    mapper: Optional[Callable[..., Mapper[Any]]] = None,
    cls: Type[Any] = object,
    name: str = "Base",
    class_registry: Optional[clsregistry._ClsRegistryType] = None,
    type_annotation_map: Optional[_TypeAnnotationMapType] = None,
    constructor: Callable[..., None] = _declarative_constructor,
    metaclass: Type[Any] = DeclarativeMeta,
) -> Any:
    r"""Construct a base class for declarative class definitions.

    The new base class will be given a metaclass that produces
    appropriate :class:`~sqlalchemy.schema.Table` objects and makes
    the appropriate :class:`_orm.Mapper` calls based on the
    information provided declaratively in the class and any subclasses
    of the class.

    .. versionchanged:: 2.0 Note that the :func:`_orm.declarative_base`
       function is superseded by the new :class:`_orm.DeclarativeBase` class,
       which generates a new "base" class using subclassing, rather than
       return value of a function.  This allows an approach that is compatible
       with :pep:`484` typing tools.

    The :func:`_orm.declarative_base` function is a shorthand version
    of using the :meth:`_orm.registry.generate_base`
    method.  That is, the following::

        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

    Is equivalent to::

        from sqlalchemy.orm import registry

        mapper_registry = registry()
        Base = mapper_registry.generate_base()

    See the docstring for :class:`_orm.registry`
    and :meth:`_orm.registry.generate_base`
    for more details.

    .. versionchanged:: 1.4  The :func:`_orm.declarative_base`
       function is now a specialization of the more generic
       :class:`_orm.registry` class.  The function also moves to the
       ``sqlalchemy.orm`` package from the ``declarative.ext`` package.


    :param metadata:
      An optional :class:`~sqlalchemy.schema.MetaData` instance.  All
      :class:`~sqlalchemy.schema.Table` objects implicitly declared by
      subclasses of the base will share this MetaData.  A MetaData instance
      will be created if none is provided.  The
      :class:`~sqlalchemy.schema.MetaData` instance will be available via the
      ``metadata`` attribute of the generated declarative base class.

    :param mapper:
      An optional callable, defaults to :class:`_orm.Mapper`. Will
      be used to map subclasses to their Tables.

    :param cls:
      Defaults to :class:`object`. A type to use as the base for the generated
      declarative base class. May be a class or tuple of classes.

    :param name:
      Defaults to ``Base``.  The display name for the generated
      class.  Customizing this is not required, but can improve clarity in
      tracebacks and debugging.

    :param constructor:
      Specify the implementation for the ``__init__`` function on a mapped
      class that has no ``__init__`` of its own.  Defaults to an
      implementation that assigns \**kwargs for declared
      fields and relationships to an instance.  If ``None`` is supplied,
      no __init__ will be provided and construction will fall back to
      cls.__init__ by way of the normal Python semantics.

    :param class_registry: optional dictionary that will serve as the
      registry of class names-> mapped classes when string names
      are used to identify classes inside of :func:`_orm.relationship`
      and others.  Allows two or more declarative base classes
      to share the same registry of class names for simplified
      inter-base relationships.

    :param type_annotation_map: optional dictionary of Python types to
        SQLAlchemy :class:`_types.TypeEngine` classes or instances.  This
        is used exclusively by the :class:`_orm.MappedColumn` construct
        to produce column types based on annotations within the
        :class:`_orm.Mapped` type.


        .. versionadded:: 2.0

        .. seealso::

            :ref:`orm_declarative_mapped_column_type_map`

    :param metaclass:
      Defaults to :class:`.DeclarativeMeta`.  A metaclass or __metaclass__
      compatible callable to use as the meta type of the generated
      declarative base class.

    .. seealso::

        :class:`_orm.registry`

    """

    return registry(
        metadata=metadata,
        class_registry=class_registry,
        constructor=constructor,
        type_annotation_map=type_annotation_map,
    ).generate_base(
        mapper=mapper,
        cls=cls,
        name=name,
        metaclass=metaclass,
    )


class registry:
    """Generalized registry for mapping classes.

    The :class:`_orm.registry` serves as the basis for maintaining a collection
    of mappings, and provides configurational hooks used to map classes.

    The three general kinds of mappings supported are Declarative Base,
    Declarative Decorator, and Imperative Mapping.   All of these mapping
    styles may be used interchangeably:

    * :meth:`_orm.registry.generate_base` returns a new declarative base
      class, and is the underlying implementation of the
      :func:`_orm.declarative_base` function.

    * :meth:`_orm.registry.mapped` provides a class decorator that will
      apply declarative mapping to a class without the use of a declarative
      base class.

    * :meth:`_orm.registry.map_imperatively` will produce a
      :class:`_orm.Mapper` for a class without scanning the class for
      declarative class attributes. This method suits the use case historically
      provided by the ``sqlalchemy.orm.mapper()`` classical mapping function,
      which is removed as of SQLAlchemy 2.0.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`orm_mapping_classes_toplevel` - overview of class mapping
        styles.

    """

    _class_registry: clsregistry._ClsRegistryType
    _managers: weakref.WeakKeyDictionary[ClassManager[Any], Literal[True]]
    _non_primary_mappers: weakref.WeakKeyDictionary[Mapper[Any], Literal[True]]
    metadata: MetaData
    constructor: CallableReference[Callable[..., None]]
    type_annotation_map: _MutableTypeAnnotationMapType
    _dependents: Set[_RegistryType]
    _dependencies: Set[_RegistryType]
    _new_mappers: bool

    def __init__(
        self,
        *,
        metadata: Optional[MetaData] = None,
        class_registry: Optional[clsregistry._ClsRegistryType] = None,
        type_annotation_map: Optional[_TypeAnnotationMapType] = None,
        constructor: Callable[..., None] = _declarative_constructor,
    ):
        r"""Construct a new :class:`_orm.registry`

        :param metadata:
          An optional :class:`_schema.MetaData` instance.  All
          :class:`_schema.Table` objects generated using declarative
          table mapping will make use of this :class:`_schema.MetaData`
          collection.  If this argument is left at its default of ``None``,
          a blank :class:`_schema.MetaData` collection is created.

        :param constructor:
          Specify the implementation for the ``__init__`` function on a mapped
          class that has no ``__init__`` of its own.  Defaults to an
          implementation that assigns \**kwargs for declared
          fields and relationships to an instance.  If ``None`` is supplied,
          no __init__ will be provided and construction will fall back to
          cls.__init__ by way of the normal Python semantics.

        :param class_registry: optional dictionary that will serve as the
          registry of class names-> mapped classes when string names
          are used to identify classes inside of :func:`_orm.relationship`
          and others.  Allows two or more declarative base classes
          to share the same registry of class names for simplified
          inter-base relationships.

        :param type_annotation_map: optional dictionary of Python types to
          SQLAlchemy :class:`_types.TypeEngine` classes or instances.
          The provided dict will update the default type mapping.  This
          is used exclusively by the :class:`_orm.MappedColumn` construct
          to produce column types based on annotations within the
          :class:`_orm.Mapped` type.

          .. versionadded:: 2.0

          .. seealso::

              :ref:`orm_declarative_mapped_column_type_map`


        """
        lcl_metadata = metadata or MetaData()

        if class_registry is None:
            class_registry = weakref.WeakValueDictionary()

        self._class_registry = class_registry
        self._managers = weakref.WeakKeyDictionary()
        self._non_primary_mappers = weakref.WeakKeyDictionary()
        self.metadata = lcl_metadata
        self.constructor = constructor
        self.type_annotation_map = {}
        if type_annotation_map is not None:
            self.update_type_annotation_map(type_annotation_map)
        self._dependents = set()
        self._dependencies = set()

        self._new_mappers = False

        with mapperlib._CONFIGURE_MUTEX:
            mapperlib._mapper_registries[self] = True

    def update_type_annotation_map(
        self,
        type_annotation_map: _TypeAnnotationMapType,
    ) -> None:
        """update the :paramref:`_orm.registry.type_annotation_map` with new
        values."""

        self.type_annotation_map.update(
            {
                de_optionalize_union_types(typ): sqltype
                for typ, sqltype in type_annotation_map.items()
            }
        )

    def _resolve_type(
        self, python_type: _MatchedOnType, _do_fallbacks: bool = True
    ) -> Optional[sqltypes.TypeEngine[Any]]:
        python_type_type: Type[Any]
        search: Iterable[Tuple[_MatchedOnType, Type[Any]]]

        if is_generic(python_type):
            if is_literal(python_type):
                python_type_type = python_type  # type: ignore[assignment]

                search = (
                    (python_type, python_type_type),
                    *((lt, python_type_type) for lt in LITERAL_TYPES),
                )
            else:
                python_type_type = python_type.__origin__
                search = ((python_type, python_type_type),)
        elif isinstance(python_type, type):
            python_type_type = python_type
            search = ((pt, pt) for pt in python_type_type.__mro__)
        else:
            python_type_type = python_type  # type: ignore[assignment]
            search = ((python_type, python_type_type),)

        for pt, flattened in search:
            # we search through full __mro__ for types.  however...
            sql_type = self.type_annotation_map.get(pt)
            if sql_type is None:
                sql_type = sqltypes._type_map_get(pt)  # type: ignore  # noqa: E501

            if sql_type is not None:
                sql_type_inst = sqltypes.to_instance(sql_type)

                # ... this additional step will reject most
                # type -> supertype matches, such as if we had
                # a MyInt(int) subclass.  note also we pass NewType()
                # here directly; these always have to be in the
                # type_annotation_map to be useful
                resolved_sql_type = sql_type_inst._resolve_for_python_type(
                    python_type_type,
                    pt,
                    flattened,
                )
                if resolved_sql_type is not None:
                    return resolved_sql_type

        # 2.0 fallbacks
        if _do_fallbacks:
            python_type_to_check: Any = None
            kind = None
            if is_pep695(python_type):
                # NOTE: assume there aren't type alias types of new types.
                python_type_to_check = python_type
                while is_pep695(python_type_to_check):
                    python_type_to_check = python_type_to_check.__value__
                python_type_to_check = de_optionalize_union_types(
                    python_type_to_check
                )
                kind = "TypeAliasType"
            if is_newtype(python_type):
                python_type_to_check = flatten_newtype(python_type)
                kind = "NewType"

            if python_type_to_check is not None:
                res_after_fallback = self._resolve_type(
                    python_type_to_check, False
                )
                if res_after_fallback is not None:
                    assert kind is not None
                    warn_deprecated(
                        f"Matching the provided {kind} '{python_type}' on "
                        "its resolved value without matching it in the "
                        "type_annotation_map is deprecated; add this type to "
                        "the type_annotation_map to allow it to match "
                        "explicitly.",
                        "2.0",
                    )
                    return res_after_fallback

        return None

    @property
    def mappers(self) -> FrozenSet[Mapper[Any]]:
        """read only collection of all :class:`_orm.Mapper` objects."""

        return frozenset(manager.mapper for manager in self._managers).union(
            self._non_primary_mappers
        )

    def _set_depends_on(self, registry: RegistryType) -> None:
        if registry is self:
            return
        registry._dependents.add(self)
        self._dependencies.add(registry)

    def _flag_new_mapper(self, mapper: Mapper[Any]) -> None:
        mapper._ready_for_configure = True
        if self._new_mappers:
            return

        for reg in self._recurse_with_dependents({self}):
            reg._new_mappers = True

    @classmethod
    def _recurse_with_dependents(
        cls, registries: Set[RegistryType]
    ) -> Iterator[RegistryType]:
        todo = registries
        done = set()
        while todo:
            reg = todo.pop()
            done.add(reg)

            # if yielding would remove dependents, make sure we have
            # them before
            todo.update(reg._dependents.difference(done))
            yield reg

            # if yielding would add dependents, make sure we have them
            # after
            todo.update(reg._dependents.difference(done))

    @classmethod
    def _recurse_with_dependencies(
        cls, registries: Set[RegistryType]
    ) -> Iterator[RegistryType]:
        todo = registries
        done = set()
        while todo:
            reg = todo.pop()
            done.add(reg)

            # if yielding would remove dependencies, make sure we have
            # them before
            todo.update(reg._dependencies.difference(done))

            yield reg

            # if yielding would remove dependencies, make sure we have
            # them before
            todo.update(reg._dependencies.difference(done))

    def _mappers_to_configure(self) -> Iterator[Mapper[Any]]:
        return itertools.chain(
            (
                manager.mapper
                for manager in list(self._managers)
                if manager.is_mapped
                and not manager.mapper.configured
                and manager.mapper._ready_for_configure
            ),
            (
                npm
                for npm in list(self._non_primary_mappers)
                if not npm.configured and npm._ready_for_configure
            ),
        )

    def _add_non_primary_mapper(self, np_mapper: Mapper[Any]) -> None:
        self._non_primary_mappers[np_mapper] = True

    def _dispose_cls(self, cls: Type[_O]) -> None:
        clsregistry.remove_class(cls.__name__, cls, self._class_registry)

    def _add_manager(self, manager: ClassManager[Any]) -> None:
        self._managers[manager] = True
        if manager.is_mapped:
            raise exc.ArgumentError(
                "Class '%s' already has a primary mapper defined. "
                % manager.class_
            )
        assert manager.registry is None
        manager.registry = self

    def configure(self, cascade: bool = False) -> None:
        """Configure all as-yet unconfigured mappers in this
        :class:`_orm.registry`.

        The configure step is used to reconcile and initialize the
        :func:`_orm.relationship` linkages between mapped classes, as well as
        to invoke configuration events such as the
        :meth:`_orm.MapperEvents.before_configured` and
        :meth:`_orm.MapperEvents.after_configured`, which may be used by ORM
        extensions or user-defined extension hooks.

        If one or more mappers in this registry contain
        :func:`_orm.relationship` constructs that refer to mapped classes in
        other registries, this registry is said to be *dependent* on those
        registries. In order to configure those dependent registries
        automatically, the :paramref:`_orm.registry.configure.cascade` flag
        should be set to ``True``. Otherwise, if they are not configured, an
        exception will be raised.  The rationale behind this behavior is to
        allow an application to programmatically invoke configuration of
        registries while controlling whether or not the process implicitly
        reaches other registries.

        As an alternative to invoking :meth:`_orm.registry.configure`, the ORM
        function :func:`_orm.configure_mappers` function may be used to ensure
        configuration is complete for all :class:`_orm.registry` objects in
        memory. This is generally simpler to use and also predates the usage of
        :class:`_orm.registry` objects overall. However, this function will
        impact all mappings throughout the running Python process and may be
        more memory/time consuming for an application that has many registries
        in use for different purposes that may not be needed immediately.

        .. seealso::

            :func:`_orm.configure_mappers`


        .. versionadded:: 1.4.0b2

        """
        mapperlib._configure_registries({self}, cascade=cascade)

    def dispose(self, cascade: bool = False) -> None:
        """Dispose of all mappers in this :class:`_orm.registry`.

        After invocation, all the classes that were mapped within this registry
        will no longer have class instrumentation associated with them. This
        method is the per-:class:`_orm.registry` analogue to the
        application-wide :func:`_orm.clear_mappers` function.

        If this registry contains mappers that are dependencies of other
        registries, typically via :func:`_orm.relationship` links, then those
        registries must be disposed as well. When such registries exist in
        relation to this one, their :meth:`_orm.registry.dispose` method will
        also be called, if the :paramref:`_orm.registry.dispose.cascade` flag
        is set to ``True``; otherwise, an error is raised if those registries
        were not already disposed.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :func:`_orm.clear_mappers`

        """

        mapperlib._dispose_registries({self}, cascade=cascade)

    def _dispose_manager_and_mapper(self, manager: ClassManager[Any]) -> None:
        if "mapper" in manager.__dict__:
            mapper = manager.mapper

            mapper._set_dispose_flags()

        class_ = manager.class_
        self._dispose_cls(class_)
        instrumentation._instrumentation_factory.unregister(class_)

    def generate_base(
        self,
        mapper: Optional[Callable[..., Mapper[Any]]] = None,
        cls: Type[Any] = object,
        name: str = "Base",
        metaclass: Type[Any] = DeclarativeMeta,
    ) -> Any:
        """Generate a declarative base class.

        Classes that inherit from the returned class object will be
        automatically mapped using declarative mapping.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()

            Base = mapper_registry.generate_base()


            class MyClass(Base):
                __tablename__ = "my_table"
                id = Column(Integer, primary_key=True)

        The above dynamically generated class is equivalent to the
        non-dynamic example below::

            from sqlalchemy.orm import registry
            from sqlalchemy.orm.decl_api import DeclarativeMeta

            mapper_registry = registry()


            class Base(metaclass=DeclarativeMeta):
                __abstract__ = True
                registry = mapper_registry
                metadata = mapper_registry.metadata

                __init__ = mapper_registry.constructor

        .. versionchanged:: 2.0 Note that the
           :meth:`_orm.registry.generate_base` method is superseded by the new
           :class:`_orm.DeclarativeBase` class, which generates a new "base"
           class using subclassing, rather than return value of a function.
           This allows an approach that is compatible with :pep:`484` typing
           tools.

        The :meth:`_orm.registry.generate_base` method provides the
        implementation for the :func:`_orm.declarative_base` function, which
        creates the :class:`_orm.registry` and base class all at once.

        See the section :ref:`orm_declarative_mapping` for background and
        examples.

        :param mapper:
          An optional callable, defaults to :class:`_orm.Mapper`.
          This function is used to generate new :class:`_orm.Mapper` objects.

        :param cls:
          Defaults to :class:`object`. A type to use as the base for the
          generated declarative base class. May be a class or tuple of classes.

        :param name:
          Defaults to ``Base``.  The display name for the generated
          class.  Customizing this is not required, but can improve clarity in
          tracebacks and debugging.

        :param metaclass:
          Defaults to :class:`.DeclarativeMeta`.  A metaclass or __metaclass__
          compatible callable to use as the meta type of the generated
          declarative base class.

        .. seealso::

            :ref:`orm_declarative_mapping`

            :func:`_orm.declarative_base`

        """
        metadata = self.metadata

        bases = not isinstance(cls, tuple) and (cls,) or cls

        class_dict: Dict[str, Any] = dict(registry=self, metadata=metadata)
        if isinstance(cls, type):
            class_dict["__doc__"] = cls.__doc__

        if self.constructor is not None:
            class_dict["__init__"] = self.constructor

        class_dict["__abstract__"] = True
        if mapper:
            class_dict["__mapper_cls__"] = mapper

        if hasattr(cls, "__class_getitem__"):

            def __class_getitem__(cls: Type[_T], key: Any) -> Type[_T]:
                # allow generic classes in py3.9+
                return cls

            class_dict["__class_getitem__"] = __class_getitem__

        return metaclass(name, bases, class_dict)

    @compat_typing.dataclass_transform(
        field_specifiers=(
            MappedColumn,
            RelationshipProperty,
            Composite,
            Synonym,
            mapped_column,
            relationship,
            composite,
            synonym,
            deferred,
        ),
    )
    @overload
    def mapped_as_dataclass(self, __cls: Type[_O]) -> Type[_O]: ...

    @overload
    def mapped_as_dataclass(
        self,
        __cls: Literal[None] = ...,
        *,
        init: Union[_NoArg, bool] = ...,
        repr: Union[_NoArg, bool] = ...,  # noqa: A002
        eq: Union[_NoArg, bool] = ...,
        order: Union[_NoArg, bool] = ...,
        unsafe_hash: Union[_NoArg, bool] = ...,
        match_args: Union[_NoArg, bool] = ...,
        kw_only: Union[_NoArg, bool] = ...,
        dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]] = ...,
    ) -> Callable[[Type[_O]], Type[_O]]: ...

    def mapped_as_dataclass(
        self,
        __cls: Optional[Type[_O]] = None,
        *,
        init: Union[_NoArg, bool] = _NoArg.NO_ARG,
        repr: Union[_NoArg, bool] = _NoArg.NO_ARG,  # noqa: A002
        eq: Union[_NoArg, bool] = _NoArg.NO_ARG,
        order: Union[_NoArg, bool] = _NoArg.NO_ARG,
        unsafe_hash: Union[_NoArg, bool] = _NoArg.NO_ARG,
        match_args: Union[_NoArg, bool] = _NoArg.NO_ARG,
        kw_only: Union[_NoArg, bool] = _NoArg.NO_ARG,
        dataclass_callable: Union[
            _NoArg, Callable[..., Type[Any]]
        ] = _NoArg.NO_ARG,
    ) -> Union[Type[_O], Callable[[Type[_O]], Type[_O]]]:
        """Class decorator that will apply the Declarative mapping process
        to a given class, and additionally convert the class to be a
        Python dataclass.

        .. seealso::

            :ref:`orm_declarative_native_dataclasses` - complete background
            on SQLAlchemy native dataclass mapping


        .. versionadded:: 2.0


        """

        def decorate(cls: Type[_O]) -> Type[_O]:
            setattr(
                cls,
                "_sa_apply_dc_transforms",
                {
                    "init": init,
                    "repr": repr,
                    "eq": eq,
                    "order": order,
                    "unsafe_hash": unsafe_hash,
                    "match_args": match_args,
                    "kw_only": kw_only,
                    "dataclass_callable": dataclass_callable,
                },
            )
            _as_declarative(self, cls, cls.__dict__)
            return cls

        if __cls:
            return decorate(__cls)
        else:
            return decorate

    def mapped(self, cls: Type[_O]) -> Type[_O]:
        """Class decorator that will apply the Declarative mapping process
        to a given class.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()


            @mapper_registry.mapped
            class Foo:
                __tablename__ = "some_table"

                id = Column(Integer, primary_key=True)
                name = Column(String)

        See the section :ref:`orm_declarative_mapping` for complete
        details and examples.

        :param cls: class to be mapped.

        :return: the class that was passed.

        .. seealso::

            :ref:`orm_declarative_mapping`

            :meth:`_orm.registry.generate_base` - generates a base class
            that will apply Declarative mapping to subclasses automatically
            using a Python metaclass.

        .. seealso::

            :meth:`_orm.registry.mapped_as_dataclass`

        """
        _as_declarative(self, cls, cls.__dict__)
        return cls

    def as_declarative_base(self, **kw: Any) -> Callable[[Type[_T]], Type[_T]]:
        """
        Class decorator which will invoke
        :meth:`_orm.registry.generate_base`
        for a given base class.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()


            @mapper_registry.as_declarative_base()
            class Base:
                @declared_attr
                def __tablename__(cls):
                    return cls.__name__.lower()

                id = Column(Integer, primary_key=True)


            class MyMappedClass(Base): ...

        All keyword arguments passed to
        :meth:`_orm.registry.as_declarative_base` are passed
        along to :meth:`_orm.registry.generate_base`.

        """

        def decorate(cls: Type[_T]) -> Type[_T]:
            kw["cls"] = cls
            kw["name"] = cls.__name__
            return self.generate_base(**kw)  # type: ignore

        return decorate

    def map_declaratively(self, cls: Type[_O]) -> Mapper[_O]:
        """Map a class declaratively.

        In this form of mapping, the class is scanned for mapping information,
        including for columns to be associated with a table, and/or an
        actual table object.

        Returns the :class:`_orm.Mapper` object.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()


            class Foo:
                __tablename__ = "some_table"

                id = Column(Integer, primary_key=True)
                name = Column(String)


            mapper = mapper_registry.map_declaratively(Foo)

        This function is more conveniently invoked indirectly via either the
        :meth:`_orm.registry.mapped` class decorator or by subclassing a
        declarative metaclass generated from
        :meth:`_orm.registry.generate_base`.

        See the section :ref:`orm_declarative_mapping` for complete
        details and examples.

        :param cls: class to be mapped.

        :return: a :class:`_orm.Mapper` object.

        .. seealso::

            :ref:`orm_declarative_mapping`

            :meth:`_orm.registry.mapped` - more common decorator interface
            to this function.

            :meth:`_orm.registry.map_imperatively`

        """
        _as_declarative(self, cls, cls.__dict__)
        return cls.__mapper__  # type: ignore

    def map_imperatively(
        self,
        class_: Type[_O],
        local_table: Optional[FromClause] = None,
        **kw: Any,
    ) -> Mapper[_O]:
        r"""Map a class imperatively.

        In this form of mapping, the class is not scanned for any mapping
        information.  Instead, all mapping constructs are passed as
        arguments.

        This method is intended to be fully equivalent to the now-removed
        SQLAlchemy ``mapper()`` function, except that it's in terms of
        a particular registry.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()

            my_table = Table(
                "my_table",
                mapper_registry.metadata,
                Column("id", Integer, primary_key=True),
            )


            class MyClass:
                pass


            mapper_registry.map_imperatively(MyClass, my_table)

        See the section :ref:`orm_imperative_mapping` for complete background
        and usage examples.

        :param class\_: The class to be mapped.  Corresponds to the
         :paramref:`_orm.Mapper.class_` parameter.

        :param local_table: the :class:`_schema.Table` or other
         :class:`_sql.FromClause` object that is the subject of the mapping.
         Corresponds to the
         :paramref:`_orm.Mapper.local_table` parameter.

        :param \**kw: all other keyword arguments are passed to the
         :class:`_orm.Mapper` constructor directly.

        .. seealso::

            :ref:`orm_imperative_mapping`

            :ref:`orm_declarative_mapping`

        """
        return _mapper(self, class_, local_table, kw)


RegistryType = registry

if not TYPE_CHECKING:
    # allow for runtime type resolution of ``ClassVar[_RegistryType]``
    _RegistryType = registry  # noqa


def as_declarative(**kw: Any) -> Callable[[Type[_T]], Type[_T]]:
    """
    Class decorator which will adapt a given class into a
    :func:`_orm.declarative_base`.

    This function makes use of the :meth:`_orm.registry.as_declarative_base`
    method, by first creating a :class:`_orm.registry` automatically
    and then invoking the decorator.

    E.g.::

        from sqlalchemy.orm import as_declarative


        @as_declarative()
        class Base:
            @declared_attr
            def __tablename__(cls):
                return cls.__name__.lower()

            id = Column(Integer, primary_key=True)


        class MyMappedClass(Base): ...

    .. seealso::

        :meth:`_orm.registry.as_declarative_base`

    """
    metadata, class_registry = (
        kw.pop("metadata", None),
        kw.pop("class_registry", None),
    )

    return registry(
        metadata=metadata, class_registry=class_registry
    ).as_declarative_base(**kw)


@inspection._inspects(
    DeclarativeMeta, DeclarativeBase, DeclarativeAttributeIntercept
)
def _inspect_decl_meta(cls: Type[Any]) -> Optional[Mapper[Any]]:
    mp: Optional[Mapper[Any]] = _inspect_mapped_class(cls)
    if mp is None:
        if _DeferredMapperConfig.has_cls(cls):
            _DeferredMapperConfig.raise_unmapped_for_cls(cls)
    return mp