
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\query.py ===
# orm/query.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""The Query class and support.

Defines the :class:`_query.Query` class, the central
construct used by the ORM to construct database queries.

The :class:`_query.Query` class should not be confused with the
:class:`_expression.Select` class, which defines database
SELECT operations at the SQL (non-ORM) level.  ``Query`` differs from
``Select`` in that it returns ORM-mapped objects and interacts with an
ORM session, whereas the ``Select`` construct interacts directly with the
database to return iterable result sets.

"""
from __future__ import annotations

import collections.abc as collections_abc
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import attributes
from . import interfaces
from . import loading
from . import util as orm_util
from ._typing import _O
from .base import _assertions
from .context import _column_descriptions
from .context import _determine_last_joined_entity
from .context import _legacy_filter_by_entity_zero
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .util import AliasedClass
from .util import object_mapper
from .util import with_parent
from .. import exc as sa_exc
from .. import inspect
from .. import inspection
from .. import log
from .. import sql
from .. import util
from ..engine import Result
from ..engine import Row
from ..event import dispatcher
from ..event import EventTarget
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import Select
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _FromClauseArgument
from ..sql._typing import _TP
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import _entity_namespace_key
from ..sql.base import _generative
from ..sql.base import _NoArg
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.elements import BooleanClauseList
from ..sql.expression import Exists
from ..sql.selectable import _MemoizedSelectEntities
from ..sql.selectable import _SelectFromElements
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import HasHints
from ..sql.selectable import HasPrefixes
from ..sql.selectable import HasSuffixes
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectLabelStyle
from ..util.typing import Literal
from ..util.typing import Self


if TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _ExternalEntityType
    from ._typing import _InternalEntityType
    from ._typing import SynchronizeSessionArgument
    from .mapper import Mapper
    from .path_registry import PathRegistry
    from .session import _PKIdentityArgument
    from .session import Session
    from .state import InstanceState
    from ..engine.cursor import CursorResult
    from ..engine.interfaces import _ImmutableExecuteOptions
    from ..engine.interfaces import CompiledCacheType
    from ..engine.interfaces import IsolationLevel
    from ..engine.interfaces import SchemaTranslateMapType
    from ..engine.result import FrozenResult
    from ..engine.result import ScalarResult
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _ColumnExpressionOrStrLabelArgument
    from ..sql._typing import _ColumnsClauseArgument
    from ..sql._typing import _DMLColumnArgument
    from ..sql._typing import _JoinTargetArgument
    from ..sql._typing import _LimitOffsetType
    from ..sql._typing import _MAYBE_ENTITY
    from ..sql._typing import _no_kw
    from ..sql._typing import _NOT_ENTITY
    from ..sql._typing import _OnClauseArgument
    from ..sql._typing import _PropagateAttrsType
    from ..sql._typing import _T0
    from ..sql._typing import _T1
    from ..sql._typing import _T2
    from ..sql._typing import _T3
    from ..sql._typing import _T4
    from ..sql._typing import _T5
    from ..sql._typing import _T6
    from ..sql._typing import _T7
    from ..sql._typing import _TypedColumnClauseArgument as _TCCA
    from ..sql.base import CacheableOptions
    from ..sql.base import ExecutableOption
    from ..sql.dml import UpdateBase
    from ..sql.elements import ColumnElement
    from ..sql.elements import Label
    from ..sql.selectable import _ForUpdateOfArgument
    from ..sql.selectable import _JoinTargetElement
    from ..sql.selectable import _SetupJoinsElement
    from ..sql.selectable import Alias
    from ..sql.selectable import CTE
    from ..sql.selectable import ExecutableReturnsRows
    from ..sql.selectable import FromClause
    from ..sql.selectable import ScalarSelect
    from ..sql.selectable import Subquery


__all__ = ["Query", "QueryContext"]

_T = TypeVar("_T", bound=Any)


@inspection._self_inspects
@log.class_logger
class Query(
    _SelectFromElements,
    SupportsCloneAnnotations,
    HasPrefixes,
    HasSuffixes,
    HasHints,
    EventTarget,
    log.Identified,
    Generative,
    Executable,
    Generic[_T],
):
    """ORM-level SQL construction object.

    .. legacy:: The ORM :class:`.Query` object is a legacy construct
       as of SQLAlchemy 2.0.   See the notes at the top of
       :ref:`query_api_toplevel` for an overview, including links to migration
       documentation.

    :class:`_query.Query` objects are normally initially generated using the
    :meth:`~.Session.query` method of :class:`.Session`, and in
    less common cases by instantiating the :class:`_query.Query` directly and
    associating with a :class:`.Session` using the
    :meth:`_query.Query.with_session`
    method.

    """

    # elements that are in Core and can be cached in the same way
    _where_criteria: Tuple[ColumnElement[Any], ...] = ()
    _having_criteria: Tuple[ColumnElement[Any], ...] = ()

    _order_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _group_by_clauses: Tuple[ColumnElement[Any], ...] = ()
    _limit_clause: Optional[ColumnElement[Any]] = None
    _offset_clause: Optional[ColumnElement[Any]] = None

    _distinct: bool = False
    _distinct_on: Tuple[ColumnElement[Any], ...] = ()

    _for_update_arg: Optional[ForUpdateArg] = None
    _correlate: Tuple[FromClause, ...] = ()
    _auto_correlate: bool = True
    _from_obj: Tuple[FromClause, ...] = ()
    _setup_joins: Tuple[_SetupJoinsElement, ...] = ()

    _label_style: SelectLabelStyle = SelectLabelStyle.LABEL_STYLE_LEGACY_ORM

    _memoized_select_entities = ()

    _compile_options: Union[Type[CacheableOptions], CacheableOptions] = (
        ORMCompileState.default_compile_options
    )

    _with_options: Tuple[ExecutableOption, ...]
    load_options = QueryContext.default_load_options + {
        "_legacy_uniquing": True
    }

    _params: util.immutabledict[str, Any] = util.EMPTY_DICT

    # local Query builder state, not needed for
    # compilation or execution
    _enable_assertions = True

    _statement: Optional[ExecutableReturnsRows] = None

    session: Session

    dispatch: dispatcher[Query[_T]]

    # mirrors that of ClauseElement, used to propagate the "orm"
    # plugin as well as the "subject" of the plugin, e.g. the mapper
    # we are querying against.
    @util.memoized_property
    def _propagate_attrs(self) -> _PropagateAttrsType:
        return util.EMPTY_DICT

    def __init__(
        self,
        entities: Union[
            _ColumnsClauseArgument[Any], Sequence[_ColumnsClauseArgument[Any]]
        ],
        session: Optional[Session] = None,
    ):
        """Construct a :class:`_query.Query` directly.

        E.g.::

            q = Query([User, Address], session=some_session)

        The above is equivalent to::

            q = some_session.query(User, Address)

        :param entities: a sequence of entities and/or SQL expressions.

        :param session: a :class:`.Session` with which the
         :class:`_query.Query`
         will be associated.   Optional; a :class:`_query.Query`
         can be associated
         with a :class:`.Session` generatively via the
         :meth:`_query.Query.with_session` method as well.

        .. seealso::

            :meth:`.Session.query`

            :meth:`_query.Query.with_session`

        """

        # session is usually present.  There's one case in subqueryloader
        # where it stores a Query without a Session and also there are tests
        # for the query(Entity).with_session(session) API which is likely in
        # some old recipes, however these are legacy as select() can now be
        # used.
        self.session = session  # type: ignore
        self._set_entities(entities)

    def _set_propagate_attrs(self, values: Mapping[str, Any]) -> Self:
        self._propagate_attrs = util.immutabledict(values)
        return self

    def _set_entities(
        self,
        entities: Union[
            _ColumnsClauseArgument[Any], Iterable[_ColumnsClauseArgument[Any]]
        ],
    ) -> None:
        self._raw_columns = [
            coercions.expect(
                roles.ColumnsClauseRole,
                ent,
                apply_propagate_attrs=self,
                post_inspect=True,
            )
            for ent in util.to_list(entities)
        ]

    def tuples(self: Query[_O]) -> Query[Tuple[_O]]:
        """return a tuple-typed form of this :class:`.Query`.

        This method invokes the :meth:`.Query.only_return_tuples`
        method with a value of ``True``, which by itself ensures that this
        :class:`.Query` will always return :class:`.Row` objects, even
        if the query is made against a single entity.  It then also
        at the typing level will return a "typed" query, if possible,
        that will type result rows as ``Tuple`` objects with typed
        elements.

        This method can be compared to the :meth:`.Result.tuples` method,
        which returns "self", but from a typing perspective returns an object
        that will yield typed ``Tuple`` objects for results.   Typing
        takes effect only if this :class:`.Query` object is a typed
        query object already.

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.Result.tuples` - v2 equivalent method.

        """
        return self.only_return_tuples(True)  # type: ignore

    def _entity_from_pre_ent_zero(self) -> Optional[_InternalEntityType[Any]]:
        if not self._raw_columns:
            return None

        ent = self._raw_columns[0]

        if "parententity" in ent._annotations:
            return ent._annotations["parententity"]  # type: ignore
        elif "bundle" in ent._annotations:
            return ent._annotations["bundle"]  # type: ignore
        else:
            # label, other SQL expression
            for element in visitors.iterate(ent):
                if "parententity" in element._annotations:
                    return element._annotations["parententity"]  # type: ignore  # noqa: E501
            else:
                return None

    def _only_full_mapper_zero(self, methname: str) -> Mapper[Any]:
        if (
            len(self._raw_columns) != 1
            or "parententity" not in self._raw_columns[0]._annotations
            or not self._raw_columns[0].is_selectable
        ):
            raise sa_exc.InvalidRequestError(
                "%s() can only be used against "
                "a single mapped class." % methname
            )

        return self._raw_columns[0]._annotations["parententity"]  # type: ignore  # noqa: E501

    def _set_select_from(
        self, obj: Iterable[_FromClauseArgument], set_base_alias: bool
    ) -> None:
        fa = [
            coercions.expect(
                roles.StrictFromClauseRole,
                elem,
                allow_select=True,
                apply_propagate_attrs=self,
            )
            for elem in obj
        ]

        self._compile_options += {"_set_base_alias": set_base_alias}
        self._from_obj = tuple(fa)

    @_generative
    def _set_lazyload_from(self, state: InstanceState[Any]) -> Self:
        self.load_options += {"_lazy_loaded_from": state}
        return self

    def _get_condition(self) -> None:
        """used by legacy BakedQuery"""
        self._no_criterion_condition("get", order_by=False, distinct=False)

    def _get_existing_condition(self) -> None:
        self._no_criterion_assertion("get", order_by=False, distinct=False)

    def _no_criterion_assertion(
        self, meth: str, order_by: bool = True, distinct: bool = True
    ) -> None:
        if not self._enable_assertions:
            return
        if (
            self._where_criteria
            or self._statement is not None
            or self._from_obj
            or self._setup_joins
            or self._limit_clause is not None
            or self._offset_clause is not None
            or self._group_by_clauses
            or (order_by and self._order_by_clauses)
            or (distinct and self._distinct)
        ):
            raise sa_exc.InvalidRequestError(
                "Query.%s() being called on a "
                "Query with existing criterion. " % meth
            )

    def _no_criterion_condition(
        self, meth: str, order_by: bool = True, distinct: bool = True
    ) -> None:
        self._no_criterion_assertion(meth, order_by, distinct)

        self._from_obj = self._setup_joins = ()
        if self._statement is not None:
            self._compile_options += {"_statement": None}
        self._where_criteria = ()
        self._distinct = False

        self._order_by_clauses = self._group_by_clauses = ()

    def _no_clauseelement_condition(self, meth: str) -> None:
        if not self._enable_assertions:
            return
        if self._order_by_clauses:
            raise sa_exc.InvalidRequestError(
                "Query.%s() being called on a "
                "Query with existing criterion. " % meth
            )
        self._no_criterion_condition(meth)

    def _no_statement_condition(self, meth: str) -> None:
        if not self._enable_assertions:
            return
        if self._statement is not None:
            raise sa_exc.InvalidRequestError(
                (
                    "Query.%s() being called on a Query with an existing full "
                    "statement - can't apply criterion."
                )
                % meth
            )

    def _no_limit_offset(self, meth: str) -> None:
        if not self._enable_assertions:
            return
        if self._limit_clause is not None or self._offset_clause is not None:
            raise sa_exc.InvalidRequestError(
                "Query.%s() being called on a Query which already has LIMIT "
                "or OFFSET applied.  Call %s() before limit() or offset() "
                "are applied." % (meth, meth)
            )

    @property
    def _has_row_limiting_clause(self) -> bool:
        return (
            self._limit_clause is not None or self._offset_clause is not None
        )

    def _get_options(
        self,
        populate_existing: Optional[bool] = None,
        version_check: Optional[bool] = None,
        only_load_props: Optional[Sequence[str]] = None,
        refresh_state: Optional[InstanceState[Any]] = None,
        identity_token: Optional[Any] = None,
    ) -> Self:
        load_options: Dict[str, Any] = {}
        compile_options: Dict[str, Any] = {}

        if version_check:
            load_options["_version_check"] = version_check
        if populate_existing:
            load_options["_populate_existing"] = populate_existing
        if refresh_state:
            load_options["_refresh_state"] = refresh_state
            compile_options["_for_refresh_state"] = True
        if only_load_props:
            compile_options["_only_load_props"] = frozenset(only_load_props)
        if identity_token:
            load_options["_identity_token"] = identity_token

        if load_options:
            self.load_options += load_options
        if compile_options:
            self._compile_options += compile_options

        return self

    def _clone(self, **kw: Any) -> Self:
        return self._generate()

    def _get_select_statement_only(self) -> Select[_T]:
        if self._statement is not None:
            raise sa_exc.InvalidRequestError(
                "Can't call this method on a Query that uses from_statement()"
            )
        return cast("Select[_T]", self.statement)

    @property
    def statement(self) -> Union[Select[_T], FromStatement[_T], UpdateBase]:
        """The full SELECT statement represented by this Query.

        The statement by default will not have disambiguating labels
        applied to the construct unless with_labels(True) is called
        first.

        """

        # .statement can return the direct future.Select() construct here, as
        # long as we are not using subsequent adaption features that
        # are made against raw entities, e.g. from_self(), with_polymorphic(),
        # select_entity_from().  If these features are being used, then
        # the Select() we return will not have the correct .selected_columns
        # collection and will not embed in subsequent queries correctly.
        # We could find a way to make this collection "correct", however
        # this would not be too different from doing the full compile as
        # we are doing in any case, the Select() would still not have the
        # proper state for other attributes like whereclause, order_by,
        # and these features are all deprecated in any case.
        #
        # for these reasons, Query is not a Select, it remains an ORM
        # object for which __clause_element__() must be called in order for
        # it to provide a real expression object.
        #
        # from there, it starts to look much like Query itself won't be
        # passed into the execute process and won't generate its own cache
        # key; this will all occur in terms of the ORM-enabled Select.
        stmt: Union[Select[_T], FromStatement[_T], UpdateBase]

        if not self._compile_options._set_base_alias:
            # if we don't have legacy top level aliasing features in use
            # then convert to a future select() directly
            stmt = self._statement_20(for_statement=True)
        else:
            stmt = self._compile_state(for_statement=True).statement

        if self._params:
            stmt = stmt.params(self._params)

        return stmt

    def _final_statement(self, legacy_query_style: bool = True) -> Select[Any]:
        """Return the 'final' SELECT statement for this :class:`.Query`.

        This is used by the testing suite only and is fairly inefficient.

        This is the Core-only select() that will be rendered by a complete
        compilation of this query, and is what .statement used to return
        in 1.3.


        """

        q = self._clone()

        return q._compile_state(
            use_legacy_query_style=legacy_query_style
        ).statement  # type: ignore

    def _statement_20(
        self, for_statement: bool = False, use_legacy_query_style: bool = True
    ) -> Union[Select[_T], FromStatement[_T]]:
        # TODO: this event needs to be deprecated, as it currently applies
        # only to ORM query and occurs at this spot that is now more
        # or less an artificial spot
        if self.dispatch.before_compile:
            for fn in self.dispatch.before_compile:
                new_query = fn(self)
                if new_query is not None and new_query is not self:
                    self = new_query
                    if not fn._bake_ok:  # type: ignore
                        self._compile_options += {"_bake_ok": False}

        compile_options = self._compile_options
        compile_options += {
            "_for_statement": for_statement,
            "_use_legacy_query_style": use_legacy_query_style,
        }

        stmt: Union[Select[_T], FromStatement[_T]]

        if self._statement is not None:
            stmt = FromStatement(self._raw_columns, self._statement)
            stmt.__dict__.update(
                _with_options=self._with_options,
                _with_context_options=self._with_context_options,
                _compile_options=compile_options,
                _execution_options=self._execution_options,
                _propagate_attrs=self._propagate_attrs,
            )
        else:
            # Query / select() internal attributes are 99% cross-compatible
            stmt = Select._create_raw_select(**self.__dict__)
            stmt.__dict__.update(
                _label_style=self._label_style,
                _compile_options=compile_options,
                _propagate_attrs=self._propagate_attrs,
            )
            stmt.__dict__.pop("session", None)

        # ensure the ORM context is used to compile the statement, even
        # if it has no ORM entities.  This is so ORM-only things like
        # _legacy_joins are picked up that wouldn't be picked up by the
        # Core statement context
        if "compile_state_plugin" not in stmt._propagate_attrs:
            stmt._propagate_attrs = stmt._propagate_attrs.union(
                {"compile_state_plugin": "orm", "plugin_subject": None}
            )

        return stmt

    def subquery(
        self,
        name: Optional[str] = None,
        with_labels: bool = False,
        reduce_columns: bool = False,
    ) -> Subquery:
        """Return the full SELECT statement represented by
        this :class:`_query.Query`, embedded within an
        :class:`_expression.Alias`.

        Eager JOIN generation within the query is disabled.

        .. seealso::

            :meth:`_sql.Select.subquery` - v2 comparable method.

        :param name: string name to be assigned as the alias;
            this is passed through to :meth:`_expression.FromClause.alias`.
            If ``None``, a name will be deterministically generated
            at compile time.

        :param with_labels: if True, :meth:`.with_labels` will be called
         on the :class:`_query.Query` first to apply table-qualified labels
         to all columns.

        :param reduce_columns: if True,
         :meth:`_expression.Select.reduce_columns` will
         be called on the resulting :func:`_expression.select` construct,
         to remove same-named columns where one also refers to the other
         via foreign key or WHERE clause equivalence.

        """
        q = self.enable_eagerloads(False)
        if with_labels:
            q = q.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

        stmt = q._get_select_statement_only()

        if TYPE_CHECKING:
            assert isinstance(stmt, Select)

        if reduce_columns:
            stmt = stmt.reduce_columns()
        return stmt.subquery(name=name)

    def cte(
        self,
        name: Optional[str] = None,
        recursive: bool = False,
        nesting: bool = False,
    ) -> CTE:
        r"""Return the full SELECT statement represented by this
        :class:`_query.Query` represented as a common table expression (CTE).

        Parameters and usage are the same as those of the
        :meth:`_expression.SelectBase.cte` method; see that method for
        further details.

        Here is the `PostgreSQL WITH
        RECURSIVE example
        <https://www.postgresql.org/docs/current/static/queries-with.html>`_.
        Note that, in this example, the ``included_parts`` cte and the
        ``incl_alias`` alias of it are Core selectables, which
        means the columns are accessed via the ``.c.`` attribute.  The
        ``parts_alias`` object is an :func:`_orm.aliased` instance of the
        ``Part`` entity, so column-mapped attributes are available
        directly::

            from sqlalchemy.orm import aliased


            class Part(Base):
                __tablename__ = "part"
                part = Column(String, primary_key=True)
                sub_part = Column(String, primary_key=True)
                quantity = Column(Integer)


            included_parts = (
                session.query(Part.sub_part, Part.part, Part.quantity)
                .filter(Part.part == "our part")
                .cte(name="included_parts", recursive=True)
            )

            incl_alias = aliased(included_parts, name="pr")
            parts_alias = aliased(Part, name="p")
            included_parts = included_parts.union_all(
                session.query(
                    parts_alias.sub_part, parts_alias.part, parts_alias.quantity
                ).filter(parts_alias.part == incl_alias.c.sub_part)
            )

            q = session.query(
                included_parts.c.sub_part,
                func.sum(included_parts.c.quantity).label("total_quantity"),
            ).group_by(included_parts.c.sub_part)

        .. seealso::

            :meth:`_sql.Select.cte` - v2 equivalent method.

        """  # noqa: E501
        return (
            self.enable_eagerloads(False)
            ._get_select_statement_only()
            .cte(name=name, recursive=recursive, nesting=nesting)
        )

    def label(self, name: Optional[str]) -> Label[Any]:
        """Return the full SELECT statement represented by this
        :class:`_query.Query`, converted
        to a scalar subquery with a label of the given name.

        .. seealso::

            :meth:`_sql.Select.label` - v2 comparable method.

        """

        return (
            self.enable_eagerloads(False)
            ._get_select_statement_only()
            .label(name)
        )

    @overload
    def as_scalar(  # type: ignore[overload-overlap]
        self: Query[Tuple[_MAYBE_ENTITY]],
    ) -> ScalarSelect[_MAYBE_ENTITY]: ...

    @overload
    def as_scalar(
        self: Query[Tuple[_NOT_ENTITY]],
    ) -> ScalarSelect[_NOT_ENTITY]: ...

    @overload
    def as_scalar(self) -> ScalarSelect[Any]: ...

    @util.deprecated(
        "1.4",
        "The :meth:`_query.Query.as_scalar` method is deprecated and will be "
        "removed in a future release.  Please refer to "
        ":meth:`_query.Query.scalar_subquery`.",
    )
    def as_scalar(self) -> ScalarSelect[Any]:
        """Return the full SELECT statement represented by this
        :class:`_query.Query`, converted to a scalar subquery.

        """
        return self.scalar_subquery()

    @overload
    def scalar_subquery(
        self: Query[Tuple[_MAYBE_ENTITY]],
    ) -> ScalarSelect[Any]: ...

    @overload
    def scalar_subquery(
        self: Query[Tuple[_NOT_ENTITY]],
    ) -> ScalarSelect[_NOT_ENTITY]: ...

    @overload
    def scalar_subquery(self) -> ScalarSelect[Any]: ...

    def scalar_subquery(self) -> ScalarSelect[Any]:
        """Return the full SELECT statement represented by this
        :class:`_query.Query`, converted to a scalar subquery.

        Analogous to
        :meth:`sqlalchemy.sql.expression.SelectBase.scalar_subquery`.

        .. versionchanged:: 1.4 The :meth:`_query.Query.scalar_subquery`
           method replaces the :meth:`_query.Query.as_scalar` method.

        .. seealso::

            :meth:`_sql.Select.scalar_subquery` - v2 comparable method.

        """

        return (
            self.enable_eagerloads(False)
            ._get_select_statement_only()
            .scalar_subquery()
        )

    @property
    def selectable(self) -> Union[Select[_T], FromStatement[_T], UpdateBase]:
        """Return the :class:`_expression.Select` object emitted by this
        :class:`_query.Query`.

        Used for :func:`_sa.inspect` compatibility, this is equivalent to::

            query.enable_eagerloads(False).with_labels().statement

        """
        return self.__clause_element__()

    def __clause_element__(
        self,
    ) -> Union[Select[_T], FromStatement[_T], UpdateBase]:
        return (
            self._with_compile_options(
                _enable_eagerloads=False, _render_for_subquery=True
            )
            .set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
            .statement
        )

    @overload
    def only_return_tuples(
        self: Query[_O], value: Literal[True]
    ) -> RowReturningQuery[Tuple[_O]]: ...

    @overload
    def only_return_tuples(
        self: Query[_O], value: Literal[False]
    ) -> Query[_O]: ...

    @_generative
    def only_return_tuples(self, value: bool) -> Query[Any]:
        """When set to True, the query results will always be a
        :class:`.Row` object.

        This can change a query that normally returns a single entity
        as a scalar to return a :class:`.Row` result in all cases.

        .. seealso::

            :meth:`.Query.tuples` - returns tuples, but also at the typing
            level will type results as ``Tuple``.

            :meth:`_query.Query.is_single_entity`

            :meth:`_engine.Result.tuples` - v2 comparable method.

        """
        self.load_options += dict(_only_return_tuples=value)
        return self

    @property
    def is_single_entity(self) -> bool:
        """Indicates if this :class:`_query.Query`
        returns tuples or single entities.

        Returns True if this query returns a single entity for each instance
        in its result list, and False if this query returns a tuple of entities
        for each result.

        .. versionadded:: 1.3.11

        .. seealso::

            :meth:`_query.Query.only_return_tuples`

        """
        return (
            not self.load_options._only_return_tuples
            and len(self._raw_columns) == 1
            and "parententity" in self._raw_columns[0]._annotations
            and isinstance(
                self._raw_columns[0]._annotations["parententity"],
                ORMColumnsClauseRole,
            )
        )

    @_generative
    def enable_eagerloads(self, value: bool) -> Self:
        """Control whether or not eager joins and subqueries are
        rendered.

        When set to False, the returned Query will not render
        eager joins regardless of :func:`~sqlalchemy.orm.joinedload`,
        :func:`~sqlalchemy.orm.subqueryload` options
        or mapper-level ``lazy='joined'``/``lazy='subquery'``
        configurations.

        This is used primarily when nesting the Query's
        statement into a subquery or other
        selectable, or when using :meth:`_query.Query.yield_per`.

        """
        self._compile_options += {"_enable_eagerloads": value}
        return self

    @_generative
    def _with_compile_options(self, **opt: Any) -> Self:
        self._compile_options += opt
        return self

    @util.became_legacy_20(
        ":meth:`_orm.Query.with_labels` and :meth:`_orm.Query.apply_labels`",
        alternative="Use set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL) "
        "instead.",
    )
    def with_labels(self) -> Self:
        return self.set_label_style(
            SelectLabelStyle.LABEL_STYLE_TABLENAME_PLUS_COL
        )

    apply_labels = with_labels

    @property
    def get_label_style(self) -> SelectLabelStyle:
        """
        Retrieve the current label style.

        .. versionadded:: 1.4

        .. seealso::

            :meth:`_sql.Select.get_label_style` - v2 equivalent method.

        """
        return self._label_style

    def set_label_style(self, style: SelectLabelStyle) -> Self:
        """Apply column labels to the return value of Query.statement.

        Indicates that this Query's `statement` accessor should return
        a SELECT statement that applies labels to all columns in the
        form <tablename>_<columnname>; this is commonly used to
        disambiguate columns from multiple tables which have the same
        name.

        When the `Query` actually issues SQL to load rows, it always
        uses column labeling.

        .. note:: The :meth:`_query.Query.set_label_style` method *only* applies
           the output of :attr:`_query.Query.statement`, and *not* to any of
           the result-row invoking systems of :class:`_query.Query` itself,
           e.g.
           :meth:`_query.Query.first`, :meth:`_query.Query.all`, etc.
           To execute
           a query using :meth:`_query.Query.set_label_style`, invoke the
           :attr:`_query.Query.statement` using :meth:`.Session.execute`::

                result = session.execute(
                    query.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).statement
                )

        .. versionadded:: 1.4


        .. seealso::

            :meth:`_sql.Select.set_label_style` - v2 equivalent method.

        """  # noqa
        if self._label_style is not style:
            self = self._generate()
            self._label_style = style
        return self

    @_generative
    def enable_assertions(self, value: bool) -> Self:
        """Control whether assertions are generated.

        When set to False, the returned Query will
        not assert its state before certain operations,
        including that LIMIT/OFFSET has not been applied
        when filter() is called, no criterion exists
        when get() is called, and no "from_statement()"
        exists when filter()/order_by()/group_by() etc.
        is called.  This more permissive mode is used by
        custom Query subclasses to specify criterion or
        other modifiers outside of the usual usage patterns.

        Care should be taken to ensure that the usage
        pattern is even possible.  A statement applied
        by from_statement() will override any criterion
        set by filter() or order_by(), for example.

        """
        self._enable_assertions = value
        return self

    @property
    def whereclause(self) -> Optional[ColumnElement[bool]]:
        """A readonly attribute which returns the current WHERE criterion for
        this Query.

        This returned value is a SQL expression construct, or ``None`` if no
        criterion has been established.

        .. seealso::

            :attr:`_sql.Select.whereclause` - v2 equivalent property.

        """
        return BooleanClauseList._construct_for_whereclause(
            self._where_criteria
        )

    @_generative
    def _with_current_path(self, path: PathRegistry) -> Self:
        """indicate that this query applies to objects loaded
        within a certain path.

        Used by deferred loaders (see strategies.py) which transfer
        query options from an originating query to a newly generated
        query intended for the deferred load.

        """
        self._compile_options += {"_current_path": path}
        return self

    @_generative
    def yield_per(self, count: int) -> Self:
        r"""Yield only ``count`` rows at a time.

        The purpose of this method is when fetching very large result sets
        (> 10K rows), to batch results in sub-collections and yield them
        out partially, so that the Python interpreter doesn't need to declare
        very large areas of memory which is both time consuming and leads
        to excessive memory use.   The performance from fetching hundreds of
        thousands of rows can often double when a suitable yield-per setting
        (e.g. approximately 1000) is used, even with DBAPIs that buffer
        rows (which are most).

        As of SQLAlchemy 1.4, the :meth:`_orm.Query.yield_per` method is
        equivalent to using the ``yield_per`` execution option at the ORM
        level. See the section :ref:`orm_queryguide_yield_per` for further
        background on this option.

        .. seealso::

            :ref:`orm_queryguide_yield_per`

        """
        self.load_options += {"_yield_per": count}
        return self

    @util.became_legacy_20(
        ":meth:`_orm.Query.get`",
        alternative="The method is now available as :meth:`_orm.Session.get`",
    )
    def get(self, ident: _PKIdentityArgument) -> Optional[Any]:
        """Return an instance based on the given primary key identifier,
        or ``None`` if not found.

        E.g.::

            my_user = session.query(User).get(5)

            some_object = session.query(VersionedFoo).get((5, 10))

            some_object = session.query(VersionedFoo).get({"id": 5, "version_id": 10})

        :meth:`_query.Query.get` is special in that it provides direct
        access to the identity map of the owning :class:`.Session`.
        If the given primary key identifier is present
        in the local identity map, the object is returned
        directly from this collection and no SQL is emitted,
        unless the object has been marked fully expired.
        If not present,
        a SELECT is performed in order to locate the object.

        :meth:`_query.Query.get` also will perform a check if
        the object is present in the identity map and
        marked as expired - a SELECT
        is emitted to refresh the object as well as to
        ensure that the row is still present.
        If not, :class:`~sqlalchemy.orm.exc.ObjectDeletedError` is raised.

        :meth:`_query.Query.get` is only used to return a single
        mapped instance, not multiple instances or
        individual column constructs, and strictly
        on a single primary key value.  The originating
        :class:`_query.Query` must be constructed in this way,
        i.e. against a single mapped entity,
        with no additional filtering criterion.  Loading
        options via :meth:`_query.Query.options` may be applied
        however, and will be used if the object is not
        yet locally present.

        :param ident: A scalar, tuple, or dictionary representing the
         primary key.  For a composite (e.g. multiple column) primary key,
         a tuple or dictionary should be passed.

         For a single-column primary key, the scalar calling form is typically
         the most expedient.  If the primary key of a row is the value "5",
         the call looks like::

            my_object = query.get(5)

         The tuple form contains primary key values typically in
         the order in which they correspond to the mapped
         :class:`_schema.Table`
         object's primary key columns, or if the
         :paramref:`_orm.Mapper.primary_key` configuration parameter were
         used, in
         the order used for that parameter. For example, if the primary key
         of a row is represented by the integer
         digits "5, 10" the call would look like::

             my_object = query.get((5, 10))

         The dictionary form should include as keys the mapped attribute names
         corresponding to each element of the primary key.  If the mapped class
         has the attributes ``id``, ``version_id`` as the attributes which
         store the object's primary key value, the call would look like::

            my_object = query.get({"id": 5, "version_id": 10})

         .. versionadded:: 1.3 the :meth:`_query.Query.get`
            method now optionally
            accepts a dictionary of attribute names to values in order to
            indicate a primary key identifier.


        :return: The object instance, or ``None``.

        """  # noqa: E501
        self._no_criterion_assertion("get", order_by=False, distinct=False)

        # we still implement _get_impl() so that baked query can override
        # it
        return self._get_impl(ident, loading.load_on_pk_identity)

    def _get_impl(
        self,
        primary_key_identity: _PKIdentityArgument,
        db_load_fn: Callable[..., Any],
        identity_token: Optional[Any] = None,
    ) -> Optional[Any]:
        mapper = self._only_full_mapper_zero("get")
        return self.session._get_impl(
            mapper,
            primary_key_identity,
            db_load_fn,
            populate_existing=self.load_options._populate_existing,
            with_for_update=self._for_update_arg,
            options=self._with_options,
            identity_token=identity_token,
            execution_options=self._execution_options,
        )

    @property
    def lazy_loaded_from(self) -> Optional[InstanceState[Any]]:
        """An :class:`.InstanceState` that is using this :class:`_query.Query`
        for a lazy load operation.

        .. deprecated:: 1.4  This attribute should be viewed via the
           :attr:`.ORMExecuteState.lazy_loaded_from` attribute, within
           the context of the :meth:`.SessionEvents.do_orm_execute`
           event.

        .. seealso::

            :attr:`.ORMExecuteState.lazy_loaded_from`

        """
        return self.load_options._lazy_loaded_from  # type: ignore

    @property
    def _current_path(self) -> PathRegistry:
        return self._compile_options._current_path  # type: ignore

    @_generative
    def correlate(
        self,
        *fromclauses: Union[Literal[None, False], _FromClauseArgument],
    ) -> Self:
        """Return a :class:`.Query` construct which will correlate the given
        FROM clauses to that of an enclosing :class:`.Query` or
        :func:`~.expression.select`.

        The method here accepts mapped classes, :func:`.aliased` constructs,
        and :class:`_orm.Mapper` constructs as arguments, which are resolved
        into expression constructs, in addition to appropriate expression
        constructs.

        The correlation arguments are ultimately passed to
        :meth:`_expression.Select.correlate`
        after coercion to expression constructs.

        The correlation arguments take effect in such cases
        as when :meth:`_query.Query.from_self` is used, or when
        a subquery as returned by :meth:`_query.Query.subquery` is
        embedded in another :func:`_expression.select` construct.

        .. seealso::

            :meth:`_sql.Select.correlate` - v2 equivalent method.

        """

        self._auto_correlate = False
        if fromclauses and fromclauses[0] in {None, False}:
            self._correlate = ()
        else:
            self._correlate = self._correlate + tuple(
                coercions.expect(roles.FromClauseRole, f) for f in fromclauses
            )
        return self

    @_generative
    def autoflush(self, setting: bool) -> Self:
        """Return a Query with a specific 'autoflush' setting.

        As of SQLAlchemy 1.4, the :meth:`_orm.Query.autoflush` method
        is equivalent to using the ``autoflush`` execution option at the
        ORM level. See the section :ref:`orm_queryguide_autoflush` for
        further background on this option.

        """
        self.load_options += {"_autoflush": setting}
        return self

    @_generative
    def populate_existing(self) -> Self:
        """Return a :class:`_query.Query`
        that will expire and refresh all instances
        as they are loaded, or reused from the current :class:`.Session`.

        As of SQLAlchemy 1.4, the :meth:`_orm.Query.populate_existing` method
        is equivalent to using the ``populate_existing`` execution option at
        the ORM level. See the section :ref:`orm_queryguide_populate_existing`
        for further background on this option.

        """
        self.load_options += {"_populate_existing": True}
        return self

    @_generative
    def _with_invoke_all_eagers(self, value: bool) -> Self:
        """Set the 'invoke all eagers' flag which causes joined- and
        subquery loaders to traverse into already-loaded related objects
        and collections.

        Default is that of :attr:`_query.Query._invoke_all_eagers`.

        """
        self.load_options += {"_invoke_all_eagers": value}
        return self

    @util.became_legacy_20(
        ":meth:`_orm.Query.with_parent`",
        alternative="Use the :func:`_orm.with_parent` standalone construct.",
    )
    @util.preload_module("sqlalchemy.orm.relationships")
    def with_parent(
        self,
        instance: object,
        property: Optional[  # noqa: A002
            attributes.QueryableAttribute[Any]
        ] = None,
        from_entity: Optional[_ExternalEntityType[Any]] = None,
    ) -> Self:
        """Add filtering criterion that relates the given instance
        to a child object or collection, using its attribute state
        as well as an established :func:`_orm.relationship()`
        configuration.

        The method uses the :func:`.with_parent` function to generate
        the clause, the result of which is passed to
        :meth:`_query.Query.filter`.

        Parameters are the same as :func:`.with_parent`, with the exception
        that the given property can be None, in which case a search is
        performed against this :class:`_query.Query` object's target mapper.

        :param instance:
          An instance which has some :func:`_orm.relationship`.

        :param property:
          Class bound attribute which indicates
          what relationship from the instance should be used to reconcile the
          parent/child relationship.

        :param from_entity:
          Entity in which to consider as the left side.  This defaults to the
          "zero" entity of the :class:`_query.Query` itself.

        """
        relationships = util.preloaded.orm_relationships

        if from_entity:
            entity_zero = inspect(from_entity)
        else:
            entity_zero = _legacy_filter_by_entity_zero(self)
        if property is None:
            # TODO: deprecate, property has to be supplied
            mapper = object_mapper(instance)

            for prop in mapper.iterate_properties:
                if (
                    isinstance(prop, relationships.RelationshipProperty)
                    and prop.mapper is entity_zero.mapper  # type: ignore
                ):
                    property = prop  # type: ignore  # noqa: A001
                    break
            else:
                raise sa_exc.InvalidRequestError(
                    "Could not locate a property which relates instances "
                    "of class '%s' to instances of class '%s'"
                    % (
                        entity_zero.mapper.class_.__name__,  # type: ignore
                        instance.__class__.__name__,
                    )
                )

        return self.filter(
            with_parent(
                instance,
                property,  # type: ignore
                entity_zero.entity,  # type: ignore
            )
        )

    @_generative
    def add_entity(
        self,
        entity: _EntityType[Any],
        alias: Optional[Union[Alias, Subquery]] = None,
    ) -> Query[Any]:
        """add a mapped entity to the list of result columns
        to be returned.

        .. seealso::

            :meth:`_sql.Select.add_columns` - v2 comparable method.
        """

        if alias is not None:
            # TODO: deprecate
            entity = AliasedClass(entity, alias)

        self._raw_columns = list(self._raw_columns)

        self._raw_columns.append(
            coercions.expect(
                roles.ColumnsClauseRole, entity, apply_propagate_attrs=self
            )
        )
        return self

    @_generative
    def with_session(self, session: Session) -> Self:
        """Return a :class:`_query.Query` that will use the given
        :class:`.Session`.

        While the :class:`_query.Query`
        object is normally instantiated using the
        :meth:`.Session.query` method, it is legal to build the
        :class:`_query.Query`
        directly without necessarily using a :class:`.Session`.  Such a
        :class:`_query.Query` object, or any :class:`_query.Query`
        already associated
        with a different :class:`.Session`, can produce a new
        :class:`_query.Query`
        object associated with a target session using this method::

            from sqlalchemy.orm import Query

            query = Query([MyClass]).filter(MyClass.id == 5)

            result = query.with_session(my_session).one()

        """

        self.session = session
        return self

    def _legacy_from_self(
        self, *entities: _ColumnsClauseArgument[Any]
    ) -> Self:
        # used for query.count() as well as for the same
        # function in BakedQuery, as well as some old tests in test_baked.py.

        fromclause = (
            self.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
            .correlate(None)
            .subquery()
            ._anonymous_fromclause()
        )

        q = self._from_selectable(fromclause)

        if entities:
            q._set_entities(entities)
        return q

    @_generative
    def _set_enable_single_crit(self, val: bool) -> Self:
        self._compile_options += {"_enable_single_crit": val}
        return self

    @_generative
    def _from_selectable(
        self, fromclause: FromClause, set_entity_from: bool = True
    ) -> Self:
        for attr in (
            "_where_criteria",
            "_order_by_clauses",
            "_group_by_clauses",
            "_limit_clause",
            "_offset_clause",
            "_last_joined_entity",
            "_setup_joins",
            "_memoized_select_entities",
            "_distinct",
            "_distinct_on",
            "_having_criteria",
            "_prefixes",
            "_suffixes",
        ):
            self.__dict__.pop(attr, None)
        self._set_select_from([fromclause], set_entity_from)
        self._compile_options += {
            "_enable_single_crit": False,
        }

        return self

    @util.deprecated(
        "1.4",
        ":meth:`_query.Query.values` "
        "is deprecated and will be removed in a "
        "future release.  Please use :meth:`_query.Query.with_entities`",
    )
    def values(self, *columns: _ColumnsClauseArgument[Any]) -> Iterable[Any]:
        """Return an iterator yielding result tuples corresponding
        to the given list of columns

        """
        return self._values_no_warn(*columns)

    _values = values

    def _values_no_warn(
        self, *columns: _ColumnsClauseArgument[Any]
    ) -> Iterable[Any]:
        if not columns:
            return iter(())
        q = self._clone().enable_eagerloads(False)
        q._set_entities(columns)
        if not q.load_options._yield_per:
            q.load_options += {"_yield_per": 10}
        return iter(q)

    @util.deprecated(
        "1.4",
        ":meth:`_query.Query.value` "
        "is deprecated and will be removed in a "
        "future release.  Please use :meth:`_query.Query.with_entities` "
        "in combination with :meth:`_query.Query.scalar`",
    )
    def value(self, column: _ColumnExpressionArgument[Any]) -> Any:
        """Return a scalar result corresponding to the given
        column expression.

        """
        try:
            return next(self._values_no_warn(column))[0]  # type: ignore
        except StopIteration:
            return None

    @overload
    def with_entities(self, _entity: _EntityType[_O]) -> Query[_O]: ...

    @overload
    def with_entities(
        self,
        _colexpr: roles.TypedColumnsClauseRole[_T],
    ) -> RowReturningQuery[Tuple[_T]]: ...

    # START OVERLOADED FUNCTIONS self.with_entities RowReturningQuery 2-8

    # code within this block is **programmatically,
    # statically generated** by tools/generate_tuple_map_overloads.py

    @overload
    def with_entities(
        self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1]
    ) -> RowReturningQuery[Tuple[_T0, _T1]]: ...

    @overload
    def with_entities(
        self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2]
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2]]: ...

    @overload
    def with_entities(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3]]: ...

    @overload
    def with_entities(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...

    @overload
    def with_entities(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

    @overload
    def with_entities(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

    @overload
    def with_entities(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
        __ent7: _TCCA[_T7],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...

    # END OVERLOADED FUNCTIONS self.with_entities

    @overload
    def with_entities(
        self, *entities: _ColumnsClauseArgument[Any]
    ) -> Query[Any]: ...

    @_generative
    def with_entities(
        self, *entities: _ColumnsClauseArgument[Any], **__kw: Any
    ) -> Query[Any]:
        r"""Return a new :class:`_query.Query`
        replacing the SELECT list with the
        given entities.

        e.g.::

            # Users, filtered on some arbitrary criterion
            # and then ordered by related email address
            q = (
                session.query(User)
                .join(User.address)
                .filter(User.name.like("%ed%"))
                .order_by(Address.email)
            )

            # given *only* User.id==5, Address.email, and 'q', what
            # would the *next* User in the result be ?
            subq = (
                q.with_entities(Address.email)
                .order_by(None)
                .filter(User.id == 5)
                .subquery()
            )
            q = q.join((subq, subq.c.email < Address.email)).limit(1)

        .. seealso::

            :meth:`_sql.Select.with_only_columns` - v2 comparable method.
        """
        if __kw:
            raise _no_kw()

        # Query has all the same fields as Select for this operation
        # this could in theory be based on a protocol but not sure if it's
        # worth it
        _MemoizedSelectEntities._generate_for_statement(self)  # type: ignore
        self._set_entities(entities)
        return self

    @_generative
    def add_columns(
        self, *column: _ColumnExpressionArgument[Any]
    ) -> Query[Any]:
        """Add one or more column expressions to the list
        of result columns to be returned.

        .. seealso::

            :meth:`_sql.Select.add_columns` - v2 comparable method.
        """

        self._raw_columns = list(self._raw_columns)

        self._raw_columns.extend(
            coercions.expect(
                roles.ColumnsClauseRole,
                c,
                apply_propagate_attrs=self,
                post_inspect=True,
            )
            for c in column
        )
        return self

    @util.deprecated(
        "1.4",
        ":meth:`_query.Query.add_column` "
        "is deprecated and will be removed in a "
        "future release.  Please use :meth:`_query.Query.add_columns`",
    )
    def add_column(self, column: _ColumnExpressionArgument[Any]) -> Query[Any]:
        """Add a column expression to the list of result columns to be
        returned.

        """
        return self.add_columns(column)

    @_generative
    def options(self, *args: ExecutableOption) -> Self:
        """Return a new :class:`_query.Query` object,
        applying the given list of
        mapper options.

        Most supplied options regard changing how column- and
        relationship-mapped attributes are loaded.

        .. seealso::

            :ref:`loading_columns`

            :ref:`relationship_loader_options`

        """

        opts = tuple(util.flatten_iterator(args))
        if self._compile_options._current_path:
            # opting for lower method overhead for the checks
            for opt in opts:
                if not opt._is_core and opt._is_legacy_option:  # type: ignore
                    opt.process_query_conditionally(self)  # type: ignore
        else:
            for opt in opts:
                if not opt._is_core and opt._is_legacy_option:  # type: ignore
                    opt.process_query(self)  # type: ignore

        self._with_options += opts
        return self

    def with_transformation(
        self, fn: Callable[[Query[Any]], Query[Any]]
    ) -> Query[Any]:
        """Return a new :class:`_query.Query` object transformed by
        the given function.

        E.g.::

            def filter_something(criterion):
                def transform(q):
                    return q.filter(criterion)

                return transform


            q = q.with_transformation(filter_something(x == 5))

        This allows ad-hoc recipes to be created for :class:`_query.Query`
        objects.

        """
        return fn(self)

    def get_execution_options(self) -> _ImmutableExecuteOptions:
        """Get the non-SQL options which will take effect during execution.

        .. versionadded:: 1.3

        .. seealso::

            :meth:`_query.Query.execution_options`

            :meth:`_sql.Select.get_execution_options` - v2 comparable method.

        """
        return self._execution_options

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
        populate_existing: bool = False,
        autoflush: bool = False,
        preserve_rowcount: bool = False,
        **opt: Any,
    ) -> Self: ...

    @overload
    def execution_options(self, **opt: Any) -> Self: ...

    @_generative
    def execution_options(self, **kwargs: Any) -> Self:
        """Set non-SQL options which take effect during execution.

        Options allowed here include all of those accepted by
        :meth:`_engine.Connection.execution_options`, as well as a series
        of ORM specific options:

        ``populate_existing=True`` - equivalent to using
        :meth:`_orm.Query.populate_existing`

        ``autoflush=True|False`` - equivalent to using
        :meth:`_orm.Query.autoflush`

        ``yield_per=<value>`` - equivalent to using
        :meth:`_orm.Query.yield_per`

        Note that the ``stream_results`` execution option is enabled
        automatically if the :meth:`~sqlalchemy.orm.query.Query.yield_per()`
        method or execution option is used.

        .. versionadded:: 1.4 - added ORM options to
           :meth:`_orm.Query.execution_options`

        The execution options may also be specified on a per execution basis
        when using :term:`2.0 style` queries via the
        :paramref:`_orm.Session.execution_options` parameter.

        .. warning:: The
           :paramref:`_engine.Connection.execution_options.stream_results`
           parameter should not be used at the level of individual ORM
           statement executions, as the :class:`_orm.Session` will not track
           objects from different schema translate maps within a single
           session.  For multiple schema translate maps within the scope of a
           single :class:`_orm.Session`, see :ref:`examples_sharding`.


        .. seealso::

            :ref:`engine_stream_results`

            :meth:`_query.Query.get_execution_options`

            :meth:`_sql.Select.execution_options` - v2 equivalent method.

        """
        self._execution_options = self._execution_options.union(kwargs)
        return self

    @_generative
    def with_for_update(
        self,
        *,
        nowait: bool = False,
        read: bool = False,
        of: Optional[_ForUpdateOfArgument] = None,
        skip_locked: bool = False,
        key_share: bool = False,
    ) -> Self:
        """return a new :class:`_query.Query`
        with the specified options for the
        ``FOR UPDATE`` clause.

        The behavior of this method is identical to that of
        :meth:`_expression.GenerativeSelect.with_for_update`.
        When called with no arguments,
        the resulting ``SELECT`` statement will have a ``FOR UPDATE`` clause
        appended.  When additional arguments are specified, backend-specific
        options such as ``FOR UPDATE NOWAIT`` or ``LOCK IN SHARE MODE``
        can take effect.

        E.g.::

            q = (
                sess.query(User)
                .populate_existing()
                .with_for_update(nowait=True, of=User)
            )

        The above query on a PostgreSQL backend will render like:

        .. sourcecode:: sql

            SELECT users.id AS users_id FROM users FOR UPDATE OF users NOWAIT

        .. warning::

            Using ``with_for_update`` in the context of eager loading
            relationships is not officially supported or recommended by
            SQLAlchemy and may not work with certain queries on various
            database backends.  When ``with_for_update`` is successfully used
            with a query that involves :func:`_orm.joinedload`, SQLAlchemy will
            attempt to emit SQL that locks all involved tables.

        .. note::  It is generally a good idea to combine the use of the
           :meth:`_orm.Query.populate_existing` method when using the
           :meth:`_orm.Query.with_for_update` method.   The purpose of
           :meth:`_orm.Query.populate_existing` is to force all the data read
           from the SELECT to be populated into the ORM objects returned,
           even if these objects are already in the :term:`identity map`.

        .. seealso::

            :meth:`_expression.GenerativeSelect.with_for_update`
            - Core level method with
            full argument and behavioral description.

            :meth:`_orm.Query.populate_existing` - overwrites attributes of
            objects already loaded in the identity map.

        """  # noqa: E501

        self._for_update_arg = ForUpdateArg(
            read=read,
            nowait=nowait,
            of=of,
            skip_locked=skip_locked,
            key_share=key_share,
        )
        return self

    @_generative
    def params(
        self, __params: Optional[Dict[str, Any]] = None, **kw: Any
    ) -> Self:
        r"""Add values for bind parameters which may have been
        specified in filter().

        Parameters may be specified using \**kwargs, or optionally a single
        dictionary as the first positional argument. The reason for both is
        that \**kwargs is convenient, however some parameter dictionaries
        contain unicode keys in which case \**kwargs cannot be used.

        """
        if __params:
            kw.update(__params)
        self._params = self._params.union(kw)
        return self

    def where(self, *criterion: _ColumnExpressionArgument[bool]) -> Self:
        """A synonym for :meth:`.Query.filter`.

        .. versionadded:: 1.4

        .. seealso::

            :meth:`_sql.Select.where` - v2 equivalent method.

        """
        return self.filter(*criterion)

    @_generative
    @_assertions(_no_statement_condition, _no_limit_offset)
    def filter(self, *criterion: _ColumnExpressionArgument[bool]) -> Self:
        r"""Apply the given filtering criterion to a copy
        of this :class:`_query.Query`, using SQL expressions.

        e.g.::

            session.query(MyClass).filter(MyClass.name == "some name")

        Multiple criteria may be specified as comma separated; the effect
        is that they will be joined together using the :func:`.and_`
        function::

            session.query(MyClass).filter(MyClass.name == "some name", MyClass.id > 5)

        The criterion is any SQL expression object applicable to the
        WHERE clause of a select.   String expressions are coerced
        into SQL expression constructs via the :func:`_expression.text`
        construct.

        .. seealso::

            :meth:`_query.Query.filter_by` - filter on keyword expressions.

            :meth:`_sql.Select.where` - v2 equivalent method.

        """  # noqa: E501
        for crit in list(criterion):
            crit = coercions.expect(
                roles.WhereHavingRole, crit, apply_propagate_attrs=self
            )

            self._where_criteria += (crit,)
        return self

    @util.memoized_property
    def _last_joined_entity(
        self,
    ) -> Optional[Union[_InternalEntityType[Any], _JoinTargetElement]]:
        if self._setup_joins:
            return _determine_last_joined_entity(
                self._setup_joins,
            )
        else:
            return None

    def _filter_by_zero(self) -> Any:
        """for the filter_by() method, return the target entity for which
        we will attempt to derive an expression from based on string name.

        """

        if self._setup_joins:
            _last_joined_entity = self._last_joined_entity
            if _last_joined_entity is not None:
                return _last_joined_entity

        # discussion related to #7239
        # special check determines if we should try to derive attributes
        # for filter_by() from the "from object", i.e., if the user
        # called query.select_from(some selectable).filter_by(some_attr=value).
        # We don't want to do that in the case that methods like
        # from_self(), select_entity_from(), or a set op like union() were
        # called; while these methods also place a
        # selectable in the _from_obj collection, they also set up
        # the _set_base_alias boolean which turns on the whole "adapt the
        # entity to this selectable" thing, meaning the query still continues
        # to construct itself in terms of the lead entity that was passed
        # to query(), e.g. query(User).from_self() is still in terms of User,
        # and not the subquery that from_self() created.   This feature of
        # "implicitly adapt all occurrences of entity X to some arbitrary
        # subquery" is the main thing I am trying to do away with in 2.0 as
        # users should now used aliased() for that, but I can't entirely get
        # rid of it due to query.union() and other set ops relying upon it.
        #
        # compare this to the base Select()._filter_by_zero() which can
        # just return self._from_obj[0] if present, because there is no
        # "_set_base_alias" feature.
        #
        # IOW, this conditional essentially detects if
        # "select_from(some_selectable)" has been called, as opposed to
        # "select_entity_from()", "from_self()"
        # or "union() / some_set_op()".
        if self._from_obj and not self._compile_options._set_base_alias:
            return self._from_obj[0]

        return self._raw_columns[0]

    def filter_by(self, **kwargs: Any) -> Self:
        r"""Apply the given filtering criterion to a copy
        of this :class:`_query.Query`, using keyword expressions.

        e.g.::

            session.query(MyClass).filter_by(name="some name")

        Multiple criteria may be specified as comma separated; the effect
        is that they will be joined together using the :func:`.and_`
        function::

            session.query(MyClass).filter_by(name="some name", id=5)

        The keyword expressions are extracted from the primary
        entity of the query, or the last entity that was the
        target of a call to :meth:`_query.Query.join`.

        .. seealso::

            :meth:`_query.Query.filter` - filter on SQL expressions.

            :meth:`_sql.Select.filter_by` - v2 comparable method.

        """
        from_entity = self._filter_by_zero()

        clauses = [
            _entity_namespace_key(from_entity, key) == value
            for key, value in kwargs.items()
        ]
        return self.filter(*clauses)

    @_generative
    def order_by(
        self,
        __first: Union[
            Literal[None, False, _NoArg.NO_ARG],
            _ColumnExpressionOrStrLabelArgument[Any],
        ] = _NoArg.NO_ARG,
        *clauses: _ColumnExpressionOrStrLabelArgument[Any],
    ) -> Self:
        """Apply one or more ORDER BY criteria to the query and return
        the newly resulting :class:`_query.Query`.

        e.g.::

            q = session.query(Entity).order_by(Entity.id, Entity.name)

        Calling this method multiple times is equivalent to calling it once
        with all the clauses concatenated. All existing ORDER BY criteria may
        be cancelled by passing ``None`` by itself.  New ORDER BY criteria may
        then be added by invoking :meth:`_orm.Query.order_by` again, e.g.::

            # will erase all ORDER BY and ORDER BY new_col alone
            q = q.order_by(None).order_by(new_col)

        .. seealso::

            These sections describe ORDER BY in terms of :term:`2.0 style`
            invocation but apply to :class:`_orm.Query` as well:

            :ref:`tutorial_order_by` - in the :ref:`unified_tutorial`

            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`

            :meth:`_sql.Select.order_by` - v2 equivalent method.

        """

        for assertion in (self._no_statement_condition, self._no_limit_offset):
            assertion("order_by")

        if not clauses and (__first is None or __first is False):
            self._order_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            criterion = tuple(
                coercions.expect(roles.OrderByRole, clause)
                for clause in (__first,) + clauses
            )
            self._order_by_clauses += criterion

        return self

    @_generative
    def group_by(
        self,
        __first: Union[
            Literal[None, False, _NoArg.NO_ARG],
            _ColumnExpressionOrStrLabelArgument[Any],
        ] = _NoArg.NO_ARG,
        *clauses: _ColumnExpressionOrStrLabelArgument[Any],
    ) -> Self:
        """Apply one or more GROUP BY criterion to the query and return
        the newly resulting :class:`_query.Query`.

        All existing GROUP BY settings can be suppressed by
        passing ``None`` - this will suppress any GROUP BY configured
        on mappers as well.

        .. seealso::

            These sections describe GROUP BY in terms of :term:`2.0 style`
            invocation but apply to :class:`_orm.Query` as well:

            :ref:`tutorial_group_by_w_aggregates` - in the
            :ref:`unified_tutorial`

            :ref:`tutorial_order_by_label` - in the :ref:`unified_tutorial`

            :meth:`_sql.Select.group_by` - v2 equivalent method.

        """

        for assertion in (self._no_statement_condition, self._no_limit_offset):
            assertion("group_by")

        if not clauses and (__first is None or __first is False):
            self._group_by_clauses = ()
        elif __first is not _NoArg.NO_ARG:
            criterion = tuple(
                coercions.expect(roles.GroupByRole, clause)
                for clause in (__first,) + clauses
            )
            self._group_by_clauses += criterion
        return self

    @_generative
    @_assertions(_no_statement_condition, _no_limit_offset)
    def having(self, *having: _ColumnExpressionArgument[bool]) -> Self:
        r"""Apply a HAVING criterion to the query and return the
        newly resulting :class:`_query.Query`.

        :meth:`_query.Query.having` is used in conjunction with
        :meth:`_query.Query.group_by`.

        HAVING criterion makes it possible to use filters on aggregate
        functions like COUNT, SUM, AVG, MAX, and MIN, eg.::

            q = (
                session.query(User.id)
                .join(User.addresses)
                .group_by(User.id)
                .having(func.count(Address.id) > 2)
            )

        .. seealso::

            :meth:`_sql.Select.having` - v2 equivalent method.

        """

        for criterion in having:
            having_criteria = coercions.expect(
                roles.WhereHavingRole, criterion
            )
            self._having_criteria += (having_criteria,)
        return self

    def _set_op(self, expr_fn: Any, *q: Query[Any]) -> Self:
        list_of_queries = (self,) + q
        return self._from_selectable(expr_fn(*(list_of_queries)).subquery())

    def union(self, *q: Query[Any]) -> Self:
        """Produce a UNION of this Query against one or more queries.

        e.g.::

            q1 = sess.query(SomeClass).filter(SomeClass.foo == "bar")
            q2 = sess.query(SomeClass).filter(SomeClass.bar == "foo")

            q3 = q1.union(q2)

        The method accepts multiple Query objects so as to control
        the level of nesting.  A series of ``union()`` calls such as::

            x.union(y).union(z).all()

        will nest on each ``union()``, and produces:

        .. sourcecode:: sql

            SELECT * FROM (SELECT * FROM (SELECT * FROM X UNION
                            SELECT * FROM y) UNION SELECT * FROM Z)

        Whereas::

            x.union(y, z).all()

        produces:

        .. sourcecode:: sql

            SELECT * FROM (SELECT * FROM X UNION SELECT * FROM y UNION
                            SELECT * FROM Z)

        Note that many database backends do not allow ORDER BY to
        be rendered on a query called within UNION, EXCEPT, etc.
        To disable all ORDER BY clauses including those configured
        on mappers, issue ``query.order_by(None)`` - the resulting
        :class:`_query.Query` object will not render ORDER BY within
        its SELECT statement.

        .. seealso::

            :meth:`_sql.Select.union` - v2 equivalent method.

        """
        return self._set_op(expression.union, *q)

    def union_all(self, *q: Query[Any]) -> Self:
        """Produce a UNION ALL of this Query against one or more queries.

        Works the same way as :meth:`~sqlalchemy.orm.query.Query.union`. See
        that method for usage examples.

        .. seealso::

            :meth:`_sql.Select.union_all` - v2 equivalent method.

        """
        return self._set_op(expression.union_all, *q)

    def intersect(self, *q: Query[Any]) -> Self:
        """Produce an INTERSECT of this Query against one or more queries.

        Works the same way as :meth:`~sqlalchemy.orm.query.Query.union`. See
        that method for usage examples.

        .. seealso::

            :meth:`_sql.Select.intersect` - v2 equivalent method.

        """
        return self._set_op(expression.intersect, *q)

    def intersect_all(self, *q: Query[Any]) -> Self:
        """Produce an INTERSECT ALL of this Query against one or more queries.

        Works the same way as :meth:`~sqlalchemy.orm.query.Query.union`. See
        that method for usage examples.

        .. seealso::

            :meth:`_sql.Select.intersect_all` - v2 equivalent method.

        """
        return self._set_op(expression.intersect_all, *q)

    def except_(self, *q: Query[Any]) -> Self:
        """Produce an EXCEPT of this Query against one or more queries.

        Works the same way as :meth:`~sqlalchemy.orm.query.Query.union`. See
        that method for usage examples.

        .. seealso::

            :meth:`_sql.Select.except_` - v2 equivalent method.

        """
        return self._set_op(expression.except_, *q)

    def except_all(self, *q: Query[Any]) -> Self:
        """Produce an EXCEPT ALL of this Query against one or more queries.

        Works the same way as :meth:`~sqlalchemy.orm.query.Query.union`. See
        that method for usage examples.

        .. seealso::

            :meth:`_sql.Select.except_all` - v2 equivalent method.

        """
        return self._set_op(expression.except_all, *q)

    @_generative
    @_assertions(_no_statement_condition, _no_limit_offset)
    def join(
        self,
        target: _JoinTargetArgument,
        onclause: Optional[_OnClauseArgument] = None,
        *,
        isouter: bool = False,
        full: bool = False,
    ) -> Self:
        r"""Create a SQL JOIN against this :class:`_query.Query`
        object's criterion
        and apply generatively, returning the newly resulting
        :class:`_query.Query`.

        **Simple Relationship Joins**

        Consider a mapping between two classes ``User`` and ``Address``,
        with a relationship ``User.addresses`` representing a collection
        of ``Address`` objects associated with each ``User``.   The most
        common usage of :meth:`_query.Query.join`
        is to create a JOIN along this
        relationship, using the ``User.addresses`` attribute as an indicator
        for how this should occur::

            q = session.query(User).join(User.addresses)

        Where above, the call to :meth:`_query.Query.join` along
        ``User.addresses`` will result in SQL approximately equivalent to:

        .. sourcecode:: sql

            SELECT user.id, user.name
            FROM user JOIN address ON user.id = address.user_id

        In the above example we refer to ``User.addresses`` as passed to
        :meth:`_query.Query.join` as the "on clause", that is, it indicates
        how the "ON" portion of the JOIN should be constructed.

        To construct a chain of joins, multiple :meth:`_query.Query.join`
        calls may be used.  The relationship-bound attribute implies both
        the left and right side of the join at once::

            q = (
                session.query(User)
                .join(User.orders)
                .join(Order.items)
                .join(Item.keywords)
            )

        .. note:: as seen in the above example, **the order in which each
           call to the join() method occurs is important**.    Query would not,
           for example, know how to join correctly if we were to specify
           ``User``, then ``Item``, then ``Order``, in our chain of joins; in
           such a case, depending on the arguments passed, it may raise an
           error that it doesn't know how to join, or it may produce invalid
           SQL in which case the database will raise an error. In correct
           practice, the
           :meth:`_query.Query.join` method is invoked in such a way that lines
           up with how we would want the JOIN clauses in SQL to be
           rendered, and each call should represent a clear link from what
           precedes it.

        **Joins to a Target Entity or Selectable**

        A second form of :meth:`_query.Query.join` allows any mapped entity or
        core selectable construct as a target.   In this usage,
        :meth:`_query.Query.join` will attempt to create a JOIN along the
        natural foreign key relationship between two entities::

            q = session.query(User).join(Address)

        In the above calling form, :meth:`_query.Query.join` is called upon to
        create the "on clause" automatically for us.  This calling form will
        ultimately raise an error if either there are no foreign keys between
        the two entities, or if there are multiple foreign key linkages between
        the target entity and the entity or entities already present on the
        left side such that creating a join requires more information.  Note
        that when indicating a join to a target without any ON clause, ORM
        configured relationships are not taken into account.

        **Joins to a Target with an ON Clause**

        The third calling form allows both the target entity as well
        as the ON clause to be passed explicitly.    A example that includes
        a SQL expression as the ON clause is as follows::

            q = session.query(User).join(Address, User.id == Address.user_id)

        The above form may also use a relationship-bound attribute as the
        ON clause as well::

            q = session.query(User).join(Address, User.addresses)

        The above syntax can be useful for the case where we wish
        to join to an alias of a particular target entity.  If we wanted
        to join to ``Address`` twice, it could be achieved using two
        aliases set up using the :func:`~sqlalchemy.orm.aliased` function::

            a1 = aliased(Address)
            a2 = aliased(Address)

            q = (
                session.query(User)
                .join(a1, User.addresses)
                .join(a2, User.addresses)
                .filter(a1.email_address == "ed@foo.com")
                .filter(a2.email_address == "ed@bar.com")
            )

        The relationship-bound calling form can also specify a target entity
        using the :meth:`_orm.PropComparator.of_type` method; a query
        equivalent to the one above would be::

            a1 = aliased(Address)
            a2 = aliased(Address)

            q = (
                session.query(User)
                .join(User.addresses.of_type(a1))
                .join(User.addresses.of_type(a2))
                .filter(a1.email_address == "ed@foo.com")
                .filter(a2.email_address == "ed@bar.com")
            )

        **Augmenting Built-in ON Clauses**

        As a substitute for providing a full custom ON condition for an
        existing relationship, the :meth:`_orm.PropComparator.and_` function
        may be applied to a relationship attribute to augment additional
        criteria into the ON clause; the additional criteria will be combined
        with the default criteria using AND::

            q = session.query(User).join(
                User.addresses.and_(Address.email_address != "foo@bar.com")
            )

        .. versionadded:: 1.4

        **Joining to Tables and Subqueries**


        The target of a join may also be any table or SELECT statement,
        which may be related to a target entity or not.   Use the
        appropriate ``.subquery()`` method in order to make a subquery
        out of a query::

            subq = (
                session.query(Address)
                .filter(Address.email_address == "ed@foo.com")
                .subquery()
            )


            q = session.query(User).join(subq, User.id == subq.c.user_id)

        Joining to a subquery in terms of a specific relationship and/or
        target entity may be achieved by linking the subquery to the
        entity using :func:`_orm.aliased`::

            subq = (
                session.query(Address)
                .filter(Address.email_address == "ed@foo.com")
                .subquery()
            )

            address_subq = aliased(Address, subq)

            q = session.query(User).join(User.addresses.of_type(address_subq))

        **Controlling what to Join From**

        In cases where the left side of the current state of
        :class:`_query.Query` is not in line with what we want to join from,
        the :meth:`_query.Query.select_from` method may be used::

            q = (
                session.query(Address)
                .select_from(User)
                .join(User.addresses)
                .filter(User.name == "ed")
            )

        Which will produce SQL similar to:

        .. sourcecode:: sql

            SELECT address.* FROM user
                JOIN address ON user.id=address.user_id
                WHERE user.name = :name_1

        .. seealso::

            :meth:`_sql.Select.join` - v2 equivalent method.

        :param \*props: Incoming arguments for :meth:`_query.Query.join`,
         the props collection in modern use should be considered to be a  one
         or two argument form, either as a single "target" entity or ORM
         attribute-bound relationship, or as a target entity plus an "on
         clause" which  may be a SQL expression or ORM attribute-bound
         relationship.

        :param isouter=False: If True, the join used will be a left outer join,
         just as if the :meth:`_query.Query.outerjoin` method were called.

        :param full=False: render FULL OUTER JOIN; implies ``isouter``.

        """

        join_target = coercions.expect(
            roles.JoinTargetRole,
            target,
            apply_propagate_attrs=self,
            legacy=True,
        )
        if onclause is not None:
            onclause_element = coercions.expect(
                roles.OnClauseRole, onclause, legacy=True
            )
        else:
            onclause_element = None

        self._setup_joins += (
            (
                join_target,
                onclause_element,
                None,
                {
                    "isouter": isouter,
                    "full": full,
                },
            ),
        )

        self.__dict__.pop("_last_joined_entity", None)
        return self

    def outerjoin(
        self,
        target: _JoinTargetArgument,
        onclause: Optional[_OnClauseArgument] = None,
        *,
        full: bool = False,
    ) -> Self:
        """Create a left outer join against this ``Query`` object's criterion
        and apply generatively, returning the newly resulting ``Query``.

        Usage is the same as the ``join()`` method.

        .. seealso::

            :meth:`_sql.Select.outerjoin` - v2 equivalent method.

        """
        return self.join(target, onclause=onclause, isouter=True, full=full)

    @_generative
    @_assertions(_no_statement_condition)
    def reset_joinpoint(self) -> Self:
        """Return a new :class:`.Query`, where the "join point" has
        been reset back to the base FROM entities of the query.

        This method is usually used in conjunction with the
        ``aliased=True`` feature of the :meth:`~.Query.join`
        method.  See the example in :meth:`~.Query.join` for how
        this is used.

        """
        self._last_joined_entity = None

        return self

    @_generative
    @_assertions(_no_clauseelement_condition)
    def select_from(self, *from_obj: _FromClauseArgument) -> Self:
        r"""Set the FROM clause of this :class:`.Query` explicitly.

        :meth:`.Query.select_from` is often used in conjunction with
        :meth:`.Query.join` in order to control which entity is selected
        from on the "left" side of the join.

        The entity or selectable object here effectively replaces the
        "left edge" of any calls to :meth:`~.Query.join`, when no
        joinpoint is otherwise established - usually, the default "join
        point" is the leftmost entity in the :class:`~.Query` object's
        list of entities to be selected.

        A typical example::

            q = (
                session.query(Address)
                .select_from(User)
                .join(User.addresses)
                .filter(User.name == "ed")
            )

        Which produces SQL equivalent to:

        .. sourcecode:: sql

            SELECT address.* FROM user
            JOIN address ON user.id=address.user_id
            WHERE user.name = :name_1

        :param \*from_obj: collection of one or more entities to apply
         to the FROM clause.  Entities can be mapped classes,
         :class:`.AliasedClass` objects, :class:`.Mapper` objects
         as well as core :class:`.FromClause` elements like subqueries.

        .. seealso::

            :meth:`~.Query.join`

            :meth:`.Query.select_entity_from`

            :meth:`_sql.Select.select_from` - v2 equivalent method.

        """

        self._set_select_from(from_obj, False)
        return self

    def __getitem__(self, item: Any) -> Any:
        return orm_util._getitem(
            self,
            item,
        )

    @_generative
    @_assertions(_no_statement_condition)
    def slice(
        self,
        start: int,
        stop: int,
    ) -> Self:
        """Computes the "slice" of the :class:`_query.Query` represented by
        the given indices and returns the resulting :class:`_query.Query`.

        The start and stop indices behave like the argument to Python's
        built-in :func:`range` function. This method provides an
        alternative to using ``LIMIT``/``OFFSET`` to get a slice of the
        query.

        For example, ::

            session.query(User).order_by(User.id).slice(1, 3)

        renders as

        .. sourcecode:: sql

           SELECT users.id AS users_id,
                  users.name AS users_name
           FROM users ORDER BY users.id
           LIMIT ? OFFSET ?
           (2, 1)

        .. seealso::

           :meth:`_query.Query.limit`

           :meth:`_query.Query.offset`

           :meth:`_sql.Select.slice` - v2 equivalent method.

        """

        self._limit_clause, self._offset_clause = sql_util._make_slice(
            self._limit_clause, self._offset_clause, start, stop
        )
        return self

    @_generative
    @_assertions(_no_statement_condition)
    def limit(self, limit: _LimitOffsetType) -> Self:
        """Apply a ``LIMIT`` to the query and return the newly resulting
        ``Query``.

        .. seealso::

            :meth:`_sql.Select.limit` - v2 equivalent method.

        """
        self._limit_clause = sql_util._offset_or_limit_clause(limit)
        return self

    @_generative
    @_assertions(_no_statement_condition)
    def offset(self, offset: _LimitOffsetType) -> Self:
        """Apply an ``OFFSET`` to the query and return the newly resulting
        ``Query``.

        .. seealso::

            :meth:`_sql.Select.offset` - v2 equivalent method.
        """
        self._offset_clause = sql_util._offset_or_limit_clause(offset)
        return self

    @_generative
    @_assertions(_no_statement_condition)
    def distinct(self, *expr: _ColumnExpressionArgument[Any]) -> Self:
        r"""Apply a ``DISTINCT`` to the query and return the newly resulting
        ``Query``.


        .. note::

            The ORM-level :meth:`.distinct` call includes logic that will
            automatically add columns from the ORDER BY of the query to the
            columns clause of the SELECT statement, to satisfy the common need
            of the database backend that ORDER BY columns be part of the SELECT
            list when DISTINCT is used.   These columns *are not* added to the
            list of columns actually fetched by the :class:`_query.Query`,
            however,
            so would not affect results. The columns are passed through when
            using the :attr:`_query.Query.statement` accessor, however.

            .. deprecated:: 2.0  This logic is deprecated and will be removed
               in SQLAlchemy 2.0.     See :ref:`migration_20_query_distinct`
               for a description of this use case in 2.0.

        .. seealso::

            :meth:`_sql.Select.distinct` - v2 equivalent method.

        :param \*expr: optional column expressions.  When present,
         the PostgreSQL dialect will render a ``DISTINCT ON (<expressions>)``
         construct.

         .. deprecated:: 1.4 Using \*expr in other dialects is deprecated
            and will raise :class:`_exc.CompileError` in a future version.

        """
        if expr:
            self._distinct = True
            self._distinct_on = self._distinct_on + tuple(
                coercions.expect(roles.ByOfRole, e) for e in expr
            )
        else:
            self._distinct = True
        return self

    def all(self) -> List[_T]:
        """Return the results represented by this :class:`_query.Query`
        as a list.

        This results in an execution of the underlying SQL statement.

        .. warning::  The :class:`_query.Query` object,
           when asked to return either
           a sequence or iterator that consists of full ORM-mapped entities,
           will **deduplicate entries based on primary key**.  See the FAQ for
           more details.

            .. seealso::

                :ref:`faq_query_deduplicating`

        .. seealso::

            :meth:`_engine.Result.all` - v2 comparable method.

            :meth:`_engine.Result.scalars` - v2 comparable method.
        """
        return self._iter().all()  # type: ignore

    @_generative
    @_assertions(_no_clauseelement_condition)
    def from_statement(self, statement: ExecutableReturnsRows) -> Self:
        """Execute the given SELECT statement and return results.

        This method bypasses all internal statement compilation, and the
        statement is executed without modification.

        The statement is typically either a :func:`_expression.text`
        or :func:`_expression.select` construct, and should return the set
        of columns
        appropriate to the entity class represented by this
        :class:`_query.Query`.

        .. seealso::

            :meth:`_sql.Select.from_statement` - v2 comparable method.

        """
        statement = coercions.expect(
            roles.SelectStatementRole, statement, apply_propagate_attrs=self
        )
        self._statement = statement
        return self

    def first(self) -> Optional[_T]:
        """Return the first result of this ``Query`` or
        None if the result doesn't contain any row.

        first() applies a limit of one within the generated SQL, so that
        only one primary entity row is generated on the server side
        (note this may consist of multiple result rows if join-loaded
        collections are present).

        Calling :meth:`_query.Query.first`
        results in an execution of the underlying
        query.

        .. seealso::

            :meth:`_query.Query.one`

            :meth:`_query.Query.one_or_none`

            :meth:`_engine.Result.first` - v2 comparable method.

            :meth:`_engine.Result.scalars` - v2 comparable method.

        """
        # replicates limit(1) behavior
        if self._statement is not None:
            return self._iter().first()  # type: ignore
        else:
            return self.limit(1)._iter().first()  # type: ignore

    def one_or_none(self) -> Optional[_T]:
        """Return at most one result or raise an exception.

        Returns ``None`` if the query selects
        no rows.  Raises ``sqlalchemy.orm.exc.MultipleResultsFound``
        if multiple object identities are returned, or if multiple
        rows are returned for a query that returns only scalar values
        as opposed to full identity-mapped entities.

        Calling :meth:`_query.Query.one_or_none`
        results in an execution of the
        underlying query.

        .. seealso::

            :meth:`_query.Query.first`

            :meth:`_query.Query.one`

            :meth:`_engine.Result.one_or_none` - v2 comparable method.

            :meth:`_engine.Result.scalar_one_or_none` - v2 comparable method.

        """
        return self._iter().one_or_none()  # type: ignore

    def one(self) -> _T:
        """Return exactly one result or raise an exception.

        Raises :class:`_exc.NoResultFound` if the query selects no rows.
        Raises :class:`_exc.MultipleResultsFound` if multiple object identities
        are returned, or if multiple rows are returned for a query that returns
        only scalar values as opposed to full identity-mapped entities.

        Calling :meth:`.one` results in an execution of the underlying query.

        .. seealso::

            :meth:`_query.Query.first`

            :meth:`_query.Query.one_or_none`

            :meth:`_engine.Result.one` - v2 comparable method.

            :meth:`_engine.Result.scalar_one` - v2 comparable method.

        """
        return self._iter().one()  # type: ignore

    def scalar(self) -> Any:
        """Return the first element of the first result or None
        if no rows present.  If multiple rows are returned,
        raises :class:`_exc.MultipleResultsFound`.

          >>> session.query(Item).scalar()
          <Item>
          >>> session.query(Item.id).scalar()
          1
          >>> session.query(Item.id).filter(Item.id < 0).scalar()
          None
          >>> session.query(Item.id, Item.name).scalar()
          1
          >>> session.query(func.count(Parent.id)).scalar()
          20

        This results in an execution of the underlying query.

        .. seealso::

            :meth:`_engine.Result.scalar` - v2 comparable method.

        """
        # TODO: not sure why we can't use result.scalar() here
        try:
            ret = self.one()
            if not isinstance(ret, collections_abc.Sequence):
                return ret
            return ret[0]
        except sa_exc.NoResultFound:
            return None

    def __iter__(self) -> Iterator[_T]:
        result = self._iter()
        try:
            yield from result  # type: ignore
        except GeneratorExit:
            # issue #8710 - direct iteration is not re-usable after
            # an iterable block is broken, so close the result
            result._soft_close()
            raise

    def _iter(self) -> Union[ScalarResult[_T], Result[_T]]:
        # new style execution.
        params = self._params

        statement = self._statement_20()
        result: Union[ScalarResult[_T], Result[_T]] = self.session.execute(
            statement,
            params,
            execution_options={"_sa_orm_load_options": self.load_options},
        )

        # legacy: automatically set scalars, unique
        if result._attributes.get("is_single_entity", False):
            result = cast("Result[_T]", result).scalars()

        if (
            result._attributes.get("filtered", False)
            and not self.load_options._yield_per
        ):
            result = result.unique()

        return result

    def __str__(self) -> str:
        statement = self._statement_20()

        try:
            bind = (
                self._get_bind_args(statement, self.session.get_bind)
                if self.session
                else None
            )
        except sa_exc.UnboundExecutionError:
            bind = None

        return str(statement.compile(bind))

    def _get_bind_args(self, statement: Any, fn: Any, **kw: Any) -> Any:
        return fn(clause=statement, **kw)

    @property
    def column_descriptions(self) -> List[ORMColumnDescription]:
        """Return metadata about the columns which would be
        returned by this :class:`_query.Query`.

        Format is a list of dictionaries::

            user_alias = aliased(User, name="user2")
            q = sess.query(User, User.id, user_alias)

            # this expression:
            q.column_descriptions

            # would return:
            [
                {
                    "name": "User",
                    "type": User,
                    "aliased": False,
                    "expr": User,
                    "entity": User,
                },
                {
                    "name": "id",
                    "type": Integer(),
                    "aliased": False,
                    "expr": User.id,
                    "entity": User,
                },
                {
                    "name": "user2",
                    "type": User,
                    "aliased": True,
                    "expr": user_alias,
                    "entity": user_alias,
                },
            ]

        .. seealso::

            This API is available using :term:`2.0 style` queries as well,
            documented at:

            * :ref:`queryguide_inspection`

            * :attr:`.Select.column_descriptions`

        """

        return _column_descriptions(self, legacy=True)

    @util.deprecated(
        "2.0",
        "The :meth:`_orm.Query.instances` method is deprecated and will "
        "be removed in a future release. "
        "Use the Select.from_statement() method or aliased() construct in "
        "conjunction with Session.execute() instead.",
    )
    def instances(
        self,
        result_proxy: CursorResult[Any],
        context: Optional[QueryContext] = None,
    ) -> Any:
        """Return an ORM result given a :class:`_engine.CursorResult` and
        :class:`.QueryContext`.

        """
        if context is None:
            util.warn_deprecated(
                "Using the Query.instances() method without a context "
                "is deprecated and will be disallowed in a future release.  "
                "Please make use of :meth:`_query.Query.from_statement` "
                "for linking ORM results to arbitrary select constructs.",
                version="1.4",
            )
            compile_state = self._compile_state(for_statement=False)

            context = QueryContext(
                compile_state,
                compile_state.statement,
                compile_state.statement,
                self._params,
                self.session,
                self.load_options,
            )

        result = loading.instances(result_proxy, context)

        # legacy: automatically set scalars, unique
        if result._attributes.get("is_single_entity", False):
            result = result.scalars()  # type: ignore

        if result._attributes.get("filtered", False):
            result = result.unique()

        # TODO: isn't this supposed to be a list?
        return result

    @util.became_legacy_20(
        ":meth:`_orm.Query.merge_result`",
        alternative="The method is superseded by the "
        ":func:`_orm.merge_frozen_result` function.",
        enable_warnings=False,  # warnings occur via loading.merge_result
    )
    def merge_result(
        self,
        iterator: Union[
            FrozenResult[Any], Iterable[Sequence[Any]], Iterable[object]
        ],
        load: bool = True,
    ) -> Union[FrozenResult[Any], Iterable[Any]]:
        """Merge a result into this :class:`_query.Query` object's Session.

        Given an iterator returned by a :class:`_query.Query`
        of the same structure
        as this one, return an identical iterator of results, with all mapped
        instances merged into the session using :meth:`.Session.merge`. This
        is an optimized method which will merge all mapped instances,
        preserving the structure of the result rows and unmapped columns with
        less method overhead than that of calling :meth:`.Session.merge`
        explicitly for each value.

        The structure of the results is determined based on the column list of
        this :class:`_query.Query` - if these do not correspond,
        unchecked errors
        will occur.

        The 'load' argument is the same as that of :meth:`.Session.merge`.

        For an example of how :meth:`_query.Query.merge_result` is used, see
        the source code for the example :ref:`examples_caching`, where
        :meth:`_query.Query.merge_result` is used to efficiently restore state
        from a cache back into a target :class:`.Session`.

        """

        return loading.merge_result(self, iterator, load)

    def exists(self) -> Exists:
        """A convenience method that turns a query into an EXISTS subquery
        of the form EXISTS (SELECT 1 FROM ... WHERE ...).

        e.g.::

            q = session.query(User).filter(User.name == "fred")
            session.query(q.exists())

        Producing SQL similar to:

        .. sourcecode:: sql

            SELECT EXISTS (
                SELECT 1 FROM users WHERE users.name = :name_1
            ) AS anon_1

        The EXISTS construct is usually used in the WHERE clause::

            session.query(User.id).filter(q.exists()).scalar()

        Note that some databases such as SQL Server don't allow an
        EXISTS expression to be present in the columns clause of a
        SELECT.    To select a simple boolean value based on the exists
        as a WHERE, use :func:`.literal`::

            from sqlalchemy import literal

            session.query(literal(True)).filter(q.exists()).scalar()

        .. seealso::

            :meth:`_sql.Select.exists` - v2 comparable method.

        """

        # .add_columns() for the case that we are a query().select_from(X),
        # so that ".statement" can be produced (#2995) but also without
        # omitting the FROM clause from a query(X) (#2818);
        # .with_only_columns() after we have a core select() so that
        # we get just "SELECT 1" without any entities.

        inner = (
            self.enable_eagerloads(False)
            .add_columns(sql.literal_column("1"))
            .set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
            ._get_select_statement_only()
            .with_only_columns(1)
        )

        ezero = self._entity_from_pre_ent_zero()
        if ezero is not None:
            inner = inner.select_from(ezero)

        return sql.exists(inner)

    def count(self) -> int:
        r"""Return a count of rows this the SQL formed by this :class:`Query`
        would return.

        This generates the SQL for this Query as follows:

        .. sourcecode:: sql

            SELECT count(1) AS count_1 FROM (
                SELECT <rest of query follows...>
            ) AS anon_1

        The above SQL returns a single row, which is the aggregate value
        of the count function; the :meth:`_query.Query.count`
        method then returns
        that single integer value.

        .. warning::

            It is important to note that the value returned by
            count() is **not the same as the number of ORM objects that this
            Query would return from a method such as the .all() method**.
            The :class:`_query.Query` object,
            when asked to return full entities,
            will **deduplicate entries based on primary key**, meaning if the
            same primary key value would appear in the results more than once,
            only one object of that primary key would be present.  This does
            not apply to a query that is against individual columns.

            .. seealso::

                :ref:`faq_query_deduplicating`

        For fine grained control over specific columns to count, to skip the
        usage of a subquery or otherwise control of the FROM clause, or to use
        other aggregate functions, use :attr:`~sqlalchemy.sql.expression.func`
        expressions in conjunction with :meth:`~.Session.query`, i.e.::

            from sqlalchemy import func

            # count User records, without
            # using a subquery.
            session.query(func.count(User.id))

            # return count of user "id" grouped
            # by "name"
            session.query(func.count(User.id)).group_by(User.name)

            from sqlalchemy import distinct

            # count distinct "name" values
            session.query(func.count(distinct(User.name)))

        .. seealso::

            :ref:`migration_20_query_usage`

        """
        col = sql.func.count(sql.literal_column("*"))
        return (  # type: ignore
            self._legacy_from_self(col).enable_eagerloads(False).scalar()
        )

    def delete(
        self,
        synchronize_session: SynchronizeSessionArgument = "auto",
        delete_args: Optional[Dict[Any, Any]] = None,
    ) -> int:
        r"""Perform a DELETE with an arbitrary WHERE clause.

        Deletes rows matched by this query from the database.

        E.g.::

            sess.query(User).filter(User.age == 25).delete(synchronize_session=False)

            sess.query(User).filter(User.age == 25).delete(
                synchronize_session="evaluate"
            )

        .. warning::

            See the section :ref:`orm_expression_update_delete` for important
            caveats and warnings, including limitations when using bulk UPDATE
            and DELETE with mapper inheritance configurations.

        :param synchronize_session: chooses the strategy to update the
         attributes on objects in the session.   See the section
         :ref:`orm_expression_update_delete` for a discussion of these
         strategies.

        :param delete_args: Optional dictionary, if present will be passed
         to the underlying :func:`_expression.delete` construct as the ``**kw``
         for the object.  May be used to pass dialect-specific arguments such
         as ``mysql_limit``.

         .. versionadded:: 2.0.37

        :return: the count of rows matched as returned by the database's
          "row count" feature.

        .. seealso::

            :ref:`orm_expression_update_delete`

        """  # noqa: E501

        bulk_del = BulkDelete(self, delete_args)
        if self.dispatch.before_compile_delete:
            for fn in self.dispatch.before_compile_delete:
                new_query = fn(bulk_del.query, bulk_del)
                if new_query is not None:
                    bulk_del.query = new_query

                self = bulk_del.query

        delete_ = sql.delete(*self._raw_columns)  # type: ignore

        if delete_args:
            delete_ = delete_.with_dialect_options(**delete_args)

        delete_._where_criteria = self._where_criteria
        result: CursorResult[Any] = self.session.execute(
            delete_,
            self._params,
            execution_options=self._execution_options.union(
                {"synchronize_session": synchronize_session}
            ),
        )
        bulk_del.result = result  # type: ignore
        self.session.dispatch.after_bulk_delete(bulk_del)
        result.close()

        return result.rowcount

    def update(
        self,
        values: Dict[_DMLColumnArgument, Any],
        synchronize_session: SynchronizeSessionArgument = "auto",
        update_args: Optional[Dict[Any, Any]] = None,
    ) -> int:
        r"""Perform an UPDATE with an arbitrary WHERE clause.

        Updates rows matched by this query in the database.

        E.g.::

            sess.query(User).filter(User.age == 25).update(
                {User.age: User.age - 10}, synchronize_session=False
            )

            sess.query(User).filter(User.age == 25).update(
                {"age": User.age - 10}, synchronize_session="evaluate"
            )

        .. warning::

            See the section :ref:`orm_expression_update_delete` for important
            caveats and warnings, including limitations when using arbitrary
            UPDATE and DELETE with mapper inheritance configurations.

        :param values: a dictionary with attributes names, or alternatively
         mapped attributes or SQL expressions, as keys, and literal
         values or sql expressions as values.   If :ref:`parameter-ordered
         mode <tutorial_parameter_ordered_updates>` is desired, the values can
         be passed as a list of 2-tuples; this requires that the
         :paramref:`~sqlalchemy.sql.expression.update.preserve_parameter_order`
         flag is passed to the :paramref:`.Query.update.update_args` dictionary
         as well.

        :param synchronize_session: chooses the strategy to update the
         attributes on objects in the session.   See the section
         :ref:`orm_expression_update_delete` for a discussion of these
         strategies.

        :param update_args: Optional dictionary, if present will be passed
         to the underlying :func:`_expression.update` construct as the ``**kw``
         for the object.  May be used to pass dialect-specific arguments such
         as ``mysql_limit``, as well as other special arguments such as
         :paramref:`~sqlalchemy.sql.expression.update.preserve_parameter_order`.

        :return: the count of rows matched as returned by the database's
         "row count" feature.


        .. seealso::

            :ref:`orm_expression_update_delete`

        """

        update_args = update_args or {}

        bulk_ud = BulkUpdate(self, values, update_args)

        if self.dispatch.before_compile_update:
            for fn in self.dispatch.before_compile_update:
                new_query = fn(bulk_ud.query, bulk_ud)
                if new_query is not None:
                    bulk_ud.query = new_query
            self = bulk_ud.query

        upd = sql.update(*self._raw_columns)  # type: ignore

        ppo = update_args.pop("preserve_parameter_order", False)
        if ppo:
            upd = upd.ordered_values(*values)  # type: ignore
        else:
            upd = upd.values(values)
        if update_args:
            upd = upd.with_dialect_options(**update_args)

        upd._where_criteria = self._where_criteria
        result: CursorResult[Any] = self.session.execute(
            upd,
            self._params,
            execution_options=self._execution_options.union(
                {"synchronize_session": synchronize_session}
            ),
        )
        bulk_ud.result = result  # type: ignore
        self.session.dispatch.after_bulk_update(bulk_ud)
        result.close()
        return result.rowcount

    def _compile_state(
        self, for_statement: bool = False, **kw: Any
    ) -> ORMCompileState:
        """Create an out-of-compiler ORMCompileState object.

        The ORMCompileState object is normally created directly as a result
        of the SQLCompiler.process() method being handed a Select()
        or FromStatement() object that uses the "orm" plugin.   This method
        provides a means of creating this ORMCompileState object directly
        without using the compiler.

        This method is used only for deprecated cases, which include
        the .from_self() method for a Query that has multiple levels
        of .from_self() in use, as well as the instances() method.  It is
        also used within the test suite to generate ORMCompileState objects
        for test purposes.

        """

        stmt = self._statement_20(for_statement=for_statement, **kw)
        assert for_statement == stmt._compile_options._for_statement

        # this chooses between ORMFromStatementCompileState and
        # ORMSelectCompileState.  We could also base this on
        # query._statement is not None as we have the ORM Query here
        # however this is the more general path.
        compile_state_cls = cast(
            ORMCompileState,
            ORMCompileState._get_plugin_class_for_plugin(stmt, "orm"),
        )

        return compile_state_cls._create_orm_context(
            stmt, toplevel=True, compiler=None
        )

    def _compile_context(self, for_statement: bool = False) -> QueryContext:
        compile_state = self._compile_state(for_statement=for_statement)
        context = QueryContext(
            compile_state,
            compile_state.statement,
            compile_state.statement,
            self._params,
            self.session,
            self.load_options,
        )

        return context


class AliasOption(interfaces.LoaderOption):
    inherit_cache = False

    @util.deprecated(
        "1.4",
        "The :class:`.AliasOption` object is not necessary "
        "for entities to be matched up to a query that is established "
        "via :meth:`.Query.from_statement` and now does nothing.",
    )
    def __init__(self, alias: Union[Alias, Subquery]):
        r"""Return a :class:`.MapperOption` that will indicate to the
        :class:`_query.Query`
        that the main table has been aliased.

        """

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        pass


class BulkUD:
    """State used for the orm.Query version of update() / delete().

    This object is now specific to Query only.

    """

    def __init__(self, query: Query[Any]):
        self.query = query.enable_eagerloads(False)
        self._validate_query_state()
        self.mapper = self.query._entity_from_pre_ent_zero()

    def _validate_query_state(self) -> None:
        for attr, methname, notset, op in (
            ("_limit_clause", "limit()", None, operator.is_),
            ("_offset_clause", "offset()", None, operator.is_),
            ("_order_by_clauses", "order_by()", (), operator.eq),
            ("_group_by_clauses", "group_by()", (), operator.eq),
            ("_distinct", "distinct()", False, operator.is_),
            (
                "_from_obj",
                "join(), outerjoin(), select_from(), or from_self()",
                (),
                operator.eq,
            ),
            (
                "_setup_joins",
                "join(), outerjoin(), select_from(), or from_self()",
                (),
                operator.eq,
            ),
        ):
            if not op(getattr(self.query, attr), notset):
                raise sa_exc.InvalidRequestError(
                    "Can't call Query.update() or Query.delete() "
                    "when %s has been called" % (methname,)
                )

    @property
    def session(self) -> Session:
        return self.query.session


class BulkUpdate(BulkUD):
    """BulkUD which handles UPDATEs."""

    def __init__(
        self,
        query: Query[Any],
        values: Dict[_DMLColumnArgument, Any],
        update_kwargs: Optional[Dict[Any, Any]],
    ):
        super().__init__(query)
        self.values = values
        self.update_kwargs = update_kwargs


class BulkDelete(BulkUD):
    """BulkUD which handles DELETEs."""

    def __init__(
        self,
        query: Query[Any],
        delete_kwargs: Optional[Dict[Any, Any]],
    ):
        super().__init__(query)
        self.delete_kwargs = delete_kwargs


class RowReturningQuery(Query[Row[_TP]]):
    if TYPE_CHECKING:

        def tuples(self) -> Query[_TP]:  # type: ignore
            ...

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\function.py ===
"""
There are three types of functions implemented in SymPy:

    1) defined functions (in the sense that they can be evaluated) like
       exp or sin; they have a name and a body:
           f = exp
    2) undefined function which have a name but no body. Undefined
       functions can be defined using a Function class as follows:
           f = Function('f')
       (the result will be a Function instance)
    3) anonymous function (or lambda function) which have a body (defined
       with dummy variables) but have no name:
           f = Lambda(x, exp(x)*x)
           f = Lambda((x, y), exp(x)*y)
    The fourth type of functions are composites, like (sin + cos)(x); these work in
    SymPy core, but are not yet part of SymPy.

    Examples
    ========

    >>> import sympy
    >>> f = sympy.Function("f")
    >>> from sympy.abc import x
    >>> f(x)
    f(x)
    >>> print(sympy.srepr(f(x).func))
    Function('f')
    >>> f(x).args
    (x,)

"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable
import copyreg

from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify

from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning, ignore_warnings)
from sympy.utilities.iterables import (has_dups, sift, iterable,
    is_sequence, uniq, topological_sort)
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name

import mpmath
from mpmath.libmp.libmpf import prec_to_dps

import inspect
from collections import Counter

def _coeff_isneg(a):
    """Return True if the leading Number is negative.

    Examples
    ========

    >>> from sympy.core.function import _coeff_isneg
    >>> from sympy import S, Symbol, oo, pi
    >>> _coeff_isneg(-3*pi)
    True
    >>> _coeff_isneg(S(3))
    False
    >>> _coeff_isneg(-oo)
    True
    >>> _coeff_isneg(Symbol('n', negative=True)) # coeff is 1
    False

    For matrix expressions:

    >>> from sympy import MatrixSymbol, sqrt
    >>> A = MatrixSymbol("A", 3, 3)
    >>> _coeff_isneg(-sqrt(2)*A)
    True
    >>> _coeff_isneg(sqrt(2)*A)
    False
    """

    if a.is_MatMul:
        a = a.args[0]
    if a.is_Mul:
        a = a.args[0]
    return a.is_Number and a.is_extended_negative


class PoleError(Exception):
    pass


class ArgumentIndexError(ValueError):
    def __str__(self):
        return ("Invalid operation with argument number %s for Function %s" %
               (self.args[1], self.args[0]))


class BadSignatureError(TypeError):
    '''Raised when a Lambda is created with an invalid signature'''
    pass


class BadArgumentsError(TypeError):
    '''Raised when a Lambda is called with an incorrect number of arguments'''
    pass


# Python 3 version that does not raise a Deprecation warning
def arity(cls):
    """Return the arity of the function if it is known, else None.

    Explanation
    ===========

    When default values are specified for some arguments, they are
    optional and the arity is reported as a tuple of possible values.

    Examples
    ========

    >>> from sympy import arity, log
    >>> arity(lambda x: x)
    1
    >>> arity(log)
    (1, 2)
    >>> arity(lambda *x: sum(x)) is None
    True
    """
    eval_ = getattr(cls, 'eval', cls)

    parameters = inspect.signature(eval_).parameters.items()
    if [p for _, p in parameters if p.kind == p.VAR_POSITIONAL]:
        return
    p_or_k = [p for _, p in parameters if p.kind == p.POSITIONAL_OR_KEYWORD]
    # how many have no default and how many have a default value
    no, yes = map(len, sift(p_or_k,
        lambda p:p.default == p.empty, binary=True))
    return no if not yes else tuple(range(no, no + yes + 1))

class FunctionClass(type):
    """
    Base class for function classes. FunctionClass is a subclass of type.

    Use Function('<function name>' [ , signature ]) to create
    undefined function classes.
    """
    _new = type.__new__

    def __init__(cls, *args, **kwargs):
        # honor kwarg value or class-defined value before using
        # the number of arguments in the eval function (if present)
        nargs = kwargs.pop('nargs', cls.__dict__.get('nargs', arity(cls)))
        if nargs is None and 'nargs' not in cls.__dict__:
            for supcls in cls.__mro__:
                if hasattr(supcls, '_nargs'):
                    nargs = supcls._nargs
                    break
                else:
                    continue

        # Canonicalize nargs here; change to set in nargs.
        if is_sequence(nargs):
            if not nargs:
                raise ValueError(filldedent('''
                    Incorrectly specified nargs as %s:
                    if there are no arguments, it should be
                    `nargs = 0`;
                    if there are any number of arguments,
                    it should be
                    `nargs = None`''' % str(nargs)))
            nargs = tuple(ordered(set(nargs)))
        elif nargs is not None:
            nargs = (as_int(nargs),)
        cls._nargs = nargs

        # When __init__ is called from UndefinedFunction it is called with
        # just one arg but when it is called from subclassing Function it is
        # called with the usual (name, bases, namespace) type() signature.
        if len(args) == 3:
            namespace = args[2]
            if 'eval' in namespace and not isinstance(namespace['eval'], classmethod):
                raise TypeError("eval on Function subclasses should be a class method (defined with @classmethod)")

    @property
    def __signature__(self):
        """
        Allow Python 3's inspect.signature to give a useful signature for
        Function subclasses.
        """
        # Python 3 only, but backports (like the one in IPython) still might
        # call this.
        try:
            from inspect import signature
        except ImportError:
            return None

        # TODO: Look at nargs
        return signature(self.eval)

    @property
    def free_symbols(self):
        return set()

    @property
    def xreplace(self):
        # Function needs args so we define a property that returns
        # a function that takes args...and then use that function
        # to return the right value
        return lambda rule, **_: rule.get(self, self)

    @property
    def nargs(self):
        """Return a set of the allowed number of arguments for the function.

        Examples
        ========

        >>> from sympy import Function
        >>> f = Function('f')

        If the function can take any number of arguments, the set of whole
        numbers is returned:

        >>> Function('f').nargs
        Naturals0

        If the function was initialized to accept one or more arguments, a
        corresponding set will be returned:

        >>> Function('f', nargs=1).nargs
        {1}
        >>> Function('f', nargs=(2, 1)).nargs
        {1, 2}

        The undefined function, after application, also has the nargs
        attribute; the actual number of arguments is always available by
        checking the ``args`` attribute:

        >>> f = Function('f')
        >>> f(1).nargs
        Naturals0
        >>> len(f(1).args)
        1
        """
        from sympy.sets.sets import FiniteSet
        # XXX it would be nice to handle this in __init__ but there are import
        # problems with trying to import FiniteSet there
        return FiniteSet(*self._nargs) if self._nargs else S.Naturals0

    def _valid_nargs(self, n : int) -> bool:
        """ Return True if the specified integer is a valid number of arguments

        The number of arguments n is guaranteed to be an integer and positive

        """
        if self._nargs:
            return n in self._nargs

        nargs = self.nargs
        return nargs is S.Naturals0 or n in nargs

    def __repr__(cls):
        return cls.__name__


class Application(Basic, metaclass=FunctionClass):
    """
    Base class for applied functions.

    Explanation
    ===========

    Instances of Application represent the result of applying an application of
    any type to any object.
    """

    is_Function = True

    @cacheit
    def __new__(cls, *args, **options):
        from sympy.sets.fancysets import Naturals0
        from sympy.sets.sets import FiniteSet

        args = list(map(sympify, args))
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        # WildFunction (and anything else like it) may have nargs defined
        # and we throw that value away here
        options.pop('nargs', None)

        if options:
            raise ValueError("Unknown options: %s" % options)

        if evaluate:
            evaluated = cls.eval(*args)
            if evaluated is not None:
                return evaluated

        obj = super().__new__(cls, *args, **options)

        # make nargs uniform here
        sentinel = object()
        objnargs = getattr(obj, "nargs", sentinel)
        if objnargs is not sentinel:
            # things passing through here:
            #  - functions subclassed from Function (e.g. myfunc(1).nargs)
            #  - functions like cos(1).nargs
            #  - AppliedUndef with given nargs like Function('f', nargs=1)(1).nargs
            # Canonicalize nargs here
            if is_sequence(objnargs):
                nargs = tuple(ordered(set(objnargs)))
            elif objnargs is not None:
                nargs = (as_int(objnargs),)
            else:
                nargs = None
        else:
            # things passing through here:
            #  - WildFunction('f').nargs
            #  - AppliedUndef with no nargs like Function('f')(1).nargs
            nargs = obj._nargs  # note the underscore here
        # convert to FiniteSet
        obj.nargs = FiniteSet(*nargs) if nargs else Naturals0()
        return obj

    @classmethod
    def eval(cls, *args):
        """
        Returns a canonical form of cls applied to arguments args.

        Explanation
        ===========

        The ``eval()`` method is called when the class ``cls`` is about to be
        instantiated and it should return either some simplified instance
        (possible of some other class), or if the class ``cls`` should be
        unmodified, return None.

        Examples of ``eval()`` for the function "sign"

        .. code-block:: python

            @classmethod
            def eval(cls, arg):
                if arg is S.NaN:
                    return S.NaN
                if arg.is_zero: return S.Zero
                if arg.is_positive: return S.One
                if arg.is_negative: return S.NegativeOne
                if isinstance(arg, Mul):
                    coeff, terms = arg.as_coeff_Mul(rational=True)
                    if coeff is not S.One:
                        return cls(coeff) * cls(terms)

        """
        return

    @property
    def func(self):
        return self.__class__

    def _eval_subs(self, old, new):
        if (old.is_Function and new.is_Function and
            callable(old) and callable(new) and
            old == self.func and len(self.args) in new.nargs):
            return new(*[i._subs(old, new) for i in self.args])


class Function(Application, Expr):
    r"""
    Base class for applied mathematical functions.

    It also serves as a constructor for undefined function classes.

    See the :ref:`custom-functions` guide for details on how to subclass
    ``Function`` and what methods can be defined.

    Examples
    ========

    **Undefined Functions**

    To create an undefined function, pass a string of the function name to
    ``Function``.

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> g = Function('g')(x)
    >>> f
    f
    >>> f(x)
    f(x)
    >>> g
    g(x)
    >>> f(x).diff(x)
    Derivative(f(x), x)
    >>> g.diff(x)
    Derivative(g(x), x)

    Assumptions can be passed to ``Function`` the same as with a
    :class:`~.Symbol`. Alternatively, you can use a ``Symbol`` with
    assumptions for the function name and the function will inherit the name
    and assumptions associated with the ``Symbol``:

    >>> f_real = Function('f', real=True)
    >>> f_real(x).is_real
    True
    >>> f_real_inherit = Function(Symbol('f', real=True))
    >>> f_real_inherit(x).is_real
    True

    Note that assumptions on a function are unrelated to the assumptions on
    the variables it is called on. If you want to add a relationship, subclass
    ``Function`` and define custom assumptions handler methods. See the
    :ref:`custom-functions-assumptions` section of the :ref:`custom-functions`
    guide for more details.

    **Custom Function Subclasses**

    The :ref:`custom-functions` guide has several
    :ref:`custom-functions-complete-examples` of how to subclass ``Function``
    to create a custom function.

    """

    @property
    def _diff_wrt(self):
        return False

    @cacheit
    def __new__(cls, *args, **options) -> type[AppliedUndef]:  # type: ignore
        # Handle calls like Function('f')
        if cls is Function:
            return UndefinedFunction(*args, **options)  # type: ignore
        else:
            return cls._new_(*args, **options)  # type: ignore

    @classmethod
    def _new_(cls, *args, **options) -> Expr:
        n = len(args)

        if not cls._valid_nargs(n):
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            temp = ('%(name)s takes %(qual)s %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise TypeError(temp % {
                'name': cls,
                'qual': 'exactly' if len(cls.nargs) == 1 else 'at least',
                'args': min(cls.nargs),
                'plural': 's'*(min(cls.nargs) != 1),
                'given': n})

        evaluate = options.get('evaluate', global_parameters.evaluate)
        result = super().__new__(cls, *args, **options)
        if evaluate and isinstance(result, cls) and result.args:
            _should_evalf = [cls._should_evalf(a) for a in result.args]
            pr2 = min(_should_evalf)
            if pr2 > 0:
                pr = max(_should_evalf)
                result = result.evalf(prec_to_dps(pr))

        return _sympify(result)

    @classmethod
    def _should_evalf(cls, arg):
        """
        Decide if the function should automatically evalf().

        Explanation
        ===========

        By default (in this implementation), this happens if (and only if) the
        ARG is a floating point number (including complex numbers).
        This function is used by __new__.

        Returns the precision to evalf to, or -1 if it should not evalf.
        """
        if arg.is_Float:
            return arg._prec
        if not arg.is_Add:
            return -1
        m = pure_complex(arg)
        if m is None:
            return -1
        # the elements of m are of type Number, so have a _prec
        return max(m[0]._prec, m[1]._prec)

    @classmethod
    def class_key(cls):
        from sympy.sets.fancysets import Naturals0
        funcs = {
            'exp': 10,
            'log': 11,
            'sin': 20,
            'cos': 21,
            'tan': 22,
            'cot': 23,
            'sinh': 30,
            'cosh': 31,
            'tanh': 32,
            'coth': 33,
            'conjugate': 40,
            're': 41,
            'im': 42,
            'arg': 43,
        }
        name = cls.__name__

        try:
            i = funcs[name]
        except KeyError:
            i = 0 if isinstance(cls.nargs, Naturals0) else 10000

        return 4, i, name

    def _eval_evalf(self, prec):

        def _get_mpmath_func(fname):
            """Lookup mpmath function based on name"""
            if isinstance(self, AppliedUndef):
                # Shouldn't lookup in mpmath but might have ._imp_
                return None

            if not hasattr(mpmath, fname):
                fname = MPMATH_TRANSLATIONS.get(fname, None)
                if fname is None:
                    return None
            return getattr(mpmath, fname)

        _eval_mpmath = getattr(self, '_eval_mpmath', None)
        if _eval_mpmath is None:
            func = _get_mpmath_func(self.func.__name__)
            args = self.args
        else:
            func, args = _eval_mpmath()

        # Fall-back evaluation
        if func is None:
            imp = getattr(self, '_imp_', None)
            if imp is None:
                return None
            try:
                return Float(imp(*[i.evalf(prec) for i in self.args]), prec)
            except (TypeError, ValueError):
                return None

        # Convert all args to mpf or mpc
        # Convert the arguments to *higher* precision than requested for the
        # final result.
        # XXX + 5 is a guess, it is similar to what is used in evalf.py. Should
        #     we be more intelligent about it?
        try:
            args = [arg._to_mpmath(prec + 5) for arg in args]
            def bad(m):
                from mpmath import mpf, mpc
                # the precision of an mpf value is the last element
                # if that is 1 (and m[1] is not 1 which would indicate a
                # power of 2), then the eval failed; so check that none of
                # the arguments failed to compute to a finite precision.
                # Note: An mpc value has two parts, the re and imag tuple;
                # check each of those parts, too. Anything else is allowed to
                # pass
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] !=1 and m[-1] == 1
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] !=1 and m[-1] == 1 and \
                        n[1] !=1 and n[-1] == 1
                else:
                    return False
            if any(bad(a) for a in args):
                raise ValueError  # one or more args failed to compute with significance
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = func(*args)

        return Expr._from_mpmath(v, prec)

    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
            i += 1
            da = a.diff(s)
            if da.is_zero:
                continue
            try:
                df = self.fdiff(i)
            except ArgumentIndexError:
                df = Function.fdiff(self, i)
            l.append(df * da)
        return Add(*l)

    def _eval_is_commutative(self):
        return fuzzy_and(a.is_commutative for a in self.args)

    def _eval_is_meromorphic(self, x, a):
        if not self.args:
            return True
        if any(arg.has(x) for arg in self.args[1:]):
            return False

        arg = self.args[0]
        if not arg._eval_is_meromorphic(x, a):
            return None

        return fuzzy_not(type(self).is_singular(arg.subs(x, a)))

    _singularities: FuzzyBool | tuple[Expr, ...] = None

    @classmethod
    def is_singular(cls, a):
        """
        Tests whether the argument is an essential singularity
        or a branch point, or the functions is non-holomorphic.
        """
        ss = cls._singularities
        if ss in (True, None, False):
            return ss

        return fuzzy_or(a.is_infinite if s is S.ComplexInfinity
                        else (a - s).is_zero for s in ss)

    def _eval_aseries(self, n, args0, x, logx):
        """
        Compute an asymptotic expansion around args0, in terms of self.args.
        This function is only used internally by _eval_nseries and should not
        be called directly; derived classes can overwrite this to implement
        asymptotic expansions.
        """
        raise PoleError(filldedent('''
            Asymptotic expansion of %s around %s is
            not implemented.''' % (type(self), args0)))

    def _eval_nseries(self, x, n, logx, cdir=0):
        """
        This function does compute series for multivariate functions,
        but the expansion is always in terms of *one* variable.

        Examples
        ========

        >>> from sympy import atan2
        >>> from sympy.abc import x, y
        >>> atan2(x, y).series(x, n=2)
        atan2(0, y) + x/y + O(x**2)
        >>> atan2(x, y).series(y, n=2)
        -y/x + atan2(x, 0) + O(y**2)

        This function also computes asymptotic expansions, if necessary
        and possible:

        >>> from sympy import loggamma
        >>> loggamma(1/x)._eval_nseries(x,0,None)
        -1/x - log(x)/x + log(x)/2 + O(1)

        """
        from .symbol import uniquely_named_symbol
        from sympy.series.order import Order
        from sympy.sets.sets import FiniteSet
        args = self.args
        args0 = [t.limit(x, 0) for t in args]
        if any(t.is_finite is False for t in args0):
            from .numbers import oo, zoo, nan
            a = [t.as_leading_term(x, logx=logx) for t in args]
            a0 = [t.limit(x, 0) for t in a]
            if any(t.has(oo, -oo, zoo, nan) for t in a0):
                return self._eval_aseries(n, args0, x, logx)
            # Careful: the argument goes to oo, but only logarithmically so. We
            # are supposed to do a power series expansion "around the
            # logarithmic term". e.g.
            #      f(1+x+log(x))
            #     -> f(1+logx) + x*f'(1+logx) + O(x**2)
            # where 'logx' is given in the argument
            a = [t._eval_nseries(x, n, logx) for t in args]
            z = [r - r0 for (r, r0) in zip(a, a0)]
            p = [Dummy() for _ in z]
            q = []
            v = None
            for ai, zi, pi in zip(a0, z, p):
                if zi.has(x):
                    if v is not None:
                        raise NotImplementedError
                    q.append(ai + pi)
                    v = pi
                else:
                    q.append(ai)
            e1 = self.func(*q)
            if v is None:
                return e1
            s = e1._eval_nseries(v, n, logx)
            o = s.getO()
            s = s.removeO()
            s = s.subs(v, zi).expand() + Order(o.expr.subs(v, zi), x)
            return s
        if (self.func.nargs is S.Naturals0
                or (self.func.nargs == FiniteSet(1) and args0[0])
                or any(c > 1 for c in self.func.nargs)):
            e = self
            e1 = e.expand()
            if e == e1:
                #for example when e = sin(x+1) or e = sin(cos(x))
                #let's try the general algorithm
                if len(e.args) == 1:
                    # issue 14411
                    e = e.func(e.args[0].cancel())
                term = e.subs(x, S.Zero)
                if term.is_finite is False or term is S.NaN:
                    raise PoleError("Cannot expand %s around 0" % (self))
                series = term
                fact = S.One

                _x = uniquely_named_symbol('xi', self)
                e = e.subs(x, _x)
                for i in range(1, n):
                    fact *= Rational(i)
                    e = e.diff(_x)
                    subs = e.subs(_x, S.Zero)
                    if subs is S.NaN:
                        # try to evaluate a limit if we have to
                        subs = e.limit(_x, S.Zero)
                    if subs.is_finite is False:
                        raise PoleError("Cannot expand %s around 0" % (self))
                    term = subs*(x**i)/fact
                    term = term.expand()
                    series += term
                return series + Order(x**n, x)
            return e1.nseries(x, n=n, logx=logx)
        arg = self.args[0]
        l = []
        g = None
        # try to predict a number of terms needed
        nterms = n + 2
        cf = Order(arg.as_leading_term(x), x).getn()
        if cf != 0:
            nterms = (n/cf).ceiling()
        for i in range(nterms):
            g = self.taylor_term(i, arg, g)
            g = g.nseries(x, n=n, logx=logx)
            l.append(g)
        return Add(*l) + Order(x**n, x)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        ix = argindex - 1
        A = self.args[ix]
        if A._diff_wrt:
            if len(self.args) == 1 or not A.is_Symbol:
                return _derivative_dispatch(self, A)
            for i, v in enumerate(self.args):
                if i != ix and A in v.free_symbols:
                    # it can't be in any other argument's free symbols
                    # issue 8510
                    break
            else:
                return _derivative_dispatch(self, A)

        # See issue 4624 and issue 4719, 5600 and 8510
        D = Dummy('xi_%i' % argindex, dummy_index=hash(A))
        args = self.args[:ix] + (D,) + self.args[ix + 1:]
        return Subs(Derivative(self.func(*args), D), D, A)

    def _eval_as_leading_term(self, x, logx, cdir):
        """Stub that should be overridden by new Functions to return
        the first non-zero term in a series if ever an x-dependent
        argument whose leading term vanishes as x -> 0 might be encountered.
        See, for example, cos._eval_as_leading_term.
        """
        from sympy.series.order import Order
        args = [a.as_leading_term(x, logx=logx) for a in self.args]
        o = Order(1, x)
        if any(x in a.free_symbols and o.contains(a) for a in args):
            # Whereas x and any finite number are contained in O(1, x),
            # expressions like 1/x are not. If any arg simplified to a
            # vanishing expression as x -> 0 (like x or x**2, but not
            # 3, 1/x, etc...) then the _eval_as_leading_term is needed
            # to supply the first non-zero term of the series,
            #
            # e.g. expression    leading term
            #      ----------    ------------
            #      cos(1/x)      cos(1/x)
            #      cos(cos(x))   cos(1)
            #      cos(x)        1        <- _eval_as_leading_term needed
            #      sin(x)        x        <- _eval_as_leading_term needed
            #
            raise NotImplementedError(
                '%s has no _eval_as_leading_term routine' % self.func)
        else:
            return self


class DefinedFunction(Function):
    """Base class for defined functions like ``sin``, ``cos``, ..."""

    @cacheit
    def __new__(cls, *args, **options) -> Expr:  # type: ignore
        return cls._new_(*args, **options)


class AppliedUndef(Function):
    """
    Base class for expressions resulting from the application of an undefined
    function.
    """

    is_number = False

    name: str

    def __new__(cls, *args, **options) -> Expr:  # type: ignore
        args = tuple(map(sympify, args))
        u = [a.name for a in args if isinstance(a, UndefinedFunction)]
        if u:
            raise TypeError('Invalid argument: expecting an expression, not UndefinedFunction%s: %s' % (
                's'*(len(u) > 1), ', '.join(u)))
        obj: Expr = super().__new__(cls, *args, **options)  # type: ignore
        return obj

    def _eval_as_leading_term(self, x, logx, cdir):
        return self

    @property
    def _diff_wrt(self):
        """
        Allow derivatives wrt to undefined functions.

        Examples
        ========

        >>> from sympy import Function, Symbol
        >>> f = Function('f')
        >>> x = Symbol('x')
        >>> f(x)._diff_wrt
        True
        >>> f(x).diff(x)
        Derivative(f(x), x)
        """
        return True


class UndefSageHelper:
    """
    Helper to facilitate Sage conversion.
    """
    def __get__(self, ins, typ):
        import sage.all as sage
        if ins is None:
            return lambda: sage.function(typ.__name__)
        else:
            args = [arg._sage_() for arg in ins.args]
            return lambda : sage.function(ins.__class__.__name__)(*args)

_undef_sage_helper = UndefSageHelper()


class UndefinedFunction(FunctionClass):
    """
    The (meta)class of undefined functions.
    """
    name: str
    _sage_: UndefSageHelper

    def __new__(mcl, name, bases=(AppliedUndef,), __dict__=None, **kwargs) -> type[AppliedUndef]:
        from .symbol import _filter_assumptions
        # Allow Function('f', real=True)
        # and/or Function(Symbol('f', real=True))
        assumptions, kwargs = _filter_assumptions(kwargs)
        if isinstance(name, Symbol):
            assumptions = name._merge(assumptions)
            name = name.name
        elif not isinstance(name, str):
            raise TypeError('expecting string or Symbol for name')
        else:
            commutative = assumptions.get('commutative', None)
            assumptions = Symbol(name, **assumptions).assumptions0
            if commutative is None:
                assumptions.pop('commutative')
        __dict__ = __dict__ or {}
        # put the `is_*` for into __dict__
        __dict__.update({'is_%s' % k: v for k, v in assumptions.items()})
        # You can add other attributes, although they do have to be hashable
        # (but seriously, if you want to add anything other than assumptions,
        # just subclass Function)
        __dict__.update(kwargs)
        # add back the sanitized assumptions without the is_ prefix
        kwargs.update(assumptions)
        # Save these for __eq__
        __dict__.update({'_kwargs': kwargs})
        # do this for pickling
        __dict__['__module__'] = None
        obj = super().__new__(mcl, name, bases, __dict__)  # type: ignore
        obj.name = name
        obj._sage_ = _undef_sage_helper
        return obj  # type: ignore

    def __instancecheck__(cls, instance):
        return cls in type(instance).__mro__

    _kwargs: dict[str, bool | None] = {}

    def __hash__(self):
        return hash((self.class_key(), frozenset(self._kwargs.items())))

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            self.class_key() == other.class_key() and
            self._kwargs == other._kwargs)

    def __ne__(self, other):
        return not self == other

    @property
    def _diff_wrt(self):
        return False


# Using copyreg is the only way to make a dynamically generated instance of a
# metaclass picklable without using a custom pickler. It is not possible to
# define e.g. __reduce__ on the metaclass because obj.__reduce__ will retrieve
# the __reduce__ method for reducing instances of the type rather than for the
# type itself.
def _reduce_undef(f):
    return (_rebuild_undef, (f.name, f._kwargs))

def _rebuild_undef(name, kwargs):
    return Function(name, **kwargs)

copyreg.pickle(UndefinedFunction, _reduce_undef)


# XXX: The type: ignore on WildFunction is because mypy complains:
#
# sympy/core/function.py:939: error: Cannot determine type of 'sort_key' in
# base class 'Expr'
#
# Somehow this is because of the @cacheit decorator but it is not clear how to
# fix it.


class WildFunction(Function, AtomicExpr):  # type: ignore
    """
    A WildFunction function matches any function (with its arguments).

    Examples
    ========

    >>> from sympy import WildFunction, Function, cos
    >>> from sympy.abc import x, y
    >>> F = WildFunction('F')
    >>> f = Function('f')
    >>> F.nargs
    Naturals0
    >>> x.match(F)
    >>> F.match(F)
    {F_: F_}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> cos(x).match(F)
    {F_: cos(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a given number of arguments, set ``nargs`` to the
    desired value at instantiation:

    >>> F = WildFunction('F', nargs=2)
    >>> F.nargs
    {2}
    >>> f(x).match(F)
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a range of arguments, set ``nargs`` to a tuple
    containing the desired number of arguments, e.g. if ``nargs = (1, 2)``
    then functions with 1 or 2 arguments will be matched.

    >>> F = WildFunction('F', nargs=(1, 2))
    >>> F.nargs
    {1, 2}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}
    >>> f(x, y, 1).match(F)

    """

    # XXX: What is this class attribute used for?
    include: set[Any] = set()

    def __init__(cls, name, **assumptions):
        from sympy.sets.sets import Set, FiniteSet
        cls.name = name
        nargs = assumptions.pop('nargs', S.Naturals0)
        if not isinstance(nargs, Set):
            # Canonicalize nargs here.  See also FunctionClass.
            if is_sequence(nargs):
                nargs = tuple(ordered(set(nargs)))
            elif nargs is not None:
                nargs = (as_int(nargs),)
            nargs = FiniteSet(*nargs)
        cls.nargs = nargs

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, (AppliedUndef, Function)):
            return None
        if len(expr.args) not in self.nargs:
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        repl_dict[self] = expr
        return repl_dict


class Derivative(Expr):
    """
    Carries out differentiation of the given expression with respect to symbols.

    Examples
    ========

    >>> from sympy import Derivative, Function, symbols, Subs
    >>> from sympy.abc import x, y
    >>> f, g = symbols('f g', cls=Function)

    >>> Derivative(x**2, x, evaluate=True)
    2*x

    Denesting of derivatives retains the ordering of variables:

        >>> Derivative(Derivative(f(x, y), y), x)
        Derivative(f(x, y), y, x)

    Contiguously identical symbols are merged into a tuple giving
    the symbol and the count:

        >>> Derivative(f(x), x, x, y, x)
        Derivative(f(x), (x, 2), y, x)

    If the derivative cannot be performed, and evaluate is True, the
    order of the variables of differentiation will be made canonical:

        >>> Derivative(f(x, y), y, x, evaluate=True)
        Derivative(f(x, y), x, y)

    Derivatives with respect to undefined functions can be calculated:

        >>> Derivative(f(x)**2, f(x), evaluate=True)
        2*f(x)

    Such derivatives will show up when the chain rule is used to
    evaluate a derivative:

        >>> f(g(x)).diff(x)
        Derivative(f(g(x)), g(x))*Derivative(g(x), x)

    Substitution is used to represent derivatives of functions with
    arguments that are not symbols or functions:

        >>> f(2*x + 3).diff(x) == 2*Subs(f(y).diff(y), y, 2*x + 3)
        True

    Notes
    =====

    Simplification of high-order derivatives:

    Because there can be a significant amount of simplification that can be
    done when multiple differentiations are performed, results will be
    automatically simplified in a fairly conservative fashion unless the
    keyword ``simplify`` is set to False.

        >>> from sympy import sqrt, diff, Function, symbols
        >>> from sympy.abc import x, y, z
        >>> f, g = symbols('f,g', cls=Function)

        >>> e = sqrt((x + 1)**2 + x)
        >>> diff(e, (x, 5), simplify=False).count_ops()
        136
        >>> diff(e, (x, 5)).count_ops()
        30

    Ordering of variables:

    If evaluate is set to True and the expression cannot be evaluated, the
    list of differentiation symbols will be sorted, that is, the expression is
    assumed to have continuous derivatives up to the order asked.

    Derivative wrt non-Symbols:

    For the most part, one may not differentiate wrt non-symbols.
    For example, we do not allow differentiation wrt `x*y` because
    there are multiple ways of structurally defining where x*y appears
    in an expression: a very strict definition would make
    (x*y*z).diff(x*y) == 0. Derivatives wrt defined functions (like
    cos(x)) are not allowed, either:

        >>> (x*y*z).diff(x*y)
        Traceback (most recent call last):
        ...
        ValueError: Can't calculate derivative wrt x*y.

    To make it easier to work with variational calculus, however,
    derivatives wrt AppliedUndef and Derivatives are allowed.
    For example, in the Euler-Lagrange method one may write
    F(t, u, v) where u = f(t) and v = f'(t). These variables can be
    written explicitly as functions of time::

        >>> from sympy.abc import t
        >>> F = Function('F')
        >>> U = f(t)
        >>> V = U.diff(t)

    The derivative wrt f(t) can be obtained directly:

        >>> direct = F(t, U, V).diff(U)

    When differentiation wrt a non-Symbol is attempted, the non-Symbol
    is temporarily converted to a Symbol while the differentiation
    is performed and the same answer is obtained:

        >>> indirect = F(t, U, V).subs(U, x).diff(x).subs(x, U)
        >>> assert direct == indirect

    The implication of this non-symbol replacement is that all
    functions are treated as independent of other functions and the
    symbols are independent of the functions that contain them::

        >>> x.diff(f(x))
        0
        >>> g(x).diff(f(x))
        0

    It also means that derivatives are assumed to depend only
    on the variables of differentiation, not on anything contained
    within the expression being differentiated::

        >>> F = f(x)
        >>> Fx = F.diff(x)
        >>> Fx.diff(F)  # derivative depends on x, not F
        0
        >>> Fxx = Fx.diff(x)
        >>> Fxx.diff(Fx)  # derivative depends on x, not Fx
        0

    The last example can be made explicit by showing the replacement
    of Fx in Fxx with y:

        >>> Fxx.subs(Fx, y)
        Derivative(y, x)

        Since that in itself will evaluate to zero, differentiating
        wrt Fx will also be zero:

        >>> _.doit()
        0

    Replacing undefined functions with concrete expressions

    One must be careful to replace undefined functions with expressions
    that contain variables consistent with the function definition and
    the variables of differentiation or else insconsistent result will
    be obtained. Consider the following example:

    >>> eq = f(x)*g(y)
    >>> eq.subs(f(x), x*y).diff(x, y).doit()
    y*Derivative(g(y), y) + g(y)
    >>> eq.diff(x, y).subs(f(x), x*y).doit()
    y*Derivative(g(y), y)

    The results differ because `f(x)` was replaced with an expression
    that involved both variables of differentiation. In the abstract
    case, differentiation of `f(x)` by `y` is 0; in the concrete case,
    the presence of `y` made that derivative nonvanishing and produced
    the extra `g(y)` term.

    Defining differentiation for an object

    An object must define ._eval_derivative(symbol) method that returns
    the differentiation result. This function only needs to consider the
    non-trivial case where expr contains symbol and it should call the diff()
    method internally (not _eval_derivative); Derivative should be the only
    one to call _eval_derivative.

    Any class can allow derivatives to be taken with respect to
    itself (while indicating its scalar nature). See the
    docstring of Expr._diff_wrt.

    See Also
    ========
    _sort_variable_count
    """

    is_Derivative = True

    @property
    def _diff_wrt(self):
        """An expression may be differentiated wrt a Derivative if
        it is in elementary form.

        Examples
        ========

        >>> from sympy import Function, Derivative, cos
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> Derivative(f(x), x)._diff_wrt
        True
        >>> Derivative(cos(x), x)._diff_wrt
        False
        >>> Derivative(x + 1, x)._diff_wrt
        False

        A Derivative might be an unevaluated form of what will not be
        a valid variable of differentiation if evaluated. For example,

        >>> Derivative(f(f(x)), x).doit()
        Derivative(f(x), x)*Derivative(f(f(x)), f(x))

        Such an expression will present the same ambiguities as arise
        when dealing with any other product, like ``2*x``, so ``_diff_wrt``
        is False:

        >>> Derivative(f(f(x)), x)._diff_wrt
        False
        """
        return self.expr._diff_wrt and isinstance(self.doit(), Derivative)

    def __new__(cls, expr, *variables, **kwargs):
        expr = sympify(expr)
        if not isinstance(expr, Basic):
            raise TypeError(f"Cannot represent derivative of {type(expr)}")
        symbols_or_none = getattr(expr, "free_symbols", None)
        has_symbol_set = isinstance(symbols_or_none, set)

        if not has_symbol_set:
            raise ValueError(filldedent('''
                Since there are no variables in the expression %s,
                it cannot be differentiated.''' % expr))

        # determine value for variables if it wasn't given
        if not variables:
            variables = expr.free_symbols
            if len(variables) != 1:
                if expr.is_number:
                    return S.Zero
                if len(variables) == 0:
                    raise ValueError(filldedent('''
                        Since there are no variables in the expression,
                        the variable(s) of differentiation must be supplied
                        to differentiate %s''' % expr))
                else:
                    raise ValueError(filldedent('''
                        Since there is more than one variable in the
                        expression, the variable(s) of differentiation
                        must be supplied to differentiate %s''' % expr))

        # Split the list of variables into a list of the variables we are diff
        # wrt, where each element of the list has the form (s, count) where
        # s is the entity to diff wrt and count is the order of the
        # derivative.
        variable_count = []
        array_likes = (tuple, list, Tuple)

        from sympy.tensor.array import Array, NDimArray

        for i, v in enumerate(variables):
            if isinstance(v, UndefinedFunction):
                raise TypeError(
                    "cannot differentiate wrt "
                    "UndefinedFunction: %s" % v)

            if isinstance(v, array_likes):
                if len(v) == 0:
                    # Ignore empty tuples: Derivative(expr, ... , (), ... )
                    continue
                if isinstance(v[0], array_likes):
                    # Derive by array: Derivative(expr, ... , [[x, y, z]], ... )
                    if len(v) == 1:
                        v = Array(v[0])
                        count = 1
                    else:
                        v, count = v
                        v = Array(v)
                else:
                    v, count = v
                if count == 0:
                    continue
                variable_count.append(Tuple(v, count))
                continue

            v = sympify(v)
            if isinstance(v, Integer):
                if i == 0:
                    raise ValueError("First variable cannot be a number: %i" % v)
                count = v
                prev, prevcount = variable_count[-1]
                if prevcount != 1:
                    raise TypeError("tuple {} followed by number {}".format((prev, prevcount), v))
                if count == 0:
                    variable_count.pop()
                else:
                    variable_count[-1] = Tuple(prev, count)
            else:
                count = 1
                variable_count.append(Tuple(v, count))

        # light evaluation of contiguous, identical
        # items: (x, 1), (x, 1) -> (x, 2)
        merged = []
        for t in variable_count:
            v, c = t
            if c.is_negative:
                raise ValueError(
                    'order of differentiation must be nonnegative')
            if merged and merged[-1][0] == v:
                c += merged[-1][1]
                if not c:
                    merged.pop()
                else:
                    merged[-1] = Tuple(v, c)
            else:
                merged.append(t)
        variable_count = merged

        # sanity check of variables of differentation; we waited
        # until the counts were computed since some variables may
        # have been removed because the count was 0
        for v, c in variable_count:
            # v must have _diff_wrt True
            if not v._diff_wrt:
                __ = ''  # filler to make error message neater
                raise ValueError(filldedent('''
                    Can't calculate derivative wrt %s.%s''' % (v,
                    __)))

        # We make a special case for 0th derivative, because there is no
        # good way to unambiguously print this.
        if len(variable_count) == 0:
            return expr

        evaluate = kwargs.get('evaluate', False)

        if evaluate:
            if isinstance(expr, Derivative):
                expr = expr.canonical
            variable_count = [
                (v.canonical if isinstance(v, Derivative) else v, c)
                for v, c in variable_count]

            # Look for a quick exit if there are symbols that don't appear in
            # expression at all. Note, this cannot check non-symbols like
            # Derivatives as those can be created by intermediate
            # derivatives.
            zero = False
            free = expr.free_symbols
            from sympy.matrices.expressions.matexpr import MatrixExpr

            for v, c in variable_count:
                vfree = v.free_symbols
                if c.is_positive and vfree:
                    if isinstance(v, AppliedUndef):
                        # these match exactly since
                        # x.diff(f(x)) == g(x).diff(f(x)) == 0
                        # and are not created by differentiation
                        D = Dummy()
                        if not expr.xreplace({v: D}).has(D):
                            zero = True
                            break
                    elif isinstance(v, MatrixExpr):
                        zero = False
                        break
                    elif isinstance(v, Symbol) and v not in free:
                        zero = True
                        break
                    else:
                        if not free & vfree:
                            # e.g. v is IndexedBase or Matrix
                            zero = True
                            break
            if zero:
                return cls._get_zero_with_shape_like(expr)

            # make the order of symbols canonical
            #TODO: check if assumption of discontinuous derivatives exist
            variable_count = cls._sort_variable_count(variable_count)

        # denest
        if isinstance(expr, Derivative):
            variable_count = list(expr.variable_count) + variable_count
            expr = expr.expr
            return _derivative_dispatch(expr, *variable_count, **kwargs)

        # we return here if evaluate is False or if there is no
        # _eval_derivative method
        if not evaluate or not hasattr(expr, '_eval_derivative'):
            # return an unevaluated Derivative
            if evaluate and variable_count == [(expr, 1)] and expr.is_scalar:
                # special hack providing evaluation for classes
                # that have defined is_scalar=True but have no
                # _eval_derivative defined
                return S.One
            return Expr.__new__(cls, expr, *variable_count)

        # evaluate the derivative by calling _eval_derivative method
        # of expr for each variable
        # -------------------------------------------------------------
        nderivs = 0  # how many derivatives were performed
        unhandled = []
        from sympy.matrices.matrixbase import MatrixBase
        for i, (v, count) in enumerate(variable_count):

            old_expr = expr
            old_v = None

            is_symbol = v.is_symbol or isinstance(v,
                (Iterable, Tuple, MatrixBase, NDimArray))

            if not is_symbol:
                old_v = v
                v = Dummy('xi')
                expr = expr.xreplace({old_v: v})
                # Derivatives and UndefinedFunctions are independent
                # of all others
                clashing = not (isinstance(old_v, (Derivative, AppliedUndef)))
                if v not in expr.free_symbols and not clashing:
                    return expr.diff(v)  # expr's version of 0
                if not old_v.is_scalar and not hasattr(
                        old_v, '_eval_derivative'):
                    # special hack providing evaluation for classes
                    # that have defined is_scalar=True but have no
                    # _eval_derivative defined
                    expr *= old_v.diff(old_v)

            obj = cls._dispatch_eval_derivative_n_times(expr, v, count)
            if obj is not None and obj.is_zero:
                return obj

            nderivs += count

            if old_v is not None:
                if obj is not None:
                    # remove the dummy that was used
                    obj = obj.subs(v, old_v)
                # restore expr
                expr = old_expr

            if obj is None:
                # we've already checked for quick-exit conditions
                # that give 0 so the remaining variables
                # are contained in the expression but the expression
                # did not compute a derivative so we stop taking
                # derivatives
                unhandled = variable_count[i:]
                break

            expr = obj

        # what we have so far can be made canonical
        expr = expr.replace(
            lambda x: isinstance(x, Derivative),
            lambda x: x.canonical)

        if unhandled:
            if isinstance(expr, Derivative):
                unhandled = list(expr.variable_count) + unhandled
                expr = expr.expr
            expr = Expr.__new__(cls, expr, *unhandled)

        if (nderivs > 1) == True and kwargs.get('simplify', True):
            from .exprtools import factor_terms
            from sympy.simplify.simplify import signsimp
            expr = factor_terms(signsimp(expr))
        return expr

    @property
    def canonical(cls):
        return cls.func(cls.expr,
            *Derivative._sort_variable_count(cls.variable_count))

    @classmethod
    def _sort_variable_count(cls, vc):
        """
        Sort (variable, count) pairs into canonical order while
        retaining order of variables that do not commute during
        differentiation:

        * symbols and functions commute with each other
        * derivatives commute with each other
        * a derivative does not commute with anything it contains
        * any other object is not allowed to commute if it has
          free symbols in common with another object

        Examples
        ========

        >>> from sympy import Derivative, Function, symbols
        >>> vsort = Derivative._sort_variable_count
        >>> x, y, z = symbols('x y z')
        >>> f, g, h = symbols('f g h', cls=Function)

        Contiguous items are collapsed into one pair:

        >>> vsort([(x, 1), (x, 1)])
        [(x, 2)]
        >>> vsort([(y, 1), (f(x), 1), (y, 1), (f(x), 1)])
        [(y, 2), (f(x), 2)]

        Ordering is canonical.

        >>> def vsort0(*v):
        ...     # docstring helper to
        ...     # change vi -> (vi, 0), sort, and return vi vals
        ...     return [i[0] for i in vsort([(i, 0) for i in v])]

        >>> vsort0(y, x)
        [x, y]
        >>> vsort0(g(y), g(x), f(y))
        [f(y), g(x), g(y)]

        Symbols are sorted as far to the left as possible but never
        move to the left of a derivative having the same symbol in
        its variables; the same applies to AppliedUndef which are
        always sorted after Symbols:

        >>> dfx = f(x).diff(x)
        >>> assert vsort0(dfx, y) == [y, dfx]
        >>> assert vsort0(dfx, x) == [dfx, x]
        """
        if not vc:
            return []
        vc = list(vc)
        if len(vc) == 1:
            return [Tuple(*vc[0])]
        V = list(range(len(vc)))
        E = []
        v = lambda i: vc[i][0]
        D = Dummy()
        def _block(d, v, wrt=False):
            # return True if v should not come before d else False
            if d == v:
                return wrt
            if d.is_Symbol:
                return False
            if isinstance(d, Derivative):
                # a derivative blocks if any of it's variables contain
                # v; the wrt flag will return True for an exact match
                # and will cause an AppliedUndef to block if v is in
                # the arguments
                if any(_block(k, v, wrt=True)
                        for k in d._wrt_variables):
                    return True
                return False
            if not wrt and isinstance(d, AppliedUndef):
                return False
            if v.is_Symbol:
                return v in d.free_symbols
            if isinstance(v, AppliedUndef):
                return _block(d.xreplace({v: D}), D)
            return d.free_symbols & v.free_symbols
        for i in range(len(vc)):
            for j in range(i):
                if _block(v(j), v(i)):
                    E.append((j,i))
        # this is the default ordering to use in case of ties
        O = dict(zip(ordered(uniq([i for i, c in vc])), range(len(vc))))
        ix = topological_sort((V, E), key=lambda i: O[v(i)])
        # merge counts of contiguously identical items
        merged = []
        for v, c in [vc[i] for i in ix]:
            if merged and merged[-1][0] == v:
                merged[-1][1] += c
            else:
                merged.append([v, c])
        return [Tuple(*i) for i in merged]

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def _eval_derivative(self, v):
        # If v (the variable of differentiation) is not in
        # self.variables, we might be able to take the derivative.
        if v not in self._wrt_variables:
            dedv = self.expr.diff(v)
            if isinstance(dedv, Derivative):
                return dedv.func(dedv.expr, *(self.variable_count + dedv.variable_count))
            # dedv (d(self.expr)/dv) could have simplified things such that the
            # derivative wrt things in self.variables can now be done. Thus,
            # we set evaluate=True to see if there are any other derivatives
            # that can be done. The most common case is when dedv is a simple
            # number so that the derivative wrt anything else will vanish.
            return self.func(dedv, *self.variables, evaluate=True)
        # In this case v was in self.variables so the derivative wrt v has
        # already been attempted and was not computed, either because it
        # couldn't be or evaluate=False originally.
        variable_count = list(self.variable_count)
        variable_count.append((v, 1))
        return self.func(self.expr, *variable_count, evaluate=False)

    def doit(self, **hints):
        expr = self.expr
        if hints.get('deep', True):
            expr = expr.doit(**hints)
        hints['evaluate'] = True
        rv = self.func(expr, *self.variable_count, **hints)
        if rv!= self and rv.has(Derivative):
            rv =  rv.doit(**hints)
        return rv

    @_sympifyit('z0', NotImplementedError)
    def doit_numerically(self, z0):
        """
        Evaluate the derivative at z numerically.

        When we can represent derivatives at a point, this should be folded
        into the normal evalf. For now, we need a special method.
        """
        if len(self.free_symbols) != 1 or len(self.variables) != 1:
            raise NotImplementedError('partials and higher order derivatives')
        z = list(self.free_symbols)[0]

        def eval(x):
            f0 = self.expr.subs(z, Expr._from_mpmath(x, prec=mpmath.mp.prec))
            f0 = f0.evalf(prec_to_dps(mpmath.mp.prec))
            return f0._to_mpmath(mpmath.mp.prec)
        return Expr._from_mpmath(mpmath.diff(eval,
                                             z0._to_mpmath(mpmath.mp.prec)),
                                 mpmath.mp.prec)

    @property
    def expr(self):
        return self._args[0]

    @property
    def _wrt_variables(self):
        # return the variables of differentiation without
        # respect to the type of count (int or symbolic)
        return [i[0] for i in self.variable_count]

    @property
    def variables(self):
        # TODO: deprecate?  YES, make this 'enumerated_variables' and
        #       name _wrt_variables as variables
        # TODO: support for `d^n`?
        rv = []
        for v, count in self.variable_count:
            if not count.is_Integer:
                raise TypeError(filldedent('''
                Cannot give expansion for symbolic count. If you just
                want a list of all variables of differentiation, use
                _wrt_variables.'''))
            rv.extend([v]*count)
        return tuple(rv)

    @property
    def variable_count(self):
        return self._args[1:]

    @property
    def derivative_count(self):
        return sum([count for _, count in self.variable_count], 0)

    @property
    def free_symbols(self):
        ret = self.expr.free_symbols
        # Add symbolic counts to free_symbols
        for _, count in self.variable_count:
            ret.update(count.free_symbols)
        return ret

    @property
    def kind(self):
        return self.args[0].kind

    def _eval_subs(self, old, new):
        # The substitution (old, new) cannot be done inside
        # Derivative(expr, vars) for a variety of reasons
        # as handled below.
        if old in self._wrt_variables:
            # first handle the counts
            expr = self.func(self.expr, *[(v, c.subs(old, new))
                for v, c in self.variable_count])
            if expr != self:
                return expr._eval_subs(old, new)
            # quick exit case
            if not getattr(new, '_diff_wrt', False):
                # case (0): new is not a valid variable of
                # differentiation
                if isinstance(old, Symbol):
                    # don't introduce a new symbol if the old will do
                    return Subs(self, old, new)
                else:
                    xi = Dummy('xi')
                    return Subs(self.xreplace({old: xi}), xi, new)

        # If both are Derivatives with the same expr, check if old is
        # equivalent to self or if old is a subderivative of self.
        if old.is_Derivative and old.expr == self.expr:
            if self.canonical == old.canonical:
                return new

            # collections.Counter doesn't have __le__
            def _subset(a, b):
                return all((a[i] <= b[i]) == True for i in a)

            old_vars = Counter(dict(reversed(old.variable_count)))
            self_vars = Counter(dict(reversed(self.variable_count)))
            if _subset(old_vars, self_vars):
                return _derivative_dispatch(new, *(self_vars - old_vars).items()).canonical

        args = list(self.args)
        newargs = [x._subs(old, new) for x in args]
        if args[0] == old:
            # complete replacement of self.expr
            # we already checked that the new is valid so we know
            # it won't be a problem should it appear in variables
            return _derivative_dispatch(*newargs)

        if newargs[0] != args[0]:
            # case (1) can't change expr by introducing something that is in
            # the _wrt_variables if it was already in the expr
            # e.g.
            # for Derivative(f(x, g(y)), y), x cannot be replaced with
            # anything that has y in it; for f(g(x), g(y)).diff(g(y))
            # g(x) cannot be replaced with anything that has g(y)
            syms = {vi: Dummy() for vi in self._wrt_variables
                if not vi.is_Symbol}
            wrt = {syms.get(vi, vi) for vi in self._wrt_variables}
            forbidden = args[0].xreplace(syms).free_symbols & wrt
            nfree = new.xreplace(syms).free_symbols
            ofree = old.xreplace(syms).free_symbols
            if (nfree - ofree) & forbidden:
                return Subs(self, old, new)

        viter = ((i, j) for ((i, _), (j, _)) in zip(newargs[1:], args[1:]))
        if any(i != j for i, j in viter):  # a wrt-variable change
            # case (2) can't change vars by introducing a variable
            # that is contained in expr, e.g.
            # for Derivative(f(z, g(h(x), y)), y), y cannot be changed to
            # x, h(x), or g(h(x), y)
            for a in _atomic(self.expr, recursive=True):
                for i in range(1, len(newargs)):
                    vi, _ = newargs[i]
                    if a == vi and vi != args[i][0]:
                        return Subs(self, old, new)
            # more arg-wise checks
            vc = newargs[1:]
            oldv = self._wrt_variables
            newe = self.expr
            subs = []
            for i, (vi, ci) in enumerate(vc):
                if not vi._diff_wrt:
                    # case (3) invalid differentiation expression so
                    # create a replacement dummy
                    xi = Dummy('xi_%i' % i)
                    # replace the old valid variable with the dummy
                    # in the expression
                    newe = newe.xreplace({oldv[i]: xi})
                    # and replace the bad variable with the dummy
                    vc[i] = (xi, ci)
                    # and record the dummy with the new (invalid)
                    # differentiation expression
                    subs.append((xi, vi))

            if subs:
                # handle any residual substitution in the expression
                newe = newe._subs(old, new)
                # return the Subs-wrapped derivative
                return Subs(Derivative(newe, *vc), *zip(*subs))

        # everything was ok
        return _derivative_dispatch(*newargs)

    def _eval_lseries(self, x, logx, cdir=0):
        dx = self.variables
        for term in self.expr.lseries(x, logx=logx, cdir=cdir):
            yield self.func(term, *dx)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.expr.nseries(x, n=n, logx=logx)
        o = arg.getO()
        dx = self.variables
        rv = [self.func(a, *dx) for a in Add.make_args(arg.removeO())]
        if o:
            rv.append(o/x)
        return Add(*rv)

    def _eval_as_leading_term(self, x, logx, cdir):
        series_gen = self.expr.lseries(x)
        d = S.Zero
        for leading_term in series_gen:
            d = diff(leading_term, *self.variables)
            if d != 0:
                break
        return d

    def as_finite_difference(self, points=1, x0=None, wrt=None):
        """ Expresses a Derivative instance as a finite difference.

        Parameters
        ==========

        points : sequence or coefficient, optional
            If sequence: discrete values (length >= order+1) of the
            independent variable used for generating the finite
            difference weights.
            If it is a coefficient, it will be used as the step-size
            for generating an equidistant sequence of length order+1
            centered around ``x0``. Default: 1 (step-size 1)

        x0 : number or Symbol, optional
            the value of the independent variable (``wrt``) at which the
            derivative is to be approximated. Default: same as ``wrt``.

        wrt : Symbol, optional
            "with respect to" the variable for which the (partial)
            derivative is to be approximated for. If not provided it
            is required that the derivative is ordinary. Default: ``None``.


        Examples
        ========

        >>> from sympy import symbols, Function, exp, sqrt, Symbol
        >>> x, h = symbols('x h')
        >>> f = Function('f')
        >>> f(x).diff(x).as_finite_difference()
        -f(x - 1/2) + f(x + 1/2)

        The default step size and number of points are 1 and
        ``order + 1`` respectively. We can change the step size by
        passing a symbol as a parameter:

        >>> f(x).diff(x).as_finite_difference(h)
        -f(-h/2 + x)/h + f(h/2 + x)/h

        We can also specify the discretized values to be used in a
        sequence:

        >>> f(x).diff(x).as_finite_difference([x, x+h, x+2*h])
        -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)

        The algorithm is not restricted to use equidistant spacing, nor
        do we need to make the approximation around ``x0``, but we can get
        an expression estimating the derivative at an offset:

        >>> e, sq2 = exp(1), sqrt(2)
        >>> xl = [x-h, x+h, x+e*h]
        >>> f(x).diff(x, 1).as_finite_difference(xl, x+h*sq2)  # doctest: +ELLIPSIS
        2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/...

        To approximate ``Derivative`` around ``x0`` using a non-equidistant
        spacing step, the algorithm supports assignment of undefined
        functions to ``points``:

        >>> dx = Function('dx')
        >>> f(x).diff(x).as_finite_difference(points=dx(x), x0=x-h)
        -f(-h + x - dx(-h + x)/2)/dx(-h + x) + f(-h + x + dx(-h + x)/2)/dx(-h + x)

        Partial derivatives are also supported:

        >>> y = Symbol('y')
        >>> d2fdxdy=f(x,y).diff(x,y)
        >>> d2fdxdy.as_finite_difference(wrt=x)
        -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)

        We can apply ``as_finite_difference`` to ``Derivative`` instances in
        compound expressions using ``replace``:

        >>> (1 + 42**f(x).diff(x)).replace(lambda arg: arg.is_Derivative,
        ...     lambda arg: arg.as_finite_difference())
        42**(-f(x - 1/2) + f(x + 1/2)) + 1


        See also
        ========

        sympy.calculus.finite_diff.apply_finite_diff
        sympy.calculus.finite_diff.differentiate_finite
        sympy.calculus.finite_diff.finite_diff_weights

        """
        from sympy.calculus.finite_diff import _as_finite_diff
        return _as_finite_diff(self, points, x0, wrt)

    @classmethod
    def _get_zero_with_shape_like(cls, expr):
        return S.Zero

    @classmethod
    def _dispatch_eval_derivative_n_times(cls, expr, v, count):
        # Evaluate the derivative `n` times.  If
        # `_eval_derivative_n_times` is not overridden by the current
        # object, the default in `Basic` will call a loop over
        # `_eval_derivative`:
        return expr._eval_derivative_n_times(v, count)


def _derivative_dispatch(expr, *variables, **kwargs):
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.matrices.expressions.matexpr import MatrixExpr
    from sympy.tensor.array import NDimArray
    array_types = (MatrixBase, MatrixExpr, NDimArray, list, tuple, Tuple)
    if isinstance(expr, array_types) or any(isinstance(i[0], array_types) if isinstance(i, (tuple, list, Tuple)) else isinstance(i, array_types) for i in variables):
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        return ArrayDerivative(expr, *variables, **kwargs)
    return Derivative(expr, *variables, **kwargs)


class Lambda(Expr):
    """
    Lambda(x, expr) represents a lambda function similar to Python's
    'lambda x: expr'. A function of several variables is written as
    Lambda((x, y, ...), expr).

    Examples
    ========

    A simple example:

    >>> from sympy import Lambda
    >>> from sympy.abc import x
    >>> f = Lambda(x, x**2)
    >>> f(4)
    16

    For multivariate functions, use:

    >>> from sympy.abc import y, z, t
    >>> f2 = Lambda((x, y, z, t), x + y**z + t**z)
    >>> f2(1, 2, 3, 4)
    73

    It is also possible to unpack tuple arguments:

    >>> f = Lambda(((x, y), z), x + y + z)
    >>> f((1, 2), 3)
    6

    A handy shortcut for lots of arguments:

    >>> p = x, y, z
    >>> f = Lambda(p, x + y*z)
    >>> f(*p)
    x + y*z

    """
    is_Function = True

    def __new__(cls, signature, expr) -> Lambda:
        if iterable(signature) and not isinstance(signature, (tuple, Tuple)):
            sympy_deprecation_warning(
                """
                Using a non-tuple iterable as the first argument to Lambda
                is deprecated. Use Lambda(tuple(args), expr) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-non-tuple-lambda",
            )
            signature = tuple(signature)
        _sig = signature if iterable(signature) else (signature,)
        sig: Tuple = sympify(_sig) # type: ignore
        cls._check_signature(sig)

        if len(sig) == 1 and sig[0] == expr:
            return S.IdentityFunction

        return Expr.__new__(cls, sig, sympify(expr))

    @classmethod
    def _check_signature(cls, sig):
        syms = set()

        def rcheck(args):
            for a in args:
                if a.is_symbol:
                    if a in syms:
                        raise BadSignatureError("Duplicate symbol %s" % a)
                    syms.add(a)
                elif isinstance(a, Tuple):
                    rcheck(a)
                else:
                    raise BadSignatureError("Lambda signature should be only tuples"
                        " and symbols, not %s" % a)

        if not isinstance(sig, Tuple):
            raise BadSignatureError("Lambda signature should be a tuple not %s" % sig)
        # Recurse through the signature:
        rcheck(sig)

    @property
    def signature(self):
        """The expected form of the arguments to be unpacked into variables"""
        return self._args[0]

    @property
    def expr(self):
        """The return value of the function"""
        return self._args[1]

    @property
    def variables(self):
        """The variables used in the internal representation of the function"""
        def _variables(args):
            if isinstance(args, Tuple):
                for arg in args:
                    yield from _variables(arg)
            else:
                yield args
        return tuple(_variables(self.signature))

    @property
    def nargs(self):
        from sympy.sets.sets import FiniteSet
        return FiniteSet(len(self.signature))

    bound_symbols = variables

    @property
    def free_symbols(self):
        return self.expr.free_symbols - set(self.variables)

    def __call__(self, *args):
        n = len(args)
        if n not in self.nargs:  # Lambda only ever has 1 value in nargs
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            ## XXX does this apply to Lambda? If not, remove this comment.
            temp = ('%(name)s takes exactly %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise BadArgumentsError(temp % {
                'name': self,
                'args': list(self.nargs)[0],
                'plural': 's'*(list(self.nargs)[0] != 1),
                'given': n})

        d = self._match_signature(self.signature, args)

        return self.expr.xreplace(d)

    def _match_signature(self, sig, args):

        symargmap = {}

        def rmatch(pars, args):
            for par, arg in zip(pars, args):
                if par.is_symbol:
                    symargmap[par] = arg
                elif isinstance(par, Tuple):
                    if not isinstance(arg, (tuple, Tuple)) or len(args) != len(pars):
                        raise BadArgumentsError("Can't match %s and %s" % (args, pars))
                    rmatch(par, arg)

        rmatch(sig, args)

        return symargmap

    @property
    def is_identity(self):
        """Return ``True`` if this ``Lambda`` is an identity function. """
        return self.signature == self.expr

    def _eval_evalf(self, prec):
        return self.func(self.args[0], self.args[1].evalf(n=prec_to_dps(prec)))


class Subs(Expr):
    """
    Represents unevaluated substitutions of an expression.

    ``Subs(expr, x, x0)`` represents the expression resulting
    from substituting x with x0 in expr.

    Parameters
    ==========

    expr : Expr
        An expression.

    x : tuple, variable
        A variable or list of distinct variables.

    x0 : tuple or list of tuples
        A point or list of evaluation points
        corresponding to those variables.

    Examples
    ========

    >>> from sympy import Subs, Function, sin, cos
    >>> from sympy.abc import x, y, z
    >>> f = Function('f')

    Subs are created when a particular substitution cannot be made. The
    x in the derivative cannot be replaced with 0 because 0 is not a
    valid variables of differentiation:

    >>> f(x).diff(x).subs(x, 0)
    Subs(Derivative(f(x), x), x, 0)

    Once f is known, the derivative and evaluation at 0 can be done:

    >>> _.subs(f, sin).doit() == sin(x).diff(x).subs(x, 0) == cos(0)
    True

    Subs can also be created directly with one or more variables:

    >>> Subs(f(x)*sin(y) + z, (x, y), (0, 1))
    Subs(z + f(x)*sin(y), (x, y), (0, 1))
    >>> _.doit()
    z + f(0)*sin(1)

    Notes
    =====

    ``Subs`` objects are generally useful to represent unevaluated derivatives
    calculated at a point.

    The variables may be expressions, but they are subjected to the limitations
    of subs(), so it is usually a good practice to use only symbols for
    variables, since in that case there can be no ambiguity.

    There's no automatic expansion - use the method .doit() to effect all
    possible substitutions of the object and also of objects inside the
    expression.

    When evaluating derivatives at a point that is not a symbol, a Subs object
    is returned. One is also able to calculate derivatives of Subs objects - in
    this case the expression is always expanded (for the unevaluated form, use
    Derivative()).

    In order to allow expressions to combine before doit is done, a
    representation of the Subs expression is used internally to make
    expressions that are superficially different compare the same:

    >>> a, b = Subs(x, x, 0), Subs(y, y, 0)
    >>> a + b
    2*Subs(x, x, 0)

    This can lead to unexpected consequences when using methods
    like `has` that are cached:

    >>> s = Subs(x, x, 0)
    >>> s.has(x), s.has(y)
    (True, False)
    >>> ss = s.subs(x, y)
    >>> ss.has(x), ss.has(y)
    (True, False)
    >>> s, ss
    (Subs(x, x, 0), Subs(y, y, 0))
    """
    def __new__(cls, expr, variables, point, **assumptions):
        if not is_sequence(variables, Tuple):
            variables = [variables]
        variables = Tuple(*variables)

        if has_dups(variables):
            repeated = [str(v) for v, i in Counter(variables).items() if i > 1]
            __ = ', '.join(repeated)
            raise ValueError(filldedent('''
                The following expressions appear more than once: %s
                ''' % __))

        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))

        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as '
                             'the number of variables.')

        if not point:
            return sympify(expr)

        # denest
        if isinstance(expr, Subs):
            variables = expr.variables + variables
            point = expr.point + point
            expr = expr.expr
        else:
            expr = sympify(expr)

        # use symbols with names equal to the point value (with prepended _)
        # to give a variable-independent expression
        pre = "_"
        pts = sorted(set(point), key=default_sort_key)
        from sympy.printing.str import StrPrinter
        class CustomStrPrinter(StrPrinter):
            def _print_Dummy(self, expr):
                return str(expr) + str(expr.dummy_index)
        def mystr(expr, **settings):
            p = CustomStrPrinter(settings)
            return p.doprint(expr)
        while 1:
            s_pts = {p: Symbol(pre + mystr(p)) for p in pts}
            reps = [(v, s_pts[p])
                for v, p in zip(variables, point)]
            # if any underscore-prepended symbol is already a free symbol
            # and is a variable with a different point value, then there
            # is a clash, e.g. _0 clashes in Subs(_0 + _1, (_0, _1), (1, 0))
            # because the new symbol that would be created is _1 but _1
            # is already mapped to 0 so __0 and __1 are used for the new
            # symbols
            if any(r in expr.free_symbols and
                   r in variables and
                   Symbol(pre + mystr(point[variables.index(r)])) != r
                   for _, r in reps):
                pre += "_"
                continue
            break

        obj = Expr.__new__(cls, expr, Tuple(*variables), point)
        obj._expr = expr.xreplace(dict(reps))
        return obj

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def doit(self, **hints):
        e, v, p = self.args

        # remove self mappings
        for i, (vi, pi) in enumerate(zip(v, p)):
            if vi == pi:
                v = v[:i] + v[i + 1:]
                p = p[:i] + p[i + 1:]
        if not v:
            return self.expr

        if isinstance(e, Derivative):
            # apply functions first, e.g. f -> cos
            undone = []
            for i, vi in enumerate(v):
                if isinstance(vi, FunctionClass):
                    e = e.subs(vi, p[i])
                else:
                    undone.append((vi, p[i]))
            if not isinstance(e, Derivative):
                e = e.doit()
            if isinstance(e, Derivative):
                # do Subs that aren't related to differentiation
                undone2 = []
                D = Dummy()
                arg = e.args[0]
                for vi, pi in undone:
                    if D not in e.xreplace({vi: D}).free_symbols:
                        if arg.has(vi):
                            e = e.subs(vi, pi)
                    else:
                        undone2.append((vi, pi))
                undone = undone2
                # differentiate wrt variables that are present
                wrt = []
                D = Dummy()
                expr = e.expr
                free = expr.free_symbols
                for vi, ci in e.variable_count:
                    if isinstance(vi, Symbol) and vi in free:
                        expr = expr.diff((vi, ci))
                    elif D in expr.subs(vi, D).free_symbols:
                        expr = expr.diff((vi, ci))
                    else:
                        wrt.append((vi, ci))
                # inject remaining subs
                rv = expr.subs(undone)
                # do remaining differentiation *in order given*
                for vc in wrt:
                    rv = rv.diff(vc)
            else:
                # inject remaining subs
                rv = e.subs(undone)
        else:
            rv = e.doit(**hints).subs(list(zip(v, p)))

        if hints.get('deep', True) and rv != self:
            rv = rv.doit(**hints)
        return rv

    def evalf(self, prec=None, **options):
        return self.doit().evalf(prec, **options)

    n = evalf  # type:ignore

    @property
    def variables(self):
        """The variables to be evaluated"""
        return self._args[1]

    bound_symbols = variables

    @property
    def expr(self):
        """The expression on which the substitution operates"""
        return self._args[0]

    @property
    def point(self):
        """The values for which the variables are to be substituted"""
        return self._args[2]

    @property
    def free_symbols(self):
        return (self.expr.free_symbols - set(self.variables) |
            set(self.point.free_symbols))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        # Don't show the warning twice from the recursive call
        with ignore_warnings(SymPyDeprecationWarning):
            return (self.expr.expr_free_symbols - set(self.variables) |
                    set(self.point.expr_free_symbols))

    def __eq__(self, other):
        if not isinstance(other, Subs):
            return False
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other):
        return not(self == other)

    def __hash__(self):
        return super().__hash__()

    def _hashable_content(self):
        return (self._expr.xreplace(self.canonical_variables),
            ) + tuple(ordered([(v, p) for v, p in
            zip(self.variables, self.point) if not self.expr.has(v)]))

    def _eval_subs(self, old, new):
        # Subs doit will do the variables in order; the semantics
        # of subs for Subs is have the following invariant for
        # Subs object foo:
        #    foo.doit().subs(reps) == foo.subs(reps).doit()
        pt = list(self.point)
        if old in self.variables:
            if _atomic(new) == {new} and not any(
                    i.has(new) for i in self.args):
                # the substitution is neutral
                return self.xreplace({old: new})
            # any occurrence of old before this point will get
            # handled by replacements from here on
            i = self.variables.index(old)
            for j in range(i, len(self.variables)):
                pt[j] = pt[j]._subs(old, new)
            return self.func(self.expr, self.variables, pt)
        v = [i._subs(old, new) for i in self.variables]
        if v != list(self.variables):
            return self.func(self.expr, self.variables + (old,), pt + [new])
        expr = self.expr._subs(old, new)
        pt = [i._subs(old, new) for i in self.point]
        return self.func(expr, v, pt)

    def _eval_derivative(self, s):
        # Apply the chain rule of the derivative on the substitution variables:
        f = self.expr
        vp = V, P = self.variables, self.point
        val = Add.fromiter(p.diff(s)*Subs(f.diff(v), *vp).doit()
            for v, p in zip(V, P))

        # these are all the free symbols in the expr
        efree = f.free_symbols
        # some symbols like IndexedBase include themselves and args
        # as free symbols
        compound = {i for i in efree if len(i.free_symbols) > 1}
        # hide them and see what independent free symbols remain
        dums = {Dummy() for i in compound}
        masked = f.xreplace(dict(zip(compound, dums)))
        ifree = masked.free_symbols - dums
        # include the compound symbols
        free = ifree | compound
        # remove the variables already handled
        free -= set(V)
        # add back any free symbols of remaining compound symbols
        free |= {i for j in free & compound for i in j.free_symbols}
        # if symbols of s are in free then there is more to do
        if free & s.free_symbols:
            val += Subs(f.diff(s), self.variables, self.point).doit()
        return val

    def _eval_nseries(self, x, n, logx, cdir=0):
        if x in self.point:
            # x is the variable being substituted into
            apos = self.point.index(x)
            other = self.variables[apos]
        else:
            other = x
        arg = self.expr.nseries(other, n=n, logx=logx)
        o = arg.getO()
        terms = Add.make_args(arg.removeO())
        rv = Add(*[self.func(a, *self.args[1:]) for a in terms])
        if o:
            rv += o.subs(other, x)
        return rv

    def _eval_as_leading_term(self, x, logx, cdir):
        if x in self.point:
            ipos = self.point.index(x)
            xvar = self.variables[ipos]
            return self.expr.as_leading_term(xvar)
        if x in self.variables:
            # if `x` is a dummy variable, it means it won't exist after the
            # substitution has been performed:
            return self
        # The variable is independent of the substitution:
        return self.expr.as_leading_term(x)


def diff(f, *symbols, **kwargs):
    """
    Differentiate f with respect to symbols.

    Explanation
    ===========

    This is just a wrapper to unify .diff() and the Derivative class; its
    interface is similar to that of integrate().  You can use the same
    shortcuts for multiple variables as with Derivative.  For example,
    diff(f(x), x, x, x) and diff(f(x), x, 3) both return the third derivative
    of f(x).

    You can pass evaluate=False to get an unevaluated Derivative class.  Note
    that if there are 0 symbols (such as diff(f(x), x, 0), then the result will
    be the function (the zeroth derivative), even if evaluate=False.

    Examples
    ========

    >>> from sympy import sin, cos, Function, diff
    >>> from sympy.abc import x, y
    >>> f = Function('f')

    >>> diff(sin(x), x)
    cos(x)
    >>> diff(f(x), x, x, x)
    Derivative(f(x), (x, 3))
    >>> diff(f(x), x, 3)
    Derivative(f(x), (x, 3))
    >>> diff(sin(x)*cos(y), x, 2, y, 2)
    sin(x)*cos(y)

    >>> type(diff(sin(x), x))
    cos
    >>> type(diff(sin(x), x, evaluate=False))
    <class 'sympy.core.function.Derivative'>
    >>> type(diff(sin(x), x, 0))
    sin
    >>> type(diff(sin(x), x, 0, evaluate=False))
    sin

    >>> diff(sin(x))
    cos(x)
    >>> diff(sin(x*y))
    Traceback (most recent call last):
    ...
    ValueError: specify differentiation variables to differentiate sin(x*y)

    Note that ``diff(sin(x))`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

    References
    ==========

    .. [1] https://reference.wolfram.com/legacy/v5_2/Built-inFunctions/AlgebraicComputation/Calculus/D.html

    See Also
    ========

    Derivative
    idiff: computes the derivative implicitly

    """
    if hasattr(f, 'diff'):
        return f.diff(*symbols, **kwargs)
    kwargs.setdefault('evaluate', True)
    return _derivative_dispatch(f, *symbols, **kwargs)


def expand(e, deep=True, modulus=None, power_base=True, power_exp=True,
        mul=True, log=True, multinomial=True, basic=True, **hints):
    r"""
    Expand an expression using methods given as hints.

    Explanation
    ===========

    Hints evaluated unless explicitly set to False are:  ``basic``, ``log``,
    ``multinomial``, ``mul``, ``power_base``, and ``power_exp`` The following
    hints are supported but not applied unless set to True:  ``complex``,
    ``func``, and ``trig``.  In addition, the following meta-hints are
    supported by some or all of the other hints:  ``frac``, ``numer``,
    ``denom``, ``modulus``, and ``force``.  ``deep`` is supported by all
    hints.  Additionally, subclasses of Expr may define their own hints or
    meta-hints.

    The ``basic`` hint is used for any special rewriting of an object that
    should be done automatically (along with the other hints like ``mul``)
    when expand is called. This is a catch-all hint to handle any sort of
    expansion that may not be described by the existing hint names. To use
    this hint an object should override the ``_eval_expand_basic`` method.
    Objects may also define their own expand methods, which are not run by
    default.  See the API section below.

    If ``deep`` is set to ``True`` (the default), things like arguments of
    functions are recursively expanded.  Use ``deep=False`` to only expand on
    the top level.

    If the ``force`` hint is used, assumptions about variables will be ignored
    in making the expansion.

    Hints
    =====

    These hints are run by default

    mul
    ---

    Distributes multiplication over addition:

    >>> from sympy import cos, exp, sin
    >>> from sympy.abc import x, y, z
    >>> (y*(x + z)).expand(mul=True)
    x*y + y*z

    multinomial
    -----------

    Expand (x + y + ...)**n where n is a positive integer.

    >>> ((x + y + z)**2).expand(multinomial=True)
    x**2 + 2*x*y + 2*x*z + y**2 + 2*y*z + z**2

    power_exp
    ---------

    Expand addition in exponents into multiplied bases.

    >>> exp(x + y).expand(power_exp=True)
    exp(x)*exp(y)
    >>> (2**(x + y)).expand(power_exp=True)
    2**x*2**y

    power_base
    ----------

    Split powers of multiplied bases.

    This only happens by default if assumptions allow, or if the
    ``force`` meta-hint is used:

    >>> ((x*y)**z).expand(power_base=True)
    (x*y)**z
    >>> ((x*y)**z).expand(power_base=True, force=True)
    x**z*y**z
    >>> ((2*y)**z).expand(power_base=True)
    2**z*y**z

    Note that in some cases where this expansion always holds, SymPy performs
    it automatically:

    >>> (x*y)**2
    x**2*y**2

    log
    ---

    Pull out power of an argument as a coefficient and split logs products
    into sums of logs.

    Note that these only work if the arguments of the log function have the
    proper assumptions--the arguments must be positive and the exponents must
    be real--or else the ``force`` hint must be True:

    >>> from sympy import log, symbols
    >>> log(x**2*y).expand(log=True)
    log(x**2*y)
    >>> log(x**2*y).expand(log=True, force=True)
    2*log(x) + log(y)
    >>> x, y = symbols('x,y', positive=True)
    >>> log(x**2*y).expand(log=True)
    2*log(x) + log(y)

    basic
    -----

    This hint is intended primarily as a way for custom subclasses to enable
    expansion by default.

    These hints are not run by default:

    complex
    -------

    Split an expression into real and imaginary parts.

    >>> x, y = symbols('x,y')
    >>> (x + y).expand(complex=True)
    re(x) + re(y) + I*im(x) + I*im(y)
    >>> cos(x).expand(complex=True)
    -I*sin(re(x))*sinh(im(x)) + cos(re(x))*cosh(im(x))

    Note that this is just a wrapper around ``as_real_imag()``.  Most objects
    that wish to redefine ``_eval_expand_complex()`` should consider
    redefining ``as_real_imag()`` instead.

    func
    ----

    Expand other functions.

    >>> from sympy import gamma
    >>> gamma(x + 1).expand(func=True)
    x*gamma(x)

    trig
    ----

    Do trigonometric expansions.

    >>> cos(x + y).expand(trig=True)
    -sin(x)*sin(y) + cos(x)*cos(y)
    >>> sin(2*x).expand(trig=True)
    2*sin(x)*cos(x)

    Note that the forms of ``sin(n*x)`` and ``cos(n*x)`` in terms of ``sin(x)``
    and ``cos(x)`` are not unique, due to the identity `\sin^2(x) + \cos^2(x)
    = 1`.  The current implementation uses the form obtained from Chebyshev
    polynomials, but this may change.  See `this MathWorld article
    <https://mathworld.wolfram.com/Multiple-AngleFormulas.html>`_ for more
    information.

    Notes
    =====

    - You can shut off unwanted methods::

        >>> (exp(x + y)*(x + y)).expand()
        x*exp(x)*exp(y) + y*exp(x)*exp(y)
        >>> (exp(x + y)*(x + y)).expand(power_exp=False)
        x*exp(x + y) + y*exp(x + y)
        >>> (exp(x + y)*(x + y)).expand(mul=False)
        (x + y)*exp(x)*exp(y)

    - Use deep=False to only expand on the top level::

        >>> exp(x + exp(x + y)).expand()
        exp(x)*exp(exp(x)*exp(y))
        >>> exp(x + exp(x + y)).expand(deep=False)
        exp(x)*exp(exp(x + y))

    - Hints are applied in an arbitrary, but consistent order (in the current
      implementation, they are applied in alphabetical order, except
      multinomial comes before mul, but this may change).  Because of this,
      some hints may prevent expansion by other hints if they are applied
      first. For example, ``mul`` may distribute multiplications and prevent
      ``log`` and ``power_base`` from expanding them. Also, if ``mul`` is
      applied before ``multinomial`, the expression might not be fully
      distributed. The solution is to use the various ``expand_hint`` helper
      functions or to use ``hint=False`` to this function to finely control
      which hints are applied. Here are some examples::

        >>> from sympy import expand, expand_mul, expand_power_base
        >>> x, y, z = symbols('x,y,z', positive=True)

        >>> expand(log(x*(y + z)))
        log(x) + log(y + z)

      Here, we see that ``log`` was applied before ``mul``.  To get the mul
      expanded form, either of the following will work::

        >>> expand_mul(log(x*(y + z)))
        log(x*y + x*z)
        >>> expand(log(x*(y + z)), log=False)
        log(x*y + x*z)

      A similar thing can happen with the ``power_base`` hint::

        >>> expand((x*(y + z))**x)
        (x*y + x*z)**x

      To get the ``power_base`` expanded form, either of the following will
      work::

        >>> expand((x*(y + z))**x, mul=False)
        x**x*(y + z)**x
        >>> expand_power_base((x*(y + z))**x)
        x**x*(y + z)**x

        >>> expand((x + y)*y/x)
        y + y**2/x

      The parts of a rational expression can be targeted::

        >>> expand((x + y)*y/x/(x + 1), frac=True)
        (x*y + y**2)/(x**2 + x)
        >>> expand((x + y)*y/x/(x + 1), numer=True)
        (x*y + y**2)/(x*(x + 1))
        >>> expand((x + y)*y/x/(x + 1), denom=True)
        y*(x + y)/(x**2 + x)

    - The ``modulus`` meta-hint can be used to reduce the coefficients of an
      expression post-expansion::

        >>> expand((3*x + 1)**2)
        9*x**2 + 6*x + 1
        >>> expand((3*x + 1)**2, modulus=5)
        4*x**2 + x + 1

    - Either ``expand()`` the function or ``.expand()`` the method can be
      used.  Both are equivalent::

        >>> expand((x + 1)**2)
        x**2 + 2*x + 1
        >>> ((x + 1)**2).expand()
        x**2 + 2*x + 1

    API
    ===

    Objects can define their own expand hints by defining
    ``_eval_expand_hint()``.  The function should take the form::

        def _eval_expand_hint(self, **hints):
            # Only apply the method to the top-level expression
            ...

    See also the example below.  Objects should define ``_eval_expand_hint()``
    methods only if ``hint`` applies to that specific object.  The generic
    ``_eval_expand_hint()`` method defined in Expr will handle the no-op case.

    Each hint should be responsible for expanding that hint only.
    Furthermore, the expansion should be applied to the top-level expression
    only.  ``expand()`` takes care of the recursion that happens when
    ``deep=True``.

    You should only call ``_eval_expand_hint()`` methods directly if you are
    100% sure that the object has the method, as otherwise you are liable to
    get unexpected ``AttributeError``s.  Note, again, that you do not need to
    recursively apply the hint to args of your object: this is handled
    automatically by ``expand()``.  ``_eval_expand_hint()`` should
    generally not be used at all outside of an ``_eval_expand_hint()`` method.
    If you want to apply a specific expansion from within another method, use
    the public ``expand()`` function, method, or ``expand_hint()`` functions.

    In order for expand to work, objects must be rebuildable by their args,
    i.e., ``obj.func(*obj.args) == obj`` must hold.

    Expand methods are passed ``**hints`` so that expand hints may use
    'metahints'--hints that control how different expand methods are applied.
    For example, the ``force=True`` hint described above that causes
    ``expand(log=True)`` to ignore assumptions is such a metahint.  The
    ``deep`` meta-hint is handled exclusively by ``expand()`` and is not
    passed to ``_eval_expand_hint()`` methods.

    Note that expansion hints should generally be methods that perform some
    kind of 'expansion'.  For hints that simply rewrite an expression, use the
    .rewrite() API.

    Examples
    ========

    >>> from sympy import Expr, sympify
    >>> class MyClass(Expr):
    ...     def __new__(cls, *args):
    ...         args = sympify(args)
    ...         return Expr.__new__(cls, *args)
    ...
    ...     def _eval_expand_double(self, *, force=False, **hints):
    ...         '''
    ...         Doubles the args of MyClass.
    ...
    ...         If there more than four args, doubling is not performed,
    ...         unless force=True is also used (False by default).
    ...         '''
    ...         if not force and len(self.args) > 4:
    ...             return self
    ...         return self.func(*(self.args + self.args))
    ...
    >>> a = MyClass(1, 2, MyClass(3, 4))
    >>> a
    MyClass(1, 2, MyClass(3, 4))
    >>> a.expand(double=True)
    MyClass(1, 2, MyClass(3, 4, 3, 4), 1, 2, MyClass(3, 4, 3, 4))
    >>> a.expand(double=True, deep=False)
    MyClass(1, 2, MyClass(3, 4), 1, 2, MyClass(3, 4))

    >>> b = MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True)
    MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True, force=True)
    MyClass(1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

    See Also
    ========

    expand_log, expand_mul, expand_multinomial, expand_complex, expand_trig,
    expand_power_base, expand_power_exp, expand_func, sympy.simplify.hyperexpand.hyperexpand

    """
    # don't modify this; modify the Expr.expand method
    hints['power_base'] = power_base
    hints['power_exp'] = power_exp
    hints['mul'] = mul
    hints['log'] = log
    hints['multinomial'] = multinomial
    hints['basic'] = basic
    return sympify(e).expand(deep=deep, modulus=modulus, **hints)

# This is a special application of two hints

def _mexpand(expr, recursive=False):
    # expand multinomials and then expand products; this may not always
    # be sufficient to give a fully expanded expression (see
    # test_issue_8247_8354 in test_arit)
    if expr is None:
        return
    was = None
    while was != expr:
        was, expr = expr, expand_mul(expand_multinomial(expr))
        if not recursive:
            break
    return expr


# These are simple wrappers around single hints.


def expand_mul(expr, deep=True):
    """
    Wrapper around expand that only uses the mul hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_mul, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_mul(exp(x+y)*(x+y)*log(x*y**2))
    x*exp(x + y)*log(x*y**2) + y*exp(x + y)*log(x*y**2)

    """
    return sympify(expr).expand(deep=deep, mul=True, power_exp=False,
    power_base=False, basic=False, multinomial=False, log=False)


def expand_multinomial(expr, deep=True):
    """
    Wrapper around expand that only uses the multinomial hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_multinomial, exp
    >>> x, y = symbols('x y', positive=True)
    >>> expand_multinomial((x + exp(x + 1))**2)
    x**2 + 2*x*exp(x + 1) + exp(2*x + 2)

    """
    return sympify(expr).expand(deep=deep, mul=False, power_exp=False,
    power_base=False, basic=False, multinomial=True, log=False)


def expand_log(expr, deep=True, force=False, factor=False):
    """
    Wrapper around expand that only uses the log hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_log, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_log(exp(x+y)*(x+y)*log(x*y**2))
    (x + y)*(log(x) + 2*log(y))*exp(x + y)

    """
    from sympy.functions.elementary.exponential import log
    from sympy.simplify.radsimp import fraction
    if factor is False:
        def _handleMul(x):
            # look for the simple case of expanded log(b**a)/log(b) -> a in args
            n, d = fraction(x)
            n = [i for i in n.atoms(log) if i.args[0].is_Integer]
            d = [i for i in d.atoms(log) if i.args[0].is_Integer]
            if len(n) == 1 and len(d) == 1:
                n = n[0]
                d = d[0]
                from sympy import multiplicity
                m = multiplicity(d.args[0], n.args[0])
                if m:
                    r = m + log(n.args[0]//d.args[0]**m)/d
                    x = x.subs(n, d*r)
            x1 = expand_mul(expand_log(x, deep=deep, force=force, factor=True))
            if x1.count(log) <= x.count(log):
                return x1
            return x

        expr = expr.replace(
        lambda x: x.is_Mul and all(any(isinstance(i, log) and i.args[0].is_Rational
        for i in Mul.make_args(j)) for j in x.as_numer_denom()),
        _handleMul)

    return sympify(expr).expand(deep=deep, log=True, mul=False,
        power_exp=False, power_base=False, multinomial=False,
        basic=False, force=force, factor=factor)


def expand_func(expr, deep=True):
    """
    Wrapper around expand that only uses the func hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_func, gamma
    >>> from sympy.abc import x
    >>> expand_func(gamma(x + 2))
    x*(x + 1)*gamma(x)

    """
    return sympify(expr).expand(deep=deep, func=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_trig(expr, deep=True):
    """
    Wrapper around expand that only uses the trig hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_trig, sin
    >>> from sympy.abc import x, y
    >>> expand_trig(sin(x+y)*(x+y))
    (x + y)*(sin(x)*cos(y) + sin(y)*cos(x))

    """
    return sympify(expr).expand(deep=deep, trig=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_complex(expr, deep=True):
    """
    Wrapper around expand that only uses the complex hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_complex, exp, sqrt, I
    >>> from sympy.abc import z
    >>> expand_complex(exp(z))
    I*exp(re(z))*sin(im(z)) + exp(re(z))*cos(im(z))
    >>> expand_complex(sqrt(I))
    sqrt(2)/2 + sqrt(2)*I/2

    See Also
    ========

    sympy.core.expr.Expr.as_real_imag
    """
    return sympify(expr).expand(deep=deep, complex=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_power_base(expr, deep=True, force=False):
    """
    Wrapper around expand that only uses the power_base hint.

    A wrapper to expand(power_base=True) which separates a power with a base
    that is a Mul into a product of powers, without performing any other
    expansions, provided that assumptions about the power's base and exponent
    allow.

    deep=False (default is True) will only apply to the top-level expression.

    force=True (default is False) will cause the expansion to ignore
    assumptions about the base and exponent. When False, the expansion will
    only happen if the base is non-negative or the exponent is an integer.

    >>> from sympy.abc import x, y, z
    >>> from sympy import expand_power_base, sin, cos, exp, Symbol

    >>> (x*y)**2
    x**2*y**2

    >>> (2*x)**y
    (2*x)**y
    >>> expand_power_base(_)
    2**y*x**y

    >>> expand_power_base((x*y)**z)
    (x*y)**z
    >>> expand_power_base((x*y)**z, force=True)
    x**z*y**z
    >>> expand_power_base(sin((x*y)**z), deep=False)
    sin((x*y)**z)
    >>> expand_power_base(sin((x*y)**z), force=True)
    sin(x**z*y**z)

    >>> expand_power_base((2*sin(x))**y + (2*cos(x))**y)
    2**y*sin(x)**y + 2**y*cos(x)**y

    >>> expand_power_base((2*exp(y))**x)
    2**x*exp(y)**x

    >>> expand_power_base((2*cos(x))**y)
    2**y*cos(x)**y

    Notice that sums are left untouched. If this is not the desired behavior,
    apply full ``expand()`` to the expression:

    >>> expand_power_base(((x+y)*z)**2)
    z**2*(x + y)**2
    >>> (((x+y)*z)**2).expand()
    x**2*z**2 + 2*x*y*z**2 + y**2*z**2

    >>> expand_power_base((2*y)**(1+z))
    2**(z + 1)*y**(z + 1)
    >>> ((2*y)**(1+z)).expand()
    2*2**z*y**(z + 1)

    The power that is unexpanded can be expanded safely when
    ``y != 0``, otherwise different values might be obtained for the expression:

    >>> prev = _

    If we indicate that ``y`` is positive but then replace it with
    a value of 0 after expansion, the expression becomes 0:

    >>> p = Symbol('p', positive=True)
    >>> prev.subs(y, p).expand().subs(p, 0)
    0

    But if ``z = -1`` the expression would not be zero:

    >>> prev.subs(y, 0).subs(z, -1)
    1

    See Also
    ========

    expand

    """
    return sympify(expr).expand(deep=deep, log=False, mul=False,
        power_exp=False, power_base=True, multinomial=False,
        basic=False, force=force)


def expand_power_exp(expr, deep=True):
    """
    Wrapper around expand that only uses the power_exp hint.

    See the expand docstring for more information.

    Examples
    ========

    >>> from sympy import expand_power_exp, Symbol
    >>> from sympy.abc import x, y
    >>> expand_power_exp(3**(y + 2))
    9*3**y
    >>> expand_power_exp(x**(y + 2))
    x**(y + 2)

    If ``x = 0`` the value of the expression depends on the
    value of ``y``; if the expression were expanded the result
    would be 0. So expansion is only done if ``x != 0``:

    >>> expand_power_exp(Symbol('x', zero=False)**(y + 2))
    x**2*x**y
    """
    return sympify(expr).expand(deep=deep, complex=False, basic=False,
    log=False, mul=False, power_exp=True, power_base=False, multinomial=False)


def count_ops(expr, visual=False):
    """
    Return a representation (integer or expression) of the operations in expr.

    Parameters
    ==========

    expr : Expr
        If expr is an iterable, the sum of the op counts of the
        items will be returned.

    visual : bool, optional
        If ``False`` (default) then the sum of the coefficients of the
        visual expression will be returned.
        If ``True`` then the number of each type of operation is shown
        with the core class types (or their virtual equivalent) multiplied by the
        number of times they occur.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y
    >>> from sympy import sin, count_ops

    Although there is not a SUB object, minus signs are interpreted as
    either negations or subtractions:

    >>> (x - y).count_ops(visual=True)
    SUB
    >>> (-x).count_ops(visual=True)
    NEG

    Here, there are two Adds and a Pow:

    >>> (1 + a + b**2).count_ops(visual=True)
    2*ADD + POW

    In the following, an Add, Mul, Pow and two functions:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=True)
    ADD + MUL + POW + 2*SIN

    for a total of 5:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=False)
    5

    Note that "what you type" is not always what you get. The expression
    1/x/y is translated by sympy into 1/(x*y) so it gives a DIV and MUL rather
    than two DIVs:

    >>> (1/x/y).count_ops(visual=True)
    DIV + MUL

    The visual option can be used to demonstrate the difference in
    operations for expressions in different forms. Here, the Horner
    representation is compared with the expanded form of a polynomial:

    >>> eq=x*(1 + x*(2 + x*(3 + x)))
    >>> count_ops(eq.expand(), visual=True) - count_ops(eq, visual=True)
    -MUL + 3*POW

    The count_ops function also handles iterables:

    >>> count_ops([x, sin(x), None, True, x + 2], visual=False)
    2
    >>> count_ops([x, sin(x), None, True, x + 2], visual=True)
    ADD + SIN
    >>> count_ops({x: sin(x), x + 2: y + 1}, visual=True)
    2*ADD + SIN

    """
    from .relational import Relational
    from sympy.concrete.summations import Sum
    from sympy.integrals.integrals import Integral
    from sympy.logic.boolalg import BooleanFunction
    from sympy.simplify.radsimp import fraction

    expr = sympify(expr)
    if isinstance(expr, Expr) and not expr.is_Relational:

        ops = []
        args = [expr]
        NEG = Symbol('NEG')
        DIV = Symbol('DIV')
        SUB = Symbol('SUB')
        ADD = Symbol('ADD')
        EXP = Symbol('EXP')
        while args:
            a = args.pop()

            # if the following fails because the object is
            # not Basic type, then the object should be fixed
            # since it is the intention that all args of Basic
            # should themselves be Basic
            if a.is_Rational:
                #-1/3 = NEG + DIV
                if a is not S.One:
                    if a.p < 0:
                        ops.append(NEG)
                    if a.q != 1:
                        ops.append(DIV)
                    continue
            elif a.is_Mul or a.is_MatMul:
                if _coeff_isneg(a):
                    ops.append(NEG)
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                n, d = fraction(a)
                if n.is_Integer:
                    ops.append(DIV)
                    if n < 0:
                        ops.append(NEG)
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    ops.append(DIV)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add or a.is_MatAdd:
                aargs = list(a.args)
                negs = 0
                for i, ai in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                        if i > 0:
                            ops.append(SUB)
                    else:
                        args.append(ai)
                        if i > 0:
                            ops.append(ADD)
                if negs == len(aargs):  # -x - y = NEG + SUB
                    ops.append(NEG)
                elif _coeff_isneg(aargs[0]):  # -x + y = SUB, but already recorded ADD
                    ops.append(SUB - ADD)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                ops.append(DIV)
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if a == S.Exp1:
                ops.append(EXP)
                continue
            if a.is_Pow and a.base == S.Exp1:
                ops.append(EXP)
                args.append(a.exp)
                continue
            if a.is_Mul or isinstance(a, LatticeOp):
                o = Symbol(a.func.__name__.upper())
                # count the args
                ops.append(o*(len(a.args) - 1))
            elif a.args and (
                    a.is_Pow or a.is_Function or isinstance(a, (Derivative, Integral, Sum))):
                # if it's not in the list above we don't
                # consider a.func something to count, e.g.
                # Tuple, MatrixSymbol, etc...
                if isinstance(a.func, UndefinedFunction):
                    o = Symbol("FUNC_" + a.func.__name__.upper())
                else:
                    o = Symbol(a.func.__name__.upper())
                ops.append(o)

            if not a.is_Symbol:
                args.extend(a.args)

    elif isinstance(expr, Dict):
        ops = [count_ops(k, visual=visual) +
               count_ops(v, visual=visual) for k, v in expr.items()]
    elif iterable(expr):
        ops = [count_ops(i, visual=visual) for i in expr]
    elif isinstance(expr, (Relational, BooleanFunction)):
        ops = []
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        o = Symbol(func_name(expr, short=True).upper())
        ops.append(o)
    elif not isinstance(expr, Basic):
        ops = []
    else:  # it's Basic not isinstance(expr, Expr):
        if not isinstance(expr, Basic):
            raise TypeError("Invalid type of expr")
        else:
            ops = []
            args = [expr]
            while args:
                a = args.pop()

                if a.args:
                    o = Symbol(type(a).__name__.upper())
                    if a.is_Boolean:
                        ops.append(o*(len(a.args)-1))
                    else:
                        ops.append(o)
                    args.extend(a.args)

    if not ops:
        if visual:
            return S.Zero
        return 0

    ops = Add(*ops)

    if visual:
        return ops

    if ops.is_Number:
        return int(ops)

    return sum(int((a.args or [1])[0]) for a in Add.make_args(ops))


def nfloat(expr, n=15, exponent=False, dkeys=False):
    """Make all Rationals in expr Floats except those in exponents
    (unless the exponents flag is set to True) and those in undefined
    functions. When processing dictionaries, do not modify the keys
    unless ``dkeys=True``.

    Examples
    ========

    >>> from sympy import nfloat, cos, pi, sqrt
    >>> from sympy.abc import x, y
    >>> nfloat(x**4 + x/2 + cos(pi/3) + 1 + sqrt(y))
    x**4 + 0.5*x + sqrt(y) + 1.5
    >>> nfloat(x**4 + sqrt(y), exponent=True)
    x**4.0 + y**0.5

    Container types are not modified:

    >>> type(nfloat((1, 2))) is tuple
    True
    """
    from sympy.matrices.matrixbase import MatrixBase

    kw = {"n": n, "exponent": exponent, "dkeys": dkeys}

    if isinstance(expr, MatrixBase):
        return expr.applyfunc(lambda e: nfloat(e, **kw))

    # handling of iterable containers
    if iterable(expr, exclude=str):
        if isinstance(expr, (dict, Dict)):
            if dkeys:
                args = [tuple((nfloat(i, **kw) for i in a))
                    for a in expr.items()]
            else:
                args = [(k, nfloat(v, **kw)) for k, v in expr.items()]
            if isinstance(expr, dict):
                return type(expr)(args)
            else:
                return expr.func(*args)
        elif isinstance(expr, Basic):
            return expr.func(*[nfloat(a, **kw) for a in expr.args])
        return type(expr)([nfloat(a, **kw) for a in expr])

    rv = sympify(expr)

    if rv.is_Number:
        return Float(rv, n)
    elif rv.is_number:
        # evalf doesn't always set the precision
        rv = rv.n(n)
        if rv.is_Number:
            rv = Float(rv.n(n), n)
        else:
            pass  # pure_complex(rv) is likely True
        return rv
    elif rv.is_Atom:
        return rv
    elif rv.is_Relational:
        args_nfloat = (nfloat(arg, **kw) for arg in rv.args)
        return rv.func(*args_nfloat)


    # watch out for RootOf instances that don't like to have
    # their exponents replaced with Dummies and also sometimes have
    # problems with evaluating at low precision (issue 6393)
    from sympy.polys.rootoftools import RootOf
    rv = rv.xreplace({ro: ro.n(n) for ro in rv.atoms(RootOf)})

    from .power import Pow
    if not exponent:
        reps = [(p, Pow(p.base, Dummy())) for p in rv.atoms(Pow)]
        rv = rv.xreplace(dict(reps))
    rv = rv.n(n)
    if not exponent:
        rv = rv.xreplace({d.exp: p.exp for p, d in reps})
    else:
        # Pow._eval_evalf special cases Integer exponents so if
        # exponent is suppose to be handled we have to do so here
        rv = rv.xreplace(Transform(
            lambda x: Pow(x.base, Float(x.exp, n)),
            lambda x: x.is_Pow and x.exp.is_Integer))

    return rv.xreplace(Transform(
        lambda x: x.func(*nfloat(x.args, n, exponent)),
        lambda x: isinstance(x, Function) and not isinstance(x, AppliedUndef)))


from .symbol import Dummy, Symbol

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\interfaces.py ===
# engine/interfaces.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Define core interfaces used by the engine system."""

from __future__ import annotations

from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict

if TYPE_CHECKING:
    from .base import Connection
    from .base import Engine
    from .cursor import CursorResult
    from .url import URL
    from ..event import _ListenerFnType
    from ..event import dispatcher
    from ..exc import StatementError
    from ..sql import Executable
    from ..sql.compiler import _InsertManyValuesBatch
    from ..sql.compiler import DDLCompiler
    from ..sql.compiler import IdentifierPreparer
    from ..sql.compiler import InsertmanyvaluesSentinelOpts
    from ..sql.compiler import Linting
    from ..sql.compiler import SQLCompiler
    from ..sql.elements import BindParameter
    from ..sql.elements import ClauseElement
    from ..sql.schema import Column
    from ..sql.schema import DefaultGenerator
    from ..sql.schema import SchemaItem
    from ..sql.schema import Sequence as Sequence_SchemaItem
    from ..sql.sqltypes import Integer
    from ..sql.type_api import _TypeMemoDict
    from ..sql.type_api import TypeEngine

ConnectArgsType = Tuple[Sequence[str], MutableMapping[str, Any]]

_T = TypeVar("_T", bound="Any")


class CacheStats(Enum):
    CACHE_HIT = 0
    CACHE_MISS = 1
    CACHING_DISABLED = 2
    NO_CACHE_KEY = 3
    NO_DIALECT_SUPPORT = 4


class ExecuteStyle(Enum):
    """indicates the :term:`DBAPI` cursor method that will be used to invoke
    a statement."""

    EXECUTE = 0
    """indicates cursor.execute() will be used"""

    EXECUTEMANY = 1
    """indicates cursor.executemany() will be used."""

    INSERTMANYVALUES = 2
    """indicates cursor.execute() will be used with an INSERT where the
    VALUES expression will be expanded to accommodate for multiple
    parameter sets

    .. seealso::

        :ref:`engine_insertmanyvalues`

    """


class DBAPIConnection(Protocol):
    """protocol representing a :pep:`249` database connection.

    .. versionadded:: 2.0

    .. seealso::

        `Connection Objects <https://www.python.org/dev/peps/pep-0249/#connection-objects>`_
        - in :pep:`249`

    """  # noqa: E501

    def close(self) -> None: ...

    def commit(self) -> None: ...

    def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor: ...

    def rollback(self) -> None: ...

    autocommit: bool


class DBAPIType(Protocol):
    """protocol representing a :pep:`249` database type.

    .. versionadded:: 2.0

    .. seealso::

        `Type Objects <https://www.python.org/dev/peps/pep-0249/#type-objects>`_
        - in :pep:`249`

    """  # noqa: E501


class DBAPICursor(Protocol):
    """protocol representing a :pep:`249` database cursor.

    .. versionadded:: 2.0

    .. seealso::

        `Cursor Objects <https://www.python.org/dev/peps/pep-0249/#cursor-objects>`_
        - in :pep:`249`

    """  # noqa: E501

    @property
    def description(
        self,
    ) -> _DBAPICursorDescription:
        """The description attribute of the Cursor.

        .. seealso::

            `cursor.description <https://www.python.org/dev/peps/pep-0249/#description>`_
            - in :pep:`249`


        """  # noqa: E501
        ...

    @property
    def rowcount(self) -> int: ...

    arraysize: int

    lastrowid: int

    def close(self) -> None: ...

    def execute(
        self,
        operation: Any,
        parameters: Optional[_DBAPISingleExecuteParams] = None,
    ) -> Any: ...

    def executemany(
        self,
        operation: Any,
        parameters: _DBAPIMultiExecuteParams,
    ) -> Any: ...

    def fetchone(self) -> Optional[Any]: ...

    def fetchmany(self, size: int = ...) -> Sequence[Any]: ...

    def fetchall(self) -> Sequence[Any]: ...

    def setinputsizes(self, sizes: Sequence[Any]) -> None: ...

    def setoutputsize(self, size: Any, column: Any) -> None: ...

    def callproc(
        self, procname: str, parameters: Sequence[Any] = ...
    ) -> Any: ...

    def nextset(self) -> Optional[bool]: ...

    def __getattr__(self, key: str) -> Any: ...


_CoreSingleExecuteParams = Mapping[str, Any]
_MutableCoreSingleExecuteParams = MutableMapping[str, Any]
_CoreMultiExecuteParams = Sequence[_CoreSingleExecuteParams]
_CoreAnyExecuteParams = Union[
    _CoreMultiExecuteParams, _CoreSingleExecuteParams
]

_DBAPISingleExecuteParams = Union[Sequence[Any], _CoreSingleExecuteParams]

_DBAPIMultiExecuteParams = Union[
    Sequence[Sequence[Any]], _CoreMultiExecuteParams
]
_DBAPIAnyExecuteParams = Union[
    _DBAPIMultiExecuteParams, _DBAPISingleExecuteParams
]
_DBAPICursorDescription = Sequence[
    Tuple[
        str,
        "DBAPIType",
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[bool],
    ]
]

_AnySingleExecuteParams = _DBAPISingleExecuteParams
_AnyMultiExecuteParams = _DBAPIMultiExecuteParams
_AnyExecuteParams = _DBAPIAnyExecuteParams

CompiledCacheType = MutableMapping[Any, "Compiled"]
SchemaTranslateMapType = Mapping[Optional[str], Optional[str]]

_ImmutableExecuteOptions = immutabledict[str, Any]

_ParamStyle = Literal[
    "qmark", "numeric", "named", "format", "pyformat", "numeric_dollar"
]

_GenericSetInputSizesType = List[Tuple[str, Any, "TypeEngine[Any]"]]

IsolationLevel = Literal[
    "SERIALIZABLE",
    "REPEATABLE READ",
    "READ COMMITTED",
    "READ UNCOMMITTED",
    "AUTOCOMMIT",
]


class _CoreKnownExecutionOptions(TypedDict, total=False):
    compiled_cache: Optional[CompiledCacheType]
    logging_token: str
    isolation_level: IsolationLevel
    no_parameters: bool
    stream_results: bool
    max_row_buffer: int
    yield_per: int
    insertmanyvalues_page_size: int
    schema_translate_map: Optional[SchemaTranslateMapType]
    preserve_rowcount: bool


_ExecuteOptions = immutabledict[str, Any]
CoreExecuteOptionsParameter = Union[
    _CoreKnownExecutionOptions, Mapping[str, Any]
]


class ReflectedIdentity(TypedDict):
    """represent the reflected IDENTITY structure of a column, corresponding
    to the :class:`_schema.Identity` construct.

    The :class:`.ReflectedIdentity` structure is part of the
    :class:`.ReflectedColumn` structure, which is returned by the
    :meth:`.Inspector.get_columns` method.

    """

    always: bool
    """type of identity column"""

    on_null: bool
    """indicates ON NULL"""

    start: int
    """starting index of the sequence"""

    increment: int
    """increment value of the sequence"""

    minvalue: int
    """the minimum value of the sequence."""

    maxvalue: int
    """the maximum value of the sequence."""

    nominvalue: bool
    """no minimum value of the sequence."""

    nomaxvalue: bool
    """no maximum value of the sequence."""

    cycle: bool
    """allows the sequence to wrap around when the maxvalue
    or minvalue has been reached."""

    cache: Optional[int]
    """number of future values in the
    sequence which are calculated in advance."""

    order: bool
    """if true, renders the ORDER keyword."""


class ReflectedComputed(TypedDict):
    """Represent the reflected elements of a computed column, corresponding
    to the :class:`_schema.Computed` construct.

    The :class:`.ReflectedComputed` structure is part of the
    :class:`.ReflectedColumn` structure, which is returned by the
    :meth:`.Inspector.get_columns` method.

    """

    sqltext: str
    """the expression used to generate this column returned
    as a string SQL expression"""

    persisted: NotRequired[bool]
    """indicates if the value is stored in the table or computed on demand"""


class ReflectedColumn(TypedDict):
    """Dictionary representing the reflected elements corresponding to
    a :class:`_schema.Column` object.

    The :class:`.ReflectedColumn` structure is returned by the
    :class:`.Inspector.get_columns` method.

    """

    name: str
    """column name"""

    type: TypeEngine[Any]
    """column type represented as a :class:`.TypeEngine` instance."""

    nullable: bool
    """boolean flag if the column is NULL or NOT NULL"""

    default: Optional[str]
    """column default expression as a SQL string"""

    autoincrement: NotRequired[bool]
    """database-dependent autoincrement flag.

    This flag indicates if the column has a database-side "autoincrement"
    flag of some kind.   Within SQLAlchemy, other kinds of columns may
    also act as an "autoincrement" column without necessarily having
    such a flag on them.

    See :paramref:`_schema.Column.autoincrement` for more background on
    "autoincrement".

    """

    comment: NotRequired[Optional[str]]
    """comment for the column, if present.
    Only some dialects return this key
    """

    computed: NotRequired[ReflectedComputed]
    """indicates that this column is computed by the database.
    Only some dialects return this key.

    .. versionadded:: 1.3.16 - added support for computed reflection.
    """

    identity: NotRequired[ReflectedIdentity]
    """indicates this column is an IDENTITY column.
    Only some dialects return this key.

    .. versionadded:: 1.4 - added support for identity column reflection.
    """

    dialect_options: NotRequired[Dict[str, Any]]
    """Additional dialect-specific options detected for this reflected
    object"""


class ReflectedConstraint(TypedDict):
    """Dictionary representing the reflected elements corresponding to
    :class:`.Constraint`

    A base class for all constraints
    """

    name: Optional[str]
    """constraint name"""

    comment: NotRequired[Optional[str]]
    """comment for the constraint, if present"""


class ReflectedCheckConstraint(ReflectedConstraint):
    """Dictionary representing the reflected elements corresponding to
    :class:`.CheckConstraint`.

    The :class:`.ReflectedCheckConstraint` structure is returned by the
    :meth:`.Inspector.get_check_constraints` method.

    """

    sqltext: str
    """the check constraint's SQL expression"""

    dialect_options: NotRequired[Dict[str, Any]]
    """Additional dialect-specific options detected for this check constraint

    .. versionadded:: 1.3.8
    """


class ReflectedUniqueConstraint(ReflectedConstraint):
    """Dictionary representing the reflected elements corresponding to
    :class:`.UniqueConstraint`.

    The :class:`.ReflectedUniqueConstraint` structure is returned by the
    :meth:`.Inspector.get_unique_constraints` method.

    """

    column_names: List[str]
    """column names which comprise the unique constraint"""

    duplicates_index: NotRequired[Optional[str]]
    "Indicates if this unique constraint duplicates an index with this name"

    dialect_options: NotRequired[Dict[str, Any]]
    """Additional dialect-specific options detected for this unique
    constraint"""


class ReflectedPrimaryKeyConstraint(ReflectedConstraint):
    """Dictionary representing the reflected elements corresponding to
    :class:`.PrimaryKeyConstraint`.

    The :class:`.ReflectedPrimaryKeyConstraint` structure is returned by the
    :meth:`.Inspector.get_pk_constraint` method.

    """

    constrained_columns: List[str]
    """column names which comprise the primary key"""

    dialect_options: NotRequired[Dict[str, Any]]
    """Additional dialect-specific options detected for this primary key"""


class ReflectedForeignKeyConstraint(ReflectedConstraint):
    """Dictionary representing the reflected elements corresponding to
    :class:`.ForeignKeyConstraint`.

    The :class:`.ReflectedForeignKeyConstraint` structure is returned by
    the :meth:`.Inspector.get_foreign_keys` method.

    """

    constrained_columns: List[str]
    """local column names which comprise the foreign key"""

    referred_schema: Optional[str]
    """schema name of the table being referred"""

    referred_table: str
    """name of the table being referred"""

    referred_columns: List[str]
    """referred column names that correspond to ``constrained_columns``"""

    options: NotRequired[Dict[str, Any]]
    """Additional options detected for this foreign key constraint"""


class ReflectedIndex(TypedDict):
    """Dictionary representing the reflected elements corresponding to
    :class:`.Index`.

    The :class:`.ReflectedIndex` structure is returned by the
    :meth:`.Inspector.get_indexes` method.

    """

    name: Optional[str]
    """index name"""

    column_names: List[Optional[str]]
    """column names which the index references.
    An element of this list is ``None`` if it's an expression and is
    returned in the ``expressions`` list.
    """

    expressions: NotRequired[List[str]]
    """Expressions that compose the index. This list, when present, contains
    both plain column names (that are also in ``column_names``) and
    expressions (that are ``None`` in ``column_names``).
    """

    unique: bool
    """whether or not the index has a unique flag"""

    duplicates_constraint: NotRequired[Optional[str]]
    "Indicates if this index mirrors a constraint with this name"

    include_columns: NotRequired[List[str]]
    """columns to include in the INCLUDE clause for supporting databases.

    .. deprecated:: 2.0

        Legacy value, will be replaced with
        ``index_dict["dialect_options"]["<dialect name>_include"]``

    """

    column_sorting: NotRequired[Dict[str, Tuple[str]]]
    """optional dict mapping column names or expressions to tuple of sort
    keywords, which may include ``asc``, ``desc``, ``nulls_first``,
    ``nulls_last``.

    .. versionadded:: 1.3.5
    """

    dialect_options: NotRequired[Dict[str, Any]]
    """Additional dialect-specific options detected for this index"""


class ReflectedTableComment(TypedDict):
    """Dictionary representing the reflected comment corresponding to
    the :attr:`_schema.Table.comment` attribute.

    The :class:`.ReflectedTableComment` structure is returned by the
    :meth:`.Inspector.get_table_comment` method.

    """

    text: Optional[str]
    """text of the comment"""


class BindTyping(Enum):
    """Define different methods of passing typing information for
    bound parameters in a statement to the database driver.

    .. versionadded:: 2.0

    """

    NONE = 1
    """No steps are taken to pass typing information to the database driver.

    This is the default behavior for databases such as SQLite, MySQL / MariaDB,
    SQL Server.

    """

    SETINPUTSIZES = 2
    """Use the pep-249 setinputsizes method.

    This is only implemented for DBAPIs that support this method and for which
    the SQLAlchemy dialect has the appropriate infrastructure for that dialect
    set up.  Current dialects include python-oracledb, cx_Oracle as well as
    optional support for SQL Server using pyodbc.

    When using setinputsizes, dialects also have a means of only using the
    method for certain datatypes using include/exclude lists.

    When SETINPUTSIZES is used, the :meth:`.Dialect.do_set_input_sizes` method
    is called for each statement executed which has bound parameters.

    """

    RENDER_CASTS = 3
    """Render casts or other directives in the SQL string.

    This method is used for all PostgreSQL dialects, including asyncpg,
    pg8000, psycopg, psycopg2.   Dialects which implement this can choose
    which kinds of datatypes are explicitly cast in SQL statements and which
    aren't.

    When RENDER_CASTS is used, the compiler will invoke the
    :meth:`.SQLCompiler.render_bind_cast` method for the rendered
    string representation of each :class:`.BindParameter` object whose
    dialect-level type sets the :attr:`.TypeEngine.render_bind_cast` attribute.

    The :meth:`.SQLCompiler.render_bind_cast` is also used to render casts
    for one form of "insertmanyvalues" query, when both
    :attr:`.InsertmanyvaluesSentinelOpts.USE_INSERT_FROM_SELECT` and
    :attr:`.InsertmanyvaluesSentinelOpts.RENDER_SELECT_COL_CASTS` are set,
    where the casts are applied to the intermediary columns e.g.
    "INSERT INTO t (a, b, c) SELECT p0::TYP, p1::TYP, p2::TYP "
    "FROM (VALUES (?, ?), (?, ?), ...)".

    .. versionadded:: 2.0.10 - :meth:`.SQLCompiler.render_bind_cast` is now
       used within some elements of the "insertmanyvalues" implementation.


    """


VersionInfoType = Tuple[Union[int, str], ...]
TableKey = Tuple[Optional[str], str]


class Dialect(EventTarget):
    """Define the behavior of a specific database and DB-API combination.

    Any aspect of metadata definition, SQL query generation,
    execution, result-set handling, or anything else which varies
    between databases is defined under the general category of the
    Dialect.  The Dialect acts as a factory for other
    database-specific object implementations including
    ExecutionContext, Compiled, DefaultGenerator, and TypeEngine.

    .. note:: Third party dialects should not subclass :class:`.Dialect`
       directly.  Instead, subclass :class:`.default.DefaultDialect` or
       descendant class.

    """

    CACHE_HIT = CacheStats.CACHE_HIT
    CACHE_MISS = CacheStats.CACHE_MISS
    CACHING_DISABLED = CacheStats.CACHING_DISABLED
    NO_CACHE_KEY = CacheStats.NO_CACHE_KEY
    NO_DIALECT_SUPPORT = CacheStats.NO_DIALECT_SUPPORT

    dispatch: dispatcher[Dialect]

    name: str
    """identifying name for the dialect from a DBAPI-neutral point of view
      (i.e. 'sqlite')
    """

    driver: str
    """identifying name for the dialect's DBAPI"""

    dialect_description: str

    dbapi: Optional[ModuleType]
    """A reference to the DBAPI module object itself.

    SQLAlchemy dialects import DBAPI modules using the classmethod
    :meth:`.Dialect.import_dbapi`. The rationale is so that any dialect
    module can be imported and used to generate SQL statements without the
    need for the actual DBAPI driver to be installed.  Only when an
    :class:`.Engine` is constructed using :func:`.create_engine` does the
    DBAPI get imported; at that point, the creation process will assign
    the DBAPI module to this attribute.

    Dialects should therefore implement :meth:`.Dialect.import_dbapi`
    which will import the necessary module and return it, and then refer
    to ``self.dbapi`` in dialect code in order to refer to the DBAPI module
    contents.

    .. versionchanged:: The :attr:`.Dialect.dbapi` attribute is exclusively
       used as the per-:class:`.Dialect`-instance reference to the DBAPI
       module.   The previous not-fully-documented ``.Dialect.dbapi()``
       classmethod is deprecated and replaced by :meth:`.Dialect.import_dbapi`.

    """

    @util.non_memoized_property
    def loaded_dbapi(self) -> ModuleType:
        """same as .dbapi, but is never None; will raise an error if no
        DBAPI was set up.

        .. versionadded:: 2.0

        """
        raise NotImplementedError()

    positional: bool
    """True if the paramstyle for this Dialect is positional."""

    paramstyle: str
    """the paramstyle to be used (some DB-APIs support multiple
      paramstyles).
    """

    compiler_linting: Linting

    statement_compiler: Type[SQLCompiler]
    """a :class:`.Compiled` class used to compile SQL statements"""

    ddl_compiler: Type[DDLCompiler]
    """a :class:`.Compiled` class used to compile DDL statements"""

    type_compiler_cls: ClassVar[Type[TypeCompiler]]
    """a :class:`.Compiled` class used to compile SQL type objects

    .. versionadded:: 2.0

    """

    type_compiler_instance: TypeCompiler
    """instance of a :class:`.Compiled` class used to compile SQL type
    objects

    .. versionadded:: 2.0

    """

    type_compiler: Any
    """legacy; this is a TypeCompiler class at the class level, a
    TypeCompiler instance at the instance level.

    Refer to type_compiler_instance instead.

    """

    preparer: Type[IdentifierPreparer]
    """a :class:`.IdentifierPreparer` class used to
    quote identifiers.
    """

    identifier_preparer: IdentifierPreparer
    """This element will refer to an instance of :class:`.IdentifierPreparer`
    once a :class:`.DefaultDialect` has been constructed.

    """

    server_version_info: Optional[Tuple[Any, ...]]
    """a tuple containing a version number for the DB backend in use.

    This value is only available for supporting dialects, and is
    typically populated during the initial connection to the database.
    """

    default_schema_name: Optional[str]
    """the name of the default schema.  This value is only available for
    supporting dialects, and is typically populated during the
    initial connection to the database.

    """

    # NOTE: this does not take into effect engine-level isolation level.
    # not clear if this should be changed, seems like it should
    default_isolation_level: Optional[IsolationLevel]
    """the isolation that is implicitly present on new connections"""

    # create_engine()  -> isolation_level  currently goes here
    _on_connect_isolation_level: Optional[IsolationLevel]

    execution_ctx_cls: Type[ExecutionContext]
    """a :class:`.ExecutionContext` class used to handle statement execution"""

    execute_sequence_format: Union[
        Type[Tuple[Any, ...]], Type[Tuple[List[Any]]]
    ]
    """either the 'tuple' or 'list' type, depending on what cursor.execute()
    accepts for the second argument (they vary)."""

    supports_alter: bool
    """``True`` if the database supports ``ALTER TABLE`` - used only for
    generating foreign key constraints in certain circumstances
    """

    max_identifier_length: int
    """The maximum length of identifier names."""
    max_index_name_length: Optional[int]
    """The maximum length of index names if different from
    ``max_identifier_length``."""
    max_constraint_name_length: Optional[int]
    """The maximum length of constraint names if different from
    ``max_identifier_length``."""

    supports_server_side_cursors: bool
    """indicates if the dialect supports server side cursors"""

    server_side_cursors: bool
    """deprecated; indicates if the dialect should attempt to use server
    side cursors by default"""

    supports_sane_rowcount: bool
    """Indicate whether the dialect properly implements rowcount for
      ``UPDATE`` and ``DELETE`` statements.
    """

    supports_sane_multi_rowcount: bool
    """Indicate whether the dialect properly implements rowcount for
      ``UPDATE`` and ``DELETE`` statements when executed via
      executemany.
    """

    supports_empty_insert: bool
    """dialect supports INSERT () VALUES (), i.e. a plain INSERT with no
    columns in it.

    This is not usually supported; an "empty" insert is typically
    suited using either "INSERT..DEFAULT VALUES" or
    "INSERT ... (col) VALUES (DEFAULT)".

    """

    supports_default_values: bool
    """dialect supports INSERT... DEFAULT VALUES syntax"""

    supports_default_metavalue: bool
    """dialect supports INSERT...(col) VALUES (DEFAULT) syntax.

    Most databases support this in some way, e.g. SQLite supports it using
    ``VALUES (NULL)``.    MS SQL Server supports the syntax also however
    is the only included dialect where we have this disabled, as
    MSSQL does not support the field for the IDENTITY column, which is
    usually where we like to make use of the feature.

    """

    default_metavalue_token: str = "DEFAULT"
    """for INSERT... VALUES (DEFAULT) syntax, the token to put in the
    parenthesis.

    E.g. for SQLite this is the keyword "NULL".

    """

    supports_multivalues_insert: bool
    """Target database supports INSERT...VALUES with multiple value
    sets, i.e. INSERT INTO table (cols) VALUES (...), (...), (...), ...

    """

    insert_executemany_returning: bool
    """dialect / driver / database supports some means of providing
    INSERT...RETURNING support when dialect.do_executemany() is used.

    """

    insert_executemany_returning_sort_by_parameter_order: bool
    """dialect / driver / database supports some means of providing
    INSERT...RETURNING support when dialect.do_executemany() is used
    along with the :paramref:`_dml.Insert.returning.sort_by_parameter_order`
    parameter being set.

    """

    update_executemany_returning: bool
    """dialect supports UPDATE..RETURNING with executemany."""

    delete_executemany_returning: bool
    """dialect supports DELETE..RETURNING with executemany."""

    use_insertmanyvalues: bool
    """if True, indicates "insertmanyvalues" functionality should be used
    to allow for ``insert_executemany_returning`` behavior, if possible.

    In practice, setting this to True means:

    if ``supports_multivalues_insert``, ``insert_returning`` and
    ``use_insertmanyvalues`` are all True, the SQL compiler will produce
    an INSERT that will be interpreted by the :class:`.DefaultDialect`
    as an :attr:`.ExecuteStyle.INSERTMANYVALUES` execution that allows
    for INSERT of many rows with RETURNING by rewriting a single-row
    INSERT statement to have multiple VALUES clauses, also executing
    the statement multiple times for a series of batches when large numbers
    of rows are given.

    The parameter is False for the default dialect, and is set to True for
    SQLAlchemy internal dialects SQLite, MySQL/MariaDB, PostgreSQL, SQL Server.
    It remains at False for Oracle Database, which provides native "executemany
    with RETURNING" support and also does not support
    ``supports_multivalues_insert``.  For MySQL/MariaDB, those MySQL dialects
    that don't support RETURNING will not report
    ``insert_executemany_returning`` as True.

    .. versionadded:: 2.0

    .. seealso::

        :ref:`engine_insertmanyvalues`

    """

    use_insertmanyvalues_wo_returning: bool
    """if True, and use_insertmanyvalues is also True, INSERT statements
    that don't include RETURNING will also use "insertmanyvalues".

    .. versionadded:: 2.0

    .. seealso::

        :ref:`engine_insertmanyvalues`

    """

    insertmanyvalues_implicit_sentinel: InsertmanyvaluesSentinelOpts
    """Options indicating the database supports a form of bulk INSERT where
    the autoincrement integer primary key can be reliably used as an ordering
    for INSERTed rows.

    .. versionadded:: 2.0.10

    .. seealso::

        :ref:`engine_insertmanyvalues_returning_order`

    """

    insertmanyvalues_page_size: int
    """Number of rows to render into an individual INSERT..VALUES() statement
    for :attr:`.ExecuteStyle.INSERTMANYVALUES` executions.

    The default dialect defaults this to 1000.

    .. versionadded:: 2.0

    .. seealso::

        :paramref:`_engine.Connection.execution_options.insertmanyvalues_page_size` -
        execution option available on :class:`_engine.Connection`, statements

    """  # noqa: E501

    insertmanyvalues_max_parameters: int
    """Alternate to insertmanyvalues_page_size, will additionally limit
    page size based on number of parameters total in the statement.


    """

    preexecute_autoincrement_sequences: bool
    """True if 'implicit' primary key functions must be executed separately
      in order to get their value, if RETURNING is not used.

      This is currently oriented towards PostgreSQL when the
      ``implicit_returning=False`` parameter is used on a :class:`.Table`
      object.

    """

    insert_returning: bool
    """if the dialect supports RETURNING with INSERT

    .. versionadded:: 2.0

    """

    update_returning: bool
    """if the dialect supports RETURNING with UPDATE

    .. versionadded:: 2.0

    """

    update_returning_multifrom: bool
    """if the dialect supports RETURNING with UPDATE..FROM

    .. versionadded:: 2.0

    """

    delete_returning: bool
    """if the dialect supports RETURNING with DELETE

    .. versionadded:: 2.0

    """

    delete_returning_multifrom: bool
    """if the dialect supports RETURNING with DELETE..FROM

    .. versionadded:: 2.0

    """

    favor_returning_over_lastrowid: bool
    """for backends that support both a lastrowid and a RETURNING insert
    strategy, favor RETURNING for simple single-int pk inserts.

    cursor.lastrowid tends to be more performant on most backends.

    """

    supports_identity_columns: bool
    """target database supports IDENTITY"""

    cte_follows_insert: bool
    """target database, when given a CTE with an INSERT statement, needs
    the CTE to be below the INSERT"""

    colspecs: MutableMapping[Type[TypeEngine[Any]], Type[TypeEngine[Any]]]
    """A dictionary of TypeEngine classes from sqlalchemy.types mapped
      to subclasses that are specific to the dialect class.  This
      dictionary is class-level only and is not accessed from the
      dialect instance itself.
    """

    supports_sequences: bool
    """Indicates if the dialect supports CREATE SEQUENCE or similar."""

    sequences_optional: bool
    """If True, indicates if the :paramref:`_schema.Sequence.optional`
      parameter on the :class:`_schema.Sequence` construct
      should signal to not generate a CREATE SEQUENCE. Applies only to
      dialects that support sequences. Currently used only to allow PostgreSQL
      SERIAL to be used on a column that specifies Sequence() for usage on
      other backends.
    """

    default_sequence_base: int
    """the default value that will be rendered as the "START WITH" portion of
    a CREATE SEQUENCE DDL statement.

    """

    supports_native_enum: bool
    """Indicates if the dialect supports a native ENUM construct.
      This will prevent :class:`_types.Enum` from generating a CHECK
      constraint when that type is used in "native" mode.
    """

    supports_native_boolean: bool
    """Indicates if the dialect supports a native boolean construct.
      This will prevent :class:`_types.Boolean` from generating a CHECK
      constraint when that type is used.
    """

    supports_native_decimal: bool
    """indicates if Decimal objects are handled and returned for precision
    numeric types, or if floats are returned"""

    supports_native_uuid: bool
    """indicates if Python UUID() objects are handled natively by the
    driver for SQL UUID datatypes.

    .. versionadded:: 2.0

    """

    returns_native_bytes: bool
    """indicates if Python bytes() objects are returned natively by the
    driver for SQL "binary" datatypes.

    .. versionadded:: 2.0.11

    """

    construct_arguments: Optional[
        List[Tuple[Type[Union[SchemaItem, ClauseElement]], Mapping[str, Any]]]
    ] = None
    """Optional set of argument specifiers for various SQLAlchemy
    constructs, typically schema items.

    To implement, establish as a series of tuples, as in::

        construct_arguments = [
            (schema.Index, {"using": False, "where": None, "ops": None}),
        ]

    If the above construct is established on the PostgreSQL dialect,
    the :class:`.Index` construct will now accept the keyword arguments
    ``postgresql_using``, ``postgresql_where``, nad ``postgresql_ops``.
    Any other argument specified to the constructor of :class:`.Index`
    which is prefixed with ``postgresql_`` will raise :class:`.ArgumentError`.

    A dialect which does not include a ``construct_arguments`` member will
    not participate in the argument validation system.  For such a dialect,
    any argument name is accepted by all participating constructs, within
    the namespace of arguments prefixed with that dialect name.  The rationale
    here is so that third-party dialects that haven't yet implemented this
    feature continue to function in the old way.

    .. seealso::

        :class:`.DialectKWArgs` - implementing base class which consumes
        :attr:`.DefaultDialect.construct_arguments`


    """

    reflection_options: Sequence[str] = ()
    """Sequence of string names indicating keyword arguments that can be
    established on a :class:`.Table` object which will be passed as
    "reflection options" when using :paramref:`.Table.autoload_with`.

    Current example is "oracle_resolve_synonyms" in the Oracle Database
    dialects.

    """

    dbapi_exception_translation_map: Mapping[str, str] = util.EMPTY_DICT
    """A dictionary of names that will contain as values the names of
       pep-249 exceptions ("IntegrityError", "OperationalError", etc)
       keyed to alternate class names, to support the case where a
       DBAPI has exception classes that aren't named as they are
       referred to (e.g. IntegrityError = MyException).   In the vast
       majority of cases this dictionary is empty.
    """

    supports_comments: bool
    """Indicates the dialect supports comment DDL on tables and columns."""

    inline_comments: bool
    """Indicates the dialect supports comment DDL that's inline with the
    definition of a Table or Column.  If False, this implies that ALTER must
    be used to set table and column comments."""

    supports_constraint_comments: bool
    """Indicates if the dialect supports comment DDL on constraints.

    .. versionadded:: 2.0
    """

    _has_events = False

    supports_statement_cache: bool = True
    """indicates if this dialect supports caching.

    All dialects that are compatible with statement caching should set this
    flag to True directly on each dialect class and subclass that supports
    it.  SQLAlchemy tests that this flag is locally present on each dialect
    subclass before it will use statement caching.  This is to provide
    safety for legacy or new dialects that are not yet fully tested to be
    compliant with SQL statement caching.

    .. versionadded:: 1.4.5

    .. seealso::

        :ref:`engine_thirdparty_caching`

    """

    _supports_statement_cache: bool
    """internal evaluation for supports_statement_cache"""

    bind_typing = BindTyping.NONE
    """define a means of passing typing information to the database and/or
    driver for bound parameters.

    See :class:`.BindTyping` for values.

    .. versionadded:: 2.0

    """

    is_async: bool
    """Whether or not this dialect is intended for asyncio use."""

    has_terminate: bool
    """Whether or not this dialect has a separate "terminate" implementation
    that does not block or require awaiting."""

    engine_config_types: Mapping[str, Any]
    """a mapping of string keys that can be in an engine config linked to
    type conversion functions.

    """

    label_length: Optional[int]
    """optional user-defined max length for SQL labels"""

    include_set_input_sizes: Optional[Set[Any]]
    """set of DBAPI type objects that should be included in
    automatic cursor.setinputsizes() calls.

    This is only used if bind_typing is BindTyping.SET_INPUT_SIZES

    """

    exclude_set_input_sizes: Optional[Set[Any]]
    """set of DBAPI type objects that should be excluded in
    automatic cursor.setinputsizes() calls.

    This is only used if bind_typing is BindTyping.SET_INPUT_SIZES

    """

    supports_simple_order_by_label: bool
    """target database supports ORDER BY <labelname>, where <labelname>
    refers to a label in the columns clause of the SELECT"""

    div_is_floordiv: bool
    """target database treats the / division operator as "floor division" """

    tuple_in_values: bool
    """target database supports tuple IN, i.e. (x, y) IN ((q, p), (r, z))"""

    _bind_typing_render_casts: bool

    _type_memos: MutableMapping[TypeEngine[Any], _TypeMemoDict]

    def _builtin_onconnect(self) -> Optional[_ListenerFnType]:
        raise NotImplementedError()

    def create_connect_args(self, url: URL) -> ConnectArgsType:
        """Build DB-API compatible connection arguments.

        Given a :class:`.URL` object, returns a tuple
        consisting of a ``(*args, **kwargs)`` suitable to send directly
        to the dbapi's connect function.   The arguments are sent to the
        :meth:`.Dialect.connect` method which then runs the DBAPI-level
        ``connect()`` function.

        The method typically makes use of the
        :meth:`.URL.translate_connect_args`
        method in order to generate a dictionary of options.

        The default implementation is::

            def create_connect_args(self, url):
                opts = url.translate_connect_args()
                opts.update(url.query)
                return ([], opts)

        :param url: a :class:`.URL` object

        :return: a tuple of ``(*args, **kwargs)`` which will be passed to the
         :meth:`.Dialect.connect` method.

        .. seealso::

            :meth:`.URL.translate_connect_args`

        """

        raise NotImplementedError()

    @classmethod
    def import_dbapi(cls) -> ModuleType:
        """Import the DBAPI module that is used by this dialect.

        The Python module object returned here will be assigned as an
        instance variable to a constructed dialect under the name
        ``.dbapi``.

        .. versionchanged:: 2.0  The :meth:`.Dialect.import_dbapi` class
           method is renamed from the previous method ``.Dialect.dbapi()``,
           which would be replaced at dialect instantiation time by the
           DBAPI module itself, thus using the same name in two different ways.
           If a ``.Dialect.dbapi()`` classmethod is present on a third-party
           dialect, it will be used and a deprecation warning will be emitted.

        """
        raise NotImplementedError()

    def type_descriptor(self, typeobj: TypeEngine[_T]) -> TypeEngine[_T]:
        """Transform a generic type to a dialect-specific type.

        Dialect classes will usually use the
        :func:`_types.adapt_type` function in the types module to
        accomplish this.

        The returned result is cached *per dialect class* so can
        contain no dialect-instance state.

        """

        raise NotImplementedError()

    def initialize(self, connection: Connection) -> None:
        """Called during strategized creation of the dialect with a
        connection.

        Allows dialects to configure options based on server version info or
        other properties.

        The connection passed here is a SQLAlchemy Connection object,
        with full capabilities.

        The initialize() method of the base dialect should be called via
        super().

        .. note:: as of SQLAlchemy 1.4, this method is called **before**
           any :meth:`_engine.Dialect.on_connect` hooks are called.

        """

    if TYPE_CHECKING:

        def _overrides_default(self, method_name: str) -> bool: ...

    def get_columns(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> List[ReflectedColumn]:
        """Return information about columns in ``table_name``.

        Given a :class:`_engine.Connection`, a string
        ``table_name``, and an optional string ``schema``, return column
        information as a list of dictionaries
        corresponding to the :class:`.ReflectedColumn` dictionary.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_columns`.

        """

        raise NotImplementedError()

    def get_multi_columns(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, List[ReflectedColumn]]]:
        """Return information about columns in all tables in the
        given ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_multi_columns`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def get_pk_constraint(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> ReflectedPrimaryKeyConstraint:
        """Return information about the primary key constraint on
        table_name`.

        Given a :class:`_engine.Connection`, a string
        ``table_name``, and an optional string ``schema``, return primary
        key information as a dictionary corresponding to the
        :class:`.ReflectedPrimaryKeyConstraint` dictionary.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_pk_constraint`.

        """
        raise NotImplementedError()

    def get_multi_pk_constraint(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, ReflectedPrimaryKeyConstraint]]:
        """Return information about primary key constraints in
        all tables in the given ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_multi_pk_constraint`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """
        raise NotImplementedError()

    def get_foreign_keys(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> List[ReflectedForeignKeyConstraint]:
        """Return information about foreign_keys in ``table_name``.

        Given a :class:`_engine.Connection`, a string
        ``table_name``, and an optional string ``schema``, return foreign
        key information as a list of dicts corresponding to the
        :class:`.ReflectedForeignKeyConstraint` dictionary.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_foreign_keys`.
        """

        raise NotImplementedError()

    def get_multi_foreign_keys(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, List[ReflectedForeignKeyConstraint]]]:
        """Return information about foreign_keys in all tables
        in the given ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_multi_foreign_keys`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def get_table_names(
        self, connection: Connection, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        """Return a list of table names for ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_table_names`.

        """

        raise NotImplementedError()

    def get_temp_table_names(
        self, connection: Connection, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        """Return a list of temporary table names on the given connection,
        if supported by the underlying backend.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_temp_table_names`.

        """

        raise NotImplementedError()

    def get_view_names(
        self, connection: Connection, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        """Return a list of all non-materialized view names available in the
        database.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_view_names`.

        :param schema: schema name to query, if not the default schema.

        """

        raise NotImplementedError()

    def get_materialized_view_names(
        self, connection: Connection, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        """Return a list of all materialized view names available in the
        database.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_materialized_view_names`.

        :param schema: schema name to query, if not the default schema.

         .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def get_sequence_names(
        self, connection: Connection, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        """Return a list of all sequence names available in the database.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_sequence_names`.

        :param schema: schema name to query, if not the default schema.

        .. versionadded:: 1.4
        """

        raise NotImplementedError()

    def get_temp_view_names(
        self, connection: Connection, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        """Return a list of temporary view names on the given connection,
        if supported by the underlying backend.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_temp_view_names`.

        """

        raise NotImplementedError()

    def get_schema_names(self, connection: Connection, **kw: Any) -> List[str]:
        """Return a list of all schema names available in the database.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_schema_names`.
        """
        raise NotImplementedError()

    def get_view_definition(
        self,
        connection: Connection,
        view_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> str:
        """Return plain or materialized view definition.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_view_definition`.

        Given a :class:`_engine.Connection`, a string
        ``view_name``, and an optional string ``schema``, return the view
        definition.
        """

        raise NotImplementedError()

    def get_indexes(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> List[ReflectedIndex]:
        """Return information about indexes in ``table_name``.

        Given a :class:`_engine.Connection`, a string
        ``table_name`` and an optional string ``schema``, return index
        information as a list of dictionaries corresponding to the
        :class:`.ReflectedIndex` dictionary.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_indexes`.
        """

        raise NotImplementedError()

    def get_multi_indexes(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, List[ReflectedIndex]]]:
        """Return information about indexes in in all tables
        in the given ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_multi_indexes`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def get_unique_constraints(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> List[ReflectedUniqueConstraint]:
        r"""Return information about unique constraints in ``table_name``.

        Given a string ``table_name`` and an optional string ``schema``, return
        unique constraint information as a list of dicts corresponding
        to the :class:`.ReflectedUniqueConstraint` dictionary.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_unique_constraints`.
        """

        raise NotImplementedError()

    def get_multi_unique_constraints(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, List[ReflectedUniqueConstraint]]]:
        """Return information about unique constraints in all tables
        in the given ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_multi_unique_constraints`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def get_check_constraints(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> List[ReflectedCheckConstraint]:
        r"""Return information about check constraints in ``table_name``.

        Given a string ``table_name`` and an optional string ``schema``, return
        check constraint information as a list of dicts corresponding
        to the :class:`.ReflectedCheckConstraint` dictionary.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_check_constraints`.

        """

        raise NotImplementedError()

    def get_multi_check_constraints(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, List[ReflectedCheckConstraint]]]:
        """Return information about check constraints in all tables
        in the given ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_multi_check_constraints`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def get_table_options(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Return a dictionary of options specified when ``table_name``
        was created.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_table_options`.
        """
        raise NotImplementedError()

    def get_multi_table_options(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, Dict[str, Any]]]:
        """Return a dictionary of options specified when the tables in the
        given schema were created.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_multi_table_options`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """
        raise NotImplementedError()

    def get_table_comment(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> ReflectedTableComment:
        r"""Return the "comment" for the table identified by ``table_name``.

        Given a string ``table_name`` and an optional string ``schema``, return
        table comment information as a dictionary corresponding to the
        :class:`.ReflectedTableComment` dictionary.

        This is an internal dialect method. Applications should use
        :meth:`.Inspector.get_table_comment`.

        :raise: ``NotImplementedError`` for dialects that don't support
         comments.

        .. versionadded:: 1.2

        """

        raise NotImplementedError()

    def get_multi_table_comment(
        self,
        connection: Connection,
        *,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        **kw: Any,
    ) -> Iterable[Tuple[TableKey, ReflectedTableComment]]:
        """Return information about the table comment in all tables
        in the given ``schema``.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.get_multi_table_comment`.

        .. note:: The :class:`_engine.DefaultDialect` provides a default
          implementation that will call the single table method for
          each object returned by :meth:`Dialect.get_table_names`,
          :meth:`Dialect.get_view_names` or
          :meth:`Dialect.get_materialized_view_names` depending on the
          provided ``kind``. Dialects that want to support a faster
          implementation should implement this method.

        .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def normalize_name(self, name: str) -> str:
        """convert the given name to lowercase if it is detected as
        case insensitive.

        This method is only used if the dialect defines
        requires_name_normalize=True.

        """
        raise NotImplementedError()

    def denormalize_name(self, name: str) -> str:
        """convert the given name to a case insensitive identifier
        for the backend if it is an all-lowercase name.

        This method is only used if the dialect defines
        requires_name_normalize=True.

        """
        raise NotImplementedError()

    def has_table(
        self,
        connection: Connection,
        table_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> bool:
        """For internal dialect use, check the existence of a particular table
        or view in the database.

        Given a :class:`_engine.Connection` object, a string table_name and
        optional schema name, return True if the given table exists in the
        database, False otherwise.

        This method serves as the underlying implementation of the
        public facing :meth:`.Inspector.has_table` method, and is also used
        internally to implement the "checkfirst" behavior for methods like
        :meth:`_schema.Table.create` and :meth:`_schema.MetaData.create_all`.

        .. note:: This method is used internally by SQLAlchemy, and is
           published so that third-party dialects may provide an
           implementation. It is **not** the public API for checking for table
           presence. Please use the :meth:`.Inspector.has_table` method.

        .. versionchanged:: 2.0:: :meth:`_engine.Dialect.has_table` now
           formally supports checking for additional table-like objects:

           * any type of views (plain or materialized)
           * temporary tables of any kind

           Previously, these two checks were not formally specified and
           different dialects would vary in their behavior.   The dialect
           testing suite now includes tests for all of these object types,
           and dialects to the degree that the backing database supports views
           or temporary tables should seek to support locating these objects
           for full compliance.

        """

        raise NotImplementedError()

    def has_index(
        self,
        connection: Connection,
        table_name: str,
        index_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> bool:
        """Check the existence of a particular index name in the database.

        Given a :class:`_engine.Connection` object, a string
        ``table_name`` and string index name, return ``True`` if an index of
        the given name on the given table exists, ``False`` otherwise.

        The :class:`.DefaultDialect` implements this in terms of the
        :meth:`.Dialect.has_table` and :meth:`.Dialect.get_indexes` methods,
        however dialects can implement a more performant version.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.has_index`.

        .. versionadded:: 1.4

        """

        raise NotImplementedError()

    def has_sequence(
        self,
        connection: Connection,
        sequence_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> bool:
        """Check the existence of a particular sequence in the database.

        Given a :class:`_engine.Connection` object and a string
        `sequence_name`, return ``True`` if the given sequence exists in
        the database, ``False`` otherwise.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.has_sequence`.
        """

        raise NotImplementedError()

    def has_schema(
        self, connection: Connection, schema_name: str, **kw: Any
    ) -> bool:
        """Check the existence of a particular schema name in the database.

        Given a :class:`_engine.Connection` object, a string
        ``schema_name``, return ``True`` if a schema of the
        given exists, ``False`` otherwise.

        The :class:`.DefaultDialect` implements this by checking
        the presence of ``schema_name`` among the schemas returned by
        :meth:`.Dialect.get_schema_names`,
        however dialects can implement a more performant version.

        This is an internal dialect method. Applications should use
        :meth:`_engine.Inspector.has_schema`.

        .. versionadded:: 2.0

        """

        raise NotImplementedError()

    def _get_server_version_info(self, connection: Connection) -> Any:
        """Retrieve the server version info from the given connection.

        This is used by the default implementation to populate the
        "server_version_info" attribute and is called exactly
        once upon first connect.

        """

        raise NotImplementedError()

    def _get_default_schema_name(self, connection: Connection) -> str:
        """Return the string name of the currently selected schema from
        the given connection.

        This is used by the default implementation to populate the
        "default_schema_name" attribute and is called exactly
        once upon first connect.

        """

        raise NotImplementedError()

    def do_begin(self, dbapi_connection: PoolProxiedConnection) -> None:
        """Provide an implementation of ``connection.begin()``, given a
        DB-API connection.

        The DBAPI has no dedicated "begin" method and it is expected
        that transactions are implicit.  This hook is provided for those
        DBAPIs that might need additional help in this area.

        :param dbapi_connection: a DBAPI connection, typically
         proxied within a :class:`.ConnectionFairy`.

        """

        raise NotImplementedError()

    def do_rollback(self, dbapi_connection: PoolProxiedConnection) -> None:
        """Provide an implementation of ``connection.rollback()``, given
        a DB-API connection.

        :param dbapi_connection: a DBAPI connection, typically
         proxied within a :class:`.ConnectionFairy`.

        """

        raise NotImplementedError()

    def do_commit(self, dbapi_connection: PoolProxiedConnection) -> None:
        """Provide an implementation of ``connection.commit()``, given a
        DB-API connection.

        :param dbapi_connection: a DBAPI connection, typically
         proxied within a :class:`.ConnectionFairy`.

        """

        raise NotImplementedError()

    def do_terminate(self, dbapi_connection: DBAPIConnection) -> None:
        """Provide an implementation of ``connection.close()`` that tries as
        much as possible to not block, given a DBAPI
        connection.

        In the vast majority of cases this just calls .close(), however
        for some asyncio dialects may call upon different API features.

        This hook is called by the :class:`_pool.Pool`
        when a connection is being recycled or has been invalidated.

        .. versionadded:: 1.4.41

        """

        raise NotImplementedError()

    def do_close(self, dbapi_connection: DBAPIConnection) -> None:
        """Provide an implementation of ``connection.close()``, given a DBAPI
        connection.

        This hook is called by the :class:`_pool.Pool`
        when a connection has been
        detached from the pool, or is being returned beyond the normal
        capacity of the pool.

        """

        raise NotImplementedError()

    def _do_ping_w_event(self, dbapi_connection: DBAPIConnection) -> bool:
        raise NotImplementedError()

    def do_ping(self, dbapi_connection: DBAPIConnection) -> bool:
        """ping the DBAPI connection and return True if the connection is
        usable."""
        raise NotImplementedError()

    def do_set_input_sizes(
        self,
        cursor: DBAPICursor,
        list_of_tuples: _GenericSetInputSizesType,
        context: ExecutionContext,
    ) -> Any:
        """invoke the cursor.setinputsizes() method with appropriate arguments

        This hook is called if the :attr:`.Dialect.bind_typing` attribute is
        set to the
        :attr:`.BindTyping.SETINPUTSIZES` value.
        Parameter data is passed in a list of tuples (paramname, dbtype,
        sqltype), where ``paramname`` is the key of the parameter in the
        statement, ``dbtype`` is the DBAPI datatype and ``sqltype`` is the
        SQLAlchemy type. The order of tuples is in the correct parameter order.

        .. versionadded:: 1.4

        .. versionchanged:: 2.0  - setinputsizes mode is now enabled by
           setting :attr:`.Dialect.bind_typing` to
           :attr:`.BindTyping.SETINPUTSIZES`.  Dialects which accept
           a ``use_setinputsizes`` parameter should set this value
           appropriately.


        """
        raise NotImplementedError()

    def create_xid(self) -> Any:
        """Create a two-phase transaction ID.

        This id will be passed to do_begin_twophase(),
        do_rollback_twophase(), do_commit_twophase().  Its format is
        unspecified.
        """

        raise NotImplementedError()

    def do_savepoint(self, connection: Connection, name: str) -> None:
        """Create a savepoint with the given name.

        :param connection: a :class:`_engine.Connection`.
        :param name: savepoint name.

        """

        raise NotImplementedError()

    def do_rollback_to_savepoint(
        self, connection: Connection, name: str
    ) -> None:
        """Rollback a connection to the named savepoint.

        :param connection: a :class:`_engine.Connection`.
        :param name: savepoint name.

        """

        raise NotImplementedError()

    def do_release_savepoint(self, connection: Connection, name: str) -> None:
        """Release the named savepoint on a connection.

        :param connection: a :class:`_engine.Connection`.
        :param name: savepoint name.
        """

        raise NotImplementedError()

    def do_begin_twophase(self, connection: Connection, xid: Any) -> None:
        """Begin a two phase transaction on the given connection.

        :param connection: a :class:`_engine.Connection`.
        :param xid: xid

        """

        raise NotImplementedError()

    def do_prepare_twophase(self, connection: Connection, xid: Any) -> None:
        """Prepare a two phase transaction on the given connection.

        :param connection: a :class:`_engine.Connection`.
        :param xid: xid

        """

        raise NotImplementedError()

    def do_rollback_twophase(
        self,
        connection: Connection,
        xid: Any,
        is_prepared: bool = True,
        recover: bool = False,
    ) -> None:
        """Rollback a two phase transaction on the given connection.

        :param connection: a :class:`_engine.Connection`.
        :param xid: xid
        :param is_prepared: whether or not
         :meth:`.TwoPhaseTransaction.prepare` was called.
        :param recover: if the recover flag was passed.

        """

        raise NotImplementedError()

    def do_commit_twophase(
        self,
        connection: Connection,
        xid: Any,
        is_prepared: bool = True,
        recover: bool = False,
    ) -> None:
        """Commit a two phase transaction on the given connection.


        :param connection: a :class:`_engine.Connection`.
        :param xid: xid
        :param is_prepared: whether or not
         :meth:`.TwoPhaseTransaction.prepare` was called.
        :param recover: if the recover flag was passed.

        """

        raise NotImplementedError()

    def do_recover_twophase(self, connection: Connection) -> List[Any]:
        """Recover list of uncommitted prepared two phase transaction
        identifiers on the given connection.

        :param connection: a :class:`_engine.Connection`.

        """

        raise NotImplementedError()

    def _deliver_insertmanyvalues_batches(
        self,
        connection: Connection,
        cursor: DBAPICursor,
        statement: str,
        parameters: _DBAPIMultiExecuteParams,
        generic_setinputsizes: Optional[_GenericSetInputSizesType],
        context: ExecutionContext,
    ) -> Iterator[_InsertManyValuesBatch]:
        """convert executemany parameters for an INSERT into an iterator
        of statement/single execute values, used by the insertmanyvalues
        feature.

        """
        raise NotImplementedError()

    def do_executemany(
        self,
        cursor: DBAPICursor,
        statement: str,
        parameters: _DBAPIMultiExecuteParams,
        context: Optional[ExecutionContext] = None,
    ) -> None:
        """Provide an implementation of ``cursor.executemany(statement,
        parameters)``."""

        raise NotImplementedError()

    def do_execute(
        self,
        cursor: DBAPICursor,
        statement: str,
        parameters: Optional[_DBAPISingleExecuteParams],
        context: Optional[ExecutionContext] = None,
    ) -> None:
        """Provide an implementation of ``cursor.execute(statement,
        parameters)``."""

        raise NotImplementedError()

    def do_execute_no_params(
        self,
        cursor: DBAPICursor,
        statement: str,
        context: Optional[ExecutionContext] = None,
    ) -> None:
        """Provide an implementation of ``cursor.execute(statement)``.

        The parameter collection should not be sent.

        """

        raise NotImplementedError()

    def is_disconnect(
        self,
        e: Exception,
        connection: Optional[Union[PoolProxiedConnection, DBAPIConnection]],
        cursor: Optional[DBAPICursor],
    ) -> bool:
        """Return True if the given DB-API error indicates an invalid
        connection"""

        raise NotImplementedError()

    def connect(self, *cargs: Any, **cparams: Any) -> DBAPIConnection:
        r"""Establish a connection using this dialect's DBAPI.

        The default implementation of this method is::

            def connect(self, *cargs, **cparams):
                return self.dbapi.connect(*cargs, **cparams)

        The ``*cargs, **cparams`` parameters are generated directly
        from this dialect's :meth:`.Dialect.create_connect_args` method.

        This method may be used for dialects that need to perform programmatic
        per-connection steps when a new connection is procured from the
        DBAPI.


        :param \*cargs: positional parameters returned from the
         :meth:`.Dialect.create_connect_args` method

        :param \*\*cparams: keyword parameters returned from the
         :meth:`.Dialect.create_connect_args` method.

        :return: a DBAPI connection, typically from the :pep:`249` module
         level ``.connect()`` function.

        .. seealso::

            :meth:`.Dialect.create_connect_args`

            :meth:`.Dialect.on_connect`

        """
        raise NotImplementedError()

    def on_connect_url(self, url: URL) -> Optional[Callable[[Any], Any]]:
        """return a callable which sets up a newly created DBAPI connection.

        This method is a new hook that supersedes the
        :meth:`_engine.Dialect.on_connect` method when implemented by a
        dialect.   When not implemented by a dialect, it invokes the
        :meth:`_engine.Dialect.on_connect` method directly to maintain
        compatibility with existing dialects.   There is no deprecation
        for :meth:`_engine.Dialect.on_connect` expected.

        The callable should accept a single argument "conn" which is the
        DBAPI connection itself.  The inner callable has no
        return value.

        E.g.::

            class MyDialect(default.DefaultDialect):
                # ...

                def on_connect_url(self, url):
                    def do_on_connect(connection):
                        connection.execute("SET SPECIAL FLAGS etc")

                    return do_on_connect

        This is used to set dialect-wide per-connection options such as
        isolation modes, Unicode modes, etc.

        This method differs from :meth:`_engine.Dialect.on_connect` in that
        it is passed the :class:`_engine.URL` object that's relevant to the
        connect args.  Normally the only way to get this is from the
        :meth:`_engine.Dialect.on_connect` hook is to look on the
        :class:`_engine.Engine` itself, however this URL object may have been
        replaced by plugins.

        .. note::

            The default implementation of
            :meth:`_engine.Dialect.on_connect_url` is to invoke the
            :meth:`_engine.Dialect.on_connect` method. Therefore if a dialect
            implements this method, the :meth:`_engine.Dialect.on_connect`
            method **will not be called** unless the overriding dialect calls
            it directly from here.

        .. versionadded:: 1.4.3 added :meth:`_engine.Dialect.on_connect_url`
           which normally calls into :meth:`_engine.Dialect.on_connect`.

        :param url: a :class:`_engine.URL` object representing the
         :class:`_engine.URL` that was passed to the
         :meth:`_engine.Dialect.create_connect_args` method.

        :return: a callable that accepts a single DBAPI connection as an
         argument, or None.

        .. seealso::

            :meth:`_engine.Dialect.on_connect`

        """
        return self.on_connect()

    def on_connect(self) -> Optional[Callable[[Any], Any]]:
        """return a callable which sets up a newly created DBAPI connection.

        The callable should accept a single argument "conn" which is the
        DBAPI connection itself.  The inner callable has no
        return value.

        E.g.::

            class MyDialect(default.DefaultDialect):
                # ...

                def on_connect(self):
                    def do_on_connect(connection):
                        connection.execute("SET SPECIAL FLAGS etc")

                    return do_on_connect

        This is used to set dialect-wide per-connection options such as
        isolation modes, Unicode modes, etc.

        The "do_on_connect" callable is invoked by using the
        :meth:`_events.PoolEvents.connect` event
        hook, then unwrapping the DBAPI connection and passing it into the
        callable.

        .. versionchanged:: 1.4 the on_connect hook is no longer called twice
           for the first connection of a dialect.  The on_connect hook is still
           called before the :meth:`_engine.Dialect.initialize` method however.

        .. versionchanged:: 1.4.3 the on_connect hook is invoked from a new
           method on_connect_url that passes the URL that was used to create
           the connect args.   Dialects can implement on_connect_url instead
           of on_connect if they need the URL object that was used for the
           connection in order to get additional context.

        If None is returned, no event listener is generated.

        :return: a callable that accepts a single DBAPI connection as an
         argument, or None.

        .. seealso::

            :meth:`.Dialect.connect` - allows the DBAPI ``connect()`` sequence
            itself to be controlled.

            :meth:`.Dialect.on_connect_url` - supersedes
            :meth:`.Dialect.on_connect` to also receive the
            :class:`_engine.URL` object in context.

        """
        return None

    def reset_isolation_level(self, dbapi_connection: DBAPIConnection) -> None:
        """Given a DBAPI connection, revert its isolation to the default.

        Note that this is a dialect-level method which is used as part
        of the implementation of the :class:`_engine.Connection` and
        :class:`_engine.Engine`
        isolation level facilities; these APIs should be preferred for
        most typical use cases.

        .. seealso::

            :meth:`_engine.Connection.get_isolation_level`
            - view current level

            :attr:`_engine.Connection.default_isolation_level`
            - view default level

            :paramref:`.Connection.execution_options.isolation_level` -
            set per :class:`_engine.Connection` isolation level

            :paramref:`_sa.create_engine.isolation_level` -
            set per :class:`_engine.Engine` isolation level

        """

        raise NotImplementedError()

    def set_isolation_level(
        self, dbapi_connection: DBAPIConnection, level: IsolationLevel
    ) -> None:
        """Given a DBAPI connection, set its isolation level.

        Note that this is a dialect-level method which is used as part
        of the implementation of the :class:`_engine.Connection` and
        :class:`_engine.Engine`
        isolation level facilities; these APIs should be preferred for
        most typical use cases.

        If the dialect also implements the
        :meth:`.Dialect.get_isolation_level_values` method, then the given
        level is guaranteed to be one of the string names within that sequence,
        and the method will not need to anticipate a lookup failure.

        .. seealso::

            :meth:`_engine.Connection.get_isolation_level`
            - view current level

            :attr:`_engine.Connection.default_isolation_level`
            - view default level

            :paramref:`.Connection.execution_options.isolation_level` -
            set per :class:`_engine.Connection` isolation level

            :paramref:`_sa.create_engine.isolation_level` -
            set per :class:`_engine.Engine` isolation level

        """

        raise NotImplementedError()

    def get_isolation_level(
        self, dbapi_connection: DBAPIConnection
    ) -> IsolationLevel:
        """Given a DBAPI connection, return its isolation level.

        When working with a :class:`_engine.Connection` object,
        the corresponding
        DBAPI connection may be procured using the
        :attr:`_engine.Connection.connection` accessor.

        Note that this is a dialect-level method which is used as part
        of the implementation of the :class:`_engine.Connection` and
        :class:`_engine.Engine` isolation level facilities;
        these APIs should be preferred for most typical use cases.


        .. seealso::

            :meth:`_engine.Connection.get_isolation_level`
            - view current level

            :attr:`_engine.Connection.default_isolation_level`
            - view default level

            :paramref:`.Connection.execution_options.isolation_level` -
            set per :class:`_engine.Connection` isolation level

            :paramref:`_sa.create_engine.isolation_level` -
            set per :class:`_engine.Engine` isolation level


        """

        raise NotImplementedError()

    def get_default_isolation_level(
        self, dbapi_conn: DBAPIConnection
    ) -> IsolationLevel:
        """Given a DBAPI connection, return its isolation level, or
        a default isolation level if one cannot be retrieved.

        This method may only raise NotImplementedError and
        **must not raise any other exception**, as it is used implicitly upon
        first connect.

        The method **must return a value** for a dialect that supports
        isolation level settings, as this level is what will be reverted
        towards when a per-connection isolation level change is made.

        The method defaults to using the :meth:`.Dialect.get_isolation_level`
        method unless overridden by a dialect.

        .. versionadded:: 1.3.22

        """
        raise NotImplementedError()

    def get_isolation_level_values(
        self, dbapi_conn: DBAPIConnection
    ) -> Sequence[IsolationLevel]:
        """return a sequence of string isolation level names that are accepted
        by this dialect.

        The available names should use the following conventions:

        * use UPPERCASE names.   isolation level methods will accept lowercase
          names but these are normalized into UPPERCASE before being passed
          along to the dialect.
        * separate words should be separated by spaces, not underscores, e.g.
          ``REPEATABLE READ``.  isolation level names will have underscores
          converted to spaces before being passed along to the dialect.
        * The names for the four standard isolation names to the extent that
          they are supported by the backend should be ``READ UNCOMMITTED``,
          ``READ COMMITTED``, ``REPEATABLE READ``, ``SERIALIZABLE``
        * if the dialect supports an autocommit option it should be provided
          using the isolation level name ``AUTOCOMMIT``.
        * Other isolation modes may also be present, provided that they
          are named in UPPERCASE and use spaces not underscores.

        This function is used so that the default dialect can check that
        a given isolation level parameter is valid, else raises an
        :class:`_exc.ArgumentError`.

        A DBAPI connection is passed to the method, in the unlikely event that
        the dialect needs to interrogate the connection itself to determine
        this list, however it is expected that most backends will return
        a hardcoded list of values.  If the dialect supports "AUTOCOMMIT",
        that value should also be present in the sequence returned.

        The method raises ``NotImplementedError`` by default.  If a dialect
        does not implement this method, then the default dialect will not
        perform any checking on a given isolation level value before passing
        it onto the :meth:`.Dialect.set_isolation_level` method.  This is
        to allow backwards-compatibility with third party dialects that may
        not yet be implementing this method.

        .. versionadded:: 2.0

        """
        raise NotImplementedError()

    def _assert_and_set_isolation_level(
        self, dbapi_conn: DBAPIConnection, level: IsolationLevel
    ) -> None:
        raise NotImplementedError()

    @classmethod
    def get_dialect_cls(cls, url: URL) -> Type[Dialect]:
        """Given a URL, return the :class:`.Dialect` that will be used.

        This is a hook that allows an external plugin to provide functionality
        around an existing dialect, by allowing the plugin to be loaded
        from the url based on an entrypoint, and then the plugin returns
        the actual dialect to be used.

        By default this just returns the cls.

        """
        return cls

    @classmethod
    def get_async_dialect_cls(cls, url: URL) -> Type[Dialect]:
        """Given a URL, return the :class:`.Dialect` that will be used by
        an async engine.

        By default this is an alias of :meth:`.Dialect.get_dialect_cls` and
        just returns the cls. It may be used if a dialect provides
        both a sync and async version under the same name, like the
        ``psycopg`` driver.

        .. versionadded:: 2

        .. seealso::

            :meth:`.Dialect.get_dialect_cls`

        """
        return cls.get_dialect_cls(url)

    @classmethod
    def load_provisioning(cls) -> None:
        """set up the provision.py module for this dialect.

        For dialects that include a provision.py module that sets up
        provisioning followers, this method should initiate that process.

        A typical implementation would be::

            @classmethod
            def load_provisioning(cls):
                __import__("mydialect.provision")

        The default method assumes a module named ``provision.py`` inside
        the owning package of the current dialect, based on the ``__module__``
        attribute::

            @classmethod
            def load_provisioning(cls):
                package = ".".join(cls.__module__.split(".")[0:-1])
                try:
                    __import__(package + ".provision")
                except ImportError:
                    pass

        .. versionadded:: 1.3.14

        """

    @classmethod
    def engine_created(cls, engine: Engine) -> None:
        """A convenience hook called before returning the final
        :class:`_engine.Engine`.

        If the dialect returned a different class from the
        :meth:`.get_dialect_cls`
        method, then the hook is called on both classes, first on
        the dialect class returned by the :meth:`.get_dialect_cls` method and
        then on the class on which the method was called.

        The hook should be used by dialects and/or wrappers to apply special
        events to the engine or its components.   In particular, it allows
        a dialect-wrapping class to apply dialect-level events.

        """

    def get_driver_connection(self, connection: DBAPIConnection) -> Any:
        """Returns the connection object as returned by the external driver
        package.

        For normal dialects that use a DBAPI compliant driver this call
        will just return the ``connection`` passed as argument.
        For dialects that instead adapt a non DBAPI compliant driver, like
        when adapting an asyncio driver, this call will return the
        connection-like object as returned by the driver.

        .. versionadded:: 1.4.24

        """
        raise NotImplementedError()

    def set_engine_execution_options(
        self, engine: Engine, opts: CoreExecuteOptionsParameter
    ) -> None:
        """Establish execution options for a given engine.

        This is implemented by :class:`.DefaultDialect` to establish
        event hooks for new :class:`.Connection` instances created
        by the given :class:`.Engine` which will then invoke the
        :meth:`.Dialect.set_connection_execution_options` method for that
        connection.

        """
        raise NotImplementedError()

    def set_connection_execution_options(
        self, connection: Connection, opts: CoreExecuteOptionsParameter
    ) -> None:
        """Establish execution options for a given connection.

        This is implemented by :class:`.DefaultDialect` in order to implement
        the :paramref:`_engine.Connection.execution_options.isolation_level`
        execution option.  Dialects can intercept various execution options
        which may need to modify state on a particular DBAPI connection.

        .. versionadded:: 1.4

        """
        raise NotImplementedError()

    def get_dialect_pool_class(self, url: URL) -> Type[Pool]:
        """return a Pool class to use for a given URL"""
        raise NotImplementedError()

    def validate_identifier(self, ident: str) -> None:
        """Validates an identifier name, raising an exception if invalid"""


class CreateEnginePlugin:
    """A set of hooks intended to augment the construction of an
    :class:`_engine.Engine` object based on entrypoint names in a URL.

    The purpose of :class:`_engine.CreateEnginePlugin` is to allow third-party
    systems to apply engine, pool and dialect level event listeners without
    the need for the target application to be modified; instead, the plugin
    names can be added to the database URL.  Target applications for
    :class:`_engine.CreateEnginePlugin` include:

    * connection and SQL performance tools, e.g. which use events to track
      number of checkouts and/or time spent with statements

    * connectivity plugins such as proxies

    A rudimentary :class:`_engine.CreateEnginePlugin` that attaches a logger
    to an :class:`_engine.Engine` object might look like::


        import logging

        from sqlalchemy.engine import CreateEnginePlugin
        from sqlalchemy import event


        class LogCursorEventsPlugin(CreateEnginePlugin):
            def __init__(self, url, kwargs):
                # consume the parameter "log_cursor_logging_name" from the
                # URL query
                logging_name = url.query.get(
                    "log_cursor_logging_name", "log_cursor"
                )

                self.log = logging.getLogger(logging_name)

            def update_url(self, url):
                "update the URL to one that no longer includes our parameters"
                return url.difference_update_query(["log_cursor_logging_name"])

            def engine_created(self, engine):
                "attach an event listener after the new Engine is constructed"
                event.listen(engine, "before_cursor_execute", self._log_event)

            def _log_event(
                self,
                conn,
                cursor,
                statement,
                parameters,
                context,
                executemany,
            ):

                self.log.info("Plugin logged cursor event: %s", statement)

    Plugins are registered using entry points in a similar way as that
    of dialects::

        entry_points = {
            "sqlalchemy.plugins": [
                "log_cursor_plugin = myapp.plugins:LogCursorEventsPlugin"
            ]
        }

    A plugin that uses the above names would be invoked from a database
    URL as in::

        from sqlalchemy import create_engine

        engine = create_engine(
            "mysql+pymysql://scott:tiger@localhost/test?"
            "plugin=log_cursor_plugin&log_cursor_logging_name=mylogger"
        )

    The ``plugin`` URL parameter supports multiple instances, so that a URL
    may specify multiple plugins; they are loaded in the order stated
    in the URL::

        engine = create_engine(
            "mysql+pymysql://scott:tiger@localhost/test?"
            "plugin=plugin_one&plugin=plugin_twp&plugin=plugin_three"
        )

    The plugin names may also be passed directly to :func:`_sa.create_engine`
    using the :paramref:`_sa.create_engine.plugins` argument::

        engine = create_engine(
            "mysql+pymysql://scott:tiger@localhost/test", plugins=["myplugin"]
        )

    .. versionadded:: 1.2.3  plugin names can also be specified
       to :func:`_sa.create_engine` as a list

    A plugin may consume plugin-specific arguments from the
    :class:`_engine.URL` object as well as the ``kwargs`` dictionary, which is
    the dictionary of arguments passed to the :func:`_sa.create_engine`
    call.  "Consuming" these arguments includes that they must be removed
    when the plugin initializes, so that the arguments are not passed along
    to the :class:`_engine.Dialect` constructor, where they will raise an
    :class:`_exc.ArgumentError` because they are not known by the dialect.

    As of version 1.4 of SQLAlchemy, arguments should continue to be consumed
    from the ``kwargs`` dictionary directly, by removing the values with a
    method such as ``dict.pop``. Arguments from the :class:`_engine.URL` object
    should be consumed by implementing the
    :meth:`_engine.CreateEnginePlugin.update_url` method, returning a new copy
    of the :class:`_engine.URL` with plugin-specific parameters removed::

        class MyPlugin(CreateEnginePlugin):
            def __init__(self, url, kwargs):
                self.my_argument_one = url.query["my_argument_one"]
                self.my_argument_two = url.query["my_argument_two"]
                self.my_argument_three = kwargs.pop("my_argument_three", None)

            def update_url(self, url):
                return url.difference_update_query(
                    ["my_argument_one", "my_argument_two"]
                )

    Arguments like those illustrated above would be consumed from a
    :func:`_sa.create_engine` call such as::

        from sqlalchemy import create_engine

        engine = create_engine(
            "mysql+pymysql://scott:tiger@localhost/test?"
            "plugin=myplugin&my_argument_one=foo&my_argument_two=bar",
            my_argument_three="bat",
        )

    .. versionchanged:: 1.4

        The :class:`_engine.URL` object is now immutable; a
        :class:`_engine.CreateEnginePlugin` that needs to alter the
        :class:`_engine.URL` should implement the newly added
        :meth:`_engine.CreateEnginePlugin.update_url` method, which
        is invoked after the plugin is constructed.

        For migration, construct the plugin in the following way, checking
        for the existence of the :meth:`_engine.CreateEnginePlugin.update_url`
        method to detect which version is running::

            class MyPlugin(CreateEnginePlugin):
                def __init__(self, url, kwargs):
                    if hasattr(CreateEnginePlugin, "update_url"):
                        # detect the 1.4 API
                        self.my_argument_one = url.query["my_argument_one"]
                        self.my_argument_two = url.query["my_argument_two"]
                    else:
                        # detect the 1.3 and earlier API - mutate the
                        # URL directly
                        self.my_argument_one = url.query.pop("my_argument_one")
                        self.my_argument_two = url.query.pop("my_argument_two")

                    self.my_argument_three = kwargs.pop("my_argument_three", None)

                def update_url(self, url):
                    # this method is only called in the 1.4 version
                    return url.difference_update_query(
                        ["my_argument_one", "my_argument_two"]
                    )

        .. seealso::

            :ref:`change_5526` - overview of the :class:`_engine.URL` change which
            also includes notes regarding :class:`_engine.CreateEnginePlugin`.


    When the engine creation process completes and produces the
    :class:`_engine.Engine` object, it is again passed to the plugin via the
    :meth:`_engine.CreateEnginePlugin.engine_created` hook.  In this hook, additional
    changes can be made to the engine, most typically involving setup of
    events (e.g. those defined in :ref:`core_event_toplevel`).

    """  # noqa: E501

    def __init__(self, url: URL, kwargs: Dict[str, Any]):
        """Construct a new :class:`.CreateEnginePlugin`.

        The plugin object is instantiated individually for each call
        to :func:`_sa.create_engine`.  A single :class:`_engine.
        Engine` will be
        passed to the :meth:`.CreateEnginePlugin.engine_created` method
        corresponding to this URL.

        :param url: the :class:`_engine.URL` object.  The plugin may inspect
         the :class:`_engine.URL` for arguments.  Arguments used by the
         plugin should be removed, by returning an updated :class:`_engine.URL`
         from the :meth:`_engine.CreateEnginePlugin.update_url` method.

         .. versionchanged::  1.4

            The :class:`_engine.URL` object is now immutable, so a
            :class:`_engine.CreateEnginePlugin` that needs to alter the
            :class:`_engine.URL` object should implement the
            :meth:`_engine.CreateEnginePlugin.update_url` method.

        :param kwargs: The keyword arguments passed to
         :func:`_sa.create_engine`.

        """
        self.url = url

    def update_url(self, url: URL) -> URL:
        """Update the :class:`_engine.URL`.

        A new :class:`_engine.URL` should be returned.   This method is
        typically used to consume configuration arguments from the
        :class:`_engine.URL` which must be removed, as they will not be
        recognized by the dialect.  The
        :meth:`_engine.URL.difference_update_query` method is available
        to remove these arguments.   See the docstring at
        :class:`_engine.CreateEnginePlugin` for an example.


        .. versionadded:: 1.4

        """
        raise NotImplementedError()

    def handle_dialect_kwargs(
        self, dialect_cls: Type[Dialect], dialect_args: Dict[str, Any]
    ) -> None:
        """parse and modify dialect kwargs"""

    def handle_pool_kwargs(
        self, pool_cls: Type[Pool], pool_args: Dict[str, Any]
    ) -> None:
        """parse and modify pool kwargs"""

    def engine_created(self, engine: Engine) -> None:
        """Receive the :class:`_engine.Engine`
        object when it is fully constructed.

        The plugin may make additional changes to the engine, such as
        registering engine or connection pool events.

        """


class ExecutionContext:
    """A messenger object for a Dialect that corresponds to a single
    execution.

    """

    engine: Engine
    """engine which the Connection is associated with"""

    connection: Connection
    """Connection object which can be freely used by default value
      generators to execute SQL.  This Connection should reference the
      same underlying connection/transactional resources of
      root_connection."""

    root_connection: Connection
    """Connection object which is the source of this ExecutionContext."""

    dialect: Dialect
    """dialect which created this ExecutionContext."""

    cursor: DBAPICursor
    """DB-API cursor procured from the connection"""

    compiled: Optional[Compiled]
    """if passed to constructor, sqlalchemy.engine.base.Compiled object
      being executed"""

    statement: str
    """string version of the statement to be executed.  Is either
      passed to the constructor, or must be created from the
      sql.Compiled object by the time pre_exec() has completed."""

    invoked_statement: Optional[Executable]
    """The Executable statement object that was given in the first place.

    This should be structurally equivalent to compiled.statement, but not
    necessarily the same object as in a caching scenario the compiled form
    will have been extracted from the cache.

    """

    parameters: _AnyMultiExecuteParams
    """bind parameters passed to the execute() or exec_driver_sql() methods.

    These are always stored as a list of parameter entries.  A single-element
    list corresponds to a ``cursor.execute()`` call and a multiple-element
    list corresponds to ``cursor.executemany()``, except in the case
    of :attr:`.ExecuteStyle.INSERTMANYVALUES` which will use
    ``cursor.execute()`` one or more times.

    """

    no_parameters: bool
    """True if the execution style does not use parameters"""

    isinsert: bool
    """True if the statement is an INSERT."""

    isupdate: bool
    """True if the statement is an UPDATE."""

    execute_style: ExecuteStyle
    """the style of DBAPI cursor method that will be used to execute
    a statement.

    .. versionadded:: 2.0

    """

    executemany: bool
    """True if the context has a list of more than one parameter set.

    Historically this attribute links to whether ``cursor.execute()`` or
    ``cursor.executemany()`` will be used.  It also can now mean that
    "insertmanyvalues" may be used which indicates one or more
    ``cursor.execute()`` calls.

    """

    prefetch_cols: util.generic_fn_descriptor[Optional[Sequence[Column[Any]]]]
    """a list of Column objects for which a client-side default
      was fired off.  Applies to inserts and updates."""

    postfetch_cols: util.generic_fn_descriptor[Optional[Sequence[Column[Any]]]]
    """a list of Column objects for which a server-side default or
      inline SQL expression value was fired off.  Applies to inserts
      and updates."""

    execution_options: _ExecuteOptions
    """Execution options associated with the current statement execution"""

    @classmethod
    def _init_ddl(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
        compiled_ddl: DDLCompiler,
    ) -> ExecutionContext:
        raise NotImplementedError()

    @classmethod
    def _init_compiled(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
        compiled: SQLCompiler,
        parameters: _CoreMultiExecuteParams,
        invoked_statement: Executable,
        extracted_parameters: Optional[Sequence[BindParameter[Any]]],
        cache_hit: CacheStats = CacheStats.CACHING_DISABLED,
    ) -> ExecutionContext:
        raise NotImplementedError()

    @classmethod
    def _init_statement(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
        statement: str,
        parameters: _DBAPIMultiExecuteParams,
    ) -> ExecutionContext:
        raise NotImplementedError()

    @classmethod
    def _init_default(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
    ) -> ExecutionContext:
        raise NotImplementedError()

    def _exec_default(
        self,
        column: Optional[Column[Any]],
        default: DefaultGenerator,
        type_: Optional[TypeEngine[Any]],
    ) -> Any:
        raise NotImplementedError()

    def _prepare_set_input_sizes(
        self,
    ) -> Optional[List[Tuple[str, Any, TypeEngine[Any]]]]:
        raise NotImplementedError()

    def _get_cache_stats(self) -> str:
        raise NotImplementedError()

    def _setup_result_proxy(self) -> CursorResult[Any]:
        raise NotImplementedError()

    def fire_sequence(self, seq: Sequence_SchemaItem, type_: Integer) -> int:
        """given a :class:`.Sequence`, invoke it and return the next int
        value"""
        raise NotImplementedError()

    def create_cursor(self) -> DBAPICursor:
        """Return a new cursor generated from this ExecutionContext's
        connection.

        Some dialects may wish to change the behavior of
        connection.cursor(), such as postgresql which may return a PG
        "server side" cursor.
        """

        raise NotImplementedError()

    def pre_exec(self) -> None:
        """Called before an execution of a compiled statement.

        If a compiled statement was passed to this ExecutionContext,
        the `statement` and `parameters` datamembers must be
        initialized after this statement is complete.
        """

        raise NotImplementedError()

    def get_out_parameter_values(
        self, out_param_names: Sequence[str]
    ) -> Sequence[Any]:
        """Return a sequence of OUT parameter values from a cursor.

        For dialects that support OUT parameters, this method will be called
        when there is a :class:`.SQLCompiler` object which has the
        :attr:`.SQLCompiler.has_out_parameters` flag set.  This flag in turn
        will be set to True if the statement itself has :class:`.BindParameter`
        objects that have the ``.isoutparam`` flag set which are consumed by
        the :meth:`.SQLCompiler.visit_bindparam` method.  If the dialect
        compiler produces :class:`.BindParameter` objects with ``.isoutparam``
        set which are not handled by :meth:`.SQLCompiler.visit_bindparam`, it
        should set this flag explicitly.

        The list of names that were rendered for each bound parameter
        is passed to the method.  The method should then return a sequence of
        values corresponding to the list of parameter objects. Unlike in
        previous SQLAlchemy versions, the values can be the **raw values** from
        the DBAPI; the execution context will apply the appropriate type
        handler based on what's present in self.compiled.binds and update the
        values.  The processed dictionary will then be made available via the
        ``.out_parameters`` collection on the result object.  Note that
        SQLAlchemy 1.4 has multiple kinds of result object as part of the 2.0
        transition.

        .. versionadded:: 1.4 - added
           :meth:`.ExecutionContext.get_out_parameter_values`, which is invoked
           automatically by the :class:`.DefaultExecutionContext` when there
           are :class:`.BindParameter` objects with the ``.isoutparam`` flag
           set.  This replaces the practice of setting out parameters within
           the now-removed ``get_result_proxy()`` method.

        """
        raise NotImplementedError()

    def post_exec(self) -> None:
        """Called after the execution of a compiled statement.

        If a compiled statement was passed to this ExecutionContext,
        the `last_insert_ids`, `last_inserted_params`, etc.
        datamembers should be available after this method completes.
        """

        raise NotImplementedError()

    def handle_dbapi_exception(self, e: BaseException) -> None:
        """Receive a DBAPI exception which occurred upon execute, result
        fetch, etc."""

        raise NotImplementedError()

    def lastrow_has_defaults(self) -> bool:
        """Return True if the last INSERT or UPDATE row contained
        inlined or database-side defaults.
        """

        raise NotImplementedError()

    def get_rowcount(self) -> Optional[int]:
        """Return the DBAPI ``cursor.rowcount`` value, or in some
        cases an interpreted value.

        See :attr:`_engine.CursorResult.rowcount` for details on this.

        """

        raise NotImplementedError()

    def fetchall_for_returning(self, cursor: DBAPICursor) -> Sequence[Any]:
        """For a RETURNING result, deliver cursor.fetchall() from the
        DBAPI cursor.

        This is a dialect-specific hook for dialects that have special
        considerations when calling upon the rows delivered for a
        "RETURNING" statement.   Default implementation is
        ``cursor.fetchall()``.

        This hook is currently used only by the :term:`insertmanyvalues`
        feature.   Dialects that don't set ``use_insertmanyvalues=True``
        don't need to consider this hook.

        .. versionadded:: 2.0.10

        """
        raise NotImplementedError()


class ConnectionEventsTarget(EventTarget):
    """An object which can accept events from :class:`.ConnectionEvents`.

    Includes :class:`_engine.Connection` and :class:`_engine.Engine`.

    .. versionadded:: 2.0

    """

    dispatch: dispatcher[ConnectionEventsTarget]


Connectable = ConnectionEventsTarget


class ExceptionContext:
    """Encapsulate information about an error condition in progress.

    This object exists solely to be passed to the
    :meth:`_events.DialectEvents.handle_error` event,
    supporting an interface that
    can be extended without backwards-incompatibility.


    """

    __slots__ = ()

    dialect: Dialect
    """The :class:`_engine.Dialect` in use.

    This member is present for all invocations of the event hook.

    .. versionadded:: 2.0

    """

    connection: Optional[Connection]
    """The :class:`_engine.Connection` in use during the exception.

    This member is present, except in the case of a failure when
    first connecting.

    .. seealso::

        :attr:`.ExceptionContext.engine`


    """

    engine: Optional[Engine]
    """The :class:`_engine.Engine` in use during the exception.

    This member is present in all cases except for when handling an error
    within the connection pool "pre-ping" process.

    """

    cursor: Optional[DBAPICursor]
    """The DBAPI cursor object.

    May be None.

    """

    statement: Optional[str]
    """String SQL statement that was emitted directly to the DBAPI.

    May be None.

    """

    parameters: Optional[_DBAPIAnyExecuteParams]
    """Parameter collection that was emitted directly to the DBAPI.

    May be None.

    """

    original_exception: BaseException
    """The exception object which was caught.

    This member is always present.

    """

    sqlalchemy_exception: Optional[StatementError]
    """The :class:`sqlalchemy.exc.StatementError` which wraps the original,
    and will be raised if exception handling is not circumvented by the event.

    May be None, as not all exception types are wrapped by SQLAlchemy.
    For DBAPI-level exceptions that subclass the dbapi's Error class, this
    field will always be present.

    """

    chained_exception: Optional[BaseException]
    """The exception that was returned by the previous handler in the
    exception chain, if any.

    If present, this exception will be the one ultimately raised by
    SQLAlchemy unless a subsequent handler replaces it.

    May be None.

    """

    execution_context: Optional[ExecutionContext]
    """The :class:`.ExecutionContext` corresponding to the execution
    operation in progress.

    This is present for statement execution operations, but not for
    operations such as transaction begin/end.  It also is not present when
    the exception was raised before the :class:`.ExecutionContext`
    could be constructed.

    Note that the :attr:`.ExceptionContext.statement` and
    :attr:`.ExceptionContext.parameters` members may represent a
    different value than that of the :class:`.ExecutionContext`,
    potentially in the case where a
    :meth:`_events.ConnectionEvents.before_cursor_execute` event or similar
    modified the statement/parameters to be sent.

    May be None.

    """

    is_disconnect: bool
    """Represent whether the exception as occurred represents a "disconnect"
    condition.

    This flag will always be True or False within the scope of the
    :meth:`_events.DialectEvents.handle_error` handler.

    SQLAlchemy will defer to this flag in order to determine whether or not
    the connection should be invalidated subsequently.    That is, by
    assigning to this flag, a "disconnect" event which then results in
    a connection and pool invalidation can be invoked or prevented by
    changing this flag.


    .. note:: The pool "pre_ping" handler enabled using the
        :paramref:`_sa.create_engine.pool_pre_ping` parameter does **not**
        consult this event before deciding if the "ping" returned false,
        as opposed to receiving an unhandled error.   For this use case, the
        :ref:`legacy recipe based on engine_connect() may be used
        <pool_disconnects_pessimistic_custom>`.  A future API allow more
        comprehensive customization of the "disconnect" detection mechanism
        across all functions.

    """

    invalidate_pool_on_disconnect: bool
    """Represent whether all connections in the pool should be invalidated
    when a "disconnect" condition is in effect.

    Setting this flag to False within the scope of the
    :meth:`_events.DialectEvents.handle_error`
    event will have the effect such
    that the full collection of connections in the pool will not be
    invalidated during a disconnect; only the current connection that is the
    subject of the error will actually be invalidated.

    The purpose of this flag is for custom disconnect-handling schemes where
    the invalidation of other connections in the pool is to be performed
    based on other conditions, or even on a per-connection basis.

    """

    is_pre_ping: bool
    """Indicates if this error is occurring within the "pre-ping" step
    performed when :paramref:`_sa.create_engine.pool_pre_ping` is set to
    ``True``.  In this mode, the :attr:`.ExceptionContext.engine` attribute
    will be ``None``.  The dialect in use is accessible via the
    :attr:`.ExceptionContext.dialect` attribute.

    .. versionadded:: 2.0.5

    """


class AdaptedConnection:
    """Interface of an adapted connection object to support the DBAPI protocol.

    Used by asyncio dialects to provide a sync-style pep-249 facade on top
    of the asyncio connection/cursor API provided by the driver.

    .. versionadded:: 1.4.24

    """

    __slots__ = ("_connection",)

    _connection: Any

    @property
    def driver_connection(self) -> Any:
        """The connection object as returned by the driver after a connect."""
        return self._connection

    def run_async(self, fn: Callable[[Any], Awaitable[_T]]) -> _T:
        """Run the awaitable returned by the given function, which is passed
        the raw asyncio driver connection.

        This is used to invoke awaitable-only methods on the driver connection
        within the context of a "synchronous" method, like a connection
        pool event handler.

        E.g.::

            engine = create_async_engine(...)


            @event.listens_for(engine.sync_engine, "connect")
            def register_custom_types(
                dbapi_connection,  # ...
            ):
                dbapi_connection.run_async(
                    lambda connection: connection.set_type_codec(
                        "MyCustomType", encoder, decoder, ...
                    )
                )

        .. versionadded:: 1.4.30

        .. seealso::

            :ref:`asyncio_events_run_async`

        """
        return await_only(fn(self._connection))

    def __repr__(self) -> str:
        return "<AdaptedConnection %s>" % self._connection