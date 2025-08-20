
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\relationships.py ===
# orm/relationships.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Heuristics related to join conditions as used in
:func:`_orm.relationship`.

Provides the :class:`.JoinCondition` object, which encapsulates
SQL annotation and aliasing behavior focused on the `primaryjoin`
and `secondaryjoin` aspects of :func:`_orm.relationship`.

"""
from __future__ import annotations

import collections
from collections import abc
import dataclasses
import inspect as _py_inspect
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref

from . import attributes
from . import strategy_options
from ._typing import insp_is_aliased_class
from ._typing import is_has_collection_adapter
from .base import _DeclarativeMapped
from .base import _is_mapped_class
from .base import class_mapper
from .base import DynamicMapped
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .base import state_str
from .base import WriteOnlyMapped
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .interfaces import PropComparator
from .interfaces import RelationshipDirection
from .interfaces import StrategizedProperty
from .util import _orm_annotate
from .util import _orm_deannotate
from .util import CascadeOptions
from .. import exc as sa_exc
from .. import Exists
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..inspection import inspect
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql._typing import _ColumnExpressionArgument
from ..sql._typing import _HasClauseElement
from ..sql.annotation import _safe_annotate
from ..sql.elements import ColumnClause
from ..sql.elements import ColumnElement
from ..sql.util import _deep_annotate
from ..sql.util import _deep_deannotate
from ..sql.util import _shallow_annotate
from ..sql.util import adapt_criterion_to_null
from ..sql.util import ClauseAdapter
from ..sql.util import join_condition
from ..sql.util import selectables_overlap
from ..sql.util import visit_binary_product
from ..util.typing import de_optionalize_union_types
from ..util.typing import Literal
from ..util.typing import resolve_name_to_real_class_name

if typing.TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _ExternalEntityType
    from ._typing import _IdentityKeyType
    from ._typing import _InstanceDict
    from ._typing import _InternalEntityType
    from ._typing import _O
    from ._typing import _RegistryType
    from .base import Mapped
    from .clsregistry import _class_resolver
    from .clsregistry import _ModNS
    from .decl_base import _ClassScanMapperConfig
    from .dependency import DependencyProcessor
    from .mapper import Mapper
    from .query import Query
    from .session import Session
    from .state import InstanceState
    from .strategies import LazyLoader
    from .util import AliasedClass
    from .util import AliasedInsp
    from ..sql._typing import _CoreAdapterProto
    from ..sql._typing import _EquivalentColumnMap
    from ..sql._typing import _InfoType
    from ..sql.annotation import _AnnotationDict
    from ..sql.annotation import SupportsAnnotations
    from ..sql.elements import BinaryExpression
    from ..sql.elements import BindParameter
    from ..sql.elements import ClauseElement
    from ..sql.schema import Table
    from ..sql.selectable import FromClause
    from ..util.typing import _AnnotationScanType
    from ..util.typing import RODescriptorReference

_T = TypeVar("_T", bound=Any)
_T1 = TypeVar("_T1", bound=Any)
_T2 = TypeVar("_T2", bound=Any)

_PT = TypeVar("_PT", bound=Any)

_PT2 = TypeVar("_PT2", bound=Any)


_RelationshipArgumentType = Union[
    str,
    Type[_T],
    Callable[[], Type[_T]],
    "Mapper[_T]",
    "AliasedClass[_T]",
    Callable[[], "Mapper[_T]"],
    Callable[[], "AliasedClass[_T]"],
]

_LazyLoadArgumentType = Literal[
    "select",
    "joined",
    "selectin",
    "subquery",
    "raise",
    "raise_on_sql",
    "noload",
    "immediate",
    "write_only",
    "dynamic",
    True,
    False,
    None,
]


_RelationshipJoinConditionArgument = Union[
    str, _ColumnExpressionArgument[bool]
]
_RelationshipSecondaryArgument = Union[
    "FromClause", str, Callable[[], "FromClause"]
]
_ORMOrderByArgument = Union[
    Literal[False],
    str,
    _ColumnExpressionArgument[Any],
    Callable[[], _ColumnExpressionArgument[Any]],
    Callable[[], Iterable[_ColumnExpressionArgument[Any]]],
    Iterable[Union[str, _ColumnExpressionArgument[Any]]],
]
ORMBackrefArgument = Union[str, Tuple[str, Dict[str, Any]]]

_ORMColCollectionElement = Union[
    ColumnClause[Any],
    _HasClauseElement[Any],
    roles.DMLColumnRole,
    "Mapped[Any]",
]
_ORMColCollectionArgument = Union[
    str,
    Sequence[_ORMColCollectionElement],
    Callable[[], Sequence[_ORMColCollectionElement]],
    Callable[[], _ORMColCollectionElement],
    _ORMColCollectionElement,
]


_CEA = TypeVar("_CEA", bound=_ColumnExpressionArgument[Any])

_CE = TypeVar("_CE", bound="ColumnElement[Any]")


_ColumnPairIterable = Iterable[Tuple[ColumnElement[Any], ColumnElement[Any]]]

_ColumnPairs = Sequence[Tuple[ColumnElement[Any], ColumnElement[Any]]]

_MutableColumnPairs = List[Tuple[ColumnElement[Any], ColumnElement[Any]]]


def remote(expr: _CEA) -> _CEA:
    """Annotate a portion of a primaryjoin expression
    with a 'remote' annotation.

    See the section :ref:`relationship_custom_foreign` for a
    description of use.

    .. seealso::

        :ref:`relationship_custom_foreign`

        :func:`.foreign`

    """
    return _annotate_columns(  # type: ignore
        coercions.expect(roles.ColumnArgumentRole, expr), {"remote": True}
    )


def foreign(expr: _CEA) -> _CEA:
    """Annotate a portion of a primaryjoin expression
    with a 'foreign' annotation.

    See the section :ref:`relationship_custom_foreign` for a
    description of use.

    .. seealso::

        :ref:`relationship_custom_foreign`

        :func:`.remote`

    """

    return _annotate_columns(  # type: ignore
        coercions.expect(roles.ColumnArgumentRole, expr), {"foreign": True}
    )


@dataclasses.dataclass
class _RelationshipArg(Generic[_T1, _T2]):
    """stores a user-defined parameter value that must be resolved and
    parsed later at mapper configuration time.

    """

    __slots__ = "name", "argument", "resolved"
    name: str
    argument: _T1
    resolved: Optional[_T2]

    def _is_populated(self) -> bool:
        return self.argument is not None

    def _resolve_against_registry(
        self, clsregistry_resolver: Callable[[str, bool], _class_resolver]
    ) -> None:
        attr_value = self.argument

        if isinstance(attr_value, str):
            self.resolved = clsregistry_resolver(
                attr_value, self.name == "secondary"
            )()
        elif callable(attr_value) and not _is_mapped_class(attr_value):
            self.resolved = attr_value()
        else:
            self.resolved = attr_value


_RelationshipOrderByArg = Union[Literal[False], Tuple[ColumnElement[Any], ...]]


class _RelationshipArgs(NamedTuple):
    """stores user-passed parameters that are resolved at mapper configuration
    time.

    """

    secondary: _RelationshipArg[
        Optional[_RelationshipSecondaryArgument],
        Optional[FromClause],
    ]
    primaryjoin: _RelationshipArg[
        Optional[_RelationshipJoinConditionArgument],
        Optional[ColumnElement[Any]],
    ]
    secondaryjoin: _RelationshipArg[
        Optional[_RelationshipJoinConditionArgument],
        Optional[ColumnElement[Any]],
    ]
    order_by: _RelationshipArg[_ORMOrderByArgument, _RelationshipOrderByArg]
    foreign_keys: _RelationshipArg[
        Optional[_ORMColCollectionArgument], Set[ColumnElement[Any]]
    ]
    remote_side: _RelationshipArg[
        Optional[_ORMColCollectionArgument], Set[ColumnElement[Any]]
    ]


@log.class_logger
class RelationshipProperty(
    _IntrospectsAnnotations, StrategizedProperty[_T], log.Identified
):
    """Describes an object property that holds a single item or list
    of items that correspond to a related database table.

    Public constructor is the :func:`_orm.relationship` function.

    .. seealso::

        :ref:`relationship_config_toplevel`

    """

    strategy_wildcard_key = strategy_options._RELATIONSHIP_TOKEN
    inherit_cache = True
    """:meta private:"""

    _links_to_entity = True
    _is_relationship = True

    _overlaps: Sequence[str]

    _lazy_strategy: LazyLoader

    _persistence_only = dict(
        passive_deletes=False,
        passive_updates=True,
        enable_typechecks=True,
        active_history=False,
        cascade_backrefs=False,
    )

    _dependency_processor: Optional[DependencyProcessor] = None

    primaryjoin: ColumnElement[bool]
    secondaryjoin: Optional[ColumnElement[bool]]
    secondary: Optional[FromClause]
    _join_condition: JoinCondition
    order_by: _RelationshipOrderByArg

    _user_defined_foreign_keys: Set[ColumnElement[Any]]
    _calculated_foreign_keys: Set[ColumnElement[Any]]

    remote_side: Set[ColumnElement[Any]]
    local_columns: Set[ColumnElement[Any]]

    synchronize_pairs: _ColumnPairs
    secondary_synchronize_pairs: Optional[_ColumnPairs]

    local_remote_pairs: Optional[_ColumnPairs]

    direction: RelationshipDirection

    _init_args: _RelationshipArgs

    def __init__(
        self,
        argument: Optional[_RelationshipArgumentType[_T]] = None,
        secondary: Optional[_RelationshipSecondaryArgument] = None,
        *,
        uselist: Optional[bool] = None,
        collection_class: Optional[
            Union[Type[Collection[Any]], Callable[[], Collection[Any]]]
        ] = None,
        primaryjoin: Optional[_RelationshipJoinConditionArgument] = None,
        secondaryjoin: Optional[_RelationshipJoinConditionArgument] = None,
        back_populates: Optional[str] = None,
        order_by: _ORMOrderByArgument = False,
        backref: Optional[ORMBackrefArgument] = None,
        overlaps: Optional[str] = None,
        post_update: bool = False,
        cascade: str = "save-update, merge",
        viewonly: bool = False,
        attribute_options: Optional[_AttributeOptions] = None,
        lazy: _LazyLoadArgumentType = "select",
        passive_deletes: Union[Literal["all"], bool] = False,
        passive_updates: bool = True,
        active_history: bool = False,
        enable_typechecks: bool = True,
        foreign_keys: Optional[_ORMColCollectionArgument] = None,
        remote_side: Optional[_ORMColCollectionArgument] = None,
        join_depth: Optional[int] = None,
        comparator_factory: Optional[
            Type[RelationshipProperty.Comparator[Any]]
        ] = None,
        single_parent: bool = False,
        innerjoin: bool = False,
        distinct_target_key: Optional[bool] = None,
        load_on_pending: bool = False,
        query_class: Optional[Type[Query[Any]]] = None,
        info: Optional[_InfoType] = None,
        omit_join: Literal[None, False] = None,
        sync_backref: Optional[bool] = None,
        doc: Optional[str] = None,
        bake_queries: Literal[True] = True,
        cascade_backrefs: Literal[False] = False,
        _local_remote_pairs: Optional[_ColumnPairs] = None,
        _legacy_inactive_history_style: bool = False,
    ):
        super().__init__(attribute_options=attribute_options)

        self.uselist = uselist
        self.argument = argument

        self._init_args = _RelationshipArgs(
            _RelationshipArg("secondary", secondary, None),
            _RelationshipArg("primaryjoin", primaryjoin, None),
            _RelationshipArg("secondaryjoin", secondaryjoin, None),
            _RelationshipArg("order_by", order_by, None),
            _RelationshipArg("foreign_keys", foreign_keys, None),
            _RelationshipArg("remote_side", remote_side, None),
        )

        self.post_update = post_update
        self.viewonly = viewonly
        if viewonly:
            self._warn_for_persistence_only_flags(
                passive_deletes=passive_deletes,
                passive_updates=passive_updates,
                enable_typechecks=enable_typechecks,
                active_history=active_history,
                cascade_backrefs=cascade_backrefs,
            )
        if viewonly and sync_backref:
            raise sa_exc.ArgumentError(
                "sync_backref and viewonly cannot both be True"
            )
        self.sync_backref = sync_backref
        self.lazy = lazy
        self.single_parent = single_parent
        self.collection_class = collection_class
        self.passive_deletes = passive_deletes

        if cascade_backrefs:
            raise sa_exc.ArgumentError(
                "The 'cascade_backrefs' parameter passed to "
                "relationship() may only be set to False."
            )

        self.passive_updates = passive_updates
        self.enable_typechecks = enable_typechecks
        self.query_class = query_class
        self.innerjoin = innerjoin
        self.distinct_target_key = distinct_target_key
        self.doc = doc
        self.active_history = active_history
        self._legacy_inactive_history_style = _legacy_inactive_history_style

        self.join_depth = join_depth
        if omit_join:
            util.warn(
                "setting omit_join to True is not supported; selectin "
                "loading of this relationship may not work correctly if this "
                "flag is set explicitly.  omit_join optimization is "
                "automatically detected for conditions under which it is "
                "supported."
            )

        self.omit_join = omit_join
        self.local_remote_pairs = _local_remote_pairs
        self.load_on_pending = load_on_pending
        self.comparator_factory = (
            comparator_factory or RelationshipProperty.Comparator
        )
        util.set_creation_order(self)

        if info is not None:
            self.info.update(info)

        self.strategy_key = (("lazy", self.lazy),)

        self._reverse_property: Set[RelationshipProperty[Any]] = set()

        if overlaps:
            self._overlaps = set(re.split(r"\s*,\s*", overlaps))  # type: ignore  # noqa: E501
        else:
            self._overlaps = ()

        # mypy ignoring the @property setter
        self.cascade = cascade  # type: ignore

        self.back_populates = back_populates

        if self.back_populates:
            if backref:
                raise sa_exc.ArgumentError(
                    "backref and back_populates keyword arguments "
                    "are mutually exclusive"
                )
            self.backref = None
        else:
            self.backref = backref

    def _warn_for_persistence_only_flags(self, **kw: Any) -> None:
        for k, v in kw.items():
            if v != self._persistence_only[k]:
                # we are warning here rather than warn deprecated as this is a
                # configuration mistake, and Python shows regular warnings more
                # aggressively than deprecation warnings by default. Unlike the
                # case of setting viewonly with cascade, the settings being
                # warned about here are not actively doing the wrong thing
                # against viewonly=True, so it is not as urgent to have these
                # raise an error.
                util.warn(
                    "Setting %s on relationship() while also "
                    "setting viewonly=True does not make sense, as a "
                    "viewonly=True relationship does not perform persistence "
                    "operations. This configuration may raise an error "
                    "in a future release." % (k,)
                )

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        attributes.register_descriptor(
            mapper.class_,
            self.key,
            comparator=self.comparator_factory(self, mapper),
            parententity=mapper,
            doc=self.doc,
        )

    class Comparator(util.MemoizedSlots, PropComparator[_PT]):
        """Produce boolean, comparison, and other operators for
        :class:`.RelationshipProperty` attributes.

        See the documentation for :class:`.PropComparator` for a brief
        overview of ORM level operator definition.

        .. seealso::

            :class:`.PropComparator`

            :class:`.ColumnProperty.Comparator`

            :class:`.ColumnOperators`

            :ref:`types_operators`

            :attr:`.TypeEngine.comparator_factory`

        """

        __slots__ = (
            "entity",
            "mapper",
            "property",
            "_of_type",
            "_extra_criteria",
        )

        prop: RODescriptorReference[RelationshipProperty[_PT]]
        _of_type: Optional[_EntityType[_PT]]

        def __init__(
            self,
            prop: RelationshipProperty[_PT],
            parentmapper: _InternalEntityType[Any],
            adapt_to_entity: Optional[AliasedInsp[Any]] = None,
            of_type: Optional[_EntityType[_PT]] = None,
            extra_criteria: Tuple[ColumnElement[bool], ...] = (),
        ):
            """Construction of :class:`.RelationshipProperty.Comparator`
            is internal to the ORM's attribute mechanics.

            """
            self.prop = prop
            self._parententity = parentmapper
            self._adapt_to_entity = adapt_to_entity
            if of_type:
                self._of_type = of_type
            else:
                self._of_type = None
            self._extra_criteria = extra_criteria

        def adapt_to_entity(
            self, adapt_to_entity: AliasedInsp[Any]
        ) -> RelationshipProperty.Comparator[Any]:
            return self.__class__(
                self.prop,
                self._parententity,
                adapt_to_entity=adapt_to_entity,
                of_type=self._of_type,
            )

        entity: _InternalEntityType[_PT]
        """The target entity referred to by this
        :class:`.RelationshipProperty.Comparator`.

        This is either a :class:`_orm.Mapper` or :class:`.AliasedInsp`
        object.

        This is the "target" or "remote" side of the
        :func:`_orm.relationship`.

        """

        mapper: Mapper[_PT]
        """The target :class:`_orm.Mapper` referred to by this
        :class:`.RelationshipProperty.Comparator`.

        This is the "target" or "remote" side of the
        :func:`_orm.relationship`.

        """

        def _memoized_attr_entity(self) -> _InternalEntityType[_PT]:
            if self._of_type:
                return inspect(self._of_type)  # type: ignore
            else:
                return self.prop.entity

        def _memoized_attr_mapper(self) -> Mapper[_PT]:
            return self.entity.mapper

        def _source_selectable(self) -> FromClause:
            if self._adapt_to_entity:
                return self._adapt_to_entity.selectable
            else:
                return self.property.parent._with_polymorphic_selectable

        def __clause_element__(self) -> ColumnElement[bool]:
            adapt_from = self._source_selectable()
            if self._of_type:
                of_type_entity = inspect(self._of_type)
            else:
                of_type_entity = None

            (
                pj,
                sj,
                source,
                dest,
                secondary,
                target_adapter,
            ) = self.prop._create_joins(
                source_selectable=adapt_from,
                source_polymorphic=True,
                of_type_entity=of_type_entity,
                alias_secondary=True,
                extra_criteria=self._extra_criteria,
            )
            if sj is not None:
                return pj & sj
            else:
                return pj

        def of_type(self, class_: _EntityType[Any]) -> PropComparator[_PT]:
            r"""Redefine this object in terms of a polymorphic subclass.

            See :meth:`.PropComparator.of_type` for an example.


            """
            return RelationshipProperty.Comparator(
                self.prop,
                self._parententity,
                adapt_to_entity=self._adapt_to_entity,
                of_type=class_,
                extra_criteria=self._extra_criteria,
            )

        def and_(
            self, *criteria: _ColumnExpressionArgument[bool]
        ) -> PropComparator[Any]:
            """Add AND criteria.

            See :meth:`.PropComparator.and_` for an example.

            .. versionadded:: 1.4

            """
            exprs = tuple(
                coercions.expect(roles.WhereHavingRole, clause)
                for clause in util.coerce_generator_arg(criteria)
            )

            return RelationshipProperty.Comparator(
                self.prop,
                self._parententity,
                adapt_to_entity=self._adapt_to_entity,
                of_type=self._of_type,
                extra_criteria=self._extra_criteria + exprs,
            )

        def in_(self, other: Any) -> NoReturn:
            """Produce an IN clause - this is not implemented
            for :func:`_orm.relationship`-based attributes at this time.

            """
            raise NotImplementedError(
                "in_() not yet supported for "
                "relationships.  For a simple "
                "many-to-one, use in_() against "
                "the set of foreign key values."
            )

        # https://github.com/python/mypy/issues/4266
        __hash__ = None  # type: ignore

        def __eq__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
            """Implement the ``==`` operator.

            In a many-to-one context, such as:

            .. sourcecode:: text

              MyClass.some_prop == <some object>

            this will typically produce a
            clause such as:

            .. sourcecode:: text

              mytable.related_id == <some id>

            Where ``<some id>`` is the primary key of the given
            object.

            The ``==`` operator provides partial functionality for non-
            many-to-one comparisons:

            * Comparisons against collections are not supported.
              Use :meth:`~.Relationship.Comparator.contains`.
            * Compared to a scalar one-to-many, will produce a
              clause that compares the target columns in the parent to
              the given target.
            * Compared to a scalar many-to-many, an alias
              of the association table will be rendered as
              well, forming a natural join that is part of the
              main body of the query. This will not work for
              queries that go beyond simple AND conjunctions of
              comparisons, such as those which use OR. Use
              explicit joins, outerjoins, or
              :meth:`~.Relationship.Comparator.has` for
              more comprehensive non-many-to-one scalar
              membership tests.
            * Comparisons against ``None`` given in a one-to-many
              or many-to-many context produce a NOT EXISTS clause.

            """
            if other is None or isinstance(other, expression.Null):
                if self.property.direction in [ONETOMANY, MANYTOMANY]:
                    return ~self._criterion_exists()
                else:
                    return _orm_annotate(
                        self.property._optimized_compare(
                            None, adapt_source=self.adapter
                        )
                    )
            elif self.property.uselist:
                raise sa_exc.InvalidRequestError(
                    "Can't compare a collection to an object or collection; "
                    "use contains() to test for membership."
                )
            else:
                return _orm_annotate(
                    self.property._optimized_compare(
                        other, adapt_source=self.adapter
                    )
                )

        def _criterion_exists(
            self,
            criterion: Optional[_ColumnExpressionArgument[bool]] = None,
            **kwargs: Any,
        ) -> Exists:
            where_criteria = (
                coercions.expect(roles.WhereHavingRole, criterion)
                if criterion is not None
                else None
            )

            if getattr(self, "_of_type", None):
                info: Optional[_InternalEntityType[Any]] = inspect(
                    self._of_type
                )
                assert info is not None
                target_mapper, to_selectable, is_aliased_class = (
                    info.mapper,
                    info.selectable,
                    info.is_aliased_class,
                )
                if self.property._is_self_referential and not is_aliased_class:
                    to_selectable = to_selectable._anonymous_fromclause()

                single_crit = target_mapper._single_table_criterion
                if single_crit is not None:
                    if where_criteria is not None:
                        where_criteria = single_crit & where_criteria
                    else:
                        where_criteria = single_crit
            else:
                is_aliased_class = False
                to_selectable = None

            if self.adapter:
                source_selectable = self._source_selectable()
            else:
                source_selectable = None

            (
                pj,
                sj,
                source,
                dest,
                secondary,
                target_adapter,
            ) = self.property._create_joins(
                dest_selectable=to_selectable,
                source_selectable=source_selectable,
            )

            for k in kwargs:
                crit = getattr(self.property.mapper.class_, k) == kwargs[k]
                if where_criteria is None:
                    where_criteria = crit
                else:
                    where_criteria = where_criteria & crit

            # annotate the *local* side of the join condition, in the case
            # of pj + sj this is the full primaryjoin, in the case of just
            # pj its the local side of the primaryjoin.
            if sj is not None:
                j = _orm_annotate(pj) & sj
            else:
                j = _orm_annotate(pj, exclude=self.property.remote_side)

            if (
                where_criteria is not None
                and target_adapter
                and not is_aliased_class
            ):
                # limit this adapter to annotated only?
                where_criteria = target_adapter.traverse(where_criteria)

            # only have the "joined left side" of what we
            # return be subject to Query adaption.  The right
            # side of it is used for an exists() subquery and
            # should not correlate or otherwise reach out
            # to anything in the enclosing query.
            if where_criteria is not None:
                where_criteria = where_criteria._annotate(
                    {"no_replacement_traverse": True}
                )

            crit = j & sql.True_._ifnone(where_criteria)

            if secondary is not None:
                ex = (
                    sql.exists(1)
                    .where(crit)
                    .select_from(dest, secondary)
                    .correlate_except(dest, secondary)
                )
            else:
                ex = (
                    sql.exists(1)
                    .where(crit)
                    .select_from(dest)
                    .correlate_except(dest)
                )
            return ex

        def any(
            self,
            criterion: Optional[_ColumnExpressionArgument[bool]] = None,
            **kwargs: Any,
        ) -> ColumnElement[bool]:
            """Produce an expression that tests a collection against
            particular criterion, using EXISTS.

            An expression like::

                session.query(MyClass).filter(
                    MyClass.somereference.any(SomeRelated.x == 2)
                )

            Will produce a query like:

            .. sourcecode:: sql

                SELECT * FROM my_table WHERE
                EXISTS (SELECT 1 FROM related WHERE related.my_id=my_table.id
                AND related.x=2)

            Because :meth:`~.Relationship.Comparator.any` uses
            a correlated subquery, its performance is not nearly as
            good when compared against large target tables as that of
            using a join.

            :meth:`~.Relationship.Comparator.any` is particularly
            useful for testing for empty collections::

                session.query(MyClass).filter(~MyClass.somereference.any())

            will produce:

            .. sourcecode:: sql

                SELECT * FROM my_table WHERE
                NOT (EXISTS (SELECT 1 FROM related WHERE
                related.my_id=my_table.id))

            :meth:`~.Relationship.Comparator.any` is only
            valid for collections, i.e. a :func:`_orm.relationship`
            that has ``uselist=True``.  For scalar references,
            use :meth:`~.Relationship.Comparator.has`.

            """
            if not self.property.uselist:
                raise sa_exc.InvalidRequestError(
                    "'any()' not implemented for scalar "
                    "attributes. Use has()."
                )

            return self._criterion_exists(criterion, **kwargs)

        def has(
            self,
            criterion: Optional[_ColumnExpressionArgument[bool]] = None,
            **kwargs: Any,
        ) -> ColumnElement[bool]:
            """Produce an expression that tests a scalar reference against
            particular criterion, using EXISTS.

            An expression like::

                session.query(MyClass).filter(
                    MyClass.somereference.has(SomeRelated.x == 2)
                )

            Will produce a query like:

            .. sourcecode:: sql

                SELECT * FROM my_table WHERE
                EXISTS (SELECT 1 FROM related WHERE
                related.id==my_table.related_id AND related.x=2)

            Because :meth:`~.Relationship.Comparator.has` uses
            a correlated subquery, its performance is not nearly as
            good when compared against large target tables as that of
            using a join.

            :meth:`~.Relationship.Comparator.has` is only
            valid for scalar references, i.e. a :func:`_orm.relationship`
            that has ``uselist=False``.  For collection references,
            use :meth:`~.Relationship.Comparator.any`.

            """
            if self.property.uselist:
                raise sa_exc.InvalidRequestError(
                    "'has()' not implemented for collections. Use any()."
                )
            return self._criterion_exists(criterion, **kwargs)

        def contains(
            self, other: _ColumnExpressionArgument[Any], **kwargs: Any
        ) -> ColumnElement[bool]:
            """Return a simple expression that tests a collection for
            containment of a particular item.

            :meth:`~.Relationship.Comparator.contains` is
            only valid for a collection, i.e. a
            :func:`_orm.relationship` that implements
            one-to-many or many-to-many with ``uselist=True``.

            When used in a simple one-to-many context, an
            expression like::

                MyClass.contains(other)

            Produces a clause like:

            .. sourcecode:: sql

                mytable.id == <some id>

            Where ``<some id>`` is the value of the foreign key
            attribute on ``other`` which refers to the primary
            key of its parent object. From this it follows that
            :meth:`~.Relationship.Comparator.contains` is
            very useful when used with simple one-to-many
            operations.

            For many-to-many operations, the behavior of
            :meth:`~.Relationship.Comparator.contains`
            has more caveats. The association table will be
            rendered in the statement, producing an "implicit"
            join, that is, includes multiple tables in the FROM
            clause which are equated in the WHERE clause::

                query(MyClass).filter(MyClass.contains(other))

            Produces a query like:

            .. sourcecode:: sql

                SELECT * FROM my_table, my_association_table AS
                my_association_table_1 WHERE
                my_table.id = my_association_table_1.parent_id
                AND my_association_table_1.child_id = <some id>

            Where ``<some id>`` would be the primary key of
            ``other``. From the above, it is clear that
            :meth:`~.Relationship.Comparator.contains`
            will **not** work with many-to-many collections when
            used in queries that move beyond simple AND
            conjunctions, such as multiple
            :meth:`~.Relationship.Comparator.contains`
            expressions joined by OR. In such cases subqueries or
            explicit "outer joins" will need to be used instead.
            See :meth:`~.Relationship.Comparator.any` for
            a less-performant alternative using EXISTS, or refer
            to :meth:`_query.Query.outerjoin`
            as well as :ref:`orm_queryguide_joins`
            for more details on constructing outer joins.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            if not self.prop.uselist:
                raise sa_exc.InvalidRequestError(
                    "'contains' not implemented for scalar "
                    "attributes.  Use =="
                )

            clause = self.prop._optimized_compare(
                other, adapt_source=self.adapter
            )

            if self.prop.secondaryjoin is not None:
                clause.negation_clause = self.__negated_contains_or_equals(
                    other
                )

            return clause

        def __negated_contains_or_equals(
            self, other: Any
        ) -> ColumnElement[bool]:
            if self.prop.direction == MANYTOONE:
                state = attributes.instance_state(other)

                def state_bindparam(
                    local_col: ColumnElement[Any],
                    state: InstanceState[Any],
                    remote_col: ColumnElement[Any],
                ) -> BindParameter[Any]:
                    dict_ = state.dict
                    return sql.bindparam(
                        local_col.key,
                        type_=local_col.type,
                        unique=True,
                        callable_=self.prop._get_attr_w_warn_on_none(
                            self.prop.mapper, state, dict_, remote_col
                        ),
                    )

                def adapt(col: _CE) -> _CE:
                    if self.adapter:
                        return self.adapter(col)
                    else:
                        return col

                if self.property._use_get:
                    return sql.and_(
                        *[
                            sql.or_(
                                adapt(x)
                                != state_bindparam(adapt(x), state, y),
                                adapt(x) == None,
                            )
                            for (x, y) in self.property.local_remote_pairs
                        ]
                    )

            criterion = sql.and_(
                *[
                    x == y
                    for (x, y) in zip(
                        self.property.mapper.primary_key,
                        self.property.mapper.primary_key_from_instance(other),
                    )
                ]
            )

            return ~self._criterion_exists(criterion)

        def __ne__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
            """Implement the ``!=`` operator.

            In a many-to-one context, such as:

            .. sourcecode:: text

              MyClass.some_prop != <some object>

            This will typically produce a clause such as:

            .. sourcecode:: sql

              mytable.related_id != <some id>

            Where ``<some id>`` is the primary key of the
            given object.

            The ``!=`` operator provides partial functionality for non-
            many-to-one comparisons:

            * Comparisons against collections are not supported.
              Use
              :meth:`~.Relationship.Comparator.contains`
              in conjunction with :func:`_expression.not_`.
            * Compared to a scalar one-to-many, will produce a
              clause that compares the target columns in the parent to
              the given target.
            * Compared to a scalar many-to-many, an alias
              of the association table will be rendered as
              well, forming a natural join that is part of the
              main body of the query. This will not work for
              queries that go beyond simple AND conjunctions of
              comparisons, such as those which use OR. Use
              explicit joins, outerjoins, or
              :meth:`~.Relationship.Comparator.has` in
              conjunction with :func:`_expression.not_` for
              more comprehensive non-many-to-one scalar
              membership tests.
            * Comparisons against ``None`` given in a one-to-many
              or many-to-many context produce an EXISTS clause.

            """
            if other is None or isinstance(other, expression.Null):
                if self.property.direction == MANYTOONE:
                    return _orm_annotate(
                        ~self.property._optimized_compare(
                            None, adapt_source=self.adapter
                        )
                    )

                else:
                    return self._criterion_exists()
            elif self.property.uselist:
                raise sa_exc.InvalidRequestError(
                    "Can't compare a collection"
                    " to an object or collection; use "
                    "contains() to test for membership."
                )
            else:
                return _orm_annotate(self.__negated_contains_or_equals(other))

        def _memoized_attr_property(self) -> RelationshipProperty[_PT]:
            self.prop.parent._check_configure()
            return self.prop

    def _with_parent(
        self,
        instance: object,
        alias_secondary: bool = True,
        from_entity: Optional[_EntityType[Any]] = None,
    ) -> ColumnElement[bool]:
        assert instance is not None
        adapt_source: Optional[_CoreAdapterProto] = None
        if from_entity is not None:
            insp: Optional[_InternalEntityType[Any]] = inspect(from_entity)
            assert insp is not None
            if insp_is_aliased_class(insp):
                adapt_source = insp._adapter.adapt_clause
        return self._optimized_compare(
            instance,
            value_is_parent=True,
            adapt_source=adapt_source,
            alias_secondary=alias_secondary,
        )

    def _optimized_compare(
        self,
        state: Any,
        value_is_parent: bool = False,
        adapt_source: Optional[_CoreAdapterProto] = None,
        alias_secondary: bool = True,
    ) -> ColumnElement[bool]:
        if state is not None:
            try:
                state = inspect(state)
            except sa_exc.NoInspectionAvailable:
                state = None

            if state is None or not getattr(state, "is_instance", False):
                raise sa_exc.ArgumentError(
                    "Mapped instance expected for relationship "
                    "comparison to object.   Classes, queries and other "
                    "SQL elements are not accepted in this context; for "
                    "comparison with a subquery, "
                    "use %s.has(**criteria)." % self
                )
        reverse_direction = not value_is_parent

        if state is None:
            return self._lazy_none_clause(
                reverse_direction, adapt_source=adapt_source
            )

        if not reverse_direction:
            criterion, bind_to_col = (
                self._lazy_strategy._lazywhere,
                self._lazy_strategy._bind_to_col,
            )
        else:
            criterion, bind_to_col = (
                self._lazy_strategy._rev_lazywhere,
                self._lazy_strategy._rev_bind_to_col,
            )

        if reverse_direction:
            mapper = self.mapper
        else:
            mapper = self.parent

        dict_ = attributes.instance_dict(state.obj())

        def visit_bindparam(bindparam: BindParameter[Any]) -> None:
            if bindparam._identifying_key in bind_to_col:
                bindparam.callable = self._get_attr_w_warn_on_none(
                    mapper,
                    state,
                    dict_,
                    bind_to_col[bindparam._identifying_key],
                )

        if self.secondary is not None and alias_secondary:
            criterion = ClauseAdapter(
                self.secondary._anonymous_fromclause()
            ).traverse(criterion)

        criterion = visitors.cloned_traverse(
            criterion, {}, {"bindparam": visit_bindparam}
        )

        if adapt_source:
            criterion = adapt_source(criterion)
        return criterion

    def _get_attr_w_warn_on_none(
        self,
        mapper: Mapper[Any],
        state: InstanceState[Any],
        dict_: _InstanceDict,
        column: ColumnElement[Any],
    ) -> Callable[[], Any]:
        """Create the callable that is used in a many-to-one expression.

        E.g.::

            u1 = s.query(User).get(5)

            expr = Address.user == u1

        Above, the SQL should be "address.user_id = 5". The callable
        returned by this method produces the value "5" based on the identity
        of ``u1``.

        """

        # in this callable, we're trying to thread the needle through
        # a wide variety of scenarios, including:
        #
        # * the object hasn't been flushed yet and there's no value for
        #   the attribute as of yet
        #
        # * the object hasn't been flushed yet but it has a user-defined
        #   value
        #
        # * the object has a value but it's expired and not locally present
        #
        # * the object has a value but it's expired and not locally present,
        #   and the object is also detached
        #
        # * The object hadn't been flushed yet, there was no value, but
        #   later, the object has been expired and detached, and *now*
        #   they're trying to evaluate it
        #
        # * the object had a value, but it was changed to a new value, and
        #   then expired
        #
        # * the object had a value, but it was changed to a new value, and
        #   then expired, then the object was detached
        #
        # * the object has a user-set value, but it's None and we don't do
        #   the comparison correctly for that so warn
        #

        prop = mapper.get_property_by_column(column)

        # by invoking this method, InstanceState will track the last known
        # value for this key each time the attribute is to be expired.
        # this feature was added explicitly for use in this method.
        state._track_last_known_value(prop.key)

        lkv_fixed = state._last_known_values

        def _go() -> Any:
            assert lkv_fixed is not None
            last_known = to_return = lkv_fixed[prop.key]
            existing_is_available = (
                last_known is not LoaderCallableStatus.NO_VALUE
            )

            # we support that the value may have changed.  so here we
            # try to get the most recent value including re-fetching.
            # only if we can't get a value now due to detachment do we return
            # the last known value
            current_value = mapper._get_state_attr_by_column(
                state,
                dict_,
                column,
                passive=(
                    PassiveFlag.PASSIVE_OFF
                    if state.persistent
                    else PassiveFlag.PASSIVE_NO_FETCH ^ PassiveFlag.INIT_OK
                ),
            )

            if current_value is LoaderCallableStatus.NEVER_SET:
                if not existing_is_available:
                    raise sa_exc.InvalidRequestError(
                        "Can't resolve value for column %s on object "
                        "%s; no value has been set for this column"
                        % (column, state_str(state))
                    )
            elif current_value is LoaderCallableStatus.PASSIVE_NO_RESULT:
                if not existing_is_available:
                    raise sa_exc.InvalidRequestError(
                        "Can't resolve value for column %s on object "
                        "%s; the object is detached and the value was "
                        "expired" % (column, state_str(state))
                    )
            else:
                to_return = current_value
            if to_return is None:
                util.warn(
                    "Got None for value of column %s; this is unsupported "
                    "for a relationship comparison and will not "
                    "currently produce an IS comparison "
                    "(but may in a future release)" % column
                )
            return to_return

        return _go

    def _lazy_none_clause(
        self,
        reverse_direction: bool = False,
        adapt_source: Optional[_CoreAdapterProto] = None,
    ) -> ColumnElement[bool]:
        if not reverse_direction:
            criterion, bind_to_col = (
                self._lazy_strategy._lazywhere,
                self._lazy_strategy._bind_to_col,
            )
        else:
            criterion, bind_to_col = (
                self._lazy_strategy._rev_lazywhere,
                self._lazy_strategy._rev_bind_to_col,
            )

        criterion = adapt_criterion_to_null(criterion, bind_to_col)

        if adapt_source:
            criterion = adapt_source(criterion)
        return criterion

    def __str__(self) -> str:
        return str(self.parent.class_.__name__) + "." + self.key

    def merge(
        self,
        session: Session,
        source_state: InstanceState[Any],
        source_dict: _InstanceDict,
        dest_state: InstanceState[Any],
        dest_dict: _InstanceDict,
        load: bool,
        _recursive: Dict[Any, object],
        _resolve_conflict_map: Dict[_IdentityKeyType[Any], object],
    ) -> None:
        if load:
            for r in self._reverse_property:
                if (source_state, r) in _recursive:
                    return

        if "merge" not in self._cascade:
            return

        if self.key not in source_dict:
            return

        if self.uselist:
            impl = source_state.get_impl(self.key)

            assert is_has_collection_adapter(impl)
            instances_iterable = impl.get_collection(source_state, source_dict)

            # if this is a CollectionAttributeImpl, then empty should
            # be False, otherwise "self.key in source_dict" should not be
            # True
            assert not instances_iterable.empty if impl.collection else True

            if load:
                # for a full merge, pre-load the destination collection,
                # so that individual _merge of each item pulls from identity
                # map for those already present.
                # also assumes CollectionAttributeImpl behavior of loading
                # "old" list in any case
                dest_state.get_impl(self.key).get(
                    dest_state, dest_dict, passive=PassiveFlag.PASSIVE_MERGE
                )

            dest_list = []
            for current in instances_iterable:
                current_state = attributes.instance_state(current)
                current_dict = attributes.instance_dict(current)
                _recursive[(current_state, self)] = True
                obj = session._merge(
                    current_state,
                    current_dict,
                    load=load,
                    _recursive=_recursive,
                    _resolve_conflict_map=_resolve_conflict_map,
                )
                if obj is not None:
                    dest_list.append(obj)

            if not load:
                coll = attributes.init_state_collection(
                    dest_state, dest_dict, self.key
                )
                for c in dest_list:
                    coll.append_without_event(c)
            else:
                dest_impl = dest_state.get_impl(self.key)
                assert is_has_collection_adapter(dest_impl)
                dest_impl.set(
                    dest_state,
                    dest_dict,
                    dest_list,
                    _adapt=False,
                    passive=PassiveFlag.PASSIVE_MERGE,
                )
        else:
            current = source_dict[self.key]
            if current is not None:
                current_state = attributes.instance_state(current)
                current_dict = attributes.instance_dict(current)
                _recursive[(current_state, self)] = True
                obj = session._merge(
                    current_state,
                    current_dict,
                    load=load,
                    _recursive=_recursive,
                    _resolve_conflict_map=_resolve_conflict_map,
                )
            else:
                obj = None

            if not load:
                dest_dict[self.key] = obj
            else:
                dest_state.get_impl(self.key).set(
                    dest_state, dest_dict, obj, None
                )

    def _value_as_iterable(
        self,
        state: InstanceState[_O],
        dict_: _InstanceDict,
        key: str,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
    ) -> Sequence[Tuple[InstanceState[_O], _O]]:
        """Return a list of tuples (state, obj) for the given
        key.

        returns an empty list if the value is None/empty/PASSIVE_NO_RESULT
        """

        impl = state.manager[key].impl
        x = impl.get(state, dict_, passive=passive)
        if x is LoaderCallableStatus.PASSIVE_NO_RESULT or x is None:
            return []
        elif is_has_collection_adapter(impl):
            return [
                (attributes.instance_state(o), o)
                for o in impl.get_collection(state, dict_, x, passive=passive)
            ]
        else:
            return [(attributes.instance_state(x), x)]

    def cascade_iterator(
        self,
        type_: str,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        visited_states: Set[InstanceState[Any]],
        halt_on: Optional[Callable[[InstanceState[Any]], bool]] = None,
    ) -> Iterator[Tuple[Any, Mapper[Any], InstanceState[Any], _InstanceDict]]:
        # assert type_ in self._cascade

        # only actively lazy load on the 'delete' cascade
        if type_ != "delete" or self.passive_deletes:
            passive = PassiveFlag.PASSIVE_NO_INITIALIZE
        else:
            passive = PassiveFlag.PASSIVE_OFF | PassiveFlag.NO_RAISE

        if type_ == "save-update":
            tuples = state.manager[self.key].impl.get_all_pending(state, dict_)
        else:
            tuples = self._value_as_iterable(
                state, dict_, self.key, passive=passive
            )

        skip_pending = (
            type_ == "refresh-expire" and "delete-orphan" not in self._cascade
        )

        for instance_state, c in tuples:
            if instance_state in visited_states:
                continue

            if c is None:
                # would like to emit a warning here, but
                # would not be consistent with collection.append(None)
                # current behavior of silently skipping.
                # see [ticket:2229]
                continue

            assert instance_state is not None
            instance_dict = attributes.instance_dict(c)

            if halt_on and halt_on(instance_state):
                continue

            if skip_pending and not instance_state.key:
                continue

            instance_mapper = instance_state.manager.mapper

            if not instance_mapper.isa(self.mapper.class_manager.mapper):
                raise AssertionError(
                    "Attribute '%s' on class '%s' "
                    "doesn't handle objects "
                    "of type '%s'"
                    % (self.key, self.parent.class_, c.__class__)
                )

            visited_states.add(instance_state)

            yield c, instance_mapper, instance_state, instance_dict

    @property
    def _effective_sync_backref(self) -> bool:
        if self.viewonly:
            return False
        else:
            return self.sync_backref is not False

    @staticmethod
    def _check_sync_backref(
        rel_a: RelationshipProperty[Any], rel_b: RelationshipProperty[Any]
    ) -> None:
        if rel_a.viewonly and rel_b.sync_backref:
            raise sa_exc.InvalidRequestError(
                "Relationship %s cannot specify sync_backref=True since %s "
                "includes viewonly=True." % (rel_b, rel_a)
            )
        if (
            rel_a.viewonly
            and not rel_b.viewonly
            and rel_b.sync_backref is not False
        ):
            rel_b.sync_backref = False

    def _add_reverse_property(self, key: str) -> None:
        other = self.mapper.get_property(key, _configure_mappers=False)
        if not isinstance(other, RelationshipProperty):
            raise sa_exc.InvalidRequestError(
                "back_populates on relationship '%s' refers to attribute '%s' "
                "that is not a relationship.  The back_populates parameter "
                "should refer to the name of a relationship on the target "
                "class." % (self, other)
            )
        # viewonly and sync_backref cases
        # 1. self.viewonly==True and other.sync_backref==True -> error
        # 2. self.viewonly==True and other.viewonly==False and
        #    other.sync_backref==None -> warn sync_backref=False, set to False
        self._check_sync_backref(self, other)
        # 3. other.viewonly==True and self.sync_backref==True -> error
        # 4. other.viewonly==True and self.viewonly==False and
        #    self.sync_backref==None -> warn sync_backref=False, set to False
        self._check_sync_backref(other, self)

        self._reverse_property.add(other)
        other._reverse_property.add(self)

        other._setup_entity()

        if not other.mapper.common_parent(self.parent):
            raise sa_exc.ArgumentError(
                "reverse_property %r on "
                "relationship %s references relationship %s, which "
                "does not reference mapper %s"
                % (key, self, other, self.parent)
            )

        if (
            other._configure_started
            and self.direction in (ONETOMANY, MANYTOONE)
            and self.direction == other.direction
        ):
            raise sa_exc.ArgumentError(
                "%s and back-reference %s are "
                "both of the same direction %r.  Did you mean to "
                "set remote_side on the many-to-one side ?"
                % (other, self, self.direction)
            )

    @util.memoized_property
    def entity(self) -> _InternalEntityType[_T]:
        """Return the target mapped entity, which is an inspect() of the
        class or aliased class that is referenced by this
        :class:`.RelationshipProperty`.

        """
        self.parent._check_configure()
        return self.entity

    @util.memoized_property
    def mapper(self) -> Mapper[_T]:
        """Return the targeted :class:`_orm.Mapper` for this
        :class:`.RelationshipProperty`.

        """
        return self.entity.mapper

    def do_init(self) -> None:
        self._check_conflicts()
        self._process_dependent_arguments()
        self._setup_entity()
        self._setup_registry_dependencies()
        self._setup_join_conditions()
        self._check_cascade_settings(self._cascade)
        self._post_init()
        self._generate_backref()
        self._join_condition._warn_for_conflicting_sync_targets()
        super().do_init()
        self._lazy_strategy = cast(
            "LazyLoader", self._get_strategy((("lazy", "select"),))
        )

    def _setup_registry_dependencies(self) -> None:
        self.parent.mapper.registry._set_depends_on(
            self.entity.mapper.registry
        )

    def _process_dependent_arguments(self) -> None:
        """Convert incoming configuration arguments to their
        proper form.

        Callables are resolved, ORM annotations removed.

        """

        # accept callables for other attributes which may require
        # deferred initialization.  This technique is used
        # by declarative "string configs" and some recipes.
        init_args = self._init_args

        for attr in (
            "order_by",
            "primaryjoin",
            "secondaryjoin",
            "secondary",
            "foreign_keys",
            "remote_side",
        ):
            rel_arg = getattr(init_args, attr)

            rel_arg._resolve_against_registry(self._clsregistry_resolvers[1])

        # remove "annotations" which are present if mapped class
        # descriptors are used to create the join expression.
        for attr in "primaryjoin", "secondaryjoin":
            rel_arg = getattr(init_args, attr)
            val = rel_arg.resolved
            if val is not None:
                rel_arg.resolved = _orm_deannotate(
                    coercions.expect(
                        roles.ColumnArgumentRole, val, argname=attr
                    )
                )

        secondary = init_args.secondary.resolved
        if secondary is not None and _is_mapped_class(secondary):
            raise sa_exc.ArgumentError(
                "secondary argument %s passed to to relationship() %s must "
                "be a Table object or other FROM clause; can't send a mapped "
                "class directly as rows in 'secondary' are persisted "
                "independently of a class that is mapped "
                "to that same table." % (secondary, self)
            )

        # ensure expressions in self.order_by, foreign_keys,
        # remote_side are all columns, not strings.
        if (
            init_args.order_by.resolved is not False
            and init_args.order_by.resolved is not None
        ):
            self.order_by = tuple(
                coercions.expect(
                    roles.ColumnArgumentRole, x, argname="order_by"
                )
                for x in util.to_list(init_args.order_by.resolved)
            )
        else:
            self.order_by = False

        self._user_defined_foreign_keys = util.column_set(
            coercions.expect(
                roles.ColumnArgumentRole, x, argname="foreign_keys"
            )
            for x in util.to_column_set(init_args.foreign_keys.resolved)
        )

        self.remote_side = util.column_set(
            coercions.expect(
                roles.ColumnArgumentRole, x, argname="remote_side"
            )
            for x in util.to_column_set(init_args.remote_side.resolved)
        )

    def declarative_scan(
        self,
        decl_scan: _ClassScanMapperConfig,
        registry: _RegistryType,
        cls: Type[Any],
        originating_module: Optional[str],
        key: str,
        mapped_container: Optional[Type[Mapped[Any]]],
        annotation: Optional[_AnnotationScanType],
        extracted_mapped_annotation: Optional[_AnnotationScanType],
        is_dataclass_field: bool,
    ) -> None:
        if extracted_mapped_annotation is None:
            if self.argument is None:
                self._raise_for_required(key, cls)
            else:
                return

        argument = extracted_mapped_annotation
        assert originating_module is not None

        if mapped_container is not None:
            is_write_only = issubclass(mapped_container, WriteOnlyMapped)
            is_dynamic = issubclass(mapped_container, DynamicMapped)
            if is_write_only:
                self.lazy = "write_only"
                self.strategy_key = (("lazy", self.lazy),)
            elif is_dynamic:
                self.lazy = "dynamic"
                self.strategy_key = (("lazy", self.lazy),)
        else:
            is_write_only = is_dynamic = False

        argument = de_optionalize_union_types(argument)

        if hasattr(argument, "__origin__"):
            arg_origin = argument.__origin__
            if isinstance(arg_origin, type) and issubclass(
                arg_origin, abc.Collection
            ):
                if self.collection_class is None:
                    if _py_inspect.isabstract(arg_origin):
                        raise sa_exc.ArgumentError(
                            f"Collection annotation type {arg_origin} cannot "
                            "be instantiated; please provide an explicit "
                            "'collection_class' parameter "
                            "(e.g. list, set, etc.) to the "
                            "relationship() function to accompany this "
                            "annotation"
                        )

                    self.collection_class = arg_origin

            elif not is_write_only and not is_dynamic:
                self.uselist = False

            if argument.__args__:  # type: ignore
                if isinstance(arg_origin, type) and issubclass(
                    arg_origin, typing.Mapping
                ):
                    type_arg = argument.__args__[-1]  # type: ignore
                else:
                    type_arg = argument.__args__[0]  # type: ignore
                if hasattr(type_arg, "__forward_arg__"):
                    str_argument = type_arg.__forward_arg__

                    argument = resolve_name_to_real_class_name(
                        str_argument, originating_module
                    )
                else:
                    argument = type_arg
            else:
                raise sa_exc.ArgumentError(
                    f"Generic alias {argument} requires an argument"
                )
        elif hasattr(argument, "__forward_arg__"):
            argument = argument.__forward_arg__

            argument = resolve_name_to_real_class_name(
                argument, originating_module
            )

        if (
            self.collection_class is None
            and not is_write_only
            and not is_dynamic
        ):
            self.uselist = False

        # ticket #8759
        # if a lead argument was given to relationship(), like
        # `relationship("B")`, use that, don't replace it with class we
        # found in the annotation.  The declarative_scan() method call here is
        # still useful, as we continue to derive collection type and do
        # checking of the annotation in any case.
        if self.argument is None:
            self.argument = cast("_RelationshipArgumentType[_T]", argument)

    @util.preload_module("sqlalchemy.orm.mapper")
    def _setup_entity(self, __argument: Any = None) -> None:
        if "entity" in self.__dict__:
            return

        mapperlib = util.preloaded.orm_mapper

        if __argument:
            argument = __argument
        else:
            argument = self.argument

        resolved_argument: _ExternalEntityType[Any]

        if isinstance(argument, str):
            # we might want to cleanup clsregistry API to make this
            # more straightforward
            resolved_argument = cast(
                "_ExternalEntityType[Any]",
                self._clsregistry_resolve_name(argument)(),
            )
        elif callable(argument) and not isinstance(
            argument, (type, mapperlib.Mapper)
        ):
            resolved_argument = argument()
        else:
            resolved_argument = argument

        entity: _InternalEntityType[Any]

        if isinstance(resolved_argument, type):
            entity = class_mapper(resolved_argument, configure=False)
        else:
            try:
                entity = inspect(resolved_argument)
            except sa_exc.NoInspectionAvailable:
                entity = None  # type: ignore

            if not hasattr(entity, "mapper"):
                raise sa_exc.ArgumentError(
                    "relationship '%s' expects "
                    "a class or a mapper argument (received: %s)"
                    % (self.key, type(resolved_argument))
                )

        self.entity = entity
        self.target = self.entity.persist_selectable

    def _setup_join_conditions(self) -> None:
        self._join_condition = jc = JoinCondition(
            parent_persist_selectable=self.parent.persist_selectable,
            child_persist_selectable=self.entity.persist_selectable,
            parent_local_selectable=self.parent.local_table,
            child_local_selectable=self.entity.local_table,
            primaryjoin=self._init_args.primaryjoin.resolved,
            secondary=self._init_args.secondary.resolved,
            secondaryjoin=self._init_args.secondaryjoin.resolved,
            parent_equivalents=self.parent._equivalent_columns,
            child_equivalents=self.mapper._equivalent_columns,
            consider_as_foreign_keys=self._user_defined_foreign_keys,
            local_remote_pairs=self.local_remote_pairs,
            remote_side=self.remote_side,
            self_referential=self._is_self_referential,
            prop=self,
            support_sync=not self.viewonly,
            can_be_synced_fn=self._columns_are_mapped,
        )
        self.primaryjoin = jc.primaryjoin
        self.secondaryjoin = jc.secondaryjoin
        self.secondary = jc.secondary
        self.direction = jc.direction
        self.local_remote_pairs = jc.local_remote_pairs
        self.remote_side = jc.remote_columns
        self.local_columns = jc.local_columns
        self.synchronize_pairs = jc.synchronize_pairs
        self._calculated_foreign_keys = jc.foreign_key_columns
        self.secondary_synchronize_pairs = jc.secondary_synchronize_pairs

    @property
    def _clsregistry_resolve_arg(
        self,
    ) -> Callable[[str, bool], _class_resolver]:
        return self._clsregistry_resolvers[1]

    @property
    def _clsregistry_resolve_name(
        self,
    ) -> Callable[[str], Callable[[], Union[Type[Any], Table, _ModNS]]]:
        return self._clsregistry_resolvers[0]

    @util.memoized_property
    @util.preload_module("sqlalchemy.orm.clsregistry")
    def _clsregistry_resolvers(
        self,
    ) -> Tuple[
        Callable[[str], Callable[[], Union[Type[Any], Table, _ModNS]]],
        Callable[[str, bool], _class_resolver],
    ]:
        _resolver = util.preloaded.orm_clsregistry._resolver

        return _resolver(self.parent.class_, self)

    def _check_conflicts(self) -> None:
        """Test that this relationship is legal, warn about
        inheritance conflicts."""
        if self.parent.non_primary and not class_mapper(
            self.parent.class_, configure=False
        ).has_property(self.key):
            raise sa_exc.ArgumentError(
                "Attempting to assign a new "
                "relationship '%s' to a non-primary mapper on "
                "class '%s'.  New relationships can only be added "
                "to the primary mapper, i.e. the very first mapper "
                "created for class '%s' "
                % (
                    self.key,
                    self.parent.class_.__name__,
                    self.parent.class_.__name__,
                )
            )

    @property
    def cascade(self) -> CascadeOptions:
        """Return the current cascade setting for this
        :class:`.RelationshipProperty`.
        """
        return self._cascade

    @cascade.setter
    def cascade(self, cascade: Union[str, CascadeOptions]) -> None:
        self._set_cascade(cascade)

    def _set_cascade(self, cascade_arg: Union[str, CascadeOptions]) -> None:
        cascade = CascadeOptions(cascade_arg)

        if self.viewonly:
            cascade = CascadeOptions(
                cascade.intersection(CascadeOptions._viewonly_cascades)
            )

        if "mapper" in self.__dict__:
            self._check_cascade_settings(cascade)
        self._cascade = cascade

        if self._dependency_processor:
            self._dependency_processor.cascade = cascade

    def _check_cascade_settings(self, cascade: CascadeOptions) -> None:
        if (
            cascade.delete_orphan
            and not self.single_parent
            and (self.direction is MANYTOMANY or self.direction is MANYTOONE)
        ):
            raise sa_exc.ArgumentError(
                "For %(direction)s relationship %(rel)s, delete-orphan "
                "cascade is normally "
                'configured only on the "one" side of a one-to-many '
                "relationship, "
                'and not on the "many" side of a many-to-one or many-to-many '
                "relationship.  "
                "To force this relationship to allow a particular "
                '"%(relatedcls)s" object to be referenced by only '
                'a single "%(clsname)s" object at a time via the '
                "%(rel)s relationship, which "
                "would allow "
                "delete-orphan cascade to take place in this direction, set "
                "the single_parent=True flag."
                % {
                    "rel": self,
                    "direction": (
                        "many-to-one"
                        if self.direction is MANYTOONE
                        else "many-to-many"
                    ),
                    "clsname": self.parent.class_.__name__,
                    "relatedcls": self.mapper.class_.__name__,
                },
                code="bbf0",
            )

        if self.passive_deletes == "all" and (
            "delete" in cascade or "delete-orphan" in cascade
        ):
            raise sa_exc.ArgumentError(
                "On %s, can't set passive_deletes='all' in conjunction "
                "with 'delete' or 'delete-orphan' cascade" % self
            )

        if cascade.delete_orphan:
            self.mapper.primary_mapper()._delete_orphans.append(
                (self.key, self.parent.class_)
            )

    def _persists_for(self, mapper: Mapper[Any]) -> bool:
        """Return True if this property will persist values on behalf
        of the given mapper.

        """

        return (
            self.key in mapper.relationships
            and mapper.relationships[self.key] is self
        )

    def _columns_are_mapped(self, *cols: ColumnElement[Any]) -> bool:
        """Return True if all columns in the given collection are
        mapped by the tables referenced by this :class:`.RelationshipProperty`.

        """

        secondary = self._init_args.secondary.resolved
        for c in cols:
            if secondary is not None and secondary.c.contains_column(c):
                continue
            if not self.parent.persist_selectable.c.contains_column(
                c
            ) and not self.target.c.contains_column(c):
                return False
        return True

    def _generate_backref(self) -> None:
        """Interpret the 'backref' instruction to create a
        :func:`_orm.relationship` complementary to this one."""

        if self.parent.non_primary:
            return
        if self.backref is not None and not self.back_populates:
            kwargs: Dict[str, Any]
            if isinstance(self.backref, str):
                backref_key, kwargs = self.backref, {}
            else:
                backref_key, kwargs = self.backref
            mapper = self.mapper.primary_mapper()

            if not mapper.concrete:
                check = set(mapper.iterate_to_root()).union(
                    mapper.self_and_descendants
                )
                for m in check:
                    if m.has_property(backref_key) and not m.concrete:
                        raise sa_exc.ArgumentError(
                            "Error creating backref "
                            "'%s' on relationship '%s': property of that "
                            "name exists on mapper '%s'"
                            % (backref_key, self, m)
                        )

            # determine primaryjoin/secondaryjoin for the
            # backref.  Use the one we had, so that
            # a custom join doesn't have to be specified in
            # both directions.
            if self.secondary is not None:
                # for many to many, just switch primaryjoin/
                # secondaryjoin.   use the annotated
                # pj/sj on the _join_condition.
                pj = kwargs.pop(
                    "primaryjoin",
                    self._join_condition.secondaryjoin_minus_local,
                )
                sj = kwargs.pop(
                    "secondaryjoin",
                    self._join_condition.primaryjoin_minus_local,
                )
            else:
                pj = kwargs.pop(
                    "primaryjoin",
                    self._join_condition.primaryjoin_reverse_remote,
                )
                sj = kwargs.pop("secondaryjoin", None)
                if sj:
                    raise sa_exc.InvalidRequestError(
                        "Can't assign 'secondaryjoin' on a backref "
                        "against a non-secondary relationship."
                    )

            foreign_keys = kwargs.pop(
                "foreign_keys", self._user_defined_foreign_keys
            )
            parent = self.parent.primary_mapper()
            kwargs.setdefault("viewonly", self.viewonly)
            kwargs.setdefault("post_update", self.post_update)
            kwargs.setdefault("passive_updates", self.passive_updates)
            kwargs.setdefault("sync_backref", self.sync_backref)
            self.back_populates = backref_key
            relationship = RelationshipProperty(
                parent,
                self.secondary,
                primaryjoin=pj,
                secondaryjoin=sj,
                foreign_keys=foreign_keys,
                back_populates=self.key,
                **kwargs,
            )
            mapper._configure_property(
                backref_key, relationship, warn_for_existing=True
            )

        if self.back_populates:
            self._add_reverse_property(self.back_populates)

    @util.preload_module("sqlalchemy.orm.dependency")
    def _post_init(self) -> None:
        dependency = util.preloaded.orm_dependency

        if self.uselist is None:
            self.uselist = self.direction is not MANYTOONE
        if not self.viewonly:
            self._dependency_processor = (  # type: ignore
                dependency.DependencyProcessor.from_relationship
            )(self)

    @util.memoized_property
    def _use_get(self) -> bool:
        """memoize the 'use_get' attribute of this RelationshipLoader's
        lazyloader."""

        strategy = self._lazy_strategy
        return strategy.use_get

    @util.memoized_property
    def _is_self_referential(self) -> bool:
        return self.mapper.common_parent(self.parent)

    def _create_joins(
        self,
        source_polymorphic: bool = False,
        source_selectable: Optional[FromClause] = None,
        dest_selectable: Optional[FromClause] = None,
        of_type_entity: Optional[_InternalEntityType[Any]] = None,
        alias_secondary: bool = False,
        extra_criteria: Tuple[ColumnElement[bool], ...] = (),
    ) -> Tuple[
        ColumnElement[bool],
        Optional[ColumnElement[bool]],
        FromClause,
        FromClause,
        Optional[FromClause],
        Optional[ClauseAdapter],
    ]:
        aliased = False

        if alias_secondary and self.secondary is not None:
            aliased = True

        if source_selectable is None:
            if source_polymorphic and self.parent.with_polymorphic:
                source_selectable = self.parent._with_polymorphic_selectable

        if of_type_entity:
            dest_mapper = of_type_entity.mapper
            if dest_selectable is None:
                dest_selectable = of_type_entity.selectable
                aliased = True
        else:
            dest_mapper = self.mapper

        if dest_selectable is None:
            dest_selectable = self.entity.selectable
            if self.mapper.with_polymorphic:
                aliased = True

            if self._is_self_referential and source_selectable is None:
                dest_selectable = dest_selectable._anonymous_fromclause()
                aliased = True
        elif (
            dest_selectable is not self.mapper._with_polymorphic_selectable
            or self.mapper.with_polymorphic
        ):
            aliased = True

        single_crit = dest_mapper._single_table_criterion
        aliased = aliased or (
            source_selectable is not None
            and (
                source_selectable
                is not self.parent._with_polymorphic_selectable
                or source_selectable._is_subquery
            )
        )

        (
            primaryjoin,
            secondaryjoin,
            secondary,
            target_adapter,
            dest_selectable,
        ) = self._join_condition.join_targets(
            source_selectable,
            dest_selectable,
            aliased,
            single_crit,
            extra_criteria,
        )
        if source_selectable is None:
            source_selectable = self.parent.local_table
        if dest_selectable is None:
            dest_selectable = self.entity.local_table
        return (
            primaryjoin,
            secondaryjoin,
            source_selectable,
            dest_selectable,
            secondary,
            target_adapter,
        )


def _annotate_columns(element: _CE, annotations: _AnnotationDict) -> _CE:
    def clone(elem: _CE) -> _CE:
        if isinstance(elem, expression.ColumnClause):
            elem = elem._annotate(annotations.copy())  # type: ignore
        elem._copy_internals(clone=clone)
        return elem

    if element is not None:
        element = clone(element)
    clone = None  # type: ignore # remove gc cycles
    return element


class JoinCondition:
    primaryjoin_initial: Optional[ColumnElement[bool]]
    primaryjoin: ColumnElement[bool]
    secondaryjoin: Optional[ColumnElement[bool]]
    secondary: Optional[FromClause]
    prop: RelationshipProperty[Any]

    synchronize_pairs: _ColumnPairs
    secondary_synchronize_pairs: _ColumnPairs
    direction: RelationshipDirection

    parent_persist_selectable: FromClause
    child_persist_selectable: FromClause
    parent_local_selectable: FromClause
    child_local_selectable: FromClause

    _local_remote_pairs: Optional[_ColumnPairs]

    def __init__(
        self,
        parent_persist_selectable: FromClause,
        child_persist_selectable: FromClause,
        parent_local_selectable: FromClause,
        child_local_selectable: FromClause,
        *,
        primaryjoin: Optional[ColumnElement[bool]] = None,
        secondary: Optional[FromClause] = None,
        secondaryjoin: Optional[ColumnElement[bool]] = None,
        parent_equivalents: Optional[_EquivalentColumnMap] = None,
        child_equivalents: Optional[_EquivalentColumnMap] = None,
        consider_as_foreign_keys: Any = None,
        local_remote_pairs: Optional[_ColumnPairs] = None,
        remote_side: Any = None,
        self_referential: Any = False,
        prop: RelationshipProperty[Any],
        support_sync: bool = True,
        can_be_synced_fn: Callable[..., bool] = lambda *c: True,
    ):
        self.parent_persist_selectable = parent_persist_selectable
        self.parent_local_selectable = parent_local_selectable
        self.child_persist_selectable = child_persist_selectable
        self.child_local_selectable = child_local_selectable
        self.parent_equivalents = parent_equivalents
        self.child_equivalents = child_equivalents
        self.primaryjoin_initial = primaryjoin
        self.secondaryjoin = secondaryjoin
        self.secondary = secondary
        self.consider_as_foreign_keys = consider_as_foreign_keys
        self._local_remote_pairs = local_remote_pairs
        self._remote_side = remote_side
        self.prop = prop
        self.self_referential = self_referential
        self.support_sync = support_sync
        self.can_be_synced_fn = can_be_synced_fn

        self._determine_joins()
        assert self.primaryjoin is not None

        self._sanitize_joins()
        self._annotate_fks()
        self._annotate_remote()
        self._annotate_local()
        self._annotate_parentmapper()
        self._setup_pairs()
        self._check_foreign_cols(self.primaryjoin, True)
        if self.secondaryjoin is not None:
            self._check_foreign_cols(self.secondaryjoin, False)
        self._determine_direction()
        self._check_remote_side()
        self._log_joins()

    def _log_joins(self) -> None:
        log = self.prop.logger
        log.info("%s setup primary join %s", self.prop, self.primaryjoin)
        log.info("%s setup secondary join %s", self.prop, self.secondaryjoin)
        log.info(
            "%s synchronize pairs [%s]",
            self.prop,
            ",".join(
                "(%s => %s)" % (l, r) for (l, r) in self.synchronize_pairs
            ),
        )
        log.info(
            "%s secondary synchronize pairs [%s]",
            self.prop,
            ",".join(
                "(%s => %s)" % (l, r)
                for (l, r) in self.secondary_synchronize_pairs or []
            ),
        )
        log.info(
            "%s local/remote pairs [%s]",
            self.prop,
            ",".join(
                "(%s / %s)" % (l, r) for (l, r) in self.local_remote_pairs
            ),
        )
        log.info(
            "%s remote columns [%s]",
            self.prop,
            ",".join("%s" % col for col in self.remote_columns),
        )
        log.info(
            "%s local columns [%s]",
            self.prop,
            ",".join("%s" % col for col in self.local_columns),
        )
        log.info("%s relationship direction %s", self.prop, self.direction)

    def _sanitize_joins(self) -> None:
        """remove the parententity annotation from our join conditions which
        can leak in here based on some declarative patterns and maybe others.

        "parentmapper" is relied upon both by the ORM evaluator as well as
        the use case in _join_fixture_inh_selfref_w_entity
        that relies upon it being present, see :ticket:`3364`.

        """

        self.primaryjoin = _deep_deannotate(
            self.primaryjoin, values=("parententity", "proxy_key")
        )
        if self.secondaryjoin is not None:
            self.secondaryjoin = _deep_deannotate(
                self.secondaryjoin, values=("parententity", "proxy_key")
            )

    def _determine_joins(self) -> None:
        """Determine the 'primaryjoin' and 'secondaryjoin' attributes,
        if not passed to the constructor already.

        This is based on analysis of the foreign key relationships
        between the parent and target mapped selectables.

        """
        if self.secondaryjoin is not None and self.secondary is None:
            raise sa_exc.ArgumentError(
                "Property %s specified with secondary "
                "join condition but "
                "no secondary argument" % self.prop
            )

        # find a join between the given mapper's mapped table and
        # the given table. will try the mapper's local table first
        # for more specificity, then if not found will try the more
        # general mapped table, which in the case of inheritance is
        # a join.
        try:
            consider_as_foreign_keys = self.consider_as_foreign_keys or None
            if self.secondary is not None:
                if self.secondaryjoin is None:
                    self.secondaryjoin = join_condition(
                        self.child_persist_selectable,
                        self.secondary,
                        a_subset=self.child_local_selectable,
                        consider_as_foreign_keys=consider_as_foreign_keys,
                    )
                if self.primaryjoin_initial is None:
                    self.primaryjoin = join_condition(
                        self.parent_persist_selectable,
                        self.secondary,
                        a_subset=self.parent_local_selectable,
                        consider_as_foreign_keys=consider_as_foreign_keys,
                    )
                else:
                    self.primaryjoin = self.primaryjoin_initial
            else:
                if self.primaryjoin_initial is None:
                    self.primaryjoin = join_condition(
                        self.parent_persist_selectable,
                        self.child_persist_selectable,
                        a_subset=self.parent_local_selectable,
                        consider_as_foreign_keys=consider_as_foreign_keys,
                    )
                else:
                    self.primaryjoin = self.primaryjoin_initial
        except sa_exc.NoForeignKeysError as nfe:
            if self.secondary is not None:
                raise sa_exc.NoForeignKeysError(
                    "Could not determine join "
                    "condition between parent/child tables on "
                    "relationship %s - there are no foreign keys "
                    "linking these tables via secondary table '%s'.  "
                    "Ensure that referencing columns are associated "
                    "with a ForeignKey or ForeignKeyConstraint, or "
                    "specify 'primaryjoin' and 'secondaryjoin' "
                    "expressions." % (self.prop, self.secondary)
                ) from nfe
            else:
                raise sa_exc.NoForeignKeysError(
                    "Could not determine join "
                    "condition between parent/child tables on "
                    "relationship %s - there are no foreign keys "
                    "linking these tables.  "
                    "Ensure that referencing columns are associated "
                    "with a ForeignKey or ForeignKeyConstraint, or "
                    "specify a 'primaryjoin' expression." % self.prop
                ) from nfe
        except sa_exc.AmbiguousForeignKeysError as afe:
            if self.secondary is not None:
                raise sa_exc.AmbiguousForeignKeysError(
                    "Could not determine join "
                    "condition between parent/child tables on "
                    "relationship %s - there are multiple foreign key "
                    "paths linking the tables via secondary table '%s'.  "
                    "Specify the 'foreign_keys' "
                    "argument, providing a list of those columns which "
                    "should be counted as containing a foreign key "
                    "reference from the secondary table to each of the "
                    "parent and child tables." % (self.prop, self.secondary)
                ) from afe
            else:
                raise sa_exc.AmbiguousForeignKeysError(
                    "Could not determine join "
                    "condition between parent/child tables on "
                    "relationship %s - there are multiple foreign key "
                    "paths linking the tables.  Specify the "
                    "'foreign_keys' argument, providing a list of those "
                    "columns which should be counted as containing a "
                    "foreign key reference to the parent table." % self.prop
                ) from afe

    @property
    def primaryjoin_minus_local(self) -> ColumnElement[bool]:
        return _deep_deannotate(self.primaryjoin, values=("local", "remote"))

    @property
    def secondaryjoin_minus_local(self) -> ColumnElement[bool]:
        assert self.secondaryjoin is not None
        return _deep_deannotate(self.secondaryjoin, values=("local", "remote"))

    @util.memoized_property
    def primaryjoin_reverse_remote(self) -> ColumnElement[bool]:
        """Return the primaryjoin condition suitable for the
        "reverse" direction.

        If the primaryjoin was delivered here with pre-existing
        "remote" annotations, the local/remote annotations
        are reversed.  Otherwise, the local/remote annotations
        are removed.

        """
        if self._has_remote_annotations:

            def replace(element: _CE, **kw: Any) -> Optional[_CE]:
                if "remote" in element._annotations:
                    v = dict(element._annotations)
                    del v["remote"]
                    v["local"] = True
                    return element._with_annotations(v)
                elif "local" in element._annotations:
                    v = dict(element._annotations)
                    del v["local"]
                    v["remote"] = True
                    return element._with_annotations(v)

                return None

            return visitors.replacement_traverse(self.primaryjoin, {}, replace)
        else:
            if self._has_foreign_annotations:
                # TODO: coverage
                return _deep_deannotate(
                    self.primaryjoin, values=("local", "remote")
                )
            else:
                return _deep_deannotate(self.primaryjoin)

    def _has_annotation(self, clause: ClauseElement, annotation: str) -> bool:
        for col in visitors.iterate(clause, {}):
            if annotation in col._annotations:
                return True
        else:
            return False

    @util.memoized_property
    def _has_foreign_annotations(self) -> bool:
        return self._has_annotation(self.primaryjoin, "foreign")

    @util.memoized_property
    def _has_remote_annotations(self) -> bool:
        return self._has_annotation(self.primaryjoin, "remote")

    def _annotate_fks(self) -> None:
        """Annotate the primaryjoin and secondaryjoin
        structures with 'foreign' annotations marking columns
        considered as foreign.

        """
        if self._has_foreign_annotations:
            return

        if self.consider_as_foreign_keys:
            self._annotate_from_fk_list()
        else:
            self._annotate_present_fks()

    def _annotate_from_fk_list(self) -> None:
        def check_fk(element: _CE, **kw: Any) -> Optional[_CE]:
            if element in self.consider_as_foreign_keys:
                return element._annotate({"foreign": True})
            return None

        self.primaryjoin = visitors.replacement_traverse(
            self.primaryjoin, {}, check_fk
        )
        if self.secondaryjoin is not None:
            self.secondaryjoin = visitors.replacement_traverse(
                self.secondaryjoin, {}, check_fk
            )

    def _annotate_present_fks(self) -> None:
        if self.secondary is not None:
            secondarycols = util.column_set(self.secondary.c)
        else:
            secondarycols = set()

        def is_foreign(
            a: ColumnElement[Any], b: ColumnElement[Any]
        ) -> Optional[ColumnElement[Any]]:
            if isinstance(a, schema.Column) and isinstance(b, schema.Column):
                if a.references(b):
                    return a
                elif b.references(a):
                    return b

            if secondarycols:
                if a in secondarycols and b not in secondarycols:
                    return a
                elif b in secondarycols and a not in secondarycols:
                    return b

            return None

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            if not isinstance(
                binary.left, sql.ColumnElement
            ) or not isinstance(binary.right, sql.ColumnElement):
                return

            if (
                "foreign" not in binary.left._annotations
                and "foreign" not in binary.right._annotations
            ):
                col = is_foreign(binary.left, binary.right)
                if col is not None:
                    if col.compare(binary.left):
                        binary.left = binary.left._annotate({"foreign": True})
                    elif col.compare(binary.right):
                        binary.right = binary.right._annotate(
                            {"foreign": True}
                        )

        self.primaryjoin = visitors.cloned_traverse(
            self.primaryjoin, {}, {"binary": visit_binary}
        )
        if self.secondaryjoin is not None:
            self.secondaryjoin = visitors.cloned_traverse(
                self.secondaryjoin, {}, {"binary": visit_binary}
            )

    def _refers_to_parent_table(self) -> bool:
        """Return True if the join condition contains column
        comparisons where both columns are in both tables.

        """
        pt = self.parent_persist_selectable
        mt = self.child_persist_selectable
        result = False

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            nonlocal result
            c, f = binary.left, binary.right
            if (
                isinstance(c, expression.ColumnClause)
                and isinstance(f, expression.ColumnClause)
                and pt.is_derived_from(c.table)
                and pt.is_derived_from(f.table)
                and mt.is_derived_from(c.table)
                and mt.is_derived_from(f.table)
            ):
                result = True

        visitors.traverse(self.primaryjoin, {}, {"binary": visit_binary})
        return result

    def _tables_overlap(self) -> bool:
        """Return True if parent/child tables have some overlap."""

        return selectables_overlap(
            self.parent_persist_selectable, self.child_persist_selectable
        )

    def _annotate_remote(self) -> None:
        """Annotate the primaryjoin and secondaryjoin
        structures with 'remote' annotations marking columns
        considered as part of the 'remote' side.

        """
        if self._has_remote_annotations:
            return

        if self.secondary is not None:
            self._annotate_remote_secondary()
        elif self._local_remote_pairs or self._remote_side:
            self._annotate_remote_from_args()
        elif self._refers_to_parent_table():
            self._annotate_selfref(
                lambda col: "foreign" in col._annotations, False
            )
        elif self._tables_overlap():
            self._annotate_remote_with_overlap()
        else:
            self._annotate_remote_distinct_selectables()

    def _annotate_remote_secondary(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when 'secondary' is present.

        """

        assert self.secondary is not None
        fixed_secondary = self.secondary

        def repl(element: _CE, **kw: Any) -> Optional[_CE]:
            if fixed_secondary.c.contains_column(element):
                return element._annotate({"remote": True})
            return None

        self.primaryjoin = visitors.replacement_traverse(
            self.primaryjoin, {}, repl
        )

        assert self.secondaryjoin is not None
        self.secondaryjoin = visitors.replacement_traverse(
            self.secondaryjoin, {}, repl
        )

    def _annotate_selfref(
        self, fn: Callable[[ColumnElement[Any]], bool], remote_side_given: bool
    ) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the relationship is detected as self-referential.

        """

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            equated = binary.left.compare(binary.right)
            if isinstance(binary.left, expression.ColumnClause) and isinstance(
                binary.right, expression.ColumnClause
            ):
                # assume one to many - FKs are "remote"
                if fn(binary.left):
                    binary.left = binary.left._annotate({"remote": True})
                if fn(binary.right) and not equated:
                    binary.right = binary.right._annotate({"remote": True})
            elif not remote_side_given:
                self._warn_non_column_elements()

        self.primaryjoin = visitors.cloned_traverse(
            self.primaryjoin, {}, {"binary": visit_binary}
        )

    def _annotate_remote_from_args(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the 'remote_side' or '_local_remote_pairs'
        arguments are used.

        """
        if self._local_remote_pairs:
            if self._remote_side:
                raise sa_exc.ArgumentError(
                    "remote_side argument is redundant "
                    "against more detailed _local_remote_side "
                    "argument."
                )

            remote_side = [r for (l, r) in self._local_remote_pairs]
        else:
            remote_side = self._remote_side

        if self._refers_to_parent_table():
            self._annotate_selfref(lambda col: col in remote_side, True)
        else:

            def repl(element: _CE, **kw: Any) -> Optional[_CE]:
                # use set() to avoid generating ``__eq__()`` expressions
                # against each element
                if element in set(remote_side):
                    return element._annotate({"remote": True})
                return None

            self.primaryjoin = visitors.replacement_traverse(
                self.primaryjoin, {}, repl
            )

    def _annotate_remote_with_overlap(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the parent/child tables have some set of
        tables in common, though is not a fully self-referential
        relationship.

        """

        def visit_binary(binary: BinaryExpression[Any]) -> None:
            binary.left, binary.right = proc_left_right(
                binary.left, binary.right
            )
            binary.right, binary.left = proc_left_right(
                binary.right, binary.left
            )

        check_entities = (
            self.prop is not None and self.prop.mapper is not self.prop.parent
        )

        def proc_left_right(
            left: ColumnElement[Any], right: ColumnElement[Any]
        ) -> Tuple[ColumnElement[Any], ColumnElement[Any]]:
            if isinstance(left, expression.ColumnClause) and isinstance(
                right, expression.ColumnClause
            ):
                if self.child_persist_selectable.c.contains_column(
                    right
                ) and self.parent_persist_selectable.c.contains_column(left):
                    right = right._annotate({"remote": True})
            elif (
                check_entities
                and right._annotations.get("parentmapper") is self.prop.mapper
            ):
                right = right._annotate({"remote": True})
            elif (
                check_entities
                and left._annotations.get("parentmapper") is self.prop.mapper
            ):
                left = left._annotate({"remote": True})
            else:
                self._warn_non_column_elements()

            return left, right

        self.primaryjoin = visitors.cloned_traverse(
            self.primaryjoin, {}, {"binary": visit_binary}
        )

    def _annotate_remote_distinct_selectables(self) -> None:
        """annotate 'remote' in primaryjoin, secondaryjoin
        when the parent/child tables are entirely
        separate.

        """

        def repl(element: _CE, **kw: Any) -> Optional[_CE]:
            if self.child_persist_selectable.c.contains_column(element) and (
                not self.parent_local_selectable.c.contains_column(element)
                or self.child_local_selectable.c.contains_column(element)
            ):
                return element._annotate({"remote": True})
            return None

        self.primaryjoin = visitors.replacement_traverse(
            self.primaryjoin, {}, repl
        )

    def _warn_non_column_elements(self) -> None:
        util.warn(
            "Non-simple column elements in primary "
            "join condition for property %s - consider using "
            "remote() annotations to mark the remote side." % self.prop
        )

    def _annotate_local(self) -> None:
        """Annotate the primaryjoin and secondaryjoin
        structures with 'local' annotations.

        This annotates all column elements found
        simultaneously in the parent table
        and the join condition that don't have a
        'remote' annotation set up from
        _annotate_remote() or user-defined.

        """
        if self._has_annotation(self.primaryjoin, "local"):
            return

        if self._local_remote_pairs:
            local_side = util.column_set(
                [l for (l, r) in self._local_remote_pairs]
            )
        else:
            local_side = util.column_set(self.parent_persist_selectable.c)

        def locals_(element: _CE, **kw: Any) -> Optional[_CE]:
            if "remote" not in element._annotations and element in local_side:
                return element._annotate({"local": True})
            return None

        self.primaryjoin = visitors.replacement_traverse(
            self.primaryjoin, {}, locals_
        )

    def _annotate_parentmapper(self) -> None:
        def parentmappers_(element: _CE, **kw: Any) -> Optional[_CE]:
            if "remote" in element._annotations:
                return element._annotate({"parentmapper": self.prop.mapper})
            elif "local" in element._annotations:
                return element._annotate({"parentmapper": self.prop.parent})
            return None

        self.primaryjoin = visitors.replacement_traverse(
            self.primaryjoin, {}, parentmappers_
        )

    def _check_remote_side(self) -> None:
        if not self.local_remote_pairs:
            raise sa_exc.ArgumentError(
                "Relationship %s could "
                "not determine any unambiguous local/remote column "
                "pairs based on join condition and remote_side "
                "arguments.  "
                "Consider using the remote() annotation to "
                "accurately mark those elements of the join "
                "condition that are on the remote side of "
                "the relationship." % (self.prop,)
            )
        else:
            not_target = util.column_set(
                self.parent_persist_selectable.c
            ).difference(self.child_persist_selectable.c)

            for _, rmt in self.local_remote_pairs:
                if rmt in not_target:
                    util.warn(
                        "Expression %s is marked as 'remote', but these "
                        "column(s) are local to the local side.  The "
                        "remote() annotation is needed only for a "
                        "self-referential relationship where both sides "
                        "of the relationship refer to the same tables."
                        % (rmt,)
                    )

    def _check_foreign_cols(
        self, join_condition: ColumnElement[bool], primary: bool
    ) -> None:
        """Check the foreign key columns collected and emit error
        messages."""
        foreign_cols = self._gather_columns_with_annotation(
            join_condition, "foreign"
        )

        has_foreign = bool(foreign_cols)

        if primary:
            can_sync = bool(self.synchronize_pairs)
        else:
            can_sync = bool(self.secondary_synchronize_pairs)

        if (
            self.support_sync
            and can_sync
            or (not self.support_sync and has_foreign)
        ):
            return

        # from here below is just determining the best error message
        # to report.  Check for a join condition using any operator
        # (not just ==), perhaps they need to turn on "viewonly=True".
        if self.support_sync and has_foreign and not can_sync:
            err = (
                "Could not locate any simple equality expressions "
                "involving locally mapped foreign key columns for "
                "%s join condition "
                "'%s' on relationship %s."
                % (
                    primary and "primary" or "secondary",
                    join_condition,
                    self.prop,
                )
            )
            err += (
                "  Ensure that referencing columns are associated "
                "with a ForeignKey or ForeignKeyConstraint, or are "
                "annotated in the join condition with the foreign() "
                "annotation. To allow comparison operators other than "
                "'==', the relationship can be marked as viewonly=True."
            )

            raise sa_exc.ArgumentError(err)
        else:
            err = (
                "Could not locate any relevant foreign key columns "
                "for %s join condition '%s' on relationship %s."
                % (
                    primary and "primary" or "secondary",
                    join_condition,
                    self.prop,
                )
            )
            err += (
                "  Ensure that referencing columns are associated "
                "with a ForeignKey or ForeignKeyConstraint, or are "
                "annotated in the join condition with the foreign() "
                "annotation."
            )
            raise sa_exc.ArgumentError(err)

    def _determine_direction(self) -> None:
        """Determine if this relationship is one to many, many to one,
        many to many.

        """
        if self.secondaryjoin is not None:
            self.direction = MANYTOMANY
        else:
            parentcols = util.column_set(self.parent_persist_selectable.c)
            targetcols = util.column_set(self.child_persist_selectable.c)

            # fk collection which suggests ONETOMANY.
            onetomany_fk = targetcols.intersection(self.foreign_key_columns)

            # fk collection which suggests MANYTOONE.

            manytoone_fk = parentcols.intersection(self.foreign_key_columns)

            if onetomany_fk and manytoone_fk:
                # fks on both sides.  test for overlap of local/remote
                # with foreign key.
                # we will gather columns directly from their annotations
                # without deannotating, so that we can distinguish on a column
                # that refers to itself.

                # 1. columns that are both remote and FK suggest
                # onetomany.
                onetomany_local = self._gather_columns_with_annotation(
                    self.primaryjoin, "remote", "foreign"
                )

                # 2. columns that are FK but are not remote (e.g. local)
                # suggest manytoone.
                manytoone_local = {
                    c
                    for c in self._gather_columns_with_annotation(
                        self.primaryjoin, "foreign"
                    )
                    if "remote" not in c._annotations
                }

                # 3. if both collections are present, remove columns that
                # refer to themselves.  This is for the case of
                # and_(Me.id == Me.remote_id, Me.version == Me.version)
                if onetomany_local and manytoone_local:
                    self_equated = self.remote_columns.intersection(
                        self.local_columns
                    )
                    onetomany_local = onetomany_local.difference(self_equated)
                    manytoone_local = manytoone_local.difference(self_equated)

                # at this point, if only one or the other collection is
                # present, we know the direction, otherwise it's still
                # ambiguous.

                if onetomany_local and not manytoone_local:
                    self.direction = ONETOMANY
                elif manytoone_local and not onetomany_local:
                    self.direction = MANYTOONE
                else:
                    raise sa_exc.ArgumentError(
                        "Can't determine relationship"
                        " direction for relationship '%s' - foreign "
                        "key columns within the join condition are present "
                        "in both the parent and the child's mapped tables.  "
                        "Ensure that only those columns referring "
                        "to a parent column are marked as foreign, "
                        "either via the foreign() annotation or "
                        "via the foreign_keys argument." % self.prop
                    )
            elif onetomany_fk:
                self.direction = ONETOMANY
            elif manytoone_fk:
                self.direction = MANYTOONE
            else:
                raise sa_exc.ArgumentError(
                    "Can't determine relationship "
                    "direction for relationship '%s' - foreign "
                    "key columns are present in neither the parent "
                    "nor the child's mapped tables" % self.prop
                )

    def _deannotate_pairs(
        self, collection: _ColumnPairIterable
    ) -> _MutableColumnPairs:
        """provide deannotation for the various lists of
        pairs, so that using them in hashes doesn't incur
        high-overhead __eq__() comparisons against
        original columns mapped.

        """
        return [(x._deannotate(), y._deannotate()) for x, y in collection]

    def _setup_pairs(self) -> None:
        sync_pairs: _MutableColumnPairs = []
        lrp: util.OrderedSet[Tuple[ColumnElement[Any], ColumnElement[Any]]] = (
            util.OrderedSet([])
        )
        secondary_sync_pairs: _MutableColumnPairs = []

        def go(
            joincond: ColumnElement[bool],
            collection: _MutableColumnPairs,
        ) -> None:
            def visit_binary(
                binary: BinaryExpression[Any],
                left: ColumnElement[Any],
                right: ColumnElement[Any],
            ) -> None:
                if (
                    "remote" in right._annotations
                    and "remote" not in left._annotations
                    and self.can_be_synced_fn(left)
                ):
                    lrp.add((left, right))
                elif (
                    "remote" in left._annotations
                    and "remote" not in right._annotations
                    and self.can_be_synced_fn(right)
                ):
                    lrp.add((right, left))
                if binary.operator is operators.eq and self.can_be_synced_fn(
                    left, right
                ):
                    if "foreign" in right._annotations:
                        collection.append((left, right))
                    elif "foreign" in left._annotations:
                        collection.append((right, left))

            visit_binary_product(visit_binary, joincond)

        for joincond, collection in [
            (self.primaryjoin, sync_pairs),
            (self.secondaryjoin, secondary_sync_pairs),
        ]:
            if joincond is None:
                continue
            go(joincond, collection)

        self.local_remote_pairs = self._deannotate_pairs(lrp)
        self.synchronize_pairs = self._deannotate_pairs(sync_pairs)
        self.secondary_synchronize_pairs = self._deannotate_pairs(
            secondary_sync_pairs
        )

    _track_overlapping_sync_targets: weakref.WeakKeyDictionary[
        ColumnElement[Any],
        weakref.WeakKeyDictionary[
            RelationshipProperty[Any], ColumnElement[Any]
        ],
    ] = weakref.WeakKeyDictionary()

    def _warn_for_conflicting_sync_targets(self) -> None:
        if not self.support_sync:
            return

        # we would like to detect if we are synchronizing any column
        # pairs in conflict with another relationship that wishes to sync
        # an entirely different column to the same target.   This is a
        # very rare edge case so we will try to minimize the memory/overhead
        # impact of this check
        for from_, to_ in [
            (from_, to_) for (from_, to_) in self.synchronize_pairs
        ] + [
            (from_, to_) for (from_, to_) in self.secondary_synchronize_pairs
        ]:
            # save ourselves a ton of memory and overhead by only
            # considering columns that are subject to a overlapping
            # FK constraints at the core level.   This condition can arise
            # if multiple relationships overlap foreign() directly, but
            # we're going to assume it's typically a ForeignKeyConstraint-
            # level configuration that benefits from this warning.

            if to_ not in self._track_overlapping_sync_targets:
                self._track_overlapping_sync_targets[to_] = (
                    weakref.WeakKeyDictionary({self.prop: from_})
                )
            else:
                other_props = []
                prop_to_from = self._track_overlapping_sync_targets[to_]

                for pr, fr_ in prop_to_from.items():
                    if (
                        not pr.mapper._dispose_called
                        and pr not in self.prop._reverse_property
                        and pr.key not in self.prop._overlaps
                        and self.prop.key not in pr._overlaps
                        # note: the "__*" symbol is used internally by
                        # SQLAlchemy as a general means of suppressing the
                        # overlaps warning for some extension cases, however
                        # this is not currently
                        # a publicly supported symbol and may change at
                        # any time.
                        and "__*" not in self.prop._overlaps
                        and "__*" not in pr._overlaps
                        and not self.prop.parent.is_sibling(pr.parent)
                        and not self.prop.mapper.is_sibling(pr.mapper)
                        and not self.prop.parent.is_sibling(pr.mapper)
                        and not self.prop.mapper.is_sibling(pr.parent)
                        and (
                            self.prop.key != pr.key
                            or not self.prop.parent.common_parent(pr.parent)
                        )
                    ):
                        other_props.append((pr, fr_))

                if other_props:
                    util.warn(
                        "relationship '%s' will copy column %s to column %s, "
                        "which conflicts with relationship(s): %s. "
                        "If this is not the intention, consider if these "
                        "relationships should be linked with "
                        "back_populates, or if viewonly=True should be "
                        "applied to one or more if they are read-only. "
                        "For the less common case that foreign key "
                        "constraints are partially overlapping, the "
                        "orm.foreign() "
                        "annotation can be used to isolate the columns that "
                        "should be written towards.   To silence this "
                        "warning, add the parameter 'overlaps=\"%s\"' to the "
                        "'%s' relationship."
                        % (
                            self.prop,
                            from_,
                            to_,
                            ", ".join(
                                sorted(
                                    "'%s' (copies %s to %s)" % (pr, fr_, to_)
                                    for (pr, fr_) in other_props
                                )
                            ),
                            ",".join(sorted(pr.key for pr, fr in other_props)),
                            self.prop,
                        ),
                        code="qzyx",
                    )
                self._track_overlapping_sync_targets[to_][self.prop] = from_

    @util.memoized_property
    def remote_columns(self) -> Set[ColumnElement[Any]]:
        return self._gather_join_annotations("remote")

    @util.memoized_property
    def local_columns(self) -> Set[ColumnElement[Any]]:
        return self._gather_join_annotations("local")

    @util.memoized_property
    def foreign_key_columns(self) -> Set[ColumnElement[Any]]:
        return self._gather_join_annotations("foreign")

    def _gather_join_annotations(
        self, annotation: str
    ) -> Set[ColumnElement[Any]]:
        s = set(
            self._gather_columns_with_annotation(self.primaryjoin, annotation)
        )
        if self.secondaryjoin is not None:
            s.update(
                self._gather_columns_with_annotation(
                    self.secondaryjoin, annotation
                )
            )
        return {x._deannotate() for x in s}

    def _gather_columns_with_annotation(
        self, clause: ColumnElement[Any], *annotation: Iterable[str]
    ) -> Set[ColumnElement[Any]]:
        annotation_set = set(annotation)
        return {
            cast(ColumnElement[Any], col)
            for col in visitors.iterate(clause, {})
            if annotation_set.issubset(col._annotations)
        }

    @util.memoized_property
    def _secondary_lineage_set(self) -> FrozenSet[ColumnElement[Any]]:
        if self.secondary is not None:
            return frozenset(
                itertools.chain(*[c.proxy_set for c in self.secondary.c])
            )
        else:
            return util.EMPTY_SET

    def join_targets(
        self,
        source_selectable: Optional[FromClause],
        dest_selectable: FromClause,
        aliased: bool,
        single_crit: Optional[ColumnElement[bool]] = None,
        extra_criteria: Tuple[ColumnElement[bool], ...] = (),
    ) -> Tuple[
        ColumnElement[bool],
        Optional[ColumnElement[bool]],
        Optional[FromClause],
        Optional[ClauseAdapter],
        FromClause,
    ]:
        """Given a source and destination selectable, create a
        join between them.

        This takes into account aliasing the join clause
        to reference the appropriate corresponding columns
        in the target objects, as well as the extra child
        criterion, equivalent column sets, etc.

        """
        # place a barrier on the destination such that
        # replacement traversals won't ever dig into it.
        # its internal structure remains fixed
        # regardless of context.
        dest_selectable = _shallow_annotate(
            dest_selectable, {"no_replacement_traverse": True}
        )

        primaryjoin, secondaryjoin, secondary = (
            self.primaryjoin,
            self.secondaryjoin,
            self.secondary,
        )

        # adjust the join condition for single table inheritance,
        # in the case that the join is to a subclass
        # this is analogous to the
        # "_adjust_for_single_table_inheritance()" method in Query.

        if single_crit is not None:
            if secondaryjoin is not None:
                secondaryjoin = secondaryjoin & single_crit
            else:
                primaryjoin = primaryjoin & single_crit

        if extra_criteria:

            def mark_exclude_cols(
                elem: SupportsAnnotations, annotations: _AnnotationDict
            ) -> SupportsAnnotations:
                """note unrelated columns in the "extra criteria" as either
                should be adapted or not adapted, even though they are not
                part of our "local" or "remote" side.

                see #9779 for this case, as well as #11010 for a follow up

                """

                parentmapper_for_element = elem._annotations.get(
                    "parentmapper", None
                )

                if (
                    parentmapper_for_element is not self.prop.parent
                    and parentmapper_for_element is not self.prop.mapper
                    and elem not in self._secondary_lineage_set
                ):
                    return _safe_annotate(elem, annotations)
                else:
                    return elem

            extra_criteria = tuple(
                _deep_annotate(
                    elem,
                    {"should_not_adapt": True},
                    annotate_callable=mark_exclude_cols,
                )
                for elem in extra_criteria
            )

            if secondaryjoin is not None:
                secondaryjoin = secondaryjoin & sql.and_(*extra_criteria)
            else:
                primaryjoin = primaryjoin & sql.and_(*extra_criteria)

        if aliased:
            if secondary is not None:
                secondary = secondary._anonymous_fromclause(flat=True)
                primary_aliasizer = ClauseAdapter(
                    secondary,
                    exclude_fn=_local_col_exclude,
                )
                secondary_aliasizer = ClauseAdapter(
                    dest_selectable, equivalents=self.child_equivalents
                ).chain(primary_aliasizer)
                if source_selectable is not None:
                    primary_aliasizer = ClauseAdapter(
                        secondary,
                        exclude_fn=_local_col_exclude,
                    ).chain(
                        ClauseAdapter(
                            source_selectable,
                            equivalents=self.parent_equivalents,
                        )
                    )

                secondaryjoin = secondary_aliasizer.traverse(secondaryjoin)
            else:
                primary_aliasizer = ClauseAdapter(
                    dest_selectable,
                    exclude_fn=_local_col_exclude,
                    equivalents=self.child_equivalents,
                )
                if source_selectable is not None:
                    primary_aliasizer.chain(
                        ClauseAdapter(
                            source_selectable,
                            exclude_fn=_remote_col_exclude,
                            equivalents=self.parent_equivalents,
                        )
                    )
                secondary_aliasizer = None

            primaryjoin = primary_aliasizer.traverse(primaryjoin)
            target_adapter = secondary_aliasizer or primary_aliasizer
            target_adapter.exclude_fn = None
        else:
            target_adapter = None
        return (
            primaryjoin,
            secondaryjoin,
            secondary,
            target_adapter,
            dest_selectable,
        )

    def create_lazy_clause(self, reverse_direction: bool = False) -> Tuple[
        ColumnElement[bool],
        Dict[str, ColumnElement[Any]],
        Dict[ColumnElement[Any], ColumnElement[Any]],
    ]:
        binds: Dict[ColumnElement[Any], BindParameter[Any]] = {}
        equated_columns: Dict[ColumnElement[Any], ColumnElement[Any]] = {}

        has_secondary = self.secondaryjoin is not None

        if has_secondary:
            lookup = collections.defaultdict(list)
            for l, r in self.local_remote_pairs:
                lookup[l].append((l, r))
                equated_columns[r] = l
        elif not reverse_direction:
            for l, r in self.local_remote_pairs:
                equated_columns[r] = l
        else:
            for l, r in self.local_remote_pairs:
                equated_columns[l] = r

        def col_to_bind(
            element: ColumnElement[Any], **kw: Any
        ) -> Optional[BindParameter[Any]]:
            if (
                (not reverse_direction and "local" in element._annotations)
                or reverse_direction
                and (
                    (has_secondary and element in lookup)
                    or (not has_secondary and "remote" in element._annotations)
                )
            ):
                if element not in binds:
                    binds[element] = sql.bindparam(
                        None, None, type_=element.type, unique=True
                    )
                return binds[element]
            return None

        lazywhere = self.primaryjoin
        if self.secondaryjoin is None or not reverse_direction:
            lazywhere = visitors.replacement_traverse(
                lazywhere, {}, col_to_bind
            )

        if self.secondaryjoin is not None:
            secondaryjoin = self.secondaryjoin
            if reverse_direction:
                secondaryjoin = visitors.replacement_traverse(
                    secondaryjoin, {}, col_to_bind
                )
            lazywhere = sql.and_(lazywhere, secondaryjoin)

        bind_to_col = {binds[col].key: col for col in binds}

        return lazywhere, bind_to_col, equated_columns


class _ColInAnnotations:
    """Serializable object that tests for names in c._annotations.

    TODO: does this need to be serializable anymore?  can we find what the
    use case was for that?

    """

    __slots__ = ("names",)

    def __init__(self, *names: str):
        self.names = frozenset(names)

    def __call__(self, c: ClauseElement) -> bool:
        return bool(self.names.intersection(c._annotations))


_local_col_exclude = _ColInAnnotations("local", "should_not_adapt")
_remote_col_exclude = _ColInAnnotations("remote", "should_not_adapt")


class Relationship(
    RelationshipProperty[_T],
    _DeclarativeMapped[_T],
):
    """Describes an object property that holds a single item or list
    of items that correspond to a related database table.

    Public constructor is the :func:`_orm.relationship` function.

    .. seealso::

        :ref:`relationship_config_toplevel`

    .. versionchanged:: 2.0 Added :class:`_orm.Relationship` as a Declarative
       compatible subclass for :class:`_orm.RelationshipProperty`.

    """

    inherit_cache = True
    """:meta private:"""


class _RelationshipDeclared(  # type: ignore[misc]
    Relationship[_T],
    WriteOnlyMapped[_T],  # not compatible with Mapped[_T]
    DynamicMapped[_T],  # not compatible with Mapped[_T]
):
    """Relationship subclass used implicitly for declarative mapping."""

    inherit_cache = True
    """:meta private:"""

    @classmethod
    def _mapper_property_name(cls) -> str:
        return "Relationship"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\ply\yacc.py ===
# -----------------------------------------------------------------------------
# ply: yacc.py
#
# Copyright (C) 2001-2017
# David M. Beazley (Dabeaz LLC)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the David Beazley or Dabeaz LLC may be used to
#   endorse or promote products derived from this software without
#  specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
#
# This implements an LR parser that is constructed from grammar rules defined
# as Python functions. The grammer is specified by supplying the BNF inside
# Python documentation strings.  The inspiration for this technique was borrowed
# from John Aycock's Spark parsing system.  PLY might be viewed as cross between
# Spark and the GNU bison utility.
#
# The current implementation is only somewhat object-oriented. The
# LR parser itself is defined in terms of an object (which allows multiple
# parsers to co-exist).  However, most of the variables used during table
# construction are defined in terms of global variables.  Users shouldn't
# notice unless they are trying to define multiple parsers at the same
# time using threads (in which case they should have their head examined).
#
# This implementation supports both SLR and LALR(1) parsing.  LALR(1)
# support was originally implemented by Elias Ioup (ezioup@alumni.uchicago.edu),
# using the algorithm found in Aho, Sethi, and Ullman "Compilers: Principles,
# Techniques, and Tools" (The Dragon Book).  LALR(1) has since been replaced
# by the more efficient DeRemer and Pennello algorithm.
#
# :::::::: WARNING :::::::
#
# Construction of LR parsing tables is fairly complicated and expensive.
# To make this module run fast, a *LOT* of work has been put into
# optimization---often at the expensive of readability and what might
# consider to be good Python "coding style."   Modify the code at your
# own risk!
# ----------------------------------------------------------------------------

import re
import types
import sys
import os.path
import inspect
import base64
import warnings

__version__    = '3.10'
__tabversion__ = '3.10'

#-----------------------------------------------------------------------------
#                     === User configurable parameters ===
#
# Change these to modify the default behavior of yacc (if you wish)
#-----------------------------------------------------------------------------

yaccdebug   = True             # Debugging mode.  If set, yacc generates a
                               # a 'parser.out' file in the current directory

debug_file  = 'parser.out'     # Default name of the debugging file
tab_module  = 'parsetab'       # Default name of the table module
default_lr  = 'LALR'           # Default LR table generation method

error_count = 3                # Number of symbols that must be shifted to leave recovery mode

yaccdevel   = False            # Set to True if developing yacc.  This turns off optimized
                               # implementations of certain functions.

resultlimit = 40               # Size limit of results when running in debug mode.

pickle_protocol = 0            # Protocol to use when writing pickle files

# String type-checking compatibility
if sys.version_info[0] < 3:
    string_types = basestring
else:
    string_types = str

MAXINT = sys.maxsize

# This object is a stand-in for a logging object created by the
# logging module.   PLY will use this by default to create things
# such as the parser.out file.  If a user wants more detailed
# information, they can create their own logging object and pass
# it into PLY.

class PlyLogger(object):
    def __init__(self, f):
        self.f = f

    def debug(self, msg, *args, **kwargs):
        self.f.write((msg % args) + '\n')

    info = debug

    def warning(self, msg, *args, **kwargs):
        self.f.write('WARNING: ' + (msg % args) + '\n')

    def error(self, msg, *args, **kwargs):
        self.f.write('ERROR: ' + (msg % args) + '\n')

    critical = debug

# Null logger is used when no output is generated. Does nothing.
class NullLogger(object):
    def __getattribute__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

# Exception raised for yacc-related errors
class YaccError(Exception):
    pass

# Format the result message that the parser produces when running in debug mode.
def format_result(r):
    repr_str = repr(r)
    if '\n' in repr_str:
        repr_str = repr(repr_str)
    if len(repr_str) > resultlimit:
        repr_str = repr_str[:resultlimit] + ' ...'
    result = '<%s @ 0x%x> (%s)' % (type(r).__name__, id(r), repr_str)
    return result

# Format stack entries when the parser is running in debug mode
def format_stack_entry(r):
    repr_str = repr(r)
    if '\n' in repr_str:
        repr_str = repr(repr_str)
    if len(repr_str) < 16:
        return repr_str
    else:
        return '<%s @ 0x%x>' % (type(r).__name__, id(r))

# Panic mode error recovery support.   This feature is being reworked--much of the
# code here is to offer a deprecation/backwards compatible transition

_errok = None
_token = None
_restart = None
_warnmsg = '''PLY: Don't use global functions errok(), token(), and restart() in p_error().
Instead, invoke the methods on the associated parser instance:

    def p_error(p):
        ...
        # Use parser.errok(), parser.token(), parser.restart()
        ...

    parser = yacc.yacc()
'''

def errok():
    warnings.warn(_warnmsg)
    return _errok()

def restart():
    warnings.warn(_warnmsg)
    return _restart()

def token():
    warnings.warn(_warnmsg)
    return _token()

# Utility function to call the p_error() function with some deprecation hacks
def call_errorfunc(errorfunc, token, parser):
    global _errok, _token, _restart
    _errok = parser.errok
    _token = parser.token
    _restart = parser.restart
    r = errorfunc(token)
    try:
        del _errok, _token, _restart
    except NameError:
        pass
    return r

#-----------------------------------------------------------------------------
#                        ===  LR Parsing Engine ===
#
# The following classes are used for the LR parser itself.  These are not
# used during table construction and are independent of the actual LR
# table generation algorithm
#-----------------------------------------------------------------------------

# This class is used to hold non-terminal grammar symbols during parsing.
# It normally has the following attributes set:
#        .type       = Grammar symbol type
#        .value      = Symbol value
#        .lineno     = Starting line number
#        .endlineno  = Ending line number (optional, set automatically)
#        .lexpos     = Starting lex position
#        .endlexpos  = Ending lex position (optional, set automatically)

class YaccSymbol:
    def __str__(self):
        return self.type

    def __repr__(self):
        return str(self)

# This class is a wrapper around the objects actually passed to each
# grammar rule.   Index lookup and assignment actually assign the
# .value attribute of the underlying YaccSymbol object.
# The lineno() method returns the line number of a given
# item (or 0 if not defined).   The linespan() method returns
# a tuple of (startline,endline) representing the range of lines
# for a symbol.  The lexspan() method returns a tuple (lexpos,endlexpos)
# representing the range of positional information for a symbol.

class YaccProduction:
    def __init__(self, s, stack=None):
        self.slice = s
        self.stack = stack
        self.lexer = None
        self.parser = None

    def __getitem__(self, n):
        if isinstance(n, slice):
            return [s.value for s in self.slice[n]]
        elif n >= 0:
            return self.slice[n].value
        else:
            return self.stack[n].value

    def __setitem__(self, n, v):
        self.slice[n].value = v

    def __getslice__(self, i, j):
        return [s.value for s in self.slice[i:j]]

    def __len__(self):
        return len(self.slice)

    def lineno(self, n):
        return getattr(self.slice[n], 'lineno', 0)

    def set_lineno(self, n, lineno):
        self.slice[n].lineno = lineno

    def linespan(self, n):
        startline = getattr(self.slice[n], 'lineno', 0)
        endline = getattr(self.slice[n], 'endlineno', startline)
        return startline, endline

    def lexpos(self, n):
        return getattr(self.slice[n], 'lexpos', 0)

    def lexspan(self, n):
        startpos = getattr(self.slice[n], 'lexpos', 0)
        endpos = getattr(self.slice[n], 'endlexpos', startpos)
        return startpos, endpos

    def error(self):
        raise SyntaxError

# -----------------------------------------------------------------------------
#                               == LRParser ==
#
# The LR Parsing engine.
# -----------------------------------------------------------------------------

class LRParser:
    def __init__(self, lrtab, errorf):
        self.productions = lrtab.lr_productions
        self.action = lrtab.lr_action
        self.goto = lrtab.lr_goto
        self.errorfunc = errorf
        self.set_defaulted_states()
        self.errorok = True

    def errok(self):
        self.errorok = True

    def restart(self):
        del self.statestack[:]
        del self.symstack[:]
        sym = YaccSymbol()
        sym.type = '$end'
        self.symstack.append(sym)
        self.statestack.append(0)

    # Defaulted state support.
    # This method identifies parser states where there is only one possible reduction action.
    # For such states, the parser can make a choose to make a rule reduction without consuming
    # the next look-ahead token.  This delayed invocation of the tokenizer can be useful in
    # certain kinds of advanced parsing situations where the lexer and parser interact with
    # each other or change states (i.e., manipulation of scope, lexer states, etc.).
    #
    # See:  https://www.gnu.org/software/bison/manual/html_node/Default-Reductions.html#Default-Reductions
    def set_defaulted_states(self):
        self.defaulted_states = {}
        for state, actions in self.action.items():
            rules = list(actions.values())
            if len(rules) == 1 and rules[0] < 0:
                self.defaulted_states[state] = rules[0]

    def disable_defaulted_states(self):
        self.defaulted_states = {}

    def parse(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        if debug or yaccdevel:
            if isinstance(debug, int):
                debug = PlyLogger(sys.stderr)
            return self.parsedebug(input, lexer, debug, tracking, tokenfunc)
        elif tracking:
            return self.parseopt(input, lexer, debug, tracking, tokenfunc)
        else:
            return self.parseopt_notrack(input, lexer, debug, tracking, tokenfunc)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # parsedebug().
    #
    # This is the debugging enabled version of parse().  All changes made to the
    # parsing engine should be made here.   Optimized versions of this function
    # are automatically created by the ply/ygen.py script.  This script cuts out
    # sections enclosed in markers such as this:
    #
    #      #--! DEBUG
    #      statements
    #      #--! DEBUG
    #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def parsedebug(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        #--! parsedebug-start
        lookahead = None                         # Current lookahead symbol
        lookaheadstack = []                      # Stack of lookahead symbols
        actions = self.action                    # Local reference to action table (to avoid lookup on self.)
        goto    = self.goto                      # Local reference to goto table (to avoid lookup on self.)
        prod    = self.productions               # Local reference to production list (to avoid lookup on self.)
        defaulted_states = self.defaulted_states # Local reference to defaulted states
        pslice  = YaccProduction(None)           # Production object passed to grammar rules
        errorcount = 0                           # Used during error recovery

        #--! DEBUG
        debug.info('PLY: PARSE DEBUG START')
        #--! DEBUG

        # If no lexer was given, we will try to use the lex module
        if not lexer:
            from . import lex
            lexer = lex.lexer

        # Set up the lexer and parser objects on pslice
        pslice.lexer = lexer
        pslice.parser = self

        # If input was supplied, pass to lexer
        if input is not None:
            lexer.input(input)

        if tokenfunc is None:
            # Tokenize function
            get_token = lexer.token
        else:
            get_token = tokenfunc

        # Set the parser() token method (sometimes used in error recovery)
        self.token = get_token

        # Set up the state and symbol stacks

        statestack = []                # Stack of parsing states
        self.statestack = statestack
        symstack   = []                # Stack of grammar symbols
        self.symstack = symstack

        pslice.stack = symstack         # Put in the production
        errtoken   = None               # Err token

        # The start state is assumed to be (0,$end)

        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            # Get the next symbol on the input.  If a lookahead symbol
            # is already set, we just use that. Otherwise, we'll pull
            # the next token off of the lookaheadstack or from the lexer

            #--! DEBUG
            debug.debug('')
            debug.debug('State  : %s', state)
            #--! DEBUG

            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()     # Get the next token
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'

                # Check the action table
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]
                #--! DEBUG
                debug.debug('Defaulted state %s: Reduce using %d', state, -t)
                #--! DEBUG

            #--! DEBUG
            debug.debug('Stack  : %s',
                        ('%s . %s' % (' '.join([xx.type for xx in symstack][1:]), str(lookahead))).lstrip())
            #--! DEBUG

            if t is not None:
                if t > 0:
                    # shift a symbol on the stack
                    statestack.append(t)
                    state = t

                    #--! DEBUG
                    debug.debug('Action : Shift and goto state %s', t)
                    #--! DEBUG

                    symstack.append(lookahead)
                    lookahead = None

                    # Decrease error count on successful shift
                    if errorcount:
                        errorcount -= 1
                    continue

                if t < 0:
                    # reduce a symbol on the stack, emit a production
                    p = prod[-t]
                    pname = p.name
                    plen  = p.len

                    # Get production function
                    sym = YaccSymbol()
                    sym.type = pname       # Production name
                    sym.value = None

                    #--! DEBUG
                    if plen:
                        debug.info('Action : Reduce rule [%s] with %s and goto state %d', p.str,
                                   '['+','.join([format_stack_entry(_v.value) for _v in symstack[-plen:]])+']',
                                   goto[statestack[-1-plen]][pname])
                    else:
                        debug.info('Action : Reduce rule [%s] with %s and goto state %d', p.str, [],
                                   goto[statestack[-1]][pname])

                    #--! DEBUG

                    if plen:
                        targ = symstack[-plen-1:]
                        targ[0] = sym

                        #--! TRACKING
                        if tracking:
                            t1 = targ[1]
                            sym.lineno = t1.lineno
                            sym.lexpos = t1.lexpos
                            t1 = targ[-1]
                            sym.endlineno = getattr(t1, 'endlineno', t1.lineno)
                            sym.endlexpos = getattr(t1, 'endlexpos', t1.lexpos)
                        #--! TRACKING

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # The code enclosed in this section is duplicated
                        # below as a performance optimization.  Make sure
                        # changes get made in both locations.

                        pslice.slice = targ

                        try:
                            # Call the grammar rule with our special slice object
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            #--! DEBUG
                            debug.info('Result : %s', format_result(pslice[0]))
                            #--! DEBUG
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            # If an error was set. Enter error recovery state
                            lookaheadstack.append(lookahead)    # Save the current lookahead token
                            symstack.extend(targ[1:-1])         # Put the production slice back on the stack
                            statestack.pop()                    # Pop back one state (before the reduce)
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False

                        continue
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    else:

                        #--! TRACKING
                        if tracking:
                            sym.lineno = lexer.lineno
                            sym.lexpos = lexer.lexpos
                        #--! TRACKING

                        targ = [sym]

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # The code enclosed in this section is duplicated
                        # above as a performance optimization.  Make sure
                        # changes get made in both locations.

                        pslice.slice = targ

                        try:
                            # Call the grammar rule with our special slice object
                            self.state = state
                            p.callable(pslice)
                            #--! DEBUG
                            debug.info('Result : %s', format_result(pslice[0]))
                            #--! DEBUG
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            # If an error was set. Enter error recovery state
                            lookaheadstack.append(lookahead)    # Save the current lookahead token
                            statestack.pop()                    # Pop back one state (before the reduce)
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False

                        continue
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    #--! DEBUG
                    debug.info('Done   : Returning %s', format_result(result))
                    debug.info('PLY: PARSE DEBUG END')
                    #--! DEBUG
                    return result

            if t is None:

                #--! DEBUG
                debug.error('Error  : %s',
                            ('%s . %s' % (' '.join([xx.type for xx in symstack][1:]), str(lookahead))).lstrip())
                #--! DEBUG

                # We have some kind of parsing error here.  To handle
                # this, we are going to push the current token onto
                # the tokenstack and replace it with an 'error' token.
                # If there are any synchronization rules, they may
                # catch it.
                #
                # In addition to pushing the error token, we call call
                # the user defined p_error() function if this is the
                # first syntax error.  This function is only called if
                # errorcount == 0.
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None               # End of file!
                    if self.errorfunc:
                        if errtoken and not hasattr(errtoken, 'lexer'):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            # User must have done some kind of panic
                            # mode recovery on their own.  The
                            # returned token is the next lookahead
                            lookahead = tok
                            errtoken = None
                            continue
                    else:
                        if errtoken:
                            if hasattr(errtoken, 'lineno'):
                                lineno = lookahead.lineno
                            else:
                                lineno = 0
                            if lineno:
                                sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                            else:
                                sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                        else:
                            sys.stderr.write('yacc: Parse error in input. EOF\n')
                            return

                else:
                    errorcount = error_count

                # case 1:  the statestack only has 1 entry on it.  If we're in this state, the
                # entire parse has been rolled back and we're completely hosed.   The token is
                # discarded and we just keep going.

                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    # Nuke the pushback stack
                    del lookaheadstack[:]
                    continue

                # case 2: the statestack has a couple of entries on it, but we're
                # at the end of the file. nuke the top entry and generate an error token

                # Start nuking entries on the stack
                if lookahead.type == '$end':
                    # Whoa. We're really hosed here. Bail out
                    return

                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        # Hmmm. Error is on top of stack, we'll just nuke input
                        # symbol and continue
                        #--! TRACKING
                        if tracking:
                            sym.endlineno = getattr(lookahead, 'lineno', sym.lineno)
                            sym.endlexpos = getattr(lookahead, 'lexpos', sym.lexpos)
                        #--! TRACKING
                        lookahead = None
                        continue

                    # Create the error symbol for the first time and make it the new lookahead symbol
                    t = YaccSymbol()
                    t.type = 'error'

                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    #--! TRACKING
                    if tracking:
                        lookahead.lineno = sym.lineno
                        lookahead.lexpos = sym.lexpos
                    #--! TRACKING
                    statestack.pop()
                    state = statestack[-1]

                continue

            # Call an error function here
            raise RuntimeError('yacc: internal parser error!!!\n')

        #--! parsedebug-end

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # parseopt().
    #
    # Optimized version of parse() method.  DO NOT EDIT THIS CODE DIRECTLY!
    # This code is automatically generated by the ply/ygen.py script. Make
    # changes to the parsedebug() method instead.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def parseopt(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        #--! parseopt-start
        lookahead = None                         # Current lookahead symbol
        lookaheadstack = []                      # Stack of lookahead symbols
        actions = self.action                    # Local reference to action table (to avoid lookup on self.)
        goto    = self.goto                      # Local reference to goto table (to avoid lookup on self.)
        prod    = self.productions               # Local reference to production list (to avoid lookup on self.)
        defaulted_states = self.defaulted_states # Local reference to defaulted states
        pslice  = YaccProduction(None)           # Production object passed to grammar rules
        errorcount = 0                           # Used during error recovery


        # If no lexer was given, we will try to use the lex module
        if not lexer:
            from . import lex
            lexer = lex.lexer

        # Set up the lexer and parser objects on pslice
        pslice.lexer = lexer
        pslice.parser = self

        # If input was supplied, pass to lexer
        if input is not None:
            lexer.input(input)

        if tokenfunc is None:
            # Tokenize function
            get_token = lexer.token
        else:
            get_token = tokenfunc

        # Set the parser() token method (sometimes used in error recovery)
        self.token = get_token

        # Set up the state and symbol stacks

        statestack = []                # Stack of parsing states
        self.statestack = statestack
        symstack   = []                # Stack of grammar symbols
        self.symstack = symstack

        pslice.stack = symstack         # Put in the production
        errtoken   = None               # Err token

        # The start state is assumed to be (0,$end)

        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            # Get the next symbol on the input.  If a lookahead symbol
            # is already set, we just use that. Otherwise, we'll pull
            # the next token off of the lookaheadstack or from the lexer


            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()     # Get the next token
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'

                # Check the action table
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]


            if t is not None:
                if t > 0:
                    # shift a symbol on the stack
                    statestack.append(t)
                    state = t


                    symstack.append(lookahead)
                    lookahead = None

                    # Decrease error count on successful shift
                    if errorcount:
                        errorcount -= 1
                    continue

                if t < 0:
                    # reduce a symbol on the stack, emit a production
                    p = prod[-t]
                    pname = p.name
                    plen  = p.len

                    # Get production function
                    sym = YaccSymbol()
                    sym.type = pname       # Production name
                    sym.value = None


                    if plen:
                        targ = symstack[-plen-1:]
                        targ[0] = sym

                        #--! TRACKING
                        if tracking:
                            t1 = targ[1]
                            sym.lineno = t1.lineno
                            sym.lexpos = t1.lexpos
                            t1 = targ[-1]
                            sym.endlineno = getattr(t1, 'endlineno', t1.lineno)
                            sym.endlexpos = getattr(t1, 'endlexpos', t1.lexpos)
                        #--! TRACKING

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # The code enclosed in this section is duplicated
                        # below as a performance optimization.  Make sure
                        # changes get made in both locations.

                        pslice.slice = targ

                        try:
                            # Call the grammar rule with our special slice object
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            # If an error was set. Enter error recovery state
                            lookaheadstack.append(lookahead)    # Save the current lookahead token
                            symstack.extend(targ[1:-1])         # Put the production slice back on the stack
                            statestack.pop()                    # Pop back one state (before the reduce)
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False

                        continue
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    else:

                        #--! TRACKING
                        if tracking:
                            sym.lineno = lexer.lineno
                            sym.lexpos = lexer.lexpos
                        #--! TRACKING

                        targ = [sym]

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # The code enclosed in this section is duplicated
                        # above as a performance optimization.  Make sure
                        # changes get made in both locations.

                        pslice.slice = targ

                        try:
                            # Call the grammar rule with our special slice object
                            self.state = state
                            p.callable(pslice)
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            # If an error was set. Enter error recovery state
                            lookaheadstack.append(lookahead)    # Save the current lookahead token
                            statestack.pop()                    # Pop back one state (before the reduce)
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False

                        continue
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    return result

            if t is None:


                # We have some kind of parsing error here.  To handle
                # this, we are going to push the current token onto
                # the tokenstack and replace it with an 'error' token.
                # If there are any synchronization rules, they may
                # catch it.
                #
                # In addition to pushing the error token, we call call
                # the user defined p_error() function if this is the
                # first syntax error.  This function is only called if
                # errorcount == 0.
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None               # End of file!
                    if self.errorfunc:
                        if errtoken and not hasattr(errtoken, 'lexer'):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            # User must have done some kind of panic
                            # mode recovery on their own.  The
                            # returned token is the next lookahead
                            lookahead = tok
                            errtoken = None
                            continue
                    else:
                        if errtoken:
                            if hasattr(errtoken, 'lineno'):
                                lineno = lookahead.lineno
                            else:
                                lineno = 0
                            if lineno:
                                sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                            else:
                                sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                        else:
                            sys.stderr.write('yacc: Parse error in input. EOF\n')
                            return

                else:
                    errorcount = error_count

                # case 1:  the statestack only has 1 entry on it.  If we're in this state, the
                # entire parse has been rolled back and we're completely hosed.   The token is
                # discarded and we just keep going.

                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    # Nuke the pushback stack
                    del lookaheadstack[:]
                    continue

                # case 2: the statestack has a couple of entries on it, but we're
                # at the end of the file. nuke the top entry and generate an error token

                # Start nuking entries on the stack
                if lookahead.type == '$end':
                    # Whoa. We're really hosed here. Bail out
                    return

                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        # Hmmm. Error is on top of stack, we'll just nuke input
                        # symbol and continue
                        #--! TRACKING
                        if tracking:
                            sym.endlineno = getattr(lookahead, 'lineno', sym.lineno)
                            sym.endlexpos = getattr(lookahead, 'lexpos', sym.lexpos)
                        #--! TRACKING
                        lookahead = None
                        continue

                    # Create the error symbol for the first time and make it the new lookahead symbol
                    t = YaccSymbol()
                    t.type = 'error'

                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    #--! TRACKING
                    if tracking:
                        lookahead.lineno = sym.lineno
                        lookahead.lexpos = sym.lexpos
                    #--! TRACKING
                    statestack.pop()
                    state = statestack[-1]

                continue

            # Call an error function here
            raise RuntimeError('yacc: internal parser error!!!\n')

        #--! parseopt-end

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # parseopt_notrack().
    #
    # Optimized version of parseopt() with line number tracking removed.
    # DO NOT EDIT THIS CODE DIRECTLY. This code is automatically generated
    # by the ply/ygen.py script. Make changes to the parsedebug() method instead.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def parseopt_notrack(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        #--! parseopt-notrack-start
        lookahead = None                         # Current lookahead symbol
        lookaheadstack = []                      # Stack of lookahead symbols
        actions = self.action                    # Local reference to action table (to avoid lookup on self.)
        goto    = self.goto                      # Local reference to goto table (to avoid lookup on self.)
        prod    = self.productions               # Local reference to production list (to avoid lookup on self.)
        defaulted_states = self.defaulted_states # Local reference to defaulted states
        pslice  = YaccProduction(None)           # Production object passed to grammar rules
        errorcount = 0                           # Used during error recovery


        # If no lexer was given, we will try to use the lex module
        if not lexer:
            from . import lex
            lexer = lex.lexer

        # Set up the lexer and parser objects on pslice
        pslice.lexer = lexer
        pslice.parser = self

        # If input was supplied, pass to lexer
        if input is not None:
            lexer.input(input)

        if tokenfunc is None:
            # Tokenize function
            get_token = lexer.token
        else:
            get_token = tokenfunc

        # Set the parser() token method (sometimes used in error recovery)
        self.token = get_token

        # Set up the state and symbol stacks

        statestack = []                # Stack of parsing states
        self.statestack = statestack
        symstack   = []                # Stack of grammar symbols
        self.symstack = symstack

        pslice.stack = symstack         # Put in the production
        errtoken   = None               # Err token

        # The start state is assumed to be (0,$end)

        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            # Get the next symbol on the input.  If a lookahead symbol
            # is already set, we just use that. Otherwise, we'll pull
            # the next token off of the lookaheadstack or from the lexer


            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()     # Get the next token
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'

                # Check the action table
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]


            if t is not None:
                if t > 0:
                    # shift a symbol on the stack
                    statestack.append(t)
                    state = t


                    symstack.append(lookahead)
                    lookahead = None

                    # Decrease error count on successful shift
                    if errorcount:
                        errorcount -= 1
                    continue

                if t < 0:
                    # reduce a symbol on the stack, emit a production
                    p = prod[-t]
                    pname = p.name
                    plen  = p.len

                    # Get production function
                    sym = YaccSymbol()
                    sym.type = pname       # Production name
                    sym.value = None


                    if plen:
                        targ = symstack[-plen-1:]
                        targ[0] = sym


                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # The code enclosed in this section is duplicated
                        # below as a performance optimization.  Make sure
                        # changes get made in both locations.

                        pslice.slice = targ

                        try:
                            # Call the grammar rule with our special slice object
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            # If an error was set. Enter error recovery state
                            lookaheadstack.append(lookahead)    # Save the current lookahead token
                            symstack.extend(targ[1:-1])         # Put the production slice back on the stack
                            statestack.pop()                    # Pop back one state (before the reduce)
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False

                        continue
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    else:


                        targ = [sym]

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # The code enclosed in this section is duplicated
                        # above as a performance optimization.  Make sure
                        # changes get made in both locations.

                        pslice.slice = targ

                        try:
                            # Call the grammar rule with our special slice object
                            self.state = state
                            p.callable(pslice)
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            # If an error was set. Enter error recovery state
                            lookaheadstack.append(lookahead)    # Save the current lookahead token
                            statestack.pop()                    # Pop back one state (before the reduce)
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False

                        continue
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    return result

            if t is None:


                # We have some kind of parsing error here.  To handle
                # this, we are going to push the current token onto
                # the tokenstack and replace it with an 'error' token.
                # If there are any synchronization rules, they may
                # catch it.
                #
                # In addition to pushing the error token, we call call
                # the user defined p_error() function if this is the
                # first syntax error.  This function is only called if
                # errorcount == 0.
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None               # End of file!
                    if self.errorfunc:
                        if errtoken and not hasattr(errtoken, 'lexer'):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            # User must have done some kind of panic
                            # mode recovery on their own.  The
                            # returned token is the next lookahead
                            lookahead = tok
                            errtoken = None
                            continue
                    else:
                        if errtoken:
                            if hasattr(errtoken, 'lineno'):
                                lineno = lookahead.lineno
                            else:
                                lineno = 0
                            if lineno:
                                sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                            else:
                                sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                        else:
                            sys.stderr.write('yacc: Parse error in input. EOF\n')
                            return

                else:
                    errorcount = error_count

                # case 1:  the statestack only has 1 entry on it.  If we're in this state, the
                # entire parse has been rolled back and we're completely hosed.   The token is
                # discarded and we just keep going.

                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    # Nuke the pushback stack
                    del lookaheadstack[:]
                    continue

                # case 2: the statestack has a couple of entries on it, but we're
                # at the end of the file. nuke the top entry and generate an error token

                # Start nuking entries on the stack
                if lookahead.type == '$end':
                    # Whoa. We're really hosed here. Bail out
                    return

                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        # Hmmm. Error is on top of stack, we'll just nuke input
                        # symbol and continue
                        lookahead = None
                        continue

                    # Create the error symbol for the first time and make it the new lookahead symbol
                    t = YaccSymbol()
                    t.type = 'error'

                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    statestack.pop()
                    state = statestack[-1]

                continue

            # Call an error function here
            raise RuntimeError('yacc: internal parser error!!!\n')

        #--! parseopt-notrack-end

# -----------------------------------------------------------------------------
#                          === Grammar Representation ===
#
# The following functions, classes, and variables are used to represent and
# manipulate the rules that make up a grammar.
# -----------------------------------------------------------------------------

# regex matching identifiers
_is_identifier = re.compile(r'^[a-zA-Z0-9_-]+$')

# -----------------------------------------------------------------------------
# class Production:
#
# This class stores the raw information about a single production or grammar rule.
# A grammar rule refers to a specification such as this:
#
#       expr : expr PLUS term
#
# Here are the basic attributes defined on all productions
#
#       name     - Name of the production.  For example 'expr'
#       prod     - A list of symbols on the right side ['expr','PLUS','term']
#       prec     - Production precedence level
#       number   - Production number.
#       func     - Function that executes on reduce
#       file     - File where production function is defined
#       lineno   - Line number where production function is defined
#
# The following attributes are defined or optional.
#
#       len       - Length of the production (number of symbols on right hand side)
#       usyms     - Set of unique symbols found in the production
# -----------------------------------------------------------------------------

class Production(object):
    reduced = 0
    def __init__(self, number, name, prod, precedence=('right', 0), func=None, file='', line=0):
        self.name     = name
        self.prod     = tuple(prod)
        self.number   = number
        self.func     = func
        self.callable = None
        self.file     = file
        self.line     = line
        self.prec     = precedence

        # Internal settings used during table construction

        self.len  = len(self.prod)   # Length of the production

        # Create a list of unique production symbols used in the production
        self.usyms = []
        for s in self.prod:
            if s not in self.usyms:
                self.usyms.append(s)

        # List of all LR items for the production
        self.lr_items = []
        self.lr_next = None

        # Create a string representation
        if self.prod:
            self.str = '%s -> %s' % (self.name, ' '.join(self.prod))
        else:
            self.str = '%s -> <empty>' % self.name

    def __str__(self):
        return self.str

    def __repr__(self):
        return 'Production(' + str(self) + ')'

    def __len__(self):
        return len(self.prod)

    def __nonzero__(self):
        return 1

    def __getitem__(self, index):
        return self.prod[index]

    # Return the nth lr_item from the production (or None if at the end)
    def lr_item(self, n):
        if n > len(self.prod):
            return None
        p = LRItem(self, n)
        # Precompute the list of productions immediately following.
        try:
            p.lr_after = Prodnames[p.prod[n+1]]
        except (IndexError, KeyError):
            p.lr_after = []
        try:
            p.lr_before = p.prod[n-1]
        except IndexError:
            p.lr_before = None
        return p

    # Bind the production function name to a callable
    def bind(self, pdict):
        if self.func:
            self.callable = pdict[self.func]

# This class serves as a minimal standin for Production objects when
# reading table data from files.   It only contains information
# actually used by the LR parsing engine, plus some additional
# debugging information.
class MiniProduction(object):
    def __init__(self, str, name, len, func, file, line):
        self.name     = name
        self.len      = len
        self.func     = func
        self.callable = None
        self.file     = file
        self.line     = line
        self.str      = str

    def __str__(self):
        return self.str

    def __repr__(self):
        return 'MiniProduction(%s)' % self.str

    # Bind the production function name to a callable
    def bind(self, pdict):
        if self.func:
            self.callable = pdict[self.func]


# -----------------------------------------------------------------------------
# class LRItem
#
# This class represents a specific stage of parsing a production rule.  For
# example:
#
#       expr : expr . PLUS term
#
# In the above, the "." represents the current location of the parse.  Here
# basic attributes:
#
#       name       - Name of the production.  For example 'expr'
#       prod       - A list of symbols on the right side ['expr','.', 'PLUS','term']
#       number     - Production number.
#
#       lr_next      Next LR item. Example, if we are ' expr -> expr . PLUS term'
#                    then lr_next refers to 'expr -> expr PLUS . term'
#       lr_index   - LR item index (location of the ".") in the prod list.
#       lookaheads - LALR lookahead symbols for this item
#       len        - Length of the production (number of symbols on right hand side)
#       lr_after    - List of all productions that immediately follow
#       lr_before   - Grammar symbol immediately before
# -----------------------------------------------------------------------------

class LRItem(object):
    def __init__(self, p, n):
        self.name       = p.name
        self.prod       = list(p.prod)
        self.number     = p.number
        self.lr_index   = n
        self.lookaheads = {}
        self.prod.insert(n, '.')
        self.prod       = tuple(self.prod)
        self.len        = len(self.prod)
        self.usyms      = p.usyms

    def __str__(self):
        if self.prod:
            s = '%s -> %s' % (self.name, ' '.join(self.prod))
        else:
            s = '%s -> <empty>' % self.name
        return s

    def __repr__(self):
        return 'LRItem(' + str(self) + ')'

# -----------------------------------------------------------------------------
# rightmost_terminal()
#
# Return the rightmost terminal from a list of symbols.  Used in add_production()
# -----------------------------------------------------------------------------
def rightmost_terminal(symbols, terminals):
    i = len(symbols) - 1
    while i >= 0:
        if symbols[i] in terminals:
            return symbols[i]
        i -= 1
    return None

# -----------------------------------------------------------------------------
#                           === GRAMMAR CLASS ===
#
# The following class represents the contents of the specified grammar along
# with various computed properties such as first sets, follow sets, LR items, etc.
# This data is used for critical parts of the table generation process later.
# -----------------------------------------------------------------------------

class GrammarError(YaccError):
    pass

class Grammar(object):
    def __init__(self, terminals):
        self.Productions  = [None]  # A list of all of the productions.  The first
                                    # entry is always reserved for the purpose of
                                    # building an augmented grammar

        self.Prodnames    = {}      # A dictionary mapping the names of nonterminals to a list of all
                                    # productions of that nonterminal.

        self.Prodmap      = {}      # A dictionary that is only used to detect duplicate
                                    # productions.

        self.Terminals    = {}      # A dictionary mapping the names of terminal symbols to a
                                    # list of the rules where they are used.

        for term in terminals:
            self.Terminals[term] = []

        self.Terminals['error'] = []

        self.Nonterminals = {}      # A dictionary mapping names of nonterminals to a list
                                    # of rule numbers where they are used.

        self.First        = {}      # A dictionary of precomputed FIRST(x) symbols

        self.Follow       = {}      # A dictionary of precomputed FOLLOW(x) symbols

        self.Precedence   = {}      # Precedence rules for each terminal. Contains tuples of the
                                    # form ('right',level) or ('nonassoc', level) or ('left',level)

        self.UsedPrecedence = set() # Precedence rules that were actually used by the grammer.
                                    # This is only used to provide error checking and to generate
                                    # a warning about unused precedence rules.

        self.Start = None           # Starting symbol for the grammar


    def __len__(self):
        return len(self.Productions)

    def __getitem__(self, index):
        return self.Productions[index]

    # -----------------------------------------------------------------------------
    # set_precedence()
    #
    # Sets the precedence for a given terminal. assoc is the associativity such as
    # 'left','right', or 'nonassoc'.  level is a numeric level.
    #
    # -----------------------------------------------------------------------------

    def set_precedence(self, term, assoc, level):
        assert self.Productions == [None], 'Must call set_precedence() before add_production()'
        if term in self.Precedence:
            raise GrammarError('Precedence already specified for terminal %r' % term)
        if assoc not in ['left', 'right', 'nonassoc']:
            raise GrammarError("Associativity must be one of 'left','right', or 'nonassoc'")
        self.Precedence[term] = (assoc, level)

    # -----------------------------------------------------------------------------
    # add_production()
    #
    # Given an action function, this function assembles a production rule and
    # computes its precedence level.
    #
    # The production rule is supplied as a list of symbols.   For example,
    # a rule such as 'expr : expr PLUS term' has a production name of 'expr' and
    # symbols ['expr','PLUS','term'].
    #
    # Precedence is determined by the precedence of the right-most non-terminal
    # or the precedence of a terminal specified by %prec.
    #
    # A variety of error checks are performed to make sure production symbols
    # are valid and that %prec is used correctly.
    # -----------------------------------------------------------------------------

    def add_production(self, prodname, syms, func=None, file='', line=0):

        if prodname in self.Terminals:
            raise GrammarError('%s:%d: Illegal rule name %r. Already defined as a token' % (file, line, prodname))
        if prodname == 'error':
            raise GrammarError('%s:%d: Illegal rule name %r. error is a reserved word' % (file, line, prodname))
        if not _is_identifier.match(prodname):
            raise GrammarError('%s:%d: Illegal rule name %r' % (file, line, prodname))

        # Look for literal tokens
        for n, s in enumerate(syms):
            if s[0] in "'\"":
                try:
                    c = eval(s)
                    if (len(c) > 1):
                        raise GrammarError('%s:%d: Literal token %s in rule %r may only be a single character' %
                                           (file, line, s, prodname))
                    if c not in self.Terminals:
                        self.Terminals[c] = []
                    syms[n] = c
                    continue
                except SyntaxError:
                    pass
            if not _is_identifier.match(s) and s != '%prec':
                raise GrammarError('%s:%d: Illegal name %r in rule %r' % (file, line, s, prodname))

        # Determine the precedence level
        if '%prec' in syms:
            if syms[-1] == '%prec':
                raise GrammarError('%s:%d: Syntax error. Nothing follows %%prec' % (file, line))
            if syms[-2] != '%prec':
                raise GrammarError('%s:%d: Syntax error. %%prec can only appear at the end of a grammar rule' %
                                   (file, line))
            precname = syms[-1]
            prodprec = self.Precedence.get(precname)
            if not prodprec:
                raise GrammarError('%s:%d: Nothing known about the precedence of %r' % (file, line, precname))
            else:
                self.UsedPrecedence.add(precname)
            del syms[-2:]     # Drop %prec from the rule
        else:
            # If no %prec, precedence is determined by the rightmost terminal symbol
            precname = rightmost_terminal(syms, self.Terminals)
            prodprec = self.Precedence.get(precname, ('right', 0))

        # See if the rule is already in the rulemap
        map = '%s -> %s' % (prodname, syms)
        if map in self.Prodmap:
            m = self.Prodmap[map]
            raise GrammarError('%s:%d: Duplicate rule %s. ' % (file, line, m) +
                               'Previous definition at %s:%d' % (m.file, m.line))

        # From this point on, everything is valid.  Create a new Production instance
        pnumber  = len(self.Productions)
        if prodname not in self.Nonterminals:
            self.Nonterminals[prodname] = []

        # Add the production number to Terminals and Nonterminals
        for t in syms:
            if t in self.Terminals:
                self.Terminals[t].append(pnumber)
            else:
                if t not in self.Nonterminals:
                    self.Nonterminals[t] = []
                self.Nonterminals[t].append(pnumber)

        # Create a production and add it to the list of productions
        p = Production(pnumber, prodname, syms, prodprec, func, file, line)
        self.Productions.append(p)
        self.Prodmap[map] = p

        # Add to the global productions list
        try:
            self.Prodnames[prodname].append(p)
        except KeyError:
            self.Prodnames[prodname] = [p]

    # -----------------------------------------------------------------------------
    # set_start()
    #
    # Sets the starting symbol and creates the augmented grammar.  Production
    # rule 0 is S' -> start where start is the start symbol.
    # -----------------------------------------------------------------------------

    def set_start(self, start=None):
        if not start:
            start = self.Productions[1].name
        if start not in self.Nonterminals:
            raise GrammarError('start symbol %s undefined' % start)
        self.Productions[0] = Production(0, "S'", [start])
        self.Nonterminals[start].append(0)
        self.Start = start

    # -----------------------------------------------------------------------------
    # find_unreachable()
    #
    # Find all of the nonterminal symbols that can't be reached from the starting
    # symbol.  Returns a list of nonterminals that can't be reached.
    # -----------------------------------------------------------------------------

    def find_unreachable(self):

        # Mark all symbols that are reachable from a symbol s
        def mark_reachable_from(s):
            if s in reachable:
                return
            reachable.add(s)
            for p in self.Prodnames.get(s, []):
                for r in p.prod:
                    mark_reachable_from(r)

        reachable = set()
        mark_reachable_from(self.Productions[0].prod[0])
        return [s for s in self.Nonterminals if s not in reachable]

    # -----------------------------------------------------------------------------
    # infinite_cycles()
    #
    # This function looks at the various parsing rules and tries to detect
    # infinite recursion cycles (grammar rules where there is no possible way
    # to derive a string of only terminals).
    # -----------------------------------------------------------------------------

    def infinite_cycles(self):
        terminates = {}

        # Terminals:
        for t in self.Terminals:
            terminates[t] = True

        terminates['$end'] = True

        # Nonterminals:

        # Initialize to false:
        for n in self.Nonterminals:
            terminates[n] = False

        # Then propagate termination until no change:
        while True:
            some_change = False
            for (n, pl) in self.Prodnames.items():
                # Nonterminal n terminates iff any of its productions terminates.
                for p in pl:
                    # Production p terminates iff all of its rhs symbols terminate.
                    for s in p.prod:
                        if not terminates[s]:
                            # The symbol s does not terminate,
                            # so production p does not terminate.
                            p_terminates = False
                            break
                    else:
                        # didn't break from the loop,
                        # so every symbol s terminates
                        # so production p terminates.
                        p_terminates = True

                    if p_terminates:
                        # symbol n terminates!
                        if not terminates[n]:
                            terminates[n] = True
                            some_change = True
                        # Don't need to consider any more productions for this n.
                        break

            if not some_change:
                break

        infinite = []
        for (s, term) in terminates.items():
            if not term:
                if s not in self.Prodnames and s not in self.Terminals and s != 'error':
                    # s is used-but-not-defined, and we've already warned of that,
                    # so it would be overkill to say that it's also non-terminating.
                    pass
                else:
                    infinite.append(s)

        return infinite

    # -----------------------------------------------------------------------------
    # undefined_symbols()
    #
    # Find all symbols that were used the grammar, but not defined as tokens or
    # grammar rules.  Returns a list of tuples (sym, prod) where sym in the symbol
    # and prod is the production where the symbol was used.
    # -----------------------------------------------------------------------------
    def undefined_symbols(self):
        result = []
        for p in self.Productions:
            if not p:
                continue

            for s in p.prod:
                if s not in self.Prodnames and s not in self.Terminals and s != 'error':
                    result.append((s, p))
        return result

    # -----------------------------------------------------------------------------
    # unused_terminals()
    #
    # Find all terminals that were defined, but not used by the grammar.  Returns
    # a list of all symbols.
    # -----------------------------------------------------------------------------
    def unused_terminals(self):
        unused_tok = []
        for s, v in self.Terminals.items():
            if s != 'error' and not v:
                unused_tok.append(s)

        return unused_tok

    # ------------------------------------------------------------------------------
    # unused_rules()
    #
    # Find all grammar rules that were defined,  but not used (maybe not reachable)
    # Returns a list of productions.
    # ------------------------------------------------------------------------------

    def unused_rules(self):
        unused_prod = []
        for s, v in self.Nonterminals.items():
            if not v:
                p = self.Prodnames[s][0]
                unused_prod.append(p)
        return unused_prod

    # -----------------------------------------------------------------------------
    # unused_precedence()
    #
    # Returns a list of tuples (term,precedence) corresponding to precedence
    # rules that were never used by the grammar.  term is the name of the terminal
    # on which precedence was applied and precedence is a string such as 'left' or
    # 'right' corresponding to the type of precedence.
    # -----------------------------------------------------------------------------

    def unused_precedence(self):
        unused = []
        for termname in self.Precedence:
            if not (termname in self.Terminals or termname in self.UsedPrecedence):
                unused.append((termname, self.Precedence[termname][0]))

        return unused

    # -------------------------------------------------------------------------
    # _first()
    #
    # Compute the value of FIRST1(beta) where beta is a tuple of symbols.
    #
    # During execution of compute_first1, the result may be incomplete.
    # Afterward (e.g., when called from compute_follow()), it will be complete.
    # -------------------------------------------------------------------------
    def _first(self, beta):

        # We are computing First(x1,x2,x3,...,xn)
        result = []
        for x in beta:
            x_produces_empty = False

            # Add all the non-<empty> symbols of First[x] to the result.
            for f in self.First[x]:
                if f == '<empty>':
                    x_produces_empty = True
                else:
                    if f not in result:
                        result.append(f)

            if x_produces_empty:
                # We have to consider the next x in beta,
                # i.e. stay in the loop.
                pass
            else:
                # We don't have to consider any further symbols in beta.
                break
        else:
            # There was no 'break' from the loop,
            # so x_produces_empty was true for all x in beta,
            # so beta produces empty as well.
            result.append('<empty>')

        return result

    # -------------------------------------------------------------------------
    # compute_first()
    #
    # Compute the value of FIRST1(X) for all symbols
    # -------------------------------------------------------------------------
    def compute_first(self):
        if self.First:
            return self.First

        # Terminals:
        for t in self.Terminals:
            self.First[t] = [t]

        self.First['$end'] = ['$end']

        # Nonterminals:

        # Initialize to the empty set:
        for n in self.Nonterminals:
            self.First[n] = []

        # Then propagate symbols until no change:
        while True:
            some_change = False
            for n in self.Nonterminals:
                for p in self.Prodnames[n]:
                    for f in self._first(p.prod):
                        if f not in self.First[n]:
                            self.First[n].append(f)
                            some_change = True
            if not some_change:
                break

        return self.First

    # ---------------------------------------------------------------------
    # compute_follow()
    #
    # Computes all of the follow sets for every non-terminal symbol.  The
    # follow set is the set of all symbols that might follow a given
    # non-terminal.  See the Dragon book, 2nd Ed. p. 189.
    # ---------------------------------------------------------------------
    def compute_follow(self, start=None):
        # If already computed, return the result
        if self.Follow:
            return self.Follow

        # If first sets not computed yet, do that first.
        if not self.First:
            self.compute_first()

        # Add '$end' to the follow list of the start symbol
        for k in self.Nonterminals:
            self.Follow[k] = []

        if not start:
            start = self.Productions[1].name

        self.Follow[start] = ['$end']

        while True:
            didadd = False
            for p in self.Productions[1:]:
                # Here is the production set
                for i, B in enumerate(p.prod):
                    if B in self.Nonterminals:
                        # Okay. We got a non-terminal in a production
                        fst = self._first(p.prod[i+1:])
                        hasempty = False
                        for f in fst:
                            if f != '<empty>' and f not in self.Follow[B]:
                                self.Follow[B].append(f)
                                didadd = True
                            if f == '<empty>':
                                hasempty = True
                        if hasempty or i == (len(p.prod)-1):
                            # Add elements of follow(a) to follow(b)
                            for f in self.Follow[p.name]:
                                if f not in self.Follow[B]:
                                    self.Follow[B].append(f)
                                    didadd = True
            if not didadd:
                break
        return self.Follow


    # -----------------------------------------------------------------------------
    # build_lritems()
    #
    # This function walks the list of productions and builds a complete set of the
    # LR items.  The LR items are stored in two ways:  First, they are uniquely
    # numbered and placed in the list _lritems.  Second, a linked list of LR items
    # is built for each production.  For example:
    #
    #   E -> E PLUS E
    #
    # Creates the list
    #
    #  [E -> . E PLUS E, E -> E . PLUS E, E -> E PLUS . E, E -> E PLUS E . ]
    # -----------------------------------------------------------------------------

    def build_lritems(self):
        for p in self.Productions:
            lastlri = p
            i = 0
            lr_items = []
            while True:
                if i > len(p):
                    lri = None
                else:
                    lri = LRItem(p, i)
                    # Precompute the list of productions immediately following
                    try:
                        lri.lr_after = self.Prodnames[lri.prod[i+1]]
                    except (IndexError, KeyError):
                        lri.lr_after = []
                    try:
                        lri.lr_before = lri.prod[i-1]
                    except IndexError:
                        lri.lr_before = None

                lastlri.lr_next = lri
                if not lri:
                    break
                lr_items.append(lri)
                lastlri = lri
                i += 1
            p.lr_items = lr_items

# -----------------------------------------------------------------------------
#                            == Class LRTable ==
#
# This basic class represents a basic table of LR parsing information.
# Methods for generating the tables are not defined here.  They are defined
# in the derived class LRGeneratedTable.
# -----------------------------------------------------------------------------

class VersionError(YaccError):
    pass

class LRTable(object):
    def __init__(self):
        self.lr_action = None
        self.lr_goto = None
        self.lr_productions = None
        self.lr_method = None

    def read_table(self, module):
        if isinstance(module, types.ModuleType):
            parsetab = module
        else:
            exec('import %s' % module)
            parsetab = sys.modules[module]

        if parsetab._tabversion != __tabversion__:
            raise VersionError('yacc table file version is out of date')

        self.lr_action = parsetab._lr_action
        self.lr_goto = parsetab._lr_goto

        self.lr_productions = []
        for p in parsetab._lr_productions:
            self.lr_productions.append(MiniProduction(*p))

        self.lr_method = parsetab._lr_method
        return parsetab._lr_signature

    def read_pickle(self, filename):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        if not os.path.exists(filename):
          raise ImportError

        in_f = open(filename, 'rb')

        tabversion = pickle.load(in_f)
        if tabversion != __tabversion__:
            raise VersionError('yacc table file version is out of date')
        self.lr_method = pickle.load(in_f)
        signature      = pickle.load(in_f)
        self.lr_action = pickle.load(in_f)
        self.lr_goto   = pickle.load(in_f)
        productions    = pickle.load(in_f)

        self.lr_productions = []
        for p in productions:
            self.lr_productions.append(MiniProduction(*p))

        in_f.close()
        return signature

    # Bind all production function names to callable objects in pdict
    def bind_callables(self, pdict):
        for p in self.lr_productions:
            p.bind(pdict)


# -----------------------------------------------------------------------------
#                           === LR Generator ===
#
# The following classes and functions are used to generate LR parsing tables on
# a grammar.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# digraph()
# traverse()
#
# The following two functions are used to compute set valued functions
# of the form:
#
#     F(x) = F'(x) U U{F(y) | x R y}
#
# This is used to compute the values of Read() sets as well as FOLLOW sets
# in LALR(1) generation.
#
# Inputs:  X    - An input set
#          R    - A relation
#          FP   - Set-valued function
# ------------------------------------------------------------------------------

def digraph(X, R, FP):
    N = {}
    for x in X:
        N[x] = 0
    stack = []
    F = {}
    for x in X:
        if N[x] == 0:
            traverse(x, N, stack, F, X, R, FP)
    return F

def traverse(x, N, stack, F, X, R, FP):
    stack.append(x)
    d = len(stack)
    N[x] = d
    F[x] = FP(x)             # F(X) <- F'(x)

    rel = R(x)               # Get y's related to x
    for y in rel:
        if N[y] == 0:
            traverse(y, N, stack, F, X, R, FP)
        N[x] = min(N[x], N[y])
        for a in F.get(y, []):
            if a not in F[x]:
                F[x].append(a)
    if N[x] == d:
        N[stack[-1]] = MAXINT
        F[stack[-1]] = F[x]
        element = stack.pop()
        while element != x:
            N[stack[-1]] = MAXINT
            F[stack[-1]] = F[x]
            element = stack.pop()

class LALRError(YaccError):
    pass

# -----------------------------------------------------------------------------
#                             == LRGeneratedTable ==
#
# This class implements the LR table generation algorithm.  There are no
# public methods except for write()
# -----------------------------------------------------------------------------

class LRGeneratedTable(LRTable):
    def __init__(self, grammar, method='LALR', log=None):
        if method not in ['SLR', 'LALR']:
            raise LALRError('Unsupported method %s' % method)

        self.grammar = grammar
        self.lr_method = method

        # Set up the logger
        if not log:
            log = NullLogger()
        self.log = log

        # Internal attributes
        self.lr_action     = {}        # Action table
        self.lr_goto       = {}        # Goto table
        self.lr_productions  = grammar.Productions    # Copy of grammar Production array
        self.lr_goto_cache = {}        # Cache of computed gotos
        self.lr0_cidhash   = {}        # Cache of closures

        self._add_count    = 0         # Internal counter used to detect cycles

        # Diagonistic information filled in by the table generator
        self.sr_conflict   = 0
        self.rr_conflict   = 0
        self.conflicts     = []        # List of conflicts

        self.sr_conflicts  = []
        self.rr_conflicts  = []

        # Build the tables
        self.grammar.build_lritems()
        self.grammar.compute_first()
        self.grammar.compute_follow()
        self.lr_parse_table()

    # Compute the LR(0) closure operation on I, where I is a set of LR(0) items.

    def lr0_closure(self, I):
        self._add_count += 1

        # Add everything in I to J
        J = I[:]
        didadd = True
        while didadd:
            didadd = False
            for j in J:
                for x in j.lr_after:
                    if getattr(x, 'lr0_added', 0) == self._add_count:
                        continue
                    # Add B --> .G to J
                    J.append(x.lr_next)
                    x.lr0_added = self._add_count
                    didadd = True

        return J

    # Compute the LR(0) goto function goto(I,X) where I is a set
    # of LR(0) items and X is a grammar symbol.   This function is written
    # in a way that guarantees uniqueness of the generated goto sets
    # (i.e. the same goto set will never be returned as two different Python
    # objects).  With uniqueness, we can later do fast set comparisons using
    # id(obj) instead of element-wise comparison.

    def lr0_goto(self, I, x):
        # First we look for a previously cached entry
        g = self.lr_goto_cache.get((id(I), x))
        if g:
            return g

        # Now we generate the goto set in a way that guarantees uniqueness
        # of the result

        s = self.lr_goto_cache.get(x)
        if not s:
            s = {}
            self.lr_goto_cache[x] = s

        gs = []
        for p in I:
            n = p.lr_next
            if n and n.lr_before == x:
                s1 = s.get(id(n))
                if not s1:
                    s1 = {}
                    s[id(n)] = s1
                gs.append(n)
                s = s1
        g = s.get('$end')
        if not g:
            if gs:
                g = self.lr0_closure(gs)
                s['$end'] = g
            else:
                s['$end'] = gs
        self.lr_goto_cache[(id(I), x)] = g
        return g

    # Compute the LR(0) sets of item function
    def lr0_items(self):
        C = [self.lr0_closure([self.grammar.Productions[0].lr_next])]
        i = 0
        for I in C:
            self.lr0_cidhash[id(I)] = i
            i += 1

        # Loop over the items in C and each grammar symbols
        i = 0
        while i < len(C):
            I = C[i]
            i += 1

            # Collect all of the symbols that could possibly be in the goto(I,X) sets
            asyms = {}
            for ii in I:
                for s in ii.usyms:
                    asyms[s] = None

            for x in asyms:
                g = self.lr0_goto(I, x)
                if not g or id(g) in self.lr0_cidhash:
                    continue
                self.lr0_cidhash[id(g)] = len(C)
                C.append(g)

        return C

    # -----------------------------------------------------------------------------
    #                       ==== LALR(1) Parsing ====
    #
    # LALR(1) parsing is almost exactly the same as SLR except that instead of
    # relying upon Follow() sets when performing reductions, a more selective
    # lookahead set that incorporates the state of the LR(0) machine is utilized.
    # Thus, we mainly just have to focus on calculating the lookahead sets.
    #
    # The method used here is due to DeRemer and Pennelo (1982).
    #
    # DeRemer, F. L., and T. J. Pennelo: "Efficient Computation of LALR(1)
    #     Lookahead Sets", ACM Transactions on Programming Languages and Systems,
    #     Vol. 4, No. 4, Oct. 1982, pp. 615-649
    #
    # Further details can also be found in:
    #
    #  J. Tremblay and P. Sorenson, "The Theory and Practice of Compiler Writing",
    #      McGraw-Hill Book Company, (1985).
    #
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # compute_nullable_nonterminals()
    #
    # Creates a dictionary containing all of the non-terminals that might produce
    # an empty production.
    # -----------------------------------------------------------------------------

    def compute_nullable_nonterminals(self):
        nullable = set()
        num_nullable = 0
        while True:
            for p in self.grammar.Productions[1:]:
                if p.len == 0:
                    nullable.add(p.name)
                    continue
                for t in p.prod:
                    if t not in nullable:
                        break
                else:
                    nullable.add(p.name)
            if len(nullable) == num_nullable:
                break
            num_nullable = len(nullable)
        return nullable

    # -----------------------------------------------------------------------------
    # find_nonterminal_trans(C)
    #
    # Given a set of LR(0) items, this functions finds all of the non-terminal
    # transitions.    These are transitions in which a dot appears immediately before
    # a non-terminal.   Returns a list of tuples of the form (state,N) where state
    # is the state number and N is the nonterminal symbol.
    #
    # The input C is the set of LR(0) items.
    # -----------------------------------------------------------------------------

    def find_nonterminal_transitions(self, C):
        trans = []
        for stateno, state in enumerate(C):
            for p in state:
                if p.lr_index < p.len - 1:
                    t = (stateno, p.prod[p.lr_index+1])
                    if t[1] in self.grammar.Nonterminals:
                        if t not in trans:
                            trans.append(t)
        return trans

    # -----------------------------------------------------------------------------
    # dr_relation()
    #
    # Computes the DR(p,A) relationships for non-terminal transitions.  The input
    # is a tuple (state,N) where state is a number and N is a nonterminal symbol.
    #
    # Returns a list of terminals.
    # -----------------------------------------------------------------------------

    def dr_relation(self, C, trans, nullable):
        dr_set = {}
        state, N = trans
        terms = []

        g = self.lr0_goto(C[state], N)
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index+1]
                if a in self.grammar.Terminals:
                    if a not in terms:
                        terms.append(a)

        # This extra bit is to handle the start state
        if state == 0 and N == self.grammar.Productions[0].prod[0]:
            terms.append('$end')

        return terms

    # -----------------------------------------------------------------------------
    # reads_relation()
    #
    # Computes the READS() relation (p,A) READS (t,C).
    # -----------------------------------------------------------------------------

    def reads_relation(self, C, trans, empty):
        # Look for empty transitions
        rel = []
        state, N = trans

        g = self.lr0_goto(C[state], N)
        j = self.lr0_cidhash.get(id(g), -1)
        for p in g:
            if p.lr_index < p.len - 1:
                a = p.prod[p.lr_index + 1]
                if a in empty:
                    rel.append((j, a))

        return rel

    # -----------------------------------------------------------------------------
    # compute_lookback_includes()
    #
    # Determines the lookback and includes relations
    #
    # LOOKBACK:
    #
    # This relation is determined by running the LR(0) state machine forward.
    # For example, starting with a production "N : . A B C", we run it forward
    # to obtain "N : A B C ."   We then build a relationship between this final
    # state and the starting state.   These relationships are stored in a dictionary
    # lookdict.
    #
    # INCLUDES:
    #
    # Computes the INCLUDE() relation (p,A) INCLUDES (p',B).
    #
    # This relation is used to determine non-terminal transitions that occur
    # inside of other non-terminal transition states.   (p,A) INCLUDES (p', B)
    # if the following holds:
    #
    #       B -> LAT, where T -> epsilon and p' -L-> p
    #
    # L is essentially a prefix (which may be empty), T is a suffix that must be
    # able to derive an empty string.  State p' must lead to state p with the string L.
    #
    # -----------------------------------------------------------------------------

    def compute_lookback_includes(self, C, trans, nullable):
        lookdict = {}          # Dictionary of lookback relations
        includedict = {}       # Dictionary of include relations

        # Make a dictionary of non-terminal transitions
        dtrans = {}
        for t in trans:
            dtrans[t] = 1

        # Loop over all transitions and compute lookbacks and includes
        for state, N in trans:
            lookb = []
            includes = []
            for p in C[state]:
                if p.name != N:
                    continue

                # Okay, we have a name match.  We now follow the production all the way
                # through the state machine until we get the . on the right hand side

                lr_index = p.lr_index
                j = state
                while lr_index < p.len - 1:
                    lr_index = lr_index + 1
                    t = p.prod[lr_index]

                    # Check to see if this symbol and state are a non-terminal transition
                    if (j, t) in dtrans:
                        # Yes.  Okay, there is some chance that this is an includes relation
                        # the only way to know for certain is whether the rest of the
                        # production derives empty

                        li = lr_index + 1
                        while li < p.len:
                            if p.prod[li] in self.grammar.Terminals:
                                break      # No forget it
                            if p.prod[li] not in nullable:
                                break
                            li = li + 1
                        else:
                            # Appears to be a relation between (j,t) and (state,N)
                            includes.append((j, t))

                    g = self.lr0_goto(C[j], t)               # Go to next set
                    j = self.lr0_cidhash.get(id(g), -1)      # Go to next state

                # When we get here, j is the final state, now we have to locate the production
                for r in C[j]:
                    if r.name != p.name:
                        continue
                    if r.len != p.len:
                        continue
                    i = 0
                    # This look is comparing a production ". A B C" with "A B C ."
                    while i < r.lr_index:
                        if r.prod[i] != p.prod[i+1]:
                            break
                        i = i + 1
                    else:
                        lookb.append((j, r))
            for i in includes:
                if i not in includedict:
                    includedict[i] = []
                includedict[i].append((state, N))
            lookdict[(state, N)] = lookb

        return lookdict, includedict

    # -----------------------------------------------------------------------------
    # compute_read_sets()
    #
    # Given a set of LR(0) items, this function computes the read sets.
    #
    # Inputs:  C        =  Set of LR(0) items
    #          ntrans   = Set of nonterminal transitions
    #          nullable = Set of empty transitions
    #
    # Returns a set containing the read sets
    # -----------------------------------------------------------------------------

    def compute_read_sets(self, C, ntrans, nullable):
        FP = lambda x: self.dr_relation(C, x, nullable)
        R =  lambda x: self.reads_relation(C, x, nullable)
        F = digraph(ntrans, R, FP)
        return F

    # -----------------------------------------------------------------------------
    # compute_follow_sets()
    #
    # Given a set of LR(0) items, a set of non-terminal transitions, a readset,
    # and an include set, this function computes the follow sets
    #
    # Follow(p,A) = Read(p,A) U U {Follow(p',B) | (p,A) INCLUDES (p',B)}
    #
    # Inputs:
    #            ntrans     = Set of nonterminal transitions
    #            readsets   = Readset (previously computed)
    #            inclsets   = Include sets (previously computed)
    #
    # Returns a set containing the follow sets
    # -----------------------------------------------------------------------------

    def compute_follow_sets(self, ntrans, readsets, inclsets):
        FP = lambda x: readsets[x]
        R  = lambda x: inclsets.get(x, [])
        F = digraph(ntrans, R, FP)
        return F

    # -----------------------------------------------------------------------------
    # add_lookaheads()
    #
    # Attaches the lookahead symbols to grammar rules.
    #
    # Inputs:    lookbacks         -  Set of lookback relations
    #            followset         -  Computed follow set
    #
    # This function directly attaches the lookaheads to productions contained
    # in the lookbacks set
    # -----------------------------------------------------------------------------

    def add_lookaheads(self, lookbacks, followset):
        for trans, lb in lookbacks.items():
            # Loop over productions in lookback
            for state, p in lb:
                if state not in p.lookaheads:
                    p.lookaheads[state] = []
                f = followset.get(trans, [])
                for a in f:
                    if a not in p.lookaheads[state]:
                        p.lookaheads[state].append(a)

    # -----------------------------------------------------------------------------
    # add_lalr_lookaheads()
    #
    # This function does all of the work of adding lookahead information for use
    # with LALR parsing
    # -----------------------------------------------------------------------------

    def add_lalr_lookaheads(self, C):
        # Determine all of the nullable nonterminals
        nullable = self.compute_nullable_nonterminals()

        # Find all non-terminal transitions
        trans = self.find_nonterminal_transitions(C)

        # Compute read sets
        readsets = self.compute_read_sets(C, trans, nullable)

        # Compute lookback/includes relations
        lookd, included = self.compute_lookback_includes(C, trans, nullable)

        # Compute LALR FOLLOW sets
        followsets = self.compute_follow_sets(trans, readsets, included)

        # Add all of the lookaheads
        self.add_lookaheads(lookd, followsets)

    # -----------------------------------------------------------------------------
    # lr_parse_table()
    #
    # This function constructs the parse tables for SLR or LALR
    # -----------------------------------------------------------------------------
    def lr_parse_table(self):
        Productions = self.grammar.Productions
        Precedence  = self.grammar.Precedence
        goto   = self.lr_goto         # Goto array
        action = self.lr_action       # Action array
        log    = self.log             # Logger for output

        actionp = {}                  # Action production array (temporary)

        log.info('Parsing method: %s', self.lr_method)

        # Step 1: Construct C = { I0, I1, ... IN}, collection of LR(0) items
        # This determines the number of states

        C = self.lr0_items()

        if self.lr_method == 'LALR':
            self.add_lalr_lookaheads(C)

        # Build the parser table, state by state
        st = 0
        for I in C:
            # Loop over each production in I
            actlist = []              # List of actions
            st_action  = {}
            st_actionp = {}
            st_goto    = {}
            log.info('')
            log.info('state %d', st)
            log.info('')
            for p in I:
                log.info('    (%d) %s', p.number, p)
            log.info('')

            for p in I:
                    if p.len == p.lr_index + 1:
                        if p.name == "S'":
                            # Start symbol. Accept!
                            st_action['$end'] = 0
                            st_actionp['$end'] = p
                        else:
                            # We are at the end of a production.  Reduce!
                            if self.lr_method == 'LALR':
                                laheads = p.lookaheads[st]
                            else:
                                laheads = self.grammar.Follow[p.name]
                            for a in laheads:
                                actlist.append((a, p, 'reduce using rule %d (%s)' % (p.number, p)))
                                r = st_action.get(a)
                                if r is not None:
                                    # Whoa. Have a shift/reduce or reduce/reduce conflict
                                    if r > 0:
                                        # Need to decide on shift or reduce here
                                        # By default we favor shifting. Need to add
                                        # some precedence rules here.

                                        # Shift precedence comes from the token
                                        sprec, slevel = Precedence.get(a, ('right', 0))

                                        # Reduce precedence comes from rule being reduced (p)
                                        rprec, rlevel = Productions[p.number].prec

                                        if (slevel < rlevel) or ((slevel == rlevel) and (rprec == 'left')):
                                            # We really need to reduce here.
                                            st_action[a] = -p.number
                                            st_actionp[a] = p
                                            if not slevel and not rlevel:
                                                log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                                self.sr_conflicts.append((st, a, 'reduce'))
                                            Productions[p.number].reduced += 1
                                        elif (slevel == rlevel) and (rprec == 'nonassoc'):
                                            st_action[a] = None
                                        else:
                                            # Hmmm. Guess we'll keep the shift
                                            if not rlevel:
                                                log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                                self.sr_conflicts.append((st, a, 'shift'))
                                    elif r < 0:
                                        # Reduce/reduce conflict.   In this case, we favor the rule
                                        # that was defined first in the grammar file
                                        oldp = Productions[-r]
                                        pp = Productions[p.number]
                                        if oldp.line > pp.line:
                                            st_action[a] = -p.number
                                            st_actionp[a] = p
                                            chosenp, rejectp = pp, oldp
                                            Productions[p.number].reduced += 1
                                            Productions[oldp.number].reduced -= 1
                                        else:
                                            chosenp, rejectp = oldp, pp
                                        self.rr_conflicts.append((st, chosenp, rejectp))
                                        log.info('  ! reduce/reduce conflict for %s resolved using rule %d (%s)',
                                                 a, st_actionp[a].number, st_actionp[a])
                                    else:
                                        raise LALRError('Unknown conflict in state %d' % st)
                                else:
                                    st_action[a] = -p.number
                                    st_actionp[a] = p
                                    Productions[p.number].reduced += 1
                    else:
                        i = p.lr_index
                        a = p.prod[i+1]       # Get symbol right after the "."
                        if a in self.grammar.Terminals:
                            g = self.lr0_goto(I, a)
                            j = self.lr0_cidhash.get(id(g), -1)
                            if j >= 0:
                                # We are in a shift state
                                actlist.append((a, p, 'shift and go to state %d' % j))
                                r = st_action.get(a)
                                if r is not None:
                                    # Whoa have a shift/reduce or shift/shift conflict
                                    if r > 0:
                                        if r != j:
                                            raise LALRError('Shift/shift conflict in state %d' % st)
                                    elif r < 0:
                                        # Do a precedence check.
                                        #   -  if precedence of reduce rule is higher, we reduce.
                                        #   -  if precedence of reduce is same and left assoc, we reduce.
                                        #   -  otherwise we shift

                                        # Shift precedence comes from the token
                                        sprec, slevel = Precedence.get(a, ('right', 0))

                                        # Reduce precedence comes from the rule that could have been reduced
                                        rprec, rlevel = Productions[st_actionp[a].number].prec

                                        if (slevel > rlevel) or ((slevel == rlevel) and (rprec == 'right')):
                                            # We decide to shift here... highest precedence to shift
                                            Productions[st_actionp[a].number].reduced -= 1
                                            st_action[a] = j
                                            st_actionp[a] = p
                                            if not rlevel:
                                                log.info('  ! shift/reduce conflict for %s resolved as shift', a)
                                                self.sr_conflicts.append((st, a, 'shift'))
                                        elif (slevel == rlevel) and (rprec == 'nonassoc'):
                                            st_action[a] = None
                                        else:
                                            # Hmmm. Guess we'll keep the reduce
                                            if not slevel and not rlevel:
                                                log.info('  ! shift/reduce conflict for %s resolved as reduce', a)
                                                self.sr_conflicts.append((st, a, 'reduce'))

                                    else:
                                        raise LALRError('Unknown conflict in state %d' % st)
                                else:
                                    st_action[a] = j
                                    st_actionp[a] = p

            # Print the actions associated with each terminal
            _actprint = {}
            for a, p, m in actlist:
                if a in st_action:
                    if p is st_actionp[a]:
                        log.info('    %-15s %s', a, m)
                        _actprint[(a, m)] = 1
            log.info('')
            # Print the actions that were not used. (debugging)
            not_used = 0
            for a, p, m in actlist:
                if a in st_action:
                    if p is not st_actionp[a]:
                        if not (a, m) in _actprint:
                            log.debug('  ! %-15s [ %s ]', a, m)
                            not_used = 1
                            _actprint[(a, m)] = 1
            if not_used:
                log.debug('')

            # Construct the goto table for this state

            nkeys = {}
            for ii in I:
                for s in ii.usyms:
                    if s in self.grammar.Nonterminals:
                        nkeys[s] = None
            for n in nkeys:
                g = self.lr0_goto(I, n)
                j = self.lr0_cidhash.get(id(g), -1)
                if j >= 0:
                    st_goto[n] = j
                    log.info('    %-30s shift and go to state %d', n, j)

            action[st] = st_action
            actionp[st] = st_actionp
            goto[st] = st_goto
            st += 1

    # -----------------------------------------------------------------------------
    # write()
    #
    # This function writes the LR parsing tables to a file
    # -----------------------------------------------------------------------------

    def write_table(self, tabmodule, outputdir='', signature=''):
        if isinstance(tabmodule, types.ModuleType):
            raise IOError("Won't overwrite existing tabmodule")

        basemodulename = tabmodule.split('.')[-1]
        filename = os.path.join(outputdir, basemodulename) + '.py'
        try:
            f = open(filename, 'w')

            f.write('''
# %s
# This file is automatically generated. Do not edit.
_tabversion = %r

_lr_method = %r

_lr_signature = %r
    ''' % (os.path.basename(filename), __tabversion__, self.lr_method, signature))

            # Change smaller to 0 to go back to original tables
            smaller = 1

            # Factor out names to try and make smaller
            if smaller:
                items = {}

                for s, nd in self.lr_action.items():
                    for name, v in nd.items():
                        i = items.get(name)
                        if not i:
                            i = ([], [])
                            items[name] = i
                        i[0].append(s)
                        i[1].append(v)

                f.write('\n_lr_action_items = {')
                for k, v in items.items():
                    f.write('%r:([' % k)
                    for i in v[0]:
                        f.write('%r,' % i)
                    f.write('],[')
                    for i in v[1]:
                        f.write('%r,' % i)

                    f.write(']),')
                f.write('}\n')

                f.write('''
_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items
''')

            else:
                f.write('\n_lr_action = { ')
                for k, v in self.lr_action.items():
                    f.write('(%r,%r):%r,' % (k[0], k[1], v))
                f.write('}\n')

            if smaller:
                # Factor out names to try and make smaller
                items = {}

                for s, nd in self.lr_goto.items():
                    for name, v in nd.items():
                        i = items.get(name)
                        if not i:
                            i = ([], [])
                            items[name] = i
                        i[0].append(s)
                        i[1].append(v)

                f.write('\n_lr_goto_items = {')
                for k, v in items.items():
                    f.write('%r:([' % k)
                    for i in v[0]:
                        f.write('%r,' % i)
                    f.write('],[')
                    for i in v[1]:
                        f.write('%r,' % i)

                    f.write(']),')
                f.write('}\n')

                f.write('''
_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
''')
            else:
                f.write('\n_lr_goto = { ')
                for k, v in self.lr_goto.items():
                    f.write('(%r,%r):%r,' % (k[0], k[1], v))
                f.write('}\n')

            # Write production table
            f.write('_lr_productions = [\n')
            for p in self.lr_productions:
                if p.func:
                    f.write('  (%r,%r,%d,%r,%r,%d),\n' % (p.str, p.name, p.len,
                                                          p.func, os.path.basename(p.file), p.line))
                else:
                    f.write('  (%r,%r,%d,None,None,None),\n' % (str(p), p.name, p.len))
            f.write(']\n')
            f.close()

        except IOError as e:
            raise


    # -----------------------------------------------------------------------------
    # pickle_table()
    #
    # This function pickles the LR parsing tables to a supplied file object
    # -----------------------------------------------------------------------------

    def pickle_table(self, filename, signature=''):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        with open(filename, 'wb') as outf:
            pickle.dump(__tabversion__, outf, pickle_protocol)
            pickle.dump(self.lr_method, outf, pickle_protocol)
            pickle.dump(signature, outf, pickle_protocol)
            pickle.dump(self.lr_action, outf, pickle_protocol)
            pickle.dump(self.lr_goto, outf, pickle_protocol)

            outp = []
            for p in self.lr_productions:
                if p.func:
                    outp.append((p.str, p.name, p.len, p.func, os.path.basename(p.file), p.line))
                else:
                    outp.append((str(p), p.name, p.len, None, None, None))
            pickle.dump(outp, outf, pickle_protocol)

# -----------------------------------------------------------------------------
#                            === INTROSPECTION ===
#
# The following functions and classes are used to implement the PLY
# introspection features followed by the yacc() function itself.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# get_caller_module_dict()
#
# This function returns a dictionary containing all of the symbols defined within
# a caller further down the call stack.  This is used to get the environment
# associated with the yacc() call if none was provided.
# -----------------------------------------------------------------------------

def get_caller_module_dict(levels):
    f = sys._getframe(levels)
    ldict = f.f_globals.copy()
    if f.f_globals != f.f_locals:
        ldict.update(f.f_locals)
    return ldict

# -----------------------------------------------------------------------------
# parse_grammar()
#
# This takes a raw grammar rule string and parses it into production data
# -----------------------------------------------------------------------------
def parse_grammar(doc, file, line):
    grammar = []
    # Split the doc string into lines
    pstrings = doc.splitlines()
    lastp = None
    dline = line
    for ps in pstrings:
        dline += 1
        p = ps.split()
        if not p:
            continue
        try:
            if p[0] == '|':
                # This is a continuation of a previous rule
                if not lastp:
                    raise SyntaxError("%s:%d: Misplaced '|'" % (file, dline))
                prodname = lastp
                syms = p[1:]
            else:
                prodname = p[0]
                lastp = prodname
                syms   = p[2:]
                assign = p[1]
                if assign != ':' and assign != '::=':
                    raise SyntaxError("%s:%d: Syntax error. Expected ':'" % (file, dline))

            grammar.append((file, dline, prodname, syms))
        except SyntaxError:
            raise
        except Exception:
            raise SyntaxError('%s:%d: Syntax error in rule %r' % (file, dline, ps.strip()))

    return grammar

# -----------------------------------------------------------------------------
# ParserReflect()
#
# This class represents information extracted for building a parser including
# start symbol, error function, tokens, precedence list, action functions,
# etc.
# -----------------------------------------------------------------------------
class ParserReflect(object):
    def __init__(self, pdict, log=None):
        self.pdict      = pdict
        self.start      = None
        self.error_func = None
        self.tokens     = None
        self.modules    = set()
        self.grammar    = []
        self.error      = False

        if log is None:
            self.log = PlyLogger(sys.stderr)
        else:
            self.log = log

    # Get all of the basic information
    def get_all(self):
        self.get_start()
        self.get_error_func()
        self.get_tokens()
        self.get_precedence()
        self.get_pfunctions()

    # Validate all of the information
    def validate_all(self):
        self.validate_start()
        self.validate_error_func()
        self.validate_tokens()
        self.validate_precedence()
        self.validate_pfunctions()
        self.validate_modules()
        return self.error

    # Compute a signature over the grammar
    def signature(self):
        parts = []
        try:
            if self.start:
                parts.append(self.start)
            if self.prec:
                parts.append(''.join([''.join(p) for p in self.prec]))
            if self.tokens:
                parts.append(' '.join(self.tokens))
            for f in self.pfuncs:
                if f[3]:
                    parts.append(f[3])
        except (TypeError, ValueError):
            pass
        return ''.join(parts)

    # -----------------------------------------------------------------------------
    # validate_modules()
    #
    # This method checks to see if there are duplicated p_rulename() functions
    # in the parser module file.  Without this function, it is really easy for
    # users to make mistakes by cutting and pasting code fragments (and it's a real
    # bugger to try and figure out why the resulting parser doesn't work).  Therefore,
    # we just do a little regular expression pattern matching of def statements
    # to try and detect duplicates.
    # -----------------------------------------------------------------------------

    def validate_modules(self):
        # Match def p_funcname(
        fre = re.compile(r'\s*def\s+(p_[a-zA-Z_0-9]*)\(')

        for module in self.modules:
            try:
                lines, linen = inspect.getsourcelines(module)
            except IOError:
                continue

            counthash = {}
            for linen, line in enumerate(lines):
                linen += 1
                m = fre.match(line)
                if m:
                    name = m.group(1)
                    prev = counthash.get(name)
                    if not prev:
                        counthash[name] = linen
                    else:
                        filename = inspect.getsourcefile(module)
                        self.log.warning('%s:%d: Function %s redefined. Previously defined on line %d',
                                         filename, linen, name, prev)

    # Get the start symbol
    def get_start(self):
        self.start = self.pdict.get('start')

    # Validate the start symbol
    def validate_start(self):
        if self.start is not None:
            if not isinstance(self.start, string_types):
                self.log.error("'start' must be a string")

    # Look for error handler
    def get_error_func(self):
        self.error_func = self.pdict.get('p_error')

    # Validate the error function
    def validate_error_func(self):
        if self.error_func:
            if isinstance(self.error_func, types.FunctionType):
                ismethod = 0
            elif isinstance(self.error_func, types.MethodType):
                ismethod = 1
            else:
                self.log.error("'p_error' defined, but is not a function or method")
                self.error = True
                return

            eline = self.error_func.__code__.co_firstlineno
            efile = self.error_func.__code__.co_filename
            module = inspect.getmodule(self.error_func)
            self.modules.add(module)

            argcount = self.error_func.__code__.co_argcount - ismethod
            if argcount != 1:
                self.log.error('%s:%d: p_error() requires 1 argument', efile, eline)
                self.error = True

    # Get the tokens map
    def get_tokens(self):
        tokens = self.pdict.get('tokens')
        if not tokens:
            self.log.error('No token list is defined')
            self.error = True
            return

        if not isinstance(tokens, (list, tuple)):
            self.log.error('tokens must be a list or tuple')
            self.error = True
            return

        if not tokens:
            self.log.error('tokens is empty')
            self.error = True
            return

        self.tokens = tokens

    # Validate the tokens
    def validate_tokens(self):
        # Validate the tokens.
        if 'error' in self.tokens:
            self.log.error("Illegal token name 'error'. Is a reserved word")
            self.error = True
            return

        terminals = set()
        for n in self.tokens:
            if n in terminals:
                self.log.warning('Token %r multiply defined', n)
            terminals.add(n)

    # Get the precedence map (if any)
    def get_precedence(self):
        self.prec = self.pdict.get('precedence')

    # Validate and parse the precedence map
    def validate_precedence(self):
        preclist = []
        if self.prec:
            if not isinstance(self.prec, (list, tuple)):
                self.log.error('precedence must be a list or tuple')
                self.error = True
                return
            for level, p in enumerate(self.prec):
                if not isinstance(p, (list, tuple)):
                    self.log.error('Bad precedence table')
                    self.error = True
                    return

                if len(p) < 2:
                    self.log.error('Malformed precedence entry %s. Must be (assoc, term, ..., term)', p)
                    self.error = True
                    return
                assoc = p[0]
                if not isinstance(assoc, string_types):
                    self.log.error('precedence associativity must be a string')
                    self.error = True
                    return
                for term in p[1:]:
                    if not isinstance(term, string_types):
                        self.log.error('precedence items must be strings')
                        self.error = True
                        return
                    preclist.append((term, assoc, level+1))
        self.preclist = preclist

    # Get all p_functions from the grammar
    def get_pfunctions(self):
        p_functions = []
        for name, item in self.pdict.items():
            if not name.startswith('p_') or name == 'p_error':
                continue
            if isinstance(item, (types.FunctionType, types.MethodType)):
                line = getattr(item, 'co_firstlineno', item.__code__.co_firstlineno)
                module = inspect.getmodule(item)
                p_functions.append((line, module, name, item.__doc__))

        # Sort all of the actions by line number; make sure to stringify
        # modules to make them sortable, since `line` may not uniquely sort all
        # p functions
        p_functions.sort(key=lambda p_function: (
            p_function[0],
            str(p_function[1]),
            p_function[2],
            p_function[3]))
        self.pfuncs = p_functions

    # Validate all of the p_functions
    def validate_pfunctions(self):
        grammar = []
        # Check for non-empty symbols
        if len(self.pfuncs) == 0:
            self.log.error('no rules of the form p_rulename are defined')
            self.error = True
            return

        for line, module, name, doc in self.pfuncs:
            file = inspect.getsourcefile(module)
            func = self.pdict[name]
            if isinstance(func, types.MethodType):
                reqargs = 2
            else:
                reqargs = 1
            if func.__code__.co_argcount > reqargs:
                self.log.error('%s:%d: Rule %r has too many arguments', file, line, func.__name__)
                self.error = True
            elif func.__code__.co_argcount < reqargs:
                self.log.error('%s:%d: Rule %r requires an argument', file, line, func.__name__)
                self.error = True
            elif not func.__doc__:
                self.log.warning('%s:%d: No documentation string specified in function %r (ignored)',
                                 file, line, func.__name__)
            else:
                try:
                    parsed_g = parse_grammar(doc, file, line)
                    for g in parsed_g:
                        grammar.append((name, g))
                except SyntaxError as e:
                    self.log.error(str(e))
                    self.error = True

                # Looks like a valid grammar rule
                # Mark the file in which defined.
                self.modules.add(module)

        # Secondary validation step that looks for p_ definitions that are not functions
        # or functions that look like they might be grammar rules.

        for n, v in self.pdict.items():
            if n.startswith('p_') and isinstance(v, (types.FunctionType, types.MethodType)):
                continue
            if n.startswith('t_'):
                continue
            if n.startswith('p_') and n != 'p_error':
                self.log.warning('%r not defined as a function', n)
            if ((isinstance(v, types.FunctionType) and v.__code__.co_argcount == 1) or
                   (isinstance(v, types.MethodType) and v.__func__.__code__.co_argcount == 2)):
                if v.__doc__:
                    try:
                        doc = v.__doc__.split(' ')
                        if doc[1] == ':':
                            self.log.warning('%s:%d: Possible grammar rule %r defined without p_ prefix',
                                             v.__code__.co_filename, v.__code__.co_firstlineno, n)
                    except IndexError:
                        pass

        self.grammar = grammar

# -----------------------------------------------------------------------------
# yacc(module)
#
# Build a parser
# -----------------------------------------------------------------------------

def yacc(method='LALR', debug=yaccdebug, module=None, tabmodule=tab_module, start=None,
         check_recursion=True, optimize=False, write_tables=True, debugfile=debug_file,
         outputdir=None, debuglog=None, errorlog=None, picklefile=None):

    if tabmodule is None:
        tabmodule = tab_module

    # Reference to the parsing method of the last built parser
    global parse

    # If pickling is enabled, table files are not created
    if picklefile:
        write_tables = 0

    if errorlog is None:
        errorlog = PlyLogger(sys.stderr)

    # Get the module dictionary used for the parser
    if module:
        _items = [(k, getattr(module, k)) for k in dir(module)]
        pdict = dict(_items)
        # If no __file__ attribute is available, try to obtain it from the __module__ instead
        if '__file__' not in pdict:
            pdict['__file__'] = sys.modules[pdict['__module__']].__file__
    else:
        pdict = get_caller_module_dict(2)

    if outputdir is None:
        # If no output directory is set, the location of the output files
        # is determined according to the following rules:
        #     - If tabmodule specifies a package, files go into that package directory
        #     - Otherwise, files go in the same directory as the specifying module
        if isinstance(tabmodule, types.ModuleType):
            srcfile = tabmodule.__file__
        else:
            if '.' not in tabmodule:
                srcfile = pdict['__file__']
            else:
                parts = tabmodule.split('.')
                pkgname = '.'.join(parts[:-1])
                exec('import %s' % pkgname)
                srcfile = getattr(sys.modules[pkgname], '__file__', '')
        outputdir = os.path.dirname(srcfile)

    # Determine if the module is package of a package or not.
    # If so, fix the tabmodule setting so that tables load correctly
    pkg = pdict.get('__package__')
    if pkg and isinstance(tabmodule, str):
        if '.' not in tabmodule:
            tabmodule = pkg + '.' + tabmodule



    # Set start symbol if it's specified directly using an argument
    if start is not None:
        pdict['start'] = start

    # Collect parser information from the dictionary
    pinfo = ParserReflect(pdict, log=errorlog)
    pinfo.get_all()

    if pinfo.error:
        raise YaccError('Unable to build parser')

    # Check signature against table files (if any)
    signature = pinfo.signature()

    # Read the tables
    try:
        lr = LRTable()
        if picklefile:
            read_signature = lr.read_pickle(picklefile)
        else:
            read_signature = lr.read_table(tabmodule)
        if optimize or (read_signature == signature):
            try:
                lr.bind_callables(pinfo.pdict)
                parser = LRParser(lr, pinfo.error_func)
                parse = parser.parse
                return parser
            except Exception as e:
                errorlog.warning('There was a problem loading the table file: %r', e)
    except VersionError as e:
        errorlog.warning(str(e))
    except ImportError:
        pass

    if debuglog is None:
        if debug:
            try:
                debuglog = PlyLogger(open(os.path.join(outputdir, debugfile), 'w'))
            except IOError as e:
                errorlog.warning("Couldn't open %r. %s" % (debugfile, e))
                debuglog = NullLogger()
        else:
            debuglog = NullLogger()

    debuglog.info('Created by PLY version %s (http://www.dabeaz.com/ply)', __version__)

    errors = False

    # Validate the parser information
    if pinfo.validate_all():
        raise YaccError('Unable to build parser')

    if not pinfo.error_func:
        errorlog.warning('no p_error() function is defined')

    # Create a grammar object
    grammar = Grammar(pinfo.tokens)

    # Set precedence level for terminals
    for term, assoc, level in pinfo.preclist:
        try:
            grammar.set_precedence(term, assoc, level)
        except GrammarError as e:
            errorlog.warning('%s', e)

    # Add productions to the grammar
    for funcname, gram in pinfo.grammar:
        file, line, prodname, syms = gram
        try:
            grammar.add_production(prodname, syms, funcname, file, line)
        except GrammarError as e:
            errorlog.error('%s', e)
            errors = True

    # Set the grammar start symbols
    try:
        if start is None:
            grammar.set_start(pinfo.start)
        else:
            grammar.set_start(start)
    except GrammarError as e:
        errorlog.error(str(e))
        errors = True

    if errors:
        raise YaccError('Unable to build parser')

    # Verify the grammar structure
    undefined_symbols = grammar.undefined_symbols()
    for sym, prod in undefined_symbols:
        errorlog.error('%s:%d: Symbol %r used, but not defined as a token or a rule', prod.file, prod.line, sym)
        errors = True

    unused_terminals = grammar.unused_terminals()
    if unused_terminals:
        debuglog.info('')
        debuglog.info('Unused terminals:')
        debuglog.info('')
        for term in unused_terminals:
            errorlog.warning('Token %r defined, but not used', term)
            debuglog.info('    %s', term)

    # Print out all productions to the debug log
    if debug:
        debuglog.info('')
        debuglog.info('Grammar')
        debuglog.info('')
        for n, p in enumerate(grammar.Productions):
            debuglog.info('Rule %-5d %s', n, p)

    # Find unused non-terminals
    unused_rules = grammar.unused_rules()
    for prod in unused_rules:
        errorlog.warning('%s:%d: Rule %r defined, but not used', prod.file, prod.line, prod.name)

    if len(unused_terminals) == 1:
        errorlog.warning('There is 1 unused token')
    if len(unused_terminals) > 1:
        errorlog.warning('There are %d unused tokens', len(unused_terminals))

    if len(unused_rules) == 1:
        errorlog.warning('There is 1 unused rule')
    if len(unused_rules) > 1:
        errorlog.warning('There are %d unused rules', len(unused_rules))

    if debug:
        debuglog.info('')
        debuglog.info('Terminals, with rules where they appear')
        debuglog.info('')
        terms = list(grammar.Terminals)
        terms.sort()
        for term in terms:
            debuglog.info('%-20s : %s', term, ' '.join([str(s) for s in grammar.Terminals[term]]))

        debuglog.info('')
        debuglog.info('Nonterminals, with rules where they appear')
        debuglog.info('')
        nonterms = list(grammar.Nonterminals)
        nonterms.sort()
        for nonterm in nonterms:
            debuglog.info('%-20s : %s', nonterm, ' '.join([str(s) for s in grammar.Nonterminals[nonterm]]))
        debuglog.info('')

    if check_recursion:
        unreachable = grammar.find_unreachable()
        for u in unreachable:
            errorlog.warning('Symbol %r is unreachable', u)

        infinite = grammar.infinite_cycles()
        for inf in infinite:
            errorlog.error('Infinite recursion detected for symbol %r', inf)
            errors = True

    unused_prec = grammar.unused_precedence()
    for term, assoc in unused_prec:
        errorlog.error('Precedence rule %r defined for unknown symbol %r', assoc, term)
        errors = True

    if errors:
        raise YaccError('Unable to build parser')

    # Run the LRGeneratedTable on the grammar
    if debug:
        errorlog.debug('Generating %s tables', method)

    lr = LRGeneratedTable(grammar, method, debuglog)

    if debug:
        num_sr = len(lr.sr_conflicts)

        # Report shift/reduce and reduce/reduce conflicts
        if num_sr == 1:
            errorlog.warning('1 shift/reduce conflict')
        elif num_sr > 1:
            errorlog.warning('%d shift/reduce conflicts', num_sr)

        num_rr = len(lr.rr_conflicts)
        if num_rr == 1:
            errorlog.warning('1 reduce/reduce conflict')
        elif num_rr > 1:
            errorlog.warning('%d reduce/reduce conflicts', num_rr)

    # Write out conflicts to the output file
    if debug and (lr.sr_conflicts or lr.rr_conflicts):
        debuglog.warning('')
        debuglog.warning('Conflicts:')
        debuglog.warning('')

        for state, tok, resolution in lr.sr_conflicts:
            debuglog.warning('shift/reduce conflict for %s in state %d resolved as %s',  tok, state, resolution)

        already_reported = set()
        for state, rule, rejected in lr.rr_conflicts:
            if (state, id(rule), id(rejected)) in already_reported:
                continue
            debuglog.warning('reduce/reduce conflict in state %d resolved using rule (%s)', state, rule)
            debuglog.warning('rejected rule (%s) in state %d', rejected, state)
            errorlog.warning('reduce/reduce conflict in state %d resolved using rule (%s)', state, rule)
            errorlog.warning('rejected rule (%s) in state %d', rejected, state)
            already_reported.add((state, id(rule), id(rejected)))

        warned_never = []
        for state, rule, rejected in lr.rr_conflicts:
            if not rejected.reduced and (rejected not in warned_never):
                debuglog.warning('Rule (%s) is never reduced', rejected)
                errorlog.warning('Rule (%s) is never reduced', rejected)
                warned_never.append(rejected)

    # Write the table file if requested
    if write_tables:
        try:
            lr.write_table(tabmodule, outputdir, signature)
        except IOError as e:
            errorlog.warning("Couldn't create %r. %s" % (tabmodule, e))

    # Write a pickled version of the tables
    if picklefile:
        try:
            lr.pickle_table(picklefile, signature)
        except IOError as e:
            errorlog.warning("Couldn't create %r. %s" % (picklefile, e))

    # Build the parser
    lr.bind_callables(pinfo.pdict)
    parser = LRParser(lr, pinfo.error_func)

    parse = parser.parse
    return parser

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\strategies.py ===
# orm/strategies.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""sqlalchemy.orm.interfaces.LoaderStrategy
   implementations, and related MapperOptions."""

from __future__ import annotations

import collections
import itertools
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from . import attributes
from . import exc as orm_exc
from . import interfaces
from . import loading
from . import path_registry
from . import properties
from . import query
from . import relationships
from . import unitofwork
from . import util as orm_util
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import ATTR_WAS_SET
from .base import LoaderCallableStatus
from .base import PASSIVE_OFF
from .base import PassiveFlag
from .context import _column_descriptions
from .context import ORMCompileState
from .context import ORMSelectCompileState
from .context import QueryContext
from .interfaces import LoaderStrategy
from .interfaces import StrategizedProperty
from .session import _state_session
from .state import InstanceState
from .strategy_options import Load
from .util import _none_only_set
from .util import AliasedClass
from .. import event
from .. import exc as sa_exc
from .. import inspect
from .. import log
from .. import sql
from .. import util
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..util.typing import Literal

if TYPE_CHECKING:
    from .mapper import Mapper
    from .relationships import RelationshipProperty
    from ..sql.elements import ColumnElement


def _register_attribute(
    prop,
    mapper,
    useobject,
    compare_function=None,
    typecallable=None,
    callable_=None,
    proxy_property=None,
    active_history=False,
    impl_class=None,
    **kw,
):
    listen_hooks = []

    uselist = useobject and prop.uselist

    if useobject and prop.single_parent:
        listen_hooks.append(single_parent_validator)

    if prop.key in prop.parent.validators:
        fn, opts = prop.parent.validators[prop.key]
        listen_hooks.append(
            lambda desc, prop: orm_util._validator_events(
                desc, prop.key, fn, **opts
            )
        )

    if useobject:
        listen_hooks.append(unitofwork.track_cascade_events)

    # need to assemble backref listeners
    # after the singleparentvalidator, mapper validator
    if useobject:
        backref = prop.back_populates
        if backref and prop._effective_sync_backref:
            listen_hooks.append(
                lambda desc, prop: attributes.backref_listeners(
                    desc, backref, uselist
                )
            )

    # a single MapperProperty is shared down a class inheritance
    # hierarchy, so we set up attribute instrumentation and backref event
    # for each mapper down the hierarchy.

    # typically, "mapper" is the same as prop.parent, due to the way
    # the configure_mappers() process runs, however this is not strongly
    # enforced, and in the case of a second configure_mappers() run the
    # mapper here might not be prop.parent; also, a subclass mapper may
    # be called here before a superclass mapper.  That is, can't depend
    # on mappers not already being set up so we have to check each one.

    for m in mapper.self_and_descendants:
        if prop is m._props.get(
            prop.key
        ) and not m.class_manager._attr_has_impl(prop.key):
            desc = attributes.register_attribute_impl(
                m.class_,
                prop.key,
                parent_token=prop,
                uselist=uselist,
                compare_function=compare_function,
                useobject=useobject,
                trackparent=useobject
                and (
                    prop.single_parent
                    or prop.direction is interfaces.ONETOMANY
                ),
                typecallable=typecallable,
                callable_=callable_,
                active_history=active_history,
                impl_class=impl_class,
                send_modified_events=not useobject or not prop.viewonly,
                doc=prop.doc,
                **kw,
            )

            for hook in listen_hooks:
                hook(desc, prop)


@properties.ColumnProperty.strategy_for(instrument=False, deferred=False)
class UninstrumentedColumnLoader(LoaderStrategy):
    """Represent a non-instrumented MapperProperty.

    The polymorphic_on argument of mapper() often results in this,
    if the argument is against the with_polymorphic selectable.

    """

    __slots__ = ("columns",)

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.columns = self.parent_property.columns

    def setup_query(
        self,
        compile_state,
        query_entity,
        path,
        loadopt,
        adapter,
        column_collection=None,
        **kwargs,
    ):
        for c in self.columns:
            if adapter:
                c = adapter.columns[c]
            compile_state._append_dedupe_col_collection(c, column_collection)

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        pass


@log.class_logger
@properties.ColumnProperty.strategy_for(instrument=True, deferred=False)
class ColumnLoader(LoaderStrategy):
    """Provide loading behavior for a :class:`.ColumnProperty`."""

    __slots__ = "columns", "is_composite"

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.columns = self.parent_property.columns
        self.is_composite = hasattr(self.parent_property, "composite_class")

    def setup_query(
        self,
        compile_state,
        query_entity,
        path,
        loadopt,
        adapter,
        column_collection,
        memoized_populators,
        check_for_adapt=False,
        **kwargs,
    ):
        for c in self.columns:
            if adapter:
                if check_for_adapt:
                    c = adapter.adapt_check_present(c)
                    if c is None:
                        return
                else:
                    c = adapter.columns[c]

            compile_state._append_dedupe_col_collection(c, column_collection)

        fetch = self.columns[0]
        if adapter:
            fetch = adapter.columns[fetch]
            if fetch is None:
                # None happens here only for dml bulk_persistence cases
                # when context.DMLReturningColFilter is used
                return

        memoized_populators[self.parent_property] = fetch

    def init_class_attribute(self, mapper):
        self.is_class_level = True
        coltype = self.columns[0].type
        # TODO: check all columns ?  check for foreign key as well?
        active_history = (
            self.parent_property.active_history
            or self.columns[0].primary_key
            or (
                mapper.version_id_col is not None
                and mapper._columntoproperty.get(mapper.version_id_col, None)
                is self.parent_property
            )
        )

        _register_attribute(
            self.parent_property,
            mapper,
            useobject=False,
            compare_function=coltype.compare_values,
            active_history=active_history,
        )

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        # look through list of columns represented here
        # to see which, if any, is present in the row.

        for col in self.columns:
            if adapter:
                col = adapter.columns[col]
            getter = result._getter(col, False)
            if getter:
                populators["quick"].append((self.key, getter))
                break
        else:
            populators["expire"].append((self.key, True))


@log.class_logger
@properties.ColumnProperty.strategy_for(query_expression=True)
class ExpressionColumnLoader(ColumnLoader):
    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)

        # compare to the "default" expression that is mapped in
        # the column.   If it's sql.null, we don't need to render
        # unless an expr is passed in the options.
        null = sql.null().label(None)
        self._have_default_expression = any(
            not c.compare(null) for c in self.parent_property.columns
        )

    def setup_query(
        self,
        compile_state,
        query_entity,
        path,
        loadopt,
        adapter,
        column_collection,
        memoized_populators,
        **kwargs,
    ):
        columns = None
        if loadopt and loadopt._extra_criteria:
            columns = loadopt._extra_criteria

        elif self._have_default_expression:
            columns = self.parent_property.columns

        if columns is None:
            return

        for c in columns:
            if adapter:
                c = adapter.columns[c]
            compile_state._append_dedupe_col_collection(c, column_collection)

        fetch = columns[0]
        if adapter:
            fetch = adapter.columns[fetch]
            if fetch is None:
                # None is not expected to be the result of any
                # adapter implementation here, however there may be theoretical
                # usages of returning() with context.DMLReturningColFilter
                return

        memoized_populators[self.parent_property] = fetch

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        # look through list of columns represented here
        # to see which, if any, is present in the row.
        if loadopt and loadopt._extra_criteria:
            columns = loadopt._extra_criteria

            for col in columns:
                if adapter:
                    col = adapter.columns[col]
                getter = result._getter(col, False)
                if getter:
                    populators["quick"].append((self.key, getter))
                    break
            else:
                populators["expire"].append((self.key, True))

    def init_class_attribute(self, mapper):
        self.is_class_level = True

        _register_attribute(
            self.parent_property,
            mapper,
            useobject=False,
            compare_function=self.columns[0].type.compare_values,
            accepts_scalar_loader=False,
        )


@log.class_logger
@properties.ColumnProperty.strategy_for(deferred=True, instrument=True)
@properties.ColumnProperty.strategy_for(
    deferred=True, instrument=True, raiseload=True
)
@properties.ColumnProperty.strategy_for(do_nothing=True)
class DeferredColumnLoader(LoaderStrategy):
    """Provide loading behavior for a deferred :class:`.ColumnProperty`."""

    __slots__ = "columns", "group", "raiseload"

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        if hasattr(self.parent_property, "composite_class"):
            raise NotImplementedError(
                "Deferred loading for composite types not implemented yet"
            )
        self.raiseload = self.strategy_opts.get("raiseload", False)
        self.columns = self.parent_property.columns
        self.group = self.parent_property.group

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        # for a DeferredColumnLoader, this method is only used during a
        # "row processor only" query; see test_deferred.py ->
        # tests with "rowproc_only" in their name.  As of the 1.0 series,
        # loading._instance_processor doesn't use a "row processing" function
        # to populate columns, instead it uses data in the "populators"
        # dictionary.  Normally, the DeferredColumnLoader.setup_query()
        # sets up that data in the "memoized_populators" dictionary
        # and "create_row_processor()" here is never invoked.

        if (
            context.refresh_state
            and context.query._compile_options._only_load_props
            and self.key in context.query._compile_options._only_load_props
        ):
            self.parent_property._get_strategy(
                (("deferred", False), ("instrument", True))
            ).create_row_processor(
                context,
                query_entity,
                path,
                loadopt,
                mapper,
                result,
                adapter,
                populators,
            )

        elif not self.is_class_level:
            if self.raiseload:
                set_deferred_for_local_state = (
                    self.parent_property._raise_column_loader
                )
            else:
                set_deferred_for_local_state = (
                    self.parent_property._deferred_column_loader
                )
            populators["new"].append((self.key, set_deferred_for_local_state))
        else:
            populators["expire"].append((self.key, False))

    def init_class_attribute(self, mapper):
        self.is_class_level = True

        _register_attribute(
            self.parent_property,
            mapper,
            useobject=False,
            compare_function=self.columns[0].type.compare_values,
            callable_=self._load_for_state,
            load_on_unexpire=False,
        )

    def setup_query(
        self,
        compile_state,
        query_entity,
        path,
        loadopt,
        adapter,
        column_collection,
        memoized_populators,
        only_load_props=None,
        **kw,
    ):
        if (
            (
                compile_state.compile_options._render_for_subquery
                and self.parent_property._renders_in_subqueries
            )
            or (
                loadopt
                and set(self.columns).intersection(
                    self.parent._should_undefer_in_wildcard
                )
            )
            or (
                loadopt
                and self.group
                and loadopt.local_opts.get(
                    "undefer_group_%s" % self.group, False
                )
            )
            or (only_load_props and self.key in only_load_props)
        ):
            self.parent_property._get_strategy(
                (("deferred", False), ("instrument", True))
            ).setup_query(
                compile_state,
                query_entity,
                path,
                loadopt,
                adapter,
                column_collection,
                memoized_populators,
                **kw,
            )
        elif self.is_class_level:
            memoized_populators[self.parent_property] = _SET_DEFERRED_EXPIRED
        elif not self.raiseload:
            memoized_populators[self.parent_property] = _DEFER_FOR_STATE
        else:
            memoized_populators[self.parent_property] = _RAISE_FOR_STATE

    def _load_for_state(self, state, passive):
        if not state.key:
            return LoaderCallableStatus.ATTR_EMPTY

        if not passive & PassiveFlag.SQL_OK:
            return LoaderCallableStatus.PASSIVE_NO_RESULT

        localparent = state.manager.mapper

        if self.group:
            toload = [
                p.key
                for p in localparent.iterate_properties
                if isinstance(p, StrategizedProperty)
                and isinstance(p.strategy, DeferredColumnLoader)
                and p.group == self.group
            ]
        else:
            toload = [self.key]

        # narrow the keys down to just those which have no history
        group = [k for k in toload if k in state.unmodified]

        session = _state_session(state)
        if session is None:
            raise orm_exc.DetachedInstanceError(
                "Parent instance %s is not bound to a Session; "
                "deferred load operation of attribute '%s' cannot proceed"
                % (orm_util.state_str(state), self.key)
            )

        if self.raiseload:
            self._invoke_raise_load(state, passive, "raise")

        loading.load_scalar_attributes(
            state.mapper, state, set(group), PASSIVE_OFF
        )

        return LoaderCallableStatus.ATTR_WAS_SET

    def _invoke_raise_load(self, state, passive, lazy):
        raise sa_exc.InvalidRequestError(
            "'%s' is not available due to raiseload=True" % (self,)
        )


class LoadDeferredColumns:
    """serializable loader object used by DeferredColumnLoader"""

    def __init__(self, key: str, raiseload: bool = False):
        self.key = key
        self.raiseload = raiseload

    def __call__(self, state, passive=attributes.PASSIVE_OFF):
        key = self.key

        localparent = state.manager.mapper
        prop = localparent._props[key]
        if self.raiseload:
            strategy_key = (
                ("deferred", True),
                ("instrument", True),
                ("raiseload", True),
            )
        else:
            strategy_key = (("deferred", True), ("instrument", True))
        strategy = prop._get_strategy(strategy_key)
        return strategy._load_for_state(state, passive)


class AbstractRelationshipLoader(LoaderStrategy):
    """LoaderStratgies which deal with related objects."""

    __slots__ = "mapper", "target", "uselist", "entity"

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.mapper = self.parent_property.mapper
        self.entity = self.parent_property.entity
        self.target = self.parent_property.target
        self.uselist = self.parent_property.uselist

    def _immediateload_create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        return self.parent_property._get_strategy(
            (("lazy", "immediate"),)
        ).create_row_processor(
            context,
            query_entity,
            path,
            loadopt,
            mapper,
            result,
            adapter,
            populators,
        )


@log.class_logger
@relationships.RelationshipProperty.strategy_for(do_nothing=True)
class DoNothingLoader(LoaderStrategy):
    """Relationship loader that makes no change to the object's state.

    Compared to NoLoader, this loader does not initialize the
    collection/attribute to empty/none; the usual default LazyLoader will
    take effect.

    """


@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy="noload")
@relationships.RelationshipProperty.strategy_for(lazy=None)
class NoLoader(AbstractRelationshipLoader):
    """Provide loading behavior for a :class:`.Relationship`
    with "lazy=None".

    """

    __slots__ = ()

    def init_class_attribute(self, mapper):
        self.is_class_level = True

        _register_attribute(
            self.parent_property,
            mapper,
            useobject=True,
            typecallable=self.parent_property.collection_class,
        )

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        def invoke_no_load(state, dict_, row):
            if self.uselist:
                attributes.init_state_collection(state, dict_, self.key)
            else:
                dict_[self.key] = None

        populators["new"].append((self.key, invoke_no_load))


@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy=True)
@relationships.RelationshipProperty.strategy_for(lazy="select")
@relationships.RelationshipProperty.strategy_for(lazy="raise")
@relationships.RelationshipProperty.strategy_for(lazy="raise_on_sql")
@relationships.RelationshipProperty.strategy_for(lazy="baked_select")
class LazyLoader(
    AbstractRelationshipLoader, util.MemoizedSlots, log.Identified
):
    """Provide loading behavior for a :class:`.Relationship`
    with "lazy=True", that is loads when first accessed.

    """

    __slots__ = (
        "_lazywhere",
        "_rev_lazywhere",
        "_lazyload_reverse_option",
        "_order_by",
        "use_get",
        "is_aliased_class",
        "_bind_to_col",
        "_equated_columns",
        "_rev_bind_to_col",
        "_rev_equated_columns",
        "_simple_lazy_clause",
        "_raise_always",
        "_raise_on_sql",
    )

    _lazywhere: ColumnElement[bool]
    _bind_to_col: Dict[str, ColumnElement[Any]]
    _rev_lazywhere: ColumnElement[bool]
    _rev_bind_to_col: Dict[str, ColumnElement[Any]]

    parent_property: RelationshipProperty[Any]

    def __init__(
        self, parent: RelationshipProperty[Any], strategy_key: Tuple[Any, ...]
    ):
        super().__init__(parent, strategy_key)
        self._raise_always = self.strategy_opts["lazy"] == "raise"
        self._raise_on_sql = self.strategy_opts["lazy"] == "raise_on_sql"

        self.is_aliased_class = inspect(self.entity).is_aliased_class

        join_condition = self.parent_property._join_condition
        (
            self._lazywhere,
            self._bind_to_col,
            self._equated_columns,
        ) = join_condition.create_lazy_clause()

        (
            self._rev_lazywhere,
            self._rev_bind_to_col,
            self._rev_equated_columns,
        ) = join_condition.create_lazy_clause(reverse_direction=True)

        if self.parent_property.order_by:
            self._order_by = [
                sql_util._deep_annotate(elem, {"_orm_adapt": True})
                for elem in util.to_list(self.parent_property.order_by)
            ]
        else:
            self._order_by = None

        self.logger.info("%s lazy loading clause %s", self, self._lazywhere)

        # determine if our "lazywhere" clause is the same as the mapper's
        # get() clause.  then we can just use mapper.get()
        #
        # TODO: the "not self.uselist" can be taken out entirely; a m2o
        # load that populates for a list (very unusual, but is possible with
        # the API) can still set for "None" and the attribute system will
        # populate as an empty list.
        self.use_get = (
            not self.is_aliased_class
            and not self.uselist
            and self.entity._get_clause[0].compare(
                self._lazywhere,
                use_proxies=True,
                compare_keys=False,
                equivalents=self.mapper._equivalent_columns,
            )
        )

        if self.use_get:
            for col in list(self._equated_columns):
                if col in self.mapper._equivalent_columns:
                    for c in self.mapper._equivalent_columns[col]:
                        self._equated_columns[c] = self._equated_columns[col]

            self.logger.info(
                "%s will use Session.get() to optimize instance loads", self
            )

    def init_class_attribute(self, mapper):
        self.is_class_level = True

        _legacy_inactive_history_style = (
            self.parent_property._legacy_inactive_history_style
        )

        if self.parent_property.active_history:
            active_history = True
            _deferred_history = False

        elif (
            self.parent_property.direction is not interfaces.MANYTOONE
            or not self.use_get
        ):
            if _legacy_inactive_history_style:
                active_history = True
                _deferred_history = False
            else:
                active_history = False
                _deferred_history = True
        else:
            active_history = _deferred_history = False

        _register_attribute(
            self.parent_property,
            mapper,
            useobject=True,
            callable_=self._load_for_state,
            typecallable=self.parent_property.collection_class,
            active_history=active_history,
            _deferred_history=_deferred_history,
        )

    def _memoized_attr__simple_lazy_clause(self):
        lazywhere = sql_util._deep_annotate(
            self._lazywhere, {"_orm_adapt": True}
        )

        criterion, bind_to_col = (lazywhere, self._bind_to_col)

        params = []

        def visit_bindparam(bindparam):
            bindparam.unique = False

        visitors.traverse(criterion, {}, {"bindparam": visit_bindparam})

        def visit_bindparam(bindparam):
            if bindparam._identifying_key in bind_to_col:
                params.append(
                    (
                        bindparam.key,
                        bind_to_col[bindparam._identifying_key],
                        None,
                    )
                )
            elif bindparam.callable is None:
                params.append((bindparam.key, None, bindparam.value))

        criterion = visitors.cloned_traverse(
            criterion, {}, {"bindparam": visit_bindparam}
        )

        return criterion, params

    def _generate_lazy_clause(self, state, passive):
        criterion, param_keys = self._simple_lazy_clause

        if state is None:
            return sql_util.adapt_criterion_to_null(
                criterion, [key for key, ident, value in param_keys]
            )

        mapper = self.parent_property.parent

        o = state.obj()  # strong ref
        dict_ = attributes.instance_dict(o)

        if passive & PassiveFlag.INIT_OK:
            passive ^= PassiveFlag.INIT_OK

        params = {}
        for key, ident, value in param_keys:
            if ident is not None:
                if passive and passive & PassiveFlag.LOAD_AGAINST_COMMITTED:
                    value = mapper._get_committed_state_attr_by_column(
                        state, dict_, ident, passive
                    )
                else:
                    value = mapper._get_state_attr_by_column(
                        state, dict_, ident, passive
                    )

            params[key] = value

        return criterion, params

    def _invoke_raise_load(self, state, passive, lazy):
        raise sa_exc.InvalidRequestError(
            "'%s' is not available due to lazy='%s'" % (self, lazy)
        )

    def _load_for_state(
        self,
        state,
        passive,
        loadopt=None,
        extra_criteria=(),
        extra_options=(),
        alternate_effective_path=None,
        execution_options=util.EMPTY_DICT,
    ):
        if not state.key and (
            (
                not self.parent_property.load_on_pending
                and not state._load_pending
            )
            or not state.session_id
        ):
            return LoaderCallableStatus.ATTR_EMPTY

        pending = not state.key
        primary_key_identity = None

        use_get = self.use_get and (not loadopt or not loadopt._extra_criteria)

        if (not passive & PassiveFlag.SQL_OK and not use_get) or (
            not passive & attributes.NON_PERSISTENT_OK and pending
        ):
            return LoaderCallableStatus.PASSIVE_NO_RESULT

        if (
            # we were given lazy="raise"
            self._raise_always
            # the no_raise history-related flag was not passed
            and not passive & PassiveFlag.NO_RAISE
            and (
                # if we are use_get and related_object_ok is disabled,
                # which means we are at most looking in the identity map
                # for history purposes or otherwise returning
                # PASSIVE_NO_RESULT, don't raise.  This is also a
                # history-related flag
                not use_get
                or passive & PassiveFlag.RELATED_OBJECT_OK
            )
        ):
            self._invoke_raise_load(state, passive, "raise")

        session = _state_session(state)
        if not session:
            if passive & PassiveFlag.NO_RAISE:
                return LoaderCallableStatus.PASSIVE_NO_RESULT

            raise orm_exc.DetachedInstanceError(
                "Parent instance %s is not bound to a Session; "
                "lazy load operation of attribute '%s' cannot proceed"
                % (orm_util.state_str(state), self.key)
            )

        # if we have a simple primary key load, check the
        # identity map without generating a Query at all
        if use_get:
            primary_key_identity = self._get_ident_for_use_get(
                session, state, passive
            )
            if LoaderCallableStatus.PASSIVE_NO_RESULT in primary_key_identity:
                return LoaderCallableStatus.PASSIVE_NO_RESULT
            elif LoaderCallableStatus.NEVER_SET in primary_key_identity:
                return LoaderCallableStatus.NEVER_SET

            # test for None alone in primary_key_identity based on
            # allow_partial_pks preference.   PASSIVE_NO_RESULT and NEVER_SET
            # have already been tested above
            if not self.mapper.allow_partial_pks:
                if _none_only_set.intersection(primary_key_identity):
                    return None
            else:
                if _none_only_set.issuperset(primary_key_identity):
                    return None

            if (
                self.key in state.dict
                and not passive & PassiveFlag.DEFERRED_HISTORY_LOAD
            ):
                return LoaderCallableStatus.ATTR_WAS_SET

            # look for this identity in the identity map.  Delegate to the
            # Query class in use, as it may have special rules for how it
            # does this, including how it decides what the correct
            # identity_token would be for this identity.

            instance = session._identity_lookup(
                self.entity,
                primary_key_identity,
                passive=passive,
                lazy_loaded_from=state,
            )

            if instance is not None:
                if instance is LoaderCallableStatus.PASSIVE_CLASS_MISMATCH:
                    return None
                else:
                    return instance
            elif (
                not passive & PassiveFlag.SQL_OK
                or not passive & PassiveFlag.RELATED_OBJECT_OK
            ):
                return LoaderCallableStatus.PASSIVE_NO_RESULT

        return self._emit_lazyload(
            session,
            state,
            primary_key_identity,
            passive,
            loadopt,
            extra_criteria,
            extra_options,
            alternate_effective_path,
            execution_options,
        )

    def _get_ident_for_use_get(self, session, state, passive):
        instance_mapper = state.manager.mapper

        if passive & PassiveFlag.LOAD_AGAINST_COMMITTED:
            get_attr = instance_mapper._get_committed_state_attr_by_column
        else:
            get_attr = instance_mapper._get_state_attr_by_column

        dict_ = state.dict

        return [
            get_attr(state, dict_, self._equated_columns[pk], passive=passive)
            for pk in self.mapper.primary_key
        ]

    @util.preload_module("sqlalchemy.orm.strategy_options")
    def _emit_lazyload(
        self,
        session,
        state,
        primary_key_identity,
        passive,
        loadopt,
        extra_criteria,
        extra_options,
        alternate_effective_path,
        execution_options,
    ):
        strategy_options = util.preloaded.orm_strategy_options

        clauseelement = self.entity.__clause_element__()
        stmt = Select._create_raw_select(
            _raw_columns=[clauseelement],
            _propagate_attrs=clauseelement._propagate_attrs,
            _label_style=LABEL_STYLE_TABLENAME_PLUS_COL,
            _compile_options=ORMCompileState.default_compile_options,
        )
        load_options = QueryContext.default_load_options

        load_options += {
            "_invoke_all_eagers": False,
            "_lazy_loaded_from": state,
        }

        if self.parent_property.secondary is not None:
            stmt = stmt.select_from(
                self.mapper, self.parent_property.secondary
            )

        pending = not state.key

        # don't autoflush on pending
        if pending or passive & attributes.NO_AUTOFLUSH:
            stmt._execution_options = util.immutabledict({"autoflush": False})

        use_get = self.use_get

        if state.load_options or (loadopt and loadopt._extra_criteria):
            if alternate_effective_path is None:
                effective_path = state.load_path[self.parent_property]
            else:
                effective_path = alternate_effective_path[self.parent_property]

            opts = state.load_options

            if loadopt and loadopt._extra_criteria:
                use_get = False
                opts += (
                    orm_util.LoaderCriteriaOption(self.entity, extra_criteria),
                )

            stmt._with_options = opts
        elif alternate_effective_path is None:
            # this path is used if there are not already any options
            # in the query, but an event may want to add them
            effective_path = state.mapper._path_registry[self.parent_property]
        else:
            # added by immediateloader
            effective_path = alternate_effective_path[self.parent_property]

        if extra_options:
            stmt._with_options += extra_options

        stmt._compile_options += {"_current_path": effective_path}

        if use_get:
            if self._raise_on_sql and not passive & PassiveFlag.NO_RAISE:
                self._invoke_raise_load(state, passive, "raise_on_sql")

            return loading.load_on_pk_identity(
                session,
                stmt,
                primary_key_identity,
                load_options=load_options,
                execution_options=execution_options,
            )

        if self._order_by:
            stmt._order_by_clauses = self._order_by

        def _lazyload_reverse(compile_context):
            for rev in self.parent_property._reverse_property:
                # reverse props that are MANYTOONE are loading *this*
                # object from get(), so don't need to eager out to those.
                if (
                    rev.direction is interfaces.MANYTOONE
                    and rev._use_get
                    and not isinstance(rev.strategy, LazyLoader)
                ):
                    strategy_options.Load._construct_for_existing_path(
                        compile_context.compile_options._current_path[
                            rev.parent
                        ]
                    ).lazyload(rev).process_compile_state(compile_context)

        stmt._with_context_options += (
            (_lazyload_reverse, self.parent_property),
        )

        lazy_clause, params = self._generate_lazy_clause(state, passive)

        if execution_options:
            execution_options = util.EMPTY_DICT.merge_with(
                execution_options,
                {
                    "_sa_orm_load_options": load_options,
                },
            )
        else:
            execution_options = {
                "_sa_orm_load_options": load_options,
            }

        if (
            self.key in state.dict
            and not passive & PassiveFlag.DEFERRED_HISTORY_LOAD
        ):
            return LoaderCallableStatus.ATTR_WAS_SET

        if pending:
            if util.has_intersection(orm_util._none_set, params.values()):
                return None

        elif util.has_intersection(orm_util._never_set, params.values()):
            return None

        if self._raise_on_sql and not passive & PassiveFlag.NO_RAISE:
            self._invoke_raise_load(state, passive, "raise_on_sql")

        stmt._where_criteria = (lazy_clause,)

        result = session.execute(
            stmt, params, execution_options=execution_options
        )

        result = result.unique().scalars().all()

        if self.uselist:
            return result
        else:
            l = len(result)
            if l:
                if l > 1:
                    util.warn(
                        "Multiple rows returned with "
                        "uselist=False for lazily-loaded attribute '%s' "
                        % self.parent_property
                    )

                return result[0]
            else:
                return None

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        key = self.key

        if (
            context.load_options._is_user_refresh
            and context.query._compile_options._only_load_props
            and self.key in context.query._compile_options._only_load_props
        ):
            return self._immediateload_create_row_processor(
                context,
                query_entity,
                path,
                loadopt,
                mapper,
                result,
                adapter,
                populators,
            )

        if not self.is_class_level or (loadopt and loadopt._extra_criteria):
            # we are not the primary manager for this attribute
            # on this class - set up a
            # per-instance lazyloader, which will override the
            # class-level behavior.
            # this currently only happens when using a
            # "lazyload" option on a "no load"
            # attribute - "eager" attributes always have a
            # class-level lazyloader installed.
            set_lazy_callable = (
                InstanceState._instance_level_callable_processor
            )(
                mapper.class_manager,
                LoadLazyAttribute(
                    key,
                    self,
                    loadopt,
                    (
                        loadopt._generate_extra_criteria(context)
                        if loadopt._extra_criteria
                        else None
                    ),
                ),
                key,
            )

            populators["new"].append((self.key, set_lazy_callable))
        elif context.populate_existing or mapper.always_refresh:

            def reset_for_lazy_callable(state, dict_, row):
                # we are the primary manager for this attribute on
                # this class - reset its
                # per-instance attribute state, so that the class-level
                # lazy loader is
                # executed when next referenced on this instance.
                # this is needed in
                # populate_existing() types of scenarios to reset
                # any existing state.
                state._reset(dict_, key)

            populators["new"].append((self.key, reset_for_lazy_callable))


class LoadLazyAttribute:
    """semi-serializable loader object used by LazyLoader

    Historically, this object would be carried along with instances that
    needed to run lazyloaders, so it had to be serializable to support
    cached instances.

    this is no longer a general requirement, and the case where this object
    is used is exactly the case where we can't really serialize easily,
    which is when extra criteria in the loader option is present.

    We can't reliably serialize that as it refers to mapped entities and
    AliasedClass objects that are local to the current process, which would
    need to be matched up on deserialize e.g. the sqlalchemy.ext.serializer
    approach.

    """

    def __init__(self, key, initiating_strategy, loadopt, extra_criteria):
        self.key = key
        self.strategy_key = initiating_strategy.strategy_key
        self.loadopt = loadopt
        self.extra_criteria = extra_criteria

    def __getstate__(self):
        if self.extra_criteria is not None:
            util.warn(
                "Can't reliably serialize a lazyload() option that "
                "contains additional criteria; please use eager loading "
                "for this case"
            )
        return {
            "key": self.key,
            "strategy_key": self.strategy_key,
            "loadopt": self.loadopt,
            "extra_criteria": (),
        }

    def __call__(self, state, passive=attributes.PASSIVE_OFF):
        key = self.key
        instance_mapper = state.manager.mapper
        prop = instance_mapper._props[key]
        strategy = prop._strategies[self.strategy_key]

        return strategy._load_for_state(
            state,
            passive,
            loadopt=self.loadopt,
            extra_criteria=self.extra_criteria,
        )


class PostLoader(AbstractRelationshipLoader):
    """A relationship loader that emits a second SELECT statement."""

    __slots__ = ()

    def _setup_for_recursion(self, context, path, loadopt, join_depth=None):
        effective_path = (
            context.compile_state.current_path or orm_util.PathRegistry.root
        ) + path

        top_level_context = context._get_top_level_context()
        execution_options = util.immutabledict(
            {"sa_top_level_orm_context": top_level_context}
        )

        if loadopt:
            recursion_depth = loadopt.local_opts.get("recursion_depth", None)
            unlimited_recursion = recursion_depth == -1
        else:
            recursion_depth = None
            unlimited_recursion = False

        if recursion_depth is not None:
            if not self.parent_property._is_self_referential:
                raise sa_exc.InvalidRequestError(
                    f"recursion_depth option on relationship "
                    f"{self.parent_property} not valid for "
                    "non-self-referential relationship"
                )
            recursion_depth = context.execution_options.get(
                f"_recursion_depth_{id(self)}", recursion_depth
            )

            if not unlimited_recursion and recursion_depth < 0:
                return (
                    effective_path,
                    False,
                    execution_options,
                    recursion_depth,
                )

            if not unlimited_recursion:
                execution_options = execution_options.union(
                    {
                        f"_recursion_depth_{id(self)}": recursion_depth - 1,
                    }
                )

        if loading.PostLoad.path_exists(
            context, effective_path, self.parent_property
        ):
            return effective_path, False, execution_options, recursion_depth

        path_w_prop = path[self.parent_property]
        effective_path_w_prop = effective_path[self.parent_property]

        if not path_w_prop.contains(context.attributes, "loader"):
            if join_depth:
                if effective_path_w_prop.length / 2 > join_depth:
                    return (
                        effective_path,
                        False,
                        execution_options,
                        recursion_depth,
                    )
            elif effective_path_w_prop.contains_mapper(self.mapper):
                return (
                    effective_path,
                    False,
                    execution_options,
                    recursion_depth,
                )

        return effective_path, True, execution_options, recursion_depth


@relationships.RelationshipProperty.strategy_for(lazy="immediate")
class ImmediateLoader(PostLoader):
    __slots__ = ("join_depth",)

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy(
            (("lazy", "select"),)
        ).init_class_attribute(mapper)

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        if not context.compile_state.compile_options._enable_eagerloads:
            return

        (
            effective_path,
            run_loader,
            execution_options,
            recursion_depth,
        ) = self._setup_for_recursion(context, path, loadopt, self.join_depth)

        if not run_loader:
            # this will not emit SQL and will only emit for a many-to-one
            # "use get" load.   the "_RELATED" part means it may return
            # instance even if its expired, since this is a mutually-recursive
            # load operation.
            flags = attributes.PASSIVE_NO_FETCH_RELATED | PassiveFlag.NO_RAISE
        else:
            flags = attributes.PASSIVE_OFF | PassiveFlag.NO_RAISE

        loading.PostLoad.callable_for_path(
            context,
            effective_path,
            self.parent,
            self.parent_property,
            self._load_for_path,
            loadopt,
            flags,
            recursion_depth,
            execution_options,
        )

    def _load_for_path(
        self,
        context,
        path,
        states,
        load_only,
        loadopt,
        flags,
        recursion_depth,
        execution_options,
    ):
        if recursion_depth:
            new_opt = Load(loadopt.path.entity)
            new_opt.context = (
                loadopt,
                loadopt._recurse(),
            )
            alternate_effective_path = path._truncate_recursive()
            extra_options = (new_opt,)
        else:
            alternate_effective_path = path
            extra_options = ()

        key = self.key
        lazyloader = self.parent_property._get_strategy((("lazy", "select"),))
        for state, overwrite in states:
            dict_ = state.dict

            if overwrite or key not in dict_:
                value = lazyloader._load_for_state(
                    state,
                    flags,
                    extra_options=extra_options,
                    alternate_effective_path=alternate_effective_path,
                    execution_options=execution_options,
                )
                if value not in (
                    ATTR_WAS_SET,
                    LoaderCallableStatus.PASSIVE_NO_RESULT,
                ):
                    state.get_impl(key).set_committed_value(
                        state, dict_, value
                    )


@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy="subquery")
class SubqueryLoader(PostLoader):
    __slots__ = ("join_depth",)

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy(
            (("lazy", "select"),)
        ).init_class_attribute(mapper)

    def _get_leftmost(
        self,
        orig_query_entity_index,
        subq_path,
        current_compile_state,
        is_root,
    ):
        given_subq_path = subq_path
        subq_path = subq_path.path
        subq_mapper = orm_util._class_to_mapper(subq_path[0])

        # determine attributes of the leftmost mapper
        if (
            self.parent.isa(subq_mapper)
            and self.parent_property is subq_path[1]
        ):
            leftmost_mapper, leftmost_prop = self.parent, self.parent_property
        else:
            leftmost_mapper, leftmost_prop = subq_mapper, subq_path[1]

        if is_root:
            # the subq_path is also coming from cached state, so when we start
            # building up this path, it has to also be converted to be in terms
            # of the current state. this is for the specific case of the entity
            # is an AliasedClass against a subquery that's not otherwise going
            # to adapt
            new_subq_path = current_compile_state._entities[
                orig_query_entity_index
            ].entity_zero._path_registry[leftmost_prop]
            additional = len(subq_path) - len(new_subq_path)
            if additional:
                new_subq_path += path_registry.PathRegistry.coerce(
                    subq_path[-additional:]
                )
        else:
            new_subq_path = given_subq_path

        leftmost_cols = leftmost_prop.local_columns

        leftmost_attr = [
            getattr(
                new_subq_path.path[0].entity,
                leftmost_mapper._columntoproperty[c].key,
            )
            for c in leftmost_cols
        ]

        return leftmost_mapper, leftmost_attr, leftmost_prop, new_subq_path

    def _generate_from_original_query(
        self,
        orig_compile_state,
        orig_query,
        leftmost_mapper,
        leftmost_attr,
        leftmost_relationship,
        orig_entity,
    ):
        # reformat the original query
        # to look only for significant columns
        q = orig_query._clone().correlate(None)

        # LEGACY: make a Query back from the select() !!
        # This suits at least two legacy cases:
        # 1. applications which expect before_compile() to be called
        #    below when we run .subquery() on this query (Keystone)
        # 2. applications which are doing subqueryload with complex
        #    from_self() queries, as query.subquery() / .statement
        #    has to do the full compile context for multiply-nested
        #    from_self() (Neutron) - see test_subqload_from_self
        #    for demo.
        q2 = query.Query.__new__(query.Query)
        q2.__dict__.update(q.__dict__)
        q = q2

        # set the query's "FROM" list explicitly to what the
        # FROM list would be in any case, as we will be limiting
        # the columns in the SELECT list which may no longer include
        # all entities mentioned in things like WHERE, JOIN, etc.
        if not q._from_obj:
            q._enable_assertions = False
            q.select_from.non_generative(
                q,
                *{
                    ent["entity"]
                    for ent in _column_descriptions(
                        orig_query, compile_state=orig_compile_state
                    )
                    if ent["entity"] is not None
                },
            )

        # select from the identity columns of the outer (specifically, these
        # are the 'local_cols' of the property).  This will remove other
        # columns from the query that might suggest the right entity which is
        # why we do set select_from above.   The attributes we have are
        # coerced and adapted using the original query's adapter, which is
        # needed only for the case of adapting a subclass column to
        # that of a polymorphic selectable, e.g. we have
        # Engineer.primary_language and the entity is Person.  All other
        # adaptations, e.g. from_self, select_entity_from(), will occur
        # within the new query when it compiles, as the compile_state we are
        # using here is only a partial one.  If the subqueryload is from a
        # with_polymorphic() or other aliased() object, left_attr will already
        # be the correct attributes so no adaptation is needed.
        target_cols = orig_compile_state._adapt_col_list(
            [
                sql.coercions.expect(sql.roles.ColumnsClauseRole, o)
                for o in leftmost_attr
            ],
            orig_compile_state._get_current_adapter(),
        )
        q._raw_columns = target_cols

        distinct_target_key = leftmost_relationship.distinct_target_key

        if distinct_target_key is True:
            q._distinct = True
        elif distinct_target_key is None:
            # if target_cols refer to a non-primary key or only
            # part of a composite primary key, set the q as distinct
            for t in {c.table for c in target_cols}:
                if not set(target_cols).issuperset(t.primary_key):
                    q._distinct = True
                    break

        # don't need ORDER BY if no limit/offset
        if not q._has_row_limiting_clause:
            q._order_by_clauses = ()

        if q._distinct is True and q._order_by_clauses:
            # the logic to automatically add the order by columns to the query
            # when distinct is True is deprecated in the query
            to_add = sql_util.expand_column_list_from_order_by(
                target_cols, q._order_by_clauses
            )
            if to_add:
                q._set_entities(target_cols + to_add)

        # the original query now becomes a subquery
        # which we'll join onto.
        # LEGACY: as "q" is a Query, the before_compile() event is invoked
        # here.
        embed_q = q.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery()
        left_alias = orm_util.AliasedClass(
            leftmost_mapper, embed_q, use_mapper_path=True
        )
        return left_alias

    def _prep_for_joins(self, left_alias, subq_path):
        # figure out what's being joined.  a.k.a. the fun part
        to_join = []
        pairs = list(subq_path.pairs())

        for i, (mapper, prop) in enumerate(pairs):
            if i > 0:
                # look at the previous mapper in the chain -
                # if it is as or more specific than this prop's
                # mapper, use that instead.
                # note we have an assumption here that
                # the non-first element is always going to be a mapper,
                # not an AliasedClass

                prev_mapper = pairs[i - 1][1].mapper
                to_append = prev_mapper if prev_mapper.isa(mapper) else mapper
            else:
                to_append = mapper

            to_join.append((to_append, prop.key))

        # determine the immediate parent class we are joining from,
        # which needs to be aliased.

        if len(to_join) < 2:
            # in the case of a one level eager load, this is the
            # leftmost "left_alias".
            parent_alias = left_alias
        else:
            info = inspect(to_join[-1][0])
            if info.is_aliased_class:
                parent_alias = info.entity
            else:
                # alias a plain mapper as we may be
                # joining multiple times
                parent_alias = orm_util.AliasedClass(
                    info.entity, use_mapper_path=True
                )

        local_cols = self.parent_property.local_columns

        local_attr = [
            getattr(parent_alias, self.parent._columntoproperty[c].key)
            for c in local_cols
        ]
        return to_join, local_attr, parent_alias

    def _apply_joins(
        self, q, to_join, left_alias, parent_alias, effective_entity
    ):
        ltj = len(to_join)
        if ltj == 1:
            to_join = [
                getattr(left_alias, to_join[0][1]).of_type(effective_entity)
            ]
        elif ltj == 2:
            to_join = [
                getattr(left_alias, to_join[0][1]).of_type(parent_alias),
                getattr(parent_alias, to_join[-1][1]).of_type(
                    effective_entity
                ),
            ]
        elif ltj > 2:
            middle = [
                (
                    (
                        orm_util.AliasedClass(item[0])
                        if not inspect(item[0]).is_aliased_class
                        else item[0].entity
                    ),
                    item[1],
                )
                for item in to_join[1:-1]
            ]
            inner = []

            while middle:
                item = middle.pop(0)
                attr = getattr(item[0], item[1])
                if middle:
                    attr = attr.of_type(middle[0][0])
                else:
                    attr = attr.of_type(parent_alias)

                inner.append(attr)

            to_join = (
                [getattr(left_alias, to_join[0][1]).of_type(inner[0].parent)]
                + inner
                + [
                    getattr(parent_alias, to_join[-1][1]).of_type(
                        effective_entity
                    )
                ]
            )

        for attr in to_join:
            q = q.join(attr)

        return q

    def _setup_options(
        self,
        context,
        q,
        subq_path,
        rewritten_path,
        orig_query,
        effective_entity,
        loadopt,
    ):
        # note that because the subqueryload object
        # does not re-use the cached query, instead always making
        # use of the current invoked query, while we have two queries
        # here (orig and context.query), they are both non-cached
        # queries and we can transfer the options as is without
        # adjusting for new criteria.   Some work on #6881 / #6889
        # brought this into question.
        new_options = orig_query._with_options

        if loadopt and loadopt._extra_criteria:
            new_options += (
                orm_util.LoaderCriteriaOption(
                    self.entity,
                    loadopt._generate_extra_criteria(context),
                ),
            )

        # propagate loader options etc. to the new query.
        # these will fire relative to subq_path.
        q = q._with_current_path(rewritten_path)
        q = q.options(*new_options)

        return q

    def _setup_outermost_orderby(self, q):
        if self.parent_property.order_by:

            def _setup_outermost_orderby(compile_context):
                compile_context.eager_order_by += tuple(
                    util.to_list(self.parent_property.order_by)
                )

            q = q._add_context_option(
                _setup_outermost_orderby, self.parent_property
            )

        return q

    class _SubqCollections:
        """Given a :class:`_query.Query` used to emit the "subquery load",
        provide a load interface that executes the query at the
        first moment a value is needed.

        """

        __slots__ = (
            "session",
            "execution_options",
            "load_options",
            "params",
            "subq",
            "_data",
        )

        def __init__(self, context, subq):
            # avoid creating a cycle by storing context
            # even though that's preferable
            self.session = context.session
            self.execution_options = context.execution_options
            self.load_options = context.load_options
            self.params = context.params or {}
            self.subq = subq
            self._data = None

        def get(self, key, default):
            if self._data is None:
                self._load()
            return self._data.get(key, default)

        def _load(self):
            self._data = collections.defaultdict(list)

            q = self.subq
            assert q.session is None

            q = q.with_session(self.session)

            if self.load_options._populate_existing:
                q = q.populate_existing()
            # to work with baked query, the parameters may have been
            # updated since this query was created, so take these into account

            rows = list(q.params(self.params))
            for k, v in itertools.groupby(rows, lambda x: x[1:]):
                self._data[k].extend(vv[0] for vv in v)

        def loader(self, state, dict_, row):
            if self._data is None:
                self._load()

    def _setup_query_from_rowproc(
        self,
        context,
        query_entity,
        path,
        entity,
        loadopt,
        adapter,
    ):
        compile_state = context.compile_state
        if (
            not compile_state.compile_options._enable_eagerloads
            or compile_state.compile_options._for_refresh_state
        ):
            return

        orig_query_entity_index = compile_state._entities.index(query_entity)
        context.loaders_require_buffering = True

        path = path[self.parent_property]

        # build up a path indicating the path from the leftmost
        # entity to the thing we're subquery loading.
        with_poly_entity = path.get(
            compile_state.attributes, "path_with_polymorphic", None
        )
        if with_poly_entity is not None:
            effective_entity = with_poly_entity
        else:
            effective_entity = self.entity

        subq_path, rewritten_path = context.query._execution_options.get(
            ("subquery_paths", None),
            (orm_util.PathRegistry.root, orm_util.PathRegistry.root),
        )
        is_root = subq_path is orm_util.PathRegistry.root
        subq_path = subq_path + path
        rewritten_path = rewritten_path + path

        # use the current query being invoked, not the compile state
        # one.  this is so that we get the current parameters.  however,
        # it means we can't use the existing compile state, we have to make
        # a new one.    other approaches include possibly using the
        # compiled query but swapping the params, seems only marginally
        # less time spent but more complicated
        orig_query = context.query._execution_options.get(
            ("orig_query", SubqueryLoader), context.query
        )

        # make a new compile_state for the query that's probably cached, but
        # we're sort of undoing a bit of that caching :(
        compile_state_cls = ORMCompileState._get_plugin_class_for_plugin(
            orig_query, "orm"
        )

        if orig_query._is_lambda_element:
            if context.load_options._lazy_loaded_from is None:
                util.warn(
                    'subqueryloader for "%s" must invoke lambda callable '
                    "at %r in "
                    "order to produce a new query, decreasing the efficiency "
                    "of caching for this statement.  Consider using "
                    "selectinload() for more effective full-lambda caching"
                    % (self, orig_query)
                )
            orig_query = orig_query._resolved

        # this is the more "quick" version, however it's not clear how
        # much of this we need.    in particular I can't get a test to
        # fail if the "set_base_alias" is missing and not sure why that is.
        orig_compile_state = compile_state_cls._create_entities_collection(
            orig_query, legacy=False
        )

        (
            leftmost_mapper,
            leftmost_attr,
            leftmost_relationship,
            rewritten_path,
        ) = self._get_leftmost(
            orig_query_entity_index,
            rewritten_path,
            orig_compile_state,
            is_root,
        )

        # generate a new Query from the original, then
        # produce a subquery from it.
        left_alias = self._generate_from_original_query(
            orig_compile_state,
            orig_query,
            leftmost_mapper,
            leftmost_attr,
            leftmost_relationship,
            entity,
        )

        # generate another Query that will join the
        # left alias to the target relationships.
        # basically doing a longhand
        # "from_self()".  (from_self() itself not quite industrial
        # strength enough for all contingencies...but very close)

        q = query.Query(effective_entity)

        q._execution_options = context.query._execution_options.merge_with(
            context.execution_options,
            {
                ("orig_query", SubqueryLoader): orig_query,
                ("subquery_paths", None): (subq_path, rewritten_path),
            },
        )

        q = q._set_enable_single_crit(False)
        to_join, local_attr, parent_alias = self._prep_for_joins(
            left_alias, subq_path
        )

        q = q.add_columns(*local_attr)
        q = self._apply_joins(
            q, to_join, left_alias, parent_alias, effective_entity
        )

        q = self._setup_options(
            context,
            q,
            subq_path,
            rewritten_path,
            orig_query,
            effective_entity,
            loadopt,
        )
        q = self._setup_outermost_orderby(q)

        return q

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        if (
            loadopt
            and context.compile_state.statement is not None
            and context.compile_state.statement.is_dml
        ):
            util.warn_deprecated(
                "The subqueryload loader option is not compatible with DML "
                "statements such as INSERT, UPDATE.  Only SELECT may be used."
                "This warning will become an exception in a future release.",
                "2.0",
            )

        if context.refresh_state:
            return self._immediateload_create_row_processor(
                context,
                query_entity,
                path,
                loadopt,
                mapper,
                result,
                adapter,
                populators,
            )

        _, run_loader, _, _ = self._setup_for_recursion(
            context, path, loadopt, self.join_depth
        )
        if not run_loader:
            return

        if not isinstance(context.compile_state, ORMSelectCompileState):
            # issue 7505 - subqueryload() in 1.3 and previous would silently
            # degrade for from_statement() without warning. this behavior
            # is restored here
            return

        if not self.parent.class_manager[self.key].impl.supports_population:
            raise sa_exc.InvalidRequestError(
                "'%s' does not support object "
                "population - eager loading cannot be applied." % self
            )

        # a little dance here as the "path" is still something that only
        # semi-tracks the exact series of things we are loading, still not
        # telling us about with_polymorphic() and stuff like that when it's at
        # the root..  the initial MapperEntity is more accurate for this case.
        if len(path) == 1:
            if not orm_util._entity_isa(query_entity.entity_zero, self.parent):
                return
        elif not orm_util._entity_isa(path[-1], self.parent):
            return

        subq = self._setup_query_from_rowproc(
            context,
            query_entity,
            path,
            path[-1],
            loadopt,
            adapter,
        )

        if subq is None:
            return

        assert subq.session is None

        path = path[self.parent_property]

        local_cols = self.parent_property.local_columns

        # cache the loaded collections in the context
        # so that inheriting mappers don't re-load when they
        # call upon create_row_processor again
        collections = path.get(context.attributes, "collections")
        if collections is None:
            collections = self._SubqCollections(context, subq)
            path.set(context.attributes, "collections", collections)

        if adapter:
            local_cols = [adapter.columns[c] for c in local_cols]

        if self.uselist:
            self._create_collection_loader(
                context, result, collections, local_cols, populators
            )
        else:
            self._create_scalar_loader(
                context, result, collections, local_cols, populators
            )

    def _create_collection_loader(
        self, context, result, collections, local_cols, populators
    ):
        tuple_getter = result._tuple_getter(local_cols)

        def load_collection_from_subq(state, dict_, row):
            collection = collections.get(tuple_getter(row), ())
            state.get_impl(self.key).set_committed_value(
                state, dict_, collection
            )

        def load_collection_from_subq_existing_row(state, dict_, row):
            if self.key not in dict_:
                load_collection_from_subq(state, dict_, row)

        populators["new"].append((self.key, load_collection_from_subq))
        populators["existing"].append(
            (self.key, load_collection_from_subq_existing_row)
        )

        if context.invoke_all_eagers:
            populators["eager"].append((self.key, collections.loader))

    def _create_scalar_loader(
        self, context, result, collections, local_cols, populators
    ):
        tuple_getter = result._tuple_getter(local_cols)

        def load_scalar_from_subq(state, dict_, row):
            collection = collections.get(tuple_getter(row), (None,))
            if len(collection) > 1:
                util.warn(
                    "Multiple rows returned with "
                    "uselist=False for eagerly-loaded attribute '%s' " % self
                )

            scalar = collection[0]
            state.get_impl(self.key).set_committed_value(state, dict_, scalar)

        def load_scalar_from_subq_existing_row(state, dict_, row):
            if self.key not in dict_:
                load_scalar_from_subq(state, dict_, row)

        populators["new"].append((self.key, load_scalar_from_subq))
        populators["existing"].append(
            (self.key, load_scalar_from_subq_existing_row)
        )
        if context.invoke_all_eagers:
            populators["eager"].append((self.key, collections.loader))


@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy="joined")
@relationships.RelationshipProperty.strategy_for(lazy=False)
class JoinedLoader(AbstractRelationshipLoader):
    """Provide loading behavior for a :class:`.Relationship`
    using joined eager loading.

    """

    __slots__ = "join_depth"

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy(
            (("lazy", "select"),)
        ).init_class_attribute(mapper)

    def setup_query(
        self,
        compile_state,
        query_entity,
        path,
        loadopt,
        adapter,
        column_collection=None,
        parentmapper=None,
        chained_from_outerjoin=False,
        **kwargs,
    ):
        """Add a left outer join to the statement that's being constructed."""

        if not compile_state.compile_options._enable_eagerloads:
            return
        elif (
            loadopt
            and compile_state.statement is not None
            and compile_state.statement.is_dml
        ):
            util.warn_deprecated(
                "The joinedload loader option is not compatible with DML "
                "statements such as INSERT, UPDATE.  Only SELECT may be used."
                "This warning will become an exception in a future release.",
                "2.0",
            )
        elif self.uselist:
            compile_state.multi_row_eager_loaders = True

        path = path[self.parent_property]

        user_defined_adapter = (
            self._init_user_defined_eager_proc(
                loadopt, compile_state, compile_state.attributes
            )
            if loadopt
            else False
        )

        if user_defined_adapter is not False:
            # setup an adapter but dont create any JOIN, assume it's already
            # in the query
            (
                clauses,
                adapter,
                add_to_collection,
            ) = self._setup_query_on_user_defined_adapter(
                compile_state,
                query_entity,
                path,
                adapter,
                user_defined_adapter,
            )

            # don't do "wrap" for multi-row, we want to wrap
            # limited/distinct SELECT,
            # because we want to put the JOIN on the outside.

        else:
            # if not via query option, check for
            # a cycle
            if not path.contains(compile_state.attributes, "loader"):
                if self.join_depth:
                    if path.length / 2 > self.join_depth:
                        return
                elif path.contains_mapper(self.mapper):
                    return

            # add the JOIN and create an adapter
            (
                clauses,
                adapter,
                add_to_collection,
                chained_from_outerjoin,
            ) = self._generate_row_adapter(
                compile_state,
                query_entity,
                path,
                loadopt,
                adapter,
                column_collection,
                parentmapper,
                chained_from_outerjoin,
            )

            # for multi-row, we want to wrap limited/distinct SELECT,
            # because we want to put the JOIN on the outside.
            compile_state.eager_adding_joins = True

        with_poly_entity = path.get(
            compile_state.attributes, "path_with_polymorphic", None
        )
        if with_poly_entity is not None:
            with_polymorphic = inspect(
                with_poly_entity
            ).with_polymorphic_mappers
        else:
            with_polymorphic = None

        path = path[self.entity]

        loading._setup_entity_query(
            compile_state,
            self.mapper,
            query_entity,
            path,
            clauses,
            add_to_collection,
            with_polymorphic=with_polymorphic,
            parentmapper=self.mapper,
            chained_from_outerjoin=chained_from_outerjoin,
        )

        has_nones = util.NONE_SET.intersection(compile_state.secondary_columns)

        if has_nones:
            if with_poly_entity is not None:
                raise sa_exc.InvalidRequestError(
                    "Detected unaliased columns when generating joined "
                    "load.  Make sure to use aliased=True or flat=True "
                    "when using joined loading with with_polymorphic()."
                )
            else:
                compile_state.secondary_columns = [
                    c for c in compile_state.secondary_columns if c is not None
                ]

    def _init_user_defined_eager_proc(
        self, loadopt, compile_state, target_attributes
    ):
        # check if the opt applies at all
        if "eager_from_alias" not in loadopt.local_opts:
            # nope
            return False

        path = loadopt.path.parent

        # the option applies.  check if the "user_defined_eager_row_processor"
        # has been built up.
        adapter = path.get(
            compile_state.attributes, "user_defined_eager_row_processor", False
        )
        if adapter is not False:
            # just return it
            return adapter

        # otherwise figure it out.
        alias = loadopt.local_opts["eager_from_alias"]
        root_mapper, prop = path[-2:]

        if alias is not None:
            if isinstance(alias, str):
                alias = prop.target.alias(alias)
            adapter = orm_util.ORMAdapter(
                orm_util._TraceAdaptRole.JOINEDLOAD_USER_DEFINED_ALIAS,
                prop.mapper,
                selectable=alias,
                equivalents=prop.mapper._equivalent_columns,
                limit_on_entity=False,
            )
        else:
            if path.contains(
                compile_state.attributes, "path_with_polymorphic"
            ):
                with_poly_entity = path.get(
                    compile_state.attributes, "path_with_polymorphic"
                )
                adapter = orm_util.ORMAdapter(
                    orm_util._TraceAdaptRole.JOINEDLOAD_PATH_WITH_POLYMORPHIC,
                    with_poly_entity,
                    equivalents=prop.mapper._equivalent_columns,
                )
            else:
                adapter = compile_state._polymorphic_adapters.get(
                    prop.mapper, None
                )
        path.set(
            target_attributes,
            "user_defined_eager_row_processor",
            adapter,
        )

        return adapter

    def _setup_query_on_user_defined_adapter(
        self, context, entity, path, adapter, user_defined_adapter
    ):
        # apply some more wrapping to the "user defined adapter"
        # if we are setting up the query for SQL render.
        adapter = entity._get_entity_clauses(context)

        if adapter and user_defined_adapter:
            user_defined_adapter = user_defined_adapter.wrap(adapter)
            path.set(
                context.attributes,
                "user_defined_eager_row_processor",
                user_defined_adapter,
            )
        elif adapter:
            user_defined_adapter = adapter
            path.set(
                context.attributes,
                "user_defined_eager_row_processor",
                user_defined_adapter,
            )

        add_to_collection = context.primary_columns
        return user_defined_adapter, adapter, add_to_collection

    def _generate_row_adapter(
        self,
        compile_state,
        entity,
        path,
        loadopt,
        adapter,
        column_collection,
        parentmapper,
        chained_from_outerjoin,
    ):
        with_poly_entity = path.get(
            compile_state.attributes, "path_with_polymorphic", None
        )
        if with_poly_entity:
            to_adapt = with_poly_entity
        else:
            insp = inspect(self.entity)
            if insp.is_aliased_class:
                alt_selectable = insp.selectable
            else:
                alt_selectable = None

            to_adapt = orm_util.AliasedClass(
                self.mapper,
                alias=(
                    alt_selectable._anonymous_fromclause(flat=True)
                    if alt_selectable is not None
                    else None
                ),
                flat=True,
                use_mapper_path=True,
            )

        to_adapt_insp = inspect(to_adapt)

        clauses = to_adapt_insp._memo(
            ("joinedloader_ormadapter", self),
            orm_util.ORMAdapter,
            orm_util._TraceAdaptRole.JOINEDLOAD_MEMOIZED_ADAPTER,
            to_adapt_insp,
            equivalents=self.mapper._equivalent_columns,
            adapt_required=True,
            allow_label_resolve=False,
            anonymize_labels=True,
        )

        assert clauses.is_aliased_class

        innerjoin = (
            loadopt.local_opts.get("innerjoin", self.parent_property.innerjoin)
            if loadopt is not None
            else self.parent_property.innerjoin
        )

        if not innerjoin:
            # if this is an outer join, all non-nested eager joins from
            # this path must also be outer joins
            chained_from_outerjoin = True

        compile_state.create_eager_joins.append(
            (
                self._create_eager_join,
                entity,
                path,
                adapter,
                parentmapper,
                clauses,
                innerjoin,
                chained_from_outerjoin,
                loadopt._extra_criteria if loadopt else (),
            )
        )

        add_to_collection = compile_state.secondary_columns
        path.set(compile_state.attributes, "eager_row_processor", clauses)

        return clauses, adapter, add_to_collection, chained_from_outerjoin

    def _create_eager_join(
        self,
        compile_state,
        query_entity,
        path,
        adapter,
        parentmapper,
        clauses,
        innerjoin,
        chained_from_outerjoin,
        extra_criteria,
    ):
        if parentmapper is None:
            localparent = query_entity.mapper
        else:
            localparent = parentmapper

        # whether or not the Query will wrap the selectable in a subquery,
        # and then attach eager load joins to that (i.e., in the case of
        # LIMIT/OFFSET etc.)
        should_nest_selectable = (
            compile_state.multi_row_eager_loaders
            and compile_state._should_nest_selectable
        )

        query_entity_key = None

        if (
            query_entity not in compile_state.eager_joins
            and not should_nest_selectable
            and compile_state.from_clauses
        ):
            indexes = sql_util.find_left_clause_that_matches_given(
                compile_state.from_clauses, query_entity.selectable
            )

            if len(indexes) > 1:
                # for the eager load case, I can't reproduce this right
                # now.   For query.join() I can.
                raise sa_exc.InvalidRequestError(
                    "Can't identify which query entity in which to joined "
                    "eager load from.   Please use an exact match when "
                    "specifying the join path."
                )

            if indexes:
                clause = compile_state.from_clauses[indexes[0]]
                # join to an existing FROM clause on the query.
                # key it to its list index in the eager_joins dict.
                # Query._compile_context will adapt as needed and
                # append to the FROM clause of the select().
                query_entity_key, default_towrap = indexes[0], clause

        if query_entity_key is None:
            query_entity_key, default_towrap = (
                query_entity,
                query_entity.selectable,
            )

        towrap = compile_state.eager_joins.setdefault(
            query_entity_key, default_towrap
        )

        if adapter:
            if getattr(adapter, "is_aliased_class", False):
                # joining from an adapted entity.  The adapted entity
                # might be a "with_polymorphic", so resolve that to our
                # specific mapper's entity before looking for our attribute
                # name on it.
                efm = adapter.aliased_insp._entity_for_mapper(
                    localparent
                    if localparent.isa(self.parent)
                    else self.parent
                )

                # look for our attribute on the adapted entity, else fall back
                # to our straight property
                onclause = getattr(efm.entity, self.key, self.parent_property)
            else:
                onclause = getattr(
                    orm_util.AliasedClass(
                        self.parent, adapter.selectable, use_mapper_path=True
                    ),
                    self.key,
                    self.parent_property,
                )

        else:
            onclause = self.parent_property

        assert clauses.is_aliased_class

        attach_on_outside = (
            not chained_from_outerjoin
            or not innerjoin
            or innerjoin == "unnested"
            or query_entity.entity_zero.represents_outer_join
        )

        extra_join_criteria = extra_criteria
        additional_entity_criteria = compile_state.global_attributes.get(
            ("additional_entity_criteria", self.mapper), ()
        )
        if additional_entity_criteria:
            extra_join_criteria += tuple(
                ae._resolve_where_criteria(self.mapper)
                for ae in additional_entity_criteria
                if ae.propagate_to_loaders
            )

        if attach_on_outside:
            # this is the "classic" eager join case.
            eagerjoin = orm_util._ORMJoin(
                towrap,
                clauses.aliased_insp,
                onclause,
                isouter=not innerjoin
                or query_entity.entity_zero.represents_outer_join
                or (chained_from_outerjoin and isinstance(towrap, sql.Join)),
                _left_memo=self.parent,
                _right_memo=path[self.mapper],
                _extra_criteria=extra_join_criteria,
            )
        else:
            # all other cases are innerjoin=='nested' approach
            eagerjoin = self._splice_nested_inner_join(
                path, path[-2], towrap, clauses, onclause, extra_join_criteria
            )

        compile_state.eager_joins[query_entity_key] = eagerjoin

        # send a hint to the Query as to where it may "splice" this join
        eagerjoin.stop_on = query_entity.selectable

        if not parentmapper:
            # for parentclause that is the non-eager end of the join,
            # ensure all the parent cols in the primaryjoin are actually
            # in the
            # columns clause (i.e. are not deferred), so that aliasing applied
            # by the Query propagates those columns outward.
            # This has the effect
            # of "undefering" those columns.
            for col in sql_util._find_columns(
                self.parent_property.primaryjoin
            ):
                if localparent.persist_selectable.c.contains_column(col):
                    if adapter:
                        col = adapter.columns[col]
                    compile_state._append_dedupe_col_collection(
                        col, compile_state.primary_columns
                    )

        if self.parent_property.order_by:
            compile_state.eager_order_by += tuple(
                (eagerjoin._target_adapter.copy_and_process)(
                    util.to_list(self.parent_property.order_by)
                )
            )

    def _splice_nested_inner_join(
        self,
        path,
        entity_we_want_to_splice_onto,
        join_obj,
        clauses,
        onclause,
        extra_criteria,
        entity_inside_join_structure: Union[
            Mapper, None, Literal[False]
        ] = False,
        detected_existing_path: Optional[path_registry.PathRegistry] = None,
    ):
        # recursive fn to splice a nested join into an existing one.
        # entity_inside_join_structure=False means this is the outermost call,
        # and it should return a value.  entity_inside_join_structure=<mapper>
        # indicates we've descended into a join and are looking at a FROM
        # clause representing this mapper; if this is not
        # entity_we_want_to_splice_onto then return None to end the recursive
        # branch

        assert entity_we_want_to_splice_onto is path[-2]

        if entity_inside_join_structure is False:
            assert isinstance(join_obj, orm_util._ORMJoin)

        if isinstance(join_obj, sql.selectable.FromGrouping):
            # FromGrouping - continue descending into the structure
            return self._splice_nested_inner_join(
                path,
                entity_we_want_to_splice_onto,
                join_obj.element,
                clauses,
                onclause,
                extra_criteria,
                entity_inside_join_structure,
            )
        elif isinstance(join_obj, orm_util._ORMJoin):
            # _ORMJoin - continue descending into the structure

            join_right_path = join_obj._right_memo

            # see if right side of join is viable
            target_join = self._splice_nested_inner_join(
                path,
                entity_we_want_to_splice_onto,
                join_obj.right,
                clauses,
                onclause,
                extra_criteria,
                entity_inside_join_structure=(
                    join_right_path[-1].mapper
                    if join_right_path is not None
                    else None
                ),
            )

            if target_join is not None:
                # for a right splice, attempt to flatten out
                # a JOIN b JOIN c JOIN .. to avoid needless
                # parenthesis nesting
                if not join_obj.isouter and not target_join.isouter:
                    eagerjoin = join_obj._splice_into_center(target_join)
                else:
                    eagerjoin = orm_util._ORMJoin(
                        join_obj.left,
                        target_join,
                        join_obj.onclause,
                        isouter=join_obj.isouter,
                        _left_memo=join_obj._left_memo,
                    )

                eagerjoin._target_adapter = target_join._target_adapter
                return eagerjoin

            else:
                # see if left side of join is viable
                target_join = self._splice_nested_inner_join(
                    path,
                    entity_we_want_to_splice_onto,
                    join_obj.left,
                    clauses,
                    onclause,
                    extra_criteria,
                    entity_inside_join_structure=join_obj._left_memo,
                    detected_existing_path=join_right_path,
                )

                if target_join is not None:
                    eagerjoin = orm_util._ORMJoin(
                        target_join,
                        join_obj.right,
                        join_obj.onclause,
                        isouter=join_obj.isouter,
                        _right_memo=join_obj._right_memo,
                    )
                    eagerjoin._target_adapter = target_join._target_adapter
                    return eagerjoin

            # neither side viable, return None, or fail if this was the top
            # most call
            if entity_inside_join_structure is False:
                assert (
                    False
                ), "assertion failed attempting to produce joined eager loads"
            return None

        # reached an endpoint (e.g. a table that's mapped, or an alias of that
        # table).  determine if we can use this endpoint to splice onto

        # is this the entity we want to splice onto in the first place?
        if not entity_we_want_to_splice_onto.isa(entity_inside_join_structure):
            return None

        # path check.  if we know the path how this join endpoint got here,
        # lets look at our path we are satisfying and see if we're in the
        # wrong place.  This is specifically for when our entity may
        # appear more than once in the path, issue #11449
        # updated in issue #11965.
        if detected_existing_path and len(detected_existing_path) > 2:
            # this assertion is currently based on how this call is made,
            # where given a join_obj, the call will have these parameters as
            # entity_inside_join_structure=join_obj._left_memo
            # and entity_inside_join_structure=join_obj._right_memo.mapper
            assert detected_existing_path[-3] is entity_inside_join_structure

            # from that, see if the path we are targeting matches the
            # "existing" path of this join all the way up to the midpoint
            # of this join object (e.g. the relationship).
            # if not, then this is not our target
            #
            # a test condition where this test is false looks like:
            #
            # desired splice:         Node->kind->Kind
            # path of desired splice: NodeGroup->nodes->Node->kind
            # path we've located:     NodeGroup->nodes->Node->common_node->Node
            #
            # above, because we want to splice kind->Kind onto
            # NodeGroup->nodes->Node, this is not our path because it actually
            # goes more steps than we want into self-referential
            # ->common_node->Node
            #
            # a test condition where this test is true looks like:
            #
            # desired splice:         B->c2s->C2
            # path of desired splice: A->bs->B->c2s
            # path we've located:     A->bs->B->c1s->C1
            #
            # above, we want to splice c2s->C2 onto B, and the located path
            # shows that the join ends with B->c1s->C1.  so we will
            # add another join onto that, which would create a "branch" that
            # we might represent in a pseudopath as:
            #
            # B->c1s->C1
            #  ->c2s->C2
            #
            # i.e. A JOIN B ON <bs> JOIN C1 ON <c1s>
            #                       JOIN C2 ON <c2s>
            #

            if detected_existing_path[0:-2] != path.path[0:-1]:
                return None

        return orm_util._ORMJoin(
            join_obj,
            clauses.aliased_insp,
            onclause,
            isouter=False,
            _left_memo=entity_inside_join_structure,
            _right_memo=path[path[-1].mapper],
            _extra_criteria=extra_criteria,
        )

    def _create_eager_adapter(self, context, result, adapter, path, loadopt):
        compile_state = context.compile_state

        user_defined_adapter = (
            self._init_user_defined_eager_proc(
                loadopt, compile_state, context.attributes
            )
            if loadopt
            else False
        )

        if user_defined_adapter is not False:
            decorator = user_defined_adapter
            # user defined eagerloads are part of the "primary"
            # portion of the load.
            # the adapters applied to the Query should be honored.
            if compile_state.compound_eager_adapter and decorator:
                decorator = decorator.wrap(
                    compile_state.compound_eager_adapter
                )
            elif compile_state.compound_eager_adapter:
                decorator = compile_state.compound_eager_adapter
        else:
            decorator = path.get(
                compile_state.attributes, "eager_row_processor"
            )
            if decorator is None:
                return False

        if self.mapper._result_has_identity_key(result, decorator):
            return decorator
        else:
            # no identity key - don't return a row
            # processor, will cause a degrade to lazy
            return False

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):

        if not context.compile_state.compile_options._enable_eagerloads:
            return

        if not self.parent.class_manager[self.key].impl.supports_population:
            raise sa_exc.InvalidRequestError(
                "'%s' does not support object "
                "population - eager loading cannot be applied." % self
            )

        if self.uselist:
            context.loaders_require_uniquing = True

        our_path = path[self.parent_property]

        eager_adapter = self._create_eager_adapter(
            context, result, adapter, our_path, loadopt
        )

        if eager_adapter is not False:
            key = self.key

            _instance = loading._instance_processor(
                query_entity,
                self.mapper,
                context,
                result,
                our_path[self.entity],
                eager_adapter,
            )

            if not self.uselist:
                self._create_scalar_loader(context, key, _instance, populators)
            else:
                self._create_collection_loader(
                    context, key, _instance, populators
                )
        else:
            self.parent_property._get_strategy(
                (("lazy", "select"),)
            ).create_row_processor(
                context,
                query_entity,
                path,
                loadopt,
                mapper,
                result,
                adapter,
                populators,
            )

    def _create_collection_loader(self, context, key, _instance, populators):
        def load_collection_from_joined_new_row(state, dict_, row):
            # note this must unconditionally clear out any existing collection.
            # an existing collection would be present only in the case of
            # populate_existing().
            collection = attributes.init_state_collection(state, dict_, key)
            result_list = util.UniqueAppender(
                collection, "append_without_event"
            )
            context.attributes[(state, key)] = result_list
            inst = _instance(row)
            if inst is not None:
                result_list.append(inst)

        def load_collection_from_joined_existing_row(state, dict_, row):
            if (state, key) in context.attributes:
                result_list = context.attributes[(state, key)]
            else:
                # appender_key can be absent from context.attributes
                # with isnew=False when self-referential eager loading
                # is used; the same instance may be present in two
                # distinct sets of result columns
                collection = attributes.init_state_collection(
                    state, dict_, key
                )
                result_list = util.UniqueAppender(
                    collection, "append_without_event"
                )
                context.attributes[(state, key)] = result_list
            inst = _instance(row)
            if inst is not None:
                result_list.append(inst)

        def load_collection_from_joined_exec(state, dict_, row):
            _instance(row)

        populators["new"].append(
            (self.key, load_collection_from_joined_new_row)
        )
        populators["existing"].append(
            (self.key, load_collection_from_joined_existing_row)
        )
        if context.invoke_all_eagers:
            populators["eager"].append(
                (self.key, load_collection_from_joined_exec)
            )

    def _create_scalar_loader(self, context, key, _instance, populators):
        def load_scalar_from_joined_new_row(state, dict_, row):
            # set a scalar object instance directly on the parent
            # object, bypassing InstrumentedAttribute event handlers.
            dict_[key] = _instance(row)

        def load_scalar_from_joined_existing_row(state, dict_, row):
            # call _instance on the row, even though the object has
            # been created, so that we further descend into properties
            existing = _instance(row)

            # conflicting value already loaded, this shouldn't happen
            if key in dict_:
                if existing is not dict_[key]:
                    util.warn(
                        "Multiple rows returned with "
                        "uselist=False for eagerly-loaded attribute '%s' "
                        % self
                    )
            else:
                # this case is when one row has multiple loads of the
                # same entity (e.g. via aliasing), one has an attribute
                # that the other doesn't.
                dict_[key] = existing

        def load_scalar_from_joined_exec(state, dict_, row):
            _instance(row)

        populators["new"].append((self.key, load_scalar_from_joined_new_row))
        populators["existing"].append(
            (self.key, load_scalar_from_joined_existing_row)
        )
        if context.invoke_all_eagers:
            populators["eager"].append(
                (self.key, load_scalar_from_joined_exec)
            )


@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy="selectin")
class SelectInLoader(PostLoader, util.MemoizedSlots):
    __slots__ = (
        "join_depth",
        "omit_join",
        "_parent_alias",
        "_query_info",
        "_fallback_query_info",
    )

    query_info = collections.namedtuple(
        "queryinfo",
        [
            "load_only_child",
            "load_with_join",
            "in_expr",
            "pk_cols",
            "zero_idx",
            "child_lookup_cols",
        ],
    )

    _chunksize = 500

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth
        is_m2o = self.parent_property.direction is interfaces.MANYTOONE

        if self.parent_property.omit_join is not None:
            self.omit_join = self.parent_property.omit_join
        else:
            lazyloader = self.parent_property._get_strategy(
                (("lazy", "select"),)
            )
            if is_m2o:
                self.omit_join = lazyloader.use_get
            else:
                self.omit_join = self.parent._get_clause[0].compare(
                    lazyloader._rev_lazywhere,
                    use_proxies=True,
                    compare_keys=False,
                    equivalents=self.parent._equivalent_columns,
                )

        if self.omit_join:
            if is_m2o:
                self._query_info = self._init_for_omit_join_m2o()
                self._fallback_query_info = self._init_for_join()
            else:
                self._query_info = self._init_for_omit_join()
        else:
            self._query_info = self._init_for_join()

    def _init_for_omit_join(self):
        pk_to_fk = dict(
            self.parent_property._join_condition.local_remote_pairs
        )
        pk_to_fk.update(
            (equiv, pk_to_fk[k])
            for k in list(pk_to_fk)
            for equiv in self.parent._equivalent_columns.get(k, ())
        )

        pk_cols = fk_cols = [
            pk_to_fk[col] for col in self.parent.primary_key if col in pk_to_fk
        ]
        if len(fk_cols) > 1:
            in_expr = sql.tuple_(*fk_cols)
            zero_idx = False
        else:
            in_expr = fk_cols[0]
            zero_idx = True

        return self.query_info(False, False, in_expr, pk_cols, zero_idx, None)

    def _init_for_omit_join_m2o(self):
        pk_cols = self.mapper.primary_key
        if len(pk_cols) > 1:
            in_expr = sql.tuple_(*pk_cols)
            zero_idx = False
        else:
            in_expr = pk_cols[0]
            zero_idx = True

        lazyloader = self.parent_property._get_strategy((("lazy", "select"),))
        lookup_cols = [lazyloader._equated_columns[pk] for pk in pk_cols]

        return self.query_info(
            True, False, in_expr, pk_cols, zero_idx, lookup_cols
        )

    def _init_for_join(self):
        self._parent_alias = AliasedClass(self.parent.class_)
        pa_insp = inspect(self._parent_alias)
        pk_cols = [
            pa_insp._adapt_element(col) for col in self.parent.primary_key
        ]
        if len(pk_cols) > 1:
            in_expr = sql.tuple_(*pk_cols)
            zero_idx = False
        else:
            in_expr = pk_cols[0]
            zero_idx = True
        return self.query_info(False, True, in_expr, pk_cols, zero_idx, None)

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy(
            (("lazy", "select"),)
        ).init_class_attribute(mapper)

    def create_row_processor(
        self,
        context,
        query_entity,
        path,
        loadopt,
        mapper,
        result,
        adapter,
        populators,
    ):
        if context.refresh_state:
            return self._immediateload_create_row_processor(
                context,
                query_entity,
                path,
                loadopt,
                mapper,
                result,
                adapter,
                populators,
            )

        (
            effective_path,
            run_loader,
            execution_options,
            recursion_depth,
        ) = self._setup_for_recursion(
            context, path, loadopt, join_depth=self.join_depth
        )

        if not run_loader:
            return

        if not context.compile_state.compile_options._enable_eagerloads:
            return

        if not self.parent.class_manager[self.key].impl.supports_population:
            raise sa_exc.InvalidRequestError(
                "'%s' does not support object "
                "population - eager loading cannot be applied." % self
            )

        # a little dance here as the "path" is still something that only
        # semi-tracks the exact series of things we are loading, still not
        # telling us about with_polymorphic() and stuff like that when it's at
        # the root..  the initial MapperEntity is more accurate for this case.
        if len(path) == 1:
            if not orm_util._entity_isa(query_entity.entity_zero, self.parent):
                return
        elif not orm_util._entity_isa(path[-1], self.parent):
            return

        selectin_path = effective_path

        path_w_prop = path[self.parent_property]

        # build up a path indicating the path from the leftmost
        # entity to the thing we're subquery loading.
        with_poly_entity = path_w_prop.get(
            context.attributes, "path_with_polymorphic", None
        )
        if with_poly_entity is not None:
            effective_entity = inspect(with_poly_entity)
        else:
            effective_entity = self.entity

        loading.PostLoad.callable_for_path(
            context,
            selectin_path,
            self.parent,
            self.parent_property,
            self._load_for_path,
            effective_entity,
            loadopt,
            recursion_depth,
            execution_options,
        )

    def _load_for_path(
        self,
        context,
        path,
        states,
        load_only,
        effective_entity,
        loadopt,
        recursion_depth,
        execution_options,
    ):
        if load_only and self.key not in load_only:
            return

        query_info = self._query_info

        if query_info.load_only_child:
            our_states = collections.defaultdict(list)
            none_states = []

            mapper = self.parent

            for state, overwrite in states:
                state_dict = state.dict
                related_ident = tuple(
                    mapper._get_state_attr_by_column(
                        state,
                        state_dict,
                        lk,
                        passive=attributes.PASSIVE_NO_FETCH,
                    )
                    for lk in query_info.child_lookup_cols
                )
                # if the loaded parent objects do not have the foreign key
                # to the related item loaded, then degrade into the joined
                # version of selectinload
                if LoaderCallableStatus.PASSIVE_NO_RESULT in related_ident:
                    query_info = self._fallback_query_info
                    break

                # organize states into lists keyed to particular foreign
                # key values.
                if None not in related_ident:
                    our_states[related_ident].append(
                        (state, state_dict, overwrite)
                    )
                else:
                    # For FK values that have None, add them to a
                    # separate collection that will be populated separately
                    none_states.append((state, state_dict, overwrite))

        # note the above conditional may have changed query_info
        if not query_info.load_only_child:
            our_states = [
                (state.key[1], state, state.dict, overwrite)
                for state, overwrite in states
            ]

        pk_cols = query_info.pk_cols
        in_expr = query_info.in_expr

        if not query_info.load_with_join:
            # in "omit join" mode, the primary key column and the
            # "in" expression are in terms of the related entity.  So
            # if the related entity is polymorphic or otherwise aliased,
            # we need to adapt our "pk_cols" and "in_expr" to that
            # entity.   in non-"omit join" mode, these are against the
            # parent entity and do not need adaption.
            if effective_entity.is_aliased_class:
                pk_cols = [
                    effective_entity._adapt_element(col) for col in pk_cols
                ]
                in_expr = effective_entity._adapt_element(in_expr)

        bundle_ent = orm_util.Bundle("pk", *pk_cols)
        bundle_sql = bundle_ent.__clause_element__()

        entity_sql = effective_entity.__clause_element__()
        q = Select._create_raw_select(
            _raw_columns=[bundle_sql, entity_sql],
            _label_style=LABEL_STYLE_TABLENAME_PLUS_COL,
            _compile_options=ORMCompileState.default_compile_options,
            _propagate_attrs={
                "compile_state_plugin": "orm",
                "plugin_subject": effective_entity,
            },
        )

        if not query_info.load_with_join:
            # the Bundle we have in the "omit_join" case is against raw, non
            # annotated columns, so to ensure the Query knows its primary
            # entity, we add it explicitly.  If we made the Bundle against
            # annotated columns, we hit a performance issue in this specific
            # case, which is detailed in issue #4347.
            q = q.select_from(effective_entity)
        else:
            # in the non-omit_join case, the Bundle is against the annotated/
            # mapped column of the parent entity, but the #4347 issue does not
            # occur in this case.
            q = q.select_from(self._parent_alias).join(
                getattr(self._parent_alias, self.parent_property.key).of_type(
                    effective_entity
                )
            )

        q = q.filter(in_expr.in_(sql.bindparam("primary_keys")))

        # a test which exercises what these comments talk about is
        # test_selectin_relations.py -> test_twolevel_selectin_w_polymorphic
        #
        # effective_entity above is given to us in terms of the cached
        # statement, namely this one:
        orig_query = context.compile_state.select_statement

        # the actual statement that was requested is this one:
        #  context_query = context.user_passed_query
        #
        # that's not the cached one, however.  So while it is of the identical
        # structure, if it has entities like AliasedInsp, which we get from
        # aliased() or with_polymorphic(), the AliasedInsp will likely be a
        # different object identity each time, and will not match up
        # hashing-wise to the corresponding AliasedInsp that's in the
        # cached query, meaning it won't match on paths and loader lookups
        # and loaders like this one will be skipped if it is used in options.
        #
        # as it turns out, standard loader options like selectinload(),
        # lazyload() that have a path need
        # to come from the cached query so that the AliasedInsp etc. objects
        # that are in the query line up with the object that's in the path
        # of the strategy object. however other options like
        # with_loader_criteria() that doesn't have a path (has a fixed entity)
        # and needs to have access to the latest closure state in order to
        # be correct, we need to use the uncached one.
        #
        # as of #8399 we let the loader option itself figure out what it
        # wants to do given cached and uncached version of itself.

        effective_path = path[self.parent_property]

        if orig_query is context.user_passed_query:
            new_options = orig_query._with_options
        else:
            cached_options = orig_query._with_options
            uncached_options = context.user_passed_query._with_options

            # propagate compile state options from the original query,
            # updating their "extra_criteria" as necessary.
            # note this will create a different cache key than
            # "orig" options if extra_criteria is present, because the copy
            # of extra_criteria will have different boundparam than that of
            # the QueryableAttribute in the path
            new_options = [
                orig_opt._adapt_cached_option_to_uncached_option(
                    context, uncached_opt
                )
                for orig_opt, uncached_opt in zip(
                    cached_options, uncached_options
                )
            ]

        if loadopt and loadopt._extra_criteria:
            new_options += (
                orm_util.LoaderCriteriaOption(
                    effective_entity,
                    loadopt._generate_extra_criteria(context),
                ),
            )

        if recursion_depth is not None:
            effective_path = effective_path._truncate_recursive()

        q = q.options(*new_options)

        q = q._update_compile_options({"_current_path": effective_path})
        if context.populate_existing:
            q = q.execution_options(populate_existing=True)

        if self.parent_property.order_by:
            if not query_info.load_with_join:
                eager_order_by = self.parent_property.order_by
                if effective_entity.is_aliased_class:
                    eager_order_by = [
                        effective_entity._adapt_element(elem)
                        for elem in eager_order_by
                    ]
                q = q.order_by(*eager_order_by)
            else:

                def _setup_outermost_orderby(compile_context):
                    compile_context.eager_order_by += tuple(
                        util.to_list(self.parent_property.order_by)
                    )

                q = q._add_context_option(
                    _setup_outermost_orderby, self.parent_property
                )

        if query_info.load_only_child:
            self._load_via_child(
                our_states,
                none_states,
                query_info,
                q,
                context,
                execution_options,
            )
        else:
            self._load_via_parent(
                our_states, query_info, q, context, execution_options
            )

    def _load_via_child(
        self,
        our_states,
        none_states,
        query_info,
        q,
        context,
        execution_options,
    ):
        uselist = self.uselist

        # this sort is really for the benefit of the unit tests
        our_keys = sorted(our_states)
        while our_keys:
            chunk = our_keys[0 : self._chunksize]
            our_keys = our_keys[self._chunksize :]
            data = {
                k: v
                for k, v in context.session.execute(
                    q,
                    params={
                        "primary_keys": [
                            key[0] if query_info.zero_idx else key
                            for key in chunk
                        ]
                    },
                    execution_options=execution_options,
                ).unique()
            }

            for key in chunk:
                # for a real foreign key and no concurrent changes to the
                # DB while running this method, "key" is always present in
                # data.  However, for primaryjoins without real foreign keys
                # a non-None primaryjoin condition may still refer to no
                # related object.
                related_obj = data.get(key, None)
                for state, dict_, overwrite in our_states[key]:
                    if not overwrite and self.key in dict_:
                        continue

                    state.get_impl(self.key).set_committed_value(
                        state,
                        dict_,
                        related_obj if not uselist else [related_obj],
                    )
        # populate none states with empty value / collection
        for state, dict_, overwrite in none_states:
            if not overwrite and self.key in dict_:
                continue

            # note it's OK if this is a uselist=True attribute, the empty
            # collection will be populated
            state.get_impl(self.key).set_committed_value(state, dict_, None)

    def _load_via_parent(
        self, our_states, query_info, q, context, execution_options
    ):
        uselist = self.uselist
        _empty_result = () if uselist else None

        while our_states:
            chunk = our_states[0 : self._chunksize]
            our_states = our_states[self._chunksize :]

            primary_keys = [
                key[0] if query_info.zero_idx else key
                for key, state, state_dict, overwrite in chunk
            ]

            data = collections.defaultdict(list)
            for k, v in itertools.groupby(
                context.session.execute(
                    q,
                    params={"primary_keys": primary_keys},
                    execution_options=execution_options,
                ).unique(),
                lambda x: x[0],
            ):
                data[k].extend(vv[1] for vv in v)

            for key, state, state_dict, overwrite in chunk:
                if not overwrite and self.key in state_dict:
                    continue

                collection = data.get(key, _empty_result)

                if not uselist and collection:
                    if len(collection) > 1:
                        util.warn(
                            "Multiple rows returned with "
                            "uselist=False for eagerly-loaded "
                            "attribute '%s' " % self
                        )
                    state.get_impl(self.key).set_committed_value(
                        state, state_dict, collection[0]
                    )
                else:
                    # note that empty tuple set on uselist=False sets the
                    # value to None
                    state.get_impl(self.key).set_committed_value(
                        state, state_dict, collection
                    )


def single_parent_validator(desc, prop):
    def _do_check(state, value, oldvalue, initiator):
        if value is not None and initiator.key == prop.key:
            hasparent = initiator.hasparent(attributes.instance_state(value))
            if hasparent and oldvalue is not value:
                raise sa_exc.InvalidRequestError(
                    "Instance %s is already associated with an instance "
                    "of %s via its %s attribute, and is only allowed a "
                    "single parent."
                    % (orm_util.instance_str(value), state.class_, prop),
                    code="bbf1",
                )
        return value

    def append(state, value, initiator):
        return _do_check(state, value, None, initiator)

    def set_(state, value, oldvalue, initiator):
        return _do_check(state, value, oldvalue, initiator)

    event.listen(
        desc, "append", append, raw=True, retval=True, active_history=True
    )
    event.listen(desc, "set", set_, raw=True, retval=True, active_history=True)