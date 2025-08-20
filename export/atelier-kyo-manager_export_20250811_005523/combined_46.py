
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\strategy_options.py ===
# orm/strategy_options.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""

"""

from __future__ import annotations

import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self

_RELATIONSHIP_TOKEN: Final[Literal["relationship"]] = "relationship"
_COLUMN_TOKEN: Final[Literal["column"]] = "column"

_FN = TypeVar("_FN", bound="Callable[..., Any]")

if typing.TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _InternalEntityType
    from .context import _MapperEntity
    from .context import ORMCompileState
    from .context import QueryContext
    from .interfaces import _StrategyKey
    from .interfaces import MapperProperty
    from .interfaces import ORMOption
    from .mapper import Mapper
    from .path_registry import _PathRepresentation
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _FromClauseArgument
    from ..sql.cache_key import _CacheKeyTraversalType
    from ..sql.cache_key import CacheKey


_AttrType = Union[Literal["*"], "QueryableAttribute[Any]"]

_WildcardKeyType = Literal["relationship", "column"]
_StrategySpec = Dict[str, Any]
_OptsType = Dict[str, Any]
_AttrGroupType = Tuple[_AttrType, ...]


class _AbstractLoad(traversals.GenerativeOnTraversal, LoaderOption):
    __slots__ = ("propagate_to_loaders",)

    _is_strategy_option = True
    propagate_to_loaders: bool

    def contains_eager(
        self,
        attr: _AttrType,
        alias: Optional[_FromClauseArgument] = None,
        _is_chain: bool = False,
        _propagate_to_loaders: bool = False,
    ) -> Self:
        r"""Indicate that the given attribute should be eagerly loaded from
        columns stated manually in the query.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        The option is used in conjunction with an explicit join that loads
        the desired rows, i.e.::

            sess.query(Order).join(Order.user).options(contains_eager(Order.user))

        The above query would join from the ``Order`` entity to its related
        ``User`` entity, and the returned ``Order`` objects would have the
        ``Order.user`` attribute pre-populated.

        It may also be used for customizing the entries in an eagerly loaded
        collection; queries will normally want to use the
        :ref:`orm_queryguide_populate_existing` execution option assuming the
        primary collection of parent objects may already have been loaded::

            sess.query(User).join(User.addresses).filter(
                Address.email_address.like("%@aol.com")
            ).options(contains_eager(User.addresses)).populate_existing()

        See the section :ref:`contains_eager` for complete usage details.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`contains_eager`

        """
        if alias is not None:
            if not isinstance(alias, str):
                coerced_alias = coercions.expect(roles.FromClauseRole, alias)
            else:
                util.warn_deprecated(
                    "Passing a string name for the 'alias' argument to "
                    "'contains_eager()` is deprecated, and will not work in a "
                    "future release.  Please use a sqlalchemy.alias() or "
                    "sqlalchemy.orm.aliased() construct.",
                    version="1.4",
                )
                coerced_alias = alias

        elif getattr(attr, "_of_type", None):
            assert isinstance(attr, QueryableAttribute)
            ot: Optional[_InternalEntityType[Any]] = inspect(attr._of_type)
            assert ot is not None
            coerced_alias = ot.selectable
        else:
            coerced_alias = None

        cloned = self._set_relationship_strategy(
            attr,
            {"lazy": "joined"},
            propagate_to_loaders=_propagate_to_loaders,
            opts={"eager_from_alias": coerced_alias},
            _reconcile_to_other=True if _is_chain else None,
        )
        return cloned

    def load_only(self, *attrs: _AttrType, raiseload: bool = False) -> Self:
        r"""Indicate that for a particular entity, only the given list
        of column-based attribute names should be loaded; all others will be
        deferred.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        Example - given a class ``User``, load only the ``name`` and
        ``fullname`` attributes::

            session.query(User).options(load_only(User.name, User.fullname))

        Example - given a relationship ``User.addresses -> Address``, specify
        subquery loading for the ``User.addresses`` collection, but on each
        ``Address`` object load only the ``email_address`` attribute::

            session.query(User).options(
                subqueryload(User.addresses).load_only(Address.email_address)
            )

        For a statement that has multiple entities,
        the lead entity can be
        specifically referred to using the :class:`_orm.Load` constructor::

            stmt = (
                select(User, Address)
                .join(User.addresses)
                .options(
                    Load(User).load_only(User.name, User.fullname),
                    Load(Address).load_only(Address.email_address),
                )
            )

        When used together with the
        :ref:`populate_existing <orm_queryguide_populate_existing>`
        execution option only the attributes listed will be refreshed.

        :param \*attrs: Attributes to be loaded, all others will be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when a deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

         .. versionadded:: 2.0

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

        :param \*attrs: Attributes to be loaded, all others will be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when a deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

         .. versionadded:: 2.0

        """
        cloned = self._set_column_strategy(
            attrs,
            {"deferred": False, "instrument": True},
        )

        wildcard_strategy = {"deferred": True, "instrument": True}
        if raiseload:
            wildcard_strategy["raiseload"] = True

        cloned = cloned._set_column_strategy(
            ("*",),
            wildcard_strategy,
        )
        return cloned

    def joinedload(
        self,
        attr: _AttrType,
        innerjoin: Optional[bool] = None,
    ) -> Self:
        """Indicate that the given attribute should be loaded using joined
        eager loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        examples::

            # joined-load the "orders" collection on "User"
            select(User).options(joinedload(User.orders))

            # joined-load Order.items and then Item.keywords
            select(Order).options(joinedload(Order.items).joinedload(Item.keywords))

            # lazily load Order.items, but when Items are loaded,
            # joined-load the keywords collection
            select(Order).options(lazyload(Order.items).joinedload(Item.keywords))

        :param innerjoin: if ``True``, indicates that the joined eager load
         should use an inner join instead of the default of left outer join::

            select(Order).options(joinedload(Order.user, innerjoin=True))

        In order to chain multiple eager joins together where some may be
        OUTER and others INNER, right-nested joins are used to link them::

            select(A).options(
                joinedload(A.bs, innerjoin=False).joinedload(B.cs, innerjoin=True)
            )

        The above query, linking A.bs via "outer" join and B.cs via "inner"
        join would render the joins as "a LEFT OUTER JOIN (b JOIN c)". When
        using older versions of SQLite (< 3.7.16), this form of JOIN is
        translated to use full subqueries as this syntax is otherwise not
        directly supported.

        The ``innerjoin`` flag can also be stated with the term ``"unnested"``.
        This indicates that an INNER JOIN should be used, *unless* the join
        is linked to a LEFT OUTER JOIN to the left, in which case it
        will render as LEFT OUTER JOIN.  For example, supposing ``A.bs``
        is an outerjoin::

            select(A).options(joinedload(A.bs).joinedload(B.cs, innerjoin="unnested"))

        The above join will render as "a LEFT OUTER JOIN b LEFT OUTER JOIN c",
        rather than as "a LEFT OUTER JOIN (b JOIN c)".

        .. note:: The "unnested" flag does **not** affect the JOIN rendered
            from a many-to-many association table, e.g. a table configured as
            :paramref:`_orm.relationship.secondary`, to the target table; for
            correctness of results, these joins are always INNER and are
            therefore right-nested if linked to an OUTER join.

        .. note::

            The joins produced by :func:`_orm.joinedload` are **anonymously
            aliased**. The criteria by which the join proceeds cannot be
            modified, nor can the ORM-enabled :class:`_sql.Select` or legacy
            :class:`_query.Query` refer to these joins in any way, including
            ordering. See :ref:`zen_of_eager_loading` for further detail.

            To produce a specific SQL JOIN which is explicitly available, use
            :meth:`_sql.Select.join` and :meth:`_query.Query.join`. To combine
            explicit JOINs with eager loading of collections, use
            :func:`_orm.contains_eager`; see :ref:`contains_eager`.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`joined_eager_loading`

        """  # noqa: E501
        loader = self._set_relationship_strategy(
            attr,
            {"lazy": "joined"},
            opts=(
                {"innerjoin": innerjoin}
                if innerjoin is not None
                else util.EMPTY_DICT
            ),
        )
        return loader

    def subqueryload(self, attr: _AttrType) -> Self:
        """Indicate that the given attribute should be loaded using
        subquery eager loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        examples::

            # subquery-load the "orders" collection on "User"
            select(User).options(subqueryload(User.orders))

            # subquery-load Order.items and then Item.keywords
            select(Order).options(
                subqueryload(Order.items).subqueryload(Item.keywords)
            )

            # lazily load Order.items, but when Items are loaded,
            # subquery-load the keywords collection
            select(Order).options(lazyload(Order.items).subqueryload(Item.keywords))

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`subquery_eager_loading`

        """
        return self._set_relationship_strategy(attr, {"lazy": "subquery"})

    def selectinload(
        self,
        attr: _AttrType,
        recursion_depth: Optional[int] = None,
    ) -> Self:
        """Indicate that the given attribute should be loaded using
        SELECT IN eager loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        examples::

            # selectin-load the "orders" collection on "User"
            select(User).options(selectinload(User.orders))

            # selectin-load Order.items and then Item.keywords
            select(Order).options(
                selectinload(Order.items).selectinload(Item.keywords)
            )

            # lazily load Order.items, but when Items are loaded,
            # selectin-load the keywords collection
            select(Order).options(lazyload(Order.items).selectinload(Item.keywords))

        :param recursion_depth: optional int; when set to a positive integer
         in conjunction with a self-referential relationship,
         indicates "selectin" loading will continue that many levels deep
         automatically until no items are found.

         .. note:: The :paramref:`_orm.selectinload.recursion_depth` option
            currently supports only self-referential relationships.  There
            is not yet an option to automatically traverse recursive structures
            with more than one relationship involved.

            Additionally, the :paramref:`_orm.selectinload.recursion_depth`
            parameter is new and experimental and should be treated as "alpha"
            status for the 2.0 series.

         .. versionadded:: 2.0 added
            :paramref:`_orm.selectinload.recursion_depth`


        .. seealso::

            :ref:`loading_toplevel`

            :ref:`selectin_eager_loading`

        """
        return self._set_relationship_strategy(
            attr,
            {"lazy": "selectin"},
            opts={"recursion_depth": recursion_depth},
        )

    def lazyload(self, attr: _AttrType) -> Self:
        """Indicate that the given attribute should be loaded using "lazy"
        loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`lazy_loading`

        """
        return self._set_relationship_strategy(attr, {"lazy": "select"})

    def immediateload(
        self,
        attr: _AttrType,
        recursion_depth: Optional[int] = None,
    ) -> Self:
        """Indicate that the given attribute should be loaded using
        an immediate load with a per-attribute SELECT statement.

        The load is achieved using the "lazyloader" strategy and does not
        fire off any additional eager loaders.

        The :func:`.immediateload` option is superseded in general
        by the :func:`.selectinload` option, which performs the same task
        more efficiently by emitting a SELECT for all loaded objects.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        :param recursion_depth: optional int; when set to a positive integer
         in conjunction with a self-referential relationship,
         indicates "selectin" loading will continue that many levels deep
         automatically until no items are found.

         .. note:: The :paramref:`_orm.immediateload.recursion_depth` option
            currently supports only self-referential relationships.  There
            is not yet an option to automatically traverse recursive structures
            with more than one relationship involved.

         .. warning:: This parameter is new and experimental and should be
            treated as "alpha" status

         .. versionadded:: 2.0 added
            :paramref:`_orm.immediateload.recursion_depth`


        .. seealso::

            :ref:`loading_toplevel`

            :ref:`selectin_eager_loading`

        """
        loader = self._set_relationship_strategy(
            attr,
            {"lazy": "immediate"},
            opts={"recursion_depth": recursion_depth},
        )
        return loader

    def noload(self, attr: _AttrType) -> Self:
        """Indicate that the given relationship attribute should remain
        unloaded.

        The relationship attribute will return ``None`` when accessed without
        producing any loading effect.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        :func:`_orm.noload` applies to :func:`_orm.relationship` attributes
        only.

        .. legacy:: The :func:`_orm.noload` option is **legacy**.  As it
           forces collections to be empty, which invariably leads to
           non-intuitive and difficult to predict results.  There are no
           legitimate uses for this option in modern SQLAlchemy.

        .. seealso::

            :ref:`loading_toplevel`

        """

        return self._set_relationship_strategy(attr, {"lazy": "noload"})

    def raiseload(self, attr: _AttrType, sql_only: bool = False) -> Self:
        """Indicate that the given attribute should raise an error if accessed.

        A relationship attribute configured with :func:`_orm.raiseload` will
        raise an :exc:`~sqlalchemy.exc.InvalidRequestError` upon access. The
        typical way this is useful is when an application is attempting to
        ensure that all relationship attributes that are accessed in a
        particular context would have been already loaded via eager loading.
        Instead of having to read through SQL logs to ensure lazy loads aren't
        occurring, this strategy will cause them to raise immediately.

        :func:`_orm.raiseload` applies to :func:`_orm.relationship` attributes
        only. In order to apply raise-on-SQL behavior to a column-based
        attribute, use the :paramref:`.orm.defer.raiseload` parameter on the
        :func:`.defer` loader option.

        :param sql_only: if True, raise only if the lazy load would emit SQL,
         but not if it is only checking the identity map, or determining that
         the related value should just be None due to missing keys. When False,
         the strategy will raise for all varieties of relationship loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`prevent_lazy_with_raiseload`

            :ref:`orm_queryguide_deferred_raiseload`

        """

        return self._set_relationship_strategy(
            attr, {"lazy": "raise_on_sql" if sql_only else "raise"}
        )

    def defaultload(self, attr: _AttrType) -> Self:
        """Indicate an attribute should load using its predefined loader style.

        The behavior of this loading option is to not change the current
        loading style of the attribute, meaning that the previously configured
        one is used or, if no previous style was selected, the default
        loading will be used.

        This method is used to link to other loader options further into
        a chain of attributes without altering the loader style of the links
        along the chain.  For example, to set joined eager loading for an
        element of an element::

            session.query(MyClass).options(
                defaultload(MyClass.someattribute).joinedload(
                    MyOtherClass.someotherattribute
                )
            )

        :func:`.defaultload` is also useful for setting column-level options on
        a related class, namely that of :func:`.defer` and :func:`.undefer`::

            session.scalars(
                select(MyClass).options(
                    defaultload(MyClass.someattribute)
                    .defer("some_column")
                    .undefer("some_other_column")
                )
            )

        .. seealso::

            :ref:`orm_queryguide_relationship_sub_options`

            :meth:`_orm.Load.options`

        """
        return self._set_relationship_strategy(attr, None)

    def defer(self, key: _AttrType, raiseload: bool = False) -> Self:
        r"""Indicate that the given column-oriented attribute should be
        deferred, e.g. not loaded until accessed.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        e.g.::

            from sqlalchemy.orm import defer

            session.query(MyClass).options(
                defer(MyClass.attribute_one), defer(MyClass.attribute_two)
            )

        To specify a deferred load of an attribute on a related class,
        the path can be specified one token at a time, specifying the loading
        style for each link along the chain.  To leave the loading style
        for a link unchanged, use :func:`_orm.defaultload`::

            session.query(MyClass).options(
                defaultload(MyClass.someattr).defer(RelatedClass.some_column)
            )

        Multiple deferral options related to a relationship can be bundled
        at once using :meth:`_orm.Load.options`::


            select(MyClass).options(
                defaultload(MyClass.someattr).options(
                    defer(RelatedClass.some_column),
                    defer(RelatedClass.some_other_column),
                    defer(RelatedClass.another_column),
                )
            )

        :param key: Attribute to be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when the deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

        .. versionadded:: 1.4

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

            :func:`_orm.load_only`

            :func:`_orm.undefer`

        """
        strategy = {"deferred": True, "instrument": True}
        if raiseload:
            strategy["raiseload"] = True
        return self._set_column_strategy((key,), strategy)

    def undefer(self, key: _AttrType) -> Self:
        r"""Indicate that the given column-oriented attribute should be
        undeferred, e.g. specified within the SELECT statement of the entity
        as a whole.

        The column being undeferred is typically set up on the mapping as a
        :func:`.deferred` attribute.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        Examples::

            # undefer two columns
            session.query(MyClass).options(
                undefer(MyClass.col1), undefer(MyClass.col2)
            )

            # undefer all columns specific to a single class using Load + *
            session.query(MyClass, MyOtherClass).options(Load(MyClass).undefer("*"))

            # undefer a column on a related object
            select(MyClass).options(defaultload(MyClass.items).undefer(MyClass.text))

        :param key: Attribute to be undeferred.

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

            :func:`_orm.defer`

            :func:`_orm.undefer_group`

        """  # noqa: E501
        return self._set_column_strategy(
            (key,), {"deferred": False, "instrument": True}
        )

    def undefer_group(self, name: str) -> Self:
        """Indicate that columns within the given deferred group name should be
        undeferred.

        The columns being undeferred are set up on the mapping as
        :func:`.deferred` attributes and include a "group" name.

        E.g::

            session.query(MyClass).options(undefer_group("large_attrs"))

        To undefer a group of attributes on a related entity, the path can be
        spelled out using relationship loader options, such as
        :func:`_orm.defaultload`::

            select(MyClass).options(
                defaultload("someattr").undefer_group("large_attrs")
            )

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

            :func:`_orm.defer`

            :func:`_orm.undefer`

        """
        return self._set_column_strategy(
            (_WILDCARD_TOKEN,), None, {f"undefer_group_{name}": True}
        )

    def with_expression(
        self,
        key: _AttrType,
        expression: _ColumnExpressionArgument[Any],
    ) -> Self:
        r"""Apply an ad-hoc SQL expression to a "deferred expression"
        attribute.

        This option is used in conjunction with the
        :func:`_orm.query_expression` mapper-level construct that indicates an
        attribute which should be the target of an ad-hoc SQL expression.

        E.g.::

            stmt = select(SomeClass).options(
                with_expression(SomeClass.x_y_expr, SomeClass.x + SomeClass.y)
            )

        .. versionadded:: 1.2

        :param key: Attribute to be populated

        :param expr: SQL expression to be applied to the attribute.

        .. seealso::

            :ref:`orm_queryguide_with_expression` - background and usage
            examples

        """

        expression = _orm_full_deannotate(
            coercions.expect(roles.LabeledColumnExprRole, expression)
        )

        return self._set_column_strategy(
            (key,), {"query_expression": True}, extra_criteria=(expression,)
        )

    def selectin_polymorphic(self, classes: Iterable[Type[Any]]) -> Self:
        """Indicate an eager load should take place for all attributes
        specific to a subclass.

        This uses an additional SELECT with IN against all matched primary
        key values, and is the per-query analogue to the ``"selectin"``
        setting on the :paramref:`.mapper.polymorphic_load` parameter.

        .. versionadded:: 1.2

        .. seealso::

            :ref:`polymorphic_selectin`

        """
        self = self._set_class_strategy(
            {"selectinload_polymorphic": True},
            opts={
                "entities": tuple(
                    sorted((inspect(cls) for cls in classes), key=id)
                )
            },
        )
        return self

    @overload
    def _coerce_strat(self, strategy: _StrategySpec) -> _StrategyKey: ...

    @overload
    def _coerce_strat(self, strategy: Literal[None]) -> None: ...

    def _coerce_strat(
        self, strategy: Optional[_StrategySpec]
    ) -> Optional[_StrategyKey]:
        if strategy is not None:
            strategy_key = tuple(sorted(strategy.items()))
        else:
            strategy_key = None
        return strategy_key

    @_generative
    def _set_relationship_strategy(
        self,
        attr: _AttrType,
        strategy: Optional[_StrategySpec],
        propagate_to_loaders: bool = True,
        opts: Optional[_OptsType] = None,
        _reconcile_to_other: Optional[bool] = None,
    ) -> Self:
        strategy_key = self._coerce_strat(strategy)

        self._clone_for_bind_strategy(
            (attr,),
            strategy_key,
            _RELATIONSHIP_TOKEN,
            opts=opts,
            propagate_to_loaders=propagate_to_loaders,
            reconcile_to_other=_reconcile_to_other,
        )
        return self

    @_generative
    def _set_column_strategy(
        self,
        attrs: Tuple[_AttrType, ...],
        strategy: Optional[_StrategySpec],
        opts: Optional[_OptsType] = None,
        extra_criteria: Optional[Tuple[Any, ...]] = None,
    ) -> Self:
        strategy_key = self._coerce_strat(strategy)

        self._clone_for_bind_strategy(
            attrs,
            strategy_key,
            _COLUMN_TOKEN,
            opts=opts,
            attr_group=attrs,
            extra_criteria=extra_criteria,
        )
        return self

    @_generative
    def _set_generic_strategy(
        self,
        attrs: Tuple[_AttrType, ...],
        strategy: _StrategySpec,
        _reconcile_to_other: Optional[bool] = None,
    ) -> Self:
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy(
            attrs,
            strategy_key,
            None,
            propagate_to_loaders=True,
            reconcile_to_other=_reconcile_to_other,
        )
        return self

    @_generative
    def _set_class_strategy(
        self, strategy: _StrategySpec, opts: _OptsType
    ) -> Self:
        strategy_key = self._coerce_strat(strategy)

        self._clone_for_bind_strategy(None, strategy_key, None, opts=opts)
        return self

    def _apply_to_parent(self, parent: Load) -> None:
        """apply this :class:`_orm._AbstractLoad` object as a sub-option o
        a :class:`_orm.Load` object.

        Implementation is provided by subclasses.

        """
        raise NotImplementedError()

    def options(self, *opts: _AbstractLoad) -> Self:
        r"""Apply a series of options as sub-options to this
        :class:`_orm._AbstractLoad` object.

        Implementation is provided by subclasses.

        """
        raise NotImplementedError()

    def _clone_for_bind_strategy(
        self,
        attrs: Optional[Tuple[_AttrType, ...]],
        strategy: Optional[_StrategyKey],
        wildcard_key: Optional[_WildcardKeyType],
        opts: Optional[_OptsType] = None,
        attr_group: Optional[_AttrGroupType] = None,
        propagate_to_loaders: bool = True,
        reconcile_to_other: Optional[bool] = None,
        extra_criteria: Optional[Tuple[Any, ...]] = None,
    ) -> Self:
        raise NotImplementedError()

    def process_compile_state_replaced_entities(
        self,
        compile_state: ORMCompileState,
        mapper_entities: Sequence[_MapperEntity],
    ) -> None:
        if not compile_state.compile_options._enable_eagerloads:
            return

        # process is being run here so that the options given are validated
        # against what the lead entities were, as well as to accommodate
        # for the entities having been replaced with equivalents
        self._process(
            compile_state,
            mapper_entities,
            not bool(compile_state.current_path),
        )

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        if not compile_state.compile_options._enable_eagerloads:
            return

        self._process(
            compile_state,
            compile_state._lead_mapper_entities,
            not bool(compile_state.current_path)
            and not compile_state.compile_options._for_refresh_state,
        )

    def _process(
        self,
        compile_state: ORMCompileState,
        mapper_entities: Sequence[_MapperEntity],
        raiseerr: bool,
    ) -> None:
        """implemented by subclasses"""
        raise NotImplementedError()

    @classmethod
    def _chop_path(
        cls,
        to_chop: _PathRepresentation,
        path: PathRegistry,
        debug: bool = False,
    ) -> Optional[_PathRepresentation]:
        i = -1

        for i, (c_token, p_token) in enumerate(
            zip(to_chop, path.natural_path)
        ):
            if isinstance(c_token, str):
                if i == 0 and (
                    c_token.endswith(f":{_DEFAULT_TOKEN}")
                    or c_token.endswith(f":{_WILDCARD_TOKEN}")
                ):
                    return to_chop
                elif (
                    c_token != f"{_RELATIONSHIP_TOKEN}:{_WILDCARD_TOKEN}"
                    and c_token != p_token.key  # type: ignore
                ):
                    return None

            if c_token is p_token:
                continue
            elif (
                isinstance(c_token, InspectionAttr)
                and insp_is_mapper(c_token)
                and insp_is_mapper(p_token)
                and c_token.isa(p_token)
            ):
                continue

            else:
                return None
        return to_chop[i + 1 :]


class Load(_AbstractLoad):
    """Represents loader options which modify the state of a
    ORM-enabled :class:`_sql.Select` or a legacy :class:`_query.Query` in
    order to affect how various mapped attributes are loaded.

    The :class:`_orm.Load` object is in most cases used implicitly behind the
    scenes when one makes use of a query option like :func:`_orm.joinedload`,
    :func:`_orm.defer`, or similar.   It typically is not instantiated directly
    except for in some very specific cases.

    .. seealso::

        :ref:`orm_queryguide_relationship_per_entity_wildcard` - illustrates an
        example where direct use of :class:`_orm.Load` may be useful

    """

    __slots__ = (
        "path",
        "context",
        "additional_source_entities",
    )

    _traverse_internals = [
        ("path", visitors.ExtendedInternalTraversal.dp_has_cache_key),
        (
            "context",
            visitors.InternalTraversal.dp_has_cache_key_list,
        ),
        ("propagate_to_loaders", visitors.InternalTraversal.dp_boolean),
        (
            "additional_source_entities",
            visitors.InternalTraversal.dp_has_cache_key_list,
        ),
    ]
    _cache_key_traversal = None

    path: PathRegistry
    context: Tuple[_LoadElement, ...]
    additional_source_entities: Tuple[_InternalEntityType[Any], ...]

    def __init__(self, entity: _EntityType[Any]):
        insp = cast("Union[Mapper[Any], AliasedInsp[Any]]", inspect(entity))
        insp._post_inspect

        self.path = insp._path_registry
        self.context = ()
        self.propagate_to_loaders = False
        self.additional_source_entities = ()

    def __str__(self) -> str:
        return f"Load({self.path[0]})"

    @classmethod
    def _construct_for_existing_path(
        cls, path: AbstractEntityRegistry
    ) -> Load:
        load = cls.__new__(cls)
        load.path = path
        load.context = ()
        load.propagate_to_loaders = False
        load.additional_source_entities = ()
        return load

    def _adapt_cached_option_to_uncached_option(
        self, context: QueryContext, uncached_opt: ORMOption
    ) -> ORMOption:
        if uncached_opt is self:
            return self
        return self._adjust_for_extra_criteria(context)

    def _prepend_path(self, path: PathRegistry) -> Load:
        cloned = self._clone()
        cloned.context = tuple(
            element._prepend_path(path) for element in self.context
        )
        return cloned

    def _adjust_for_extra_criteria(self, context: QueryContext) -> Load:
        """Apply the current bound parameters in a QueryContext to all
        occurrences "extra_criteria" stored within this ``Load`` object,
        returning a new instance of this ``Load`` object.

        """

        # avoid generating cache keys for the queries if we don't
        # actually have any extra_criteria options, which is the
        # common case
        for value in self.context:
            if value._extra_criteria:
                break
        else:
            return self

        replacement_cache_key = context.user_passed_query._generate_cache_key()

        if replacement_cache_key is None:
            return self

        orig_query = context.compile_state.select_statement
        orig_cache_key = orig_query._generate_cache_key()
        assert orig_cache_key is not None

        def process(
            opt: _LoadElement,
            replacement_cache_key: CacheKey,
            orig_cache_key: CacheKey,
        ) -> _LoadElement:
            cloned_opt = opt._clone()

            cloned_opt._extra_criteria = tuple(
                replacement_cache_key._apply_params_to_element(
                    orig_cache_key, crit
                )
                for crit in cloned_opt._extra_criteria
            )

            return cloned_opt

        cloned = self._clone()
        cloned.context = tuple(
            (
                process(value, replacement_cache_key, orig_cache_key)
                if value._extra_criteria
                else value
            )
            for value in self.context
        )
        return cloned

    def _reconcile_query_entities_with_us(self, mapper_entities, raiseerr):
        """called at process time to allow adjustment of the root
        entity inside of _LoadElement objects.

        """
        path = self.path

        for ent in mapper_entities:
            ezero = ent.entity_zero
            if ezero and orm_util._entity_corresponds_to(
                # technically this can be a token also, but this is
                # safe to pass to _entity_corresponds_to()
                ezero,
                cast("_InternalEntityType[Any]", path[0]),
            ):
                return ezero

        return None

    def _process(
        self,
        compile_state: ORMCompileState,
        mapper_entities: Sequence[_MapperEntity],
        raiseerr: bool,
    ) -> None:
        reconciled_lead_entity = self._reconcile_query_entities_with_us(
            mapper_entities, raiseerr
        )

        # if the context has a current path, this is a lazy load
        has_current_path = bool(compile_state.compile_options._current_path)

        for loader in self.context:
            # issue #11292
            # historically, propagate_to_loaders was only considered at
            # object loading time, whether or not to carry along options
            # onto an object's loaded state where it would be used by lazyload.
            # however, the defaultload() option needs to propagate in case
            # its sub-options propagate_to_loaders, but its sub-options
            # that dont propagate should not be applied for lazy loaders.
            # so we check again
            if has_current_path and not loader.propagate_to_loaders:
                continue
            loader.process_compile_state(
                self,
                compile_state,
                mapper_entities,
                reconciled_lead_entity,
                raiseerr,
            )

    def _apply_to_parent(self, parent: Load) -> None:
        """apply this :class:`_orm.Load` object as a sub-option of another
        :class:`_orm.Load` object.

        This method is used by the :meth:`_orm.Load.options` method.

        """
        cloned = self._generate()

        assert cloned.propagate_to_loaders == self.propagate_to_loaders

        if not any(
            orm_util._entity_corresponds_to_use_path_impl(
                elem, cloned.path.odd_element(0)
            )
            for elem in (parent.path.odd_element(-1),)
            + parent.additional_source_entities
        ):
            if len(cloned.path) > 1:
                attrname = cloned.path[1]
                parent_entity = cloned.path[0]
            else:
                attrname = cloned.path[0]
                parent_entity = cloned.path[0]
            _raise_for_does_not_link(parent.path, attrname, parent_entity)

        cloned.path = PathRegistry.coerce(parent.path[0:-1] + cloned.path[:])

        if self.context:
            cloned.context = tuple(
                value._prepend_path_from(parent) for value in self.context
            )

        if cloned.context:
            parent.context += cloned.context
            parent.additional_source_entities += (
                cloned.additional_source_entities
            )

    @_generative
    def options(self, *opts: _AbstractLoad) -> Self:
        r"""Apply a series of options as sub-options to this
        :class:`_orm.Load`
        object.

        E.g.::

            query = session.query(Author)
            query = query.options(
                joinedload(Author.book).options(
                    load_only(Book.summary, Book.excerpt),
                    joinedload(Book.citations).options(joinedload(Citation.author)),
                )
            )

        :param \*opts: A series of loader option objects (ultimately
         :class:`_orm.Load` objects) which should be applied to the path
         specified by this :class:`_orm.Load` object.

        .. versionadded:: 1.3.6

        .. seealso::

            :func:`.defaultload`

            :ref:`orm_queryguide_relationship_sub_options`

        """
        for opt in opts:
            try:
                opt._apply_to_parent(self)
            except AttributeError as ae:
                if not isinstance(opt, _AbstractLoad):
                    raise sa_exc.ArgumentError(
                        f"Loader option {opt} is not compatible with the "
                        "Load.options() method."
                    ) from ae
                else:
                    raise
        return self

    def _clone_for_bind_strategy(
        self,
        attrs: Optional[Tuple[_AttrType, ...]],
        strategy: Optional[_StrategyKey],
        wildcard_key: Optional[_WildcardKeyType],
        opts: Optional[_OptsType] = None,
        attr_group: Optional[_AttrGroupType] = None,
        propagate_to_loaders: bool = True,
        reconcile_to_other: Optional[bool] = None,
        extra_criteria: Optional[Tuple[Any, ...]] = None,
    ) -> Self:
        # for individual strategy that needs to propagate, set the whole
        # Load container to also propagate, so that it shows up in
        # InstanceState.load_options
        if propagate_to_loaders:
            self.propagate_to_loaders = True

        if self.path.is_token:
            raise sa_exc.ArgumentError(
                "Wildcard token cannot be followed by another entity"
            )

        elif path_is_property(self.path):
            # re-use the lookup which will raise a nicely formatted
            # LoaderStrategyException
            if strategy:
                self.path.prop._strategy_lookup(self.path.prop, strategy[0])
            else:
                raise sa_exc.ArgumentError(
                    f"Mapped attribute '{self.path.prop}' does not "
                    "refer to a mapped entity"
                )

        if attrs is None:
            load_element = _ClassStrategyLoad.create(
                self.path,
                None,
                strategy,
                wildcard_key,
                opts,
                propagate_to_loaders,
                attr_group=attr_group,
                reconcile_to_other=reconcile_to_other,
                extra_criteria=extra_criteria,
            )
            if load_element:
                self.context += (load_element,)
                assert opts is not None
                self.additional_source_entities += cast(
                    "Tuple[_InternalEntityType[Any]]", opts["entities"]
                )

        else:
            for attr in attrs:
                if isinstance(attr, str):
                    load_element = _TokenStrategyLoad.create(
                        self.path,
                        attr,
                        strategy,
                        wildcard_key,
                        opts,
                        propagate_to_loaders,
                        attr_group=attr_group,
                        reconcile_to_other=reconcile_to_other,
                        extra_criteria=extra_criteria,
                    )
                else:
                    load_element = _AttributeStrategyLoad.create(
                        self.path,
                        attr,
                        strategy,
                        wildcard_key,
                        opts,
                        propagate_to_loaders,
                        attr_group=attr_group,
                        reconcile_to_other=reconcile_to_other,
                        extra_criteria=extra_criteria,
                    )

                if load_element:
                    # for relationship options, update self.path on this Load
                    # object with the latest path.
                    if wildcard_key is _RELATIONSHIP_TOKEN:
                        self.path = load_element.path
                    self.context += (load_element,)

                    # this seems to be effective for selectinloader,
                    # giving the extra match to one more level deep.
                    # but does not work for immediateloader, which still
                    # must add additional options at load time
                    if load_element.local_opts.get("recursion_depth", False):
                        r1 = load_element._recurse()
                        self.context += (r1,)

        return self

    def __getstate__(self):
        d = self._shallow_to_dict()
        d["path"] = self.path.serialize()
        return d

    def __setstate__(self, state):
        state["path"] = PathRegistry.deserialize(state["path"])
        self._shallow_from_dict(state)


class _WildcardLoad(_AbstractLoad):
    """represent a standalone '*' load operation"""

    __slots__ = ("strategy", "path", "local_opts")

    _traverse_internals = [
        ("strategy", visitors.ExtendedInternalTraversal.dp_plain_obj),
        ("path", visitors.ExtendedInternalTraversal.dp_plain_obj),
        (
            "local_opts",
            visitors.ExtendedInternalTraversal.dp_string_multi_dict,
        ),
    ]
    cache_key_traversal: _CacheKeyTraversalType = None

    strategy: Optional[Tuple[Any, ...]]
    local_opts: _OptsType
    path: Union[Tuple[()], Tuple[str]]
    propagate_to_loaders = False

    def __init__(self) -> None:
        self.path = ()
        self.strategy = None
        self.local_opts = util.EMPTY_DICT

    def _clone_for_bind_strategy(
        self,
        attrs,
        strategy,
        wildcard_key,
        opts=None,
        attr_group=None,
        propagate_to_loaders=True,
        reconcile_to_other=None,
        extra_criteria=None,
    ):
        assert attrs is not None
        attr = attrs[0]
        assert (
            wildcard_key
            and isinstance(attr, str)
            and attr in (_WILDCARD_TOKEN, _DEFAULT_TOKEN)
        )

        attr = f"{wildcard_key}:{attr}"

        self.strategy = strategy
        self.path = (attr,)
        if opts:
            self.local_opts = util.immutabledict(opts)

        assert extra_criteria is None

    def options(self, *opts: _AbstractLoad) -> Self:
        raise NotImplementedError("Star option does not support sub-options")

    def _apply_to_parent(self, parent: Load) -> None:
        """apply this :class:`_orm._WildcardLoad` object as a sub-option of
        a :class:`_orm.Load` object.

        This method is used by the :meth:`_orm.Load.options` method.   Note
        that :class:`_orm.WildcardLoad` itself can't have sub-options, but
        it may be used as the sub-option of a :class:`_orm.Load` object.

        """
        assert self.path
        attr = self.path[0]
        if attr.endswith(_DEFAULT_TOKEN):
            attr = f"{attr.split(':')[0]}:{_WILDCARD_TOKEN}"

        effective_path = cast(AbstractEntityRegistry, parent.path).token(attr)

        assert effective_path.is_token

        loader = _TokenStrategyLoad.create(
            effective_path,
            None,
            self.strategy,
            None,
            self.local_opts,
            self.propagate_to_loaders,
        )

        parent.context += (loader,)

    def _process(self, compile_state, mapper_entities, raiseerr):
        is_refresh = compile_state.compile_options._for_refresh_state

        if is_refresh and not self.propagate_to_loaders:
            return

        entities = [ent.entity_zero for ent in mapper_entities]
        current_path = compile_state.current_path

        start_path: _PathRepresentation = self.path

        if current_path:
            # TODO: no cases in test suite where we actually get
            # None back here
            new_path = self._chop_path(start_path, current_path)
            if new_path is None:
                return

            # chop_path does not actually "chop" a wildcard token path,
            # just returns it
            assert new_path == start_path

        # start_path is a single-token tuple
        assert start_path and len(start_path) == 1

        token = start_path[0]
        assert isinstance(token, str)
        entity = self._find_entity_basestring(entities, token, raiseerr)

        if not entity:
            return

        path_element = entity

        # transfer our entity-less state into a Load() object
        # with a real entity path.  Start with the lead entity
        # we just located, then go through the rest of our path
        # tokens and populate into the Load().

        assert isinstance(token, str)
        loader = _TokenStrategyLoad.create(
            path_element._path_registry,
            token,
            self.strategy,
            None,
            self.local_opts,
            self.propagate_to_loaders,
            raiseerr=raiseerr,
        )
        if not loader:
            return

        assert loader.path.is_token

        # don't pass a reconciled lead entity here
        loader.process_compile_state(
            self, compile_state, mapper_entities, None, raiseerr
        )

        return loader

    def _find_entity_basestring(
        self,
        entities: Iterable[_InternalEntityType[Any]],
        token: str,
        raiseerr: bool,
    ) -> Optional[_InternalEntityType[Any]]:
        if token.endswith(f":{_WILDCARD_TOKEN}"):
            if len(list(entities)) != 1:
                if raiseerr:
                    raise sa_exc.ArgumentError(
                        "Can't apply wildcard ('*') or load_only() "
                        f"loader option to multiple entities "
                        f"{', '.join(str(ent) for ent in entities)}. Specify "
                        "loader options for each entity individually, such as "
                        f"""{
                            ", ".join(
                                f"Load({ent}).some_option('*')"
                                for ent in entities
                            )
                        }."""
                    )
        elif token.endswith(_DEFAULT_TOKEN):
            raiseerr = False

        for ent in entities:
            # return only the first _MapperEntity when searching
            # based on string prop name.   Ideally object
            # attributes are used to specify more exactly.
            return ent
        else:
            if raiseerr:
                raise sa_exc.ArgumentError(
                    "Query has only expression-based entities - "
                    f'can\'t find property named "{token}".'
                )
            else:
                return None

    def __getstate__(self) -> Dict[str, Any]:
        d = self._shallow_to_dict()
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._shallow_from_dict(state)


class _LoadElement(
    cache_key.HasCacheKey, traversals.HasShallowCopy, visitors.Traversible
):
    """represents strategy information to select for a LoaderStrategy
    and pass options to it.

    :class:`._LoadElement` objects provide the inner datastructure
    stored by a :class:`_orm.Load` object and are also the object passed
    to methods like :meth:`.LoaderStrategy.setup_query`.

    .. versionadded:: 2.0

    """

    __slots__ = (
        "path",
        "strategy",
        "propagate_to_loaders",
        "local_opts",
        "_extra_criteria",
        "_reconcile_to_other",
    )
    __visit_name__ = "load_element"

    _traverse_internals = [
        ("path", visitors.ExtendedInternalTraversal.dp_has_cache_key),
        ("strategy", visitors.ExtendedInternalTraversal.dp_plain_obj),
        (
            "local_opts",
            visitors.ExtendedInternalTraversal.dp_string_multi_dict,
        ),
        ("_extra_criteria", visitors.InternalTraversal.dp_clauseelement_list),
        ("propagate_to_loaders", visitors.InternalTraversal.dp_plain_obj),
        ("_reconcile_to_other", visitors.InternalTraversal.dp_plain_obj),
    ]
    _cache_key_traversal = None

    _extra_criteria: Tuple[Any, ...]

    _reconcile_to_other: Optional[bool]
    strategy: Optional[_StrategyKey]
    path: PathRegistry
    propagate_to_loaders: bool

    local_opts: util.immutabledict[str, Any]

    is_token_strategy: bool
    is_class_strategy: bool

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other):
        return traversals.compare(self, other)

    @property
    def is_opts_only(self) -> bool:
        return bool(self.local_opts and self.strategy is None)

    def _clone(self, **kw: Any) -> _LoadElement:
        cls = self.__class__
        s = cls.__new__(cls)

        self._shallow_copy_to(s)
        return s

    def _update_opts(self, **kw: Any) -> _LoadElement:
        new = self._clone()
        new.local_opts = new.local_opts.union(kw)
        return new

    def __getstate__(self) -> Dict[str, Any]:
        d = self._shallow_to_dict()
        d["path"] = self.path.serialize()
        return d

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state["path"] = PathRegistry.deserialize(state["path"])
        self._shallow_from_dict(state)

    def _raise_for_no_match(self, parent_loader, mapper_entities):
        path = parent_loader.path

        found_entities = False
        for ent in mapper_entities:
            ezero = ent.entity_zero
            if ezero:
                found_entities = True
                break

        if not found_entities:
            raise sa_exc.ArgumentError(
                "Query has only expression-based entities; "
                f"attribute loader options for {path[0]} can't "
                "be applied here."
            )
        else:
            raise sa_exc.ArgumentError(
                f"Mapped class {path[0]} does not apply to any of the "
                f"root entities in this query, e.g. "
                f"""{
                    ", ".join(
                        str(x.entity_zero)
                        for x in mapper_entities if x.entity_zero
                    )}. Please """
                "specify the full path "
                "from one of the root entities to the target "
                "attribute. "
            )

    def _adjust_effective_path_for_current_path(
        self, effective_path: PathRegistry, current_path: PathRegistry
    ) -> Optional[PathRegistry]:
        """receives the 'current_path' entry from an :class:`.ORMCompileState`
        instance, which is set during lazy loads and secondary loader strategy
        loads, and adjusts the given path to be relative to the
        current_path.

        E.g. given a loader path and current path:

        .. sourcecode:: text

            lp: User -> orders -> Order -> items -> Item -> keywords -> Keyword

            cp: User -> orders -> Order -> items

        The adjusted path would be:

        .. sourcecode:: text

            Item -> keywords -> Keyword


        """
        chopped_start_path = Load._chop_path(
            effective_path.natural_path, current_path
        )
        if not chopped_start_path:
            return None

        tokens_removed_from_start_path = len(effective_path) - len(
            chopped_start_path
        )

        loader_lead_path_element = self.path[tokens_removed_from_start_path]

        effective_path = PathRegistry.coerce(
            (loader_lead_path_element,) + chopped_start_path[1:]
        )

        return effective_path

    def _init_path(
        self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria
    ):
        """Apply ORM attributes and/or wildcard to an existing path, producing
        a new path.

        This method is used within the :meth:`.create` method to initialize
        a :class:`._LoadElement` object.

        """
        raise NotImplementedError()

    def _prepare_for_compile_state(
        self,
        parent_loader,
        compile_state,
        mapper_entities,
        reconciled_lead_entity,
        raiseerr,
    ):
        """implemented by subclasses."""
        raise NotImplementedError()

    def process_compile_state(
        self,
        parent_loader,
        compile_state,
        mapper_entities,
        reconciled_lead_entity,
        raiseerr,
    ):
        """populate ORMCompileState.attributes with loader state for this
        _LoadElement.

        """
        keys = self._prepare_for_compile_state(
            parent_loader,
            compile_state,
            mapper_entities,
            reconciled_lead_entity,
            raiseerr,
        )
        for key in keys:
            if key in compile_state.attributes:
                compile_state.attributes[key] = _LoadElement._reconcile(
                    self, compile_state.attributes[key]
                )
            else:
                compile_state.attributes[key] = self

    @classmethod
    def create(
        cls,
        path: PathRegistry,
        attr: Union[_AttrType, _StrPathToken, None],
        strategy: Optional[_StrategyKey],
        wildcard_key: Optional[_WildcardKeyType],
        local_opts: Optional[_OptsType],
        propagate_to_loaders: bool,
        raiseerr: bool = True,
        attr_group: Optional[_AttrGroupType] = None,
        reconcile_to_other: Optional[bool] = None,
        extra_criteria: Optional[Tuple[Any, ...]] = None,
    ) -> _LoadElement:
        """Create a new :class:`._LoadElement` object."""

        opt = cls.__new__(cls)
        opt.path = path
        opt.strategy = strategy
        opt.propagate_to_loaders = propagate_to_loaders
        opt.local_opts = (
            util.immutabledict(local_opts) if local_opts else util.EMPTY_DICT
        )
        opt._extra_criteria = ()

        if reconcile_to_other is not None:
            opt._reconcile_to_other = reconcile_to_other
        elif strategy is None and not local_opts:
            opt._reconcile_to_other = True
        else:
            opt._reconcile_to_other = None

        path = opt._init_path(
            path, attr, wildcard_key, attr_group, raiseerr, extra_criteria
        )

        if not path:
            return None  # type: ignore

        assert opt.is_token_strategy == path.is_token

        opt.path = path
        return opt

    def __init__(self) -> None:
        raise NotImplementedError()

    def _recurse(self) -> _LoadElement:
        cloned = self._clone()
        cloned.path = PathRegistry.coerce(self.path[:] + self.path[-2:])

        return cloned

    def _prepend_path_from(self, parent: Load) -> _LoadElement:
        """adjust the path of this :class:`._LoadElement` to be
        a subpath of that of the given parent :class:`_orm.Load` object's
        path.

        This is used by the :meth:`_orm.Load._apply_to_parent` method,
        which is in turn part of the :meth:`_orm.Load.options` method.

        """

        if not any(
            orm_util._entity_corresponds_to_use_path_impl(
                elem,
                self.path.odd_element(0),
            )
            for elem in (parent.path.odd_element(-1),)
            + parent.additional_source_entities
        ):
            raise sa_exc.ArgumentError(
                f'Attribute "{self.path[1]}" does not link '
                f'from element "{parent.path[-1]}".'
            )

        return self._prepend_path(parent.path)

    def _prepend_path(self, path: PathRegistry) -> _LoadElement:
        cloned = self._clone()

        assert cloned.strategy == self.strategy
        assert cloned.local_opts == self.local_opts
        assert cloned.is_class_strategy == self.is_class_strategy

        cloned.path = PathRegistry.coerce(path[0:-1] + cloned.path[:])

        return cloned

    @staticmethod
    def _reconcile(
        replacement: _LoadElement, existing: _LoadElement
    ) -> _LoadElement:
        """define behavior for when two Load objects are to be put into
        the context.attributes under the same key.

        :param replacement: ``_LoadElement`` that seeks to replace the
         existing one

        :param existing: ``_LoadElement`` that is already present.

        """
        # mapper inheritance loading requires fine-grained "block other
        # options" / "allow these options to be overridden" behaviors
        # see test_poly_loading.py

        if replacement._reconcile_to_other:
            return existing
        elif replacement._reconcile_to_other is False:
            return replacement
        elif existing._reconcile_to_other:
            return replacement
        elif existing._reconcile_to_other is False:
            return existing

        if existing is replacement:
            return replacement
        elif (
            existing.strategy == replacement.strategy
            and existing.local_opts == replacement.local_opts
        ):
            return replacement
        elif replacement.is_opts_only:
            existing = existing._clone()
            existing.local_opts = existing.local_opts.union(
                replacement.local_opts
            )
            existing._extra_criteria += replacement._extra_criteria
            return existing
        elif existing.is_opts_only:
            replacement = replacement._clone()
            replacement.local_opts = replacement.local_opts.union(
                existing.local_opts
            )
            replacement._extra_criteria += existing._extra_criteria
            return replacement
        elif replacement.path.is_token:
            # use 'last one wins' logic for wildcard options.  this is also
            # kind of inconsistent vs. options that are specific paths which
            # will raise as below
            return replacement

        raise sa_exc.InvalidRequestError(
            f"Loader strategies for {replacement.path} conflict"
        )


class _AttributeStrategyLoad(_LoadElement):
    """Loader strategies against specific relationship or column paths.

    e.g.::

        joinedload(User.addresses)
        defer(Order.name)
        selectinload(User.orders).lazyload(Order.items)

    """

    __slots__ = ("_of_type", "_path_with_polymorphic_path")

    __visit_name__ = "attribute_strategy_load_element"

    _traverse_internals = _LoadElement._traverse_internals + [
        ("_of_type", visitors.ExtendedInternalTraversal.dp_multi),
        (
            "_path_with_polymorphic_path",
            visitors.ExtendedInternalTraversal.dp_has_cache_key,
        ),
    ]

    _of_type: Union[Mapper[Any], AliasedInsp[Any], None]
    _path_with_polymorphic_path: Optional[PathRegistry]

    is_class_strategy = False
    is_token_strategy = False

    def _init_path(
        self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria
    ):
        assert attr is not None
        self._of_type = None
        self._path_with_polymorphic_path = None
        insp, _, prop = _parse_attr_argument(attr)

        if insp.is_property:
            # direct property can be sent from internal strategy logic
            # that sets up specific loaders, such as
            # emit_lazyload->_lazyload_reverse
            # prop = found_property = attr
            prop = attr
            path = path[prop]

            if path.has_entity:
                path = path.entity_path
            return path

        elif not insp.is_attribute:
            # should not reach here;
            assert False

        # here we assume we have user-passed InstrumentedAttribute
        if not orm_util._entity_corresponds_to_use_path_impl(
            path[-1], attr.parent
        ):
            if raiseerr:
                if attr_group and attr is not attr_group[0]:
                    raise sa_exc.ArgumentError(
                        "Can't apply wildcard ('*') or load_only() "
                        "loader option to multiple entities in the "
                        "same option. Use separate options per entity."
                    )
                else:
                    _raise_for_does_not_link(path, str(attr), attr.parent)
            else:
                return None

        # note the essential logic of this attribute was very different in
        # 1.4, where there were caching failures in e.g.
        # test_relationship_criteria.py::RelationshipCriteriaTest::
        # test_selectinload_nested_criteria[True] if an existing
        # "_extra_criteria" on a Load object were replaced with that coming
        # from an attribute.   This appears to have been an artifact of how
        # _UnboundLoad / Load interacted together, which was opaque and
        # poorly defined.
        if extra_criteria:
            assert not attr._extra_criteria
            self._extra_criteria = extra_criteria
        else:
            self._extra_criteria = attr._extra_criteria

        if getattr(attr, "_of_type", None):
            ac = attr._of_type
            ext_info = inspect(ac)
            self._of_type = ext_info

            self._path_with_polymorphic_path = path.entity_path[prop]

            path = path[prop][ext_info]

        else:
            path = path[prop]

        if path.has_entity:
            path = path.entity_path

        return path

    def _generate_extra_criteria(self, context):
        """Apply the current bound parameters in a QueryContext to the
        immediate "extra_criteria" stored with this Load object.

        Load objects are typically pulled from the cached version of
        the statement from a QueryContext.  The statement currently being
        executed will have new values (and keys) for bound parameters in the
        extra criteria which need to be applied by loader strategies when
        they handle this criteria for a result set.

        """

        assert (
            self._extra_criteria
        ), "this should only be called if _extra_criteria is present"

        orig_query = context.compile_state.select_statement
        current_query = context.query

        # NOTE: while it seems like we should not do the "apply" operation
        # here if orig_query is current_query, skipping it in the "optimized"
        # case causes the query to be different from a cache key perspective,
        # because we are creating a copy of the criteria which is no longer
        # the same identity of the _extra_criteria in the loader option
        # itself.  cache key logic produces a different key for
        # (A, copy_of_A) vs. (A, A), because in the latter case it shortens
        # the second part of the key to just indicate on identity.

        # if orig_query is current_query:
        # not cached yet.   just do the and_()
        #    return and_(*self._extra_criteria)

        k1 = orig_query._generate_cache_key()
        k2 = current_query._generate_cache_key()

        return k2._apply_params_to_element(k1, and_(*self._extra_criteria))

    def _set_of_type_info(self, context, current_path):
        assert self._path_with_polymorphic_path

        pwpi = self._of_type
        assert pwpi
        if not pwpi.is_aliased_class:
            pwpi = inspect(
                orm_util.AliasedInsp._with_polymorphic_factory(
                    pwpi.mapper.base_mapper,
                    (pwpi.mapper,),
                    aliased=True,
                    _use_mapper_path=True,
                )
            )
        start_path = self._path_with_polymorphic_path
        if current_path:
            new_path = self._adjust_effective_path_for_current_path(
                start_path, current_path
            )
            if new_path is None:
                return
            start_path = new_path

        key = ("path_with_polymorphic", start_path.natural_path)
        if key in context:
            existing_aliased_insp = context[key]
            this_aliased_insp = pwpi
            new_aliased_insp = existing_aliased_insp._merge_with(
                this_aliased_insp
            )
            context[key] = new_aliased_insp
        else:
            context[key] = pwpi

    def _prepare_for_compile_state(
        self,
        parent_loader,
        compile_state,
        mapper_entities,
        reconciled_lead_entity,
        raiseerr,
    ):
        # _AttributeStrategyLoad

        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state
        assert not self.path.is_token

        if is_refresh and not self.propagate_to_loaders:
            return []

        if self._of_type:
            # apply additional with_polymorphic alias that may have been
            # generated.  this has to happen even if this is a defaultload
            self._set_of_type_info(compile_state.attributes, current_path)

        # omit setting loader attributes for a "defaultload" type of option
        if not self.strategy and not self.local_opts:
            return []

        if raiseerr and not reconciled_lead_entity:
            self._raise_for_no_match(parent_loader, mapper_entities)

        if self.path.has_entity:
            effective_path = self.path.parent
        else:
            effective_path = self.path

        if current_path:
            assert effective_path is not None
            effective_path = self._adjust_effective_path_for_current_path(
                effective_path, current_path
            )
            if effective_path is None:
                return []

        return [("loader", cast(PathRegistry, effective_path).natural_path)]

    def __getstate__(self):
        d = super().__getstate__()

        # can't pickle this.  See
        # test_pickled.py -> test_lazyload_extra_criteria_not_supported
        # where we should be emitting a warning for the usual case where this
        # would be non-None
        d["_extra_criteria"] = ()

        if self._path_with_polymorphic_path:
            d["_path_with_polymorphic_path"] = (
                self._path_with_polymorphic_path.serialize()
            )

        if self._of_type:
            if self._of_type.is_aliased_class:
                d["_of_type"] = None
            elif self._of_type.is_mapper:
                d["_of_type"] = self._of_type.class_
            else:
                assert False, "unexpected object for _of_type"

        return d

    def __setstate__(self, state):
        super().__setstate__(state)

        if state.get("_path_with_polymorphic_path", None):
            self._path_with_polymorphic_path = PathRegistry.deserialize(
                state["_path_with_polymorphic_path"]
            )
        else:
            self._path_with_polymorphic_path = None

        if state.get("_of_type", None):
            self._of_type = inspect(state["_of_type"])
        else:
            self._of_type = None


class _TokenStrategyLoad(_LoadElement):
    """Loader strategies against wildcard attributes

    e.g.::

        raiseload("*")
        Load(User).lazyload("*")
        defer("*")
        load_only(User.name, User.email)  # will create a defer('*')
        joinedload(User.addresses).raiseload("*")

    """

    __visit_name__ = "token_strategy_load_element"

    inherit_cache = True
    is_class_strategy = False
    is_token_strategy = True

    def _init_path(
        self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria
    ):
        # assert isinstance(attr, str) or attr is None
        if attr is not None:
            default_token = attr.endswith(_DEFAULT_TOKEN)
            if attr.endswith(_WILDCARD_TOKEN) or default_token:
                if wildcard_key:
                    attr = f"{wildcard_key}:{attr}"

                path = path.token(attr)
                return path
            else:
                raise sa_exc.ArgumentError(
                    "Strings are not accepted for attribute names in loader "
                    "options; please use class-bound attributes directly."
                )
        return path

    def _prepare_for_compile_state(
        self,
        parent_loader,
        compile_state,
        mapper_entities,
        reconciled_lead_entity,
        raiseerr,
    ):
        # _TokenStrategyLoad

        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state

        assert self.path.is_token

        if is_refresh and not self.propagate_to_loaders:
            return []

        # omit setting attributes for a "defaultload" type of option
        if not self.strategy and not self.local_opts:
            return []

        effective_path = self.path
        if reconciled_lead_entity:
            effective_path = PathRegistry.coerce(
                (reconciled_lead_entity,) + effective_path.path[1:]
            )

        if current_path:
            new_effective_path = self._adjust_effective_path_for_current_path(
                effective_path, current_path
            )
            if new_effective_path is None:
                return []
            effective_path = new_effective_path

        # for a wildcard token, expand out the path we set
        # to encompass everything from the query entity on
        # forward.  not clear if this is necessary when current_path
        # is set.

        return [
            ("loader", natural_path)
            for natural_path in (
                cast(
                    TokenRegistry, effective_path
                )._generate_natural_for_superclasses()
            )
        ]


class _ClassStrategyLoad(_LoadElement):
    """Loader strategies that deals with a class as a target, not
    an attribute path

    e.g.::

        q = s.query(Person).options(
            selectin_polymorphic(Person, [Engineer, Manager])
        )

    """

    inherit_cache = True
    is_class_strategy = True
    is_token_strategy = False

    __visit_name__ = "class_strategy_load_element"

    def _init_path(
        self, path, attr, wildcard_key, attr_group, raiseerr, extra_criteria
    ):
        return path

    def _prepare_for_compile_state(
        self,
        parent_loader,
        compile_state,
        mapper_entities,
        reconciled_lead_entity,
        raiseerr,
    ):
        # _ClassStrategyLoad

        current_path = compile_state.current_path
        is_refresh = compile_state.compile_options._for_refresh_state

        if is_refresh and not self.propagate_to_loaders:
            return []

        # omit setting attributes for a "defaultload" type of option
        if not self.strategy and not self.local_opts:
            return []

        effective_path = self.path

        if current_path:
            new_effective_path = self._adjust_effective_path_for_current_path(
                effective_path, current_path
            )
            if new_effective_path is None:
                return []
            effective_path = new_effective_path

        return [("loader", effective_path.natural_path)]


def _generate_from_keys(
    meth: Callable[..., _AbstractLoad],
    keys: Tuple[_AttrType, ...],
    chained: bool,
    kw: Any,
) -> _AbstractLoad:
    lead_element: Optional[_AbstractLoad] = None

    attr: Any
    for is_default, _keys in (True, keys[0:-1]), (False, keys[-1:]):
        for attr in _keys:
            if isinstance(attr, str):
                if attr.startswith("." + _WILDCARD_TOKEN):
                    util.warn_deprecated(
                        "The undocumented `.{WILDCARD}` format is "
                        "deprecated "
                        "and will be removed in a future version as "
                        "it is "
                        "believed to be unused. "
                        "If you have been using this functionality, "
                        "please "
                        "comment on Issue #4390 on the SQLAlchemy project "
                        "tracker.",
                        version="1.4",
                    )
                    attr = attr[1:]

                if attr == _WILDCARD_TOKEN:
                    if is_default:
                        raise sa_exc.ArgumentError(
                            "Wildcard token cannot be followed by "
                            "another entity",
                        )

                    if lead_element is None:
                        lead_element = _WildcardLoad()

                    lead_element = meth(lead_element, _DEFAULT_TOKEN, **kw)

                else:
                    raise sa_exc.ArgumentError(
                        "Strings are not accepted for attribute names in "
                        "loader options; please use class-bound "
                        "attributes directly.",
                    )
            else:
                if lead_element is None:
                    _, lead_entity, _ = _parse_attr_argument(attr)
                    lead_element = Load(lead_entity)

                if is_default:
                    if not chained:
                        lead_element = lead_element.defaultload(attr)
                    else:
                        lead_element = meth(
                            lead_element, attr, _is_chain=True, **kw
                        )
                else:
                    lead_element = meth(lead_element, attr, **kw)

    assert lead_element
    return lead_element


def _parse_attr_argument(
    attr: _AttrType,
) -> Tuple[InspectionAttr, _InternalEntityType[Any], MapperProperty[Any]]:
    """parse an attribute or wildcard argument to produce an
    :class:`._AbstractLoad` instance.

    This is used by the standalone loader strategy functions like
    ``joinedload()``, ``defer()``, etc. to produce :class:`_orm.Load` or
    :class:`._WildcardLoad` objects.

    """
    try:
        # TODO: need to figure out this None thing being returned by
        # inspect(), it should not have None as an option in most cases
        # if at all
        insp: InspectionAttr = inspect(attr)  # type: ignore
    except sa_exc.NoInspectionAvailable as err:
        raise sa_exc.ArgumentError(
            "expected ORM mapped attribute for loader strategy argument"
        ) from err

    lead_entity: _InternalEntityType[Any]

    if insp_is_mapper_property(insp):
        lead_entity = insp.parent
        prop = insp
    elif insp_is_attribute(insp):
        lead_entity = insp.parent
        prop = insp.prop
    else:
        raise sa_exc.ArgumentError(
            "expected ORM mapped attribute for loader strategy argument"
        )

    return insp, lead_entity, prop


def loader_unbound_fn(fn: _FN) -> _FN:
    """decorator that applies docstrings between standalone loader functions
    and the loader methods on :class:`._AbstractLoad`.

    """
    bound_fn = getattr(_AbstractLoad, fn.__name__)
    fn_doc = bound_fn.__doc__
    bound_fn.__doc__ = f"""Produce a new :class:`_orm.Load` object with the
:func:`_orm.{fn.__name__}` option applied.

See :func:`_orm.{fn.__name__}` for usage examples.

"""

    fn.__doc__ = fn_doc
    return fn


# standalone functions follow.  docstrings are filled in
# by the ``@loader_unbound_fn`` decorator.


@loader_unbound_fn
def contains_eager(*keys: _AttrType, **kw: Any) -> _AbstractLoad:
    return _generate_from_keys(Load.contains_eager, keys, True, kw)


@loader_unbound_fn
def load_only(*attrs: _AttrType, raiseload: bool = False) -> _AbstractLoad:
    # TODO: attrs against different classes.  we likely have to
    # add some extra state to Load of some kind
    _, lead_element, _ = _parse_attr_argument(attrs[0])
    return Load(lead_element).load_only(*attrs, raiseload=raiseload)


@loader_unbound_fn
def joinedload(*keys: _AttrType, **kw: Any) -> _AbstractLoad:
    return _generate_from_keys(Load.joinedload, keys, False, kw)


@loader_unbound_fn
def subqueryload(*keys: _AttrType) -> _AbstractLoad:
    return _generate_from_keys(Load.subqueryload, keys, False, {})


@loader_unbound_fn
def selectinload(
    *keys: _AttrType, recursion_depth: Optional[int] = None
) -> _AbstractLoad:
    return _generate_from_keys(
        Load.selectinload, keys, False, {"recursion_depth": recursion_depth}
    )


@loader_unbound_fn
def lazyload(*keys: _AttrType) -> _AbstractLoad:
    return _generate_from_keys(Load.lazyload, keys, False, {})


@loader_unbound_fn
def immediateload(
    *keys: _AttrType, recursion_depth: Optional[int] = None
) -> _AbstractLoad:
    return _generate_from_keys(
        Load.immediateload, keys, False, {"recursion_depth": recursion_depth}
    )


@loader_unbound_fn
def noload(*keys: _AttrType) -> _AbstractLoad:
    return _generate_from_keys(Load.noload, keys, False, {})


@loader_unbound_fn
def raiseload(*keys: _AttrType, **kw: Any) -> _AbstractLoad:
    return _generate_from_keys(Load.raiseload, keys, False, kw)


@loader_unbound_fn
def defaultload(*keys: _AttrType) -> _AbstractLoad:
    return _generate_from_keys(Load.defaultload, keys, False, {})


@loader_unbound_fn
def defer(
    key: _AttrType, *addl_attrs: _AttrType, raiseload: bool = False
) -> _AbstractLoad:
    if addl_attrs:
        util.warn_deprecated(
            "The *addl_attrs on orm.defer is deprecated.  Please use "
            "method chaining in conjunction with defaultload() to "
            "indicate a path.",
            version="1.3",
        )

    if raiseload:
        kw = {"raiseload": raiseload}
    else:
        kw = {}

    return _generate_from_keys(Load.defer, (key,) + addl_attrs, False, kw)


@loader_unbound_fn
def undefer(key: _AttrType, *addl_attrs: _AttrType) -> _AbstractLoad:
    if addl_attrs:
        util.warn_deprecated(
            "The *addl_attrs on orm.undefer is deprecated.  Please use "
            "method chaining in conjunction with defaultload() to "
            "indicate a path.",
            version="1.3",
        )
    return _generate_from_keys(Load.undefer, (key,) + addl_attrs, False, {})


@loader_unbound_fn
def undefer_group(name: str) -> _AbstractLoad:
    element = _WildcardLoad()
    return element.undefer_group(name)


@loader_unbound_fn
def with_expression(
    key: _AttrType, expression: _ColumnExpressionArgument[Any]
) -> _AbstractLoad:
    return _generate_from_keys(
        Load.with_expression, (key,), False, {"expression": expression}
    )


@loader_unbound_fn
def selectin_polymorphic(
    base_cls: _EntityType[Any], classes: Iterable[Type[Any]]
) -> _AbstractLoad:
    ul = Load(base_cls)
    return ul.selectin_polymorphic(classes)


def _raise_for_does_not_link(path, attrname, parent_entity):
    if len(path) > 1:
        path_is_of_type = path[-1].entity is not path[-2].mapper.class_
        if insp_is_aliased_class(parent_entity):
            parent_entity_str = str(parent_entity)
        else:
            parent_entity_str = parent_entity.class_.__name__

        raise sa_exc.ArgumentError(
            f'ORM mapped entity or attribute "{attrname}" does not '
            f'link from relationship "{path[-2]}%s".%s'
            % (
                f".of_type({path[-1]})" if path_is_of_type else "",
                (
                    "  Did you mean to use "
                    f'"{path[-2]}'
                    f'.of_type({parent_entity_str})" or "loadopt.options('
                    f"selectin_polymorphic({path[-2].mapper.class_.__name__}, "
                    f'[{parent_entity_str}]), ...)" ?'
                    if not path_is_of_type
                    and not path[-1].is_aliased_class
                    and orm_util._entity_corresponds_to(
                        path.entity, inspect(parent_entity).mapper
                    )
                    else ""
                ),
            )
        )
    else:
        raise sa_exc.ArgumentError(
            f'ORM mapped attribute "{attrname}" does not '
            f'link mapped class "{path[-1]}"'
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\galoistools.py ===
"""Dense univariate polynomials with coefficients in Galois fields. """

from math import ceil as _ceil, sqrt as _sqrt, prod

from sympy.core.random import uniform, _randint
from sympy.external.gmpy import SYMPY_INTS, MPZ, invert
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors


def gf_crt(U, M, K=None):
    """
    Chinese Remainder Theorem.

    Given a set of integer residues ``u_0,...,u_n`` and a set of
    co-prime integer moduli ``m_0,...,m_n``, returns an integer
    ``u``, such that ``u = u_i mod m_i`` for ``i = ``0,...,n``.

    Examples
    ========

    Consider a set of residues ``U = [49, 76, 65]``
    and a set of moduli ``M = [99, 97, 95]``. Then we have::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_crt

       >>> gf_crt([49, 76, 65], [99, 97, 95], ZZ)
       639985

    This is the correct result because::

       >>> [639985 % m for m in [99, 97, 95]]
       [49, 76, 65]

    Note: this is a low-level routine with no error checking.

    See Also
    ========

    sympy.ntheory.modular.crt : a higher level crt routine
    sympy.ntheory.modular.solve_congruence

    """
    p = prod(M, start=K.one)
    v = K.zero

    for u, m in zip(U, M):
        e = p // m
        s, _, _ = K.gcdex(e, m)
        v += e*(u*s % m)

    return v % p


def gf_crt1(M, K):
    """
    First part of the Chinese Remainder Theorem.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
    >>> U = [49, 76, 65]
    >>> M = [99, 97, 95]

    The following two codes have the same result.

    >>> gf_crt(U, M, ZZ)
    639985

    >>> p, E, S = gf_crt1(M, ZZ)
    >>> gf_crt2(U, M, p, E, S, ZZ)
    639985

    However, it is faster when we want to fix ``M`` and
    compute for multiple U, i.e. the following cases:

    >>> p, E, S = gf_crt1(M, ZZ)
    >>> Us = [[49, 76, 65], [23, 42, 67]]
    >>> for U in Us:
    ...     print(gf_crt2(U, M, p, E, S, ZZ))
    639985
    236237

    See Also
    ========

    sympy.ntheory.modular.crt1 : a higher level crt routine
    sympy.polys.galoistools.gf_crt
    sympy.polys.galoistools.gf_crt2

    """
    E, S = [], []
    p = prod(M, start=K.one)

    for m in M:
        E.append(p // m)
        S.append(K.gcdex(E[-1], m)[0] % m)

    return p, E, S


def gf_crt2(U, M, p, E, S, K):
    """
    Second part of the Chinese Remainder Theorem.

    See ``gf_crt1`` for usage.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_crt2

    >>> U = [49, 76, 65]
    >>> M = [99, 97, 95]
    >>> p = 912285
    >>> E = [9215, 9405, 9603]
    >>> S = [62, 24, 12]

    >>> gf_crt2(U, M, p, E, S, ZZ)
    639985

    See Also
    ========

    sympy.ntheory.modular.crt2 : a higher level crt routine
    sympy.polys.galoistools.gf_crt
    sympy.polys.galoistools.gf_crt1

    """
    v = K.zero

    for u, m, e, s in zip(U, M, E, S):
        v += e*(u*s % m)

    return v % p


def gf_int(a, p):
    """
    Coerce ``a mod p`` to an integer in the range ``[-p/2, p/2]``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_int

    >>> gf_int(2, 7)
    2
    >>> gf_int(5, 7)
    -2

    """
    if a <= p // 2:
        return a
    else:
        return a - p


def gf_degree(f):
    """
    Return the leading degree of ``f``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_degree

    >>> gf_degree([1, 1, 2, 0])
    3
    >>> gf_degree([])
    -1

    """
    return len(f) - 1


def gf_LC(f, K):
    """
    Return the leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_LC

    >>> gf_LC([3, 0, 1], ZZ)
    3

    """
    if not f:
        return K.zero
    else:
        return f[0]


def gf_TC(f, K):
    """
    Return the trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_TC

    >>> gf_TC([3, 0, 1], ZZ)
    1

    """
    if not f:
        return K.zero
    else:
        return f[-1]


def gf_strip(f):
    """
    Remove leading zeros from ``f``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_strip

    >>> gf_strip([0, 0, 0, 3, 0, 1])
    [3, 0, 1]

    """
    if not f or f[0]:
        return f

    k = 0

    for coeff in f:
        if coeff:
            break
        else:
            k += 1

    return f[k:]


def gf_trunc(f, p):
    """
    Reduce all coefficients modulo ``p``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_trunc

    >>> gf_trunc([7, -2, 3], 5)
    [2, 3, 3]

    """
    return gf_strip([ a % p for a in f ])


def gf_normal(f, p, K):
    """
    Normalize all coefficients in ``K``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_normal

    >>> gf_normal([5, 10, 21, -3], 5, ZZ)
    [1, 2]

    """
    return gf_trunc(list(map(K, f)), p)


def gf_from_dict(f, p, K):
    """
    Create a ``GF(p)[x]`` polynomial from a dict.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_from_dict

    >>> gf_from_dict({10: ZZ(4), 4: ZZ(33), 0: ZZ(-1)}, 5, ZZ)
    [4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4]

    """
    n, h = max(f.keys()), []

    if isinstance(n, SYMPY_INTS):
        for k in range(n, -1, -1):
            h.append(f.get(k, K.zero) % p)
    else:
        (n,) = n

        for k in range(n, -1, -1):
            h.append(f.get((k,), K.zero) % p)

    return gf_trunc(h, p)


def gf_to_dict(f, p, symmetric=True):
    """
    Convert a ``GF(p)[x]`` polynomial to a dict.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_to_dict

    >>> gf_to_dict([4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4], 5)
    {0: -1, 4: -2, 10: -1}
    >>> gf_to_dict([4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4], 5, symmetric=False)
    {0: 4, 4: 3, 10: 4}

    """
    n, result = gf_degree(f), {}

    for k in range(0, n + 1):
        if symmetric:
            a = gf_int(f[n - k], p)
        else:
            a = f[n - k]

        if a:
            result[k] = a

    return result


def gf_from_int_poly(f, p):
    """
    Create a ``GF(p)[x]`` polynomial from ``Z[x]``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_from_int_poly

    >>> gf_from_int_poly([7, -2, 3], 5)
    [2, 3, 3]

    """
    return gf_trunc(f, p)


def gf_to_int_poly(f, p, symmetric=True):
    """
    Convert a ``GF(p)[x]`` polynomial to ``Z[x]``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_to_int_poly

    >>> gf_to_int_poly([2, 3, 3], 5)
    [2, -2, -2]
    >>> gf_to_int_poly([2, 3, 3], 5, symmetric=False)
    [2, 3, 3]

    """
    if symmetric:
        return [ gf_int(c, p) for c in f ]
    else:
        return f


def gf_neg(f, p, K):
    """
    Negate a polynomial in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_neg

    >>> gf_neg([3, 2, 1, 0], 5, ZZ)
    [2, 3, 4, 0]

    """
    return [ -coeff % p for coeff in f ]


def gf_add_ground(f, a, p, K):
    """
    Compute ``f + a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_add_ground

    >>> gf_add_ground([3, 2, 4], 2, 5, ZZ)
    [3, 2, 1]

    """
    if not f:
        a = a % p
    else:
        a = (f[-1] + a) % p

        if len(f) > 1:
            return f[:-1] + [a]

    if not a:
        return []
    else:
        return [a]


def gf_sub_ground(f, a, p, K):
    """
    Compute ``f - a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub_ground

    >>> gf_sub_ground([3, 2, 4], 2, 5, ZZ)
    [3, 2, 2]

    """
    if not f:
        a = -a % p
    else:
        a = (f[-1] - a) % p

        if len(f) > 1:
            return f[:-1] + [a]

    if not a:
        return []
    else:
        return [a]


def gf_mul_ground(f, a, p, K):
    """
    Compute ``f * a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_mul_ground

    >>> gf_mul_ground([3, 2, 4], 2, 5, ZZ)
    [1, 4, 3]

    """
    if not a:
        return []
    else:
        return [ (a*b) % p for b in f ]


def gf_quo_ground(f, a, p, K):
    """
    Compute ``f/a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_quo_ground

    >>> gf_quo_ground(ZZ.map([3, 2, 4]), ZZ(2), 5, ZZ)
    [4, 1, 2]

    """
    return gf_mul_ground(f, K.invert(a, p), p, K)


def gf_add(f, g, p, K):
    """
    Add polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_add

    >>> gf_add([3, 2, 4], [2, 2, 2], 5, ZZ)
    [4, 1]

    """
    if not f:
        return g
    if not g:
        return f

    df = gf_degree(f)
    dg = gf_degree(g)

    if df == dg:
        return gf_strip([ (a + b) % p for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = g[:k], g[k:]

        return h + [ (a + b) % p for a, b in zip(f, g) ]


def gf_sub(f, g, p, K):
    """
    Subtract polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub

    >>> gf_sub([3, 2, 4], [2, 2, 2], 5, ZZ)
    [1, 0, 2]

    """
    if not g:
        return f
    if not f:
        return gf_neg(g, p, K)

    df = gf_degree(f)
    dg = gf_degree(g)

    if df == dg:
        return gf_strip([ (a - b) % p for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = gf_neg(g[:k], p, K), g[k:]

        return h + [ (a - b) % p for a, b in zip(f, g) ]


def gf_mul(f, g, p, K):
    """
    Multiply polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_mul

    >>> gf_mul([3, 2, 4], [2, 2, 2], 5, ZZ)
    [1, 0, 3, 2, 3]

    """
    df = gf_degree(f)
    dg = gf_degree(g)

    dh = df + dg
    h = [0]*(dh + 1)

    for i in range(0, dh + 1):
        coeff = K.zero

        for j in range(max(0, i - dg), min(i, df) + 1):
            coeff += f[j]*g[i - j]

        h[i] = coeff % p

    return gf_strip(h)


def gf_sqr(f, p, K):
    """
    Square polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqr

    >>> gf_sqr([3, 2, 4], 5, ZZ)
    [4, 2, 3, 1, 1]

    """
    df = gf_degree(f)

    dh = 2*df
    h = [0]*(dh + 1)

    for i in range(0, dh + 1):
        coeff = K.zero

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            coeff += f[j]*f[i - j]

        coeff += coeff

        if n & 1:
            elem = f[jmax + 1]
            coeff += elem**2

        h[i] = coeff % p

    return gf_strip(h)


def gf_add_mul(f, g, h, p, K):
    """
    Returns ``f + g*h`` where ``f``, ``g``, ``h`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_add_mul
    >>> gf_add_mul([3, 2, 4], [2, 2, 2], [1, 4], 5, ZZ)
    [2, 3, 2, 2]
    """
    return gf_add(f, gf_mul(g, h, p, K), p, K)


def gf_sub_mul(f, g, h, p, K):
    """
    Compute ``f - g*h`` where ``f``, ``g``, ``h`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub_mul

    >>> gf_sub_mul([3, 2, 4], [2, 2, 2], [1, 4], 5, ZZ)
    [3, 3, 2, 1]

    """
    return gf_sub(f, gf_mul(g, h, p, K), p, K)


def gf_expand(F, p, K):
    """
    Expand results of :func:`~.factor` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_expand

    >>> gf_expand([([3, 2, 4], 1), ([2, 2], 2), ([3, 1], 3)], 5, ZZ)
    [4, 3, 0, 3, 0, 1, 4, 1]

    """
    if isinstance(F, tuple):
        lc, F = F
    else:
        lc = K.one

    g = [lc]

    for f, k in F:
        f = gf_pow(f, k, p, K)
        g = gf_mul(g, f, p, K)

    return g


def gf_div(f, g, p, K):
    """
    Division with remainder in ``GF(p)[x]``.

    Given univariate polynomials ``f`` and ``g`` with coefficients in a
    finite field with ``p`` elements, returns polynomials ``q`` and ``r``
    (quotient and remainder) such that ``f = q*g + r``.

    Consider polynomials ``x**3 + x + 1`` and ``x**2 + x`` in GF(2)::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_div, gf_add_mul

       >>> gf_div(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
       ([1, 1], [1])

    As result we obtained quotient ``x + 1`` and remainder ``1``, thus::

       >>> gf_add_mul(ZZ.map([1]), ZZ.map([1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
       [1, 0, 1, 1]

    References
    ==========

    .. [1] [Monagan93]_
    .. [2] [Gathen99]_

    """
    df = gf_degree(f)
    dg = gf_degree(g)

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return [], f

    inv = K.invert(g[0], p)

    h, dq, dr = list(f), df - dg, dg - 1

    for i in range(0, df + 1):
        coeff = h[i]

        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]

        if i <= dq:
            coeff *= inv

        h[i] = coeff % p

    return h[:dq + 1], gf_strip(h[dq + 1:])


def gf_rem(f, g, p, K):
    """
    Compute polynomial remainder in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_rem

    >>> gf_rem(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    [1]

    """
    return gf_div(f, g, p, K)[1]


def gf_quo(f, g, p, K):
    """
    Compute exact quotient in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_quo

    >>> gf_quo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    [1, 1]
    >>> gf_quo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)
    [3, 2, 4]

    """
    df = gf_degree(f)
    dg = gf_degree(g)

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return []

    inv = K.invert(g[0], p)

    h, dq, dr = f[:], df - dg, dg - 1

    for i in range(0, dq + 1):
        coeff = h[i]

        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]

        h[i] = (coeff * inv) % p

    return h[:dq + 1]


def gf_exquo(f, g, p, K):
    """
    Compute polynomial quotient in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_exquo

    >>> gf_exquo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)
    [3, 2, 4]

    >>> gf_exquo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [1, 1, 0] does not divide [1, 0, 1, 1]

    """
    q, r = gf_div(f, g, p, K)

    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)


def gf_lshift(f, n, K):
    """
    Efficiently multiply ``f`` by ``x**n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_lshift

    >>> gf_lshift([3, 2, 4], 4, ZZ)
    [3, 2, 4, 0, 0, 0, 0]

    """
    if not f:
        return f
    else:
        return f + [K.zero]*n


def gf_rshift(f, n, K):
    """
    Efficiently divide ``f`` by ``x**n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_rshift

    >>> gf_rshift([1, 2, 3, 4, 0], 3, ZZ)
    ([1, 2], [3, 4, 0])

    """
    if not n:
        return f, []
    else:
        return f[:-n], f[-n:]


def gf_pow(f, n, p, K):
    """
    Compute ``f**n`` in ``GF(p)[x]`` using repeated squaring.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_pow

    >>> gf_pow([3, 2, 4], 3, 5, ZZ)
    [2, 4, 4, 2, 2, 1, 4]

    """
    if not n:
        return [K.one]
    elif n == 1:
        return f
    elif n == 2:
        return gf_sqr(f, p, K)

    h = [K.one]

    while True:
        if n & 1:
            h = gf_mul(h, f, p, K)
            n -= 1

        n >>= 1

        if not n:
            break

        f = gf_sqr(f, p, K)

    return h

def gf_frobenius_monomial_base(g, p, K):
    """
    return the list of ``x**(i*p) mod g in Z_p`` for ``i = 0, .., n - 1``
    where ``n = gf_degree(g)``

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_frobenius_monomial_base
    >>> g = ZZ.map([1, 0, 2, 1])
    >>> gf_frobenius_monomial_base(g, 5, ZZ)
    [[1], [4, 4, 2], [1, 2]]

    """
    n = gf_degree(g)
    if n == 0:
        return []
    b = [0]*n
    b[0] = [1]
    if p < n:
        for i in range(1, n):
            mon = gf_lshift(b[i - 1], p, K)
            b[i] = gf_rem(mon, g, p, K)
    elif n > 1:
        b[1] = gf_pow_mod([K.one, K.zero], p, g, p, K)
        for i in range(2, n):
            b[i] = gf_mul(b[i - 1], b[1], p, K)
            b[i] = gf_rem(b[i], g, p, K)

    return b

def gf_frobenius_map(f, g, b, p, K):
    """
    compute gf_pow_mod(f, p, g, p, K) using the Frobenius map

    Parameters
    ==========

    f, g : polynomials in ``GF(p)[x]``
    b : frobenius monomial base
    p : prime number
    K : domain

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_frobenius_monomial_base, gf_frobenius_map
    >>> f = ZZ.map([2, 1, 0, 1])
    >>> g = ZZ.map([1, 0, 2, 1])
    >>> p = 5
    >>> b = gf_frobenius_monomial_base(g, p, ZZ)
    >>> r = gf_frobenius_map(f, g, b, p, ZZ)
    >>> gf_frobenius_map(f, g, b, p, ZZ)
    [4, 0, 3]
    """
    m = gf_degree(g)
    if gf_degree(f) >= m:
        f = gf_rem(f, g, p, K)
    if not f:
        return []
    n = gf_degree(f)
    sf = [f[-1]]
    for i in range(1, n + 1):
        v = gf_mul_ground(b[i], f[n - i], p, K)
        sf = gf_add(sf, v, p, K)
    return sf

def _gf_pow_pnm1d2(f, n, g, b, p, K):
    """
    utility function for ``gf_edf_zassenhaus``
    Compute ``f**((p**n - 1) // 2)`` in ``GF(p)[x]/(g)``
    ``f**((p**n - 1) // 2) = (f*f**p*...*f**(p**n - 1))**((p - 1) // 2)``
    """
    f = gf_rem(f, g, p, K)
    h = f
    r = f
    for i in range(1, n):
        h = gf_frobenius_map(h, g, b, p, K)
        r = gf_mul(r, h, p, K)
        r = gf_rem(r, g, p, K)

    res = gf_pow_mod(r, (p - 1)//2, g, p, K)
    return res

def gf_pow_mod(f, n, g, p, K):
    """
    Compute ``f**n`` in ``GF(p)[x]/(g)`` using repeated squaring.

    Given polynomials ``f`` and ``g`` in ``GF(p)[x]`` and a non-negative
    integer ``n``, efficiently computes ``f**n (mod g)`` i.e. the remainder
    of ``f**n`` from division by ``g``, using the repeated squaring algorithm.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_pow_mod

    >>> gf_pow_mod(ZZ.map([3, 2, 4]), 3, ZZ.map([1, 1]), 5, ZZ)
    []

    References
    ==========

    .. [1] [Gathen99]_

    """
    if not n:
        return [K.one]
    elif n == 1:
        return gf_rem(f, g, p, K)
    elif n == 2:
        return gf_rem(gf_sqr(f, p, K), g, p, K)

    h = [K.one]

    while True:
        if n & 1:
            h = gf_mul(h, f, p, K)
            h = gf_rem(h, g, p, K)
            n -= 1

        n >>= 1

        if not n:
            break

        f = gf_sqr(f, p, K)
        f = gf_rem(f, g, p, K)

    return h


def gf_gcd(f, g, p, K):
    """
    Euclidean Algorithm in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_gcd

    >>> gf_gcd(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    [1, 3]

    """
    while g:
        f, g = g, gf_rem(f, g, p, K)

    return gf_monic(f, p, K)[1]


def gf_lcm(f, g, p, K):
    """
    Compute polynomial LCM in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_lcm

    >>> gf_lcm(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    [1, 2, 0, 4]

    """
    if not f or not g:
        return []

    h = gf_quo(gf_mul(f, g, p, K),
               gf_gcd(f, g, p, K), p, K)

    return gf_monic(h, p, K)[1]


def gf_cofactors(f, g, p, K):
    """
    Compute polynomial GCD and cofactors in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_cofactors

    >>> gf_cofactors(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    ([1, 3], [3, 3], [2, 1])

    """
    if not f and not g:
        return ([], [], [])

    h = gf_gcd(f, g, p, K)

    return (h, gf_quo(f, h, p, K),
            gf_quo(g, h, p, K))


def gf_gcdex(f, g, p, K):
    """
    Extended Euclidean Algorithm in ``GF(p)[x]``.

    Given polynomials ``f`` and ``g`` in ``GF(p)[x]``, computes polynomials
    ``s``, ``t`` and ``h``, such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.
    The typical application of EEA is solving polynomial diophantine equations.

    Consider polynomials ``f = (x + 7) (x + 1)``, ``g = (x + 7) (x**2 + 1)``
    in ``GF(11)[x]``. Application of Extended Euclidean Algorithm gives::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_gcdex, gf_mul, gf_add

       >>> s, t, g = gf_gcdex(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ)
       >>> s, t, g
       ([5, 6], [6], [1, 7])

    As result we obtained polynomials ``s = 5*x + 6`` and ``t = 6``, and
    additionally ``gcd(f, g) = x + 7``. This is correct because::

       >>> S = gf_mul(s, ZZ.map([1, 8, 7]), 11, ZZ)
       >>> T = gf_mul(t, ZZ.map([1, 7, 1, 7]), 11, ZZ)

       >>> gf_add(S, T, 11, ZZ) == [1, 7]
       True

    References
    ==========

    .. [1] [Gathen99]_

    """
    if not (f or g):
        return [K.one], [], []

    p0, r0 = gf_monic(f, p, K)
    p1, r1 = gf_monic(g, p, K)

    if not f:
        return [], [K.invert(p1, p)], r1
    if not g:
        return [K.invert(p0, p)], [], r0

    s0, s1 = [K.invert(p0, p)], []
    t0, t1 = [], [K.invert(p1, p)]

    while True:
        Q, R = gf_div(r0, r1, p, K)

        if not R:
            break

        (lc, r1), r0 = gf_monic(R, p, K), r1

        inv = K.invert(lc, p)

        s = gf_sub_mul(s0, s1, Q, p, K)
        t = gf_sub_mul(t0, t1, Q, p, K)

        s1, s0 = gf_mul_ground(s, inv, p, K), s1
        t1, t0 = gf_mul_ground(t, inv, p, K), t1

    return s1, t1, r1


def gf_monic(f, p, K):
    """
    Compute LC and a monic polynomial in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_monic

    >>> gf_monic(ZZ.map([3, 2, 4]), 5, ZZ)
    (3, [1, 4, 3])

    """
    if not f:
        return K.zero, []
    else:
        lc = f[0]

        if K.is_one(lc):
            return lc, list(f)
        else:
            return lc, gf_quo_ground(f, lc, p, K)


def gf_diff(f, p, K):
    """
    Differentiate polynomial in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_diff

    >>> gf_diff([3, 2, 4], 5, ZZ)
    [1, 2]

    """
    df = gf_degree(f)

    h, n = [K.zero]*df, df

    for coeff in f[:-1]:
        coeff *= K(n)
        coeff %= p

        if coeff:
            h[df - n] = coeff

        n -= 1

    return gf_strip(h)


def gf_eval(f, a, p, K):
    """
    Evaluate ``f(a)`` in ``GF(p)`` using Horner scheme.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_eval

    >>> gf_eval([3, 2, 4], 2, 5, ZZ)
    0

    """
    result = K.zero

    for c in f:
        result *= a
        result += c
        result %= p

    return result


def gf_multi_eval(f, A, p, K):
    """
    Evaluate ``f(a)`` for ``a`` in ``[a_1, ..., a_n]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_multi_eval

    >>> gf_multi_eval([3, 2, 4], [0, 1, 2, 3, 4], 5, ZZ)
    [4, 4, 0, 2, 0]

    """
    return [ gf_eval(f, a, p, K) for a in A ]


def gf_compose(f, g, p, K):
    """
    Compute polynomial composition ``f(g)`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_compose

    >>> gf_compose([3, 2, 4], [2, 2, 2], 5, ZZ)
    [2, 4, 0, 3, 0]

    """
    if len(g) <= 1:
        return gf_strip([gf_eval(f, gf_LC(g, K), p, K)])

    if not f:
        return []

    h = [f[0]]

    for c in f[1:]:
        h = gf_mul(h, g, p, K)
        h = gf_add_ground(h, c, p, K)

    return h


def gf_compose_mod(g, h, f, p, K):
    """
    Compute polynomial composition ``g(h)`` in ``GF(p)[x]/(f)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_compose_mod

    >>> gf_compose_mod(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 2]), ZZ.map([4, 3]), 5, ZZ)
    [4]

    """
    if not g:
        return []

    comp = [g[0]]

    for a in g[1:]:
        comp = gf_mul(comp, h, p, K)
        comp = gf_add_ground(comp, a, p, K)
        comp = gf_rem(comp, f, p, K)

    return comp


def gf_trace_map(a, b, c, n, f, p, K):
    """
    Compute polynomial trace map in ``GF(p)[x]/(f)``.

    Given a polynomial ``f`` in ``GF(p)[x]``, polynomials ``a``, ``b``,
    ``c`` in the quotient ring ``GF(p)[x]/(f)`` such that ``b = c**t
    (mod f)`` for some positive power ``t`` of ``p``, and a positive
    integer ``n``, returns a mapping::

       a -> a**t**n, a + a**t + a**t**2 + ... + a**t**n (mod f)

    In factorization context, ``b = x**p mod f`` and ``c = x mod f``.
    This way we can efficiently compute trace polynomials in equal
    degree factorization routine, much faster than with other methods,
    like iterated Frobenius algorithm, for large degrees.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_trace_map

    >>> gf_trace_map([1, 2], [4, 4], [1, 1], 4, [3, 2, 4], 5, ZZ)
    ([1, 3], [1, 3])

    References
    ==========

    .. [1] [Gathen92]_

    """
    u = gf_compose_mod(a, b, f, p, K)
    v = b

    if n & 1:
        U = gf_add(a, u, p, K)
        V = b
    else:
        U = a
        V = c

    n >>= 1

    while n:
        u = gf_add(u, gf_compose_mod(u, v, f, p, K), p, K)
        v = gf_compose_mod(v, v, f, p, K)

        if n & 1:
            U = gf_add(U, gf_compose_mod(u, V, f, p, K), p, K)
            V = gf_compose_mod(v, V, f, p, K)

        n >>= 1

    return gf_compose_mod(a, V, f, p, K), U

def _gf_trace_map(f, n, g, b, p, K):
    """
    utility for ``gf_edf_shoup``
    """
    f = gf_rem(f, g, p, K)
    h = f
    r = f
    for i in range(1, n):
        h = gf_frobenius_map(h, g, b, p, K)
        r = gf_add(r, h, p, K)
        r = gf_rem(r, g, p, K)
    return r


def gf_random(n, p, K):
    """
    Generate a random polynomial in ``GF(p)[x]`` of degree ``n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_random
    >>> gf_random(10, 5, ZZ) #doctest: +SKIP
    [1, 2, 3, 2, 1, 1, 1, 2, 0, 4, 2]

    """
    pi = int(p)
    return [K.one] + [ K(int(uniform(0, pi))) for i in range(0, n) ]


def gf_irreducible(n, p, K):
    """
    Generate random irreducible polynomial of degree ``n`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irreducible
    >>> gf_irreducible(10, 5, ZZ) #doctest: +SKIP
    [1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]

    """
    while True:
        f = gf_random(n, p, K)
        if gf_irreducible_p(f, p, K):
            return f


def gf_irred_p_ben_or(f, p, K):
    """
    Ben-Or's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_ben_or

    >>> gf_irred_p_ben_or(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_ben_or(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)

    if n <= 1:
        return True

    _, f = gf_monic(f, p, K)
    if n < 5:
        H = h = gf_pow_mod([K.one, K.zero], p, f, p, K)

        for i in range(0, n//2):
            g = gf_sub(h, [K.one, K.zero], p, K)

            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_compose_mod(h, H, f, p, K)
            else:
                return False
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
        for i in range(0, n//2):
            g = gf_sub(h, [K.one, K.zero], p, K)
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_frobenius_map(h, f, b, p, K)
            else:
                return False

    return True


def gf_irred_p_rabin(f, p, K):
    """
    Rabin's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_rabin

    >>> gf_irred_p_rabin(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_rabin(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)

    if n <= 1:
        return True

    _, f = gf_monic(f, p, K)

    x = [K.one, K.zero]

    from sympy.ntheory import factorint

    indices = { n//d for d in factorint(n) }

    b = gf_frobenius_monomial_base(f, p, K)
    h = b[1]

    for i in range(1, n):
        if i in indices:
            g = gf_sub(h, x, p, K)

            if gf_gcd(f, g, p, K) != [K.one]:
                return False

        h = gf_frobenius_map(h, f, b, p, K)

    return h == x

_irred_methods = {
    'ben-or': gf_irred_p_ben_or,
    'rabin': gf_irred_p_rabin,
}


def gf_irreducible_p(f, p, K):
    """
    Test irreducibility of a polynomial ``f`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irreducible_p

    >>> gf_irreducible_p(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irreducible_p(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    method = query('GF_IRRED_METHOD')

    if method is not None:
        irred = _irred_methods[method](f, p, K)
    else:
        irred = gf_irred_p_rabin(f, p, K)

    return irred


def gf_sqf_p(f, p, K):
    """
    Return ``True`` if ``f`` is square-free in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_p

    >>> gf_sqf_p(ZZ.map([3, 2, 4]), 5, ZZ)
    True
    >>> gf_sqf_p(ZZ.map([2, 4, 4, 2, 2, 1, 4]), 5, ZZ)
    False

    """
    _, f = gf_monic(f, p, K)

    if not f:
        return True
    else:
        return gf_gcd(f, gf_diff(f, p, K), p, K) == [K.one]


def gf_sqf_part(f, p, K):
    """
    Return square-free part of a ``GF(p)[x]`` polynomial.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_part

    >>> gf_sqf_part(ZZ.map([1, 1, 3, 0, 1, 0, 2, 2, 1]), 5, ZZ)
    [1, 4, 3]

    """
    _, sqf = gf_sqf_list(f, p, K)

    g = [K.one]

    for f, _ in sqf:
        g = gf_mul(g, f, p, K)

    return g


def gf_sqf_list(f, p, K, all=False):
    """
    Return the square-free decomposition of a ``GF(p)[x]`` polynomial.

    Given a polynomial ``f`` in ``GF(p)[x]``, returns the leading coefficient
    of ``f`` and a square-free decomposition ``f_1**e_1 f_2**e_2 ... f_k**e_k``
    such that all ``f_i`` are monic polynomials and ``(f_i, f_j)`` for ``i != j``
    are co-prime and ``e_1 ... e_k`` are given in increasing order. All trivial
    terms (i.e. ``f_i = 1``) are not included in the output.

    Consider polynomial ``f = x**11 + 1`` over ``GF(11)[x]``::

       >>> from sympy.polys.domains import ZZ

       >>> from sympy.polys.galoistools import (
       ...     gf_from_dict, gf_diff, gf_sqf_list, gf_pow,
       ... )
       ... # doctest: +NORMALIZE_WHITESPACE

       >>> f = gf_from_dict({11: ZZ(1), 0: ZZ(1)}, 11, ZZ)

    Note that ``f'(x) = 0``::

       >>> gf_diff(f, 11, ZZ)
       []

    This phenomenon does not happen in characteristic zero. However we can
    still compute square-free decomposition of ``f`` using ``gf_sqf()``::

       >>> gf_sqf_list(f, 11, ZZ)
       (1, [([1, 1], 11)])

    We obtained factorization ``f = (x + 1)**11``. This is correct because::

       >>> gf_pow([1, 1], 11, 11, ZZ) == f
       True

    References
    ==========

    .. [1] [Geddes92]_

    """
    n, sqf, factors, r = 1, False, [], int(p)

    lc, f = gf_monic(f, p, K)

    if gf_degree(f) < 1:
        return lc, []

    while True:
        F = gf_diff(f, p, K)

        if F != []:
            g = gf_gcd(f, F, p, K)
            h = gf_quo(f, g, p, K)

            i = 1

            while h != [K.one]:
                G = gf_gcd(g, h, p, K)
                H = gf_quo(h, G, p, K)

                if gf_degree(H) > 0:
                    factors.append((H, i*n))

                g, h, i = gf_quo(g, G, p, K), G, i + 1

            if g == [K.one]:
                sqf = True
            else:
                f = g

        if not sqf:
            d = gf_degree(f) // r

            for i in range(0, d + 1):
                f[i] = f[i*r]

            f, n = f[:d + 1], n*r
        else:
            break

    if all:
        raise ValueError("'all=True' is not supported yet")

    return lc, factors


def gf_Qmatrix(f, p, K):
    """
    Calculate Berlekamp's ``Q`` matrix.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix

    >>> gf_Qmatrix([3, 2, 4], 5, ZZ)
    [[1, 0],
     [3, 4]]

    >>> gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 0, 0],
     [0, 4, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 4]]

    """
    n, r = gf_degree(f), int(p)

    q = [K.one] + [K.zero]*(n - 1)
    Q = [list(q)] + [[]]*(n - 1)

    for i in range(1, (n - 1)*r + 1):
        qq, c = [(-q[-1]*f[-1]) % p], q[-1]

        for j in range(1, n):
            qq.append((q[j - 1] - c*f[-j - 1]) % p)

        if not (i % r):
            Q[i//r] = list(qq)

        q = qq

    return Q


def gf_Qbasis(Q, p, K):
    """
    Compute a basis of the kernel of ``Q``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix, gf_Qbasis

    >>> gf_Qbasis(gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ), 5, ZZ)
    [[1, 0, 0, 0], [0, 0, 1, 0]]

    >>> gf_Qbasis(gf_Qmatrix([3, 2, 4], 5, ZZ), 5, ZZ)
    [[1, 0]]

    """
    Q, n = [ list(q) for q in Q ], len(Q)

    for k in range(0, n):
        Q[k][k] = (Q[k][k] - K.one) % p

    for k in range(0, n):
        for i in range(k, n):
            if Q[k][i]:
                break
        else:
            continue

        inv = K.invert(Q[k][i], p)

        for j in range(0, n):
            Q[j][i] = (Q[j][i]*inv) % p

        for j in range(0, n):
            t = Q[j][k]
            Q[j][k] = Q[j][i]
            Q[j][i] = t

        for i in range(0, n):
            if i != k:
                q = Q[k][i]

                for j in range(0, n):
                    Q[j][i] = (Q[j][i] - Q[j][k]*q) % p

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                Q[i][j] = (K.one - Q[i][j]) % p
            else:
                Q[i][j] = (-Q[i][j]) % p

    basis = []

    for q in Q:
        if any(q):
            basis.append(q)

    return basis


def gf_berlekamp(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for small ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_berlekamp

    >>> gf_berlekamp([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 2], [1, 0, 3]]

    """
    Q = gf_Qmatrix(f, p, K)
    V = gf_Qbasis(Q, p, K)

    for i, v in enumerate(V):
        V[i] = gf_strip(list(reversed(v)))

    factors = [f]

    for k in range(1, len(V)):
        for f in list(factors):
            s = K.zero

            while s < p:
                g = gf_sub_ground(V[k], s, p, K)
                h = gf_gcd(f, g, p, K)

                if h != [K.one] and h != f:
                    factors.remove(f)

                    f = gf_quo(f, h, p, K)
                    factors.extend([f, h])

                if len(factors) == len(V):
                    return _sort_factors(factors, multiple=False)

                s += K.one

    return _sort_factors(factors, multiple=False)


def gf_ddf_zassenhaus(f, p, K):
    """
    Cantor-Zassenhaus: Deterministic Distinct Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes
    partial distinct degree factorization ``f_1 ... f_d`` of ``f`` where
    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a
    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``
    is an argument to the equal degree factorization routine.

    Consider the polynomial ``x**15 - 1`` in ``GF(11)[x]``::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_from_dict

       >>> f = gf_from_dict({15: ZZ(1), 0: ZZ(-1)}, 11, ZZ)

    Distinct degree factorization gives::

       >>> from sympy.polys.galoistools import gf_ddf_zassenhaus

       >>> gf_ddf_zassenhaus(f, 11, ZZ)
       [([1, 0, 0, 0, 0, 10], 1), ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 2)]

    which means ``x**15 - 1 = (x**5 - 1) (x**10 + x**5 + 1)``. To obtain
    factorization into irreducibles, use equal degree factorization
    procedure (EDF) with each of the factors.

    References
    ==========

    .. [1] [Gathen99]_
    .. [2] [Geddes92]_

    """
    i, g, factors = 1, [K.one, K.zero], []

    b = gf_frobenius_monomial_base(f, p, K)
    while 2*i <= gf_degree(f):
        g = gf_frobenius_map(g, f, b, p, K)
        h = gf_gcd(f, gf_sub(g, [K.one, K.zero], p, K), p, K)

        if h != [K.one]:
            factors.append((h, i))

            f = gf_quo(f, h, p, K)
            g = gf_rem(g, f, p, K)
            b = gf_frobenius_monomial_base(f, p, K)

        i += 1

    if f != [K.one]:
        return factors + [(f, gf_degree(f))]
    else:
        return factors


def gf_edf_zassenhaus(f, n, p, K):
    """
    Cantor-Zassenhaus: Probabilistic Equal Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and
    an integer ``n``, such that ``n`` divides ``deg(f)``, returns all
    irreducible factors ``f_1,...,f_d`` of ``f``, each of degree ``n``.
    EDF procedure gives complete factorization over Galois fields.

    Consider the square-free polynomial ``f = x**3 + x**2 + x + 1`` in
    ``GF(5)[x]``. Let's compute its irreducible factors of degree one::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_edf_zassenhaus

       >>> gf_edf_zassenhaus([1,1,1,1], 1, 5, ZZ)
       [[1, 1], [1, 2], [1, 3]]

    Notes
    =====

    The case p == 2 is handled by Cohen's Algorithm 3.4.8. The case p odd is
    as in Geddes Algorithm 8.9 (or Cohen's Algorithm 3.4.6).

    References
    ==========

    .. [1] [Gathen99]_
    .. [2] [Geddes92]_ Algorithm 8.9
    .. [3] [Cohen93]_ Algorithm 3.4.8

    """
    factors = [f]

    if gf_degree(f) <= n:
        return factors

    N = gf_degree(f) // n
    if p != 2:
        b = gf_frobenius_monomial_base(f, p, K)

    t = [K.one, K.zero]
    while len(factors) < N:
        if p == 2:
            h = r = t

            for i in range(n - 1):
                r = gf_pow_mod(r, 2, f, p, K)
                h = gf_add(h, r, p, K)

            g = gf_gcd(f, h, p, K)
            t += [K.zero, K.zero]
        else:
            r = gf_random(2 * n - 1, p, K)
            h = _gf_pow_pnm1d2(r, n, f, b, p, K)
            g = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)

        if g != [K.one] and g != f:
            factors = gf_edf_zassenhaus(g, n, p, K) \
                + gf_edf_zassenhaus(gf_quo(f, g, p, K), n, p, K)

    return _sort_factors(factors, multiple=False)


def gf_ddf_shoup(f, p, K):
    """
    Kaltofen-Shoup: Deterministic Distinct Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes
    partial distinct degree factorization ``f_1,...,f_d`` of ``f`` where
    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a
    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``
    is an argument to the equal degree factorization routine.

    This algorithm is an improved version of Zassenhaus algorithm for
    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_ddf_shoup, gf_from_dict

    >>> f = gf_from_dict({6: ZZ(1), 5: ZZ(-1), 4: ZZ(1), 3: ZZ(1), 1: ZZ(-1)}, 3, ZZ)

    >>> gf_ddf_shoup(f, 3, ZZ)
    [([1, 1, 0], 1), ([1, 1, 0, 1, 2], 2)]

    References
    ==========

    .. [1] [Kaltofen98]_
    .. [2] [Shoup95]_
    .. [3] [Gathen92]_

    """
    n = gf_degree(f)
    k = int(_ceil(_sqrt(n//2)))
    b = gf_frobenius_monomial_base(f, p, K)
    h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
    # U[i] = x**(p**i)
    U = [[K.one, K.zero], h] + [K.zero]*(k - 1)

    for i in range(2, k + 1):
        U[i] = gf_frobenius_map(U[i-1], f, b, p, K)

    h, U = U[k], U[:k]
    # V[i] = x**(p**(k*(i+1)))
    V = [h] + [K.zero]*(k - 1)

    for i in range(1, k):
        V[i] = gf_compose_mod(V[i - 1], h, f, p, K)

    factors = []

    for i, v in enumerate(V):
        h, j = [K.one], k - 1

        for u in U:
            g = gf_sub(v, u, p, K)
            h = gf_mul(h, g, p, K)
            h = gf_rem(h, f, p, K)

        g = gf_gcd(f, h, p, K)
        f = gf_quo(f, g, p, K)

        for u in reversed(U):
            h = gf_sub(v, u, p, K)
            F = gf_gcd(g, h, p, K)

            if F != [K.one]:
                factors.append((F, k*(i + 1) - j))

            g, j = gf_quo(g, F, p, K), j - 1

    if f != [K.one]:
        factors.append((f, gf_degree(f)))

    return factors

def gf_edf_shoup(f, n, p, K):
    """
    Gathen-Shoup: Probabilistic Equal Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and integer
    ``n`` such that ``n`` divides ``deg(f)``, returns all irreducible factors
    ``f_1,...,f_d`` of ``f``, each of degree ``n``. This is a complete
    factorization over Galois fields.

    This algorithm is an improved version of Zassenhaus algorithm for
    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_edf_shoup

    >>> gf_edf_shoup(ZZ.map([1, 2837, 2277]), 1, 2917, ZZ)
    [[1, 852], [1, 1985]]

    References
    ==========

    .. [1] [Shoup91]_
    .. [2] [Gathen92]_

    """
    N, q = gf_degree(f), int(p)

    if not N:
        return []
    if N <= n:
        return [f]

    factors, x = [f], [K.one, K.zero]

    r = gf_random(N - 1, p, K)

    if p == 2:
        h = gf_pow_mod(x, q, f, p, K)
        H = gf_trace_map(r, h, x, n - 1, f, p, K)[1]
        h1 = gf_gcd(f, H, p, K)
        h2 = gf_quo(f, h1, p, K)

        factors = gf_edf_shoup(h1, n, p, K) \
            + gf_edf_shoup(h2, n, p, K)
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = _gf_trace_map(r, n, f, b, p, K)
        h = gf_pow_mod(H, (q - 1)//2, f, p, K)

        h1 = gf_gcd(f, h, p, K)
        h2 = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)
        h3 = gf_quo(f, gf_mul(h1, h2, p, K), p, K)

        factors = gf_edf_shoup(h1, n, p, K) \
            + gf_edf_shoup(h2, n, p, K) \
            + gf_edf_shoup(h3, n, p, K)

    return _sort_factors(factors, multiple=False)


def gf_zassenhaus(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for medium ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_zassenhaus

    >>> gf_zassenhaus(ZZ.map([1, 4, 3]), 5, ZZ)
    [[1, 1], [1, 3]]

    """
    factors = []

    for factor, n in gf_ddf_zassenhaus(f, p, K):
        factors += gf_edf_zassenhaus(factor, n, p, K)

    return _sort_factors(factors, multiple=False)


def gf_shoup(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for large ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_shoup

    >>> gf_shoup(ZZ.map([1, 4, 3]), 5, ZZ)
    [[1, 1], [1, 3]]

    """
    factors = []

    for factor, n in gf_ddf_shoup(f, p, K):
        factors += gf_edf_shoup(factor, n, p, K)

    return _sort_factors(factors, multiple=False)

_factor_methods = {
    'berlekamp': gf_berlekamp,  # ``p`` : small
    'zassenhaus': gf_zassenhaus,  # ``p`` : medium
    'shoup': gf_shoup,      # ``p`` : large
}


def gf_factor_sqf(f, p, K, method=None):
    """
    Factor a square-free polynomial ``f`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_factor_sqf

    >>> gf_factor_sqf(ZZ.map([3, 2, 4]), 5, ZZ)
    (3, [[1, 1], [1, 3]])

    """
    lc, f = gf_monic(f, p, K)

    if gf_degree(f) < 1:
        return lc, []

    method = method or query('GF_FACTOR_METHOD')

    if method is not None:
        factors = _factor_methods[method](f, p, K)
    else:
        factors = gf_zassenhaus(f, p, K)

    return lc, factors


def gf_factor(f, p, K):
    """
    Factor (non square-free) polynomials in ``GF(p)[x]``.

    Given a possibly non square-free polynomial ``f`` in ``GF(p)[x]``,
    returns its complete factorization into irreducibles::

                 f_1(x)**e_1 f_2(x)**e_2 ... f_d(x)**e_d

    where each ``f_i`` is a monic polynomial and ``gcd(f_i, f_j) == 1``,
    for ``i != j``.  The result is given as a tuple consisting of the
    leading coefficient of ``f`` and a list of factors of ``f`` with
    their multiplicities.

    The algorithm proceeds by first computing square-free decomposition
    of ``f`` and then iteratively factoring each of square-free factors.

    Consider a non square-free polynomial ``f = (7*x + 1) (x + 2)**2`` in
    ``GF(11)[x]``. We obtain its factorization into irreducibles as follows::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_factor

       >>> gf_factor(ZZ.map([5, 2, 7, 2]), 11, ZZ)
       (5, [([1, 2], 1), ([1, 8], 2)])

    We arrived with factorization ``f = 5 (x + 2) (x + 8)**2``. We did not
    recover the exact form of the input polynomial because we requested to
    get monic factors of ``f`` and its leading coefficient separately.

    Square-free factors of ``f`` can be factored into irreducibles over
    ``GF(p)`` using three very different methods:

    Berlekamp
        efficient for very small values of ``p`` (usually ``p < 25``)
    Cantor-Zassenhaus
        efficient on average input and with "typical" ``p``
    Shoup-Kaltofen-Gathen
        efficient with very large inputs and modulus

    If you want to use a specific factorization method, instead of the default
    one, set ``GF_FACTOR_METHOD`` with one of ``berlekamp``, ``zassenhaus`` or
    ``shoup`` values.

    References
    ==========

    .. [1] [Gathen99]_

    """
    lc, f = gf_monic(f, p, K)

    if gf_degree(f) < 1:
        return lc, []

    factors = []

    for g, n in gf_sqf_list(f, p, K)[1]:
        for h in gf_factor_sqf(g, p, K)[1]:
            factors.append((h, n))

    return lc, _sort_factors(factors)


def gf_value(f, a):
    """
    Value of polynomial 'f' at 'a' in field R.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_value

    >>> gf_value([1, 7, 2, 4], 11)
    2204

    """
    result = 0
    for c in f:
        result *= a
        result += c
    return result


def linear_congruence(a, b, m):
    """
    Returns the values of x satisfying a*x congruent b mod(m)

    Here m is positive integer and a, b are natural numbers.
    This function returns only those values of x which are distinct mod(m).

    Examples
    ========

    >>> from sympy.polys.galoistools import linear_congruence

    >>> linear_congruence(3, 12, 15)
    [4, 9, 14]

    There are 3 solutions distinct mod(15) since gcd(a, m) = gcd(3, 15) = 3.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Linear_congruence_theorem

    """
    from sympy.polys.polytools import gcdex
    if a % m == 0:
        if b % m == 0:
            return list(range(m))
        else:
            return []
    r, _, g = gcdex(a, m)
    if b % g != 0:
        return []
    return [(r * b // g + t * m // g) % m for t in range(g)]


def _raise_mod_power(x, s, p, f):
    """
    Used in gf_csolve to generate solutions of f(x) cong 0 mod(p**(s + 1))
    from the solutions of f(x) cong 0 mod(p**s).

    Examples
    ========

    >>> from sympy.polys.galoistools import _raise_mod_power
    >>> from sympy.polys.galoistools import csolve_prime

    These is the solutions of f(x) = x**2 + x + 7 cong 0 mod(3)

    >>> f = [1, 1, 7]
    >>> csolve_prime(f, 3)
    [1]
    >>> [ i for i in range(3) if not (i**2 + i + 7) % 3]
    [1]

    The solutions of f(x) cong 0 mod(9) are constructed from the
    values returned from _raise_mod_power:

    >>> x, s, p = 1, 1, 3
    >>> V = _raise_mod_power(x, s, p, f)
    >>> [x + v * p**s for v in V]
    [1, 4, 7]

    And these are confirmed with the following:

    >>> [ i for i in range(3**2) if not (i**2 + i + 7) % 3**2]
    [1, 4, 7]

    """
    from sympy.polys.domains import ZZ
    f_f = gf_diff(f, p, ZZ)
    alpha = gf_value(f_f, x)
    beta = - gf_value(f, x) // p**s
    return linear_congruence(alpha, beta, p)


def _csolve_prime_las_vegas(f, p, seed=None):
    r""" Solutions of `f(x) \equiv 0 \pmod{p}`, `f(0) \not\equiv 0 \pmod{p}`.

    Explanation
    ===========

    This algorithm is classified as the Las Vegas method.
    That is, it always returns the correct answer and solves the problem
    fast in many cases, but if it is unlucky, it does not answer forever.

    Suppose the polynomial f is not a zero polynomial. Assume further
    that it is of degree at most p-1 and `f(0)\not\equiv 0 \pmod{p}`.
    These assumptions are not an essential part of the algorithm,
    only that it is more convenient for the function calling this
    function to resolve them.

    Note that `x^{p-1} - 1 \equiv \prod_{a=1}^{p-1}(x - a) \pmod{p}`.
    Thus, the greatest common divisor with f is `\prod_{s \in S}(x - s)`,
    with S being the set of solutions to f. Furthermore,
    when a is randomly determined, `(x+a)^{(p-1)/2}-1` is
    a polynomial with (p-1)/2 randomly chosen solutions.
    The greatest common divisor of f may be a nontrivial factor of f.

    When p is large and the degree of f is small,
    it is faster than naive solution methods.

    Parameters
    ==========

    f : polynomial
    p : prime number

    Returns
    =======

    list[int]
        a list of solutions, sorted in ascending order
        by integers in the range [1, p). The same value
        does not exist in the list even if there is
        a multiple solution. If no solution exists, returns [].

    Examples
    ========

    >>> from sympy.polys.galoistools import _csolve_prime_las_vegas
    >>> _csolve_prime_las_vegas([1, 4, 3], 7) # x^2 + 4x + 3 = 0 (mod 7)
    [4, 6]
    >>> _csolve_prime_las_vegas([5, 7, 1, 9], 11) # 5x^3 + 7x^2 + x + 9 = 0 (mod 11)
    [1, 5, 8]

    References
    ==========

    .. [1] R. Crandall and C. Pomerance "Prime Numbers", 2nd Ed., Algorithm 2.3.10

    """
    from sympy.polys.domains import ZZ
    from sympy.ntheory import sqrt_mod
    randint = _randint(seed)
    root = set()
    g = gf_pow_mod([1, 0], p - 1, f, p, ZZ)
    g = gf_sub_ground(g, 1, p, ZZ)
    # We want to calculate gcd(x**(p-1) - 1, f(x))
    factors = [gf_gcd(f, g, p, ZZ)]
    while factors:
        f = factors.pop()
        # If the degree is small, solve directly
        if len(f) <= 1:
            continue
        if len(f) == 2:
            root.add(-invert(f[0], p) * f[1] % p)
            continue
        if len(f) == 3:
            inv = invert(f[0], p)
            b = f[1] * inv % p
            b = (b + p * (b % 2)) // 2
            root.update((r - b) % p for r in
                        sqrt_mod(b**2 - f[2] * inv, p, all_roots=True))
            continue
        while True:
            # Determine `a` randomly and
            # compute gcd((x+a)**((p-1)//2)-1, f(x))
            a = randint(0, p - 1)
            g = gf_pow_mod([1, a], (p - 1) // 2, f, p, ZZ)
            g = gf_sub_ground(g, 1, p, ZZ)
            g = gf_gcd(f, g, p, ZZ)
            if 1 < len(g) < len(f):
                factors.append(g)
                factors.append(gf_div(f, g, p, ZZ)[0])
                break
    return sorted(root)


def csolve_prime(f, p, e=1):
    r""" Solutions of `f(x) \equiv 0 \pmod{p^e}`.

    Parameters
    ==========

    f : polynomial
    p : prime number
    e : positive integer

    Returns
    =======

    list[int]
        a list of solutions, sorted in ascending order
        by integers in the range [1, p**e). The same value
        does not exist in the list even if there is
        a multiple solution. If no solution exists, returns [].

    Examples
    ========

    >>> from sympy.polys.galoistools import csolve_prime
    >>> csolve_prime([1, 1, 7], 3, 1)
    [1]
    >>> csolve_prime([1, 1, 7], 3, 2)
    [1, 4, 7]

    Solutions [7, 4, 1] (mod 3**2) are generated by ``_raise_mod_power()``
    from solution [1] (mod 3).
    """
    from sympy.polys.domains import ZZ
    g = [MPZ(int(c)) for c in f]
    # Convert to polynomial of degree at most p-1
    for i in range(len(g) - p):
        g[i + p - 1] += g[i]
        g[i] = 0
    g = gf_trunc(g, p)
    # Checks whether g(x) is divisible by x
    k = 0
    while k < len(g) and g[len(g) - k - 1] == 0:
        k += 1
    if k:
        g = g[:-k]
        root_zero = [0]
    else:
        root_zero = []
    if g == []:
        X1 = list(range(p))
    elif len(g)**2 < p:
        # The conditions under which `_csolve_prime_las_vegas` is faster than
        # a naive solution are worth considering.
        X1 = root_zero + _csolve_prime_las_vegas(g, p)
    else:
        X1 = root_zero + [i for i in range(p) if gf_eval(g, i, p, ZZ) == 0]
    if e == 1:
        return X1
    X = []
    S = list(zip(X1, [1]*len(X1)))
    while S:
        x, s = S.pop()
        if s == e:
            X.append(x)
        else:
            s1 = s + 1
            ps = p**s
            S.extend([(x + v*ps, s1) for v in _raise_mod_power(x, s, p, f)])
    return sorted(X)


def gf_csolve(f, n):
    """
    To solve f(x) congruent 0 mod(n).

    n is divided into canonical factors and f(x) cong 0 mod(p**e) will be
    solved for each factor. Applying the Chinese Remainder Theorem to the
    results returns the final answers.

    Examples
    ========

    Solve [1, 1, 7] congruent 0 mod(189):

    >>> from sympy.polys.galoistools import gf_csolve
    >>> gf_csolve([1, 1, 7], 189)
    [13, 49, 76, 112, 139, 175]

    See Also
    ========

    sympy.ntheory.residue_ntheory.polynomial_congruence : a higher level solving routine

    References
    ==========

    .. [1] 'An introduction to the Theory of Numbers' 5th Edition by Ivan Niven,
           Zuckerman and Montgomery.

    """
    from sympy.polys.domains import ZZ
    from sympy.ntheory import factorint
    P = factorint(n)
    X = [csolve_prime(f, p, e) for p, e in P.items()]
    pools = list(map(tuple, X))
    perms = [[]]
    for pool in pools:
        perms = [x + [y] for x in perms for y in pool]
    dist_factors = [pow(p, e) for p, e in P.items()]
    return sorted([gf_crt(per, dist_factors, ZZ) for per in perms])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\style_render.py ===
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Optional,
    TypedDict,
    Union,
)
from uuid import uuid4

import numpy as np

from pandas._config import get_option

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.common import (
    is_complex,
    is_float,
    is_integer,
)
from pandas.core.dtypes.generic import ABCSeries

from pandas import (
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    Series,
    isna,
)
from pandas.api.types import is_list_like
import pandas.core.common as com

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        Level,
    )
jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency

BaseFormatter = Union[str, Callable]
ExtFormatter = Union[BaseFormatter, dict[Any, Optional[BaseFormatter]]]
CSSPair = tuple[str, Union[str, float]]
CSSList = list[CSSPair]
CSSProperties = Union[str, CSSList]


class CSSDict(TypedDict):
    selector: str
    props: CSSProperties


CSSStyles = list[CSSDict]
Subset = Union[slice, Sequence, Index]


class StylerRenderer:
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """

    loader = jinja2.PackageLoader("pandas", "io/formats/templates")
    env = jinja2.Environment(loader=loader, trim_blocks=True)
    template_html = env.get_template("html.tpl")
    template_html_table = env.get_template("html_table.tpl")
    template_html_style = env.get_template("html_style.tpl")
    template_latex = env.get_template("latex.tpl")
    template_string = env.get_template("string.tpl")

    def __init__(
        self,
        data: DataFrame | Series,
        uuid: str | None = None,
        uuid_len: int = 5,
        table_styles: CSSStyles | None = None,
        table_attributes: str | None = None,
        caption: str | tuple | list | None = None,
        cell_ids: bool = True,
        precision: int | None = None,
    ) -> None:
        # validate ordered args
        if isinstance(data, Series):
            data = data.to_frame()
        if not isinstance(data, DataFrame):
            raise TypeError("``data`` must be a Series or DataFrame")
        self.data: DataFrame = data
        self.index: Index = data.index
        self.columns: Index = data.columns
        if not isinstance(uuid_len, int) or uuid_len < 0:
            raise TypeError("``uuid_len`` must be an integer in range [0, 32].")
        self.uuid = uuid or uuid4().hex[: min(32, uuid_len)]
        self.uuid_len = len(self.uuid)
        self.table_styles = table_styles
        self.table_attributes = table_attributes
        self.caption = caption
        self.cell_ids = cell_ids
        self.css = {
            "row_heading": "row_heading",
            "col_heading": "col_heading",
            "index_name": "index_name",
            "col": "col",
            "row": "row",
            "col_trim": "col_trim",
            "row_trim": "row_trim",
            "level": "level",
            "data": "data",
            "blank": "blank",
            "foot": "foot",
        }
        self.concatenated: list[StylerRenderer] = []
        # add rendering variables
        self.hide_index_names: bool = False
        self.hide_column_names: bool = False
        self.hide_index_: list = [False] * self.index.nlevels
        self.hide_columns_: list = [False] * self.columns.nlevels
        self.hidden_rows: Sequence[int] = []  # sequence for specific hidden rows/cols
        self.hidden_columns: Sequence[int] = []
        self.ctx: DefaultDict[tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_index: DefaultDict[tuple[int, int], CSSList] = defaultdict(list)
        self.ctx_columns: DefaultDict[tuple[int, int], CSSList] = defaultdict(list)
        self.cell_context: DefaultDict[tuple[int, int], str] = defaultdict(str)
        self._todo: list[tuple[Callable, tuple, dict]] = []
        self.tooltips: Tooltips | None = None
        precision = (
            get_option("styler.format.precision") if precision is None else precision
        )
        self._display_funcs: DefaultDict[  # maps (row, col) -> format func
            tuple[int, int], Callable[[Any], str]
        ] = defaultdict(lambda: partial(_default_formatter, precision=precision))
        self._display_funcs_index: DefaultDict[  # maps (row, level) -> format func
            tuple[int, int], Callable[[Any], str]
        ] = defaultdict(lambda: partial(_default_formatter, precision=precision))
        self._display_funcs_columns: DefaultDict[  # maps (level, col) -> format func
            tuple[int, int], Callable[[Any], str]
        ] = defaultdict(lambda: partial(_default_formatter, precision=precision))

    def _render(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        blank: str = "",
    ):
        """
        Computes and applies styles and then generates the general render dicts.

        Also extends the `ctx` and `ctx_index` attributes with those of concatenated
        stylers for use within `_translate_latex`
        """
        self._compute()
        dxs = []
        ctx_len = len(self.index)
        for i, concatenated in enumerate(self.concatenated):
            concatenated.hide_index_ = self.hide_index_
            concatenated.hidden_columns = self.hidden_columns
            foot = f"{self.css['foot']}{i}"
            concatenated.css = {
                **self.css,
                "data": f"{foot}_data",
                "row_heading": f"{foot}_row_heading",
                "row": f"{foot}_row",
                "foot": f"{foot}_foot",
            }
            dx = concatenated._render(
                sparse_index, sparse_columns, max_rows, max_cols, blank
            )
            dxs.append(dx)

            for (r, c), v in concatenated.ctx.items():
                self.ctx[(r + ctx_len, c)] = v
            for (r, c), v in concatenated.ctx_index.items():
                self.ctx_index[(r + ctx_len, c)] = v

            ctx_len += len(concatenated.index)

        d = self._translate(
            sparse_index, sparse_columns, max_rows, max_cols, blank, dxs
        )
        return d

    def _render_html(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs,
    ) -> str:
        """
        Renders the ``Styler`` including all applied styles to HTML.
        Generates a dict with necessary kwargs passed to jinja2 template.
        """
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols, "&nbsp;")
        d.update(kwargs)
        return self.template_html.render(
            **d,
            html_table_tpl=self.template_html_table,
            html_style_tpl=self.template_html_style,
        )

    def _render_latex(
        self, sparse_index: bool, sparse_columns: bool, clines: str | None, **kwargs
    ) -> str:
        """
        Render a Styler in latex format
        """
        d = self._render(sparse_index, sparse_columns, None, None)
        self._translate_latex(d, clines=clines)
        self.template_latex.globals["parse_wrap"] = _parse_latex_table_wrapping
        self.template_latex.globals["parse_table"] = _parse_latex_table_styles
        self.template_latex.globals["parse_cell"] = _parse_latex_cell_styles
        self.template_latex.globals["parse_header"] = _parse_latex_header_span
        d.update(kwargs)
        return self.template_latex.render(**d)

    def _render_string(
        self,
        sparse_index: bool,
        sparse_columns: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        **kwargs,
    ) -> str:
        """
        Render a Styler in string format
        """
        d = self._render(sparse_index, sparse_columns, max_rows, max_cols)
        d.update(kwargs)
        return self.template_string.render(**d)

    def _compute(self):
        """
        Execute the style functions built up in `self._todo`.

        Relies on the conventions that all style functions go through
        .apply or .map. The append styles to apply as tuples of

        (application method, *args, **kwargs)
        """
        self.ctx.clear()
        self.ctx_index.clear()
        self.ctx_columns.clear()
        r = self
        for func, args, kwargs in self._todo:
            r = func(self)(*args, **kwargs)
        return r

    def _translate(
        self,
        sparse_index: bool,
        sparse_cols: bool,
        max_rows: int | None = None,
        max_cols: int | None = None,
        blank: str = "&nbsp;",
        dxs: list[dict] | None = None,
    ):
        """
        Process Styler data and settings into a dict for template rendering.

        Convert data and settings from ``Styler`` attributes such as ``self.data``,
        ``self.tooltips`` including applying any methods in ``self._todo``.

        Parameters
        ----------
        sparse_index : bool
            Whether to sparsify the index or print all hierarchical index elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.index`.
        sparse_cols : bool
            Whether to sparsify the columns or print all hierarchical column elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.columns`.
        max_rows, max_cols : int, optional
            Specific max rows and cols. max_elements always take precedence in render.
        blank : str
            Entry to top-left blank cells.
        dxs : list[dict]
            The render dicts of the concatenated Stylers.

        Returns
        -------
        d : dict
            The following structure: {uuid, table_styles, caption, head, body,
            cellstyle, table_attributes}
        """
        if dxs is None:
            dxs = []
        self.css["blank_value"] = blank

        # construct render dict
        d = {
            "uuid": self.uuid,
            "table_styles": format_table_styles(self.table_styles or []),
            "caption": self.caption,
        }

        max_elements = get_option("styler.render.max_elements")
        max_rows = max_rows if max_rows else get_option("styler.render.max_rows")
        max_cols = max_cols if max_cols else get_option("styler.render.max_columns")
        max_rows, max_cols = _get_trimming_maximums(
            len(self.data.index),
            len(self.data.columns),
            max_elements,
            max_rows,
            max_cols,
        )

        self.cellstyle_map_columns: DefaultDict[
            tuple[CSSPair, ...], list[str]
        ] = defaultdict(list)
        head = self._translate_header(sparse_cols, max_cols)
        d.update({"head": head})

        # for sparsifying a MultiIndex and for use with latex clines
        idx_lengths = _get_level_lengths(
            self.index, sparse_index, max_rows, self.hidden_rows
        )
        d.update({"index_lengths": idx_lengths})

        self.cellstyle_map: DefaultDict[tuple[CSSPair, ...], list[str]] = defaultdict(
            list
        )
        self.cellstyle_map_index: DefaultDict[
            tuple[CSSPair, ...], list[str]
        ] = defaultdict(list)
        body: list = self._translate_body(idx_lengths, max_rows, max_cols)
        d.update({"body": body})

        ctx_maps = {
            "cellstyle": "cellstyle_map",
            "cellstyle_index": "cellstyle_map_index",
            "cellstyle_columns": "cellstyle_map_columns",
        }  # add the cell_ids styles map to the render dictionary in right format
        for k, attr in ctx_maps.items():
            map = [
                {"props": list(props), "selectors": selectors}
                for props, selectors in getattr(self, attr).items()
            ]
            d.update({k: map})

        for dx in dxs:  # self.concatenated is not empty
            d["body"].extend(dx["body"])  # type: ignore[union-attr]
            d["cellstyle"].extend(dx["cellstyle"])  # type: ignore[union-attr]
            d["cellstyle_index"].extend(  # type: ignore[union-attr]
                dx["cellstyle_index"]
            )

        table_attr = self.table_attributes
        if not get_option("styler.html.mathjax"):
            table_attr = table_attr or ""
            if 'class="' in table_attr:
                table_attr = table_attr.replace('class="', 'class="tex2jax_ignore ')
            else:
                table_attr += ' class="tex2jax_ignore"'
        d.update({"table_attributes": table_attr})

        if self.tooltips:
            d = self.tooltips._translate(self, d)

        return d

    def _translate_header(self, sparsify_cols: bool, max_cols: int):
        """
        Build each <tr> within table <head> as a list

        Using the structure:
             +----------------------------+---------------+---------------------------+
             |  index_blanks ...          | column_name_0 |  column_headers (level_0) |
          1) |       ..                   |       ..      |             ..            |
             |  index_blanks ...          | column_name_n |  column_headers (level_n) |
             +----------------------------+---------------+---------------------------+
          2) |  index_names (level_0 to level_n) ...      | column_blanks ...         |
             +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        sparsify_cols : bool
            Whether column_headers section will add colspan attributes (>1) to elements.
        max_cols : int
            Maximum number of columns to render. If exceeded will contain `...` filler.

        Returns
        -------
        head : list
            The associated HTML elements needed for template rendering.
        """
        # for sparsifying a MultiIndex
        col_lengths = _get_level_lengths(
            self.columns, sparsify_cols, max_cols, self.hidden_columns
        )

        clabels = self.data.columns.tolist()
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))

        head = []
        # 1) column headers
        for r, hide in enumerate(self.hide_columns_):
            if hide or not clabels:
                continue

            header_row = self._generate_col_header_row(
                (r, clabels), max_cols, col_lengths
            )
            head.append(header_row)

        # 2) index names
        if (
            self.data.index.names
            and com.any_not_none(*self.data.index.names)
            and not all(self.hide_index_)
            and not self.hide_index_names
        ):
            index_names_row = self._generate_index_names_row(
                clabels, max_cols, col_lengths
            )
            head.append(index_names_row)

        return head

    def _generate_col_header_row(
        self, iter: Sequence, max_cols: int, col_lengths: dict
    ):
        """
        Generate the row containing column headers:

         +----------------------------+---------------+---------------------------+
         |  index_blanks ...          | column_name_i |  column_headers (level_i) |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns
        col_lengths :
            c

        Returns
        -------
        list of elements
        """

        r, clabels = iter

        # number of index blanks is governed by number of hidden index levels
        index_blanks = [
            _element("th", self.css["blank"], self.css["blank_value"], True)
        ] * (self.index.nlevels - sum(self.hide_index_) - 1)

        name = self.data.columns.names[r]
        column_name = [
            _element(
                "th",
                (
                    f"{self.css['blank']} {self.css['level']}{r}"
                    if name is None
                    else f"{self.css['index_name']} {self.css['level']}{r}"
                ),
                name
                if (name is not None and not self.hide_column_names)
                else self.css["blank_value"],
                not all(self.hide_index_),
            )
        ]

        column_headers: list = []
        visible_col_count: int = 0
        for c, value in enumerate(clabels[r]):
            header_element_visible = _is_visible(c, r, col_lengths)
            if header_element_visible:
                visible_col_count += col_lengths.get((r, c), 0)
            if self._check_trim(
                visible_col_count,
                max_cols,
                column_headers,
                "th",
                f"{self.css['col_heading']} {self.css['level']}{r} "
                f"{self.css['col_trim']}",
            ):
                break

            header_element = _element(
                "th",
                (
                    f"{self.css['col_heading']} {self.css['level']}{r} "
                    f"{self.css['col']}{c}"
                ),
                value,
                header_element_visible,
                display_value=self._display_funcs_columns[(r, c)](value),
                attributes=(
                    f'colspan="{col_lengths.get((r, c), 0)}"'
                    if col_lengths.get((r, c), 0) > 1
                    else ""
                ),
            )

            if self.cell_ids:
                header_element["id"] = f"{self.css['level']}{r}_{self.css['col']}{c}"
            if (
                header_element_visible
                and (r, c) in self.ctx_columns
                and self.ctx_columns[r, c]
            ):
                header_element["id"] = f"{self.css['level']}{r}_{self.css['col']}{c}"
                self.cellstyle_map_columns[tuple(self.ctx_columns[r, c])].append(
                    f"{self.css['level']}{r}_{self.css['col']}{c}"
                )

            column_headers.append(header_element)

        return index_blanks + column_name + column_headers

    def _generate_index_names_row(
        self, iter: Sequence, max_cols: int, col_lengths: dict
    ):
        """
        Generate the row containing index names

         +----------------------------+---------------+---------------------------+
         |  index_names (level_0 to level_n) ...      | column_blanks ...         |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns

        Returns
        -------
        list of elements
        """

        clabels = iter

        index_names = [
            _element(
                "th",
                f"{self.css['index_name']} {self.css['level']}{c}",
                self.css["blank_value"] if name is None else name,
                not self.hide_index_[c],
            )
            for c, name in enumerate(self.data.index.names)
        ]

        column_blanks: list = []
        visible_col_count: int = 0
        if clabels:
            last_level = self.columns.nlevels - 1  # use last level since never sparsed
            for c, value in enumerate(clabels[last_level]):
                header_element_visible = _is_visible(c, last_level, col_lengths)
                if header_element_visible:
                    visible_col_count += 1
                if self._check_trim(
                    visible_col_count,
                    max_cols,
                    column_blanks,
                    "th",
                    f"{self.css['blank']} {self.css['col']}{c} {self.css['col_trim']}",
                    self.css["blank_value"],
                ):
                    break

                column_blanks.append(
                    _element(
                        "th",
                        f"{self.css['blank']} {self.css['col']}{c}",
                        self.css["blank_value"],
                        c not in self.hidden_columns,
                    )
                )

        return index_names + column_blanks

    def _translate_body(self, idx_lengths: dict, max_rows: int, max_cols: int):
        """
        Build each <tr> within table <body> as a list

        Use the following structure:
          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Also add elements to the cellstyle_map for more efficient grouped elements in
        <style></style> block

        Parameters
        ----------
        sparsify_index : bool
            Whether index_headers section will add rowspan attributes (>1) to elements.

        Returns
        -------
        body : list
            The associated HTML elements needed for template rendering.
        """
        rlabels = self.data.index.tolist()
        if not isinstance(self.data.index, MultiIndex):
            rlabels = [[x] for x in rlabels]

        body: list = []
        visible_row_count: int = 0
        for r, row_tup in [
            z for z in enumerate(self.data.itertuples()) if z[0] not in self.hidden_rows
        ]:
            visible_row_count += 1
            if self._check_trim(
                visible_row_count,
                max_rows,
                body,
                "row",
            ):
                break

            body_row = self._generate_body_row(
                (r, row_tup, rlabels), max_cols, idx_lengths
            )
            body.append(body_row)
        return body

    def _check_trim(
        self,
        count: int,
        max: int,
        obj: list,
        element: str,
        css: str | None = None,
        value: str = "...",
    ) -> bool:
        """
        Indicates whether to break render loops and append a trimming indicator

        Parameters
        ----------
        count : int
            The loop count of previous visible items.
        max : int
            The allowable rendered items in the loop.
        obj : list
            The current render collection of the rendered items.
        element : str
            The type of element to append in the case a trimming indicator is needed.
        css : str, optional
            The css to add to the trimming indicator element.
        value : str, optional
            The value of the elements display if necessary.

        Returns
        -------
        result : bool
            Whether a trimming element was required and appended.
        """
        if count > max:
            if element == "row":
                obj.append(self._generate_trimmed_row(max))
            else:
                obj.append(_element(element, css, value, True, attributes=""))
            return True
        return False

    def _generate_trimmed_row(self, max_cols: int) -> list:
        """
        When a render has too many rows we generate a trimming row containing "..."

        Parameters
        ----------
        max_cols : int
            Number of permissible columns

        Returns
        -------
        list of elements
        """
        index_headers = [
            _element(
                "th",
                (
                    f"{self.css['row_heading']} {self.css['level']}{c} "
                    f"{self.css['row_trim']}"
                ),
                "...",
                not self.hide_index_[c],
                attributes="",
            )
            for c in range(self.data.index.nlevels)
        ]

        data: list = []
        visible_col_count: int = 0
        for c, _ in enumerate(self.columns):
            data_element_visible = c not in self.hidden_columns
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(
                visible_col_count,
                max_cols,
                data,
                "td",
                f"{self.css['data']} {self.css['row_trim']} {self.css['col_trim']}",
            ):
                break

            data.append(
                _element(
                    "td",
                    f"{self.css['data']} {self.css['col']}{c} {self.css['row_trim']}",
                    "...",
                    data_element_visible,
                    attributes="",
                )
            )

        return index_headers + data

    def _generate_body_row(
        self,
        iter: tuple,
        max_cols: int,
        idx_lengths: dict,
    ):
        """
        Generate a regular row for the body section of appropriate format.

          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Iterable from outer scope: row number, row data tuple, row index labels.
        max_cols : int
            Number of permissible columns.
        idx_lengths : dict
            A map of the sparsification structure of the index

        Returns
        -------
            list of elements
        """
        r, row_tup, rlabels = iter

        index_headers = []
        for c, value in enumerate(rlabels[r]):
            header_element_visible = (
                _is_visible(r, c, idx_lengths) and not self.hide_index_[c]
            )
            header_element = _element(
                "th",
                (
                    f"{self.css['row_heading']} {self.css['level']}{c} "
                    f"{self.css['row']}{r}"
                ),
                value,
                header_element_visible,
                display_value=self._display_funcs_index[(r, c)](value),
                attributes=(
                    f'rowspan="{idx_lengths.get((c, r), 0)}"'
                    if idx_lengths.get((c, r), 0) > 1
                    else ""
                ),
            )

            if self.cell_ids:
                header_element[
                    "id"
                ] = f"{self.css['level']}{c}_{self.css['row']}{r}"  # id is given
            if (
                header_element_visible
                and (r, c) in self.ctx_index
                and self.ctx_index[r, c]
            ):
                # always add id if a style is specified
                header_element["id"] = f"{self.css['level']}{c}_{self.css['row']}{r}"
                self.cellstyle_map_index[tuple(self.ctx_index[r, c])].append(
                    f"{self.css['level']}{c}_{self.css['row']}{r}"
                )

            index_headers.append(header_element)

        data: list = []
        visible_col_count: int = 0
        for c, value in enumerate(row_tup[1:]):
            data_element_visible = (
                c not in self.hidden_columns and r not in self.hidden_rows
            )
            if data_element_visible:
                visible_col_count += 1
            if self._check_trim(
                visible_col_count,
                max_cols,
                data,
                "td",
                f"{self.css['data']} {self.css['row']}{r} {self.css['col_trim']}",
            ):
                break

            # add custom classes from cell context
            cls = ""
            if (r, c) in self.cell_context:
                cls = " " + self.cell_context[r, c]

            data_element = _element(
                "td",
                (
                    f"{self.css['data']} {self.css['row']}{r} "
                    f"{self.css['col']}{c}{cls}"
                ),
                value,
                data_element_visible,
                attributes="",
                display_value=self._display_funcs[(r, c)](value),
            )

            if self.cell_ids:
                data_element["id"] = f"{self.css['row']}{r}_{self.css['col']}{c}"
            if data_element_visible and (r, c) in self.ctx and self.ctx[r, c]:
                # always add id if needed due to specified style
                data_element["id"] = f"{self.css['row']}{r}_{self.css['col']}{c}"
                self.cellstyle_map[tuple(self.ctx[r, c])].append(
                    f"{self.css['row']}{r}_{self.css['col']}{c}"
                )

            data.append(data_element)

        return index_headers + data

    def _translate_latex(self, d: dict, clines: str | None) -> None:
        r"""
        Post-process the default render dict for the LaTeX template format.

        Processing items included are:
          - Remove hidden columns from the non-headers part of the body.
          - Place cellstyles directly in td cells rather than use cellstyle_map.
          - Remove hidden indexes or reinsert missing th elements if part of multiindex
            or multirow sparsification (so that \multirow and \multicol work correctly).
        """
        index_levels = self.index.nlevels
        visible_index_level_n = index_levels - sum(self.hide_index_)
        d["head"] = [
            [
                {**col, "cellstyle": self.ctx_columns[r, c - visible_index_level_n]}
                for c, col in enumerate(row)
                if col["is_visible"]
            ]
            for r, row in enumerate(d["head"])
        ]

        def _concatenated_visible_rows(obj, n, row_indices):
            """
            Extract all visible row indices recursively from concatenated stylers.
            """
            row_indices.extend(
                [r + n for r in range(len(obj.index)) if r not in obj.hidden_rows]
            )
            n += len(obj.index)
            for concatenated in obj.concatenated:
                n = _concatenated_visible_rows(concatenated, n, row_indices)
            return n

        def concatenated_visible_rows(obj):
            row_indices: list[int] = []
            _concatenated_visible_rows(obj, 0, row_indices)
            # TODO try to consolidate the concat visible rows
            # methods to a single function / recursion for simplicity
            return row_indices

        body = []
        for r, row in zip(concatenated_visible_rows(self), d["body"]):
            # note: cannot enumerate d["body"] because rows were dropped if hidden
            # during _translate_body so must zip to acquire the true r-index associated
            # with the ctx obj which contains the cell styles.
            if all(self.hide_index_):
                row_body_headers = []
            else:
                row_body_headers = [
                    {
                        **col,
                        "display_value": col["display_value"]
                        if col["is_visible"]
                        else "",
                        "cellstyle": self.ctx_index[r, c],
                    }
                    for c, col in enumerate(row[:index_levels])
                    if (col["type"] == "th" and not self.hide_index_[c])
                ]

            row_body_cells = [
                {**col, "cellstyle": self.ctx[r, c]}
                for c, col in enumerate(row[index_levels:])
                if (col["is_visible"] and col["type"] == "td")
            ]

            body.append(row_body_headers + row_body_cells)
        d["body"] = body

        # clines are determined from info on index_lengths and hidden_rows and input
        # to a dict defining which row clines should be added in the template.
        if clines not in [
            None,
            "all;data",
            "all;index",
            "skip-last;data",
            "skip-last;index",
        ]:
            raise ValueError(
                f"`clines` value of {clines} is invalid. Should either be None or one "
                f"of 'all;data', 'all;index', 'skip-last;data', 'skip-last;index'."
            )
        if clines is not None:
            data_len = len(row_body_cells) if "data" in clines and d["body"] else 0

            d["clines"] = defaultdict(list)
            visible_row_indexes: list[int] = [
                r for r in range(len(self.data.index)) if r not in self.hidden_rows
            ]
            visible_index_levels: list[int] = [
                i for i in range(index_levels) if not self.hide_index_[i]
            ]
            for rn, r in enumerate(visible_row_indexes):
                for lvln, lvl in enumerate(visible_index_levels):
                    if lvl == index_levels - 1 and "skip-last" in clines:
                        continue
                    idx_len = d["index_lengths"].get((lvl, r), None)
                    if idx_len is not None:  # i.e. not a sparsified entry
                        d["clines"][rn + idx_len].append(
                            f"\\cline{{{lvln+1}-{len(visible_index_levels)+data_len}}}"
                        )

    def format(
        self,
        formatter: ExtFormatter | None = None,
        subset: Subset | None = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ) -> StylerRenderer:
        r"""
        Format the text display value of cells.

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.
        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.

            .. versionadded:: 1.3.0

        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers.

            .. versionadded:: 1.3.0

        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers.

            .. versionadded:: 1.3.0

        escape : str, optional
            Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\`` in the cell display string with
            LaTeX-safe sequences.
            Use 'latex-math' to replace the characters the same way as in 'latex' mode,
            except for math substrings, which either are surrounded
            by two characters ``$`` or start with the character ``\(`` and
            end with ``\)``. Escaping is done before ``formatter``.

            .. versionadded:: 1.3.0

        hyperlinks : {"html", "latex"}, optional
            Convert string patterns containing https://, http://, ftp:// or www. to
            HTML <a> tags as clickable URL hyperlinks if "html", or LaTeX \href
            commands if "latex".

            .. versionadded:: 1.4.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.format_index: Format the text display value of index labels.

        Notes
        -----
        This method assigns a formatting function, ``formatter``, to each cell in the
        DataFrame. If ``formatter`` is ``None``, then the default formatter is used.
        If a callable then that function should take a data value as input and return
        a displayable representation, such as a string. If ``formatter`` is
        given as a string this is assumed to be a valid Python format specification
        and is wrapped to a callable as ``string.format(x)``. If a ``dict`` is given,
        keys should correspond to column names, and values should be string or
        callable, as above.

        The default formatter currently expresses floats and complex numbers with the
        pandas display precision unless using the ``precision`` argument here. The
        default formatter does not adjust the representation of missing values unless
        the ``na_rep`` argument is used.

        The ``subset`` argument defines which region to apply the formatting function
        to. If the ``formatter`` argument is given in dict form but does not include
        all columns within the subset then these columns will have the default formatter
        applied. Any columns in the formatter dict excluded from the subset will
        be ignored.

        When using a ``formatter`` string the dtypes must be compatible, otherwise a
        `ValueError` will be raised.

        When instantiating a Styler, default formatting can be applied be setting the
        ``pandas.options``:

          - ``styler.format.formatter``: default None.
          - ``styler.format.na_rep``: default None.
          - ``styler.format.precision``: default 6.
          - ``styler.format.decimal``: default ".".
          - ``styler.format.thousands``: default None.
          - ``styler.format.escape``: default None.

        .. warning::
           `Styler.format` is ignored when using the output format `Styler.to_excel`,
           since Excel and Python have inherrently different formatting structures.
           However, it is possible to use the `number-format` pseudo CSS attribute
           to force Excel permissible formatting. See examples.

        Examples
        --------
        Using ``na_rep`` and ``precision`` with the default ``formatter``

        >>> df = pd.DataFrame([[np.nan, 1.0, 'A'], [2.0, np.nan, 3.0]])
        >>> df.style.format(na_rep='MISS', precision=3)  # doctest: +SKIP
                0       1       2
        0    MISS   1.000       A
        1   2.000    MISS   3.000

        Using a ``formatter`` specification on consistent column dtypes

        >>> df.style.format('{:.2f}', na_rep='MISS', subset=[0,1])  # doctest: +SKIP
                0      1          2
        0    MISS   1.00          A
        1    2.00   MISS   3.000000

        Using the default ``formatter`` for unspecified columns

        >>> df.style.format({0: '{:.2f}', 1: '£ {:.1f}'}, na_rep='MISS', precision=1)
        ...  # doctest: +SKIP
                 0      1     2
        0    MISS   £ 1.0     A
        1    2.00    MISS   3.0

        Multiple ``na_rep`` or ``precision`` specifications under the default
        ``formatter``.

        >>> (df.style.format(na_rep='MISS', precision=1, subset=[0])
        ...     .format(na_rep='PASS', precision=2, subset=[1, 2]))  # doctest: +SKIP
                0      1      2
        0    MISS   1.00      A
        1     2.0   PASS   3.00

        Using a callable ``formatter`` function.

        >>> func = lambda s: 'STRING' if isinstance(s, str) else 'FLOAT'
        >>> df.style.format({0: '{:.1f}', 2: func}, precision=4, na_rep='MISS')
        ...  # doctest: +SKIP
                0        1        2
        0    MISS   1.0000   STRING
        1     2.0     MISS    FLOAT

        Using a ``formatter`` with HTML ``escape`` and ``na_rep``.

        >>> df = pd.DataFrame([['<div></div>', '"A&B"', None]])
        >>> s = df.style.format(
        ...     '<a href="a.com/{0}">{0}</a>', escape="html", na_rep="NA"
        ...     )
        >>> s.to_html()  # doctest: +SKIP
        ...
        <td .. ><a href="a.com/&lt;div&gt;&lt;/div&gt;">&lt;div&gt;&lt;/div&gt;</a></td>
        <td .. ><a href="a.com/&#34;A&amp;B&#34;">&#34;A&amp;B&#34;</a></td>
        <td .. >NA</td>
        ...

        Using a ``formatter`` with ``escape`` in 'latex' mode.

        >>> df = pd.DataFrame([["123"], ["~ ^"], ["$%#"]])
        >>> df.style.format("\\textbf{{{}}}", escape="latex").to_latex()
        ...  # doctest: +SKIP
        \begin{tabular}{ll}
         & 0 \\
        0 & \textbf{123} \\
        1 & \textbf{\textasciitilde \space \textasciicircum } \\
        2 & \textbf{\$\%\#} \\
        \end{tabular}

        Applying ``escape`` in 'latex-math' mode. In the example below
        we enter math mode using the character ``$``.

        >>> df = pd.DataFrame([[r"$\sum_{i=1}^{10} a_i$ a~b $\alpha \
        ...     = \frac{\beta}{\zeta^2}$"], ["%#^ $ \$x^2 $"]])
        >>> df.style.format(escape="latex-math").to_latex()
        ...  # doctest: +SKIP
        \begin{tabular}{ll}
         & 0 \\
        0 & $\sum_{i=1}^{10} a_i$ a\textasciitilde b $\alpha = \frac{\beta}{\zeta^2}$ \\
        1 & \%\#\textasciicircum \space $ \$x^2 $ \\
        \end{tabular}

        We can use the character ``\(`` to enter math mode and the character ``\)``
        to close math mode.

        >>> df = pd.DataFrame([[r"\(\sum_{i=1}^{10} a_i\) a~b \(\alpha \
        ...     = \frac{\beta}{\zeta^2}\)"], ["%#^ \( \$x^2 \)"]])
        >>> df.style.format(escape="latex-math").to_latex()
        ...  # doctest: +SKIP
        \begin{tabular}{ll}
         & 0 \\
        0 & \(\sum_{i=1}^{10} a_i\) a\textasciitilde b \(\alpha
        = \frac{\beta}{\zeta^2}\) \\
        1 & \%\#\textasciicircum \space \( \$x^2 \) \\
        \end{tabular}

        If we have in one DataFrame cell a combination of both shorthands
        for math formulas, the shorthand with the sign ``$`` will be applied.

        >>> df = pd.DataFrame([[r"\( x^2 \)  $x^2$"], \
        ...     [r"$\frac{\beta}{\zeta}$ \(\frac{\beta}{\zeta}\)"]])
        >>> df.style.format(escape="latex-math").to_latex()
        ...  # doctest: +SKIP
        \begin{tabular}{ll}
         & 0 \\
        0 & \textbackslash ( x\textasciicircum 2 \textbackslash )  $x^2$ \\
        1 & $\frac{\beta}{\zeta}$ \textbackslash (\textbackslash
        frac\{\textbackslash beta\}\{\textbackslash zeta\}\textbackslash ) \\
        \end{tabular}

        Pandas defines a `number-format` pseudo CSS attribute instead of the `.format`
        method to create `to_excel` permissible formatting. Note that semi-colons are
        CSS protected characters but used as separators in Excel's format string.
        Replace semi-colons with the section separator character (ASCII-245) when
        defining the formatting here.

        >>> df = pd.DataFrame({"A": [1, 0, -1]})
        >>> pseudo_css = "number-format: 0§[Red](0)§-§@;"
        >>> filename = "formatted_file.xlsx"
        >>> df.style.map(lambda v: pseudo_css).to_excel(filename) # doctest: +SKIP

        .. figure:: ../../_static/style/format_excel_css.png
        """
        if all(
            (
                formatter is None,
                subset is None,
                precision is None,
                decimal == ".",
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None,
            )
        ):
            self._display_funcs.clear()
            return self  # clear the formatter / revert to default and avoid looping

        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]

        if not isinstance(formatter, dict):
            formatter = {col: formatter for col in data.columns}

        cis = self.columns.get_indexer_for(data.columns)
        ris = self.index.get_indexer_for(data.index)
        for ci in cis:
            format_func = _maybe_wrap_formatter(
                formatter.get(self.columns[ci]),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks,
            )
            for ri in ris:
                self._display_funcs[(ri, ci)] = format_func

        return self

    def format_index(
        self,
        formatter: ExtFormatter | None = None,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
        hyperlinks: str | None = None,
    ) -> StylerRenderer:
        r"""
        Format the text display value of index labels or column headers.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        axis : {0, "index", 1, "columns"}
            Whether to apply the formatter to the index or column headers.
        level : int, str, list
            The level(s) over which to apply the generic formatter.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.
        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.
        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers.
        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers.
        escape : str, optional
            Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\`` in the cell display string with
            LaTeX-safe sequences.
            Escaping is done before ``formatter``.
        hyperlinks : {"html", "latex"}, optional
            Convert string patterns containing https://, http://, ftp:// or www. to
            HTML <a> tags as clickable URL hyperlinks if "html", or LaTeX \href
            commands if "latex".

        Returns
        -------
        Styler

        See Also
        --------
        Styler.format: Format the text display value of data cells.

        Notes
        -----
        This method assigns a formatting function, ``formatter``, to each level label
        in the DataFrame's index or column headers. If ``formatter`` is ``None``,
        then the default formatter is used.
        If a callable then that function should take a label value as input and return
        a displayable representation, such as a string. If ``formatter`` is
        given as a string this is assumed to be a valid Python format specification
        and is wrapped to a callable as ``string.format(x)``. If a ``dict`` is given,
        keys should correspond to MultiIndex level numbers or names, and values should
        be string or callable, as above.

        The default formatter currently expresses floats and complex numbers with the
        pandas display precision unless using the ``precision`` argument here. The
        default formatter does not adjust the representation of missing values unless
        the ``na_rep`` argument is used.

        The ``level`` argument defines which levels of a MultiIndex to apply the
        method to. If the ``formatter`` argument is given in dict form but does
        not include all levels within the level argument then these unspecified levels
        will have the default formatter applied. Any levels in the formatter dict
        specifically excluded from the level argument will be ignored.

        When using a ``formatter`` string the dtypes must be compatible, otherwise a
        `ValueError` will be raised.

        .. warning::
           `Styler.format_index` is ignored when using the output format
           `Styler.to_excel`, since Excel and Python have inherrently different
           formatting structures.
           However, it is possible to use the `number-format` pseudo CSS attribute
           to force Excel permissible formatting. See documentation for `Styler.format`.

        Examples
        --------
        Using ``na_rep`` and ``precision`` with the default ``formatter``

        >>> df = pd.DataFrame([[1, 2, 3]], columns=[2.0, np.nan, 4.0])
        >>> df.style.format_index(axis=1, na_rep='MISS', precision=3)  # doctest: +SKIP
            2.000    MISS   4.000
        0       1       2       3

        Using a ``formatter`` specification on consistent dtypes in a level

        >>> df.style.format_index('{:.2f}', axis=1, na_rep='MISS')  # doctest: +SKIP
             2.00   MISS    4.00
        0       1      2       3

        Using the default ``formatter`` for unspecified levels

        >>> df = pd.DataFrame([[1, 2, 3]],
        ...     columns=pd.MultiIndex.from_arrays([["a", "a", "b"],[2, np.nan, 4]]))
        >>> df.style.format_index({0: lambda v: v.upper()}, axis=1, precision=1)
        ...  # doctest: +SKIP
                       A       B
              2.0    nan     4.0
        0       1      2       3

        Using a callable ``formatter`` function.

        >>> func = lambda s: 'STRING' if isinstance(s, str) else 'FLOAT'
        >>> df.style.format_index(func, axis=1, na_rep='MISS')
        ...  # doctest: +SKIP
                  STRING  STRING
            FLOAT   MISS   FLOAT
        0       1      2       3

        Using a ``formatter`` with HTML ``escape`` and ``na_rep``.

        >>> df = pd.DataFrame([[1, 2, 3]], columns=['"A"', 'A&B', None])
        >>> s = df.style.format_index('$ {0}', axis=1, escape="html", na_rep="NA")
        ...  # doctest: +SKIP
        <th .. >$ &#34;A&#34;</th>
        <th .. >$ A&amp;B</th>
        <th .. >NA</td>
        ...

        Using a ``formatter`` with LaTeX ``escape``.

        >>> df = pd.DataFrame([[1, 2, 3]], columns=["123", "~", "$%#"])
        >>> df.style.format_index("\\textbf{{{}}}", escape="latex", axis=1).to_latex()
        ...  # doctest: +SKIP
        \begin{tabular}{lrrr}
        {} & {\textbf{123}} & {\textbf{\textasciitilde }} & {\textbf{\$\%\#}} \\
        0 & 1 & 2 & 3 \\
        \end{tabular}
        """
        axis = self.data._get_axis_number(axis)
        if axis == 0:
            display_funcs_, obj = self._display_funcs_index, self.index
        else:
            display_funcs_, obj = self._display_funcs_columns, self.columns
        levels_ = refactor_levels(level, obj)

        if all(
            (
                formatter is None,
                level is None,
                precision is None,
                decimal == ".",
                thousands is None,
                na_rep is None,
                escape is None,
                hyperlinks is None,
            )
        ):
            display_funcs_.clear()
            return self  # clear the formatter / revert to default and avoid looping

        if not isinstance(formatter, dict):
            formatter = {level: formatter for level in levels_}
        else:
            formatter = {
                obj._get_level_number(level): formatter_
                for level, formatter_ in formatter.items()
            }

        for lvl in levels_:
            format_func = _maybe_wrap_formatter(
                formatter.get(lvl),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
                hyperlinks=hyperlinks,
            )

            for idx in [(i, lvl) if axis == 0 else (lvl, i) for i in range(len(obj))]:
                display_funcs_[idx] = format_func

        return self

    def relabel_index(
        self,
        labels: Sequence | Index,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
    ) -> StylerRenderer:
        r"""
        Relabel the index, or column header, keys to display a set of specified values.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        labels : list-like or Index
            New labels to display. Must have same length as the underlying values not
            hidden.
        axis : {"index", 0, "columns", 1}
            Apply to the index or columns.
        level : int, str, list, optional
            The level(s) over which to apply the new labels. If `None` will apply
            to all levels of an Index or MultiIndex which are not hidden.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.format_index: Format the text display value of index or column headers.
        Styler.hide: Hide the index, column headers, or specified data from display.

        Notes
        -----
        As part of Styler, this method allows the display of an index to be
        completely user-specified without affecting the underlying DataFrame data,
        index, or column headers. This means that the flexibility of indexing is
        maintained whilst the final display is customisable.

        Since Styler is designed to be progressively constructed with method chaining,
        this method is adapted to react to the **currently specified hidden elements**.
        This is useful because it means one does not have to specify all the new
        labels if the majority of an index, or column headers, have already been hidden.
        The following produce equivalent display (note the length of ``labels`` in
        each case).

        .. code-block:: python

            # relabel first, then hide
            df = pd.DataFrame({"col": ["a", "b", "c"]})
            df.style.relabel_index(["A", "B", "C"]).hide([0,1])
            # hide first, then relabel
            df = pd.DataFrame({"col": ["a", "b", "c"]})
            df.style.hide([0,1]).relabel_index(["C"])

        This method should be used, rather than :meth:`Styler.format_index`, in one of
        the following cases (see examples):

          - A specified set of labels are required which are not a function of the
            underlying index keys.
          - The function of the underlying index keys requires a counter variable,
            such as those available upon enumeration.

        Examples
        --------
        Basic use

        >>> df = pd.DataFrame({"col": ["a", "b", "c"]})
        >>> df.style.relabel_index(["A", "B", "C"])  # doctest: +SKIP
             col
        A      a
        B      b
        C      c

        Chaining with pre-hidden elements

        >>> df.style.hide([0,1]).relabel_index(["C"])  # doctest: +SKIP
             col
        C      c

        Using a MultiIndex

        >>> midx = pd.MultiIndex.from_product([[0, 1], [0, 1], [0, 1]])
        >>> df = pd.DataFrame({"col": list(range(8))}, index=midx)
        >>> styler = df.style  # doctest: +SKIP
                  col
        0  0  0     0
              1     1
           1  0     2
              1     3
        1  0  0     4
              1     5
           1  0     6
              1     7
        >>> styler.hide((midx.get_level_values(0)==0)|(midx.get_level_values(1)==0))
        ...  # doctest: +SKIP
        >>> styler.hide(level=[0,1])  # doctest: +SKIP
        >>> styler.relabel_index(["binary6", "binary7"])  # doctest: +SKIP
                  col
        binary6     6
        binary7     7

        We can also achieve the above by indexing first and then re-labeling

        >>> styler = df.loc[[(1,1,0), (1,1,1)]].style
        >>> styler.hide(level=[0,1]).relabel_index(["binary6", "binary7"])
        ...  # doctest: +SKIP
                  col
        binary6     6
        binary7     7

        Defining a formatting function which uses an enumeration counter. Also note
        that the value of the index key is passed in the case of string labels so it
        can also be inserted into the label, using curly brackets (or double curly
        brackets if the string if pre-formatted),

        >>> df = pd.DataFrame({"samples": np.random.rand(10)})
        >>> styler = df.loc[np.random.randint(0,10,3)].style
        >>> styler.relabel_index([f"sample{i+1} ({{}})" for i in range(3)])
        ...  # doctest: +SKIP
                         samples
        sample1 (5)     0.315811
        sample2 (0)     0.495941
        sample3 (2)     0.067946
        """
        axis = self.data._get_axis_number(axis)
        if axis == 0:
            display_funcs_, obj = self._display_funcs_index, self.index
            hidden_labels, hidden_lvls = self.hidden_rows, self.hide_index_
        else:
            display_funcs_, obj = self._display_funcs_columns, self.columns
            hidden_labels, hidden_lvls = self.hidden_columns, self.hide_columns_
        visible_len = len(obj) - len(set(hidden_labels))
        if len(labels) != visible_len:
            raise ValueError(
                "``labels`` must be of length equal to the number of "
                f"visible labels along ``axis`` ({visible_len})."
            )

        if level is None:
            level = [i for i in range(obj.nlevels) if not hidden_lvls[i]]
        levels_ = refactor_levels(level, obj)

        def alias_(x, value):
            if isinstance(value, str):
                return value.format(x)
            return value

        for ai, i in enumerate([i for i in range(len(obj)) if i not in hidden_labels]):
            if len(levels_) == 1:
                idx = (i, levels_[0]) if axis == 0 else (levels_[0], i)
                display_funcs_[idx] = partial(alias_, value=labels[ai])
            else:
                for aj, lvl in enumerate(levels_):
                    idx = (i, lvl) if axis == 0 else (lvl, i)
                    display_funcs_[idx] = partial(alias_, value=labels[ai][aj])

        return self


def _element(
    html_element: str,
    html_class: str | None,
    value: Any,
    is_visible: bool,
    **kwargs,
) -> dict:
    """
    Template to return container with information for a <td></td> or <th></th> element.
    """
    if "display_value" not in kwargs:
        kwargs["display_value"] = value
    return {
        "type": html_element,
        "value": value,
        "class": html_class,
        "is_visible": is_visible,
        **kwargs,
    }


def _get_trimming_maximums(
    rn,
    cn,
    max_elements,
    max_rows=None,
    max_cols=None,
    scaling_factor: float = 0.8,
) -> tuple[int, int]:
    """
    Recursively reduce the number of rows and columns to satisfy max elements.

    Parameters
    ----------
    rn, cn : int
        The number of input rows / columns
    max_elements : int
        The number of allowable elements
    max_rows, max_cols : int, optional
        Directly specify an initial maximum rows or columns before compression.
    scaling_factor : float
        Factor at which to reduce the number of rows / columns to fit.

    Returns
    -------
    rn, cn : tuple
        New rn and cn values that satisfy the max_elements constraint
    """

    def scale_down(rn, cn):
        if cn >= rn:
            return rn, int(cn * scaling_factor)
        else:
            return int(rn * scaling_factor), cn

    if max_rows:
        rn = max_rows if rn > max_rows else rn
    if max_cols:
        cn = max_cols if cn > max_cols else cn

    while rn * cn > max_elements:
        rn, cn = scale_down(rn, cn)

    return rn, cn


def _get_level_lengths(
    index: Index,
    sparsify: bool,
    max_index: int,
    hidden_elements: Sequence[int] | None = None,
):
    """
    Given an index, find the level length for each element.

    Parameters
    ----------
    index : Index
        Index or columns to determine lengths of each element
    sparsify : bool
        Whether to hide or show each distinct element in a MultiIndex
    max_index : int
        The maximum number of elements to analyse along the index due to trimming
    hidden_elements : sequence of int
        Index positions of elements hidden from display in the index affecting
        length

    Returns
    -------
    Dict :
        Result is a dictionary of (level, initial_position): span
    """
    if isinstance(index, MultiIndex):
        levels = index._format_multi(sparsify=lib.no_default, include_names=False)
    else:
        levels = index._format_flat(include_name=False)

    if hidden_elements is None:
        hidden_elements = []

    lengths = {}
    if not isinstance(index, MultiIndex):
        for i, value in enumerate(levels):
            if i not in hidden_elements:
                lengths[(0, i)] = 1
        return lengths

    for i, lvl in enumerate(levels):
        visible_row_count = 0  # used to break loop due to display trimming
        for j, row in enumerate(lvl):
            if visible_row_count > max_index:
                break
            if not sparsify:
                # then lengths will always equal 1 since no aggregation.
                if j not in hidden_elements:
                    lengths[(i, j)] = 1
                    visible_row_count += 1
            elif (row is not lib.no_default) and (j not in hidden_elements):
                # this element has not been sparsified so must be the start of section
                last_label = j
                lengths[(i, last_label)] = 1
                visible_row_count += 1
            elif row is not lib.no_default:
                # even if the above is hidden, keep track of it in case length > 1 and
                # later elements are visible
                last_label = j
                lengths[(i, last_label)] = 0
            elif j not in hidden_elements:
                # then element must be part of sparsified section and is visible
                visible_row_count += 1
                if visible_row_count > max_index:
                    break  # do not add a length since the render trim limit reached
                if lengths[(i, last_label)] == 0:
                    # if previous iteration was first-of-section but hidden then offset
                    last_label = j
                    lengths[(i, last_label)] = 1
                else:
                    # else add to previous iteration
                    lengths[(i, last_label)] += 1

    non_zero_lengths = {
        element: length for element, length in lengths.items() if length >= 1
    }

    return non_zero_lengths


def _is_visible(idx_row, idx_col, lengths) -> bool:
    """
    Index -> {(idx_row, idx_col): bool}).
    """
    return (idx_col, idx_row) in lengths


def format_table_styles(styles: CSSStyles) -> CSSStyles:
    """
    looks for multiple CSS selectors and separates them:
    [{'selector': 'td, th', 'props': 'a:v;'}]
        ---> [{'selector': 'td', 'props': 'a:v;'},
              {'selector': 'th', 'props': 'a:v;'}]
    """
    return [
        {"selector": selector, "props": css_dict["props"]}
        for css_dict in styles
        for selector in css_dict["selector"].split(",")
    ]


def _default_formatter(x: Any, precision: int, thousands: bool = False) -> Any:
    """
    Format the display of a value

    Parameters
    ----------
    x : Any
        Input variable to be formatted
    precision : Int
        Floating point precision used if ``x`` is float or complex.
    thousands : bool, default False
        Whether to group digits with thousands separated with ",".

    Returns
    -------
    value : Any
        Matches input type, or string if input is float or complex or int with sep.
    """
    if is_float(x) or is_complex(x):
        return f"{x:,.{precision}f}" if thousands else f"{x:.{precision}f}"
    elif is_integer(x):
        return f"{x:,}" if thousands else str(x)
    return x


def _wrap_decimal_thousands(
    formatter: Callable, decimal: str, thousands: str | None
) -> Callable:
    """
    Takes a string formatting function and wraps logic to deal with thousands and
    decimal parameters, in the case that they are non-standard and that the input
    is a (float, complex, int).
    """

    def wrapper(x):
        if is_float(x) or is_integer(x) or is_complex(x):
            if decimal != "." and thousands is not None and thousands != ",":
                return (
                    formatter(x)
                    .replace(",", "§_§-")  # rare string to avoid "," <-> "." clash.
                    .replace(".", decimal)
                    .replace("§_§-", thousands)
                )
            elif decimal != "." and (thousands is None or thousands == ","):
                return formatter(x).replace(".", decimal)
            elif decimal == "." and thousands is not None and thousands != ",":
                return formatter(x).replace(",", thousands)
        return formatter(x)

    return wrapper


def _str_escape(x, escape):
    """if escaping: only use on str, else return input"""
    if isinstance(x, str):
        if escape == "html":
            return escape_html(x)
        elif escape == "latex":
            return _escape_latex(x)
        elif escape == "latex-math":
            return _escape_latex_math(x)
        else:
            raise ValueError(
                f"`escape` only permitted in {{'html', 'latex', 'latex-math'}}, \
got {escape}"
            )
    return x


def _render_href(x, format):
    """uses regex to detect a common URL pattern and converts to href tag in format."""
    if isinstance(x, str):
        if format == "html":
            href = '<a href="{0}" target="_blank">{0}</a>'
        elif format == "latex":
            href = r"\href{{{0}}}{{{0}}}"
        else:
            raise ValueError("``hyperlinks`` format can only be 'html' or 'latex'")
        pat = r"((http|ftp)s?:\/\/|www.)[\w/\-?=%.:@]+\.[\w/\-&?=%.,':;~!@#$*()\[\]]+"
        return re.sub(pat, lambda m: href.format(m.group(0)), x)
    return x


def _maybe_wrap_formatter(
    formatter: BaseFormatter | None = None,
    na_rep: str | None = None,
    precision: int | None = None,
    decimal: str = ".",
    thousands: str | None = None,
    escape: str | None = None,
    hyperlinks: str | None = None,
) -> Callable:
    """
    Allows formatters to be expressed as str, callable or None, where None returns
    a default formatting function. wraps with na_rep, and precision where they are
    available.
    """
    # Get initial func from input string, input callable, or from default factory
    if isinstance(formatter, str):
        func_0 = lambda x: formatter.format(x)
    elif callable(formatter):
        func_0 = formatter
    elif formatter is None:
        precision = (
            get_option("styler.format.precision") if precision is None else precision
        )
        func_0 = partial(
            _default_formatter, precision=precision, thousands=(thousands is not None)
        )
    else:
        raise TypeError(f"'formatter' expected str or callable, got {type(formatter)}")

    # Replace chars if escaping
    if escape is not None:
        func_1 = lambda x: func_0(_str_escape(x, escape=escape))
    else:
        func_1 = func_0

    # Replace decimals and thousands if non-standard inputs detected
    if decimal != "." or (thousands is not None and thousands != ","):
        func_2 = _wrap_decimal_thousands(func_1, decimal=decimal, thousands=thousands)
    else:
        func_2 = func_1

    # Render links
    if hyperlinks is not None:
        func_3 = lambda x: func_2(_render_href(x, format=hyperlinks))
    else:
        func_3 = func_2

    # Replace missing values if na_rep
    if na_rep is None:
        return func_3
    else:
        return lambda x: na_rep if (isna(x) is True) else func_3(x)


def non_reducing_slice(slice_: Subset):
    """
    Ensure that a slice doesn't reduce to a Series or Scalar.

    Any user-passed `subset` should have this called on it
    to make sure we're always working with DataFrames.
    """
    # default to column slice, like DataFrame
    # ['A', 'B'] -> IndexSlices[:, ['A', 'B']]
    kinds = (ABCSeries, np.ndarray, Index, list, str)
    if isinstance(slice_, kinds):
        slice_ = IndexSlice[:, slice_]

    def pred(part) -> bool:
        """
        Returns
        -------
        bool
            True if slice does *not* reduce,
            False if `part` is a tuple.
        """
        # true when slice does *not* reduce, False when part is a tuple,
        # i.e. MultiIndex slice
        if isinstance(part, tuple):
            # GH#39421 check for sub-slice:
            return any((isinstance(s, slice) or is_list_like(s)) for s in part)
        else:
            return isinstance(part, slice) or is_list_like(part)

    if not is_list_like(slice_):
        if not isinstance(slice_, slice):
            # a 1-d slice, like df.loc[1]
            slice_ = [[slice_]]
        else:
            # slice(a, b, c)
            slice_ = [slice_]  # to tuplize later
    else:
        # error: Item "slice" of "Union[slice, Sequence[Any]]" has no attribute
        # "__iter__" (not iterable) -> is specifically list_like in conditional
        slice_ = [p if pred(p) else [p] for p in slice_]  # type: ignore[union-attr]
    return tuple(slice_)


def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList:
    """
    Convert css-string to sequence of tuples format if needed.
    'color:red; border:1px solid black;' -> [('color', 'red'),
                                             ('border','1px solid red')]
    """
    if isinstance(style, str):
        s = style.split(";")
        try:
            return [
                (x.split(":")[0].strip(), x.split(":")[1].strip())
                for x in s
                if x.strip() != ""
            ]
        except IndexError:
            raise ValueError(
                "Styles supplied as string must follow CSS rule formats, "
                f"for example 'attr: val;'. '{style}' was given."
            )
    return style


def refactor_levels(
    level: Level | list[Level] | None,
    obj: Index,
) -> list[int]:
    """
    Returns a consistent levels arg for use in ``hide_index`` or ``hide_columns``.

    Parameters
    ----------
    level : int, str, list
        Original ``level`` arg supplied to above methods.
    obj:
        Either ``self.index`` or ``self.columns``

    Returns
    -------
    list : refactored arg with a list of levels to hide
    """
    if level is None:
        levels_: list[int] = list(range(obj.nlevels))
    elif isinstance(level, int):
        levels_ = [level]
    elif isinstance(level, str):
        levels_ = [obj._get_level_number(level)]
    elif isinstance(level, list):
        levels_ = [
            obj._get_level_number(lev) if not isinstance(lev, int) else lev
            for lev in level
        ]
    else:
        raise ValueError("`level` must be of type `int`, `str` or list of such")
    return levels_


class Tooltips:
    """
    An extension to ``Styler`` that allows for and manipulates tooltips on hover
    of ``<td>`` cells in the HTML result.

    Parameters
    ----------
    css_name: str, default "pd-t"
        Name of the CSS class that controls visualisation of tooltips.
    css_props: list-like, default; see Notes
        List of (attr, value) tuples defining properties of the CSS class.
    tooltips: DataFrame, default empty
        DataFrame of strings aligned with underlying Styler data for tooltip
        display.

    Notes
    -----
    The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

    Hidden visibility is a key prerequisite to the hover functionality, and should
    always be included in any manual properties specification.
    """

    def __init__(
        self,
        css_props: CSSProperties = [
            ("visibility", "hidden"),
            ("position", "absolute"),
            ("z-index", 1),
            ("background-color", "black"),
            ("color", "white"),
            ("transform", "translate(-20px, -20px)"),
        ],
        css_name: str = "pd-t",
        tooltips: DataFrame = DataFrame(),
    ) -> None:
        self.class_name = css_name
        self.class_properties = css_props
        self.tt_data = tooltips
        self.table_styles: CSSStyles = []

    @property
    def _class_styles(self):
        """
        Combine the ``_Tooltips`` CSS class name and CSS properties to the format
        required to extend the underlying ``Styler`` `table_styles` to allow
        tooltips to render in HTML.

        Returns
        -------
        styles : List
        """
        return [
            {
                "selector": f".{self.class_name}",
                "props": maybe_convert_css_to_tuples(self.class_properties),
            }
        ]

    def _pseudo_css(self, uuid: str, name: str, row: int, col: int, text: str):
        """
        For every table data-cell that has a valid tooltip (not None, NaN or
        empty string) must create two pseudo CSS entries for the specific
        <td> element id which are added to overall table styles:
        an on hover visibility change and a content change
        dependent upon the user's chosen display string.

        For example:
            [{"selector": "T__row1_col1:hover .pd-t",
             "props": [("visibility", "visible")]},
            {"selector": "T__row1_col1 .pd-t::after",
             "props": [("content", "Some Valid Text String")]}]

        Parameters
        ----------
        uuid: str
            The uuid of the Styler instance
        name: str
            The css-name of the class used for styling tooltips
        row : int
            The row index of the specified tooltip string data
        col : int
            The col index of the specified tooltip string data
        text : str
            The textual content of the tooltip to be displayed in HTML.

        Returns
        -------
        pseudo_css : List
        """
        selector_id = "#T_" + uuid + "_row" + str(row) + "_col" + str(col)
        return [
            {
                "selector": selector_id + f":hover .{name}",
                "props": [("visibility", "visible")],
            },
            {
                "selector": selector_id + f" .{name}::after",
                "props": [("content", f'"{text}"')],
            },
        ]

    def _translate(self, styler: StylerRenderer, d: dict):
        """
        Mutate the render dictionary to allow for tooltips:

        - Add ``<span>`` HTML element to each data cells ``display_value``. Ignores
          headers.
        - Add table level CSS styles to control pseudo classes.

        Parameters
        ----------
        styler_data : DataFrame
            Underlying ``Styler`` DataFrame used for reindexing.
        uuid : str
            The underlying ``Styler`` uuid for CSS id.
        d : dict
            The dictionary prior to final render

        Returns
        -------
        render_dict : Dict
        """
        self.tt_data = self.tt_data.reindex_like(styler.data)
        if self.tt_data.empty:
            return d

        name = self.class_name
        mask = (self.tt_data.isna()) | (self.tt_data.eq(""))  # empty string = no ttip
        self.table_styles = [
            style
            for sublist in [
                self._pseudo_css(styler.uuid, name, i, j, str(self.tt_data.iloc[i, j]))
                for i in range(len(self.tt_data.index))
                for j in range(len(self.tt_data.columns))
                if not (
                    mask.iloc[i, j]
                    or i in styler.hidden_rows
                    or j in styler.hidden_columns
                )
            ]
            for style in sublist
        ]

        if self.table_styles:
            # add span class to every cell only if at least 1 non-empty tooltip
            for row in d["body"]:
                for item in row:
                    if item["type"] == "td":
                        item["display_value"] = (
                            str(item["display_value"])
                            + f'<span class="{self.class_name}"></span>'
                        )
            d["table_styles"].extend(self._class_styles)
            d["table_styles"].extend(self.table_styles)

        return d


def _parse_latex_table_wrapping(table_styles: CSSStyles, caption: str | None) -> bool:
    """
    Indicate whether LaTeX {tabular} should be wrapped with a {table} environment.

    Parses the `table_styles` and detects any selectors which must be included outside
    of {tabular}, i.e. indicating that wrapping must occur, and therefore return True,
    or if a caption exists and requires similar.
    """
    IGNORED_WRAPPERS = ["toprule", "midrule", "bottomrule", "column_format"]
    # ignored selectors are included with {tabular} so do not need wrapping
    return (
        table_styles is not None
        and any(d["selector"] not in IGNORED_WRAPPERS for d in table_styles)
    ) or caption is not None


def _parse_latex_table_styles(table_styles: CSSStyles, selector: str) -> str | None:
    """
    Return the first 'props' 'value' from ``tables_styles`` identified by ``selector``.

    Examples
    --------
    >>> table_styles = [{'selector': 'foo', 'props': [('attr','value')]},
    ...                 {'selector': 'bar', 'props': [('attr', 'overwritten')]},
    ...                 {'selector': 'bar', 'props': [('a1', 'baz'), ('a2', 'ignore')]}]
    >>> _parse_latex_table_styles(table_styles, selector='bar')
    'baz'

    Notes
    -----
    The replacement of "§" with ":" is to avoid the CSS problem where ":" has structural
    significance and cannot be used in LaTeX labels, but is often required by them.
    """
    for style in table_styles[::-1]:  # in reverse for most recently applied style
        if style["selector"] == selector:
            return str(style["props"][0][1]).replace("§", ":")
    return None


def _parse_latex_cell_styles(
    latex_styles: CSSList, display_value: str, convert_css: bool = False
) -> str:
    r"""
    Mutate the ``display_value`` string including LaTeX commands from ``latex_styles``.

    This method builds a recursive latex chain of commands based on the
    CSSList input, nested around ``display_value``.

    If a CSS style is given as ('<command>', '<options>') this is translated to
    '\<command><options>{display_value}', and this value is treated as the
    display value for the next iteration.

    The most recent style forms the inner component, for example for styles:
    `[('c1', 'o1'), ('c2', 'o2')]` this returns: `\c1o1{\c2o2{display_value}}`

    Sometimes latex commands have to be wrapped with curly braces in different ways:
    We create some parsing flags to identify the different behaviours:

     - `--rwrap`        : `\<command><options>{<display_value>}`
     - `--wrap`         : `{\<command><options> <display_value>}`
     - `--nowrap`       : `\<command><options> <display_value>`
     - `--lwrap`        : `{\<command><options>} <display_value>`
     - `--dwrap`        : `{\<command><options>}{<display_value>}`

    For example for styles:
    `[('c1', 'o1--wrap'), ('c2', 'o2')]` this returns: `{\c1o1 \c2o2{display_value}}
    """
    if convert_css:
        latex_styles = _parse_latex_css_conversion(latex_styles)
    for command, options in latex_styles[::-1]:  # in reverse for most recent style
        formatter = {
            "--wrap": f"{{\\{command}--to_parse {display_value}}}",
            "--nowrap": f"\\{command}--to_parse {display_value}",
            "--lwrap": f"{{\\{command}--to_parse}} {display_value}",
            "--rwrap": f"\\{command}--to_parse{{{display_value}}}",
            "--dwrap": f"{{\\{command}--to_parse}}{{{display_value}}}",
        }
        display_value = f"\\{command}{options} {display_value}"
        for arg in ["--nowrap", "--wrap", "--lwrap", "--rwrap", "--dwrap"]:
            if arg in str(options):
                display_value = formatter[arg].replace(
                    "--to_parse", _parse_latex_options_strip(value=options, arg=arg)
                )
                break  # only ever one purposeful entry
    return display_value


def _parse_latex_header_span(
    cell: dict[str, Any],
    multirow_align: str,
    multicol_align: str,
    wrap: bool = False,
    convert_css: bool = False,
) -> str:
    r"""
    Refactor the cell `display_value` if a 'colspan' or 'rowspan' attribute is present.

    'rowspan' and 'colspan' do not occur simultaneouly. If they are detected then
    the `display_value` is altered to a LaTeX `multirow` or `multicol` command
    respectively, with the appropriate cell-span.

    ``wrap`` is used to enclose the `display_value` in braces which is needed for
    column headers using an siunitx package.

    Requires the package {multirow}, whereas multicol support is usually built in
    to the {tabular} environment.

    Examples
    --------
    >>> cell = {'cellstyle': '', 'display_value':'text', 'attributes': 'colspan="3"'}
    >>> _parse_latex_header_span(cell, 't', 'c')
    '\\multicolumn{3}{c}{text}'
    """
    display_val = _parse_latex_cell_styles(
        cell["cellstyle"], cell["display_value"], convert_css
    )
    if "attributes" in cell:
        attrs = cell["attributes"]
        if 'colspan="' in attrs:
            colspan = attrs[attrs.find('colspan="') + 9 :]  # len('colspan="') = 9
            colspan = int(colspan[: colspan.find('"')])
            if "naive-l" == multicol_align:
                out = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks = " & {}" if wrap else " &"
                return out + blanks * (colspan - 1)
            elif "naive-r" == multicol_align:
                out = f"{{{display_val}}}" if wrap else f"{display_val}"
                blanks = "{} & " if wrap else "& "
                return blanks * (colspan - 1) + out
            return f"\\multicolumn{{{colspan}}}{{{multicol_align}}}{{{display_val}}}"
        elif 'rowspan="' in attrs:
            if multirow_align == "naive":
                return display_val
            rowspan = attrs[attrs.find('rowspan="') + 9 :]
            rowspan = int(rowspan[: rowspan.find('"')])
            return f"\\multirow[{multirow_align}]{{{rowspan}}}{{*}}{{{display_val}}}"
    if wrap:
        return f"{{{display_val}}}"
    else:
        return display_val


def _parse_latex_options_strip(value: str | float, arg: str) -> str:
    """
    Strip a css_value which may have latex wrapping arguments, css comment identifiers,
    and whitespaces, to a valid string for latex options parsing.

    For example: 'red /* --wrap */  ' --> 'red'
    """
    return str(value).replace(arg, "").replace("/*", "").replace("*/", "").strip()


def _parse_latex_css_conversion(styles: CSSList) -> CSSList:
    """
    Convert CSS (attribute,value) pairs to equivalent LaTeX (command,options) pairs.

    Ignore conversion if tagged with `--latex` option, skipped if no conversion found.
    """

    def font_weight(value, arg):
        if value in ("bold", "bolder"):
            return "bfseries", f"{arg}"
        return None

    def font_style(value, arg):
        if value == "italic":
            return "itshape", f"{arg}"
        if value == "oblique":
            return "slshape", f"{arg}"
        return None

    def color(value, user_arg, command, comm_arg):
        """
        CSS colors have 5 formats to process:

         - 6 digit hex code: "#ff23ee"     --> [HTML]{FF23EE}
         - 3 digit hex code: "#f0e"        --> [HTML]{FF00EE}
         - rgba: rgba(128, 255, 0, 0.5)    --> [rgb]{0.502, 1.000, 0.000}
         - rgb: rgb(128, 255, 0,)          --> [rbg]{0.502, 1.000, 0.000}
         - string: red                     --> {red}

        Additionally rgb or rgba can be expressed in % which is also parsed.
        """
        arg = user_arg if user_arg != "" else comm_arg

        if value[0] == "#" and len(value) == 7:  # color is hex code
            return command, f"[HTML]{{{value[1:].upper()}}}{arg}"
        if value[0] == "#" and len(value) == 4:  # color is short hex code
            val = f"{value[1].upper()*2}{value[2].upper()*2}{value[3].upper()*2}"
            return command, f"[HTML]{{{val}}}{arg}"
        elif value[:3] == "rgb":  # color is rgb or rgba
            r = re.findall("(?<=\\()[0-9\\s%]+(?=,)", value)[0].strip()
            r = float(r[:-1]) / 100 if "%" in r else int(r) / 255
            g = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[0].strip()
            g = float(g[:-1]) / 100 if "%" in g else int(g) / 255
            if value[3] == "a":  # color is rgba
                b = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[1].strip()
            else:  # color is rgb
                b = re.findall("(?<=,)[0-9\\s%]+(?=\\))", value)[0].strip()
            b = float(b[:-1]) / 100 if "%" in b else int(b) / 255
            return command, f"[rgb]{{{r:.3f}, {g:.3f}, {b:.3f}}}{arg}"
        else:
            return command, f"{{{value}}}{arg}"  # color is likely string-named

    CONVERTED_ATTRIBUTES: dict[str, Callable] = {
        "font-weight": font_weight,
        "background-color": partial(color, command="cellcolor", comm_arg="--lwrap"),
        "color": partial(color, command="color", comm_arg=""),
        "font-style": font_style,
    }

    latex_styles: CSSList = []
    for attribute, value in styles:
        if isinstance(value, str) and "--latex" in value:
            # return the style without conversion but drop '--latex'
            latex_styles.append((attribute, value.replace("--latex", "")))
        if attribute in CONVERTED_ATTRIBUTES:
            arg = ""
            for x in ["--wrap", "--nowrap", "--lwrap", "--dwrap", "--rwrap"]:
                if x in str(value):
                    arg, value = x, _parse_latex_options_strip(value, x)
                    break
            latex_style = CONVERTED_ATTRIBUTES[attribute](value, arg)
            if latex_style is not None:
                latex_styles.extend([latex_style])
    return latex_styles


def _escape_latex(s: str) -> str:
    r"""
    Replace the characters ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``,
    ``~``, ``^``, and ``\`` in the string with LaTeX-safe sequences.

    Use this if you need to display text that might contain such characters in LaTeX.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    return (
        s.replace("\\", "ab2§=§8yz")  # rare string for final conversion: avoid \\ clash
        .replace("ab2§=§8yz ", "ab2§=§8yz\\space ")  # since \backslash gobbles spaces
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~ ", "~\\space ")  # since \textasciitilde gobbles spaces
        .replace("~", "\\textasciitilde ")
        .replace("^ ", "^\\space ")  # since \textasciicircum gobbles spaces
        .replace("^", "\\textasciicircum ")
        .replace("ab2§=§8yz", "\\textbackslash ")
    )


def _math_mode_with_dollar(s: str) -> str:
    r"""
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``$`` and end with ``$``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace(r"\$", r"rt8§=§7wz")
    pattern = re.compile(r"\$.*?\$")
    pos = 0
    ps = pattern.search(s, pos)
    res = []
    while ps:
        res.append(_escape_latex(s[pos : ps.span()[0]]))
        res.append(ps.group())
        pos = ps.span()[1]
        ps = pattern.search(s, pos)

    res.append(_escape_latex(s[pos : len(s)]))
    return "".join(res).replace(r"rt8§=§7wz", r"\$")


def _math_mode_with_parentheses(s: str) -> str:
    r"""
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``\(`` and end with ``\)``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace(r"\(", r"LEFT§=§6yzLEFT").replace(r"\)", r"RIGHTab5§=§RIGHT")
    res = []
    for item in re.split(r"LEFT§=§6yz|ab5§=§RIGHT", s):
        if item.startswith("LEFT") and item.endswith("RIGHT"):
            res.append(item.replace("LEFT", r"\(").replace("RIGHT", r"\)"))
        elif "LEFT" in item and "RIGHT" in item:
            res.append(
                _escape_latex(item).replace("LEFT", r"\(").replace("RIGHT", r"\)")
            )
        else:
            res.append(
                _escape_latex(item)
                .replace("LEFT", r"\textbackslash (")
                .replace("RIGHT", r"\textbackslash )")
            )
    return "".join(res)


def _escape_latex_math(s: str) -> str:
    r"""
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which either are surrounded
    by two characters ``$`` or start with the character ``\(`` and end with ``\)``,
    are preserved without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace(r"\$", r"rt8§=§7wz")
    ps_d = re.compile(r"\$.*?\$").search(s, 0)
    ps_p = re.compile(r"\(.*?\)").search(s, 0)
    mode = []
    if ps_d:
        mode.append(ps_d.span()[0])
    if ps_p:
        mode.append(ps_p.span()[0])
    if len(mode) == 0:
        return _escape_latex(s.replace(r"rt8§=§7wz", r"\$"))
    if s[mode[0]] == r"$":
        return _math_mode_with_dollar(s.replace(r"rt8§=§7wz", r"\$"))
    if s[mode[0] - 1 : mode[0] + 1] == r"\(":
        return _math_mode_with_parentheses(s.replace(r"rt8§=§7wz", r"\$"))
    else:
        return _escape_latex(s.replace(r"rt8§=§7wz", r"\$"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\hyperexpand.py ===
"""
Expand Hypergeometric (and Meijer G) functions into named
special functions.

The algorithm for doing this uses a collection of lookup tables of
hypergeometric functions, and various of their properties, to expand
many hypergeometric functions in terms of special functions.

It is based on the following paper:
      Kelly B. Roach.  Meijer G Function Representations.
      In: Proceedings of the 1997 International Symposium on Symbolic and
      Algebraic Computation, pages 205-211, New York, 1997. ACM.

It is described in great(er) detail in the Sphinx documentation.
"""
# SUMMARY OF EXTENSIONS FOR MEIJER G FUNCTIONS
#
# o z**rho G(ap, bq; z) = G(ap + rho, bq + rho; z)
#
# o denote z*d/dz by D
#
# o It is helpful to keep in mind that ap and bq play essentially symmetric
#   roles: G(1/z) has slightly altered parameters, with ap and bq interchanged.
#
# o There are four shift operators:
#   A_J = b_J - D,     J = 1, ..., n
#   B_J = 1 - a_j + D, J = 1, ..., m
#   C_J = -b_J + D,    J = m+1, ..., q
#   D_J = a_J - 1 - D, J = n+1, ..., p
#
#   A_J, C_J increment b_J
#   B_J, D_J decrement a_J
#
# o The corresponding four inverse-shift operators are defined if there
#   is no cancellation. Thus e.g. an index a_J (upper or lower) can be
#   incremented if a_J != b_i for i = 1, ..., q.
#
# o Order reduction: if b_j - a_i is a non-negative integer, where
#   j <= m and i > n, the corresponding quotient of gamma functions reduces
#   to a polynomial. Hence the G function can be expressed using a G-function
#   of lower order.
#   Similarly if j > m and i <= n.
#
#   Secondly, there are paired index theorems [Adamchik, The evaluation of
#   integrals of Bessel functions via G-function identities]. Suppose there
#   are three parameters a, b, c, where a is an a_i, i <= n, b is a b_j,
#   j <= m and c is a denominator parameter (i.e. a_i, i > n or b_j, j > m).
#   Suppose further all three differ by integers.
#   Then the order can be reduced.
#   TODO work this out in detail.
#
# o An index quadruple is called suitable if its order cannot be reduced.
#   If there exists a sequence of shift operators transforming one index
#   quadruple into another, we say one is reachable from the other.
#
# o Deciding if one index quadruple is reachable from another is tricky. For
#   this reason, we use hand-built routines to match and instantiate formulas.
#
from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod

from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
    EulerGamma, oo, zoo, expand_func, Add, nan, Expr, Rational)
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
        besseli, gamma, uppergamma, expint, erf, sin, besselj, Ei, Ci, Si, Shi,
        sinh, cosh, Chi, fresnels, fresnelc, polar_lift, exp_polar, floor, ceiling,
        rf, factorial, lerchphi, Piecewise, re, elliptic_k, elliptic_e)
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
        HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1,
        HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2,
        HyperRep_cosasin, HyperRep_sinasin, meijerg)
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift

# function to define "buckets"
def _mod1(x):
    # TODO see if this can work as Mod(x, 1); this will require
    # different handling of the "buckets" since these need to
    # be sorted and that fails when there is a mixture of
    # integers and expressions with parameters. With the current
    # Mod behavior, Mod(k, 1) == Mod(1, 1) == 0 if k is an integer.
    # Although the sorting can be done with Basic.compare, this may
    # still require different handling of the sorted buckets.
    if x.is_Number:
        return Mod(x, 1)
    c, x = x.as_coeff_Add()
    return Mod(c, 1) + x


# leave add formulae at the top for easy reference
def add_formulae(formulae):
    """ Create our knowledge base. """
    a, b, c, z = symbols('a b c, z', cls=Dummy)

    def add(ap, bq, res):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, res, (a, b, c)))

    def addb(ap, bq, B, C, M):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, None, (a, b, c), B, C, M))

    # Luke, Y. L. (1969), The Special Functions and Their Approximations,
    # Volume 1, section 6.2

    # 0F0
    add((), (), exp(z))

    # 1F0
    add((a, ), (), HyperRep_power1(-a, z))

    # 2F1
    addb((a, a - S.Half), (2*a, ),
         Matrix([HyperRep_power2(a, z),
                 HyperRep_power2(a + S.Half, z)/2]),
         Matrix([[1, 0]]),
         Matrix([[(a - S.Half)*z/(1 - z), (S.Half - a)*z/(1 - z)],
                 [a/(1 - z), a*(z - 2)/(1 - z)]]))
    addb((1, 1), (2, ),
         Matrix([HyperRep_log1(z), 1]), Matrix([[-1/z, 0]]),
         Matrix([[0, z/(z - 1)], [0, 0]]))
    addb((S.Half, 1), (S('3/2'), ),
         Matrix([HyperRep_atanh(z), 1]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), 1/(1 - z)/2], [0, 0]]))
    addb((S.Half, S.Half), (S('3/2'), ),
         Matrix([HyperRep_asin1(z), HyperRep_power1(Rational(-1, 2), z)]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), S.Half], [0, z/(1 - z)/2]]))
    addb((a, S.Half + a), (S.Half, ),
         Matrix([HyperRep_sqrts1(-a, z), -HyperRep_sqrts2(-a - S.Half, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a],
                 [z*(-2*a - 1)/2/(1 - z), S.Half - z*(-2*a - 1)/(1 - z)]]))

    # A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    # Integrals and Series: More Special Functions, Vol. 3,.
    # Gordon and Breach Science Publisher
    addb([a, -a], [S.Half],
         Matrix([HyperRep_cosasin(a, z), HyperRep_sinasin(a, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a], [a*z/(1 - z), 1/(1 - z)/2]]))
    addb([1, 1], [3*S.Half],
         Matrix([HyperRep_asin2(z), 1]), Matrix([[1, 0]]),
         Matrix([[(z - S.Half)/(1 - z), 1/(1 - z)/2], [0, 0]]))

    # Complete elliptic integrals K(z) and E(z), both a 2F1 function
    addb([S.Half, S.Half], [S.One],
         Matrix([elliptic_k(z), elliptic_e(z)]),
         Matrix([[2/pi, 0]]),
         Matrix([[Rational(-1, 2), -1/(2*z-2)],
                 [Rational(-1, 2), S.Half]]))
    addb([Rational(-1, 2), S.Half], [S.One],
         Matrix([elliptic_k(z), elliptic_e(z)]),
         Matrix([[0, 2/pi]]),
         Matrix([[Rational(-1, 2), -1/(2*z-2)],
                 [Rational(-1, 2), S.Half]]))

    # 3F2
    addb([Rational(-1, 2), 1, 1], [S.Half, 2],
         Matrix([z*HyperRep_atanh(z), HyperRep_log1(z), 1]),
         Matrix([[Rational(-2, 3), -S.One/(3*z), Rational(2, 3)]]),
         Matrix([[S.Half, 0, z/(1 - z)/2],
                 [0, 0, z/(z - 1)],
                 [0, 0, 0]]))
    # actually the formula for 3/2 is much nicer ...
    addb([Rational(-1, 2), 1, 1], [2, 2],
         Matrix([HyperRep_power1(S.Half, z), HyperRep_log2(z), 1]),
         Matrix([[Rational(4, 9) - 16/(9*z), 4/(3*z), 16/(9*z)]]),
         Matrix([[z/2/(z - 1), 0, 0], [1/(2*(z - 1)), 0, S.Half], [0, 0, 0]]))

    # 1F1
    addb([1], [b], Matrix([z**(1 - b) * exp(z) * lowergamma(b - 1, z), 1]),
         Matrix([[b - 1, 0]]), Matrix([[1 - b + z, 1], [0, 0]]))
    addb([a], [2*a],
         Matrix([z**(S.Half - a)*exp(z/2)*besseli(a - S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a),
                 z**(S.Half - a)*exp(z/2)*besseli(a + S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a)]),
         Matrix([[1, 0]]),
         Matrix([[z/2, z/2], [z/2, (z/2 - 2*a)]]))
    mz = polar_lift(-1)*z
    addb([a], [a + 1],
         Matrix([mz**(-a)*a*lowergamma(a, mz), a*exp(z)]),
         Matrix([[1, 0]]),
         Matrix([[-a, 1], [0, z]]))
    # This one is redundant.
    add([Rational(-1, 2)], [S.Half], exp(z) - sqrt(pi*z)*(-I)*erf(I*sqrt(z)))

    # Added to get nice results for Laplace transform of Fresnel functions
    # https://functions.wolfram.com/07.22.03.6437.01
    # Basic rule
    #add([1], [Rational(3, 4), Rational(5, 4)],
    #    sqrt(pi) * (cos(2*sqrt(polar_lift(-1)*z))*fresnelc(2*root(polar_lift(-1)*z,4)/sqrt(pi)) +
    #                sin(2*sqrt(polar_lift(-1)*z))*fresnels(2*root(polar_lift(-1)*z,4)/sqrt(pi)))
    #    / (2*root(polar_lift(-1)*z,4)))
    # Manually tuned rule
    addb([1], [Rational(3, 4), Rational(5, 4)],
         Matrix([ sqrt(pi)*(I*sinh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                            + cosh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  * exp(-I*pi/4)/(2*root(z, 4)),
                  sqrt(pi)*root(z, 4)*(sinh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                                      + I*cosh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  *exp(-I*pi/4)/2,
                  1 ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),              1, Rational(1, 4)],
                 [              z, Rational(1, 4),              0],
                 [              0,              0,              0]]))

    # 2F2
    addb([S.Half, a], [Rational(3, 2), a + 1],
         Matrix([a/(2*a - 1)*(-I)*sqrt(pi/z)*erf(I*sqrt(z)),
                 a/(2*a - 1)*(polar_lift(-1)*z)**(-a)*
                 lowergamma(a, polar_lift(-1)*z),
                 a/(2*a - 1)*exp(z)]),
         Matrix([[1, -1, 0]]),
         Matrix([[Rational(-1, 2), 0, 1], [0, -a, 1], [0, 0, z]]))
    # We make a "basis" of four functions instead of three, and give EulerGamma
    # an extra slot (it could just be a coefficient to 1). The advantage is
    # that this way Polys will not see multivariate polynomials (it treats
    # EulerGamma as an indeterminate), which is *way* faster.
    addb([1, 1], [2, 2],
         Matrix([Ei(z) - log(z), exp(z), 1, EulerGamma]),
         Matrix([[1/z, 0, 0, -1/z]]),
         Matrix([[0, 1, -1, 0], [0, z, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

    # 0F1
    add((), (S.Half, ), cosh(2*sqrt(z)))
    addb([], [b],
         Matrix([gamma(b)*z**((1 - b)/2)*besseli(b - 1, 2*sqrt(z)),
                 gamma(b)*z**(1 - b/2)*besseli(b, 2*sqrt(z))]),
         Matrix([[1, 0]]), Matrix([[0, 1], [z, (1 - b)]]))

    # 0F3
    x = 4*z**Rational(1, 4)

    def fp(a, z):
        return besseli(a, x) + besselj(a, x)

    def fm(a, z):
        return besseli(a, x) - besselj(a, x)

    # TODO branching
    addb([], [S.Half, a, a + S.Half],
         Matrix([fp(2*a - 1, z), fm(2*a, z)*z**Rational(1, 4),
                 fm(2*a - 1, z)*sqrt(z), fp(2*a, z)*z**Rational(3, 4)])
         * 2**(-2*a)*gamma(2*a)*z**((1 - 2*a)/4),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, 1, 0, 0],
                 [0, S.Half - a, 1, 0],
                 [0, 0, S.Half, 1],
                 [z, 0, 0, 1 - a]]))
    x = 2*(4*z)**Rational(1, 4)*exp_polar(I*pi/4)
    addb([], [a, a + S.Half, 2*a],
         (2*sqrt(polar_lift(-1)*z))**(1 - 2*a)*gamma(2*a)**2 *
         Matrix([besselj(2*a - 1, x)*besseli(2*a - 1, x),
                 x*(besseli(2*a, x)*besselj(2*a - 1, x)
                    - besseli(2*a - 1, x)*besselj(2*a, x)),
                 x**2*besseli(2*a, x)*besselj(2*a, x),
                 x**3*(besseli(2*a, x)*besselj(2*a - 1, x)
                       + besseli(2*a - 1, x)*besselj(2*a, x))]),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, Rational(1, 4), 0, 0],
                 [0, (1 - 2*a)/2, Rational(-1, 2), 0],
                 [0, 0, 1 - 2*a, Rational(1, 4)],
                 [-32*z, 0, 0, 1 - a]]))

    # 1F2
    addb([a], [a - S.Half, 2*a],
         Matrix([z**(S.Half - a)*besseli(a - S.Half, sqrt(z))**2,
                 z**(1 - a)*besseli(a - S.Half, sqrt(z))
                 *besseli(a - Rational(3, 2), sqrt(z)),
                 z**(Rational(3, 2) - a)*besseli(a - Rational(3, 2), sqrt(z))**2]),
         Matrix([[-gamma(a + S.Half)**2/4**(S.Half - a),
                 2*gamma(a - S.Half)*gamma(a + S.Half)/4**(1 - a),
                 0]]),
         Matrix([[1 - 2*a, 1, 0], [z/2, S.Half - a, S.Half], [0, z, 0]]))
    addb([S.Half], [b, 2 - b],
         pi*(1 - b)/sin(pi*b)*
         Matrix([besseli(1 - b, sqrt(z))*besseli(b - 1, sqrt(z)),
                 sqrt(z)*(besseli(-b, sqrt(z))*besseli(b - 1, sqrt(z))
                          + besseli(1 - b, sqrt(z))*besseli(b, sqrt(z))),
                 besseli(-b, sqrt(z))*besseli(b, sqrt(z))]),
         Matrix([[1, 0, 0]]),
         Matrix([[b - 1, S.Half, 0],
                 [z, 0, z],
                 [0, S.Half, -b]]))
    addb([S.Half], [Rational(3, 2), Rational(3, 2)],
         Matrix([Shi(2*sqrt(z))/2/sqrt(z), sinh(2*sqrt(z))/2/sqrt(z),
                 cosh(2*sqrt(z))]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 2), S.Half, 0], [0, Rational(-1, 2), S.Half], [0, 2*z, 0]]))

    # FresnelS
    # Basic rule
    #add([Rational(3, 4)], [Rational(3, 2),Rational(7, 4)], 6*fresnels( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) / ( pi * (exp(pi*I/4)*root(z,4)*2/sqrt(pi))**3 ) )
    # Manually tuned rule
    addb([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)],
         Matrix(
             [ fresnels(
                 exp(
                     pi*I/4)*root(
                         z, 4)*2/sqrt(
                             pi) ) / (
                                 pi * (exp(pi*I/4)*root(z, 4)*2/sqrt(pi))**3 ),
               sinh(2*sqrt(z))/sqrt(z),
               cosh(2*sqrt(z)) ]),
         Matrix([[6, 0, 0]]),
         Matrix([[Rational(-3, 4),  Rational(1, 16), 0],
                 [ 0,      Rational(-1, 2),  1],
                 [ 0,       z,       0]]))

    # FresnelC
    # Basic rule
    #add([Rational(1, 4)], [S.Half,Rational(5, 4)], fresnelc( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) / ( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) )
    # Manually tuned rule
    addb([Rational(1, 4)], [S.Half, Rational(5, 4)],
         Matrix(
             [ sqrt(
                 pi)*exp(
                     -I*pi/4)*fresnelc(
                         2*root(z, 4)*exp(I*pi/4)/sqrt(pi))/(2*root(z, 4)),
               cosh(2*sqrt(z)),
               sinh(2*sqrt(z))*sqrt(z) ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),  Rational(1, 4), 0     ],
                 [ 0,       0,      1     ],
                 [ 0,       z,      S.Half]]))

    # 2F3
    # XXX with this five-parameter formula is pretty slow with the current
    #     Formula.find_instantiations (creates 2!*3!*3**(2+3) ~ 3000
    #     instantiations ... But it's not too bad.
    addb([a, a + S.Half], [2*a, b, 2*a - b + 1],
         gamma(b)*gamma(2*a - b + 1) * (sqrt(z)/2)**(1 - 2*a) *
         Matrix([besseli(b - 1, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b - 1, sqrt(z))*besseli(2*a - b + 1, sqrt(z)),
                 besseli(b, sqrt(z))*besseli(2*a - b + 1, sqrt(z))]),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, S.Half, S.Half, 0],
                 [z/2, 1 - b, 0, z/2],
                 [z/2, 0, b - 2*a, z/2],
                 [0, S.Half, S.Half, -2*a]]))
    # (C/f above comment about eulergamma in the basis).
    addb([1, 1], [2, 2, Rational(3, 2)],
         Matrix([Chi(2*sqrt(z)) - log(2*sqrt(z)),
                 cosh(2*sqrt(z)), sqrt(z)*sinh(2*sqrt(z)), 1, EulerGamma]),
         Matrix([[1/z, 0, 0, 0, -1/z]]),
         Matrix([[0, S.Half, 0, Rational(-1, 2), 0],
                 [0, 0, 1, 0, 0],
                 [0, z, S.Half, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]))

    # 3F3
    # This is rule: https://functions.wolfram.com/07.31.03.0134.01
    # Initial reason to add it was a nice solution for
    # integrate(erf(a*z)/z**2, z) and same for erfc and erfi.
    # Basic rule
    # add([1, 1, a], [2, 2, a+1], (a/(z*(a-1)**2)) *
    #     (1 - (-z)**(1-a) * (gamma(a) - uppergamma(a,-z))
    #      - (a-1) * (EulerGamma + uppergamma(0,-z) + log(-z))
    #      - exp(z)))
    # Manually tuned rule
    addb([1, 1, a], [2, 2, a+1],
         Matrix([a*(log(-z) + expint(1, -z) + EulerGamma)/(z*(a**2 - 2*a + 1)),
                 a*(-z)**(-a)*(gamma(a) - uppergamma(a, -z))/(a - 1)**2,
                 a*exp(z)/(a**2 - 2*a + 1),
                 a/(z*(a**2 - 2*a + 1))]),
         Matrix([[1-a, 1, -1/z, 1]]),
         Matrix([[-1,0,-1/z,1],
                 [0,-a,1,0],
                 [0,0,z,0],
                 [0,0,0,-1]]))


def add_meijerg_formulae(formulae):
    a, b, c, z = list(map(Dummy, 'abcz'))
    rho = Dummy('rho')

    def add(an, ap, bm, bq, B, C, M, matcher):
        formulae.append(MeijerFormula(an, ap, bm, bq, z, [a, b, c, rho],
                                      B, C, M, matcher))

    def detect_uppergamma(func):
        x = func.an[0]
        y, z = func.bm
        swapped = False
        if not _mod1((x - y).simplify()):
            swapped = True
            (y, z) = (z, y)
        if _mod1((x - z).simplify()) or x - z > 0:
            return None
        l = [y, x]
        if swapped:
            l = [x, y]
        return {rho: y, a: x - y}, G_Function([x], [], l, [])

    add([a + rho], [], [rho, a + rho], [],
        Matrix([gamma(1 - a)*z**rho*exp(z)*uppergamma(a, z),
                gamma(1 - a)*z**(a + rho)]),
        Matrix([[1, 0]]),
        Matrix([[rho + z, -1], [0, a + rho]]),
        detect_uppergamma)

    def detect_3113(func):
        """https://functions.wolfram.com/07.34.03.0984.01"""
        x = func.an[0]
        u, v, w = func.bm
        if _mod1((u - v).simplify()) == 0:
            if _mod1((v - w).simplify()) == 0:
                return
            sig = (S.Half, S.Half, S.Zero)
            x1, x2, y = u, v, w
        else:
            if _mod1((x - u).simplify()) == 0:
                sig = (S.Half, S.Zero, S.Half)
                x1, y, x2 = u, v, w
            else:
                sig = (S.Zero, S.Half, S.Half)
                y, x1, x2 = u, v, w

        if (_mod1((x - x1).simplify()) != 0 or
            _mod1((x - x2).simplify()) != 0 or
            _mod1((x - y).simplify()) != S.Half or
                x - x1 > 0 or x - x2 > 0):
            return

        return {a: x}, G_Function([x], [], [x - S.Half + t for t in sig], [])

    s = sin(2*sqrt(z))
    c_ = cos(2*sqrt(z))
    S_ = Si(2*sqrt(z)) - pi/2
    C = Ci(2*sqrt(z))
    add([a], [], [a, a, a - S.Half], [],
        Matrix([sqrt(pi)*z**(a - S.Half)*(c_*S_ - s*C),
                sqrt(pi)*z**a*(s*S_ + c_*C),
                sqrt(pi)*z**a]),
        Matrix([[-2, 0, 0]]),
        Matrix([[a - S.Half, -1, 0], [z, a, S.Half], [0, 0, a]]),
        detect_3113)


def make_simp(z):
    """ Create a function that simplifies rational functions in ``z``. """

    def simp(expr):
        """ Efficiently simplify the rational function ``expr``. """
        numer, denom = expr.as_numer_denom()
        numer = numer.expand()
        # denom = denom.expand()  # is this needed?
        c, numer, denom = poly(numer, z).cancel(poly(denom, z))
        return c * numer.as_expr() / denom.as_expr()

    return simp


def debug(*args):
    if SYMPY_DEBUG:
        for a in args:
            print(a, end="")
        print()


class Hyper_Function(Expr):
    """ A generalized hypergeometric function. """

    def __new__(cls, ap, bq):
        obj = super().__new__(cls)
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.ap, self.bq)

    @property
    def sizes(self):
        return (len(self.ap), len(self.bq))

    @property
    def gamma(self):
        """
        Number of upper parameters that are negative integers

        This is a transformation invariant.
        """
        return sum(bool(x.is_integer and x.is_negative) for x in self.ap)

    def _hashable_content(self):
        return super()._hashable_content() + (self.ap,
                self.bq)

    def __call__(self, arg):
        return hyper(self.ap, self.bq, arg)

    def build_invariants(self):
        """
        Compute the invariant vector.

        Explanation
        ===========

        The invariant vector is:
            (gamma, ((s1, n1), ..., (sk, nk)), ((t1, m1), ..., (tr, mr)))
        where gamma is the number of integer a < 0,
              s1 < ... < sk
              nl is the number of parameters a_i congruent to sl mod 1
              t1 < ... < tr
              ml is the number of parameters b_i congruent to tl mod 1

        If the index pair contains parameters, then this is not truly an
        invariant, since the parameters cannot be sorted uniquely mod1.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Hyper_Function
        >>> from sympy import S
        >>> ap = (S.Half, S.One/3, S(-1)/2, -2)
        >>> bq = (1, 2)

        Here gamma = 1,
             k = 3, s1 = 0, s2 = 1/3, s3 = 1/2
                    n1 = 1, n2 = 1,   n2 = 2
             r = 1, t1 = 0
                    m1 = 2:

        >>> Hyper_Function(ap, bq).build_invariants()
        (1, ((0, 1), (1/3, 1), (1/2, 2)), ((0, 2),))
        """
        abuckets, bbuckets = sift(self.ap, _mod1), sift(self.bq, _mod1)

        def tr(bucket):
            bucket = list(bucket.items())
            if not any(isinstance(x[0], Mod) for x in bucket):
                bucket.sort(key=lambda x: default_sort_key(x[0]))
            bucket = tuple([(mod, len(values)) for mod, values in bucket if
                    values])
            return bucket

        return (self.gamma, tr(abuckets), tr(bbuckets))

    def difficulty(self, func):
        """ Estimate how many steps it takes to reach ``func`` from self.
            Return -1 if impossible. """
        if self.gamma != func.gamma:
            return -1
        oabuckets, obbuckets, abuckets, bbuckets = [sift(params, _mod1) for
                params in (self.ap, self.bq, func.ap, func.bq)]

        diff = 0
        for bucket, obucket in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
            for mod in set(list(bucket.keys()) + list(obucket.keys())):
                if (mod not in bucket) or (mod not in obucket) \
                        or len(bucket[mod]) != len(obucket[mod]):
                    return -1
                l1 = list(bucket[mod])
                l2 = list(obucket[mod])
                l1.sort()
                l2.sort()
                for i, j in zip(l1, l2):
                    diff += abs(i - j)

        return diff

    def _is_suitable_origin(self):
        """
        Decide if ``self`` is a suitable origin.

        Explanation
        ===========

        A function is a suitable origin iff:
        * none of the ai equals bj + n, with n a non-negative integer
        * none of the ai is zero
        * none of the bj is a non-positive integer

        Note that this gives meaningful results only when none of the indices
        are symbolic.

        """
        for a in self.ap:
            for b in self.bq:
                if (a - b).is_integer and (a - b).is_negative is False:
                    return False
        for a in self.ap:
            if a == 0:
                return False
        for b in self.bq:
            if b.is_integer and b.is_nonpositive:
                return False
        return True


class G_Function(Expr):
    """ A Meijer G-function. """

    def __new__(cls, an, ap, bm, bq):
        obj = super().__new__(cls)
        obj.an = Tuple(*list(map(expand, an)))
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bm = Tuple(*list(map(expand, bm)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.an, self.ap, self.bm, self.bq)

    def _hashable_content(self):
        return super()._hashable_content() + self.args

    def __call__(self, z):
        return meijerg(self.an, self.ap, self.bm, self.bq, z)

    def compute_buckets(self):
        """
        Compute buckets for the fours sets of parameters.

        Explanation
        ===========

        We guarantee that any two equal Mod objects returned are actually the
        same, and that the buckets are sorted by real part (an and bq
        descendending, bm and ap ascending).

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import G_Function
        >>> from sympy.abc import y
        >>> from sympy import S

        >>> a, b = [1, 3, 2, S(3)/2], [1 + y, y, 2, y + 3]
        >>> G_Function(a, b, [2], [y]).compute_buckets()
        ({0: [3, 2, 1], 1/2: [3/2]},
        {0: [2], y: [y, y + 1, y + 3]}, {0: [2]}, {y: [y]})

        """
        dicts = pan, pap, pbm, pbq = [defaultdict(list) for i in range(4)]
        for dic, lis in zip(dicts, (self.an, self.ap, self.bm, self.bq)):
            for x in lis:
                dic[_mod1(x)].append(x)

        for dic, flip in zip(dicts, (True, False, False, True)):
            for m, items in dic.items():
                x0 = items[0]
                items.sort(key=lambda x: x - x0, reverse=flip)
                dic[m] = items

        return tuple([dict(w) for w in dicts])

    @property
    def signature(self):
        return (len(self.an), len(self.ap), len(self.bm), len(self.bq))


# Dummy variable.
_x = Dummy('x')

class Formula:
    """
    This class represents hypergeometric formulae.

    Explanation
    ===========

    Its data members are:
    - z, the argument
    - closed_form, the closed form expression
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (see _compute_basis)

    Examples
    ========

    >>> from sympy.abc import a, b, z
    >>> from sympy.simplify.hyperexpand import Formula, Hyper_Function
    >>> func = Hyper_Function((a/2, a/3 + b, (1+a)/2), (a, b, (a+b)/7))
    >>> f = Formula(func, z, None, [a, b])

    """

    def _compute_basis(self, closed_form):
        """
        Compute a set of functions B=(f1, ..., fn), a nxn matrix M
        and a 1xn matrix C such that:
           closed_form = C B
           z d/dz B = M B.
        """
        afactors = [_x + a for a in self.func.ap]
        bfactors = [_x + b - 1 for b in self.func.bq]
        expr = _x*Mul(*bfactors) - self.z*Mul(*afactors)
        poly = Poly(expr, _x)

        n = poly.degree() - 1
        b = [closed_form]
        for _ in range(n):
            b.append(self.z*b[-1].diff(self.z))

        self.B = Matrix(b)
        self.C = Matrix([[1] + [0]*n])

        m = eye(n)
        m = m.col_insert(0, zeros(n, 1))
        l = poly.all_coeffs()[1:]
        l.reverse()
        self.M = m.row_insert(n, -Matrix([l])/poly.all_coeffs()[0])

    def __init__(self, func, z, res, symbols, B=None, C=None, M=None):
        z = sympify(z)
        res = sympify(res)
        symbols = [x for x in sympify(symbols) if func.has(x)]

        self.z = z
        self.symbols = symbols
        self.B = B
        self.C = C
        self.M = M
        self.func = func

        # TODO with symbolic parameters, it could be advantageous
        #      (for prettier answers) to compute a basis only *after*
        #      instantiation
        if res is not None:
            self._compute_basis(res)

    @property
    def closed_form(self):
        return reduce(lambda s,m: s+m[0]*m[1], zip(self.C, self.B), S.Zero)

    def find_instantiations(self, func):
        """
        Find substitutions of the free symbols that match ``func``.

        Return the substitution dictionaries as a list. Note that the returned
        instantiations need not actually match, or be valid!

        """
        from sympy.solvers import solve
        ap = func.ap
        bq = func.bq
        if len(ap) != len(self.func.ap) or len(bq) != len(self.func.bq):
            raise TypeError('Cannot instantiate other number of parameters')
        symbol_values = []
        for a in self.symbols:
            if a in self.func.ap.args:
                symbol_values.append(ap)
            elif a in self.func.bq.args:
                symbol_values.append(bq)
            else:
                raise ValueError("At least one of the parameters of the "
                        "formula must be equal to %s" % (a,))
        base_repl = [dict(list(zip(self.symbols, values)))
                for values in product(*symbol_values)]
        abuckets, bbuckets = [sift(params, _mod1) for params in [ap, bq]]
        a_inv, b_inv = [{a: len(vals) for a, vals in bucket.items()}
                for bucket in [abuckets, bbuckets]]
        critical_values = [[0] for _ in self.symbols]
        result = []
        _n = Dummy()
        for repl in base_repl:
            symb_a, symb_b = [sift(params, lambda x: _mod1(x.xreplace(repl)))
                for params in [self.func.ap, self.func.bq]]
            for bucket, obucket in [(abuckets, symb_a), (bbuckets, symb_b)]:
                for mod in set(list(bucket.keys()) + list(obucket.keys())):
                    if (mod not in bucket) or (mod not in obucket) \
                            or len(bucket[mod]) != len(obucket[mod]):
                        break
                    for a, vals in zip(self.symbols, critical_values):
                        if repl[a].free_symbols:
                            continue
                        exprs = [expr for expr in obucket[mod] if expr.has(a)]
                        repl0 = repl.copy()
                        repl0[a] += _n
                        for expr in exprs:
                            for target in bucket[mod]:
                                n0, = solve(expr.xreplace(repl0) - target, _n)
                                if n0.free_symbols:
                                    raise ValueError("Value should not be true")
                                vals.append(n0)
            else:
                values = []
                for a, vals in zip(self.symbols, critical_values):
                    a0 = repl[a]
                    min_ = floor(min(vals))
                    max_ = ceiling(max(vals))
                    values.append([a0 + n for n in range(min_, max_ + 1)])
                result.extend(dict(list(zip(self.symbols, l))) for l in product(*values))
        return result




class FormulaCollection:
    """ A collection of formulae to use as origins. """

    def __init__(self):
        """ Doing this globally at module init time is a pain ... """
        self.symbolic_formulae = {}
        self.concrete_formulae = {}
        self.formulae = []

        add_formulae(self.formulae)

        # Now process the formulae into a helpful form.
        # These dicts are indexed by (p, q).

        for f in self.formulae:
            sizes = f.func.sizes
            if len(f.symbols) > 0:
                self.symbolic_formulae.setdefault(sizes, []).append(f)
            else:
                inv = f.func.build_invariants()
                self.concrete_formulae.setdefault(sizes, {})[inv] = f

    def lookup_origin(self, func):
        """
        Given the suitable target ``func``, try to find an origin in our
        knowledge base.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import (FormulaCollection,
        ...     Hyper_Function)
        >>> f = FormulaCollection()
        >>> f.lookup_origin(Hyper_Function((), ())).closed_form
        exp(_z)
        >>> f.lookup_origin(Hyper_Function([1], ())).closed_form
        HyperRep_power1(-1, _z)

        >>> from sympy import S
        >>> i = Hyper_Function([S('1/4'), S('3/4 + 4')], [S.Half])
        >>> f.lookup_origin(i).closed_form
        HyperRep_sqrts1(-1/4, _z)
        """
        inv = func.build_invariants()
        sizes = func.sizes
        if sizes in self.concrete_formulae and \
                inv in self.concrete_formulae[sizes]:
            return self.concrete_formulae[sizes][inv]

        # We don't have a concrete formula. Try to instantiate.
        if sizes not in self.symbolic_formulae:
            return None  # Too bad...

        possible = []
        for f in self.symbolic_formulae[sizes]:
            repls = f.find_instantiations(func)
            for repl in repls:
                func2 = f.func.xreplace(repl)
                if not func2._is_suitable_origin():
                    continue
                diff = func2.difficulty(func)
                if diff == -1:
                    continue
                possible.append((diff, repl, f, func2))

        # find the nearest origin
        possible.sort(key=lambda x: x[0])
        for _, repl, f, func2 in possible:
            f2 = Formula(func2, f.z, None, [], f.B.subs(repl),
                    f.C.subs(repl), f.M.subs(repl))
            if not any(e.has(S.NaN, oo, -oo, zoo) for e in [f2.B, f2.M, f2.C]):
                return f2

        return None


class MeijerFormula:
    """
    This class represents a Meijer G-function formula.

    Its data members are:
    - z, the argument
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (c/f ordinary Formula)
    """

    def __init__(self, an, ap, bm, bq, z, symbols, B, C, M, matcher):
        an, ap, bm, bq = [Tuple(*list(map(expand, w))) for w in [an, ap, bm, bq]]
        self.func = G_Function(an, ap, bm, bq)
        self.z = z
        self.symbols = symbols
        self._matcher = matcher
        self.B = B
        self.C = C
        self.M = M

    @property
    def closed_form(self):
        return reduce(lambda s,m: s+m[0]*m[1], zip(self.C, self.B), S.Zero)

    def try_instantiate(self, func):
        """
        Try to instantiate the current formula to (almost) match func.
        This uses the _matcher passed on init.
        """
        if func.signature != self.func.signature:
            return None
        res = self._matcher(func)
        if res is not None:
            subs, newfunc = res
            return MeijerFormula(newfunc.an, newfunc.ap, newfunc.bm, newfunc.bq,
                                 self.z, [],
                                 self.B.subs(subs), self.C.subs(subs),
                                 self.M.subs(subs), None)


class MeijerFormulaCollection:
    """
    This class holds a collection of meijer g formulae.
    """

    def __init__(self):
        formulae = []
        add_meijerg_formulae(formulae)
        self.formulae = defaultdict(list)
        for formula in formulae:
            self.formulae[formula.func.signature].append(formula)
        self.formulae = dict(self.formulae)

    def lookup_origin(self, func):
        """ Try to find a formula that matches func. """
        if func.signature not in self.formulae:
            return None
        for formula in self.formulae[func.signature]:
            res = formula.try_instantiate(func)
            if res is not None:
                return res


class Operator:
    """
    Base class for operators to be applied to our functions.

    Explanation
    ===========

    These operators are differential operators. They are by convention
    expressed in the variable D = z*d/dz (although this base class does
    not actually care).
    Note that when the operator is applied to an object, we typically do
    *not* blindly differentiate but instead use a different representation
    of the z*d/dz operator (see make_derivative_operator).

    To subclass from this, define a __init__ method that initializes a
    self._poly variable. This variable stores a polynomial. By convention
    the generator is z*d/dz, and acts to the right of all coefficients.

    Thus this poly
        x**2 + 2*z*x + 1
    represents the differential operator
        (z*d/dz)**2 + 2*z**2*d/dz.

    This class is used only in the implementation of the hypergeometric
    function expansion algorithm.
    """

    def apply(self, obj, op):
        """
        Apply ``self`` to the object ``obj``, where the generator is ``op``.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Operator
        >>> from sympy.polys.polytools import Poly
        >>> from sympy.abc import x, y, z
        >>> op = Operator()
        >>> op._poly = Poly(x**2 + z*x + y, x)
        >>> op.apply(z**7, lambda f: f.diff(z))
        y*z**7 + 7*z**7 + 42*z**5
        """
        coeffs = self._poly.all_coeffs()
        coeffs.reverse()
        diffs = [obj]
        for c in coeffs[1:]:
            diffs.append(op(diffs[-1]))
        r = coeffs[0]*diffs[0]
        for c, d in zip(coeffs[1:], diffs[1:]):
            r += c*d
        return r


class MultOperator(Operator):
    """ Simply multiply by a "constant" """

    def __init__(self, p):
        self._poly = Poly(p, _x)


class ShiftA(Operator):
    """ Increment an upper index. """

    def __init__(self, ai):
        ai = sympify(ai)
        if ai == 0:
            raise ValueError('Cannot increment zero upper index.')
        self._poly = Poly(_x/ai + 1, _x)

    def __str__(self):
        return '<Increment upper %s.>' % (1/self._poly.all_coeffs()[0])


class ShiftB(Operator):
    """ Decrement a lower index. """

    def __init__(self, bi):
        bi = sympify(bi)
        if bi == 1:
            raise ValueError('Cannot decrement unit lower index.')
        self._poly = Poly(_x/(bi - 1) + 1, _x)

    def __str__(self):
        return '<Decrement lower %s.>' % (1/self._poly.all_coeffs()[0] + 1)


class UnShiftA(Operator):
    """ Decrement an upper index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        self._ap = ap
        self._bq = bq
        self._i = i

        ap = list(ap)
        bq = list(bq)
        ai = ap.pop(i) - 1

        if ai == 0:
            raise ValueError('Cannot decrement unit upper index.')

        m = Poly(z*ai, _x)
        for a in ap:
            m *= Poly(_x + a, _x)

        A = Dummy('A')
        n = D = Poly(ai*A - ai, A)
        for b in bq:
            n *= D + (b - 1).as_poly(A)

        b0 = -n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper index: '
                             'cancels with lower')

        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, _x/ai + 1), _x)

        self._poly = Poly((n - m)/b0, _x)

    def __str__(self):
        return '<Decrement upper index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class UnShiftB(Operator):
    """ Increment a lower index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        self._ap = ap
        self._bq = bq
        self._i = i

        ap = list(ap)
        bq = list(bq)
        bi = bq.pop(i) + 1

        if bi == 0:
            raise ValueError('Cannot increment -1 lower index.')

        m = Poly(_x*(bi - 1), _x)
        for b in bq:
            m *= Poly(_x + b - 1, _x)

        B = Dummy('B')
        D = Poly((bi - 1)*B - bi + 1, B)
        n = Poly(z, B)
        for a in ap:
            n *= (D + a.as_poly(B))

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment index: cancels with upper')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, _x/(bi - 1) + 1), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class MeijerShiftA(Operator):
    """ Increment an upper b index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(bi - _x, _x)

    def __str__(self):
        return '<Increment upper b=%s.>' % (self._poly.all_coeffs()[1])


class MeijerShiftB(Operator):
    """ Decrement an upper a index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(1 - bi + _x, _x)

    def __str__(self):
        return '<Decrement upper a=%s.>' % (1 - self._poly.all_coeffs()[1])


class MeijerShiftC(Operator):
    """ Increment a lower b index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(-bi + _x, _x)

    def __str__(self):
        return '<Increment lower b=%s.>' % (-self._poly.all_coeffs()[1])


class MeijerShiftD(Operator):
    """ Decrement a lower a index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(bi - 1 - _x, _x)

    def __str__(self):
        return '<Decrement lower a=%s.>' % (self._poly.all_coeffs()[1] + 1)


class MeijerUnShiftA(Operator):
    """ Decrement an upper b index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bm.pop(i) - 1

        m = Poly(1, _x) * prod(Poly(b - _x, _x) for b in bm) * prod(Poly(_x - b, _x) for b in bq)

        A = Dummy('A')
        D = Poly(bi - A, A)
        n = Poly(z, A) * prod((D + 1 - a) for a in an) * prod((-D + a - 1) for a in ap)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper b index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, bi - _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Decrement upper b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftB(Operator):
    """ Increment an upper a index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = an.pop(i) + 1

        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        B = Dummy('B')
        D = Poly(B + ai - 1, B)
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment upper a index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, 1 - ai + _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment upper a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftC(Operator):
    """ Decrement a lower b index. """
    # XXX this is "essentially" the same as MeijerUnShiftA. This "essentially"
    #     can be made rigorous using the functional equation G(1/z) = G'(z),
    #     where G' denotes a G function of slightly altered parameters.
    #     However, sorting out the details seems harder than just coding it
    #     again.

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bq.pop(i) - 1

        m = Poly(1, _x)
        for b in bm:
            m *= Poly(b - _x, _x)
        for b in bq:
            m *= Poly(_x - b, _x)

        C = Dummy('C')
        D = Poly(bi + C, C)
        n = Poly(z, C)
        for a in an:
            n *= (D + 1 - a)
        for a in ap:
            n *= (-D + a - 1)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement lower b index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], C).as_expr().subs(C, _x - bi), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Decrement lower b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftD(Operator):
    """ Increment a lower a index. """
    # XXX This is essentially the same as MeijerUnShiftA.
    #     See comment at MeijerUnShiftC.

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = ap.pop(i) + 1

        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        B = Dummy('B')  # - this is the shift operator `D_I`
        D = Poly(ai - 1 - B, B)
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment lower a index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, ai - 1 - _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class ReduceOrder(Operator):
    """ Reduce Order by cancelling an upper and a lower index. """

    def __new__(cls, ai, bj):
        """ For convenience if reduction is not possible, return None. """
        ai = sympify(ai)
        bj = sympify(bj)
        n = ai - bj
        if not n.is_Integer or n < 0:
            return None
        if bj.is_integer and bj.is_nonpositive:
            return None

        expr = Operator.__new__(cls)

        p = S.One
        for k in range(n):
            p *= (_x + bj + k)/(bj + k)

        expr._poly = Poly(p, _x)
        expr._a = ai
        expr._b = bj

        return expr

    @classmethod
    def _meijer(cls, b, a, sign):
        """ Cancel b + sign*s and a + sign*s
            This is for meijer G functions. """
        b = sympify(b)
        a = sympify(a)
        n = b - a
        if n.is_negative or not n.is_Integer:
            return None

        expr = Operator.__new__(cls)

        p = S.One
        for k in range(n):
            p *= (sign*_x + a + k)

        expr._poly = Poly(p, _x)
        if sign == -1:
            expr._a = b
            expr._b = a
        else:
            expr._b = Add(1, a - 1, evaluate=False)
            expr._a = Add(1, b - 1, evaluate=False)

        return expr

    @classmethod
    def meijer_minus(cls, b, a):
        return cls._meijer(b, a, -1)

    @classmethod
    def meijer_plus(cls, a, b):
        return cls._meijer(1 - a, 1 - b, 1)

    def __str__(self):
        return '<Reduce order by cancelling upper %s with lower %s.>' % \
            (self._a, self._b)


def _reduce_order(ap, bq, gen, key):
    """ Order reduction algorithm used in Hypergeometric and Meijer G """
    ap = list(ap)
    bq = list(bq)

    ap.sort(key=key)
    bq.sort(key=key)

    nap = []
    # we will edit bq in place
    operators = []
    for a in ap:
        op = None
        for i in range(len(bq)):
            op = gen(a, bq[i])
            if op is not None:
                bq.pop(i)
                break
        if op is None:
            nap.append(a)
        else:
            operators.append(op)

    return nap, bq, operators


def reduce_order(func):
    """
    Given the hypergeometric function ``func``, find a sequence of operators to
    reduces order as much as possible.

    Explanation
    ===========

    Return (newfunc, [operators]), where applying the operators to the
    hypergeometric function newfunc yields func.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import reduce_order, Hyper_Function
    >>> reduce_order(Hyper_Function((1, 2), (3, 4)))
    (Hyper_Function((1, 2), (3, 4)), [])
    >>> reduce_order(Hyper_Function((1,), (1,)))
    (Hyper_Function((), ()), [<Reduce order by cancelling upper 1 with lower 1.>])
    >>> reduce_order(Hyper_Function((2, 4), (3, 3)))
    (Hyper_Function((2,), (3,)), [<Reduce order by cancelling
    upper 4 with lower 3.>])
    """
    nap, nbq, operators = _reduce_order(func.ap, func.bq, ReduceOrder, default_sort_key)

    return Hyper_Function(Tuple(*nap), Tuple(*nbq)), operators


def reduce_order_meijer(func):
    """
    Given the Meijer G function parameters, ``func``, find a sequence of
    operators that reduces order as much as possible.

    Return newfunc, [operators].

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import (reduce_order_meijer,
    ...                                         G_Function)
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 2]))[0]
    G_Function((4, 3), (5, 6), (3, 4), (2, 1))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 8]))[0]
    G_Function((3,), (5, 6), (3, 4), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [1, 5]))[0]
    G_Function((3,), (), (), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [5, 3]))[0]
    G_Function((), (), (), ())
    """

    nan, nbq, ops1 = _reduce_order(func.an, func.bq, ReduceOrder.meijer_plus,
                                   lambda x: default_sort_key(-x))
    nbm, nap, ops2 = _reduce_order(func.bm, func.ap, ReduceOrder.meijer_minus,
                                   default_sort_key)

    return G_Function(nan, nap, nbm, nbq), ops1 + ops2


def make_derivative_operator(M, z):
    """ Create a derivative operator, to be passed to Operator.apply. """
    def doit(C):
        r = z*C.diff(z) + C*M
        r = r.applyfunc(make_simp(z))
        return r
    return doit


def apply_operators(obj, ops, op):
    """
    Apply the list of operators ``ops`` to object ``obj``, substituting
    ``op`` for the generator.
    """
    res = obj
    for o in reversed(ops):
        res = o.apply(res, op)
    return res


def devise_plan(target, origin, z):
    """
    Devise a plan (consisting of shift and un-shift operators) to be applied
    to the hypergeometric function ``target`` to yield ``origin``.
    Returns a list of operators.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import devise_plan, Hyper_Function
    >>> from sympy.abc import z

    Nothing to do:

    >>> devise_plan(Hyper_Function((1, 2), ()), Hyper_Function((1, 2), ()), z)
    []
    >>> devise_plan(Hyper_Function((), (1, 2)), Hyper_Function((), (1, 2)), z)
    []

    Very simple plans:

    >>> devise_plan(Hyper_Function((2,), ()), Hyper_Function((1,), ()), z)
    [<Increment upper 1.>]
    >>> devise_plan(Hyper_Function((), (2,)), Hyper_Function((), (1,)), z)
    [<Increment lower index #0 of [], [1].>]

    Several buckets:

    >>> from sympy import S
    >>> devise_plan(Hyper_Function((1, S.Half), ()),
    ...             Hyper_Function((2, S('3/2')), ()), z) #doctest: +NORMALIZE_WHITESPACE
    [<Decrement upper index #0 of [3/2, 1], [].>,
    <Decrement upper index #0 of [2, 3/2], [].>]

    A slightly more complicated plan:

    >>> devise_plan(Hyper_Function((1, 3), ()), Hyper_Function((2, 2), ()), z)
    [<Increment upper 2.>, <Decrement upper index #0 of [2, 2], [].>]

    Another more complicated plan: (note that the ap have to be shifted first!)

    >>> devise_plan(Hyper_Function((1, -1), (2,)), Hyper_Function((3, -2), (4,)), z)
    [<Decrement lower 3.>, <Decrement lower 4.>,
    <Decrement upper index #1 of [-1, 2], [4].>,
    <Decrement upper index #1 of [-1, 3], [4].>, <Increment upper -2.>]
    """
    abuckets, bbuckets, nabuckets, nbbuckets = [sift(params, _mod1) for
            params in (target.ap, target.bq, origin.ap, origin.bq)]

    if len(list(abuckets.keys())) != len(list(nabuckets.keys())) or \
            len(list(bbuckets.keys())) != len(list(nbbuckets.keys())):
        raise ValueError('%s not reachable from %s' % (target, origin))

    ops = []

    def do_shifts(fro, to, inc, dec):
        ops = []
        for i in range(len(fro)):
            if to[i] - fro[i] > 0:
                sh = inc
                ch = 1
            else:
                sh = dec
                ch = -1

            while to[i] != fro[i]:
                ops += [sh(fro, i)]
                fro[i] += ch

        return ops

    def do_shifts_a(nal, nbk, al, aother, bother):
        """ Shift us from (nal, nbk) to (al, nbk). """
        return do_shifts(nal, al, lambda p, i: ShiftA(p[i]),
                         lambda p, i: UnShiftA(p + aother, nbk + bother, i, z))

    def do_shifts_b(nal, nbk, bk, aother, bother):
        """ Shift us from (nal, nbk) to (nal, bk). """
        return do_shifts(nbk, bk,
                         lambda p, i: UnShiftB(nal + aother, p + bother, i, z),
                         lambda p, i: ShiftB(p[i]))

    for r in sorted(list(abuckets.keys()) + list(bbuckets.keys()), key=default_sort_key):
        al = ()
        nal = ()
        bk = ()
        nbk = ()
        if r in abuckets:
            al = abuckets[r]
            nal = nabuckets[r]
        if r in bbuckets:
            bk = bbuckets[r]
            nbk = nbbuckets[r]
        if len(al) != len(nal) or len(bk) != len(nbk):
            raise ValueError('%s not reachable from %s' % (target, origin))

        al, nal, bk, nbk = [sorted(w, key=default_sort_key)
            for w in [al, nal, bk, nbk]]

        def others(dic, key):
            l = []
            for k in dic:
                if k != key:
                    l.extend(dic[k])
            return l
        aother = others(nabuckets, r)
        bother = others(nbbuckets, r)

        if len(al) == 0:
            # there can be no complications, just shift the bs as we please
            ops += do_shifts_b([], nbk, bk, aother, bother)
        elif len(bk) == 0:
            # there can be no complications, just shift the as as we please
            ops += do_shifts_a(nal, [], al, aother, bother)
        else:
            namax = nal[-1]
            amax = al[-1]

            if nbk[0] - namax <= 0 or bk[0] - amax <= 0:
                raise ValueError('Non-suitable parameters.')

            if namax - amax > 0:
                # we are going to shift down - first do the as, then the bs
                ops += do_shifts_a(nal, nbk, al, aother, bother)
                ops += do_shifts_b(al, nbk, bk, aother, bother)
            else:
                # we are going to shift up - first do the bs, then the as
                ops += do_shifts_b(nal, nbk, bk, aother, bother)
                ops += do_shifts_a(nal, bk, al, aother, bother)

        nabuckets[r] = al
        nbbuckets[r] = bk

    ops.reverse()
    return ops


def try_shifted_sum(func, z):
    """ Try to recognise a hypergeometric sum that starts from k > 0. """
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    if len(abuckets[S.Zero]) != 1:
        return None
    r = abuckets[S.Zero][0]
    if r <= 0:
        return None
    if S.Zero not in bbuckets:
        return None
    l = list(bbuckets[S.Zero])
    l.sort()
    k = l[0]
    if k <= 0:
        return None

    nap = list(func.ap)
    nap.remove(r)
    nbq = list(func.bq)
    nbq.remove(k)
    k -= 1
    nap = [x - k for x in nap]
    nbq = [x - k for x in nbq]

    ops = []
    for n in range(r - 1):
        ops.append(ShiftA(n + 1))
    ops.reverse()

    fac = factorial(k)/z**k
    fac *= Mul(*[rf(b, k) for b in nbq])
    fac /= Mul(*[rf(a, k) for a in nap])

    ops += [MultOperator(fac)]

    p = 0
    for n in range(k):
        m = z**n/factorial(n)
        m *= Mul(*[rf(a, n) for a in nap])
        m /= Mul(*[rf(b, n) for b in nbq])
        p += m

    return Hyper_Function(nap, nbq), ops, -p


def try_polynomial(func, z):
    """ Recognise polynomial cases. Returns None if not such a case.
        Requires order to be fully reduced. """
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    a0 = abuckets[S.Zero]
    b0 = bbuckets[S.Zero]
    a0.sort()
    b0.sort()
    al0 = [x for x in a0 if x <= 0]
    bl0 = [x for x in b0 if x <= 0]

    if bl0 and all(a < bl0[-1] for a in al0):
        return oo
    if not al0:
        return None

    a = al0[-1]
    fac = 1
    res = S.One
    for n in Tuple(*list(range(-a))):
        fac *= z
        fac /= n + 1
        fac *= Mul(*[a + n for a in func.ap])
        fac /= Mul(*[b + n for b in func.bq])
        res += fac
    return res


def try_lerchphi(func):
    """
    Try to find an expression for Hyper_Function ``func`` in terms of Lerch
    Transcendents.

    Return None if no such expression can be found.
    """
    # This is actually quite simple, and is described in Roach's paper,
    # section 18.
    # We don't need to implement the reduction to polylog here, this
    # is handled by expand_func.

    # First we need to figure out if the summation coefficient is a rational
    # function of the summation index, and construct that rational function.
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)

    paired = {}
    for key, value in abuckets.items():
        if key != 0 and key not in bbuckets:
            return None
        bvalue = bbuckets[key]
        paired[key] = (list(value), list(bvalue))
        bbuckets.pop(key, None)
    if bbuckets != {}:
        return None
    if S.Zero not in abuckets:
        return None
    aints, bints = paired[S.Zero]
    # Account for the additional n! in denominator
    paired[S.Zero] = (aints, bints + [1])

    t = Dummy('t')
    numer = S.One
    denom = S.One
    for key, (avalue, bvalue) in paired.items():
        if len(avalue) != len(bvalue):
            return None
        # Note that since order has been reduced fully, all the b are
        # bigger than all the a they differ from by an integer. In particular
        # if there are any negative b left, this function is not well-defined.
        for a, b in zip(avalue, bvalue):
            if (a - b).is_positive:
                k = a - b
                numer *= rf(b + t, k)
                denom *= rf(b, k)
            else:
                k = b - a
                numer *= rf(a, k)
                denom *= rf(a + t, k)

    # Now do a partial fraction decomposition.
    # We assemble two structures: a list monomials of pairs (a, b) representing
    # a*t**b (b a non-negative integer), and a dict terms, where
    # terms[a] = [(b, c)] means that there is a term b/(t-a)**c.
    part = apart(numer/denom, t)
    args = Add.make_args(part)
    monomials = []
    terms = {}
    for arg in args:
        numer, denom = arg.as_numer_denom()
        if not denom.has(t):
            p = Poly(numer, t)
            if not p.is_monomial:
                raise TypeError("p should be monomial")
            ((b, ), a) = p.LT()
            monomials += [(a/denom, b)]
            continue
        if numer.has(t):
            raise NotImplementedError('Need partial fraction decomposition'
                                      ' with linear denominators')
        indep, [dep] = denom.as_coeff_mul(t)
        n = 1
        if dep.is_Pow:
            n = dep.exp
            dep = dep.base
        if dep == t:
            a = 0
        elif dep.is_Add:
            a, tmp = dep.as_independent(t)
            b = 1
            if tmp != t:
                b, _ = tmp.as_independent(t)
            if dep != b*t + a:
                raise NotImplementedError('unrecognised form %s' % dep)
            a /= b
            indep *= b**n
        else:
            raise NotImplementedError('unrecognised form of partial fraction')
        terms.setdefault(a, []).append((numer/indep, n))

    # Now that we have this information, assemble our formula. All the
    # monomials yield rational functions and go into one basis element.
    # The terms[a] are related by differentiation. If the largest exponent is
    # n, we need lerchphi(z, k, a) for k = 1, 2, ..., n.
    # deriv maps a basis to its derivative, expressed as a C(z)-linear
    # combination of other basis elements.
    deriv = {}
    coeffs = {}
    z = Dummy('z')
    monomials.sort(key=lambda x: x[1])
    mon = {0: 1/(1 - z)}
    if monomials:
        for k in range(monomials[-1][1]):
            mon[k + 1] = z*mon[k].diff(z)
    for a, n in monomials:
        coeffs.setdefault(S.One, []).append(a*mon[n])
    for a, l in terms.items():
        for c, k in l:
            coeffs.setdefault(lerchphi(z, k, a), []).append(c)
        l.sort(key=lambda x: x[1])
        for k in range(2, l[-1][1] + 1):
            deriv[lerchphi(z, k, a)] = [(-a, lerchphi(z, k, a)),
                                        (1, lerchphi(z, k - 1, a))]
        deriv[lerchphi(z, 1, a)] = [(-a, lerchphi(z, 1, a)),
                                    (1/(1 - z), S.One)]
    trans = {}
    for n, b in enumerate([S.One] + list(deriv.keys())):
        trans[b] = n
    basis = [expand_func(b) for (b, _) in sorted(trans.items(),
                                                 key=lambda x:x[1])]
    B = Matrix(basis)
    C = Matrix([[0]*len(B)])
    for b, c in coeffs.items():
        C[trans[b]] = Add(*c)
    M = zeros(len(B))
    for b, l in deriv.items():
        for c, b2 in l:
            M[trans[b], trans[b2]] = c
    return Formula(func, z, None, [], B, C, M)


def build_hypergeometric_formula(func):
    """
    Create a formula object representing the hypergeometric function ``func``.

    """
    # We know that no `ap` are negative integers, otherwise "detect poly"
    # would have kicked in. However, `ap` could be empty. In this case we can
    # use a different basis.
    # I'm not aware of a basis that works in all cases.
    z = Dummy('z')
    if func.ap:
        afactors = [_x + a for a in func.ap]
        bfactors = [_x + b - 1 for b in func.bq]
        expr = _x*Mul(*bfactors) - z*Mul(*afactors)
        poly = Poly(expr, _x)
        n = poly.degree()
        basis = []
        M = zeros(n)
        for k in range(n):
            a = func.ap[0] + k
            basis += [hyper([a] + list(func.ap[1:]), func.bq, z)]
            if k < n - 1:
                M[k, k] = -a
                M[k, k + 1] = a
        B = Matrix(basis)
        C = Matrix([[1] + [0]*(n - 1)])
        derivs = [eye(n)]
        for k in range(n):
            derivs.append(M*derivs[k])
        l = poly.all_coeffs()
        l.reverse()
        res = [0]*n
        for k, c in enumerate(l):
            for r, d in enumerate(C*derivs[k]):
                res[r] += c*d
        for k, c in enumerate(res):
            M[n - 1, k] = -c/derivs[n - 1][0, n - 1]/poly.all_coeffs()[0]
        return Formula(func, z, None, [], B, C, M)
    else:
        # Since there are no `ap`, none of the `bq` can be non-positive
        # integers.
        basis = []
        bq = list(func.bq[:])
        for i in range(len(bq)):
            basis += [hyper([], bq, z)]
            bq[i] += 1
        basis += [hyper([], bq, z)]
        B = Matrix(basis)
        n = len(B)
        C = Matrix([[1] + [0]*(n - 1)])
        M = zeros(n)
        M[0, n - 1] = z/Mul(*func.bq)
        for k in range(1, n):
            M[k, k - 1] = func.bq[k - 1]
            M[k, k] = -func.bq[k - 1]
        return Formula(func, z, None, [], B, C, M)


def hyperexpand_special(ap, bq, z):
    """
    Try to find a closed-form expression for hyper(ap, bq, z), where ``z``
    is supposed to be a "special" value, e.g. 1.

    This function tries various of the classical summation formulae
    (Gauss, Saalschuetz, etc).
    """
    # This code is very ad-hoc. There are many clever algorithms
    # (notably Zeilberger's) related to this problem.
    # For now we just want a few simple cases to work.
    p, q = len(ap), len(bq)
    z_ = z
    z = unpolarify(z)
    if z == 0:
        return S.One
    from sympy.simplify.simplify import simplify
    if p == 2 and q == 1:
        # 2F1
        a, b, c = ap + bq
        if z == 1:
            # Gauss
            return gamma(c - a - b)*gamma(c)/gamma(c - a)/gamma(c - b)
        if z == -1 and simplify(b - a + c) == 1:
            b, a = a, b
        if z == -1 and simplify(a - b + c) == 1:
            # Kummer
            if b.is_integer and b.is_negative:
                return 2*cos(pi*b/2)*gamma(-b)*gamma(b - a + 1) \
                    /gamma(-b/2)/gamma(b/2 - a + 1)
            else:
                return gamma(b/2 + 1)*gamma(b - a + 1) \
                    /gamma(b + 1)/gamma(b/2 - a + 1)
    # TODO tons of more formulae
    #      investigate what algorithms exist
    return hyper(ap, bq, z_)

_collection = None


def _hyperexpand(func, z, ops0=[], z0=Dummy('z0'), premult=1, prem=0,
                 rewrite='default'):
    """
    Try to find an expression for the hypergeometric function ``func``.

    Explanation
    ===========

    The result is expressed in terms of a dummy variable ``z0``. Then it
    is multiplied by ``premult``. Then ``ops0`` is applied.
    ``premult`` must be a*z**prem for some a independent of ``z``.
    """

    if z.is_zero:
        return S.One

    from sympy.simplify.simplify import simplify

    z = polarify(z, subs=False)
    if rewrite == 'default':
        rewrite = 'nonrepsmall'

    def carryout_plan(f, ops):
        C = apply_operators(f.C.subs(f.z, z0), ops,
                            make_derivative_operator(f.M.subs(f.z, z0), z0))
        C = apply_operators(C, ops0,
                            make_derivative_operator(f.M.subs(f.z, z0)
                                         + prem*eye(f.M.shape[0]), z0))

        if premult == 1:
            C = C.applyfunc(make_simp(z0))
        r = reduce(lambda s,m: s+m[0]*m[1], zip(C, f.B.subs(f.z, z0)), S.Zero)*premult
        res = r.subs(z0, z)
        if rewrite:
            res = res.rewrite(rewrite)
        return res

    # TODO
    # The following would be possible:
    # *) PFD Duplication (see Kelly Roach's paper)
    # *) In a similar spirit, try_lerchphi() can be generalised considerably.

    global _collection
    if _collection is None:
        _collection = FormulaCollection()

    debug('Trying to expand hypergeometric function ', func)

    # First reduce order as much as possible.
    func, ops = reduce_order(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')

    # Now try polynomial cases
    res = try_polynomial(func, z0)
    if res is not None:
        debug('  Recognised polynomial.')
        p = apply_operators(res, ops, lambda f: z0*f.diff(z0))
        p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
        return unpolarify(simplify(p).subs(z0, z))

    # Try to recognise a shifted sum.
    p = S.Zero
    res = try_shifted_sum(func, z0)
    if res is not None:
        func, nops, p = res
        debug('  Recognised shifted sum, reduced order to ', func)
        ops += nops

    # apply the plan for poly
    p = apply_operators(p, ops, lambda f: z0*f.diff(z0))
    p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
    p = simplify(p).subs(z0, z)

    # Try special expansions early.
    if unpolarify(z) in [1, -1] and (len(func.ap), len(func.bq)) == (2, 1):
        f = build_hypergeometric_formula(func)
        r = carryout_plan(f, ops).replace(hyper, hyperexpand_special)
        if not r.has(hyper):
            return r + p

    # Try to find a formula in our collection
    formula = _collection.lookup_origin(func)

    # Now try a lerch phi formula
    if formula is None:
        formula = try_lerchphi(func)

    if formula is None:
        debug('  Could not find an origin. ',
              'Will return answer in terms of '
              'simpler hypergeometric functions.')
        formula = build_hypergeometric_formula(func)

    debug('  Found an origin: ', formula.closed_form, ' ', formula.func)

    # We need to find the operators that convert formula into func.
    ops += devise_plan(func, formula.func, z0)

    # Now carry out the plan.
    r = carryout_plan(formula, ops) + p

    return powdenest(r, polar=True).replace(hyper, hyperexpand_special)


def devise_plan_meijer(fro, to, z):
    """
    Find operators to convert G-function ``fro`` into G-function ``to``.

    Explanation
    ===========

    It is assumed that ``fro`` and ``to`` have the same signatures, and that in fact
    any corresponding pair of parameters differs by integers, and a direct path
    is possible. I.e. if there are parameters a1 b1 c1  and a2 b2 c2 it is
    assumed that a1 can be shifted to a2, etc. The only thing this routine
    determines is the order of shifts to apply, nothing clever will be tried.
    It is also assumed that ``fro`` is suitable.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import (devise_plan_meijer,
    ...                                         G_Function)
    >>> from sympy.abc import z

    Empty plan:

    >>> devise_plan_meijer(G_Function([1], [2], [3], [4]),
    ...                    G_Function([1], [2], [3], [4]), z)
    []

    Very simple plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([1], [], [], []), z)
    [<Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([-1], [], [], []), z)
    [<Decrement upper a=0.>]
    >>> devise_plan_meijer(G_Function([], [1], [], []),
    ...                    G_Function([], [2], [], []), z)
    [<Increment lower a index #0 of [], [1], [], [].>]

    Slightly more complicated plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([2], [], [], []), z)
    [<Increment upper a index #0 of [1], [], [], [].>,
    <Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([-1], [], [1], []), z)
    [<Increment upper b=0.>, <Decrement upper a=0.>]

    Order matters:

    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([1], [], [1], []), z)
    [<Increment upper a index #0 of [0], [], [1], [].>, <Increment upper b=0.>]
    """
    # TODO for now, we use the following simple heuristic: inverse-shift
    #      when possible, shift otherwise. Give up if we cannot make progress.

    def try_shift(f, t, shifter, diff, counter):
        """ Try to apply ``shifter`` in order to bring some element in ``f``
            nearer to its counterpart in ``to``. ``diff`` is +/- 1 and
            determines the effect of ``shifter``. Counter is a list of elements
            blocking the shift.

            Return an operator if change was possible, else None.
        """
        for idx, (a, b) in enumerate(zip(f, t)):
            if (
                (a - b).is_integer and (b - a)/diff > 0 and
                    all(a != x for x in counter)):
                sh = shifter(idx)
                f[idx] += diff
                return sh
    fan = list(fro.an)
    fap = list(fro.ap)
    fbm = list(fro.bm)
    fbq = list(fro.bq)
    ops = []
    change = True
    while change:
        change = False
        op = try_shift(fan, to.an,
                       lambda i: MeijerUnShiftB(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap,
                       lambda i: MeijerUnShiftD(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm,
                       lambda i: MeijerUnShiftA(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq,
                       lambda i: MeijerUnShiftC(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fan, to.an, lambda i: MeijerShiftB(fan[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap, lambda i: MeijerShiftD(fap[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm, lambda i: MeijerShiftA(fbm[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq, lambda i: MeijerShiftC(fbq[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
    if fan != list(to.an) or fap != list(to.ap) or fbm != list(to.bm) or \
            fbq != list(to.bq):
        raise NotImplementedError('Could not devise plan.')
    ops.reverse()
    return ops

_meijercollection = None


def _meijergexpand(func, z0, allow_hyper=False, rewrite='default',
                   place=None):
    """
    Try to find an expression for the Meijer G function specified
    by the G_Function ``func``. If ``allow_hyper`` is True, then returning
    an expression in terms of hypergeometric functions is allowed.

    Currently this just does Slater's theorem.
    If expansions exist both at zero and at infinity, ``place``
    can be set to ``0`` or ``zoo`` for the preferred choice.
    """
    global _meijercollection
    if _meijercollection is None:
        _meijercollection = MeijerFormulaCollection()
    if rewrite == 'default':
        rewrite = None

    func0 = func
    debug('Try to expand Meijer G function corresponding to ', func)

    # We will play games with analytic continuation - rather use a fresh symbol
    z = Dummy('z')

    func, ops = reduce_order_meijer(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')

    # Try to find a direct formula
    f = _meijercollection.lookup_origin(func)
    if f is not None:
        debug('  Found a Meijer G formula: ', f.func)
        ops += devise_plan_meijer(f.func, func, z)

        # Now carry out the plan.
        C = apply_operators(f.C.subs(f.z, z), ops,
                            make_derivative_operator(f.M.subs(f.z, z), z))

        C = C.applyfunc(make_simp(z))
        r = C*f.B.subs(f.z, z)
        r = r[0].subs(z, z0)
        return powdenest(r, polar=True)

    debug("  Could not find a direct formula. Trying Slater's theorem.")

    # TODO the following would be possible:
    # *) Paired Index Theorems
    # *) PFD Duplication
    #    (See Kelly Roach's paper for details on either.)
    #
    # TODO Also, we tend to create combinations of gamma functions that can be
    #      simplified.

    def can_do(pbm, pap):
        """ Test if slater applies. """
        for i in pbm:
            if len(pbm[i]) > 1:
                l = 0
                if i in pap:
                    l = len(pap[i])
                if l + 1 < len(pbm[i]):
                    return False
        return True

    def do_slater(an, bm, ap, bq, z, zfinal):
        # zfinal is the value that will eventually be substituted for z.
        # We pass it to _hyperexpand to improve performance.
        func = G_Function(an, bm, ap, bq)
        _, pbm, pap, _ = func.compute_buckets()
        if not can_do(pbm, pap):
            return S.Zero, False

        cond = len(an) + len(ap) < len(bm) + len(bq)
        if len(an) + len(ap) == len(bm) + len(bq):
            cond = abs(z) < 1
        if cond is False:
            return S.Zero, False

        res = S.Zero
        for m in pbm:
            if len(pbm[m]) == 1:
                bh = pbm[m][0]
                fac = 1
                bo = list(bm)
                bo.remove(bh)
                for bj in bo:
                    fac *= gamma(bj - bh)
                for aj in an:
                    fac *= gamma(1 + bh - aj)
                for bj in bq:
                    fac /= gamma(1 + bh - bj)
                for aj in ap:
                    fac /= gamma(aj - bh)
                nap = [1 + bh - a for a in list(an) + list(ap)]
                nbq = [1 + bh - b for b in list(bo) + list(bq)]

                k = polar_lift(S.NegativeOne**(len(ap) - len(bm)))
                harg = k*zfinal
                # NOTE even though k "is" +-1, this has to be t/k instead of
                #      t*k ... we are using polar numbers for consistency!
                premult = (t/k)**bh
                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops,
                                   t, premult, bh, rewrite=None)
                res += fac * hyp
            else:
                b_ = pbm[m][0]
                ki = [bi - b_ for bi in pbm[m][1:]]
                u = len(ki)
                li = [ai - b_ for ai in pap[m][:u + 1]]
                bo = list(bm)
                for b in pbm[m]:
                    bo.remove(b)
                ao = list(ap)
                for a in pap[m][:u]:
                    ao.remove(a)
                lu = li[-1]
                di = [l - k for (l, k) in zip(li, ki)]

                # We first work out the integrand:
                s = Dummy('s')
                integrand = z**s
                for b in bm:
                    if not Mod(b, 1) and b.is_Number:
                        b = int(round(b))
                    integrand *= gamma(b - s)
                for a in an:
                    integrand *= gamma(1 - a + s)
                for b in bq:
                    integrand /= gamma(1 - b + s)
                for a in ap:
                    integrand /= gamma(a - s)

                # Now sum the finitely many residues:
                # XXX This speeds up some cases - is it a good idea?
                integrand = expand_func(integrand)
                for r in range(int(round(lu))):
                    resid = residue(integrand, s, b_ + r)
                    resid = apply_operators(resid, ops, lambda f: z*f.diff(z))
                    res -= resid

                # Now the hypergeometric term.
                au = b_ + lu
                k = polar_lift(S.NegativeOne**(len(ao) + len(bo) + 1))
                harg = k*zfinal
                premult = (t/k)**au
                nap = [1 + au - a for a in list(an) + list(ap)] + [1]
                nbq = [1 + au - b for b in list(bm) + list(bq)]

                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops,
                                   t, premult, au, rewrite=None)

                C = S.NegativeOne**(lu)/factorial(lu)
                for i in range(u):
                    C *= S.NegativeOne**di[i]/rf(lu - li[i] + 1, di[i])
                for a in an:
                    C *= gamma(1 - a + au)
                for b in bo:
                    C *= gamma(b - au)
                for a in ao:
                    C /= gamma(a - au)
                for b in bq:
                    C /= gamma(1 - b + au)

                res += C*hyp

        return res, cond

    t = Dummy('t')
    slater1, cond1 = do_slater(func.an, func.bm, func.ap, func.bq, z, z0)

    def tr(l):
        return [1 - x for x in l]

    for op in ops:
        op._poly = Poly(op._poly.subs({z: 1/t, _x: -_x}), _x)
    slater2, cond2 = do_slater(tr(func.bm), tr(func.an), tr(func.bq), tr(func.ap),
                               t, 1/z0)

    slater1 = powdenest(slater1.subs(z, z0), polar=True)
    slater2 = powdenest(slater2.subs(t, 1/z0), polar=True)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(t, 1/z)

    m = func(z)
    if m.delta > 0 or \
        (m.delta == 0 and len(m.ap) == len(m.bq) and
            (re(m.nu) < -1) is not False and polar_lift(z0) == polar_lift(1)):
        # The condition delta > 0 means that the convergence region is
        # connected. Any expression we find can be continued analytically
        # to the entire convergence region.
        # The conditions delta==0, p==q, re(nu) < -1 imply that G is continuous
        # on the positive reals, so the values at z=1 agree.
        if cond1 is not False:
            cond1 = True
        if cond2 is not False:
            cond2 = True

    if cond1 is True:
        slater1 = slater1.rewrite(rewrite or 'nonrep')
    else:
        slater1 = slater1.rewrite(rewrite or 'nonrepsmall')
    if cond2 is True:
        slater2 = slater2.rewrite(rewrite or 'nonrep')
    else:
        slater2 = slater2.rewrite(rewrite or 'nonrepsmall')

    if cond1 is not False and cond2 is not False:
        # If one condition is False, there is no choice.
        if place == 0:
            cond2 = False
        if place == zoo:
            cond1 = False

    if not isinstance(cond1, bool):
        cond1 = cond1.subs(z, z0)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(z, z0)

    def weight(expr, cond):
        if cond is True:
            c0 = 0
        elif cond is False:
            c0 = 1
        else:
            c0 = 2
        if expr.has(oo, zoo, -oo, nan):
            # XXX this actually should not happen, but consider
            # S('meijerg(((0, -1/2, 0, -1/2, 1/2), ()), ((0,),
            #   (-1/2, -1/2, -1/2, -1)), exp_polar(I*pi))/4')
            c0 = 3
        return (c0, expr.count(hyper), expr.count_ops())

    w1 = weight(slater1, cond1)
    w2 = weight(slater2, cond2)
    if min(w1, w2) <= (0, 1, oo):
        if w1 < w2:
            return slater1
        else:
            return slater2
    if max(w1[0], w2[0]) <= 1 and max(w1[1], w2[1]) <= 1:
        return Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))

    # We couldn't find an expression without hypergeometric functions.
    # TODO it would be helpful to give conditions under which the integral
    #      is known to diverge.
    r = Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    if r.has(hyper) and not allow_hyper:
        debug('  Could express using hypergeometric functions, '
              'but not allowed.')
    if not r.has(hyper) or allow_hyper:
        return r

    return func0(z0)


def hyperexpand(f, allow_hyper=False, rewrite='default', place=None):
    """
    Expand hypergeometric functions. If allow_hyper is True, allow partial
    simplification (that is a result different from input,
    but still containing hypergeometric functions).

    If a G-function has expansions both at zero and at infinity,
    ``place`` can be set to ``0`` or ``zoo`` to indicate the
    preferred choice.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import hyperexpand
    >>> from sympy.functions import hyper
    >>> from sympy.abc import z
    >>> hyperexpand(hyper([], [], z))
    exp(z)

    Non-hyperegeometric parts of the expression and hypergeometric expressions
    that are not recognised are left unchanged:

    >>> hyperexpand(1 + hyper([1, 1, 1], [], z))
    hyper((1, 1, 1), (), z) + 1
    """
    f = sympify(f)

    def do_replace(ap, bq, z):
        r = _hyperexpand(Hyper_Function(ap, bq), z, rewrite=rewrite)
        if r is None:
            return hyper(ap, bq, z)
        else:
            return r

    def do_meijer(ap, bq, z):
        r = _meijergexpand(G_Function(ap[0], ap[1], bq[0], bq[1]), z,
                   allow_hyper, rewrite=rewrite, place=place)
        if not r.has(nan, zoo, oo, -oo):
            return r
    return f.replace(hyper, do_replace).replace(meijerg, do_meijer)