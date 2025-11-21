
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\attributes.py ===
# orm/attributes.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Defines instrumentation for class attributes and their interaction
with instances.

This module is usually not directly visible to user applications, but
defines a large part of the ORM's interactivity.


"""

from __future__ import annotations

import dataclasses
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import collections
from . import exc as orm_exc
from . import interfaces
from ._typing import insp_is_aliased_class
from .base import _DeclarativeMapped
from .base import ATTR_EMPTY
from .base import ATTR_WAS_SET
from .base import CALLABLES_OK
from .base import DEFERRED_HISTORY_LOAD
from .base import INCLUDE_PENDING_MUTATIONS  # noqa
from .base import INIT_OK
from .base import instance_dict as instance_dict
from .base import instance_state as instance_state
from .base import instance_str
from .base import LOAD_AGAINST_COMMITTED
from .base import LoaderCallableStatus
from .base import manager_of_class as manager_of_class
from .base import Mapped as Mapped  # noqa
from .base import NEVER_SET  # noqa
from .base import NO_AUTOFLUSH
from .base import NO_CHANGE  # noqa
from .base import NO_KEY
from .base import NO_RAISE
from .base import NO_VALUE
from .base import NON_PERSISTENT_OK  # noqa
from .base import opt_manager_of_class as opt_manager_of_class
from .base import PASSIVE_CLASS_MISMATCH  # noqa
from .base import PASSIVE_NO_FETCH
from .base import PASSIVE_NO_FETCH_RELATED  # noqa
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import PASSIVE_ONLY_PERSISTENT
from .base import PASSIVE_RETURN_NO_VALUE
from .base import PassiveFlag
from .base import RELATED_OBJECT_OK  # noqa
from .base import SQL_OK  # noqa
from .base import SQLORMExpression
from .base import state_str
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
from ..sql.visitors import _TraverseInternalsType
from ..sql.visitors import InternalTraversal
from ..util.typing import Literal
from ..util.typing import Self
from ..util.typing import TypeGuard

if TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _ExternalEntityType
    from ._typing import _InstanceDict
    from ._typing import _InternalEntityType
    from ._typing import _LoaderCallable
    from ._typing import _O
    from .collections import _AdaptedCollectionProtocol
    from .collections import CollectionAdapter
    from .interfaces import MapperProperty
    from .relationships import RelationshipProperty
    from .state import InstanceState
    from .util import AliasedInsp
    from .writeonly import WriteOnlyAttributeImpl
    from ..event.base import _Dispatch
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _DMLColumnArgument
    from ..sql._typing import _InfoType
    from ..sql._typing import _PropagateAttrsType
    from ..sql.annotation import _AnnotationDict
    from ..sql.elements import ColumnElement
    from ..sql.elements import Label
    from ..sql.operators import OperatorType
    from ..sql.selectable import FromClause


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", bound=Any, covariant=True)


_AllPendingType = Sequence[
    Tuple[Optional["InstanceState[Any]"], Optional[object]]
]


_UNKNOWN_ATTR_KEY = object()


@inspection._self_inspects
class QueryableAttribute(
    _DeclarativeMapped[_T_co],
    SQLORMExpression[_T_co],
    interfaces.InspectionAttr,
    interfaces.PropComparator[_T_co],
    roles.JoinTargetRole,
    roles.OnClauseRole,
    sql_base.Immutable,
    cache_key.SlotsMemoizedHasCacheKey,
    util.MemoizedSlots,
    EventTarget,
):
    """Base class for :term:`descriptor` objects that intercept
    attribute events on behalf of a :class:`.MapperProperty`
    object.  The actual :class:`.MapperProperty` is accessible
    via the :attr:`.QueryableAttribute.property`
    attribute.


    .. seealso::

        :class:`.InstrumentedAttribute`

        :class:`.MapperProperty`

        :attr:`_orm.Mapper.all_orm_descriptors`

        :attr:`_orm.Mapper.attrs`
    """

    __slots__ = (
        "class_",
        "key",
        "impl",
        "comparator",
        "property",
        "parent",
        "expression",
        "_of_type",
        "_extra_criteria",
        "_slots_dispatch",
        "_propagate_attrs",
        "_doc",
    )

    is_attribute = True

    dispatch: dispatcher[QueryableAttribute[_T_co]]

    class_: _ExternalEntityType[Any]
    key: str
    parententity: _InternalEntityType[Any]
    impl: AttributeImpl
    comparator: interfaces.PropComparator[_T_co]
    _of_type: Optional[_InternalEntityType[Any]]
    _extra_criteria: Tuple[ColumnElement[bool], ...]
    _doc: Optional[str]

    # PropComparator has a __visit_name__ to participate within
    # traversals.   Disambiguate the attribute vs. a comparator.
    __visit_name__ = "orm_instrumented_attribute"

    def __init__(
        self,
        class_: _ExternalEntityType[_O],
        key: str,
        parententity: _InternalEntityType[_O],
        comparator: interfaces.PropComparator[_T_co],
        impl: Optional[AttributeImpl] = None,
        of_type: Optional[_InternalEntityType[Any]] = None,
        extra_criteria: Tuple[ColumnElement[bool], ...] = (),
    ):
        self.class_ = class_
        self.key = key

        self._parententity = self.parent = parententity

        # this attribute is non-None after mappers are set up, however in the
        # interim class manager setup, there's a check for None to see if it
        # needs to be populated, so we assign None here leaving the attribute
        # in a temporarily not-type-correct state
        self.impl = impl  # type: ignore

        assert comparator is not None
        self.comparator = comparator
        self._of_type = of_type
        self._extra_criteria = extra_criteria
        self._doc = None

        manager = opt_manager_of_class(class_)
        # manager is None in the case of AliasedClass
        if manager:
            # propagate existing event listeners from
            # immediate superclass
            for base in manager._bases:
                if key in base:
                    self.dispatch._update(base[key].dispatch)
                    if base[key].dispatch._active_history:
                        self.dispatch._active_history = True  # type: ignore

    _cache_key_traversal = [
        ("key", visitors.ExtendedInternalTraversal.dp_string),
        ("_parententity", visitors.ExtendedInternalTraversal.dp_multi),
        ("_of_type", visitors.ExtendedInternalTraversal.dp_multi),
        ("_extra_criteria", visitors.InternalTraversal.dp_clauseelement_list),
    ]

    def __reduce__(self) -> Any:
        # this method is only used in terms of the
        # sqlalchemy.ext.serializer extension
        return (
            _queryable_attribute_unreduce,
            (
                self.key,
                self._parententity.mapper.class_,
                self._parententity,
                self._parententity.entity,
            ),
        )

    @property
    def _impl_uses_objects(self) -> bool:
        return self.impl.uses_objects

    def get_history(
        self, instance: Any, passive: PassiveFlag = PASSIVE_OFF
    ) -> History:
        return self.impl.get_history(
            instance_state(instance), instance_dict(instance), passive
        )

    @property
    def info(self) -> _InfoType:
        """Return the 'info' dictionary for the underlying SQL element.

        The behavior here is as follows:

        * If the attribute is a column-mapped property, i.e.
          :class:`.ColumnProperty`, which is mapped directly
          to a schema-level :class:`_schema.Column` object, this attribute
          will return the :attr:`.SchemaItem.info` dictionary associated
          with the core-level :class:`_schema.Column` object.

        * If the attribute is a :class:`.ColumnProperty` but is mapped to
          any other kind of SQL expression other than a
          :class:`_schema.Column`,
          the attribute will refer to the :attr:`.MapperProperty.info`
          dictionary associated directly with the :class:`.ColumnProperty`,
          assuming the SQL expression itself does not have its own ``.info``
          attribute (which should be the case, unless a user-defined SQL
          construct has defined one).

        * If the attribute refers to any other kind of
          :class:`.MapperProperty`, including :class:`.Relationship`,
          the attribute will refer to the :attr:`.MapperProperty.info`
          dictionary associated with that :class:`.MapperProperty`.

        * To access the :attr:`.MapperProperty.info` dictionary of the
          :class:`.MapperProperty` unconditionally, including for a
          :class:`.ColumnProperty` that's associated directly with a
          :class:`_schema.Column`, the attribute can be referred to using
          :attr:`.QueryableAttribute.property` attribute, as
          ``MyClass.someattribute.property.info``.

        .. seealso::

            :attr:`.SchemaItem.info`

            :attr:`.MapperProperty.info`

        """
        return self.comparator.info

    parent: _InternalEntityType[Any]
    """Return an inspection instance representing the parent.

    This will be either an instance of :class:`_orm.Mapper`
    or :class:`.AliasedInsp`, depending upon the nature
    of the parent entity which this attribute is associated
    with.

    """

    expression: ColumnElement[_T_co]
    """The SQL expression object represented by this
    :class:`.QueryableAttribute`.

    This will typically be an instance of a :class:`_sql.ColumnElement`
    subclass representing a column expression.

    """

    def _memoized_attr_expression(self) -> ColumnElement[_T]:
        annotations: _AnnotationDict

        # applies only to Proxy() as used by hybrid.
        # currently is an exception to typing rather than feeding through
        # non-string keys.
        # ideally Proxy() would have a separate set of methods to deal
        # with this case.
        entity_namespace = self._entity_namespace
        assert isinstance(entity_namespace, HasCacheKey)

        if self.key is _UNKNOWN_ATTR_KEY:
            annotations = {"entity_namespace": entity_namespace}
        else:
            annotations = {
                "proxy_key": self.key,
                "proxy_owner": self._parententity,
                "entity_namespace": entity_namespace,
            }

        ce = self.comparator.__clause_element__()
        try:
            if TYPE_CHECKING:
                assert isinstance(ce, ColumnElement)
            anno = ce._annotate
        except AttributeError as ae:
            raise exc.InvalidRequestError(
                'When interpreting attribute "%s" as a SQL expression, '
                "expected __clause_element__() to return "
                "a ClauseElement object, got: %r" % (self, ce)
            ) from ae
        else:
            return anno(annotations)

    def _memoized_attr__propagate_attrs(self) -> _PropagateAttrsType:
        # this suits the case in coercions where we don't actually
        # call ``__clause_element__()`` but still need to get
        # resolved._propagate_attrs.  See #6558.
        return util.immutabledict(
            {
                "compile_state_plugin": "orm",
                "plugin_subject": self._parentmapper,
            }
        )

    @property
    def _entity_namespace(self) -> _InternalEntityType[Any]:
        return self._parententity

    @property
    def _annotations(self) -> _AnnotationDict:
        return self.__clause_element__()._annotations

    def __clause_element__(self) -> ColumnElement[_T_co]:
        return self.expression

    @property
    def _from_objects(self) -> List[FromClause]:
        return self.expression._from_objects

    def _bulk_update_tuples(
        self, value: Any
    ) -> Sequence[Tuple[_DMLColumnArgument, Any]]:
        """Return setter tuples for a bulk UPDATE."""

        return self.comparator._bulk_update_tuples(value)

    def adapt_to_entity(self, adapt_to_entity: AliasedInsp[Any]) -> Self:
        assert not self._of_type
        return self.__class__(
            adapt_to_entity.entity,
            self.key,
            impl=self.impl,
            comparator=self.comparator.adapt_to_entity(adapt_to_entity),
            parententity=adapt_to_entity,
        )

    def of_type(self, entity: _EntityType[_T]) -> QueryableAttribute[_T]:
        return QueryableAttribute(
            self.class_,
            self.key,
            self._parententity,
            impl=self.impl,
            comparator=self.comparator.of_type(entity),
            of_type=inspection.inspect(entity),
            extra_criteria=self._extra_criteria,
        )

    def and_(
        self, *clauses: _ColumnExpressionArgument[bool]
    ) -> QueryableAttribute[bool]:
        if TYPE_CHECKING:
            assert isinstance(self.comparator, RelationshipProperty.Comparator)

        exprs = tuple(
            coercions.expect(roles.WhereHavingRole, clause)
            for clause in util.coerce_generator_arg(clauses)
        )

        return QueryableAttribute(
            self.class_,
            self.key,
            self._parententity,
            impl=self.impl,
            comparator=self.comparator.and_(*exprs),
            of_type=self._of_type,
            extra_criteria=self._extra_criteria + exprs,
        )

    def _clone(self, **kw: Any) -> QueryableAttribute[_T]:
        return QueryableAttribute(
            self.class_,
            self.key,
            self._parententity,
            impl=self.impl,
            comparator=self.comparator,
            of_type=self._of_type,
            extra_criteria=self._extra_criteria,
        )

    def label(self, name: Optional[str]) -> Label[_T_co]:
        return self.__clause_element__().label(name)

    def operate(
        self, op: OperatorType, *other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        return op(self.comparator, *other, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

    def reverse_operate(
        self, op: OperatorType, other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        return op(other, self.comparator, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

    def hasparent(
        self, state: InstanceState[Any], optimistic: bool = False
    ) -> bool:
        return self.impl.hasparent(state, optimistic=optimistic) is not False

    def __getattr__(self, key: str) -> Any:
        try:
            return util.MemoizedSlots.__getattr__(self, key)
        except AttributeError:
            pass

        try:
            return getattr(self.comparator, key)
        except AttributeError as err:
            raise AttributeError(
                "Neither %r object nor %r object associated with %s "
                "has an attribute %r"
                % (
                    type(self).__name__,
                    type(self.comparator).__name__,
                    self,
                    key,
                )
            ) from err

    def __str__(self) -> str:
        return f"{self.class_.__name__}.{self.key}"

    def _memoized_attr_property(self) -> Optional[MapperProperty[Any]]:
        return self.comparator.property


def _queryable_attribute_unreduce(
    key: str,
    mapped_class: Type[_O],
    parententity: _InternalEntityType[_O],
    entity: _ExternalEntityType[Any],
) -> Any:
    # this method is only used in terms of the
    # sqlalchemy.ext.serializer extension
    if insp_is_aliased_class(parententity):
        return entity._get_from_serialized(key, mapped_class, parententity)
    else:
        return getattr(entity, key)


class InstrumentedAttribute(QueryableAttribute[_T_co]):
    """Class bound instrumented attribute which adds basic
    :term:`descriptor` methods.

    See :class:`.QueryableAttribute` for a description of most features.


    """

    __slots__ = ()

    inherit_cache = True
    """:meta private:"""

    # hack to make __doc__ writeable on instances of
    # InstrumentedAttribute, while still keeping classlevel
    # __doc__ correct

    @util.rw_hybridproperty
    def __doc__(self) -> Optional[str]:
        return self._doc

    @__doc__.setter  # type: ignore
    def __doc__(self, value: Optional[str]) -> None:
        self._doc = value

    @__doc__.classlevel  # type: ignore
    def __doc__(cls) -> Optional[str]:
        return super().__doc__

    def __set__(self, instance: object, value: Any) -> None:
        self.impl.set(
            instance_state(instance), instance_dict(instance), value, None
        )

    def __delete__(self, instance: object) -> None:
        self.impl.delete(instance_state(instance), instance_dict(instance))

    @overload
    def __get__(
        self, instance: None, owner: Any
    ) -> InstrumentedAttribute[_T_co]: ...

    @overload
    def __get__(self, instance: object, owner: Any) -> _T_co: ...

    def __get__(
        self, instance: Optional[object], owner: Any
    ) -> Union[InstrumentedAttribute[_T_co], _T_co]:
        if instance is None:
            return self

        dict_ = instance_dict(instance)
        if self.impl.supports_population and self.key in dict_:
            return dict_[self.key]  # type: ignore[no-any-return]
        else:
            try:
                state = instance_state(instance)
            except AttributeError as err:
                raise orm_exc.UnmappedInstanceError(instance) from err
            return self.impl.get(state, dict_)  # type: ignore[no-any-return]


@dataclasses.dataclass(frozen=True)
class AdHocHasEntityNamespace(HasCacheKey):
    _traverse_internals: ClassVar[_TraverseInternalsType] = [
        ("_entity_namespace", InternalTraversal.dp_has_cache_key),
    ]

    # py37 compat, no slots=True on dataclass
    __slots__ = ("_entity_namespace",)
    _entity_namespace: _InternalEntityType[Any]
    is_mapper: ClassVar[bool] = False
    is_aliased_class: ClassVar[bool] = False

    @property
    def entity_namespace(self):
        return self._entity_namespace.entity_namespace


def create_proxied_attribute(
    descriptor: Any,
) -> Callable[..., QueryableAttribute[Any]]:
    """Create an QueryableAttribute / user descriptor hybrid.

    Returns a new QueryableAttribute type that delegates descriptor
    behavior and getattr() to the given descriptor.
    """

    # TODO: can move this to descriptor_props if the need for this
    # function is removed from ext/hybrid.py

    class Proxy(QueryableAttribute[Any]):
        """Presents the :class:`.QueryableAttribute` interface as a
        proxy on top of a Python descriptor / :class:`.PropComparator`
        combination.

        """

        _extra_criteria = ()

        # the attribute error catches inside of __getattr__ basically create a
        # singularity if you try putting slots on this too
        # __slots__ = ("descriptor", "original_property", "_comparator")

        def __init__(
            self,
            class_,
            key,
            descriptor,
            comparator,
            adapt_to_entity=None,
            doc=None,
            original_property=None,
        ):
            self.class_ = class_
            self.key = key
            self.descriptor = descriptor
            self.original_property = original_property
            self._comparator = comparator
            self._adapt_to_entity = adapt_to_entity
            self._doc = self.__doc__ = doc

        @property
        def _parententity(self):
            return inspection.inspect(self.class_, raiseerr=False)

        @property
        def parent(self):
            return inspection.inspect(self.class_, raiseerr=False)

        _is_internal_proxy = True

        _cache_key_traversal = [
            ("key", visitors.ExtendedInternalTraversal.dp_string),
            ("_parententity", visitors.ExtendedInternalTraversal.dp_multi),
        ]

        @property
        def _impl_uses_objects(self):
            return (
                self.original_property is not None
                and getattr(self.class_, self.key).impl.uses_objects
            )

        @property
        def _entity_namespace(self):
            if hasattr(self._comparator, "_parententity"):
                return self._comparator._parententity
            else:
                # used by hybrid attributes which try to remain
                # agnostic of any ORM concepts like mappers
                return AdHocHasEntityNamespace(self._parententity)

        @property
        def property(self):
            return self.comparator.property

        @util.memoized_property
        def comparator(self):
            if callable(self._comparator):
                self._comparator = self._comparator()
            if self._adapt_to_entity:
                self._comparator = self._comparator.adapt_to_entity(
                    self._adapt_to_entity
                )
            return self._comparator

        def adapt_to_entity(self, adapt_to_entity):
            return self.__class__(
                adapt_to_entity.entity,
                self.key,
                self.descriptor,
                self._comparator,
                adapt_to_entity,
            )

        def _clone(self, **kw):
            return self.__class__(
                self.class_,
                self.key,
                self.descriptor,
                self._comparator,
                adapt_to_entity=self._adapt_to_entity,
                original_property=self.original_property,
            )

        def __get__(self, instance, owner):
            retval = self.descriptor.__get__(instance, owner)
            # detect if this is a plain Python @property, which just returns
            # itself for class level access.  If so, then return us.
            # Otherwise, return the object returned by the descriptor.
            if retval is self.descriptor and instance is None:
                return self
            else:
                return retval

        def __str__(self) -> str:
            return f"{self.class_.__name__}.{self.key}"

        def __getattr__(self, attribute):
            """Delegate __getattr__ to the original descriptor and/or
            comparator."""

            # this is unfortunately very complicated, and is easily prone
            # to recursion overflows when implementations of related
            # __getattr__ schemes are changed

            try:
                return util.MemoizedSlots.__getattr__(self, attribute)
            except AttributeError:
                pass

            try:
                return getattr(descriptor, attribute)
            except AttributeError as err:
                if attribute == "comparator":
                    raise AttributeError("comparator") from err
                try:
                    # comparator itself might be unreachable
                    comparator = self.comparator
                except AttributeError as err2:
                    raise AttributeError(
                        "Neither %r object nor unconfigured comparator "
                        "object associated with %s has an attribute %r"
                        % (type(descriptor).__name__, self, attribute)
                    ) from err2
                else:
                    try:
                        return getattr(comparator, attribute)
                    except AttributeError as err3:
                        raise AttributeError(
                            "Neither %r object nor %r object "
                            "associated with %s has an attribute %r"
                            % (
                                type(descriptor).__name__,
                                type(comparator).__name__,
                                self,
                                attribute,
                            )
                        ) from err3

    Proxy.__name__ = type(descriptor).__name__ + "Proxy"

    util.monkeypatch_proxied_specials(
        Proxy, type(descriptor), name="descriptor", from_instance=descriptor
    )
    return Proxy


OP_REMOVE = util.symbol("REMOVE")
OP_APPEND = util.symbol("APPEND")
OP_REPLACE = util.symbol("REPLACE")
OP_BULK_REPLACE = util.symbol("BULK_REPLACE")
OP_MODIFIED = util.symbol("MODIFIED")


class AttributeEventToken:
    """A token propagated throughout the course of a chain of attribute
    events.

    Serves as an indicator of the source of the event and also provides
    a means of controlling propagation across a chain of attribute
    operations.

    The :class:`.Event` object is sent as the ``initiator`` argument
    when dealing with events such as :meth:`.AttributeEvents.append`,
    :meth:`.AttributeEvents.set`,
    and :meth:`.AttributeEvents.remove`.

    The :class:`.Event` object is currently interpreted by the backref
    event handlers, and is used to control the propagation of operations
    across two mutually-dependent attributes.

    .. versionchanged:: 2.0  Changed the name from ``AttributeEvent``
       to ``AttributeEventToken``.

    :attribute impl: The :class:`.AttributeImpl` which is the current event
     initiator.

    :attribute op: The symbol :attr:`.OP_APPEND`, :attr:`.OP_REMOVE`,
     :attr:`.OP_REPLACE`, or :attr:`.OP_BULK_REPLACE`, indicating the
     source operation.

    """

    __slots__ = "impl", "op", "parent_token"

    def __init__(self, attribute_impl: AttributeImpl, op: util.symbol):
        self.impl = attribute_impl
        self.op = op
        self.parent_token = self.impl.parent_token

    def __eq__(self, other):
        return (
            isinstance(other, AttributeEventToken)
            and other.impl is self.impl
            and other.op == self.op
        )

    @property
    def key(self):
        return self.impl.key

    def hasparent(self, state):
        return self.impl.hasparent(state)


AttributeEvent = AttributeEventToken  # legacy
Event = AttributeEventToken  # legacy


class AttributeImpl:
    """internal implementation for instrumented attributes."""

    collection: bool
    default_accepts_scalar_loader: bool
    uses_objects: bool
    supports_population: bool
    dynamic: bool

    _is_has_collection_adapter = False

    _replace_token: AttributeEventToken
    _remove_token: AttributeEventToken
    _append_token: AttributeEventToken

    def __init__(
        self,
        class_: _ExternalEntityType[_O],
        key: str,
        callable_: Optional[_LoaderCallable],
        dispatch: _Dispatch[QueryableAttribute[Any]],
        trackparent: bool = False,
        compare_function: Optional[Callable[..., bool]] = None,
        active_history: bool = False,
        parent_token: Optional[AttributeEventToken] = None,
        load_on_unexpire: bool = True,
        send_modified_events: bool = True,
        accepts_scalar_loader: Optional[bool] = None,
        **kwargs: Any,
    ):
        r"""Construct an AttributeImpl.

        :param \class_: associated class

        :param key: string name of the attribute

        :param \callable_:
          optional function which generates a callable based on a parent
          instance, which produces the "default" values for a scalar or
          collection attribute when it's first accessed, if not present
          already.

        :param trackparent:
          if True, attempt to track if an instance has a parent attached
          to it via this attribute.

        :param compare_function:
          a function that compares two values which are normally
          assignable to this attribute.

        :param active_history:
          indicates that get_history() should always return the "old" value,
          even if it means executing a lazy callable upon attribute change.

        :param parent_token:
          Usually references the MapperProperty, used as a key for
          the hasparent() function to identify an "owning" attribute.
          Allows multiple AttributeImpls to all match a single
          owner attribute.

        :param load_on_unexpire:
          if False, don't include this attribute in a load-on-expired
          operation, i.e. the "expired_attribute_loader" process.
          The attribute can still be in the "expired" list and be
          considered to be "expired".   Previously, this flag was called
          "expire_missing" and is only used by a deferred column
          attribute.

        :param send_modified_events:
          if False, the InstanceState._modified_event method will have no
          effect; this means the attribute will never show up as changed in a
          history entry.

        """
        self.class_ = class_
        self.key = key
        self.callable_ = callable_
        self.dispatch = dispatch
        self.trackparent = trackparent
        self.parent_token = parent_token or self
        self.send_modified_events = send_modified_events
        if compare_function is None:
            self.is_equal = operator.eq
        else:
            self.is_equal = compare_function

        if accepts_scalar_loader is not None:
            self.accepts_scalar_loader = accepts_scalar_loader
        else:
            self.accepts_scalar_loader = self.default_accepts_scalar_loader

        _deferred_history = kwargs.pop("_deferred_history", False)
        self._deferred_history = _deferred_history

        if active_history:
            self.dispatch._active_history = True

        self.load_on_unexpire = load_on_unexpire
        self._modified_token = AttributeEventToken(self, OP_MODIFIED)

    __slots__ = (
        "class_",
        "key",
        "callable_",
        "dispatch",
        "trackparent",
        "parent_token",
        "send_modified_events",
        "is_equal",
        "load_on_unexpire",
        "_modified_token",
        "accepts_scalar_loader",
        "_deferred_history",
    )

    def __str__(self) -> str:
        return f"{self.class_.__name__}.{self.key}"

    def _get_active_history(self):
        """Backwards compat for impl.active_history"""

        return self.dispatch._active_history

    def _set_active_history(self, value):
        self.dispatch._active_history = value

    active_history = property(_get_active_history, _set_active_history)

    def hasparent(
        self, state: InstanceState[Any], optimistic: bool = False
    ) -> bool:
        """Return the boolean value of a `hasparent` flag attached to
        the given state.

        The `optimistic` flag determines what the default return value
        should be if no `hasparent` flag can be located.

        As this function is used to determine if an instance is an
        *orphan*, instances that were loaded from storage should be
        assumed to not be orphans, until a True/False value for this
        flag is set.

        An instance attribute that is loaded by a callable function
        will also not have a `hasparent` flag.

        """
        msg = "This AttributeImpl is not configured to track parents."
        assert self.trackparent, msg

        return (
            state.parents.get(id(self.parent_token), optimistic) is not False
        )

    def sethasparent(
        self,
        state: InstanceState[Any],
        parent_state: InstanceState[Any],
        value: bool,
    ) -> None:
        """Set a boolean flag on the given item corresponding to
        whether or not it is attached to a parent object via the
        attribute represented by this ``InstrumentedAttribute``.

        """
        msg = "This AttributeImpl is not configured to track parents."
        assert self.trackparent, msg

        id_ = id(self.parent_token)
        if value:
            state.parents[id_] = parent_state
        else:
            if id_ in state.parents:
                last_parent = state.parents[id_]

                if (
                    last_parent is not False
                    and last_parent.key != parent_state.key
                ):
                    if last_parent.obj() is None:
                        raise orm_exc.StaleDataError(
                            "Removing state %s from parent "
                            "state %s along attribute '%s', "
                            "but the parent record "
                            "has gone stale, can't be sure this "
                            "is the most recent parent."
                            % (
                                state_str(state),
                                state_str(parent_state),
                                self.key,
                            )
                        )

                    return

            state.parents[id_] = False

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> History:
        raise NotImplementedError()

    def get_all_pending(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_NO_INITIALIZE,
    ) -> _AllPendingType:
        """Return a list of tuples of (state, obj)
        for all objects in this attribute's current state
        + history.

        Only applies to object-based attributes.

        This is an inlining of existing functionality
        which roughly corresponds to:

            get_state_history(
                        state,
                        key,
                        passive=PASSIVE_NO_INITIALIZE).sum()

        """
        raise NotImplementedError()

    def _default_value(
        self, state: InstanceState[Any], dict_: _InstanceDict
    ) -> Any:
        """Produce an empty value for an uninitialized scalar attribute."""

        assert self.key not in dict_, (
            "_default_value should only be invoked for an "
            "uninitialized or expired attribute"
        )

        value = None
        for fn in self.dispatch.init_scalar:
            ret = fn(state, value, dict_)
            if ret is not ATTR_EMPTY:
                value = ret

        return value

    def get(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> Any:
        """Retrieve a value from the given object.
        If a callable is assembled on this object's attribute, and
        passive is False, the callable will be executed and the
        resulting value will be set as the new value for this attribute.
        """
        if self.key in dict_:
            return dict_[self.key]
        else:
            # if history present, don't load
            key = self.key
            if (
                key not in state.committed_state
                or state.committed_state[key] is NO_VALUE
            ):
                if not passive & CALLABLES_OK:
                    return PASSIVE_NO_RESULT

                value = self._fire_loader_callables(state, key, passive)

                if value is PASSIVE_NO_RESULT or value is NO_VALUE:
                    return value
                elif value is ATTR_WAS_SET:
                    try:
                        return dict_[key]
                    except KeyError as err:
                        # TODO: no test coverage here.
                        raise KeyError(
                            "Deferred loader for attribute "
                            "%r failed to populate "
                            "correctly" % key
                        ) from err
                elif value is not ATTR_EMPTY:
                    return self.set_committed_value(state, dict_, value)

            if not passive & INIT_OK:
                return NO_VALUE
            else:
                return self._default_value(state, dict_)

    def _fire_loader_callables(
        self, state: InstanceState[Any], key: str, passive: PassiveFlag
    ) -> Any:
        if (
            self.accepts_scalar_loader
            and self.load_on_unexpire
            and key in state.expired_attributes
        ):
            return state._load_expired(state, passive)
        elif key in state.callables:
            callable_ = state.callables[key]
            return callable_(state, passive)
        elif self.callable_:
            return self.callable_(state, passive)
        else:
            return ATTR_EMPTY

    def append(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> None:
        self.set(state, dict_, value, initiator, passive=passive)

    def remove(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> None:
        self.set(
            state, dict_, None, initiator, passive=passive, check_old=value
        )

    def pop(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> None:
        self.set(
            state,
            dict_,
            None,
            initiator,
            passive=passive,
            check_old=value,
            pop=True,
        )

    def set(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken] = None,
        passive: PassiveFlag = PASSIVE_OFF,
        check_old: Any = None,
        pop: bool = False,
    ) -> None:
        raise NotImplementedError()

    def delete(self, state: InstanceState[Any], dict_: _InstanceDict) -> None:
        raise NotImplementedError()

    def get_committed_value(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> Any:
        """return the unchanged value of this attribute"""

        if self.key in state.committed_state:
            value = state.committed_state[self.key]
            if value is NO_VALUE:
                return None
            else:
                return value
        else:
            return self.get(state, dict_, passive=passive)

    def set_committed_value(self, state, dict_, value):
        """set an attribute value on the given instance and 'commit' it."""

        dict_[self.key] = value
        state._commit(dict_, [self.key])
        return value


class ScalarAttributeImpl(AttributeImpl):
    """represents a scalar value-holding InstrumentedAttribute."""

    default_accepts_scalar_loader = True
    uses_objects = False
    supports_population = True
    collection = False
    dynamic = False

    __slots__ = "_replace_token", "_append_token", "_remove_token"

    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        self._replace_token = self._append_token = AttributeEventToken(
            self, OP_REPLACE
        )
        self._remove_token = AttributeEventToken(self, OP_REMOVE)

    def delete(self, state: InstanceState[Any], dict_: _InstanceDict) -> None:
        if self.dispatch._active_history:
            old = self.get(state, dict_, PASSIVE_RETURN_NO_VALUE)
        else:
            old = dict_.get(self.key, NO_VALUE)

        if self.dispatch.remove:
            self.fire_remove_event(state, dict_, old, self._remove_token)
        state._modified_event(dict_, self, old)

        existing = dict_.pop(self.key, NO_VALUE)
        if (
            existing is NO_VALUE
            and old is NO_VALUE
            and not state.expired
            and self.key not in state.expired_attributes
        ):
            raise AttributeError("%s object does not have a value" % self)

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: Dict[str, Any],
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> History:
        if self.key in dict_:
            return History.from_scalar_attribute(self, state, dict_[self.key])
        elif self.key in state.committed_state:
            return History.from_scalar_attribute(self, state, NO_VALUE)
        else:
            if passive & INIT_OK:
                passive ^= INIT_OK
            current = self.get(state, dict_, passive=passive)
            if current is PASSIVE_NO_RESULT:
                return HISTORY_BLANK
            else:
                return History.from_scalar_attribute(self, state, current)

    def set(
        self,
        state: InstanceState[Any],
        dict_: Dict[str, Any],
        value: Any,
        initiator: Optional[AttributeEventToken] = None,
        passive: PassiveFlag = PASSIVE_OFF,
        check_old: Optional[object] = None,
        pop: bool = False,
    ) -> None:
        if self.dispatch._active_history:
            old = self.get(state, dict_, PASSIVE_RETURN_NO_VALUE)
        else:
            old = dict_.get(self.key, NO_VALUE)

        if self.dispatch.set:
            value = self.fire_replace_event(
                state, dict_, value, old, initiator
            )
        state._modified_event(dict_, self, old)
        dict_[self.key] = value

    def fire_replace_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: _T,
        previous: Any,
        initiator: Optional[AttributeEventToken],
    ) -> _T:
        for fn in self.dispatch.set:
            value = fn(
                state, value, previous, initiator or self._replace_token
            )
        return value

    def fire_remove_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
    ) -> None:
        for fn in self.dispatch.remove:
            fn(state, value, initiator or self._remove_token)


class ScalarObjectAttributeImpl(ScalarAttributeImpl):
    """represents a scalar-holding InstrumentedAttribute,
    where the target object is also instrumented.

    Adds events to delete/set operations.

    """

    default_accepts_scalar_loader = False
    uses_objects = True
    supports_population = True
    collection = False

    __slots__ = ()

    def delete(self, state: InstanceState[Any], dict_: _InstanceDict) -> None:
        if self.dispatch._active_history:
            old = self.get(
                state,
                dict_,
                passive=PASSIVE_ONLY_PERSISTENT
                | NO_AUTOFLUSH
                | LOAD_AGAINST_COMMITTED,
            )
        else:
            old = self.get(
                state,
                dict_,
                passive=PASSIVE_NO_FETCH ^ INIT_OK
                | LOAD_AGAINST_COMMITTED
                | NO_RAISE,
            )

        self.fire_remove_event(state, dict_, old, self._remove_token)

        existing = dict_.pop(self.key, NO_VALUE)

        # if the attribute is expired, we currently have no way to tell
        # that an object-attribute was expired vs. not loaded.   So
        # for this test, we look to see if the object has a DB identity.
        if (
            existing is NO_VALUE
            and old is not PASSIVE_NO_RESULT
            and state.key is None
        ):
            raise AttributeError("%s object does not have a value" % self)

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> History:
        if self.key in dict_:
            current = dict_[self.key]
        else:
            if passive & INIT_OK:
                passive ^= INIT_OK
            current = self.get(state, dict_, passive=passive)
            if current is PASSIVE_NO_RESULT:
                return HISTORY_BLANK

        if not self._deferred_history:
            return History.from_object_attribute(self, state, current)
        else:
            original = state.committed_state.get(self.key, _NO_HISTORY)
            if original is PASSIVE_NO_RESULT:
                loader_passive = passive | (
                    PASSIVE_ONLY_PERSISTENT
                    | NO_AUTOFLUSH
                    | LOAD_AGAINST_COMMITTED
                    | NO_RAISE
                    | DEFERRED_HISTORY_LOAD
                )
                original = self._fire_loader_callables(
                    state, self.key, loader_passive
                )
            return History.from_object_attribute(
                self, state, current, original=original
            )

    def get_all_pending(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_NO_INITIALIZE,
    ) -> _AllPendingType:
        if self.key in dict_:
            current = dict_[self.key]
        elif passive & CALLABLES_OK:
            current = self.get(state, dict_, passive=passive)
        else:
            return []

        ret: _AllPendingType

        # can't use __hash__(), can't use __eq__() here
        if (
            current is not None
            and current is not PASSIVE_NO_RESULT
            and current is not NO_VALUE
        ):
            ret = [(instance_state(current), current)]
        else:
            ret = [(None, None)]

        if self.key in state.committed_state:
            original = state.committed_state[self.key]
            if (
                original is not None
                and original is not PASSIVE_NO_RESULT
                and original is not NO_VALUE
                and original is not current
            ):
                ret.append((instance_state(original), original))
        return ret

    def set(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken] = None,
        passive: PassiveFlag = PASSIVE_OFF,
        check_old: Any = None,
        pop: bool = False,
    ) -> None:
        """Set a value on the given InstanceState."""

        if self.dispatch._active_history:
            old = self.get(
                state,
                dict_,
                passive=PASSIVE_ONLY_PERSISTENT
                | NO_AUTOFLUSH
                | LOAD_AGAINST_COMMITTED,
            )
        else:
            old = self.get(
                state,
                dict_,
                passive=PASSIVE_NO_FETCH ^ INIT_OK
                | LOAD_AGAINST_COMMITTED
                | NO_RAISE,
            )

        if (
            check_old is not None
            and old is not PASSIVE_NO_RESULT
            and check_old is not old
        ):
            if pop:
                return
            else:
                raise ValueError(
                    "Object %s not associated with %s on attribute '%s'"
                    % (instance_str(check_old), state_str(state), self.key)
                )

        value = self.fire_replace_event(state, dict_, value, old, initiator)
        dict_[self.key] = value

    def fire_remove_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
    ) -> None:
        if self.trackparent and value not in (
            None,
            PASSIVE_NO_RESULT,
            NO_VALUE,
        ):
            self.sethasparent(instance_state(value), state, False)

        for fn in self.dispatch.remove:
            fn(state, value, initiator or self._remove_token)

        state._modified_event(dict_, self, value)

    def fire_replace_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: _T,
        previous: Any,
        initiator: Optional[AttributeEventToken],
    ) -> _T:
        if self.trackparent:
            if previous is not value and previous not in (
                None,
                PASSIVE_NO_RESULT,
                NO_VALUE,
            ):
                self.sethasparent(instance_state(previous), state, False)

        for fn in self.dispatch.set:
            value = fn(
                state, value, previous, initiator or self._replace_token
            )

        state._modified_event(dict_, self, previous)

        if self.trackparent:
            if value is not None:
                self.sethasparent(instance_state(value), state, True)

        return value


class HasCollectionAdapter:
    __slots__ = ()

    collection: bool
    _is_has_collection_adapter = True

    def _dispose_previous_collection(
        self,
        state: InstanceState[Any],
        collection: _AdaptedCollectionProtocol,
        adapter: CollectionAdapter,
        fire_event: bool,
    ) -> None:
        raise NotImplementedError()

    @overload
    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: Literal[None] = ...,
        passive: Literal[PassiveFlag.PASSIVE_OFF] = ...,
    ) -> CollectionAdapter: ...

    @overload
    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: _AdaptedCollectionProtocol = ...,
        passive: PassiveFlag = ...,
    ) -> CollectionAdapter: ...

    @overload
    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: Optional[_AdaptedCollectionProtocol] = ...,
        passive: PassiveFlag = ...,
    ) -> Union[
        Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter
    ]: ...

    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: Optional[_AdaptedCollectionProtocol] = None,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
    ) -> Union[
        Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter
    ]:
        raise NotImplementedError()

    def set(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken] = None,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
        check_old: Any = None,
        pop: bool = False,
        _adapt: bool = True,
    ) -> None:
        raise NotImplementedError()


if TYPE_CHECKING:

    def _is_collection_attribute_impl(
        impl: AttributeImpl,
    ) -> TypeGuard[CollectionAttributeImpl]: ...

else:
    _is_collection_attribute_impl = operator.attrgetter("collection")


class CollectionAttributeImpl(HasCollectionAdapter, AttributeImpl):
    """A collection-holding attribute that instruments changes in membership.

    Only handles collections of instrumented objects.

    InstrumentedCollectionAttribute holds an arbitrary, user-specified
    container object (defaulting to a list) and brokers access to the
    CollectionAdapter, a "view" onto that object that presents consistent bag
    semantics to the orm layer independent of the user data implementation.

    """

    uses_objects = True
    collection = True
    default_accepts_scalar_loader = False
    supports_population = True
    dynamic = False

    _bulk_replace_token: AttributeEventToken

    __slots__ = (
        "copy",
        "collection_factory",
        "_append_token",
        "_remove_token",
        "_bulk_replace_token",
        "_duck_typed_as",
    )

    def __init__(
        self,
        class_,
        key,
        callable_,
        dispatch,
        typecallable=None,
        trackparent=False,
        copy_function=None,
        compare_function=None,
        **kwargs,
    ):
        super().__init__(
            class_,
            key,
            callable_,
            dispatch,
            trackparent=trackparent,
            compare_function=compare_function,
            **kwargs,
        )

        if copy_function is None:
            copy_function = self.__copy
        self.copy = copy_function
        self.collection_factory = typecallable
        self._append_token = AttributeEventToken(self, OP_APPEND)
        self._remove_token = AttributeEventToken(self, OP_REMOVE)
        self._bulk_replace_token = AttributeEventToken(self, OP_BULK_REPLACE)
        self._duck_typed_as = util.duck_type_collection(
            self.collection_factory()
        )

        if getattr(self.collection_factory, "_sa_linker", None):

            @event.listens_for(self, "init_collection")
            def link(target, collection, collection_adapter):
                collection._sa_linker(collection_adapter)

            @event.listens_for(self, "dispose_collection")
            def unlink(target, collection, collection_adapter):
                collection._sa_linker(None)

    def __copy(self, item):
        return [y for y in collections.collection_adapter(item)]

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> History:
        current = self.get(state, dict_, passive=passive)

        if current is PASSIVE_NO_RESULT:
            if (
                passive & PassiveFlag.INCLUDE_PENDING_MUTATIONS
                and self.key in state._pending_mutations
            ):
                pending = state._pending_mutations[self.key]
                return pending.merge_with_history(HISTORY_BLANK)
            else:
                return HISTORY_BLANK
        else:
            if passive & PassiveFlag.INCLUDE_PENDING_MUTATIONS:
                # this collection is loaded / present.  should not be any
                # pending mutations
                assert self.key not in state._pending_mutations

            return History.from_collection(self, state, current)

    def get_all_pending(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PASSIVE_NO_INITIALIZE,
    ) -> _AllPendingType:
        # NOTE: passive is ignored here at the moment

        if self.key not in dict_:
            return []

        current = dict_[self.key]
        current = getattr(current, "_sa_adapter")

        if self.key in state.committed_state:
            original = state.committed_state[self.key]
            if original is not NO_VALUE:
                current_states = [
                    ((c is not None) and instance_state(c) or None, c)
                    for c in current
                ]
                original_states = [
                    ((c is not None) and instance_state(c) or None, c)
                    for c in original
                ]

                current_set = dict(current_states)
                original_set = dict(original_states)

                return (
                    [
                        (s, o)
                        for s, o in current_states
                        if s not in original_set
                    ]
                    + [(s, o) for s, o in current_states if s in original_set]
                    + [
                        (s, o)
                        for s, o in original_states
                        if s not in current_set
                    ]
                )

        return [(instance_state(o), o) for o in current]

    def fire_append_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: _T,
        initiator: Optional[AttributeEventToken],
        key: Optional[Any],
    ) -> _T:
        for fn in self.dispatch.append:
            value = fn(state, value, initiator or self._append_token, key=key)

        state._modified_event(dict_, self, NO_VALUE, True)

        if self.trackparent and value is not None:
            self.sethasparent(instance_state(value), state, True)

        return value

    def fire_append_wo_mutation_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: _T,
        initiator: Optional[AttributeEventToken],
        key: Optional[Any],
    ) -> _T:
        for fn in self.dispatch.append_wo_mutation:
            value = fn(state, value, initiator or self._append_token, key=key)

        return value

    def fire_pre_remove_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        initiator: Optional[AttributeEventToken],
        key: Optional[Any],
    ) -> None:
        """A special event used for pop() operations.

        The "remove" event needs to have the item to be removed passed to
        it, which in the case of pop from a set, we don't have a way to access
        the item before the operation.   the event is used for all pop()
        operations (even though set.pop is the one where it is really needed).

        """
        state._modified_event(dict_, self, NO_VALUE, True)

    def fire_remove_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        key: Optional[Any],
    ) -> None:
        if self.trackparent and value is not None:
            self.sethasparent(instance_state(value), state, False)

        for fn in self.dispatch.remove:
            fn(state, value, initiator or self._remove_token, key=key)

        state._modified_event(dict_, self, NO_VALUE, True)

    def delete(self, state: InstanceState[Any], dict_: _InstanceDict) -> None:
        if self.key not in dict_:
            return

        state._modified_event(dict_, self, NO_VALUE, True)

        collection = self.get_collection(state, state.dict)
        collection.clear_with_event()

        # key is always present because we checked above.  e.g.
        # del is a no-op if collection not present.
        del dict_[self.key]

    def _default_value(
        self, state: InstanceState[Any], dict_: _InstanceDict
    ) -> _AdaptedCollectionProtocol:
        """Produce an empty collection for an un-initialized attribute"""

        assert self.key not in dict_, (
            "_default_value should only be invoked for an "
            "uninitialized or expired attribute"
        )

        if self.key in state._empty_collections:
            return state._empty_collections[self.key]

        adapter, user_data = self._initialize_collection(state)
        adapter._set_empty(user_data)
        return user_data

    def _initialize_collection(
        self, state: InstanceState[Any]
    ) -> Tuple[CollectionAdapter, _AdaptedCollectionProtocol]:
        adapter, collection = state.manager.initialize_collection(
            self.key, state, self.collection_factory
        )

        self.dispatch.init_collection(state, collection, adapter)

        return adapter, collection

    def append(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> None:
        collection = self.get_collection(
            state, dict_, user_data=None, passive=passive
        )
        if collection is PASSIVE_NO_RESULT:
            value = self.fire_append_event(
                state, dict_, value, initiator, key=NO_KEY
            )
            assert (
                self.key not in dict_
            ), "Collection was loaded during event handling."
            state._get_pending_mutation(self.key).append(value)
        else:
            if TYPE_CHECKING:
                assert isinstance(collection, CollectionAdapter)
            collection.append_with_event(value, initiator)

    def remove(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> None:
        collection = self.get_collection(
            state, state.dict, user_data=None, passive=passive
        )
        if collection is PASSIVE_NO_RESULT:
            self.fire_remove_event(state, dict_, value, initiator, key=NO_KEY)
            assert (
                self.key not in dict_
            ), "Collection was loaded during event handling."
            state._get_pending_mutation(self.key).remove(value)
        else:
            if TYPE_CHECKING:
                assert isinstance(collection, CollectionAdapter)
            collection.remove_with_event(value, initiator)

    def pop(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> None:
        try:
            # TODO: better solution here would be to add
            # a "popper" role to collections.py to complement
            # "remover".
            self.remove(state, dict_, value, initiator, passive=passive)
        except (ValueError, KeyError, IndexError):
            pass

    def set(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken] = None,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
        check_old: Any = None,
        pop: bool = False,
        _adapt: bool = True,
    ) -> None:
        iterable = orig_iterable = value
        new_keys = None

        # pulling a new collection first so that an adaptation exception does
        # not trigger a lazy load of the old collection.
        new_collection, user_data = self._initialize_collection(state)
        if _adapt:
            if new_collection._converter is not None:
                iterable = new_collection._converter(iterable)
            else:
                setting_type = util.duck_type_collection(iterable)
                receiving_type = self._duck_typed_as

                if setting_type is not receiving_type:
                    given = (
                        iterable is None
                        and "None"
                        or iterable.__class__.__name__
                    )
                    wanted = self._duck_typed_as.__name__
                    raise TypeError(
                        "Incompatible collection type: %s is not %s-like"
                        % (given, wanted)
                    )

                # If the object is an adapted collection, return the (iterable)
                # adapter.
                if hasattr(iterable, "_sa_iterator"):
                    iterable = iterable._sa_iterator()
                elif setting_type is dict:
                    new_keys = list(iterable)
                    iterable = iterable.values()
                else:
                    iterable = iter(iterable)
        elif util.duck_type_collection(iterable) is dict:
            new_keys = list(value)

        new_values = list(iterable)

        evt = self._bulk_replace_token

        self.dispatch.bulk_replace(state, new_values, evt, keys=new_keys)

        # propagate NO_RAISE in passive through to the get() for the
        # existing object (ticket #8862)
        old = self.get(
            state,
            dict_,
            passive=PASSIVE_ONLY_PERSISTENT ^ (passive & PassiveFlag.NO_RAISE),
        )
        if old is PASSIVE_NO_RESULT:
            old = self._default_value(state, dict_)
        elif old is orig_iterable:
            # ignore re-assignment of the current collection, as happens
            # implicitly with in-place operators (foo.collection |= other)
            return

        # place a copy of "old" in state.committed_state
        state._modified_event(dict_, self, old, True)

        old_collection = old._sa_adapter

        dict_[self.key] = user_data

        collections.bulk_replace(
            new_values, old_collection, new_collection, initiator=evt
        )

        self._dispose_previous_collection(state, old, old_collection, True)

    def _dispose_previous_collection(
        self,
        state: InstanceState[Any],
        collection: _AdaptedCollectionProtocol,
        adapter: CollectionAdapter,
        fire_event: bool,
    ) -> None:
        del collection._sa_adapter

        # discarding old collection make sure it is not referenced in empty
        # collections.
        state._empty_collections.pop(self.key, None)
        if fire_event:
            self.dispatch.dispose_collection(state, collection, adapter)

    def _invalidate_collection(
        self, collection: _AdaptedCollectionProtocol
    ) -> None:
        adapter = getattr(collection, "_sa_adapter")
        adapter.invalidated = True

    def set_committed_value(
        self, state: InstanceState[Any], dict_: _InstanceDict, value: Any
    ) -> _AdaptedCollectionProtocol:
        """Set an attribute value on the given instance and 'commit' it."""

        collection, user_data = self._initialize_collection(state)

        if value:
            collection.append_multiple_without_event(value)

        state.dict[self.key] = user_data

        state._commit(dict_, [self.key])

        if self.key in state._pending_mutations:
            # pending items exist.  issue a modified event,
            # add/remove new items.
            state._modified_event(dict_, self, user_data, True)

            pending = state._pending_mutations.pop(self.key)
            added = pending.added_items
            removed = pending.deleted_items
            for item in added:
                collection.append_without_event(item)
            for item in removed:
                collection.remove_without_event(item)

        return user_data

    @overload
    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: Literal[None] = ...,
        passive: Literal[PassiveFlag.PASSIVE_OFF] = ...,
    ) -> CollectionAdapter: ...

    @overload
    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: _AdaptedCollectionProtocol = ...,
        passive: PassiveFlag = ...,
    ) -> CollectionAdapter: ...

    @overload
    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: Optional[_AdaptedCollectionProtocol] = ...,
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> Union[
        Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter
    ]: ...

    def get_collection(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        user_data: Optional[_AdaptedCollectionProtocol] = None,
        passive: PassiveFlag = PASSIVE_OFF,
    ) -> Union[
        Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter
    ]:
        """Retrieve the CollectionAdapter associated with the given state.

        if user_data is None, retrieves it from the state using normal
        "get()" rules, which will fire lazy callables or return the "empty"
        collection value.

        """
        if user_data is None:
            fetch_user_data = self.get(state, dict_, passive=passive)
            if fetch_user_data is LoaderCallableStatus.PASSIVE_NO_RESULT:
                return fetch_user_data
            else:
                user_data = cast("_AdaptedCollectionProtocol", fetch_user_data)

        return user_data._sa_adapter


def backref_listeners(
    attribute: QueryableAttribute[Any], key: str, uselist: bool
) -> None:
    """Apply listeners to synchronize a two-way relationship."""

    # use easily recognizable names for stack traces.

    # in the sections marked "tokens to test for a recursive loop",
    # this is somewhat brittle and very performance-sensitive logic
    # that is specific to how we might arrive at each event.  a marker
    # that can target us directly to arguments being invoked against
    # the impl might be simpler, but could interfere with other systems.

    parent_token = attribute.impl.parent_token
    parent_impl = attribute.impl

    def _acceptable_key_err(child_state, initiator, child_impl):
        raise ValueError(
            "Bidirectional attribute conflict detected: "
            'Passing object %s to attribute "%s" '
            'triggers a modify event on attribute "%s" '
            'via the backref "%s".'
            % (
                state_str(child_state),
                initiator.parent_token,
                child_impl.parent_token,
                attribute.impl.parent_token,
            )
        )

    def emit_backref_from_scalar_set_event(
        state, child, oldchild, initiator, **kw
    ):
        if oldchild is child:
            return child
        if (
            oldchild is not None
            and oldchild is not PASSIVE_NO_RESULT
            and oldchild is not NO_VALUE
        ):
            # With lazy=None, there's no guarantee that the full collection is
            # present when updating via a backref.
            old_state, old_dict = (
                instance_state(oldchild),
                instance_dict(oldchild),
            )
            impl = old_state.manager[key].impl

            # tokens to test for a recursive loop.
            if not impl.collection and not impl.dynamic:
                check_recursive_token = impl._replace_token
            else:
                check_recursive_token = impl._remove_token

            if initiator is not check_recursive_token:
                impl.pop(
                    old_state,
                    old_dict,
                    state.obj(),
                    parent_impl._append_token,
                    passive=PASSIVE_NO_FETCH,
                )

        if child is not None:
            child_state, child_dict = (
                instance_state(child),
                instance_dict(child),
            )
            child_impl = child_state.manager[key].impl

            if (
                initiator.parent_token is not parent_token
                and initiator.parent_token is not child_impl.parent_token
            ):
                _acceptable_key_err(state, initiator, child_impl)

            # tokens to test for a recursive loop.
            check_append_token = child_impl._append_token
            check_bulk_replace_token = (
                child_impl._bulk_replace_token
                if _is_collection_attribute_impl(child_impl)
                else None
            )

            if (
                initiator is not check_append_token
                and initiator is not check_bulk_replace_token
            ):
                child_impl.append(
                    child_state,
                    child_dict,
                    state.obj(),
                    initiator,
                    passive=PASSIVE_NO_FETCH,
                )
        return child

    def emit_backref_from_collection_append_event(
        state, child, initiator, **kw
    ):
        if child is None:
            return

        child_state, child_dict = instance_state(child), instance_dict(child)
        child_impl = child_state.manager[key].impl

        if (
            initiator.parent_token is not parent_token
            and initiator.parent_token is not child_impl.parent_token
        ):
            _acceptable_key_err(state, initiator, child_impl)

        # tokens to test for a recursive loop.
        check_append_token = child_impl._append_token
        check_bulk_replace_token = (
            child_impl._bulk_replace_token
            if _is_collection_attribute_impl(child_impl)
            else None
        )

        if (
            initiator is not check_append_token
            and initiator is not check_bulk_replace_token
        ):
            child_impl.append(
                child_state,
                child_dict,
                state.obj(),
                initiator,
                passive=PASSIVE_NO_FETCH,
            )
        return child

    def emit_backref_from_collection_remove_event(
        state, child, initiator, **kw
    ):
        if (
            child is not None
            and child is not PASSIVE_NO_RESULT
            and child is not NO_VALUE
        ):
            child_state, child_dict = (
                instance_state(child),
                instance_dict(child),
            )
            child_impl = child_state.manager[key].impl

            check_replace_token: Optional[AttributeEventToken]

            # tokens to test for a recursive loop.
            if not child_impl.collection and not child_impl.dynamic:
                check_remove_token = child_impl._remove_token
                check_replace_token = child_impl._replace_token
                check_for_dupes_on_remove = uselist and not parent_impl.dynamic
            else:
                check_remove_token = child_impl._remove_token
                check_replace_token = (
                    child_impl._bulk_replace_token
                    if _is_collection_attribute_impl(child_impl)
                    else None
                )
                check_for_dupes_on_remove = False

            if (
                initiator is not check_remove_token
                and initiator is not check_replace_token
            ):
                if not check_for_dupes_on_remove or not util.has_dupes(
                    # when this event is called, the item is usually
                    # present in the list, except for a pop() operation.
                    state.dict[parent_impl.key],
                    child,
                ):
                    child_impl.pop(
                        child_state,
                        child_dict,
                        state.obj(),
                        initiator,
                        passive=PASSIVE_NO_FETCH,
                    )

    if uselist:
        event.listen(
            attribute,
            "append",
            emit_backref_from_collection_append_event,
            retval=True,
            raw=True,
            include_key=True,
        )
    else:
        event.listen(
            attribute,
            "set",
            emit_backref_from_scalar_set_event,
            retval=True,
            raw=True,
            include_key=True,
        )
    # TODO: need coverage in test/orm/ of remove event
    event.listen(
        attribute,
        "remove",
        emit_backref_from_collection_remove_event,
        retval=True,
        raw=True,
        include_key=True,
    )


_NO_HISTORY = util.symbol("NO_HISTORY")
_NO_STATE_SYMBOLS = frozenset([id(PASSIVE_NO_RESULT), id(NO_VALUE)])


class History(NamedTuple):
    """A 3-tuple of added, unchanged and deleted values,
    representing the changes which have occurred on an instrumented
    attribute.

    The easiest way to get a :class:`.History` object for a particular
    attribute on an object is to use the :func:`_sa.inspect` function::

        from sqlalchemy import inspect

        hist = inspect(myobject).attrs.myattribute.history

    Each tuple member is an iterable sequence:

    * ``added`` - the collection of items added to the attribute (the first
      tuple element).

    * ``unchanged`` - the collection of items that have not changed on the
      attribute (the second tuple element).

    * ``deleted`` - the collection of items that have been removed from the
      attribute (the third tuple element).

    """

    added: Union[Tuple[()], List[Any]]
    unchanged: Union[Tuple[()], List[Any]]
    deleted: Union[Tuple[()], List[Any]]

    def __bool__(self) -> bool:
        return self != HISTORY_BLANK

    def empty(self) -> bool:
        """Return True if this :class:`.History` has no changes
        and no existing, unchanged state.

        """

        return not bool((self.added or self.deleted) or self.unchanged)

    def sum(self) -> Sequence[Any]:
        """Return a collection of added + unchanged + deleted."""

        return (
            (self.added or []) + (self.unchanged or []) + (self.deleted or [])
        )

    def non_deleted(self) -> Sequence[Any]:
        """Return a collection of added + unchanged."""

        return (self.added or []) + (self.unchanged or [])

    def non_added(self) -> Sequence[Any]:
        """Return a collection of unchanged + deleted."""

        return (self.unchanged or []) + (self.deleted or [])

    def has_changes(self) -> bool:
        """Return True if this :class:`.History` has changes."""

        return bool(self.added or self.deleted)

    def _merge(self, added: Iterable[Any], deleted: Iterable[Any]) -> History:
        return History(
            list(self.added) + list(added),
            self.unchanged,
            list(self.deleted) + list(deleted),
        )

    def as_state(self) -> History:
        return History(
            [
                (c is not None) and instance_state(c) or None
                for c in self.added
            ],
            [
                (c is not None) and instance_state(c) or None
                for c in self.unchanged
            ],
            [
                (c is not None) and instance_state(c) or None
                for c in self.deleted
            ],
        )

    @classmethod
    def from_scalar_attribute(
        cls,
        attribute: ScalarAttributeImpl,
        state: InstanceState[Any],
        current: Any,
    ) -> History:
        original = state.committed_state.get(attribute.key, _NO_HISTORY)

        deleted: Union[Tuple[()], List[Any]]

        if original is _NO_HISTORY:
            if current is NO_VALUE:
                return cls((), (), ())
            else:
                return cls((), [current], ())
        # don't let ClauseElement expressions here trip things up
        elif (
            current is not NO_VALUE
            and attribute.is_equal(current, original) is True
        ):
            return cls((), [current], ())
        else:
            # current convention on native scalars is to not
            # include information
            # about missing previous value in "deleted", but
            # we do include None, which helps in some primary
            # key situations
            if id(original) in _NO_STATE_SYMBOLS:
                deleted = ()
                # indicate a "del" operation occurred when we don't have
                # the previous value as: ([None], (), ())
                if id(current) in _NO_STATE_SYMBOLS:
                    current = None
            else:
                deleted = [original]
            if current is NO_VALUE:
                return cls((), (), deleted)
            else:
                return cls([current], (), deleted)

    @classmethod
    def from_object_attribute(
        cls,
        attribute: ScalarObjectAttributeImpl,
        state: InstanceState[Any],
        current: Any,
        original: Any = _NO_HISTORY,
    ) -> History:
        deleted: Union[Tuple[()], List[Any]]

        if original is _NO_HISTORY:
            original = state.committed_state.get(attribute.key, _NO_HISTORY)

        if original is _NO_HISTORY:
            if current is NO_VALUE:
                return cls((), (), ())
            else:
                return cls((), [current], ())
        elif current is original and current is not NO_VALUE:
            return cls((), [current], ())
        else:
            # current convention on related objects is to not
            # include information
            # about missing previous value in "deleted", and
            # to also not include None - the dependency.py rules
            # ignore the None in any case.
            if id(original) in _NO_STATE_SYMBOLS or original is None:
                deleted = ()
                # indicate a "del" operation occurred when we don't have
                # the previous value as: ([None], (), ())
                if id(current) in _NO_STATE_SYMBOLS:
                    current = None
            else:
                deleted = [original]
            if current is NO_VALUE:
                return cls((), (), deleted)
            else:
                return cls([current], (), deleted)

    @classmethod
    def from_collection(
        cls,
        attribute: CollectionAttributeImpl,
        state: InstanceState[Any],
        current: Any,
    ) -> History:
        original = state.committed_state.get(attribute.key, _NO_HISTORY)
        if current is NO_VALUE:
            return cls((), (), ())

        current = getattr(current, "_sa_adapter")
        if original is NO_VALUE:
            return cls(list(current), (), ())
        elif original is _NO_HISTORY:
            return cls((), list(current), ())
        else:
            current_states = [
                ((c is not None) and instance_state(c) or None, c)
                for c in current
            ]
            original_states = [
                ((c is not None) and instance_state(c) or None, c)
                for c in original
            ]

            current_set = dict(current_states)
            original_set = dict(original_states)

            return cls(
                [o for s, o in current_states if s not in original_set],
                [o for s, o in current_states if s in original_set],
                [o for s, o in original_states if s not in current_set],
            )


HISTORY_BLANK = History((), (), ())


def get_history(
    obj: object, key: str, passive: PassiveFlag = PASSIVE_OFF
) -> History:
    """Return a :class:`.History` record for the given object
    and attribute key.

    This is the **pre-flush** history for a given attribute, which is
    reset each time the :class:`.Session` flushes changes to the
    current database transaction.

    .. note::

        Prefer to use the :attr:`.AttributeState.history` and
        :meth:`.AttributeState.load_history` accessors to retrieve the
        :class:`.History` for instance attributes.


    :param obj: an object whose class is instrumented by the
      attributes package.

    :param key: string attribute name.

    :param passive: indicates loading behavior for the attribute
       if the value is not already present.   This is a
       bitflag attribute, which defaults to the symbol
       :attr:`.PASSIVE_OFF` indicating all necessary SQL
       should be emitted.

    .. seealso::

        :attr:`.AttributeState.history`

        :meth:`.AttributeState.load_history` - retrieve history
        using loader callables if the value is not locally present.

    """

    return get_state_history(instance_state(obj), key, passive)


def get_state_history(
    state: InstanceState[Any], key: str, passive: PassiveFlag = PASSIVE_OFF
) -> History:
    return state.get_history(key, passive)


def has_parent(
    cls: Type[_O], obj: _O, key: str, optimistic: bool = False
) -> bool:
    """TODO"""
    manager = manager_of_class(cls)
    state = instance_state(obj)
    return manager.has_parent(state, key, optimistic)


def register_attribute(
    class_: Type[_O],
    key: str,
    *,
    comparator: interfaces.PropComparator[_T],
    parententity: _InternalEntityType[_O],
    doc: Optional[str] = None,
    **kw: Any,
) -> InstrumentedAttribute[_T]:
    desc = register_descriptor(
        class_, key, comparator=comparator, parententity=parententity, doc=doc
    )
    register_attribute_impl(class_, key, **kw)
    return desc


def register_attribute_impl(
    class_: Type[_O],
    key: str,
    uselist: bool = False,
    callable_: Optional[_LoaderCallable] = None,
    useobject: bool = False,
    impl_class: Optional[Type[AttributeImpl]] = None,
    backref: Optional[str] = None,
    **kw: Any,
) -> QueryableAttribute[Any]:
    manager = manager_of_class(class_)
    if uselist:
        factory = kw.pop("typecallable", None)
        typecallable = manager.instrument_collection_class(
            key, factory or list
        )
    else:
        typecallable = kw.pop("typecallable", None)

    dispatch = cast(
        "_Dispatch[QueryableAttribute[Any]]", manager[key].dispatch
    )  # noqa: E501

    impl: AttributeImpl

    if impl_class:
        # TODO: this appears to be the WriteOnlyAttributeImpl /
        # DynamicAttributeImpl constructor which is hardcoded
        impl = cast("Type[WriteOnlyAttributeImpl]", impl_class)(
            class_, key, dispatch, **kw
        )
    elif uselist:
        impl = CollectionAttributeImpl(
            class_, key, callable_, dispatch, typecallable=typecallable, **kw
        )
    elif useobject:
        impl = ScalarObjectAttributeImpl(
            class_, key, callable_, dispatch, **kw
        )
    else:
        impl = ScalarAttributeImpl(class_, key, callable_, dispatch, **kw)

    manager[key].impl = impl

    if backref:
        backref_listeners(manager[key], backref, uselist)

    manager.post_configure_attribute(key)
    return manager[key]


def register_descriptor(
    class_: Type[Any],
    key: str,
    *,
    comparator: interfaces.PropComparator[_T],
    parententity: _InternalEntityType[Any],
    doc: Optional[str] = None,
) -> InstrumentedAttribute[_T]:
    manager = manager_of_class(class_)

    descriptor = InstrumentedAttribute(
        class_, key, comparator=comparator, parententity=parententity
    )

    descriptor.__doc__ = doc  # type: ignore

    manager.instrument_attribute(key, descriptor)
    return descriptor


def unregister_attribute(class_: Type[Any], key: str) -> None:
    manager_of_class(class_).uninstrument_attribute(key)


def init_collection(obj: object, key: str) -> CollectionAdapter:
    """Initialize a collection attribute and return the collection adapter.

    This function is used to provide direct access to collection internals
    for a previously unloaded attribute.  e.g.::

        collection_adapter = init_collection(someobject, "elements")
        for elem in values:
            collection_adapter.append_without_event(elem)

    For an easier way to do the above, see
    :func:`~sqlalchemy.orm.attributes.set_committed_value`.

    :param obj: a mapped object

    :param key: string attribute name where the collection is located.

    """
    state = instance_state(obj)
    dict_ = state.dict
    return init_state_collection(state, dict_, key)


def init_state_collection(
    state: InstanceState[Any], dict_: _InstanceDict, key: str
) -> CollectionAdapter:
    """Initialize a collection attribute and return the collection adapter.

    Discards any existing collection which may be there.

    """
    attr = state.manager[key].impl

    if TYPE_CHECKING:
        assert isinstance(attr, HasCollectionAdapter)

    old = dict_.pop(key, None)  # discard old collection
    if old is not None:
        old_collection = old._sa_adapter
        attr._dispose_previous_collection(state, old, old_collection, False)

    user_data = attr._default_value(state, dict_)
    adapter: CollectionAdapter = attr.get_collection(
        state, dict_, user_data, passive=PassiveFlag.PASSIVE_NO_FETCH
    )
    adapter._reset_empty()

    return adapter


def set_committed_value(instance, key, value):
    """Set the value of an attribute with no history events.

    Cancels any previous history present.  The value should be
    a scalar value for scalar-holding attributes, or
    an iterable for any collection-holding attribute.

    This is the same underlying method used when a lazy loader
    fires off and loads additional data from the database.
    In particular, this method can be used by application code
    which has loaded additional attributes or collections through
    separate queries, which can then be attached to an instance
    as though it were part of its original loaded state.

    """
    state, dict_ = instance_state(instance), instance_dict(instance)
    state.manager[key].impl.set_committed_value(state, dict_, value)


def set_attribute(
    instance: object,
    key: str,
    value: Any,
    initiator: Optional[AttributeEventToken] = None,
) -> None:
    """Set the value of an attribute, firing history events.

    This function may be used regardless of instrumentation
    applied directly to the class, i.e. no descriptors are required.
    Custom attribute management schemes will need to make usage
    of this method to establish attribute state as understood
    by SQLAlchemy.

    :param instance: the object that will be modified

    :param key: string name of the attribute

    :param value: value to assign

    :param initiator: an instance of :class:`.Event` that would have
     been propagated from a previous event listener.  This argument
     is used when the :func:`.set_attribute` function is being used within
     an existing event listening function where an :class:`.Event` object
     is being supplied; the object may be used to track the origin of the
     chain of events.

     .. versionadded:: 1.2.3

    """
    state, dict_ = instance_state(instance), instance_dict(instance)
    state.manager[key].impl.set(state, dict_, value, initiator)


def get_attribute(instance: object, key: str) -> Any:
    """Get the value of an attribute, firing any callables required.

    This function may be used regardless of instrumentation
    applied directly to the class, i.e. no descriptors are required.
    Custom attribute management schemes will need to make usage
    of this method to make usage of attribute state as understood
    by SQLAlchemy.

    """
    state, dict_ = instance_state(instance), instance_dict(instance)
    return state.manager[key].impl.get(state, dict_)


def del_attribute(instance: object, key: str) -> None:
    """Delete the value of an attribute, firing history events.

    This function may be used regardless of instrumentation
    applied directly to the class, i.e. no descriptors are required.
    Custom attribute management schemes will need to make usage
    of this method to establish attribute state as understood
    by SQLAlchemy.

    """
    state, dict_ = instance_state(instance), instance_dict(instance)
    state.manager[key].impl.delete(state, dict_)


def flag_modified(instance: object, key: str) -> None:
    """Mark an attribute on an instance as 'modified'.

    This sets the 'modified' flag on the instance and
    establishes an unconditional change event for the given attribute.
    The attribute must have a value present, else an
    :class:`.InvalidRequestError` is raised.

    To mark an object "dirty" without referring to any specific attribute
    so that it is considered within a flush, use the
    :func:`.attributes.flag_dirty` call.

    .. seealso::

        :func:`.attributes.flag_dirty`

    """
    state, dict_ = instance_state(instance), instance_dict(instance)
    impl = state.manager[key].impl
    impl.dispatch.modified(state, impl._modified_token)
    state._modified_event(dict_, impl, NO_VALUE, is_userland=True)


def flag_dirty(instance: object) -> None:
    """Mark an instance as 'dirty' without any specific attribute mentioned.

    This is a special operation that will allow the object to travel through
    the flush process for interception by events such as
    :meth:`.SessionEvents.before_flush`.   Note that no SQL will be emitted in
    the flush process for an object that has no changes, even if marked dirty
    via this method.  However, a :meth:`.SessionEvents.before_flush` handler
    will be able to see the object in the :attr:`.Session.dirty` collection and
    may establish changes on it, which will then be included in the SQL
    emitted.

    .. versionadded:: 1.2

    .. seealso::

        :func:`.attributes.flag_modified`

    """

    state, dict_ = instance_state(instance), instance_dict(instance)
    state._modified_event(dict_, None, NO_VALUE, is_userland=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\sets.py ===
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING, overload
from functools import reduce
from collections import defaultdict
from collections.abc import Mapping, Iterable
import inspect

from sympy.core.kind import Kind, UndefinedKind, NumberKind
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,
    fuzzy_not)
from sympy.core.numbers import Float, Integer
from sympy.core.operations import LatticeOp
from sympy.core.parameters import global_parameters
from sympy.core.relational import Eq, Ne, is_lt
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,
                                       subsets)
from sympy.utilities.misc import func_name, filldedent

from mpmath import mpi, mpf

from mpmath.libmp.libmpf import prec_to_dps


tfn = defaultdict(lambda: None, {
    True: S.true,
    S.true: S.true,
    False: S.false,
    S.false: S.false})


@sympify_method_args
class Set(Basic, EvalfMixin):
    """
    The base class for any kind of set.

    Explanation
    ===========

    This is not meant to be used directly as a container of items. It does not
    behave like the builtin ``set``; see :class:`FiniteSet` for that.

    Real intervals are represented by the :class:`Interval` class and unions of
    sets by the :class:`Union` class. The empty set is represented by the
    :class:`EmptySet` class and available as a singleton as ``S.EmptySet``.
    """

    __slots__: tuple[()] = ()

    is_number = False
    is_iterable = False
    is_interval = False

    is_FiniteSet = False
    is_Interval = False
    is_ProductSet = False
    is_Union = False
    is_Intersection: FuzzyBool = None
    is_UniversalSet: FuzzyBool = None
    is_Complement: FuzzyBool = None
    is_ComplexRegion = False

    is_empty: FuzzyBool = None
    is_finite_set: FuzzyBool = None

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    def is_EmptySet(self):
        return None

    if TYPE_CHECKING:

        def __new__(cls, *args: Basic | complex) -> Set:
            ...

        @overload # type: ignore
        def subs(self, arg1: Mapping[Basic | complex, Set | complex], arg2: None=None) -> Set: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Set | complex]], arg2: None=None, **kwargs: Any) -> Set: ...
        @overload
        def subs(self, arg1: Set | complex, arg2: Set | complex) -> Set: ...
        @overload
        def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

        def subs(self, arg1: Mapping[Basic | complex, Basic | complex] | Basic | complex, # type: ignore
                 arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
            ...

        def simplify(self, **kwargs) -> Set:
            assert False

        def evalf(self, n: int = 15, subs: dict[Basic, Basic | float] | None = None,
                  maxn: int = 100, chop: bool = False, strict: bool  = False,
                  quad: str | None = None, verbose: bool = False) -> Set:
            ...

        n = evalf

    @staticmethod
    def _infimum_key(expr):
        """
        Return infimum (if possible) else S.Infinity.
        """
        try:
            infimum = expr.inf
            assert infimum.is_comparable
            infimum = infimum.evalf()  # issue #18505
        except (NotImplementedError,
                AttributeError, AssertionError, ValueError):
            infimum = S.Infinity
        return infimum

    def union(self, other):
        """
        Returns the union of ``self`` and ``other``.

        Examples
        ========

        As a shortcut it is possible to use the ``+`` operator:

        >>> from sympy import Interval, FiniteSet
        >>> Interval(0, 1).union(Interval(2, 3))
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(0, 1) + Interval(2, 3)
        Union(Interval(0, 1), Interval(2, 3))
        >>> Interval(1, 2, True, True) + FiniteSet(2, 3)
        Union({3}, Interval.Lopen(1, 2))

        Similarly it is possible to use the ``-`` operator for set differences:

        >>> Interval(0, 2) - Interval(0, 1)
        Interval.Lopen(1, 2)
        >>> Interval(1, 3) - FiniteSet(2)
        Union(Interval.Ropen(1, 2), Interval.Lopen(2, 3))

        """
        return Union(self, other)

    def intersect(self, other):
        """
        Returns the intersection of 'self' and 'other'.

        Examples
        ========

        >>> from sympy import Interval

        >>> Interval(1, 3).intersect(Interval(1, 2))
        Interval(1, 2)

        >>> from sympy import imageset, Lambda, symbols, S
        >>> n, m = symbols('n m')
        >>> a = imageset(Lambda(n, 2*n), S.Integers)
        >>> a.intersect(imageset(Lambda(m, 2*m + 1), S.Integers))
        EmptySet

        """
        return Intersection(self, other)

    def intersection(self, other):
        """
        Alias for :meth:`intersect()`
        """
        return self.intersect(other)

    def is_disjoint(self, other):
        """
        Returns True if ``self`` and ``other`` are disjoint.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 2).is_disjoint(Interval(1, 2))
        False
        >>> Interval(0, 2).is_disjoint(Interval(3, 4))
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Disjoint_sets
        """
        return self.intersect(other) == S.EmptySet

    def isdisjoint(self, other):
        """
        Alias for :meth:`is_disjoint()`
        """
        return self.is_disjoint(other)

    def complement(self, universe):
        r"""
        The complement of 'self' w.r.t the given universe.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(0, 1).complement(S.Reals)
        Union(Interval.open(-oo, 0), Interval.open(1, oo))

        >>> Interval(0, 1).complement(S.UniversalSet)
        Complement(UniversalSet, Interval(0, 1))

        """
        return Complement(universe, self)

    def _complement(self, other):
        # this behaves as other - self
        if isinstance(self, ProductSet) and isinstance(other, ProductSet):
            # If self and other are disjoint then other - self == self
            if len(self.sets) != len(other.sets):
                return other

            # There can be other ways to represent this but this gives:
            # (A x B) - (C x D) = ((A - C) x B) U (A x (B - D))
            overlaps = []
            pairs = list(zip(self.sets, other.sets))
            for n in range(len(pairs)):
                sets = (o if i != n else o-s for i, (s, o) in enumerate(pairs))
                overlaps.append(ProductSet(*sets))
            return Union(*overlaps)

        elif isinstance(other, Interval):
            if isinstance(self, (Interval, FiniteSet)):
                return Intersection(other, self.complement(S.Reals))

        elif isinstance(other, Union):
            return Union(*(o - self for o in other.args))

        elif isinstance(other, Complement):
            return Complement(other.args[0], Union(other.args[1], self), evaluate=False)

        elif other is S.EmptySet:
            return S.EmptySet

        elif isinstance(other, FiniteSet):
            sifted = sift(other, lambda x: fuzzy_bool(self.contains(x)))
            # ignore those that are contained in self
            return Union(FiniteSet(*(sifted[False])),
                Complement(FiniteSet(*(sifted[None])), self, evaluate=False)
                if sifted[None] else S.EmptySet)

    def symmetric_difference(self, other):
        """
        Returns symmetric difference of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy import Interval, S
        >>> Interval(1, 3).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(3, oo))
        >>> Interval(1, 10).symmetric_difference(S.Reals)
        Union(Interval.open(-oo, 1), Interval.open(10, oo))

        >>> from sympy import S, EmptySet
        >>> S.Reals.symmetric_difference(EmptySet)
        Reals

        References
        ==========
        .. [1] https://en.wikipedia.org/wiki/Symmetric_difference

        """
        return SymmetricDifference(self, other)

    def _symmetric_difference(self, other):
        return Union(Complement(self, other), Complement(other, self))

    @property
    def inf(self):
        """
        The infimum of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).inf
        0
        >>> Union(Interval(0, 1), Interval(2, 3)).inf
        0

        """
        return self._inf

    @property
    def _inf(self):
        raise NotImplementedError("(%s)._inf" % self)

    @property
    def sup(self):
        """
        The supremum of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).sup
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).sup
        3

        """
        return self._sup

    @property
    def _sup(self):
        raise NotImplementedError("(%s)._sup" % self)

    def contains(self, other):
        """
        Returns a SymPy value indicating whether ``other`` is contained
        in ``self``: ``true`` if it is, ``false`` if it is not, else
        an unevaluated ``Contains`` expression (or, as in the case of
        ConditionSet and a union of FiniteSet/Intervals, an expression
        indicating the conditions for containment).

        Examples
        ========

        >>> from sympy import Interval, S
        >>> from sympy.abc import x

        >>> Interval(0, 1).contains(0.5)
        True

        As a shortcut it is possible to use the ``in`` operator, but that
        will raise an error unless an affirmative true or false is not
        obtained.

        >>> Interval(0, 1).contains(x)
        (0 <= x) & (x <= 1)
        >>> x in Interval(0, 1)
        Traceback (most recent call last):
        ...
        TypeError: did not evaluate to a bool: None

        The result of 'in' is a bool, not a SymPy value

        >>> 1 in Interval(0, 2)
        True
        >>> _ is S.true
        False
        """
        from .contains import Contains
        other = sympify(other, strict=True)

        c = self._contains(other)
        if isinstance(c, Contains):
            return c
        if c is None:
            return Contains(other, self, evaluate=False)
        b = tfn[c]
        if b is None:
            return c
        return b

    def _contains(self, other):
        """Test if ``other`` is an element of the set ``self``.

        This is an internal method that is expected to be overridden by
        subclasses of ``Set`` and will be called by the public
        :func:`Set.contains` method or the :class:`Contains` expression.

        Parameters
        ==========

        other: Sympified :class:`Basic` instance
            The object whose membership in ``self`` is to be tested.

        Returns
        =======

        Symbolic :class:`Boolean` or ``None``.

        A return value of ``None`` indicates that it is unknown whether
        ``other`` is contained in ``self``. Returning ``None`` from here
        ensures that ``self.contains(other)`` or ``Contains(self, other)`` will
        return an unevaluated :class:`Contains` expression.

        If not ``None`` then the returned value is a :class:`Boolean` that is
        logically equivalent to the statement that ``other`` is an element of
        ``self``. Usually this would be either ``S.true`` or ``S.false`` but
        not always.
        """
        raise NotImplementedError(f"{type(self).__name__}._contains")

    def is_subset(self, other):
        """
        Returns True if ``self`` is a subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_subset(Interval(0, 1, left_open=True))
        False

        """
        if not isinstance(other, Set):
            raise ValueError("Unknown argument '%s'" % other)

        # Handle the trivial cases
        if self == other:
            return True
        is_empty = self.is_empty
        if is_empty is True:
            return True
        elif fuzzy_not(is_empty) and other.is_empty:
            return False
        if self.is_finite_set is False and other.is_finite_set:
            return False

        # Dispatch on subclass rules
        ret = self._eval_is_subset(other)
        if ret is not None:
            return ret
        ret = other._eval_is_superset(self)
        if ret is not None:
            return ret

        # Use pairwise rules from multiple dispatch
        from sympy.sets.handlers.issubset import is_subset_sets
        ret = is_subset_sets(self, other)
        if ret is not None:
            return ret

        # Fall back on computing the intersection
        # XXX: We shouldn't do this. A query like this should be handled
        # without evaluating new Set objects. It should be the other way round
        # so that the intersect method uses is_subset for evaluation.
        if self.intersect(other) == self:
            return True

    def _eval_is_subset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    def _eval_is_superset(self, other):
        '''Returns a fuzzy bool for whether self is a subset of other.'''
        return None

    # This should be deprecated:
    def issubset(self, other):
        """
        Alias for :meth:`is_subset()`
        """
        return self.is_subset(other)

    def is_proper_subset(self, other):
        """
        Returns True if ``self`` is a proper subset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_proper_subset(Interval(0, 1))
        True
        >>> Interval(0, 1).is_proper_subset(Interval(0, 1))
        False

        """
        if isinstance(other, Set):
            return self != other and self.is_subset(other)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    def is_superset(self, other):
        """
        Returns True if ``self`` is a superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 0.5).is_superset(Interval(0, 1))
        False
        >>> Interval(0, 1).is_superset(Interval(0, 1, left_open=True))
        True

        """
        if isinstance(other, Set):
            return other.is_subset(self)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    # This should be deprecated:
    def issuperset(self, other):
        """
        Alias for :meth:`is_superset()`
        """
        return self.is_superset(other)

    def is_proper_superset(self, other):
        """
        Returns True if ``self`` is a proper superset of ``other``.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).is_proper_superset(Interval(0, 0.5))
        True
        >>> Interval(0, 1).is_proper_superset(Interval(0, 1))
        False

        """
        if isinstance(other, Set):
            return self != other and self.is_superset(other)
        else:
            raise ValueError("Unknown argument '%s'" % other)

    def _eval_powerset(self):
        from .powerset import PowerSet
        return PowerSet(self)

    def powerset(self):
        """
        Find the Power set of ``self``.

        Examples
        ========

        >>> from sympy import EmptySet, FiniteSet, Interval

        A power set of an empty set:

        >>> A = EmptySet
        >>> A.powerset()
        {EmptySet}

        A power set of a finite set:

        >>> A = FiniteSet(1, 2)
        >>> a, b, c = FiniteSet(1), FiniteSet(2), FiniteSet(1, 2)
        >>> A.powerset() == FiniteSet(a, b, c, EmptySet)
        True

        A power set of an interval:

        >>> Interval(1, 2).powerset()
        PowerSet(Interval(1, 2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Power_set

        """
        return self._eval_powerset()

    @property
    def measure(self):
        """
        The (Lebesgue) measure of ``self``.

        Examples
        ========

        >>> from sympy import Interval, Union
        >>> Interval(0, 1).measure
        1
        >>> Union(Interval(0, 1), Interval(2, 3)).measure
        2

        """
        return self._measure

    @property
    def kind(self):
        """
        The kind of a Set

        Explanation
        ===========

        Any :class:`Set` will have kind :class:`SetKind` which is
        parametrised by the kind of the elements of the set. For example
        most sets are sets of numbers and will have kind
        ``SetKind(NumberKind)``. If elements of sets are different in kind than
        their kind will ``SetKind(UndefinedKind)``. See
        :class:`sympy.core.kind.Kind` for an explanation of the kind system.

        Examples
        ========

        >>> from sympy import Interval, Matrix, FiniteSet, EmptySet, ProductSet, PowerSet

        >>> FiniteSet(Matrix([1, 2])).kind
        SetKind(MatrixKind(NumberKind))

        >>> Interval(1, 2).kind
        SetKind(NumberKind)

        >>> EmptySet.kind
        SetKind()

        A :class:`sympy.sets.powerset.PowerSet` is a set of sets:

        >>> PowerSet({1, 2, 3}).kind
        SetKind(SetKind(NumberKind))

        A :class:`ProductSet` represents the set of tuples of elements of
        other sets. Its kind is :class:`sympy.core.containers.TupleKind`
        parametrised by the kinds of the elements of those sets:

        >>> p = ProductSet(FiniteSet(1, 2), FiniteSet(3, 4))
        >>> list(p)
        [(1, 3), (2, 3), (1, 4), (2, 4)]
        >>> p.kind
        SetKind(TupleKind(NumberKind, NumberKind))

        When all elements of the set do not have same kind, the kind
        will be returned as ``SetKind(UndefinedKind)``:

        >>> FiniteSet(0, Matrix([1, 2])).kind
        SetKind(UndefinedKind)

        The kind of the elements of a set are given by the ``element_kind``
        attribute of ``SetKind``:

        >>> Interval(1, 2).kind.element_kind
        NumberKind

        See Also
        ========

        NumberKind
        sympy.core.kind.UndefinedKind
        sympy.core.containers.TupleKind
        MatrixKind
        sympy.matrices.expressions.sets.MatrixSet
        sympy.sets.conditionset.ConditionSet
        Rationals
        Naturals
        Integers
        sympy.sets.fancysets.ImageSet
        sympy.sets.fancysets.Range
        sympy.sets.fancysets.ComplexRegion
        sympy.sets.powerset.PowerSet
        sympy.sets.sets.ProductSet
        sympy.sets.sets.Interval
        sympy.sets.sets.Union
        sympy.sets.sets.Intersection
        sympy.sets.sets.Complement
        sympy.sets.sets.EmptySet
        sympy.sets.sets.UniversalSet
        sympy.sets.sets.FiniteSet
        sympy.sets.sets.SymmetricDifference
        sympy.sets.sets.DisjointUnion
        """
        return self._kind()

    @property
    def boundary(self):
        """
        The boundary or frontier of a set.

        Explanation
        ===========

        A point x is on the boundary of a set S if

        1.  x is in the closure of S.
            I.e. Every neighborhood of x contains a point in S.
        2.  x is not in the interior of S.
            I.e. There does not exist an open set centered on x contained
            entirely within S.

        There are the points on the outer rim of S.  If S is open then these
        points need not actually be contained within S.

        For example, the boundary of an interval is its start and end points.
        This is true regardless of whether or not the interval is open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).boundary
        {0, 1}
        >>> Interval(0, 1, True, False).boundary
        {0, 1}
        """
        return self._boundary

    @property
    def is_open(self):
        """
        Property method to check whether a set is open.

        Explanation
        ===========

        A set is open if and only if it has an empty intersection with its
        boundary. In particular, a subset A of the reals is open if and only
        if each one of its points is contained in an open interval that is a
        subset of A.

        Examples
        ========
        >>> from sympy import S
        >>> S.Reals.is_open
        True
        >>> S.Rationals.is_open
        False
        """
        return Intersection(self, self.boundary).is_empty

    @property
    def is_closed(self):
        """
        A property method to check whether a set is closed.

        Explanation
        ===========

        A set is closed if its complement is an open set. The closedness of a
        subset of the reals is determined with respect to R and its standard
        topology.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).is_closed
        True
        """
        return self.boundary.is_subset(self)

    @property
    def closure(self):
        """
        Property method which returns the closure of a set.
        The closure is defined as the union of the set itself and its
        boundary.

        Examples
        ========
        >>> from sympy import S, Interval
        >>> S.Reals.closure
        Reals
        >>> Interval(0, 1).closure
        Interval(0, 1)
        """
        return self + self.boundary

    @property
    def interior(self):
        """
        Property method which returns the interior of a set.
        The interior of a set S consists all points of S that do not
        belong to the boundary of S.

        Examples
        ========
        >>> from sympy import Interval
        >>> Interval(0, 1).interior
        Interval.open(0, 1)
        >>> Interval(0, 1).boundary.interior
        EmptySet
        """
        return self - self.boundary

    @property
    def _boundary(self):
        raise NotImplementedError()

    @property
    def _measure(self):
        raise NotImplementedError("(%s)._measure" % self)

    def _kind(self):
        return SetKind(UndefinedKind)

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return self.func(*[arg.evalf(n=dps) for arg in self.args])

    @sympify_return([('other', 'Set')], NotImplemented)
    def __add__(self, other):
        return self.union(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __or__(self, other):
        return self.union(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __and__(self, other):
        return self.intersect(other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __mul__(self, other):
        return ProductSet(self, other)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __xor__(self, other):
        return SymmetricDifference(self, other)

    @sympify_return([('exp', Expr)], NotImplemented)
    def __pow__(self, exp):
        if not (exp.is_Integer and exp >= 0):
            raise ValueError("%s: Exponent must be a positive Integer" % exp)
        return ProductSet(*[self]*exp)

    @sympify_return([('other', 'Set')], NotImplemented)
    def __sub__(self, other):
        return Complement(self, other)

    def __contains__(self, other):
        other = _sympify(other)
        c = self._contains(other)
        b = tfn[c]
        if b is None:
            # x in y must evaluate to T or F; to entertain a None
            # result with Set use y.contains(x)
            raise TypeError('did not evaluate to a bool: %r' % c)
        return b


class ProductSet(Set):
    """
    Represents a Cartesian Product of Sets.

    Explanation
    ===========

    Returns a Cartesian product given several sets as either an iterable
    or individual arguments.

    Can use ``*`` operator on any sets for convenient shorthand.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet, ProductSet
    >>> I = Interval(0, 5); S = FiniteSet(1, 2, 3)
    >>> ProductSet(I, S)
    ProductSet(Interval(0, 5), {1, 2, 3})

    >>> (2, 2) in ProductSet(I, S)
    True

    >>> Interval(0, 1) * Interval(0, 1) # The unit square
    ProductSet(Interval(0, 1), Interval(0, 1))

    >>> coin = FiniteSet('H', 'T')
    >>> set(coin**2)
    {(H, H), (H, T), (T, H), (T, T)}

    The Cartesian product is not commutative or associative e.g.:

    >>> I*S == S*I
    False
    >>> (I*I)*I == I*(I*I)
    False

    Notes
    =====

    - Passes most operations down to the argument sets

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cartesian_product
    """
    is_ProductSet = True

    def __new__(cls, *sets, **assumptions):
        if len(sets) == 1 and iterable(sets[0]) and not isinstance(sets[0], (Set, set)):
            sympy_deprecation_warning(
                """
ProductSet(iterable) is deprecated. Use ProductSet(*iterable) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-productset-iterable",
            )
            sets = tuple(sets[0])

        sets = [sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Arguments to ProductSet should be of type Set")

        # Nullary product of sets is *not* the empty set
        if len(sets) == 0:
            return FiniteSet(())

        if S.EmptySet in sets:
            return S.EmptySet

        return Basic.__new__(cls, *sets, **assumptions)

    @property
    def sets(self):
        return self.args

    def flatten(self):
        def _flatten(sets):
            for s in sets:
                if s.is_ProductSet:
                    yield from _flatten(s.sets)
                else:
                    yield s
        return ProductSet(*_flatten(self.sets))



    def _contains(self, element):
        """
        ``in`` operator for ProductSets.

        Examples
        ========

        >>> from sympy import Interval
        >>> (2, 3) in Interval(0, 5) * Interval(0, 5)
        True

        >>> (10, 10) in Interval(0, 5) * Interval(0, 5)
        False

        Passes operation on to constituent sets
        """
        if element.is_Symbol:
            return None

        if not isinstance(element, Tuple) or len(element) != len(self.sets):
            return S.false

        return And(*[s.contains(e) for s, e in zip(self.sets, element)])

    def as_relational(self, *symbols):
        symbols = [_sympify(s) for s in symbols]
        if len(symbols) != len(self.sets) or not all(
                i.is_Symbol for i in symbols):
            raise ValueError(
                'number of symbols must match the number of sets')
        return And(*[s.as_relational(i) for s, i in zip(self.sets, symbols)])

    @property
    def _boundary(self):
        return Union(*(ProductSet(*(b + b.boundary if i != j else b.boundary
                                for j, b in enumerate(self.sets)))
                                for i, a in enumerate(self.sets)))

    @property
    def is_iterable(self):
        """
        A property method which tests whether a set is iterable or not.
        Returns True if set is iterable, otherwise returns False.

        Examples
        ========

        >>> from sympy import FiniteSet, Interval
        >>> I = Interval(0, 1)
        >>> A = FiniteSet(1, 2, 3, 4, 5)
        >>> I.is_iterable
        False
        >>> A.is_iterable
        True

        """
        return all(set.is_iterable for set in self.sets)

    def __iter__(self):
        """
        A method which implements is_iterable property method.
        If self.is_iterable returns True (both constituent sets are iterable),
        then return the Cartesian Product. Otherwise, raise TypeError.
        """
        return iproduct(*self.sets)

    @property
    def is_empty(self):
        return fuzzy_or(s.is_empty for s in self.sets)

    @property
    def is_finite_set(self):
        all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
        return fuzzy_or([self.is_empty, all_finite])

    @property
    def _measure(self):
        measure = 1
        for s in self.sets:
            measure *= s.measure
        return measure

    def _kind(self):
        return SetKind(TupleKind(*(i.kind.element_kind for i in self.args)))

    def __len__(self):
        return reduce(lambda a, b: a*b, (len(s) for s in self.args))

    def __bool__(self):
        return all(self.sets)


class Interval(Set):
    """
    Represents a real interval as a Set.

    Usage:
        Returns an interval with end points ``start`` and ``end``.

        For ``left_open=True`` (default ``left_open`` is ``False``) the interval
        will be open on the left. Similarly, for ``right_open=True`` the interval
        will be open on the right.

    Examples
    ========

    >>> from sympy import Symbol, Interval
    >>> Interval(0, 1)
    Interval(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Ropen(0, 1)
    Interval.Ropen(0, 1)
    >>> Interval.Lopen(0, 1)
    Interval.Lopen(0, 1)
    >>> Interval.open(0, 1)
    Interval.open(0, 1)

    >>> a = Symbol('a', real=True)
    >>> Interval(0, a)
    Interval(0, a)

    Notes
    =====
    - Only real end points are supported
    - ``Interval(a, b)`` with $a > b$ will return the empty set
    - Use the ``evalf()`` method to turn an Interval into an mpmath
      ``mpi`` interval instance

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_%28mathematics%29
    """
    is_Interval = True

    def __new__(cls, start, end, left_open=False, right_open=False):

        start = _sympify(start)
        end = _sympify(end)
        left_open = _sympify(left_open)
        right_open = _sympify(right_open)

        if not all(isinstance(a, (type(true), type(false)))
            for a in [left_open, right_open]):
            raise NotImplementedError(
                "left_open and right_open can have only true/false values, "
                "got %s and %s" % (left_open, right_open))

        # Only allow real intervals
        if fuzzy_not(fuzzy_and(i.is_extended_real for i in (start, end, end-start))):
            raise ValueError("Non-real intervals are not supported")

        # evaluate if possible
        if is_lt(end, start):
            return S.EmptySet
        elif (end - start).is_negative:
            return S.EmptySet

        if end == start and (left_open or right_open):
            return S.EmptySet
        if end == start and not (left_open or right_open):
            if start is S.Infinity or start is S.NegativeInfinity:
                return S.EmptySet
            return FiniteSet(end)

        # Make sure infinite interval end points are open.
        if start is S.NegativeInfinity:
            left_open = true
        if end is S.Infinity:
            right_open = true
        if start == S.Infinity or end == S.NegativeInfinity:
            return S.EmptySet

        return Basic.__new__(cls, start, end, left_open, right_open)

    @property
    def start(self):
        """
        The left end point of the interval.

        This property takes the same value as the ``inf`` property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).start
        0

        """
        return self._args[0]

    @property
    def end(self):
        """
        The right end point of the interval.

        This property takes the same value as the ``sup`` property.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1).end
        1

        """
        return self._args[1]

    @property
    def left_open(self):
        """
        True if interval is left-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, left_open=True).left_open
        True
        >>> Interval(0, 1, left_open=False).left_open
        False

        """
        return self._args[2]

    @property
    def right_open(self):
        """
        True if interval is right-open.

        Examples
        ========

        >>> from sympy import Interval
        >>> Interval(0, 1, right_open=True).right_open
        True
        >>> Interval(0, 1, right_open=False).right_open
        False

        """
        return self._args[3]

    @classmethod
    def open(cls, a, b):
        """Return an interval including neither boundary."""
        return cls(a, b, True, True)

    @classmethod
    def Lopen(cls, a, b):
        """Return an interval not including the left boundary."""
        return cls(a, b, True, False)

    @classmethod
    def Ropen(cls, a, b):
        """Return an interval not including the right boundary."""
        return cls(a, b, False, True)

    @property
    def _inf(self):
        return self.start

    @property
    def _sup(self):
        return self.end

    @property
    def left(self):
        return self.start

    @property
    def right(self):
        return self.end

    @property
    def is_empty(self):
        if self.left_open or self.right_open:
            cond = self.start >= self.end  # One/both bounds open
        else:
            cond = self.start > self.end  # Both bounds closed
        return fuzzy_bool(cond)

    @property
    def is_finite_set(self):
        return self.measure.is_zero

    def _complement(self, other):
        if other == S.Reals:
            a = Interval(S.NegativeInfinity, self.start,
                         True, not self.left_open)
            b = Interval(self.end, S.Infinity, not self.right_open, True)
            return Union(a, b)

        if isinstance(other, FiniteSet):
            nums = [m for m in other.args if m.is_number]
            if nums == []:
                return None

        return Set._complement(self, other)

    @property
    def _boundary(self):
        finite_points = [p for p in (self.start, self.end)
                         if abs(p) != S.Infinity]
        return FiniteSet(*finite_points)

    def _contains(self, other):
        if (not isinstance(other, Expr) or other is S.NaN
            or other.is_real is False or other.has(S.ComplexInfinity)):
                # if an expression has zoo it will be zoo or nan
                # and neither of those is real
                return false

        if self.start is S.NegativeInfinity and self.end is S.Infinity:
            if other.is_real is not None:
                return tfn[other.is_real]

        d = Dummy()
        return self.as_relational(d).subs(d, other)

    def as_relational(self, x):
        """Rewrite an interval in terms of inequalities and logic operators."""
        x = sympify(x)
        if self.right_open:
            right = x < self.end
        else:
            right = x <= self.end
        if self.left_open:
            left = self.start < x
        else:
            left = self.start <= x
        return And(left, right)

    @property
    def _measure(self):
        return self.end - self.start

    def _kind(self):
        return SetKind(NumberKind)

    def to_mpi(self, prec=53):
        return mpi(mpf(self.start._eval_evalf(prec)),
            mpf(self.end._eval_evalf(prec)))

    def _eval_evalf(self, prec):
        return Interval(self.left._evalf(prec), self.right._evalf(prec),
            left_open=self.left_open, right_open=self.right_open)

    def _is_comparable(self, other):
        is_comparable = self.start.is_comparable
        is_comparable &= self.end.is_comparable
        is_comparable &= other.start.is_comparable
        is_comparable &= other.end.is_comparable

        return is_comparable

    @property
    def is_left_unbounded(self):
        """Return ``True`` if the left endpoint is negative infinity. """
        return self.left is S.NegativeInfinity or self.left == Float("-inf")

    @property
    def is_right_unbounded(self):
        """Return ``True`` if the right endpoint is positive infinity. """
        return self.right is S.Infinity or self.right == Float("+inf")

    def _eval_Eq(self, other):
        if not isinstance(other, Interval):
            if isinstance(other, FiniteSet):
                return false
            elif isinstance(other, Set):
                return None
            return false


class Union(Set, LatticeOp):
    """
    Represents a union of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Union, Interval
    >>> Union(Interval(1, 2), Interval(3, 4))
    Union(Interval(1, 2), Interval(3, 4))

    The Union constructor will always try to merge overlapping intervals,
    if possible. For example:

    >>> Union(Interval(1, 2), Interval(2, 3))
    Interval(1, 3)

    See Also
    ========

    Intersection

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Union_%28set_theory%29
    """
    is_Union = True

    @property
    def identity(self):
        return S.EmptySet

    @property
    def zero(self):
        return S.UniversalSet

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs to merge intersections and iterables
        args = _sympify(args)

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_union(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    def args(self):
        return self._args

    def _complement(self, universe):
        # DeMorgan's Law
        return Intersection(s.complement(universe) for s in self.args)

    @property
    def _inf(self):
        # We use Min so that sup is meaningful in combination with symbolic
        # interval end points.
        return Min(*[set.inf for set in self.args])

    @property
    def _sup(self):
        # We use Max so that sup is meaningful in combination with symbolic
        # end points.
        return Max(*[set.sup for set in self.args])

    @property
    def is_empty(self):
        return fuzzy_and(set.is_empty for set in self.args)

    @property
    def is_finite_set(self):
        return fuzzy_and(set.is_finite_set for set in self.args)

    @property
    def _measure(self):
        # Measure of a union is the sum of the measures of the sets minus
        # the sum of their pairwise intersections plus the sum of their
        # triple-wise intersections minus ... etc...

        # Sets is a collection of intersections and a set of elementary
        # sets which made up those intersections (called "sos" for set of sets)
        # An example element might of this list might be:
        #    ( {A,B,C}, A.intersect(B).intersect(C) )

        # Start with just elementary sets (  ({A}, A), ({B}, B), ... )
        # Then get and subtract (  ({A,B}, (A int B), ... ) while non-zero
        sets = [(FiniteSet(s), s) for s in self.args]
        measure = 0
        parity = 1
        while sets:
            # Add up the measure of these sets and add or subtract it to total
            measure += parity * sum(inter.measure for sos, inter in sets)

            # For each intersection in sets, compute the intersection with every
            # other set not already part of the intersection.
            sets = ((sos + FiniteSet(newset), newset.intersect(intersection))
                    for sos, intersection in sets for newset in self.args
                    if newset not in sos)

            # Clear out sets with no measure
            sets = [(sos, inter) for sos, inter in sets if inter.measure != 0]

            # Clear out duplicates
            sos_list = []
            sets_list = []
            for _set in sets:
                if _set[0] in sos_list:
                    continue
                else:
                    sos_list.append(_set[0])
                    sets_list.append(_set)
            sets = sets_list

            # Flip Parity - next time subtract/add if we added/subtracted here
            parity *= -1
        return measure

    def _kind(self):
        kinds = tuple(arg.kind for arg in self.args if arg is not S.EmptySet)
        if not kinds:
            return SetKind()
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]
        else:
            return SetKind(UndefinedKind)

    @property
    def _boundary(self):
        def boundary_of_set(i):
            """ The boundary of set i minus interior of all other sets """
            b = self.args[i].boundary
            for j, a in enumerate(self.args):
                if j != i:
                    b = b - a.interior
            return b
        return Union(*map(boundary_of_set, range(len(self.args))))

    def _contains(self, other):
        return Or(*[s.contains(other) for s in self.args])

    def is_subset(self, other):
        return fuzzy_and(s.is_subset(other) for s in self.args)

    def as_relational(self, symbol):
        """Rewrite a Union in terms of equalities and logic operators. """
        if (len(self.args) == 2 and
                all(isinstance(i, Interval) for i in self.args)):
            # optimization to give 3 args as (x > 1) & (x < 5) & Ne(x, 3)
            # instead of as 4, ((1 <= x) & (x < 3)) | ((x <= 5) & (3 < x))
            # XXX: This should be ideally be improved to handle any number of
            # intervals and also not to assume that the intervals are in any
            # particular sorted order.
            a, b = self.args
            if a.sup == b.inf and a.right_open and b.left_open:
                mincond = symbol > a.inf if a.left_open else symbol >= a.inf
                maxcond = symbol < b.sup if b.right_open else symbol <= b.sup
                necond = Ne(symbol, a.sup)
                return And(necond, mincond, maxcond)
        return Or(*[i.as_relational(symbol) for i in self.args])

    @property
    def is_iterable(self):
        return all(arg.is_iterable for arg in self.args)

    def __iter__(self):
        return roundrobin(*(iter(arg) for arg in self.args))


class Intersection(Set, LatticeOp):
    """
    Represents an intersection of sets as a :class:`Set`.

    Examples
    ========

    >>> from sympy import Intersection, Interval
    >>> Intersection(Interval(1, 3), Interval(2, 4))
    Interval(2, 3)

    We often use the .intersect method

    >>> Interval(1,3).intersect(Interval(2,4))
    Interval(2, 3)

    See Also
    ========

    Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Intersection_%28set_theory%29
    """
    is_Intersection = True

    @property
    def identity(self):
        return S.UniversalSet

    @property
    def zero(self):
        return S.EmptySet

    def __new__(cls, *args , evaluate=None):
        if evaluate is None:
            evaluate = global_parameters.evaluate

        # flatten inputs to merge intersections and iterables
        args = list(ordered(set(_sympify(args))))

        # Reduce sets using known rules
        if evaluate:
            args = list(cls._new_args_filter(args))
            return simplify_intersection(args)

        args = list(ordered(args, Set._infimum_key))

        obj = Basic.__new__(cls, *args)
        obj._argset = frozenset(args)
        return obj

    @property
    def args(self):
        return self._args

    @property
    def is_iterable(self):
        return any(arg.is_iterable for arg in self.args)

    @property
    def is_finite_set(self):
        if fuzzy_or(arg.is_finite_set for arg in self.args):
            return True

    def _kind(self):
        kinds = tuple(arg.kind for arg in self.args if arg is not S.UniversalSet)
        if not kinds:
            return SetKind(UndefinedKind)
        elif all(i == kinds[0] for i in kinds):
            return kinds[0]
        else:
            return SetKind()

    @property
    def _inf(self):
        raise NotImplementedError()

    @property
    def _sup(self):
        raise NotImplementedError()

    def _contains(self, other):
        return And(*[set.contains(other) for set in self.args])

    def __iter__(self):
        sets_sift = sift(self.args, lambda x: x.is_iterable)

        completed = False
        candidates = sets_sift[True] + sets_sift[None]

        finite_candidates, others = [], []
        for candidate in candidates:
            length = None
            try:
                length = len(candidate)
            except TypeError:
                others.append(candidate)

            if length is not None:
                finite_candidates.append(candidate)
        finite_candidates.sort(key=len)

        for s in finite_candidates + others:
            other_sets = set(self.args) - {s}
            other = Intersection(*other_sets, evaluate=False)
            completed = True
            for x in s:
                try:
                    if x in other:
                        yield x
                except TypeError:
                    completed = False
            if completed:
                return

        if not completed:
            if not candidates:
                raise TypeError("None of the constituent sets are iterable")
            raise TypeError(
                "The computation had not completed because of the "
                "undecidable set membership is found in every candidates.")

    @staticmethod
    def _handle_finite_sets(args):
        '''Simplify intersection of one or more FiniteSets and other sets'''

        # First separate the FiniteSets from the others
        fs_args, others = sift(args, lambda x: x.is_FiniteSet, binary=True)

        # Let the caller handle intersection of non-FiniteSets
        if not fs_args:
            return

        # Convert to Python sets and build the set of all elements
        fs_sets = [set(fs) for fs in fs_args]
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Extract elements that are definitely in or definitely not in the
        # intersection. Here we check contains for all of args.
        definite = set()
        for e in all_elements:
            inall = fuzzy_and(s.contains(e) for s in args)
            if inall is True:
                definite.add(e)
            if inall is not None:
                for s in fs_sets:
                    s.discard(e)

        # At this point all elements in all of fs_sets are possibly in the
        # intersection. In some cases this is because they are definitely in
        # the intersection of the finite sets but it's not clear if they are
        # members of others. We might have {m, n}, {m}, and Reals where we
        # don't know if m or n is real. We want to remove n here but it is
        # possibly in because it might be equal to m. So what we do now is
        # extract the elements that are definitely in the remaining finite
        # sets iteratively until we end up with {n}, {}. At that point if we
        # get any empty set all remaining elements are discarded.

        fs_elements = reduce(lambda a, b: a | b, fs_sets, set())

        # Need fuzzy containment testing
        fs_symsets = [FiniteSet(*s) for s in fs_sets]

        while fs_elements:
            for e in fs_elements:
                infs = fuzzy_and(s.contains(e) for s in fs_symsets)
                if infs is True:
                    definite.add(e)
                if infs is not None:
                    for n, s in enumerate(fs_sets):
                        # Update Python set and FiniteSet
                        if e in s:
                            s.remove(e)
                            fs_symsets[n] = FiniteSet(*s)
                    fs_elements.remove(e)
                    break
            # If we completed the for loop without removing anything we are
            # done so quit the outer while loop
            else:
                break

        # If any of the sets of remainder elements is empty then we discard
        # all of them for the intersection.
        if not all(fs_sets):
            fs_sets = [set()]

        # Here we fold back the definitely included elements into each fs.
        # Since they are definitely included they must have been members of
        # each FiniteSet to begin with. We could instead fold these in with a
        # Union at the end to get e.g. {3}|({x}&{y}) rather than {3,x}&{3,y}.
        if definite:
            fs_sets = [fs | definite for fs in fs_sets]

        if fs_sets == [set()]:
            return S.EmptySet

        sets = [FiniteSet(*s) for s in fs_sets]

        # Any set in others is redundant if it contains all the elements that
        # are in the finite sets so we don't need it in the Intersection
        all_elements = reduce(lambda a, b: a | b, fs_sets, set())
        is_redundant = lambda o: all(fuzzy_bool(o.contains(e)) for e in all_elements)
        others = [o for o in others if not is_redundant(o)]

        if others:
            rest = Intersection(*others)
            # XXX: Maybe this shortcut should be at the beginning. For large
            # FiniteSets it could much more efficient to process the other
            # sets first...
            if rest is S.EmptySet:
                return S.EmptySet
            # Flatten the Intersection
            if rest.is_Intersection:
                sets.extend(rest.args)
            else:
                sets.append(rest)

        if len(sets) == 1:
            return sets[0]
        else:
            return Intersection(*sets, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite an Intersection in terms of equalities and logic operators"""
        return And(*[set.as_relational(symbol) for set in self.args])


class Complement(Set):
    r"""Represents the set difference or relative complement of a set with
    another set.

    $$A - B = \{x \in A \mid x \notin B\}$$


    Examples
    ========

    >>> from sympy import Complement, FiniteSet
    >>> Complement(FiniteSet(0, 1, 2), FiniteSet(1))
    {0, 2}

    See Also
    =========

    Intersection, Union

    References
    ==========

    .. [1] https://mathworld.wolfram.com/ComplementSet.html
    """

    is_Complement = True

    def __new__(cls, a, b, evaluate=True):
        a, b = map(_sympify, (a, b))
        if evaluate:
            return Complement.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        """
        Simplify a :class:`Complement`.

        """
        if B == S.UniversalSet or A.is_subset(B):
            return S.EmptySet

        if isinstance(B, Union):
            return Intersection(*(s.complement(A) for s in B.args))

        result = B._complement(A)
        if result is not None:
            return result
        else:
            return Complement(A, B, evaluate=False)

    def _contains(self, other):
        A = self.args[0]
        B = self.args[1]
        return And(A.contains(other), Not(B.contains(other)))

    def as_relational(self, symbol):
        """Rewrite a complement in terms of equalities and logic
        operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = Not(B.as_relational(symbol))

        return And(A_rel, B_rel)

    def _kind(self):
        return self.args[0].kind

    @property
    def is_iterable(self):
        if self.args[0].is_iterable:
            return True

    @property
    def is_finite_set(self):
        A, B = self.args
        a_finite = A.is_finite_set
        if a_finite is True:
            return True
        elif a_finite is False and B.is_finite_set:
            return False

    def __iter__(self):
        A, B = self.args
        for a in A:
            if a not in B:
                    yield a
            else:
                continue


class EmptySet(Set, metaclass=Singleton):
    """
    Represents the empty set. The empty set is available as a singleton
    as ``S.EmptySet``.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> S.EmptySet
    EmptySet

    >>> Interval(1, 2).intersect(S.EmptySet)
    EmptySet

    See Also
    ========

    UniversalSet

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Empty_set
    """
    is_empty = True
    is_finite_set = True
    is_FiniteSet = True

    @property  # type: ignore
    @deprecated(
        """
        The is_EmptySet attribute of Set objects is deprecated.
        Use 's is S.EmptySet" or 's.is_empty' instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-is-emptyset",
    )
    def is_EmptySet(self):
        return True

    @property
    def _measure(self):
        return 0

    def _contains(self, other):
        return false

    def as_relational(self, symbol):
        return false

    def __len__(self):
        return 0

    def __iter__(self):
        return iter([])

    def _eval_powerset(self):
        return FiniteSet(self)

    @property
    def _boundary(self):
        return self

    def _complement(self, other):
        return other

    def _kind(self):
        return SetKind()

    def _symmetric_difference(self, other):
        return other


class UniversalSet(Set, metaclass=Singleton):
    """
    Represents the set of all things.
    The universal set is available as a singleton as ``S.UniversalSet``.

    Examples
    ========

    >>> from sympy import S, Interval
    >>> S.UniversalSet
    UniversalSet

    >>> Interval(1, 2).intersect(S.UniversalSet)
    Interval(1, 2)

    See Also
    ========

    EmptySet

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Universal_set
    """

    is_UniversalSet = True
    is_empty = False
    is_finite_set = False

    def _complement(self, other):
        return S.EmptySet

    def _symmetric_difference(self, other):
        return other

    @property
    def _measure(self):
        return S.Infinity

    def _kind(self):
        return SetKind(UndefinedKind)

    def _contains(self, other):
        return true

    def as_relational(self, symbol):
        return true

    @property
    def _boundary(self):
        return S.EmptySet


class FiniteSet(Set):
    """
    Represents a finite set of Sympy expressions.

    Examples
    ========

    >>> from sympy import FiniteSet, Symbol, Interval, Naturals0
    >>> FiniteSet(1, 2, 3, 4)
    {1, 2, 3, 4}
    >>> 3 in FiniteSet(1, 2, 3, 4)
    True
    >>> FiniteSet(1, (1, 2), Symbol('x'))
    {1, x, (1, 2)}
    >>> FiniteSet(Interval(1, 2), Naturals0, {1, 2})
    FiniteSet({1, 2}, Interval(1, 2), Naturals0)
    >>> members = [1, 2, 3, 4]
    >>> f = FiniteSet(*members)
    >>> f
    {1, 2, 3, 4}
    >>> f - FiniteSet(2)
    {1, 3, 4}
    >>> f + FiniteSet(2, 5)
    {1, 2, 3, 4, 5}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite_set
    """
    is_FiniteSet = True
    is_iterable = True
    is_empty = False
    is_finite_set = True

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if evaluate:
            args = list(map(sympify, args))

            if len(args) == 0:
                return S.EmptySet
        else:
            args = list(map(sympify, args))

        # keep the form of the first canonical arg
        dargs = {}
        for i in reversed(list(ordered(args))):
            if i.is_Symbol:
                dargs[i] = i
            else:
                try:
                    dargs[i.as_dummy()] = i
                except TypeError:
                    # e.g. i = class without args like `Interval`
                    dargs[i] = i
        _args_set = set(dargs.values())
        args = list(ordered(_args_set, Set._infimum_key))
        obj = Basic.__new__(cls, *args)
        obj._args_set = _args_set
        return obj


    def __iter__(self):
        return iter(self.args)

    def _complement(self, other):
        if isinstance(other, Interval):
            # Splitting in sub-intervals is only done for S.Reals;
            # other cases that need splitting will first pass through
            # Set._complement().
            nums, syms = [], []
            for m in self.args:
                if m.is_number and m.is_real:
                    nums.append(m)
                elif m.is_real == False:
                    pass  # drop non-reals
                else:
                    syms.append(m)  # various symbolic expressions
            if other == S.Reals and nums != []:
                nums.sort()
                intervals = []  # Build up a list of intervals between the elements
                intervals += [Interval(S.NegativeInfinity, nums[0], True, True)]
                for a, b in zip(nums[:-1], nums[1:]):
                    intervals.append(Interval(a, b, True, True))  # both open
                intervals.append(Interval(nums[-1], S.Infinity, True, True))
                if syms != []:
                    return Complement(Union(*intervals, evaluate=False),
                            FiniteSet(*syms), evaluate=False)
                else:
                    return Union(*intervals, evaluate=False)
            elif nums == []:  # no splitting necessary or possible:
                if syms:
                    return Complement(other, FiniteSet(*syms), evaluate=False)
                else:
                    return other

        elif isinstance(other, FiniteSet):
            unk = []
            for i in self:
                c = sympify(other.contains(i))
                if c is not S.true and c is not S.false:
                    unk.append(i)
            unk = FiniteSet(*unk)
            if unk == self:
                return
            not_true = []
            for i in other:
                c = sympify(self.contains(i))
                if c is not S.true:
                    not_true.append(i)
            return Complement(FiniteSet(*not_true), unk)

        return Set._complement(self, other)

    def _contains(self, other):
        """
        Tests whether an element, other, is in the set.

        Explanation
        ===========

        The actual test is for mathematical equality (as opposed to
        syntactical equality). In the worst case all elements of the
        set must be checked.

        Examples
        ========

        >>> from sympy import FiniteSet
        >>> 1 in FiniteSet(1, 2)
        True
        >>> 5 in FiniteSet(1, 2)
        False

        """
        if other in self._args_set:
            return S.true
        else:
            # evaluate=True is needed to override evaluate=False context;
            # we need Eq to do the evaluation
            return Or(*[Eq(e, other, evaluate=True) for e in self.args])

    def _eval_is_subset(self, other):
        return fuzzy_and(other._contains(e) for e in self.args)

    @property
    def _boundary(self):
        return self

    @property
    def _inf(self):
        return Min(*self)

    @property
    def _sup(self):
        return Max(*self)

    @property
    def measure(self):
        return 0

    def _kind(self):
        if not self.args:
            return SetKind()
        elif all(i.kind == self.args[0].kind for i in self.args):
            return SetKind(self.args[0].kind)
        else:
            return SetKind(UndefinedKind)

    def __len__(self):
        return len(self.args)

    def as_relational(self, symbol):
        """Rewrite a FiniteSet in terms of equalities and logic operators. """
        return Or(*[Eq(symbol, elem) for elem in self])

    def compare(self, other):
        return (hash(self) - hash(other))

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return FiniteSet(*[elem.evalf(n=dps) for elem in self])

    def _eval_simplify(self, **kwargs):
        from sympy.simplify import simplify
        return FiniteSet(*[simplify(elem, **kwargs) for elem in self])

    @property
    def _sorted_args(self):
        return self.args

    def _eval_powerset(self):
        return self.func(*[self.func(*s) for s in subsets(self.args)])

    def _eval_rewrite_as_PowerSet(self, *args, **kwargs):
        """Rewriting method for a finite set to a power set."""
        from .powerset import PowerSet

        is2pow = lambda n: bool(n and not n & (n - 1))
        if not is2pow(len(self)):
            return None

        fs_test = lambda arg: isinstance(arg, Set) and arg.is_FiniteSet
        if not all(fs_test(arg) for arg in args):
            return None

        biggest = max(args, key=len)
        for arg in subsets(biggest.args):
            arg_set = FiniteSet(*arg)
            if arg_set not in args:
                return None
        return PowerSet(biggest)

    def __ge__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return other.is_subset(self)

    def __gt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_superset(other)

    def __le__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_subset(other)

    def __lt__(self, other):
        if not isinstance(other, Set):
            raise TypeError("Invalid comparison of set with %s" % func_name(other))
        return self.is_proper_subset(other)

    def __eq__(self, other):
        if isinstance(other, (set, frozenset)):
            return self._args_set == other
        return super().__eq__(other)

    __hash__ : Callable[[Basic], Any] = Basic.__hash__

_sympy_converter[set] = lambda x: FiniteSet(*x)
_sympy_converter[frozenset] = lambda x: FiniteSet(*x)


class SymmetricDifference(Set):
    """Represents the set of elements which are in either of the
    sets and not in their intersection.

    Examples
    ========

    >>> from sympy import SymmetricDifference, FiniteSet
    >>> SymmetricDifference(FiniteSet(1, 2, 3), FiniteSet(3, 4, 5))
    {1, 2, 4, 5}

    See Also
    ========

    Complement, Union

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_difference
    """

    is_SymmetricDifference = True

    def __new__(cls, a, b, evaluate=True):
        if evaluate:
            return SymmetricDifference.reduce(a, b)

        return Basic.__new__(cls, a, b)

    @staticmethod
    def reduce(A, B):
        result = B._symmetric_difference(A)
        if result is not None:
            return result
        else:
            return SymmetricDifference(A, B, evaluate=False)

    def as_relational(self, symbol):
        """Rewrite a symmetric_difference in terms of equalities and
        logic operators"""
        A, B = self.args

        A_rel = A.as_relational(symbol)
        B_rel = B.as_relational(symbol)

        return Xor(A_rel, B_rel)

    @property
    def is_iterable(self):
        if all(arg.is_iterable for arg in self.args):
            return True

    def __iter__(self):

        args = self.args
        union = roundrobin(*(iter(arg) for arg in args))

        for item in union:
            count = 0
            for s in args:
                if item in s:
                    count += 1

            if count % 2 == 1:
                yield item



class DisjointUnion(Set):
    """ Represents the disjoint union (also known as the external disjoint union)
    of a finite number of sets.

    Examples
    ========

    >>> from sympy import DisjointUnion, FiniteSet, Interval, Union, Symbol
    >>> A = FiniteSet(1, 2, 3)
    >>> B = Interval(0, 5)
    >>> DisjointUnion(A, B)
    DisjointUnion({1, 2, 3}, Interval(0, 5))
    >>> DisjointUnion(A, B).rewrite(Union)
    Union(ProductSet({1, 2, 3}, {0}), ProductSet(Interval(0, 5), {1}))
    >>> C = FiniteSet(Symbol('x'), Symbol('y'), Symbol('z'))
    >>> DisjointUnion(C, C)
    DisjointUnion({x, y, z}, {x, y, z})
    >>> DisjointUnion(C, C).rewrite(Union)
    ProductSet({x, y, z}, {0, 1})

    References
    ==========

    https://en.wikipedia.org/wiki/Disjoint_union
    """

    def __new__(cls, *sets):
        dj_collection = []
        for set_i in sets:
            if isinstance(set_i, Set):
                dj_collection.append(set_i)
            else:
                raise TypeError("Invalid input: '%s', input args \
                    to DisjointUnion must be Sets" % set_i)
        obj = Basic.__new__(cls, *dj_collection)
        return obj

    @property
    def sets(self):
        return self.args

    @property
    def is_empty(self):
        return fuzzy_and(s.is_empty for s in self.sets)

    @property
    def is_finite_set(self):
        all_finite = fuzzy_and(s.is_finite_set for s in self.sets)
        return fuzzy_or([self.is_empty, all_finite])

    @property
    def is_iterable(self):
        if self.is_empty:
            return False
        iter_flag = True
        for set_i in self.sets:
            if not set_i.is_empty:
                iter_flag = iter_flag and set_i.is_iterable
        return iter_flag

    def _eval_rewrite_as_Union(self, *sets, **kwargs):
        """
        Rewrites the disjoint union as the union of (``set`` x {``i``})
        where ``set`` is the element in ``sets`` at index = ``i``
        """

        dj_union = S.EmptySet
        index = 0
        for set_i in sets:
            if isinstance(set_i, Set):
                cross = ProductSet(set_i, FiniteSet(index))
                dj_union = Union(dj_union, cross)
                index = index + 1
        return dj_union

    def _contains(self, element):
        """
        ``in`` operator for DisjointUnion

        Examples
        ========

        >>> from sympy import Interval, DisjointUnion
        >>> D = DisjointUnion(Interval(0, 1), Interval(0, 2))
        >>> (0.5, 0) in D
        True
        >>> (0.5, 1) in D
        True
        >>> (1.5, 0) in D
        False
        >>> (1.5, 1) in D
        True

        Passes operation on to constituent sets
        """
        if not isinstance(element, Tuple) or len(element) != 2:
            return S.false

        if not element[1].is_Integer:
            return S.false

        if element[1] >= len(self.sets) or element[1] < 0:
            return S.false

        return self.sets[element[1]]._contains(element[0])

    def _kind(self):
        if not self.args:
            return SetKind()
        elif all(i.kind == self.args[0].kind for i in self.args):
            return self.args[0].kind
        else:
            return SetKind(UndefinedKind)

    def __iter__(self):
        if self.is_iterable:

            iters = []
            for i, s in enumerate(self.sets):
                iters.append(iproduct(s, {Integer(i)}))

            return iter(roundrobin(*iters))
        else:
            raise ValueError("'%s' is not iterable." % self)

    def __len__(self):
        """
        Returns the length of the disjoint union, i.e., the number of elements in the set.

        Examples
        ========

        >>> from sympy import FiniteSet, DisjointUnion, EmptySet
        >>> D1 = DisjointUnion(FiniteSet(1, 2, 3, 4), EmptySet, FiniteSet(3, 4, 5))
        >>> len(D1)
        7
        >>> D2 = DisjointUnion(FiniteSet(3, 5, 7), EmptySet, FiniteSet(3, 5, 7))
        >>> len(D2)
        6
        >>> D3 = DisjointUnion(EmptySet, EmptySet)
        >>> len(D3)
        0

        Adds up the lengths of the constituent sets.
        """

        if self.is_finite_set:
            size = 0
            for set in self.sets:
                size += len(set)
            return size
        else:
            raise ValueError("'%s' is not a finite set." % self)


def imageset(*args):
    r"""
    Return an image of the set under transformation ``f``.

    Explanation
    ===========

    If this function cannot compute the image, it returns an
    unevaluated ImageSet object.

    .. math::
        \{ f(x) \mid x \in \mathrm{self} \}

    Examples
    ========

    >>> from sympy import S, Interval, imageset, sin, Lambda
    >>> from sympy.abc import x

    >>> imageset(x, 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(lambda x: 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(Lambda(x, sin(x)), Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))

    >>> imageset(sin, Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))
    >>> imageset(lambda y: x + y, Interval(-2, 1))
    ImageSet(Lambda(y, x + y), Interval(-2, 1))

    Expressions applied to the set of Integers are simplified
    to show as few negatives as possible and linear expressions
    are converted to a canonical form. If this is not desirable
    then the unevaluated ImageSet should be used.

    >>> imageset(x, -2*x + 5, S.Integers)
    ImageSet(Lambda(x, 2*x + 1), Integers)

    See Also
    ========

    sympy.sets.fancysets.ImageSet

    """
    from .fancysets import ImageSet
    from .setexpr import set_function

    if len(args) < 2:
        raise ValueError('imageset expects at least 2 args, got: %s' % len(args))

    if isinstance(args[0], (Symbol, tuple)) and len(args) > 2:
        f = Lambda(args[0], args[1])
        set_list = args[2:]
    else:
        f = args[0]
        set_list = args[1:]

    if isinstance(f, Lambda):
        pass
    elif callable(f):
        nargs = getattr(f, 'nargs', {})
        if nargs:
            if len(nargs) != 1:
                raise NotImplementedError(filldedent('''
                    This function can take more than 1 arg
                    but the potentially complicated set input
                    has not been analyzed at this point to
                    know its dimensions. TODO
                    '''))
            N = nargs.args[0]
            if N == 1:
                s = 'x'
            else:
                s = [Symbol('x%i' % i) for i in range(1, N + 1)]
        else:
            s = inspect.signature(f).parameters

        dexpr = _sympify(f(*[Dummy() for i in s]))
        var = tuple(uniquely_named_symbol(
            Symbol(i), dexpr) for i in s)
        f = Lambda(var, f(*var))
    else:
        raise TypeError(filldedent('''
            expecting lambda, Lambda, or FunctionClass,
            not \'%s\'.''' % func_name(f)))

    if any(not isinstance(s, Set) for s in set_list):
        name = [func_name(s) for s in set_list]
        raise ValueError(
            'arguments after mapping should be sets, not %s' % name)

    if len(set_list) == 1:
        set = set_list[0]
        try:
            # TypeError if arg count != set dimensions
            r = set_function(f, set)
            if r is None:
                raise TypeError
            if not r:
                return r
        except TypeError:
            r = ImageSet(f, set)
        if isinstance(r, ImageSet):
            f, set = r.args

        if f.variables[0] == f.expr:
            return set

        if isinstance(set, ImageSet):
            # XXX: Maybe this should just be:
            # f2 = set.lambda
            # fun = Lambda(f2.signature, f(*f2.expr))
            # return imageset(fun, *set.base_sets)
            if len(set.lamda.variables) == 1 and len(f.variables) == 1:
                x = set.lamda.variables[0]
                y = f.variables[0]
                return imageset(
                    Lambda(x, f.expr.subs(y, set.lamda.expr)), *set.base_sets)

        if r is not None:
            return r

    return ImageSet(f, *set_list)


def is_function_invertible_in_set(func, setv):
    """
    Checks whether function ``func`` is invertible when the domain is
    restricted to set ``setv``.
    """
    # Functions known to always be invertible:
    if func in (exp, log):
        return True
    u = Dummy("u")
    fdiff = func(u).diff(u)
    # monotonous functions:
    # TODO: check subsets (`func` in `setv`)
    if (fdiff > 0) == True or (fdiff < 0) == True:
        return True
    # TODO: support more
    return None


def simplify_union(args):
    """
    Simplify a :class:`Union` using known rules.

    Explanation
    ===========

    We first start with global rules like 'Merge all FiniteSets'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent.  This process depends
    on ``union_sets(a, b)`` functions.
    """
    from sympy.sets.handlers.union import union_sets

    # ===== Global Rules =====
    if not args:
        return S.EmptySet

    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")

    # Merge all finite sets
    finite_sets = [x for x in args if x.is_FiniteSet]
    if len(finite_sets) > 1:
        a = (x for set in finite_sets for x in set)
        finite_set = FiniteSet(*a)
        args = [finite_set] + [x for x in args if not x.is_FiniteSet]

    # ===== Pair-wise Rules =====
    # Here we depend on rules built into the constituent sets
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = union_sets(s, t)
                # This returns None if s does not know how to intersect
                # with t. Returns the newly intersected set otherwise
                if new_set is not None:
                    if not isinstance(new_set, set):
                        new_set = {new_set}
                    new_args = (args - {s, t}).union(new_set)
                    break
            if new_args:
                args = new_args
                break

    if len(args) == 1:
        return args.pop()
    else:
        return Union(*args, evaluate=False)


def simplify_intersection(args):
    """
    Simplify an intersection using known rules.

    Explanation
    ===========

    We first start with global rules like
    'if any empty sets return empty set' and 'distribute any unions'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent
    """

    # ===== Global Rules =====
    if not args:
        return S.UniversalSet

    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError("Input args to Union must be Sets")

    # If any EmptySets return EmptySet
    if S.EmptySet in args:
        return S.EmptySet

    # Handle Finite sets
    rv = Intersection._handle_finite_sets(args)

    if rv is not None:
        return rv

    # If any of the sets are unions, return a Union of Intersections
    for s in args:
        if s.is_Union:
            other_sets = set(args) - {s}
            if len(other_sets) > 0:
                other = Intersection(*other_sets)
                return Union(*(Intersection(arg, other) for arg in s.args))
            else:
                return Union(*s.args)

    for s in args:
        if s.is_Complement:
            args.remove(s)
            other_sets = args + [s.args[0]]
            return Complement(Intersection(*other_sets), s.args[1])

    from sympy.sets.handlers.intersection import intersection_sets

    # At this stage we are guaranteed not to have any
    # EmptySets, FiniteSets, or Unions in the intersection

    # ===== Pair-wise Rules =====
    # Here we depend on rules built into the constituent sets
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = intersection_sets(s, t)
                # This returns None if s does not know how to intersect
                # with t. Returns the newly intersected set otherwise

                if new_set is not None:
                    new_args = (args - {s, t}).union({new_set})
                    break
            if new_args:
                args = new_args
                break

    if len(args) == 1:
        return args.pop()
    else:
        return Intersection(*args, evaluate=False)


def _handle_finite_sets(op, x, y, commutative):
    # Handle finite sets:
    fs_args, other = sift([x, y], lambda x: isinstance(x, FiniteSet), binary=True)
    if len(fs_args) == 2:
        return FiniteSet(*[op(i, j) for i in fs_args[0] for j in fs_args[1]])
    elif len(fs_args) == 1:
        sets = [_apply_operation(op, other[0], i, commutative) for i in fs_args[0]]
        return Union(*sets)
    else:
        return None


def _apply_operation(op, x, y, commutative):
    from .fancysets import ImageSet
    d = Dummy('d')

    out = _handle_finite_sets(op, x, y, commutative)
    if out is None:
        out = op(x, y)

    if out is None and commutative:
        out = op(y, x)
    if out is None:
        _x, _y = symbols("x y")
        if isinstance(x, Set) and not isinstance(y, Set):
            out = ImageSet(Lambda(d, op(d, y)), x).doit()
        elif not isinstance(x, Set) and isinstance(y, Set):
            out = ImageSet(Lambda(d, op(x, d)), y).doit()
        else:
            out = ImageSet(Lambda((_x, _y), op(_x, _y)), x, y)
    return out


def set_add(x, y):
    from sympy.sets.handlers.add import _set_add
    return _apply_operation(_set_add, x, y, commutative=True)


def set_sub(x, y):
    from sympy.sets.handlers.add import _set_sub
    return _apply_operation(_set_sub, x, y, commutative=False)


def set_mul(x, y):
    from sympy.sets.handlers.mul import _set_mul
    return _apply_operation(_set_mul, x, y, commutative=True)


def set_div(x, y):
    from sympy.sets.handlers.mul import _set_div
    return _apply_operation(_set_div, x, y, commutative=False)


def set_pow(x, y):
    from sympy.sets.handlers.power import _set_pow
    return _apply_operation(_set_pow, x, y, commutative=False)


def set_function(f, x):
    from sympy.sets.handlers.functions import _set_function
    return _set_function(f, x)


class SetKind(Kind):
    """
    SetKind is kind for all Sets

    Every instance of Set will have kind ``SetKind`` parametrised by the kind
    of the elements of the ``Set``. The kind of the elements might be
    ``NumberKind``, or ``TupleKind`` or something else. When not all elements
    have the same kind then the kind of the elements will be given as
    ``UndefinedKind``.

    Parameters
    ==========

    element_kind: Kind (optional)
        The kind of the elements of the set. In a well defined set all elements
        will have the same kind. Otherwise the kind should
        :class:`sympy.core.kind.UndefinedKind`. The ``element_kind`` argument is optional but
        should only be omitted in the case of ``EmptySet`` whose kind is simply
        ``SetKind()``

    Examples
    ========

    >>> from sympy import Interval
    >>> Interval(1, 2).kind
    SetKind(NumberKind)
    >>> Interval(1,2).kind.element_kind
    NumberKind

    See Also
    ========

    sympy.core.kind.NumberKind
    sympy.matrices.kind.MatrixKind
    sympy.core.containers.TupleKind
    """
    def __new__(cls, element_kind=None):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        if not self.element_kind:
            return "SetKind()"
        else:
            return "SetKind(%s)" % self.element_kind

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\error_functions.py ===
""" This module contains various functions that are special cases
    of incomplete gamma functions. It should probably be renamed. """

from sympy.core import EulerGamma # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import DefinedFunction, ArgumentIndexError, expand_mul
from sympy.core.logic import fuzzy_or
from sympy.core.numbers import I, pi, Rational, Integer
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg

# TODO series expansions
# TODO see the "Note:" in Ei

# Helper function
def real_to_real_as_real_imag(self, deep=True, **hints):
    if self.args[0].is_extended_real:
        if deep:
            hints['complex'] = False
            return (self.expand(deep, **hints), S.Zero)
        else:
            return (self, S.Zero)
    if deep:
        x, y = self.args[0].expand(deep, **hints).as_real_imag()
    else:
        x, y = self.args[0].as_real_imag()
    re = (self.func(x + I*y) + self.func(x - I*y))/2
    im = (self.func(x + I*y) - self.func(x - I*y))/(2*I)
    return (re, im)


###############################################################################
################################ ERROR FUNCTION ###############################
###############################################################################


class erf(DefinedFunction):
    r"""
    The Gauss error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \mathrm{d}t.

    Examples
    ========

    >>> from sympy import I, oo, erf
    >>> from sympy.abc import z

    Several special values are known:

    >>> erf(0)
    0
    >>> erf(oo)
    1
    >>> erf(-oo)
    -1
    >>> erf(I*oo)
    oo*I
    >>> erf(-I*oo)
    -oo*I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erf(-z)
    -erf(z)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf(z))
    erf(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erf(z), z)
    2*exp(-z**2)/sqrt(pi)

    We can numerically evaluate the error function to arbitrary precision
    on the whole complex plane:

    >>> erf(4).evalf(30)
    0.999999984582742099719981147840

    >>> erf(-4*I).evalf(30)
    -1296959.73071763923152794095062*I

    See Also
    ========

    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erf.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erf

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*exp(-self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfinv

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.Zero

        if isinstance(arg, erfinv):
            return arg.args[0]

        if isinstance(arg, erfcinv):
            return S.One - arg.args[0]

        if arg.is_zero:
            return S.Zero

        # Only happens with unevaluated erf2inv
        if isinstance(arg, erf2inv) and arg.args[0].is_zero:
            return arg.args[1]

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return arg

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return 2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        if self.args[0].is_extended_real is True:
            return True
        # There are cases where erf(z) becomes a real number
        # even if z is a complex number

    def _eval_is_imaginary(self):
        if self.args[0].is_imaginary is True:
            return True

    def _eval_is_finite(self):
        z = self.args[0]
        return fuzzy_or([z.is_finite, z.is_extended_real])

    def _eval_is_zero(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_zero

    def _eval_is_positive(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_extended_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real is True:
            return self.args[0].is_extended_negative

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(z**2)/z - z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        from sympy.series.limits import limit
        if limitvar:
            lim = limit(z, limitvar, S.Infinity)
            if lim is S.NegativeInfinity:
                return S.NegativeOne + _erfs(-z)*exp(-z**2)
        return S.One - _erfs(z)*exp(-z**2)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return S.One - erfc(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return -I*erfi(I*z)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if x in arg.free_symbols and arg0.is_zero:
            return 2*arg/sqrt(pi)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in [S.Infinity, S.NegativeInfinity]:
            z = self.args[0]

            try:
                _, ex = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self

            ex = -ex # as x->1/x for aseries
            if ex.is_positive:
                newn = ceiling(n/ex)
                s = [S.NegativeOne**k * factorial2(2*k - 1) / (z**(2*k + 1) * 2**k)
                     for k in range(newn)] + [Order(1/z**newn, x)]
                return S.One - (exp(-z**2)/sqrt(pi)) * Add(*s)

        return super(erf, self)._eval_aseries(n, args0, x, logx)

    as_real_imag = real_to_real_as_real_imag


class erfc(DefinedFunction):
    r"""
    Complementary Error Function.

    Explanation
    ===========

    The function is defined as:

    .. math ::
        \mathrm{erfc}(x) = \frac{2}{\sqrt{\pi}} \int_x^\infty e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfc
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfc(0)
    1
    >>> erfc(oo)
    0
    >>> erfc(-oo)
    2
    >>> erfc(I*oo)
    -oo*I
    >>> erfc(-I*oo)
    oo*I

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erfc(z))
    erfc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfc(z), z)
    -2*exp(-z**2)/sqrt(pi)

    It also follows

    >>> erfc(-z)
    2 - erfc(z)

    We can numerically evaluate the complementary error function to arbitrary
    precision on the whole complex plane:

    >>> erfc(4).evalf(30)
    0.0000000154172579002800188521596734869

    >>> erfc(4*I).evalf(30)
    1.0 - 1296959.73071763923152794095062*I

    See Also
    ========

    erf: Gaussian error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erfc.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erfc

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -2*exp(-self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfcinv

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg.is_zero:
                return S.One

        if isinstance(arg, erfinv):
            return S.One - arg.args[0]

        if isinstance(arg, erfcinv):
            return arg.args[0]

        if arg.is_zero:
            return S.One

        # Try to pull out factors of I
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return -arg

        # Try to pull out factors of -1
        if arg.could_extract_minus_sign():
            return 2 - cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return S.One
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return -2*S.NegativeOne**k * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        if self.args[0].is_extended_real is True:
            return True
        if self.args[0].is_imaginary is True:
            return False

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return S.One - erf(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return S.One + I*erfi(I*z)

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One-I)*z/sqrt(pi)
        return S.One - (S.One + I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return S.One - z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return S.One - 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], -z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return S.One - sqrt(z**2)/z*(S.One - uppergamma(S.Half, z**2)/sqrt(pi))

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if arg0.is_zero:
            return S.One
        else:
            return self.func(arg0)

    as_real_imag = real_to_real_as_real_imag

    def _eval_aseries(self, n, args0, x, logx):
        return S.One - erf(*self.args)._eval_aseries(n, args0, x, logx)


class erfi(DefinedFunction):
    r"""
    Imaginary error function.

    Explanation
    ===========

    The function erfi is defined as:

    .. math ::
        \mathrm{erfi}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfi
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfi(0)
    0
    >>> erfi(oo)
    oo
    >>> erfi(-oo)
    -oo
    >>> erfi(I*oo)
    I
    >>> erfi(-I*oo)
    -I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erfi(-z)
    -erfi(z)

    >>> from sympy import conjugate
    >>> conjugate(erfi(z))
    erfi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfi(z), z)
    2*exp(z**2)/sqrt(pi)

    We can numerically evaluate the imaginary error function to arbitrary
    precision on the whole complex plane:

    >>> erfi(2).evalf(30)
    18.5648024145755525987042919132

    >>> erfi(-2*I).evalf(30)
    -0.995322265018952734162069256367*I

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://mathworld.wolfram.com/Erfi.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/Erfi

    """

    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2*exp(self.args[0]**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, z):
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z.is_zero:
                return S.Zero
            elif z is S.Infinity:
                return S.Infinity

        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(-z)

        # Try to pull out factors of I
        nz = z.extract_multiplicatively(I)
        if nz is not None:
            if nz is S.Infinity:
                return I
            if isinstance(nz, erfinv):
                return I*nz.args[0]
            if isinstance(nz, erfcinv):
                return I*(S.One - nz.args[0])
            # Only happens with unevaluated erf2inv
            if isinstance(nz, erf2inv) and nz.args[0].is_zero:
                return I*nz.args[1]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1)/S(2))
            if len(previous_terms) > 2:
                return previous_terms[-2] * x**2 * (n - 2)/(n*k)
            else:
                return 2 * x**n/(n*factorial(k)*sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite("tractable", deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return -I*erf(I*z)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return I*erfc(I*z) - I

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One + I)*z/sqrt(pi)
        return (S.One - I)*(fresnelc(arg) - I*fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z/sqrt(pi)*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2*z/sqrt(pi)*hyper([S.Half], [3*S.Half], z**2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(-z**2)/z*(uppergamma(S.Half, -z**2)/sqrt(pi) - S.One)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    as_real_imag = real_to_real_as_real_imag

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if x in arg.free_symbols and arg0.is_zero:
            return 2*arg/sqrt(pi)
        elif arg0.is_finite:
            return self.func(arg0)
        return self.func(arg)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point is S.Infinity:
            z = self.args[0]
            s = [factorial2(2*k - 1) / (2**k * z**(2*k + 1))
                    for k in range(n)] + [Order(1/z**n, x)]
            return -I + (exp(z**2)/sqrt(pi)) * Add(*s)

        return super(erfi, self)._eval_aseries(n, args0, x, logx)


class erf2(DefinedFunction):
    r"""
    Two-argument error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \mathrm{erf2}(x, y) = \frac{2}{\sqrt{\pi}} \int_x^y e^{-t^2} \mathrm{d}t

    Examples
    ========

    >>> from sympy import oo, erf2
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2(0, 0)
    0
    >>> erf2(x, x)
    0
    >>> erf2(x, oo)
    1 - erf(x)
    >>> erf2(x, -oo)
    -erf(x) - 1
    >>> erf2(oo, y)
    erf(y) - 1
    >>> erf2(-oo, y)
    erf(y) + 1

    In general one can pull out factors of -1:

    >>> erf2(-x, -y)
    -erf2(x, y)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf2(x, y))
    erf2(conjugate(x), conjugate(y))

    Differentiation with respect to $x$, $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2(x, y), x)
    -2*exp(-x**2)/sqrt(pi)
    >>> diff(erf2(x, y), y)
    2*exp(-y**2)/sqrt(pi)

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/Erf2/

    """


    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return -2*exp(-x**2)/sqrt(pi)
        elif argindex == 2:
            return 2*exp(-y**2)/sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        chk = (S.Infinity, S.NegativeInfinity, S.Zero)
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x == y:
            return S.Zero
        elif x in chk or y in chk:
            return erf(y) - erf(x)

        if isinstance(y, erf2inv) and y.args[0] == x:
            return y.args[1]

        if x.is_zero or y.is_zero or x.is_extended_real and x.is_infinite or \
                y.is_extended_real and y.is_infinite:
            return erf(y) - erf(x)

        #Try to pull out -1 factor
        sign_x = x.could_extract_minus_sign()
        sign_y = y.could_extract_minus_sign()
        if (sign_x and sign_y):
            return -cls(-x, -y)
        elif (sign_x or sign_y):
            return erf(y)-erf(x)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_rewrite_as_erf(self, x, y, **kwargs):
        return erf(y) - erf(x)

    def _eval_rewrite_as_erfc(self, x, y, **kwargs):
        return erfc(x) - erfc(y)

    def _eval_rewrite_as_erfi(self, x, y, **kwargs):
        return I*(erfi(I*x)-erfi(I*y))

    def _eval_rewrite_as_fresnels(self, x, y, **kwargs):
        return erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)

    def _eval_rewrite_as_fresnelc(self, x, y, **kwargs):
        return erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)

    def _eval_rewrite_as_meijerg(self, x, y, **kwargs):
        return erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)

    def _eval_rewrite_as_hyper(self, x, y, **kwargs):
        return erf(y).rewrite(hyper) - erf(x).rewrite(hyper)

    def _eval_rewrite_as_uppergamma(self, x, y, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return (sqrt(y**2)/y*(S.One - uppergamma(S.Half, y**2)/sqrt(pi)) -
            sqrt(x**2)/x*(S.One - uppergamma(S.Half, x**2)/sqrt(pi)))

    def _eval_rewrite_as_expint(self, x, y, **kwargs):
        return erf(y).rewrite(expint) - erf(x).rewrite(expint)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_is_zero(self):
        return is_eq(*self.args)

class erfinv(DefinedFunction):
    r"""
    Inverse Error Function. The erfinv function is defined as:

    .. math ::
        \mathrm{erf}(x) = y \quad \Rightarrow \quad \mathrm{erfinv}(y) = x

    Examples
    ========

    >>> from sympy import erfinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfinv(0)
    0
    >>> erfinv(1)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfinv(x), x)
    sqrt(pi)*exp(erfinv(x)**2)/2

    We can numerically evaluate the inverse error function to arbitrary
    precision on [-1, 1]:

    >>> erfinv(0.2).evalf(30)
    0.179143454621291692285822705344

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErf/

    """


    def fdiff(self, argindex =1):
        if argindex == 1:
            return sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else :
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erf

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z is S.NegativeOne:
            return S.NegativeInfinity
        elif z.is_zero:
            return S.Zero
        elif z is S.One:
            return S.Infinity

        if isinstance(z, erf) and z.args[0].is_extended_real:
            return z.args[0]

        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1
        nz = z.extract_multiplicatively(-1)
        if nz is not None and (isinstance(nz, erf) and (nz.args[0]).is_extended_real):
            return -nz.args[0]

    def _eval_rewrite_as_erfcinv(self, z, **kwargs):
        return erfcinv(1-z)

    def _eval_is_zero(self):
        return self.args[0].is_zero


class erfcinv (DefinedFunction):
    r"""
    Inverse Complementary Error Function. The erfcinv function is defined as:

    .. math ::
        \mathrm{erfc}(x) = y \quad \Rightarrow \quad \mathrm{erfcinv}(y) = x

    Examples
    ========

    >>> from sympy import erfcinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfcinv(1)
    0
    >>> erfcinv(0)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfcinv(x), x)
    -sqrt(pi)*exp(erfcinv(x)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErfc/

    """


    def fdiff(self, argindex =1):
        if argindex == 1:
            return -sqrt(pi)*exp(self.func(self.args[0])**2)*S.Half
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfc

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z.is_zero:
            return S.Infinity
        elif z is S.One:
            return S.Zero
        elif z == 2:
            return S.NegativeInfinity

        if z.is_zero:
            return S.Infinity

    def _eval_rewrite_as_erfinv(self, z, **kwargs):
        return erfinv(1-z)

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_infinite(self):
        z = self.args[0]
        return fuzzy_or([z.is_zero, is_eq(z, Integer(2))])


class erf2inv(DefinedFunction):
    r"""
    Two-argument Inverse error function. The erf2inv function is defined as:

    .. math ::
        \mathrm{erf2}(x, w) = y \quad \Rightarrow \quad \mathrm{erf2inv}(x, y) = w

    Examples
    ========

    >>> from sympy import erf2inv, oo
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2inv(0, 0)
    0
    >>> erf2inv(1, 0)
    1
    >>> erf2inv(0, 1)
    oo
    >>> erf2inv(0, y)
    erfinv(y)
    >>> erf2inv(oo, y)
    erfcinv(-y)

    Differentiation with respect to $x$ and $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2inv(x, y), x)
    exp(-x**2 + erf2inv(x, y)**2)
    >>> diff(erf2inv(x, y), y)
    sqrt(pi)*exp(erf2inv(x, y)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse complementary error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/InverseErf2/

    """


    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return exp(self.func(x,y)**2-x**2)
        elif argindex == 2:
            return sqrt(pi)*S.Half*exp(self.func(x,y)**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x.is_zero and y.is_zero:
            return S.Zero
        elif x.is_zero and y is S.One:
            return S.Infinity
        elif x is S.One and y.is_zero:
            return S.One
        elif x.is_zero:
            return erfinv(y)
        elif x is S.Infinity:
            return erfcinv(-y)
        elif y.is_zero:
            return x
        elif y is S.Infinity:
            return erfinv(x)

        if x.is_zero:
            if y.is_zero:
                return S.Zero
            else:
                return erfinv(y)
        if y.is_zero:
            return x

    def _eval_is_zero(self):
        x, y = self.args
        if x.is_zero and y.is_zero:
            return True

###############################################################################
#################### EXPONENTIAL INTEGRALS ####################################
###############################################################################

class Ei(DefinedFunction):
    r"""
    The classical exponential integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Ei}(x) = \sum_{n=1}^\infty \frac{x^n}{n\, n!}
                                     + \log(x) + \gamma,

    where $\gamma$ is the Euler-Mascheroni constant.

    If $x$ is a polar number, this defines an analytic function on the
    Riemann surface of the logarithm. Otherwise this defines an analytic
    function in the cut plane $\mathbb{C} \setminus (-\infty, 0]$.

    **Background**

    The name exponential integral comes from the following statement:

    .. math:: \operatorname{Ei}(x) = \int_{-\infty}^x \frac{e^t}{t} \mathrm{d}t

    If the integral is interpreted as a Cauchy principal value, this statement
    holds for $x > 0$ and $\operatorname{Ei}(x)$ as defined above.

    Examples
    ========

    >>> from sympy import Ei, polar_lift, exp_polar, I, pi
    >>> from sympy.abc import x

    >>> Ei(-1)
    Ei(-1)

    This yields a real value:

    >>> Ei(-1).n(chop=True)
    -0.219383934395520

    On the other hand the analytic continuation is not real:

    >>> Ei(polar_lift(-1)).n(chop=True)
    -0.21938393439552 + 3.14159265358979*I

    The exponential integral has a logarithmic branch point at the origin:

    >>> Ei(x*exp_polar(2*I*pi))
    Ei(x) + 2*I*pi

    Differentiation is supported:

    >>> Ei(x).diff(x)
    exp(x)/x

    The exponential integral is related to many other special functions.
    For example:

    >>> from sympy import expint, Shi
    >>> Ei(x).rewrite(expint)
    -expint(1, x*exp_polar(I*pi)) - I*pi
    >>> Ei(x).rewrite(Shi)
    Chi(x) + Shi(x)

    See Also
    ========

    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma: Upper incomplete gamma function.

    References
    ==========

    .. [1] https://dlmf.nist.gov/6.6
    .. [2] https://en.wikipedia.org/wiki/Exponential_integral
    .. [3] Abramowitz & Stegun, section 5: https://web.archive.org/web/20201128173312/http://people.math.sfu.ca/~cbm/aands/page_228.htm

    """


    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        elif z is S.NegativeInfinity:
            return S.Zero

        if z.is_zero:
            return S.NegativeInfinity

        nz, n = z.extract_branch_factor()
        if n:
            return Ei(nz) + 2*I*pi*n

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return exp(arg)/arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        if (self.args[0]/polar_lift(-1)).is_positive:
            return super()._eval_evalf(prec) + (I*pi)._eval_evalf(prec)
        return super()._eval_evalf(prec)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        # XXX this does not currently work usefully because uppergamma
        #     immediately turns into expint
        return -uppergamma(0, polar_lift(-1)*z) - I*pi

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -expint(1, polar_lift(-1)*z) - I*pi

    def _eval_rewrite_as_li(self, z, **kwargs):
        if isinstance(z, log):
            return li(z.args[0])
        # TODO:
        # Actually it only holds that:
        #  Ei(z) = li(exp(z))
        # for -pi < imag(z) <= pi
        return li(exp(z))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        if z.is_negative:
            return Shi(z) + Chi(z) - I*pi
        else:
            return Shi(z) + Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(z) * _eis(z)

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(S.Exp1**t/t, (t, S.NegativeInfinity, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy import re
        x0 = self.args[0].limit(x, 0)
        arg = self.args[0].as_leading_term(x, cdir=cdir)
        cdir = arg.dir(x, cdir)
        if x0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma - (
                I*pi if re(cdir).is_negative else S.Zero)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_Si(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in (S.Infinity, S.NegativeInfinity):
            z = self.args[0]
            s = [factorial(k) / (z)**k for k in range(n)] + \
                    [Order(1/z**n, x)]
            return (exp(z)/z) * Add(*s)

        return super(Ei, self)._eval_aseries(n, args0, x, logx)


class expint(DefinedFunction):
    r"""
    Generalized exponential integral.

    Explanation
    ===========

    This function is defined as

    .. math:: \operatorname{E}_\nu(z) = z^{\nu - 1} \Gamma(1 - \nu, z),

    where $\Gamma(1 - \nu, z)$ is the upper incomplete gamma function
    (``uppergamma``).

    Hence for $z$ with positive real part we have

    .. math:: \operatorname{E}_\nu(z)
              =   \int_1^\infty \frac{e^{-zt}}{t^\nu} \mathrm{d}t,

    which explains the name.

    The representation as an incomplete gamma function provides an analytic
    continuation for $\operatorname{E}_\nu(z)$. If $\nu$ is a
    non-positive integer, the exponential integral is thus an unbranched
    function of $z$, otherwise there is a branch point at the origin.
    Refer to the incomplete gamma function documentation for details of the
    branching behavior.

    Examples
    ========

    >>> from sympy import expint, S
    >>> from sympy.abc import nu, z

    Differentiation is supported. Differentiation with respect to $z$ further
    explains the name: for integral orders, the exponential integral is an
    iterated integral of the exponential function.

    >>> expint(nu, z).diff(z)
    -expint(nu - 1, z)

    Differentiation with respect to $\nu$ has no classical expression:

    >>> expint(nu, z).diff(nu)
    -z**(nu - 1)*meijerg(((), (1, 1)), ((0, 0, 1 - nu), ()), z)

    At non-postive integer orders, the exponential integral reduces to the
    exponential function:

    >>> expint(0, z)
    exp(-z)/z
    >>> expint(-1, z)
    exp(-z)/z + exp(-z)/z**2

    At half-integers it reduces to error functions:

    >>> expint(S(1)/2, z)
    sqrt(pi)*erfc(sqrt(z))/sqrt(z)

    At positive integer orders it can be rewritten in terms of exponentials
    and ``expint(1, z)``. Use ``expand_func()`` to do this:

    >>> from sympy import expand_func
    >>> expand_func(expint(5, z))
    z**4*expint(1, z)/24 + (-z**3 + z**2 - 2*z + 6)*exp(-z)/24

    The generalised exponential integral is essentially equivalent to the
    incomplete gamma function:

    >>> from sympy import uppergamma
    >>> expint(nu, z).rewrite(uppergamma)
    z**(nu - 1)*uppergamma(1 - nu, z)

    As such it is branched at the origin:

    >>> from sympy import exp_polar, pi, I
    >>> expint(4, z*exp_polar(2*pi*I))
    I*pi*z**3/3 + expint(4, z)
    >>> expint(nu, z*exp_polar(2*pi*I))
    z**(nu - 1)*(exp(2*I*pi*nu) - 1)*gamma(1 - nu) + expint(nu, z)

    See Also
    ========

    Ei: Another related function called exponential integral.
    E1: The classical case, returns expint(1, z).
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma

    References
    ==========

    .. [1] https://dlmf.nist.gov/8.19
    .. [2] https://functions.wolfram.com/GammaBetaErf/ExpIntegralE/
    .. [3] https://en.wikipedia.org/wiki/Exponential_integral

    """


    @classmethod
    def eval(cls, nu, z):
        from sympy.functions.special.gamma_functions import (gamma, uppergamma)
        nu2 = unpolarify(nu)
        if nu != nu2:
            return expint(nu2, z)
        if nu.is_Integer and nu <= 0 or (not nu.is_Integer and (2*nu).is_Integer):
            return unpolarify(expand_mul(z**(nu - 1)*uppergamma(1 - nu, z)))

        # Extract branching information. This can be deduced from what is
        # explained in lowergamma.eval().
        z, n = z.extract_branch_factor()
        if n is S.Zero:
            return
        if nu.is_integer:
            if not nu > 0:
                return
            return expint(nu, z) \
                - 2*pi*I*n*S.NegativeOne**(nu - 1)/factorial(nu - 1)*unpolarify(z)**(nu - 1)
        else:
            return (exp(2*I*pi*nu*n) - 1)*z**(nu - 1)*gamma(1 - nu) + expint(nu, z)

    def fdiff(self, argindex):
        nu, z = self.args
        if argindex == 1:
            return -z**(nu - 1)*meijerg([], [1, 1], [0, 0, 1 - nu], [], z)
        elif argindex == 2:
            return -expint(nu - 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_uppergamma(self, nu, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return z**(nu - 1)*uppergamma(1 - nu, z)

    def _eval_rewrite_as_Ei(self, nu, z, **kwargs):
        if nu == 1:
            return -Ei(z*exp_polar(-I*pi)) - I*pi
        elif nu.is_Integer and nu > 1:
            # DLMF, 8.19.7
            x = -unpolarify(z)
            return x**(nu - 1)/factorial(nu - 1)*E1(z).rewrite(Ei) + \
                exp(x)/factorial(nu - 1) * \
                Add(*[factorial(nu - k - 2)*x**k for k in range(nu - 1)])
        else:
            return self

    def _eval_expand_func(self, **hints):
        return self.rewrite(Ei).rewrite(expint, **hints)

    def _eval_rewrite_as_Si(self, nu, z, **kwargs):
        if nu != 1:
            return self
        return Shi(z) - Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_nseries(self, x, n, logx, cdir=0):
        if not self.args[0].has(x):
            nu = self.args[0]
            if nu == 1:
                f = self._eval_rewrite_as_Si(*self.args)
                return f._eval_nseries(x, n, logx)
            elif nu.is_Integer and nu > 1:
                f = self._eval_rewrite_as_Ei(*self.args)
                return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[1]
        nu = self.args[0]

        if point is S.Infinity:
            z = self.args[1]
            s = [S.NegativeOne**k * RisingFactorial(nu, k) / z**k for k in range(n)] + [Order(1/z**n, x)]
            return (exp(-z)/z) * Add(*s)

        return super(expint, self)._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        n, x = self.args
        t = Dummy(uniquely_named_symbol('t', args).name)
        return Integral(t**-n * exp(-t*x), (t, 1, S.Infinity))


def E1(z):
    """
    Classical case of the generalized exponential integral.

    Explanation
    ===========

    This is equivalent to ``expint(1, z)``.

    Examples
    ========

    >>> from sympy import E1
    >>> E1(0)
    expint(1, 0)

    >>> E1(5)
    expint(1, 5)

    See Also
    ========

    Ei: Exponential integral.
    expint: Generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    """
    return expint(1, z)


class li(DefinedFunction):
    r"""
    The classical logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{li}(x) = \int_0^x \frac{1}{\log(t)} \mathrm{d}t \,.

    Examples
    ========

    >>> from sympy import I, oo, li
    >>> from sympy.abc import z

    Several special values are known:

    >>> li(0)
    0
    >>> li(1)
    -oo
    >>> li(oo)
    oo

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(li(z), z)
    1/log(z)

    Defining the ``li`` function via an integral:
    >>> from sympy import integrate
    >>> integrate(li(z))
    z*li(z) - Ei(2*log(z))

    >>> integrate(li(z),z)
    z*li(z) - Ei(2*log(z))


    The logarithmic integral can also be defined in terms of ``Ei``:

    >>> from sympy import Ei
    >>> li(z).rewrite(Ei)
    Ei(log(z))
    >>> diff(li(z).rewrite(Ei), z)
    1/log(z)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> li(2).evalf(30)
    1.04516378011749278484458888919

    >>> li(2*I).evalf(30)
    1.0652795784357498247001125598 + 3.08346052231061726610939702133*I

    We can even compute Soldner's constant by the help of mpmath:

    >>> from mpmath import findroot
    >>> findroot(li, 2)
    1.45136923488338

    Further transformations include rewriting ``li`` in terms of
    the trigonometric integrals ``Si``, ``Ci``, ``Shi`` and ``Chi``:

    >>> from sympy import Si, Ci, Shi, Chi
    >>> li(z).rewrite(Si)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Ci)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Shi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))
    >>> li(z).rewrite(Chi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))

    See Also
    ========

    Li: Offset logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6
    .. [4] https://mathworld.wolfram.com/SoldnersConstant.html

    """


    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.Zero
        elif z is S.One:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        if z.is_zero:
            return S.Zero

    def fdiff(self, argindex=1):
        arg = self.args[0]
        if argindex == 1:
            return S.One / log(arg)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        z = self.args[0]
        # Exclude values on the branch cut (-oo, 0)
        if not z.is_extended_negative:
            return self.func(z.conjugate())

    def _eval_rewrite_as_Li(self, z, **kwargs):
        return Li(z) + li(2)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return Ei(log(z))

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return (-uppergamma(0, -log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) - log(-log(z)))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        return (Ci(I*log(z)) - I*Si(I*log(z)) -
                S.Half*(log(S.One/log(z)) - log(log(z))) - log(I*log(z)))

    _eval_rewrite_as_Ci = _eval_rewrite_as_Si

    def _eval_rewrite_as_Shi(self, z, **kwargs):
        return (Chi(log(z)) - Shi(log(z)) - S.Half*(log(S.One/log(z)) - log(log(z))))

    _eval_rewrite_as_Chi = _eval_rewrite_as_Shi

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return (log(z)*hyper((1, 1), (2, 2), log(z)) +
                S.Half*(log(log(z)) - log(S.One/log(z))) + EulerGamma)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (-log(-log(z)) - S.Half*(log(S.One/log(z)) - log(log(z)))
                - meijerg(((), (1,)), ((0, 0), ()), -log(z)))

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return z * _eis(log(z))

    def _eval_nseries(self, x, n, logx, cdir=0):
        z = self.args[0]
        s = [(log(z))**k / (factorial(k) * k) for k in range(1, n)]
        return EulerGamma + log(log(z)) + Add(*s)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True

class Li(DefinedFunction):
    r"""
    The offset logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \operatorname{Li}(x) = \operatorname{li}(x) - \operatorname{li}(2)

    Examples
    ========

    >>> from sympy import Li
    >>> from sympy.abc import z

    The following special value is known:

    >>> Li(2)
    0

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(Li(z), z)
    1/log(z)

    The shifted logarithmic integral can be written in terms of $li(z)$:

    >>> from sympy import li
    >>> Li(z).rewrite(li)
    li(z) - li(2)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> Li(2).evalf(30)
    0

    >>> Li(4).evalf(30)
    1.92242131492155809316615998938

    See Also
    ========

    li: Logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6

    """


    @classmethod
    def eval(cls, z):
        if z is S.Infinity:
            return S.Infinity
        elif z == S(2):
            return S.Zero

    def fdiff(self, argindex=1):
        arg = self.args[0]
        if argindex == 1:
            return S.One / log(arg)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        return self.rewrite(li).evalf(prec)

    def _eval_rewrite_as_li(self, z, **kwargs):
        return li(z) - li(2)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(li).rewrite("tractable", deep=True)

    def _eval_nseries(self, x, n, logx, cdir=0):
        f = self._eval_rewrite_as_li(*self.args)
        return f._eval_nseries(x, n, logx)

###############################################################################
#################### TRIGONOMETRIC INTEGRALS ##################################
###############################################################################

class TrigonometricIntegral(DefinedFunction):
    """ Base class for trigonometric integrals. """


    @classmethod
    def eval(cls, z):
        if z is S.Zero:
            return cls._atzero
        elif z is S.Infinity:
            return cls._atinf()
        elif z is S.NegativeInfinity:
            return cls._atneginf()

        if z.is_zero:
            return cls._atzero

        nz = z.extract_multiplicatively(polar_lift(I))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(I)
        if nz is not None:
            return cls._Ifactor(nz, 1)
        nz = z.extract_multiplicatively(polar_lift(-I))
        if nz is not None:
            return cls._Ifactor(nz, -1)

        nz = z.extract_multiplicatively(polar_lift(-1))
        if nz is None and cls._trigfunc(0) == 0:
            nz = z.extract_multiplicatively(-1)
        if nz is not None:
            return cls._minusfactor(nz)

        nz, n = z.extract_branch_factor()
        if n == 0 and nz == z:
            return
        return 2*pi*I*n*cls._trigfunc(0) + cls(nz)

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return self._trigfunc(arg)/arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return self._eval_rewrite_as_expint(z).rewrite(Ei)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return self._eval_rewrite_as_expint(z).rewrite(uppergamma)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE this is fairly inefficient
        if self.args[0].subs(x, 0) != 0:
            return super()._eval_nseries(x, n, logx)
        baseseries = self._trigfunc(x)._eval_nseries(x, n, logx)
        if self._trigfunc(0) != 0:
            baseseries -= 1
        baseseries = baseseries.replace(Pow, lambda t, n: t**n/n, simultaneous=False)
        if self._trigfunc(0) != 0:
            baseseries += EulerGamma + log(x)
        return baseseries.subs(x, self.args[0])._eval_nseries(x, n, logx)


class Si(TrigonometricIntegral):
    r"""
    Sine integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Si}(z) = \int_0^z \frac{\sin{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Si
    >>> from sympy.abc import z

    The sine integral is an antiderivative of $sin(z)/z$:

    >>> Si(z).diff(z)
    sin(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Si(z*exp_polar(2*I*pi))
    Si(z)

    Sine integral behaves much like ordinary sine under multiplication by ``I``:

    >>> Si(I*z)
    I*Shi(z)
    >>> Si(-z)
    -Si(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Si(z).rewrite(expint)
    -I*(-expint(1, z*exp_polar(-I*pi/2))/2 +
         expint(1, z*exp_polar(I*pi/2))/2) + pi/2

    It can be rewritten in the form of sinc function (by definition):

    >>> from sympy import sinc
    >>> Si(z).rewrite(sinc)
    Integral(sinc(_t), (_t, 0, z))

    See Also
    ========

    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    sinc: unnormalized sinc function
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sin
    _atzero = S.Zero

    @classmethod
    def _atinf(cls):
        return pi*S.Half

    @classmethod
    def _atneginf(cls):
        return -pi*S.Half

    @classmethod
    def _minusfactor(cls, z):
        return -Si(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        return I*Shi(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # XXX should we polarify z?
        return pi/2 + (E1(polar_lift(I)*z) - E1(polar_lift(-I)*z))/2/I

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sinc(t), (t, 0, z))

    _eval_rewrite_as_sinc =  _eval_rewrite_as_Integral

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif not arg0.is_infinite:
            return self.func(arg0)
        else:
            return self

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point is S.Infinity:
            z = self.args[0]
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            return pi/2 - cos(z)*Add(*p) - sin(z)*Add(*q)

        # All other points are not handled
        return super(Si, self)._eval_aseries(n, args0, x, logx)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True


class Ci(TrigonometricIntegral):
    r"""
    Cosine integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Ci}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cos{t} - 1}{t} \mathrm{d}t
           = -\int_x^\infty \frac{\cos{t}}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Ci}(z) =
        -\frac{\operatorname{E}_1\left(e^{i\pi/2} z\right)
               + \operatorname{E}_1\left(e^{-i \pi/2} z\right)}{2}

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.

    The formula also holds as stated
    for $z \in \mathbb{C}$ with $\Re(z) > 0$.
    By lifting to the principal branch, we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Ci
    >>> from sympy.abc import z

    The cosine integral is a primitive of $\cos(z)/z$:

    >>> Ci(z).diff(z)
    cos(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Ci(z*exp_polar(2*I*pi))
    Ci(z) + 2*I*pi

    The cosine integral behaves somewhat like ordinary $\cos$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Ci(polar_lift(I)*z)
    Chi(z) + I*pi/2
    >>> Ci(polar_lift(-1)*z)
    Ci(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Ci(z).rewrite(expint)
    -expint(1, z*exp_polar(-I*pi/2))/2 - expint(1, z*exp_polar(I*pi/2))/2

    See Also
    ========

    Si: Sine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cos
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Zero

    @classmethod
    def _atneginf(cls):
        return I*pi

    @classmethod
    def _minusfactor(cls, z):
        return Ci(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Chi(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -(E1(polar_lift(I)*z) + E1(polar_lift(-I)*z))/2

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return S.EulerGamma + log(z) - Integral((1-cos(t))/t, (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        if point in (S.Infinity, S.NegativeInfinity):
            z = self.args[0]
            p = [S.NegativeOne**k * factorial(2*k) / z**(2*k + 1)
                    for k in range(n//2 + 1)] + [Order(1/z**n, x)]
            q = [S.NegativeOne**k * factorial(2*k + 1) / z**(2*(k + 1))
                    for k in range(n//2)] + [Order(1/z**n, x)]
            result = sin(z)*(Add(*p)) - cos(z)*(Add(*q))

            if point is S.NegativeInfinity:
                result += I*pi
            return result

        return super(Ci, self)._eval_aseries(n, args0, x, logx)

class Shi(TrigonometricIntegral):
    r"""
    Sinh integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{Shi}(z) = \int_0^z \frac{\sinh{t}}{t} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Shi
    >>> from sympy.abc import z

    The Sinh integral is a primitive of $\sinh(z)/z$:

    >>> Shi(z).diff(z)
    sinh(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Shi(z*exp_polar(2*I*pi))
    Shi(z)

    The $\sinh$ integral behaves much like ordinary $\sinh$ under
    multiplication by $i$:

    >>> Shi(I*z)
    I*Si(z)
    >>> Shi(-z)
    -Shi(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Shi(z).rewrite(expint)
    expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = sinh
    _atzero = S.Zero

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.NegativeInfinity

    @classmethod
    def _minusfactor(cls, z):
        return -Shi(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        return I*Si(z)*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        # XXX should we polarify z?
        return (E1(z) - E1(exp_polar(I*pi)*z))/2 - I*pi/2

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif not arg0.is_infinite:
            return self.func(arg0)
        else:
            return self


class Chi(TrigonometricIntegral):
    r"""
    Cosh integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \operatorname{Chi}(x) = \gamma + \log{x}
                         + \int_0^x \frac{\cosh{t} - 1}{t} \mathrm{d}t,

    where $\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \operatorname{Chi}(z) = \operatorname{Ci}\left(e^{i \pi/2}z\right)
                         - i\frac{\pi}{2},

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.
    By lifting to the principal branch we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Chi
    >>> from sympy.abc import z

    The $\cosh$ integral is a primitive of $\cosh(z)/z$:

    >>> Chi(z).diff(z)
    cosh(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Chi(z*exp_polar(2*I*pi))
    Chi(z) + 2*I*pi

    The $\cosh$ integral behaves somewhat like ordinary $\cosh$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Chi(polar_lift(I)*z)
    Ci(z) + I*pi/2
    >>> Chi(polar_lift(-1)*z)
    Chi(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Chi(z).rewrite(expint)
    -expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """

    _trigfunc = cosh
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.Infinity

    @classmethod
    def _minusfactor(cls, z):
        return Chi(z) + I*pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Ci(z) + I*pi/2*sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -I*pi/2 - (E1(z) + E1(exp_polar(I*pi)*z))/2

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e*logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self


###############################################################################
#################### FRESNEL INTEGRALS ########################################
###############################################################################

class FresnelIntegral(DefinedFunction):
    """ Base class for the Fresnel integrals."""

    unbranched = True

    @classmethod
    def eval(cls, z):
        # Values at positive infinities signs
        # if any were extracted automatically
        if z is S.Infinity:
            return S.Half

        # Value at zero
        if z.is_zero:
            return S.Zero

        # Try to pull out factors of -1 and I
        prefact = S.One
        newarg = z
        changed = False

        nz = newarg.extract_multiplicatively(-1)
        if nz is not None:
            prefact = -prefact
            newarg = nz
            changed = True

        nz = newarg.extract_multiplicatively(I)
        if nz is not None:
            prefact = cls._sign*I*prefact
            newarg = nz
            changed = True

        if changed:
            return prefact*cls(newarg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self._trigfunc(S.Half*pi*self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    _eval_is_finite = _eval_is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    as_real_imag = real_to_real_as_real_imag


class fresnels(FresnelIntegral):
    r"""
    Fresnel integral S.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{S}(z) = \int_0^z \sin{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnels
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnels(0)
    0
    >>> fresnels(oo)
    1/2
    >>> fresnels(-oo)
    -1/2
    >>> fresnels(I*oo)
    -I/2
    >>> fresnels(-I*oo)
    I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnels(-z)
    -fresnels(z)
    >>> fresnels(I*z)
    -I*fresnels(z)

    The Fresnel S integral obeys the mirror symmetry
    $\overline{S(z)} = S(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnels(z))
    fresnels(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnels(z), z)
    sin(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, sin, expand_func
    >>> integrate(sin(pi*z**2/2), z)
    3*fresnels(z)*gamma(3/4)/(4*gamma(7/4))
    >>> expand_func(integrate(sin(pi*z**2/2), z))
    fresnels(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnels(2).evalf(30)
    0.343415678363698242195300815958

    >>> fresnels(-2*I).evalf(30)
    0.343415678363698242195300815958*I

    See Also
    ========

    fresnelc: Fresnel cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelS
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = sin
    _sign = -S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (-pi**2*x**4*(4*n - 1)/(8*n*(2*n + 1)*(4*n + 3))) * p
            else:
                return x**3 * (-x**4)**n * (S(2)**(-2*n - 1)*pi**(2*n + 1)) / ((4*n + 3)*factorial(2*n + 1))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One + I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return pi*z**3/6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**Rational(9, 4) / (sqrt(2)*(z**2)**Rational(3, 4)*(-z)**Rational(3, 4))
                * meijerg([], [1], [Rational(3, 4)], [Rational(1, 4), 0], -pi**2*z**4/16))

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(sin(pi*t**2/2), (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return pi*arg**3/6
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo and -oo
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # expansion of S(x) = S1(x*sqrt(pi/2)), see reference[5] page 1-8
            # as only real infinities are dealt with, sin and cos are O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(0, n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            p = [-sqrt(2/pi)*t for t in p]
            q = [-sqrt(2/pi)*t for t in q]
            s = 1 if point is S.Infinity else -1
            # The expansion at oo is 1/2 + some odd powers of z
            # To get the expansion at -oo, replace z by -z and flip the sign
            # The result -1/2 + the same odd powers of z as before.
            return s*S.Half + (sin(z**2)*Add(*p) + cos(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)


class fresnelc(FresnelIntegral):
    r"""
    Fresnel integral C.

    Explanation
    ===========

    This function is defined by

    .. math:: \operatorname{C}(z) = \int_0^z \cos{\frac{\pi}{2} t^2} \mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnelc
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnelc(0)
    0
    >>> fresnelc(oo)
    1/2
    >>> fresnelc(-oo)
    -1/2
    >>> fresnelc(I*oo)
    I/2
    >>> fresnelc(-I*oo)
    -I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnelc(-z)
    -fresnelc(z)
    >>> fresnelc(I*z)
    I*fresnelc(z)

    The Fresnel C integral obeys the mirror symmetry
    $\overline{C(z)} = C(\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnelc(z))
    fresnelc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnelc(z), z)
    cos(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, cos, expand_func
    >>> integrate(cos(pi*z**2/2), z)
    fresnelc(z)*gamma(1/4)/(4*gamma(5/4))
    >>> expand_func(integrate(cos(pi*z**2/2), z))
    fresnelc(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnelc(2).evalf(30)
    0.488253406075340754500223503357

    >>> fresnelc(-2*I).evalf(30)
    -0.488253406075340754500223503357*I

    See Also
    ========

    fresnels: Fresnel sine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelC
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = cos
    _sign = S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (-pi**2*x**4*(4*n - 3)/(8*n*(2*n - 1)*(4*n + 1))) * p
            else:
                return x * (-x**4)**n * (S(2)**(-2*n)*pi**(2*n)) / ((4*n + 1)*factorial(2*n))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One - I)/4 * (erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi**2*z**4/16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return (pi*z**Rational(3, 4) / (sqrt(2)*root(z**2, 4)*root(-z, 4))
                * meijerg([], [1], [Rational(1, 4)], [Rational(3, 4), 0], -pi**2*z**4/16))

    def _eval_rewrite_as_Integral(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [z]).name)
        return Integral(cos(pi*t**2/2), (t, 0, z))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s*S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]

            # expansion of C(x) = C1(x*sqrt(pi/2)), see reference[5] page 1-8
            # as only real infinities are dealt with, sin and cos are O(1)
            p = [S.NegativeOne**k * factorial(4*k + 1) /
                 (2**(2*k + 2) * z**(4*k + 3) * 2**(2*k)*factorial(2*k))
                 for k in range(n) if 4*k + 3 < n]
            q = [1/(2*z)] + [S.NegativeOne**k * factorial(4*k - 1) /
                 (2**(2*k + 1) * z**(4*k + 1) * 2**(2*k - 1)*factorial(2*k - 1))
                 for k in range(1, n) if 4*k + 1 < n]

            p = [-sqrt(2/pi)*t for t in p]
            q = [ sqrt(2/pi)*t for t in q]
            s = 1 if point is S.Infinity else -1
            # The expansion at oo is 1/2 + some odd powers of z
            # To get the expansion at -oo, replace z by -z and flip the sign
            # The result -1/2 + the same odd powers of z as before.
            return s*S.Half + (cos(z**2)*Add(*p) + sin(z**2)*Add(*q)
                ).subs(x, sqrt(2/pi)*x) + Order(1/z**n, x)

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)


###############################################################################
#################### HELPER FUNCTIONS #########################################
###############################################################################


class _erfs(DefinedFunction):
    """
    Helper function to make the $\\mathrm{erf}(z)$ function
    tractable for the Gruntz algorithm.

    """
    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]

        # Expansion at oo
        if point is S.Infinity:
            z = self.args[0]
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # It is very inefficient to first add the order and then do the nseries
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # Expansion at I*oo
        t = point.extract_multiplicatively(I)
        if t is S.Infinity:
            z = self.args[0]
            # TODO: is the series really correct?
            l = [1/sqrt(pi) * factorial(2*k)*(-S(
                 4))**(-k)/factorial(k) * (1/z)**(2*k + 1) for k in range(n)]
            o = Order(1/z**(2*n + 1), x)
            # It is very inefficient to first add the order and then do the nseries
            return (Add(*l))._eval_nseries(x, n, logx) + o

        # All other points are not handled
        return super()._eval_aseries(n, args0, x, logx)

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -2/sqrt(pi) + 2*z*_erfs(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return (S.One - erf(z))*exp(z**2)


class _eis(DefinedFunction):
    """
    Helper function to make the $\\mathrm{Ei}(z)$ and $\\mathrm{li}(z)$
    functions tractable for the Gruntz algorithm.

    """


    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[0] not in (S.Infinity, S.NegativeInfinity):
            return super()._eval_aseries(n, args0, x, logx)

        z = self.args[0]
        l = [factorial(k) * (1/z)**(k + 1) for k in range(n)]
        o = Order(1/z**(n + 1), x)
        # It is very inefficient to first add the order and then do the nseries
        return (Add(*l))._eval_nseries(x, n, logx) + o


    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return S.One / z - _eis(z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return exp(-z)*Ei(z)

    def _eval_as_leading_term(self, x, logx, cdir):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexing.py ===
from __future__ import annotations

from contextlib import suppress
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
    AbstractMethodError,
    ChainedAssignmentError,
    IndexingError,
    InvalidIndexError,
    LossySetitemError,
    _chained_assignment_msg,
    _chained_assignment_warning_msg,
    _check_cacher,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    can_hold_element,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_sequence,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    construct_1d_array_from_inferred_fill_value,
    infer_fill_value,
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    is_list_like_indexer,
    is_scalar_indexer,
    length_of_indexer,
)
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        Axis,
        AxisInt,
        Self,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )

T = TypeVar("T")
# "null slice"
_NS = slice(None, None)
_one_ellipsis_message = "indexer may only contain one '...' entry"


# the public IndexSlicerMaker
class _IndexSlice:
    """
    Create an object to more easily perform multi-index slicing.

    See Also
    --------
    MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.

    Notes
    -----
    See :ref:`Defined Levels <advanced.shown_levels>`
    for further info on slicing a MultiIndex.

    Examples
    --------
    >>> midx = pd.MultiIndex.from_product([['A0','A1'], ['B0','B1','B2','B3']])
    >>> columns = ['foo', 'bar']
    >>> dfmi = pd.DataFrame(np.arange(16).reshape((len(midx), len(columns))),
    ...                     index=midx, columns=columns)

    Using the default slice command:

    >>> dfmi.loc[(slice(None), slice('B0', 'B1')), :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11

    Using the IndexSlice class for a more intuitive command:

    >>> idx = pd.IndexSlice
    >>> dfmi.loc[idx[:, 'B0':'B1'], :]
               foo  bar
        A0 B0    0    1
           B1    2    3
        A1 B0    8    9
           B1   10   11
    """

    def __getitem__(self, arg):
        return arg


IndexSlice = _IndexSlice()


class IndexingMixin:
    """
    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.
    """

    @property
    def iloc(self) -> _iLocIndexer:
        """
        Purely integer-location based indexing for selection by position.

        .. deprecated:: 2.2.0

           Returning a tuple from a callable is deprecated.

        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean
        array.

        Allowed inputs are:

        - An integer, e.g. ``5``.
        - A list or array of integers, e.g. ``[4, 3, 0]``.
        - A slice object with ints, e.g. ``1:7``.
        - A boolean array.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above).
          This is useful in method chains, when you don't have a reference to the
          calling object, but would like to base your selection on
          some value.
        - A tuple of row and column indexes. The tuple elements consist of one of the
          above inputs, e.g. ``(0, 1)``.

        ``.iloc`` will raise ``IndexError`` if a requested indexer is
        out-of-bounds, except *slice* indexers which allow out-of-bounds
        indexing (this conforms with python/numpy *slice* semantics).

        See more at :ref:`Selection by Position <indexing.integer>`.

        See Also
        --------
        DataFrame.iat : Fast integer location scalar accessor.
        DataFrame.loc : Purely label-location based indexer for selection by label.
        Series.iloc : Purely integer-location based indexing for
                       selection by position.

        Examples
        --------
        >>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
        ...           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
        ...           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]
        >>> df = pd.DataFrame(mydict)
        >>> df
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        **Indexing just the rows**

        With a scalar integer.

        >>> type(df.iloc[0])
        <class 'pandas.core.series.Series'>
        >>> df.iloc[0]
        a    1
        b    2
        c    3
        d    4
        Name: 0, dtype: int64

        With a list of integers.

        >>> df.iloc[[0]]
           a  b  c  d
        0  1  2  3  4
        >>> type(df.iloc[[0]])
        <class 'pandas.core.frame.DataFrame'>

        >>> df.iloc[[0, 1]]
             a    b    c    d
        0    1    2    3    4
        1  100  200  300  400

        With a `slice` object.

        >>> df.iloc[:3]
              a     b     c     d
        0     1     2     3     4
        1   100   200   300   400
        2  1000  2000  3000  4000

        With a boolean mask the same length as the index.

        >>> df.iloc[[True, False, True]]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        With a callable, useful in method chains. The `x` passed
        to the ``lambda`` is the DataFrame being sliced. This selects
        the rows whose index label even.

        >>> df.iloc[lambda x: x.index % 2 == 0]
              a     b     c     d
        0     1     2     3     4
        2  1000  2000  3000  4000

        **Indexing both axes**

        You can mix the indexer types for the index and columns. Use ``:`` to
        select the entire axis.

        With scalar integers.

        >>> df.iloc[0, 1]
        2

        With lists of integers.

        >>> df.iloc[[0, 2], [1, 3]]
              b     d
        0     2     4
        2  2000  4000

        With `slice` objects.

        >>> df.iloc[1:3, 0:3]
              a     b     c
        1   100   200   300
        2  1000  2000  3000

        With a boolean array whose length matches the columns.

        >>> df.iloc[:, [True, False, True, False]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000

        With a callable function that expects the Series or DataFrame.

        >>> df.iloc[:, lambda df: [0, 2]]
              a     c
        0     1     3
        1   100   300
        2  1000  3000
        """
        return _iLocIndexer("iloc", self)

    @property
    def loc(self) -> _LocIndexer:
        """
        Access a group of rows and columns by label(s) or a boolean array.

        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.

        Allowed inputs are:

        - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
          interpreted as a *label* of the index, and **never** as an
          integer position along the index).
        - A list or array of labels, e.g. ``['a', 'b', 'c']``.
        - A slice object with labels, e.g. ``'a':'f'``.

          .. warning:: Note that contrary to usual python slices, **both** the
              start and the stop are included

        - A boolean array of the same length as the axis being sliced,
          e.g. ``[True, False, True]``.
        - An alignable boolean Series. The index of the key will be aligned before
          masking.
        - An alignable Index. The Index of the returned selection will be the input.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above)

        See more at :ref:`Selection by Label <indexing.label>`.

        Raises
        ------
        KeyError
            If any items are not found.
        IndexingError
            If an indexed key is passed and its index is unalignable to the frame index.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.iloc : Access group of rows and columns by integer position(s).
        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
                       Series/DataFrame.
        Series.loc : Access group of values using labels.

        Examples
        --------
        **Getting values**

        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', 'sidewinder'],
        ...                   columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8

        Single label. Note this returns the row as a Series.

        >>> df.loc['viper']
        max_speed    4
        shield       5
        Name: viper, dtype: int64

        List of labels. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[['viper', 'sidewinder']]
                    max_speed  shield
        viper               4       5
        sidewinder          7       8

        Single label for row and column

        >>> df.loc['cobra', 'shield']
        2

        Slice with labels for row and single label for column. As mentioned
        above, note that both the start and stop of the slice are included.

        >>> df.loc['cobra':'viper', 'max_speed']
        cobra    1
        viper    4
        Name: max_speed, dtype: int64

        Boolean list with the same length as the row axis

        >>> df.loc[[False, False, True]]
                    max_speed  shield
        sidewinder          7       8

        Alignable boolean Series:

        >>> df.loc[pd.Series([False, True, False],
        ...                  index=['viper', 'sidewinder', 'cobra'])]
                             max_speed  shield
        sidewinder          7       8

        Index (same behavior as ``df.reindex``)

        >>> df.loc[pd.Index(["cobra", "viper"], name="foo")]
               max_speed  shield
        foo
        cobra          1       2
        viper          4       5

        Conditional that returns a boolean Series

        >>> df.loc[df['shield'] > 6]
                    max_speed  shield
        sidewinder          7       8

        Conditional that returns a boolean Series with column labels specified

        >>> df.loc[df['shield'] > 6, ['max_speed']]
                    max_speed
        sidewinder          7

        Multiple conditional using ``&`` that returns a boolean Series

        >>> df.loc[(df['max_speed'] > 1) & (df['shield'] < 8)]
                    max_speed  shield
        viper          4       5

        Multiple conditional using ``|`` that returns a boolean Series

        >>> df.loc[(df['max_speed'] > 4) | (df['shield'] < 5)]
                    max_speed  shield
        cobra               1       2
        sidewinder          7       8

        Please ensure that each condition is wrapped in parentheses ``()``.
        See the :ref:`user guide<indexing.boolean>`
        for more details and explanations of Boolean indexing.

        .. note::
            If you find yourself using 3 or more conditionals in ``.loc[]``,
            consider using :ref:`advanced indexing<advanced.advanced_hierarchical>`.

            See below for using ``.loc[]`` on MultiIndex DataFrames.

        Callable that returns a boolean Series

        >>> df.loc[lambda df: df['shield'] == 8]
                    max_speed  shield
        sidewinder          7       8

        **Setting values**

        Set value for all items matching the list of labels

        >>> df.loc[['viper', 'sidewinder'], ['shield']] = 50
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4      50
        sidewinder          7      50

        Set value for an entire row

        >>> df.loc['cobra'] = 10
        >>> df
                    max_speed  shield
        cobra              10      10
        viper               4      50
        sidewinder          7      50

        Set value for an entire column

        >>> df.loc[:, 'max_speed'] = 30
        >>> df
                    max_speed  shield
        cobra              30      10
        viper              30      50
        sidewinder         30      50

        Set value for rows matching callable condition

        >>> df.loc[df['shield'] > 35] = 0
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       0
        sidewinder          0       0

        Add value matching location

        >>> df.loc["viper", "shield"] += 5
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       5
        sidewinder          0       0

        Setting using a ``Series`` or a ``DataFrame`` sets the values matching the
        index labels, not the index positions.

        >>> shuffled_df = df.loc[["viper", "cobra", "sidewinder"]]
        >>> df.loc[:] += shuffled_df
        >>> df
                    max_speed  shield
        cobra              60      20
        viper               0      10
        sidewinder          0       0

        **Getting values on a DataFrame with an index that has integer labels**

        Another example using integers for the index

        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=[7, 8, 9], columns=['max_speed', 'shield'])
        >>> df
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        Slice with integer labels for rows. As mentioned above, note that both
        the start and stop of the slice are included.

        >>> df.loc[7:9]
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8

        **Getting values with a MultiIndex**

        A number of examples using a DataFrame with a MultiIndex

        >>> tuples = [
        ...     ('cobra', 'mark i'), ('cobra', 'mark ii'),
        ...     ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
        ...     ('viper', 'mark ii'), ('viper', 'mark iii')
        ... ]
        >>> index = pd.MultiIndex.from_tuples(tuples)
        >>> values = [[12, 2], [0, 4], [10, 20],
        ...           [1, 4], [7, 1], [16, 36]]
        >>> df = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)
        >>> df
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Single label. Note this returns a DataFrame with a single index.

        >>> df.loc['cobra']
                 max_speed  shield
        mark i          12       2
        mark ii          0       4

        Single index tuple. Note this returns a Series.

        >>> df.loc[('cobra', 'mark ii')]
        max_speed    0
        shield       4
        Name: (cobra, mark ii), dtype: int64

        Single label for row and column. Similar to passing in a tuple, this
        returns a Series.

        >>> df.loc['cobra', 'mark i']
        max_speed    12
        shield        2
        Name: (cobra, mark i), dtype: int64

        Single tuple. Note using ``[[]]`` returns a DataFrame.

        >>> df.loc[[('cobra', 'mark ii')]]
                       max_speed  shield
        cobra mark ii          0       4

        Single tuple for the index with a single label for the column

        >>> df.loc[('cobra', 'mark i'), 'shield']
        2

        Slice from index tuple to single label

        >>> df.loc[('cobra', 'mark i'):'viper']
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36

        Slice from index tuple to index tuple

        >>> df.loc[('cobra', 'mark i'):('viper', 'mark ii')]
                            max_speed  shield
        cobra      mark i          12       2
                   mark ii          0       4
        sidewinder mark i          10      20
                   mark ii          1       4
        viper      mark ii          7       1

        Please see the :ref:`user guide<advanced.advanced_hierarchical>`
        for more details and explanations of advanced indexing.
        """
        return _LocIndexer("loc", self)

    @property
    def at(self) -> _AtIndexer:
        """
        Access a single value for a row/column label pair.

        Similar to ``loc``, in that both provide label-based lookups. Use
        ``at`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        KeyError
            If getting a value and 'label' does not exist in a DataFrame or Series.

        ValueError
            If row/column label pair is not a tuple or if any label
            from the pair is not a scalar for DataFrame.
            If label is list-like (*excluding* NamedTuple) for Series.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column pair by label.
        DataFrame.iat : Access a single value for a row/column pair by integer
            position.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer
            position(s).
        Series.at : Access a single value by label.
        Series.iat : Access a single value by integer position.
        Series.loc : Access a group of rows by label(s).
        Series.iloc : Access a group of rows by integer position(s).

        Notes
        -----
        See :ref:`Fast scalar value getting and setting <indexing.basics.get_value>`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
        ...                   index=[4, 5, 6], columns=['A', 'B', 'C'])
        >>> df
            A   B   C
        4   0   2   3
        5   0   4   1
        6  10  20  30

        Get value at specified row/column pair

        >>> df.at[4, 'B']
        2

        Set value at specified row/column pair

        >>> df.at[4, 'B'] = 10
        >>> df.at[4, 'B']
        10

        Get value within a Series

        >>> df.loc[5].at['B']
        4
        """
        return _AtIndexer("at", self)

    @property
    def iat(self) -> _iAtIndexer:
        """
        Access a single value for a row/column pair by integer position.

        Similar to ``iloc``, in that both provide integer-based lookups. Use
        ``iat`` if you only need to get or set a single value in a DataFrame
        or Series.

        Raises
        ------
        IndexError
            When integer position is out of bounds.

        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.loc : Access a group of rows and columns by label(s).
        DataFrame.iloc : Access a group of rows and columns by integer position(s).

        Examples
        --------
        >>> df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
        ...                   columns=['A', 'B', 'C'])
        >>> df
            A   B   C
        0   0   2   3
        1   0   4   1
        2  10  20  30

        Get value at specified row/column pair

        >>> df.iat[1, 2]
        1

        Set value at specified row/column pair

        >>> df.iat[1, 2] = 10
        >>> df.iat[1, 2]
        10

        Get value within a series

        >>> df.loc[0].iat[1]
        2
        """
        return _iAtIndexer("iat", self)


class _LocationIndexer(NDFrameIndexerBase):
    _valid_types: str
    axis: AxisInt | None = None

    # sub-classes need to set _takeable
    _takeable: bool

    @final
    def __call__(self, axis: Axis | None = None) -> Self:
        # we need to return a copy of ourselves
        new_self = type(self)(self.name, self.obj)

        if axis is not None:
            axis_int_none = self.obj._get_axis_number(axis)
        else:
            axis_int_none = axis
        new_self.axis = axis_int_none
        return new_self

    def _get_setitem_indexer(self, key):
        """
        Convert a potentially-label-based key into a positional indexer.
        """
        if self.name == "loc":
            # always holds here bc iloc overrides _get_setitem_indexer
            self._ensure_listlike_indexer(key)

        if isinstance(key, tuple):
            for x in key:
                check_dict_or_set_indexers(x)

        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)

        ax = self.obj._get_axis(0)

        if (
            isinstance(ax, MultiIndex)
            and self.name != "iloc"
            and is_hashable(key)
            and not isinstance(key, slice)
        ):
            with suppress(KeyError, InvalidIndexError):
                # TypeError e.g. passed a bool
                return ax.get_loc(key)

        if isinstance(key, tuple):
            with suppress(IndexingError):
                # suppress "Too many indexers"
                return self._convert_tuple(key)

        if isinstance(key, range):
            # GH#45479 test_loc_setitem_range_key
            key = list(key)

        return self._convert_to_indexer(key, axis=0)

    @final
    def _maybe_mask_setitem_value(self, indexer, value):
        """
        If we have obj.iloc[mask] = series_or_frame and series_or_frame has the
        same length as obj, we treat this as obj.iloc[mask] = series_or_frame[mask],
        similar to Series.__setitem__.

        Note this is only for loc, not iloc.
        """

        if (
            isinstance(indexer, tuple)
            and len(indexer) == 2
            and isinstance(value, (ABCSeries, ABCDataFrame))
        ):
            pi, icols = indexer
            ndim = value.ndim
            if com.is_bool_indexer(pi) and len(value) == len(pi):
                newkey = pi.nonzero()[0]

                if is_scalar_indexer(icols, self.ndim - 1) and ndim == 1:
                    # e.g. test_loc_setitem_boolean_mask_allfalse
                    # test_loc_setitem_ndframe_values_alignment
                    value = self.obj.iloc._align_series(indexer, value)
                    indexer = (newkey, icols)

                elif (
                    isinstance(icols, np.ndarray)
                    and icols.dtype.kind == "i"
                    and len(icols) == 1
                ):
                    if ndim == 1:
                        # We implicitly broadcast, though numpy does not, see
                        # github.com/pandas-dev/pandas/pull/45501#discussion_r789071825
                        # test_loc_setitem_ndframe_values_alignment
                        value = self.obj.iloc._align_series(indexer, value)
                        indexer = (newkey, icols)

                    elif ndim == 2 and value.shape[1] == 1:
                        # test_loc_setitem_ndframe_values_alignment
                        value = self.obj.iloc._align_frame(indexer, value)
                        indexer = (newkey, icols)
        elif com.is_bool_indexer(indexer):
            indexer = indexer.nonzero()[0]

        return indexer, value

    @final
    def _ensure_listlike_indexer(self, key, axis=None, value=None) -> None:
        """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.

        Parameters
        ----------
        key : list-like of column labels
            Target labels.
        axis : key axis if known
        """
        column_axis = 1

        # column only exists in 2-dimensional DataFrame
        if self.ndim != 2:
            return

        if isinstance(key, tuple) and len(key) > 1:
            # key may be a tuple if we are .loc
            # if length of key is > 1 set key to column part
            key = key[column_axis]
            axis = column_axis

        if (
            axis == column_axis
            and not isinstance(self.obj.columns, MultiIndex)
            and is_list_like_indexer(key)
            and not com.is_bool_indexer(key)
            and all(is_hashable(k) for k in key)
        ):
            # GH#38148
            keys = self.obj.columns.union(key, sort=False)
            diff = Index(key).difference(self.obj.columns, sort=False)

            if len(diff):
                # e.g. if we are doing df.loc[:, ["A", "B"]] = 7 and "B"
                #  is a new column, add the new columns with dtype=np.void
                #  so that later when we go through setitem_single_column
                #  we will use isetitem. Without this, the reindex_axis
                #  below would create float64 columns in this example, which
                #  would successfully hold 7, so we would end up with the wrong
                #  dtype.
                indexer = np.arange(len(keys), dtype=np.intp)
                indexer[len(self.obj.columns) :] = -1
                new_mgr = self.obj._mgr.reindex_indexer(
                    keys, indexer=indexer, axis=0, only_slice=True, use_na_proxy=True
                )
                self.obj._mgr = new_mgr
                return

            self.obj._mgr = self.obj._mgr.reindex_axis(keys, axis=0, only_slice=True)

    @final
    def __setitem__(self, key, value) -> None:
        if not PYPY and using_copy_on_write():
            if sys.getrefcount(self.obj) <= 2:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )
        elif not PYPY and not using_copy_on_write():
            ctr = sys.getrefcount(self.obj)
            ref_count = 2
            if not warn_copy_on_write() and _check_cacher(self.obj):
                # see https://github.com/pandas-dev/pandas/pull/56060#discussion_r1399245221
                ref_count += 1
            if ctr <= ref_count:
                warnings.warn(
                    _chained_assignment_warning_msg, FutureWarning, stacklevel=2
                )

        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            key = tuple(list(x) if is_iterator(x) else x for x in key)
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            maybe_callable = com.apply_if_callable(key, self.obj)
            key = self._check_deprecated_callable_usage(key, maybe_callable)
        indexer = self._get_setitem_indexer(key)
        self._has_valid_setitem_indexer(key)

        iloc = self if self.name == "iloc" else self.obj.iloc
        iloc._setitem_with_indexer(indexer, value, self.name)

    def _validate_key(self, key, axis: AxisInt):
        """
        Ensure that key is valid for current indexer.

        Parameters
        ----------
        key : scalar, slice or list-like
            Key requested.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        TypeError
            If the key (or some element of it) has wrong type.
        IndexError
            If the key (or some element of it) is out of bounds.
        KeyError
            If the key was not found.
        """
        raise AbstractMethodError(self)

    @final
    def _expand_ellipsis(self, tup: tuple) -> tuple:
        """
        If a tuple key includes an Ellipsis, replace it with an appropriate
        number of null slices.
        """
        if any(x is Ellipsis for x in tup):
            if tup.count(Ellipsis) > 1:
                raise IndexingError(_one_ellipsis_message)

            if len(tup) == self.ndim:
                # It is unambiguous what axis this Ellipsis is indexing,
                #  treat as a single null slice.
                i = tup.index(Ellipsis)
                # FIXME: this assumes only one Ellipsis
                new_key = tup[:i] + (_NS,) + tup[i + 1 :]
                return new_key

            # TODO: other cases?  only one test gets here, and that is covered
            #  by _validate_key_length
        return tup

    @final
    def _validate_tuple_indexer(self, key: tuple) -> tuple:
        """
        Check the key for valid keys across my indexer.
        """
        key = self._validate_key_length(key)
        key = self._expand_ellipsis(key)
        for i, k in enumerate(key):
            try:
                self._validate_key(k, i)
            except ValueError as err:
                raise ValueError(
                    "Location based indexing can only have "
                    f"[{self._valid_types}] types"
                ) from err
        return key

    @final
    def _is_nested_tuple_indexer(self, tup: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
        if any(isinstance(ax, MultiIndex) for ax in self.obj.axes):
            return any(is_nested_tuple(tup, ax) for ax in self.obj.axes)
        return False

    @final
    def _convert_tuple(self, key: tuple) -> tuple:
        # Note: we assume _tupleize_axis_indexer has been called, if necessary.
        self._validate_key_length(key)
        keyidx = [self._convert_to_indexer(k, axis=i) for i, k in enumerate(key)]
        return tuple(keyidx)

    @final
    def _validate_key_length(self, key: tuple) -> tuple:
        if len(key) > self.ndim:
            if key[0] is Ellipsis:
                # e.g. Series.iloc[..., 3] reduces to just Series.iloc[3]
                key = key[1:]
                if Ellipsis in key:
                    raise IndexingError(_one_ellipsis_message)
                return self._validate_key_length(key)
            raise IndexingError("Too many indexers")
        return key

    @final
    def _getitem_tuple_same_dim(self, tup: tuple):
        """
        Index with indexers that should return an object of the same dimension
        as self.obj.

        This is only called after a failed call to _getitem_lowerdim.
        """
        retval = self.obj
        # Selecting columns before rows is significantly faster
        start_val = (self.ndim - len(tup)) + 1
        for i, key in enumerate(reversed(tup)):
            i = self.ndim - i - start_val
            if com.is_null_slice(key):
                continue

            retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
            # We should never have retval.ndim < self.ndim, as that should
            #  be handled by the _getitem_lowerdim call above.
            assert retval.ndim == self.ndim

        if retval is self.obj:
            # if all axes were a null slice (`df.loc[:, :]`), ensure we still
            # return a new object (https://github.com/pandas-dev/pandas/pull/49469)
            retval = retval.copy(deep=False)

        return retval

    @final
    def _getitem_lowerdim(self, tup: tuple):
        # we can directly get the axis result since the axis is specified
        if self.axis is not None:
            axis = self.obj._get_axis_number(self.axis)
            return self._getitem_axis(tup, axis=axis)

        # we may have a nested tuples indexer here
        if self._is_nested_tuple_indexer(tup):
            return self._getitem_nested_tuple(tup)

        # we maybe be using a tuple to represent multiple dimensions here
        ax0 = self.obj._get_axis(0)
        # ...but iloc should handle the tuple as simple integer-location
        # instead of checking it as multiindex representation (GH 13797)
        if (
            isinstance(ax0, MultiIndex)
            and self.name != "iloc"
            and not any(isinstance(x, slice) for x in tup)
        ):
            # Note: in all extant test cases, replacing the slice condition with
            #  `all(is_hashable(x) or com.is_null_slice(x) for x in tup)`
            #  is equivalent.
            #  (see the other place where we call _handle_lowerdim_multi_index_axis0)
            with suppress(IndexingError):
                return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)

        tup = self._validate_key_length(tup)

        for i, key in enumerate(tup):
            if is_label_like(key):
                # We don't need to check for tuples here because those are
                #  caught by the _is_nested_tuple_indexer check above.
                section = self._getitem_axis(key, axis=i)

                # We should never have a scalar section here, because
                #  _getitem_lowerdim is only called after a check for
                #  is_scalar_access, which that would be.
                if section.ndim == self.ndim:
                    # we're in the middle of slicing through a MultiIndex
                    # revise the key wrt to `section` by inserting an _NS
                    new_key = tup[:i] + (_NS,) + tup[i + 1 :]

                else:
                    # Note: the section.ndim == self.ndim check above
                    #  rules out having DataFrame here, so we dont need to worry
                    #  about transposing.
                    new_key = tup[:i] + tup[i + 1 :]

                    if len(new_key) == 1:
                        new_key = new_key[0]

                # Slices should return views, but calling iloc/loc with a null
                # slice returns a new object.
                if com.is_null_slice(new_key):
                    return section
                # This is an elided recursive call to iloc/loc
                return getattr(section, self.name)[new_key]

        raise IndexingError("not applicable")

    @final
    def _getitem_nested_tuple(self, tup: tuple):
        # we have a nested tuple so have at least 1 multi-index level
        # we should be able to match up the dimensionality here

        def _contains_slice(x: object) -> bool:
            # Check if object is a slice or a tuple containing a slice
            if isinstance(x, tuple):
                return any(isinstance(v, slice) for v in x)
            elif isinstance(x, slice):
                return True
            return False

        for key in tup:
            check_dict_or_set_indexers(key)

        # we have too many indexers for our dim, but have at least 1
        # multi-index dimension, try to see if we have something like
        # a tuple passed to a series with a multi-index
        if len(tup) > self.ndim:
            if self.name != "loc":
                # This should never be reached, but let's be explicit about it
                raise ValueError("Too many indices")  # pragma: no cover
            if all(
                (is_hashable(x) and not _contains_slice(x)) or com.is_null_slice(x)
                for x in tup
            ):
                # GH#10521 Series should reduce MultiIndex dimensions instead of
                #  DataFrame, IndexingError is not raised when slice(None,None,None)
                #  with one row.
                with suppress(IndexingError):
                    return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(
                        tup
                    )
            elif isinstance(self.obj, ABCSeries) and any(
                isinstance(k, tuple) for k in tup
            ):
                # GH#35349 Raise if tuple in tuple for series
                # Do this after the all-hashable-or-null-slice check so that
                #  we are only getting non-hashable tuples, in particular ones
                #  that themselves contain a slice entry
                # See test_loc_series_getitem_too_many_dimensions
                raise IndexingError("Too many indexers")

            # this is a series with a multi-index specified a tuple of
            # selectors
            axis = self.axis or 0
            return self._getitem_axis(tup, axis=axis)

        # handle the multi-axis by taking sections and reducing
        # this is iterative
        obj = self.obj
        # GH#41369 Loop in reverse order ensures indexing along columns before rows
        # which selects only necessary blocks which avoids dtype conversion if possible
        axis = len(tup) - 1
        for key in tup[::-1]:
            if com.is_null_slice(key):
                axis -= 1
                continue

            obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
            axis -= 1

            # if we have a scalar, we are done
            if is_scalar(obj) or not hasattr(obj, "ndim"):
                break

        return obj

    def _convert_to_indexer(self, key, axis: AxisInt):
        raise AbstractMethodError(self)

    def _check_deprecated_callable_usage(self, key: Any, maybe_callable: T) -> T:
        # GH53533
        if self.name == "iloc" and callable(key) and isinstance(maybe_callable, tuple):
            warnings.warn(
                "Returning a tuple from a callable with iloc "
                "is deprecated and will be removed in a future version",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return maybe_callable

    @final
    def __getitem__(self, key):
        check_dict_or_set_indexers(key)
        if type(key) is tuple:
            key = tuple(list(x) if is_iterator(x) else x for x in key)
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
            if self._is_scalar_access(key):
                return self.obj._get_value(*key, takeable=self._takeable)
            return self._getitem_tuple(key)
        else:
            # we by definition only have the 0th axis
            axis = self.axis or 0

            maybe_callable = com.apply_if_callable(key, self.obj)
            maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _is_scalar_access(self, key: tuple):
        raise NotImplementedError()

    def _getitem_tuple(self, tup: tuple):
        raise AbstractMethodError(self)

    def _getitem_axis(self, key, axis: AxisInt):
        raise NotImplementedError()

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        raise AbstractMethodError(self)

    @final
    def _getbool_axis(self, key, axis: AxisInt):
        # caller is responsible for ensuring non-None axis
        labels = self.obj._get_axis(axis)
        key = check_bool_indexer(labels, key)
        inds = key.nonzero()[0]
        return self.obj._take_with_is_copy(inds, axis=axis)


@doc(IndexingMixin.loc)
class _LocIndexer(_LocationIndexer):
    _takeable: bool = False
    _valid_types = (
        "labels (MUST BE IN THE INDEX), slices of labels (BOTH "
        "endpoints included! Can be slices of integers if the "
        "index is integers), listlike of labels, boolean"
    )

    # -------------------------------------------------------------------
    # Key Checks

    @doc(_LocationIndexer._validate_key)
    def _validate_key(self, key, axis: Axis):
        # valid for a collection of labels (we check their presence later)
        # slice of labels (where start-end in labels)
        # slice of integers (only if in the labels)
        # boolean not in slice and with boolean index
        ax = self.obj._get_axis(axis)
        if isinstance(key, bool) and not (
            is_bool_dtype(ax.dtype)
            or ax.dtype.name == "boolean"
            or isinstance(ax, MultiIndex)
            and is_bool_dtype(ax.get_level_values(0).dtype)
        ):
            raise KeyError(
                f"{key}: boolean label can not be used without a boolean index"
            )

        if isinstance(key, slice) and (
            isinstance(key.start, bool) or isinstance(key.stop, bool)
        ):
            raise TypeError(f"{key}: boolean values can not be used in a slice")

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        return True

    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
        # this is a shortcut accessor to both .loc and .iloc
        # that provide the equivalent access of .at and .iat
        # a) avoid getting things via sections and (to minimize dtype changes)
        # b) provide a performant path
        if len(key) != self.ndim:
            return False

        for i, k in enumerate(key):
            if not is_scalar(k):
                return False

            ax = self.obj.axes[i]
            if isinstance(ax, MultiIndex):
                return False

            if isinstance(k, str) and ax._supports_partial_string_indexing:
                # partial string indexing, df.loc['2000', 'A']
                # should not be considered scalar
                return False

            if not ax._index_as_unique:
                return False

        return True

    # -------------------------------------------------------------------
    # MultiIndex Handling

    def _multi_take_opportunity(self, tup: tuple) -> bool:
        """
        Check whether there is the possibility to use ``_multi_take``.

        Currently the limit is that all axes being indexed, must be indexed with
        list-likes.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        bool
            Whether the current indexing,
            can be passed through `_multi_take`.
        """
        if not all(is_list_like_indexer(x) for x in tup):
            return False

        # just too complicated
        return not any(com.is_bool_indexer(x) for x in tup)

    def _multi_take(self, tup: tuple):
        """
        Create the indexers for the passed tuple of keys, and
        executes the take operation. This allows the take operation to be
        executed all at once, rather than once for each dimension.
        Improving efficiency.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        values: same type as the object being indexed
        """
        # GH 836
        d = {
            axis: self._get_listlike_indexer(key, axis)
            for (key, axis) in zip(tup, self.obj._AXIS_ORDERS)
        }
        return self.obj._reindex_with_indexers(d, copy=True, allow_dups=True)

    # -------------------------------------------------------------------

    def _getitem_iterable(self, key, axis: AxisInt):
        """
        Index current object with an iterable collection of keys.

        Parameters
        ----------
        key : iterable
            Targeted labels.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If no key was found. Will change in the future to raise if not all
            keys were found.

        Returns
        -------
        scalar, DataFrame, or Series: indexed value(s).
        """
        # we assume that not com.is_bool_indexer(key), as that is
        #  handled before we get here.
        self._validate_key(key, axis)

        # A collection of keys
        keyarr, indexer = self._get_listlike_indexer(key, axis)
        return self.obj._reindex_with_indexers(
            {axis: [keyarr, indexer]}, copy=True, allow_dups=True
        )

    def _getitem_tuple(self, tup: tuple):
        with suppress(IndexingError):
            tup = self._expand_ellipsis(tup)
            return self._getitem_lowerdim(tup)

        # no multi-index, so validate all of the indexers
        tup = self._validate_tuple_indexer(tup)

        # ugly hack for GH #836
        if self._multi_take_opportunity(tup):
            return self._multi_take(tup)

        return self._getitem_tuple_same_dim(tup)

    def _get_label(self, label, axis: AxisInt):
        # GH#5567 this will fail if the label is not present in the axis.
        return self.obj.xs(label, axis=axis)

    def _handle_lowerdim_multi_index_axis0(self, tup: tuple):
        # we have an axis0 multi-index, handle or raise
        axis = self.axis or 0
        try:
            # fast path for series or for tup devoid of slices
            return self._get_label(tup, axis=axis)

        except KeyError as ek:
            # raise KeyError if number of indexers match
            # else IndexingError will be raised
            if self.ndim < len(tup) <= self.obj.index.nlevels:
                raise ek
            raise IndexingError("No label returned") from ek

    def _getitem_axis(self, key, axis: AxisInt):
        key = item_from_zerodim(key)
        if is_iterator(key):
            key = list(key)
        if key is Ellipsis:
            key = slice(None)

        labels = self.obj._get_axis(axis)

        if isinstance(key, tuple) and isinstance(labels, MultiIndex):
            key = tuple(key)

        if isinstance(key, slice):
            self._validate_key(key, axis)
            return self._get_slice_axis(key, axis=axis)
        elif com.is_bool_indexer(key):
            return self._getbool_axis(key, axis=axis)
        elif is_list_like_indexer(key):
            # an iterable multi-selection
            if not (isinstance(key, tuple) and isinstance(labels, MultiIndex)):
                if hasattr(key, "ndim") and key.ndim > 1:
                    raise ValueError("Cannot index with multidimensional key")

                return self._getitem_iterable(key, axis=axis)

            # nested tuple slicing
            if is_nested_tuple(key, labels):
                locs = labels.get_locs(key)
                indexer: list[slice | npt.NDArray[np.intp]] = [slice(None)] * self.ndim
                indexer[axis] = locs
                return self.obj.iloc[tuple(indexer)]

        # fall thru to straight lookup
        self._validate_key(key, axis)
        return self._get_label(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        """
        This is pretty simple as we just have to deal with labels.
        """
        # caller is responsible for ensuring non-None axis
        obj = self.obj
        if not need_slice(slice_obj):
            return obj.copy(deep=False)

        labels = obj._get_axis(axis)
        indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop, slice_obj.step)

        if isinstance(indexer, slice):
            return self.obj._slice(indexer, axis=axis)
        else:
            # DatetimeIndex overrides Index.slice_indexer and may
            #  return a DatetimeIndex instead of a slice object.
            return self.obj.take(indexer, axis=axis)

    def _convert_to_indexer(self, key, axis: AxisInt):
        """
        Convert indexing key into something we can use to do actual fancy
        indexing on a ndarray.

        Examples
        ix[:5] -> slice(0, 5)
        ix[[1,2,3]] -> [1,2,3]
        ix[['foo', 'bar', 'baz']] -> [i, j, k] (indices of foo, bar, baz)

        Going by Zen of Python?
        'In the face of ambiguity, refuse the temptation to guess.'
        raise AmbiguousIndexError with integer labels?
        - No, prefer label-based indexing
        """
        labels = self.obj._get_axis(axis)

        if isinstance(key, slice):
            return labels._convert_slice_indexer(key, kind="loc")

        if (
            isinstance(key, tuple)
            and not isinstance(labels, MultiIndex)
            and self.ndim < 2
            and len(key) > 1
        ):
            raise IndexingError("Too many indexers")

        # Slices are not valid keys passed in by the user,
        # even though they are hashable in Python 3.12
        contains_slice = False
        if isinstance(key, tuple):
            contains_slice = any(isinstance(v, slice) for v in key)

        if is_scalar(key) or (
            isinstance(labels, MultiIndex) and is_hashable(key) and not contains_slice
        ):
            # Otherwise get_loc will raise InvalidIndexError

            # if we are a label return me
            try:
                return labels.get_loc(key)
            except LookupError:
                if isinstance(key, tuple) and isinstance(labels, MultiIndex):
                    if len(key) == labels.nlevels:
                        return {"key": key}
                    raise
            except InvalidIndexError:
                # GH35015, using datetime as column indices raises exception
                if not isinstance(labels, MultiIndex):
                    raise
            except ValueError:
                if not is_integer(key):
                    raise
                return {"key": key}

        if is_nested_tuple(key, labels):
            if self.ndim == 1 and any(isinstance(k, tuple) for k in key):
                # GH#35349 Raise if tuple in tuple for series
                raise IndexingError("Too many indexers")
            return labels.get_locs(key)

        elif is_list_like_indexer(key):
            if is_iterator(key):
                key = list(key)

            if com.is_bool_indexer(key):
                key = check_bool_indexer(labels, key)
                return key
            else:
                return self._get_listlike_indexer(key, axis)[1]
        else:
            try:
                return labels.get_loc(key)
            except LookupError:
                # allow a not found key only if we are a setter
                if not is_list_like_indexer(key):
                    return {"key": key}
                raise

    def _get_listlike_indexer(self, key, axis: AxisInt):
        """
        Transform a list-like of keys into a new index and an indexer.

        Parameters
        ----------
        key : list-like
            Targeted labels.
        axis:  int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.

        Returns
        -------
        keyarr: Index
            New index (coinciding with 'key' if the axis is unique).
        values : array-like
            Indexer for the return object, -1 denotes keys not found.
        """
        ax = self.obj._get_axis(axis)
        axis_name = self.obj._get_axis_name(axis)

        keyarr, indexer = ax._get_indexer_strict(key, axis_name)

        return keyarr, indexer


@doc(IndexingMixin.iloc)
class _iLocIndexer(_LocationIndexer):
    _valid_types = (
        "integer, integer slice (START point is INCLUDED, END "
        "point is EXCLUDED), listlike of integers, boolean array"
    )
    _takeable = True

    # -------------------------------------------------------------------
    # Key Checks

    def _validate_key(self, key, axis: AxisInt):
        if com.is_bool_indexer(key):
            if hasattr(key, "index") and isinstance(key.index, Index):
                if key.index.inferred_type == "integer":
                    raise NotImplementedError(
                        "iLocation based boolean "
                        "indexing on an integer type "
                        "is not available"
                    )
                raise ValueError(
                    "iLocation based boolean indexing cannot use "
                    "an indexable as a mask"
                )
            return

        if isinstance(key, slice):
            return
        elif is_integer(key):
            self._validate_integer(key, axis)
        elif isinstance(key, tuple):
            # a tuple should already have been caught by this point
            # so don't treat a tuple as a valid indexer
            raise IndexingError("Too many indexers")
        elif is_list_like_indexer(key):
            if isinstance(key, ABCSeries):
                arr = key._values
            elif is_array_like(key):
                arr = key
            else:
                arr = np.array(key)
            len_axis = len(self.obj._get_axis(axis))

            # check that the key has a numeric dtype
            if not is_numeric_dtype(arr.dtype):
                raise IndexError(f".iloc requires numeric indexers, got {arr}")

            # check that the key does not exceed the maximum size of the index
            if len(arr) and (arr.max() >= len_axis or arr.min() < -len_axis):
                raise IndexError("positional indexers are out-of-bounds")
        else:
            raise ValueError(f"Can only index by location with a [{self._valid_types}]")

    def _has_valid_setitem_indexer(self, indexer) -> bool:
        """
        Validate that a positional indexer cannot enlarge its target
        will raise if needed, does not modify the indexer externally.

        Returns
        -------
        bool
        """
        if isinstance(indexer, dict):
            raise IndexError("iloc cannot enlarge its target object")

        if isinstance(indexer, ABCDataFrame):
            raise TypeError(
                "DataFrame indexer for .iloc is not supported. "
                "Consider using .loc with a DataFrame indexer for automatic alignment.",
            )

        if not isinstance(indexer, tuple):
            indexer = _tuplify(self.ndim, indexer)

        for ax, i in zip(self.obj.axes, indexer):
            if isinstance(i, slice):
                # should check the stop slice?
                pass
            elif is_list_like_indexer(i):
                # should check the elements?
                pass
            elif is_integer(i):
                if i >= len(ax):
                    raise IndexError("iloc cannot enlarge its target object")
            elif isinstance(i, dict):
                raise IndexError("iloc cannot enlarge its target object")

        return True

    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
        # this is a shortcut accessor to both .loc and .iloc
        # that provide the equivalent access of .at and .iat
        # a) avoid getting things via sections and (to minimize dtype changes)
        # b) provide a performant path
        if len(key) != self.ndim:
            return False

        return all(is_integer(k) for k in key)

    def _validate_integer(self, key: int | np.integer, axis: AxisInt) -> None:
        """
        Check that 'key' is a valid position in the desired axis.

        Parameters
        ----------
        key : int
            Requested position.
        axis : int
            Desired axis.

        Raises
        ------
        IndexError
            If 'key' is not a valid position in axis 'axis'.
        """
        len_axis = len(self.obj._get_axis(axis))
        if key >= len_axis or key < -len_axis:
            raise IndexError("single positional indexer is out-of-bounds")

    # -------------------------------------------------------------------

    def _getitem_tuple(self, tup: tuple):
        tup = self._validate_tuple_indexer(tup)
        with suppress(IndexingError):
            return self._getitem_lowerdim(tup)

        return self._getitem_tuple_same_dim(tup)

    def _get_list_axis(self, key, axis: AxisInt):
        """
        Return Series values by list or array of integers.

        Parameters
        ----------
        key : list-like positional indexer
        axis : int

        Returns
        -------
        Series object

        Notes
        -----
        `axis` can only be zero.
        """
        try:
            return self.obj._take_with_is_copy(key, axis=axis)
        except IndexError as err:
            # re-raise with different error message, e.g. test_getitem_ndarray_3d
            raise IndexError("positional indexers are out-of-bounds") from err

    def _getitem_axis(self, key, axis: AxisInt):
        if key is Ellipsis:
            key = slice(None)
        elif isinstance(key, ABCDataFrame):
            raise IndexError(
                "DataFrame indexer is not allowed for .iloc\n"
                "Consider using .loc for automatic alignment."
            )

        if isinstance(key, slice):
            return self._get_slice_axis(key, axis=axis)

        if is_iterator(key):
            key = list(key)

        if isinstance(key, list):
            key = np.asarray(key)

        if com.is_bool_indexer(key):
            self._validate_key(key, axis)
            return self._getbool_axis(key, axis=axis)

        # a list of integers
        elif is_list_like_indexer(key):
            return self._get_list_axis(key, axis=axis)

        # a single integer
        else:
            key = item_from_zerodim(key)
            if not is_integer(key):
                raise TypeError("Cannot index by location index with a non-integer key")

            # validate the location
            self._validate_integer(key, axis)

            return self.obj._ixs(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        # caller is responsible for ensuring non-None axis
        obj = self.obj

        if not need_slice(slice_obj):
            return obj.copy(deep=False)

        labels = obj._get_axis(axis)
        labels._validate_positional_slice(slice_obj)
        return self.obj._slice(slice_obj, axis=axis)

    def _convert_to_indexer(self, key, axis: AxisInt):
        """
        Much simpler as we only have to deal with our valid types.
        """
        return key

    def _get_setitem_indexer(self, key):
        # GH#32257 Fall through to let numpy do validation
        if is_iterator(key):
            key = list(key)

        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)

        return key

    # -------------------------------------------------------------------

    def _setitem_with_indexer(self, indexer, value, name: str = "iloc"):
        """
        _setitem_with_indexer is for setting values on a Series/DataFrame
        using positional indexers.

        If the relevant keys are not present, the Series/DataFrame may be
        expanded.

        This method is currently broken when dealing with non-unique Indexes,
        since it goes from positional indexers back to labels when calling
        BlockManager methods, see GH#12991, GH#22046, GH#15686.
        """
        info_axis = self.obj._info_axis_number

        # maybe partial set
        take_split_path = not self.obj._mgr.is_single_block

        if not take_split_path and isinstance(value, ABCDataFrame):
            # Avoid cast of values
            take_split_path = not value._mgr.is_single_block

        # if there is only one block/type, still have to take split path
        # unless the block is one-dimensional or it can hold the value
        if not take_split_path and len(self.obj._mgr.arrays) and self.ndim > 1:
            # in case of dict, keys are indices
            val = list(value.values()) if isinstance(value, dict) else value
            arr = self.obj._mgr.arrays[0]
            take_split_path = not can_hold_element(
                arr, extract_array(val, extract_numpy=True)
            )

        # if we have any multi-indexes that have non-trivial slices
        # (not null slices) then we must take the split path, xref
        # GH 10360, GH 27841
        if isinstance(indexer, tuple) and len(indexer) == len(self.obj.axes):
            for i, ax in zip(indexer, self.obj.axes):
                if isinstance(ax, MultiIndex) and not (
                    is_integer(i) or com.is_null_slice(i)
                ):
                    take_split_path = True
                    break

        if isinstance(indexer, tuple):
            nindexer = []
            for i, idx in enumerate(indexer):
                if isinstance(idx, dict):
                    # reindex the axis to the new value
                    # and set inplace
                    key, _ = convert_missing_indexer(idx)

                    # if this is the items axes, then take the main missing
                    # path first
                    # this correctly sets the dtype and avoids cache issues
                    # essentially this separates out the block that is needed
                    # to possibly be modified
                    if self.ndim > 1 and i == info_axis:
                        # add the new item, and set the value
                        # must have all defined axes if we have a scalar
                        # or a list-like on the non-info axes if we have a
                        # list-like
                        if not len(self.obj):
                            if not is_list_like_indexer(value):
                                raise ValueError(
                                    "cannot set a frame with no "
                                    "defined index and a scalar"
                                )
                            self.obj[key] = value
                            return

                        # add a new item with the dtype setup
                        if com.is_null_slice(indexer[0]):
                            # We are setting an entire column
                            self.obj[key] = value
                            return
                        elif is_array_like(value):
                            # GH#42099
                            arr = extract_array(value, extract_numpy=True)
                            taker = -1 * np.ones(len(self.obj), dtype=np.intp)
                            empty_value = algos.take_nd(arr, taker)
                            if not isinstance(value, ABCSeries):
                                # if not Series (in which case we need to align),
                                #  we can short-circuit
                                if (
                                    isinstance(arr, np.ndarray)
                                    and arr.ndim == 1
                                    and len(arr) == 1
                                ):
                                    # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
                                    arr = arr[0, ...]
                                empty_value[indexer[0]] = arr
                                self.obj[key] = empty_value
                                return

                            self.obj[key] = empty_value
                        elif not is_list_like(value):
                            self.obj[key] = construct_1d_array_from_inferred_fill_value(
                                value, len(self.obj)
                            )
                        else:
                            # FIXME: GH#42099#issuecomment-864326014
                            self.obj[key] = infer_fill_value(value)

                        new_indexer = convert_from_missing_indexer_tuple(
                            indexer, self.obj.axes
                        )
                        self._setitem_with_indexer(new_indexer, value, name)

                        return

                    # reindex the axis
                    # make sure to clear the cache because we are
                    # just replacing the block manager here
                    # so the object is the same
                    index = self.obj._get_axis(i)
                    with warnings.catch_warnings():
                        # TODO: re-issue this with setitem-specific message?
                        warnings.filterwarnings(
                            "ignore",
                            "The behavior of Index.insert with object-dtype "
                            "is deprecated",
                            category=FutureWarning,
                        )
                        labels = index.insert(len(index), key)

                    # We are expanding the Series/DataFrame values to match
                    #  the length of thenew index `labels`.  GH#40096 ensure
                    #  this is valid even if the index has duplicates.
                    taker = np.arange(len(index) + 1, dtype=np.intp)
                    taker[-1] = -1
                    reindexers = {i: (labels, taker)}
                    new_obj = self.obj._reindex_with_indexers(
                        reindexers, allow_dups=True
                    )
                    self.obj._mgr = new_obj._mgr
                    self.obj._maybe_update_cacher(clear=True)
                    self.obj._is_copy = None

                    nindexer.append(labels.get_loc(key))

                else:
                    nindexer.append(idx)

            indexer = tuple(nindexer)
        else:
            indexer, missing = convert_missing_indexer(indexer)

            if missing:
                self._setitem_with_indexer_missing(indexer, value)
                return

        if name == "loc":
            # must come after setting of missing
            indexer, value = self._maybe_mask_setitem_value(indexer, value)

        # align and set the values
        if take_split_path:
            # We have to operate column-wise
            self._setitem_with_indexer_split_path(indexer, value, name)
        else:
            self._setitem_single_block(indexer, value, name)

    def _setitem_with_indexer_split_path(self, indexer, value, name: str):
        """
        Setitem column-wise.
        """
        # Above we only set take_split_path to True for 2D cases
        assert self.ndim == 2

        if not isinstance(indexer, tuple):
            indexer = _tuplify(self.ndim, indexer)
        if len(indexer) > self.ndim:
            raise IndexError("too many indices for array")
        if isinstance(indexer[0], np.ndarray) and indexer[0].ndim > 2:
            raise ValueError(r"Cannot set values with ndim > 2")

        if (isinstance(value, ABCSeries) and name != "iloc") or isinstance(value, dict):
            from pandas import Series

            value = self._align_series(indexer, Series(value))

        # Ensure we have something we can iterate over
        info_axis = indexer[1]
        ilocs = self._ensure_iterable_column_indexer(info_axis)

        pi = indexer[0]
        lplane_indexer = length_of_indexer(pi, self.obj.index)
        # lplane_indexer gives the expected length of obj[indexer[0]]

        # we need an iterable, with a ndim of at least 1
        # eg. don't pass through np.array(0)
        if is_list_like_indexer(value) and getattr(value, "ndim", 1) > 0:
            if isinstance(value, ABCDataFrame):
                self._setitem_with_indexer_frame_value(indexer, value, name)

            elif np.ndim(value) == 2:
                # TODO: avoid np.ndim call in case it isn't an ndarray, since
                #  that will construct an ndarray, which will be wasteful
                self._setitem_with_indexer_2d_value(indexer, value)

            elif len(ilocs) == 1 and lplane_indexer == len(value) and not is_scalar(pi):
                # We are setting multiple rows in a single column.
                self._setitem_single_column(ilocs[0], value, pi)

            elif len(ilocs) == 1 and 0 != lplane_indexer != len(value):
                # We are trying to set N values into M entries of a single
                #  column, which is invalid for N != M
                # Exclude zero-len for e.g. boolean masking that is all-false

                if len(value) == 1 and not is_integer(info_axis):
                    # This is a case like df.iloc[:3, [1]] = [0]
                    #  where we treat as df.iloc[:3, 1] = 0
                    return self._setitem_with_indexer((pi, info_axis[0]), value[0])

                raise ValueError(
                    "Must have equal len keys and value "
                    "when setting with an iterable"
                )

            elif lplane_indexer == 0 and len(value) == len(self.obj.index):
                # We get here in one case via .loc with a all-False mask
                pass

            elif self._is_scalar_access(indexer) and is_object_dtype(
                self.obj.dtypes._values[ilocs[0]]
            ):
                # We are setting nested data, only possible for object dtype data
                self._setitem_single_column(indexer[1], value, pi)

            elif len(ilocs) == len(value):
                # We are setting multiple columns in a single row.
                for loc, v in zip(ilocs, value):
                    self._setitem_single_column(loc, v, pi)

            elif len(ilocs) == 1 and com.is_null_slice(pi) and len(self.obj) == 0:
                # This is a setitem-with-expansion, see
                #  test_loc_setitem_empty_append_expands_rows_mixed_dtype
                # e.g. df = DataFrame(columns=["x", "y"])
                #  df["x"] = df["x"].astype(np.int64)
                #  df.loc[:, "x"] = [1, 2, 3]
                self._setitem_single_column(ilocs[0], value, pi)

            else:
                raise ValueError(
                    "Must have equal len keys and value "
                    "when setting with an iterable"
                )

        else:
            # scalar value
            for loc in ilocs:
                self._setitem_single_column(loc, value, pi)

    def _setitem_with_indexer_2d_value(self, indexer, value):
        # We get here with np.ndim(value) == 2, excluding DataFrame,
        #  which goes through _setitem_with_indexer_frame_value
        pi = indexer[0]

        ilocs = self._ensure_iterable_column_indexer(indexer[1])

        if not is_array_like(value):
            # cast lists to array
            value = np.array(value, dtype=object)
        if len(ilocs) != value.shape[1]:
            raise ValueError(
                "Must have equal len keys and value when setting with an ndarray"
            )

        for i, loc in enumerate(ilocs):
            value_col = value[:, i]
            if is_object_dtype(value_col.dtype):
                # casting to list so that we do type inference in setitem_single_column
                value_col = value_col.tolist()
            self._setitem_single_column(loc, value_col, pi)

    def _setitem_with_indexer_frame_value(self, indexer, value: DataFrame, name: str):
        ilocs = self._ensure_iterable_column_indexer(indexer[1])

        sub_indexer = list(indexer)
        pi = indexer[0]

        multiindex_indexer = isinstance(self.obj.columns, MultiIndex)

        unique_cols = value.columns.is_unique

        # We do not want to align the value in case of iloc GH#37728
        if name == "iloc":
            for i, loc in enumerate(ilocs):
                val = value.iloc[:, i]
                self._setitem_single_column(loc, val, pi)

        elif not unique_cols and value.columns.equals(self.obj.columns):
            # We assume we are already aligned, see
            # test_iloc_setitem_frame_duplicate_columns_multiple_blocks
            for loc in ilocs:
                item = self.obj.columns[loc]
                if item in value:
                    sub_indexer[1] = item
                    val = self._align_series(
                        tuple(sub_indexer),
                        value.iloc[:, loc],
                        multiindex_indexer,
                    )
                else:
                    val = np.nan

                self._setitem_single_column(loc, val, pi)

        elif not unique_cols:
            raise ValueError("Setting with non-unique columns is not allowed.")

        else:
            for loc in ilocs:
                item = self.obj.columns[loc]
                if item in value:
                    sub_indexer[1] = item
                    val = self._align_series(
                        tuple(sub_indexer),
                        value[item],
                        multiindex_indexer,
                        using_cow=using_copy_on_write(),
                    )
                else:
                    val = np.nan

                self._setitem_single_column(loc, val, pi)

    def _setitem_single_column(self, loc: int, value, plane_indexer) -> None:
        """

        Parameters
        ----------
        loc : int
            Indexer for column position
        plane_indexer : int, slice, listlike[int]
            The indexer we use for setitem along axis=0.
        """
        pi = plane_indexer

        is_full_setter = com.is_null_slice(pi) or com.is_full_slice(pi, len(self.obj))

        is_null_setter = com.is_empty_slice(pi) or is_array_like(pi) and len(pi) == 0

        if is_null_setter:
            # no-op, don't cast dtype later
            return

        elif is_full_setter:
            try:
                self.obj._mgr.column_setitem(
                    loc, plane_indexer, value, inplace_only=True
                )
            except (ValueError, TypeError, LossySetitemError):
                # If we're setting an entire column and we can't do it inplace,
                #  then we can use value's dtype (or inferred dtype)
                #  instead of object
                dtype = self.obj.dtypes.iloc[loc]
                if dtype not in (np.void, object) and not self.obj.empty:
                    # - Exclude np.void, as that is a special case for expansion.
                    #   We want to warn for
                    #       df = pd.DataFrame({'a': [1, 2]})
                    #       df.loc[:, 'a'] = .3
                    #   but not for
                    #       df = pd.DataFrame({'a': [1, 2]})
                    #       df.loc[:, 'b'] = .3
                    # - Exclude `object`, as then no upcasting happens.
                    # - Exclude empty initial object with enlargement,
                    #   as then there's nothing to be inconsistent with.
                    warnings.warn(
                        f"Setting an item of incompatible dtype is deprecated "
                        "and will raise in a future error of pandas. "
                        f"Value '{value}' has dtype incompatible with {dtype}, "
                        "please explicitly cast to a compatible dtype first.",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )
                self.obj.isetitem(loc, value)
        else:
            # set value into the column (first attempting to operate inplace, then
            #  falling back to casting if necessary)
            dtype = self.obj.dtypes.iloc[loc]
            if dtype == np.void:
                # This means we're expanding, with multiple columns, e.g.
                #     df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
                #     df.loc[df.index <= 2, ['F', 'G']] = (1, 'abc')
                # Columns F and G will initially be set to np.void.
                # Here, we replace those temporary `np.void` columns with
                # columns of the appropriate dtype, based on `value`.
                self.obj.iloc[:, loc] = construct_1d_array_from_inferred_fill_value(
                    value, len(self.obj)
                )
            self.obj._mgr.column_setitem(loc, plane_indexer, value)

        self.obj._clear_item_cache()

    def _setitem_single_block(self, indexer, value, name: str) -> None:
        """
        _setitem_with_indexer for the case when we have a single Block.
        """
        from pandas import Series

        if (isinstance(value, ABCSeries) and name != "iloc") or isinstance(value, dict):
            # TODO(EA): ExtensionBlock.setitem this causes issues with
            # setting for extensionarrays that store dicts. Need to decide
            # if it's worth supporting that.
            value = self._align_series(indexer, Series(value))

        info_axis = self.obj._info_axis_number
        item_labels = self.obj._get_axis(info_axis)
        if isinstance(indexer, tuple):
            # if we are setting on the info axis ONLY
            # set using those methods to avoid block-splitting
            # logic here
            if (
                self.ndim == len(indexer) == 2
                and is_integer(indexer[1])
                and com.is_null_slice(indexer[0])
            ):
                col = item_labels[indexer[info_axis]]
                if len(item_labels.get_indexer_for([col])) == 1:
                    # e.g. test_loc_setitem_empty_append_expands_rows
                    loc = item_labels.get_loc(col)
                    self._setitem_single_column(loc, value, indexer[0])
                    return

            indexer = maybe_convert_ix(*indexer)  # e.g. test_setitem_frame_align

        if isinstance(value, ABCDataFrame) and name != "iloc":
            value = self._align_frame(indexer, value)._values

        # check for chained assignment
        self.obj._check_is_chained_assignment_possible()

        # actually do the set
        self.obj._mgr = self.obj._mgr.setitem(indexer=indexer, value=value)
        self.obj._maybe_update_cacher(clear=True, inplace=True)

    def _setitem_with_indexer_missing(self, indexer, value):
        """
        Insert new row(s) or column(s) into the Series or DataFrame.
        """
        from pandas import Series

        # reindex the axis to the new value
        # and set inplace
        if self.ndim == 1:
            index = self.obj.index
            with warnings.catch_warnings():
                # TODO: re-issue this with setitem-specific message?
                warnings.filterwarnings(
                    "ignore",
                    "The behavior of Index.insert with object-dtype is deprecated",
                    category=FutureWarning,
                )
                new_index = index.insert(len(index), indexer)

            # we have a coerced indexer, e.g. a float
            # that matches in an int64 Index, so
            # we will not create a duplicate index, rather
            # index to that element
            # e.g. 0.0 -> 0
            # GH#12246
            if index.is_unique:
                # pass new_index[-1:] instead if [new_index[-1]]
                #  so that we retain dtype
                new_indexer = index.get_indexer(new_index[-1:])
                if (new_indexer != -1).any():
                    # We get only here with loc, so can hard code
                    return self._setitem_with_indexer(new_indexer, value, "loc")

            # this preserves dtype of the value and of the object
            if not is_scalar(value):
                new_dtype = None

            elif is_valid_na_for_dtype(value, self.obj.dtype):
                if not is_object_dtype(self.obj.dtype):
                    # Every NA value is suitable for object, no conversion needed
                    value = na_value_for_dtype(self.obj.dtype, compat=False)

                new_dtype = maybe_promote(self.obj.dtype, value)[0]

            elif isna(value):
                new_dtype = None
            elif not self.obj.empty and not is_object_dtype(self.obj.dtype):
                # We should not cast, if we have object dtype because we can
                # set timedeltas into object series
                curr_dtype = self.obj.dtype
                curr_dtype = getattr(curr_dtype, "numpy_dtype", curr_dtype)
                new_dtype = maybe_promote(curr_dtype, value)[0]
            else:
                new_dtype = None

            new_values = Series([value], dtype=new_dtype)._values

            if len(self.obj._values):
                # GH#22717 handle casting compatibility that np.concatenate
                #  does incorrectly
                new_values = concat_compat([self.obj._values, new_values])
            self.obj._mgr = self.obj._constructor(
                new_values, index=new_index, name=self.obj.name
            )._mgr
            self.obj._maybe_update_cacher(clear=True)

        elif self.ndim == 2:
            if not len(self.obj.columns):
                # no columns and scalar
                raise ValueError("cannot set a frame with no defined columns")

            has_dtype = hasattr(value, "dtype")
            if isinstance(value, ABCSeries):
                # append a Series
                value = value.reindex(index=self.obj.columns, copy=True)
                value.name = indexer
            elif isinstance(value, dict):
                value = Series(
                    value, index=self.obj.columns, name=indexer, dtype=object
                )
            else:
                # a list-list
                if is_list_like_indexer(value):
                    # must have conforming columns
                    if len(value) != len(self.obj.columns):
                        raise ValueError("cannot set a row with mismatched columns")

                value = Series(value, index=self.obj.columns, name=indexer)

            if not len(self.obj):
                # We will ignore the existing dtypes instead of using
                #  internals.concat logic
                df = value.to_frame().T

                idx = self.obj.index
                if isinstance(idx, MultiIndex):
                    name = idx.names
                else:
                    name = idx.name

                df.index = Index([indexer], name=name)
                if not has_dtype:
                    # i.e. if we already had a Series or ndarray, keep that
                    #  dtype.  But if we had a list or dict, then do inference
                    df = df.infer_objects(copy=False)
                self.obj._mgr = df._mgr
            else:
                self.obj._mgr = self.obj._append(value)._mgr
            self.obj._maybe_update_cacher(clear=True)

    def _ensure_iterable_column_indexer(self, column_indexer):
        """
        Ensure that our column indexer is something that can be iterated over.
        """
        ilocs: Sequence[int | np.integer] | np.ndarray
        if is_integer(column_indexer):
            ilocs = [column_indexer]
        elif isinstance(column_indexer, slice):
            ilocs = np.arange(len(self.obj.columns))[column_indexer]
        elif (
            isinstance(column_indexer, np.ndarray) and column_indexer.dtype.kind == "b"
        ):
            ilocs = np.arange(len(column_indexer))[column_indexer]
        else:
            ilocs = column_indexer
        return ilocs

    def _align_series(
        self,
        indexer,
        ser: Series,
        multiindex_indexer: bool = False,
        using_cow: bool = False,
    ):
        """
        Parameters
        ----------
        indexer : tuple, slice, scalar
            Indexer used to get the locations that will be set to `ser`.
        ser : pd.Series
            Values to assign to the locations specified by `indexer`.
        multiindex_indexer : bool, optional
            Defaults to False. Should be set to True if `indexer` was from
            a `pd.MultiIndex`, to avoid unnecessary broadcasting.

        Returns
        -------
        `np.array` of `ser` broadcast to the appropriate shape for assignment
        to the locations selected by `indexer`
        """
        if isinstance(indexer, (slice, np.ndarray, list, Index)):
            indexer = (indexer,)

        if isinstance(indexer, tuple):
            # flatten np.ndarray indexers
            def ravel(i):
                return i.ravel() if isinstance(i, np.ndarray) else i

            indexer = tuple(map(ravel, indexer))

            aligners = [not com.is_null_slice(idx) for idx in indexer]
            sum_aligners = sum(aligners)
            single_aligner = sum_aligners == 1
            is_frame = self.ndim == 2
            obj = self.obj

            # are we a single alignable value on a non-primary
            # dim (e.g. panel: 1,2, or frame: 0) ?
            # hence need to align to a single axis dimension
            # rather that find all valid dims

            # frame
            if is_frame:
                single_aligner = single_aligner and aligners[0]

            # we have a frame, with multiple indexers on both axes; and a
            # series, so need to broadcast (see GH5206)
            if sum_aligners == self.ndim and all(is_sequence(_) for _ in indexer):
                ser_values = ser.reindex(obj.axes[0][indexer[0]], copy=True)._values

                # single indexer
                if len(indexer) > 1 and not multiindex_indexer:
                    len_indexer = len(indexer[1])
                    ser_values = (
                        np.tile(ser_values, len_indexer).reshape(len_indexer, -1).T
                    )

                return ser_values

            for i, idx in enumerate(indexer):
                ax = obj.axes[i]

                # multiple aligners (or null slices)
                if is_sequence(idx) or isinstance(idx, slice):
                    if single_aligner and com.is_null_slice(idx):
                        continue
                    new_ix = ax[idx]
                    if not is_list_like_indexer(new_ix):
                        new_ix = Index([new_ix])
                    else:
                        new_ix = Index(new_ix)
                    if ser.index.equals(new_ix):
                        if using_cow:
                            return ser
                        return ser._values.copy()

                    return ser.reindex(new_ix)._values

                # 2 dims
                elif single_aligner:
                    # reindex along index
                    ax = self.obj.axes[1]
                    if ser.index.equals(ax) or not len(ax):
                        return ser._values.copy()
                    return ser.reindex(ax)._values

        elif is_integer(indexer) and self.ndim == 1:
            if is_object_dtype(self.obj.dtype):
                return ser
            ax = self.obj._get_axis(0)

            if ser.index.equals(ax):
                return ser._values.copy()

            return ser.reindex(ax)._values[indexer]

        elif is_integer(indexer):
            ax = self.obj._get_axis(1)

            if ser.index.equals(ax):
                return ser._values.copy()

            return ser.reindex(ax)._values

        raise ValueError("Incompatible indexer with Series")

    def _align_frame(self, indexer, df: DataFrame) -> DataFrame:
        is_frame = self.ndim == 2

        if isinstance(indexer, tuple):
            idx, cols = None, None
            sindexers = []
            for i, ix in enumerate(indexer):
                ax = self.obj.axes[i]
                if is_sequence(ix) or isinstance(ix, slice):
                    if isinstance(ix, np.ndarray):
                        ix = ix.ravel()
                    if idx is None:
                        idx = ax[ix]
                    elif cols is None:
                        cols = ax[ix]
                    else:
                        break
                else:
                    sindexers.append(i)

            if idx is not None and cols is not None:
                if df.index.equals(idx) and df.columns.equals(cols):
                    val = df.copy()
                else:
                    val = df.reindex(idx, columns=cols)
                return val

        elif (isinstance(indexer, slice) or is_list_like_indexer(indexer)) and is_frame:
            ax = self.obj.index[indexer]
            if df.index.equals(ax):
                val = df.copy()
            else:
                # we have a multi-index and are trying to align
                # with a particular, level GH3738
                if (
                    isinstance(ax, MultiIndex)
                    and isinstance(df.index, MultiIndex)
                    and ax.nlevels != df.index.nlevels
                ):
                    raise TypeError(
                        "cannot align on a multi-index with out "
                        "specifying the join levels"
                    )

                val = df.reindex(index=ax)
            return val

        raise ValueError("Incompatible indexer with DataFrame")


class _ScalarAccessIndexer(NDFrameIndexerBase):
    """
    Access scalars quickly.
    """

    # sub-classes need to set _takeable
    _takeable: bool

    def _convert_key(self, key):
        raise AbstractMethodError(self)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            # we could have a convertible item here (e.g. Timestamp)
            if not is_list_like_indexer(key):
                key = (key,)
            else:
                raise ValueError("Invalid call for scalar access (getting)!")

        key = self._convert_key(key)
        return self.obj._get_value(*key, takeable=self._takeable)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            key = tuple(com.apply_if_callable(x, self.obj) for x in key)
        else:
            # scalar callable may return tuple
            key = com.apply_if_callable(key, self.obj)

        if not isinstance(key, tuple):
            key = _tuplify(self.ndim, key)
        key = list(self._convert_key(key))
        if len(key) != self.ndim:
            raise ValueError("Not enough indexers for scalar access (setting)!")

        self.obj._set_value(*key, value=value, takeable=self._takeable)


@doc(IndexingMixin.at)
class _AtIndexer(_ScalarAccessIndexer):
    _takeable = False

    def _convert_key(self, key):
        """
        Require they keys to be the same type as the index. (so we don't
        fallback)
        """
        # GH 26989
        # For series, unpacking key needs to result in the label.
        # This is already the case for len(key) == 1; e.g. (1,)
        if self.ndim == 1 and len(key) > 1:
            key = (key,)

        return key

    @property
    def _axes_are_unique(self) -> bool:
        # Only relevant for self.ndim == 2
        assert self.ndim == 2
        return self.obj.index.is_unique and self.obj.columns.is_unique

    def __getitem__(self, key):
        if self.ndim == 2 and not self._axes_are_unique:
            # GH#33041 fall back to .loc
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                raise ValueError("Invalid call for scalar access (getting)!")
            return self.obj.loc[key]

        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        if self.ndim == 2 and not self._axes_are_unique:
            # GH#33041 fall back to .loc
            if not isinstance(key, tuple) or not all(is_scalar(x) for x in key):
                raise ValueError("Invalid call for scalar access (setting)!")

            self.obj.loc[key] = value
            return

        return super().__setitem__(key, value)


@doc(IndexingMixin.iat)
class _iAtIndexer(_ScalarAccessIndexer):
    _takeable = True

    def _convert_key(self, key):
        """
        Require integer args. (and convert to label arguments)
        """
        for i in key:
            if not is_integer(i):
                raise ValueError("iAt based indexing can only have integer indexers")
        return key


def _tuplify(ndim: int, loc: Hashable) -> tuple[Hashable | slice, ...]:
    """
    Given an indexer for the first dimension, create an equivalent tuple
    for indexing over all dimensions.

    Parameters
    ----------
    ndim : int
    loc : object

    Returns
    -------
    tuple
    """
    _tup: list[Hashable | slice]
    _tup = [slice(None, None) for _ in range(ndim)]
    _tup[0] = loc
    return tuple(_tup)


def _tupleize_axis_indexer(ndim: int, axis: AxisInt, key) -> tuple:
    """
    If we have an axis, adapt the given key to be axis-independent.
    """
    new_key = [slice(None)] * ndim
    new_key[axis] = key
    return tuple(new_key)


def check_bool_indexer(index: Index, key) -> np.ndarray:
    """
    Check if key is a valid boolean indexer for an object with such index and
    perform reindexing or conversion if needed.

    This function assumes that is_bool_indexer(key) == True.

    Parameters
    ----------
    index : Index
        Index of the object on which the indexing is done.
    key : list-like
        Boolean indexer to check.

    Returns
    -------
    np.array
        Resulting key.

    Raises
    ------
    IndexError
        If the key does not have the same length as index.
    IndexingError
        If the index of the key is unalignable to index.
    """
    result = key
    if isinstance(key, ABCSeries) and not key.index.equals(index):
        indexer = result.index.get_indexer_for(index)
        if -1 in indexer:
            raise IndexingError(
                "Unalignable boolean Series provided as "
                "indexer (index of the boolean Series and of "
                "the indexed object do not match)."
            )

        result = result.take(indexer)

        # fall through for boolean
        if not isinstance(result.dtype, ExtensionDtype):
            return result.astype(bool)._values

    if is_object_dtype(key):
        # key might be object-dtype bool, check_array_indexer needs bool array
        result = np.asarray(result, dtype=bool)
    elif not is_array_like(result):
        # GH 33924
        # key may contain nan elements, check_array_indexer needs bool array
        result = pd_array(result, dtype=bool)
    return check_array_indexer(index, result)


def convert_missing_indexer(indexer):
    """
    Reverse convert a missing indexer, which is a dict
    return the scalar indexer and a boolean indicating if we converted
    """
    if isinstance(indexer, dict):
        # a missing key (but not a tuple indexer)
        indexer = indexer["key"]

        if isinstance(indexer, bool):
            raise KeyError("cannot use a single bool to index into setitem")
        return indexer, True

    return indexer, False


def convert_from_missing_indexer_tuple(indexer, axes):
    """
    Create a filtered indexer that doesn't have any missing indexers.
    """

    def get_indexer(_i, _idx):
        return axes[_i].get_loc(_idx["key"]) if isinstance(_idx, dict) else _idx

    return tuple(get_indexer(_i, _idx) for _i, _idx in enumerate(indexer))


def maybe_convert_ix(*args):
    """
    We likely want to take the cross-product.
    """
    for arg in args:
        if not isinstance(arg, (np.ndarray, list, ABCSeries, Index)):
            return args
    return np.ix_(*args)


def is_nested_tuple(tup, labels) -> bool:
    """
    Returns
    -------
    bool
    """
    # check for a compatible nested tuple and multiindexes among the axes
    if not isinstance(tup, tuple):
        return False

    for k in tup:
        if is_list_like(k) or isinstance(k, slice):
            return isinstance(labels, MultiIndex)

    return False


def is_label_like(key) -> bool:
    """
    Returns
    -------
    bool
    """
    # select a label or row
    return (
        not isinstance(key, slice)
        and not is_list_like_indexer(key)
        and key is not Ellipsis
    )


def need_slice(obj: slice) -> bool:
    """
    Returns
    -------
    bool
    """
    return (
        obj.start is not None
        or obj.stop is not None
        or (obj.step is not None and obj.step != 1)
    )


def check_dict_or_set_indexers(key) -> None:
    """
    Check if the indexer is or contains a dict or set, which is no longer allowed.
    """
    if (
        isinstance(key, set)
        or isinstance(key, tuple)
        and any(isinstance(x, set) for x in key)
    ):
        raise TypeError(
            "Passing a set as an indexer is not supported. Use a list instead."
        )

    if (
        isinstance(key, dict)
        or isinstance(key, tuple)
        and any(isinstance(x, dict) for x in key)
    ):
        raise TypeError(
            "Passing a dict as an indexer is not supported. Use a list instead."
        )