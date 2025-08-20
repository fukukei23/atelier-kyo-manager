
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\writeonly.py ===
# orm/writeonly.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Write-only collection API.

This is an alternate mapped attribute style that only supports single-item
collection mutation operations.   To read the collection, a select()
object must be executed each time.

.. versionadded:: 2.0


"""

from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from sqlalchemy.sql import bindparam
from . import attributes
from . import interfaces
from . import relationships
from . import strategies
from .base import NEVER_SET
from .base import object_mapper
from .base import PassiveFlag
from .base import RelationshipDirection
from .. import exc
from .. import inspect
from .. import log
from .. import util
from ..sql import delete
from ..sql import insert
from ..sql import select
from ..sql import update
from ..sql.dml import Delete
from ..sql.dml import Insert
from ..sql.dml import Update
from ..util.typing import Literal

if TYPE_CHECKING:
    from . import QueryableAttribute
    from ._typing import _InstanceDict
    from .attributes import AttributeEventToken
    from .base import LoaderCallableStatus
    from .collections import _AdaptedCollectionProtocol
    from .collections import CollectionAdapter
    from .mapper import Mapper
    from .relationships import _RelationshipOrderByArg
    from .state import InstanceState
    from .util import AliasedClass
    from ..event import _Dispatch
    from ..sql.selectable import FromClause
    from ..sql.selectable import Select

_T = TypeVar("_T", bound=Any)


class WriteOnlyHistory(Generic[_T]):
    """Overrides AttributeHistory to receive append/remove events directly."""

    unchanged_items: util.OrderedIdentitySet
    added_items: util.OrderedIdentitySet
    deleted_items: util.OrderedIdentitySet
    _reconcile_collection: bool

    def __init__(
        self,
        attr: WriteOnlyAttributeImpl,
        state: InstanceState[_T],
        passive: PassiveFlag,
        apply_to: Optional[WriteOnlyHistory[_T]] = None,
    ) -> None:
        if apply_to:
            if passive & PassiveFlag.SQL_OK:
                raise exc.InvalidRequestError(
                    f"Attribute {attr} can't load the existing state from the "
                    "database for this operation; full iteration is not "
                    "permitted.  If this is a delete operation, configure "
                    f"passive_deletes=True on the {attr} relationship in "
                    "order to resolve this error."
                )

            self.unchanged_items = apply_to.unchanged_items
            self.added_items = apply_to.added_items
            self.deleted_items = apply_to.deleted_items
            self._reconcile_collection = apply_to._reconcile_collection
        else:
            self.deleted_items = util.OrderedIdentitySet()
            self.added_items = util.OrderedIdentitySet()
            self.unchanged_items = util.OrderedIdentitySet()
            self._reconcile_collection = False

    @property
    def added_plus_unchanged(self) -> List[_T]:
        return list(self.added_items.union(self.unchanged_items))

    @property
    def all_items(self) -> List[_T]:
        return list(
            self.added_items.union(self.unchanged_items).union(
                self.deleted_items
            )
        )

    def as_history(self) -> attributes.History:
        if self._reconcile_collection:
            added = self.added_items.difference(self.unchanged_items)
            deleted = self.deleted_items.intersection(self.unchanged_items)
            unchanged = self.unchanged_items.difference(deleted)
        else:
            added, unchanged, deleted = (
                self.added_items,
                self.unchanged_items,
                self.deleted_items,
            )
        return attributes.History(list(added), list(unchanged), list(deleted))

    def indexed(self, index: Union[int, slice]) -> Union[List[_T], _T]:
        return list(self.added_items)[index]

    def add_added(self, value: _T) -> None:
        self.added_items.add(value)

    def add_removed(self, value: _T) -> None:
        if value in self.added_items:
            self.added_items.remove(value)
        else:
            self.deleted_items.add(value)


class WriteOnlyAttributeImpl(
    attributes.HasCollectionAdapter, attributes.AttributeImpl
):
    uses_objects: bool = True
    default_accepts_scalar_loader: bool = False
    supports_population: bool = False
    _supports_dynamic_iteration: bool = False
    collection: bool = False
    dynamic: bool = True
    order_by: _RelationshipOrderByArg = ()
    collection_history_cls: Type[WriteOnlyHistory[Any]] = WriteOnlyHistory

    query_class: Type[WriteOnlyCollection[Any]]

    def __init__(
        self,
        class_: Union[Type[Any], AliasedClass[Any]],
        key: str,
        dispatch: _Dispatch[QueryableAttribute[Any]],
        target_mapper: Mapper[_T],
        order_by: _RelationshipOrderByArg,
        **kw: Any,
    ):
        super().__init__(class_, key, None, dispatch, **kw)
        self.target_mapper = target_mapper
        self.query_class = WriteOnlyCollection
        if order_by:
            self.order_by = tuple(order_by)

    def get(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
    ) -> Union[util.OrderedIdentitySet, WriteOnlyCollection[Any]]:
        if not passive & PassiveFlag.SQL_OK:
            return self._get_collection_history(
                state, PassiveFlag.PASSIVE_NO_INITIALIZE
            ).added_items
        else:
            return self.query_class(self, state)

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
        data: Collection[Any]
        if not passive & PassiveFlag.SQL_OK:
            data = self._get_collection_history(state, passive).added_items
        else:
            history = self._get_collection_history(state, passive)
            data = history.added_plus_unchanged
        return DynamicCollectionAdapter(data)  # type: ignore[return-value]

    @util.memoized_property
    def _append_token(  # type:ignore[override]
        self,
    ) -> attributes.AttributeEventToken:
        return attributes.AttributeEventToken(self, attributes.OP_APPEND)

    @util.memoized_property
    def _remove_token(  # type:ignore[override]
        self,
    ) -> attributes.AttributeEventToken:
        return attributes.AttributeEventToken(self, attributes.OP_REMOVE)

    def fire_append_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        collection_history: Optional[WriteOnlyHistory[Any]] = None,
    ) -> None:
        if collection_history is None:
            collection_history = self._modified_event(state, dict_)

        collection_history.add_added(value)

        for fn in self.dispatch.append:
            value = fn(state, value, initiator or self._append_token)

        if self.trackparent and value is not None:
            self.sethasparent(attributes.instance_state(value), state, True)

    def fire_remove_event(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        collection_history: Optional[WriteOnlyHistory[Any]] = None,
    ) -> None:
        if collection_history is None:
            collection_history = self._modified_event(state, dict_)

        collection_history.add_removed(value)

        if self.trackparent and value is not None:
            self.sethasparent(attributes.instance_state(value), state, False)

        for fn in self.dispatch.remove:
            fn(state, value, initiator or self._remove_token)

    def _modified_event(
        self, state: InstanceState[Any], dict_: _InstanceDict
    ) -> WriteOnlyHistory[Any]:
        if self.key not in state.committed_state:
            state.committed_state[self.key] = self.collection_history_cls(
                self, state, PassiveFlag.PASSIVE_NO_FETCH
            )

        state._modified_event(dict_, self, NEVER_SET)

        # this is a hack to allow the entities.ComparableEntity fixture
        # to work
        dict_[self.key] = True
        return state.committed_state[self.key]  # type: ignore[no-any-return]

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
        if initiator and initiator.parent_token is self.parent_token:
            return

        if pop and value is None:
            return

        iterable = value
        new_values = list(iterable)
        if state.has_identity:
            if not self._supports_dynamic_iteration:
                raise exc.InvalidRequestError(
                    f'Collection "{self}" does not support implicit '
                    "iteration; collection replacement operations "
                    "can't be used"
                )
            old_collection = util.IdentitySet(
                self.get(state, dict_, passive=passive)
            )

        collection_history = self._modified_event(state, dict_)
        if not state.has_identity:
            old_collection = collection_history.added_items
        else:
            old_collection = old_collection.union(
                collection_history.added_items
            )

        constants = old_collection.intersection(new_values)
        additions = util.IdentitySet(new_values).difference(constants)
        removals = old_collection.difference(constants)

        for member in new_values:
            if member in additions:
                self.fire_append_event(
                    state,
                    dict_,
                    member,
                    None,
                    collection_history=collection_history,
                )

        for member in removals:
            self.fire_remove_event(
                state,
                dict_,
                member,
                None,
                collection_history=collection_history,
            )

    def delete(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError()

    def set_committed_value(
        self, state: InstanceState[Any], dict_: _InstanceDict, value: Any
    ) -> NoReturn:
        raise NotImplementedError(
            "Dynamic attributes don't support collection population."
        )

    def get_history(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PassiveFlag.PASSIVE_NO_FETCH,
    ) -> attributes.History:
        c = self._get_collection_history(state, passive)
        return c.as_history()

    def get_all_pending(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        passive: PassiveFlag = PassiveFlag.PASSIVE_NO_INITIALIZE,
    ) -> List[Tuple[InstanceState[Any], Any]]:
        c = self._get_collection_history(state, passive)
        return [(attributes.instance_state(x), x) for x in c.all_items]

    def _get_collection_history(
        self, state: InstanceState[Any], passive: PassiveFlag
    ) -> WriteOnlyHistory[Any]:
        c: WriteOnlyHistory[Any]
        if self.key in state.committed_state:
            c = state.committed_state[self.key]
        else:
            c = self.collection_history_cls(
                self, state, PassiveFlag.PASSIVE_NO_FETCH
            )

        if state.has_identity and (passive & PassiveFlag.INIT_OK):
            return self.collection_history_cls(
                self, state, passive, apply_to=c
            )
        else:
            return c

    def append(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PassiveFlag.PASSIVE_NO_FETCH,
    ) -> None:
        if initiator is not self:
            self.fire_append_event(state, dict_, value, initiator)

    def remove(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PassiveFlag.PASSIVE_NO_FETCH,
    ) -> None:
        if initiator is not self:
            self.fire_remove_event(state, dict_, value, initiator)

    def pop(
        self,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        value: Any,
        initiator: Optional[AttributeEventToken],
        passive: PassiveFlag = PassiveFlag.PASSIVE_NO_FETCH,
    ) -> None:
        self.remove(state, dict_, value, initiator, passive=passive)


@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy="write_only")
class WriteOnlyLoader(strategies.AbstractRelationshipLoader, log.Identified):
    impl_class = WriteOnlyAttributeImpl

    def init_class_attribute(self, mapper: Mapper[Any]) -> None:
        self.is_class_level = True
        if not self.uselist or self.parent_property.direction not in (
            interfaces.ONETOMANY,
            interfaces.MANYTOMANY,
        ):
            raise exc.InvalidRequestError(
                "On relationship %s, 'dynamic' loaders cannot be used with "
                "many-to-one/one-to-one relationships and/or "
                "uselist=False." % self.parent_property
            )

        strategies._register_attribute(  # type: ignore[no-untyped-call]
            self.parent_property,
            mapper,
            useobject=True,
            impl_class=self.impl_class,
            target_mapper=self.parent_property.mapper,
            order_by=self.parent_property.order_by,
            query_class=self.parent_property.query_class,
        )


class DynamicCollectionAdapter:
    """simplified CollectionAdapter for internal API consistency"""

    data: Collection[Any]

    def __init__(self, data: Collection[Any]):
        self.data = data

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def _reset_empty(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return True


class AbstractCollectionWriter(Generic[_T]):
    """Virtual collection which includes append/remove methods that synchronize
    into the attribute event system.

    """

    if not TYPE_CHECKING:
        __slots__ = ()

    instance: _T
    _from_obj: Tuple[FromClause, ...]

    def __init__(self, attr: WriteOnlyAttributeImpl, state: InstanceState[_T]):
        instance = state.obj()
        if TYPE_CHECKING:
            assert instance
        self.instance = instance
        self.attr = attr

        mapper = object_mapper(instance)
        prop = mapper._props[self.attr.key]

        if prop.secondary is not None:
            # this is a hack right now.  The Query only knows how to
            # make subsequent joins() without a given left-hand side
            # from self._from_obj[0].  We need to ensure prop.secondary
            # is in the FROM.  So we purposely put the mapper selectable
            # in _from_obj[0] to ensure a user-defined join() later on
            # doesn't fail, and secondary is then in _from_obj[1].

            # note also, we are using the official ORM-annotated selectable
            # from __clause_element__(), see #7868
            self._from_obj = (prop.mapper.__clause_element__(), prop.secondary)
        else:
            self._from_obj = ()

        self._where_criteria = (
            prop._with_parent(instance, alias_secondary=False),
        )

        if self.attr.order_by:
            self._order_by_clauses = self.attr.order_by
        else:
            self._order_by_clauses = ()

    def _add_all_impl(self, iterator: Iterable[_T]) -> None:
        for item in iterator:
            self.attr.append(
                attributes.instance_state(self.instance),
                attributes.instance_dict(self.instance),
                item,
                None,
            )

    def _remove_impl(self, item: _T) -> None:
        self.attr.remove(
            attributes.instance_state(self.instance),
            attributes.instance_dict(self.instance),
            item,
            None,
        )


class WriteOnlyCollection(AbstractCollectionWriter[_T]):
    """Write-only collection which can synchronize changes into the
    attribute event system.

    The :class:`.WriteOnlyCollection` is used in a mapping by
    using the ``"write_only"`` lazy loading strategy with
    :func:`_orm.relationship`.     For background on this configuration,
    see :ref:`write_only_relationship`.

    .. versionadded:: 2.0

    .. seealso::

        :ref:`write_only_relationship`

    """

    __slots__ = (
        "instance",
        "attr",
        "_where_criteria",
        "_from_obj",
        "_order_by_clauses",
    )

    def __iter__(self) -> NoReturn:
        raise TypeError(
            "WriteOnly collections don't support iteration in-place; "
            "to query for collection items, use the select() method to "
            "produce a SQL statement and execute it with session.scalars()."
        )

    def select(self) -> Select[Tuple[_T]]:
        """Produce a :class:`_sql.Select` construct that represents the
        rows within this instance-local :class:`_orm.WriteOnlyCollection`.

        """
        stmt = select(self.attr.target_mapper).where(*self._where_criteria)
        if self._from_obj:
            stmt = stmt.select_from(*self._from_obj)
        if self._order_by_clauses:
            stmt = stmt.order_by(*self._order_by_clauses)
        return stmt

    def insert(self) -> Insert:
        """For one-to-many collections, produce a :class:`_dml.Insert` which
        will insert new rows in terms of this this instance-local
        :class:`_orm.WriteOnlyCollection`.

        This construct is only supported for a :class:`_orm.Relationship`
        that does **not** include the :paramref:`_orm.relationship.secondary`
        parameter.  For relationships that refer to a many-to-many table,
        use ordinary bulk insert techniques to produce new objects, then
        use :meth:`_orm.AbstractCollectionWriter.add_all` to associate them
        with the collection.


        """

        state = inspect(self.instance)
        mapper = state.mapper
        prop = mapper._props[self.attr.key]

        if prop.direction is not RelationshipDirection.ONETOMANY:
            raise exc.InvalidRequestError(
                "Write only bulk INSERT only supported for one-to-many "
                "collections; for many-to-many, use a separate bulk "
                "INSERT along with add_all()."
            )

        dict_: Dict[str, Any] = {}

        for l, r in prop.synchronize_pairs:
            fn = prop._get_attr_w_warn_on_none(
                mapper,
                state,
                state.dict,
                l,
            )

            dict_[r.key] = bindparam(None, callable_=fn)

        return insert(self.attr.target_mapper).values(**dict_)

    def update(self) -> Update:
        """Produce a :class:`_dml.Update` which will refer to rows in terms
        of this instance-local :class:`_orm.WriteOnlyCollection`.

        """
        return update(self.attr.target_mapper).where(*self._where_criteria)

    def delete(self) -> Delete:
        """Produce a :class:`_dml.Delete` which will refer to rows in terms
        of this instance-local :class:`_orm.WriteOnlyCollection`.

        """
        return delete(self.attr.target_mapper).where(*self._where_criteria)

    def add_all(self, iterator: Iterable[_T]) -> None:
        """Add an iterable of items to this :class:`_orm.WriteOnlyCollection`.

        The given items will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        """
        self._add_all_impl(iterator)

    def add(self, item: _T) -> None:
        """Add an item to this :class:`_orm.WriteOnlyCollection`.

        The given item will be persisted to the database in terms of
        the parent instance's collection on the next flush.

        """
        self._add_all_impl([item])

    def remove(self, item: _T) -> None:
        """Remove an item from this :class:`_orm.WriteOnlyCollection`.

        The given item will be removed from the parent instance's collection on
        the next flush.

        """
        self._remove_impl(item)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\sqrtdenest.py ===
from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError


def is_sqrt(expr):
    """Return True if expr is a sqrt, otherwise False."""

    return expr.is_Pow and expr.exp.is_Rational and abs(expr.exp) is S.Half


def sqrt_depth(p) -> int:
    """Return the maximum depth of any square root argument of p.

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import sqrt_depth

    Neither of these square roots contains any other square roots
    so the depth is 1:

    >>> sqrt_depth(1 + sqrt(2)*(1 + sqrt(3)))
    1

    The sqrt(3) is contained within a square root so the depth is
    2:

    >>> sqrt_depth(1 + sqrt(2)*sqrt(1 + sqrt(3)))
    2
    """
    if p is S.ImaginaryUnit:
        return 1
    if p.is_Atom:
        return 0
    if p.is_Add or p.is_Mul:
        return max(sqrt_depth(x) for x in p.args)
    if is_sqrt(p):
        return sqrt_depth(p.base) + 1
    return 0


def is_algebraic(p):
    """Return True if p is comprised of only Rationals or square roots
    of Rationals and algebraic operations.

    Examples
    ========

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import is_algebraic
    >>> from sympy import cos
    >>> is_algebraic(sqrt(2)*(3/(sqrt(7) + sqrt(5)*sqrt(2))))
    True
    >>> is_algebraic(sqrt(2)*(3/(sqrt(7) + sqrt(5)*cos(2))))
    False
    """

    if p.is_Rational:
        return True
    elif p.is_Atom:
        return False
    elif is_sqrt(p) or p.is_Pow and p.exp.is_Integer:
        return is_algebraic(p.base)
    elif p.is_Add or p.is_Mul:
        return all(is_algebraic(x) for x in p.args)
    else:
        return False


def _subsets(n):
    """
    Returns all possible subsets of the set (0, 1, ..., n-1) except the
    empty set, listed in reversed lexicographical order according to binary
    representation, so that the case of the fourth root is treated last.

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import _subsets
    >>> _subsets(2)
    [[1, 0], [0, 1], [1, 1]]

    """
    if n == 1:
        a = [[1]]
    elif n == 2:
        a = [[1, 0], [0, 1], [1, 1]]
    elif n == 3:
        a = [[1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    else:
        b = _subsets(n - 1)
        a0 = [x + [0] for x in b]
        a1 = [x + [1] for x in b]
        a = a0 + [[0]*(n - 1) + [1]] + a1
    return a


def sqrtdenest(expr, max_iter=3):
    """Denests sqrts in an expression that contain other square roots
    if possible, otherwise returns the expr unchanged. This is based on the
    algorithms of [1].

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import sqrtdenest
    >>> from sympy import sqrt
    >>> sqrtdenest(sqrt(5 + 2 * sqrt(6)))
    sqrt(2) + sqrt(3)

    See Also
    ========

    sympy.solvers.solvers.unrad

    References
    ==========

    .. [1] https://web.archive.org/web/20210806201615/https://researcher.watson.ibm.com/researcher/files/us-fagin/symb85.pdf

    .. [2] D. J. Jeffrey and A. D. Rich, 'Symplifying Square Roots of Square Roots
           by Denesting' (available at https://www.cybertester.com/data/denest.pdf)

    """
    expr = expand_mul(expr)
    for i in range(max_iter):
        z = _sqrtdenest0(expr)
        if expr == z:
            return expr
        expr = z
    return expr


def _sqrt_match(p):
    """Return [a, b, r] for p.match(a + b*sqrt(r)) where, in addition to
    matching, sqrt(r) also has then maximal sqrt_depth among addends of p.

    Examples
    ========

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match
    >>> _sqrt_match(1 + sqrt(2) + sqrt(2)*sqrt(3) +  2*sqrt(1+sqrt(5)))
    [1 + sqrt(2) + sqrt(6), 2, 1 + sqrt(5)]
    """
    from sympy.simplify.radsimp import split_surds

    p = _mexpand(p)
    if p.is_Number:
        res = (p, S.Zero, S.Zero)
    elif p.is_Add:
        pargs = sorted(p.args, key=default_sort_key)
        sqargs = [x**2 for x in pargs]
        if all(sq.is_Rational and sq.is_positive for sq in sqargs):
            r, b, a = split_surds(p)
            res = a, b, r
            return list(res)
        # to make the process canonical, the argument is included in the tuple
        # so when the max is selected, it will be the largest arg having a
        # given depth
        v = [(sqrt_depth(x), x, i) for i, x in enumerate(pargs)]
        nmax = max(v, key=default_sort_key)
        if nmax[0] == 0:
            res = []
        else:
            # select r
            depth, _, i = nmax
            r = pargs.pop(i)
            v.pop(i)
            b = S.One
            if r.is_Mul:
                bv = []
                rv = []
                for x in r.args:
                    if sqrt_depth(x) < depth:
                        bv.append(x)
                    else:
                        rv.append(x)
                b = Mul._from_args(bv)
                r = Mul._from_args(rv)
            # collect terms containing r
            a1 = []
            b1 = [b]
            for x in v:
                if x[0] < depth:
                    a1.append(x[1])
                else:
                    x1 = x[1]
                    if x1 == r:
                        b1.append(1)
                    else:
                        if x1.is_Mul:
                            x1args = list(x1.args)
                            if r in x1args:
                                x1args.remove(r)
                                b1.append(Mul(*x1args))
                            else:
                                a1.append(x[1])
                        else:
                            a1.append(x[1])
            a = Add(*a1)
            b = Add(*b1)
            res = (a, b, r**2)
    else:
        b, r = p.as_coeff_Mul()
        if is_sqrt(r):
            res = (S.Zero, b, r**2)
        else:
            res = []
    return list(res)


class SqrtdenestStopIteration(StopIteration):
    pass


def _sqrtdenest0(expr):
    """Returns expr after denesting its arguments."""

    if is_sqrt(expr):
        n, d = expr.as_numer_denom()
        if d is S.One:  # n is a square root
            if n.base.is_Add:
                args = sorted(n.base.args, key=default_sort_key)
                if len(args) > 2 and all((x**2).is_Integer for x in args):
                    try:
                        return _sqrtdenest_rec(n)
                    except SqrtdenestStopIteration:
                        pass
                expr = sqrt(_mexpand(Add(*[_sqrtdenest0(x) for x in args])))
            return _sqrtdenest1(expr)
        else:
            n, d = [_sqrtdenest0(i) for i in (n, d)]
            return n/d

    if isinstance(expr, Add):
        cs = []
        args = []
        for arg in expr.args:
            c, a = arg.as_coeff_Mul()
            cs.append(c)
            args.append(a)

        if all(c.is_Rational for c in cs) and all(is_sqrt(arg) for arg in args):
            return _sqrt_ratcomb(cs, args)

    if isinstance(expr, Expr):
        args = expr.args
        if args:
            return expr.func(*[_sqrtdenest0(a) for a in args])
    return expr


def _sqrtdenest_rec(expr):
    """Helper that denests the square root of three or more surds.

    Explanation
    ===========

    It returns the denested expression; if it cannot be denested it
    throws SqrtdenestStopIteration

    Algorithm: expr.base is in the extension Q_m = Q(sqrt(r_1),..,sqrt(r_k));
    split expr.base = a + b*sqrt(r_k), where `a` and `b` are on
    Q_(m-1) = Q(sqrt(r_1),..,sqrt(r_(k-1))); then a**2 - b**2*r_k is
    on Q_(m-1); denest sqrt(a**2 - b**2*r_k) and so on.
    See [1], section 6.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrtdenest_rec
    >>> _sqrtdenest_rec(sqrt(-72*sqrt(2) + 158*sqrt(5) + 498))
    -sqrt(10) + sqrt(2) + 9 + 9*sqrt(5)
    >>> w=-6*sqrt(55)-6*sqrt(35)-2*sqrt(22)-2*sqrt(14)+2*sqrt(77)+6*sqrt(10)+65
    >>> _sqrtdenest_rec(sqrt(w))
    -sqrt(11) - sqrt(7) + sqrt(2) + 3*sqrt(5)
    """
    from sympy.simplify.radsimp import radsimp, rad_rationalize, split_surds
    if not expr.is_Pow:
        return sqrtdenest(expr)
    if expr.base < 0:
        return sqrt(-1)*_sqrtdenest_rec(sqrt(-expr.base))
    g, a, b = split_surds(expr.base)
    a = a*sqrt(g)
    if a < b:
        a, b = b, a
    c2 = _mexpand(a**2 - b**2)
    if len(c2.args) > 2:
        g, a1, b1 = split_surds(c2)
        a1 = a1*sqrt(g)
        if a1 < b1:
            a1, b1 = b1, a1
        c2_1 = _mexpand(a1**2 - b1**2)
        c_1 = _sqrtdenest_rec(sqrt(c2_1))
        d_1 = _sqrtdenest_rec(sqrt(a1 + c_1))
        num, den = rad_rationalize(b1, d_1)
        c = _mexpand(d_1/sqrt(2) + num/(den*sqrt(2)))
    else:
        c = _sqrtdenest1(sqrt(c2))

    if sqrt_depth(c) > 1:
        raise SqrtdenestStopIteration
    ac = a + c
    if len(ac.args) >= len(expr.args):
        if count_ops(ac) >= count_ops(expr.base):
            raise SqrtdenestStopIteration
    d = sqrtdenest(sqrt(ac))
    if sqrt_depth(d) > 1:
        raise SqrtdenestStopIteration
    num, den = rad_rationalize(b, d)
    r = d/sqrt(2) + num/(den*sqrt(2))
    r = radsimp(r)
    return _mexpand(r)


def _sqrtdenest1(expr, denester=True):
    """Return denested expr after denesting with simpler methods or, that
    failing, using the denester."""

    from sympy.simplify.simplify import radsimp

    if not is_sqrt(expr):
        return expr

    a = expr.base
    if a.is_Atom:
        return expr
    val = _sqrt_match(a)
    if not val:
        return expr

    a, b, r = val
    # try a quick numeric denesting
    d2 = _mexpand(a**2 - b**2*r)
    if d2.is_Rational:
        if d2.is_positive:
            z = _sqrt_numeric_denest(a, b, r, d2)
            if z is not None:
                return z
        else:
            # fourth root case
            # sqrtdenest(sqrt(3 + 2*sqrt(3))) =
            # sqrt(2)*3**(1/4)/2 + sqrt(2)*3**(3/4)/2
            dr2 = _mexpand(-d2*r)
            dr = sqrt(dr2)
            if dr.is_Rational:
                z = _sqrt_numeric_denest(_mexpand(b*r), a, r, dr2)
                if z is not None:
                    return z/root(r, 4)

    else:
        z = _sqrt_symbolic_denest(a, b, r)
        if z is not None:
            return z

    if not denester or not is_algebraic(expr):
        return expr

    res = sqrt_biquadratic_denest(expr, a, b, r, d2)
    if res:
        return res

    # now call to the denester
    av0 = [a, b, r, d2]
    z = _denester([radsimp(expr**2)], av0, 0, sqrt_depth(expr))[0]
    if av0[1] is None:
        return expr
    if z is not None:
        if sqrt_depth(z) == sqrt_depth(expr) and count_ops(z) > count_ops(expr):
            return expr
        return z
    return expr


def _sqrt_symbolic_denest(a, b, r):
    """Given an expression, sqrt(a + b*sqrt(b)), return the denested
    expression or None.

    Explanation
    ===========

    If r = ra + rb*sqrt(rr), try replacing sqrt(rr) in ``a`` with
    (y**2 - ra)/rb, and if the result is a quadratic, ca*y**2 + cb*y + cc, and
    (cb + b)**2 - 4*ca*cc is 0, then sqrt(a + b*sqrt(r)) can be rewritten as
    sqrt(ca*(sqrt(r) + (cb + b)/(2*ca))**2).

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import _sqrt_symbolic_denest, sqrtdenest
    >>> from sympy import sqrt, Symbol
    >>> from sympy.abc import x

    >>> a, b, r = 16 - 2*sqrt(29), 2, -10*sqrt(29) + 55
    >>> _sqrt_symbolic_denest(a, b, r)
    sqrt(11 - 2*sqrt(29)) + sqrt(5)

    If the expression is numeric, it will be simplified:

    >>> w = sqrt(sqrt(sqrt(3) + 1) + 1) + 1 + sqrt(2)
    >>> sqrtdenest(sqrt((w**2).expand()))
    1 + sqrt(2) + sqrt(1 + sqrt(1 + sqrt(3)))

    Otherwise, it will only be simplified if assumptions allow:

    >>> w = w.subs(sqrt(3), sqrt(x + 3))
    >>> sqrtdenest(sqrt((w**2).expand()))
    sqrt((sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2))**2)

    Notice that the argument of the sqrt is a square. If x is made positive
    then the sqrt of the square is resolved:

    >>> _.subs(x, Symbol('x', positive=True))
    sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2)
    """

    a, b, r = map(sympify, (a, b, r))
    rval = _sqrt_match(r)
    if not rval:
        return None
    ra, rb, rr = rval
    if rb:
        y = Dummy('y', positive=True)
        try:
            newa = Poly(a.subs(sqrt(rr), (y**2 - ra)/rb), y)
        except PolynomialError:
            return None
        if newa.degree() == 2:
            ca, cb, cc = newa.all_coeffs()
            cb += b
            if _mexpand(cb**2 - 4*ca*cc).equals(0):
                z = sqrt(ca*(sqrt(r) + cb/(2*ca))**2)
                if z.is_number:
                    z = _mexpand(Mul._from_args(z.as_content_primitive()))
                return z


def _sqrt_numeric_denest(a, b, r, d2):
    r"""Helper that denest
    $\sqrt{a + b \sqrt{r}}, d^2 = a^2 - b^2 r > 0$

    If it cannot be denested, it returns ``None``.
    """
    d = sqrt(d2)
    s = a + d
    # sqrt_depth(res) <= sqrt_depth(s) + 1
    # sqrt_depth(expr) = sqrt_depth(r) + 2
    # there is denesting if sqrt_depth(s) + 1 < sqrt_depth(r) + 2
    # if s**2 is Number there is a fourth root
    if sqrt_depth(s) < sqrt_depth(r) + 1 or (s**2).is_Rational:
        s1, s2 = sign(s), sign(b)
        if s1 == s2 == -1:
            s1 = s2 = 1
        res = (s1 * sqrt(a + d) + s2 * sqrt(a - d)) * sqrt(2) / 2
        return res.expand()


def sqrt_biquadratic_denest(expr, a, b, r, d2):
    """denest expr = sqrt(a + b*sqrt(r))
    where a, b, r are linear combinations of square roots of
    positive rationals on the rationals (SQRR) and r > 0, b != 0,
    d2 = a**2 - b**2*r > 0

    If it cannot denest it returns None.

    Explanation
    ===========

    Search for a solution A of type SQRR of the biquadratic equation
    4*A**4 - 4*a*A**2 + b**2*r = 0                               (1)
    sqd = sqrt(a**2 - b**2*r)
    Choosing the sqrt to be positive, the possible solutions are
    A = sqrt(a/2 +/- sqd/2)
    Since a, b, r are SQRR, then a**2 - b**2*r is a SQRR,
    so if sqd can be denested, it is done by
    _sqrtdenest_rec, and the result is a SQRR.
    Similarly for A.
    Examples of solutions (in both cases a and sqd are positive):

      Example of expr with solution sqrt(a/2 + sqd/2) but not
      solution sqrt(a/2 - sqd/2):
      expr = sqrt(-sqrt(15) - sqrt(2)*sqrt(-sqrt(5) + 5) - sqrt(3) + 8)
      a = -sqrt(15) - sqrt(3) + 8; sqd = -2*sqrt(5) - 2 + 4*sqrt(3)

      Example of expr with solution sqrt(a/2 - sqd/2) but not
      solution sqrt(a/2 + sqd/2):
      w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
      expr = sqrt((w**2).expand())
      a = 4*sqrt(6) + 8*sqrt(2) + 47 + 28*sqrt(3)
      sqd = 29 + 20*sqrt(3)

    Define B = b/2*A; eq.(1) implies a = A**2 + B**2*r; then
    expr**2 = a + b*sqrt(r) = (A + B*sqrt(r))**2

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match, sqrt_biquadratic_denest
    >>> z = sqrt((2*sqrt(2) + 4)*sqrt(2 + sqrt(2)) + 5*sqrt(2) + 8)
    >>> a, b, r = _sqrt_match(z**2)
    >>> d2 = a**2 - b**2*r
    >>> sqrt_biquadratic_denest(z, a, b, r, d2)
    sqrt(2) + sqrt(sqrt(2) + 2) + 2
    """
    from sympy.simplify.radsimp import radsimp, rad_rationalize
    if r <= 0 or d2 < 0 or not b or sqrt_depth(expr.base) < 2:
        return None
    for x in (a, b, r):
        for y in x.args:
            y2 = y**2
            if not y2.is_Integer or not y2.is_positive:
                return None
    sqd = _mexpand(sqrtdenest(sqrt(radsimp(d2))))
    if sqrt_depth(sqd) > 1:
        return None
    x1, x2 = [a/2 + sqd/2, a/2 - sqd/2]
    # look for a solution A with depth 1
    for x in (x1, x2):
        A = sqrtdenest(sqrt(x))
        if sqrt_depth(A) > 1:
            continue
        Bn, Bd = rad_rationalize(b, _mexpand(2*A))
        B = Bn/Bd
        z = A + B*sqrt(r)
        if z < 0:
            z = -z
        return _mexpand(z)
    return None


def _denester(nested, av0, h, max_depth_level):
    """Denests a list of expressions that contain nested square roots.

    Explanation
    ===========

    Algorithm based on <http://www.almaden.ibm.com/cs/people/fagin/symb85.pdf>.

    It is assumed that all of the elements of 'nested' share the same
    bottom-level radicand. (This is stated in the paper, on page 177, in
    the paragraph immediately preceding the algorithm.)

    When evaluating all of the arguments in parallel, the bottom-level
    radicand only needs to be denested once. This means that calling
    _denester with x arguments results in a recursive invocation with x+1
    arguments; hence _denester has polynomial complexity.

    However, if the arguments were evaluated separately, each call would
    result in two recursive invocations, and the algorithm would have
    exponential complexity.

    This is discussed in the paper in the middle paragraph of page 179.
    """
    from sympy.simplify.simplify import radsimp
    if h > max_depth_level:
        return None, None
    if av0[1] is None:
        return None, None
    if (av0[0] is None and
            all(n.is_Number for n in nested)):  # no arguments are nested
        for f in _subsets(len(nested)):  # test subset 'f' of nested
            p = _mexpand(Mul(*[nested[i] for i in range(len(f)) if f[i]]))
            if f.count(1) > 1 and f[-1]:
                p = -p
            sqp = sqrt(p)
            if sqp.is_Rational:
                return sqp, f  # got a perfect square so return its square root.
        # Otherwise, return the radicand from the previous invocation.
        return sqrt(nested[-1]), [0]*len(nested)
    else:
        R = None
        if av0[0] is not None:
            values = [av0[:2]]
            R = av0[2]
            nested2 = [av0[3], R]
            av0[0] = None
        else:
            values = list(filter(None, [_sqrt_match(expr) for expr in nested]))
            for v in values:
                if v[2]:  # Since if b=0, r is not defined
                    if R is not None:
                        if R != v[2]:
                            av0[1] = None
                            return None, None
                    else:
                        R = v[2]
            if R is None:
                # return the radicand from the previous invocation
                return sqrt(nested[-1]), [0]*len(nested)
            nested2 = [_mexpand(v[0]**2) -
                       _mexpand(R*v[1]**2) for v in values] + [R]
        d, f = _denester(nested2, av0, h + 1, max_depth_level)
        if not f:
            return None, None
        if not any(f[i] for i in range(len(nested))):
            v = values[-1]
            return sqrt(v[0] + _mexpand(v[1]*d)), f
        else:
            p = Mul(*[nested[i] for i in range(len(nested)) if f[i]])
            v = _sqrt_match(p)
            if 1 in f and f.index(1) < len(nested) - 1 and f[len(nested) - 1]:
                v[0] = -v[0]
                v[1] = -v[1]
            if not f[len(nested)]:  # Solution denests with square roots
                vad = _mexpand(v[0] + d)
                if vad <= 0:
                    # return the radicand from the previous invocation.
                    return sqrt(nested[-1]), [0]*len(nested)
                if not(sqrt_depth(vad) <= sqrt_depth(R) + 1 or
                       (vad**2).is_Number):
                    av0[1] = None
                    return None, None

                sqvad = _sqrtdenest1(sqrt(vad), denester=False)
                if not (sqrt_depth(sqvad) <= sqrt_depth(R) + 1):
                    av0[1] = None
                    return None, None
                sqvad1 = radsimp(1/sqvad)
                res = _mexpand(sqvad/sqrt(2) + (v[1]*sqrt(R)*sqvad1/sqrt(2)))
                return res, f

                      #          sign(v[1])*sqrt(_mexpand(v[1]**2*R*vad1/2))), f
            else:  # Solution requires a fourth root
                s2 = _mexpand(v[1]*R) + d
                if s2 <= 0:
                    return sqrt(nested[-1]), [0]*len(nested)
                FR, s = root(_mexpand(R), 4), sqrt(s2)
                return _mexpand(s/(sqrt(2)*FR) + v[0]*FR/(sqrt(2)*s)), f


def _sqrt_ratcomb(cs, args):
    """Denest rational combinations of radicals.

    Based on section 5 of [1].

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import sqrtdenest
    >>> z = sqrt(1+sqrt(3)) + sqrt(3+3*sqrt(3)) - sqrt(10+6*sqrt(3))
    >>> sqrtdenest(z)
    0
    """
    from sympy.simplify.radsimp import radsimp

    # check if there exists a pair of sqrt that can be denested
    def find(a):
        n = len(a)
        for i in range(n - 1):
            for j in range(i + 1, n):
                s1 = a[i].base
                s2 = a[j].base
                p = _mexpand(s1 * s2)
                s = sqrtdenest(sqrt(p))
                if s != sqrt(p):
                    return s, i, j

    indices = find(args)
    if indices is None:
        return Add(*[c * arg for c, arg in zip(cs, args)])

    s, i1, i2 = indices

    c2 = cs.pop(i2)
    args.pop(i2)
    a1 = args[i1]

    # replace a2 by s/a1
    cs[i1] += radsimp(c2 * s / a1.base)

    return _sqrt_ratcomb(cs, args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\reflection.py ===
# dialects/mysql/reflection.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


import re

from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util


class ReflectedState:
    """Stores raw information about a SHOW CREATE TABLE statement."""

    def __init__(self):
        self.columns = []
        self.table_options = {}
        self.table_name = None
        self.keys = []
        self.fk_constraints = []
        self.ck_constraints = []


@log.class_logger
class MySQLTableDefinitionParser:
    """Parses the results of a SHOW CREATE TABLE statement."""

    def __init__(self, dialect, preparer):
        self.dialect = dialect
        self.preparer = preparer
        self._prep_regexes()

    def parse(self, show_create, charset):
        state = ReflectedState()
        state.charset = charset
        for line in re.split(r"\r?\n", show_create):
            if line.startswith("  " + self.preparer.initial_quote):
                self._parse_column(line, state)
            # a regular table options line
            elif line.startswith(") "):
                self._parse_table_options(line, state)
            # an ANSI-mode table options line
            elif line == ")":
                pass
            elif line.startswith("CREATE "):
                self._parse_table_name(line, state)
            elif "PARTITION" in line:
                self._parse_partition_options(line, state)
            # Not present in real reflection, but may be if
            # loading from a file.
            elif not line:
                pass
            else:
                type_, spec = self._parse_constraints(line)
                if type_ is None:
                    util.warn("Unknown schema content: %r" % line)
                elif type_ == "key":
                    state.keys.append(spec)
                elif type_ == "fk_constraint":
                    state.fk_constraints.append(spec)
                elif type_ == "ck_constraint":
                    state.ck_constraints.append(spec)
                else:
                    pass
        return state

    def _check_view(self, sql: str) -> bool:
        return bool(self._re_is_view.match(sql))

    def _parse_constraints(self, line):
        """Parse a KEY or CONSTRAINT line.

        :param line: A line of SHOW CREATE TABLE output
        """

        # KEY
        m = self._re_key.match(line)
        if m:
            spec = m.groupdict()
            # convert columns into name, length pairs
            # NOTE: we may want to consider SHOW INDEX as the
            # format of indexes in MySQL becomes more complex
            spec["columns"] = self._parse_keyexprs(spec["columns"])
            if spec["version_sql"]:
                m2 = self._re_key_version_sql.match(spec["version_sql"])
                if m2 and m2.groupdict()["parser"]:
                    spec["parser"] = m2.groupdict()["parser"]
            if spec["parser"]:
                spec["parser"] = self.preparer.unformat_identifiers(
                    spec["parser"]
                )[0]
            return "key", spec

        # FOREIGN KEY CONSTRAINT
        m = self._re_fk_constraint.match(line)
        if m:
            spec = m.groupdict()
            spec["table"] = self.preparer.unformat_identifiers(spec["table"])
            spec["local"] = [c[0] for c in self._parse_keyexprs(spec["local"])]
            spec["foreign"] = [
                c[0] for c in self._parse_keyexprs(spec["foreign"])
            ]
            return "fk_constraint", spec

        # CHECK constraint
        m = self._re_ck_constraint.match(line)
        if m:
            spec = m.groupdict()
            return "ck_constraint", spec

        # PARTITION and SUBPARTITION
        m = self._re_partition.match(line)
        if m:
            # Punt!
            return "partition", line

        # No match.
        return (None, line)

    def _parse_table_name(self, line, state):
        """Extract the table name.

        :param line: The first line of SHOW CREATE TABLE
        """

        regex, cleanup = self._pr_name
        m = regex.match(line)
        if m:
            state.table_name = cleanup(m.group("name"))

    def _parse_table_options(self, line, state):
        """Build a dictionary of all reflected table-level options.

        :param line: The final line of SHOW CREATE TABLE output.
        """

        options = {}

        if line and line != ")":
            rest_of_line = line
            for regex, cleanup in self._pr_options:
                m = regex.search(rest_of_line)
                if not m:
                    continue
                directive, value = m.group("directive"), m.group("val")
                if cleanup:
                    value = cleanup(value)
                options[directive.lower()] = value
                rest_of_line = regex.sub("", rest_of_line)

        for nope in ("auto_increment", "data directory", "index directory"):
            options.pop(nope, None)

        for opt, val in options.items():
            state.table_options["%s_%s" % (self.dialect.name, opt)] = val

    def _parse_partition_options(self, line, state):
        options = {}
        new_line = line[:]

        while new_line.startswith("(") or new_line.startswith(" "):
            new_line = new_line[1:]

        for regex, cleanup in self._pr_options:
            m = regex.search(new_line)
            if not m or "PARTITION" not in regex.pattern:
                continue

            directive = m.group("directive")
            directive = directive.lower()
            is_subpartition = directive == "subpartition"

            if directive == "partition" or is_subpartition:
                new_line = new_line.replace(") */", "")
                new_line = new_line.replace(",", "")
                if is_subpartition and new_line.endswith(")"):
                    new_line = new_line[:-1]
                if self.dialect.name == "mariadb" and new_line.endswith(")"):
                    if (
                        "MAXVALUE" in new_line
                        or "MINVALUE" in new_line
                        or "ENGINE" in new_line
                    ):
                        # final line of MariaDB partition endswith ")"
                        new_line = new_line[:-1]

                defs = "%s_%s_definitions" % (self.dialect.name, directive)
                options[defs] = new_line

            else:
                directive = directive.replace(" ", "_")
                value = m.group("val")
                if cleanup:
                    value = cleanup(value)
                options[directive] = value
            break

        for opt, val in options.items():
            part_def = "%s_partition_definitions" % (self.dialect.name)
            subpart_def = "%s_subpartition_definitions" % (self.dialect.name)
            if opt == part_def or opt == subpart_def:
                # builds a string of definitions
                if opt not in state.table_options:
                    state.table_options[opt] = val
                else:
                    state.table_options[opt] = "%s, %s" % (
                        state.table_options[opt],
                        val,
                    )
            else:
                state.table_options["%s_%s" % (self.dialect.name, opt)] = val

    def _parse_column(self, line, state):
        """Extract column details.

        Falls back to a 'minimal support' variant if full parse fails.

        :param line: Any column-bearing line from SHOW CREATE TABLE
        """

        spec = None
        m = self._re_column.match(line)
        if m:
            spec = m.groupdict()
            spec["full"] = True
        else:
            m = self._re_column_loose.match(line)
            if m:
                spec = m.groupdict()
                spec["full"] = False
        if not spec:
            util.warn("Unknown column definition %r" % line)
            return
        if not spec["full"]:
            util.warn("Incomplete reflection of column definition %r" % line)

        name, type_, args = spec["name"], spec["coltype"], spec["arg"]

        try:
            col_type = self.dialect.ischema_names[type_]
        except KeyError:
            util.warn(
                "Did not recognize type '%s' of column '%s'" % (type_, name)
            )
            col_type = sqltypes.NullType

        # Column type positional arguments eg. varchar(32)
        if args is None or args == "":
            type_args = []
        elif args[0] == "'" and args[-1] == "'":
            type_args = self._re_csv_str.findall(args)
        else:
            type_args = [int(v) for v in self._re_csv_int.findall(args)]

        # Column type keyword options
        type_kw = {}

        if issubclass(col_type, (DATETIME, TIME, TIMESTAMP)):
            if type_args:
                type_kw["fsp"] = type_args.pop(0)

        for kw in ("unsigned", "zerofill"):
            if spec.get(kw, False):
                type_kw[kw] = True
        for kw in ("charset", "collate"):
            if spec.get(kw, False):
                type_kw[kw] = spec[kw]
        if issubclass(col_type, (ENUM, SET)):
            type_args = _strip_values(type_args)

            if issubclass(col_type, SET) and "" in type_args:
                type_kw["retrieve_as_bitwise"] = True

        type_instance = col_type(*type_args, **type_kw)

        col_kw = {}

        # NOT NULL
        col_kw["nullable"] = True
        # this can be "NULL" in the case of TIMESTAMP
        if spec.get("notnull", False) == "NOT NULL":
            col_kw["nullable"] = False
        # For generated columns, the nullability is marked in a different place
        if spec.get("notnull_generated", False) == "NOT NULL":
            col_kw["nullable"] = False

        # AUTO_INCREMENT
        if spec.get("autoincr", False):
            col_kw["autoincrement"] = True
        elif issubclass(col_type, sqltypes.Integer):
            col_kw["autoincrement"] = False

        # DEFAULT
        default = spec.get("default", None)

        if default == "NULL":
            # eliminates the need to deal with this later.
            default = None

        comment = spec.get("comment", None)

        if comment is not None:
            comment = cleanup_text(comment)

        sqltext = spec.get("generated")
        if sqltext is not None:
            computed = dict(sqltext=sqltext)
            persisted = spec.get("persistence")
            if persisted is not None:
                computed["persisted"] = persisted == "STORED"
            col_kw["computed"] = computed

        col_d = dict(
            name=name, type=type_instance, default=default, comment=comment
        )
        col_d.update(col_kw)
        state.columns.append(col_d)

    def _describe_to_create(self, table_name, columns):
        """Re-format DESCRIBE output as a SHOW CREATE TABLE string.

        DESCRIBE is a much simpler reflection and is sufficient for
        reflecting views for runtime use.  This method formats DDL
        for columns only- keys are omitted.

        :param columns: A sequence of DESCRIBE or SHOW COLUMNS 6-tuples.
          SHOW FULL COLUMNS FROM rows must be rearranged for use with
          this function.
        """

        buffer = []
        for row in columns:
            (name, col_type, nullable, default, extra) = (
                row[i] for i in (0, 1, 2, 4, 5)
            )

            line = [" "]
            line.append(self.preparer.quote_identifier(name))
            line.append(col_type)
            if not nullable:
                line.append("NOT NULL")
            if default:
                if "auto_increment" in default:
                    pass
                elif col_type.startswith("timestamp") and default.startswith(
                    "C"
                ):
                    line.append("DEFAULT")
                    line.append(default)
                elif default == "NULL":
                    line.append("DEFAULT")
                    line.append(default)
                else:
                    line.append("DEFAULT")
                    line.append("'%s'" % default.replace("'", "''"))
            if extra:
                line.append(extra)

            buffer.append(" ".join(line))

        return "".join(
            [
                (
                    "CREATE TABLE %s (\n"
                    % self.preparer.quote_identifier(table_name)
                ),
                ",\n".join(buffer),
                "\n) ",
            ]
        )

    def _parse_keyexprs(self, identifiers):
        """Unpack '"col"(2),"col" ASC'-ish strings into components."""

        return [
            (colname, int(length) if length else None, modifiers)
            for colname, length, modifiers in self._re_keyexprs.findall(
                identifiers
            )
        ]

    def _prep_regexes(self):
        """Pre-compile regular expressions."""

        self._re_columns = []
        self._pr_options = []

        _final = self.preparer.final_quote

        quotes = dict(
            zip(
                ("iq", "fq", "esc_fq"),
                [
                    re.escape(s)
                    for s in (
                        self.preparer.initial_quote,
                        _final,
                        self.preparer._escape_identifier(_final),
                    )
                ],
            )
        )

        self._pr_name = _pr_compile(
            r"^CREATE (?:\w+ +)?TABLE +"
            r"%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +\($" % quotes,
            self.preparer._unescape_identifier,
        )

        self._re_is_view = _re_compile(r"^CREATE(?! TABLE)(\s.*)?\sVIEW")

        # `col`,`col2`(32),`col3`(15) DESC
        #
        self._re_keyexprs = _re_compile(
            r"(?:"
            r"(?:%(iq)s((?:%(esc_fq)s|[^%(fq)s])+)%(fq)s)"
            r"(?:\((\d+)\))?(?: +(ASC|DESC))?(?=\,|$))+" % quotes
        )

        # 'foo' or 'foo','bar' or 'fo,o','ba''a''r'
        self._re_csv_str = _re_compile(r"\x27(?:\x27\x27|[^\x27])*\x27")

        # 123 or 123,456
        self._re_csv_int = _re_compile(r"\d+")

        # `colname` <type> [type opts]
        #  (NOT NULL | NULL)
        #   DEFAULT ('value' | CURRENT_TIMESTAMP...)
        #   COMMENT 'comment'
        #  COLUMN_FORMAT (FIXED|DYNAMIC|DEFAULT)
        #  STORAGE (DISK|MEMORY)
        self._re_column = _re_compile(
            r"  "
            r"%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +"
            r"(?P<coltype>\w+)"
            r"(?:\((?P<arg>(?:\d+|\d+,\d+|"
            r"(?:'(?:''|[^'])*',?)+))\))?"
            r"(?: +(?P<unsigned>UNSIGNED))?"
            r"(?: +(?P<zerofill>ZEROFILL))?"
            r"(?: +CHARACTER SET +(?P<charset>[\w_]+))?"
            r"(?: +COLLATE +(?P<collate>[\w_]+))?"
            r"(?: +(?P<notnull>(?:NOT )?NULL))?"
            r"(?: +DEFAULT +(?P<default>"
            r"(?:NULL|'(?:''|[^'])*'|\(.+?\)|[\-\w\.\(\)]+"
            r"(?: +ON UPDATE [\-\w\.\(\)]+)?)"
            r"))?"
            r"(?: +(?:GENERATED ALWAYS)? ?AS +(?P<generated>\("
            r".*\))? ?(?P<persistence>VIRTUAL|STORED)?"
            r"(?: +(?P<notnull_generated>(?:NOT )?NULL))?"
            r")?"
            r"(?: +(?P<autoincr>AUTO_INCREMENT))?"
            r"(?: +COMMENT +'(?P<comment>(?:''|[^'])*)')?"
            r"(?: +COLUMN_FORMAT +(?P<colfmt>\w+))?"
            r"(?: +STORAGE +(?P<storage>\w+))?"
            r"(?: +(?P<extra>.*))?"
            r",?$" % quotes
        )

        # Fallback, try to parse as little as possible
        self._re_column_loose = _re_compile(
            r"  "
            r"%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +"
            r"(?P<coltype>\w+)"
            r"(?:\((?P<arg>(?:\d+|\d+,\d+|\x27(?:\x27\x27|[^\x27])+\x27))\))?"
            r".*?(?P<notnull>(?:NOT )NULL)?" % quotes
        )

        # (PRIMARY|UNIQUE|FULLTEXT|SPATIAL) INDEX `name` (USING (BTREE|HASH))?
        # (`col` (ASC|DESC)?, `col` (ASC|DESC)?)
        # KEY_BLOCK_SIZE size | WITH PARSER name  /*!50100 WITH PARSER name */
        self._re_key = _re_compile(
            r"  "
            r"(?:(?P<type>\S+) )?KEY"
            r"(?: +%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s)?"
            r"(?: +USING +(?P<using_pre>\S+))?"
            r" +\((?P<columns>.+?)\)"
            r"(?: +USING +(?P<using_post>\S+))?"
            r"(?: +KEY_BLOCK_SIZE *[ =]? *(?P<keyblock>\S+))?"
            r"(?: +WITH PARSER +(?P<parser>\S+))?"
            r"(?: +COMMENT +(?P<comment>(\x27\x27|\x27([^\x27])*?\x27)+))?"
            r"(?: +/\*(?P<version_sql>.+)\*/ *)?"
            r",?$" % quotes
        )

        # https://forums.mysql.com/read.php?20,567102,567111#msg-567111
        # It means if the MySQL version >= \d+, execute what's in the comment
        self._re_key_version_sql = _re_compile(
            r"\!\d+ " r"(?: *WITH PARSER +(?P<parser>\S+) *)?"
        )

        # CONSTRAINT `name` FOREIGN KEY (`local_col`)
        # REFERENCES `remote` (`remote_col`)
        # MATCH FULL | MATCH PARTIAL | MATCH SIMPLE
        # ON DELETE CASCADE ON UPDATE RESTRICT
        #
        # unique constraints come back as KEYs
        kw = quotes.copy()
        kw["on"] = "RESTRICT|CASCADE|SET NULL|NO ACTION|SET DEFAULT"
        self._re_fk_constraint = _re_compile(
            r"  "
            r"CONSTRAINT +"
            r"%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +"
            r"FOREIGN KEY +"
            r"\((?P<local>[^\)]+?)\) REFERENCES +"
            r"(?P<table>%(iq)s[^%(fq)s]+%(fq)s"
            r"(?:\.%(iq)s[^%(fq)s]+%(fq)s)?) +"
            r"\((?P<foreign>(?:%(iq)s[^%(fq)s]+%(fq)s(?: *, *)?)+)\)"
            r"(?: +(?P<match>MATCH \w+))?"
            r"(?: +ON DELETE (?P<ondelete>%(on)s))?"
            r"(?: +ON UPDATE (?P<onupdate>%(on)s))?" % kw
        )

        # CONSTRAINT `CONSTRAINT_1` CHECK (`x` > 5)'
        # testing on MariaDB 10.2 shows that the CHECK constraint
        # is returned on a line by itself, so to match without worrying
        # about parenthesis in the expression we go to the end of the line
        self._re_ck_constraint = _re_compile(
            r"  "
            r"CONSTRAINT +"
            r"%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +"
            r"CHECK +"
            r"\((?P<sqltext>.+)\),?" % kw
        )

        # PARTITION
        #
        # punt!
        self._re_partition = _re_compile(r"(?:.*)(?:SUB)?PARTITION(?:.*)")

        # Table-level options (COLLATE, ENGINE, etc.)
        # Do the string options first, since they have quoted
        # strings we need to get rid of.
        for option in _options_of_type_string:
            self._add_option_string(option)

        for option in (
            "ENGINE",
            "TYPE",
            "AUTO_INCREMENT",
            "AVG_ROW_LENGTH",
            "CHARACTER SET",
            "DEFAULT CHARSET",
            "CHECKSUM",
            "COLLATE",
            "DELAY_KEY_WRITE",
            "INSERT_METHOD",
            "MAX_ROWS",
            "MIN_ROWS",
            "PACK_KEYS",
            "ROW_FORMAT",
            "KEY_BLOCK_SIZE",
            "STATS_SAMPLE_PAGES",
        ):
            self._add_option_word(option)

        for option in (
            "PARTITION BY",
            "SUBPARTITION BY",
            "PARTITIONS",
            "SUBPARTITIONS",
            "PARTITION",
            "SUBPARTITION",
        ):
            self._add_partition_option_word(option)

        self._add_option_regex("UNION", r"\([^\)]+\)")
        self._add_option_regex("TABLESPACE", r".*? STORAGE DISK")
        self._add_option_regex(
            "RAID_TYPE",
            r"\w+\s+RAID_CHUNKS\s*\=\s*\w+RAID_CHUNKSIZE\s*=\s*\w+",
        )

    _optional_equals = r"(?:\s*(?:=\s*)|\s+)"

    def _add_option_string(self, directive):
        regex = r"(?P<directive>%s)%s" r"'(?P<val>(?:[^']|'')*?)'(?!')" % (
            re.escape(directive),
            self._optional_equals,
        )
        self._pr_options.append(_pr_compile(regex, cleanup_text))

    def _add_option_word(self, directive):
        regex = r"(?P<directive>%s)%s" r"(?P<val>\w+)" % (
            re.escape(directive),
            self._optional_equals,
        )
        self._pr_options.append(_pr_compile(regex))

    def _add_partition_option_word(self, directive):
        if directive == "PARTITION BY" or directive == "SUBPARTITION BY":
            regex = r"(?<!\S)(?P<directive>%s)%s" r"(?P<val>\w+.*)" % (
                re.escape(directive),
                self._optional_equals,
            )
        elif directive == "SUBPARTITIONS" or directive == "PARTITIONS":
            regex = r"(?<!\S)(?P<directive>%s)%s" r"(?P<val>\d+)" % (
                re.escape(directive),
                self._optional_equals,
            )
        else:
            regex = r"(?<!\S)(?P<directive>%s)(?!\S)" % (re.escape(directive),)
        self._pr_options.append(_pr_compile(regex))

    def _add_option_regex(self, directive, regex):
        regex = r"(?P<directive>%s)%s" r"(?P<val>%s)" % (
            re.escape(directive),
            self._optional_equals,
            regex,
        )
        self._pr_options.append(_pr_compile(regex))


_options_of_type_string = (
    "COMMENT",
    "DATA DIRECTORY",
    "INDEX DIRECTORY",
    "PASSWORD",
    "CONNECTION",
)


def _pr_compile(regex, cleanup=None):
    """Prepare a 2-tuple of compiled regex and callable."""

    return (_re_compile(regex), cleanup)


def _re_compile(regex):
    """Compile a string to regex, I and UNICODE."""

    return re.compile(regex, re.I | re.UNICODE)


def _strip_values(values):
    "Strip reflected values quotes"
    strip_values = []
    for a in values:
        if a[0:1] == '"' or a[0:1] == "'":
            # strip enclosing quotes and unquote interior
            a = a[1:-1].replace(a[0] * 2, a[0])
        strip_values.append(a)
    return strip_values


def cleanup_text(raw_text: str) -> str:
    if "\\" in raw_text:
        raw_text = re.sub(
            _control_char_regexp, lambda s: _control_char_map[s[0]], raw_text
        )
    return raw_text.replace("''", "'")


_control_char_map = {
    "\\\\": "\\",
    "\\0": "\0",
    "\\a": "\a",
    "\\b": "\b",
    "\\t": "\t",
    "\\n": "\n",
    "\\v": "\v",
    "\\f": "\f",
    "\\r": "\r",
    # '\\e':'\e',
}
_control_char_regexp = re.compile(
    "|".join(re.escape(k) for k in _control_char_map)
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_app.py ===
import inspect
import selectors
import socket
import threading
import time
from typing import Any, Callable, Optional, Union

from . import _logging
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import (
    WebSocketConnectionClosedException,
    WebSocketException,
    WebSocketTimeoutException,
)
from ._ssl_compat import SSLEOFError
from ._url import parse_url

"""
_app.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__all__ = ["WebSocketApp"]

RECONNECT = 0


def setReconnect(reconnectInterval: int) -> None:
    global RECONNECT
    RECONNECT = reconnectInterval


class DispatcherBase:
    """
    DispatcherBase
    """

    def __init__(self, app: Any, ping_timeout: Union[float, int, None]) -> None:
        self.app = app
        self.ping_timeout = ping_timeout

    def timeout(self, seconds: Union[float, int, None], callback: Callable) -> None:
        time.sleep(seconds)
        callback()

    def reconnect(self, seconds: int, reconnector: Callable) -> None:
        try:
            _logging.info(
                f"reconnect() - retrying in {seconds} seconds [{len(inspect.stack())} frames in stack]"
            )
            time.sleep(seconds)
            reconnector(reconnecting=True)
        except KeyboardInterrupt as e:
            _logging.info(f"User exited {e}")
            raise e


class Dispatcher(DispatcherBase):
    """
    Dispatcher
    """

    def read(
        self,
        sock: socket.socket,
        read_callback: Callable,
        check_callback: Callable,
    ) -> None:
        sel = selectors.DefaultSelector()
        sel.register(self.app.sock.sock, selectors.EVENT_READ)
        try:
            while self.app.keep_running:
                if sel.select(self.ping_timeout):
                    if not read_callback():
                        break
                check_callback()
        finally:
            sel.close()


class SSLDispatcher(DispatcherBase):
    """
    SSLDispatcher
    """

    def read(
        self,
        sock: socket.socket,
        read_callback: Callable,
        check_callback: Callable,
    ) -> None:
        sock = self.app.sock.sock
        sel = selectors.DefaultSelector()
        sel.register(sock, selectors.EVENT_READ)
        try:
            while self.app.keep_running:
                if self.select(sock, sel):
                    if not read_callback():
                        break
                check_callback()
        finally:
            sel.close()

    def select(self, sock, sel: selectors.DefaultSelector):
        sock = self.app.sock.sock
        if sock.pending():
            return [
                sock,
            ]

        r = sel.select(self.ping_timeout)

        if len(r) > 0:
            return r[0][0]


class WrappedDispatcher:
    """
    WrappedDispatcher
    """

    def __init__(self, app, ping_timeout: Union[float, int, None], dispatcher) -> None:
        self.app = app
        self.ping_timeout = ping_timeout
        self.dispatcher = dispatcher
        dispatcher.signal(2, dispatcher.abort)  # keyboard interrupt

    def read(
        self,
        sock: socket.socket,
        read_callback: Callable,
        check_callback: Callable,
    ) -> None:
        self.dispatcher.read(sock, read_callback)
        self.ping_timeout and self.timeout(self.ping_timeout, check_callback)

    def timeout(self, seconds: float, callback: Callable) -> None:
        self.dispatcher.timeout(seconds, callback)

    def reconnect(self, seconds: int, reconnector: Callable) -> None:
        self.timeout(seconds, reconnector)


class WebSocketApp:
    """
    Higher level of APIs are provided. The interface is like JavaScript WebSocket object.
    """

    def __init__(
        self,
        url: str,
        header: Union[list, dict, Callable, None] = None,
        on_open: Optional[Callable[[WebSocket], None]] = None,
        on_reconnect: Optional[Callable[[WebSocket], None]] = None,
        on_message: Optional[Callable[[WebSocket, Any], None]] = None,
        on_error: Optional[Callable[[WebSocket, Any], None]] = None,
        on_close: Optional[Callable[[WebSocket, Any, Any], None]] = None,
        on_ping: Optional[Callable] = None,
        on_pong: Optional[Callable] = None,
        on_cont_message: Optional[Callable] = None,
        keep_running: bool = True,
        get_mask_key: Optional[Callable] = None,
        cookie: Optional[str] = None,
        subprotocols: Optional[list] = None,
        on_data: Optional[Callable] = None,
        socket: Optional[socket.socket] = None,
    ) -> None:
        """
        WebSocketApp initialization

        Parameters
        ----------
        url: str
            Websocket url.
        header: list or dict or Callable
            Custom header for websocket handshake.
            If the parameter is a callable object, it is called just before the connection attempt.
            The returned dict or list is used as custom header value.
            This could be useful in order to properly setup timestamp dependent headers.
        on_open: function
            Callback object which is called at opening websocket.
            on_open has one argument.
            The 1st argument is this class object.
        on_reconnect: function
            Callback object which is called at reconnecting websocket.
            on_reconnect has one argument.
            The 1st argument is this class object.
        on_message: function
            Callback object which is called when received data.
            on_message has 2 arguments.
            The 1st argument is this class object.
            The 2nd argument is utf-8 data received from the server.
        on_error: function
            Callback object which is called when we get error.
            on_error has 2 arguments.
            The 1st argument is this class object.
            The 2nd argument is exception object.
        on_close: function
            Callback object which is called when connection is closed.
            on_close has 3 arguments.
            The 1st argument is this class object.
            The 2nd argument is close_status_code.
            The 3rd argument is close_msg.
        on_cont_message: function
            Callback object which is called when a continuation
            frame is received.
            on_cont_message has 3 arguments.
            The 1st argument is this class object.
            The 2nd argument is utf-8 string which we get from the server.
            The 3rd argument is continue flag. if 0, the data continue
            to next frame data
        on_data: function
            Callback object which is called when a message received.
            This is called before on_message or on_cont_message,
            and then on_message or on_cont_message is called.
            on_data has 4 argument.
            The 1st argument is this class object.
            The 2nd argument is utf-8 string which we get from the server.
            The 3rd argument is data type. ABNF.OPCODE_TEXT or ABNF.OPCODE_BINARY will be came.
            The 4th argument is continue flag. If 0, the data continue
        keep_running: bool
            This parameter is obsolete and ignored.
        get_mask_key: function
            A callable function to get new mask keys, see the
            WebSocket.set_mask_key's docstring for more information.
        cookie: str
            Cookie value.
        subprotocols: list
            List of available sub protocols. Default is None.
        socket: socket
            Pre-initialized stream socket.
        """
        self.url = url
        self.header = header if header is not None else []
        self.cookie = cookie

        self.on_open = on_open
        self.on_reconnect = on_reconnect
        self.on_message = on_message
        self.on_data = on_data
        self.on_error = on_error
        self.on_close = on_close
        self.on_ping = on_ping
        self.on_pong = on_pong
        self.on_cont_message = on_cont_message
        self.keep_running = False
        self.get_mask_key = get_mask_key
        self.sock: Optional[WebSocket] = None
        self.last_ping_tm = float(0)
        self.last_pong_tm = float(0)
        self.ping_thread: Optional[threading.Thread] = None
        self.stop_ping: Optional[threading.Event] = None
        self.ping_interval = float(0)
        self.ping_timeout: Union[float, int, None] = None
        self.ping_payload = ""
        self.subprotocols = subprotocols
        self.prepared_socket = socket
        self.has_errored = False
        self.has_done_teardown = False
        self.has_done_teardown_lock = threading.Lock()

    def send(self, data: Union[bytes, str], opcode: int = ABNF.OPCODE_TEXT) -> None:
        """
        send message

        Parameters
        ----------
        data: str
            Message to send. If you set opcode to OPCODE_TEXT,
            data must be utf-8 string or unicode.
        opcode: int
            Operation code of data. Default is OPCODE_TEXT.
        """

        if not self.sock or self.sock.send(data, opcode) == 0:
            raise WebSocketConnectionClosedException("Connection is already closed.")

    def send_text(self, text_data: str) -> None:
        """
        Sends UTF-8 encoded text.
        """
        if not self.sock or self.sock.send(text_data, ABNF.OPCODE_TEXT) == 0:
            raise WebSocketConnectionClosedException("Connection is already closed.")

    def send_bytes(self, data: Union[bytes, bytearray]) -> None:
        """
        Sends a sequence of bytes.
        """
        if not self.sock or self.sock.send(data, ABNF.OPCODE_BINARY) == 0:
            raise WebSocketConnectionClosedException("Connection is already closed.")

    def close(self, **kwargs) -> None:
        """
        Close websocket connection.
        """
        self.keep_running = False
        if self.sock:
            self.sock.close(**kwargs)
            self.sock = None

    def _start_ping_thread(self) -> None:
        self.last_ping_tm = self.last_pong_tm = float(0)
        self.stop_ping = threading.Event()
        self.ping_thread = threading.Thread(target=self._send_ping)
        self.ping_thread.daemon = True
        self.ping_thread.start()

    def _stop_ping_thread(self) -> None:
        if self.stop_ping:
            self.stop_ping.set()
        if self.ping_thread and self.ping_thread.is_alive():
            self.ping_thread.join(3)
        self.last_ping_tm = self.last_pong_tm = float(0)

    def _send_ping(self) -> None:
        if self.stop_ping.wait(self.ping_interval) or self.keep_running is False:
            return
        while not self.stop_ping.wait(self.ping_interval) and self.keep_running is True:
            if self.sock:
                self.last_ping_tm = time.time()
                try:
                    _logging.debug("Sending ping")
                    self.sock.ping(self.ping_payload)
                except Exception as e:
                    _logging.debug(f"Failed to send ping: {e}")

    def run_forever(
        self,
        sockopt: tuple = None,
        sslopt: dict = None,
        ping_interval: Union[float, int] = 0,
        ping_timeout: Union[float, int, None] = None,
        ping_payload: str = "",
        http_proxy_host: str = None,
        http_proxy_port: Union[int, str] = None,
        http_no_proxy: list = None,
        http_proxy_auth: tuple = None,
        http_proxy_timeout: Optional[float] = None,
        skip_utf8_validation: bool = False,
        host: str = None,
        origin: str = None,
        dispatcher=None,
        suppress_origin: bool = False,
        proxy_type: str = None,
        reconnect: int = None,
    ) -> bool:
        """
        Run event loop for WebSocket framework.

        This loop is an infinite loop and is alive while websocket is available.

        Parameters
        ----------
        sockopt: tuple
            Values for socket.setsockopt.
            sockopt must be tuple
            and each element is argument of sock.setsockopt.
        sslopt: dict
            Optional dict object for ssl socket option.
        ping_interval: int or float
            Automatically send "ping" command
            every specified period (in seconds).
            If set to 0, no ping is sent periodically.
        ping_timeout: int or float
            Timeout (in seconds) if the pong message is not received.
        ping_payload: str
            Payload message to send with each ping.
        http_proxy_host: str
            HTTP proxy host name.
        http_proxy_port: int or str
            HTTP proxy port. If not set, set to 80.
        http_no_proxy: list
            Whitelisted host names that don't use the proxy.
        http_proxy_timeout: int or float
            HTTP proxy timeout, default is 60 sec as per python-socks.
        http_proxy_auth: tuple
            HTTP proxy auth information. tuple of username and password. Default is None.
        skip_utf8_validation: bool
            skip utf8 validation.
        host: str
            update host header.
        origin: str
            update origin header.
        dispatcher: Dispatcher object
            customize reading data from socket.
        suppress_origin: bool
            suppress outputting origin header.
        proxy_type: str
            type of proxy from: http, socks4, socks4a, socks5, socks5h
        reconnect: int
            delay interval when reconnecting

        Returns
        -------
        teardown: bool
            False if the `WebSocketApp` is closed or caught KeyboardInterrupt,
            True if any other exception was raised during a loop.
        """

        if reconnect is None:
            reconnect = RECONNECT

        if ping_timeout is not None and ping_timeout <= 0:
            raise WebSocketException("Ensure ping_timeout > 0")
        if ping_interval is not None and ping_interval < 0:
            raise WebSocketException("Ensure ping_interval >= 0")
        if ping_timeout and ping_interval and ping_interval <= ping_timeout:
            raise WebSocketException("Ensure ping_interval > ping_timeout")
        if not sockopt:
            sockopt = ()
        if not sslopt:
            sslopt = {}
        if self.sock:
            raise WebSocketException("socket is already opened")

        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.ping_payload = ping_payload
        self.has_done_teardown = False
        self.keep_running = True

        def teardown(close_frame: ABNF = None):
            """
            Tears down the connection.

            Parameters
            ----------
            close_frame: ABNF frame
                If close_frame is set, the on_close handler is invoked
                with the statusCode and reason from the provided frame.
            """

            # teardown() is called in many code paths to ensure resources are cleaned up and on_close is fired.
            # To ensure the work is only done once, we use this bool and lock.
            with self.has_done_teardown_lock:
                if self.has_done_teardown:
                    return
                self.has_done_teardown = True

            self._stop_ping_thread()
            self.keep_running = False
            if self.sock:
                self.sock.close()
            close_status_code, close_reason = self._get_close_args(
                close_frame if close_frame else None
            )
            self.sock = None

            # Finally call the callback AFTER all teardown is complete
            self._callback(self.on_close, close_status_code, close_reason)

        def setSock(reconnecting: bool = False) -> None:
            if reconnecting and self.sock:
                self.sock.shutdown()

            self.sock = WebSocket(
                self.get_mask_key,
                sockopt=sockopt,
                sslopt=sslopt,
                fire_cont_frame=self.on_cont_message is not None,
                skip_utf8_validation=skip_utf8_validation,
                enable_multithread=True,
            )

            self.sock.settimeout(getdefaulttimeout())
            try:
                header = self.header() if callable(self.header) else self.header

                self.sock.connect(
                    self.url,
                    header=header,
                    cookie=self.cookie,
                    http_proxy_host=http_proxy_host,
                    http_proxy_port=http_proxy_port,
                    http_no_proxy=http_no_proxy,
                    http_proxy_auth=http_proxy_auth,
                    http_proxy_timeout=http_proxy_timeout,
                    subprotocols=self.subprotocols,
                    host=host,
                    origin=origin,
                    suppress_origin=suppress_origin,
                    proxy_type=proxy_type,
                    socket=self.prepared_socket,
                )

                _logging.info("Websocket connected")

                if self.ping_interval:
                    self._start_ping_thread()

                if reconnecting and self.on_reconnect:
                    self._callback(self.on_reconnect)
                else:
                    self._callback(self.on_open)

                dispatcher.read(self.sock.sock, read, check)
            except (
                WebSocketConnectionClosedException,
                ConnectionRefusedError,
                KeyboardInterrupt,
                SystemExit,
                Exception,
            ) as e:
                handleDisconnect(e, reconnecting)

        def read() -> bool:
            if not self.keep_running:
                return teardown()

            try:
                op_code, frame = self.sock.recv_data_frame(True)
            except (
                WebSocketConnectionClosedException,
                KeyboardInterrupt,
                SSLEOFError,
            ) as e:
                if custom_dispatcher:
                    return handleDisconnect(e, bool(reconnect))
                else:
                    raise e

            if op_code == ABNF.OPCODE_CLOSE:
                return teardown(frame)
            elif op_code == ABNF.OPCODE_PING:
                self._callback(self.on_ping, frame.data)
            elif op_code == ABNF.OPCODE_PONG:
                self.last_pong_tm = time.time()
                self._callback(self.on_pong, frame.data)
            elif op_code == ABNF.OPCODE_CONT and self.on_cont_message:
                self._callback(self.on_data, frame.data, frame.opcode, frame.fin)
                self._callback(self.on_cont_message, frame.data, frame.fin)
            else:
                data = frame.data
                if op_code == ABNF.OPCODE_TEXT and not skip_utf8_validation:
                    data = data.decode("utf-8")
                self._callback(self.on_data, data, frame.opcode, True)
                self._callback(self.on_message, data)

            return True

        def check() -> bool:
            if self.ping_timeout:
                has_timeout_expired = (
                    time.time() - self.last_ping_tm > self.ping_timeout
                )
                has_pong_not_arrived_after_last_ping = (
                    self.last_pong_tm - self.last_ping_tm < 0
                )
                has_pong_arrived_too_late = (
                    self.last_pong_tm - self.last_ping_tm > self.ping_timeout
                )

                if (
                    self.last_ping_tm
                    and has_timeout_expired
                    and (
                        has_pong_not_arrived_after_last_ping
                        or has_pong_arrived_too_late
                    )
                ):
                    raise WebSocketTimeoutException("ping/pong timed out")
            return True

        def handleDisconnect(
            e: Union[
                WebSocketConnectionClosedException,
                ConnectionRefusedError,
                KeyboardInterrupt,
                SystemExit,
                Exception,
            ],
            reconnecting: bool = False,
        ) -> bool:
            self.has_errored = True
            self._stop_ping_thread()
            if not reconnecting:
                self._callback(self.on_error, e)

            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                teardown()
                # Propagate further
                raise

            if reconnect:
                _logging.info(f"{e} - reconnect")
                if custom_dispatcher:
                    _logging.debug(
                        f"Calling custom dispatcher reconnect [{len(inspect.stack())} frames in stack]"
                    )
                    dispatcher.reconnect(reconnect, setSock)
            else:
                _logging.error(f"{e} - goodbye")
                teardown()

        custom_dispatcher = bool(dispatcher)
        dispatcher = self.create_dispatcher(
            ping_timeout, dispatcher, parse_url(self.url)[3]
        )

        try:
            setSock()
            if not custom_dispatcher and reconnect:
                while self.keep_running:
                    _logging.debug(
                        f"Calling dispatcher reconnect [{len(inspect.stack())} frames in stack]"
                    )
                    dispatcher.reconnect(reconnect, setSock)
        except (KeyboardInterrupt, Exception) as e:
            _logging.info(f"tearing down on exception {e}")
            teardown()
        finally:
            if not custom_dispatcher:
                # Ensure teardown was called before returning from run_forever
                teardown()

        return self.has_errored

    def create_dispatcher(
        self,
        ping_timeout: Union[float, int, None],
        dispatcher: Optional[DispatcherBase] = None,
        is_ssl: bool = False,
    ) -> Union[Dispatcher, SSLDispatcher, WrappedDispatcher]:
        if dispatcher:  # If custom dispatcher is set, use WrappedDispatcher
            return WrappedDispatcher(self, ping_timeout, dispatcher)
        timeout = ping_timeout or 10
        if is_ssl:
            return SSLDispatcher(self, timeout)
        return Dispatcher(self, timeout)

    def _get_close_args(self, close_frame: ABNF) -> list:
        """
        _get_close_args extracts the close code and reason from the close body
        if it exists (RFC6455 says WebSocket Connection Close Code is optional)
        """
        # Need to catch the case where close_frame is None
        # Otherwise the following if statement causes an error
        if not self.on_close or not close_frame:
            return [None, None]

        # Extract close frame status code
        if close_frame.data and len(close_frame.data) >= 2:
            close_status_code = 256 * int(close_frame.data[0]) + int(
                close_frame.data[1]
            )
            reason = close_frame.data[2:]
            if isinstance(reason, bytes):
                reason = reason.decode("utf-8")
            return [close_status_code, reason]
        else:
            # Most likely reached this because len(close_frame_data.data) < 2
            return [None, None]

    def _callback(self, callback, *args) -> None:
        if callback:
            try:
                callback(self, *args)

            except Exception as e:
                _logging.error(f"error from callback {callback}: {e}")
                if self.on_error:
                    self.on_error(self, e)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\galois_resolvents.py ===
r"""
Galois resolvents

Each of the functions in ``sympy.polys.numberfields.galoisgroups`` that
computes Galois groups for a particular degree $n$ uses resolvents. Given the
polynomial $T$ whose Galois group is to be computed, a resolvent is a
polynomial $R$ whose roots are defined as functions of the roots of $T$.

One way to compute the coefficients of $R$ is by approximating the roots of $T$
to sufficient precision. This module defines a :py:class:`~.Resolvent` class
that handles this job, determining the necessary precision, and computing $R$.

In some cases, the coefficients of $R$ are symmetric in the roots of $T$,
meaning they are equal to fixed functions of the coefficients of $T$. Therefore
another approach is to compute these functions once and for all, and record
them in a lookup table. This module defines code that can compute such tables.
The tables for polynomials $T$ of degrees 4 through 6, produced by this code,
are recorded in the resolvent_lookup.py module.

"""

from sympy.core.evalf import (
    evalf, fastlog, _evalf_with_bounded_error, quad_to_mpmath,
)
from sympy.core.symbol import symbols, Dummy
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import ZZ
from sympy.polys.orderings import lex
from sympy.polys.polyroots import preprocess_roots
from sympy.polys.polytools import Poly
from sympy.polys.rings import xring
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.lambdify import lambdify

from mpmath import MPContext
from mpmath.libmp.libmpf import prec_to_dps


class GaloisGroupException(Exception):
    ...


class ResolventException(GaloisGroupException):
    ...


class Resolvent:
    r"""
    If $G$ is a subgroup of the symmetric group $S_n$,
    $F$ a multivariate polynomial in $\mathbb{Z}[X_1, \ldots, X_n]$,
    $H$ the stabilizer of $F$ in $G$ (i.e. the permutations $\sigma$ such that
    $F(X_{\sigma(1)}, \ldots, X_{\sigma(n)}) = F(X_1, \ldots, X_n)$), and $s$
    a set of left coset representatives of $H$ in $G$, then the resolvent
    polynomial $R(Y)$ is the product over $\sigma \in s$ of
    $Y - F(X_{\sigma(1)}, \ldots, X_{\sigma(n)})$.

    For example, consider the resolvent for the form
    $$F = X_0 X_2 + X_1 X_3$$
    and the group $G = S_4$. In this case, the stabilizer $H$ is the dihedral
    group $D4 = < (0123), (02) >$, and a set of representatives of $G/H$ is
    $\{I, (01), (03)\}$. The resolvent can be constructed as follows:

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.core.symbol import symbols
    >>> from sympy.polys.numberfields.galoisgroups import Resolvent
    >>> X = symbols('X0 X1 X2 X3')
    >>> F = X[0]*X[2] + X[1]*X[3]
    >>> s = [Permutation([0, 1, 2, 3]), Permutation([1, 0, 2, 3]),
    ... Permutation([3, 1, 2, 0])]
    >>> R = Resolvent(F, X, s)

    This resolvent has three roots, which are the conjugates of ``F`` under the
    three permutations in ``s``:

    >>> R.root_lambdas[0](*X)
    X0*X2 + X1*X3
    >>> R.root_lambdas[1](*X)
    X0*X3 + X1*X2
    >>> R.root_lambdas[2](*X)
    X0*X1 + X2*X3

    Resolvents are useful for computing Galois groups. Given a polynomial $T$
    of degree $n$, we will use a resolvent $R$ where $Gal(T) \leq G \leq S_n$.
    We will then want to substitute the roots of $T$ for the variables $X_i$
    in $R$, and study things like the discriminant of $R$, and the way $R$
    factors over $\mathbb{Q}$.

    From the symmetry in $R$'s construction, and since $Gal(T) \leq G$, we know
    from Galois theory that the coefficients of $R$ must lie in $\mathbb{Z}$.
    This allows us to compute the coefficients of $R$ by approximating the
    roots of $T$ to sufficient precision, plugging these values in for the
    variables $X_i$ in the coefficient expressions of $R$, and then simply
    rounding to the nearest integer.

    In order to determine a sufficient precision for the roots of $T$, this
    ``Resolvent`` class imposes certain requirements on the form ``F``. It
    could be possible to design a different ``Resolvent`` class, that made
    different precision estimates, and different assumptions about ``F``.

    ``F`` must be homogeneous, and all terms must have unit coefficient.
    Furthermore, if $r$ is the number of terms in ``F``, and $t$ the total
    degree, and if $m$ is the number of conjugates of ``F``, i.e. the number
    of permutations in ``s``, then we require that $m < r 2^t$. Again, it is
    not impossible to work with forms ``F`` that violate these assumptions, but
    this ``Resolvent`` class requires them.

    Since determining the integer coefficients of the resolvent for a given
    polynomial $T$ is one of the main problems this class solves, we take some
    time to explain the precision bounds it uses.

    The general problem is:
    Given a multivariate polynomial $P \in \mathbb{Z}[X_1, \ldots, X_n]$, and a
    bound $M \in \mathbb{R}_+$, compute an $\varepsilon > 0$ such that for any
    complex numbers $a_1, \ldots, a_n$ with $|a_i| < M$, if the $a_i$ are
    approximated to within an accuracy of $\varepsilon$ by $b_i$, that is,
    $|a_i - b_i| < \varepsilon$ for $i = 1, \ldots, n$, then
    $|P(a_1, \ldots, a_n) - P(b_1, \ldots, b_n)| < 1/2$. In other words, if it
    is known that $P(a_1, \ldots, a_n) = c$ for some $c \in \mathbb{Z}$, then
    $P(b_1, \ldots, b_n)$ can be rounded to the nearest integer in order to
    determine $c$.

    To derive our error bound, consider the monomial $xyz$. Defining
    $d_i = b_i - a_i$, our error is
    $|(a_1 + d_1)(a_2 + d_2)(a_3 + d_3) - a_1 a_2 a_3|$, which is bounded
    above by $|(M + \varepsilon)^3 - M^3|$. Passing to a general monomial of
    total degree $t$, this expression is bounded by
    $M^{t-1}\varepsilon(t + 2^t\varepsilon/M)$ provided $\varepsilon < M$,
    and by $(t+1)M^{t-1}\varepsilon$ provided $\varepsilon < M/2^t$.
    But since our goal is to make the error less than $1/2$, we will choose
    $\varepsilon < 1/(2(t+1)M^{t-1})$, which implies the condition that
    $\varepsilon < M/2^t$, as long as $M \geq 2$.

    Passing from the general monomial to the general polynomial is easy, by
    scaling and summing error bounds.

    In our specific case, we are given a homogeneous polynomial $F$ of
    $r$ terms and total degree $t$, all of whose coefficients are $\pm 1$. We
    are given the $m$ permutations that make the conjugates of $F$, and
    we want to bound the error in the coefficients of the monic polynomial
    $R(Y)$ having $F$ and its conjugates as roots (i.e. the resolvent).

    For $j$ from $1$ to $m$, the coefficient of $Y^{m-j}$ in $R(Y)$ is the
    $j$th elementary symmetric polynomial in the conjugates of $F$. This sums
    the products of these conjugates, taken $j$ at a time, in all possible
    combinations. There are $\binom{m}{j}$ such combinations, and each product
    of $j$ conjugates of $F$ expands to a sum of $r^j$ terms, each of unit
    coefficient, and total degree $jt$. An error bound for the $j$th coeff of
    $R$ is therefore
    $$\binom{m}{j} r^j (jt + 1) M^{jt - 1} \varepsilon$$
    When our goal is to evaluate all the coefficients of $R$, we will want to
    use the maximum of these error bounds. It is clear that this bound is
    strictly increasing for $j$ up to the ceiling of $m/2$. After that point,
    the first factor $\binom{m}{j}$ begins to decrease, while the others
    continue to increase. However, the binomial coefficient never falls by more
    than a factor of $1/m$ at a time, so our assumptions that $M \geq 2$ and
    $m < r 2^t$ are enough to tell us that the constant coefficient of $R$,
    i.e. that where $j = m$, has the largest error bound. Therefore we can use
    $$r^m (mt + 1) M^{mt - 1} \varepsilon$$
    as our error bound for all the coefficients.

    Note that this bound is also (more than) adequate to determine whether any
    of the roots of $R$ is an integer. Each of these roots is a single
    conjugate of $F$, which contains less error than the trace, i.e. the
    coefficient of $Y^{m - 1}$. By rounding the roots of $R$ to the nearest
    integers, we therefore get all the candidates for integer roots of $R$. By
    plugging these candidates into $R$, we can check whether any of them
    actually is a root.

    Note: We take the definition of resolvent from Cohen, but the error bound
    is ours.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.
       (Def 6.3.2)

    """

    def __init__(self, F, X, s):
        r"""
        Parameters
        ==========

        F : :py:class:`~.Expr`
            polynomial in the symbols in *X*
        X : list of :py:class:`~.Symbol`
        s : list of :py:class:`~.Permutation`
            representing the cosets of the stabilizer of *F* in
            some subgroup $G$ of $S_n$, where $n$ is the length of *X*.
        """
        self.F = F
        self.X = X
        self.s = s

        # Number of conjugates:
        self.m = len(s)
        # Total degree of F (computed below):
        self.t = None
        # Number of terms in F (computed below):
        self.r = 0

        for monom, coeff in Poly(F).terms():
            if abs(coeff) != 1:
                raise ResolventException('Resolvent class expects forms with unit coeffs')
            t = sum(monom)
            if t != self.t and self.t is not None:
                raise ResolventException('Resolvent class expects homogeneous forms')
            self.t = t
            self.r += 1

        m, t, r = self.m, self.t, self.r
        if not m < r * 2**t:
            raise ResolventException('Resolvent class expects m < r*2^t')
        M = symbols('M')
        # Precision sufficient for computing the coeffs of the resolvent:
        self.coeff_prec_func = Poly(r**m*(m*t + 1)*M**(m*t - 1))
        # Precision sufficient for checking whether any of the roots of the
        # resolvent are integers:
        self.root_prec_func = Poly(r*(t + 1)*M**(t - 1))

        # The conjugates of F are the roots of the resolvent.
        # For evaluating these to required numerical precisions, we need
        # lambdified versions.
        # Note: for a given permutation sigma, the conjugate (sigma F) is
        # equivalent to lambda [sigma^(-1) X]: F.
        self.root_lambdas = [
            lambdify((~s[j])(X), F)
            for j in range(self.m)
        ]

        # For evaluating the coeffs, we'll also need lambdified versions of
        # the elementary symmetric functions for degree m.
        Y = symbols('Y')
        R = symbols(' '.join(f'R{i}' for i in range(m)))
        f = 1
        for r in R:
            f *= (Y - r)
        C = Poly(f, Y).coeffs()
        self.esf_lambdas = [lambdify(R, c) for c in C]

    def get_prec(self, M, target='coeffs'):
        r"""
        For a given upper bound *M* on the magnitude of the complex numbers to
        be plugged in for this resolvent's symbols, compute a sufficient
        precision for evaluating those complex numbers, such that the
        coefficients, or the integer roots, of the resolvent can be determined.

        Parameters
        ==========

        M : real number
            Upper bound on magnitude of the complex numbers to be plugged in.

        target : str, 'coeffs' or 'roots', default='coeffs'
            Name the task for which a sufficient precision is desired.
            This is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        int $m$
            such that $2^{-m}$ is a sufficient upper bound on the
            error in approximating the complex numbers to be plugged in.

        """
        # As explained in the docstring for this class, our precision estimates
        # require that M be at least 2.
        M = max(M, 2)
        f = self.coeff_prec_func if target == 'coeffs' else self.root_prec_func
        r, _, _, _ = evalf(2*f(M), 1, {})
        return fastlog(r) + 1

    def approximate_roots_of_poly(self, T, target='coeffs'):
        """
        Approximate the roots of a given polynomial *T* to sufficient precision
        in order to evaluate this resolvent's coefficients, or determine
        whether the resolvent has an integer root.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        target : str, 'coeffs' or 'roots', default='coeffs'
            Set the approximation precision to be sufficient for the desired
            task, which is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        list of elements of :ref:`CC`

        """
        ctx = MPContext()
        # Because sympy.polys.polyroots._integer_basis() is called when a CRootOf
        # is formed, we proactively extract the integer basis now. This means that
        # when we call T.all_roots(), every root will be a CRootOf, not a Mul
        # of Integer*CRootOf.
        coeff, T = preprocess_roots(T)
        coeff = ctx.mpf(str(coeff))

        scaled_roots = T.all_roots(radicals=False)

        # Since we're going to be approximating the roots of T anyway, we can
        # get a good upper bound on the magnitude of the roots by starting with
        # a very low precision approx.
        approx0 = [coeff * quad_to_mpmath(_evalf_with_bounded_error(r, m=0)) for r in scaled_roots]
        # Here we add 1 to account for the possible error in our initial approximation.
        M = max(abs(b) for b in approx0) + 1
        m = self.get_prec(M, target=target)
        n = fastlog(M._mpf_) + 1
        p = m + n + 1
        ctx.prec = p
        d = prec_to_dps(p)

        approx1 = [r.eval_approx(d, return_mpmath=True) for r in scaled_roots]
        approx1 = [coeff*ctx.mpc(r) for r in approx1]

        return approx1

    @staticmethod
    def round_mpf(a):
        if isinstance(a, int):
            return a
        # If we use python's built-in `round()`, we lose precision.
        # If we use `ZZ` directly, we may add or subtract 1.
        #
        # XXX: We have to convert to int before converting to ZZ because
        # flint.fmpz cannot convert a mpmath mpf.
        return ZZ(int(a.context.nint(a)))

    def round_roots_to_integers_for_poly(self, T):
        """
        For a given polynomial *T*, round the roots of this resolvent to the
        nearest integers.

        Explanation
        ===========

        None of the integers returned by this method is guaranteed to be a
        root of the resolvent; however, if the resolvent has any integer roots
        (for the given polynomial *T*), then they must be among these.

        If the coefficients of the resolvent are also desired, then this method
        should not be used. Instead, use the ``eval_for_poly`` method. This
        method may be significantly faster than ``eval_for_poly``.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        Returns
        =======

        dict
            Keys are the indices of those permutations in ``self.s`` such that
            the corresponding root did round to a rational integer.

            Values are :ref:`ZZ`.


        """
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='roots')
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        return {
            i: self.round_mpf(r.real)
            for i, r in enumerate(approx_roots_of_self)
            if self.round_mpf(r.imag) == 0
        }

    def eval_for_poly(self, T, find_integer_root=False):
        r"""
        Compute the integer values of the coefficients of this resolvent, when
        plugging in the roots of a given polynomial.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        find_integer_root : ``bool``, default ``False``
            If ``True``, then also determine whether the resolvent has an
            integer root, and return the first one found, along with its
            index, i.e. the index of the permutation ``self.s[i]`` it
            corresponds to.

        Returns
        =======

        Tuple ``(R, a, i)``

            ``R`` is this resolvent as a dense univariate polynomial over
            :ref:`ZZ`, i.e. a list of :ref:`ZZ`.

            If *find_integer_root* was ``True``, then ``a`` and ``i`` are the
            first integer root found, and its index, if one exists.
            Otherwise ``a`` and ``i`` are both ``None``.

        """
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='coeffs')
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        approx_coeffs_of_self = [c(*approx_roots_of_self) for c in self.esf_lambdas]

        R = []
        for c in approx_coeffs_of_self:
            if self.round_mpf(c.imag) != 0:
                # If precision was enough, this should never happen.
                raise ResolventException(f"Got non-integer coeff for resolvent: {c}")
            R.append(self.round_mpf(c.real))

        a0, i0 = None, None

        if find_integer_root:
            for i, r in enumerate(approx_roots_of_self):
                if self.round_mpf(r.imag) != 0:
                    continue
                if not dup_eval(R, (a := self.round_mpf(r.real)), ZZ):
                    a0, i0 = a, i
                    break

        return R, a0, i0


def wrap(text, width=80):
    """Line wrap a polynomial expression. """
    out = ''
    col = 0
    for c in text:
        if c == ' ' and col > width:
            c, col = '\n', 0
        else:
            col += 1
        out += c
    return out


def s_vars(n):
    """Form the symbols s1, s2, ..., sn to stand for elem. symm. polys. """
    return symbols([f's{i + 1}' for i in range(n)])


def sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=False):
    """
    Compute the coefficients of a resolvent as functions of the coefficients of
    the associated polynomial.

    F must be a sparse polynomial.
    """
    import time, sys
    # Roots of resolvent as multivariate forms over vars X:
    root_forms = [
        F.compose(list(zip(X, sigma(X))))
        for sigma in s
    ]

    # Coeffs of resolvent (besides lead coeff of 1) as symmetric forms over vars X:
    Y = [Dummy(f'Y{i}') for i in range(len(s))]
    coeff_forms = []
    for i in range(1, len(s) + 1):
        if verbose:
            print('----')
            print(f'Computing symmetric poly of degree {i}...')
            sys.stdout.flush()
        t0 = time.time()
        G = symmetric_poly(i, *Y)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            print('lambdifying...')
            sys.stdout.flush()
        t0 = time.time()
        C = lambdify(Y, (-1)**i*G)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()
        coeff_forms.append(C)

    coeffs = []
    for i, f in enumerate(coeff_forms):
        if verbose:
            print('----')
            print(f'Plugging root forms into elem symm poly {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        g = f(*root_forms)
        t1 = time.time()
        coeffs.append(g)
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    # Now symmetrize these coeffs. This means recasting them as polynomials in
    # the elementary symmetric polys over X.
    symmetrized = []
    symmetrization_times = []
    ss = s_vars(len(X))
    for i, A in list(enumerate(coeffs)):
        if verbose:
            print('-----')
            print(f'Coeff {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        B, rem, _ = A.symmetrize()
        t1 = time.time()
        if rem != 0:
            msg = f"Got nonzero remainder {rem} for resolvent (F, X, s) = ({F}, {X}, {s})"
            raise ResolventException(msg)
        B_str = str(B.as_expr(*ss))
        symmetrized.append(B_str)
        symmetrization_times.append(t1 - t0)
        if verbose:
            print(wrap(B_str))
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    return symmetrized, symmetrization_times


def define_resolvents():
    """Define all the resolvents for polys T of degree 4 through 6. """
    from sympy.combinatorics.galois import PGL2F5
    from sympy.combinatorics.permutations import Permutation

    R4, X4 = xring("X0,X1,X2,X3", ZZ, lex)
    X = X4

    # The one resolvent used in `_galois_group_degree_4_lookup()`:
    F40 = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    s40 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 2),
        Permutation(3)(0, 3),
        Permutation(3)(1, 2),
        Permutation(3)(2, 3),
    ]

    # First resolvent used in `_galois_group_degree_4_root_approx()`:
    F41 = X[0]*X[2] + X[1]*X[3]
    s41 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 3)
    ]

    R5, X5 = xring("X0,X1,X2,X3,X4", ZZ, lex)
    X = X5

    # First resolvent used in `_galois_group_degree_5_hybrid()`,
    # and only one used in `_galois_group_degree_5_lookup_ext_factor()`:
    F51 = (  X[0]**2*(X[1]*X[4] + X[2]*X[3])
           + X[1]**2*(X[2]*X[0] + X[3]*X[4])
           + X[2]**2*(X[3]*X[1] + X[4]*X[0])
           + X[3]**2*(X[4]*X[2] + X[0]*X[1])
           + X[4]**2*(X[0]*X[3] + X[1]*X[2]))
    s51 = [
        Permutation(4),
        Permutation(4)(0, 1),
        Permutation(4)(0, 2),
        Permutation(4)(0, 3),
        Permutation(4)(0, 4),
        Permutation(4)(1, 4)
    ]

    R6, X6 = xring("X0,X1,X2,X3,X4,X5", ZZ, lex)
    X = X6

    # First resolvent used in `_galois_group_degree_6_lookup()`:
    H = PGL2F5()
    term0 = X[0]**2*X[5]**2*(X[1]*X[4] + X[2]*X[3])
    terms = {term0.compose(list(zip(X, s(X)))) for s in H.elements}
    F61 = sum(terms)
    s61 = [Permutation(5)] + [Permutation(5)(0, n) for n in range(1, 6)]

    # Second resolvent used in `_galois_group_degree_6_lookup()`:
    F62 = X[0]*X[1]*X[2] + X[3]*X[4]*X[5]
    s62 = [Permutation(5)] + [
        Permutation(5)(i, j + 3) for i in range(3) for j in range(3)
    ]

    return {
        (4, 0): (F40, X4, s40),
        (4, 1): (F41, X4, s41),
        (5, 1): (F51, X5, s51),
        (6, 1): (F61, X6, s61),
        (6, 2): (F62, X6, s62),
    }


def generate_lambda_lookup(verbose=False, trial_run=False):
    """
    Generate the whole lookup table of coeff lambdas, for all resolvents.
    """
    jobs = define_resolvents()
    lambda_lists = {}
    total_time = 0
    time_for_61 = 0
    time_for_61_last = 0
    for k, (F, X, s) in jobs.items():
        symmetrized, times = sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=verbose)

        total_time += sum(times)
        if k == (6, 1):
            time_for_61 = sum(times)
            time_for_61_last = times[-1]

        sv = s_vars(len(X))
        head = f'lambda {", ".join(str(v) for v in sv)}:'
        lambda_lists[k] = ',\n        '.join([
            f'{head} ({wrap(f)})'
            for f in symmetrized
        ])

        if trial_run:
            break

    table = (
         "# This table was generated by a call to\n"
         "# `sympy.polys.numberfields.galois_resolvents.generate_lambda_lookup()`.\n"
        f"# The entire job took {total_time:.2f}s.\n"
        f"# Of this, Case (6, 1) took {time_for_61:.2f}s.\n"
        f"# The final polynomial of Case (6, 1) alone took {time_for_61_last:.2f}s.\n"
         "resolvent_coeff_lambdas = {\n")

    for k, L in lambda_lists.items():
        table += f"    {k}: [\n"
        table +=  "        " + L + '\n'
        table +=  "    ],\n"
    table += "}\n"
    return table


def get_resolvent_by_lookup(T, number):
    """
    Use the lookup table, to return a resolvent (as dup) for a given
    polynomial *T*.

    Parameters
    ==========

    T : Poly
        The polynomial whose resolvent is needed

    number : int
        For some degrees, there are multiple resolvents.
        Use this to indicate which one you want.

    Returns
    =======

    dup

    """
    from sympy.polys.numberfields.resolvent_lookup import resolvent_coeff_lambdas
    degree = T.degree()
    L = resolvent_coeff_lambdas[(degree, number)]
    T_coeffs = T.rep.to_list()[1:]
    return [ZZ(1)] + [c(*T_coeffs) for c in L]


# Use
#   (.venv) $ python -m sympy.polys.numberfields.galois_resolvents
# to reproduce the table found in resolvent_lookup.py
if __name__ == "__main__":
    import sys
    verbose = '-v' in sys.argv[1:]
    trial_run = '-t' in sys.argv[1:]
    table = generate_lambda_lookup(verbose=verbose, trial_run=trial_run)
    print(table)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\pauli.py ===
"""Pauli operators and states"""

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum import Operator, Ket, Bra
from sympy.physics.quantum import ComplexSpace
from sympy.matrices import Matrix
from sympy.functions.special.tensor_functions import KroneckerDelta

__all__ = [
    'SigmaX', 'SigmaY', 'SigmaZ', 'SigmaMinus', 'SigmaPlus', 'SigmaZKet',
    'SigmaZBra', 'qsimplify_pauli'
]


class SigmaOpBase(Operator):
    """Pauli sigma operator, base class"""

    @property
    def name(self):
        return self.args[0]

    @property
    def use_name(self):
        return bool(self.args[0]) is not False

    @classmethod
    def default_args(self):
        return (False,)

    def __new__(cls, *args, **hints):
        return Operator.__new__(cls, *args, **hints)

    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero


class SigmaX(SigmaOpBase):
    """Pauli sigma x operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaX
    >>> sx = SigmaX()
    >>> sx
    SigmaX()
    >>> represent(sx)
    Matrix([
    [0, 1],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args, **hints)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return - 2 * I * SigmaY(self.name)

    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_x^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_x}'

    def _print_contents(self, printer, *args):
        return 'SigmaX()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaX(self.name).__pow__(int(e) % 2)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaY(SigmaOpBase):
    """Pauli sigma y operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaY
    >>> sy = SigmaY()
    >>> sy
    SigmaY()
    >>> represent(sy)
    Matrix([
    [0, -I],
    [I,  0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaX(self.name)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return - 2 * I * SigmaZ(self.name)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_y^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_y}'

    def _print_contents(self, printer, *args):
        return 'SigmaY()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaY(self.name).__pow__(int(e) % 2)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, -I], [I, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZ(SigmaOpBase):
    """Pauli sigma z operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaZ
    >>> sz = SigmaZ()
    >>> sz ** 3
    SigmaZ()
    >>> represent(sz)
    Matrix([
    [1,  0],
    [0, -1]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaY(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return - 2 * I * SigmaX(self.name)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_z^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_z}'

    def _print_contents(self, printer, *args):
        return 'SigmaZ()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaZ(self.name).__pow__(int(e) % 2)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[1, 0], [0, -1]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaMinus(SigmaOpBase):
    """Pauli sigma minus operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent, Dagger
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> sm = SigmaMinus()
    >>> sm
    SigmaMinus()
    >>> Dagger(sm)
    SigmaPlus()
    >>> represent(sm)
    Matrix([
    [0, 0],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return -SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        return 2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        return SigmaZ(self.name)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.One

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return I * S.NegativeOne

    def _eval_anticommutator_SigmaPlus(self, other, **hints):
        return S.One

    def _eval_adjoint(self):
        return SigmaPlus(self.name)

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return S.Zero

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_-^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_-}'

    def _print_contents(self, printer, *args):
        return 'SigmaMinus()'

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 0], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaPlus(SigmaOpBase):
    """Pauli sigma plus operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent, Dagger
    >>> from sympy.physics.quantum.pauli import SigmaPlus
    >>> sp = SigmaPlus()
    >>> sp
    SigmaPlus()
    >>> Dagger(sp)
    SigmaMinus()
    >>> represent(sp)
    Matrix([
    [0, 1],
    [0, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return -2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        return SigmaZ(self.name)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.One

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return I

    def _eval_anticommutator_SigmaMinus(self, other, **hints):
        return S.One

    def _eval_adjoint(self):
        return SigmaMinus(self.name)

    def _eval_mul(self, other):
        return self * other

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return S.Zero

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_+^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_+}'

    def _print_contents(self, printer, *args):
        return 'SigmaPlus()'

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [0, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZKet(Ket):
    """Ket for a two-level system quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return SigmaZBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2)

    def _eval_innerproduct_SigmaZBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_SigmaZ(self, op, **options):
        if self.n == 0:
            return self
        else:
            return S.NegativeOne * self

    def _apply_from_right_to_SigmaX(self, op, **options):
        return SigmaZKet(1) if self.n == 0 else SigmaZKet(0)

    def _apply_from_right_to_SigmaY(self, op, **options):
        return I * SigmaZKet(1) if self.n == 0 else (-I) * SigmaZKet(0)

    def _apply_from_right_to_SigmaMinus(self, op, **options):
        if self.n == 0:
            return SigmaZKet(1)
        else:
            return S.Zero

    def _apply_from_right_to_SigmaPlus(self, op, **options):
        if self.n == 0:
            return S.Zero
        else:
            return SigmaZKet(0)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[1], [0]]) if self.n == 0 else Matrix([[0], [1]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZBra(Bra):
    """Bra for a two-level quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return SigmaZKet


def _qsimplify_pauli_product(a, b):
    """
    Internal helper function for simplifying products of Pauli operators.
    """
    if not (isinstance(a, SigmaOpBase) and isinstance(b, SigmaOpBase)):
        return Mul(a, b)

    if a.name != b.name:
        # Pauli matrices with different labels commute; sort by name
        if a.name < b.name:
            return Mul(a, b)
        else:
            return Mul(b, a)

    elif isinstance(a, SigmaX):

        if isinstance(b, SigmaX):
            return S.One

        if isinstance(b, SigmaY):
            return I * SigmaZ(a.name)

        if isinstance(b, SigmaZ):
            return - I * SigmaY(a.name)

        if isinstance(b, SigmaMinus):
            return (S.Half + SigmaZ(a.name)/2)

        if isinstance(b, SigmaPlus):
            return (S.Half - SigmaZ(a.name)/2)

    elif isinstance(a, SigmaY):

        if isinstance(b, SigmaX):
            return - I * SigmaZ(a.name)

        if isinstance(b, SigmaY):
            return S.One

        if isinstance(b, SigmaZ):
            return I * SigmaX(a.name)

        if isinstance(b, SigmaMinus):
            return -I * (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaPlus):
            return I * (S.One - SigmaZ(a.name))/2

    elif isinstance(a, SigmaZ):

        if isinstance(b, SigmaX):
            return I * SigmaY(a.name)

        if isinstance(b, SigmaY):
            return - I * SigmaX(a.name)

        if isinstance(b, SigmaZ):
            return S.One

        if isinstance(b, SigmaMinus):
            return - SigmaMinus(a.name)

        if isinstance(b, SigmaPlus):
            return SigmaPlus(a.name)

    elif isinstance(a, SigmaMinus):

        if isinstance(b, SigmaX):
            return (S.One - SigmaZ(a.name))/2

        if isinstance(b, SigmaY):
            return - I * (S.One - SigmaZ(a.name))/2

        if isinstance(b, SigmaZ):
            # (SigmaX(a.name) - I * SigmaY(a.name))/2
            return SigmaMinus(b.name)

        if isinstance(b, SigmaMinus):
            return S.Zero

        if isinstance(b, SigmaPlus):
            return S.Half - SigmaZ(a.name)/2

    elif isinstance(a, SigmaPlus):

        if isinstance(b, SigmaX):
            return (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaY):
            return I * (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaZ):
            #-(SigmaX(a.name) + I * SigmaY(a.name))/2
            return -SigmaPlus(a.name)

        if isinstance(b, SigmaMinus):
            return (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaPlus):
            return S.Zero

    else:
        return a * b


def qsimplify_pauli(e):
    """
    Simplify an expression that includes products of pauli operators.

    Parameters
    ==========

    e : expression
        An expression that contains products of Pauli operators that is
        to be simplified.

    Examples
    ========

    >>> from sympy.physics.quantum.pauli import SigmaX, SigmaY
    >>> from sympy.physics.quantum.pauli import qsimplify_pauli
    >>> sx, sy = SigmaX(), SigmaY()
    >>> sx * sy
    SigmaX()*SigmaY()
    >>> qsimplify_pauli(sx * sy)
    I*SigmaZ()
    """
    if isinstance(e, Operator):
        return e

    if isinstance(e, (Add, Pow, exp)):
        t = type(e)
        return t(*(qsimplify_pauli(arg) for arg in e.args))

    if isinstance(e, Mul):

        c, nc = e.args_cnc()

        nc_s = []
        while nc:
            curr = nc.pop(0)

            while (len(nc) and
                   isinstance(curr, SigmaOpBase) and
                   isinstance(nc[0], SigmaOpBase) and
                   curr.name == nc[0].name):

                x = nc.pop(0)
                y = _qsimplify_pauli_product(curr, x)
                c1, nc1 = y.args_cnc()
                curr = Mul(*nc1)
                c = c + c1

            nc_s.append(curr)

        return Mul(*c) * Mul(*nc_s)

    return e

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\coloredlogs\tests.py ===
# Automated tests for the `coloredlogs' package.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: June 11, 2021
# URL: https://coloredlogs.readthedocs.io

"""Automated tests for the `coloredlogs` package."""

# Standard library modules.
import contextlib
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import tempfile

# External dependencies.
from humanfriendly.compat import StringIO
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_CSI, ansi_style, ansi_wrap
from humanfriendly.testing import PatchedAttribute, PatchedItem, TestCase, retry
from humanfriendly.text import format, random_string

# The module we're testing.
import coloredlogs
import coloredlogs.cli
from coloredlogs import (
    CHROOT_FILES,
    ColoredFormatter,
    NameNormalizer,
    decrease_verbosity,
    find_defined_levels,
    find_handler,
    find_hostname,
    find_program_name,
    find_username,
    get_level,
    increase_verbosity,
    install,
    is_verbose,
    level_to_number,
    match_stream_handler,
    parse_encoded_styles,
    set_level,
    walk_propagation_tree,
)
from coloredlogs.demo import demonstrate_colored_logging
from coloredlogs.syslog import SystemLogging, is_syslog_supported, match_syslog_handler
from coloredlogs.converter import (
    ColoredCronMailer,
    EIGHT_COLOR_PALETTE,
    capture,
    convert,
)

# External test dependencies.
from capturer import CaptureOutput
from verboselogs import VerboseLogger

# Compiled regular expression that matches a single line of output produced by
# the default log format (does not include matching of ANSI escape sequences).
PLAIN_TEXT_PATTERN = re.compile(r'''
    (?P<date> \d{4}-\d{2}-\d{2} )
    \s (?P<time> \d{2}:\d{2}:\d{2} )
    \s (?P<hostname> \S+ )
    \s (?P<logger_name> \w+ )
    \[ (?P<process_id> \d+ ) \]
    \s (?P<severity> [A-Z]+ )
    \s (?P<message> .* )
''', re.VERBOSE)

# Compiled regular expression that matches a single line of output produced by
# the default log format with milliseconds=True.
PATTERN_INCLUDING_MILLISECONDS = re.compile(r'''
    (?P<date> \d{4}-\d{2}-\d{2} )
    \s (?P<time> \d{2}:\d{2}:\d{2},\d{3} )
    \s (?P<hostname> \S+ )
    \s (?P<logger_name> \w+ )
    \[ (?P<process_id> \d+ ) \]
    \s (?P<severity> [A-Z]+ )
    \s (?P<message> .* )
''', re.VERBOSE)


def setUpModule():
    """Speed up the tests by disabling the demo's artificial delay."""
    os.environ['COLOREDLOGS_DEMO_DELAY'] = '0'
    coloredlogs.demo.DEMO_DELAY = 0


class ColoredLogsTestCase(TestCase):

    """Container for the `coloredlogs` tests."""

    def find_system_log(self):
        """Find the system log file or skip the current test."""
        filename = ('/var/log/system.log' if sys.platform == 'darwin' else (
            '/var/log/syslog' if 'linux' in sys.platform else None
        ))
        if not filename:
            self.skipTest("Location of system log file unknown!")
        elif not os.path.isfile(filename):
            self.skipTest("System log file not found! (%s)" % filename)
        elif not os.access(filename, os.R_OK):
            self.skipTest("Insufficient permissions to read system log file! (%s)" % filename)
        else:
            return filename

    def test_level_to_number(self):
        """Make sure :func:`level_to_number()` works as intended."""
        # Make sure the default levels are translated as expected.
        assert level_to_number('debug') == logging.DEBUG
        assert level_to_number('info') == logging.INFO
        assert level_to_number('warning') == logging.WARNING
        assert level_to_number('error') == logging.ERROR
        assert level_to_number('fatal') == logging.FATAL
        # Make sure bogus level names don't blow up.
        assert level_to_number('bogus-level') == logging.INFO

    def test_find_hostname(self):
        """Make sure :func:`~find_hostname()` works correctly."""
        assert find_hostname()
        # Create a temporary file as a placeholder for e.g. /etc/debian_chroot.
        fd, temporary_file = tempfile.mkstemp()
        try:
            with open(temporary_file, 'w') as handle:
                handle.write('first line\n')
                handle.write('second line\n')
            CHROOT_FILES.insert(0, temporary_file)
            # Make sure the chroot file is being read.
            assert find_hostname() == 'first line'
        finally:
            # Clean up.
            CHROOT_FILES.pop(0)
            os.unlink(temporary_file)
        # Test that unreadable chroot files don't break coloredlogs.
        try:
            CHROOT_FILES.insert(0, temporary_file)
            # Make sure that a usable value is still produced.
            assert find_hostname()
        finally:
            # Clean up.
            CHROOT_FILES.pop(0)

    def test_host_name_filter(self):
        """Make sure :func:`install()` integrates with :class:`~coloredlogs.HostNameFilter()`."""
        install(fmt='%(hostname)s')
        with CaptureOutput() as capturer:
            logging.info("A truly insignificant message ..")
            output = capturer.get_text()
            assert find_hostname() in output

    def test_program_name_filter(self):
        """Make sure :func:`install()` integrates with :class:`~coloredlogs.ProgramNameFilter()`."""
        install(fmt='%(programname)s')
        with CaptureOutput() as capturer:
            logging.info("A truly insignificant message ..")
            output = capturer.get_text()
            assert find_program_name() in output

    def test_username_filter(self):
        """Make sure :func:`install()` integrates with :class:`~coloredlogs.UserNameFilter()`."""
        install(fmt='%(username)s')
        with CaptureOutput() as capturer:
            logging.info("A truly insignificant message ..")
            output = capturer.get_text()
            assert find_username() in output

    def test_system_logging(self):
        """Make sure the :class:`coloredlogs.syslog.SystemLogging` context manager works."""
        system_log_file = self.find_system_log()
        expected_message = random_string(50)
        with SystemLogging(programname='coloredlogs-test-suite') as syslog:
            if not syslog:
                return self.skipTest("couldn't connect to syslog daemon")
            # When I tried out the system logging support on macOS 10.13.1 on
            # 2018-01-05 I found that while WARNING and ERROR messages show up
            # in the system log DEBUG and INFO messages don't. This explains
            # the importance of the level of the log message below.
            logging.error("%s", expected_message)
        # Retry the following assertion (for up to 60 seconds) to give the
        # logging daemon time to write our log message to disk. This
        # appears to be needed on MacOS workers on Travis CI, see:
        # https://travis-ci.org/xolox/python-coloredlogs/jobs/325245853
        retry(lambda: check_contents(system_log_file, expected_message, True))

    def test_system_logging_override(self):
        """Make sure the :class:`coloredlogs.syslog.is_syslog_supported` respects the override."""
        with PatchedItem(os.environ, 'COLOREDLOGS_SYSLOG', 'true'):
            assert is_syslog_supported() is True
        with PatchedItem(os.environ, 'COLOREDLOGS_SYSLOG', 'false'):
            assert is_syslog_supported() is False

    def test_syslog_shortcut_simple(self):
        """Make sure that ``coloredlogs.install(syslog=True)`` works."""
        system_log_file = self.find_system_log()
        expected_message = random_string(50)
        with cleanup_handlers():
            # See test_system_logging() for the importance of this log level.
            coloredlogs.install(syslog=True)
            logging.error("%s", expected_message)
        # See the comments in test_system_logging() on why this is retried.
        retry(lambda: check_contents(system_log_file, expected_message, True))

    def test_syslog_shortcut_enhanced(self):
        """Make sure that ``coloredlogs.install(syslog='warning')`` works."""
        system_log_file = self.find_system_log()
        the_expected_message = random_string(50)
        not_an_expected_message = random_string(50)
        with cleanup_handlers():
            # See test_system_logging() for the importance of these log levels.
            coloredlogs.install(syslog='error')
            logging.warning("%s", not_an_expected_message)
            logging.error("%s", the_expected_message)
        # See the comments in test_system_logging() on why this is retried.
        retry(lambda: check_contents(system_log_file, the_expected_message, True))
        retry(lambda: check_contents(system_log_file, not_an_expected_message, False))

    def test_name_normalization(self):
        """Make sure :class:`~coloredlogs.NameNormalizer` works as intended."""
        nn = NameNormalizer()
        for canonical_name in ['debug', 'info', 'warning', 'error', 'critical']:
            assert nn.normalize_name(canonical_name) == canonical_name
            assert nn.normalize_name(canonical_name.upper()) == canonical_name
        assert nn.normalize_name('warn') == 'warning'
        assert nn.normalize_name('fatal') == 'critical'

    def test_style_parsing(self):
        """Make sure :func:`~coloredlogs.parse_encoded_styles()` works as intended."""
        encoded_styles = 'debug=green;warning=yellow;error=red;critical=red,bold'
        decoded_styles = parse_encoded_styles(encoded_styles, normalize_key=lambda k: k.upper())
        assert sorted(decoded_styles.keys()) == sorted(['debug', 'warning', 'error', 'critical'])
        assert decoded_styles['debug']['color'] == 'green'
        assert decoded_styles['warning']['color'] == 'yellow'
        assert decoded_styles['error']['color'] == 'red'
        assert decoded_styles['critical']['color'] == 'red'
        assert decoded_styles['critical']['bold'] is True

    def test_is_verbose(self):
        """Make sure is_verbose() does what it should :-)."""
        set_level(logging.INFO)
        assert not is_verbose()
        set_level(logging.DEBUG)
        assert is_verbose()
        set_level(logging.VERBOSE)
        assert is_verbose()

    def test_increase_verbosity(self):
        """Make sure increase_verbosity() respects default and custom levels."""
        # Start from a known state.
        set_level(logging.INFO)
        assert get_level() == logging.INFO
        # INFO -> VERBOSE.
        increase_verbosity()
        assert get_level() == logging.VERBOSE
        # VERBOSE -> DEBUG.
        increase_verbosity()
        assert get_level() == logging.DEBUG
        # DEBUG -> SPAM.
        increase_verbosity()
        assert get_level() == logging.SPAM
        # SPAM -> NOTSET.
        increase_verbosity()
        assert get_level() == logging.NOTSET
        # NOTSET -> NOTSET.
        increase_verbosity()
        assert get_level() == logging.NOTSET

    def test_decrease_verbosity(self):
        """Make sure decrease_verbosity() respects default and custom levels."""
        # Start from a known state.
        set_level(logging.INFO)
        assert get_level() == logging.INFO
        # INFO -> NOTICE.
        decrease_verbosity()
        assert get_level() == logging.NOTICE
        # NOTICE -> WARNING.
        decrease_verbosity()
        assert get_level() == logging.WARNING
        # WARNING -> SUCCESS.
        decrease_verbosity()
        assert get_level() == logging.SUCCESS
        # SUCCESS -> ERROR.
        decrease_verbosity()
        assert get_level() == logging.ERROR
        # ERROR -> CRITICAL.
        decrease_verbosity()
        assert get_level() == logging.CRITICAL
        # CRITICAL -> CRITICAL.
        decrease_verbosity()
        assert get_level() == logging.CRITICAL

    def test_level_discovery(self):
        """Make sure find_defined_levels() always reports the levels defined in Python's standard library."""
        defined_levels = find_defined_levels()
        level_values = defined_levels.values()
        for number in (0, 10, 20, 30, 40, 50):
            assert number in level_values

    def test_walk_propagation_tree(self):
        """Make sure walk_propagation_tree() properly walks the tree of loggers."""
        root, parent, child, grand_child = self.get_logger_tree()
        # Check the default mode of operation.
        loggers = list(walk_propagation_tree(grand_child))
        assert loggers == [grand_child, child, parent, root]
        # Now change the propagation (non-default mode of operation).
        child.propagate = False
        loggers = list(walk_propagation_tree(grand_child))
        assert loggers == [grand_child, child]

    def test_find_handler(self):
        """Make sure find_handler() works as intended."""
        root, parent, child, grand_child = self.get_logger_tree()
        # Add some handlers to the tree.
        stream_handler = logging.StreamHandler()
        syslog_handler = logging.handlers.SysLogHandler()
        child.addHandler(stream_handler)
        parent.addHandler(syslog_handler)
        # Make sure the first matching handler is returned.
        matched_handler, matched_logger = find_handler(grand_child, lambda h: isinstance(h, logging.Handler))
        assert matched_handler is stream_handler
        # Make sure the first matching handler of the given type is returned.
        matched_handler, matched_logger = find_handler(child, lambda h: isinstance(h, logging.handlers.SysLogHandler))
        assert matched_handler is syslog_handler

    def get_logger_tree(self):
        """Create and return a tree of loggers."""
        # Get the root logger.
        root = logging.getLogger()
        # Create a top level logger for ourselves.
        parent_name = random_string()
        parent = logging.getLogger(parent_name)
        # Create a child logger.
        child_name = '%s.%s' % (parent_name, random_string())
        child = logging.getLogger(child_name)
        # Create a grand child logger.
        grand_child_name = '%s.%s' % (child_name, random_string())
        grand_child = logging.getLogger(grand_child_name)
        return root, parent, child, grand_child

    def test_support_for_milliseconds(self):
        """Make sure milliseconds are hidden by default but can be easily enabled."""
        # Check that the default log format doesn't include milliseconds.
        stream = StringIO()
        install(reconfigure=True, stream=stream)
        logging.info("This should not include milliseconds.")
        assert all(map(PLAIN_TEXT_PATTERN.match, stream.getvalue().splitlines()))
        # Check that milliseconds can be enabled via a shortcut.
        stream = StringIO()
        install(milliseconds=True, reconfigure=True, stream=stream)
        logging.info("This should include milliseconds.")
        assert all(map(PATTERN_INCLUDING_MILLISECONDS.match, stream.getvalue().splitlines()))

    def test_support_for_milliseconds_directive(self):
        """Make sure milliseconds using the ``%f`` directive are supported."""
        stream = StringIO()
        install(reconfigure=True, stream=stream, datefmt='%Y-%m-%dT%H:%M:%S.%f%z')
        logging.info("This should be timestamped according to #45.")
        assert re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4}\s', stream.getvalue())

    def test_plain_text_output_format(self):
        """Inspect the plain text output of coloredlogs."""
        logger = VerboseLogger(random_string(25))
        stream = StringIO()
        install(level=logging.NOTSET, logger=logger, stream=stream)
        # Test that filtering on severity works.
        logger.setLevel(logging.INFO)
        logger.debug("No one should see this message.")
        assert len(stream.getvalue().strip()) == 0
        # Test that the default output format looks okay in plain text.
        logger.setLevel(logging.NOTSET)
        for method, severity in ((logger.debug, 'DEBUG'),
                                 (logger.info, 'INFO'),
                                 (logger.verbose, 'VERBOSE'),
                                 (logger.warning, 'WARNING'),
                                 (logger.error, 'ERROR'),
                                 (logger.critical, 'CRITICAL')):
            # XXX Workaround for a regression in Python 3.7 caused by the
            # Logger.isEnabledFor() method using stale cache entries. If we
            # don't clear the cache then logger.isEnabledFor(logging.DEBUG)
            # returns False and no DEBUG message is emitted.
            try:
                logger._cache.clear()
            except AttributeError:
                pass
            # Prepare the text.
            text = "This is a message with severity %r." % severity.lower()
            # Log the message with the given severity.
            method(text)
            # Get the line of output generated by the handler.
            output = stream.getvalue()
            lines = output.splitlines()
            last_line = lines[-1]
            assert text in last_line
            assert severity in last_line
            assert PLAIN_TEXT_PATTERN.match(last_line)

    def test_dynamic_stderr_lookup(self):
        """Make sure coloredlogs.install() uses StandardErrorHandler when possible."""
        coloredlogs.install()
        # Redirect sys.stderr to a temporary buffer.
        initial_stream = StringIO()
        initial_text = "Which stream will receive this text?"
        with PatchedAttribute(sys, 'stderr', initial_stream):
            logging.info(initial_text)
        assert initial_text in initial_stream.getvalue()
        # Redirect sys.stderr again, to a different destination.
        subsequent_stream = StringIO()
        subsequent_text = "And which stream will receive this other text?"
        with PatchedAttribute(sys, 'stderr', subsequent_stream):
            logging.info(subsequent_text)
        assert subsequent_text in subsequent_stream.getvalue()

    def test_force_enable(self):
        """Make sure ANSI escape sequences can be forced (bypassing auto-detection)."""
        interpreter = subprocess.Popen([
            sys.executable, "-c", ";".join([
                "import coloredlogs, logging",
                "coloredlogs.install(isatty=True)",
                "logging.info('Hello world')",
            ]),
        ], stderr=subprocess.PIPE)
        stdout, stderr = interpreter.communicate()
        assert ANSI_CSI in stderr.decode('UTF-8')

    def test_auto_disable(self):
        """
        Make sure ANSI escape sequences are not emitted when logging output is being redirected.

        This is a regression test for https://github.com/xolox/python-coloredlogs/issues/100.

        It works as follows:

        1. We mock an interactive terminal using 'capturer' to ensure that this
           test works inside test drivers that capture output (like pytest).

        2. We launch a subprocess (to ensure a clean process state) where
           stderr is captured but stdout is not, emulating issue #100.

        3. The output captured on stderr contained ANSI escape sequences after
           this test was written and before the issue was fixed, so now this
           serves as a regression test for issue #100.
        """
        with CaptureOutput():
            interpreter = subprocess.Popen([
                sys.executable, "-c", ";".join([
                    "import coloredlogs, logging",
                    "coloredlogs.install()",
                    "logging.info('Hello world')",
                ]),
            ], stderr=subprocess.PIPE)
            stdout, stderr = interpreter.communicate()
            assert ANSI_CSI not in stderr.decode('UTF-8')

    def test_env_disable(self):
        """Make sure ANSI escape sequences can be disabled using ``$NO_COLOR``."""
        with PatchedItem(os.environ, 'NO_COLOR', 'I like monochrome'):
            with CaptureOutput() as capturer:
                subprocess.check_call([
                    sys.executable, "-c", ";".join([
                        "import coloredlogs, logging",
                        "coloredlogs.install()",
                        "logging.info('Hello world')",
                    ]),
                ])
                output = capturer.get_text()
                assert ANSI_CSI not in output

    def test_html_conversion(self):
        """Check the conversion from ANSI escape sequences to HTML."""
        # Check conversion of colored text.
        for color_name, ansi_code in ANSI_COLOR_CODES.items():
            ansi_encoded_text = 'plain text followed by %s text' % ansi_wrap(color_name, color=color_name)
            expected_html = format(
                '<code>plain text followed by <span style="color:{css}">{name}</span> text</code>',
                css=EIGHT_COLOR_PALETTE[ansi_code], name=color_name,
            )
            self.assertEqual(expected_html, convert(ansi_encoded_text))
        # Check conversion of bright colored text.
        expected_html = '<code><span style="color:#FF0">bright yellow</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('bright yellow', color='yellow', bright=True)))
        # Check conversion of text with a background color.
        expected_html = '<code><span style="background-color:#DE382B">red background</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('red background', background='red')))
        # Check conversion of text with a bright background color.
        expected_html = '<code><span style="background-color:#F00">bright red background</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('bright red background', background='red', bright=True)))
        # Check conversion of text that uses the 256 color mode palette as a foreground color.
        expected_html = '<code><span style="color:#FFAF00">256 color mode foreground</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('256 color mode foreground', color=214)))
        # Check conversion of text that uses the 256 color mode palette as a background color.
        expected_html = '<code><span style="background-color:#AF0000">256 color mode background</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('256 color mode background', background=124)))
        # Check that invalid 256 color mode indexes don't raise exceptions.
        expected_html = '<code>plain text expected</code>'
        self.assertEqual(expected_html, convert('\x1b[38;5;256mplain text expected\x1b[0m'))
        # Check conversion of bold text.
        expected_html = '<code><span style="font-weight:bold">bold text</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('bold text', bold=True)))
        # Check conversion of underlined text.
        expected_html = '<code><span style="text-decoration:underline">underlined text</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('underlined text', underline=True)))
        # Check conversion of strike-through text.
        expected_html = '<code><span style="text-decoration:line-through">strike-through text</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('strike-through text', strike_through=True)))
        # Check conversion of inverse text.
        expected_html = '<code><span style="background-color:#FFC706;color:#000">inverse</span></code>'
        self.assertEqual(expected_html, convert(ansi_wrap('inverse', color='yellow', inverse=True)))
        # Check conversion of URLs.
        for sample_text in 'www.python.org', 'http://coloredlogs.rtfd.org', 'https://coloredlogs.rtfd.org':
            sample_url = sample_text if '://' in sample_text else ('http://' + sample_text)
            expected_html = '<code><a href="%s" style="color:inherit">%s</a></code>' % (sample_url, sample_text)
            self.assertEqual(expected_html, convert(sample_text))
        # Check that the capture pattern for URLs doesn't match ANSI escape
        # sequences and also check that the short hand for the 0 reset code is
        # supported. These are tests for regressions of bugs found in
        # coloredlogs <= 8.0.
        reset_short_hand = '\x1b[0m'
        blue_underlined = ansi_style(color='blue', underline=True)
        ansi_encoded_text = '<%shttps://coloredlogs.readthedocs.io%s>' % (blue_underlined, reset_short_hand)
        expected_html = (
            '<code>&lt;<span style="color:#006FB8;text-decoration:underline">'
            '<a href="https://coloredlogs.readthedocs.io" style="color:inherit">'
            'https://coloredlogs.readthedocs.io'
            '</a></span>&gt;</code>'
        )
        self.assertEqual(expected_html, convert(ansi_encoded_text))

    def test_output_interception(self):
        """Test capturing of output from external commands."""
        expected_output = 'testing, 1, 2, 3 ..'
        actual_output = capture(['echo', expected_output])
        assert actual_output.strip() == expected_output.strip()

    def test_enable_colored_cron_mailer(self):
        """Test that automatic ANSI to HTML conversion when running under ``cron`` can be enabled."""
        with PatchedItem(os.environ, 'CONTENT_TYPE', 'text/html'):
            with ColoredCronMailer() as mailer:
                assert mailer.is_enabled

    def test_disable_colored_cron_mailer(self):
        """Test that automatic ANSI to HTML conversion when running under ``cron`` can be disabled."""
        with PatchedItem(os.environ, 'CONTENT_TYPE', 'text/plain'):
            with ColoredCronMailer() as mailer:
                assert not mailer.is_enabled

    def test_auto_install(self):
        """Test :func:`coloredlogs.auto_install()`."""
        needle = random_string()
        command_line = [sys.executable, '-c', 'import logging; logging.info(%r)' % needle]
        # Sanity check that log messages aren't enabled by default.
        with CaptureOutput() as capturer:
            os.environ['COLOREDLOGS_AUTO_INSTALL'] = 'false'
            subprocess.check_call(command_line)
            output = capturer.get_text()
        assert needle not in output
        # Test that the $COLOREDLOGS_AUTO_INSTALL environment variable can be
        # used to automatically call coloredlogs.install() during initialization.
        with CaptureOutput() as capturer:
            os.environ['COLOREDLOGS_AUTO_INSTALL'] = 'true'
            subprocess.check_call(command_line)
            output = capturer.get_text()
        assert needle in output

    def test_cli_demo(self):
        """Test the command line colored logging demonstration."""
        with CaptureOutput() as capturer:
            main('coloredlogs', '--demo')
            output = capturer.get_text()
        # Make sure the output contains all of the expected logging level names.
        for name in 'debug', 'info', 'warning', 'error', 'critical':
            assert name.upper() in output

    def test_cli_conversion(self):
        """Test the command line HTML conversion."""
        output = main('coloredlogs', '--convert', 'coloredlogs', '--demo', capture=True)
        # Make sure the output is encoded as HTML.
        assert '<span' in output

    def test_empty_conversion(self):
        """
        Test that conversion of empty output produces no HTML.

        This test was added because I found that ``coloredlogs --convert`` when
        used in a cron job could cause cron to send out what appeared to be
        empty emails. On more careful inspection the body of those emails was
        ``<code></code>``. By not emitting the wrapper element when no other
        HTML is generated, cron will not send out an email.
        """
        output = main('coloredlogs', '--convert', 'true', capture=True)
        assert not output.strip()

    def test_implicit_usage_message(self):
        """Test that the usage message is shown when no actions are given."""
        assert 'Usage:' in main('coloredlogs', capture=True)

    def test_explicit_usage_message(self):
        """Test that the usage message is shown when ``--help`` is given."""
        assert 'Usage:' in main('coloredlogs', '--help', capture=True)

    def test_custom_record_factory(self):
        """
        Test that custom LogRecord factories are supported.

        This test is a bit convoluted because the logging module suppresses
        exceptions. We monkey patch the method suspected of encountering
        exceptions so that we can tell after it was called whether any
        exceptions occurred (despite the exceptions not propagating).
        """
        if not hasattr(logging, 'getLogRecordFactory'):
            return self.skipTest("this test requires Python >= 3.2")

        exceptions = []
        original_method = ColoredFormatter.format
        original_factory = logging.getLogRecordFactory()

        def custom_factory(*args, **kwargs):
            record = original_factory(*args, **kwargs)
            record.custom_attribute = 0xdecafbad
            return record

        def custom_method(*args, **kw):
            try:
                return original_method(*args, **kw)
            except Exception as e:
                exceptions.append(e)
                raise

        with PatchedAttribute(ColoredFormatter, 'format', custom_method):
            logging.setLogRecordFactory(custom_factory)
            try:
                demonstrate_colored_logging()
            finally:
                logging.setLogRecordFactory(original_factory)

        # Ensure that no exceptions were triggered.
        assert not exceptions


def check_contents(filename, contents, match):
    """Check if a line in a file contains an expected string."""
    with open(filename) as handle:
        assert any(contents in line for line in handle) == match


def main(*arguments, **options):
    """Wrap the command line interface to make it easier to test."""
    capture = options.get('capture', False)
    saved_argv = sys.argv
    saved_stdout = sys.stdout
    try:
        sys.argv = arguments
        if capture:
            sys.stdout = StringIO()
        coloredlogs.cli.main()
        if capture:
            return sys.stdout.getvalue()
    finally:
        sys.argv = saved_argv
        sys.stdout = saved_stdout


@contextlib.contextmanager
def cleanup_handlers():
    """Context manager to cleanup output handlers."""
    # There's nothing to set up so we immediately yield control.
    yield
    # After the with block ends we cleanup any output handlers.
    for match_func in match_stream_handler, match_syslog_handler:
        handler, logger = find_handler(logging.getLogger(), match_func)
        if handler and logger:
            logger.removeHandler(handler)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wsproto\frame_protocol.py ===
"""
wsproto/frame_protocol
~~~~~~~~~~~~~~~~~~~~~~

WebSocket frame protocol implementation.
"""

import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .extensions import Extension  # pragma: no cover


_XOR_TABLE = [bytes(a ^ b for a in range(256)) for b in range(256)]


class XorMaskerSimple:
    def __init__(self, masking_key: bytes) -> None:
        self._masking_key = masking_key

    def process(self, data: bytes) -> bytes:
        if data:
            data_array = bytearray(data)
            a, b, c, d = (_XOR_TABLE[n] for n in self._masking_key)
            data_array[::4] = data_array[::4].translate(a)
            data_array[1::4] = data_array[1::4].translate(b)
            data_array[2::4] = data_array[2::4].translate(c)
            data_array[3::4] = data_array[3::4].translate(d)

            # Rotate the masking key so that the next usage continues
            # with the next key element, rather than restarting.
            key_rotation = len(data) % 4
            self._masking_key = (
                self._masking_key[key_rotation:] + self._masking_key[:key_rotation]
            )

            return bytes(data_array)
        return data


class XorMaskerNull:
    def process(self, data: bytes) -> bytes:
        return data


# RFC6455, Section 5.2 - Base Framing Protocol

# Payload length constants
PAYLOAD_LENGTH_TWO_BYTE = 126
PAYLOAD_LENGTH_EIGHT_BYTE = 127
MAX_PAYLOAD_NORMAL = 125
MAX_PAYLOAD_TWO_BYTE = 2**16 - 1
MAX_PAYLOAD_EIGHT_BYTE = 2**64 - 1
MAX_FRAME_PAYLOAD = MAX_PAYLOAD_EIGHT_BYTE

# MASK and PAYLOAD LEN are packed into a byte
MASK_MASK = 0x80
PAYLOAD_LEN_MASK = 0x7F

# FIN, RSV[123] and OPCODE are packed into a single byte
FIN_MASK = 0x80
RSV1_MASK = 0x40
RSV2_MASK = 0x20
RSV3_MASK = 0x10
OPCODE_MASK = 0x0F


class Opcode(IntEnum):
    """
    RFC 6455, Section 5.2 - Base Framing Protocol
    """

    #: Continuation frame
    CONTINUATION = 0x0

    #: Text message
    TEXT = 0x1

    #: Binary message
    BINARY = 0x2

    #: Close frame
    CLOSE = 0x8

    #: Ping frame
    PING = 0x9

    #: Pong frame
    PONG = 0xA

    def iscontrol(self) -> bool:
        return bool(self & 0x08)


class CloseReason(IntEnum):
    """
    RFC 6455, Section 7.4.1 - Defined Status Codes
    """

    #: indicates a normal closure, meaning that the purpose for
    #: which the connection was established has been fulfilled.
    NORMAL_CLOSURE = 1000

    #: indicates that an endpoint is "going away", such as a server
    #: going down or a browser having navigated away from a page.
    GOING_AWAY = 1001

    #: indicates that an endpoint is terminating the connection due
    #: to a protocol error.
    PROTOCOL_ERROR = 1002

    #: indicates that an endpoint is terminating the connection
    #: because it has received a type of data it cannot accept (e.g., an
    #: endpoint that understands only text data MAY send this if it
    #: receives a binary message).
    UNSUPPORTED_DATA = 1003

    #: Reserved.  The specific meaning might be defined in the future.
    # DON'T DEFINE THIS: RESERVED_1004 = 1004

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that no status
    #: code was actually present.
    NO_STATUS_RCVD = 1005

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that the
    #: connection was closed abnormally, e.g., without sending or
    #: receiving a Close control frame.
    ABNORMAL_CLOSURE = 1006

    #: indicates that an endpoint is terminating the connection
    #: because it has received data within a message that was not
    #: consistent with the type of the message (e.g., non-UTF-8 [RFC3629]
    #: data within a text message).
    INVALID_FRAME_PAYLOAD_DATA = 1007

    #: indicates that an endpoint is terminating the connection
    #: because it has received a message that violates its policy.  This
    #: is a generic status code that can be returned when there is no
    #: other more suitable status code (e.g., 1003 or 1009) or if there
    #: is a need to hide specific details about the policy.
    POLICY_VIOLATION = 1008

    #: indicates that an endpoint is terminating the connection
    #: because it has received a message that is too big for it to
    #: process.
    MESSAGE_TOO_BIG = 1009

    #: indicates that an endpoint (client) is terminating the
    #: connection because it has expected the server to negotiate one or
    #: more extension, but the server didn't return them in the response
    #: message of the WebSocket handshake.  The list of extensions that
    #: are needed SHOULD appear in the /reason/ part of the Close frame.
    #: Note that this status code is not used by the server, because it
    #: can fail the WebSocket handshake instead.
    MANDATORY_EXT = 1010

    #: indicates that a server is terminating the connection because
    #: it encountered an unexpected condition that prevented it from
    #: fulfilling the request.
    INTERNAL_ERROR = 1011

    #: Server/service is restarting
    #: (not part of RFC6455)
    SERVICE_RESTART = 1012

    #: Temporary server condition forced blocking client's request
    #: (not part of RFC6455)
    TRY_AGAIN_LATER = 1013

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that the
    #: connection was closed due to a failure to perform a TLS handshake
    #: (e.g., the server certificate can't be verified).
    TLS_HANDSHAKE_FAILED = 1015


# RFC 6455, Section 7.4.1 - Defined Status Codes
LOCAL_ONLY_CLOSE_REASONS = (
    CloseReason.NO_STATUS_RCVD,
    CloseReason.ABNORMAL_CLOSURE,
    CloseReason.TLS_HANDSHAKE_FAILED,
)


# RFC 6455, Section 7.4.2 - Status Code Ranges
MIN_CLOSE_REASON = 1000
MIN_PROTOCOL_CLOSE_REASON = 1000
MAX_PROTOCOL_CLOSE_REASON = 2999
MIN_LIBRARY_CLOSE_REASON = 3000
MAX_LIBRARY_CLOSE_REASON = 3999
MIN_PRIVATE_CLOSE_REASON = 4000
MAX_PRIVATE_CLOSE_REASON = 4999
MAX_CLOSE_REASON = 4999


NULL_MASK = struct.pack("!I", 0)


class ParseFailed(Exception):
    def __init__(
        self, msg: str, code: CloseReason = CloseReason.PROTOCOL_ERROR
    ) -> None:
        super().__init__(msg)
        self.code = code


class RsvBits(NamedTuple):
    rsv1: bool
    rsv2: bool
    rsv3: bool


class Header(NamedTuple):
    fin: bool
    rsv: RsvBits
    opcode: Opcode
    payload_len: int
    masking_key: Optional[bytes]


class Frame(NamedTuple):
    opcode: Opcode
    payload: Union[bytes, str, Tuple[int, str]]
    frame_finished: bool
    message_finished: bool


def _truncate_utf8(data: bytes, nbytes: int) -> bytes:
    if len(data) <= nbytes:
        return data

    # Truncate
    data = data[:nbytes]
    # But we might have cut a codepoint in half, in which case we want to
    # discard the partial character so the data is at least
    # well-formed. This is a little inefficient since it processes the
    # whole message twice when in theory we could just peek at the last
    # few characters, but since this is only used for close messages (max
    # length = 125 bytes) it really doesn't matter.
    data = data.decode("utf-8", errors="ignore").encode("utf-8")
    return data


class Buffer:
    def __init__(self, initial_bytes: Optional[bytes] = None) -> None:
        self.buffer = bytearray()
        self.bytes_used = 0
        if initial_bytes:
            self.feed(initial_bytes)

    def feed(self, new_bytes: bytes) -> None:
        self.buffer += new_bytes

    def consume_at_most(self, nbytes: int) -> bytes:
        if not nbytes:
            return bytearray()

        data = self.buffer[self.bytes_used : self.bytes_used + nbytes]
        self.bytes_used += len(data)
        return data

    def consume_exactly(self, nbytes: int) -> Optional[bytes]:
        if len(self.buffer) - self.bytes_used < nbytes:
            return None

        return self.consume_at_most(nbytes)

    def commit(self) -> None:
        # In CPython 3.4+, del[:n] is amortized O(n), *not* quadratic
        del self.buffer[: self.bytes_used]
        self.bytes_used = 0

    def rollback(self) -> None:
        self.bytes_used = 0

    def __len__(self) -> int:
        return len(self.buffer)


class MessageDecoder:
    def __init__(self) -> None:
        self.opcode: Optional[Opcode] = None
        self.decoder: Optional[IncrementalDecoder] = None

    def process_frame(self, frame: Frame) -> Frame:
        assert not frame.opcode.iscontrol()

        if self.opcode is None:
            if frame.opcode is Opcode.CONTINUATION:
                raise ParseFailed("unexpected CONTINUATION")
            self.opcode = frame.opcode
        elif frame.opcode is not Opcode.CONTINUATION:
            raise ParseFailed("expected CONTINUATION, got %r" % frame.opcode)

        if frame.opcode is Opcode.TEXT:
            self.decoder = getincrementaldecoder("utf-8")()

        finished = frame.frame_finished and frame.message_finished

        if self.decoder is None:
            data = frame.payload
        else:
            assert isinstance(frame.payload, (bytes, bytearray))
            try:
                data = self.decoder.decode(frame.payload, finished)
            except UnicodeDecodeError as exc:
                raise ParseFailed(str(exc), CloseReason.INVALID_FRAME_PAYLOAD_DATA)

        frame = Frame(self.opcode, data, frame.frame_finished, finished)

        if finished:
            self.opcode = None
            self.decoder = None

        return frame


class FrameDecoder:
    def __init__(
        self, client: bool, extensions: Optional[List["Extension"]] = None
    ) -> None:
        self.client = client
        self.extensions = extensions or []

        self.buffer = Buffer()

        self.header: Optional[Header] = None
        self.effective_opcode: Optional[Opcode] = None
        self.masker: Union[None, XorMaskerNull, XorMaskerSimple] = None
        self.payload_required = 0
        self.payload_consumed = 0

    def receive_bytes(self, data: bytes) -> None:
        self.buffer.feed(data)

    def process_buffer(self) -> Optional[Frame]:
        if not self.header:
            if not self.parse_header():
                return None
        # parse_header() sets these.
        assert self.header is not None
        assert self.masker is not None
        assert self.effective_opcode is not None

        if len(self.buffer) < self.payload_required:
            return None

        payload_remaining = self.header.payload_len - self.payload_consumed
        payload = self.buffer.consume_at_most(payload_remaining)
        if not payload and self.header.payload_len > 0:
            return None
        self.buffer.commit()

        self.payload_consumed += len(payload)
        finished = self.payload_consumed == self.header.payload_len

        payload = self.masker.process(payload)

        for extension in self.extensions:
            payload_ = extension.frame_inbound_payload_data(self, payload)
            if isinstance(payload_, CloseReason):
                raise ParseFailed("error in extension", payload_)
            payload = payload_

        if finished:
            final = bytearray()
            for extension in self.extensions:
                result = extension.frame_inbound_complete(self, self.header.fin)
                if isinstance(result, CloseReason):
                    raise ParseFailed("error in extension", result)
                if result is not None:
                    final += result
            payload += final

        frame = Frame(self.effective_opcode, payload, finished, self.header.fin)

        if finished:
            self.header = None
            self.effective_opcode = None
            self.masker = None
        else:
            self.effective_opcode = Opcode.CONTINUATION

        return frame

    def parse_header(self) -> bool:
        data = self.buffer.consume_exactly(2)
        if data is None:
            self.buffer.rollback()
            return False

        fin = bool(data[0] & FIN_MASK)
        rsv = RsvBits(
            bool(data[0] & RSV1_MASK),
            bool(data[0] & RSV2_MASK),
            bool(data[0] & RSV3_MASK),
        )
        opcode = data[0] & OPCODE_MASK
        try:
            opcode = Opcode(opcode)
        except ValueError:
            raise ParseFailed(f"Invalid opcode {opcode:#x}")

        if opcode.iscontrol() and not fin:
            raise ParseFailed("Invalid attempt to fragment control frame")

        has_mask = bool(data[1] & MASK_MASK)
        payload_len_short = data[1] & PAYLOAD_LEN_MASK
        payload_len = self.parse_extended_payload_length(opcode, payload_len_short)
        if payload_len is None:
            self.buffer.rollback()
            return False

        self.extension_processing(opcode, rsv, payload_len)

        if has_mask and self.client:
            raise ParseFailed("client received unexpected masked frame")
        if not has_mask and not self.client:
            raise ParseFailed("server received unexpected unmasked frame")
        if has_mask:
            masking_key = self.buffer.consume_exactly(4)
            if masking_key is None:
                self.buffer.rollback()
                return False
            self.masker = XorMaskerSimple(masking_key)
        else:
            self.masker = XorMaskerNull()

        self.buffer.commit()
        self.header = Header(fin, rsv, opcode, payload_len, None)
        self.effective_opcode = self.header.opcode
        if self.header.opcode.iscontrol():
            self.payload_required = payload_len
        else:
            self.payload_required = 0
        self.payload_consumed = 0
        return True

    def parse_extended_payload_length(
        self, opcode: Opcode, payload_len: int
    ) -> Optional[int]:
        if opcode.iscontrol() and payload_len > MAX_PAYLOAD_NORMAL:
            raise ParseFailed("Control frame with payload len > 125")
        if payload_len == PAYLOAD_LENGTH_TWO_BYTE:
            data = self.buffer.consume_exactly(2)
            if data is None:
                return None
            (payload_len,) = struct.unpack("!H", data)
            if payload_len <= MAX_PAYLOAD_NORMAL:
                raise ParseFailed(
                    "Payload length used 2 bytes when 1 would have sufficed"
                )
        elif payload_len == PAYLOAD_LENGTH_EIGHT_BYTE:
            data = self.buffer.consume_exactly(8)
            if data is None:
                return None
            (payload_len,) = struct.unpack("!Q", data)
            if payload_len <= MAX_PAYLOAD_TWO_BYTE:
                raise ParseFailed(
                    "Payload length used 8 bytes when 2 would have sufficed"
                )
            if payload_len >> 63:
                # I'm not sure why this is illegal, but that's what the RFC
                # says, so...
                raise ParseFailed("8-byte payload length with non-zero MSB")

        return payload_len

    def extension_processing(
        self, opcode: Opcode, rsv: RsvBits, payload_len: int
    ) -> None:
        rsv_used = [False, False, False]
        for extension in self.extensions:
            result = extension.frame_inbound_header(self, opcode, rsv, payload_len)
            if isinstance(result, CloseReason):
                raise ParseFailed("error in extension", result)
            for bit, used in enumerate(result):
                if used:
                    rsv_used[bit] = True
        for expected, found in zip(rsv_used, rsv):
            if found and not expected:
                raise ParseFailed("Reserved bit set unexpectedly")


class FrameProtocol:
    def __init__(self, client: bool, extensions: List["Extension"]) -> None:
        self.client = client
        self.extensions = [ext for ext in extensions if ext.enabled()]

        # Global state
        self._frame_decoder = FrameDecoder(self.client, self.extensions)
        self._message_decoder = MessageDecoder()
        self._parse_more = self._parse_more_gen()

        self._outbound_opcode: Optional[Opcode] = None

    def _process_close(self, frame: Frame) -> Frame:
        data = frame.payload
        assert isinstance(data, (bytes, bytearray))

        if not data:
            # "If this Close control frame contains no status code, _The
            # WebSocket Connection Close Code_ is considered to be 1005"
            data = (CloseReason.NO_STATUS_RCVD, "")
        elif len(data) == 1:
            raise ParseFailed("CLOSE with 1 byte payload")
        else:
            (code,) = struct.unpack("!H", data[:2])
            if code < MIN_CLOSE_REASON or code > MAX_CLOSE_REASON:
                raise ParseFailed("CLOSE with invalid code")
            try:
                code = CloseReason(code)
            except ValueError:
                pass
            if code in LOCAL_ONLY_CLOSE_REASONS:
                raise ParseFailed("remote CLOSE with local-only reason")
            if not isinstance(code, CloseReason) and code <= MAX_PROTOCOL_CLOSE_REASON:
                raise ParseFailed("CLOSE with unknown reserved code")
            try:
                reason = data[2:].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ParseFailed(
                    "Error decoding CLOSE reason: " + str(exc),
                    CloseReason.INVALID_FRAME_PAYLOAD_DATA,
                )
            data = (code, reason)

        return Frame(frame.opcode, data, frame.frame_finished, frame.message_finished)

    def _parse_more_gen(self) -> Generator[Optional[Frame], None, None]:
        # Consume as much as we can from self._buffer, yielding events, and
        # then yield None when we need more data. Or raise ParseFailed.

        # XX FIXME this should probably be refactored so that we never see
        # disabled extensions in the first place...
        self.extensions = [ext for ext in self.extensions if ext.enabled()]
        closed = False

        while not closed:
            frame = self._frame_decoder.process_buffer()

            if frame is not None:
                if not frame.opcode.iscontrol():
                    frame = self._message_decoder.process_frame(frame)
                elif frame.opcode == Opcode.CLOSE:
                    frame = self._process_close(frame)
                    closed = True

            yield frame

    def receive_bytes(self, data: bytes) -> None:
        self._frame_decoder.receive_bytes(data)

    def received_frames(self) -> Generator[Frame, None, None]:
        for event in self._parse_more:
            if event is None:
                break
            else:
                yield event

    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> bytes:
        payload = bytearray()
        if code is CloseReason.NO_STATUS_RCVD:
            code = None
        if code is None and reason:
            raise TypeError("cannot specify a reason without a code")
        if code in LOCAL_ONLY_CLOSE_REASONS:
            code = CloseReason.NORMAL_CLOSURE
        if code is not None:
            payload += bytearray(struct.pack("!H", code))
            if reason is not None:
                payload += _truncate_utf8(
                    reason.encode("utf-8"), MAX_PAYLOAD_NORMAL - 2
                )

        return self._serialize_frame(Opcode.CLOSE, payload)

    def ping(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PING, payload)

    def pong(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PONG, payload)

    def send_data(
        self, payload: Union[bytes, bytearray, str] = b"", fin: bool = True
    ) -> bytes:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            opcode = Opcode.BINARY
        elif isinstance(payload, str):
            opcode = Opcode.TEXT
            payload = payload.encode("utf-8")
        else:
            raise ValueError("Must provide bytes or text")

        if self._outbound_opcode is None:
            self._outbound_opcode = opcode
        elif self._outbound_opcode is not opcode:
            raise TypeError("Data type mismatch inside message")
        else:
            opcode = Opcode.CONTINUATION

        if fin:
            self._outbound_opcode = None

        return self._serialize_frame(opcode, payload, fin)

    def _make_fin_rsv_opcode(self, fin: bool, rsv: RsvBits, opcode: Opcode) -> int:
        fin_bits = int(fin) << 7
        rsv_bits = (int(rsv.rsv1) << 6) + (int(rsv.rsv2) << 5) + (int(rsv.rsv3) << 4)
        opcode_bits = int(opcode)

        return fin_bits | rsv_bits | opcode_bits

    def _serialize_frame(
        self, opcode: Opcode, payload: bytes = b"", fin: bool = True
    ) -> bytes:
        rsv = RsvBits(False, False, False)
        for extension in reversed(self.extensions):
            rsv, payload = extension.frame_outbound(self, opcode, rsv, payload, fin)

        fin_rsv_opcode = self._make_fin_rsv_opcode(fin, rsv, opcode)

        payload_length = len(payload)
        quad_payload = False
        if payload_length <= MAX_PAYLOAD_NORMAL:
            first_payload = payload_length
            second_payload = None
        elif payload_length <= MAX_PAYLOAD_TWO_BYTE:
            first_payload = PAYLOAD_LENGTH_TWO_BYTE
            second_payload = payload_length
        else:
            first_payload = PAYLOAD_LENGTH_EIGHT_BYTE
            second_payload = payload_length
            quad_payload = True

        if self.client:
            first_payload |= 1 << 7

        header = bytearray([fin_rsv_opcode, first_payload])
        if second_payload is not None:
            if opcode.iscontrol():
                raise ValueError("payload too long for control frame")
            if quad_payload:
                header += bytearray(struct.pack("!Q", second_payload))
            else:
                header += bytearray(struct.pack("!H", second_payload))

        if self.client:
            # "The masking key is a 32-bit value chosen at random by the
            # client.  When preparing a masked frame, the client MUST pick a
            # fresh masking key from the set of allowed 32-bit values.  The
            # masking key needs to be unpredictable; thus, the masking key
            # MUST be derived from a strong source of entropy, and the masking
            # key for a given frame MUST NOT make it simple for a server/proxy
            # to predict the masking key for a subsequent frame.  The
            # unpredictability of the masking key is essential to prevent
            # authors of malicious applications from selecting the bytes that
            # appear on the wire."
            #   -- https://tools.ietf.org/html/rfc6455#section-5.3
            masking_key = os.urandom(4)
            masker = XorMaskerSimple(masking_key)
            return header + masking_key + masker.process(payload)

        return header + payload

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\math2.py ===
"""
This module complements the math and cmath builtin modules by providing
fast machine precision versions of some additional functions (gamma, ...)
and wrapping math/cmath functions so that they can be called with either
real or complex arguments.
"""

import operator
import math
import cmath

# Irrational (?) constants
pi = 3.1415926535897932385
e = 2.7182818284590452354
sqrt2 = 1.4142135623730950488
sqrt5 = 2.2360679774997896964
phi = 1.6180339887498948482
ln2 = 0.69314718055994530942
ln10 = 2.302585092994045684
euler = 0.57721566490153286061
catalan = 0.91596559417721901505
khinchin = 2.6854520010653064453
apery = 1.2020569031595942854

logpi = 1.1447298858494001741

def _mathfun_real(f_real, f_complex):
    def f(x, **kwargs):
        if type(x) is float:
            return f_real(x)
        if type(x) is complex:
            return f_complex(x)
        try:
            x = float(x)
            return f_real(x)
        except (TypeError, ValueError):
            x = complex(x)
            return f_complex(x)
    f.__name__ = f_real.__name__
    return f

def _mathfun(f_real, f_complex):
    def f(x, **kwargs):
        if type(x) is complex:
            return f_complex(x)
        try:
            return f_real(float(x))
        except (TypeError, ValueError):
            return f_complex(complex(x))
    f.__name__ = f_real.__name__
    return f

def _mathfun_n(f_real, f_complex):
    def f(*args, **kwargs):
        try:
            return f_real(*(float(x) for x in args))
        except (TypeError, ValueError):
            return f_complex(*(complex(x) for x in args))
    f.__name__ = f_real.__name__
    return f

# Workaround for non-raising log and sqrt in Python 2.5 and 2.4
# on Unix system
try:
    math.log(-2.0)
    def math_log(x):
        if x <= 0.0:
            raise ValueError("math domain error")
        return math.log(x)
    def math_sqrt(x):
        if x < 0.0:
            raise ValueError("math domain error")
        return math.sqrt(x)
except (ValueError, TypeError):
    math_log = math.log
    math_sqrt = math.sqrt

pow = _mathfun_n(operator.pow, lambda x, y: complex(x)**y)
log = _mathfun_n(math_log, cmath.log)
sqrt = _mathfun(math_sqrt, cmath.sqrt)
exp = _mathfun_real(math.exp, cmath.exp)

cos = _mathfun_real(math.cos, cmath.cos)
sin = _mathfun_real(math.sin, cmath.sin)
tan = _mathfun_real(math.tan, cmath.tan)

acos = _mathfun(math.acos, cmath.acos)
asin = _mathfun(math.asin, cmath.asin)
atan = _mathfun_real(math.atan, cmath.atan)

cosh = _mathfun_real(math.cosh, cmath.cosh)
sinh = _mathfun_real(math.sinh, cmath.sinh)
tanh = _mathfun_real(math.tanh, cmath.tanh)

floor = _mathfun_real(math.floor,
    lambda z: complex(math.floor(z.real), math.floor(z.imag)))
ceil = _mathfun_real(math.ceil,
    lambda z: complex(math.ceil(z.real), math.ceil(z.imag)))


cos_sin = _mathfun_real(lambda x: (math.cos(x), math.sin(x)),
                        lambda z: (cmath.cos(z), cmath.sin(z)))

cbrt = _mathfun(lambda x: x**(1./3), lambda z: z**(1./3))

def nthroot(x, n):
    r = 1./n
    try:
        return float(x) ** r
    except (ValueError, TypeError):
        return complex(x) ** r

def _sinpi_real(x):
    if x < 0:
        return -_sinpi_real(-x)
    n, r = divmod(x, 0.5)
    r *= pi
    n %= 4
    if n == 0: return math.sin(r)
    if n == 1: return math.cos(r)
    if n == 2: return -math.sin(r)
    if n == 3: return -math.cos(r)

def _cospi_real(x):
    if x < 0:
        x = -x
    n, r = divmod(x, 0.5)
    r *= pi
    n %= 4
    if n == 0: return math.cos(r)
    if n == 1: return -math.sin(r)
    if n == 2: return -math.cos(r)
    if n == 3: return math.sin(r)

def _sinpi_complex(z):
    if z.real < 0:
        return -_sinpi_complex(-z)
    n, r = divmod(z.real, 0.5)
    z = pi*complex(r, z.imag)
    n %= 4
    if n == 0: return cmath.sin(z)
    if n == 1: return cmath.cos(z)
    if n == 2: return -cmath.sin(z)
    if n == 3: return -cmath.cos(z)

def _cospi_complex(z):
    if z.real < 0:
        z = -z
    n, r = divmod(z.real, 0.5)
    z = pi*complex(r, z.imag)
    n %= 4
    if n == 0: return cmath.cos(z)
    if n == 1: return -cmath.sin(z)
    if n == 2: return -cmath.cos(z)
    if n == 3: return cmath.sin(z)

cospi = _mathfun_real(_cospi_real, _cospi_complex)
sinpi = _mathfun_real(_sinpi_real, _sinpi_complex)

def tanpi(x):
    try:
        return sinpi(x) / cospi(x)
    except OverflowError:
        if complex(x).imag > 10:
            return 1j
        if complex(x).imag < 10:
            return -1j
        raise

def cotpi(x):
    try:
        return cospi(x) / sinpi(x)
    except OverflowError:
        if complex(x).imag > 10:
            return -1j
        if complex(x).imag < 10:
            return 1j
        raise

INF = 1e300*1e300
NINF = -INF
NAN = INF-INF
EPS = 2.2204460492503131e-16

_exact_gamma = (INF, 1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
  362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
  1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0,
  121645100408832000.0, 2432902008176640000.0)

_max_exact_gamma = len(_exact_gamma)-1

# Lanczos coefficients used by the GNU Scientific Library
_lanczos_g = 7
_lanczos_p = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
     771.32342877765313, -176.61502916214059, 12.507343278686905,
     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)

def _gamma_real(x):
    _intx = int(x)
    if _intx == x:
        if _intx <= 0:
            #return (-1)**_intx * INF
            raise ZeroDivisionError("gamma function pole")
        if _intx <= _max_exact_gamma:
            return _exact_gamma[_intx]
    if x < 0.5:
        # TODO: sinpi
        return pi / (_sinpi_real(x)*_gamma_real(1-x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g+2):
            r += _lanczos_p[i]/(x+i)
        t = x + _lanczos_g + 0.5
        return 2.506628274631000502417 * t**(x+0.5) * math.exp(-t) * r

def _gamma_complex(x):
    if not x.imag:
        return complex(_gamma_real(x.real))
    if x.real < 0.5:
        # TODO: sinpi
        return pi / (_sinpi_complex(x)*_gamma_complex(1-x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g+2):
            r += _lanczos_p[i]/(x+i)
        t = x + _lanczos_g + 0.5
        return 2.506628274631000502417 * t**(x+0.5) * cmath.exp(-t) * r

gamma = _mathfun_real(_gamma_real, _gamma_complex)

def rgamma(x):
    try:
        return 1./gamma(x)
    except ZeroDivisionError:
        return x*0.0

def factorial(x):
    return gamma(x+1.0)

def arg(x):
    if type(x) is float:
        return math.atan2(0.0,x)
    return math.atan2(x.imag,x.real)

# XXX: broken for negatives
def loggamma(x):
    if type(x) not in (float, complex):
        try:
            x = float(x)
        except (ValueError, TypeError):
            x = complex(x)
    try:
        xreal = x.real
        ximag = x.imag
    except AttributeError:   # py2.5
        xreal = x
        ximag = 0.0
    # Reflection formula
    # http://functions.wolfram.com/GammaBetaErf/LogGamma/16/01/01/0003/
    if xreal < 0.0:
        if abs(x) < 0.5:
            v = log(gamma(x))
            if ximag == 0:
                v = v.conjugate()
            return v
        z = 1-x
        try:
            re = z.real
            im = z.imag
        except AttributeError:   # py2.5
            re = z
            im = 0.0
        refloor = floor(re)
        if im == 0.0:
            imsign = 0
        elif im < 0.0:
            imsign = -1
        else:
            imsign = 1
        return (-pi*1j)*abs(refloor)*(1-abs(imsign)) + logpi - \
            log(sinpi(z-refloor)) - loggamma(z) + 1j*pi*refloor*imsign
    if x == 1.0 or x == 2.0:
        return x*0
    p = 0.
    while abs(x) < 11:
        p -= log(x)
        x += 1.0
    s = 0.918938533204672742 + (x-0.5)*log(x) - x
    r = 1./x
    r2 = r*r
    s += 0.083333333333333333333*r; r *= r2
    s += -0.0027777777777777777778*r; r *= r2
    s += 0.00079365079365079365079*r; r *= r2
    s += -0.0005952380952380952381*r; r *= r2
    s += 0.00084175084175084175084*r; r *= r2
    s += -0.0019175269175269175269*r; r *= r2
    s += 0.0064102564102564102564*r; r *= r2
    s += -0.02955065359477124183*r
    return s + p

_psi_coeff = [
0.083333333333333333333,
-0.0083333333333333333333,
0.003968253968253968254,
-0.0041666666666666666667,
0.0075757575757575757576,
-0.021092796092796092796,
0.083333333333333333333,
-0.44325980392156862745,
3.0539543302701197438,
-26.456212121212121212]

def _digamma_real(x):
    _intx = int(x)
    if _intx == x:
        if _intx <= 0:
            raise ZeroDivisionError("polygamma pole")
    if x < 0.5:
        x = 1.0-x
        s = pi*cotpi(x)
    else:
        s = 0.0
    while x < 10.0:
        s -= 1.0/x
        x += 1.0
    x2 = x**-2
    t = x2
    for c in _psi_coeff:
        s -= c*t
        if t < 1e-20:
            break
        t *= x2
    return s + math_log(x) - 0.5/x

def _digamma_complex(x):
    if not x.imag:
        return complex(_digamma_real(x.real))
    if x.real < 0.5:
        x = 1.0-x
        s = pi*cotpi(x)
    else:
        s = 0.0
    while abs(x) < 10.0:
        s -= 1.0/x
        x += 1.0
    x2 = x**-2
    t = x2
    for c in _psi_coeff:
        s -= c*t
        if abs(t) < 1e-20:
            break
        t *= x2
    return s + cmath.log(x) - 0.5/x

digamma = _mathfun_real(_digamma_real, _digamma_complex)

# TODO: could implement complex erf and erfc here. Need
# to find an accurate method (avoiding cancellation)
# for approx. 1 < abs(x) < 9.

_erfc_coeff_P = [
    1.0000000161203922312,
    2.1275306946297962644,
    2.2280433377390253297,
    1.4695509105618423961,
    0.66275911699770787537,
    0.20924776504163751585,
    0.045459713768411264339,
    0.0063065951710717791934,
    0.00044560259661560421715][::-1]

_erfc_coeff_Q = [
    1.0000000000000000000,
    3.2559100272784894318,
    4.9019435608903239131,
    4.4971472894498014205,
    2.7845640601891186528,
    1.2146026030046904138,
    0.37647108453729465912,
    0.080970149639040548613,
    0.011178148899483545902,
    0.00078981003831980423513][::-1]

def _polyval(coeffs, x):
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + x*p
    return p

def _erf_taylor(x):
    # Taylor series assuming 0 <= x <= 1
    x2 = x*x
    s = t = x
    n = 1
    while abs(t) > 1e-17:
        t *= x2/n
        s -= t/(n+n+1)
        n += 1
        t *= x2/n
        s += t/(n+n+1)
        n += 1
    return 1.1283791670955125739*s

def _erfc_mid(x):
    # Rational approximation assuming 0 <= x <= 9
    return exp(-x*x)*_polyval(_erfc_coeff_P,x)/_polyval(_erfc_coeff_Q,x)

def _erfc_asymp(x):
    # Asymptotic expansion assuming x >= 9
    x2 = x*x
    v = exp(-x2)/x*0.56418958354775628695
    r = t = 0.5 / x2
    s = 1.0
    for n in range(1,22,4):
        s -= t
        t *= r * (n+2)
        s += t
        t *= r * (n+4)
        if abs(t) < 1e-17:
            break
    return s * v

def erf(x):
    """
    erf of a real number.
    """
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        return -erf(-x)
    if x >= 1.0:
        if x >= 6.0:
            return 1.0
        return 1.0 - _erfc_mid(x)
    return _erf_taylor(x)

def erfc(x):
    """
    erfc of a real number.
    """
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        if x < -6.0:
            return 2.0
        return 2.0-erfc(-x)
    if x > 9.0:
        return _erfc_asymp(x)
    if x >= 1.0:
        return _erfc_mid(x)
    return 1.0 - _erf_taylor(x)

gauss42 = [\
(0.99839961899006235, 0.0041059986046490839),
(-0.99839961899006235, 0.0041059986046490839),
(0.9915772883408609, 0.009536220301748501),
(-0.9915772883408609,0.009536220301748501),
(0.97934250806374812, 0.014922443697357493),
(-0.97934250806374812, 0.014922443697357493),
(0.96175936533820439,0.020227869569052644),
(-0.96175936533820439, 0.020227869569052644),
(0.93892355735498811, 0.025422959526113047),
(-0.93892355735498811,0.025422959526113047),
(0.91095972490412735, 0.030479240699603467),
(-0.91095972490412735, 0.030479240699603467),
(0.87802056981217269,0.03536907109759211),
(-0.87802056981217269, 0.03536907109759211),
(0.8402859832618168, 0.040065735180692258),
(-0.8402859832618168,0.040065735180692258),
(0.7979620532554873, 0.044543577771965874),
(-0.7979620532554873, 0.044543577771965874),
(0.75127993568948048,0.048778140792803244),
(-0.75127993568948048, 0.048778140792803244),
(0.70049459055617114, 0.052746295699174064),
(-0.70049459055617114,0.052746295699174064),
(0.64588338886924779, 0.056426369358018376),
(-0.64588338886924779, 0.056426369358018376),
(0.58774459748510932, 0.059798262227586649),
(-0.58774459748510932, 0.059798262227586649),
(0.5263957499311922, 0.062843558045002565),
(-0.5263957499311922, 0.062843558045002565),
(0.46217191207042191, 0.065545624364908975),
(-0.46217191207042191, 0.065545624364908975),
(0.39542385204297503, 0.067889703376521934),
(-0.39542385204297503, 0.067889703376521934),
(0.32651612446541151, 0.069862992492594159),
(-0.32651612446541151, 0.069862992492594159),
(0.25582507934287907, 0.071454714265170971),
(-0.25582507934287907, 0.071454714265170971),
(0.18373680656485453, 0.072656175243804091),
(-0.18373680656485453, 0.072656175243804091),
(0.11064502720851986, 0.073460813453467527),
(-0.11064502720851986, 0.073460813453467527),
(0.036948943165351772, 0.073864234232172879),
(-0.036948943165351772, 0.073864234232172879)]

EI_ASYMP_CONVERGENCE_RADIUS = 40.0

def ei_asymp(z, _e1=False):
    r = 1./z
    s = t = 1.0
    k = 1
    while 1:
        t *= k*r
        s += t
        if abs(t) < 1e-16:
            break
        k += 1
    v = s*exp(z)/z
    if _e1:
        if type(z) is complex:
            zreal = z.real
            zimag = z.imag
        else:
            zreal = z
            zimag = 0.0
        if zimag == 0.0 and zreal > 0.0:
            v += pi*1j
    else:
        if type(z) is complex:
            if z.imag > 0:
                v += pi*1j
            if z.imag < 0:
                v -= pi*1j
    return v

def ei_taylor(z, _e1=False):
    s = t = z
    k = 2
    while 1:
        t = t*z/k
        term = t/k
        if abs(term) < 1e-17:
            break
        s += term
        k += 1
    s += euler
    if _e1:
        s += log(-z)
    else:
        if type(z) is float or z.imag == 0.0:
            s += math_log(abs(z))
        else:
            s += cmath.log(z)
    return s

def ei(z, _e1=False):
    typez = type(z)
    if typez not in (float, complex):
        try:
            z = float(z)
            typez = float
        except (TypeError, ValueError):
            z = complex(z)
            typez = complex
    if not z:
        return -INF
    absz = abs(z)
    if absz > EI_ASYMP_CONVERGENCE_RADIUS:
        return ei_asymp(z, _e1)
    elif absz <= 2.0 or (typez is float and z > 0.0):
        return ei_taylor(z, _e1)
    # Integrate, starting from whichever is smaller of a Taylor
    # series value or an asymptotic series value
    if typez is complex and z.real > 0.0:
        zref = z / absz
        ref = ei_taylor(zref, _e1)
    else:
        zref = EI_ASYMP_CONVERGENCE_RADIUS * z / absz
        ref = ei_asymp(zref, _e1)
    C = (zref-z)*0.5
    D = (zref+z)*0.5
    s = 0.0
    if type(z) is complex:
        _exp = cmath.exp
    else:
        _exp = math.exp
    for x,w in gauss42:
        t = C*x+D
        s += w*_exp(t)/t
    ref -= C*s
    return ref

def e1(z):
    # hack to get consistent signs if the imaginary part if 0
    # and signed
    typez = type(z)
    if type(z) not in (float, complex):
        try:
            z = float(z)
            typez = float
        except (TypeError, ValueError):
            z = complex(z)
            typez = complex
    if typez is complex and not z.imag:
        z = complex(z.real, 0.0)
    # end hack
    return -ei(-z, _e1=True)

_zeta_int = [\
-0.5,
0.0,
1.6449340668482264365,1.2020569031595942854,1.0823232337111381915,
1.0369277551433699263,1.0173430619844491397,1.0083492773819228268,
1.0040773561979443394,1.0020083928260822144,1.0009945751278180853,
1.0004941886041194646,1.0002460865533080483,1.0001227133475784891,
1.0000612481350587048,1.0000305882363070205,1.0000152822594086519,
1.0000076371976378998,1.0000038172932649998,1.0000019082127165539,
1.0000009539620338728,1.0000004769329867878,1.0000002384505027277,
1.0000001192199259653,1.0000000596081890513,1.0000000298035035147,
1.0000000149015548284]

_zeta_P = [-3.50000000087575873, -0.701274355654678147,
-0.0672313458590012612, -0.00398731457954257841,
-0.000160948723019303141, -4.67633010038383371e-6,
-1.02078104417700585e-7, -1.68030037095896287e-9,
-1.85231868742346722e-11][::-1]

_zeta_Q = [1.00000000000000000, -0.936552848762465319,
-0.0588835413263763741, -0.00441498861482948666,
-0.000143416758067432622, -5.10691659585090782e-6,
-9.58813053268913799e-8, -1.72963791443181972e-9,
-1.83527919681474132e-11][::-1]

_zeta_1 = [3.03768838606128127e-10, -1.21924525236601262e-8,
2.01201845887608893e-7, -1.53917240683468381e-6,
-5.09890411005967954e-7, 0.000122464707271619326,
-0.000905721539353130232, -0.00239315326074843037,
0.084239750013159168, 0.418938517907442414, 0.500000001921884009]

_zeta_0 = [-3.46092485016748794e-10, -6.42610089468292485e-9,
1.76409071536679773e-7, -1.47141263991560698e-6, -6.38880222546167613e-7,
0.000122641099800668209, -0.000905894913516772796, -0.00239303348507992713,
0.0842396947501199816, 0.418938533204660256, 0.500000000000000052]

def zeta(s):
    """
    Riemann zeta function, real argument
    """
    if not isinstance(s, (float, int)):
        try:
            s = float(s)
        except (ValueError, TypeError):
            try:
                s = complex(s)
                if not s.imag:
                    return complex(zeta(s.real))
            except (ValueError, TypeError):
                pass
            raise NotImplementedError
    if s == 1:
        raise ValueError("zeta(1) pole")
    if s >= 27:
        return 1.0 + 2.0**(-s) + 3.0**(-s)
    n = int(s)
    if n == s:
        if n >= 0:
            return _zeta_int[n]
        if not (n % 2):
            return 0.0
    if s <= 0.0:
        return 2.**s*pi**(s-1)*_sinpi_real(0.5*s)*_gamma_real(1-s)*zeta(1-s)
    if s <= 2.0:
        if s <= 1.0:
            return _polyval(_zeta_0,s)/(s-1)
        return _polyval(_zeta_1,s)/(s-1)
    z = _polyval(_zeta_P,s) / _polyval(_zeta_Q,s)
    return 1.0 + 2.0**(-s) + 3.0**(-s) + 4.0**(-s)*z

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\testing.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: March 6, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Utility classes and functions that make it easy to write :mod:`unittest` compatible test suites.

Over the years I've developed the habit of writing test suites for Python
projects using the :mod:`unittest` module. During those years I've come to know
:pypi:`pytest` and in fact I use :pypi:`pytest` to run my test suites (due to
its much better error reporting) but I've yet to publish a test suite that
*requires* :pypi:`pytest`. I have several reasons for doing so:

- It's nice to keep my test suites as simple and accessible as possible and
  not requiring a specific test runner is part of that attitude.

- Whereas :mod:`unittest` is quite explicit, :pypi:`pytest` contains a lot of
  magic, which kind of contradicts the Python mantra "explicit is better than
  implicit" (IMHO).
"""

# Standard library module
import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest

# Modules included in our package.
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string

# Initialize a logger for this module.
logger = logging.getLogger(__name__)

# A unique object reference used to detect missing attributes.
NOTHING = object()

# Public identifiers that require documentation.
__all__ = (
    'CallableTimedOut',
    'CaptureBuffer',
    'CaptureOutput',
    'ContextManager',
    'CustomSearchPath',
    'MockedProgram',
    'PatchedAttribute',
    'PatchedItem',
    'TemporaryDirectory',
    'TestCase',
    'configure_logging',
    'make_dirs',
    'retry',
    'run_cli',
    'skip_on_raise',
    'touch',
)


def configure_logging(log_level=logging.DEBUG):
    """configure_logging(log_level=logging.DEBUG)
    Automatically configure logging to the terminal.

    :param log_level: The log verbosity (a number, defaults
                      to :mod:`logging.DEBUG <logging>`).

    When :mod:`coloredlogs` is installed :func:`coloredlogs.install()` will be
    used to configure logging to the terminal. When this fails with an
    :exc:`~exceptions.ImportError` then :func:`logging.basicConfig()` is used
    as a fall back.
    """
    try:
        import coloredlogs
        coloredlogs.install(level=log_level)
    except ImportError:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')


def make_dirs(pathname):
    """
    Create missing directories.

    :param pathname: The pathname of a directory (a string).
    """
    if not os.path.isdir(pathname):
        os.makedirs(pathname)


def retry(func, timeout=60, exc_type=AssertionError):
    """retry(func, timeout=60, exc_type=AssertionError)
    Retry a function until assertions no longer fail.

    :param func: A callable. When the callable returns
                 :data:`False` it will also be retried.
    :param timeout: The number of seconds after which to abort (a number,
                    defaults to 60).
    :param exc_type: The type of exceptions to retry (defaults
                     to :exc:`~exceptions.AssertionError`).
    :returns: The value returned by `func`.
    :raises: Once the timeout has expired :func:`retry()` will raise the
             previously retried assertion error. When `func` keeps returning
             :data:`False` until `timeout` expires :exc:`CallableTimedOut`
             will be raised.

    This function sleeps between retries to avoid claiming CPU cycles we don't
    need. It starts by sleeping for 0.1 second but adjusts this to one second
    as the number of retries grows.
    """
    pause = 0.1
    timeout += time.time()
    while True:
        try:
            result = func()
            if result is not False:
                return result
        except exc_type:
            if time.time() > timeout:
                raise
        else:
            if time.time() > timeout:
                raise CallableTimedOut()
        time.sleep(pause)
        if pause < 1:
            pause *= 2


def run_cli(entry_point, *arguments, **options):
    """
    Test a command line entry point.

    :param entry_point: The function that implements the command line interface
                        (a callable).
    :param arguments: Any positional arguments (strings) become the command
                      line arguments (:data:`sys.argv` items 1-N).
    :param options: The following keyword arguments are supported:

                    **capture**
                     Whether to use :class:`CaptureOutput`. Defaults
                     to :data:`True` but can be disabled by passing
                     :data:`False` instead.
                    **input**
                     Refer to :class:`CaptureOutput`.
                    **merged**
                     Refer to :class:`CaptureOutput`.
                    **program_name**
                     Used to set :data:`sys.argv` item 0.
    :returns: A tuple with two values:

              1. The return code (an integer).
              2. The captured output (a string).
    """
    # Add the `program_name' option to the arguments.
    arguments = list(arguments)
    arguments.insert(0, options.pop('program_name', sys.executable))
    # Log the command line arguments (and the fact that we're about to call the
    # command line entry point function).
    logger.debug("Calling command line entry point with arguments: %s", arguments)
    # Prepare to capture the return code and output even if the command line
    # interface raises an exception (whether the exception type is SystemExit
    # or something else).
    returncode = 0
    stdout = None
    stderr = None
    try:
        # Temporarily override sys.argv.
        with PatchedAttribute(sys, 'argv', arguments):
            # Manipulate the standard input/output/error streams?
            options['enabled'] = options.pop('capture', True)
            with CaptureOutput(**options) as capturer:
                try:
                    # Call the command line interface.
                    entry_point()
                finally:
                    # Get the output even if an exception is raised.
                    stdout = capturer.stdout.getvalue()
                    stderr = capturer.stderr.getvalue()
                    # Reconfigure logging to the terminal because it is very
                    # likely that the entry point function has changed the
                    # configured log level.
                    configure_logging()
    except BaseException as e:
        if isinstance(e, SystemExit):
            logger.debug("Intercepting return code %s from SystemExit exception.", e.code)
            returncode = e.code
        else:
            logger.warning("Defaulting return code to 1 due to raised exception.", exc_info=True)
            returncode = 1
    else:
        logger.debug("Command line entry point returned successfully!")
    # Always log the output captured on stdout/stderr, to make it easier to
    # diagnose test failures (but avoid duplicate logging when merged=True).
    is_merged = options.get('merged', False)
    merged_streams = [('merged streams', stdout)]
    separate_streams = [('stdout', stdout), ('stderr', stderr)]
    streams = merged_streams if is_merged else separate_streams
    for name, value in streams:
        if value:
            logger.debug("Output on %s:\n%s", name, value)
        else:
            logger.debug("No output on %s.", name)
    return returncode, stdout


def skip_on_raise(*exc_types):
    """
    Decorate a test function to translation specific exception types to :exc:`unittest.SkipTest`.

    :param exc_types: One or more positional arguments give the exception
                      types to be translated to :exc:`unittest.SkipTest`.
    :returns: A decorator function specialized to `exc_types`.
    """
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kw):
            try:
                return function(*args, **kw)
            except exc_types as e:
                logger.debug("Translating exception to unittest.SkipTest ..", exc_info=True)
                raise unittest.SkipTest("skipping test because %s was raised" % type(e))
        return wrapper
    return decorator


def touch(filename):
    """
    The equivalent of the UNIX :man:`touch` program in Python.

    :param filename: The pathname of the file to touch (a string).

    Note that missing directories are automatically created using
    :func:`make_dirs()`.
    """
    make_dirs(os.path.dirname(filename))
    with open(filename, 'a'):
        os.utime(filename, None)


class CallableTimedOut(Exception):

    """Raised by :func:`retry()` when the timeout expires."""


class ContextManager(object):

    """Base class to enable composition of context managers."""

    def __enter__(self):
        """Enable use as context managers."""
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Enable use as context managers."""


class PatchedAttribute(ContextManager):

    """Context manager that temporary replaces an object attribute using :func:`setattr()`."""

    def __init__(self, obj, name, value):
        """
        Initialize a :class:`PatchedAttribute` object.

        :param obj: The object to patch.
        :param name: An attribute name.
        :param value: The value to set.
        """
        self.object_to_patch = obj
        self.attribute_to_patch = name
        self.patched_value = value
        self.original_value = NOTHING

    def __enter__(self):
        """
        Replace (patch) the attribute.

        :returns: The object whose attribute was patched.
        """
        # Enable composition of context managers.
        super(PatchedAttribute, self).__enter__()
        # Patch the object's attribute.
        self.original_value = getattr(self.object_to_patch, self.attribute_to_patch, NOTHING)
        setattr(self.object_to_patch, self.attribute_to_patch, self.patched_value)
        return self.object_to_patch

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Restore the attribute to its original value."""
        # Enable composition of context managers.
        super(PatchedAttribute, self).__exit__(exc_type, exc_value, traceback)
        # Restore the object's attribute.
        if self.original_value is NOTHING:
            delattr(self.object_to_patch, self.attribute_to_patch)
        else:
            setattr(self.object_to_patch, self.attribute_to_patch, self.original_value)


class PatchedItem(ContextManager):

    """Context manager that temporary replaces an object item using :meth:`~object.__setitem__()`."""

    def __init__(self, obj, item, value):
        """
        Initialize a :class:`PatchedItem` object.

        :param obj: The object to patch.
        :param item: The item to patch.
        :param value: The value to set.
        """
        self.object_to_patch = obj
        self.item_to_patch = item
        self.patched_value = value
        self.original_value = NOTHING

    def __enter__(self):
        """
        Replace (patch) the item.

        :returns: The object whose item was patched.
        """
        # Enable composition of context managers.
        super(PatchedItem, self).__enter__()
        # Patch the object's item.
        try:
            self.original_value = self.object_to_patch[self.item_to_patch]
        except KeyError:
            self.original_value = NOTHING
        self.object_to_patch[self.item_to_patch] = self.patched_value
        return self.object_to_patch

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Restore the item to its original value."""
        # Enable composition of context managers.
        super(PatchedItem, self).__exit__(exc_type, exc_value, traceback)
        # Restore the object's item.
        if self.original_value is NOTHING:
            del self.object_to_patch[self.item_to_patch]
        else:
            self.object_to_patch[self.item_to_patch] = self.original_value


class TemporaryDirectory(ContextManager):

    """
    Easy temporary directory creation & cleanup using the :keyword:`with` statement.

    Here's an example of how to use this:

    .. code-block:: python

       with TemporaryDirectory() as directory:
           # Do something useful here.
           assert os.path.isdir(directory)
    """

    def __init__(self, **options):
        """
        Initialize a :class:`TemporaryDirectory` object.

        :param options: Any keyword arguments are passed on to
                        :func:`tempfile.mkdtemp()`.
        """
        self.mkdtemp_options = options
        self.temporary_directory = None

    def __enter__(self):
        """
        Create the temporary directory using :func:`tempfile.mkdtemp()`.

        :returns: The pathname of the directory (a string).
        """
        # Enable composition of context managers.
        super(TemporaryDirectory, self).__enter__()
        # Create the temporary directory.
        self.temporary_directory = tempfile.mkdtemp(**self.mkdtemp_options)
        return self.temporary_directory

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Cleanup the temporary directory using :func:`shutil.rmtree()`."""
        # Enable composition of context managers.
        super(TemporaryDirectory, self).__exit__(exc_type, exc_value, traceback)
        # Cleanup the temporary directory.
        if self.temporary_directory is not None:
            shutil.rmtree(self.temporary_directory)
            self.temporary_directory = None


class MockedHomeDirectory(PatchedItem, TemporaryDirectory):

    """
    Context manager to temporarily change ``$HOME`` (the current user's profile directory).

    This class is a composition of the :class:`PatchedItem` and
    :class:`TemporaryDirectory` context managers.
    """

    def __init__(self):
        """Initialize a :class:`MockedHomeDirectory` object."""
        PatchedItem.__init__(self, os.environ, 'HOME', os.environ.get('HOME'))
        TemporaryDirectory.__init__(self)

    def __enter__(self):
        """
        Activate the custom ``$PATH``.

        :returns: The pathname of the directory that has
                  been added to ``$PATH`` (a string).
        """
        # Get the temporary directory.
        directory = TemporaryDirectory.__enter__(self)
        # Override the value to patch now that we have
        # the pathname of the temporary directory.
        self.patched_value = directory
        # Temporary patch $HOME.
        PatchedItem.__enter__(self)
        # Pass the pathname of the temporary directory to the caller.
        return directory

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Deactivate the custom ``$HOME``."""
        super(MockedHomeDirectory, self).__exit__(exc_type, exc_value, traceback)


class CustomSearchPath(PatchedItem, TemporaryDirectory):

    """
    Context manager to temporarily customize ``$PATH`` (the executable search path).

    This class is a composition of the :class:`PatchedItem` and
    :class:`TemporaryDirectory` context managers.
    """

    def __init__(self, isolated=False):
        """
        Initialize a :class:`CustomSearchPath` object.

        :param isolated: :data:`True` to clear the original search path,
                         :data:`False` to add the temporary directory to the
                         start of the search path.
        """
        # Initialize our own instance variables.
        self.isolated_search_path = isolated
        # Selectively initialize our superclasses.
        PatchedItem.__init__(self, os.environ, 'PATH', self.current_search_path)
        TemporaryDirectory.__init__(self)

    def __enter__(self):
        """
        Activate the custom ``$PATH``.

        :returns: The pathname of the directory that has
                  been added to ``$PATH`` (a string).
        """
        # Get the temporary directory.
        directory = TemporaryDirectory.__enter__(self)
        # Override the value to patch now that we have
        # the pathname of the temporary directory.
        self.patched_value = (
            directory if self.isolated_search_path
            else os.pathsep.join([directory] + self.current_search_path.split(os.pathsep))
        )
        # Temporary patch the $PATH.
        PatchedItem.__enter__(self)
        # Pass the pathname of the temporary directory to the caller
        # because they may want to `install' custom executables.
        return directory

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Deactivate the custom ``$PATH``."""
        super(CustomSearchPath, self).__exit__(exc_type, exc_value, traceback)

    @property
    def current_search_path(self):
        """The value of ``$PATH`` or :data:`os.defpath` (a string)."""
        return os.environ.get('PATH', os.defpath)


class MockedProgram(CustomSearchPath):

    """
    Context manager to mock the existence of a program (executable).

    This class extends the functionality of :class:`CustomSearchPath`.
    """

    def __init__(self, name, returncode=0, script=None):
        """
        Initialize a :class:`MockedProgram` object.

        :param name: The name of the program (a string).
        :param returncode: The return code that the program should emit (a
                           number, defaults to zero).
        :param script: Shell script code to include in the mocked program (a
                       string or :data:`None`). This can be used to mock a
                       program that is expected to generate specific output.
        """
        # Initialize our own instance variables.
        self.program_name = name
        self.program_returncode = returncode
        self.program_script = script
        self.program_signal_file = None
        # Initialize our superclasses.
        super(MockedProgram, self).__init__()

    def __enter__(self):
        """
        Create the mock program.

        :returns: The pathname of the directory that has
                  been added to ``$PATH`` (a string).
        """
        directory = super(MockedProgram, self).__enter__()
        self.program_signal_file = os.path.join(directory, 'program-was-run-%s' % random_string(10))
        pathname = os.path.join(directory, self.program_name)
        with open(pathname, 'w') as handle:
            handle.write('#!/bin/sh\n')
            handle.write('echo > %s\n' % pipes.quote(self.program_signal_file))
            if self.program_script:
                handle.write('%s\n' % self.program_script.strip())
            handle.write('exit %i\n' % self.program_returncode)
        os.chmod(pathname, 0o755)
        return directory

    def __exit__(self, *args, **kw):
        """
        Ensure that the mock program was run.

        :raises: :exc:`~exceptions.AssertionError` when
                 the mock program hasn't been run.
        """
        try:
            assert self.program_signal_file and os.path.isfile(self.program_signal_file), \
                ("It looks like %r was never run!" % self.program_name)
        finally:
            return super(MockedProgram, self).__exit__(*args, **kw)


class CaptureOutput(ContextManager):

    """
    Context manager that captures what's written to :data:`sys.stdout` and :data:`sys.stderr`.

    .. attribute:: stdin

       The :class:`~humanfriendly.compat.StringIO` object used to feed the standard input stream.

    .. attribute:: stdout

       The :class:`CaptureBuffer` object used to capture the standard output stream.

    .. attribute:: stderr

       The :class:`CaptureBuffer` object used to capture the standard error stream.
    """

    def __init__(self, merged=False, input='', enabled=True):
        """
        Initialize a :class:`CaptureOutput` object.

        :param merged: :data:`True` to merge the streams,
                       :data:`False` to capture them separately.
        :param input: The data that reads from :data:`sys.stdin`
                      should return (a string).
        :param enabled: :data:`True` to enable capturing (the default),
                        :data:`False` otherwise. This makes it easy to
                        unconditionally use :class:`CaptureOutput` in
                        a :keyword:`with` block while preserving the
                        choice to opt out of capturing output.
        """
        self.stdin = StringIO(input)
        self.stdout = CaptureBuffer()
        self.stderr = self.stdout if merged else CaptureBuffer()
        self.patched_attributes = []
        if enabled:
            self.patched_attributes.extend(
                PatchedAttribute(sys, name, getattr(self, name))
                for name in ('stdin', 'stdout', 'stderr')
            )

    def __enter__(self):
        """Start capturing what's written to :data:`sys.stdout` and :data:`sys.stderr`."""
        super(CaptureOutput, self).__enter__()
        for context in self.patched_attributes:
            context.__enter__()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Stop capturing what's written to :data:`sys.stdout` and :data:`sys.stderr`."""
        super(CaptureOutput, self).__exit__(exc_type, exc_value, traceback)
        for context in self.patched_attributes:
            context.__exit__(exc_type, exc_value, traceback)

    def get_lines(self):
        """Get the contents of :attr:`stdout` split into separate lines."""
        return self.get_text().splitlines()

    def get_text(self):
        """Get the contents of :attr:`stdout` as a Unicode string."""
        return self.stdout.get_text()

    def getvalue(self):
        """Get the text written to :data:`sys.stdout`."""
        return self.stdout.getvalue()


class CaptureBuffer(StringIO):

    """
    Helper for :class:`CaptureOutput` to provide an easy to use API.

    The two methods defined by this subclass were specifically chosen to match
    the names of the methods provided by my :pypi:`capturer` package which
    serves a similar role as :class:`CaptureOutput` but knows how to simulate
    an interactive terminal (tty).
    """

    def get_lines(self):
        """Get the contents of the buffer split into separate lines."""
        return self.get_text().splitlines()

    def get_text(self):
        """Get the contents of the buffer as a Unicode string."""
        return self.getvalue()


class TestCase(unittest.TestCase):

    """Subclass of :class:`unittest.TestCase` with automatic logging and other miscellaneous features."""

    def __init__(self, *args, **kw):
        """
        Initialize a :class:`TestCase` object.

        Any positional and/or keyword arguments are passed on to the
        initializer of the superclass.
        """
        super(TestCase, self).__init__(*args, **kw)

    def setUp(self, log_level=logging.DEBUG):
        """setUp(log_level=logging.DEBUG)
        Automatically configure logging to the terminal.

        :param log_level: Refer to :func:`configure_logging()`.

        The :func:`setUp()` method is automatically called by
        :class:`unittest.TestCase` before each test method starts.
        It does two things:

        - Logging to the terminal is configured using
          :func:`configure_logging()`.

        - Before the test method starts a newline is emitted, to separate the
          name of the test method (which will be printed to the terminal by
          :mod:`unittest` or :pypi:`pytest`) from the first line of logging
          output that the test method is likely going to generate.
        """
        # Configure logging to the terminal.
        configure_logging(log_level)
        # Separate the name of the test method (printed by the superclass
        # and/or py.test without a newline at the end) from the first line of
        # logging output that the test method is likely going to generate.
        sys.stderr.write("\n")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\api.py ===
from __future__ import annotations

import logging
from os import PathLike
from typing import BinaryIO

from .cd import (
    coherence_ratio,
    encoding_languages,
    mb_encoding_languages,
    merge_coherence_ratios,
)
from .constant import IANA_SUPPORTED, TOO_BIG_SEQUENCE, TOO_SMALL_SEQUENCE, TRACE
from .md import mess_ratio
from .models import CharsetMatch, CharsetMatches
from .utils import (
    any_specified_encoding,
    cut_sequence_chunks,
    iana_name,
    identify_sig_or_bom,
    is_cp_similar,
    is_multi_byte_encoding,
    should_strip_sig_or_bom,
)

logger = logging.getLogger("charset_normalizer")
explain_handler = logging.StreamHandler()
explain_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)


def from_bytes(
    sequences: bytes | bytearray,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.2,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    """
    Given a raw bytes sequence, return the best possibles charset usable to render str objects.
    If there is no results, it is a strong indicator that the source is binary/not text.
    By default, the process will extract 5 blocks of 512o each to assess the mess and coherence of a given sequence.
    And will give up a particular code page after 20% of measured mess. Those criteria are customizable at will.

    The preemptive behavior DOES NOT replace the traditional detection workflow, it prioritize a particular code page
    but never take it for granted. Can improve the performance.

    You may want to focus your attention to some code page or/and not others, use cp_isolation and cp_exclusion for that
    purpose.

    This function will strip the SIG in the payload/sequence every time except on UTF-16, UTF-32.
    By default the library does not setup any handler other than the NullHandler, if you choose to set the 'explain'
    toggle to True it will alter the logger configuration to add a StreamHandler that is suitable for debugging.
    Custom logging format and handler can be set manually.
    """

    if not isinstance(sequences, (bytearray, bytes)):
        raise TypeError(
            "Expected object of type bytes or bytearray, got: {}".format(
                type(sequences)
            )
        )

    if explain:
        previous_logger_level: int = logger.level
        logger.addHandler(explain_handler)
        logger.setLevel(TRACE)

    length: int = len(sequences)

    if length == 0:
        logger.debug("Encoding detection on empty bytes, assuming utf_8 intention.")
        if explain:  # Defensive: ensure exit path clean handler
            logger.removeHandler(explain_handler)
            logger.setLevel(previous_logger_level or logging.WARNING)
        return CharsetMatches([CharsetMatch(sequences, "utf_8", 0.0, False, [], "")])

    if cp_isolation is not None:
        logger.log(
            TRACE,
            "cp_isolation is set. use this flag for debugging purpose. "
            "limited list of encoding allowed : %s.",
            ", ".join(cp_isolation),
        )
        cp_isolation = [iana_name(cp, False) for cp in cp_isolation]
    else:
        cp_isolation = []

    if cp_exclusion is not None:
        logger.log(
            TRACE,
            "cp_exclusion is set. use this flag for debugging purpose. "
            "limited list of encoding excluded : %s.",
            ", ".join(cp_exclusion),
        )
        cp_exclusion = [iana_name(cp, False) for cp in cp_exclusion]
    else:
        cp_exclusion = []

    if length <= (chunk_size * steps):
        logger.log(
            TRACE,
            "override steps (%i) and chunk_size (%i) as content does not fit (%i byte(s) given) parameters.",
            steps,
            chunk_size,
            length,
        )
        steps = 1
        chunk_size = length

    if steps > 1 and length / steps < chunk_size:
        chunk_size = int(length / steps)

    is_too_small_sequence: bool = len(sequences) < TOO_SMALL_SEQUENCE
    is_too_large_sequence: bool = len(sequences) >= TOO_BIG_SEQUENCE

    if is_too_small_sequence:
        logger.log(
            TRACE,
            "Trying to detect encoding from a tiny portion of ({}) byte(s).".format(
                length
            ),
        )
    elif is_too_large_sequence:
        logger.log(
            TRACE,
            "Using lazy str decoding because the payload is quite large, ({}) byte(s).".format(
                length
            ),
        )

    prioritized_encodings: list[str] = []

    specified_encoding: str | None = (
        any_specified_encoding(sequences) if preemptive_behaviour else None
    )

    if specified_encoding is not None:
        prioritized_encodings.append(specified_encoding)
        logger.log(
            TRACE,
            "Detected declarative mark in sequence. Priority +1 given for %s.",
            specified_encoding,
        )

    tested: set[str] = set()
    tested_but_hard_failure: list[str] = []
    tested_but_soft_failure: list[str] = []

    fallback_ascii: CharsetMatch | None = None
    fallback_u8: CharsetMatch | None = None
    fallback_specified: CharsetMatch | None = None

    results: CharsetMatches = CharsetMatches()

    early_stop_results: CharsetMatches = CharsetMatches()

    sig_encoding, sig_payload = identify_sig_or_bom(sequences)

    if sig_encoding is not None:
        prioritized_encodings.append(sig_encoding)
        logger.log(
            TRACE,
            "Detected a SIG or BOM mark on first %i byte(s). Priority +1 given for %s.",
            len(sig_payload),
            sig_encoding,
        )

    prioritized_encodings.append("ascii")

    if "utf_8" not in prioritized_encodings:
        prioritized_encodings.append("utf_8")

    for encoding_iana in prioritized_encodings + IANA_SUPPORTED:
        if cp_isolation and encoding_iana not in cp_isolation:
            continue

        if cp_exclusion and encoding_iana in cp_exclusion:
            continue

        if encoding_iana in tested:
            continue

        tested.add(encoding_iana)

        decoded_payload: str | None = None
        bom_or_sig_available: bool = sig_encoding == encoding_iana
        strip_sig_or_bom: bool = bom_or_sig_available and should_strip_sig_or_bom(
            encoding_iana
        )

        if encoding_iana in {"utf_16", "utf_32"} and not bom_or_sig_available:
            logger.log(
                TRACE,
                "Encoding %s won't be tested as-is because it require a BOM. Will try some sub-encoder LE/BE.",
                encoding_iana,
            )
            continue
        if encoding_iana in {"utf_7"} and not bom_or_sig_available:
            logger.log(
                TRACE,
                "Encoding %s won't be tested as-is because detection is unreliable without BOM/SIG.",
                encoding_iana,
            )
            continue

        try:
            is_multi_byte_decoder: bool = is_multi_byte_encoding(encoding_iana)
        except (ModuleNotFoundError, ImportError):
            logger.log(
                TRACE,
                "Encoding %s does not provide an IncrementalDecoder",
                encoding_iana,
            )
            continue

        try:
            if is_too_large_sequence and is_multi_byte_decoder is False:
                str(
                    (
                        sequences[: int(50e4)]
                        if strip_sig_or_bom is False
                        else sequences[len(sig_payload) : int(50e4)]
                    ),
                    encoding=encoding_iana,
                )
            else:
                decoded_payload = str(
                    (
                        sequences
                        if strip_sig_or_bom is False
                        else sequences[len(sig_payload) :]
                    ),
                    encoding=encoding_iana,
                )
        except (UnicodeDecodeError, LookupError) as e:
            if not isinstance(e, LookupError):
                logger.log(
                    TRACE,
                    "Code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
            tested_but_hard_failure.append(encoding_iana)
            continue

        similar_soft_failure_test: bool = False

        for encoding_soft_failed in tested_but_soft_failure:
            if is_cp_similar(encoding_iana, encoding_soft_failed):
                similar_soft_failure_test = True
                break

        if similar_soft_failure_test:
            logger.log(
                TRACE,
                "%s is deemed too similar to code page %s and was consider unsuited already. Continuing!",
                encoding_iana,
                encoding_soft_failed,
            )
            continue

        r_ = range(
            0 if not bom_or_sig_available else len(sig_payload),
            length,
            int(length / steps),
        )

        multi_byte_bonus: bool = (
            is_multi_byte_decoder
            and decoded_payload is not None
            and len(decoded_payload) < length
        )

        if multi_byte_bonus:
            logger.log(
                TRACE,
                "Code page %s is a multi byte encoding table and it appear that at least one character "
                "was encoded using n-bytes.",
                encoding_iana,
            )

        max_chunk_gave_up: int = int(len(r_) / 4)

        max_chunk_gave_up = max(max_chunk_gave_up, 2)
        early_stop_count: int = 0
        lazy_str_hard_failure = False

        md_chunks: list[str] = []
        md_ratios = []

        try:
            for chunk in cut_sequence_chunks(
                sequences,
                encoding_iana,
                r_,
                chunk_size,
                bom_or_sig_available,
                strip_sig_or_bom,
                sig_payload,
                is_multi_byte_decoder,
                decoded_payload,
            ):
                md_chunks.append(chunk)

                md_ratios.append(
                    mess_ratio(
                        chunk,
                        threshold,
                        explain is True and 1 <= len(cp_isolation) <= 2,
                    )
                )

                if md_ratios[-1] >= threshold:
                    early_stop_count += 1

                if (early_stop_count >= max_chunk_gave_up) or (
                    bom_or_sig_available and strip_sig_or_bom is False
                ):
                    break
        except (
            UnicodeDecodeError
        ) as e:  # Lazy str loading may have missed something there
            logger.log(
                TRACE,
                "LazyStr Loading: After MD chunk decode, code page %s does not fit given bytes sequence at ALL. %s",
                encoding_iana,
                str(e),
            )
            early_stop_count = max_chunk_gave_up
            lazy_str_hard_failure = True

        # We might want to check the sequence again with the whole content
        # Only if initial MD tests passes
        if (
            not lazy_str_hard_failure
            and is_too_large_sequence
            and not is_multi_byte_decoder
        ):
            try:
                sequences[int(50e3) :].decode(encoding_iana, errors="strict")
            except UnicodeDecodeError as e:
                logger.log(
                    TRACE,
                    "LazyStr Loading: After final lookup, code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
                tested_but_hard_failure.append(encoding_iana)
                continue

        mean_mess_ratio: float = sum(md_ratios) / len(md_ratios) if md_ratios else 0.0
        if mean_mess_ratio >= threshold or early_stop_count >= max_chunk_gave_up:
            tested_but_soft_failure.append(encoding_iana)
            logger.log(
                TRACE,
                "%s was excluded because of initial chaos probing. Gave up %i time(s). "
                "Computed mean chaos is %f %%.",
                encoding_iana,
                early_stop_count,
                round(mean_mess_ratio * 100, ndigits=3),
            )
            # Preparing those fallbacks in case we got nothing.
            if (
                enable_fallback
                and encoding_iana in ["ascii", "utf_8", specified_encoding]
                and not lazy_str_hard_failure
            ):
                fallback_entry = CharsetMatch(
                    sequences,
                    encoding_iana,
                    threshold,
                    False,
                    [],
                    decoded_payload,
                    preemptive_declaration=specified_encoding,
                )
                if encoding_iana == specified_encoding:
                    fallback_specified = fallback_entry
                elif encoding_iana == "ascii":
                    fallback_ascii = fallback_entry
                else:
                    fallback_u8 = fallback_entry
            continue

        logger.log(
            TRACE,
            "%s passed initial chaos probing. Mean measured chaos is %f %%",
            encoding_iana,
            round(mean_mess_ratio * 100, ndigits=3),
        )

        if not is_multi_byte_decoder:
            target_languages: list[str] = encoding_languages(encoding_iana)
        else:
            target_languages = mb_encoding_languages(encoding_iana)

        if target_languages:
            logger.log(
                TRACE,
                "{} should target any language(s) of {}".format(
                    encoding_iana, str(target_languages)
                ),
            )

        cd_ratios = []

        # We shall skip the CD when its about ASCII
        # Most of the time its not relevant to run "language-detection" on it.
        if encoding_iana != "ascii":
            for chunk in md_chunks:
                chunk_languages = coherence_ratio(
                    chunk,
                    language_threshold,
                    ",".join(target_languages) if target_languages else None,
                )

                cd_ratios.append(chunk_languages)

        cd_ratios_merged = merge_coherence_ratios(cd_ratios)

        if cd_ratios_merged:
            logger.log(
                TRACE,
                "We detected language {} using {}".format(
                    cd_ratios_merged, encoding_iana
                ),
            )

        current_match = CharsetMatch(
            sequences,
            encoding_iana,
            mean_mess_ratio,
            bom_or_sig_available,
            cd_ratios_merged,
            (
                decoded_payload
                if (
                    is_too_large_sequence is False
                    or encoding_iana in [specified_encoding, "ascii", "utf_8"]
                )
                else None
            ),
            preemptive_declaration=specified_encoding,
        )

        results.append(current_match)

        if (
            encoding_iana in [specified_encoding, "ascii", "utf_8"]
            and mean_mess_ratio < 0.1
        ):
            # If md says nothing to worry about, then... stop immediately!
            if mean_mess_ratio == 0.0:
                logger.debug(
                    "Encoding detection: %s is most likely the one.",
                    current_match.encoding,
                )
                if explain:  # Defensive: ensure exit path clean handler
                    logger.removeHandler(explain_handler)
                    logger.setLevel(previous_logger_level)
                return CharsetMatches([current_match])

            early_stop_results.append(current_match)

        if (
            len(early_stop_results)
            and (specified_encoding is None or specified_encoding in tested)
            and "ascii" in tested
            and "utf_8" in tested
        ):
            probable_result: CharsetMatch = early_stop_results.best()  # type: ignore[assignment]
            logger.debug(
                "Encoding detection: %s is most likely the one.",
                probable_result.encoding,
            )
            if explain:  # Defensive: ensure exit path clean handler
                logger.removeHandler(explain_handler)
                logger.setLevel(previous_logger_level)

            return CharsetMatches([probable_result])

        if encoding_iana == sig_encoding:
            logger.debug(
                "Encoding detection: %s is most likely the one as we detected a BOM or SIG within "
                "the beginning of the sequence.",
                encoding_iana,
            )
            if explain:  # Defensive: ensure exit path clean handler
                logger.removeHandler(explain_handler)
                logger.setLevel(previous_logger_level)
            return CharsetMatches([results[encoding_iana]])

    if len(results) == 0:
        if fallback_u8 or fallback_ascii or fallback_specified:
            logger.log(
                TRACE,
                "Nothing got out of the detection process. Using ASCII/UTF-8/Specified fallback.",
            )

        if fallback_specified:
            logger.debug(
                "Encoding detection: %s will be used as a fallback match",
                fallback_specified.encoding,
            )
            results.append(fallback_specified)
        elif (
            (fallback_u8 and fallback_ascii is None)
            or (
                fallback_u8
                and fallback_ascii
                and fallback_u8.fingerprint != fallback_ascii.fingerprint
            )
            or (fallback_u8 is not None)
        ):
            logger.debug("Encoding detection: utf_8 will be used as a fallback match")
            results.append(fallback_u8)
        elif fallback_ascii:
            logger.debug("Encoding detection: ascii will be used as a fallback match")
            results.append(fallback_ascii)

    if results:
        logger.debug(
            "Encoding detection: Found %s as plausible (best-candidate) for content. With %i alternatives.",
            results.best().encoding,  # type: ignore
            len(results) - 1,
        )
    else:
        logger.debug("Encoding detection: Unable to determine any suitable charset.")

    if explain:
        logger.removeHandler(explain_handler)
        logger.setLevel(previous_logger_level)

    return results


def from_fp(
    fp: BinaryIO,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    """
    Same thing than the function from_bytes but using a file pointer that is already ready.
    Will not close the file pointer.
    """
    return from_bytes(
        fp.read(),
        steps,
        chunk_size,
        threshold,
        cp_isolation,
        cp_exclusion,
        preemptive_behaviour,
        explain,
        language_threshold,
        enable_fallback,
    )


def from_path(
    path: str | bytes | PathLike,  # type: ignore[type-arg]
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    """
    Same thing than the function from_bytes but with one extra step. Opening and reading given file path in binary mode.
    Can raise IOError.
    """
    with open(path, "rb") as fp:
        return from_fp(
            fp,
            steps,
            chunk_size,
            threshold,
            cp_isolation,
            cp_exclusion,
            preemptive_behaviour,
            explain,
            language_threshold,
            enable_fallback,
        )


def is_binary(
    fp_or_path_or_payload: PathLike | str | BinaryIO | bytes,  # type: ignore[type-arg]
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = False,
) -> bool:
    """
    Detect if the given input (file, bytes, or path) points to a binary file. aka. not a string.
    Based on the same main heuristic algorithms and default kwargs at the sole exception that fallbacks match
    are disabled to be stricter around ASCII-compatible but unlikely to be a string.
    """
    if isinstance(fp_or_path_or_payload, (str, PathLike)):
        guesses = from_path(
            fp_or_path_or_payload,
            steps=steps,
            chunk_size=chunk_size,
            threshold=threshold,
            cp_isolation=cp_isolation,
            cp_exclusion=cp_exclusion,
            preemptive_behaviour=preemptive_behaviour,
            explain=explain,
            language_threshold=language_threshold,
            enable_fallback=enable_fallback,
        )
    elif isinstance(
        fp_or_path_or_payload,
        (
            bytes,
            bytearray,
        ),
    ):
        guesses = from_bytes(
            fp_or_path_or_payload,
            steps=steps,
            chunk_size=chunk_size,
            threshold=threshold,
            cp_isolation=cp_isolation,
            cp_exclusion=cp_exclusion,
            preemptive_behaviour=preemptive_behaviour,
            explain=explain,
            language_threshold=language_threshold,
            enable_fallback=enable_fallback,
        )
    else:
        guesses = from_fp(
            fp_or_path_or_payload,
            steps=steps,
            chunk_size=chunk_size,
            threshold=threshold,
            cp_isolation=cp_isolation,
            cp_exclusion=cp_exclusion,
            preemptive_behaviour=preemptive_behaviour,
            explain=explain,
            language_threshold=language_threshold,
            enable_fallback=enable_fallback,
        )

    return not guesses

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_mha_mmdit.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy as np
from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionMultiHeadAttentionMMDit(Fusion):
    """
    Fuse MultiHeadAttention for Multimodal Diffusion Transformer (MMDiT).
    """

    def __init__(self, model: OnnxModel):
        super().__init__(model, fused_op_type="MultiHeadAttention", search_op_types=["Softmax"])
        self.unsqueeze_update_map = {}

    def get_num_heads(self, start_node: NodeProto, output_name_to_node, input_index=0) -> int:
        """
        Detect num_heads from Reshape & Transpose of q/k/v for both Stable Diffusion 3.x and Flux 1.x:

                MatMul    .. [-1] [24] ..
                 |        |  |  /   /
                Add     Concat(axis=0)
                  |      /
                  Reshape
                     |
                 Transpose(perm=0,1,3,2)
                     |
               (start_node)
        """
        nodes = self.model.match_parent_path(
            start_node, ["Transpose", "Reshape", "Concat"], [input_index, 0, 1], output_name_to_node=output_name_to_node
        )
        if nodes is None:
            return 0

        concat_shape = nodes[-1]
        if len(concat_shape.input) != 4:
            return 0

        value = self.model.get_constant_value(concat_shape.input[2])
        if value is None:
            return 0

        if len(value.shape) != 1:
            return 0

        return int(value[0])

    def get_num_heads_from_k(self, transpose_k: NodeProto, output_name_to_node, concat_before_transpose: bool) -> int:
        """
                Detect num_heads from subgraph like the following (num_heads=24 in this example):
                               MatMu    .. [-1] [24] ..
                                 |       |  |  /   /
                                Add     Concat
                                  |      /
                                 Reshape
                                    |
                             Transpose(perm=0,2,1,3)
                                    |
                             SimplifiedLayerNormalization
                                    |
                            Transpose(perm=0,1,3,2)

                Another variant is to an extra Concat node to join two symmetrical subgraphs:

                           |              |
                          MatMul        MatMul   .. [-1] [24] ..
                           |              |       |  |  /   /
                          Add  Concat    Add      Concat
                            |  /          |      /
                          Reshape         Reshape
                            |              |
                         Transpose     Transpose(perm=0,2,1,3)
                            |              |
        SimplifiedLayerNormalization  SimplifiedLayerNormalization
                                |     /
                               Concat
                                 |
                            Transpose(perm=0,1,3,2)

                    Both patterns are used in stable diffusion 3.5 model.
        """
        if concat_before_transpose:
            nodes = self.model.match_parent_path(
                transpose_k, ["Concat", "SimplifiedLayerNormalization"], [0, 1], output_name_to_node=output_name_to_node
            )
            if nodes:
                return self.get_num_heads(nodes[1], output_name_to_node)
        else:
            nodes = self.model.match_parent_path(
                transpose_k, ["SimplifiedLayerNormalization"], [0], output_name_to_node=output_name_to_node
            )
            if nodes:
                return self.get_num_heads(nodes[0], output_name_to_node)

        return 0

    def reshape_to_3d(self, input_name: str, output_name: str) -> str:
        """Add a Reshape node to convert 4D BxSxNxH to 3D BxSxD.

        Args:
            input_name (str): input name for the 4D tensor of shape BxSxNxH.
            output_name (str): output name for the 3D tensor of shape BxSxD, where D = N * H.

        Returns:
            str: the output name
        """

        new_dims_name = "bsnh_to_bsd_reshape_dims"
        new_dims = self.model.get_initializer(new_dims_name)
        if new_dims is None:
            new_dims = numpy_helper.from_array(np.array([0, 0, -1], dtype="int64"), name=new_dims_name)
            self.model.add_initializer(new_dims, self.this_graph_name)
        reshape_q = helper.make_node(
            "Reshape",
            inputs=[input_name, new_dims_name],
            outputs=[output_name],
            name=self.model.create_node_name("Reshape"),
        )
        self.nodes_to_add.append(reshape_q)
        self.node_name_to_graph_name[reshape_q.name] = self.this_graph_name
        return reshape_q.output[0]

    def adjust_query_from_bnsh_to_bsd_no_concat(self, mul_q: NodeProto, output_name_to_node) -> str | None:
        """
        MultiHeadAttenion requires query in BSD format. This function adjusts query from BNSH to BSD format.

        Before:
                               MatMul
                                 |
                               Add      Concat
                                 |      /
                                 Reshape
                                  |
                               Transpose(perm=0,2,1,3)
                                  |
                       SimplifiedLayerNorm
                                  |
                                 Mul

        After:
                               MatMul
                                 |
                                Add      Concat
                                 |      /
                                 Reshape
                                   |
                           SimplifiedLayerNorm
                                   |
                        Reshape (shape=[0, 0, -1])
        """

        path = self.model.match_parent_path(
            mul_q,
            ["SimplifiedLayerNormalization", "Transpose"],
            [0, 0],
        )
        if path is None:
            return None
        sln_a, transpose_a = path

        if not FusionUtils.check_node_attribute(transpose_a, "perm", [0, 2, 1, 3]):
            return None

        # Update the graph
        sln_a.input[0] = transpose_a.input[0]
        sln_output = sln_a.output[0]
        sln_a.output[0] = sln_output + "_BSNH"

        return self.reshape_to_3d(sln_a.output[0], sln_output + "_BSD")

    def adjust_query_from_bnsh_to_bsd(self, mul_q: NodeProto, output_name_to_node) -> str | None:
        """
        MultiHeadAttenion requires query in BSD format. This function adjusts query from BNSH to BSD format.

            Before:
                      MatMul      MatMul
                        |            |
                        Add Concat  Add    Concat
                         |    /      |      /
                         Reshape     Reshape
                            |           |
        Transpose(perm=0,2,1,3)      Transpose(perm=0,2,1,3)
                            |           |
            SimplifiedLayerNorm  SimplifiedLayerNorm
                            |     /
                            Concat(axis=2)
                             |
                            Mul

            After:
                   MatMul        MatMul
                     |              |
                    Add Concat     Add     Concat
                     |    /         |     /
                     Reshape       Reshape
                        |            |
           SimplifiedLayerNorm  SimplifiedLayerNorm
                        |       /
                      Concat(axis=1)
                         |
                      Reshape (shape=[0, 0, -1])
        """

        path = self.model.match_parent_path(
            mul_q,
            ["Concat", "SimplifiedLayerNormalization", "Transpose"],
            [0, 0, 0],
        )
        if path is None:
            return None
        concat, sln_a, transpose_a = path

        if len(concat.input) != 2:
            return None

        path = self.model.match_parent_path(
            concat,
            ["SimplifiedLayerNormalization", "Transpose"],
            [1, 0],
        )
        if path is None:
            return None
        sln_b, transpose_b = path

        if not FusionUtils.check_node_attribute(transpose_a, "perm", [0, 2, 1, 3]):
            return None

        if not FusionUtils.check_node_attribute(transpose_b, "perm", [0, 2, 1, 3]):
            return None

        if not FusionUtils.check_node_attribute(concat, "axis", 2):
            return None

        # Update the graph
        sln_a.input[0] = transpose_a.input[0]
        sln_b.input[0] = transpose_b.input[0]

        new_concat_node = helper.make_node(
            "Concat",
            inputs=[sln_a.output[0], sln_b.output[0]],
            outputs=[concat.output[0] + "_BSNH"],
            name=self.model.create_node_name("Concat"),
            axis=1,
        )
        self.nodes_to_add.append(new_concat_node)
        self.node_name_to_graph_name[new_concat_node.name] = self.this_graph_name

        return self.reshape_to_3d(new_concat_node.output[0], concat.output[0] + "_BSD")

    def update_unsqueeze_axes_1_to_2(self, unsqueeze: NodeProto) -> str:
        updated_unsqueeze_output = self.unsqueeze_update_map.get(unsqueeze.name)
        if updated_unsqueeze_output is None:
            if len(unsqueeze.input) == 1:
                new_node = helper.make_node(
                    "Unsqueeze",
                    inputs=unsqueeze.input,
                    outputs=[unsqueeze.output[0] + "_BSNH"],
                    name=self.model.create_node_name("Unsqueeze"),
                    axes=[2],
                )
            else:
                initializer_name = "unsqueeze_axes_2"
                if self.model.get_initializer(initializer_name) is None:
                    unsqueeze_axes_2 = helper.make_tensor(
                        name=initializer_name,
                        data_type=TensorProto.INT64,
                        dims=[1],  # Shape of the tensor
                        vals=[2],  # Tensor values
                    )
                    self.model.add_initializer(unsqueeze_axes_2, self.this_graph_name)

                new_node = helper.make_node(
                    "Unsqueeze",
                    inputs=[unsqueeze.input[0], initializer_name],
                    outputs=[unsqueeze.output[0] + "_BSNH"],
                    name=self.model.create_node_name("Unsqueeze"),
                )

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name
            updated_unsqueeze_output = new_node.output[0]
            self.unsqueeze_update_map[unsqueeze.name] = updated_unsqueeze_output

        return updated_unsqueeze_output

    def update_unsqueeze_axes(self, add: NodeProto, output_name_to_node: dict[str, NodeProto]) -> bool:
        """
        Update axes of Unsqueeze from [1] to [2] in the following pattern:
                  Unsqueeze        Unsqueeze
                  (axes=[0])       (axes=[0])
                     |              |
                  Unsqueeze        Unsqueeze
              ... (axes=[1])  ...  (axes=[1])
                |     /        |   /
                   Mul         Mul
                    |       /
                     Add
        Args:
            add (NodeProto): the Add node
            output_name_to_node (Dict[str, NodeProto]): mapping from output name to node

        Returns:
            bool: True if the pattern is matched and updated successfully, False otherwise.
        """
        if len(add.input) != 2:
            return False

        # Check axes of Unsqueeze nodes are [0] and [1], and change to [0] and [2] respectively.
        nodes_b = self.model.match_parent_path(add, ["Mul", "Unsqueeze", "Unsqueeze"], [1, 1, 0], output_name_to_node)
        if nodes_b is None:
            return False

        fusion_utils = FusionUtils(self.model)
        axes_1 = fusion_utils.get_squeeze_or_unsqueeze_axes(nodes_b[1])
        if axes_1 is None or axes_1 != [1]:
            return False

        axes_0 = fusion_utils.get_squeeze_or_unsqueeze_axes(nodes_b[2])
        if axes_0 is None or axes_0 != [0]:
            return False

        # Check axes of Unsqueeze nodes are [0] and [1], and change to [0] and [2] respectively.
        nodes_a = self.model.match_parent_path(add, ["Mul", "Unsqueeze", "Unsqueeze"], [0, 1, 0], output_name_to_node)
        if nodes_a is None:
            return False

        axes_1 = fusion_utils.get_squeeze_or_unsqueeze_axes(nodes_a[1])
        if axes_1 is None or axes_1 != [1]:
            return False

        axes_0 = fusion_utils.get_squeeze_or_unsqueeze_axes(nodes_a[2])
        if axes_0 is None or axes_0 != [0]:
            return False

        nodes_a[0].input[1] = self.update_unsqueeze_axes_1_to_2(nodes_a[1])
        nodes_b[0].input[1] = self.update_unsqueeze_axes_1_to_2(nodes_b[1])
        return True

    def adjust_flux_query_from_bnsh_to_bsd(self, mul_q: NodeProto, output_name_to_node) -> str | None:
        """
        Adjust graph to change query format from BNSH to BSD for Flux model.
        Note that the graph pattern is complex, and we only do a shallow match here.

        Before:
                       |               |
        Transpose(perm=0,2,1,3)    Transpose(perm=0,2,1,3)
                        |              |
        SimplifiedLayerNorm  SimplifiedLayerNorm
                        |             /
                        Concat(axis=2)
                         |
                        Mul     Mul
                         |    /
                          Add
                           |
                          Mul

        After (Transpose nods are removed, and a Reshape is added):

                        |           |
            SimplifiedLayerNorm  SimplifiedLayerNorm
                        |         /
                    Concat(axis=1)
                        |
                        Mul    Mul
                         |    /
                          Add
                           |
                       Reshape (shape=[0, 0, -1])
        """

        path = self.model.match_parent_path(
            mul_q,
            ["Add", "Mul", "Concat", "SimplifiedLayerNormalization", "Transpose"],
            [0, 0, 0, 0, 0],
        )
        if path is None:
            return None
        add, _mul_a, concat, sln_a, transpose_a = path

        if len(concat.input) != 2:
            return None

        path = self.model.match_parent_path(
            concat,
            ["SimplifiedLayerNormalization", "Transpose"],
            [1, 0],
        )
        if path is None:
            return None
        sln_b, transpose_b = path

        if not FusionUtils.check_node_attribute(transpose_a, "perm", [0, 2, 1, 3]):
            return None

        if not FusionUtils.check_node_attribute(transpose_b, "perm", [0, 2, 1, 3]):
            return None

        if not FusionUtils.check_node_attribute(concat, "axis", 2):
            return None

        # Need adjust axes of Unsqueeze nodes from [1] to [2] so that the tensors to Mul nodes are BSNH instead of BNSH.
        if not self.update_unsqueeze_axes(add, output_name_to_node):
            return None

        # Update the graph
        sln_a.input[0] = transpose_a.input[0]
        sln_b.input[0] = transpose_b.input[0]

        new_concat_node = helper.make_node(
            "Concat",
            inputs=[sln_a.output[0], sln_b.output[0]],
            outputs=[concat.output[0] + "_BSNH"],
            name=self.model.create_node_name("Concat"),
            axis=1,
        )
        self.nodes_to_add.append(new_concat_node)
        self.node_name_to_graph_name[new_concat_node.name] = self.this_graph_name
        self.model.replace_input_of_all_nodes(concat.output[0], new_concat_node.output[0])

        return self.reshape_to_3d(add.output[0], add.output[0] + "_BSD")

    def adjust_flux_single_query_from_bnsh_to_bsd(self, mul_q: NodeProto, output_name_to_node) -> str | None:
        """
        Adjust graph to change query format from BNSH to BSD for Flux model.
        Note that the graph pattern is complex, and we only do a shallow match here.

        Before:
                      |
                    Transpose(perm=0,2,1,3)
                      |
                    SimplifiedLayerNorm
                      |
                     Mul     Mul
                       |   /
                       Add
                        |
                       Mul

        After (Transpose is removed, and a Reshape is added):

                        |
                      SimplifiedLayerNorm
                        |
                        Mul   Mul
                         |   /
                         Add
                          |
                       Reshape (shape=[0, 0, -1])
        """

        path = self.model.match_parent_path(
            mul_q,
            ["Add", "Mul", "SimplifiedLayerNormalization", "Transpose"],
            [0, 0, 0, 0],
        )
        if path is None:
            return None
        add, _mul_a, sln_a, transpose_a = path

        if not FusionUtils.check_node_attribute(transpose_a, "perm", [0, 2, 1, 3]):
            return None

        # Need adjust axes of Unsqueeze nodes from [1] to [2] so that the tensors to Mul nodes are BSNH instead of BNSH.
        if not self.update_unsqueeze_axes(add, output_name_to_node):
            return None

        # Update the graph
        sln_a.input[0] = transpose_a.input[0]
        add.output[0] = add.output[0] + "_BSNH"

        return self.reshape_to_3d(add.output[0], add.output[0] + "_BSD")

    def transpose_reshape_bnsh_to_bsd(self, q: str, output_name_to_node) -> str | None:
        transpose_q = helper.make_node(
            "Transpose",
            [q],
            [q + "_BSNH"],
            name=self.model.create_node_name("Transpose", name_prefix="Transpose_BNSH_to_BSNH"),
            perm=[0, 2, 1, 3],
        )
        self.nodes_to_add.append(transpose_q)
        self.node_name_to_graph_name[transpose_q.name] = self.this_graph_name

        return self.reshape_to_3d(q + "_BSNH", q + "_BSD")

    def create_multihead_attention_node(
        self,
        q: str,
        k: str,
        v: str,
        output: str,
        num_heads: int,
    ) -> NodeProto:
        """
        Create a MultiHeadAttention node.

        Args:
            q (str): name of q
            k (str): name of k
            v (str): name of v
            output (str): output name of MHA
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.

        Returns:
            NodeProto: the node created.
        """

        assert num_heads > 0

        # Add inputs for MHA: Query, Key, Value (Proj_Bias, Mask, Attention_Bias, Past_K, Past_V are optional)
        mha_inputs = [q, k, v]

        # Add outputs for MHA (Present_K, Present_V are optional)
        mha_outputs = [output]

        mha_node = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=mha_outputs,
            name=self.model.create_node_name("MultiHeadAttention"),
        )

        mha_node.domain = "com.microsoft"
        mha_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        # No mask is used in MMDit model, so we need not set the optional mask_filter_value attribute.
        return mha_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        assert node.op_type == "Softmax"
        softmax = node

        # Softmax output shall not be graph output.
        if self.model.find_graph_output(softmax.output[0]):
            return

        nodes = self.model.match_child_path(
            softmax, ["MatMul", "Transpose", "Reshape"], [(0, 0), (0, 0), (0, 0)], input_name_to_nodes
        )
        if nodes is None:
            return

        matmul_s_v, transpose_out, reshape_out = nodes
        if not FusionUtils.check_node_attribute(transpose_out, "perm", [0, 2, 1, 3]):
            return

        q_nodes = self.model.match_parent_path(
            softmax,
            ["MatMul", "Mul", "Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape"],
            [0, 0, 1, 0, 1, 0, 0, 0],
        )

        if q_nodes is None:
            return

        matmul_qk, mul_q, sqrt_q_2, div_q, sqrt_q, _, _, shape_q = q_nodes

        q_bnsh = mul_q.input[0]
        if q_bnsh != shape_q.input[0]:
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose"], [1, 0])
        if k_nodes is None:
            return

        mul_k, transpose_k = k_nodes
        k = transpose_k.input[0]
        if not FusionUtils.check_node_attribute(transpose_k, "perm", [0, 1, 3, 2]):
            return

        k_scale_nodes = self.model.match_parent_path(mul_k, ["Sqrt", "Div"], [1, 0])
        if k_scale_nodes is None:
            return
        if k_scale_nodes[0].input[0] != sqrt_q_2.input[0]:
            return

        v = matmul_s_v.input[1]

        # Here we sanity check the v path to make sure it is in the expected BNSH format.
        concat_v = self.model.match_parent(matmul_s_v, "Concat", input_index=1, output_name_to_node=output_name_to_node)
        if concat_v is not None:
            # Match v path like:
            #   -- Transpose (perm=[0,2,1,3]) ----+
            #                                     |
            #                                     v
            #   -- Transpose (perm=[0,2,1,3]) -> Concat -> (v)
            transpose_1 = self.model.match_parent(
                concat_v, "Transpose", input_index=0, output_name_to_node=output_name_to_node
            )
            if transpose_1 is None:
                return
            if not FusionUtils.check_node_attribute(transpose_1, "perm", [0, 2, 1, 3]):
                return

            transpose_2 = self.model.match_parent(
                concat_v, "Transpose", input_index=1, output_name_to_node=output_name_to_node
            )
            if transpose_2 is None:
                return
            if not FusionUtils.check_node_attribute(transpose_2, "perm", [0, 2, 1, 3]):
                return
        else:
            # Match v path like:
            #   -- Transpose (perm=[0,2,1,3]) -> (v)
            transpose_1 = self.model.match_parent(
                matmul_s_v, "Transpose", input_index=1, output_name_to_node=output_name_to_node
            )
            if transpose_1 is None:
                return
            if not FusionUtils.check_node_attribute(transpose_1, "perm", [0, 2, 1, 3]):
                return

        # Match patterns for Flux.
        num_heads = (
            self.get_num_heads(concat_v, output_name_to_node)
            if concat_v
            else self.get_num_heads(matmul_s_v, output_name_to_node, input_index=1)
        )

        if num_heads == 0:
            # Match patterns for Stable Diffusion 3.5.
            num_heads = self.get_num_heads_from_k(transpose_k, output_name_to_node, concat_v is not None)
            if num_heads <= 0:
                return

        # Q is in BNSH format, we need to adjust it to BSD format due to limitation of MHA op.
        # TODO: MHA op support BNSH format to reduce the effort in fusion.
        if concat_v is not None:
            query = self.adjust_query_from_bnsh_to_bsd(mul_q, output_name_to_node)
        else:
            query = self.adjust_query_from_bnsh_to_bsd_no_concat(mul_q, output_name_to_node)

        if query is None:
            query = self.adjust_flux_query_from_bnsh_to_bsd(mul_q, output_name_to_node)
            if query is None:
                query = self.adjust_flux_single_query_from_bnsh_to_bsd(mul_q, output_name_to_node)
                if query is None:
                    # fallback to use Transpose and Add to adjust query from BNSH to BSD
                    # This is more general approach.
                    # However, it might be slower if the extra Transpose node cannot be removed by ORT optimizer.
                    query = self.transpose_reshape_bnsh_to_bsd(q_bnsh, output_name_to_node)

        new_node = self.create_multihead_attention_node(
            q=query,
            k=k,
            v=v,
            output=reshape_out.output[0],
            num_heads=num_heads,
        )
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([matmul_s_v, transpose_out, reshape_out])

        # Use prune graph to remove nodes
        self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\pytables.py ===
""" manage PyTables query interface via Expressions """
from __future__ import annotations

import ast
from decimal import (
    Decimal,
    InvalidOperation,
)
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

import numpy as np

from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)
from pandas.errors import UndefinedVariableError

from pandas.core.dtypes.common import is_list_like

import pandas.core.common as com
from pandas.core.computation import (
    expr,
    ops,
    scope as _scope,
)
from pandas.core.computation.common import ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor
from pandas.core.computation.ops import is_term
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index

from pandas.io.formats.printing import (
    pprint_thing,
    pprint_thing_encoded,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Self,
        npt,
    )


class PyTablesScope(_scope.Scope):
    __slots__ = ("queryables",)

    queryables: dict[str, Any]

    def __init__(
        self,
        level: int,
        global_dict=None,
        local_dict=None,
        queryables: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}


class Term(ops.Term):
    env: PyTablesScope

    def __new__(cls, name, env, side=None, encoding=None):
        if isinstance(name, str):
            klass = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        # must be a queryables
        if self.side == "left":
            # Note: The behavior of __new__ ensures that self.name is a str here
            if self.name not in self.env.queryables:
                raise NameError(f"name {repr(self.name)} is not defined")
            return self.name

        # resolve the rhs (and allow it to be None)
        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    # read-only property overwriting read/write property
    @property  # type: ignore[misc]
    def value(self):
        return self._value


class Constant(Term):
    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        return self._name


class BinOp(ops.BinOp):
    _max_selectors = 31

    op: str
    queryables: dict[str, Any]
    condition: str | None

    def __init__(self, op: str, lhs, rhs, queryables: dict[str, Any], encoding) -> None:
        super().__init__(op, lhs, rhs)
        self.queryables = queryables
        self.encoding = encoding
        self.condition = None

    def _disallow_scalar_only_bool_ops(self) -> None:
        pass

    def prune(self, klass):
        def pr(left, right):
            """create and return a new specialized BinOp from myself"""
            if left is None:
                return right
            elif right is None:
                return left

            k = klass
            if isinstance(left, ConditionBinOp):
                if isinstance(right, ConditionBinOp):
                    k = JointConditionBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right

            elif isinstance(left, FilterBinOp):
                if isinstance(right, FilterBinOp):
                    k = JointFilterBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right

            return k(
                self.op, left, right, queryables=self.queryables, encoding=self.encoding
            ).evaluate()

        left, right = self.lhs, self.rhs

        if is_term(left) and is_term(right):
            res = pr(left.value, right.value)
        elif not is_term(left) and is_term(right):
            res = pr(left.prune(klass), right.value)
        elif is_term(left) and not is_term(right):
            res = pr(left.value, right.prune(klass))
        elif not (is_term(left) or is_term(right)):
            res = pr(left.prune(klass), right.prune(klass))

        return res

    def conform(self, rhs):
        """inplace conform rhs"""
        if not is_list_like(rhs):
            rhs = [rhs]
        if isinstance(rhs, np.ndarray):
            rhs = rhs.ravel()
        return rhs

    @property
    def is_valid(self) -> bool:
        """return True if this is a valid field"""
        return self.lhs in self.queryables

    @property
    def is_in_table(self) -> bool:
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
        return self.queryables.get(self.lhs) is not None

    @property
    def kind(self):
        """the kind of my field"""
        return getattr(self.queryables.get(self.lhs), "kind", None)

    @property
    def meta(self):
        """the meta of my field"""
        return getattr(self.queryables.get(self.lhs), "meta", None)

    @property
    def metadata(self):
        """the metadata of my field"""
        return getattr(self.queryables.get(self.lhs), "metadata", None)

    def generate(self, v) -> str:
        """create and return the op string for this TermValue"""
        val = v.tostring(self.encoding)
        return f"({self.lhs} {self.op} {val})"

    def convert_value(self, v) -> TermValue:
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """

        def stringify(value):
            if self.encoding is not None:
                return pprint_thing_encoded(value, encoding=self.encoding)
            return pprint_thing(value)

        kind = ensure_decoded(self.kind)
        meta = ensure_decoded(self.meta)
        if kind == "datetime" or (kind and kind.startswith("datetime64")):
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = ensure_decoded(v)
            v = Timestamp(v).as_unit("ns")
            if v.tz is not None:
                v = v.tz_convert("UTC")
            return TermValue(v, v._value, kind)
        elif kind in ("timedelta64", "timedelta"):
            if isinstance(v, str):
                v = Timedelta(v)
            else:
                v = Timedelta(v, unit="s")
            v = v.as_unit("ns")._value
            return TermValue(int(v), v, kind)
        elif meta == "category":
            metadata = extract_array(self.metadata, extract_numpy=True)
            result: npt.NDArray[np.intp] | np.intp | int
            if v not in metadata:
                result = -1
            else:
                result = metadata.searchsorted(v, side="left")
            return TermValue(result, result, "integer")
        elif kind == "integer":
            try:
                v_dec = Decimal(v)
            except InvalidOperation:
                # GH 54186
                # convert v to float to raise float's ValueError
                float(v)
            else:
                v = int(v_dec.to_integral_exact(rounding="ROUND_HALF_EVEN"))
            return TermValue(v, v, kind)
        elif kind == "float":
            v = float(v)
            return TermValue(v, v, kind)
        elif kind == "bool":
            if isinstance(v, str):
                v = v.strip().lower() not in [
                    "false",
                    "f",
                    "no",
                    "n",
                    "none",
                    "0",
                    "[]",
                    "{}",
                    "",
                ]
            else:
                v = bool(v)
            return TermValue(v, v, kind)
        elif isinstance(v, str):
            # string quoting
            return TermValue(v, stringify(v), "string")
        else:
            raise TypeError(f"Cannot compare {v} of type {type(v)} to {kind} column")

    def convert_values(self) -> None:
        pass


class FilterBinOp(BinOp):
    filter: tuple[Any, Any, Index] | None = None

    def __repr__(self) -> str:
        if self.filter is None:
            return "Filter: Not Initialized"
        return pprint_thing(f"[Filter : [{self.filter[0]}] -> [{self.filter[1]}]")

    def invert(self) -> Self:
        """invert the filter"""
        if self.filter is not None:
            self.filter = (
                self.filter[0],
                self.generate_filter_op(invert=True),
                self.filter[2],
            )
        return self

    def format(self):
        """return the actual filter format"""
        return [self.filter]

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        rhs = self.conform(self.rhs)
        values = list(rhs)

        if self.is_in_table:
            # if too many values to create the expression, use a filter instead
            if self.op in ["==", "!="] and len(values) > self._max_selectors:
                filter_op = self.generate_filter_op()
                self.filter = (self.lhs, filter_op, Index(values))

                return self
            return None

        # equality conditions
        if self.op in ["==", "!="]:
            filter_op = self.generate_filter_op()
            self.filter = (self.lhs, filter_op, Index(values))

        else:
            raise TypeError(
                f"passing a filterable condition to a non-table indexer [{self}]"
            )

        return self

    def generate_filter_op(self, invert: bool = False):
        if (self.op == "!=" and not invert) or (self.op == "==" and invert):
            return lambda axis, vals: ~axis.isin(vals)
        else:
            return lambda axis, vals: axis.isin(vals)


class JointFilterBinOp(FilterBinOp):
    def format(self):
        raise NotImplementedError("unable to collapse Joint Filters")

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self:  # type: ignore[override]
        return self


class ConditionBinOp(BinOp):
    def __repr__(self) -> str:
        return pprint_thing(f"[Condition : [{self.condition}]]")

    def invert(self):
        """invert the condition"""
        # if self.condition is not None:
        #    self.condition = "~(%s)" % self.condition
        # return self
        raise NotImplementedError(
            "cannot use an invert condition when passing to numexpr"
        )

    def format(self):
        """return the actual ne format"""
        return self.condition

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        # convert values if we are in the table
        if not self.is_in_table:
            return None

        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]

        # equality conditions
        if self.op in ["==", "!="]:
            # too many values to create the expression?
            if len(values) <= self._max_selectors:
                vs = [self.generate(v) for v in values]
                self.condition = f"({' | '.join(vs)})"

            # use a filter after reading
            else:
                return None
        else:
            self.condition = self.generate(values[0])

        return self


class JointConditionBinOp(ConditionBinOp):
    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self:  # type: ignore[override]
        self.condition = f"({self.lhs.condition} {self.op} {self.rhs.condition})"
        return self


class UnaryOp(ops.UnaryOp):
    def prune(self, klass):
        if self.op != "~":
            raise NotImplementedError("UnaryOp only support invert type ops")

        operand = self.operand
        operand = operand.prune(klass)

        if operand is not None and (
            issubclass(klass, ConditionBinOp)
            and operand.condition is not None
            or not issubclass(klass, ConditionBinOp)
            and issubclass(klass, FilterBinOp)
            and operand.filter is not None
        ):
            return operand.invert()
        return None


class PyTablesExprVisitor(BaseExprVisitor):
    const_type: ClassVar[type[ops.Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    def __init__(self, env, engine, parser, **kwargs) -> None:
        super().__init__(env, engine, parser)
        for bin_op in self.binary_ops:
            bin_node = self.binary_op_nodes_map[bin_op]
            setattr(
                self,
                f"visit_{bin_node}",
                lambda node, bin_op=bin_op: partial(BinOp, bin_op, **kwargs),
            )

    def visit_UnaryOp(self, node, **kwargs) -> ops.Term | UnaryOp | None:
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return UnaryOp("~", self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            return self.const_type(-self.visit(node.operand).value, self.env)
        elif isinstance(node.op, ast.UAdd):
            raise NotImplementedError("Unary addition not supported")
        # TODO: return None might never be reached
        return None

    def visit_Index(self, node, **kwargs):
        return self.visit(node.value).value

    def visit_Assign(self, node, **kwargs):
        cmpr = ast.Compare(
            ops=[ast.Eq()], left=node.targets[0], comparators=[node.value]
        )
        return self.visit(cmpr)

    def visit_Subscript(self, node, **kwargs) -> ops.Term:
        # only allow simple subscripts

        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        try:
            value = value.value
        except AttributeError:
            pass

        if isinstance(slobj, Term):
            # In py39 np.ndarray lookups with Term containing int raise
            slobj = slobj.value

        try:
            return self.const_type(value[slobj], self.env)
        except TypeError as err:
            raise ValueError(
                f"cannot subscript {repr(value)} with {repr(slobj)}"
            ) from err

    def visit_Attribute(self, node, **kwargs):
        attr = node.attr
        value = node.value

        ctx = type(node.ctx)
        if ctx == ast.Load:
            # resolve the value
            resolved = self.visit(value)

            # try to get the value to see if we are another expression
            try:
                resolved = resolved.value
            except AttributeError:
                pass

            try:
                return self.term_type(getattr(resolved, attr), self.env)
            except AttributeError:
                # something like datetime.datetime where scope is overridden
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved

        raise ValueError(f"Invalid Attribute context {ctx.__name__}")

    def translate_In(self, op):
        return ast.Eq() if isinstance(op, ast.In) else op

    def _rewrite_membership_op(self, node, left, right):
        return self.visit(node.op), node.op, left, right


def _validate_where(w):
    """
    Validate that the where statement is of the right type.

    The type may either be String, Expr, or list-like of Exprs.

    Parameters
    ----------
    w : String term expression, Expr, or list-like of Exprs.

    Returns
    -------
    where : The original where clause if the check was successful.

    Raises
    ------
    TypeError : An invalid data type was passed in for w (e.g. dict).
    """
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        raise TypeError(
            "where must be passed as a string, PyTablesExpr, "
            "or list-like of PyTablesExpr"
        )

    return w


class PyTablesExpr(expr.Expr):
    """
    Hold a pytables-like expression, comprised of possibly multiple 'terms'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    'index>=date'
    "columns=['A', 'D']"
    'columns=A'
    'columns==A'
    "~(columns=['A','B'])"
    'index>df.index[3] & string="bar"'
    '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    "ts>=Timestamp('2012-02-01')"
    "major_axis>=20130101"
    """

    _visitor: PyTablesExprVisitor | None
    env: PyTablesScope
    expr: str

    def __init__(
        self,
        where,
        queryables: dict[str, Any] | None = None,
        encoding=None,
        scope_level: int = 0,
    ) -> None:
        where = _validate_where(where)

        self.encoding = encoding
        self.condition = None
        self.filter = None
        self.terms = None
        self._visitor = None

        # capture the environment if needed
        local_dict: _scope.DeepChainMap[Any, Any] | None = None

        if isinstance(where, PyTablesExpr):
            local_dict = where.env.scope
            _where = where.expr

        elif is_list_like(where):
            where = list(where)
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    local_dict = w.env.scope
                else:
                    where[idx] = _validate_where(w)
            _where = " & ".join([f"({w})" for w in com.flatten(where)])
        else:
            # _validate_where ensures we otherwise have a string
            _where = where

        self.expr = _where
        self.env = PyTablesScope(scope_level + 1, local_dict=local_dict)

        if queryables is not None and isinstance(self.expr, str):
            self.env.queryables.update(queryables)
            self._visitor = PyTablesExprVisitor(
                self.env,
                queryables=queryables,
                parser="pytables",
                engine="pytables",
                encoding=encoding,
            )
            self.terms = self.parse()

    def __repr__(self) -> str:
        if self.terms is not None:
            return pprint_thing(self.terms)
        return pprint_thing(self.expr)

    def evaluate(self):
        """create and return the numexpr condition and filter"""
        try:
            self.condition = self.terms.prune(ConditionBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid condition"
            ) from err
        try:
            self.filter = self.terms.prune(FilterBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid filter"
            ) from err

        return self.condition, self.filter


class TermValue:
    """hold a term value the we use to construct a condition/filter"""

    def __init__(self, value, converted, kind: str) -> None:
        assert isinstance(kind, str), kind
        self.value = value
        self.converted = converted
        self.kind = kind

    def tostring(self, encoding) -> str:
        """quote the string if not encoded else encode and return"""
        if self.kind == "string":
            if encoding is not None:
                return str(self.converted)
            return f'"{self.converted}"'
        elif self.kind == "float":
            # python 2 str(float) is not always
            # round-trippable so use repr()
            return repr(self.converted)
        return str(self.converted)


def maybe_expression(s) -> bool:
    """loose checking if s is a pytables-acceptable expression"""
    if not isinstance(s, str):
        return False
    operations = PyTablesExprVisitor.binary_ops + PyTablesExprVisitor.unary_ops + ("=",)

    # make sure we have an op at least
    return any(op in s for op in operations)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\pg8000.py ===
# dialects/postgresql/pg8000.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors <see AUTHORS
# file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r"""
.. dialect:: postgresql+pg8000
    :name: pg8000
    :dbapi: pg8000
    :connectstring: postgresql+pg8000://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://pypi.org/project/pg8000/

.. versionchanged:: 1.4  The pg8000 dialect has been updated for version
   1.16.6 and higher, and is again part of SQLAlchemy's continuous integration
   with full feature support.

.. _pg8000_unicode:

Unicode
-------

pg8000 will encode / decode string values between it and the server using the
PostgreSQL ``client_encoding`` parameter; by default this is the value in
the ``postgresql.conf`` file, which often defaults to ``SQL_ASCII``.
Typically, this can be changed to ``utf-8``, as a more useful default::

    # client_encoding = sql_ascii # actually, defaults to database encoding
    client_encoding = utf8

The ``client_encoding`` can be overridden for a session by executing the SQL:

.. sourcecode:: sql

    SET CLIENT_ENCODING TO 'utf8';

SQLAlchemy will execute this SQL on all new connections based on the value
passed to :func:`_sa.create_engine` using the ``client_encoding`` parameter::

    engine = create_engine(
        "postgresql+pg8000://user:pass@host/dbname", client_encoding="utf8"
    )

.. _pg8000_ssl:

SSL Connections
---------------

pg8000 accepts a Python ``SSLContext`` object which may be specified using the
:paramref:`_sa.create_engine.connect_args` dictionary::

    import ssl

    ssl_context = ssl.create_default_context()
    engine = sa.create_engine(
        "postgresql+pg8000://scott:tiger@192.168.0.199/test",
        connect_args={"ssl_context": ssl_context},
    )

If the server uses an automatically-generated certificate that is self-signed
or does not match the host name (as seen from the client), it may also be
necessary to disable hostname checking::

    import ssl

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    engine = sa.create_engine(
        "postgresql+pg8000://scott:tiger@192.168.0.199/test",
        connect_args={"ssl_context": ssl_context},
    )

.. _pg8000_isolation_level:

pg8000 Transaction Isolation Level
-------------------------------------

The pg8000 dialect offers the same isolation level settings as that
of the :ref:`psycopg2 <psycopg2_isolation_level>` dialect:

* ``READ COMMITTED``
* ``READ UNCOMMITTED``
* ``REPEATABLE READ``
* ``SERIALIZABLE``
* ``AUTOCOMMIT``

.. seealso::

    :ref:`postgresql_isolation_level`

    :ref:`psycopg2_isolation_level`


"""  # noqa
import decimal
import re

from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .pg_catalog import _SpaceVector
from .pg_catalog import OIDVECTOR
from .types import CITEXT
from ... import exc
from ... import util
from ...engine import processors
from ...sql import sqltypes
from ...sql.elements import quoted_name


class _PGString(sqltypes.String):
    render_bind_cast = True


class _PGNumeric(sqltypes.Numeric):
    render_bind_cast = True

    def result_processor(self, dialect, coltype):
        if self.asdecimal:
            if coltype in _FLOAT_TYPES:
                return processors.to_decimal_processor_factory(
                    decimal.Decimal, self._effective_decimal_return_scale
                )
            elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
                # pg8000 returns Decimal natively for 1700
                return None
            else:
                raise exc.InvalidRequestError(
                    "Unknown PG numeric type: %d" % coltype
                )
        else:
            if coltype in _FLOAT_TYPES:
                # pg8000 returns float natively for 701
                return None
            elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
                return processors.to_float
            else:
                raise exc.InvalidRequestError(
                    "Unknown PG numeric type: %d" % coltype
                )


class _PGFloat(_PGNumeric, sqltypes.Float):
    __visit_name__ = "float"
    render_bind_cast = True


class _PGNumericNoBind(_PGNumeric):
    def bind_processor(self, dialect):
        return None


class _PGJSON(JSON):
    render_bind_cast = True

    def result_processor(self, dialect, coltype):
        return None


class _PGJSONB(JSONB):
    render_bind_cast = True

    def result_processor(self, dialect, coltype):
        return None


class _PGJSONIndexType(sqltypes.JSON.JSONIndexType):
    def get_dbapi_type(self, dbapi):
        raise NotImplementedError("should not be here")


class _PGJSONIntIndexType(sqltypes.JSON.JSONIntIndexType):
    __visit_name__ = "json_int_index"

    render_bind_cast = True


class _PGJSONStrIndexType(sqltypes.JSON.JSONStrIndexType):
    __visit_name__ = "json_str_index"

    render_bind_cast = True


class _PGJSONPathType(JSONPathType):
    pass

    # DBAPI type 1009


class _PGEnum(ENUM):
    def get_dbapi_type(self, dbapi):
        return dbapi.UNKNOWN


class _PGInterval(INTERVAL):
    render_bind_cast = True

    def get_dbapi_type(self, dbapi):
        return dbapi.INTERVAL

    @classmethod
    def adapt_emulated_to_native(cls, interval, **kw):
        return _PGInterval(precision=interval.second_precision)


class _PGTimeStamp(sqltypes.DateTime):
    render_bind_cast = True


class _PGDate(sqltypes.Date):
    render_bind_cast = True


class _PGTime(sqltypes.Time):
    render_bind_cast = True


class _PGInteger(sqltypes.Integer):
    render_bind_cast = True


class _PGSmallInteger(sqltypes.SmallInteger):
    render_bind_cast = True


class _PGNullType(sqltypes.NullType):
    pass


class _PGBigInteger(sqltypes.BigInteger):
    render_bind_cast = True


class _PGBoolean(sqltypes.Boolean):
    render_bind_cast = True


class _PGARRAY(PGARRAY):
    render_bind_cast = True


class _PGOIDVECTOR(_SpaceVector, OIDVECTOR):
    pass


class _Pg8000Range(ranges.AbstractSingleRangeImpl):
    def bind_processor(self, dialect):
        pg8000_Range = dialect.dbapi.Range

        def to_range(value):
            if isinstance(value, ranges.Range):
                value = pg8000_Range(
                    value.lower, value.upper, value.bounds, value.empty
                )
            return value

        return to_range

    def result_processor(self, dialect, coltype):
        def to_range(value):
            if value is not None:
                value = ranges.Range(
                    value.lower,
                    value.upper,
                    bounds=value.bounds,
                    empty=value.is_empty,
                )
            return value

        return to_range


class _Pg8000MultiRange(ranges.AbstractMultiRangeImpl):
    def bind_processor(self, dialect):
        pg8000_Range = dialect.dbapi.Range

        def to_multirange(value):
            if isinstance(value, list):
                mr = []
                for v in value:
                    if isinstance(v, ranges.Range):
                        mr.append(
                            pg8000_Range(v.lower, v.upper, v.bounds, v.empty)
                        )
                    else:
                        mr.append(v)
                return mr
            else:
                return value

        return to_multirange

    def result_processor(self, dialect, coltype):
        def to_multirange(value):
            if value is None:
                return None
            else:
                return ranges.MultiRange(
                    ranges.Range(
                        v.lower, v.upper, bounds=v.bounds, empty=v.is_empty
                    )
                    for v in value
                )

        return to_multirange


_server_side_id = util.counter()


class PGExecutionContext_pg8000(PGExecutionContext):
    def create_server_side_cursor(self):
        ident = "c_%s_%s" % (hex(id(self))[2:], hex(_server_side_id())[2:])
        return ServerSideCursor(self._dbapi_connection.cursor(), ident)

    def pre_exec(self):
        if not self.compiled:
            return


class ServerSideCursor:
    server_side = True

    def __init__(self, cursor, ident):
        self.ident = ident
        self.cursor = cursor

    @property
    def connection(self):
        return self.cursor.connection

    @property
    def rowcount(self):
        return self.cursor.rowcount

    @property
    def description(self):
        return self.cursor.description

    def execute(self, operation, args=(), stream=None):
        op = "DECLARE " + self.ident + " NO SCROLL CURSOR FOR " + operation
        self.cursor.execute(op, args, stream=stream)
        return self

    def executemany(self, operation, param_sets):
        self.cursor.executemany(operation, param_sets)
        return self

    def fetchone(self):
        self.cursor.execute("FETCH FORWARD 1 FROM " + self.ident)
        return self.cursor.fetchone()

    def fetchmany(self, num=None):
        if num is None:
            return self.fetchall()
        else:
            self.cursor.execute(
                "FETCH FORWARD " + str(int(num)) + " FROM " + self.ident
            )
            return self.cursor.fetchall()

    def fetchall(self):
        self.cursor.execute("FETCH FORWARD ALL FROM " + self.ident)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.execute("CLOSE " + self.ident)
        self.cursor.close()

    def setinputsizes(self, *sizes):
        self.cursor.setinputsizes(*sizes)

    def setoutputsize(self, size, column=None):
        pass


class PGCompiler_pg8000(PGCompiler):
    def visit_mod_binary(self, binary, operator, **kw):
        return (
            self.process(binary.left, **kw)
            + " %% "
            + self.process(binary.right, **kw)
        )


class PGIdentifierPreparer_pg8000(PGIdentifierPreparer):
    def __init__(self, *args, **kwargs):
        PGIdentifierPreparer.__init__(self, *args, **kwargs)
        self._double_percents = False


class PGDialect_pg8000(PGDialect):
    driver = "pg8000"
    supports_statement_cache = True

    supports_unicode_statements = True

    supports_unicode_binds = True

    default_paramstyle = "format"
    supports_sane_multi_rowcount = True
    execution_ctx_cls = PGExecutionContext_pg8000
    statement_compiler = PGCompiler_pg8000
    preparer = PGIdentifierPreparer_pg8000
    supports_server_side_cursors = True

    render_bind_cast = True

    # reversed as of pg8000 1.16.6.  1.16.5 and lower
    # are no longer compatible
    description_encoding = None
    # description_encoding = "use_encoding"

    colspecs = util.update_copy(
        PGDialect.colspecs,
        {
            sqltypes.String: _PGString,
            sqltypes.Numeric: _PGNumericNoBind,
            sqltypes.Float: _PGFloat,
            sqltypes.JSON: _PGJSON,
            sqltypes.Boolean: _PGBoolean,
            sqltypes.NullType: _PGNullType,
            JSONB: _PGJSONB,
            CITEXT: CITEXT,
            sqltypes.JSON.JSONPathType: _PGJSONPathType,
            sqltypes.JSON.JSONIndexType: _PGJSONIndexType,
            sqltypes.JSON.JSONIntIndexType: _PGJSONIntIndexType,
            sqltypes.JSON.JSONStrIndexType: _PGJSONStrIndexType,
            sqltypes.Interval: _PGInterval,
            INTERVAL: _PGInterval,
            sqltypes.DateTime: _PGTimeStamp,
            sqltypes.DateTime: _PGTimeStamp,
            sqltypes.Date: _PGDate,
            sqltypes.Time: _PGTime,
            sqltypes.Integer: _PGInteger,
            sqltypes.SmallInteger: _PGSmallInteger,
            sqltypes.BigInteger: _PGBigInteger,
            sqltypes.Enum: _PGEnum,
            sqltypes.ARRAY: _PGARRAY,
            OIDVECTOR: _PGOIDVECTOR,
            ranges.INT4RANGE: _Pg8000Range,
            ranges.INT8RANGE: _Pg8000Range,
            ranges.NUMRANGE: _Pg8000Range,
            ranges.DATERANGE: _Pg8000Range,
            ranges.TSRANGE: _Pg8000Range,
            ranges.TSTZRANGE: _Pg8000Range,
            ranges.INT4MULTIRANGE: _Pg8000MultiRange,
            ranges.INT8MULTIRANGE: _Pg8000MultiRange,
            ranges.NUMMULTIRANGE: _Pg8000MultiRange,
            ranges.DATEMULTIRANGE: _Pg8000MultiRange,
            ranges.TSMULTIRANGE: _Pg8000MultiRange,
            ranges.TSTZMULTIRANGE: _Pg8000MultiRange,
        },
    )

    def __init__(self, client_encoding=None, **kwargs):
        PGDialect.__init__(self, **kwargs)
        self.client_encoding = client_encoding

        if self._dbapi_version < (1, 16, 6):
            raise NotImplementedError("pg8000 1.16.6 or greater is required")

        if self._native_inet_types:
            raise NotImplementedError(
                "The pg8000 dialect does not fully implement "
                "ipaddress type handling; INET is supported by default, "
                "CIDR is not"
            )

    @util.memoized_property
    def _dbapi_version(self):
        if self.dbapi and hasattr(self.dbapi, "__version__"):
            return tuple(
                [
                    int(x)
                    for x in re.findall(
                        r"(\d+)(?:[-\.]?|$)", self.dbapi.__version__
                    )
                ]
            )
        else:
            return (99, 99, 99)

    @classmethod
    def import_dbapi(cls):
        return __import__("pg8000")

    def create_connect_args(self, url):
        opts = url.translate_connect_args(username="user")
        if "port" in opts:
            opts["port"] = int(opts["port"])
        opts.update(url.query)
        return ([], opts)

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.InterfaceError) and "network error" in str(
            e
        ):
            # new as of pg8000 1.19.0 for broken connections
            return True

        # connection was closed normally
        return "connection is closed" in str(e)

    def get_isolation_level_values(self, dbapi_connection):
        return (
            "AUTOCOMMIT",
            "READ COMMITTED",
            "READ UNCOMMITTED",
            "REPEATABLE READ",
            "SERIALIZABLE",
        )

    def set_isolation_level(self, dbapi_connection, level):
        level = level.replace("_", " ")

        if level == "AUTOCOMMIT":
            dbapi_connection.autocommit = True
        else:
            dbapi_connection.autocommit = False
            cursor = dbapi_connection.cursor()
            cursor.execute(
                "SET SESSION CHARACTERISTICS AS TRANSACTION "
                f"ISOLATION LEVEL {level}"
            )
            cursor.execute("COMMIT")
            cursor.close()

    def set_readonly(self, connection, value):
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SET SESSION CHARACTERISTICS AS TRANSACTION %s"
                % ("READ ONLY" if value else "READ WRITE")
            )
            cursor.execute("COMMIT")
        finally:
            cursor.close()

    def get_readonly(self, connection):
        cursor = connection.cursor()
        try:
            cursor.execute("show transaction_read_only")
            val = cursor.fetchone()[0]
        finally:
            cursor.close()

        return val == "on"

    def set_deferrable(self, connection, value):
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SET SESSION CHARACTERISTICS AS TRANSACTION %s"
                % ("DEFERRABLE" if value else "NOT DEFERRABLE")
            )
            cursor.execute("COMMIT")
        finally:
            cursor.close()

    def get_deferrable(self, connection):
        cursor = connection.cursor()
        try:
            cursor.execute("show transaction_deferrable")
            val = cursor.fetchone()[0]
        finally:
            cursor.close()

        return val == "on"

    def _set_client_encoding(self, dbapi_connection, client_encoding):
        cursor = dbapi_connection.cursor()
        cursor.execute(
            f"""SET CLIENT_ENCODING TO '{
                client_encoding.replace("'", "''")
            }'"""
        )
        cursor.execute("COMMIT")
        cursor.close()

    def do_begin_twophase(self, connection, xid):
        connection.connection.tpc_begin((0, xid, ""))

    def do_prepare_twophase(self, connection, xid):
        connection.connection.tpc_prepare()

    def do_rollback_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        connection.connection.tpc_rollback((0, xid, ""))

    def do_commit_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        connection.connection.tpc_commit((0, xid, ""))

    def do_recover_twophase(self, connection):
        return [row[1] for row in connection.connection.tpc_recover()]

    def on_connect(self):
        fns = []

        def on_connect(conn):
            conn.py_types[quoted_name] = conn.py_types[str]

        fns.append(on_connect)

        if self.client_encoding is not None:

            def on_connect(conn):
                self._set_client_encoding(conn, self.client_encoding)

            fns.append(on_connect)

        if self._native_inet_types is False:

            def on_connect(conn):
                # inet
                conn.register_in_adapter(869, lambda s: s)

                # cidr
                conn.register_in_adapter(650, lambda s: s)

            fns.append(on_connect)

        if self._json_deserializer:

            def on_connect(conn):
                # json
                conn.register_in_adapter(114, self._json_deserializer)

                # jsonb
                conn.register_in_adapter(3802, self._json_deserializer)

            fns.append(on_connect)

        if len(fns) > 0:

            def on_connect(conn):
                for fn in fns:
                    fn(conn)

            return on_connect
        else:
            return None

    @util.memoized_property
    def _dialect_specific_select_one(self):
        return ";"


dialect = PGDialect_pg8000

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\cb_rules.py ===
"""
Build call-back mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
from . import __version__, cfuncs
from .auxfuncs import (
    applyrules,
    debugcapi,
    dictappend,
    errmess,
    getargs,
    hasnote,
    isarray,
    iscomplex,
    iscomplexarray,
    iscomplexfunction,
    isfunction,
    isintent_c,
    isintent_hide,
    isintent_in,
    isintent_inout,
    isintent_nothide,
    isintent_out,
    isoptional,
    isrequired,
    isscalar,
    isstring,
    isstringfunction,
    issubroutine,
    l_and,
    l_not,
    l_or,
    outmess,
    replace,
    stripcomma,
    throw_error,
)

f2py_version = __version__.version


################## Rules for callback function ##############

cb_routine_rules = {
    'cbtypedefs': 'typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);',
    'body': """
#begintitle#
typedef struct {
    PyObject *capi;
    PyTupleObject *args_capi;
    int nofargs;
    jmp_buf jmpbuf;
} #name#_t;

#if defined(F2PY_THREAD_LOCAL_DECL) && !defined(F2PY_USE_PYTHON_TLS)

static F2PY_THREAD_LOCAL_DECL #name#_t *_active_#name# = NULL;

static #name#_t *swap_active_#name#(#name#_t *ptr) {
    #name#_t *prev = _active_#name#;
    _active_#name# = ptr;
    return prev;
}

static #name#_t *get_active_#name#(void) {
    return _active_#name#;
}

#else

static #name#_t *swap_active_#name#(#name#_t *ptr) {
    char *key = "__f2py_cb_#name#";
    return (#name#_t *)F2PySwapThreadLocalCallbackPtr(key, ptr);
}

static #name#_t *get_active_#name#(void) {
    char *key = "__f2py_cb_#name#";
    return (#name#_t *)F2PyGetThreadLocalCallbackPtr(key);
}

#endif

/*typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);*/
#static# #rctype# #callbackname# (#optargs##args##strarglens##noargs#) {
    #name#_t cb_local = { NULL, NULL, 0 };
    #name#_t *cb = NULL;
    PyTupleObject *capi_arglist = NULL;
    PyObject *capi_return = NULL;
    PyObject *capi_tmp = NULL;
    PyObject *capi_arglist_list = NULL;
    int capi_j,capi_i = 0;
    int capi_longjmp_ok = 1;
#decl#
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_start_clock();
#endif
    cb = get_active_#name#();
    if (cb == NULL) {
        capi_longjmp_ok = 0;
        cb = &cb_local;
    }
    capi_arglist = cb->args_capi;
    CFUNCSMESS(\"cb:Call-back function #name# (maxnofargs=#maxnofargs#(-#nofoptargs#))\\n\");
    CFUNCSMESSPY(\"cb:#name#_capi=\",cb->capi);
    if (cb->capi==NULL) {
        capi_longjmp_ok = 0;
        cb->capi = PyObject_GetAttrString(#modulename#_module,\"#argname#\");
        CFUNCSMESSPY(\"cb:#name#_capi=\",cb->capi);
    }
    if (cb->capi==NULL) {
        PyErr_SetString(#modulename#_error,\"cb: Callback #argname# not defined (as an argument or module #modulename# attribute).\\n\");
        goto capi_fail;
    }
    if (F2PyCapsule_Check(cb->capi)) {
    #name#_typedef #name#_cptr;
    #name#_cptr = F2PyCapsule_AsVoidPtr(cb->capi);
    #returncptr#(*#name#_cptr)(#optargs_nm##args_nm##strarglens_nm#);
    #return#
    }
    if (capi_arglist==NULL) {
        capi_longjmp_ok = 0;
        capi_tmp = PyObject_GetAttrString(#modulename#_module,\"#argname#_extra_args\");
        if (capi_tmp) {
            capi_arglist = (PyTupleObject *)PySequence_Tuple(capi_tmp);
            Py_DECREF(capi_tmp);
            if (capi_arglist==NULL) {
                PyErr_SetString(#modulename#_error,\"Failed to convert #modulename#.#argname#_extra_args to tuple.\\n\");
                goto capi_fail;
            }
        } else {
            PyErr_Clear();
            capi_arglist = (PyTupleObject *)Py_BuildValue(\"()\");
        }
    }
    if (capi_arglist == NULL) {
        PyErr_SetString(#modulename#_error,\"Callback #argname# argument list is not set.\\n\");
        goto capi_fail;
    }
#setdims#
#ifdef PYPY_VERSION
#define CAPI_ARGLIST_SETITEM(idx, value) PyList_SetItem((PyObject *)capi_arglist_list, idx, value)
    capi_arglist_list = PySequence_List((PyObject *)capi_arglist);
    if (capi_arglist_list == NULL) goto capi_fail;
#else
#define CAPI_ARGLIST_SETITEM(idx, value) PyTuple_SetItem((PyObject *)capi_arglist, idx, value)
#endif
#pyobjfrom#
#undef CAPI_ARGLIST_SETITEM
#ifdef PYPY_VERSION
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist_list);
#else
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist);
#endif
    CFUNCSMESS(\"cb:Call-back calling Python function #argname#.\\n\");
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_start_call_clock();
#endif
#ifdef PYPY_VERSION
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist_list);
    Py_DECREF(capi_arglist_list);
    capi_arglist_list = NULL;
#else
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist);
#endif
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_stop_call_clock();
#endif
    CFUNCSMESSPY(\"cb:capi_return=\",capi_return);
    if (capi_return == NULL) {
        fprintf(stderr,\"capi_return is NULL\\n\");
        goto capi_fail;
    }
    if (capi_return == Py_None) {
        Py_DECREF(capi_return);
        capi_return = Py_BuildValue(\"()\");
    }
    else if (!PyTuple_Check(capi_return)) {
        capi_return = Py_BuildValue(\"(N)\",capi_return);
    }
    capi_j = PyTuple_Size(capi_return);
    capi_i = 0;
#frompyobj#
    CFUNCSMESS(\"cb:#name#:successful\\n\");
    Py_DECREF(capi_return);
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_stop_clock();
#endif
    goto capi_return_pt;
capi_fail:
    fprintf(stderr,\"Call-back #name# failed.\\n\");
    Py_XDECREF(capi_return);
    Py_XDECREF(capi_arglist_list);
    if (capi_longjmp_ok) {
        longjmp(cb->jmpbuf,-1);
    }
capi_return_pt:
    ;
#return#
}
#endtitle#
""",
    'need': ['setjmp.h', 'CFUNCSMESS', 'F2PY_THREAD_LOCAL_DECL'],
    'maxnofargs': '#maxnofargs#',
    'nofoptargs': '#nofoptargs#',
    'docstr': """\
    def #argname#(#docsignature#): return #docreturn#\\n\\
#docstrsigns#""",
    'latexdocstr': """
{{}\\verb@def #argname#(#latexdocsignature#): return #docreturn#@{}}
#routnote#

#latexdocstrsigns#""",
    'docstrshort': 'def #argname#(#docsignature#): return #docreturn#'
}
cb_rout_rules = [
    {  # Init
        'separatorsfor': {'decl': '\n',
                          'args': ',', 'optargs': '', 'pyobjfrom': '\n', 'freemem': '\n',
                          'args_td': ',', 'optargs_td': '',
                          'args_nm': ',', 'optargs_nm': '',
                          'frompyobj': '\n', 'setdims': '\n',
                          'docstrsigns': '\\n"\n"',
                          'latexdocstrsigns': '\n',
                          'latexdocstrreq': '\n', 'latexdocstropt': '\n',
                          'latexdocstrout': '\n', 'latexdocstrcbs': '\n',
                          },
        'decl': '/*decl*/', 'pyobjfrom': '/*pyobjfrom*/', 'frompyobj': '/*frompyobj*/',
        'args': [], 'optargs': '', 'return': '', 'strarglens': '', 'freemem': '/*freemem*/',
        'args_td': [], 'optargs_td': '', 'strarglens_td': '',
        'args_nm': [], 'optargs_nm': '', 'strarglens_nm': '',
        'noargs': '',
        'setdims': '/*setdims*/',
        'docstrsigns': '', 'latexdocstrsigns': '',
        'docstrreq': '    Required arguments:',
        'docstropt': '    Optional arguments:',
        'docstrout': '    Return objects:',
        'docstrcbs': '    Call-back functions:',
        'docreturn': '', 'docsign': '', 'docsignopt': '',
        'latexdocstrreq': '\\noindent Required arguments:',
        'latexdocstropt': '\\noindent Optional arguments:',
        'latexdocstrout': '\\noindent Return objects:',
        'latexdocstrcbs': '\\noindent Call-back functions:',
        'routnote': {hasnote: '--- #note#', l_not(hasnote): ''},
    }, {  # Function
        'decl': '    #ctype# return_value = 0;',
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},
            '''\
    if (capi_j>capi_i) {
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,
          "#ctype#_from_pyobj failed in converting return_value of"
          " call-back function #name# to C #ctype#\\n");
    } else {
        fprintf(stderr,"Warning: call-back function #name# did not provide"
                       " return value (index=%d, type=#ctype#)\\n",capi_i);
    }''',
            {debugcapi:
             '    fprintf(stderr,"#showvalueformat#.\\n",return_value);'}
        ],
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'}, 'GETSCALARFROMPYTUPLE'],
        'return': '    return return_value;',
        '_check': l_and(isfunction, l_not(isstringfunction), l_not(iscomplexfunction))
    },
    {  # String function
        'pyobjfrom': {debugcapi: '    fprintf(stderr,"debug-capi:cb:#name#:%d:\\n",return_value_len);'},
        'args': '#ctype# return_value,int return_value_len',
        'args_nm': 'return_value,&return_value_len',
        'args_td': '#ctype# ,int',
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->\\"");'},
            """\
    if (capi_j>capi_i) {
        GETSTRFROMPYTUPLE(capi_return,capi_i++,return_value,return_value_len);
    } else {
        fprintf(stderr,"Warning: call-back function #name# did not provide"
                       " return value (index=%d, type=#ctype#)\\n",capi_i);
    }""",
            {debugcapi:
             '    fprintf(stderr,"#showvalueformat#\\".\\n",return_value);'}
        ],
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
                 'string.h', 'GETSTRFROMPYTUPLE'],
        'return': 'return;',
        '_check': isstringfunction
    },
    {  # Complex function
        'optargs': """
#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *return_value
#endif
""",
        'optargs_nm': """
#ifndef F2PY_CB_RETURNCOMPLEX
return_value
#endif
""",
        'optargs_td': """
#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *
#endif
""",
        'decl': """
#ifdef F2PY_CB_RETURNCOMPLEX
    #ctype# return_value = {0, 0};
#endif
""",
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},
            """\
    if (capi_j>capi_i) {
#ifdef F2PY_CB_RETURNCOMPLEX
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,
          \"#ctype#_from_pyobj failed in converting return_value of call-back\"
          \" function #name# to C #ctype#\\n\");
#else
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,return_value,#ctype#,
          \"#ctype#_from_pyobj failed in converting return_value of call-back\"
          \" function #name# to C #ctype#\\n\");
#endif
    } else {
        fprintf(stderr,
                \"Warning: call-back function #name# did not provide\"
                \" return value (index=%d, type=#ctype#)\\n\",capi_i);
    }""",
            {debugcapi: """\
#ifdef F2PY_CB_RETURNCOMPLEX
    fprintf(stderr,\"#showvalueformat#.\\n\",(return_value).r,(return_value).i);
#else
    fprintf(stderr,\"#showvalueformat#.\\n\",(*return_value).r,(*return_value).i);
#endif
"""}
        ],
        'return': """
#ifdef F2PY_CB_RETURNCOMPLEX
    return return_value;
#else
    return;
#endif
""",
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
                 'string.h', 'GETSCALARFROMPYTUPLE', '#ctype#'],
        '_check': iscomplexfunction
    },
    {'docstrout': '        #pydocsignout#',
     'latexdocstrout': ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                        {hasnote: '--- #note#'}],
     'docreturn': '#rname#,',
     '_check': isfunction},
    {'_check': issubroutine, 'return': 'return;'}
]

cb_arg_rules = [
    {  # Doc
        'docstropt': {l_and(isoptional, isintent_nothide): '        #pydocsign#'},
        'docstrreq': {l_and(isrequired, isintent_nothide): '        #pydocsign#'},
        'docstrout': {isintent_out: '        #pydocsignout#'},
        'latexdocstropt': {l_and(isoptional, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrreq': {l_and(isrequired, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrout': {isintent_out: ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                                          {l_and(hasnote, isintent_hide): '--- #note#',
                                           l_and(hasnote, isintent_nothide): '--- See above.'}]},
        'docsign': {l_and(isrequired, isintent_nothide): '#varname#,'},
        'docsignopt': {l_and(isoptional, isintent_nothide): '#varname#,'},
        'depend': ''
    },
    {
        'args': {
            l_and(isscalar, isintent_c): '#ctype# #varname_i#',
            l_and(isscalar, l_not(isintent_c)): '#ctype# *#varname_i#_cb_capi',
            isarray: '#ctype# *#varname_i#',
            isstring: '#ctype# #varname_i#'
        },
        'args_nm': {
            l_and(isscalar, isintent_c): '#varname_i#',
            l_and(isscalar, l_not(isintent_c)): '#varname_i#_cb_capi',
            isarray: '#varname_i#',
            isstring: '#varname_i#'
        },
        'args_td': {
            l_and(isscalar, isintent_c): '#ctype#',
            l_and(isscalar, l_not(isintent_c)): '#ctype# *',
            isarray: '#ctype# *',
            isstring: '#ctype#'
        },
        'need': {l_or(isscalar, isarray, isstring): '#ctype#'},
        # untested with multiple args
        'strarglens': {isstring: ',int #varname_i#_cb_len'},
        'strarglens_td': {isstring: ',int'},  # untested with multiple args
        # untested with multiple args
        'strarglens_nm': {isstring: ',#varname_i#_cb_len'},
    },
    {  # Scalars
        'decl': {l_not(isintent_c): '    #ctype# #varname_i#=(*#varname_i#_cb_capi);'},
        'error': {l_and(isintent_c, isintent_out,
                        throw_error('intent(c,out) is forbidden for callback scalar arguments')):
                  ''},
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->");'},
                      {isintent_out:
                       '    if (capi_j>capi_i)\n        GETSCALARFROMPYTUPLE(capi_return,capi_i++,#varname_i#_cb_capi,#ctype#,"#ctype#_from_pyobj failed in converting argument #varname# of call-back function #name# to C #ctype#\\n");'},
                      {l_and(debugcapi, l_and(l_not(iscomplex), isintent_c)):
                          '    fprintf(stderr,"#showvalueformat#.\\n",#varname_i#);'},
                      {l_and(debugcapi, l_and(l_not(iscomplex), l_not(isintent_c))):
                          '    fprintf(stderr,"#showvalueformat#.\\n",*#varname_i#_cb_capi);'},
                      {l_and(debugcapi, l_and(iscomplex, isintent_c)):
                          '    fprintf(stderr,"#showvalueformat#.\\n",(#varname_i#).r,(#varname_i#).i);'},
                      {l_and(debugcapi, l_and(iscomplex, l_not(isintent_c))):
                          '    fprintf(stderr,"#showvalueformat#.\\n",(*#varname_i#_cb_capi).r,(*#varname_i#_cb_capi).i);'},
                      ],
        'need': [{isintent_out: ['#ctype#_from_pyobj', 'GETSCALARFROMPYTUPLE']},
                 {debugcapi: 'CFUNCSMESS'}],
        '_check': isscalar
    }, {
        'pyobjfrom': [{isintent_in: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyobj_from_#ctype#1(#varname_i#)))
            goto capi_fail;"""},
                      {isintent_inout: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyarr_from_p_#ctype#1(#varname_i#_cb_capi)))
            goto capi_fail;"""}],
        'need': [{isintent_in: 'pyobj_from_#ctype#1'},
                 {isintent_inout: 'pyarr_from_p_#ctype#1'},
                 {iscomplex: '#ctype#'}],
        '_check': l_and(isscalar, isintent_nothide),
        '_optional': ''
    }, {  # String
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->\\"");'},
                      """    if (capi_j>capi_i)
        GETSTRFROMPYTUPLE(capi_return,capi_i++,#varname_i#,#varname_i#_cb_len);""",
                      {debugcapi:
                       '    fprintf(stderr,"#showvalueformat#\\":%d:.\\n",#varname_i#,#varname_i#_cb_len);'},
                      ],
        'need': ['#ctype#', 'GETSTRFROMPYTUPLE',
                 {debugcapi: 'CFUNCSMESS'}, 'string.h'],
        '_check': l_and(isstring, isintent_out)
    }, {
        'pyobjfrom': [
            {debugcapi:
             ('    fprintf(stderr,"debug-capi:cb:#varname#=#showvalueformat#:'
              '%d:\\n",#varname_i#,#varname_i#_cb_len);')},
            {isintent_in: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyobj_from_#ctype#1size(#varname_i#,#varname_i#_cb_len)))
            goto capi_fail;"""},
                      {isintent_inout: """\
    if (cb->nofargs>capi_i) {
        int #varname_i#_cb_dims[] = {#varname_i#_cb_len};
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyarr_from_p_#ctype#1(#varname_i#,#varname_i#_cb_dims)))
            goto capi_fail;
    }"""}],
        'need': [{isintent_in: 'pyobj_from_#ctype#1size'},
                 {isintent_inout: 'pyarr_from_p_#ctype#1'}],
        '_check': l_and(isstring, isintent_nothide),
        '_optional': ''
    },
    # Array ...
    {
        'decl': '    npy_intp #varname_i#_Dims[#rank#] = {#rank*[-1]#};',
        'setdims': '    #cbsetdims#;',
        '_check': isarray,
        '_depend': ''
    },
    {
        'pyobjfrom': [{debugcapi: '    fprintf(stderr,"debug-capi:cb:#varname#\\n");'},
                      {isintent_c: """\
    if (cb->nofargs>capi_i) {
        /* tmp_arr will be inserted to capi_arglist_list that will be
           destroyed when leaving callback function wrapper together
           with tmp_arr. */
        PyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,
          #rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,#elsize#,
          NPY_ARRAY_CARRAY,NULL);
""",
                       l_not(isintent_c): """\
    if (cb->nofargs>capi_i) {
        /* tmp_arr will be inserted to capi_arglist_list that will be
           destroyed when leaving callback function wrapper together
           with tmp_arr. */
        PyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,
          #rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,#elsize#,
          NPY_ARRAY_FARRAY,NULL);
""",
                       },
                      """
        if (tmp_arr==NULL)
            goto capi_fail;
        if (CAPI_ARGLIST_SETITEM(capi_i++,(PyObject *)tmp_arr))
            goto capi_fail;
}"""],
        '_check': l_and(isarray, isintent_nothide, l_or(isintent_in, isintent_inout)),
        '_optional': '',
    }, {
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->");'},
                      """    if (capi_j>capi_i) {
        PyArrayObject *rv_cb_arr = NULL;
        if ((capi_tmp = PyTuple_GetItem(capi_return,capi_i++))==NULL) goto capi_fail;
        rv_cb_arr =  array_from_pyobj(#atype#,#varname_i#_Dims,#rank#,F2PY_INTENT_IN""",
                      {isintent_c: '|F2PY_INTENT_C'},
                      """,capi_tmp);
        if (rv_cb_arr == NULL) {
            fprintf(stderr,\"rv_cb_arr is NULL\\n\");
            goto capi_fail;
        }
        MEMCOPY(#varname_i#,PyArray_DATA(rv_cb_arr),PyArray_NBYTES(rv_cb_arr));
        if (capi_tmp != (PyObject *)rv_cb_arr) {
            Py_DECREF(rv_cb_arr);
        }
    }""",
                      {debugcapi: '    fprintf(stderr,"<-.\\n");'},
                      ],
        'need': ['MEMCOPY', {iscomplexarray: '#ctype#'}],
        '_check': l_and(isarray, isintent_out)
    }, {
        'docreturn': '#varname#,',
        '_check': isintent_out
    }
]

################## Build call-back module #############
cb_map = {}


def buildcallbacks(m):
    cb_map[m['name']] = []
    for bi in m['body']:
        if bi['block'] == 'interface':
            for b in bi['body']:
                if b:
                    buildcallback(b, m['name'])
                else:
                    errmess(f"warning: empty body for {m['name']}\n")


def buildcallback(rout, um):
    from . import capi_maps

    outmess(f"    Constructing call-back function \"cb_{rout['name']}_in_{um}\"\n")
    args, depargs = getargs(rout)
    capi_maps.depargs = depargs
    var = rout['vars']
    vrd = capi_maps.cb_routsign2map(rout, um)
    rd = dictappend({}, vrd)
    cb_map[um].append([rout['name'], rd['name']])
    for r in cb_rout_rules:
        if ('_check' in r and r['_check'](rout)) or ('_check' not in r):
            ar = applyrules(r, vrd, rout)
            rd = dictappend(rd, ar)
    savevrd = {}
    for i, a in enumerate(args):
        vrd = capi_maps.cb_sign2map(a, var[a], index=i)
        savevrd[a] = vrd
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if '_optional' in r and isoptional(var[a]):
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in args:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' in r:
                continue
            if ('_optional' not in r) or ('_optional' in r and isrequired(var[a])):
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in depargs:
        vrd = savevrd[a]
        for r in cb_arg_rules:
            if '_depend' not in r:
                continue
            if '_optional' in r:
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    if 'args' in rd and 'optargs' in rd:
        if isinstance(rd['optargs'], list):
            rd['optargs'] = rd['optargs'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,
#endif
"""]
            rd['optargs_nm'] = rd['optargs_nm'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,
#endif
"""]
            rd['optargs_td'] = rd['optargs_td'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,
#endif
"""]
    if isinstance(rd['docreturn'], list):
        rd['docreturn'] = stripcomma(
            replace('#docreturn#', {'docreturn': rd['docreturn']}))
    optargs = stripcomma(replace('#docsignopt#',
                                 {'docsignopt': rd['docsignopt']}
                                 ))
    if optargs == '':
        rd['docsignature'] = stripcomma(
            replace('#docsign#', {'docsign': rd['docsign']}))
    else:
        rd['docsignature'] = replace('#docsign#[#docsignopt#]',
                                     {'docsign': rd['docsign'],
                                      'docsignopt': optargs,
                                      })
    rd['latexdocsignature'] = rd['docsignature'].replace('_', '\\_')
    rd['latexdocsignature'] = rd['latexdocsignature'].replace(',', ', ')
    rd['docstrsigns'] = []
    rd['latexdocstrsigns'] = []
    for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
        if k in rd and isinstance(rd[k], list):
            rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
        k = 'latex' + k
        if k in rd and isinstance(rd[k], list):
            rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] +\
                ['\\begin{description}'] + rd[k][1:] +\
                ['\\end{description}']
    if 'args' not in rd:
        rd['args'] = ''
        rd['args_td'] = ''
        rd['args_nm'] = ''
    if not (rd.get('args') or rd.get('optargs') or rd.get('strarglens')):
        rd['noargs'] = 'void'

    ar = applyrules(cb_routine_rules, rd)
    cfuncs.callbacks[rd['name']] = ar['body']
    if isinstance(ar['need'], str):
        ar['need'] = [ar['need']]

    if 'need' in rd:
        for t in cfuncs.typedefs.keys():
            if t in rd['need']:
                ar['need'].append(t)

    cfuncs.typedefs_generated[rd['name'] + '_typedef'] = ar['cbtypedefs']
    ar['need'].append(rd['name'] + '_typedef')
    cfuncs.needs[rd['name']] = ar['need']

    capi_maps.lcb2_map[rd['name']] = {'maxnofargs': ar['maxnofargs'],
                                      'nofoptargs': ar['nofoptargs'],
                                      'docstr': ar['docstr'],
                                      'latexdocstr': ar['latexdocstr'],
                                      'argname': rd['argname']
                                      }
    outmess(f"      {ar['docstrshort']}\n")
################## Build call-back function #############