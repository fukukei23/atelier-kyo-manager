
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\visitors.py ===
# sql/visitors.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Visitor/traversal interface and library functions.


"""

from __future__ import annotations

from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self

if TYPE_CHECKING:
    from .annotation import _AnnotationDict
    from .elements import ColumnElement

if typing.TYPE_CHECKING or not HAS_CYEXTENSION:
    from ._py_util import prefix_anon_map as prefix_anon_map
    from ._py_util import cache_anon_map as anon_map
else:
    from sqlalchemy.cyextension.util import (  # noqa: F401,E501
        prefix_anon_map as prefix_anon_map,
    )
    from sqlalchemy.cyextension.util import (  # noqa: F401,E501
        cache_anon_map as anon_map,
    )


__all__ = [
    "iterate",
    "traverse_using",
    "traverse",
    "cloned_traverse",
    "replacement_traverse",
    "Visitable",
    "ExternalTraversal",
    "InternalTraversal",
    "anon_map",
]


class _CompilerDispatchType(Protocol):
    def __call__(_self, self: Visitable, visitor: Any, **kw: Any) -> Any: ...


class Visitable:
    """Base class for visitable objects.

    :class:`.Visitable` is used to implement the SQL compiler dispatch
    functions.    Other forms of traversal such as for cache key generation
    are implemented separately using the :class:`.HasTraverseInternals`
    interface.

    .. versionchanged:: 2.0  The :class:`.Visitable` class was named
       :class:`.Traversible` in the 1.4 series; the name is changed back
       to :class:`.Visitable` in 2.0 which is what it was prior to 1.4.

       Both names remain importable in both 1.4 and 2.0 versions.

    """

    __slots__ = ()

    __visit_name__: str

    _original_compiler_dispatch: _CompilerDispatchType

    if typing.TYPE_CHECKING:

        def _compiler_dispatch(self, visitor: Any, **kw: Any) -> str: ...

    def __init_subclass__(cls) -> None:
        if "__visit_name__" in cls.__dict__:
            cls._generate_compiler_dispatch()
        super().__init_subclass__()

    @classmethod
    def _generate_compiler_dispatch(cls) -> None:
        visit_name = cls.__visit_name__

        if "_compiler_dispatch" in cls.__dict__:
            # class has a fixed _compiler_dispatch() method.
            # copy it to "original" so that we can get it back if
            # sqlalchemy.ext.compiles overrides it.
            cls._original_compiler_dispatch = cls._compiler_dispatch
            return

        if not isinstance(visit_name, str):
            raise exc.InvalidRequestError(
                f"__visit_name__ on class {cls.__name__} must be a string "
                "at the class level"
            )

        name = "visit_%s" % visit_name
        getter = operator.attrgetter(name)

        def _compiler_dispatch(
            self: Visitable, visitor: Any, **kw: Any
        ) -> str:
            """Look for an attribute named "visit_<visit_name>" on the
            visitor, and call it with the same kw params.

            """
            try:
                meth = getter(visitor)
            except AttributeError as err:
                return visitor.visit_unsupported_compilation(self, err, **kw)  # type: ignore  # noqa: E501
            else:
                return meth(self, **kw)  # type: ignore  # noqa: E501

        cls._compiler_dispatch = (  # type: ignore
            cls._original_compiler_dispatch
        ) = _compiler_dispatch

    def __class_getitem__(cls, key: Any) -> Any:
        # allow generic classes in py3.9+
        return cls


class InternalTraversal(Enum):
    r"""Defines visitor symbols used for internal traversal.

    The :class:`.InternalTraversal` class is used in two ways.  One is that
    it can serve as the superclass for an object that implements the
    various visit methods of the class.   The other is that the symbols
    themselves of :class:`.InternalTraversal` are used within
    the ``_traverse_internals`` collection.   Such as, the :class:`.Case`
    object defines ``_traverse_internals`` as ::

        class Case(ColumnElement[_T]):
            _traverse_internals = [
                ("value", InternalTraversal.dp_clauseelement),
                ("whens", InternalTraversal.dp_clauseelement_tuples),
                ("else_", InternalTraversal.dp_clauseelement),
            ]

    Above, the :class:`.Case` class indicates its internal state as the
    attributes named ``value``, ``whens``, and ``else_``.    They each
    link to an :class:`.InternalTraversal` method which indicates the type
    of datastructure to which each attribute refers.

    Using the ``_traverse_internals`` structure, objects of type
    :class:`.InternalTraversible` will have the following methods automatically
    implemented:

    * :meth:`.HasTraverseInternals.get_children`

    * :meth:`.HasTraverseInternals._copy_internals`

    * :meth:`.HasCacheKey._gen_cache_key`

    Subclasses can also implement these methods directly, particularly for the
    :meth:`.HasTraverseInternals._copy_internals` method, when special steps
    are needed.

    .. versionadded:: 1.4

    """

    dp_has_cache_key = "HC"
    """Visit a :class:`.HasCacheKey` object."""

    dp_has_cache_key_list = "HL"
    """Visit a list of :class:`.HasCacheKey` objects."""

    dp_clauseelement = "CE"
    """Visit a :class:`_expression.ClauseElement` object."""

    dp_fromclause_canonical_column_collection = "FC"
    """Visit a :class:`_expression.FromClause` object in the context of the
    ``columns`` attribute.

    The column collection is "canonical", meaning it is the originally
    defined location of the :class:`.ColumnClause` objects.   Right now
    this means that the object being visited is a
    :class:`_expression.TableClause`
    or :class:`_schema.Table` object only.

    """

    dp_clauseelement_tuples = "CTS"
    """Visit a list of tuples which contain :class:`_expression.ClauseElement`
    objects.

    """

    dp_clauseelement_list = "CL"
    """Visit a list of :class:`_expression.ClauseElement` objects.

    """

    dp_clauseelement_tuple = "CT"
    """Visit a tuple of :class:`_expression.ClauseElement` objects.

    """

    dp_executable_options = "EO"

    dp_with_context_options = "WC"

    dp_fromclause_ordered_set = "CO"
    """Visit an ordered set of :class:`_expression.FromClause` objects. """

    dp_string = "S"
    """Visit a plain string value.

    Examples include table and column names, bound parameter keys, special
    keywords such as "UNION", "UNION ALL".

    The string value is considered to be significant for cache key
    generation.

    """

    dp_string_list = "SL"
    """Visit a list of strings."""

    dp_anon_name = "AN"
    """Visit a potentially "anonymized" string value.

    The string value is considered to be significant for cache key
    generation.

    """

    dp_boolean = "B"
    """Visit a boolean value.

    The boolean value is considered to be significant for cache key
    generation.

    """

    dp_operator = "O"
    """Visit an operator.

    The operator is a function from the :mod:`sqlalchemy.sql.operators`
    module.

    The operator value is considered to be significant for cache key
    generation.

    """

    dp_type = "T"
    """Visit a :class:`.TypeEngine` object

    The type object is considered to be significant for cache key
    generation.

    """

    dp_plain_dict = "PD"
    """Visit a dictionary with string keys.

    The keys of the dictionary should be strings, the values should
    be immutable and hashable.   The dictionary is considered to be
    significant for cache key generation.

    """

    dp_dialect_options = "DO"
    """Visit a dialect options structure."""

    dp_string_clauseelement_dict = "CD"
    """Visit a dictionary of string keys to :class:`_expression.ClauseElement`
    objects.

    """

    dp_string_multi_dict = "MD"
    """Visit a dictionary of string keys to values which may either be
    plain immutable/hashable or :class:`.HasCacheKey` objects.

    """

    dp_annotations_key = "AK"
    """Visit the _annotations_cache_key element.

    This is a dictionary of additional information about a ClauseElement
    that modifies its role.  It should be included when comparing or caching
    objects, however generating this key is relatively expensive.   Visitors
    should check the "_annotations" dict for non-None first before creating
    this key.

    """

    dp_plain_obj = "PO"
    """Visit a plain python object.

    The value should be immutable and hashable, such as an integer.
    The value is considered to be significant for cache key generation.

    """

    dp_named_ddl_element = "DD"
    """Visit a simple named DDL element.

    The current object used by this method is the :class:`.Sequence`.

    The object is only considered to be important for cache key generation
    as far as its name, but not any other aspects of it.

    """

    dp_prefix_sequence = "PS"
    """Visit the sequence represented by :class:`_expression.HasPrefixes`
    or :class:`_expression.HasSuffixes`.

    """

    dp_table_hint_list = "TH"
    """Visit the ``_hints`` collection of a :class:`_expression.Select`
    object.

    """

    dp_setup_join_tuple = "SJ"

    dp_memoized_select_entities = "ME"

    dp_statement_hint_list = "SH"
    """Visit the ``_statement_hints`` collection of a
    :class:`_expression.Select`
    object.

    """

    dp_unknown_structure = "UK"
    """Visit an unknown structure.

    """

    dp_dml_ordered_values = "DML_OV"
    """Visit the values() ordered tuple list of an
    :class:`_expression.Update` object."""

    dp_dml_values = "DML_V"
    """Visit the values() dictionary of a :class:`.ValuesBase`
    (e.g. Insert or Update) object.

    """

    dp_dml_multi_values = "DML_MV"
    """Visit the values() multi-valued list of dictionaries of an
    :class:`_expression.Insert` object.

    """

    dp_propagate_attrs = "PA"
    """Visit the propagate attrs dict.  This hardcodes to the particular
    elements we care about right now."""

    """Symbols that follow are additional symbols that are useful in
    caching applications.

    Traversals for :class:`_expression.ClauseElement` objects only need to use
    those symbols present in :class:`.InternalTraversal`.  However, for
    additional caching use cases within the ORM, symbols dealing with the
    :class:`.HasCacheKey` class are added here.

    """

    dp_ignore = "IG"
    """Specify an object that should be ignored entirely.

    This currently applies function call argument caching where some
    arguments should not be considered to be part of a cache key.

    """

    dp_inspectable = "IS"
    """Visit an inspectable object where the return value is a
    :class:`.HasCacheKey` object."""

    dp_multi = "M"
    """Visit an object that may be a :class:`.HasCacheKey` or may be a
    plain hashable object."""

    dp_multi_list = "MT"
    """Visit a tuple containing elements that may be :class:`.HasCacheKey` or
    may be a plain hashable object."""

    dp_has_cache_key_tuples = "HT"
    """Visit a list of tuples which contain :class:`.HasCacheKey`
    objects.

    """

    dp_inspectable_list = "IL"
    """Visit a list of inspectable objects which upon inspection are
    HasCacheKey objects."""


_TraverseInternalsType = List[Tuple[str, InternalTraversal]]
"""a structure that defines how a HasTraverseInternals should be
traversed.

This structure consists of a list of (attributename, internaltraversal)
tuples, where the "attributename" refers to the name of an attribute on an
instance of the HasTraverseInternals object, and "internaltraversal" refers
to an :class:`.InternalTraversal` enumeration symbol defining what kind
of data this attribute stores, which indicates to the traverser how it should
be handled.

"""


class HasTraverseInternals:
    """base for classes that have a "traverse internals" element,
    which defines all kinds of ways of traversing the elements of an object.

    Compared to :class:`.Visitable`, which relies upon an external visitor to
    define how the object is travered (i.e. the :class:`.SQLCompiler`), the
    :class:`.HasTraverseInternals` interface allows classes to define their own
    traversal, that is, what attributes are accessed and in what order.

    """

    __slots__ = ()

    _traverse_internals: _TraverseInternalsType

    _is_immutable: bool = False

    @util.preload_module("sqlalchemy.sql.traversals")
    def get_children(
        self, *, omit_attrs: Tuple[str, ...] = (), **kw: Any
    ) -> Iterable[HasTraverseInternals]:
        r"""Return immediate child :class:`.visitors.HasTraverseInternals`
        elements of this :class:`.visitors.HasTraverseInternals`.

        This is used for visit traversal.

        \**kw may contain flags that change the collection that is
        returned, for example to return a subset of items in order to
        cut down on larger traversals, or to return child items from a
        different context (such as schema-level collections instead of
        clause-level).

        """

        traversals = util.preloaded.sql_traversals

        try:
            traverse_internals = self._traverse_internals
        except AttributeError:
            # user-defined classes may not have a _traverse_internals
            return []

        dispatch = traversals._get_children.run_generated_dispatch
        return itertools.chain.from_iterable(
            meth(obj, **kw)
            for attrname, obj, meth in dispatch(
                self, traverse_internals, "_generated_get_children_traversal"
            )
            if attrname not in omit_attrs and obj is not None
        )


class _InternalTraversalDispatchType(Protocol):
    def __call__(s, self: object, visitor: HasTraversalDispatch) -> Any: ...


class HasTraversalDispatch:
    r"""Define infrastructure for classes that perform internal traversals

    .. versionadded:: 2.0

    """

    __slots__ = ()

    _dispatch_lookup: ClassVar[Dict[Union[InternalTraversal, str], str]] = {}

    def dispatch(self, visit_symbol: InternalTraversal) -> Callable[..., Any]:
        """Given a method from :class:`.HasTraversalDispatch`, return the
        corresponding method on a subclass.

        """
        name = _dispatch_lookup[visit_symbol]
        return getattr(self, name, None)  # type: ignore

    def run_generated_dispatch(
        self,
        target: object,
        internal_dispatch: _TraverseInternalsType,
        generate_dispatcher_name: str,
    ) -> Any:
        dispatcher: _InternalTraversalDispatchType
        try:
            dispatcher = target.__class__.__dict__[generate_dispatcher_name]
        except KeyError:
            # traversals.py -> _preconfigure_traversals()
            # may be used to run these ahead of time, but
            # is not enabled right now.
            # this block will generate any remaining dispatchers.
            dispatcher = self.generate_dispatch(
                target.__class__, internal_dispatch, generate_dispatcher_name
            )
        return dispatcher(target, self)

    def generate_dispatch(
        self,
        target_cls: Type[object],
        internal_dispatch: _TraverseInternalsType,
        generate_dispatcher_name: str,
    ) -> _InternalTraversalDispatchType:
        dispatcher = self._generate_dispatcher(
            internal_dispatch, generate_dispatcher_name
        )
        # assert isinstance(target_cls, type)
        setattr(target_cls, generate_dispatcher_name, dispatcher)
        return dispatcher

    def _generate_dispatcher(
        self, internal_dispatch: _TraverseInternalsType, method_name: str
    ) -> _InternalTraversalDispatchType:
        names = []
        for attrname, visit_sym in internal_dispatch:
            meth = self.dispatch(visit_sym)
            if meth is not None:
                visit_name = _dispatch_lookup[visit_sym]
                names.append((attrname, visit_name))

        code = (
            ("    return [\n")
            + (
                ", \n".join(
                    "        (%r, self.%s, visitor.%s)"
                    % (attrname, attrname, visit_name)
                    for attrname, visit_name in names
                )
            )
            + ("\n    ]\n")
        )
        meth_text = ("def %s(self, visitor):\n" % method_name) + code + "\n"
        return cast(
            _InternalTraversalDispatchType,
            langhelpers._exec_code_in_env(meth_text, {}, method_name),
        )


ExtendedInternalTraversal = InternalTraversal


def _generate_traversal_dispatch() -> None:
    lookup = _dispatch_lookup

    for sym in InternalTraversal:
        key = sym.name
        if key.startswith("dp_"):
            visit_key = key.replace("dp_", "visit_")
            sym_name = sym.value
            assert sym_name not in lookup, sym_name
            lookup[sym] = lookup[sym_name] = visit_key


_dispatch_lookup = HasTraversalDispatch._dispatch_lookup
_generate_traversal_dispatch()


class ExternallyTraversible(HasTraverseInternals, Visitable):
    __slots__ = ()

    _annotations: Mapping[Any, Any] = util.EMPTY_DICT

    if typing.TYPE_CHECKING:

        def _annotate(self, values: _AnnotationDict) -> Self: ...

        def get_children(
            self, *, omit_attrs: Tuple[str, ...] = (), **kw: Any
        ) -> Iterable[ExternallyTraversible]: ...

    def _clone(self, **kw: Any) -> Self:
        """clone this element"""
        raise NotImplementedError()

    def _copy_internals(
        self, *, omit_attrs: Tuple[str, ...] = (), **kw: Any
    ) -> None:
        """Reassign internal elements to be clones of themselves.

        Called during a copy-and-traverse operation on newly
        shallow-copied elements to create a deep copy.

        The given clone function should be used, which may be applying
        additional transformations to the element (i.e. replacement
        traversal, cloned traversal, annotations).

        """
        raise NotImplementedError()


_ET = TypeVar("_ET", bound=ExternallyTraversible)

_CE = TypeVar("_CE", bound="ColumnElement[Any]")

_TraverseCallableType = Callable[[_ET], None]


class _CloneCallableType(Protocol):
    def __call__(self, element: _ET, **kw: Any) -> _ET: ...


class _TraverseTransformCallableType(Protocol[_ET]):
    def __call__(self, element: _ET, **kw: Any) -> Optional[_ET]: ...


_ExtT = TypeVar("_ExtT", bound="ExternalTraversal")


class ExternalTraversal(util.MemoizedSlots):
    """Base class for visitor objects which can traverse externally using
    the :func:`.visitors.traverse` function.

    Direct usage of the :func:`.visitors.traverse` function is usually
    preferred.

    """

    __slots__ = ("_visitor_dict", "_next")

    __traverse_options__: Dict[str, Any] = {}
    _next: Optional[ExternalTraversal]

    def traverse_single(self, obj: Visitable, **kw: Any) -> Any:
        for v in self.visitor_iterator:
            meth = getattr(v, "visit_%s" % obj.__visit_name__, None)
            if meth:
                return meth(obj, **kw)

    def iterate(
        self, obj: Optional[ExternallyTraversible]
    ) -> Iterator[ExternallyTraversible]:
        """Traverse the given expression structure, returning an iterator
        of all elements.

        """
        return iterate(obj, self.__traverse_options__)

    @overload
    def traverse(self, obj: Literal[None]) -> None: ...

    @overload
    def traverse(
        self, obj: ExternallyTraversible
    ) -> ExternallyTraversible: ...

    def traverse(
        self, obj: Optional[ExternallyTraversible]
    ) -> Optional[ExternallyTraversible]:
        """Traverse and visit the given expression structure."""

        return traverse(obj, self.__traverse_options__, self._visitor_dict)

    def _memoized_attr__visitor_dict(
        self,
    ) -> Dict[str, _TraverseCallableType[Any]]:
        visitors = {}

        for name in dir(self):
            if name.startswith("visit_"):
                visitors[name[6:]] = getattr(self, name)
        return visitors

    @property
    def visitor_iterator(self) -> Iterator[ExternalTraversal]:
        """Iterate through this visitor and each 'chained' visitor."""

        v: Optional[ExternalTraversal] = self
        while v:
            yield v
            v = getattr(v, "_next", None)

    def chain(self: _ExtT, visitor: ExternalTraversal) -> _ExtT:
        """'Chain' an additional ExternalTraversal onto this ExternalTraversal

        The chained visitor will receive all visit events after this one.

        """
        tail = list(self.visitor_iterator)[-1]
        tail._next = visitor
        return self


class CloningExternalTraversal(ExternalTraversal):
    """Base class for visitor objects which can traverse using
    the :func:`.visitors.cloned_traverse` function.

    Direct usage of the :func:`.visitors.cloned_traverse` function is usually
    preferred.


    """

    __slots__ = ()

    def copy_and_process(
        self, list_: List[ExternallyTraversible]
    ) -> List[ExternallyTraversible]:
        """Apply cloned traversal to the given list of elements, and return
        the new list.

        """
        return [self.traverse(x) for x in list_]

    @overload
    def traverse(self, obj: Literal[None]) -> None: ...

    @overload
    def traverse(
        self, obj: ExternallyTraversible
    ) -> ExternallyTraversible: ...

    def traverse(
        self, obj: Optional[ExternallyTraversible]
    ) -> Optional[ExternallyTraversible]:
        """Traverse and visit the given expression structure."""

        return cloned_traverse(
            obj, self.__traverse_options__, self._visitor_dict
        )


class ReplacingExternalTraversal(CloningExternalTraversal):
    """Base class for visitor objects which can traverse using
    the :func:`.visitors.replacement_traverse` function.

    Direct usage of the :func:`.visitors.replacement_traverse` function is
    usually preferred.

    """

    __slots__ = ()

    def replace(
        self, elem: ExternallyTraversible
    ) -> Optional[ExternallyTraversible]:
        """Receive pre-copied elements during a cloning traversal.

        If the method returns a new element, the element is used
        instead of creating a simple copy of the element.  Traversal
        will halt on the newly returned element if it is re-encountered.
        """
        return None

    @overload
    def traverse(self, obj: Literal[None]) -> None: ...

    @overload
    def traverse(
        self, obj: ExternallyTraversible
    ) -> ExternallyTraversible: ...

    def traverse(
        self, obj: Optional[ExternallyTraversible]
    ) -> Optional[ExternallyTraversible]:
        """Traverse and visit the given expression structure."""

        def replace(
            element: ExternallyTraversible,
            **kw: Any,
        ) -> Optional[ExternallyTraversible]:
            for v in self.visitor_iterator:
                e = cast(ReplacingExternalTraversal, v).replace(element)
                if e is not None:
                    return e

            return None

        return replacement_traverse(obj, self.__traverse_options__, replace)


# backwards compatibility
Traversible = Visitable

ClauseVisitor = ExternalTraversal
CloningVisitor = CloningExternalTraversal
ReplacingCloningVisitor = ReplacingExternalTraversal


def iterate(
    obj: Optional[ExternallyTraversible],
    opts: Mapping[str, Any] = util.EMPTY_DICT,
) -> Iterator[ExternallyTraversible]:
    r"""Traverse the given expression structure, returning an iterator.

    Traversal is configured to be breadth-first.

    The central API feature used by the :func:`.visitors.iterate`
    function is the
    :meth:`_expression.ClauseElement.get_children` method of
    :class:`_expression.ClauseElement` objects.  This method should return all
    the :class:`_expression.ClauseElement` objects which are associated with a
    particular :class:`_expression.ClauseElement` object. For example, a
    :class:`.Case` structure will refer to a series of
    :class:`_expression.ColumnElement` objects within its "whens" and "else\_"
    member variables.

    :param obj: :class:`_expression.ClauseElement` structure to be traversed

    :param opts: dictionary of iteration options.   This dictionary is usually
     empty in modern usage.

    """
    if obj is None:
        return

    yield obj
    children = obj.get_children(**opts)

    if not children:
        return

    stack = deque([children])
    while stack:
        t_iterator = stack.popleft()
        for t in t_iterator:
            yield t
            stack.append(t.get_children(**opts))


@overload
def traverse_using(
    iterator: Iterable[ExternallyTraversible],
    obj: Literal[None],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> None: ...


@overload
def traverse_using(
    iterator: Iterable[ExternallyTraversible],
    obj: ExternallyTraversible,
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> ExternallyTraversible: ...


def traverse_using(
    iterator: Iterable[ExternallyTraversible],
    obj: Optional[ExternallyTraversible],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> Optional[ExternallyTraversible]:
    """Visit the given expression structure using the given iterator of
    objects.

    :func:`.visitors.traverse_using` is usually called internally as the result
    of the :func:`.visitors.traverse` function.

    :param iterator: an iterable or sequence which will yield
     :class:`_expression.ClauseElement`
     structures; the iterator is assumed to be the
     product of the :func:`.visitors.iterate` function.

    :param obj: the :class:`_expression.ClauseElement`
     that was used as the target of the
     :func:`.iterate` function.

    :param visitors: dictionary of visit functions.  See :func:`.traverse`
     for details on this dictionary.

    .. seealso::

        :func:`.traverse`


    """
    for target in iterator:
        meth = visitors.get(target.__visit_name__, None)
        if meth:
            meth(target)
    return obj


@overload
def traverse(
    obj: Literal[None],
    opts: Mapping[str, Any],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> None: ...


@overload
def traverse(
    obj: ExternallyTraversible,
    opts: Mapping[str, Any],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> ExternallyTraversible: ...


def traverse(
    obj: Optional[ExternallyTraversible],
    opts: Mapping[str, Any],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> Optional[ExternallyTraversible]:
    """Traverse and visit the given expression structure using the default
    iterator.

     e.g.::

        from sqlalchemy.sql import visitors

        stmt = select(some_table).where(some_table.c.foo == "bar")


        def visit_bindparam(bind_param):
            print("found bound value: %s" % bind_param.value)


        visitors.traverse(stmt, {}, {"bindparam": visit_bindparam})

    The iteration of objects uses the :func:`.visitors.iterate` function,
    which does a breadth-first traversal using a stack.

    :param obj: :class:`_expression.ClauseElement` structure to be traversed

    :param opts: dictionary of iteration options.   This dictionary is usually
     empty in modern usage.

    :param visitors: dictionary of visit functions.   The dictionary should
     have strings as keys, each of which would correspond to the
     ``__visit_name__`` of a particular kind of SQL expression object, and
     callable functions  as values, each of which represents a visitor function
     for that kind of object.

    """
    return traverse_using(iterate(obj, opts), obj, visitors)


@overload
def cloned_traverse(
    obj: Literal[None],
    opts: Mapping[str, Any],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> None: ...


# a bit of controversy here, as the clone of the lead element
# *could* in theory replace with an entirely different kind of element.
# however this is really not how cloned_traverse is ever used internally
# at least.
@overload
def cloned_traverse(
    obj: _ET,
    opts: Mapping[str, Any],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> _ET: ...


def cloned_traverse(
    obj: Optional[ExternallyTraversible],
    opts: Mapping[str, Any],
    visitors: Mapping[str, _TraverseCallableType[Any]],
) -> Optional[ExternallyTraversible]:
    """Clone the given expression structure, allowing modifications by
    visitors for mutable objects.

    Traversal usage is the same as that of :func:`.visitors.traverse`.
    The visitor functions present in the ``visitors`` dictionary may also
    modify the internals of the given structure as the traversal proceeds.

    The :func:`.cloned_traverse` function does **not** provide objects that are
    part of the :class:`.Immutable` interface to the visit methods (this
    primarily includes :class:`.ColumnClause`, :class:`.Column`,
    :class:`.TableClause` and :class:`.Table` objects). As this traversal is
    only intended to allow in-place mutation of objects, :class:`.Immutable`
    objects are skipped. The :meth:`.Immutable._clone` method is still called
    on each object to allow for objects to replace themselves with a different
    object based on a clone of their sub-internals (e.g. a
    :class:`.ColumnClause` that clones its subquery to return a new
    :class:`.ColumnClause`).

    .. versionchanged:: 2.0  The :func:`.cloned_traverse` function omits
       objects that are part of the :class:`.Immutable` interface.

    The central API feature used by the :func:`.visitors.cloned_traverse`
    and :func:`.visitors.replacement_traverse` functions, in addition to the
    :meth:`_expression.ClauseElement.get_children`
    function that is used to achieve
    the iteration, is the :meth:`_expression.ClauseElement._copy_internals`
    method.
    For a :class:`_expression.ClauseElement`
    structure to support cloning and replacement
    traversals correctly, it needs to be able to pass a cloning function into
    its internal members in order to make copies of them.

    .. seealso::

        :func:`.visitors.traverse`

        :func:`.visitors.replacement_traverse`

    """

    cloned: Dict[int, ExternallyTraversible] = {}
    stop_on = set(opts.get("stop_on", []))

    def deferred_copy_internals(
        obj: ExternallyTraversible,
    ) -> ExternallyTraversible:
        return cloned_traverse(obj, opts, visitors)

    def clone(elem: ExternallyTraversible, **kw: Any) -> ExternallyTraversible:
        if elem in stop_on:
            return elem
        else:
            if id(elem) not in cloned:
                if "replace" in kw:
                    newelem = cast(
                        Optional[ExternallyTraversible], kw["replace"](elem)
                    )
                    if newelem is not None:
                        cloned[id(elem)] = newelem
                        return newelem

                # the _clone method for immutable normally returns "self".
                # however, the method is still allowed to return a
                # different object altogether; ColumnClause._clone() will
                # based on options clone the subquery to which it is associated
                # and return the new corresponding column.
                cloned[id(elem)] = newelem = elem._clone(clone=clone, **kw)
                newelem._copy_internals(clone=clone, **kw)

                # however, visit methods which are tasked with in-place
                # mutation of the object should not get access to the immutable
                # object.
                if not elem._is_immutable:
                    meth = visitors.get(newelem.__visit_name__, None)
                    if meth:
                        meth(newelem)
            return cloned[id(elem)]

    if obj is not None:
        obj = clone(
            obj, deferred_copy_internals=deferred_copy_internals, **opts
        )
    clone = None  # type: ignore[assignment]  # remove gc cycles
    return obj


@overload
def replacement_traverse(
    obj: Literal[None],
    opts: Mapping[str, Any],
    replace: _TraverseTransformCallableType[Any],
) -> None: ...


@overload
def replacement_traverse(
    obj: _CE,
    opts: Mapping[str, Any],
    replace: _TraverseTransformCallableType[Any],
) -> _CE: ...


@overload
def replacement_traverse(
    obj: ExternallyTraversible,
    opts: Mapping[str, Any],
    replace: _TraverseTransformCallableType[Any],
) -> ExternallyTraversible: ...


def replacement_traverse(
    obj: Optional[ExternallyTraversible],
    opts: Mapping[str, Any],
    replace: _TraverseTransformCallableType[Any],
) -> Optional[ExternallyTraversible]:
    """Clone the given expression structure, allowing element
    replacement by a given replacement function.

    This function is very similar to the :func:`.visitors.cloned_traverse`
    function, except instead of being passed a dictionary of visitors, all
    elements are unconditionally passed into the given replace function.
    The replace function then has the option to return an entirely new object
    which will replace the one given.  If it returns ``None``, then the object
    is kept in place.

    The difference in usage between :func:`.visitors.cloned_traverse` and
    :func:`.visitors.replacement_traverse` is that in the former case, an
    already-cloned object is passed to the visitor function, and the visitor
    function can then manipulate the internal state of the object.
    In the case of the latter, the visitor function should only return an
    entirely different object, or do nothing.

    The use case for :func:`.visitors.replacement_traverse` is that of
    replacing a FROM clause inside of a SQL structure with a different one,
    as is a common use case within the ORM.

    """

    cloned = {}
    stop_on = {id(x) for x in opts.get("stop_on", [])}

    def deferred_copy_internals(
        obj: ExternallyTraversible,
    ) -> ExternallyTraversible:
        return replacement_traverse(obj, opts, replace)

    def clone(elem: ExternallyTraversible, **kw: Any) -> ExternallyTraversible:
        if (
            id(elem) in stop_on
            or "no_replacement_traverse" in elem._annotations
        ):
            return elem
        else:
            newelem = replace(elem)
            if newelem is not None:
                stop_on.add(id(newelem))
                return newelem  # type: ignore
            else:
                # base "already seen" on id(), not hash, so that we don't
                # replace an Annotated element with its non-annotated one, and
                # vice versa
                id_elem = id(elem)
                if id_elem not in cloned:
                    if "replace" in kw:
                        newelem = kw["replace"](elem)
                        if newelem is not None:
                            cloned[id_elem] = newelem
                            return newelem  # type: ignore

                    cloned[id_elem] = newelem = elem._clone(**kw)
                    newelem._copy_internals(clone=clone, **kw)
                return cloned[id_elem]  # type: ignore

    if obj is not None:
        obj = clone(
            obj, deferred_copy_internals=deferred_copy_internals, **opts
        )
    clone = None  # type: ignore[assignment]  # remove gc cycles
    return obj

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\types.py ===
from __future__ import annotations

import collections.abc as cabc
import enum
import os
import stat
import sys
import typing as t
from datetime import datetime
from gettext import gettext as _
from gettext import ngettext

from ._compat import _get_argv_encoding
from ._compat import open_stream
from .exceptions import BadParameter
from .utils import format_filename
from .utils import LazyFile
from .utils import safecall

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .core import Context
    from .core import Parameter
    from .shell_completion import CompletionItem

ParamTypeValue = t.TypeVar("ParamTypeValue")


class ParamType:
    """Represents the type of a parameter. Validates and converts values
    from the command line or Python into the correct type.

    To implement a custom type, subclass and implement at least the
    following:

    -   The :attr:`name` class attribute must be set.
    -   Calling an instance of the type with ``None`` must return
        ``None``. This is already implemented by default.
    -   :meth:`convert` must convert string values to the correct type.
    -   :meth:`convert` must accept values that are already the correct
        type.
    -   It must be able to convert a value if the ``ctx`` and ``param``
        arguments are ``None``. This can occur when converting prompt
        input.
    """

    is_composite: t.ClassVar[bool] = False
    arity: t.ClassVar[int] = 1

    #: the descriptive name of this type
    name: str

    #: if a list of this type is expected and the value is pulled from a
    #: string environment variable, this is what splits it up.  `None`
    #: means any whitespace.  For all parameters the general rule is that
    #: whitespace splits them up.  The exception are paths and files which
    #: are split by ``os.path.pathsep`` by default (":" on Unix and ";" on
    #: Windows).
    envvar_list_splitter: t.ClassVar[str | None] = None

    def to_info_dict(self) -> dict[str, t.Any]:
        """Gather information that could be useful for a tool generating
        user-facing documentation.

        Use :meth:`click.Context.to_info_dict` to traverse the entire
        CLI structure.

        .. versionadded:: 8.0
        """
        # The class name without the "ParamType" suffix.
        param_type = type(self).__name__.partition("ParamType")[0]
        param_type = param_type.partition("ParameterType")[0]

        # Custom subclasses might not remember to set a name.
        if hasattr(self, "name"):
            name = self.name
        else:
            name = param_type

        return {"param_type": param_type, "name": name}

    def __call__(
        self,
        value: t.Any,
        param: Parameter | None = None,
        ctx: Context | None = None,
    ) -> t.Any:
        if value is not None:
            return self.convert(value, param, ctx)

    def get_metavar(self, param: Parameter, ctx: Context) -> str | None:
        """Returns the metavar default for this param if it provides one."""

    def get_missing_message(self, param: Parameter, ctx: Context | None) -> str | None:
        """Optionally might return extra information about a missing
        parameter.

        .. versionadded:: 2.0
        """

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        """Convert the value to the correct type. This is not called if
        the value is ``None`` (the missing value).

        This must accept string values from the command line, as well as
        values that are already the correct type. It may also convert
        other compatible types.

        The ``param`` and ``ctx`` arguments may be ``None`` in certain
        situations, such as when converting prompt input.

        If the value cannot be converted, call :meth:`fail` with a
        descriptive message.

        :param value: The value to convert.
        :param param: The parameter that is using this type to convert
            its value. May be ``None``.
        :param ctx: The current context that arrived at this value. May
            be ``None``.
        """
        return value

    def split_envvar_value(self, rv: str) -> cabc.Sequence[str]:
        """Given a value from an environment variable this splits it up
        into small chunks depending on the defined envvar list splitter.

        If the splitter is set to `None`, which means that whitespace splits,
        then leading and trailing whitespace is ignored.  Otherwise, leading
        and trailing splitters usually lead to empty items being included.
        """
        return (rv or "").split(self.envvar_list_splitter)

    def fail(
        self,
        message: str,
        param: Parameter | None = None,
        ctx: Context | None = None,
    ) -> t.NoReturn:
        """Helper method to fail with an invalid value message."""
        raise BadParameter(message, ctx=ctx, param=param)

    def shell_complete(
        self, ctx: Context, param: Parameter, incomplete: str
    ) -> list[CompletionItem]:
        """Return a list of
        :class:`~click.shell_completion.CompletionItem` objects for the
        incomplete value. Most types do not provide completions, but
        some do, and this allows custom types to provide custom
        completions as well.

        :param ctx: Invocation context for this command.
        :param param: The parameter that is requesting completion.
        :param incomplete: Value being completed. May be empty.

        .. versionadded:: 8.0
        """
        return []


class CompositeParamType(ParamType):
    is_composite = True

    @property
    def arity(self) -> int:  # type: ignore
        raise NotImplementedError()


class FuncParamType(ParamType):
    def __init__(self, func: t.Callable[[t.Any], t.Any]) -> None:
        self.name: str = func.__name__
        self.func = func

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict["func"] = self.func
        return info_dict

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        try:
            return self.func(value)
        except ValueError:
            try:
                value = str(value)
            except UnicodeError:
                value = value.decode("utf-8", "replace")

            self.fail(value, param, ctx)


class UnprocessedParamType(ParamType):
    name = "text"

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        return value

    def __repr__(self) -> str:
        return "UNPROCESSED"


class StringParamType(ParamType):
    name = "text"

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        if isinstance(value, bytes):
            enc = _get_argv_encoding()
            try:
                value = value.decode(enc)
            except UnicodeError:
                fs_enc = sys.getfilesystemencoding()
                if fs_enc != enc:
                    try:
                        value = value.decode(fs_enc)
                    except UnicodeError:
                        value = value.decode("utf-8", "replace")
                else:
                    value = value.decode("utf-8", "replace")
            return value
        return str(value)

    def __repr__(self) -> str:
        return "STRING"


class Choice(ParamType, t.Generic[ParamTypeValue]):
    """The choice type allows a value to be checked against a fixed set
    of supported values.

    You may pass any iterable value which will be converted to a tuple
    and thus will only be iterated once.

    The resulting value will always be one of the originally passed choices.
    See :meth:`normalize_choice` for more info on the mapping of strings
    to choices. See :ref:`choice-opts` for an example.

    :param case_sensitive: Set to false to make choices case
        insensitive. Defaults to true.

    .. versionchanged:: 8.2.0
        Non-``str`` ``choices`` are now supported. It can additionally be any
        iterable. Before you were not recommended to pass anything but a list or
        tuple.

    .. versionadded:: 8.2.0
        Choice normalization can be overridden via :meth:`normalize_choice`.
    """

    name = "choice"

    def __init__(
        self, choices: cabc.Iterable[ParamTypeValue], case_sensitive: bool = True
    ) -> None:
        self.choices: cabc.Sequence[ParamTypeValue] = tuple(choices)
        self.case_sensitive = case_sensitive

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict["choices"] = self.choices
        info_dict["case_sensitive"] = self.case_sensitive
        return info_dict

    def _normalized_mapping(
        self, ctx: Context | None = None
    ) -> cabc.Mapping[ParamTypeValue, str]:
        """
        Returns mapping where keys are the original choices and the values are
        the normalized values that are accepted via the command line.

        This is a simple wrapper around :meth:`normalize_choice`, use that
        instead which is supported.
        """
        return {
            choice: self.normalize_choice(
                choice=choice,
                ctx=ctx,
            )
            for choice in self.choices
        }

    def normalize_choice(self, choice: ParamTypeValue, ctx: Context | None) -> str:
        """
        Normalize a choice value, used to map a passed string to a choice.
        Each choice must have a unique normalized value.

        By default uses :meth:`Context.token_normalize_func` and if not case
        sensitive, convert it to a casefolded value.

        .. versionadded:: 8.2.0
        """
        normed_value = choice.name if isinstance(choice, enum.Enum) else str(choice)

        if ctx is not None and ctx.token_normalize_func is not None:
            normed_value = ctx.token_normalize_func(normed_value)

        if not self.case_sensitive:
            normed_value = normed_value.casefold()

        return normed_value

    def get_metavar(self, param: Parameter, ctx: Context) -> str | None:
        if param.param_type_name == "option" and not param.show_choices:  # type: ignore
            choice_metavars = [
                convert_type(type(choice)).name.upper() for choice in self.choices
            ]
            choices_str = "|".join([*dict.fromkeys(choice_metavars)])
        else:
            choices_str = "|".join(
                [str(i) for i in self._normalized_mapping(ctx=ctx).values()]
            )

        # Use curly braces to indicate a required argument.
        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"

        # Use square braces to indicate an option or optional argument.
        return f"[{choices_str}]"

    def get_missing_message(self, param: Parameter, ctx: Context | None) -> str:
        """
        Message shown when no choice is passed.

        .. versionchanged:: 8.2.0 Added ``ctx`` argument.
        """
        return _("Choose from:\n\t{choices}").format(
            choices=",\n\t".join(self._normalized_mapping(ctx=ctx).values())
        )

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> ParamTypeValue:
        """
        For a given value from the parser, normalize it and find its
        matching normalized value in the list of choices. Then return the
        matched "original" choice.
        """
        normed_value = self.normalize_choice(choice=value, ctx=ctx)
        normalized_mapping = self._normalized_mapping(ctx=ctx)

        try:
            return next(
                original
                for original, normalized in normalized_mapping.items()
                if normalized == normed_value
            )
        except StopIteration:
            self.fail(
                self.get_invalid_choice_message(value=value, ctx=ctx),
                param=param,
                ctx=ctx,
            )

    def get_invalid_choice_message(self, value: t.Any, ctx: Context | None) -> str:
        """Get the error message when the given choice is invalid.

        :param value: The invalid value.

        .. versionadded:: 8.2
        """
        choices_str = ", ".join(map(repr, self._normalized_mapping(ctx=ctx).values()))
        return ngettext(
            "{value!r} is not {choice}.",
            "{value!r} is not one of {choices}.",
            len(self.choices),
        ).format(value=value, choice=choices_str, choices=choices_str)

    def __repr__(self) -> str:
        return f"Choice({list(self.choices)})"

    def shell_complete(
        self, ctx: Context, param: Parameter, incomplete: str
    ) -> list[CompletionItem]:
        """Complete choices that start with the incomplete value.

        :param ctx: Invocation context for this command.
        :param param: The parameter that is requesting completion.
        :param incomplete: Value being completed. May be empty.

        .. versionadded:: 8.0
        """
        from click.shell_completion import CompletionItem

        str_choices = map(str, self.choices)

        if self.case_sensitive:
            matched = (c for c in str_choices if c.startswith(incomplete))
        else:
            incomplete = incomplete.lower()
            matched = (c for c in str_choices if c.lower().startswith(incomplete))

        return [CompletionItem(c) for c in matched]


class DateTime(ParamType):
    """The DateTime type converts date strings into `datetime` objects.

    The format strings which are checked are configurable, but default to some
    common (non-timezone aware) ISO 8601 formats.

    When specifying *DateTime* formats, you should only pass a list or a tuple.
    Other iterables, like generators, may lead to surprising results.

    The format strings are processed using ``datetime.strptime``, and this
    consequently defines the format strings which are allowed.

    Parsing is tried using each format, in order, and the first format which
    parses successfully is used.

    :param formats: A list or tuple of date format strings, in the order in
                    which they should be tried. Defaults to
                    ``'%Y-%m-%d'``, ``'%Y-%m-%dT%H:%M:%S'``,
                    ``'%Y-%m-%d %H:%M:%S'``.
    """

    name = "datetime"

    def __init__(self, formats: cabc.Sequence[str] | None = None):
        self.formats: cabc.Sequence[str] = formats or [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict["formats"] = self.formats
        return info_dict

    def get_metavar(self, param: Parameter, ctx: Context) -> str | None:
        return f"[{'|'.join(self.formats)}]"

    def _try_to_convert_date(self, value: t.Any, format: str) -> datetime | None:
        try:
            return datetime.strptime(value, format)
        except ValueError:
            return None

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        if isinstance(value, datetime):
            return value

        for format in self.formats:
            converted = self._try_to_convert_date(value, format)

            if converted is not None:
                return converted

        formats_str = ", ".join(map(repr, self.formats))
        self.fail(
            ngettext(
                "{value!r} does not match the format {format}.",
                "{value!r} does not match the formats {formats}.",
                len(self.formats),
            ).format(value=value, format=formats_str, formats=formats_str),
            param,
            ctx,
        )

    def __repr__(self) -> str:
        return "DateTime"


class _NumberParamTypeBase(ParamType):
    _number_class: t.ClassVar[type[t.Any]]

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        try:
            return self._number_class(value)
        except ValueError:
            self.fail(
                _("{value!r} is not a valid {number_type}.").format(
                    value=value, number_type=self.name
                ),
                param,
                ctx,
            )


class _NumberRangeBase(_NumberParamTypeBase):
    def __init__(
        self,
        min: float | None = None,
        max: float | None = None,
        min_open: bool = False,
        max_open: bool = False,
        clamp: bool = False,
    ) -> None:
        self.min = min
        self.max = max
        self.min_open = min_open
        self.max_open = max_open
        self.clamp = clamp

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict.update(
            min=self.min,
            max=self.max,
            min_open=self.min_open,
            max_open=self.max_open,
            clamp=self.clamp,
        )
        return info_dict

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        import operator

        rv = super().convert(value, param, ctx)
        lt_min: bool = self.min is not None and (
            operator.le if self.min_open else operator.lt
        )(rv, self.min)
        gt_max: bool = self.max is not None and (
            operator.ge if self.max_open else operator.gt
        )(rv, self.max)

        if self.clamp:
            if lt_min:
                return self._clamp(self.min, 1, self.min_open)  # type: ignore

            if gt_max:
                return self._clamp(self.max, -1, self.max_open)  # type: ignore

        if lt_min or gt_max:
            self.fail(
                _("{value} is not in the range {range}.").format(
                    value=rv, range=self._describe_range()
                ),
                param,
                ctx,
            )

        return rv

    def _clamp(self, bound: float, dir: t.Literal[1, -1], open: bool) -> float:
        """Find the valid value to clamp to bound in the given
        direction.

        :param bound: The boundary value.
        :param dir: 1 or -1 indicating the direction to move.
        :param open: If true, the range does not include the bound.
        """
        raise NotImplementedError

    def _describe_range(self) -> str:
        """Describe the range for use in help text."""
        if self.min is None:
            op = "<" if self.max_open else "<="
            return f"x{op}{self.max}"

        if self.max is None:
            op = ">" if self.min_open else ">="
            return f"x{op}{self.min}"

        lop = "<" if self.min_open else "<="
        rop = "<" if self.max_open else "<="
        return f"{self.min}{lop}x{rop}{self.max}"

    def __repr__(self) -> str:
        clamp = " clamped" if self.clamp else ""
        return f"<{type(self).__name__} {self._describe_range()}{clamp}>"


class IntParamType(_NumberParamTypeBase):
    name = "integer"
    _number_class = int

    def __repr__(self) -> str:
        return "INT"


class IntRange(_NumberRangeBase, IntParamType):
    """Restrict an :data:`click.INT` value to a range of accepted
    values. See :ref:`ranges`.

    If ``min`` or ``max`` are not passed, any value is accepted in that
    direction. If ``min_open`` or ``max_open`` are enabled, the
    corresponding boundary is not included in the range.

    If ``clamp`` is enabled, a value outside the range is clamped to the
    boundary instead of failing.

    .. versionchanged:: 8.0
        Added the ``min_open`` and ``max_open`` parameters.
    """

    name = "integer range"

    def _clamp(  # type: ignore
        self, bound: int, dir: t.Literal[1, -1], open: bool
    ) -> int:
        if not open:
            return bound

        return bound + dir


class FloatParamType(_NumberParamTypeBase):
    name = "float"
    _number_class = float

    def __repr__(self) -> str:
        return "FLOAT"


class FloatRange(_NumberRangeBase, FloatParamType):
    """Restrict a :data:`click.FLOAT` value to a range of accepted
    values. See :ref:`ranges`.

    If ``min`` or ``max`` are not passed, any value is accepted in that
    direction. If ``min_open`` or ``max_open`` are enabled, the
    corresponding boundary is not included in the range.

    If ``clamp`` is enabled, a value outside the range is clamped to the
    boundary instead of failing. This is not supported if either
    boundary is marked ``open``.

    .. versionchanged:: 8.0
        Added the ``min_open`` and ``max_open`` parameters.
    """

    name = "float range"

    def __init__(
        self,
        min: float | None = None,
        max: float | None = None,
        min_open: bool = False,
        max_open: bool = False,
        clamp: bool = False,
    ) -> None:
        super().__init__(
            min=min, max=max, min_open=min_open, max_open=max_open, clamp=clamp
        )

        if (min_open or max_open) and clamp:
            raise TypeError("Clamping is not supported for open bounds.")

    def _clamp(self, bound: float, dir: t.Literal[1, -1], open: bool) -> float:
        if not open:
            return bound

        # Could use math.nextafter here, but clamping an
        # open float range doesn't seem to be particularly useful. It's
        # left up to the user to write a callback to do it if needed.
        raise RuntimeError("Clamping is not supported for open bounds.")


class BoolParamType(ParamType):
    name = "boolean"

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        if value in {False, True}:
            return bool(value)

        norm = value.strip().lower()

        if norm in {"1", "true", "t", "yes", "y", "on"}:
            return True

        if norm in {"0", "false", "f", "no", "n", "off"}:
            return False

        self.fail(
            _("{value!r} is not a valid boolean.").format(value=value), param, ctx
        )

    def __repr__(self) -> str:
        return "BOOL"


class UUIDParameterType(ParamType):
    name = "uuid"

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        import uuid

        if isinstance(value, uuid.UUID):
            return value

        value = value.strip()

        try:
            return uuid.UUID(value)
        except ValueError:
            self.fail(
                _("{value!r} is not a valid UUID.").format(value=value), param, ctx
            )

    def __repr__(self) -> str:
        return "UUID"


class File(ParamType):
    """Declares a parameter to be a file for reading or writing.  The file
    is automatically closed once the context tears down (after the command
    finished working).

    Files can be opened for reading or writing.  The special value ``-``
    indicates stdin or stdout depending on the mode.

    By default, the file is opened for reading text data, but it can also be
    opened in binary mode or for writing.  The encoding parameter can be used
    to force a specific encoding.

    The `lazy` flag controls if the file should be opened immediately or upon
    first IO. The default is to be non-lazy for standard input and output
    streams as well as files opened for reading, `lazy` otherwise. When opening a
    file lazily for reading, it is still opened temporarily for validation, but
    will not be held open until first IO. lazy is mainly useful when opening
    for writing to avoid creating the file until it is needed.

    Files can also be opened atomically in which case all writes go into a
    separate file in the same folder and upon completion the file will
    be moved over to the original location.  This is useful if a file
    regularly read by other users is modified.

    See :ref:`file-args` for more information.

    .. versionchanged:: 2.0
        Added the ``atomic`` parameter.
    """

    name = "filename"
    envvar_list_splitter: t.ClassVar[str] = os.path.pathsep

    def __init__(
        self,
        mode: str = "r",
        encoding: str | None = None,
        errors: str | None = "strict",
        lazy: bool | None = None,
        atomic: bool = False,
    ) -> None:
        self.mode = mode
        self.encoding = encoding
        self.errors = errors
        self.lazy = lazy
        self.atomic = atomic

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict.update(mode=self.mode, encoding=self.encoding)
        return info_dict

    def resolve_lazy_flag(self, value: str | os.PathLike[str]) -> bool:
        if self.lazy is not None:
            return self.lazy
        if os.fspath(value) == "-":
            return False
        elif "w" in self.mode:
            return True
        return False

    def convert(
        self,
        value: str | os.PathLike[str] | t.IO[t.Any],
        param: Parameter | None,
        ctx: Context | None,
    ) -> t.IO[t.Any]:
        if _is_file_like(value):
            return value

        value = t.cast("str | os.PathLike[str]", value)

        try:
            lazy = self.resolve_lazy_flag(value)

            if lazy:
                lf = LazyFile(
                    value, self.mode, self.encoding, self.errors, atomic=self.atomic
                )

                if ctx is not None:
                    ctx.call_on_close(lf.close_intelligently)

                return t.cast("t.IO[t.Any]", lf)

            f, should_close = open_stream(
                value, self.mode, self.encoding, self.errors, atomic=self.atomic
            )

            # If a context is provided, we automatically close the file
            # at the end of the context execution (or flush out).  If a
            # context does not exist, it's the caller's responsibility to
            # properly close the file.  This for instance happens when the
            # type is used with prompts.
            if ctx is not None:
                if should_close:
                    ctx.call_on_close(safecall(f.close))
                else:
                    ctx.call_on_close(safecall(f.flush))

            return f
        except OSError as e:
            self.fail(f"'{format_filename(value)}': {e.strerror}", param, ctx)

    def shell_complete(
        self, ctx: Context, param: Parameter, incomplete: str
    ) -> list[CompletionItem]:
        """Return a special completion marker that tells the completion
        system to use the shell to provide file path completions.

        :param ctx: Invocation context for this command.
        :param param: The parameter that is requesting completion.
        :param incomplete: Value being completed. May be empty.

        .. versionadded:: 8.0
        """
        from click.shell_completion import CompletionItem

        return [CompletionItem(incomplete, type="file")]


def _is_file_like(value: t.Any) -> te.TypeGuard[t.IO[t.Any]]:
    return hasattr(value, "read") or hasattr(value, "write")


class Path(ParamType):
    """The ``Path`` type is similar to the :class:`File` type, but
    returns the filename instead of an open file. Various checks can be
    enabled to validate the type of file and permissions.

    :param exists: The file or directory needs to exist for the value to
        be valid. If this is not set to ``True``, and the file does not
        exist, then all further checks are silently skipped.
    :param file_okay: Allow a file as a value.
    :param dir_okay: Allow a directory as a value.
    :param readable: if true, a readable check is performed.
    :param writable: if true, a writable check is performed.
    :param executable: if true, an executable check is performed.
    :param resolve_path: Make the value absolute and resolve any
        symlinks. A ``~`` is not expanded, as this is supposed to be
        done by the shell only.
    :param allow_dash: Allow a single dash as a value, which indicates
        a standard stream (but does not open it). Use
        :func:`~click.open_file` to handle opening this value.
    :param path_type: Convert the incoming path value to this type. If
        ``None``, keep Python's default, which is ``str``. Useful to
        convert to :class:`pathlib.Path`.

    .. versionchanged:: 8.1
        Added the ``executable`` parameter.

    .. versionchanged:: 8.0
        Allow passing ``path_type=pathlib.Path``.

    .. versionchanged:: 6.0
        Added the ``allow_dash`` parameter.
    """

    envvar_list_splitter: t.ClassVar[str] = os.path.pathsep

    def __init__(
        self,
        exists: bool = False,
        file_okay: bool = True,
        dir_okay: bool = True,
        writable: bool = False,
        readable: bool = True,
        resolve_path: bool = False,
        allow_dash: bool = False,
        path_type: type[t.Any] | None = None,
        executable: bool = False,
    ):
        self.exists = exists
        self.file_okay = file_okay
        self.dir_okay = dir_okay
        self.readable = readable
        self.writable = writable
        self.executable = executable
        self.resolve_path = resolve_path
        self.allow_dash = allow_dash
        self.type = path_type

        if self.file_okay and not self.dir_okay:
            self.name: str = _("file")
        elif self.dir_okay and not self.file_okay:
            self.name = _("directory")
        else:
            self.name = _("path")

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict.update(
            exists=self.exists,
            file_okay=self.file_okay,
            dir_okay=self.dir_okay,
            writable=self.writable,
            readable=self.readable,
            allow_dash=self.allow_dash,
        )
        return info_dict

    def coerce_path_result(
        self, value: str | os.PathLike[str]
    ) -> str | bytes | os.PathLike[str]:
        if self.type is not None and not isinstance(value, self.type):
            if self.type is str:
                return os.fsdecode(value)
            elif self.type is bytes:
                return os.fsencode(value)
            else:
                return t.cast("os.PathLike[str]", self.type(value))

        return value

    def convert(
        self,
        value: str | os.PathLike[str],
        param: Parameter | None,
        ctx: Context | None,
    ) -> str | bytes | os.PathLike[str]:
        rv = value

        is_dash = self.file_okay and self.allow_dash and rv in (b"-", "-")

        if not is_dash:
            if self.resolve_path:
                rv = os.path.realpath(rv)

            try:
                st = os.stat(rv)
            except OSError:
                if not self.exists:
                    return self.coerce_path_result(rv)
                self.fail(
                    _("{name} {filename!r} does not exist.").format(
                        name=self.name.title(), filename=format_filename(value)
                    ),
                    param,
                    ctx,
                )

            if not self.file_okay and stat.S_ISREG(st.st_mode):
                self.fail(
                    _("{name} {filename!r} is a file.").format(
                        name=self.name.title(), filename=format_filename(value)
                    ),
                    param,
                    ctx,
                )
            if not self.dir_okay and stat.S_ISDIR(st.st_mode):
                self.fail(
                    _("{name} {filename!r} is a directory.").format(
                        name=self.name.title(), filename=format_filename(value)
                    ),
                    param,
                    ctx,
                )

            if self.readable and not os.access(rv, os.R_OK):
                self.fail(
                    _("{name} {filename!r} is not readable.").format(
                        name=self.name.title(), filename=format_filename(value)
                    ),
                    param,
                    ctx,
                )

            if self.writable and not os.access(rv, os.W_OK):
                self.fail(
                    _("{name} {filename!r} is not writable.").format(
                        name=self.name.title(), filename=format_filename(value)
                    ),
                    param,
                    ctx,
                )

            if self.executable and not os.access(value, os.X_OK):
                self.fail(
                    _("{name} {filename!r} is not executable.").format(
                        name=self.name.title(), filename=format_filename(value)
                    ),
                    param,
                    ctx,
                )

        return self.coerce_path_result(rv)

    def shell_complete(
        self, ctx: Context, param: Parameter, incomplete: str
    ) -> list[CompletionItem]:
        """Return a special completion marker that tells the completion
        system to use the shell to provide path completions for only
        directories or any paths.

        :param ctx: Invocation context for this command.
        :param param: The parameter that is requesting completion.
        :param incomplete: Value being completed. May be empty.

        .. versionadded:: 8.0
        """
        from click.shell_completion import CompletionItem

        type = "dir" if self.dir_okay and not self.file_okay else "file"
        return [CompletionItem(incomplete, type=type)]


class Tuple(CompositeParamType):
    """The default behavior of Click is to apply a type on a value directly.
    This works well in most cases, except for when `nargs` is set to a fixed
    count and different types should be used for different items.  In this
    case the :class:`Tuple` type can be used.  This type can only be used
    if `nargs` is set to a fixed number.

    For more information see :ref:`tuple-type`.

    This can be selected by using a Python tuple literal as a type.

    :param types: a list of types that should be used for the tuple items.
    """

    def __init__(self, types: cabc.Sequence[type[t.Any] | ParamType]) -> None:
        self.types: cabc.Sequence[ParamType] = [convert_type(ty) for ty in types]

    def to_info_dict(self) -> dict[str, t.Any]:
        info_dict = super().to_info_dict()
        info_dict["types"] = [t.to_info_dict() for t in self.types]
        return info_dict

    @property
    def name(self) -> str:  # type: ignore
        return f"<{' '.join(ty.name for ty in self.types)}>"

    @property
    def arity(self) -> int:  # type: ignore
        return len(self.types)

    def convert(
        self, value: t.Any, param: Parameter | None, ctx: Context | None
    ) -> t.Any:
        len_type = len(self.types)
        len_value = len(value)

        if len_value != len_type:
            self.fail(
                ngettext(
                    "{len_type} values are required, but {len_value} was given.",
                    "{len_type} values are required, but {len_value} were given.",
                    len_value,
                ).format(len_type=len_type, len_value=len_value),
                param=param,
                ctx=ctx,
            )

        return tuple(
            ty(x, param, ctx) for ty, x in zip(self.types, value, strict=False)
        )


def convert_type(ty: t.Any | None, default: t.Any | None = None) -> ParamType:
    """Find the most appropriate :class:`ParamType` for the given Python
    type. If the type isn't provided, it can be inferred from a default
    value.
    """
    guessed_type = False

    if ty is None and default is not None:
        if isinstance(default, (tuple, list)):
            # If the default is empty, ty will remain None and will
            # return STRING.
            if default:
                item = default[0]

                # A tuple of tuples needs to detect the inner types.
                # Can't call convert recursively because that would
                # incorrectly unwind the tuple to a single type.
                if isinstance(item, (tuple, list)):
                    ty = tuple(map(type, item))
                else:
                    ty = type(item)
        else:
            ty = type(default)

        guessed_type = True

    if isinstance(ty, tuple):
        return Tuple(ty)

    if isinstance(ty, ParamType):
        return ty

    if ty is str or ty is None:
        return STRING

    if ty is int:
        return INT

    if ty is float:
        return FLOAT

    if ty is bool:
        return BOOL

    if guessed_type:
        return STRING

    if __debug__:
        try:
            if issubclass(ty, ParamType):
                raise AssertionError(
                    f"Attempted to use an uninstantiated parameter type ({ty})."
                )
        except TypeError:
            # ty is an instance (correct), so issubclass fails.
            pass

    return FuncParamType(ty)


#: A dummy parameter type that just does nothing.  From a user's
#: perspective this appears to just be the same as `STRING` but
#: internally no string conversion takes place if the input was bytes.
#: This is usually useful when working with file paths as they can
#: appear in bytes and unicode.
#:
#: For path related uses the :class:`Path` type is a better choice but
#: there are situations where an unprocessed type is useful which is why
#: it is is provided.
#:
#: .. versionadded:: 4.0
UNPROCESSED = UnprocessedParamType()

#: A unicode string parameter type which is the implicit default.  This
#: can also be selected by using ``str`` as type.
STRING = StringParamType()

#: An integer parameter.  This can also be selected by using ``int`` as
#: type.
INT = IntParamType()

#: A floating point value parameter.  This can also be selected by using
#: ``float`` as type.
FLOAT = FloatParamType()

#: A boolean parameter.  This is the default for boolean flags.  This can
#: also be selected by using ``bool`` as a type.
BOOL = BoolParamType()

#: A UUID parameter.
UUID = UUIDParameterType()


class OptionHelpExtra(t.TypedDict, total=False):
    envvars: tuple[str, ...]
    default: str
    range: str
    required: str

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\missing.py ===
"""
Routines for filling missing data.
"""
from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    NaT,
    algos,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    F,
    ReindexMethod,
    npt,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_numeric_dtype,
    is_numeric_v_string_like,
    is_object_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

if TYPE_CHECKING:
    from pandas import Index


def check_value_size(value, mask: npt.NDArray[np.bool_], length: int):
    """
    Validate the size of the values passed to ExtensionArray.fillna.
    """
    if is_array_like(value):
        if len(value) != length:
            raise ValueError(
                f"Length of 'value' does not match. Got ({len(value)}) "
                f" expected {length}"
            )
        value = value[mask]

    return value


def mask_missing(arr: ArrayLike, values_to_mask) -> npt.NDArray[np.bool_]:
    """
    Return a masking array of same size/shape as arr
    with entries equaling any member of values_to_mask set to True

    Parameters
    ----------
    arr : ArrayLike
    values_to_mask: list, tuple, or scalar

    Returns
    -------
    np.ndarray[bool]
    """
    # When called from Block.replace/replace_list, values_to_mask is a scalar
    #  known to be holdable by arr.
    # When called from Series._single_replace, values_to_mask is tuple or list
    dtype, values_to_mask = infer_dtype_from(values_to_mask)

    if isinstance(dtype, np.dtype):
        values_to_mask = np.array(values_to_mask, dtype=dtype)
    else:
        cls = dtype.construct_array_type()
        if not lib.is_list_like(values_to_mask):
            values_to_mask = [values_to_mask]
        values_to_mask = cls._from_sequence(values_to_mask, dtype=dtype, copy=False)

    potential_na = False
    if is_object_dtype(arr.dtype):
        # pre-compute mask to avoid comparison to NA
        potential_na = True
        arr_mask = ~isna(arr)

    na_mask = isna(values_to_mask)
    nonna = values_to_mask[~na_mask]

    # GH 21977
    mask = np.zeros(arr.shape, dtype=bool)
    if (
        is_numeric_dtype(arr.dtype)
        and not is_bool_dtype(arr.dtype)
        and is_bool_dtype(nonna.dtype)
    ):
        pass
    elif (
        is_bool_dtype(arr.dtype)
        and is_numeric_dtype(nonna.dtype)
        and not is_bool_dtype(nonna.dtype)
    ):
        pass
    else:
        for x in nonna:
            if is_numeric_v_string_like(arr, x):
                # GH#29553 prevent numpy deprecation warnings
                pass
            else:
                if potential_na:
                    new_mask = np.zeros(arr.shape, dtype=np.bool_)
                    new_mask[arr_mask] = arr[arr_mask] == x
                else:
                    new_mask = arr == x

                    if not isinstance(new_mask, np.ndarray):
                        # usually BooleanArray
                        new_mask = new_mask.to_numpy(dtype=bool, na_value=False)
                mask |= new_mask

    if na_mask.any():
        mask |= isna(arr)

    return mask


@overload
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill"],
    *,
    allow_nearest: Literal[False] = ...,
) -> Literal["pad", "backfill"]:
    ...


@overload
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: Literal[True],
) -> Literal["pad", "backfill", "nearest"]:
    ...


def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: bool = False,
) -> Literal["pad", "backfill", "nearest"]:
    if isinstance(method, str):
        # error: Incompatible types in assignment (expression has type "str", variable
        # has type "Literal['ffill', 'pad', 'bfill', 'backfill', 'nearest']")
        method = method.lower()  # type: ignore[assignment]
        if method == "ffill":
            method = "pad"
        elif method == "bfill":
            method = "backfill"

    valid_methods = ["pad", "backfill"]
    expecting = "pad (ffill) or backfill (bfill)"
    if allow_nearest:
        valid_methods.append("nearest")
        expecting = "pad (ffill), backfill (bfill) or nearest"
    if method not in valid_methods:
        raise ValueError(f"Invalid fill method. Expecting {expecting}. Got {method}")
    return method


# interpolation methods that dispatch to np.interp

NP_METHODS = ["linear", "time", "index", "values"]

# interpolation methods that dispatch to _interpolate_scipy_wrapper

SP_METHODS = [
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "krogh",
    "spline",
    "polynomial",
    "from_derivatives",
    "piecewise_polynomial",
    "pchip",
    "akima",
    "cubicspline",
]


def clean_interp_method(method: str, index: Index, **kwargs) -> str:
    order = kwargs.get("order")

    if method in ("spline", "polynomial") and order is None:
        raise ValueError("You must specify the order of the spline or polynomial.")

    valid = NP_METHODS + SP_METHODS
    if method not in valid:
        raise ValueError(f"method must be one of {valid}. Got '{method}' instead.")

    if method in ("krogh", "piecewise_polynomial", "pchip"):
        if not index.is_monotonic_increasing:
            raise ValueError(
                f"{method} interpolation requires that the index be monotonic."
            )

    return method


def find_valid_index(how: str, is_valid: npt.NDArray[np.bool_]) -> int | None:
    """
    Retrieves the positional index of the first valid value.

    Parameters
    ----------
    how : {'first', 'last'}
        Use this parameter to change between the first or last valid index.
    is_valid: np.ndarray
        Mask to find na_values.

    Returns
    -------
    int or None
    """
    assert how in ["first", "last"]

    if len(is_valid) == 0:  # early stop
        return None

    if is_valid.ndim == 2:
        is_valid = is_valid.any(axis=1)  # reduce axis 1

    if how == "first":
        idxpos = is_valid[::].argmax()

    elif how == "last":
        idxpos = len(is_valid) - 1 - is_valid[::-1].argmax()

    chk_notna = is_valid[idxpos]

    if not chk_notna:
        return None
    # Incompatible return value type (got "signedinteger[Any]",
    # expected "Optional[int]")
    return idxpos  # type: ignore[return-value]


def validate_limit_direction(
    limit_direction: str,
) -> Literal["forward", "backward", "both"]:
    valid_limit_directions = ["forward", "backward", "both"]
    limit_direction = limit_direction.lower()
    if limit_direction not in valid_limit_directions:
        raise ValueError(
            "Invalid limit_direction: expecting one of "
            f"{valid_limit_directions}, got '{limit_direction}'."
        )
    # error: Incompatible return value type (got "str", expected
    # "Literal['forward', 'backward', 'both']")
    return limit_direction  # type: ignore[return-value]


def validate_limit_area(limit_area: str | None) -> Literal["inside", "outside"] | None:
    if limit_area is not None:
        valid_limit_areas = ["inside", "outside"]
        limit_area = limit_area.lower()
        if limit_area not in valid_limit_areas:
            raise ValueError(
                f"Invalid limit_area: expecting one of {valid_limit_areas}, got "
                f"{limit_area}."
            )
    # error: Incompatible return value type (got "Optional[str]", expected
    # "Optional[Literal['inside', 'outside']]")
    return limit_area  # type: ignore[return-value]


def infer_limit_direction(
    limit_direction: Literal["backward", "forward", "both"] | None, method: str
) -> Literal["backward", "forward", "both"]:
    # Set `limit_direction` depending on `method`
    if limit_direction is None:
        if method in ("backfill", "bfill"):
            limit_direction = "backward"
        else:
            limit_direction = "forward"
    else:
        if method in ("pad", "ffill") and limit_direction != "forward":
            raise ValueError(
                f"`limit_direction` must be 'forward' for method `{method}`"
            )
        if method in ("backfill", "bfill") and limit_direction != "backward":
            raise ValueError(
                f"`limit_direction` must be 'backward' for method `{method}`"
            )
    return limit_direction


def get_interp_index(method, index: Index) -> Index:
    # create/use the index
    if method == "linear":
        # prior default
        from pandas import Index

        index = Index(np.arange(len(index)))
    else:
        methods = {"index", "values", "nearest", "time"}
        is_numeric_or_datetime = (
            is_numeric_dtype(index.dtype)
            or isinstance(index.dtype, DatetimeTZDtype)
            or lib.is_np_dtype(index.dtype, "mM")
        )
        if method not in methods and not is_numeric_or_datetime:
            raise ValueError(
                "Index column must be numeric or datetime type when "
                f"using {method} method other than linear. "
                "Try setting a numeric or datetime index column before "
                "interpolating."
            )

    if isna(index).any():
        raise NotImplementedError(
            "Interpolation with NaNs in the index "
            "has not been implemented. Try filling "
            "those NaNs before interpolating."
        )
    return index


def interpolate_2d_inplace(
    data: np.ndarray,  # floating dtype
    index: Index,
    axis: AxisInt,
    method: str = "linear",
    limit: int | None = None,
    limit_direction: str = "forward",
    limit_area: str | None = None,
    fill_value: Any | None = None,
    mask=None,
    **kwargs,
) -> None:
    """
    Column-wise application of _interpolate_1d.

    Notes
    -----
    Alters 'data' in-place.

    The signature does differ from _interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """
    # validate the interp method
    clean_interp_method(method, index, **kwargs)

    if is_valid_na_for_dtype(fill_value, data.dtype):
        fill_value = na_value_for_dtype(data.dtype, compat=False)

    if method == "time":
        if not needs_i8_conversion(index.dtype):
            raise ValueError(
                "time-weighted interpolation only works "
                "on Series or DataFrames with a "
                "DatetimeIndex"
            )
        method = "values"

    limit_direction = validate_limit_direction(limit_direction)
    limit_area_validated = validate_limit_area(limit_area)

    # default limit is unlimited GH #16282
    limit = algos.validate_limit(nobs=None, limit=limit)

    indices = _index_to_interp_indices(index, method)

    def func(yvalues: np.ndarray) -> None:
        # process 1-d slices in the axis direction

        _interpolate_1d(
            indices=indices,
            yvalues=yvalues,
            method=method,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area_validated,
            fill_value=fill_value,
            bounds_error=False,
            mask=mask,
            **kwargs,
        )

    # error: Argument 1 to "apply_along_axis" has incompatible type
    # "Callable[[ndarray[Any, Any]], None]"; expected "Callable[...,
    # Union[_SupportsArray[dtype[<nothing>]], Sequence[_SupportsArray
    # [dtype[<nothing>]]], Sequence[Sequence[_SupportsArray[dtype[<nothing>]]]],
    # Sequence[Sequence[Sequence[_SupportsArray[dtype[<nothing>]]]]],
    # Sequence[Sequence[Sequence[Sequence[_SupportsArray[dtype[<nothing>]]]]]]]]"
    np.apply_along_axis(func, axis, data)  # type: ignore[arg-type]


def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
    xarr = index._values
    if needs_i8_conversion(xarr.dtype):
        # GH#1646 for dt64tz
        xarr = xarr.view("i8")

    if method == "linear":
        inds = xarr
        inds = cast(np.ndarray, inds)
    else:
        inds = np.asarray(xarr)

        if method in ("values", "index"):
            if inds.dtype == np.object_:
                inds = lib.maybe_convert_objects(inds)

    return inds


def _interpolate_1d(
    indices: np.ndarray,
    yvalues: np.ndarray,
    method: str = "linear",
    limit: int | None = None,
    limit_direction: str = "forward",
    limit_area: Literal["inside", "outside"] | None = None,
    fill_value: Any | None = None,
    bounds_error: bool = False,
    order: int | None = None,
    mask=None,
    **kwargs,
) -> None:
    """
    Logic for the 1-d interpolation.  The input
    indices and yvalues will each be 1-d arrays of the same length.

    Bounds_error is currently hardcoded to False since non-scipy ones don't
    take it as an argument.

    Notes
    -----
    Fills 'yvalues' in-place.
    """
    if mask is not None:
        invalid = mask
    else:
        invalid = isna(yvalues)
    valid = ~invalid

    if not valid.any():
        return

    if valid.all():
        return

    # These are sets of index pointers to invalid values... i.e. {0, 1, etc...
    all_nans = set(np.flatnonzero(invalid))

    first_valid_index = find_valid_index(how="first", is_valid=valid)
    if first_valid_index is None:  # no nan found in start
        first_valid_index = 0
    start_nans = set(range(first_valid_index))

    last_valid_index = find_valid_index(how="last", is_valid=valid)
    if last_valid_index is None:  # no nan found in end
        last_valid_index = len(yvalues)
    end_nans = set(range(1 + last_valid_index, len(valid)))

    # Like the sets above, preserve_nans contains indices of invalid values,
    # but in this case, it is the final set of indices that need to be
    # preserved as NaN after the interpolation.

    # For example if limit_direction='forward' then preserve_nans will
    # contain indices of NaNs at the beginning of the series, and NaNs that
    # are more than 'limit' away from the prior non-NaN.

    # set preserve_nans based on direction using _interp_limit
    preserve_nans: list | set
    if limit_direction == "forward":
        preserve_nans = start_nans | set(_interp_limit(invalid, limit, 0))
    elif limit_direction == "backward":
        preserve_nans = end_nans | set(_interp_limit(invalid, 0, limit))
    else:
        # both directions... just use _interp_limit
        preserve_nans = set(_interp_limit(invalid, limit, limit))

    # if limit_area is set, add either mid or outside indices
    # to preserve_nans GH #16284
    if limit_area == "inside":
        # preserve NaNs on the outside
        preserve_nans |= start_nans | end_nans
    elif limit_area == "outside":
        # preserve NaNs on the inside
        mid_nans = all_nans - start_nans - end_nans
        preserve_nans |= mid_nans

    # sort preserve_nans and convert to list
    preserve_nans = sorted(preserve_nans)

    is_datetimelike = yvalues.dtype.kind in "mM"

    if is_datetimelike:
        yvalues = yvalues.view("i8")

    if method in NP_METHODS:
        # np.interp requires sorted X values, #21037

        indexer = np.argsort(indices[valid])
        yvalues[invalid] = np.interp(
            indices[invalid], indices[valid][indexer], yvalues[valid][indexer]
        )
    else:
        yvalues[invalid] = _interpolate_scipy_wrapper(
            indices[valid],
            yvalues[valid],
            indices[invalid],
            method=method,
            fill_value=fill_value,
            bounds_error=bounds_error,
            order=order,
            **kwargs,
        )

    if mask is not None:
        mask[:] = False
        mask[preserve_nans] = True
    elif is_datetimelike:
        yvalues[preserve_nans] = NaT.value
    else:
        yvalues[preserve_nans] = np.nan
    return


def _interpolate_scipy_wrapper(
    x: np.ndarray,
    y: np.ndarray,
    new_x: np.ndarray,
    method: str,
    fill_value=None,
    bounds_error: bool = False,
    order=None,
    **kwargs,
):
    """
    Passed off to scipy.interpolate.interp1d. method is scipy's kind.
    Returns an array interpolated at new_x.  Add any new methods to
    the list in _clean_interp_method.
    """
    extra = f"{method} interpolation requires SciPy."
    import_optional_dependency("scipy", extra=extra)
    from scipy import interpolate

    new_x = np.asarray(new_x)

    # ignores some kwargs that could be passed along.
    alt_methods = {
        "barycentric": interpolate.barycentric_interpolate,
        "krogh": interpolate.krogh_interpolate,
        "from_derivatives": _from_derivatives,
        "piecewise_polynomial": _from_derivatives,
        "cubicspline": _cubicspline_interpolate,
        "akima": _akima_interpolate,
        "pchip": interpolate.pchip_interpolate,
    }

    interp1d_methods = [
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "polynomial",
    ]
    if method in interp1d_methods:
        if method == "polynomial":
            kind = order
        else:
            kind = method
        terp = interpolate.interp1d(
            x, y, kind=kind, fill_value=fill_value, bounds_error=bounds_error
        )
        new_y = terp(new_x)
    elif method == "spline":
        # GH #10633, #24014
        if isna(order) or (order <= 0):
            raise ValueError(
                f"order needs to be specified and greater than 0; got order: {order}"
            )
        terp = interpolate.UnivariateSpline(x, y, k=order, **kwargs)
        new_y = terp(new_x)
    else:
        # GH 7295: need to be able to write for some reason
        # in some circumstances: check all three
        if not x.flags.writeable:
            x = x.copy()
        if not y.flags.writeable:
            y = y.copy()
        if not new_x.flags.writeable:
            new_x = new_x.copy()
        terp = alt_methods[method]
        new_y = terp(x, y, new_x, **kwargs)
    return new_y


def _from_derivatives(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    order=None,
    der: int | list[int] | None = 0,
    extrapolate: bool = False,
):
    """
    Convenience function for interpolate.BPoly.from_derivatives.

    Construct a piecewise polynomial in the Bernstein basis, compatible
    with the specified values and derivatives at breakpoints.

    Parameters
    ----------
    xi : array-like
        sorted 1D array of x-coordinates
    yi : array-like or list of array-likes
        yi[i][j] is the j-th derivative known at xi[i]
    order: None or int or array-like of ints. Default: None.
        Specifies the degree of local polynomials. If not None, some
        derivatives are ignored.
    der : int or list
        How many derivatives to extract; None for all potentially nonzero
        derivatives (that is a number equal to the number of points), or a
        list of derivatives to extract. This number includes the function
        value as 0th derivative.
     extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points based on first and last
        intervals, or to return NaNs. Default: True.

    See Also
    --------
    scipy.interpolate.BPoly.from_derivatives

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R.
    """
    from scipy import interpolate

    # return the method for compat with scipy version & backwards compat
    method = interpolate.BPoly.from_derivatives
    m = method(xi, yi.reshape(-1, 1), orders=order, extrapolate=extrapolate)

    return m(x)


def _akima_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    der: int | list[int] | None = 0,
    axis: AxisInt = 0,
):
    """
    Convenience function for akima interpolation.
    xi and yi are arrays of values used to approximate some function f,
    with ``yi = f(xi)``.

    See `Akima1DInterpolator` for details.

    Parameters
    ----------
    xi : np.ndarray
        A sorted list of x-coordinates, of length N.
    yi : np.ndarray
        A 1-D array of real values.  `yi`'s length along the interpolation
        axis must be equal to the length of `xi`. If N-D array, use axis
        parameter to select correct axis.
    x : np.ndarray
        Of length M.
    der : int, optional
        How many derivatives to extract; None for all potentially
        nonzero derivatives (that is a number equal to the number
        of points), or a list of derivatives to extract. This number
        includes the function value as 0th derivative.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values.

    See Also
    --------
    scipy.interpolate.Akima1DInterpolator

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R,

    """
    from scipy import interpolate

    P = interpolate.Akima1DInterpolator(xi, yi, axis=axis)

    return P(x, nu=der)


def _cubicspline_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    axis: AxisInt = 0,
    bc_type: str | tuple[Any, Any] = "not-a-knot",
    extrapolate=None,
):
    """
    Convenience function for cubic spline data interpolator.

    See `scipy.interpolate.CubicSpline` for details.

    Parameters
    ----------
    xi : np.ndarray, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    yi : np.ndarray
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    x : np.ndarray, shape (m,)
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type : string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.
        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:
        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped': The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural': The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.
        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple `(order, deriv_values)` allowing to specify arbitrary
        derivatives at curve ends:
        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array-like containing derivative values, shape must
          be the same as `y`, excluding ``axis`` dimension. For example, if
          `y` is 1D, then `deriv_value` must be a scalar. If `y` is 3D with
          the shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D
          and have the shape (n0, n1).
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

    See Also
    --------
    scipy.interpolate.CubicHermiteSpline

    Returns
    -------
    y : scalar or array-like
        The result, of shape (m,)

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """
    from scipy import interpolate

    P = interpolate.CubicSpline(
        xi, yi, axis=axis, bc_type=bc_type, extrapolate=extrapolate
    )

    return P(x)


def _interpolate_with_limit_area(
    values: np.ndarray,
    method: Literal["pad", "backfill"],
    limit: int | None,
    limit_area: Literal["inside", "outside"],
) -> None:
    """
    Apply interpolation and limit_area logic to values along a to-be-specified axis.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str
        Interpolation method. Could be "bfill" or "pad"
    limit: int, optional
        Index limit on interpolation.
    limit_area: {'inside', 'outside'}
        Limit area for interpolation.

    Notes
    -----
    Modifies values in-place.
    """

    invalid = isna(values)
    is_valid = ~invalid

    if not invalid.all():
        first = find_valid_index(how="first", is_valid=is_valid)
        if first is None:
            first = 0
        last = find_valid_index(how="last", is_valid=is_valid)
        if last is None:
            last = len(values)

        pad_or_backfill_inplace(
            values,
            method=method,
            limit=limit,
            limit_area=limit_area,
        )

        if limit_area == "inside":
            invalid[first : last + 1] = False
        elif limit_area == "outside":
            invalid[:first] = invalid[last + 1 :] = False
        else:
            raise ValueError("limit_area should be 'inside' or 'outside'")

        values[invalid] = np.nan


def pad_or_backfill_inplace(
    values: np.ndarray,
    method: Literal["pad", "backfill"] = "pad",
    axis: AxisInt = 0,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
) -> None:
    """
    Perform an actual interpolation of values, values will be make 2-d if
    needed fills inplace, returns the result.

    Parameters
    ----------
    values: np.ndarray
        Input array.
    method: str, default "pad"
        Interpolation method. Could be "bfill" or "pad"
    axis: 0 or 1
        Interpolation axis
    limit: int, optional
        Index limit on interpolation.
    limit_area: str, optional
        Limit area for interpolation. Can be "inside" or "outside"

    Notes
    -----
    Modifies values in-place.
    """
    transf = (lambda x: x) if axis == 0 else (lambda x: x.T)

    # reshape a 1 dim if needed
    if values.ndim == 1:
        if axis != 0:  # pragma: no cover
            raise AssertionError("cannot interpolate on a ndim == 1 with axis != 0")
        values = values.reshape(tuple((1,) + values.shape))

    method = clean_fill_method(method)
    tvalues = transf(values)

    func = get_fill_func(method, ndim=2)
    # _pad_2d and _backfill_2d both modify tvalues inplace
    func(tvalues, limit=limit, limit_area=limit_area)


def _fillna_prep(
    values, mask: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.bool_]:
    # boilerplate for _pad_1d, _backfill_1d, _pad_2d, _backfill_2d

    if mask is None:
        mask = isna(values)

    return mask


def _datetimelike_compat(func: F) -> F:
    """
    Wrapper to handle datetime64 and timedelta64 dtypes.
    """

    @wraps(func)
    def new_func(
        values,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        mask=None,
    ):
        if needs_i8_conversion(values.dtype):
            if mask is None:
                # This needs to occur before casting to int64
                mask = isna(values)

            result, mask = func(
                values.view("i8"), limit=limit, limit_area=limit_area, mask=mask
            )
            return result.view(values.dtype), mask

        return func(values, limit=limit, limit_area=limit_area, mask=mask)

    return cast(F, new_func)


@_datetimelike_compat
def _pad_1d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    mask = _fillna_prep(values, mask)
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    algos.pad_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _backfill_1d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    mask = _fillna_prep(values, mask)
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    algos.backfill_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _pad_2d(
    values: np.ndarray,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
):
    mask = _fillna_prep(values, mask)
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)

    if values.size:
        algos.pad_2d_inplace(values, mask, limit=limit)
    else:
        # for test coverage
        pass
    return values, mask


@_datetimelike_compat
def _backfill_2d(
    values,
    limit: int | None = None,
    limit_area: Literal["inside", "outside"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
):
    mask = _fillna_prep(values, mask)
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)

    if values.size:
        algos.backfill_2d_inplace(values, mask, limit=limit)
    else:
        # for test coverage
        pass
    return values, mask


def _fill_limit_area_1d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    """Prepare 1d mask for ffill/bfill with limit_area.

    Caller is responsible for checking at least one value of mask is False.
    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    neg_mask = ~mask
    first = neg_mask.argmax()
    last = len(neg_mask) - neg_mask[::-1].argmax() - 1
    if limit_area == "inside":
        mask[:first] = False
        mask[last + 1 :] = False
    elif limit_area == "outside":
        mask[first + 1 : last] = False


def _fill_limit_area_2d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    """Prepare 2d mask for ffill/bfill with limit_area.

    When called, mask will no longer faithfully represent when
    the corresponding are NA or not.

    Parameters
    ----------
    mask : np.ndarray[bool, ndim=1]
        Mask representing NA values when filling.
    limit_area : { "outside", "inside" }
        Whether to limit filling to outside or inside the outer most non-NA value.
    """
    neg_mask = ~mask.T
    if limit_area == "outside":
        # Identify inside
        la_mask = (
            np.maximum.accumulate(neg_mask, axis=0)
            & np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    else:
        # Identify outside
        la_mask = (
            ~np.maximum.accumulate(neg_mask, axis=0)
            | ~np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    mask[la_mask.T] = False


_fill_methods = {"pad": _pad_1d, "backfill": _backfill_1d}


def get_fill_func(method, ndim: int = 1):
    method = clean_fill_method(method)
    if ndim == 1:
        return _fill_methods[method]
    return {"pad": _pad_2d, "backfill": _backfill_2d}[method]


def clean_reindex_fill_method(method) -> ReindexMethod | None:
    if method is None:
        return None
    return clean_fill_method(method, allow_nearest=True)


def _interp_limit(
    invalid: npt.NDArray[np.bool_], fw_limit: int | None, bw_limit: int | None
):
    """
    Get indexers of values that won't be filled
    because they exceed the limits.

    Parameters
    ----------
    invalid : np.ndarray[bool]
    fw_limit : int or None
        forward limit to index
    bw_limit : int or None
        backward limit to index

    Returns
    -------
    set of indexers

    Notes
    -----
    This is equivalent to the more readable, but slower

    .. code-block:: python

        def _interp_limit(invalid, fw_limit, bw_limit):
            for x in np.where(invalid)[0]:
                if invalid[max(0, x - fw_limit):x + bw_limit + 1].all():
                    yield x
    """
    # handle forward first; the backward direction is the same except
    # 1. operate on the reversed array
    # 2. subtract the returned indices from N - 1
    N = len(invalid)
    f_idx = set()
    b_idx = set()

    def inner(invalid, limit: int):
        limit = min(limit, N)
        windowed = _rolling_window(invalid, limit + 1).all(1)
        idx = set(np.where(windowed)[0] + limit) | set(
            np.where((~invalid[: limit + 1]).cumsum() == 0)[0]
        )
        return idx

    if fw_limit is not None:
        if fw_limit == 0:
            f_idx = set(np.where(invalid)[0])
        else:
            f_idx = inner(invalid, fw_limit)

    if bw_limit is not None:
        if bw_limit == 0:
            # then we don't even need to care about backwards
            # just use forwards
            return f_idx
        else:
            b_idx_inv = list(inner(invalid[::-1], bw_limit))
            b_idx = set(N - 1 - np.asarray(b_idx_inv))
            if fw_limit == 0:
                return b_idx

    return f_idx & b_idx


def _rolling_window(a: npt.NDArray[np.bool_], window: int) -> npt.NDArray[np.bool_]:
    """
    [True, True, False, True, False], 2 ->

    [
        [True,  True],
        [True, False],
        [False, True],
        [True, False],
    ]
    """
    # https://stackoverflow.com/a/6811241
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\generate.py ===
"""
Generating and counting primes.

"""

from bisect import bisect, bisect_left
from itertools import count
# Using arrays for sieving instead of lists greatly reduces
# memory consumption
from array import array as _array

from sympy.core.random import randint
from sympy.external.gmpy import sqrt
from .primetest import isprime
from sympy.utilities.decorator import deprecated
from sympy.utilities.misc import as_int


def _as_int_ceiling(a):
    """ Wrapping ceiling in as_int will raise an error if there was a problem
        determining whether the expression was exactly an integer or not."""
    from sympy.functions.elementary.integers import ceiling
    return as_int(ceiling(a))


class Sieve:
    """A list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number. Implementation details limit the number of
    primes to ``2^32-1``.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    # data shared (and updated) by all Sieve instances
    def __init__(self, sieve_interval=1_000_000):
        """ Initial parameters for the Sieve class.

        Parameters
        ==========

        sieve_interval (int): Amount of memory to be used

        Raises
        ======

        ValueError
            If ``sieve_interval`` is not positive.

        """
        self._n = 6
        self._list = _array('L', [2, 3, 5, 7, 11, 13]) # primes
        self._tlist = _array('L', [0, 1, 1, 2, 2, 4]) # totient
        self._mlist = _array('i', [0, 1, -1, -1, 0, -1]) # mobius
        if sieve_interval <= 0:
            raise ValueError("sieve_interval should be a positive integer")
        self.sieve_interval = sieve_interval
        assert all(len(i) == self._n for i in (self._list, self._tlist, self._mlist))

    def __repr__(self):
        return ("<%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i>") % (
             'prime', len(self._list),
                 self._list[0], self._list[1], self._list[2],
                 self._list[-2], self._list[-1],
             'totient', len(self._tlist),
                 self._tlist[0], self._tlist[1],
                 self._tlist[2], self._tlist[-2], self._tlist[-1],
             'mobius', len(self._mlist),
                 self._mlist[0], self._mlist[1],
                 self._mlist[2], self._mlist[-2], self._mlist[-1])

    def _reset(self, prime=None, totient=None, mobius=None):
        """Reset all caches (default). To reset one or more set the
            desired keyword to True."""
        if all(i is None for i in (prime, totient, mobius)):
            prime = totient = mobius = True
        if prime:
            self._list = self._list[:self._n]
        if totient:
            self._tlist = self._tlist[:self._n]
        if mobius:
            self._mlist = self._mlist[:self._n]

    def extend(self, n):
        """Grow the sieve to cover all primes <= n.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend(30)
        >>> sieve[10] == 29
        True
        """
        n = int(n)
        # `num` is even at any point in the function.
        # This satisfies the condition required by `self._primerange`.
        num = self._list[-1] + 1
        if n < num:
            return
        num2 = num**2
        while num2 <= n:
            self._list += _array('L', self._primerange(num, num2))
            num, num2 = num2, num2**2
        # Merge the sieves
        self._list += _array('L', self._primerange(num, n + 1))

    def _primerange(self, a, b):
        """ Generate all prime numbers in the range (a, b).

        Parameters
        ==========

        a, b : positive integers assuming the following conditions
                * a is an even number
                * 2 < self._list[-1] < a < b < nextprime(self._list[-1])**2

        Yields
        ======

        p (int): prime numbers such that ``a < p < b``

        Examples
        ========

        >>> from sympy.ntheory.generate import Sieve
        >>> s = Sieve()
        >>> s._list[-1]
        13
        >>> list(s._primerange(18, 31))
        [19, 23, 29]

        """
        if b % 2:
            b -= 1
        while a < b:
            block_size = min(self.sieve_interval, (b - a) // 2)
            # Create the list such that block[x] iff (a + 2x + 1) is prime.
            # Note that even numbers are not considered here.
            block = [True] * block_size
            for p in self._list[1:bisect(self._list, sqrt(a + 2 * block_size + 1))]:
                for t in range((-(a + 1 + p) // 2) % p, block_size, p):
                    block[t] = False
            for idx, p in enumerate(block):
                if p:
                    yield a + 2 * idx + 1
            a += 2 * block_size

    def extend_to_no(self, i):
        """Extend to include the ith prime number.

        Parameters
        ==========

        i : integer

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend_to_no(9)
        >>> sieve._list
        array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])

        Notes
        =====

        The list is extended by 50% if it is too short, so it is
        likely that it will be longer than requested.
        """
        i = as_int(i)
        while len(self._list) < i:
            self.extend(int(self._list[-1] * 1.5))

    def primerange(self, a, b=None):
        """Generate all prime numbers in the range [2, a) or [a, b).

        Examples
        ========

        >>> from sympy import sieve, prime

        All primes less than 19:

        >>> print([i for i in sieve.primerange(19)])
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> print([i for i in sieve.primerange(7, 19)])
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(sieve.primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        """
        if b is None:
            b = _as_int_ceiling(a)
            a = 2
        else:
            a = max(2, _as_int_ceiling(a))
            b = _as_int_ceiling(b)
        if a >= b:
            return
        self.extend(b)
        yield from self._list[bisect_left(self._list, a):
                              bisect_left(self._list, b)]

    def totientrange(self, a, b):
        """Generate all totient numbers for the range [a, b).

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.totientrange(7, 18)])
        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._tlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._tlist[i]
        else:
            self._tlist += _array('L', range(n, b))
            for i in range(1, n):
                ti = self._tlist[i]
                if ti == i - 1:
                    startindex = (n + i - 1) // i * i
                    for j in range(startindex, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield ti

            for i in range(n, b):
                ti = self._tlist[i]
                if ti == i:
                    for j in range(i, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield self._tlist[i]

    def mobiusrange(self, a, b):
        """Generate all mobius numbers for the range [a, b).

        Parameters
        ==========

        a : integer
            First number in range

        b : integer
            First number outside of range

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.mobiusrange(7, 18)])
        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._mlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._mlist[i]
        else:
            self._mlist += _array('i', [0]*(b - n))
            for i in range(1, n):
                mi = self._mlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

            for i in range(n, b):
                mi = self._mlist[i]
                for j in range(2 * i, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

    def search(self, n):
        """Return the indices i, j of the primes that bound n.

        If n is prime then i == j.

        Although n can be an expression, if ceiling cannot convert
        it to an integer then an n error will be raised.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve.search(25)
        (9, 10)
        >>> sieve.search(23)
        (9, 9)
        """
        test = _as_int_ceiling(n)
        n = as_int(n)
        if n < 2:
            raise ValueError("n should be >= 2 but got: %s" % n)
        if n > self._list[-1]:
            self.extend(n)
        b = bisect(self._list, n)
        if self._list[b - 1] == test:
            return b, b
        else:
            return b, b + 1

    def __contains__(self, n):
        try:
            n = as_int(n)
            assert n >= 2
        except (ValueError, AssertionError):
            return False
        if n % 2 == 0:
            return n == 2
        a, b = self.search(n)
        return a == b

    def __iter__(self):
        for n in count(1):
            yield self[n]

    def __getitem__(self, n):
        """Return the nth prime number"""
        if isinstance(n, slice):
            self.extend_to_no(n.stop)
            start = n.start if n.start is not None else 0
            if start < 1:
                # sieve[:5] would be empty (starting at -1), let's
                # just be explicit and raise.
                raise IndexError("Sieve indices start at 1.")
            return self._list[start - 1:n.stop - 1:n.step]
        else:
            if n < 1:
                # offset is one, so forbid explicit access to sieve[0]
                # (would surprisingly return the last one).
                raise IndexError("Sieve indices start at 1.")
            n = as_int(n)
            self.extend_to_no(n)
            return self._list[n - 1]

# Generate a global object for repeated use in trial division etc
sieve = Sieve()

def prime(nth):
    r"""
    Return the nth prime number, where primes are indexed starting from 1:
    prime(1) = 2, prime(2) = 3, etc.

    Parameters
    ==========

    nth : int
        The position of the prime number to return (must be a positive integer).

    Returns
    =======

    int
        The nth prime number.

    Examples
    ========

    >>> from sympy import prime
    >>> prime(10)
    29
    >>> prime(1)
    2
    >>> prime(100000)
    1299709

    See Also
    ========

    sympy.ntheory.primetest.isprime : Test if a number is prime.
    primerange : Generate all primes in a given range.
    primepi : Return the number of primes less than or equal to a given number.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Prime_number_theorem
    .. [2] https://en.wikipedia.org/wiki/Logarithmic_integral_function
    .. [3] https://en.wikipedia.org/wiki/Skewes%27_number
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; prime(1) == 2")

    # Check if n is within the sieve range
    if n <= len(sieve._list):
        return sieve[n]

    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li

    if n < 1000:
        # Extend sieve up to 8*n as this is empirically sufficient
        sieve.extend(8 * n)
        return sieve[n]

    a = 2
    # Estimate an upper bound for the nth prime using the prime number theorem
    b = int(n * (log(n).evalf() + log(log(n)).evalf()))

    # Binary search for the least m such that li(m) > n
    while a < b:
        mid = (a + b) >> 1
        if li(mid).evalf() > n:
            b = mid
        else:
            a = mid + 1

    return nextprime(a - 1, n - _primepi(a - 1))


@deprecated("""\
The `sympy.ntheory.generate.primepi` has been moved to `sympy.functions.combinatorial.numbers.primepi`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primepi(n):
    r""" Represents the prime counting function pi(n) = the number
        of prime numbers less than or equal to n.

        .. deprecated:: 1.13

            The ``primepi`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.primepi`
            instead. See its documentation for more information. See
            :ref:`deprecated-ntheory-symbolic-functions` for details.

        Algorithm Description:

        In sieve method, we remove all multiples of prime p
        except p itself.

        Let phi(i,j) be the number of integers 2 <= k <= i
        which remain after sieving from primes less than
        or equal to j.
        Clearly, pi(n) = phi(n, sqrt(n))

        If j is not a prime,
        phi(i,j) = phi(i, j - 1)

        if j is a prime,
        We remove all numbers(except j) whose
        smallest prime factor is j.

        Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
        Now, after sieving from primes $\le j - 1$,
        a must remain
        (because x, and hence a has no prime factor $\le j - 1$)
        Clearly, there are phi(i / j, j - 1) such a
        which remain on sieving from primes $\le j - 1$

        Now, if a is a prime less than equal to j - 1,
        $x= j \times a$ has smallest prime factor = a, and
        has already been removed(by sieving from a).
        So, we do not need to remove it again.
        (Note: there will be pi(j - 1) such x)

        Thus, number of x, that will be removed are:
        phi(i / j, j - 1) - phi(j - 1, j - 1)
        (Note that pi(j - 1) = phi(j - 1, j - 1))

        $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

        So,following recursion is used and implemented as dp:

        phi(a, b) = phi(a, b - 1), if b is not a prime
        phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

        Clearly a is always of the form floor(n / k),
        which can take at most $2\sqrt{n}$ values.
        Two arrays arr1,arr2 are maintained
        arr1[i] = phi(i, j),
        arr2[i] = phi(n // i, j)

        Finally the answer is arr2[1]

        Examples
        ========

        >>> from sympy import primepi, prime, prevprime, isprime
        >>> primepi(25)
        9

        So there are 9 primes less than or equal to 25. Is 25 prime?

        >>> isprime(25)
        False

        It is not. So the first prime less than 25 must be the
        9th prime:

        >>> prevprime(25) == prime(9)
        True

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        prime : Return the nth prime
    """
    from sympy.functions.combinatorial.numbers import primepi as func_primepi
    return func_primepi(n)


def _primepi(n:int) -> int:
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Explanation
    ===========

    In sieve method, we remove all multiples of prime p
    except p itself.

    Let phi(i,j) be the number of integers 2 <= k <= i
    which remain after sieving from primes less than
    or equal to j.
    Clearly, pi(n) = phi(n, sqrt(n))

    If j is not a prime,
    phi(i,j) = phi(i, j - 1)

    if j is a prime,
    We remove all numbers(except j) whose
    smallest prime factor is j.

    Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
    Now, after sieving from primes $\le j - 1$,
    a must remain
    (because x, and hence a has no prime factor $\le j - 1$)
    Clearly, there are phi(i / j, j - 1) such a
    which remain on sieving from primes $\le j - 1$

    Now, if a is a prime less than equal to j - 1,
    $x= j \times a$ has smallest prime factor = a, and
    has already been removed(by sieving from a).
    So, we do not need to remove it again.
    (Note: there will be pi(j - 1) such x)

    Thus, number of x, that will be removed are:
    phi(i / j, j - 1) - phi(j - 1, j - 1)
    (Note that pi(j - 1) = phi(j - 1, j - 1))

    $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

    So,following recursion is used and implemented as dp:

    phi(a, b) = phi(a, b - 1), if b is not a prime
    phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

    Clearly a is always of the form floor(n / k),
    which can take at most $2\sqrt{n}$ values.
    Two arrays arr1,arr2 are maintained
    arr1[i] = phi(i, j),
    arr2[i] = phi(n // i, j)

    Finally the answer is arr2[1]

    Parameters
    ==========

    n : int

    """
    if n < 2:
        return 0
    if n <= sieve._list[-1]:
        return sieve.search(n)[0]
    lim = sqrt(n)
    arr1 = [(i + 1) >> 1 for i in range(lim + 1)]
    arr2 = [0] + [(n//i + 1) >> 1 for i in range(1, lim + 1)]
    skip = [False] * (lim + 1)
    for i in range(3, lim + 1, 2):
        # Presently, arr1[k]=phi(k,i - 1),
        # arr2[k] = phi(n // k,i - 1) # not all k's do this
        if skip[i]:
            # skip if i is a composite number
            continue
        p = arr1[i - 1]
        for j in range(i, lim + 1, i):
            skip[j] = True
        # update arr2
        # phi(n/j, i) = phi(n/j, i-1) - phi(n/(i*j), i-1) + phi(i-1, i-1)
        for j in range(1, min(n // (i * i), lim) + 1, 2):
            # No need for arr2[j] in j such that skip[j] is True to
            # compute the final required arr2[1].
            if skip[j]:
                continue
            st = i * j
            if st <= lim:
                arr2[j] -= arr2[st] - p
            else:
                arr2[j] -= arr1[n // st] - p
        # update arr1
        # phi(j, i) = phi(j, i-1) - phi(j/i, i-1) + phi(i-1, i-1)
        # where the range below i**2 is fixed and
        # does not need to be calculated.
        for j in range(lim, min(lim, i*i - 1), -1):
            arr1[j] -= arr1[j // i] - p
    return arr2[1]


def nextprime(n, ith=1):
    """ Return the ith prime greater than n.

        Parameters
        ==========

        n : integer
        ith : positive integer

        Returns
        =======

        int : Return the ith prime greater than n

        Raises
        ======

        ValueError
            If ``ith <= 0``.
            If ``n`` or ``ith`` is not an integer.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import nextprime
        >>> [(i, nextprime(i)) for i in range(10, 15)]
        [(10, 11), (11, 13), (12, 13), (13, 17), (14, 17)]
        >>> nextprime(2, ith=2) # the 2nd prime after 2
        5

        See Also
        ========

        prevprime : Return the largest prime smaller than n
        primerange : Generate all primes in a given range

    """
    n = int(n)
    i = as_int(ith)
    if i <= 0:
        raise ValueError("ith should be positive")
    if n < 2:
        n = 2
        i -= 1
    if n <= sieve._list[-2]:
        l, _ = sieve.search(n)
        if l + i - 1 < len(sieve._list):
            return sieve._list[l + i - 1]
        n = sieve._list[-1]
        i += l - len(sieve._list)
    nn = 6*(n//6)
    if nn == n:
        n += 1
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    elif n - nn == 5:
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    else:
        n = nn + 5
    while 1:
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4


def prevprime(n):
    """ Return the largest prime smaller than n.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import prevprime
        >>> [(i, prevprime(i)) for i in range(10, 15)]
        [(10, 7), (11, 7), (12, 11), (13, 11), (14, 13)]

        See Also
        ========

        nextprime : Return the ith prime greater than n
        primerange : Generates all primes in a given range
    """
    n = _as_int_ceiling(n)
    if n < 3:
        raise ValueError("no preceding primes")
    if n < 8:
        return {3: 2, 4: 3, 5: 3, 6: 5, 7: 5}[n]
    if n <= sieve._list[-1]:
        l, u = sieve.search(n)
        if l == u:
            return sieve[l-1]
        else:
            return sieve[l]
    nn = 6*(n//6)
    if n - nn <= 1:
        n = nn - 1
        if isprime(n):
            return n
        n -= 4
    else:
        n = nn + 1
    while 1:
        if isprime(n):
            return n
        n -= 2
        if isprime(n):
            return n
        n -= 4


def primerange(a, b=None):
    """ Generate a list of all prime numbers in the range [2, a),
        or [a, b).

        If the range exists in the default sieve, the values will
        be returned from there; otherwise values will be returned
        but will not modify the sieve.

        Examples
        ========

        >>> from sympy import primerange, prime

        All primes less than 19:

        >>> list(primerange(19))
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> list(primerange(7, 19))
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        The Sieve method, primerange, is generally faster but it will
        occupy more memory as the sieve stores values. The default
        instance of Sieve, named sieve, can be used:

        >>> from sympy import sieve
        >>> list(sieve.primerange(1, 30))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        Notes
        =====

        Some famous conjectures about the occurrence of primes in a given
        range are [1]:

        - Twin primes: though often not, the following will give 2 primes
                    an infinite number of times:
                        primerange(6*n - 1, 6*n + 2)
        - Legendre's: the following always yields at least one prime
                        primerange(n**2, (n+1)**2+1)
        - Bertrand's (proven): there is always a prime in the range
                        primerange(n, 2*n)
        - Brocard's: there are at least four primes in the range
                        primerange(prime(n)**2, prime(n+1)**2)

        The average gap between primes is log(n) [2]; the gap between
        primes can be arbitrarily large since sequences of composite
        numbers are arbitrarily large, e.g. the numbers in the sequence
        n! + 2, n! + 3 ... n! + n are all composite.

        See Also
        ========

        prime : Return the nth prime
        nextprime : Return the ith prime greater than n
        prevprime : Return the largest prime smaller than n
        randprime : Returns a random prime in a given range
        primorial : Returns the product of primes based on condition
        Sieve.primerange : return range from already computed primes
                           or extend the sieve to contain the requested
                           range.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Prime_number
        .. [2] https://primes.utm.edu/notes/gaps.html
    """
    if b is None:
        a, b = 2, a
    if a >= b:
        return
    # If we already have the range, return it.
    largest_known_prime = sieve._list[-1]
    if b <= largest_known_prime:
        yield from sieve.primerange(a, b)
        return
    # If we know some of it, return it.
    if a <= largest_known_prime:
        yield from sieve._list[bisect_left(sieve._list, a):]
        a = largest_known_prime + 1
    elif a % 2:
        a -= 1
    tail = min(b, (largest_known_prime)**2)
    if a < tail:
        yield from sieve._primerange(a, tail)
        a = tail
    if b <= a:
        return
    # otherwise compute, without storing, the desired range.
    while 1:
        a = nextprime(a)
        if a < b:
            yield a
        else:
            return


def randprime(a, b):
    """ Return a random prime number in the range [a, b).

        Bertrand's postulate assures that
        randprime(a, 2*a) will always succeed for a > 1.

        Note that due to implementation difficulties,
        the prime numbers chosen are not uniformly random.
        For example, there are two primes in the range [112, 128),
        ``113`` and ``127``, but ``randprime(112, 128)`` returns ``127``
        with a probability of 15/17.

        Examples
        ========

        >>> from sympy import randprime, isprime
        >>> randprime(1, 30) #doctest: +SKIP
        13
        >>> isprime(randprime(1, 30))
        True

        See Also
        ========

        primerange : Generate all primes in a given range

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Bertrand's_postulate

    """
    if a >= b:
        return
    a, b = map(int, (a, b))
    n = randint(a - 1, b)
    p = nextprime(n)
    if p >= b:
        p = prevprime(b)
    if p < a:
        raise ValueError("no primes exist in the specified range")
    return p


def primorial(n, nth=True):
    """
    Returns the product of the first n primes (default) or
    the primes less than or equal to n (when ``nth=False``).

    Examples
    ========

    >>> from sympy.ntheory.generate import primorial, primerange
    >>> from sympy import factorint, Mul, primefactors, sqrt
    >>> primorial(4) # the first 4 primes are 2, 3, 5, 7
    210
    >>> primorial(4, nth=False) # primes <= 4 are 2 and 3
    6
    >>> primorial(1)
    2
    >>> primorial(1, nth=False)
    1
    >>> primorial(sqrt(101), nth=False)
    210

    One can argue that the primes are infinite since if you take
    a set of primes and multiply them together (e.g. the primorial) and
    then add or subtract 1, the result cannot be divided by any of the
    original factors, hence either 1 or more new primes must divide this
    product of primes.

    In this case, the number itself is a new prime:

    >>> factorint(primorial(4) + 1)
    {211: 1}

    In this case two new primes are the factors:

    >>> factorint(primorial(4) - 1)
    {11: 1, 19: 1}

    Here, some primes smaller and larger than the primes multiplied together
    are obtained:

    >>> p = list(primerange(10, 20))
    >>> sorted(set(primefactors(Mul(*p) + 1)).difference(set(p)))
    [2, 5, 31, 149]

    See Also
    ========

    primerange : Generate all primes in a given range

    """
    if nth:
        n = as_int(n)
    else:
        n = int(n)
    if n < 1:
        raise ValueError("primorial argument must be >= 1")
    p = 1
    if nth:
        for i in range(1, n + 1):
            p *= prime(i)
    else:
        for i in primerange(2, n + 1):
            p *= i
    return p


def cycle_length(f, x0, nmax=None, values=False):
    """For a given iterated sequence, return a generator that gives
    the length of the iterated cycle (lambda) and the length of terms
    before the cycle begins (mu); if ``values`` is True then the
    terms of the sequence will be returned instead. The sequence is
    started with value ``x0``.

    Note: more than the first lambda + mu terms may be returned and this
    is the cost of cycle detection with Brent's method; there are, however,
    generally less terms calculated than would have been calculated if the
    proper ending point were determined, e.g. by using Floyd's method.

    >>> from sympy.ntheory.generate import cycle_length

    This will yield successive values of i <-- func(i):

        >>> def gen(func, i):
        ...     while 1:
        ...         yield i
        ...         i = func(i)
        ...

    A function is defined:

        >>> func = lambda i: (i**2 + 1) % 51

    and given a seed of 4 and the mu and lambda terms calculated:

        >>> next(cycle_length(func, 4))
        (6, 3)

    We can see what is meant by looking at the output:

        >>> iter = cycle_length(func, 4, values=True)
        >>> list(iter)
        [4, 17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]

    There are 6 repeating values after the first 3.

    If a sequence is suspected of being longer than you might wish, ``nmax``
    can be used to exit early (and mu will be returned as None):

        >>> next(cycle_length(func, 4, nmax = 4))
        (4, None)
        >>> list(cycle_length(func, 4, nmax = 4, values=True))
        [4, 17, 35, 2]

    Code modified from:
        https://en.wikipedia.org/wiki/Cycle_detection.
    """

    nmax = int(nmax or 0)

    # main phase: search successive powers of two
    power = lam = 1
    tortoise, hare = x0, f(x0)  # f(x0) is the element/node next to x0.
    i = 1
    if values:
        yield tortoise
    while tortoise != hare and (not nmax or i < nmax):
        i += 1
        if power == lam:   # time to start a new power of two?
            tortoise = hare
            power *= 2
            lam = 0
        if values:
            yield hare
        hare = f(hare)
        lam += 1
    if nmax and i == nmax:
        if values:
            return
        else:
            yield nmax, None
            return
    if not values:
        # Find the position of the first repetition of length lambda
        mu = 0
        tortoise = hare = x0
        for i in range(lam):
            hare = f(hare)
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)
            mu += 1
        yield lam, mu


def composite(nth):
    """ Return the nth composite number, with the composite numbers indexed as
        composite(1) = 4, composite(2) = 6, etc....

        Examples
        ========

        >>> from sympy import composite
        >>> composite(36)
        52
        >>> composite(1)
        4
        >>> composite(17737)
        20000

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        primepi : Return the number of primes less than or equal to n
        prime : Return the nth prime
        compositepi : Return the number of positive composite numbers less than or equal to n
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; composite(1) == 4")
    composite_arr = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    if n <= 10:
        return composite_arr[n - 1]

    a, b = 4, sieve._list[-1]
    if n <= b - _primepi(b) - 1:
        while a < b - 1:
            mid = (a + b) >> 1
            if mid - _primepi(mid) - 1 > n:
                b = mid
            else:
                a = mid
        if isprime(a):
            a -= 1
        return a

    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li
    a = 4 # Lower bound for binary search
    b = int(n*(log(n) + log(log(n)))) # Upper bound for the search.

    while a < b:
        mid = (a + b) >> 1
        if mid - li(mid) - 1 > n:
            b = mid
        else:
            a = mid + 1

    n_composites = a - _primepi(a) - 1
    while n_composites > n:
        if not isprime(a):
            n_composites -= 1
        a -= 1
    if isprime(a):
        a -= 1
    return a


def compositepi(n):
    """ Return the number of positive composite numbers less than or equal to n.
        The first positive composite is 4, i.e. compositepi(4) = 1.

        Examples
        ========

        >>> from sympy import compositepi
        >>> compositepi(25)
        15
        >>> compositepi(1000)
        831

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        prime : Return the nth prime
        primepi : Return the number of primes less than or equal to n
        composite : Return the nth composite number
    """
    n = int(n)
    if n < 4:
        return 0
    return n - _primepi(n) - 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\enumerative.py ===
"""
Algorithms and classes to support enumerative combinatorics.

Currently just multiset partitions, but more could be added.

Terminology (following Knuth, algorithm 7.1.2.5M TAOCP)
*multiset* aaabbcccc has a *partition* aaabc | bccc

The submultisets, aaabc and bccc of the partition are called
*parts*, or sometimes *vectors*.  (Knuth notes that multiset
partitions can be thought of as partitions of vectors of integers,
where the ith element of the vector gives the multiplicity of
element i.)

The values a, b and c are *components* of the multiset.  These
correspond to elements of a set, but in a multiset can be present
with a multiplicity greater than 1.

The algorithm deserves some explanation.

Think of the part aaabc from the multiset above.  If we impose an
ordering on the components of the multiset, we can represent a part
with a vector, in which the value of the first element of the vector
corresponds to the multiplicity of the first component in that
part. Thus, aaabc can be represented by the vector [3, 1, 1].  We
can also define an ordering on parts, based on the lexicographic
ordering of the vector (leftmost vector element, i.e., the element
with the smallest component number, is the most significant), so
that [3, 1, 1] > [3, 1, 0] and [3, 1, 1] > [2, 1, 4].  The ordering
on parts can be extended to an ordering on partitions: First, sort
the parts in each partition, left-to-right in decreasing order. Then
partition A is greater than partition B if A's leftmost/greatest
part is greater than B's leftmost part.  If the leftmost parts are
equal, compare the second parts, and so on.

In this ordering, the greatest partition of a given multiset has only
one part.  The least partition is the one in which the components
are spread out, one per part.

The enumeration algorithms in this file yield the partitions of the
argument multiset in decreasing order.  The main data structure is a
stack of parts, corresponding to the current partition.  An
important invariant is that the parts on the stack are themselves in
decreasing order.  This data structure is decremented to find the
next smaller partition.  Most often, decrementing the partition will
only involve adjustments to the smallest parts at the top of the
stack, much as adjacent integers *usually* differ only in their last
few digits.

Knuth's algorithm uses two main operations on parts:

Decrement - change the part so that it is smaller in the
  (vector) lexicographic order, but reduced by the smallest amount possible.
  For example, if the multiset has vector [5,
  3, 1], and the bottom/greatest part is [4, 2, 1], this part would
  decrement to [4, 2, 0], while [4, 0, 0] would decrement to [3, 3,
  1].  A singleton part is never decremented -- [1, 0, 0] is not
  decremented to [0, 3, 1].  Instead, the decrement operator needs
  to fail for this case.  In Knuth's pseudocode, the decrement
  operator is step m5.

Spread unallocated multiplicity - Once a part has been decremented,
  it cannot be the rightmost part in the partition.  There is some
  multiplicity that has not been allocated, and new parts must be
  created above it in the stack to use up this multiplicity.  To
  maintain the invariant that the parts on the stack are in
  decreasing order, these new parts must be less than or equal to
  the decremented part.
  For example, if the multiset is [5, 3, 1], and its most
  significant part has just been decremented to [5, 3, 0], the
  spread operation will add a new part so that the stack becomes
  [[5, 3, 0], [0, 0, 1]].  If the most significant part (for the
  same multiset) has been decremented to [2, 0, 0] the stack becomes
  [[2, 0, 0], [2, 0, 0], [1, 3, 1]].  In the pseudocode, the spread
  operation for one part is step m2.  The complete spread operation
  is a loop of steps m2 and m3.

In order to facilitate the spread operation, Knuth stores, for each
component of each part, not just the multiplicity of that component
in the part, but also the total multiplicity available for this
component in this part or any lesser part above it on the stack.

One added twist is that Knuth does not represent the part vectors as
arrays. Instead, he uses a sparse representation, in which a
component of a part is represented as a component number (c), plus
the multiplicity of the component in that part (v) as well as the
total multiplicity available for that component (u).  This saves
time that would be spent skipping over zeros.

"""

class PartComponent:
    """Internal class used in support of the multiset partitions
    enumerators and the associated visitor functions.

    Represents one component of one part of the current partition.

    A stack of these, plus an auxiliary frame array, f, represents a
    partition of the multiset.

    Knuth's pseudocode makes c, u, and v separate arrays.
    """

    __slots__ = ('c', 'u', 'v')

    def __init__(self):
        self.c = 0   # Component number
        self.u = 0   # The as yet unpartitioned amount in component c
                     # *before* it is allocated by this triple
        self.v = 0   # Amount of c component in the current part
                     # (v<=u).  An invariant of the representation is
                     # that the next higher triple for this component
                     # (if there is one) will have a value of u-v in
                     # its u attribute.

    def __repr__(self):
        "for debug/algorithm animation purposes"
        return 'c:%d u:%d v:%d' % (self.c, self.u, self.v)

    def __eq__(self, other):
        """Define  value oriented equality, which is useful for testers"""
        return (isinstance(other, self.__class__) and
                self.c == other.c and
                self.u == other.u and
                self.v == other.v)

    def __ne__(self, other):
        """Defined for consistency with __eq__"""
        return not self == other


# This function tries to be a faithful implementation of algorithm
# 7.1.2.5M in Volume 4A, Combinatoral Algorithms, Part 1, of The Art
# of Computer Programming, by Donald Knuth.  This includes using
# (mostly) the same variable names, etc.  This makes for rather
# low-level Python.

# Changes from Knuth's pseudocode include
# - use PartComponent struct/object instead of 3 arrays
# - make the function a generator
# - map (with some difficulty) the GOTOs to Python control structures.
# - Knuth uses 1-based numbering for components, this code is 0-based
# - renamed variable l to lpart.
# - flag variable x takes on values True/False instead of 1/0
#
def multiset_partitions_taocp(multiplicities):
    """Enumerates partitions of a multiset.

    Parameters
    ==========

    multiplicities
         list of integer multiplicities of the components of the multiset.

    Yields
    ======

    state
        Internal data structure which encodes a particular partition.
        This output is then usually processed by a visitor function
        which combines the information from this data structure with
        the components themselves to produce an actual partition.

        Unless they wish to create their own visitor function, users will
        have little need to look inside this data structure.  But, for
        reference, it is a 3-element list with components:

        f
            is a frame array, which is used to divide pstack into parts.

        lpart
            points to the base of the topmost part.

        pstack
            is an array of PartComponent objects.

        The ``state`` output offers a peek into the internal data
        structures of the enumeration function.  The client should
        treat this as read-only; any modification of the data
        structure will cause unpredictable (and almost certainly
        incorrect) results.  Also, the components of ``state`` are
        modified in place at each iteration.  Hence, the visitor must
        be called at each loop iteration.  Accumulating the ``state``
        instances and processing them later will not work.

    Examples
    ========

    >>> from sympy.utilities.enumerative import list_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> # variables components and multiplicities represent the multiset 'abb'
    >>> components = 'ab'
    >>> multiplicities = [1, 2]
    >>> states = multiset_partitions_taocp(multiplicities)
    >>> list(list_visitor(state, components) for state in states)
    [[['a', 'b', 'b']],
    [['a', 'b'], ['b']],
    [['a'], ['b', 'b']],
    [['a'], ['b'], ['b']]]

    See Also
    ========

    sympy.utilities.iterables.multiset_partitions: Takes a multiset
        as input and directly yields multiset partitions.  It
        dispatches to a number of functions, including this one, for
        implementation.  Most users will find it more convenient to
        use than multiset_partitions_taocp.

    """

    # Important variables.
    # m is the number of components, i.e., number of distinct elements
    m = len(multiplicities)
    # n is the cardinality, total number of elements whether or not distinct
    n = sum(multiplicities)

    # The main data structure, f segments pstack into parts.  See
    # list_visitor() for example code indicating how this internal
    # state corresponds to a partition.

    # Note: allocation of space for stack is conservative.  Knuth's
    # exercise 7.2.1.5.68 gives some indication of how to tighten this
    # bound, but this is not implemented.
    pstack = [PartComponent() for i in range(n * m + 1)]
    f = [0] * (n + 1)

    # Step M1 in Knuth (Initialize)
    # Initial state - entire multiset in one part.
    for j in range(m):
        ps = pstack[j]
        ps.c = j
        ps.u = multiplicities[j]
        ps.v = multiplicities[j]

    # Other variables
    f[0] = 0
    a = 0
    lpart = 0
    f[1] = m
    b = m  # in general, current stack frame is from a to b - 1

    while True:
        while True:
            # Step M2 (Subtract v from u)
            k = b
            x = False
            for j in range(a, b):
                pstack[k].u = pstack[j].u - pstack[j].v
                if pstack[k].u == 0:
                    x = True
                elif not x:
                    pstack[k].c = pstack[j].c
                    pstack[k].v = min(pstack[j].v, pstack[k].u)
                    x = pstack[k].u < pstack[j].v
                    k = k + 1
                else:  # x is True
                    pstack[k].c = pstack[j].c
                    pstack[k].v = pstack[k].u
                    k = k + 1
                # Note: x is True iff v has changed

            # Step M3 (Push if nonzero.)
            if k > b:
                a = b
                b = k
                lpart = lpart + 1
                f[lpart + 1] = b
                # Return to M2
            else:
                break  # Continue to M4

        # M4  Visit a partition
        state = [f, lpart, pstack]
        yield state

        # M5 (Decrease v)
        while True:
            j = b-1
            while (pstack[j].v == 0):
                j = j - 1
            if j == a and pstack[j].v == 1:
                # M6 (Backtrack)
                if lpart == 0:
                    return
                lpart = lpart - 1
                b = a
                a = f[lpart]
                # Return to M5
            else:
                pstack[j].v = pstack[j].v - 1
                for k in range(j + 1, b):
                    pstack[k].v = pstack[k].u
                break  # GOTO M2

# --------------- Visitor functions for multiset partitions ---------------
# A visitor takes the partition state generated by
# multiset_partitions_taocp or other enumerator, and produces useful
# output (such as the actual partition).


def factoring_visitor(state, primes):
    """Use with multiset_partitions_taocp to enumerate the ways a
    number can be expressed as a product of factors.  For this usage,
    the exponents of the prime factors of a number are arguments to
    the partition enumerator, while the corresponding prime factors
    are input here.

    Examples
    ========

    To enumerate the factorings of a number we can think of the elements of the
    partition as being the prime factors and the multiplicities as being their
    exponents.

    >>> from sympy.utilities.enumerative import factoring_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> from sympy import factorint
    >>> primes, multiplicities = zip(*factorint(24).items())
    >>> primes
    (2, 3)
    >>> multiplicities
    (3, 1)
    >>> states = multiset_partitions_taocp(multiplicities)
    >>> list(factoring_visitor(state, primes) for state in states)
    [[24], [8, 3], [12, 2], [4, 6], [4, 2, 3], [6, 2, 2], [2, 2, 2, 3]]
    """
    f, lpart, pstack = state
    factoring = []
    for i in range(lpart + 1):
        factor = 1
        for ps in pstack[f[i]: f[i + 1]]:
            if ps.v > 0:
                factor *= primes[ps.c] ** ps.v
        factoring.append(factor)
    return factoring


def list_visitor(state, components):
    """Return a list of lists to represent the partition.

    Examples
    ========

    >>> from sympy.utilities.enumerative import list_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> states = multiset_partitions_taocp([1, 2, 1])
    >>> s = next(states)
    >>> list_visitor(s, 'abc')  # for multiset 'a b b c'
    [['a', 'b', 'b', 'c']]
    >>> s = next(states)
    >>> list_visitor(s, [1, 2, 3])  # for multiset '1 2 2 3
    [[1, 2, 2], [3]]
    """
    f, lpart, pstack = state

    partition = []
    for i in range(lpart+1):
        part = []
        for ps in pstack[f[i]:f[i+1]]:
            if ps.v > 0:
                part.extend([components[ps.c]] * ps.v)
        partition.append(part)

    return partition


class MultisetPartitionTraverser():
    """
    Has methods to ``enumerate`` and ``count`` the partitions of a multiset.

    This implements a refactored and extended version of Knuth's algorithm
    7.1.2.5M [AOCP]_."

    The enumeration methods of this class are generators and return
    data structures which can be interpreted by the same visitor
    functions used for the output of ``multiset_partitions_taocp``.

    Examples
    ========

    >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
    >>> m = MultisetPartitionTraverser()
    >>> m.count_partitions([4,4,4,2])
    127750
    >>> m.count_partitions([3,3,3])
    686

    See Also
    ========

    multiset_partitions_taocp
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [AOCP] Algorithm 7.1.2.5M in Volume 4A, Combinatoral Algorithms,
           Part 1, of The Art of Computer Programming, by Donald Knuth.

    .. [Factorisatio] On a Problem of Oppenheim concerning
           "Factorisatio Numerorum" E. R. Canfield, Paul Erdos, Carl
           Pomerance, JOURNAL OF NUMBER THEORY, Vol. 17, No. 1. August
           1983.  See section 7 for a description of an algorithm
           similar to Knuth's.

    .. [Yorgey] Generating Multiset Partitions, Brent Yorgey, The
           Monad.Reader, Issue 8, September 2007.

    """

    def __init__(self):
        self.debug = False
        # TRACING variables.  These are useful for gathering
        # statistics on the algorithm itself, but have no particular
        # benefit to a user of the code.
        self.k1 = 0
        self.k2 = 0
        self.p1 = 0
        self.pstack = None
        self.f = None
        self.lpart = 0
        self.discarded = 0
        # dp_stack is list of lists of (part_key, start_count) pairs
        self.dp_stack = []

        # dp_map is map part_key-> count, where count represents the
        # number of multiset which are descendants of a part with this
        # key, **or any of its decrements**

        # Thus, when we find a part in the map, we add its count
        # value to the running total, cut off the enumeration, and
        # backtrack

        if not hasattr(self, 'dp_map'):
            self.dp_map = {}

    def db_trace(self, msg):
        """Useful for understanding/debugging the algorithms.  Not
        generally activated in end-user code."""
        if self.debug:
            # XXX: animation_visitor is undefined... Clearly this does not
            # work and was not tested. Previous code in comments below.
            raise RuntimeError
            #letters = 'abcdefghijklmnopqrstuvwxyz'
            #state = [self.f, self.lpart, self.pstack]
            #print("DBG:", msg,
            #      ["".join(part) for part in list_visitor(state, letters)],
            #      animation_visitor(state))

    #
    # Helper methods for enumeration
    #
    def _initialize_enumeration(self, multiplicities):
        """Allocates and initializes the partition stack.

        This is called from the enumeration/counting routines, so
        there is no need to call it separately."""

        num_components = len(multiplicities)
        # cardinality is the total number of elements, whether or not distinct
        cardinality = sum(multiplicities)

        # pstack is the partition stack, which is segmented by
        # f into parts.
        self.pstack = [PartComponent() for i in
                       range(num_components * cardinality + 1)]
        self.f = [0] * (cardinality + 1)

        # Initial state - entire multiset in one part.
        for j in range(num_components):
            ps = self.pstack[j]
            ps.c = j
            ps.u = multiplicities[j]
            ps.v = multiplicities[j]

        self.f[0] = 0
        self.f[1] = num_components
        self.lpart = 0

    # The decrement_part() method corresponds to step M5 in Knuth's
    # algorithm.  This is the base version for enum_all().  Modified
    # versions of this method are needed if we want to restrict
    # sizes of the partitions produced.
    def decrement_part(self, part):
        """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        If you think of the v values in the part as a multi-digit
        integer (least significant digit on the right) this is
        basically decrementing that integer, but with the extra
        constraint that the leftmost digit cannot be decremented to 0.

        Parameters
        ==========

        part
           The part, represented as a list of PartComponent objects,
           which is to be decremented.

        """
        plen = len(part)
        for j in range(plen - 1, -1, -1):
            if j == 0 and part[j].v > 1 or j > 0 and part[j].v > 0:
                # found val to decrement
                part[j].v -= 1
                # Reset trailing parts back to maximum
                for k in range(j + 1, plen):
                    part[k].v = part[k].u
                return True
        return False

    # Version to allow number of parts to be bounded from above.
    # Corresponds to (a modified) step M5.
    def decrement_part_small(self, part, ub):
        """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        ub
            the maximum number of parts allowed in a partition
            returned by the calling traversal.

        Notes
        =====

        The goal of this modification of the ordinary decrement method
        is to fail (meaning that the subtree rooted at this part is to
        be skipped) when it can be proved that this part can only have
        child partitions which are larger than allowed by ``ub``. If a
        decision is made to fail, it must be accurate, otherwise the
        enumeration will miss some partitions.  But, it is OK not to
        capture all the possible failures -- if a part is passed that
        should not be, the resulting too-large partitions are filtered
        by the enumeration one level up.  However, as is usual in
        constrained enumerations, failing early is advantageous.

        The tests used by this method catch the most common cases,
        although this implementation is by no means the last word on
        this problem.  The tests include:

        1) ``lpart`` must be less than ``ub`` by at least 2.  This is because
           once a part has been decremented, the partition
           will gain at least one child in the spread step.

        2) If the leading component of the part is about to be
           decremented, check for how many parts will be added in
           order to use up the unallocated multiplicity in that
           leading component, and fail if this number is greater than
           allowed by ``ub``.  (See code for the exact expression.)  This
           test is given in the answer to Knuth's problem 7.2.1.5.69.

        3) If there is *exactly* enough room to expand the leading
           component by the above test, check the next component (if
           it exists) once decrementing has finished.  If this has
           ``v == 0``, this next component will push the expansion over the
           limit by 1, so fail.
        """
        if self.lpart >= ub - 1:
            self.p1 += 1  # increment to keep track of usefulness of tests
            return False
        plen = len(part)
        for j in range(plen - 1, -1, -1):
            # Knuth's mod, (answer to problem 7.2.1.5.69)
            if j == 0 and (part[0].v - 1)*(ub - self.lpart) < part[0].u:
                self.k1 += 1
                return False

            if j == 0 and part[j].v > 1 or j > 0 and part[j].v > 0:
                # found val to decrement
                part[j].v -= 1
                # Reset trailing parts back to maximum
                for k in range(j + 1, plen):
                    part[k].v = part[k].u

                # Have now decremented part, but are we doomed to
                # failure when it is expanded?  Check one oddball case
                # that turns out to be surprisingly common - exactly
                # enough room to expand the leading component, but no
                # room for the second component, which has v=0.
                if (plen > 1 and part[1].v == 0 and
                    (part[0].u - part[0].v) ==
                        ((ub - self.lpart - 1) * part[0].v)):
                    self.k2 += 1
                    self.db_trace("Decrement fails test 3")
                    return False
                return True
        return False

    def decrement_part_large(self, part, amt, lb):
        """Decrements part, while respecting size constraint.

        A part can have no children which are of sufficient size (as
        indicated by ``lb``) unless that part has sufficient
        unallocated multiplicity.  When enforcing the size constraint,
        this method will decrement the part (if necessary) by an
        amount needed to ensure sufficient unallocated multiplicity.

        Returns True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        amt
            Can only take values 0 or 1.  A value of 1 means that the
            part must be decremented, and then the size constraint is
            enforced.  A value of 0 means just to enforce the ``lb``
            size constraint.

        lb
            The partitions produced by the calling enumeration must
            have more parts than this value.

        """

        if amt == 1:
            # In this case we always need to decrement, *before*
            # enforcing the "sufficient unallocated multiplicity"
            # constraint.  Easiest for this is just to call the
            # regular decrement method.
            if not self.decrement_part(part):
                return False

        # Next, perform any needed additional decrementing to respect
        # "sufficient unallocated multiplicity" (or fail if this is
        # not possible).
        min_unalloc = lb - self.lpart
        if min_unalloc <= 0:
            return True
        total_mult = sum(pc.u for pc in part)
        total_alloc = sum(pc.v for pc in part)
        if total_mult <= min_unalloc:
            return False

        deficit = min_unalloc - (total_mult - total_alloc)
        if deficit <= 0:
            return True

        for i in range(len(part) - 1, -1, -1):
            if i == 0:
                if part[0].v > deficit:
                    part[0].v -= deficit
                    return True
                else:
                    return False  # This shouldn't happen, due to above check
            else:
                if part[i].v >= deficit:
                    part[i].v -= deficit
                    return True
                else:
                    deficit -= part[i].v
                    part[i].v = 0

    def decrement_part_range(self, part, lb, ub):
        """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        ub
            the maximum number of parts allowed in a partition
            returned by the calling traversal.

        lb
            The partitions produced by the calling enumeration must
            have more parts than this value.

        Notes
        =====

        Combines the constraints of _small and _large decrement
        methods.  If returns success, part has been decremented at
        least once, but perhaps by quite a bit more if needed to meet
        the lb constraint.
        """

        # Constraint in the range case is just enforcing both the
        # constraints from _small and _large cases.  Note the 0 as the
        # second argument to the _large call -- this is the signal to
        # decrement only as needed to for constraint enforcement.  The
        # short circuiting and left-to-right order of the 'and'
        # operator is important for this to work correctly.
        return self.decrement_part_small(part, ub) and \
            self.decrement_part_large(part, 0, lb)

    def spread_part_multiplicity(self):
        """Returns True if a new part has been created, and
        adjusts pstack, f and lpart as needed.

        Notes
        =====

        Spreads unallocated multiplicity from the current top part
        into a new part created above the current on the stack.  This
        new part is constrained to be less than or equal to the old in
        terms of the part ordering.

        This call does nothing (and returns False) if the current top
        part has no unallocated multiplicity.

        """
        j = self.f[self.lpart]  # base of current top part
        k = self.f[self.lpart + 1]  # ub of current; potential base of next
        base = k  # save for later comparison

        changed = False  # Set to true when the new part (so far) is
                         # strictly less than (as opposed to less than
                         # or equal) to the old.
        for j in range(self.f[self.lpart], self.f[self.lpart + 1]):
            self.pstack[k].u = self.pstack[j].u - self.pstack[j].v
            if self.pstack[k].u == 0:
                changed = True
            else:
                self.pstack[k].c = self.pstack[j].c
                if changed:  # Put all available multiplicity in this part
                    self.pstack[k].v = self.pstack[k].u
                else:  # Still maintaining ordering constraint
                    if self.pstack[k].u < self.pstack[j].v:
                        self.pstack[k].v = self.pstack[k].u
                        changed = True
                    else:
                        self.pstack[k].v = self.pstack[j].v
                k = k + 1
        if k > base:
            # Adjust for the new part on stack
            self.lpart = self.lpart + 1
            self.f[self.lpart + 1] = k
            return True
        return False

    def top_part(self):
        """Return current top part on the stack, as a slice of pstack.

        """
        return self.pstack[self.f[self.lpart]:self.f[self.lpart + 1]]

    # Same interface and functionality as multiset_partitions_taocp(),
    # but some might find this refactored version easier to follow.
    def enum_all(self, multiplicities):
        """Enumerate the partitions of a multiset.

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_all([2,2])
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b', 'b']],
        [['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'a'], ['b'], ['b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']],
        [['a', 'b'], ['a'], ['b']],
        [['a'], ['a'], ['b', 'b']],
        [['a'], ['a'], ['b'], ['b']]]

        See Also
        ========

        multiset_partitions_taocp:
            which provides the same result as this method, but is
            about twice as fast.  Hence, enum_all is primarily useful
            for testing.  Also see the function for a discussion of
            states and visitors.

        """
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                pass

            # M4  Visit a partition
            state = [self.f, self.lpart, self.pstack]
            yield state

            # M5 (Decrease v)
            while not self.decrement_part(self.top_part()):
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1

    def enum_small(self, multiplicities, ub):
        """Enumerate multiset partitions with no more than ``ub`` parts.

        Equivalent to enum_range(multiplicities, 0, ub)

        Parameters
        ==========

        multiplicities
             list of multiplicities of the components of the multiset.

        ub
            Maximum number of parts

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_small([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b', 'b']],
        [['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        The implementation is based, in part, on the answer given to
        exercise 69, in Knuth [AOCP]_.

        See Also
        ========

        enum_all, enum_large, enum_range

        """

        # Keep track of iterations which do not yield a partition.
        # Clearly, we would like to keep this number small.
        self.discarded = 0
        if ub <= 0:
            return
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                self.db_trace('spread 1')
                if self.lpart >= ub:
                    self.discarded += 1
                    self.db_trace('  Discarding')
                    self.lpart = ub - 2
                    break
            else:
                # M4  Visit a partition
                state = [self.f, self.lpart, self.pstack]
                yield state

            # M5 (Decrease v)
            while not self.decrement_part_small(self.top_part(), ub):
                self.db_trace("Failed decrement, going to backtrack")
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1
                self.db_trace("Backtracked to")
            self.db_trace("decrement ok, about to expand")

    def enum_large(self, multiplicities, lb):
        """Enumerate the partitions of a multiset with lb < num(parts)

        Equivalent to enum_range(multiplicities, lb, sum(multiplicities))

        Parameters
        ==========

        multiplicities
            list of multiplicities of the components of the multiset.

        lb
            Number of parts in the partition must be greater than
            this lower bound.


        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_large([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a'], ['b'], ['b']],
        [['a', 'b'], ['a'], ['b']],
        [['a'], ['a'], ['b', 'b']],
        [['a'], ['a'], ['b'], ['b']]]

        See Also
        ========

        enum_all, enum_small, enum_range

        """
        self.discarded = 0
        if lb >= sum(multiplicities):
            return
        self._initialize_enumeration(multiplicities)
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            while self.spread_part_multiplicity():
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    # Failure here should be rare/impossible
                    self.discarded += 1
                    good_partition = False
                    break

            # M4  Visit a partition
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state

            # M5 (Decrease v)
            while not self.decrement_part_large(self.top_part(), 1, lb):
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1

    def enum_range(self, multiplicities, lb, ub):

        """Enumerate the partitions of a multiset with
        ``lb < num(parts) <= ub``.

        In particular, if partitions with exactly ``k`` parts are
        desired, call with ``(multiplicities, k - 1, k)``.  This
        method generalizes enum_all, enum_small, and enum_large.

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_range([2,2], 1, 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        """
        # combine the constraints of the _large and _small
        # enumerations.
        self.discarded = 0
        if ub <= 0 or lb >= sum(multiplicities):
            return
        self._initialize_enumeration(multiplicities)
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            while self.spread_part_multiplicity():
                self.db_trace("spread 1")
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    # Failure here - possible in range case?
                    self.db_trace("  Discarding (large cons)")
                    self.discarded += 1
                    good_partition = False
                    break
                elif self.lpart >= ub:
                    self.discarded += 1
                    good_partition = False
                    self.db_trace("  Discarding small cons")
                    self.lpart = ub - 2
                    break

            # M4  Visit a partition
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state

            # M5 (Decrease v)
            while not self.decrement_part_range(self.top_part(), lb, ub):
                self.db_trace("Failed decrement, going to backtrack")
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1
                self.db_trace("Backtracked to")
            self.db_trace("decrement ok, about to expand")

    def count_partitions_slow(self, multiplicities):
        """Returns the number of partitions of a multiset whose elements
        have the multiplicities given in ``multiplicities``.

        Primarily for comparison purposes.  It follows the same path as
        enumerate, and counts, rather than generates, the partitions.

        See Also
        ========

        count_partitions
            Has the same calling interface, but is much faster.

        """
        # number of partitions so far in the enumeration
        self.pcount = 0
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                pass

            # M4  Visit (count) a partition
            self.pcount += 1

            # M5 (Decrease v)
            while not self.decrement_part(self.top_part()):
                # M6 (Backtrack)
                if self.lpart == 0:
                    return self.pcount
                self.lpart -= 1

    def count_partitions(self, multiplicities):
        """Returns the number of partitions of a multiset whose components
        have the multiplicities given in ``multiplicities``.

        For larger counts, this method is much faster than calling one
        of the enumerators and counting the result.  Uses dynamic
        programming to cut down on the number of nodes actually
        explored.  The dictionary used in order to accelerate the
        counting process is stored in the ``MultisetPartitionTraverser``
        object and persists across calls.  If the user does not
        expect to call ``count_partitions`` for any additional
        multisets, the object should be cleared to save memory.  On
        the other hand, the cache built up from one count run can
        significantly speed up subsequent calls to ``count_partitions``,
        so it may be advantageous not to clear the object.

        Examples
        ========

        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> m.count_partitions([9,8,2])
        288716
        >>> m.count_partitions([2,2])
        9
        >>> del m

        Notes
        =====

        If one looks at the workings of Knuth's algorithm M [AOCP]_, it
        can be viewed as a traversal of a binary tree of parts.  A
        part has (up to) two children, the left child resulting from
        the spread operation, and the right child from the decrement
        operation.  The ordinary enumeration of multiset partitions is
        an in-order traversal of this tree, and with the partitions
        corresponding to paths from the root to the leaves. The
        mapping from paths to partitions is a little complicated,
        since the partition would contain only those parts which are
        leaves or the parents of a spread link, not those which are
        parents of a decrement link.

        For counting purposes, it is sufficient to count leaves, and
        this can be done with a recursive in-order traversal.  The
        number of leaves of a subtree rooted at a particular part is a
        function only of that part itself, so memoizing has the
        potential to speed up the counting dramatically.

        This method follows a computational approach which is similar
        to the hypothetical memoized recursive function, but with two
        differences:

        1) This method is iterative, borrowing its structure from the
           other enumerations and maintaining an explicit stack of
           parts which are in the process of being counted.  (There
           may be multisets which can be counted reasonably quickly by
           this implementation, but which would overflow the default
           Python recursion limit with a recursive implementation.)

        2) Instead of using the part data structure directly, a more
           compact key is constructed.  This saves space, but more
           importantly coalesces some parts which would remain
           separate with physical keys.

        Unlike the enumeration functions, there is currently no _range
        version of count_partitions.  If someone wants to stretch
        their brain, it should be possible to construct one by
        memoizing with a histogram of counts rather than a single
        count, and combining the histograms.
        """
        # number of partitions so far in the enumeration
        self.pcount = 0

        # dp_stack is list of lists of (part_key, start_count) pairs
        self.dp_stack = []

        self._initialize_enumeration(multiplicities)
        pkey = part_key(self.top_part())
        self.dp_stack.append([(pkey, 0), ])
        while True:
            while self.spread_part_multiplicity():
                pkey = part_key(self.top_part())
                if pkey in self.dp_map:
                    # Already have a cached value for the count of the
                    # subtree rooted at this part.  Add it to the
                    # running counter, and break out of the spread
                    # loop.  The -1 below is to compensate for the
                    # leaf that this code path would otherwise find,
                    # and which gets incremented for below.

                    self.pcount += (self.dp_map[pkey] - 1)
                    self.lpart -= 1
                    break
                else:
                    self.dp_stack.append([(pkey, self.pcount), ])

            # M4  count a leaf partition
            self.pcount += 1

            # M5 (Decrease v)
            while not self.decrement_part(self.top_part()):
                # M6 (Backtrack)
                for key, oldcount in self.dp_stack.pop():
                    self.dp_map[key] = self.pcount - oldcount
                if self.lpart == 0:
                    return self.pcount
                self.lpart -= 1

            # At this point have successfully decremented the part on
            # the stack and it does not appear in the cache.  It needs
            # to be added to the list at the top of dp_stack
            pkey = part_key(self.top_part())
            self.dp_stack[-1].append((pkey, self.pcount),)


def part_key(part):
    """Helper for MultisetPartitionTraverser.count_partitions that
    creates a key for ``part``, that only includes information which can
    affect the count for that part.  (Any irrelevant information just
    reduces the effectiveness of dynamic programming.)

    Notes
    =====

    This member function is a candidate for future exploration. There
    are likely symmetries that can be exploited to coalesce some
    ``part_key`` values, and thereby save space and improve
    performance.

    """
    # The component number is irrelevant for counting partitions, so
    # leave it out of the memo key.
    rval = []
    for ps in part:
        rval.append(ps.u)
        rval.append(ps.v)
    return tuple(rval)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\zeta.py ===
from __future__ import print_function

from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static

@defun
def stieltjes(ctx, n, a=1):
    n = ctx.convert(n)
    a = ctx.convert(a)
    if n < 0:
        return ctx.bad_domain("Stieltjes constants defined for n >= 0")
    if hasattr(ctx, "stieltjes_cache"):
        stieltjes_cache = ctx.stieltjes_cache
    else:
        stieltjes_cache = ctx.stieltjes_cache = {}
    if a == 1:
        if n == 0:
            return +ctx.euler
        if n in stieltjes_cache:
            prec, s = stieltjes_cache[n]
            if prec >= ctx.prec:
                return +s
    mag = 1
    def f(x):
        xa = x/a
        v = (xa-ctx.j)*ctx.ln(a-ctx.j*x)**n/(1+xa**2)/(ctx.exp(2*ctx.pi*x)-1)
        return ctx._re(v) / mag
    orig = ctx.prec
    try:
        # Normalize integrand by approx. magnitude to
        # speed up quadrature (which uses absolute error)
        if n > 50:
            ctx.prec = 20
            mag = ctx.quad(f, [0,ctx.inf], maxdegree=3)
        ctx.prec = orig + 10 + int(n**0.5)
        s = ctx.quad(f, [0,ctx.inf], maxdegree=20)
        v = ctx.ln(a)**n/(2*a) - ctx.ln(a)**(n+1)/(n+1) + 2*s/a*mag
    finally:
        ctx.prec = orig
    if a == 1 and ctx.isint(n):
        stieltjes_cache[n] = (ctx.prec, v)
    return +v

@defun_wrapped
def siegeltheta(ctx, t, derivative=0):
    d = int(derivative)
    if  (t == ctx.inf or t == ctx.ninf):
        if d < 2:
            if t == ctx.ninf and d == 0:
                return ctx.ninf
            return ctx.inf
        else:
            return ctx.zero
    if d == 0:
        if ctx._im(t):
            # XXX: cancellation occurs
            a = ctx.loggamma(0.25+0.5j*t)
            b = ctx.loggamma(0.25-0.5j*t)
            return -ctx.ln(ctx.pi)/2*t - 0.5j*(a-b)
        else:
            if ctx.isinf(t):
                return t
            return ctx._im(ctx.loggamma(0.25+0.5j*t)) - ctx.ln(ctx.pi)/2*t
    if d > 0:
        a = (-0.5j)**(d-1)*ctx.polygamma(d-1, 0.25-0.5j*t)
        b = (0.5j)**(d-1)*ctx.polygamma(d-1, 0.25+0.5j*t)
        if ctx._im(t):
            if d == 1:
                return -0.5*ctx.log(ctx.pi)+0.25*(a+b)
            else:
                return 0.25*(a+b)
        else:
            if d == 1:
                return ctx._re(-0.5*ctx.log(ctx.pi)+0.25*(a+b))
            else:
                return ctx._re(0.25*(a+b))

@defun_wrapped
def grampoint(ctx, n):
    # asymptotic expansion, from
    # http://mathworld.wolfram.com/GramPoint.html
    g = 2*ctx.pi*ctx.exp(1+ctx.lambertw((8*n+1)/(8*ctx.e)))
    return ctx.findroot(lambda t: ctx.siegeltheta(t)-ctx.pi*n, g)


@defun_wrapped
def siegelz(ctx, t, **kwargs):
    d = int(kwargs.get("derivative", 0))
    t = ctx.convert(t)
    t1 = ctx._re(t)
    t2 = ctx._im(t)
    prec = ctx.prec
    try:
        if abs(t1) > 500*prec and t2**2 < t1:
            v = ctx.rs_z(t, d)
            if ctx._is_real_type(t):
                return ctx._re(v)
            return v
    except NotImplementedError:
        pass
    ctx.prec += 21
    e1 = ctx.expj(ctx.siegeltheta(t))
    z = ctx.zeta(0.5+ctx.j*t)
    if d == 0:
        v = e1*z
        ctx.prec=prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z1 = ctx.zeta(0.5+ctx.j*t, derivative=1)
    theta1 = ctx.siegeltheta(t, derivative=1)
    if d == 1:
        v =  ctx.j*e1*(z1+z*theta1)
        ctx.prec=prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z2 = ctx.zeta(0.5+ctx.j*t, derivative=2)
    theta2 = ctx.siegeltheta(t, derivative=2)
    comb1 = theta1**2-ctx.j*theta2
    if d == 2:
        def terms():
            return [2*z1*theta1, z2, z*comb1]
        v = ctx.sum_accurately(terms, 1)
        v =  -e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    ctx.prec += 10
    z3 = ctx.zeta(0.5+ctx.j*t, derivative=3)
    theta3 = ctx.siegeltheta(t, derivative=3)
    comb2 = theta1**3-3*ctx.j*theta1*theta2-theta3
    if d == 3:
        def terms():
            return  [3*theta1*z2, 3*z1*comb1, z3+z*comb2]
        v = ctx.sum_accurately(terms, 1)
        v =  -ctx.j*e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    z4 = ctx.zeta(0.5+ctx.j*t, derivative=4)
    theta4 = ctx.siegeltheta(t, derivative=4)
    def terms():
        return [theta1**4, -6*ctx.j*theta1**2*theta2, -3*theta2**2,
            -4*theta1*theta3, ctx.j*theta4]
    comb3 = ctx.sum_accurately(terms, 1)
    if d == 4:
        def terms():
            return  [6*theta1**2*z2, -6*ctx.j*z2*theta2, 4*theta1*z3,
                 4*z1*comb2, z4, z*comb3]
        v = ctx.sum_accurately(terms, 1)
        v =  e1*v
        ctx.prec = prec
        if ctx._is_real_type(t):
            return ctx._re(v)
        return +v
    if d > 4:
        h = lambda x: ctx.siegelz(x, derivative=4)
        return ctx.diff(h, t, n=d-4)


_zeta_zeros = [
14.134725142,21.022039639,25.010857580,30.424876126,32.935061588,
37.586178159,40.918719012,43.327073281,48.005150881,49.773832478,
52.970321478,56.446247697,59.347044003,60.831778525,65.112544048,
67.079810529,69.546401711,72.067157674,75.704690699,77.144840069,
79.337375020,82.910380854,84.735492981,87.425274613,88.809111208,
92.491899271,94.651344041,95.870634228,98.831194218,101.317851006,
103.725538040,105.446623052,107.168611184,111.029535543,111.874659177,
114.320220915,116.226680321,118.790782866,121.370125002,122.946829294,
124.256818554,127.516683880,129.578704200,131.087688531,133.497737203,
134.756509753,138.116042055,139.736208952,141.123707404,143.111845808,
146.000982487,147.422765343,150.053520421,150.925257612,153.024693811,
156.112909294,157.597591818,158.849988171,161.188964138,163.030709687,
165.537069188,167.184439978,169.094515416,169.911976479,173.411536520,
174.754191523,176.441434298,178.377407776,179.916484020,182.207078484,
184.874467848,185.598783678,187.228922584,189.416158656,192.026656361,
193.079726604,195.265396680,196.876481841,198.015309676,201.264751944,
202.493594514,204.189671803,205.394697202,207.906258888,209.576509717,
211.690862595,213.347919360,214.547044783,216.169538508,219.067596349,
220.714918839,221.430705555,224.007000255,224.983324670,227.421444280,
229.337413306,231.250188700,231.987235253,233.693404179,236.524229666,
]

def _load_zeta_zeros(url):
    import urllib
    d = urllib.urlopen(url)
    L = [float(x) for x in d.readlines()]
    # Sanity check
    assert round(L[0]) == 14
    _zeta_zeros[:] = L

@defun
def oldzetazero(ctx, n, url='http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1'):
    n = int(n)
    if n < 0:
        return ctx.zetazero(-n).conjugate()
    if n == 0:
        raise ValueError("n must be nonzero")
    if n > len(_zeta_zeros) and n <= 100000:
        _load_zeta_zeros(url)
    if n > len(_zeta_zeros):
        raise NotImplementedError("n too large for zetazeros")
    return ctx.mpc(0.5, ctx.findroot(ctx.siegelz, _zeta_zeros[n-1]))

@defun_wrapped
def riemannr(ctx, x):
    if x == 0:
        return ctx.zero
    # Check if a simple asymptotic estimate is accurate enough
    if abs(x) > 1000:
        a = ctx.li(x)
        b = 0.5*ctx.li(ctx.sqrt(x))
        if abs(b) < abs(a)*ctx.eps:
            return a
    if abs(x) < 0.01:
        # XXX
        ctx.prec += int(-ctx.log(abs(x),2))
    # Sum Gram's series
    s = t = ctx.one
    u = ctx.ln(x)
    k = 1
    while abs(t) > abs(s)*ctx.eps:
        t = t * u / k
        s += t / (k * ctx._zeta_int(k+1))
        k += 1
    return s

@defun_static
def primepi(ctx, x):
    x = int(x)
    if x < 2:
        return 0
    return len(ctx.list_primes(x))

# TODO: fix the interface wrt contexts
@defun_wrapped
def primepi2(ctx, x):
    x = int(x)
    if x < 2:
        return ctx._iv.zero
    if x < 2657:
        return ctx._iv.mpf(ctx.primepi(x))
    mid = ctx.li(x)
    # Schoenfeld's estimate for x >= 2657, assuming RH
    err = ctx.sqrt(x,rounding='u')*ctx.ln(x,rounding='u')/8/ctx.pi(rounding='d')
    a = ctx.floor((ctx._iv.mpf(mid)-err).a, rounding='d')
    b = ctx.ceil((ctx._iv.mpf(mid)+err).b, rounding='u')
    return ctx._iv.mpf([a,b])

@defun_wrapped
def primezeta(ctx, s):
    if ctx.isnan(s):
        return s
    if ctx.re(s) <= 0:
        raise ValueError("prime zeta function defined only for re(s) > 0")
    if s == 1:
        return ctx.inf
    if s == 0.5:
        return ctx.mpc(ctx.ninf, ctx.pi)
    r = ctx.re(s)
    if r > ctx.prec:
        return 0.5**s
    else:
        wp = ctx.prec + int(r)
        def terms():
            orig = ctx.prec
            # zeta ~ 1+eps; need to set precision
            # to get logarithm accurately
            k = 0
            while 1:
                k += 1
                u = ctx.moebius(k)
                if not u:
                    continue
                ctx.prec = wp
                t = u*ctx.ln(ctx.zeta(k*s))/k
                if not t:
                    return
                #print ctx.prec, ctx.nstr(t)
                ctx.prec = orig
                yield t
    return ctx.sum_accurately(terms)

# TODO: for bernpoly and eulerpoly, ensure that all exact zeros are covered

@defun_wrapped
def bernpoly(ctx, n, z):
    # Slow implementation:
    #return sum(ctx.binomial(n,k)*ctx.bernoulli(k)*z**(n-k) for k in xrange(0,n+1))
    n = int(n)
    if n < 0:
        raise ValueError("Bernoulli polynomials only defined for n >= 0")
    if z == 0 or (z == 1 and n > 1):
        return ctx.bernoulli(n)
    if z == 0.5:
        return (ctx.ldexp(1,1-n)-1)*ctx.bernoulli(n)
    if n <= 3:
        if n == 0: return z ** 0
        if n == 1: return z - 0.5
        if n == 2: return (6*z*(z-1)+1)/6
        if n == 3: return z*(z*(z-1.5)+0.5)
    if ctx.isinf(z):
        return z ** n
    if ctx.isnan(z):
        return z
    if abs(z) > 2:
        def terms():
            t = ctx.one
            yield t
            r = ctx.one/z
            k = 1
            while k <= n:
                t = t*(n+1-k)/k*r
                if not (k > 2 and k & 1):
                    yield t*ctx.bernoulli(k)
                k += 1
        return ctx.sum_accurately(terms) * z**n
    else:
        def terms():
            yield ctx.bernoulli(n)
            t = ctx.one
            k = 1
            while k <= n:
                t = t*(n+1-k)/k * z
                m = n-k
                if not (m > 2 and m & 1):
                    yield t*ctx.bernoulli(m)
                k += 1
        return ctx.sum_accurately(terms)

@defun_wrapped
def eulerpoly(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError("Euler polynomials only defined for n >= 0")
    if n <= 2:
        if n == 0: return z ** 0
        if n == 1: return z - 0.5
        if n == 2: return z*(z-1)
    if ctx.isinf(z):
        return z**n
    if ctx.isnan(z):
        return z
    m = n+1
    if z == 0:
        return -2*(ctx.ldexp(1,m)-1)*ctx.bernoulli(m)/m * z**0
    if z == 1:
        return 2*(ctx.ldexp(1,m)-1)*ctx.bernoulli(m)/m * z**0
    if z == 0.5:
        if n % 2:
            return ctx.zero
        # Use exact code for Euler numbers
        if n < 100 or n*ctx.mag(0.46839865*n) < ctx.prec*0.25:
            return ctx.ldexp(ctx._eulernum(n), -n)
    # http://functions.wolfram.com/Polynomials/EulerE2/06/01/02/01/0002/
    def terms():
        t = ctx.one
        k = 0
        w = ctx.ldexp(1,n+2)
        while 1:
            v = n-k+1
            if not (v > 2 and v & 1):
                yield (2-w)*ctx.bernoulli(v)*t
            k += 1
            if k > n:
                break
            t = t*z*(n-k+2)/k
            w *= 0.5
    return ctx.sum_accurately(terms) / m

@defun
def eulernum(ctx, n, exact=False):
    n = int(n)
    if exact:
        return int(ctx._eulernum(n))
    if n < 100:
        return ctx.mpf(ctx._eulernum(n))
    if n % 2:
        return ctx.zero
    return ctx.ldexp(ctx.eulerpoly(n,0.5), n)

# TODO: this should be implemented low-level
def polylog_series(ctx, s, z):
    tol = +ctx.eps
    l = ctx.zero
    k = 1
    zk = z
    while 1:
        term = zk / k**s
        l += term
        if abs(term) < tol:
            break
        zk *= z
        k += 1
    return l

def polylog_continuation(ctx, n, z):
    if n < 0:
        return z*0
    twopij = 2j * ctx.pi
    a = -twopij**n/ctx.fac(n) * ctx.bernpoly(n, ctx.ln(z)/twopij)
    if ctx._is_real_type(z) and z < 0:
        a = ctx._re(a)
    if ctx._im(z) < 0 or (ctx._im(z) == 0 and ctx._re(z) >= 1):
        a -= twopij*ctx.ln(z)**(n-1)/ctx.fac(n-1)
    return a

def polylog_unitcircle(ctx, n, z):
    tol = +ctx.eps
    if n > 1:
        l = ctx.zero
        logz = ctx.ln(z)
        logmz = ctx.one
        m = 0
        while 1:
            if (n-m) != 1:
                term = ctx.zeta(n-m) * logmz / ctx.fac(m)
                if term and abs(term) < tol:
                    break
                l += term
            logmz *= logz
            m += 1
        l += ctx.ln(z)**(n-1)/ctx.fac(n-1)*(ctx.harmonic(n-1)-ctx.ln(-ctx.ln(z)))
    elif n < 1:  # else
        l = ctx.fac(-n)*(-ctx.ln(z))**(n-1)
        logz = ctx.ln(z)
        logkz = ctx.one
        k = 0
        while 1:
            b = ctx.bernoulli(k-n+1)
            if b:
                term = b*logkz/(ctx.fac(k)*(k-n+1))
                if abs(term) < tol:
                    break
                l -= term
            logkz *= logz
            k += 1
    else:
        raise ValueError
    if ctx._is_real_type(z) and z < 0:
        l = ctx._re(l)
    return l

def polylog_general(ctx, s, z):
    v = ctx.zero
    u = ctx.ln(z)
    if not abs(u) < 5: # theoretically |u| < 2*pi
        j = ctx.j
        v = 1-s
        y = ctx.ln(-z)/(2*ctx.pi*j)
        return ctx.gamma(v)*(j**v*ctx.zeta(v,0.5+y) + j**-v*ctx.zeta(v,0.5-y))/(2*ctx.pi)**v
    t = 1
    k = 0
    while 1:
        term = ctx.zeta(s-k) * t
        if abs(term) < ctx.eps:
            break
        v += term
        k += 1
        t *= u
        t /= k
    return ctx.gamma(1-s)*(-u)**(s-1) + v

@defun_wrapped
def polylog(ctx, s, z):
    s = ctx.convert(s)
    z = ctx.convert(z)
    if z == 1:
        return ctx.zeta(s)
    if z == -1:
        return -ctx.altzeta(s)
    if s == 0:
        return z/(1-z)
    if s == 1:
        return -ctx.ln(1-z)
    if s == -1:
        return z/(1-z)**2
    if abs(z) <= 0.75 or (not ctx.isint(s) and abs(z) < 0.9):
        return polylog_series(ctx, s, z)
    if abs(z) >= 1.4 and ctx.isint(s):
        return (-1)**(s+1)*polylog_series(ctx, s, 1/z) + polylog_continuation(ctx, int(ctx.re(s)), z)
    if ctx.isint(s):
        return polylog_unitcircle(ctx, int(ctx.re(s)), z)
    return polylog_general(ctx, s, z)

@defun_wrapped
def clsin(ctx, s, z, pi=False):
    if ctx.isint(s) and s < 0 and int(s) % 2 == 1:
        return z*0
    if pi:
        a = ctx.expjpi(z)
    else:
        a = ctx.expj(z)
    if ctx._is_real_type(z) and ctx._is_real_type(s):
        return ctx.im(ctx.polylog(s,a))
    b = 1/a
    return (-0.5j)*(ctx.polylog(s,a) - ctx.polylog(s,b))

@defun_wrapped
def clcos(ctx, s, z, pi=False):
    if ctx.isint(s) and s < 0 and int(s) % 2 == 0:
        return z*0
    if pi:
        a = ctx.expjpi(z)
    else:
        a = ctx.expj(z)
    if ctx._is_real_type(z) and ctx._is_real_type(s):
        return ctx.re(ctx.polylog(s,a))
    b = 1/a
    return 0.5*(ctx.polylog(s,a) + ctx.polylog(s,b))

@defun
def altzeta(ctx, s, **kwargs):
    try:
        return ctx._altzeta(s, **kwargs)
    except NotImplementedError:
        return ctx._altzeta_generic(s)

@defun_wrapped
def _altzeta_generic(ctx, s):
    if s == 1:
        return ctx.ln2 + 0*s
    return -ctx.powm1(2, 1-s) * ctx.zeta(s)

@defun
def zeta(ctx, s, a=1, derivative=0, method=None, **kwargs):
    d = int(derivative)
    if a == 1 and not (d or method):
        try:
            return ctx._zeta(s, **kwargs)
        except NotImplementedError:
            pass
    s = ctx.convert(s)
    prec = ctx.prec
    method = kwargs.get('method')
    verbose = kwargs.get('verbose')
    if (not s) and (not derivative):
        return ctx.mpf(0.5) - ctx._convert_param(a)[0]
    if a == 1 and method != 'euler-maclaurin':
        im = abs(ctx._im(s))
        re = abs(ctx._re(s))
        #if (im < prec or method == 'borwein') and not derivative:
        #    try:
        #        if verbose:
        #            print "zeta: Attempting to use the Borwein algorithm"
        #        return ctx._zeta(s, **kwargs)
        #    except NotImplementedError:
        #        if verbose:
        #            print "zeta: Could not use the Borwein algorithm"
        #        pass
        if abs(im) > 500*prec and 10*re < prec and derivative <= 4 or \
            method == 'riemann-siegel':
            try:   #  py2.4 compatible try block
                try:
                    if verbose:
                        print("zeta: Attempting to use the Riemann-Siegel algorithm")
                    return ctx.rs_zeta(s, derivative, **kwargs)
                except NotImplementedError:
                    if verbose:
                        print("zeta: Could not use the Riemann-Siegel algorithm")
                    pass
            finally:
                ctx.prec = prec
    if s == 1:
        return ctx.inf
    abss = abs(s)
    if abss == ctx.inf:
        if ctx.re(s) == ctx.inf:
            if d == 0:
                return ctx.one
            return ctx.zero
        return s*0
    elif ctx.isnan(abss):
        return 1/s
    if ctx.re(s) > 2*ctx.prec and a == 1 and not derivative:
        return ctx.one + ctx.power(2, -s)
    return +ctx._hurwitz(s, a, d, **kwargs)

@defun
def _hurwitz(ctx, s, a=1, d=0, **kwargs):
    prec = ctx.prec
    verbose = kwargs.get('verbose')
    try:
        extraprec = 10
        ctx.prec += extraprec
        # We strongly want to special-case rational a
        a, atype = ctx._convert_param(a)
        if ctx.re(s) < 0:
            if verbose:
                print("zeta: Attempting reflection formula")
            try:
                return _hurwitz_reflection(ctx, s, a, d, atype)
            except NotImplementedError:
                pass
            if verbose:
                print("zeta: Reflection formula failed")
        if verbose:
            print("zeta: Using the Euler-Maclaurin algorithm")
        while 1:
            ctx.prec = prec + extraprec
            T1, T2 = _hurwitz_em(ctx, s, a, d, prec+10, verbose)
            cancellation = ctx.mag(T1) - ctx.mag(T1+T2)
            if verbose:
                print("Term 1:", T1)
                print("Term 2:", T2)
                print("Cancellation:", cancellation, "bits")
            if cancellation < extraprec:
                return T1 + T2
            else:
                extraprec = max(2*extraprec, min(cancellation + 5, 100*prec))
                if extraprec > kwargs.get('maxprec', 100*prec):
                    raise ctx.NoConvergence("zeta: too much cancellation")
    finally:
        ctx.prec = prec

def _hurwitz_reflection(ctx, s, a, d, atype):
    # TODO: implement for derivatives
    if d != 0:
        raise NotImplementedError
    res = ctx.re(s)
    negs = -s
    # Integer reflection formula
    if ctx.isnpint(s):
        n = int(res)
        if n <= 0:
            return ctx.bernpoly(1-n, a) / (n-1)
    if not (atype == 'Q' or atype == 'Z'):
        raise NotImplementedError
    t = 1-s
    # We now require a to be standardized
    v = 0
    shift = 0
    b = a
    while ctx.re(b) > 1:
        b -= 1
        v -= b**negs
        shift -= 1
    while ctx.re(b) <= 0:
        v += b**negs
        b += 1
        shift += 1
    # Rational reflection formula
    try:
        p, q = a._mpq_
    except:
        assert a == int(a)
        p = int(a)
        q = 1
    p += shift*q
    assert 1 <= p <= q
    g = ctx.fsum(ctx.cospi(t/2-2*k*b)*ctx._hurwitz(t,(k,q)) \
        for k in range(1,q+1))
    g *= 2*ctx.gamma(t)/(2*ctx.pi*q)**t
    v += g
    return v

def _hurwitz_em(ctx, s, a, d, prec, verbose):
    # May not be converted at this point
    a = ctx.convert(a)
    tol = -prec
    # Estimate number of terms for Euler-Maclaurin summation; could be improved
    M1 = 0
    M2 = prec // 3
    N = M2
    lsum = 0
    # This speeds up the recurrence for derivatives
    if ctx.isint(s):
        s = int(ctx._re(s))
    s1 = s-1
    while 1:
        # Truncated L-series
        l = ctx._zetasum(s, M1+a, M2-M1-1, [d])[0][0]
        #if d:
        #    l = ctx.fsum((-ctx.ln(n+a))**d * (n+a)**negs for n in range(M1,M2))
        #else:
        #    l = ctx.fsum((n+a)**negs for n in range(M1,M2))
        lsum += l
        M2a = M2+a
        logM2a = ctx.ln(M2a)
        logM2ad = logM2a**d
        logs = [logM2ad]
        logr = 1/logM2a
        rM2a = 1/M2a
        M2as = M2a**(-s)
        if d:
            tailsum = ctx.gammainc(d+1, s1*logM2a) / s1**(d+1)
        else:
            tailsum = 1/((s1)*(M2a)**s1)
        tailsum += 0.5 * logM2ad * M2as
        U = [1]
        r = M2as
        fact = 2
        for j in range(1, N+1):
            # TODO: the following could perhaps be tidied a bit
            j2 = 2*j
            if j == 1:
                upds = [1]
            else:
                upds = [j2-2, j2-1]
            for m in upds:
                D = min(m,d+1)
                if m <= d:
                    logs.append(logs[-1] * logr)
                Un = [0]*(D+1)
                for i in xrange(D): Un[i] = (1-m-s)*U[i]
                for i in xrange(1,D+1): Un[i] += (d-(i-1))*U[i-1]
                U = Un
                r *= rM2a
            t = ctx.fdot(U, logs) * r * ctx.bernoulli(j2)/(-fact)
            tailsum += t
            if ctx.mag(t) < tol:
                return lsum, (-1)**d * tailsum
            fact *= (j2+1)*(j2+2)
        if verbose:
            print("Sum range:", M1, M2, "term magnitude", ctx.mag(t), "tolerance", tol)
        M1, M2 = M2, M2*2
        if ctx.re(s) < 0:
            N += N//2



@defun
def _zetasum(ctx, s, a, n, derivatives=[0], reflect=False):
    """
    Returns [xd0,xd1,...,xdr], [yd0,yd1,...ydr] where

    xdk = D^k     ( 1/a^s     +  1/(a+1)^s      +  ...  +  1/(a+n)^s     )
    ydk = D^k conj( 1/a^(1-s) +  1/(a+1)^(1-s)  +  ...  +  1/(a+n)^(1-s) )

    D^k = kth derivative with respect to s, k ranges over the given list of
    derivatives (which should consist of either a single element
    or a range 0,1,...r). If reflect=False, the ydks are not computed.
    """
    #print "zetasum", s, a, n
    # don't use the fixed-point code if there are large exponentials
    if abs(ctx.re(s)) < 0.5 * ctx.prec:
        try:
            return ctx._zetasum_fast(s, a, n, derivatives, reflect)
        except NotImplementedError:
            pass
    negs = ctx.fneg(s, exact=True)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1
    if not reflect:
        if not have_derivatives:
            return [ctx.fsum((a+k)**negs for k in xrange(n+1))], []
        if have_one_derivative:
            d = derivatives[0]
            x = ctx.fsum(ctx.ln(a+k)**d * (a+k)**negs for k in xrange(n+1))
            return [(-1)**d * x], []
    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd+1)
    xs = [ctx.zero for d in derivatives]
    if reflect:
        ys = [ctx.zero for d in derivatives]
    else:
        ys = []
    for k in xrange(n+1):
        w = a + k
        xterm = w ** negs
        if reflect:
            yterm = ctx.conj(ctx.one / (w * xterm))
        if have_derivatives:
            logw = -ctx.ln(w)
            if have_one_derivative:
                logw = logw ** maxd
                xs[0] += xterm * logw
                if reflect:
                    ys[0] += yterm * logw
            else:
                t = ctx.one
                for d in derivatives:
                    xs[d] += xterm * t
                    if reflect:
                        ys[d] += yterm * t
                    t *= logw
        else:
            xs[0] += xterm
            if reflect:
                ys[0] += yterm
    return xs, ys

@defun
def dirichlet(ctx, s, chi=[1], derivative=0):
    s = ctx.convert(s)
    q = len(chi)
    d = int(derivative)
    if d > 2:
        raise NotImplementedError("arbitrary order derivatives")
    prec = ctx.prec
    try:
        ctx.prec += 10
        if s == 1:
            have_pole = True
            for x in chi:
                if x and x != 1:
                    have_pole = False
                    h = +ctx.eps
                    ctx.prec *= 2*(d+1)
                    s += h
            if have_pole:
                return +ctx.inf
        z = ctx.zero
        for p in range(1,q+1):
            if chi[p%q]:
                if d == 1:
                    z += chi[p%q] * (ctx.zeta(s, (p,q), 1) - \
                        ctx.zeta(s, (p,q))*ctx.log(q))
                else:
                    z += chi[p%q] * ctx.zeta(s, (p,q))
        z /= q**s
    finally:
        ctx.prec = prec
    return +z


def secondzeta_main_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5*s, a*gamm**2, regularized=True)*gamm**(-s)
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        gamm = ctx.im(ctx.zetazero_memoized(n))
        term = f(n)
        mg = abs(term)
    err = 0
    if kwargs.get("error"):
        sg = ctx.re(s)
        err = 0.5*ctx.pi**(-1)*max(1,sg)*a**(sg-0.5)*ctx.log(gamm/(2*ctx.pi))*\
             ctx.gammainc(-0.5, a*gamm**2)/abs(ctx.gamma(s/2))
        err = abs(err)
    return +totsum, err, n

def secondzeta_prime_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5*(1-s),0.25*ctx.log(n)**2 * a**(-1))*\
        ((0.5*ctx.log(n))**(s-1))*ctx.mangoldt(n)/ctx.sqrt(n)/\
        (2*ctx.gamma(0.5*s)*ctx.sqrt(ctx.pi))
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 1
    while mg > tol or n < 9:
        totsum += term
        n += 1
        term = f(n)
        if term == 0:
            mg = ctx.inf
        else:
            mg = abs(term)
    if kwargs.get("error"):
        err = mg
    return +totsum, err, n

def secondzeta_exp_term(ctx, s, a):
    if ctx.isint(s) and ctx.re(s) <= 0:
        m = int(round(ctx.re(s)))
        if not m & 1:
            return ctx.mpf('-0.25')**(-m//2)
    tol = ctx.eps
    f = lambda n: (0.25*a)**n/((n+0.5*s)*ctx.fac(n))
    totsum = ctx.zero
    term = f(0)
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        term = f(n)
        mg = abs(term)
    v = a**(0.5*s)*totsum/ctx.gamma(0.5*s)
    return v

def secondzeta_singular_term(ctx, s, a, **kwargs):
    factor = a**(0.5*(s-1))/(4*ctx.sqrt(ctx.pi)*ctx.gamma(0.5*s))
    extraprec = ctx.mag(factor)
    ctx.prec += extraprec
    factor = a**(0.5*(s-1))/(4*ctx.sqrt(ctx.pi)*ctx.gamma(0.5*s))
    tol = ctx.eps
    f = lambda n: ctx.bernpoly(n,0.75)*(4*ctx.sqrt(a))**n*\
       ctx.gamma(0.5*n)/((s+n-1)*ctx.fac(n))
    totsum = ctx.zero
    mg1 = ctx.inf
    n = 1
    term = f(n)
    mg2 = abs(term)
    while mg2 > tol and mg2 <= mg1:
        totsum += term
        n += 1
        term = f(n)
        totsum += term
        n +=1
        term = f(n)
        mg1 = mg2
        mg2 = abs(term)
    totsum += term
    pole = -2*(s-1)**(-2)+(ctx.euler+ctx.log(16*ctx.pi**2*a))*(s-1)**(-1)
    st = factor*(pole+totsum)
    err = 0
    if kwargs.get("error"):
        if not ((mg2 > tol) and (mg2 <= mg1)):
            if mg2 <= tol:
                err = ctx.mpf(10)**int(ctx.log(abs(factor*tol),10))
            if mg2 > mg1:
                err = ctx.mpf(10)**int(ctx.log(abs(factor*mg1),10))
        err = max(err, ctx.eps*1.)
    ctx.prec -= extraprec
    return +st, err

@defun
def secondzeta(ctx, s, a = 0.015, **kwargs):
    r"""
    Evaluates the secondary zeta function `Z(s)`, defined for
    `\mathrm{Re}(s)>1` by

    .. math ::

        Z(s) = \sum_{n=1}^{\infty} \frac{1}{\tau_n^s}

    where `\frac12+i\tau_n` runs through the zeros of `\zeta(s)` with
    imaginary part positive.

    `Z(s)` extends to a meromorphic function on `\mathbb{C}`  with a
    double pole at `s=1` and  simple poles at the points `-2n` for
    `n=0`,  1, 2, ...

    **Examples**

        >>> from mpmath import *
        >>> mp.pretty = True; mp.dps = 15
        >>> secondzeta(2)
        0.023104993115419
        >>> xi = lambda s: 0.5*s*(s-1)*pi**(-0.5*s)*gamma(0.5*s)*zeta(s)
        >>> Xi = lambda t: xi(0.5+t*j)
        >>> chop(-0.5*diff(Xi,0,n=2)/Xi(0))
        0.023104993115419

    We may ask for an approximate error value::

        >>> secondzeta(0.5+100j, error=True)
        ((-0.216272011276718 - 0.844952708937228j), 2.22044604925031e-16)

    The function has poles at the negative odd integers,
    and dyadic rational values at the negative even integers::

        >>> mp.dps = 30
        >>> secondzeta(-8)
        -0.67236328125
        >>> secondzeta(-7)
        +inf

    **Implementation notes**

    The function is computed as sum of four terms `Z(s)=A(s)-P(s)+E(s)-S(s)`
    respectively main, prime, exponential and singular terms.
    The main term `A(s)` is computed from the zeros of zeta.
    The prime term depends on the von Mangoldt function.
    The singular term is responsible for the poles of the function.

    The four terms depends on a small parameter `a`. We may change the
    value of `a`. Theoretically this has no effect on the sum of the four
    terms, but in practice may be important.

    A smaller value of the parameter `a` makes `A(s)` depend on
    a smaller number of zeros of zeta, but `P(s)`  uses more values of
    von Mangoldt function.

    We may also add a verbose option to obtain data about the
    values of the four terms.

        >>> mp.dps = 10
        >>> secondzeta(0.5 + 40j, error=True, verbose=True)
        main term = (-30190318549.138656312556 - 13964804384.624622876523j)
            computed using 19 zeros of zeta
        prime term = (132717176.89212754625045 + 188980555.17563978290601j)
            computed using 9 values of the von Mangoldt function
        exponential term = (542447428666.07179812536 + 362434922978.80192435203j)
        singular term = (512124392939.98154322355 + 348281138038.65531023921j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

        >>> secondzeta(0.5 + 40j, a=0.04, error=True, verbose=True)
        main term = (-151962888.19606243907725 - 217930683.90210294051982j)
            computed using 9 zeros of zeta
        prime term = (2476659342.3038722372461 + 28711581821.921627163136j)
            computed using 37 values of the von Mangoldt function
        exponential term = (178506047114.7838188264 + 819674143244.45677330576j)
        singular term = (175877424884.22441310708 + 790744630738.28669174871j)
        ((0.059471043 + 0.3463514534j), 1.455191523e-11)

    Notice the great cancellation between the four terms. Changing `a`, the
    four terms are very different numbers but the cancellation gives
    the good value of Z(s).

    **References**

    A. Voros, Zeta functions for the Riemann zeros, Ann. Institute Fourier,
    53, (2003) 665--699.

    A. Voros, Zeta functions over Zeros of Zeta Functions, Lecture Notes
    of the Unione Matematica Italiana, Springer, 2009.
    """
    s = ctx.convert(s)
    a = ctx.convert(a)
    tol = ctx.eps
    if ctx.isint(s) and ctx.re(s) <= 1:
        if abs(s-1) < tol*1000:
            return ctx.inf
        m = int(round(ctx.re(s)))
        if m & 1:
            return ctx.inf
        else:
            return ((-1)**(-m//2)*\
                   ctx.fraction(8-ctx.eulernum(-m,exact=True),2**(-m+3)))
    prec = ctx.prec
    try:
        t3 = secondzeta_exp_term(ctx, s, a)
        extraprec = max(ctx.mag(t3),0)
        ctx.prec += extraprec + 3
        t1, r1, gt = secondzeta_main_term(ctx,s,a,error='True', verbose='True')
        t2, r2, pt = secondzeta_prime_term(ctx,s,a,error='True', verbose='True')
        t4, r4 = secondzeta_singular_term(ctx,s,a,error='True')
        t3 = secondzeta_exp_term(ctx, s, a)
        err = r1+r2+r4
        t = t1-t2+t3-t4
        if kwargs.get("verbose"):
            print('main term =', t1)
            print('    computed using', gt, 'zeros of zeta')
            print('prime term =', t2)
            print('    computed using', pt, 'values of the von Mangoldt function')
            print('exponential term =', t3)
            print('singular term =', t4)
    finally:
        ctx.prec = prec
    if kwargs.get("error"):
        w = max(ctx.mag(abs(t)),0)
        err = max(err*2**w, ctx.eps*1.*2**w)
        return +t, err
    return +t


@defun_wrapped
def lerchphi(ctx, z, s, a):
    r"""
    Gives the Lerch transcendent, defined for `|z| < 1` and
    `\Re{a} > 0` by

    .. math ::

        \Phi(z,s,a) = \sum_{k=0}^{\infty} \frac{z^k}{(a+k)^s}

    and generally by the recurrence `\Phi(z,s,a) = z \Phi(z,s,a+1) + a^{-s}`
    along with the integral representation valid for `\Re{a} > 0`

    .. math ::

        \Phi(z,s,a) = \frac{1}{2 a^s} +
                \int_0^{\infty} \frac{z^t}{(a+t)^s} dt -
                2 \int_0^{\infty} \frac{\sin(t \log z - s
                    \operatorname{arctan}(t/a)}{(a^2 + t^2)^{s/2}
                    (e^{2 \pi t}-1)} dt.

    The Lerch transcendent generalizes the Hurwitz zeta function :func:`zeta`
    (`z = 1`) and the polylogarithm :func:`polylog` (`a = 1`).

    **Examples**

    Several evaluations in terms of simpler functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> lerchphi(-1,2,0.5); 4*catalan
        3.663862376708876060218414
        3.663862376708876060218414
        >>> diff(lerchphi, (-1,-2,1), (0,1,0)); 7*zeta(3)/(4*pi**2)
        0.2131391994087528954617607
        0.2131391994087528954617607
        >>> lerchphi(-4,1,1); log(5)/4
        0.4023594781085250936501898
        0.4023594781085250936501898
        >>> lerchphi(-3+2j,1,0.5); 2*atanh(sqrt(-3+2j))/sqrt(-3+2j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)
        (1.142423447120257137774002 + 0.2118232380980201350495795j)

    Evaluation works for complex arguments and `|z| \ge 1`::

        >>> lerchphi(1+2j, 3-j, 4+2j)
        (0.002025009957009908600539469 + 0.003327897536813558807438089j)
        >>> lerchphi(-2,2,-2.5)
        -12.28676272353094275265944
        >>> lerchphi(10,10,10)
        (-4.462130727102185701817349e-11 - 1.575172198981096218823481e-12j)
        >>> lerchphi(10,10,-10.5)
        (112658784011940.5605789002 - 498113185.5756221777743631j)

    Some degenerate cases::

        >>> lerchphi(0,1,2)
        0.5
        >>> lerchphi(0,1,-2)
        -0.5

    Reduction to simpler functions::

        >>> lerchphi(1, 4.25+1j, 1)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> zeta(4.25+1j)
        (1.044674457556746668033975 - 0.04674508654012658932271226j)
        >>> lerchphi(1 - 0.5**10, 4.25+1j, 1)
        (1.044629338021507546737197 - 0.04667768813963388181708101j)
        >>> lerchphi(3, 4, 1)
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> polylog(4, 3) / 3
        (1.249503297023366545192592 - 0.2314252413375664776474462j)
        >>> lerchphi(3, 4, 1 - 0.5**10)
        (1.253978063946663945672674 - 0.2316736622836535468765376j)

    **References**

    1. [DLMF]_ section 25.14

    """
    if z == 0:
        return a ** (-s)
    # Faster, but these cases are useful for testing right now
    if z == 1:
        return ctx.zeta(s, a)
    if a == 1:
        return ctx.polylog(s, z) / z
    if ctx.re(a) < 1:
        if ctx.isnpint(a):
            raise ValueError("Lerch transcendent complex infinity")
        m = int(ctx.ceil(1-ctx.re(a)))
        v = ctx.zero
        zpow = ctx.one
        for n in xrange(m):
            v += zpow / (a+n)**s
            zpow *= z
        return zpow * ctx.lerchphi(z,s, a+m) + v
    g = ctx.ln(z)
    v = 1/(2*a**s) + ctx.gammainc(1-s, -a*g) * (-g)**(s-1) / z**a
    h = s / 2
    r = 2*ctx.pi
    f = lambda t: ctx.sin(s*ctx.atan(t/a)-t*g) / \
        ((a**2+t**2)**h * ctx.expm1(r*t))
    v += 2*ctx.quad(f, [0, ctx.inf])
    if not ctx.im(z) and not ctx.im(s) and not ctx.im(a) and ctx.re(z) < 1:
        v = ctx.chop(v)
    return v

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import collections
import collections.abc
import os
import typing
import warnings
from collections.abc import Sequence
from typing import Any

from onnxruntime.capi import _pybind_state as C

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    import onnxruntime


def get_ort_device_type(device_type: str, device_index) -> C.OrtDevice:
    if device_type == "cuda":
        return C.OrtDevice.cuda()
    elif device_type == "cann":
        return C.OrtDevice.cann()
    elif device_type == "cpu":
        return C.OrtDevice.cpu()
    elif device_type == "dml":
        return C.OrtDevice.dml()
    elif device_type == "webgpu":
        return C.OrtDevice.webgpu()
    elif device_type == "ort":
        return C.get_ort_device(device_index).device_type()
    else:
        raise Exception("Unsupported device type: " + device_type)


class AdapterFormat:
    """
    This class is used to create adapter files from python structures
    """

    def __init__(self, adapter=None) -> None:
        if adapter is None:
            self._adapter = C.AdapterFormat()
        else:
            self._adapter = adapter

    @staticmethod
    def read_adapter(file_path: os.PathLike) -> AdapterFormat:
        return AdapterFormat(C.AdapterFormat.read_adapter(file_path))

    def export_adapter(self, file_path: os.PathLike):
        """
        This function writes a file at the specified location
        in onnxrunitme adapter format containing Lora parameters.

        :param file_path: absolute path for the adapter
        """
        self._adapter.export_adapter(file_path)

    def get_format_version(self) -> int:
        return self._adapter.format_version

    def set_adapter_version(self, adapter_version: int) -> None:
        self._adapter.adapter_version = adapter_version

    def get_adapter_version(self) -> int:
        return self._adapter.adapter_version

    def set_model_version(self, model_version: int) -> None:
        self._adapter.model_version = model_version

    def get_model_version(self) -> int:
        return self._adapter.model_version

    def set_parameters(self, params: dict[str, OrtValue]) -> None:
        self._adapter.parameters = {k: v._ortvalue for k, v in params.items()}

    def get_parameters(self) -> dict[str, OrtValue]:
        return {k: OrtValue(v) for k, v in self._adapter.parameters.items()}


def check_and_normalize_provider_args(
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None,
    provider_options: Sequence[dict[Any, Any]] | None,
    available_provider_names: Sequence[str],
):
    """
    Validates the 'providers' and 'provider_options' arguments and returns a
        normalized version.

    :param providers: Optional sequence of providers in order of decreasing
        precedence. Values can either be provider names or tuples of
        (provider name, options dict).
    :param provider_options: Optional sequence of options dicts corresponding
        to the providers listed in 'providers'.
    :param available_provider_names: The available provider names.

    :return: Tuple of (normalized 'providers' sequence, normalized
        'provider_options' sequence).

    'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

    The normalized result is a tuple of:
    1. Sequence of provider names in the same order as 'providers'.
    2. Sequence of corresponding provider options dicts with string keys and
        values. Unspecified provider options yield empty dicts.
    """
    if providers is None:
        return [], []

    provider_name_to_options = collections.OrderedDict()

    def set_provider_options(name, options):
        if name not in available_provider_names:
            warnings.warn(
                "Specified provider '{}' is not in available provider names.Available providers: '{}'".format(
                    name, ", ".join(available_provider_names)
                )
            )

        if name in provider_name_to_options:
            warnings.warn(f"Duplicate provider '{name}' encountered, ignoring.")
            return

        normalized_options = {str(key): str(value) for key, value in options.items()}
        provider_name_to_options[name] = normalized_options

    if not isinstance(providers, collections.abc.Sequence):
        raise ValueError("'providers' should be a sequence.")

    if provider_options is not None:
        if not isinstance(provider_options, collections.abc.Sequence):
            raise ValueError("'provider_options' should be a sequence.")

        if len(providers) != len(provider_options):
            raise ValueError("'providers' and 'provider_options' should be the same length if both are given.")

        if not all(isinstance(provider, str) for provider in providers):
            raise ValueError("Only string values for 'providers' are supported if 'provider_options' is given.")

        if not all(isinstance(options_for_provider, dict) for options_for_provider in provider_options):
            raise ValueError("'provider_options' values must be dicts.")

        for name, options in zip(providers, provider_options, strict=False):
            set_provider_options(name, options)

    else:
        for provider in providers:
            if isinstance(provider, str):
                set_provider_options(provider, {})
            elif (
                isinstance(provider, tuple)
                and len(provider) == 2
                and isinstance(provider[0], str)
                and isinstance(provider[1], dict)
            ):
                set_provider_options(provider[0], provider[1])
            else:
                raise ValueError("'providers' values must be either strings or (string, dict) tuples.")

    return list(provider_name_to_options.keys()), list(provider_name_to_options.values())


class Session:
    """
    This is the main class used to run a model.
    """

    def __init__(self):
        # self._sess is managed by the derived class and relies on bindings from C.InferenceSession
        self._sess = None
        self._enable_fallback = True

    def get_session_options(self) -> onnxruntime.SessionOptions:
        "Return the session options. See :class:`onnxruntime.SessionOptions`."
        return self._sess_options

    def get_inputs(self) -> Sequence[onnxruntime.NodeArg]:
        "Return the inputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._inputs_meta

    def get_outputs(self) -> Sequence[onnxruntime.NodeArg]:
        "Return the outputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._outputs_meta

    def get_overridable_initializers(self) -> Sequence[onnxruntime.NodeArg]:
        "Return the inputs (including initializers) metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._overridable_initializers

    def get_modelmeta(self) -> onnxruntime.ModelMetadata:
        "Return the metadata. See :class:`onnxruntime.ModelMetadata`."
        return self._model_meta

    def get_providers(self) -> Sequence[str]:
        "Return list of registered execution providers."
        return self._providers

    def get_provider_options(self):
        "Return registered execution providers' configurations."
        return self._provider_options

    def set_providers(self, providers=None, provider_options=None) -> None:
        """
        Register the input list of execution providers. The underlying session is re-created.

        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example
        `['CUDAExecutionProvider', 'CPUExecutionProvider']`
        means execute a node using CUDAExecutionProvider if capable,
        otherwise execute using CPUExecutionProvider.
        """
        # recreate the underlying C.InferenceSession
        self._reset_session(providers, provider_options)

    def disable_fallback(self) -> None:
        """
        Disable session.run() fallback mechanism.
        """
        self._enable_fallback = False

    def enable_fallback(self) -> None:
        """
        Enable session.Run() fallback mechanism. If session.Run() fails due to an internal Execution Provider failure,
        reset the Execution Providers enabled for this session.
        If GPU is enabled, fall back to CUDAExecutionProvider.
        otherwise fall back to CPUExecutionProvider.
        """
        self._enable_fallback = True

    def _validate_input(self, feed_input_names):
        missing_input_names = []
        for input in self._inputs_meta:
            if input.name not in feed_input_names and not input.type.startswith("optional"):
                missing_input_names.append(input.name)
        if missing_input_names:
            raise ValueError(
                f"Required inputs ({missing_input_names}) are missing from input feed ({feed_input_names})."
            )

    def run(self, output_names, input_feed, run_options=None) -> Sequence[np.ndarray | SparseTensor | list | dict]:
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param run_options: See :class:`onnxruntime.RunOptions`.
        :return: list of results, every result is either a numpy array,
            a sparse tensor, a list or a dictionary.

        ::

            sess.run([output_name], {input_name: x})
        """
        self._validate_input(list(input_feed.keys()))
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return self._sess.run(output_names, input_feed, run_options)
        except C.EPFail as err:
            if self._enable_fallback:
                print(f"EP Error: {err!s} using {self._providers}")
                print(f"Falling back to {self._fallback_providers} and retrying.")
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                return self._sess.run(output_names, input_feed, run_options)
            raise

    def run_async(self, output_names, input_feed, callback, user_data, run_options=None):
        """
        Compute the predictions asynchronously in a separate cxx thread from ort intra-op threadpool.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param callback: python function that accept array of results, and a status string on error.
            The callback will be invoked by a cxx thread from ort intra-op threadpool.
        :param run_options: See :class:`onnxruntime.RunOptions`.

        ::
            class MyData:
                def __init__(self):
                    # ...
                def save_results(self, results):
                    # ...

            def callback(results: np.ndarray, user_data: MyData, err: str) -> None:
              if err:
                 print (err)
              else:
                # save results to user_data

            sess.run_async([output_name], {input_name: x}, callback)
        """
        self._validate_input(list(input_feed.keys()))
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        return self._sess.run_async(output_names, input_feed, callback, user_data, run_options)

    def run_with_ort_values(self, output_names, input_dict_ort_values, run_options=None) -> Sequence[OrtValue]:
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_dict_ort_values: dictionary ``{ input_name: input_ort_value }``
            See ``OrtValue`` class how to create `OrtValue`
            from numpy array or `SparseTensor`
        :param run_options: See :class:`onnxruntime.RunOptions`.
        :return: an array of `OrtValue`

        ::

            sess.run([output_name], {input_name: x})
        """

        def invoke(sess, output_names, input_dict_ort_values, run_options):
            input_dict = {}
            for n, v in input_dict_ort_values.items():
                input_dict[n] = v._get_c_value()
            result = sess.run_with_ort_values(input_dict, output_names, run_options)
            if not isinstance(result, C.OrtValueVector):
                raise TypeError("run_with_ort_values() must return a instance of type 'OrtValueVector'.")
            ort_values = [OrtValue(v) for v in result]
            return ort_values

        self._validate_input(list(input_dict_ort_values.keys()))
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return invoke(self._sess, output_names, input_dict_ort_values, run_options)
        except C.EPFail as err:
            if self._enable_fallback:
                print(f"EP Error: {err!s} using {self._providers}")
                print(f"Falling back to {self._fallback_providers} and retrying.")
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                return invoke(self._sess, output_names, input_dict_ort_values, run_options)
            raise

    def end_profiling(self):
        """
        End profiling and return results in a file.

        The results are stored in a filename if the option
        :meth:`onnxruntime.SessionOptions.enable_profiling`.
        """
        return self._sess.end_profiling()

    def get_profiling_start_time_ns(self):
        """
        Return the nanoseconds of profiling's start time
        Comparable to time.monotonic_ns() after Python 3.3
        On some platforms, this timer may not be as precise as nanoseconds
        For instance, on Windows and MacOS, the precision will be ~100ns
        """
        return self._sess.get_profiling_start_time_ns

    def io_binding(self) -> IOBinding:
        "Return an onnxruntime.IOBinding object`."
        return IOBinding(self)

    def run_with_iobinding(self, iobinding, run_options=None):
        """
        Compute the predictions.

        :param iobinding: the iobinding object that has graph inputs/outputs bind.
        :param run_options: See :class:`onnxruntime.RunOptions`.
        """
        self._sess.run_with_iobinding(iobinding._iobinding, run_options)

    def get_tuning_results(self):
        return self._sess.get_tuning_results()

    def set_tuning_results(self, results, *, error_on_invalid=False):
        return self._sess.set_tuning_results(results, error_on_invalid)

    def run_with_ortvaluevector(self, run_options, feed_names, feeds, fetch_names, fetches, fetch_devices):
        """
        Compute the predictions similar to other run_*() methods but with minimal C++/Python conversion overhead.

        :param run_options: See :class:`onnxruntime.RunOptions`.
        :param feed_names: list of input names.
        :param feeds: list of input OrtValue.
        :param fetch_names: list of output names.
        :param fetches: list of output OrtValue.
        :param fetch_devices: list of output devices.
        """
        self._sess.run_with_ortvaluevector(run_options, feed_names, feeds, fetch_names, fetches, fetch_devices)


class InferenceSession(Session):
    """
    This is the main class used to run a model.
    """

    def __init__(
        self,
        path_or_bytes: str | bytes | os.PathLike,
        sess_options: onnxruntime.SessionOptions | None = None,
        providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        **kwargs,
    ) -> None:
        """
        :param path_or_bytes: Filename or serialized ONNX or ORT format model in a byte string.
        :param sess_options: Session options.
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:

        ::

            so = onnxruntime.SessionOptions()
            # so.add_session_config_entry('session.load_model_format', 'ONNX') or
            so.add_session_config_entry('session.load_model_format', 'ORT')

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example
        `['CUDAExecutionProvider', 'CPUExecutionProvider']`
        means execute a node using `CUDAExecutionProvider`
        if capable, otherwise execute using `CPUExecutionProvider`.
        """
        super().__init__()

        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._model_path = os.fspath(path_or_bytes)
            self._model_bytes = None
        elif isinstance(path_or_bytes, bytes):
            self._model_path = None
            self._model_bytes = path_or_bytes  # TODO: This is bad as we're holding the memory indefinitely
        else:
            raise TypeError(f"Unable to load from type '{type(path_or_bytes)}'")

        self._sess_options = sess_options
        self._sess_options_initial = sess_options
        self._enable_fallback = True
        if "read_config_from_model" in kwargs:
            self._read_config_from_model = int(kwargs["read_config_from_model"]) == 1
        else:
            self._read_config_from_model = os.environ.get("ORT_LOAD_CONFIG_FROM_MODEL") == "1"

        # internal parameters that we don't expect to be used in general so aren't documented
        disabled_optimizers = kwargs.get("disabled_optimizers")

        try:
            self._create_inference_session(providers, provider_options, disabled_optimizers)
        except (ValueError, RuntimeError) as e:
            if self._enable_fallback:
                try:
                    print("*************** EP Error ***************")
                    print(f"EP Error {e} when using {providers}")
                    print(f"Falling back to {self._fallback_providers} and retrying.")
                    print("****************************************")
                    self._create_inference_session(self._fallback_providers, None)
                    # Fallback only once.
                    self.disable_fallback()
                    return
                except Exception as fallback_error:
                    raise fallback_error from e
            # Fallback is disabled. Raise the original error.
            raise e

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        if "NvTensorRTRTXExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "NvTensorRTRTXExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "NvExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
        sess.initialize_session(providers, provider_options, disabled_optimizers)

        self._sess = sess
        self._sess_options = self._sess.session_options
        self._inputs_meta = self._sess.inputs_meta
        self._outputs_meta = self._sess.outputs_meta
        self._overridable_initializers = self._sess.overridable_initializers
        self._model_meta = self._sess.model_meta
        self._providers = self._sess.get_providers()
        self._provider_options = self._sess.get_provider_options()
        self._profiling_start_time_ns = self._sess.get_profiling_start_time_ns

    def _reset_session(self, providers, provider_options) -> None:
        "release underlying session object."
        # meta data references session internal structures
        # so they must be set to None to decrement _sess reference count.
        self._sess_options = None
        self._inputs_meta = None
        self._outputs_meta = None
        self._overridable_initializers = None
        self._model_meta = None
        self._providers = None
        self._provider_options = None
        self._profiling_start_time_ns = None

        # create a new C.InferenceSession
        self._sess = None
        self._sess_options = self._sess_options_initial
        self._create_inference_session(providers, provider_options)

    def _register_ep_custom_ops(self, session_options, providers, provider_options, available_providers):
        for i in range(len(providers)):
            if providers[i] in available_providers and providers[i] == "TensorrtExecutionProvider":
                C.register_tensorrt_plugins_as_custom_ops(session_options, provider_options[i])
            elif (
                isinstance(providers[i], tuple)
                and providers[i][0] in available_providers
                and providers[i][0] == "TensorrtExecutionProvider"
            ):
                C.register_tensorrt_plugins_as_custom_ops(session_options, providers[i][1])

            if providers[i] in available_providers and providers[i] == "NvTensorRTRTXExecutionProvider":
                C.register_nv_tensorrt_rtx_plugins_as_custom_ops(session_options, provider_options[i])
            elif (
                isinstance(providers[i], tuple)
                and providers[i][0] in available_providers
                and providers[i][0] == "NvTensorrtRTXExecutionProvider"
            ):
                C.register_nv_tensorrt_rtx_plugins_as_custom_ops(session_options, providers[i][1])


class IOBinding:
    """
    This class provides API to bind input/output to a specified device, e.g. GPU.
    """

    def __init__(self, session: Session):
        self._iobinding = C.SessionIOBinding(session._sess)
        self._numpy_obj_references = {}

    def bind_cpu_input(self, name, arr_on_cpu):
        """
        bind an input to array on CPU
        :param name: input name
        :param arr_on_cpu: input values as a python array on CPU
        """
        # Hold a reference to the numpy object as the bound OrtValue is backed
        # directly by the data buffer of the numpy object and so the numpy object
        # must be around until this IOBinding instance is around
        self._numpy_obj_references[name] = arr_on_cpu
        self._iobinding.bind_input(name, arr_on_cpu)

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        """
        :param name: input name
        :param device_type: e.g. cpu, cuda, cann
        :param device_id: device id, e.g. 0
        :param element_type: input element type. It can be either numpy type (like numpy.float32) or an integer for onnx type (like onnx.TensorProto.BFLOAT16)
        :param shape: input shape
        :param buffer_ptr: memory pointer to input data
        """
        self._iobinding.bind_input(
            name,
            C.OrtDevice(
                get_ort_device_type(device_type, device_id),
                C.OrtDevice.default_memory(),
                device_id,
            ),
            element_type,
            shape,
            buffer_ptr,
        )

    def bind_ortvalue_input(self, name, ortvalue):
        """
        :param name: input name
        :param ortvalue: OrtValue instance to bind
        """
        self._iobinding.bind_ortvalue_input(name, ortvalue._ortvalue)

    def synchronize_inputs(self):
        self._iobinding.synchronize_inputs()

    def bind_output(
        self,
        name,
        device_type="cpu",
        device_id=0,
        element_type=None,
        shape=None,
        buffer_ptr=None,
    ):
        """
        :param name: output name
        :param device_type: e.g. cpu, cuda, cann, cpu by default
        :param device_id: device id, e.g. 0
        :param element_type: output element type. It can be either numpy type (like numpy.float32) or an integer for onnx type (like onnx.TensorProto.BFLOAT16)
        :param shape: output shape
        :param buffer_ptr: memory pointer to output data
        """

        # Follow the `if` path when the user has not provided any pre-allocated buffer but still
        # would like to bind an output to a specific device (e.g. cuda).
        # Pre-allocating an output buffer may not be an option for the user as :
        # (1) They may not want to use a custom allocator specific to the device they want to bind the output to,
        # in which case ORT will allocate the memory for the user
        # (2) The output has a dynamic shape and hence the size of the buffer may not be fixed across runs
        if buffer_ptr is None:
            self._iobinding.bind_output(
                name,
                C.OrtDevice(
                    get_ort_device_type(device_type, device_id),
                    C.OrtDevice.default_memory(),
                    device_id,
                ),
            )
        else:
            if element_type is None or shape is None:
                raise ValueError("`element_type` and `shape` are to be provided if pre-allocated memory is provided")
            self._iobinding.bind_output(
                name,
                C.OrtDevice(
                    get_ort_device_type(device_type, device_id),
                    C.OrtDevice.default_memory(),
                    device_id,
                ),
                element_type,
                shape,
                buffer_ptr,
            )

    def bind_ortvalue_output(self, name, ortvalue):
        """
        :param name: output name
        :param ortvalue: OrtValue instance to bind
        """
        self._iobinding.bind_ortvalue_output(name, ortvalue._ortvalue)

    def synchronize_outputs(self):
        self._iobinding.synchronize_outputs()

    def get_outputs(self):
        """
        Returns the output OrtValues from the Run() that preceded the call.
        The data buffer of the obtained OrtValues may not reside on CPU memory
        """
        outputs = self._iobinding.get_outputs()
        if not isinstance(outputs, C.OrtValueVector):
            raise TypeError("get_outputs() must return an instance of type 'OrtValueVector'.")
        return [OrtValue(ortvalue) for ortvalue in outputs]

    def get_outputs_as_ortvaluevector(self):
        return self._iobinding.get_outputs()

    def copy_outputs_to_cpu(self):
        """Copy output contents to CPU."""
        return self._iobinding.copy_outputs_to_cpu()

    def clear_binding_inputs(self):
        self._iobinding.clear_binding_inputs()

    def clear_binding_outputs(self):
        self._iobinding.clear_binding_outputs()


class OrtValue:
    """
    A data structure that supports all ONNX data formats (tensors and non-tensors) that allows users
    to place the data backing these on a device, for example, on a CUDA supported device.
    This class provides APIs to construct and deal with OrtValues.
    """

    def __init__(self, ortvalue: C.OrtValue, numpy_obj: np.ndarray | None = None):
        if isinstance(ortvalue, C.OrtValue):
            self._ortvalue = ortvalue
            # Hold a ref count to the numpy object if the OrtValue is backed directly
            # by its data buffer so that it isn't destroyed when the OrtValue is in use
            self._numpy_obj = numpy_obj
        else:
            # An end user won't hit this error
            raise ValueError(
                "`Provided ortvalue` needs to be of type `onnxruntime.capi.onnxruntime_pybind11_state.OrtValue`"
            )

    def _get_c_value(self) -> C.OrtValue:
        return self._ortvalue

    @classmethod
    def ortvalue_from_numpy(cls, numpy_obj: np.ndarray, /, device_type="cpu", device_id=0) -> OrtValue:
        """
        Factory method to construct an OrtValue (which holds a Tensor) from a given Numpy object
        A copy of the data in the Numpy object is held by the OrtValue only if the device is NOT cpu

        :param numpy_obj: The Numpy object to construct the OrtValue from
        :param device_type: e.g. cpu, cuda, cann, cpu by default
        :param device_id: device id, e.g. 0
        """
        # Hold a reference to the numpy object (if device_type is 'cpu') as the OrtValue
        # is backed directly by the data buffer of the numpy object and so the numpy object
        # must be around until this OrtValue instance is around
        return cls(
            C.OrtValue.ortvalue_from_numpy(
                numpy_obj,
                C.OrtDevice(
                    get_ort_device_type(device_type, device_id),
                    C.OrtDevice.default_memory(),
                    device_id,
                ),
            ),
            numpy_obj if device_type.lower() == "cpu" else None,
        )

    @classmethod
    def ortvalue_from_numpy_with_onnx_type(cls, data: np.ndarray, /, onnx_element_type: int) -> OrtValue:
        """
        This method creates an instance of OrtValue on top of the numpy array.
        No data copy is made and the lifespan of the resulting OrtValue should never
        exceed the lifespan of bytes object. The API attempts to reinterpret
        the data type which is expected to be the same size. This is useful
        when we want to use an ONNX data type that is not supported by numpy.

        :param data: numpy.ndarray.
        :param onnx_element_type: a valid onnx TensorProto::DataType enum value
        """
        return cls(C.OrtValue.ortvalue_from_numpy_with_onnx_type(data, onnx_element_type), data)

    @classmethod
    def ortvalue_from_shape_and_type(
        cls, shape: Sequence[int], element_type, device_type: str = "cpu", device_id: int = 0
    ) -> OrtValue:
        """
        Factory method to construct an OrtValue (which holds a Tensor) from given shape and element_type

        :param shape: List of integers indicating the shape of the OrtValue
        :param element_type: The data type of the elements. It can be either numpy type (like numpy.float32) or an integer for onnx type (like onnx.TensorProto.BFLOAT16).
        :param device_type: e.g. cpu, cuda, cann, cpu by default
        :param device_id: device id, e.g. 0
        """
        # Integer for onnx element type (see https://onnx.ai/onnx/api/mapping.html).
        # This is helpful for some data type (like TensorProto.BFLOAT16) that is not available in numpy.
        if isinstance(element_type, int):
            return cls(
                C.OrtValue.ortvalue_from_shape_and_onnx_type(
                    shape,
                    element_type,
                    C.OrtDevice(
                        get_ort_device_type(device_type, device_id),
                        C.OrtDevice.default_memory(),
                        device_id,
                    ),
                )
            )

        return cls(
            C.OrtValue.ortvalue_from_shape_and_type(
                shape,
                element_type,
                C.OrtDevice(
                    get_ort_device_type(device_type, device_id),
                    C.OrtDevice.default_memory(),
                    device_id,
                ),
            )
        )

    @classmethod
    def ort_value_from_sparse_tensor(cls, sparse_tensor: SparseTensor) -> OrtValue:
        """
        The function will construct an OrtValue instance from a valid SparseTensor
        The new instance of OrtValue will assume the ownership of sparse_tensor
        """
        return cls(C.OrtValue.ort_value_from_sparse_tensor(sparse_tensor._get_c_tensor()))

    def as_sparse_tensor(self) -> SparseTensor:
        """
        The function will return SparseTensor contained in this OrtValue
        """
        return SparseTensor(self._ortvalue.as_sparse_tensor())

    def data_ptr(self) -> int:
        """
        Returns the address of the first element in the OrtValue's data buffer
        """
        return self._ortvalue.data_ptr()

    def device_name(self) -> str:
        """
        Returns the name of the device where the OrtValue's data buffer resides e.g. cpu, cuda, cann
        """
        return self._ortvalue.device_name().lower()

    def shape(self) -> Sequence[int]:
        """
        Returns the shape of the data in the OrtValue
        """
        return self._ortvalue.shape()

    def data_type(self) -> str:
        """
        Returns the data type of the data in the OrtValue. E.g. 'tensor(int64)'
        """
        return self._ortvalue.data_type()

    def element_type(self) -> int:
        """
        Returns the proto type of the data in the OrtValue
        if the OrtValue is a tensor.
        """
        return self._ortvalue.element_type()

    def has_value(self) -> bool:
        """
        Returns True if the OrtValue corresponding to an
        optional type contains data, else returns False
        """
        return self._ortvalue.has_value()

    def is_tensor(self) -> bool:
        """
        Returns True if the OrtValue contains a Tensor, else returns False
        """
        return self._ortvalue.is_tensor()

    def is_sparse_tensor(self) -> bool:
        """
        Returns True if the OrtValue contains a SparseTensor, else returns False
        """
        return self._ortvalue.is_sparse_tensor()

    def is_tensor_sequence(self) -> bool:
        """
        Returns True if the OrtValue contains a Tensor Sequence, else returns False
        """
        return self._ortvalue.is_tensor_sequence()

    def numpy(self) -> np.ndarray:
        """
        Returns a Numpy object from the OrtValue.
        Valid only for OrtValues holding Tensors. Throws for OrtValues holding non-Tensors.
        Use accessors to gain a reference to non-Tensor objects such as SparseTensor
        """
        return self._ortvalue.numpy()

    def update_inplace(self, np_arr) -> None:
        """
        Update the OrtValue in place with a new Numpy array. The numpy contents
        are copied over to the device memory backing the OrtValue. It can be used
        to update the input valuess for an InferenceSession with CUDA graph
        enabled or other scenarios where the OrtValue needs to be updated while
        the memory address can not be changed.
        """
        self._ortvalue.update_inplace(np_arr)


class OrtDevice:
    """
    A data structure that exposes the underlying C++ OrtDevice
    """

    def __init__(self, c_ort_device):
        """
        Internal constructor
        """
        if isinstance(c_ort_device, C.OrtDevice):
            self._ort_device = c_ort_device
        else:
            raise ValueError(
                "`Provided object` needs to be of type `onnxruntime.capi.onnxruntime_pybind11_state.OrtDevice`"
            )

    def _get_c_device(self):
        """
        Internal accessor to underlying object
        """
        return self._ort_device

    @staticmethod
    def make(ort_device_name, device_id):
        return OrtDevice(
            C.OrtDevice(
                get_ort_device_type(ort_device_name, device_id),
                C.OrtDevice.default_memory(),
                device_id,
            )
        )

    def device_id(self):
        return self._ort_device.device_id()

    def device_type(self):
        return self._ort_device.device_type()


class SparseTensor:
    """
    A data structure that project the C++ SparseTensor object
    The class provides API to work with the object.
    Depending on the format, the class will hold more than one buffer
    depending on the format
    """

    def __init__(self, sparse_tensor: C.SparseTensor):
        """
        Internal constructor
        """
        if isinstance(sparse_tensor, C.SparseTensor):
            self._tensor = sparse_tensor
        else:
            # An end user won't hit this error
            raise ValueError(
                "`Provided object` needs to be of type `onnxruntime.capi.onnxruntime_pybind11_state.SparseTensor`"
            )

    def _get_c_tensor(self) -> C.SparseTensor:
        return self._tensor

    @classmethod
    def sparse_coo_from_numpy(
        cls,
        dense_shape: npt.NDArray[np.int64],
        values: np.ndarray,
        coo_indices: npt.NDArray[np.int64],
        ort_device: OrtDevice,
    ) -> SparseTensor:
        """
        Factory method to construct a SparseTensor in COO format from given arguments

        :param dense_shape: 1-D  numpy array(int64) or a python list that contains a dense_shape of the sparse tensor
            must be on cpu memory
        :param values: a homogeneous, contiguous 1-D numpy array that contains non-zero elements of the tensor
            of a type.
        :param coo_indices:  contiguous numpy array(int64) that contains COO indices for the tensor. coo_indices may
            have a 1-D shape when it contains a linear index of non-zero values and its length must be equal to
            that of the values. It can also be of 2-D shape, in which has it contains pairs of coordinates for
            each of the nnz values and its length must be exactly twice of the values length.
        :param ort_device: - describes the backing memory owned by the supplied nummpy arrays. Only CPU memory is
            suppored for non-numeric data types.

        For primitive types, the method will map values and coo_indices arrays into native memory and will use
        them as backing storage. It will increment the reference count for numpy arrays and it will decrement it
        on GC. The buffers may reside in any storage either CPU or GPU.
        For strings and objects, it will create a copy of the arrays in CPU memory as ORT does not support those
        on other devices and their memory can not be mapped.
        """
        return cls(C.SparseTensor.sparse_coo_from_numpy(dense_shape, values, coo_indices, ort_device._get_c_device()))

    @classmethod
    def sparse_csr_from_numpy(
        cls,
        dense_shape: npt.NDArray[np.int64],
        values: np.ndarray,
        inner_indices: npt.NDArray[np.int64],
        outer_indices: npt.NDArray[np.int64],
        ort_device: OrtDevice,
    ) -> SparseTensor:
        """
        Factory method to construct a SparseTensor in CSR format from given arguments

        :param dense_shape: 1-D numpy array(int64) or a python list that contains a dense_shape of the
            sparse tensor (rows, cols) must be on cpu memory
        :param values: a  contiguous, homogeneous 1-D numpy array that contains non-zero elements of the tensor
            of a type.
        :param inner_indices:  contiguous 1-D numpy array(int64) that contains CSR inner indices for the tensor.
            Its length must be equal to that of the values.
        :param outer_indices:  contiguous 1-D numpy array(int64) that contains CSR outer indices for the tensor.
            Its length must be equal to the number of rows + 1.
        :param ort_device: - describes the backing memory owned by the supplied nummpy arrays. Only CPU memory is
            suppored for non-numeric data types.

        For primitive types, the method will map values and indices arrays into native memory and will use them as
        backing storage. It will increment the reference count and it will decrement then count when it is GCed.
        The buffers may reside in any storage either CPU or GPU.
        For strings and objects, it will create a copy of the arrays in CPU memory as ORT does not support those
        on other devices and their memory can not be mapped.
        """
        return cls(
            C.SparseTensor.sparse_csr_from_numpy(
                dense_shape,
                values,
                inner_indices,
                outer_indices,
                ort_device._get_c_device(),
            )
        )

    def values(self) -> np.ndarray:
        """
        The method returns a numpy array that is backed by the native memory
        if the data type is numeric. Otherwise, the returned numpy array that contains
        copies of the strings.
        """
        return self._tensor.values()

    def as_coo_view(self):
        """
        The method will return coo representation of the sparse tensor which will enable
        querying COO indices. If the instance did not contain COO format, it would throw.
        You can query coo indices as:

        ::

            coo_indices = sparse_tensor.as_coo_view().indices()

        which will return a numpy array that is backed by the native memory.
        """
        return self._tensor.get_coo_data()

    def as_csrc_view(self):
        """
        The method will return CSR(C) representation of the sparse tensor which will enable
        querying CRS(C) indices. If the instance dit not contain CSR(C) format, it would throw.
        You can query indices as:

        ::

            inner_ndices = sparse_tensor.as_csrc_view().inner()
            outer_ndices = sparse_tensor.as_csrc_view().outer()

        returning numpy arrays backed by the native memory.
        """
        return self._tensor.get_csrc_data()

    def as_blocksparse_view(self):
        """
        The method will return coo representation of the sparse tensor which will enable
        querying BlockSparse indices. If the instance did not contain BlockSparse format, it would throw.
        You can query coo indices as:

        ::

            block_sparse_indices = sparse_tensor.as_blocksparse_view().indices()

        which will return a numpy array that is backed by the native memory
        """
        return self._tensor.get_blocksparse_data()

    def to_cuda(self, ort_device):
        """
        Returns a copy of this instance on the specified cuda device

        :param ort_device: with name 'cuda' and valid gpu device id

        The method will throw if:

        - this instance contains strings
        - this instance is already on GPU. Cross GPU copy is not supported
        - CUDA is not present in this build
        - if the specified device is not valid
        """
        return SparseTensor(self._tensor.to_cuda(ort_device._get_c_device()))

    def format(self):
        """
        Returns a OrtSparseFormat enumeration
        """
        return self._tensor.format

    def dense_shape(self) -> npt.NDArray[np.int64]:
        """
        Returns a numpy array(int64) containing a dense shape of a sparse tensor
        """
        return self._tensor.dense_shape()

    def data_type(self) -> str:
        """
        Returns a string data type of the data in the OrtValue
        """
        return self._tensor.data_type()

    def device_name(self) -> str:
        """
        Returns the name of the device where the SparseTensor data buffers reside e.g. cpu, cuda
        """
        return self._tensor.device_name().lower()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\compatibility.py ===
"""Compatibility interface between dense and sparse polys. """

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.polys.domains.domain import Domain
    from sympy.polys.orderings import MonomialOrder
    from sympy.polys.rings import PolyElement

from sympy.polys.densearith import dup_add_term
from sympy.polys.densearith import dmp_add_term
from sympy.polys.densearith import dup_sub_term
from sympy.polys.densearith import dmp_sub_term
from sympy.polys.densearith import dup_mul_term
from sympy.polys.densearith import dmp_mul_term
from sympy.polys.densearith import dup_add_ground
from sympy.polys.densearith import dmp_add_ground
from sympy.polys.densearith import dup_sub_ground
from sympy.polys.densearith import dmp_sub_ground
from sympy.polys.densearith import dup_mul_ground
from sympy.polys.densearith import dmp_mul_ground
from sympy.polys.densearith import dup_quo_ground
from sympy.polys.densearith import dmp_quo_ground
from sympy.polys.densearith import dup_exquo_ground
from sympy.polys.densearith import dmp_exquo_ground
from sympy.polys.densearith import dup_lshift
from sympy.polys.densearith import dup_rshift
from sympy.polys.densearith import dup_abs
from sympy.polys.densearith import dmp_abs
from sympy.polys.densearith import dup_neg
from sympy.polys.densearith import dmp_neg
from sympy.polys.densearith import dup_add
from sympy.polys.densearith import dmp_add
from sympy.polys.densearith import dup_sub
from sympy.polys.densearith import dmp_sub
from sympy.polys.densearith import dup_add_mul
from sympy.polys.densearith import dmp_add_mul
from sympy.polys.densearith import dup_sub_mul
from sympy.polys.densearith import dmp_sub_mul
from sympy.polys.densearith import dup_mul
from sympy.polys.densearith import dmp_mul
from sympy.polys.densearith import dup_sqr
from sympy.polys.densearith import dmp_sqr
from sympy.polys.densearith import dup_pow
from sympy.polys.densearith import dmp_pow
from sympy.polys.densearith import dup_pdiv
from sympy.polys.densearith import dup_prem
from sympy.polys.densearith import dup_pquo
from sympy.polys.densearith import dup_pexquo
from sympy.polys.densearith import dmp_pdiv
from sympy.polys.densearith import dmp_prem
from sympy.polys.densearith import dmp_pquo
from sympy.polys.densearith import dmp_pexquo
from sympy.polys.densearith import dup_rr_div
from sympy.polys.densearith import dmp_rr_div
from sympy.polys.densearith import dup_ff_div
from sympy.polys.densearith import dmp_ff_div
from sympy.polys.densearith import dup_div
from sympy.polys.densearith import dup_rem
from sympy.polys.densearith import dup_quo
from sympy.polys.densearith import dup_exquo
from sympy.polys.densearith import dmp_div
from sympy.polys.densearith import dmp_rem
from sympy.polys.densearith import dmp_quo
from sympy.polys.densearith import dmp_exquo
from sympy.polys.densearith import dup_max_norm
from sympy.polys.densearith import dmp_max_norm
from sympy.polys.densearith import dup_l1_norm
from sympy.polys.densearith import dmp_l1_norm
from sympy.polys.densearith import dup_l2_norm_squared
from sympy.polys.densearith import dmp_l2_norm_squared
from sympy.polys.densearith import dup_expand
from sympy.polys.densearith import dmp_expand
from sympy.polys.densebasic import dup_LC
from sympy.polys.densebasic import dmp_LC
from sympy.polys.densebasic import dup_TC
from sympy.polys.densebasic import dmp_TC
from sympy.polys.densebasic import dmp_ground_LC
from sympy.polys.densebasic import dmp_ground_TC
from sympy.polys.densebasic import dup_degree
from sympy.polys.densebasic import dmp_degree
from sympy.polys.densebasic import dmp_degree_in
from sympy.polys.densebasic import dmp_to_dict
from sympy.polys.densetools import dup_integrate
from sympy.polys.densetools import dmp_integrate
from sympy.polys.densetools import dmp_integrate_in
from sympy.polys.densetools import dup_diff
from sympy.polys.densetools import dmp_diff
from sympy.polys.densetools import dmp_diff_in
from sympy.polys.densetools import dup_eval
from sympy.polys.densetools import dmp_eval
from sympy.polys.densetools import dmp_eval_in
from sympy.polys.densetools import dmp_eval_tail
from sympy.polys.densetools import dmp_diff_eval_in
from sympy.polys.densetools import dup_trunc
from sympy.polys.densetools import dmp_trunc
from sympy.polys.densetools import dmp_ground_trunc
from sympy.polys.densetools import dup_monic
from sympy.polys.densetools import dmp_ground_monic
from sympy.polys.densetools import dup_content
from sympy.polys.densetools import dmp_ground_content
from sympy.polys.densetools import dup_primitive
from sympy.polys.densetools import dmp_ground_primitive
from sympy.polys.densetools import dup_extract
from sympy.polys.densetools import dmp_ground_extract
from sympy.polys.densetools import dup_real_imag
from sympy.polys.densetools import dup_mirror
from sympy.polys.densetools import dup_scale
from sympy.polys.densetools import dup_shift
from sympy.polys.densetools import dmp_shift
from sympy.polys.densetools import dup_transform
from sympy.polys.densetools import dup_compose
from sympy.polys.densetools import dmp_compose
from sympy.polys.densetools import dup_decompose
from sympy.polys.densetools import dmp_lift
from sympy.polys.densetools import dup_sign_variations
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.densetools import dmp_clear_denoms
from sympy.polys.densetools import dup_revert
from sympy.polys.euclidtools import dup_half_gcdex
from sympy.polys.euclidtools import dmp_half_gcdex
from sympy.polys.euclidtools import dup_gcdex
from sympy.polys.euclidtools import dmp_gcdex
from sympy.polys.euclidtools import dup_invert
from sympy.polys.euclidtools import dmp_invert
from sympy.polys.euclidtools import dup_euclidean_prs
from sympy.polys.euclidtools import dmp_euclidean_prs
from sympy.polys.euclidtools import dup_primitive_prs
from sympy.polys.euclidtools import dmp_primitive_prs
from sympy.polys.euclidtools import dup_inner_subresultants
from sympy.polys.euclidtools import dup_subresultants
from sympy.polys.euclidtools import dup_prs_resultant
from sympy.polys.euclidtools import dup_resultant
from sympy.polys.euclidtools import dmp_inner_subresultants
from sympy.polys.euclidtools import dmp_subresultants
from sympy.polys.euclidtools import dmp_prs_resultant
from sympy.polys.euclidtools import dmp_zz_modular_resultant
from sympy.polys.euclidtools import dmp_zz_collins_resultant
from sympy.polys.euclidtools import dmp_qq_collins_resultant
from sympy.polys.euclidtools import dmp_resultant
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.euclidtools import dmp_discriminant
from sympy.polys.euclidtools import dup_rr_prs_gcd
from sympy.polys.euclidtools import dup_ff_prs_gcd
from sympy.polys.euclidtools import dmp_rr_prs_gcd
from sympy.polys.euclidtools import dmp_ff_prs_gcd
from sympy.polys.euclidtools import dup_zz_heu_gcd
from sympy.polys.euclidtools import dmp_zz_heu_gcd
from sympy.polys.euclidtools import dup_qq_heu_gcd
from sympy.polys.euclidtools import dmp_qq_heu_gcd
from sympy.polys.euclidtools import dup_inner_gcd
from sympy.polys.euclidtools import dmp_inner_gcd
from sympy.polys.euclidtools import dup_gcd
from sympy.polys.euclidtools import dmp_gcd
from sympy.polys.euclidtools import dup_rr_lcm
from sympy.polys.euclidtools import dup_ff_lcm
from sympy.polys.euclidtools import dup_lcm
from sympy.polys.euclidtools import dmp_rr_lcm
from sympy.polys.euclidtools import dmp_ff_lcm
from sympy.polys.euclidtools import dmp_lcm
from sympy.polys.euclidtools import dmp_content
from sympy.polys.euclidtools import dmp_primitive
from sympy.polys.euclidtools import dup_cancel
from sympy.polys.euclidtools import dmp_cancel
from sympy.polys.factortools import dup_trial_division
from sympy.polys.factortools import dmp_trial_division
from sympy.polys.factortools import dup_zz_mignotte_bound
from sympy.polys.factortools import dmp_zz_mignotte_bound
from sympy.polys.factortools import dup_zz_hensel_step
from sympy.polys.factortools import dup_zz_hensel_lift
from sympy.polys.factortools import dup_zz_zassenhaus
from sympy.polys.factortools import dup_zz_irreducible_p
from sympy.polys.factortools import dup_cyclotomic_p
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.factortools import dup_zz_cyclotomic_factor
from sympy.polys.factortools import dup_zz_factor_sqf
from sympy.polys.factortools import dup_zz_factor
from sympy.polys.factortools import dmp_zz_wang_non_divisors
from sympy.polys.factortools import dmp_zz_wang_lead_coeffs
from sympy.polys.factortools import dup_zz_diophantine
from sympy.polys.factortools import dmp_zz_diophantine
from sympy.polys.factortools import dmp_zz_wang_hensel_lifting
from sympy.polys.factortools import dmp_zz_wang
from sympy.polys.factortools import dmp_zz_factor
from sympy.polys.factortools import dup_qq_i_factor
from sympy.polys.factortools import dup_zz_i_factor
from sympy.polys.factortools import dmp_qq_i_factor
from sympy.polys.factortools import dmp_zz_i_factor
from sympy.polys.factortools import dup_ext_factor
from sympy.polys.factortools import dmp_ext_factor
from sympy.polys.factortools import dup_gf_factor
from sympy.polys.factortools import dmp_gf_factor
from sympy.polys.factortools import dup_factor_list
from sympy.polys.factortools import dup_factor_list_include
from sympy.polys.factortools import dmp_factor_list
from sympy.polys.factortools import dmp_factor_list_include
from sympy.polys.factortools import dup_irreducible_p
from sympy.polys.factortools import dmp_irreducible_p
from sympy.polys.rootisolation import dup_sturm
from sympy.polys.rootisolation import dup_root_upper_bound
from sympy.polys.rootisolation import dup_root_lower_bound
from sympy.polys.rootisolation import dup_step_refine_real_root
from sympy.polys.rootisolation import dup_inner_refine_real_root
from sympy.polys.rootisolation import dup_outer_refine_real_root
from sympy.polys.rootisolation import dup_refine_real_root
from sympy.polys.rootisolation import dup_inner_isolate_real_roots
from sympy.polys.rootisolation import dup_inner_isolate_positive_roots
from sympy.polys.rootisolation import dup_inner_isolate_negative_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_sqf
from sympy.polys.rootisolation import dup_isolate_real_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.polys.rootisolation import dup_count_real_roots
from sympy.polys.rootisolation import dup_count_complex_roots
from sympy.polys.rootisolation import dup_isolate_complex_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots

from sympy.polys.sqfreetools import (
    dup_sqf_p, dmp_sqf_p, dmp_norm, dup_sqf_norm, dmp_sqf_norm,
    dup_gf_sqf_part, dmp_gf_sqf_part, dup_sqf_part, dmp_sqf_part,
    dup_gf_sqf_list, dmp_gf_sqf_list, dup_sqf_list, dup_sqf_list_include,
    dmp_sqf_list, dmp_sqf_list_include, dup_gff_list, dmp_gff_list)

from sympy.polys.galoistools import (
    gf_degree, gf_LC, gf_TC, gf_strip, gf_from_dict,
    gf_to_dict, gf_from_int_poly, gf_to_int_poly, gf_neg, gf_add_ground, gf_sub_ground,
    gf_mul_ground, gf_quo_ground, gf_add, gf_sub, gf_mul, gf_sqr, gf_add_mul, gf_sub_mul,
    gf_expand, gf_div, gf_rem, gf_quo, gf_exquo, gf_lshift, gf_rshift, gf_pow, gf_pow_mod,
    gf_gcd, gf_lcm, gf_cofactors, gf_gcdex, gf_monic, gf_diff, gf_eval, gf_multi_eval,
    gf_compose, gf_compose_mod, gf_trace_map, gf_random, gf_irreducible, gf_irred_p_ben_or,
    gf_irred_p_rabin, gf_irreducible_p, gf_sqf_p, gf_sqf_part, gf_Qmatrix,
    gf_berlekamp, gf_ddf_zassenhaus, gf_edf_zassenhaus, gf_ddf_shoup, gf_edf_shoup,
    gf_zassenhaus, gf_shoup, gf_factor_sqf, gf_factor)

from sympy.utilities import public

@public
class IPolys:

    gens: tuple[PolyElement, ...]
    symbols: tuple[Expr, ...]
    ngens: int
    domain: Domain
    order: MonomialOrder

    def drop(self, gen):
        pass

    def clone(self, symbols=None, domain=None, order=None):
        pass

    def to_ground(self):
        pass

    def ground_new(self, element):
        pass

    def domain_new(self, element):
        pass

    def from_dict(self, d):
        pass

    def wrap(self, element):
        from sympy.polys.rings import PolyElement
        if isinstance(element, PolyElement):
            if element.ring == self:
                return element
            else:
                raise NotImplementedError("domain conversions")
        else:
            return self.ground_new(element)

    def to_dense(self, element):
        return self.wrap(element).to_dense()

    def from_dense(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))

    def dup_add_term(self, f, c, i):
        return self.from_dense(dup_add_term(self.to_dense(f), c, i, self.domain))
    def dmp_add_term(self, f, c, i):
        return self.from_dense(dmp_add_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))
    def dup_sub_term(self, f, c, i):
        return self.from_dense(dup_sub_term(self.to_dense(f), c, i, self.domain))
    def dmp_sub_term(self, f, c, i):
        return self.from_dense(dmp_sub_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))
    def dup_mul_term(self, f, c, i):
        return self.from_dense(dup_mul_term(self.to_dense(f), c, i, self.domain))
    def dmp_mul_term(self, f, c, i):
        return self.from_dense(dmp_mul_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))

    def dup_add_ground(self, f, c):
        return self.from_dense(dup_add_ground(self.to_dense(f), c, self.domain))
    def dmp_add_ground(self, f, c):
        return self.from_dense(dmp_add_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_sub_ground(self, f, c):
        return self.from_dense(dup_sub_ground(self.to_dense(f), c, self.domain))
    def dmp_sub_ground(self, f, c):
        return self.from_dense(dmp_sub_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_mul_ground(self, f, c):
        return self.from_dense(dup_mul_ground(self.to_dense(f), c, self.domain))
    def dmp_mul_ground(self, f, c):
        return self.from_dense(dmp_mul_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_quo_ground(self, f, c):
        return self.from_dense(dup_quo_ground(self.to_dense(f), c, self.domain))
    def dmp_quo_ground(self, f, c):
        return self.from_dense(dmp_quo_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_exquo_ground(self, f, c):
        return self.from_dense(dup_exquo_ground(self.to_dense(f), c, self.domain))
    def dmp_exquo_ground(self, f, c):
        return self.from_dense(dmp_exquo_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    def dup_lshift(self, f, n):
        return self.from_dense(dup_lshift(self.to_dense(f), n, self.domain))
    def dup_rshift(self, f, n):
        return self.from_dense(dup_rshift(self.to_dense(f), n, self.domain))

    def dup_abs(self, f):
        return self.from_dense(dup_abs(self.to_dense(f), self.domain))
    def dmp_abs(self, f):
        return self.from_dense(dmp_abs(self.to_dense(f), self.ngens-1, self.domain))

    def dup_neg(self, f):
        return self.from_dense(dup_neg(self.to_dense(f), self.domain))
    def dmp_neg(self, f):
        return self.from_dense(dmp_neg(self.to_dense(f), self.ngens-1, self.domain))

    def dup_add(self, f, g):
        return self.from_dense(dup_add(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_add(self, f, g):
        return self.from_dense(dmp_add(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_sub(self, f, g):
        return self.from_dense(dup_sub(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_sub(self, f, g):
        return self.from_dense(dmp_sub(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_add_mul(self, f, g, h):
        return self.from_dense(dup_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))
    def dmp_add_mul(self, f, g, h):
        return self.from_dense(dmp_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))
    def dup_sub_mul(self, f, g, h):
        return self.from_dense(dup_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))
    def dmp_sub_mul(self, f, g, h):
        return self.from_dense(dmp_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))

    def dup_mul(self, f, g):
        return self.from_dense(dup_mul(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_mul(self, f, g):
        return self.from_dense(dmp_mul(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_sqr(self, f):
        return self.from_dense(dup_sqr(self.to_dense(f), self.domain))
    def dmp_sqr(self, f):
        return self.from_dense(dmp_sqr(self.to_dense(f), self.ngens-1, self.domain))
    def dup_pow(self, f, n):
        return self.from_dense(dup_pow(self.to_dense(f), n, self.domain))
    def dmp_pow(self, f, n):
        return self.from_dense(dmp_pow(self.to_dense(f), n, self.ngens-1, self.domain))

    def dup_pdiv(self, f, g):
        q, r = dup_pdiv(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_prem(self, f, g):
        return self.from_dense(dup_prem(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_pquo(self, f, g):
        return self.from_dense(dup_pquo(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_pexquo(self, f, g):
        return self.from_dense(dup_pexquo(self.to_dense(f), self.to_dense(g), self.domain))

    def dmp_pdiv(self, f, g):
        q, r = dmp_pdiv(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_prem(self, f, g):
        return self.from_dense(dmp_prem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_pquo(self, f, g):
        return self.from_dense(dmp_pquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_pexquo(self, f, g):
        return self.from_dense(dmp_pexquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_rr_div(self, f, g):
        q, r = dup_rr_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_rr_div(self, f, g):
        q, r = dmp_rr_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_ff_div(self, f, g):
        q, r = dup_ff_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_ff_div(self, f, g):
        q, r = dmp_ff_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))

    def dup_div(self, f, g):
        q, r = dup_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_rem(self, f, g):
        return self.from_dense(dup_rem(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_quo(self, f, g):
        return self.from_dense(dup_quo(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_exquo(self, f, g):
        return self.from_dense(dup_exquo(self.to_dense(f), self.to_dense(g), self.domain))

    def dmp_div(self, f, g):
        q, r = dmp_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_rem(self, f, g):
        return self.from_dense(dmp_rem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_quo(self, f, g):
        return self.from_dense(dmp_quo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_exquo(self, f, g):
        return self.from_dense(dmp_exquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_max_norm(self, f):
        return dup_max_norm(self.to_dense(f), self.domain)
    def dmp_max_norm(self, f):
        return dmp_max_norm(self.to_dense(f), self.ngens-1, self.domain)

    def dup_l1_norm(self, f):
        return dup_l1_norm(self.to_dense(f), self.domain)
    def dmp_l1_norm(self, f):
        return dmp_l1_norm(self.to_dense(f), self.ngens-1, self.domain)

    def dup_l2_norm_squared(self, f):
        return dup_l2_norm_squared(self.to_dense(f), self.domain)
    def dmp_l2_norm_squared(self, f):
        return dmp_l2_norm_squared(self.to_dense(f), self.ngens-1, self.domain)

    def dup_expand(self, polys):
        return self.from_dense(dup_expand(list(map(self.to_dense, polys)), self.domain))
    def dmp_expand(self, polys):
        return self.from_dense(dmp_expand(list(map(self.to_dense, polys)), self.ngens-1, self.domain))

    def dup_LC(self, f):
        return dup_LC(self.to_dense(f), self.domain)
    def dmp_LC(self, f):
        LC = dmp_LC(self.to_dense(f), self.domain)
        if isinstance(LC, list):
            return self[1:].from_dense(LC)
        else:
            return LC
    def dup_TC(self, f):
        return dup_TC(self.to_dense(f), self.domain)
    def dmp_TC(self, f):
        TC = dmp_TC(self.to_dense(f), self.domain)
        if isinstance(TC, list):
            return self[1:].from_dense(TC)
        else:
            return TC

    def dmp_ground_LC(self, f):
        return dmp_ground_LC(self.to_dense(f), self.ngens-1, self.domain)
    def dmp_ground_TC(self, f):
        return dmp_ground_TC(self.to_dense(f), self.ngens-1, self.domain)

    def dup_degree(self, f):
        return dup_degree(self.to_dense(f))
    def dmp_degree(self, f):
        return dmp_degree(self.to_dense(f), self.ngens-1)
    def dmp_degree_in(self, f, j):
        return dmp_degree_in(self.to_dense(f), j, self.ngens-1)
    def dup_integrate(self, f, m):
        return self.from_dense(dup_integrate(self.to_dense(f), m, self.domain))
    def dmp_integrate(self, f, m):
        return self.from_dense(dmp_integrate(self.to_dense(f), m, self.ngens-1, self.domain))

    def dup_diff(self, f, m):
        return self.from_dense(dup_diff(self.to_dense(f), m, self.domain))
    def dmp_diff(self, f, m):
        return self.from_dense(dmp_diff(self.to_dense(f), m, self.ngens-1, self.domain))

    def dmp_diff_in(self, f, m, j):
        return self.from_dense(dmp_diff_in(self.to_dense(f), m, j, self.ngens-1, self.domain))
    def dmp_integrate_in(self, f, m, j):
        return self.from_dense(dmp_integrate_in(self.to_dense(f), m, j, self.ngens-1, self.domain))

    def dup_eval(self, f, a):
        return dup_eval(self.to_dense(f), a, self.domain)
    def dmp_eval(self, f, a):
        result = dmp_eval(self.to_dense(f), a, self.ngens-1, self.domain)
        return self[1:].from_dense(result)

    def dmp_eval_in(self, f, a, j):
        result = dmp_eval_in(self.to_dense(f), a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)
    def dmp_diff_eval_in(self, f, m, a, j):
        result = dmp_diff_eval_in(self.to_dense(f), m, a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)

    def dmp_eval_tail(self, f, A):
        result = dmp_eval_tail(self.to_dense(f), A, self.ngens-1, self.domain)
        if isinstance(result, list):
            return self[:-len(A)].from_dense(result)
        else:
            return result

    def dup_trunc(self, f, p):
        return self.from_dense(dup_trunc(self.to_dense(f), p, self.domain))
    def dmp_trunc(self, f, g):
        return self.from_dense(dmp_trunc(self.to_dense(f), self[1:].to_dense(g), self.ngens-1, self.domain))
    def dmp_ground_trunc(self, f, p):
        return self.from_dense(dmp_ground_trunc(self.to_dense(f), p, self.ngens-1, self.domain))

    def dup_monic(self, f):
        return self.from_dense(dup_monic(self.to_dense(f), self.domain))
    def dmp_ground_monic(self, f):
        return self.from_dense(dmp_ground_monic(self.to_dense(f), self.ngens-1, self.domain))

    def dup_extract(self, f, g):
        c, F, G = dup_extract(self.to_dense(f), self.to_dense(g), self.domain)
        return (c, self.from_dense(F), self.from_dense(G))
    def dmp_ground_extract(self, f, g):
        c, F, G = dmp_ground_extract(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (c, self.from_dense(F), self.from_dense(G))

    def dup_real_imag(self, f):
        p, q = dup_real_imag(self.wrap(f).drop(1).to_dense(), self.domain)
        return (self.from_dense(p), self.from_dense(q))

    def dup_mirror(self, f):
        return self.from_dense(dup_mirror(self.to_dense(f), self.domain))
    def dup_scale(self, f, a):
        return self.from_dense(dup_scale(self.to_dense(f), a, self.domain))
    def dup_shift(self, f, a):
        return self.from_dense(dup_shift(self.to_dense(f), a, self.domain))
    def dmp_shift(self, f, a):
        return self.from_dense(dmp_shift(self.to_dense(f), a, self.ngens-1, self.domain))
    def dup_transform(self, f, p, q):
        return self.from_dense(dup_transform(self.to_dense(f), self.to_dense(p), self.to_dense(q), self.domain))

    def dup_compose(self, f, g):
        return self.from_dense(dup_compose(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_compose(self, f, g):
        return self.from_dense(dmp_compose(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_decompose(self, f):
        components = dup_decompose(self.to_dense(f), self.domain)
        return list(map(self.from_dense, components))

    def dmp_lift(self, f):
        result = dmp_lift(self.to_dense(f), self.ngens-1, self.domain)
        return self.to_ground().from_dense(result)

    def dup_sign_variations(self, f):
        return dup_sign_variations(self.to_dense(f), self.domain)

    def dup_clear_denoms(self, f, convert=False):
        c, F = dup_clear_denoms(self.to_dense(f), self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))
    def dmp_clear_denoms(self, f, convert=False):
        c, F = dmp_clear_denoms(self.to_dense(f), self.ngens-1, self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))

    def dup_revert(self, f, n):
        return self.from_dense(dup_revert(self.to_dense(f), n, self.domain))

    def dup_half_gcdex(self, f, g):
        s, h = dup_half_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(h))
    def dmp_half_gcdex(self, f, g):
        s, h = dmp_half_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(h))
    def dup_gcdex(self, f, g):
        s, t, h = dup_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))
    def dmp_gcdex(self, f, g):
        s, t, h = dmp_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))

    def dup_invert(self, f, g):
        return self.from_dense(dup_invert(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_invert(self, f, g):
        return self.from_dense(dmp_invert(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_euclidean_prs(self, f, g):
        prs = dup_euclidean_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_euclidean_prs(self, f, g):
        prs = dmp_euclidean_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))
    def dup_primitive_prs(self, f, g):
        prs = dup_primitive_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_primitive_prs(self, f, g):
        prs = dmp_primitive_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    def dup_inner_subresultants(self, f, g):
        prs, sres = dup_inner_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return (list(map(self.from_dense, prs)), sres)
    def dmp_inner_subresultants(self, f, g):
        prs, sres  = dmp_inner_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (list(map(self.from_dense, prs)), sres)

    def dup_subresultants(self, f, g):
        prs = dup_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_subresultants(self, f, g):
        prs = dmp_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    def dup_prs_resultant(self, f, g):
        res, prs = dup_prs_resultant(self.to_dense(f), self.to_dense(g), self.domain)
        return (res, list(map(self.from_dense, prs)))
    def dmp_prs_resultant(self, f, g):
        res, prs = dmp_prs_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self[1:].from_dense(res), list(map(self.from_dense, prs)))

    def dmp_zz_modular_resultant(self, f, g, p):
        res = dmp_zz_modular_resultant(self.to_dense(f), self.to_dense(g), self.domain_new(p), self.ngens-1, self.domain)
        return self[1:].from_dense(res)
    def dmp_zz_collins_resultant(self, f, g):
        res = dmp_zz_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self[1:].from_dense(res)
    def dmp_qq_collins_resultant(self, f, g):
        res = dmp_qq_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self[1:].from_dense(res)

    def dup_resultant(self, f, g): #, includePRS=False):
        return dup_resultant(self.to_dense(f), self.to_dense(g), self.domain) #, includePRS=includePRS)
    def dmp_resultant(self, f, g): #, includePRS=False):
        res = dmp_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain) #, includePRS=includePRS)
        if isinstance(res, list):
            return self[1:].from_dense(res)
        else:
            return res

    def dup_discriminant(self, f):
        return dup_discriminant(self.to_dense(f), self.domain)
    def dmp_discriminant(self, f):
        disc = dmp_discriminant(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(disc, list):
            return self[1:].from_dense(disc)
        else:
            return disc

    def dup_rr_prs_gcd(self, f, g):
        H, F, G = dup_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_ff_prs_gcd(self, f, g):
        H, F, G = dup_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_rr_prs_gcd(self, f, g):
        H, F, G = dmp_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_ff_prs_gcd(self, f, g):
        H, F, G = dmp_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_zz_heu_gcd(self, f, g):
        H, F, G = dup_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_zz_heu_gcd(self, f, g):
        H, F, G = dmp_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_qq_heu_gcd(self, f, g):
        H, F, G = dup_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_qq_heu_gcd(self, f, g):
        H, F, G = dmp_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_inner_gcd(self, f, g):
        H, F, G = dup_inner_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_inner_gcd(self, f, g):
        H, F, G = dmp_inner_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_gcd(self, f, g):
        H = dup_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dmp_gcd(self, f, g):
        H = dmp_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dup_rr_lcm(self, f, g):
        H = dup_rr_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dup_ff_lcm(self, f, g):
        H = dup_ff_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dup_lcm(self, f, g):
        H = dup_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dmp_rr_lcm(self, f, g):
        H = dmp_rr_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dmp_ff_lcm(self, f, g):
        H = dmp_ff_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dmp_lcm(self, f, g):
        H = dmp_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)

    def dup_content(self, f):
        cont = dup_content(self.to_dense(f), self.domain)
        return cont
    def dup_primitive(self, f):
        cont, prim = dup_primitive(self.to_dense(f), self.domain)
        return cont, self.from_dense(prim)

    def dmp_content(self, f):
        cont = dmp_content(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return self[1:].from_dense(cont)
        else:
            return cont
    def dmp_primitive(self, f):
        cont, prim = dmp_primitive(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return (self[1:].from_dense(cont), self.from_dense(prim))
        else:
            return (cont, self.from_dense(prim))

    def dmp_ground_content(self, f):
        cont = dmp_ground_content(self.to_dense(f), self.ngens-1, self.domain)
        return cont
    def dmp_ground_primitive(self, f):
        cont, prim = dmp_ground_primitive(self.to_dense(f), self.ngens-1, self.domain)
        return (cont, self.from_dense(prim))

    def dup_cancel(self, f, g, include=True):
        result = dup_cancel(self.to_dense(f), self.to_dense(g), self.domain, include=include)
        if not include:
            cf, cg, F, G = result
            return (cf, cg, self.from_dense(F), self.from_dense(G))
        else:
            F, G = result
            return (self.from_dense(F), self.from_dense(G))
    def dmp_cancel(self, f, g, include=True):
        result = dmp_cancel(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain, include=include)
        if not include:
            cf, cg, F, G = result
            return (cf, cg, self.from_dense(F), self.from_dense(G))
        else:
            F, G = result
            return (self.from_dense(F), self.from_dense(G))

    def dup_trial_division(self, f, factors):
        factors = dup_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_trial_division(self, f, factors):
        factors = dmp_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_zz_mignotte_bound(self, f):
        return dup_zz_mignotte_bound(self.to_dense(f), self.domain)
    def dmp_zz_mignotte_bound(self, f):
        return dmp_zz_mignotte_bound(self.to_dense(f), self.ngens-1, self.domain)

    def dup_zz_hensel_step(self, m, f, g, h, s, t):
        D = self.to_dense
        G, H, S, T = dup_zz_hensel_step(m, D(f), D(g), D(h), D(s), D(t), self.domain)
        return (self.from_dense(G), self.from_dense(H), self.from_dense(S), self.from_dense(T))
    def dup_zz_hensel_lift(self, p, f, f_list, l):
        D = self.to_dense
        polys = dup_zz_hensel_lift(p, D(f), list(map(D, f_list)), l, self.domain)
        return list(map(self.from_dense, polys))

    def dup_zz_zassenhaus(self, f):
        factors = dup_zz_zassenhaus(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_zz_irreducible_p(self, f):
        return dup_zz_irreducible_p(self.to_dense(f), self.domain)
    def dup_cyclotomic_p(self, f, irreducible=False):
        return dup_cyclotomic_p(self.to_dense(f), self.domain, irreducible=irreducible)
    def dup_zz_cyclotomic_poly(self, n):
        F = dup_zz_cyclotomic_poly(n, self.domain)
        return self.from_dense(F)
    def dup_zz_cyclotomic_factor(self, f):
        result = dup_zz_cyclotomic_factor(self.to_dense(f), self.domain)
        if result is None:
            return result
        else:
            return list(map(self.from_dense, result))

    # E: List[ZZ], cs: ZZ, ct: ZZ
    def dmp_zz_wang_non_divisors(self, E, cs, ct):
        return dmp_zz_wang_non_divisors(E, cs, ct, self.domain)

    # f: Poly, T: List[(Poly, int)], ct: ZZ, A: List[ZZ]
    #def dmp_zz_wang_test_points(f, T, ct, A):
    #   dmp_zz_wang_test_points(self.to_dense(f), T, ct, A, self.ngens-1, self.domain)

    # f: Poly, T: List[(Poly, int)], cs: ZZ, E: List[ZZ], H: List[Poly], A: List[ZZ]
    def dmp_zz_wang_lead_coeffs(self, f, T, cs, E, H, A):
        mv = self[1:]
        T = [ (mv.to_dense(t), k) for t, k in T ]
        uv = self[:1]
        H = list(map(uv.to_dense, H))
        f, HH, CC = dmp_zz_wang_lead_coeffs(self.to_dense(f), T, cs, E, H, A, self.ngens-1, self.domain)
        return self.from_dense(f), list(map(uv.from_dense, HH)), list(map(mv.from_dense, CC))

    # f: List[Poly], m: int, p: ZZ
    def dup_zz_diophantine(self, F, m, p):
        result = dup_zz_diophantine(list(map(self.to_dense, F)), m, p, self.domain)
        return list(map(self.from_dense, result))

    # f: List[Poly], c: List[Poly], A: List[ZZ], d: int, p: ZZ
    def dmp_zz_diophantine(self, F, c, A, d, p):
        result = dmp_zz_diophantine(list(map(self.to_dense, F)), self.to_dense(c), A, d, p, self.ngens-1, self.domain)
        return list(map(self.from_dense, result))

    # f: Poly, H: List[Poly], LC: List[Poly], A: List[ZZ], p: ZZ
    def dmp_zz_wang_hensel_lifting(self, f, H, LC, A, p):
        uv = self[:1]
        mv = self[1:]
        H = list(map(uv.to_dense, H))
        LC = list(map(mv.to_dense, LC))
        result = dmp_zz_wang_hensel_lifting(self.to_dense(f), H, LC, A, p, self.ngens-1, self.domain)
        return list(map(self.from_dense, result))

    def dmp_zz_wang(self, f, mod=None, seed=None):
        factors = dmp_zz_wang(self.to_dense(f), self.ngens-1, self.domain, mod=mod, seed=seed)
        return [ self.from_dense(g) for g in factors ]

    def dup_zz_factor_sqf(self, f):
        coeff, factors = dup_zz_factor_sqf(self.to_dense(f), self.domain)
        return (coeff, [ self.from_dense(g) for g in factors ])

    def dup_zz_factor(self, f):
        coeff, factors = dup_zz_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_zz_factor(self, f):
        coeff, factors = dmp_zz_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_qq_i_factor(self, f):
        coeff, factors = dup_qq_i_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_qq_i_factor(self, f):
        coeff, factors = dmp_qq_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_zz_i_factor(self, f):
        coeff, factors = dup_zz_i_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_zz_i_factor(self, f):
        coeff, factors = dmp_zz_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_ext_factor(self, f):
        coeff, factors = dup_ext_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_ext_factor(self, f):
        coeff, factors = dmp_ext_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_gf_factor(self, f):
        coeff, factors = dup_gf_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_gf_factor(self, f):
        coeff, factors = dmp_gf_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_factor_list(self, f):
        coeff, factors = dup_factor_list(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dup_factor_list_include(self, f):
        factors = dup_factor_list_include(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dmp_factor_list(self, f):
        coeff, factors = dmp_factor_list(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_factor_list_include(self, f):
        factors = dmp_factor_list_include(self.to_dense(f), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_irreducible_p(self, f):
        return dup_irreducible_p(self.to_dense(f), self.domain)
    def dmp_irreducible_p(self, f):
        return dmp_irreducible_p(self.to_dense(f), self.ngens-1, self.domain)

    def dup_sturm(self, f):
        seq = dup_sturm(self.to_dense(f), self.domain)
        return list(map(self.from_dense, seq))

    def dup_sqf_p(self, f):
        return dup_sqf_p(self.to_dense(f), self.domain)
    def dmp_sqf_p(self, f):
        return dmp_sqf_p(self.to_dense(f), self.ngens-1, self.domain)

    def dmp_norm(self, f):
        n = dmp_norm(self.to_dense(f), self.ngens-1, self.domain)
        return self.to_ground().from_dense(n)

    def dup_sqf_norm(self, f):
        s, F, R = dup_sqf_norm(self.to_dense(f), self.domain)
        return (s, self.from_dense(F), self.to_ground().from_dense(R))
    def dmp_sqf_norm(self, f):
        s, F, R = dmp_sqf_norm(self.to_dense(f), self.ngens-1, self.domain)
        return (s, self.from_dense(F), self.to_ground().from_dense(R))

    def dup_gf_sqf_part(self, f):
        return self.from_dense(dup_gf_sqf_part(self.to_dense(f), self.domain))
    def dmp_gf_sqf_part(self, f):
        return self.from_dense(dmp_gf_sqf_part(self.to_dense(f), self.domain))
    def dup_sqf_part(self, f):
        return self.from_dense(dup_sqf_part(self.to_dense(f), self.domain))
    def dmp_sqf_part(self, f):
        return self.from_dense(dmp_sqf_part(self.to_dense(f), self.ngens-1, self.domain))

    def dup_gf_sqf_list(self, f, all=False):
        coeff, factors = dup_gf_sqf_list(self.to_dense(f), self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_gf_sqf_list(self, f, all=False):
        coeff, factors = dmp_gf_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_sqf_list(self, f, all=False):
        coeff, factors = dup_sqf_list(self.to_dense(f), self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dup_sqf_list_include(self, f, all=False):
        factors = dup_sqf_list_include(self.to_dense(f), self.domain, all=all)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_sqf_list(self, f, all=False):
        coeff, factors = dmp_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_sqf_list_include(self, f, all=False):
        factors = dmp_sqf_list_include(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_gff_list(self, f):
        factors = dup_gff_list(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_gff_list(self, f):
        factors = dmp_gff_list(self.to_dense(f), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_root_upper_bound(self, f):
        return dup_root_upper_bound(self.to_dense(f), self.domain)
    def dup_root_lower_bound(self, f):
        return dup_root_lower_bound(self.to_dense(f), self.domain)

    def dup_step_refine_real_root(self, f, M, fast=False):
        return dup_step_refine_real_root(self.to_dense(f), M, self.domain, fast=fast)
    def dup_inner_refine_real_root(self, f, M, eps=None, steps=None, disjoint=None, fast=False, mobius=False):
        return dup_inner_refine_real_root(self.to_dense(f), M, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast, mobius=mobius)
    def dup_outer_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        return dup_outer_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
    def dup_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        return dup_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
    def dup_inner_isolate_real_roots(self, f, eps=None, fast=False):
        return dup_inner_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, fast=fast)
    def dup_inner_isolate_positive_roots(self, f, eps=None, inf=None, sup=None, fast=False, mobius=False):
        return dup_inner_isolate_positive_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, mobius=mobius)
    def dup_inner_isolate_negative_roots(self, f, inf=None, sup=None, eps=None, fast=False, mobius=False):
        return dup_inner_isolate_negative_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, eps=eps, fast=fast, mobius=mobius)
    def dup_isolate_real_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_real_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)
    def dup_isolate_real_roots(self, f, eps=None, inf=None, sup=None, basis=False, fast=False):
        return dup_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, basis=basis, fast=fast)
    def dup_isolate_real_roots_list(self, polys, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
        return dup_isolate_real_roots_list(list(map(self.to_dense, polys)), self.domain, eps=eps, inf=inf, sup=sup, strict=strict, basis=basis, fast=fast)
    def dup_count_real_roots(self, f, inf=None, sup=None):
        return dup_count_real_roots(self.to_dense(f), self.domain, inf=inf, sup=sup)
    def dup_count_complex_roots(self, f, inf=None, sup=None, exclude=None):
        return dup_count_complex_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, exclude=exclude)
    def dup_isolate_complex_roots_sqf(self, f, eps=None, inf=None, sup=None, blackbox=False):
        return dup_isolate_complex_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, blackbox=blackbox)
    def dup_isolate_all_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_all_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)
    def dup_isolate_all_roots(self, f, eps=None, inf=None, sup=None, fast=False):
        return dup_isolate_all_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast)

    def fateman_poly_F_1(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_1
        return tuple(map(self.from_dense, dmp_fateman_poly_F_1(self.ngens-1, self.domain)))
    def fateman_poly_F_2(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_2
        return tuple(map(self.from_dense, dmp_fateman_poly_F_2(self.ngens-1, self.domain)))
    def fateman_poly_F_3(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_3
        return tuple(map(self.from_dense, dmp_fateman_poly_F_3(self.ngens-1, self.domain)))

    def to_gf_dense(self, element):
        return gf_strip([ self.domain.dom.convert(c, self.domain) for c in self.wrap(element).to_dense() ])

    def from_gf_dense(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain.dom))

    def gf_degree(self, f):
        return gf_degree(self.to_gf_dense(f))

    def gf_LC(self, f):
        return gf_LC(self.to_gf_dense(f), self.domain.dom)
    def gf_TC(self, f):
        return gf_TC(self.to_gf_dense(f), self.domain.dom)

    def gf_strip(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f)))
    def gf_trunc(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod))
    def gf_normal(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_from_dict(self, f):
        return self.from_gf_dense(gf_from_dict(f, self.domain.mod, self.domain.dom))
    def gf_to_dict(self, f, symmetric=True):
        return gf_to_dict(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_from_int_poly(self, f):
        return self.from_gf_dense(gf_from_int_poly(f, self.domain.mod))
    def gf_to_int_poly(self, f, symmetric=True):
        return gf_to_int_poly(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_neg(self, f):
        return self.from_gf_dense(gf_neg(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_ground(self, f, a):
        return self.from_gf_dense(gf_add_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_sub_ground(self, f, a):
        return self.from_gf_dense(gf_sub_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_mul_ground(self, f, a):
        return self.from_gf_dense(gf_mul_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_quo_ground(self, f, a):
        return self.from_gf_dense(gf_quo_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))

    def gf_add(self, f, g):
        return self.from_gf_dense(gf_add(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_sub(self, f, g):
        return self.from_gf_dense(gf_sub(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_mul(self, f, g):
        return self.from_gf_dense(gf_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_sqr(self, f):
        return self.from_gf_dense(gf_sqr(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_mul(self, f, g, h):
        return self.from_gf_dense(gf_add_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))
    def gf_sub_mul(self, f, g, h):
        return self.from_gf_dense(gf_sub_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))

    def gf_expand(self, F):
        return self.from_gf_dense(gf_expand(list(map(self.to_gf_dense, F)), self.domain.mod, self.domain.dom))

    def gf_div(self, f, g):
        q, r = gf_div(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(q), self.from_gf_dense(r)
    def gf_rem(self, f, g):
        return self.from_gf_dense(gf_rem(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_quo(self, f, g):
        return self.from_gf_dense(gf_quo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_exquo(self, f, g):
        return self.from_gf_dense(gf_exquo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_lshift(self, f, n):
        return self.from_gf_dense(gf_lshift(self.to_gf_dense(f), n, self.domain.dom))
    def gf_rshift(self, f, n):
        return self.from_gf_dense(gf_rshift(self.to_gf_dense(f), n, self.domain.dom))

    def gf_pow(self, f, n):
        return self.from_gf_dense(gf_pow(self.to_gf_dense(f), n, self.domain.mod, self.domain.dom))
    def gf_pow_mod(self, f, n, g):
        return self.from_gf_dense(gf_pow_mod(self.to_gf_dense(f), n, self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_cofactors(self, f, g):
        h, cff, cfg = gf_cofactors(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(h), self.from_gf_dense(cff), self.from_gf_dense(cfg)
    def gf_gcd(self, f, g):
        return self.from_gf_dense(gf_gcd(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_lcm(self, f, g):
        return self.from_gf_dense(gf_lcm(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_gcdex(self, f, g):
        return self.from_gf_dense(gf_gcdex(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_monic(self, f):
        return self.from_gf_dense(gf_monic(self.to_gf_dense(f), self.domain.mod, self.domain.dom))
    def gf_diff(self, f):
        return self.from_gf_dense(gf_diff(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_eval(self, f, a):
        return gf_eval(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom)
    def gf_multi_eval(self, f, A):
        return gf_multi_eval(self.to_gf_dense(f), A, self.domain.mod, self.domain.dom)

    def gf_compose(self, f, g):
        return self.from_gf_dense(gf_compose(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_compose_mod(self, g, h, f):
        return self.from_gf_dense(gf_compose_mod(self.to_gf_dense(g), self.to_gf_dense(h), self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_trace_map(self, a, b, c, n, f):
        a = self.to_gf_dense(a)
        b = self.to_gf_dense(b)
        c = self.to_gf_dense(c)
        f = self.to_gf_dense(f)
        U, V = gf_trace_map(a, b, c, n, f, self.domain.mod, self.domain.dom)
        return self.from_gf_dense(U), self.from_gf_dense(V)

    def gf_random(self, n):
        return self.from_gf_dense(gf_random(n, self.domain.mod, self.domain.dom))
    def gf_irreducible(self, n):
        return self.from_gf_dense(gf_irreducible(n, self.domain.mod, self.domain.dom))

    def gf_irred_p_ben_or(self, f):
        return gf_irred_p_ben_or(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_irred_p_rabin(self, f):
        return gf_irred_p_rabin(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_irreducible_p(self, f):
        return gf_irreducible_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_sqf_p(self, f):
        return gf_sqf_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)

    def gf_sqf_part(self, f):
        return self.from_gf_dense(gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom))
    def gf_sqf_list(self, f, all=False):
        coeff, factors = gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]

    def gf_Qmatrix(self, f):
        return gf_Qmatrix(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_berlekamp(self, f):
        factors = gf_berlekamp(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_ddf_zassenhaus(self, f):
        factors = gf_ddf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]
    def gf_edf_zassenhaus(self, f, n):
        factors = gf_edf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_ddf_shoup(self, f):
        factors = gf_ddf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]
    def gf_edf_shoup(self, f, n):
        factors = gf_edf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_zassenhaus(self, f):
        factors = gf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]
    def gf_shoup(self, f):
        factors = gf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_factor_sqf(self, f, method=None):
        coeff, factors = gf_factor_sqf(self.to_gf_dense(f), self.domain.mod, self.domain.dom, method=method)
        return coeff, [ self.from_gf_dense(g) for g in factors ]
    def gf_factor(self, f):
        coeff, factors = gf_factor(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\libhyper.py ===
"""
This module implements computation of hypergeometric and related
functions. In particular, it provides code for generic summation
of hypergeometric series. Optimized versions for various special
cases are also provided.
"""

import operator
import math

from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_

from .libintmath import gcd

from .libmpf import (\
    ComplexResult, round_fast, round_nearest,
    negative_rnd, bitcount, to_fixed, from_man_exp, from_int, to_int,
    from_rational,
    fzero, fone, fnone, ftwo, finf, fninf, fnan,
    mpf_sign, mpf_add, mpf_abs, mpf_pos,
    mpf_cmp, mpf_lt, mpf_le, mpf_gt, mpf_min_max,
    mpf_perturb, mpf_neg, mpf_shift, mpf_sub, mpf_mul, mpf_div,
    sqrt_fixed, mpf_sqrt, mpf_rdiv_int, mpf_pow_int,
    to_rational,
)

from .libelefun import (\
    mpf_pi, mpf_exp, mpf_log, pi_fixed, mpf_cos_sin, mpf_cos, mpf_sin,
    mpf_sqrt, agm_fixed,
)

from .libmpc import (\
    mpc_one, mpc_sub, mpc_mul_mpf, mpc_mul, mpc_neg, complex_int_pow,
    mpc_div, mpc_add_mpf, mpc_sub_mpf,
    mpc_log, mpc_add, mpc_pos, mpc_shift,
    mpc_is_infnan, mpc_zero, mpc_sqrt, mpc_abs,
    mpc_mpf_div, mpc_square, mpc_exp
)

from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed

class NoConvergence(Exception):
    pass


#-----------------------------------------------------------------------#
#                                                                       #
#                     Generic hypergeometric series                     #
#                                                                       #
#-----------------------------------------------------------------------#

"""
TODO:

1. proper mpq parsing
2. imaginary z special-cased (also: rational, integer?)
3. more clever handling of series that don't converge because of stupid
   upwards rounding
4. checking for cancellation

"""

def make_hyp_summator(key):
    """
    Returns a function that sums a generalized hypergeometric series,
    for given parameter types (integer, rational, real, complex).

    """
    p, q, param_types, ztype = key

    pstring = "".join(param_types)
    fname = "hypsum_%i_%i_%s_%s_%s" % (p, q, pstring[:p], pstring[p:], ztype)
    #print "generating hypsum", fname

    have_complex_param = 'C' in param_types
    have_complex_arg = ztype == 'C'
    have_complex = have_complex_param or have_complex_arg

    source = []
    add = source.append

    aint = []
    arat = []
    bint = []
    brat = []
    areal = []
    breal = []
    acomplex = []
    bcomplex = []

    #add("wp = prec + 40")
    add("MAX = kwargs.get('maxterms', wp*100)")
    add("HIGH = MPZ_ONE<<epsshift")
    add("LOW = -HIGH")

    # Setup code
    add("SRE = PRE = one = (MPZ_ONE << wp)")
    if have_complex:
        add("SIM = PIM = MPZ_ZERO")

    if have_complex_arg:
        add("xsign, xm, xe, xbc = z[0]")
        add("if xsign: xm = -xm")
        add("ysign, ym, ye, ybc = z[1]")
        add("if ysign: ym = -ym")
    else:
        add("xsign, xm, xe, xbc = z")
        add("if xsign: xm = -xm")

    add("offset = xe + wp")
    add("if offset >= 0:")
    add("    ZRE = xm << offset")
    add("else:")
    add("    ZRE = xm >> (-offset)")
    if have_complex_arg:
        add("offset = ye + wp")
        add("if offset >= 0:")
        add("    ZIM = ym << offset")
        add("else:")
        add("    ZIM = ym >> (-offset)")

    for i, flag in enumerate(param_types):
        W = ["A", "B"][i >= p]
        if flag == 'Z':
            ([aint,bint][i >= p]).append(i)
            add("%sINT_%i = coeffs[%i]" % (W, i, i))
        elif flag == 'Q':
            ([arat,brat][i >= p]).append(i)
            add("%sP_%i, %sQ_%i = coeffs[%i]._mpq_" % (W, i, W, i, i))
        elif flag == 'R':
            ([areal,breal][i >= p]).append(i)
            add("xsign, xm, xe, xbc = coeffs[%i]._mpf_" % i)
            add("if xsign: xm = -xm")
            add("offset = xe + wp")
            add("if offset >= 0:")
            add("    %sREAL_%i = xm << offset" % (W, i))
            add("else:")
            add("    %sREAL_%i = xm >> (-offset)" % (W, i))
        elif flag == 'C':
            ([acomplex,bcomplex][i >= p]).append(i)
            add("__re, __im = coeffs[%i]._mpc_" % i)
            add("xsign, xm, xe, xbc = __re")
            add("if xsign: xm = -xm")
            add("ysign, ym, ye, ybc = __im")
            add("if ysign: ym = -ym")

            add("offset = xe + wp")
            add("if offset >= 0:")
            add("    %sCRE_%i = xm << offset" % (W, i))
            add("else:")
            add("    %sCRE_%i = xm >> (-offset)" % (W, i))
            add("offset = ye + wp")
            add("if offset >= 0:")
            add("    %sCIM_%i = ym << offset" % (W, i))
            add("else:")
            add("    %sCIM_%i = ym >> (-offset)" % (W, i))
        else:
            raise ValueError

    l_areal = len(areal)
    l_breal = len(breal)
    cancellable_real = min(l_areal, l_breal)
    noncancellable_real_num = areal[cancellable_real:]
    noncancellable_real_den = breal[cancellable_real:]

    # LOOP
    add("for n in xrange(1,10**8):")

    add("    if n in magnitude_check:")
    add("        p_mag = bitcount(abs(PRE))")
    if have_complex:
        add("        p_mag = max(p_mag, bitcount(abs(PIM)))")
    add("        magnitude_check[n] = wp-p_mag")

    # Real factors
    multiplier = " * ".join(["AINT_#".replace("#", str(i)) for i in aint] + \
                            ["AP_#".replace("#", str(i)) for i in arat] + \
                            ["BQ_#".replace("#", str(i)) for i in brat])

    divisor    = " * ".join(["BINT_#".replace("#", str(i)) for i in bint] + \
                            ["BP_#".replace("#", str(i)) for i in brat] + \
                            ["AQ_#".replace("#", str(i)) for i in arat] + ["n"])

    if multiplier:
        add("    mul = " + multiplier)
    add("    div = " + divisor)

    # Check for singular terms
    add("    if not div:")
    if multiplier:
        add("        if not mul:")
        add("            break")
    add("        raise ZeroDivisionError")

    # Update product
    if have_complex:

        # TODO: when there are several real parameters and just a few complex
        # (maybe just the complex argument), we only need to do about
        # half as many ops if we accumulate the real factor in a single real variable
        for k in range(cancellable_real): add("    PRE = PRE * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PRE = (PRE * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PRE = (PRE << wp) // BREAL_#".replace("#", str(i)))
        for k in range(cancellable_real): add("    PIM = PIM * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PIM = (PIM * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PIM = (PIM << wp) // BREAL_#".replace("#", str(i)))

        if multiplier:
            if have_complex_arg:
                add("    PRE, PIM = (mul*(PRE*ZRE-PIM*ZIM))//div, (mul*(PIM*ZRE+PRE*ZIM))//div")
                add("    PRE >>= wp")
                add("    PIM >>= wp")
            else:
                add("    PRE = ((mul * PRE * ZRE) >> wp) // div")
                add("    PIM = ((mul * PIM * ZRE) >> wp) // div")
        else:
            if have_complex_arg:
                add("    PRE, PIM = (PRE*ZRE-PIM*ZIM)//div, (PIM*ZRE+PRE*ZIM)//div")
                add("    PRE >>= wp")
                add("    PIM >>= wp")
            else:
                add("    PRE = ((PRE * ZRE) >> wp) // div")
                add("    PIM = ((PIM * ZRE) >> wp) // div")

        for i in acomplex:
            add("    PRE, PIM = PRE*ACRE_#-PIM*ACIM_#, PIM*ACRE_#+PRE*ACIM_#".replace("#", str(i)))
            add("    PRE >>= wp")
            add("    PIM >>= wp")

        for i in bcomplex:
            add("    mag = BCRE_#*BCRE_#+BCIM_#*BCIM_#".replace("#", str(i)))
            add("    re = PRE*BCRE_# + PIM*BCIM_#".replace("#", str(i)))
            add("    im = PIM*BCRE_# - PRE*BCIM_#".replace("#", str(i)))
            add("    PRE = (re << wp) // mag".replace("#", str(i)))
            add("    PIM = (im << wp) // mag".replace("#", str(i)))

    else:
        for k in range(cancellable_real): add("    PRE = PRE * AREAL_%i // BREAL_%i" % (areal[k], breal[k]))
        for i in noncancellable_real_num: add("    PRE = (PRE * AREAL_#) >> wp".replace("#", str(i)))
        for i in noncancellable_real_den: add("    PRE = (PRE << wp) // BREAL_#".replace("#", str(i)))
        if multiplier:
            add("    PRE = ((PRE * mul * ZRE) >> wp) // div")
        else:
            add("    PRE = ((PRE * ZRE) >> wp) // div")

    # Add product to sum
    if have_complex:
        add("    SRE += PRE")
        add("    SIM += PIM")
        add("    if (HIGH > PRE > LOW) and (HIGH > PIM > LOW):")
        add("        break")
    else:
        add("    SRE += PRE")
        add("    if HIGH > PRE > LOW:")
        add("        break")

    #add("    from mpmath import nprint, log, ldexp")
    #add("    nprint([n, log(abs(PRE),2), ldexp(PRE,-wp)])")

    add("    if n > MAX:")
    add("        raise NoConvergence('Hypergeometric series converges too slowly. Try increasing maxterms.')")

    # +1 all parameters for next loop
    for i in aint:     add("    AINT_# += 1".replace("#", str(i)))
    for i in bint:     add("    BINT_# += 1".replace("#", str(i)))
    for i in arat:     add("    AP_# += AQ_#".replace("#", str(i)))
    for i in brat:     add("    BP_# += BQ_#".replace("#", str(i)))
    for i in areal:    add("    AREAL_# += one".replace("#", str(i)))
    for i in breal:    add("    BREAL_# += one".replace("#", str(i)))
    for i in acomplex: add("    ACRE_# += one".replace("#", str(i)))
    for i in bcomplex: add("    BCRE_# += one".replace("#", str(i)))

    if have_complex:
        add("a = from_man_exp(SRE, -wp, prec, 'n')")
        add("b = from_man_exp(SIM, -wp, prec, 'n')")

        add("if SRE:")
        add("    if SIM:")
        add("        magn = max(a[2]+a[3], b[2]+b[3])")
        add("    else:")
        add("        magn = a[2]+a[3]")
        add("elif SIM:")
        add("    magn = b[2]+b[3]")
        add("else:")
        add("    magn = -wp+1")

        add("return (a, b), True, magn")
    else:
        add("a = from_man_exp(SRE, -wp, prec, 'n')")

        add("if SRE:")
        add("    magn = a[2]+a[3]")
        add("else:")
        add("    magn = -wp+1")

        add("return a, False, magn")

    source = "\n".join(("    " + line) for line in source)
    source = ("def %s(coeffs, z, prec, wp, epsshift, magnitude_check, **kwargs):\n" % fname) + source

    namespace = {}

    exec_(source, globals(), namespace)

    #print source
    return source, namespace[fname]


if BACKEND == 'sage':

    def make_hyp_summator(key):
        """
        Returns a function that sums a generalized hypergeometric series,
        for given parameter types (integer, rational, real, complex).
        """
        from sage.libs.mpmath.ext_main import hypsum_internal
        p, q, param_types, ztype = key
        def _hypsum(coeffs, z, prec, wp, epsshift, magnitude_check, **kwargs):
            return hypsum_internal(p, q, param_types, ztype, coeffs, z,
                prec, wp, epsshift, magnitude_check, kwargs)

        return "(none)", _hypsum


#-----------------------------------------------------------------------#
#                                                                       #
#                              Error functions                          #
#                                                                       #
#-----------------------------------------------------------------------#

# TODO: mpf_erf should call mpf_erfc when appropriate (currently
#    only the converse delegation is implemented)

def mpf_erf(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fzero
        if x == finf: return fone
        if x== fninf: return fnone
        return fnan
    size = exp + bc
    lg = math.log
    # The approximation erf(x) = 1 is accurate to > x^2 * log(e,2) bits
    if size > 3 and 2*(size-1) + 0.528766 > lg(prec,2):
        if sign:
            return mpf_perturb(fnone, 0, prec, rnd)
        else:
            return mpf_perturb(fone, 1, prec, rnd)
    # erf(x) ~ 2*x/sqrt(pi) close to 0
    if size < -prec:
        # 2*x
        x = mpf_shift(x,1)
        c = mpf_sqrt(mpf_pi(prec+20), prec+20)
        # TODO: interval rounding
        return mpf_div(x, c, prec, rnd)
    wp = prec + abs(size) + 25
    # Taylor series for erf, fixed-point summation
    t = abs(to_fixed(x, wp))
    t2 = (t*t) >> wp
    s, term, k = t, 12345, 1
    while term:
        t = ((t * t2) >> wp) // k
        term = t // (2*k+1)
        if k & 1:
            s -= term
        else:
            s += term
        k += 1
    s = (s << (wp+1)) // sqrt_fixed(pi_fixed(wp), wp)
    if sign:
        s = -s
    return from_man_exp(s, -wp, prec, rnd)

# If possible, we use the asymptotic series for erfc.
# This is an alternating divergent asymptotic series, so
# the error is at most equal to the first omitted term.
# Here we check if the smallest term is small enough
# for a given x and precision
def erfc_check_series(x, prec):
    n = to_int(x)
    if n**2 * 1.44 > prec:
        return True
    return False

def mpf_erfc(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fone
        if x == finf: return fzero
        if x == fninf: return ftwo
        return fnan
    wp = prec + 20
    mag = bc+exp
    # Preserve full accuracy when exponent grows huge
    wp += max(0, 2*mag)
    regular_erf = sign or mag < 2
    if regular_erf or not erfc_check_series(x, wp):
        if regular_erf:
            return mpf_sub(fone, mpf_erf(x, prec+10, negative_rnd[rnd]), prec, rnd)
        # 1-erf(x) ~ exp(-x^2), increase prec to deal with cancellation
        n = to_int(x)+1
        return mpf_sub(fone, mpf_erf(x, prec + int(n**2*1.44) + 10), prec, rnd)
    s = term = MPZ_ONE << wp
    term_prev = 0
    t = (2 * to_fixed(x, wp) ** 2) >> wp
    k = 1
    while 1:
        term = ((term * (2*k - 1)) << wp) // t
        if k > 4 and term > term_prev or not term:
            break
        if k & 1:
            s -= term
        else:
            s += term
        term_prev = term
        #print k, to_str(from_man_exp(term, -wp, 50), 10)
        k += 1
    s = (s << wp) // sqrt_fixed(pi_fixed(wp), wp)
    s = from_man_exp(s, -wp, wp)
    z = mpf_exp(mpf_neg(mpf_mul(x,x,wp),wp),wp)
    y = mpf_div(mpf_mul(z, s, wp), x, prec, rnd)
    return y


#-----------------------------------------------------------------------#
#                                                                       #
#                         Exponential integrals                         #
#                                                                       #
#-----------------------------------------------------------------------#

def ei_taylor(x, prec):
    s = t = x
    k = 2
    while t:
        t = ((t*x) >> prec) // k
        s += t // k
        k += 1
    return s

def complex_ei_taylor(zre, zim, prec):
    _abs = abs
    sre = tre = zre
    sim = tim = zim
    k = 2
    while _abs(tre) + _abs(tim) > 5:
        tre, tim = ((tre*zre-tim*zim)//k)>>prec, ((tre*zim+tim*zre)//k)>>prec
        sre += tre // k
        sim += tim // k
        k += 1
    return sre, sim

def ei_asymptotic(x, prec):
    one = MPZ_ONE << prec
    x = t = ((one << prec) // x)
    s = one + x
    k = 2
    while t:
        t = (k*t*x) >> prec
        s += t
        k += 1
    return s

def complex_ei_asymptotic(zre, zim, prec):
    _abs = abs
    one = MPZ_ONE << prec
    M = (zim*zim + zre*zre) >> prec
    # 1 / z
    xre = tre = (zre << prec) // M
    xim = tim = ((-zim) << prec) // M
    sre = one + xre
    sim = xim
    k = 2
    while _abs(tre) + _abs(tim) > 1000:
        #print tre, tim
        tre, tim = ((tre*xre-tim*xim)*k)>>prec, ((tre*xim+tim*xre)*k)>>prec
        sre += tre
        sim += tim
        k += 1
        if k > prec:
            raise NoConvergence
    return sre, sim

def mpf_ei(x, prec, rnd=round_fast, e1=False):
    if e1:
        x = mpf_neg(x)
    sign, man, exp, bc = x
    if e1 and not sign:
        if x == fzero:
            return finf
        raise ComplexResult("E1(x) for x < 0")
    if man:
        xabs = 0, man, exp, bc
        xmag = exp+bc
        wp = prec + 20
        can_use_asymp = xmag > wp
        if not can_use_asymp:
            if exp >= 0:
                xabsint = man << exp
            else:
                xabsint = man >> (-exp)
            can_use_asymp = xabsint > int(wp*0.693) + 10
        if can_use_asymp:
            if xmag > wp:
                v = fone
            else:
                v = from_man_exp(ei_asymptotic(to_fixed(x, wp), wp), -wp)
            v = mpf_mul(v, mpf_exp(x, wp), wp)
            v = mpf_div(v, x, prec, rnd)
        else:
            wp += 2*int(to_int(xabs))
            u = to_fixed(x, wp)
            v = ei_taylor(u, wp) + euler_fixed(wp)
            t1 = from_man_exp(v,-wp)
            t2 = mpf_log(xabs,wp)
            v = mpf_add(t1, t2, prec, rnd)
    else:
        if x == fzero: v = fninf
        elif x == finf: v = finf
        elif x == fninf: v = fzero
        else: v = fnan
    if e1:
        v = mpf_neg(v)
    return v

def mpc_ei(z, prec, rnd=round_fast, e1=False):
    if e1:
        z = mpc_neg(z)
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero:
        if e1:
            x = mpf_neg(mpf_ei(a, prec, rnd))
            if not asign:
                y = mpf_neg(mpf_pi(prec, rnd))
            else:
                y = fzero
            return x, y
        else:
            return mpf_ei(a, prec, rnd), fzero
    if a != fzero:
        if not aman or not bman:
            return (fnan, fnan)
    wp = prec + 40
    amag = aexp+abc
    bmag = bexp+bbc
    zmag = max(amag, bmag)
    can_use_asymp = zmag > wp
    if not can_use_asymp:
        zabsint = abs(to_int(a)) + abs(to_int(b))
        can_use_asymp = zabsint > int(wp*0.693) + 20
    try:
        if can_use_asymp:
            if zmag > wp:
                v = fone, fzero
            else:
                zre = to_fixed(a, wp)
                zim = to_fixed(b, wp)
                vre, vim = complex_ei_asymptotic(zre, zim, wp)
                v = from_man_exp(vre, -wp), from_man_exp(vim, -wp)
            v = mpc_mul(v, mpc_exp(z, wp), wp)
            v = mpc_div(v, z, wp)
            if e1:
                v = mpc_neg(v, prec, rnd)
            else:
                x, y = v
                if bsign:
                    v = mpf_pos(x, prec, rnd), mpf_sub(y, mpf_pi(wp), prec, rnd)
                else:
                    v = mpf_pos(x, prec, rnd), mpf_add(y, mpf_pi(wp), prec, rnd)
            return v
    except NoConvergence:
        pass
    #wp += 2*max(0,zmag)
    wp += 2*int(to_int(mpc_abs(z, 5)))
    zre = to_fixed(a, wp)
    zim = to_fixed(b, wp)
    vre, vim = complex_ei_taylor(zre, zim, wp)
    vre += euler_fixed(wp)
    v = from_man_exp(vre,-wp), from_man_exp(vim,-wp)
    if e1:
        u = mpc_log(mpc_neg(z),wp)
    else:
        u = mpc_log(z,wp)
    v = mpc_add(v, u, prec, rnd)
    if e1:
        v = mpc_neg(v)
    return v

def mpf_e1(x, prec, rnd=round_fast):
    return mpf_ei(x, prec, rnd, True)

def mpc_e1(x, prec, rnd=round_fast):
    return mpc_ei(x, prec, rnd, True)

def mpf_expint(n, x, prec, rnd=round_fast, gamma=False):
    """
    E_n(x), n an integer, x real

    With gamma=True, computes Gamma(n,x)   (upper incomplete gamma function)

    Returns (real, None) if real, otherwise (real, imag)
    The imaginary part is an optional branch cut term

    """
    sign, man, exp, bc = x
    if not man:
        if gamma:
            if x == fzero:
                # Actually gamma function pole
                if n <= 0:
                    return finf, None
                return mpf_gamma_int(n, prec, rnd), None
            if x == finf:
                return fzero, None
            # TODO: could return finite imaginary value at -inf
            return fnan, fnan
        else:
            if x == fzero:
                if n > 1:
                    return from_rational(1, n-1, prec, rnd), None
                else:
                    return finf, None
            if x == finf:
                return fzero, None
            return fnan, fnan
    n_orig = n
    if gamma:
        n = 1-n
    wp = prec + 20
    xmag = exp + bc
    # Beware of near-poles
    if xmag < -10:
        raise NotImplementedError
    nmag = bitcount(abs(n))
    have_imag = n > 0 and sign
    negx = mpf_neg(x)
    # Skip series if direct convergence
    if n == 0 or 2*nmag - xmag < -wp:
        if gamma:
            v = mpf_exp(negx, wp)
            re = mpf_mul(v, mpf_pow_int(x, n_orig-1, wp), prec, rnd)
        else:
            v = mpf_exp(negx, wp)
            re = mpf_div(v, x, prec, rnd)
    else:
        # Finite number of terms, or...
        can_use_asymptotic_series = -3*wp < n <= 0
        # ...large enough?
        if not can_use_asymptotic_series:
            xi = abs(to_int(x))
            m = min(max(1, xi-n), 2*wp)
            siz = -n*nmag + (m+n)*bitcount(abs(m+n)) - m*xmag - (144*m//100)
            tol = -wp-10
            can_use_asymptotic_series = siz < tol
        if can_use_asymptotic_series:
            r = ((-MPZ_ONE) << (wp+wp)) // to_fixed(x, wp)
            m = n
            t = r*m
            s = MPZ_ONE << wp
            while m and t:
                s += t
                m += 1
                t = (m*r*t) >> wp
            v = mpf_exp(negx, wp)
            if gamma:
                # ~ exp(-x) * x^(n-1) * (1 + ...)
                v = mpf_mul(v, mpf_pow_int(x, n_orig-1, wp), wp)
            else:
                # ~ exp(-x)/x * (1 + ...)
                v = mpf_div(v, x, wp)
            re = mpf_mul(v, from_man_exp(s, -wp), prec, rnd)
        elif n == 1:
            re = mpf_neg(mpf_ei(negx, prec, rnd))
        elif n > 0 and n < 3*wp:
            T1 = mpf_neg(mpf_ei(negx, wp))
            if gamma:
                if n_orig & 1:
                    T1 = mpf_neg(T1)
            else:
                T1 = mpf_mul(T1, mpf_pow_int(negx, n-1, wp), wp)
            r = t = to_fixed(x, wp)
            facs = [1] * (n-1)
            for k in range(1,n-1):
                facs[k] = facs[k-1] * k
            facs = facs[::-1]
            s = facs[0] << wp
            for k in range(1, n-1):
                if k & 1:
                    s -= facs[k] * t
                else:
                    s += facs[k] * t
                t = (t*r) >> wp
            T2 = from_man_exp(s, -wp, wp)
            T2 = mpf_mul(T2, mpf_exp(negx, wp))
            if gamma:
                T2 = mpf_mul(T2, mpf_pow_int(x, n_orig, wp), wp)
            R = mpf_add(T1, T2)
            re = mpf_div(R, from_int(ifac(n-1)), prec, rnd)
        else:
            raise NotImplementedError
    if have_imag:
        M = from_int(-ifac(n-1))
        if gamma:
            im = mpf_div(mpf_pi(wp), M, prec, rnd)
            if n_orig & 1:
                im = mpf_neg(im)
        else:
            im = mpf_div(mpf_mul(mpf_pi(wp), mpf_pow_int(negx, n_orig-1, wp), wp), M, prec, rnd)
        return re, im
    else:
        return re, None

def mpf_ci_si_taylor(x, wp, which=0):
    """
    0 - Ci(x) - (euler+log(x))
    1 - Si(x)
    """
    x = to_fixed(x, wp)
    x2 = -(x*x) >> wp
    if which == 0:
        s, t, k = 0, (MPZ_ONE<<wp), 2
    else:
        s, t, k = x, x, 3
    while t:
        t = (t*x2//(k*(k-1)))>>wp
        s += t//k
        k += 2
    return from_man_exp(s, -wp)

def mpc_ci_si_taylor(re, im, wp, which=0):
    # The following code is only designed for small arguments,
    # and not too small arguments (for relative accuracy)
    if re[1]:
        mag = re[2]+re[3]
    elif im[1]:
        mag = im[2]+im[3]
    if im[1]:
        mag = max(mag, im[2]+im[3])
    if mag > 2 or mag < -wp:
        raise NotImplementedError
    wp += (2-mag)
    zre = to_fixed(re, wp)
    zim = to_fixed(im, wp)
    z2re = (zim*zim-zre*zre)>>wp
    z2im = (-2*zre*zim)>>wp
    tre = zre
    tim = zim
    one = MPZ_ONE<<wp
    if which == 0:
        sre, sim, tre, tim, k = 0, 0, (MPZ_ONE<<wp), 0, 2
    else:
        sre, sim, tre, tim, k = zre, zim, zre, zim, 3
    while max(abs(tre), abs(tim)) > 2:
        f = k*(k-1)
        tre, tim = ((tre*z2re-tim*z2im)//f)>>wp, ((tre*z2im+tim*z2re)//f)>>wp
        sre += tre//k
        sim += tim//k
        k += 2
    return from_man_exp(sre, -wp), from_man_exp(sim, -wp)

def mpf_ci_si(x, prec, rnd=round_fast, which=2):
    """
    Calculation of Ci(x), Si(x) for real x.

    which = 0 -- returns (Ci(x), -)
    which = 1 -- returns (Si(x), -)
    which = 2 -- returns (Ci(x), Si(x))

    Note: if x < 0, Ci(x) needs an additional imaginary term, pi*i.
    """
    wp = prec + 20
    sign, man, exp, bc = x
    ci, si = None, None
    if not man:
        if x == fzero:
            return (fninf, fzero)
        if x == fnan:
            return (x, x)
        ci = fzero
        if which != 0:
            if x == finf:
                si = mpf_shift(mpf_pi(prec, rnd), -1)
            if x == fninf:
                si = mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))
        return (ci, si)
    # For small x: Ci(x) ~ euler + log(x), Si(x) ~ x
    mag = exp+bc
    if mag < -wp:
        if which != 0:
            si = mpf_perturb(x, 1-sign, prec, rnd)
        if which != 1:
            y = mpf_euler(wp)
            xabs = mpf_abs(x)
            ci = mpf_add(y, mpf_log(xabs, wp), prec, rnd)
        return ci, si
    # For huge x: Ci(x) ~ sin(x)/x, Si(x) ~ pi/2
    elif mag > wp:
        if which != 0:
            if sign:
                si = mpf_neg(mpf_pi(prec, negative_rnd[rnd]))
            else:
                si = mpf_pi(prec, rnd)
            si = mpf_shift(si, -1)
        if which != 1:
            ci = mpf_div(mpf_sin(x, wp), x, prec, rnd)
        return ci, si
    else:
        wp += abs(mag)
    # Use an asymptotic series? The smallest value of n!/x^n
    # occurs for n ~ x, where the magnitude is ~ exp(-x).
    asymptotic = mag-1 > math.log(wp, 2)
    # Case 1: convergent series near 0
    if not asymptotic:
        if which != 0:
            si = mpf_pos(mpf_ci_si_taylor(x, wp, 1), prec, rnd)
        if which != 1:
            ci = mpf_ci_si_taylor(x, wp, 0)
            ci = mpf_add(ci, mpf_euler(wp), wp)
            ci = mpf_add(ci, mpf_log(mpf_abs(x), wp), prec, rnd)
        return ci, si
    x = mpf_abs(x)
    # Case 2: asymptotic series for x >> 1
    xf = to_fixed(x, wp)
    xr = (MPZ_ONE<<(2*wp)) // xf   # 1/x
    s1 = (MPZ_ONE << wp)
    s2 = xr
    t = xr
    k = 2
    while t:
        t = -t
        t = (t*xr*k)>>wp
        k += 1
        s1 += t
        t = (t*xr*k)>>wp
        k += 1
        s2 += t
    s1 = from_man_exp(s1, -wp)
    s2 = from_man_exp(s2, -wp)
    s1 = mpf_div(s1, x, wp)
    s2 = mpf_div(s2, x, wp)
    cos, sin = mpf_cos_sin(x, wp)
    # Ci(x) = sin(x)*s1-cos(x)*s2
    # Si(x) = pi/2-cos(x)*s1-sin(x)*s2
    if which != 0:
        si = mpf_add(mpf_mul(cos, s1), mpf_mul(sin, s2), wp)
        si = mpf_sub(mpf_shift(mpf_pi(wp), -1), si, wp)
        if sign:
            si = mpf_neg(si)
        si = mpf_pos(si, prec, rnd)
    if which != 1:
        ci = mpf_sub(mpf_mul(sin, s1), mpf_mul(cos, s2), prec, rnd)
    return ci, si

def mpf_ci(x, prec, rnd=round_fast):
    if mpf_sign(x) < 0:
        raise ComplexResult
    return mpf_ci_si(x, prec, rnd, 0)[0]

def mpf_si(x, prec, rnd=round_fast):
    return mpf_ci_si(x, prec, rnd, 1)[1]

def mpc_ci(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        ci = mpf_ci_si(re, prec, rnd, 0)[0]
        if mpf_sign(re) < 0:
            return (ci, mpf_pi(prec, rnd))
        return (ci, fzero)
    wp = prec + 20
    cre, cim = mpc_ci_si_taylor(re, im, wp, 0)
    cre = mpf_add(cre, mpf_euler(wp), wp)
    ci = mpc_add((cre, cim), mpc_log(z, wp), prec, rnd)
    return ci

def mpc_si(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        return (mpf_ci_si(re, prec, rnd, 1)[1], fzero)
    wp = prec + 20
    z = mpc_ci_si_taylor(re, im, wp, 1)
    return mpc_pos(z, prec, rnd)


#-----------------------------------------------------------------------#
#                                                                       #
#                             Bessel functions                          #
#                                                                       #
#-----------------------------------------------------------------------#

# A Bessel function of the first kind of integer order, J_n(x), is
# given by the power series

#             oo
#             ___         k         2 k + n
#            \        (-1)     / x \
#    J_n(x) = )    ----------- | - |
#            /___  k! (k + n)! \ 2 /
#            k = 0

# Simplifying the quotient between two successive terms gives the
# ratio x^2 / (-4*k*(k+n)). Hence, we only need one full-precision
# multiplication and one division by a small integer per term.
# The complex version is very similar, the only difference being
# that the multiplication is actually 4 multiplies.

# In the general case, we have
# J_v(x) = (x/2)**v / v! * 0F1(v+1, (-1/4)*z**2)

# TODO: for extremely large x, we could use an asymptotic
# trigonometric approximation.

# TODO: recompute at higher precision if the fixed-point mantissa
# is very small

def mpf_besseljn(n, x, prec, rounding=round_fast):
    prec += 50
    negate = n < 0 and n & 1
    mag = x[2]+x[3]
    n = abs(n)
    wp = prec + 20 + n*bitcount(n)
    if mag < 0:
        wp -= n * mag
    x = to_fixed(x, wp)
    x2 = (x**2) >> wp
    if not n:
        s = t = MPZ_ONE << wp
    else:
        s = t = (x**n // ifac(n)) >> ((n-1)*wp + n)
    k = 1
    while t:
        t = ((t * x2) // (-4*k*(k+n))) >> wp
        s += t
        k += 1
    if negate:
        s = -s
    return from_man_exp(s, -wp, prec, rounding)

def mpc_besseljn(n, z, prec, rounding=round_fast):
    negate = n < 0 and n & 1
    n = abs(n)
    origprec = prec
    zre, zim = z
    mag = max(zre[2]+zre[3], zim[2]+zim[3])
    prec += 20 + n*bitcount(n) + abs(mag)
    if mag < 0:
        prec -= n * mag
    zre = to_fixed(zre, prec)
    zim = to_fixed(zim, prec)
    z2re = (zre**2 - zim**2) >> prec
    z2im = (zre*zim) >> (prec-1)
    if not n:
        sre = tre = MPZ_ONE << prec
        sim = tim = MPZ_ZERO
    else:
        re, im = complex_int_pow(zre, zim, n)
        sre = tre = (re // ifac(n)) >> ((n-1)*prec + n)
        sim = tim = (im // ifac(n)) >> ((n-1)*prec + n)
    k = 1
    while abs(tre) + abs(tim) > 3:
        p = -4*k*(k+n)
        tre, tim = tre*z2re - tim*z2im, tim*z2re + tre*z2im
        tre = (tre // p) >> prec
        tim = (tim // p) >> prec
        sre += tre
        sim += tim
        k += 1
    if negate:
        sre = -sre
        sim = -sim
    re = from_man_exp(sre, -prec, origprec, rounding)
    im = from_man_exp(sim, -prec, origprec, rounding)
    return (re, im)

def mpf_agm(a, b, prec, rnd=round_fast):
    """
    Computes the arithmetic-geometric mean agm(a,b) for
    nonnegative mpf values a, b.
    """
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign or bsign:
        raise ComplexResult("agm of a negative number")
    # Handle inf, nan or zero in either operand
    if not (aman and bman):
        if a == fnan or b == fnan:
            return fnan
        if a == finf:
            if b == fzero:
                return fnan
            return finf
        if b == finf:
            if a == fzero:
                return fnan
            return finf
        # agm(0,x) = agm(x,0) = 0
        return fzero
    wp = prec + 20
    amag = aexp+abc
    bmag = bexp+bbc
    mag_delta = amag - bmag
    # Reduce to roughly the same magnitude using floating-point AGM
    abs_mag_delta = abs(mag_delta)
    if abs_mag_delta > 10:
        while abs_mag_delta > 10:
            a, b = mpf_shift(mpf_add(a,b,wp),-1), \
                mpf_sqrt(mpf_mul(a,b,wp),wp)
            abs_mag_delta //= 2
        asign, aman, aexp, abc = a
        bsign, bman, bexp, bbc = b
        amag = aexp+abc
        bmag = bexp+bbc
        mag_delta = amag - bmag
    #print to_float(a), to_float(b)
    # Use agm(a,b) = agm(x*a,x*b)/x to obtain a, b ~= 1
    min_mag = min(amag,bmag)
    max_mag = max(amag,bmag)
    n = 0
    # If too small, we lose precision when going to fixed-point
    if min_mag < -8:
        n = -min_mag
    # If too large, we waste time using fixed-point with large numbers
    elif max_mag > 20:
        n = -max_mag
    if n:
        a = mpf_shift(a, n)
        b = mpf_shift(b, n)
    #print to_float(a), to_float(b)
    af = to_fixed(a, wp)
    bf = to_fixed(b, wp)
    g = agm_fixed(af, bf, wp)
    return from_man_exp(g, -wp-n, prec, rnd)

def mpf_agm1(a, prec, rnd=round_fast):
    """
    Computes the arithmetic-geometric mean agm(1,a) for a nonnegative
    mpf value a.
    """
    return mpf_agm(fone, a, prec, rnd)

def mpc_agm(a, b, prec, rnd=round_fast):
    """
    Complex AGM.

    TODO:
    * check that convergence works as intended
    * optimize
    * select a nonarbitrary branch
    """
    if mpc_is_infnan(a) or mpc_is_infnan(b):
        return fnan, fnan
    if mpc_zero in (a, b):
        return fzero, fzero
    if mpc_neg(a) == b:
        return fzero, fzero
    wp = prec+20
    eps = mpf_shift(fone, -wp+10)
    while 1:
        a1 = mpc_shift(mpc_add(a, b, wp), -1)
        b1 = mpc_sqrt(mpc_mul(a, b, wp), wp)
        a, b = a1, b1
        size = mpf_min_max([mpc_abs(a,10), mpc_abs(b,10)])[1]
        err = mpc_abs(mpc_sub(a, b, 10), 10)
        if size == fzero or mpf_lt(err, mpf_mul(eps, size)):
            return a

def mpc_agm1(a, prec, rnd=round_fast):
    return mpc_agm(mpc_one, a, prec, rnd)

def mpf_ellipk(x, prec, rnd=round_fast):
    if not x[1]:
        if x == fzero:
            return mpf_shift(mpf_pi(prec, rnd), -1)
        if x == fninf:
            return fzero
        if x == fnan:
            return x
    if x == fone:
        return finf
    # TODO: for |x| << 1/2, one could use fall back to
    # pi/2 * hyp2f1_rat((1,2),(1,2),(1,1), x)
    wp = prec + 15
    # Use K(x) = pi/2/agm(1,a) where a = sqrt(1-x)
    # The sqrt raises ComplexResult if x > 0
    a = mpf_sqrt(mpf_sub(fone, x, wp), wp)
    v = mpf_agm1(a, wp)
    r = mpf_div(mpf_pi(wp), v, prec, rnd)
    return mpf_shift(r, -1)

def mpc_ellipk(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        if re == finf:
            return mpc_zero
        if mpf_le(re, fone):
            return mpf_ellipk(re, prec, rnd), fzero
    wp = prec + 15
    a = mpc_sqrt(mpc_sub(mpc_one, z, wp), wp)
    v = mpc_agm1(a, wp)
    r = mpc_mpf_div(mpf_pi(wp), v, prec, rnd)
    return mpc_shift(r, -1)

def mpf_ellipe(x, prec, rnd=round_fast):
    # http://functions.wolfram.com/EllipticIntegrals/
    # EllipticK/20/01/0001/
    # E = (1-m)*(K'(m)*2*m + K(m))
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return mpf_shift(mpf_pi(prec, rnd), -1)
        if x == fninf:
            return finf
        if x == fnan:
            return x
        if x == finf:
            raise ComplexResult
    if x == fone:
        return fone
    wp = prec+20
    mag = exp+bc
    if mag < -wp:
        return mpf_shift(mpf_pi(prec, rnd), -1)
    # Compute a finite difference for K'
    p = max(mag, 0) - wp
    h = mpf_shift(fone, p)
    K = mpf_ellipk(x, 2*wp)
    Kh = mpf_ellipk(mpf_sub(x, h), 2*wp)
    Kdiff = mpf_shift(mpf_sub(K, Kh), -p)
    t = mpf_sub(fone, x)
    b = mpf_mul(Kdiff, mpf_shift(x,1), wp)
    return mpf_mul(t, mpf_add(K, b), prec, rnd)

def mpc_ellipe(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        if re == finf:
            return (fzero, finf)
        if mpf_le(re, fone):
            return mpf_ellipe(re, prec, rnd), fzero
    wp = prec + 15
    mag = mpc_abs(z, 1)
    p = max(mag[2]+mag[3], 0) - wp
    h = mpf_shift(fone, p)
    K = mpc_ellipk(z, 2*wp)
    Kh = mpc_ellipk(mpc_add_mpf(z, h, 2*wp), 2*wp)
    Kdiff = mpc_shift(mpc_sub(Kh, K, wp), -p)
    t = mpc_sub(mpc_one, z, wp)
    b = mpc_mul(Kdiff, mpc_shift(z,1), wp)
    return mpc_mul(t, mpc_add(K, b, wp), prec, rnd)