
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\clsregistry.py ===
# orm/clsregistry.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Routines to handle the string class registry used by declarative.

This system allows specification of classes and expressions used in
:func:`_orm.relationship` using strings.

"""

from __future__ import annotations

import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import attributes
from . import interfaces
from .descriptor_props import SynonymProperty
from .properties import ColumnProperty
from .util import class_mapper
from .. import exc
from .. import inspection
from .. import util
from ..sql.schema import _get_table_key
from ..util.typing import CallableReference

if TYPE_CHECKING:
    from .relationships import RelationshipProperty
    from ..sql.schema import MetaData
    from ..sql.schema import Table

_T = TypeVar("_T", bound=Any)

_ClsRegistryType = MutableMapping[str, Union[type, "ClsRegistryToken"]]

# strong references to registries which we place in
# the _decl_class_registry, which is usually weak referencing.
# the internal registries here link to classes with weakrefs and remove
# themselves when all references to contained classes are removed.
_registries: Set[ClsRegistryToken] = set()


def add_class(
    classname: str, cls: Type[_T], decl_class_registry: _ClsRegistryType
) -> None:
    """Add a class to the _decl_class_registry associated with the
    given declarative class.

    """
    if classname in decl_class_registry:
        # class already exists.
        existing = decl_class_registry[classname]
        if not isinstance(existing, _MultipleClassMarker):
            decl_class_registry[classname] = _MultipleClassMarker(
                [cls, cast("Type[Any]", existing)]
            )
    else:
        decl_class_registry[classname] = cls

    try:
        root_module = cast(
            _ModuleMarker, decl_class_registry["_sa_module_registry"]
        )
    except KeyError:
        decl_class_registry["_sa_module_registry"] = root_module = (
            _ModuleMarker("_sa_module_registry", None)
        )

    tokens = cls.__module__.split(".")

    # build up a tree like this:
    # modulename:  myapp.snacks.nuts
    #
    # myapp->snack->nuts->(classes)
    # snack->nuts->(classes)
    # nuts->(classes)
    #
    # this allows partial token paths to be used.
    while tokens:
        token = tokens.pop(0)
        module = root_module.get_module(token)
        for token in tokens:
            module = module.get_module(token)

        try:
            module.add_class(classname, cls)
        except AttributeError as ae:
            if not isinstance(module, _ModuleMarker):
                raise exc.InvalidRequestError(
                    f'name "{classname}" matches both a '
                    "class name and a module name"
                ) from ae
            else:
                raise


def remove_class(
    classname: str, cls: Type[Any], decl_class_registry: _ClsRegistryType
) -> None:
    if classname in decl_class_registry:
        existing = decl_class_registry[classname]
        if isinstance(existing, _MultipleClassMarker):
            existing.remove_item(cls)
        else:
            del decl_class_registry[classname]

    try:
        root_module = cast(
            _ModuleMarker, decl_class_registry["_sa_module_registry"]
        )
    except KeyError:
        return

    tokens = cls.__module__.split(".")

    while tokens:
        token = tokens.pop(0)
        module = root_module.get_module(token)
        for token in tokens:
            module = module.get_module(token)
        try:
            module.remove_class(classname, cls)
        except AttributeError:
            if not isinstance(module, _ModuleMarker):
                pass
            else:
                raise


def _key_is_empty(
    key: str,
    decl_class_registry: _ClsRegistryType,
    test: Callable[[Any], bool],
) -> bool:
    """test if a key is empty of a certain object.

    used for unit tests against the registry to see if garbage collection
    is working.

    "test" is a callable that will be passed an object should return True
    if the given object is the one we were looking for.

    We can't pass the actual object itself b.c. this is for testing garbage
    collection; the caller will have to have removed references to the
    object itself.

    """
    if key not in decl_class_registry:
        return True

    thing = decl_class_registry[key]
    if isinstance(thing, _MultipleClassMarker):
        for sub_thing in thing.contents:
            if test(sub_thing):
                return False
        else:
            raise NotImplementedError("unknown codepath")
    else:
        return not test(thing)


class ClsRegistryToken:
    """an object that can be in the registry._class_registry as a value."""

    __slots__ = ()


class _MultipleClassMarker(ClsRegistryToken):
    """refers to multiple classes of the same name
    within _decl_class_registry.

    """

    __slots__ = "on_remove", "contents", "__weakref__"

    contents: Set[weakref.ref[Type[Any]]]
    on_remove: CallableReference[Optional[Callable[[], None]]]

    def __init__(
        self,
        classes: Iterable[Type[Any]],
        on_remove: Optional[Callable[[], None]] = None,
    ):
        self.on_remove = on_remove
        self.contents = {
            weakref.ref(item, self._remove_item) for item in classes
        }
        _registries.add(self)

    def remove_item(self, cls: Type[Any]) -> None:
        self._remove_item(weakref.ref(cls))

    def __iter__(self) -> Generator[Optional[Type[Any]], None, None]:
        return (ref() for ref in self.contents)

    def attempt_get(self, path: List[str], key: str) -> Type[Any]:
        if len(self.contents) > 1:
            raise exc.InvalidRequestError(
                'Multiple classes found for path "%s" '
                "in the registry of this declarative "
                "base. Please use a fully module-qualified path."
                % (".".join(path + [key]))
            )
        else:
            ref = list(self.contents)[0]
            cls = ref()
            if cls is None:
                raise NameError(key)
            return cls

    def _remove_item(self, ref: weakref.ref[Type[Any]]) -> None:
        self.contents.discard(ref)
        if not self.contents:
            _registries.discard(self)
            if self.on_remove:
                self.on_remove()

    def add_item(self, item: Type[Any]) -> None:
        # protect against class registration race condition against
        # asynchronous garbage collection calling _remove_item,
        # [ticket:3208] and [ticket:10782]
        modules = {
            cls.__module__
            for cls in [ref() for ref in list(self.contents)]
            if cls is not None
        }
        if item.__module__ in modules:
            util.warn(
                "This declarative base already contains a class with the "
                "same class name and module name as %s.%s, and will "
                "be replaced in the string-lookup table."
                % (item.__module__, item.__name__)
            )
        self.contents.add(weakref.ref(item, self._remove_item))


class _ModuleMarker(ClsRegistryToken):
    """Refers to a module name within
    _decl_class_registry.

    """

    __slots__ = "parent", "name", "contents", "mod_ns", "path", "__weakref__"

    parent: Optional[_ModuleMarker]
    contents: Dict[str, Union[_ModuleMarker, _MultipleClassMarker]]
    mod_ns: _ModNS
    path: List[str]

    def __init__(self, name: str, parent: Optional[_ModuleMarker]):
        self.parent = parent
        self.name = name
        self.contents = {}
        self.mod_ns = _ModNS(self)
        if self.parent:
            self.path = self.parent.path + [self.name]
        else:
            self.path = []
        _registries.add(self)

    def __contains__(self, name: str) -> bool:
        return name in self.contents

    def __getitem__(self, name: str) -> ClsRegistryToken:
        return self.contents[name]

    def _remove_item(self, name: str) -> None:
        self.contents.pop(name, None)
        if not self.contents:
            if self.parent is not None:
                self.parent._remove_item(self.name)
            _registries.discard(self)

    def resolve_attr(self, key: str) -> Union[_ModNS, Type[Any]]:
        return self.mod_ns.__getattr__(key)

    def get_module(self, name: str) -> _ModuleMarker:
        if name not in self.contents:
            marker = _ModuleMarker(name, self)
            self.contents[name] = marker
        else:
            marker = cast(_ModuleMarker, self.contents[name])
        return marker

    def add_class(self, name: str, cls: Type[Any]) -> None:
        if name in self.contents:
            existing = cast(_MultipleClassMarker, self.contents[name])
            try:
                existing.add_item(cls)
            except AttributeError as ae:
                if not isinstance(existing, _MultipleClassMarker):
                    raise exc.InvalidRequestError(
                        f'name "{name}" matches both a '
                        "class name and a module name"
                    ) from ae
                else:
                    raise
        else:
            self.contents[name] = _MultipleClassMarker(
                [cls], on_remove=lambda: self._remove_item(name)
            )

    def remove_class(self, name: str, cls: Type[Any]) -> None:
        if name in self.contents:
            existing = cast(_MultipleClassMarker, self.contents[name])
            existing.remove_item(cls)


class _ModNS:
    __slots__ = ("__parent",)

    __parent: _ModuleMarker

    def __init__(self, parent: _ModuleMarker):
        self.__parent = parent

    def __getattr__(self, key: str) -> Union[_ModNS, Type[Any]]:
        try:
            value = self.__parent.contents[key]
        except KeyError:
            pass
        else:
            if value is not None:
                if isinstance(value, _ModuleMarker):
                    return value.mod_ns
                else:
                    assert isinstance(value, _MultipleClassMarker)
                    return value.attempt_get(self.__parent.path, key)
        raise NameError(
            "Module %r has no mapped classes "
            "registered under the name %r" % (self.__parent.name, key)
        )


class _GetColumns:
    __slots__ = ("cls",)

    cls: Type[Any]

    def __init__(self, cls: Type[Any]):
        self.cls = cls

    def __getattr__(self, key: str) -> Any:
        mp = class_mapper(self.cls, configure=False)
        if mp:
            if key not in mp.all_orm_descriptors:
                raise AttributeError(
                    "Class %r does not have a mapped column named %r"
                    % (self.cls, key)
                )

            desc = mp.all_orm_descriptors[key]
            if desc.extension_type is interfaces.NotExtension.NOT_EXTENSION:
                assert isinstance(desc, attributes.QueryableAttribute)
                prop = desc.property
                if isinstance(prop, SynonymProperty):
                    key = prop.name
                elif not isinstance(prop, ColumnProperty):
                    raise exc.InvalidRequestError(
                        "Property %r is not an instance of"
                        " ColumnProperty (i.e. does not correspond"
                        " directly to a Column)." % key
                    )
        return getattr(self.cls, key)


inspection._inspects(_GetColumns)(
    lambda target: inspection.inspect(target.cls)
)


class _GetTable:
    __slots__ = "key", "metadata"

    key: str
    metadata: MetaData

    def __init__(self, key: str, metadata: MetaData):
        self.key = key
        self.metadata = metadata

    def __getattr__(self, key: str) -> Table:
        return self.metadata.tables[_get_table_key(key, self.key)]


def _determine_container(key: str, value: Any) -> _GetColumns:
    if isinstance(value, _MultipleClassMarker):
        value = value.attempt_get([], key)
    return _GetColumns(value)


class _class_resolver:
    __slots__ = (
        "cls",
        "prop",
        "arg",
        "fallback",
        "_dict",
        "_resolvers",
        "favor_tables",
    )

    cls: Type[Any]
    prop: RelationshipProperty[Any]
    fallback: Mapping[str, Any]
    arg: str
    favor_tables: bool
    _resolvers: Tuple[Callable[[str], Any], ...]

    def __init__(
        self,
        cls: Type[Any],
        prop: RelationshipProperty[Any],
        fallback: Mapping[str, Any],
        arg: str,
        favor_tables: bool = False,
    ):
        self.cls = cls
        self.prop = prop
        self.arg = arg
        self.fallback = fallback
        self._dict = util.PopulateDict(self._access_cls)
        self._resolvers = ()
        self.favor_tables = favor_tables

    def _access_cls(self, key: str) -> Any:
        cls = self.cls

        manager = attributes.manager_of_class(cls)
        decl_base = manager.registry
        assert decl_base is not None
        decl_class_registry = decl_base._class_registry
        metadata = decl_base.metadata

        if self.favor_tables:
            if key in metadata.tables:
                return metadata.tables[key]
            elif key in metadata._schemas:
                return _GetTable(key, getattr(cls, "metadata", metadata))

        if key in decl_class_registry:
            return _determine_container(key, decl_class_registry[key])

        if not self.favor_tables:
            if key in metadata.tables:
                return metadata.tables[key]
            elif key in metadata._schemas:
                return _GetTable(key, getattr(cls, "metadata", metadata))

        if "_sa_module_registry" in decl_class_registry and key in cast(
            _ModuleMarker, decl_class_registry["_sa_module_registry"]
        ):
            registry = cast(
                _ModuleMarker, decl_class_registry["_sa_module_registry"]
            )
            return registry.resolve_attr(key)
        elif self._resolvers:
            for resolv in self._resolvers:
                value = resolv(key)
                if value is not None:
                    return value

        return self.fallback[key]

    def _raise_for_name(self, name: str, err: Exception) -> NoReturn:
        generic_match = re.match(r"(.+)\[(.+)\]", name)

        if generic_match:
            clsarg = generic_match.group(2).strip("'")
            raise exc.InvalidRequestError(
                f"When initializing mapper {self.prop.parent}, "
                f'expression "relationship({self.arg!r})" seems to be '
                "using a generic class as the argument to relationship(); "
                "please state the generic argument "
                "using an annotation, e.g. "
                f'"{self.prop.key}: Mapped[{generic_match.group(1)}'
                f"['{clsarg}']] = relationship()\""
            ) from err
        else:
            raise exc.InvalidRequestError(
                "When initializing mapper %s, expression %r failed to "
                "locate a name (%r). If this is a class name, consider "
                "adding this relationship() to the %r class after "
                "both dependent classes have been defined."
                % (self.prop.parent, self.arg, name, self.cls)
            ) from err

    def _resolve_name(self) -> Union[Table, Type[Any], _ModNS]:
        name = self.arg
        d = self._dict
        rval = None
        try:
            for token in name.split("."):
                if rval is None:
                    rval = d[token]
                else:
                    rval = getattr(rval, token)
        except KeyError as err:
            self._raise_for_name(name, err)
        except NameError as n:
            self._raise_for_name(n.args[0], n)
        else:
            if isinstance(rval, _GetColumns):
                return rval.cls
            else:
                if TYPE_CHECKING:
                    assert isinstance(rval, (type, Table, _ModNS))
                return rval

    def __call__(self) -> Any:
        try:
            x = eval(self.arg, globals(), self._dict)

            if isinstance(x, _GetColumns):
                return x.cls
            else:
                return x
        except NameError as n:
            self._raise_for_name(n.args[0], n)


_fallback_dict: Mapping[str, Any] = None  # type: ignore


def _resolver(cls: Type[Any], prop: RelationshipProperty[Any]) -> Tuple[
    Callable[[str], Callable[[], Union[Type[Any], Table, _ModNS]]],
    Callable[[str, bool], _class_resolver],
]:
    global _fallback_dict

    if _fallback_dict is None:
        import sqlalchemy
        from . import foreign
        from . import remote

        _fallback_dict = util.immutabledict(sqlalchemy.__dict__).union(
            {"foreign": foreign, "remote": remote}
        )

    def resolve_arg(arg: str, favor_tables: bool = False) -> _class_resolver:
        return _class_resolver(
            cls, prop, _fallback_dict, arg, favor_tables=favor_tables
        )

    def resolve_name(
        arg: str,
    ) -> Callable[[], Union[Type[Any], Table, _ModNS]]:
        return _class_resolver(cls, prop, _fallback_dict, arg)._resolve_name

    return resolve_name, resolve_arg

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\theanocode.py ===
"""
.. deprecated:: 1.8

  ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
  Aesara. Use ``sympy.printing.aesaracode`` instead. See
  :ref:`theanocode-deprecated` for more information.

"""
from __future__ import annotations
import math
from typing import Any

from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial

from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning


__doctest_requires__ = {('theano_function',): ['theano']}


theano = import_module('theano')


if theano:
    ts = theano.scalar
    tt = theano.tensor
    from theano.sandbox import linalg as tlinalg

    mapping = {
            sympy.Add: tt.add,
            sympy.Mul: tt.mul,
            sympy.Abs: tt.abs_,
            sympy.sign: tt.sgn,
            sympy.ceiling: tt.ceil,
            sympy.floor: tt.floor,
            sympy.log: tt.log,
            sympy.exp: tt.exp,
            sympy.sqrt: tt.sqrt,
            sympy.cos: tt.cos,
            sympy.acos: tt.arccos,
            sympy.sin: tt.sin,
            sympy.asin: tt.arcsin,
            sympy.tan: tt.tan,
            sympy.atan: tt.arctan,
            sympy.atan2: tt.arctan2,
            sympy.cosh: tt.cosh,
            sympy.acosh: tt.arccosh,
            sympy.sinh: tt.sinh,
            sympy.asinh: tt.arcsinh,
            sympy.tanh: tt.tanh,
            sympy.atanh: tt.arctanh,
            sympy.re: tt.real,
            sympy.im: tt.imag,
            sympy.arg: tt.angle,
            sympy.erf: tt.erf,
            sympy.gamma: tt.gamma,
            sympy.loggamma: tt.gammaln,
            sympy.Pow: tt.pow,
            sympy.Eq: tt.eq,
            sympy.StrictGreaterThan: tt.gt,
            sympy.StrictLessThan: tt.lt,
            sympy.LessThan: tt.le,
            sympy.GreaterThan: tt.ge,
            sympy.And: tt.and_,
            sympy.Or: tt.or_,
            sympy.Max: tt.maximum,  # SymPy accept >2 inputs, Theano only 2
            sympy.Min: tt.minimum,  # SymPy accept >2 inputs, Theano only 2
            sympy.conjugate: tt.conj,
            sympy.core.numbers.ImaginaryUnit: lambda:tt.complex(0,1),
            # Matrices
            sympy.MatAdd: tt.Elemwise(ts.add),
            sympy.HadamardProduct: tt.Elemwise(ts.mul),
            sympy.Trace: tlinalg.trace,
            sympy.Determinant : tlinalg.det,
            sympy.Inverse: tlinalg.matrix_inverse,
            sympy.Transpose: tt.DimShuffle((False, False), [1, 0]),
    }


class TheanoPrinter(Printer):
    """ Code printer which creates Theano symbolic expression graphs.

    Parameters
    ==========

    cache : dict
        Cache dictionary to use. If None (default) will use
        the global cache. To create a printer which does not depend on or alter
        global state pass an empty dictionary. Note: the dictionary is not
        copied on initialization of the printer and will be updated in-place,
        so using the same dict object when creating multiple printers or making
        multiple calls to :func:`.theano_code` or :func:`.theano_function` means
        the cache is shared between all these applications.

    Attributes
    ==========

    cache : dict
        A cache of Theano variables which have been created for SymPy
        symbol-like objects (e.g. :class:`sympy.core.symbol.Symbol` or
        :class:`sympy.matrices.expressions.MatrixSymbol`). This is used to
        ensure that all references to a given symbol in an expression (or
        multiple expressions) are printed as the same Theano variable, which is
        created only once. Symbols are differentiated only by name and type. The
        format of the cache's contents should be considered opaque to the user.
    """
    printmethod = "_theano"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', {})
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        """ Get the cache key for a SymPy object.

        Parameters
        ==========

        s : sympy.core.basic.Basic
            SymPy object to get key for.

        name : str
            Name of object, if it does not have a ``name`` attribute.
        """

        if name is None:
            name = s.name

        return (name, type(s), s.args, dtype, broadcastable)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Theano variable for a SymPy symbol from the cache, or create it
        if it does not exist.
        """

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # Integers already taken care of below, interpret as float
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        if not hasattr(tt, 'stacklists'):
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Theano")

        return tt.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = tt.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = 1
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = tt.dot(result, children[0])
        else:
            raise NotImplementedError('''Only non-negative integer
           powers of matrices can be handled by Theano at the moment''')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        return tt.join(0, *[tt.join(1, *row) for row in blocks])


    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return math.pi

    def _print_Exp1(self, expr, **kwargs):
        return ts.exp(1)

    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        e, cond = expr.args[0].args  # First condition and corresponding value

        # Print conditional expression and value for first condition
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # One condition only
        if len(expr.args) == 1:
            # Return value if condition else NaN
            return tt.switch(p_cond, p_e, np.nan)

        # Return value_1 if condition_1 else evaluate remaining conditions
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return tt.switch(p_cond, p_e, p_remaining)

    def _print_Rational(self, expr, **kwargs):
        return tt.true_div(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            rv = tt.Rop(rv, var, tt.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        return expr

    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a SymPy expression to a Theano graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Theano variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``theano.tensor.Tensor``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Theano.

        .. __: http://deeplearning.net/software/theano/tutorial/broadcasting.html

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict
            Mapping from SymPy symbols to Theano datatypes to use when creating
            new Theano variables for those symbols. Corresponds to the ``dtype``
            argument to ``theano.tensor.Tensor``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``theano.tensor.Tensor`` to use when creating Theano
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        theano.gof.graph.Variable
            A variable corresponding to the expression's value in a Theano
            symbolic expression graph.

        """
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}

        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)


global_cache: dict[Any, Any] = {}


def theano_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Theano graph variable.

    .. deprecated:: 1.8

      ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
      Aesara. Use ``sympy.printing.aesaracode`` instead. See
      :ref:`theanocode-deprecated` for more information.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Theano variables (see :class:`TheanoPrinter.cache
        <TheanoPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    Returns
    =======

    theano.gof.graph.Variable
        A variable corresponding to the expression's value in a Theano symbolic
        expression graph.

    """
    sympy_deprecation_warning(
        """
        sympy.printing.theanocode is deprecated. Theano has been renamed to
        Aesara. Use sympy.printing.aesaracode instead.""",
        deprecated_since_version="1.8",
    active_deprecations_target='theanocode-deprecated')

    if not theano:
        raise ImportError("theano is required for theano_code")

    if cache is None:
        cache = global_cache

    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    r"""
    Get value of ``broadcastables`` argument to :func:`.theano_code` from
    keyword arguments to :func:`.theano_function`.

    Included for backwards compatibility.

    Parameters
    ==========

    inputs
        Sequence of input symbols.

    dim : int
        Common number of dimensions for all inputs. Overrides other arguments
        if given.

    dims : dict
        Mapping from input symbols to number of dimensions. Overrides
        ``broadcastables`` argument if given.

    broadcastables : dict
        Explicit value of ``broadcastables`` argument to
        :meth:`.TheanoPrinter.doprint`. If not None function will return this value unchanged.

    Returns
    =======
    dict
        Dictionary mapping elements of ``inputs`` to their "broadcastable"
        values (tuple of ``bool``\ s).
    """
    if dim is not None:
        return dict.fromkeys(inputs, (False,) * dim)

    if dims is not None:
        maxdim = max(dims.values())
        return {
            s: (False,) * d + (True,) * (maxdim - d)
            for s, d in dims.items()
        }

    if broadcastables is not None:
        return broadcastables

    return {}


@doctest_depends_on(modules=('theano',))
def theano_function(inputs, outputs, scalar=False, *,
        dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Theano function from SymPy expressions.

    .. deprecated:: 1.8

      ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
      Aesara. Use ``sympy.printing.aesaracode`` instead. See
      :ref:`theanocode-deprecated` for more information.

    The inputs and outputs are converted to Theano variables using
    :func:`.theano_code` and then passed to ``theano.function``.

    Parameters
    ==========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Sequence of expressions which constitute the outputs(s) of the
        function. The free symbols of each expression must be a subset of
        ``inputs``.

    scalar : bool
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Theano function object.

    cache : dict
        Cached Theano variables (see :class:`TheanoPrinter.cache
        <TheanoPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.

    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``theano_function([x, y], [...], dim=2)`` is equivalent to using
        ``broadcastables={x: (False, False), y: (False, False)}``.

    Returns
    =======
    callable
        A callable object which takes values of ``inputs`` as positional
        arguments and returns an output array for each of the expressions
        in ``outputs``. If ``outputs`` is a single expression the function will
        return a Numpy array, if it is a list of multiple expressions the
        function will return a list of arrays. See description of the ``squeeze``
        argument above for the behavior when a single output is passed in a list.
        The returned object will either be an instance of
        ``theano.compile.function_module.Function`` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``theano_function`` attribute which points to the return value of
        ``theano.function``.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.printing.theanocode import theano_function

    A simple function with one input and one output:

    >>> f1 = theano_function([x], [x**2 - 1], scalar=True)
    >>> f1(3)
    8.0

    A function with multiple inputs and one output:

    >>> f2 = theano_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
    >>> f2(3, 4, 2)
    5.0

    A function with multiple inputs and multiple outputs:

    >>> f3 = theano_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    >>> f3(2, 3)
    [13.0, -5.0]

    See also
    ========

    dim_handling

    """
    sympy_deprecation_warning(
        """
        sympy.printing.theanocode is deprecated. Theano has been renamed to Aesara. Use sympy.printing.aesaracode instead""",
        deprecated_since_version="1.8",
    active_deprecations_target='theanocode-deprecated')

    if not theano:
        raise ImportError("theano is required for theano_function")

    # Pop off non-theano keyword args
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})

    broadcastables = dim_handling(
        inputs, dim=dim, dims=dims, broadcastables=broadcastables,
    )

    # Print inputs/outputs
    code = partial(theano_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))

    #fix constant expressions as variables
    toutputs = [output if isinstance(output, theano.Variable) else tt.as_tensor_variable(output) for output in toutputs]

    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # Compile theano func
    func = theano.function(tinputs, toutputs, **kwargs)

    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]

    # No wrapper required
    if not scalar or not any(is_0d):
        func.theano_function = func
        return func

    # Create wrapper to convert 0-dimensional outputs to scalars
    def wrapper(*args):
        out = func(*args)
        # out can be array(1.0) or [array(1.0), array(2.0)]

        if is_sequence(out):
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]
        else:
            return out[()]

    wrapper.__wrapped__ = func
    wrapper.__doc__ = func.__doc__
    wrapper.theano_function = func
    return wrapper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\baked.py ===
# ext/baked.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""Baked query extension.

Provides a creational pattern for the :class:`.query.Query` object which
allows the fully constructed object, Core select statement, and string
compiled result to be fully cached.


"""

import collections.abc as collections_abc
import logging

from .. import exc as sa_exc
from .. import util
from ..orm import exc as orm_exc
from ..orm.query import Query
from ..orm.session import Session
from ..sql import func
from ..sql import literal_column
from ..sql import util as sql_util


log = logging.getLogger(__name__)


class Bakery:
    """Callable which returns a :class:`.BakedQuery`.

    This object is returned by the class method
    :meth:`.BakedQuery.bakery`.  It exists as an object
    so that the "cache" can be easily inspected.

    .. versionadded:: 1.2


    """

    __slots__ = "cls", "cache"

    def __init__(self, cls_, cache):
        self.cls = cls_
        self.cache = cache

    def __call__(self, initial_fn, *args):
        return self.cls(self.cache, initial_fn, args)


class BakedQuery:
    """A builder object for :class:`.query.Query` objects."""

    __slots__ = "steps", "_bakery", "_cache_key", "_spoiled"

    def __init__(self, bakery, initial_fn, args=()):
        self._cache_key = ()
        self._update_cache_key(initial_fn, args)
        self.steps = [initial_fn]
        self._spoiled = False
        self._bakery = bakery

    @classmethod
    def bakery(cls, size=200, _size_alert=None):
        """Construct a new bakery.

        :return: an instance of :class:`.Bakery`

        """

        return Bakery(cls, util.LRUCache(size, size_alert=_size_alert))

    def _clone(self):
        b1 = BakedQuery.__new__(BakedQuery)
        b1._cache_key = self._cache_key
        b1.steps = list(self.steps)
        b1._bakery = self._bakery
        b1._spoiled = self._spoiled
        return b1

    def _update_cache_key(self, fn, args=()):
        self._cache_key += (fn.__code__,) + args

    def __iadd__(self, other):
        if isinstance(other, tuple):
            self.add_criteria(*other)
        else:
            self.add_criteria(other)
        return self

    def __add__(self, other):
        if isinstance(other, tuple):
            return self.with_criteria(*other)
        else:
            return self.with_criteria(other)

    def add_criteria(self, fn, *args):
        """Add a criteria function to this :class:`.BakedQuery`.

        This is equivalent to using the ``+=`` operator to
        modify a :class:`.BakedQuery` in-place.

        """
        self._update_cache_key(fn, args)
        self.steps.append(fn)
        return self

    def with_criteria(self, fn, *args):
        """Add a criteria function to a :class:`.BakedQuery` cloned from this
        one.

        This is equivalent to using the ``+`` operator to
        produce a new :class:`.BakedQuery` with modifications.

        """
        return self._clone().add_criteria(fn, *args)

    def for_session(self, session):
        """Return a :class:`_baked.Result` object for this
        :class:`.BakedQuery`.

        This is equivalent to calling the :class:`.BakedQuery` as a
        Python callable, e.g. ``result = my_baked_query(session)``.

        """
        return Result(self, session)

    def __call__(self, session):
        return self.for_session(session)

    def spoil(self, full=False):
        """Cancel any query caching that will occur on this BakedQuery object.

        The BakedQuery can continue to be used normally, however additional
        creational functions will not be cached; they will be called
        on every invocation.

        This is to support the case where a particular step in constructing
        a baked query disqualifies the query from being cacheable, such
        as a variant that relies upon some uncacheable value.

        :param full: if False, only functions added to this
         :class:`.BakedQuery` object subsequent to the spoil step will be
         non-cached; the state of the :class:`.BakedQuery` up until
         this point will be pulled from the cache.   If True, then the
         entire :class:`_query.Query` object is built from scratch each
         time, with all creational functions being called on each
         invocation.

        """
        if not full and not self._spoiled:
            _spoil_point = self._clone()
            _spoil_point._cache_key += ("_query_only",)
            self.steps = [_spoil_point._retrieve_baked_query]
        self._spoiled = True
        return self

    def _effective_key(self, session):
        """Return the key that actually goes into the cache dictionary for
        this :class:`.BakedQuery`, taking into account the given
        :class:`.Session`.

        This basically means we also will include the session's query_class,
        as the actual :class:`_query.Query` object is part of what's cached
        and needs to match the type of :class:`_query.Query` that a later
        session will want to use.

        """
        return self._cache_key + (session._query_cls,)

    def _with_lazyload_options(self, options, effective_path, cache_path=None):
        """Cloning version of _add_lazyload_options."""
        q = self._clone()
        q._add_lazyload_options(options, effective_path, cache_path=cache_path)
        return q

    def _add_lazyload_options(self, options, effective_path, cache_path=None):
        """Used by per-state lazy loaders to add options to the
        "lazy load" query from a parent query.

        Creates a cache key based on given load path and query options;
        if a repeatable cache key cannot be generated, the query is
        "spoiled" so that it won't use caching.

        """

        key = ()

        if not cache_path:
            cache_path = effective_path

        for opt in options:
            if opt._is_legacy_option or opt._is_compile_state:
                ck = opt._generate_cache_key()
                if ck is None:
                    self.spoil(full=True)
                else:
                    assert not ck[1], (
                        "loader options with variable bound parameters "
                        "not supported with baked queries.  Please "
                        "use new-style select() statements for cached "
                        "ORM queries."
                    )
                    key += ck[0]

        self.add_criteria(
            lambda q: q._with_current_path(effective_path).options(*options),
            cache_path.path,
            key,
        )

    def _retrieve_baked_query(self, session):
        query = self._bakery.get(self._effective_key(session), None)
        if query is None:
            query = self._as_query(session)
            self._bakery[self._effective_key(session)] = query.with_session(
                None
            )
        return query.with_session(session)

    def _bake(self, session):
        query = self._as_query(session)
        query.session = None

        # in 1.4, this is where before_compile() event is
        # invoked
        statement = query._statement_20()

        # if the query is not safe to cache, we still do everything as though
        # we did cache it, since the receiver of _bake() assumes subqueryload
        # context was set up, etc.
        #
        # note also we want to cache the statement itself because this
        # allows the statement itself to hold onto its cache key that is
        # used by the Connection, which in itself is more expensive to
        # generate than what BakedQuery was able to provide in 1.3 and prior

        if statement._compile_options._bake_ok:
            self._bakery[self._effective_key(session)] = (
                query,
                statement,
            )

        return query, statement

    def to_query(self, query_or_session):
        """Return the :class:`_query.Query` object for use as a subquery.

        This method should be used within the lambda callable being used
        to generate a step of an enclosing :class:`.BakedQuery`.   The
        parameter should normally be the :class:`_query.Query` object that
        is passed to the lambda::

            sub_bq = self.bakery(lambda s: s.query(User.name))
            sub_bq += lambda q: q.filter(User.id == Address.user_id).correlate(Address)

            main_bq = self.bakery(lambda s: s.query(Address))
            main_bq += lambda q: q.filter(sub_bq.to_query(q).exists())

        In the case where the subquery is used in the first callable against
        a :class:`.Session`, the :class:`.Session` is also accepted::

            sub_bq = self.bakery(lambda s: s.query(User.name))
            sub_bq += lambda q: q.filter(User.id == Address.user_id).correlate(Address)

            main_bq = self.bakery(
                lambda s: s.query(Address.id, sub_bq.to_query(q).scalar_subquery())
            )

        :param query_or_session: a :class:`_query.Query` object or a class
         :class:`.Session` object, that is assumed to be within the context
         of an enclosing :class:`.BakedQuery` callable.


         .. versionadded:: 1.3


        """  # noqa: E501

        if isinstance(query_or_session, Session):
            session = query_or_session
        elif isinstance(query_or_session, Query):
            session = query_or_session.session
            if session is None:
                raise sa_exc.ArgumentError(
                    "Given Query needs to be associated with a Session"
                )
        else:
            raise TypeError(
                "Query or Session object expected, got %r."
                % type(query_or_session)
            )
        return self._as_query(session)

    def _as_query(self, session):
        query = self.steps[0](session)

        for step in self.steps[1:]:
            query = step(query)

        return query


class Result:
    """Invokes a :class:`.BakedQuery` against a :class:`.Session`.

    The :class:`_baked.Result` object is where the actual :class:`.query.Query`
    object gets created, or retrieved from the cache,
    against a target :class:`.Session`, and is then invoked for results.

    """

    __slots__ = "bq", "session", "_params", "_post_criteria"

    def __init__(self, bq, session):
        self.bq = bq
        self.session = session
        self._params = {}
        self._post_criteria = []

    def params(self, *args, **kw):
        """Specify parameters to be replaced into the string SQL statement."""

        if len(args) == 1:
            kw.update(args[0])
        elif len(args) > 0:
            raise sa_exc.ArgumentError(
                "params() takes zero or one positional argument, "
                "which is a dictionary."
            )
        self._params.update(kw)
        return self

    def _using_post_criteria(self, fns):
        if fns:
            self._post_criteria.extend(fns)
        return self

    def with_post_criteria(self, fn):
        """Add a criteria function that will be applied post-cache.

        This adds a function that will be run against the
        :class:`_query.Query` object after it is retrieved from the
        cache.    This currently includes **only** the
        :meth:`_query.Query.params` and :meth:`_query.Query.execution_options`
        methods.

        .. warning::  :meth:`_baked.Result.with_post_criteria`
           functions are applied
           to the :class:`_query.Query`
           object **after** the query's SQL statement
           object has been retrieved from the cache.   Only
           :meth:`_query.Query.params` and
           :meth:`_query.Query.execution_options`
           methods should be used.


        .. versionadded:: 1.2


        """
        return self._using_post_criteria([fn])

    def _as_query(self):
        q = self.bq._as_query(self.session).params(self._params)
        for fn in self._post_criteria:
            q = fn(q)
        return q

    def __str__(self):
        return str(self._as_query())

    def __iter__(self):
        return self._iter().__iter__()

    def _iter(self):
        bq = self.bq

        if not self.session.enable_baked_queries or bq._spoiled:
            return self._as_query()._iter()

        query, statement = bq._bakery.get(
            bq._effective_key(self.session), (None, None)
        )
        if query is None:
            query, statement = bq._bake(self.session)

        if self._params:
            q = query.params(self._params)
        else:
            q = query
        for fn in self._post_criteria:
            q = fn(q)

        params = q._params
        execution_options = dict(q._execution_options)
        execution_options.update(
            {
                "_sa_orm_load_options": q.load_options,
                "compiled_cache": bq._bakery,
            }
        )

        result = self.session.execute(
            statement, params, execution_options=execution_options
        )
        if result._attributes.get("is_single_entity", False):
            result = result.scalars()

        if result._attributes.get("filtered", False):
            result = result.unique()

        return result

    def count(self):
        """return the 'count'.

        Equivalent to :meth:`_query.Query.count`.

        Note this uses a subquery to ensure an accurate count regardless
        of the structure of the original statement.

        """

        col = func.count(literal_column("*"))
        bq = self.bq.with_criteria(lambda q: q._legacy_from_self(col))
        return bq.for_session(self.session).params(self._params).scalar()

    def scalar(self):
        """Return the first element of the first result or None
        if no rows present.  If multiple rows are returned,
        raises MultipleResultsFound.

        Equivalent to :meth:`_query.Query.scalar`.

        """
        try:
            ret = self.one()
            if not isinstance(ret, collections_abc.Sequence):
                return ret
            return ret[0]
        except orm_exc.NoResultFound:
            return None

    def first(self):
        """Return the first row.

        Equivalent to :meth:`_query.Query.first`.

        """

        bq = self.bq.with_criteria(lambda q: q.slice(0, 1))
        return (
            bq.for_session(self.session)
            .params(self._params)
            ._using_post_criteria(self._post_criteria)
            ._iter()
            .first()
        )

    def one(self):
        """Return exactly one result or raise an exception.

        Equivalent to :meth:`_query.Query.one`.

        """
        return self._iter().one()

    def one_or_none(self):
        """Return one or zero results, or raise an exception for multiple
        rows.

        Equivalent to :meth:`_query.Query.one_or_none`.

        """
        return self._iter().one_or_none()

    def all(self):
        """Return all rows.

        Equivalent to :meth:`_query.Query.all`.

        """
        return self._iter().all()

    def get(self, ident):
        """Retrieve an object based on identity.

        Equivalent to :meth:`_query.Query.get`.

        """

        query = self.bq.steps[0](self.session)
        return query._get_impl(ident, self._load_on_pk_identity)

    def _load_on_pk_identity(self, session, query, primary_key_identity, **kw):
        """Load the given primary key identity from the database."""

        mapper = query._raw_columns[0]._annotations["parententity"]

        _get_clause, _get_params = mapper._get_clause

        def setup(query):
            _lcl_get_clause = _get_clause
            q = query._clone()
            q._get_condition()
            q._order_by = None

            # None present in ident - turn those comparisons
            # into "IS NULL"
            if None in primary_key_identity:
                nones = {
                    _get_params[col].key
                    for col, value in zip(
                        mapper.primary_key, primary_key_identity
                    )
                    if value is None
                }
                _lcl_get_clause = sql_util.adapt_criterion_to_null(
                    _lcl_get_clause, nones
                )

            # TODO: can mapper._get_clause be pre-adapted?
            q._where_criteria = (
                sql_util._deep_annotate(_lcl_get_clause, {"_orm_adapt": True}),
            )

            for fn in self._post_criteria:
                q = fn(q)
            return q

        # cache the query against a key that includes
        # which positions in the primary key are NULL
        # (remember, we can map to an OUTER JOIN)
        bq = self.bq

        # add the clause we got from mapper._get_clause to the cache
        # key so that if a race causes multiple calls to _get_clause,
        # we've cached on ours
        bq = bq._clone()
        bq._cache_key += (_get_clause,)

        bq = bq.with_criteria(
            setup, tuple(elem is None for elem in primary_key_identity)
        )

        params = {
            _get_params[primary_key].key: id_val
            for id_val, primary_key in zip(
                primary_key_identity, mapper.primary_key
            )
        }

        result = list(bq.for_session(self.session).params(**params))
        l = len(result)
        if l > 1:
            raise orm_exc.MultipleResultsFound()
        elif l:
            return result[0]
        else:
            return None


bakery = BakedQuery.bakery

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\crv.py ===
"""
Continuous Random Variables Module

See Also
========
sympy.stats.crv_types
sympy.stats.rv
sympy.stats.frv
"""


from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda, PoleError
from sympy.core.numbers import (I, nan, oo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Union)
from sympy.solvers.solveset import solveset
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats.rv import (RandomDomain, SingleDomain, ConditionalDomain, is_random,
        ProductDomain, PSpace, SinglePSpace, random_symbols, NamedArgsMixin, Distribution)


class ContinuousDomain(RandomDomain):
    """
    A domain with continuous support

    Represented using symbols and Intervals.
    """
    is_Continuous = True

    def as_boolean(self):
        raise NotImplementedError("Not Implemented for generic Domains")


class SingleContinuousDomain(ContinuousDomain, SingleDomain):
    """
    A univariate domain with continuous support

    Represented using a single symbol and interval.
    """
    def compute_expectation(self, expr, variables=None, **kwargs):
        if variables is None:
            variables = self.symbols
        if not variables:
            return expr
        if frozenset(variables) != frozenset(self.symbols):
            raise ValueError("Values should be equal")
        # assumes only intervals
        return Integral(expr, (self.symbol, self.set), **kwargs)

    def as_boolean(self):
        return self.set.as_relational(self.symbol)


class ProductContinuousDomain(ProductDomain, ContinuousDomain):
    """
    A collection of independent domains with continuous support
    """

    def compute_expectation(self, expr, variables=None, **kwargs):
        if variables is None:
            variables = self.symbols
        for domain in self.domains:
            domain_vars = frozenset(variables) & frozenset(domain.symbols)
            if domain_vars:
                expr = domain.compute_expectation(expr, domain_vars, **kwargs)
        return expr

    def as_boolean(self):
        return And(*[domain.as_boolean() for domain in self.domains])


class ConditionalContinuousDomain(ContinuousDomain, ConditionalDomain):
    """
    A domain with continuous support that has been further restricted by a
    condition such as $x > 3$.
    """

    def compute_expectation(self, expr, variables=None, **kwargs):
        if variables is None:
            variables = self.symbols
        if not variables:
            return expr
        # Extract the full integral
        fullintgrl = self.fulldomain.compute_expectation(expr, variables)
        # separate into integrand and limits
        integrand, limits = fullintgrl.function, list(fullintgrl.limits)

        conditions = [self.condition]
        while conditions:
            cond = conditions.pop()
            if cond.is_Boolean:
                if isinstance(cond, And):
                    conditions.extend(cond.args)
                elif isinstance(cond, Or):
                    raise NotImplementedError("Or not implemented here")
            elif cond.is_Relational:
                if cond.is_Equality:
                    # Add the appropriate Delta to the integrand
                    integrand *= DiracDelta(cond.lhs - cond.rhs)
                else:
                    symbols = cond.free_symbols & set(self.symbols)
                    if len(symbols) != 1:  # Can't handle x > y
                        raise NotImplementedError(
                            "Multivariate Inequalities not yet implemented")
                    # Can handle x > 0
                    symbol = symbols.pop()
                    # Find the limit with x, such as (x, -oo, oo)
                    for i, limit in enumerate(limits):
                        if limit[0] == symbol:
                            # Make condition into an Interval like [0, oo]
                            cintvl = reduce_rational_inequalities_wrap(
                                cond, symbol)
                            # Make limit into an Interval like [-oo, oo]
                            lintvl = Interval(limit[1], limit[2])
                            # Intersect them to get [0, oo]
                            intvl = cintvl.intersect(lintvl)
                            # Put back into limits list
                            limits[i] = (symbol, intvl.left, intvl.right)
            else:
                raise TypeError(
                    "Condition %s is not a relational or Boolean" % cond)

        return Integral(integrand, *limits, **kwargs)

    def as_boolean(self):
        return And(self.fulldomain.as_boolean(), self.condition)

    @property
    def set(self):
        if len(self.symbols) == 1:
            return (self.fulldomain.set & reduce_rational_inequalities_wrap(
                self.condition, tuple(self.symbols)[0]))
        else:
            raise NotImplementedError(
                "Set of Conditional Domain not Implemented")


class ContinuousDistribution(Distribution):
    def __call__(self, *args):
        return self.pdf(*args)


class SingleContinuousDistribution(ContinuousDistribution, NamedArgsMixin):
    """ Continuous distribution of a single variable.

    Explanation
    ===========

    Serves as superclass for Normal/Exponential/UniformDistribution etc....

    Represented by parameters for each of the specific classes.  E.g
    NormalDistribution is represented by a mean and standard deviation.

    Provides methods for pdf, cdf, and sampling.

    See Also
    ========

    sympy.stats.crv_types.*
    """

    set = Interval(-oo, oo)

    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    @cacheit
    def compute_cdf(self, **kwargs):
        """ Compute the CDF from the PDF.

        Returns a Lambda.
        """
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.set.start

        # CDF is integral of PDF from left bound to z
        pdf = self.pdf(x)
        cdf = integrate(pdf.doit(), (x, left_bound, z), **kwargs)
        # CDF Ensure that CDF left of left_bound is zero
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    def _cdf(self, x):
        return None

    def cdf(self, x, **kwargs):
        """ Cumulative density function """
        if len(kwargs) == 0:
            cdf = self._cdf(x)
            if cdf is not None:
                return cdf
        return self.compute_cdf(**kwargs)(x)

    @cacheit
    def compute_characteristic_function(self, **kwargs):
        """ Compute the characteristic function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        cf = integrate(exp(I*t*x)*pdf, (x, self.set))
        return Lambda(t, cf)

    def _characteristic_function(self, t):
        return None

    def characteristic_function(self, t, **kwargs):
        """ Characteristic function """
        if len(kwargs) == 0:
            cf = self._characteristic_function(t)
            if cf is not None:
                return cf
        return self.compute_characteristic_function(**kwargs)(t)

    @cacheit
    def compute_moment_generating_function(self, **kwargs):
        """ Compute the moment generating function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        mgf = integrate(exp(t * x) * pdf, (x, self.set))
        return Lambda(t, mgf)

    def _moment_generating_function(self, t):
        return None

    def moment_generating_function(self, t, **kwargs):
        """ Moment generating function """
        if not kwargs:
                mgf = self._moment_generating_function(t)
                if mgf is not None:
                    return mgf
        return self.compute_moment_generating_function(**kwargs)(t)

    def expectation(self, expr, var, evaluate=True, **kwargs):
        """ Expectation of expression over distribution """
        if evaluate:
            try:
                p = poly(expr, var)
                if p.is_zero:
                    return S.Zero
                t = Dummy('t', real=True)
                mgf = self._moment_generating_function(t)
                if mgf is None:
                    return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
                deg = p.degree()
                taylor = poly(series(mgf, t, 0, deg + 1).removeO(), t)
                result = 0
                for k in range(deg+1):
                    result += p.coeff_monomial(var ** k) * taylor.coeff_monomial(t ** k) * factorial(k)
                return result
            except PolynomialError:
                return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
        else:
            return Integral(expr * self.pdf(var), (var, self.set), **kwargs)

    @cacheit
    def compute_quantile(self, **kwargs):
        """ Compute the Quantile from the PDF.

        Returns a Lambda.
        """
        x, p = symbols('x, p', real=True, cls=Dummy)
        left_bound = self.set.start

        pdf = self.pdf(x)
        cdf = integrate(pdf, (x, left_bound, x), **kwargs)
        quantile = solveset(cdf - p, x, self.set)
        return Lambda(p, Piecewise((quantile, (p >= 0) & (p <= 1) ), (nan, True)))

    def _quantile(self, x):
        return None

    def quantile(self, x, **kwargs):
        """ Cumulative density function """
        if len(kwargs) == 0:
            quantile = self._quantile(x)
            if quantile is not None:
                return quantile
        return self.compute_quantile(**kwargs)(x)


class ContinuousPSpace(PSpace):
    """ Continuous Probability Space

    Represents the likelihood of an event space defined over a continuum.

    Represented with a ContinuousDomain and a PDF (Lambda-Like)
    """

    is_Continuous = True
    is_real = True

    @property
    def pdf(self):
        return self.density(*self.domain.symbols)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        if rvs is None:
            rvs = self.values
        else:
            rvs = frozenset(rvs)

        expr = expr.xreplace({rv: rv.symbol for rv in rvs})

        domain_symbols = frozenset(rv.symbol for rv in rvs)

        return self.domain.compute_expectation(self.pdf * expr,
                domain_symbols, **kwargs)

    def compute_density(self, expr, **kwargs):
        # Common case Density(X) where X in self.values
        if expr in self.values:
            # Marginalize all other random symbols out of the density
            randomsymbols = tuple(set(self.values) - frozenset([expr]))
            symbols = tuple(rs.symbol for rs in randomsymbols)
            pdf = self.domain.compute_expectation(self.pdf, symbols, **kwargs)
            return Lambda(expr.symbol, pdf)

        z = Dummy('z', real=True)
        return Lambda(z, self.compute_expectation(DiracDelta(expr - z), **kwargs))

    @cacheit
    def compute_cdf(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise ValueError(
                "CDF not well defined on multivariate expressions")

        d = self.compute_density(expr, **kwargs)
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.domain.set.start

        # CDF is integral of PDF from left bound to z
        cdf = integrate(d(x), (x, left_bound, z), **kwargs)
        # CDF Ensure that CDF left of left_bound is zero
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    @cacheit
    def compute_characteristic_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError("Characteristic function of multivariate expressions not implemented")

        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        cf = integrate(exp(I*t*x)*d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, cf)

    @cacheit
    def compute_moment_generating_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError("Moment generating function of multivariate expressions not implemented")

        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        mgf = integrate(exp(t * x) * d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, mgf)

    @cacheit
    def compute_quantile(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise ValueError(
                "Quantile not well defined on multivariate expressions")

        d = self.compute_cdf(expr, **kwargs)
        x = Dummy('x', real=True)
        p = Dummy('p', positive=True)

        quantile = solveset(d(x) - p, x, self.set)

        return Lambda(p, quantile)

    def probability(self, condition, **kwargs):
        z = Dummy('z', real=True)
        cond_inv = False
        if isinstance(condition, Ne):
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        # Univariate case can be handled by where
        try:
            domain = self.where(condition)
            rv = [rv for rv in self.values if rv.symbol == domain.symbol][0]
            # Integrate out all other random variables
            pdf = self.compute_density(rv, **kwargs)
            # return S.Zero if `domain` is empty set
            if domain.set is S.EmptySet or isinstance(domain.set, FiniteSet):
                return S.Zero if not cond_inv else S.One
            if isinstance(domain.set, Union):
                return sum(
                     Integral(pdf(z), (z, subset), **kwargs) for subset in
                     domain.set.args if isinstance(subset, Interval))
            # Integrate out the last variable over the special domain
            return Integral(pdf(z), (z, domain.set), **kwargs)

        # Other cases can be turned into univariate case
        # by computing a density handled by density computation
        except NotImplementedError:
            from sympy.stats.rv import density
            expr = condition.lhs - condition.rhs
            if not is_random(expr):
                dens = self.density
                comp = condition.rhs
            else:
                dens = density(expr, **kwargs)
                comp = 0
            if not isinstance(dens, ContinuousDistribution):
                from sympy.stats.crv_types import ContinuousDistributionHandmade
                dens = ContinuousDistributionHandmade(dens, set=self.domain.set)
            # Turn problem into univariate case
            space = SingleContinuousPSpace(z, dens)
            result = space.probability(condition.__class__(space.value, comp))
            return result if not cond_inv else S.One - result

    def where(self, condition):
        rvs = frozenset(random_symbols(condition))
        if not (len(rvs) == 1 and rvs.issubset(self.values)):
            raise NotImplementedError(
                "Multiple continuous random variables not supported")
        rv = tuple(rvs)[0]
        interval = reduce_rational_inequalities_wrap(condition, rv)
        interval = interval.intersect(self.domain.set)
        return SingleContinuousDomain(rv.symbol, interval)

    def conditional_space(self, condition, normalize=True, **kwargs):
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        domain = ConditionalContinuousDomain(self.domain, condition)
        if normalize:
            # create a clone of the variable to
            # make sure that variables in nested integrals are different
            # from the variables outside the integral
            # this makes sure that they are evaluated separately
            # and in the correct order
            replacement  = {rv: Dummy(str(rv)) for rv in self.symbols}
            norm = domain.compute_expectation(self.pdf, **kwargs)
            pdf = self.pdf / norm.xreplace(replacement)
            # XXX: Converting set to tuple. The order matters to Lambda though
            # so we shouldn't be starting with a set here...
            density = Lambda(tuple(domain.symbols), pdf)

        return ContinuousPSpace(domain, density)


class SingleContinuousPSpace(ContinuousPSpace, SinglePSpace):
    """
    A continuous probability space over a single univariate variable.

    These consist of a Symbol and a SingleContinuousDistribution

    This class is normally accessed through the various random variable
    functions, Normal, Exponential, Uniform, etc....
    """

    @property
    def set(self):
        return self.distribution.set

    @property
    def domain(self):
        return SingleContinuousDomain(sympify(self.symbol), self.set)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method.

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        rvs = rvs or (self.value,)
        if self.value not in rvs:
            return expr

        expr = _sympify(expr)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})

        x = self.value.symbol
        try:
            return self.distribution.expectation(expr, x, evaluate=evaluate, **kwargs)
        except PoleError:
            return Integral(expr * self.pdf, (x, self.set), **kwargs)

    def compute_cdf(self, expr, **kwargs):
        if expr == self.value:
            z = Dummy("z", real=True)
            return Lambda(z, self.distribution.cdf(z, **kwargs))
        else:
            return ContinuousPSpace.compute_cdf(self, expr, **kwargs)

    def compute_characteristic_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.characteristic_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_characteristic_function(self, expr, **kwargs)

    def compute_moment_generating_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.moment_generating_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_moment_generating_function(self, expr, **kwargs)

    def compute_density(self, expr, **kwargs):
        # https://en.wikipedia.org/wiki/Random_variable#Functions_of_random_variables
        if expr == self.value:
            return self.density
        y = Dummy('y', real=True)

        gs = solveset(expr - y, self.value, S.Reals)

        if isinstance(gs, Intersection):
            if len(gs.args) == 2 and gs.args[0] is S.Reals:
                gs = gs.args[1]
        if not gs.is_FiniteSet:
            raise ValueError("Can not solve %s for %s" % (expr, self.value))
        fx = self.compute_density(self.value)
        fy = sum(fx(g) * abs(g.diff(y)) for g in gs)
        return Lambda(y, fy)

    def compute_quantile(self, expr, **kwargs):

        if expr == self.value:
            p = Dummy("p", real=True)
            return Lambda(p, self.distribution.quantile(p, **kwargs))
        else:
            return ContinuousPSpace.compute_quantile(self, expr, **kwargs)

def _reduce_inequalities(conditions, var, **kwargs):
    try:
        return reduce_rational_inequalities(conditions, var, **kwargs)
    except PolynomialError:
        raise ValueError("Reduction of condition failed %s\n" % conditions[0])


def reduce_rational_inequalities_wrap(condition, var):
    if condition.is_Relational:
        return _reduce_inequalities([[condition]], var, relational=False)
    if isinstance(condition, Or):
        return Union(*[_reduce_inequalities([[arg]], var, relational=False)
            for arg in condition.args])
    if isinstance(condition, And):
        intervals = [_reduce_inequalities([[arg]], var, relational=False)
            for arg in condition.args]
        I = intervals[0]
        for i in intervals:
            I = I.intersect(i)
        return I

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\_check_streams.py ===
# Generic stream tests
from __future__ import annotations

import random
import sys
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager, suppress
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
)

from .. import CancelScope, _core
from .._abc import AsyncResource, HalfCloseableStream, ReceiveStream, SendStream, Stream
from .._highlevel_generic import aclose_forcefully
from ._checkpoints import assert_checkpoints

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import ParamSpec, TypeAlias

    ArgsT = ParamSpec("ArgsT")

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

Res1 = TypeVar("Res1", bound=AsyncResource)
Res2 = TypeVar("Res2", bound=AsyncResource)
StreamMaker: TypeAlias = Callable[[], Awaitable[tuple[Res1, Res2]]]


class _ForceCloseBoth(Generic[Res1, Res2]):
    def __init__(self, both: tuple[Res1, Res2]) -> None:
        self._first, self._second = both

    async def __aenter__(self) -> tuple[Res1, Res2]:
        return self._first, self._second

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await aclose_forcefully(self._first)
        finally:
            await aclose_forcefully(self._second)


# This is used in this file instead of pytest.raises in order to avoid a dependency
# on pytest, as the check_* functions are publicly exported.
@contextmanager
def _assert_raises(
    expected_exc: type[BaseException],
    wrapped: bool = False,
) -> Generator[None, None, None]:
    __tracebackhide__ = True
    try:
        yield
    except BaseExceptionGroup as exc:
        assert wrapped, "caught exceptiongroup, but expected an unwrapped exception"
        # assert in except block ignored below
        assert len(exc.exceptions) == 1  # noqa: PT017
        assert isinstance(exc.exceptions[0], expected_exc)  # noqa: PT017
    except expected_exc:
        assert not wrapped, "caught exception, but expected an exceptiongroup"
    else:
        raise AssertionError(f"expected exception: {expected_exc}")


async def check_one_way_stream(
    stream_maker: StreamMaker[SendStream, ReceiveStream],
    clogged_stream_maker: StreamMaker[SendStream, ReceiveStream] | None,
) -> None:
    """Perform a number of generic tests on a custom one-way stream
    implementation.

    Args:
      stream_maker: An async (!) function which returns a connected
          (:class:`~trio.abc.SendStream`, :class:`~trio.abc.ReceiveStream`)
          pair.
      clogged_stream_maker: Either None, or an async function similar to
          stream_maker, but with the extra property that the returned stream
          is in a state where ``send_all`` and
          ``wait_send_all_might_not_block`` will block until ``receive_some``
          has been called. This allows for more thorough testing of some edge
          cases, especially around ``wait_send_all_might_not_block``.

    Raises:
      AssertionError: if a test fails.

    """
    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        assert isinstance(s, SendStream)
        assert isinstance(r, ReceiveStream)

        async def do_send_all(data: bytes | bytearray | memoryview) -> None:
            with assert_checkpoints():  # We're testing that it doesn't return anything.
                assert await s.send_all(data) is None  # type: ignore[func-returns-value]

        async def do_receive_some(max_bytes: int | None = None) -> bytes | bytearray:
            with assert_checkpoints():
                return await r.receive_some(max_bytes)

        async def checked_receive_1(expected: bytes) -> None:
            assert await do_receive_some(1) == expected

        async def do_aclose(resource: AsyncResource) -> None:
            with assert_checkpoints():
                await resource.aclose()

        # Simple sending/receiving
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, b"x")
            nursery.start_soon(checked_receive_1, b"x")

        async def send_empty_then_y() -> None:
            # Streams should tolerate sending b"" without giving it any
            # special meaning.
            await do_send_all(b"")
            await do_send_all(b"y")

        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_empty_then_y)
            nursery.start_soon(checked_receive_1, b"y")

        # ---- Checking various argument types ----

        # send_all accepts bytearray and memoryview
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, bytearray(b"1"))
            nursery.start_soon(checked_receive_1, b"1")

        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, memoryview(b"2"))
            nursery.start_soon(checked_receive_1, b"2")

        # max_bytes must be a positive integer
        with _assert_raises(ValueError):
            await r.receive_some(-1)
        with _assert_raises(ValueError):
            await r.receive_some(0)
        with _assert_raises(TypeError):
            await r.receive_some(1.5)  # type: ignore[arg-type]
        # it can also be missing or None
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, b"x")
            assert await do_receive_some() == b"x"
        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_send_all, b"x")
            assert await do_receive_some(None) == b"x"

        with _assert_raises(_core.BusyResourceError, wrapped=True):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(do_receive_some, 1)
                nursery.start_soon(do_receive_some, 1)

        # Method always has to exist, and an empty stream with a blocked
        # receive_some should *always* allow send_all. (Technically it's legal
        # for send_all to wait until receive_some is called to run, though; a
        # stream doesn't *have* to have any internal buffering. That's why we
        # start a concurrent receive_some call, then cancel it.)
        async def simple_check_wait_send_all_might_not_block(
            scope: CancelScope,
        ) -> None:
            with assert_checkpoints():
                await s.wait_send_all_might_not_block()
            scope.cancel()

        async with _core.open_nursery() as nursery:
            nursery.start_soon(
                simple_check_wait_send_all_might_not_block,
                nursery.cancel_scope,
            )
            nursery.start_soon(do_receive_some, 1)

        # closing the r side leads to BrokenResourceError on the s side
        # (eventually)
        async def expect_broken_stream_on_send() -> None:
            with _assert_raises(_core.BrokenResourceError):
                while True:
                    await do_send_all(b"x" * 100)

        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect_broken_stream_on_send)
            nursery.start_soon(do_aclose, r)

        # once detected, the stream stays broken
        with _assert_raises(_core.BrokenResourceError):
            await do_send_all(b"x" * 100)

        # r closed -> ClosedResourceError on the receive side
        with _assert_raises(_core.ClosedResourceError):
            await do_receive_some(4096)

        # we can close the same stream repeatedly, it's fine
        await do_aclose(r)
        await do_aclose(r)

        # closing the sender side
        await do_aclose(s)

        # now trying to send raises ClosedResourceError
        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b"x" * 100)

        # even if it's an empty send
        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b"")

        # ditto for wait_send_all_might_not_block
        with _assert_raises(_core.ClosedResourceError):
            with assert_checkpoints():
                await s.wait_send_all_might_not_block()

        # and again, repeated closing is fine
        await do_aclose(s)
        await do_aclose(s)

    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        # if send-then-graceful-close, receiver gets data then b""
        async def send_then_close() -> None:
            await do_send_all(b"y")
            await do_aclose(s)

        async def receive_send_then_close() -> None:
            # We want to make sure that if the sender closes the stream before
            # we read anything, then we still get all the data. But some
            # streams might block on the do_send_all call. So we let the
            # sender get as far as it can, then we receive.
            await _core.wait_all_tasks_blocked()
            await checked_receive_1(b"y")
            await checked_receive_1(b"")
            await do_aclose(r)

        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_then_close)
            nursery.start_soon(receive_send_then_close)

    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        await aclose_forcefully(r)

        with _assert_raises(_core.BrokenResourceError):
            while True:
                await do_send_all(b"x" * 100)

        with _assert_raises(_core.ClosedResourceError):
            await do_receive_some(4096)

    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        await aclose_forcefully(s)

        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b"123")

        # after the sender does a forceful close, the receiver might either
        # get BrokenResourceError or a clean b""; either is OK. Not OK would be
        # if it freezes, or returns data.
        with suppress(_core.BrokenResourceError):
            await checked_receive_1(b"")

    # cancelled aclose still closes
    async with _ForceCloseBoth(await stream_maker()) as (s, r):
        with _core.CancelScope() as scope:
            scope.cancel()
            await r.aclose()

        with _core.CancelScope() as scope:
            scope.cancel()
            await s.aclose()

        with _assert_raises(_core.ClosedResourceError):
            await do_send_all(b"123")

        with _assert_raises(_core.ClosedResourceError):
            await do_receive_some(4096)

    # Check that we can still gracefully close a stream after an operation has
    # been cancelled. This can be challenging if cancellation can leave the
    # stream internals in an inconsistent state, e.g. for
    # SSLStream. Unfortunately this test isn't very thorough; the really
    # challenging case for something like SSLStream is it gets cancelled
    # *while* it's sending data on the underlying, not before. But testing
    # that requires some special-case handling of the particular stream setup;
    # we can't do it here. Maybe we could do a bit better with
    #     https://github.com/python-trio/trio/issues/77
    async with _ForceCloseBoth(await stream_maker()) as (s, r):

        async def expect_cancelled(
            afn: Callable[ArgsT, Awaitable[object]],
            *args: ArgsT.args,
            **kwargs: ArgsT.kwargs,
        ) -> None:
            with _assert_raises(_core.Cancelled):
                await afn(*args, **kwargs)

        with _core.CancelScope() as scope:
            scope.cancel()
            async with _core.open_nursery() as nursery:
                nursery.start_soon(expect_cancelled, do_send_all, b"x")
                nursery.start_soon(expect_cancelled, do_receive_some, 1)

        async with _core.open_nursery() as nursery:
            nursery.start_soon(do_aclose, s)
            nursery.start_soon(do_aclose, r)

    # Check that if a task is blocked in receive_some, then closing the
    # receive stream causes it to wake up.
    async with _ForceCloseBoth(await stream_maker()) as (s, r):

        async def receive_expecting_closed() -> None:
            with _assert_raises(_core.ClosedResourceError):
                await r.receive_some(10)

        async with _core.open_nursery() as nursery:
            nursery.start_soon(receive_expecting_closed)
            await _core.wait_all_tasks_blocked()
            await aclose_forcefully(r)

    # check wait_send_all_might_not_block, if we can
    if clogged_stream_maker is not None:
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            record: list[str] = []

            async def waiter(cancel_scope: CancelScope) -> None:
                record.append("waiter sleeping")
                with assert_checkpoints():
                    await s.wait_send_all_might_not_block()
                record.append("waiter wokeup")
                cancel_scope.cancel()

            async def receiver() -> None:
                # give wait_send_all_might_not_block a chance to block
                await _core.wait_all_tasks_blocked()
                record.append("receiver starting")
                while True:
                    await r.receive_some(16834)

            async with _core.open_nursery() as nursery:
                nursery.start_soon(waiter, nursery.cancel_scope)
                await _core.wait_all_tasks_blocked()
                nursery.start_soon(receiver)

            assert record == [
                "waiter sleeping",
                "receiver starting",
                "waiter wokeup",
            ]

        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            # simultaneous wait_send_all_might_not_block fails
            with _assert_raises(_core.BusyResourceError, wrapped=True):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s.wait_send_all_might_not_block)
                    nursery.start_soon(s.wait_send_all_might_not_block)

            # and simultaneous send_all and wait_send_all_might_not_block (NB
            # this test might destroy the stream b/c we end up cancelling
            # send_all and e.g. SSLStream can't handle that, so we have to
            # recreate afterwards)
            with _assert_raises(_core.BusyResourceError, wrapped=True):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s.wait_send_all_might_not_block)
                    nursery.start_soon(s.send_all, b"123")

        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            # send_all and send_all blocked simultaneously should also raise
            # (but again this might destroy the stream)
            with _assert_raises(_core.BusyResourceError, wrapped=True):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s.send_all, b"123")
                    nursery.start_soon(s.send_all, b"123")

        # closing the receiver causes wait_send_all_might_not_block to return,
        # with or without an exception
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):

            async def sender() -> None:
                try:
                    with assert_checkpoints():
                        await s.wait_send_all_might_not_block()
                except _core.BrokenResourceError:  # pragma: no cover
                    pass

            async def receiver() -> None:
                await _core.wait_all_tasks_blocked()
                await aclose_forcefully(r)

            async with _core.open_nursery() as nursery:
                nursery.start_soon(sender)
                nursery.start_soon(receiver)

        # and again with the call starting after the close
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            await aclose_forcefully(r)
            try:
                with assert_checkpoints():
                    await s.wait_send_all_might_not_block()
            except _core.BrokenResourceError:  # pragma: no cover
                pass

        # Check that if a task is blocked in a send-side method, then closing
        # the send stream causes it to wake up.
        async def close_soon(s: SendStream) -> None:
            await _core.wait_all_tasks_blocked()
            await aclose_forcefully(s)

        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(close_soon, s)
                with _assert_raises(_core.ClosedResourceError):
                    await s.send_all(b"xyzzy")

        async with _ForceCloseBoth(await clogged_stream_maker()) as (s, r):
            async with _core.open_nursery() as nursery:
                nursery.start_soon(close_soon, s)
                with _assert_raises(_core.ClosedResourceError):
                    await s.wait_send_all_might_not_block()


async def check_two_way_stream(
    stream_maker: StreamMaker[Stream, Stream],
    clogged_stream_maker: StreamMaker[Stream, Stream] | None,
) -> None:
    """Perform a number of generic tests on a custom two-way stream
    implementation.

    This is similar to :func:`check_one_way_stream`, except that the maker
    functions are expected to return objects implementing the
    :class:`~trio.abc.Stream` interface.

    This function tests a *superset* of what :func:`check_one_way_stream`
    checks – if you call this, then you don't need to also call
    :func:`check_one_way_stream`.

    """
    await check_one_way_stream(stream_maker, clogged_stream_maker)

    async def flipped_stream_maker() -> tuple[Stream, Stream]:
        return (await stream_maker())[::-1]

    flipped_clogged_stream_maker: Callable[[], Awaitable[tuple[Stream, Stream]]] | None

    if clogged_stream_maker is not None:

        async def flipped_clogged_stream_maker() -> tuple[Stream, Stream]:
            return (await clogged_stream_maker())[::-1]

    else:
        flipped_clogged_stream_maker = None
    await check_one_way_stream(flipped_stream_maker, flipped_clogged_stream_maker)

    async with _ForceCloseBoth(await stream_maker()) as (s1, s2):
        assert isinstance(s1, Stream)
        assert isinstance(s2, Stream)

        # Duplex can be a bit tricky, might as well check it as well
        DUPLEX_TEST_SIZE = 2**20
        CHUNK_SIZE_MAX = 2**14

        r = random.Random(0)
        i = r.getrandbits(8 * DUPLEX_TEST_SIZE)
        test_data = i.to_bytes(DUPLEX_TEST_SIZE, "little")

        async def sender(
            s: Stream,
            data: bytes | bytearray | memoryview,
            seed: int,
        ) -> None:
            r = random.Random(seed)
            m = memoryview(data)
            while m:
                chunk_size = r.randint(1, CHUNK_SIZE_MAX)
                await s.send_all(m[:chunk_size])
                m = m[chunk_size:]

        async def receiver(s: Stream, data: bytes | bytearray, seed: int) -> None:
            r = random.Random(seed)
            got = bytearray()
            while len(got) < len(data):
                chunk = await s.receive_some(r.randint(1, CHUNK_SIZE_MAX))
                assert chunk
                got += chunk
            assert got == data

        async with _core.open_nursery() as nursery:
            nursery.start_soon(sender, s1, test_data, 0)
            nursery.start_soon(sender, s2, test_data[::-1], 1)
            nursery.start_soon(receiver, s1, test_data[::-1], 2)
            nursery.start_soon(receiver, s2, test_data, 3)

        async def expect_receive_some_empty() -> None:
            assert await s2.receive_some(10) == b""
            await s2.aclose()

        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect_receive_some_empty)
            nursery.start_soon(s1.aclose)


async def check_half_closeable_stream(
    stream_maker: StreamMaker[HalfCloseableStream, HalfCloseableStream],
    clogged_stream_maker: StreamMaker[HalfCloseableStream, HalfCloseableStream] | None,
) -> None:
    """Perform a number of generic tests on a custom half-closeable stream
    implementation.

    This is similar to :func:`check_two_way_stream`, except that the maker
    functions are expected to return objects that implement the
    :class:`~trio.abc.HalfCloseableStream` interface.

    This function tests a *superset* of what :func:`check_two_way_stream`
    checks – if you call this, then you don't need to also call
    :func:`check_two_way_stream`.

    """
    await check_two_way_stream(stream_maker, clogged_stream_maker)

    async with _ForceCloseBoth(await stream_maker()) as (s1, s2):
        assert isinstance(s1, HalfCloseableStream)
        assert isinstance(s2, HalfCloseableStream)

        async def send_x_then_eof(s: HalfCloseableStream) -> None:
            await s.send_all(b"x")
            with assert_checkpoints():
                await s.send_eof()

        async def expect_x_then_eof(r: HalfCloseableStream) -> None:
            await _core.wait_all_tasks_blocked()
            assert await r.receive_some(10) == b"x"
            assert await r.receive_some(10) == b""

        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_x_then_eof, s1)
            nursery.start_soon(expect_x_then_eof, s2)

        # now sending is disallowed
        with _assert_raises(_core.ClosedResourceError):
            await s1.send_all(b"y")

        # but we can do send_eof again
        with assert_checkpoints():
            await s1.send_eof()

        # and we can still send stuff back the other way
        async with _core.open_nursery() as nursery:
            nursery.start_soon(send_x_then_eof, s2)
            nursery.start_soon(expect_x_then_eof, s1)

    if clogged_stream_maker is not None:
        async with _ForceCloseBoth(await clogged_stream_maker()) as (s1, s2):
            # send_all and send_eof simultaneously is not ok
            with _assert_raises(_core.BusyResourceError, wrapped=True):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s1.send_all, b"x")
                    await _core.wait_all_tasks_blocked()
                    nursery.start_soon(s1.send_eof)

        async with _ForceCloseBoth(await clogged_stream_maker()) as (s1, s2):
            # wait_send_all_might_not_block and send_eof simultaneously is not
            # ok either
            with _assert_raises(_core.BusyResourceError, wrapped=True):
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(s1.wait_send_all_might_not_block)
                    await _core.wait_all_tasks_blocked()
                    nursery.start_soon(s1.send_eof)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\widgets\core.py ===
from markupsafe import escape
from markupsafe import Markup

__all__ = (
    "CheckboxInput",
    "ColorInput",
    "DateInput",
    "DateTimeInput",
    "DateTimeLocalInput",
    "EmailInput",
    "FileInput",
    "HiddenInput",
    "ListWidget",
    "MonthInput",
    "NumberInput",
    "Option",
    "PasswordInput",
    "RadioInput",
    "RangeInput",
    "SearchInput",
    "Select",
    "SubmitInput",
    "TableWidget",
    "TextArea",
    "TextInput",
    "TelInput",
    "TimeInput",
    "URLInput",
    "WeekInput",
)


def clean_key(key):
    key = key.rstrip("_")
    if key.startswith("data_") or key.startswith("aria_"):
        key = key.replace("_", "-")
    return key


def html_params(**kwargs):
    """
    Generate HTML attribute syntax from inputted keyword arguments.

    The output value is sorted by the passed keys, to provide consistent output
    each time this function is called with the same parameters. Because of the
    frequent use of the normally reserved keywords `class` and `for`, suffixing
    these with an underscore will allow them to be used.

    In order to facilitate the use of ``data-`` and ``aria-`` attributes, if the
    name of the attribute begins with ``data_`` or ``aria_``, then every
    underscore will be replaced with a hyphen in the generated attribute.

    >>> html_params(data_attr='user.name', aria_labeledby='name')
    'data-attr="user.name" aria-labeledby="name"'

    In addition, the values ``True`` and ``False`` are special:
      * ``attr=True`` generates the HTML compact output of a boolean attribute,
        e.g. ``checked=True`` will generate simply ``checked``
      * ``attr=False`` will be ignored and generate no output.

    >>> html_params(name='text1', id='f', class_='text')
    'class="text" id="f" name="text1"'
    >>> html_params(checked=True, readonly=False, name="text1", abc="hello")
    'abc="hello" checked name="text1"'

    .. versionchanged:: 3.0
        ``aria_`` args convert underscores to hyphens like ``data_``
        args.

    .. versionchanged:: 2.2
        ``data_`` args convert all underscores to hyphens, instead of
        only the first one.
    """
    params = []
    for k, v in sorted(kwargs.items()):
        k = clean_key(k)
        if v is True:
            params.append(k)
        elif v is False:
            pass
        else:
            params.append(f'{str(k)}="{escape(v)}"')  # noqa: B907
    return " ".join(params)


class ListWidget:
    """
    Renders a list of fields as a `ul` or `ol` list.

    This is used for fields which encapsulate many inner fields as subfields.
    The widget will try to iterate the field to get access to the subfields and
    call them to render them.

    If `prefix_label` is set, the subfield's label is printed before the field,
    otherwise afterwards. The latter is useful for iterating radios or
    checkboxes.
    """

    def __init__(self, html_tag="ul", prefix_label=True):
        assert html_tag in ("ol", "ul")
        self.html_tag = html_tag
        self.prefix_label = prefix_label

    def __call__(self, field, **kwargs):
        kwargs.setdefault("id", field.id)
        html = [f"<{self.html_tag} {html_params(**kwargs)}>"]
        for subfield in field:
            if self.prefix_label:
                html.append(f"<li>{subfield.label} {subfield()}</li>")
            else:
                html.append(f"<li>{subfield()} {subfield.label}</li>")
        html.append(f"</{self.html_tag}>")
        return Markup("".join(html))


class TableWidget:
    """
    Renders a list of fields as a set of table rows with th/td pairs.

    If `with_table_tag` is True, then an enclosing <table> is placed around the
    rows.

    Hidden fields will not be displayed with a row, instead the field will be
    pushed into a subsequent table row to ensure XHTML validity. Hidden fields
    at the end of the field list will appear outside the table.
    """

    def __init__(self, with_table_tag=True):
        self.with_table_tag = with_table_tag

    def __call__(self, field, **kwargs):
        html = []
        if self.with_table_tag:
            kwargs.setdefault("id", field.id)
            table_params = html_params(**kwargs)
            html.append(f"<table {table_params}>")
        hidden = ""
        for subfield in field:
            if subfield.type in ("HiddenField", "CSRFTokenField"):
                hidden += str(subfield)
            else:
                html.append(
                    f"<tr><th>{subfield.label}</th><td>{hidden}{subfield}</td></tr>"
                )
                hidden = ""
        if self.with_table_tag:
            html.append("</table>")
        if hidden:
            html.append(hidden)
        return Markup("".join(html))


class Input:
    """
    Render a basic ``<input>`` field.

    This is used as the basis for most of the other input fields.

    By default, the `_value()` method will be called upon the associated field
    to provide the ``value=`` HTML attribute.
    """

    html_params = staticmethod(html_params)

    def __init__(self, input_type=None):
        if input_type is not None:
            self.input_type = input_type

    def __call__(self, field, **kwargs):
        kwargs.setdefault("id", field.id)
        kwargs.setdefault("type", self.input_type)
        if "value" not in kwargs:
            kwargs["value"] = field._value()
        flags = getattr(field, "flags", {})
        for k in dir(flags):
            if k in self.validation_attrs and k not in kwargs:
                kwargs[k] = getattr(flags, k)
        input_params = self.html_params(name=field.name, **kwargs)
        return Markup(f"<input {input_params}>")


class TextInput(Input):
    """
    Render a single-line text input.
    """

    input_type = "text"
    validation_attrs = [
        "required",
        "disabled",
        "readonly",
        "maxlength",
        "minlength",
        "pattern",
    ]


class PasswordInput(Input):
    """
    Render a password input.

    For security purposes, this field will not reproduce the value on a form
    submit by default. To have the value filled in, set `hide_value` to
    `False`.
    """

    input_type = "password"
    validation_attrs = [
        "required",
        "disabled",
        "readonly",
        "maxlength",
        "minlength",
        "pattern",
    ]

    def __init__(self, hide_value=True):
        self.hide_value = hide_value

    def __call__(self, field, **kwargs):
        if self.hide_value:
            kwargs["value"] = ""
        return super().__call__(field, **kwargs)


class HiddenInput(Input):
    """
    Render a hidden input.
    """

    input_type = "hidden"
    validation_attrs = ["disabled"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_flags = {"hidden": True}


class CheckboxInput(Input):
    """
    Render a checkbox.

    The ``checked`` HTML attribute is set if the field's data is a non-false value.
    """

    input_type = "checkbox"
    validation_attrs = ["required", "disabled"]

    def __call__(self, field, **kwargs):
        if getattr(field, "checked", field.data):
            kwargs["checked"] = True
        return super().__call__(field, **kwargs)


class RadioInput(Input):
    """
    Render a single radio button.

    This widget is most commonly used in conjunction with ListWidget or some
    other listing, as singular radio buttons are not very useful.
    """

    input_type = "radio"
    validation_attrs = ["required", "disabled"]

    def __call__(self, field, **kwargs):
        if field.checked:
            kwargs["checked"] = True
        return super().__call__(field, **kwargs)


class FileInput(Input):
    """Render a file chooser input.

    :param multiple: allow choosing multiple files
    """

    input_type = "file"
    validation_attrs = ["required", "disabled", "accept"]

    def __init__(self, multiple=False):
        super().__init__()
        self.multiple = multiple

    def __call__(self, field, **kwargs):
        # browser ignores value of file input for security
        kwargs["value"] = False

        if self.multiple:
            kwargs["multiple"] = True

        return super().__call__(field, **kwargs)


class SubmitInput(Input):
    """
    Renders a submit button.

    The field's label is used as the text of the submit button instead of the
    data on the field.
    """

    input_type = "submit"
    validation_attrs = ["required", "disabled"]

    def __call__(self, field, **kwargs):
        kwargs.setdefault("value", field.label.text)
        return super().__call__(field, **kwargs)


class TextArea:
    """
    Renders a multi-line text area.

    `rows` and `cols` ought to be passed as keyword args when rendering.
    """

    validation_attrs = ["required", "disabled", "readonly", "maxlength", "minlength"]

    def __call__(self, field, **kwargs):
        kwargs.setdefault("id", field.id)
        flags = getattr(field, "flags", {})
        for k in dir(flags):
            if k in self.validation_attrs and k not in kwargs:
                kwargs[k] = getattr(flags, k)
        textarea_params = html_params(name=field.name, **kwargs)
        textarea_innerhtml = escape(field._value())
        return Markup(
            f"<textarea {textarea_params}>\r\n{textarea_innerhtml}</textarea>"
        )


class Select:
    """
    Renders a select field.

    If `multiple` is True, then the `size` property should be specified on
    rendering to make the field useful.

    The field must provide an `iter_choices()` method which the widget will
    call on rendering; this method must yield tuples of
    `(value, label, selected)` or `(value, label, selected, render_kw)`.
    It also must provide a `has_groups()` method which tells whether choices
    are divided into groups, and if they do, the field must have an
    `iter_groups()` method that yields tuples of `(label, choices)`, where
    `choices` is a iterable of `(value, label, selected)` tuples.
    """

    validation_attrs = ["required", "disabled"]

    def __init__(self, multiple=False):
        self.multiple = multiple

    def __call__(self, field, **kwargs):
        kwargs.setdefault("id", field.id)
        if self.multiple:
            kwargs["multiple"] = True
        flags = getattr(field, "flags", {})
        for k in dir(flags):
            if k in self.validation_attrs and k not in kwargs:
                kwargs[k] = getattr(flags, k)
        select_params = html_params(name=field.name, **kwargs)
        html = [f"<select {select_params}>"]
        if field.has_groups():
            for group, choices in field.iter_groups():
                optgroup_params = html_params(label=group)
                html.append(f"<optgroup {optgroup_params}>")
                for choice in choices:
                    val, label, selected, render_kw = choice
                    html.append(self.render_option(val, label, selected, **render_kw))
                html.append("</optgroup>")
        else:
            for choice in field.iter_choices():
                val, label, selected, render_kw = choice
                html.append(self.render_option(val, label, selected, **render_kw))
        html.append("</select>")
        return Markup("".join(html))

    @classmethod
    def render_option(cls, value, label, selected, **kwargs):
        if value is True:
            # Handle the special case of a 'True' value.
            value = str(value)

        options = dict(kwargs, value=value)
        if selected:
            options["selected"] = True
        return Markup(f"<option {html_params(**options)}>{escape(label)}</option>")


class Option:
    """
    Renders the individual option from a select field.

    This is just a convenience for various custom rendering situations, and an
    option by itself does not constitute an entire field.
    """

    def __call__(self, field, **kwargs):
        return Select.render_option(
            field._value(), field.label.text, field.checked, **kwargs
        )


class SearchInput(Input):
    """
    Renders an input with type "search".
    """

    input_type = "search"
    validation_attrs = [
        "required",
        "disabled",
        "readonly",
        "maxlength",
        "minlength",
        "pattern",
    ]


class TelInput(Input):
    """
    Renders an input with type "tel".
    """

    input_type = "tel"
    validation_attrs = [
        "required",
        "disabled",
        "readonly",
        "maxlength",
        "minlength",
        "pattern",
    ]


class URLInput(Input):
    """
    Renders an input with type "url".
    """

    input_type = "url"
    validation_attrs = [
        "required",
        "disabled",
        "readonly",
        "maxlength",
        "minlength",
        "pattern",
    ]


class EmailInput(Input):
    """
    Renders an input with type "email".
    """

    input_type = "email"
    validation_attrs = [
        "required",
        "disabled",
        "readonly",
        "maxlength",
        "minlength",
        "pattern",
    ]


class DateTimeInput(Input):
    """
    Renders an input with type "datetime".
    """

    input_type = "datetime"
    validation_attrs = ["required", "disabled", "readonly", "max", "min", "step"]


class DateInput(Input):
    """
    Renders an input with type "date".
    """

    input_type = "date"
    validation_attrs = ["required", "disabled", "readonly", "max", "min", "step"]


class MonthInput(Input):
    """
    Renders an input with type "month".
    """

    input_type = "month"
    validation_attrs = ["required", "disabled", "readonly", "max", "min", "step"]


class WeekInput(Input):
    """
    Renders an input with type "week".
    """

    input_type = "week"
    validation_attrs = ["required", "disabled", "readonly", "max", "min", "step"]


class TimeInput(Input):
    """
    Renders an input with type "time".
    """

    input_type = "time"
    validation_attrs = ["required", "disabled", "readonly", "max", "min", "step"]


class DateTimeLocalInput(Input):
    """
    Renders an input with type "datetime-local".
    """

    input_type = "datetime-local"
    validation_attrs = ["required", "disabled", "readonly", "max", "min", "step"]


class NumberInput(Input):
    """
    Renders an input with type "number".
    """

    input_type = "number"
    validation_attrs = ["required", "disabled", "readonly", "max", "min", "step"]

    def __init__(self, step=None, min=None, max=None):
        self.step = step
        self.min = min
        self.max = max

    def __call__(self, field, **kwargs):
        if self.step is not None:
            kwargs.setdefault("step", self.step)
        if self.min is not None:
            kwargs.setdefault("min", self.min)
        if self.max is not None:
            kwargs.setdefault("max", self.max)
        return super().__call__(field, **kwargs)


class RangeInput(Input):
    """
    Renders an input with type "range".
    """

    input_type = "range"
    validation_attrs = ["disabled", "max", "min", "step"]

    def __init__(self, step=None):
        self.step = step

    def __call__(self, field, **kwargs):
        if self.step is not None:
            kwargs.setdefault("step", self.step)
        return super().__call__(field, **kwargs)


class ColorInput(Input):
    """
    Renders an input with type "color".
    """

    input_type = "color"
    validation_attrs = ["disabled"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\truststore\_windows.py ===
import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_ulong,
    c_void_p,
    c_wchar_p,
    cast,
    create_unicode_buffer,
    pointer,
    sizeof,
)
from ctypes.wintypes import (
    BOOL,
    DWORD,
    HANDLE,
    LONG,
    LPCSTR,
    LPCVOID,
    LPCWSTR,
    LPFILETIME,
    LPSTR,
    LPWSTR,
)
from typing import TYPE_CHECKING, Any

from ._ssl_constants import _set_ssl_context_verify_mode

HCERTCHAINENGINE = HANDLE
HCERTSTORE = HANDLE
HCRYPTPROV_LEGACY = HANDLE


class CERT_CONTEXT(Structure):
    _fields_ = (
        ("dwCertEncodingType", DWORD),
        ("pbCertEncoded", c_void_p),
        ("cbCertEncoded", DWORD),
        ("pCertInfo", c_void_p),
        ("hCertStore", HCERTSTORE),
    )


PCERT_CONTEXT = POINTER(CERT_CONTEXT)
PCCERT_CONTEXT = POINTER(PCERT_CONTEXT)


class CERT_ENHKEY_USAGE(Structure):
    _fields_ = (
        ("cUsageIdentifier", DWORD),
        ("rgpszUsageIdentifier", POINTER(LPSTR)),
    )


PCERT_ENHKEY_USAGE = POINTER(CERT_ENHKEY_USAGE)


class CERT_USAGE_MATCH(Structure):
    _fields_ = (
        ("dwType", DWORD),
        ("Usage", CERT_ENHKEY_USAGE),
    )


class CERT_CHAIN_PARA(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("RequestedUsage", CERT_USAGE_MATCH),
        ("RequestedIssuancePolicy", CERT_USAGE_MATCH),
        ("dwUrlRetrievalTimeout", DWORD),
        ("fCheckRevocationFreshnessTime", BOOL),
        ("dwRevocationFreshnessTime", DWORD),
        ("pftCacheResync", LPFILETIME),
        ("pStrongSignPara", c_void_p),
        ("dwStrongSignFlags", DWORD),
    )


if TYPE_CHECKING:
    PCERT_CHAIN_PARA = pointer[CERT_CHAIN_PARA]  # type: ignore[misc]
else:
    PCERT_CHAIN_PARA = POINTER(CERT_CHAIN_PARA)


class CERT_TRUST_STATUS(Structure):
    _fields_ = (
        ("dwErrorStatus", DWORD),
        ("dwInfoStatus", DWORD),
    )


class CERT_CHAIN_ELEMENT(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("pCertContext", PCERT_CONTEXT),
        ("TrustStatus", CERT_TRUST_STATUS),
        ("pRevocationInfo", c_void_p),
        ("pIssuanceUsage", PCERT_ENHKEY_USAGE),
        ("pApplicationUsage", PCERT_ENHKEY_USAGE),
        ("pwszExtendedErrorInfo", LPCWSTR),
    )


PCERT_CHAIN_ELEMENT = POINTER(CERT_CHAIN_ELEMENT)


class CERT_SIMPLE_CHAIN(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("TrustStatus", CERT_TRUST_STATUS),
        ("cElement", DWORD),
        ("rgpElement", POINTER(PCERT_CHAIN_ELEMENT)),
        ("pTrustListInfo", c_void_p),
        ("fHasRevocationFreshnessTime", BOOL),
        ("dwRevocationFreshnessTime", DWORD),
    )


PCERT_SIMPLE_CHAIN = POINTER(CERT_SIMPLE_CHAIN)


class CERT_CHAIN_CONTEXT(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("TrustStatus", CERT_TRUST_STATUS),
        ("cChain", DWORD),
        ("rgpChain", POINTER(PCERT_SIMPLE_CHAIN)),
        ("cLowerQualityChainContext", DWORD),
        ("rgpLowerQualityChainContext", c_void_p),
        ("fHasRevocationFreshnessTime", BOOL),
        ("dwRevocationFreshnessTime", DWORD),
    )


PCERT_CHAIN_CONTEXT = POINTER(CERT_CHAIN_CONTEXT)
PCCERT_CHAIN_CONTEXT = POINTER(PCERT_CHAIN_CONTEXT)


class SSL_EXTRA_CERT_CHAIN_POLICY_PARA(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("dwAuthType", DWORD),
        ("fdwChecks", DWORD),
        ("pwszServerName", LPCWSTR),
    )


class CERT_CHAIN_POLICY_PARA(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("dwFlags", DWORD),
        ("pvExtraPolicyPara", c_void_p),
    )


PCERT_CHAIN_POLICY_PARA = POINTER(CERT_CHAIN_POLICY_PARA)


class CERT_CHAIN_POLICY_STATUS(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("dwError", DWORD),
        ("lChainIndex", LONG),
        ("lElementIndex", LONG),
        ("pvExtraPolicyStatus", c_void_p),
    )


PCERT_CHAIN_POLICY_STATUS = POINTER(CERT_CHAIN_POLICY_STATUS)


class CERT_CHAIN_ENGINE_CONFIG(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("hRestrictedRoot", HCERTSTORE),
        ("hRestrictedTrust", HCERTSTORE),
        ("hRestrictedOther", HCERTSTORE),
        ("cAdditionalStore", DWORD),
        ("rghAdditionalStore", c_void_p),
        ("dwFlags", DWORD),
        ("dwUrlRetrievalTimeout", DWORD),
        ("MaximumCachedCertificates", DWORD),
        ("CycleDetectionModulus", DWORD),
        ("hExclusiveRoot", HCERTSTORE),
        ("hExclusiveTrustedPeople", HCERTSTORE),
        ("dwExclusiveFlags", DWORD),
    )


PCERT_CHAIN_ENGINE_CONFIG = POINTER(CERT_CHAIN_ENGINE_CONFIG)
PHCERTCHAINENGINE = POINTER(HCERTCHAINENGINE)

X509_ASN_ENCODING = 0x00000001
PKCS_7_ASN_ENCODING = 0x00010000
CERT_STORE_PROV_MEMORY = b"Memory"
CERT_STORE_ADD_USE_EXISTING = 2
USAGE_MATCH_TYPE_OR = 1
OID_PKIX_KP_SERVER_AUTH = c_char_p(b"1.3.6.1.5.5.7.3.1")
CERT_CHAIN_REVOCATION_CHECK_END_CERT = 0x10000000
CERT_CHAIN_REVOCATION_CHECK_CHAIN = 0x20000000
CERT_CHAIN_POLICY_IGNORE_ALL_NOT_TIME_VALID_FLAGS = 0x00000007
CERT_CHAIN_POLICY_IGNORE_INVALID_BASIC_CONSTRAINTS_FLAG = 0x00000008
CERT_CHAIN_POLICY_ALLOW_UNKNOWN_CA_FLAG = 0x00000010
CERT_CHAIN_POLICY_IGNORE_INVALID_NAME_FLAG = 0x00000040
CERT_CHAIN_POLICY_IGNORE_WRONG_USAGE_FLAG = 0x00000020
CERT_CHAIN_POLICY_IGNORE_INVALID_POLICY_FLAG = 0x00000080
CERT_CHAIN_POLICY_IGNORE_ALL_REV_UNKNOWN_FLAGS = 0x00000F00
CERT_CHAIN_POLICY_ALLOW_TESTROOT_FLAG = 0x00008000
CERT_CHAIN_POLICY_TRUST_TESTROOT_FLAG = 0x00004000
SECURITY_FLAG_IGNORE_CERT_CN_INVALID = 0x00001000
AUTHTYPE_SERVER = 2
CERT_CHAIN_POLICY_SSL = 4
FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200

# Flags to set for SSLContext.verify_mode=CERT_NONE
CERT_CHAIN_POLICY_VERIFY_MODE_NONE_FLAGS = (
    CERT_CHAIN_POLICY_IGNORE_ALL_NOT_TIME_VALID_FLAGS
    | CERT_CHAIN_POLICY_IGNORE_INVALID_BASIC_CONSTRAINTS_FLAG
    | CERT_CHAIN_POLICY_ALLOW_UNKNOWN_CA_FLAG
    | CERT_CHAIN_POLICY_IGNORE_INVALID_NAME_FLAG
    | CERT_CHAIN_POLICY_IGNORE_WRONG_USAGE_FLAG
    | CERT_CHAIN_POLICY_IGNORE_INVALID_POLICY_FLAG
    | CERT_CHAIN_POLICY_IGNORE_ALL_REV_UNKNOWN_FLAGS
    | CERT_CHAIN_POLICY_ALLOW_TESTROOT_FLAG
    | CERT_CHAIN_POLICY_TRUST_TESTROOT_FLAG
)

wincrypt = WinDLL("crypt32.dll")
kernel32 = WinDLL("kernel32.dll")


def _handle_win_error(result: bool, _: Any, args: Any) -> Any:
    if not result:
        # Note, actually raises OSError after calling GetLastError and FormatMessage
        raise WinError()
    return args


CertCreateCertificateChainEngine = wincrypt.CertCreateCertificateChainEngine
CertCreateCertificateChainEngine.argtypes = (
    PCERT_CHAIN_ENGINE_CONFIG,
    PHCERTCHAINENGINE,
)
CertCreateCertificateChainEngine.errcheck = _handle_win_error

CertOpenStore = wincrypt.CertOpenStore
CertOpenStore.argtypes = (LPCSTR, DWORD, HCRYPTPROV_LEGACY, DWORD, c_void_p)
CertOpenStore.restype = HCERTSTORE
CertOpenStore.errcheck = _handle_win_error

CertAddEncodedCertificateToStore = wincrypt.CertAddEncodedCertificateToStore
CertAddEncodedCertificateToStore.argtypes = (
    HCERTSTORE,
    DWORD,
    c_char_p,
    DWORD,
    DWORD,
    PCCERT_CONTEXT,
)
CertAddEncodedCertificateToStore.restype = BOOL

CertCreateCertificateContext = wincrypt.CertCreateCertificateContext
CertCreateCertificateContext.argtypes = (DWORD, c_char_p, DWORD)
CertCreateCertificateContext.restype = PCERT_CONTEXT
CertCreateCertificateContext.errcheck = _handle_win_error

CertGetCertificateChain = wincrypt.CertGetCertificateChain
CertGetCertificateChain.argtypes = (
    HCERTCHAINENGINE,
    PCERT_CONTEXT,
    LPFILETIME,
    HCERTSTORE,
    PCERT_CHAIN_PARA,
    DWORD,
    c_void_p,
    PCCERT_CHAIN_CONTEXT,
)
CertGetCertificateChain.restype = BOOL
CertGetCertificateChain.errcheck = _handle_win_error

CertVerifyCertificateChainPolicy = wincrypt.CertVerifyCertificateChainPolicy
CertVerifyCertificateChainPolicy.argtypes = (
    c_ulong,
    PCERT_CHAIN_CONTEXT,
    PCERT_CHAIN_POLICY_PARA,
    PCERT_CHAIN_POLICY_STATUS,
)
CertVerifyCertificateChainPolicy.restype = BOOL

CertCloseStore = wincrypt.CertCloseStore
CertCloseStore.argtypes = (HCERTSTORE, DWORD)
CertCloseStore.restype = BOOL
CertCloseStore.errcheck = _handle_win_error

CertFreeCertificateChain = wincrypt.CertFreeCertificateChain
CertFreeCertificateChain.argtypes = (PCERT_CHAIN_CONTEXT,)

CertFreeCertificateContext = wincrypt.CertFreeCertificateContext
CertFreeCertificateContext.argtypes = (PCERT_CONTEXT,)

CertFreeCertificateChainEngine = wincrypt.CertFreeCertificateChainEngine
CertFreeCertificateChainEngine.argtypes = (HCERTCHAINENGINE,)

FormatMessageW = kernel32.FormatMessageW
FormatMessageW.argtypes = (
    DWORD,
    LPCVOID,
    DWORD,
    DWORD,
    LPWSTR,
    DWORD,
    c_void_p,
)
FormatMessageW.restype = DWORD


def _verify_peercerts_impl(
    ssl_context: ssl.SSLContext,
    cert_chain: list[bytes],
    server_hostname: str | None = None,
) -> None:
    """Verify the cert_chain from the server using Windows APIs."""

    # If the peer didn't send any certificates then
    # we can't do verification. Raise an error.
    if not cert_chain:
        raise ssl.SSLCertVerificationError("Peer sent no certificates to verify")

    pCertContext = None
    hIntermediateCertStore = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, None, 0, None)
    try:
        # Add intermediate certs to an in-memory cert store
        for cert_bytes in cert_chain[1:]:
            CertAddEncodedCertificateToStore(
                hIntermediateCertStore,
                X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
                cert_bytes,
                len(cert_bytes),
                CERT_STORE_ADD_USE_EXISTING,
                None,
            )

        # Cert context for leaf cert
        leaf_cert = cert_chain[0]
        pCertContext = CertCreateCertificateContext(
            X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, leaf_cert, len(leaf_cert)
        )

        # Chain params to match certs for serverAuth extended usage
        cert_enhkey_usage = CERT_ENHKEY_USAGE()
        cert_enhkey_usage.cUsageIdentifier = 1
        cert_enhkey_usage.rgpszUsageIdentifier = (c_char_p * 1)(OID_PKIX_KP_SERVER_AUTH)
        cert_usage_match = CERT_USAGE_MATCH()
        cert_usage_match.Usage = cert_enhkey_usage
        chain_params = CERT_CHAIN_PARA()
        chain_params.RequestedUsage = cert_usage_match
        chain_params.cbSize = sizeof(chain_params)
        pChainPara = pointer(chain_params)

        if ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_CHAIN:
            chain_flags = CERT_CHAIN_REVOCATION_CHECK_CHAIN
        elif ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF:
            chain_flags = CERT_CHAIN_REVOCATION_CHECK_END_CERT
        else:
            chain_flags = 0

        try:
            # First attempt to verify using the default Windows system trust roots
            # (default chain engine).
            _get_and_verify_cert_chain(
                ssl_context,
                None,
                hIntermediateCertStore,
                pCertContext,
                pChainPara,
                server_hostname,
                chain_flags=chain_flags,
            )
        except ssl.SSLCertVerificationError as e:
            # If that fails but custom CA certs have been added
            # to the SSLContext using load_verify_locations,
            # try verifying using a custom chain engine
            # that trusts the custom CA certs.
            custom_ca_certs: list[bytes] | None = ssl_context.get_ca_certs(
                binary_form=True
            )
            if custom_ca_certs:
                try:
                    _verify_using_custom_ca_certs(
                        ssl_context,
                        custom_ca_certs,
                        hIntermediateCertStore,
                        pCertContext,
                        pChainPara,
                        server_hostname,
                        chain_flags=chain_flags,
                    )
                # Raise the original error, not the new error.
                except ssl.SSLCertVerificationError:
                    raise e from None
            else:
                raise
    finally:
        CertCloseStore(hIntermediateCertStore, 0)
        if pCertContext:
            CertFreeCertificateContext(pCertContext)


def _get_and_verify_cert_chain(
    ssl_context: ssl.SSLContext,
    hChainEngine: HCERTCHAINENGINE | None,
    hIntermediateCertStore: HCERTSTORE,
    pPeerCertContext: c_void_p,
    pChainPara: PCERT_CHAIN_PARA,  # type: ignore[valid-type]
    server_hostname: str | None,
    chain_flags: int,
) -> None:
    ppChainContext = None
    try:
        # Get cert chain
        ppChainContext = pointer(PCERT_CHAIN_CONTEXT())
        CertGetCertificateChain(
            hChainEngine,  # chain engine
            pPeerCertContext,  # leaf cert context
            None,  # current system time
            hIntermediateCertStore,  # additional in-memory cert store
            pChainPara,  # chain-building parameters
            chain_flags,
            None,  # reserved
            ppChainContext,  # the resulting chain context
        )
        pChainContext = ppChainContext.contents

        # Verify cert chain
        ssl_extra_cert_chain_policy_para = SSL_EXTRA_CERT_CHAIN_POLICY_PARA()
        ssl_extra_cert_chain_policy_para.cbSize = sizeof(
            ssl_extra_cert_chain_policy_para
        )
        ssl_extra_cert_chain_policy_para.dwAuthType = AUTHTYPE_SERVER
        ssl_extra_cert_chain_policy_para.fdwChecks = 0
        if ssl_context.check_hostname is False:
            ssl_extra_cert_chain_policy_para.fdwChecks = (
                SECURITY_FLAG_IGNORE_CERT_CN_INVALID
            )
        if server_hostname:
            ssl_extra_cert_chain_policy_para.pwszServerName = c_wchar_p(server_hostname)

        chain_policy = CERT_CHAIN_POLICY_PARA()
        chain_policy.pvExtraPolicyPara = cast(
            pointer(ssl_extra_cert_chain_policy_para), c_void_p
        )
        if ssl_context.verify_mode == ssl.CERT_NONE:
            chain_policy.dwFlags |= CERT_CHAIN_POLICY_VERIFY_MODE_NONE_FLAGS
        chain_policy.cbSize = sizeof(chain_policy)

        pPolicyPara = pointer(chain_policy)
        policy_status = CERT_CHAIN_POLICY_STATUS()
        policy_status.cbSize = sizeof(policy_status)
        pPolicyStatus = pointer(policy_status)
        CertVerifyCertificateChainPolicy(
            CERT_CHAIN_POLICY_SSL,
            pChainContext,
            pPolicyPara,
            pPolicyStatus,
        )

        # Check status
        error_code = policy_status.dwError
        if error_code:
            # Try getting a human readable message for an error code.
            error_message_buf = create_unicode_buffer(1024)
            error_message_chars = FormatMessageW(
                FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                None,
                error_code,
                0,
                error_message_buf,
                sizeof(error_message_buf),
                None,
            )

            # See if we received a message for the error,
            # otherwise we use a generic error with the
            # error code and hope that it's search-able.
            if error_message_chars <= 0:
                error_message = f"Certificate chain policy error {error_code:#x} [{policy_status.lElementIndex}]"
            else:
                error_message = error_message_buf.value.strip()

            err = ssl.SSLCertVerificationError(error_message)
            err.verify_message = error_message
            err.verify_code = error_code
            raise err from None
    finally:
        if ppChainContext:
            CertFreeCertificateChain(ppChainContext.contents)


def _verify_using_custom_ca_certs(
    ssl_context: ssl.SSLContext,
    custom_ca_certs: list[bytes],
    hIntermediateCertStore: HCERTSTORE,
    pPeerCertContext: c_void_p,
    pChainPara: PCERT_CHAIN_PARA,  # type: ignore[valid-type]
    server_hostname: str | None,
    chain_flags: int,
) -> None:
    hChainEngine = None
    hRootCertStore = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, None, 0, None)
    try:
        # Add custom CA certs to an in-memory cert store
        for cert_bytes in custom_ca_certs:
            CertAddEncodedCertificateToStore(
                hRootCertStore,
                X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
                cert_bytes,
                len(cert_bytes),
                CERT_STORE_ADD_USE_EXISTING,
                None,
            )

        # Create a custom cert chain engine which exclusively trusts
        # certs from our hRootCertStore
        cert_chain_engine_config = CERT_CHAIN_ENGINE_CONFIG()
        cert_chain_engine_config.cbSize = sizeof(cert_chain_engine_config)
        cert_chain_engine_config.hExclusiveRoot = hRootCertStore
        pConfig = pointer(cert_chain_engine_config)
        phChainEngine = pointer(HCERTCHAINENGINE())
        CertCreateCertificateChainEngine(
            pConfig,
            phChainEngine,
        )
        hChainEngine = phChainEngine.contents

        # Get and verify a cert chain using the custom chain engine
        _get_and_verify_cert_chain(
            ssl_context,
            hChainEngine,
            hIntermediateCertStore,
            pPeerCertContext,
            pChainPara,
            server_hostname,
            chain_flags,
        )
    finally:
        if hChainEngine:
            CertFreeCertificateChainEngine(hChainEngine)
        CertCloseStore(hRootCertStore, 0)


@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    check_hostname = ctx.check_hostname
    verify_mode = ctx.verify_mode
    ctx.check_hostname = False
    _set_ssl_context_verify_mode(ctx, ssl.CERT_NONE)
    try:
        yield
    finally:
        ctx.check_hostname = check_hostname
        _set_ssl_context_verify_mode(ctx, verify_mode)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\network\auth.py ===
"""Network Authentication Helpers

Contains interface (MultiDomainBasicAuth) and associated glue code for
providing credentials in the context of network requests.
"""

import logging
import os
import shutil
import subprocess
import sysconfig
import typing
import urllib.parse
from abc import ABC, abstractmethod
from functools import lru_cache
from os.path import commonprefix
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from pip._vendor.requests.auth import AuthBase, HTTPBasicAuth
from pip._vendor.requests.models import Request, Response
from pip._vendor.requests.utils import get_netrc_auth

from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
    ask,
    ask_input,
    ask_password,
    remove_auth_from_url,
    split_auth_netloc_from_url,
)
from pip._internal.vcs.versioncontrol import AuthInfo

logger = getLogger(__name__)

KEYRING_DISABLED = False


class Credentials(NamedTuple):
    url: str
    username: str
    password: str


class KeyRingBaseProvider(ABC):
    """Keyring base provider interface"""

    has_keyring: bool

    @abstractmethod
    def get_auth_info(
        self, url: str, username: Optional[str]
    ) -> Optional[AuthInfo]: ...

    @abstractmethod
    def save_auth_info(self, url: str, username: str, password: str) -> None: ...


class KeyRingNullProvider(KeyRingBaseProvider):
    """Keyring null provider"""

    has_keyring = False

    def get_auth_info(self, url: str, username: Optional[str]) -> Optional[AuthInfo]:
        return None

    def save_auth_info(self, url: str, username: str, password: str) -> None:
        return None


class KeyRingPythonProvider(KeyRingBaseProvider):
    """Keyring interface which uses locally imported `keyring`"""

    has_keyring = True

    def __init__(self) -> None:
        import keyring

        self.keyring = keyring

    def get_auth_info(self, url: str, username: Optional[str]) -> Optional[AuthInfo]:
        # Support keyring's get_credential interface which supports getting
        # credentials without a username. This is only available for
        # keyring>=15.2.0.
        if hasattr(self.keyring, "get_credential"):
            logger.debug("Getting credentials from keyring for %s", url)
            cred = self.keyring.get_credential(url, username)
            if cred is not None:
                return cred.username, cred.password
            return None

        if username is not None:
            logger.debug("Getting password from keyring for %s", url)
            password = self.keyring.get_password(url, username)
            if password:
                return username, password
        return None

    def save_auth_info(self, url: str, username: str, password: str) -> None:
        self.keyring.set_password(url, username, password)


class KeyRingCliProvider(KeyRingBaseProvider):
    """Provider which uses `keyring` cli

    Instead of calling the keyring package installed alongside pip
    we call keyring on the command line which will enable pip to
    use which ever installation of keyring is available first in
    PATH.
    """

    has_keyring = True

    def __init__(self, cmd: str) -> None:
        self.keyring = cmd

    def get_auth_info(self, url: str, username: Optional[str]) -> Optional[AuthInfo]:
        # This is the default implementation of keyring.get_credential
        # https://github.com/jaraco/keyring/blob/97689324abcf01bd1793d49063e7ca01e03d7d07/keyring/backend.py#L134-L139
        if username is not None:
            password = self._get_password(url, username)
            if password is not None:
                return username, password
        return None

    def save_auth_info(self, url: str, username: str, password: str) -> None:
        return self._set_password(url, username, password)

    def _get_password(self, service_name: str, username: str) -> Optional[str]:
        """Mirror the implementation of keyring.get_password using cli"""
        if self.keyring is None:
            return None

        cmd = [self.keyring, "get", service_name, username]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            env=env,
        )
        if res.returncode:
            return None
        return res.stdout.decode("utf-8").strip(os.linesep)

    def _set_password(self, service_name: str, username: str, password: str) -> None:
        """Mirror the implementation of keyring.set_password using cli"""
        if self.keyring is None:
            return None
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(
            [self.keyring, "set", service_name, username],
            input=f"{password}{os.linesep}".encode(),
            env=env,
            check=True,
        )
        return None


@lru_cache(maxsize=None)
def get_keyring_provider(provider: str) -> KeyRingBaseProvider:
    logger.verbose("Keyring provider requested: %s", provider)

    # keyring has previously failed and been disabled
    if KEYRING_DISABLED:
        provider = "disabled"
    if provider in ["import", "auto"]:
        try:
            impl = KeyRingPythonProvider()
            logger.verbose("Keyring provider set: import")
            return impl
        except ImportError:
            pass
        except Exception as exc:
            # In the event of an unexpected exception
            # we should warn the user
            msg = "Installed copy of keyring fails with exception %s"
            if provider == "auto":
                msg = msg + ", trying to find a keyring executable as a fallback"
            logger.warning(msg, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
    if provider in ["subprocess", "auto"]:
        cli = shutil.which("keyring")
        if cli and cli.startswith(sysconfig.get_path("scripts")):
            # all code within this function is stolen from shutil.which implementation
            @typing.no_type_check
            def PATH_as_shutil_which_determines_it() -> str:
                path = os.environ.get("PATH", None)
                if path is None:
                    try:
                        path = os.confstr("CS_PATH")
                    except (AttributeError, ValueError):
                        # os.confstr() or CS_PATH is not available
                        path = os.defpath
                # bpo-35755: Don't use os.defpath if the PATH environment variable is
                # set to an empty string

                return path

            scripts = Path(sysconfig.get_path("scripts"))

            paths = []
            for path in PATH_as_shutil_which_determines_it().split(os.pathsep):
                p = Path(path)
                try:
                    if not p.samefile(scripts):
                        paths.append(path)
                except FileNotFoundError:
                    pass

            path = os.pathsep.join(paths)

            cli = shutil.which("keyring", path=path)

        if cli:
            logger.verbose("Keyring provider set: subprocess with executable %s", cli)
            return KeyRingCliProvider(cli)

    logger.verbose("Keyring provider set: disabled")
    return KeyRingNullProvider()


class MultiDomainBasicAuth(AuthBase):
    def __init__(
        self,
        prompting: bool = True,
        index_urls: Optional[List[str]] = None,
        keyring_provider: str = "auto",
    ) -> None:
        self.prompting = prompting
        self.index_urls = index_urls
        self.keyring_provider = keyring_provider  # type: ignore[assignment]
        self.passwords: Dict[str, AuthInfo] = {}
        # When the user is prompted to enter credentials and keyring is
        # available, we will offer to save them. If the user accepts,
        # this value is set to the credentials they entered. After the
        # request authenticates, the caller should call
        # ``save_credentials`` to save these.
        self._credentials_to_save: Optional[Credentials] = None

    @property
    def keyring_provider(self) -> KeyRingBaseProvider:
        return get_keyring_provider(self._keyring_provider)

    @keyring_provider.setter
    def keyring_provider(self, provider: str) -> None:
        # The free function get_keyring_provider has been decorated with
        # functools.cache. If an exception occurs in get_keyring_auth that
        # cache will be cleared and keyring disabled, take that into account
        # if you want to remove this indirection.
        self._keyring_provider = provider

    @property
    def use_keyring(self) -> bool:
        # We won't use keyring when --no-input is passed unless
        # a specific provider is requested because it might require
        # user interaction
        return self.prompting or self._keyring_provider not in ["auto", "disabled"]

    def _get_keyring_auth(
        self,
        url: Optional[str],
        username: Optional[str],
    ) -> Optional[AuthInfo]:
        """Return the tuple auth for a given url from keyring."""
        # Do nothing if no url was provided
        if not url:
            return None

        try:
            return self.keyring_provider.get_auth_info(url, username)
        except Exception as exc:
            # Log the full exception (with stacktrace) at debug, so it'll only
            # show up when running in verbose mode.
            logger.debug("Keyring is skipped due to an exception", exc_info=True)
            # Always log a shortened version of the exception.
            logger.warning(
                "Keyring is skipped due to an exception: %s",
                str(exc),
            )
            global KEYRING_DISABLED
            KEYRING_DISABLED = True
            get_keyring_provider.cache_clear()
            return None

    def _get_index_url(self, url: str) -> Optional[str]:
        """Return the original index URL matching the requested URL.

        Cached or dynamically generated credentials may work against
        the original index URL rather than just the netloc.

        The provided url should have had its username and password
        removed already. If the original index url had credentials then
        they will be included in the return value.

        Returns None if no matching index was found, or if --no-index
        was specified by the user.
        """
        if not url or not self.index_urls:
            return None

        url = remove_auth_from_url(url).rstrip("/") + "/"
        parsed_url = urllib.parse.urlsplit(url)

        candidates = []

        for index in self.index_urls:
            index = index.rstrip("/") + "/"
            parsed_index = urllib.parse.urlsplit(remove_auth_from_url(index))
            if parsed_url == parsed_index:
                return index

            if parsed_url.netloc != parsed_index.netloc:
                continue

            candidate = urllib.parse.urlsplit(index)
            candidates.append(candidate)

        if not candidates:
            return None

        candidates.sort(
            reverse=True,
            key=lambda candidate: commonprefix(
                [
                    parsed_url.path,
                    candidate.path,
                ]
            ).rfind("/"),
        )

        return urllib.parse.urlunsplit(candidates[0])

    def _get_new_credentials(
        self,
        original_url: str,
        *,
        allow_netrc: bool = True,
        allow_keyring: bool = False,
    ) -> AuthInfo:
        """Find and return credentials for the specified URL."""
        # Split the credentials and netloc from the url.
        url, netloc, url_user_password = split_auth_netloc_from_url(
            original_url,
        )

        # Start with the credentials embedded in the url
        username, password = url_user_password
        if username is not None and password is not None:
            logger.debug("Found credentials in url for %s", netloc)
            return url_user_password

        # Find a matching index url for this request
        index_url = self._get_index_url(url)
        if index_url:
            # Split the credentials from the url.
            index_info = split_auth_netloc_from_url(index_url)
            if index_info:
                index_url, _, index_url_user_password = index_info
                logger.debug("Found index url %s", index_url)

        # If an index URL was found, try its embedded credentials
        if index_url and index_url_user_password[0] is not None:
            username, password = index_url_user_password
            if username is not None and password is not None:
                logger.debug("Found credentials in index url for %s", netloc)
                return index_url_user_password

        # Get creds from netrc if we still don't have them
        if allow_netrc:
            netrc_auth = get_netrc_auth(original_url)
            if netrc_auth:
                logger.debug("Found credentials in netrc for %s", netloc)
                return netrc_auth

        # If we don't have a password and keyring is available, use it.
        if allow_keyring:
            # The index url is more specific than the netloc, so try it first
            # fmt: off
            kr_auth = (
                self._get_keyring_auth(index_url, username) or
                self._get_keyring_auth(netloc, username)
            )
            # fmt: on
            if kr_auth:
                logger.debug("Found credentials in keyring for %s", netloc)
                return kr_auth

        return username, password

    def _get_url_and_credentials(
        self, original_url: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Return the credentials to use for the provided URL.

        If allowed, netrc and keyring may be used to obtain the
        correct credentials.

        Returns (url_without_credentials, username, password). Note
        that even if the original URL contains credentials, this
        function may return a different username and password.
        """
        url, netloc, _ = split_auth_netloc_from_url(original_url)

        # Try to get credentials from original url
        username, password = self._get_new_credentials(original_url)

        # If credentials not found, use any stored credentials for this netloc.
        # Do this if either the username or the password is missing.
        # This accounts for the situation in which the user has specified
        # the username in the index url, but the password comes from keyring.
        if (username is None or password is None) and netloc in self.passwords:
            un, pw = self.passwords[netloc]
            # It is possible that the cached credentials are for a different username,
            # in which case the cache should be ignored.
            if username is None or username == un:
                username, password = un, pw

        if username is not None or password is not None:
            # Convert the username and password if they're None, so that
            # this netloc will show up as "cached" in the conditional above.
            # Further, HTTPBasicAuth doesn't accept None, so it makes sense to
            # cache the value that is going to be used.
            username = username or ""
            password = password or ""

            # Store any acquired credentials.
            self.passwords[netloc] = (username, password)

        assert (
            # Credentials were found
            (username is not None and password is not None)
            # Credentials were not found
            or (username is None and password is None)
        ), f"Could not load credentials from url: {original_url}"

        return url, username, password

    def __call__(self, req: Request) -> Request:
        # Get credentials for this request
        url, username, password = self._get_url_and_credentials(req.url)

        # Set the url of the request to the url without any credentials
        req.url = url

        if username is not None and password is not None:
            # Send the basic auth with this request
            req = HTTPBasicAuth(username, password)(req)

        # Attach a hook to handle 401 responses
        req.register_hook("response", self.handle_401)

        return req

    # Factored out to allow for easy patching in tests
    def _prompt_for_password(
        self, netloc: str
    ) -> Tuple[Optional[str], Optional[str], bool]:
        username = ask_input(f"User for {netloc}: ") if self.prompting else None
        if not username:
            return None, None, False
        if self.use_keyring:
            auth = self._get_keyring_auth(netloc, username)
            if auth and auth[0] is not None and auth[1] is not None:
                return auth[0], auth[1], False
        password = ask_password("Password: ")
        return username, password, True

    # Factored out to allow for easy patching in tests
    def _should_save_password_to_keyring(self) -> bool:
        if (
            not self.prompting
            or not self.use_keyring
            or not self.keyring_provider.has_keyring
        ):
            return False
        return ask("Save credentials to keyring [y/N]: ", ["y", "n"]) == "y"

    def handle_401(self, resp: Response, **kwargs: Any) -> Response:
        # We only care about 401 responses, anything else we want to just
        #   pass through the actual response
        if resp.status_code != 401:
            return resp

        username, password = None, None

        # Query the keyring for credentials:
        if self.use_keyring:
            username, password = self._get_new_credentials(
                resp.url,
                allow_netrc=False,
                allow_keyring=True,
            )

        # We are not able to prompt the user so simply return the response
        if not self.prompting and not username and not password:
            return resp

        parsed = urllib.parse.urlparse(resp.url)

        # Prompt the user for a new username and password
        save = False
        if not username and not password:
            username, password, save = self._prompt_for_password(parsed.netloc)

        # Store the new username and password to use for future requests
        self._credentials_to_save = None
        if username is not None and password is not None:
            self.passwords[parsed.netloc] = (username, password)

            # Prompt to save the password to keyring
            if save and self._should_save_password_to_keyring():
                self._credentials_to_save = Credentials(
                    url=parsed.netloc,
                    username=username,
                    password=password,
                )

        # Consume content and release the original connection to allow our new
        #   request to reuse the same one.
        # The result of the assignment isn't used, it's just needed to consume
        # the content.
        _ = resp.content
        resp.raw.release_conn()

        # Add our new username and password to the request
        req = HTTPBasicAuth(username or "", password or "")(resp.request)
        req.register_hook("response", self.warn_on_401)

        # On successful request, save the credentials that were used to
        # keyring. (Note that if the user responded "no" above, this member
        # is not set and nothing will be saved.)
        if self._credentials_to_save:
            req.register_hook("response", self.save_credentials)

        # Send our new request
        new_resp = resp.connection.send(req, **kwargs)
        new_resp.history.append(resp)

        return new_resp

    def warn_on_401(self, resp: Response, **kwargs: Any) -> None:
        """Response callback to warn about incorrect credentials."""
        if resp.status_code == 401:
            logger.warning(
                "401 Error, Credentials not correct for %s",
                resp.request.url,
            )

    def save_credentials(self, resp: Response, **kwargs: Any) -> None:
        """Response callback to save credentials on success."""
        assert (
            self.keyring_provider.has_keyring
        ), "should never reach here without keyring"

        creds = self._credentials_to_save
        self._credentials_to_save = None
        if creds and resp.status_code < 400:
            try:
                logger.info("Saving credentials to keyring")
                self.keyring_provider.save_auth_info(
                    creds.url, creds.username, creds.password
                )
            except Exception:
                logger.exception("Failed to save credentials")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\testing.py ===
from __future__ import annotations

import collections.abc as cabc
import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import typing as t
from types import TracebackType

from . import _compat
from . import formatting
from . import termui
from . import utils
from ._compat import _find_binary_reader

if t.TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from .core import Command


class EchoingStdin:
    def __init__(self, input: t.BinaryIO, output: t.BinaryIO) -> None:
        self._input = input
        self._output = output
        self._paused = False

    def __getattr__(self, x: str) -> t.Any:
        return getattr(self._input, x)

    def _echo(self, rv: bytes) -> bytes:
        if not self._paused:
            self._output.write(rv)

        return rv

    def read(self, n: int = -1) -> bytes:
        return self._echo(self._input.read(n))

    def read1(self, n: int = -1) -> bytes:
        return self._echo(self._input.read1(n))  # type: ignore

    def readline(self, n: int = -1) -> bytes:
        return self._echo(self._input.readline(n))

    def readlines(self) -> list[bytes]:
        return [self._echo(x) for x in self._input.readlines()]

    def __iter__(self) -> cabc.Iterator[bytes]:
        return iter(self._echo(x) for x in self._input)

    def __repr__(self) -> str:
        return repr(self._input)


@contextlib.contextmanager
def _pause_echo(stream: EchoingStdin | None) -> cabc.Iterator[None]:
    if stream is None:
        yield
    else:
        stream._paused = True
        yield
        stream._paused = False


class BytesIOCopy(io.BytesIO):
    """Patch ``io.BytesIO`` to let the written stream be copied to another.

    .. versionadded:: 8.2
    """

    def __init__(self, copy_to: io.BytesIO) -> None:
        super().__init__()
        self.copy_to = copy_to

    def flush(self) -> None:
        super().flush()
        self.copy_to.flush()

    def write(self, b: ReadableBuffer) -> int:
        self.copy_to.write(b)
        return super().write(b)


class StreamMixer:
    """Mixes `<stdout>` and `<stderr>` streams.

    The result is available in the ``output`` attribute.

    .. versionadded:: 8.2
    """

    def __init__(self) -> None:
        self.output: io.BytesIO = io.BytesIO()
        self.stdout: io.BytesIO = BytesIOCopy(copy_to=self.output)
        self.stderr: io.BytesIO = BytesIOCopy(copy_to=self.output)


class _NamedTextIOWrapper(io.TextIOWrapper):
    def __init__(
        self, buffer: t.BinaryIO, name: str, mode: str, **kwargs: t.Any
    ) -> None:
        super().__init__(buffer, **kwargs)
        self._name = name
        self._mode = mode

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    def __next__(self) -> str:  # type: ignore
        try:
            line = super().__next__()
        except StopIteration as e:
            raise EOFError() from e
        return line


def make_input_stream(
    input: str | bytes | t.IO[t.Any] | None, charset: str
) -> t.BinaryIO:
    # Is already an input stream.
    if hasattr(input, "read"):
        rv = _find_binary_reader(t.cast("t.IO[t.Any]", input))

        if rv is not None:
            return rv

        raise TypeError("Could not find binary reader for input stream.")

    if input is None:
        input = b""
    elif isinstance(input, str):
        input = input.encode(charset)

    return io.BytesIO(input)


class Result:
    """Holds the captured result of an invoked CLI script.

    :param runner: The runner that created the result
    :param stdout_bytes: The standard output as bytes.
    :param stderr_bytes: The standard error as bytes.
    :param output_bytes: A mix of ``stdout_bytes`` and ``stderr_bytes``, as the
        user would see  it in its terminal.
    :param return_value: The value returned from the invoked command.
    :param exit_code: The exit code as integer.
    :param exception: The exception that happened if one did.
    :param exc_info: Exception information (exception type, exception instance,
        traceback type).

    .. versionchanged:: 8.2
        ``stderr_bytes`` no longer optional, ``output_bytes`` introduced and
        ``mix_stderr`` has been removed.

    .. versionadded:: 8.0
        Added ``return_value``.
    """

    def __init__(
        self,
        runner: CliRunner,
        stdout_bytes: bytes,
        stderr_bytes: bytes,
        output_bytes: bytes,
        return_value: t.Any,
        exit_code: int,
        exception: BaseException | None,
        exc_info: tuple[type[BaseException], BaseException, TracebackType]
        | None = None,
    ):
        self.runner = runner
        self.stdout_bytes = stdout_bytes
        self.stderr_bytes = stderr_bytes
        self.output_bytes = output_bytes
        self.return_value = return_value
        self.exit_code = exit_code
        self.exception = exception
        self.exc_info = exc_info

    @property
    def output(self) -> str:
        """The terminal output as unicode string, as the user would see it.

        .. versionchanged:: 8.2
            No longer a proxy for ``self.stdout``. Now has its own independent stream
            that is mixing `<stdout>` and `<stderr>`, in the order they were written.
        """
        return self.output_bytes.decode(self.runner.charset, "replace").replace(
            "\r\n", "\n"
        )

    @property
    def stdout(self) -> str:
        """The standard output as unicode string."""
        return self.stdout_bytes.decode(self.runner.charset, "replace").replace(
            "\r\n", "\n"
        )

    @property
    def stderr(self) -> str:
        """The standard error as unicode string.

        .. versionchanged:: 8.2
            No longer raise an exception, always returns the `<stderr>` string.
        """
        return self.stderr_bytes.decode(self.runner.charset, "replace").replace(
            "\r\n", "\n"
        )

    def __repr__(self) -> str:
        exc_str = repr(self.exception) if self.exception else "okay"
        return f"<{type(self).__name__} {exc_str}>"


class CliRunner:
    """The CLI runner provides functionality to invoke a Click command line
    script for unittesting purposes in a isolated environment.  This only
    works in single-threaded systems without any concurrency as it changes the
    global interpreter state.

    :param charset: the character set for the input and output data.
    :param env: a dictionary with environment variables for overriding.
    :param echo_stdin: if this is set to `True`, then reading from `<stdin>` writes
                       to `<stdout>`.  This is useful for showing examples in
                       some circumstances.  Note that regular prompts
                       will automatically echo the input.
    :param catch_exceptions: Whether to catch any exceptions other than
                             ``SystemExit`` when running :meth:`~CliRunner.invoke`.

    .. versionchanged:: 8.2
        Added the ``catch_exceptions`` parameter.

    .. versionchanged:: 8.2
        ``mix_stderr`` parameter has been removed.
    """

    def __init__(
        self,
        charset: str = "utf-8",
        env: cabc.Mapping[str, str | None] | None = None,
        echo_stdin: bool = False,
        catch_exceptions: bool = True,
    ) -> None:
        self.charset = charset
        self.env: cabc.Mapping[str, str | None] = env or {}
        self.echo_stdin = echo_stdin
        self.catch_exceptions = catch_exceptions

    def get_default_prog_name(self, cli: Command) -> str:
        """Given a command object it will return the default program name
        for it.  The default is the `name` attribute or ``"root"`` if not
        set.
        """
        return cli.name or "root"

    def make_env(
        self, overrides: cabc.Mapping[str, str | None] | None = None
    ) -> cabc.Mapping[str, str | None]:
        """Returns the environment overrides for invoking a script."""
        rv = dict(self.env)
        if overrides:
            rv.update(overrides)
        return rv

    @contextlib.contextmanager
    def isolation(
        self,
        input: str | bytes | t.IO[t.Any] | None = None,
        env: cabc.Mapping[str, str | None] | None = None,
        color: bool = False,
    ) -> cabc.Iterator[tuple[io.BytesIO, io.BytesIO, io.BytesIO]]:
        """A context manager that sets up the isolation for invoking of a
        command line tool.  This sets up `<stdin>` with the given input data
        and `os.environ` with the overrides from the given dictionary.
        This also rebinds some internals in Click to be mocked (like the
        prompt functionality).

        This is automatically done in the :meth:`invoke` method.

        :param input: the input stream to put into `sys.stdin`.
        :param env: the environment overrides as dictionary.
        :param color: whether the output should contain color codes. The
                      application can still override this explicitly.

        .. versionadded:: 8.2
            An additional output stream is returned, which is a mix of
            `<stdout>` and `<stderr>` streams.

        .. versionchanged:: 8.2
            Always returns the `<stderr>` stream.

        .. versionchanged:: 8.0
            `<stderr>` is opened with ``errors="backslashreplace"``
            instead of the default ``"strict"``.

        .. versionchanged:: 4.0
            Added the ``color`` parameter.
        """
        bytes_input = make_input_stream(input, self.charset)
        echo_input = None

        old_stdin = sys.stdin
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_forced_width = formatting.FORCED_WIDTH
        formatting.FORCED_WIDTH = 80

        env = self.make_env(env)

        stream_mixer = StreamMixer()

        if self.echo_stdin:
            bytes_input = echo_input = t.cast(
                t.BinaryIO, EchoingStdin(bytes_input, stream_mixer.stdout)
            )

        sys.stdin = text_input = _NamedTextIOWrapper(
            bytes_input, encoding=self.charset, name="<stdin>", mode="r"
        )

        if self.echo_stdin:
            # Force unbuffered reads, otherwise TextIOWrapper reads a
            # large chunk which is echoed early.
            text_input._CHUNK_SIZE = 1  # type: ignore

        sys.stdout = _NamedTextIOWrapper(
            stream_mixer.stdout, encoding=self.charset, name="<stdout>", mode="w"
        )

        sys.stderr = _NamedTextIOWrapper(
            stream_mixer.stderr,
            encoding=self.charset,
            name="<stderr>",
            mode="w",
            errors="backslashreplace",
        )

        @_pause_echo(echo_input)  # type: ignore
        def visible_input(prompt: str | None = None) -> str:
            sys.stdout.write(prompt or "")
            val = next(text_input).rstrip("\r\n")
            sys.stdout.write(f"{val}\n")
            sys.stdout.flush()
            return val

        @_pause_echo(echo_input)  # type: ignore
        def hidden_input(prompt: str | None = None) -> str:
            sys.stdout.write(f"{prompt or ''}\n")
            sys.stdout.flush()
            return next(text_input).rstrip("\r\n")

        @_pause_echo(echo_input)  # type: ignore
        def _getchar(echo: bool) -> str:
            char = sys.stdin.read(1)

            if echo:
                sys.stdout.write(char)

            sys.stdout.flush()
            return char

        default_color = color

        def should_strip_ansi(
            stream: t.IO[t.Any] | None = None, color: bool | None = None
        ) -> bool:
            if color is None:
                return not default_color
            return not color

        old_visible_prompt_func = termui.visible_prompt_func
        old_hidden_prompt_func = termui.hidden_prompt_func
        old__getchar_func = termui._getchar
        old_should_strip_ansi = utils.should_strip_ansi  # type: ignore
        old__compat_should_strip_ansi = _compat.should_strip_ansi
        termui.visible_prompt_func = visible_input
        termui.hidden_prompt_func = hidden_input
        termui._getchar = _getchar
        utils.should_strip_ansi = should_strip_ansi  # type: ignore
        _compat.should_strip_ansi = should_strip_ansi

        old_env = {}
        try:
            for key, value in env.items():
                old_env[key] = os.environ.get(key)
                if value is None:
                    try:
                        del os.environ[key]
                    except Exception:
                        pass
                else:
                    os.environ[key] = value
            yield (stream_mixer.stdout, stream_mixer.stderr, stream_mixer.output)
        finally:
            for key, value in old_env.items():
                if value is None:
                    try:
                        del os.environ[key]
                    except Exception:
                        pass
                else:
                    os.environ[key] = value
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stdin = old_stdin
            termui.visible_prompt_func = old_visible_prompt_func
            termui.hidden_prompt_func = old_hidden_prompt_func
            termui._getchar = old__getchar_func
            utils.should_strip_ansi = old_should_strip_ansi  # type: ignore
            _compat.should_strip_ansi = old__compat_should_strip_ansi
            formatting.FORCED_WIDTH = old_forced_width

    def invoke(
        self,
        cli: Command,
        args: str | cabc.Sequence[str] | None = None,
        input: str | bytes | t.IO[t.Any] | None = None,
        env: cabc.Mapping[str, str | None] | None = None,
        catch_exceptions: bool | None = None,
        color: bool = False,
        **extra: t.Any,
    ) -> Result:
        """Invokes a command in an isolated environment.  The arguments are
        forwarded directly to the command line script, the `extra` keyword
        arguments are passed to the :meth:`~clickpkg.Command.main` function of
        the command.

        This returns a :class:`Result` object.

        :param cli: the command to invoke
        :param args: the arguments to invoke. It may be given as an iterable
                     or a string. When given as string it will be interpreted
                     as a Unix shell command. More details at
                     :func:`shlex.split`.
        :param input: the input data for `sys.stdin`.
        :param env: the environment overrides.
        :param catch_exceptions: Whether to catch any other exceptions than
                                 ``SystemExit``. If :data:`None`, the value
                                 from :class:`CliRunner` is used.
        :param extra: the keyword arguments to pass to :meth:`main`.
        :param color: whether the output should contain color codes. The
                      application can still override this explicitly.

        .. versionadded:: 8.2
            The result object has the ``output_bytes`` attribute with
            the mix of ``stdout_bytes`` and ``stderr_bytes``, as the user would
            see it in its terminal.

        .. versionchanged:: 8.2
            The result object always returns the ``stderr_bytes`` stream.

        .. versionchanged:: 8.0
            The result object has the ``return_value`` attribute with
            the value returned from the invoked command.

        .. versionchanged:: 4.0
            Added the ``color`` parameter.

        .. versionchanged:: 3.0
            Added the ``catch_exceptions`` parameter.

        .. versionchanged:: 3.0
            The result object has the ``exc_info`` attribute with the
            traceback if available.
        """
        exc_info = None
        if catch_exceptions is None:
            catch_exceptions = self.catch_exceptions

        with self.isolation(input=input, env=env, color=color) as outstreams:
            return_value = None
            exception: BaseException | None = None
            exit_code = 0

            if isinstance(args, str):
                args = shlex.split(args)

            try:
                prog_name = extra.pop("prog_name")
            except KeyError:
                prog_name = self.get_default_prog_name(cli)

            try:
                return_value = cli.main(args=args or (), prog_name=prog_name, **extra)
            except SystemExit as e:
                exc_info = sys.exc_info()
                e_code = t.cast("int | t.Any | None", e.code)

                if e_code is None:
                    e_code = 0

                if e_code != 0:
                    exception = e

                if not isinstance(e_code, int):
                    sys.stdout.write(str(e_code))
                    sys.stdout.write("\n")
                    e_code = 1

                exit_code = e_code

            except Exception as e:
                if not catch_exceptions:
                    raise
                exception = e
                exit_code = 1
                exc_info = sys.exc_info()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                stdout = outstreams[0].getvalue()
                stderr = outstreams[1].getvalue()
                output = outstreams[2].getvalue()

        return Result(
            runner=self,
            stdout_bytes=stdout,
            stderr_bytes=stderr,
            output_bytes=output,
            return_value=return_value,
            exit_code=exit_code,
            exception=exception,
            exc_info=exc_info,  # type: ignore
        )

    @contextlib.contextmanager
    def isolated_filesystem(
        self, temp_dir: str | os.PathLike[str] | None = None
    ) -> cabc.Iterator[str]:
        """A context manager that creates a temporary directory and
        changes the current working directory to it. This isolates tests
        that affect the contents of the CWD to prevent them from
        interfering with each other.

        :param temp_dir: Create the temporary directory under this
            directory. If given, the created directory is not removed
            when exiting.

        .. versionchanged:: 8.0
            Added the ``temp_dir`` parameter.
        """
        cwd = os.getcwd()
        dt = tempfile.mkdtemp(dir=temp_dir)
        os.chdir(dt)

        try:
            yield dt
        finally:
            os.chdir(cwd)

            if temp_dir is None:
                try:
                    shutil.rmtree(dt)
                except OSError:
                    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\debug\__init__.py ===
from __future__ import annotations

import getpass
import hashlib
import json
import os
import pkgutil
import re
import sys
import time
import typing as t
import uuid
from contextlib import ExitStack
from io import BytesIO
from itertools import chain
from multiprocessing import Value
from os.path import basename
from os.path import join
from zlib import adler32

from .._internal import _log
from ..exceptions import NotFound
from ..exceptions import SecurityError
from ..http import parse_cookie
from ..sansio.utils import host_is_trusted
from ..security import gen_salt
from ..utils import send_file
from ..wrappers.request import Request
from ..wrappers.response import Response
from .console import Console
from .tbtools import DebugFrameSummary
from .tbtools import DebugTraceback
from .tbtools import render_console_html

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

# A week
PIN_TIME = 60 * 60 * 24 * 7


def hash_pin(pin: str) -> str:
    return hashlib.sha1(f"{pin} added salt".encode("utf-8", "replace")).hexdigest()[:12]


_machine_id: str | bytes | None = None


def get_machine_id() -> str | bytes | None:
    global _machine_id

    if _machine_id is not None:
        return _machine_id

    def _generate() -> str | bytes | None:
        linux = b""

        # machine-id is stable across boots, boot_id is not.
        for filename in "/etc/machine-id", "/proc/sys/kernel/random/boot_id":
            try:
                with open(filename, "rb") as f:
                    value = f.readline().strip()
            except OSError:
                continue

            if value:
                linux += value
                break

        # Containers share the same machine id, add some cgroup
        # information. This is used outside containers too but should be
        # relatively stable across boots.
        try:
            with open("/proc/self/cgroup", "rb") as f:
                linux += f.readline().strip().rpartition(b"/")[2]
        except OSError:
            pass

        if linux:
            return linux

        # On OS X, use ioreg to get the computer's serial number.
        try:
            # subprocess may not be available, e.g. Google App Engine
            # https://github.com/pallets/werkzeug/issues/925
            from subprocess import PIPE
            from subprocess import Popen

            dump = Popen(
                ["ioreg", "-c", "IOPlatformExpertDevice", "-d", "2"], stdout=PIPE
            ).communicate()[0]
            match = re.search(b'"serial-number" = <([^>]+)', dump)

            if match is not None:
                return match.group(1)
        except (OSError, ImportError):
            pass

        # On Windows, use winreg to get the machine guid.
        if sys.platform == "win32":
            import winreg

            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    "SOFTWARE\\Microsoft\\Cryptography",
                    0,
                    winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
                ) as rk:
                    guid: str | bytes
                    guid_type: int
                    guid, guid_type = winreg.QueryValueEx(rk, "MachineGuid")

                    if guid_type == winreg.REG_SZ:
                        return guid.encode()

                    return guid
            except OSError:
                pass

        return None

    _machine_id = _generate()
    return _machine_id


class _ConsoleFrame:
    """Helper class so that we can reuse the frame console code for the
    standalone console.
    """

    def __init__(self, namespace: dict[str, t.Any]):
        self.console = Console(namespace)
        self.id = 0

    def eval(self, code: str) -> t.Any:
        return self.console.eval(code)


def get_pin_and_cookie_name(
    app: WSGIApplication,
) -> tuple[str, str] | tuple[None, None]:
    """Given an application object this returns a semi-stable 9 digit pin
    code and a random key.  The hope is that this is stable between
    restarts to not make debugging particularly frustrating.  If the pin
    was forcefully disabled this returns `None`.

    Second item in the resulting tuple is the cookie name for remembering.
    """
    pin = os.environ.get("WERKZEUG_DEBUG_PIN")
    rv = None
    num = None

    # Pin was explicitly disabled
    if pin == "off":
        return None, None

    # Pin was provided explicitly
    if pin is not None and pin.replace("-", "").isdecimal():
        # If there are separators in the pin, return it directly
        if "-" in pin:
            rv = pin
        else:
            num = pin

    modname = getattr(app, "__module__", t.cast(object, app).__class__.__module__)
    username: str | None

    try:
        # getuser imports the pwd module, which does not exist in Google
        # App Engine. It may also raise a KeyError if the UID does not
        # have a username, such as in Docker.
        username = getpass.getuser()
    # Python >= 3.13 only raises OSError
    except (ImportError, KeyError, OSError):
        username = None

    mod = sys.modules.get(modname)

    # This information only exists to make the cookie unique on the
    # computer, not as a security feature.
    probably_public_bits = [
        username,
        modname,
        getattr(app, "__name__", type(app).__name__),
        getattr(mod, "__file__", None),
    ]

    # This information is here to make it harder for an attacker to
    # guess the cookie name.  They are unlikely to be contained anywhere
    # within the unauthenticated debug page.
    private_bits = [str(uuid.getnode()), get_machine_id()]

    h = hashlib.sha1()
    for bit in chain(probably_public_bits, private_bits):
        if not bit:
            continue
        if isinstance(bit, str):
            bit = bit.encode()
        h.update(bit)
    h.update(b"cookiesalt")

    cookie_name = f"__wzd{h.hexdigest()[:20]}"

    # If we need to generate a pin we salt it a bit more so that we don't
    # end up with the same value and generate out 9 digits
    if num is None:
        h.update(b"pinsalt")
        num = f"{int(h.hexdigest(), 16):09d}"[:9]

    # Format the pincode in groups of digits for easier remembering if
    # we don't have a result yet.
    if rv is None:
        for group_size in 5, 4, 3:
            if len(num) % group_size == 0:
                rv = "-".join(
                    num[x : x + group_size].rjust(group_size, "0")
                    for x in range(0, len(num), group_size)
                )
                break
        else:
            rv = num

    return rv, cookie_name


class DebuggedApplication:
    """Enables debugging support for a given application::

        from werkzeug.debug import DebuggedApplication
        from myapp import app
        app = DebuggedApplication(app, evalex=True)

    The ``evalex`` argument allows evaluating expressions in any frame
    of a traceback. This works by preserving each frame with its local
    state. Some state, such as context globals, cannot be restored with
    the frame by default. When ``evalex`` is enabled,
    ``environ["werkzeug.debug.preserve_context"]`` will be a callable
    that takes a context manager, and can be called multiple times.
    Each context manager will be entered before evaluating code in the
    frame, then exited again, so they can perform setup and cleanup for
    each call.

    :param app: the WSGI application to run debugged.
    :param evalex: enable exception evaluation feature (interactive
                   debugging).  This requires a non-forking server.
    :param request_key: The key that points to the request object in this
                        environment.  This parameter is ignored in current
                        versions.
    :param console_path: the URL for a general purpose console.
    :param console_init_func: the function that is executed before starting
                              the general purpose console.  The return value
                              is used as initial namespace.
    :param show_hidden_frames: by default hidden traceback frames are skipped.
                               You can show them by setting this parameter
                               to `True`.
    :param pin_security: can be used to disable the pin based security system.
    :param pin_logging: enables the logging of the pin system.

    .. versionchanged:: 2.2
        Added the ``werkzeug.debug.preserve_context`` environ key.
    """

    _pin: str
    _pin_cookie: str

    def __init__(
        self,
        app: WSGIApplication,
        evalex: bool = False,
        request_key: str = "werkzeug.request",
        console_path: str = "/console",
        console_init_func: t.Callable[[], dict[str, t.Any]] | None = None,
        show_hidden_frames: bool = False,
        pin_security: bool = True,
        pin_logging: bool = True,
    ) -> None:
        if not console_init_func:
            console_init_func = None
        self.app = app
        self.evalex = evalex
        self.frames: dict[int, DebugFrameSummary | _ConsoleFrame] = {}
        self.frame_contexts: dict[int, list[t.ContextManager[None]]] = {}
        self.request_key = request_key
        self.console_path = console_path
        self.console_init_func = console_init_func
        self.show_hidden_frames = show_hidden_frames
        self.secret = gen_salt(20)
        self._failed_pin_auth = Value("B")

        self.pin_logging = pin_logging
        if pin_security:
            # Print out the pin for the debugger on standard out.
            if os.environ.get("WERKZEUG_RUN_MAIN") == "true" and pin_logging:
                _log("warning", " * Debugger is active!")
                if self.pin is None:
                    _log("warning", " * Debugger PIN disabled. DEBUGGER UNSECURED!")
                else:
                    _log("info", " * Debugger PIN: %s", self.pin)
        else:
            self.pin = None

        self.trusted_hosts: list[str] = [".localhost", "127.0.0.1"]
        """List of domains to allow requests to the debugger from. A leading dot
        allows all subdomains. This only allows ``".localhost"`` domains by
        default.

        .. versionadded:: 3.0.3
        """

    @property
    def pin(self) -> str | None:
        if not hasattr(self, "_pin"):
            pin_cookie = get_pin_and_cookie_name(self.app)
            self._pin, self._pin_cookie = pin_cookie  # type: ignore
        return self._pin

    @pin.setter
    def pin(self, value: str) -> None:
        self._pin = value

    @property
    def pin_cookie_name(self) -> str:
        """The name of the pin cookie."""
        if not hasattr(self, "_pin_cookie"):
            pin_cookie = get_pin_and_cookie_name(self.app)
            self._pin, self._pin_cookie = pin_cookie  # type: ignore
        return self._pin_cookie

    def debug_application(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterator[bytes]:
        """Run the application and conserve the traceback frames."""
        contexts: list[t.ContextManager[t.Any]] = []

        if self.evalex:
            environ["werkzeug.debug.preserve_context"] = contexts.append

        app_iter = None
        try:
            app_iter = self.app(environ, start_response)
            yield from app_iter
            if hasattr(app_iter, "close"):
                app_iter.close()
        except Exception as e:
            if hasattr(app_iter, "close"):
                app_iter.close()  # type: ignore

            tb = DebugTraceback(e, skip=1, hide=not self.show_hidden_frames)

            for frame in tb.all_frames:
                self.frames[id(frame)] = frame
                self.frame_contexts[id(frame)] = contexts

            is_trusted = bool(self.check_pin_trust(environ))
            html = tb.render_debugger_html(
                evalex=self.evalex and self.check_host_trust(environ),
                secret=self.secret,
                evalex_trusted=is_trusted,
            )
            response = Response(html, status=500, mimetype="text/html")

            try:
                yield from response(environ, start_response)
            except Exception:
                # if we end up here there has been output but an error
                # occurred.  in that situation we can do nothing fancy any
                # more, better log something into the error log and fall
                # back gracefully.
                environ["wsgi.errors"].write(
                    "Debugging middleware caught exception in streamed "
                    "response at a point where response headers were already "
                    "sent.\n"
                )

            environ["wsgi.errors"].write("".join(tb.render_traceback_text()))

    def execute_command(
        self,
        request: Request,
        command: str,
        frame: DebugFrameSummary | _ConsoleFrame,
    ) -> Response:
        """Execute a command in a console."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        contexts = self.frame_contexts.get(id(frame), [])

        with ExitStack() as exit_stack:
            for cm in contexts:
                exit_stack.enter_context(cm)

            return Response(frame.eval(command), mimetype="text/html")

    def display_console(self, request: Request) -> Response:
        """Display a standalone shell."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        if 0 not in self.frames:
            if self.console_init_func is None:
                ns = {}
            else:
                ns = dict(self.console_init_func())
            ns.setdefault("app", self.app)
            self.frames[0] = _ConsoleFrame(ns)
        is_trusted = bool(self.check_pin_trust(request.environ))
        return Response(
            render_console_html(secret=self.secret, evalex_trusted=is_trusted),
            mimetype="text/html",
        )

    def get_resource(self, request: Request, filename: str) -> Response:
        """Return a static resource from the shared folder."""
        path = join("shared", basename(filename))

        try:
            data = pkgutil.get_data(__package__, path)
        except OSError:
            return NotFound()  # type: ignore[return-value]
        else:
            if data is None:
                return NotFound()  # type: ignore[return-value]

            etag = str(adler32(data) & 0xFFFFFFFF)
            return send_file(
                BytesIO(data), request.environ, download_name=filename, etag=etag
            )

    def check_pin_trust(self, environ: WSGIEnvironment) -> bool | None:
        """Checks if the request passed the pin test.  This returns `True` if the
        request is trusted on a pin/cookie basis and returns `False` if not.
        Additionally if the cookie's stored pin hash is wrong it will return
        `None` so that appropriate action can be taken.
        """
        if self.pin is None:
            return True
        val = parse_cookie(environ).get(self.pin_cookie_name)
        if not val or "|" not in val:
            return False
        ts_str, pin_hash = val.split("|", 1)

        try:
            ts = int(ts_str)
        except ValueError:
            return False

        if pin_hash != hash_pin(self.pin):
            return None
        return (time.time() - PIN_TIME) < ts

    def check_host_trust(self, environ: WSGIEnvironment) -> bool:
        return host_is_trusted(environ.get("HTTP_HOST"), self.trusted_hosts)

    def _fail_pin_auth(self) -> None:
        with self._failed_pin_auth.get_lock():
            count = self._failed_pin_auth.value
            self._failed_pin_auth.value = count + 1

        time.sleep(5.0 if count > 5 else 0.5)

    def pin_auth(self, request: Request) -> Response:
        """Authenticates with the pin."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        exhausted = False
        auth = False
        trust = self.check_pin_trust(request.environ)
        pin = t.cast(str, self.pin)

        # If the trust return value is `None` it means that the cookie is
        # set but the stored pin hash value is bad.  This means that the
        # pin was changed.  In this case we count a bad auth and unset the
        # cookie.  This way it becomes harder to guess the cookie name
        # instead of the pin as we still count up failures.
        bad_cookie = False
        if trust is None:
            self._fail_pin_auth()
            bad_cookie = True

        # If we're trusted, we're authenticated.
        elif trust:
            auth = True

        # If we failed too many times, then we're locked out.
        elif self._failed_pin_auth.value > 10:
            exhausted = True

        # Otherwise go through pin based authentication
        else:
            entered_pin = request.args["pin"]

            if entered_pin.strip().replace("-", "") == pin.replace("-", ""):
                self._failed_pin_auth.value = 0
                auth = True
            else:
                self._fail_pin_auth()

        rv = Response(
            json.dumps({"auth": auth, "exhausted": exhausted}),
            mimetype="application/json",
        )
        if auth:
            rv.set_cookie(
                self.pin_cookie_name,
                f"{int(time.time())}|{hash_pin(pin)}",
                httponly=True,
                samesite="Strict",
                secure=request.is_secure,
            )
        elif bad_cookie:
            rv.delete_cookie(self.pin_cookie_name)
        return rv

    def log_pin_request(self, request: Request) -> Response:
        """Log the pin if needed."""
        if not self.check_host_trust(request.environ):
            return SecurityError()  # type: ignore[return-value]

        if self.pin_logging and self.pin is not None:
            _log(
                "info", " * To enable the debugger you need to enter the security pin:"
            )
            _log("info", " * Debugger pin code: %s", self.pin)
        return Response("")

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        """Dispatch the requests."""
        # important: don't ever access a function here that reads the incoming
        # form data!  Otherwise the application won't have access to that data
        # any more!
        request = Request(environ)
        response = self.debug_application
        if request.args.get("__debugger__") == "yes":
            cmd = request.args.get("cmd")
            arg = request.args.get("f")
            secret = request.args.get("s")
            frame = self.frames.get(request.args.get("frm", type=int))  # type: ignore
            if cmd == "resource" and arg:
                response = self.get_resource(request, arg)  # type: ignore
            elif cmd == "pinauth" and secret == self.secret:
                response = self.pin_auth(request)  # type: ignore
            elif cmd == "printpin" and secret == self.secret:
                response = self.log_pin_request(request)  # type: ignore
            elif (
                self.evalex
                and cmd is not None
                and frame is not None
                and self.secret == secret
                and self.check_pin_trust(environ)
            ):
                response = self.execute_command(request, cmd, frame)  # type: ignore
        elif (
            self.evalex
            and self.console_path is not None
            and request.path == self.console_path
        ):
            response = self.display_console(request)  # type: ignore
        return response(environ, start_response)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\base_quantizer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging
from typing import Any

import numpy as np
import onnx
import onnx.numpy_helper

try:
    from onnx.reference.op_run import to_array_extended
except ImportError:
    # old version of onnx.
    to_array_extended = None

from .calibrate import TensorData
from .onnx_model import ONNXModel
from .quant_utils import (
    DEQUANT_OP_NAME,
    ONNX_TYPE_TO_NP_TYPE,
    QUANT_OP_NAME,
    TENSOR_NAME_QUANT_SUFFIX,
    find_by_name,
    model_has_infer_metadata,
    normalize_axis,
    pack_bytes_to_4bit,
    quantize_data,
    quantize_nparray,
    save_and_reload_model_with_shape_infer,
    tensor_proto_to_array,
)
from .tensor_quant_overrides import TensorQuantOverridesHelper


class QuantizationParams:
    def __init__(self, **data: dict[str, Any]):
        self.data = {}
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"Keys must be strings not {type(k)} for k={k!r}.")
            if k != "axis" and not isinstance(v, (int, str, np.ndarray)):
                raise TypeError(f"Values must be numpy arrays, int, float, str not {type(v)} for k={k!r}.")
            if k == "axis" and not isinstance(v, int) and v is not None:
                raise TypeError(f"Axis value must be an int or None, not {type(v)}.")
            if k == "scale" and v.dtype not in (np.float32, np.float16):
                raise ValueError(f"scale must a float32 or float16 numpy element but is {v.dtype} for k={k!r}")
            self.data[k] = v

    def get(self, key, default_value=None):
        return self.data.get(key, default_value)

    def __iter__(self):
        yield from self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)


class BaseQuantizer:
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        if not model_has_infer_metadata(model):
            model = save_and_reload_model_with_shape_infer(model)
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})

        self.model = ONNXModel(model)
        self.per_channel = per_channel  # weight-pack per channel
        self.reduce_range = reduce_range

        self.extra_options = extra_options if extra_options else {}
        self.enable_subgraph_quantization = (
            "EnableSubgraph" in self.extra_options and self.extra_options["EnableSubgraph"]
        )
        self.parent = None
        self.force_quantize_no_input_check = (
            "ForceQuantizeNoInputCheck" in self.extra_options and self.extra_options["ForceQuantizeNoInputCheck"]
        )

        # If user does not explicitly set "WeightSymmetric", then the weight's quantization type determines
        # the symmetry (i.e., signed integer types will use symmetric quantization). See `def is_weight_symmetric()`
        self._is_weight_symmetric: bool | None = self.extra_options.get("WeightSymmetric", None)
        self.is_activation_symmetric = self.extra_options.get("ActivationSymmetric", False)
        self.min_real_range = self.extra_options.get("MinimumRealRange")

        self.activation_qType = getattr(activation_qType, "tensor_type", activation_qType)
        self.weight_qType = getattr(weight_qType, "tensor_type", weight_qType)

        """
            Dictionary specifying the min and max values for tensors. It has following format:
                {
                    "param_name": [min, max]
                }
            example:
                {
                    'Conv_3:0': [np.float32(0), np.float32(0.5)],
                    'Conv_4:0': [np.float32(1), np.float32(3.5)]
                }
        """
        if tensors_range is not None and any(not isinstance(t, TensorData) for t in tensors_range.values()):
            raise TypeError(
                f"tensors_range contains unexpected types { {type(v) for v in tensors_range.values()} }, not TensorData."
            )
        self.tensors_range = tensors_range
        self.nodes_to_quantize = nodes_to_quantize  # specific nodes to quantize
        self.nodes_to_exclude = nodes_to_exclude  # specific nodes to exclude
        self.op_types_to_quantize = op_types_to_quantize

        self.opset_version = self.check_opset_version()

        # Get tensor-level quantization overrides and ensure they are valid.
        self.tensor_quant_overrides = TensorQuantOverridesHelper(self.extra_options.get("TensorQuantOverrides", {}))

        self.initializers = {initzer.name: initzer for initzer in self.model.initializer()}
        overrides_valid, overrides_err = self.tensor_quant_overrides.is_valid(
            self.initializers, self.value_infos.keys(), activation_qType
        )
        if not overrides_valid:
            raise ValueError(overrides_err)

        self.tensor_quant_override_qtypes = self.tensor_quant_overrides.get_quant_types()

    def is_weight_symmetric(self, weight_quant_type: onnx.TensorProto.DataType) -> bool:
        if self._is_weight_symmetric is not None:
            return self._is_weight_symmetric  # Return value explicitly set by user.
        return weight_quant_type in (
            onnx.TensorProto.INT4,
            onnx.TensorProto.INT8,
            onnx.TensorProto.INT16,
            onnx.TensorProto.FLOAT8E4M3FN,
        )

    def quantize_model(self):
        raise NotImplementedError

    def is_input_a_initializer(self, input_name):
        initializer = find_by_name(input_name, self.model.initializer())
        return initializer is not None

    def is_per_channel(self):
        return self.per_channel

    def is_valid_quantize_weight(self, weight_name):
        weight = find_by_name(weight_name, self.model.initializer())
        if weight is not None:
            return weight.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)
        if (not self.enable_subgraph_quantization) or (self.parent is None):
            return False
        return self.parent.is_valid_quantize_weight(weight_name)

    def should_quantize_node(self, node):
        if (
            self.nodes_to_quantize is not None
            and len(self.nodes_to_quantize) != 0
            and node.name not in self.nodes_to_quantize
        ):
            return False

        if node.op_type not in self.op_types_to_quantize:
            return False

        if node.op_type in (DEQUANT_OP_NAME, QUANT_OP_NAME):
            return False

        if self.nodes_to_exclude is not None and node.name in self.nodes_to_exclude:
            return False

        return True

    def check_opset_version(self):
        ai_onnx_domain = [
            opset for opset in self.model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        if len(ai_onnx_domain) != 1:
            raise ValueError("Failed to find proper ai.onnx domain")
        opset_version = ai_onnx_domain[0].version

        if opset_version == 10:
            logging.warning(
                f"The original model opset version is {opset_version}, which does not support node fusions. Please update the model to opset >= 11 for better performance."
            )
            return 10

        if opset_version < 10:
            logging.warning(
                f"The original model opset version is {opset_version}, which does not support quantization. Please update the model to opset >= 11. Updating the model automatically to opset 11. Please verify the quantized model."
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
            opset_version = 11

        if opset_version < 19 and self.weight_qType == onnx.TensorProto.FLOAT8E4M3FN:
            logging.warning(
                f"The original model opset version is {opset_version}, which does not support quantization to float 8. "
                "Please update the model to opset >= 19. Updating the model automatically to opset 19. "
                "Please verify the quantized model."
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 19)])
            self.model.model.ir_version = 9
            opset_version = 19

        return opset_version

    def quantize_bias_static_impl(self, bias_name, input_scale, weight_scale, beta=1.0):
        """
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """

        # get bias
        bias_initializer = find_by_name(bias_name, self.model.initializer())
        bias_data = tensor_proto_to_array(bias_initializer)
        quantized_bias_name = bias_name + TENSOR_NAME_QUANT_SUFFIX

        # quantize bias
        if self.weight_qType == onnx.TensorProto.FLOAT8E4M3FN:
            data = np.asarray(bias_data)
            if data.dtype == np.float16:
                node_qtype = onnx.TensorProto.FLOAT16
            elif data.dtype == np.float32:
                node_qtype = onnx.TensorProto.FLOAT
            else:
                raise TypeError(f"Only float16 or float32 are supported with float 8 but bias dtype is {data.dtype}.")
            quantized_data = data.astype(np.float32)
            bias_scale = np.array([1], dtype=quantized_data.dtype)
            bias_scale_data = bias_scale.reshape(-1)
            packed_bias_initializer = onnx.numpy_helper.from_array(quantized_data, quantized_bias_name)
            self.model.initializer_extend([packed_bias_initializer])
            node_type = "Cast"
        else:
            # calculate scale for bias
            # TODO: This formula should be explained including why the scale is not estimated for the bias as well.
            bias_scale = input_scale * weight_scale * beta

            # Quantize by dividing by bias_scale
            quantized_data = np.asarray(bias_data, dtype=np.float64) / np.asarray(bias_scale, dtype=np.float64)
            quantized_data = quantized_data.round()

            # Clip quantized data to the range of a int32
            int32_min = np.float64(np.iinfo(np.int32).min)
            int32_max = np.float64(np.iinfo(np.int32).max)
            if np.any(quantized_data < int32_min) or np.any(quantized_data > int32_max):
                logging.warning(
                    f"Quantized bias `{bias_name}` exceeds the range of a int32. The bias scale is too small."
                )

            quantized_data = np.clip(quantized_data, int32_min, int32_max).astype(np.int32)

            # update bias initializer
            bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
            packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
            self.model.initializer_extend([packed_bias_initializer])

            # Bias's scale dtype should match the original bias data's unquantized type (float32 or float16).
            bias_scale_data = np.asarray(bias_scale, dtype=bias_data.dtype).reshape(-1)
            node_type = "DequantizeLinear"
            node_qtype = self.weight_qType

        # update scale initializer
        quantized_bias_scale_name = quantized_bias_name + "_scale"
        packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        self.model.initializer_extend([packed_bias_scale_initializer])

        # update zero initializer
        if self.weight_qType == onnx.TensorProto.FLOAT8E4M3FN:
            tensor_type = self.weight_qType
        else:
            tensor_type = onnx.TensorProto.INT32

        quantized_bias_zp_name = quantized_bias_name + "_zero_point"
        if self.weight_qType == onnx.TensorProto.FLOAT8E4M3FN:
            packed_bias_zp_initializer = onnx.helper.make_tensor(quantized_bias_zp_name, self.weight_qType, [1], [0.0])
        elif bias_scale.size > 1:
            bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
            packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        else:
            packed_bias_zp_initializer = onnx.helper.make_tensor(quantized_bias_zp_name, tensor_type, [], [0])
        self.model.initializer_extend([packed_bias_zp_initializer])

        return (
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            bias_scale_data,
            node_type,
            node_qtype,
        )

    def quantize_initializer_impl(self, weight, qType, reduce_range=False, keep_float_weight=False):
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                  If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        # TODO(adrianlizarraga): This function is now only used by onnx_quantizer.py, so move it there.
        q_weight_name = weight.name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Quantize weight data. Use quantization overrides if provided by the user.
        weight_data = tensor_proto_to_array(weight)
        quant_overrides = self.tensor_quant_overrides.get_per_tensor_overrides(weight.name, default_val={})
        if "quant_type" in quant_overrides:
            qType = quant_overrides["quant_type"].tensor_type  # noqa: N806

        if "scale" in quant_overrides and "zero_point" in quant_overrides:
            zero_point = np.array(quant_overrides["zero_point"], dtype=ONNX_TYPE_TO_NP_TYPE[qType])
            scale = np.array(quant_overrides["scale"])
            q_weight_data = quantize_nparray(qType, weight_data.flatten(), scale, zero_point)
            assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
            assert zero_point.dtype != np.float32 and zero_point.dtype != np.float16, (
                f"Unexpected dtype {zero_point.dtype}"
            )
            assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"

        else:
            symmetric = self.is_weight_symmetric(qType) if qType == self.weight_qType else self.is_activation_symmetric
            zero_point, scale, q_weight_data = quantize_data(
                weight_data.flatten(),
                qType,
                quant_overrides.get("symmetric", symmetric),
                reduce_range=quant_overrides.get("reduce_range", self.reduce_range and reduce_range),
                min_real_range=self.min_real_range,
                rmin_override=quant_overrides.get("rmin"),
                rmax_override=quant_overrides.get("rmax"),
            )

            assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
            assert zero_point.dtype != np.float32 and zero_point.dtype != np.float16, (
                f"Unexpected dtype {zero_point.dtype}"
            )
            assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"

        scale_dtype = weight.data_type
        scale_initializer = onnx.helper.make_tensor(scale_name, scale_dtype, [], scale.reshape((-1,)).tolist())
        zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], zero_point.reshape((-1,)).tolist())
        self.model.initializer_extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            if self.weight_qType == onnx.TensorProto.FLOAT8E4M3FN:
                q_weight_initializer = onnx.TensorProto()
                q_weight_initializer.data_type = self.weight_qType
                q_weight_initializer.dims.extend(weight.dims)
                q_weight_initializer.name = q_weight_name
                # Do not remove .flatten().copy() numpy is not clear about data persistence.
                q_weight_initializer.raw_data = q_weight_data.flatten().copy().tobytes()
                if to_array_extended is not None:
                    # This test should not be needed but it helped catch some issues
                    # with data persistence and tobytes.
                    check = to_array_extended(q_weight_initializer)
                    if check.shape != weight_data.shape or check.tobytes() != q_weight_data.tobytes():
                        raise RuntimeError(
                            f"The initializer of shape {weight_data.shape} could not be created, expecting "
                            f"{q_weight_data.tobytes()[:10]}, got {check.tobytes()[:10]} and shape={weight.shape}"
                            f"\nraw={str(q_weight_initializer)[:200]}."
                        )
            elif qType in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4):
                if q_weight_data.dtype not in (np.int8, np.uint8):
                    raise RuntimeError(
                        f"Quantized weights for {q_weight_name} must be 8-bit before packing as 4-bit values."
                    )

                # We do not use onnx.helper.pack_float32_to_4bit() due to performance.
                # This can be the difference between a large model taking 30 minutes to quantize vs 5 minutes.
                packed_data = bytes(pack_bytes_to_4bit(q_weight_data.tobytes()))

                # We only use onnx.helper.make_tensor with raw data due to bug: https://github.com/onnx/onnx/pull/6161
                q_weight_initializer = onnx.helper.make_tensor(q_weight_name, qType, weight.dims, packed_data, raw=True)
            else:
                q_weight_data = np.asarray(q_weight_data, dtype=onnx.helper.tensor_dtype_to_np_dtype(qType)).reshape(
                    weight.dims
                )
                q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)
            self.model.initializer_extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel_impl(
        self,
        weight_name,
        weight_qType,
        channel_axis,
        reduce_range=True,
        keep_float_weight=False,
    ):
        # TODO(adrianlizarraga): This function is now only used by onnx_quantizer.py, so move it there.
        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = tensor_proto_to_array(initializer)
        weights_rank = len(weights.shape)
        is_axis_valid, axis_norm = normalize_axis(channel_axis, weights_rank)
        if not is_axis_valid:
            raise ValueError(
                f"Weight {weight_name} has a per-channel axis with value {channel_axis} that is "
                f"out-of-bounds for rank {weights_rank}"
            )

        channel_axis = axis_norm
        channel_count = weights.shape[channel_axis]
        quant_overrides_for_channels = self.tensor_quant_overrides.get_per_channel_overrides(
            weight_name, default_val=[{"axis": channel_axis}]
        )

        num_channel_overrides = len(quant_overrides_for_channels)
        if num_channel_overrides != 1 and num_channel_overrides != channel_count:
            raise ValueError(
                f"Per-channel tensor quantization overrides for {weight_name} must have "
                f"either 1 or {channel_count} elements in the list of dictionaries."
            )

        is_axis_override_valid, axis_override = normalize_axis(quant_overrides_for_channels[0]["axis"], weights_rank)
        if not is_axis_override_valid or axis_override != channel_axis:
            raise ValueError(
                f"Tensor quantization overrides for {weight_name} specify an unexpected axis. "
                f"Expected {channel_axis}, but got {quant_overrides_for_channels[0]['axis']}."
            )

        # If user provides per-channel quantization overrides, all channels must use the same quant_type,
        # axis, symmetric, and reduce_range values. So, just use the first channel's values.
        if "quant_type" in quant_overrides_for_channels[0]:
            weight_qType = quant_overrides_for_channels[0]["quant_type"].tensor_type  # noqa: N806

        symmetric = quant_overrides_for_channels[0].get("symmetric", self.is_weight_symmetric(weight_qType))
        reduce_range = quant_overrides_for_channels[0].get("reduce_range", self.reduce_range and reduce_range)
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        weights_shape = list(weights.shape)
        reshape_dims = list(weights_shape)  # deep copy
        reshape_dims[channel_axis] = 1  # only one per channel for reshape
        for i in range(channel_count):
            per_channel_data = weights.take(i, channel_axis)
            channel_override_index = i if i < num_channel_overrides else 0
            channel_quant_overrides = quant_overrides_for_channels[channel_override_index]

            if "scale" in channel_quant_overrides and "zero_point" in channel_quant_overrides:
                zero_point = np.array(channel_quant_overrides["zero_point"], dtype=ONNX_TYPE_TO_NP_TYPE[weight_qType])
                scale = np.array(channel_quant_overrides["scale"])
                quantized_per_channel_data = quantize_nparray(
                    weight_qType, per_channel_data.flatten(), scale, zero_point
                )
                assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
                assert zero_point.dtype != np.float32 and zero_point.dtype != np.float16, (
                    f"Unexpected dtype {zero_point.dtype}"
                )
                assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"
                assert isinstance(quantized_per_channel_data, np.ndarray), (
                    f"Unexpected type {type(quantized_per_channel_data)}"
                )

            else:
                zero_point, scale, quantized_per_channel_data = quantize_data(
                    per_channel_data.flatten(),
                    weight_qType,
                    symmetric,
                    reduce_range=reduce_range,
                    min_real_range=self.min_real_range,
                    rmin_override=channel_quant_overrides.get("rmin"),
                    rmax_override=channel_quant_overrides.get("rmax"),
                )

                assert isinstance(zero_point, np.ndarray), f"Unexpected type {type(zero_point)}"
                assert zero_point.dtype != np.float32 and zero_point.dtype != np.float16, (
                    f"Unexpected dtype {zero_point.dtype}"
                )
                assert isinstance(scale, np.ndarray), f"Unexpected type {type(scale)}"
                assert isinstance(quantized_per_channel_data, np.ndarray), (
                    f"Unexpected type {type(quantized_per_channel_data)}"
                )

            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(np.asarray(quantized_per_channel_data).reshape(reshape_dims))

        # combine per_channel_data into one
        quantized_weights = np.concatenate(quantized_per_channel_data_list, channel_axis)
        q_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight_name + "_zero_point"
        scale_name = weight_name + "_scale"

        # Update packed weight, zero point, and scale initializers
        zero_scale_shape = [initializer.dims[channel_axis]]
        scale_initializer = onnx.helper.make_tensor(
            scale_name, initializer.data_type, zero_scale_shape, np.hstack(scale_list).tolist()
        )
        zero_initializer = onnx.helper.make_tensor(
            zp_name, weight_qType, zero_scale_shape, np.hstack(zero_point_list).tolist()
        )

        self.model.initializer_extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            if weight_qType in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4):
                if quantized_weights.dtype not in (np.int8, np.uint8):
                    raise RuntimeError(
                        f"Quantized weights for {q_weight_name} must be 8-bit before packing as 4-bit values."
                    )

                # We do not use onnx.helper.pack_float32_to_4bit() due to performance.
                # This can be the difference between a large model taking 30 minutes to quantize vs 5 minutes.
                packed_data = bytes(pack_bytes_to_4bit(quantized_weights.tobytes()))

                # We only use onnx.helper.make_tensor with raw data due to bug: https://github.com/onnx/onnx/pull/6161
                q_weight_initializer = onnx.helper.make_tensor(
                    q_weight_name, weight_qType, weights_shape, packed_data, raw=True
                )
                self.model.initializer_extend([q_weight_initializer])
            else:
                quantized_weights = np.asarray(
                    quantized_weights,
                    dtype=onnx.helper.tensor_dtype_to_np_dtype(weight_qType),
                ).reshape(initializer.dims)
                q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
                self.model.initializer_extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def adjust_tensor_ranges(self):
        if self.tensors_range is None:
            return

        for node in self.model.nodes():
            # adjust tensor_ranges for input of Clip and Relu node
            if node.op_type in ["Clip", "Relu"]:
                if not self.should_quantize_node(node):
                    continue
                if len(self.model.input_name_to_nodes()[node.input[0]]) != 1:
                    continue
                if node.input[0] not in self.tensors_range or node.output[0] not in self.tensors_range:
                    continue
                td = self.tensors_range[node.output[0]]
                if not isinstance(td, TensorData):
                    raise TypeError(f"Unexpected type {type(td)} for {node.output[0]!r}.")
                self.tensors_range[node.input[0]] = td
            # Adjust Softmax to range from 0.0 to 1.0
            elif node.op_type == "Softmax":
                if not self.should_quantize_node(node):
                    continue
                self.tensors_range[node.output[0]] = TensorData(lowest=np.float32(0.0), highest=np.float32(1.0))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\declarative\extensions.py ===
# ext/declarative/extensions.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""Public API functions and helpers for declarative."""
from __future__ import annotations

import collections
import contextlib
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from typing import Union

from ... import exc as sa_exc
from ...engine import Connection
from ...engine import Engine
from ...orm import exc as orm_exc
from ...orm import relationships
from ...orm.base import _mapper_or_none
from ...orm.clsregistry import _resolver
from ...orm.decl_base import _DeferredMapperConfig
from ...orm.util import polymorphic_union
from ...schema import Table
from ...util import OrderedDict

if TYPE_CHECKING:
    from ...sql.schema import MetaData


class ConcreteBase:
    """A helper class for 'concrete' declarative mappings.

    :class:`.ConcreteBase` will use the :func:`.polymorphic_union`
    function automatically, against all tables mapped as a subclass
    to this class.   The function is called via the
    ``__declare_last__()`` function, which is essentially
    a hook for the :meth:`.after_configured` event.

    :class:`.ConcreteBase` produces a mapped
    table for the class itself.  Compare to :class:`.AbstractConcreteBase`,
    which does not.

    Example::

        from sqlalchemy.ext.declarative import ConcreteBase


        class Employee(ConcreteBase, Base):
            __tablename__ = "employee"
            employee_id = Column(Integer, primary_key=True)
            name = Column(String(50))
            __mapper_args__ = {
                "polymorphic_identity": "employee",
                "concrete": True,
            }


        class Manager(Employee):
            __tablename__ = "manager"
            employee_id = Column(Integer, primary_key=True)
            name = Column(String(50))
            manager_data = Column(String(40))
            __mapper_args__ = {
                "polymorphic_identity": "manager",
                "concrete": True,
            }

    The name of the discriminator column used by :func:`.polymorphic_union`
    defaults to the name ``type``.  To suit the use case of a mapping where an
    actual column in a mapped table is already named ``type``, the
    discriminator name can be configured by setting the
    ``_concrete_discriminator_name`` attribute::

        class Employee(ConcreteBase, Base):
            _concrete_discriminator_name = "_concrete_discriminator"

    .. versionadded:: 1.3.19 Added the ``_concrete_discriminator_name``
       attribute to :class:`_declarative.ConcreteBase` so that the
       virtual discriminator column name can be customized.

    .. versionchanged:: 1.4.2 The ``_concrete_discriminator_name`` attribute
       need only be placed on the basemost class to take correct effect for
       all subclasses.   An explicit error message is now raised if the
       mapped column names conflict with the discriminator name, whereas
       in the 1.3.x series there would be some warnings and then a non-useful
       query would be generated.

    .. seealso::

        :class:`.AbstractConcreteBase`

        :ref:`concrete_inheritance`


    """

    @classmethod
    def _create_polymorphic_union(cls, mappers, discriminator_name):
        return polymorphic_union(
            OrderedDict(
                (mp.polymorphic_identity, mp.local_table) for mp in mappers
            ),
            discriminator_name,
            "pjoin",
        )

    @classmethod
    def __declare_first__(cls):
        m = cls.__mapper__
        if m.with_polymorphic:
            return

        discriminator_name = (
            getattr(cls, "_concrete_discriminator_name", None) or "type"
        )

        mappers = list(m.self_and_descendants)
        pjoin = cls._create_polymorphic_union(mappers, discriminator_name)
        m._set_with_polymorphic(("*", pjoin))
        m._set_polymorphic_on(pjoin.c[discriminator_name])


class AbstractConcreteBase(ConcreteBase):
    """A helper class for 'concrete' declarative mappings.

    :class:`.AbstractConcreteBase` will use the :func:`.polymorphic_union`
    function automatically, against all tables mapped as a subclass
    to this class.   The function is called via the
    ``__declare_first__()`` function, which is essentially
    a hook for the :meth:`.before_configured` event.

    :class:`.AbstractConcreteBase` applies :class:`_orm.Mapper` for its
    immediately inheriting class, as would occur for any other
    declarative mapped class. However, the :class:`_orm.Mapper` is not
    mapped to any particular :class:`.Table` object.  Instead, it's
    mapped directly to the "polymorphic" selectable produced by
    :func:`.polymorphic_union`, and performs no persistence operations on its
    own.  Compare to :class:`.ConcreteBase`, which maps its
    immediately inheriting class to an actual
    :class:`.Table` that stores rows directly.

    .. note::

        The :class:`.AbstractConcreteBase` delays the mapper creation of the
        base class until all the subclasses have been defined,
        as it needs to create a mapping against a selectable that will include
        all subclass tables.  In order to achieve this, it waits for the
        **mapper configuration event** to occur, at which point it scans
        through all the configured subclasses and sets up a mapping that will
        query against all subclasses at once.

        While this event is normally invoked automatically, in the case of
        :class:`.AbstractConcreteBase`, it may be necessary to invoke it
        explicitly after **all** subclass mappings are defined, if the first
        operation is to be a query against this base class. To do so, once all
        the desired classes have been configured, the
        :meth:`_orm.registry.configure` method on the :class:`_orm.registry`
        in use can be invoked, which is available in relation to a particular
        declarative base class::

            Base.registry.configure()

    Example::

        from sqlalchemy.orm import DeclarativeBase
        from sqlalchemy.ext.declarative import AbstractConcreteBase


        class Base(DeclarativeBase):
            pass


        class Employee(AbstractConcreteBase, Base):
            pass


        class Manager(Employee):
            __tablename__ = "manager"
            employee_id = Column(Integer, primary_key=True)
            name = Column(String(50))
            manager_data = Column(String(40))

            __mapper_args__ = {
                "polymorphic_identity": "manager",
                "concrete": True,
            }


        Base.registry.configure()

    The abstract base class is handled by declarative in a special way;
    at class configuration time, it behaves like a declarative mixin
    or an ``__abstract__`` base class.   Once classes are configured
    and mappings are produced, it then gets mapped itself, but
    after all of its descendants.  This is a very unique system of mapping
    not found in any other SQLAlchemy API feature.

    Using this approach, we can specify columns and properties
    that will take place on mapped subclasses, in the way that
    we normally do as in :ref:`declarative_mixins`::

        from sqlalchemy.ext.declarative import AbstractConcreteBase


        class Company(Base):
            __tablename__ = "company"
            id = Column(Integer, primary_key=True)


        class Employee(AbstractConcreteBase, Base):
            strict_attrs = True

            employee_id = Column(Integer, primary_key=True)

            @declared_attr
            def company_id(cls):
                return Column(ForeignKey("company.id"))

            @declared_attr
            def company(cls):
                return relationship("Company")


        class Manager(Employee):
            __tablename__ = "manager"

            name = Column(String(50))
            manager_data = Column(String(40))

            __mapper_args__ = {
                "polymorphic_identity": "manager",
                "concrete": True,
            }


        Base.registry.configure()

    When we make use of our mappings however, both ``Manager`` and
    ``Employee`` will have an independently usable ``.company`` attribute::

        session.execute(select(Employee).filter(Employee.company.has(id=5)))

    :param strict_attrs: when specified on the base class, "strict" attribute
     mode is enabled which attempts to limit ORM mapped attributes on the
     base class to only those that are immediately present, while still
     preserving "polymorphic" loading behavior.

     .. versionadded:: 2.0

    .. seealso::

        :class:`.ConcreteBase`

        :ref:`concrete_inheritance`

        :ref:`abstract_concrete_base`

    """

    __no_table__ = True

    @classmethod
    def __declare_first__(cls):
        cls._sa_decl_prepare_nocascade()

    @classmethod
    def _sa_decl_prepare_nocascade(cls):
        if getattr(cls, "__mapper__", None):
            return

        to_map = _DeferredMapperConfig.config_for_cls(cls)

        # can't rely on 'self_and_descendants' here
        # since technically an immediate subclass
        # might not be mapped, but a subclass
        # may be.
        mappers = []
        stack = list(cls.__subclasses__())
        while stack:
            klass = stack.pop()
            stack.extend(klass.__subclasses__())
            mn = _mapper_or_none(klass)
            if mn is not None:
                mappers.append(mn)

        discriminator_name = (
            getattr(cls, "_concrete_discriminator_name", None) or "type"
        )
        pjoin = cls._create_polymorphic_union(mappers, discriminator_name)

        # For columns that were declared on the class, these
        # are normally ignored with the "__no_table__" mapping,
        # unless they have a different attribute key vs. col name
        # and are in the properties argument.
        # In that case, ensure we update the properties entry
        # to the correct column from the pjoin target table.
        declared_cols = set(to_map.declared_columns)
        declared_col_keys = {c.key for c in declared_cols}
        for k, v in list(to_map.properties.items()):
            if v in declared_cols:
                to_map.properties[k] = pjoin.c[v.key]
                declared_col_keys.remove(v.key)

        to_map.local_table = pjoin

        strict_attrs = cls.__dict__.get("strict_attrs", False)

        m_args = to_map.mapper_args_fn or dict

        def mapper_args():
            args = m_args()
            args["polymorphic_on"] = pjoin.c[discriminator_name]
            args["polymorphic_abstract"] = True
            if strict_attrs:
                args["include_properties"] = (
                    set(pjoin.primary_key)
                    | declared_col_keys
                    | {discriminator_name}
                )
                args["with_polymorphic"] = ("*", pjoin)
            return args

        to_map.mapper_args_fn = mapper_args

        to_map.map()

        stack = [cls]
        while stack:
            scls = stack.pop(0)
            stack.extend(scls.__subclasses__())
            sm = _mapper_or_none(scls)
            if sm and sm.concrete and sm.inherits is None:
                for sup_ in scls.__mro__[1:]:
                    sup_sm = _mapper_or_none(sup_)
                    if sup_sm:
                        sm._set_concrete_base(sup_sm)
                        break

    @classmethod
    def _sa_raise_deferred_config(cls):
        raise orm_exc.UnmappedClassError(
            cls,
            msg="Class %s is a subclass of AbstractConcreteBase and "
            "has a mapping pending until all subclasses are defined. "
            "Call the sqlalchemy.orm.configure_mappers() function after "
            "all subclasses have been defined to "
            "complete the mapping of this class."
            % orm_exc._safe_cls_name(cls),
        )


class DeferredReflection:
    """A helper class for construction of mappings based on
    a deferred reflection step.

    Normally, declarative can be used with reflection by
    setting a :class:`_schema.Table` object using autoload_with=engine
    as the ``__table__`` attribute on a declarative class.
    The caveat is that the :class:`_schema.Table` must be fully
    reflected, or at the very least have a primary key column,
    at the point at which a normal declarative mapping is
    constructed, meaning the :class:`_engine.Engine` must be available
    at class declaration time.

    The :class:`.DeferredReflection` mixin moves the construction
    of mappers to be at a later point, after a specific
    method is called which first reflects all :class:`_schema.Table`
    objects created so far.   Classes can define it as such::

        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.ext.declarative import DeferredReflection

        Base = declarative_base()


        class MyClass(DeferredReflection, Base):
            __tablename__ = "mytable"

    Above, ``MyClass`` is not yet mapped.   After a series of
    classes have been defined in the above fashion, all tables
    can be reflected and mappings created using
    :meth:`.prepare`::

        engine = create_engine("someengine://...")
        DeferredReflection.prepare(engine)

    The :class:`.DeferredReflection` mixin can be applied to individual
    classes, used as the base for the declarative base itself,
    or used in a custom abstract class.   Using an abstract base
    allows that only a subset of classes to be prepared for a
    particular prepare step, which is necessary for applications
    that use more than one engine.  For example, if an application
    has two engines, you might use two bases, and prepare each
    separately, e.g.::

        class ReflectedOne(DeferredReflection, Base):
            __abstract__ = True


        class ReflectedTwo(DeferredReflection, Base):
            __abstract__ = True


        class MyClass(ReflectedOne):
            __tablename__ = "mytable"


        class MyOtherClass(ReflectedOne):
            __tablename__ = "myothertable"


        class YetAnotherClass(ReflectedTwo):
            __tablename__ = "yetanothertable"


        # ... etc.

    Above, the class hierarchies for ``ReflectedOne`` and
    ``ReflectedTwo`` can be configured separately::

        ReflectedOne.prepare(engine_one)
        ReflectedTwo.prepare(engine_two)

    .. seealso::

        :ref:`orm_declarative_reflected_deferred_reflection` - in the
        :ref:`orm_declarative_table_config_toplevel` section.

    """

    @classmethod
    def prepare(
        cls, bind: Union[Engine, Connection], **reflect_kw: Any
    ) -> None:
        r"""Reflect all :class:`_schema.Table` objects for all current
        :class:`.DeferredReflection` subclasses

        :param bind: :class:`_engine.Engine` or :class:`_engine.Connection`
         instance

         ..versionchanged:: 2.0.16 a :class:`_engine.Connection` is also
         accepted.

        :param \**reflect_kw: additional keyword arguments passed to
         :meth:`_schema.MetaData.reflect`, such as
         :paramref:`_schema.MetaData.reflect.views`.

         .. versionadded:: 2.0.16

        """

        to_map = _DeferredMapperConfig.classes_for_base(cls)

        metadata_to_table = collections.defaultdict(set)

        # first collect the primary __table__ for each class into a
        # collection of metadata/schemaname -> table names
        for thingy in to_map:
            if thingy.local_table is not None:
                metadata_to_table[
                    (thingy.local_table.metadata, thingy.local_table.schema)
                ].add(thingy.local_table.name)

        # then reflect all those tables into their metadatas

        if isinstance(bind, Connection):
            conn = bind
            ctx = contextlib.nullcontext(enter_result=conn)
        elif isinstance(bind, Engine):
            ctx = bind.connect()
        else:
            raise sa_exc.ArgumentError(
                f"Expected Engine or Connection, got {bind!r}"
            )

        with ctx as conn:
            for (metadata, schema), table_names in metadata_to_table.items():
                metadata.reflect(
                    conn,
                    only=table_names,
                    schema=schema,
                    extend_existing=True,
                    autoload_replace=False,
                    **reflect_kw,
                )

            metadata_to_table.clear()

            # .map() each class, then go through relationships and look
            # for secondary
            for thingy in to_map:
                thingy.map()

                mapper = thingy.cls.__mapper__
                metadata = mapper.class_.metadata

                for rel in mapper._props.values():
                    if (
                        isinstance(rel, relationships.RelationshipProperty)
                        and rel._init_args.secondary._is_populated()
                    ):
                        secondary_arg = rel._init_args.secondary

                        if isinstance(secondary_arg.argument, Table):
                            secondary_table = secondary_arg.argument
                            metadata_to_table[
                                (
                                    secondary_table.metadata,
                                    secondary_table.schema,
                                )
                            ].add(secondary_table.name)
                        elif isinstance(secondary_arg.argument, str):
                            _, resolve_arg = _resolver(rel.parent.class_, rel)

                            resolver = resolve_arg(
                                secondary_arg.argument, True
                            )
                            metadata_to_table[
                                (metadata, thingy.local_table.schema)
                            ].add(secondary_arg.argument)

                            resolver._resolvers += (
                                cls._sa_deferred_table_resolver(metadata),
                            )

                            secondary_arg.argument = resolver()

            for (metadata, schema), table_names in metadata_to_table.items():
                metadata.reflect(
                    conn,
                    only=table_names,
                    schema=schema,
                    extend_existing=True,
                    autoload_replace=False,
                )

    @classmethod
    def _sa_deferred_table_resolver(
        cls, metadata: MetaData
    ) -> Callable[[str], Table]:
        def _resolve(key: str) -> Table:
            # reflection has already occurred so this Table would have
            # its contents already
            return Table(key, metadata)

        return _resolve

    _sa_decl_prepare = True

    @classmethod
    def _sa_raise_deferred_config(cls):
        raise orm_exc.UnmappedClassError(
            cls,
            msg="Class %s is a subclass of DeferredReflection.  "
            "Mappings are not produced until the .prepare() "
            "method is called on the class hierarchy."
            % orm_exc._safe_cls_name(cls),
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\misc.py ===
"""Miscellaneous stuff that does not really fit anywhere else."""

from __future__ import annotations

import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent


class Undecidable(ValueError):
    # an error to be raised when a decision cannot be made definitively
    # where a definitive answer is needed
    pass


def filldedent(s, w=70, **kwargs):
    """
    Strips leading and trailing empty lines from a copy of ``s``, then dedents,
    fills and returns it.

    Empty line stripping serves to deal with docstrings like this one that
    start with a newline after the initial triple quote, inserting an empty
    line at the beginning of the string.

    Additional keyword arguments will be passed to ``textwrap.fill()``.

    See Also
    ========
    strlines, rawlines

    """
    return '\n' + fill(dedent(str(s)).strip('\n'), width=w, **kwargs)


def strlines(s, c=64, short=False):
    """Return a cut-and-pastable string that, when printed, is
    equivalent to the input.  The lines will be surrounded by
    parentheses and no line will be longer than c (default 64)
    characters. If the line contains newlines characters, the
    `rawlines` result will be returned.  If ``short`` is True
    (default is False) then if there is one line it will be
    returned without bounding parentheses.

    Examples
    ========

    >>> from sympy.utilities.misc import strlines
    >>> q = 'this is a long string that should be broken into shorter lines'
    >>> print(strlines(q, 40))
    (
    'this is a long string that should be b'
    'roken into shorter lines'
    )
    >>> q == (
    ... 'this is a long string that should be b'
    ... 'roken into shorter lines'
    ... )
    True

    See Also
    ========
    filldedent, rawlines
    """
    if not isinstance(s, str):
        raise ValueError('expecting string input')
    if '\n' in s:
        return rawlines(s)
    q = '"' if repr(s).startswith('"') else "'"
    q = (q,)*2
    if '\\' in s:  # use r-string
        m = '(\nr%s%%s%s\n)' % q
        j = '%s\nr%s' % q
        c -= 3
    else:
        m = '(\n%s%%s%s\n)' % q
        j = '%s\n%s' % q
        c -= 2
    out = []
    while s:
        out.append(s[:c])
        s=s[c:]
    if short and len(out) == 1:
        return (m % out[0]).splitlines()[1]  # strip bounding (\n...\n)
    return m % j.join(out)


def rawlines(s):
    """Return a cut-and-pastable string that, when printed, is equivalent
    to the input. Use this when there is more than one line in the
    string. The string returned is formatted so it can be indented
    nicely within tests; in some cases it is wrapped in the dedent
    function which has to be imported from textwrap.

    Examples
    ========

    Note: because there are characters in the examples below that need
    to be escaped because they are themselves within a triple quoted
    docstring, expressions below look more complicated than they would
    be if they were printed in an interpreter window.

    >>> from sympy.utilities.misc import rawlines
    >>> from sympy import TableForm
    >>> s = str(TableForm([[1, 10]], headings=(None, ['a', 'bee'])))
    >>> print(rawlines(s))
    (
        'a bee\\n'
        '-----\\n'
        '1 10 '
    )
    >>> print(rawlines('''this
    ... that'''))
    dedent('''\\
        this
        that''')

    >>> print(rawlines('''this
    ... that
    ... '''))
    dedent('''\\
        this
        that
        ''')

    >>> s = \"\"\"this
    ... is a triple '''
    ... \"\"\"
    >>> print(rawlines(s))
    dedent(\"\"\"\\
        this
        is a triple '''
        \"\"\")

    >>> print(rawlines('''this
    ... that
    ...     '''))
    (
        'this\\n'
        'that\\n'
        '    '
    )

    See Also
    ========
    filldedent, strlines
    """
    lines = s.split('\n')
    if len(lines) == 1:
        return repr(lines[0])
    triple = ["'''" in s, '"""' in s]
    if any(li.endswith(' ') for li in lines) or '\\' in s or all(triple):
        rv = []
        # add on the newlines
        trailing = s.endswith('\n')
        last = len(lines) - 1
        for i, li in enumerate(lines):
            if i != last or trailing:
                rv.append(repr(li + '\n'))
            else:
                rv.append(repr(li))
        return '(\n    %s\n)' % '\n    '.join(rv)
    else:
        rv = '\n    '.join(lines)
        if triple[0]:
            return 'dedent("""\\\n    %s""")' % rv
        else:
            return "dedent('''\\\n    %s''')" % rv

ARCH = str(struct.calcsize('P') * 8) + "-bit"


# XXX: PyPy does not support hash randomization
HASH_RANDOMIZATION = getattr(sys.flags, 'hash_randomization', False)

_debug_tmp: list[str] = []
_debug_iter = 0

def debug_decorator(func):
    """If SYMPY_DEBUG is True, it will print a nice execution tree with
    arguments and results of all decorated functions, else do nothing.
    """
    from sympy import SYMPY_DEBUG

    if not SYMPY_DEBUG:
        return func

    def maketree(f, *args, **kw):
        global _debug_tmp, _debug_iter
        oldtmp = _debug_tmp
        _debug_tmp = []
        _debug_iter += 1

        def tree(subtrees):
            def indent(s, variant=1):
                x = s.split("\n")
                r = "+-%s\n" % x[0]
                for a in x[1:]:
                    if a == "":
                        continue
                    if variant == 1:
                        r += "| %s\n" % a
                    else:
                        r += "  %s\n" % a
                return r
            if len(subtrees) == 0:
                return ""
            f = []
            for a in subtrees[:-1]:
                f.append(indent(a))
            f.append(indent(subtrees[-1], 2))
            return ''.join(f)

        # If there is a bug and the algorithm enters an infinite loop, enable the
        # following lines. It will print the names and parameters of all major functions
        # that are called, *before* they are called
        #from functools import reduce
        #print("%s%s %s%s" % (_debug_iter, reduce(lambda x, y: x + y, \
        #    map(lambda x: '-', range(1, 2 + _debug_iter))), f.__name__, args))

        r = f(*args, **kw)

        _debug_iter -= 1
        s = "%s%s = %s\n" % (f.__name__, args, r)
        if _debug_tmp != []:
            s += tree(_debug_tmp)
        _debug_tmp = oldtmp
        _debug_tmp.append(s)
        if _debug_iter == 0:
            print(_debug_tmp[0])
            _debug_tmp = []
        return r

    def decorated(*args, **kwargs):
        return maketree(func, *args, **kwargs)

    return decorated


def debug(*args):
    """
    Print ``*args`` if SYMPY_DEBUG is True, else do nothing.
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(*args, file=sys.stderr)


def debugf(string, args):
    """
    Print ``string%args`` if SYMPY_DEBUG is True, else do nothing. This is
    intended for debug messages using formatted strings.
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(string%args, file=sys.stderr)


def find_executable(executable, path=None):
    """Try to find 'executable' in the directories listed in 'path' (a
    string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH']).  Returns the complete filename or None if not
    found
    """
    from .exceptions import sympy_deprecation_warning
    sympy_deprecation_warning(
        """
        sympy.utilities.misc.find_executable() is deprecated. Use the standard
        library shutil.which() function instead.
        """,
        deprecated_since_version="1.7",
        active_deprecations_target="deprecated-find-executable",
    )
    if path is None:
        path = os.environ['PATH']
    paths = path.split(os.pathsep)
    extlist = ['']
    if os.name == 'os2':
        (base, ext) = os.path.splitext(executable)
        # executable files on OS/2 can have an arbitrary extension, but
        # .exe is automatically appended if no dot is present in the name
        if not ext:
            executable = executable + ".exe"
    elif sys.platform == 'win32':
        pathext = os.environ['PATHEXT'].lower().split(os.pathsep)
        (base, ext) = os.path.splitext(executable)
        if ext.lower() not in pathext:
            extlist = pathext
    for ext in extlist:
        execname = executable + ext
        if os.path.isfile(execname):
            return execname
        else:
            for p in paths:
                f = os.path.join(p, execname)
                if os.path.isfile(f):
                    return f

    return None


def func_name(x, short=False):
    """Return function name of `x` (if defined) else the `type(x)`.
    If short is True and there is a shorter alias for the result,
    return the alias.

    Examples
    ========

    >>> from sympy.utilities.misc import func_name
    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> func_name(Matrix.eye(3))
    'MutableDenseMatrix'
    >>> func_name(x < 1)
    'StrictLessThan'
    >>> func_name(x < 1, short=True)
    'Lt'
    """
    alias = {
    'GreaterThan': 'Ge',
    'StrictGreaterThan': 'Gt',
    'LessThan': 'Le',
    'StrictLessThan': 'Lt',
    'Equality': 'Eq',
    'Unequality': 'Ne',
    }
    typ = type(x)
    if str(typ).startswith("<type '"):
        typ = str(typ).split("'")[1].split("'")[0]
    elif str(typ).startswith("<class '"):
        typ = str(typ).split("'")[1].split("'")[0]
    rv = getattr(getattr(x, 'func', x), '__name__', typ)
    if '.' in rv:
        rv = rv.split('.')[-1]
    if short:
        rv = alias.get(rv, rv)
    return rv


def _replace(reps):
    """Return a function that can make the replacements, given in
    ``reps``, on a string. The replacements should be given as mapping.

    Examples
    ========

    >>> from sympy.utilities.misc import _replace
    >>> f = _replace(dict(foo='bar', d='t'))
    >>> f('food')
    'bart'
    >>> f = _replace({})
    >>> f('food')
    'food'
    """
    if not reps:
        return lambda x: x
    D = lambda match: reps[match.group(0)]
    pattern = _re.compile("|".join(
        [_re.escape(k) for k, v in reps.items()]), _re.MULTILINE)
    return lambda string: pattern.sub(D, string)


def replace(string, *reps):
    """Return ``string`` with all keys in ``reps`` replaced with
    their corresponding values, longer strings first, irrespective
    of the order they are given.  ``reps`` may be passed as tuples
    or a single mapping.

    Examples
    ========

    >>> from sympy.utilities.misc import replace
    >>> replace('foo', {'oo': 'ar', 'f': 'b'})
    'bar'
    >>> replace("spamham sha", ("spam", "eggs"), ("sha","md5"))
    'eggsham md5'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap (i.e. are the same
    length and have some identical sequence at the
    beginning/end):

    >>> reps = [
    ...     ('ab', 'x'),
    ...     ('bc', 'y')]
    >>> replace('abc', *reps) in ('xc', 'ay')
    True

    References
    ==========

    .. [1] https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
    """
    if len(reps) == 1:
        kv = reps[0]
        if isinstance(kv, dict):
            reps = kv
        else:
            return string.replace(*kv)
    else:
        reps = dict(reps)
    return _replace(reps)(string)


def translate(s, a, b=None, c=None):
    """Return ``s`` where characters have been replaced or deleted.

    SYNTAX
    ======

    translate(s, None, deletechars):
        all characters in ``deletechars`` are deleted
    translate(s, map [,deletechars]):
        all characters in ``deletechars`` (if provided) are deleted
        then the replacements defined by map are made; if the keys
        of map are strings then the longer ones are handled first.
        Multicharacter deletions should have a value of ''.
    translate(s, oldchars, newchars, deletechars)
        all characters in ``deletechars`` are deleted
        then each character in ``oldchars`` is replaced with the
        corresponding character in ``newchars``

    Examples
    ========

    >>> from sympy.utilities.misc import translate
    >>> abc = 'abc'
    >>> translate(abc, None, 'a')
    'bc'
    >>> translate(abc, {'a': 'x'}, 'c')
    'xb'
    >>> translate(abc, {'abc': 'x', 'a': 'y'})
    'x'

    >>> translate('abcd', 'ac', 'AC', 'd')
    'AbC'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap are the same
    length and have some identical sequences at the
    beginning/end:

    >>> translate(abc, {'ab': 'x', 'bc': 'y'}) in ('xc', 'ay')
    True
    """

    mr = {}
    if a is None:
        if c is not None:
            raise ValueError('c should be None when a=None is passed, instead got %s' % c)
        if b is None:
            return s
        c = b
        a = b = ''
    else:
        if isinstance(a, dict):
            short = {}
            for k in list(a.keys()):
                if len(k) == 1 and len(a[k]) == 1:
                    short[k] = a.pop(k)
            mr = a
            c = b
            if short:
                a, b = [''.join(i) for i in list(zip(*short.items()))]
            else:
                a = b = ''
        elif len(a) != len(b):
            raise ValueError('oldchars and newchars have different lengths')

    if c:
        val = str.maketrans('', '', c)
        s = s.translate(val)
    s = replace(s, mr)
    n = str.maketrans(a, b)
    return s.translate(n)


def ordinal(num):
    """Return ordinal number string of num, e.g. 1 becomes 1st.
    """
    # modified from https://codereview.stackexchange.com/questions/41298/producing-ordinal-numbers
    n = as_int(num)
    k = abs(n) % 100
    if 11 <= k <= 13:
        suffix = 'th'
    elif k % 10 == 1:
        suffix = 'st'
    elif k % 10 == 2:
        suffix = 'nd'
    elif k % 10 == 3:
        suffix = 'rd'
    else:
        suffix = 'th'
    return str(n) + suffix


def as_int(n, strict=True):
    """
    Convert the argument to a builtin integer.

    The return value is guaranteed to be equal to the input. ValueError is
    raised if the input has a non-integral value. When ``strict`` is True, this
    uses `__index__ <https://docs.python.org/3/reference/datamodel.html#object.__index__>`_
    and when it is False it uses ``int``.


    Examples
    ========

    >>> from sympy.utilities.misc import as_int
    >>> from sympy import sqrt, S

    The function is primarily concerned with sanitizing input for
    functions that need to work with builtin integers, so anything that
    is unambiguously an integer should be returned as an int:

    >>> as_int(S(3))
    3

    Floats, being of limited precision, are not assumed to be exact and
    will raise an error unless the ``strict`` flag is False. This
    precision issue becomes apparent for large floating point numbers:

    >>> big = 1e23
    >>> type(big) is float
    True
    >>> big == int(big)
    True
    >>> as_int(big)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    >>> as_int(big, strict=False)
    99999999999999991611392

    Input that might be a complex representation of an integer value is
    also rejected by default:

    >>> one = sqrt(3 + 2*sqrt(2)) - sqrt(2)
    >>> int(one) == 1
    True
    >>> as_int(one)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    """
    if strict:
        try:
            if isinstance(n, bool):
                raise TypeError
            return operator.index(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
    else:
        try:
            result = int(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
        if n - result:
            raise ValueError('%s is not an integer' % (n,))
        return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\contrib\pyopenssl.py ===
"""
Module for using pyOpenSSL as a TLS backend. This module was relevant before
the standard library ``ssl`` module supported SNI, but now that we've dropped
support for Python 2.7 all relevant Python versions support SNI so
**this module is no longer recommended**.

This needs the following packages installed:

* `pyOpenSSL`_ (tested with 16.0.0)
* `cryptography`_ (minimum 1.3.4, from pyopenssl)
* `idna`_ (minimum 2.0)

However, pyOpenSSL depends on cryptography, so while we use all three directly here we
end up having relatively few packages required.

You can install them with the following command:

.. code-block:: bash

    $ python -m pip install pyopenssl cryptography idna

To activate certificate checking, call
:func:`~urllib3.contrib.pyopenssl.inject_into_urllib3` from your Python code
before you begin making HTTP requests. This can be done in a ``sitecustomize``
module, or at any other time before your application begins using ``urllib3``,
like this:

.. code-block:: python

    try:
        import urllib3.contrib.pyopenssl
        urllib3.contrib.pyopenssl.inject_into_urllib3()
    except ImportError:
        pass

.. _pyopenssl: https://www.pyopenssl.org
.. _cryptography: https://cryptography.io
.. _idna: https://github.com/kjd/idna
"""

from __future__ import annotations

import OpenSSL.SSL  # type: ignore[import-untyped]
from cryptography import x509

try:
    from cryptography.x509 import UnsupportedExtension  # type: ignore[attr-defined]
except ImportError:
    # UnsupportedExtension is gone in cryptography >= 2.1.0
    class UnsupportedExtension(Exception):  # type: ignore[no-redef]
        pass


import logging
import ssl
import typing
from io import BytesIO
from socket import socket as socket_cls
from socket import timeout

from .. import util

if typing.TYPE_CHECKING:
    from OpenSSL.crypto import X509  # type: ignore[import-untyped]


__all__ = ["inject_into_urllib3", "extract_from_urllib3"]

# Map from urllib3 to PyOpenSSL compatible parameter-values.
_openssl_versions: dict[int, int] = {
    util.ssl_.PROTOCOL_TLS: OpenSSL.SSL.SSLv23_METHOD,  # type: ignore[attr-defined]
    util.ssl_.PROTOCOL_TLS_CLIENT: OpenSSL.SSL.SSLv23_METHOD,  # type: ignore[attr-defined]
    ssl.PROTOCOL_TLSv1: OpenSSL.SSL.TLSv1_METHOD,
}

if hasattr(ssl, "PROTOCOL_TLSv1_1") and hasattr(OpenSSL.SSL, "TLSv1_1_METHOD"):
    _openssl_versions[ssl.PROTOCOL_TLSv1_1] = OpenSSL.SSL.TLSv1_1_METHOD

if hasattr(ssl, "PROTOCOL_TLSv1_2") and hasattr(OpenSSL.SSL, "TLSv1_2_METHOD"):
    _openssl_versions[ssl.PROTOCOL_TLSv1_2] = OpenSSL.SSL.TLSv1_2_METHOD


_stdlib_to_openssl_verify = {
    ssl.CERT_NONE: OpenSSL.SSL.VERIFY_NONE,
    ssl.CERT_OPTIONAL: OpenSSL.SSL.VERIFY_PEER,
    ssl.CERT_REQUIRED: OpenSSL.SSL.VERIFY_PEER
    + OpenSSL.SSL.VERIFY_FAIL_IF_NO_PEER_CERT,
}
_openssl_to_stdlib_verify = {v: k for k, v in _stdlib_to_openssl_verify.items()}

# The SSLvX values are the most likely to be missing in the future
# but we check them all just to be sure.
_OP_NO_SSLv2_OR_SSLv3: int = getattr(OpenSSL.SSL, "OP_NO_SSLv2", 0) | getattr(
    OpenSSL.SSL, "OP_NO_SSLv3", 0
)
_OP_NO_TLSv1: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1", 0)
_OP_NO_TLSv1_1: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1_1", 0)
_OP_NO_TLSv1_2: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1_2", 0)
_OP_NO_TLSv1_3: int = getattr(OpenSSL.SSL, "OP_NO_TLSv1_3", 0)

_openssl_to_ssl_minimum_version: dict[int, int] = {
    ssl.TLSVersion.MINIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3,
    ssl.TLSVersion.TLSv1: _OP_NO_SSLv2_OR_SSLv3,
    ssl.TLSVersion.TLSv1_1: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1,
    ssl.TLSVersion.TLSv1_2: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1,
    ssl.TLSVersion.TLSv1_3: (
        _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2
    ),
    ssl.TLSVersion.MAXIMUM_SUPPORTED: (
        _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2
    ),
}
_openssl_to_ssl_maximum_version: dict[int, int] = {
    ssl.TLSVersion.MINIMUM_SUPPORTED: (
        _OP_NO_SSLv2_OR_SSLv3
        | _OP_NO_TLSv1
        | _OP_NO_TLSv1_1
        | _OP_NO_TLSv1_2
        | _OP_NO_TLSv1_3
    ),
    ssl.TLSVersion.TLSv1: (
        _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2 | _OP_NO_TLSv1_3
    ),
    ssl.TLSVersion.TLSv1_1: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_2 | _OP_NO_TLSv1_3,
    ssl.TLSVersion.TLSv1_2: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_3,
    ssl.TLSVersion.TLSv1_3: _OP_NO_SSLv2_OR_SSLv3,
    ssl.TLSVersion.MAXIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3,
}

# OpenSSL will only write 16K at a time
SSL_WRITE_BLOCKSIZE = 16384

orig_util_SSLContext = util.ssl_.SSLContext


log = logging.getLogger(__name__)


def inject_into_urllib3() -> None:
    "Monkey-patch urllib3 with PyOpenSSL-backed SSL-support."

    _validate_dependencies_met()

    util.SSLContext = PyOpenSSLContext  # type: ignore[assignment]
    util.ssl_.SSLContext = PyOpenSSLContext  # type: ignore[assignment]
    util.IS_PYOPENSSL = True
    util.ssl_.IS_PYOPENSSL = True


def extract_from_urllib3() -> None:
    "Undo monkey-patching by :func:`inject_into_urllib3`."

    util.SSLContext = orig_util_SSLContext
    util.ssl_.SSLContext = orig_util_SSLContext
    util.IS_PYOPENSSL = False
    util.ssl_.IS_PYOPENSSL = False


def _validate_dependencies_met() -> None:
    """
    Verifies that PyOpenSSL's package-level dependencies have been met.
    Throws `ImportError` if they are not met.
    """
    # Method added in `cryptography==1.1`; not available in older versions
    from cryptography.x509.extensions import Extensions

    if getattr(Extensions, "get_extension_for_class", None) is None:
        raise ImportError(
            "'cryptography' module missing required functionality.  "
            "Try upgrading to v1.3.4 or newer."
        )

    # pyOpenSSL 0.14 and above use cryptography for OpenSSL bindings. The _x509
    # attribute is only present on those versions.
    from OpenSSL.crypto import X509

    x509 = X509()
    if getattr(x509, "_x509", None) is None:
        raise ImportError(
            "'pyOpenSSL' module missing required functionality. "
            "Try upgrading to v0.14 or newer."
        )


def _dnsname_to_stdlib(name: str) -> str | None:
    """
    Converts a dNSName SubjectAlternativeName field to the form used by the
    standard library on the given Python version.

    Cryptography produces a dNSName as a unicode string that was idna-decoded
    from ASCII bytes. We need to idna-encode that string to get it back, and
    then on Python 3 we also need to convert to unicode via UTF-8 (the stdlib
    uses PyUnicode_FromStringAndSize on it, which decodes via UTF-8).

    If the name cannot be idna-encoded then we return None signalling that
    the name given should be skipped.
    """

    def idna_encode(name: str) -> bytes | None:
        """
        Borrowed wholesale from the Python Cryptography Project. It turns out
        that we can't just safely call `idna.encode`: it can explode for
        wildcard names. This avoids that problem.
        """
        import idna

        try:
            for prefix in ["*.", "."]:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    return prefix.encode("ascii") + idna.encode(name)
            return idna.encode(name)
        except idna.core.IDNAError:
            return None

    # Don't send IPv6 addresses through the IDNA encoder.
    if ":" in name:
        return name

    encoded_name = idna_encode(name)
    if encoded_name is None:
        return None
    return encoded_name.decode("utf-8")


def get_subj_alt_name(peer_cert: X509) -> list[tuple[str, str]]:
    """
    Given an PyOpenSSL certificate, provides all the subject alternative names.
    """
    cert = peer_cert.to_cryptography()

    # We want to find the SAN extension. Ask Cryptography to locate it (it's
    # faster than looping in Python)
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        # No such extension, return the empty list.
        return []
    except (
        x509.DuplicateExtension,
        UnsupportedExtension,
        x509.UnsupportedGeneralNameType,
        UnicodeError,
    ) as e:
        # A problem has been found with the quality of the certificate. Assume
        # no SAN field is present.
        log.warning(
            "A problem was encountered with the certificate that prevented "
            "urllib3 from finding the SubjectAlternativeName field. This can "
            "affect certificate validation. The error was %s",
            e,
        )
        return []

    # We want to return dNSName and iPAddress fields. We need to cast the IPs
    # back to strings because the match_hostname function wants them as
    # strings.
    # Sadly the DNS names need to be idna encoded and then, on Python 3, UTF-8
    # decoded. This is pretty frustrating, but that's what the standard library
    # does with certificates, and so we need to attempt to do the same.
    # We also want to skip over names which cannot be idna encoded.
    names = [
        ("DNS", name)
        for name in map(_dnsname_to_stdlib, ext.get_values_for_type(x509.DNSName))
        if name is not None
    ]
    names.extend(
        ("IP Address", str(name)) for name in ext.get_values_for_type(x509.IPAddress)
    )

    return names


class WrappedSocket:
    """API-compatibility wrapper for Python OpenSSL's Connection-class."""

    def __init__(
        self,
        connection: OpenSSL.SSL.Connection,
        socket: socket_cls,
        suppress_ragged_eofs: bool = True,
    ) -> None:
        self.connection = connection
        self.socket = socket
        self.suppress_ragged_eofs = suppress_ragged_eofs
        self._io_refs = 0
        self._closed = False

    def fileno(self) -> int:
        return self.socket.fileno()

    # Copy-pasted from Python 3.5 source code
    def _decref_socketios(self) -> None:
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._closed:
            self.close()

    def recv(self, *args: typing.Any, **kwargs: typing.Any) -> bytes:
        try:
            data = self.connection.recv(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, "Unexpected EOF"):
                return b""
            else:
                raise OSError(e.args[0], str(e)) from e
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return b""
            else:
                raise
        except OpenSSL.SSL.WantReadError as e:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise timeout("The read operation timed out") from e
            else:
                return self.recv(*args, **kwargs)

        # TLS 1.3 post-handshake authentication
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"read error: {e!r}") from e
        else:
            return data  # type: ignore[no-any-return]

    def recv_into(self, *args: typing.Any, **kwargs: typing.Any) -> int:
        try:
            return self.connection.recv_into(*args, **kwargs)  # type: ignore[no-any-return]
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, "Unexpected EOF"):
                return 0
            else:
                raise OSError(e.args[0], str(e)) from e
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return 0
            else:
                raise
        except OpenSSL.SSL.WantReadError as e:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise timeout("The read operation timed out") from e
            else:
                return self.recv_into(*args, **kwargs)

        # TLS 1.3 post-handshake authentication
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"read error: {e!r}") from e

    def settimeout(self, timeout: float) -> None:
        return self.socket.settimeout(timeout)

    def _send_until_done(self, data: bytes) -> int:
        while True:
            try:
                return self.connection.send(data)  # type: ignore[no-any-return]
            except OpenSSL.SSL.WantWriteError as e:
                if not util.wait_for_write(self.socket, self.socket.gettimeout()):
                    raise timeout() from e
                continue
            except OpenSSL.SSL.SysCallError as e:
                raise OSError(e.args[0], str(e)) from e

    def sendall(self, data: bytes) -> None:
        total_sent = 0
        while total_sent < len(data):
            sent = self._send_until_done(
                data[total_sent : total_sent + SSL_WRITE_BLOCKSIZE]
            )
            total_sent += sent

    def shutdown(self, how: int) -> None:
        try:
            self.connection.shutdown()
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"shutdown error: {e!r}") from e

    def close(self) -> None:
        self._closed = True
        if self._io_refs <= 0:
            self._real_close()

    def _real_close(self) -> None:
        try:
            return self.connection.close()  # type: ignore[no-any-return]
        except OpenSSL.SSL.Error:
            return

    def getpeercert(
        self, binary_form: bool = False
    ) -> dict[str, list[typing.Any]] | None:
        x509 = self.connection.get_peer_certificate()

        if not x509:
            return x509  # type: ignore[no-any-return]

        if binary_form:
            return OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_ASN1, x509)  # type: ignore[no-any-return]

        return {
            "subject": ((("commonName", x509.get_subject().CN),),),  # type: ignore[dict-item]
            "subjectAltName": get_subj_alt_name(x509),
        }

    def version(self) -> str:
        return self.connection.get_protocol_version_name()  # type: ignore[no-any-return]

    def selected_alpn_protocol(self) -> str | None:
        alpn_proto = self.connection.get_alpn_proto_negotiated()
        return alpn_proto.decode() if alpn_proto else None


WrappedSocket.makefile = socket_cls.makefile  # type: ignore[attr-defined]


class PyOpenSSLContext:
    """
    I am a wrapper class for the PyOpenSSL ``Context`` object. I am responsible
    for translating the interface of the standard library ``SSLContext`` object
    to calls into PyOpenSSL.
    """

    def __init__(self, protocol: int) -> None:
        self.protocol = _openssl_versions[protocol]
        self._ctx = OpenSSL.SSL.Context(self.protocol)
        self._options = 0
        self.check_hostname = False
        self._minimum_version: int = ssl.TLSVersion.MINIMUM_SUPPORTED
        self._maximum_version: int = ssl.TLSVersion.MAXIMUM_SUPPORTED
        self._verify_flags: int = ssl.VERIFY_X509_TRUSTED_FIRST

    @property
    def options(self) -> int:
        return self._options

    @options.setter
    def options(self, value: int) -> None:
        self._options = value
        self._set_ctx_options()

    @property
    def verify_flags(self) -> int:
        return self._verify_flags

    @verify_flags.setter
    def verify_flags(self, value: int) -> None:
        self._verify_flags = value
        self._ctx.get_cert_store().set_flags(self._verify_flags)

    @property
    def verify_mode(self) -> int:
        return _openssl_to_stdlib_verify[self._ctx.get_verify_mode()]

    @verify_mode.setter
    def verify_mode(self, value: ssl.VerifyMode) -> None:
        self._ctx.set_verify(_stdlib_to_openssl_verify[value], _verify_callback)

    def set_default_verify_paths(self) -> None:
        self._ctx.set_default_verify_paths()

    def set_ciphers(self, ciphers: bytes | str) -> None:
        if isinstance(ciphers, str):
            ciphers = ciphers.encode("utf-8")
        self._ctx.set_cipher_list(ciphers)

    def load_verify_locations(
        self,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: bytes | None = None,
    ) -> None:
        if cafile is not None:
            cafile = cafile.encode("utf-8")  # type: ignore[assignment]
        if capath is not None:
            capath = capath.encode("utf-8")  # type: ignore[assignment]
        try:
            self._ctx.load_verify_locations(cafile, capath)
            if cadata is not None:
                self._ctx.load_verify_locations(BytesIO(cadata))
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"unable to load trusted certificates: {e!r}") from e

    def load_cert_chain(
        self,
        certfile: str,
        keyfile: str | None = None,
        password: str | None = None,
    ) -> None:
        try:
            self._ctx.use_certificate_chain_file(certfile)
            if password is not None:
                if not isinstance(password, bytes):
                    password = password.encode("utf-8")  # type: ignore[assignment]
                self._ctx.set_passwd_cb(lambda *_: password)
            self._ctx.use_privatekey_file(keyfile or certfile)
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f"Unable to load certificate chain: {e!r}") from e

    def set_alpn_protocols(self, protocols: list[bytes | str]) -> None:
        protocols = [util.util.to_bytes(p, "ascii") for p in protocols]
        return self._ctx.set_alpn_protos(protocols)  # type: ignore[no-any-return]

    def wrap_socket(
        self,
        sock: socket_cls,
        server_side: bool = False,
        do_handshake_on_connect: bool = True,
        suppress_ragged_eofs: bool = True,
        server_hostname: bytes | str | None = None,
    ) -> WrappedSocket:
        cnx = OpenSSL.SSL.Connection(self._ctx, sock)

        # If server_hostname is an IP, don't use it for SNI, per RFC6066 Section 3
        if server_hostname and not util.ssl_.is_ipaddress(server_hostname):
            if isinstance(server_hostname, str):
                server_hostname = server_hostname.encode("utf-8")
            cnx.set_tlsext_host_name(server_hostname)

        cnx.set_connect_state()

        while True:
            try:
                cnx.do_handshake()
            except OpenSSL.SSL.WantReadError as e:
                if not util.wait_for_read(sock, sock.gettimeout()):
                    raise timeout("select timed out") from e
                continue
            except OpenSSL.SSL.Error as e:
                raise ssl.SSLError(f"bad handshake: {e!r}") from e
            break

        return WrappedSocket(cnx, sock)

    def _set_ctx_options(self) -> None:
        self._ctx.set_options(
            self._options
            | _openssl_to_ssl_minimum_version[self._minimum_version]
            | _openssl_to_ssl_maximum_version[self._maximum_version]
        )

    @property
    def minimum_version(self) -> int:
        return self._minimum_version

    @minimum_version.setter
    def minimum_version(self, minimum_version: int) -> None:
        self._minimum_version = minimum_version
        self._set_ctx_options()

    @property
    def maximum_version(self) -> int:
        return self._maximum_version

    @maximum_version.setter
    def maximum_version(self, maximum_version: int) -> None:
        self._maximum_version = maximum_version
        self._set_ctx_options()


def _verify_callback(
    cnx: OpenSSL.SSL.Connection,
    x509: X509,
    err_no: int,
    err_depth: int,
    return_code: int,
) -> bool:
    return err_no == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\aesaracode.py ===
from __future__ import annotations
import math
from typing import Any

from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial


aesara = import_module('aesara')

if aesara:
    aes = aesara.scalar
    aet = aesara.tensor
    from aesara.tensor import nlinalg
    from aesara.tensor.elemwise import Elemwise
    from aesara.tensor.elemwise import DimShuffle

    # `true_divide` replaced `true_div` in Aesara 2.8.11 (released 2023) to
    # match NumPy
    # XXX: Remove this when not needed to support older versions.
    true_divide = getattr(aet, 'true_divide', None)
    if true_divide is None:
        true_divide = aet.true_div

    mapping = {
            sympy.Add: aet.add,
            sympy.Mul: aet.mul,
            sympy.Abs: aet.abs,
            sympy.sign: aet.sgn,
            sympy.ceiling: aet.ceil,
            sympy.floor: aet.floor,
            sympy.log: aet.log,
            sympy.exp: aet.exp,
            sympy.sqrt: aet.sqrt,
            sympy.cos: aet.cos,
            sympy.acos: aet.arccos,
            sympy.sin: aet.sin,
            sympy.asin: aet.arcsin,
            sympy.tan: aet.tan,
            sympy.atan: aet.arctan,
            sympy.atan2: aet.arctan2,
            sympy.cosh: aet.cosh,
            sympy.acosh: aet.arccosh,
            sympy.sinh: aet.sinh,
            sympy.asinh: aet.arcsinh,
            sympy.tanh: aet.tanh,
            sympy.atanh: aet.arctanh,
            sympy.re: aet.real,
            sympy.im: aet.imag,
            sympy.arg: aet.angle,
            sympy.erf: aet.erf,
            sympy.gamma: aet.gamma,
            sympy.loggamma: aet.gammaln,
            sympy.Pow: aet.pow,
            sympy.Eq: aet.eq,
            sympy.StrictGreaterThan: aet.gt,
            sympy.StrictLessThan: aet.lt,
            sympy.LessThan: aet.le,
            sympy.GreaterThan: aet.ge,
            sympy.And: aet.bitwise_and,  # bitwise
            sympy.Or: aet.bitwise_or,  # bitwise
            sympy.Not: aet.invert,  # bitwise
            sympy.Xor: aet.bitwise_xor,  # bitwise
            sympy.Max: aet.maximum,  # Sympy accept >2 inputs, Aesara only 2
            sympy.Min: aet.minimum,  # Sympy accept >2 inputs, Aesara only 2
            sympy.conjugate: aet.conj,
            sympy.core.numbers.ImaginaryUnit: lambda:aet.complex(0,1),
            # Matrices
            sympy.MatAdd: Elemwise(aes.add),
            sympy.HadamardProduct: Elemwise(aes.mul),
            sympy.Trace: nlinalg.trace,
            sympy.Determinant : nlinalg.det,
            sympy.Inverse: nlinalg.matrix_inverse,
            sympy.Transpose: DimShuffle((False, False), [1, 0]),
    }


class AesaraPrinter(Printer):
    """
    .. deprecated:: 1.14.
        The ``Aesara Code printing`` is deprecated.See its documentation for
        more information. See :ref:`deprecated-aesaraprinter` for details.

    Code printer which creates Aesara symbolic expression graphs.

    Parameters
    ==========

    cache : dict
        Cache dictionary to use. If None (default) will use
        the global cache. To create a printer which does not depend on or alter
        global state pass an empty dictionary. Note: the dictionary is not
        copied on initialization of the printer and will be updated in-place,
        so using the same dict object when creating multiple printers or making
        multiple calls to :func:`.aesara_code` or :func:`.aesara_function` means
        the cache is shared between all these applications.

    Attributes
    ==========

    cache : dict
        A cache of Aesara variables which have been created for SymPy
        symbol-like objects (e.g. :class:`sympy.core.symbol.Symbol` or
        :class:`sympy.matrices.expressions.MatrixSymbol`). This is used to
        ensure that all references to a given symbol in an expression (or
        multiple expressions) are printed as the same Aesara variable, which is
        created only once. Symbols are differentiated only by name and type. The
        format of the cache's contents should be considered opaque to the user.
    """
    printmethod = "_aesara"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', {})
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        """ Get the cache key for a SymPy object.

        Parameters
        ==========

        s : sympy.core.basic.Basic
            SymPy object to get key for.

        name : str
            Name of object, if it does not have a ``name`` attribute.
        """

        if name is None:
            name = s.name

        return (name, type(s), s.args, dtype, broadcastable)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Aesara variable for a SymPy symbol from the cache, or create it
        if it does not exist.
        """

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        value = aet.tensor(name=name, dtype=dtype, shape=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # Integers already taken care of below, interpret as float
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        if not hasattr(aet, 'stacklists'):
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Aesara")

        return aet.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = aet.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = 1
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = aet.dot(result, children[0])
        else:
            raise NotImplementedError('''Only non-negative integer
           powers of matrices can be handled by Aesara at the moment''')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        return aet.join(0, *[aet.join(1, *row) for row in blocks])


    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return math.pi

    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        e, cond = expr.args[0].args  # First condition and corresponding value

        # Print conditional expression and value for first condition
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # One condition only
        if len(expr.args) == 1:
            # Return value if condition else NaN
            return aet.switch(p_cond, p_e, np.nan)

        # Return value_1 if condition_1 else evaluate remaining conditions
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return aet.switch(p_cond, p_e, p_remaining)

    def _print_Rational(self, expr, **kwargs):
        return true_divide(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        from aesara.gradient import Rop

        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            rv = Rop(rv, var, aet.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        return expr

    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a SymPy expression to a Aesara graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Aesara variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``aesara.tensor.var.TensorVariable``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Aesara.


        .. __: https://aesara.readthedocs.io/en/latest/reference/tensor/shapes.html#broadcasting

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict
            Mapping from SymPy symbols to Aesara datatypes to use when creating
            new Aesara variables for those symbols. Corresponds to the ``dtype``
            argument to ``aesara.tensor.var.TensorVariable``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``aesara.tensor.var.TensorVariable`` to use when creating Aesara
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        aesara.graph.basic.Variable
            A variable corresponding to the expression's value in a Aesara
            symbolic expression graph.

        """
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}

        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)


global_cache: dict[Any, Any] = {}


def aesara_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Aesara graph variable.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Aesara variables (see :class:`AesaraPrinter.cache
        <AesaraPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    Returns
    =======

    aesara.graph.basic.Variable
        A variable corresponding to the expression's value in a Aesara symbolic
        expression graph.

    """
    sympy_deprecation_warning(
        """
        The aesara_code function is deprecated.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-aesaraprinter',
    )

    if not aesara:
        raise ImportError("aesara is required for aesara_code")

    if cache is None:
        cache = global_cache

    return AesaraPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    r"""
    Get value of ``broadcastables`` argument to :func:`.aesara_code` from
    keyword arguments to :func:`.aesara_function`.

    Included for backwards compatibility.

    Parameters
    ==========

    inputs
        Sequence of input symbols.

    dim : int
        Common number of dimensions for all inputs. Overrides other arguments
        if given.

    dims : dict
        Mapping from input symbols to number of dimensions. Overrides
        ``broadcastables`` argument if given.

    broadcastables : dict
        Explicit value of ``broadcastables`` argument to
        :meth:`.AesaraPrinter.doprint`. If not None function will return this value unchanged.

    Returns
    =======
    dict
        Dictionary mapping elements of ``inputs`` to their "broadcastable"
        values (tuple of ``bool``\ s).
    """
    if dim is not None:
        return dict.fromkeys(inputs, (False,) * dim)

    if dims is not None:
        maxdim = max(dims.values())
        return {
            s: (False,) * d + (True,) * (maxdim - d)
            for s, d in dims.items()
        }

    if broadcastables is not None:
        return broadcastables

    return {}


def aesara_function(inputs, outputs, scalar=False, *,
        dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Aesara function from SymPy expressions.

    The inputs and outputs are converted to Aesara variables using
    :func:`.aesara_code` and then passed to ``aesara.function``.

    Parameters
    ==========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Sequence of expressions which constitute the outputs(s) of the
        function. The free symbols of each expression must be a subset of
        ``inputs``.

    scalar : bool
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Aesara function object.

    cache : dict
        Cached Aesara variables (see :class:`AesaraPrinter.cache
        <AesaraPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.

    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``aesara_function([x, y], [...], dim=2)`` is equivalent to using
        ``broadcastables={x: (False, False), y: (False, False)}``.

    Returns
    =======
    callable
        A callable object which takes values of ``inputs`` as positional
        arguments and returns an output array for each of the expressions
        in ``outputs``. If ``outputs`` is a single expression the function will
        return a Numpy array, if it is a list of multiple expressions the
        function will return a list of arrays. See description of the ``squeeze``
        argument above for the behavior when a single output is passed in a list.
        The returned object will either be an instance of
        ``aesara.compile.function.types.Function`` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``aesara_function`` attribute which points to the return value of
        ``aesara.function``.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.printing.aesaracode import aesara_function

    A simple function with one input and one output:

    >>> f1 = aesara_function([x], [x**2 - 1], scalar=True)
    >>> f1(3)
    8.0

    A function with multiple inputs and one output:

    >>> f2 = aesara_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
    >>> f2(3, 4, 2)
    5.0

    A function with multiple inputs and multiple outputs:

    >>> f3 = aesara_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    >>> f3(2, 3)
    [13.0, -5.0]

    See also
    ========

    dim_handling

    """
    sympy_deprecation_warning(
        """
        The aesara_function function is deprecated.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-aesaraprinter',
    )

    if not aesara:
        raise ImportError("Aesara is required for aesara_function")

    # Pop off non-aesara keyword args
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})

    broadcastables = dim_handling(
        inputs, dim=dim, dims=dims, broadcastables=broadcastables,
    )

    # Print inputs/outputs
    code = partial(aesara_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))

    #fix constant expressions as variables
    toutputs = [output if isinstance(output, aesara.graph.basic.Variable) else aet.as_tensor_variable(output) for output in toutputs]

    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # Compile aesara func
    func = aesara.function(tinputs, toutputs, **kwargs)

    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]

    # No wrapper required
    if not scalar or not any(is_0d):
        func.aesara_function = func
        return func

    # Create wrapper to convert 0-dimensional outputs to scalars
    def wrapper(*args):
        out = func(*args)
        # out can be array(1.0) or [array(1.0), array(2.0)]

        if is_sequence(out):
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]
        else:
            return out[()]

    wrapper.__wrapped__ = func
    wrapper.__doc__ = func.__doc__
    wrapper.aesara_function = func
    return wrapper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\cookies.py ===
"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `http.cookiejar.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

import calendar
import copy
import time

from ._internal_utils import to_native_string
from .compat import Morsel, MutableMapping, cookielib, urlparse, urlunparse

try:
    import threading
except ImportError:
    import dummy_threading as threading


class MockRequest:
    """Wraps a `requests.Request` to mimic a `urllib2.Request`.

    The code in `http.cookiejar.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """

    def __init__(self, request):
        self._r = request
        self._new_headers = {}
        self.type = urlparse(self._r.url).scheme

    def get_type(self):
        return self.type

    def get_host(self):
        return urlparse(self._r.url).netloc

    def get_origin_req_host(self):
        return self.get_host()

    def get_full_url(self):
        # Only return the response's URL if the user hadn't set the Host
        # header
        if not self._r.headers.get("Host"):
            return self._r.url
        # If they did set it, retrieve it and reconstruct the expected domain
        host = to_native_string(self._r.headers["Host"], encoding="utf-8")
        parsed = urlparse(self._r.url)
        # Reconstruct the URL as we expect it
        return urlunparse(
            [
                parsed.scheme,
                host,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ]
        )

    def is_unverifiable(self):
        return True

    def has_header(self, name):
        return name in self._r.headers or name in self._new_headers

    def get_header(self, name, default=None):
        return self._r.headers.get(name, self._new_headers.get(name, default))

    def add_header(self, key, val):
        """cookiejar has no legitimate use for this method; add it back if you find one."""
        raise NotImplementedError(
            "Cookie headers should be added with add_unredirected_header()"
        )

    def add_unredirected_header(self, name, value):
        self._new_headers[name] = value

    def get_new_headers(self):
        return self._new_headers

    @property
    def unverifiable(self):
        return self.is_unverifiable()

    @property
    def origin_req_host(self):
        return self.get_origin_req_host()

    @property
    def host(self):
        return self.get_host()


class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `http.cookiejar` expects to see them.
    """

    def __init__(self, headers):
        """Make a MockResponse for `cookiejar` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
        self._headers = headers

    def info(self):
        return self._headers

    def getheaders(self, name):
        self._headers.getheaders(name)


def extract_cookies_to_jar(jar, request, response):
    """Extract the cookies from the response into a CookieJar.

    :param jar: http.cookiejar.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
    if not (hasattr(response, "_original_response") and response._original_response):
        return
    # the _original_response field is the wrapped httplib.HTTPResponse object,
    req = MockRequest(request)
    # pull out the HTTPMessage with the headers and put it in the mock:
    res = MockResponse(response._original_response.msg)
    jar.extract_cookies(res, req)


def get_cookie_header(jar, request):
    """
    Produce an appropriate Cookie header string to be sent with `request`, or None.

    :rtype: str
    """
    r = MockRequest(request)
    jar.add_cookie_header(r)
    return r.get_new_headers().get("Cookie")


def remove_cookie_by_name(cookiejar, name, domain=None, path=None):
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """
    clearables = []
    for cookie in cookiejar:
        if cookie.name != name:
            continue
        if domain is not None and domain != cookie.domain:
            continue
        if path is not None and path != cookie.path:
            continue
        clearables.append((cookie.domain, cookie.path, cookie.name))

    for domain, path, name in clearables:
        cookiejar.clear(domain, path, name)


class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """


class RequestsCookieJar(cookielib.CookieJar, MutableMapping):
    """Compatibility class; is a http.cookiejar.CookieJar, but exposes a dict
    interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Requests does not use the dict interface internally; it's just for
    compatibility with external client code. All requests code should work
    out of the box with externally provided instances of ``CookieJar``, e.g.
    ``LWPCookieJar`` and ``FileCookieJar``.

    Unlike a regular CookieJar, this class is pickleable.

    .. warning:: dictionary operations that are normally O(1) may be O(n).
    """

    def get(self, name, default=None, domain=None, path=None):
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.

        .. warning:: operation is O(n), not O(1).
        """
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default

    def set(self, name, value, **kwargs):
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.
        """
        # support client code that unsets cookies by assignment of a None value:
        if value is None:
            remove_cookie_by_name(
                self, name, domain=kwargs.get("domain"), path=kwargs.get("path")
            )
            return

        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c

    def iterkeys(self):
        """Dict-like iterkeys() that returns an iterator of names of cookies
        from the jar.

        .. seealso:: itervalues() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.name

    def keys(self):
        """Dict-like keys() that returns a list of names of cookies from the
        jar.

        .. seealso:: values() and items().
        """
        return list(self.iterkeys())

    def itervalues(self):
        """Dict-like itervalues() that returns an iterator of values of cookies
        from the jar.

        .. seealso:: iterkeys() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.value

    def values(self):
        """Dict-like values() that returns a list of values of cookies from the
        jar.

        .. seealso:: keys() and items().
        """
        return list(self.itervalues())

    def iteritems(self):
        """Dict-like iteritems() that returns an iterator of name-value tuples
        from the jar.

        .. seealso:: iterkeys() and itervalues().
        """
        for cookie in iter(self):
            yield cookie.name, cookie.value

    def items(self):
        """Dict-like items() that returns a list of name-value tuples from the
        jar. Allows client-code to call ``dict(RequestsCookieJar)`` and get a
        vanilla python dict of key value pairs.

        .. seealso:: keys() and values().
        """
        return list(self.iteritems())

    def list_domains(self):
        """Utility method to list all the domains in the jar."""
        domains = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains

    def list_paths(self):
        """Utility method to list all the paths in the jar."""
        paths = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths

    def multiple_domains(self):
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
        domains = []
        for cookie in iter(self):
            if cookie.domain is not None and cookie.domain in domains:
                return True
            domains.append(cookie.domain)
        return False  # there is only one domain in jar

    def get_dict(self, domain=None, path=None):
        """Takes as an argument an optional domain and path and returns a plain
        old Python dict of name-value pairs of cookies that meet the
        requirements.

        :rtype: dict
        """
        dictionary = {}
        for cookie in iter(self):
            if (domain is None or cookie.domain == domain) and (
                path is None or cookie.path == path
            ):
                dictionary[cookie.name] = cookie.value
        return dictionary

    def __contains__(self, name):
        try:
            return super().__contains__(name)
        except CookieConflictError:
            return True

    def __getitem__(self, name):
        """Dict-like __getitem__() for compatibility with client code. Throws
        exception if there are more than one cookie with name. In that case,
        use the more explicit get() method instead.

        .. warning:: operation is O(n), not O(1).
        """
        return self._find_no_duplicates(name)

    def __setitem__(self, name, value):
        """Dict-like __setitem__ for compatibility with client code. Throws
        exception if there is already a cookie of that name in the jar. In that
        case, use the more explicit set() method instead.
        """
        self.set(name, value)

    def __delitem__(self, name):
        """Deletes a cookie given a name. Wraps ``http.cookiejar.CookieJar``'s
        ``remove_cookie_by_name()``.
        """
        remove_cookie_by_name(self, name)

    def set_cookie(self, cookie, *args, **kwargs):
        if (
            hasattr(cookie.value, "startswith")
            and cookie.value.startswith('"')
            and cookie.value.endswith('"')
        ):
            cookie.value = cookie.value.replace('\\"', "")
        return super().set_cookie(cookie, *args, **kwargs)

    def update(self, other):
        """Updates this jar with cookies from another CookieJar or dict-like"""
        if isinstance(other, cookielib.CookieJar):
            for cookie in other:
                self.set_cookie(copy.copy(cookie))
        else:
            super().update(other)

    def _find(self, name, domain=None, path=None):
        """Requests uses this method internally to get cookie values.

        If there are conflicting cookies, _find arbitrarily chooses one.
        See _find_no_duplicates if you want an exception thrown if there are
        conflicting cookies.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :return: cookie.value
        """
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        return cookie.value

        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def _find_no_duplicates(self, name, domain=None, path=None):
        """Both ``__get_item__`` and ``get`` call this function: it's never
        used elsewhere in Requests.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :raises KeyError: if cookie is not found
        :raises CookieConflictError: if there are multiple cookies
            that match name and optionally domain and path
        :return: cookie.value
        """
        toReturn = None
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if toReturn is not None:
                            # if there are multiple cookies that meet passed in criteria
                            raise CookieConflictError(
                                f"There are multiple cookies with name, {name!r}"
                            )
                        # we will eventually return this as long as no cookie conflict
                        toReturn = cookie.value

        if toReturn:
            return toReturn
        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def __getstate__(self):
        """Unlike a normal CookieJar, this class is pickleable."""
        state = self.__dict__.copy()
        # remove the unpickleable RLock object
        state.pop("_cookies_lock")
        return state

    def __setstate__(self, state):
        """Unlike a normal CookieJar, this class is pickleable."""
        self.__dict__.update(state)
        if "_cookies_lock" not in self.__dict__:
            self._cookies_lock = threading.RLock()

    def copy(self):
        """Return a copy of this RequestsCookieJar."""
        new_cj = RequestsCookieJar()
        new_cj.set_policy(self.get_policy())
        new_cj.update(self)
        return new_cj

    def get_policy(self):
        """Return the CookiePolicy instance used."""
        return self._policy


def _copy_cookie_jar(jar):
    if jar is None:
        return None

    if hasattr(jar, "copy"):
        # We're dealing with an instance of RequestsCookieJar
        return jar.copy()
    # We're dealing with a generic CookieJar instance
    new_jar = copy.copy(jar)
    new_jar.clear()
    for cookie in jar:
        new_jar.set_cookie(copy.copy(cookie))
    return new_jar


def create_cookie(name, value, **kwargs):
    """Make a cookie from underspecified parameters.

    By default, the pair of `name` and `value` will be set for the domain ''
    and sent on every request (this is sometimes called a "supercookie").
    """
    result = {
        "version": 0,
        "name": name,
        "value": value,
        "port": None,
        "domain": "",
        "path": "/",
        "secure": False,
        "expires": None,
        "discard": True,
        "comment": None,
        "comment_url": None,
        "rest": {"HttpOnly": None},
        "rfc2109": False,
    }

    badargs = set(kwargs) - set(result)
    if badargs:
        raise TypeError(
            f"create_cookie() got unexpected keyword arguments: {list(badargs)}"
        )

    result.update(kwargs)
    result["port_specified"] = bool(result["port"])
    result["domain_specified"] = bool(result["domain"])
    result["domain_initial_dot"] = result["domain"].startswith(".")
    result["path_specified"] = bool(result["path"])

    return cookielib.Cookie(**result)


def morsel_to_cookie(morsel):
    """Convert a Morsel object into a Cookie containing the one k/v pair."""

    expires = None
    if morsel["max-age"]:
        try:
            expires = int(time.time() + int(morsel["max-age"]))
        except ValueError:
            raise TypeError(f"max-age: {morsel['max-age']} must be integer")
    elif morsel["expires"]:
        time_template = "%a, %d-%b-%Y %H:%M:%S GMT"
        expires = calendar.timegm(time.strptime(morsel["expires"], time_template))
    return create_cookie(
        comment=morsel["comment"],
        comment_url=bool(morsel["comment"]),
        discard=False,
        domain=morsel["domain"],
        expires=expires,
        name=morsel.key,
        path=morsel["path"],
        port=None,
        rest={"HttpOnly": morsel["httponly"]},
        rfc2109=False,
        secure=bool(morsel["secure"]),
        value=morsel.value,
        version=morsel["version"] or 0,
    )


def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :param cookiejar: (optional) A cookiejar to add the cookies to.
    :param overwrite: (optional) If False, will not replace cookies
        already in the jar with new ones.
    :rtype: CookieJar
    """
    if cookiejar is None:
        cookiejar = RequestsCookieJar()

    if cookie_dict is not None:
        names_from_jar = [cookie.name for cookie in cookiejar]
        for name in cookie_dict:
            if overwrite or (name not in names_from_jar):
                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))

    return cookiejar


def merge_cookies(cookiejar, cookies):
    """Add cookies to cookiejar and returns a merged CookieJar.

    :param cookiejar: CookieJar object to add the cookies to.
    :param cookies: Dictionary or CookieJar object to be added.
    :rtype: CookieJar
    """
    if not isinstance(cookiejar, cookielib.CookieJar):
        raise ValueError("You can only merge into CookieJar")

    if isinstance(cookies, dict):
        cookiejar = cookiejar_from_dict(cookies, cookiejar=cookiejar, overwrite=False)
    elif isinstance(cookies, cookielib.CookieJar):
        try:
            cookiejar.update(cookies)
        except AttributeError:
            for cookie_in_jar in cookies:
                cookiejar.set_cookie(cookie_in_jar)

    return cookiejar

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\cookies.py ===
"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `http.cookiejar.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

import calendar
import copy
import time

from ._internal_utils import to_native_string
from .compat import Morsel, MutableMapping, cookielib, urlparse, urlunparse

try:
    import threading
except ImportError:
    import dummy_threading as threading


class MockRequest:
    """Wraps a `requests.Request` to mimic a `urllib2.Request`.

    The code in `http.cookiejar.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """

    def __init__(self, request):
        self._r = request
        self._new_headers = {}
        self.type = urlparse(self._r.url).scheme

    def get_type(self):
        return self.type

    def get_host(self):
        return urlparse(self._r.url).netloc

    def get_origin_req_host(self):
        return self.get_host()

    def get_full_url(self):
        # Only return the response's URL if the user hadn't set the Host
        # header
        if not self._r.headers.get("Host"):
            return self._r.url
        # If they did set it, retrieve it and reconstruct the expected domain
        host = to_native_string(self._r.headers["Host"], encoding="utf-8")
        parsed = urlparse(self._r.url)
        # Reconstruct the URL as we expect it
        return urlunparse(
            [
                parsed.scheme,
                host,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            ]
        )

    def is_unverifiable(self):
        return True

    def has_header(self, name):
        return name in self._r.headers or name in self._new_headers

    def get_header(self, name, default=None):
        return self._r.headers.get(name, self._new_headers.get(name, default))

    def add_header(self, key, val):
        """cookiejar has no legitimate use for this method; add it back if you find one."""
        raise NotImplementedError(
            "Cookie headers should be added with add_unredirected_header()"
        )

    def add_unredirected_header(self, name, value):
        self._new_headers[name] = value

    def get_new_headers(self):
        return self._new_headers

    @property
    def unverifiable(self):
        return self.is_unverifiable()

    @property
    def origin_req_host(self):
        return self.get_origin_req_host()

    @property
    def host(self):
        return self.get_host()


class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `http.cookiejar` expects to see them.
    """

    def __init__(self, headers):
        """Make a MockResponse for `cookiejar` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
        self._headers = headers

    def info(self):
        return self._headers

    def getheaders(self, name):
        self._headers.getheaders(name)


def extract_cookies_to_jar(jar, request, response):
    """Extract the cookies from the response into a CookieJar.

    :param jar: http.cookiejar.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
    if not (hasattr(response, "_original_response") and response._original_response):
        return
    # the _original_response field is the wrapped httplib.HTTPResponse object,
    req = MockRequest(request)
    # pull out the HTTPMessage with the headers and put it in the mock:
    res = MockResponse(response._original_response.msg)
    jar.extract_cookies(res, req)


def get_cookie_header(jar, request):
    """
    Produce an appropriate Cookie header string to be sent with `request`, or None.

    :rtype: str
    """
    r = MockRequest(request)
    jar.add_cookie_header(r)
    return r.get_new_headers().get("Cookie")


def remove_cookie_by_name(cookiejar, name, domain=None, path=None):
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """
    clearables = []
    for cookie in cookiejar:
        if cookie.name != name:
            continue
        if domain is not None and domain != cookie.domain:
            continue
        if path is not None and path != cookie.path:
            continue
        clearables.append((cookie.domain, cookie.path, cookie.name))

    for domain, path, name in clearables:
        cookiejar.clear(domain, path, name)


class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """


class RequestsCookieJar(cookielib.CookieJar, MutableMapping):
    """Compatibility class; is a http.cookiejar.CookieJar, but exposes a dict
    interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Requests does not use the dict interface internally; it's just for
    compatibility with external client code. All requests code should work
    out of the box with externally provided instances of ``CookieJar``, e.g.
    ``LWPCookieJar`` and ``FileCookieJar``.

    Unlike a regular CookieJar, this class is pickleable.

    .. warning:: dictionary operations that are normally O(1) may be O(n).
    """

    def get(self, name, default=None, domain=None, path=None):
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.

        .. warning:: operation is O(n), not O(1).
        """
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default

    def set(self, name, value, **kwargs):
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.
        """
        # support client code that unsets cookies by assignment of a None value:
        if value is None:
            remove_cookie_by_name(
                self, name, domain=kwargs.get("domain"), path=kwargs.get("path")
            )
            return

        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c

    def iterkeys(self):
        """Dict-like iterkeys() that returns an iterator of names of cookies
        from the jar.

        .. seealso:: itervalues() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.name

    def keys(self):
        """Dict-like keys() that returns a list of names of cookies from the
        jar.

        .. seealso:: values() and items().
        """
        return list(self.iterkeys())

    def itervalues(self):
        """Dict-like itervalues() that returns an iterator of values of cookies
        from the jar.

        .. seealso:: iterkeys() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.value

    def values(self):
        """Dict-like values() that returns a list of values of cookies from the
        jar.

        .. seealso:: keys() and items().
        """
        return list(self.itervalues())

    def iteritems(self):
        """Dict-like iteritems() that returns an iterator of name-value tuples
        from the jar.

        .. seealso:: iterkeys() and itervalues().
        """
        for cookie in iter(self):
            yield cookie.name, cookie.value

    def items(self):
        """Dict-like items() that returns a list of name-value tuples from the
        jar. Allows client-code to call ``dict(RequestsCookieJar)`` and get a
        vanilla python dict of key value pairs.

        .. seealso:: keys() and values().
        """
        return list(self.iteritems())

    def list_domains(self):
        """Utility method to list all the domains in the jar."""
        domains = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains

    def list_paths(self):
        """Utility method to list all the paths in the jar."""
        paths = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths

    def multiple_domains(self):
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
        domains = []
        for cookie in iter(self):
            if cookie.domain is not None and cookie.domain in domains:
                return True
            domains.append(cookie.domain)
        return False  # there is only one domain in jar

    def get_dict(self, domain=None, path=None):
        """Takes as an argument an optional domain and path and returns a plain
        old Python dict of name-value pairs of cookies that meet the
        requirements.

        :rtype: dict
        """
        dictionary = {}
        for cookie in iter(self):
            if (domain is None or cookie.domain == domain) and (
                path is None or cookie.path == path
            ):
                dictionary[cookie.name] = cookie.value
        return dictionary

    def __contains__(self, name):
        try:
            return super().__contains__(name)
        except CookieConflictError:
            return True

    def __getitem__(self, name):
        """Dict-like __getitem__() for compatibility with client code. Throws
        exception if there are more than one cookie with name. In that case,
        use the more explicit get() method instead.

        .. warning:: operation is O(n), not O(1).
        """
        return self._find_no_duplicates(name)

    def __setitem__(self, name, value):
        """Dict-like __setitem__ for compatibility with client code. Throws
        exception if there is already a cookie of that name in the jar. In that
        case, use the more explicit set() method instead.
        """
        self.set(name, value)

    def __delitem__(self, name):
        """Deletes a cookie given a name. Wraps ``http.cookiejar.CookieJar``'s
        ``remove_cookie_by_name()``.
        """
        remove_cookie_by_name(self, name)

    def set_cookie(self, cookie, *args, **kwargs):
        if (
            hasattr(cookie.value, "startswith")
            and cookie.value.startswith('"')
            and cookie.value.endswith('"')
        ):
            cookie.value = cookie.value.replace('\\"', "")
        return super().set_cookie(cookie, *args, **kwargs)

    def update(self, other):
        """Updates this jar with cookies from another CookieJar or dict-like"""
        if isinstance(other, cookielib.CookieJar):
            for cookie in other:
                self.set_cookie(copy.copy(cookie))
        else:
            super().update(other)

    def _find(self, name, domain=None, path=None):
        """Requests uses this method internally to get cookie values.

        If there are conflicting cookies, _find arbitrarily chooses one.
        See _find_no_duplicates if you want an exception thrown if there are
        conflicting cookies.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :return: cookie.value
        """
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        return cookie.value

        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def _find_no_duplicates(self, name, domain=None, path=None):
        """Both ``__get_item__`` and ``get`` call this function: it's never
        used elsewhere in Requests.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :raises KeyError: if cookie is not found
        :raises CookieConflictError: if there are multiple cookies
            that match name and optionally domain and path
        :return: cookie.value
        """
        toReturn = None
        for cookie in iter(self):
            if cookie.name == name:
                if domain is None or cookie.domain == domain:
                    if path is None or cookie.path == path:
                        if toReturn is not None:
                            # if there are multiple cookies that meet passed in criteria
                            raise CookieConflictError(
                                f"There are multiple cookies with name, {name!r}"
                            )
                        # we will eventually return this as long as no cookie conflict
                        toReturn = cookie.value

        if toReturn:
            return toReturn
        raise KeyError(f"name={name!r}, domain={domain!r}, path={path!r}")

    def __getstate__(self):
        """Unlike a normal CookieJar, this class is pickleable."""
        state = self.__dict__.copy()
        # remove the unpickleable RLock object
        state.pop("_cookies_lock")
        return state

    def __setstate__(self, state):
        """Unlike a normal CookieJar, this class is pickleable."""
        self.__dict__.update(state)
        if "_cookies_lock" not in self.__dict__:
            self._cookies_lock = threading.RLock()

    def copy(self):
        """Return a copy of this RequestsCookieJar."""
        new_cj = RequestsCookieJar()
        new_cj.set_policy(self.get_policy())
        new_cj.update(self)
        return new_cj

    def get_policy(self):
        """Return the CookiePolicy instance used."""
        return self._policy


def _copy_cookie_jar(jar):
    if jar is None:
        return None

    if hasattr(jar, "copy"):
        # We're dealing with an instance of RequestsCookieJar
        return jar.copy()
    # We're dealing with a generic CookieJar instance
    new_jar = copy.copy(jar)
    new_jar.clear()
    for cookie in jar:
        new_jar.set_cookie(copy.copy(cookie))
    return new_jar


def create_cookie(name, value, **kwargs):
    """Make a cookie from underspecified parameters.

    By default, the pair of `name` and `value` will be set for the domain ''
    and sent on every request (this is sometimes called a "supercookie").
    """
    result = {
        "version": 0,
        "name": name,
        "value": value,
        "port": None,
        "domain": "",
        "path": "/",
        "secure": False,
        "expires": None,
        "discard": True,
        "comment": None,
        "comment_url": None,
        "rest": {"HttpOnly": None},
        "rfc2109": False,
    }

    badargs = set(kwargs) - set(result)
    if badargs:
        raise TypeError(
            f"create_cookie() got unexpected keyword arguments: {list(badargs)}"
        )

    result.update(kwargs)
    result["port_specified"] = bool(result["port"])
    result["domain_specified"] = bool(result["domain"])
    result["domain_initial_dot"] = result["domain"].startswith(".")
    result["path_specified"] = bool(result["path"])

    return cookielib.Cookie(**result)


def morsel_to_cookie(morsel):
    """Convert a Morsel object into a Cookie containing the one k/v pair."""

    expires = None
    if morsel["max-age"]:
        try:
            expires = int(time.time() + int(morsel["max-age"]))
        except ValueError:
            raise TypeError(f"max-age: {morsel['max-age']} must be integer")
    elif morsel["expires"]:
        time_template = "%a, %d-%b-%Y %H:%M:%S GMT"
        expires = calendar.timegm(time.strptime(morsel["expires"], time_template))
    return create_cookie(
        comment=morsel["comment"],
        comment_url=bool(morsel["comment"]),
        discard=False,
        domain=morsel["domain"],
        expires=expires,
        name=morsel.key,
        path=morsel["path"],
        port=None,
        rest={"HttpOnly": morsel["httponly"]},
        rfc2109=False,
        secure=bool(morsel["secure"]),
        value=morsel.value,
        version=morsel["version"] or 0,
    )


def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :param cookiejar: (optional) A cookiejar to add the cookies to.
    :param overwrite: (optional) If False, will not replace cookies
        already in the jar with new ones.
    :rtype: CookieJar
    """
    if cookiejar is None:
        cookiejar = RequestsCookieJar()

    if cookie_dict is not None:
        names_from_jar = [cookie.name for cookie in cookiejar]
        for name in cookie_dict:
            if overwrite or (name not in names_from_jar):
                cookiejar.set_cookie(create_cookie(name, cookie_dict[name]))

    return cookiejar


def merge_cookies(cookiejar, cookies):
    """Add cookies to cookiejar and returns a merged CookieJar.

    :param cookiejar: CookieJar object to add the cookies to.
    :param cookies: Dictionary or CookieJar object to be added.
    :rtype: CookieJar
    """
    if not isinstance(cookiejar, cookielib.CookieJar):
        raise ValueError("You can only merge into CookieJar")

    if isinstance(cookies, dict):
        cookiejar = cookiejar_from_dict(cookies, cookiejar=cookiejar, overwrite=False)
    elif isinstance(cookies, cookielib.CookieJar):
        try:
            cookiejar.update(cookies)
        except AttributeError:
            for cookie_in_jar in cookies:
                cookiejar.set_cookie(cookie_in_jar)

    return cookiejar

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\xml.py ===
"""
:mod:`pandas.io.formats.xml` is a module for formatting data in XML.
"""
from __future__ import annotations

import codecs
import io
from typing import (
    TYPE_CHECKING,
    Any,
    final,
)
import warnings

from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna

from pandas.core.shared_docs import _shared_docs

from pandas.io.common import get_handle
from pandas.io.xml import (
    get_data_from_filepath,
    preprocess_data,
)

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import DataFrame


@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "path_or_buffer",
)
class _BaseXMLFormatter:
    """
    Subclass for formatting data in XML.

    Parameters
    ----------
    path_or_buffer : str or file-like
        This can be either a string of raw XML, a valid URL,
        file or file-like object.

    index : bool
        Whether to include index in xml document.

    row_name : str
        Name for root of xml document. Default is 'data'.

    root_name : str
        Name for row elements of xml document. Default is 'row'.

    na_rep : str
        Missing data representation.

    attrs_cols : list
        List of columns to write as attributes in row element.

    elem_cols : list
        List of columns to write as children in row element.

    namespaces : dict
        The namespaces to define in XML document as dicts with key
        being namespace and value the URI.

    prefix : str
        The prefix for each element in XML document including root.

    encoding : str
        Encoding of xml object or document.

    xml_declaration : bool
        Whether to include xml declaration at top line item in xml.

    pretty_print : bool
        Whether to write xml document with line breaks and indentation.

    stylesheet : str or file-like
        A URL, file, file-like object, or a raw string containing XSLT.

    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    See also
    --------
    pandas.io.formats.xml.EtreeXMLFormatter
    pandas.io.formats.xml.LxmlXMLFormatter

    """

    def __init__(
        self,
        frame: DataFrame,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        index: bool = True,
        root_name: str | None = "data",
        row_name: str | None = "row",
        na_rep: str | None = None,
        attr_cols: list[str] | None = None,
        elem_cols: list[str] | None = None,
        namespaces: dict[str | None, str] | None = None,
        prefix: str | None = None,
        encoding: str = "utf-8",
        xml_declaration: bool | None = True,
        pretty_print: bool | None = True,
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
    ) -> None:
        self.frame = frame
        self.path_or_buffer = path_or_buffer
        self.index = index
        self.root_name = root_name
        self.row_name = row_name
        self.na_rep = na_rep
        self.attr_cols = attr_cols
        self.elem_cols = elem_cols
        self.namespaces = namespaces
        self.prefix = prefix
        self.encoding = encoding
        self.xml_declaration = xml_declaration
        self.pretty_print = pretty_print
        self.stylesheet = stylesheet
        self.compression: CompressionOptions = compression
        self.storage_options = storage_options

        self.orig_cols = self.frame.columns.tolist()
        self.frame_dicts = self._process_dataframe()

        self._validate_columns()
        self._validate_encoding()
        self.prefix_uri = self._get_prefix_uri()
        self._handle_indexes()

    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        raise AbstractMethodError(self)

    @final
    def _validate_columns(self) -> None:
        """
        Validate elems_cols and attrs_cols.

        This method will check if columns is list-like.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
        if self.attr_cols and not is_list_like(self.attr_cols):
            raise TypeError(
                f"{type(self.attr_cols).__name__} is not a valid type for attr_cols"
            )

        if self.elem_cols and not is_list_like(self.elem_cols):
            raise TypeError(
                f"{type(self.elem_cols).__name__} is not a valid type for elem_cols"
            )

    @final
    def _validate_encoding(self) -> None:
        """
        Validate encoding.

        This method will check if encoding is among listed under codecs.

        Raises
        ------
        LookupError
            * If encoding is not available in codecs.
        """

        codecs.lookup(self.encoding)

    @final
    def _process_dataframe(self) -> dict[int | str, dict[str, Any]]:
        """
        Adjust Data Frame to fit xml output.

        This method will adjust underlying data frame for xml output,
        including optionally replacing missing values and including indexes.
        """

        df = self.frame

        if self.index:
            df = df.reset_index()

        if self.na_rep is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Downcasting object dtype arrays",
                    category=FutureWarning,
                )
                df = df.fillna(self.na_rep)

        return df.to_dict(orient="index")

    @final
    def _handle_indexes(self) -> None:
        """
        Handle indexes.

        This method will add indexes into attr_cols or elem_cols.
        """

        if not self.index:
            return

        first_key = next(iter(self.frame_dicts))
        indexes: list[str] = [
            x for x in self.frame_dicts[first_key].keys() if x not in self.orig_cols
        ]

        if self.attr_cols:
            self.attr_cols = indexes + self.attr_cols

        if self.elem_cols:
            self.elem_cols = indexes + self.elem_cols

    def _get_prefix_uri(self) -> str:
        """
        Get uri of namespace prefix.

        This method retrieves corresponding URI to prefix in namespaces.

        Raises
        ------
        KeyError
            *If prefix is not included in namespace dict.
        """

        raise AbstractMethodError(self)

    @final
    def _other_namespaces(self) -> dict:
        """
        Define other namespaces.

        This method will build dictionary of namespaces attributes
        for root element, conditionally with optional namespaces and
        prefix.
        """

        nmsp_dict: dict[str, str] = {}
        if self.namespaces:
            nmsp_dict = {
                f"xmlns{p if p=='' else f':{p}'}": n
                for p, n in self.namespaces.items()
                if n != self.prefix_uri[1:-1]
            }

        return nmsp_dict

    @final
    def _build_attribs(self, d: dict[str, Any], elem_row: Any) -> Any:
        """
        Create attributes of row.

        This method adds attributes using attr_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """

        if not self.attr_cols:
            return elem_row

        for col in self.attr_cols:
            attr_name = self._get_flat_col_name(col)
            try:
                if not isna(d[col]):
                    elem_row.attrib[attr_name] = str(d[col])
            except KeyError:
                raise KeyError(f"no valid column, {col}")
        return elem_row

    @final
    def _get_flat_col_name(self, col: str | tuple) -> str:
        flat_col = col
        if isinstance(col, tuple):
            flat_col = (
                "".join([str(c) for c in col]).strip()
                if "" in col
                else "_".join([str(c) for c in col]).strip()
            )
        return f"{self.prefix_uri}{flat_col}"

    @cache_readonly
    def _sub_element_cls(self):
        raise AbstractMethodError(self)

    @final
    def _build_elems(self, d: dict[str, Any], elem_row: Any) -> None:
        """
        Create child elements of row.

        This method adds child elements using elem_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
        sub_element_cls = self._sub_element_cls

        if not self.elem_cols:
            return

        for col in self.elem_cols:
            elem_name = self._get_flat_col_name(col)
            try:
                val = None if isna(d[col]) or d[col] == "" else str(d[col])
                sub_element_cls(elem_row, elem_name).text = val
            except KeyError:
                raise KeyError(f"no valid column, {col}")

    @final
    def write_output(self) -> str | None:
        xml_doc = self._build_tree()

        if self.path_or_buffer is not None:
            with get_handle(
                self.path_or_buffer,
                "wb",
                compression=self.compression,
                storage_options=self.storage_options,
                is_text=False,
            ) as handles:
                handles.handle.write(xml_doc)
            return None

        else:
            return xml_doc.decode(self.encoding).rstrip()


class EtreeXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def _build_tree(self) -> bytes:
        from xml.etree.ElementTree import (
            Element,
            SubElement,
            tostring,
        )

        self.root = Element(
            f"{self.prefix_uri}{self.root_name}", attrib=self._other_namespaces()
        )

        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)

            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        self.out_xml = tostring(
            self.root,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        if self.pretty_print:
            self.out_xml = self._prettify_tree()

        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )

        return self.out_xml

    def _get_prefix_uri(self) -> str:
        from xml.etree.ElementTree import register_namespace

        uri = ""
        if self.namespaces:
            for p, n in self.namespaces.items():
                if isinstance(p, str) and isinstance(n, str):
                    register_namespace(p, n)
            if self.prefix:
                try:
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError:
                    raise KeyError(f"{self.prefix} is not included in namespaces")
            elif "" in self.namespaces:
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ""

        return uri

    @cache_readonly
    def _sub_element_cls(self):
        from xml.etree.ElementTree import SubElement

        return SubElement

    def _prettify_tree(self) -> bytes:
        """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """

        from xml.dom.minidom import parseString

        dom = parseString(self.out_xml)

        return dom.toprettyxml(indent="  ", encoding=self.encoding)


class LxmlXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._convert_empty_str_key()

    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        from lxml.etree import (
            Element,
            SubElement,
            tostring,
        )

        self.root = Element(f"{self.prefix_uri}{self.root_name}", nsmap=self.namespaces)

        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)

            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        self.out_xml = tostring(
            self.root,
            pretty_print=self.pretty_print,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        if self.stylesheet is not None:
            self.out_xml = self._transform_doc()

        return self.out_xml

    def _convert_empty_str_key(self) -> None:
        """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """

        if self.namespaces and "" in self.namespaces.keys():
            self.namespaces[None] = self.namespaces.pop("", "default")

    def _get_prefix_uri(self) -> str:
        uri = ""
        if self.namespaces:
            if self.prefix:
                try:
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError:
                    raise KeyError(f"{self.prefix} is not included in namespaces")
            elif "" in self.namespaces:
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ""

        return uri

    @cache_readonly
    def _sub_element_cls(self):
        from lxml.etree import SubElement

        return SubElement

    def _transform_doc(self) -> bytes:
        """
        Parse stylesheet from file or buffer and run it.

        This method will parse stylesheet object into tree for parsing
        conditionally by its specific object type, then transforms
        original tree with XSLT script.
        """
        from lxml.etree import (
            XSLT,
            XMLParser,
            fromstring,
            parse,
        )

        style_doc = self.stylesheet
        assert style_doc is not None  # is ensured by caller

        handle_data = get_data_from_filepath(
            filepath_or_buffer=style_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)

            if isinstance(xml_data, io.StringIO):
                xsl_doc = fromstring(
                    xml_data.getvalue().encode(self.encoding), parser=curr_parser
                )
            else:
                xsl_doc = parse(xml_data, parser=curr_parser)

        transformer = XSLT(xsl_doc)
        new_doc = transformer(self.root)

        return bytes(new_doc)