
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\json\tag.py ===
"""
Tagged JSON
~~~~~~~~~~~

A compact representation for lossless serialization of non-standard JSON
types. :class:`~flask.sessions.SecureCookieSessionInterface` uses this
to serialize the session data, but it may be useful in other places. It
can be extended to support other types.

.. autoclass:: TaggedJSONSerializer
    :members:

.. autoclass:: JSONTag
    :members:

Let's see an example that adds support for
:class:`~collections.OrderedDict`. Dicts don't have an order in JSON, so
to handle this we will dump the items as a list of ``[key, value]``
pairs. Subclass :class:`JSONTag` and give it the new key ``' od'`` to
identify the type. The session serializer processes dicts first, so
insert the new tag at the front of the order since ``OrderedDict`` must
be processed before ``dict``.

.. code-block:: python

    from flask.json.tag import JSONTag

    class TagOrderedDict(JSONTag):
        __slots__ = ('serializer',)
        key = ' od'

        def check(self, value):
            return isinstance(value, OrderedDict)

        def to_json(self, value):
            return [[k, self.serializer.tag(v)] for k, v in iteritems(value)]

        def to_python(self, value):
            return OrderedDict(value)

    app.session_interface.serializer.register(TagOrderedDict, index=0)
"""

from __future__ import annotations

import typing as t
from base64 import b64decode
from base64 import b64encode
from datetime import datetime
from uuid import UUID

from markupsafe import Markup
from werkzeug.http import http_date
from werkzeug.http import parse_date

from ..json import dumps
from ..json import loads


class JSONTag:
    """Base class for defining type tags for :class:`TaggedJSONSerializer`."""

    __slots__ = ("serializer",)

    #: The tag to mark the serialized object with. If empty, this tag is
    #: only used as an intermediate step during tagging.
    key: str = ""

    def __init__(self, serializer: TaggedJSONSerializer) -> None:
        """Create a tagger for the given serializer."""
        self.serializer = serializer

    def check(self, value: t.Any) -> bool:
        """Check if the given value should be tagged by this tag."""
        raise NotImplementedError

    def to_json(self, value: t.Any) -> t.Any:
        """Convert the Python object to an object that is a valid JSON type.
        The tag will be added later."""
        raise NotImplementedError

    def to_python(self, value: t.Any) -> t.Any:
        """Convert the JSON representation back to the correct type. The tag
        will already be removed."""
        raise NotImplementedError

    def tag(self, value: t.Any) -> dict[str, t.Any]:
        """Convert the value to a valid JSON type and add the tag structure
        around it."""
        return {self.key: self.to_json(value)}


class TagDict(JSONTag):
    """Tag for 1-item dicts whose only key matches a registered tag.

    Internally, the dict key is suffixed with `__`, and the suffix is removed
    when deserializing.
    """

    __slots__ = ()
    key = " di"

    def check(self, value: t.Any) -> bool:
        return (
            isinstance(value, dict)
            and len(value) == 1
            and next(iter(value)) in self.serializer.tags
        )

    def to_json(self, value: t.Any) -> t.Any:
        key = next(iter(value))
        return {f"{key}__": self.serializer.tag(value[key])}

    def to_python(self, value: t.Any) -> t.Any:
        key = next(iter(value))
        return {key[:-2]: value[key]}


class PassDict(JSONTag):
    __slots__ = ()

    def check(self, value: t.Any) -> bool:
        return isinstance(value, dict)

    def to_json(self, value: t.Any) -> t.Any:
        # JSON objects may only have string keys, so don't bother tagging the
        # key here.
        return {k: self.serializer.tag(v) for k, v in value.items()}

    tag = to_json


class TagTuple(JSONTag):
    __slots__ = ()
    key = " t"

    def check(self, value: t.Any) -> bool:
        return isinstance(value, tuple)

    def to_json(self, value: t.Any) -> t.Any:
        return [self.serializer.tag(item) for item in value]

    def to_python(self, value: t.Any) -> t.Any:
        return tuple(value)


class PassList(JSONTag):
    __slots__ = ()

    def check(self, value: t.Any) -> bool:
        return isinstance(value, list)

    def to_json(self, value: t.Any) -> t.Any:
        return [self.serializer.tag(item) for item in value]

    tag = to_json


class TagBytes(JSONTag):
    __slots__ = ()
    key = " b"

    def check(self, value: t.Any) -> bool:
        return isinstance(value, bytes)

    def to_json(self, value: t.Any) -> t.Any:
        return b64encode(value).decode("ascii")

    def to_python(self, value: t.Any) -> t.Any:
        return b64decode(value)


class TagMarkup(JSONTag):
    """Serialize anything matching the :class:`~markupsafe.Markup` API by
    having a ``__html__`` method to the result of that method. Always
    deserializes to an instance of :class:`~markupsafe.Markup`."""

    __slots__ = ()
    key = " m"

    def check(self, value: t.Any) -> bool:
        return callable(getattr(value, "__html__", None))

    def to_json(self, value: t.Any) -> t.Any:
        return str(value.__html__())

    def to_python(self, value: t.Any) -> t.Any:
        return Markup(value)


class TagUUID(JSONTag):
    __slots__ = ()
    key = " u"

    def check(self, value: t.Any) -> bool:
        return isinstance(value, UUID)

    def to_json(self, value: t.Any) -> t.Any:
        return value.hex

    def to_python(self, value: t.Any) -> t.Any:
        return UUID(value)


class TagDateTime(JSONTag):
    __slots__ = ()
    key = " d"

    def check(self, value: t.Any) -> bool:
        return isinstance(value, datetime)

    def to_json(self, value: t.Any) -> t.Any:
        return http_date(value)

    def to_python(self, value: t.Any) -> t.Any:
        return parse_date(value)


class TaggedJSONSerializer:
    """Serializer that uses a tag system to compactly represent objects that
    are not JSON types. Passed as the intermediate serializer to
    :class:`itsdangerous.Serializer`.

    The following extra types are supported:

    * :class:`dict`
    * :class:`tuple`
    * :class:`bytes`
    * :class:`~markupsafe.Markup`
    * :class:`~uuid.UUID`
    * :class:`~datetime.datetime`
    """

    __slots__ = ("tags", "order")

    #: Tag classes to bind when creating the serializer. Other tags can be
    #: added later using :meth:`~register`.
    default_tags = [
        TagDict,
        PassDict,
        TagTuple,
        PassList,
        TagBytes,
        TagMarkup,
        TagUUID,
        TagDateTime,
    ]

    def __init__(self) -> None:
        self.tags: dict[str, JSONTag] = {}
        self.order: list[JSONTag] = []

        for cls in self.default_tags:
            self.register(cls)

    def register(
        self,
        tag_class: type[JSONTag],
        force: bool = False,
        index: int | None = None,
    ) -> None:
        """Register a new tag with this serializer.

        :param tag_class: tag class to register. Will be instantiated with this
            serializer instance.
        :param force: overwrite an existing tag. If false (default), a
            :exc:`KeyError` is raised.
        :param index: index to insert the new tag in the tag order. Useful when
            the new tag is a special case of an existing tag. If ``None``
            (default), the tag is appended to the end of the order.

        :raise KeyError: if the tag key is already registered and ``force`` is
            not true.
        """
        tag = tag_class(self)
        key = tag.key

        if key:
            if not force and key in self.tags:
                raise KeyError(f"Tag '{key}' is already registered.")

            self.tags[key] = tag

        if index is None:
            self.order.append(tag)
        else:
            self.order.insert(index, tag)

    def tag(self, value: t.Any) -> t.Any:
        """Convert a value to a tagged representation if necessary."""
        for tag in self.order:
            if tag.check(value):
                return tag.tag(value)

        return value

    def untag(self, value: dict[str, t.Any]) -> t.Any:
        """Convert a tagged representation back to the original type."""
        if len(value) != 1:
            return value

        key = next(iter(value))

        if key not in self.tags:
            return value

        return self.tags[key].to_python(value[key])

    def _untag_scan(self, value: t.Any) -> t.Any:
        if isinstance(value, dict):
            # untag each item recursively
            value = {k: self._untag_scan(v) for k, v in value.items()}
            # untag the dict itself
            value = self.untag(value)
        elif isinstance(value, list):
            # untag each item recursively
            value = [self._untag_scan(item) for item in value]

        return value

    def dumps(self, value: t.Any) -> str:
        """Tag the value and dump it to a compact JSON string."""
        return dumps(self.tag(value), separators=(",", ":"))

    def loads(self, value: str) -> t.Any:
        """Load data from a JSON string and deserialized any tagged objects."""
        return self._untag_scan(loads(value))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\delta.py ===
"""
This module implements sums and products containing the Kronecker Delta function.

References
==========

.. [1] https://mathworld.wolfram.com/KroneckerDelta.html

"""
from .products import product
from .summations import Sum, summation
from sympy.core import Add, Mul, S, Dummy
from sympy.core.cache import cacheit
from sympy.core.sorting import default_sort_key
from sympy.functions import KroneckerDelta, Piecewise, piecewise_fold
from sympy.polys.polytools import factor
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve


@cacheit
def _expand_delta(expr, index):
    """
    Expand the first Add containing a simple KroneckerDelta.
    """
    if not expr.is_Mul:
        return expr
    delta = None
    func = Add
    terms = [S.One]
    for h in expr.args:
        if delta is None and h.is_Add and _has_simple_delta(h, index):
            delta = True
            func = h.func
            terms = [terms[0]*t for t in h.args]
        else:
            terms = [t*h for t in terms]
    return func(*terms)


@cacheit
def _extract_delta(expr, index):
    """
    Extract a simple KroneckerDelta from the expression.

    Explanation
    ===========

    Returns the tuple ``(delta, newexpr)`` where:

      - ``delta`` is a simple KroneckerDelta expression if one was found,
        or ``None`` if no simple KroneckerDelta expression was found.

      - ``newexpr`` is a Mul containing the remaining terms; ``expr`` is
        returned unchanged if no simple KroneckerDelta expression was found.

    Examples
    ========

    >>> from sympy import KroneckerDelta
    >>> from sympy.concrete.delta import _extract_delta
    >>> from sympy.abc import x, y, i, j, k
    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), i)
    (KroneckerDelta(i, j), 4*x*y)
    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), k)
    (None, 4*x*y*KroneckerDelta(i, j))

    See Also
    ========

    sympy.functions.special.tensor_functions.KroneckerDelta
    deltaproduct
    deltasummation
    """
    if not _has_simple_delta(expr, index):
        return (None, expr)
    if isinstance(expr, KroneckerDelta):
        return (expr, S.One)
    if not expr.is_Mul:
        raise ValueError("Incorrect expr")
    delta = None
    terms = []

    for arg in expr.args:
        if delta is None and _is_simple_delta(arg, index):
            delta = arg
        else:
            terms.append(arg)
    return (delta, expr.func(*terms))


@cacheit
def _has_simple_delta(expr, index):
    """
    Returns True if ``expr`` is an expression that contains a KroneckerDelta
    that is simple in the index ``index``, meaning that this KroneckerDelta
    is nonzero for a single value of the index ``index``.
    """
    if expr.has(KroneckerDelta):
        if _is_simple_delta(expr, index):
            return True
        if expr.is_Add or expr.is_Mul:
            return any(_has_simple_delta(arg, index) for arg in expr.args)
    return False


@cacheit
def _is_simple_delta(delta, index):
    """
    Returns True if ``delta`` is a KroneckerDelta and is nonzero for a single
    value of the index ``index``.
    """
    if isinstance(delta, KroneckerDelta) and delta.has(index):
        p = (delta.args[0] - delta.args[1]).as_poly(index)
        if p:
            return p.degree() == 1
    return False


@cacheit
def _remove_multiple_delta(expr):
    """
    Evaluate products of KroneckerDelta's.
    """
    if expr.is_Add:
        return expr.func(*list(map(_remove_multiple_delta, expr.args)))
    if not expr.is_Mul:
        return expr
    eqs = []
    newargs = []
    for arg in expr.args:
        if isinstance(arg, KroneckerDelta):
            eqs.append(arg.args[0] - arg.args[1])
        else:
            newargs.append(arg)
    if not eqs:
        return expr
    solns = solve(eqs, dict=True)
    if len(solns) == 0:
        return S.Zero
    elif len(solns) == 1:
        newargs += [KroneckerDelta(k, v) for k, v in solns[0].items()]
        expr2 = expr.func(*newargs)
        if expr != expr2:
            return _remove_multiple_delta(expr2)
    return expr


@cacheit
def _simplify_delta(expr):
    """
    Rewrite a KroneckerDelta's indices in its simplest form.
    """
    if isinstance(expr, KroneckerDelta):
        try:
            slns = solve(expr.args[0] - expr.args[1], dict=True)
            if slns and len(slns) == 1:
                return Mul(*[KroneckerDelta(*(key, value))
                            for key, value in slns[0].items()])
        except NotImplementedError:
            pass
    return expr


@cacheit
def deltaproduct(f, limit):
    """
    Handle products containing a KroneckerDelta.

    See Also
    ========

    deltasummation
    sympy.functions.special.tensor_functions.KroneckerDelta
    sympy.concrete.products.product
    """
    if ((limit[2] - limit[1]) < 0) == True:
        return S.One

    if not f.has(KroneckerDelta):
        return product(f, limit)

    if f.is_Add:
        # Identify the term in the Add that has a simple KroneckerDelta
        delta = None
        terms = []
        for arg in sorted(f.args, key=default_sort_key):
            if delta is None and _has_simple_delta(arg, limit[0]):
                delta = arg
            else:
                terms.append(arg)
        newexpr = f.func(*terms)
        k = Dummy("kprime", integer=True)
        if isinstance(limit[1], int) and isinstance(limit[2], int):
            result = deltaproduct(newexpr, limit) + sum(deltaproduct(newexpr, (limit[0], limit[1], ik - 1)) *
                delta.subs(limit[0], ik) *
                deltaproduct(newexpr, (limit[0], ik + 1, limit[2])) for ik in range(int(limit[1]), int(limit[2] + 1))
            )
        else:
            result = deltaproduct(newexpr, limit) + deltasummation(
                deltaproduct(newexpr, (limit[0], limit[1], k - 1)) *
                delta.subs(limit[0], k) *
                deltaproduct(newexpr, (limit[0], k + 1, limit[2])),
                (k, limit[1], limit[2]),
                no_piecewise=_has_simple_delta(newexpr, limit[0])
            )
        return _remove_multiple_delta(result)

    delta, _ = _extract_delta(f, limit[0])

    if not delta:
        g = _expand_delta(f, limit[0])
        if f != g:
            try:
                return factor(deltaproduct(g, limit))
            except AssertionError:
                return deltaproduct(g, limit)
        return product(f, limit)

    return _remove_multiple_delta(f.subs(limit[0], limit[1])*KroneckerDelta(limit[2], limit[1])) + \
        S.One*_simplify_delta(KroneckerDelta(limit[2], limit[1] - 1))


@cacheit
def deltasummation(f, limit, no_piecewise=False):
    """
    Handle summations containing a KroneckerDelta.

    Explanation
    ===========

    The idea for summation is the following:

    - If we are dealing with a KroneckerDelta expression, i.e. KroneckerDelta(g(x), j),
      we try to simplify it.

      If we could simplify it, then we sum the resulting expression.
      We already know we can sum a simplified expression, because only
      simple KroneckerDelta expressions are involved.

      If we could not simplify it, there are two cases:

      1) The expression is a simple expression: we return the summation,
         taking care if we are dealing with a Derivative or with a proper
         KroneckerDelta.

      2) The expression is not simple (i.e. KroneckerDelta(cos(x))): we can do
         nothing at all.

    - If the expr is a multiplication expr having a KroneckerDelta term:

      First we expand it.

      If the expansion did work, then we try to sum the expansion.

      If not, we try to extract a simple KroneckerDelta term, then we have two
      cases:

      1) We have a simple KroneckerDelta term, so we return the summation.

      2) We did not have a simple term, but we do have an expression with
         simplified KroneckerDelta terms, so we sum this expression.

    Examples
    ========

    >>> from sympy import oo, symbols
    >>> from sympy.abc import k
    >>> i, j = symbols('i, j', integer=True, finite=True)
    >>> from sympy.concrete.delta import deltasummation
    >>> from sympy import KroneckerDelta
    >>> deltasummation(KroneckerDelta(i, k), (k, -oo, oo))
    1
    >>> deltasummation(KroneckerDelta(i, k), (k, 0, oo))
    Piecewise((1, i >= 0), (0, True))
    >>> deltasummation(KroneckerDelta(i, k), (k, 1, 3))
    Piecewise((1, (i >= 1) & (i <= 3)), (0, True))
    >>> deltasummation(k*KroneckerDelta(i, j)*KroneckerDelta(j, k), (k, -oo, oo))
    j*KroneckerDelta(i, j)
    >>> deltasummation(j*KroneckerDelta(i, j), (j, -oo, oo))
    i
    >>> deltasummation(i*KroneckerDelta(i, j), (i, -oo, oo))
    j

    See Also
    ========

    deltaproduct
    sympy.functions.special.tensor_functions.KroneckerDelta
    sympy.concrete.sums.summation
    """
    if ((limit[2] - limit[1]) < 0) == True:
        return S.Zero

    if not f.has(KroneckerDelta):
        return summation(f, limit)

    x = limit[0]

    g = _expand_delta(f, x)
    if g.is_Add:
        return piecewise_fold(
            g.func(*[deltasummation(h, limit, no_piecewise) for h in g.args]))

    # try to extract a simple KroneckerDelta term
    delta, expr = _extract_delta(g, x)

    if (delta is not None) and (delta.delta_range is not None):
        dinf, dsup = delta.delta_range
        if (limit[1] - dinf <= 0) == True and (limit[2] - dsup >= 0) == True:
            no_piecewise = True

    if not delta:
        return summation(f, limit)

    solns = solve(delta.args[0] - delta.args[1], x)
    if len(solns) == 0:
        return S.Zero
    elif len(solns) != 1:
        return Sum(f, limit)
    value = solns[0]
    if no_piecewise:
        return expr.subs(x, value)
    return Piecewise(
        (expr.subs(x, value), Interval(*limit[1:3]).as_relational(value)),
        (S.Zero, True)
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\pivot\fields.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    DateTime,
    Bool,
    Float,
    String,
    Integer,
    Sequence,
)
from openpyxl.descriptors.excel import HexBinary

class Index(Serialisable):

    tagname = "x"

    v = Integer(allow_none=True)

    def __init__(self,
                 v=0,
                ):
        self.v = v


class Tuple(Serialisable):

    tagname = "tpl"

    fld = Integer(allow_none=True)
    hier = Integer(allow_none=True)
    item = Integer()

    def __init__(self,
                 fld=None,
                 hier=None,
                 item=None,
                ):
        self.fld = fld
        self.hier = hier
        self.item = item


class TupleList(Serialisable):

    tagname = "tpls"

    c = Integer(allow_none=True)
    tpl = Typed(expected_type=Tuple, )

    __elements__ = ('tpl',)

    def __init__(self,
                 c=None,
                 tpl=None,
                ):
        self.c = c
        self.tpl = tpl


class Missing(Serialisable):

    tagname = "m"

    tpls = Sequence(expected_type=TupleList)
    x = Sequence(expected_type=Index)
    u = Bool(allow_none=True)
    f = Bool(allow_none=True)
    c = String(allow_none=True)
    cp = Integer(allow_none=True)
    _in = Integer(allow_none=True)
    bc = HexBinary(allow_none=True)
    fc = HexBinary(allow_none=True)
    i = Bool(allow_none=True)
    un = Bool(allow_none=True)
    st = Bool(allow_none=True)
    b = Bool(allow_none=True)

    __elements__ = ('tpls', 'x')

    def __init__(self,
                 tpls=(),
                 x=(),
                 u=None,
                 f=None,
                 c=None,
                 cp=None,
                 _in=None,
                 bc=None,
                 fc=None,
                 i=None,
                 un=None,
                 st=None,
                 b=None,
                ):
        self.tpls = tpls
        self.x = x
        self.u = u
        self.f = f
        self.c = c
        self.cp = cp
        self._in = _in
        self.bc = bc
        self.fc = fc
        self.i = i
        self.un = un
        self.st = st
        self.b = b


class Number(Serialisable):

    tagname = "n"

    tpls = Sequence(expected_type=TupleList)
    x = Sequence(expected_type=Index)
    v = Float()
    u = Bool(allow_none=True)
    f = Bool(allow_none=True)
    c = String(allow_none=True)
    cp = Integer(allow_none=True)
    _in = Integer(allow_none=True)
    bc = HexBinary(allow_none=True)
    fc = HexBinary(allow_none=True)
    i = Bool(allow_none=True)
    un = Bool(allow_none=True)
    st = Bool(allow_none=True)
    b = Bool(allow_none=True)

    __elements__ = ('tpls', 'x')

    def __init__(self,
                 tpls=(),
                 x=(),
                 v=None,
                 u=None,
                 f=None,
                 c=None,
                 cp=None,
                 _in=None,
                 bc=None,
                 fc=None,
                 i=None,
                 un=None,
                 st=None,
                 b=None,
                ):
        self.tpls = tpls
        self.x = x
        self.v = v
        self.u = u
        self.f = f
        self.c = c
        self.cp = cp
        self._in = _in
        self.bc = bc
        self.fc = fc
        self.i = i
        self.un = un
        self.st = st
        self.b = b


class Error(Serialisable):

    tagname = "e"

    tpls = Typed(expected_type=TupleList, allow_none=True)
    x = Sequence(expected_type=Index)
    v = String()
    u = Bool(allow_none=True)
    f = Bool(allow_none=True)
    c = String(allow_none=True)
    cp = Integer(allow_none=True)
    _in = Integer(allow_none=True)
    bc = HexBinary(allow_none=True)
    fc = HexBinary(allow_none=True)
    i = Bool(allow_none=True)
    un = Bool(allow_none=True)
    st = Bool(allow_none=True)
    b = Bool(allow_none=True)

    __elements__ = ('tpls', 'x')

    def __init__(self,
                 tpls=None,
                 x=(),
                 v=None,
                 u=None,
                 f=None,
                 c=None,
                 cp=None,
                 _in=None,
                 bc=None,
                 fc=None,
                 i=None,
                 un=None,
                 st=None,
                 b=None,
                ):
        self.tpls = tpls
        self.x = x
        self.v = v
        self.u = u
        self.f = f
        self.c = c
        self.cp = cp
        self._in = _in
        self.bc = bc
        self.fc = fc
        self.i = i
        self.un = un
        self.st = st
        self.b = b


class Boolean(Serialisable):

    tagname = "b"

    x = Sequence(expected_type=Index)
    v = Bool()
    u = Bool(allow_none=True)
    f = Bool(allow_none=True)
    c = String(allow_none=True)
    cp = Integer(allow_none=True)

    __elements__ = ('x',)

    def __init__(self,
                 x=(),
                 v=None,
                 u=None,
                 f=None,
                 c=None,
                 cp=None,
                ):
        self.x = x
        self.v = v
        self.u = u
        self.f = f
        self.c = c
        self.cp = cp


class Text(Serialisable):

    tagname = "s"

    tpls = Sequence(expected_type=TupleList)
    x = Sequence(expected_type=Index)
    v = String()
    u = Bool(allow_none=True)
    f = Bool(allow_none=True)
    c = String(allow_none=True)
    cp = Integer(allow_none=True)
    _in = Integer(allow_none=True)
    bc = HexBinary(allow_none=True)
    fc = HexBinary(allow_none=True)
    i = Bool(allow_none=True)
    un = Bool(allow_none=True)
    st = Bool(allow_none=True)
    b = Bool(allow_none=True)

    __elements__ = ('tpls', 'x')

    def __init__(self,
                 tpls=(),
                 x=(),
                 v=None,
                 u=None,
                 f=None,
                 c=None,
                 cp=None,
                 _in=None,
                 bc=None,
                 fc=None,
                 i=None,
                 un=None,
                 st=None,
                 b=None,
                 ):
        self.tpls = tpls
        self.x = x
        self.v = v
        self.u = u
        self.f = f
        self.c = c
        self.cp = cp
        self._in = _in
        self.bc = bc
        self.fc = fc
        self.i = i
        self.un = un
        self.st = st
        self.b = b


class DateTimeField(Serialisable):

    tagname = "d"

    x = Sequence(expected_type=Index)
    v = DateTime()
    u = Bool(allow_none=True)
    f = Bool(allow_none=True)
    c = String(allow_none=True)
    cp = Integer(allow_none=True)

    __elements__ = ('x',)

    def __init__(self,
                 x=(),
                 v=None,
                 u=None,
                 f=None,
                 c=None,
                 cp=None,
                 ):
        self.x = x
        self.v = v
        self.u = u
        self.f = f
        self.c = c
        self.cp = cp

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\build_env.py ===
"""Build Environment used for isolation during sdist building"""

import logging
import os
import pathlib
import site
import sys
import textwrap
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Type, Union

from pip._vendor.packaging.version import Version

from pip import __file__ as pip_location
from pip._internal.cli.spinners import open_spinner
from pip._internal.locations import get_platlib, get_purelib, get_scheme
from pip._internal.metadata import get_default_environment, get_environment
from pip._internal.utils.logging import VERBOSE
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder

logger = logging.getLogger(__name__)


def _dedup(a: str, b: str) -> Union[Tuple[str], Tuple[str, str]]:
    return (a, b) if a != b else (a,)


class _Prefix:
    def __init__(self, path: str) -> None:
        self.path = path
        self.setup = False
        scheme = get_scheme("", prefix=path)
        self.bin_dir = scheme.scripts
        self.lib_dirs = _dedup(scheme.purelib, scheme.platlib)


def get_runnable_pip() -> str:
    """Get a file to pass to a Python executable, to run the currently-running pip.

    This is used to run a pip subprocess, for installing requirements into the build
    environment.
    """
    source = pathlib.Path(pip_location).resolve().parent

    if not source.is_dir():
        # This would happen if someone is using pip from inside a zip file. In that
        # case, we can use that directly.
        return str(source)

    return os.fsdecode(source / "__pip-runner__.py")


def _get_system_sitepackages() -> Set[str]:
    """Get system site packages

    Usually from site.getsitepackages,
    but fallback on `get_purelib()/get_platlib()` if unavailable
    (e.g. in a virtualenv created by virtualenv<20)

    Returns normalized set of strings.
    """
    if hasattr(site, "getsitepackages"):
        system_sites = site.getsitepackages()
    else:
        # virtualenv < 20 overwrites site.py without getsitepackages
        # fallback on get_purelib/get_platlib.
        # this is known to miss things, but shouldn't in the cases
        # where getsitepackages() has been removed (inside a virtualenv)
        system_sites = [get_purelib(), get_platlib()]
    return {os.path.normcase(path) for path in system_sites}


class BuildEnvironment:
    """Creates and manages an isolated environment to install build deps"""

    def __init__(self) -> None:
        temp_dir = TempDirectory(kind=tempdir_kinds.BUILD_ENV, globally_managed=True)

        self._prefixes = OrderedDict(
            (name, _Prefix(os.path.join(temp_dir.path, name)))
            for name in ("normal", "overlay")
        )

        self._bin_dirs: List[str] = []
        self._lib_dirs: List[str] = []
        for prefix in reversed(list(self._prefixes.values())):
            self._bin_dirs.append(prefix.bin_dir)
            self._lib_dirs.extend(prefix.lib_dirs)

        # Customize site to:
        # - ensure .pth files are honored
        # - prevent access to system site packages
        system_sites = _get_system_sitepackages()

        self._site_dir = os.path.join(temp_dir.path, "site")
        if not os.path.exists(self._site_dir):
            os.mkdir(self._site_dir)
        with open(
            os.path.join(self._site_dir, "sitecustomize.py"), "w", encoding="utf-8"
        ) as fp:
            fp.write(
                textwrap.dedent(
                    """
                import os, site, sys

                # First, drop system-sites related paths.
                original_sys_path = sys.path[:]
                known_paths = set()
                for path in {system_sites!r}:
                    site.addsitedir(path, known_paths=known_paths)
                system_paths = set(
                    os.path.normcase(path)
                    for path in sys.path[len(original_sys_path):]
                )
                original_sys_path = [
                    path for path in original_sys_path
                    if os.path.normcase(path) not in system_paths
                ]
                sys.path = original_sys_path

                # Second, add lib directories.
                # ensuring .pth file are processed.
                for path in {lib_dirs!r}:
                    assert not path in sys.path
                    site.addsitedir(path)
                """
                ).format(system_sites=system_sites, lib_dirs=self._lib_dirs)
            )

    def __enter__(self) -> None:
        self._save_env = {
            name: os.environ.get(name, None)
            for name in ("PATH", "PYTHONNOUSERSITE", "PYTHONPATH")
        }

        path = self._bin_dirs[:]
        old_path = self._save_env["PATH"]
        if old_path:
            path.extend(old_path.split(os.pathsep))

        pythonpath = [self._site_dir]

        os.environ.update(
            {
                "PATH": os.pathsep.join(path),
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(pythonpath),
            }
        )

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        for varname, old_value in self._save_env.items():
            if old_value is None:
                os.environ.pop(varname, None)
            else:
                os.environ[varname] = old_value

    def check_requirements(
        self, reqs: Iterable[str]
    ) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        """Return 2 sets:
        - conflicting requirements: set of (installed, wanted) reqs tuples
        - missing requirements: set of reqs
        """
        missing = set()
        conflicting = set()
        if reqs:
            env = (
                get_environment(self._lib_dirs)
                if hasattr(self, "_lib_dirs")
                else get_default_environment()
            )
            for req_str in reqs:
                req = get_requirement(req_str)
                # We're explicitly evaluating with an empty extra value, since build
                # environments are not provided any mechanism to select specific extras.
                if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                    continue
                dist = env.get_distribution(req.name)
                if not dist:
                    missing.add(req_str)
                    continue
                if isinstance(dist.version, Version):
                    installed_req_str = f"{req.name}=={dist.version}"
                else:
                    installed_req_str = f"{req.name}==={dist.version}"
                if not req.specifier.contains(dist.version, prereleases=True):
                    conflicting.add((installed_req_str, req_str))
                # FIXME: Consider direct URL?
        return conflicting, missing

    def install_requirements(
        self,
        finder: "PackageFinder",
        requirements: Iterable[str],
        prefix_as_string: str,
        *,
        kind: str,
    ) -> None:
        prefix = self._prefixes[prefix_as_string]
        assert not prefix.setup
        prefix.setup = True
        if not requirements:
            return
        self._install_requirements(
            get_runnable_pip(),
            finder,
            requirements,
            prefix,
            kind=kind,
        )

    @staticmethod
    def _install_requirements(
        pip_runnable: str,
        finder: "PackageFinder",
        requirements: Iterable[str],
        prefix: _Prefix,
        *,
        kind: str,
    ) -> None:
        args: List[str] = [
            sys.executable,
            pip_runnable,
            "install",
            "--ignore-installed",
            "--no-user",
            "--prefix",
            prefix.path,
            "--no-warn-script-location",
            "--disable-pip-version-check",
            # As the build environment is ephemeral, it's wasteful to
            # pre-compile everything, especially as not every Python
            # module will be used/compiled in most cases.
            "--no-compile",
            # The prefix specified two lines above, thus
            # target from config file or env var should be ignored
            "--target",
            "",
        ]
        if logger.getEffectiveLevel() <= logging.DEBUG:
            args.append("-vv")
        elif logger.getEffectiveLevel() <= VERBOSE:
            args.append("-v")
        for format_control in ("no_binary", "only_binary"):
            formats = getattr(finder.format_control, format_control)
            args.extend(
                (
                    "--" + format_control.replace("_", "-"),
                    ",".join(sorted(formats or {":none:"})),
                )
            )

        index_urls = finder.index_urls
        if index_urls:
            args.extend(["-i", index_urls[0]])
            for extra_index in index_urls[1:]:
                args.extend(["--extra-index-url", extra_index])
        else:
            args.append("--no-index")
        for link in finder.find_links:
            args.extend(["--find-links", link])

        if finder.proxy:
            args.extend(["--proxy", finder.proxy])
        for host in finder.trusted_hosts:
            args.extend(["--trusted-host", host])
        if finder.custom_cert:
            args.extend(["--cert", finder.custom_cert])
        if finder.client_cert:
            args.extend(["--client-cert", finder.client_cert])
        if finder.allow_all_prereleases:
            args.append("--pre")
        if finder.prefer_binary:
            args.append("--prefer-binary")
        args.append("--")
        args.extend(requirements)
        with open_spinner(f"Installing {kind}") as spinner:
            call_subprocess(
                args,
                command_desc=f"pip subprocess to install {kind}",
                spinner=spinner,
            )


class NoOpBuildEnvironment(BuildEnvironment):
    """A no-op drop-in replacement for BuildEnvironment"""

    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def install_requirements(
        self,
        finder: "PackageFinder",
        requirements: Iterable[str],
        prefix_as_string: str,
        *,
        kind: str,
    ) -> None:
        raise NotImplementedError()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\vcs\subversion.py ===
import logging
import os
import re
from typing import List, Optional, Tuple

from pip._internal.utils.misc import (
    HiddenText,
    display_path,
    is_console_interactive,
    is_installable_dir,
    split_auth_from_netloc,
)
from pip._internal.utils.subprocess import CommandArgs, make_command
from pip._internal.vcs.versioncontrol import (
    AuthInfo,
    RemoteNotFoundError,
    RevOptions,
    VersionControl,
    vcs,
)

logger = logging.getLogger(__name__)

_svn_xml_url_re = re.compile('url="([^"]+)"')
_svn_rev_re = re.compile(r'committed-rev="(\d+)"')
_svn_info_xml_rev_re = re.compile(r'\s*revision="(\d+)"')
_svn_info_xml_url_re = re.compile(r"<url>(.*)</url>")


class Subversion(VersionControl):
    name = "svn"
    dirname = ".svn"
    repo_name = "checkout"
    schemes = ("svn+ssh", "svn+http", "svn+https", "svn+svn", "svn+file")

    @classmethod
    def should_add_vcs_url_prefix(cls, remote_url: str) -> bool:
        return True

    @staticmethod
    def get_base_rev_args(rev: str) -> List[str]:
        return ["-r", rev]

    @classmethod
    def get_revision(cls, location: str) -> str:
        """
        Return the maximum revision for all files under a given location
        """
        # Note: taken from setuptools.command.egg_info
        revision = 0

        for base, dirs, _ in os.walk(location):
            if cls.dirname not in dirs:
                dirs[:] = []
                continue  # no sense walking uncontrolled subdirs
            dirs.remove(cls.dirname)
            entries_fn = os.path.join(base, cls.dirname, "entries")
            if not os.path.exists(entries_fn):
                # FIXME: should we warn?
                continue

            dirurl, localrev = cls._get_svn_url_rev(base)

            if base == location:
                assert dirurl is not None
                base = dirurl + "/"  # save the root url
            elif not dirurl or not dirurl.startswith(base):
                dirs[:] = []
                continue  # not part of the same svn tree, skip it
            revision = max(revision, localrev)
        return str(revision)

    @classmethod
    def get_netloc_and_auth(
        cls, netloc: str, scheme: str
    ) -> Tuple[str, Tuple[Optional[str], Optional[str]]]:
        """
        This override allows the auth information to be passed to svn via the
        --username and --password options instead of via the URL.
        """
        if scheme == "ssh":
            # The --username and --password options can't be used for
            # svn+ssh URLs, so keep the auth information in the URL.
            return super().get_netloc_and_auth(netloc, scheme)

        return split_auth_from_netloc(netloc)

    @classmethod
    def get_url_rev_and_auth(cls, url: str) -> Tuple[str, Optional[str], AuthInfo]:
        # hotfix the URL scheme after removing svn+ from svn+ssh:// re-add it
        url, rev, user_pass = super().get_url_rev_and_auth(url)
        if url.startswith("ssh://"):
            url = "svn+" + url
        return url, rev, user_pass

    @staticmethod
    def make_rev_args(
        username: Optional[str], password: Optional[HiddenText]
    ) -> CommandArgs:
        extra_args: CommandArgs = []
        if username:
            extra_args += ["--username", username]
        if password:
            extra_args += ["--password", password]

        return extra_args

    @classmethod
    def get_remote_url(cls, location: str) -> str:
        # In cases where the source is in a subdirectory, we have to look up in
        # the location until we find a valid project root.
        orig_location = location
        while not is_installable_dir(location):
            last_location = location
            location = os.path.dirname(location)
            if location == last_location:
                # We've traversed up to the root of the filesystem without
                # finding a Python project.
                logger.warning(
                    "Could not find Python project for directory %s (tried all "
                    "parent directories)",
                    orig_location,
                )
                raise RemoteNotFoundError

        url, _rev = cls._get_svn_url_rev(location)
        if url is None:
            raise RemoteNotFoundError

        return url

    @classmethod
    def _get_svn_url_rev(cls, location: str) -> Tuple[Optional[str], int]:
        from pip._internal.exceptions import InstallationError

        entries_path = os.path.join(location, cls.dirname, "entries")
        if os.path.exists(entries_path):
            with open(entries_path) as f:
                data = f.read()
        else:  # subversion >= 1.7 does not have the 'entries' file
            data = ""

        url = None
        if data.startswith("8") or data.startswith("9") or data.startswith("10"):
            entries = list(map(str.splitlines, data.split("\n\x0c\n")))
            del entries[0][0]  # get rid of the '8'
            url = entries[0][3]
            revs = [int(d[9]) for d in entries if len(d) > 9 and d[9]] + [0]
        elif data.startswith("<?xml"):
            match = _svn_xml_url_re.search(data)
            if not match:
                raise ValueError(f"Badly formatted data: {data!r}")
            url = match.group(1)  # get repository URL
            revs = [int(m.group(1)) for m in _svn_rev_re.finditer(data)] + [0]
        else:
            try:
                # subversion >= 1.7
                # Note that using get_remote_call_options is not necessary here
                # because `svn info` is being run against a local directory.
                # We don't need to worry about making sure interactive mode
                # is being used to prompt for passwords, because passwords
                # are only potentially needed for remote server requests.
                xml = cls.run_command(
                    ["info", "--xml", location],
                    show_stdout=False,
                    stdout_only=True,
                )
                match = _svn_info_xml_url_re.search(xml)
                assert match is not None
                url = match.group(1)
                revs = [int(m.group(1)) for m in _svn_info_xml_rev_re.finditer(xml)]
            except InstallationError:
                url, revs = None, []

        if revs:
            rev = max(revs)
        else:
            rev = 0

        return url, rev

    @classmethod
    def is_commit_id_equal(cls, dest: str, name: Optional[str]) -> bool:
        """Always assume the versions don't match"""
        return False

    def __init__(self, use_interactive: Optional[bool] = None) -> None:
        if use_interactive is None:
            use_interactive = is_console_interactive()
        self.use_interactive = use_interactive

        # This member is used to cache the fetched version of the current
        # ``svn`` client.
        # Special value definitions:
        #   None: Not evaluated yet.
        #   Empty tuple: Could not parse version.
        self._vcs_version: Optional[Tuple[int, ...]] = None

        super().__init__()

    def call_vcs_version(self) -> Tuple[int, ...]:
        """Query the version of the currently installed Subversion client.

        :return: A tuple containing the parts of the version information or
            ``()`` if the version returned from ``svn`` could not be parsed.
        :raises: BadCommand: If ``svn`` is not installed.
        """
        # Example versions:
        #   svn, version 1.10.3 (r1842928)
        #      compiled Feb 25 2019, 14:20:39 on x86_64-apple-darwin17.0.0
        #   svn, version 1.7.14 (r1542130)
        #      compiled Mar 28 2018, 08:49:13 on x86_64-pc-linux-gnu
        #   svn, version 1.12.0-SlikSvn (SlikSvn/1.12.0)
        #      compiled May 28 2019, 13:44:56 on x86_64-microsoft-windows6.2
        version_prefix = "svn, version "
        version = self.run_command(["--version"], show_stdout=False, stdout_only=True)
        if not version.startswith(version_prefix):
            return ()

        version = version[len(version_prefix) :].split()[0]
        version_list = version.partition("-")[0].split(".")
        try:
            parsed_version = tuple(map(int, version_list))
        except ValueError:
            return ()

        return parsed_version

    def get_vcs_version(self) -> Tuple[int, ...]:
        """Return the version of the currently installed Subversion client.

        If the version of the Subversion client has already been queried,
        a cached value will be used.

        :return: A tuple containing the parts of the version information or
            ``()`` if the version returned from ``svn`` could not be parsed.
        :raises: BadCommand: If ``svn`` is not installed.
        """
        if self._vcs_version is not None:
            # Use cached version, if available.
            # If parsing the version failed previously (empty tuple),
            # do not attempt to parse it again.
            return self._vcs_version

        vcs_version = self.call_vcs_version()
        self._vcs_version = vcs_version
        return vcs_version

    def get_remote_call_options(self) -> CommandArgs:
        """Return options to be used on calls to Subversion that contact the server.

        These options are applicable for the following ``svn`` subcommands used
        in this class.

            - checkout
            - switch
            - update

        :return: A list of command line arguments to pass to ``svn``.
        """
        if not self.use_interactive:
            # --non-interactive switch is available since Subversion 0.14.4.
            # Subversion < 1.8 runs in interactive mode by default.
            return ["--non-interactive"]

        svn_version = self.get_vcs_version()
        # By default, Subversion >= 1.8 runs in non-interactive mode if
        # stdin is not a TTY. Since that is how pip invokes SVN, in
        # call_subprocess(), pip must pass --force-interactive to ensure
        # the user can be prompted for a password, if required.
        #   SVN added the --force-interactive option in SVN 1.8. Since
        # e.g. RHEL/CentOS 7, which is supported until 2024, ships with
        # SVN 1.7, pip should continue to support SVN 1.7. Therefore, pip
        # can't safely add the option if the SVN version is < 1.8 (or unknown).
        if svn_version >= (1, 8):
            return ["--force-interactive"]

        return []

    def fetch_new(
        self, dest: str, url: HiddenText, rev_options: RevOptions, verbosity: int
    ) -> None:
        rev_display = rev_options.to_display()
        logger.info(
            "Checking out %s%s to %s",
            url,
            rev_display,
            display_path(dest),
        )
        if verbosity <= 0:
            flags = ["--quiet"]
        else:
            flags = []
        cmd_args = make_command(
            "checkout",
            *flags,
            self.get_remote_call_options(),
            rev_options.to_args(),
            url,
            dest,
        )
        self.run_command(cmd_args)

    def switch(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        cmd_args = make_command(
            "switch",
            self.get_remote_call_options(),
            rev_options.to_args(),
            url,
            dest,
        )
        self.run_command(cmd_args)

    def update(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        cmd_args = make_command(
            "update",
            self.get_remote_call_options(),
            rev_options.to_args(),
            dest,
        )
        self.run_command(cmd_args)


vcs.register(Subversion)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\util.py ===
"""
    pygments.util
    ~~~~~~~~~~~~~

    Utility functions.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re
from io import TextIOWrapper


split_path_re = re.compile(r'[/\\ ]')
doctype_lookup_re = re.compile(r'''
    <!DOCTYPE\s+(
     [a-zA-Z_][a-zA-Z0-9]*
     (?: \s+      # optional in HTML5
     [a-zA-Z_][a-zA-Z0-9]*\s+
     "[^"]*")?
     )
     [^>]*>
''', re.DOTALL | re.MULTILINE | re.VERBOSE)
tag_re = re.compile(r'<(.+?)(\s.*?)?>.*?</.+?>',
                    re.IGNORECASE | re.DOTALL | re.MULTILINE)
xml_decl_re = re.compile(r'\s*<\?xml[^>]*\?>', re.I)


class ClassNotFound(ValueError):
    """Raised if one of the lookup functions didn't find a matching class."""


class OptionError(Exception):
    """
    This exception will be raised by all option processing functions if
    the type or value of the argument is not correct.
    """

def get_choice_opt(options, optname, allowed, default=None, normcase=False):
    """
    If the key `optname` from the dictionary is not in the sequence
    `allowed`, raise an error, otherwise return it.
    """
    string = options.get(optname, default)
    if normcase:
        string = string.lower()
    if string not in allowed:
        raise OptionError('Value for option {} must be one of {}'.format(optname, ', '.join(map(str, allowed))))
    return string


def get_bool_opt(options, optname, default=None):
    """
    Intuitively, this is `options.get(optname, default)`, but restricted to
    Boolean value. The Booleans can be represented as string, in order to accept
    Boolean value from the command line arguments. If the key `optname` is
    present in the dictionary `options` and is not associated with a Boolean,
    raise an `OptionError`. If it is absent, `default` is returned instead.

    The valid string values for ``True`` are ``1``, ``yes``, ``true`` and
    ``on``, the ones for ``False`` are ``0``, ``no``, ``false`` and ``off``
    (matched case-insensitively).
    """
    string = options.get(optname, default)
    if isinstance(string, bool):
        return string
    elif isinstance(string, int):
        return bool(string)
    elif not isinstance(string, str):
        raise OptionError(f'Invalid type {string!r} for option {optname}; use '
                          '1/0, yes/no, true/false, on/off')
    elif string.lower() in ('1', 'yes', 'true', 'on'):
        return True
    elif string.lower() in ('0', 'no', 'false', 'off'):
        return False
    else:
        raise OptionError(f'Invalid value {string!r} for option {optname}; use '
                          '1/0, yes/no, true/false, on/off')


def get_int_opt(options, optname, default=None):
    """As :func:`get_bool_opt`, but interpret the value as an integer."""
    string = options.get(optname, default)
    try:
        return int(string)
    except TypeError:
        raise OptionError(f'Invalid type {string!r} for option {optname}; you '
                          'must give an integer value')
    except ValueError:
        raise OptionError(f'Invalid value {string!r} for option {optname}; you '
                          'must give an integer value')

def get_list_opt(options, optname, default=None):
    """
    If the key `optname` from the dictionary `options` is a string,
    split it at whitespace and return it. If it is already a list
    or a tuple, it is returned as a list.
    """
    val = options.get(optname, default)
    if isinstance(val, str):
        return val.split()
    elif isinstance(val, (list, tuple)):
        return list(val)
    else:
        raise OptionError(f'Invalid type {val!r} for option {optname}; you '
                          'must give a list value')


def docstring_headline(obj):
    if not obj.__doc__:
        return ''
    res = []
    for line in obj.__doc__.strip().splitlines():
        if line.strip():
            res.append(" " + line.strip())
        else:
            break
    return ''.join(res).lstrip()


def make_analysator(f):
    """Return a static text analyser function that returns float values."""
    def text_analyse(text):
        try:
            rv = f(text)
        except Exception:
            return 0.0
        if not rv:
            return 0.0
        try:
            return min(1.0, max(0.0, float(rv)))
        except (ValueError, TypeError):
            return 0.0
    text_analyse.__doc__ = f.__doc__
    return staticmethod(text_analyse)


def shebang_matches(text, regex):
    r"""Check if the given regular expression matches the last part of the
    shebang if one exists.

        >>> from pygments.util import shebang_matches
        >>> shebang_matches('#!/usr/bin/env python', r'python(2\.\d)?')
        True
        >>> shebang_matches('#!/usr/bin/python2.4', r'python(2\.\d)?')
        True
        >>> shebang_matches('#!/usr/bin/python-ruby', r'python(2\.\d)?')
        False
        >>> shebang_matches('#!/usr/bin/python/ruby', r'python(2\.\d)?')
        False
        >>> shebang_matches('#!/usr/bin/startsomethingwith python',
        ...                 r'python(2\.\d)?')
        True

    It also checks for common windows executable file extensions::

        >>> shebang_matches('#!C:\\Python2.4\\Python.exe', r'python(2\.\d)?')
        True

    Parameters (``'-f'`` or ``'--foo'`` are ignored so ``'perl'`` does
    the same as ``'perl -e'``)

    Note that this method automatically searches the whole string (eg:
    the regular expression is wrapped in ``'^$'``)
    """
    index = text.find('\n')
    if index >= 0:
        first_line = text[:index].lower()
    else:
        first_line = text.lower()
    if first_line.startswith('#!'):
        try:
            found = [x for x in split_path_re.split(first_line[2:].strip())
                     if x and not x.startswith('-')][-1]
        except IndexError:
            return False
        regex = re.compile(rf'^{regex}(\.(exe|cmd|bat|bin))?$', re.IGNORECASE)
        if regex.search(found) is not None:
            return True
    return False


def doctype_matches(text, regex):
    """Check if the doctype matches a regular expression (if present).

    Note that this method only checks the first part of a DOCTYPE.
    eg: 'html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"'
    """
    m = doctype_lookup_re.search(text)
    if m is None:
        return False
    doctype = m.group(1)
    return re.compile(regex, re.I).match(doctype.strip()) is not None


def html_doctype_matches(text):
    """Check if the file looks like it has a html doctype."""
    return doctype_matches(text, r'html')


_looks_like_xml_cache = {}


def looks_like_xml(text):
    """Check if a doctype exists or if we have some tags."""
    if xml_decl_re.match(text):
        return True
    key = hash(text)
    try:
        return _looks_like_xml_cache[key]
    except KeyError:
        m = doctype_lookup_re.search(text)
        if m is not None:
            return True
        rv = tag_re.search(text[:1000]) is not None
        _looks_like_xml_cache[key] = rv
        return rv


def surrogatepair(c):
    """Given a unicode character code with length greater than 16 bits,
    return the two 16 bit surrogate pair.
    """
    # From example D28 of:
    # http://www.unicode.org/book/ch03.pdf
    return (0xd7c0 + (c >> 10), (0xdc00 + (c & 0x3ff)))


def format_lines(var_name, seq, raw=False, indent_level=0):
    """Formats a sequence of strings for output."""
    lines = []
    base_indent = ' ' * indent_level * 4
    inner_indent = ' ' * (indent_level + 1) * 4
    lines.append(base_indent + var_name + ' = (')
    if raw:
        # These should be preformatted reprs of, say, tuples.
        for i in seq:
            lines.append(inner_indent + i + ',')
    else:
        for i in seq:
            # Force use of single quotes
            r = repr(i + '"')
            lines.append(inner_indent + r[:-2] + r[-1] + ',')
    lines.append(base_indent + ')')
    return '\n'.join(lines)


def duplicates_removed(it, already_seen=()):
    """
    Returns a list with duplicates removed from the iterable `it`.

    Order is preserved.
    """
    lst = []
    seen = set()
    for i in it:
        if i in seen or i in already_seen:
            continue
        lst.append(i)
        seen.add(i)
    return lst


class Future:
    """Generic class to defer some work.

    Handled specially in RegexLexerMeta, to support regex string construction at
    first use.
    """
    def get(self):
        raise NotImplementedError


def guess_decode(text):
    """Decode *text* with guessed encoding.

    First try UTF-8; this should fail for non-UTF-8 encodings.
    Then try the preferred locale encoding.
    Fall back to latin-1, which always works.
    """
    try:
        text = text.decode('utf-8')
        return text, 'utf-8'
    except UnicodeDecodeError:
        try:
            import locale
            prefencoding = locale.getpreferredencoding()
            text = text.decode()
            return text, prefencoding
        except (UnicodeDecodeError, LookupError):
            text = text.decode('latin1')
            return text, 'latin1'


def guess_decode_from_terminal(text, term):
    """Decode *text* coming from terminal *term*.

    First try the terminal encoding, if given.
    Then try UTF-8.  Then try the preferred locale encoding.
    Fall back to latin-1, which always works.
    """
    if getattr(term, 'encoding', None):
        try:
            text = text.decode(term.encoding)
        except UnicodeDecodeError:
            pass
        else:
            return text, term.encoding
    return guess_decode(text)


def terminal_encoding(term):
    """Return our best guess of encoding for the given *term*."""
    if getattr(term, 'encoding', None):
        return term.encoding
    import locale
    return locale.getpreferredencoding()


class UnclosingTextIOWrapper(TextIOWrapper):
    # Don't close underlying buffer on destruction.
    def close(self):
        self.flush()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\mypy\apply.py ===
# ext/mypy/apply.py
# Copyright (C) 2021-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import List
from typing import Optional
from typing import Union

from mypy.nodes import ARG_NAMED_OPT
from mypy.nodes import Argument
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import MDEF
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import SymbolTableNode
from mypy.nodes import TempNode
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.plugins.common import add_method_to_class
from mypy.types import AnyType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneTyp
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnboundType
from mypy.types import UnionType

from . import infer
from . import util
from .names import expr_to_mapped_constructor
from .names import NAMED_TYPE_SQLA_MAPPED


def apply_mypy_mapped_attr(
    cls: ClassDef,
    api: SemanticAnalyzerPluginInterface,
    item: Union[NameExpr, StrExpr],
    attributes: List[util.SQLAlchemyAttribute],
) -> None:
    if isinstance(item, NameExpr):
        name = item.name
    elif isinstance(item, StrExpr):
        name = item.value
    else:
        return None

    for stmt in cls.defs.body:
        if (
            isinstance(stmt, AssignmentStmt)
            and isinstance(stmt.lvalues[0], NameExpr)
            and stmt.lvalues[0].name == name
        ):
            break
    else:
        util.fail(api, f"Can't find mapped attribute {name}", cls)
        return None

    if stmt.type is None:
        util.fail(
            api,
            "Statement linked from _mypy_mapped_attrs has no "
            "typing information",
            stmt,
        )
        return None

    left_hand_explicit_type = get_proper_type(stmt.type)
    assert isinstance(
        left_hand_explicit_type, (Instance, UnionType, UnboundType)
    )

    attributes.append(
        util.SQLAlchemyAttribute(
            name=name,
            line=item.line,
            column=item.column,
            typ=left_hand_explicit_type,
            info=cls.info,
        )
    )

    apply_type_to_mapped_statement(
        api, stmt, stmt.lvalues[0], left_hand_explicit_type, None
    )


def re_apply_declarative_assignments(
    cls: ClassDef,
    api: SemanticAnalyzerPluginInterface,
    attributes: List[util.SQLAlchemyAttribute],
) -> None:
    """For multiple class passes, re-apply our left-hand side types as mypy
    seems to reset them in place.

    """
    mapped_attr_lookup = {attr.name: attr for attr in attributes}
    update_cls_metadata = False

    for stmt in cls.defs.body:
        # for a re-apply, all of our statements are AssignmentStmt;
        # @declared_attr calls will have been converted and this
        # currently seems to be preserved by mypy (but who knows if this
        # will change).
        if (
            isinstance(stmt, AssignmentStmt)
            and isinstance(stmt.lvalues[0], NameExpr)
            and stmt.lvalues[0].name in mapped_attr_lookup
            and isinstance(stmt.lvalues[0].node, Var)
        ):
            left_node = stmt.lvalues[0].node

            python_type_for_type = mapped_attr_lookup[
                stmt.lvalues[0].name
            ].type

            left_node_proper_type = get_proper_type(left_node.type)

            # if we have scanned an UnboundType and now there's a more
            # specific type than UnboundType, call the re-scan so we
            # can get that set up correctly
            if (
                isinstance(python_type_for_type, UnboundType)
                and not isinstance(left_node_proper_type, UnboundType)
                and (
                    isinstance(stmt.rvalue, CallExpr)
                    and isinstance(stmt.rvalue.callee, MemberExpr)
                    and isinstance(stmt.rvalue.callee.expr, NameExpr)
                    and stmt.rvalue.callee.expr.node is not None
                    and stmt.rvalue.callee.expr.node.fullname
                    == NAMED_TYPE_SQLA_MAPPED
                    and stmt.rvalue.callee.name == "_empty_constructor"
                    and isinstance(stmt.rvalue.args[0], CallExpr)
                    and isinstance(stmt.rvalue.args[0].callee, RefExpr)
                )
            ):
                new_python_type_for_type = (
                    infer.infer_type_from_right_hand_nameexpr(
                        api,
                        stmt,
                        left_node,
                        left_node_proper_type,
                        stmt.rvalue.args[0].callee,
                    )
                )

                if new_python_type_for_type is not None and not isinstance(
                    new_python_type_for_type, UnboundType
                ):
                    python_type_for_type = new_python_type_for_type

                    # update the SQLAlchemyAttribute with the better
                    # information
                    mapped_attr_lookup[stmt.lvalues[0].name].type = (
                        python_type_for_type
                    )

                    update_cls_metadata = True

            if (
                not isinstance(left_node.type, Instance)
                or left_node.type.type.fullname != NAMED_TYPE_SQLA_MAPPED
            ):
                assert python_type_for_type is not None
                left_node.type = api.named_type(
                    NAMED_TYPE_SQLA_MAPPED, [python_type_for_type]
                )

    if update_cls_metadata:
        util.set_mapped_attributes(cls.info, attributes)


def apply_type_to_mapped_statement(
    api: SemanticAnalyzerPluginInterface,
    stmt: AssignmentStmt,
    lvalue: NameExpr,
    left_hand_explicit_type: Optional[ProperType],
    python_type_for_type: Optional[ProperType],
) -> None:
    """Apply the Mapped[<type>] annotation and right hand object to a
    declarative assignment statement.

    This converts a Python declarative class statement such as::

        class User(Base):
            # ...

            attrname = Column(Integer)

    To one that describes the final Python behavior to Mypy::

    ... format: off

        class User(Base):
            # ...

            attrname : Mapped[Optional[int]] = <meaningless temp node>

    ... format: on

    """
    left_node = lvalue.node
    assert isinstance(left_node, Var)

    # to be completely honest I have no idea what the difference between
    # left_node.type and stmt.type is, what it means if these are different
    # vs. the same, why in order to get tests to pass I have to assign
    # to stmt.type for the second case and not the first.  this is complete
    # trying every combination until it works stuff.

    if left_hand_explicit_type is not None:
        lvalue.is_inferred_def = False
        left_node.type = api.named_type(
            NAMED_TYPE_SQLA_MAPPED, [left_hand_explicit_type]
        )
    else:
        lvalue.is_inferred_def = False
        left_node.type = api.named_type(
            NAMED_TYPE_SQLA_MAPPED,
            (
                [AnyType(TypeOfAny.special_form)]
                if python_type_for_type is None
                else [python_type_for_type]
            ),
        )

    # so to have it skip the right side totally, we can do this:
    # stmt.rvalue = TempNode(AnyType(TypeOfAny.special_form))

    # however, if we instead manufacture a new node that uses the old
    # one, then we can still get type checking for the call itself,
    # e.g. the Column, relationship() call, etc.

    # rewrite the node as:
    # <attr> : Mapped[<typ>] =
    # _sa_Mapped._empty_constructor(<original CallExpr from rvalue>)
    # the original right-hand side is maintained so it gets type checked
    # internally
    stmt.rvalue = expr_to_mapped_constructor(stmt.rvalue)

    if stmt.type is not None and python_type_for_type is not None:
        stmt.type = python_type_for_type


def add_additional_orm_attributes(
    cls: ClassDef,
    api: SemanticAnalyzerPluginInterface,
    attributes: List[util.SQLAlchemyAttribute],
) -> None:
    """Apply __init__, __table__ and other attributes to the mapped class."""

    info = util.info_for_cls(cls, api)

    if info is None:
        return

    is_base = util.get_is_base(info)

    if "__init__" not in info.names and not is_base:
        mapped_attr_names = {attr.name: attr.type for attr in attributes}

        for base in info.mro[1:-1]:
            if "sqlalchemy" not in info.metadata:
                continue

            base_cls_attributes = util.get_mapped_attributes(base, api)
            if base_cls_attributes is None:
                continue

            for attr in base_cls_attributes:
                mapped_attr_names.setdefault(attr.name, attr.type)

        arguments = []
        for name, typ in mapped_attr_names.items():
            if typ is None:
                typ = AnyType(TypeOfAny.special_form)
            arguments.append(
                Argument(
                    variable=Var(name, typ),
                    type_annotation=typ,
                    initializer=TempNode(typ),
                    kind=ARG_NAMED_OPT,
                )
            )

        add_method_to_class(api, cls, "__init__", arguments, NoneTyp())

    if "__table__" not in info.names and util.get_has_table(info):
        _apply_placeholder_attr_to_class(
            api, cls, "sqlalchemy.sql.schema.Table", "__table__"
        )
    if not is_base:
        _apply_placeholder_attr_to_class(
            api, cls, "sqlalchemy.orm.mapper.Mapper", "__mapper__"
        )


def _apply_placeholder_attr_to_class(
    api: SemanticAnalyzerPluginInterface,
    cls: ClassDef,
    qualified_name: str,
    attrname: str,
) -> None:
    sym = api.lookup_fully_qualified_or_none(qualified_name)
    if sym:
        assert isinstance(sym.node, TypeInfo)
        type_: ProperType = Instance(sym.node, [])
    else:
        type_ = AnyType(TypeOfAny.special_form)
    var = Var(attrname)
    var._fullname = cls.fullname + "." + attrname
    var.info = cls.info
    var.type = type_
    cls.info.names[attrname] = SymbolTableNode(MDEF, var)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\profiling.py ===
# testing/profiling.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""Profiling support for unit and performance tests.

These are special purpose profiling methods which operate
in a more fine-grained way than nose's profiling plugin.

"""

from __future__ import annotations

import collections
import contextlib
import os
import platform
import pstats
import re
import sys

from . import config
from .util import gc_collect
from ..util import has_compiled_ext


try:
    import cProfile
except ImportError:
    cProfile = None

_profile_stats = None
"""global ProfileStatsFileInstance.

plugin_base assigns this at the start of all tests.

"""


_current_test = None
"""String id of current test.

plugin_base assigns this at the start of each test using
_start_current_test.

"""


def _start_current_test(id_):
    global _current_test
    _current_test = id_

    if _profile_stats.force_write:
        _profile_stats.reset_count()


class ProfileStatsFile:
    """Store per-platform/fn profiling results in a file.

    There was no json module available when this was written, but now
    the file format which is very deterministically line oriented is kind of
    handy in any case for diffs and merges.

    """

    def __init__(self, filename, sort="cumulative", dump=None):
        self.force_write = (
            config.options is not None and config.options.force_write_profiles
        )
        self.write = self.force_write or (
            config.options is not None and config.options.write_profiles
        )
        self.fname = os.path.abspath(filename)
        self.short_fname = os.path.split(self.fname)[-1]
        self.data = collections.defaultdict(
            lambda: collections.defaultdict(dict)
        )
        self.dump = dump
        self.sort = sort
        self._read()
        if self.write:
            # rewrite for the case where features changed,
            # etc.
            self._write()

    @property
    def platform_key(self):
        dbapi_key = config.db.name + "_" + config.db.driver

        if config.db.name == "sqlite" and config.db.dialect._is_url_file_db(
            config.db.url
        ):
            dbapi_key += "_file"

        # keep it at 2.7, 3.1, 3.2, etc. for now.
        py_version = ".".join([str(v) for v in sys.version_info[0:2]])

        platform_tokens = [
            platform.machine(),
            platform.system().lower(),
            platform.python_implementation().lower(),
            py_version,
            dbapi_key,
        ]

        platform_tokens.append("dbapiunicode")
        _has_cext = has_compiled_ext()
        platform_tokens.append(_has_cext and "cextensions" or "nocextensions")
        return "_".join(platform_tokens)

    def has_stats(self):
        test_key = _current_test
        return (
            test_key in self.data and self.platform_key in self.data[test_key]
        )

    def result(self, callcount):
        test_key = _current_test
        per_fn = self.data[test_key]
        per_platform = per_fn[self.platform_key]

        if "counts" not in per_platform:
            per_platform["counts"] = counts = []
        else:
            counts = per_platform["counts"]

        if "current_count" not in per_platform:
            per_platform["current_count"] = current_count = 0
        else:
            current_count = per_platform["current_count"]

        has_count = len(counts) > current_count

        if not has_count:
            counts.append(callcount)
            if self.write:
                self._write()
            result = None
        else:
            result = per_platform["lineno"], counts[current_count]
        per_platform["current_count"] += 1
        return result

    def reset_count(self):
        test_key = _current_test
        # since self.data is a defaultdict, don't access a key
        # if we don't know it's there first.
        if test_key not in self.data:
            return
        per_fn = self.data[test_key]
        if self.platform_key not in per_fn:
            return
        per_platform = per_fn[self.platform_key]
        if "counts" in per_platform:
            per_platform["counts"][:] = []

    def replace(self, callcount):
        test_key = _current_test
        per_fn = self.data[test_key]
        per_platform = per_fn[self.platform_key]
        counts = per_platform["counts"]
        current_count = per_platform["current_count"]
        if current_count < len(counts):
            counts[current_count - 1] = callcount
        else:
            counts[-1] = callcount
        if self.write:
            self._write()

    def _header(self):
        return (
            "# %s\n"
            "# This file is written out on a per-environment basis.\n"
            "# For each test in aaa_profiling, the corresponding "
            "function and \n"
            "# environment is located within this file.  "
            "If it doesn't exist,\n"
            "# the test is skipped.\n"
            "# If a callcount does exist, it is compared "
            "to what we received. \n"
            "# assertions are raised if the counts do not match.\n"
            "# \n"
            "# To add a new callcount test, apply the function_call_count \n"
            "# decorator and re-run the tests using the --write-profiles \n"
            "# option - this file will be rewritten including the new count.\n"
            "# \n"
        ) % (self.fname)

    def _read(self):
        try:
            profile_f = open(self.fname)
        except OSError:
            return
        for lineno, line in enumerate(profile_f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            test_key, platform_key, counts = line.split()
            per_fn = self.data[test_key]
            per_platform = per_fn[platform_key]
            c = [int(count) for count in counts.split(",")]
            per_platform["counts"] = c
            per_platform["lineno"] = lineno + 1
            per_platform["current_count"] = 0
        profile_f.close()

    def _write(self):
        print("Writing profile file %s" % self.fname)
        profile_f = open(self.fname, "w")
        profile_f.write(self._header())
        for test_key in sorted(self.data):
            per_fn = self.data[test_key]
            profile_f.write("\n# TEST: %s\n\n" % test_key)
            for platform_key in sorted(per_fn):
                per_platform = per_fn[platform_key]
                c = ",".join(str(count) for count in per_platform["counts"])
                profile_f.write("%s %s %s\n" % (test_key, platform_key, c))
        profile_f.close()


def function_call_count(variance=0.05, times=1, warmup=0):
    """Assert a target for a test case's function call count.

    The main purpose of this assertion is to detect changes in
    callcounts for various functions - the actual number is not as important.
    Callcounts are stored in a file keyed to Python version and OS platform
    information.  This file is generated automatically for new tests,
    and versioned so that unexpected changes in callcounts will be detected.

    """

    # use signature-rewriting decorator function so that pytest fixtures
    # still work on py27.  In Py3, update_wrapper() alone is good enough,
    # likely due to the introduction of __signature__.

    from sqlalchemy.util import decorator

    @decorator
    def wrap(fn, *args, **kw):
        for warm in range(warmup):
            fn(*args, **kw)

        timerange = range(times)
        with count_functions(variance=variance):
            for time in timerange:
                rv = fn(*args, **kw)
            return rv

    return wrap


@contextlib.contextmanager
def count_functions(variance=0.05):
    if cProfile is None:
        raise config._skip_test_exception("cProfile is not installed")

    if not _profile_stats.has_stats() and not _profile_stats.write:
        config.skip_test(
            "No profiling stats available on this "
            "platform for this function.  Run tests with "
            "--write-profiles to add statistics to %s for "
            "this platform." % _profile_stats.short_fname
        )

    gc_collect()

    pr = cProfile.Profile()
    pr.enable()
    # began = time.time()
    yield
    # ended = time.time()
    pr.disable()

    # s = StringIO()
    stats = pstats.Stats(pr, stream=sys.stdout)

    # timespent = ended - began
    callcount = stats.total_calls

    expected = _profile_stats.result(callcount)

    if expected is None:
        expected_count = None
    else:
        line_no, expected_count = expected

    print("Pstats calls: %d Expected %s" % (callcount, expected_count))
    stats.sort_stats(*re.split(r"[, ]", _profile_stats.sort))
    stats.print_stats()
    if _profile_stats.dump:
        base, ext = os.path.splitext(_profile_stats.dump)
        test_name = _current_test.split(".")[-1]
        dumpfile = "%s_%s%s" % (base, test_name, ext or ".profile")
        stats.dump_stats(dumpfile)
        print("Dumped stats to file %s" % dumpfile)
    # stats.print_callers()
    if _profile_stats.force_write:
        _profile_stats.replace(callcount)
    elif expected_count:
        deviance = int(callcount * variance)
        failed = abs(callcount - expected_count) > deviance

        if failed:
            if _profile_stats.write:
                _profile_stats.replace(callcount)
            else:
                raise AssertionError(
                    "Adjusted function call count %s not within %s%% "
                    "of expected %s, platform %s. Rerun with "
                    "--write-profiles to "
                    "regenerate this callcount."
                    % (
                        callcount,
                        (variance * 100),
                        expected_count,
                        _profile_stats.platform_key,
                    )
                )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\exceptions.py ===
from __future__ import absolute_import

from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead

# Base Exceptions


class HTTPError(Exception):
    """Base exception used by this module."""

    pass


class HTTPWarning(Warning):
    """Base warning used by this module."""

    pass


class PoolError(HTTPError):
    """Base exception for errors caused within a pool."""

    def __init__(self, pool, message):
        self.pool = pool
        HTTPError.__init__(self, "%s: %s" % (pool, message))

    def __reduce__(self):
        # For pickling purposes.
        return self.__class__, (None, None)


class RequestError(PoolError):
    """Base exception for PoolErrors that have associated URLs."""

    def __init__(self, pool, url, message):
        self.url = url
        PoolError.__init__(self, pool, message)

    def __reduce__(self):
        # For pickling purposes.
        return self.__class__, (None, self.url, None)


class SSLError(HTTPError):
    """Raised when SSL certificate fails in an HTTPS connection."""

    pass


class ProxyError(HTTPError):
    """Raised when the connection to a proxy fails."""

    def __init__(self, message, error, *args):
        super(ProxyError, self).__init__(message, error, *args)
        self.original_error = error


class DecodeError(HTTPError):
    """Raised when automatic decoding based on Content-Type fails."""

    pass


class ProtocolError(HTTPError):
    """Raised when something unexpected happens mid-request/response."""

    pass


#: Renamed to ProtocolError but aliased for backwards compatibility.
ConnectionError = ProtocolError


# Leaf Exceptions


class MaxRetryError(RequestError):
    """Raised when the maximum number of retries is exceeded.

    :param pool: The connection pool
    :type pool: :class:`~urllib3.connectionpool.HTTPConnectionPool`
    :param string url: The requested Url
    :param exceptions.Exception reason: The underlying error

    """

    def __init__(self, pool, url, reason=None):
        self.reason = reason

        message = "Max retries exceeded with url: %s (Caused by %r)" % (url, reason)

        RequestError.__init__(self, pool, url, message)


class HostChangedError(RequestError):
    """Raised when an existing pool gets a request for a foreign host."""

    def __init__(self, pool, url, retries=3):
        message = "Tried to open a foreign host with url: %s" % url
        RequestError.__init__(self, pool, url, message)
        self.retries = retries


class TimeoutStateError(HTTPError):
    """Raised when passing an invalid state to a timeout"""

    pass


class TimeoutError(HTTPError):
    """Raised when a socket timeout error occurs.

    Catching this error will catch both :exc:`ReadTimeoutErrors
    <ReadTimeoutError>` and :exc:`ConnectTimeoutErrors <ConnectTimeoutError>`.
    """

    pass


class ReadTimeoutError(TimeoutError, RequestError):
    """Raised when a socket timeout occurs while receiving data from a server"""

    pass


# This timeout error does not have a URL attached and needs to inherit from the
# base HTTPError
class ConnectTimeoutError(TimeoutError):
    """Raised when a socket timeout occurs while connecting to a server"""

    pass


class NewConnectionError(ConnectTimeoutError, PoolError):
    """Raised when we fail to establish a new connection. Usually ECONNREFUSED."""

    pass


class EmptyPoolError(PoolError):
    """Raised when a pool runs out of connections and no more are allowed."""

    pass


class ClosedPoolError(PoolError):
    """Raised when a request enters a pool after the pool has been closed."""

    pass


class LocationValueError(ValueError, HTTPError):
    """Raised when there is something wrong with a given URL input."""

    pass


class LocationParseError(LocationValueError):
    """Raised when get_host or similar fails to parse the URL input."""

    def __init__(self, location):
        message = "Failed to parse: %s" % location
        HTTPError.__init__(self, message)

        self.location = location


class URLSchemeUnknown(LocationValueError):
    """Raised when a URL input has an unsupported scheme."""

    def __init__(self, scheme):
        message = "Not supported URL scheme %s" % scheme
        super(URLSchemeUnknown, self).__init__(message)

        self.scheme = scheme


class ResponseError(HTTPError):
    """Used as a container for an error reason supplied in a MaxRetryError."""

    GENERIC_ERROR = "too many error responses"
    SPECIFIC_ERROR = "too many {status_code} error responses"


class SecurityWarning(HTTPWarning):
    """Warned when performing security reducing actions"""

    pass


class SubjectAltNameWarning(SecurityWarning):
    """Warned when connecting to a host with a certificate missing a SAN."""

    pass


class InsecureRequestWarning(SecurityWarning):
    """Warned when making an unverified HTTPS request."""

    pass


class SystemTimeWarning(SecurityWarning):
    """Warned when system time is suspected to be wrong"""

    pass


class InsecurePlatformWarning(SecurityWarning):
    """Warned when certain TLS/SSL configuration is not available on a platform."""

    pass


class SNIMissingWarning(HTTPWarning):
    """Warned when making a HTTPS request without SNI available."""

    pass


class DependencyWarning(HTTPWarning):
    """
    Warned when an attempt is made to import a module with missing optional
    dependencies.
    """

    pass


class ResponseNotChunked(ProtocolError, ValueError):
    """Response needs to be chunked in order to read it as chunks."""

    pass


class BodyNotHttplibCompatible(HTTPError):
    """
    Body should be :class:`http.client.HTTPResponse` like
    (have an fp attribute which returns raw chunks) for read_chunked().
    """

    pass


class IncompleteRead(HTTPError, httplib_IncompleteRead):
    """
    Response length doesn't match expected Content-Length

    Subclass of :class:`http.client.IncompleteRead` to allow int value
    for ``partial`` to avoid creating large objects on streamed reads.
    """

    def __init__(self, partial, expected):
        super(IncompleteRead, self).__init__(partial, expected)

    def __repr__(self):
        return "IncompleteRead(%i bytes read, %i more expected)" % (
            self.partial,
            self.expected,
        )


class InvalidChunkLength(HTTPError, httplib_IncompleteRead):
    """Invalid chunk length in a chunked response."""

    def __init__(self, response, length):
        super(InvalidChunkLength, self).__init__(
            response.tell(), response.length_remaining
        )
        self.response = response
        self.length = length

    def __repr__(self):
        return "InvalidChunkLength(got length %r, %i bytes read)" % (
            self.length,
            self.partial,
        )


class InvalidHeader(HTTPError):
    """The header provided was somehow invalid."""

    pass


class ProxySchemeUnknown(AssertionError, URLSchemeUnknown):
    """ProxyManager does not support the supplied scheme"""

    # TODO(t-8ch): Stop inheriting from AssertionError in v2.0.

    def __init__(self, scheme):
        # 'localhost' is here because our URL parser parses
        # localhost:8080 -> scheme=localhost, remove if we fix this.
        if scheme == "localhost":
            scheme = None
        if scheme is None:
            message = "Proxy URL had no scheme, should start with http:// or https://"
        else:
            message = (
                "Proxy URL had unsupported scheme %s, should use http:// or https://"
                % scheme
            )
        super(ProxySchemeUnknown, self).__init__(message)


class ProxySchemeUnsupported(ValueError):
    """Fetching HTTPS resources through HTTPS proxies is unsupported"""

    pass


class HeaderParsingError(HTTPError):
    """Raised by assert_header_parsing, but we convert it to a log.warning statement."""

    def __init__(self, defects, unparsed_data):
        message = "%s, unparsed data: %r" % (defects or "Unknown", unparsed_data)
        super(HeaderParsingError, self).__init__(message)


class UnrewindableBodyError(HTTPError):
    """urllib3 encountered an error when trying to rewind a body"""

    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\roles.py ===
# sql/roles.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

from typing import Any
from typing import Generic
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar

from .. import util
from ..util.typing import Literal

if TYPE_CHECKING:
    from ._typing import _PropagateAttrsType
    from .elements import Label
    from .selectable import _SelectIterable
    from .selectable import FromClause
    from .selectable import Subquery

_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)


class SQLRole:
    """Define a "role" within a SQL statement structure.

    Classes within SQL Core participate within SQLRole hierarchies in order
    to more accurately indicate where they may be used within SQL statements
    of all types.

    .. versionadded:: 1.4

    """

    __slots__ = ()
    allows_lambda = False
    uses_inspection = False


class UsesInspection:
    __slots__ = ()
    _post_inspect: Literal[None] = None
    uses_inspection = True


class AllowsLambdaRole:
    __slots__ = ()
    allows_lambda = True


class HasCacheKeyRole(SQLRole):
    __slots__ = ()
    _role_name = "Cacheable Core or ORM object"


class ExecutableOptionRole(SQLRole):
    __slots__ = ()
    _role_name = "ExecutionOption Core or ORM object"


class LiteralValueRole(SQLRole):
    __slots__ = ()
    _role_name = "Literal Python value"


class ColumnArgumentRole(SQLRole):
    __slots__ = ()
    _role_name = "Column expression"


class ColumnArgumentOrKeyRole(ColumnArgumentRole):
    __slots__ = ()
    _role_name = "Column expression or string key"


class StrAsPlainColumnRole(ColumnArgumentRole):
    __slots__ = ()
    _role_name = "Column expression or string key"


class ColumnListRole(SQLRole):
    """Elements suitable for forming comma separated lists of expressions."""

    __slots__ = ()


class StringRole(SQLRole):
    """mixin indicating a role that results in strings"""

    __slots__ = ()


class TruncatedLabelRole(StringRole, SQLRole):
    __slots__ = ()
    _role_name = "String SQL identifier"


class ColumnsClauseRole(AllowsLambdaRole, UsesInspection, ColumnListRole):
    __slots__ = ()
    _role_name = (
        "Column expression, FROM clause, or other columns clause element"
    )

    @property
    def _select_iterable(self) -> _SelectIterable:
        raise NotImplementedError()


class TypedColumnsClauseRole(Generic[_T_co], SQLRole):
    """element-typed form of ColumnsClauseRole"""

    __slots__ = ()


class LimitOffsetRole(SQLRole):
    __slots__ = ()
    _role_name = "LIMIT / OFFSET expression"


class ByOfRole(ColumnListRole):
    __slots__ = ()
    _role_name = "GROUP BY / OF / etc. expression"


class GroupByRole(AllowsLambdaRole, UsesInspection, ByOfRole):
    __slots__ = ()
    # note there's a special case right now where you can pass a whole
    # ORM entity to group_by() and it splits out.   we may not want to keep
    # this around

    _role_name = "GROUP BY expression"


class OrderByRole(AllowsLambdaRole, ByOfRole):
    __slots__ = ()
    _role_name = "ORDER BY expression"


class StructuralRole(SQLRole):
    __slots__ = ()


class StatementOptionRole(StructuralRole):
    __slots__ = ()
    _role_name = "statement sub-expression element"


class OnClauseRole(AllowsLambdaRole, StructuralRole):
    __slots__ = ()
    _role_name = (
        "ON clause, typically a SQL expression or "
        "ORM relationship attribute"
    )


class WhereHavingRole(OnClauseRole):
    __slots__ = ()
    _role_name = "SQL expression for WHERE/HAVING role"


class ExpressionElementRole(TypedColumnsClauseRole[_T_co]):
    # note when using generics for ExpressionElementRole,
    # the generic type needs to be in
    # sqlalchemy.sql.coercions._impl_lookup mapping also.
    # these are set up for basic types like int, bool, str, float
    # right now

    __slots__ = ()
    _role_name = "SQL expression element"

    def label(self, name: Optional[str]) -> Label[_T]:
        raise NotImplementedError()


class ConstExprRole(ExpressionElementRole[_T]):
    __slots__ = ()
    _role_name = "Constant True/False/None expression"


class LabeledColumnExprRole(ExpressionElementRole[_T]):
    __slots__ = ()


class BinaryElementRole(ExpressionElementRole[_T]):
    __slots__ = ()
    _role_name = "SQL expression element or literal value"


class InElementRole(SQLRole):
    __slots__ = ()
    _role_name = (
        "IN expression list, SELECT construct, or bound parameter object"
    )


class JoinTargetRole(AllowsLambdaRole, UsesInspection, StructuralRole):
    __slots__ = ()
    _role_name = (
        "Join target, typically a FROM expression, or ORM "
        "relationship attribute"
    )


class FromClauseRole(ColumnsClauseRole, JoinTargetRole):
    __slots__ = ()
    _role_name = "FROM expression, such as a Table or alias() object"

    _is_subquery = False

    named_with_column: bool


class StrictFromClauseRole(FromClauseRole):
    __slots__ = ()
    # does not allow text() or select() objects


class AnonymizedFromClauseRole(StrictFromClauseRole):
    __slots__ = ()

    if TYPE_CHECKING:

        def _anonymous_fromclause(
            self, *, name: Optional[str] = None, flat: bool = False
        ) -> FromClause: ...


class ReturnsRowsRole(SQLRole):
    __slots__ = ()
    _role_name = (
        "Row returning expression such as a SELECT, a FROM clause, or an "
        "INSERT/UPDATE/DELETE with RETURNING"
    )


class StatementRole(SQLRole):
    __slots__ = ()
    _role_name = "Executable SQL or text() construct"

    if TYPE_CHECKING:

        @util.memoized_property
        def _propagate_attrs(self) -> _PropagateAttrsType: ...

    else:
        _propagate_attrs = util.EMPTY_DICT


class SelectStatementRole(StatementRole, ReturnsRowsRole):
    __slots__ = ()
    _role_name = "SELECT construct or equivalent text() construct"

    def subquery(self) -> Subquery:
        raise NotImplementedError(
            "All SelectStatementRole objects should implement a "
            ".subquery() method."
        )


class HasCTERole(ReturnsRowsRole):
    __slots__ = ()


class IsCTERole(SQLRole):
    __slots__ = ()
    _role_name = "CTE object"


class CompoundElementRole(AllowsLambdaRole, SQLRole):
    """SELECT statements inside a CompoundSelect, e.g. UNION, EXTRACT, etc."""

    __slots__ = ()
    _role_name = (
        "SELECT construct for inclusion in a UNION or other set construct"
    )


# TODO: are we using this?
class DMLRole(StatementRole):
    __slots__ = ()


class DMLTableRole(FromClauseRole):
    __slots__ = ()
    _role_name = "subject table for an INSERT, UPDATE or DELETE"


class DMLColumnRole(SQLRole):
    __slots__ = ()
    _role_name = "SET/VALUES column expression or string key"


class DMLSelectRole(SQLRole):
    """A SELECT statement embedded in DML, typically INSERT from SELECT"""

    __slots__ = ()
    _role_name = "SELECT statement or equivalent textual object"


class DDLRole(StatementRole):
    __slots__ = ()


class DDLExpressionRole(StructuralRole):
    __slots__ = ()
    _role_name = "SQL expression element for DDL constraint"


class DDLConstraintColumnRole(SQLRole):
    __slots__ = ()
    _role_name = "String column name or column expression for DDL constraint"


class DDLReferredColumnRole(DDLConstraintColumnRole):
    __slots__ = ()
    _role_name = (
        "String column name or Column object for DDL foreign key constraint"
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\utils.py ===
from sympy.core.containers import Tuple
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanFunction
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import FiniteSet
from sympy.tensor.indexed import Indexed


def _get_free_symbols(exprs):
    """Returns the free symbols of a symbolic expression.

    If the expression contains any of these elements, assume that they are
    the "free symbols" of the expression:

    * indexed objects
    * applied undefined function (useful for sympy.physics.mechanics module)
    """
    if not isinstance(exprs, (list, tuple, set)):
        exprs = [exprs]
    if all(callable(e) for e in exprs):
        return set()

    free = set().union(*[e.atoms(Indexed) for e in exprs])
    free = free.union(*[e.atoms(AppliedUndef) for e in exprs])
    return free or set().union(*[e.free_symbols for e in exprs])


def extract_solution(set_sol, n=10):
    """Extract numerical solutions from a set solution (computed by solveset,
    linsolve, nonlinsolve). Often, it is not trivial do get something useful
    out of them.

    Parameters
    ==========

    n : int, optional
        In order to replace ImageSet with FiniteSet, an iterator is created
        for each ImageSet contained in `set_sol`, starting from 0 up to `n`.
        Default value: 10.
    """
    images = set_sol.find(ImageSet)
    for im in images:
        it = iter(im)
        s = FiniteSet(*[next(it) for n in range(0, n)])
        set_sol = set_sol.subs(im, s)
    return set_sol


def _plot_sympify(args):
    """This function recursively loop over the arguments passed to the plot
    functions: the sympify function will be applied to all arguments except
    those of type string/dict.

    Generally, users can provide the following arguments to a plot function:

    expr, range1 [tuple, opt], ..., label [str, opt], rendering_kw [dict, opt]

    `expr, range1, ...` can be sympified, whereas `label, rendering_kw` can't.
    In particular, whenever a special character like $, {, }, ... is used in
    the `label`, sympify will raise an error.
    """
    if isinstance(args, Expr):
        return args

    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, (list, tuple)):
            args[i] = Tuple(*_plot_sympify(a), sympify=False)
        elif not (isinstance(a, (str, dict)) or callable(a)
            # NOTE: check if it is a vector from sympy.physics.vector module
            # without importing the module (because it slows down SymPy's
            # import process and triggers SymPy's optional-dependencies
            # tests to fail).
            or ((a.__class__.__name__ == "Vector") and not isinstance(a, Basic))
        ):
            args[i] = sympify(a)
    return args


def _create_ranges(exprs, ranges, npar, label="", params=None):
    """This function does two things:

    1. Check if the number of free symbols is in agreement with the type of
       plot chosen. For example, plot() requires 1 free symbol;
       plot3d() requires 2 free symbols.
    2. Sometime users create plots without providing ranges for the variables.
       Here we create the necessary ranges.

    Parameters
    ==========

    exprs : iterable
        The expressions from which to extract the free symbols
    ranges : iterable
        The limiting ranges provided by the user
    npar : int
        The number of free symbols required by the plot functions.
        For example,
        npar=1 for plot, npar=2 for plot3d, ...
    params : dict
        A dictionary mapping symbols to parameters for interactive plot.
    """
    get_default_range = lambda symbol: Tuple(symbol, -10, 10)

    free_symbols = _get_free_symbols(exprs)
    if params is not None:
        free_symbols = free_symbols.difference(params.keys())

    if len(free_symbols) > npar:
        raise ValueError(
            "Too many free symbols.\n"
            + "Expected {} free symbols.\n".format(npar)
            + "Received {}: {}".format(len(free_symbols), free_symbols)
        )

    if len(ranges) > npar:
        raise ValueError(
            "Too many ranges. Received %s, expected %s" % (len(ranges), npar))

    # free symbols in the ranges provided by the user
    rfs = set().union([r[0] for r in ranges])
    if len(rfs) != len(ranges):
        raise ValueError("Multiple ranges with the same symbol")

    if len(ranges) < npar:
        symbols = free_symbols.difference(rfs)
        if symbols != set():
            # add a range for each missing free symbols
            for s in symbols:
                ranges.append(get_default_range(s))
        # if there is still room, fill them with dummys
        for i in range(npar - len(ranges)):
            ranges.append(get_default_range(Dummy()))

    if len(free_symbols) == npar:
        # there could be times when this condition is not met, for example
        # plotting the function f(x, y) = x (which is a plane); in this case,
        # free_symbols = {x} whereas rfs = {x, y} (or x and Dummy)
        rfs = set().union([r[0] for r in ranges])
        if len(free_symbols.difference(rfs)) > 0:
            raise ValueError(
                "Incompatible free symbols of the expressions with "
                "the ranges.\n"
                + "Free symbols in the expressions: {}\n".format(free_symbols)
                + "Free symbols in the ranges: {}".format(rfs)
            )
    return ranges


def _is_range(r):
    """A range is defined as (symbol, start, end). start and end should
    be numbers.
    """
    # TODO: prange check goes here
    return (
        isinstance(r, Tuple)
        and (len(r) == 3)
        and (not isinstance(r.args[1], str)) and r.args[1].is_number
        and (not isinstance(r.args[2], str)) and r.args[2].is_number
    )


def _unpack_args(*args):
    """Given a list/tuple of arguments previously processed by _plot_sympify()
    and/or _check_arguments(), separates and returns its components:
    expressions, ranges, label and rendering keywords.

    Examples
    ========

    >>> from sympy import cos, sin, symbols
    >>> from sympy.plotting.utils import _plot_sympify, _unpack_args
    >>> x, y = symbols('x, y')
    >>> args = (sin(x), (x, -10, 10), "f1")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x)], [(x, -10, 10)], 'f1', None)

    >>> args = (sin(x**2 + y**2), (x, -2, 2), (y, -3, 3), "f2")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x**2 + y**2)], [(x, -2, 2), (y, -3, 3)], 'f2', None)

    >>> args = (sin(x + y), cos(x - y), x + y, (x, -2, 2), (y, -3, 3), "f3")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x + y), cos(x - y), x + y], [(x, -2, 2), (y, -3, 3)], 'f3', None)
    """
    ranges = [t for t in args if _is_range(t)]
    labels = [t for t in args if isinstance(t, str)]
    label = None if not labels else labels[0]
    rendering_kw = [t for t in args if isinstance(t, dict)]
    rendering_kw = None if not rendering_kw else rendering_kw[0]
    # NOTE: why None? because args might have been preprocessed by
    # _check_arguments, so None might represent the rendering_kw
    results = [not (_is_range(a) or isinstance(a, (str, dict)) or (a is None)) for a in args]
    exprs = [a for a, b in zip(args, results) if b]
    return exprs, ranges, label, rendering_kw


def _check_arguments(args, nexpr, npar, **kwargs):
    """Checks the arguments and converts into tuples of the
    form (exprs, ranges, label, rendering_kw).

    Parameters
    ==========

    args
        The arguments provided to the plot functions
    nexpr
        The number of sub-expression forming an expression to be plotted.
        For example:
        nexpr=1 for plot.
        nexpr=2 for plot_parametric: a curve is represented by a tuple of two
            elements.
        nexpr=1 for plot3d.
        nexpr=3 for plot3d_parametric_line: a curve is represented by a tuple
            of three elements.
    npar
        The number of free symbols required by the plot functions. For example,
        npar=1 for plot, npar=2 for plot3d, ...
    **kwargs :
        keyword arguments passed to the plotting function. It will be used to
        verify if ``params`` has ben provided.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import _check_arguments
       >>> x = symbols('x')
       >>> _check_arguments([cos(x), sin(x)], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), None, None)]

       >>> _check_arguments([cos(x), sin(x), "test"], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), 'test', None)]

       >>> _check_arguments([cos(x), sin(x), "test", {"a": 0, "b": 1}], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), 'test', {'a': 0, 'b': 1})]

       >>> _check_arguments([x, x**2], 1, 1)
       [(x, (x, -10, 10), None, None), (x**2, (x, -10, 10), None, None)]
    """
    if not args:
        return []
    output = []
    params = kwargs.get("params", None)

    if all(isinstance(a, (Expr, Relational, BooleanFunction)) for a in args[:nexpr]):
        # In this case, with a single plot command, we are plotting either:
        #   1. one expression
        #   2. multiple expressions over the same range

        exprs, ranges, label, rendering_kw = _unpack_args(*args)
        free_symbols = set().union(*[e.free_symbols for e in exprs])
        ranges = _create_ranges(exprs, ranges, npar, label, params)

        if nexpr > 1:
            # in case of plot_parametric or plot3d_parametric_line, there will
            # be 2 or 3 expressions defining a curve. Group them together.
            if len(exprs) == nexpr:
                exprs = (tuple(exprs),)
        for expr in exprs:
            # need this if-else to deal with both plot/plot3d and
            # plot_parametric/plot3d_parametric_line
            is_expr = isinstance(expr, (Expr, Relational, BooleanFunction))
            e = (expr,) if is_expr else expr
            output.append((*e, *ranges, label, rendering_kw))

    else:
        # In this case, we are plotting multiple expressions, each one with its
        # range. Each "expression" to be plotted has the following form:
        # (expr, range, label) where label is optional

        _, ranges, labels, rendering_kw = _unpack_args(*args)
        labels = [labels] if labels else []

        # number of expressions
        n = (len(ranges) + len(labels) +
            (len(rendering_kw) if rendering_kw is not None else 0))
        new_args = args[:-n] if n > 0 else args

        # at this point, new_args might just be [expr]. But I need it to be
        # [[expr]] in order to be able to loop over
        # [expr, range [opt], label [opt]]
        if not isinstance(new_args[0], (list, tuple, Tuple)):
            new_args = [new_args]

        # Each arg has the form (expr1, expr2, ..., range1 [optional], ...,
        #   label [optional], rendering_kw [optional])
        for arg in new_args:
            # look for "local" range and label. If there is not, use "global".
            l = [a for a in arg if isinstance(a, str)]
            if not l:
                l = labels
            r = [a for a in arg if _is_range(a)]
            if not r:
                r = ranges.copy()
            rend_kw = [a for a in arg if isinstance(a, dict)]
            rend_kw = rendering_kw if len(rend_kw) == 0 else rend_kw[0]

            # NOTE: arg = arg[:nexpr] may raise an exception if lambda
            # functions are used. Execute the following instead:
            arg = [arg[i] for i in range(nexpr)]
            free_symbols = set()
            if all(not callable(a) for a in arg):
                free_symbols = free_symbols.union(*[a.free_symbols for a in arg])
            if len(r) != npar:
                r = _create_ranges(arg, r, npar, "", params)

            label = None if not l else l[0]
            output.append((*arg, *r, label, rend_kw))
    return output

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\sansio\multipart.py ===
from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass
from enum import auto
from enum import Enum

from ..datastructures import Headers
from ..exceptions import RequestEntityTooLarge
from ..http import parse_options_header


class Event:
    pass


@dataclass(frozen=True)
class Preamble(Event):
    data: bytes


@dataclass(frozen=True)
class Field(Event):
    name: str
    headers: Headers


@dataclass(frozen=True)
class File(Event):
    name: str
    filename: str
    headers: Headers


@dataclass(frozen=True)
class Data(Event):
    data: bytes
    more_data: bool


@dataclass(frozen=True)
class Epilogue(Event):
    data: bytes


class NeedData(Event):
    pass


NEED_DATA = NeedData()


class State(Enum):
    PREAMBLE = auto()
    PART = auto()
    DATA = auto()
    DATA_START = auto()
    EPILOGUE = auto()
    COMPLETE = auto()


# Multipart line breaks MUST be CRLF (\r\n) by RFC-7578, except that
# many implementations break this and either use CR or LF alone.
LINE_BREAK = b"(?:\r\n|\n|\r)"
BLANK_LINE_RE = re.compile(b"(?:\r\n\r\n|\r\r|\n\n)", re.MULTILINE)
LINE_BREAK_RE = re.compile(LINE_BREAK, re.MULTILINE)
# Header values can be continued via a space or tab after the linebreak, as
# per RFC2231
HEADER_CONTINUATION_RE = re.compile(b"%s[ \t]" % LINE_BREAK, re.MULTILINE)
# This must be long enough to contain any line breaks plus any
# additional boundary markers (--) such that they will be found in a
# subsequent search
SEARCH_EXTRA_LENGTH = 8


class MultipartDecoder:
    """Decodes a multipart message as bytes into Python events.

    The part data is returned as available to allow the caller to save
    the data from memory to disk, if desired.
    """

    def __init__(
        self,
        boundary: bytes,
        max_form_memory_size: int | None = None,
        *,
        max_parts: int | None = None,
    ) -> None:
        self.buffer = bytearray()
        self.complete = False
        self.max_form_memory_size = max_form_memory_size
        self.max_parts = max_parts
        self.state = State.PREAMBLE
        self.boundary = boundary

        # Note in the below \h i.e. horizontal whitespace is used
        # as [^\S\n\r] as \h isn't supported in python.

        # The preamble must end with a boundary where the boundary is
        # prefixed by a line break, RFC2046. Except that many
        # implementations including Werkzeug's tests omit the line
        # break prefix. In addition the first boundary could be the
        # epilogue boundary (for empty form-data) hence the matching
        # group to understand if it is an epilogue boundary.
        self.preamble_re = re.compile(
            rb"%s?--%s(--[^\S\n\r]*%s?|[^\S\n\r]*%s)"
            % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK),
            re.MULTILINE,
        )
        # A boundary must include a line break prefix and suffix, and
        # may include trailing whitespace. In addition the boundary
        # could be the epilogue boundary hence the matching group to
        # understand if it is an epilogue boundary.
        self.boundary_re = re.compile(
            rb"%s--%s(--[^\S\n\r]*%s?|[^\S\n\r]*%s)"
            % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK),
            re.MULTILINE,
        )
        self._search_position = 0
        self._parts_decoded = 0

    def last_newline(self, data: bytes) -> int:
        try:
            last_nl = data.rindex(b"\n")
        except ValueError:
            last_nl = len(data)
        try:
            last_cr = data.rindex(b"\r")
        except ValueError:
            last_cr = len(data)

        return min(last_nl, last_cr)

    def receive_data(self, data: bytes | None) -> None:
        if data is None:
            self.complete = True
        elif (
            self.max_form_memory_size is not None
            and len(self.buffer) + len(data) > self.max_form_memory_size
        ):
            # Ensure that data within single event does not exceed limit.
            # Also checked across accumulated events in MultiPartParser.
            raise RequestEntityTooLarge()
        else:
            self.buffer.extend(data)

    def next_event(self) -> Event:
        event: Event = NEED_DATA

        if self.state == State.PREAMBLE:
            match = self.preamble_re.search(self.buffer, self._search_position)
            if match is not None:
                if match.group(1).startswith(b"--"):
                    self.state = State.EPILOGUE
                else:
                    self.state = State.PART
                data = bytes(self.buffer[: match.start()])
                del self.buffer[: match.end()]
                event = Preamble(data=data)
                self._search_position = 0
            else:
                # Update the search start position to be equal to the
                # current buffer length (already searched) minus a
                # safe buffer for part of the search target.
                self._search_position = max(
                    0, len(self.buffer) - len(self.boundary) - SEARCH_EXTRA_LENGTH
                )

        elif self.state == State.PART:
            match = BLANK_LINE_RE.search(self.buffer, self._search_position)
            if match is not None:
                headers = self._parse_headers(self.buffer[: match.start()])
                # The final header ends with a single CRLF, however a
                # blank line indicates the start of the
                # body. Therefore the end is after the first CRLF.
                headers_end = (match.start() + match.end()) // 2
                del self.buffer[:headers_end]

                if "content-disposition" not in headers:
                    raise ValueError("Missing Content-Disposition header")

                disposition, extra = parse_options_header(
                    headers["content-disposition"]
                )
                name = t.cast(str, extra.get("name"))
                filename = extra.get("filename")
                if filename is not None:
                    event = File(
                        filename=filename,
                        headers=headers,
                        name=name,
                    )
                else:
                    event = Field(
                        headers=headers,
                        name=name,
                    )
                self.state = State.DATA_START
                self._search_position = 0
                self._parts_decoded += 1

                if self.max_parts is not None and self._parts_decoded > self.max_parts:
                    raise RequestEntityTooLarge()
            else:
                # Update the search start position to be equal to the
                # current buffer length (already searched) minus a
                # safe buffer for part of the search target.
                self._search_position = max(0, len(self.buffer) - SEARCH_EXTRA_LENGTH)

        elif self.state == State.DATA_START:
            data, del_index, more_data = self._parse_data(self.buffer, start=True)
            del self.buffer[:del_index]
            event = Data(data=data, more_data=more_data)
            if more_data:
                self.state = State.DATA

        elif self.state == State.DATA:
            data, del_index, more_data = self._parse_data(self.buffer, start=False)
            del self.buffer[:del_index]
            if data or not more_data:
                event = Data(data=data, more_data=more_data)

        elif self.state == State.EPILOGUE and self.complete:
            event = Epilogue(data=bytes(self.buffer))
            del self.buffer[:]
            self.state = State.COMPLETE

        if self.complete and isinstance(event, NeedData):
            raise ValueError(f"Invalid form-data cannot parse beyond {self.state}")

        return event

    def _parse_headers(self, data: bytes) -> Headers:
        headers: list[tuple[str, str]] = []
        # Merge the continued headers into one line
        data = HEADER_CONTINUATION_RE.sub(b" ", data)
        # Now there is one header per line
        for line in data.splitlines():
            line = line.strip()

            if line != b"":
                name, _, value = line.decode().partition(":")
                headers.append((name.strip(), value.strip()))
        return Headers(headers)

    def _parse_data(self, data: bytes, *, start: bool) -> tuple[bytes, int, bool]:
        # Body parts must start with CRLF (or CR or LF)
        if start:
            match = LINE_BREAK_RE.match(data)
            data_start = t.cast(t.Match[bytes], match).end()
        else:
            data_start = 0

        boundary = b"--" + self.boundary

        if self.buffer.find(boundary) == -1:
            # No complete boundary in the buffer, but there may be
            # a partial boundary at the end. As the boundary
            # starts with either a nl or cr find the earliest and
            # return up to that as data.
            data_end = del_index = self.last_newline(data[data_start:]) + data_start
            # If amount of data after last newline is far from
            # possible length of partial boundary, we should
            # assume that there is no partial boundary in the buffer
            # and return all pending data.
            if (len(data) - data_end) > len(b"\n" + boundary):
                data_end = del_index = len(data)
            more_data = True
        else:
            match = self.boundary_re.search(data)
            if match is not None:
                if match.group(1).startswith(b"--"):
                    self.state = State.EPILOGUE
                else:
                    self.state = State.PART
                data_end = match.start()
                del_index = match.end()
            else:
                data_end = del_index = self.last_newline(data[data_start:]) + data_start
            more_data = match is None

        return bytes(data[data_start:data_end]), del_index, more_data


class MultipartEncoder:
    def __init__(self, boundary: bytes) -> None:
        self.boundary = boundary
        self.state = State.PREAMBLE

    def send_event(self, event: Event) -> bytes:
        if isinstance(event, Preamble) and self.state == State.PREAMBLE:
            self.state = State.PART
            return event.data
        elif isinstance(event, (Field, File)) and self.state in {
            State.PREAMBLE,
            State.PART,
            State.DATA,
        }:
            data = b"\r\n--" + self.boundary + b"\r\n"
            data += b'Content-Disposition: form-data; name="%s"' % event.name.encode()
            if isinstance(event, File):
                data += b'; filename="%s"' % event.filename.encode()
            data += b"\r\n"
            for name, value in t.cast(Field, event).headers:
                if name.lower() != "content-disposition":
                    data += f"{name}: {value}\r\n".encode()
            self.state = State.DATA_START
            return data
        elif isinstance(event, Data) and self.state == State.DATA_START:
            self.state = State.DATA
            if len(event.data) > 0:
                return b"\r\n" + event.data
            else:
                return event.data
        elif isinstance(event, Data) and self.state == State.DATA:
            return event.data
        elif isinstance(event, Epilogue):
            self.state = State.COMPLETE
            return b"\r\n--" + self.boundary + b"--\r\n" + event.data
        else:
            raise ValueError(f"Cannot generate {event} in state: {self.state}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\queue.py ===
# util/queue.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""An adaptation of Py2.3/2.4's Queue module which supports reentrant
behavior, using RLock instead of Lock for its mutex object.  The
Queue object is used exclusively by the sqlalchemy.pool.QueuePool
class.

This is to support the connection pool's usage of weakref callbacks to return
connections to the underlying Queue, which can in extremely
rare cases be invoked within the ``get()`` method of the Queue itself,
producing a ``put()`` inside the ``get()`` and therefore a reentrant
condition.

"""
from __future__ import annotations

import asyncio
from collections import deque
import threading
from time import time as _time
import typing
from typing import Any
from typing import Awaitable
from typing import Deque
from typing import Generic
from typing import Optional
from typing import TypeVar

from .concurrency import await_fallback
from .concurrency import await_only
from .langhelpers import memoized_property


_T = TypeVar("_T", bound=Any)
__all__ = ["Empty", "Full", "Queue"]


class Empty(Exception):
    "Exception raised by Queue.get(block=0)/get_nowait()."

    pass


class Full(Exception):
    "Exception raised by Queue.put(block=0)/put_nowait()."

    pass


class QueueCommon(Generic[_T]):
    maxsize: int
    use_lifo: bool

    def __init__(self, maxsize: int = 0, use_lifo: bool = False): ...

    def empty(self) -> bool:
        raise NotImplementedError()

    def full(self) -> bool:
        raise NotImplementedError()

    def qsize(self) -> int:
        raise NotImplementedError()

    def put_nowait(self, item: _T) -> None:
        raise NotImplementedError()

    def put(
        self, item: _T, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        raise NotImplementedError()

    def get_nowait(self) -> _T:
        raise NotImplementedError()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> _T:
        raise NotImplementedError()


class Queue(QueueCommon[_T]):
    queue: Deque[_T]

    def __init__(self, maxsize: int = 0, use_lifo: bool = False):
        """Initialize a queue object with a given maximum size.

        If `maxsize` is <= 0, the queue size is infinite.

        If `use_lifo` is True, this Queue acts like a Stack (LIFO).
        """

        self._init(maxsize)
        # mutex must be held whenever the queue is mutating.  All methods
        # that acquire mutex must release it before returning.  mutex
        # is shared between the two conditions, so acquiring and
        # releasing the conditions also acquires and releases mutex.
        self.mutex = threading.RLock()
        # Notify not_empty whenever an item is added to the queue; a
        # thread waiting to get is notified then.
        self.not_empty = threading.Condition(self.mutex)
        # Notify not_full whenever an item is removed from the queue;
        # a thread waiting to put is notified then.
        self.not_full = threading.Condition(self.mutex)
        # If this queue uses LIFO or FIFO
        self.use_lifo = use_lifo

    def qsize(self) -> int:
        """Return the approximate size of the queue (not reliable!)."""

        with self.mutex:
            return self._qsize()

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise (not
        reliable!)."""

        with self.mutex:
            return self._empty()

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise (not
        reliable!)."""

        with self.mutex:
            return self._full()

    def put(
        self, item: _T, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        """Put an item into the queue.

        If optional args `block` is True and `timeout` is None (the
        default), block if necessary until a free slot is
        available. If `timeout` is a positive number, it blocks at
        most `timeout` seconds and raises the ``Full`` exception if no
        free slot was available within that time.  Otherwise (`block`
        is false), put an item on the queue if a free slot is
        immediately available, else raise the ``Full`` exception
        (`timeout` is ignored in that case).
        """

        with self.not_full:
            if not block:
                if self._full():
                    raise Full
            elif timeout is None:
                while self._full():
                    self.not_full.wait()
            else:
                if timeout < 0:
                    raise ValueError("'timeout' must be a positive number")
                endtime = _time() + timeout
                while self._full():
                    remaining = endtime - _time()
                    if remaining <= 0.0:
                        raise Full
                    self.not_full.wait(remaining)
            self._put(item)
            self.not_empty.notify()

    def put_nowait(self, item: _T) -> None:
        """Put an item into the queue without blocking.

        Only enqueue the item if a free slot is immediately available.
        Otherwise raise the ``Full`` exception.
        """
        return self.put(item, False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> _T:
        """Remove and return an item from the queue.

        If optional args `block` is True and `timeout` is None (the
        default), block if necessary until an item is available. If
        `timeout` is a positive number, it blocks at most `timeout`
        seconds and raises the ``Empty`` exception if no item was
        available within that time.  Otherwise (`block` is false),
        return an item if one is immediately available, else raise the
        ``Empty`` exception (`timeout` is ignored in that case).

        """
        with self.not_empty:
            if not block:
                if self._empty():
                    raise Empty
            elif timeout is None:
                while self._empty():
                    self.not_empty.wait()
            else:
                if timeout < 0:
                    raise ValueError("'timeout' must be a positive number")
                endtime = _time() + timeout
                while self._empty():
                    remaining = endtime - _time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            self.not_full.notify()
            return item

    def get_nowait(self) -> _T:
        """Remove and return an item from the queue without blocking.

        Only get an item if one is immediately available. Otherwise
        raise the ``Empty`` exception.
        """

        return self.get(False)

    def _init(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self.queue = deque()

    def _qsize(self) -> int:
        return len(self.queue)

    def _empty(self) -> bool:
        return not self.queue

    def _full(self) -> bool:
        return self.maxsize > 0 and len(self.queue) == self.maxsize

    def _put(self, item: _T) -> None:
        self.queue.append(item)

    def _get(self) -> _T:
        if self.use_lifo:
            # LIFO
            return self.queue.pop()
        else:
            # FIFO
            return self.queue.popleft()


class AsyncAdaptedQueue(QueueCommon[_T]):
    if typing.TYPE_CHECKING:

        @staticmethod
        def await_(coroutine: Awaitable[Any]) -> _T: ...

    else:
        await_ = staticmethod(await_only)

    def __init__(self, maxsize: int = 0, use_lifo: bool = False):
        self.use_lifo = use_lifo
        self.maxsize = maxsize

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self):
        return self._queue.full()

    def qsize(self):
        return self._queue.qsize()

    @memoized_property
    def _queue(self) -> asyncio.Queue[_T]:
        # Delay creation of the queue until it is first used, to avoid
        # binding it to a possibly wrong event loop.
        # By delaying the creation of the pool we accommodate the common
        # usage pattern of instantiating the engine at module level, where a
        # different event loop is in present compared to when the application
        # is actually run.

        queue: asyncio.Queue[_T]

        if self.use_lifo:
            queue = asyncio.LifoQueue(maxsize=self.maxsize)
        else:
            queue = asyncio.Queue(maxsize=self.maxsize)
        return queue

    def put_nowait(self, item: _T) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as err:
            raise Full() from err

    def put(
        self, item: _T, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        if not block:
            return self.put_nowait(item)

        try:
            if timeout is not None:
                self.await_(asyncio.wait_for(self._queue.put(item), timeout))
            else:
                self.await_(self._queue.put(item))
        except (asyncio.QueueFull, asyncio.TimeoutError) as err:
            raise Full() from err

    def get_nowait(self) -> _T:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty as err:
            raise Empty() from err

    def get(self, block: bool = True, timeout: Optional[float] = None) -> _T:
        if not block:
            return self.get_nowait()

        try:
            if timeout is not None:
                return self.await_(
                    asyncio.wait_for(self._queue.get(), timeout)
                )
            else:
                return self.await_(self._queue.get())
        except (asyncio.QueueEmpty, asyncio.TimeoutError) as err:
            raise Empty() from err


class FallbackAsyncAdaptedQueue(AsyncAdaptedQueue[_T]):
    if not typing.TYPE_CHECKING:
        await_ = staticmethod(await_fallback)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\sathandlers.py ===
from collections import defaultdict

from sympy.assumptions.ask import Q
from sympy.core import (Add, Mul, Pow, Number, NumberSymbol, Symbol)
from sympy.core.numbers import ImaginaryUnit
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import (Equivalent, And, Or, Implies)
from sympy.matrices.expressions import MatMul

# APIs here may be subject to change


### Helper functions ###

def allargs(symbol, fact, expr):
    """
    Apply all arguments of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import allargs
    >>> from sympy.abc import x, y
    >>> allargs(x, Q.negative(x) | Q.positive(x), x*y)
    (Q.negative(x) | Q.positive(x)) & (Q.negative(y) | Q.positive(y))

    """
    return And(*[fact.subs(symbol, arg) for arg in expr.args])


def anyarg(symbol, fact, expr):
    """
    Apply any argument of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import anyarg
    >>> from sympy.abc import x, y
    >>> anyarg(x, Q.negative(x) & Q.positive(x), x*y)
    (Q.negative(x) & Q.positive(x)) | (Q.negative(y) & Q.positive(y))

    """
    return Or(*[fact.subs(symbol, arg) for arg in expr.args])


def exactlyonearg(symbol, fact, expr):
    """
    Apply exactly one argument of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import exactlyonearg
    >>> from sympy.abc import x, y
    >>> exactlyonearg(x, Q.positive(x), x*y)
    (Q.positive(x) & ~Q.positive(y)) | (Q.positive(y) & ~Q.positive(x))

    """
    pred_args = [fact.subs(symbol, arg) for arg in expr.args]
    res = Or(*[And(pred_args[i], *[~lit for lit in pred_args[:i] +
        pred_args[i+1:]]) for i in range(len(pred_args))])
    return res


### Fact registry ###

class ClassFactRegistry:
    """
    Register handlers against classes.

    Explanation
    ===========

    ``register`` method registers the handler function for a class. Here,
    handler function should return a single fact. ``multiregister`` method
    registers the handler function for multiple classes. Here, handler function
    should return a container of multiple facts.

    ``registry(expr)`` returns a set of facts for *expr*.

    Examples
    ========

    Here, we register the facts for ``Abs``.

    >>> from sympy import Abs, Equivalent, Q
    >>> from sympy.assumptions.sathandlers import ClassFactRegistry
    >>> reg = ClassFactRegistry()
    >>> @reg.register(Abs)
    ... def f1(expr):
    ...     return Q.nonnegative(expr)
    >>> @reg.register(Abs)
    ... def f2(expr):
    ...     arg = expr.args[0]
    ...     return Equivalent(~Q.zero(arg), ~Q.zero(expr))

    Calling the registry with expression returns the defined facts for the
    expression.

    >>> from sympy.abc import x
    >>> reg(Abs(x))
    {Q.nonnegative(Abs(x)), Equivalent(~Q.zero(x), ~Q.zero(Abs(x)))}

    Multiple facts can be registered at once by ``multiregister`` method.

    >>> reg2 = ClassFactRegistry()
    >>> @reg2.multiregister(Abs)
    ... def _(expr):
    ...     arg = expr.args[0]
    ...     return [Q.even(arg) >> Q.even(expr), Q.odd(arg) >> Q.odd(expr)]
    >>> reg2(Abs(x))
    {Implies(Q.even(x), Q.even(Abs(x))), Implies(Q.odd(x), Q.odd(Abs(x)))}

    """
    def __init__(self):
        self.singlefacts = defaultdict(frozenset)
        self.multifacts = defaultdict(frozenset)

    def register(self, cls):
        def _(func):
            self.singlefacts[cls] |= {func}
            return func
        return _

    def multiregister(self, *classes):
        def _(func):
            for cls in classes:
                self.multifacts[cls] |= {func}
            return func
        return _

    def __getitem__(self, key):
        ret1 = self.singlefacts[key]
        for k in self.singlefacts:
            if issubclass(key, k):
                ret1 |= self.singlefacts[k]

        ret2 = self.multifacts[key]
        for k in self.multifacts:
            if issubclass(key, k):
                ret2 |= self.multifacts[k]

        return ret1, ret2

    def __call__(self, expr):
        ret = set()

        handlers1, handlers2 = self[type(expr)]

        ret.update(h(expr) for h in handlers1)
        for h in handlers2:
            ret.update(h(expr))
        return ret

class_fact_registry = ClassFactRegistry()



### Class fact registration ###

x = Symbol('x')

## Abs ##

@class_fact_registry.multiregister(Abs)
def _(expr):
    arg = expr.args[0]
    return [Q.nonnegative(expr),
            Equivalent(~Q.zero(arg), ~Q.zero(expr)),
            Q.even(arg) >> Q.even(expr),
            Q.odd(arg) >> Q.odd(expr),
            Q.integer(arg) >> Q.integer(expr),
            ]


### Add ##

@class_fact_registry.multiregister(Add)
def _(expr):
    return [allargs(x, Q.positive(x), expr) >> Q.positive(expr),
            allargs(x, Q.negative(x), expr) >> Q.negative(expr),
            allargs(x, Q.real(x), expr) >> Q.real(expr),
            allargs(x, Q.rational(x), expr) >> Q.rational(expr),
            allargs(x, Q.integer(x), expr) >> Q.integer(expr),
            exactlyonearg(x, ~Q.integer(x), expr) >> ~Q.integer(expr),
            ]

@class_fact_registry.register(Add)
def _(expr):
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))


### Mul ###

@class_fact_registry.multiregister(Mul)
def _(expr):
    return [Equivalent(Q.zero(expr), anyarg(x, Q.zero(x), expr)),
            allargs(x, Q.positive(x), expr) >> Q.positive(expr),
            allargs(x, Q.real(x), expr) >> Q.real(expr),
            allargs(x, Q.rational(x), expr) >> Q.rational(expr),
            allargs(x, Q.integer(x), expr) >> Q.integer(expr),
            exactlyonearg(x, ~Q.rational(x), expr) >> ~Q.integer(expr),
            allargs(x, Q.commutative(x), expr) >> Q.commutative(expr),
            ]

@class_fact_registry.register(Mul)
def _(expr):
    # Implicitly assumes Mul has more than one arg
    # Would be allargs(x, Q.prime(x) | Q.composite(x)) except 1 is composite
    # More advanced prime assumptions will require inequalities, as 1 provides
    # a corner case.
    allargs_prime = allargs(x, Q.prime(x), expr)
    return Implies(allargs_prime, ~Q.prime(expr))

@class_fact_registry.register(Mul)
def _(expr):
    # General Case: Odd number of imaginary args implies mul is imaginary(To be implemented)
    allargs_imag_or_real = allargs(x, Q.imaginary(x) | Q.real(x), expr)
    onearg_imaginary = exactlyonearg(x, Q.imaginary(x), expr)
    return Implies(allargs_imag_or_real, Implies(onearg_imaginary, Q.imaginary(expr)))

@class_fact_registry.register(Mul)
def _(expr):
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))

@class_fact_registry.register(Mul)
def _(expr):
    # Including the integer qualification means we don't need to add any facts
    # for odd, since the assumptions already know that every integer is
    # exactly one of even or odd.
    allargs_integer = allargs(x, Q.integer(x), expr)
    anyarg_even = anyarg(x, Q.even(x), expr)
    return Implies(allargs_integer, Equivalent(anyarg_even, Q.even(expr)))


### MatMul ###

@class_fact_registry.register(MatMul)
def _(expr):
    allargs_square = allargs(x, Q.square(x), expr)
    allargs_invertible = allargs(x, Q.invertible(x), expr)
    return Implies(allargs_square, Equivalent(Q.invertible(expr), allargs_invertible))


### Pow ###

@class_fact_registry.multiregister(Pow)
def _(expr):
    base, exp = expr.base, expr.exp
    return [
        (Q.real(base) & Q.even(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr),
        (Q.nonnegative(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr),
        (Q.nonpositive(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonpositive(expr),
        Equivalent(Q.zero(expr), Q.zero(base) & Q.positive(exp))
    ]


### Numbers ###

_old_assump_getters = {
    Q.positive: lambda o: o.is_positive,
    Q.zero: lambda o: o.is_zero,
    Q.negative: lambda o: o.is_negative,
    Q.rational: lambda o: o.is_rational,
    Q.irrational: lambda o: o.is_irrational,
    Q.even: lambda o: o.is_even,
    Q.odd: lambda o: o.is_odd,
    Q.imaginary: lambda o: o.is_imaginary,
    Q.prime: lambda o: o.is_prime,
    Q.composite: lambda o: o.is_composite,
}

@class_fact_registry.multiregister(Number, NumberSymbol, ImaginaryUnit)
def _(expr):
    ret = []
    for p, getter in _old_assump_getters.items():
        pred = p(expr)
        prop = getter(expr)
        if prop is not None:
            ret.append(Equivalent(pred, prop))
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\sam2_demo.py ===
# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2_image_onnx_predictor import SAM2ImageOnnxPredictor
from sam2_utils import load_sam2_model

import onnxruntime


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
    output_image_file_prefix=None,
    image_files=None,
):
    for i, (mask, score) in enumerate(zip(masks, scores, strict=False)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())

        if box_coords is not None:
            show_box(box_coords, plt.gca())

        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)

        plt.axis("off")
        if output_image_file_prefix:
            filename = f"{output_image_file_prefix}_{i}.png"
            if os.path.exists(filename):
                os.remove(filename)
            plt.savefig(filename, format="png", bbox_inches="tight", pad_inches=0)
            if isinstance(image_files, list):
                image_files.append(filename)
        plt.show(block=False)
        plt.close()


def get_predictor(
    sam2_dir: str,
    device: str | torch.device,
    dtype: torch.dtype,
    model_type="sam2_hiera_large",
    engine="torch",
    image_encoder_onnx_path: str = "",
    image_decoder_onnx_path: str = "",
    image_decoder_multi_onnx_path: str = "",
    provider: str = "CUDAExecutionProvider",
):
    sam2_model = load_sam2_model(sam2_dir, model_type, device=device)
    if engine == "torch":
        predictor = SAM2ImagePredictor(sam2_model)
    else:
        predictor = SAM2ImageOnnxPredictor(
            sam2_model,
            image_encoder_onnx_path=image_encoder_onnx_path,
            image_decoder_onnx_path=image_decoder_onnx_path,
            image_decoder_multi_onnx_path=image_decoder_multi_onnx_path,
            provider=provider,
            device=device,
            onnx_dtype=dtype,
        )
    return predictor


def run_demo(
    sam2_dir: str,
    model_type: str = "sam2_hiera_large",
    engine: str = "torch",
    dtype: torch.dtype = torch.float32,
    image_encoder_onnx_path: str = "",
    image_decoder_onnx_path: str = "",
    image_decoder_multi_onnx_path: str = "",
    use_gpu: bool = True,
    enable_batch: bool = False,
):
    if use_gpu:
        assert torch.cuda.is_available()
        assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    device = torch.device("cuda" if use_gpu else "cpu")

    if use_gpu and engine == "torch" and torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    np.random.seed(3)
    image = Image.open("truck.jpg")
    image = np.array(image.convert("RGB"))

    predictor = get_predictor(
        sam2_dir,
        device,
        dtype,
        model_type,
        engine,
        image_encoder_onnx_path,
        image_decoder_onnx_path,
        image_decoder_multi_onnx_path,
        provider=provider,
    )

    predictor.set_image(image)
    prefix = f"sam2_demo_{engine}_"

    #  The model returns masks, quality predictions for those masks,
    #     and low resolution mask logits that can be passed to the next iteration of prediction.
    #  With multimask_output=True (the default setting), SAM 2 outputs 3 masks, where
    #     scores gives the model's own estimation of the quality of these masks.
    #  For ambiguous prompts such as a single point, it is recommended to use multimask_output=True
    #     even if only a single mask is desired;
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    image_files = []
    show_masks(
        image,
        masks,
        scores,
        point_coords=input_point,
        input_labels=input_label,
        borders=True,
        output_image_file_prefix=prefix + "multimask",
        image_files=image_files,
    )

    # Multiple points.
    input_point = np.array([[500, 375], [1125, 625]])
    input_label = np.array([1, 1])
    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    show_masks(
        image,
        masks,
        scores,
        point_coords=input_point,
        input_labels=input_label,
        output_image_file_prefix=prefix + "multi_points",
        image_files=image_files,
    )

    # Specify a window and a background point.
    input_point = np.array([[500, 375], [1125, 625]])
    input_label = np.array([1, 0])
    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    show_masks(
        image,
        masks,
        scores,
        point_coords=input_point,
        input_labels=input_label,
        output_image_file_prefix=prefix + "background_point",
        image_files=image_files,
    )

    # Take a box as input
    input_box = np.array([425, 600, 700, 875])
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    show_masks(
        image,
        masks,
        scores,
        box_coords=input_box,
        output_image_file_prefix=prefix + "box",
        image_files=image_files,
    )

    # Combining points and boxes
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )
    show_masks(
        image,
        masks,
        scores,
        box_coords=input_box,
        point_coords=input_point,
        input_labels=input_label,
        output_image_file_prefix=prefix + "box_and_point",
        image_files=image_files,
    )

    # TODO: support batched prompt inputs
    if enable_batch:
        input_boxes = np.array(
            [
                [75, 275, 1725, 850],
                [425, 600, 700, 875],
                [1375, 550, 1650, 800],
                [1240, 675, 1400, 750],
            ]
        )
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.squeeze(0), plt.gca(), random_color=True)
        for box in input_boxes:
            show_box(box, plt.gca())
        plt.axis("off")
        plt.show()
        plt.savefig(prefix + "batch_prompt.png")
        image_files.append(prefix + "batch_prompt.png")
    return image_files


def show_all_images(left_images, right_images, suffix=""):
    # Show images in two rows since display screen is horizontal in most cases.
    fig, axes = plt.subplots(nrows=2, ncols=len(left_images), figsize=(19.20, 10.80))
    for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images, strict=False)):
        left_img = mpimg.imread(left_img_path)
        right_img = mpimg.imread(right_img_path)

        axes[0, i].imshow(left_img)
        axes[0, i].set_title(left_img_path.replace("sam2_demo_", "").replace(".png", ""), fontsize=10)
        axes[0, i].axis("off")
        axes[0, i].set_aspect(left_img.shape[1] / left_img.shape[0])

        axes[1, i].imshow(right_img)
        axes[1, i].set_title(right_img_path.replace("sam2_demo_", "").replace(".png", ""), fontsize=10)
        axes[1, i].axis("off")
        axes[1, i].set_aspect(right_img.shape[1] / right_img.shape[0])

    plt.tight_layout()
    plt.savefig(f"sam2_demo{suffix}.png", format="png", bbox_inches="tight", dpi=1000)
    plt.show()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\polyfuncs.py ===
"""High-level polynomials manipulation functions. """


from sympy.core import S, Basic, symbols, Dummy
from sympy.polys.polyerrors import (
    PolificationFailed, ComputationFailed,
    MultivariatePolynomialError, OptionError)
from sympy.polys.polyoptions import allowed_flags, build_options
from sympy.polys.polytools import poly_from_expr, Poly
from sympy.polys.specialpolys import (
    symmetric_poly, interpolating_poly)
from sympy.polys.rings import sring
from sympy.utilities import numbered_symbols, take, public

@public
def symmetrize(F, *gens, **args):
    r"""
    Rewrite a polynomial in terms of elementary symmetric polynomials.

    A symmetric polynomial is a multivariate polynomial that remains invariant
    under any variable permutation, i.e., if `f = f(x_1, x_2, \dots, x_n)`,
    then `f = f(x_{i_1}, x_{i_2}, \dots, x_{i_n})`, where
    `(i_1, i_2, \dots, i_n)` is a permutation of `(1, 2, \dots, n)` (an
    element of the group `S_n`).

    Returns a tuple of symmetric polynomials ``(f1, f2, ..., fn)`` such that
    ``f = f1 + f2 + ... + fn``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import symmetrize
    >>> from sympy.abc import x, y

    >>> symmetrize(x**2 + y**2)
    (-2*x*y + (x + y)**2, 0)

    >>> symmetrize(x**2 + y**2, formal=True)
    (s1**2 - 2*s2, 0, [(s1, x + y), (s2, x*y)])

    >>> symmetrize(x**2 - y**2)
    (-2*x*y + (x + y)**2, -2*y**2)

    >>> symmetrize(x**2 - y**2, formal=True)
    (s1**2 - 2*s2, -2*y**2, [(s1, x + y), (s2, x*y)])

    """
    allowed_flags(args, ['formal', 'symbols'])

    iterable = True

    if not hasattr(F, '__iter__'):
        iterable = False
        F = [F]

    R, F = sring(F, *gens, **args)
    gens = R.symbols

    opt = build_options(gens, args)
    symbols = opt.symbols
    symbols = [next(symbols) for i in range(len(gens))]

    result = []

    for f in F:
        p, r, m = f.symmetrize()
        result.append((p.as_expr(*symbols), r.as_expr(*gens)))

    polys = [(s, g.as_expr()) for s, (_, g) in zip(symbols, m)]

    if not opt.formal:
        for i, (sym, non_sym) in enumerate(result):
            result[i] = (sym.subs(polys), non_sym)

    if not iterable:
        result, = result

    if not opt.formal:
        return result
    else:
        if iterable:
            return result, polys
        else:
            return result + (polys,)


@public
def horner(f, *gens, **args):
    """
    Rewrite a polynomial in Horner form.

    Among other applications, evaluation of a polynomial at a point is optimal
    when it is applied using the Horner scheme ([1]).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import horner
    >>> from sympy.abc import x, y, a, b, c, d, e

    >>> horner(9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5)
    x*(x*(x*(9*x + 8) + 7) + 6) + 5

    >>> horner(a*x**4 + b*x**3 + c*x**2 + d*x + e)
    e + x*(d + x*(c + x*(a*x + b)))

    >>> f = 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y

    >>> horner(f, wrt=x)
    x*(x*y*(4*y + 2) + y*(2*y + 1))

    >>> horner(f, wrt=y)
    y*(x*y*(4*x + 2) + x*(2*x + 1))

    References
    ==========
    [1] - https://en.wikipedia.org/wiki/Horner_scheme

    """
    allowed_flags(args, [])

    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr

    form, gen = S.Zero, F.gen

    if F.is_univariate:
        for coeff in F.all_coeffs():
            form = form*gen + coeff
    else:
        F, gens = Poly(F, gen), gens[1:]

        for coeff in F.all_coeffs():
            form = form*gen + horner(coeff, *gens, **args)

    return form


@public
def interpolate(data, x):
    """
    Construct an interpolating polynomial for the data points
    evaluated at point x (which can be symbolic or numeric).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import interpolate
    >>> from sympy.abc import a, b, x

    A list is interpreted as though it were paired with a range starting
    from 1:

    >>> interpolate([1, 4, 9, 16], x)
    x**2

    This can be made explicit by giving a list of coordinates:

    >>> interpolate([(1, 1), (2, 4), (3, 9)], x)
    x**2

    The (x, y) coordinates can also be given as keys and values of a
    dictionary (and the points need not be equispaced):

    >>> interpolate([(-1, 2), (1, 2), (2, 5)], x)
    x**2 + 1
    >>> interpolate({-1: 2, 1: 2, 2: 5}, x)
    x**2 + 1

    If the interpolation is going to be used only once then the
    value of interest can be passed instead of passing a symbol:

    >>> interpolate([1, 4, 9], 5)
    25

    Symbolic coordinates are also supported:

    >>> [(i,interpolate((a, b), i)) for i in range(1, 4)]
    [(1, a), (2, b), (3, -a + 2*b)]
    """
    n = len(data)

    if isinstance(data, dict):
        if x in data:
            return S(data[x])
        X, Y = list(zip(*data.items()))
    else:
        if isinstance(data[0], tuple):
            X, Y = list(zip(*data))
            if x in X:
                return S(Y[X.index(x)])
        else:
            if x in range(1, n + 1):
                return S(data[x - 1])
            Y = list(data)
            X = list(range(1, n + 1))

    try:
        return interpolating_poly(n, x, X, Y).expand()
    except ValueError:
        d = Dummy()
        return interpolating_poly(n, d, X, Y).expand().subs(d, x)


@public
def rational_interpolate(data, degnum, X=symbols('x')):
    """
    Returns a rational interpolation, where the data points are element of
    any integral domain.

    The first argument  contains the data (as a list of coordinates). The
    ``degnum`` argument is the degree in the numerator of the rational
    function. Setting it too high will decrease the maximal degree in the
    denominator for the same amount of data.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import rational_interpolate

    >>> data = [(1, -210), (2, -35), (3, 105), (4, 231), (5, 350), (6, 465)]
    >>> rational_interpolate(data, 2)
    (105*x**2 - 525)/(x + 1)

    Values do not need to be integers:

    >>> from sympy import sympify
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = sympify("[-1, 0, 2, 22/5, 7, 68/7]")
    >>> rational_interpolate(zip(x, y), 2)
    (3*x**2 - 7*x + 2)/(x + 1)

    The symbol for the variable can be changed if needed:
    >>> from sympy import symbols
    >>> z = symbols('z')
    >>> rational_interpolate(data, 2, X=z)
    (105*z**2 - 525)/(z + 1)

    References
    ==========

    .. [1] Algorithm is adapted from:
           http://axiom-wiki.newsynthesis.org/RationalInterpolation

    """
    from sympy.matrices.dense import ones

    xdata, ydata = list(zip(*data))

    k = len(xdata) - degnum - 1
    if k < 0:
        raise OptionError("Too few values for the required degree.")
    c = ones(degnum + k + 1, degnum + k + 2)
    for j in range(max(degnum, k)):
        for i in range(degnum + k + 1):
            c[i, j + 1] = c[i, j]*xdata[i]
    for j in range(k + 1):
        for i in range(degnum + k + 1):
            c[i, degnum + k + 1 - j] = -c[i, k - j]*ydata[i]
    r = c.nullspace()[0]
    return (sum(r[i] * X**i for i in range(degnum + 1))
            / sum(r[i + degnum + 1] * X**i for i in range(k + 1)))


@public
def viete(f, roots=None, *gens, **args):
    """
    Generate Viete's formulas for ``f``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import viete
    >>> from sympy import symbols

    >>> x, a, b, c, r1, r2 = symbols('x,a:c,r1:3')

    >>> viete(a*x**2 + b*x + c, [r1, r2], x)
    [(r1 + r2, -b/a), (r1*r2, c/a)]

    """
    allowed_flags(args, [])

    if isinstance(roots, Basic):
        gens, roots = (roots,) + gens, None

    try:
        f, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('viete', 1, exc)

    if f.is_multivariate:
        raise MultivariatePolynomialError(
            "multivariate polynomials are not allowed")

    n = f.degree()

    if n < 1:
        raise ValueError(
            "Cannot derive Viete's formulas for a constant polynomial")

    if roots is None:
        roots = numbered_symbols('r', start=1)

    roots = take(roots, n)

    if n != len(roots):
        raise ValueError("required %s roots, got %s" % (n, len(roots)))

    lc, coeffs = f.LC(), f.all_coeffs()
    result, sign = [], -1

    for i, coeff in enumerate(coeffs[1:]):
        poly = symmetric_poly(i + 1, roots)
        coeff = sign*(coeff/lc)
        result.append((poly, coeff))
        sign = -sign

    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\Graph.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Graph(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Graph()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsGraph(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def GraphBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Graph
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Graph
    def Initializers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def InitializersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def InitializersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # Graph
    def NodeArgs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.ValueInfo import ValueInfo
            obj = ValueInfo()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def NodeArgsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def NodeArgsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # Graph
    def Nodes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Node import Node
            obj = Node()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def NodesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def NodesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Graph
    def MaxNodeIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Graph
    def NodeEdges(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.NodeEdge import NodeEdge
            obj = NodeEdge()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def NodeEdgesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def NodeEdgesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # Graph
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Graph
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Graph
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Graph
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # Graph
    def SparseInitializers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.SparseTensor import SparseTensor
            obj = SparseTensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Graph
    def SparseInitializersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Graph
    def SparseInitializersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

    # Graph
    def RuntimeOptimizations(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.RuntimeOptimizations import RuntimeOptimizations
            obj = RuntimeOptimizations()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def GraphStart(builder):
    builder.StartObject(9)

def Start(builder):
    GraphStart(builder)

def GraphAddInitializers(builder, initializers):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(initializers), 0)

def AddInitializers(builder, initializers):
    GraphAddInitializers(builder, initializers)

def GraphStartInitializersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInitializersVector(builder, numElems: int) -> int:
    return GraphStartInitializersVector(builder, numElems)

def GraphAddNodeArgs(builder, nodeArgs):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(nodeArgs), 0)

def AddNodeArgs(builder, nodeArgs):
    GraphAddNodeArgs(builder, nodeArgs)

def GraphStartNodeArgsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodeArgsVector(builder, numElems: int) -> int:
    return GraphStartNodeArgsVector(builder, numElems)

def GraphAddNodes(builder, nodes):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(nodes), 0)

def AddNodes(builder, nodes):
    GraphAddNodes(builder, nodes)

def GraphStartNodesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodesVector(builder, numElems: int) -> int:
    return GraphStartNodesVector(builder, numElems)

def GraphAddMaxNodeIndex(builder, maxNodeIndex):
    builder.PrependUint32Slot(3, maxNodeIndex, 0)

def AddMaxNodeIndex(builder, maxNodeIndex):
    GraphAddMaxNodeIndex(builder, maxNodeIndex)

def GraphAddNodeEdges(builder, nodeEdges):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(nodeEdges), 0)

def AddNodeEdges(builder, nodeEdges):
    GraphAddNodeEdges(builder, nodeEdges)

def GraphStartNodeEdgesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodeEdgesVector(builder, numElems: int) -> int:
    return GraphStartNodeEdgesVector(builder, numElems)

def GraphAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)

def AddInputs(builder, inputs):
    GraphAddInputs(builder, inputs)

def GraphStartInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputsVector(builder, numElems: int) -> int:
    return GraphStartInputsVector(builder, numElems)

def GraphAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)

def AddOutputs(builder, outputs):
    GraphAddOutputs(builder, outputs)

def GraphStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOutputsVector(builder, numElems: int) -> int:
    return GraphStartOutputsVector(builder, numElems)

def GraphAddSparseInitializers(builder, sparseInitializers):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(sparseInitializers), 0)

def AddSparseInitializers(builder, sparseInitializers):
    GraphAddSparseInitializers(builder, sparseInitializers)

def GraphStartSparseInitializersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSparseInitializersVector(builder, numElems: int) -> int:
    return GraphStartSparseInitializersVector(builder, numElems)

def GraphAddRuntimeOptimizations(builder, runtimeOptimizations):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(runtimeOptimizations), 0)

def AddRuntimeOptimizations(builder, runtimeOptimizations):
    GraphAddRuntimeOptimizations(builder, runtimeOptimizations)

def GraphEnd(builder):
    return builder.EndObject()

def End(builder):
    return GraphEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\resolvelib\resolver.py ===
import contextlib
import functools
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, cast

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.resolvelib import BaseReporter, ResolutionImpossible, ResolutionTooDeep
from pip._vendor.resolvelib import Resolver as RLResolver
from pip._vendor.resolvelib.structs import DirectedGraph

from pip._internal.cache import WheelCache
from pip._internal.exceptions import ResolutionTooDeepError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import install_req_extend_extras
from pip._internal.req.req_install import InstallRequirement
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver, InstallRequirementProvider
from pip._internal.resolution.resolvelib.provider import PipProvider
from pip._internal.resolution.resolvelib.reporter import (
    PipDebuggingReporter,
    PipReporter,
)
from pip._internal.utils.packaging import get_requirement

from .base import Candidate, Requirement
from .factory import Factory

if TYPE_CHECKING:
    from pip._vendor.resolvelib.resolvers import Result as RLResult

    Result = RLResult[Requirement, Candidate, str]


logger = logging.getLogger(__name__)


class Resolver(BaseResolver):
    _allowed_strategies = {"eager", "only-if-needed", "to-satisfy-only"}

    def __init__(
        self,
        preparer: RequirementPreparer,
        finder: PackageFinder,
        wheel_cache: Optional[WheelCache],
        make_install_req: InstallRequirementProvider,
        use_user_site: bool,
        ignore_dependencies: bool,
        ignore_installed: bool,
        ignore_requires_python: bool,
        force_reinstall: bool,
        upgrade_strategy: str,
        py_version_info: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        assert upgrade_strategy in self._allowed_strategies

        self.factory = Factory(
            finder=finder,
            preparer=preparer,
            make_install_req=make_install_req,
            wheel_cache=wheel_cache,
            use_user_site=use_user_site,
            force_reinstall=force_reinstall,
            ignore_installed=ignore_installed,
            ignore_requires_python=ignore_requires_python,
            py_version_info=py_version_info,
        )
        self.ignore_dependencies = ignore_dependencies
        self.upgrade_strategy = upgrade_strategy
        self._result: Optional[Result] = None

    def resolve(
        self, root_reqs: List[InstallRequirement], check_supported_wheels: bool
    ) -> RequirementSet:
        collected = self.factory.collect_root_requirements(root_reqs)
        provider = PipProvider(
            factory=self.factory,
            constraints=collected.constraints,
            ignore_dependencies=self.ignore_dependencies,
            upgrade_strategy=self.upgrade_strategy,
            user_requested=collected.user_requested,
        )
        if "PIP_RESOLVER_DEBUG" in os.environ:
            reporter: BaseReporter[Requirement, Candidate, str] = PipDebuggingReporter()
        else:
            reporter = PipReporter()
        resolver: RLResolver[Requirement, Candidate, str] = RLResolver(
            provider,
            reporter,
        )

        try:
            limit_how_complex_resolution_can_be = 200000
            result = self._result = resolver.resolve(
                collected.requirements, max_rounds=limit_how_complex_resolution_can_be
            )

        except ResolutionImpossible as e:
            error = self.factory.get_installation_error(
                cast("ResolutionImpossible[Requirement, Candidate]", e),
                collected.constraints,
            )
            raise error from e
        except ResolutionTooDeep:
            raise ResolutionTooDeepError from None

        req_set = RequirementSet(check_supported_wheels=check_supported_wheels)
        # process candidates with extras last to ensure their base equivalent is
        # already in the req_set if appropriate.
        # Python's sort is stable so using a binary key function keeps relative order
        # within both subsets.
        for candidate in sorted(
            result.mapping.values(), key=lambda c: c.name != c.project_name
        ):
            ireq = candidate.get_install_requirement()
            if ireq is None:
                if candidate.name != candidate.project_name:
                    # extend existing req's extras
                    with contextlib.suppress(KeyError):
                        req = req_set.get_requirement(candidate.project_name)
                        req_set.add_named_requirement(
                            install_req_extend_extras(
                                req, get_requirement(candidate.name).extras
                            )
                        )
                continue

            # Check if there is already an installation under the same name,
            # and set a flag for later stages to uninstall it, if needed.
            installed_dist = self.factory.get_dist_to_uninstall(candidate)
            if installed_dist is None:
                # There is no existing installation -- nothing to uninstall.
                ireq.should_reinstall = False
            elif self.factory.force_reinstall:
                # The --force-reinstall flag is set -- reinstall.
                ireq.should_reinstall = True
            elif installed_dist.version != candidate.version:
                # The installation is different in version -- reinstall.
                ireq.should_reinstall = True
            elif candidate.is_editable or installed_dist.editable:
                # The incoming distribution is editable, or different in
                # editable-ness to installation -- reinstall.
                ireq.should_reinstall = True
            elif candidate.source_link and candidate.source_link.is_file:
                # The incoming distribution is under file://
                if candidate.source_link.is_wheel:
                    # is a local wheel -- do nothing.
                    logger.info(
                        "%s is already installed with the same version as the "
                        "provided wheel. Use --force-reinstall to force an "
                        "installation of the wheel.",
                        ireq.name,
                    )
                    continue

                # is a local sdist or path -- reinstall
                ireq.should_reinstall = True
            else:
                continue

            link = candidate.source_link
            if link and link.is_yanked:
                # The reason can contain non-ASCII characters, Unicode
                # is required for Python 2.
                msg = (
                    "The candidate selected for download or install is a "
                    "yanked version: {name!r} candidate (version {version} "
                    "at {link})\nReason for being yanked: {reason}"
                ).format(
                    name=candidate.name,
                    version=candidate.version,
                    link=link,
                    reason=link.yanked_reason or "<none given>",
                )
                logger.warning(msg)

            req_set.add_named_requirement(ireq)

        reqs = req_set.all_requirements
        self.factory.preparer.prepare_linked_requirements_more(reqs)
        for req in reqs:
            req.prepared = True
            req.needs_more_preparation = False
        return req_set

    def get_installation_order(
        self, req_set: RequirementSet
    ) -> List[InstallRequirement]:
        """Get order for installation of requirements in RequirementSet.

        The returned list contains a requirement before another that depends on
        it. This helps ensure that the environment is kept consistent as they
        get installed one-by-one.

        The current implementation creates a topological ordering of the
        dependency graph, giving more weight to packages with less
        or no dependencies, while breaking any cycles in the graph at
        arbitrary points. We make no guarantees about where the cycle
        would be broken, other than it *would* be broken.
        """
        assert self._result is not None, "must call resolve() first"

        if not req_set.requirements:
            # Nothing is left to install, so we do not need an order.
            return []

        graph = self._result.graph
        weights = get_topological_weights(graph, set(req_set.requirements.keys()))

        sorted_items = sorted(
            req_set.requirements.items(),
            key=functools.partial(_req_set_item_sorter, weights=weights),
            reverse=True,
        )
        return [ireq for _, ireq in sorted_items]


def get_topological_weights(
    graph: "DirectedGraph[Optional[str]]", requirement_keys: Set[str]
) -> Dict[Optional[str], int]:
    """Assign weights to each node based on how "deep" they are.

    This implementation may change at any point in the future without prior
    notice.

    We first simplify the dependency graph by pruning any leaves and giving them
    the highest weight: a package without any dependencies should be installed
    first. This is done again and again in the same way, giving ever less weight
    to the newly found leaves. The loop stops when no leaves are left: all
    remaining packages have at least one dependency left in the graph.

    Then we continue with the remaining graph, by taking the length for the
    longest path to any node from root, ignoring any paths that contain a single
    node twice (i.e. cycles). This is done through a depth-first search through
    the graph, while keeping track of the path to the node.

    Cycles in the graph result would result in node being revisited while also
    being on its own path. In this case, take no action. This helps ensure we
    don't get stuck in a cycle.

    When assigning weight, the longer path (i.e. larger length) is preferred.

    We are only interested in the weights of packages that are in the
    requirement_keys.
    """
    path: Set[Optional[str]] = set()
    weights: Dict[Optional[str], int] = {}

    def visit(node: Optional[str]) -> None:
        if node in path:
            # We hit a cycle, so we'll break it here.
            return

        # Time to visit the children!
        path.add(node)
        for child in graph.iter_children(node):
            visit(child)
        path.remove(node)

        if node not in requirement_keys:
            return

        last_known_parent_count = weights.get(node, 0)
        weights[node] = max(last_known_parent_count, len(path))

    # Simplify the graph, pruning leaves that have no dependencies.
    # This is needed for large graphs (say over 200 packages) because the
    # `visit` function is exponentially slower then, taking minutes.
    # See https://github.com/pypa/pip/issues/10557
    # We will loop until we explicitly break the loop.
    while True:
        leaves = set()
        for key in graph:
            if key is None:
                continue
            for _child in graph.iter_children(key):
                # This means we have at least one child
                break
            else:
                # No child.
                leaves.add(key)
        if not leaves:
            # We are done simplifying.
            break
        # Calculate the weight for the leaves.
        weight = len(graph) - 1
        for leaf in leaves:
            if leaf not in requirement_keys:
                continue
            weights[leaf] = weight
        # Remove the leaves from the graph, making it simpler.
        for leaf in leaves:
            graph.remove(leaf)

    # Visit the remaining graph.
    # `None` is guaranteed to be the root node by resolvelib.
    visit(None)

    # Sanity check: all requirement keys should be in the weights,
    # and no other keys should be in the weights.
    difference = set(weights.keys()).difference(requirement_keys)
    assert not difference, difference

    return weights


def _req_set_item_sorter(
    item: Tuple[str, InstallRequirement],
    weights: Dict[Optional[str], int],
) -> Tuple[int, str]:
    """Key function used to sort install requirements for installation.

    Based on the "weight" mapping calculated in ``get_installation_order()``.
    The canonical package name is returned as the second member as a tie-
    breaker to ensure the result is predictable, which is useful in tests.
    """
    name = canonicalize_name(item[0])
    return weights[name], name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\idtracking.py ===
import typing as t

from . import nodes
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    import typing_extensions as te

VAR_LOAD_PARAMETER = "param"
VAR_LOAD_RESOLVE = "resolve"
VAR_LOAD_ALIAS = "alias"
VAR_LOAD_UNDEFINED = "undefined"


def find_symbols(
    nodes: t.Iterable[nodes.Node], parent_symbols: t.Optional["Symbols"] = None
) -> "Symbols":
    sym = Symbols(parent=parent_symbols)
    visitor = FrameSymbolVisitor(sym)
    for node in nodes:
        visitor.visit(node)
    return sym


def symbols_for_node(
    node: nodes.Node, parent_symbols: t.Optional["Symbols"] = None
) -> "Symbols":
    sym = Symbols(parent=parent_symbols)
    sym.analyze_node(node)
    return sym


class Symbols:
    def __init__(
        self, parent: t.Optional["Symbols"] = None, level: t.Optional[int] = None
    ) -> None:
        if level is None:
            if parent is None:
                level = 0
            else:
                level = parent.level + 1

        self.level: int = level
        self.parent = parent
        self.refs: t.Dict[str, str] = {}
        self.loads: t.Dict[str, t.Any] = {}
        self.stores: t.Set[str] = set()

    def analyze_node(self, node: nodes.Node, **kwargs: t.Any) -> None:
        visitor = RootVisitor(self)
        visitor.visit(node, **kwargs)

    def _define_ref(
        self, name: str, load: t.Optional[t.Tuple[str, t.Optional[str]]] = None
    ) -> str:
        ident = f"l_{self.level}_{name}"
        self.refs[name] = ident
        if load is not None:
            self.loads[ident] = load
        return ident

    def find_load(self, target: str) -> t.Optional[t.Any]:
        if target in self.loads:
            return self.loads[target]

        if self.parent is not None:
            return self.parent.find_load(target)

        return None

    def find_ref(self, name: str) -> t.Optional[str]:
        if name in self.refs:
            return self.refs[name]

        if self.parent is not None:
            return self.parent.find_ref(name)

        return None

    def ref(self, name: str) -> str:
        rv = self.find_ref(name)
        if rv is None:
            raise AssertionError(
                "Tried to resolve a name to a reference that was"
                f" unknown to the frame ({name!r})"
            )
        return rv

    def copy(self) -> "te.Self":
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.refs = self.refs.copy()
        rv.loads = self.loads.copy()
        rv.stores = self.stores.copy()
        return rv

    def store(self, name: str) -> None:
        self.stores.add(name)

        # If we have not see the name referenced yet, we need to figure
        # out what to set it to.
        if name not in self.refs:
            # If there is a parent scope we check if the name has a
            # reference there.  If it does it means we might have to alias
            # to a variable there.
            if self.parent is not None:
                outer_ref = self.parent.find_ref(name)
                if outer_ref is not None:
                    self._define_ref(name, load=(VAR_LOAD_ALIAS, outer_ref))
                    return

            # Otherwise we can just set it to undefined.
            self._define_ref(name, load=(VAR_LOAD_UNDEFINED, None))

    def declare_parameter(self, name: str) -> str:
        self.stores.add(name)
        return self._define_ref(name, load=(VAR_LOAD_PARAMETER, None))

    def load(self, name: str) -> None:
        if self.find_ref(name) is None:
            self._define_ref(name, load=(VAR_LOAD_RESOLVE, name))

    def branch_update(self, branch_symbols: t.Sequence["Symbols"]) -> None:
        stores: t.Set[str] = set()

        for branch in branch_symbols:
            stores.update(branch.stores)

        stores.difference_update(self.stores)

        for sym in branch_symbols:
            self.refs.update(sym.refs)
            self.loads.update(sym.loads)
            self.stores.update(sym.stores)

        for name in stores:
            target = self.find_ref(name)
            assert target is not None, "should not happen"

            if self.parent is not None:
                outer_target = self.parent.find_ref(name)
                if outer_target is not None:
                    self.loads[target] = (VAR_LOAD_ALIAS, outer_target)
                    continue
            self.loads[target] = (VAR_LOAD_RESOLVE, name)

    def dump_stores(self) -> t.Dict[str, str]:
        rv: t.Dict[str, str] = {}
        node: t.Optional[Symbols] = self

        while node is not None:
            for name in sorted(node.stores):
                if name not in rv:
                    rv[name] = self.find_ref(name)  # type: ignore

            node = node.parent

        return rv

    def dump_param_targets(self) -> t.Set[str]:
        rv = set()
        node: t.Optional[Symbols] = self

        while node is not None:
            for target, (instr, _) in self.loads.items():
                if instr == VAR_LOAD_PARAMETER:
                    rv.add(target)

            node = node.parent

        return rv


class RootVisitor(NodeVisitor):
    def __init__(self, symbols: "Symbols") -> None:
        self.sym_visitor = FrameSymbolVisitor(symbols)

    def _simple_visit(self, node: nodes.Node, **kwargs: t.Any) -> None:
        for child in node.iter_child_nodes():
            self.sym_visitor.visit(child)

    visit_Template = _simple_visit
    visit_Block = _simple_visit
    visit_Macro = _simple_visit
    visit_FilterBlock = _simple_visit
    visit_Scope = _simple_visit
    visit_If = _simple_visit
    visit_ScopedEvalContextModifier = _simple_visit

    def visit_AssignBlock(self, node: nodes.AssignBlock, **kwargs: t.Any) -> None:
        for child in node.body:
            self.sym_visitor.visit(child)

    def visit_CallBlock(self, node: nodes.CallBlock, **kwargs: t.Any) -> None:
        for child in node.iter_child_nodes(exclude=("call",)):
            self.sym_visitor.visit(child)

    def visit_OverlayScope(self, node: nodes.OverlayScope, **kwargs: t.Any) -> None:
        for child in node.body:
            self.sym_visitor.visit(child)

    def visit_For(
        self, node: nodes.For, for_branch: str = "body", **kwargs: t.Any
    ) -> None:
        if for_branch == "body":
            self.sym_visitor.visit(node.target, store_as_param=True)
            branch = node.body
        elif for_branch == "else":
            branch = node.else_
        elif for_branch == "test":
            self.sym_visitor.visit(node.target, store_as_param=True)
            if node.test is not None:
                self.sym_visitor.visit(node.test)
            return
        else:
            raise RuntimeError("Unknown for branch")

        if branch:
            for item in branch:
                self.sym_visitor.visit(item)

    def visit_With(self, node: nodes.With, **kwargs: t.Any) -> None:
        for target in node.targets:
            self.sym_visitor.visit(target)
        for child in node.body:
            self.sym_visitor.visit(child)

    def generic_visit(self, node: nodes.Node, *args: t.Any, **kwargs: t.Any) -> None:
        raise NotImplementedError(f"Cannot find symbols for {type(node).__name__!r}")


class FrameSymbolVisitor(NodeVisitor):
    """A visitor for `Frame.inspect`."""

    def __init__(self, symbols: "Symbols") -> None:
        self.symbols = symbols

    def visit_Name(
        self, node: nodes.Name, store_as_param: bool = False, **kwargs: t.Any
    ) -> None:
        """All assignments to names go through this function."""
        if store_as_param or node.ctx == "param":
            self.symbols.declare_parameter(node.name)
        elif node.ctx == "store":
            self.symbols.store(node.name)
        elif node.ctx == "load":
            self.symbols.load(node.name)

    def visit_NSRef(self, node: nodes.NSRef, **kwargs: t.Any) -> None:
        self.symbols.load(node.name)

    def visit_If(self, node: nodes.If, **kwargs: t.Any) -> None:
        self.visit(node.test, **kwargs)
        original_symbols = self.symbols

        def inner_visit(nodes: t.Iterable[nodes.Node]) -> "Symbols":
            self.symbols = rv = original_symbols.copy()

            for subnode in nodes:
                self.visit(subnode, **kwargs)

            self.symbols = original_symbols
            return rv

        body_symbols = inner_visit(node.body)
        elif_symbols = inner_visit(node.elif_)
        else_symbols = inner_visit(node.else_ or ())
        self.symbols.branch_update([body_symbols, elif_symbols, else_symbols])

    def visit_Macro(self, node: nodes.Macro, **kwargs: t.Any) -> None:
        self.symbols.store(node.name)

    def visit_Import(self, node: nodes.Import, **kwargs: t.Any) -> None:
        self.generic_visit(node, **kwargs)
        self.symbols.store(node.target)

    def visit_FromImport(self, node: nodes.FromImport, **kwargs: t.Any) -> None:
        self.generic_visit(node, **kwargs)

        for name in node.names:
            if isinstance(name, tuple):
                self.symbols.store(name[1])
            else:
                self.symbols.store(name)

    def visit_Assign(self, node: nodes.Assign, **kwargs: t.Any) -> None:
        """Visit assignments in the correct order."""
        self.visit(node.node, **kwargs)
        self.visit(node.target, **kwargs)

    def visit_For(self, node: nodes.For, **kwargs: t.Any) -> None:
        """Visiting stops at for blocks.  However the block sequence
        is visited as part of the outer scope.
        """
        self.visit(node.iter, **kwargs)

    def visit_CallBlock(self, node: nodes.CallBlock, **kwargs: t.Any) -> None:
        self.visit(node.call, **kwargs)

    def visit_FilterBlock(self, node: nodes.FilterBlock, **kwargs: t.Any) -> None:
        self.visit(node.filter, **kwargs)

    def visit_With(self, node: nodes.With, **kwargs: t.Any) -> None:
        for target in node.values:
            self.visit(target)

    def visit_AssignBlock(self, node: nodes.AssignBlock, **kwargs: t.Any) -> None:
        """Stop visiting at block assigns."""
        self.visit(node.target, **kwargs)

    def visit_Scope(self, node: nodes.Scope, **kwargs: t.Any) -> None:
        """Stop visiting at scopes."""

    def visit_Block(self, node: nodes.Block, **kwargs: t.Any) -> None:
        """Stop visiting at blocks."""

    def visit_OverlayScope(self, node: nodes.OverlayScope, **kwargs: t.Any) -> None:
        """Do not visit into overlay scopes."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\t5\convert_to_onnx.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import copy
import logging
import os

import torch
from benchmark_helper import (
    Precision,
    create_onnxruntime_session,
    prepare_environment,
    setup_logger,
)
from onnx.shape_inference import infer_shapes_path
from t5_helper import PRETRAINED_MT5_MODELS, PRETRAINED_T5_MODELS, T5Helper
from transformers import MT5Config, T5Config

logger = logging.getLogger("")


def parse_arguments():
    parser = argparse.ArgumentParser()

    pretrained_models = PRETRAINED_T5_MODELS + PRETRAINED_MT5_MODELS
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=False,
        default=PRETRAINED_T5_MODELS[0],
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(pretrained_models),
    )

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="t5",
        choices=["t5", "mt5"],
        help="Model type: either t5 (default) or mt5",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=os.path.join(".", "onnx_models"),
        help="Output directory",
    )

    parser.add_argument(
        "-o",
        "--optimize_onnx",
        required=False,
        action="store_true",
        help="Use optimizer.py to optimize onnx model",
    )
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=str,
        default=Precision.FLOAT32.value,
        choices=[Precision.FLOAT32.value, Precision.FLOAT16.value],
        help="Precision of model to run. fp32 for full precision, fp16 for half precision",
    )

    parser.add_argument("--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("-e", "--use_external_data_format", required=False, action="store_true")
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument(
        "-s",
        "--use_decoder_start_token",
        required=False,
        action="store_true",
        help="Use config.decoder_start_token_id. Otherwise, add an extra graph input for decoder_input_ids.",
    )
    parser.set_defaults(use_decoder_start_token=False)

    parser.add_argument(
        "-w",
        "--overwrite",
        required=False,
        action="store_true",
        help="overwrite existing ONNX model",
    )
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "--disable_auto_mixed_precision",
        required=False,
        action="store_true",
        help="do not use auto mixed precision conversion",
    )
    parser.set_defaults(disable_auto_mixed_precision=False)

    parser.add_argument(
        "--force_fp16_io",
        required=False,
        action="store_true",
        help="Force to convert all float inputs and outputs to fp16 when precision is fp16.",
    )
    parser.set_defaults(force_fp16_io=False)

    parser.add_argument(
        "--use_int64_inputs",
        required=False,
        action="store_true",
        help="Use int64 instead of int32 for input_ids, position_ids and attention_mask.",
    )
    parser.set_defaults(use_int64_inputs=False)

    parser.add_argument(
        "--state_dict_path",
        type=str,
        default="",
        help="filepath to load pre-trained model with custom state dictionary (e.g. pytorch_model.bin)",
    )

    parser.add_argument(
        "--encoder_decoder_init",
        required=False,
        action="store_true",
        help="Combine encoder and decoder kv cache initialization into one model. It is legacy format that will be deprecated.",
    )
    parser.set_defaults(encoder_decoder_init=False)

    args = parser.parse_args()

    return args


def export_onnx_models(
    model_name_or_path: str,
    cache_dir: str,
    output_dir: str,
    use_gpu: bool = False,
    use_external_data_format: bool = False,
    optimize_onnx: bool = False,
    precision: str = Precision.FLOAT32.value,
    verbose: bool = False,
    use_decoder_start_token: bool = False,
    overwrite: bool = False,
    disable_auto_mixed_precision: bool = False,
    use_int32_inputs: bool = True,
    model_type: str = "t5",
    state_dict_path: str = "",
    encoder_decoder_init: bool = False,
    force_fp16_io: bool = False,
    shape_infer_before_optimization: bool = False,
):
    assert precision in [Precision.FLOAT32.value, Precision.FLOAT16.value], (
        f"Invalid precision: {precision}. Use 'fp32' or 'fp16'."
    )
    device = torch.device("cuda:0" if use_gpu else "cpu")

    models = T5Helper.load_model(
        model_name_or_path,
        cache_dir,
        device,
        model_type,
        state_dict_path,
        encoder_decoder_init=encoder_decoder_init,
    )
    config: T5Config | MT5Config = models["decoder"].config

    if (not use_external_data_format) and (config.num_layers > 24):
        logger.info("Try use_external_data_format when model size > 2GB")

    output_paths = []
    for name, model in models.items():
        model.to(device)
        filename_suffix = "_" + name

        onnx_path = T5Helper.get_onnx_path(
            output_dir,
            model_name_or_path,
            suffix=filename_suffix,
            new_folder=False,
        )

        if overwrite or not os.path.exists(onnx_path):
            logger.info(f"Exporting ONNX model to {onnx_path}")
            # We have to clone model before exporting onnx, otherwise verify_onnx will report large difference.
            cloned_model = copy.deepcopy(model).to(device)
            T5Helper.export_onnx(
                cloned_model,
                device,
                onnx_path,
                verbose,
                use_external_data_format,
                use_decoder_input_ids=not use_decoder_start_token,
                use_int32_inputs=use_int32_inputs,
            )
        else:
            logger.info(f"Skip exporting: existed ONNX model {onnx_path}")

        # Optimize ONNX graph.
        # The precision shall be compared with string value. It is because the Precision enum loaded from local file
        # (like by transformers test in CI pipeline) are not same as Precision enum from package.
        if optimize_onnx or precision != Precision.FLOAT32.value:
            onnx_shape_path = None
            if shape_infer_before_optimization:
                onnx_shape_path = T5Helper.get_onnx_path(
                    output_dir,
                    model_name_or_path,
                    suffix=filename_suffix + "_shape",
                    new_folder=False,
                )
                infer_shapes_path(onnx_path, onnx_shape_path)

            output_path = T5Helper.get_onnx_path(
                output_dir,
                model_name_or_path,
                suffix=filename_suffix + "_" + str(precision),
                new_folder=False,
            )

            if overwrite or not os.path.exists(output_path):
                logger.info(f"Optimizing model to {output_path}")
                T5Helper.optimize_onnx(
                    onnx_shape_path or onnx_path,
                    output_path,
                    precision == Precision.FLOAT16.value,
                    config.num_heads,
                    config.hidden_size,
                    use_external_data_format,
                    auto_mixed_precision=not disable_auto_mixed_precision,
                    use_gpu=use_gpu,
                    force_fp16_io=force_fp16_io,
                )
            else:
                logger.info(f"Skip optimizing: existed ONNX model {output_path}")
        else:
            output_path = onnx_path

        ort_session = create_onnxruntime_session(
            output_path,
            use_gpu=use_gpu,
            verbose=verbose,
        )
        if ort_session is None:
            break

        with torch.no_grad():
            max_diff = T5Helper.verify_onnx(model, ort_session, device, use_int32_inputs)
        logger.info(f"PyTorch and OnnxRuntime results max difference = {max_diff}")

        # The threshold cannot apply to fp16 model, which need a larger threshold.
        if precision == Precision.FLOAT32.value and max_diff > 1e-4:
            logger.warning("PyTorch and OnnxRuntime results are NOT close")

        output_paths.append(output_path)

    return output_paths


def main():
    args = parse_arguments()

    setup_logger(args.verbose)

    logger.info(f"Arguments:{args}")

    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".onnx") else os.path.dirname(args.output)
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    if args.precision != Precision.FLOAT32.value:
        assert args.optimize_onnx, "fp16/int8 requires --optimize_onnx"

    if args.precision == Precision.FLOAT16.value:
        assert args.use_gpu, "fp16 requires --use_gpu"

    output_paths = export_onnx_models(
        args.model_name_or_path,
        cache_dir,
        output_dir,
        args.use_gpu,
        args.use_external_data_format,
        args.optimize_onnx,
        args.precision,
        args.verbose,
        args.use_decoder_start_token,
        args.overwrite,
        args.disable_auto_mixed_precision,
        not args.use_int64_inputs,
        args.model_type,
        encoder_decoder_init=args.encoder_decoder_init,
        force_fp16_io=args.force_fp16_io,
    )

    logger.info(f"Done! Outputs: {output_paths}")


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\panel.py ===
from typing import TYPE_CHECKING, Optional

from .align import AlignMethod
from .box import ROUNDED, Box
from .cells import cell_len
from .jupyter import JupyterMixin
from .measure import Measurement, measure_renderables
from .padding import Padding, PaddingDimensions
from .segment import Segment
from .style import Style, StyleType
from .text import Text, TextType

if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderableType, RenderResult


class Panel(JupyterMixin):
    """A console renderable that draws a border around its contents.

    Example:
        >>> console.print(Panel("Hello, World!"))

    Args:
        renderable (RenderableType): A console renderable object.
        box (Box): A Box instance that defines the look of the border (see :ref:`appendix_box`. Defaults to box.ROUNDED.
        title (Optional[TextType], optional): Optional title displayed in panel header. Defaults to None.
        title_align (AlignMethod, optional): Alignment of title. Defaults to "center".
        subtitle (Optional[TextType], optional): Optional subtitle displayed in panel footer. Defaults to None.
        subtitle_align (AlignMethod, optional): Alignment of subtitle. Defaults to "center".
        safe_box (bool, optional): Disable box characters that don't display on windows legacy terminal with *raster* fonts. Defaults to True.
        expand (bool, optional): If True the panel will stretch to fill the console width, otherwise it will be sized to fit the contents. Defaults to True.
        style (str, optional): The style of the panel (border and contents). Defaults to "none".
        border_style (str, optional): The style of the border. Defaults to "none".
        width (Optional[int], optional): Optional width of panel. Defaults to None to auto-detect.
        height (Optional[int], optional): Optional height of panel. Defaults to None to auto-detect.
        padding (Optional[PaddingDimensions]): Optional padding around renderable. Defaults to 0.
        highlight (bool, optional): Enable automatic highlighting of panel title (if str). Defaults to False.
    """

    def __init__(
        self,
        renderable: "RenderableType",
        box: Box = ROUNDED,
        *,
        title: Optional[TextType] = None,
        title_align: AlignMethod = "center",
        subtitle: Optional[TextType] = None,
        subtitle_align: AlignMethod = "center",
        safe_box: Optional[bool] = None,
        expand: bool = True,
        style: StyleType = "none",
        border_style: StyleType = "none",
        width: Optional[int] = None,
        height: Optional[int] = None,
        padding: PaddingDimensions = (0, 1),
        highlight: bool = False,
    ) -> None:
        self.renderable = renderable
        self.box = box
        self.title = title
        self.title_align: AlignMethod = title_align
        self.subtitle = subtitle
        self.subtitle_align = subtitle_align
        self.safe_box = safe_box
        self.expand = expand
        self.style = style
        self.border_style = border_style
        self.width = width
        self.height = height
        self.padding = padding
        self.highlight = highlight

    @classmethod
    def fit(
        cls,
        renderable: "RenderableType",
        box: Box = ROUNDED,
        *,
        title: Optional[TextType] = None,
        title_align: AlignMethod = "center",
        subtitle: Optional[TextType] = None,
        subtitle_align: AlignMethod = "center",
        safe_box: Optional[bool] = None,
        style: StyleType = "none",
        border_style: StyleType = "none",
        width: Optional[int] = None,
        height: Optional[int] = None,
        padding: PaddingDimensions = (0, 1),
        highlight: bool = False,
    ) -> "Panel":
        """An alternative constructor that sets expand=False."""
        return cls(
            renderable,
            box,
            title=title,
            title_align=title_align,
            subtitle=subtitle,
            subtitle_align=subtitle_align,
            safe_box=safe_box,
            style=style,
            border_style=border_style,
            width=width,
            height=height,
            padding=padding,
            highlight=highlight,
            expand=False,
        )

    @property
    def _title(self) -> Optional[Text]:
        if self.title:
            title_text = (
                Text.from_markup(self.title)
                if isinstance(self.title, str)
                else self.title.copy()
            )
            title_text.end = ""
            title_text.plain = title_text.plain.replace("\n", " ")
            title_text.no_wrap = True
            title_text.expand_tabs()
            title_text.pad(1)
            return title_text
        return None

    @property
    def _subtitle(self) -> Optional[Text]:
        if self.subtitle:
            subtitle_text = (
                Text.from_markup(self.subtitle)
                if isinstance(self.subtitle, str)
                else self.subtitle.copy()
            )
            subtitle_text.end = ""
            subtitle_text.plain = subtitle_text.plain.replace("\n", " ")
            subtitle_text.no_wrap = True
            subtitle_text.expand_tabs()
            subtitle_text.pad(1)
            return subtitle_text
        return None

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        _padding = Padding.unpack(self.padding)
        renderable = (
            Padding(self.renderable, _padding) if any(_padding) else self.renderable
        )
        style = console.get_style(self.style)
        partial_border_style = console.get_style(self.border_style)
        border_style = style + partial_border_style
        width = (
            options.max_width
            if self.width is None
            else min(options.max_width, self.width)
        )

        safe_box: bool = console.safe_box if self.safe_box is None else self.safe_box
        box = self.box.substitute(options, safe=safe_box)

        def align_text(
            text: Text, width: int, align: str, character: str, style: Style
        ) -> Text:
            """Gets new aligned text.

            Args:
                text (Text): Title or subtitle text.
                width (int): Desired width.
                align (str): Alignment.
                character (str): Character for alignment.
                style (Style): Border style

            Returns:
                Text: New text instance
            """
            text = text.copy()
            text.truncate(width)
            excess_space = width - cell_len(text.plain)
            if text.style:
                text.stylize(console.get_style(text.style))

            if excess_space:
                if align == "left":
                    return Text.assemble(
                        text,
                        (character * excess_space, style),
                        no_wrap=True,
                        end="",
                    )
                elif align == "center":
                    left = excess_space // 2
                    return Text.assemble(
                        (character * left, style),
                        text,
                        (character * (excess_space - left), style),
                        no_wrap=True,
                        end="",
                    )
                else:
                    return Text.assemble(
                        (character * excess_space, style),
                        text,
                        no_wrap=True,
                        end="",
                    )
            return text

        title_text = self._title
        if title_text is not None:
            title_text.stylize_before(partial_border_style)

        child_width = (
            width - 2
            if self.expand
            else console.measure(
                renderable, options=options.update_width(width - 2)
            ).maximum
        )
        child_height = self.height or options.height or None
        if child_height:
            child_height -= 2
        if title_text is not None:
            child_width = min(
                options.max_width - 2, max(child_width, title_text.cell_len + 2)
            )

        width = child_width + 2
        child_options = options.update(
            width=child_width, height=child_height, highlight=self.highlight
        )
        lines = console.render_lines(renderable, child_options, style=style)

        line_start = Segment(box.mid_left, border_style)
        line_end = Segment(f"{box.mid_right}", border_style)
        new_line = Segment.line()
        if title_text is None or width <= 4:
            yield Segment(box.get_top([width - 2]), border_style)
        else:
            title_text = align_text(
                title_text,
                width - 4,
                self.title_align,
                box.top,
                border_style,
            )
            yield Segment(box.top_left + box.top, border_style)
            yield from console.render(title_text, child_options.update_width(width - 4))
            yield Segment(box.top + box.top_right, border_style)

        yield new_line
        for line in lines:
            yield line_start
            yield from line
            yield line_end
            yield new_line

        subtitle_text = self._subtitle
        if subtitle_text is not None:
            subtitle_text.stylize_before(partial_border_style)

        if subtitle_text is None or width <= 4:
            yield Segment(box.get_bottom([width - 2]), border_style)
        else:
            subtitle_text = align_text(
                subtitle_text,
                width - 4,
                self.subtitle_align,
                box.bottom,
                border_style,
            )
            yield Segment(box.bottom_left + box.bottom, border_style)
            yield from console.render(
                subtitle_text, child_options.update_width(width - 4)
            )
            yield Segment(box.bottom + box.bottom_right, border_style)

        yield new_line

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "Measurement":
        _title = self._title
        _, right, _, left = Padding.unpack(self.padding)
        padding = left + right
        renderables = [self.renderable, _title] if _title else [self.renderable]

        if self.width is None:
            width = (
                measure_renderables(
                    console,
                    options.update_width(options.max_width - padding - 2),
                    renderables,
                ).maximum
                + padding
                + 2
            )
        else:
            width = self.width
        return Measurement(width, width)


if __name__ == "__main__":  # pragma: no cover
    from .console import Console

    c = Console()

    from .box import DOUBLE, ROUNDED
    from .padding import Padding

    p = Panel(
        "Hello, World!",
        title="rich.Panel",
        style="white on blue",
        box=DOUBLE,
        padding=1,
    )

    c.print()
    c.print(p)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\jointsmethod.py ===
from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
                                    RigidBody, Particle)
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.method import _Methods
from sympy import Matrix
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['JointsMethod']


class JointsMethod(_Methods):
    """Method for formulating the equations of motion using a set of interconnected bodies with joints.

    .. deprecated:: 1.13
        The JointsMethod class is deprecated. Its functionality has been
        replaced by the new :class:`~.System` class.

    Parameters
    ==========

    newtonion : Body or ReferenceFrame
        The newtonion(inertial) frame.
    *joints : Joint
        The joints in the system

    Attributes
    ==========

    q, u : iterable
        Iterable of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Body objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    mass_matrix : Matrix, shape(n, n)
        The system's mass matrix
    forcing : Matrix, shape(n, 1)
        The system's forcing vector
    mass_matrix_full : Matrix, shape(2*n, 2*n)
        The "mass matrix" for the u's and q's
    forcing_full : Matrix, shape(2*n, 1)
        The "forcing vector" for the u's and q's
    method : KanesMethod or Lagrange's method
        Method's object.
    kdes : iterable
        Iterable of kde in they system.

    Examples
    ========

    As Body and JointsMethod have been deprecated, the following examples are
    for illustrative purposes only. The functionality of Body is fully captured
    by :class:`~.RigidBody` and :class:`~.Particle` and the functionality of
    JointsMethod is fully captured by :class:`~.System`. To ignore the
    deprecation warning we can use the ignore_warnings context manager.

    >>> from sympy.utilities.exceptions import ignore_warnings

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Body, JointsMethod, PrismaticJoint
    >>> from sympy.physics.vector import dynamicsymbols
    >>> c, k = symbols('c k')
    >>> x, v = dynamicsymbols('x v')
    >>> with ignore_warnings(DeprecationWarning):
    ...     wall = Body('W')
    ...     body = Body('B')
    >>> J = PrismaticJoint('J', wall, body, coordinates=x, speeds=v)
    >>> wall.apply_force(c*v*wall.x, reaction_body=body)
    >>> wall.apply_force(k*x*wall.x, reaction_body=body)
    >>> with ignore_warnings(DeprecationWarning):
    ...     method = JointsMethod(wall, J)
    >>> method.form_eoms()
    Matrix([[-B_mass*Derivative(v(t), t) - c*v(t) - k*x(t)]])
    >>> M = method.mass_matrix_full
    >>> F = method.forcing_full
    >>> rhs = M.LUsolve(F)
    >>> rhs
    Matrix([
    [                     v(t)],
    [(-c*v(t) - k*x(t))/B_mass]])

    Notes
    =====

    ``JointsMethod`` currently only works with systems that do not have any
    configuration or motion constraints.

    """

    def __init__(self, newtonion, *joints):
        sympy_deprecation_warning(
            """
            The JointsMethod class is deprecated.
            Its functionality has been replaced by the new System class.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-mechanics-jointsmethod"
        )
        if isinstance(newtonion, BodyBase):
            self.frame = newtonion.frame
        else:
            self.frame = newtonion

        self._joints = joints
        self._bodies = self._generate_bodylist()
        self._loads = self._generate_loadlist()
        self._q = self._generate_q()
        self._u = self._generate_u()
        self._kdes = self._generate_kdes()

        self._method = None

    @property
    def bodies(self):
        """List of bodies in they system."""
        return self._bodies

    @property
    def loads(self):
        """List of loads on the system."""
        return self._loads

    @property
    def q(self):
        """List of the generalized coordinates."""
        return self._q

    @property
    def u(self):
        """List of the generalized speeds."""
        return self._u

    @property
    def kdes(self):
        """List of the generalized coordinates."""
        return self._kdes

    @property
    def forcing_full(self):
        """The "forcing vector" for the u's and q's."""
        return self.method.forcing_full

    @property
    def mass_matrix_full(self):
        """The "mass matrix" for the u's and q's."""
        return self.method.mass_matrix_full

    @property
    def mass_matrix(self):
        """The system's mass matrix."""
        return self.method.mass_matrix

    @property
    def forcing(self):
        """The system's forcing vector."""
        return self.method.forcing

    @property
    def method(self):
        """Object of method used to form equations of systems."""
        return self._method

    def _generate_bodylist(self):
        bodies = []
        for joint in self._joints:
            if joint.child not in bodies:
                bodies.append(joint.child)
            if joint.parent not in bodies:
                bodies.append(joint.parent)
        return bodies

    def _generate_loadlist(self):
        load_list = []
        for body in self.bodies:
            if isinstance(body, Body):
                load_list.extend(body.loads)
        return load_list

    def _generate_q(self):
        q_ind = []
        for joint in self._joints:
            for coordinate in joint.coordinates:
                if coordinate in q_ind:
                    raise ValueError('Coordinates of joints should be unique.')
                q_ind.append(coordinate)
        return Matrix(q_ind)

    def _generate_u(self):
        u_ind = []
        for joint in self._joints:
            for speed in joint.speeds:
                if speed in u_ind:
                    raise ValueError('Speeds of joints should be unique.')
                u_ind.append(speed)
        return Matrix(u_ind)

    def _generate_kdes(self):
        kd_ind = Matrix(1, 0, []).T
        for joint in self._joints:
            kd_ind = kd_ind.col_join(joint.kdes)
        return kd_ind

    def _convert_bodies(self):
        # Convert `Body` to `Particle` and `RigidBody`
        bodylist = []
        for body in self.bodies:
            if not isinstance(body, Body):
                bodylist.append(body)
                continue
            if body.is_rigidbody:
                rb = RigidBody(body.name, body.masscenter, body.frame, body.mass,
                    (body.central_inertia, body.masscenter))
                rb.potential_energy = body.potential_energy
                bodylist.append(rb)
            else:
                part = Particle(body.name, body.masscenter, body.mass)
                part.potential_energy = body.potential_energy
                bodylist.append(part)
        return bodylist

    def form_eoms(self, method=KanesMethod):
        """Method to form system's equation of motions.

        Parameters
        ==========

        method : Class
            Class name of method.

        Returns
        ========

        Matrix
            Vector of equations of motions.

        Examples
        ========

        As Body and JointsMethod have been deprecated, the following examples
        are for illustrative purposes only. The functionality of Body is fully
        captured by :class:`~.RigidBody` and :class:`~.Particle` and the
        functionality of JointsMethod is fully captured by :class:`~.System`. To
        ignore the deprecation warning we can use the ignore_warnings context
        manager.

        >>> from sympy.utilities.exceptions import ignore_warnings

        This is a simple example for a one degree of freedom translational
        spring-mass-damper.

        >>> from sympy import S, symbols
        >>> from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols, Body
        >>> from sympy.physics.mechanics import PrismaticJoint, JointsMethod
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> with ignore_warnings(DeprecationWarning):
        ...     wall = Body('W')
        ...     part = Body('P', mass=m)
        >>> part.potential_energy = k * q**2 / S(2)
        >>> J = PrismaticJoint('J', wall, part, coordinates=q, speeds=qd)
        >>> wall.apply_force(b * qd * wall.x, reaction_body=part)
        >>> with ignore_warnings(DeprecationWarning):
        ...     method = JointsMethod(wall, J)
        >>> method.form_eoms(LagrangesMethod)
        Matrix([[b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])

        We can also solve for the states using the 'rhs' method.

        >>> method.rhs()
        Matrix([
        [                Derivative(q(t), t)],
        [(-b*Derivative(q(t), t) - k*q(t))/m]])

        """

        bodylist = self._convert_bodies()
        if issubclass(method, LagrangesMethod): #LagrangesMethod or similar
            L = Lagrangian(self.frame, *bodylist)
            self._method = method(L, self.q, self.loads, bodylist, self.frame)
        else: #KanesMethod or similar
            self._method = method(self.frame, q_ind=self.q, u_ind=self.u, kd_eqs=self.kdes,
                                    forcelist=self.loads, bodies=bodylist)
        soln = self.method._form_eoms()
        return soln

    def rhs(self, inv_method=None):
        """Returns equations that can be solved numerically.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        Returns
        ========

        Matrix
            Numerically solvable equations.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's rhs function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's rhs function.

        """

        return self.method.rhs(inv_method=inv_method)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\backends\matplotlibbackend\matplotlib.py ===
from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.external import import_module
import sympy.plotting.backends.base_backend as base_backend
from sympy.printing.latex import latex


# N.B.
# When changing the minimum module version for matplotlib, please change
# the same in the `SymPyDocTestFinder`` in `sympy/testing/runtests.py`


def _str_or_latex(label):
    if isinstance(label, Basic):
        return latex(label, mode='inline')
    return str(label)


def _matplotlib_list(interval_list):
    """
    Returns lists for matplotlib ``fill`` command from a list of bounding
    rectangular intervals
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend([intervalx.start, intervalx.start,
                          intervalx.end, intervalx.end, None])
            ylist.extend([intervaly.start, intervaly.end,
                          intervaly.end, intervaly.start, None])
    else:
        #XXX Ugly hack. Matplotlib does not accept empty lists for ``fill``
        xlist.extend((None, None, None, None))
        ylist.extend((None, None, None, None))
    return xlist, ylist


# Don't have to check for the success of importing matplotlib in each case;
# we will only be using this backend if we can successfully import matploblib
class MatplotlibBackend(base_backend.Plot):
    """ This class implements the functionalities to use Matplotlib with SymPy
    plotting functions.
    """

    def __init__(self, *series, **kwargs):
        super().__init__(*series, **kwargs)
        self.matplotlib = import_module('matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']},
            min_module_version='1.1.0', catch=(RuntimeError,))
        self.plt = self.matplotlib.pyplot
        self.cm = self.matplotlib.cm
        self.LineCollection = self.matplotlib.collections.LineCollection
        self.aspect = kwargs.get('aspect_ratio', 'auto')
        if self.aspect != 'auto':
            self.aspect = float(self.aspect[1]) / self.aspect[0]
        # PlotGrid can provide its figure and axes to be populated with
        # the data from the series.
        self._plotgrid_fig = kwargs.pop("fig", None)
        self._plotgrid_ax = kwargs.pop("ax", None)

    def _create_figure(self):
        def set_spines(ax):
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        if self._plotgrid_fig is not None:
            self.fig = self._plotgrid_fig
            self.ax = self._plotgrid_ax
            if not any(s.is_3D for s in self._series):
                set_spines(self.ax)
        else:
            self.fig = self.plt.figure(figsize=self.size)
            if any(s.is_3D for s in self._series):
                self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
            else:
                self.ax = self.fig.add_subplot(1, 1, 1)
                set_spines(self.ax)

    @staticmethod
    def get_segments(x, y, z=None):
        """ Convert two list of coordinates to a list of segments to be used
        with Matplotlib's :external:class:`~matplotlib.collections.LineCollection`.

        Parameters
        ==========
            x : list
                List of x-coordinates

            y : list
                List of y-coordinates

            z : list
                List of z-coordinates for a 3D line.
        """
        np = import_module('numpy')
        if z is not None:
            dim = 3
            points = (x, y, z)
        else:
            dim = 2
            points = (x, y)
        points = np.ma.array(points).T.reshape(-1, 1, dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _process_series(self, series, ax):
        np = import_module('numpy')
        mpl_toolkits = import_module(
            'mpl_toolkits', import_kwargs={'fromlist': ['mplot3d']})

        # XXX Workaround for matplotlib issue
        # https://github.com/matplotlib/matplotlib/issues/17130
        xlims, ylims, zlims = [], [], []

        for s in series:
            # Create the collections
            if s.is_2Dline:
                if s.is_parametric:
                    x, y, param = s.get_data()
                else:
                    x, y = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    segments = self.get_segments(x, y)
                    collection = self.LineCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    lbl = _str_or_latex(s.label)
                    line, = ax.plot(x, y, label=lbl, color=s.line_color)
            elif s.is_contour:
                ax.contour(*s.get_data())
            elif s.is_3Dline:
                x, y, z, param = s.get_data()
                if (isinstance(s.line_color, (int, float)) or
                        callable(s.line_color)):
                    art3d = mpl_toolkits.mplot3d.art3d
                    segments = self.get_segments(x, y, z)
                    collection = art3d.Line3DCollection(segments)
                    collection.set_array(s.get_color_array())
                    ax.add_collection(collection)
                else:
                    lbl = _str_or_latex(s.label)
                    ax.plot(x, y, z, label=lbl, color=s.line_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_3Dsurface:
                if s.is_parametric:
                    x, y, z, u, v = s.get_data()
                else:
                    x, y, z = s.get_data()
                collection = ax.plot_surface(x, y, z,
                    cmap=getattr(self.cm, 'viridis', self.cm.jet),
                    rstride=1, cstride=1, linewidth=0.1)
                if isinstance(s.surface_color, (float, int, Callable)):
                    color_array = s.get_color_array()
                    color_array = color_array.reshape(color_array.size)
                    collection.set_array(color_array)
                else:
                    collection.set_color(s.surface_color)

                xlims.append(s._xlim)
                ylims.append(s._ylim)
                zlims.append(s._zlim)
            elif s.is_implicit:
                points = s.get_data()
                if len(points) == 2:
                    # interval math plotting
                    x, y = _matplotlib_list(points[0])
                    ax.fill(x, y, facecolor=s.line_color, edgecolor='None')
                else:
                    # use contourf or contour depending on whether it is
                    # an inequality or equality.
                    # XXX: ``contour`` plots multiple lines. Should be fixed.
                    ListedColormap = self.matplotlib.colors.ListedColormap
                    colormap = ListedColormap(["white", s.line_color])
                    xarray, yarray, zarray, plot_type = points
                    if plot_type == 'contour':
                        ax.contour(xarray, yarray, zarray, cmap=colormap)
                    else:
                        ax.contourf(xarray, yarray, zarray, cmap=colormap)
            elif s.is_generic:
                if s.type == "markers":
                    # s.rendering_kw["color"] = s.line_color
                    ax.plot(*s.args, **s.rendering_kw)
                elif s.type == "annotations":
                    ax.annotate(*s.args, **s.rendering_kw)
                elif s.type == "fill":
                    # s.rendering_kw["color"] = s.line_color
                    ax.fill_between(*s.args, **s.rendering_kw)
                elif s.type == "rectangles":
                    # s.rendering_kw["color"] = s.line_color
                    ax.add_patch(
                        self.matplotlib.patches.Rectangle(
                            *s.args, **s.rendering_kw))
            else:
                raise NotImplementedError(
                    '{} is not supported in the SymPy plotting module '
                    'with matplotlib backend. Please report this issue.'
                    .format(ax))

        Axes3D = mpl_toolkits.mplot3d.Axes3D
        if not isinstance(ax, Axes3D):
            ax.autoscale_view(
                scalex=ax.get_autoscalex_on(),
                scaley=ax.get_autoscaley_on())
        else:
            # XXX Workaround for matplotlib issue
            # https://github.com/matplotlib/matplotlib/issues/17130
            if xlims:
                xlims = np.array(xlims)
                xlim = (np.amin(xlims[:, 0]), np.amax(xlims[:, 1]))
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([0, 1])

            if ylims:
                ylims = np.array(ylims)
                ylim = (np.amin(ylims[:, 0]), np.amax(ylims[:, 1]))
                ax.set_ylim(ylim)
            else:
                ax.set_ylim([0, 1])

            if zlims:
                zlims = np.array(zlims)
                zlim = (np.amin(zlims[:, 0]), np.amax(zlims[:, 1]))
                ax.set_zlim(zlim)
            else:
                ax.set_zlim([0, 1])

        # Set global options.
        # TODO The 3D stuff
        # XXX The order of those is important.
        if self.xscale and not isinstance(ax, Axes3D):
            ax.set_xscale(self.xscale)
        if self.yscale and not isinstance(ax, Axes3D):
            ax.set_yscale(self.yscale)
        if not isinstance(ax, Axes3D) or self.matplotlib.__version__ >= '1.2.0':  # XXX in the distant future remove this check
            ax.set_autoscale_on(self.autoscale)
        if self.axis_center:
            val = self.axis_center
            if isinstance(ax, Axes3D):
                pass
            elif val == 'center':
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
            elif val == 'auto':
                xl, xh = ax.get_xlim()
                yl, yh = ax.get_ylim()
                pos_left = ('data', 0) if xl*xh <= 0 else 'center'
                pos_bottom = ('data', 0) if yl*yh <= 0 else 'center'
                ax.spines['left'].set_position(pos_left)
                ax.spines['bottom'].set_position(pos_bottom)
            else:
                ax.spines['left'].set_position(('data', val[0]))
                ax.spines['bottom'].set_position(('data', val[1]))
        if not self.axis:
            ax.set_axis_off()
        if self.legend:
            if ax.legend():
                ax.legend_.set_visible(self.legend)
        if self.margin:
            ax.set_xmargin(self.margin)
            ax.set_ymargin(self.margin)
        if self.title:
            ax.set_title(self.title)
        if self.xlabel:
            xlbl = _str_or_latex(self.xlabel)
            ax.set_xlabel(xlbl, position=(1, 0))
        if self.ylabel:
            ylbl = _str_or_latex(self.ylabel)
            ax.set_ylabel(ylbl, position=(0, 1))
        if isinstance(ax, Axes3D) and self.zlabel:
            zlbl = _str_or_latex(self.zlabel)
            ax.set_zlabel(zlbl, position=(0, 1))

        # xlim and ylim should always be set at last so that plot limits
        # doesn't get altered during the process.
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)
        self.ax.set_aspect(self.aspect)


    def process_series(self):
        """
        Iterates over every ``Plot`` object and further calls
        _process_series()
        """
        self._create_figure()
        self._process_series(self._series, self.ax)

    def show(self):
        self.process_series()
        #TODO after fixing https://github.com/ipython/ipython/issues/1255
        # you can uncomment the next line and remove the pyplot.show() call
        #self.fig.show()
        if base_backend._show:
            self.fig.tight_layout()
            self.plt.show()
        else:
            self.close()

    def save(self, path):
        self.process_series()
        self.fig.savefig(path)

    def close(self):
        self.plt.close(self.fig)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\Node.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Node(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Node()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsNode(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def NodeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Node
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Node
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def Domain(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def SinceVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Node
    def Index(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Node
    def OpType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Node
    def ExecutionProviderType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Node
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

    # Node
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Node
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    # Node
    def Attributes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Attribute import Attribute
            obj = Attribute()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Node
    def AttributesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def AttributesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    # Node
    def InputArgCounts(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Node
    def InputArgCountsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Node
    def InputArgCountsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def InputArgCountsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

    # Node
    def ImplicitInputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Node
    def ImplicitInputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def ImplicitInputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        return o == 0

def NodeStart(builder):
    builder.StartObject(13)

def Start(builder):
    NodeStart(builder)

def NodeAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    NodeAddName(builder, name)

def NodeAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    NodeAddDocString(builder, docString)

def NodeAddDomain(builder, domain):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(domain), 0)

def AddDomain(builder, domain):
    NodeAddDomain(builder, domain)

def NodeAddSinceVersion(builder, sinceVersion):
    builder.PrependInt32Slot(3, sinceVersion, 0)

def AddSinceVersion(builder, sinceVersion):
    NodeAddSinceVersion(builder, sinceVersion)

def NodeAddIndex(builder, index):
    builder.PrependUint32Slot(4, index, 0)

def AddIndex(builder, index):
    NodeAddIndex(builder, index)

def NodeAddOpType(builder, opType):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(opType), 0)

def AddOpType(builder, opType):
    NodeAddOpType(builder, opType)

def NodeAddType(builder, type):
    builder.PrependInt32Slot(6, type, 0)

def AddType(builder, type):
    NodeAddType(builder, type)

def NodeAddExecutionProviderType(builder, executionProviderType):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(executionProviderType), 0)

def AddExecutionProviderType(builder, executionProviderType):
    NodeAddExecutionProviderType(builder, executionProviderType)

def NodeAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)

def AddInputs(builder, inputs):
    NodeAddInputs(builder, inputs)

def NodeStartInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputsVector(builder, numElems: int) -> int:
    return NodeStartInputsVector(builder, numElems)

def NodeAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)

def AddOutputs(builder, outputs):
    NodeAddOutputs(builder, outputs)

def NodeStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOutputsVector(builder, numElems: int) -> int:
    return NodeStartOutputsVector(builder, numElems)

def NodeAddAttributes(builder, attributes):
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(attributes), 0)

def AddAttributes(builder, attributes):
    NodeAddAttributes(builder, attributes)

def NodeStartAttributesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartAttributesVector(builder, numElems: int) -> int:
    return NodeStartAttributesVector(builder, numElems)

def NodeAddInputArgCounts(builder, inputArgCounts):
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(inputArgCounts), 0)

def AddInputArgCounts(builder, inputArgCounts):
    NodeAddInputArgCounts(builder, inputArgCounts)

def NodeStartInputArgCountsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputArgCountsVector(builder, numElems: int) -> int:
    return NodeStartInputArgCountsVector(builder, numElems)

def NodeAddImplicitInputs(builder, implicitInputs):
    builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(implicitInputs), 0)

def AddImplicitInputs(builder, implicitInputs):
    NodeAddImplicitInputs(builder, implicitInputs)

def NodeStartImplicitInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartImplicitInputsVector(builder, numElems: int) -> int:
    return NodeStartImplicitInputsVector(builder, numElems)

def NodeEnd(builder):
    return builder.EndObject()

def End(builder):
    return NodeEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_utils.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy
from numpy import array_equal, ndarray
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx import onnx_pb as onnx_proto
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionUtils:
    def __init__(self, model: OnnxModel):
        self.model: OnnxModel = model

    def cast_graph_input_to_int32(self, input_name: str) -> tuple[bool, str]:
        graph_input = self.model.find_graph_input(input_name)
        if graph_input is not None and graph_input.type.tensor_type.elem_type != TensorProto.INT32:
            cast_output, cast_node = self.cast_input_to_int32(input_name)
            logger.debug(f"Casted graph input {input_name} to int32")
            return True, cast_output

        logger.debug(f"Did not cast graph input {input_name} to int32: found {graph_input is not None}")
        return False, input_name

    def cast_input(self, input_name: str, target_type="int32"):
        output_name = input_name + "_" + target_type

        if target_type == "int32":
            to_type = int(TensorProto.INT32)
        elif target_type == "float32":
            to_type = int(TensorProto.FLOAT)
        elif target_type == "float16":
            to_type = int(TensorProto.FLOAT16)
        else:
            raise ValueError("Invalid target_type: {target_type}")

        cast_node = self.add_cast_node(input_name, to_type, output_name)

        return output_name, cast_node

    def add_cast_node(
        self,
        input_name: str,
        to_type: int,
        output_name: str | None = None,
        output_name_to_node=None,
        graph_name: str | None = None,
    ):
        if output_name is None:
            output_name = input_name + f"_cast_to_{to_type}"

        # Avoid consequent Cast nodes.
        inputs = [input_name]
        if output_name_to_node is None:
            output_name_to_node = self.model.output_name_to_node()
        if input_name in output_name_to_node:
            parent_node = output_name_to_node[input_name]
            if parent_node and parent_node.op_type == "Cast":
                inputs = [parent_node.input[0]]

        cast_node = helper.make_node("Cast", inputs=inputs, outputs=[output_name])

        cast_node.attribute.extend([helper.make_attribute("to", to_type)])
        self.model.add_node(cast_node, graph_name=graph_name)

        return cast_node

    def cast_input_to_int32(self, input_name: str):
        return self.cast_input(input_name, "int32")

    def remove_cast_int32(self, input_name: str):
        input_name_to_nodes = self.model.input_name_to_nodes()
        nodes = input_name_to_nodes[input_name]
        for node in nodes:
            if node.op_type == "Cast":
                is_int32 = False
                for att in node.attribute:
                    if att.name == "to" and att.i == int(TensorProto.INT32):
                        is_int32 = True
                        break
                if is_int32:
                    output_name = node.output[0]
                    self.model.remove_node(node)
                    self.model.replace_input_of_all_nodes(output_name, input_name)

    @staticmethod
    def update_node_input(node, i, new_input_name, input_name_to_nodes):
        old_input_reference = 0
        if (node.input[i] in input_name_to_nodes) and node in input_name_to_nodes[node.input[i]]:
            input_name_to_nodes[node.input[i]].remove(node)
            old_input_reference = len(input_name_to_nodes[node.input[i]])

        node.input[i] = new_input_name

        if new_input_name in input_name_to_nodes:
            input_name_to_nodes[new_input_name].append(node)
        else:
            input_name_to_nodes[new_input_name] = [node]

        return old_input_reference

    @staticmethod
    def skip_parent(model: OnnxModel, node, parent_node, input_name_to_nodes, node_input_index=0, parent_input_index=0):
        """
        Before:
              (input)-->parent-->node-->(output)
        After:
              (input)-->parent-->
                |
                +----->node-->(output)

        This function returns a flag whether the parent node can be removed.
        """

        old_input_name = node.input[node_input_index]
        new_input_name = parent_node.input[parent_input_index]
        old_input_reference = FusionUtils.update_node_input(node, node_input_index, new_input_name, input_name_to_nodes)

        # We can remove the first Transpose if its output is not used (linked to graph output or other nodes) anymore.
        parent_can_be_removed = (old_input_reference == 0) and not model.find_graph_output(old_input_name)

        return parent_can_be_removed

    def get_squeeze_or_unsqueeze_axes(self, node: NodeProto) -> ndarray | None:
        assert node.op_type in ["Squeeze", "Unsqueeze"]

        # For opset >= 13, axes is an input instead of an attribute.
        if len(node.input) > 1:
            return self.model.get_constant_value(node.input[1])

        axes = None
        for attr in node.attribute:
            if attr.name == "axes":
                axes = helper.get_attribute_value(attr)
        return axes

    @staticmethod
    def check_node_attribute(node, attribute_name: str, expected_value, default_value=None):
        """Verify that a node has expected value for an attribute.

        Args:
            node (NodeProto): a node to check
            attribute_name (str): name of attribute
            expected_value (Any): expected value of the attribute
            default_value (Any, optional): default value if the attribute does not exist. Defaults to None.

        Returns:
            bool: whether the check is passed or not
        """
        value = default_value
        for attr in node.attribute:
            if attr.name == attribute_name:
                value = helper.get_attribute_value(attr)

        if isinstance(expected_value, list):
            return (isinstance(value, (ndarray, list))) and array_equal(expected_value, value, equal_nan=False)
        else:
            return value == expected_value

    @staticmethod
    def transpose_2d_int8_tensor(tensor: onnx_proto.TensorProto):
        """Transpose a 2-D INT8 TensorProto
        Args:
            tensor (TensorProto): tensor to be transposed
        Returns:
            tensor (TensorProto): transposed tensor
        """
        if not isinstance(tensor, onnx_proto.TensorProto):
            raise ValueError(f"Expected input type is an ONNX TensorProto but got {type(tensor)}")

        if len(tensor.dims) != 2 or tensor.data_type != onnx_proto.TensorProto.INT8:
            raise ValueError("Only INT8 2-D tensors can be transposed")

        if tensor.raw_data:
            int32_data = numpy.reshape(numpy.frombuffer(tensor.raw_data, dtype="int8"), tensor.dims)
            int32_transposed_data = numpy.transpose(int32_data, [1, 0])
            tensor.raw_data = int32_transposed_data.tobytes()

        else:
            raise ValueError("only raw buffer supported")

        return tensor

    @staticmethod
    def check_qdq_node_for_fusion(node: NodeProto, model: OnnxModel, allow_per_tensor_quantization_only=True):
        """Verify if a provided QuantizeLinear (Q) / DequantizeLinear (DQ) node is a good candidate for fusion.
           It is a good candidate for fusion if:
           (1) The Q/DQ node is for per-tensor quantization if allow_per_tensor_quantization_only is `True`
           (2) The Q/DQ node should have constant scale
           (3) The Q/DQ node should have a zero point of 0
        Args:
            node (NodeProto): a Q/DQ node to check
        Returns:
            bool: whether the check is passed or not
        """
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            logger.debug(f"Provided node is not a Q/DQ node. Op Type: {node.op_type}")

        scale = model.get_constant_value(node.input[1])

        # Scale is not constant
        if scale is None:
            return False

        # Not per-tensor quantization
        scale_has_single_element = scale.ndim == 0 or (scale.ndim == 1 and scale.shape[0] == 1)
        if allow_per_tensor_quantization_only and not scale_has_single_element:
            return False

        # If the Q/DQ node has no zero point input, it is assumed to be 0 (per ONNX spec)
        if len(node.input) == 2:
            return True

        # Zero point should be constant and should have a value of 0
        zero_point = model.get_constant_value(node.input[2])

        # Zero point and scale should have same number of dims
        if scale.ndim != zero_point.ndim:
            return False

        # Zero point is not constant or zero point is not zero
        if zero_point is None:
            return False

        return numpy.all(zero_point == 0)

    def check_node_input_value(self, node, input_index: int, expected_value):
        """Verify that a node has expected input value

        Args:
            node (NodeProto): a node to check
            input_index (int): index of its input to be verified
            expected_value (Any): expected value of the input

        Returns:
            bool: whether the check is passed or not
        """
        assert len(node.input) > input_index

        value = self.model.get_constant_value(node.input[input_index])

        if isinstance(expected_value, list):
            return (isinstance(value, (ndarray, list))) and array_equal(expected_value, value, equal_nan=False)
        else:
            return value == expected_value

    def remove_identity_nodes(self):
        """Remove Identity nodes, except those right before graph output."""
        nodes_to_remove = []
        graph_output_names = self.model.get_graphs_output_names()
        for node in self.model.nodes():
            if node.op_type == "Identity":
                if node.output[0] not in graph_output_names:
                    self.model.replace_input_of_all_nodes(node.output[0], node.input[0])
                    nodes_to_remove.append(node)

        if nodes_to_remove:
            self.model.remove_nodes(nodes_to_remove)
            logger.info(f"Removed {len(nodes_to_remove)} Identity nodes")

    def remove_cascaded_cast_nodes(self):
        self.model.remove_cascaded_cast_nodes()

    def remove_useless_cast_nodes(self):
        self.model.remove_useless_cast_nodes()

    def remove_useless_reshape_nodes(self):
        """Remove reshape node that is not needed based on symbolic shape inference: input and output has same shape"""
        shape_infer = self.model.infer_runtime_shape(update=True)
        if shape_infer is None:
            return

        nodes_to_remove = []
        for node in self.model.nodes():
            if node.op_type == "Reshape":
                input_shape = shape_infer.get_edge_shape(node.input[0])
                output_shape = shape_infer.get_edge_shape(node.output[0])
                if input_shape and output_shape and input_shape == output_shape:
                    logger.info(
                        f"Remove reshape node {node.name} since its input shape is same as output: {input_shape}"
                    )
                    nodes_to_remove.append(node)

        if nodes_to_remove:
            graph_input_names = set(self.model.get_graphs_input_names())
            graph_output_names = set(self.model.get_graphs_output_names())
            for node in nodes_to_remove:
                if bool(set(node.output) & graph_output_names):
                    if (
                        not bool(set(node.input) & graph_input_names)
                        and len(self.model.input_name_to_nodes()[node.input[0]]) == 1  # parent has only one child
                    ):
                        self.model.replace_output_of_all_nodes(node.input[0], node.output[0])
                    else:
                        continue
                else:
                    self.model.replace_input_of_all_nodes(node.output[0], node.input[0])
                self.model.remove_node(node)


class NumpyHelper:
    @staticmethod
    def to_array(tensor: TensorProto, fill_zeros: bool = False) -> ndarray:
        # When weights are in external data format but not presented, we can still test the optimizer with two changes:
        # (1) set fill_zeros = True  (2) change load_external_data=False in optimizer.py
        if fill_zeros:
            return ndarray(
                shape=tensor.dims,
                dtype=helper.tensor_dtype_to_np_dtype(tensor.data_type),
            )

        return numpy_helper.to_array(tensor)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_parking_lot.py ===
# ParkingLot provides an abstraction for a fair waitqueue with cancellation
# and requeuing support. Inspiration:
#
#    https://webkit.org/blog/6161/locking-in-webkit/
#    https://amanieu.github.io/parking_lot/
#
# which were in turn heavily influenced by
#
#    http://gee.cs.oswego.edu/dl/papers/aqs.pdf
#
# Compared to these, our use of cooperative scheduling allows some
# simplifications (no need for internal locking). On the other hand, the need
# to support Trio's strong cancellation semantics adds some complications
# (tasks need to know where they're queued so they can cancel). Also, in the
# above work, the ParkingLot is a global structure that holds a collection of
# waitqueues keyed by lock address, and which are opportunistically allocated
# and destroyed as contention arises; this allows the worst-case memory usage
# for all waitqueues to be O(#tasks). Here we allocate a separate wait queue
# for each synchronization object, so we're O(#objects + #tasks). This isn't
# *so* bad since compared to our synchronization objects are heavier than
# theirs and our tasks are lighter, so for us #objects is smaller and #tasks
# is larger.
#
# This is in the core because for two reasons. First, it's used by
# UnboundedQueue, and UnboundedQueue is used for a number of things in the
# core. And second, it's responsible for providing fairness to all of our
# high-level synchronization primitives (locks, queues, etc.). For now with
# our FIFO scheduler this is relatively trivial (it's just a FIFO waitqueue),
# but in the future we ever start support task priorities or fair scheduling
#
#    https://github.com/python-trio/trio/issues/32
#
# then all we'll have to do is update this. (Well, full-fledged task
# priorities might also require priority inheritance, which would require more
# work.)
#
# For discussion of data structures to use here, see:
#
#     https://github.com/dabeaz/curio/issues/136
#
# (and also the articles above). Currently we use a SortedDict ordered by a
# global monotonic counter that ensures FIFO ordering. The main advantage of
# this is that it's easy to implement :-). An intrusive doubly-linked list
# would also be a natural approach, so long as we only handle FIFO ordering.
#
# XX: should we switch to the shared global ParkingLot approach?
#
# XX: we should probably add support for "parking tokens" to allow for
# task-fair RWlock (basically: when parking a task needs to be able to mark
# itself as a reader or a writer, and then a task-fair wakeup policy is, wake
# the next task, and if it's a reader than keep waking tasks so long as they
# are readers). Without this I think you can implement write-biased or
# read-biased RWlocks (by using two parking lots and drawing from whichever is
# preferred), but not task-fair -- and task-fair plays much more nicely with
# WFQ. (Consider what happens in the two-lot implementation if you're
# write-biased but all the pending writers are blocked at the scheduler level
# by the WFQ logic...)
# ...alternatively, "phase-fair" RWlocks are pretty interesting:
#    http://www.cs.unc.edu/~anderson/papers/ecrts09b.pdf
# Useful summary:
# https://docs.oracle.com/javase/7/docs/api/java/util/concurrent/locks/ReadWriteLock.html
#
# XX: if we do add WFQ, then we might have to drop the current feature where
# unpark returns the tasks that were unparked. Rationale: suppose that at the
# time we call unpark, the next task is deprioritized... and then, before it
# becomes runnable, a new task parks which *is* runnable. Ideally we should
# immediately wake the new task, and leave the old task on the queue for
# later. But this means we can't commit to which task we are unparking when
# unpark is called.
#
# See: https://github.com/python-trio/trio/issues/53
from __future__ import annotations

import inspect
import math
from collections import OrderedDict
from typing import TYPE_CHECKING

import attrs
import outcome

from .. import _core
from .._util import final

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._run import Task


GLOBAL_PARKING_LOT_BREAKER: dict[Task, list[ParkingLot]] = {}


def add_parking_lot_breaker(task: Task, lot: ParkingLot) -> None:
    """Register a task as a breaker for a lot. See :meth:`ParkingLot.break_lot`.

    raises:
      trio.BrokenResourceError: if the task has already exited.
    """
    if inspect.getcoroutinestate(task.coro) == inspect.CORO_CLOSED:
        raise _core._exceptions.BrokenResourceError(
            "Attempted to add already exited task as lot breaker.",
        )
    if task not in GLOBAL_PARKING_LOT_BREAKER:
        GLOBAL_PARKING_LOT_BREAKER[task] = [lot]
    else:
        GLOBAL_PARKING_LOT_BREAKER[task].append(lot)


def remove_parking_lot_breaker(task: Task, lot: ParkingLot) -> None:
    """Deregister a task as a breaker for a lot. See :meth:`ParkingLot.break_lot`"""
    try:
        GLOBAL_PARKING_LOT_BREAKER[task].remove(lot)
    except (KeyError, ValueError):
        raise RuntimeError(
            "Attempted to remove task as breaker for a lot it is not registered for",
        ) from None
    if not GLOBAL_PARKING_LOT_BREAKER[task]:
        del GLOBAL_PARKING_LOT_BREAKER[task]


@attrs.frozen
class ParkingLotStatistics:
    """An object containing debugging information for a ParkingLot.

    Currently, the following fields are defined:

    * ``tasks_waiting`` (int): The number of tasks blocked on this lot's
      :meth:`trio.lowlevel.ParkingLot.park` method.

    """

    tasks_waiting: int


@final
@attrs.define(eq=False)
class ParkingLot:
    """A fair wait queue with cancellation and requeuing.

    This class encapsulates the tricky parts of implementing a wait
    queue. It's useful for implementing higher-level synchronization
    primitives like queues and locks.

    In addition to the methods below, you can use ``len(parking_lot)`` to get
    the number of parked tasks, and ``if parking_lot: ...`` to check whether
    there are any parked tasks.

    """

    # {task: None}, we just want a deque where we can quickly delete random
    # items
    _parked: OrderedDict[Task, None] = attrs.field(factory=OrderedDict, init=False)
    broken_by: list[Task] = attrs.field(factory=list, init=False)

    def __len__(self) -> int:
        """Returns the number of parked tasks."""
        return len(self._parked)

    def __bool__(self) -> bool:
        """True if there are parked tasks, False otherwise."""
        return bool(self._parked)

    # XX this currently returns None
    # if we ever add the ability to repark while one's resuming place in
    # line (for false wakeups), then we could have it return a ticket that
    # abstracts the "place in line" concept.
    @_core.enable_ki_protection
    async def park(self) -> None:
        """Park the current task until woken by a call to :meth:`unpark` or
        :meth:`unpark_all`.

        Raises:
          BrokenResourceError: if attempting to park in a broken lot, or the lot
            breaks before we get to unpark.

        """
        if self.broken_by:
            raise _core.BrokenResourceError(
                f"Attempted to park in parking lot broken by {self.broken_by}",
            )
        task = _core.current_task()
        self._parked[task] = None
        task.custom_sleep_data = self

        def abort_fn(_: _core.RaiseCancelT) -> _core.Abort:
            del task.custom_sleep_data._parked[task]
            return _core.Abort.SUCCEEDED

        await _core.wait_task_rescheduled(abort_fn)

    def _pop_several(self, count: int | float) -> Iterator[Task]:  # noqa: PYI041
        if isinstance(count, float):
            if math.isinf(count):
                count = len(self._parked)
            else:
                raise ValueError("Cannot pop a non-integer number of tasks.")
        else:
            count = min(count, len(self._parked))
        for _ in range(count):
            task, _ = self._parked.popitem(last=False)
            yield task

    @_core.enable_ki_protection
    def unpark(self, *, count: int | float = 1) -> list[Task]:  # noqa: PYI041
        """Unpark one or more tasks.

        This wakes up ``count`` tasks that are blocked in :meth:`park`. If
        there are fewer than ``count`` tasks parked, then wakes as many tasks
        are available and then returns successfully.

        Args:
          count (int | math.inf): the number of tasks to unpark.

        """
        tasks = list(self._pop_several(count))
        for task in tasks:
            _core.reschedule(task)
        return tasks

    def unpark_all(self) -> list[Task]:
        """Unpark all parked tasks."""
        return self.unpark(count=len(self))

    @_core.enable_ki_protection
    def repark(
        self,
        new_lot: ParkingLot,
        *,
        count: int | float = 1,  # noqa: PYI041
    ) -> None:
        """Move parked tasks from one :class:`ParkingLot` object to another.

        This dequeues ``count`` tasks from one lot, and requeues them on
        another, preserving order. For example::

           async def parker(lot):
               print("sleeping")
               await lot.park()
               print("woken")

           async def main():
               lot1 = trio.lowlevel.ParkingLot()
               lot2 = trio.lowlevel.ParkingLot()
               async with trio.open_nursery() as nursery:
                   nursery.start_soon(parker, lot1)
                   await trio.testing.wait_all_tasks_blocked()
                   assert len(lot1) == 1
                   assert len(lot2) == 0
                   lot1.repark(lot2)
                   assert len(lot1) == 0
                   assert len(lot2) == 1
                   # This wakes up the task that was originally parked in lot1
                   lot2.unpark()

        If there are fewer than ``count`` tasks parked, then reparks as many
        tasks as are available and then returns successfully.

        Args:
          new_lot (ParkingLot): the parking lot to move tasks to.
          count (int|math.inf): the number of tasks to move.

        """
        if not isinstance(new_lot, ParkingLot):
            raise TypeError("new_lot must be a ParkingLot")
        for task in self._pop_several(count):
            new_lot._parked[task] = None
            task.custom_sleep_data = new_lot

    def repark_all(self, new_lot: ParkingLot) -> None:
        """Move all parked tasks from one :class:`ParkingLot` object to
        another.

        See :meth:`repark` for details.

        """
        return self.repark(new_lot, count=len(self))

    def break_lot(self, task: Task | None = None) -> None:
        """Break this lot, with ``task`` noted as the task that broke it.

        This causes all parked tasks to raise an error, and any
        future tasks attempting to park to error. Unpark & repark become no-ops as the
        parking lot is empty.

        The error raised contains a reference to the task sent as a parameter. The task
        is also saved in the parking lot in the ``broken_by`` attribute.
        """
        if task is None:
            task = _core.current_task()

        # if lot is already broken, just mark this as another breaker and return
        if self.broken_by:
            self.broken_by.append(task)
            return

        self.broken_by.append(task)

        for parked_task in self._parked:
            _core.reschedule(
                parked_task,
                outcome.Error(
                    _core.BrokenResourceError(f"Parking lot broken by {task}"),
                ),
            )
        self._parked.clear()

    def statistics(self) -> ParkingLotStatistics:
        """Return an object containing debugging information.

        Currently the following fields are defined:

        * ``tasks_waiting``: The number of tasks blocked on this lot's
          :meth:`park` method.

        """
        return ParkingLotStatistics(tasks_waiting=len(self._parked))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_fakenet.py ===
import errno
import re
import socket
import sys

import pytest

import trio
from trio.testing._fake_net import FakeNet

# ENOTCONN gives different messages on different platforms
if sys.platform == "linux":
    ENOTCONN_MSG = r"^\[Errno 107\] (Transport endpoint is|Socket) not connected$"
elif sys.platform == "darwin":
    ENOTCONN_MSG = r"^\[Errno 57\] Socket is not connected$"
else:
    ENOTCONN_MSG = r"^\[Errno 10057\] Unknown error$"


def fn() -> FakeNet:
    fn = FakeNet()
    fn.enable()
    return fn


async def test_basic_udp() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)

    await s1.bind(("127.0.0.1", 0))
    ip, port = s1.getsockname()
    assert ip == "127.0.0.1"
    assert port != 0

    with pytest.raises(
        OSError,
        match=r"^\[\w+ \d+\] Invalid argument$",
    ) as exc:  # Cannot rebind.
        await s1.bind(("192.0.2.1", 0))
    assert exc.value.errno == errno.EINVAL

    # Cannot bind multiple sockets to the same address
    with pytest.raises(
        OSError,
        match=r"^\[\w+ \d+\] (Address (already )?in use|Unknown error)$",
    ) as exc:
        await s2.bind(("127.0.0.1", port))
    assert exc.value.errno == errno.EADDRINUSE

    await s2.sendto(b"xyz", s1.getsockname())
    data, addr = await s1.recvfrom(10)
    assert data == b"xyz"
    assert addr == s2.getsockname()
    await s1.sendto(b"abc", s2.getsockname())
    data, addr = await s2.recvfrom(10)
    assert data == b"abc"
    assert addr == s1.getsockname()


async def test_msg_trunc() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    await s1.bind(("127.0.0.1", 0))
    await s2.sendto(b"xyz", s1.getsockname())
    await s1.recvfrom(10)


async def test_recv_methods() -> None:
    """Test all recv methods for codecov"""
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)

    # receiving on an unbound socket is a bad idea (I think?)
    with pytest.raises(NotImplementedError, match="code will most likely hang"):
        await s2.recv(10)

    await s1.bind(("127.0.0.1", 0))
    ip, port = s1.getsockname()
    assert ip == "127.0.0.1"
    assert port != 0

    # recvfrom
    await s2.sendto(b"abc", s1.getsockname())
    data, addr = await s1.recvfrom(10)
    assert data == b"abc"
    assert addr == s2.getsockname()

    # recv
    await s1.sendto(b"def", s2.getsockname())
    data = await s2.recv(10)
    assert data == b"def"

    # recvfrom_into
    assert await s1.sendto(b"ghi", s2.getsockname()) == 3
    buf = bytearray(10)

    with pytest.raises(NotImplementedError, match=r"^partial recvfrom_into$"):
        (nbytes, addr) = await s2.recvfrom_into(buf, nbytes=2)

    (nbytes, addr) = await s2.recvfrom_into(buf)
    assert nbytes == 3
    assert buf == b"ghi" + b"\x00" * 7
    assert addr == s1.getsockname()

    # recv_into
    assert await s1.sendto(b"jkl", s2.getsockname()) == 3
    buf2 = bytearray(10)
    nbytes = await s2.recv_into(buf2)
    assert nbytes == 3
    assert buf2 == b"jkl" + b"\x00" * 7

    if sys.platform == "linux" and sys.implementation.name == "cpython":
        flags: int = socket.MSG_MORE
    else:
        flags = 1

    # Send seems explicitly non-functional
    with pytest.raises(OSError, match=ENOTCONN_MSG) as exc:
        await s2.send(b"mno")
    assert exc.value.errno == errno.ENOTCONN
    with pytest.raises(
        NotImplementedError, match=r"^FakeNet send flags must be 0, not"
    ):
        await s2.send(b"mno", flags)

    # sendto errors
    # it's successfully used earlier
    with pytest.raises(
        NotImplementedError, match=r"^FakeNet send flags must be 0, not"
    ):
        await s2.sendto(b"mno", flags, s1.getsockname())
    with pytest.raises(TypeError, match=r"wrong number of arguments$"):
        await s2.sendto(b"mno", flags, s1.getsockname(), "extra arg")  # type: ignore[call-overload]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="functions not in socket on windows",
)
async def test_nonwindows_functionality() -> None:
    # mypy doesn't support a good way of aborting typechecking on different platforms
    if sys.platform != "win32":  # pragma: no branch
        fn()
        s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        await s2.bind(("127.0.0.1", 0))

        # sendmsg
        with pytest.raises(OSError, match=ENOTCONN_MSG) as exc:
            await s2.sendmsg([b"mno"])
        assert exc.value.errno == errno.ENOTCONN

        assert await s1.sendmsg([b"jkl"], (), 0, s2.getsockname()) == 3
        (data, ancdata, msg_flags, addr) = await s2.recvmsg(10)
        assert data == b"jkl"
        assert ancdata == []
        assert msg_flags == 0
        assert addr == s1.getsockname()

        # TODO: recvmsg

        # recvmsg_into
        assert await s1.sendto(b"xyzw", s2.getsockname()) == 4
        buf1 = bytearray(2)
        buf2 = bytearray(3)
        ret = await s2.recvmsg_into([buf1, buf2])
        (nbytes, ancdata, msg_flags, addr) = ret
        assert nbytes == 4
        assert buf1 == b"xy"
        assert buf2 == b"zw" + b"\x00"
        assert ancdata == []
        assert msg_flags == 0
        assert addr == s1.getsockname()

        # recvmsg_into with MSG_TRUNC set
        assert await s1.sendto(b"xyzwv", s2.getsockname()) == 5
        buf1 = bytearray(2)
        ret = await s2.recvmsg_into([buf1])
        (nbytes, ancdata, msg_flags, addr) = ret
        assert nbytes == 2
        assert buf1 == b"xy"
        assert ancdata == []
        assert msg_flags == socket.MSG_TRUNC
        assert addr == s1.getsockname()

        with pytest.raises(
            AttributeError,
            match=r"^'FakeSocket' object has no attribute 'share'$",
        ):
            await s1.share(0)  # type: ignore[attr-defined]


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="windows-specific fakesocket testing",
)
async def test_windows_functionality() -> None:
    # mypy doesn't support a good way of aborting typechecking on different platforms
    if sys.platform == "win32":  # pragma: no branch
        fn()
        s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        await s1.bind(("127.0.0.1", 0))
        with pytest.raises(
            AttributeError,
            match=r"^'FakeSocket' object has no attribute 'sendmsg'$",
        ):
            await s1.sendmsg([b"jkl"], (), 0, s2.getsockname())  # type: ignore[attr-defined]
        with pytest.raises(
            AttributeError,
            match=r"^'FakeSocket' object has no attribute 'recvmsg'$",
        ):
            s2.recvmsg(0)  # type: ignore[attr-defined]
        with pytest.raises(
            AttributeError,
            match=r"^'FakeSocket' object has no attribute 'recvmsg_into'$",
        ):
            s2.recvmsg_into([])  # type: ignore[attr-defined]
        with pytest.raises(NotImplementedError):
            s1.share(0)


async def test_basic_tcp() -> None:
    fn()
    with pytest.raises(NotImplementedError):
        trio.socket.socket()


async def test_not_implemented_functions() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)

    # getsockopt
    with pytest.raises(
        OSError,
        match=r"^FakeNet doesn't implement getsockopt\(\d, \d\)$",
    ):
        s1.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)

    # setsockopt
    with pytest.raises(
        NotImplementedError,
        match=r"^FakeNet always has IPV6_V6ONLY=True$",
    ):
        s1.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, False)
    with pytest.raises(
        OSError,
        match=r"^FakeNet doesn't implement setsockopt\(\d+, \d+, \.\.\.\)$",
    ):
        s1.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, True)
    with pytest.raises(
        OSError,
        match=r"^FakeNet doesn't implement setsockopt\(\d+, \d+, \.\.\.\)$",
    ):
        s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # set_inheritable
    s1.set_inheritable(False)
    with pytest.raises(
        NotImplementedError,
        match=r"^FakeNet can't make inheritable sockets$",
    ):
        s1.set_inheritable(True)

    # get_inheritable
    assert not s1.get_inheritable()


async def test_getpeername() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    with pytest.raises(OSError, match=ENOTCONN_MSG) as exc:
        s1.getpeername()
    assert exc.value.errno == errno.ENOTCONN

    await s1.bind(("127.0.0.1", 0))

    with pytest.raises(
        AssertionError,
        match=r"^This method seems to assume that self._binding has a remote UDPEndpoint$",
    ):
        s1.getpeername()


async def test_init() -> None:
    fn()
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"FakeNet doesn't (yet) support type={trio.socket.SOCK_STREAM}",
        ),
    ):
        s1 = trio.socket.socket()

    # getsockname on unbound ipv4 socket
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    assert s1.getsockname() == ("0.0.0.0", 0)

    # getsockname on bound ipv4 socket
    await s1.bind(("0.0.0.0", 0))
    ip, port = s1.getsockname()
    assert ip == "127.0.0.1"
    assert port != 0

    # getsockname on unbound ipv6 socket
    s2 = trio.socket.socket(family=socket.AF_INET6, type=socket.SOCK_DGRAM)
    assert s2.getsockname() == ("::", 0)

    # getsockname on bound ipv6 socket
    await s2.bind(("::", 0))
    ip, port, *_ = s2.getsockname()
    assert ip == "::1"
    assert port != 0
    assert _ == [0, 0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\auth.py ===
from __future__ import annotations

import base64
import binascii
import collections.abc as cabc
import typing as t

from ..http import dump_header
from ..http import parse_dict_header
from ..http import quote_header_value
from .structures import CallbackDict

if t.TYPE_CHECKING:
    import typing_extensions as te


class Authorization:
    """Represents the parts of an ``Authorization`` request header.

    :attr:`.Request.authorization` returns an instance if the header is set.

    An instance can be used with the test :class:`.Client` request methods' ``auth``
    parameter to send the header in test requests.

    Depending on the auth scheme, either :attr:`parameters` or :attr:`token` will be
    set. The ``Basic`` scheme's token is decoded into the ``username`` and ``password``
    parameters.

    For convenience, ``auth["key"]`` and ``auth.key`` both access the key in the
    :attr:`parameters` dict, along with ``auth.get("key")`` and ``"key" in auth``.

    .. versionchanged:: 2.3
        The ``token`` parameter and attribute was added to support auth schemes that use
        a token instead of parameters, such as ``Bearer``.

    .. versionchanged:: 2.3
        The object is no longer a ``dict``.

    .. versionchanged:: 0.5
        The object is an immutable dict.
    """

    def __init__(
        self,
        auth_type: str,
        data: dict[str, str | None] | None = None,
        token: str | None = None,
    ) -> None:
        self.type = auth_type
        """The authorization scheme, like ``basic``, ``digest``, or ``bearer``."""

        if data is None:
            data = {}

        self.parameters = data
        """A dict of parameters parsed from the header. Either this or :attr:`token`
        will have a value for a given scheme.
        """

        self.token = token
        """A token parsed from the header. Either this or :attr:`parameters` will have a
        value for a given scheme.

        .. versionadded:: 2.3
        """

    def __getattr__(self, name: str) -> str | None:
        return self.parameters.get(name)

    def __getitem__(self, name: str) -> str | None:
        return self.parameters.get(name)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.parameters.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Authorization):
            return NotImplemented

        return (
            other.type == self.type
            and other.token == self.token
            and other.parameters == self.parameters
        )

    @classmethod
    def from_header(cls, value: str | None) -> te.Self | None:
        """Parse an ``Authorization`` header value and return an instance, or ``None``
        if the value is empty.

        :param value: The header value to parse.

        .. versionadded:: 2.3
        """
        if not value:
            return None

        scheme, _, rest = value.partition(" ")
        scheme = scheme.lower()
        rest = rest.strip()

        if scheme == "basic":
            try:
                username, _, password = base64.b64decode(rest).decode().partition(":")
            except (binascii.Error, UnicodeError):
                return None

            return cls(scheme, {"username": username, "password": password})

        if "=" in rest.rstrip("="):
            # = that is not trailing, this is parameters.
            return cls(scheme, parse_dict_header(rest), None)

        # No = or only trailing =, this is a token.
        return cls(scheme, None, rest)

    def to_header(self) -> str:
        """Produce an ``Authorization`` header value representing this data.

        .. versionadded:: 2.0
        """
        if self.type == "basic":
            value = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode("ascii")
            return f"Basic {value}"

        if self.token is not None:
            return f"{self.type.title()} {self.token}"

        return f"{self.type.title()} {dump_header(self.parameters)}"

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.to_header()}>"


class WWWAuthenticate:
    """Represents the parts of a ``WWW-Authenticate`` response header.

    Set :attr:`.Response.www_authenticate` to an instance of list of instances to set
    values for this header in the response. Modifying this instance will modify the
    header value.

    Depending on the auth scheme, either :attr:`parameters` or :attr:`token` should be
    set. The ``Basic`` scheme will encode ``username`` and ``password`` parameters to a
    token.

    For convenience, ``auth["key"]`` and ``auth.key`` both act on the :attr:`parameters`
    dict, and can be used to get, set, or delete parameters. ``auth.get("key")`` and
    ``"key" in auth`` are also provided.

    .. versionchanged:: 2.3
        The ``token`` parameter and attribute was added to support auth schemes that use
        a token instead of parameters, such as ``Bearer``.

    .. versionchanged:: 2.3
        The object is no longer a ``dict``.

    .. versionchanged:: 2.3
        The ``on_update`` parameter was removed.
    """

    def __init__(
        self,
        auth_type: str,
        values: dict[str, str | None] | None = None,
        token: str | None = None,
    ):
        self._type = auth_type.lower()
        self._parameters: dict[str, str | None] = CallbackDict(
            values, lambda _: self._trigger_on_update()
        )
        self._token = token
        self._on_update: cabc.Callable[[WWWAuthenticate], None] | None = None

    def _trigger_on_update(self) -> None:
        if self._on_update is not None:
            self._on_update(self)

    @property
    def type(self) -> str:
        """The authorization scheme, like ``basic``, ``digest``, or ``bearer``."""
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        self._type = value
        self._trigger_on_update()

    @property
    def parameters(self) -> dict[str, str | None]:
        """A dict of parameters for the header. Only one of this or :attr:`token` should
        have a value for a given scheme.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: dict[str, str]) -> None:
        self._parameters = CallbackDict(value, lambda _: self._trigger_on_update())
        self._trigger_on_update()

    @property
    def token(self) -> str | None:
        """A dict of parameters for the header. Only one of this or :attr:`token` should
        have a value for a given scheme.
        """
        return self._token

    @token.setter
    def token(self, value: str | None) -> None:
        """A token for the header. Only one of this or :attr:`parameters` should have a
        value for a given scheme.

        .. versionadded:: 2.3
        """
        self._token = value
        self._trigger_on_update()

    def __getitem__(self, key: str) -> str | None:
        return self.parameters.get(key)

    def __setitem__(self, key: str, value: str | None) -> None:
        if value is None:
            if key in self.parameters:
                del self.parameters[key]
        else:
            self.parameters[key] = value

        self._trigger_on_update()

    def __delitem__(self, key: str) -> None:
        if key in self.parameters:
            del self.parameters[key]
            self._trigger_on_update()

    def __getattr__(self, name: str) -> str | None:
        return self[name]

    def __setattr__(self, name: str, value: str | None) -> None:
        if name in {"_type", "_parameters", "_token", "_on_update"}:
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WWWAuthenticate):
            return NotImplemented

        return (
            other.type == self.type
            and other.token == self.token
            and other.parameters == self.parameters
        )

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.parameters.get(key, default)

    @classmethod
    def from_header(cls, value: str | None) -> te.Self | None:
        """Parse a ``WWW-Authenticate`` header value and return an instance, or ``None``
        if the value is empty.

        :param value: The header value to parse.

        .. versionadded:: 2.3
        """
        if not value:
            return None

        scheme, _, rest = value.partition(" ")
        scheme = scheme.lower()
        rest = rest.strip()

        if "=" in rest.rstrip("="):
            # = that is not trailing, this is parameters.
            return cls(scheme, parse_dict_header(rest), None)

        # No = or only trailing =, this is a token.
        return cls(scheme, None, rest)

    def to_header(self) -> str:
        """Produce a ``WWW-Authenticate`` header value representing this data."""
        if self.token is not None:
            return f"{self.type.title()} {self.token}"

        if self.type == "digest":
            items = []

            for key, value in self.parameters.items():
                if key in {"realm", "domain", "nonce", "opaque", "qop"}:
                    value = quote_header_value(value, allow_token=False)
                else:
                    value = quote_header_value(value)

                items.append(f"{key}={value}")

            return f"Digest {', '.join(items)}"

        return f"{self.type.title()} {dump_header(self.parameters)}"

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.to_header()}>"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\mixins.py ===
from __future__ import annotations

import collections.abc as cabc
import typing as t
from functools import update_wrapper
from itertools import repeat

from .._internal import _missing

if t.TYPE_CHECKING:
    import typing_extensions as te

K = t.TypeVar("K")
V = t.TypeVar("V")
T = t.TypeVar("T")
F = t.TypeVar("F", bound=cabc.Callable[..., t.Any])


def _immutable_error(self: t.Any) -> t.NoReturn:
    raise TypeError(f"{type(self).__name__!r} objects are immutable")


class ImmutableListMixin:
    """Makes a :class:`list` immutable.

    .. versionadded:: 0.5

    :private:
    """

    _hash_cache: int | None = None

    def __hash__(self) -> int:
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(tuple(self))  # type: ignore[arg-type]
        return rv

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (list(self),)  # type: ignore[call-overload]

    def __delitem__(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __iadd__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __imul__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __setitem__(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def append(self, item: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def remove(self, item: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def extend(self, iterable: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def insert(self, pos: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def pop(self, index: t.Any = -1) -> t.NoReturn:
        _immutable_error(self)

    def reverse(self: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def sort(self, key: t.Any = None, reverse: t.Any = False) -> t.NoReturn:
        _immutable_error(self)


class ImmutableDictMixin(t.Generic[K, V]):
    """Makes a :class:`dict` immutable.

    .. versionchanged:: 3.1
        Disallow ``|=`` operator.

    .. versionadded:: 0.5

    :private:
    """

    _hash_cache: int | None = None

    @classmethod
    @t.overload
    def fromkeys(
        cls, keys: cabc.Iterable[K], value: None
    ) -> ImmutableDictMixin[K, t.Any | None]: ...
    @classmethod
    @t.overload
    def fromkeys(cls, keys: cabc.Iterable[K], value: V) -> ImmutableDictMixin[K, V]: ...
    @classmethod
    def fromkeys(
        cls, keys: cabc.Iterable[K], value: V | None = None
    ) -> ImmutableDictMixin[K, t.Any | None] | ImmutableDictMixin[K, V]:
        instance = super().__new__(cls)
        instance.__init__(zip(keys, repeat(value)))  # type: ignore[misc]
        return instance

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (dict(self),)  # type: ignore[call-overload]

    def _iter_hashitems(self) -> t.Iterable[t.Any]:
        return self.items()  # type: ignore[attr-defined,no-any-return]

    def __hash__(self) -> int:
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(frozenset(self._iter_hashitems()))
        return rv

    def setdefault(self, key: t.Any, default: t.Any = None) -> t.NoReturn:
        _immutable_error(self)

    def update(self, arg: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __ior__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def pop(self, key: t.Any, default: t.Any = None) -> t.NoReturn:
        _immutable_error(self)

    def popitem(self) -> t.NoReturn:
        _immutable_error(self)

    def __setitem__(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __delitem__(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def clear(self) -> t.NoReturn:
        _immutable_error(self)


class ImmutableMultiDictMixin(ImmutableDictMixin[K, V]):
    """Makes a :class:`MultiDict` immutable.

    .. versionadded:: 0.5

    :private:
    """

    def __reduce_ex__(self, protocol: t.SupportsIndex) -> t.Any:
        return type(self), (list(self.items(multi=True)),)  # type: ignore[attr-defined]

    def _iter_hashitems(self) -> t.Iterable[t.Any]:
        return self.items(multi=True)  # type: ignore[attr-defined,no-any-return]

    def add(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def popitemlist(self) -> t.NoReturn:
        _immutable_error(self)

    def poplist(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlist(self, key: t.Any, new_list: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlistdefault(self, key: t.Any, default_list: t.Any = None) -> t.NoReturn:
        _immutable_error(self)


class ImmutableHeadersMixin:
    """Makes a :class:`Headers` immutable.  We do not mark them as
    hashable though since the only usecase for this datastructure
    in Werkzeug is a view on a mutable structure.

    .. versionchanged:: 3.1
        Disallow ``|=`` operator.

    .. versionadded:: 0.5

    :private:
    """

    def __delitem__(self, key: t.Any, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __setitem__(self, key: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def set(self, key: t.Any, value: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlist(self, key: t.Any, values: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def add(self, key: t.Any, value: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def add_header(self, key: t.Any, value: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def remove(self, key: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def extend(self, arg: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def update(self, arg: t.Any, /, **kwargs: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def __ior__(self, other: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def insert(self, pos: t.Any, value: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def pop(self, key: t.Any = None, default: t.Any = _missing) -> t.NoReturn:
        _immutable_error(self)

    def popitem(self) -> t.NoReturn:
        _immutable_error(self)

    def setdefault(self, key: t.Any, default: t.Any) -> t.NoReturn:
        _immutable_error(self)

    def setlistdefault(self, key: t.Any, default: t.Any) -> t.NoReturn:
        _immutable_error(self)


def _always_update(f: F) -> F:
    def wrapper(
        self: UpdateDictMixin[t.Any, t.Any], /, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        rv = f(self, *args, **kwargs)

        if self.on_update is not None:
            self.on_update(self)

        return rv

    return update_wrapper(wrapper, f)  # type: ignore[return-value]


class UpdateDictMixin(dict[K, V]):
    """Makes dicts call `self.on_update` on modifications.

    .. versionchanged:: 3.1
        Implement ``|=`` operator.

    .. versionadded:: 0.5

    :private:
    """

    on_update: cabc.Callable[[te.Self], None] | None = None

    def setdefault(self: te.Self, key: K, default: V | None = None) -> V:
        modified = key not in self
        rv = super().setdefault(key, default)  # type: ignore[arg-type]
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv

    @t.overload
    def pop(self: te.Self, key: K) -> V: ...
    @t.overload
    def pop(self: te.Self, key: K, default: V) -> V: ...
    @t.overload
    def pop(self: te.Self, key: K, default: T) -> T: ...
    def pop(
        self: te.Self,
        key: K,
        default: V | T = _missing,  # type: ignore[assignment]
    ) -> V | T:
        modified = key in self
        if default is _missing:
            rv = super().pop(key)
        else:
            rv = super().pop(key, default)  # type: ignore[arg-type]
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv

    @_always_update
    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)

    @_always_update
    def __delitem__(self, key: K) -> None:
        super().__delitem__(key)

    @_always_update
    def clear(self) -> None:
        super().clear()

    @_always_update
    def popitem(self) -> tuple[K, V]:
        return super().popitem()

    @_always_update
    def update(  # type: ignore[override]
        self,
        arg: cabc.Mapping[K, V] | cabc.Iterable[tuple[K, V]] | None = None,
        /,
        **kwargs: V,
    ) -> None:
        if arg is None:
            super().update(**kwargs)
        else:
            super().update(arg, **kwargs)

    @_always_update
    def __ior__(  # type: ignore[override]
        self, other: cabc.Mapping[K, V] | cabc.Iterable[tuple[K, V]]
    ) -> te.Self:
        return super().__ior__(other)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\types.py ===
# dialects/oracle/types.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from __future__ import annotations

import datetime as dt
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING

from ... import exc
from ...sql import sqltypes
from ...types import NVARCHAR
from ...types import VARCHAR

if TYPE_CHECKING:
    from ...engine.interfaces import Dialect
    from ...sql.type_api import _LiteralProcessorType


class RAW(sqltypes._Binary):
    __visit_name__ = "RAW"


OracleRaw = RAW


class NCLOB(sqltypes.Text):
    __visit_name__ = "NCLOB"


class VARCHAR2(VARCHAR):
    __visit_name__ = "VARCHAR2"


NVARCHAR2 = NVARCHAR


class NUMBER(sqltypes.Numeric, sqltypes.Integer):
    __visit_name__ = "NUMBER"

    def __init__(self, precision=None, scale=None, asdecimal=None):
        if asdecimal is None:
            asdecimal = bool(scale and scale > 0)

        super().__init__(precision=precision, scale=scale, asdecimal=asdecimal)

    def adapt(self, impltype):
        ret = super().adapt(impltype)
        # leave a hint for the DBAPI handler
        ret._is_oracle_number = True
        return ret

    @property
    def _type_affinity(self):
        if bool(self.scale and self.scale > 0):
            return sqltypes.Numeric
        else:
            return sqltypes.Integer


class FLOAT(sqltypes.FLOAT):
    """Oracle Database FLOAT.

    This is the same as :class:`_sqltypes.FLOAT` except that
    an Oracle Database -specific :paramref:`_oracle.FLOAT.binary_precision`
    parameter is accepted, and
    the :paramref:`_sqltypes.Float.precision` parameter is not accepted.

    Oracle Database FLOAT types indicate precision in terms of "binary
    precision", which defaults to 126. For a REAL type, the value is 63. This
    parameter does not cleanly map to a specific number of decimal places but
    is roughly equivalent to the desired number of decimal places divided by
    0.3103.

    .. versionadded:: 2.0

    """

    __visit_name__ = "FLOAT"

    def __init__(
        self,
        binary_precision=None,
        asdecimal=False,
        decimal_return_scale=None,
    ):
        r"""
        Construct a FLOAT

        :param binary_precision: Oracle Database binary precision value to be
         rendered in DDL. This may be approximated to the number of decimal
         characters using the formula "decimal precision = 0.30103 * binary
         precision".  The default value used by Oracle Database for FLOAT /
         DOUBLE PRECISION is 126.

        :param asdecimal: See :paramref:`_sqltypes.Float.asdecimal`

        :param decimal_return_scale: See
         :paramref:`_sqltypes.Float.decimal_return_scale`

        """
        super().__init__(
            asdecimal=asdecimal, decimal_return_scale=decimal_return_scale
        )
        self.binary_precision = binary_precision


class BINARY_DOUBLE(sqltypes.Double):
    """Implement the Oracle ``BINARY_DOUBLE`` datatype.

    This datatype differs from the Oracle ``DOUBLE`` datatype in that it
    delivers a true 8-byte FP value.   The datatype may be combined with a
    generic :class:`.Double` datatype using :meth:`.TypeEngine.with_variant`.

    .. seealso::

        :ref:`oracle_float_support`


    """

    __visit_name__ = "BINARY_DOUBLE"


class BINARY_FLOAT(sqltypes.Float):
    """Implement the Oracle ``BINARY_FLOAT`` datatype.

    This datatype differs from the Oracle ``FLOAT`` datatype in that it
    delivers a true 4-byte FP value.   The datatype may be combined with a
    generic :class:`.Float` datatype using :meth:`.TypeEngine.with_variant`.

    .. seealso::

        :ref:`oracle_float_support`


    """

    __visit_name__ = "BINARY_FLOAT"


class BFILE(sqltypes.LargeBinary):
    __visit_name__ = "BFILE"


class LONG(sqltypes.Text):
    __visit_name__ = "LONG"


class _OracleDateLiteralRender:
    def _literal_processor_datetime(self, dialect):
        def process(value):
            if getattr(value, "microsecond", None):
                value = (
                    f"""TO_TIMESTAMP"""
                    f"""('{value.isoformat().replace("T", " ")}', """
                    """'YYYY-MM-DD HH24:MI:SS.FF')"""
                )
            else:
                value = (
                    f"""TO_DATE"""
                    f"""('{value.isoformat().replace("T", " ")}', """
                    """'YYYY-MM-DD HH24:MI:SS')"""
                )
            return value

        return process

    def _literal_processor_date(self, dialect):
        def process(value):
            if getattr(value, "microsecond", None):
                value = (
                    f"""TO_TIMESTAMP"""
                    f"""('{value.isoformat().split("T")[0]}', """
                    """'YYYY-MM-DD')"""
                )
            else:
                value = (
                    f"""TO_DATE"""
                    f"""('{value.isoformat().split("T")[0]}', """
                    """'YYYY-MM-DD')"""
                )
            return value

        return process


class DATE(_OracleDateLiteralRender, sqltypes.DateTime):
    """Provide the Oracle Database DATE type.

    This type has no special Python behavior, except that it subclasses
    :class:`_types.DateTime`; this is to suit the fact that the Oracle Database
    ``DATE`` type supports a time value.

    """

    __visit_name__ = "DATE"

    def literal_processor(self, dialect):
        return self._literal_processor_datetime(dialect)

    def _compare_type_affinity(self, other):
        return other._type_affinity in (sqltypes.DateTime, sqltypes.Date)


class _OracleDate(_OracleDateLiteralRender, sqltypes.Date):
    def literal_processor(self, dialect):
        return self._literal_processor_date(dialect)


class INTERVAL(sqltypes.NativeForEmulated, sqltypes._AbstractInterval):
    __visit_name__ = "INTERVAL"

    def __init__(self, day_precision=None, second_precision=None):
        """Construct an INTERVAL.

        Note that only DAY TO SECOND intervals are currently supported.
        This is due to a lack of support for YEAR TO MONTH intervals
        within available DBAPIs.

        :param day_precision: the day precision value.  this is the number of
          digits to store for the day field.  Defaults to "2"
        :param second_precision: the second precision value.  this is the
          number of digits to store for the fractional seconds field.
          Defaults to "6".

        """
        self.day_precision = day_precision
        self.second_precision = second_precision

    @classmethod
    def _adapt_from_generic_interval(cls, interval):
        return INTERVAL(
            day_precision=interval.day_precision,
            second_precision=interval.second_precision,
        )

    @classmethod
    def adapt_emulated_to_native(
        cls, interval: sqltypes.Interval, **kw  # type: ignore[override]
    ):
        return INTERVAL(
            day_precision=interval.day_precision,
            second_precision=interval.second_precision,
        )

    @property
    def _type_affinity(self):
        return sqltypes.Interval

    def as_generic(self, allow_nulltype=False):
        return sqltypes.Interval(
            native=True,
            second_precision=self.second_precision,
            day_precision=self.day_precision,
        )

    @property
    def python_type(self) -> Type[dt.timedelta]:
        return dt.timedelta

    def literal_processor(
        self, dialect: Dialect
    ) -> Optional[_LiteralProcessorType[dt.timedelta]]:
        def process(value: dt.timedelta) -> str:
            return f"NUMTODSINTERVAL({value.total_seconds()}, 'SECOND')"

        return process


class TIMESTAMP(sqltypes.TIMESTAMP):
    """Oracle Database implementation of ``TIMESTAMP``, which supports
    additional Oracle Database-specific modes

    .. versionadded:: 2.0

    """

    def __init__(self, timezone: bool = False, local_timezone: bool = False):
        """Construct a new :class:`_oracle.TIMESTAMP`.

        :param timezone: boolean.  Indicates that the TIMESTAMP type should
         use Oracle Database's ``TIMESTAMP WITH TIME ZONE`` datatype.

        :param local_timezone: boolean.  Indicates that the TIMESTAMP type
         should use Oracle Database's ``TIMESTAMP WITH LOCAL TIME ZONE``
         datatype.


        """
        if timezone and local_timezone:
            raise exc.ArgumentError(
                "timezone and local_timezone are mutually exclusive"
            )
        super().__init__(timezone=timezone)
        self.local_timezone = local_timezone


class ROWID(sqltypes.TypeEngine):
    """Oracle Database ROWID type.

    When used in a cast() or similar, generates ROWID.

    """

    __visit_name__ = "ROWID"


class _OracleBoolean(sqltypes.Boolean):
    def get_dbapi_type(self, dbapi):
        return dbapi.NUMBER

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\sphinx.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: June 11, 2021
# URL: https://humanfriendly.readthedocs.io

"""
Customizations for and integration with the Sphinx_ documentation generator.

The :mod:`humanfriendly.sphinx` module uses the `Sphinx extension API`_ to
customize the process of generating Sphinx based Python documentation. To
explore the functionality this module offers its best to start reading
from the :func:`setup()` function.

.. _Sphinx: http://www.sphinx-doc.org/
.. _Sphinx extension API: http://sphinx-doc.org/extdev/appapi.html
"""

# Standard library modules.
import logging
import types

# External dependencies (if Sphinx is installed docutils will be installed).
import docutils.nodes
import docutils.utils

# Modules included in our package.
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage

# Public identifiers that require documentation.
__all__ = (
    "deprecation_note_callback",
    "enable_deprecation_notes",
    "enable_man_role",
    "enable_pypi_role",
    "enable_special_methods",
    "enable_usage_formatting",
    "logger",
    "man_role",
    "pypi_role",
    "setup",
    "special_methods_callback",
    "usage_message_callback",
)

# Initialize a logger for this module.
logger = logging.getLogger(__name__)


def deprecation_note_callback(app, what, name, obj, options, lines):
    """
    Automatically document aliases defined using :func:`~humanfriendly.deprecation.define_aliases()`.

    Refer to :func:`enable_deprecation_notes()` to enable the use of this
    function (you probably don't want to call :func:`deprecation_note_callback()`
    directly).

    This function implements a callback for ``autodoc-process-docstring`` that
    reformats module docstrings to append an overview of aliases defined by the
    module.

    The parameters expected by this function are those defined for Sphinx event
    callback functions (i.e. I'm not going to document them here :-).
    """
    if isinstance(obj, types.ModuleType) and lines:
        aliases = get_aliases(obj.__name__)
        if aliases:
            # Convert the existing docstring to a string and remove leading
            # indentation from that string, otherwise our generated content
            # would have to match the existing indentation in order not to
            # break docstring parsing (because indentation is significant
            # in the reStructuredText format).
            blocks = [dedent("\n".join(lines))]
            # Use an admonition to group the deprecated aliases together and
            # to distinguish them from the autodoc entries that follow.
            blocks.append(".. note:: Deprecated names")
            indent = " " * 3
            if len(aliases) == 1:
                explanation = """
                    The following alias exists to preserve backwards compatibility,
                    however a :exc:`~exceptions.DeprecationWarning` is triggered
                    when it is accessed, because this alias will be removed
                    in a future release.
                """
            else:
                explanation = """
                    The following aliases exist to preserve backwards compatibility,
                    however a :exc:`~exceptions.DeprecationWarning` is triggered
                    when they are accessed, because these aliases will be
                    removed in a future release.
                """
            blocks.append(indent + compact(explanation))
            for name, target in aliases.items():
                blocks.append(format("%s.. data:: %s", indent, name))
                blocks.append(format("%sAlias for :obj:`%s`.", indent * 2, target))
            update_lines(lines, "\n\n".join(blocks))


def enable_deprecation_notes(app):
    """
    Enable documenting backwards compatibility aliases using the autodoc_ extension.

    :param app: The Sphinx application object.

    This function connects the :func:`deprecation_note_callback()` function to
    ``autodoc-process-docstring`` events.

    .. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
    """
    app.connect("autodoc-process-docstring", deprecation_note_callback)


def enable_man_role(app):
    """
    Enable the ``:man:`` role for linking to Debian Linux manual pages.

    :param app: The Sphinx application object.

    This function registers the :func:`man_role()` function to handle the
    ``:man:`` role.
    """
    app.add_role("man", man_role)


def enable_pypi_role(app):
    """
    Enable the ``:pypi:`` role for linking to the Python Package Index.

    :param app: The Sphinx application object.

    This function registers the :func:`pypi_role()` function to handle the
    ``:pypi:`` role.
    """
    app.add_role("pypi", pypi_role)


def enable_special_methods(app):
    """
    Enable documenting "special methods" using the autodoc_ extension.

    :param app: The Sphinx application object.

    This function connects the :func:`special_methods_callback()` function to
    ``autodoc-skip-member`` events.

    .. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
    """
    app.connect("autodoc-skip-member", special_methods_callback)


def enable_usage_formatting(app):
    """
    Reformat human friendly usage messages to reStructuredText_.

    :param app: The Sphinx application object (as given to ``setup()``).

    This function connects the :func:`usage_message_callback()` function to
    ``autodoc-process-docstring`` events.

    .. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
    """
    app.connect("autodoc-process-docstring", usage_message_callback)


def man_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Convert a Linux manual topic to a hyperlink.

    Using the ``:man:`` role is very simple, here's an example:

    .. code-block:: rst

        See the :man:`python` documentation.

    This results in the following:

      See the :man:`python` documentation.

    As the example shows you can use the role inline, embedded in sentences of
    text. In the generated documentation the ``:man:`` text is omitted and a
    hyperlink pointing to the Debian Linux manual pages is emitted.
    """
    man_url = "https://manpages.debian.org/%s" % text
    reference = docutils.nodes.reference(rawtext, docutils.utils.unescape(text), refuri=man_url, **options)
    return [reference], []


def pypi_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Generate hyperlinks to the Python Package Index.

    Using the ``:pypi:`` role is very simple, here's an example:

    .. code-block:: rst

        See the :pypi:`humanfriendly` package.

    This results in the following:

      See the :pypi:`humanfriendly` package.

    As the example shows you can use the role inline, embedded in sentences of
    text. In the generated documentation the ``:pypi:`` text is omitted and a
    hyperlink pointing to the Python Package Index is emitted.
    """
    pypi_url = "https://pypi.org/project/%s/" % text
    reference = docutils.nodes.reference(rawtext, docutils.utils.unescape(text), refuri=pypi_url, **options)
    return [reference], []


def setup(app):
    """
    Enable all of the provided Sphinx_ customizations.

    :param app: The Sphinx application object.

    The :func:`setup()` function makes it easy to enable all of the Sphinx
    customizations provided by the :mod:`humanfriendly.sphinx` module with the
    least amount of code. All you need to do is to add the module name to the
    ``extensions`` variable in your ``conf.py`` file:

    .. code-block:: python

       # Sphinx extension module names.
       extensions = [
           'sphinx.ext.autodoc',
           'sphinx.ext.doctest',
           'sphinx.ext.intersphinx',
           'humanfriendly.sphinx',
       ]

    When Sphinx sees the :mod:`humanfriendly.sphinx` name it will import the
    module and call its :func:`setup()` function. This function will then call
    the following:

    - :func:`enable_deprecation_notes()`
    - :func:`enable_man_role()`
    - :func:`enable_pypi_role()`
    - :func:`enable_special_methods()`
    - :func:`enable_usage_formatting()`

    Of course more functionality may be added at a later stage. If you don't
    like that idea you may be better of calling the individual functions from
    your own ``setup()`` function.
    """
    from humanfriendly import __version__

    enable_deprecation_notes(app)
    enable_man_role(app)
    enable_pypi_role(app)
    enable_special_methods(app)
    enable_usage_formatting(app)

    return dict(parallel_read_safe=True, parallel_write_safe=True, version=__version__)


def special_methods_callback(app, what, name, obj, skip, options):
    """
    Enable documenting "special methods" using the autodoc_ extension.

    Refer to :func:`enable_special_methods()` to enable the use of this
    function (you probably don't want to call
    :func:`special_methods_callback()` directly).

    This function implements a callback for ``autodoc-skip-member`` events to
    include documented "special methods" (method names with two leading and two
    trailing underscores) in your documentation. The result is similar to the
    use of the ``special-members`` flag with one big difference: Special
    methods are included but other types of members are ignored. This means
    that attributes like ``__weakref__`` will always be ignored (this was my
    main annoyance with the ``special-members`` flag).

    The parameters expected by this function are those defined for Sphinx event
    callback functions (i.e. I'm not going to document them here :-).
    """
    if getattr(obj, "__doc__", None) and isinstance(obj, (types.FunctionType, types.MethodType)):
        return False
    else:
        return skip


def update_lines(lines, text):
    """Private helper for ``autodoc-process-docstring`` callbacks."""
    while lines:
        lines.pop()
    lines.extend(text.splitlines())


def usage_message_callback(app, what, name, obj, options, lines):
    """
    Reformat human friendly usage messages to reStructuredText_.

    Refer to :func:`enable_usage_formatting()` to enable the use of this
    function (you probably don't want to call :func:`usage_message_callback()`
    directly).

    This function implements a callback for ``autodoc-process-docstring`` that
    reformats module docstrings using :func:`.render_usage()` so that Sphinx
    doesn't mangle usage messages that were written to be human readable
    instead of machine readable. Only module docstrings whose first line starts
    with :data:`.USAGE_MARKER` are reformatted.

    The parameters expected by this function are those defined for Sphinx event
    callback functions (i.e. I'm not going to document them here :-).
    """
    # Make sure we only modify the docstrings of modules.
    if isinstance(obj, types.ModuleType) and lines:
        # Make sure we only modify docstrings containing a usage message.
        if lines[0].startswith(USAGE_MARKER):
            # Convert the usage message to reStructuredText.
            text = render_usage("\n".join(lines))
            # Fill up the buffer with our modified docstring.
            update_lines(lines, text)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\density.py ===
from itertools import product

from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_numpy
from sympy.physics.quantum.trace import Tr


class Density(HermitianOperator):
    """Density operator for representing mixed states.

    TODO: Density operator support for Qubits

    Parameters
    ==========

    values : tuples/lists
    Each tuple/list should be of form (state, prob) or [state,prob]

    Examples
    ========

    Create a density operator with 2 states represented by Kets.

    >>> from sympy.physics.quantum.state import Ket
    >>> from sympy.physics.quantum.density import Density
    >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
    >>> d
    Density((|0>, 0.5),(|1>, 0.5))

    """
    @classmethod
    def _eval_args(cls, args):
        # call this to qsympify the args
        args = super()._eval_args(args)

        for arg in args:
            # Check if arg is a tuple
            if not (isinstance(arg, Tuple) and len(arg) == 2):
                raise ValueError("Each argument should be of form [state,prob]"
                                 " or ( state, prob )")

        return args

    def states(self):
        """Return list of all states.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()
        (|0>, |1>)

        """
        return Tuple(*[arg[0] for arg in self.args])

    def probs(self):
        """Return list of all probabilities.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()
        (0.5, 0.5)

        """
        return Tuple(*[arg[1] for arg in self.args])

    def get_state(self, index):
        """Return specific state by index.

        Parameters
        ==========

        index : index of state to be returned

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()[1]
        |1>

        """
        state = self.args[index][0]
        return state

    def get_prob(self, index):
        """Return probability of specific state by index.

        Parameters
        ===========

        index : index of states whose probability is returned.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()[1]
        0.500000000000000

        """
        prob = self.args[index][1]
        return prob

    def apply_op(self, op):
        """op will operate on each individual state.

        Parameters
        ==========

        op : Operator

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.apply_op(A)
        Density((A*|0>, 0.5),(A*|1>, 0.5))

        """
        new_args = [(op*state, prob) for (state, prob) in self.args]
        return Density(*new_args)

    def doit(self, **hints):
        """Expand the density operator into an outer product format.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.doit()
        0.5*|0><0| + 0.5*|1><1|

        """

        terms = []
        for (state, prob) in self.args:
            state = state.expand()  # needed to break up (a+b)*c
            if (isinstance(state, Add)):
                for arg in product(state.args, repeat=2):
                    terms.append(prob*self._generate_outer_prod(arg[0],
                                                                arg[1]))
            else:
                terms.append(prob*self._generate_outer_prod(state, state))

        return Add(*terms)

    def _generate_outer_prod(self, arg1, arg2):
        c_part1, nc_part1 = arg1.args_cnc()
        c_part2, nc_part2 = arg2.args_cnc()

        if (len(nc_part1) == 0 or len(nc_part2) == 0):
            raise ValueError('Atleast one-pair of'
                             ' Non-commutative instance required'
                             ' for outer product.')

        # We were able to remove some tensor product simplifications that
        # used to be here as those transformations are not automatically
        # applied by transforms.py.
        op = Mul(*nc_part1)*Dagger(Mul(*nc_part2))

        return Mul(*c_part1)*Mul(*c_part2) * op

    def _represent(self, **options):
        return represent(self.doit(), **options)

    def _print_operator_name_latex(self, printer, *args):
        return r'\rho'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('\N{GREEK SMALL LETTER RHO}')

    def _eval_trace(self, **kwargs):
        indices = kwargs.get('indices', [])
        return Tr(self.doit(), indices).doit()

    def entropy(self):
        """ Compute the entropy of a density matrix.

        Refer to density.entropy() method  for examples.
        """
        return entropy(self)


def entropy(density):
    """Compute the entropy of a matrix/density object.

    This computes -Tr(density*ln(density)) using the eigenvalue decomposition
    of density, which is given as either a Density instance or a matrix
    (numpy.ndarray, sympy.Matrix or scipy.sparse).

    Parameters
    ==========

    density : density matrix of type Density, SymPy matrix,
    scipy.sparse or numpy.ndarray

    Examples
    ========

    >>> from sympy.physics.quantum.density import Density, entropy
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy import S
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> d = Density((up,S(1)/2),(down,S(1)/2))
    >>> entropy(d)
    log(2)/2

    """
    if isinstance(density, Density):
        density = represent(density)  # represent in Matrix

    if isinstance(density, scipy_sparse_matrix):
        density = to_numpy(density)

    if isinstance(density, Matrix):
        eigvals = density.eigenvals().keys()
        return expand(-sum(e*log(e) for e in eigvals))
    elif isinstance(density, numpy_ndarray):
        import numpy as np
        eigvals = np.linalg.eigvals(density)
        return -np.sum(eigvals*np.log(eigvals))
    else:
        raise ValueError(
            "numpy.ndarray, scipy.sparse or SymPy matrix expected")


def fidelity(state1, state2):
    """ Computes the fidelity [1]_ between two quantum states

    The arguments provided to this function should be a square matrix or a
    Density object. If it is a square matrix, it is assumed to be diagonalizable.

    Parameters
    ==========

    state1, state2 : a density matrix or Matrix


    Examples
    ========

    >>> from sympy import S, sqrt
    >>> from sympy.physics.quantum.dagger import Dagger
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy.physics.quantum.density import fidelity
    >>> from sympy.physics.quantum.represent import represent
    >>>
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> amp = 1/sqrt(2)
    >>> updown = (amp*up) + (amp*down)
    >>>
    >>> # represent turns Kets into matrices
    >>> up_dm = represent(up*Dagger(up))
    >>> down_dm = represent(down*Dagger(down))
    >>> updown_dm = represent(updown*Dagger(updown))
    >>>
    >>> fidelity(up_dm, up_dm)
    1
    >>> fidelity(up_dm, down_dm) #orthogonal states
    0
    >>> fidelity(up_dm, updown_dm).evalf().round(3)
    0.707

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fidelity_of_quantum_states

    """
    state1 = represent(state1) if isinstance(state1, Density) else state1
    state2 = represent(state2) if isinstance(state2, Density) else state2

    if not isinstance(state1, Matrix) or not isinstance(state2, Matrix):
        raise ValueError("state1 and state2 must be of type Density or Matrix "
                         "received type=%s for state1 and type=%s for state2" %
                         (type(state1), type(state2)))

    if state1.shape != state2.shape and state1.is_square:
        raise ValueError("The dimensions of both args should be equal and the "
                         "matrix obtained should be a square matrix")

    sqrt_state1 = state1**S.Half
    return Tr((sqrt_state1*state2*sqrt_state1)**S.Half).doit()