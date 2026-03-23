
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest10.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a, b = _sm.symbols('a b', real=True)
e = a*(b*x+y)**2
m = _sm.Matrix([e,e]).reshape(2, 1)
e = e.expand()
m = _sm.Matrix([i.expand() for i in m]).reshape((m).shape[0], (m).shape[1])
e = _sm.factor(e, x)
m = _sm.Matrix([_sm.factor(i,x) for i in m]).reshape((m).shape[0], (m).shape[1])
eqn = _sm.Matrix([[0]])
eqn[0] = a*x+b*y
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = 2*a*x-3*b*y
print(_sm.solve(eqn,x,y))
rhs_y = _sm.solve(eqn,x,y)[y]
e = (x+y)**2+2*x**2
e.collect(x)
a, b, c = _sm.symbols('a b c', real=True)
m = _sm.Matrix([a,b,c,0]).reshape(2, 2)
m2 = _sm.Matrix([i.subs({a:1,b:2,c:3}) for i in m]).reshape((m).shape[0], (m).shape[1])
eigvalue = _sm.Matrix([i.evalf() for i in (m2).eigenvals().keys()])
eigvec = _sm.Matrix([i[2][0].evalf() for i in (m2).eigenvects()]).reshape(m2.shape[0], m2.shape[1])
frame_n = _me.ReferenceFrame('n')
frame_a = _me.ReferenceFrame('a')
frame_a.orient(frame_n, 'Axis', [x, frame_n.x])
frame_a.orient(frame_n, 'Axis', [_sm.pi/2, frame_n.x])
c1, c2, c3 = _sm.symbols('c1 c2 c3', real=True)
v = c1*frame_a.x+c2*frame_a.y+c3*frame_a.z
point_o = _me.Point('o')
point_p = _me.Point('p')
point_o.set_pos(point_p, c1*frame_a.x)
v = (v).express(frame_n)
point_o.set_pos(point_p, (point_o.pos_from(point_p)).express(frame_n))
frame_a.set_ang_vel(frame_n, c3*frame_a.z)
print(frame_n.ang_vel_in(frame_a))
point_p.v2pt_theory(point_o,frame_n,frame_a)
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
particle_p2.point.v2pt_theory(particle_p1.point,frame_n,frame_a)
point_p.a2pt_theory(particle_p1.point,frame_n,frame_a)
body_b1_cm = _me.Point('b1_cm')
body_b1_cm.set_vel(frame_n, 0)
body_b1_f = _me.ReferenceFrame('b1_f')
body_b1 = _me.RigidBody('b1', body_b1_cm, body_b1_f, _sm.symbols('m'), (_me.outer(body_b1_f.x,body_b1_f.x),body_b1_cm))
body_b2_cm = _me.Point('b2_cm')
body_b2_cm.set_vel(frame_n, 0)
body_b2_f = _me.ReferenceFrame('b2_f')
body_b2 = _me.RigidBody('b2', body_b2_cm, body_b2_f, _sm.symbols('m'), (_me.outer(body_b2_f.x,body_b2_f.x),body_b2_cm))
g = _sm.symbols('g', real=True)
force_p1 = particle_p1.mass*(g*frame_n.x)
force_p2 = particle_p2.mass*(g*frame_n.x)
force_b1 = body_b1.mass*(g*frame_n.x)
force_b2 = body_b2.mass*(g*frame_n.x)
z = _me.dynamicsymbols('z')
v = x*frame_a.x+y*frame_a.z
point_o.set_pos(point_p, x*frame_a.x+y*frame_a.y)
v = (v).subs({x:2*z, y:z})
point_o.set_pos(point_p, (point_o.pos_from(point_p)).subs({x:2*z, y:z}))
force_o = -1*(x*y*frame_a.x)
force_p1 = particle_p1.mass*(g*frame_n.x)+ x*y*frame_a.x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_deprecate_strict_exception_groups_false.py ===
from collections.abc import Awaitable, Callable

import pytest

import trio


async def test_deprecation_warning_open_nursery() -> None:
    with pytest.warns(
        trio.TrioDeprecationWarning,
        match="strict_exception_groups=False",
    ) as record:
        async with trio.open_nursery(strict_exception_groups=False):
            ...
    assert len(record) == 1
    async with trio.open_nursery(strict_exception_groups=True):
        ...
    async with trio.open_nursery():
        ...


def test_deprecation_warning_run() -> None:
    async def foo() -> None: ...

    async def foo_nursery() -> None:
        # this should not raise a warning, even if it's implied loose
        async with trio.open_nursery():
            ...

    async def foo_loose_nursery() -> None:
        # this should raise a warning, even if specifying the parameter is redundant
        async with trio.open_nursery(strict_exception_groups=False):
            ...

    def helper(fun: Callable[[], Awaitable[None]], num: int) -> None:
        with pytest.warns(
            trio.TrioDeprecationWarning,
            match="strict_exception_groups=False",
        ) as record:
            trio.run(fun, strict_exception_groups=False)
        assert len(record) == num

    helper(foo, 1)
    helper(foo_nursery, 1)
    helper(foo_loose_nursery, 2)


def test_deprecation_warning_start_guest_run() -> None:
    # "The simplest possible "host" loop."
    from .._core._tests.test_guest_mode import trivial_guest_run

    async def trio_return(in_host: object) -> str:
        await trio.lowlevel.checkpoint()
        return "ok"

    with pytest.warns(
        trio.TrioDeprecationWarning,
        match="strict_exception_groups=False",
    ) as record:
        trivial_guest_run(
            trio_return,
            strict_exception_groups=False,
        )
    assert len(record) == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\__init__.py ===
from __future__ import annotations

import typing as t

from .accept import Accept as Accept
from .accept import CharsetAccept as CharsetAccept
from .accept import LanguageAccept as LanguageAccept
from .accept import MIMEAccept as MIMEAccept
from .auth import Authorization as Authorization
from .auth import WWWAuthenticate as WWWAuthenticate
from .cache_control import RequestCacheControl as RequestCacheControl
from .cache_control import ResponseCacheControl as ResponseCacheControl
from .csp import ContentSecurityPolicy as ContentSecurityPolicy
from .etag import ETags as ETags
from .file_storage import FileMultiDict as FileMultiDict
from .file_storage import FileStorage as FileStorage
from .headers import EnvironHeaders as EnvironHeaders
from .headers import Headers as Headers
from .mixins import ImmutableDictMixin as ImmutableDictMixin
from .mixins import ImmutableHeadersMixin as ImmutableHeadersMixin
from .mixins import ImmutableListMixin as ImmutableListMixin
from .mixins import ImmutableMultiDictMixin as ImmutableMultiDictMixin
from .mixins import UpdateDictMixin as UpdateDictMixin
from .range import ContentRange as ContentRange
from .range import IfRange as IfRange
from .range import Range as Range
from .structures import CallbackDict as CallbackDict
from .structures import CombinedMultiDict as CombinedMultiDict
from .structures import HeaderSet as HeaderSet
from .structures import ImmutableDict as ImmutableDict
from .structures import ImmutableList as ImmutableList
from .structures import ImmutableMultiDict as ImmutableMultiDict
from .structures import ImmutableTypeConversionDict as ImmutableTypeConversionDict
from .structures import iter_multi_items as iter_multi_items
from .structures import MultiDict as MultiDict
from .structures import TypeConversionDict as TypeConversionDict


def __getattr__(name: str) -> t.Any:
    import warnings

    if name == "OrderedMultiDict":
        from .structures import _OrderedMultiDict

        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'MultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _OrderedMultiDict

    if name == "ImmutableOrderedMultiDict":
        from .structures import _ImmutableOrderedMultiDict

        warnings.warn(
            "'OrderedMultiDict' is deprecated and will be removed in Werkzeug"
            " 3.2. Use 'ImmutableMultiDict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ImmutableOrderedMultiDict

    raise AttributeError(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\test_inference.py ===
"""
These the test the public routines exposed in types/common.py
related to inference and not otherwise tested in types/test_common.py

"""
import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
    Generic,
    TypeVar,
)

import numpy as np
import pytest
import pytz

from pandas._libs import (
    lib,
    missing as libmissing,
    ops as libops,
)
from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
    ensure_int32,
    is_bool,
    is_complex,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_float,
    is_integer,
    is_number,
    is_scalar,
    is_scipy_sparse,
    is_timedelta64_dtype,
    is_timedelta64_ns_dtype,
)

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DateOffset,
    DatetimeIndex,
    Index,
    Interval,
    Period,
    PeriodIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    FloatingArray,
    IntegerArray,
)


@pytest.fixture(params=[True, False], ids=str)
def coerce(request):
    return request.param


class MockNumpyLikeArray:
    """
    A class which is numpy-like (e.g. Pint's Quantity) but not actually numpy

    The key is that it is not actually a numpy array so
    ``util.is_array(mock_numpy_like_array_instance)`` returns ``False``. Other
    important properties are that the class defines a :meth:`__iter__` method
    (so that ``isinstance(abc.Iterable)`` returns ``True``) and has a
    :meth:`ndim` property, as pandas special-cases 0-dimensional arrays in some
    cases.

    We expect pandas to behave with respect to such duck arrays exactly as
    with real numpy arrays. In particular, a 0-dimensional duck array is *NOT*
    a scalar (`is_scalar(np.array(1)) == False`), but it is not list-like either.
    """

    def __init__(self, values) -> None:
        self._values = values

    def __iter__(self) -> Iterator:
        iter_values = iter(self._values)

        def it_outer():
            yield from iter_values

        return it_outer()

    def __len__(self) -> int:
        return len(self._values)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._values, dtype=dtype)

    @property
    def ndim(self):
        return self._values.ndim

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def size(self):
        return self._values.size

    @property
    def shape(self):
        return self._values.shape


# collect all objects to be tested for list-like-ness; use tuples of objects,
# whether they are list-like or not (special casing for sets), and their ID
ll_params = [
    ([1], True, "list"),
    ([], True, "list-empty"),
    ((1,), True, "tuple"),
    ((), True, "tuple-empty"),
    ({"a": 1}, True, "dict"),
    ({}, True, "dict-empty"),
    ({"a", 1}, "set", "set"),
    (set(), "set", "set-empty"),
    (frozenset({"a", 1}), "set", "frozenset"),
    (frozenset(), "set", "frozenset-empty"),
    (iter([1, 2]), True, "iterator"),
    (iter([]), True, "iterator-empty"),
    ((x for x in [1, 2]), True, "generator"),
    ((_ for _ in []), True, "generator-empty"),
    (Series([1]), True, "Series"),
    (Series([], dtype=object), True, "Series-empty"),
    # Series.str will still raise a TypeError if iterated
    (Series(["a"]).str, True, "StringMethods"),
    (Series([], dtype="O").str, True, "StringMethods-empty"),
    (Index([1]), True, "Index"),
    (Index([]), True, "Index-empty"),
    (DataFrame([[1]]), True, "DataFrame"),
    (DataFrame(), True, "DataFrame-empty"),
    (np.ndarray((2,) * 1), True, "ndarray-1d"),
    (np.array([]), True, "ndarray-1d-empty"),
    (np.ndarray((2,) * 2), True, "ndarray-2d"),
    (np.array([[]]), True, "ndarray-2d-empty"),
    (np.ndarray((2,) * 3), True, "ndarray-3d"),
    (np.array([[[]]]), True, "ndarray-3d-empty"),
    (np.ndarray((2,) * 4), True, "ndarray-4d"),
    (np.array([[[[]]]]), True, "ndarray-4d-empty"),
    (np.array(2), False, "ndarray-0d"),
    (MockNumpyLikeArray(np.ndarray((2,) * 1)), True, "duck-ndarray-1d"),
    (MockNumpyLikeArray(np.array([])), True, "duck-ndarray-1d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 2)), True, "duck-ndarray-2d"),
    (MockNumpyLikeArray(np.array([[]])), True, "duck-ndarray-2d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 3)), True, "duck-ndarray-3d"),
    (MockNumpyLikeArray(np.array([[[]]])), True, "duck-ndarray-3d-empty"),
    (MockNumpyLikeArray(np.ndarray((2,) * 4)), True, "duck-ndarray-4d"),
    (MockNumpyLikeArray(np.array([[[[]]]])), True, "duck-ndarray-4d-empty"),
    (MockNumpyLikeArray(np.array(2)), False, "duck-ndarray-0d"),
    (1, False, "int"),
    (b"123", False, "bytes"),
    (b"", False, "bytes-empty"),
    ("123", False, "string"),
    ("", False, "string-empty"),
    (str, False, "string-type"),
    (object(), False, "object"),
    (np.nan, False, "NaN"),
    (None, False, "None"),
]
objs, expected, ids = zip(*ll_params)


@pytest.fixture(params=zip(objs, expected), ids=ids)
def maybe_list_like(request):
    return request.param


def test_is_list_like(maybe_list_like):
    obj, expected = maybe_list_like
    expected = True if expected == "set" else expected
    assert inference.is_list_like(obj) == expected


def test_is_list_like_disallow_sets(maybe_list_like):
    obj, expected = maybe_list_like
    expected = False if expected == "set" else expected
    assert inference.is_list_like(obj, allow_sets=False) == expected


def test_is_list_like_recursion():
    # GH 33721
    # interpreter would crash with SIGABRT
    def list_like():
        inference.is_list_like([])
        list_like()

    rec_limit = sys.getrecursionlimit()
    try:
        # Limit to avoid stack overflow on Windows CI
        sys.setrecursionlimit(100)
        with tm.external_error_raised(RecursionError):
            list_like()
    finally:
        sys.setrecursionlimit(rec_limit)


def test_is_list_like_iter_is_none():
    # GH 43373
    # is_list_like was yielding false positives with __iter__ == None
    class NotListLike:
        def __getitem__(self, item):
            return self

        __iter__ = None

    assert not inference.is_list_like(NotListLike())


def test_is_list_like_generic():
    # GH 49649
    # is_list_like was yielding false positives for Generic classes in python 3.11
    T = TypeVar("T")

    class MyDataFrame(DataFrame, Generic[T]):
        ...

    tstc = MyDataFrame[int]
    tst = MyDataFrame[int]({"x": [1, 2, 3]})

    assert not inference.is_list_like(tstc)
    assert isinstance(tst, DataFrame)
    assert inference.is_list_like(tst)


def test_is_sequence():
    is_seq = inference.is_sequence
    assert is_seq((1, 2))
    assert is_seq([1, 2])
    assert not is_seq("abcd")
    assert not is_seq(np.int64)

    class A:
        def __getitem__(self, item):
            return 1

    assert not is_seq(A())


def test_is_array_like():
    assert inference.is_array_like(Series([], dtype=object))
    assert inference.is_array_like(Series([1, 2]))
    assert inference.is_array_like(np.array(["a", "b"]))
    assert inference.is_array_like(Index(["2016-01-01"]))
    assert inference.is_array_like(np.array([2, 3]))
    assert inference.is_array_like(MockNumpyLikeArray(np.array([2, 3])))

    class DtypeList(list):
        dtype = "special"

    assert inference.is_array_like(DtypeList())

    assert not inference.is_array_like([1, 2, 3])
    assert not inference.is_array_like(())
    assert not inference.is_array_like("foo")
    assert not inference.is_array_like(123)


@pytest.mark.parametrize(
    "inner",
    [
        [],
        [1],
        (1,),
        (1, 2),
        {"a": 1},
        {1, "a"},
        Series([1]),
        Series([], dtype=object),
        Series(["a"]).str,
        (x for x in range(5)),
    ],
)
@pytest.mark.parametrize("outer", [list, Series, np.array, tuple])
def test_is_nested_list_like_passes(inner, outer):
    result = outer([inner for _ in range(5)])
    assert inference.is_list_like(result)


@pytest.mark.parametrize(
    "obj",
    [
        "abc",
        [],
        [1],
        (1,),
        ["a"],
        "a",
        {"a"},
        [1, 2, 3],
        Series([1]),
        DataFrame({"A": [1]}),
        ([1, 2] for _ in range(5)),
    ],
)
def test_is_nested_list_like_fails(obj):
    assert not inference.is_nested_list_like(obj)


@pytest.mark.parametrize("ll", [{}, {"A": 1}, Series([1]), collections.defaultdict()])
def test_is_dict_like_passes(ll):
    assert inference.is_dict_like(ll)


@pytest.mark.parametrize(
    "ll",
    [
        "1",
        1,
        [1, 2],
        (1, 2),
        range(2),
        Index([1]),
        dict,
        collections.defaultdict,
        Series,
    ],
)
def test_is_dict_like_fails(ll):
    assert not inference.is_dict_like(ll)


@pytest.mark.parametrize("has_keys", [True, False])
@pytest.mark.parametrize("has_getitem", [True, False])
@pytest.mark.parametrize("has_contains", [True, False])
def test_is_dict_like_duck_type(has_keys, has_getitem, has_contains):
    class DictLike:
        def __init__(self, d) -> None:
            self.d = d

        if has_keys:

            def keys(self):
                return self.d.keys()

        if has_getitem:

            def __getitem__(self, key):
                return self.d.__getitem__(key)

        if has_contains:

            def __contains__(self, key) -> bool:
                return self.d.__contains__(key)

    d = DictLike({1: 2})
    result = inference.is_dict_like(d)
    expected = has_keys and has_getitem and has_contains

    assert result is expected


def test_is_file_like():
    class MockFile:
        pass

    is_file = inference.is_file_like

    data = StringIO("data")
    assert is_file(data)

    # No read / write attributes
    # No iterator attributes
    m = MockFile()
    assert not is_file(m)

    MockFile.write = lambda self: 0

    # Write attribute but not an iterator
    m = MockFile()
    assert not is_file(m)

    # gh-16530: Valid iterator just means we have the
    # __iter__ attribute for our purposes.
    MockFile.__iter__ = lambda self: self

    # Valid write-only file
    m = MockFile()
    assert is_file(m)

    del MockFile.write
    MockFile.read = lambda self: 0

    # Valid read-only file
    m = MockFile()
    assert is_file(m)

    # Iterator but no read / write attributes
    data = [1, 2, 3]
    assert not is_file(data)


test_tuple = collections.namedtuple("test_tuple", ["a", "b", "c"])


@pytest.mark.parametrize("ll", [test_tuple(1, 2, 3)])
def test_is_names_tuple_passes(ll):
    assert inference.is_named_tuple(ll)


@pytest.mark.parametrize("ll", [(1, 2, 3), "a", Series({"pi": 3.14})])
def test_is_names_tuple_fails(ll):
    assert not inference.is_named_tuple(ll)


def test_is_hashable():
    # all new-style classes are hashable by default
    class HashableClass:
        pass

    class UnhashableClass1:
        __hash__ = None

    class UnhashableClass2:
        def __hash__(self):
            raise TypeError("Not hashable")

    hashable = (1, 3.14, np.float64(3.14), "a", (), (1,), HashableClass())
    not_hashable = ([], UnhashableClass1())
    abc_hashable_not_really_hashable = (([],), UnhashableClass2())

    for i in hashable:
        assert inference.is_hashable(i)
    for i in not_hashable:
        assert not inference.is_hashable(i)
    for i in abc_hashable_not_really_hashable:
        assert not inference.is_hashable(i)

    # numpy.array is no longer collections.abc.Hashable as of
    # https://github.com/numpy/numpy/pull/5326, just test
    # is_hashable()
    assert not inference.is_hashable(np.array([]))


@pytest.mark.parametrize("ll", [re.compile("ad")])
def test_is_re_passes(ll):
    assert inference.is_re(ll)


@pytest.mark.parametrize("ll", ["x", 2, 3, object()])
def test_is_re_fails(ll):
    assert not inference.is_re(ll)


@pytest.mark.parametrize(
    "ll", [r"a", "x", r"asdf", re.compile("adsf"), r"\u2233\s*", re.compile(r"")]
)
def test_is_recompilable_passes(ll):
    assert inference.is_re_compilable(ll)


@pytest.mark.parametrize("ll", [1, [], object()])
def test_is_recompilable_fails(ll):
    assert not inference.is_re_compilable(ll)


class TestInference:
    @pytest.mark.parametrize(
        "arr",
        [
            np.array(list("abc"), dtype="S1"),
            np.array(list("abc"), dtype="S1").astype(object),
            [b"a", np.nan, b"c"],
        ],
    )
    def test_infer_dtype_bytes(self, arr):
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "bytes"

    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("inf"), True),
            (np.inf, True),
            (-np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isposinf_scalar(self, value, expected):
        # GH 11352
        result = libmissing.isposinf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (float("-inf"), True),
            (-np.inf, True),
            (np.inf, False),
            (1, False),
            ("a", False),
        ],
    )
    def test_isneginf_scalar(self, value, expected):
        result = libmissing.isneginf_scalar(value)
        assert result is expected

    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (
                True,
                BooleanArray(
                    np.array([True, False], dtype="bool"), np.array([False, True])
                ),
            ),
            (False, np.array([True, np.nan], dtype="object")),
        ],
    )
    def test_maybe_convert_nullable_boolean(self, convert_to_masked_nullable, exp):
        # GH 40687
        arr = np.array([True, np.nan], dtype=object)
        result = libops.maybe_convert_bool(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(BooleanArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    @pytest.mark.parametrize("coerce_numeric", [True, False])
    @pytest.mark.parametrize(
        "infinity", ["inf", "inF", "iNf", "Inf", "iNF", "InF", "INf", "INF"]
    )
    @pytest.mark.parametrize("prefix", ["", "-", "+"])
    def test_maybe_convert_numeric_infinities(
        self, coerce_numeric, infinity, prefix, convert_to_masked_nullable
    ):
        # see gh-13274
        result, _ = lib.maybe_convert_numeric(
            np.array([prefix + infinity], dtype=object),
            na_values={"", "NULL", "nan"},
            coerce_numeric=coerce_numeric,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        expected = np.array([np.inf if prefix in ["", "+"] else -np.inf])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_maybe_convert_numeric_infinities_raises(self, convert_to_masked_nullable):
        msg = "Unable to parse string"
        with pytest.raises(ValueError, match=msg):
            lib.maybe_convert_numeric(
                np.array(["foo_inf"], dtype=object),
                na_values={"", "NULL", "nan"},
                coerce_numeric=False,
                convert_to_masked_nullable=convert_to_masked_nullable,
            )

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_maybe_convert_numeric_post_floatify_nan(
        self, coerce, convert_to_masked_nullable
    ):
        # see gh-13314
        data = np.array(["1.200", "-999.000", "4.500"], dtype=object)
        expected = np.array([1.2, np.nan, 4.5], dtype=np.float64)
        nan_values = {-999, -999.0}

        out = lib.maybe_convert_numeric(
            data,
            nan_values,
            coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable:
            expected = FloatingArray(expected, np.isnan(expected))
            tm.assert_extension_array_equal(expected, FloatingArray(*out))
        else:
            out = out[0]
            tm.assert_numpy_array_equal(out, expected)

    def test_convert_infs(self):
        arr = np.array(["inf", "inf", "inf"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

        arr = np.array(["-inf", "-inf", "-inf"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False)
        assert result.dtype == np.float64

    def test_scientific_no_exponent(self):
        # See PR 12215
        arr = np.array(["42E", "2E", "99e", "6e"], dtype="O")
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        assert np.all(np.isnan(result))

    def test_convert_non_hashable(self):
        # GH13324
        # make sure that we are handing non-hashables
        arr = np.array([[10.0, 2], 1.0, "apple"], dtype=object)
        result, _ = lib.maybe_convert_numeric(arr, set(), False, True)
        tm.assert_numpy_array_equal(result, np.array([np.nan, 1.0, np.nan]))

    def test_convert_numeric_uint64(self):
        arr = np.array([2**63], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        arr = np.array([str(2**63)], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

        arr = np.array([np.uint64(2**63)], dtype=object)
        exp = np.array([2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(lib.maybe_convert_numeric(arr, set())[0], exp)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([2**63, np.nan], dtype=object),
            np.array([str(2**63), np.nan], dtype=object),
            np.array([np.nan, 2**63], dtype=object),
            np.array([np.nan, str(2**63)], dtype=object),
        ],
    )
    def test_convert_numeric_uint64_nan(self, coerce, arr):
        expected = arr.astype(float) if coerce else arr.copy()
        result, _ = lib.maybe_convert_numeric(arr, set(), coerce_numeric=coerce)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_uint64_nan_values(
        self, coerce, convert_to_masked_nullable
    ):
        arr = np.array([2**63, 2**63 + 1], dtype=object)
        na_values = {2**63}

        expected = (
            np.array([np.nan, 2**63 + 1], dtype=float) if coerce else arr.copy()
        )
        result = lib.maybe_convert_numeric(
            arr,
            na_values,
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable and coerce:
            expected = IntegerArray(
                np.array([0, 2**63 + 1], dtype="u8"),
                np.array([True, False], dtype="bool"),
            )
            result = IntegerArray(*result)
        else:
            result = result[0]  # discard mask
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "case",
        [
            np.array([2**63, -1], dtype=object),
            np.array([str(2**63), -1], dtype=object),
            np.array([str(2**63), str(-1)], dtype=object),
            np.array([-1, 2**63], dtype=object),
            np.array([-1, str(2**63)], dtype=object),
            np.array([str(-1), str(2**63)], dtype=object),
        ],
    )
    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_int64_uint64(
        self, case, coerce, convert_to_masked_nullable
    ):
        expected = case.astype(float) if coerce else case.copy()
        result, _ = lib.maybe_convert_numeric(
            case,
            set(),
            coerce_numeric=coerce,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )

        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("convert_to_masked_nullable", [True, False])
    def test_convert_numeric_string_uint64(self, convert_to_masked_nullable):
        # GH32394
        result = lib.maybe_convert_numeric(
            np.array(["uint64"], dtype=object),
            set(),
            coerce_numeric=True,
            convert_to_masked_nullable=convert_to_masked_nullable,
        )
        if convert_to_masked_nullable:
            result = FloatingArray(*result)
        else:
            result = result[0]
        assert np.isnan(result)

    @pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
    def test_convert_int_overflow(self, value):
        # see gh-18584
        arr = np.array([value], dtype=object)
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(arr, result)

    @pytest.mark.parametrize("val", [None, np.nan, float("nan")])
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_maybe_convert_objects_nat_inference(self, val, dtype):
        dtype = np.dtype(dtype)
        vals = np.array([pd.NaT, val], dtype=object)
        result = lib.maybe_convert_objects(
            vals,
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        assert result.dtype == dtype
        assert np.isnat(result).all()

        result = lib.maybe_convert_objects(
            vals[::-1],
            convert_non_numeric=True,
            dtype_if_all_nat=dtype,
        )
        assert result.dtype == dtype
        assert np.isnat(result).all()

    @pytest.mark.parametrize(
        "value, expected_dtype",
        [
            # see gh-4471
            ([2**63], np.uint64),
            # NumPy bug: can't compare uint64 to int64, as that
            # results in both casting to float64, so we should
            # make sure that this function is robust against it
            ([np.uint64(2**63)], np.uint64),
            ([2, -1], np.int64),
            ([2**63, -1], object),
            # GH#47294
            ([np.uint8(1)], np.uint8),
            ([np.uint16(1)], np.uint16),
            ([np.uint32(1)], np.uint32),
            ([np.uint64(1)], np.uint64),
            ([np.uint8(2), np.uint16(1)], np.uint16),
            ([np.uint32(2), np.uint16(1)], np.uint32),
            ([np.uint32(2), -1], object),
            ([np.uint32(2), 1], np.uint64),
            ([np.uint32(2), np.int32(1)], object),
        ],
    )
    def test_maybe_convert_objects_uint(self, value, expected_dtype):
        arr = np.array(value, dtype=object)
        exp = np.array(value, dtype=expected_dtype)
        tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)

    def test_maybe_convert_objects_datetime(self):
        # GH27438
        arr = np.array(
            [np.datetime64("2000-01-01"), np.timedelta64(1, "s")], dtype=object
        )
        exp = arr.copy()
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

        arr = np.array([pd.NaT, np.timedelta64(1, "s")], dtype=object)
        exp = np.array([np.timedelta64("NaT"), np.timedelta64(1, "s")], dtype="m8[ns]")
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

        # with convert_non_numeric=True, the nan is a valid NA value for td64
        arr = np.array([np.timedelta64(1, "s"), np.nan], dtype=object)
        exp = exp[::-1]
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat(self):
        arr = np.array([pd.NaT, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no dtype_if_all_nat passed -> we dont guess
        tm.assert_numpy_array_equal(out, arr)

        out = lib.maybe_convert_objects(
            arr,
            convert_non_numeric=True,
            dtype_if_all_nat=np.dtype("timedelta64[ns]"),
        )
        exp = np.array(["NaT", "NaT"], dtype="timedelta64[ns]")
        tm.assert_numpy_array_equal(out, exp)

        out = lib.maybe_convert_objects(
            arr,
            convert_non_numeric=True,
            dtype_if_all_nat=np.dtype("datetime64[ns]"),
        )
        exp = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self):
        # we accept datetime64[ns], timedelta64[ns], and EADtype
        arr = np.array([pd.NaT, pd.NaT], dtype=object)

        with pytest.raises(ValueError, match="int64"):
            lib.maybe_convert_objects(
                arr,
                convert_non_numeric=True,
                dtype_if_all_nat=np.dtype("int64"),
            )

    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_maybe_convert_objects_datetime_overflow_safe(self, dtype):
        stamp = datetime(2363, 10, 4)  # Enterprise-D launch date
        if dtype == "timedelta64[ns]":
            stamp = stamp - datetime(1970, 1, 1)
        arr = np.array([stamp], dtype=object)

        out = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        # no OutOfBoundsDatetime/OutOfBoundsTimedeltas
        tm.assert_numpy_array_equal(out, arr)

    def test_maybe_convert_objects_mixed_datetimes(self):
        ts = Timestamp("now")
        vals = [ts, ts.to_pydatetime(), ts.to_datetime64(), pd.NaT, np.nan, None]

        for data in itertools.permutations(vals):
            data = np.array(list(data), dtype=object)
            expected = DatetimeIndex(data)._data._ndarray
            result = lib.maybe_convert_objects(data, convert_non_numeric=True)
            tm.assert_numpy_array_equal(result, expected)

    def test_maybe_convert_objects_timedelta64_nat(self):
        obj = np.timedelta64("NaT", "ns")
        arr = np.array([obj], dtype=object)
        assert arr[0] is obj

        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)

        expected = np.array([obj], dtype="m8[ns]")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "exp",
        [
            IntegerArray(np.array([2, 0], dtype="i8"), np.array([False, True])),
            IntegerArray(np.array([2, 0], dtype="int64"), np.array([False, True])),
        ],
    )
    def test_maybe_convert_objects_nullable_integer(self, exp):
        # GH27335
        arr = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)

        tm.assert_extension_array_equal(result, exp)

    @pytest.mark.parametrize(
        "dtype, val", [("int64", 1), ("uint64", np.iinfo(np.int64).max + 1)]
    )
    def test_maybe_convert_objects_nullable_none(self, dtype, val):
        # GH#50043
        arr = np.array([val, None, 3], dtype="object")
        result = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        expected = IntegerArray(
            np.array([val, 0, 3], dtype=dtype), np.array([False, True, False])
        )
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (True, IntegerArray(np.array([2, 0], dtype="i8"), np.array([False, True]))),
            (False, np.array([2, np.nan], dtype="float64")),
        ],
    )
    def test_maybe_convert_numeric_nullable_integer(
        self, convert_to_masked_nullable, exp
    ):
        # GH 40687
        arr = np.array([2, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            result = IntegerArray(*result)
            tm.assert_extension_array_equal(result, exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize(
        "convert_to_masked_nullable, exp",
        [
            (
                True,
                FloatingArray(
                    np.array([2.0, 0.0], dtype="float64"), np.array([False, True])
                ),
            ),
            (False, np.array([2.0, np.nan], dtype="float64")),
        ],
    )
    def test_maybe_convert_numeric_floating_array(
        self, convert_to_masked_nullable, exp
    ):
        # GH 40687
        arr = np.array([2.0, np.nan], dtype=object)
        result = lib.maybe_convert_numeric(
            arr, set(), convert_to_masked_nullable=convert_to_masked_nullable
        )
        if convert_to_masked_nullable:
            tm.assert_extension_array_equal(FloatingArray(*result), exp)
        else:
            result = result[0]
            tm.assert_numpy_array_equal(result, exp)

    def test_maybe_convert_objects_bool_nan(self):
        # GH32146
        ind = Index([True, False, np.nan], dtype=object)
        exp = np.array([True, False, np.nan], dtype=object)
        out = lib.maybe_convert_objects(ind.values, safe=1)
        tm.assert_numpy_array_equal(out, exp)

    def test_maybe_convert_objects_nullable_boolean(self):
        # GH50047
        arr = np.array([True, False], dtype=object)
        exp = np.array([True, False])
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_numpy_array_equal(out, exp)

        arr = np.array([True, False, pd.NaT], dtype=object)
        exp = np.array([True, False, pd.NaT], dtype=object)
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_numpy_array_equal(out, exp)

    @pytest.mark.parametrize("val", [None, np.nan])
    def test_maybe_convert_objects_nullable_boolean_na(self, val):
        # GH50047
        arr = np.array([True, False, val], dtype=object)
        exp = BooleanArray(
            np.array([True, False, False]), np.array([False, False, True])
        )
        out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
        tm.assert_extension_array_equal(out, exp)

    @pytest.mark.parametrize(
        "data0",
        [
            True,
            1,
            1.0,
            1.0 + 1.0j,
            np.int8(1),
            np.int16(1),
            np.int32(1),
            np.int64(1),
            np.float16(1),
            np.float32(1),
            np.float64(1),
            np.complex64(1),
            np.complex128(1),
        ],
    )
    @pytest.mark.parametrize(
        "data1",
        [
            True,
            1,
            1.0,
            1.0 + 1.0j,
            np.int8(1),
            np.int16(1),
            np.int32(1),
            np.int64(1),
            np.float16(1),
            np.float32(1),
            np.float64(1),
            np.complex64(1),
            np.complex128(1),
        ],
    )
    def test_maybe_convert_objects_itemsize(self, data0, data1):
        # GH 40908
        data = [data0, data1]
        arr = np.array(data, dtype="object")

        common_kind = np.result_type(type(data0), type(data1)).kind
        kind0 = "python" if not hasattr(data0, "dtype") else data0.dtype.kind
        kind1 = "python" if not hasattr(data1, "dtype") else data1.dtype.kind
        if kind0 != "python" and kind1 != "python":
            kind = common_kind
            itemsize = max(data0.dtype.itemsize, data1.dtype.itemsize)
        elif is_bool(data0) or is_bool(data1):
            kind = "bool" if (is_bool(data0) and is_bool(data1)) else "object"
            itemsize = ""
        elif is_complex(data0) or is_complex(data1):
            kind = common_kind
            itemsize = 16
        else:
            kind = common_kind
            itemsize = 8

        expected = np.array(data, dtype=f"{kind}{itemsize}")
        result = lib.maybe_convert_objects(arr)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_dtypes_remain_object_array(self):
        # GH14956
        arr = np.array([datetime(2015, 1, 1, tzinfo=pytz.utc), 1], dtype=object)
        result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
        tm.assert_numpy_array_equal(result, arr)

    @pytest.mark.parametrize(
        "idx",
        [
            pd.IntervalIndex.from_breaks(range(5), closed="both"),
            pd.period_range("2016-01-01", periods=3, freq="D"),
        ],
    )
    def test_maybe_convert_objects_ea(self, idx):
        result = lib.maybe_convert_objects(
            np.array(idx, dtype=object),
            convert_non_numeric=True,
        )
        tm.assert_extension_array_equal(result, idx._data)


class TestTypeInference:
    # Dummy class used for testing with Python objects
    class Dummy:
        pass

    def test_inferred_dtype_fixture(self, any_skipna_inferred_dtype):
        # see pandas/conftest.py
        inferred_dtype, values = any_skipna_inferred_dtype

        # make sure the inferred dtype of the fixture is as requested
        assert inferred_dtype == lib.infer_dtype(values, skipna=True)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_length_zero(self, skipna):
        result = lib.infer_dtype(np.array([], dtype="i4"), skipna=skipna)
        assert result == "integer"

        result = lib.infer_dtype([], skipna=skipna)
        assert result == "empty"

        # GH 18004
        arr = np.array([np.array([], dtype=object), np.array([], dtype=object)])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "empty"

    def test_integers(self):
        arr = np.array([1, 2, 3, np.int64(4), np.int32(5)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "integer"

        arr = np.array([1, 2, 3, np.int64(4), np.int32(5), "foo"], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed-integer"

        arr = np.array([1, 2, 3, 4, 5], dtype="i4")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "integer"

    @pytest.mark.parametrize(
        "arr, skipna",
        [
            (np.array([1, 2, np.nan, np.nan, 3], dtype="O"), False),
            (np.array([1, 2, np.nan, np.nan, 3], dtype="O"), True),
            (np.array([1, 2, 3, np.int64(4), np.int32(5), np.nan], dtype="O"), False),
            (np.array([1, 2, 3, np.int64(4), np.int32(5), np.nan], dtype="O"), True),
        ],
    )
    def test_integer_na(self, arr, skipna):
        # GH 27392
        result = lib.infer_dtype(arr, skipna=skipna)
        expected = "integer" if skipna else "integer-na"
        assert result == expected

    def test_infer_dtype_skipna_default(self):
        # infer_dtype `skipna` default deprecated in GH#24050,
        #  changed to True in GH#29876
        arr = np.array([1, 2, 3, np.nan], dtype=object)

        result = lib.infer_dtype(arr)
        assert result == "integer"

    def test_bools(self):
        arr = np.array([True, False, True, True, True], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        arr = np.array([np.bool_(True), np.bool_(False)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        arr = np.array([True, False, True, "foo"], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed"

        arr = np.array([True, False, True], dtype=bool)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        arr = np.array([True, np.nan, False], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "boolean"

        result = lib.infer_dtype(arr, skipna=False)
        assert result == "mixed"

    def test_floats(self):
        arr = np.array([1.0, 2.0, 3.0, np.float64(4), np.float32(5)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "floating"

        arr = np.array([1, 2, 3, np.float64(4), np.float32(5), "foo"], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed-integer"

        arr = np.array([1, 2, 3, 4, 5], dtype="f4")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "floating"

        arr = np.array([1, 2, 3, 4, 5], dtype="f8")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "floating"

    def test_decimals(self):
        # GH15690
        arr = np.array([Decimal(1), Decimal(2), Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "decimal"

        arr = np.array([1.0, 2.0, Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed"

        result = lib.infer_dtype(arr[::-1], skipna=True)
        assert result == "mixed"

        arr = np.array([Decimal(1), Decimal("NaN"), Decimal(3)])
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "decimal"

        arr = np.array([Decimal(1), np.nan, Decimal(3)], dtype="O")
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "decimal"

    # complex is compatible with nan, so skipna has no effect
    @pytest.mark.parametrize("skipna", [True, False])
    def test_complex(self, skipna):
        # gets cast to complex on array construction
        arr = np.array([1.0, 2.0, 1 + 1j])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

        arr = np.array([1.0, 2.0, 1 + 1j], dtype="O")
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "mixed"

        result = lib.infer_dtype(arr[::-1], skipna=skipna)
        assert result == "mixed"

        # gets cast to complex on array construction
        arr = np.array([1, np.nan, 1 + 1j])
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

        arr = np.array([1.0, np.nan, 1 + 1j], dtype="O")
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "mixed"

        # complex with nans stays complex
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype="O")
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

        # test smaller complex dtype; will pass through _try_infer_map fastpath
        arr = np.array([1 + 1j, np.nan, 3 + 3j], dtype=np.complex64)
        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == "complex"

    def test_string(self):
        pass

    def test_unicode(self):
        arr = ["a", np.nan, "c"]
        result = lib.infer_dtype(arr, skipna=False)
        # This currently returns "mixed", but it's not clear that's optimal.
        # This could also return "string" or "mixed-string"
        assert result == "mixed"

        # even though we use skipna, we are only skipping those NAs that are
        #  considered matching by is_string_array
        arr = ["a", np.nan, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "string"

        arr = ["a", pd.NA, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "string"

        arr = ["a", pd.NaT, "c"]
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "mixed"

        arr = ["a", "c"]
        result = lib.infer_dtype(arr, skipna=False)
        assert result == "string"

    @pytest.mark.parametrize(
        "dtype, missing, skipna, expected",
        [
            (float, np.nan, False, "floating"),
            (float, np.nan, True, "floating"),
            (object, np.nan, False, "floating"),
            (object, np.nan, True, "empty"),
            (object, None, False, "mixed"),
            (object, None, True, "empty"),
        ],
    )
    @pytest.mark.parametrize("box", [Series, np.array])
    def test_object_empty(self, box, missing, dtype, skipna, expected):
        # GH 23421
        arr = box([missing, missing], dtype=dtype)

        result = lib.infer_dtype(arr, skipna=skipna)
        assert result == expected

    def test_datetime(self):
        dates = [datetime(2012, 1, x) for x in range(1, 20)]
        index = Index(dates)
        assert index.inferred_type == "datetime64"

    def test_infer_dtype_datetime64(self):
        arr = np.array(
            [np.datetime64("2011-01-01"), np.datetime64("2011-01-01")], dtype=object
        )
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    def test_infer_dtype_datetime64_with_na(self, na_value):
        # starts with nan
        arr = np.array([na_value, np.datetime64("2011-01-02")])
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

        arr = np.array([na_value, np.datetime64("2011-01-02"), na_value])
        assert lib.infer_dtype(arr, skipna=True) == "datetime64"

    @pytest.mark.parametrize(
        "arr",
        [
            np.array(
                [np.timedelta64("nat"), np.datetime64("2011-01-02")], dtype=object
            ),
            np.array(
                [np.datetime64("2011-01-02"), np.timedelta64("nat")], dtype=object
            ),
            np.array([np.datetime64("2011-01-01"), Timestamp("2011-01-02")]),
            np.array([Timestamp("2011-01-02"), np.datetime64("2011-01-01")]),
            np.array([np.nan, Timestamp("2011-01-02"), 1.1]),
            np.array([np.nan, "2011-01-01", Timestamp("2011-01-02")], dtype=object),
            np.array([np.datetime64("nat"), np.timedelta64(1, "D")], dtype=object),
            np.array([np.timedelta64(1, "D"), np.datetime64("nat")], dtype=object),
        ],
    )
    def test_infer_datetimelike_dtype_mixed(self, arr):
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    def test_infer_dtype_mixed_integer(self):
        arr = np.array([np.nan, Timestamp("2011-01-02"), 1])
        assert lib.infer_dtype(arr, skipna=True) == "mixed-integer"

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")]),
            np.array([datetime(2011, 1, 1), datetime(2012, 2, 1)]),
            np.array([datetime(2011, 1, 1), Timestamp("2011-01-02")]),
        ],
    )
    def test_infer_dtype_datetime(self, arr):
        assert lib.infer_dtype(arr, skipna=True) == "datetime"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    @pytest.mark.parametrize(
        "time_stamp", [Timestamp("2011-01-01"), datetime(2011, 1, 1)]
    )
    def test_infer_dtype_datetime_with_na(self, na_value, time_stamp):
        # starts with nan
        arr = np.array([na_value, time_stamp])
        assert lib.infer_dtype(arr, skipna=True) == "datetime"

        arr = np.array([na_value, time_stamp, na_value])
        assert lib.infer_dtype(arr, skipna=True) == "datetime"

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([Timedelta("1 days"), Timedelta("2 days")]),
            np.array([np.timedelta64(1, "D"), np.timedelta64(2, "D")], dtype=object),
            np.array([timedelta(1), timedelta(2)]),
        ],
    )
    def test_infer_dtype_timedelta(self, arr):
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    @pytest.mark.parametrize(
        "delta", [Timedelta("1 days"), np.timedelta64(1, "D"), timedelta(1)]
    )
    def test_infer_dtype_timedelta_with_na(self, na_value, delta):
        # starts with nan
        arr = np.array([na_value, delta])
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

        arr = np.array([na_value, delta, na_value])
        assert lib.infer_dtype(arr, skipna=True) == "timedelta"

    def test_infer_dtype_period(self):
        # GH 13664
        arr = np.array([Period("2011-01", freq="D"), Period("2011-02", freq="D")])
        assert lib.infer_dtype(arr, skipna=True) == "period"

        # non-homogeneous freqs -> mixed
        arr = np.array([Period("2011-01", freq="D"), Period("2011-02", freq="M")])
        assert lib.infer_dtype(arr, skipna=True) == "mixed"

    @pytest.mark.parametrize("klass", [pd.array, Series, Index])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_infer_dtype_period_array(self, klass, skipna):
        # https://github.com/pandas-dev/pandas/issues/23553
        values = klass(
            [
                Period("2011-01-01", freq="D"),
                Period("2011-01-02", freq="D"),
                pd.NaT,
            ]
        )
        assert lib.infer_dtype(values, skipna=skipna) == "period"

        # periods but mixed freq
        values = klass(
            [
                Period("2011-01-01", freq="D"),
                Period("2011-01-02", freq="M"),
                pd.NaT,
            ]
        )
        # with pd.array this becomes NumpyExtensionArray which ends up
        #  as "unknown-array"
        exp = "unknown-array" if klass is pd.array else "mixed"
        assert lib.infer_dtype(values, skipna=skipna) == exp

    def test_infer_dtype_period_mixed(self):
        arr = np.array(
            [Period("2011-01", freq="M"), np.datetime64("nat")], dtype=object
        )
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array(
            [np.datetime64("nat"), Period("2011-01", freq="M")], dtype=object
        )
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    @pytest.mark.parametrize("na_value", [pd.NaT, np.nan])
    def test_infer_dtype_period_with_na(self, na_value):
        # starts with nan
        arr = np.array([na_value, Period("2011-01", freq="D")])
        assert lib.infer_dtype(arr, skipna=True) == "period"

        arr = np.array([na_value, Period("2011-01", freq="D"), na_value])
        assert lib.infer_dtype(arr, skipna=True) == "period"

    def test_infer_dtype_all_nan_nat_like(self):
        arr = np.array([np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == "floating"

        # nan and None mix are result in mixed
        arr = np.array([np.nan, np.nan, None])
        assert lib.infer_dtype(arr, skipna=True) == "empty"
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array([None, np.nan, np.nan])
        assert lib.infer_dtype(arr, skipna=True) == "empty"
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # pd.NaT
        arr = np.array([pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([np.nan, pd.NaT])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([np.nan, pd.NaT, np.nan])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        arr = np.array([None, pd.NaT, None])
        assert lib.infer_dtype(arr, skipna=False) == "datetime"

        # np.datetime64(nat)
        arr = np.array([np.datetime64("nat")])
        assert lib.infer_dtype(arr, skipna=False) == "datetime64"

        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.datetime64("nat"), n])
            assert lib.infer_dtype(arr, skipna=False) == "datetime64"

            arr = np.array([pd.NaT, n, np.datetime64("nat"), n])
            assert lib.infer_dtype(arr, skipna=False) == "datetime64"

        arr = np.array([np.timedelta64("nat")], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == "timedelta"

        for n in [np.nan, pd.NaT, None]:
            arr = np.array([n, np.timedelta64("nat"), n])
            assert lib.infer_dtype(arr, skipna=False) == "timedelta"

            arr = np.array([pd.NaT, n, np.timedelta64("nat"), n])
            assert lib.infer_dtype(arr, skipna=False) == "timedelta"

        # datetime / timedelta mixed
        arr = np.array([pd.NaT, np.datetime64("nat"), np.timedelta64("nat"), np.nan])
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array([np.timedelta64("nat"), np.datetime64("nat")], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

    def test_is_datetimelike_array_all_nan_nat_like(self):
        arr = np.array([np.nan, pd.NaT, np.datetime64("nat")])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array([np.nan, pd.NaT, np.timedelta64("nat")])
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array([np.nan, pd.NaT, np.datetime64("nat"), np.timedelta64("nat")])
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array([np.nan, pd.NaT])
        assert lib.is_datetime_array(arr)
        assert lib.is_datetime64_array(arr)
        assert lib.is_timedelta_or_timedelta64_array(arr)

        arr = np.array([np.nan, np.nan], dtype=object)
        assert not lib.is_datetime_array(arr)
        assert not lib.is_datetime64_array(arr)
        assert not lib.is_timedelta_or_timedelta64_array(arr)

        assert lib.is_datetime_with_singletz_array(
            np.array(
                [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130102", tz="US/Eastern"),
                ],
                dtype=object,
            )
        )
        assert not lib.is_datetime_with_singletz_array(
            np.array(
                [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130102", tz="CET"),
                ],
                dtype=object,
            )
        )

    @pytest.mark.parametrize(
        "func",
        [
            "is_datetime_array",
            "is_datetime64_array",
            "is_bool_array",
            "is_timedelta_or_timedelta64_array",
            "is_date_array",
            "is_time_array",
            "is_interval_array",
        ],
    )
    def test_other_dtypes_for_array(self, func):
        func = getattr(lib, func)
        arr = np.array(["foo", "bar"])
        assert not func(arr)
        assert not func(arr.reshape(2, 1))

        arr = np.array([1, 2])
        assert not func(arr)
        assert not func(arr.reshape(2, 1))

    def test_date(self):
        dates = [date(2012, 1, day) for day in range(1, 20)]
        index = Index(dates)
        assert index.inferred_type == "date"

        dates = [date(2012, 1, day) for day in range(1, 20)] + [np.nan]
        result = lib.infer_dtype(dates, skipna=False)
        assert result == "mixed"

        result = lib.infer_dtype(dates, skipna=True)
        assert result == "date"

    @pytest.mark.parametrize(
        "values",
        [
            [date(2020, 1, 1), Timestamp("2020-01-01")],
            [Timestamp("2020-01-01"), date(2020, 1, 1)],
            [date(2020, 1, 1), pd.NaT],
            [pd.NaT, date(2020, 1, 1)],
        ],
    )
    @pytest.mark.parametrize("skipna", [True, False])
    def test_infer_dtype_date_order_invariant(self, values, skipna):
        # https://github.com/pandas-dev/pandas/issues/33741
        result = lib.infer_dtype(values, skipna=skipna)
        assert result == "date"

    def test_is_numeric_array(self):
        assert lib.is_float_array(np.array([1, 2.0]))
        assert lib.is_float_array(np.array([1, 2.0, np.nan]))
        assert not lib.is_float_array(np.array([1, 2]))

        assert lib.is_integer_array(np.array([1, 2]))
        assert not lib.is_integer_array(np.array([1, 2.0]))

    def test_is_string_array(self):
        # We should only be accepting pd.NA, np.nan,
        # other floating point nans e.g. float('nan')]
        # when skipna is True.
        assert lib.is_string_array(np.array(["foo", "bar"]))
        assert not lib.is_string_array(
            np.array(["foo", "bar", pd.NA], dtype=object), skipna=False
        )
        assert lib.is_string_array(
            np.array(["foo", "bar", pd.NA], dtype=object), skipna=True
        )
        # we allow NaN/None in the StringArray constructor, so its allowed here
        assert lib.is_string_array(
            np.array(["foo", "bar", None], dtype=object), skipna=True
        )
        assert lib.is_string_array(
            np.array(["foo", "bar", np.nan], dtype=object), skipna=True
        )
        # But not e.g. datetimelike or Decimal NAs
        assert not lib.is_string_array(
            np.array(["foo", "bar", pd.NaT], dtype=object), skipna=True
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", np.datetime64("NaT")], dtype=object), skipna=True
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", Decimal("NaN")], dtype=object), skipna=True
        )

        assert not lib.is_string_array(
            np.array(["foo", "bar", None], dtype=object), skipna=False
        )
        assert not lib.is_string_array(
            np.array(["foo", "bar", np.nan], dtype=object), skipna=False
        )
        assert not lib.is_string_array(np.array([1, 2]))

    @pytest.mark.parametrize(
        "func",
        [
            "is_bool_array",
            "is_date_array",
            "is_datetime_array",
            "is_datetime64_array",
            "is_float_array",
            "is_integer_array",
            "is_interval_array",
            "is_string_array",
            "is_time_array",
            "is_timedelta_or_timedelta64_array",
        ],
    )
    def test_is_dtype_array_empty_obj(self, func):
        # https://github.com/pandas-dev/pandas/pull/60796
        func = getattr(lib, func)

        arr = np.empty((2, 0), dtype=object)
        assert not func(arr)

        arr = np.empty((0, 2), dtype=object)
        assert not func(arr)

    def test_to_object_array_tuples(self):
        r = (5, 6)
        values = [r]
        lib.to_object_array_tuples(values)

        # make sure record array works
        record = namedtuple("record", "x y")
        r = record(5, 6)
        values = [r]
        lib.to_object_array_tuples(values)

    def test_object(self):
        # GH 7431
        # cannot infer more than this as only a single element
        arr = np.array([None], dtype="O")
        result = lib.infer_dtype(arr, skipna=False)
        assert result == "mixed"
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "empty"

    def test_to_object_array_width(self):
        # see gh-13320
        rows = [[1, 2, 3], [4, 5, 6]]

        expected = np.array(rows, dtype=object)
        out = lib.to_object_array(rows)
        tm.assert_numpy_array_equal(out, expected)

        expected = np.array(rows, dtype=object)
        out = lib.to_object_array(rows, min_width=1)
        tm.assert_numpy_array_equal(out, expected)

        expected = np.array(
            [[1, 2, 3, None, None], [4, 5, 6, None, None]], dtype=object
        )
        out = lib.to_object_array(rows, min_width=5)
        tm.assert_numpy_array_equal(out, expected)

    def test_is_period(self):
        # GH#55264
        msg = "is_period is deprecated and will be removed in a future version"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert lib.is_period(Period("2011-01", freq="M"))
            assert not lib.is_period(PeriodIndex(["2011-01"], freq="M"))
            assert not lib.is_period(Timestamp("2011-01"))
            assert not lib.is_period(1)
            assert not lib.is_period(np.nan)

    def test_is_interval(self):
        # GH#55264
        msg = "is_interval is deprecated and will be removed in a future version"
        item = Interval(1, 2)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert lib.is_interval(item)
            assert not lib.is_interval(pd.IntervalIndex([item]))
            assert not lib.is_interval(pd.IntervalIndex([item])._engine)

    def test_categorical(self):
        # GH 8974
        arr = Categorical(list("abc"))
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "categorical"

        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == "categorical"

        arr = Categorical(list("abc"), categories=["cegfab"], ordered=True)
        result = lib.infer_dtype(arr, skipna=True)
        assert result == "categorical"

        result = lib.infer_dtype(Series(arr), skipna=True)
        assert result == "categorical"

    @pytest.mark.parametrize("asobject", [True, False])
    def test_interval(self, asobject):
        idx = pd.IntervalIndex.from_breaks(range(5), closed="both")
        if asobject:
            idx = idx.astype(object)

        inferred = lib.infer_dtype(idx, skipna=False)
        assert inferred == "interval"

        inferred = lib.infer_dtype(idx._data, skipna=False)
        assert inferred == "interval"

        inferred = lib.infer_dtype(Series(idx, dtype=idx.dtype), skipna=False)
        assert inferred == "interval"

    @pytest.mark.parametrize("value", [Timestamp(0), Timedelta(0), 0, 0.0])
    def test_interval_mismatched_closed(self, value):
        first = Interval(value, value, closed="left")
        second = Interval(value, value, closed="right")

        # if closed match, we should infer "interval"
        arr = np.array([first, first], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == "interval"

        # if closed dont match, we should _not_ get "interval"
        arr2 = np.array([first, second], dtype=object)
        assert lib.infer_dtype(arr2, skipna=False) == "mixed"

    def test_interval_mismatched_subtype(self):
        first = Interval(0, 1, closed="left")
        second = Interval(Timestamp(0), Timestamp(1), closed="left")
        third = Interval(Timedelta(0), Timedelta(1), closed="left")

        arr = np.array([first, second])
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array([second, third])
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        arr = np.array([first, third])
        assert lib.infer_dtype(arr, skipna=False) == "mixed"

        # float vs int subdtype are compatible
        flt_interval = Interval(1.5, 2.5, closed="left")
        arr = np.array([first, flt_interval], dtype=object)
        assert lib.infer_dtype(arr, skipna=False) == "interval"

    @pytest.mark.parametrize("klass", [pd.array, Series])
    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("data", [["a", "b", "c"], ["a", "b", pd.NA]])
    def test_string_dtype(self, data, skipna, klass, nullable_string_dtype):
        # StringArray
        val = klass(data, dtype=nullable_string_dtype)
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == "string"

    @pytest.mark.parametrize("klass", [pd.array, Series])
    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("data", [[True, False, True], [True, False, pd.NA]])
    def test_boolean_dtype(self, data, skipna, klass):
        # BooleanArray
        val = klass(data, dtype="boolean")
        inferred = lib.infer_dtype(val, skipna=skipna)
        assert inferred == "boolean"


class TestNumberScalar:
    def test_is_number(self):
        assert is_number(True)
        assert is_number(1)
        assert is_number(1.1)
        assert is_number(1 + 3j)
        assert is_number(np.int64(1))
        assert is_number(np.float64(1.1))
        assert is_number(np.complex128(1 + 3j))
        assert is_number(np.nan)

        assert not is_number(None)
        assert not is_number("x")
        assert not is_number(datetime(2011, 1, 1))
        assert not is_number(np.datetime64("2011-01-01"))
        assert not is_number(Timestamp("2011-01-01"))
        assert not is_number(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_number(timedelta(1000))
        assert not is_number(Timedelta("1 days"))

        # questionable
        assert not is_number(np.bool_(False))
        assert is_number(np.timedelta64(1, "D"))

    def test_is_bool(self):
        assert is_bool(True)
        assert is_bool(False)
        assert is_bool(np.bool_(False))

        assert not is_bool(1)
        assert not is_bool(1.1)
        assert not is_bool(1 + 3j)
        assert not is_bool(np.int64(1))
        assert not is_bool(np.float64(1.1))
        assert not is_bool(np.complex128(1 + 3j))
        assert not is_bool(np.nan)
        assert not is_bool(None)
        assert not is_bool("x")
        assert not is_bool(datetime(2011, 1, 1))
        assert not is_bool(np.datetime64("2011-01-01"))
        assert not is_bool(Timestamp("2011-01-01"))
        assert not is_bool(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_bool(timedelta(1000))
        assert not is_bool(np.timedelta64(1, "D"))
        assert not is_bool(Timedelta("1 days"))

    def test_is_integer(self):
        assert is_integer(1)
        assert is_integer(np.int64(1))

        assert not is_integer(True)
        assert not is_integer(1.1)
        assert not is_integer(1 + 3j)
        assert not is_integer(False)
        assert not is_integer(np.bool_(False))
        assert not is_integer(np.float64(1.1))
        assert not is_integer(np.complex128(1 + 3j))
        assert not is_integer(np.nan)
        assert not is_integer(None)
        assert not is_integer("x")
        assert not is_integer(datetime(2011, 1, 1))
        assert not is_integer(np.datetime64("2011-01-01"))
        assert not is_integer(Timestamp("2011-01-01"))
        assert not is_integer(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_integer(timedelta(1000))
        assert not is_integer(Timedelta("1 days"))
        assert not is_integer(np.timedelta64(1, "D"))

    def test_is_float(self):
        assert is_float(1.1)
        assert is_float(np.float64(1.1))
        assert is_float(np.nan)

        assert not is_float(True)
        assert not is_float(1)
        assert not is_float(1 + 3j)
        assert not is_float(False)
        assert not is_float(np.bool_(False))
        assert not is_float(np.int64(1))
        assert not is_float(np.complex128(1 + 3j))
        assert not is_float(None)
        assert not is_float("x")
        assert not is_float(datetime(2011, 1, 1))
        assert not is_float(np.datetime64("2011-01-01"))
        assert not is_float(Timestamp("2011-01-01"))
        assert not is_float(Timestamp("2011-01-01", tz="US/Eastern"))
        assert not is_float(timedelta(1000))
        assert not is_float(np.timedelta64(1, "D"))
        assert not is_float(Timedelta("1 days"))

    def test_is_datetime_dtypes(self):
        ts = pd.date_range("20130101", periods=3)
        tsa = pd.date_range("20130101", periods=3, tz="US/Eastern")

        msg = "is_datetime64tz_dtype is deprecated"

        assert is_datetime64_dtype("datetime64")
        assert is_datetime64_dtype("datetime64[ns]")
        assert is_datetime64_dtype(ts)
        assert not is_datetime64_dtype(tsa)

        assert not is_datetime64_ns_dtype("datetime64")
        assert is_datetime64_ns_dtype("datetime64[ns]")
        assert is_datetime64_ns_dtype(ts)
        assert is_datetime64_ns_dtype(tsa)

        assert is_datetime64_any_dtype("datetime64")
        assert is_datetime64_any_dtype("datetime64[ns]")
        assert is_datetime64_any_dtype(ts)
        assert is_datetime64_any_dtype(tsa)

        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert not is_datetime64tz_dtype("datetime64")
            assert not is_datetime64tz_dtype("datetime64[ns]")
            assert not is_datetime64tz_dtype(ts)
            assert is_datetime64tz_dtype(tsa)

    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC"])
    def test_is_datetime_dtypes_with_tz(self, tz):
        dtype = f"datetime64[ns, {tz}]"
        assert not is_datetime64_dtype(dtype)

        msg = "is_datetime64tz_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_any_dtype(dtype)

    def test_is_timedelta(self):
        assert is_timedelta64_dtype("timedelta64")
        assert is_timedelta64_dtype("timedelta64[ns]")
        assert not is_timedelta64_ns_dtype("timedelta64")
        assert is_timedelta64_ns_dtype("timedelta64[ns]")

        tdi = TimedeltaIndex([1e14, 2e14], dtype="timedelta64[ns]")
        assert is_timedelta64_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi)
        assert is_timedelta64_ns_dtype(tdi.astype("timedelta64[ns]"))

        assert not is_timedelta64_ns_dtype(Index([], dtype=np.float64))
        assert not is_timedelta64_ns_dtype(Index([], dtype=np.int64))


class TestIsScalar:
    def test_is_scalar_builtin_scalars(self):
        assert is_scalar(None)
        assert is_scalar(True)
        assert is_scalar(False)
        assert is_scalar(Fraction())
        assert is_scalar(0.0)
        assert is_scalar(1)
        assert is_scalar(complex(2))
        assert is_scalar(float("NaN"))
        assert is_scalar(np.nan)
        assert is_scalar("foobar")
        assert is_scalar(b"foobar")
        assert is_scalar(datetime(2014, 1, 1))
        assert is_scalar(date(2014, 1, 1))
        assert is_scalar(time(12, 0))
        assert is_scalar(timedelta(hours=1))
        assert is_scalar(pd.NaT)
        assert is_scalar(pd.NA)

    def test_is_scalar_builtin_nonscalars(self):
        assert not is_scalar({})
        assert not is_scalar([])
        assert not is_scalar([1])
        assert not is_scalar(())
        assert not is_scalar((1,))
        assert not is_scalar(slice(None))
        assert not is_scalar(Ellipsis)

    def test_is_scalar_numpy_array_scalars(self):
        assert is_scalar(np.int64(1))
        assert is_scalar(np.float64(1.0))
        assert is_scalar(np.int32(1))
        assert is_scalar(np.complex64(2))
        assert is_scalar(np.object_("foobar"))
        assert is_scalar(np.str_("foobar"))
        assert is_scalar(np.bytes_(b"foobar"))
        assert is_scalar(np.datetime64("2014-01-01"))
        assert is_scalar(np.timedelta64(1, "h"))

    @pytest.mark.parametrize(
        "zerodim",
        [
            np.array(1),
            np.array("foobar"),
            np.array(np.datetime64("2014-01-01")),
            np.array(np.timedelta64(1, "h")),
            np.array(np.datetime64("NaT")),
        ],
    )
    def test_is_scalar_numpy_zerodim_arrays(self, zerodim):
        assert not is_scalar(zerodim)
        assert is_scalar(lib.item_from_zerodim(zerodim))

    @pytest.mark.parametrize("arr", [np.array([]), np.array([[]])])
    def test_is_scalar_numpy_arrays(self, arr):
        assert not is_scalar(arr)
        assert not is_scalar(MockNumpyLikeArray(arr))

    def test_is_scalar_pandas_scalars(self):
        assert is_scalar(Timestamp("2014-01-01"))
        assert is_scalar(Timedelta(hours=1))
        assert is_scalar(Period("2014-01-01"))
        assert is_scalar(Interval(left=0, right=1))
        assert is_scalar(DateOffset(days=1))
        assert is_scalar(pd.offsets.Minute(3))

    def test_is_scalar_pandas_containers(self):
        assert not is_scalar(Series(dtype=object))
        assert not is_scalar(Series([1]))
        assert not is_scalar(DataFrame())
        assert not is_scalar(DataFrame([[1]]))
        assert not is_scalar(Index([]))
        assert not is_scalar(Index([1]))
        assert not is_scalar(Categorical([]))
        assert not is_scalar(DatetimeIndex([])._data)
        assert not is_scalar(TimedeltaIndex([])._data)
        assert not is_scalar(DatetimeIndex([])._data.to_period("D"))
        assert not is_scalar(pd.array([1, 2, 3]))

    def test_is_scalar_number(self):
        # Number() is not recognied by PyNumber_Check, so by extension
        #  is not recognized by is_scalar, but instances of non-abstract
        #  subclasses are.

        class Numeric(Number):
            def __init__(self, value) -> None:
                self.value = value

            def __int__(self) -> int:
                return self.value

        num = Numeric(1)
        assert is_scalar(num)


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_datetimeindex_from_empty_datetime64_array(unit):
    idx = DatetimeIndex(np.array([], dtype=f"datetime64[{unit}]"))
    assert len(idx) == 0


def test_nan_to_nat_conversions():
    df = DataFrame(
        {"A": np.asarray(range(10), dtype="float64"), "B": Timestamp("20010101")}
    )
    df.iloc[3:6, :] = np.nan
    result = df.loc[4, "B"]
    assert result is pd.NaT

    s = df["B"].copy()
    s[8:9] = np.nan
    assert s[8] is pd.NaT


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_is_scipy_sparse(spmatrix):
    pytest.importorskip("scipy")
    assert is_scipy_sparse(spmatrix([[0, 1]]))
    assert not is_scipy_sparse(np.array([1]))


def test_ensure_int32():
    values = np.arange(10, dtype=np.int32)
    result = ensure_int32(values)
    assert result.dtype == np.int32

    values = np.arange(10, dtype=np.int64)
    result = ensure_int32(values)
    assert result.dtype == np.int32


@pytest.mark.parametrize(
    "right,result",
    [
        (0, np.uint8),
        (-1, np.int16),
        (300, np.uint16),
        # For floats, we just upcast directly to float64 instead of trying to
        # find a smaller floating dtype
        (300.0, np.uint16),  # for integer floats, we convert them to ints
        (300.1, np.float64),
        (np.int16(300), np.int16 if np_version_gt2 else np.uint16),
    ],
)
def test_find_result_type_uint_int(right, result):
    left_dtype = np.dtype("uint8")
    assert find_result_type(left_dtype, right) == result


@pytest.mark.parametrize(
    "right,result",
    [
        (0, np.int8),
        (-1, np.int8),
        (300, np.int16),
        # For floats, we just upcast directly to float64 instead of trying to
        # find a smaller floating dtype
        (300.0, np.int16),  # for integer floats, we convert them to ints
        (300.1, np.float64),
        (np.int16(300), np.int16),
    ],
)
def test_find_result_type_int_int(right, result):
    left_dtype = np.dtype("int8")
    assert find_result_type(left_dtype, right) == result


@pytest.mark.parametrize(
    "right,result",
    [
        (300.0, np.float64),
        (np.float32(300), np.float32),
    ],
)
def test_find_result_type_floats(right, result):
    left_dtype = np.dtype("float16")
    assert find_result_type(left_dtype, right) == result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_methods.py ===
import numpy as np
import pytest

from pandas.compat import HAS_PYARROW
from pandas.errors import SettingWithCopyWarning

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timestamp,
    date_range,
    option_context,
    period_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_copy(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_copy = df.copy()

    # the deep copy by defaults takes a shallow copy of the Index
    assert df_copy.index is not df.index
    assert df_copy.columns is not df.columns
    assert df_copy.index.is_(df.index)
    assert df_copy.columns.is_(df.columns)

    # the deep copy doesn't share memory
    assert not np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    if using_copy_on_write:
        assert not df_copy._mgr.blocks[0].refs.has_reference()
        assert not df_copy._mgr.blocks[1].refs.has_reference()

    # mutating copy doesn't mutate original
    df_copy.iloc[0, 0] = 0
    assert df.iloc[0, 0] == 1


def test_copy_shallow(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_copy = df.copy(deep=False)

    # the shallow copy also makes a shallow copy of the index
    if using_copy_on_write:
        assert df_copy.index is not df.index
        assert df_copy.columns is not df.columns
        assert df_copy.index.is_(df.index)
        assert df_copy.columns.is_(df.columns)
    else:
        assert df_copy.index is df.index
        assert df_copy.columns is df.columns

    # the shallow copy still shares memory
    assert np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    if using_copy_on_write:
        assert df_copy._mgr.blocks[0].refs.has_reference()
        assert df_copy._mgr.blocks[1].refs.has_reference()

    if using_copy_on_write:
        # mutating shallow copy doesn't mutate original
        df_copy.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 1
        # mutating triggered a copy-on-write -> no longer shares memory
        assert not np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
        # but still shares memory for the other columns/blocks
        assert np.shares_memory(get_array(df_copy, "c"), get_array(df, "c"))
    else:
        # mutating shallow copy does mutate original
        with tm.assert_cow_warning(warn_copy_on_write):
            df_copy.iloc[0, 0] = 0
        assert df.iloc[0, 0] == 0
        # and still shares memory
        assert np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))


@pytest.mark.parametrize("copy", [True, None, False])
@pytest.mark.parametrize(
    "method",
    [
        lambda df, copy: df.rename(columns=str.lower, copy=copy),
        lambda df, copy: df.reindex(columns=["a", "c"], copy=copy),
        lambda df, copy: df.reindex_like(df, copy=copy),
        lambda df, copy: df.align(df, copy=copy)[0],
        lambda df, copy: df.set_axis(["a", "b", "c"], axis="index", copy=copy),
        lambda df, copy: df.rename_axis(index="test", copy=copy),
        lambda df, copy: df.rename_axis(columns="test", copy=copy),
        lambda df, copy: df.astype({"b": "int64"}, copy=copy),
        # lambda df, copy: df.swaplevel(0, 0, copy=copy),
        lambda df, copy: df.swapaxes(0, 0, copy=copy),
        lambda df, copy: df.truncate(0, 5, copy=copy),
        lambda df, copy: df.infer_objects(copy=copy),
        lambda df, copy: df.to_timestamp(copy=copy),
        lambda df, copy: df.to_period(freq="D", copy=copy),
        lambda df, copy: df.tz_localize("US/Central", copy=copy),
        lambda df, copy: df.tz_convert("US/Central", copy=copy),
        lambda df, copy: df.set_flags(allows_duplicate_labels=False, copy=copy),
    ],
    ids=[
        "rename",
        "reindex",
        "reindex_like",
        "align",
        "set_axis",
        "rename_axis0",
        "rename_axis1",
        "astype",
        # "swaplevel",  # only series
        "swapaxes",
        "truncate",
        "infer_objects",
        "to_timestamp",
        "to_period",
        "tz_localize",
        "tz_convert",
        "set_flags",
    ],
)
def test_methods_copy_keyword(
    request, method, copy, using_copy_on_write, using_array_manager
):
    index = None
    if "to_timestamp" in request.node.callspec.id:
        index = period_range("2012-01-01", freq="D", periods=3)
    elif "to_period" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_localize" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_convert" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3, tz="Europe/Brussels")

    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]}, index=index)

    if "swapaxes" in request.node.callspec.id:
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df2 = method(df, copy=copy)
    else:
        df2 = method(df, copy=copy)

    share_memory = using_copy_on_write or copy is False

    if request.node.callspec.id.startswith("reindex-"):
        # TODO copy=False without CoW still returns a copy in this case
        if not using_copy_on_write and not using_array_manager and copy is False:
            share_memory = False

    if share_memory:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


@pytest.mark.parametrize("copy", [True, None, False])
@pytest.mark.parametrize(
    "method",
    [
        lambda ser, copy: ser.rename(index={0: 100}, copy=copy),
        lambda ser, copy: ser.rename(None, copy=copy),
        lambda ser, copy: ser.reindex(index=ser.index, copy=copy),
        lambda ser, copy: ser.reindex_like(ser, copy=copy),
        lambda ser, copy: ser.align(ser, copy=copy)[0],
        lambda ser, copy: ser.set_axis(["a", "b", "c"], axis="index", copy=copy),
        lambda ser, copy: ser.rename_axis(index="test", copy=copy),
        lambda ser, copy: ser.astype("int64", copy=copy),
        lambda ser, copy: ser.swaplevel(0, 1, copy=copy),
        lambda ser, copy: ser.swapaxes(0, 0, copy=copy),
        lambda ser, copy: ser.truncate(0, 5, copy=copy),
        lambda ser, copy: ser.infer_objects(copy=copy),
        lambda ser, copy: ser.to_timestamp(copy=copy),
        lambda ser, copy: ser.to_period(freq="D", copy=copy),
        lambda ser, copy: ser.tz_localize("US/Central", copy=copy),
        lambda ser, copy: ser.tz_convert("US/Central", copy=copy),
        lambda ser, copy: ser.set_flags(allows_duplicate_labels=False, copy=copy),
    ],
    ids=[
        "rename (dict)",
        "rename",
        "reindex",
        "reindex_like",
        "align",
        "set_axis",
        "rename_axis0",
        "astype",
        "swaplevel",
        "swapaxes",
        "truncate",
        "infer_objects",
        "to_timestamp",
        "to_period",
        "tz_localize",
        "tz_convert",
        "set_flags",
    ],
)
def test_methods_series_copy_keyword(request, method, copy, using_copy_on_write):
    index = None
    if "to_timestamp" in request.node.callspec.id:
        index = period_range("2012-01-01", freq="D", periods=3)
    elif "to_period" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_localize" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_convert" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3, tz="Europe/Brussels")
    elif "swaplevel" in request.node.callspec.id:
        index = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])

    ser = Series([1, 2, 3], index=index)

    if "swapaxes" in request.node.callspec.id:
        msg = "'Series.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser2 = method(ser, copy=copy)
    else:
        ser2 = method(ser, copy=copy)

    share_memory = using_copy_on_write or copy is False

    if share_memory:
        assert np.shares_memory(get_array(ser2), get_array(ser))
    else:
        assert not np.shares_memory(get_array(ser2), get_array(ser))


@pytest.mark.parametrize("copy", [True, None, False])
def test_transpose_copy_keyword(using_copy_on_write, copy, using_array_manager):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.transpose(copy=copy)
    share_memory = using_copy_on_write or copy is False or copy is None
    share_memory = share_memory and not using_array_manager

    if share_memory:
        assert np.shares_memory(get_array(df, "a"), get_array(result, 0))
    else:
        assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))


# -----------------------------------------------------------------------------
# DataFrame methods returning new DataFrame using shallow copy


def test_reset_index(using_copy_on_write):
    # Case: resetting the index (i.e. adding a new column) + mutating the
    # resulting dataframe
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]}, index=[10, 11, 12]
    )
    df_orig = df.copy()
    df2 = df.reset_index()
    df2._mgr._verify_integrity()

    if using_copy_on_write:
        # still shares memory (df2 is a shallow copy)
        assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 2] = 0
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("index", [pd.RangeIndex(0, 2), Index([1, 2])])
def test_reset_index_series_drop(using_copy_on_write, index):
    ser = Series([1, 2], index=index)
    ser_orig = ser.copy()
    ser2 = ser.reset_index(drop=True)
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(ser2))
        assert not ser._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(ser), get_array(ser2))

    ser2.iloc[0] = 100
    tm.assert_series_equal(ser, ser_orig)


def test_groupby_column_index_in_references():
    df = DataFrame(
        {"A": ["a", "b", "c", "d"], "B": [1, 2, 3, 4], "C": ["a", "a", "b", "b"]}
    )
    df = df.set_index("A")
    key = df["C"]
    result = df.groupby(key, observed=True).sum()
    expected = df.groupby("C", observed=True).sum()
    tm.assert_frame_equal(result, expected)


def test_rename_columns(using_copy_on_write):
    # Case: renaming columns returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.rename(columns=str.upper)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "C"), get_array(df, "c"))
    expected = DataFrame({"A": [0, 2, 3], "B": [4, 5, 6], "C": [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(df2, expected)
    tm.assert_frame_equal(df, df_orig)


def test_rename_columns_modify_parent(using_copy_on_write):
    # Case: renaming columns returns a new dataframe
    # + afterwards modifying the original (parent) dataframe
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df2 = df.rename(columns=str.upper)
    df2_orig = df2.copy()

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "C"), get_array(df, "c"))
    expected = DataFrame({"a": [0, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(df, expected)
    tm.assert_frame_equal(df2, df2_orig)


def test_pipe(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    df_orig = df.copy()

    def testfunc(df):
        return df

    df2 = df.pipe(testfunc)

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column
    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        expected = DataFrame({"a": [0, 2, 3], "b": 1.5})
        tm.assert_frame_equal(df, expected)

        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))


def test_pipe_modify_df(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    df_orig = df.copy()

    def testfunc(df):
        df.iloc[0, 0] = 100
        return df

    df2 = df.pipe(testfunc)

    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        expected = DataFrame({"a": [100, 2, 3], "b": 1.5})
        tm.assert_frame_equal(df, expected)

        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))


def test_reindex_columns(using_copy_on_write):
    # Case: reindexing the column returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.reindex(columns=["a", "c"])

    if using_copy_on_write:
        # still shares memory (df2 is a shallow copy)
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # mutating df2 triggers a copy-on-write for that column
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "index",
    [
        lambda idx: idx,
        lambda idx: idx.view(),
        lambda idx: idx.copy(),
        lambda idx: list(idx),
    ],
    ids=["identical", "view", "copy", "values"],
)
def test_reindex_rows(index, using_copy_on_write):
    # Case: reindexing the rows with an index that matches the current index
    # can use a shallow copy
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.reindex(index=index(df.index))

    if using_copy_on_write:
        # still shares memory (df2 is a shallow copy)
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # mutating df2 triggers a copy-on-write for that column
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


def test_drop_on_column(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.drop(columns="a")
    df2._mgr._verify_integrity()

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    else:
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        assert not np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


def test_select_dtypes(using_copy_on_write):
    # Case: selecting columns using `select_dtypes()` returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.select_dtypes("int64")
    df2._mgr._verify_integrity()

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column/block
    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "filter_kwargs", [{"items": ["a"]}, {"like": "a"}, {"regex": "a"}]
)
def test_filter(using_copy_on_write, filter_kwargs):
    # Case: selecting columns using `filter()` returns a new dataframe
    # + afterwards modifying the result
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.filter(**filter_kwargs)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column/block
    if using_copy_on_write:
        df2.iloc[0, 0] = 0
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_shift_no_op(using_copy_on_write):
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df_orig = df.copy()
    df2 = df.shift(periods=0)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
        assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    tm.assert_frame_equal(df2, df_orig)


def test_shift_index(using_copy_on_write):
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df2 = df.shift(periods=1, axis=0)

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_shift_rows_freq(using_copy_on_write):
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df_orig = df.copy()
    df_orig.index = date_range("2020-01-02", "2020-01-04")
    df2 = df.shift(periods=1, freq="1D")

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    tm.assert_frame_equal(df2, df_orig)


def test_shift_columns(using_copy_on_write, warn_copy_on_write):
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]], columns=date_range("2020-01-01", "2020-01-02")
    )
    df2 = df.shift(periods=1, axis=1)

    assert np.shares_memory(get_array(df2, "2020-01-02"), get_array(df, "2020-01-01"))
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(
            get_array(df2, "2020-01-02"), get_array(df, "2020-01-01")
        )
        expected = DataFrame(
            [[np.nan, 1], [np.nan, 3], [np.nan, 5]],
            columns=date_range("2020-01-01", "2020-01-02"),
        )
        tm.assert_frame_equal(df2, expected)


def test_pop(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    view_original = df[:]
    result = df.pop("a")

    assert np.shares_memory(result.values, get_array(view_original, "a"))
    assert np.shares_memory(get_array(df, "b"), get_array(view_original, "b"))

    if using_copy_on_write:
        result.iloc[0] = 0
        assert not np.shares_memory(result.values, get_array(view_original, "a"))
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "b"), get_array(view_original, "b"))
        tm.assert_frame_equal(view_original, df_orig)
    else:
        expected = DataFrame({"a": [1, 2, 3], "b": [0, 5, 6], "c": [0.1, 0.2, 0.3]})
        tm.assert_frame_equal(view_original, expected)


@pytest.mark.parametrize(
    "func",
    [
        lambda x, y: x.align(y),
        lambda x, y: x.align(y.a, axis=0),
        lambda x, y: x.align(y.a.iloc[slice(0, 1)], axis=1),
    ],
)
def test_align_frame(using_copy_on_write, func):
    df = DataFrame({"a": [1, 2, 3], "b": "a"})
    df_orig = df.copy()
    df_changed = df[["b", "a"]].copy()
    df2, _ = func(df, df_changed)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_align_series(using_copy_on_write):
    ser = Series([1, 2])
    ser_orig = ser.copy()
    ser_other = ser.copy()
    ser2, ser_other_result = ser.align(ser_other)

    if using_copy_on_write:
        assert np.shares_memory(ser2.values, ser.values)
        assert np.shares_memory(ser_other_result.values, ser_other.values)
    else:
        assert not np.shares_memory(ser2.values, ser.values)
        assert not np.shares_memory(ser_other_result.values, ser_other.values)

    ser2.iloc[0] = 0
    ser_other_result.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(ser2.values, ser.values)
        assert not np.shares_memory(ser_other_result.values, ser_other.values)
    tm.assert_series_equal(ser, ser_orig)
    tm.assert_series_equal(ser_other, ser_orig)


def test_align_copy_false(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_orig = df.copy()
    df2, df3 = df.align(df, copy=False)

    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    if using_copy_on_write:
        df2.loc[0, "a"] = 0
        tm.assert_frame_equal(df, df_orig)  # Original is unchanged

        df3.loc[0, "a"] = 0
        tm.assert_frame_equal(df, df_orig)  # Original is unchanged


def test_align_with_series_copy_false(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = Series([1, 2, 3], name="x")
    ser_orig = ser.copy()
    df_orig = df.copy()
    df2, ser2 = df.align(ser, copy=False, axis=0)

    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    assert np.shares_memory(get_array(ser, "x"), get_array(ser2, "x"))

    if using_copy_on_write:
        df2.loc[0, "a"] = 0
        tm.assert_frame_equal(df, df_orig)  # Original is unchanged

        ser2.loc[0] = 0
        tm.assert_series_equal(ser, ser_orig)  # Original is unchanged


def test_to_frame(using_copy_on_write, warn_copy_on_write):
    # Case: converting a Series to a DataFrame with to_frame
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()

    df = ser[:].to_frame()

    # currently this always returns a "view"
    assert np.shares_memory(ser.values, get_array(df, 0))

    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 0

    if using_copy_on_write:
        # mutating df triggers a copy-on-write for that column
        assert not np.shares_memory(ser.values, get_array(df, 0))
        tm.assert_series_equal(ser, ser_orig)
    else:
        # but currently select_dtypes() actually returns a view -> mutates parent
        expected = ser_orig.copy()
        expected.iloc[0] = 0
        tm.assert_series_equal(ser, expected)

    # modify original series -> don't modify dataframe
    df = ser[:].to_frame()
    with tm.assert_cow_warning(warn_copy_on_write):
        ser.iloc[0] = 0

    if using_copy_on_write:
        tm.assert_frame_equal(df, ser_orig.to_frame())
    else:
        expected = ser_orig.copy().to_frame()
        expected.iloc[0, 0] = 0
        tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("ax", ["index", "columns"])
def test_swapaxes_noop(using_copy_on_write, ax):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_orig = df.copy()
    msg = "'DataFrame.swapaxes' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df2 = df.swapaxes(ax, ax)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column/block
    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_swapaxes_single_block(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])
    df_orig = df.copy()
    msg = "'DataFrame.swapaxes' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df2 = df.swapaxes("index", "columns")

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "x"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "x"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column/block
    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "x"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_swapaxes_read_only_array():
    df = DataFrame({"a": [1, 2], "b": 3})
    msg = "'DataFrame.swapaxes' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df = df.swapaxes(axis1="index", axis2="columns")
    df.iloc[0, 0] = 100
    expected = DataFrame({0: [100, 3], 1: [2, 3]}, index=["a", "b"])
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "method, idx",
    [
        (lambda df: df.copy(deep=False).copy(deep=False), 0),
        (lambda df: df.reset_index().reset_index(), 2),
        (lambda df: df.rename(columns=str.upper).rename(columns=str.lower), 0),
        (lambda df: df.copy(deep=False).select_dtypes(include="number"), 0),
    ],
    ids=["shallow-copy", "reset_index", "rename", "select_dtypes"],
)
def test_chained_methods(request, method, idx, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    # when not using CoW, only the copy() variant actually gives a view
    df2_is_view = not using_copy_on_write and request.node.callspec.id == "shallow-copy"

    # modify df2 -> don't modify df
    df2 = method(df)
    with tm.assert_cow_warning(warn_copy_on_write and df2_is_view):
        df2.iloc[0, idx] = 0
    if not df2_is_view:
        tm.assert_frame_equal(df, df_orig)

    # modify df -> don't modify df2
    df2 = method(df)
    with tm.assert_cow_warning(warn_copy_on_write and df2_is_view):
        df.iloc[0, 0] = 0
    if not df2_is_view:
        tm.assert_frame_equal(df2.iloc[:, idx:], df_orig)


@pytest.mark.parametrize("obj", [Series([1, 2], name="a"), DataFrame({"a": [1, 2]})])
def test_to_timestamp(using_copy_on_write, obj):
    obj.index = Index([Period("2012-1-1", freq="D"), Period("2012-1-2", freq="D")])

    obj_orig = obj.copy()
    obj2 = obj.to_timestamp()

    if using_copy_on_write:
        assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    else:
        assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # mutating obj2 triggers a copy-on-write for that column / block
    obj2.iloc[0] = 0
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize("obj", [Series([1, 2], name="a"), DataFrame({"a": [1, 2]})])
def test_to_period(using_copy_on_write, obj):
    obj.index = Index([Timestamp("2019-12-31"), Timestamp("2020-12-31")])

    obj_orig = obj.copy()
    obj2 = obj.to_period(freq="Y")

    if using_copy_on_write:
        assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    else:
        assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # mutating obj2 triggers a copy-on-write for that column / block
    obj2.iloc[0] = 0
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


def test_set_index(using_copy_on_write):
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.set_index("a")

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    else:
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 1] = 0
    assert not np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    tm.assert_frame_equal(df, df_orig)


def test_set_index_mutating_parent_does_not_mutate_index():
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    result = df.set_index("a")
    expected = result.copy()

    df.iloc[0, 0] = 100
    tm.assert_frame_equal(result, expected)


def test_add_prefix(using_copy_on_write):
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.add_prefix("CoW_")

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "CoW_a"), get_array(df, "a"))
    df2.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(df2, "CoW_a"), get_array(df, "a"))

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "CoW_c"), get_array(df, "c"))
    expected = DataFrame(
        {"CoW_a": [0, 2, 3], "CoW_b": [4, 5, 6], "CoW_c": [0.1, 0.2, 0.3]}
    )
    tm.assert_frame_equal(df2, expected)
    tm.assert_frame_equal(df, df_orig)


def test_add_suffix(using_copy_on_write):
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.add_suffix("_CoW")
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a_CoW"), get_array(df, "a"))
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a_CoW"), get_array(df, "a"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "c_CoW"), get_array(df, "c"))
    expected = DataFrame(
        {"a_CoW": [0, 2, 3], "b_CoW": [4, 5, 6], "c_CoW": [0.1, 0.2, 0.3]}
    )
    tm.assert_frame_equal(df2, expected)
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("axis, val", [(0, 5.5), (1, np.nan)])
def test_dropna(using_copy_on_write, axis, val):
    df = DataFrame({"a": [1, 2, 3], "b": [4, val, 6], "c": "d"})
    df_orig = df.copy()
    df2 = df.dropna(axis=axis)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("val", [5, 5.5])
def test_dropna_series(using_copy_on_write, val):
    ser = Series([1, val, 4])
    ser_orig = ser.copy()
    ser2 = ser.dropna()

    if using_copy_on_write:
        assert np.shares_memory(ser2.values, ser.values)
    else:
        assert not np.shares_memory(ser2.values, ser.values)

    ser2.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df.head(),
        lambda df: df.head(2),
        lambda df: df.tail(),
        lambda df: df.tail(3),
    ],
)
def test_head_tail(method, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = method(df)
    df2._mgr._verify_integrity()

    if using_copy_on_write:
        # We are explicitly deviating for CoW here to make an eager copy (avoids
        # tracking references for very cheap ops)
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # modify df2 to trigger CoW for that block
    with tm.assert_cow_warning(warn_copy_on_write):
        df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        # without CoW enabled, head and tail return views. Mutating df2 also mutates df.
        assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        with tm.assert_cow_warning(warn_copy_on_write):
            df2.iloc[0, 0] = 1
    tm.assert_frame_equal(df, df_orig)


def test_infer_objects(using_copy_on_write, using_infer_string):
    df = DataFrame(
        {"a": [1, 2], "b": Series(["x", "y"], dtype=object), "c": 1, "d": "x"}
    )
    df_orig = df.copy()
    df2 = df.infer_objects()

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
        if using_infer_string:
            assert not tm.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        else:
            assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    df2.iloc[0, 0] = 0
    df2.iloc[0, 1] = "d"
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    tm.assert_frame_equal(df, df_orig)


def test_infer_objects_no_reference(using_copy_on_write, using_infer_string):
    df = DataFrame(
        {
            "a": [1, 2],
            "b": Series(["x", "y"], dtype=object),
            "c": 1,
            "d": Series(
                [Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype="object"
            ),
            "e": Series(["z", "w"], dtype=object),
        }
    )
    df = df.infer_objects()

    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")
    arr_d = get_array(df, "d")

    df.iloc[0, 0] = 0
    df.iloc[0, 1] = "d"
    df.iloc[0, 3] = Timestamp("2018-12-31")
    if using_copy_on_write:
        assert np.shares_memory(arr_a, get_array(df, "a"))
        if using_infer_string:
            # note that the underlying memory of arr_b has been copied anyway
            # because of the assignment, but the EA is updated inplace so still
            # appears the share memory
            assert tm.shares_memory(arr_b, get_array(df, "b"))
        else:
            # TODO(CoW): Block splitting causes references here
            assert not np.shares_memory(arr_b, get_array(df, "b"))
        assert np.shares_memory(arr_d, get_array(df, "d"))


def test_infer_objects_reference(using_copy_on_write, using_infer_string):
    df = DataFrame(
        {
            "a": [1, 2],
            "b": Series(["x", "y"], dtype=object),
            "c": 1,
            "d": Series(
                [Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype="object"
            ),
        }
    )
    view = df[:]  # noqa: F841
    df = df.infer_objects()

    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")
    arr_d = get_array(df, "d")

    df.iloc[0, 0] = 0
    df.iloc[0, 1] = "d"
    df.iloc[0, 3] = Timestamp("2018-12-31")
    if using_copy_on_write:
        assert not np.shares_memory(arr_a, get_array(df, "a"))
        if not using_infer_string or HAS_PYARROW:
            assert not np.shares_memory(arr_b, get_array(df, "b"))
        assert np.shares_memory(arr_d, get_array(df, "d"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"before": "a", "after": "b", "axis": 1},
        {"before": 0, "after": 1, "axis": 0},
    ],
)
def test_truncate(using_copy_on_write, kwargs):
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2})
    df_orig = df.copy()
    df2 = df.truncate(**kwargs)
    df2._mgr._verify_integrity()

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("method", ["assign", "drop_duplicates"])
def test_assign_drop_duplicates(using_copy_on_write, method):
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()
    df2 = getattr(df, method)()
    df2._mgr._verify_integrity()

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("obj", [Series([1, 2]), DataFrame({"a": [1, 2]})])
def test_take(using_copy_on_write, obj):
    # Check that no copy is made when we take all rows in original order
    obj_orig = obj.copy()
    obj2 = obj.take([0, 1])

    if using_copy_on_write:
        assert np.shares_memory(obj2.values, obj.values)
    else:
        assert not np.shares_memory(obj2.values, obj.values)

    obj2.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(obj2.values, obj.values)
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize("obj", [Series([1, 2]), DataFrame({"a": [1, 2]})])
def test_between_time(using_copy_on_write, obj):
    obj.index = date_range("2018-04-09", periods=2, freq="1D20min")
    obj_orig = obj.copy()
    obj2 = obj.between_time("0:00", "1:00")

    if using_copy_on_write:
        assert np.shares_memory(obj2.values, obj.values)
    else:
        assert not np.shares_memory(obj2.values, obj.values)

    obj2.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(obj2.values, obj.values)
    tm.assert_equal(obj, obj_orig)


def test_reindex_like(using_copy_on_write):
    df = DataFrame({"a": [1, 2], "b": "a"})
    other = DataFrame({"b": "a", "a": [1, 2]})

    df_orig = df.copy()
    df2 = df.reindex_like(other)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 1] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_sort_index(using_copy_on_write):
    # GH 49473
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    ser2 = ser.sort_index()

    if using_copy_on_write:
        assert np.shares_memory(ser.values, ser2.values)
    else:
        assert not np.shares_memory(ser.values, ser2.values)

    # mutating ser triggers a copy-on-write for the column / block
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize(
    "obj, kwargs",
    [(Series([1, 2, 3], name="a"), {}), (DataFrame({"a": [1, 2, 3]}), {"by": "a"})],
)
def test_sort_values(using_copy_on_write, obj, kwargs):
    obj_orig = obj.copy()
    obj2 = obj.sort_values(**kwargs)

    if using_copy_on_write:
        assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    else:
        assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # mutating df triggers a copy-on-write for the column / block
    obj2.iloc[0] = 0
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize(
    "obj, kwargs",
    [(Series([1, 2, 3], name="a"), {}), (DataFrame({"a": [1, 2, 3]}), {"by": "a"})],
)
def test_sort_values_inplace(using_copy_on_write, obj, kwargs, warn_copy_on_write):
    obj_orig = obj.copy()
    view = obj[:]
    obj.sort_values(inplace=True, **kwargs)

    assert np.shares_memory(get_array(obj, "a"), get_array(view, "a"))

    # mutating obj triggers a copy-on-write for the column / block
    with tm.assert_cow_warning(warn_copy_on_write):
        obj.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(obj, "a"), get_array(view, "a"))
        tm.assert_equal(view, obj_orig)
    else:
        assert np.shares_memory(get_array(obj, "a"), get_array(view, "a"))


@pytest.mark.parametrize("decimals", [-1, 0, 1])
def test_round(using_copy_on_write, warn_copy_on_write, decimals):
    df = DataFrame({"a": [1, 2], "b": "c"})
    df_orig = df.copy()
    df2 = df.round(decimals=decimals)

    if using_copy_on_write:
        assert tm.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        # TODO: Make inplace by using out parameter of ndarray.round?
        if decimals >= 0:
            # Ensure lazy copy if no-op
            assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
        else:
            assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 1] = "d"
    df2.iloc[0, 0] = 4
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_reorder_levels(using_copy_on_write):
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=["one", "two"]
    )
    df = DataFrame({"a": [1, 2, 3, 4]}, index=index)
    df_orig = df.copy()
    df2 = df.reorder_levels(order=["two", "one"])

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_series_reorder_levels(using_copy_on_write):
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=["one", "two"]
    )
    ser = Series([1, 2, 3, 4], index=index)
    ser_orig = ser.copy()
    ser2 = ser.reorder_levels(order=["two", "one"])

    if using_copy_on_write:
        assert np.shares_memory(ser2.values, ser.values)
    else:
        assert not np.shares_memory(ser2.values, ser.values)

    ser2.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("obj", [Series([1, 2, 3]), DataFrame({"a": [1, 2, 3]})])
def test_swaplevel(using_copy_on_write, obj):
    index = MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1)], names=["one", "two"])
    obj.index = index
    obj_orig = obj.copy()
    obj2 = obj.swaplevel()

    if using_copy_on_write:
        assert np.shares_memory(obj2.values, obj.values)
    else:
        assert not np.shares_memory(obj2.values, obj.values)

    obj2.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(obj2.values, obj.values)
    tm.assert_equal(obj, obj_orig)


def test_frame_set_axis(using_copy_on_write):
    # GH 49473
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2 = df.set_axis(["a", "b", "c"], axis="index")

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_series_set_axis(using_copy_on_write):
    # GH 49473
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    ser2 = ser.set_axis(["a", "b", "c"], axis="index")

    if using_copy_on_write:
        assert np.shares_memory(ser, ser2)
    else:
        assert not np.shares_memory(ser, ser2)

    # mutating ser triggers a copy-on-write for the column / block
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2, ser)
    tm.assert_series_equal(ser, ser_orig)


def test_set_flags(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    ser2 = ser.set_flags(allows_duplicate_labels=False)

    assert np.shares_memory(ser, ser2)

    # mutating ser triggers a copy-on-write for the column / block
    with tm.assert_cow_warning(warn_copy_on_write):
        ser2.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(ser2, ser)
        tm.assert_series_equal(ser, ser_orig)
    else:
        assert np.shares_memory(ser2, ser)
        expected = Series([0, 2, 3])
        tm.assert_series_equal(ser, expected)


@pytest.mark.parametrize("kwargs", [{"mapper": "test"}, {"index": "test"}])
def test_rename_axis(using_copy_on_write, kwargs):
    df = DataFrame({"a": [1, 2, 3, 4]}, index=Index([1, 2, 3, 4], name="a"))
    df_orig = df.copy()
    df2 = df.rename_axis(**kwargs)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df2.iloc[0, 0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "func, tz", [("tz_convert", "Europe/Berlin"), ("tz_localize", None)]
)
def test_tz_convert_localize(using_copy_on_write, func, tz):
    # GH 49473
    ser = Series(
        [1, 2], index=date_range(start="2014-08-01 09:00", freq="h", periods=2, tz=tz)
    )
    ser_orig = ser.copy()
    ser2 = getattr(ser, func)("US/Central")

    if using_copy_on_write:
        assert np.shares_memory(ser.values, ser2.values)
    else:
        assert not np.shares_memory(ser.values, ser2.values)

    # mutating ser triggers a copy-on-write for the column / block
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    tm.assert_series_equal(ser, ser_orig)


def test_droplevel(using_copy_on_write):
    # GH 49473
    index = MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1)], names=["one", "two"])
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=index)
    df_orig = df.copy()
    df2 = df.droplevel(0)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    tm.assert_frame_equal(df, df_orig)


def test_squeeze(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]})
    df_orig = df.copy()
    series = df.squeeze()

    # Should share memory regardless of CoW since squeeze is just an iloc
    assert np.shares_memory(series.values, get_array(df, "a"))

    # mutating squeezed df triggers a copy-on-write for that column/block
    with tm.assert_cow_warning(warn_copy_on_write):
        series.iloc[0] = 0
    if using_copy_on_write:
        assert not np.shares_memory(series.values, get_array(df, "a"))
        tm.assert_frame_equal(df, df_orig)
    else:
        # Without CoW the original will be modified
        assert np.shares_memory(series.values, get_array(df, "a"))
        assert df.loc[0, "a"] == 0


def test_items(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()

    # Test this twice, since the second time, the item cache will be
    # triggered, and we want to make sure it still works then.
    for i in range(2):
        for name, ser in df.items():
            assert np.shares_memory(get_array(ser, name), get_array(df, name))

            # mutating df triggers a copy-on-write for that column / block
            with tm.assert_cow_warning(warn_copy_on_write):
                ser.iloc[0] = 0

            if using_copy_on_write:
                assert not np.shares_memory(get_array(ser, name), get_array(df, name))
                tm.assert_frame_equal(df, df_orig)
            else:
                # Original frame will be modified
                assert df.loc[0, name] == 0


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_putmask(using_copy_on_write, dtype, warn_copy_on_write):
    df = DataFrame({"a": [1, 2], "b": 1, "c": 2}, dtype=dtype)
    view = df[:]
    df_orig = df.copy()
    with tm.assert_cow_warning(warn_copy_on_write):
        df[df == df] = 5

    if using_copy_on_write:
        assert not np.shares_memory(get_array(view, "a"), get_array(df, "a"))
        tm.assert_frame_equal(view, df_orig)
    else:
        # Without CoW the original will be modified
        assert np.shares_memory(get_array(view, "a"), get_array(df, "a"))
        assert view.iloc[0, 0] == 5


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_putmask_no_reference(using_copy_on_write, dtype):
    df = DataFrame({"a": [1, 2], "b": 1, "c": 2}, dtype=dtype)
    arr_a = get_array(df, "a")
    df[df == df] = 5

    if using_copy_on_write:
        assert np.shares_memory(arr_a, get_array(df, "a"))


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_putmask_aligns_rhs_no_reference(using_copy_on_write, dtype):
    df = DataFrame({"a": [1.5, 2], "b": 1.5}, dtype=dtype)
    arr_a = get_array(df, "a")
    df[df == df] = DataFrame({"a": [5.5, 5]})

    if using_copy_on_write:
        assert np.shares_memory(arr_a, get_array(df, "a"))


@pytest.mark.parametrize(
    "val, exp, warn", [(5.5, True, FutureWarning), (5, False, None)]
)
def test_putmask_dont_copy_some_blocks(
    using_copy_on_write, val, exp, warn, warn_copy_on_write
):
    df = DataFrame({"a": [1, 2], "b": 1, "c": 1.5})
    view = df[:]
    df_orig = df.copy()
    indexer = DataFrame(
        [[True, False, False], [True, False, False]], columns=list("abc")
    )
    if warn_copy_on_write:
        with tm.assert_cow_warning():
            df[indexer] = val
    else:
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            df[indexer] = val

    if using_copy_on_write:
        assert not np.shares_memory(get_array(view, "a"), get_array(df, "a"))
        # TODO(CoW): Could split blocks to avoid copying the whole block
        assert np.shares_memory(get_array(view, "b"), get_array(df, "b")) is exp
        assert np.shares_memory(get_array(view, "c"), get_array(df, "c"))
        assert df._mgr._has_no_reference(1) is not exp
        assert not df._mgr._has_no_reference(2)
        tm.assert_frame_equal(view, df_orig)
    elif val == 5:
        # Without CoW the original will be modified, the other case upcasts, e.g. copy
        assert np.shares_memory(get_array(view, "a"), get_array(df, "a"))
        assert np.shares_memory(get_array(view, "c"), get_array(df, "c"))
        assert view.iloc[0, 0] == 5


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize(
    "func",
    [
        lambda ser: ser.where(ser > 0, 10),
        lambda ser: ser.mask(ser <= 0, 10),
    ],
)
def test_where_mask_noop(using_copy_on_write, dtype, func):
    ser = Series([1, 2, 3], dtype=dtype)
    ser_orig = ser.copy()

    result = func(ser)

    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(result))
    else:
        assert not np.shares_memory(get_array(ser), get_array(result))

    result.iloc[0] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), get_array(result))
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize(
    "func",
    [
        lambda ser: ser.where(ser < 0, 10),
        lambda ser: ser.mask(ser >= 0, 10),
    ],
)
def test_where_mask(using_copy_on_write, dtype, func):
    ser = Series([1, 2, 3], dtype=dtype)
    ser_orig = ser.copy()

    result = func(ser)

    assert not np.shares_memory(get_array(ser), get_array(result))
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("dtype, val", [("int64", 10.5), ("Int64", 10)])
@pytest.mark.parametrize(
    "func",
    [
        lambda df, val: df.where(df < 0, val),
        lambda df, val: df.mask(df >= 0, val),
    ],
)
def test_where_mask_noop_on_single_column(using_copy_on_write, dtype, val, func):
    df = DataFrame({"a": [1, 2, 3], "b": [-4, -5, -6]}, dtype=dtype)
    df_orig = df.copy()

    result = func(df, val)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "b"), get_array(result, "b"))
        assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    else:
        assert not np.shares_memory(get_array(df, "b"), get_array(result, "b"))

    result.iloc[0, 1] = 10
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "b"), get_array(result, "b"))
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("func", ["mask", "where"])
def test_chained_where_mask(using_copy_on_write, func):
    df = DataFrame({"a": [1, 4, 2], "b": 1})
    df_orig = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            getattr(df["a"], func)(df["a"] > 2, 5, inplace=True)
        tm.assert_frame_equal(df, df_orig)

        with tm.raises_chained_assignment_error():
            getattr(df[["a"]], func)(df["a"] > 2, 5, inplace=True)
        tm.assert_frame_equal(df, df_orig)
    else:
        with tm.assert_produces_warning(FutureWarning, match="inplace method"):
            getattr(df["a"], func)(df["a"] > 2, 5, inplace=True)

        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                getattr(df[["a"]], func)(df["a"] > 2, 5, inplace=True)

        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                getattr(df[df["a"] > 1], func)(df["a"] > 2, 5, inplace=True)


def test_asfreq_noop(using_copy_on_write):
    df = DataFrame(
        {"a": [0.0, None, 2.0, 3.0]},
        index=date_range("1/1/2000", periods=4, freq="min"),
    )
    df_orig = df.copy()
    df2 = df.asfreq(freq="min")

    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # mutating df2 triggers a copy-on-write for that column / block
    df2.iloc[0, 0] = 0

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_iterrows(using_copy_on_write):
    df = DataFrame({"a": 0, "b": 1}, index=[1, 2, 3])
    df_orig = df.copy()

    for _, sub in df.iterrows():
        sub.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)


def test_interpolate_creates_copy(using_copy_on_write, warn_copy_on_write):
    # GH#51126
    df = DataFrame({"a": [1.5, np.nan, 3]})
    view = df[:]
    expected = df.copy()

    with tm.assert_cow_warning(warn_copy_on_write):
        df.ffill(inplace=True)
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100.5

    if using_copy_on_write:
        tm.assert_frame_equal(view, expected)
    else:
        expected = DataFrame({"a": [100.5, 1.5, 3]})
        tm.assert_frame_equal(view, expected)


def test_isetitem(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()
    df2 = df.copy(deep=None)  # Trigger a CoW
    df2.isetitem(1, np.array([-1, -2, -3]))  # This is inplace

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
        assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    else:
        assert not np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    else:
        assert not np.shares_memory(get_array(df, "c"), get_array(df2, "c"))


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_isetitem_series(using_copy_on_write, dtype):
    df = DataFrame({"a": [1, 2, 3], "b": np.array([4, 5, 6], dtype=dtype)})
    ser = Series([7, 8, 9])
    ser_orig = ser.copy()
    df.isetitem(0, ser)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "a"), get_array(ser))
        assert not df._mgr._has_no_reference(0)

    # mutating dataframe doesn't update series
    df.loc[0, "a"] = 0
    tm.assert_series_equal(ser, ser_orig)

    # mutating series doesn't update dataframe
    df = DataFrame({"a": [1, 2, 3], "b": np.array([4, 5, 6], dtype=dtype)})
    ser = Series([7, 8, 9])
    df.isetitem(0, ser)

    ser.loc[0] = 0
    expected = DataFrame({"a": [7, 8, 9], "b": np.array([4, 5, 6], dtype=dtype)})
    tm.assert_frame_equal(df, expected)


def test_isetitem_frame(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2})
    rhs = DataFrame({"a": [4, 5, 6], "b": 2})
    df.isetitem([0, 1], rhs)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "a"), get_array(rhs, "a"))
        assert np.shares_memory(get_array(df, "b"), get_array(rhs, "b"))
        assert not df._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(df, "a"), get_array(rhs, "a"))
        assert not np.shares_memory(get_array(df, "b"), get_array(rhs, "b"))
    expected = df.copy()
    rhs.iloc[0, 0] = 100
    rhs.iloc[0, 1] = 100
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("key", ["a", ["a"]])
def test_get(using_copy_on_write, warn_copy_on_write, key):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df_orig = df.copy()

    result = df.get(key)

    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
        result.iloc[0] = 0
        assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
        tm.assert_frame_equal(df, df_orig)
    else:
        # for non-CoW it depends on whether we got a Series or DataFrame if it
        # is a view or copy or triggers a warning or not
        if warn_copy_on_write:
            warn = FutureWarning if isinstance(key, str) else None
        else:
            warn = SettingWithCopyWarning if isinstance(key, list) else None
        with option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(warn):
                result.iloc[0] = 0

        if isinstance(key, list):
            tm.assert_frame_equal(df, df_orig)
        else:
            assert df.iloc[0, 0] == 0


@pytest.mark.parametrize("axis, key", [(0, 0), (1, "a")])
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_xs(
    using_copy_on_write, warn_copy_on_write, using_array_manager, axis, key, dtype
):
    single_block = (dtype == "int64") and not using_array_manager
    is_view = single_block or (using_array_manager and axis == 1)
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    df_orig = df.copy()

    result = df.xs(key, axis=axis)

    if axis == 1 or single_block:
        assert np.shares_memory(get_array(df, "a"), get_array(result))
    elif using_copy_on_write:
        assert result._mgr._has_no_reference(0)

    if using_copy_on_write or (is_view and not warn_copy_on_write):
        result.iloc[0] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning(single_block or axis == 1):
            result.iloc[0] = 0
    else:
        with option_context("chained_assignment", "warn"):
            with tm.assert_produces_warning(SettingWithCopyWarning):
                result.iloc[0] = 0

    if using_copy_on_write or (not single_block and axis == 0):
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.iloc[0, 0] == 0


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("key, level", [("l1", 0), (2, 1)])
def test_xs_multiindex(
    using_copy_on_write, warn_copy_on_write, using_array_manager, key, level, axis
):
    arr = np.arange(18).reshape(6, 3)
    index = MultiIndex.from_product([["l1", "l2"], [1, 2, 3]], names=["lev1", "lev2"])
    df = DataFrame(arr, index=index, columns=list("abc"))
    if axis == 1:
        df = df.transpose().copy()
    df_orig = df.copy()

    result = df.xs(key, level=level, axis=axis)

    if level == 0:
        assert np.shares_memory(
            get_array(df, df.columns[0]), get_array(result, result.columns[0])
        )

    if warn_copy_on_write:
        warn = FutureWarning if level == 0 else None
    elif not using_copy_on_write and not using_array_manager:
        warn = SettingWithCopyWarning
    else:
        warn = None
    with option_context("chained_assignment", "warn"):
        with tm.assert_produces_warning(warn):
            result.iloc[0, 0] = 0

    tm.assert_frame_equal(df, df_orig)


def test_update_frame(using_copy_on_write, warn_copy_on_write):
    df1 = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df2 = DataFrame({"b": [100.0]}, index=[1])
    df1_orig = df1.copy()
    view = df1[:]

    # TODO(CoW) better warning message?
    with tm.assert_cow_warning(warn_copy_on_write):
        df1.update(df2)

    expected = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 100.0, 6.0]})
    tm.assert_frame_equal(df1, expected)
    if using_copy_on_write:
        # df1 is updated, but its view not
        tm.assert_frame_equal(view, df1_orig)
        assert np.shares_memory(get_array(df1, "a"), get_array(view, "a"))
        assert not np.shares_memory(get_array(df1, "b"), get_array(view, "b"))
    else:
        tm.assert_frame_equal(view, expected)


def test_update_series(using_copy_on_write, warn_copy_on_write):
    ser1 = Series([1.0, 2.0, 3.0])
    ser2 = Series([100.0], index=[1])
    ser1_orig = ser1.copy()
    view = ser1[:]

    if warn_copy_on_write:
        with tm.assert_cow_warning():
            ser1.update(ser2)
    else:
        ser1.update(ser2)

    expected = Series([1.0, 100.0, 3.0])
    tm.assert_series_equal(ser1, expected)
    if using_copy_on_write:
        # ser1 is updated, but its view not
        tm.assert_series_equal(view, ser1_orig)
    else:
        tm.assert_series_equal(view, expected)


def test_update_chained_assignment(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]})
    ser2 = Series([100.0], index=[1])
    df_orig = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["a"].update(ser2)
        tm.assert_frame_equal(df, df_orig)

        with tm.raises_chained_assignment_error():
            df[["a"]].update(ser2.to_frame())
        tm.assert_frame_equal(df, df_orig)
    else:
        with tm.assert_produces_warning(FutureWarning, match="inplace method"):
            df["a"].update(ser2)

        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                df[["a"]].update(ser2.to_frame())

        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                df[df["a"] > 1].update(ser2.to_frame())


def test_inplace_arithmetic_series(using_copy_on_write):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    data = get_array(ser)
    ser *= 2
    if using_copy_on_write:
        # https://github.com/pandas-dev/pandas/pull/55745
        # changed to NOT update inplace because there is no benefit (actual
        # operation already done non-inplace). This was only for the optics
        # of updating the backing array inplace, but we no longer want to make
        # that guarantee
        assert not np.shares_memory(get_array(ser), data)
        tm.assert_numpy_array_equal(data, get_array(ser_orig))
    else:
        assert np.shares_memory(get_array(ser), data)
        tm.assert_numpy_array_equal(data, get_array(ser))


def test_inplace_arithmetic_series_with_reference(
    using_copy_on_write, warn_copy_on_write
):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    view = ser[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        ser *= 2
    if using_copy_on_write:
        assert not np.shares_memory(get_array(ser), get_array(view))
        tm.assert_series_equal(ser_orig, view)
    else:
        assert np.shares_memory(get_array(ser), get_array(view))


@pytest.mark.parametrize("copy", [True, False])
def test_transpose(using_copy_on_write, copy, using_array_manager):
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()
    result = df.transpose(copy=copy)

    if not copy and not using_array_manager or using_copy_on_write:
        assert np.shares_memory(get_array(df, "a"), get_array(result, 0))
    else:
        assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))

    result.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)


def test_transpose_different_dtypes(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    df_orig = df.copy()
    result = df.T

    assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))
    result.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)


def test_transpose_ea_single_column(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    result = df.T

    assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))


def test_transform_frame(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()

    def func(ser):
        ser.iloc[0] = 100
        return ser

    with tm.assert_cow_warning(warn_copy_on_write):
        df.transform(func)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)


def test_transform_series(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()

    def func(ser):
        ser.iloc[0] = 100
        return ser

    with tm.assert_cow_warning(warn_copy_on_write):
        ser.transform(func)
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)


def test_count_read_only_array():
    df = DataFrame({"a": [1, 2], "b": 3})
    result = df.count()
    result.iloc[0] = 100
    expected = Series([100, 2], index=["a", "b"])
    tm.assert_series_equal(result, expected)


def test_series_view(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()

    with tm.assert_produces_warning(FutureWarning, match="is deprecated"):
        ser2 = ser.view()
    assert np.shares_memory(get_array(ser), get_array(ser2))
    if using_copy_on_write:
        assert not ser2._mgr._has_no_reference(0)

    with tm.assert_cow_warning(warn_copy_on_write):
        ser2.iloc[0] = 100

    if using_copy_on_write:
        tm.assert_series_equal(ser_orig, ser)
    else:
        expected = Series([100, 2, 3])
        tm.assert_series_equal(ser, expected)


def test_insert_series(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]})
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()
    df.insert(loc=1, value=ser, column="b")
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(df, "b"))
        assert not df._mgr._has_no_reference(1)
    else:
        assert not np.shares_memory(get_array(ser), get_array(df, "b"))

    df.iloc[0, 1] = 100
    tm.assert_series_equal(ser, ser_orig)


def test_eval(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()

    result = df.eval("c = a+b")
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    else:
        assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))

    result.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df_orig)


def test_eval_inplace(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    df_orig = df.copy()
    df_view = df[:]

    df.eval("c = a+b", inplace=True)
    assert np.shares_memory(get_array(df, "a"), get_array(df_view, "a"))

    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100
    if using_copy_on_write:
        tm.assert_frame_equal(df_view, df_orig)


def test_apply_modify_row(using_copy_on_write, warn_copy_on_write):
    # Case: applying a function on each row as a Series object, where the
    # function mutates the row object (which needs to trigger CoW if row is a view)
    df = DataFrame({"A": [1, 2], "B": [3, 4]})
    df_orig = df.copy()

    def transform(row):
        row["B"] = 100
        return row

    with tm.assert_cow_warning(warn_copy_on_write):
        df.apply(transform, axis=1)

    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        assert df.loc[0, "B"] == 100

    # row Series is a copy
    df = DataFrame({"A": [1, 2], "B": ["b", "c"]})
    df_orig = df.copy()

    with tm.assert_produces_warning(None):
        df.apply(transform, axis=1)

    tm.assert_frame_equal(df, df_orig)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_algos.py ===
from datetime import datetime
import struct

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas._libs import (
    algos as libalgos,
    hashtable as ht,
)

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_complex_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
import pandas.core.common as com


class TestFactorize:
    def test_factorize_complex(self):
        # GH#17927
        array = [1, 2, 2 + 1j]
        msg = "factorize with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            labels, uniques = algos.factorize(array)

        expected_labels = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(labels, expected_labels)

        # Should return a complex dtype in the future
        expected_uniques = np.array([(1 + 0j), (2 + 0j), (2 + 1j)], dtype=object)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)", strict=False)
    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize(self, index_or_series_obj, sort):
        obj = index_or_series_obj
        result_codes, result_uniques = obj.factorize(sort=sort)

        constructor = Index
        if isinstance(obj, MultiIndex):
            constructor = MultiIndex.from_tuples
        expected_arr = obj.unique()
        if expected_arr.dtype == np.float16:
            expected_arr = expected_arr.astype(np.float32)
        expected_uniques = constructor(expected_arr)
        if (
            isinstance(obj, Index)
            and expected_uniques.dtype == bool
            and obj.dtype == object
        ):
            expected_uniques = expected_uniques.astype(object)

        if sort:
            expected_uniques = expected_uniques.sort_values()

        # construct an integer ndarray so that
        # `expected_uniques.take(expected_codes)` is equal to `obj`
        expected_uniques_list = list(expected_uniques)
        expected_codes = [expected_uniques_list.index(val) for val in obj]
        expected_codes = np.asarray(expected_codes, dtype=np.intp)

        tm.assert_numpy_array_equal(result_codes, expected_codes)
        tm.assert_index_equal(result_uniques, expected_uniques, exact=True)

    def test_series_factorize_use_na_sentinel_false(self):
        # GH#35667
        values = np.array([1, 2, 1, np.nan])
        ser = Series(values)
        codes, uniques = ser.factorize(use_na_sentinel=False)

        expected_codes = np.array([0, 1, 0, 2], dtype=np.intp)
        expected_uniques = Index([1.0, 2.0, np.nan])

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_index_equal(uniques, expected_uniques)

    def test_basic(self):
        items = np.array(["a", "b", "b", "a", "a", "c", "c", "c"], dtype=object)
        codes, uniques = algos.factorize(items)
        tm.assert_numpy_array_equal(uniques, np.array(["a", "b", "c"], dtype=object))

        codes, uniques = algos.factorize(items, sort=True)
        exp = np.array([0, 1, 1, 0, 0, 2, 2, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(uniques, exp)

        arr = np.arange(5, dtype=np.intp)[::-1]

        codes, uniques = algos.factorize(arr)
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([4, 3, 2, 1, 0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        codes, uniques = algos.factorize(arr, sort=True)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([0, 1, 2, 3, 4], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        arr = np.arange(5.0)[::-1]

        codes, uniques = algos.factorize(arr)
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

        codes, uniques = algos.factorize(arr, sort=True)
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=arr.dtype)
        tm.assert_numpy_array_equal(uniques, exp)

    def test_mixed(self):
        # doc example reshaping.rst
        x = Series(["A", "A", np.nan, "B", 3.14, np.inf])
        codes, uniques = algos.factorize(x)

        exp = np.array([0, 0, -1, 1, 2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = Index(["A", "B", 3.14, np.inf])
        tm.assert_index_equal(uniques, exp)

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([2, 2, -1, 3, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = Index([3.14, np.inf, "A", "B"])
        tm.assert_index_equal(uniques, exp)

    def test_factorize_datetime64(self):
        # M8
        v1 = Timestamp("20130101 09:00:00.00004")
        v2 = Timestamp("20130101")
        x = Series([v1, v1, v1, v2, v2, v1])
        codes, uniques = algos.factorize(x)

        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = DatetimeIndex([v1, v2])
        tm.assert_index_equal(uniques, exp)

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        exp = DatetimeIndex([v2, v1])
        tm.assert_index_equal(uniques, exp)

    def test_factorize_period(self):
        # period
        v1 = Period("201302", freq="M")
        v2 = Period("201303", freq="M")
        x = Series([v1, v1, v1, v2, v2, v1])

        # periods are not 'sorted' as they are converted back into an index
        codes, uniques = algos.factorize(x)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

    def test_factorize_timedelta(self):
        # GH 5986
        v1 = to_timedelta("1 day 1 min")
        v2 = to_timedelta("1 day")
        x = Series([v1, v2, v1, v1, v2, v2, v1])
        codes, uniques = algos.factorize(x)
        exp = np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, to_timedelta([v1, v2]))

        codes, uniques = algos.factorize(x, sort=True)
        exp = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, exp)
        tm.assert_index_equal(uniques, to_timedelta([v2, v1]))

    def test_factorize_nan(self):
        # nan should map to na_sentinel, not reverse_indexer[na_sentinel]
        # rizer.factorize should not raise an exception if na_sentinel indexes
        # outside of reverse_indexer
        key = np.array([1, 2, 1, np.nan], dtype="O")
        rizer = ht.ObjectFactorizer(len(key))
        for na_sentinel in (-1, 20):
            ids = rizer.factorize(key, na_sentinel=na_sentinel)
            expected = np.array([0, 1, 0, na_sentinel], dtype=np.intp)
            assert len(set(key)) == len(set(expected))
            tm.assert_numpy_array_equal(pd.isna(key), expected == na_sentinel)
            tm.assert_numpy_array_equal(ids, expected)

    def test_factorizer_with_mask(self):
        # GH#49549
        data = np.array([1, 2, 3, 1, 1, 0], dtype="int64")
        mask = np.array([False, False, False, False, False, True])
        rizer = ht.Int64Factorizer(len(data))
        result = rizer.factorize(data, mask=mask)
        expected = np.array([0, 1, 2, 0, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        expected_uniques = np.array([1, 2, 3], dtype="int64")
        tm.assert_numpy_array_equal(rizer.uniques.to_array(), expected_uniques)

    def test_factorizer_object_with_nan(self):
        # GH#49549
        data = np.array([1, 2, 3, 1, np.nan])
        rizer = ht.ObjectFactorizer(len(data))
        result = rizer.factorize(data.astype(object))
        expected = np.array([0, 1, 2, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        expected_uniques = np.array([1, 2, 3], dtype=object)
        tm.assert_numpy_array_equal(rizer.uniques.to_array(), expected_uniques)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), "nonsense"],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), "nonsense"],
            ),
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), (1, 2, 3)],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), (1, 2, 3)],
            ),
            ([(1, 1), (1, 2), (0, 0), (1, 2)], [0, 1, 2, 1], [(1, 1), (1, 2), (0, 0)]),
        ],
    )
    def test_factorize_tuple_list(self, data, expected_codes, expected_uniques):
        # GH9454
        msg = "factorize with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            codes, uniques = pd.factorize(data)

        tm.assert_numpy_array_equal(codes, np.array(expected_codes, dtype=np.intp))

        expected_uniques_array = com.asarray_tuplesafe(expected_uniques, dtype=object)
        tm.assert_numpy_array_equal(uniques, expected_uniques_array)

    def test_complex_sorting(self):
        # gh 12666 - check no segfault
        x17 = np.array([complex(i) for i in range(17)], dtype=object)

        msg = "'[<>]' not supported between instances of .*"
        with pytest.raises(TypeError, match=msg):
            algos.factorize(x17[::-1], sort=True)

    def test_numeric_dtype_factorize(self, any_real_numpy_dtype):
        # GH41132
        dtype = any_real_numpy_dtype
        data = np.array([1, 2, 2, 1], dtype=dtype)
        expected_codes = np.array([0, 1, 1, 0], dtype=np.intp)
        expected_uniques = np.array([1, 2], dtype=dtype)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_float64_factorize(self, writable):
        data = np.array([1.0, 1e8, 1.0, 1e-8, 1e8, 1.0], dtype=np.float64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        expected_uniques = np.array([1.0, 1e8, 1e-8], dtype=np.float64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_uint64_factorize(self, writable):
        data = np.array([2**64 - 1, 1, 2**64 - 1], dtype=np.uint64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        expected_uniques = np.array([2**64 - 1, 1], dtype=np.uint64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_int64_factorize(self, writable):
        data = np.array([2**63 - 1, -(2**63), 2**63 - 1], dtype=np.int64)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        expected_uniques = np.array([2**63 - 1, -(2**63)], dtype=np.int64)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_string_factorize(self, writable):
        data = np.array(["a", "c", "a", "b", "c"], dtype=object)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, 0, 2, 1], dtype=np.intp)
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_object_factorize(self, writable):
        data = np.array(["a", "c", None, np.nan, "a", "b", NaT, "c"], dtype=object)
        data.setflags(write=writable)
        expected_codes = np.array([0, 1, -1, -1, 0, 2, -1, 1], dtype=np.intp)
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_datetime64_factorize(self, writable):
        # GH35650 Verify whether read-only datetime64 array can be factorized
        data = np.array([np.datetime64("2020-01-01T00:00:00.000")], dtype="M8[ns]")
        data.setflags(write=writable)
        expected_codes = np.array([0], dtype=np.intp)
        expected_uniques = np.array(
            ["2020-01-01T00:00:00.000000000"], dtype="datetime64[ns]"
        )

        codes, uniques = pd.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize_rangeindex(self, sort):
        # increasing -> sort doesn't matter
        ri = pd.RangeIndex.from_range(range(10))
        expected = np.arange(10, dtype=np.intp), ri

        result = algos.factorize(ri, sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

        result = ri.factorize(sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize_rangeindex_decreasing(self, sort):
        # decreasing -> sort matters
        ri = pd.RangeIndex.from_range(range(10))
        expected = np.arange(10, dtype=np.intp), ri

        ri2 = ri[::-1]
        expected = expected[0], ri2
        if sort:
            expected = expected[0][::-1], expected[1][::-1]

        result = algos.factorize(ri2, sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

        result = ri2.factorize(sort=sort)
        tm.assert_numpy_array_equal(result[0], expected[0])
        tm.assert_index_equal(result[1], expected[1], exact=True)

    def test_deprecate_order(self):
        # gh 19727 - check warning is raised for deprecated keyword, order.
        # Test not valid once order keyword is removed.
        data = np.array([2**63, 1, 2**63], dtype=np.uint64)
        with pytest.raises(TypeError, match="got an unexpected keyword"):
            algos.factorize(data, order=True)
        with tm.assert_produces_warning(False):
            algos.factorize(data)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([0, 1, 0], dtype="u8"),
            np.array([-(2**63), 1, -(2**63)], dtype="i8"),
            np.array(["__nan__", "foo", "__nan__"], dtype="object"),
        ],
    )
    def test_parametrized_factorize_na_value_default(self, data):
        # arrays that include the NA default for that type, but isn't used.
        codes, uniques = algos.factorize(data)
        expected_uniques = data[[0, 1]]
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "data, na_value",
        [
            (np.array([0, 1, 0, 2], dtype="u8"), 0),
            (np.array([1, 0, 1, 2], dtype="u8"), 1),
            (np.array([-(2**63), 1, -(2**63), 0], dtype="i8"), -(2**63)),
            (np.array([1, -(2**63), 1, 0], dtype="i8"), 1),
            (np.array(["a", "", "a", "b"], dtype=object), "a"),
            (np.array([(), ("a", 1), (), ("a", 2)], dtype=object), ()),
            (np.array([("a", 1), (), ("a", 1), ("a", 2)], dtype=object), ("a", 1)),
        ],
    )
    def test_parametrized_factorize_na_value(self, data, na_value):
        codes, uniques = algos.factorize_array(data, na_value=na_value)
        expected_uniques = data[[1, 3]]
        expected_codes = np.array([-1, 0, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize("sort", [True, False])
    @pytest.mark.parametrize(
        "data, uniques",
        [
            (
                np.array(["b", "a", None, "b"], dtype=object),
                np.array(["b", "a"], dtype=object),
            ),
            (
                pd.array([2, 1, np.nan, 2], dtype="Int64"),
                pd.array([2, 1], dtype="Int64"),
            ),
        ],
        ids=["numpy_array", "extension_array"],
    )
    def test_factorize_use_na_sentinel(self, sort, data, uniques):
        codes, uniques = algos.factorize(data, sort=sort, use_na_sentinel=True)
        if sort:
            expected_codes = np.array([1, 0, -1, 1], dtype=np.intp)
            expected_uniques = algos.safe_sort(uniques)
        else:
            expected_codes = np.array([0, 1, -1, 0], dtype=np.intp)
            expected_uniques = uniques
        tm.assert_numpy_array_equal(codes, expected_codes)
        if isinstance(data, np.ndarray):
            tm.assert_numpy_array_equal(uniques, expected_uniques)
        else:
            tm.assert_extension_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                ["a", None, "b", "a"],
                np.array([0, 1, 2, 0], dtype=np.dtype("intp")),
                np.array(["a", np.nan, "b"], dtype=object),
            ),
            (
                ["a", np.nan, "b", "a"],
                np.array([0, 1, 2, 0], dtype=np.dtype("intp")),
                np.array(["a", np.nan, "b"], dtype=object),
            ),
        ],
    )
    def test_object_factorize_use_na_sentinel_false(
        self, data, expected_codes, expected_uniques
    ):
        codes, uniques = algos.factorize(
            np.array(data, dtype=object), use_na_sentinel=False
        )

        tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)
        tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                [1, None, 1, 2],
                np.array([0, 1, 0, 2], dtype=np.dtype("intp")),
                np.array([1, np.nan, 2], dtype="O"),
            ),
            (
                [1, np.nan, 1, 2],
                np.array([0, 1, 0, 2], dtype=np.dtype("intp")),
                np.array([1, np.nan, 2], dtype=np.float64),
            ),
        ],
    )
    def test_int_factorize_use_na_sentinel_false(
        self, data, expected_codes, expected_uniques
    ):
        msg = "factorize with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            codes, uniques = algos.factorize(data, use_na_sentinel=False)

        tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)
        tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                Index(Categorical(["a", "a", "b"])),
                np.array([0, 0, 1], dtype=np.intp),
                CategoricalIndex(["a", "b"], categories=["a", "b"], dtype="category"),
            ),
            (
                Series(Categorical(["a", "a", "b"])),
                np.array([0, 0, 1], dtype=np.intp),
                CategoricalIndex(["a", "b"], categories=["a", "b"], dtype="category"),
            ),
            (
                Series(DatetimeIndex(["2017", "2017"], tz="US/Eastern")),
                np.array([0, 0], dtype=np.intp),
                DatetimeIndex(["2017"], tz="US/Eastern"),
            ),
        ],
    )
    def test_factorize_mixed_values(self, data, expected_codes, expected_uniques):
        # GH 19721
        codes, uniques = algos.factorize(data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_index_equal(uniques, expected_uniques)

    def test_factorize_interval_non_nano(self, unit):
        # GH#56099
        left = DatetimeIndex(["2016-01-01", np.nan, "2015-10-11"]).as_unit(unit)
        right = DatetimeIndex(["2016-01-02", np.nan, "2015-10-15"]).as_unit(unit)
        idx = IntervalIndex.from_arrays(left, right)
        codes, cats = idx.factorize()
        assert cats.dtype == f"interval[datetime64[{unit}], right]"

        ts = Timestamp(0).as_unit(unit)
        idx2 = IntervalIndex.from_arrays(left - ts, right - ts)
        codes2, cats2 = idx2.factorize()
        assert cats2.dtype == f"interval[timedelta64[{unit}], right]"

        idx3 = IntervalIndex.from_arrays(
            left.tz_localize("US/Pacific"), right.tz_localize("US/Pacific")
        )
        codes3, cats3 = idx3.factorize()
        assert cats3.dtype == f"interval[datetime64[{unit}, US/Pacific], right]"


class TestUnique:
    def test_ints(self):
        arr = np.random.default_rng(2).integers(0, 100, size=50)

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_objects(self):
        arr = np.random.default_rng(2).integers(0, 100, size=50).astype("O")

        result = algos.unique(arr)
        assert isinstance(result, np.ndarray)

    def test_object_refcount_bug(self):
        lst = np.array(["A", "B", "C", "D", "E"], dtype=object)
        for i in range(1000):
            len(algos.unique(lst))

    def test_on_index_object(self):
        mindex = MultiIndex.from_arrays(
            [np.arange(5).repeat(5), np.tile(np.arange(5), 5)]
        )
        expected = mindex.values
        expected.sort()

        mindex = mindex.repeat(2)

        result = pd.unique(mindex)
        result.sort()

        tm.assert_almost_equal(result, expected)

    def test_dtype_preservation(self, any_numpy_dtype):
        # GH 15442
        if any_numpy_dtype in (tm.BYTES_DTYPES + tm.STRING_DTYPES):
            data = [1, 2, 2]
            uniques = [1, 2]
        elif is_integer_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1, 2]
        elif is_float_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1.0, 2.0]
        elif is_complex_dtype(any_numpy_dtype):
            data = [complex(1, 0), complex(2, 0), complex(2, 0)]
            uniques = [complex(1, 0), complex(2, 0)]
        elif is_bool_dtype(any_numpy_dtype):
            data = [True, True, False]
            uniques = [True, False]
        elif is_object_dtype(any_numpy_dtype):
            data = ["A", "B", "B"]
            uniques = ["A", "B"]
        else:
            # datetime64[ns]/M8[ns]/timedelta64[ns]/m8[ns] tested elsewhere
            data = [1, 2, 2]
            uniques = [1, 2]

        result = Series(data, dtype=any_numpy_dtype).unique()
        expected = np.array(uniques, dtype=any_numpy_dtype)

        if any_numpy_dtype in tm.STRING_DTYPES:
            expected = expected.astype(object)

        if expected.dtype.kind in ["m", "M"]:
            # We get TimedeltaArray/DatetimeArray
            assert isinstance(result, (DatetimeArray, TimedeltaArray))
            result = np.array(result)
        tm.assert_numpy_array_equal(result, expected)

    def test_datetime64_dtype_array_returned(self):
        # GH 9431
        expected = np.array(
            [
                "2015-01-03T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
            ],
            dtype="M8[ns]",
        )

        dt_index = to_datetime(
            [
                "2015-01-03T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
            ]
        )
        result = algos.unique(dt_index)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        s = Series(dt_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_datetime_non_ns(self):
        a = np.array(["2000", "2000", "2001"], dtype="datetime64[s]")
        result = pd.unique(a)
        expected = np.array(["2000", "2001"], dtype="datetime64[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta_non_ns(self):
        a = np.array(["2000", "2000", "2001"], dtype="timedelta64[s]")
        result = pd.unique(a)
        expected = np.array([2000, 2001], dtype="timedelta64[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta64_dtype_array_returned(self):
        # GH 9431
        expected = np.array([31200, 45678, 10000], dtype="m8[ns]")

        td_index = to_timedelta([31200, 45678, 31200, 10000, 45678])
        result = algos.unique(td_index)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        s = Series(td_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_uint64_overflow(self):
        s = Series([1, 2, 2**63, 2**63], dtype=np.uint64)
        exp = np.array([1, 2, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(algos.unique(s), exp)

    def test_nan_in_object_array(self):
        duplicated_items = ["a", np.nan, "c", "c"]
        result = pd.unique(np.array(duplicated_items, dtype=object))
        expected = np.array(["a", np.nan, "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_categorical(self):
        # we are expecting to return in the order
        # of appearance
        expected = Categorical(list("bac"))

        # we are expecting to return in the order
        # of the categories
        expected_o = Categorical(list("bac"), categories=list("abc"), ordered=True)

        # GH 15939
        c = Categorical(list("baabc"))
        result = c.unique()
        tm.assert_categorical_equal(result, expected)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected)

        c = Categorical(list("baabc"), ordered=True)
        result = c.unique()
        tm.assert_categorical_equal(result, expected_o)

        result = algos.unique(c)
        tm.assert_categorical_equal(result, expected_o)

        # Series of categorical dtype
        s = Series(Categorical(list("baabc")), name="foo")
        result = s.unique()
        tm.assert_categorical_equal(result, expected)

        result = pd.unique(s)
        tm.assert_categorical_equal(result, expected)

        # CI -> return CI
        ci = CategoricalIndex(Categorical(list("baabc"), categories=list("abc")))
        expected = CategoricalIndex(expected)
        result = ci.unique()
        tm.assert_index_equal(result, expected)

        result = pd.unique(ci)
        tm.assert_index_equal(result, expected)

    def test_datetime64tz_aware(self, unit):
        # GH 15939

        dti = Index(
            [
                Timestamp("20160101", tz="US/Eastern"),
                Timestamp("20160101", tz="US/Eastern"),
            ]
        ).as_unit(unit)
        ser = Series(dti)

        result = ser.unique()
        expected = dti[:1]._data
        tm.assert_extension_array_equal(result, expected)

        result = dti.unique()
        expected = dti[:1]
        tm.assert_index_equal(result, expected)

        result = pd.unique(ser)
        expected = dti[:1]._data
        tm.assert_extension_array_equal(result, expected)

        result = pd.unique(dti)
        expected = dti[:1]
        tm.assert_index_equal(result, expected)

    def test_order_of_appearance(self):
        # 9346
        # light testing of guarantee of order of appearance
        # these also are the doc-examples
        result = pd.unique(Series([2, 1, 3, 3]))
        tm.assert_numpy_array_equal(result, np.array([2, 1, 3], dtype="int64"))

        result = pd.unique(Series([2] + [1] * 5))
        tm.assert_numpy_array_equal(result, np.array([2, 1], dtype="int64"))

        msg = "unique with argument that is not not a Series, Index,"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pd.unique(list("aabc"))
        expected = np.array(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = pd.unique(Series(Categorical(list("aabc"))))
        expected = Categorical(list("abc"))
        tm.assert_categorical_equal(result, expected)

    def test_order_of_appearance_dt64(self, unit):
        ser = Series([Timestamp("20160101"), Timestamp("20160101")]).dt.as_unit(unit)
        result = pd.unique(ser)
        expected = np.array(["2016-01-01T00:00:00.000000000"], dtype=f"M8[{unit}]")
        tm.assert_numpy_array_equal(result, expected)

    def test_order_of_appearance_dt64tz(self, unit):
        dti = DatetimeIndex(
            [
                Timestamp("20160101", tz="US/Eastern"),
                Timestamp("20160101", tz="US/Eastern"),
            ]
        ).as_unit(unit)
        result = pd.unique(dti)
        expected = DatetimeIndex(
            ["2016-01-01 00:00:00"], dtype=f"datetime64[{unit}, US/Eastern]", freq=None
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "arg ,expected",
        [
            (("1", "1", "2"), np.array(["1", "2"], dtype=object)),
            (("foo",), np.array(["foo"], dtype=object)),
        ],
    )
    def test_tuple_with_strings(self, arg, expected):
        # see GH 17108
        msg = "unique with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pd.unique(arg)
        tm.assert_numpy_array_equal(result, expected)

    def test_obj_none_preservation(self):
        # GH 20866
        arr = np.array(["foo", None], dtype=object)
        result = pd.unique(arr)
        expected = np.array(["foo", None], dtype=object)

        tm.assert_numpy_array_equal(result, expected, strict_nan=True)

    def test_signed_zero(self):
        # GH 21866
        a = np.array([-0.0, 0.0])
        result = pd.unique(a)
        expected = np.array([-0.0])  # 0.0 and -0.0 are equivalent
        tm.assert_numpy_array_equal(result, expected)

    def test_different_nans(self):
        # GH 21866
        # create different nans from bit-patterns:
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        a = np.array([NAN1, NAN2])  # NAN1 and NAN2 are equivalent
        result = pd.unique(a)
        expected = np.array([np.nan])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("el_type", [np.float64, object])
    def test_first_nan_kept(self, el_type):
        # GH 22295
        # create different nans from bit-patterns:
        bits_for_nan1 = 0xFFF8000000000001
        bits_for_nan2 = 0x7FF8000000000001
        NAN1 = struct.unpack("d", struct.pack("=Q", bits_for_nan1))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", bits_for_nan2))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        a = np.array([NAN1, NAN2], dtype=el_type)
        result = pd.unique(a)
        assert result.size == 1
        # use bit patterns to identify which nan was kept:
        result_nan_bits = struct.unpack("=Q", struct.pack("d", result[0]))[0]
        assert result_nan_bits == bits_for_nan1

    def test_do_not_mangle_na_values(self, unique_nulls_fixture, unique_nulls_fixture2):
        # GH 22295
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # skip it, values not unique
        a = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
        result = pd.unique(a)
        assert result.size == 2
        assert a[0] is unique_nulls_fixture
        assert a[1] is unique_nulls_fixture2

    def test_unique_masked(self, any_numeric_ea_dtype):
        # GH#48019
        ser = Series([1, pd.NA, 2] * 3, dtype=any_numeric_ea_dtype)
        result = pd.unique(ser)
        expected = pd.array([1, pd.NA, 2], dtype=any_numeric_ea_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_nunique_ints(index_or_series_or_array):
    # GH#36327
    values = index_or_series_or_array(np.random.default_rng(2).integers(0, 20, 30))
    result = algos.nunique_ints(values)
    expected = len(algos.unique(values))
    assert result == expected


class TestIsin:
    def test_invalid(self):
        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a `int`"
        )
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, 1)
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, [1])
        with pytest.raises(TypeError, match=msg):
            algos.isin([1], 1)

    def test_basic(self):
        msg = "isin with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin([1, 2], [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(np.array([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), Series([1]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series([1, 2]), {1})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(["a", "b"], ["a"])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(["a", "b"]), Series(["a"]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(Series(["a", "b"]), {"a"})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(["a", "b"], [1])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_i8(self):
        arr = date_range("20130101", periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        arr = timedelta_range("1 day", periods=3).values
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype1", ["m8[ns]", "M8[ns]", "M8[ns, UTC]", "period[D]"])
    @pytest.mark.parametrize("dtype", ["i8", "f8", "u8"])
    def test_isin_datetimelike_values_numeric_comps(self, dtype, dtype1):
        # Anything but object and we get all-False shortcut

        dta = date_range("2013-01-01", periods=3)._values
        arr = Series(dta.view("i8")).array.view(dtype1)

        comps = arr.view("i8").astype(dtype)

        result = algos.isin(comps, arr)
        expected = np.zeros(comps.shape, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_large(self):
        s = date_range("20000101", periods=2000000, freq="s").values
        result = algos.isin(s, s[0:2])
        expected = np.zeros(len(s), dtype=bool)
        expected[0] = True
        expected[1] = True
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]", "M8[ns, UTC]", "period[D]"])
    def test_isin_datetimelike_all_nat(self, dtype):
        # GH#56427
        dta = date_range("2013-01-01", periods=3)._values
        arr = Series(dta.view("i8")).array.view(dtype)

        arr[0] = NaT
        result = algos.isin(arr, [NaT])
        expected = np.array([True, False, False], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]", "M8[ns, UTC]"])
    def test_isin_datetimelike_strings_deprecated(self, dtype):
        # GH#53111
        dta = date_range("2013-01-01", periods=3)._values
        arr = Series(dta.view("i8")).array.view(dtype)

        vals = [str(x) for x in arr]
        msg = "The behavior of 'isin' with dtype=.* is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = algos.isin(arr, vals)
        assert res.all()

        vals2 = np.array(vals, dtype=str)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res2 = algos.isin(arr, vals2)
        assert res2.all()

    def test_isin_dt64tz_with_nat(self):
        # the all-NaT values used to get inferred to tznaive, which was evaluated
        #  as non-matching GH#56427
        dti = date_range("2016-01-01", periods=3, tz="UTC")
        ser = Series(dti)
        ser[0] = NaT

        res = algos.isin(ser._values, [NaT])
        exp = np.array([True, False, False], dtype=bool)
        tm.assert_numpy_array_equal(res, exp)

    def test_categorical_from_codes(self):
        # GH 16639
        vals = np.array([0, 1, 2, 0])
        cats = ["a", "b", "c"]
        Sd = Series(Categorical([1]).from_codes(vals, cats))
        St = Series(Categorical([1]).from_codes(np.array([0, 1]), cats))
        expected = np.array([True, True, False, True])
        result = algos.isin(Sd, St)
        tm.assert_numpy_array_equal(expected, result)

    def test_categorical_isin(self):
        vals = np.array([0, 1, 2, 0])
        cats = ["a", "b", "c"]
        cat = Categorical([1]).from_codes(vals, cats)
        other = Categorical([1]).from_codes(np.array([0, 1]), cats)

        expected = np.array([True, True, False, True])
        result = algos.isin(cat, other)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in(self):
        # GH 22160
        # nan is special, because from " a is b" doesn't follow "a == b"
        # at least, isin() should follow python's "np.nan in [nan] == True"
        # casting to -> np.float64 -> another float-object somewhere on
        # the way could lead jeopardize this behavior
        comps = [np.nan]  # could be casted to float64
        values = [np.nan]
        expected = np.array([True])
        msg = "isin with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in_large(self):
        # https://github.com/pandas-dev/pandas/issues/22205
        s = np.tile(1.0, 1_000_001)
        s[0] = np.nan
        result = algos.isin(s, np.array([np.nan, 1]))
        expected = np.ones(len(s), dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    def test_same_nan_is_in_large_series(self):
        # https://github.com/pandas-dev/pandas/issues/22205
        s = np.tile(1.0, 1_000_001)
        series = Series(s)
        s[0] = np.nan
        result = series.isin(np.array([np.nan, 1]))
        expected = Series(np.ones(len(s), dtype=bool))
        tm.assert_series_equal(result, expected)

    def test_same_object_is_in(self):
        # GH 22160
        # there could be special treatment for nans
        # the user however could define a custom class
        # with similar behavior, then we at least should
        # fall back to usual python's behavior: "a in [a] == True"
        class LikeNan:
            def __eq__(self, other) -> bool:
                return False

            def __hash__(self):
                return 0

        a, b = LikeNan(), LikeNan()

        msg = "isin with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # same object -> True
            tm.assert_numpy_array_equal(algos.isin([a], [a]), np.array([True]))
            # different objects -> False
            tm.assert_numpy_array_equal(algos.isin([a], [b]), np.array([False]))

    def test_different_nans(self):
        # GH 22160
        # all nans are handled as equivalent

        comps = [float("nan")]
        values = [float("nan")]
        assert comps[0] is not values[0]  # different nan-objects

        # as list of python-objects:
        result = algos.isin(np.array(comps), values)
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as object-array:
        result = algos.isin(
            np.asarray(comps, dtype=object), np.asarray(values, dtype=object)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

        # as float64-array:
        result = algos.isin(
            np.asarray(comps, dtype=np.float64), np.asarray(values, dtype=np.float64)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

    def test_no_cast(self):
        # GH 22160
        # ensure 42 is not casted to a string
        comps = ["ss", 42]
        values = ["42"]
        expected = np.array([False, False])
        msg = "isin with argument that is not not a Series, Index"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_empty(self, empty):
        # see gh-16991
        vals = Index(["a", "b"])
        expected = np.array([False, False])

        result = algos.isin(vals, empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nan_objects(self):
        # GH 22119
        comps = np.array(["nan", np.nan * 1j, float("nan")], dtype=object)
        vals = np.array([float("nan")], dtype=object)
        expected = np.array([False, False, True])
        result = algos.isin(comps, vals)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nans_as_float64(self):
        # GH 21866
        # create different nans from bit-patterns,
        # these nans will land in different buckets in the hash-table
        # if no special care is taken
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2

        # check that NAN1 and NAN2 are equivalent:
        arr = np.array([NAN1, NAN2], dtype=np.float64)
        lookup1 = np.array([NAN1], dtype=np.float64)
        result = algos.isin(arr, lookup1)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        lookup2 = np.array([NAN2], dtype=np.float64)
        result = algos.isin(arr, lookup2)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_isin_int_df_string_search(self):
        """Comparing df with int`s (1,2) with a string at isin() ("1")
        -> should not match values because int 1 is not equal str 1"""
        df = DataFrame({"values": [1, 2]})
        result = df.isin(["1"])
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_nan_df_string_search(self):
        """Comparing df with nan value (np.nan,2) with a string at isin() ("NaN")
        -> should not match values because np.nan is not equal str NaN"""
        df = DataFrame({"values": [np.nan, 2]})
        result = df.isin(np.array(["NaN"], dtype=object))
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_float_df_string_search(self):
        """Comparing df with floats (1.4245,2.32441) with a string at isin() ("1.4245")
        -> should not match values because float 1.4245 is not equal str 1.4245"""
        df = DataFrame({"values": [1.4245, 2.32441]})
        result = df.isin(np.array(["1.4245"], dtype=object))
        expected_false = DataFrame({"values": [False, False]})
        tm.assert_frame_equal(result, expected_false)

    def test_isin_unsigned_dtype(self):
        # GH#46485
        ser = Series([1378774140726870442], dtype=np.uint64)
        result = ser.isin([1378774140726870528])
        expected = Series(False)
        tm.assert_series_equal(result, expected)


class TestValueCounts:
    def test_value_counts(self):
        arr = np.random.default_rng(1234).standard_normal(4)
        factor = cut(arr, 4)

        # assert isinstance(factor, n)
        msg = "pandas.value_counts is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.value_counts(factor)
        breaks = [-1.606, -1.018, -0.431, 0.155, 0.741]
        index = IntervalIndex.from_breaks(breaks).astype(CategoricalDtype(ordered=True))
        expected = Series([1, 0, 2, 1], index=index, name="count")
        tm.assert_series_equal(result.sort_index(), expected.sort_index())

    def test_value_counts_bins(self):
        s = [1, 2, 3, 4]
        msg = "pandas.value_counts is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.value_counts(s, bins=1)
        expected = Series(
            [4], index=IntervalIndex.from_tuples([(0.996, 4.0)]), name="count"
        )
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.value_counts(s, bins=2, sort=False)
        expected = Series(
            [2, 2],
            index=IntervalIndex.from_tuples([(0.996, 2.5), (2.5, 4.0)]),
            name="count",
        )
        tm.assert_series_equal(result, expected)

    def test_value_counts_dtypes(self):
        msg2 = "pandas.value_counts is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            result = algos.value_counts(np.array([1, 1.0]))
        assert len(result) == 1

        with tm.assert_produces_warning(FutureWarning, match=msg2):
            result = algos.value_counts(np.array([1, 1.0]), bins=1)
        assert len(result) == 1

        with tm.assert_produces_warning(FutureWarning, match=msg2):
            result = algos.value_counts(Series([1, 1.0, "1"]))  # object
        assert len(result) == 2

        msg = "bins argument only works with numeric data"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                algos.value_counts(np.array(["1", 1], dtype=object), bins=1)

    def test_value_counts_nat(self):
        td = Series([np.timedelta64(10000), NaT], dtype="timedelta64[ns]")
        dt = to_datetime(["NaT", "2014-01-01"])

        msg = "pandas.value_counts is deprecated"

        for ser in [td, dt]:
            with tm.assert_produces_warning(FutureWarning, match=msg):
                vc = algos.value_counts(ser)
                vc_with_na = algos.value_counts(ser, dropna=False)
            assert len(vc) == 1
            assert len(vc_with_na) == 2

        exp_dt = Series({Timestamp("2014-01-01 00:00:00"): 1}, name="count")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result_dt = algos.value_counts(dt)
        tm.assert_series_equal(result_dt, exp_dt)

        exp_td = Series([1], index=[np.timedelta64(10000)], name="count")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result_td = algos.value_counts(td)
        tm.assert_series_equal(result_td, exp_td)

    @pytest.mark.parametrize("dtype", [object, "M8[us]"])
    def test_value_counts_datetime_outofbounds(self, dtype):
        # GH 13663
        ser = Series(
            [
                datetime(3000, 1, 1),
                datetime(5000, 1, 1),
                datetime(5000, 1, 1),
                datetime(6000, 1, 1),
                datetime(3000, 1, 1),
                datetime(3000, 1, 1),
            ],
            dtype=dtype,
        )
        res = ser.value_counts()

        exp_index = Index(
            [datetime(3000, 1, 1), datetime(5000, 1, 1), datetime(6000, 1, 1)],
            dtype=dtype,
        )
        exp = Series([3, 2, 1], index=exp_index, name="count")
        tm.assert_series_equal(res, exp)

    def test_categorical(self):
        s = Series(Categorical(list("aaabbc")))
        result = s.value_counts()
        expected = Series(
            [3, 2, 1], index=CategoricalIndex(["a", "b", "c"]), name="count"
        )

        tm.assert_series_equal(result, expected, check_index_type=True)

        # preserve order?
        s = s.cat.as_ordered()
        result = s.value_counts()
        expected.index = expected.index.as_ordered()
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_nans(self):
        s = Series(Categorical(list("aaaaabbbcc")))  # 4,3,2,1 (nan)
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c"]),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)
        result = s.value_counts(dropna=False)
        expected = Series(
            [4, 3, 2, 1], index=CategoricalIndex(["a", "b", "c", np.nan]), name="count"
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

        # out of order
        s = Series(
            Categorical(list("aaaaabbbcc"), ordered=True, categories=["b", "a", "c"])
        )
        s.iloc[1] = np.nan
        result = s.value_counts()
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(
                ["a", "b", "c"],
                categories=["b", "a", "c"],
                ordered=True,
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

        result = s.value_counts(dropna=False)
        expected = Series(
            [4, 3, 2, 1],
            index=CategoricalIndex(
                ["a", "b", "c", np.nan], categories=["b", "a", "c"], ordered=True
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_categorical_zeroes(self):
        # keep the `d` category with 0
        s = Series(Categorical(list("bbbaac"), categories=list("abcd"), ordered=True))
        result = s.value_counts()
        expected = Series(
            [3, 2, 1, 0],
            index=Categorical(
                ["b", "a", "c", "d"], categories=list("abcd"), ordered=True
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_value_counts_dropna(self):
        # https://github.com/pandas-dev/pandas/issues/9443#issuecomment-73719328

        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=True),
            Series([2, 1], index=[True, False], name="count"),
        )
        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=False),
            Series([2, 1], index=[True, False], name="count"),
        )

        tm.assert_series_equal(
            Series([True] * 3 + [False] * 2 + [None] * 5).value_counts(dropna=True),
            Series([3, 2], index=Index([True, False], dtype=object), name="count"),
        )
        tm.assert_series_equal(
            Series([True] * 5 + [False] * 3 + [None] * 2).value_counts(dropna=False),
            Series([5, 3, 2], index=[True, False, None], name="count"),
        )
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=False),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )

        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0, None]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )

        result = Series([10.3, 10.3, 5.0, 5.0, 5.0, None]).value_counts(dropna=False)
        expected = Series([3, 2, 1], index=[5.0, 10.3, None], name="count")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", (np.float64, object, "M8[ns]"))
    def test_value_counts_normalized(self, dtype):
        # GH12558
        s = Series([1] * 2 + [2] * 3 + [np.nan] * 5)
        s_typed = s.astype(dtype)
        result = s_typed.value_counts(normalize=True, dropna=False)
        expected = Series(
            [0.5, 0.3, 0.2],
            index=Series([np.nan, 2.0, 1.0], dtype=dtype),
            name="proportion",
        )
        tm.assert_series_equal(result, expected)

        result = s_typed.value_counts(normalize=True, dropna=True)
        expected = Series(
            [0.6, 0.4], index=Series([2.0, 1.0], dtype=dtype), name="proportion"
        )
        tm.assert_series_equal(result, expected)

    def test_value_counts_uint64(self):
        arr = np.array([2**63], dtype=np.uint64)
        expected = Series([1], index=[2**63], name="count")
        msg = "pandas.value_counts is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.value_counts(arr)

        tm.assert_series_equal(result, expected)

        arr = np.array([-1, 2**63], dtype=object)
        expected = Series([1, 1], index=[-1, 2**63], name="count")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = algos.value_counts(arr)

        tm.assert_series_equal(result, expected)

    def test_value_counts_series(self):
        # GH#54857
        values = np.array([3, 1, 2, 3, 4, np.nan])
        result = Series(values).value_counts(bins=3)
        expected = Series(
            [2, 2, 1],
            index=IntervalIndex.from_tuples(
                [(0.996, 2.0), (2.0, 3.0), (3.0, 4.0)], dtype="interval[float64, right]"
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected)


class TestDuplicated:
    def test_duplicated_with_nas(self):
        keys = np.array([0, 1, np.nan, 0, 2, np.nan], dtype=object)

        result = algos.duplicated(keys)
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="first")
        expected = np.array([False, False, False, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="last")
        expected = np.array([True, False, True, False, False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array([True, False, True, True, False, True])
        tm.assert_numpy_array_equal(result, expected)

        keys = np.empty(8, dtype=object)
        for i, t in enumerate(
            zip([0, 0, np.nan, np.nan] * 2, [0, np.nan, 0, np.nan] * 2)
        ):
            keys[i] = t

        result = algos.duplicated(keys)
        falses = [False] * 4
        trues = [True] * 4
        expected = np.array(falses + trues)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep="last")
        expected = np.array(trues + falses)
        tm.assert_numpy_array_equal(result, expected)

        result = algos.duplicated(keys, keep=False)
        expected = np.array(trues + trues)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "case",
        [
            np.array([1, 2, 1, 5, 3, 2, 4, 1, 5, 6]),
            np.array([1.1, 2.2, 1.1, np.nan, 3.3, 2.2, 4.4, 1.1, np.nan, 6.6]),
            np.array(
                [
                    1 + 1j,
                    2 + 2j,
                    1 + 1j,
                    5 + 5j,
                    3 + 3j,
                    2 + 2j,
                    4 + 4j,
                    1 + 1j,
                    5 + 5j,
                    6 + 6j,
                ]
            ),
            np.array(["a", "b", "a", "e", "c", "b", "d", "a", "e", "f"], dtype=object),
            np.array(
                [1, 2**63, 1, 3**5, 10, 2**63, 39, 1, 3**5, 7], dtype=np.uint64
            ),
        ],
    )
    def test_numeric_object_likes(self, case):
        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        exp_false = exp_first | exp_last

        res_first = algos.duplicated(case, keep="first")
        tm.assert_numpy_array_equal(res_first, exp_first)

        res_last = algos.duplicated(case, keep="last")
        tm.assert_numpy_array_equal(res_last, exp_last)

        res_false = algos.duplicated(case, keep=False)
        tm.assert_numpy_array_equal(res_false, exp_false)

        # index
        for idx in [Index(case), Index(case, dtype="category")]:
            res_first = idx.duplicated(keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = idx.duplicated(keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = idx.duplicated(keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

        # series
        for s in [Series(case), Series(case, dtype="category")]:
            res_first = s.duplicated(keep="first")
            tm.assert_series_equal(res_first, Series(exp_first))

            res_last = s.duplicated(keep="last")
            tm.assert_series_equal(res_last, Series(exp_last))

            res_false = s.duplicated(keep=False)
            tm.assert_series_equal(res_false, Series(exp_false))

    def test_datetime_likes(self):
        dt = [
            "2011-01-01",
            "2011-01-02",
            "2011-01-01",
            "NaT",
            "2011-01-03",
            "2011-01-02",
            "2011-01-04",
            "2011-01-01",
            "NaT",
            "2011-01-06",
        ]
        td = [
            "1 days",
            "2 days",
            "1 days",
            "NaT",
            "3 days",
            "2 days",
            "4 days",
            "1 days",
            "NaT",
            "6 days",
        ]

        cases = [
            np.array([Timestamp(d) for d in dt]),
            np.array([Timestamp(d, tz="US/Eastern") for d in dt]),
            np.array([Period(d, freq="D") for d in dt]),
            np.array([np.datetime64(d) for d in dt]),
            np.array([Timedelta(d) for d in td]),
        ]

        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        exp_false = exp_first | exp_last

        for case in cases:
            res_first = algos.duplicated(case, keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = algos.duplicated(case, keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = algos.duplicated(case, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

            # index
            for idx in [
                Index(case),
                Index(case, dtype="category"),
                Index(case, dtype=object),
            ]:
                res_first = idx.duplicated(keep="first")
                tm.assert_numpy_array_equal(res_first, exp_first)

                res_last = idx.duplicated(keep="last")
                tm.assert_numpy_array_equal(res_last, exp_last)

                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)

            # series
            for s in [
                Series(case),
                Series(case, dtype="category"),
                Series(case, dtype=object),
            ]:
                res_first = s.duplicated(keep="first")
                tm.assert_series_equal(res_first, Series(exp_first))

                res_last = s.duplicated(keep="last")
                tm.assert_series_equal(res_last, Series(exp_last))

                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

    @pytest.mark.parametrize("case", [Index([1, 2, 3]), pd.RangeIndex(0, 3)])
    def test_unique_index(self, case):
        assert case.is_unique is True
        tm.assert_numpy_array_equal(case.duplicated(), np.array([False, False, False]))

    @pytest.mark.parametrize(
        "arr, uniques",
        [
            (
                [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)],
                [(0, 0), (0, 1), (1, 0), (1, 1)],
            ),
            (
                [("b", "c"), ("a", "b"), ("a", "b"), ("b", "c")],
                [("b", "c"), ("a", "b")],
            ),
            ([("a", 1), ("b", 2), ("a", 3), ("a", 1)], [("a", 1), ("b", 2), ("a", 3)]),
        ],
    )
    def test_unique_tuples(self, arr, uniques):
        # https://github.com/pandas-dev/pandas/issues/16519
        expected = np.empty(len(uniques), dtype=object)
        expected[:] = uniques

        msg = "unique with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pd.unique(arr)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "array,expected",
        [
            (
                [1 + 1j, 0, 1, 1j, 1 + 2j, 1 + 2j],
                # Should return a complex dtype in the future
                np.array([(1 + 1j), 0j, (1 + 0j), 1j, (1 + 2j)], dtype=object),
            )
        ],
    )
    def test_unique_complex_numbers(self, array, expected):
        # GH 17927
        msg = "unique with argument that is not not a Series"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pd.unique(array)
        tm.assert_numpy_array_equal(result, expected)


class TestHashTable:
    @pytest.mark.parametrize(
        "htable, data",
        [
            (
                ht.PyObjectHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (
                ht.StringHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (ht.Float64HashTable, np.arange(1000, dtype=np.float64)),
            (ht.Int64HashTable, np.arange(1000, dtype=np.int64)),
            (ht.UInt64HashTable, np.arange(1000, dtype=np.uint64)),
        ],
    )
    def test_hashtable_unique(self, htable, data, writable):
        # output of maker has guaranteed unique elements
        s = Series(data, dtype=data.dtype)
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.unique()
        expected_unique = s_duplicated.drop_duplicates(keep="first").values
        result_unique = htable().unique(s_duplicated.values)
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # test return_inverse=True
        # reconstruction can only succeed if the inverse is correct
        result_unique, result_inverse = htable().unique(
            s_duplicated.values, return_inverse=True
        )
        tm.assert_numpy_array_equal(result_unique, expected_unique)
        reconstr = result_unique[result_inverse]
        tm.assert_numpy_array_equal(reconstr, s_duplicated.values)

    @pytest.mark.parametrize(
        "htable, data",
        [
            (
                ht.PyObjectHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (
                ht.StringHashTable,
                np.array([f"foo_{i}" for i in range(1000)], dtype=object),
            ),
            (ht.Float64HashTable, np.arange(1000, dtype=np.float64)),
            (ht.Int64HashTable, np.arange(1000, dtype=np.int64)),
            (ht.UInt64HashTable, np.arange(1000, dtype=np.uint64)),
        ],
    )
    def test_hashtable_factorize(self, htable, writable, data):
        # output of maker has guaranteed unique elements
        s = Series(data, dtype=data.dtype)
        if htable == ht.Float64HashTable:
            # add NaN for float column
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # use different NaN types for object column
            s.loc[500:502] = [np.nan, None, NaT]

        # create duplicated selection
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        s_duplicated.values.setflags(write=writable)
        na_mask = s_duplicated.isna().values

        result_unique, result_inverse = htable().factorize(s_duplicated.values)

        # drop_duplicates has own cython code (hash_table_func_helper.pxi)
        # and is tested separately; keeps first occurrence like ht.factorize()
        # since factorize removes all NaNs, we do the same here
        expected_unique = s_duplicated.dropna().drop_duplicates().values
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # reconstruction can only succeed if the inverse is correct. Since
        # factorize removes the NaNs, those have to be excluded here as well
        result_reconstruct = result_unique[result_inverse[~na_mask]]
        expected_reconstruct = s_duplicated.dropna().values
        tm.assert_numpy_array_equal(result_reconstruct, expected_reconstruct)


class TestRank:
    @pytest.mark.parametrize(
        "arr",
        [
            [np.nan, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 3, np.nan],
            [4.0, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 4.0, np.nan],
        ],
    )
    def test_scipy_compat(self, arr):
        sp_stats = pytest.importorskip("scipy.stats")

        arr = np.array(arr)

        mask = ~np.isfinite(arr)
        arr = arr.copy()
        result = libalgos.rank_1d(arr)
        arr[mask] = np.inf
        exp = sp_stats.rankdata(arr)
        exp[mask] = np.nan
        tm.assert_almost_equal(result, exp)

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_basic(self, writable, dtype):
        exp = np.array([1, 2], dtype=np.float64)

        data = np.array([1, 100], dtype=dtype)
        data.setflags(write=writable)
        ser = Series(data)
        result = algos.rank(ser)
        tm.assert_numpy_array_equal(result, exp)

    @pytest.mark.parametrize("dtype", [np.float64, np.uint64])
    def test_uint64_overflow(self, dtype):
        exp = np.array([1, 2], dtype=np.float64)

        s = Series([1, 2**63], dtype=dtype)
        tm.assert_numpy_array_equal(algos.rank(s), exp)

    def test_too_many_ndims(self):
        arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        msg = "Array with ndim > 2 are not supported"

        with pytest.raises(TypeError, match=msg):
            algos.rank(arr)

    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self):
        # GH 18271
        values = np.arange(2**24 + 1)
        result = algos.rank(values, pct=True).max()
        assert result == 1

        values = np.arange(2**25 + 2).reshape(2**24 + 1, 2)
        result = algos.rank(values, pct=True).max()
        assert result == 1


class TestMode:
    def test_no_mode(self):
        exp = Series([], dtype=np.float64, index=Index([], dtype=int))
        tm.assert_numpy_array_equal(algos.mode(np.array([])), exp.values)

    @pytest.mark.parametrize("dt", np.typecodes["AllInteger"] + np.typecodes["Float"])
    def test_mode_single(self, dt):
        # GH 15714
        exp_single = [1]
        data_single = [1]

        exp_multi = [1]
        data_multi = [1, 1]

        ser = Series(data_single, dtype=dt)
        exp = Series(exp_single, dtype=dt)
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

        ser = Series(data_multi, dtype=dt)
        exp = Series(exp_multi, dtype=dt)
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_mode_obj_int(self):
        exp = Series([1], dtype=int)
        tm.assert_numpy_array_equal(algos.mode(exp.values), exp.values)

        exp = Series(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(algos.mode(exp.values), exp.values)

    @pytest.mark.parametrize("dt", np.typecodes["AllInteger"] + np.typecodes["Float"])
    def test_number_mode(self, dt):
        exp_single = [1]
        data_single = [1] * 5 + [2] * 3

        exp_multi = [1, 3]
        data_multi = [1] * 5 + [2] * 3 + [3] * 5

        ser = Series(data_single, dtype=dt)
        exp = Series(exp_single, dtype=dt)
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

        ser = Series(data_multi, dtype=dt)
        exp = Series(exp_multi, dtype=dt)
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_strobj_mode(self):
        exp = ["b"]
        data = ["a"] * 2 + ["b"] * 3

        ser = Series(data, dtype="c")
        exp = Series(exp, dtype="c")
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    @pytest.mark.parametrize("dt", [str, object])
    def test_strobj_multi_char(self, dt, using_infer_string):
        exp = ["bar"]
        data = ["foo"] * 2 + ["bar"] * 3

        ser = Series(data, dtype=dt)
        exp = Series(exp, dtype=dt)
        if using_infer_string and dt is str:
            tm.assert_extension_array_equal(algos.mode(ser.values), exp.values)
        else:
            tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_datelike_mode(self):
        exp = Series(["1900-05-03", "2011-01-03", "2013-01-02"], dtype="M8[ns]")
        ser = Series(["2011-01-03", "2013-01-02", "1900-05-03"], dtype="M8[ns]")
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

        exp = Series(["2011-01-03", "2013-01-02"], dtype="M8[ns]")
        ser = Series(
            ["2011-01-03", "2013-01-02", "1900-05-03", "2011-01-03", "2013-01-02"],
            dtype="M8[ns]",
        )
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_timedelta_mode(self):
        exp = Series(["-1 days", "0 days", "1 days"], dtype="timedelta64[ns]")
        ser = Series(["1 days", "-1 days", "0 days"], dtype="timedelta64[ns]")
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

        exp = Series(["2 min", "1 day"], dtype="timedelta64[ns]")
        ser = Series(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_mixed_dtype(self):
        exp = Series(["foo"], dtype=object)
        ser = Series([1, "foo", "foo"])
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_uint64_overflow(self):
        exp = Series([2**63], dtype=np.uint64)
        ser = Series([1, 2**63, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

        exp = Series([1, 2**63], dtype=np.uint64)
        ser = Series([1, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        tm.assert_series_equal(ser.mode(), exp)

    def test_categorical(self):
        c = Categorical([1, 2])
        exp = c
        res = Series(c).mode()._values
        tm.assert_categorical_equal(res, exp)

        c = Categorical([1, "a", "a"])
        exp = Categorical(["a"], categories=[1, "a"])
        res = Series(c).mode()._values
        tm.assert_categorical_equal(res, exp)

        c = Categorical([1, 1, 2, 3, 3])
        exp = Categorical([1, 3], categories=[1, 2, 3])
        res = Series(c).mode()._values
        tm.assert_categorical_equal(res, exp)

    def test_index(self):
        idx = Index([1, 2, 3])
        exp = Series([1, 2, 3], dtype=np.int64)
        tm.assert_numpy_array_equal(algos.mode(idx), exp.values)

        idx = Index([1, "a", "a"])
        exp = Series(["a"], dtype=object)
        tm.assert_numpy_array_equal(algos.mode(idx), exp.values)

        idx = Index([1, 1, 2, 3, 3])
        exp = Series([1, 3], dtype=np.int64)
        tm.assert_numpy_array_equal(algos.mode(idx), exp.values)

        idx = Index(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        with pytest.raises(AttributeError, match="TimedeltaIndex"):
            # algos.mode expects Arraylike, does *not* unwrap TimedeltaIndex
            algos.mode(idx)

    def test_ser_mode_with_name(self):
        # GH 46737
        ser = Series([1, 1, 3], name="foo")
        result = ser.mode()
        expected = Series([1], name="foo")
        tm.assert_series_equal(result, expected)


class TestDiff:
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_diff_datetimelike_nat(self, dtype):
        # NaT - NaT is NaT, not 0
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(3, 4)
        arr[:, 2] = arr.dtype.type("NaT", "ns")
        result = algos.diff(arr, 1, axis=0)

        expected = np.ones(arr.shape, dtype="timedelta64[ns]") * 4
        expected[:, 2] = np.timedelta64("NaT", "ns")
        expected[0, :] = np.timedelta64("NaT", "ns")

        tm.assert_numpy_array_equal(result, expected)

        result = algos.diff(arr.T, 1, axis=1)
        tm.assert_numpy_array_equal(result, expected.T)

    def test_diff_ea_axis(self):
        dta = date_range("2016-01-01", periods=3, tz="US/Pacific")._data

        msg = "cannot diff DatetimeArray on axis=1"
        with pytest.raises(ValueError, match=msg):
            algos.diff(dta, 1, axis=1)

    @pytest.mark.parametrize("dtype", ["int8", "int16"])
    def test_diff_low_precision_int(self, dtype):
        arr = np.array([0, 1, 1, 0, 0], dtype=dtype)
        result = algos.diff(arr, 1)
        expected = np.array([np.nan, 1, 0, -1, 0], dtype="float32")
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("op", [np.array, pd.array])
def test_union_with_duplicates(op):
    # GH#36289
    lvals = op([3, 1, 3, 4])
    rvals = op([2, 3, 1, 1])
    expected = op([3, 3, 1, 1, 4, 2])
    if isinstance(expected, np.ndarray):
        result = algos.union_with_duplicates(lvals, rvals)
        tm.assert_numpy_array_equal(result, expected)
    else:
        result = algos.union_with_duplicates(lvals, rvals)
        tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_mathml.py ===
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, Lambda, diff, Function
from sympy.core.numbers import (zoo, Float, Integer, I, oo, pi, E,
    Rational)
from sympy.core.relational import Lt, Ge, Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (factorial2,
    binomial, factorial)
from sympy.functions.combinatorial.numbers import (lucas, bell,
    catalan, euler, tribonacci, fibonacci, bernoulli, primenu, primeomega,
    totient, reduced_totient)
from sympy.functions.elementary.complexes import re, im, conjugate, Abs
from sympy.functions.elementary.exponential import exp, LambertW, log
from sympy.functions.elementary.hyperbolic import (tanh, acoth, atanh,
    coth, asinh, acsch, asech, acosh, csch, sinh, cosh, sech)
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import (csc, sec, tan,
    atan, sin, asec, cot, cos, acot, acsc, asin, acos)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.elliptic_integrals import (elliptic_pi,
    elliptic_f, elliptic_k, elliptic_e)
from sympy.functions.special.error_functions import (fresnelc,
    fresnels, Ei, expint)
from sympy.functions.special.gamma_functions import (gamma, uppergamma,
    lowergamma)
from sympy.functions.special.mathieu_functions import (mathieusprime,
    mathieus, mathieucprime, mathieuc)
from sympy.functions.special.polynomials import (jacobi, chebyshevu,
    chebyshevt, hermite, assoc_legendre, gegenbauer, assoc_laguerre,
    legendre, laguerre)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import (polylog, stieltjes,
    lerchphi, dirichlet_eta, zeta)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (Xor, Or, false, true, And, Equivalent,
    Implies, Not)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.physics.quantum import (ComplexSpace, FockSpace, hbar,
    HilbertSpace, Dagger)
from sympy.printing.mathml import (MathMLPresentationPrinter,
    MathMLPrinter, MathMLContentPrinter, mathml)
from sympy.series.limits import Limit
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, Union, SymmetricDifference,
    Complement, FiniteSet, Intersection, ProductSet)
from sympy.stats.rv import RandomSymbol
from sympy.tensor.indexed import IndexedBase
from sympy.vector import (Divergence, CoordSys3D, Cross, Curl, Dot,
    Laplacian, Gradient)
from sympy.testing.pytest import raises, XFAIL

x, y, z, a, b, c, d, e, n = symbols('x:z a:e n')
mp = MathMLContentPrinter()
mpp = MathMLPresentationPrinter()


def test_mathml_printer():
    m = MathMLPrinter()
    assert m.doprint(1+x) == mp.doprint(1+x)


def test_content_printmethod():
    assert mp.doprint(1 + x) == '<apply><plus/><ci>x</ci><cn>1</cn></apply>'


def test_content_mathml_core():
    mml_1 = mp._print(1 + x)
    assert mml_1.nodeName == 'apply'
    nodes = mml_1.childNodes
    assert len(nodes) == 3
    assert nodes[0].nodeName == 'plus'
    assert nodes[0].hasChildNodes() is False
    assert nodes[0].nodeValue is None
    assert nodes[1].nodeName in ['cn', 'ci']
    if nodes[1].nodeName == 'cn':
        assert nodes[1].childNodes[0].nodeValue == '1'
        assert nodes[2].childNodes[0].nodeValue == 'x'
    else:
        assert nodes[1].childNodes[0].nodeValue == 'x'
        assert nodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mp._print(x**2)
    assert mml_2.nodeName == 'apply'
    nodes = mml_2.childNodes
    assert nodes[1].childNodes[0].nodeValue == 'x'
    assert nodes[2].childNodes[0].nodeValue == '2'

    mml_3 = mp._print(2*x)
    assert mml_3.nodeName == 'apply'
    nodes = mml_3.childNodes
    assert nodes[0].nodeName == 'times'
    assert nodes[1].childNodes[0].nodeValue == '2'
    assert nodes[2].childNodes[0].nodeValue == 'x'

    mml = mp._print(Float(1.0, 2)*x)
    assert mml.nodeName == 'apply'
    nodes = mml.childNodes
    assert nodes[0].nodeName == 'times'
    assert nodes[1].childNodes[0].nodeValue == '1.0'
    assert nodes[2].childNodes[0].nodeValue == 'x'


def test_content_mathml_functions():
    mml_1 = mp._print(sin(x))
    assert mml_1.nodeName == 'apply'
    assert mml_1.childNodes[0].nodeName == 'sin'
    assert mml_1.childNodes[1].nodeName == 'ci'

    mml_2 = mp._print(diff(sin(x), x, evaluate=False))
    assert mml_2.nodeName == 'apply'
    assert mml_2.childNodes[0].nodeName == 'diff'
    assert mml_2.childNodes[1].nodeName == 'bvar'
    assert mml_2.childNodes[1].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>x/ci>

    mml_3 = mp._print(diff(cos(x*y), x, evaluate=False))
    assert mml_3.nodeName == 'apply'
    assert mml_3.childNodes[0].nodeName == 'partialdiff'
    assert mml_3.childNodes[1].nodeName == 'bvar'
    assert mml_3.childNodes[1].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>x/ci>

    mml_4 = mp._print(Lambda((x, y), x * y))
    assert mml_4.nodeName == 'lambda'
    assert mml_4.childNodes[0].nodeName == 'bvar'
    assert mml_4.childNodes[0].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>x/ci>
    assert mml_4.childNodes[1].nodeName == 'bvar'
    assert mml_4.childNodes[1].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>y/ci>
    assert mml_4.childNodes[2].nodeName == 'apply'


def test_content_mathml_limits():
    # XXX No unevaluated limits
    lim_fun = sin(x)/x
    mml_1 = mp._print(Limit(lim_fun, x, 0))
    assert mml_1.childNodes[0].nodeName == 'limit'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    assert mml_1.childNodes[3].toxml() == mp._print(lim_fun).toxml()


def test_content_mathml_integrals():
    integrand = x
    mml_1 = mp._print(Integral(integrand, (x, 0, 1)))
    assert mml_1.childNodes[0].nodeName == 'int'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    assert mml_1.childNodes[3].nodeName == 'uplimit'
    assert mml_1.childNodes[4].toxml() == mp._print(integrand).toxml()


def test_content_mathml_matrices():
    A = Matrix([1, 2, 3])
    B = Matrix([[0, 5, 4], [2, 3, 1], [9, 7, 9]])
    mll_1 = mp._print(A)
    assert mll_1.childNodes[0].nodeName == 'matrixrow'
    assert mll_1.childNodes[0].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[0].childNodes[0].childNodes[0].nodeValue == '1'
    assert mll_1.childNodes[1].nodeName == 'matrixrow'
    assert mll_1.childNodes[1].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_1.childNodes[2].nodeName == 'matrixrow'
    assert mll_1.childNodes[2].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[2].childNodes[0].childNodes[0].nodeValue == '3'
    mll_2 = mp._print(B)
    assert mll_2.childNodes[0].nodeName == 'matrixrow'
    assert mll_2.childNodes[0].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[0].childNodes[0].nodeValue == '0'
    assert mll_2.childNodes[0].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[1].childNodes[0].nodeValue == '5'
    assert mll_2.childNodes[0].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[2].childNodes[0].nodeValue == '4'
    assert mll_2.childNodes[1].nodeName == 'matrixrow'
    assert mll_2.childNodes[1].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_2.childNodes[1].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[1].childNodes[0].nodeValue == '3'
    assert mll_2.childNodes[1].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[2].childNodes[0].nodeValue == '1'
    assert mll_2.childNodes[2].nodeName == 'matrixrow'
    assert mll_2.childNodes[2].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[0].childNodes[0].nodeValue == '9'
    assert mll_2.childNodes[2].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[1].childNodes[0].nodeValue == '7'
    assert mll_2.childNodes[2].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[2].childNodes[0].nodeValue == '9'


def test_content_mathml_sums():
    summand = x
    mml_1 = mp._print(Sum(summand, (x, 1, 10)))
    assert mml_1.childNodes[0].nodeName == 'sum'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    assert mml_1.childNodes[3].nodeName == 'uplimit'
    assert mml_1.childNodes[4].toxml() == mp._print(summand).toxml()


def test_content_mathml_tuples():
    mml_1 = mp._print([2])
    assert mml_1.nodeName == 'list'
    assert mml_1.childNodes[0].nodeName == 'cn'
    assert len(mml_1.childNodes) == 1

    mml_2 = mp._print([2, Integer(1)])
    assert mml_2.nodeName == 'list'
    assert mml_2.childNodes[0].nodeName == 'cn'
    assert mml_2.childNodes[1].nodeName == 'cn'
    assert len(mml_2.childNodes) == 2


def test_content_mathml_add():
    mml = mp._print(x**5 - x**4 + x)
    assert mml.childNodes[0].nodeName == 'plus'
    assert mml.childNodes[1].childNodes[0].nodeName == 'minus'
    assert mml.childNodes[1].childNodes[1].nodeName == 'apply'


def test_content_mathml_Rational():
    mml_1 = mp._print(Rational(1, 1))
    """should just return a number"""
    assert mml_1.nodeName == 'cn'

    mml_2 = mp._print(Rational(2, 5))
    assert mml_2.childNodes[0].nodeName == 'divide'


def test_content_mathml_constants():
    mml = mp._print(I)
    assert mml.nodeName == 'imaginaryi'

    mml = mp._print(E)
    assert mml.nodeName == 'exponentiale'

    mml = mp._print(oo)
    assert mml.nodeName == 'infinity'

    mml = mp._print(pi)
    assert mml.nodeName == 'pi'

    assert mathml(hbar) == '<hbar/>'
    assert mathml(S.TribonacciConstant) == '<tribonacciconstant/>'
    assert mathml(S.GoldenRatio) == '<cn>&#966;</cn>'
    mml = mathml(S.EulerGamma)
    assert mml == '<eulergamma/>'

    mml = mathml(S.EmptySet)
    assert mml == '<emptyset/>'

    mml = mathml(S.true)
    assert mml == '<true/>'

    mml = mathml(S.false)
    assert mml == '<false/>'

    mml = mathml(S.NaN)
    assert mml == '<notanumber/>'


def test_content_mathml_trig():
    mml = mp._print(sin(x))
    assert mml.childNodes[0].nodeName == 'sin'

    mml = mp._print(cos(x))
    assert mml.childNodes[0].nodeName == 'cos'

    mml = mp._print(tan(x))
    assert mml.childNodes[0].nodeName == 'tan'

    mml = mp._print(cot(x))
    assert mml.childNodes[0].nodeName == 'cot'

    mml = mp._print(csc(x))
    assert mml.childNodes[0].nodeName == 'csc'

    mml = mp._print(sec(x))
    assert mml.childNodes[0].nodeName == 'sec'

    mml = mp._print(asin(x))
    assert mml.childNodes[0].nodeName == 'arcsin'

    mml = mp._print(acos(x))
    assert mml.childNodes[0].nodeName == 'arccos'

    mml = mp._print(atan(x))
    assert mml.childNodes[0].nodeName == 'arctan'

    mml = mp._print(acot(x))
    assert mml.childNodes[0].nodeName == 'arccot'

    mml = mp._print(acsc(x))
    assert mml.childNodes[0].nodeName == 'arccsc'

    mml = mp._print(asec(x))
    assert mml.childNodes[0].nodeName == 'arcsec'

    mml = mp._print(sinh(x))
    assert mml.childNodes[0].nodeName == 'sinh'

    mml = mp._print(cosh(x))
    assert mml.childNodes[0].nodeName == 'cosh'

    mml = mp._print(tanh(x))
    assert mml.childNodes[0].nodeName == 'tanh'

    mml = mp._print(coth(x))
    assert mml.childNodes[0].nodeName == 'coth'

    mml = mp._print(csch(x))
    assert mml.childNodes[0].nodeName == 'csch'

    mml = mp._print(sech(x))
    assert mml.childNodes[0].nodeName == 'sech'

    mml = mp._print(asinh(x))
    assert mml.childNodes[0].nodeName == 'arcsinh'

    mml = mp._print(atanh(x))
    assert mml.childNodes[0].nodeName == 'arctanh'

    mml = mp._print(acosh(x))
    assert mml.childNodes[0].nodeName == 'arccosh'

    mml = mp._print(acoth(x))
    assert mml.childNodes[0].nodeName == 'arccoth'

    mml = mp._print(acsch(x))
    assert mml.childNodes[0].nodeName == 'arccsch'

    mml = mp._print(asech(x))
    assert mml.childNodes[0].nodeName == 'arcsech'


def test_content_mathml_relational():
    mml_1 = mp._print(Eq(x, 1))
    assert mml_1.nodeName == 'apply'
    assert mml_1.childNodes[0].nodeName == 'eq'
    assert mml_1.childNodes[1].nodeName == 'ci'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == 'x'
    assert mml_1.childNodes[2].nodeName == 'cn'
    assert mml_1.childNodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mp._print(Ne(1, x))
    assert mml_2.nodeName == 'apply'
    assert mml_2.childNodes[0].nodeName == 'neq'
    assert mml_2.childNodes[1].nodeName == 'cn'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_2.childNodes[2].nodeName == 'ci'
    assert mml_2.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_3 = mp._print(Ge(1, x))
    assert mml_3.nodeName == 'apply'
    assert mml_3.childNodes[0].nodeName == 'geq'
    assert mml_3.childNodes[1].nodeName == 'cn'
    assert mml_3.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_3.childNodes[2].nodeName == 'ci'
    assert mml_3.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_4 = mp._print(Lt(1, x))
    assert mml_4.nodeName == 'apply'
    assert mml_4.childNodes[0].nodeName == 'lt'
    assert mml_4.childNodes[1].nodeName == 'cn'
    assert mml_4.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_4.childNodes[2].nodeName == 'ci'
    assert mml_4.childNodes[2].childNodes[0].nodeValue == 'x'


def test_content_symbol():
    mml = mp._print(x)
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeValue == 'x'
    del mml

    mml = mp._print(Symbol("x^2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mp._print(Symbol("x__2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mp._print(Symbol("x_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msub'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mp._print(Symbol("x^3_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msubsup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mp._print(Symbol("x__3_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msubsup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mp._print(Symbol("x_2_a"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msub'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[
        0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[
        0].nodeValue == ' '
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[
        0].nodeValue == 'a'
    del mml

    mml = mp._print(Symbol("x^2^a"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[
        0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[
        0].nodeValue == ' '
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[
        0].nodeValue == 'a'
    del mml

    mml = mp._print(Symbol("x__2__a"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[
        0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[
        0].nodeValue == ' '
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[
        0].nodeValue == 'a'
    del mml


def test_content_mathml_greek():
    mml = mp._print(Symbol('alpha'))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeValue == '\N{GREEK SMALL LETTER ALPHA}'

    assert mp.doprint(Symbol('alpha')) == '<ci>&#945;</ci>'
    assert mp.doprint(Symbol('beta')) == '<ci>&#946;</ci>'
    assert mp.doprint(Symbol('gamma')) == '<ci>&#947;</ci>'
    assert mp.doprint(Symbol('delta')) == '<ci>&#948;</ci>'
    assert mp.doprint(Symbol('epsilon')) == '<ci>&#949;</ci>'
    assert mp.doprint(Symbol('zeta')) == '<ci>&#950;</ci>'
    assert mp.doprint(Symbol('eta')) == '<ci>&#951;</ci>'
    assert mp.doprint(Symbol('theta')) == '<ci>&#952;</ci>'
    assert mp.doprint(Symbol('iota')) == '<ci>&#953;</ci>'
    assert mp.doprint(Symbol('kappa')) == '<ci>&#954;</ci>'
    assert mp.doprint(Symbol('lambda')) == '<ci>&#955;</ci>'
    assert mp.doprint(Symbol('mu')) == '<ci>&#956;</ci>'
    assert mp.doprint(Symbol('nu')) == '<ci>&#957;</ci>'
    assert mp.doprint(Symbol('xi')) == '<ci>&#958;</ci>'
    assert mp.doprint(Symbol('omicron')) == '<ci>&#959;</ci>'
    assert mp.doprint(Symbol('pi')) == '<ci>&#960;</ci>'
    assert mp.doprint(Symbol('rho')) == '<ci>&#961;</ci>'
    assert mp.doprint(Symbol('varsigma')) == '<ci>&#962;</ci>'
    assert mp.doprint(Symbol('sigma')) == '<ci>&#963;</ci>'
    assert mp.doprint(Symbol('tau')) == '<ci>&#964;</ci>'
    assert mp.doprint(Symbol('upsilon')) == '<ci>&#965;</ci>'
    assert mp.doprint(Symbol('phi')) == '<ci>&#966;</ci>'
    assert mp.doprint(Symbol('chi')) == '<ci>&#967;</ci>'
    assert mp.doprint(Symbol('psi')) == '<ci>&#968;</ci>'
    assert mp.doprint(Symbol('omega')) == '<ci>&#969;</ci>'

    assert mp.doprint(Symbol('Alpha')) == '<ci>&#913;</ci>'
    assert mp.doprint(Symbol('Beta')) == '<ci>&#914;</ci>'
    assert mp.doprint(Symbol('Gamma')) == '<ci>&#915;</ci>'
    assert mp.doprint(Symbol('Delta')) == '<ci>&#916;</ci>'
    assert mp.doprint(Symbol('Epsilon')) == '<ci>&#917;</ci>'
    assert mp.doprint(Symbol('Zeta')) == '<ci>&#918;</ci>'
    assert mp.doprint(Symbol('Eta')) == '<ci>&#919;</ci>'
    assert mp.doprint(Symbol('Theta')) == '<ci>&#920;</ci>'
    assert mp.doprint(Symbol('Iota')) == '<ci>&#921;</ci>'
    assert mp.doprint(Symbol('Kappa')) == '<ci>&#922;</ci>'
    assert mp.doprint(Symbol('Lambda')) == '<ci>&#923;</ci>'
    assert mp.doprint(Symbol('Mu')) == '<ci>&#924;</ci>'
    assert mp.doprint(Symbol('Nu')) == '<ci>&#925;</ci>'
    assert mp.doprint(Symbol('Xi')) == '<ci>&#926;</ci>'
    assert mp.doprint(Symbol('Omicron')) == '<ci>&#927;</ci>'
    assert mp.doprint(Symbol('Pi')) == '<ci>&#928;</ci>'
    assert mp.doprint(Symbol('Rho')) == '<ci>&#929;</ci>'
    assert mp.doprint(Symbol('Sigma')) == '<ci>&#931;</ci>'
    assert mp.doprint(Symbol('Tau')) == '<ci>&#932;</ci>'
    assert mp.doprint(Symbol('Upsilon')) == '<ci>&#933;</ci>'
    assert mp.doprint(Symbol('Phi')) == '<ci>&#934;</ci>'
    assert mp.doprint(Symbol('Chi')) == '<ci>&#935;</ci>'
    assert mp.doprint(Symbol('Psi')) == '<ci>&#936;</ci>'
    assert mp.doprint(Symbol('Omega')) == '<ci>&#937;</ci>'


def test_content_mathml_order():
    expr = x**3 + x**2*y + 3*x*y**3 + y**4

    mp = MathMLContentPrinter({'order': 'lex'})
    mml = mp._print(expr)

    assert mml.childNodes[1].childNodes[0].nodeName == 'power'
    assert mml.childNodes[1].childNodes[1].childNodes[0].data == 'x'
    assert mml.childNodes[1].childNodes[2].childNodes[0].data == '3'

    assert mml.childNodes[4].childNodes[0].nodeName == 'power'
    assert mml.childNodes[4].childNodes[1].childNodes[0].data == 'y'
    assert mml.childNodes[4].childNodes[2].childNodes[0].data == '4'

    mp = MathMLContentPrinter({'order': 'rev-lex'})
    mml = mp._print(expr)

    assert mml.childNodes[1].childNodes[0].nodeName == 'power'
    assert mml.childNodes[1].childNodes[1].childNodes[0].data == 'y'
    assert mml.childNodes[1].childNodes[2].childNodes[0].data == '4'

    assert mml.childNodes[4].childNodes[0].nodeName == 'power'
    assert mml.childNodes[4].childNodes[1].childNodes[0].data == 'x'
    assert mml.childNodes[4].childNodes[2].childNodes[0].data == '3'


def test_content_settings():
    raises(TypeError, lambda: mathml(x, method="garbage"))


def test_content_mathml_logic():
    assert mathml(And(x, y)) == '<apply><and/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Or(x, y)) == '<apply><or/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Xor(x, y)) == '<apply><xor/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Implies(x, y)) == '<apply><implies/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Not(x)) == '<apply><not/><ci>x</ci></apply>'


def test_content_finite_sets():
    assert mathml(FiniteSet(a)) == '<set><ci>a</ci></set>'
    assert mathml(FiniteSet(a, b)) == '<set><ci>a</ci><ci>b</ci></set>'
    assert mathml(FiniteSet(FiniteSet(a, b), c)) == \
        '<set><ci>c</ci><set><ci>a</ci><ci>b</ci></set></set>'

    A = FiniteSet(a)
    B = FiniteSet(b)
    C = FiniteSet(c)
    D = FiniteSet(d)

    U1 = Union(A, B, evaluate=False)
    U2 = Union(C, D, evaluate=False)
    I1 = Intersection(A, B, evaluate=False)
    I2 = Intersection(C, D, evaluate=False)
    C1 = Complement(A, B, evaluate=False)
    C2 = Complement(C, D, evaluate=False)
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(A, B)
    P2 = ProductSet(C, D)

    assert mathml(U1) == \
        '<apply><union/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(I1) == \
        '<apply><intersect/><set><ci>a</ci></set><set><ci>b</ci></set>' \
        '</apply>'
    assert mathml(C1) == \
        '<apply><setdiff/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(P1) == \
        '<apply><cartesianproduct/><set><ci>a</ci></set><set><ci>b</ci>' \
        '</set></apply>'

    assert mathml(Intersection(A, U2, evaluate=False)) == \
        '<apply><intersect/><set><ci>a</ci></set><apply><union/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Intersection(U1, U2, evaluate=False)) == \
        '<apply><intersect/><apply><union/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'

    # XXX Does the parenthesis appear correctly for these examples in mathjax?
    assert mathml(Intersection(C1, C2, evaluate=False)) == \
        '<apply><intersect/><apply><setdiff/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Intersection(P1, P2, evaluate=False)) == \
        '<apply><intersect/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    assert mathml(Union(A, I2, evaluate=False)) == \
        '<apply><union/><set><ci>a</ci></set><apply><intersect/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(I1, I2, evaluate=False)) == \
        '<apply><union/><apply><intersect/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(C1, C2, evaluate=False)) == \
        '<apply><union/><apply><setdiff/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(P1, P2, evaluate=False)) == \
        '<apply><union/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    assert mathml(Complement(A, C2, evaluate=False)) == \
        '<apply><setdiff/><set><ci>a</ci></set><apply><setdiff/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(U1, U2, evaluate=False)) == \
        '<apply><setdiff/><apply><union/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(I1, I2, evaluate=False)) == \
        '<apply><setdiff/><apply><intersect/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(P1, P2, evaluate=False)) == \
        '<apply><setdiff/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    assert mathml(ProductSet(A, P2)) == \
        '<apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<apply><cartesianproduct/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(U1, U2)) == \
        '<apply><cartesianproduct/><apply><union/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(I1, I2)) == \
        '<apply><cartesianproduct/><apply><intersect/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><intersect/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(C1, C2)) == \
        '<apply><cartesianproduct/><apply><setdiff/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><setdiff/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'


def test_presentation_printmethod():
    assert mpp.doprint(1 + x) == '<mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow>'
    assert mpp.doprint(x**2) == '<msup><mi>x</mi><mn>2</mn></msup>'
    assert mpp.doprint(x**-1) == '<mfrac><mn>1</mn><mi>x</mi></mfrac>'
    assert mpp.doprint(x**-2) == \
        '<mfrac><mn>1</mn><msup><mi>x</mi><mn>2</mn></msup></mfrac>'
    assert mpp.doprint(2*x) == \
        '<mrow><mn>2</mn><mo>&InvisibleTimes;</mo><mi>x</mi></mrow>'


def test_presentation_mathml_core():
    mml_1 = mpp._print(1 + x)
    assert mml_1.nodeName == 'mrow'
    nodes = mml_1.childNodes
    assert len(nodes) == 3
    assert nodes[0].nodeName in ['mi', 'mn']
    assert nodes[1].nodeName == 'mo'
    if nodes[0].nodeName == 'mn':
        assert nodes[0].childNodes[0].nodeValue == '1'
        assert nodes[2].childNodes[0].nodeValue == 'x'
    else:
        assert nodes[0].childNodes[0].nodeValue == 'x'
        assert nodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mpp._print(x**2)
    assert mml_2.nodeName == 'msup'
    nodes = mml_2.childNodes
    assert nodes[0].childNodes[0].nodeValue == 'x'
    assert nodes[1].childNodes[0].nodeValue == '2'

    mml_3 = mpp._print(2*x)
    assert mml_3.nodeName == 'mrow'
    nodes = mml_3.childNodes
    assert nodes[0].childNodes[0].nodeValue == '2'
    assert nodes[1].childNodes[0].nodeValue == '&InvisibleTimes;'
    assert nodes[2].childNodes[0].nodeValue == 'x'

    mml = mpp._print(Float(1.0, 2)*x)
    assert mml.nodeName == 'mrow'
    nodes = mml.childNodes
    assert nodes[0].childNodes[0].nodeValue == '1.0'
    assert nodes[1].childNodes[0].nodeValue == '&InvisibleTimes;'
    assert nodes[2].childNodes[0].nodeValue == 'x'


def test_presentation_mathml_functions():
    mml_1 = mpp._print(sin(x))
    assert mml_1.childNodes[0].childNodes[0
        ].nodeValue == 'sin'
    assert mml_1.childNodes[1].childNodes[1
        ].childNodes[0].nodeValue == 'x'

    mml_2 = mpp._print(diff(sin(x), x, evaluate=False))
    assert mml_2.nodeName == 'mrow'
    assert mml_2.childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '&dd;'
    assert mml_2.childNodes[1].childNodes[1
        ].nodeName == 'mrow'
    assert mml_2.childNodes[0].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '&dd;'

    mml_3 = mpp._print(diff(cos(x*y), x, evaluate=False))
    assert mml_3.childNodes[0].nodeName == 'mfrac'
    assert mml_3.childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '&#x2202;'
    assert mml_3.childNodes[1].childNodes[0
        ].childNodes[0].nodeValue == 'cos'


def test_print_derivative():
    f = Function('f')
    d = Derivative(f(x, y, z), x, z, x, z, z, y)
    assert mathml(d) == \
        '<apply><partialdiff/><bvar><ci>y</ci><ci>z</ci><degree><cn>2</cn></degree><ci>x</ci><ci>z</ci><ci>x</ci></bvar><apply><f/><ci>x</ci><ci>y</ci><ci>z</ci></apply></apply>'
    assert mathml(d, printer='presentation') == \
        '<mrow><mfrac><mrow><msup><mo>&#x2202;</mo><mn>6</mn></msup></mrow><mrow><mo>&#x2202;</mo><mi>y</mi><msup><mo>&#x2202;</mo><mn>2</mn></msup><mi>z</mi><mo>&#x2202;</mo><mi>x</mi><mo>&#x2202;</mo><mi>z</mi><mo>&#x2202;</mo><mi>x</mi></mrow></mfrac><mrow><mi>f</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow></mrow>'


def test_presentation_mathml_limits():
    lim_fun = sin(x)/x
    mml_1 = mpp._print(Limit(lim_fun, x, 0))
    assert mml_1.childNodes[0].nodeName == 'munder'
    assert mml_1.childNodes[0].childNodes[0
        ].childNodes[0].nodeValue == 'lim'
    assert mml_1.childNodes[0].childNodes[1
        ].childNodes[0].childNodes[0
        ].nodeValue == 'x'
    assert mml_1.childNodes[0].childNodes[1
        ].childNodes[1].childNodes[0
        ].nodeValue == '&#x2192;'
    assert mml_1.childNodes[0].childNodes[1
        ].childNodes[2].childNodes[0
        ].nodeValue == '0'


def test_presentation_mathml_integrals():
    assert mpp.doprint(Integral(x, (x, 0, 1))) == \
        '<mrow><msubsup><mo>&#x222B;</mo><mn>0</mn><mn>1</mn></msubsup>'\
        '<mi>x</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(log(x), x)) == \
        '<mrow><mo>&#x222B;</mo><mrow><mi>log</mi><mrow><mo>(</mo><mi>x</mi>' \
        '<mo>)</mo></mrow></mrow><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x*y, x, y)) == \
        '<mrow><mo>&#x222C;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>y</mi></mrow><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    z, w = symbols('z w')
    assert mpp.doprint(Integral(x*y*z, x, y, z)) == \
        '<mrow><mo>&#x222D;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>y</mi><mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo>'\
        '<mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x*y*z*w, x, y, z, w)) == \
        '<mrow><mo>&#x222B;</mo><mo>&#x222B;</mo><mo>&#x222B;</mo>'\
        '<mo>&#x222B;</mo><mrow><mi>w</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi>'\
        '<mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo><mi>w</mi>'\
        '<mo>&dd;</mo><mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x, x, y, (z, 0, 1))) == \
        '<mrow><msubsup><mo>&#x222B;</mo><mn>0</mn><mn>1</mn></msubsup>'\
        '<mo>&#x222B;</mo><mo>&#x222B;</mo><mi>x</mi><mo>&dd;</mo><mi>z</mi>'\
        '<mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x, (x, 0))) == \
        '<mrow><msup><mo>&#x222B;</mo><mn>0</mn></msup><mi>x</mi><mo>&dd;</mo>'\
        '<mi>x</mi></mrow>'


def test_presentation_mathml_matrices():
    A = Matrix([1, 2, 3])
    B = Matrix([[0, 5, 4], [2, 3, 1], [9, 7, 9]])
    mll_1 = mpp._print(A)
    assert mll_1.childNodes[1].nodeName == 'mtable'
    assert mll_1.childNodes[1].childNodes[0].nodeName == 'mtr'
    assert len(mll_1.childNodes[1].childNodes) == 3
    assert mll_1.childNodes[1].childNodes[0].childNodes[0].nodeName == 'mtd'
    assert len(mll_1.childNodes[1].childNodes[0].childNodes) == 1
    assert mll_1.childNodes[1].childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '1'
    assert mll_1.childNodes[1].childNodes[1].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_1.childNodes[1].childNodes[2].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '3'
    mll_2 = mpp._print(B)
    assert mll_2.childNodes[1].nodeName == 'mtable'
    assert mll_2.childNodes[1].childNodes[0].nodeName == 'mtr'
    assert len(mll_2.childNodes[1].childNodes) == 3
    assert mll_2.childNodes[1].childNodes[0].childNodes[0].nodeName == 'mtd'
    assert len(mll_2.childNodes[1].childNodes[0].childNodes) == 3
    assert mll_2.childNodes[1].childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '0'
    assert mll_2.childNodes[1].childNodes[0].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '5'
    assert mll_2.childNodes[1].childNodes[0].childNodes[2
        ].childNodes[0].childNodes[0].nodeValue == '4'
    assert mll_2.childNodes[1].childNodes[1].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_2.childNodes[1].childNodes[1].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '3'
    assert mll_2.childNodes[1].childNodes[1].childNodes[2
        ].childNodes[0].childNodes[0].nodeValue == '1'
    assert mll_2.childNodes[1].childNodes[2].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '9'
    assert mll_2.childNodes[1].childNodes[2].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '7'
    assert mll_2.childNodes[1].childNodes[2].childNodes[2
        ].childNodes[0].childNodes[0].nodeValue == '9'


def test_presentation_mathml_sums():
    mml_1 = mpp._print(Sum(x, (x, 1, 10)))
    assert mml_1.childNodes[0].nodeName == 'munderover'
    assert len(mml_1.childNodes[0].childNodes) == 3
    assert mml_1.childNodes[0].childNodes[0].childNodes[0
        ].nodeValue == '&#x2211;'
    assert len(mml_1.childNodes[0].childNodes[1].childNodes) == 3
    assert mml_1.childNodes[0].childNodes[2].childNodes[0
        ].nodeValue == '10'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == 'x'

    assert mpp.doprint(Sum(x, (x, 1, 10))) == \
        '<mrow><munderover><mo>&#x2211;</mo><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow><mn>10</mn></munderover><mi>x</mi></mrow>'
    assert mpp.doprint(Sum(x + y, (x, 1, 10))) == \
        '<mrow><munderover><mo>&#x2211;</mo><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow><mn>10</mn></munderover><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow><mo>)</mo></mrow></mrow>'


def test_presentation_mathml_add():
    mml = mpp._print(x**5 - x**4 + x)
    assert len(mml.childNodes) == 5
    assert mml.childNodes[0].childNodes[0].childNodes[0
        ].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].childNodes[0
        ].nodeValue == '5'
    assert mml.childNodes[1].childNodes[0].nodeValue == '-'
    assert mml.childNodes[2].childNodes[0].childNodes[0
        ].nodeValue == 'x'
    assert mml.childNodes[2].childNodes[1].childNodes[0
        ].nodeValue == '4'
    assert mml.childNodes[3].childNodes[0].nodeValue == '+'
    assert mml.childNodes[4].childNodes[0].nodeValue == 'x'


def test_presentation_mathml_Rational():
    mml_1 = mpp._print(Rational(1, 1))
    assert mml_1.nodeName == 'mn'

    mml_2 = mpp._print(Rational(2, 5))
    assert mml_2.nodeName == 'mfrac'
    assert mml_2.childNodes[0].childNodes[0].nodeValue == '2'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '5'


def test_presentation_mathml_constants():
    mml = mpp._print(I)
    assert mml.childNodes[0].nodeValue == '&ImaginaryI;'

    mml = mpp._print(E)
    assert mml.childNodes[0].nodeValue == '&ExponentialE;'

    mml = mpp._print(oo)
    assert mml.childNodes[0].nodeValue == '&#x221E;'

    mml = mpp._print(pi)
    assert mml.childNodes[0].nodeValue == '&pi;'

    assert mathml(hbar, printer='presentation') == '<mi>&#x210F;</mi>'
    assert mathml(S.TribonacciConstant, printer='presentation'
        ) == '<mi>TribonacciConstant</mi>'
    assert mathml(S.EulerGamma, printer='presentation'
        ) == '<mi>&#x3B3;</mi>'
    assert mathml(S.GoldenRatio, printer='presentation'
        ) == '<mi>&#x3A6;</mi>'

    assert mathml(zoo, printer='presentation') == \
        '<mover><mo>&#x221E;</mo><mo>~</mo></mover>'

    assert mathml(S.NaN, printer='presentation') == '<mi>NaN</mi>'

def test_presentation_mathml_trig():
    mml = mpp._print(sin(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'sin'

    mml = mpp._print(cos(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'cos'

    mml = mpp._print(tan(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'tan'

    mml = mpp._print(asin(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arcsin'

    mml = mpp._print(acos(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arccos'

    mml = mpp._print(atan(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arctan'

    mml = mpp._print(sinh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'sinh'

    mml = mpp._print(cosh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'cosh'

    mml = mpp._print(tanh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'tanh'

    mml = mpp._print(asinh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arcsinh'

    mml = mpp._print(atanh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arctanh'

    mml = mpp._print(acosh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arccosh'


def test_presentation_mathml_relational():
    mml_1 = mpp._print(Eq(x, 1))
    assert len(mml_1.childNodes) == 3
    assert mml_1.childNodes[0].nodeName == 'mi'
    assert mml_1.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml_1.childNodes[1].nodeName == 'mo'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == '='
    assert mml_1.childNodes[2].nodeName == 'mn'
    assert mml_1.childNodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mpp._print(Ne(1, x))
    assert len(mml_2.childNodes) == 3
    assert mml_2.childNodes[0].nodeName == 'mn'
    assert mml_2.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_2.childNodes[1].nodeName == 'mo'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '&#x2260;'
    assert mml_2.childNodes[2].nodeName == 'mi'
    assert mml_2.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_3 = mpp._print(Ge(1, x))
    assert len(mml_3.childNodes) == 3
    assert mml_3.childNodes[0].nodeName == 'mn'
    assert mml_3.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_3.childNodes[1].nodeName == 'mo'
    assert mml_3.childNodes[1].childNodes[0].nodeValue == '&#x2265;'
    assert mml_3.childNodes[2].nodeName == 'mi'
    assert mml_3.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_4 = mpp._print(Lt(1, x))
    assert len(mml_4.childNodes) == 3
    assert mml_4.childNodes[0].nodeName == 'mn'
    assert mml_4.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_4.childNodes[1].nodeName == 'mo'
    assert mml_4.childNodes[1].childNodes[0].nodeValue == '<'
    assert mml_4.childNodes[2].nodeName == 'mi'
    assert mml_4.childNodes[2].childNodes[0].nodeValue == 'x'


def test_presentation_symbol():
    mml = mpp._print(x)
    assert mml.nodeName == 'mi'
    assert mml.childNodes[0].nodeValue == 'x'
    del mml

    mml = mpp._print(Symbol("x^2"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mpp._print(Symbol("x__2"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mpp._print(Symbol("x_2"))
    assert mml.nodeName == 'msub'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mpp._print(Symbol("x^3_2"))
    assert mml.nodeName == 'msubsup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[2].nodeName == 'mi'
    assert mml.childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mpp._print(Symbol("x__3_2"))
    assert mml.nodeName == 'msubsup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[2].nodeName == 'mi'
    assert mml.childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mpp._print(Symbol("x_2_a"))
    assert mml.nodeName == 'msub'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mrow'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    del mml

    mml = mpp._print(Symbol("x^2^a"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mrow'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    del mml

    mml = mpp._print(Symbol("x__2__a"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mrow'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    del mml


def test_presentation_mathml_greek():
    mml = mpp._print(Symbol('alpha'))
    assert mml.nodeName == 'mi'
    assert mml.childNodes[0].nodeValue == '\N{GREEK SMALL LETTER ALPHA}'

    assert mpp.doprint(Symbol('alpha')) == '<mi>&#945;</mi>'
    assert mpp.doprint(Symbol('beta')) == '<mi>&#946;</mi>'
    assert mpp.doprint(Symbol('gamma')) == '<mi>&#947;</mi>'
    assert mpp.doprint(Symbol('delta')) == '<mi>&#948;</mi>'
    assert mpp.doprint(Symbol('epsilon')) == '<mi>&#949;</mi>'
    assert mpp.doprint(Symbol('zeta')) == '<mi>&#950;</mi>'
    assert mpp.doprint(Symbol('eta')) == '<mi>&#951;</mi>'
    assert mpp.doprint(Symbol('theta')) == '<mi>&#952;</mi>'
    assert mpp.doprint(Symbol('iota')) == '<mi>&#953;</mi>'
    assert mpp.doprint(Symbol('kappa')) == '<mi>&#954;</mi>'
    assert mpp.doprint(Symbol('lambda')) == '<mi>&#955;</mi>'
    assert mpp.doprint(Symbol('mu')) == '<mi>&#956;</mi>'
    assert mpp.doprint(Symbol('nu')) == '<mi>&#957;</mi>'
    assert mpp.doprint(Symbol('xi')) == '<mi>&#958;</mi>'
    assert mpp.doprint(Symbol('omicron')) == '<mi>&#959;</mi>'
    assert mpp.doprint(Symbol('pi')) == '<mi>&#960;</mi>'
    assert mpp.doprint(Symbol('rho')) == '<mi>&#961;</mi>'
    assert mpp.doprint(Symbol('varsigma')) == '<mi>&#962;</mi>'
    assert mpp.doprint(Symbol('sigma')) == '<mi>&#963;</mi>'
    assert mpp.doprint(Symbol('tau')) == '<mi>&#964;</mi>'
    assert mpp.doprint(Symbol('upsilon')) == '<mi>&#965;</mi>'
    assert mpp.doprint(Symbol('phi')) == '<mi>&#966;</mi>'
    assert mpp.doprint(Symbol('chi')) == '<mi>&#967;</mi>'
    assert mpp.doprint(Symbol('psi')) == '<mi>&#968;</mi>'
    assert mpp.doprint(Symbol('omega')) == '<mi>&#969;</mi>'

    assert mpp.doprint(Symbol('Alpha')) == '<mi>&#913;</mi>'
    assert mpp.doprint(Symbol('Beta')) == '<mi>&#914;</mi>'
    assert mpp.doprint(Symbol('Gamma')) == '<mi>&#915;</mi>'
    assert mpp.doprint(Symbol('Delta')) == '<mi>&#916;</mi>'
    assert mpp.doprint(Symbol('Epsilon')) == '<mi>&#917;</mi>'
    assert mpp.doprint(Symbol('Zeta')) == '<mi>&#918;</mi>'
    assert mpp.doprint(Symbol('Eta')) == '<mi>&#919;</mi>'
    assert mpp.doprint(Symbol('Theta')) == '<mi>&#920;</mi>'
    assert mpp.doprint(Symbol('Iota')) == '<mi>&#921;</mi>'
    assert mpp.doprint(Symbol('Kappa')) == '<mi>&#922;</mi>'
    assert mpp.doprint(Symbol('Lambda')) == '<mi>&#923;</mi>'
    assert mpp.doprint(Symbol('Mu')) == '<mi>&#924;</mi>'
    assert mpp.doprint(Symbol('Nu')) == '<mi>&#925;</mi>'
    assert mpp.doprint(Symbol('Xi')) == '<mi>&#926;</mi>'
    assert mpp.doprint(Symbol('Omicron')) == '<mi>&#927;</mi>'
    assert mpp.doprint(Symbol('Pi')) == '<mi>&#928;</mi>'
    assert mpp.doprint(Symbol('Rho')) == '<mi>&#929;</mi>'
    assert mpp.doprint(Symbol('Sigma')) == '<mi>&#931;</mi>'
    assert mpp.doprint(Symbol('Tau')) == '<mi>&#932;</mi>'
    assert mpp.doprint(Symbol('Upsilon')) == '<mi>&#933;</mi>'
    assert mpp.doprint(Symbol('Phi')) == '<mi>&#934;</mi>'
    assert mpp.doprint(Symbol('Chi')) == '<mi>&#935;</mi>'
    assert mpp.doprint(Symbol('Psi')) == '<mi>&#936;</mi>'
    assert mpp.doprint(Symbol('Omega')) == '<mi>&#937;</mi>'


def test_presentation_mathml_order():
    expr = x**3 + x**2*y + 3*x*y**3 + y**4

    mp = MathMLPresentationPrinter({'order': 'lex'})
    mml = mp._print(expr)
    assert mml.childNodes[0].nodeName == 'msup'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '3'

    assert mml.childNodes[6].nodeName == 'msup'
    assert mml.childNodes[6].childNodes[0].childNodes[0].nodeValue == 'y'
    assert mml.childNodes[6].childNodes[1].childNodes[0].nodeValue == '4'

    mp = MathMLPresentationPrinter({'order': 'rev-lex'})
    mml = mp._print(expr)

    assert mml.childNodes[0].nodeName == 'msup'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'y'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '4'

    assert mml.childNodes[6].nodeName == 'msup'
    assert mml.childNodes[6].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[6].childNodes[1].childNodes[0].nodeValue == '3'


def test_print_intervals():
    a = Symbol('a', real=True)
    assert mpp.doprint(Interval(0, a)) == \
        '<mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>]</mo></mrow>'
    assert mpp.doprint(Interval(0, a, False, False)) == \
        '<mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>]</mo></mrow>'
    assert mpp.doprint(Interval(0, a, True, False)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>]</mo></mrow>'
    assert mpp.doprint(Interval(0, a, False, True)) == \
        '<mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'
    assert mpp.doprint(Interval(0, a, True, True)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'


def test_print_tuples():
    assert mpp.doprint(Tuple(0,)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, a)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, a, a)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, 1, 2, 3, 4)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, 1, Tuple(2, 3, 4))) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mrow><mo>(</mo><mn>2</mn><mo>,</mo><mn>3'\
        '</mn><mo>,</mo><mn>4</mn><mo>)</mo></mrow><mo>)</mo></mrow>'


def test_print_re_im():
    assert mpp.doprint(re(x)) == \
        '<mrow><mi>&#8476;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(im(x)) == \
        '<mrow><mi>&#8465;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(re(x + 1, evaluate=False)) == \
        '<mrow><mi>&#8476;</mi><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(im(x + 1, evaluate=False)) == \
        '<mrow><mi>&#8465;</mi><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow></mrow>'


def test_print_Abs():
    assert mpp.doprint(Abs(x)) == \
        '<mrow><mo>|</mo><mi>x</mi><mo>|</mo></mrow>'
    assert mpp.doprint(Abs(x + 1)) == \
        '<mrow><mo>|</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>|</mo></mrow>'


def test_print_Determinant():
    assert mpp.doprint(Determinant(Matrix([[1, 2], [3, 4]]))) == \
        '<mrow><mo>|</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>|</mo></mrow>'


def test_presentation_settings():
    raises(TypeError, lambda: mathml(x, printer='presentation',
                                     method="garbage"))


def test_print_domains():
    from sympy.sets import Integers, Naturals, Naturals0, Reals, Complexes

    assert mpp.doprint(Complexes) == '<mi mathvariant="normal">&#x2102;</mi>'
    assert mpp.doprint(Integers) == '<mi mathvariant="normal">&#x2124;</mi>'
    assert mpp.doprint(Naturals) == '<mi mathvariant="normal">&#x2115;</mi>'
    assert mpp.doprint(Naturals0) == \
        '<msub><mi mathvariant="normal">&#x2115;</mi><mn>0</mn></msub>'
    assert mpp.doprint(Reals) == '<mi mathvariant="normal">&#x211D;</mi>'


def test_print_expression_with_minus():
    assert mpp.doprint(-x) == '<mrow><mo>-</mo><mi>x</mi></mrow>'
    assert mpp.doprint(-x/y) == \
        '<mrow><mo>-</mo><mfrac><mi>x</mi><mi>y</mi></mfrac></mrow>'
    assert mpp.doprint(-Rational(1, 2)) == \
        '<mrow><mo>-</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow>'


def test_print_AssocOp():
    from sympy.core.operations import AssocOp

    class TestAssocOp(AssocOp):
        identity = 0

    expr = TestAssocOp(1, 2)
    assert mpp.doprint(expr) == \
        '<mrow><mi>testassocop</mi><mn>1</mn><mn>2</mn></mrow>'


def test_print_basic():
    expr = Basic(S(1), S(2))
    assert mpp.doprint(expr) == \
        '<mrow><mi>basic</mi><mrow><mo>(</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>)</mo></mrow></mrow>'
    assert mp.doprint(expr) == '<basic><cn>1</cn><cn>2</cn></basic>'


def test_mat_delim_print():
    expr = Matrix([[1, 2], [3, 4]])
    assert mathml(expr, printer='presentation', mat_delim='[') == \
        '<mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd>'\
        '<mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn>'\
        '</mtd></mtr></mtable><mo>]</mo></mrow>'
    assert mathml(expr, printer='presentation', mat_delim='(') == \
        '<mrow><mo>(</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd>'\
        '</mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable><mo>)</mo></mrow>'
    assert mathml(expr, printer='presentation', mat_delim='') == \
        '<mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr>'\
        '<mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable>'


def test_ln_notation_print():
    expr = log(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mi>log</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(expr, printer='presentation', ln_notation=False) == \
        '<mrow><mi>log</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(expr, printer='presentation', ln_notation=True) == \
        '<mrow><mi>ln</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'


def test_mul_symbol_print():
    expr = x * y
    assert mathml(expr, printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol=None) == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol='dot') == \
        '<mrow><mi>x</mi><mo>&#xB7;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol='ldot') == \
        '<mrow><mi>x</mi><mo>&#x2024;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol='times') == \
        '<mrow><mi>x</mi><mo>&#xD7;</mo><mi>y</mi></mrow>'


def test_print_lerchphi():
    assert mpp.doprint(lerchphi(1, 2, 3)) == \
        '<mrow><mi>&#x3A6;</mi><mrow><mo>(</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>)</mo></mrow></mrow>'


def test_print_polylog():
    assert mp.doprint(polylog(x, y)) == \
        '<apply><polylog/><ci>x</ci><ci>y</ci></apply>'
    assert mpp.doprint(polylog(x, y)) == \
        '<mrow><msub><mi>Li</mi><mi>x</mi></msub><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow>'


def test_print_set_frozenset():
    f = frozenset({1, 5, 3})
    assert mpp.doprint(f) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>5</mn><mo>}</mo></mrow>'
    s = set({1, 2, 3})
    assert mpp.doprint(s) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>}</mo></mrow>'


def test_print_FiniteSet():
    f1 = FiniteSet(x, 1, 3)
    assert mpp.doprint(f1) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi><mo>}</mo></mrow>'


def test_print_LambertW():
    assert mpp.doprint(LambertW(x)) == '<mrow><mi>W</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(LambertW(x, y)) == '<mrow><mi>W</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'


def test_print_EmptySet():
    assert mpp.doprint(S.EmptySet) == '<mo>&#x2205;</mo>'


def test_print_UniversalSet():
    assert mpp.doprint(S.UniversalSet) == '<mo>&#x1D54C;</mo>'


def test_print_spaces():
    assert mpp.doprint(HilbertSpace()) == '<mi>&#x210B;</mi>'
    assert mpp.doprint(ComplexSpace(2)) == '<msup>&#x1D49E;<mn>2</mn></msup>'
    assert mpp.doprint(FockSpace()) == '<mi>&#x2131;</mi>'


def test_print_constants():
    assert mpp.doprint(hbar) == '<mi>&#x210F;</mi>'
    assert mpp.doprint(S.TribonacciConstant) == '<mi>TribonacciConstant</mi>'
    assert mpp.doprint(S.GoldenRatio) == '<mi>&#x3A6;</mi>'
    assert mpp.doprint(S.EulerGamma) == '<mi>&#x3B3;</mi>'


def test_print_Contains():
    assert mpp.doprint(Contains(x, S.Naturals)) == \
        '<mrow><mi>x</mi><mo>&#x2208;</mo><mi mathvariant="normal">&#x2115;</mi></mrow>'


def test_print_Dagger():
    x = symbols('x', commutative=False)
    assert mpp.doprint(Dagger(x)) == '<msup><mi>x</mi>&#x2020;</msup>'


def test_print_SetOp():
    f1 = FiniteSet(x, 1, 3)
    f2 = FiniteSet(y, 2, 4)

    prntr = lambda x: mathml(x, printer='presentation')

    assert prntr(Union(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x222A;</mo><mrow><mo>{</mo><mn>2</mn><mo>,</mo>'\
    '<mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'
    assert prntr(Intersection(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x2229;</mo><mrow><mo>{</mo><mn>2</mn>'\
    '<mo>,</mo><mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'
    assert prntr(Complement(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x2216;</mo><mrow><mo>{</mo><mn>2</mn>'\
    '<mo>,</mo><mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'
    assert prntr(SymmetricDifference(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x2206;</mo><mrow><mo>{</mo><mn>2</mn>'\
    '<mo>,</mo><mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'

    A = FiniteSet(a)
    C = FiniteSet(c)
    D = FiniteSet(d)

    U1 = Union(C, D, evaluate=False)
    I1 = Intersection(C, D, evaluate=False)
    C1 = Complement(C, D, evaluate=False)
    D1 = SymmetricDifference(C, D, evaluate=False)
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(C, D)

    assert prntr(Union(A, I1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x222A;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x2229;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(Intersection(A, C1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x2229;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x2216;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(Complement(A, D1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x2216;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x2206;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(SymmetricDifference(A, P1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x2206;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x00d7;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(ProductSet(A, U1)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x00d7;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x222A;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'


def test_print_logic():
    assert mpp.doprint(And(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x2227;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Or(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x2228;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Xor(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x22BB;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Implies(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x21D2;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Equivalent(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x21D4;</mo><mi>y</mi></mrow>'

    assert mpp.doprint(And(Eq(x, y), x > 4)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2227;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    assert mpp.doprint(And(Eq(x, 3), y < 3, x > y + 1)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow><mo>&#x2227;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mrow><mi>y</mi><mo>+</mo><mn>1</mn></mrow>'\
        '</mrow><mo>&#x2227;</mo><mrow><mi>y</mi><mo><</mo><mn>3</mn></mrow></mrow>'
    assert mpp.doprint(Or(Eq(x, y), x > 4)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2228;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    assert mpp.doprint(And(Eq(x, 3), Or(y < 3, x > y + 1))) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow>'\
        '<mo>&#x2227;</mo><mrow><mo>(</mo><mrow><mrow><mi>x</mi><mo>></mo><mrow>'\
        '<mi>y</mi><mo>+</mo><mn>1</mn></mrow></mrow><mo>&#x2228;</mo><mrow>'\
        '<mi>y</mi><mo><</mo><mn>3</mn></mrow></mrow><mo>)</mo></mrow></mrow>'

    assert mpp.doprint(Not(x)) == '<mrow><mo>&#xAC;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Not(And(x, y))) == \
        '<mrow><mo>&#xAC;</mo><mrow><mo>(</mo><mrow><mi>x</mi><mo>&#x2227;</mo><mi>y</mi></mrow><mo>)</mo></mrow></mrow>'


def test_root_notation_print():
    assert mathml(x**(S.One/3), printer='presentation') == \
        '<mroot><mi>x</mi><mn>3</mn></mroot>'
    assert mathml(x**(S.One/3), printer='presentation', root_notation=False) ==\
        '<msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup>'
    assert mathml(x**(S.One/3), printer='content') == \
        '<apply><root/><degree><cn>3</cn></degree><ci>x</ci></apply>'
    assert mathml(x**(S.One/3), printer='content', root_notation=False) == \
        '<apply><power/><ci>x</ci><apply><divide/><cn>1</cn><cn>3</cn></apply></apply>'
    assert mathml(x**(Rational(-1, 3)), printer='presentation') == \
        '<mfrac><mn>1</mn><mroot><mi>x</mi><mn>3</mn></mroot></mfrac>'
    assert mathml(x**(Rational(-1, 3)), printer='presentation', root_notation=False) \
        == '<mfrac><mn>1</mn><msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup></mfrac>'


def test_fold_frac_powers_print():
    expr = x ** Rational(5, 2)
    assert mathml(expr, printer='presentation') == \
        '<msup><mi>x</mi><mfrac><mn>5</mn><mn>2</mn></mfrac></msup>'
    assert mathml(expr, printer='presentation', fold_frac_powers=True) == \
        '<msup><mi>x</mi><mfrac bevelled="true"><mn>5</mn><mn>2</mn></mfrac></msup>'
    assert mathml(expr, printer='presentation', fold_frac_powers=False) == \
        '<msup><mi>x</mi><mfrac><mn>5</mn><mn>2</mn></mfrac></msup>'


def test_fold_short_frac_print():
    expr = Rational(2, 5)
    assert mathml(expr, printer='presentation') == \
        '<mfrac><mn>2</mn><mn>5</mn></mfrac>'
    assert mathml(expr, printer='presentation', fold_short_frac=True) == \
        '<mfrac bevelled="true"><mn>2</mn><mn>5</mn></mfrac>'
    assert mathml(expr, printer='presentation', fold_short_frac=False) == \
        '<mfrac><mn>2</mn><mn>5</mn></mfrac>'


def test_print_factorials():
    assert mpp.doprint(factorial(x)) == '<mrow><mi>x</mi><mo>!</mo></mrow>'
    assert mpp.doprint(factorial(x + 1)) == \
        '<mrow><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow><mo>!</mo></mrow>'
    assert mpp.doprint(factorial2(x)) == '<mrow><mi>x</mi><mo>!!</mo></mrow>'
    assert mpp.doprint(factorial2(x + 1)) == \
        '<mrow><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow><mo>!!</mo></mrow>'
    assert mpp.doprint(binomial(x, y)) == \
        '<mrow><mo>(</mo><mfrac linethickness="0"><mi>x</mi><mi>y</mi></mfrac><mo>)</mo></mrow>'
    assert mpp.doprint(binomial(4, x + y)) == \
        '<mrow><mo>(</mo><mfrac linethickness="0"><mn>4</mn><mrow><mi>x</mi>'\
        '<mo>+</mo><mi>y</mi></mrow></mfrac><mo>)</mo></mrow>'


def test_print_floor():
    expr = floor(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>&#8970;</mo><mi>x</mi><mo>&#8971;</mo></mrow>'


def test_print_ceiling():
    expr = ceiling(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>&#8968;</mo><mi>x</mi><mo>&#8969;</mo></mrow>'


def test_print_Lambda():
    expr = Lambda(x, x+1)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>(</mo><mi>x</mi><mo>&#x21A6;</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow>'
    expr = Lambda((x, y), x + y)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>(</mo><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow><mo>&#x21A6;</mo><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow><mo>)</mo></mrow>'


def test_print_conjugate():
    assert mpp.doprint(conjugate(x)) == \
        '<menclose notation="top"><mi>x</mi></menclose>'
    assert mpp.doprint(conjugate(x + 1)) == \
        '<mrow><menclose notation="top"><mi>x</mi></menclose><mo>+</mo><mn>1</mn></mrow>'


def test_print_AccumBounds():
    a = Symbol('a', real=True)
    assert mpp.doprint(AccumBounds(0, 1)) == '<mrow><mo>&#10216;</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>&#10217;</mo></mrow>'
    assert mpp.doprint(AccumBounds(0, a)) == '<mrow><mo>&#10216;</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>&#10217;</mo></mrow>'
    assert mpp.doprint(AccumBounds(a + 1, a + 2)) == '<mrow><mo>&#10216;</mo><mrow><mi>a</mi><mo>+</mo><mn>1</mn></mrow><mo>,</mo><mrow><mi>a</mi><mo>+</mo><mn>2</mn></mrow><mo>&#10217;</mo></mrow>'


def test_print_Float():
    assert mpp.doprint(Float(1e100)) == '<mrow><mn>1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>100</mn></msup></mrow>'
    assert mpp.doprint(Float(1e-100)) == '<mrow><mn>1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>-100</mn></msup></mrow>'
    assert mpp.doprint(Float(-1e100)) == '<mrow><mn>-1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>100</mn></msup></mrow>'
    assert mpp.doprint(Float(1.0*oo)) == '<mi>&#x221E;</mi>'
    assert mpp.doprint(Float(-1.0*oo)) == '<mrow><mo>-</mo><mi>&#x221E;</mi></mrow>'


def test_print_different_functions():
    assert mpp.doprint(gamma(x)) == '<mrow><mi>&#x393;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(lowergamma(x, y)) == '<mrow><mi>&#x3B3;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(uppergamma(x, y)) == '<mrow><mi>&#x393;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(zeta(x)) == '<mrow><mi>&#x3B6;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(zeta(x, y)) == '<mrow><mi>&#x3B6;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(dirichlet_eta(x)) ==  '<mrow><mi>&#x3B7;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(elliptic_k(x)) == '<mrow><mi>&#x39A;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(totient(x)) == '<mrow><mi>&#x3D5;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(reduced_totient(x)) == '<mrow><mi>&#x3BB;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(primenu(x)) == '<mrow><mi>&#x3BD;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(primeomega(x)) == '<mrow><mi>&#x3A9;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(fresnels(x)) == '<mrow><mi>S</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(fresnelc(x)) ==  '<mrow><mi>C</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Heaviside(x)) == '<mrow><mi>&#x398;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>)</mo></mrow></mrow>'


def test_mathml_builtins():
    assert mpp.doprint(None) == '<mi>None</mi>'
    assert mpp.doprint(true) == '<mi>True</mi>'
    assert mpp.doprint(false) == '<mi>False</mi>'


def test_mathml_Range():
    assert mpp.doprint(Range(1, 51)) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mi>&#8230;</mi><mo>,</mo><mn>50</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(1, 4)) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(0, 3, 1)) == \
        '<mrow><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(0, 30, 1)) == \
        '<mrow><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mi>&#8230;</mi><mo>,</mo><mn>29</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(30, 1, -1)) == \
        '<mrow><mo>{</mo><mn>30</mn><mo>,</mo><mn>29</mn><mo>,</mo><mi>&#8230;</mi><mo>,</mo><mn>2</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(0, oo, 2)) == \
        '<mrow><mo>{</mo><mn>0</mn><mo>,</mo><mn>2</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'
    assert mpp.doprint(Range(oo, -2, -2)) == \
        '<mrow><mo>{</mo><mi>&#8230;</mi><mo>,</mo><mn>2</mn><mo>,</mo><mn>0</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(-2, -oo, -1)) == \
        '<mrow><mo>{</mo><mn>-2</mn><mo>,</mo><mn>-3</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'


def test_print_exp():
    assert mpp.doprint(exp(x)) == \
        '<msup><mi>&ExponentialE;</mi><mi>x</mi></msup>'
    assert mpp.doprint(exp(1) + exp(2)) == \
        '<mrow><mi>&ExponentialE;</mi><mo>+</mo><msup><mi>&ExponentialE;</mi><mn>2</mn></msup></mrow>'


def test_print_MinMax():
    assert mpp.doprint(Min(x, y)) == \
        '<mrow><mo>min</mo><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Min(x, 2, x**3)) == \
        '<mrow><mo>min</mo><mrow><mo>(</mo><mn>2</mn><mo>,</mo><mi>x</mi><mo>,</mo><msup><mi>x</mi><mn>3</mn></msup><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Max(x, y)) == \
        '<mrow><mo>max</mo><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Max(x, 2, x**3)) == \
        '<mrow><mo>max</mo><mrow><mo>(</mo><mn>2</mn><mo>,</mo><mi>x</mi><mo>,</mo><msup><mi>x</mi><mn>3</mn></msup><mo>)</mo></mrow></mrow>'


def test_mathml_presentation_numbers():
    n = Symbol('n')
    assert mathml(catalan(n), printer='presentation') == \
        '<msub><mi>C</mi><mi>n</mi></msub>'
    assert mathml(bernoulli(n), printer='presentation') == \
        '<msub><mi>B</mi><mi>n</mi></msub>'
    assert mathml(bell(n), printer='presentation') == \
        '<msub><mi>B</mi><mi>n</mi></msub>'
    assert mathml(euler(n), printer='presentation') == \
        '<msub><mi>E</mi><mi>n</mi></msub>'
    assert mathml(fibonacci(n), printer='presentation') == \
        '<msub><mi>F</mi><mi>n</mi></msub>'
    assert mathml(lucas(n), printer='presentation') == \
        '<msub><mi>L</mi><mi>n</mi></msub>'
    assert mathml(tribonacci(n), printer='presentation') == \
        '<msub><mi>T</mi><mi>n</mi></msub>'
    assert mathml(bernoulli(n, x), printer='presentation') == \
        mathml(bell(n, x), printer='presentation') == \
        '<mrow><msub><mi>B</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(euler(n, x), printer='presentation') == \
        '<mrow><msub><mi>E</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(fibonacci(n, x), printer='presentation') == \
        '<mrow><msub><mi>F</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(tribonacci(n, x), printer='presentation') == \
        '<mrow><msub><mi>T</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'


def test_mathml_presentation_mathieu():
    assert mathml(mathieuc(x, y, z), printer='presentation') == \
        '<mrow><mi>C</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'
    assert mathml(mathieus(x, y, z), printer='presentation') == \
        '<mrow><mi>S</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'
    assert mathml(mathieucprime(x, y, z), printer='presentation') == \
        '<mrow><mi>C&#x2032;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'
    assert mathml(mathieusprime(x, y, z), printer='presentation') == \
        '<mrow><mi>S&#x2032;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'


def test_mathml_presentation_stieltjes():
    assert mathml(stieltjes(n), printer='presentation') == \
         '<msub><mi>&#x03B3;</mi><mi>n</mi></msub>'
    assert mathml(stieltjes(n, x), printer='presentation') == \
         '<mrow><msub><mi>&#x03B3;</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'


def test_print_matrix_symbol():
    A = MatrixSymbol('A', 1, 2)
    assert mpp.doprint(A) == '<mi>A</mi>'
    assert mp.doprint(A) == '<ci>A</ci>'
    assert mathml(A, printer='presentation', mat_symbol_style="bold") == \
        '<mi mathvariant="bold">A</mi>'
    # No effect in content printer
    assert mathml(A, mat_symbol_style="bold") == '<ci>A</ci>'


def test_print_hadamard():
    from sympy.matrices.expressions import HadamardProduct
    from sympy.matrices.expressions import Transpose

    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)

    assert mathml(HadamardProduct(X, Y*Y), printer="presentation") == \
        '<mrow>' \
        '<mi>X</mi>' \
        '<mo>&#x2218;</mo>' \
        '<msup><mi>Y</mi><mn>2</mn></msup>' \
        '</mrow>'

    assert mathml(HadamardProduct(X, Y)*Y, printer="presentation") == \
        '<mrow>' \
        '<mrow><mo>(</mo>' \
        '<mrow><mi>X</mi><mo>&#x2218;</mo><mi>Y</mi></mrow>' \
        '<mo>)</mo></mrow>' \
        '<mo>&InvisibleTimes;</mo><mi>Y</mi>' \
        '</mrow>'

    assert mathml(HadamardProduct(X, Y, Y), printer="presentation") == \
        '<mrow>' \
        '<mi>X</mi><mo>&#x2218;</mo>' \
        '<mi>Y</mi><mo>&#x2218;</mo>' \
        '<mi>Y</mi>' \
        '</mrow>'

    assert mathml(
        Transpose(HadamardProduct(X, Y)), printer="presentation") == \
            '<msup>' \
            '<mrow><mo>(</mo>' \
            '<mrow><mi>X</mi><mo>&#x2218;</mo><mi>Y</mi></mrow>' \
            '<mo>)</mo></mrow>' \
            '<mo>T</mo>' \
            '</msup>'


def test_print_random_symbol():
    R = RandomSymbol(Symbol('R'))
    assert mpp.doprint(R) == '<mi>R</mi>'
    assert mp.doprint(R) == '<ci>R</ci>'


def test_print_IndexedBase():
    assert mathml(IndexedBase(a)[b], printer='presentation') == \
        '<msub><mi>a</mi><mi>b</mi></msub>'
    assert mathml(IndexedBase(a)[b, c, d], printer='presentation') == \
        '<msub><mi>a</mi><mrow><mo>(</mo><mi>b</mi><mo>,</mo><mi>c</mi><mo>,</mo><mi>d</mi><mo>)</mo></mrow></msub>'
    assert mathml(IndexedBase(a)[b]*IndexedBase(c)[d]*IndexedBase(e),
                  printer='presentation') == \
                  '<mrow><msub><mi>a</mi><mi>b</mi></msub><mo>&InvisibleTimes;'\
                  '</mo><msub><mi>c</mi><mi>d</mi></msub><mo>&InvisibleTimes;</mo><mi>e</mi></mrow>'


def test_print_Indexed():
    assert mathml(IndexedBase(a), printer='presentation') == '<mi>a</mi>'
    assert mathml(IndexedBase(a/b), printer='presentation') == \
        '<mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow>'
    assert mathml(IndexedBase((a, b)), printer='presentation') == \
        '<mrow><mo>(</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>)</mo></mrow>'

def test_print_MatrixElement():
    i, j = symbols('i j')
    A = MatrixSymbol('A', i, j)
    assert mathml(A[0,0],printer = 'presentation') == \
        '<msub><mi>A</mi><mrow><mn>0</mn><mo>,</mo><mn>0</mn></mrow></msub>'
    assert mathml(A[i,j], printer = 'presentation') == \
        '<msub><mi>A</mi><mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub>'
    assert mathml(A[i*j,0], printer = 'presentation') == \
        '<msub><mi>A</mi><mrow><mrow><mi>i</mi><mo>&InvisibleTimes;</mo><mi>j</mi></mrow><mo>,</mo><mn>0</mn></mrow></msub>'


def test_print_Vector():
    ACS = CoordSys3D('A')
    assert mathml(Cross(ACS.i, ACS.j*ACS.x*3 + ACS.k), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><mrow><mo>(</mo><mrow>'\
        '<mrow><mo>(</mo><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '</mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover>'\
        '<mi mathvariant="bold">k</mi><mo>^</mo></mover><mi mathvariant="bold">'\
        'A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Cross(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(x*Cross(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Cross(x*ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mo>-</mo><mrow><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub>'\
        '<mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">i</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '<mo>)</mo></mrow></mrow></mrow>'
    assert mathml(Curl(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo>'\
        '<mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo>'\
        '<msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Curl(3*x*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Curl(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2207;</mo>'\
        '<mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo>'\
        '</mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Curl(3*x*ACS.x*ACS.j + ACS.i), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mo>(</mo><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Divergence(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo>'\
        '<mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Divergence(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2207;</mo>'\
        '<mo>&#xB7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '<mo>)</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Divergence(3*x*ACS.x*ACS.j + ACS.i), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mo>(</mo><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '<mo>&InvisibleTimes;</mo><mi>x</mi></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '<mo>)</mo></mrow></mrow>'
    assert mathml(Dot(ACS.i, ACS.j*ACS.x*3+ACS.k), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mrow><mo>(</mo><mrow>'\
        '<mrow><mo>(</mo><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '</mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover>'\
        '<mi mathvariant="bold">k</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Dot(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Dot(x*ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mrow><mo>(</mo><mrow>'\
        '<mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Dot(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Gradient(ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Gradient(ACS.x + 3*ACS.y), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Gradient(ACS.x), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2207;</mo>'\
        '<msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi>'\
        '</msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Gradient(x*ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Cross(ACS.z, ACS.x), printer='presentation') == \
        '<mrow><mo>-</mo><mrow><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub>'\
        '<mi mathvariant="bold">z</mi><mi mathvariant="bold">A</mi></msub></mrow></mrow>'
    assert mathml(Laplacian(ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Laplacian(ACS.x + 3*ACS.y), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Laplacian(ACS.x), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2206;</mo>'\
        '<msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi>'\
        '</msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Laplacian(x*ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow></mrow>'

@XFAIL
def test_vector_cross_xfail():
    ACS = CoordSys3D('A')
    assert mathml(Cross(ACS.x, ACS.z) + Cross(ACS.z, ACS.x), printer='presentation') == \
        '<mover><mi mathvariant="bold">0</mi><mo>^</mo></mover>'

def test_print_elliptic_f():
    assert mathml(elliptic_f(x, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a5;</mi><mrow><mo>(</mo><mi>x</mi><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mathml(elliptic_f(x/y, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a5;</mi><mrow><mo>(</mo><mrow><mfrac><mi>x</mi><mi>y</mi></mfrac></mrow><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'

def test_print_elliptic_e():
    assert mathml(elliptic_e(x), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a4;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(elliptic_e(x, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a4;</mi><mrow><mo>(</mo><mi>x</mi><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'

def test_print_elliptic_pi():
    assert mathml(elliptic_pi(x, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d6f1;</mi><mrow><mo>(</mo><mi>x</mi><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mathml(elliptic_pi(x, y, z), printer = 'presentation') == \
        '<mrow><mi>&#x1d6f1;</mi><mrow><mo>(</mo><mi>x</mi><mo>;</mo><mi>y</mi><mo>|</mo><mi>z</mi><mo>)</mo></mrow></mrow>'

def test_print_Ei():
    assert mathml(Ei(x), printer = 'presentation') == \
        '<mrow><mi>Ei</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(Ei(x**y), printer = 'presentation') == \
        '<mrow><mi>Ei</mi><mrow><mo>(</mo><msup><mi>x</mi><mi>y</mi></msup><mo>)</mo></mrow></mrow>'

def test_print_expint():
    assert mathml(expint(x, y), printer = 'presentation') == \
        '<mrow><msub><mo>E</mo><mi>x</mi></msub><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mathml(expint(IndexedBase(x)[1], IndexedBase(x)[2]), printer = 'presentation') == \
        '<mrow><msub><mo>E</mo><msub><mi>x</mi><mn>1</mn></msub></msub><mrow><mo>(</mo><msub><mi>x</mi><mn>2</mn></msub><mo>)</mo></mrow></mrow>'

def test_print_jacobi():
    assert mathml(jacobi(n, a, b, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>P</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_gegenbauer():
    assert mathml(gegenbauer(n, a, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>C</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_chebyshevt():
    assert mathml(chebyshevt(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>T</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_chebyshevu():
    assert mathml(chebyshevu(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>U</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_legendre():
    assert mathml(legendre(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>P</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_assoc_legendre():
    assert mathml(assoc_legendre(n, a, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>P</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_laguerre():
    assert mathml(laguerre(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>L</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_assoc_laguerre():
    assert mathml(assoc_laguerre(n, a, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>L</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_hermite():
    assert mathml(hermite(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>H</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_mathml_SingularityFunction():
    assert mathml(SingularityFunction(x, 4, 5), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>-</mo><mn>4</mn></mrow><mo>&#10217;</mo></mrow><mn>5</mn></msup>'
    assert mathml(SingularityFunction(x, -3, 4), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>+</mo><mn>3</mn></mrow><mo>&#10217;</mo></mrow><mn>4</mn></msup>'
    assert mathml(SingularityFunction(x, 0, 4), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mi>x</mi><mo>&#10217;</mo></mrow><mn>4</mn></msup>'
    assert mathml(SingularityFunction(x, a, n), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mrow><mo>-</mo><mi>a</mi></mrow><mo>+</mo><mi>x</mi></mrow><mo>&#10217;</mo></mrow><mi>n</mi></msup>'
    assert mathml(SingularityFunction(x, 4, -2), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>-</mo><mn>4</mn></mrow><mo>&#10217;</mo></mrow><mn>-2</mn></msup>'
    assert mathml(SingularityFunction(x, 4, -1), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>-</mo><mn>4</mn></mrow><mo>&#10217;</mo></mrow><mn>-1</mn></msup>'


def test_mathml_matrix_functions():
    from sympy.matrices import Adjoint, Inverse, Transpose
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert mathml(Adjoint(X), printer='presentation') == \
        '<msup><mi>X</mi><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X + Y), printer='presentation') == \
        '<msup><mrow><mo>(</mo><mrow><mi>X</mi><mo>+</mo><mi>Y</mi></mrow><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X) + Adjoint(Y), printer='presentation') == \
        '<mrow><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>+</mo><msup>' \
        '<mi>Y</mi><mo>&#x2020;</mo></msup></mrow>'
    assert mathml(Adjoint(X*Y), printer='presentation') == \
        '<msup><mrow><mo>(</mo><mrow><mi>X</mi><mo>&InvisibleTimes;</mo>' \
        '<mi>Y</mi></mrow><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(Y)*Adjoint(X), printer='presentation') == \
        '<mrow><msup><mi>Y</mi><mo>&#x2020;</mo></msup><mo>&InvisibleTimes;' \
        '</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup></mrow>'
    assert mathml(Adjoint(X**2), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mn>2</mn></msup><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X)**2, printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>)</mo></mrow><mn>2</mn></msup>'
    assert mathml(Adjoint(Inverse(X)), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mn>-1</mn></msup><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Inverse(Adjoint(X)), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>)</mo></mrow><mn>-1</mn></msup>'
    assert mathml(Adjoint(Transpose(X)), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>T</mo></msup><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Transpose(Adjoint(X)), printer='presentation') ==  \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>)</mo></mrow><mo>T</mo></msup>'
    assert mathml(Transpose(Adjoint(X) + Y), printer='presentation') ==  \
        '<msup><mrow><mo>(</mo><mrow><msup><mi>X</mi><mo>&#x2020;</mo></msup>' \
        '<mo>+</mo><mi>Y</mi></mrow><mo>)</mo></mrow><mo>T</mo></msup>'
    assert mathml(Transpose(X), printer='presentation') == \
        '<msup><mi>X</mi><mo>T</mo></msup>'
    assert mathml(Transpose(X + Y), printer='presentation') == \
        '<msup><mrow><mo>(</mo><mrow><mi>X</mi><mo>+</mo><mi>Y</mi></mrow><mo>)</mo></mrow><mo>T</mo></msup>'


def test_mathml_special_matrices():
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    assert mathml(Identity(4), printer='presentation') == '<mi>&#x1D540;</mi>'
    assert mathml(ZeroMatrix(2, 2), printer='presentation') == '<mn>&#x1D7D8</mn>'
    assert mathml(OneMatrix(2, 2), printer='presentation') == '<mn>&#x1D7D9</mn>'

def test_mathml_piecewise():
    from sympy.functions.elementary.piecewise import Piecewise
    # Content MathML
    assert mathml(Piecewise((x, x <= 1), (x**2, True))) == \
        '<piecewise><piece><ci>x</ci><apply><leq/><ci>x</ci><cn>1</cn></apply></piece><otherwise><apply><power/><ci>x</ci><cn>2</cn></apply></otherwise></piecewise>'

    raises(ValueError, lambda: mathml(Piecewise((x, x <= 1))))


def test_issue_17857():
    assert mathml(Range(-oo, oo), printer='presentation') == \
        '<mrow><mo>{</mo><mi>&#8230;</mi><mo>,</mo><mn>-1</mn><mo>,</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'
    assert mathml(Range(oo, -oo, -1), printer='presentation') == \
        '<mrow><mo>{</mo><mi>&#8230;</mi><mo>,</mo><mn>1</mn><mo>,</mo><mn>0</mn><mo>,</mo><mn>-1</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'


def test_float_roundtrip():
    x = sympify(0.8975979010256552)
    y = float(mp.doprint(x).strip('</cn>'))
    assert x == y


def test_content_mathml_disable_split_super_sub():
    mp = MathMLContentPrinter()
    assert mp.doprint(Symbol('u_b')) == '<ci><mml:msub><mml:mi>u</mml:mi><mml:mi>b</mml:mi></mml:msub></ci>'
    mp = MathMLContentPrinter({'disable_split_super_sub': False})
    assert mp.doprint(Symbol('u_b')) == '<ci><mml:msub><mml:mi>u</mml:mi><mml:mi>b</mml:mi></mml:msub></ci>'
    mp = MathMLContentPrinter({'disable_split_super_sub': True})
    assert mp.doprint(Symbol('u_b')) == '<ci>u_b</ci>'

def test_presentation_mathml_disable_split_super_sub():
    mpp = MathMLPresentationPrinter()
    assert mpp.doprint(Symbol('u_b')) == '<msub><mi>u</mi><mi>b</mi></msub>'
    mpp = MathMLPresentationPrinter({'disable_split_super_sub': False})
    assert mpp.doprint(Symbol('u_b')) == '<msub><mi>u</mi><mi>b</mi></msub>'
    mpp = MathMLPresentationPrinter({'disable_split_super_sub': True})
    assert mpp.doprint(Symbol('u_b')) == '<mi>u_b</mi>'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\npyio.py ===
from ._npyio_impl import DataSource, NpzFile, __doc__  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\stride_tricks.py ===
from ._stride_tricks_impl import __doc__, as_strided, sliding_window_view  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\user_array.py ===
from ._user_array_impl import __doc__, container  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\split.py ===
import onnx

from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase


class QSplit(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0])
        if quantized_input_names is None:
            return super().quantize()

        quantized_node_name = ""
        if node.name:
            quantized_node_name = node.name + "_quant"
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))

        # Output just derive the scale/zero from input
        quantized_output_names = []
        for output_name in node.output:
            quantized_output_name = output_name + "quantized"
            quantized_output_names.append(quantized_output_name)
            q_output = QuantizedValue(
                output_name,
                quantized_output_name,
                scale_names[0],
                zero_point_names[0],
                QuantizedValueType.Input,
            )
            self.quantizer.quantized_value_map[output_name] = q_output

        if len(node.input) > 1:
            quantized_input_names.extend(node.input[1:])
        quantized_node = onnx.helper.make_node(
            node.op_type, quantized_input_names, quantized_output_names, quantized_node_name, **kwargs
        )

        nodes.append(quantized_node)
        self.quantizer.new_nodes += nodes


class QDQSplit(QDQOperatorBase):
    def quantize(self):
        node = self.node
        assert node.op_type == "Split"

        if not self.quantizer.is_tensor_quantized(node.input[0]):
            self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            for output in node.output:
                self.quantizer.quantize_output_same_as_input(output, node.input[0], node.name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\_utils.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs import lib
from pandas.errors import LossySetitemError

from pandas.core.dtypes.cast import np_can_hold_element
from pandas.core.dtypes.common import is_numeric_dtype

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        npt,
    )


def to_numpy_dtype_inference(
    arr: ArrayLike, dtype: npt.DTypeLike | None, na_value, hasna: bool
) -> tuple[npt.DTypeLike, Any]:
    if dtype is None and is_numeric_dtype(arr.dtype):
        dtype_given = False
        if hasna:
            if arr.dtype.kind == "b":
                dtype = np.dtype(np.object_)
            else:
                if arr.dtype.kind in "iu":
                    dtype = np.dtype(np.float64)
                else:
                    dtype = arr.dtype.numpy_dtype  # type: ignore[union-attr]
                if na_value is lib.no_default:
                    na_value = np.nan
        else:
            dtype = arr.dtype.numpy_dtype  # type: ignore[union-attr]
    elif dtype is not None:
        dtype = np.dtype(dtype)
        dtype_given = True
    else:
        dtype_given = True

    if na_value is lib.no_default:
        if dtype is None or not hasna:
            na_value = arr.dtype.na_value
        elif dtype.kind == "f":  # type: ignore[union-attr]
            na_value = np.nan
        elif dtype.kind == "M":  # type: ignore[union-attr]
            na_value = np.datetime64("nat")
        elif dtype.kind == "m":  # type: ignore[union-attr]
            na_value = np.timedelta64("nat")
        else:
            na_value = arr.dtype.na_value

    if not dtype_given and hasna:
        try:
            np_can_hold_element(dtype, na_value)  # type: ignore[arg-type]
        except LossySetitemError:
            dtype = np.dtype(np.object_)
    return dtype, na_value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\tethering.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Tethering (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def bind(
        port: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Request browser port binding.

    :param port: Port number to bind.
    '''
    params: T_JSON_DICT = dict()
    params['port'] = port
    cmd_dict: T_JSON_DICT = {
        'method': 'Tethering.bind',
        'params': params,
    }
    json = yield cmd_dict


def unbind(
        port: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Request browser port unbinding.

    :param port: Port number to unbind.
    '''
    params: T_JSON_DICT = dict()
    params['port'] = port
    cmd_dict: T_JSON_DICT = {
        'method': 'Tethering.unbind',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Tethering.accepted')
@dataclass
class Accepted:
    '''
    Informs that port was successfully bound and got a specified connection id.
    '''
    #: Port number that was successfully bound.
    port: int
    #: Connection id to be used.
    connection_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Accepted:
        return cls(
            port=int(json['port']),
            connection_id=str(json['connectionId'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\tethering.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Tethering (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def bind(
        port: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Request browser port binding.

    :param port: Port number to bind.
    '''
    params: T_JSON_DICT = dict()
    params['port'] = port
    cmd_dict: T_JSON_DICT = {
        'method': 'Tethering.bind',
        'params': params,
    }
    json = yield cmd_dict


def unbind(
        port: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Request browser port unbinding.

    :param port: Port number to unbind.
    '''
    params: T_JSON_DICT = dict()
    params['port'] = port
    cmd_dict: T_JSON_DICT = {
        'method': 'Tethering.unbind',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Tethering.accepted')
@dataclass
class Accepted:
    '''
    Informs that port was successfully bound and got a specified connection id.
    '''
    #: Port number that was successfully bound.
    port: int
    #: Connection id to be used.
    connection_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Accepted:
        return cls(
            port=int(json['port']),
            connection_id=str(json['connectionId'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\tethering.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Tethering (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

def bind(
        port: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Request browser port binding.

    :param port: Port number to bind.
    '''
    params: T_JSON_DICT = dict()
    params['port'] = port
    cmd_dict: T_JSON_DICT = {
        'method': 'Tethering.bind',
        'params': params,
    }
    json = yield cmd_dict


def unbind(
        port: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Request browser port unbinding.

    :param port: Port number to unbind.
    '''
    params: T_JSON_DICT = dict()
    params['port'] = port
    cmd_dict: T_JSON_DICT = {
        'method': 'Tethering.unbind',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Tethering.accepted')
@dataclass
class Accepted:
    '''
    Informs that port was successfully bound and got a specified connection id.
    '''
    #: Port number that was successfully bound.
    port: int
    #: Connection id to be used.
    connection_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Accepted:
        return cls(
            port=int(json['port']),
            connection_id=str(json['connectionId'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\fedcm\dialog.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import List, Optional

from .account import Account


class Dialog:
    """Represents a FedCM dialog that can be interacted with."""

    DIALOG_TYPE_ACCOUNT_LIST = "AccountChooser"
    DIALOG_TYPE_AUTO_REAUTH = "AutoReauthn"

    def __init__(self, driver) -> None:
        self._driver = driver

    @property
    def type(self) -> Optional[str]:
        """Gets the type of the dialog currently being shown."""
        return self._driver.fedcm.dialog_type

    @property
    def title(self) -> str:
        """Gets the title of the dialog."""
        return self._driver.fedcm.title

    @property
    def subtitle(self) -> Optional[str]:
        """Gets the subtitle of the dialog."""
        result = self._driver.fedcm.subtitle
        return result.get("subtitle") if result else None

    def get_accounts(self) -> List[Account]:
        """Gets the list of accounts shown in the dialog."""
        accounts = self._driver.fedcm.account_list
        return [Account(account) for account in accounts]

    def select_account(self, index: int) -> None:
        """Selects an account from the dialog by index."""
        self._driver.fedcm.select_account(index)

    def accept(self) -> None:
        """Clicks the continue button in the dialog."""
        self._driver.fedcm.accept()

    def dismiss(self) -> None:
        """Cancels/dismisses the dialog."""
        self._driver.fedcm.dismiss()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\firefox\service.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import List, Mapping, Optional

from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import service, utils


class Service(service.Service):
    """A Service class that is responsible for the starting and stopping of
    `geckodriver`.

    :param executable_path: install path of the geckodriver executable, defaults to `geckodriver`.
    :param port: Port for the service to run on, defaults to 0 where the operating system will decide.
    :param service_args: (Optional) List of args to be passed to the subprocess when launching the executable.
    :param log_output: (Optional) int representation of STDOUT/DEVNULL, any IO instance or String path to file.
    :param env: (Optional) Mapping of environment variables for the new process, defaults to `os.environ`.
    :param driver_path_env_key: (Optional) Environment variable to use to get the path to the driver executable.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        port: int = 0,
        service_args: Optional[List[str]] = None,
        log_output: Optional[SubprocessStdAlias] = None,
        env: Optional[Mapping[str, str]] = None,
        driver_path_env_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.service_args = service_args or []
        driver_path_env_key = driver_path_env_key or "SE_GECKODRIVER"

        super().__init__(
            executable_path=executable_path,
            port=port,
            log_output=log_output,
            env=env,
            driver_path_env_key=driver_path_env_key,
            **kwargs,
        )

        # Set a port for CDP
        if "--connect-existing" not in self.service_args:
            self.service_args.append("--websocket-port")
            self.service_args.append(f"{utils.free_port()}")

    def command_line_args(self) -> List[str]:
        return ["--port", f"{self.port}"] + self.service_args

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\webkitgtk\service.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import shutil
import warnings
from typing import List, Mapping, Optional

from selenium.webdriver.common import service

DEFAULT_EXECUTABLE_PATH: str = shutil.which("WebKitWebDriver")


class Service(service.Service):
    """A Service class that is responsible for the starting and stopping of
    `WebKitWebDriver`.

    :param executable_path: install path of the WebKitWebDriver executable, defaults to the first
        `WebKitWebDriver` in `$PATH`.
    :param port: Port for the service to run on, defaults to 0 where the operating system will decide.
    :param service_args: (Optional) List of args to be passed to the subprocess when launching the executable.
    :param log_output: (Optional) File path for the file to be opened and passed as the subprocess
        stdout/stderr handler.
    :param env: (Optional) Mapping of environment variables for the new process, defaults to `os.environ`.
    """

    def __init__(
        self,
        executable_path: str = DEFAULT_EXECUTABLE_PATH,
        port: int = 0,
        log_path: Optional[str] = None,
        log_output: Optional[str] = None,
        service_args: Optional[List[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        **kwargs,
    ) -> None:
        self.service_args = service_args or []
        if log_path is not None:
            warnings.warn("log_path is deprecated, use log_output instead", DeprecationWarning, stacklevel=2)
            log_path = open(log_path, "wb")
        log_output = open(log_output, "wb") if log_output else None
        super().__init__(
            executable_path=executable_path,
            port=port,
            log_output=log_path or log_output,
            env=env,
            **kwargs,
        )

    def command_line_args(self) -> List[str]:
        return ["-p", f"{self.port}"] + self.service_args

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mssql\aioodbc.py ===
# dialects/mssql/aioodbc.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
r"""
.. dialect:: mssql+aioodbc
    :name: aioodbc
    :dbapi: aioodbc
    :connectstring: mssql+aioodbc://<username>:<password>@<dsnname>
    :url: https://pypi.org/project/aioodbc/


Support for the SQL Server database in asyncio style, using the aioodbc
driver which itself is a thread-wrapper around pyodbc.

.. versionadded:: 2.0.23  Added the mssql+aioodbc dialect which builds
   on top of the pyodbc and general aio* dialect architecture.

Using a special asyncio mediation layer, the aioodbc dialect is usable
as the backend for the :ref:`SQLAlchemy asyncio <asyncio_toplevel>`
extension package.

Most behaviors and caveats for this driver are the same as that of the
pyodbc dialect used on SQL Server; see :ref:`mssql_pyodbc` for general
background.

This dialect should normally be used only with the
:func:`_asyncio.create_async_engine` engine creation function; connection
styles are otherwise equivalent to those documented in the pyodbc section::

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        "mssql+aioodbc://scott:tiger@mssql2017:1433/test?"
        "driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )

"""

from __future__ import annotations

from .pyodbc import MSDialect_pyodbc
from .pyodbc import MSExecutionContext_pyodbc
from ...connectors.aioodbc import aiodbcConnector


class MSExecutionContext_aioodbc(MSExecutionContext_pyodbc):
    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(server_side=True)


class MSDialectAsync_aioodbc(aiodbcConnector, MSDialect_pyodbc):
    driver = "aioodbc"

    supports_statement_cache = True

    execution_ctx_cls = MSExecutionContext_aioodbc


dialect = MSDialectAsync_aioodbc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\singularityfunctions.py ===
from sympy.functions import SingularityFunction, DiracDelta
from sympy.integrals import integrate


def singularityintegrate(f, x):
    """
    This function handles the indefinite integrations of Singularity functions.
    The ``integrate`` function calls this function internally whenever an
    instance of SingularityFunction is passed as argument.

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a SingularityFunction expression,
      i.e. ``SingularityFunction(x, a, n)``, we just return
      ``SingularityFunction(x, a, n + 1)/(n + 1)`` if ``n >= 0`` and
      ``SingularityFunction(x, a, n + 1)`` if ``n < 0``.

    - If the node is a multiplication or power node having a
      SingularityFunction term we rewrite the whole expression in terms of
      Heaviside and DiracDelta and then integrate the output. Lastly, we
      rewrite the output of integration back in terms of SingularityFunction.

    - If none of the above case arises, we return None.

    Examples
    ========

    >>> from sympy.integrals.singularityfunctions import singularityintegrate
    >>> from sympy import SingularityFunction, symbols, Function
    >>> x, a, n, y = symbols('x a n y')
    >>> f = Function('f')
    >>> singularityintegrate(SingularityFunction(x, a, 3), x)
    SingularityFunction(x, a, 4)/4
    >>> singularityintegrate(5*SingularityFunction(x, 5, -2), x)
    5*SingularityFunction(x, 5, -1)
    >>> singularityintegrate(6*SingularityFunction(x, 5, -1), x)
    6*SingularityFunction(x, 5, 0)
    >>> singularityintegrate(x*SingularityFunction(x, 0, -1), x)
    0
    >>> singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x)
    f(1)*SingularityFunction(x, 1, 0)

    """

    if not f.has(SingularityFunction):
        return None

    if isinstance(f, SingularityFunction):
        x, a, n = f.args
        if n.is_positive or n.is_zero:
            return SingularityFunction(x, a, n + 1)/(n + 1)
        elif n in (-1, -2, -3, -4):
            return SingularityFunction(x, a, n + 1)

    if f.is_Mul or f.is_Pow:

        expr = f.rewrite(DiracDelta)
        expr = integrate(expr, x)
        return expr.rewrite(SingularityFunction)
    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\series.py ===
from sympy.core.sympify import sympify


def series(expr, x=None, x0=0, n=6, dir="+"):
    """Series expansion of expr around point `x = x0`.

    Parameters
    ==========

    expr : Expression
           The expression whose series is to be expanded.

    x : Symbol
        It is the variable of the expression to be calculated.

    x0 : Value
         The value around which ``x`` is calculated. Can be any value
         from ``-oo`` to ``oo``.

    n : Value
        The number of terms upto which the series is to be expanded.

    dir : String, optional
          The series-expansion can be bi-directional. If ``dir="+"``,
          then (x->x0+). If ``dir="-"``, then (x->x0-). For infinite
          ``x0`` (``oo`` or ``-oo``), the ``dir`` argument is determined
          from the direction of the infinity (i.e., ``dir="-"`` for
          ``oo``).

    Examples
    ========

    >>> from sympy import series, tan, oo
    >>> from sympy.abc import x
    >>> f = tan(x)
    >>> series(f, x, 2, 6, "+")
    tan(2) + (1 + tan(2)**2)*(x - 2) + (x - 2)**2*(tan(2)**3 + tan(2)) +
    (x - 2)**3*(1/3 + 4*tan(2)**2/3 + tan(2)**4) + (x - 2)**4*(tan(2)**5 +
    5*tan(2)**3/3 + 2*tan(2)/3) + (x - 2)**5*(2/15 + 17*tan(2)**2/15 +
    2*tan(2)**4 + tan(2)**6) + O((x - 2)**6, (x, 2))

    >>> series(f, x, 2, 3, "-")
    tan(2) + (2 - x)*(-tan(2)**2 - 1) + (2 - x)**2*(tan(2)**3 + tan(2))
    + O((x - 2)**3, (x, 2))

    >>> series(f, x, 2, oo, "+")
    Traceback (most recent call last):
    ...
    TypeError: 'Infinity' object cannot be interpreted as an integer

    Returns
    =======

    Expr
        Series expansion of the expression about x0

    See Also
    ========

    sympy.core.expr.Expr.series: See the docstring of Expr.series() for complete details of this wrapper.
    """
    expr = sympify(expr)
    return expr.series(x, x0, n, dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\contains.py ===
from sympy.core import S
from sympy.core.sympify import sympify
from sympy.core.relational import Eq, Ne
from sympy.core.parameters import global_parameters
from sympy.logic.boolalg import Boolean
from sympy.utilities.misc import func_name
from .sets import Set


class Contains(Boolean):
    """
    Asserts that x is an element of the set S.

    Examples
    ========

    >>> from sympy import Symbol, Integer, S, Contains
    >>> Contains(Integer(2), S.Integers)
    True
    >>> Contains(Integer(-2), S.Naturals)
    False
    >>> i = Symbol('i', integer=True)
    >>> Contains(i, S.Naturals)
    Contains(i, Naturals)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Element_%28mathematics%29
    """
    def __new__(cls, x, s, evaluate=None):
        x = sympify(x)
        s = sympify(s)

        if evaluate is None:
            evaluate = global_parameters.evaluate

        if not isinstance(s, Set):
            raise TypeError('expecting Set, not %s' % func_name(s))

        if evaluate:
            # _contains can return symbolic booleans that would be returned by
            # s.contains(x) but here for Contains(x, s) we only evaluate to
            # true, false or return the unevaluated Contains.
            result = s._contains(x)

            if isinstance(result, Boolean):
                if result in (S.true, S.false):
                    return result
            elif result is not None:
                raise TypeError("_contains() should return Boolean or None")

        return super().__new__(cls, x, s)

    @property
    def binary_symbols(self):
        return set().union(*[i.binary_symbols
            for i in self.args[1].args
            if i.is_Boolean or i.is_Symbol or
            isinstance(i, (Eq, Ne))])

    def as_set(self):
        return self.args[1]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\utils.py ===
import datetime
import os
import re
import subprocess


def get_date_diff(date1, date2, date_format):
    a = datetime.datetime.strptime(date1, date_format)
    b = datetime.datetime.strptime(
        str(date2.strftime(date_format)), date_format)

    return (b - a).days


def linux_browser_apps_to_cmd(*apps: str) -> str:
    """Create 'browser --version' command from browser app names.

    Result command example:
        chromium --version || chromium-browser --version
    """
    ignore_errors_cmd_part = " 2>/dev/null" if os.getenv(
        "WDM_LOG_LEVEL") == "0" else ""
    return " || ".join(f"{i} --version{ignore_errors_cmd_part}" for i in apps)


def windows_browser_apps_to_cmd(*apps: str) -> str:
    """Create analogue of browser --version command for windows."""
    powershell = determine_powershell()

    first_hit_template = """$tmp = {expression}; if ($tmp) {{echo $tmp; Exit;}};"""
    script = "$ErrorActionPreference='silentlycontinue'; " + " ".join(
        first_hit_template.format(expression=e) for e in apps
    )

    return f'{powershell} -NoProfile "{script}"'


def read_version_from_cmd(cmd, pattern):
    with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
    ) as stream:
        stdout = stream.communicate()[0].decode()
        version = re.search(pattern, stdout)
        version = version.group(0) if version else None
    return version


def determine_powershell():
    """Returns "True" if runs in Powershell and "False" if another console."""
    cmd = "(dir 2>&1 *`|echo CMD);&<# rem #>echo powershell"
    with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            shell=True,
    ) as stream:
        stdout = stream.communicate()[0].decode()
    return "" if stdout == "powershell" else "powershell"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_indexing.py ===
from collections import namedtuple
from datetime import (
    datetime,
    timedelta,
)
from decimal import Decimal
import re

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas._libs import iNaT
from pandas.errors import (
    InvalidIndexError,
    PerformanceWarning,
    SettingWithCopyError,
)
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    to_datetime,
)
import pandas._testing as tm

# We pass through a TypeError raised by numpy
_slice_msg = "slice indices must be integers or None or have an __index__ method"


class TestDataFrameIndexing:
    def test_getitem(self, float_frame):
        # Slicing
        sl = float_frame[:20]
        assert len(sl.index) == 20

        # Column access
        for _, series in sl.items():
            assert len(series.index) == 20
            tm.assert_index_equal(series.index, sl.index)

        for key, _ in float_frame._series.items():
            assert float_frame[key] is not None

        assert "random" not in float_frame
        with pytest.raises(KeyError, match="random"):
            float_frame["random"]

    def test_getitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        # GH51053
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        result = df[1]
        expected = DataFrame([[1, 3], [4, 6]], columns=Index([1, 1], dtype=dtype))
        tm.assert_frame_equal(result, expected, check_exact=True)

    def test_getitem2(self, float_frame):
        df = float_frame.copy()
        df["$10"] = np.random.default_rng(2).standard_normal(len(df))

        ad = np.random.default_rng(2).standard_normal(len(df))
        df["@awesome_domain"] = ad

        with pytest.raises(KeyError, match=re.escape("'df[\"$10\"]'")):
            df.__getitem__('df["$10"]')

        res = df["@awesome_domain"]
        tm.assert_numpy_array_equal(ad, res.values)

    def test_setitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        # GH51053
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        df[1] = 10
        expected = DataFrame([[10, 2, 10], [10, 5, 10]], columns=idx)
        tm.assert_frame_equal(df, expected, check_exact=True)

    def test_setitem_list(self, float_frame):
        float_frame["E"] = "foo"
        data = float_frame[["A", "B"]]
        float_frame[["B", "A"]] = data

        tm.assert_series_equal(float_frame["B"], data["A"], check_names=False)
        tm.assert_series_equal(float_frame["A"], data["B"], check_names=False)

        msg = "Columns must be same length as key"
        with pytest.raises(ValueError, match=msg):
            data[["A"]] = float_frame[["A", "B"]]
        newcolumndata = range(len(data.index) - 1)
        msg = (
            rf"Length of values \({len(newcolumndata)}\) "
            rf"does not match length of index \({len(data)}\)"
        )
        with pytest.raises(ValueError, match=msg):
            data["A"] = newcolumndata

    def test_setitem_list2(self):
        df = DataFrame(0, index=range(3), columns=["tt1", "tt2"], dtype=int)
        df.loc[1, ["tt1", "tt2"]] = [1, 2]

        result = df.loc[df.index[1], ["tt1", "tt2"]]
        expected = Series([1, 2], df.columns, dtype=int, name=1)
        tm.assert_series_equal(result, expected)

        df["tt1"] = df["tt2"] = "0"
        df.loc[df.index[1], ["tt1", "tt2"]] = ["1", "2"]
        result = df.loc[df.index[1], ["tt1", "tt2"]]
        expected = Series(["1", "2"], df.columns, name=1)
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean(self, mixed_float_frame, mixed_int_frame, datetime_frame):
        # boolean indexing
        d = datetime_frame.index[10]
        indexer = datetime_frame.index > d
        indexer_obj = indexer.astype(object)

        subindex = datetime_frame.index[indexer]
        subframe = datetime_frame[indexer]

        tm.assert_index_equal(subindex, subframe.index)
        with pytest.raises(ValueError, match="Item wrong length"):
            datetime_frame[indexer[:-1]]

        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        with pytest.raises(ValueError, match="Boolean array expected"):
            datetime_frame[datetime_frame]

        # test that Series work
        indexer_obj = Series(indexer_obj, datetime_frame.index)

        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        # test that Series indexers reindex
        # we are producing a warning that since the passed boolean
        # key is not the same as the given index, we will reindex
        # not sure this is really necessary
        with tm.assert_produces_warning(UserWarning):
            indexer_obj = indexer_obj.reindex(datetime_frame.index[::-1])
            subframe_obj = datetime_frame[indexer_obj]
            tm.assert_frame_equal(subframe_obj, subframe)

        # test df[df > 0]
        for df in [
            datetime_frame,
            mixed_float_frame,
            mixed_int_frame,
        ]:
            data = df._get_numeric_data()
            bif = df[df > 0]
            bifw = DataFrame(
                {c: np.where(data[c] > 0, data[c], np.nan) for c in data.columns},
                index=data.index,
                columns=data.columns,
            )

            # add back other columns to compare
            for c in df.columns:
                if c not in bifw:
                    bifw[c] = df[c]
            bifw = bifw.reindex(columns=df.columns)

            tm.assert_frame_equal(bif, bifw, check_dtype=False)
            for c in df.columns:
                if bif[c].dtype != bifw[c].dtype:
                    assert bif[c].dtype == df[c].dtype

    def test_getitem_boolean_casting(self, datetime_frame):
        # don't upcast if we don't need to
        df = datetime_frame.copy()
        df["E"] = 1
        df["E"] = df["E"].astype("int32")
        df["E1"] = df["E"].copy()
        df["F"] = 1
        df["F"] = df["F"].astype("int64")
        df["F1"] = df["F"].copy()

        casted = df[df > 0]
        result = casted.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")] * 2
            + [np.dtype("int64")] * 2,
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        tm.assert_series_equal(result, expected)

        # int block splitting
        df.loc[df.index[1:3], ["E1", "F1"]] = 0
        casted = df[df > 0]
        result = casted.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")]
            + [np.dtype("float64")]
            + [np.dtype("int64")]
            + [np.dtype("float64")],
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "lst", [[True, False, True], [True, True, True], [False, False, False]]
    )
    def test_getitem_boolean_list(self, lst):
        df = DataFrame(np.arange(12).reshape(3, 4))
        result = df[lst]
        expected = df.loc[df.index[lst]]
        tm.assert_frame_equal(result, expected)

    def test_getitem_boolean_iadd(self):
        arr = np.random.default_rng(2).standard_normal((5, 5))

        df = DataFrame(arr.copy(), columns=["A", "B", "C", "D", "E"])

        df[df < 0] += 1
        arr[arr < 0] += 1

        tm.assert_almost_equal(df.values, arr)

    def test_boolean_index_empty_corner(self):
        # #2096
        blah = DataFrame(np.empty([0, 1]), columns=["A"], index=DatetimeIndex([]))

        # both of these should succeed trivially
        k = np.array([], bool)

        blah[k]
        blah[k] = 0

    def test_getitem_ix_mixed_integer(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)),
            index=[1, 10, "C", "E"],
            columns=[1, 2, 3],
        )

        result = df.iloc[:-1]
        expected = df.loc[df.index[:-1]]
        tm.assert_frame_equal(result, expected)

        result = df.loc[[1, 10]]
        expected = df.loc[Index([1, 10])]
        tm.assert_frame_equal(result, expected)

    def test_getitem_ix_mixed_integer2(self):
        # 11320
        df = DataFrame(
            {
                "rna": (1.5, 2.2, 3.2, 4.5),
                -1000: [11, 21, 36, 40],
                0: [10, 22, 43, 34],
                1000: [0, 10, 20, 30],
            },
            columns=["rna", -1000, 0, 1000],
        )
        result = df[[1000]]
        expected = df.iloc[:, [3]]
        tm.assert_frame_equal(result, expected)
        result = df[[-1000]]
        expected = df.iloc[:, [1]]
        tm.assert_frame_equal(result, expected)

    def test_getattr(self, float_frame):
        tm.assert_series_equal(float_frame.A, float_frame["A"])
        msg = "'DataFrame' object has no attribute 'NONEXISTENT_NAME'"
        with pytest.raises(AttributeError, match=msg):
            float_frame.NONEXISTENT_NAME

    def test_setattr_column(self):
        df = DataFrame({"foobar": 1}, index=range(10))

        df.foobar = 5
        assert (df.foobar == 5).all()

    def test_setitem(
        self, float_frame, using_copy_on_write, warn_copy_on_write, using_infer_string
    ):
        # not sure what else to do here
        series = float_frame["A"][::2]
        float_frame["col5"] = series
        assert "col5" in float_frame

        assert len(series) == 15
        assert len(float_frame) == 30

        exp = np.ravel(np.column_stack((series.values, [np.nan] * 15)))
        exp = Series(exp, index=float_frame.index, name="col5")
        tm.assert_series_equal(float_frame["col5"], exp)

        series = float_frame["A"]
        float_frame["col6"] = series
        tm.assert_series_equal(series, float_frame["col6"], check_names=False)

        # set ndarray
        arr = np.random.default_rng(2).standard_normal(len(float_frame))
        float_frame["col9"] = arr
        assert (float_frame["col9"] == arr).all()

        float_frame["col7"] = 5
        assert (float_frame["col7"] == 5).all()

        float_frame["col0"] = 3.14
        assert (float_frame["col0"] == 3.14).all()

        float_frame["col8"] = "foo"
        assert (float_frame["col8"] == "foo").all()

        # this is partially a view (e.g. some blocks are view)
        # so raise/warn
        smaller = float_frame[:2]

        msg = r"\nA value is trying to be set on a copy of a slice from a DataFrame"
        if using_copy_on_write or warn_copy_on_write:
            # With CoW, adding a new column doesn't raise a warning
            smaller["col10"] = ["1", "2"]
        else:
            with pytest.raises(SettingWithCopyError, match=msg):
                smaller["col10"] = ["1", "2"]

        if using_infer_string:
            assert smaller["col10"].dtype == "str"
        else:
            assert smaller["col10"].dtype == np.object_
        assert (smaller["col10"] == ["1", "2"]).all()

    def test_setitem2(self):
        # dtype changing GH4204
        df = DataFrame([[0, 0]])
        df.iloc[0] = np.nan
        expected = DataFrame([[np.nan, np.nan]])
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[0, 0]])
        df.loc[0] = np.nan
        tm.assert_frame_equal(df, expected)

    def test_setitem_boolean(self, float_frame):
        df = float_frame.copy()
        values = float_frame.values.copy()

        df[df["A"] > 0] = 4
        values[values[:, 0] > 0] = 4
        tm.assert_almost_equal(df.values, values)

        # test that column reindexing works
        series = df["A"] == 4
        series = series.reindex(df.index[::-1])
        df[series] = 1
        values[values[:, 0] == 4] = 1
        tm.assert_almost_equal(df.values, values)

        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)

        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)

        # a df that needs alignment first
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)

        # indexed with same shape but rows-reversed df
        df[df[::-1] == 2] = 3
        values[values == 2] = 3
        tm.assert_almost_equal(df.values, values)

        msg = "Must pass DataFrame or 2-d ndarray with boolean values only"
        with pytest.raises(TypeError, match=msg):
            df[df * 0] = 2

        # index with DataFrame
        df_orig = df.copy()
        mask = df > np.abs(df)
        df[df > np.abs(df)] = np.nan
        values = df_orig.values.copy()
        values[mask.values] = np.nan
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        tm.assert_frame_equal(df, expected)

        # set from DataFrame
        df[df > np.abs(df)] = df * 2
        np.putmask(values, mask.values, df.values * 2)
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        tm.assert_frame_equal(df, expected)

    def test_setitem_cast(self, float_frame):
        float_frame["D"] = float_frame["D"].astype("i8")
        assert float_frame["D"].dtype == np.int64

        # #669, should not cast?
        # this is now set to int64, which means a replacement of the column to
        # the value dtype (and nothing to do with the existing dtype)
        float_frame["B"] = 0
        assert float_frame["B"].dtype == np.int64

        # cast if pass array of course
        float_frame["B"] = np.arange(len(float_frame))
        assert issubclass(float_frame["B"].dtype.type, np.integer)

        float_frame["foo"] = "bar"
        float_frame["foo"] = 0
        assert float_frame["foo"].dtype == np.int64

        float_frame["foo"] = "bar"
        float_frame["foo"] = 2.5
        assert float_frame["foo"].dtype == np.float64

        float_frame["something"] = 0
        assert float_frame["something"].dtype == np.int64
        float_frame["something"] = 2
        assert float_frame["something"].dtype == np.int64
        float_frame["something"] = 2.5
        assert float_frame["something"].dtype == np.float64

    def test_setitem_corner(self, float_frame, using_infer_string):
        # corner case
        df = DataFrame({"B": [1.0, 2.0, 3.0], "C": ["a", "b", "c"]}, index=np.arange(3))
        del df["B"]
        df["B"] = [1.0, 2.0, 3.0]
        assert "B" in df
        assert len(df.columns) == 2

        df["A"] = "beginning"
        df["E"] = "foo"
        df["D"] = "bar"
        df[datetime.now()] = "date"
        df[datetime.now()] = 5.0

        # what to do when empty frame with index
        dm = DataFrame(index=float_frame.index)
        dm["A"] = "foo"
        dm["B"] = "bar"
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.object_

        # upcast
        dm["C"] = 1
        assert dm["C"].dtype == np.int64

        dm["E"] = 1.0
        assert dm["E"].dtype == np.float64

        # set existing column
        dm["A"] = "bar"
        assert "bar" == dm["A"].iloc[0]

        dm = DataFrame(index=np.arange(3))
        dm["A"] = 1
        dm["foo"] = "bar"
        del dm["foo"]
        dm["foo"] = "bar"
        if using_infer_string:
            assert dm["foo"].dtype == "str"
        else:
            assert dm["foo"].dtype == np.object_

        dm["coercible"] = ["1", "2", "3"]
        if using_infer_string:
            assert dm["coercible"].dtype == "str"
        else:
            assert dm["coercible"].dtype == np.object_

    def test_setitem_corner2(self):
        data = {
            "title": ["foobar", "bar", "foobar"] + ["foobar"] * 17,
            "cruft": np.random.default_rng(2).random(20),
        }

        df = DataFrame(data)
        ix = df[df["title"] == "bar"].index

        df.loc[ix, ["title"]] = "foobar"
        df.loc[ix, ["cruft"]] = 0

        assert df.loc[1, "title"] == "foobar"
        assert df.loc[1, "cruft"] == 0

    def test_setitem_ambig(self, using_infer_string):
        # Difficulties with mixed-type data
        # Created as float type
        dm = DataFrame(index=range(3), columns=range(3))

        coercable_series = Series([Decimal(1) for _ in range(3)], index=range(3))
        uncoercable_series = Series(["foo", "bzr", "baz"], index=range(3))

        dm[0] = np.ones(3)
        assert len(dm.columns) == 3

        dm[1] = coercable_series
        assert len(dm.columns) == 3

        dm[2] = uncoercable_series
        assert len(dm.columns) == 3
        if using_infer_string:
            assert dm[2].dtype == "str"
        else:
            assert dm[2].dtype == np.object_

    def test_setitem_None(self, float_frame):
        # GH #766
        float_frame[None] = float_frame["A"]
        tm.assert_series_equal(
            float_frame.iloc[:, -1], float_frame["A"], check_names=False
        )
        tm.assert_series_equal(
            float_frame.loc[:, None], float_frame["A"], check_names=False
        )
        tm.assert_series_equal(float_frame[None], float_frame["A"], check_names=False)

    def test_loc_setitem_boolean_mask_allfalse(self):
        # GH 9596
        df = DataFrame(
            {"a": ["1", "2", "3"], "b": ["11", "22", "33"], "c": ["111", "222", "333"]}
        )

        result = df.copy()
        result.loc[result.b.isna(), "a"] = result.a.copy()
        tm.assert_frame_equal(result, df)

    def test_getitem_fancy_slice_integers_step(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))

        # this is OK
        df.iloc[:8:2]
        df.iloc[:8:2] = np.nan
        assert isna(df.iloc[:8:2]).values.all()

    def test_getitem_setitem_integer_slice_keyerrors(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), index=range(0, 20, 2)
        )

        # this is OK
        cp = df.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).values.all()

        # so is this
        cp = df.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()

        result = df.iloc[2:6]
        result2 = df.loc[3:11]
        expected = df.reindex([4, 6, 8, 10])

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # non-monotonic, raise KeyError
        df2 = df.iloc[list(range(5)) + list(range(5, 10))[::-1]]
        with pytest.raises(KeyError, match=r"^3$"):
            df2.loc[3:11]
        with pytest.raises(KeyError, match=r"^3$"):
            df2.loc[3:11] = 0

    @td.skip_array_manager_invalid_test  # already covered in test_iloc_col_slice_view
    def test_fancy_getitem_slice_mixed(
        self, float_frame, float_string_frame, using_copy_on_write, warn_copy_on_write
    ):
        sliced = float_string_frame.iloc[:, -3:]
        assert sliced["D"].dtype == np.float64

        # get view with single block
        # setting it triggers setting with copy
        original = float_frame.copy()
        sliced = float_frame.iloc[:, -3:]

        assert np.shares_memory(sliced["C"]._values, float_frame["C"]._values)

        with tm.assert_cow_warning(warn_copy_on_write):
            sliced.loc[:, "C"] = 4.0
        if not using_copy_on_write:
            assert (float_frame["C"] == 4).all()

            # with the enforcement of GH#45333 in 2.0, this remains a view
            np.shares_memory(sliced["C"]._values, float_frame["C"]._values)
        else:
            tm.assert_frame_equal(float_frame, original)

    def test_getitem_setitem_non_ix_labels(self):
        df = DataFrame(range(20), index=date_range("2020-01-01", periods=20))

        start, end = df.index[[5, 10]]

        result = df.loc[start:end]
        result2 = df[start:end]
        expected = df[5:11]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        result = df.copy()
        result.loc[start:end] = 0
        result2 = df.copy()
        result2[start:end] = 0
        expected = df.copy()
        expected[5:11] = 0
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_ix_multi_take(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        rs = df.loc[df.index == 0, :]
        xp = df.reindex([0])
        tm.assert_frame_equal(rs, xp)

        # GH#1321
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        rs = df.loc[df.index == 0, df.columns == 1]
        xp = df.reindex(index=[0], columns=[1])
        tm.assert_frame_equal(rs, xp)

    def test_getitem_fancy_scalar(self, float_frame):
        f = float_frame
        ix = f.loc

        # individual value
        for col in f.columns:
            ts = f[col]
            for idx in f.index[::5]:
                assert ix[idx, col] == ts[idx]

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite not using .values
    def test_setitem_fancy_scalar(self, float_frame):
        f = float_frame
        expected = float_frame.copy()
        ix = f.loc

        # individual value
        for j, col in enumerate(f.columns):
            f[col]
            for idx in f.index[::5]:
                i = f.index.get_loc(idx)
                val = np.random.default_rng(2).standard_normal()
                expected.iloc[i, j] = val

                ix[idx, col] = val
                tm.assert_frame_equal(f, expected)

    def test_getitem_fancy_boolean(self, float_frame):
        f = float_frame
        ix = f.loc

        expected = f.reindex(columns=["B", "D"])
        result = ix[:, [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        expected = f.reindex(index=f.index[5:10], columns=["B", "D"])
        result = ix[f.index[5:10], [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        boolvec = f.index > f.index[7]
        expected = f.reindex(index=f.index[boolvec])
        result = ix[boolvec]
        tm.assert_frame_equal(result, expected)
        result = ix[boolvec, :]
        tm.assert_frame_equal(result, expected)

        result = ix[boolvec, f.columns[2:]]
        expected = f.reindex(index=f.index[boolvec], columns=["C", "D"])
        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite not using .values
    def test_setitem_fancy_boolean(self, float_frame):
        # from 2d, set with booleans
        frame = float_frame.copy()
        expected = float_frame.copy()
        values = expected.values.copy()

        mask = frame["A"] > 0
        frame.loc[mask] = 0.0
        values[mask.values] = 0.0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(frame, expected)

        frame = float_frame.copy()
        expected = float_frame.copy()
        values = expected.values.copy()
        frame.loc[mask, ["A", "B"]] = 0.0
        values[mask.values, :2] = 0.0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(frame, expected)

    def test_getitem_fancy_ints(self, float_frame):
        result = float_frame.iloc[[1, 4, 7]]
        expected = float_frame.loc[float_frame.index[[1, 4, 7]]]
        tm.assert_frame_equal(result, expected)

        result = float_frame.iloc[:, [2, 0, 1]]
        expected = float_frame.loc[:, float_frame.columns[[2, 0, 1]]]
        tm.assert_frame_equal(result, expected)

    def test_getitem_setitem_boolean_misaligned(self, float_frame):
        # boolean index misaligned labels
        mask = float_frame["A"][::-1] > 1

        result = float_frame.loc[mask]
        expected = float_frame.loc[mask[::-1]]
        tm.assert_frame_equal(result, expected)

        cp = float_frame.copy()
        expected = float_frame.copy()
        cp.loc[mask] = 0
        expected.loc[mask] = 0
        tm.assert_frame_equal(cp, expected)

    def test_getitem_setitem_boolean_multi(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))

        # get
        k1 = np.array([True, False, True])
        k2 = np.array([False, True])
        result = df.loc[k1, k2]
        expected = df.loc[[0, 2], [1]]
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        df.loc[np.array([True, False, True]), np.array([False, True])] = 5
        expected.loc[[0, 2], [1]] = 5
        tm.assert_frame_equal(df, expected)

    def test_getitem_setitem_float_labels(self, using_array_manager):
        index = Index([1.5, 2, 3, 4, 5])
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)

        result = df.loc[1.5:4]
        expected = df.reindex([1.5, 2, 3, 4])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4

        result = df.loc[4:5]
        expected = df.reindex([4, 5])  # reindex with int
        tm.assert_frame_equal(result, expected, check_index_type=False)
        assert len(result) == 2

        result = df.loc[4:5]
        expected = df.reindex([4.0, 5.0])  # reindex with float
        tm.assert_frame_equal(result, expected)
        assert len(result) == 2

        # loc_float changes this to work properly
        result = df.loc[1:2]
        expected = df.iloc[0:2]
        tm.assert_frame_equal(result, expected)

        expected = df.iloc[0:2]
        msg = r"The behavior of obj\[i:j\] with a float-dtype index"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df[1:2]
        tm.assert_frame_equal(result, expected)

        # #2727
        index = Index([1.0, 2.5, 3.5, 4.5, 5.0])
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)

        # positional slicing only via iloc!
        msg = (
            "cannot do positional indexing on Index with "
            r"these indexers \[1.0\] of type float"
        )
        with pytest.raises(TypeError, match=msg):
            df.iloc[1.0:5]

        result = df.iloc[4:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1

        cp = df.copy()

        with pytest.raises(TypeError, match=_slice_msg):
            cp.iloc[1.0:5] = 0

        with pytest.raises(TypeError, match=msg):
            result = cp.iloc[1.0:5] == 0

        assert result.values.all()
        assert (cp.iloc[0:1] == df.iloc[0:1]).values.all()

        cp = df.copy()
        cp.iloc[4:5] = 0
        assert (cp.iloc[4:5] == 0).values.all()
        assert (cp.iloc[0:4] == df.iloc[0:4]).values.all()

        # float slicing
        result = df.loc[1.0:5]
        expected = df
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5

        result = df.loc[1.1:5]
        expected = df.reindex([2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4

        result = df.loc[4.51:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1

        result = df.loc[1.0:5.0]
        expected = df.reindex([1.0, 2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5

        cp = df.copy()
        cp.loc[1.0:5.0] = 0
        result = cp.loc[1.0:5.0]
        assert (result == 0).values.all()

    def test_setitem_single_column_mixed_datetime(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )

        df["timestamp"] = Timestamp("20010102")

        # check our dtypes
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")] * 3 + [np.dtype("datetime64[s]")],
            index=["foo", "bar", "baz", "timestamp"],
        )
        tm.assert_series_equal(result, expected)

        # GH#16674 iNaT is treated as an integer when given by the user
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            df.loc["b", "timestamp"] = iNaT
        assert not isna(df.loc["b", "timestamp"])
        assert df["timestamp"].dtype == np.object_
        assert df.loc["b", "timestamp"] == iNaT

        # allow this syntax (as of GH#3216)
        df.loc["c", "timestamp"] = np.nan
        assert isna(df.loc["c", "timestamp"])

        # allow this syntax
        df.loc["d", :] = np.nan
        assert not isna(df.loc["c", :]).all()

    def test_setitem_mixed_datetime(self):
        # GH 9336
        expected = DataFrame(
            {
                "a": [0, 0, 0, 0, 13, 14],
                "b": [
                    datetime(2012, 1, 1),
                    1,
                    "x",
                    "y",
                    datetime(2013, 1, 1),
                    datetime(2014, 1, 1),
                ],
            }
        )
        df = DataFrame(0, columns=list("ab"), index=range(6))
        df["b"] = pd.NaT
        df.loc[0, "b"] = datetime(2012, 1, 1)
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            df.loc[1, "b"] = 1
        df.loc[[2, 3], "b"] = "x", "y"
        A = np.array(
            [
                [13, np.datetime64("2013-01-01T00:00:00")],
                [14, np.datetime64("2014-01-01T00:00:00")],
            ]
        )
        df.loc[[4, 5], ["a", "b"]] = A
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_float(self, float_frame):
        piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]
        float_frame.loc[float_frame.index[-2] :, ["A", "B"]] = piece.values
        result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    # dtype inference
    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_setitem_frame_mixed(self, float_string_frame):
        # GH 3216

        # already aligned
        f = float_string_frame.copy()
        piece = DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], index=f.index[0:2], columns=["A", "B"]
        )
        key = (f.index[slice(None, 2)], ["A", "B"])
        f.loc[key] = piece
        tm.assert_almost_equal(f.loc[f.index[0:2], ["A", "B"]].values, piece.values)

    # dtype inference
    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_setitem_frame_mixed_rows_unaligned(self, float_string_frame):
        # GH#3216 rows unaligned
        f = float_string_frame.copy()
        piece = DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            index=list(f.index[0:2]) + ["foo", "bar"],
            columns=["A", "B"],
        )
        key = (f.index[slice(None, 2)], ["A", "B"])
        f.loc[key] = piece
        tm.assert_almost_equal(
            f.loc[f.index[0:2:], ["A", "B"]].values, piece.values[0:2]
        )

    # dtype inference
    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_setitem_frame_mixed_key_unaligned(self, float_string_frame):
        # GH#3216 key is unaligned with values
        f = float_string_frame.copy()
        piece = f.loc[f.index[:2], ["A"]]
        piece.index = f.index[-2:]
        key = (f.index[slice(-2, None)], ["A", "B"])
        f.loc[key] = piece
        piece["B"] = np.nan
        tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)

    def test_setitem_frame_mixed_ndarray(self, float_string_frame):
        # GH#3216 ndarray
        f = float_string_frame.copy()
        piece = float_string_frame.loc[f.index[:2], ["A", "B"]]
        key = (f.index[slice(-2, None)], ["A", "B"])
        f.loc[key] = piece.values
        tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)

    def test_setitem_frame_upcast(self):
        # needs upcasting
        df = DataFrame([[1, 2, "foo"], [3, 4, "bar"]], columns=["A", "B", "C"])
        df2 = df.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df2.loc[:, ["A", "B"]] = df.loc[:, ["A", "B"]] + 0.5
        expected = df.reindex(columns=["A", "B"])
        expected += 0.5
        expected["C"] = df["C"]
        tm.assert_frame_equal(df2, expected)

    def test_setitem_frame_align(self, float_frame):
        piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]
        piece.index = float_frame.index[-2:]
        piece.columns = ["A", "B"]
        float_frame.loc[float_frame.index[-2:], ["A", "B"]] = piece
        result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    def test_getitem_setitem_ix_duplicates(self):
        # #1201
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["foo", "foo", "bar", "baz", "bar"],
        )

        result = df.loc["foo"]
        expected = df[:2]
        tm.assert_frame_equal(result, expected)

        result = df.loc["bar"]
        expected = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, expected)

        result = df.loc["baz"]
        expected = df.iloc[3]
        tm.assert_series_equal(result, expected)

    def test_getitem_ix_boolean_duplicates_multiple(self):
        # #1201
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["foo", "foo", "bar", "baz", "bar"],
        )

        result = df.loc[["bar"]]
        exp = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, exp)

        result = df.loc[df[1] > 0]
        exp = df[df[1] > 0]
        tm.assert_frame_equal(result, exp)

        result = df.loc[df[0] > 0]
        exp = df[df[0] > 0]
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize("bool_value", [True, False])
    def test_getitem_setitem_ix_bool_keyerror(self, bool_value):
        # #2199
        df = DataFrame({"a": [1, 2, 3]})
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        with pytest.raises(KeyError, match=message):
            df.loc[bool_value]

        msg = "cannot use a single bool to index into setitem"
        with pytest.raises(KeyError, match=msg):
            df.loc[bool_value] = 0

    # TODO: rename?  remove?
    def test_single_element_ix_dont_upcast(self, float_frame):
        float_frame["E"] = 1
        assert issubclass(float_frame["E"].dtype.type, (int, np.integer))

        result = float_frame.loc[float_frame.index[5], "E"]
        assert is_integer(result)

        # GH 11617
        df = DataFrame({"a": [1.23]})
        df["b"] = 666

        result = df.loc[0, "b"]
        assert is_integer(result)

        expected = Series([666], [0], name="b")
        result = df.loc[[0], "b"]
        tm.assert_series_equal(result, expected)

    def test_iloc_callable_tuple_return_value(self):
        # GH53769
        df = DataFrame(np.arange(40).reshape(10, 4), index=range(0, 20, 2))
        msg = "callable with iloc"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.iloc[lambda _: (0,)]
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.iloc[lambda _: (0,)] = 1

    def test_iloc_row(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2)
        )

        result = df.iloc[1]
        exp = df.loc[2]
        tm.assert_series_equal(result, exp)

        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)

        # slice
        result = df.iloc[slice(4, 8)]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[[1, 2, 4, 6]]
        expected = df.reindex(df.index[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_row_slice_view(self, using_copy_on_write, warn_copy_on_write):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2)
        )
        original = df.copy()

        # verify slice is view
        # setting it makes it raise/warn
        subset = df.iloc[slice(4, 8)]

        assert np.shares_memory(df[2], subset[2])

        exp_col = original[2].copy()
        with tm.assert_cow_warning(warn_copy_on_write):
            subset.loc[:, 2] = 0.0
        if not using_copy_on_write:
            exp_col._values[4:8] = 0.0

            # With the enforcement of GH#45333 in 2.0, this remains a view
            assert np.shares_memory(df[2], subset[2])
        tm.assert_series_equal(df[2], exp_col)

    def test_iloc_col(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2)
        )

        result = df.iloc[:, 1]
        exp = df.loc[:, 2]
        tm.assert_series_equal(result, exp)

        result = df.iloc[:, 2]
        exp = df.loc[:, 4]
        tm.assert_series_equal(result, exp)

        # slice
        result = df.iloc[:, slice(4, 8)]
        expected = df.loc[:, 8:14]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[:, [1, 2, 4, 6]]
        expected = df.reindex(columns=df.columns[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_col_slice_view(
        self, using_array_manager, using_copy_on_write, warn_copy_on_write
    ):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2)
        )
        original = df.copy()
        subset = df.iloc[:, slice(4, 8)]

        if not using_array_manager and not using_copy_on_write:
            # verify slice is view
            assert np.shares_memory(df[8]._values, subset[8]._values)

            with tm.assert_cow_warning(warn_copy_on_write):
                subset.loc[:, 8] = 0.0

            assert (df[8] == 0).all()

            # with the enforcement of GH#45333 in 2.0, this remains a view
            assert np.shares_memory(df[8]._values, subset[8]._values)
        else:
            if using_copy_on_write:
                # verify slice is view
                assert np.shares_memory(df[8]._values, subset[8]._values)
            subset[8] = 0.0
            # subset changed
            assert (subset[8] == 0).all()
            # but df itself did not change (setitem replaces full column)
            tm.assert_frame_equal(df, original)

    def test_loc_duplicates(self):
        # gh-17105

        # insert a duplicate element to the index
        trange = date_range(
            start=Timestamp(year=2017, month=1, day=1),
            end=Timestamp(year=2017, month=1, day=5),
        )

        trange = trange.insert(loc=5, item=Timestamp(year=2017, month=1, day=5))

        df = DataFrame(0, index=trange, columns=["A", "B"])
        bool_idx = np.array([False, False, False, False, False, True])

        # assignment
        df.loc[trange[bool_idx], "A"] = 6

        expected = DataFrame(
            {"A": [0, 0, 0, 0, 6, 6], "B": [0, 0, 0, 0, 0, 0]}, index=trange
        )
        tm.assert_frame_equal(df, expected)

        # in-place
        df = DataFrame(0, index=trange, columns=["A", "B"])
        df.loc[trange[bool_idx], "A"] += 6
        tm.assert_frame_equal(df, expected)

    def test_setitem_with_unaligned_tz_aware_datetime_column(self):
        # GH 12981
        # Assignment of unaligned offset-aware datetime series.
        # Make sure timezone isn't lost
        column = Series(date_range("2015-01-01", periods=3, tz="utc"), name="dates")
        df = DataFrame({"dates": column})
        df["dates"] = column[[1, 0, 2]]
        tm.assert_series_equal(df["dates"], column)

        df = DataFrame({"dates": column})
        df.loc[[0, 1, 2], "dates"] = column[[1, 0, 2]]
        tm.assert_series_equal(df["dates"], column)

    def test_loc_setitem_datetimelike_with_inference(self):
        # GH 7592
        # assignment of timedeltas with NaT

        one_hour = timedelta(hours=1)
        df = DataFrame(index=date_range("20130101", periods=4))
        df["A"] = np.array([1 * one_hour] * 4, dtype="m8[ns]")
        df.loc[:, "B"] = np.array([2 * one_hour] * 4, dtype="m8[ns]")
        df.loc[df.index[:3], "C"] = np.array([3 * one_hour] * 3, dtype="m8[ns]")
        df.loc[:, "D"] = np.array([4 * one_hour] * 4, dtype="m8[ns]")
        df.loc[df.index[:3], "E"] = np.array([5 * one_hour] * 3, dtype="m8[ns]")
        df["F"] = np.timedelta64("NaT")
        df.loc[df.index[:-1], "F"] = np.array([6 * one_hour] * 3, dtype="m8[ns]")
        df.loc[df.index[-3] :, "G"] = date_range("20130101", periods=3)
        df["H"] = np.datetime64("NaT")
        result = df.dtypes
        expected = Series(
            [np.dtype("timedelta64[ns]")] * 6 + [np.dtype("datetime64[ns]")] * 2,
            index=Index(list("ABCDEFGH"), dtype=object),
        )
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_indexing_mixed(self):
        df = DataFrame(
            {
                0: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan},
                1: {
                    35: np.nan,
                    40: 0.32632316859446198,
                    43: np.nan,
                    49: 0.32632316859446198,
                    50: 0.39114724480578139,
                },
                2: {
                    35: np.nan,
                    40: np.nan,
                    43: 0.29012581014105987,
                    49: np.nan,
                    50: np.nan,
                },
                3: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan},
                4: {
                    35: 0.34215328467153283,
                    40: np.nan,
                    43: np.nan,
                    49: np.nan,
                    50: np.nan,
                },
                "y": {35: 0, 40: 0, 43: 0, 49: 0, 50: 1},
            }
        )

        # mixed int/float ok
        df2 = df.copy()
        df2[df2 > 0.3] = 1
        expected = df.copy()
        expected.loc[40, 1] = 1
        expected.loc[49, 1] = 1
        expected.loc[50, 1] = 1
        expected.loc[35, 4] = 1
        tm.assert_frame_equal(df2, expected)

        df["foo"] = "test"
        msg = "not supported between instances|unorderable types|Invalid comparison"

        with pytest.raises(TypeError, match=msg):
            df[df > 0.3] = 1

    def test_type_error_multiindex(self):
        # See gh-12218
        mi = MultiIndex.from_product([["x", "y"], [0, 1]], names=[None, "c"])
        dg = DataFrame(
            [[1, 1, 2, 2], [3, 3, 4, 4]], columns=mi, index=Index([0, 1], name="i")
        )
        with pytest.raises(InvalidIndexError, match="slice"):
            dg[:, 0]

        index = Index(range(2), name="i")
        columns = MultiIndex(
            levels=[["x", "y"], [0, 1]], codes=[[0, 1], [0, 0]], names=[None, "c"]
        )
        expected = DataFrame([[1, 2], [3, 4]], columns=columns, index=index)

        result = dg.loc[:, (slice(None), 0)]
        tm.assert_frame_equal(result, expected)

        name = ("x", 0)
        index = Index(range(2), name="i")
        expected = Series([1, 3], index=index, name=name)

        result = dg["x", 0]
        tm.assert_series_equal(result, expected)

    def test_getitem_interval_index_partial_indexing(self):
        # GH#36490
        df = DataFrame(
            np.ones((3, 4)), columns=pd.IntervalIndex.from_breaks(np.arange(5))
        )

        expected = df.iloc[:, 0]

        res = df[0.5]
        tm.assert_series_equal(res, expected)

        res = df.loc[:, 0.5]
        tm.assert_series_equal(res, expected)

    def test_setitem_array_as_cell_value(self):
        # GH#43422
        df = DataFrame(columns=["a", "b"], dtype=object)
        df.loc[0] = {"a": np.zeros((2,)), "b": np.zeros((2, 2))}
        expected = DataFrame({"a": [np.zeros((2,))], "b": [np.zeros((2, 2))]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_nullable_2d_values(self):
        df = DataFrame({"A": [1, 2, 3]}, dtype="Int64")
        orig = df.copy()

        df.loc[:] = df.values[:, ::-1]
        tm.assert_frame_equal(df, orig)

        df.loc[:] = pd.core.arrays.NumpyExtensionArray(df.values[:, ::-1])
        tm.assert_frame_equal(df, orig)

        df.iloc[:] = df.iloc[:, :].copy()
        tm.assert_frame_equal(df, orig)

    def test_getitem_segfault_with_empty_like_object(self):
        # GH#46848
        df = DataFrame(np.empty((1, 1), dtype=object))
        df[0] = np.empty_like(df[0])
        # this produces the segfault
        df[[0]]

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    @pytest.mark.parametrize(
        "null", [pd.NaT, pd.NaT.to_numpy("M8[ns]"), pd.NaT.to_numpy("m8[ns]")]
    )
    def test_setting_mismatched_na_into_nullable_fails(
        self, null, any_numeric_ea_dtype
    ):
        # GH#44514 don't cast mismatched nulls to pd.NA
        df = DataFrame({"A": [1, 2, 3]}, dtype=any_numeric_ea_dtype)
        ser = df["A"].copy()
        arr = ser._values

        msg = "|".join(
            [
                r"timedelta64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                r"datetime64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                "'values' contains non-numeric NA",
                r"Invalid value '.*' for dtype '(U?Int|Float)\d{1,2}'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr[0] = null

        with pytest.raises(TypeError, match=msg):
            arr[:2] = [null, null]

        with pytest.raises(TypeError, match=msg):
            ser[0] = null

        with pytest.raises(TypeError, match=msg):
            ser[:2] = [null, null]

        with pytest.raises(TypeError, match=msg):
            ser.iloc[0] = null

        with pytest.raises(TypeError, match=msg):
            ser.iloc[:2] = [null, null]

        with pytest.raises(TypeError, match=msg):
            df.iloc[0, 0] = null

        with pytest.raises(TypeError, match=msg):
            df.iloc[:2, 0] = [null, null]

        # Multi-Block
        df2 = df.copy()
        df2["B"] = ser.copy()
        with pytest.raises(TypeError, match=msg):
            df2.iloc[0, 0] = null

        with pytest.raises(TypeError, match=msg):
            df2.iloc[:2, 0] = [null, null]

    def test_loc_expand_empty_frame_keep_index_name(self):
        # GH#45621
        df = DataFrame(columns=["b"], index=Index([], name="a"))
        df.loc[0] = 1
        expected = DataFrame({"b": [1]}, index=Index([0], name="a"))
        tm.assert_frame_equal(df, expected)

    def test_loc_expand_empty_frame_keep_midx_names(self):
        # GH#46317
        df = DataFrame(
            columns=["d"], index=MultiIndex.from_tuples([], names=["a", "b", "c"])
        )
        df.loc[(1, 2, 3)] = "foo"
        expected = DataFrame(
            {"d": ["foo"]},
            index=MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"]),
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "val, idxr",
        [
            ("x", "a"),
            ("x", ["a"]),
            (1, "a"),
            (1, ["a"]),
        ],
    )
    def test_loc_setitem_rhs_frame(self, idxr, val):
        # GH#47578
        df = DataFrame({"a": [1, 2]})

        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            df.loc[:, idxr] = DataFrame({"a": [val, 11]}, index=[1, 2])
        expected = DataFrame({"a": [np.nan, val]})
        tm.assert_frame_equal(df, expected)

    @td.skip_array_manager_invalid_test
    def test_iloc_setitem_enlarge_no_warning(self, warn_copy_on_write):
        # GH#47381
        df = DataFrame(columns=["a", "b"])
        expected = df.copy()
        view = df[:]
        df.iloc[:, 0] = np.array([1, 2], dtype=np.float64)
        tm.assert_frame_equal(view, expected)

    def test_loc_internals_not_updated_correctly(self):
        # GH#47867 all steps are necessary to reproduce the initial bug
        df = DataFrame(
            {"bool_col": True, "a": 1, "b": 2.5},
            index=MultiIndex.from_arrays([[1, 2], [1, 2]], names=["idx1", "idx2"]),
        )
        idx = [(1, 1)]

        df["c"] = 3
        df.loc[idx, "c"] = 0

        df.loc[idx, "c"]
        df.loc[idx, ["a", "b"]]

        df.loc[idx, "c"] = 15
        result = df.loc[idx, "c"]
        expected = df = Series(
            15,
            index=MultiIndex.from_arrays([[1], [1]], names=["idx1", "idx2"]),
            name="c",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [None, [None], pd.NA, [pd.NA]])
    def test_iloc_setitem_string_list_na(self, val):
        # GH#45469
        df = DataFrame({"a": ["a", "b", "c"]}, dtype="string")
        df.iloc[[0], :] = val
        expected = DataFrame({"a": [pd.NA, "b", "c"]}, dtype="string")
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("val", [None, pd.NA])
    def test_iloc_setitem_string_na(self, val):
        # GH#45469
        df = DataFrame({"a": ["a", "b", "c"]}, dtype="string")
        df.iloc[0, :] = val
        expected = DataFrame({"a": [pd.NA, "b", "c"]}, dtype="string")
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("func", [list, Series, np.array])
    def test_iloc_setitem_ea_null_slice_length_one_list(self, func):
        # GH#48016
        df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
        df.iloc[:, func([0])] = 5
        expected = DataFrame({"a": [5, 5, 5]}, dtype="Int64")
        tm.assert_frame_equal(df, expected)

    def test_loc_named_tuple_for_midx(self):
        # GH#48124
        df = DataFrame(
            index=MultiIndex.from_product(
                [["A", "B"], ["a", "b", "c"]], names=["first", "second"]
            )
        )
        indexer_tuple = namedtuple("Indexer", df.index.names)
        idxr = indexer_tuple(first="A", second=["a", "b"])
        result = df.loc[idxr, :]
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["first", "second"]
            )
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("indexer", [["a"], "a"])
    @pytest.mark.parametrize("col", [{}, {"b": 1}])
    def test_set_2d_casting_date_to_int(self, col, indexer):
        # GH#49159
        df = DataFrame(
            {"a": [Timestamp("2022-12-29"), Timestamp("2022-12-30")], **col},
        )
        df.loc[[1], indexer] = df["a"] + pd.Timedelta(days=1)
        expected = DataFrame(
            {"a": [Timestamp("2022-12-29"), Timestamp("2022-12-31")], **col},
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("col", [{}, {"name": "a"}])
    def test_loc_setitem_reordering_with_all_true_indexer(self, col):
        # GH#48701
        n = 17
        df = DataFrame({**col, "x": range(n), "y": range(n)})
        expected = df.copy()
        df.loc[n * [True], ["x", "y"]] = df[["x", "y"]]
        tm.assert_frame_equal(df, expected)

    def test_loc_rhs_empty_warning(self):
        # GH48480
        df = DataFrame(columns=["a", "b"])
        expected = df.copy()
        rhs = DataFrame(columns=["a"])
        with tm.assert_produces_warning(None):
            df.loc[:, "a"] = rhs
        tm.assert_frame_equal(df, expected)

    def test_iloc_ea_series_indexer(self):
        # GH#49521
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        indexer = Series([0, 1], dtype="Int64")
        row_indexer = Series([1], dtype="Int64")
        result = df.iloc[row_indexer, indexer]
        expected = DataFrame([[5, 6]], index=[1])
        tm.assert_frame_equal(result, expected)

        result = df.iloc[row_indexer.values, indexer.values]
        tm.assert_frame_equal(result, expected)

    def test_iloc_ea_series_indexer_with_na(self):
        # GH#49521
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        indexer = Series([0, pd.NA], dtype="Int64")
        msg = "cannot convert"
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer]
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer.values]

    @pytest.mark.parametrize("indexer", [True, (True,)])
    @pytest.mark.parametrize("dtype", [bool, "boolean"])
    def test_loc_bool_multiindex(self, dtype, indexer):
        # GH#47687
        midx = MultiIndex.from_arrays(
            [
                Series([True, True, False, False], dtype=dtype),
                Series([True, False, True, False], dtype=dtype),
            ],
            names=["a", "b"],
        )
        df = DataFrame({"c": [1, 2, 3, 4]}, index=midx)
        with tm.maybe_produces_warning(PerformanceWarning, isinstance(indexer, tuple)):
            result = df.loc[indexer]
        expected = DataFrame(
            {"c": [1, 2]}, index=Index([True, False], name="b", dtype=dtype)
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("utc", [False, True])
    @pytest.mark.parametrize("indexer", ["date", ["date"]])
    def test_loc_datetime_assignment_dtype_does_not_change(self, utc, indexer):
        # GH#49837
        df = DataFrame(
            {
                "date": to_datetime(
                    [datetime(2022, 1, 20), datetime(2022, 1, 22)], utc=utc
                ),
                "update": [True, False],
            }
        )
        expected = df.copy(deep=True)

        update_df = df[df["update"]]

        df.loc[df["update"], indexer] = update_df["date"]

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer, idx", [(tm.loc, 1), (tm.iloc, 2)])
    def test_setitem_value_coercing_dtypes(self, indexer, idx):
        # GH#50467
        df = DataFrame([["1", np.nan], ["2", np.nan], ["3", np.nan]], dtype=object)
        rhs = DataFrame([[1, np.nan], [2, np.nan]])
        indexer(df)[:idx, :] = rhs
        expected = DataFrame([[1, np.nan], [2, np.nan], ["3", np.nan]], dtype=object)
        tm.assert_frame_equal(df, expected)


class TestDataFrameIndexingUInt64:
    def test_setitem(self):
        df = DataFrame(
            {"A": np.arange(3), "B": [2**63, 2**63 + 5, 2**63 + 10]},
            dtype=np.uint64,
        )
        idx = df["A"].rename("foo")

        # setitem
        assert "C" not in df.columns
        df["C"] = idx
        tm.assert_series_equal(df["C"], Series(idx, name="C"))

        assert "D" not in df.columns
        df["D"] = "foo"
        df["D"] = idx
        tm.assert_series_equal(df["D"], Series(idx, name="D"))
        del df["D"]

        # With NaN: because uint64 has no NaN element,
        # the column should be cast to object.
        df2 = df.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df2.iloc[1, 1] = pd.NaT
            df2.iloc[1, 2] = pd.NaT
        result = df2["B"]
        tm.assert_series_equal(notna(result), Series([True, False, True], name="B"))
        tm.assert_series_equal(
            df2.dtypes,
            Series(
                [np.dtype("uint64"), np.dtype("O"), np.dtype("O")],
                index=["A", "B", "C"],
            ),
        )


def test_object_casting_indexing_wraps_datetimelike(using_array_manager):
    # GH#31649, check the indexing methods all the way down the stack
    df = DataFrame(
        {
            "A": [1, 2],
            "B": date_range("2000", periods=2),
            "C": pd.timedelta_range("1 Day", periods=2),
        }
    )

    ser = df.loc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    ser = df.iloc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    ser = df.xs(0, axis=0)
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    if using_array_manager:
        # remainder of the test checking BlockManager internals
        return

    mgr = df._mgr
    mgr._rebuild_blknos_and_blklocs()
    arr = mgr.fast_xs(0).array
    assert isinstance(arr[1], Timestamp)
    assert isinstance(arr[2], pd.Timedelta)

    blk = mgr.blocks[mgr.blknos[1]]
    assert blk.dtype == "M8[ns]"  # we got the right block
    val = blk.iget((0, 0))
    assert isinstance(val, Timestamp)

    blk = mgr.blocks[mgr.blknos[2]]
    assert blk.dtype == "m8[ns]"  # we got the right block
    val = blk.iget((0, 0))
    assert isinstance(val, pd.Timedelta)


msg1 = r"Cannot setitem on a Categorical with a new category( \(.*\))?, set the"
msg2 = "Cannot set a Categorical with another, without identical categories"


class TestLocILocDataFrameCategorical:
    @pytest.fixture
    def orig(self):
        cats = Categorical(["a", "a", "a", "a", "a", "a", "a"], categories=["a", "b"])
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 1, 1, 1, 1, 1, 1]
        orig = DataFrame({"cats": cats, "values": values}, index=idx)
        return orig

    @pytest.fixture
    def exp_single_row(self):
        # The expected values if we change a single row
        cats1 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        idx1 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values1 = [1, 1, 2, 1, 1, 1, 1]
        exp_single_row = DataFrame({"cats": cats1, "values": values1}, index=idx1)
        return exp_single_row

    @pytest.fixture
    def exp_multi_row(self):
        # assign multiple rows (mixed values) (-> array) -> exp_multi_row
        # changed multiple rows
        cats2 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx2 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values2 = [1, 1, 2, 2, 1, 1, 1]
        exp_multi_row = DataFrame({"cats": cats2, "values": values2}, index=idx2)
        return exp_multi_row

    @pytest.fixture
    def exp_parts_cats_col(self):
        # changed part of the cats column
        cats3 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx3 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values3 = [1, 1, 1, 1, 1, 1, 1]
        exp_parts_cats_col = DataFrame({"cats": cats3, "values": values3}, index=idx3)
        return exp_parts_cats_col

    @pytest.fixture
    def exp_single_cats_value(self):
        # changed single value in cats col
        cats4 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        idx4 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values4 = [1, 1, 1, 1, 1, 1, 1]
        exp_single_cats_value = DataFrame(
            {"cats": cats4, "values": values4}, index=idx4
        )
        return exp_single_cats_value

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_list_of_lists(self, orig, exp_multi_row, indexer):
        #   - assign multiple rows (mixed values) -> exp_multi_row
        df = orig.copy()

        key = slice(2, 4)
        if indexer is tm.loc:
            key = slice("j", "k")

        indexer(df)[key, :] = [["b", 2], ["b", 2]]
        tm.assert_frame_equal(df, exp_multi_row)

        df = orig.copy()
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key, :] = [["c", 2], ["c", 2]]

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc, tm.at, tm.iat])
    def test_loc_iloc_at_iat_setitem_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer
    ):
        #   - assign a single value -> exp_single_cats_value
        df = orig.copy()

        key = (2, 0)
        if indexer in [tm.loc, tm.at]:
            key = (df.index[2], df.columns[0])

        # "b" is among the categories for df["cat"}]
        indexer(df)[key] = "b"
        tm.assert_frame_equal(df, exp_single_cats_value)

        # "c" is not among the categories for df["cat"]
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key] = "c"

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_mask_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer
    ):
        # mask with single True
        df = orig.copy()

        mask = df.index == "j"
        key = 0
        if indexer is tm.loc:
            key = df.columns[key]

        indexer(df)[mask, key] = "b"
        tm.assert_frame_equal(df, exp_single_cats_value)

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_full_row_non_categorical_rhs(
        self, orig, exp_single_row, indexer
    ):
        #   - assign a complete row (mixed values) -> exp_single_row
        df = orig.copy()

        key = 2
        if indexer is tm.loc:
            key = df.index[2]

        # not categorical dtype, but "b" _is_ among the categories for df["cat"]
        indexer(df)[key, :] = ["b", 2]
        tm.assert_frame_equal(df, exp_single_row)

        # "c" is not among the categories for df["cat"]
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key, :] = ["c", 2]

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_partial_col_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer
    ):
        # assign a part of a column with dtype == categorical ->
        # exp_parts_cats_col
        df = orig.copy()

        key = (slice(2, 4), 0)
        if indexer is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # same categories as we currently have in df["cats"]
        compat = Categorical(["b", "b"], categories=["a", "b"])
        indexer(df)[key] = compat
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # categories do not match df["cat"]'s, but "b" is among them
        semi_compat = Categorical(list("bb"), categories=list("abc"))
        with pytest.raises(TypeError, match=msg2):
            # different categories but holdable values
            #  -> not sure if this should fail or pass
            indexer(df)[key] = semi_compat

        # categories do not match df["cat"]'s, and "c" is not among them
        incompat = Categorical(list("cc"), categories=list("abc"))
        with pytest.raises(TypeError, match=msg2):
            # different values
            indexer(df)[key] = incompat

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_non_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer
    ):
        # assign a part of a column with dtype != categorical -> exp_parts_cats_col
        df = orig.copy()

        key = (slice(2, 4), 0)
        if indexer is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # "b" is among the categories for df["cat"]
        indexer(df)[key] = ["b", "b"]
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # "c" not part of the categories
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key] = ["c", "c"]

    @pytest.mark.parametrize("indexer", [tm.getitem, tm.loc, tm.iloc])
    def test_getitem_preserve_object_index_with_dates(self, indexer):
        # https://github.com/pandas-dev/pandas/pull/42950 - when selecting a column
        # from dataframe, don't try to infer object dtype index on Series construction
        idx = date_range("2012", periods=3).astype(object)
        df = DataFrame({0: [1, 2, 3]}, index=idx)
        assert df.index.dtype == object

        if indexer is tm.getitem:
            ser = indexer(df)[0]
        else:
            ser = indexer(df)[:, 0]

        assert ser.index.dtype == object

    def test_loc_on_multiindex_one_level(self):
        # GH#45779
        df = DataFrame(
            data=[[0], [1]],
            index=MultiIndex.from_tuples([("a",), ("b",)], names=["first"]),
        )
        expected = DataFrame(
            data=[[0]], index=MultiIndex.from_tuples([("a",)], names=["first"])
        )
        result = df.loc["a"]
        tm.assert_frame_equal(result, expected)


class TestDeprecatedIndexers:
    @pytest.mark.parametrize(
        "key", [{1}, {1: 1}, ({1}, "a"), ({1: 1}, "a"), (1, {"a"}), (1, {"a": "a"})]
    )
    def test_getitem_dict_and_set_deprecated(self, key):
        # GH#42825 enforced in 2.0
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key]

    @pytest.mark.parametrize(
        "key",
        [
            {1},
            {1: 1},
            (({1}, 2), "a"),
            (({1: 1}, 2), "a"),
            ((1, 2), {"a"}),
            ((1, 2), {"a": "a"}),
        ],
    )
    def test_getitem_dict_and_set_deprecated_multiindex(self, key):
        # GH#42825 enforced in 2.0
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=["a", "b"],
            index=MultiIndex.from_tuples([(1, 2), (3, 4)]),
        )
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key]

    @pytest.mark.parametrize(
        "key", [{1}, {1: 1}, ({1}, "a"), ({1: 1}, "a"), (1, {"a"}), (1, {"a": "a"})]
    )
    def test_setitem_dict_and_set_disallowed(self, key):
        # GH#42825 enforced in 2.0
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key] = 1

    @pytest.mark.parametrize(
        "key",
        [
            {1},
            {1: 1},
            (({1}, 2), "a"),
            (({1: 1}, 2), "a"),
            ((1, 2), {"a"}),
            ((1, 2), {"a": "a"}),
        ],
    )
    def test_setitem_dict_and_set_disallowed_multiindex(self, key):
        # GH#42825 enforced in 2.0
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=["a", "b"],
            index=MultiIndex.from_tuples([(1, 2), (3, 4)]),
        )
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key] = 1


def test_adding_new_conditional_column() -> None:
    # https://github.com/pandas-dev/pandas/issues/55025
    df = DataFrame({"x": [1]})
    df.loc[df["x"] == 1, "y"] = "1"
    expected = DataFrame({"x": [1], "y": ["1"]})
    tm.assert_frame_equal(df, expected)

    df = DataFrame({"x": [1]})
    # try inserting something which numpy would store as 'object'
    value = lambda x: x
    df.loc[df["x"] == 1, "y"] = value
    expected = DataFrame({"x": [1], "y": [value]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    ("dtype", "infer_string"),
    [
        (object, False),
        (pd.StringDtype(na_value=np.nan), True),
    ],
)
def test_adding_new_conditional_column_with_string(dtype, infer_string) -> None:
    # https://github.com/pandas-dev/pandas/issues/56204
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pd.option_context("future.infer_string", infer_string):
        df.loc[df["a"] == 1, "c"] = "1"
    expected = DataFrame({"a": [1, 2], "b": [3, 4], "c": ["1", float("nan")]}).astype(
        {"a": "int64", "b": "int64", "c": dtype}
    )
    tm.assert_frame_equal(df, expected)


def test_add_new_column_infer_string():
    # GH#55366
    df = DataFrame({"x": [1]})
    with pd.option_context("future.infer_string", True):
        df.loc[df["x"] == 1, "y"] = "1"
    expected = DataFrame(
        {"x": [1], "y": Series(["1"], dtype=pd.StringDtype(na_value=np.nan))},
        columns=Index(["x", "y"], dtype="str"),
    )
    tm.assert_frame_equal(df, expected)


class TestSetitemValidation:
    # This is adapted from pandas/tests/arrays/masked/test_indexing.py
    # but checks for warnings instead of errors.
    def _check_setitem_invalid(self, df, invalid, indexer, warn):
        msg = "Setting an item of incompatible dtype is deprecated"
        msg = re.escape(msg)

        orig_df = df.copy()

        # iloc
        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[indexer, 0] = invalid
            df = orig_df.copy()

        # loc
        with tm.assert_produces_warning(warn, match=msg):
            df.loc[indexer, "a"] = invalid
            df = orig_df.copy()

    _invalid_scalars = [
        1 + 2j,
        "True",
        "1",
        "1.0",
        pd.NaT,
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ]
    _indexers = [0, [0], slice(0, 1), [True, False, False], slice(None, None, None)]

    @pytest.mark.parametrize(
        "invalid", _invalid_scalars + [1, 1.0, np.int64(1), np.float64(1)]
    )
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_bool(self, invalid, indexer):
        df = DataFrame({"a": [True, False, False]}, dtype="bool")
        self._check_setitem_invalid(df, invalid, indexer, FutureWarning)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True, 1.5, np.float64(1.5)])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_int(self, invalid, any_int_numpy_dtype, indexer):
        df = DataFrame({"a": [1, 2, 3]}, dtype=any_int_numpy_dtype)
        if isna(invalid) and invalid is not pd.NaT and not np.isnat(invalid):
            warn = None
        else:
            warn = FutureWarning
        self._check_setitem_invalid(df, invalid, indexer, warn)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
        df = DataFrame({"a": [1, 2, None]}, dtype=float_numpy_dtype)
        self._check_setitem_invalid(df, invalid, indexer, FutureWarning)