
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_sho1d.py ===
"""Tests for sho1d.py"""

from sympy.concrete import Sum
from sympy.core import oo
from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum import Commutator
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.cartesian import X, Px
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.represent import represent
from sympy.simplify import simplify
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.testing.pytest import skip, raises

from sympy.physics.quantum.sho1d import (RaisingOp, LoweringOp,
                                        SHOKet, SHOBra,
                                        Hamiltonian, NumberOp)

ad = RaisingOp('a')
a = LoweringOp('a')
k = SHOKet('k')
kz = SHOKet(0)
kf = SHOKet(1)
k3 = SHOKet(3)
b = SHOBra('b')
b3 = SHOBra(3)
H = Hamiltonian('H')
N = NumberOp('N')
omega = Symbol('omega')
m = Symbol('m')
ndim = Integer(4)
p = Symbol('p', integer=True)
q = Symbol('q', nonnegative=True, integer=True)


np = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})

ad_rep_sympy = represent(ad, basis=N, ndim=4, format='sympy')
a_rep = represent(a, basis=N, ndim=4, format='sympy')
N_rep = represent(N, basis=N, ndim=4, format='sympy')
H_rep = represent(H, basis=N, ndim=4, format='sympy')
k3_rep = represent(k3, basis=N, ndim=4, format='sympy')
b3_rep = represent(b3, basis=N, ndim=4, format='sympy')

def test_RaisingOp():
    assert Dagger(ad) == a
    assert Commutator(ad, a).doit() == Integer(-1)
    assert Commutator(ad, N).doit() == Integer(-1)*ad
    assert qapply(ad*k) == (sqrt(k.n + 1)*SHOKet(k.n + 1)).expand()
    assert qapply(ad*kz) == (sqrt(kz.n + 1)*SHOKet(kz.n + 1)).expand()
    assert qapply(ad*kf) == (sqrt(kf.n + 1)*SHOKet(kf.n + 1)).expand()
    assert ad.rewrite('xp').doit() == \
        (Integer(1)/sqrt(Integer(2)*hbar*m*omega))*(Integer(-1)*I*Px + m*omega*X)
    assert ad.hilbert_space == ComplexSpace(S.Infinity)
    for i in range(ndim - 1):
        assert ad_rep_sympy[i + 1,i] == sqrt(i + 1)

    if not np:
        skip("numpy not installed.")

    ad_rep_numpy = represent(ad, basis=N, ndim=4, format='numpy')
    for i in range(ndim - 1):
        assert ad_rep_numpy[i + 1,i] == float(sqrt(i + 1))

    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")

    ad_rep_scipy = represent(ad, basis=N, ndim=4, format='scipy.sparse', spmatrix='lil')
    for i in range(ndim - 1):
        assert ad_rep_scipy[i + 1,i] == float(sqrt(i + 1))

    assert ad_rep_numpy.dtype == 'float64'
    assert ad_rep_scipy.dtype == 'float64'

def test_LoweringOp():
    assert Dagger(a) == ad
    assert Commutator(a, ad).doit() == Integer(1)
    assert Commutator(a, N).doit() == a
    assert qapply(a*k) == (sqrt(k.n)*SHOKet(k.n-Integer(1))).expand()
    assert qapply(a*kz) == Integer(0)
    assert qapply(a*kf) == (sqrt(kf.n)*SHOKet(kf.n-Integer(1))).expand()
    assert a.rewrite('xp').doit() == \
        (Integer(1)/sqrt(Integer(2)*hbar*m*omega))*(I*Px + m*omega*X)
    for i in range(ndim - 1):
        assert a_rep[i,i + 1] == sqrt(i + 1)

def test_NumberOp():
    assert Commutator(N, ad).doit() == ad
    assert Commutator(N, a).doit() == Integer(-1)*a
    assert Commutator(N, H).doit() == Integer(0)
    assert qapply(N*k) == (k.n*k).expand()
    assert N.rewrite('a').doit() == ad*a
    assert N.rewrite('xp').doit() == (Integer(1)/(Integer(2)*m*hbar*omega))*(
        Px**2 + (m*omega*X)**2) - Integer(1)/Integer(2)
    assert N.rewrite('H').doit() == H/(hbar*omega) - Integer(1)/Integer(2)
    for i in range(ndim):
        assert N_rep[i,i] == i
    assert N_rep == ad_rep_sympy*a_rep

def test_Hamiltonian():
    assert Commutator(H, N).doit() == Integer(0)
    assert qapply(H*k) == ((hbar*omega*(k.n + Integer(1)/Integer(2)))*k).expand()
    assert H.rewrite('a').doit() == hbar*omega*(ad*a + Integer(1)/Integer(2))
    assert H.rewrite('xp').doit() == \
        (Integer(1)/(Integer(2)*m))*(Px**2 + (m*omega*X)**2)
    assert H.rewrite('N').doit() == hbar*omega*(N + Integer(1)/Integer(2))
    for i in range(ndim):
        assert H_rep[i,i] == hbar*omega*(i + Integer(1)/Integer(2))

def test_SHOKet():
    assert SHOKet('k').dual_class() == SHOBra
    assert SHOBra('b').dual_class() == SHOKet
    assert InnerProduct(b,k).doit() == KroneckerDelta(k.n, b.n)
    assert k.hilbert_space == ComplexSpace(S.Infinity)
    assert k3_rep[k3.n, 0] == Integer(1)
    assert b3_rep[0, b3.n] == Integer(1)

def test_sho_sums():
    e1 = Sum(SHOKet(p)*SHOBra(p), (p, 0, 1))
    assert e1.doit() == SHOKet(0)*SHOBra(0) + SHOKet(1)*SHOBra(1)

    # Test qapply with Sum on the left
    assert qapply(
        Sum(SHOKet(p)*SHOBra(p), (p, 0, oo))*SHOKet(q),
        sum_doit=True
    ) == SHOKet(q)

    # Test qapply with Sum on the right
    a = IndexedBase('a')
    n = symbols('n', cls=Idx)
    result = qapply(SHOBra(q)*Sum(a[n]*SHOKet(n), (n,0,oo)), sum_doit=True)
    assert result == a[q]

    # Test qapply with a product of Sums
    result = qapply(
        SHOBra(q)*Sum(SHOKet(p)*SHOBra(p), (p, 0, oo))*Sum(a[n]*SHOKet(n), (n,0,oo)),
        sum_doit=True
    )
    assert result == a[q]

    with raises(ValueError):
        result = qapply(
            SHOBra(q)*Sum(SHOKet(p)*SHOBra(p), (p, 0, oo))*Sum(a[p]*SHOKet(p), (p,0,oo)),
            sum_doit=True
        )

def test_sho_coherant_state():
    alpha = Symbol('alpha', is_complex=True)
    cstate = exp(-Abs(alpha)**2/S(2))*Sum(((alpha**p)/sqrt(factorial(p)))*SHOKet(p), (p,0,oo))
    # Projection onto the number eigenstate
    assert qapply(SHOBra(q)*cstate, sum_doit=True) == exp(-Abs(alpha)**2/S(2))*alpha**q/sqrt(factorial(q))
    # Ensure that the coherent state is an eigenstate of annihilation operator
    assert simplify(qapply(SHOBra(q)*a*cstate, sum_doit=True)) == simplify(qapply(SHOBra(q)*alpha*cstate, sum_doit=True))

def test_issue_26495():
    nbar = Symbol('nbar', real=True, nonnegative=True)
    n = Symbol('n', integer=True)
    i = Symbol('i', integer=True, nonnegative=True)
    j = Symbol('j', integer=True, nonnegative=True)
    rho = Sum((nbar/(1+nbar))**n*SHOKet(n)*SHOBra(n), (n,0,oo))
    result = qapply(SHOBra(i)*rho*SHOKet(j), sum_doit=True)
    assert simplify(result) == (nbar/(nbar+1))**i*KroneckerDelta(i,j)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_find_common_type.py ===
import numpy as np
import pytest

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

from pandas import (
    Categorical,
    Index,
)


@pytest.mark.parametrize(
    "source_dtypes,expected_common_dtype",
    [
        ((np.int64,), np.int64),
        ((np.uint64,), np.uint64),
        ((np.float32,), np.float32),
        ((object,), object),
        # Into ints.
        ((np.int16, np.int64), np.int64),
        ((np.int32, np.uint32), np.int64),
        ((np.uint16, np.uint64), np.uint64),
        # Into floats.
        ((np.float16, np.float32), np.float32),
        ((np.float16, np.int16), np.float32),
        ((np.float32, np.int16), np.float32),
        ((np.uint64, np.int64), np.float64),
        ((np.int16, np.float64), np.float64),
        ((np.float16, np.int64), np.float64),
        # Into others.
        ((np.complex128, np.int32), np.complex128),
        ((object, np.float32), object),
        ((object, np.int16), object),
        # Bool with int.
        ((np.dtype("bool"), np.int64), object),
        ((np.dtype("bool"), np.int32), object),
        ((np.dtype("bool"), np.int16), object),
        ((np.dtype("bool"), np.int8), object),
        ((np.dtype("bool"), np.uint64), object),
        ((np.dtype("bool"), np.uint32), object),
        ((np.dtype("bool"), np.uint16), object),
        ((np.dtype("bool"), np.uint8), object),
        # Bool with float.
        ((np.dtype("bool"), np.float64), object),
        ((np.dtype("bool"), np.float32), object),
        (
            (np.dtype("datetime64[ns]"), np.dtype("datetime64[ns]")),
            np.dtype("datetime64[ns]"),
        ),
        (
            (np.dtype("timedelta64[ns]"), np.dtype("timedelta64[ns]")),
            np.dtype("timedelta64[ns]"),
        ),
        (
            (np.dtype("datetime64[ns]"), np.dtype("datetime64[ms]")),
            np.dtype("datetime64[ns]"),
        ),
        (
            (np.dtype("timedelta64[ms]"), np.dtype("timedelta64[ns]")),
            np.dtype("timedelta64[ns]"),
        ),
        ((np.dtype("datetime64[ns]"), np.dtype("timedelta64[ns]")), object),
        ((np.dtype("datetime64[ns]"), np.int64), object),
    ],
)
def test_numpy_dtypes(source_dtypes, expected_common_dtype):
    source_dtypes = [pandas_dtype(x) for x in source_dtypes]
    assert find_common_type(source_dtypes) == expected_common_dtype


def test_raises_empty_input():
    with pytest.raises(ValueError, match="no types given"):
        find_common_type([])


@pytest.mark.parametrize(
    "dtypes,exp_type",
    [
        ([CategoricalDtype()], "category"),
        ([object, CategoricalDtype()], object),
        ([CategoricalDtype(), CategoricalDtype()], "category"),
    ],
)
def test_categorical_dtype(dtypes, exp_type):
    assert find_common_type(dtypes) == exp_type


def test_datetimetz_dtype_match():
    dtype = DatetimeTZDtype(unit="ns", tz="US/Eastern")
    assert find_common_type([dtype, dtype]) == "datetime64[ns, US/Eastern]"


@pytest.mark.parametrize(
    "dtype2",
    [
        DatetimeTZDtype(unit="ns", tz="Asia/Tokyo"),
        np.dtype("datetime64[ns]"),
        object,
        np.int64,
    ],
)
def test_datetimetz_dtype_mismatch(dtype2):
    dtype = DatetimeTZDtype(unit="ns", tz="US/Eastern")
    assert find_common_type([dtype, dtype2]) == object
    assert find_common_type([dtype2, dtype]) == object


def test_period_dtype_match():
    dtype = PeriodDtype(freq="D")
    assert find_common_type([dtype, dtype]) == "period[D]"


@pytest.mark.parametrize(
    "dtype2",
    [
        DatetimeTZDtype(unit="ns", tz="Asia/Tokyo"),
        PeriodDtype(freq="2D"),
        PeriodDtype(freq="h"),
        np.dtype("datetime64[ns]"),
        object,
        np.int64,
    ],
)
def test_period_dtype_mismatch(dtype2):
    dtype = PeriodDtype(freq="D")
    assert find_common_type([dtype, dtype2]) == object
    assert find_common_type([dtype2, dtype]) == object


interval_dtypes = [
    IntervalDtype(np.int64, "right"),
    IntervalDtype(np.float64, "right"),
    IntervalDtype(np.uint64, "right"),
    IntervalDtype(DatetimeTZDtype(unit="ns", tz="US/Eastern"), "right"),
    IntervalDtype("M8[ns]", "right"),
    IntervalDtype("m8[ns]", "right"),
]


@pytest.mark.parametrize("left", interval_dtypes)
@pytest.mark.parametrize("right", interval_dtypes)
def test_interval_dtype(left, right):
    result = find_common_type([left, right])

    if left is right:
        assert result is left

    elif left.subtype.kind in ["i", "u", "f"]:
        # i.e. numeric
        if right.subtype.kind in ["i", "u", "f"]:
            # both numeric -> common numeric subtype
            expected = IntervalDtype(np.float64, "right")
            assert result == expected
        else:
            assert result == object

    else:
        assert result == object


@pytest.mark.parametrize("dtype", interval_dtypes)
def test_interval_dtype_with_categorical(dtype):
    obj = Index([], dtype=dtype)

    cat = Categorical([], categories=obj)

    result = find_common_type([dtype, cat.dtype])
    assert result == dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\interchange\test_spec_conformance.py ===
"""
A verbatim copy (vendored) of the spec tests.
Taken from https://github.com/data-apis/dataframe-api
"""
import ctypes
import math

import pytest

import pandas as pd


@pytest.fixture
def df_from_dict():
    def maker(dct, is_categorical=False):
        df = pd.DataFrame(dct)
        return df.astype("category") if is_categorical else df

    return maker


@pytest.mark.parametrize(
    "test_data",
    [
        {"a": ["foo", "bar"], "b": ["baz", "qux"]},
        {"a": [1.5, 2.5, 3.5], "b": [9.2, 10.5, 11.8]},
        {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]},
    ],
    ids=["str_data", "float_data", "int_data"],
)
def test_only_one_dtype(test_data, df_from_dict):
    columns = list(test_data.keys())
    df = df_from_dict(test_data)
    dfX = df.__dataframe__()

    column_size = len(test_data[columns[0]])
    for column in columns:
        null_count = dfX.get_column_by_name(column).null_count
        assert null_count == 0
        assert isinstance(null_count, int)
        assert dfX.get_column_by_name(column).size() == column_size
        assert dfX.get_column_by_name(column).offset == 0


def test_mixed_dtypes(df_from_dict):
    df = df_from_dict(
        {
            "a": [1, 2, 3],  # dtype kind INT = 0
            "b": [3, 4, 5],  # dtype kind INT = 0
            "c": [1.5, 2.5, 3.5],  # dtype kind FLOAT = 2
            "d": [9, 10, 11],  # dtype kind INT = 0
            "e": [True, False, True],  # dtype kind BOOLEAN = 20
            "f": ["a", "", "c"],  # dtype kind STRING = 21
        }
    )
    dfX = df.__dataframe__()
    # for meanings of dtype[0] see the spec; we cannot import the spec here as this
    # file is expected to be vendored *anywhere*;
    # values for dtype[0] are explained above
    columns = {"a": 0, "b": 0, "c": 2, "d": 0, "e": 20, "f": 21}

    for column, kind in columns.items():
        colX = dfX.get_column_by_name(column)
        assert colX.null_count == 0
        assert isinstance(colX.null_count, int)
        assert colX.size() == 3
        assert colX.offset == 0

        assert colX.dtype[0] == kind

    assert dfX.get_column_by_name("c").dtype[1] == 64


def test_na_float(df_from_dict):
    df = df_from_dict({"a": [1.0, math.nan, 2.0]})
    dfX = df.__dataframe__()
    colX = dfX.get_column_by_name("a")
    assert colX.null_count == 1
    assert isinstance(colX.null_count, int)


def test_noncategorical(df_from_dict):
    df = df_from_dict({"a": [1, 2, 3]})
    dfX = df.__dataframe__()
    colX = dfX.get_column_by_name("a")
    with pytest.raises(TypeError, match=".*categorical.*"):
        colX.describe_categorical


def test_categorical(df_from_dict):
    df = df_from_dict(
        {"weekday": ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", "Sun"]},
        is_categorical=True,
    )

    colX = df.__dataframe__().get_column_by_name("weekday")
    categorical = colX.describe_categorical
    assert isinstance(categorical["is_ordered"], bool)
    assert isinstance(categorical["is_dictionary"], bool)


def test_dataframe(df_from_dict):
    df = df_from_dict(
        {"x": [True, True, False], "y": [1, 2, 0], "z": [9.2, 10.5, 11.8]}
    )
    dfX = df.__dataframe__()

    assert dfX.num_columns() == 3
    assert dfX.num_rows() == 3
    assert dfX.num_chunks() == 1
    assert list(dfX.column_names()) == ["x", "y", "z"]
    assert list(dfX.select_columns((0, 2)).column_names()) == list(
        dfX.select_columns_by_name(("x", "z")).column_names()
    )


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
def test_df_get_chunks(size, n_chunks, df_from_dict):
    df = df_from_dict({"x": list(range(size))})
    dfX = df.__dataframe__()
    chunks = list(dfX.get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    assert sum(chunk.num_rows() for chunk in chunks) == size


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
def test_column_get_chunks(size, n_chunks, df_from_dict):
    df = df_from_dict({"x": list(range(size))})
    dfX = df.__dataframe__()
    chunks = list(dfX.get_column(0).get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    assert sum(chunk.size() for chunk in chunks) == size


def test_get_columns(df_from_dict):
    df = df_from_dict({"a": [0, 1], "b": [2.5, 3.5]})
    dfX = df.__dataframe__()
    for colX in dfX.get_columns():
        assert colX.size() == 2
        assert colX.num_chunks() == 1
    # for meanings of dtype[0] see the spec; we cannot import the spec here as this
    # file is expected to be vendored *anywhere*
    assert dfX.get_column(0).dtype[0] == 0  # INT
    assert dfX.get_column(1).dtype[0] == 2  # FLOAT


def test_buffer(df_from_dict):
    arr = [0, 1, -1]
    df = df_from_dict({"a": arr})
    dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    bufX = colX.get_buffers()

    dataBuf, dataDtype = bufX["data"]

    assert dataBuf.bufsize > 0
    assert dataBuf.ptr != 0
    device, _ = dataBuf.__dlpack_device__()

    # for meanings of dtype[0] see the spec; we cannot import the spec here as this
    # file is expected to be vendored *anywhere*
    assert dataDtype[0] == 0  # INT

    if device == 1:  # CPU-only as we're going to directly read memory here
        bitwidth = dataDtype[1]
        ctype = {
            8: ctypes.c_int8,
            16: ctypes.c_int16,
            32: ctypes.c_int32,
            64: ctypes.c_int64,
        }[bitwidth]

        for idx, truth in enumerate(arr):
            val = ctype.from_address(dataBuf.ptr + idx * (bitwidth // 8)).value
            assert val == truth, f"Buffer at index {idx} mismatch"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_http_headers.py ===
"""
Tests for the pandas custom headers in http(s) requests
"""
from functools import partial
import gzip
from io import BytesIO

import pytest

from pandas._config import using_string_dtype

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

pytestmark = [
    pytest.mark.single_cpu,
    pytest.mark.network,
    pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
    ),
]


def gzip_bytes(response_bytes):
    with BytesIO() as bio:
        with gzip.GzipFile(fileobj=bio, mode="w") as zipper:
            zipper.write(response_bytes)
        return bio.getvalue()


def csv_responder(df):
    return df.to_csv(index=False).encode("utf-8")


def gz_csv_responder(df):
    return gzip_bytes(csv_responder(df))


def json_responder(df):
    return df.to_json().encode("utf-8")


def gz_json_responder(df):
    return gzip_bytes(json_responder(df))


def html_responder(df):
    return df.to_html(index=False).encode("utf-8")


def parquetpyarrow_reponder(df):
    return df.to_parquet(index=False, engine="pyarrow")


def parquetfastparquet_responder(df):
    # the fastparquet engine doesn't like to write to a buffer
    # it can do it via the open_with function being set appropriately
    # however it automatically calls the close method and wipes the buffer
    # so just overwrite that attribute on this instance to not do that

    # protected by an importorskip in the respective test
    import fsspec

    df.to_parquet(
        "memory://fastparquet_user_agent.parquet",
        index=False,
        engine="fastparquet",
        compression=None,
    )
    with fsspec.open("memory://fastparquet_user_agent.parquet", "rb") as f:
        return f.read()


def pickle_respnder(df):
    with BytesIO() as bio:
        df.to_pickle(bio)
        return bio.getvalue()


def stata_responder(df):
    with BytesIO() as bio:
        df.to_stata(bio, write_index=False)
        return bio.getvalue()


@pytest.mark.parametrize(
    "responder, read_method",
    [
        (csv_responder, pd.read_csv),
        (json_responder, pd.read_json),
        (
            html_responder,
            lambda *args, **kwargs: pd.read_html(*args, **kwargs)[0],
        ),
        pytest.param(
            parquetpyarrow_reponder,
            partial(pd.read_parquet, engine="pyarrow"),
            marks=td.skip_if_no("pyarrow"),
        ),
        pytest.param(
            parquetfastparquet_responder,
            partial(pd.read_parquet, engine="fastparquet"),
            # TODO(ArrayManager) fastparquet
            marks=[
                td.skip_if_no("fastparquet"),
                td.skip_if_no("fsspec"),
                td.skip_array_manager_not_yet_implemented,
                pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string"),
            ],
        ),
        (pickle_respnder, pd.read_pickle),
        (stata_responder, pd.read_stata),
        (gz_csv_responder, pd.read_csv),
        (gz_json_responder, pd.read_json),
    ],
)
@pytest.mark.parametrize(
    "storage_options",
    [
        None,
        {"User-Agent": "foo"},
        {"User-Agent": "foo", "Auth": "bar"},
    ],
)
def test_request_headers(responder, read_method, httpserver, storage_options):
    expected = pd.DataFrame({"a": ["b"]})
    default_headers = ["Accept-Encoding", "Host", "Connection", "User-Agent"]
    if "gz" in responder.__name__:
        extra = {"Content-Encoding": "gzip"}
        if storage_options is None:
            storage_options = extra
        else:
            storage_options |= extra
    else:
        extra = None
    expected_headers = set(default_headers).union(
        storage_options.keys() if storage_options else []
    )
    httpserver.serve_content(content=responder(expected), headers=extra)
    result = read_method(httpserver.url, storage_options=storage_options)
    tm.assert_frame_equal(result, expected)

    request_headers = dict(httpserver.requests[0].headers)
    for header in expected_headers:
        exp = request_headers.pop(header)
        if storage_options and header in storage_options:
            assert exp == storage_options[header]
    # No extra headers added
    assert not request_headers


@pytest.mark.parametrize(
    "engine",
    [
        "pyarrow",
        "fastparquet",
    ],
)
def test_to_parquet_to_disk_with_storage_options(engine):
    headers = {
        "User-Agent": "custom",
        "Auth": "other_custom",
    }

    pytest.importorskip(engine)

    true_df = pd.DataFrame({"column_name": ["column_value"]})
    msg = (
        "storage_options passed with file object or non-fsspec file path|"
        "storage_options passed with buffer, or non-supported URL"
    )
    with pytest.raises(ValueError, match=msg):
        true_df.to_parquet("/tmp/junk.parquet", storage_options=headers, engine=engine)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_series.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm


class TestSeriesConcat:
    def test_concat_series(self):
        ts = Series(
            np.arange(20, dtype=np.float64),
            index=date_range("2020-01-01", periods=20),
            name="foo",
        )
        ts.name = "foo"

        pieces = [ts[:5], ts[5:15], ts[15:]]

        result = concat(pieces)
        tm.assert_series_equal(result, ts)
        assert result.name == ts.name

        result = concat(pieces, keys=[0, 1, 2])
        expected = ts.copy()

        ts.index = DatetimeIndex(np.array(ts.index.values, dtype="M8[ns]"))

        exp_codes = [np.repeat([0, 1, 2], [len(x) for x in pieces]), np.arange(len(ts))]
        exp_index = MultiIndex(levels=[[0, 1, 2], ts.index], codes=exp_codes)
        expected.index = exp_index
        tm.assert_series_equal(result, expected)

    def test_concat_empty_and_non_empty_series_regression(self):
        # GH 18187 regression test
        s1 = Series([1])
        s2 = Series([], dtype=object)

        expected = s1
        msg = "The behavior of array concatenation with empty entries is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = concat([s1, s2])
        tm.assert_series_equal(result, expected)

    def test_concat_series_axis1(self):
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )

        pieces = [ts[:-2], ts[2:], ts[2:-2]]

        result = concat(pieces, axis=1)
        expected = DataFrame(pieces).T
        tm.assert_frame_equal(result, expected)

        result = concat(pieces, keys=["A", "B", "C"], axis=1)
        expected = DataFrame(pieces, index=["A", "B", "C"]).T
        tm.assert_frame_equal(result, expected)

    def test_concat_series_axis1_preserves_series_names(self):
        # preserve series names, #2489
        s = Series(np.random.default_rng(2).standard_normal(5), name="A")
        s2 = Series(np.random.default_rng(2).standard_normal(5), name="B")

        result = concat([s, s2], axis=1)
        expected = DataFrame({"A": s, "B": s2})
        tm.assert_frame_equal(result, expected)

        s2.name = None
        result = concat([s, s2], axis=1)
        tm.assert_index_equal(result.columns, Index(["A", 0], dtype="object"))

    def test_concat_series_axis1_with_reindex(self, sort):
        # must reindex, #2603
        s = Series(
            np.random.default_rng(2).standard_normal(3), index=["c", "a", "b"], name="A"
        )
        s2 = Series(
            np.random.default_rng(2).standard_normal(4),
            index=["d", "a", "b", "c"],
            name="B",
        )
        result = concat([s, s2], axis=1, sort=sort)
        expected = DataFrame({"A": s, "B": s2}, index=["c", "a", "b", "d"])
        if sort:
            expected = expected.sort_index()
        tm.assert_frame_equal(result, expected)

    def test_concat_series_axis1_names_applied(self):
        # ensure names argument is not ignored on axis=1, #23490
        s = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        result = concat([s, s2], axis=1, keys=["a", "b"], names=["A"])
        expected = DataFrame(
            [[1, 4], [2, 5], [3, 6]], columns=Index(["a", "b"], name="A")
        )
        tm.assert_frame_equal(result, expected)

        result = concat([s, s2], axis=1, keys=[("a", 1), ("b", 2)], names=["A", "B"])
        expected = DataFrame(
            [[1, 4], [2, 5], [3, 6]],
            columns=MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["A", "B"]),
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_series_axis1_same_names_ignore_index(self):
        dates = date_range("01-Jan-2013", "01-Jan-2014", freq="MS")[0:-1]
        s1 = Series(
            np.random.default_rng(2).standard_normal(len(dates)),
            index=dates,
            name="value",
        )
        s2 = Series(
            np.random.default_rng(2).standard_normal(len(dates)),
            index=dates,
            name="value",
        )

        result = concat([s1, s2], axis=1, ignore_index=True)
        expected = Index(range(2))

        tm.assert_index_equal(result.columns, expected, exact=True)

    @pytest.mark.parametrize(
        "s1name,s2name", [(np.int64(190), (43, 0)), (190, (43, 0))]
    )
    def test_concat_series_name_npscalar_tuple(self, s1name, s2name):
        # GH21015
        s1 = Series({"a": 1, "b": 2}, name=s1name)
        s2 = Series({"c": 5, "d": 6}, name=s2name)
        result = concat([s1, s2])
        expected = Series({"a": 1, "b": 2, "c": 5, "d": 6})
        tm.assert_series_equal(result, expected)

    def test_concat_series_partial_columns_names(self):
        # GH10698
        named_series = Series([1, 2], name="foo")
        unnamed_series1 = Series([1, 2])
        unnamed_series2 = Series([4, 5])

        result = concat([named_series, unnamed_series1, unnamed_series2], axis=1)
        expected = DataFrame(
            {"foo": [1, 2], 0: [1, 2], 1: [4, 5]}, columns=["foo", 0, 1]
        )
        tm.assert_frame_equal(result, expected)

        result = concat(
            [named_series, unnamed_series1, unnamed_series2],
            axis=1,
            keys=["red", "blue", "yellow"],
        )
        expected = DataFrame(
            {"red": [1, 2], "blue": [1, 2], "yellow": [4, 5]},
            columns=["red", "blue", "yellow"],
        )
        tm.assert_frame_equal(result, expected)

        result = concat(
            [named_series, unnamed_series1, unnamed_series2], axis=1, ignore_index=True
        )
        expected = DataFrame({0: [1, 2], 1: [1, 2], 2: [4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_concat_series_length_one_reversed(self, frame_or_series):
        # GH39401
        obj = frame_or_series([100])
        result = concat([obj.iloc[::-1]])
        tm.assert_equal(result, obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_explode.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_basic():
    s = pd.Series([[0, 1, 2], np.nan, [], (3, 4)], index=list("abcd"), name="foo")
    result = s.explode()
    expected = pd.Series(
        [0, 1, 2, np.nan, np.nan, 3, 4], index=list("aaabcdd"), dtype=object, name="foo"
    )
    tm.assert_series_equal(result, expected)


def test_mixed_type():
    s = pd.Series(
        [[0, 1, 2], np.nan, None, np.array([]), pd.Series(["a", "b"])], name="foo"
    )
    result = s.explode()
    expected = pd.Series(
        [0, 1, 2, np.nan, None, np.nan, "a", "b"],
        index=[0, 0, 0, 1, 2, 3, 4, 4],
        dtype=object,
        name="foo",
    )
    tm.assert_series_equal(result, expected)


def test_empty():
    s = pd.Series(dtype=object)
    result = s.explode()
    expected = s.copy()
    tm.assert_series_equal(result, expected)


def test_nested_lists():
    s = pd.Series([[[1, 2, 3]], [1, 2], 1])
    result = s.explode()
    expected = pd.Series([[1, 2, 3], 1, 2, 1], index=[0, 1, 1, 2])
    tm.assert_series_equal(result, expected)


def test_multi_index():
    s = pd.Series(
        [[0, 1, 2], np.nan, [], (3, 4)],
        name="foo",
        index=pd.MultiIndex.from_product([list("ab"), range(2)], names=["foo", "bar"]),
    )
    result = s.explode()
    index = pd.MultiIndex.from_tuples(
        [("a", 0), ("a", 0), ("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 1)],
        names=["foo", "bar"],
    )
    expected = pd.Series(
        [0, 1, 2, np.nan, np.nan, 3, 4], index=index, dtype=object, name="foo"
    )
    tm.assert_series_equal(result, expected)


def test_large():
    s = pd.Series([range(256)]).explode()
    result = s.explode()
    tm.assert_series_equal(result, s)


def test_invert_array():
    df = pd.DataFrame({"a": pd.date_range("20190101", periods=3, tz="UTC")})

    listify = df.apply(lambda x: x.array, axis=1)
    result = listify.explode()
    tm.assert_series_equal(result, df["a"].rename())


@pytest.mark.parametrize(
    "s", [pd.Series([1, 2, 3]), pd.Series(pd.date_range("2019", periods=3, tz="UTC"))]
)
def test_non_object_dtype(s):
    result = s.explode()
    tm.assert_series_equal(result, s)


def test_typical_usecase():
    df = pd.DataFrame(
        [{"var1": "a,b,c", "var2": 1}, {"var1": "d,e,f", "var2": 2}],
        columns=["var1", "var2"],
    )
    exploded = df.var1.str.split(",").explode()
    result = df[["var2"]].join(exploded)
    expected = pd.DataFrame(
        {"var2": [1, 1, 1, 2, 2, 2], "var1": list("abcdef")},
        columns=["var2", "var1"],
        index=[0, 0, 0, 1, 1, 1],
    )
    tm.assert_frame_equal(result, expected)


def test_nested_EA():
    # a nested EA array
    s = pd.Series(
        [
            pd.date_range("20170101", periods=3, tz="UTC"),
            pd.date_range("20170104", periods=3, tz="UTC"),
        ]
    )
    result = s.explode()
    expected = pd.Series(
        pd.date_range("20170101", periods=6, tz="UTC"), index=[0, 0, 0, 1, 1, 1]
    )
    tm.assert_series_equal(result, expected)


def test_duplicate_index():
    # GH 28005
    s = pd.Series([[1, 2], [3, 4]], index=[0, 0])
    result = s.explode()
    expected = pd.Series([1, 2, 3, 4], index=[0, 0, 0, 0], dtype=object)
    tm.assert_series_equal(result, expected)


def test_ignore_index():
    # GH 34932
    s = pd.Series([[1, 2], [3, 4]])
    result = s.explode(ignore_index=True)
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)


def test_explode_sets():
    # https://github.com/pandas-dev/pandas/issues/35614
    s = pd.Series([{"a", "b", "c"}], index=[1])
    result = s.explode().sort_values()
    expected = pd.Series(["a", "b", "c"], index=[1, 1, 1])
    tm.assert_series_equal(result, expected)


def test_explode_scalars_can_ignore_index():
    # https://github.com/pandas-dev/pandas/issues/40487
    s = pd.Series([1, 2, 3], index=["a", "b", "c"])
    result = s.explode(ignore_index=True)
    expected = pd.Series([1, 2, 3])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ignore_index", [True, False])
def test_explode_pyarrow_list_type(ignore_index):
    # GH 53602
    pa = pytest.importorskip("pyarrow")

    data = [
        [None, None],
        [1],
        [],
        [2, 3],
        None,
    ]
    ser = pd.Series(data, dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    result = ser.explode(ignore_index=ignore_index)
    expected = pd.Series(
        data=[None, None, 1, None, 2, 3, None],
        index=None if ignore_index else [0, 0, 1, 2, 3, 3, 4],
        dtype=pd.ArrowDtype(pa.int64()),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ignore_index", [True, False])
def test_explode_pyarrow_non_list_type(ignore_index):
    pa = pytest.importorskip("pyarrow")
    data = [1, 2, 3]
    ser = pd.Series(data, dtype=pd.ArrowDtype(pa.int64()))
    result = ser.explode(ignore_index=ignore_index)
    expected = pd.Series([1, 2, 3], dtype="int64[pyarrow]", index=[0, 1, 2])
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_orthopolys.py ===
"""Tests for efficient functions for generating orthogonal polynomials. """

from sympy.core.numbers import Rational as Q
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises

from sympy.polys.orthopolys import (
    jacobi_poly,
    gegenbauer_poly,
    chebyshevt_poly,
    chebyshevu_poly,
    hermite_poly,
    hermite_prob_poly,
    legendre_poly,
    laguerre_poly,
    spherical_bessel_fn,
)

from sympy.abc import x, a, b


def test_jacobi_poly():
    raises(ValueError, lambda: jacobi_poly(-1, a, b, x))

    assert jacobi_poly(1, a, b, x, polys=True) == Poly(
        (a/2 + b/2 + 1)*x + a/2 - b/2, x, domain='ZZ(a,b)')

    assert jacobi_poly(0, a, b, x) == 1
    assert jacobi_poly(1, a, b, x) == a/2 - b/2 + x*(a/2 + b/2 + 1)
    assert jacobi_poly(2, a, b, x) == (a**2/8 - a*b/4 - a/8 + b**2/8 - b/8 +
                                       x**2*(a**2/8 + a*b/4 + a*Q(7, 8) + b**2/8 +
                                             b*Q(7, 8) + Q(3, 2)) + x*(a**2/4 +
                                            a*Q(3, 4) - b**2/4 - b*Q(3, 4)) - S.Half)

    assert jacobi_poly(1, a, b, polys=True) == Poly(
        (a/2 + b/2 + 1)*x + a/2 - b/2, x, domain='ZZ(a,b)')


def test_gegenbauer_poly():
    raises(ValueError, lambda: gegenbauer_poly(-1, a, x))

    assert gegenbauer_poly(
        1, a, x, polys=True) == Poly(2*a*x, x, domain='ZZ(a)')

    assert gegenbauer_poly(0, a, x) == 1
    assert gegenbauer_poly(1, a, x) == 2*a*x
    assert gegenbauer_poly(2, a, x) == -a + x**2*(2*a**2 + 2*a)
    assert gegenbauer_poly(
        3, a, x) == x**3*(4*a**3/3 + 4*a**2 + a*Q(8, 3)) + x*(-2*a**2 - 2*a)

    assert gegenbauer_poly(1, S.Half).dummy_eq(x)
    assert gegenbauer_poly(1, a, polys=True) == Poly(2*a*x, x, domain='ZZ(a)')


def test_chebyshevt_poly():
    raises(ValueError, lambda: chebyshevt_poly(-1, x))

    assert chebyshevt_poly(1, x, polys=True) == Poly(x)

    assert chebyshevt_poly(0, x) == 1
    assert chebyshevt_poly(1, x) == x
    assert chebyshevt_poly(2, x) == 2*x**2 - 1
    assert chebyshevt_poly(3, x) == 4*x**3 - 3*x
    assert chebyshevt_poly(4, x) == 8*x**4 - 8*x**2 + 1
    assert chebyshevt_poly(5, x) == 16*x**5 - 20*x**3 + 5*x
    assert chebyshevt_poly(6, x) == 32*x**6 - 48*x**4 + 18*x**2 - 1
    assert chebyshevt_poly(75, x) == (2*chebyshevt_poly(37, x)*chebyshevt_poly(38, x) - x).expand()
    assert chebyshevt_poly(100, x) == (2*chebyshevt_poly(50, x)**2 - 1).expand()

    assert chebyshevt_poly(1).dummy_eq(x)
    assert chebyshevt_poly(1, polys=True) == Poly(x)


def test_chebyshevu_poly():
    raises(ValueError, lambda: chebyshevu_poly(-1, x))

    assert chebyshevu_poly(1, x, polys=True) == Poly(2*x)

    assert chebyshevu_poly(0, x) == 1
    assert chebyshevu_poly(1, x) == 2*x
    assert chebyshevu_poly(2, x) == 4*x**2 - 1
    assert chebyshevu_poly(3, x) == 8*x**3 - 4*x
    assert chebyshevu_poly(4, x) == 16*x**4 - 12*x**2 + 1
    assert chebyshevu_poly(5, x) == 32*x**5 - 32*x**3 + 6*x
    assert chebyshevu_poly(6, x) == 64*x**6 - 80*x**4 + 24*x**2 - 1

    assert chebyshevu_poly(1).dummy_eq(2*x)
    assert chebyshevu_poly(1, polys=True) == Poly(2*x)


def test_hermite_poly():
    raises(ValueError, lambda: hermite_poly(-1, x))

    assert hermite_poly(1, x, polys=True) == Poly(2*x)

    assert hermite_poly(0, x) == 1
    assert hermite_poly(1, x) == 2*x
    assert hermite_poly(2, x) == 4*x**2 - 2
    assert hermite_poly(3, x) == 8*x**3 - 12*x
    assert hermite_poly(4, x) == 16*x**4 - 48*x**2 + 12
    assert hermite_poly(5, x) == 32*x**5 - 160*x**3 + 120*x
    assert hermite_poly(6, x) == 64*x**6 - 480*x**4 + 720*x**2 - 120

    assert hermite_poly(1).dummy_eq(2*x)
    assert hermite_poly(1, polys=True) == Poly(2*x)


def test_hermite_prob_poly():
    raises(ValueError, lambda: hermite_prob_poly(-1, x))

    assert hermite_prob_poly(1, x, polys=True) == Poly(x)

    assert hermite_prob_poly(0, x) == 1
    assert hermite_prob_poly(1, x) == x
    assert hermite_prob_poly(2, x) == x**2 - 1
    assert hermite_prob_poly(3, x) == x**3 - 3*x
    assert hermite_prob_poly(4, x) == x**4 - 6*x**2 + 3
    assert hermite_prob_poly(5, x) == x**5 - 10*x**3 + 15*x
    assert hermite_prob_poly(6, x) == x**6 - 15*x**4 + 45*x**2 - 15

    assert hermite_prob_poly(1).dummy_eq(x)
    assert hermite_prob_poly(1, polys=True) == Poly(x)


def test_legendre_poly():
    raises(ValueError, lambda: legendre_poly(-1, x))

    assert legendre_poly(1, x, polys=True) == Poly(x, domain='QQ')

    assert legendre_poly(0, x) == 1
    assert legendre_poly(1, x) == x
    assert legendre_poly(2, x) == Q(3, 2)*x**2 - Q(1, 2)
    assert legendre_poly(3, x) == Q(5, 2)*x**3 - Q(3, 2)*x
    assert legendre_poly(4, x) == Q(35, 8)*x**4 - Q(30, 8)*x**2 + Q(3, 8)
    assert legendre_poly(5, x) == Q(63, 8)*x**5 - Q(70, 8)*x**3 + Q(15, 8)*x
    assert legendre_poly(6, x) == Q(
        231, 16)*x**6 - Q(315, 16)*x**4 + Q(105, 16)*x**2 - Q(5, 16)

    assert legendre_poly(1).dummy_eq(x)
    assert legendre_poly(1, polys=True) == Poly(x)


def test_laguerre_poly():
    raises(ValueError, lambda: laguerre_poly(-1, x))

    assert laguerre_poly(1, x, polys=True) == Poly(-x + 1, domain='QQ')

    assert laguerre_poly(0, x) == 1
    assert laguerre_poly(1, x) == -x + 1
    assert laguerre_poly(2, x) == Q(1, 2)*x**2 - Q(4, 2)*x + 1
    assert laguerre_poly(3, x) == -Q(1, 6)*x**3 + Q(9, 6)*x**2 - Q(18, 6)*x + 1
    assert laguerre_poly(4, x) == Q(
        1, 24)*x**4 - Q(16, 24)*x**3 + Q(72, 24)*x**2 - Q(96, 24)*x + 1
    assert laguerre_poly(5, x) == -Q(1, 120)*x**5 + Q(25, 120)*x**4 - Q(
        200, 120)*x**3 + Q(600, 120)*x**2 - Q(600, 120)*x + 1
    assert laguerre_poly(6, x) == Q(1, 720)*x**6 - Q(36, 720)*x**5 + Q(450, 720)*x**4 - Q(2400, 720)*x**3 + Q(5400, 720)*x**2 - Q(4320, 720)*x + 1

    assert laguerre_poly(0, x, a) == 1
    assert laguerre_poly(1, x, a) == -x + a + 1
    assert laguerre_poly(2, x, a) == x**2/2 + (-a - 2)*x + a**2/2 + a*Q(3, 2) + 1
    assert laguerre_poly(3, x, a) == -x**3/6 + (a/2 + Q(
        3)/2)*x**2 + (-a**2/2 - a*Q(5, 2) - 3)*x + a**3/6 + a**2 + a*Q(11, 6) + 1

    assert laguerre_poly(1).dummy_eq(-x + 1)
    assert laguerre_poly(1, polys=True) == Poly(-x + 1)


def test_spherical_bessel_fn():
    x, z = symbols("x z")
    assert spherical_bessel_fn(1, z) == 1/z**2
    assert spherical_bessel_fn(2, z) == -1/z + 3/z**3
    assert spherical_bessel_fn(3, z) == -6/z**2 + 15/z**4
    assert spherical_bessel_fn(4, z) == 1/z - 45/z**3 + 105/z**5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_symbolic_probability.py ===
from sympy.concrete.summations import Sum
from sympy.core.mul import Mul
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.core.expr import unchanged
from sympy.stats import (Normal, Poisson, variance, Covariance, Variance,
                         Probability, Expectation, Moment, CentralMoment)
from sympy.stats.rv import probability, expectation


def test_literal_probability():
    X = Normal('X', 2, 3)
    Y = Normal('Y', 3, 4)
    Z = Poisson('Z', 4)
    W = Poisson('W', 3)
    x = symbols('x', real=True)
    y, w, z = symbols('y, w, z')

    assert Probability(X > 0).evaluate_integral() == probability(X > 0)
    assert Probability(X > x).evaluate_integral() == probability(X > x)
    assert Probability(X > 0).rewrite(Integral).doit() == probability(X > 0)
    assert Probability(X > x).rewrite(Integral).doit() == probability(X > x)

    assert Expectation(X).evaluate_integral() == expectation(X)
    assert Expectation(X).rewrite(Integral).doit() == expectation(X)
    assert Expectation(X**2).evaluate_integral() == expectation(X**2)
    assert Expectation(x*X).args == (x*X,)
    assert Expectation(x*X).expand() == x*Expectation(X)
    assert Expectation(2*X + 3*Y + z*X*Y).expand() == 2*Expectation(X) + 3*Expectation(Y) + z*Expectation(X*Y)
    assert Expectation(2*X + 3*Y + z*X*Y).args == (2*X + 3*Y + z*X*Y,)
    assert Expectation(sin(X)) == Expectation(sin(X)).expand()
    assert Expectation(2*x*sin(X)*Y + y*X**2 + z*X*Y).expand() == 2*x*Expectation(sin(X)*Y) \
                            + y*Expectation(X**2) + z*Expectation(X*Y)
    assert Expectation(X + Y).expand() ==  Expectation(X) + Expectation(Y)
    assert Expectation((X + Y)*(X - Y)).expand() == Expectation(X**2) - Expectation(Y**2)
    assert Expectation((X + Y)*(X - Y)).expand().doit() == -12
    assert Expectation(X + Y, evaluate=True).doit() == 5
    assert Expectation(X + Expectation(Y)).doit() == 5
    assert Expectation(X + Expectation(Y)).doit(deep=False) == 2 + Expectation(Expectation(Y))
    assert Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False) == 2 \
                                + Expectation(Expectation(Y + Expectation(2*X)))
    assert Expectation(X + Expectation(Y + Expectation(2*X))).doit() == 9
    assert Expectation(Expectation(2*X)).doit() == 4
    assert Expectation(Expectation(2*X)).doit(deep=False) == Expectation(2*X)
    assert Expectation(4*Expectation(2*X)).doit(deep=False) == 4*Expectation(2*X)
    assert Expectation((X + Y)**3).expand() == 3*Expectation(X*Y**2) +\
                3*Expectation(X**2*Y) + Expectation(X**3) + Expectation(Y**3)
    assert Expectation((X - Y)**3).expand() == 3*Expectation(X*Y**2) -\
                3*Expectation(X**2*Y) + Expectation(X**3) - Expectation(Y**3)
    assert Expectation((X - Y)**2).expand() == -2*Expectation(X*Y) +\
                Expectation(X**2) + Expectation(Y**2)

    assert Variance(w).args == (w,)
    assert Variance(w).expand() == 0
    assert Variance(X).evaluate_integral() == Variance(X).rewrite(Integral).doit() == variance(X)
    assert Variance(X + z).args == (X + z,)
    assert Variance(X + z).expand() == Variance(X)
    assert Variance(X*Y).args == (Mul(X, Y),)
    assert type(Variance(X*Y)) == Variance
    assert Variance(z*X).expand() == z**2*Variance(X)
    assert Variance(X + Y).expand() == Variance(X) + Variance(Y) + 2*Covariance(X, Y)
    assert Variance(X + Y + Z + W).expand() == (Variance(X) + Variance(Y) + Variance(Z) + Variance(W) +
                                       2 * Covariance(X, Y) + 2 * Covariance(X, Z) + 2 * Covariance(X, W) +
                                       2 * Covariance(Y, Z) + 2 * Covariance(Y, W) + 2 * Covariance(W, Z))
    assert Variance(X**2).evaluate_integral() == variance(X**2)
    assert unchanged(Variance, X**2)
    assert Variance(x*X**2).expand() == x**2*Variance(X**2)
    assert Variance(sin(X)).args == (sin(X),)
    assert Variance(sin(X)).expand() == Variance(sin(X))
    assert Variance(x*sin(X)).expand() == x**2*Variance(sin(X))

    assert Covariance(w, z).args == (w, z)
    assert Covariance(w, z).expand() == 0
    assert Covariance(X, w).expand() == 0
    assert Covariance(w, X).expand() == 0
    assert Covariance(X, Y).args == (X, Y)
    assert type(Covariance(X, Y)) == Covariance
    assert Covariance(z*X + 3, Y).expand() == z*Covariance(X, Y)
    assert Covariance(X, X).args == (X, X)
    assert Covariance(X, X).expand() == Variance(X)
    assert Covariance(z*X + 3, w*Y + 4).expand() == w*z*Covariance(X,Y)
    assert Covariance(X, Y) == Covariance(Y, X)
    assert Covariance(X + Y, Z + W).expand() == Covariance(W, X) + Covariance(W, Y) + Covariance(X, Z) + Covariance(Y, Z)
    assert Covariance(x*X + y*Y, z*Z + w*W).expand() == (x*w*Covariance(W, X) + w*y*Covariance(W, Y) +
                                                x*z*Covariance(X, Z) + y*z*Covariance(Y, Z))
    assert Covariance(x*X**2 + y*sin(Y), z*Y*Z**2 + w*W).expand() == (w*x*Covariance(W, X**2) + w*y*Covariance(sin(Y), W) +
                                                        x*z*Covariance(Y*Z**2, X**2) + y*z*Covariance(Y*Z**2, sin(Y)))
    assert Covariance(X, X**2).expand() == Covariance(X, X**2)
    assert Covariance(X, sin(X)).expand() == Covariance(sin(X), X)
    assert Covariance(X**2, sin(X)*Y).expand() == Covariance(sin(X)*Y, X**2)
    assert Covariance(w, X).evaluate_integral() == 0


def test_probability_rewrite():
    X = Normal('X', 2, 3)
    Y = Normal('Y', 3, 4)
    Z = Poisson('Z', 4)
    W = Poisson('W', 3)
    x, y, w, z = symbols('x, y, w, z')

    assert Variance(w).rewrite(Expectation) == 0
    assert Variance(X).rewrite(Expectation) == Expectation(X ** 2) - Expectation(X) ** 2
    assert Variance(X, condition=Y).rewrite(Expectation) == Expectation(X ** 2, Y) - Expectation(X, Y) ** 2
    assert Variance(X, Y) != Expectation(X**2) - Expectation(X)**2
    assert Variance(X + z).rewrite(Expectation) == Expectation((X + z) ** 2) - Expectation(X + z) ** 2
    assert Variance(X * Y).rewrite(Expectation) == Expectation(X ** 2 * Y ** 2) - Expectation(X * Y) ** 2

    assert Covariance(w, X).rewrite(Expectation) == -w*Expectation(X) + Expectation(w*X)
    assert Covariance(X, Y).rewrite(Expectation) == Expectation(X*Y) - Expectation(X)*Expectation(Y)
    assert Covariance(X, Y, condition=W).rewrite(Expectation) == Expectation(X * Y, W) - Expectation(X, W) * Expectation(Y, W)

    w, x, z = symbols("W, x, z")
    px = Probability(Eq(X, x))
    pz = Probability(Eq(Z, z))

    assert Expectation(X).rewrite(Probability) == Integral(x*px, (x, -oo, oo))
    assert Expectation(Z).rewrite(Probability) == Sum(z*pz, (z, 0, oo))
    assert Variance(X).rewrite(Probability) == Integral(x**2*px, (x, -oo, oo)) - Integral(x*px, (x, -oo, oo))**2
    assert Variance(Z).rewrite(Probability) == Sum(z**2*pz, (z, 0, oo)) - Sum(z*pz, (z, 0, oo))**2
    assert Covariance(w, X).rewrite(Probability) == \
           -w*Integral(x*Probability(Eq(X, x)), (x, -oo, oo)) + Integral(w*x*Probability(Eq(X, x)), (x, -oo, oo))

    # To test rewrite as sum function
    assert Variance(X).rewrite(Sum) == Variance(X).rewrite(Integral)
    assert Expectation(X).rewrite(Sum) == Expectation(X).rewrite(Integral)

    assert Covariance(w, X).rewrite(Sum) == 0

    assert Covariance(w, X).rewrite(Integral) == 0

    assert Variance(X, condition=Y).rewrite(Probability) == Integral(x**2*Probability(Eq(X, x), Y), (x, -oo, oo)) - \
                                                            Integral(x*Probability(Eq(X, x), Y), (x, -oo, oo))**2


def test_symbolic_Moment():
    mu = symbols('mu', real=True)
    sigma = symbols('sigma', positive=True)
    x = symbols('x')
    X = Normal('X', mu, sigma)
    M = Moment(X, 4, 2)
    assert M.rewrite(Expectation) == Expectation((X - 2)**4)
    assert M.rewrite(Probability) == Integral((x - 2)**4*Probability(Eq(X, x)),
                                    (x, -oo, oo))
    k = Dummy('k')
    expri = Integral(sqrt(2)*(k - 2)**4*exp(-(k - \
                mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo))
    assert M.rewrite(Integral).dummy_eq(expri)
    assert M.doit() == (mu**4 - 8*mu**3 + 6*mu**2*sigma**2 + \
                24*mu**2 - 24*mu*sigma**2 - 32*mu + 3*sigma**4 + 24*sigma**2 + 16)
    M = Moment(2, 5)
    assert M.doit() == 2**5


def test_symbolic_CentralMoment():
    mu = symbols('mu', real=True)
    sigma = symbols('sigma', positive=True)
    x = symbols('x')
    X = Normal('X', mu, sigma)
    CM = CentralMoment(X, 6)
    assert CM.rewrite(Expectation) == Expectation((X - Expectation(X))**6)
    assert CM.rewrite(Probability) == Integral((x - Integral(x*Probability(True),
                    (x, -oo, oo)))**6*Probability(Eq(X, x)), (x, -oo, oo))
    k = Dummy('k')
    expri = Integral(sqrt(2)*(k - Integral(sqrt(2)*k*exp(-(k - \
        mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo)))**6*exp(-(k - \
        mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo))
    assert CM.rewrite(Integral).dummy_eq(expri)
    assert CM.doit().simplify() == 15*sigma**6
    CM = Moment(5, 5)
    assert CM.doit() == 5**5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\ma.py ===
from typing import Any, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
from numpy._typing import _Shape

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
MaskedArray: TypeAlias = np.ma.MaskedArray[_Shape, np.dtype[_ScalarT]]

MAR_b: MaskedArray[np.bool] = np.ma.MaskedArray([True])
MAR_u: MaskedArray[np.uint32] = np.ma.MaskedArray([1], dtype=np.uint32)
MAR_i: MaskedArray[np.int64] = np.ma.MaskedArray([1])
MAR_f: MaskedArray[np.float64] = np.ma.MaskedArray([1.0])
MAR_c: MaskedArray[np.complex128] = np.ma.MaskedArray([1j])
MAR_td64: MaskedArray[np.timedelta64] = np.ma.MaskedArray([np.timedelta64(1, "D")])
MAR_M_dt64: MaskedArray[np.datetime64] = np.ma.MaskedArray([np.datetime64(1, "D")])
MAR_S: MaskedArray[np.bytes_] = np.ma.MaskedArray([b'foo'], dtype=np.bytes_)
MAR_U: MaskedArray[np.str_] = np.ma.MaskedArray(['foo'], dtype=np.str_)
MAR_T = cast(np.ma.MaskedArray[Any, np.dtypes.StringDType],
             np.ma.MaskedArray(["a"], dtype="T"))

AR_b: npt.NDArray[np.bool] = np.array([True, False, True])

AR_LIKE_b = [True]
AR_LIKE_u = [np.uint32(1)]
AR_LIKE_i = [1]
AR_LIKE_f = [1.0]
AR_LIKE_c = [1j]
AR_LIKE_m = [np.timedelta64(1, "D")]
AR_LIKE_M = [np.datetime64(1, "D")]

MAR_f.mask = AR_b
MAR_f.mask = np.False_

# Inplace addition

MAR_b += AR_LIKE_b

MAR_u += AR_LIKE_b
MAR_u += AR_LIKE_u

MAR_i += AR_LIKE_b
MAR_i += 2
MAR_i += AR_LIKE_i

MAR_f += AR_LIKE_b
MAR_f += 2
MAR_f += AR_LIKE_u
MAR_f += AR_LIKE_i
MAR_f += AR_LIKE_f

MAR_c += AR_LIKE_b
MAR_c += AR_LIKE_u
MAR_c += AR_LIKE_i
MAR_c += AR_LIKE_f
MAR_c += AR_LIKE_c

MAR_td64 += AR_LIKE_b
MAR_td64 += AR_LIKE_u
MAR_td64 += AR_LIKE_i
MAR_td64 += AR_LIKE_m
MAR_M_dt64 += AR_LIKE_b
MAR_M_dt64 += AR_LIKE_u
MAR_M_dt64 += AR_LIKE_i
MAR_M_dt64 += AR_LIKE_m

MAR_S += b'snakes'
MAR_U += 'snakes'
MAR_T += 'snakes'

# Inplace subtraction

MAR_u -= AR_LIKE_b
MAR_u -= AR_LIKE_u

MAR_i -= AR_LIKE_b
MAR_i -= AR_LIKE_i

MAR_f -= AR_LIKE_b
MAR_f -= AR_LIKE_u
MAR_f -= AR_LIKE_i
MAR_f -= AR_LIKE_f

MAR_c -= AR_LIKE_b
MAR_c -= AR_LIKE_u
MAR_c -= AR_LIKE_i
MAR_c -= AR_LIKE_f
MAR_c -= AR_LIKE_c

MAR_td64 -= AR_LIKE_b
MAR_td64 -= AR_LIKE_u
MAR_td64 -= AR_LIKE_i
MAR_td64 -= AR_LIKE_m
MAR_M_dt64 -= AR_LIKE_b
MAR_M_dt64 -= AR_LIKE_u
MAR_M_dt64 -= AR_LIKE_i
MAR_M_dt64 -= AR_LIKE_m

# Inplace floor division

MAR_f //= AR_LIKE_b
MAR_f //= 2
MAR_f //= AR_LIKE_u
MAR_f //= AR_LIKE_i
MAR_f //= AR_LIKE_f

MAR_td64 //= AR_LIKE_i

# Inplace true division

MAR_f /= AR_LIKE_b
MAR_f /= 2
MAR_f /= AR_LIKE_u
MAR_f /= AR_LIKE_i
MAR_f /= AR_LIKE_f

MAR_c /= AR_LIKE_b
MAR_c /= AR_LIKE_u
MAR_c /= AR_LIKE_i
MAR_c /= AR_LIKE_f
MAR_c /= AR_LIKE_c

MAR_td64 /= AR_LIKE_i

# Inplace multiplication

MAR_b *= AR_LIKE_b

MAR_u *= AR_LIKE_b
MAR_u *= AR_LIKE_u

MAR_i *= AR_LIKE_b
MAR_i *= 2
MAR_i *= AR_LIKE_i

MAR_f *= AR_LIKE_b
MAR_f *= 2
MAR_f *= AR_LIKE_u
MAR_f *= AR_LIKE_i
MAR_f *= AR_LIKE_f

MAR_c *= AR_LIKE_b
MAR_c *= AR_LIKE_u
MAR_c *= AR_LIKE_i
MAR_c *= AR_LIKE_f
MAR_c *= AR_LIKE_c

MAR_td64 *= AR_LIKE_b
MAR_td64 *= AR_LIKE_u
MAR_td64 *= AR_LIKE_i
MAR_td64 *= AR_LIKE_f

MAR_S *= 2
MAR_U *= 2
MAR_T *= 2

# Inplace power

MAR_u **= AR_LIKE_b
MAR_u **= AR_LIKE_u

MAR_i **= AR_LIKE_b
MAR_i **= AR_LIKE_i

MAR_f **= AR_LIKE_b
MAR_f **= AR_LIKE_u
MAR_f **= AR_LIKE_i
MAR_f **= AR_LIKE_f

MAR_c **= AR_LIKE_b
MAR_c **= AR_LIKE_u
MAR_c **= AR_LIKE_i
MAR_c **= AR_LIKE_f
MAR_c **= AR_LIKE_c

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_chained_assignment_deprecation.py ===
import numpy as np
import pytest

from pandas.compat import PY311
from pandas.errors import (
    ChainedAssignmentError,
    SettingWithCopyWarning,
)

from pandas import (
    DataFrame,
    option_context,
)
import pandas._testing as tm


def test_methods_iloc_warn(using_copy_on_write):
    if not using_copy_on_write:
        df = DataFrame({"a": [1, 2, 3], "b": 1})
        with tm.assert_cow_warning(match="A value"):
            df.iloc[:, 0].replace(1, 5, inplace=True)

        with tm.assert_cow_warning(match="A value"):
            df.iloc[:, 0].fillna(1, inplace=True)

        with tm.assert_cow_warning(match="A value"):
            df.iloc[:, 0].interpolate(inplace=True)

        with tm.assert_cow_warning(match="A value"):
            df.iloc[:, 0].ffill(inplace=True)

        with tm.assert_cow_warning(match="A value"):
            df.iloc[:, 0].bfill(inplace=True)


@pytest.mark.parametrize(
    "func, args",
    [
        ("replace", (4, 5)),
        ("fillna", (1,)),
        ("interpolate", ()),
        ("bfill", ()),
        ("ffill", ()),
    ],
)
def test_methods_iloc_getitem_item_cache(
    func, args, using_copy_on_write, warn_copy_on_write
):
    # ensure we don't incorrectly raise chained assignment warning because
    # of the item cache / iloc not setting the item cache
    df_orig = DataFrame({"a": [1, 2, 3], "b": 1})

    df = df_orig.copy()
    ser = df.iloc[:, 0]
    getattr(ser, func)(*args, inplace=True)

    # parent that holds item_cache is dead, so don't increase ref count
    df = df_orig.copy()
    ser = df.copy()["a"]
    getattr(ser, func)(*args, inplace=True)

    df = df_orig.copy()
    df["a"]  # populate the item_cache
    ser = df.iloc[:, 0]  # iloc creates a new object
    getattr(ser, func)(*args, inplace=True)

    df = df_orig.copy()
    df["a"]  # populate the item_cache
    ser = df["a"]
    getattr(ser, func)(*args, inplace=True)

    df = df_orig.copy()
    df["a"]  # populate the item_cache
    # TODO(CoW-warn) because of the usage of *args, this doesn't warn on Py3.11+
    if using_copy_on_write:
        with tm.raises_chained_assignment_error(not PY311):
            getattr(df["a"], func)(*args, inplace=True)
    else:
        with tm.assert_cow_warning(not PY311, match="A value"):
            getattr(df["a"], func)(*args, inplace=True)

    df = df_orig.copy()
    ser = df["a"]  # populate the item_cache and keep ref
    if using_copy_on_write:
        with tm.raises_chained_assignment_error(not PY311):
            getattr(df["a"], func)(*args, inplace=True)
    else:
        # ideally also warns on the default mode, but the ser' _cacher
        # messes up the refcount + even in warning mode this doesn't trigger
        # the warning of Py3.1+ (see above)
        with tm.assert_cow_warning(warn_copy_on_write and not PY311, match="A value"):
            getattr(df["a"], func)(*args, inplace=True)


def test_methods_iloc_getitem_item_cache_fillna(
    using_copy_on_write, warn_copy_on_write
):
    # ensure we don't incorrectly raise chained assignment warning because
    # of the item cache / iloc not setting the item cache
    df_orig = DataFrame({"a": [1, 2, 3], "b": 1})

    df = df_orig.copy()
    ser = df.iloc[:, 0]
    ser.fillna(1, inplace=True)

    # parent that holds item_cache is dead, so don't increase ref count
    df = df_orig.copy()
    ser = df.copy()["a"]
    ser.fillna(1, inplace=True)

    df = df_orig.copy()
    df["a"]  # populate the item_cache
    ser = df.iloc[:, 0]  # iloc creates a new object
    ser.fillna(1, inplace=True)

    df = df_orig.copy()
    df["a"]  # populate the item_cache
    ser = df["a"]
    ser.fillna(1, inplace=True)

    df = df_orig.copy()
    df["a"]  # populate the item_cache
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["a"].fillna(1, inplace=True)
    else:
        with tm.assert_cow_warning(match="A value"):
            df["a"].fillna(1, inplace=True)

    df = df_orig.copy()
    ser = df["a"]  # populate the item_cache and keep ref
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["a"].fillna(1, inplace=True)
    else:
        # TODO(CoW-warn) ideally also warns on the default mode, but the ser' _cacher
        # messes up the refcount
        with tm.assert_cow_warning(warn_copy_on_write, match="A value"):
            df["a"].fillna(1, inplace=True)


# TODO(CoW-warn) expand the cases
@pytest.mark.parametrize(
    "indexer", [0, [0, 1], slice(0, 2), np.array([True, False, True])]
)
def test_series_setitem(indexer, using_copy_on_write, warn_copy_on_write):
    # ensure we only get a single warning for those typical cases of chained
    # assignment
    df = DataFrame({"a": [1, 2, 3], "b": 1})

    # using custom check instead of tm.assert_produces_warning because that doesn't
    # fail if multiple warnings are raised
    with pytest.warns() as record:
        df["a"][indexer] = 0
    assert len(record) == 1
    if using_copy_on_write:
        assert record[0].category == ChainedAssignmentError
    else:
        assert record[0].category == FutureWarning
        assert "ChainedAssignmentError" in record[0].message.args[0]


@pytest.mark.filterwarnings("ignore::pandas.errors.SettingWithCopyWarning")
@pytest.mark.parametrize(
    "indexer", ["a", ["a", "b"], slice(0, 2), np.array([True, False, True])]
)
def test_frame_setitem(indexer, using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": 1})

    extra_warnings = () if using_copy_on_write else (SettingWithCopyWarning,)

    with option_context("chained_assignment", "warn"):
        with tm.raises_chained_assignment_error(extra_warnings=extra_warnings):
            df[0:3][indexer] = 10

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\groupby.py ===
import re

import pytest

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

import pandas as pd
import pandas._testing as tm


@pytest.mark.filterwarnings(
    "ignore:The default of observed=False is deprecated:FutureWarning"
)
class BaseGroupbyTests:
    """Groupby-specific tests."""

    def test_grouping_grouper(self, data_for_grouping):
        df = pd.DataFrame(
            {
                "A": pd.Series(
                    ["B", "B", None, None, "A", "A", "B", "C"], dtype=object
                ),
                "B": data_for_grouping,
            }
        )
        gr1 = df.groupby("A")._grouper.groupings[0]
        gr2 = df.groupby("B")._grouper.groupings[0]

        tm.assert_numpy_array_equal(gr1.grouping_vector, df.A.values)
        tm.assert_extension_array_equal(gr2.grouping_vector, data_for_grouping)

    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values, and the final entry has c==b
            #  (see data_for_grouping docstring)
            df = df.iloc[:-1]

        result = df.groupby("B", as_index=as_index).A.mean()
        _, uniques = pd.factorize(data_for_grouping, sort=True)

        exp_vals = [3.0, 1.0, 4.0]
        if is_bool:
            exp_vals = exp_vals[:-1]
        if as_index:
            index = pd.Index(uniques, name="B")
            expected = pd.Series(exp_vals, index=index, name="A")
            tm.assert_series_equal(result, expected)
        else:
            expected = pd.DataFrame({"B": uniques, "A": exp_vals})
            tm.assert_frame_equal(result, expected)

    def test_groupby_agg_extension(self, data_for_grouping):
        # GH#38980 groupby agg on extension type fails for non-numeric types
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        expected = df.iloc[[0, 2, 4, 7]]
        expected = expected.set_index("A")

        result = df.groupby("A").agg({"B": "first"})
        tm.assert_frame_equal(result, expected)

        result = df.groupby("A").agg("first")
        tm.assert_frame_equal(result, expected)

        result = df.groupby("A").first()
        tm.assert_frame_equal(result, expected)

    def test_groupby_extension_no_sort(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})

        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values, and the final entry has c==b
            #  (see data_for_grouping docstring)
            df = df.iloc[:-1]

        result = df.groupby("B", sort=False).A.mean()
        _, index = pd.factorize(data_for_grouping, sort=False)

        index = pd.Index(index, name="B")
        exp_vals = [1.0, 3.0, 4.0]
        if is_bool:
            exp_vals = exp_vals[:-1]
        expected = pd.Series(exp_vals, index=index, name="A")
        tm.assert_series_equal(result, expected)

    def test_groupby_extension_transform(self, data_for_grouping):
        is_bool = data_for_grouping.dtype._is_boolean

        valid = data_for_grouping[~data_for_grouping.isna()]
        df = pd.DataFrame({"A": [1, 1, 3, 3, 1, 4], "B": valid})
        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values, and the final entry has c==b
            #  (see data_for_grouping docstring)
            df = df.iloc[:-1]

        result = df.groupby("B").A.transform(len)
        expected = pd.Series([3, 3, 2, 2, 3, 1], name="A")
        if is_bool:
            expected = expected[:-1]

        tm.assert_series_equal(result, expected)

    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.groupby("B", group_keys=False, observed=False).apply(groupby_apply_op)
        df.groupby("B", group_keys=False, observed=False).A.apply(groupby_apply_op)
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.groupby("A", group_keys=False, observed=False).apply(groupby_apply_op)
        df.groupby("A", group_keys=False, observed=False).B.apply(groupby_apply_op)

    def test_groupby_apply_identity(self, data_for_grouping):
        df = pd.DataFrame({"A": [1, 1, 2, 2, 3, 3, 1, 4], "B": data_for_grouping})
        result = df.groupby("A").B.apply(lambda x: x.array)
        expected = pd.Series(
            [
                df.B.iloc[[0, 1, 6]].array,
                df.B.iloc[[2, 3]].array,
                df.B.iloc[[4, 5]].array,
                df.B.iloc[[7]].array,
            ],
            index=pd.Index([1, 2, 3, 4], name="A"),
            name="B",
        )
        tm.assert_series_equal(result, expected)

    def test_in_numeric_groupby(self, data_for_grouping):
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 1, 4],
                "B": data_for_grouping,
                "C": [1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

        dtype = data_for_grouping.dtype
        if (
            is_numeric_dtype(dtype)
            or is_bool_dtype(dtype)
            or dtype.name == "decimal"
            or is_string_dtype(dtype)
            or is_object_dtype(dtype)
            or dtype.kind == "m"  # in particular duration[*][pyarrow]
        ):
            expected = pd.Index(["B", "C"])
            result = df.groupby("A").sum().columns
        else:
            expected = pd.Index(["C"])

            msg = "|".join(
                [
                    # period/datetime
                    "does not support sum operations",
                    # all others
                    re.escape(f"agg function failed [how->sum,dtype->{dtype}"),
                ]
            )
            with pytest.raises(TypeError, match=msg):
                df.groupby("A").sum()
            result = df.groupby("A").sum(numeric_only=True).columns
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_reindex.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


def test_reindex(idx):
    result, indexer = idx.reindex(list(idx[:4]))
    assert isinstance(result, MultiIndex)
    assert result.names == ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]

    result, indexer = idx.reindex(list(idx))
    assert isinstance(result, MultiIndex)
    assert indexer is None
    assert result.names == ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]


def test_reindex_level(idx):
    index = Index(["one"])

    target, indexer = idx.reindex(index, level="second")
    target2, indexer2 = index.reindex(idx, level="second")

    exp_index = idx.join(index, level="second", how="right")
    exp_index2 = idx.join(index, level="second", how="left")

    assert target.equals(exp_index)
    exp_indexer = np.array([0, 2, 4])
    tm.assert_numpy_array_equal(indexer, exp_indexer, check_dtype=False)

    assert target2.equals(exp_index2)
    exp_indexer2 = np.array([0, -1, 0, -1, 0, -1])
    tm.assert_numpy_array_equal(indexer2, exp_indexer2, check_dtype=False)

    with pytest.raises(TypeError, match="Fill method not supported"):
        idx.reindex(idx, method="pad", level="second")


def test_reindex_preserves_names_when_target_is_list_or_ndarray(idx):
    # GH6552
    idx = idx.copy()
    target = idx.copy()
    idx.names = target.names = [None, None]

    other_dtype = MultiIndex.from_product([[1, 2], [3, 4]])

    # list & ndarray cases
    assert idx.reindex([])[0].names == [None, None]
    assert idx.reindex(np.array([]))[0].names == [None, None]
    assert idx.reindex(target.tolist())[0].names == [None, None]
    assert idx.reindex(target.values)[0].names == [None, None]
    assert idx.reindex(other_dtype.tolist())[0].names == [None, None]
    assert idx.reindex(other_dtype.values)[0].names == [None, None]

    idx.names = ["foo", "bar"]
    assert idx.reindex([])[0].names == ["foo", "bar"]
    assert idx.reindex(np.array([]))[0].names == ["foo", "bar"]
    assert idx.reindex(target.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(target.values)[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.values)[0].names == ["foo", "bar"]


def test_reindex_lvl_preserves_names_when_target_is_list_or_array():
    # GH7774
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]], names=["foo", "bar"])
    assert idx.reindex([], level=0)[0].names == ["foo", "bar"]
    assert idx.reindex([], level=1)[0].names == ["foo", "bar"]


def test_reindex_lvl_preserves_type_if_target_is_empty_list_or_array(
    using_infer_string,
):
    # GH7774
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]])
    assert idx.reindex([], level=0)[0].levels[0].dtype.type == np.int64
    exp = np.object_ if not using_infer_string else str
    assert idx.reindex([], level=1)[0].levels[1].dtype.type == exp

    # case with EA levels
    cat = pd.Categorical(["foo", "bar"])
    dti = pd.date_range("2016-01-01", periods=2, tz="US/Pacific")
    mi = MultiIndex.from_product([cat, dti])
    assert mi.reindex([], level=0)[0].levels[0].dtype == cat.dtype
    assert mi.reindex([], level=1)[0].levels[1].dtype == dti.dtype


def test_reindex_base(idx):
    expected = np.arange(idx.size, dtype=np.intp)

    actual = idx.get_indexer(idx)
    tm.assert_numpy_array_equal(expected, actual)

    with pytest.raises(ValueError, match="Invalid fill method"):
        idx.get_indexer(idx, method="invalid")


def test_reindex_non_unique():
    idx = MultiIndex.from_tuples([(0, 0), (1, 1), (1, 1), (2, 2)])
    a = pd.Series(np.arange(4), index=idx)
    new_idx = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 2)])

    msg = "cannot handle a non-unique multi-index!"
    with pytest.raises(ValueError, match=msg):
        a.reindex(new_idx)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
def test_reindex_empty_with_level(values):
    # GH41170
    idx = MultiIndex.from_arrays(values)
    result, result_indexer = idx.reindex(np.array(["b"]), level=0)
    expected = MultiIndex(levels=[["b"], values[1]], codes=[[], []])
    expected_indexer = np.array([], dtype=result_indexer.dtype)
    tm.assert_index_equal(result, expected)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_not_all_tuples():
    keys = [("i", "i"), ("i", "j"), ("j", "i"), "j"]
    mi = MultiIndex.from_tuples(keys[:-1])
    idx = Index(keys)
    res, indexer = mi.reindex(idx)

    tm.assert_index_equal(res, idx)
    expected = np.array([0, 1, 2, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)


def test_reindex_limit_arg_with_multiindex():
    # GH21247

    idx = MultiIndex.from_tuples([(3, "A"), (4, "A"), (4, "B")])

    df = pd.Series([0.02, 0.01, 0.012], index=idx)

    new_idx = MultiIndex.from_tuples(
        [
            (3, "A"),
            (3, "B"),
            (4, "A"),
            (4, "B"),
            (4, "C"),
            (5, "B"),
            (5, "C"),
            (6, "B"),
            (6, "C"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="limit argument only valid if doing pad, backfill or nearest reindexing",
    ):
        df.reindex(new_idx, fill_value=0, limit=1)


def test_reindex_with_none_in_nested_multiindex():
    # GH42883
    index = MultiIndex.from_tuples([(("a", None), 1), (("b", None), 2)])
    index2 = MultiIndex.from_tuples([(("b", None), 2), (("a", None), 1)])
    df1_dtype = pd.DataFrame([1, 2], index=index)
    df2_dtype = pd.DataFrame([2, 1], index=index2)

    result = df1_dtype.reindex_like(df2_dtype)
    expected = df2_dtype
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_timedelta_range.py ===
import numpy as np
import pytest

from pandas import (
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
    to_timedelta,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    Day,
    Second,
)


class TestTimedeltas:
    def test_timedelta_range_unit(self):
        # GH#49824
        tdi = timedelta_range("0 Days", periods=10, freq="100000D", unit="s")
        exp_arr = (np.arange(10, dtype="i8") * 100_000).view("m8[D]").astype("m8[s]")
        tm.assert_numpy_array_equal(tdi.to_numpy(), exp_arr)

    def test_timedelta_range(self):
        expected = to_timedelta(np.arange(5), unit="D")
        result = timedelta_range("0 days", periods=5, freq="D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta(np.arange(11), unit="D")
        result = timedelta_range("0 days", "10 days", freq="D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta(np.arange(5), unit="D") + Second(2) + Day()
        result = timedelta_range("1 days, 00:00:02", "5 days, 00:00:02", freq="D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta([1, 3, 5, 7, 9], unit="D") + Second(2)
        result = timedelta_range("1 days, 00:00:02", periods=5, freq="2D")
        tm.assert_index_equal(result, expected)

        expected = to_timedelta(np.arange(50), unit="min") * 30
        result = timedelta_range("0 days", freq="30min", periods=50)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "depr_unit, unit",
        [
            ("H", "hour"),
            ("T", "minute"),
            ("t", "minute"),
            ("S", "second"),
            ("L", "millisecond"),
            ("l", "millisecond"),
            ("U", "microsecond"),
            ("u", "microsecond"),
            ("N", "nanosecond"),
            ("n", "nanosecond"),
        ],
    )
    def test_timedelta_units_H_T_S_L_U_N_deprecated(self, depr_unit, unit):
        # GH#52536
        depr_msg = (
            f"'{depr_unit}' is deprecated and will be removed in a future version."
        )

        expected = to_timedelta(np.arange(5), unit=unit)
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = to_timedelta(np.arange(5), unit=depr_unit)
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, freq", [(3, "2D"), (5, "D"), (6, "19h12min"), (7, "16h"), (9, "12h")]
    )
    def test_linspace_behavior(self, periods, freq):
        # GH 20976
        result = timedelta_range(start="0 days", end="4 days", periods=periods)
        expected = timedelta_range(start="0 days", end="4 days", freq=freq)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("msg_freq, freq", [("H", "19H12min"), ("T", "19h12T")])
    def test_timedelta_range_H_T_deprecated(self, freq, msg_freq):
        # GH#52536
        msg = f"'{msg_freq}' is deprecated and will be removed in a future version."

        result = timedelta_range(start="0 days", end="4 days", periods=6)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = timedelta_range(start="0 days", end="4 days", freq=freq)
        tm.assert_index_equal(result, expected)

    def test_errors(self):
        # not enough params
        msg = (
            "Of the four parameters: start, end, periods, and freq, "
            "exactly three must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start="0 days")

        with pytest.raises(ValueError, match=msg):
            timedelta_range(end="5 days")

        with pytest.raises(ValueError, match=msg):
            timedelta_range(periods=2)

        with pytest.raises(ValueError, match=msg):
            timedelta_range()

        # too many params
        with pytest.raises(ValueError, match=msg):
            timedelta_range(start="0 days", end="5 days", periods=10, freq="h")

    @pytest.mark.parametrize(
        "start, end, freq, expected_periods",
        [
            ("1D", "10D", "2D", (10 - 1) // 2 + 1),
            ("2D", "30D", "3D", (30 - 2) // 3 + 1),
            ("2s", "50s", "5s", (50 - 2) // 5 + 1),
            # tests that worked before GH 33498:
            ("4D", "16D", "3D", (16 - 4) // 3 + 1),
            ("8D", "16D", "40s", (16 * 3600 * 24 - 8 * 3600 * 24) // 40 + 1),
        ],
    )
    def test_timedelta_range_freq_divide_end(self, start, end, freq, expected_periods):
        # GH 33498 only the cases where `(end % freq) == 0` used to fail
        res = timedelta_range(start=start, end=end, freq=freq)
        assert Timedelta(start) == res[0]
        assert Timedelta(end) >= res[-1]
        assert len(res) == expected_periods

    def test_timedelta_range_infer_freq(self):
        # https://github.com/pandas-dev/pandas/issues/35897
        result = timedelta_range("0s", "1s", periods=31)
        assert result.freq is None

    @pytest.mark.parametrize(
        "freq_depr, start, end, expected_values, expected_freq",
        [
            (
                "3.5S",
                "05:03:01",
                "05:03:10",
                ["0 days 05:03:01", "0 days 05:03:04.500000", "0 days 05:03:08"],
                "3500ms",
            ),
            (
                "2.5T",
                "5 hours",
                "5 hours 8 minutes",
                [
                    "0 days 05:00:00",
                    "0 days 05:02:30",
                    "0 days 05:05:00",
                    "0 days 05:07:30",
                ],
                "150s",
            ),
        ],
    )
    def test_timedelta_range_deprecated_freq(
        self, freq_depr, start, end, expected_values, expected_freq
    ):
        # GH#52536
        msg = (
            f"'{freq_depr[-1]}' is deprecated and will be removed in a future version."
        )

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = timedelta_range(start=start, end=end, freq=freq_depr)
        expected = TimedeltaIndex(
            expected_values, dtype="timedelta64[ns]", freq=expected_freq
        )
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_liboffsets.py ===
"""
Tests for helper functions in the cython tslibs.offsets
"""
from datetime import datetime

import pytest

from pandas._libs.tslibs.ccalendar import (
    get_firstbday,
    get_lastbday,
)
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday

from pandas import Timestamp


@pytest.fixture(params=["start", "end", "business_start", "business_end"])
def day_opt(request):
    return request.param


@pytest.mark.parametrize(
    "dt,exp_week_day,exp_last_day",
    [
        (datetime(2017, 11, 30), 3, 30),  # Business day.
        (datetime(1993, 10, 31), 6, 29),  # Non-business day.
    ],
)
def test_get_last_bday(dt, exp_week_day, exp_last_day):
    assert dt.weekday() == exp_week_day
    assert get_lastbday(dt.year, dt.month) == exp_last_day


@pytest.mark.parametrize(
    "dt,exp_week_day,exp_first_day",
    [
        (datetime(2017, 4, 1), 5, 3),  # Non-weekday.
        (datetime(1993, 10, 1), 4, 1),  # Business day.
    ],
)
def test_get_first_bday(dt, exp_week_day, exp_first_day):
    assert dt.weekday() == exp_week_day
    assert get_firstbday(dt.year, dt.month) == exp_first_day


@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (0, 15, datetime(2017, 11, 15)),
        (0, None, datetime(2017, 11, 30)),
        (1, "start", datetime(2017, 12, 1)),
        (-145, "end", datetime(2005, 10, 31)),
        (0, "business_end", datetime(2017, 11, 30)),
        (0, "business_start", datetime(2017, 11, 1)),
    ],
)
def test_shift_month_dt(months, day_opt, expected):
    dt = datetime(2017, 11, 30)
    assert liboffsets.shift_month(dt, months, day_opt=day_opt) == expected


@pytest.mark.parametrize(
    "months,day_opt,expected",
    [
        (1, "start", Timestamp("1929-06-01")),
        (-3, "end", Timestamp("1929-02-28")),
        (25, None, Timestamp("1931-06-5")),
        (-1, 31, Timestamp("1929-04-30")),
    ],
)
def test_shift_month_ts(months, day_opt, expected):
    ts = Timestamp("1929-05-05")
    assert liboffsets.shift_month(ts, months, day_opt=day_opt) == expected


def test_shift_month_error():
    dt = datetime(2017, 11, 15)
    day_opt = "this should raise"

    with pytest.raises(ValueError, match=day_opt):
        liboffsets.shift_month(dt, 3, day_opt=day_opt)


@pytest.mark.parametrize(
    "other,expected",
    [
        # Before March 1.
        (datetime(2017, 2, 10), {2: 1, -7: -7, 0: 0}),
        # After March 1.
        (Timestamp("2014-03-15", tz="US/Eastern"), {2: 2, -7: -6, 0: 1}),
    ],
)
@pytest.mark.parametrize("n", [2, -7, 0])
def test_roll_qtrday_year(other, expected, n):
    month = 3
    day_opt = "start"  # `other` will be compared to March 1.

    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]


@pytest.mark.parametrize(
    "other,expected",
    [
        # Before June 30.
        (datetime(1999, 6, 29), {5: 4, -7: -7, 0: 0}),
        # After June 30.
        (Timestamp(2072, 8, 24, 6, 17, 18), {5: 5, -7: -6, 0: 1}),
    ],
)
@pytest.mark.parametrize("n", [5, -7, 0])
def test_roll_qtrday_year2(other, expected, n):
    month = 6
    day_opt = "end"  # `other` will be compared to June 30.

    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]


def test_get_day_of_month_error():
    # get_day_of_month is not directly exposed.
    # We test it via roll_qtrday.
    dt = datetime(2017, 11, 15)
    day_opt = "foo"

    with pytest.raises(ValueError, match=day_opt):
        # To hit the raising case we need month == dt.month and n > 0.
        roll_qtrday(dt, n=3, month=11, day_opt=day_opt, modby=12)


@pytest.mark.parametrize(
    "month",
    [3, 5],  # (other.month % 3) < (month % 3)  # (other.month % 3) > (month % 3)
)
@pytest.mark.parametrize("n", [4, -3])
def test_roll_qtr_day_not_mod_unequal(day_opt, month, n):
    expected = {3: {-3: -2, 4: 4}, 5: {-3: -3, 4: 3}}

    other = Timestamp(2072, 10, 1, 6, 17, 18)  # Saturday.
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected[month][n]


@pytest.mark.parametrize(
    "other,month,exp_dict",
    [
        # Monday.
        (datetime(1999, 5, 31), 2, {-1: {"start": 0, "business_start": 0}}),
        # Saturday.
        (
            Timestamp(2072, 10, 1, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1, "business_start": 1}},
        ),
        # First business day.
        (
            Timestamp(2072, 10, 3, 6, 17, 18),
            4,
            {2: {"end": 1, "business_end": 1}, -1: {"start": 0}},
        ),
    ],
)
@pytest.mark.parametrize("n", [2, -1])
def test_roll_qtr_day_mod_equal(other, month, exp_dict, n, day_opt):
    # All cases have (other.month % 3) == (month % 3).
    expected = exp_dict.get(n, {}).get(day_opt, n)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected


@pytest.mark.parametrize(
    "n,expected", [(42, {29: 42, 1: 42, 31: 41}), (-4, {29: -4, 1: -3, 31: -4})]
)
@pytest.mark.parametrize("compare", [29, 1, 31])
def test_roll_convention(n, expected, compare):
    assert liboffsets.roll_convention(29, n, compare) == expected[compare]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_dtypes.py ===
import numpy as np
import pytest

from pandas.errors import DataError

from pandas.core.dtypes.common import pandas_dtype

from pandas import (
    NA,
    DataFrame,
    Series,
)
import pandas._testing as tm

# gh-12373 : rolling functions error on float32 data
# make sure rolling functions works for different dtypes
#
# further note that we are only checking rolling for fully dtype
# compliance (though both expanding and ewm inherit)


def get_dtype(dtype, coerce_int=None):
    if coerce_int is False and "int" in dtype:
        return None
    return pandas_dtype(dtype)


@pytest.fixture(
    params=[
        "object",
        "category",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "m8[ns]",
        "M8[ns]",
        "datetime64[ns, UTC]",
    ]
)
def dtypes(request):
    """Dtypes for window tests"""
    return request.param


@pytest.mark.parametrize(
    "method, data, expected_data, coerce_int, min_periods",
    [
        ("count", np.arange(5), [1, 2, 2, 2, 2], True, 0),
        ("count", np.arange(10, 0, -2), [1, 2, 2, 2, 2], True, 0),
        ("count", [0, 1, 2, np.nan, 4], [1, 2, 2, 1, 1], False, 0),
        ("max", np.arange(5), [np.nan, 1, 2, 3, 4], True, None),
        ("max", np.arange(10, 0, -2), [np.nan, 10, 8, 6, 4], True, None),
        ("max", [0, 1, 2, np.nan, 4], [np.nan, 1, 2, np.nan, np.nan], False, None),
        ("min", np.arange(5), [np.nan, 0, 1, 2, 3], True, None),
        ("min", np.arange(10, 0, -2), [np.nan, 8, 6, 4, 2], True, None),
        ("min", [0, 1, 2, np.nan, 4], [np.nan, 0, 1, np.nan, np.nan], False, None),
        ("sum", np.arange(5), [np.nan, 1, 3, 5, 7], True, None),
        ("sum", np.arange(10, 0, -2), [np.nan, 18, 14, 10, 6], True, None),
        ("sum", [0, 1, 2, np.nan, 4], [np.nan, 1, 3, np.nan, np.nan], False, None),
        ("mean", np.arange(5), [np.nan, 0.5, 1.5, 2.5, 3.5], True, None),
        ("mean", np.arange(10, 0, -2), [np.nan, 9, 7, 5, 3], True, None),
        ("mean", [0, 1, 2, np.nan, 4], [np.nan, 0.5, 1.5, np.nan, np.nan], False, None),
        ("std", np.arange(5), [np.nan] + [np.sqrt(0.5)] * 4, True, None),
        ("std", np.arange(10, 0, -2), [np.nan] + [np.sqrt(2)] * 4, True, None),
        (
            "std",
            [0, 1, 2, np.nan, 4],
            [np.nan] + [np.sqrt(0.5)] * 2 + [np.nan] * 2,
            False,
            None,
        ),
        ("var", np.arange(5), [np.nan, 0.5, 0.5, 0.5, 0.5], True, None),
        ("var", np.arange(10, 0, -2), [np.nan, 2, 2, 2, 2], True, None),
        ("var", [0, 1, 2, np.nan, 4], [np.nan, 0.5, 0.5, np.nan, np.nan], False, None),
        ("median", np.arange(5), [np.nan, 0.5, 1.5, 2.5, 3.5], True, None),
        ("median", np.arange(10, 0, -2), [np.nan, 9, 7, 5, 3], True, None),
        (
            "median",
            [0, 1, 2, np.nan, 4],
            [np.nan, 0.5, 1.5, np.nan, np.nan],
            False,
            None,
        ),
    ],
)
def test_series_dtypes(
    method, data, expected_data, coerce_int, dtypes, min_periods, step
):
    ser = Series(data, dtype=get_dtype(dtypes, coerce_int=coerce_int))
    rolled = ser.rolling(2, min_periods=min_periods, step=step)

    if dtypes in ("m8[ns]", "M8[ns]", "datetime64[ns, UTC]") and method != "count":
        msg = "No numeric types to aggregate"
        with pytest.raises(DataError, match=msg):
            getattr(rolled, method)()
    else:
        result = getattr(rolled, method)()
        expected = Series(expected_data, dtype="float64")[::step]
        tm.assert_almost_equal(result, expected)


def test_series_nullable_int(any_signed_int_ea_dtype, step):
    # GH 43016
    ser = Series([0, 1, NA], dtype=any_signed_int_ea_dtype)
    result = ser.rolling(2, step=step).mean()
    expected = Series([np.nan, 0.5, np.nan])[::step]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, expected_data, min_periods",
    [
        ("count", {0: Series([1, 2, 2, 2, 2]), 1: Series([1, 2, 2, 2, 2])}, 0),
        (
            "max",
            {0: Series([np.nan, 2, 4, 6, 8]), 1: Series([np.nan, 3, 5, 7, 9])},
            None,
        ),
        (
            "min",
            {0: Series([np.nan, 0, 2, 4, 6]), 1: Series([np.nan, 1, 3, 5, 7])},
            None,
        ),
        (
            "sum",
            {0: Series([np.nan, 2, 6, 10, 14]), 1: Series([np.nan, 4, 8, 12, 16])},
            None,
        ),
        (
            "mean",
            {0: Series([np.nan, 1, 3, 5, 7]), 1: Series([np.nan, 2, 4, 6, 8])},
            None,
        ),
        (
            "std",
            {
                0: Series([np.nan] + [np.sqrt(2)] * 4),
                1: Series([np.nan] + [np.sqrt(2)] * 4),
            },
            None,
        ),
        (
            "var",
            {0: Series([np.nan, 2, 2, 2, 2]), 1: Series([np.nan, 2, 2, 2, 2])},
            None,
        ),
        (
            "median",
            {0: Series([np.nan, 1, 3, 5, 7]), 1: Series([np.nan, 2, 4, 6, 8])},
            None,
        ),
    ],
)
def test_dataframe_dtypes(method, expected_data, dtypes, min_periods, step):
    df = DataFrame(np.arange(10).reshape((5, 2)), dtype=get_dtype(dtypes))
    rolled = df.rolling(2, min_periods=min_periods, step=step)

    if dtypes in ("m8[ns]", "M8[ns]", "datetime64[ns, UTC]") and method != "count":
        msg = "Cannot aggregate non-numeric type"
        with pytest.raises(DataError, match=msg):
            getattr(rolled, method)()
    else:
        result = getattr(rolled, method)()
        expected = DataFrame(expected_data, dtype="float64")[::step]
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\interface.py ===
import warnings

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_extension_array_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype

import pandas as pd
import pandas._testing as tm


class BaseInterfaceTests:
    """Tests that the basic interface is satisfied."""

    # ------------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------------

    def test_len(self, data):
        assert len(data) == 100

    def test_size(self, data):
        assert data.size == 100

    def test_ndim(self, data):
        assert data.ndim == 1

    def test_can_hold_na_valid(self, data):
        # GH-20761
        assert data._can_hold_na is True

    def test_contains(self, data, data_missing):
        # GH-37867
        # Tests for membership checks. Membership checks for nan-likes is tricky and
        # the settled on rule is: `nan_like in arr` is True if nan_like is
        # arr.dtype.na_value and arr.isna().any() is True. Else the check returns False.

        na_value = data.dtype.na_value
        # ensure data without missing values
        data = data[~data.isna()]

        # first elements are non-missing
        assert data[0] in data
        assert data_missing[0] in data_missing

        # check the presence of na_value
        assert na_value in data_missing
        assert na_value not in data

        # the data can never contain other nan-likes than na_value
        for na_value_obj in tm.NULL_OBJECTS:
            if na_value_obj is na_value or type(na_value_obj) == type(na_value):
                # type check for e.g. two instances of Decimal("NAN")
                continue
            assert na_value_obj not in data
            assert na_value_obj not in data_missing

    def test_memory_usage(self, data):
        s = pd.Series(data)
        result = s.memory_usage(index=False)
        assert result == s.nbytes

    def test_array_interface(self, data):
        result = np.array(data)
        assert result[0] == data[0]

        result = np.array(data, dtype=object)
        expected = np.array(list(data), dtype=object)
        if expected.ndim > 1:
            # nested data, explicitly construct as 1D
            expected = construct_1d_object_array_from_listlike(list(data))
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface_copy(self, data):
        result_copy1 = np.array(data, copy=True)
        result_copy2 = np.array(data, copy=True)
        assert not np.may_share_memory(result_copy1, result_copy2)

        if not np_version_gt2:
            # copy=False semantics are only supported in NumPy>=2.
            return

        warning_raised = False
        msg = "Starting with NumPy 2.0, the behavior of the 'copy' keyword has changed"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_nocopy1 = np.array(data, copy=False)
            assert len(w) <= 1
            if len(w):
                warning_raised = True
                assert msg in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_nocopy2 = np.array(data, copy=False)
            assert len(w) <= 1
            if len(w):
                warning_raised = True
                assert msg in str(w[0].message)

        if not warning_raised:
            # If copy=False was given and did not raise, these must share the same data
            assert np.may_share_memory(result_nocopy1, result_nocopy2)

    def test_is_extension_array_dtype(self, data):
        assert is_extension_array_dtype(data)
        assert is_extension_array_dtype(data.dtype)
        assert is_extension_array_dtype(pd.Series(data))
        assert isinstance(data.dtype, ExtensionDtype)

    def test_no_values_attribute(self, data):
        # GH-20735: EA's with .values attribute give problems with internal
        # code, disallowing this for now until solved
        assert not hasattr(data, "values")
        assert not hasattr(data, "_values")

    def test_is_numeric_honored(self, data):
        result = pd.Series(data)
        if hasattr(result._mgr, "blocks"):
            assert result._mgr.blocks[0].is_numeric is data.dtype._is_numeric

    def test_isna_extension_array(self, data_missing):
        # If your `isna` returns an ExtensionArray, you must also implement
        # _reduce. At the *very* least, you must implement any and all
        na = data_missing.isna()
        if is_extension_array_dtype(na):
            assert na._reduce("any")
            assert na.any()

            assert not na._reduce("all")
            assert not na.all()

            assert na.dtype._is_boolean

    def test_copy(self, data):
        # GH#27083 removing deep keyword from EA.copy
        assert data[0] != data[1]
        result = data.copy()

        if data.dtype._is_immutable:
            pytest.skip(f"test_copy assumes mutability and {data.dtype} is immutable")

        data[1] = data[0]
        assert result[1] != result[0]

    def test_view(self, data):
        # view with no dtype should return a shallow copy, *not* the same
        #  object
        assert data[1] != data[0]

        result = data.view()
        assert result is not data
        assert type(result) == type(data)

        if data.dtype._is_immutable:
            pytest.skip(f"test_view assumes mutability and {data.dtype} is immutable")

        result[1] = result[0]
        assert data[1] == data[0]

        # check specifically that the `dtype` kwarg is accepted
        data.view(dtype=None)

    def test_tolist(self, data):
        result = data.tolist()
        expected = list(data)
        assert isinstance(result, list)
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_any_index.py ===
"""
Tests that can be parametrized over _any_ Index object.
"""
import re

import numpy as np
import pytest

from pandas.errors import InvalidIndexError

import pandas._testing as tm


def test_boolean_context_compat(index):
    # GH#7897
    with pytest.raises(ValueError, match="The truth value of a"):
        if index:
            pass

    with pytest.raises(ValueError, match="The truth value of a"):
        bool(index)


def test_sort(index):
    msg = "cannot sort an Index object in-place, use sort_values instead"
    with pytest.raises(TypeError, match=msg):
        index.sort()


def test_hash_error(index):
    with pytest.raises(TypeError, match=f"unhashable type: '{type(index).__name__}'"):
        hash(index)


def test_mutability(index):
    if not len(index):
        pytest.skip("Test doesn't make sense for empty index")
    msg = "Index does not support mutable operations"
    with pytest.raises(TypeError, match=msg):
        index[0] = index[0]


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_map_identity_mapping(index, request):
    # GH#12766

    result = index.map(lambda x: x)
    if index.dtype == object and result.dtype in [bool, "string"]:
        assert (index == result).all()
        # TODO: could work that into the 'exact="equiv"'?
        return  # FIXME: doesn't belong in this file anymore!
    tm.assert_index_equal(result, index, exact="equiv")


def test_wrong_number_names(index):
    names = index.nlevels * ["apple", "banana", "carrot"]
    with pytest.raises(ValueError, match="^Length"):
        index.names = names


def test_view_preserves_name(index):
    assert index.view().name == index.name


def test_ravel(index):
    # GH#19956 ravel returning ndarray is deprecated, in 2.0 returns a view on self
    res = index.ravel()
    tm.assert_index_equal(res, index)


class TestConversion:
    def test_to_series(self, index):
        # assert that we are creating a copy of the index

        ser = index.to_series()
        assert ser.values is not index.values
        assert ser.index is not index
        assert ser.name == index.name

    def test_to_series_with_arguments(self, index):
        # GH#18699

        # index kwarg
        ser = index.to_series(index=index)

        assert ser.values is not index.values
        assert ser.index is index
        assert ser.name == index.name

        # name kwarg
        ser = index.to_series(name="__test")

        assert ser.values is not index.values
        assert ser.index is not index
        assert ser.name != index.name

    def test_tolist_matches_list(self, index):
        assert index.tolist() == list(index)


class TestRoundTrips:
    def test_pickle_roundtrip(self, index):
        result = tm.round_trip_pickle(index)
        tm.assert_index_equal(result, index, exact=True)
        if result.nlevels > 1:
            # GH#8367 round-trip with timezone
            assert index.equal_levels(result)

    def test_pickle_preserves_name(self, index):
        original_name, index.name = index.name, "foo"
        unpickled = tm.round_trip_pickle(index)
        assert index.equals(unpickled)
        index.name = original_name


class TestIndexing:
    def test_get_loc_listlike_raises_invalid_index_error(self, index):
        # and never TypeError
        key = np.array([0, 1], dtype=np.intp)

        with pytest.raises(InvalidIndexError, match=r"\[0 1\]"):
            index.get_loc(key)

        with pytest.raises(InvalidIndexError, match=r"\[False  True\]"):
            index.get_loc(key.astype(bool))

    def test_getitem_ellipsis(self, index):
        # GH#21282
        result = index[...]
        assert result.equals(index)
        assert result is not index

    def test_slice_keeps_name(self, index):
        assert index.name == index[1:].name

    @pytest.mark.parametrize("item", [101, "no_int", 2.5])
    def test_getitem_error(self, index, item):
        msg = "|".join(
            [
                r"index 101 is out of bounds for axis 0 with size [\d]+",
                re.escape(
                    "only integers, slices (`:`), ellipsis (`...`), "
                    "numpy.newaxis (`None`) and integer or boolean arrays "
                    "are valid indices"
                ),
                "index out of bounds",  # string[pyarrow]
            ]
        )
        with pytest.raises(IndexError, match=msg):
            index[item]


class TestRendering:
    def test_str(self, index):
        # test the string repr
        index.name = "foo"
        assert "'foo'" in str(index)
        assert type(index).__name__ in str(index)


class TestReductions:
    def test_argmax_axis_invalid(self, index):
        # GH#23081
        msg = r"`axis` must be fewer than the number of dimensions \(1\)"
        with pytest.raises(ValueError, match=msg):
            index.argmax(axis=1)
        with pytest.raises(ValueError, match=msg):
            index.argmin(axis=2)
        with pytest.raises(ValueError, match=msg):
            index.min(axis=-2)
        with pytest.raises(ValueError, match=msg):
            index.max(axis=-3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_rel_queries.py ===
from sympy.assumptions.lra_satask import lra_satask
from sympy.logic.algorithms.lra_theory import UnhandledInput
from sympy.assumptions.ask import Q, ask

from sympy.core import symbols, Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.core.numbers import I

from sympy.testing.pytest import raises, XFAIL
x, y, z = symbols("x y z", real=True)

def test_lra_satask():
    im = Symbol('im', imaginary=True)

    # test preprocessing of unequalities is working correctly
    assert lra_satask(Q.eq(x, 1), ~Q.ne(x, 0)) is False
    assert lra_satask(Q.eq(x, 0), ~Q.ne(x, 0)) is True
    assert lra_satask(~Q.ne(x, 0), Q.eq(x, 0)) is True
    assert lra_satask(~Q.eq(x, 0), Q.eq(x, 0)) is False
    assert lra_satask(Q.ne(x, 0), Q.eq(x, 0)) is False

    # basic tests
    assert lra_satask(Q.ne(x, x)) is False
    assert lra_satask(Q.eq(x, x)) is True
    assert lra_satask(Q.gt(x, 0), Q.gt(x, 1)) is True

    # check that True/False are handled
    assert lra_satask(Q.gt(x, 0), True) is None
    assert raises(ValueError, lambda: lra_satask(Q.gt(x, 0), False))

    # check imaginary numbers are correctly handled
    # (im * I).is_real returns True so this is an edge case
    raises(UnhandledInput, lambda: lra_satask(Q.gt(im * I, 0), Q.gt(im * I, 0)))

    # check matrix inputs
    X = MatrixSymbol("X", 2, 2)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(X, 2) & Q.gt(X, 3)))


def test_old_assumptions():
    # test unhandled old assumptions
    w = symbols("w")
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", rational=False, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", odd=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", even=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", prime=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", composite=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", integer=True, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))
    w = symbols("w", integer=False, real=True)
    raises(UnhandledInput, lambda: lra_satask(Q.lt(w, 2) & Q.gt(w, 3)))

    # test handled
    w = symbols("w", positive=True, real=True)
    assert lra_satask(Q.le(w, 0)) is False
    assert lra_satask(Q.gt(w, 0)) is True
    w = symbols("w", negative=True, real=True)
    assert lra_satask(Q.lt(w, 0)) is True
    assert lra_satask(Q.ge(w, 0)) is False
    w = symbols("w", zero=True, real=True)
    assert lra_satask(Q.eq(w, 0)) is True
    assert lra_satask(Q.ne(w, 0)) is False
    w = symbols("w", nonzero=True, real=True)
    assert lra_satask(Q.ne(w, 0)) is True
    assert lra_satask(Q.eq(w, 1)) is None
    w = symbols("w", nonpositive=True, real=True)
    assert lra_satask(Q.le(w, 0)) is True
    assert lra_satask(Q.gt(w, 0)) is False
    w = symbols("w", nonnegative=True, real=True)
    assert lra_satask(Q.ge(w, 0)) is True
    assert lra_satask(Q.lt(w, 0)) is False


def test_rel_queries():
    assert ask(Q.lt(x, 2) & Q.gt(x, 3)) is False
    assert ask(Q.positive(x - z), (x > y) & (y > z)) is True
    assert ask(x + y > 2, (x < 0) & (y <0)) is False
    assert ask(x > z, (x > y) & (y > z)) is True


def test_unhandled_queries():
    X = MatrixSymbol("X", 2, 2)
    assert ask(Q.lt(X, 2) & Q.gt(X, 3)) is None


def test_all_pred():
    # test usable pred
    assert lra_satask(Q.extended_positive(x), (x > 2)) is True
    assert lra_satask(Q.positive_infinite(x)) is False
    assert lra_satask(Q.negative_infinite(x)) is False

    # test disallowed pred
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.prime(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.composite(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.odd(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.even(x)))
    raises(UnhandledInput, lambda: lra_satask((x > 0), (x > 2) & Q.integer(x)))


def test_number_line_properties():
    # From:
    # https://en.wikipedia.org/wiki/Inequality_(mathematics)#Properties_on_the_number_line

    a, b, c = symbols("a b c", real=True)

    # Transitivity
    # If a <= b and b <= c, then a <= c.
    assert ask(a <= c, (a <= b) & (b <= c)) is True
    # If a <= b and b < c, then a < c.
    assert ask(a < c, (a <= b) & (b < c)) is True
    # If a < b and b <= c, then a < c.
    assert ask(a < c, (a < b) & (b <= c)) is True

    # Addition and subtraction
    # If a <= b, then a + c <= b + c and a - c <= b - c.
    assert ask(a + c <= b + c, a <= b) is True
    assert ask(a - c <= b - c, a <= b) is True


@XFAIL
def test_failing_number_line_properties():
    # From:
    # https://en.wikipedia.org/wiki/Inequality_(mathematics)#Properties_on_the_number_line

    a, b, c = symbols("a b c", real=True)

    # Multiplication and division
    # If a <= b and c > 0, then ac <= bc and a/c <= b/c. (True for non-zero c)
    assert ask(a*c <= b*c, (a <= b) & (c > 0) & ~ Q.zero(c)) is True
    assert ask(a/c <= b/c, (a <= b) & (c > 0) & ~ Q.zero(c)) is True
    # If a <= b and c < 0, then ac >= bc and a/c >= b/c. (True for non-zero c)
    assert ask(a*c >= b*c, (a <= b) & (c < 0) & ~ Q.zero(c)) is True
    assert ask(a/c >= b/c, (a <= b) & (c < 0) & ~ Q.zero(c)) is True

    # Additive inverse
    # If a <= b, then -a >= -b.
    assert ask(-a >= -b, a <= b) is True

    # Multiplicative inverse
    # For a, b that are both negative or both positive:
    # If a <= b, then 1/a >= 1/b .
    assert ask(1/a >= 1/b, (a <= b) & Q.positive(x) & Q.positive(b)) is True
    assert ask(1/a >= 1/b, (a <= b) & Q.negative(x) & Q.negative(b)) is True


def test_equality():
    # test symmetry and reflexivity
    assert ask(Q.eq(x, x)) is True
    assert ask(Q.eq(y, x), Q.eq(x, y)) is True
    assert ask(Q.eq(y, x), ~Q.eq(z, z) | Q.eq(x, y)) is True

    # test transitivity
    assert ask(Q.eq(x,z), Q.eq(x,y) & Q.eq(y,z)) is True


@XFAIL
def test_equality_failing():
    # Note that implementing the substitution property of equality
    # most likely requires a redesign of the new assumptions.
    # See issue #25485 for why this is the case and general ideas
    # about how things could be redesigned.

    # test substitution property
    assert ask(Q.prime(x), Q.eq(x, y) & Q.prime(y)) is True
    assert ask(Q.real(x), Q.eq(x, y) & Q.real(y)) is True
    assert ask(Q.imaginary(x), Q.eq(x, y) & Q.imaginary(y)) is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_symbolic_multivariate.py ===
from sympy.stats import Expectation, Normal, Variance, Covariance
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.symbolic_multivariate_probability import (ExpectationMatrix,
                            VarianceMatrix, CrossCovarianceMatrix)

j, k = symbols("j,k")

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)

A2 = MatrixSymbol("A2", 2, 2)
B2 = MatrixSymbol("B2", 2, 2)

X = RandomMatrixSymbol("X", k, 1)
Y = RandomMatrixSymbol("Y", k, 1)
Z = RandomMatrixSymbol("Z", k, 1)
W = RandomMatrixSymbol("W", k, 1)

R = RandomMatrixSymbol("R", k, k)

X2 = RandomMatrixSymbol("X2", 2, 1)

normal = Normal("normal", 0, 1)

m1 = Matrix([
    [1, j*Normal("normal2", 2, 1)],
    [normal, 0]
])

def test_multivariate_expectation():
    expr = Expectation(a)
    assert expr == Expectation(a) == ExpectationMatrix(a)
    assert expr.expand() == a

    expr = Expectation(X)
    assert expr == Expectation(X) == ExpectationMatrix(X)
    assert expr.shape == (k, 1)
    assert expr.rows == k
    assert expr.cols == 1
    assert isinstance(expr, ExpectationMatrix)

    expr = Expectation(A*X + b)
    assert expr == ExpectationMatrix(A*X + b)
    assert expr.expand() == A*ExpectationMatrix(X) + b
    assert isinstance(expr, ExpectationMatrix)
    assert expr.shape == (k, 1)

    expr = Expectation(m1*X2)
    assert expr.expand() == expr

    expr = Expectation(A2*m1*B2*X2)
    assert expr.args[0].args == (A2, m1, B2, X2)
    assert expr.expand() == A2*ExpectationMatrix(m1*B2*X2)

    expr = Expectation((X + Y)*(X - Y).T)
    assert expr.expand() == ExpectationMatrix(X*X.T) - ExpectationMatrix(X*Y.T) +\
                ExpectationMatrix(Y*X.T) - ExpectationMatrix(Y*Y.T)

    expr = Expectation(A*X + B*Y)
    assert expr.expand() == A*ExpectationMatrix(X) + B*ExpectationMatrix(Y)

    assert Expectation(m1).doit() == Matrix([[1, 2*j], [0, 0]])

    x1 = Matrix([
    [Normal('N11', 11, 1), Normal('N12', 12, 1)],
    [Normal('N21', 21, 1), Normal('N22', 22, 1)]
    ])
    x2 = Matrix([
    [Normal('M11', 1, 1), Normal('M12', 2, 1)],
    [Normal('M21', 3, 1), Normal('M22', 4, 1)]
    ])

    assert Expectation(Expectation(x1 + x2)).doit(deep=False) == ExpectationMatrix(x1 + x2)
    assert Expectation(Expectation(x1 + x2)).doit() == Matrix([[12, 14], [24, 26]])


def test_multivariate_variance():
    raises(ShapeError, lambda: Variance(A))

    expr = Variance(a)
    assert expr == Variance(a) == VarianceMatrix(a)
    assert expr.expand() == ZeroMatrix(k, k)
    expr = Variance(a.T)
    assert expr == Variance(a.T) == VarianceMatrix(a.T)
    assert expr.expand() == ZeroMatrix(k, k)

    expr = Variance(X)
    assert expr == Variance(X) == VarianceMatrix(X)
    assert expr.shape == (k, k)
    assert expr.rows == k
    assert expr.cols == k
    assert isinstance(expr, VarianceMatrix)

    expr = Variance(A*X)
    assert expr == VarianceMatrix(A*X)
    assert expr.expand() == A*VarianceMatrix(X)*A.T
    assert isinstance(expr, VarianceMatrix)
    assert expr.shape == (k, k)

    expr = Variance(A*B*X)
    assert expr.expand() == A*B*VarianceMatrix(X)*B.T*A.T

    expr = Variance(m1*X2)
    assert expr.expand() == expr

    expr = Variance(A2*m1*B2*X2)
    assert expr.args[0].args == (A2, m1, B2, X2)
    assert expr.expand() == expr

    expr = Variance(A*X + B*Y)
    assert expr.expand() == 2*A*CrossCovarianceMatrix(X, Y)*B.T +\
                    A*VarianceMatrix(X)*A.T + B*VarianceMatrix(Y)*B.T

def test_multivariate_crosscovariance():
    raises(ShapeError, lambda: Covariance(X, Y.T))
    raises(ShapeError, lambda: Covariance(X, A))


    expr = Covariance(a.T, b.T)
    assert expr.shape == (1, 1)
    assert expr.expand() == ZeroMatrix(1, 1)

    expr = Covariance(a, b)
    assert expr == Covariance(a, b) == CrossCovarianceMatrix(a, b)
    assert expr.expand() == ZeroMatrix(k, k)
    assert expr.shape == (k, k)
    assert expr.rows == k
    assert expr.cols == k
    assert isinstance(expr, CrossCovarianceMatrix)

    expr = Covariance(A*X + a, b)
    assert expr.expand() == ZeroMatrix(k, k)

    expr = Covariance(X, Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == expr

    expr = Covariance(X, X)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == VarianceMatrix(X)

    expr = Covariance(X + Y, Z)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)

    expr = Covariance(A*X, Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A*CrossCovarianceMatrix(X, Y)

    expr = Covariance(X, B*Y)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == CrossCovarianceMatrix(X, Y)*B.T

    expr = Covariance(A*X + a, B.T*Y + b)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A*CrossCovarianceMatrix(X, Y)*B

    expr = Covariance(A*X + B*Y + a, C.T*Z + D.T*W + b)
    assert isinstance(expr, CrossCovarianceMatrix)
    assert expr.expand() == A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C \
        + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_datetimelike.py ===
""" generic datetimelike tests """

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestDatetimeLike:
    @pytest.fixture(
        params=[
            pd.period_range("20130101", periods=5, freq="D"),
            pd.TimedeltaIndex(
                [
                    "0 days 01:00:00",
                    "1 days 01:00:00",
                    "2 days 01:00:00",
                    "3 days 01:00:00",
                    "4 days 01:00:00",
                ],
                dtype="timedelta64[ns]",
                freq="D",
            ),
            pd.DatetimeIndex(
                ["2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05"],
                dtype="datetime64[ns]",
                freq="D",
            ),
        ]
    )
    def simple_index(self, request):
        return request.param

    def test_isin(self, simple_index):
        index = simple_index[:4]
        result = index.isin(index)
        assert result.all()

        result = index.isin(list(index))
        assert result.all()

        result = index.isin([index[2], 5])
        expected = np.array([False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort_matches_array(self, simple_index):
        idx = simple_index
        idx = idx.insert(1, pd.NaT)

        result = idx.argsort()
        expected = idx._data.argsort()
        tm.assert_numpy_array_equal(result, expected)

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_shift_identity(self, simple_index):
        idx = simple_index
        tm.assert_index_equal(idx, idx.shift(0))

    def test_shift_empty(self, simple_index):
        # GH#14811
        idx = simple_index[:0]
        tm.assert_index_equal(idx, idx.shift(1))

    def test_str(self, simple_index):
        # test the string repr
        idx = simple_index.copy()
        idx.name = "foo"
        assert f"length={len(idx)}" not in str(idx)
        assert "'foo'" in str(idx)
        assert type(idx).__name__ in str(idx)

        if hasattr(idx, "tz"):
            if idx.tz is not None:
                assert idx.tz in str(idx)
        if isinstance(idx, pd.PeriodIndex):
            assert f"dtype='period[{idx.freqstr}]'" in str(idx)
        else:
            assert f"freq='{idx.freqstr}'" in str(idx)

    def test_view(self, simple_index):
        idx = simple_index

        idx_view = idx.view("i8")
        result = type(simple_index)(idx)
        tm.assert_index_equal(result, idx)

        msg = "Passing a type in .*Index.view is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx_view = idx.view(type(simple_index))
        result = type(simple_index)(idx)
        tm.assert_index_equal(result, idx_view)

    def test_map_callable(self, simple_index):
        index = simple_index
        expected = index + index.freq
        result = index.map(lambda x: x + index.freq)
        tm.assert_index_equal(result, expected)

        # map to NaT
        result = index.map(lambda x: pd.NaT if x == index[0] else x)
        expected = pd.Index([pd.NaT] + index[1:].tolist())
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: pd.Series(values, index, dtype=object),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_map_dictlike(self, mapper, simple_index):
        index = simple_index
        expected = index + index.freq

        # don't compare the freqs
        if isinstance(expected, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            expected = expected._with_freq(None)

        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

        expected = pd.Index([pd.NaT] + index[1:].tolist())
        result = index.map(mapper(expected, index))
        tm.assert_index_equal(result, expected)

        # empty map; these map to np.nan because we cannot know
        # to re-infer things
        expected = pd.Index([np.nan] * len(index))
        result = index.map(mapper([], []))
        tm.assert_index_equal(result, expected)

    def test_getitem_preserves_freq(self, simple_index):
        index = simple_index
        assert index.freq is not None

        result = index[:]
        assert result.freq == index.freq

    def test_where_cast_str(self, simple_index):
        index = simple_index

        mask = np.ones(len(index), dtype=bool)
        mask[-1] = False

        result = index.where(mask, str(index[0]))
        expected = index.where(mask, index[0])
        tm.assert_index_equal(result, expected)

        result = index.where(mask, [str(index[0])])
        tm.assert_index_equal(result, expected)

        expected = index.astype(object).where(mask, "foo")
        result = index.where(mask, "foo")
        tm.assert_index_equal(result, expected)

        result = index.where(mask, ["foo"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_diff(self, unit):
        # GH 55080
        dti = pd.to_datetime([10, 20, 30], unit=unit).as_unit(unit)
        result = dti.diff(1)
        expected = pd.to_timedelta([pd.NaT, 10, 10], unit=unit).as_unit(unit)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_iloc.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.fixture
def simple_multiindex_dataframe():
    """
    Factory function to create simple 3 x 3 dataframe with
    both columns and row MultiIndex using supplied data or
    random data by default.
    """

    data = np.random.default_rng(2).standard_normal((3, 3))
    return DataFrame(
        data, columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]]
    )


@pytest.mark.parametrize(
    "indexer, expected",
    [
        (
            lambda df: df.iloc[0],
            lambda arr: Series(arr[0], index=[[2, 2, 4], [6, 8, 10]], name=(4, 8)),
        ),
        (
            lambda df: df.iloc[2],
            lambda arr: Series(arr[2], index=[[2, 2, 4], [6, 8, 10]], name=(8, 12)),
        ),
        (
            lambda df: df.iloc[:, 2],
            lambda arr: Series(arr[:, 2], index=[[4, 4, 8], [8, 10, 12]], name=(4, 10)),
        ),
    ],
)
def test_iloc_returns_series(indexer, expected, simple_multiindex_dataframe):
    df = simple_multiindex_dataframe
    arr = df.values
    result = indexer(df)
    expected = expected(arr)
    tm.assert_series_equal(result, expected)


def test_iloc_returns_dataframe(simple_multiindex_dataframe):
    df = simple_multiindex_dataframe
    result = df.iloc[[0, 1]]
    expected = df.xs(4, drop_level=False)
    tm.assert_frame_equal(result, expected)


def test_iloc_returns_scalar(simple_multiindex_dataframe):
    df = simple_multiindex_dataframe
    arr = df.values
    result = df.iloc[2, 2]
    expected = arr[2, 2]
    assert result == expected


def test_iloc_getitem_multiple_items():
    # GH 5528
    tup = zip(*[["a", "a", "b", "b"], ["x", "y", "x", "y"]])
    index = MultiIndex.from_tuples(tup)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=index)
    result = df.iloc[[2, 3]]
    expected = df.xs("b", drop_level=False)
    tm.assert_frame_equal(result, expected)


def test_iloc_getitem_labels():
    # this is basically regular indexing
    arr = np.random.default_rng(2).standard_normal((4, 3))
    df = DataFrame(
        arr,
        columns=[["i", "i", "j"], ["A", "A", "B"]],
        index=[["i", "i", "j", "k"], ["X", "X", "Y", "Y"]],
    )
    result = df.iloc[2, 2]
    expected = arr[2, 2]
    assert result == expected


def test_frame_getitem_slice(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    result = df.iloc[:4]
    expected = df[:4]
    tm.assert_frame_equal(result, expected)


def test_frame_setitem_slice(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    df.iloc[:4] = 0

    assert (df.values[:4] == 0).all()
    assert (df.values[4:] != 0).all()


def test_indexing_ambiguity_bug_1678():
    # GH 1678
    columns = MultiIndex.from_tuples(
        [("Ohio", "Green"), ("Ohio", "Red"), ("Colorado", "Green")]
    )
    index = MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])

    df = DataFrame(np.arange(12).reshape((4, 3)), index=index, columns=columns)

    result = df.iloc[:, 1]
    expected = df.loc[:, ("Ohio", "Red")]
    tm.assert_series_equal(result, expected)


def test_iloc_integer_locations():
    # GH 13797
    data = [
        ["str00", "str01"],
        ["str10", "str11"],
        ["str20", "srt21"],
        ["str30", "str31"],
        ["str40", "str41"],
    ]

    index = MultiIndex.from_tuples(
        [("CC", "A"), ("CC", "B"), ("CC", "B"), ("BB", "a"), ("BB", "b")]
    )

    expected = DataFrame(data)
    df = DataFrame(data, index=index)

    result = DataFrame([[df.iloc[r, c] for c in range(2)] for r in range(5)])

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, indexes, values, expected_k",
    [
        # test without indexer value in first level of MultiIndex
        ([[2, 22, 5], [2, 33, 6]], [0, -1, 1], [2, 3, 1], [7, 10]),
        # test like code sample 1 in the issue
        ([[1, 22, 555], [1, 33, 666]], [0, -1, 1], [200, 300, 100], [755, 1066]),
        # test like code sample 2 in the issue
        ([[1, 3, 7], [2, 4, 8]], [0, -1, 1], [10, 10, 1000], [17, 1018]),
        # test like code sample 3 in the issue
        ([[1, 11, 4], [2, 22, 5], [3, 33, 6]], [0, -1, 1], [4, 7, 10], [8, 15, 13]),
    ],
)
def test_iloc_setitem_int_multiindex_series(data, indexes, values, expected_k):
    # GH17148
    df = DataFrame(data=data, columns=["i", "j", "k"])
    df = df.set_index(["i", "j"])

    series = df.k.copy()
    for i, v in zip(indexes, values):
        series.iloc[i] += v

    df["k"] = expected_k
    expected = df.k
    tm.assert_series_equal(series, expected)


def test_getitem_iloc(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    result = df.iloc[2]
    expected = df.xs(df.index[2])
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\tests\test_runtests_pytest.py ===
import pathlib
from typing import List

import pytest

from sympy.testing.runtests_pytest import (
    make_absolute_path,
    sympy_dir,
    update_args_with_paths,
)


class TestMakeAbsolutePath:

    @staticmethod
    @pytest.mark.parametrize(
        'partial_path', ['sympy', 'sympy/core', 'sympy/nonexistant_directory'],
    )
    def test_valid_partial_path(partial_path: str):
        """Paths that start with `sympy` are valid."""
        _ = make_absolute_path(partial_path)

    @staticmethod
    @pytest.mark.parametrize(
        'partial_path', ['not_sympy', 'also/not/sympy'],
    )
    def test_invalid_partial_path_raises_value_error(partial_path: str):
        """A `ValueError` is raises on paths that don't start with `sympy`."""
        with pytest.raises(ValueError):
            _ = make_absolute_path(partial_path)


class TestUpdateArgsWithPaths:

    @staticmethod
    def test_no_paths():
        """If no paths are passed, only `sympy` and `doc/src` are appended.

        `sympy` and `doc/src` are the `testpaths` stated in `pytest.ini`. They
        need to be manually added as if any path-related arguments are passed
        to `pytest.main` then the settings in `pytest.ini` may be ignored.

        """
        paths = []
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy')),
            str(pathlib.Path(sympy_dir(), 'doc/src')),
        ]
        assert args == expected

    @staticmethod
    @pytest.mark.parametrize(
        'path',
        ['sympy/core/tests/test_basic.py', '_basic']
    )
    def test_one_file(path: str):
        """Single files/paths, full or partial, are matched correctly."""
        args = update_args_with_paths(paths=[path], keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy/core/tests/test_basic.py')),
        ]
        assert args == expected

    @staticmethod
    def test_partial_path_from_root():
        """Partial paths from the root directly are matched correctly."""
        args = update_args_with_paths(paths=['sympy/functions'], keywords=None, args=[])
        expected = [str(pathlib.Path(sympy_dir(), 'sympy/functions'))]
        assert args == expected

    @staticmethod
    def test_multiple_paths_from_root():
        """Multiple paths, partial or full, are matched correctly."""
        paths = ['sympy/core/tests/test_basic.py', 'sympy/functions']
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        expected = [
            str(pathlib.Path(sympy_dir(), 'sympy/core/tests/test_basic.py')),
            str(pathlib.Path(sympy_dir(), 'sympy/functions')),
        ]
        assert args == expected

    @staticmethod
    @pytest.mark.parametrize(
        'paths, expected_paths',
        [
            (
                ['/core', '/util'],
                [
                    'doc/src/modules/utilities',
                    'doc/src/reference/public/utilities',
                    'sympy/core',
                    'sympy/logic/utilities',
                    'sympy/utilities',
                ]
            ),
        ]
    )
    def test_multiple_paths_from_non_root(paths: List[str], expected_paths: List[str]):
        """Multiple partial paths are matched correctly."""
        args = update_args_with_paths(paths=paths, keywords=None, args=[])
        assert len(args) == len(expected_paths)
        for arg, expected in zip(sorted(args), expected_paths):
            assert expected in arg

    @staticmethod
    @pytest.mark.parametrize(
        'paths',
        [

            [],
            ['sympy/physics'],
            ['sympy/physics/mechanics'],
            ['sympy/physics/mechanics/tests'],
            ['sympy/physics/mechanics/tests/test_kane3.py'],
        ]
    )
    def test_string_as_keyword(paths: List[str]):
        """String keywords are matched correctly."""
        keywords = ('bicycle', )
        args = update_args_with_paths(paths=paths, keywords=keywords, args=[])
        expected_args = ['sympy/physics/mechanics/tests/test_kane3.py::test_bicycle']
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    @pytest.mark.parametrize(
        'paths',
        [

            [],
            ['sympy/core'],
            ['sympy/core/tests'],
            ['sympy/core/tests/test_sympify.py'],
        ]
    )
    def test_integer_as_keyword(paths: List[str]):
        """Integer keywords are matched correctly."""
        keywords = ('3538', )
        args = update_args_with_paths(paths=paths, keywords=keywords, args=[])
        expected_args = ['sympy/core/tests/test_sympify.py::test_issue_3538']
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    def test_multiple_keywords():
        """Multiple keywords are matched correctly."""
        keywords = ('bicycle', '3538')
        args = update_args_with_paths(paths=[], keywords=keywords, args=[])
        expected_args = [
            'sympy/core/tests/test_sympify.py::test_issue_3538',
            'sympy/physics/mechanics/tests/test_kane3.py::test_bicycle',
        ]
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

    @staticmethod
    def test_keyword_match_in_multiple_files():
        """Keywords are matched across multiple files."""
        keywords = ('1130', )
        args = update_args_with_paths(paths=[], keywords=keywords, args=[])
        expected_args = [
            'sympy/integrals/tests/test_heurisch.py::test_heurisch_symbolic_coeffs_1130',
            'sympy/utilities/tests/test_lambdify.py::test_python_div_zero_issue_11306',
        ]
        assert len(args) == len(expected_args)
        for arg, expected in zip(sorted(args), expected_args):
            assert expected in arg

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_formatter.py ===
import pytest

from bs4.element import Tag
from bs4.formatter import (
    Formatter,
    HTMLFormatter,
    XMLFormatter,
)
from . import SoupTest


class TestFormatter(SoupTest):
    def test_default_attributes(self):
        # Test the default behavior of Formatter.attributes().
        formatter = Formatter()
        tag = Tag(name="tag")
        tag["b"] = "1"
        tag["a"] = "2"

        # Attributes come out sorted by name. In Python 3, attributes
        # normally come out of a dictionary in the order they were
        # added.
        assert [("a", "2"), ("b", "1")] == formatter.attributes(tag)

        # This works even if Tag.attrs is None, though this shouldn't
        # normally happen.
        tag.attrs = None
        assert [] == formatter.attributes(tag)

        assert " " == formatter.indent

    def test_sort_attributes(self):
        # Test the ability to override Formatter.attributes() to,
        # e.g., disable the normal sorting of attributes.
        class UnsortedFormatter(Formatter):
            def attributes(self, tag):
                self.called_with = tag
                for k, v in sorted(tag.attrs.items()):
                    if k == "ignore":
                        continue
                    yield k, v

        soup = self.soup('<p cval="1" aval="2" ignore="ignored"></p>')
        formatter = UnsortedFormatter()
        decoded = soup.decode(formatter=formatter)

        # attributes() was called on the <p> tag. It filtered out one
        # attribute and sorted the other two.
        assert formatter.called_with == soup.p
        assert '<p aval="2" cval="1"></p>' == decoded

    def test_empty_attributes_are_booleans(self):
        # Test the behavior of empty_attributes_are_booleans as well
        # as which Formatters have it enabled.

        for name in ("html", "minimal", None):
            formatter = HTMLFormatter.REGISTRY[name]
            assert False is formatter.empty_attributes_are_booleans

        formatter = XMLFormatter.REGISTRY[None]
        assert False is formatter.empty_attributes_are_booleans

        formatter = HTMLFormatter.REGISTRY["html5"]
        assert True is formatter.empty_attributes_are_booleans

        # Verify that the constructor sets the value.
        formatter = Formatter(empty_attributes_are_booleans=True)
        assert True is formatter.empty_attributes_are_booleans

        # Now demonstrate what it does to markup.
        for markup in ("<option selected></option>", '<option selected=""></option>'):
            soup = self.soup(markup)
            for formatter in ("html", "minimal", "xml", None):
                assert b'<option selected=""></option>' == soup.option.encode(
                    formatter="html"
                )
                assert b"<option selected></option>" == soup.option.encode(
                    formatter="html5"
                )

    @pytest.mark.parametrize(
        "indent,expect",
        [
            (None, "<a>\n<b>\ntext\n</b>\n</a>\n"),
            (-1, "<a>\n<b>\ntext\n</b>\n</a>\n"),
            (0, "<a>\n<b>\ntext\n</b>\n</a>\n"),
            ("", "<a>\n<b>\ntext\n</b>\n</a>\n"),
            (1, "<a>\n <b>\n  text\n </b>\n</a>\n"),
            (2, "<a>\n  <b>\n    text\n  </b>\n</a>\n"),
            ("\t", "<a>\n\t<b>\n\t\ttext\n\t</b>\n</a>\n"),
            ("abc", "<a>\nabc<b>\nabcabctext\nabc</b>\n</a>\n"),
            # Some invalid inputs -- the default behavior is used.
            (object(), "<a>\n <b>\n  text\n </b>\n</a>\n"),
            (b"bytes", "<a>\n <b>\n  text\n </b>\n</a>\n"),
        ],
    )
    def test_indent(self, indent, expect):
        # Pretty-print a tree with a Formatter set to
        # indent in a certain way and verify the results.
        soup = self.soup("<a><b>text</b></a>")
        formatter = Formatter(indent=indent)
        assert soup.prettify(formatter=formatter) == expect

        # Pretty-printing only happens with prettify(), not
        # encode().
        assert soup.encode(formatter=formatter) != expect

    def test_default_indent_value(self):
        formatter = Formatter()
        assert formatter.indent == " "

    @pytest.mark.parametrize("formatter,expect",
        [
            (HTMLFormatter(indent=1), "<p>\n a\n</p>\n"),
            (HTMLFormatter(indent=2), "<p>\n  a\n</p>\n"),
            (XMLFormatter(indent=1), "<p>\n a\n</p>\n"),
            (XMLFormatter(indent="\t"), "<p>\n\ta\n</p>\n"),
        ]                             )
    def test_indent_subclasses(self, formatter, expect):
        soup = self.soup("<p>a</p>")
        assert expect == soup.p.prettify(formatter=formatter)

    @pytest.mark.parametrize(
        "s,expect_html,expect_html5",
        [
            # The html5 formatter is much less aggressive about escaping ampersands
            # than the html formatter.
            ("foo & bar", "foo &amp; bar", "foo & bar"),
            ("foo&", "foo&amp;", "foo&"),
            ("foo&&& bar", "foo&amp;&amp;&amp; bar", "foo&&& bar"),
            ("x=1&y=2", "x=1&amp;y=2", "x=1&y=2"),
            ("&123", "&amp;123", "&123"),
            ("&abc", "&amp;abc", "&abc"),
            ("foo &0 bar", "foo &amp;0 bar", "foo &0 bar"),
            ("foo &lolwat bar", "foo &amp;lolwat bar", "foo &lolwat bar"),
            # But both formatters escape what the HTML5 spec considers ambiguous ampersands.
            ("&nosuchentity;", "&amp;nosuchentity;", "&amp;nosuchentity;"),
        ],
    )
    def test_entity_substitution(self, s, expect_html, expect_html5):
        assert HTMLFormatter.REGISTRY["html"].substitute(s) == expect_html
        assert HTMLFormatter.REGISTRY["html5"].substitute(s) == expect_html5
        assert HTMLFormatter.REGISTRY["html5-4.12"].substitute(s) == expect_html

    def test_entity_round_trip(self):
        # This is more an explanatory test and a way to avoid regressions than a test of functionality.

        markup = "<p>Some division signs: ÷ &divide; &#247; &#xf7;. These are made with: ÷ &amp;divide; &amp;#247;</p>"
        soup = self.soup(markup)
        assert (
            "Some division signs: ÷ ÷ ÷ ÷. These are made with: ÷ &divide; &#247;"
            == soup.p.string
        )

        # Oops, I forgot to mention the entity.
        soup.p.string = soup.p.string + " &#xf7;"

        assert (
            "Some division signs: ÷ ÷ ÷ ÷. These are made with: ÷ &divide; &#247; &#xf7;"
            == soup.p.string
        )

        expect = "<p>Some division signs: &divide; &divide; &divide; &divide;. These are made with: &divide; &amp;divide; &amp;#247; &amp;#xf7;</p>"
        assert expect == soup.p.decode(formatter="html")
        assert expect == soup.p.decode(formatter="html5")

        markup = "<p>a & b</p>"
        soup = self.soup(markup)
        assert "<p>a &amp; b</p>" == soup.p.decode(formatter="html")
        assert "<p>a & b</p>" == soup.p.decode(formatter="html5")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_util.py ===
import pytest
from sympy.core.numbers import Float
from sympy.core.function import (Derivative, Function)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions import exp, cos, sin, tan, cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Point, Point2D, Line, Polygon, Segment, convex_hull,\
    intersection, centroid, Point3D, Line3D, Ray, Ellipse
from sympy.geometry.util import idiff, closest_points, farthest_points, _ordered_points, are_coplanar
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises


def test_idiff():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    t = Symbol('t', real=True)
    f = Function('f')
    g = Function('g')
    # the use of idiff in ellipse also provides coverage
    circ = x**2 + y**2 - 4
    ans = -3*x*(x**2/y**2 + 1)/y**3
    assert ans == idiff(circ, y, x, 3), idiff(circ, y, x, 3)
    assert ans == idiff(circ, [y], x, 3)
    assert idiff(circ, y, x, 3) == ans
    explicit  = 12*x/sqrt(-x**2 + 4)**5
    assert ans.subs(y, solve(circ, y)[0]).equals(explicit)
    assert True in [sol.diff(x, 3).equals(explicit) for sol in solve(circ, y)]
    assert idiff(x + t + y, [y, t], x) == -Derivative(t, x) - 1
    assert idiff(f(x) * exp(f(x)) - x * exp(x), f(x), x) == (x + 1)*exp(x)*exp(-f(x))/(f(x) + 1)
    assert idiff(f(x) - y * exp(x), [f(x), y], x) == (y + Derivative(y, x))*exp(x)
    assert idiff(f(x) - y * exp(x), [y, f(x)], x) == -y + Derivative(f(x), x)*exp(-x)
    assert idiff(f(x) - g(x), [f(x), g(x)], x) == Derivative(g(x), x)
    # this should be fast
    fxy = y - (-10*(-sin(x) + 1/x)**2 + tan(x)**2 + 2*cosh(x/10))
    assert idiff(fxy, y, x) == -20*sin(x)*cos(x) + 2*tan(x)**3 + \
        2*tan(x) + sinh(x/10)/5 + 20*cos(x)/x - 20*sin(x)/x**2 + 20/x**3


def test_intersection():
    assert intersection(Point(0, 0)) == []
    raises(TypeError, lambda: intersection(Point(0, 0), 3))
    assert intersection(
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)),
            Line((0, 0), (0, 1)), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    assert intersection(
            Line((0, 0), (0, 1)),
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    assert intersection(
            Line((0, 0), (0, 1)),
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)),
            Line((0, 0), slope=1), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    R = 4.0
    c = intersection(
            Ray(Point2D(0.001, -1),
            Point2D(0.0008, -1.7)),
            Ellipse(center=Point2D(0, 0), hradius=R, vradius=2.0), pairwise=True)[0].coordinates
    assert c == pytest.approx(
            Point2D(0.000714285723396502, -1.99999996811224, evaluate=False).coordinates)
    # check this is responds to a lower precision parameter
    R = Float(4, 5)
    c2 = intersection(
            Ray(Point2D(0.001, -1),
            Point2D(0.0008, -1.7)),
            Ellipse(center=Point2D(0, 0), hradius=R, vradius=2.0), pairwise=True)[0].coordinates
    assert c2 == pytest.approx(
            Point2D(0.000714285723396502, -1.99999996811224, evaluate=False).coordinates)
    assert c[0]._prec == 53
    assert c2[0]._prec == 20


def test_convex_hull():
    raises(TypeError, lambda: convex_hull(Point(0, 0), 3))
    points = [(1, -1), (1, -2), (3, -1), (-5, -2), (15, -4)]
    assert convex_hull(*points, **{"polygon": False}) == (
        [Point2D(-5, -2), Point2D(1, -1), Point2D(3, -1), Point2D(15, -4)],
        [Point2D(-5, -2), Point2D(15, -4)])


def test_centroid():
    p = Polygon((0, 0), (10, 0), (10, 10))
    q = p.translate(0, 20)
    assert centroid(p, q) == Point(20, 40)/3
    p = Segment((0, 0), (2, 0))
    q = Segment((0, 0), (2, 2))
    assert centroid(p, q) == Point(1, -sqrt(2) + 2)
    assert centroid(Point(0, 0), Point(2, 0)) == Point(2, 0)/2
    assert centroid(Point(0, 0), Point(0, 0), Point(2, 0)) == Point(2, 0)/3


def test_farthest_points_closest_points():
    from sympy.core.random import randint
    from sympy.utilities.iterables import subsets

    for how in (min, max):
        if how == min:
            func = closest_points
        else:
            func = farthest_points

        raises(ValueError, lambda: func(Point2D(0, 0), Point2D(0, 0)))

        # 3rd pt dx is close and pt is closer to 1st pt
        p1 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 1)]
        # 3rd pt dx is close and pt is closer to 2nd pt
        p2 = [Point2D(0, 0), Point2D(3, 0), Point2D(2, 1)]
        # 3rd pt dx is close and but pt is not closer
        p3 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 10)]
        # 3rd pt dx is not closer and it's closer to 2nd pt
        p4 = [Point2D(0, 0), Point2D(3, 0), Point2D(4, 0)]
        # 3rd pt dx is not closer and it's closer to 1st pt
        p5 = [Point2D(0, 0), Point2D(3, 0), Point2D(-1, 0)]
        # duplicate point doesn't affect outcome
        dup = [Point2D(0, 0), Point2D(3, 0), Point2D(3, 0), Point2D(-1, 0)]
        # symbolic
        x = Symbol('x', positive=True)
        s = [Point2D(a) for a in ((x, 1), (x + 3, 2), (x + 2, 2))]

        for points in (p1, p2, p3, p4, p5, dup, s):
            d = how(i.distance(j) for i, j in subsets(set(points), 2))
            ans = a, b = list(func(*points))[0]
            assert a.distance(b) == d
            assert ans == _ordered_points(ans)

        # if the following ever fails, the above tests were not sufficient
        # and the logical error in the routine should be fixed
        points = set()
        while len(points) != 7:
            points.add(Point2D(randint(1, 100), randint(1, 100)))
        points = list(points)
        d = how(i.distance(j) for i, j in subsets(points, 2))
        ans = a, b = list(func(*points))[0]
        assert a.distance(b) == d
        assert ans == _ordered_points(ans)

    # equidistant points
    a, b, c = (
        Point2D(0, 0), Point2D(1, 0), Point2D(S.Half, sqrt(3)/2))
    ans = {_ordered_points((i, j))
        for i, j in subsets((a, b, c), 2)}
    assert closest_points(b, c, a) == ans
    assert farthest_points(b, c, a) == ans

    # unique to farthest
    points = [(1, 1), (1, 2), (3, 1), (-5, 2), (15, 4)]
    assert farthest_points(*points) == {
        (Point2D(-5, 2), Point2D(15, 4))}
    points = [(1, -1), (1, -2), (3, -1), (-5, -2), (15, -4)]
    assert farthest_points(*points) == {
        (Point2D(-5, -2), Point2D(15, -4))}
    assert farthest_points((1, 1), (0, 0)) == {
        (Point2D(0, 0), Point2D(1, 1))}
    raises(ValueError, lambda: farthest_points((1, 1)))


def test_are_coplanar():
    a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    d = Line(Point2D(0, 3), Point2D(1, 5))

    assert are_coplanar(a, b, c) == False
    assert are_coplanar(a, d) == False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_shift.py ===
from datetime import datetime

import pytest
import pytz

from pandas.errors import NullFrequencyError

import pandas as pd
from pandas import (
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestDatetimeIndexShift:
    # -------------------------------------------------------------
    # DatetimeIndex.shift is used in integer addition

    def test_dti_shift_tzaware(self, tz_naive_fixture, unit):
        # GH#9903
        tz = tz_naive_fixture
        idx = DatetimeIndex([], name="xxx", tz=tz).as_unit(unit)
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        tm.assert_index_equal(idx.shift(3, freq="h"), idx)

        idx = DatetimeIndex(
            ["2011-01-01 10:00", "2011-01-01 11:00", "2011-01-01 12:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        tm.assert_index_equal(idx.shift(0, freq="h"), idx)
        exp = DatetimeIndex(
            ["2011-01-01 13:00", "2011-01-01 14:00", "2011-01-01 15:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        tm.assert_index_equal(idx.shift(3, freq="h"), exp)
        exp = DatetimeIndex(
            ["2011-01-01 07:00", "2011-01-01 08:00", "2011-01-01 09:00"],
            name="xxx",
            tz=tz,
            freq="h",
        ).as_unit(unit)
        tm.assert_index_equal(idx.shift(-3, freq="h"), exp)

    def test_dti_shift_freqs(self, unit):
        # test shift for DatetimeIndex and non DatetimeIndex
        # GH#8083
        drange = date_range("20130101", periods=5, unit=unit)
        result = drange.shift(1)
        expected = DatetimeIndex(
            ["2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05", "2013-01-06"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)

        result = drange.shift(-1)
        expected = DatetimeIndex(
            ["2012-12-31", "2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)

        result = drange.shift(3, freq="2D")
        expected = DatetimeIndex(
            ["2013-01-07", "2013-01-08", "2013-01-09", "2013-01-10", "2013-01-11"],
            dtype=f"M8[{unit}]",
            freq="D",
        )
        tm.assert_index_equal(result, expected)

    def test_dti_shift_int(self, unit):
        rng = date_range("1/1/2000", periods=20, unit=unit)

        result = rng + 5 * rng.freq
        expected = rng.shift(5)
        tm.assert_index_equal(result, expected)

        result = rng - 5 * rng.freq
        expected = rng.shift(-5)
        tm.assert_index_equal(result, expected)

    def test_dti_shift_no_freq(self, unit):
        # GH#19147
        dti = DatetimeIndex(["2011-01-01 10:00", "2011-01-01"], freq=None).as_unit(unit)
        with pytest.raises(NullFrequencyError, match="Cannot shift with no freq"):
            dti.shift(2)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_shift_localized(self, tzstr, unit):
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI", unit=unit)
        dr_tz = dr.tz_localize(tzstr)

        result = dr_tz.shift(1, "10min")
        assert result.tz == dr_tz.tz

    def test_dti_shift_across_dst(self, unit):
        # GH 8616
        idx = date_range(
            "2013-11-03", tz="America/Chicago", periods=7, freq="h", unit=unit
        )
        ser = Series(index=idx[:-1], dtype=object)
        result = ser.shift(freq="h")
        expected = Series(index=idx[1:], dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "shift, result_time",
        [
            [0, "2014-11-14 00:00:00"],
            [-1, "2014-11-13 23:00:00"],
            [1, "2014-11-14 01:00:00"],
        ],
    )
    def test_dti_shift_near_midnight(self, shift, result_time, unit):
        # GH 8616
        dt = datetime(2014, 11, 14, 0)
        dt_est = pytz.timezone("EST").localize(dt)
        idx = DatetimeIndex([dt_est]).as_unit(unit)
        ser = Series(data=[1], index=idx)
        result = ser.shift(shift, freq="h")
        exp_index = DatetimeIndex([result_time], tz="EST").as_unit(unit)
        expected = Series(1, index=exp_index)
        tm.assert_series_equal(result, expected)

    def test_shift_periods(self, unit):
        # GH#22458 : argument 'n' was deprecated in favor of 'periods'
        idx = date_range(start=START, end=END, periods=3, unit=unit)
        tm.assert_index_equal(idx.shift(periods=0), idx)
        tm.assert_index_equal(idx.shift(0), idx)

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_shift_bday(self, freq, unit):
        rng = date_range(START, END, freq=freq, unit=unit)
        shifted = rng.shift(5)
        assert shifted[0] == rng[5]
        assert shifted.freq == rng.freq

        shifted = rng.shift(-5)
        assert shifted[5] == rng[0]
        assert shifted.freq == rng.freq

        shifted = rng.shift(0)
        assert shifted[0] == rng[0]
        assert shifted.freq == rng.freq

    def test_shift_bmonth(self, unit):
        rng = date_range(START, END, freq=pd.offsets.BMonthEnd(), unit=unit)
        shifted = rng.shift(1, freq=pd.offsets.BDay())
        assert shifted[0] == rng[0] + pd.offsets.BDay()

        rng = date_range(START, END, freq=pd.offsets.BMonthEnd(), unit=unit)
        with tm.assert_produces_warning(pd.errors.PerformanceWarning):
            shifted = rng.shift(1, freq=pd.offsets.CDay())
            assert shifted[0] == rng[0] + pd.offsets.CDay()

    def test_shift_empty(self, unit):
        # GH#14811
        dti = date_range(start="2016-10-21", end="2016-10-21", freq="BME", unit=unit)
        result = dti.shift(1)
        tm.assert_index_equal(result, dti)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_unstack.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


def test_unstack_preserves_object():
    mi = MultiIndex.from_product([["bar", "foo"], ["one", "two"]])

    ser = Series(np.arange(4.0), index=mi, dtype=object)

    res1 = ser.unstack()
    assert (res1.dtypes == object).all()

    res2 = ser.unstack(level=0)
    assert (res2.dtypes == object).all()


def test_unstack():
    index = MultiIndex(
        levels=[["bar", "foo"], ["one", "three", "two"]],
        codes=[[1, 1, 0, 0], [0, 1, 0, 2]],
    )

    s = Series(np.arange(4.0), index=index)
    unstacked = s.unstack()

    expected = DataFrame(
        [[2.0, np.nan, 3.0], [0.0, 1.0, np.nan]],
        index=["bar", "foo"],
        columns=["one", "three", "two"],
    )

    tm.assert_frame_equal(unstacked, expected)

    unstacked = s.unstack(level=0)
    tm.assert_frame_equal(unstacked, expected.T)

    index = MultiIndex(
        levels=[["bar"], ["one", "two", "three"], [0, 1]],
        codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    s = Series(np.random.default_rng(2).standard_normal(6), index=index)
    exp_index = MultiIndex(
        levels=[["one", "two", "three"], [0, 1]],
        codes=[[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    expected = DataFrame({"bar": s.values}, index=exp_index).sort_index(level=0)
    unstacked = s.unstack(0).sort_index()
    tm.assert_frame_equal(unstacked, expected)

    # GH5873
    idx = MultiIndex.from_arrays([[101, 102], [3.5, np.nan]])
    ts = Series([1, 2], index=idx)
    left = ts.unstack()
    right = DataFrame(
        [[np.nan, 1], [2, np.nan]], index=[101, 102], columns=[np.nan, 3.5]
    )
    tm.assert_frame_equal(left, right)

    idx = MultiIndex.from_arrays(
        [
            ["cat", "cat", "cat", "dog", "dog"],
            ["a", "a", "b", "a", "b"],
            [1, 2, 1, 1, np.nan],
        ]
    )
    ts = Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
    right = DataFrame(
        [[1.0, 1.3], [1.1, np.nan], [np.nan, 1.4], [1.2, np.nan]],
        columns=["cat", "dog"],
    )
    tpls = [("a", 1), ("a", 2), ("b", np.nan), ("b", 1)]
    right.index = MultiIndex.from_tuples(tpls)
    tm.assert_frame_equal(ts.unstack(level=0), right)


def test_unstack_tuplename_in_multiindex():
    # GH 19966
    idx = MultiIndex.from_product(
        [["a", "b", "c"], [1, 2, 3]], names=[("A", "a"), ("B", "b")]
    )
    ser = Series(1, index=idx)
    result = ser.unstack(("A", "a"))

    expected = DataFrame(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        columns=MultiIndex.from_tuples([("a",), ("b",), ("c",)], names=[("A", "a")]),
        index=Index([1, 2, 3], name=("B", "b")),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "unstack_idx, expected_values, expected_index, expected_columns",
    [
        (
            ("A", "a"),
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=["B", "C"]),
            MultiIndex.from_tuples([("a",), ("b",)], names=[("A", "a")]),
        ),
        (
            (("A", "a"), "B"),
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            Index([3, 4], name="C"),
            MultiIndex.from_tuples(
                [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=[("A", "a"), "B"]
            ),
        ),
    ],
)
def test_unstack_mixed_type_name_in_multiindex(
    unstack_idx, expected_values, expected_index, expected_columns
):
    # GH 19966
    idx = MultiIndex.from_product(
        [["a", "b"], [1, 2], [3, 4]], names=[("A", "a"), "B", "C"]
    )
    ser = Series(1, index=idx)
    result = ser.unstack(unstack_idx)

    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)


def test_unstack_multi_index_categorical_values():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    mi = df.stack(future_stack=True).index.rename(["major", "minor"])
    ser = Series(["foo"] * len(mi), index=mi, name="category", dtype="category")

    result = ser.unstack()

    dti = ser.index.levels[0]
    c = pd.Categorical(["foo"] * len(dti))
    expected = DataFrame(
        {"A": c.copy(), "B": c.copy(), "C": c.copy(), "D": c.copy()},
        columns=Index(list("ABCD"), name="minor"),
        index=dti.rename("major"),
    )
    tm.assert_frame_equal(result, expected)


def test_unstack_mixed_level_names():
    # GH#48763
    arrays = [["a", "a"], [1, 2], ["red", "blue"]]
    idx = MultiIndex.from_arrays(arrays, names=("x", 0, "y"))
    ser = Series([1, 2], index=idx)
    result = ser.unstack("x")
    expected = DataFrame(
        [[1], [2]],
        columns=Index(["a"], name="x"),
        index=MultiIndex.from_tuples([(1, "red"), (2, "blue")], names=[0, "y"]),
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_generator_nested.py ===

from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck

class genlet(greenlet):
    parent = None
    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds
        self.child = None

    def run(self):
        # Note the function is packed in a tuple
        # to avoid creating a bound method for it.
        fn, = self.fn
        fn(*self.args, **self.kwds)

    def __iter__(self):
        return self

    def set_child(self, child):
        self.child = child

    def __next__(self):
        if self.child:
            child = self.child
            while child.child:
                tmp = child
                child = child.child
                tmp.child = None

            result = child.switch()
        else:
            self.parent = greenlet.getcurrent()
            result = self.switch()

        if self:
            return result

        raise StopIteration

    next = __next__

def Yield(value, level=1):
    g = greenlet.getcurrent()

    while level != 0:
        if not isinstance(g, genlet):
            raise RuntimeError('yield outside a genlet')
        if level > 1:
            g.parent.set_child(g)
        g = g.parent
        level -= 1

    g.switch(value)


def Genlet(func):
    class TheGenlet(genlet):
        fn = (func,)
    return TheGenlet

# ____________________________________________________________


def g1(n, seen):
    for i in range(n):
        seen.append(i + 1)
        yield i


def g2(n, seen):
    for i in range(n):
        seen.append(i + 1)
        Yield(i)

g2 = Genlet(g2)


def nested(i):
    Yield(i)


def g3(n, seen):
    for i in range(n):
        seen.append(i + 1)
        nested(i)
g3 = Genlet(g3)


def a(n):
    if n == 0:
        return
    for ii in ax(n - 1):
        Yield(ii)
    Yield(n)
ax = Genlet(a)


def perms(l):
    if len(l) > 1:
        for e in l:
            # No syntactical sugar for generator expressions
            x = [Yield([e] + p) for p in perms([x for x in l if x != e])]
            assert x
    else:
        Yield(l)
perms = Genlet(perms)


def gr1(n):
    for ii in range(1, n):
        Yield(ii)
        Yield(ii * ii, 2)

gr1 = Genlet(gr1)


def gr2(n, seen):
    for ii in gr1(n):
        seen.append(ii)

gr2 = Genlet(gr2)


class NestedGeneratorTests(TestCase):
    def test_layered_genlets(self):
        seen = []
        for ii in gr2(5, seen):
            seen.append(ii)
        self.assertEqual(seen, [1, 1, 2, 4, 3, 9, 4, 16])

    @fails_leakcheck
    def test_permutations(self):
        gen_perms = perms(list(range(4)))
        permutations = list(gen_perms)
        self.assertEqual(len(permutations), 4 * 3 * 2 * 1)
        self.assertIn([0, 1, 2, 3], permutations)
        self.assertIn([3, 2, 1, 0], permutations)
        res = []
        for ii in zip(perms(list(range(4))), perms(list(range(3)))):
            res.append(ii)
        self.assertEqual(
            res,
            [([0, 1, 2, 3], [0, 1, 2]), ([0, 1, 3, 2], [0, 2, 1]),
             ([0, 2, 1, 3], [1, 0, 2]), ([0, 2, 3, 1], [1, 2, 0]),
             ([0, 3, 1, 2], [2, 0, 1]), ([0, 3, 2, 1], [2, 1, 0])])
        # XXX Test to make sure we are working as a generator expression

    def test_genlet_simple(self):
        for g in g1, g2, g3:
            seen = []
            for _ in range(3):
                for j in g(5, seen):
                    seen.append(j)
            self.assertEqual(seen, 3 * [1, 0, 2, 1, 3, 2, 4, 3, 5, 4])

    def test_genlet_bad(self):
        try:
            Yield(10)
        except RuntimeError:
            pass

    def test_nested_genlets(self):
        seen = []
        for ii in ax(5):
            seen.append(ii)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\simple.py ===
"""Simple expression that should pass with mypy."""
import operator

import numpy as np
import numpy.typing as npt
from collections.abc import Iterable

# Basic checks
array = np.array([1, 2])


def ndarray_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x


ndarray_func(np.array([1, 2], dtype=np.float64))
array == 1
array.dtype == float

# Dtype construction
np.dtype(float)
np.dtype(np.float64)
np.dtype(None)
np.dtype("float64")
np.dtype(np.dtype(float))
np.dtype(("U", 10))
np.dtype((np.int32, (2, 2)))
# Define the arguments on the previous line to prevent bidirectional
# type inference in mypy from broadening the types.
two_tuples_dtype = [("R", "u1"), ("G", "u1"), ("B", "u1")]
np.dtype(two_tuples_dtype)

three_tuples_dtype = [("R", "u1", 2)]
np.dtype(three_tuples_dtype)

mixed_tuples_dtype = [("R", "u1"), ("G", np.str_, 1)]
np.dtype(mixed_tuples_dtype)

shape_tuple_dtype = [("R", "u1", (2, 2))]
np.dtype(shape_tuple_dtype)

shape_like_dtype = [("R", "u1", (2, 2)), ("G", np.str_, 1)]
np.dtype(shape_like_dtype)

object_dtype = [("field1", object)]
np.dtype(object_dtype)

np.dtype((np.int32, (np.int8, 4)))

# Dtype comparison
np.dtype(float) == float
np.dtype(float) != np.float64
np.dtype(float) < None
np.dtype(float) <= "float64"
np.dtype(float) > np.dtype(float)
np.dtype(float) >= np.dtype(("U", 10))

# Iteration and indexing
def iterable_func(x: Iterable[object]) -> Iterable[object]:
    return x


iterable_func(array)
list(array)
iter(array)
zip(array, array)
array[1]
array[:]
array[...]
array[:] = 0

array_2d = np.ones((3, 3))
array_2d[:2, :2]
array_2d[:2, :2] = 0
array_2d[..., 0]
array_2d[..., 0] = 2
array_2d[-1, -1] = None

array_obj = np.zeros(1, dtype=np.object_)
array_obj[0] = slice(None)

# Other special methods
len(array)
str(array)
array_scalar = np.array(1)
int(array_scalar)
float(array_scalar)
complex(array_scalar)
bytes(array_scalar)
operator.index(array_scalar)
bool(array_scalar)

# comparisons
array < 1
array <= 1
array == 1
array != 1
array > 1
array >= 1
1 < array
1 <= array
1 == array
1 != array
1 > array
1 >= array

# binary arithmetic
array + 1
1 + array
array += 1

array - 1
1 - array
array -= 1

array * 1
1 * array
array *= 1

nonzero_array = np.array([1, 2])
array / 1
1 / nonzero_array
float_array = np.array([1.0, 2.0])
float_array /= 1

array // 1
1 // nonzero_array
array //= 1

array % 1
1 % nonzero_array
array %= 1

divmod(array, 1)
divmod(1, nonzero_array)

array ** 1
1 ** array
array **= 1

array << 1
1 << array
array <<= 1

array >> 1
1 >> array
array >>= 1

array & 1
1 & array
array &= 1

array ^ 1
1 ^ array
array ^= 1

array | 1
1 | array
array |= 1

# unary arithmetic
-array
+array
abs(array)
~array

# Other methods
np.array([1, 2]).transpose()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\numeric\test_setops.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas._testing as tm
from pandas.core.indexes.api import (
    Index,
    RangeIndex,
)


@pytest.fixture
def index_large():
    # large values used in TestUInt64Index where no compat needed with int64/float64
    large = [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25]
    return Index(large, dtype=np.uint64)


class TestSetOps:
    @pytest.mark.parametrize("dtype", ["f8", "u8", "i8"])
    def test_union_non_numeric(self, dtype):
        # corner case, non-numeric
        index = Index(np.arange(5, dtype=dtype), dtype=dtype)
        assert index.dtype == dtype

        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        result = index.union(other)
        expected = Index(np.concatenate((index, other)))
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        expected = Index(np.concatenate((other, index)))
        tm.assert_index_equal(result, expected)

    def test_intersection(self):
        index = Index(range(5), dtype=np.int64)

        other = Index([1, 2, 3, 4, 5])
        result = index.intersection(other)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "uint64"])
    def test_int_float_union_dtype(self, dtype):
        # https://github.com/pandas-dev/pandas/issues/26778
        # [u]int | float -> float
        index = Index([0, 2, 3], dtype=dtype)
        other = Index([0.5, 1.5], dtype=np.float64)
        expected = Index([0.0, 0.5, 1.5, 2.0, 3.0], dtype=np.float64)
        result = index.union(other)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_range_float_union_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/26778
        index = RangeIndex(start=0, stop=3)
        other = Index([0.5, 1.5], dtype=np.float64)
        result = index.union(other)
        expected = Index([0.0, 0.5, 1, 1.5, 2.0], dtype=np.float64)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_range_uint64_union_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/26778
        index = RangeIndex(start=0, stop=3)
        other = Index([0, 10], dtype=np.uint64)
        result = index.union(other)
        expected = Index([0, 1, 2, 10], dtype=object)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_float64_index_difference(self):
        # https://github.com/pandas-dev/pandas/issues/35217
        float_index = Index([1.0, 2, 3])
        string_index = Index(["1", "2", "3"])

        result = float_index.difference(string_index)
        tm.assert_index_equal(result, float_index)

        result = string_index.difference(float_index)
        tm.assert_index_equal(result, string_index)

    def test_intersection_uint64_outside_int64_range(self, index_large):
        other = Index([2**63, 2**63 + 5, 2**63 + 10, 2**63 + 15, 2**63 + 20])
        result = index_large.intersection(other)
        expected = Index(np.sort(np.intersect1d(index_large.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index_large)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index_large.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index2,keeps_name",
        [
            (Index([4, 7, 6, 5, 3], name="index"), True),
            (Index([4, 7, 6, 5, 3], name="other"), False),
        ],
    )
    def test_intersection_monotonic(self, index2, keeps_name, sort):
        index1 = Index([5, 3, 2, 4, 1], name="index")
        expected = Index([5, 3, 4])

        if keeps_name:
            expected.name = "index"

        result = index1.intersection(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self, sort):
        # smoke
        index1 = Index([5, 2, 3, 4], name="index1")
        index2 = Index([2, 3, 4, 1])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = Index([5, 1])
        if sort is not None:
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result, expected.sort_values())
        assert result.name is None
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)


class TestSetOpsSort:
    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_other_special(self, slice_):
        # https://github.com/pandas-dev/pandas/issues/24959

        idx = Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]
        tm.assert_index_equal(idx.union(other), idx)
        tm.assert_index_equal(other.union(idx), idx)

        # sort=False
        tm.assert_index_equal(idx.union(other, sort=False), idx)

    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_special_true(self, slice_):
        idx = Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]

        result = idx.union(other, sort=True)
        expected = Index([0, 1, 2])
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_timezones.py ===
from datetime import (
    datetime,
    timedelta,
    timezone,
)

import dateutil.tz
import pytest
import pytz

from pandas._libs.tslibs import (
    conversion,
    timezones,
)
from pandas.compat import is_platform_windows

from pandas import Timestamp


def test_is_utc(utc_fixture):
    tz = timezones.maybe_get_tz(utc_fixture)
    assert timezones.is_utc(tz)


@pytest.mark.parametrize("tz_name", list(pytz.common_timezones))
def test_cache_keys_are_distinct_for_pytz_vs_dateutil(tz_name):
    tz_p = timezones.maybe_get_tz(tz_name)
    tz_d = timezones.maybe_get_tz("dateutil/" + tz_name)

    if tz_d is None:
        pytest.skip(tz_name + ": dateutil does not know about this one")

    if not (tz_name == "UTC" and is_platform_windows()):
        # they both end up as tzwin("UTC") on windows
        assert timezones._p_tz_cache_key(tz_p) != timezones._p_tz_cache_key(tz_d)


def test_tzlocal_repr():
    # see gh-13583
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal())
    assert ts.tz == dateutil.tz.tzlocal()
    assert "tz='tzlocal()')" in repr(ts)


def test_tzlocal_maybe_get_tz():
    # see gh-13583
    tz = timezones.maybe_get_tz("tzlocal()")
    assert tz == dateutil.tz.tzlocal()


def test_tzlocal_offset():
    # see gh-13583
    #
    # Get offset using normal datetime for test.
    ts = Timestamp("2011-01-01", tz=dateutil.tz.tzlocal())

    offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
    offset = offset.total_seconds()

    assert ts._value + offset == Timestamp("2011-01-01")._value


def test_tzlocal_is_not_utc():
    # even if the machine running the test is localized to UTC
    tz = dateutil.tz.tzlocal()
    assert not timezones.is_utc(tz)

    assert not timezones.tz_compare(tz, dateutil.tz.tzutc())


def test_tz_compare_utc(utc_fixture, utc_fixture2):
    tz = timezones.maybe_get_tz(utc_fixture)
    tz2 = timezones.maybe_get_tz(utc_fixture2)
    assert timezones.tz_compare(tz, tz2)


@pytest.fixture(
    params=[
        (pytz.timezone("US/Eastern"), lambda tz, x: tz.localize(x)),
        (dateutil.tz.gettz("US/Eastern"), lambda tz, x: x.replace(tzinfo=tz)),
    ]
)
def infer_setup(request):
    eastern, localize = request.param

    start_naive = datetime(2001, 1, 1)
    end_naive = datetime(2009, 1, 1)

    start = localize(eastern, start_naive)
    end = localize(eastern, end_naive)

    return eastern, localize, start, end, start_naive, end_naive


def test_infer_tz_compat(infer_setup):
    eastern, _, start, end, start_naive, end_naive = infer_setup

    assert (
        timezones.infer_tzinfo(start, end)
        is conversion.localize_pydatetime(start_naive, eastern).tzinfo
    )
    assert (
        timezones.infer_tzinfo(start, None)
        is conversion.localize_pydatetime(start_naive, eastern).tzinfo
    )
    assert (
        timezones.infer_tzinfo(None, end)
        is conversion.localize_pydatetime(end_naive, eastern).tzinfo
    )


def test_infer_tz_utc_localize(infer_setup):
    _, _, start, end, start_naive, end_naive = infer_setup
    utc = pytz.utc

    start = utc.localize(start_naive)
    end = utc.localize(end_naive)

    assert timezones.infer_tzinfo(start, end) is utc


@pytest.mark.parametrize("ordered", [True, False])
def test_infer_tz_mismatch(infer_setup, ordered):
    eastern, _, _, _, start_naive, end_naive = infer_setup
    msg = "Inputs must both have the same timezone"

    utc = pytz.utc
    start = utc.localize(start_naive)
    end = conversion.localize_pydatetime(end_naive, eastern)

    args = (start, end) if ordered else (end, start)

    with pytest.raises(AssertionError, match=msg):
        timezones.infer_tzinfo(*args)


def test_maybe_get_tz_invalid_types():
    with pytest.raises(TypeError, match="<class 'float'>"):
        timezones.maybe_get_tz(44.0)

    with pytest.raises(TypeError, match="<class 'module'>"):
        timezones.maybe_get_tz(pytz)

    msg = "<class 'pandas._libs.tslibs.timestamps.Timestamp'>"
    with pytest.raises(TypeError, match=msg):
        timezones.maybe_get_tz(Timestamp("2021-01-01", tz="UTC"))


def test_maybe_get_tz_offset_only():
    # see gh-36004

    # timezone.utc
    tz = timezones.maybe_get_tz(timezone.utc)
    assert tz == timezone(timedelta(hours=0, minutes=0))

    # without UTC+- prefix
    tz = timezones.maybe_get_tz("+01:15")
    assert tz == timezone(timedelta(hours=1, minutes=15))

    tz = timezones.maybe_get_tz("-01:15")
    assert tz == timezone(-timedelta(hours=1, minutes=15))

    # with UTC+- prefix
    tz = timezones.maybe_get_tz("UTC+02:45")
    assert tz == timezone(timedelta(hours=2, minutes=45))

    tz = timezones.maybe_get_tz("UTC-02:45")
    assert tz == timezone(-timedelta(hours=2, minutes=45))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\fft\tests\test_helper.py ===
"""Test functions for fftpack.helper module

Copied from fftpack.helper by Pearu Peterson, October 2005

"""
import numpy as np
from numpy import fft, pi
from numpy.testing import assert_array_almost_equal


class TestFFTShift:

    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)

    def test_inverse(self):
        for n in [1, 4, 9, 100, 211]:
            x = np.random.random((n,))
            assert_array_almost_equal(fft.ifftshift(fft.fftshift(x)), x)

    def test_axes_keyword(self):
        freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
        shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shifted)
        assert_array_almost_equal(fft.fftshift(freqs, axes=0),
                                  fft.fftshift(freqs, axes=(0,)))
        assert_array_almost_equal(fft.ifftshift(shifted, axes=(0, 1)), freqs)
        assert_array_almost_equal(fft.ifftshift(shifted, axes=0),
                                  fft.ifftshift(shifted, axes=(0,)))

        assert_array_almost_equal(fft.fftshift(freqs), shifted)
        assert_array_almost_equal(fft.ifftshift(shifted), freqs)

    def test_uneven_dims(self):
        """ Test 2D input, which has uneven dimension sizes """
        freqs = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        # shift in dimension 0
        shift_dim0 = [
            [4, 5],
            [0, 1],
            [2, 3]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=0), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=0), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # shift in dimension 1
        shift_dim1 = [
            [1, 0],
            [3, 2],
            [5, 4]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=1), shift_dim1)
        assert_array_almost_equal(fft.ifftshift(shift_dim1, axes=1), freqs)

        # shift in both dimensions
        shift_dim_both = [
            [5, 4],
            [1, 0],
            [3, 2]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # axes=None (default) shift in all dimensions
        assert_array_almost_equal(fft.fftshift(freqs, axes=None), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=None), freqs)
        assert_array_almost_equal(fft.fftshift(freqs), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both), freqs)

    def test_equal_to_original(self):
        """ Test the new (>=v1.15) and old implementations are equal (see #10073) """
        from numpy._core import arange, asarray, concatenate, take

        def original_fftshift(x, axes=None):
            """ How fftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        def original_ifftshift(x, axes=None):
            """ How ifftshift was implemented in v1.14 """
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = n - (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        # create possible 2d array combinations and try all possible keywords
        # compare output to original functions
        for i in range(16):
            for j in range(16):
                for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                    inp = np.random.rand(i, j)

                    assert_array_almost_equal(fft.fftshift(inp, axes_keyword),
                                              original_fftshift(inp, axes_keyword))

                    assert_array_almost_equal(fft.ifftshift(inp, axes_keyword),
                                              original_ifftshift(inp, axes_keyword))


class TestFFTFreq:

    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        assert_array_almost_equal(9 * fft.fftfreq(9), x)
        assert_array_almost_equal(9 * pi * fft.fftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        assert_array_almost_equal(10 * fft.fftfreq(10), x)
        assert_array_almost_equal(10 * pi * fft.fftfreq(10, pi), x)


class TestRFFTFreq:

    def test_definition(self):
        x = [0, 1, 2, 3, 4]
        assert_array_almost_equal(9 * fft.rfftfreq(9), x)
        assert_array_almost_equal(9 * pi * fft.rfftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, 5]
        assert_array_almost_equal(10 * fft.rfftfreq(10), x)
        assert_array_almost_equal(10 * pi * fft.rfftfreq(10, pi), x)


class TestIRFFTN:

    def test_not_last_axis_success(self):
        ar, ai = np.random.random((2, 16, 8, 32))
        a = ar + 1j * ai

        axes = (-2,)

        # Should not raise error
        fft.irfftn(a, axes=axes)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_item_selection.py ===
import sys

import pytest

import numpy as np
from numpy.testing import HAS_REFCOUNT, assert_, assert_array_equal, assert_raises


class TestTake:
    def test_simple(self):
        a = [[1, 2], [3, 4]]
        a_str = [[b'1', b'2'], [b'3', b'4']]
        modes = ['raise', 'wrap', 'clip']
        indices = [-1, 4]
        index_arrays = [np.empty(0, dtype=np.intp),
                        np.empty((), dtype=np.intp),
                        np.empty((1, 1), dtype=np.intp)]
        real_indices = {'raise': {-1: 1, 4: IndexError},
                        'wrap': {-1: 1, 4: 0},
                        'clip': {-1: 0, 4: 1}}
        # Currently all types but object, use the same function generation.
        # So it should not be necessary to test all. However test also a non
        # refcounted struct on top of object, which has a size that hits the
        # default (non-specialized) path.
        types = int, object, np.dtype([('', 'i2', 3)])
        for t in types:
            # ta works, even if the array may be odd if buffer interface is used
            ta = np.array(a if np.issubdtype(t, np.number) else a_str, dtype=t)
            tresult = list(ta.T.copy())
            for index_array in index_arrays:
                if index_array.size != 0:
                    tresult[0].shape = (2,) + index_array.shape
                    tresult[1].shape = (2,) + index_array.shape
                for mode in modes:
                    for index in indices:
                        real_index = real_indices[mode][index]
                        if real_index is IndexError and index_array.size != 0:
                            index_array.put(0, index)
                            assert_raises(IndexError, ta.take, index_array,
                                          mode=mode, axis=1)
                        elif index_array.size != 0:
                            index_array.put(0, index)
                            res = ta.take(index_array, mode=mode, axis=1)
                            assert_array_equal(res, tresult[real_index])
                        else:
                            res = ta.take(index_array, mode=mode, axis=1)
                            assert_(res.shape == (2,) + index_array.shape)

    def test_refcounting(self):
        objects = [object() for i in range(10)]
        if HAS_REFCOUNT:
            orig_rcs = [sys.getrefcount(o) for o in objects]
        for mode in ('raise', 'clip', 'wrap'):
            a = np.array(objects)
            b = np.array([2, 2, 4, 5, 3, 5])
            a.take(b, out=a[:6], mode=mode)
            del a
            if HAS_REFCOUNT:
                assert_(all(sys.getrefcount(o) == rc + 1
                            for o, rc in zip(objects, orig_rcs)))
            # not contiguous, example:
            a = np.array(objects * 2)[::2]
            a.take(b, out=a[:6], mode=mode)
            del a
            if HAS_REFCOUNT:
                assert_(all(sys.getrefcount(o) == rc + 1
                            for o, rc in zip(objects, orig_rcs)))

    def test_unicode_mode(self):
        d = np.arange(10)
        k = b'\xc3\xa4'.decode("UTF8")
        assert_raises(ValueError, d.take, 5, mode=k)

    def test_empty_partition(self):
        # In reference to github issue #6530
        a_original = np.array([0, 2, 4, 6, 8, 10])
        a = a_original.copy()

        # An empty partition should be a successful no-op
        a.partition(np.array([], dtype=np.int16))

        assert_array_equal(a, a_original)

    def test_empty_argpartition(self):
        # In reference to github issue #6530
        a = np.array([0, 2, 4, 6, 8, 10])
        a = a.argpartition(np.array([], dtype=np.int16))

        b = np.array([0, 1, 2, 3, 4, 5])
        assert_array_equal(a, b)


class TestPutMask:
    @pytest.mark.parametrize("dtype", list(np.typecodes["All"]) + ["i,O"])
    def test_simple(self, dtype):
        if dtype.lower() == "m":
            dtype += "8[ns]"

        # putmask is weird and doesn't care about value length (even shorter)
        vals = np.arange(1001).astype(dtype=dtype)

        mask = np.random.randint(2, size=1000).astype(bool)
        # Use vals.dtype in case of flexible dtype (i.e. string)
        arr = np.zeros(1000, dtype=vals.dtype)
        zeros = arr.copy()

        np.putmask(arr, mask, vals)
        assert_array_equal(arr[mask], vals[:len(mask)][mask])
        assert_array_equal(arr[~mask], zeros[~mask])

    @pytest.mark.parametrize("dtype", list(np.typecodes["All"])[1:] + ["i,O"])
    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_empty(self, dtype, mode):
        arr = np.zeros(1000, dtype=dtype)
        arr_copy = arr.copy()
        mask = np.random.randint(2, size=1000).astype(bool)

        # Allowing empty values like this is weird...
        np.put(arr, mask, [])
        assert_array_equal(arr, arr_copy)


class TestPut:
    @pytest.mark.parametrize("dtype", list(np.typecodes["All"])[1:] + ["i,O"])
    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_simple(self, dtype, mode):
        if dtype.lower() == "m":
            dtype += "8[ns]"

        # put is weird and doesn't care about value length (even shorter)
        vals = np.arange(1001).astype(dtype=dtype)

        # Use vals.dtype in case of flexible dtype (i.e. string)
        arr = np.zeros(1000, dtype=vals.dtype)
        zeros = arr.copy()

        if mode == "clip":
            # Special because 0 and -1 value are "reserved" for clip test
            indx = np.random.permutation(len(arr) - 2)[:-500] + 1

            indx[-1] = 0
            indx[-2] = len(arr) - 1
            indx_put = indx.copy()
            indx_put[-1] = -1389
            indx_put[-2] = 1321
        else:
            # Avoid duplicates (for simplicity) and fill half only
            indx = np.random.permutation(len(arr) - 3)[:-500]
            indx_put = indx
            if mode == "wrap":
                indx_put = indx_put + len(arr)

        np.put(arr, indx_put, vals, mode=mode)
        assert_array_equal(arr[indx], vals[:len(indx)])
        untouched = np.ones(len(arr), dtype=bool)
        untouched[indx] = False
        assert_array_equal(arr[untouched], zeros[:untouched.sum()])

    @pytest.mark.parametrize("dtype", list(np.typecodes["All"])[1:] + ["i,O"])
    @pytest.mark.parametrize("mode", ["raise", "wrap", "clip"])
    def test_empty(self, dtype, mode):
        arr = np.zeros(1000, dtype=dtype)
        arr_copy = arr.copy()

        # Allowing empty values like this is weird...
        np.put(arr, [1, 2, 3], [])
        assert_array_equal(arr, arr_copy)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\sas\test_xport.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

from pandas.io.sas.sasreader import read_sas

# CSV versions of test xpt files were obtained using the R foreign library

# Numbers in a SAS xport file are always float64, so need to convert
# before making comparisons.


def numeric_as_float(data):
    for v in data.columns:
        if data[v].dtype is np.dtype("int64"):
            data[v] = data[v].astype(np.float64)


class TestXport:
    @pytest.fixture
    def file01(self, datapath):
        return datapath("io", "sas", "data", "DEMO_G.xpt")

    @pytest.fixture
    def file02(self, datapath):
        return datapath("io", "sas", "data", "SSHSV1_A.xpt")

    @pytest.fixture
    def file03(self, datapath):
        return datapath("io", "sas", "data", "DRXFCD_G.xpt")

    @pytest.fixture
    def file04(self, datapath):
        return datapath("io", "sas", "data", "paxraw_d_short.xpt")

    @pytest.fixture
    def file05(self, datapath):
        return datapath("io", "sas", "data", "DEMO_PUF.cpt")

    @pytest.mark.slow
    def test1_basic(self, file01):
        # Tests with DEMO_G.xpt (all numeric file)

        # Compare to this
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        numeric_as_float(data_csv)

        # Read full file
        data = read_sas(file01, format="xport")
        tm.assert_frame_equal(data, data_csv)
        num_rows = data.shape[0]

        # Test reading beyond end of file
        with read_sas(file01, format="xport", iterator=True) as reader:
            data = reader.read(num_rows + 100)
        assert data.shape[0] == num_rows

        # Test incremental read with `read` method.
        with read_sas(file01, format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

        # Test incremental read with `get_chunk` method.
        with read_sas(file01, format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

        # Test read in loop
        m = 0
        with read_sas(file01, format="xport", chunksize=100) as reader:
            for x in reader:
                m += x.shape[0]
        assert m == num_rows

        # Read full file with `read_sas` method
        data = read_sas(file01)
        tm.assert_frame_equal(data, data_csv)

    def test1_index(self, file01):
        # Tests with DEMO_G.xpt using index (all numeric file)

        # Compare to this
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        data_csv = data_csv.set_index("SEQN")
        numeric_as_float(data_csv)

        # Read full file
        data = read_sas(file01, index="SEQN", format="xport")
        tm.assert_frame_equal(data, data_csv, check_index_type=False)

        # Test incremental read with `read` method.
        with read_sas(file01, index="SEQN", format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

        # Test incremental read with `get_chunk` method.
        with read_sas(file01, index="SEQN", format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

    def test1_incremental(self, file01):
        # Test with DEMO_G.xpt, reading full file incrementally

        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        data_csv = data_csv.set_index("SEQN")
        numeric_as_float(data_csv)

        with read_sas(file01, index="SEQN", chunksize=1000) as reader:
            all_data = list(reader)
        data = pd.concat(all_data, axis=0)

        tm.assert_frame_equal(data, data_csv, check_index_type=False)

    def test2(self, file02):
        # Test with SSHSV1_A.xpt

        # Compare to this
        data_csv = pd.read_csv(file02.replace(".xpt", ".csv"))
        numeric_as_float(data_csv)

        data = read_sas(file02)
        tm.assert_frame_equal(data, data_csv)

    def test2_binary(self, file02):
        # Test with SSHSV1_A.xpt, read as a binary file

        # Compare to this
        data_csv = pd.read_csv(file02.replace(".xpt", ".csv"))
        numeric_as_float(data_csv)

        with open(file02, "rb") as fd:
            # GH#35693 ensure that if we pass an open file, we
            #  dont incorrectly close it in read_sas
            data = read_sas(fd, format="xport")

        tm.assert_frame_equal(data, data_csv)

    def test_multiple_types(self, file03):
        # Test with DRXFCD_G.xpt (contains text and numeric variables)

        # Compare to this
        data_csv = pd.read_csv(file03.replace(".xpt", ".csv"))

        data = read_sas(file03, encoding="utf-8")
        tm.assert_frame_equal(data, data_csv)

    def test_truncated_float_support(self, file04):
        # Test with paxraw_d_short.xpt, a shortened version of:
        # http://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/PAXRAW_D.ZIP
        # This file has truncated floats (5 bytes in this case).

        # GH 11713

        data_csv = pd.read_csv(file04.replace(".xpt", ".csv"))

        data = read_sas(file04, format="xport")
        tm.assert_frame_equal(data.astype("int64"), data_csv)

    def test_cport_header_found_raises(self, file05):
        # Test with DEMO_PUF.cpt, the beginning of puf2019_1_fall.xpt
        # from https://www.cms.gov/files/zip/puf2019.zip
        # (despite the extension, it's a cpt file)
        msg = "Header record indicates a CPORT file, which is not readable."
        with pytest.raises(ValueError, match=msg):
            read_sas(file05, format="xport")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_bsplines.py ===
from sympy.functions import bspline_basis_set, interpolating_spline
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.testing.pytest import slow

x, y = symbols('x,y')


def test_basic_degree_0():
    d = 0
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    for i in range(len(splines)):
        assert splines[i] == Piecewise((1, Interval(i, i + 1).contains(x)),
                                       (0, True))


def test_basic_degree_1():
    d = 1
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    assert splines[0] == Piecewise((x, Interval(0, 1).contains(x)),
                                   (2 - x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[1] == Piecewise((-1 + x, Interval(1, 2).contains(x)),
                                   (3 - x, Interval(2, 3).contains(x)),
                                   (0, True))
    assert splines[2] == Piecewise((-2 + x, Interval(2, 3).contains(x)),
                                   (4 - x, Interval(3, 4).contains(x)),
                                   (0, True))


def test_basic_degree_2():
    d = 2
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    b0 = Piecewise((x**2/2, Interval(0, 1).contains(x)),
                   (Rational(-3, 2) + 3*x - x**2, Interval(1, 2).contains(x)),
                   (Rational(9, 2) - 3*x + x**2/2, Interval(2, 3).contains(x)),
                   (0, True))
    b1 = Piecewise((S.Half - x + x**2/2, Interval(1, 2).contains(x)),
                   (Rational(-11, 2) + 5*x - x**2, Interval(2, 3).contains(x)),
                   (8 - 4*x + x**2/2, Interval(3, 4).contains(x)),
                   (0, True))
    assert splines[0] == b0
    assert splines[1] == b1


def test_basic_degree_3():
    d = 3
    knots = range(5)
    splines = bspline_basis_set(d, knots, x)
    b0 = Piecewise(
        (x**3/6, Interval(0, 1).contains(x)),
        (Rational(2, 3) - 2*x + 2*x**2 - x**3/2, Interval(1, 2).contains(x)),
        (Rational(-22, 3) + 10*x - 4*x**2 + x**3/2, Interval(2, 3).contains(x)),
        (Rational(32, 3) - 8*x + 2*x**2 - x**3/6, Interval(3, 4).contains(x)),
        (0, True)
    )
    assert splines[0] == b0


def test_repeated_degree_1():
    d = 1
    knots = [0, 0, 1, 2, 2, 3, 4, 4]
    splines = bspline_basis_set(d, knots, x)
    assert splines[0] == Piecewise((1 - x, Interval(0, 1).contains(x)),
                                   (0, True))
    assert splines[1] == Piecewise((x, Interval(0, 1).contains(x)),
                                   (2 - x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[2] == Piecewise((-1 + x, Interval(1, 2).contains(x)),
                                   (0, True))
    assert splines[3] == Piecewise((3 - x, Interval(2, 3).contains(x)),
                                   (0, True))
    assert splines[4] == Piecewise((-2 + x, Interval(2, 3).contains(x)),
                                   (4 - x, Interval(3, 4).contains(x)),
                                   (0, True))
    assert splines[5] == Piecewise((-3 + x, Interval(3, 4).contains(x)),
                                   (0, True))


def test_repeated_degree_2():
    d = 2
    knots = [0, 0, 1, 2, 2, 3, 4, 4]
    splines = bspline_basis_set(d, knots, x)

    assert splines[0] == Piecewise(((-3*x**2/2 + 2*x), And(x <= 1, x >= 0)),
                                   (x**2/2 - 2*x + 2, And(x <= 2, x >= 1)),
                                   (0, True))
    assert splines[1] == Piecewise((x**2/2, And(x <= 1, x >= 0)),
                                   (-3*x**2/2 + 4*x - 2, And(x <= 2, x >= 1)),
                                   (0, True))
    assert splines[2] == Piecewise((x**2 - 2*x + 1, And(x <= 2, x >= 1)),
                                   (x**2 - 6*x + 9, And(x <= 3, x >= 2)),
                                   (0, True))
    assert splines[3] == Piecewise((-3*x**2/2 + 8*x - 10, And(x <= 3, x >= 2)),
                                   (x**2/2 - 4*x + 8, And(x <= 4, x >= 3)),
                                   (0, True))
    assert splines[4] == Piecewise((x**2/2 - 2*x + 2, And(x <= 3, x >= 2)),
                                   (-3*x**2/2 + 10*x - 16, And(x <= 4, x >= 3)),
                                   (0, True))

# Tests for interpolating_spline


def test_10_points_degree_1():
    d = 1
    X = [-5, 2, 3, 4, 7, 9, 10, 30, 31, 34]
    Y = [-10, -2, 2, 4, 7, 6, 20, 45, 19, 25]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((x*Rational(8, 7) - Rational(30, 7), (x >= -5) & (x <= 2)), (4*x - 10, (x >= 2) & (x <= 3)),
                               (2*x - 4, (x >= 3) & (x <= 4)), (x, (x >= 4) & (x <= 7)),
                               (-x/2 + Rational(21, 2), (x >= 7) & (x <= 9)), (14*x - 120, (x >= 9) & (x <= 10)),
                               (x*Rational(5, 4) + Rational(15, 2), (x >= 10) & (x <= 30)), (-26*x + 825, (x >= 30) & (x <= 31)),
                               (2*x - 43, (x >= 31) & (x <= 34)))


def test_3_points_degree_2():
    d = 2
    X = [-3, 10, 19]
    Y = [3, -4, 30]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((505*x**2/2574 - x*Rational(4921, 2574) - Rational(1931, 429), (x >= -3) & (x <= 19)))


def test_5_points_degree_2():
    d = 2
    X = [-3, 2, 4, 5, 10]
    Y = [-1, 2, 5, 10, 14]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((4*x**2/329 + x*Rational(1007, 1645) + Rational(1196, 1645), (x >= -3) & (x <= 3)),
                               (2701*x**2/1645 - x*Rational(15079, 1645) + Rational(5065, 329), (x >= 3) & (x <= Rational(9, 2))),
                               (-1319*x**2/1645 + x*Rational(21101, 1645) - Rational(11216, 329), (x >= Rational(9, 2)) & (x <= 10)))


@slow
def test_6_points_degree_3():
    d = 3
    X = [-1, 0, 2, 3, 9, 12]
    Y = [-4, 3, 3, 7, 9, 20]
    spline = interpolating_spline(d, x, X, Y)

    assert spline == Piecewise((6058*x**3/5301 - 18427*x**2/5301 + x*Rational(12622, 5301) + 3, (x >= -1) & (x <= 2)),
                               (-8327*x**3/5301 + 67883*x**2/5301 - x*Rational(159998, 5301) + Rational(43661, 1767), (x >= 2) & (x <= 3)),
                               (5414*x**3/47709 - 1386*x**2/589 + x*Rational(4267, 279) - Rational(12232, 589), (x >= 3) & (x <= 12)))


def test_issue_19262():
    Delta = symbols('Delta', positive=True)
    knots = [i*Delta for i in range(4)]
    basis = bspline_basis_set(1, knots, x)
    y = symbols('y', nonnegative=True)
    basis2 = bspline_basis_set(1, knots, y)
    assert basis[0].subs(x, y) == basis2[0]
    assert interpolating_spline(1, x,
        [Delta*i for i in [1, 2, 4, 7]], [3, 6, 5, 7]
        )  == Piecewise((3*x/Delta, (Delta <= x) & (x <= 2*Delta)),
        (7 - x/(2*Delta), (x >= 2*Delta) & (x <= 4*Delta)),
        (Rational(7, 3) + 2*x/(3*Delta), (x >= 4*Delta) & (x <= 7*Delta)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_spss.py ===
import datetime
from pathlib import Path

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version

pyreadstat = pytest.importorskip("pyreadstat")


# TODO(CoW) - detection of chained assignment in cython
# https://github.com/pandas-dev/pandas/issues/51315
@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
@pytest.mark.parametrize("path_klass", [lambda p: p, Path])
def test_spss_labelled_num(path_klass, datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = path_klass(datapath("io", "data", "spss", "labelled-num.sav"))

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"VAR00002": "This is one"}, index=[0])
    expected["VAR00002"] = pd.Categorical(expected["VAR00002"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"VAR00002": 1.0}, index=[0])
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_labelled_num_na(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = datapath("io", "data", "spss", "labelled-num-na.sav")

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"VAR00002": ["This is one", None]})
    expected["VAR00002"] = pd.Categorical(expected["VAR00002"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"VAR00002": [1.0, np.nan]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_labelled_str(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = datapath("io", "data", "spss", "labelled-str.sav")

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"gender": ["Male", "Female"]})
    expected["gender"] = pd.Categorical(expected["gender"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"gender": ["M", "F"]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_umlauts(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = datapath("io", "data", "spss", "umlauts.sav")

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame(
        {"var1": ["the ä umlaut", "the ü umlaut", "the ä umlaut", "the ö umlaut"]}
    )
    expected["var1"] = pd.Categorical(expected["var1"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"var1": [1.0, 2.0, 1.0, 3.0]})
    tm.assert_frame_equal(df, expected)


def test_spss_usecols(datapath):
    # usecols must be list-like
    fname = datapath("io", "data", "spss", "labelled-num.sav")

    with pytest.raises(TypeError, match="usecols must be list-like."):
        pd.read_spss(fname, usecols="VAR00002")


def test_spss_umlauts_dtype_backend(datapath, dtype_backend):
    # test file from the Haven project (https://haven.tidyverse.org/)
    # Licence at LICENSES/HAVEN_LICENSE, LICENSES/HAVEN_MIT
    fname = datapath("io", "data", "spss", "umlauts.sav")

    df = pd.read_spss(fname, convert_categoricals=False, dtype_backend=dtype_backend)
    expected = pd.DataFrame({"var1": [1.0, 2.0, 1.0, 3.0]}, dtype="Int64")

    if dtype_backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")

        from pandas.arrays import ArrowExtensionArray

        expected = pd.DataFrame(
            {
                col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                for col in expected.columns
            }
        )

    tm.assert_frame_equal(df, expected)


def test_invalid_dtype_backend():
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        pd.read_spss("test", dtype_backend="numpy")


@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
def test_spss_metadata(datapath):
    # GH 54264
    fname = datapath("io", "data", "spss", "labelled-num.sav")

    df = pd.read_spss(fname)
    metadata = {
        "column_names": ["VAR00002"],
        "column_labels": [None],
        "column_names_to_labels": {"VAR00002": None},
        "file_encoding": "UTF-8",
        "number_columns": 1,
        "number_rows": 1,
        "variable_value_labels": {"VAR00002": {1.0: "This is one"}},
        "value_labels": {"labels0": {1.0: "This is one"}},
        "variable_to_label": {"VAR00002": "labels0"},
        "notes": [],
        "original_variable_types": {"VAR00002": "F8.0"},
        "readstat_variable_types": {"VAR00002": "double"},
        "table_name": None,
        "missing_ranges": {},
        "missing_user_values": {},
        "variable_storage_width": {"VAR00002": 8},
        "variable_display_width": {"VAR00002": 8},
        "variable_alignment": {"VAR00002": "unknown"},
        "variable_measure": {"VAR00002": "unknown"},
        "file_label": None,
        "file_format": "sav/zsav",
    }
    if Version(pyreadstat.__version__) >= Version("1.2.4"):
        metadata.update(
            {
                "creation_time": datetime.datetime(2015, 2, 6, 14, 33, 36),
                "modification_time": datetime.datetime(2015, 2, 6, 14, 33, 36),
            }
        )
    if Version(pyreadstat.__version__) >= Version("1.2.8"):
        metadata["mr_sets"] = {}
    tm.assert_dict_equal(df.attrs, metadata)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_permutation.py ===
from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
    MatMul, BlockDiagMatrix, Determinant, Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
    MatrixPermute, PermutationMatrix
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol


def test_PermutationMatrix_basic():
    p = Permutation([1, 0])
    assert unchanged(PermutationMatrix, p)
    raises(ValueError, lambda: PermutationMatrix((0, 1, 2)))
    assert PermutationMatrix(p).as_explicit() == Matrix([[0, 1], [1, 0]])
    assert isinstance(PermutationMatrix(p)*MatrixSymbol('A', 2, 2), MatMul)


def test_PermutationMatrix_matmul():
    p = Permutation([1, 2, 0])
    P = PermutationMatrix(p)
    M = Matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert (P*M).as_explicit() == P.as_explicit()*M
    assert (M*P).as_explicit() == M*P.as_explicit()

    P1 = PermutationMatrix(Permutation([1, 2, 0]))
    P2 = PermutationMatrix(Permutation([2, 1, 0]))
    P3 = PermutationMatrix(Permutation([1, 0, 2]))
    assert P1*P2 == P3


def test_PermutationMatrix_matpow():
    p1 = Permutation([1, 2, 0])
    P1 = PermutationMatrix(p1)
    p2 = Permutation([2, 0, 1])
    P2 = PermutationMatrix(p2)
    assert P1**2 == P2
    assert P1**3 == Identity(3)


def test_PermutationMatrix_identity():
    p = Permutation([0, 1])
    assert PermutationMatrix(p).is_Identity

    p = Permutation([1, 0])
    assert not PermutationMatrix(p).is_Identity


def test_PermutationMatrix_determinant():
    P = PermutationMatrix(Permutation([0, 1, 2]))
    assert Determinant(P).doit() == 1
    P = PermutationMatrix(Permutation([0, 2, 1]))
    assert Determinant(P).doit() == -1
    P = PermutationMatrix(Permutation([2, 0, 1]))
    assert Determinant(P).doit() == 1


def test_PermutationMatrix_inverse():
    P = PermutationMatrix(Permutation(0, 1, 2))
    assert Inverse(P).doit() == PermutationMatrix(Permutation(0, 2, 1))


def test_PermutationMatrix_rewrite_BlockDiagMatrix():
    P = PermutationMatrix(Permutation([0, 1, 2, 3, 4, 5]))
    P0 = PermutationMatrix(Permutation([0]))
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P0, P0, P0, P0, P0)

    P = PermutationMatrix(Permutation([0, 1, 3, 2, 4, 5]))
    P10 = PermutationMatrix(Permutation(0, 1))
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P0, P10, P0, P0)

    P = PermutationMatrix(Permutation([1, 0, 3, 2, 5, 4]))
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P10, P10, P10)

    P = PermutationMatrix(Permutation([0, 4, 3, 2, 1, 5]))
    P3210 = PermutationMatrix(Permutation([3, 2, 1, 0]))
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P3210, P0)

    P = PermutationMatrix(Permutation([0, 4, 2, 3, 1, 5]))
    P3120 = PermutationMatrix(Permutation([3, 1, 2, 0]))
    assert P.rewrite(BlockDiagMatrix) == \
        BlockDiagMatrix(P0, P3120, P0)

    P = PermutationMatrix(Permutation(0, 3)(1, 4)(2, 5))
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P)


def test_MartrixPermute_basic():
    p = Permutation(0, 1)
    P = PermutationMatrix(p)
    A = MatrixSymbol('A', 2, 2)

    raises(ValueError, lambda: MatrixPermute(Symbol('x'), p))
    raises(ValueError, lambda: MatrixPermute(A, Symbol('x')))

    assert MatrixPermute(A, P) == MatrixPermute(A, p)
    raises(ValueError, lambda: MatrixPermute(A, p, 2))

    pp = Permutation(0, 1, size=3)
    assert MatrixPermute(A, pp) == MatrixPermute(A, p)
    pp = Permutation(0, 1, 2)
    raises(ValueError, lambda: MatrixPermute(A, pp))


def test_MatrixPermute_shape():
    p = Permutation(0, 1)
    A = MatrixSymbol('A', 2, 3)
    assert MatrixPermute(A, p).shape == (2, 3)


def test_MatrixPermute_explicit():
    p = Permutation(0, 1, 2)
    A = MatrixSymbol('A', 3, 3)
    AA = A.as_explicit()
    assert MatrixPermute(A, p, 0).as_explicit() == \
        AA.permute(p, orientation='rows')
    assert MatrixPermute(A, p, 1).as_explicit() == \
        AA.permute(p, orientation='cols')


def test_MatrixPermute_rewrite_MatMul():
    p = Permutation(0, 1, 2)
    A = MatrixSymbol('A', 3, 3)

    assert MatrixPermute(A, p, 0).rewrite(MatMul).as_explicit() == \
        MatrixPermute(A, p, 0).as_explicit()
    assert MatrixPermute(A, p, 1).rewrite(MatMul).as_explicit() == \
        MatrixPermute(A, p, 1).as_explicit()


def test_MatrixPermute_doit():
    p = Permutation(0, 1, 2)
    A = MatrixSymbol('A', 3, 3)
    assert MatrixPermute(A, p).doit() == MatrixPermute(A, p)

    p = Permutation(0, size=3)
    A = MatrixSymbol('A', 3, 3)
    assert MatrixPermute(A, p).doit().as_explicit() == \
        MatrixPermute(A, p).as_explicit()

    p = Permutation(0, 1, 2)
    A = Identity(3)
    assert MatrixPermute(A, p, 0).doit().as_explicit() == \
        MatrixPermute(A, p, 0).as_explicit()
    assert MatrixPermute(A, p, 1).doit().as_explicit() == \
        MatrixPermute(A, p, 1).as_explicit()

    A = ZeroMatrix(3, 3)
    assert MatrixPermute(A, p).doit() == A
    A = OneMatrix(3, 3)
    assert MatrixPermute(A, p).doit() == A

    A = MatrixSymbol('A', 4, 4)
    p1 = Permutation(0, 1, 2, 3)
    p2 = Permutation(0, 2, 3, 1)
    expr = MatrixPermute(MatrixPermute(A, p1, 0), p2, 0)
    assert expr.as_explicit() == expr.doit().as_explicit()
    expr = MatrixPermute(MatrixPermute(A, p1, 1), p2, 1)
    assert expr.as_explicit() == expr.doit().as_explicit()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_delta_functions.py ===
from sympy.core.numbers import (I, nan, oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, sign, transpose)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.simplify.simplify import signsimp


from sympy.testing.pytest import raises

from sympy.core.expr import unchanged

from sympy.core.function import ArgumentIndexError


x, y = symbols('x y')
i = symbols('t', nonzero=True)
j = symbols('j', positive=True)
k = symbols('k', negative=True)

def test_DiracDelta():
    assert DiracDelta(1) == 0
    assert DiracDelta(5.1) == 0
    assert DiracDelta(-pi) == 0
    assert DiracDelta(5, 7) == 0
    assert DiracDelta(x, 0) == DiracDelta(x)
    assert DiracDelta(i) == 0
    assert DiracDelta(j) == 0
    assert DiracDelta(k) == 0
    assert DiracDelta(nan) is nan
    assert DiracDelta(0).func is DiracDelta
    assert DiracDelta(x).func is DiracDelta
    # FIXME: this is generally undefined @ x=0
    #         But then limit(Delta(c)*Heaviside(x),x,-oo)
    #         need's to be implemented.
    # assert 0*DiracDelta(x) == 0

    assert adjoint(DiracDelta(x)) == DiracDelta(x)
    assert adjoint(DiracDelta(x - y)) == DiracDelta(x - y)
    assert conjugate(DiracDelta(x)) == DiracDelta(x)
    assert conjugate(DiracDelta(x - y)) == DiracDelta(x - y)
    assert transpose(DiracDelta(x)) == DiracDelta(x)
    assert transpose(DiracDelta(x - y)) == DiracDelta(x - y)

    assert DiracDelta(x).diff(x) == DiracDelta(x, 1)
    assert DiracDelta(x, 1).diff(x) == DiracDelta(x, 2)

    assert DiracDelta(x).is_simple(x) is True
    assert DiracDelta(3*x).is_simple(x) is True
    assert DiracDelta(x**2).is_simple(x) is False
    assert DiracDelta(sqrt(x)).is_simple(x) is False
    assert DiracDelta(x).is_simple(y) is False

    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True, wrt=x) == (
        DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)

    assert DiracDelta(2*x) != DiracDelta(x)  # scaling property
    assert DiracDelta(x) == DiracDelta(-x)  # even function
    assert DiracDelta(-x, 2) == DiracDelta(x, 2)
    assert DiracDelta(-x, 1) == -DiracDelta(x, 1)  # odd deriv is odd
    assert DiracDelta(-oo*x) == DiracDelta(oo*x)
    assert DiracDelta(x - y) != DiracDelta(y - x)
    assert signsimp(DiracDelta(x - y) - DiracDelta(y - x)) == 0

    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True) == (
            DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)

    raises(ArgumentIndexError, lambda: DiracDelta(x).fdiff(2))
    raises(ValueError, lambda: DiracDelta(x, -1))
    raises(ValueError, lambda: DiracDelta(I))
    raises(ValueError, lambda: DiracDelta(2 + 3*I))


def test_heaviside():
    assert Heaviside(-5) == 0
    assert Heaviside(1) == 1
    assert Heaviside(0) == S.Half

    assert Heaviside(0, x) == x
    assert unchanged(Heaviside,x, nan)
    assert Heaviside(0, nan) == nan

    h0  = Heaviside(x, 0)
    h12 = Heaviside(x, S.Half)
    h1  = Heaviside(x, 1)

    assert h0.args == h0.pargs == (x, 0)
    assert h1.args == h1.pargs == (x, 1)
    assert h12.args == (x, S.Half)
    assert h12.pargs == (x,) # default 1/2 suppressed

    assert adjoint(Heaviside(x)) == Heaviside(x)
    assert adjoint(Heaviside(x - y)) == Heaviside(x - y)
    assert conjugate(Heaviside(x)) == Heaviside(x)
    assert conjugate(Heaviside(x - y)) == Heaviside(x - y)
    assert transpose(Heaviside(x)) == Heaviside(x)
    assert transpose(Heaviside(x - y)) == Heaviside(x - y)

    assert Heaviside(x).diff(x) == DiracDelta(x)
    assert Heaviside(x + I).is_Function is True
    assert Heaviside(I*x).is_Function is True

    raises(ArgumentIndexError, lambda: Heaviside(x).fdiff(2))
    raises(ValueError, lambda: Heaviside(I))
    raises(ValueError, lambda: Heaviside(2 + 3*I))


def test_rewrite():
    x, y = Symbol('x', real=True), Symbol('y')
    assert Heaviside(x).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (Heaviside(0), Eq(x, 0)), (1, True)))
    assert Heaviside(y).rewrite(Piecewise) == (
        Piecewise((0, y < 0), (Heaviside(0), Eq(y, 0)), (1, True)))
    assert Heaviside(x, y).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (y, Eq(x, 0)), (1, True)))
    assert Heaviside(x, 0).rewrite(Piecewise) == (
        Piecewise((0, x <= 0), (1, True)))
    assert Heaviside(x, 1).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (1, True)))
    assert Heaviside(x, nan).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True)))

    assert Heaviside(x).rewrite(sign) == \
        Heaviside(x, H0=Heaviside(0)).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(Heaviside(0), S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (Heaviside(0), True)), True)
        )

    assert Heaviside(y).rewrite(sign) == Heaviside(y)
    assert Heaviside(x, S.Half).rewrite(sign) == (sign(x)+1)/2
    assert Heaviside(x, y).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(y, S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (y, True)), True)
        )

    assert DiracDelta(y).rewrite(Piecewise) == Piecewise((DiracDelta(0), Eq(y, 0)), (0, True))
    assert DiracDelta(y, 1).rewrite(Piecewise) == DiracDelta(y, 1)
    assert DiracDelta(x - 5).rewrite(Piecewise) == (
        Piecewise((DiracDelta(0), Eq(x - 5, 0)), (0, True)))

    assert (x*DiracDelta(x - 10)).rewrite(SingularityFunction) == x*SingularityFunction(x, 10, -1)
    assert 5*x*y*DiracDelta(y, 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, 0, -2)
    assert DiracDelta(0).rewrite(SingularityFunction) == SingularityFunction(0, 0, -1)
    assert DiracDelta(0, 1).rewrite(SingularityFunction) == SingularityFunction(0, 0, -2)

    assert Heaviside(x).rewrite(SingularityFunction) == SingularityFunction(x, 0, 0)
    assert 5*x*y*Heaviside(y + 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, -1, 0)
    assert ((x - 3)**3*Heaviside(x - 3)).rewrite(SingularityFunction) == (x - 3)**3*SingularityFunction(x, 3, 0)
    assert Heaviside(0).rewrite(SingularityFunction) == S.Half

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_fourier.py ===
from sympy.core.add import Add
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, tan)
from sympy.series.fourier import fourier_series
from sympy.series.fourier import FourierSeries
from sympy.testing.pytest import raises
from functools import lru_cache

x, y, z = symbols('x y z')

# Don't declare these during import because they are slow
@lru_cache()
def _get_examples():
    fo = fourier_series(x, (x, -pi, pi))
    fe = fourier_series(x**2, (-pi, pi))
    fp = fourier_series(Piecewise((0, x < 0), (pi, True)), (x, -pi, pi))
    return fo, fe, fp


def test_FourierSeries():
    fo, fe, fp = _get_examples()

    assert fourier_series(1, (-pi, pi)) == 1
    assert (Piecewise((0, x < 0), (pi, True)).
            fourier_series((x, -pi, pi)).truncate()) == fp.truncate()
    assert isinstance(fo, FourierSeries)
    assert fo.function == x
    assert fo.x == x
    assert fo.period == (-pi, pi)

    assert fo.term(3) == 2*sin(3*x) / 3
    assert fe.term(3) == -4*cos(3*x) / 9
    assert fp.term(3) == 2*sin(3*x) / 3

    assert fo.as_leading_term(x) == 2*sin(x)
    assert fe.as_leading_term(x) == pi**2 / 3
    assert fp.as_leading_term(x) == pi / 2

    assert fo.truncate() == 2*sin(x) - sin(2*x) + (2*sin(3*x) / 3)
    assert fe.truncate() == -4*cos(x) + cos(2*x) + pi**2 / 3
    assert fp.truncate() == 2*sin(x) + (2*sin(3*x) / 3) + pi / 2

    fot = fo.truncate(n=None)
    s = [0, 2*sin(x), -sin(2*x)]
    for i, t in enumerate(fot):
        if i == 3:
            break
        assert s[i] == t

    def _check_iter(f, i):
        for ind, t in enumerate(f):
            assert t == f[ind]  # noqa: PLR1736
            if ind == i:
                break

    _check_iter(fo, 3)
    _check_iter(fe, 3)
    _check_iter(fp, 3)

    assert fo.subs(x, x**2) == fo

    raises(ValueError, lambda: fourier_series(x, (0, 1, 2)))
    raises(ValueError, lambda: fourier_series(x, (x, 0, oo)))
    raises(ValueError, lambda: fourier_series(x*y, (0, oo)))


def test_FourierSeries_2():
    p = Piecewise((0, x < 0), (x, True))
    f = fourier_series(p, (x, -2, 2))

    assert f.term(3) == (2*sin(3*pi*x / 2) / (3*pi) -
                         4*cos(3*pi*x / 2) / (9*pi**2))
    assert f.truncate() == (2*sin(pi*x / 2) / pi - sin(pi*x) / pi -
                            4*cos(pi*x / 2) / pi**2 + S.Half)


def test_square_wave():
    """Test if fourier_series approximates discontinuous function correctly."""
    square_wave = Piecewise((1, x < pi), (-1, True))
    s = fourier_series(square_wave, (x, 0, 2*pi))

    assert s.truncate(3) == 4 / pi * sin(x) + 4 / (3 * pi) * sin(3 * x) + \
        4 / (5 * pi) * sin(5 * x)
    assert s.sigma_approximation(4) == 4 / pi * sin(x) * sinc(pi / 4) + \
        4 / (3 * pi) * sin(3 * x) * sinc(3 * pi / 4)


def test_sawtooth_wave():
    s = fourier_series(x, (x, 0, pi))
    assert s.truncate(4) == \
        pi/2 - sin(2*x) - sin(4*x)/2 - sin(6*x)/3
    s = fourier_series(x, (x, 0, 1))
    assert s.truncate(4) == \
        S.Half - sin(2*pi*x)/pi - sin(4*pi*x)/(2*pi) - sin(6*pi*x)/(3*pi)


def test_FourierSeries__operations():
    fo, fe, fp = _get_examples()

    fes = fe.scale(-1).shift(pi**2)
    assert fes.truncate() == 4*cos(x) - cos(2*x) + 2*pi**2 / 3

    assert fp.shift(-pi/2).truncate() == (2*sin(x) + (2*sin(3*x) / 3) +
                                          (2*sin(5*x) / 5))

    fos = fo.scale(3)
    assert fos.truncate() == 6*sin(x) - 3*sin(2*x) + 2*sin(3*x)

    fx = fe.scalex(2).shiftx(1)
    assert fx.truncate() == -4*cos(2*x + 2) + cos(4*x + 4) + pi**2 / 3

    fl = fe.scalex(3).shift(-pi).scalex(2).shiftx(1).scale(4)
    assert fl.truncate() == (-16*cos(6*x + 6) + 4*cos(12*x + 12) -
                             4*pi + 4*pi**2 / 3)

    raises(ValueError, lambda: fo.shift(x))
    raises(ValueError, lambda: fo.shiftx(sin(x)))
    raises(ValueError, lambda: fo.scale(x*y))
    raises(ValueError, lambda: fo.scalex(x**2))


def test_FourierSeries__neg():
    fo, fe, fp = _get_examples()

    assert (-fo).truncate() == -2*sin(x) + sin(2*x) - (2*sin(3*x) / 3)
    assert (-fe).truncate() == +4*cos(x) - cos(2*x) - pi**2 / 3


def test_FourierSeries__add__sub():
    fo, fe, fp = _get_examples()

    assert fo + fo == fo.scale(2)
    assert fo - fo == 0
    assert -fe - fe == fe.scale(-2)

    assert (fo + fe).truncate() == 2*sin(x) - sin(2*x) - 4*cos(x) + cos(2*x) \
        + pi**2 / 3
    assert (fo - fe).truncate() == 2*sin(x) - sin(2*x) + 4*cos(x) - cos(2*x) \
        - pi**2 / 3

    assert isinstance(fo + 1, Add)

    raises(ValueError, lambda: fo + fourier_series(x, (x, 0, 2)))


def test_FourierSeries_finite():

    assert fourier_series(sin(x)).truncate(1) == sin(x)
    # assert type(fourier_series(sin(x)*log(x))).truncate() == FourierSeries
    # assert type(fourier_series(sin(x**2+6))).truncate() == FourierSeries
    assert fourier_series(sin(x)*log(y)*exp(z),(x,pi,-pi)).truncate() == sin(x)*log(y)*exp(z)
    assert fourier_series(sin(x)**6).truncate(oo) == -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 \
           + Rational(5, 16)
    assert fourier_series(sin(x) ** 6).truncate() == -15 * cos(2 * x) / 32 + 3 * cos(4 * x) / 16 \
           + Rational(5, 16)
    assert fourier_series(sin(4*x+3) + cos(3*x+4)).truncate(oo) ==  -sin(4)*sin(3*x) + sin(4*x)*cos(3) \
           + cos(4)*cos(3*x) + sin(3)*cos(4*x)
    assert fourier_series(sin(x)+cos(x)*tan(x)).truncate(oo) == 2*sin(x)
    assert fourier_series(cos(pi*x), (x, -1, 1)).truncate(oo) == cos(pi*x)
    assert fourier_series(cos(3*pi*x + 4) - sin(4*pi*x)*log(pi*y), (x, -1, 1)).truncate(oo) == -log(pi*y)*sin(4*pi*x)\
           - sin(4)*sin(3*pi*x) + cos(4)*cos(3*pi*x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\ranges\test_constructors.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas import (
    Index,
    RangeIndex,
    Series,
)
import pandas._testing as tm


class TestRangeIndexConstructors:
    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize(
        "args, kwargs, start, stop, step",
        [
            ((5,), {}, 0, 5, 1),
            ((1, 5), {}, 1, 5, 1),
            ((1, 5, 2), {}, 1, 5, 2),
            ((0,), {}, 0, 0, 1),
            ((0, 0), {}, 0, 0, 1),
            ((), {"start": 0}, 0, 0, 1),
            ((), {"stop": 0}, 0, 0, 1),
        ],
    )
    def test_constructor(self, args, kwargs, start, stop, step, name):
        result = RangeIndex(*args, name=name, **kwargs)
        expected = Index(np.arange(start, stop, step, dtype=np.int64), name=name)
        assert isinstance(result, RangeIndex)
        assert result.name is name
        assert result._range == range(start, stop, step)
        tm.assert_index_equal(result, expected, exact="equiv")

    def test_constructor_invalid_args(self):
        msg = "RangeIndex\\(\\.\\.\\.\\) must be called with integers"
        with pytest.raises(TypeError, match=msg):
            RangeIndex()

        with pytest.raises(TypeError, match=msg):
            RangeIndex(name="Foo")

        # we don't allow on a bare Index
        msg = (
            r"Index\(\.\.\.\) must be called with a collection of some "
            r"kind, 0 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            Index(0)

    @pytest.mark.parametrize(
        "args",
        [
            Index(["a", "b"]),
            Series(["a", "b"]),
            np.array(["a", "b"]),
            [],
            np.arange(0, 10),
            np.array([1]),
            [1],
        ],
    )
    def test_constructor_additional_invalid_args(self, args):
        msg = f"Value needs to be a scalar value, was type {type(args).__name__}"
        with pytest.raises(TypeError, match=msg):
            RangeIndex(args)

    @pytest.mark.parametrize("args", ["foo", datetime(2000, 1, 1, 0, 0)])
    def test_constructor_invalid_args_wrong_type(self, args):
        msg = f"Wrong type {type(args)} for value {args}"
        with pytest.raises(TypeError, match=msg):
            RangeIndex(args)

    def test_constructor_same(self):
        # pass thru w and w/o copy
        index = RangeIndex(1, 5, 2)
        result = RangeIndex(index, copy=False)
        assert result.identical(index)

        result = RangeIndex(index, copy=True)
        tm.assert_index_equal(result, index, exact=True)

        result = RangeIndex(index)
        tm.assert_index_equal(result, index, exact=True)

        with pytest.raises(
            ValueError,
            match="Incorrect `dtype` passed: expected signed integer, received float64",
        ):
            RangeIndex(index, dtype="float64")

    def test_constructor_range_object(self):
        result = RangeIndex(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

    def test_constructor_range(self):
        result = RangeIndex.from_range(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = RangeIndex.from_range(range(5, 6))
        expected = RangeIndex(5, 6, 1)
        tm.assert_index_equal(result, expected, exact=True)

        # an invalid range
        result = RangeIndex.from_range(range(5, 1))
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected, exact=True)

        result = RangeIndex.from_range(range(5))
        expected = RangeIndex(0, 5, 1)
        tm.assert_index_equal(result, expected, exact=True)

        result = Index(range(1, 5, 2))
        expected = RangeIndex(1, 5, 2)
        tm.assert_index_equal(result, expected, exact=True)

        msg = (
            r"(RangeIndex.)?from_range\(\) got an unexpected keyword argument( 'copy')?"
        )
        with pytest.raises(TypeError, match=msg):
            RangeIndex.from_range(range(10), copy=True)

    def test_constructor_name(self):
        # GH#12288
        orig = RangeIndex(10)
        orig.name = "original"

        copy = RangeIndex(orig)
        copy.name = "copy"

        assert orig.name == "original"
        assert copy.name == "copy"

        new = Index(copy)
        assert new.name == "copy"

        new.name = "new"
        assert orig.name == "original"
        assert copy.name == "copy"
        assert new.name == "new"

    def test_constructor_corner(self):
        arr = np.array([1, 2, 3, 4], dtype=object)
        index = RangeIndex(1, 5)
        assert index.values.dtype == np.int64
        expected = Index(arr).astype("int64")

        tm.assert_index_equal(index, expected, exact="equiv")

        # non-int raise Exception
        with pytest.raises(TypeError, match=r"Wrong type \<class 'str'\>"):
            RangeIndex("1", "10", "1")
        with pytest.raises(TypeError, match=r"Wrong type \<class 'float'\>"):
            RangeIndex(1.1, 10.2, 1.3)

        # invalid passed type
        with pytest.raises(
            ValueError,
            match="Incorrect `dtype` passed: expected signed integer, received float64",
        ):
            RangeIndex(1, 5, dtype="float64")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\tests\test_finite_diff.py ===
from itertools import product

from sympy.core.function import (Function, diff)
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.calculus.finite_diff import (
    apply_finite_diff, differentiate_finite, finite_diff_weights,
    _as_finite_diff
)
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_apply_finite_diff():
    x, h = symbols('x h')
    f = Function('f')
    assert (apply_finite_diff(1, [x-h, x+h], [f(x-h), f(x+h)], x) -
            (f(x+h)-f(x-h))/(2*h)).simplify() == 0

    assert (apply_finite_diff(1, [5, 6, 7], [f(5), f(6), f(7)], 5) -
            (Rational(-3, 2)*f(5) + 2*f(6) - S.Half*f(7))).simplify() == 0
    raises(ValueError, lambda: apply_finite_diff(1, [x, h], [f(x)]))


def test_finite_diff_weights():

    d = finite_diff_weights(1, [5, 6, 7], 5)
    assert d[1][2] == [Rational(-3, 2), 2, Rational(-1, 2)]

    # Table 1, p. 702 in doi:10.1090/S0025-5718-1988-0935077-0
    # --------------------------------------------------------
    xl = [0, 1, -1, 2, -2, 3, -3, 4, -4]

    # d holds all coefficients
    d = finite_diff_weights(4, xl, S.Zero)

    # Zeroeth derivative
    for i in range(5):
        assert d[0][i] == [S.One] + [S.Zero]*8

    # First derivative
    assert d[1][0] == [S.Zero]*9
    assert d[1][2] == [S.Zero, S.Half, Rational(-1, 2)] + [S.Zero]*6
    assert d[1][4] == [S.Zero, Rational(2, 3), Rational(-2, 3), Rational(-1, 12), Rational(1, 12)] + [S.Zero]*4
    assert d[1][6] == [S.Zero, Rational(3, 4), Rational(-3, 4), Rational(-3, 20), Rational(3, 20),
                       Rational(1, 60), Rational(-1, 60)] + [S.Zero]*2
    assert d[1][8] == [S.Zero, Rational(4, 5), Rational(-4, 5), Rational(-1, 5), Rational(1, 5),
                       Rational(4, 105), Rational(-4, 105), Rational(-1, 280), Rational(1, 280)]

    # Second derivative
    for i in range(2):
        assert d[2][i] == [S.Zero]*9
    assert d[2][2] == [-S(2), S.One, S.One] + [S.Zero]*6
    assert d[2][4] == [Rational(-5, 2), Rational(4, 3), Rational(4, 3), Rational(-1, 12), Rational(-1, 12)] + [S.Zero]*4
    assert d[2][6] == [Rational(-49, 18), Rational(3, 2), Rational(3, 2), Rational(-3, 20), Rational(-3, 20),
                       Rational(1, 90), Rational(1, 90)] + [S.Zero]*2
    assert d[2][8] == [Rational(-205, 72), Rational(8, 5), Rational(8, 5), Rational(-1, 5), Rational(-1, 5),
                       Rational(8, 315), Rational(8, 315), Rational(-1, 560), Rational(-1, 560)]

    # Third derivative
    for i in range(3):
        assert d[3][i] == [S.Zero]*9
    assert d[3][4] == [S.Zero, -S.One, S.One, S.Half, Rational(-1, 2)] + [S.Zero]*4
    assert d[3][6] == [S.Zero, Rational(-13, 8), Rational(13, 8), S.One, -S.One,
                       Rational(-1, 8), Rational(1, 8)] + [S.Zero]*2
    assert d[3][8] == [S.Zero, Rational(-61, 30), Rational(61, 30), Rational(169, 120), Rational(-169, 120),
                       Rational(-3, 10), Rational(3, 10), Rational(7, 240), Rational(-7, 240)]

    # Fourth derivative
    for i in range(4):
        assert d[4][i] == [S.Zero]*9
    assert d[4][4] == [S(6), -S(4), -S(4), S.One, S.One] + [S.Zero]*4
    assert d[4][6] == [Rational(28, 3), Rational(-13, 2), Rational(-13, 2), S(2), S(2),
                       Rational(-1, 6), Rational(-1, 6)] + [S.Zero]*2
    assert d[4][8] == [Rational(91, 8), Rational(-122, 15), Rational(-122, 15), Rational(169, 60), Rational(169, 60),
                       Rational(-2, 5), Rational(-2, 5), Rational(7, 240), Rational(7, 240)]

    # Table 2, p. 703 in doi:10.1090/S0025-5718-1988-0935077-0
    # --------------------------------------------------------
    xl = [[j/S(2) for j in list(range(-i*2+1, 0, 2))+list(range(1, i*2+1, 2))]
          for i in range(1, 5)]

    # d holds all coefficients
    d = [finite_diff_weights({0: 1, 1: 2, 2: 4, 3: 4}[i], xl[i], 0) for
         i in range(4)]

    # Zeroth derivative
    assert d[0][0][1] == [S.Half, S.Half]
    assert d[1][0][3] == [Rational(-1, 16), Rational(9, 16), Rational(9, 16), Rational(-1, 16)]
    assert d[2][0][5] == [Rational(3, 256), Rational(-25, 256), Rational(75, 128), Rational(75, 128),
                          Rational(-25, 256), Rational(3, 256)]
    assert d[3][0][7] == [Rational(-5, 2048), Rational(49, 2048), Rational(-245, 2048), Rational(1225, 2048),
                          Rational(1225, 2048), Rational(-245, 2048), Rational(49, 2048), Rational(-5, 2048)]

    # First derivative
    assert d[0][1][1] == [-S.One, S.One]
    assert d[1][1][3] == [Rational(1, 24), Rational(-9, 8), Rational(9, 8), Rational(-1, 24)]
    assert d[2][1][5] == [Rational(-3, 640), Rational(25, 384), Rational(-75, 64),
                          Rational(75, 64), Rational(-25, 384), Rational(3, 640)]
    assert d[3][1][7] == [Rational(5, 7168), Rational(-49, 5120),
                          Rational(245, 3072), Rational(-1225, 1024),
                          Rational(1225, 1024), Rational(-245, 3072),
                          Rational(49, 5120), Rational(-5, 7168)]

    # Reasonably the rest of the table is also correct... (testing of that
    # deemed excessive at the moment)
    raises(ValueError, lambda: finite_diff_weights(-1, [1, 2]))
    raises(ValueError, lambda: finite_diff_weights(1.2, [1, 2]))
    x = symbols('x')
    raises(ValueError, lambda: finite_diff_weights(x, [1, 2]))


def test_as_finite_diff():
    x = symbols('x')
    f = Function('f')
    dx = Function('dx')

    _as_finite_diff(f(x).diff(x), [x-2, x-1, x, x+1, x+2])

    # Use of undefined functions in ``points``
    df_true = -f(x+dx(x)/2-dx(x+dx(x)/2)/2) / dx(x+dx(x)/2) \
              + f(x+dx(x)/2+dx(x+dx(x)/2)/2) / dx(x+dx(x)/2)
    df_test = diff(f(x), x).as_finite_difference(points=dx(x), x0=x+dx(x)/2)
    assert (df_test - df_true).simplify() == 0


def test_differentiate_finite():
    x, y, h = symbols('x y h')
    f = Function('f')
    with warns_deprecated_sympy():
        res0 = differentiate_finite(f(x, y) + exp(42), x, y, evaluate=True)
    xm, xp, ym, yp = [v + sign*S.Half for v, sign in product([x, y], [-1, 1])]
    ref0 = f(xm, ym) + f(xp, yp) - f(xm, yp) - f(xp, ym)
    assert (res0 - ref0).simplify() == 0

    g = Function('g')
    with warns_deprecated_sympy():
        res1 = differentiate_finite(f(x)*g(x) + 42, x, evaluate=True)
    ref1 = (-f(x - S.Half) + f(x + S.Half))*g(x) + \
           (-g(x - S.Half) + g(x + S.Half))*f(x)
    assert (res1 - ref1).simplify() == 0

    res2 = differentiate_finite(f(x) + x**3 + 42, x, points=[x-1, x+1])
    ref2 = (f(x + 1) + (x + 1)**3 - f(x - 1) - (x - 1)**3)/2
    assert (res2 - ref2).simplify() == 0
    raises(TypeError, lambda: differentiate_finite(f(x)*g(x), x,
                                                   pints=[x-1, x+1]))

    res3 = differentiate_finite(f(x)*g(x).diff(x), x)
    ref3 = (-g(x) + g(x + 1))*f(x + S.Half) - (g(x) - g(x - 1))*f(x - S.Half)
    assert res3 == ref3

    res4 = differentiate_finite(f(x)*g(x).diff(x).diff(x), x)
    ref4 = -((g(x - Rational(3, 2)) - 2*g(x - S.Half) + g(x + S.Half))*f(x - S.Half)) \
           + (g(x - S.Half) - 2*g(x + S.Half) + g(x + Rational(3, 2)))*f(x + S.Half)
    assert res4 == ref4

    res5_expr = f(x).diff(x)*g(x).diff(x)
    res5 = differentiate_finite(res5_expr, points=[x-h, x, x+h])
    ref5 = (-2*f(x)/h + f(-h + x)/(2*h) + 3*f(h + x)/(2*h))*(-2*g(x)/h + g(-h + x)/(2*h) \
           + 3*g(h + x)/(2*h))/(2*h) - (2*f(x)/h - 3*f(-h + x)/(2*h) - \
           f(h + x)/(2*h))*(2*g(x)/h - 3*g(-h + x)/(2*h) - g(h + x)/(2*h))/(2*h)
    assert res5 == ref5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_matchpy_connector.py ===
import pickle

from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer

matchpy = import_module("matchpy")

x, y, z = symbols("x y z")


def _get_first_match(expr, pattern):
    from matchpy import ManyToOneMatcher, Pattern

    matcher = ManyToOneMatcher()
    matcher.add(Pattern(pattern))
    return next(iter(matcher.match(expr)))


def test_matchpy_connector():
    if matchpy is None:
        skip("matchpy not installed")

    from multiset import Multiset
    from matchpy import Pattern, Substitution

    w_ = WildDot("w_")
    w__ = WildPlus("w__")
    w___ = WildStar("w___")

    expr = x + y
    pattern = x + w_
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w_': y})

    expr = x + y + z
    pattern = x + w__
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w__': Multiset([y, z])})

    expr = x + y + z
    pattern = x + y + z + w___
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w___': Multiset()})


def test_matchpy_optional():
    if matchpy is None:
        skip("matchpy not installed")

    from matchpy import Pattern, Substitution
    from matchpy import ManyToOneReplacer, ReplacementRule

    p = WildDot("p", optional=1)
    q = WildDot("q", optional=0)

    pattern = p*x + q

    expr1 = 2*x
    pa, subst = _get_first_match(expr1, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 2, 'q': 0})

    expr2 = x + 3
    pa, subst = _get_first_match(expr2, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 3})

    expr3 = x
    pa, subst = _get_first_match(expr3, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 0})

    expr4 = x*y + z
    pa, subst = _get_first_match(expr4, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': y, 'q': z})

    replacer = ManyToOneReplacer()
    replacer.add(ReplacementRule(Pattern(pattern), lambda p, q: sin(p)*cos(q)))
    assert replacer.replace(expr1) == sin(2)*cos(0)
    assert replacer.replace(expr2) == sin(1)*cos(3)
    assert replacer.replace(expr3) == sin(1)*cos(0)
    assert replacer.replace(expr4) == sin(y)*cos(z)


def test_replacer():
    if matchpy is None:
        skip("matchpy not installed")

    for info in [True, False]:
        for lambdify in [True, False]:
            _perform_test_replacer(info, lambdify)


def _perform_test_replacer(info, lambdify):

    x1_ = WildDot("x1_")
    x2_ = WildDot("x2_")

    a_ = WildDot("a_", optional=S.One)
    b_ = WildDot("b_", optional=S.One)
    c_ = WildDot("c_", optional=S.Zero)

    replacer = Replacer(common_constraints=[
        matchpy.CustomConstraint(lambda a_: not a_.has(x)),
        matchpy.CustomConstraint(lambda b_: not b_.has(x)),
        matchpy.CustomConstraint(lambda c_: not c_.has(x)),
    ], lambdify=lambdify, info=info)

    # Rewrite the equation into implicit form, unless it's already solved:
    replacer.add(Eq(x1_, x2_), Eq(x1_ - x2_, 0), conditions_nonfalse=[Ne(x2_, 0), Ne(x1_, 0), Ne(x1_, x), Ne(x2_, x)], info=1)

    # Simple equation solver for real numbers:
    replacer.add(Eq(a_*x + b_, 0), Eq(x, -b_/a_), info=2)
    disc = b_**2 - 4*a_*c_
    replacer.add(
        Eq(a_*x**2 + b_*x + c_, 0),
        Eq(x, (-b_ - sqrt(disc))/(2*a_)) | Eq(x, (-b_ + sqrt(disc))/(2*a_)),
        conditions_nonfalse=[disc >= 0],
        info=3
    )
    replacer.add(
        Eq(a_*x**2 + c_, 0),
        Eq(x, sqrt(-c_/a_)) | Eq(x, -sqrt(-c_/a_)),
        conditions_nonfalse=[-c_*a_ > 0],
        info=4
    )

    g = lambda expr, infos: (expr, infos) if info else expr

    assert replacer.replace(Eq(3*x, y)) == g(Eq(x, y/3), [1, 2])
    assert replacer.replace(Eq(x**2 + 1, 0)) == g(Eq(x**2 + 1, 0), [])
    assert replacer.replace(Eq(x**2, 4)) == g((Eq(x, 2) | Eq(x, -2)), [1, 4])
    assert replacer.replace(Eq(x**2 + 4*y*x + 4*y**2, 0)) == g(Eq(x, -2*y), [3])


def test_matchpy_object_pickle():
    if matchpy is None:
        return

    a1 = WildDot("a")
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2

    a1 = WildDot("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2

    a1 = WildPlus("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2

    a1 = WildStar("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_apply_mutate.py ===
import numpy as np

import pandas as pd
import pandas._testing as tm


def test_group_by_copy():
    # GH#44803
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carl"],
            "age": [20, 21, 20],
        }
    ).set_index("name")

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grp_by_same_value = df.groupby(["age"], group_keys=False).apply(
            lambda group: group
        )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grp_by_copy = df.groupby(["age"], group_keys=False).apply(
            lambda group: group.copy()
        )
    tm.assert_frame_equal(grp_by_same_value, grp_by_copy)


def test_mutate_groups():
    # GH3380

    df = pd.DataFrame(
        {
            "cat1": ["a"] * 8 + ["b"] * 6,
            "cat2": ["c"] * 2
            + ["d"] * 2
            + ["e"] * 2
            + ["f"] * 2
            + ["c"] * 2
            + ["d"] * 2
            + ["e"] * 2,
            "cat3": [f"g{x}" for x in range(1, 15)],
            "val": np.random.default_rng(2).integers(100, size=14),
        }
    )

    def f_copy(x):
        x = x.copy()
        x["rank"] = x.val.rank(method="min")
        return x.groupby("cat2")["rank"].min()

    def f_no_copy(x):
        x["rank"] = x.val.rank(method="min")
        return x.groupby("cat2")["rank"].min()

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grpby_copy = df.groupby("cat1").apply(f_copy)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grpby_no_copy = df.groupby("cat1").apply(f_no_copy)
    tm.assert_series_equal(grpby_copy, grpby_no_copy)


def test_no_mutate_but_looks_like():
    # GH 8467
    # first show's mutation indicator
    # second does not, but should yield the same results
    df = pd.DataFrame({"key": [1, 1, 1, 2, 2, 2, 3, 3, 3], "value": range(9)})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result1 = df.groupby("key", group_keys=True).apply(lambda x: x[:].key)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result2 = df.groupby("key", group_keys=True).apply(lambda x: x.key)
    tm.assert_series_equal(result1, result2)


def test_apply_function_with_indexing(warn_copy_on_write):
    # GH: 33058
    df = pd.DataFrame(
        {"col1": ["A", "A", "A", "B", "B", "B"], "col2": [1, 2, 3, 4, 5, 6]}
    )

    def fn(x):
        x.loc[x.index[-1], "col2"] = 0
        return x.col2

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(
        FutureWarning, match=msg, raise_on_extra_warnings=not warn_copy_on_write
    ):
        result = df.groupby(["col1"], as_index=False).apply(fn)
    expected = pd.Series(
        [1, 2, 0, 4, 5, 0],
        index=pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5)]
        ),
        name="col2",
    )
    tm.assert_series_equal(result, expected)


def test_apply_mutate_columns_multiindex():
    # GH 12652
    df = pd.DataFrame(
        {
            ("C", "julian"): [1, 2, 3],
            ("B", "geoffrey"): [1, 2, 3],
            ("A", "julian"): [1, 2, 3],
            ("B", "julian"): [1, 2, 3],
            ("A", "geoffrey"): [1, 2, 3],
            ("C", "geoffrey"): [1, 2, 3],
        },
        columns=pd.MultiIndex.from_tuples(
            [
                ("A", "julian"),
                ("A", "geoffrey"),
                ("B", "julian"),
                ("B", "geoffrey"),
                ("C", "julian"),
                ("C", "geoffrey"),
            ]
        ),
    )

    def add_column(grouped):
        name = grouped.columns[0][1]
        grouped["sum", name] = grouped.sum(axis=1)
        return grouped

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=1, axis=1)
    result = gb.apply(add_column)
    expected = pd.DataFrame(
        [
            [1, 1, 1, 3, 1, 1, 1, 3],
            [2, 2, 2, 6, 2, 2, 2, 6],
            [
                3,
                3,
                3,
                9,
                3,
                3,
                3,
                9,
            ],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("geoffrey", "A", "geoffrey"),
                ("geoffrey", "B", "geoffrey"),
                ("geoffrey", "C", "geoffrey"),
                ("geoffrey", "sum", "geoffrey"),
                ("julian", "A", "julian"),
                ("julian", "B", "julian"),
                ("julian", "C", "julian"),
                ("julian", "sum", "julian"),
            ]
        ),
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_missing.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    date_range,
)
import pandas._testing as tm


@pytest.mark.parametrize("func", ["ffill", "bfill"])
def test_groupby_column_index_name_lost_fill_funcs(func):
    # GH: 29764 groupby loses index sometimes
    df = DataFrame(
        [[1, 1.0, -1.0], [1, np.nan, np.nan], [1, 2.0, -2.0]],
        columns=Index(["type", "a", "b"], name="idx"),
    )
    df_grouped = df.groupby(["type"])[["a", "b"]]
    result = getattr(df_grouped, func)().columns
    expected = Index(["a", "b"], name="idx")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("func", ["ffill", "bfill"])
def test_groupby_fill_duplicate_column_names(func):
    # GH: 25610 ValueError with duplicate column names
    df1 = DataFrame({"field1": [1, 3, 4], "field2": [1, 3, 4]})
    df2 = DataFrame({"field1": [1, np.nan, 4]})
    df_grouped = pd.concat([df1, df2], axis=1).groupby(by=["field2"])
    expected = DataFrame(
        [[1, 1.0], [3, np.nan], [4, 4.0]], columns=["field1", "field1"]
    )
    result = getattr(df_grouped, func)()
    tm.assert_frame_equal(result, expected)


def test_ffill_missing_arguments():
    # GH 14955
    df = DataFrame({"a": [1, 2], "b": [1, 1]})
    msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pytest.raises(ValueError, match="Must specify a fill"):
            df.groupby("b").fillna()


@pytest.mark.parametrize(
    "method, expected", [("ffill", [None, "a", "a"]), ("bfill", ["a", "a", None])]
)
def test_fillna_with_string_dtype(method, expected):
    # GH 40250
    df = DataFrame({"a": pd.array([None, "a", None], dtype="string"), "b": [0, 0, 0]})
    grp = df.groupby("b")
    msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grp.fillna(method=method)
    expected = DataFrame({"a": pd.array(expected, dtype="string")})
    tm.assert_frame_equal(result, expected)


def test_fill_consistency():
    # GH9221
    # pass thru keyword arguments to the generated wrapper
    # are set if the passed kw is None (only)
    df = DataFrame(
        index=pd.MultiIndex.from_product(
            [["value1", "value2"], date_range("2014-01-01", "2014-01-06")]
        ),
        columns=Index(["1", "2"], name="id"),
    )
    df["1"] = [
        np.nan,
        1,
        np.nan,
        np.nan,
        11,
        np.nan,
        np.nan,
        2,
        np.nan,
        np.nan,
        22,
        np.nan,
    ]
    df["2"] = [
        np.nan,
        3,
        np.nan,
        np.nan,
        33,
        np.nan,
        np.nan,
        4,
        np.nan,
        np.nan,
        44,
        np.nan,
    ]

    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby(level=0, axis=0).fillna(method="ffill")

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.T.groupby(level=0, axis=1).fillna(method="ffill").T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["ffill", "bfill"])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("has_nan_group", [True, False])
def test_ffill_handles_nan_groups(dropna, method, has_nan_group):
    # GH 34725

    df_without_nan_rows = DataFrame([(1, 0.1), (2, 0.2)])

    ridx = [-1, 0, -1, -1, 1, -1]
    df = df_without_nan_rows.reindex(ridx).reset_index(drop=True)

    group_b = np.nan if has_nan_group else "b"
    df["group_col"] = pd.Series(["a"] * 3 + [group_b] * 3)

    grouped = df.groupby(by="group_col", dropna=dropna)
    result = getattr(grouped, method)(limit=None)

    expected_rows = {
        ("ffill", True, True): [-1, 0, 0, -1, -1, -1],
        ("ffill", True, False): [-1, 0, 0, -1, 1, 1],
        ("ffill", False, True): [-1, 0, 0, -1, 1, 1],
        ("ffill", False, False): [-1, 0, 0, -1, 1, 1],
        ("bfill", True, True): [0, 0, -1, -1, -1, -1],
        ("bfill", True, False): [0, 0, -1, 1, 1, -1],
        ("bfill", False, True): [0, 0, -1, 1, 1, -1],
        ("bfill", False, False): [0, 0, -1, 1, 1, -1],
    }

    ridx = expected_rows.get((method, dropna, has_nan_group))
    expected = df_without_nan_rows.reindex(ridx).reset_index(drop=True)
    # columns are a 'take' on df.columns, which are object dtype
    expected.columns = expected.columns.astype(object)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("min_count, value", [(2, np.nan), (-1, 1.0)])
@pytest.mark.parametrize("func", ["first", "last", "max", "min"])
def test_min_count(func, min_count, value):
    # GH#37821
    df = DataFrame({"a": [1] * 3, "b": [1, np.nan, np.nan], "c": [np.nan] * 3})
    result = getattr(df.groupby("a"), func)(min_count=min_count)
    expected = DataFrame({"b": [value], "c": [np.nan]}, index=Index([1], name="a"))
    tm.assert_frame_equal(result, expected)


def test_indices_with_missing():
    # GH 9304
    df = DataFrame({"a": [1, 1, np.nan], "b": [2, 3, 4], "c": [5, 6, 7]})
    g = df.groupby(["a", "b"])
    result = g.indices
    expected = {(1.0, 2): np.array([0]), (1.0, 3): np.array([1])}
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\test_formats.py ===
import numpy as np
import pytest

from pandas._config import using_string_dtype
import pandas._config.config as cf

from pandas import Index
import pandas._testing as tm


class TestIndexRendering:
    def test_repr_is_valid_construction_code(self):
        # for the case of Index, where the repr is traditional rather than
        # stylized
        idx = Index(["a", "b"])
        res = eval(repr(idx))
        tm.assert_index_equal(res, idx)

    @pytest.mark.xfail(using_string_dtype(), reason="repr different")
    @pytest.mark.parametrize(
        "index,expected",
        [
            # ASCII
            # short
            (
                Index(["a", "bb", "ccc"]),
                """Index(['a', 'bb', 'ccc'], dtype='object')""",
            ),
            # multiple lines
            (
                Index(["a", "bb", "ccc"] * 10),
                "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', "
                "'bb', 'ccc', 'a', 'bb', 'ccc',\n"
                "       'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', "
                "'bb', 'ccc', 'a', 'bb', 'ccc',\n"
                "       'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n"
                "      dtype='object')",
            ),
            # truncated
            (
                Index(["a", "bb", "ccc"] * 100),
                "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',\n"
                "       ...\n"
                "       'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n"
                "      dtype='object', length=300)",
            ),
            # Non-ASCII
            # short
            (
                Index(["あ", "いい", "ううう"]),
                """Index(['あ', 'いい', 'ううう'], dtype='object')""",
            ),
            # multiple lines
            (
                Index(["あ", "いい", "ううう"] * 10),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう'],\n"
                    "      dtype='object')"
                ),
            ),
            # truncated
            (
                Index(["あ", "いい", "ううう"] * 100),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ',\n"
                    "       ...\n"
                    "       'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう'],\n"
                    "      dtype='object', length=300)"
                ),
            ),
        ],
    )
    def test_string_index_repr(self, index, expected):
        result = repr(index)
        assert result == expected

    @pytest.mark.xfail(using_string_dtype(), reason="repr different")
    @pytest.mark.parametrize(
        "index,expected",
        [
            # short
            (
                Index(["あ", "いい", "ううう"]),
                ("Index(['あ', 'いい', 'ううう'], dtype='object')"),
            ),
            # multiple lines
            (
                Index(["あ", "いい", "ううう"] * 10),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう'],\n"
                    "      dtype='object')"
                    ""
                ),
            ),
            # truncated
            (
                Index(["あ", "いい", "ううう"] * 100),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ',\n"
                    "       ...\n"
                    "       'ううう', 'あ', 'いい', 'ううう', 'あ', "
                    "'いい', 'ううう', 'あ', 'いい',\n"
                    "       'ううう'],\n"
                    "      dtype='object', length=300)"
                ),
            ),
        ],
    )
    def test_string_index_repr_with_unicode_option(self, index, expected):
        # Enable Unicode option -----------------------------------------
        with cf.option_context("display.unicode.east_asian_width", True):
            result = repr(index)
            assert result == expected

    def test_repr_summary(self):
        with cf.option_context("display.max_seq_items", 10):
            result = repr(Index(np.arange(1000)))
            assert len(result) < 200
            assert "..." in result

    def test_summary_bug(self):
        # GH#3869
        ind = Index(["{other}%s", "~:{range}:0"], name="A")
        result = ind._summary()
        # shouldn't be formatted accidentally.
        assert "~:{range}:0" in result
        assert "{other}%s" in result

    def test_index_repr_bool_nan(self):
        # GH32146
        arr = Index([True, False, np.nan], dtype=object)
        msg = "Index.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            exp1 = arr.format()
        out1 = ["True", "False", "NaN"]
        assert out1 == exp1

        exp2 = repr(arr)
        out2 = "Index([True, False, nan], dtype='object')"
        assert out2 == exp2

    def test_format_different_scalar_lengths(self):
        # GH#35439
        idx = Index(["aaaaaaaaa", "b"])
        expected = ["aaaaaaaaa", "b"]
        msg = r"Index\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert idx.format() == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\libs\test_libalgos.py ===
from datetime import datetime
from itertools import permutations

import numpy as np

from pandas._libs import algos as libalgos

import pandas._testing as tm


def test_ensure_platform_int():
    arr = np.arange(100, dtype=np.intp)

    result = libalgos.ensure_platform_int(arr)
    assert result is arr


def test_is_lexsorted():
    failure = [
        np.array(
            ([3] * 32) + ([2] * 32) + ([1] * 32) + ([0] * 32),
            dtype="int64",
        ),
        np.array(
            list(range(31))[::-1] * 4,
            dtype="int64",
        ),
    ]

    assert not libalgos.is_lexsorted(failure)


def test_groupsort_indexer():
    a = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)
    b = np.random.default_rng(2).integers(0, 1000, 100).astype(np.intp)

    result = libalgos.groupsort_indexer(a, 1000)[0]

    # need to use a stable sort
    # np.argsort returns int, groupsort_indexer
    # always returns intp
    expected = np.argsort(a, kind="mergesort")
    expected = expected.astype(np.intp)

    tm.assert_numpy_array_equal(result, expected)

    # compare with lexsort
    # np.lexsort returns int, groupsort_indexer
    # always returns intp
    key = a * 1000 + b
    result = libalgos.groupsort_indexer(key, 1000000)[0]
    expected = np.lexsort((b, a))
    expected = expected.astype(np.intp)

    tm.assert_numpy_array_equal(result, expected)


class TestPadBackfill:
    def test_backfill(self):
        old = np.array([1, 5, 10], dtype=np.int64)
        new = np.array(list(range(12)), dtype=np.int64)

        filler = libalgos.backfill["int64_t"](old, new)

        expect_filler = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

        # corner case
        old = np.array([1, 4], dtype=np.int64)
        new = np.array(list(range(5, 10)), dtype=np.int64)
        filler = libalgos.backfill["int64_t"](old, new)

        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

    def test_pad(self):
        old = np.array([1, 5, 10], dtype=np.int64)
        new = np.array(list(range(12)), dtype=np.int64)

        filler = libalgos.pad["int64_t"](old, new)

        expect_filler = np.array([-1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

        # corner case
        old = np.array([5, 10], dtype=np.int64)
        new = np.arange(5, dtype=np.int64)
        filler = libalgos.pad["int64_t"](old, new)
        expect_filler = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(filler, expect_filler)

    def test_pad_backfill_object_segfault(self):
        old = np.array([], dtype="O")
        new = np.array([datetime(2010, 12, 31)], dtype="O")

        result = libalgos.pad["object"](old, new)
        expected = np.array([-1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = libalgos.pad["object"](new, old)
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = libalgos.backfill["object"](old, new)
        expected = np.array([-1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = libalgos.backfill["object"](new, old)
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)


class TestInfinity:
    def test_infinity_sort(self):
        # GH#13445
        # numpy's argsort can be unhappy if something is less than
        # itself.  Instead, let's give our infinities a self-consistent
        # ordering, but outside the float extended real line.

        Inf = libalgos.Infinity()
        NegInf = libalgos.NegInfinity()

        ref_nums = [NegInf, float("-inf"), -1e100, 0, 1e100, float("inf"), Inf]

        assert all(Inf >= x for x in ref_nums)
        assert all(Inf > x or x is Inf for x in ref_nums)
        assert Inf >= Inf and Inf == Inf
        assert not Inf < Inf and not Inf > Inf
        assert libalgos.Infinity() == libalgos.Infinity()
        assert not libalgos.Infinity() != libalgos.Infinity()

        assert all(NegInf <= x for x in ref_nums)
        assert all(NegInf < x or x is NegInf for x in ref_nums)
        assert NegInf <= NegInf and NegInf == NegInf
        assert not NegInf < NegInf and not NegInf > NegInf
        assert libalgos.NegInfinity() == libalgos.NegInfinity()
        assert not libalgos.NegInfinity() != libalgos.NegInfinity()

        for perm in permutations(ref_nums):
            assert sorted(perm) == ref_nums

        # smoke tests
        np.array([libalgos.Infinity()] * 32).argsort()
        np.array([libalgos.NegInfinity()] * 32).argsort()

    def test_infinity_against_nan(self):
        Inf = libalgos.Infinity()
        NegInf = libalgos.NegInfinity()

        assert not Inf > np.nan
        assert not Inf >= np.nan
        assert not Inf < np.nan
        assert not Inf <= np.nan
        assert not Inf == np.nan
        assert Inf != np.nan

        assert not NegInf > np.nan
        assert not NegInf >= np.nan
        assert not NegInf < np.nan
        assert not NegInf <= np.nan
        assert not NegInf == np.nan
        assert NegInf != np.nan

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\unify\tests\test_sympy.py ===
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.logic.boolalg import And
from sympy.core.symbol import Str
from sympy.unify.core import Compound, Variable
from sympy.unify.usympy import (deconstruct, construct, unify, is_associative,
        is_commutative)
from sympy.abc import x, y, z, n

def test_deconstruct():
    expr     = Basic(S(1), S(2), S(3))
    expected = Compound(Basic, (1, 2, 3))
    assert deconstruct(expr) == expected

    assert deconstruct(1) == 1
    assert deconstruct(x) == x
    assert deconstruct(x, variables=(x,)) == Variable(x)
    assert deconstruct(Add(1, x, evaluate=False)) == Compound(Add, (1, x))
    assert deconstruct(Add(1, x, evaluate=False), variables=(x,)) == \
              Compound(Add, (1, Variable(x)))

def test_construct():
    expr     = Compound(Basic, (S(1), S(2), S(3)))
    expected = Basic(S(1), S(2), S(3))
    assert construct(expr) == expected

def test_nested():
    expr = Basic(S(1), Basic(S(2)), S(3))
    cmpd = Compound(Basic, (S(1), Compound(Basic, Tuple(2)), S(3)))
    assert deconstruct(expr) == cmpd
    assert construct(cmpd) == expr

def test_unify():
    expr = Basic(S(1), S(2), S(3))
    a, b, c = map(Symbol, 'abc')
    pattern = Basic(a, b, c)
    assert list(unify(expr, pattern, {}, (a, b, c))) == [{a: 1, b: 2, c: 3}]
    assert list(unify(expr, pattern, variables=(a, b, c))) == \
            [{a: 1, b: 2, c: 3}]

def test_unify_variables():
    assert list(unify(Basic(S(1), S(2)), Basic(S(1), x), {}, variables=(x,))) == [{x: 2}]

def test_s_input():
    expr = Basic(S(1), S(2))
    a, b = map(Symbol, 'ab')
    pattern = Basic(a, b)
    assert list(unify(expr, pattern, {}, (a, b))) == [{a: 1, b: 2}]
    assert list(unify(expr, pattern, {a: 5}, (a, b))) == []

def iterdicteq(a, b):
    a = tuple(a)
    b = tuple(b)
    return len(a) == len(b) and all(x in b for x in a)

def test_unify_commutative():
    expr = Add(1, 2, 3, evaluate=False)
    a, b, c = map(Symbol, 'abc')
    pattern = Add(a, b, c, evaluate=False)

    result  = tuple(unify(expr, pattern, {}, (a, b, c)))
    expected = ({a: 1, b: 2, c: 3},
                {a: 1, b: 3, c: 2},
                {a: 2, b: 1, c: 3},
                {a: 2, b: 3, c: 1},
                {a: 3, b: 1, c: 2},
                {a: 3, b: 2, c: 1})

    assert iterdicteq(result, expected)

def test_unify_iter():
    expr = Add(1, 2, 3, evaluate=False)
    a, b, c = map(Symbol, 'abc')
    pattern = Add(a, c, evaluate=False)
    assert is_associative(deconstruct(pattern))
    assert is_commutative(deconstruct(pattern))

    result   = list(unify(expr, pattern, {}, (a, c)))
    expected = [{a: 1, c: Add(2, 3, evaluate=False)},
                {a: 1, c: Add(3, 2, evaluate=False)},
                {a: 2, c: Add(1, 3, evaluate=False)},
                {a: 2, c: Add(3, 1, evaluate=False)},
                {a: 3, c: Add(1, 2, evaluate=False)},
                {a: 3, c: Add(2, 1, evaluate=False)},
                {a: Add(1, 2, evaluate=False), c: 3},
                {a: Add(2, 1, evaluate=False), c: 3},
                {a: Add(1, 3, evaluate=False), c: 2},
                {a: Add(3, 1, evaluate=False), c: 2},
                {a: Add(2, 3, evaluate=False), c: 1},
                {a: Add(3, 2, evaluate=False), c: 1}]

    assert iterdicteq(result, expected)

def test_hard_match():
    from sympy.functions.elementary.trigonometric import (cos, sin)
    expr = sin(x) + cos(x)**2
    p, q = map(Symbol, 'pq')
    pattern = sin(p) + cos(p)**2
    assert list(unify(expr, pattern, {}, (p, q))) == [{p: x}]

def test_matrix():
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 2, 2)
    Z = MatrixSymbol('Z', 2, 3)
    assert list(unify(X, Y, {}, variables=[n, Str('X')])) == [{Str('X'): Str('Y'), n: 2}]
    assert list(unify(X, Z, {}, variables=[n, Str('X')])) == []

def test_non_frankenAdds():
    # the is_commutative property used to fail because of Basic.__new__
    # This caused is_commutative and str calls to fail
    expr = x+y*2
    rebuilt = construct(deconstruct(expr))
    # Ensure that we can run these commands without causing an error
    str(rebuilt)
    rebuilt.is_commutative

def test_FiniteSet_commutivity():
    from sympy.sets.sets import FiniteSet
    a, b, c, x, y = symbols('a,b,c,x,y')
    s = FiniteSet(a, b, c)
    t = FiniteSet(x, y)
    variables = (x, y)
    assert {x: FiniteSet(a, c), y: b} in tuple(unify(s, t, variables=variables))

def test_FiniteSet_complex():
    from sympy.sets.sets import FiniteSet
    a, b, c, x, y, z = symbols('a,b,c,x,y,z')
    expr = FiniteSet(Basic(S(1), x), y, Basic(x, z))
    pattern = FiniteSet(a, Basic(x, b))
    variables = a, b
    expected = ({b: 1, a: FiniteSet(y, Basic(x, z))},
                      {b: z, a: FiniteSet(y, Basic(S(1), x))})
    assert iterdicteq(unify(expr, pattern, variables=variables), expected)


def test_and():
    variables = x, y
    expected = ({x: z > 0, y: n < 3},)
    assert iterdicteq(unify((z>0) & (n<3), And(x, y), variables=variables),
                      expected)

def test_Union():
    from sympy.sets.sets import Interval
    assert list(unify(Interval(0, 1) + Interval(10, 11),
                      Interval(0, 1) + Interval(12, 13),
                      variables=(Interval(12, 13),)))

def test_is_commutative():
    assert is_commutative(deconstruct(x+y))
    assert is_commutative(deconstruct(x*y))
    assert not is_commutative(deconstruct(x**y))

def test_commutative_in_commutative():
    from sympy.abc import a,b,c,d
    from sympy.functions.elementary.trigonometric import (cos, sin)
    eq = sin(3)*sin(4)*sin(5) + 4*cos(3)*cos(4)
    pat = a*cos(b)*cos(c) + d*sin(b)*sin(c)
    assert next(unify(eq, pat, variables=(a,b,c,d)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_htmlparser.py ===
"""Tests to ensure that the html.parser tree builder generates good
trees."""

import pickle
import pytest
from bs4.builder._htmlparser import (
    _DuplicateAttributeHandler,
    BeautifulSoupHTMLParser,
    HTMLParserTreeBuilder,
)
from bs4.exceptions import ParserRejectedMarkup
from typing import Any
from . import HTMLTreeBuilderSmokeTest


class TestHTMLParserTreeBuilder(HTMLTreeBuilderSmokeTest):
    default_builder = HTMLParserTreeBuilder

    def test_rejected_input(self):
        # Python's html.parser will occasionally reject markup,
        # especially when there is a problem with the initial DOCTYPE
        # declaration. Different versions of Python sound the alarm in
        # different ways, but Beautiful Soup consistently raises
        # errors as ParserRejectedMarkup exceptions.
        bad_markup = [
            # https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28873
            # https://github.com/guidovranken/python-library-fuzzers/blob/master/corp-html/519e5b4269a01185a0d5e76295251921da2f0700
            # https://github.com/python/cpython/issues/81928
            b"\n<![\xff\xfe\xfe\xcd\x00",
            # https://github.com/guidovranken/python-library-fuzzers/blob/master/corp-html/de32aa55785be29bbc72a1a8e06b00611fb3d9f8
            # https://github.com/python/cpython/issues/78661
            #
            b"<![n\x00",
            b"<![UNKNOWN[]]>",
        ]
        for markup in bad_markup:
            with pytest.raises(ParserRejectedMarkup):
                self.soup(markup)

    def test_namespaced_system_doctype(self):
        # html.parser can't handle namespaced doctypes, so skip this one.
        pass

    def test_namespaced_public_doctype(self):
        # html.parser can't handle namespaced doctypes, so skip this one.
        pass

    def test_builder_is_pickled(self):
        """Unlike most tree builders, HTMLParserTreeBuilder and will
        be restored after pickling.
        """
        tree = self.soup("<a><b>foo</a>")
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert isinstance(loaded.builder, type(tree.builder))

    def test_redundant_empty_element_closing_tags(self):
        self.assert_soup("<br></br><br></br><br></br>", "<br/><br/><br/>")
        self.assert_soup("</br></br></br>", "")

    def test_empty_element(self):
        # This verifies that any buffered data present when the parser
        # finishes working is handled.
        self.assert_soup("foo &# bar", "foo &amp;# bar")

    def test_tracking_line_numbers(self):
        # The html.parser TreeBuilder keeps track of line number and
        # position of each element.
        markup = "\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>"
        soup = self.soup(markup)
        assert 2 == soup.p.sourceline
        assert 3 == soup.p.sourcepos
        assert "sourceline" == soup.p.find("sourceline").name

        # You can deactivate this behavior.
        soup = self.soup(markup, store_line_numbers=False)
        assert None is soup.p.sourceline
        assert None is soup.p.sourcepos

    def test_on_duplicate_attribute(self):
        # The html.parser tree builder has a variety of ways of
        # handling a tag that contains the same attribute multiple times.

        markup = '<a class="cls" href="url1" href="url2" href="url3" id="id">'

        # If you don't provide any particular value for
        # on_duplicate_attribute, later values replace earlier values.
        soup = self.soup(markup)
        assert "url3" == soup.a["href"]
        assert ["cls"] == soup.a["class"]
        assert "id" == soup.a["id"]

        # You can also get this behavior explicitly.
        def assert_attribute(
            on_duplicate_attribute: _DuplicateAttributeHandler, expected: Any
        ) -> None:
            soup = self.soup(markup, on_duplicate_attribute=on_duplicate_attribute)
            assert soup.a is not None
            assert expected == soup.a["href"]

            # Verify that non-duplicate attributes are treated normally.
            assert ["cls"] == soup.a["class"]
            assert "id" == soup.a["id"]

        assert_attribute(None, "url3")
        assert_attribute(BeautifulSoupHTMLParser.REPLACE, "url3")

        # You can ignore subsequent values in favor of the first.
        assert_attribute(BeautifulSoupHTMLParser.IGNORE, "url1")

        # And you can pass in a callable that does whatever you want.
        def accumulate(attrs, key, value):
            if not isinstance(attrs[key], list):
                attrs[key] = [attrs[key]]
            attrs[key].append(value)

        assert_attribute(accumulate, ["url1", "url2", "url3"])

    def test_html5_attributes(self):
        # The html.parser TreeBuilder can convert any entity named in
        # the HTML5 spec to a sequence of Unicode characters, and
        # convert those Unicode characters to a (potentially
        # different) named entity on the way out.
        for input_element, output_unicode, output_element in (
            ("&RightArrowLeftArrow;", "\u21c4", b"&rlarr;"),
            ("&models;", "\u22a7", b"&models;"),
            ("&Nfr;", "\U0001d511", b"&Nfr;"),
            ("&ngeqq;", "\u2267\u0338", b"&ngeqq;"),
            ("&not;", "\xac", b"&not;"),
            ("&Not;", "\u2aec", b"&Not;"),
            ("&quot;", '"', b'"'),
            ("&there4;", "\u2234", b"&there4;"),
            ("&Therefore;", "\u2234", b"&there4;"),
            ("&therefore;", "\u2234", b"&there4;"),
            ("&fjlig;", "fj", b"fj"),
            ("&sqcup;", "\u2294", b"&sqcup;"),
            ("&sqcups;", "\u2294\ufe00", b"&sqcups;"),
            ("&apos;", "'", b"'"),
            ("&verbar;", "|", b"|"),
        ):
            markup = "<div>%s</div>" % input_element
            div = self.soup(markup).div
            without_element = div.encode()
            expect = b"<div>%s</div>" % output_unicode.encode("utf8")
            assert without_element == expect

            with_element = div.encode(formatter="html")
            expect = b"<div>%s</div>" % output_element
            assert with_element == expect

    def test_invalid_html_entity(self):
        # The html.parser treebuilder can't distinguish between an invalid
        # HTML entity with a semicolon and an invalid HTML entity with no
        # semicolon.
        markup = "<p>a &nosuchentity b</p>"
        soup = self.soup(markup)
        assert "<p>a &amp;nosuchentity b</p>" == soup.p.decode()

        markup = "<p>a &nosuchentity; b</p>"
        soup = self.soup(markup)
        assert "<p>a &amp;nosuchentity b</p>" == soup.p.decode()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\runtests.py ===
#!/usr/bin/env python

"""
python runtests.py -py
  Use py.test to run tests (more useful for debugging)

python runtests.py -coverage
  Generate test coverage report. Statistics are written to /tmp

python runtests.py -profile
  Generate profile stats (this is much slower)

python runtests.py -nogmpy
  Run tests without using GMPY even if it exists

python runtests.py -strict
  Enforce extra tests in normalize()

python runtests.py -local
  Insert '../..' at the beginning of sys.path to use local mpmath

python runtests.py -skip ...
  Skip tests from the listed modules

Additional arguments are used to filter the tests to run. Only files that have
one of the arguments in their name are executed.

"""

import sys, os, traceback

profile = False
if "-profile" in sys.argv:
    sys.argv.remove('-profile')
    profile = True

coverage = False
if "-coverage" in sys.argv:
    sys.argv.remove('-coverage')
    coverage = True

if "-nogmpy" in sys.argv:
    sys.argv.remove('-nogmpy')
    os.environ['MPMATH_NOGMPY'] = 'Y'

if "-strict" in sys.argv:
    sys.argv.remove('-strict')
    os.environ['MPMATH_STRICT'] = 'Y'

if "-local" in sys.argv:
    sys.argv.remove('-local')
    importdir = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]),
                                             '../..'))
else:
    importdir = ''

# TODO: add a flag for this
testdir = ''

def testit(importdir='', testdir='', exit_on_fail=False):
    """Run all tests in testdir while importing from importdir."""
    if importdir:
        sys.path.insert(1, importdir)
    if testdir:
        sys.path.insert(1, testdir)
    import os.path
    import mpmath
    print("mpmath imported from %s" % os.path.dirname(mpmath.__file__))
    print("mpmath backend: %s" % mpmath.libmp.backend.BACKEND)
    print("mpmath mp class: %s" % repr(mpmath.mp))
    print("mpmath version: %s" % mpmath.__version__)
    print("Python version: %s" % sys.version)
    print("")
    if "-py" in sys.argv:
        sys.argv.remove('-py')
        import py
        py.test.cmdline.main()
    else:
        import glob
        from timeit import default_timer as clock
        modules = []
        args = sys.argv[1:]
        excluded = []
        if '-skip' in args:
            excluded = args[args.index('-skip')+1:]
            args = args[:args.index('-skip')]
        # search for tests in directory of this file if not otherwise specified
        if not testdir:
            pattern = os.path.dirname(sys.argv[0])
        else:
            pattern = testdir
        if pattern:
            pattern += '/'
        pattern += 'test*.py'
        # look for tests (respecting specified filter)
        for f in glob.glob(pattern):
            name = os.path.splitext(os.path.basename(f))[0]
            # If run as a script, only run tests given as args, if any are given
            if args and __name__ == "__main__":
                ok = False
                for arg in args:
                    if arg in name:
                        ok = True
                        break
                if not ok:
                    continue
            elif name in excluded:
                continue
            module = __import__(name)
            priority = module.__dict__.get('priority', 100)
            if priority == 666:
                modules = [[priority, name, module]]
                break
            modules.append([priority, name, module])
        # execute tests
        modules.sort()
        tstart = clock()
        for priority, name, module in modules:
            print(name)
            for f in sorted(module.__dict__.keys()):
                if f.startswith('test_'):
                    if coverage and ('numpy' in f):
                        continue
                    sys.stdout.write("    " + f[5:].ljust(25) + " ")
                    t1 = clock()
                    try:
                        module.__dict__[f]()
                    except:
                        etype, evalue, trb = sys.exc_info()
                        if etype in (KeyboardInterrupt, SystemExit):
                            raise
                        print("")
                        print("TEST FAILED!")
                        print("")
                        traceback.print_exc()
                        if exit_on_fail:
                            return
                    t2 = clock()
                    print("ok " + "       " + ("%.7f" % (t2-t1)) + " s")
        tend = clock()
        print("")
        print("finished tests in " + ("%.2f" % (tend-tstart)) + " seconds")
        # clean sys.path
        if importdir:
            sys.path.remove(importdir)
        if testdir:
            sys.path.remove(testdir)

if __name__ == '__main__':
    if profile:
        import cProfile
        cProfile.run("testit('%s', '%s')" % (importdir, testdir), sort=1)
    elif coverage:
        import trace
        tracer = trace.Trace(ignoredirs=[sys.prefix, sys.exec_prefix],
            trace=0, count=1)
        tracer.run('testit(importdir, testdir)')
        r = tracer.results()
        r.write_results(show_missing=True, summary=True, coverdir="/tmp")
    else:
        testit(importdir, testdir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\recfunctions.py ===
"""These tests are based on the doctests from `numpy/lib/recfunctions.py`."""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions as rfn


def test_recursive_fill_fields() -> None:
    a: npt.NDArray[np.void] = np.array(
        [(1, 10.0), (2, 20.0)],
        dtype=[("A", np.int64), ("B", np.float64)],
    )
    b = np.zeros((int(3),), dtype=a.dtype)
    out = rfn.recursive_fill_fields(a, b)
    assert_type(out, np.ndarray[tuple[int], np.dtype[np.void]])


def test_get_names() -> None:
    names: tuple[str | Any, ...]
    names = rfn.get_names(np.empty((1,), dtype=[("A", int)]).dtype)
    names = rfn.get_names(np.empty((1,), dtype=[("A", int), ("B", float)]).dtype)

    adtype = np.dtype([("a", int), ("b", [("b_a", int), ("b_b", int)])])
    names = rfn.get_names(adtype)


def test_get_names_flat() -> None:
    names: tuple[str, ...]
    names = rfn.get_names_flat(np.empty((1,), dtype=[("A", int)]).dtype)
    names = rfn.get_names_flat(np.empty((1,), dtype=[("A", int), ("B", float)]).dtype)

    adtype = np.dtype([("a", int), ("b", [("b_a", int), ("b_b", int)])])
    names = rfn.get_names_flat(adtype)


def test_flatten_descr() -> None:
    ndtype = np.dtype([("a", "<i4"), ("b", [("b_a", "<f8"), ("b_b", "<i4")])])
    assert_type(rfn.flatten_descr(ndtype), tuple[tuple[str, np.dtype]])


def test_get_fieldstructure() -> None:
    ndtype = np.dtype([
        ("A", int),
        ("B", [("B_A", int), ("B_B", [("B_B_A", int), ("B_B_B", int)])]),
    ])
    assert_type(rfn.get_fieldstructure(ndtype), dict[str, list[str]])


def test_merge_arrays() -> None:
    assert_type(
        rfn.merge_arrays((
            np.ones((int(2),), np.int_),
            np.ones((int(3),), np.float64),
        )),
        np.recarray[tuple[int], np.dtype[np.void]],
    )


def test_drop_fields() -> None:
    ndtype = [("a", np.int64), ("b", [("b_a", np.double), ("b_b", np.int64)])]
    a = np.ones((int(3),), dtype=ndtype)

    assert_type(
        rfn.drop_fields(a, "a"),
        np.ndarray[tuple[int], np.dtype[np.void]],
    )
    assert_type(
        rfn.drop_fields(a, "a", asrecarray=True),
        np.rec.recarray[tuple[int], np.dtype[np.void]],
    )
    assert_type(
        rfn.rec_drop_fields(a, "a"),
        np.rec.recarray[tuple[int], np.dtype[np.void]],
    )


def test_rename_fields() -> None:
    ndtype = [("a", np.int64), ("b", [("b_a", np.double), ("b_b", np.int64)])]
    a = np.ones((int(3),), dtype=ndtype)

    assert_type(
        rfn.rename_fields(a, {"a": "A", "b_b": "B_B"}),
        np.ndarray[tuple[int], np.dtype[np.void]],
    )


def test_repack_fields() -> None:
    dt: np.dtype[np.void] = np.dtype("u1, <i8, <f8", align=True)

    assert_type(rfn.repack_fields(dt), np.dtype[np.void])
    assert_type(rfn.repack_fields(dt.type(0)), np.void)
    assert_type(
        rfn.repack_fields(np.ones((int(3),), dtype=dt)),
        np.ndarray[tuple[int], np.dtype[np.void]],
    )


def test_structured_to_unstructured() -> None:
    a = np.zeros(4, dtype=[("a", "i4"), ("b", "f4,u2"), ("c", "f4", 2)])
    assert_type(rfn.structured_to_unstructured(a), npt.NDArray[Any])


def unstructured_to_structured() -> None:
    dt: np.dtype[np.void] = np.dtype([("a", "i4"), ("b", "f4,u2"), ("c", "f4", 2)])
    a = np.arange(20, dtype=np.int32).reshape((4, 5))
    assert_type(rfn.unstructured_to_structured(a, dt), npt.NDArray[np.void])


def test_apply_along_fields() -> None:
    b = np.ones(4, dtype=[("x", "i4"), ("y", "f4"), ("z", "f8")])
    assert_type(
        rfn.apply_along_fields(np.mean, b),
        np.ndarray[tuple[int], np.dtype[np.void]],
    )


def test_assign_fields_by_name() -> None:
    b = np.ones(4, dtype=[("x", "i4"), ("y", "f4"), ("z", "f8")])
    assert_type(
        rfn.apply_along_fields(np.mean, b),
        np.ndarray[tuple[int], np.dtype[np.void]],
    )


def test_require_fields() -> None:
    a = np.ones(4, dtype=[("a", "i4"), ("b", "f8"), ("c", "u1")])
    assert_type(
        rfn.require_fields(a, [("b", "f4"), ("c", "u1")]),
        np.ndarray[tuple[int], np.dtype[np.void]],
    )


def test_stack_arrays() -> None:
    x = np.zeros((int(2),), np.int32)
    assert_type(
        rfn.stack_arrays(x),
        np.ndarray[tuple[int], np.dtype[np.int32]],
    )

    z = np.ones((int(2),), [("A", "|S3"), ("B", float)])
    zz = np.ones((int(2),), [("A", "|S3"), ("B", np.float64), ("C", np.float64)])
    assert_type(
        rfn.stack_arrays((z, zz)),
        np.ma.MaskedArray[tuple[Any, ...], np.dtype[np.void]],
    )


def test_find_duplicates() -> None:
    ndtype = np.dtype([("a", int)])

    a = np.ma.ones(7, mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
    assert_type(rfn.find_duplicates(a), np.ma.MaskedArray[Any, np.dtype[np.void]])
    assert_type(
        rfn.find_duplicates(a, ignoremask=True, return_index=True),
        tuple[
            np.ma.MaskedArray[Any, np.dtype[np.void]],
            np.ndarray[Any, np.dtype[np.int_]],
        ],
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\interval\test_interval_pyarrow.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


def test_arrow_extension_type():
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    p1 = ArrowIntervalType(pa.int64(), "left")
    p2 = ArrowIntervalType(pa.int64(), "left")
    p3 = ArrowIntervalType(pa.int64(), "right")

    assert p1.closed == "left"
    assert p1 == p2
    assert p1 != p3
    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)


def test_arrow_array():
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    intervals = pd.interval_range(1, 5, freq=1).array

    result = pa.array(intervals)
    assert isinstance(result.type, ArrowIntervalType)
    assert result.type.closed == intervals.closed
    assert result.type.subtype == pa.int64()
    assert result.storage.field("left").equals(pa.array([1, 2, 3, 4], type="int64"))
    assert result.storage.field("right").equals(pa.array([2, 3, 4, 5], type="int64"))

    expected = pa.array([{"left": i, "right": i + 1} for i in range(1, 5)])
    assert result.storage.equals(expected)

    # convert to its storage type
    result = pa.array(intervals, type=expected.type)
    assert result.equals(expected)

    # unsupported conversions
    with pytest.raises(TypeError, match="Not supported to convert IntervalArray"):
        pa.array(intervals, type="float64")

    with pytest.raises(TypeError, match="Not supported to convert IntervalArray"):
        pa.array(intervals, type=ArrowIntervalType(pa.float64(), "left"))


def test_arrow_array_missing():
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    arr = IntervalArray.from_breaks([0.0, 1.0, 2.0, 3.0])
    arr[1] = None

    result = pa.array(arr)
    assert isinstance(result.type, ArrowIntervalType)
    assert result.type.closed == arr.closed
    assert result.type.subtype == pa.float64()

    # fields have missing values (not NaN)
    left = pa.array([0.0, None, 2.0], type="float64")
    right = pa.array([1.0, None, 3.0], type="float64")
    assert result.storage.field("left").equals(left)
    assert result.storage.field("right").equals(right)

    # structarray itself also has missing values on the array level
    vals = [
        {"left": 0.0, "right": 1.0},
        {"left": None, "right": None},
        {"left": 2.0, "right": 3.0},
    ]
    expected = pa.StructArray.from_pandas(vals, mask=np.array([False, True, False]))
    assert result.storage.equals(expected)


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
@pytest.mark.parametrize(
    "breaks",
    [[0.0, 1.0, 2.0, 3.0], pd.date_range("2017", periods=4, freq="D")],
    ids=["float", "datetime64[ns]"],
)
def test_arrow_table_roundtrip(breaks):
    pa = pytest.importorskip("pyarrow")

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

    arr = IntervalArray.from_breaks(breaks)
    arr[1] = None
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    assert isinstance(table.field("a").type, ArrowIntervalType)
    result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.IntervalDtype)
    tm.assert_frame_equal(result, df)

    table2 = pa.concat_tables([table, table])
    result = table2.to_pandas()
    expected = pd.concat([df, df], ignore_index=True)
    tm.assert_frame_equal(result, expected)

    # GH#41040
    table = pa.table(
        [pa.chunked_array([], type=table.column(0).type)], schema=table.schema
    )
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected[0:0])


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
@pytest.mark.parametrize(
    "breaks",
    [[0.0, 1.0, 2.0, 3.0], pd.date_range("2017", periods=4, freq="D")],
    ids=["float", "datetime64[ns]"],
)
def test_arrow_table_roundtrip_without_metadata(breaks):
    pa = pytest.importorskip("pyarrow")

    arr = IntervalArray.from_breaks(breaks)
    arr[1] = None
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    # remove the metadata
    table = table.replace_schema_metadata()
    assert table.schema.metadata is None

    result = table.to_pandas()
    assert isinstance(result["a"].dtype, pd.IntervalDtype)
    tm.assert_frame_equal(result, df)


def test_from_arrow_from_raw_struct_array():
    # in case pyarrow lost the Interval extension type (eg on parquet roundtrip
    # with datetime64[ns] subtype, see GH-45881), still allow conversion
    # from arrow to IntervalArray
    pa = pytest.importorskip("pyarrow")

    arr = pa.array([{"left": 0, "right": 1}, {"left": 1, "right": 2}])
    dtype = pd.IntervalDtype(np.dtype("int64"), closed="neither")

    result = dtype.__from_arrow__(arr)
    expected = IntervalArray.from_breaks(
        np.array([0, 1, 2], dtype="int64"), closed="neither"
    )
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_iteration.py ===
import datetime

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)

from pandas import (
    Categorical,
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestIteration:
    def test_keys(self, float_frame):
        assert float_frame.keys() is float_frame.columns

    def test_iteritems(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        for k, v in df.items():
            assert isinstance(v, DataFrame._constructor_sliced)

    def test_items(self):
        # GH#17213, GH#13918
        cols = ["a", "b", "c"]
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=cols)
        for c, (k, v) in zip(cols, df.items()):
            assert c == k
            assert isinstance(v, Series)
            assert (df[k] == v).all()

    def test_items_names(self, float_string_frame):
        for k, v in float_string_frame.items():
            assert v.name == k

    def test_iter(self, float_frame):
        assert list(float_frame) == list(float_frame.columns)

    def test_iterrows(self, float_frame, float_string_frame):
        for k, v in float_frame.iterrows():
            exp = float_frame.loc[k]
            tm.assert_series_equal(v, exp)

        for k, v in float_string_frame.iterrows():
            exp = float_string_frame.loc[k]
            tm.assert_series_equal(v, exp)

    def test_iterrows_iso8601(self):
        # GH#19671
        s = DataFrame(
            {
                "non_iso8601": ["M1701", "M1802", "M1903", "M2004"],
                "iso8601": date_range("2000-01-01", periods=4, freq="ME"),
            }
        )
        for k, v in s.iterrows():
            exp = s.loc[k]
            tm.assert_series_equal(v, exp)

    def test_iterrows_corner(self):
        # GH#12222
        df = DataFrame(
            {
                "a": [datetime.datetime(2015, 1, 1)],
                "b": [None],
                "c": [None],
                "d": [""],
                "e": [[]],
                "f": [set()],
                "g": [{}],
            }
        )
        expected = Series(
            [datetime.datetime(2015, 1, 1), None, None, "", [], set(), {}],
            index=list("abcdefg"),
            name=0,
            dtype="object",
        )
        _, result = next(df.iterrows())
        tm.assert_series_equal(result, expected)

    def test_itertuples(self, float_frame):
        for i, tup in enumerate(float_frame.itertuples()):
            ser = DataFrame._constructor_sliced(tup[1:])
            ser.name = tup[0]
            expected = float_frame.iloc[i, :].reset_index(drop=True)
            tm.assert_series_equal(ser, expected)

    def test_itertuples_index_false(self):
        df = DataFrame(
            {"floats": np.random.default_rng(2).standard_normal(5), "ints": range(5)},
            columns=["floats", "ints"],
        )

        for tup in df.itertuples(index=False):
            assert isinstance(tup[1], int)

    def test_itertuples_duplicate_cols(self):
        df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
        dfaa = df[["a", "a"]]

        assert list(dfaa.itertuples()) == [(0, 1, 1), (1, 2, 2), (2, 3, 3)]

        # repr with int on 32-bit/windows
        if not (is_platform_windows() or not IS64):
            assert (
                repr(list(df.itertuples(name=None)))
                == "[(0, 1, 4), (1, 2, 5), (2, 3, 6)]"
            )

    def test_itertuples_tuple_name(self):
        df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
        tup = next(df.itertuples(name="TestName"))
        assert tup._fields == ("Index", "a", "b")
        assert (tup.Index, tup.a, tup.b) == tup
        assert type(tup).__name__ == "TestName"

    def test_itertuples_disallowed_col_labels(self):
        df = DataFrame(data={"def": [1, 2, 3], "return": [4, 5, 6]})
        tup2 = next(df.itertuples(name="TestName"))
        assert tup2 == (0, 1, 4)
        assert tup2._fields == ("Index", "_1", "_2")

    @pytest.mark.parametrize("limit", [254, 255, 1024])
    @pytest.mark.parametrize("index", [True, False])
    def test_itertuples_py2_3_field_limit_namedtuple(self, limit, index):
        # GH#28282
        df = DataFrame([{f"foo_{i}": f"bar_{i}" for i in range(limit)}])
        result = next(df.itertuples(index=index))
        assert isinstance(result, tuple)
        assert hasattr(result, "_fields")

    def test_sequence_like_with_categorical(self):
        # GH#7839
        # make sure can iterate
        df = DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
        )
        df["grade"] = Categorical(df["raw_grade"])

        # basic sequencing testing
        result = list(df.grade.values)
        expected = np.array(df.grade.values).tolist()
        tm.assert_almost_equal(result, expected)

        # iteration
        for t in df.itertuples(index=False):
            str(t)

        for row, s in df.iterrows():
            str(s)

        for c, col in df.items():
            str(col)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_conversion.py ===
from datetime import datetime

import numpy as np
import pytest
from pytz import UTC

from pandas._libs.tslibs import (
    OutOfBoundsTimedelta,
    astype_overflowsafe,
    conversion,
    iNaT,
    timezones,
    tz_convert_from_utc,
    tzconversion,
)

from pandas import (
    Timestamp,
    date_range,
)
import pandas._testing as tm


def _compare_utc_to_local(tz_didx):
    def f(x):
        return tzconversion.tz_convert_from_utc_single(x, tz_didx.tz)

    result = tz_convert_from_utc(tz_didx.asi8, tz_didx.tz)
    expected = np.vectorize(f)(tz_didx.asi8)

    tm.assert_numpy_array_equal(result, expected)


def _compare_local_to_utc(tz_didx, naive_didx):
    # Check that tz_localize behaves the same vectorized and pointwise.
    err1 = err2 = None
    try:
        result = tzconversion.tz_localize_to_utc(naive_didx.asi8, tz_didx.tz)
        err1 = None
    except Exception as err:
        err1 = err

    try:
        expected = naive_didx.map(lambda x: x.tz_localize(tz_didx.tz)).asi8
    except Exception as err:
        err2 = err

    if err1 is not None:
        assert type(err1) == type(err2)
    else:
        assert err2 is None
        tm.assert_numpy_array_equal(result, expected)


def test_tz_localize_to_utc_copies():
    # GH#46460
    arr = np.arange(5, dtype="i8")
    result = tz_convert_from_utc(arr, tz=UTC)
    tm.assert_numpy_array_equal(result, arr)
    assert not np.shares_memory(arr, result)

    result = tz_convert_from_utc(arr, tz=None)
    tm.assert_numpy_array_equal(result, arr)
    assert not np.shares_memory(arr, result)


def test_tz_convert_single_matches_tz_convert_hourly(tz_aware_fixture):
    tz = tz_aware_fixture
    tz_didx = date_range("2014-03-01", "2015-01-10", freq="h", tz=tz)
    naive_didx = date_range("2014-03-01", "2015-01-10", freq="h")

    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)


@pytest.mark.parametrize("freq", ["D", "YE"])
def test_tz_convert_single_matches_tz_convert(tz_aware_fixture, freq):
    tz = tz_aware_fixture
    tz_didx = date_range("2018-01-01", "2020-01-01", freq=freq, tz=tz)
    naive_didx = date_range("2018-01-01", "2020-01-01", freq=freq)

    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(np.array([], dtype=np.int64), id="empty"),
        pytest.param(np.array([iNaT], dtype=np.int64), id="all_nat"),
    ],
)
def test_tz_convert_corner(arr):
    result = tz_convert_from_utc(arr, timezones.maybe_get_tz("Asia/Tokyo"))
    tm.assert_numpy_array_equal(result, arr)


def test_tz_convert_readonly():
    # GH#35530
    arr = np.array([0], dtype=np.int64)
    arr.setflags(write=False)
    result = tz_convert_from_utc(arr, UTC)
    tm.assert_numpy_array_equal(result, arr)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("dtype", ["M8[ns]", "M8[s]"])
def test_length_zero_copy(dtype, copy):
    arr = np.array([], dtype=dtype)
    result = astype_overflowsafe(arr, copy=copy, dtype=np.dtype("M8[ns]"))
    if copy:
        assert not np.shares_memory(result, arr)
    elif arr.dtype == result.dtype:
        assert result is arr
    else:
        assert not np.shares_memory(result, arr)


def test_ensure_datetime64ns_bigendian():
    # GH#29684
    arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")
    result = astype_overflowsafe(arr, dtype=np.dtype("M8[ns]"))

    expected = np.array([np.datetime64(1, "ms")], dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)


def test_ensure_timedelta64ns_overflows():
    arr = np.arange(10).astype("m8[Y]") * 100
    msg = r"Cannot convert 300 years to timedelta64\[ns\] without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        astype_overflowsafe(arr, dtype=np.dtype("m8[ns]"))


class SubDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "dt, expected",
    [
        pytest.param(
            Timestamp("2000-01-01"), Timestamp("2000-01-01", tz=UTC), id="timestamp"
        ),
        pytest.param(
            datetime(2000, 1, 1), datetime(2000, 1, 1, tzinfo=UTC), id="datetime"
        ),
        pytest.param(
            SubDatetime(2000, 1, 1),
            SubDatetime(2000, 1, 1, tzinfo=UTC),
            id="subclassed_datetime",
        ),
    ],
)
def test_localize_pydatetime_dt_types(dt, expected):
    # GH 25851
    # ensure that subclassed datetime works with
    # localize_pydatetime
    result = conversion.localize_pydatetime(dt, UTC)
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_diff.py ===
from sympy.concrete.summations import Sum
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, diff, Subs)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.tensor.array.ndim_array import NDimArray
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, x, y, z

def test_diff():
    assert Rational(1, 3).diff(x) is S.Zero
    assert I.diff(x) is S.Zero
    assert pi.diff(x) is S.Zero
    assert x.diff(x, 0) == x
    assert (x**2).diff(x, 2, x) == 0
    assert (x**2).diff((x, 2), x) == 0
    assert (x**2).diff((x, 1), x) == 2
    assert (x**2).diff((x, 1), (x, 1)) == 2
    assert (x**2).diff((x, 2)) == 2
    assert (x**2).diff(x, y, 0) == 2*x
    assert (x**2).diff(x, (y, 0)) == 2*x
    assert (x**2).diff(x, y) == 0
    raises(ValueError, lambda: x.diff(1, x))

    p = Rational(5)
    e = a*b + b**p
    assert e.diff(a) == b
    assert e.diff(b) == a + 5*b**4
    assert e.diff(b).diff(a) == Rational(1)
    e = a*(b + c)
    assert e.diff(a) == b + c
    assert e.diff(b) == a
    assert e.diff(b).diff(a) == Rational(1)
    e = c**p
    assert e.diff(c, 6) == Rational(0)
    assert e.diff(c, 5) == Rational(120)
    e = c**Rational(2)
    assert e.diff(c) == 2*c
    e = a*b*c
    assert e.diff(c) == a*b


def test_diff2():
    n3 = Rational(3)
    n2 = Rational(2)
    n6 = Rational(6)

    e = n3*(-n2 + x**n2)*cos(x) + x*(-n6 + x**n2)*sin(x)
    assert e == 3*(-2 + x**2)*cos(x) + x*(-6 + x**2)*sin(x)
    assert e.diff(x).expand() == x**3*cos(x)

    e = (x + 1)**3
    assert e.diff(x) == 3*(x + 1)**2
    e = x*(x + 1)**3
    assert e.diff(x) == (x + 1)**3 + 3*x*(x + 1)**2
    e = 2*exp(x*x)*x
    assert e.diff(x) == 2*exp(x**2) + 4*x**2*exp(x**2)


def test_diff3():
    p = Rational(5)
    e = a*b + sin(b**p)
    assert e == a*b + sin(b**5)
    assert e.diff(a) == b
    assert e.diff(b) == a + 5*b**4*cos(b**5)
    e = tan(c)
    assert e == tan(c)
    assert e.diff(c) in [cos(c)**(-2), 1 + sin(c)**2/cos(c)**2, 1 + tan(c)**2]
    e = c*log(c) - c
    assert e == -c + c*log(c)
    assert e.diff(c) == log(c)
    e = log(sin(c))
    assert e == log(sin(c))
    assert e.diff(c) in [sin(c)**(-1)*cos(c), cot(c)]
    e = (Rational(2)**a/log(Rational(2)))
    assert e == 2**a*log(Rational(2))**(-1)
    assert e.diff(a) == 2**a


def test_diff_no_eval_derivative():
    class My(Expr):
        def __new__(cls, x):
            return Expr.__new__(cls, x)

    # My doesn't have its own _eval_derivative method
    assert My(x).diff(x).func is Derivative
    assert My(x).diff(x, 3).func is Derivative
    assert re(x).diff(x, 2) == Derivative(re(x), (x, 2))  # issue 15518
    assert diff(NDimArray([re(x), im(x)]), (x, 2)) == NDimArray(
        [Derivative(re(x), (x, 2)), Derivative(im(x), (x, 2))])
    # it doesn't have y so it shouldn't need a method for this case
    assert My(x).diff(y) == 0


def test_speed():
    # this should return in 0.0s. If it takes forever, it's wrong.
    assert x.diff(x, 10**8) == 0


def test_deriv_noncommutative():
    A = Symbol("A", commutative=False)
    f = Function("f")
    assert A*f(x)*A == f(x)*A**2
    assert A*f(x).diff(x)*A == f(x).diff(x) * A**2


def test_diff_nth_derivative():
    f =  Function("f")
    n = Symbol("n", integer=True)

    expr = diff(sin(x), (x, n))
    expr2 = diff(f(x), (x, 2))
    expr3 = diff(f(x), (x, n))

    assert expr.subs(sin(x), cos(-x)) == Derivative(cos(-x), (x, n))
    assert expr.subs(n, 1).doit() == cos(x)
    assert expr.subs(n, 2).doit() == -sin(x)

    assert expr2.subs(Derivative(f(x), x), y) == Derivative(y, x)
    # Currently not supported (cannot determine if `n > 1`):
    #assert expr3.subs(Derivative(f(x), x), y) == Derivative(y, (x, n-1))
    assert expr3 == Derivative(f(x), (x, n))

    assert diff(x, (x, n)) == Piecewise((x, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
    assert diff(2*x, (x, n)).dummy_eq(
        Sum(Piecewise((2*x*factorial(n)/(factorial(y)*factorial(-y + n)),
        Eq(y, 0) & Eq(Max(0, -y + n), 0)),
        (2*factorial(n)/(factorial(y)*factorial(-y + n)), Eq(y, 0) & Eq(Max(0,
        -y + n), 1)), (0, True)), (y, 0, n)))
    # TODO: assert diff(x**2, (x, n)) == x**(2-n)*ff(2, n)
    exprm = x*sin(x)
    mul_diff = diff(exprm, (x, n))
    assert isinstance(mul_diff, Sum)
    for i in range(5):
        assert mul_diff.subs(n, i).doit() == exprm.diff((x, i)).expand()

    exprm2 = 2*y*x*sin(x)*cos(x)*log(x)*exp(x)
    dex = exprm2.diff((x, n))
    assert isinstance(dex, Sum)
    for i in range(7):
        assert dex.subs(n, i).doit().expand() == \
        exprm2.diff((x, i)).expand()

    assert (cos(x)*sin(y)).diff([[x, y, z]]) == NDimArray([
        -sin(x)*sin(y), cos(x)*cos(y), 0])


def test_issue_16160():
    assert Derivative(x**3, (x, x)).subs(x, 2) == Subs(
        Derivative(x**3, (x, 2)), x, 2)
    assert Derivative(1 + x**3, (x, x)).subs(x, 0
        ) == Derivative(1 + y**3, (y, 0)).subs(y, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_sqfreetools.py ===
"""Tests for square-free decomposition algorithms and related tools. """

from sympy.polys.rings import ring
from sympy.polys.domains import FF, ZZ, QQ
from sympy.polys.specialpolys import f_polys

from sympy.testing.pytest import raises
from sympy.external.gmpy import MPQ

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()

def test_dup_sqf():
    R, x = ring("x", ZZ)

    assert R.dup_sqf_part(0) == 0
    assert R.dup_sqf_p(0) is True

    assert R.dup_sqf_part(7) == 1
    assert R.dup_sqf_p(7) is True

    assert R.dup_sqf_part(2*x + 2) == x + 1
    assert R.dup_sqf_p(2*x + 2) is True

    assert R.dup_sqf_part(x**3 + x + 1) == x**3 + x + 1
    assert R.dup_sqf_p(x**3 + x + 1) is True

    assert R.dup_sqf_part(-x**3 + x + 1) == x**3 - x - 1
    assert R.dup_sqf_p(-x**3 + x + 1) is True

    assert R.dup_sqf_part(2*x**3 + 3*x**2) == 2*x**2 + 3*x
    assert R.dup_sqf_p(2*x**3 + 3*x**2) is False

    assert R.dup_sqf_part(-2*x**3 + 3*x**2) == 2*x**2 - 3*x
    assert R.dup_sqf_p(-2*x**3 + 3*x**2) is False

    assert R.dup_sqf_list(0) == (0, [])
    assert R.dup_sqf_list(1) == (1, [])

    assert R.dup_sqf_list(x) == (1, [(x, 1)])
    assert R.dup_sqf_list(2*x**2) == (2, [(x, 2)])
    assert R.dup_sqf_list(3*x**3) == (3, [(x, 3)])

    assert R.dup_sqf_list(-x**5 + x**4 + x - 1) == \
        (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dup_sqf_list(x**8 + 6*x**6 + 12*x**4 + 8*x**2) == \
        ( 1, [(x, 2), (x**2 + 2, 3)])

    assert R.dup_sqf_list(2*x**2 + 4*x + 2) == (2, [(x + 1, 2)])

    R, x = ring("x", QQ)
    assert R.dup_sqf_list(2*x**2 + 4*x + 2) == (2, [(x + 1, 2)])

    R, x = ring("x", FF(2))
    assert R.dup_sqf_list(x**2 + 1) == (1, [(x + 1, 2)])

    R, x = ring("x", FF(3))
    assert R.dup_sqf_list(x**10 + 2*x**7 + 2*x**4 + x) == \
        (1, [(x, 1),
             (x + 1, 3),
             (x + 2, 6)])

    R1, x = ring("x", ZZ)
    R2, y = ring("y", FF(3))

    f = x**3 + 1
    g = y**3 + 1

    assert R1.dup_sqf_part(f) == f
    assert R2.dup_sqf_part(g) == y + 1

    assert R1.dup_sqf_p(f) is True
    assert R2.dup_sqf_p(g) is False

    R, x, y = ring("x,y", ZZ)

    A = x**4 - 3*x**2 + 6
    D = x**6 - 5*x**4 + 5*x**2 + 4

    f, g = D, R.dmp_sub(A, R.dmp_mul(R.dmp_diff(D, 1), y))
    res = R.dmp_resultant(f, g)
    h = (4*y**2 + 1).drop(x)

    assert R.drop(x).dup_sqf_list(res) == (45796, [(h, 3)])

    Rt, t = ring("t", ZZ)
    R, x = ring("x", Rt)
    assert R.dup_sqf_list_include(t**3*x**2) == [(t**3, 1), (x, 2)]


def test_dmp_sqf():
    R, x, y = ring("x,y", ZZ)
    assert R.dmp_sqf_part(0) == 0
    assert R.dmp_sqf_p(0) is True

    assert R.dmp_sqf_part(7) == 1
    assert R.dmp_sqf_p(7) is True

    assert R.dmp_sqf_list(3) == (3, [])
    assert R.dmp_sqf_list_include(3) == [(3, 1)]

    R, x, y, z = ring("x,y,z", ZZ)
    assert R.dmp_sqf_p(f_0) is True
    assert R.dmp_sqf_p(f_0**2) is False
    assert R.dmp_sqf_p(f_1) is True
    assert R.dmp_sqf_p(f_1**2) is False
    assert R.dmp_sqf_p(f_2) is True
    assert R.dmp_sqf_p(f_2**2) is False
    assert R.dmp_sqf_p(f_3) is True
    assert R.dmp_sqf_p(f_3**2) is False
    assert R.dmp_sqf_p(f_5) is False
    assert R.dmp_sqf_p(f_5**2) is False

    assert R.dmp_sqf_p(f_4) is True
    assert R.dmp_sqf_part(f_4) == -f_4

    assert R.dmp_sqf_part(f_5) == x + y - z

    R, x, y, z, t = ring("x,y,z,t", ZZ)
    assert R.dmp_sqf_p(f_6) is True
    assert R.dmp_sqf_part(f_6) == f_6

    R, x = ring("x", ZZ)
    f = -x**5 + x**4 + x - 1

    assert R.dmp_sqf_list(f) == (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list_include(f) == [(-x**3 - x**2 - x - 1, 1), (x - 1, 2)]

    R, x, y = ring("x,y", ZZ)
    f = -x**5 + x**4 + x - 1

    assert R.dmp_sqf_list(f) == (-1, [(x**3 + x**2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list_include(f) == [(-x**3 - x**2 - x - 1, 1), (x - 1, 2)]

    f = -x**2 + 2*x - 1
    assert R.dmp_sqf_list_include(f) == [(-1, 1), (x - 1, 2)]

    f = (y**2 + 1)**2*(x**2 + 2*x + 2)
    assert R.dmp_sqf_p(f) is False
    assert R.dmp_sqf_list(f) == (1, [(x**2 + 2*x + 2, 1), (y**2 + 1, 2)])

    R, x, y = ring("x,y", FF(2))
    raises(NotImplementedError, lambda: R.dmp_sqf_list(y**2 + 1))


def test_dup_gff_list():
    R, x = ring("x", ZZ)

    f = x**5 + 2*x**4 - x**3 - 2*x**2
    assert R.dup_gff_list(f) == [(x, 1), (x + 2, 4)]

    g = x**9 - 20*x**8 + 166*x**7 - 744*x**6 + 1965*x**5 - 3132*x**4 + 2948*x**3 - 1504*x**2 + 320*x
    assert R.dup_gff_list(g) == [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]

    raises(ValueError, lambda: R.dup_gff_list(0))

def test_issue_26178():
    R, x, y, z = ring(['x', 'y', 'z'], QQ)
    assert (x**2 - 2*y**2 + 1).sqf_list() == (MPQ(1,1), [(x**2 - 2*y**2 + 1, 1)])
    assert (x**2 - 2*z**2 + 1).sqf_list() == (MPQ(1,1), [(x**2 - 2*z**2 + 1, 1)])
    assert (y**2 - 2*z**2 + 1).sqf_list() == (MPQ(1,1), [(y**2 - 2*z**2 + 1, 1)])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\certifi\__init__.py ===
from .core import contents, where

__all__ = ["contents", "where"]
__version__ = "2025.06.15"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\_version.py ===
# file generated by setuptools_scm
# don't change, don't track in version control
__version__ = version = '2.9.0.post0'
__version_tuple__ = version_tuple = (2, 9, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\cell\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

from .cell import Cell, WriteOnlyCell, MergedCell
from .read_only import ReadOnlyCell

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\comments\__init__.py ===
# Copyright (c) 2010-2024 openpyxl


from .comments import Comment

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\__init__.py ===
# Copyright (c) 2010-2024 openpyxl


from .drawing import Drawing

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\__init__.py ===
# Copyright (c) 2010-2024 openpyxl


from .workbook import Workbook

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\test_series.py ===
from operator import methodcaller

import numpy as np
import pytest

import pandas as pd
from pandas import (
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestSeries:
    @pytest.mark.parametrize("func", ["rename_axis", "_set_axis_name"])
    def test_set_axis_name_mi(self, func):
        ser = Series(
            [11, 21, 31],
            index=MultiIndex.from_tuples(
                [("A", x) for x in ["a", "B", "c"]], names=["l1", "l2"]
            ),
        )

        result = methodcaller(func, ["L1", "L2"])(ser)
        assert ser.index.name is None
        assert ser.index.names == ["l1", "l2"]
        assert result.index.name is None
        assert result.index.names, ["L1", "L2"]

    def test_set_axis_name_raises(self):
        ser = Series([1])
        msg = "No axis named 1 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser._set_axis_name(name="a", axis=1)

    def test_get_bool_data_preserve_dtype(self):
        ser = Series([True, False, True])
        result = ser._get_bool_data()
        tm.assert_series_equal(result, ser)

    def test_nonzero_single_element(self):
        # allow single item via bool method
        msg_warn = (
            "Series.bool is now deprecated and will be removed "
            "in future version of pandas"
        )
        ser = Series([True])
        ser1 = Series([False])
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            assert ser.bool()
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            assert not ser1.bool()

    @pytest.mark.parametrize("data", [np.nan, pd.NaT, True, False])
    def test_nonzero_single_element_raise_1(self, data):
        # single item nan to raise
        series = Series([data])

        msg = "The truth value of a Series is ambiguous"
        with pytest.raises(ValueError, match=msg):
            bool(series)

    @pytest.mark.parametrize("data", [np.nan, pd.NaT])
    def test_nonzero_single_element_raise_2(self, data):
        msg_warn = (
            "Series.bool is now deprecated and will be removed "
            "in future version of pandas"
        )
        msg_err = "bool cannot act on a non-boolean single element Series"
        series = Series([data])
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            with pytest.raises(ValueError, match=msg_err):
                series.bool()

    @pytest.mark.parametrize("data", [(True, True), (False, False)])
    def test_nonzero_multiple_element_raise(self, data):
        # multiple bool are still an error
        msg_warn = (
            "Series.bool is now deprecated and will be removed "
            "in future version of pandas"
        )
        msg_err = "The truth value of a Series is ambiguous"
        series = Series([data])
        with pytest.raises(ValueError, match=msg_err):
            bool(series)
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            with pytest.raises(ValueError, match=msg_err):
                series.bool()

    @pytest.mark.parametrize("data", [1, 0, "a", 0.0])
    def test_nonbool_single_element_raise(self, data):
        # single non-bool are an error
        msg_warn = (
            "Series.bool is now deprecated and will be removed "
            "in future version of pandas"
        )
        msg_err1 = "The truth value of a Series is ambiguous"
        msg_err2 = "bool cannot act on a non-boolean single element Series"
        series = Series([data])
        with pytest.raises(ValueError, match=msg_err1):
            bool(series)
        with tm.assert_produces_warning(FutureWarning, match=msg_warn):
            with pytest.raises(ValueError, match=msg_err2):
                series.bool()

    def test_metadata_propagation_indiv_resample(self):
        # resample
        ts = Series(
            np.random.default_rng(2).random(1000),
            index=date_range("20130101", periods=1000, freq="s"),
            name="foo",
        )
        result = ts.resample("1min").mean()
        tm.assert_metadata_equivalent(ts, result)

        result = ts.resample("1min").min()
        tm.assert_metadata_equivalent(ts, result)

        result = ts.resample("1min").apply(lambda x: x.sum())
        tm.assert_metadata_equivalent(ts, result)

    def test_metadata_propagation_indiv(self, monkeypatch):
        # check that the metadata matches up on the resulting ops

        ser = Series(range(3), range(3))
        ser.name = "foo"
        ser2 = Series(range(3), range(3))
        ser2.name = "bar"

        result = ser.T
        tm.assert_metadata_equivalent(ser, result)

        def finalize(self, other, method=None, **kwargs):
            for name in self._metadata:
                if method == "concat" and name == "filename":
                    value = "+".join(
                        [
                            getattr(obj, name)
                            for obj in other.objs
                            if getattr(obj, name, None)
                        ]
                    )
                    object.__setattr__(self, name, value)
                else:
                    object.__setattr__(self, name, getattr(other, name, None))

            return self

        with monkeypatch.context() as m:
            m.setattr(Series, "_metadata", ["name", "filename"])
            m.setattr(Series, "__finalize__", finalize)

            ser.filename = "foo"
            ser2.filename = "bar"

            result = pd.concat([ser, ser2])
            assert result.filename == "foo+bar"
            assert result.name is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\object\test_indexing.py ===
from decimal import Decimal

import numpy as np
import pytest

from pandas._libs.missing import is_matching_na

from pandas import Index
import pandas._testing as tm


class TestGetIndexer:
    @pytest.mark.parametrize(
        "method,expected",
        [
            ("pad", np.array([-1, 0, 1, 1], dtype=np.intp)),
            ("backfill", np.array([0, 0, 1, -1], dtype=np.intp)),
        ],
    )
    def test_get_indexer_strings(self, method, expected):
        expected = np.array(expected, dtype=np.intp)
        index = Index(["b", "c"], dtype=object)
        actual = index.get_indexer(["a", "b", "c", "d"], method=method)

        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_strings_raises(self):
        index = Index(["b", "c"], dtype=object)

        msg = "|".join(
            [
                "operation 'sub' not supported for dtype 'str'",
                r"unsupported operand type\(s\) for -: 'str' and 'str'",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="nearest")

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(["a", "b", "c", "d"], method="pad", tolerance=2)

        with pytest.raises(TypeError, match=msg):
            index.get_indexer(
                ["a", "b", "c", "d"], method="pad", tolerance=[2, 2, 2, 2]
            )

    def test_get_indexer_with_NA_values(
        self, unique_nulls_fixture, unique_nulls_fixture2
    ):
        # GH#22332
        # check pairwise, that no pair of na values
        # is mangled
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # skip it, values are not unique
        arr = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
        index = Index(arr, dtype=object)
        result = index.get_indexer(
            Index(
                [unique_nulls_fixture, unique_nulls_fixture2, "Unknown"], dtype=object
            )
        )
        expected = np.array([0, 1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_infer_string_missing_values(self):
        # ensure the passed list is not cast to string but to object so that
        # the None value is matched in the index
        # https://github.com/pandas-dev/pandas/issues/55834
        idx = Index(["a", "b", None], dtype="object")
        result = idx.get_indexer([None, "x"])
        expected = np.array([2, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)


class TestGetIndexerNonUnique:
    def test_get_indexer_non_unique_nas(self, nulls_fixture):
        # even though this isn't non-unique, this should still work
        index = Index(["a", "b", nulls_fixture], dtype=object)
        indexer, missing = index.get_indexer_non_unique([nulls_fixture])

        expected_indexer = np.array([2], dtype=np.intp)
        expected_missing = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # actually non-unique
        index = Index(["a", nulls_fixture, "b", nulls_fixture], dtype=object)
        indexer, missing = index.get_indexer_non_unique([nulls_fixture])

        expected_indexer = np.array([1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # matching-but-not-identical nans
        if is_matching_na(nulls_fixture, float("NaN")):
            index = Index(["a", float("NaN"), "b", float("NaN")], dtype=object)
            match_but_not_identical = True
        elif is_matching_na(nulls_fixture, Decimal("NaN")):
            index = Index(["a", Decimal("NaN"), "b", Decimal("NaN")], dtype=object)
            match_but_not_identical = True
        else:
            match_but_not_identical = False

        if match_but_not_identical:
            indexer, missing = index.get_indexer_non_unique([nulls_fixture])

            expected_indexer = np.array([1, 3], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)

    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    def test_get_indexer_non_unique_np_nats(self, np_nat_fixture, np_nat_fixture2):
        expected_missing = np.array([], dtype=np.intp)
        # matching-but-not-identical nats
        if is_matching_na(np_nat_fixture, np_nat_fixture2):
            # ensure nats are different objects
            index = Index(
                np.array(
                    ["2021-10-02", np_nat_fixture.copy(), np_nat_fixture2.copy()],
                    dtype=object,
                ),
                dtype=object,
            )
            # pass as index to prevent target from being casted to DatetimeIndex
            indexer, missing = index.get_indexer_non_unique(
                Index([np_nat_fixture], dtype=object)
            )
            expected_indexer = np.array([1, 2], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)
        # dt64nat vs td64nat
        else:
            try:
                np_nat_fixture == np_nat_fixture2
            except (TypeError, OverflowError):
                # Numpy will raise on uncomparable types, like
                # np.datetime64('NaT', 'Y') and np.datetime64('NaT', 'ps')
                # https://github.com/numpy/numpy/issues/22762
                return
            index = Index(
                np.array(
                    [
                        "2021-10-02",
                        np_nat_fixture,
                        np_nat_fixture2,
                        np_nat_fixture,
                        np_nat_fixture2,
                    ],
                    dtype=object,
                ),
                dtype=object,
            )
            # pass as index to prevent target from being casted to DatetimeIndex
            indexer, missing = index.get_indexer_non_unique(
                Index([np_nat_fixture], dtype=object)
            )
            expected_indexer = np.array([1, 3], dtype=np.intp)
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\certifi\__init__.py ===
from .core import contents, where

__all__ = ["contents", "where"]
__version__ = "2025.01.31"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distro\__main__.py ===
from .distro import main

if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\tomli_w\__init__.py ===
__all__ = ("dumps", "dump")
__version__ = "1.2.0"  # DO NOT EDIT THIS LINE MANUALLY. LET bump2version UTILITY DO IT

from pip._vendor.tomli_w._writer import dump, dumps

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\alphabets.py ===
greeks = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
                    'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu',
                    'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
                    'phi', 'chi', 'psi', 'omega')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\__init__.py ===
"""Used for translating a string into a SymPy expression. """
__all__ = ['parse_expr']

from .sympy_parser import parse_expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_pauli.py ===
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
                                   Operator, represent)
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
                                         SigmaMinus, SigmaPlus,
                                         qsimplify_pauli)
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises


sx, sy, sz = SigmaX(), SigmaY(), SigmaZ()
sx1, sy1, sz1 = SigmaX(1), SigmaY(1), SigmaZ(1)
sx2, sy2, sz2 = SigmaX(2), SigmaY(2), SigmaZ(2)

sm, sp = SigmaMinus(), SigmaPlus()
sm1, sp1 = SigmaMinus(1), SigmaPlus(1)
A, B = Operator("A"), Operator("B")


def test_pauli_operators_types():

    assert isinstance(sx, SigmaOpBase) and isinstance(sx, SigmaX)
    assert isinstance(sy, SigmaOpBase) and isinstance(sy, SigmaY)
    assert isinstance(sz, SigmaOpBase) and isinstance(sz, SigmaZ)
    assert isinstance(sm, SigmaOpBase) and isinstance(sm, SigmaMinus)
    assert isinstance(sp, SigmaOpBase) and isinstance(sp, SigmaPlus)


def test_pauli_operators_commutator():

    assert Commutator(sx, sy).doit() == 2 * I * sz
    assert Commutator(sy, sz).doit() == 2 * I * sx
    assert Commutator(sz, sx).doit() == 2 * I * sy


def test_pauli_operators_commutator_with_labels():

    assert Commutator(sx1, sy1).doit() == 2 * I * sz1
    assert Commutator(sy1, sz1).doit() == 2 * I * sx1
    assert Commutator(sz1, sx1).doit() == 2 * I * sy1

    assert Commutator(sx2, sy2).doit() == 2 * I * sz2
    assert Commutator(sy2, sz2).doit() == 2 * I * sx2
    assert Commutator(sz2, sx2).doit() == 2 * I * sy2

    assert Commutator(sx1, sy2).doit() == 0
    assert Commutator(sy1, sz2).doit() == 0
    assert Commutator(sz1, sx2).doit() == 0


def test_pauli_operators_anticommutator():

    assert AntiCommutator(sy, sz).doit() == 0
    assert AntiCommutator(sz, sx).doit() == 0
    assert AntiCommutator(sx, sm).doit() == 1
    assert AntiCommutator(sx, sp).doit() == 1


def test_pauli_operators_adjoint():

    assert Dagger(sx) == sx
    assert Dagger(sy) == sy
    assert Dagger(sz) == sz


def test_pauli_operators_adjoint_with_labels():

    assert Dagger(sx1) == sx1
    assert Dagger(sy1) == sy1
    assert Dagger(sz1) == sz1

    assert Dagger(sx1) != sx2
    assert Dagger(sy1) != sy2
    assert Dagger(sz1) != sz2


def test_pauli_operators_multiplication():

    assert qsimplify_pauli(sx * sx) == 1
    assert qsimplify_pauli(sy * sy) == 1
    assert qsimplify_pauli(sz * sz) == 1

    assert qsimplify_pauli(sx * sy) == I * sz
    assert qsimplify_pauli(sy * sz) == I * sx
    assert qsimplify_pauli(sz * sx) == I * sy

    assert qsimplify_pauli(sy * sx) == - I * sz
    assert qsimplify_pauli(sz * sy) == - I * sx
    assert qsimplify_pauli(sx * sz) == - I * sy


def test_pauli_operators_multiplication_with_labels():

    assert qsimplify_pauli(sx1 * sx1) == 1
    assert qsimplify_pauli(sy1 * sy1) == 1
    assert qsimplify_pauli(sz1 * sz1) == 1

    assert isinstance(sx1 * sx2, Mul)
    assert isinstance(sy1 * sy2, Mul)
    assert isinstance(sz1 * sz2, Mul)

    assert qsimplify_pauli(sx1 * sy1 * sx2 * sy2) == - sz1 * sz2
    assert qsimplify_pauli(sy1 * sz1 * sz2 * sx2) == - sx1 * sy2


def test_pauli_states():
    sx, sz = SigmaX(), SigmaZ()

    up = SigmaZKet(0)
    down = SigmaZKet(1)

    assert qapply(sx * up) == down
    assert qapply(sx * down) == up
    assert qapply(sz * up) == up
    assert qapply(sz * down) == - down

    up = SigmaZBra(0)
    down = SigmaZBra(1)

    assert qapply(up * sx, dagger=True) == down
    assert qapply(down * sx, dagger=True) == up
    assert qapply(up * sz, dagger=True) == up
    assert qapply(down * sz, dagger=True) == - down

    assert Dagger(SigmaZKet(0)) == SigmaZBra(0)
    assert Dagger(SigmaZBra(1)) == SigmaZKet(1)
    raises(ValueError, lambda: SigmaZBra(2))
    raises(ValueError, lambda: SigmaZKet(2))


def test_use_name():
    assert sm.use_name is False
    assert sm1.use_name is True
    assert sx.use_name is False
    assert sx1.use_name is True


def test_printing():
    assert latex(sx) == r'{\sigma_x}'
    assert latex(sx1) == r'{\sigma_x^{(1)}}'
    assert latex(sy) == r'{\sigma_y}'
    assert latex(sy1) == r'{\sigma_y^{(1)}}'
    assert latex(sz) == r'{\sigma_z}'
    assert latex(sz1) == r'{\sigma_z^{(1)}}'
    assert latex(sm) == r'{\sigma_-}'
    assert latex(sm1) == r'{\sigma_-^{(1)}}'
    assert latex(sp) == r'{\sigma_+}'
    assert latex(sp1) == r'{\sigma_+^{(1)}}'


def test_represent():
    assert represent(sx) == Matrix([[0, 1], [1, 0]])
    assert represent(sy) == Matrix([[0, -I], [I, 0]])
    assert represent(sz) == Matrix([[1, 0], [0, -1]])
    assert represent(sm) == Matrix([[0, 0], [1, 0]])
    assert represent(sp) == Matrix([[0, 1], [0, 0]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_compound_rv.py ===
from sympy.concrete.summations import Sum
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.sets import Interval
from sympy.stats import (Normal, P, E, density, Gamma, Poisson, Rayleigh,
                        variance, Bernoulli, Beta, Uniform, cdf)
from sympy.stats.compound_rv import CompoundDistribution, CompoundPSpace
from sympy.stats.crv_types import NormalDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.frv_types import BernoulliDistribution
from sympy.testing.pytest import raises, ignore_warnings
from sympy.stats.joint_rv_types import MultivariateNormalDistribution

from sympy.abc import x


# helpers for testing troublesome unevaluated expressions
flat = lambda s: ''.join(str(s).split())
streq = lambda *a: len(set(map(flat, a))) == 1
assert streq(x, x)
assert streq(x, 'x')
assert not streq(x, x + 1)


def test_normal_CompoundDist():
    X = Normal('X', 1, 2)
    Y = Normal('X', X, 4)
    assert density(Y)(x).simplify() == sqrt(10)*exp(-x**2/40 + x/20 - S(1)/40)/(20*sqrt(pi))
    assert E(Y) == 1 # it is always equal to mean of X
    assert P(Y > 1) == S(1)/2 # as 1 is the mean
    assert P(Y > 5).simplify() ==  S(1)/2 - erf(sqrt(10)/5)/2
    assert variance(Y) == variance(X) + 4**2 # 2**2 + 4**2
    # https://math.stackexchange.com/questions/1484451/
    # (Contains proof of E and variance computation)


def test_poisson_CompoundDist():
    k, t, y = symbols('k t y', positive=True, real=True)
    G = Gamma('G', k, t)
    D = Poisson('P', G)
    assert density(D)(y).simplify() == t**y*(t + 1)**(-k - y)*gamma(k + y)/(gamma(k)*gamma(y + 1))
    # https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture
    assert E(D).simplify() == k*t # mean of NegativeBinomialDistribution


def test_bernoulli_CompoundDist():
    X = Beta('X', 1, 2)
    Y = Bernoulli('Y', X)
    assert density(Y).dict == {0: S(2)/3, 1: S(1)/3}
    assert E(Y) == P(Eq(Y, 1)) == S(1)/3
    assert variance(Y) == S(2)/9
    assert cdf(Y) == {0: S(2)/3, 1: 1}

    # test issue 8128
    a = Bernoulli('a', S(1)/2)
    b = Bernoulli('b', a)
    assert density(b).dict == {0: S(1)/2, 1: S(1)/2}
    assert P(b > 0.5) == S(1)/2

    X = Uniform('X', 0, 1)
    Y = Bernoulli('Y', X)
    assert E(Y) == S(1)/2
    assert P(Eq(Y, 1)) == E(Y)


def test_unevaluated_CompoundDist():
    # these tests need to be removed once they work with evaluation as they are currently not
    # evaluated completely in sympy.
    R = Rayleigh('R', 4)
    X = Normal('X', 3, R)
    ans = '''
        Piecewise(((-sqrt(pi)*sinh(x/4 - 3/4) + sqrt(pi)*cosh(x/4 - 3/4))/(
        8*sqrt(pi)), Abs(arg(x - 3)) <= pi/4), (Integral(sqrt(2)*exp(-(x - 3)
        **2/(2*R**2))*exp(-R**2/32)/(32*sqrt(pi)), (R, 0, oo)), True))'''
    assert streq(density(X)(x), ans)

    expre = '''
        Integral(X*Integral(sqrt(2)*exp(-(X-3)**2/(2*R**2))*exp(-R**2/32)/(32*
        sqrt(pi)),(R,0,oo)),(X,-oo,oo))'''
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert streq(E(X, evaluate=False).rewrite(Integral), expre)

    X = Poisson('X', 1)
    Y = Poisson('Y', X)
    Z = Poisson('Z', Y)
    exprd = Sum(exp(-Y)*Y**x*Sum(exp(-1)*exp(-X)*X**Y/(factorial(X)*factorial(Y)
                ), (X, 0, oo))/factorial(x), (Y, 0, oo))
    assert density(Z)(x) == exprd

    N = Normal('N', 1, 2)
    M = Normal('M', 3, 4)
    D = Normal('D', M, N)
    exprd = '''
        Integral(sqrt(2)*exp(-(N-1)**2/8)*Integral(exp(-(x-M)**2/(2*N**2))*exp
        (-(M-3)**2/32)/(8*pi*N),(M,-oo,oo))/(4*sqrt(pi)),(N,-oo,oo))'''
    assert streq(density(D, evaluate=False)(x), exprd)


def test_Compound_Distribution():
    X = Normal('X', 2, 4)
    N = NormalDistribution(X, 4)
    C = CompoundDistribution(N)
    assert C.is_Continuous
    assert C.set == Interval(-oo, oo)
    assert C.pdf(x, evaluate=True).simplify() == exp(-x**2/64 + x/16 - S(1)/16)/(8*sqrt(pi))

    assert not isinstance(CompoundDistribution(NormalDistribution(2, 3)),
                            CompoundDistribution)
    M = MultivariateNormalDistribution([1, 2], [[2, 1], [1, 2]])
    raises(NotImplementedError, lambda: CompoundDistribution(M))

    X = Beta('X', 2, 4)
    B = BernoulliDistribution(X, 1, 0)
    C = CompoundDistribution(B)
    assert C.is_Finite
    assert C.set == {0, 1}
    y = symbols('y', negative=False, integer=True)
    assert C.pdf(y, evaluate=True) == Piecewise((S(1)/(30*beta(2, 4)), Eq(y, 0)),
                (S(1)/(60*beta(2, 4)), Eq(y, 1)), (0, True))

    k, t, z = symbols('k t z', positive=True, real=True)
    G = Gamma('G', k, t)
    X = PoissonDistribution(G)
    C = CompoundDistribution(X)
    assert C.is_Discrete
    assert C.set == S.Naturals0
    assert C.pdf(z, evaluate=True).simplify() == t**z*(t + 1)**(-k - z)*gamma(k \
                    + z)/(gamma(k)*gamma(z + 1))


def test_compound_pspace():
    X = Normal('X', 2, 4)
    Y = Normal('Y', 3, 6)
    assert not isinstance(Y.pspace, CompoundPSpace)
    N = NormalDistribution(1, 2)
    D = PoissonDistribution(3)
    B = BernoulliDistribution(0.2, 1, 0)
    pspace1 = CompoundPSpace('N', N)
    pspace2 = CompoundPSpace('D', D)
    pspace3 = CompoundPSpace('B', B)
    assert not isinstance(pspace1, CompoundPSpace)
    assert not isinstance(pspace2, CompoundPSpace)
    assert not isinstance(pspace3, CompoundPSpace)
    M = MultivariateNormalDistribution([1, 2], [[2, 1], [1, 2]])
    raises(ValueError, lambda: CompoundPSpace('M', M))
    Y = Normal('Y', X, 6)
    assert isinstance(Y.pspace, CompoundPSpace)
    assert Y.pspace.distribution == CompoundDistribution(NormalDistribution(X, 6))
    assert Y.pspace.domain.set == Interval(-oo, oo)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\conv_indexed_to_array.py ===
from sympy.tensor.array.expressions import from_indexed_to_array
from sympy.tensor.array.expressions.conv_array_to_indexed import _conv_to_from_decorator

convert_indexed_to_array = _conv_to_from_decorator(from_indexed_to_array.convert_indexed_to_array)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\conv_matrix_to_array.py ===
from sympy.tensor.array.expressions import from_matrix_to_array
from sympy.tensor.array.expressions.conv_array_to_indexed import _conv_to_from_decorator

convert_matrix_to_array = _conv_to_from_decorator(from_matrix_to_array.convert_matrix_to_array)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\to_thread.py ===
from ._threads import current_default_thread_limiter, to_thread_run_sync as run_sync

# need to use __all__ for pyright --verifytypes to see re-exports when renaming them
__all__ = ["current_default_thread_limiter", "run_sync"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\type_tests\open_memory_channel.py ===
# https://github.com/python-trio/trio/issues/2873
import trio

s, r = trio.open_memory_channel[int](0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\__init__.py ===
from .serving import run_simple as run_simple
from .test import Client as Client
from .wrappers import Request as Request
from .wrappers import Response as Response

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_multi_thread.py ===
"""
Tests multithreading behaviour for reading and
parsing files for each parser defined in parsers.py
"""
from contextlib import ExitStack
from io import BytesIO
from multiprocessing.pool import ThreadPool

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.util.version import Version

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")

# We'll probably always skip these for pyarrow
# Maybe we'll add our own tests for pyarrow too
pytestmark = [
    pytest.mark.single_cpu,
    pytest.mark.slow,
]


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_multi_thread_string_io_read_csv(all_parsers, request):
    # see gh-11786
    parser = all_parsers
    if parser.engine == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        if Version(pa.__version__) < Version("16.0"):
            request.applymarker(
                pytest.mark.xfail(reason="# ValueError: Found non-unique column index")
            )
    max_row_range = 100
    num_files = 10

    bytes_to_df = (
        "\n".join([f"{i:d},{i:d},{i:d}" for i in range(max_row_range)]).encode()
        for _ in range(num_files)
    )

    # Read all files in many threads.
    with ExitStack() as stack:
        files = [stack.enter_context(BytesIO(b)) for b in bytes_to_df]

        pool = stack.enter_context(ThreadPool(8))

        results = pool.map(parser.read_csv, files)
        first_result = results[0]

        for result in results:
            tm.assert_frame_equal(first_result, result)


def _generate_multi_thread_dataframe(parser, path, num_rows, num_tasks):
    """
    Generate a DataFrame via multi-thread.

    Parameters
    ----------
    parser : BaseParser
        The parser object to use for reading the data.
    path : str
        The location of the CSV file to read.
    num_rows : int
        The number of rows to read per task.
    num_tasks : int
        The number of tasks to use for reading this DataFrame.

    Returns
    -------
    df : DataFrame
    """

    def reader(arg):
        """
        Create a reader for part of the CSV.

        Parameters
        ----------
        arg : tuple
            A tuple of the following:

            * start : int
                The starting row to start for parsing CSV
            * nrows : int
                The number of rows to read.

        Returns
        -------
        df : DataFrame
        """
        start, nrows = arg

        if not start:
            return parser.read_csv(
                path, index_col=0, header=0, nrows=nrows, parse_dates=["date"]
            )

        return parser.read_csv(
            path,
            index_col=0,
            header=None,
            skiprows=int(start) + 1,
            nrows=nrows,
            parse_dates=[9],
        )

    tasks = [
        (num_rows * i // num_tasks, num_rows // num_tasks) for i in range(num_tasks)
    ]

    with ThreadPool(processes=num_tasks) as pool:
        results = pool.map(reader, tasks)

    header = results[0].columns

    for r in results[1:]:
        r.columns = header

    final_dataframe = pd.concat(results)
    return final_dataframe


@xfail_pyarrow  # ValueError: The 'nrows' option is not supported
def test_multi_thread_path_multipart_read_csv(all_parsers):
    # see gh-11786
    num_tasks = 4
    num_rows = 48

    parser = all_parsers
    file_name = "__thread_pool_reader__.csv"
    df = DataFrame(
        {
            "a": np.random.default_rng(2).random(num_rows),
            "b": np.random.default_rng(2).random(num_rows),
            "c": np.random.default_rng(2).random(num_rows),
            "d": np.random.default_rng(2).random(num_rows),
            "e": np.random.default_rng(2).random(num_rows),
            "foo": ["foo"] * num_rows,
            "bar": ["bar"] * num_rows,
            "baz": ["baz"] * num_rows,
            "date": pd.date_range("20000101 09:00:00", periods=num_rows, freq="s"),
            "int": np.arange(num_rows, dtype="int64"),
        }
    )

    with tm.ensure_clean(file_name) as path:
        df.to_csv(path)

        final_dataframe = _generate_multi_thread_dataframe(
            parser, path, num_rows, num_tasks
        )
        tm.assert_frame_equal(df, final_dataframe)