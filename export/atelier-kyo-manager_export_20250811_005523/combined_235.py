
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_dotproduct.py ===
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.matrices import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.testing.pytest import raises


A = Matrix(3, 1, [1, 2, 3])
B = Matrix(3, 1, [1, 3, 5])
C = Matrix(4, 1, [1, 2, 4, 5])
D = Matrix(2, 2, [1, 2, 3, 4])

def test_docproduct():
    assert DotProduct(A, B).doit() == 22
    assert DotProduct(A.T, B).doit() == 22
    assert DotProduct(A, B.T).doit() == 22
    assert DotProduct(A.T, B.T).doit() == 22

    raises(TypeError, lambda: DotProduct(1, A))
    raises(TypeError, lambda: DotProduct(A, 1))
    raises(TypeError, lambda: DotProduct(A, D))
    raises(TypeError, lambda: DotProduct(D, A))

    raises(TypeError, lambda: DotProduct(B, C).doit())

def test_dotproduct_symbolic():
    A = MatrixSymbol('A', 3, 1)
    B = MatrixSymbol('B', 3, 1)

    dot = DotProduct(A, B)
    assert dot.is_scalar == True
    assert unchanged(Mul, 2, dot)
    # XXX Fix forced evaluation for arithmetics with matrix expressions
    assert dot * A == (A[0, 0]*B[0, 0] + A[1, 0]*B[1, 0] + A[2, 0]*B[2, 0])*A

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\methods\test_repeat.py ===
import numpy as np

from pandas import (
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestRepeat:
    def test_repeat(self):
        index = timedelta_range("1 days", periods=2, freq="D")
        exp = TimedeltaIndex(["1 days", "1 days", "2 days", "2 days"])
        for res in [index.repeat(2), np.repeat(index, 2)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

        index = TimedeltaIndex(["1 days", "NaT", "3 days"])
        exp = TimedeltaIndex(
            [
                "1 days",
                "1 days",
                "1 days",
                "NaT",
                "NaT",
                "NaT",
                "3 days",
                "3 days",
                "3 days",
            ]
        )
        for res in [index.repeat(3), np.repeat(index, 3)]:
            tm.assert_index_equal(res, exp)
            assert res.freq is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\sas\test_sas.py ===
from io import StringIO

import pytest

from pandas import read_sas
import pandas._testing as tm


class TestSas:
    def test_sas_buffer_format(self):
        # see gh-14947
        b = StringIO("")

        msg = (
            "If this is a buffer object rather than a string "
            "name, you must specify a format string"
        )
        with pytest.raises(ValueError, match=msg):
            read_sas(b)

    def test_sas_read_no_format_or_extension(self):
        # see gh-24548
        msg = "unable to infer format of SAS file.+"
        with tm.ensure_clean("test_file_no_extension") as path:
            with pytest.raises(ValueError, match=msg):
                read_sas(path)


def test_sas_archive(datapath):
    fname_uncompressed = datapath("io", "sas", "data", "airline.sas7bdat")
    df_uncompressed = read_sas(fname_uncompressed)
    fname_compressed = datapath("io", "sas", "data", "airline.sas7bdat.gz")
    df_compressed = read_sas(fname_compressed, format="sas7bdat")
    tm.assert_frame_equal(df_uncompressed, df_compressed)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_count.py ===
import numpy as np

import pandas as pd
from pandas import (
    Categorical,
    Series,
)
import pandas._testing as tm


class TestSeriesCount:
    def test_count(self, datetime_series):
        assert datetime_series.count() == len(datetime_series)

        datetime_series[::2] = np.nan

        assert datetime_series.count() == np.isfinite(datetime_series).sum()

    def test_count_inf_as_na(self):
        # GH#29478
        ser = Series([pd.Timestamp("1990/1/1")])
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context("use_inf_as_na", True):
                assert ser.count() == 1

    def test_count_categorical(self):
        ser = Series(
            Categorical(
                [np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True
            )
        )
        result = ser.count()
        assert result == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_adjoint.py ===
from sympy.core import symbols, S
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices.expressions import MatrixSymbol, Adjoint, trace, Transpose
from sympy.matrices import eye, Matrix

n, m, l, k, p = symbols('n m l k p', integer=True)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)


def test_adjoint():
    Sq = MatrixSymbol('Sq', n, n)

    assert Adjoint(A).shape == (m, n)
    assert Adjoint(A*B).shape == (l, n)
    assert adjoint(Adjoint(A)) == A
    assert isinstance(Adjoint(Adjoint(A)), Adjoint)

    assert conjugate(Adjoint(A)) == Transpose(A)
    assert transpose(Adjoint(A)) == Adjoint(Transpose(A))

    assert Adjoint(eye(3)).doit() == eye(3)

    assert Adjoint(S(5)).doit() == S(5)

    assert Adjoint(Matrix([[1, 2], [3, 4]])).doit() == Matrix([[1, 3], [2, 4]])

    assert adjoint(trace(Sq)) == conjugate(trace(Sq))
    assert trace(adjoint(Sq)) == conjugate(trace(Sq))

    assert Adjoint(Sq)[0, 1] == conjugate(Sq[1, 0])

    assert Adjoint(A*B).doit() == Adjoint(B) * Adjoint(A)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_modular.py ===
from sympy.ntheory.modular import crt, crt1, crt2, solve_congruence
from sympy.testing.pytest import raises


def test_crt():
    def mcrt(m, v, r, symmetric=False):
        assert crt(m, v, symmetric)[0] == r
        mm, e, s = crt1(m)
        assert crt2(m, v, mm, e, s, symmetric) == (r, mm)

    mcrt([2, 3, 5], [0, 0, 0], 0)
    mcrt([2, 3, 5], [1, 1, 1], 1)

    mcrt([2, 3, 5], [-1, -1, -1], -1, True)
    mcrt([2, 3, 5], [-1, -1, -1], 2*3*5 - 1, False)

    assert crt([656, 350], [811, 133], symmetric=True) == (-56917, 114800)


def test_modular():
    assert solve_congruence(*list(zip([3, 4, 2], [12, 35, 17]))) == (1719, 7140)
    assert solve_congruence(*list(zip([3, 4, 2], [12, 6, 17]))) is None
    assert solve_congruence(*list(zip([3, 4, 2], [13, 7, 17]))) == (172, 1547)
    assert solve_congruence(*list(zip([-10, -3, -15], [13, 7, 17]))) == (172, 1547)
    assert solve_congruence(*list(zip([-10, -3, 1, -15], [13, 7, 7, 17]))) is None
    assert solve_congruence(
        *list(zip([-10, -5, 2, -15], [13, 7, 7, 17]))) == (835, 1547)
    assert solve_congruence(
        *list(zip([-10, -5, 2, -15], [13, 7, 14, 17]))) == (2382, 3094)
    assert solve_congruence(
        *list(zip([-10, 2, 2, -15], [13, 7, 14, 17]))) == (2382, 3094)
    assert solve_congruence(*list(zip((1, 1, 2), (3, 2, 4)))) is None
    raises(
        ValueError, lambda: solve_congruence(*list(zip([3, 4, 2], [12.1, 35, 17]))))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\fail_cpp_exception.py ===
# -*- coding: utf-8 -*-
"""
Helper for testing a C++ exception throw aborts the process.

Takes one argument, the name of the function in :mod:`_test_extension_cpp` to call.
"""
import sys
import greenlet
from greenlet.tests import _test_extension_cpp
print('fail_cpp_exception is running')

def run_unhandled_exception_in_greenlet_aborts():
    def _():
        _test_extension_cpp.test_exception_switch_and_do_in_g2(
            _test_extension_cpp.test_exception_throw_nonstd
        )
    g1 = greenlet.greenlet(_)
    g1.switch()


func_name = sys.argv[1]
try:
    func = getattr(_test_extension_cpp, func_name)
except AttributeError:
    if func_name == run_unhandled_exception_in_greenlet_aborts.__name__:
        func = run_unhandled_exception_in_greenlet_aborts
    elif func_name == 'run_as_greenlet_target':
        g = greenlet.greenlet(_test_extension_cpp.test_exception_throw_std)
        func = g.switch
    else:
        raise
print('raising', func, flush=True)
func()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\array_with_attr\test_array_with_attr.py ===
import numpy as np

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.array_with_attr import FloatAttrArray


def test_concat_with_all_na():
    # https://github.com/pandas-dev/pandas/pull/47762
    # ensure that attribute of the column array is preserved (when it gets
    # preserved in reindexing the array) during merge/concat
    arr = FloatAttrArray(np.array([np.nan, np.nan], dtype="float64"), attr="test")

    df1 = pd.DataFrame({"col": arr, "key": [0, 1]})
    df2 = pd.DataFrame({"key": [0, 1], "col2": [1, 2]})
    result = pd.merge(df1, df2, on="key")
    expected = pd.DataFrame({"col": arr, "key": [0, 1], "col2": [1, 2]})
    tm.assert_frame_equal(result, expected)
    assert result["col"].array.attr == "test"

    df1 = pd.DataFrame({"col": arr, "key": [0, 1]})
    df2 = pd.DataFrame({"key": [0, 2], "col2": [1, 2]})
    result = pd.merge(df1, df2, on="key")
    expected = pd.DataFrame({"col": arr.take([0]), "key": [0], "col2": [1]})
    tm.assert_frame_equal(result, expected)
    assert result["col"].array.attr == "test"

    result = pd.concat([df1.set_index("key"), df2.set_index("key")], axis=1)
    expected = pd.DataFrame(
        {"col": arr.take([0, 1, -1]), "col2": [1, np.nan, 2], "key": [0, 1, 2]}
    ).set_index("key")
    tm.assert_frame_equal(result, expected)
    assert result["col"].array.attr == "test"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\list\test_list.py ===
import pytest

import pandas as pd
from pandas.tests.extension.list.array import (
    ListArray,
    ListDtype,
    make_data,
)


@pytest.fixture
def dtype():
    return ListDtype()


@pytest.fixture
def data():
    """Length-100 ListArray for semantics test."""
    data = make_data()

    while len(data[0]) == len(data[1]):
        data = make_data()

    return ListArray(data)


def test_to_csv(data):
    # https://github.com/pandas-dev/pandas/issues/28840
    # array with list-likes fail when doing astype(str) on the numpy array
    # which was done in get_values_for_csv
    df = pd.DataFrame({"a": data})
    res = df.to_csv()
    assert str(data[0]) in res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_easter.py ===
"""
Tests for the following offsets:
- Easter
"""
from __future__ import annotations

from datetime import datetime

import pytest

from pandas.tests.tseries.offsets.common import assert_offset_equal

from pandas.tseries.offsets import Easter


class TestEaster:
    @pytest.mark.parametrize(
        "offset,date,expected",
        [
            (Easter(), datetime(2010, 1, 1), datetime(2010, 4, 4)),
            (Easter(), datetime(2010, 4, 5), datetime(2011, 4, 24)),
            (Easter(2), datetime(2010, 1, 1), datetime(2011, 4, 24)),
            (Easter(), datetime(2010, 4, 4), datetime(2011, 4, 24)),
            (Easter(2), datetime(2010, 4, 4), datetime(2012, 4, 8)),
            (-Easter(), datetime(2011, 1, 1), datetime(2010, 4, 4)),
            (-Easter(), datetime(2010, 4, 5), datetime(2010, 4, 4)),
            (-Easter(2), datetime(2011, 1, 1), datetime(2009, 4, 12)),
            (-Easter(), datetime(2010, 4, 4), datetime(2009, 4, 12)),
            (-Easter(2), datetime(2010, 4, 4), datetime(2008, 3, 23)),
        ],
    )
    def test_offset(self, offset, date, expected):
        assert_offset_equal(offset, date, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_attr_equal.py ===
from types import SimpleNamespace

import pytest

from pandas.core.dtypes.common import is_float

import pandas._testing as tm


def test_assert_attr_equal(nulls_fixture):
    obj = SimpleNamespace()
    obj.na_value = nulls_fixture
    tm.assert_attr_equal("na_value", obj, obj)


def test_assert_attr_equal_different_nulls(nulls_fixture, nulls_fixture2):
    obj = SimpleNamespace()
    obj.na_value = nulls_fixture

    obj2 = SimpleNamespace()
    obj2.na_value = nulls_fixture2

    if nulls_fixture is nulls_fixture2:
        tm.assert_attr_equal("na_value", obj, obj2)
    elif is_float(nulls_fixture) and is_float(nulls_fixture2):
        # we consider float("nan") and np.float64("nan") to be equivalent
        tm.assert_attr_equal("na_value", obj, obj2)
    elif type(nulls_fixture) is type(nulls_fixture2):
        # e.g. Decimal("NaN")
        tm.assert_attr_equal("na_value", obj, obj2)
    else:
        with pytest.raises(AssertionError, match='"na_value" are different'):
            tm.assert_attr_equal("na_value", obj, obj2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\diffgeom\tests\test_class_structure.py ===
from sympy.diffgeom import Manifold, Patch, CoordSystem, Point
from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.testing.pytest import warns_deprecated_sympy

m = Manifold('m', 2)
p = Patch('p', m)
a, b = symbols('a b')
cs = CoordSystem('cs', p, [a, b])
x, y = symbols('x y')
f = Function('f')
s1, s2 = cs.coord_functions()
v1, v2 = cs.base_vectors()
f1, f2 = cs.base_oneforms()

def test_point():
    point = Point(cs, [x, y])
    assert point != Point(cs, [2, y])
    #TODO assert point.subs(x, 2) == Point(cs, [2, y])
    #TODO assert point.free_symbols == set([x, y])

def test_subs():
    assert s1.subs(s1, s2) == s2
    assert v1.subs(v1, v2) == v2
    assert f1.subs(f1, f2) == f2
    assert (x*f(s1) + y).subs(s1, s2) == x*f(s2) + y
    assert (f(s1)*v1).subs(v1, v2) == f(s1)*v2
    assert (y*f(s1)*f1).subs(f1, f2) == y*f(s1)*f2

def test_deprecated():
    with warns_deprecated_sympy():
        cs_wname = CoordSystem('cs', p, ['a', 'b'])
        assert cs_wname == cs_wname.func(*cs_wname.args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_mathml.py ===
import os
from textwrap import dedent
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.mathml import apply_xsl



lxml = import_module('lxml')

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_xxe.py"))


def test_xxe():
    assert os.path.isfile(path)
    if not lxml:
        skip("lxml not installed.")

    mml = dedent(
        rf"""
        <!--?xml version="1.0" ?-->
        <!DOCTYPE replace [<!ENTITY ent SYSTEM "file://{path}"> ]>
        <userInfo>
        <firstName>John</firstName>
        <lastName>&ent;</lastName>
        </userInfo>
        """
    )
    xsl = 'mathml/data/simple_mmlctop.xsl'

    res = apply_xsl(mml, xsl)
    assert res == \
        '<?xml version="1.0"?>\n<userInfo>\n<firstName>John</firstName>\n<lastName/>\n</userInfo>\n'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_visualization.py ===
"""
Limited tests of the visualization module. Right now it just makes
sure that passing custom Axes works.

"""

from mpmath import mp, fp

def test_axes():
    try:
        import matplotlib
        version = matplotlib.__version__.split("-")[0]
        version = version.split(".")[:2]
        if [int(_) for _ in version] < [0,99]:
            raise ImportError
        import pylab
    except ImportError:
        print("\nSkipping test (pylab not available or too old version)\n")
        return
    fig = pylab.figure()
    axes = fig.add_subplot(111)
    for ctx in [mp, fp]:
        ctx.plot(lambda x: x**2, [0, 3], axes=axes)
        assert axes.get_xlabel() == 'x'
        assert axes.get_ylabel() == 'f(x)'

    fig = pylab.figure()
    axes = fig.add_subplot(111)
    for ctx in [mp, fp]:
        ctx.cplot(lambda z: z, [-2, 2], [-10, 10], axes=axes)
    assert axes.get_xlabel() == 'Re(z)'
    assert axes.get_ylabel() == 'Im(z)'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\test_isfile.py ===
import os
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_

ROOT = Path(np.__file__).parents[0]
FILES = [
    ROOT / "py.typed",
    ROOT / "__init__.pyi",
    ROOT / "ctypeslib" / "__init__.pyi",
    ROOT / "_core" / "__init__.pyi",
    ROOT / "f2py" / "__init__.pyi",
    ROOT / "fft" / "__init__.pyi",
    ROOT / "lib" / "__init__.pyi",
    ROOT / "linalg" / "__init__.pyi",
    ROOT / "ma" / "__init__.pyi",
    ROOT / "matrixlib" / "__init__.pyi",
    ROOT / "polynomial" / "__init__.pyi",
    ROOT / "random" / "__init__.pyi",
    ROOT / "testing" / "__init__.pyi",
]
if sys.version_info < (3, 12):
    FILES += [ROOT / "distutils" / "__init__.pyi"]


class TestIsFile:
    def test_isfile(self):
        """Test if all ``.pyi`` files are properly installed."""
        for file in FILES:
            assert_(os.path.isfile(file))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_pyinstaller\tests\pyinstaller-smoke.py ===
"""A crude *bit of everything* smoke test to verify PyInstaller compatibility.

PyInstaller typically goes wrong by forgetting to package modules, extension
modules or shared libraries. This script should aim to touch as many of those
as possible in an attempt to trip a ModuleNotFoundError or a DLL load failure
due to an uncollected resource. Missing resources are unlikely to lead to
arithmetic errors so there's generally no need to verify any calculation's
output - merely that it made it to the end OK. This script should not
explicitly import any of numpy's submodules as that gives PyInstaller undue
hints that those submodules exist and should be collected (accessing implicitly
loaded submodules is OK).

"""
import numpy as np

a = np.arange(1., 10.).reshape((3, 3)) % 5
np.linalg.det(a)
a @ a
a @ a.T
np.linalg.inv(a)
np.sin(np.exp(a))
np.linalg.svd(a)
np.linalg.eigh(a)

np.unique(np.random.randint(0, 10, 100))
np.sort(np.random.uniform(0, 10, 100))

np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
np.ma.masked_array(np.arange(10), np.random.rand(10) < .5).sum()
np.polynomial.Legendre([7, 8, 9]).roots()

print("I made it!")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\computation\test_compat.py ===
import pytest

from pandas.compat._optional import VERSIONS

import pandas as pd
from pandas.core.computation import expr
from pandas.core.computation.engines import ENGINES
from pandas.util.version import Version


def test_compat():
    # test we have compat with our version of numexpr

    from pandas.core.computation.check import NUMEXPR_INSTALLED

    ne = pytest.importorskip("numexpr")

    ver = ne.__version__
    if Version(ver) < Version(VERSIONS["numexpr"]):
        assert not NUMEXPR_INSTALLED
    else:
        assert NUMEXPR_INSTALLED


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parser", expr.PARSERS)
def test_invalid_numexpr_version(engine, parser):
    if engine == "numexpr":
        pytest.importorskip("numexpr")
    a, b = 1, 2  # noqa: F841
    res = pd.eval("a + b", engine=engine, parser=parser)
    assert res == 3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_shares_memory.py ===
import numpy as np

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


def test_shares_memory_interval():
    obj = pd.interval_range(1, 5)

    assert tm.shares_memory(obj, obj)
    assert tm.shares_memory(obj, obj._data)
    assert tm.shares_memory(obj, obj[::-1])
    assert tm.shares_memory(obj, obj[:2])

    assert not tm.shares_memory(obj, obj._data.copy())


@td.skip_if_no("pyarrow")
def test_shares_memory_string():
    # GH#55823
    import pyarrow as pa

    obj = pd.array(["a", "b"], dtype=pd.StringDtype("pyarrow", na_value=pd.NA))
    assert tm.shares_memory(obj, obj)

    obj = pd.array(["a", "b"], dtype=pd.StringDtype("pyarrow", na_value=np.nan))
    assert tm.shares_memory(obj, obj)

    obj = pd.array(["a", "b"], dtype=pd.ArrowDtype(pa.string()))
    assert tm.shares_memory(obj, obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\tests\test_tools.py ===
from sympy.strategies.tools import subs, typed
from sympy.strategies.rl import rm_id
from sympy.core.basic import Basic
from sympy.core.singleton import S


def test_subs():
    from sympy.core.symbol import symbols
    a, b, c, d, e, f = symbols('a,b,c,d,e,f')
    mapping = {a: d, d: a, Basic(e): Basic(f)}
    expr = Basic(a, Basic(b, c), Basic(d, Basic(e)))
    result = Basic(d, Basic(b, c), Basic(a, Basic(f)))
    assert subs(mapping)(expr) == result


def test_subs_empty():
    assert subs({})(Basic(S(1), S(2))) == Basic(S(1), S(2))


def test_typed():
    class A(Basic):
        pass

    class B(Basic):
        pass

    rmzeros = rm_id(lambda x: x == S(0))
    rmones = rm_id(lambda x: x == S(1))
    remove_something = typed({A: rmzeros, B: rmones})

    assert remove_something(A(S(0), S(1))) == A(S(1))
    assert remove_something(B(S(0), S(1))) == B(S(0))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\compiler\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\pyext\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\testdata\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\util\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\platform\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\test_regression.py ===
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises


class TestRegression:
    def test_kron_matrix(self):
        # Ticket #71
        x = np.matrix('[1 0; 1 0]')
        assert_equal(type(np.kron(x, x)), type(x))

    def test_matrix_properties(self):
        # Ticket #125
        a = np.matrix([1.0], dtype=float)
        assert_(type(a.real) is np.matrix)
        assert_(type(a.imag) is np.matrix)
        c, d = np.matrix([0.0]).nonzero()
        assert_(type(c) is np.ndarray)
        assert_(type(d) is np.ndarray)

    def test_matrix_multiply_by_1d_vector(self):
        # Ticket #473
        def mul():
            np.asmatrix(np.eye(2)) * np.ones(2)

        assert_raises(ValueError, mul)

    def test_matrix_std_argmax(self):
        # Ticket #83
        x = np.asmatrix(np.random.uniform(0, 1, (3, 3)))
        assert_equal(x.std().shape, ())
        assert_equal(x.argmax().shape, ())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\testing\_private\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_pyinstaller\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\CalTableFlatBuffers\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\mobile_helpers\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\qdq_helpers\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\interchange\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\methods\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\reshape\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\sparse\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\tools\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\util\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\_numba\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_resolution.py ===
from dateutil.tz import tzlocal
import pytest

from pandas.compat import IS64

from pandas import date_range


@pytest.mark.parametrize(
    "freq,expected",
    [
        ("YE", "day"),
        ("QE", "day"),
        ("ME", "day"),
        ("D", "day"),
        ("h", "hour"),
        ("min", "minute"),
        ("s", "second"),
        ("ms", "millisecond"),
        ("us", "microsecond"),
    ],
)
def test_dti_resolution(request, tz_naive_fixture, freq, expected):
    tz = tz_naive_fixture
    if freq == "YE" and not IS64 and isinstance(tz, tzlocal):
        request.applymarker(
            pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
        )

    idx = date_range(start="2013-04-01", periods=30, freq=freq, tz=tz)
    assert idx.resolution == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_timestamp_method.py ===
# NB: This is for the Timestamp.timestamp *method* specifically, not
# the Timestamp class in general.

from pytz import utc

from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td

import pandas._testing as tm


class TestTimestampMethod:
    @td.skip_if_windows
    def test_timestamp(self, fixed_now_ts):
        # GH#17329
        # tz-naive --> treat it as if it were UTC for purposes of timestamp()
        ts = fixed_now_ts
        uts = ts.replace(tzinfo=utc)
        assert ts.timestamp() == uts.timestamp()

        tsc = Timestamp("2014-10-11 11:00:01.12345678", tz="US/Central")
        utsc = tsc.tz_convert("UTC")

        # utsc is a different representation of the same time
        assert tsc.timestamp() == utsc.timestamp()

        # datetime.timestamp() converts in the local timezone
        with tm.set_timezone("UTC"):
            # should agree with datetime.timestamp method
            dt = ts.to_pydatetime()
            assert dt.timestamp() == ts.timestamp()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_libs\window\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\build\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\legacy\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\resolvelib\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\contrib\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\contrib\_securetransport\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\packages\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\packages\backports\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\lineeditor\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sniffio\_tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\algorithms\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\hep\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\backends\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\handlers\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_rewrite.py ===
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, cot, sin)
from sympy.testing.pytest import _both_exp_pow

x, y, z, n = symbols('x,y,z,n')


@_both_exp_pow
def test_has():
    assert cot(x).has(x)
    assert cot(x).has(cot)
    assert not cot(x).has(sin)
    assert sin(x).has(x)
    assert sin(x).has(sin)
    assert not sin(x).has(cot)
    assert exp(x).has(exp)


@_both_exp_pow
def test_sin_exp_rewrite():
    assert sin(x).rewrite(sin, exp) == -I/2*(exp(I*x) - exp(-I*x))
    assert sin(x).rewrite(sin, exp).rewrite(exp, sin) == sin(x)
    assert cos(x).rewrite(cos, exp).rewrite(exp, cos) == cos(x)
    assert (sin(5*y) - sin(
        2*x)).rewrite(sin, exp).rewrite(exp, sin) == sin(5*y) - sin(2*x)
    assert sin(x + y).rewrite(sin, exp).rewrite(exp, sin) == sin(x + y)
    assert cos(x + y).rewrite(cos, exp).rewrite(exp, cos) == cos(x + y)
    # This next test currently passes... not clear whether it should or not?
    assert cos(x).rewrite(cos, exp).rewrite(exp, sin) == cos(x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\benchmarks\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\sampling\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\test_tensor_element.py ===
from sympy.tensor.tensor import (Tensor, TensorIndexType, TensorSymmetry,
        tensor_indices, TensorHead, TensorElement)
from sympy.tensor import Array
from sympy.core.symbol import Symbol


def test_tensor_element():
    L = TensorIndexType("L")
    i, j, k, l, m, n = tensor_indices("i j k l m n", L)

    A = TensorHead("A", [L, L], TensorSymmetry.no_symmetry(2))

    a = A(i, j)

    assert isinstance(TensorElement(a, {}), Tensor)
    assert isinstance(TensorElement(a, {k: 1}), Tensor)

    te1 = TensorElement(a, {Symbol("i"): 1})
    assert te1.free == [(j, 0)]
    assert te1.get_free_indices() == [j]
    assert te1.dum == []

    te2 = TensorElement(a, {i: 1})
    assert te2.free == [(j, 0)]
    assert te2.get_free_indices() == [j]
    assert te2.dum == []

    assert te1 == te2

    array = Array([[1, 2], [3, 4]])
    assert te1.replace_with_arrays({A(i, j): array}, [j]) == array[1, :]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\mathml\data\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\tools\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tools\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Africa\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\America\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\America\Argentina\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\America\Indiana\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\America\Kentucky\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\America\North_Dakota\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Antarctica\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Arctic\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Asia\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Atlantic\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Australia\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Brazil\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Canada\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Chile\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Etc\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Europe\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Indian\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Mexico\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\Pacific\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\tzdata\zoneinfo\US\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\contrib\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\drivers\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\middleware\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\sansio\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\csrf\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\extratest_zeta.py ===
from mpmath import zetazero
from timeit import default_timer as clock

def test_zetazero():
    cases = [\
    (399999999, 156762524.6750591511),
    (241389216, 97490234.2276711795),
    (526196239, 202950727.691229534),
    (542964976, 209039046.578535272),
    (1048449112, 388858885.231056486),
    (1048449113, 388858885.384337406),
    (1048449114, 388858886.002285122),
    (1048449115, 388858886.00239369),
    (1048449116, 388858886.690745053)
    ]
    for n, v in cases:
        print(n, v)
        t1 = clock()
        ok = zetazero(n).ae(complex(0.5,v))
        t2 = clock()
        print("ok =", ok, ("(time = %s)" % round(t2-t1,3)))
    print("Now computing two huge zeros (this may take hours)")
    print("Computing zetazero(8637740722917)")
    ok = zetazero(8637740722917).ae(complex(0.5,2124447368584.39296466152))
    print("ok =", ok)
    ok = zetazero(8637740722918).ae(complex(0.5,2124447368584.39298170604))
    print("ok =", ok)

if __name__ == "__main__":
    test_zetazero()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_machar.py ===
"""
Test machar. Given recent changes to hardcode type data, we might want to get
rid of both MachAr and this test at some point.

"""
import numpy._core.numerictypes as ntypes
from numpy import array, errstate
from numpy._core._machar import MachAr


class TestMachAr:
    def _run_machar_highprec(self):
        # Instantiate MachAr instance with high enough precision to cause
        # underflow
        try:
            hiprec = ntypes.float96
            MachAr(lambda v: array(v, hiprec))
        except AttributeError:
            # Fixme, this needs to raise a 'skip' exception.
            "Skipping test: no ntypes.float96 available on this platform."

    def test_underlow(self):
        # Regression test for #759:
        # instantiating MachAr for dtype = np.float96 raises spurious warning.
        with errstate(all='raise'):
            try:
                self._run_machar_highprec()
            except FloatingPointError as e:
                msg = f"Caught {e} exception, should not have been raised."
                raise AssertionError(msg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\util.py ===
from pandas import (
    Categorical,
    Index,
    Series,
)
from pandas.core.arrays import BaseMaskedArray


def get_array(obj, col=None):
    """
    Helper method to get array for a DataFrame column or a Series.

    Equivalent of df[col].values, but without going through normal getitem,
    which triggers tracking references / CoW (and we might be testing that
    this is done by some other operation).
    """
    if isinstance(obj, Index):
        arr = obj._values
    elif isinstance(obj, Series) and (col is None or obj.name == col):
        arr = obj._values
    else:
        assert col is not None
        icol = obj.columns.get_loc(col)
        assert isinstance(icol, int)
        arr = obj._get_column_array(icol)
    if isinstance(arr, BaseMaskedArray):
        return arr._data
    elif isinstance(arr, Categorical):
        return arr
    return getattr(arr, "_ndarray", arr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\index\test_periodindex.py ===
import pytest

from pandas import (
    Period,
    PeriodIndex,
    Series,
    period_range,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting a value on a view:FutureWarning"
)


@pytest.mark.parametrize(
    "cons",
    [
        lambda x: PeriodIndex(x),
        lambda x: PeriodIndex(PeriodIndex(x)),
    ],
)
def test_periodindex(using_copy_on_write, cons):
    dt = period_range("2019-12-31", periods=3, freq="D")
    ser = Series(dt)
    idx = cons(ser)
    expected = idx.copy(deep=True)
    ser.iloc[0] = Period("2020-12-31")
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\index\test_timedeltaindex.py ===
import pytest

from pandas import (
    Series,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting a value on a view:FutureWarning"
)


@pytest.mark.parametrize(
    "cons",
    [
        lambda x: TimedeltaIndex(x),
        lambda x: TimedeltaIndex(TimedeltaIndex(x)),
    ],
)
def test_timedeltaindex(using_copy_on_write, cons):
    dt = timedelta_range("1 day", periods=3)
    ser = Series(dt)
    idx = cons(ser)
    expected = idx.copy(deep=True)
    ser.iloc[0] = Timedelta("5 days")
    if using_copy_on_write:
        tm.assert_index_equal(idx, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_alter_axes.py ===
from datetime import datetime

import pytz

from pandas import DataFrame
import pandas._testing as tm


class TestDataFrameAlterAxes:
    # Tests for setting index/columns attributes directly (i.e. __setattr__)

    def test_set_axis_setattr_index(self):
        # GH 6785
        # set the index manually

        df = DataFrame([{"ts": datetime(2014, 4, 1, tzinfo=pytz.utc), "foo": 1}])
        expected = df.set_index("ts")
        df.index = df["ts"]
        df.pop("ts")
        tm.assert_frame_equal(df, expected)

    # Renaming

    def test_assign_columns(self, float_frame):
        float_frame["hi"] = "there"

        df = float_frame.copy()
        df.columns = ["foo", "bar", "baz", "quux", "foo2"]
        tm.assert_series_equal(float_frame["C"], df["baz"], check_names=False)
        tm.assert_series_equal(float_frame["hi"], df["foo2"], check_names=False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_asof.py ===
from datetime import timedelta

from pandas import (
    Index,
    Timestamp,
    date_range,
    isna,
)


class TestAsOf:
    def test_asof_partial(self):
        index = date_range("2010-01-01", periods=2, freq="ME")
        expected = Timestamp("2010-02-28")
        result = index.asof("2010-02")
        assert result == expected
        assert not isinstance(result, Index)

    def test_asof(self):
        index = date_range("2020-01-01", periods=10)

        dt = index[0]
        assert index.asof(dt) == dt
        assert isna(index.asof(dt - timedelta(1)))

        dt = index[-1]
        assert index.asof(dt + timedelta(1)) == dt

        dt = index[0].to_pydatetime()
        assert isinstance(index.asof(dt), Timestamp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_astype.py ===
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas._testing as tm


def test_astype(idx):
    expected = idx.copy()
    actual = idx.astype("O")
    tm.assert_copy(actual.levels, expected.levels)
    tm.assert_copy(actual.codes, expected.codes)
    assert actual.names == list(expected.names)

    with pytest.raises(TypeError, match="^Setting.*dtype.*object"):
        idx.astype(np.dtype(int))


@pytest.mark.parametrize("ordered", [True, False])
def test_astype_category(idx, ordered):
    # GH 18630
    msg = "> 1 ndim Categorical are not supported at this time"
    with pytest.raises(NotImplementedError, match=msg):
        idx.astype(CategoricalDtype(ordered=ordered))

    if ordered is False:
        # dtype='category' defaults to ordered=False, so only test once
        with pytest.raises(NotImplementedError, match=msg):
            idx.astype("category")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_autocorr.py ===
import numpy as np


class TestAutoCorr:
    def test_autocorr(self, datetime_series):
        # Just run the function
        corr1 = datetime_series.autocorr()

        # Now run it with the lag parameter
        corr2 = datetime_series.autocorr(lag=1)

        # corr() with lag needs Series of at least length 2
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            assert corr1 == corr2

        # Choose a random lag between 1 and length of Series - 2
        # and compare the result with the Series corr() function
        n = 1 + np.random.default_rng(2).integers(max(1, len(datetime_series) - 2))
        corr1 = datetime_series.corr(datetime_series.shift(n))
        corr2 = datetime_series.autocorr(lag=n)

        # corr() with lag needs Series of at least length 2
        if len(datetime_series) <= 2:
            assert np.isnan(corr1)
            assert np.isnan(corr2)
        else:
            assert corr1 == corr2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\fail_slp_switch.py ===
# -*- coding: utf-8 -*-
"""
A test helper for seeing what happens when slp_switch()
fails.
"""
# pragma: no cover

import greenlet


print('fail_slp_switch is running', flush=True)

runs = []
def func():
    runs.append(1)
    greenlet.getcurrent().parent.switch()
    runs.append(2)
    greenlet.getcurrent().parent.switch()
    runs.append(3)

g = greenlet._greenlet.UnswitchableGreenlet(func)
g.switch()
assert runs == [1]
g.switch()
assert runs == [1, 2]
g.force_slp_switch_error = True

# This should crash.
g.switch()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_routines.py ===
import pytest

from . import util


@pytest.mark.slow
class TestRenamedFunc(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "routines", "funcfortranname.f"),
        util.getpath("tests", "src", "routines", "funcfortranname.pyf"),
    ]
    module_name = "funcfortranname"

    def test_gh25799(self):
        assert dir(self.module)
        assert self.module.funcfortranname_default(200, 12) == 212


@pytest.mark.slow
class TestRenamedSubroutine(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "routines", "subrout.f"),
        util.getpath("tests", "src", "routines", "subrout.pyf"),
    ]
    module_name = "subrout"

    def test_renamed_subroutine(self):
        assert dir(self.module)
        assert self.module.subrout_default(200, 12) == 212

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_values.py ===
import numpy as np
import pytest

from pandas import (
    IntervalIndex,
    Series,
    period_range,
)
import pandas._testing as tm


class TestValues:
    @pytest.mark.parametrize(
        "data",
        [
            period_range("2000", periods=4),
            IntervalIndex.from_breaks([1, 2, 3, 4]),
        ],
    )
    def test_values_object_extension_dtypes(self, data):
        # https://github.com/pandas-dev/pandas/issues/23995
        result = Series(data).values
        expected = np.array(data.astype(object))
        tm.assert_numpy_array_equal(result, expected)

    def test_values(self, datetime_series):
        tm.assert_almost_equal(
            datetime_series.values, list(datetime_series), check_dtype=False
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\frequencies\test_frequencies.py ===
import pytest

from pandas._libs.tslibs import offsets

from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)


@pytest.mark.parametrize(
    "p1,p2,expected",
    [
        # Input validation.
        (offsets.MonthEnd(), None, False),
        (offsets.YearEnd(), None, False),
        (None, offsets.YearEnd(), False),
        (None, offsets.MonthEnd(), False),
        (None, None, False),
        (offsets.YearEnd(), offsets.MonthEnd(), True),
        (offsets.Hour(), offsets.Minute(), True),
        (offsets.Second(), offsets.Milli(), True),
        (offsets.Milli(), offsets.Micro(), True),
        (offsets.Micro(), offsets.Nano(), True),
    ],
)
def test_super_sub_symmetry(p1, p2, expected):
    assert is_superperiod(p1, p2) is expected
    assert is_subperiod(p2, p1) is expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_mathieu.py ===
from sympy.core.function import diff
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)

from sympy.abc import a, q, z


def test_mathieus():
    assert isinstance(mathieus(a, q, z), mathieus)
    assert mathieus(a, 0, z) == sin(sqrt(a)*z)
    assert conjugate(mathieus(a, q, z)) == mathieus(conjugate(a), conjugate(q), conjugate(z))
    assert diff(mathieus(a, q, z), z) == mathieusprime(a, q, z)

def test_mathieuc():
    assert isinstance(mathieuc(a, q, z), mathieuc)
    assert mathieuc(a, 0, z) == cos(sqrt(a)*z)
    assert diff(mathieuc(a, q, z), z) == mathieucprime(a, q, z)

def test_mathieusprime():
    assert isinstance(mathieusprime(a, q, z), mathieusprime)
    assert mathieusprime(a, 0, z) == sqrt(a)*cos(sqrt(a)*z)
    assert diff(mathieusprime(a, q, z), z) == (-a + 2*q*cos(2*z))*mathieus(a, q, z)

def test_mathieucprime():
    assert isinstance(mathieucprime(a, q, z), mathieucprime)
    assert mathieucprime(a, 0, z) == -sqrt(a)*sin(sqrt(a)*z)
    assert diff(mathieucprime(a, q, z), z) == (-a + 2*q*cos(2*z))*mathieuc(a, q, z)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_factorizations.py ===
from sympy.matrices.expressions.factorizations import lu, LofCholesky, qr, svd
from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.matexpr import MatrixSymbol

n = Symbol('n')
X = MatrixSymbol('X', n, n)

def test_LU():
    L, U = lu(X)
    assert L.shape == U.shape == X.shape
    assert ask(Q.lower_triangular(L))
    assert ask(Q.upper_triangular(U))

def test_Cholesky():
    LofCholesky(X)

def test_QR():
    Q_, R = qr(X)
    assert Q_.shape == R.shape == X.shape
    assert ask(Q.orthogonal(Q_))
    assert ask(Q.upper_triangular(R))

def test_svd():
    U, S, V = svd(X)
    assert U.shape == S.shape == V.shape == X.shape
    assert ask(Q.orthogonal(U))
    assert ask(Q.orthogonal(V))
    assert ask(Q.diagonal(S))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_piab.py ===
"""Tests for piab.py"""

from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.sets.sets import Interval
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum import L2, qapply, hbar, represent
from sympy.physics.quantum.piab import PIABHamiltonian, PIABKet, PIABBra, m, L

i, j, n, x = symbols('i j n x')


def test_H():
    assert PIABHamiltonian('H').hilbert_space == \
        L2(Interval(S.NegativeInfinity, S.Infinity))
    assert qapply(PIABHamiltonian('H')*PIABKet(n)) == \
        (n**2*pi**2*hbar**2)/(2*m*L**2)*PIABKet(n)


def test_states():
    assert PIABKet(n).dual_class() == PIABBra
    assert PIABKet(n).hilbert_space == \
        L2(Interval(S.NegativeInfinity, S.Infinity))
    assert represent(PIABKet(n)) == sqrt(2/L)*sin(n*pi*x/L)
    assert (PIABBra(i)*PIABKet(j)).doit() == KroneckerDelta(i, j)
    assert PIABBra(n).dual_class() == PIABKet

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_builder.py ===
import pytest
from unittest.mock import patch
from bs4.builder import DetectsXMLParsedAsHTML


class TestDetectsXMLParsedAsHTML:
    @pytest.mark.parametrize(
        "markup,looks_like_xml",
        [
            ("No xml declaration", False),
            ("<html>obviously HTML</html", False),
            ("<?xml ><html>Actually XHTML</html>", False),
            ("<?xml>            <    html>Tricky XHTML</html>", False),
            ("<?xml ><no-html-tag>", True),
        ],
    )
    def test_warn_if_markup_looks_like_xml(self, markup, looks_like_xml):
        # Test of our ability to guess at whether markup looks XML-ish
        # _and_ not HTML-ish.
        with patch("bs4.builder.DetectsXMLParsedAsHTML._warn") as mock:
            for data in markup, markup.encode("utf8"):
                result = DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(data)
                assert result == looks_like_xml
                if looks_like_xml:
                    assert mock.called
                else:
                    assert not mock.called
                mock.reset_mock()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\interval\test_astype.py ===
import pytest

from pandas import (
    Categorical,
    CategoricalDtype,
    Index,
    IntervalIndex,
)
import pandas._testing as tm


class TestAstype:
    @pytest.mark.parametrize("ordered", [True, False])
    def test_astype_categorical_retains_ordered(self, ordered):
        index = IntervalIndex.from_breaks(range(5))
        arr = index._data

        dtype = CategoricalDtype(None, ordered=ordered)

        expected = Categorical(list(arr), ordered=ordered)
        result = arr.astype(dtype)
        assert result.ordered is ordered
        tm.assert_categorical_equal(result, expected)

        # test IntervalIndex.astype while we're at it.
        result = index.astype(dtype)
        expected = Index(expected)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_infer_datetimelike.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timestamp,
)


@pytest.mark.parametrize(
    "data,exp_size",
    [
        # see gh-16362.
        ([[NaT, "a", "b", 0], [NaT, "b", "c", 1]], 8),
        ([[NaT, "a", 0], [NaT, "b", 1]], 6),
    ],
)
def test_maybe_infer_to_datetimelike_df_construct(data, exp_size):
    result = DataFrame(np.array(data))
    assert result.size == exp_size


def test_maybe_infer_to_datetimelike_ser_construct():
    # see gh-19671.
    result = Series(["M1701", Timestamp("20130101")])
    assert result.dtype.kind == "O"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_isocalendar.py ===
from pandas import (
    DataFrame,
    DatetimeIndex,
    date_range,
)
import pandas._testing as tm


def test_isocalendar_returns_correct_values_close_to_new_year_with_tz():
    # GH#6538: Check that DatetimeIndex and its TimeStamp elements
    # return the same weekofyear accessor close to new year w/ tz
    dates = ["2013/12/29", "2013/12/30", "2013/12/31"]
    dates = DatetimeIndex(dates, tz="Europe/Brussels")
    result = dates.isocalendar()
    expected_data_frame = DataFrame(
        [[2013, 52, 7], [2014, 1, 1], [2014, 1, 2]],
        columns=["year", "week", "day"],
        index=dates,
        dtype="UInt32",
    )
    tm.assert_frame_equal(result, expected_data_frame)


def test_dti_timestamp_isocalendar_fields():
    idx = date_range("2020-01-01", periods=10)
    expected = tuple(idx.isocalendar().iloc[-1].to_list())
    result = idx[-1].isocalendar()
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_to_frame.py ===
from pandas import (
    DataFrame,
    Index,
    date_range,
)
import pandas._testing as tm


class TestToFrame:
    def test_to_frame_datetime_tz(self):
        # GH#25809
        idx = date_range(start="2019-01-01", end="2019-01-30", freq="D", tz="UTC")
        result = idx.to_frame()
        expected = DataFrame(idx, index=idx)
        tm.assert_frame_equal(result, expected)

    def test_to_frame_respects_none_name(self):
        # GH#44212 if we explicitly pass name=None, then that should be respected,
        #  not changed to 0
        # GH-45448 this is first deprecated to only change in the future
        idx = date_range(start="2019-01-01", end="2019-01-30", freq="D", tz="UTC")
        result = idx.to_frame(name=None)
        exp_idx = Index([None], dtype=object)
        tm.assert_index_equal(exp_idx, result.columns)

        result = idx.rename("foo").to_frame(name=None)
        exp_idx = Index([None], dtype=object)
        tm.assert_index_equal(exp_idx, result.columns)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_freq_attr.py ===
import pytest

from pandas.compat import PY311

from pandas import (
    offsets,
    period_range,
)
import pandas._testing as tm


class TestFreq:
    def test_freq_setter_deprecated(self):
        # GH#20678
        idx = period_range("2018Q1", periods=4, freq="Q")

        # no warning for getter
        with tm.assert_produces_warning(None):
            idx.freq

        # warning for setter
        msg = (
            "property 'freq' of 'PeriodArray' object has no setter"
            if PY311
            else "can't set attribute"
        )
        with pytest.raises(AttributeError, match=msg):
            idx.freq = offsets.Day()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_searchsorted.py ===
import numpy as np
import pytest

from pandas import (
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm


class TestSearchSorted:
    def test_searchsorted_different_argument_classes(self, listlike_box):
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        result = idx.searchsorted(listlike_box(idx))
        expected = np.arange(len(idx), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = idx._data.searchsorted(listlike_box(idx))
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
    )
    def test_searchsorted_invalid_argument_dtype(self, arg):
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        msg = "value should be a 'Timedelta', 'NaT', or array of those. Got"
        with pytest.raises(TypeError, match=msg):
            idx.searchsorted(arg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_to_julian_date.py ===
from pandas import Timestamp


class TestTimestampToJulianDate:
    def test_compare_1700(self):
        ts = Timestamp("1700-06-23")
        res = ts.to_julian_date()
        assert res == 2_342_145.5

    def test_compare_2000(self):
        ts = Timestamp("2000-04-12")
        res = ts.to_julian_date()
        assert res == 2_451_646.5

    def test_compare_2100(self):
        ts = Timestamp("2100-08-12")
        res = ts.to_julian_date()
        assert res == 2_488_292.5

    def test_compare_hour01(self):
        ts = Timestamp("2000-08-12T01:00:00")
        res = ts.to_julian_date()
        assert res == 2_451_768.5416666666666666

    def test_compare_hour13(self):
        ts = Timestamp("2000-08-12T13:00:00")
        res = ts.to_julian_date()
        assert res == 2_451_769.0416666666666666

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_sorting.py ===
from sympy.core.sorting import default_sort_key, ordered
from sympy.testing.pytest import raises

from sympy.abc import x


def test_default_sort_key():
    func = lambda x: x
    assert sorted([func, x, func], key=default_sort_key) == [func, func, x]

    class C:
        def __repr__(self):
            return 'x.y'
    func = C()
    assert sorted([x, func], key=default_sort_key) == [func, x]


def test_ordered():
    # Issue 7210 - this had been failing with python2/3 problems
    assert (list(ordered([{1:3, 2:4, 9:10}, {1:3}])) == \
               [{1: 3}, {1: 3, 2: 4, 9: 10}])
    # warnings should not be raised for identical items
    l = [1, 1]
    assert list(ordered(l, warn=True)) == l
    l = [[1], [2], [1]]
    assert list(ordered(l, warn=True)) == [[1], [1], [2]]
    raises(ValueError, lambda: list(ordered(['a', 'ab'], keys=[lambda x: x[0]],
        default=False, warn=True)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_partitions.py ===
from sympy.ntheory.partitions_ import npartitions, _partition_rec, _partition


def test__partition_rec():
    A000041 = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135,
               176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575]
    for n, val in enumerate(A000041):
        assert _partition_rec(n) == val


def test__partition():
    assert [_partition(k) for k in range(13)] == \
        [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77]
    assert _partition(100) == 190569292
    assert _partition(200) == 3972999029388
    assert _partition(1000) == 24061467864032622473692149727991
    assert _partition(1001) == 25032297938763929621013218349796
    assert _partition(2000) == 4720819175619413888601432406799959512200344166
    assert _partition(10000) % 10**10 == 6916435144
    assert _partition(100000) % 10**10 == 9421098519
    assert _partition(10000000) % 10**10 == 7677288980


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert npartitions(0) == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_pickle.py ===
import os
import tempfile
import pickle

from mpmath import *

def pickler(obj):
    fn = tempfile.mktemp()

    f = open(fn, 'wb')
    pickle.dump(obj, f)
    f.close()

    f = open(fn, 'rb')
    obj2 = pickle.load(f)
    f.close()
    os.remove(fn)

    return obj2

def test_pickle():

    obj = mpf('0.5')
    assert obj == pickler(obj)

    obj = mpc('0.5','0.2')
    assert obj == pickler(obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\arrayterator.py ===

from __future__ import annotations

from typing import Any
import numpy as np

AR_i8: np.ndarray[Any, np.dtype[np.int_]] = np.arange(10)
ar_iter = np.lib.Arrayterator(AR_i8)

ar_iter.var
ar_iter.buf_size
ar_iter.start
ar_iter.stop
ar_iter.step
ar_iter.shape
ar_iter.flat

ar_iter.__array__()

for i in ar_iter:
    pass

ar_iter[0]
ar_iter[...]
ar_iter[:]
ar_iter[0, 0, 0]
ar_iter[..., 0, :]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_ops.py ===
import pandas as pd
import pandas._testing as tm


class TestUnaryOps:
    def test_invert(self):
        a = pd.array([True, False, None], dtype="boolean")
        expected = pd.array([False, True, None], dtype="boolean")
        tm.assert_extension_array_equal(~a, expected)

        expected = pd.Series(expected, index=["a", "b", "c"], name="name")
        result = ~pd.Series(a, index=["a", "b", "c"], name="name")
        tm.assert_series_equal(result, expected)

        df = pd.DataFrame({"A": a, "B": [True, False, False]}, index=["a", "b", "c"])
        result = ~df
        expected = pd.DataFrame(
            {"A": expected, "B": [False, True, True]}, index=["a", "b", "c"]
        )
        tm.assert_frame_equal(result, expected)

    def test_abs(self):
        # matching numpy behavior, abs is the identity function
        arr = pd.array([True, False, None], dtype="boolean")
        result = abs(arr)

        tm.assert_extension_array_equal(result, arr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_get.py ===
import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestGet:
    def test_get(self, float_frame):
        b = float_frame.get("B")
        tm.assert_series_equal(b, float_frame["B"])

        assert float_frame.get("foo") is None
        tm.assert_series_equal(
            float_frame.get("foo", float_frame["B"]), float_frame["B"]
        )

    @pytest.mark.parametrize(
        "df",
        [
            DataFrame(),
            DataFrame(columns=list("AB")),
            DataFrame(columns=list("AB"), index=range(3)),
        ],
    )
    def test_get_none(self, df):
        # see gh-5652
        assert df.get(None) is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_skew.py ===
import numpy as np

import pandas as pd
import pandas._testing as tm


def test_groupby_skew_equivalence():
    # Test that that groupby skew method (which uses libgroupby.group_skew)
    #  matches the results of operating group-by-group (which uses nanops.nanskew)
    nrows = 1000
    ngroups = 3
    ncols = 2
    nan_frac = 0.05

    arr = np.random.default_rng(2).standard_normal((nrows, ncols))
    arr[np.random.default_rng(2).random(nrows) < nan_frac] = np.nan

    df = pd.DataFrame(arr)
    grps = np.random.default_rng(2).integers(0, ngroups, size=nrows)
    gb = df.groupby(grps)

    result = gb.skew()

    grpwise = [grp.skew().to_frame(i).T for i, grp in gb]
    expected = pd.concat(grpwise, axis=0)
    expected.index = expected.index.astype(result.index.dtype)  # 32bit builds
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\conftest.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    MultiIndex,
)


# Note: identical the "multi" entry in the top-level "index" fixture
@pytest.fixture
def idx():
    # a MultiIndex used to test the general functionality of the
    # general functionality of this object
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    mi = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )
    return mi

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_libfrequencies.py ===
import pytest

from pandas._libs.tslibs.parsing import get_rule_month

from pandas.tseries import offsets


@pytest.mark.parametrize(
    "obj,expected",
    [
        ("W", "DEC"),
        (offsets.Week().freqstr, "DEC"),
        ("D", "DEC"),
        (offsets.Day().freqstr, "DEC"),
        ("Q", "DEC"),
        (offsets.QuarterEnd(startingMonth=12).freqstr, "DEC"),
        ("Q-JAN", "JAN"),
        (offsets.QuarterEnd(startingMonth=1).freqstr, "JAN"),
        ("Y-DEC", "DEC"),
        (offsets.YearEnd().freqstr, "DEC"),
        ("Y-MAY", "MAY"),
        (offsets.YearEnd(month=5).freqstr, "MAY"),
    ],
)
def test_get_rule_month(obj, expected):
    result = get_rule_month(obj)
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_npy_units.py ===
import numpy as np

from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas._libs.tslibs.vectorized import is_date_array_normalized

# a datetime64 ndarray which *is* normalized
day_arr = np.arange(10, dtype="i8").view("M8[D]")


class TestIsDateArrayNormalized:
    def test_is_date_array_normalized_day(self):
        arr = day_arr
        abbrev = "D"
        unit = abbrev_to_npy_unit(abbrev)
        result = is_date_array_normalized(arr.view("i8"), None, unit)
        assert result is True

    def test_is_date_array_normalized_seconds(self):
        abbrev = "s"
        arr = day_arr.astype(f"M8[{abbrev}]")
        unit = abbrev_to_npy_unit(abbrev)
        result = is_date_array_normalized(arr.view("i8"), None, unit)
        assert result is True

        arr[0] += np.timedelta64(1, abbrev)
        result2 = is_date_array_normalized(arr.view("i8"), None, unit)
        assert result2 is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_abstract_interface.py ===
import pytest

from numpy.f2py import crackfortran
from numpy.testing import IS_WASM

from . import util


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.slow
class TestAbstractInterface(util.F2PyTest):
    sources = [util.getpath("tests", "src", "abstract_interface", "foo.f90")]

    skip = ["add1", "add2"]

    def test_abstract_interface(self):
        assert self.module.ops_module.foo(3, 5) == (8, 13)

    def test_parse_abstract_interface(self):
        # Test gh18403
        fpath = util.getpath("tests", "src", "abstract_interface",
                             "gh18403_mod.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        assert len(mod[0]["body"]) == 1
        assert mod[0]["body"][0]["block"] == "abstract interface"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_subclass.py ===
from pandas import Categorical
import pandas._testing as tm


class SubclassedCategorical(Categorical):
    pass


class TestCategoricalSubclassing:
    def test_constructor(self):
        sc = SubclassedCategorical(["a", "b", "c"])
        assert isinstance(sc, SubclassedCategorical)
        tm.assert_categorical_equal(sc, Categorical(["a", "b", "c"]))

    def test_from_codes(self):
        sc = SubclassedCategorical.from_codes([1, 0, 2], ["a", "b", "c"])
        assert isinstance(sc, SubclassedCategorical)
        exp = Categorical.from_codes([1, 0, 2], ["a", "b", "c"])
        tm.assert_categorical_equal(sc, exp)

    def test_map(self):
        sc = SubclassedCategorical(["a", "b", "c"])
        res = sc.map(lambda x: x.upper(), na_action=None)
        assert isinstance(res, SubclassedCategorical)
        exp = Categorical(["A", "B", "C"])
        tm.assert_categorical_equal(res, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_extension.py ===
"""
Tests for behavior if an author does *not* implement EA methods.
"""
import numpy as np
import pytest

from pandas.core.arrays import ExtensionArray


class MyEA(ExtensionArray):
    def __init__(self, values) -> None:
        self._values = values


@pytest.fixture
def data():
    arr = np.arange(10)
    return MyEA(arr)


class TestExtensionArray:
    def test_errors(self, data, all_arithmetic_operators):
        # invalid ops
        op_name = all_arithmetic_operators
        with pytest.raises(AttributeError):
            getattr(data, op_name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_pickle.py ===
import numpy as np
import pytest

from pandas import (
    NaT,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestPickle:
    @pytest.mark.parametrize("freq", ["D", "M", "Y"])
    def test_pickle_round_trip(self, freq):
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.nan], freq=freq)
        result = tm.round_trip_pickle(idx)
        tm.assert_index_equal(result, idx)

    def test_pickle_freq(self):
        # GH#2891
        prng = period_range("1/1/2011", "1/1/2012", freq="M")
        new_prng = tm.round_trip_pickle(prng)
        assert new_prng.freq == offsets.MonthEnd()
        assert new_prng.freqstr == "M"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_repeat.py ===
import numpy as np
import pytest

from pandas import (
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestRepeat:
    @pytest.mark.parametrize("use_numpy", [True, False])
    @pytest.mark.parametrize(
        "index",
        [
            period_range("2000-01-01", periods=3, freq="D"),
            period_range("2001-01-01", periods=3, freq="2D"),
            PeriodIndex(["2001-01", "NaT", "2003-01"], freq="M"),
        ],
    )
    def test_repeat_freqstr(self, index, use_numpy):
        # GH#10183
        expected = PeriodIndex([per for per in index for _ in range(3)])
        result = np.repeat(index, 3) if use_numpy else index.repeat(3)
        tm.assert_index_equal(result, expected)
        assert result.freqstr == index.freqstr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_validate.py ===
import pytest


@pytest.mark.parametrize(
    "func",
    [
        "reset_index",
        "_set_name",
        "sort_values",
        "sort_index",
        "rename",
        "dropna",
        "drop_duplicates",
    ],
)
@pytest.mark.parametrize("inplace", [1, "True", [1, 2, 3], 5.0])
def test_validate_bool_args(string_series, func, inplace):
    """Tests for error handling related to data types of method arguments."""
    msg = 'For argument "inplace" expected type bool'
    kwargs = {"inplace": inplace}

    if func == "_set_name":
        kwargs["name"] = "hello"

    with pytest.raises(ValueError, match=msg):
        getattr(string_series, func)(**kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_is_monotonic.py ===
import numpy as np

from pandas import (
    Series,
    date_range,
)


class TestIsMonotonic:
    def test_is_monotonic_numeric(self):
        ser = Series(np.random.default_rng(2).integers(0, 10, size=1000))
        assert not ser.is_monotonic_increasing
        ser = Series(np.arange(1000))
        assert ser.is_monotonic_increasing is True
        assert ser.is_monotonic_increasing is True
        ser = Series(np.arange(1000, 0, -1))
        assert ser.is_monotonic_decreasing is True

    def test_is_monotonic_dt64(self):
        ser = Series(date_range("20130101", periods=10))
        assert ser.is_monotonic_increasing is True
        assert ser.is_monotonic_increasing is True

        ser = Series(list(reversed(ser)))
        assert ser.is_monotonic_increasing is False
        assert ser.is_monotonic_decreasing is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\conftest.py ===
import pytest


@pytest.fixture(params=[True, False])
def check_dtype(request):
    return request.param


@pytest.fixture(params=[True, False])
def check_exact(request):
    return request.param


@pytest.fixture(params=[True, False])
def check_index_type(request):
    return request.param


@pytest.fixture(params=[0.5e-3, 0.5e-5])
def rtol(request):
    return request.param


@pytest.fixture(params=[True, False])
def check_categorical(request):
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\test_categorical.py ===
import numpy as np

from pandas import (
    Categorical,
    Series,
)
import pandas._testing as tm


class TestCategoricalComparisons:
    def test_categorical_nan_equality(self):
        cat = Series(Categorical(["a", "b", "c", np.nan]))
        expected = Series([True, True, True, False])
        result = cat == cat
        tm.assert_series_equal(result, expected)

    def test_categorical_tuple_equality(self):
        # GH 18050
        ser = Series([(0, 0), (0, 1), (0, 0), (1, 0), (1, 1)])
        expected = Series([True, False, True, False, False])
        result = ser == (0, 0)
        tm.assert_series_equal(result, expected)

        result = ser.astype("category") == (0, 0)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\__init__.py ===
def get_groupby_method_args(name, obj):
    """
    Get required arguments for a groupby method.

    When parametrizing a test over groupby methods (e.g. "sum", "mean", "fillna"),
    it is often the case that arguments are required for certain methods.

    Parameters
    ----------
    name: str
        Name of the method.
    obj: Series or DataFrame
        pandas object that is being grouped.

    Returns
    -------
    A tuple of required arguments for the method.
    """
    if name in ("nth", "fillna", "take"):
        return (0,)
    if name == "quantile":
        return (0.5,)
    if name == "corrwith":
        return (obj,)
    return ()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\accessors\test_str_accessor.py ===
import pytest

from pandas import Series
import pandas._testing as tm


class TestStrAccessor:
    def test_str_attribute(self):
        # GH#9068
        methods = ["strip", "rstrip", "lstrip"]
        ser = Series([" jack", "jill ", " jesse ", "frank"])
        for method in methods:
            expected = Series([getattr(str, method)(x) for x in ser.values])
            tm.assert_series_equal(getattr(Series.str, method)(ser.str), expected)

        # str accessor only valid with string values
        ser = Series(range(5))
        with pytest.raises(AttributeError, match="only use .str accessor"):
            ser.str.repeat(2)

    def test_str_accessor_updates_on_inplace(self):
        ser = Series(list("abc"))
        return_value = ser.drop([0], inplace=True)
        assert return_value is None
        assert len(ser.str.lower()) == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_ast_parser.py ===
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.parsing.ast_parser import parse_expr
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
import warnings

def test_parse_expr():
    a, b = symbols('a, b')
    # tests issue_16393
    assert parse_expr('a + b', {}) == a + b
    raises(SympifyError, lambda: parse_expr('a + ', {}))

    # tests Transform.visit_Constant
    assert parse_expr('1 + 2', {}) == S(3)
    assert parse_expr('1 + 2.0', {}) == S(3.0)

    # tests Transform.visit_Name
    assert parse_expr('Rational(1, 2)', {}) == S(1)/2
    assert parse_expr('a', {'a': a}) == a

    # tests issue_23092
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert parse_expr('6 * 7', {}) == S(42)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sandbox\tests\test_indexed_integrals.py ===
from sympy.sandbox.indexed_integrals import IndexedIntegral
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.indexed import (Idx, IndexedBase)


def test_indexed_integrals():
    A = IndexedBase('A')
    i, j = symbols('i j', integer=True)
    a1, a2 = symbols('a1:3', cls=Idx)
    assert isinstance(a1, Idx)

    assert IndexedIntegral(1, A[i]).doit() == A[i]
    assert IndexedIntegral(A[i], A[i]).doit() == A[i] ** 2 / 2
    assert IndexedIntegral(A[j], A[i]).doit() == A[i] * A[j]
    assert IndexedIntegral(A[i] * A[j], A[i]).doit() == A[i] ** 2 * A[j] / 2
    assert IndexedIntegral(sin(A[i]), A[i]).doit() == -cos(A[i])
    assert IndexedIntegral(sin(A[j]), A[i]).doit() == sin(A[j]) * A[i]

    assert IndexedIntegral(1, A[a1]).doit() == A[a1]
    assert IndexedIntegral(A[a1], A[a1]).doit() == A[a1] ** 2 / 2
    assert IndexedIntegral(A[a2], A[a1]).doit() == A[a1] * A[a2]
    assert IndexedIntegral(A[a1] * A[a2], A[a1]).doit() == A[a1] ** 2 * A[a2] / 2
    assert IndexedIntegral(sin(A[a1]), A[a1]).doit() == -cos(A[a1])
    assert IndexedIntegral(sin(A[a2]), A[a1]).doit() == sin(A[a2]) * A[a1]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\examples\limited_api\setup.py ===
"""
Build an example package using the limited Python C API.
"""

import os

from setuptools import Extension, setup

import numpy as np

macros = [("NPY_NO_DEPRECATED_API", 0), ("Py_LIMITED_API", "0x03060000")]

limited_api = Extension(
    "limited_api",
    sources=[os.path.join('.', "limited_api.c")],
    include_dirs=[np.get_include()],
    define_macros=macros,
)

extensions = [limited_api]

setup(
    ext_modules=extensions
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_corrwith.py ===
import numpy as np

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


def test_corrwith_with_1_axis():
    # GH 47723
    df = DataFrame({"a": [1, 1, 2], "b": [3, 7, 4]})
    gb = df.groupby("a")

    msg = "DataFrameGroupBy.corrwith with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = gb.corrwith(df, axis=1)
    index = Index(
        data=[(1, 0), (1, 1), (1, 2), (2, 2), (2, 0), (2, 1)],
        name=("a", None),
    )
    expected = Series([np.nan] * 6, index=index)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\test_timezones.py ===
"""
Tests for Timestamp timezone-related methods
"""
from datetime import datetime

from pandas._libs.tslibs import timezones

from pandas import Timestamp


class TestTimestampTZOperations:
    # ------------------------------------------------------------------

    def test_timestamp_timetz_equivalent_with_datetime_tz(self, tz_naive_fixture):
        # GH21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)

        stamp = Timestamp("2018-06-04 10:20:30", tz=tz)
        _datetime = datetime(2018, 6, 4, hour=10, minute=20, second=30, tzinfo=tz)

        result = stamp.timetz()
        expected = _datetime.timetz()

        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_nunique.py ===
import numpy as np

from pandas import (
    Categorical,
    Series,
)


def test_nunique():
    # basics.rst doc example
    series = Series(np.random.default_rng(2).standard_normal(500))
    series[20:500] = np.nan
    series[10:20] = 5000
    result = series.nunique()
    assert result == 11


def test_nunique_categorical():
    # GH#18051
    ser = Series(Categorical([]))
    assert ser.nunique() == 0

    ser = Series(Categorical([np.nan]))
    assert ser.nunique() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_multidimensional.py ===
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import sin
from sympy.core.multidimensional import vectorize
x, y, z = symbols('x y z')
f, g, h = list(map(Function, 'fgh'))


def test_vectorize():
    @vectorize(0)
    def vsin(x):
        return sin(x)

    assert vsin([1, x, y]) == [sin(1), sin(x), sin(y)]

    @vectorize(0, 1)
    def vdiff(f, y):
        return diff(f, y)

    assert vdiff([f(x, y, z), g(x, y, z), h(x, y, z)], [x, y, z]) == \
         [[Derivative(f(x, y, z), x), Derivative(f(x, y, z), y),
           Derivative(f(x, y, z), z)], [Derivative(g(x, y, z), x),
                     Derivative(g(x, y, z), y), Derivative(g(x, y, z), z)],
         [Derivative(h(x, y, z), x), Derivative(h(x, y, z), y), Derivative(h(x, y, z), z)]]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_type_F.py ===
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix
from sympy.core.backend import S

def test_type_F():
    c = CartanType("F4")
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -2, 0, 0, -1, 2, -1, 0, 0, -1, 2])
    assert c.cartan_matrix() == m
    assert c.dimension() == 4
    assert c.simple_root(1) == [1, -1, 0, 0]
    assert c.simple_root(2) == [0, 1, -1, 0]
    assert c.simple_root(3) == [0, 0, 0, 1]
    assert c.simple_root(4) == [-S.Half, -S.Half, -S.Half, -S.Half]
    assert c.roots() == 48
    assert c.basis() == 52
    diag = "0---0=>=0---0\n" + "   ".join(str(i) for i in range(1, 5))
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0], 3: [1, 0, -1, 0],
            4: [1, 0, 1, 0], 5: [1, 0, 0, -1], 6: [1, 0, 0, 1], 7: [0, 1, -1, 0],
            8: [0, 1, 1, 0], 9: [0, 1, 0, -1], 10: [0, 1, 0, 1], 11: [0, 0, 1, -1],
            12: [0, 0, 1, 1], 13: [1, 0, 0, 0], 14: [0, 1, 0, 0], 15: [0, 0, 1, 0],
            16: [0, 0, 0, 1], 17: [S.Half, S.Half, S.Half, S.Half], 18: [S.Half, -S.Half, S.Half, S.Half],
            19: [S.Half, S.Half, -S.Half, S.Half], 20: [S.Half, S.Half, S.Half, -S.Half], 21: [S.Half, S.Half, -S.Half, -S.Half],
            22: [S.Half, -S.Half, S.Half, -S.Half], 23: [S.Half, -S.Half, -S.Half, S.Half], 24: [S.Half, -S.Half, -S.Half, -S.Half]}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_hypothesis.py ===
from hypothesis import given
from hypothesis import strategies as st
from sympy import divisors
from sympy.functions.combinatorial.numbers import divisor_sigma, totient
from sympy.ntheory.primetest import is_square


@given(n=st.integers(1, 10**10))
def test_tau_hypothesis(n):
    div = divisors(n)
    tau_n = len(div)
    assert is_square(n) == (tau_n % 2 == 1)
    sigmas = [divisor_sigma(i) for i in div]
    totients = [totient(n // i) for i in div]
    mul = [a * b for a, b in zip(sigmas, totients)]
    assert n * tau_n == sum(mul)


@given(n=st.integers(1, 10**10))
def test_totient_hypothesis(n):
    assert totient(n) <= n
    div = divisors(n)
    totients = [totient(i) for i in div]
    assert n == sum(totients)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_common.py ===
import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestCommonBlock(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "block.f")]

    def test_common_block(self):
        self.module.initcb()
        assert self.module.block.long_bn == np.array(1.0, dtype=np.float64)
        assert self.module.block.string_bn == np.array("2", dtype="|S1")
        assert self.module.block.ok == np.array(3, dtype=np.int32)


class TestCommonWithUse(util.F2PyTest):
    sources = [util.getpath("tests", "src", "common", "gh19161.f90")]

    def test_common_gh19161(self):
        assert self.module.data.x == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_resolution.py ===
import pytest

import pandas as pd


class TestResolution:
    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("Y", "year"),
            ("Q", "quarter"),
            ("M", "month"),
            ("D", "day"),
            ("h", "hour"),
            ("min", "minute"),
            ("s", "second"),
            ("ms", "millisecond"),
            ("us", "microsecond"),
        ],
    )
    def test_resolution(self, freq, expected):
        idx = pd.period_range(start="2013-04-01", periods=30, freq=freq)
        assert idx.resolution == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_is_full.py ===
import pytest

from pandas import PeriodIndex


def test_is_full():
    index = PeriodIndex([2005, 2007, 2009], freq="Y")
    assert not index.is_full

    index = PeriodIndex([2005, 2006, 2007], freq="Y")
    assert index.is_full

    index = PeriodIndex([2005, 2005, 2007], freq="Y")
    assert not index.is_full

    index = PeriodIndex([2005, 2005, 2006], freq="Y")
    assert index.is_full

    index = PeriodIndex([2006, 2005, 2005], freq="Y")
    with pytest.raises(ValueError, match="Index is not monotonic"):
        index.is_full

    assert index[:0].is_full

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\__init__.py ===
import numpy as np

import pandas as pd


def is_object_or_nan_string_dtype(dtype):
    """
    Check if string-like dtype is following NaN semantics, i.e. is object
    dtype or a NaN-variant of the StringDtype.
    """
    return (isinstance(dtype, np.dtype) and dtype == "object") or (
        dtype.na_value is np.nan
    )


def _convert_na_value(ser, expected):
    if ser.dtype != object:
        if ser.dtype.na_value is np.nan:
            expected = expected.fillna(np.nan)
        else:
            # GH#18463
            expected = expected.fillna(pd.NA)
    return expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_tzconversion.py ===
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.tzconversion import tz_localize_to_utc


class TestTZLocalizeToUTC:
    def test_tz_localize_to_utc_ambiguous_infer(self):
        # val is a timestamp that is ambiguous when localized to US/Eastern
        val = 1_320_541_200_000_000_000
        vals = np.array([val, val - 1, val], dtype=np.int64)

        with pytest.raises(pytz.AmbiguousTimeError, match="2011-11-06 01:00:00"):
            tz_localize_to_utc(vals, pytz.timezone("US/Eastern"), ambiguous="infer")

        with pytest.raises(pytz.AmbiguousTimeError, match="are no repeated times"):
            tz_localize_to_utc(vals[:1], pytz.timezone("US/Eastern"), ambiguous="infer")

        vals[1] += 1
        msg = "There are 2 dst switches when there should only be 1"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            tz_localize_to_utc(vals, pytz.timezone("US/Eastern"), ambiguous="infer")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_approximants.py ===
from sympy.series import approximants
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import (fibonacci, lucas)


def test_approximants():
    x, t = symbols("x,t")
    g = [lucas(k) for k in range(16)]
    assert list(approximants(g)) == (
        [2, -4/(x - 2), (5*x - 2)/(3*x - 1), (x - 2)/(x**2 + x - 1)] )
    g = [lucas(k)+fibonacci(k+2) for k in range(16)]
    assert list(approximants(g)) == (
        [3, -3/(x - 1), (3*x - 3)/(2*x - 1), -3/(x**2 + x - 1)] )
    g = [lucas(k)**2 for k in range(16)]
    assert list(approximants(g)) == (
        [4, -16/(x - 4), (35*x - 4)/(9*x - 1), (37*x - 28)/(13*x**2 + 11*x - 7),
        (50*x**2 + 63*x - 52)/(37*x**2 + 19*x - 13),
        (-x**2 - 7*x + 4)/(x**3 - 2*x**2 - 2*x + 1)] )
    p = [sum(binomial(k,i)*x**i for i in range(k+1)) for k in range(16)]
    y = approximants(p, t, simplify=True)
    assert next(y) == 1
    assert next(y) == -1/(t*(x + 1) - 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_kauers.py ===
from sympy.series.kauers import finite_diff
from sympy.series.kauers import finite_diff_kauers
from sympy.abc import x, y, z, m, n, w
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.concrete.summations import Sum


def test_finite_diff():
    assert finite_diff(x**2 + 2*x + 1, x) == 2*x + 3
    assert finite_diff(y**3 + 2*y**2 + 3*y + 5, y) == 3*y**2 + 7*y + 6
    assert finite_diff(z**2 - 2*z + 3, z) == 2*z - 1
    assert finite_diff(w**2 + 3*w - 2, w) == 2*w + 4
    assert finite_diff(sin(x), x, pi/6) == -sin(x) + sin(x + pi/6)
    assert finite_diff(cos(y), y, pi/3) == -cos(y) + cos(y + pi/3)
    assert finite_diff(x**2 - 2*x + 3, x, 2) == 4*x
    assert finite_diff(n**2 - 2*n + 3, n, 3) == 6*n + 3

def test_finite_diff_kauers():
    assert finite_diff_kauers(Sum(x**2, (x, 1, n))) == (n + 1)**2
    assert finite_diff_kauers(Sum(y, (y, 1, m))) == (m + 1)
    assert finite_diff_kauers(Sum((x*y), (x, 1, m), (y, 1, n))) == (m + 1)*(n + 1)
    assert finite_diff_kauers(Sum((x*y**2), (x, 1, m), (y, 1, n))) == (n + 1)**2*(m + 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\echo-server.py ===
#!/usr/bin/env python

# From https://github.com/aaugustin/websockets/blob/main/example/echo.py

import asyncio
import os

import websockets

LOCAL_WS_SERVER_PORT = int(os.environ.get("LOCAL_WS_SERVER_PORT", "8765"))


async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)


async def main():
    async with websockets.serve(echo, "localhost", LOCAL_WS_SERVER_PORT):
        await asyncio.Future()  # run forever


asyncio.run(main())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\lib_user_array.py ===
"""Based on the `if __name__ == "__main__"` test code in `lib/_user_array_impl.py`."""

from __future__ import annotations

import numpy as np
from numpy.lib.user_array import container

N = 10_000
W = H = int(N**0.5)

a: np.ndarray[tuple[int, int], np.dtype[np.int32]]
ua: container[tuple[int, int], np.dtype[np.int32]]

a = np.arange(N, dtype=np.int32).reshape(W, H)
ua = container(a)

ua_small: container[tuple[int, int], np.dtype[np.int32]] = ua[:3, :5]
ua_small[0, 0] = 10

ua_bool: container[tuple[int, int], np.dtype[np.bool]] = ua_small > 1

# shape: tuple[int, int] = np.shape(ua)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_get_value.py ===
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)


class TestGetValue:
    def test_get_set_value_no_partial_indexing(self):
        # partial w/ MultiIndex raise exception
        index = MultiIndex.from_tuples([(0, 1), (0, 2), (1, 1), (1, 2)])
        df = DataFrame(index=index, columns=range(4))
        with pytest.raises(KeyError, match=r"^0$"):
            df._get_value(0, 1)

    def test_get_value(self, float_frame):
        for idx in float_frame.index:
            for col in float_frame.columns:
                result = float_frame._get_value(idx, col)
                expected = float_frame[col][idx]
                assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\methods\test_fillna.py ===
from pandas import (
    Index,
    NaT,
    Timedelta,
    TimedeltaIndex,
)
import pandas._testing as tm


class TestFillNA:
    def test_fillna_timedelta(self):
        # GH#11343
        idx = TimedeltaIndex(["1 day", NaT, "3 day"])

        exp = TimedeltaIndex(["1 day", "2 day", "3 day"])
        tm.assert_index_equal(idx.fillna(Timedelta("2 day")), exp)

        exp = TimedeltaIndex(["1 day", "3 hour", "3 day"])
        idx.fillna(Timedelta("3 hour"))

        exp = Index([Timedelta("1 day"), "x", Timedelta("3 day")], dtype=object)
        tm.assert_index_equal(idx.fillna("x"), exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_normalize.py ===
import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit


class TestTimestampNormalize:
    @pytest.mark.parametrize("arg", ["2013-11-30", "2013-11-30 12:00:00"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_normalize(self, tz_naive_fixture, arg, unit):
        tz = tz_naive_fixture
        ts = Timestamp(arg, tz=tz).as_unit(unit)
        result = ts.normalize()
        expected = Timestamp("2013-11-30", tz=tz)
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    def test_normalize_pre_epoch_dates(self):
        # GH: 36294
        result = Timestamp("1969-01-01 09:00:00").normalize()
        expected = Timestamp("1969-01-01 00:00:00")
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_size.py ===
import pytest

from pandas import Series


@pytest.mark.parametrize(
    "data, index, expected",
    [
        ([1, 2, 3], None, 3),
        ({"a": 1, "b": 2, "c": 3}, None, 3),
        ([1, 2, 3], ["x", "y", "z"], 3),
        ([1, 2, 3, 4, 5], ["x", "y", "z", "w", "n"], 5),
        ([1, 2, 3], None, 3),
        ([1, 2, 3], ["x", "y", "z"], 3),
        ([1, 2, 3, 4], ["x", "y", "z", "w"], 4),
    ],
)
def test_series(data, index, expected):
    # GH#52897
    ser = Series(data, index=index)
    assert ser.size == expected
    assert isinstance(ser.size, int)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_singularityfunctions.py ===
from sympy.integrals.singularityfunctions import singularityintegrate
from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.special.singularity_functions import SingularityFunction

x, a, n, y = symbols('x a n y')
f = Function('f')


def test_singularityintegrate():
    assert singularityintegrate(x, x) is None
    assert singularityintegrate(x + SingularityFunction(x, 9, 1), x) is None

    assert 4*singularityintegrate(SingularityFunction(x, a, 3), x) == 4*SingularityFunction(x, a, 4)/4
    assert singularityintegrate(5*SingularityFunction(x, 5, -2), x) == 5*SingularityFunction(x, 5, -1)
    assert singularityintegrate(6*SingularityFunction(x, 5, -1), x) == 6*SingularityFunction(x, 5, 0)
    assert singularityintegrate(x*SingularityFunction(x, 0, -1), x) == 0
    assert singularityintegrate((x - 5)*SingularityFunction(x, 5, -1), x) == 0
    assert singularityintegrate(SingularityFunction(x, 0, -1) * f(x), x) == f(0) * SingularityFunction(x, 0, 0)
    assert singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x) == f(1) * SingularityFunction(x, 1, 0)
    assert singularityintegrate(y*SingularityFunction(x, 0, -1)**2, x) == \
        y*SingularityFunction(0, 0, -1)*SingularityFunction(x, 0, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_type_C.py ===
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

def test_type_C():
    c = CartanType("C4")
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -2, 2])
    assert c.cartan_matrix() == m
    assert c.dimension() == 4
    assert c.simple_root(4) == [0, 0, 0, 2]
    assert c.roots() == 32
    assert c.basis() == 36
    assert c.lie_algebra() == "sp(8)"
    t = CartanType(['C', 3])
    assert t.dimension() == 3
    diag = "0---0---0=<=0\n1   2   3   4"
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0],
            3: [1, 0, -1, 0], 4: [1, 0, 1, 0], 5: [1, 0, 0, -1],
            6: [1, 0, 0, 1], 7: [0, 1, -1, 0], 8: [0, 1, 1, 0],
            9: [0, 1, 0, -1], 10: [0, 1, 0, 1], 11: [0, 0, 1, -1],
            12: [0, 0, 1, 1], 13: [2, 0, 0, 0], 14: [0, 2, 0, 0], 15: [0, 0, 2, 0],
            16: [0, 0, 0, 2]}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_type_E.py ===
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix
from sympy.core.backend import Rational

def test_type_E():
    c = CartanType("E6")
    m = Matrix(6, 6, [2, 0, -1, 0, 0, 0, 0, 2, 0, -1, 0, 0,
        -1, 0, 2, -1, 0, 0, 0, -1, -1, 2, -1, 0, 0, 0, 0,
        -1, 2, -1, 0, 0, 0, 0, -1, 2])
    assert c.cartan_matrix() == m
    assert c.dimension() == 8
    assert c.simple_root(6) == [0, 0, 0, -1, 1, 0, 0, 0]
    assert c.roots() == 72
    assert c.basis() == 78
    diag = " "*8 + "2\n" + " "*8 + "0\n" + " "*8 + "|\n" + " "*8 + "|\n"
    diag += "---".join("0" for i in range(1, 6))+"\n"
    diag += "1   " + "   ".join(str(i) for i in range(3, 7))
    assert c.dynkin_diagram() == diag
    posroots = c.positive_roots()
    assert posroots[8] == [1, 0, 0, 0, 1, 0, 0, 0]
    assert posroots[21] == [Rational(1,2),Rational(1,2),Rational(1,2),Rational(1,2),
                            Rational(1,2),Rational(-1,2),Rational(-1,2),Rational(1,2)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_deprecated_conv_modules.py ===
from sympy import MatrixSymbol, symbols, Sum
from sympy.tensor.array.expressions import conv_array_to_indexed, from_array_to_indexed, ArrayTensorProduct, \
    ArrayContraction, conv_array_to_matrix, from_array_to_matrix, conv_matrix_to_array, from_matrix_to_array, \
    conv_indexed_to_array, from_indexed_to_array
from sympy.testing.pytest import warns
from sympy.utilities.exceptions import SymPyDeprecationWarning


def test_deprecated_conv_module_results():

    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    i, j, d = symbols("i j d")

    x = ArrayContraction(ArrayTensorProduct(M, N), (1, 2))
    y = Sum(M[i, d]*N[d, j], (d, 0, 2))

    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        assert conv_array_to_indexed.convert_array_to_indexed(x, [i, j]).dummy_eq(from_array_to_indexed.convert_array_to_indexed(x, [i, j]))
        assert conv_array_to_matrix.convert_array_to_matrix(x) == from_array_to_matrix.convert_array_to_matrix(x)
        assert conv_matrix_to_array.convert_matrix_to_array(M*N) == from_matrix_to_array.convert_matrix_to_array(M*N)
        assert conv_indexed_to_array.convert_indexed_to_array(y) == from_indexed_to_array.convert_indexed_to_array(y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\test_ndim_array_conversions.py ===
from sympy.tensor.array import (ImmutableDenseNDimArray,
        ImmutableSparseNDimArray, MutableDenseNDimArray, MutableSparseNDimArray)
from sympy.abc import x, y, z


def test_NDim_array_conv():
    MD = MutableDenseNDimArray([x, y, z])
    MS = MutableSparseNDimArray([x, y, z])
    ID = ImmutableDenseNDimArray([x, y, z])
    IS = ImmutableSparseNDimArray([x, y, z])

    assert MD.as_immutable() == ID
    assert MD.as_mutable() == MD

    assert MS.as_immutable() == IS
    assert MS.as_mutable() == MS

    assert ID.as_immutable() == ID
    assert ID.as_mutable() == MD

    assert IS.as_immutable() == IS
    assert IS.as_mutable() == MS

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_size.py ===
import numpy as np
import pytest

from pandas import DataFrame


@pytest.mark.parametrize(
    "data, index, expected",
    [
        ({"col1": [1], "col2": [3]}, None, 2),
        ({}, None, 0),
        ({"col1": [1, np.nan], "col2": [3, 4]}, None, 4),
        ({"col1": [1, 2], "col2": [3, 4]}, [["a", "b"], [1, 2]], 4),
        ({"col1": [1, 2, 3, 4], "col2": [3, 4, 5, 6]}, ["x", "y", "a", "b"], 8),
    ],
)
def test_size(data, index, expected):
    # GH#52897
    df = DataFrame(data, index=index)
    assert df.size == expected
    assert isinstance(df.size, int)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\string\test_astype.py ===
from pandas import (
    Index,
    Series,
)
import pandas._testing as tm


def test_astype_str_from_bytes():
    # https://github.com/pandas-dev/pandas/issues/38607
    # GH#49658 pre-2.0 Index called .values.astype(str) here, which effectively
    #  did a .decode() on the bytes object.  In 2.0 we go through
    #  ensure_string_array which does f"{val}"
    idx = Index(["あ", b"a"], dtype="object")
    result = idx.astype(str)
    expected = Index(["あ", "a"], dtype="str")
    tm.assert_index_equal(result, expected)

    # while we're here, check that Series.astype behaves the same
    result = Series(idx).astype(str)
    expected = Series(expected, dtype="str")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_deprecated_kwargs.py ===
"""
Tests for the deprecated keyword arguments for `read_json`.
"""
from io import StringIO

import pandas as pd
import pandas._testing as tm

from pandas.io.json import read_json


def test_good_kwargs():
    df = pd.DataFrame({"A": [2, 4, 6], "B": [3, 6, 9]}, index=[0, 1, 2])

    with tm.assert_produces_warning(None):
        data1 = StringIO(df.to_json(orient="split"))
        tm.assert_frame_equal(df, read_json(data1, orient="split"))
        data2 = StringIO(df.to_json(orient="columns"))
        tm.assert_frame_equal(df, read_json(data2, orient="columns"))
        data3 = StringIO(df.to_json(orient="index"))
        tm.assert_frame_equal(df, read_json(data3, orient="index"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_set_name.py ===
from datetime import datetime

from pandas import Series


class TestSetName:
    def test_set_name(self):
        ser = Series([1, 2, 3])
        ser2 = ser._set_name("foo")
        assert ser2.name == "foo"
        assert ser.name is None
        assert ser is not ser2

    def test_set_name_attribute(self):
        ser = Series([1, 2, 3])
        ser2 = Series([1, 2, 3], name="bar")
        for name in [7, 7.0, "name", datetime(2001, 1, 1), (1,), "\u05D0"]:
            ser.name = name
            assert ser.name == name
            ser2.name = name
            assert ser2.name == name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_shor.py ===
from sympy.testing.pytest import XFAIL

from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.shor import CMod, getr


@XFAIL
def test_CMod():
    assert qapply(CMod(4, 2, 2)*Qubit(0, 0, 1, 0, 0, 0, 0, 0)) == \
        Qubit(0, 0, 1, 0, 0, 0, 0, 0)
    assert qapply(CMod(5, 5, 7)*Qubit(0, 0, 1, 0, 0, 0, 0, 0, 0, 0)) == \
        Qubit(0, 0, 1, 0, 0, 0, 0, 0, 1, 0)
    assert qapply(CMod(3, 2, 3)*Qubit(0, 1, 0, 0, 0, 0)) == \
        Qubit(0, 1, 0, 0, 0, 1)


def test_continued_frac():
    assert getr(513, 1024, 10) == 2
    assert getr(169, 1024, 11) == 6
    assert getr(314, 4096, 16) == 13

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_sho.py ===
from sympy.core import symbols, Rational, Function, diff
from sympy.physics.sho import R_nl, E_nl
from sympy.simplify.simplify import simplify


def test_sho_R_nl():
    omega, r = symbols('omega r')
    l = symbols('l', integer=True)
    u = Function('u')

    # check that it obeys the Schrodinger equation
    for n in range(5):
        schreq = ( -diff(u(r), r, 2)/2 + ((l*(l + 1))/(2*r**2)
                    + omega**2*r**2/2 - E_nl(n, l, omega))*u(r) )
        result = schreq.subs(u(r), r*R_nl(n, l, omega/2, r))
        assert simplify(result.doit()) == 0


def test_energy():
    n, l, hw = symbols('n l hw')
    assert simplify(E_nl(n, l, hw) - (2*n + l + Rational(3, 2))*hw) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_block_docstring.py ===
import sys

import pytest

from numpy.testing import IS_PYPY

from . import util


@pytest.mark.slow
class TestBlockDocString(util.F2PyTest):
    sources = [util.getpath("tests", "src", "block_docstring", "foo.f")]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_block_docstring(self):
        expected = "bar : 'i'-array(2,3)\n"
        assert self.module.block.__doc__ == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\linalg\tests\test_deprecations.py ===
"""Test deprecation and future warnings.

"""
import numpy as np
from numpy.testing import assert_warns


def test_qr_mode_full_future_warning():
    """Check mode='full' FutureWarning.

    In numpy 1.8 the mode options 'full' and 'economic' in linalg.qr were
    deprecated. The release date will probably be sometime in the summer
    of 2013.

    """
    a = np.eye(2)
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='full')
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='f')
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='economic')
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='e')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_concat.py ===
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        (["Float64", "Float64"], "Float64"),
        (["Float32", "Float64"], "Float64"),
        (["Float32", "Float32"], "Float32"),
    ],
)
def test_concat_series(to_concat_dtypes, result_dtype):
    result = pd.concat([pd.Series([1, 2, pd.NA], dtype=t) for t in to_concat_dtypes])
    expected = pd.concat([pd.Series([1, 2, pd.NA], dtype=object)] * 2).astype(
        result_dtype
    )
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\timedeltas\test_cumulative.py ===
import pytest

import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestAccumulator:
    def test_accumulators_disallowed(self):
        # GH#50297
        arr = TimedeltaArray._from_sequence(["1D", "2D"], dtype="m8[ns]")
        with pytest.raises(TypeError, match="cumprod not supported"):
            arr._accumulate("cumprod")

    def test_cumsum(self, unit):
        # GH#50297
        dtype = f"m8[{unit}]"
        arr = TimedeltaArray._from_sequence(["1D", "2D"], dtype=dtype)
        result = arr._accumulate("cumsum")
        expected = TimedeltaArray._from_sequence(["1D", "3D"], dtype=dtype)
        tm.assert_timedelta_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_construct_object_arr.py ===
import pytest

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike


@pytest.mark.parametrize("datum1", [1, 2.0, "3", (4, 5), [6, 7], None])
@pytest.mark.parametrize("datum2", [8, 9.0, "10", (11, 12), [13, 14], None])
def test_cast_1d_array(datum1, datum2):
    data = [datum1, datum2]
    result = construct_1d_object_array_from_listlike(data)

    # Direct comparison fails: https://github.com/numpy/numpy/issues/10218
    assert result.dtype == "object"
    assert list(result) == data


@pytest.mark.parametrize("val", [1, 2.0, None])
def test_cast_1d_array_invalid_scalar(val):
    with pytest.raises(TypeError, match="has no len()"):
        construct_1d_object_array_from_listlike(val)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_elliptic_curve.py ===
from sympy.ntheory.elliptic_curve import EllipticCurve


def test_elliptic_curve():
    # Point addition and multiplication
    e3 = EllipticCurve(-1, 9)
    p = e3(0, 3)
    q = e3(-1, 3)
    r = p + q
    assert r.x == 1 and r.y == -3
    r = 2*p + q
    assert r.x == 35 and r.y == 207
    r = -p + q
    assert r.x == 37 and r.y == 225
    # Verify result in http://www.lmfdb.org/EllipticCurve/Q
    # Discriminant
    assert EllipticCurve(-1, 9).discriminant == -34928
    assert EllipticCurve(-2731, -55146, 1, 0, 1).discriminant == 25088
    # Torsion points
    assert len(EllipticCurve(0, 1).torsion_points()) == 6

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_stack_saved.py ===
import greenlet
from . import TestCase


class Test(TestCase):

    def test_stack_saved(self):
        main = greenlet.getcurrent()
        self.assertEqual(main._stack_saved, 0)

        def func():
            main.switch(main._stack_saved)

        g = greenlet.greenlet(func)
        x = g.switch()
        self.assertGreater(x, 0)
        self.assertGreater(g._stack_saved, 0)
        g.switch()
        self.assertEqual(g._stack_saved, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_identify.py ===
from mpmath import *

def test_pslq():
    mp.dps = 15
    assert pslq([3*pi+4*e/7, pi, e, log(2)]) == [7, -21, -4, 0]
    assert pslq([4.9999999999999991, 1]) == [1, -5]
    assert pslq([2,1]) == [1, -2]

def test_identify():
    mp.dps = 20
    assert identify(zeta(4), ['log(2)', 'pi**4']) == '((1/90)*pi**4)'
    mp.dps = 15
    assert identify(exp(5)) == 'exp(5)'
    assert identify(exp(4)) == 'exp(4)'
    assert identify(log(5)) == 'log(5)'
    assert identify(exp(3*pi), ['pi']) == 'exp((3*pi))'
    assert identify(3, full=True) == ['3', '3', '1/(1/3)', 'sqrt(9)',
        '1/sqrt((1/9))', '(sqrt(12)/2)**2', '1/(sqrt(12)/6)**2']
    assert identify(pi+1, {'a':+pi}) == '(1 + 1*a)'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\flatiter.py ===
import numpy as np

a = np.empty((2, 2)).flat

a.base
a.copy()
a.coords
a.index
iter(a)
next(a)
a[0]
a[[0, 1, 2]]
a[...]
a[:]
a.__array__()
a.__array__(np.dtype(np.float64))

b = np.array([1]).flat
a[b]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\lib_utils.py ===
from __future__ import annotations

from io import StringIO

import numpy as np
import numpy.lib.array_utils as array_utils

FILE = StringIO()
AR = np.arange(10, dtype=np.float64)


def func(a: int) -> bool:
    return True


array_utils.byte_bounds(AR)
array_utils.byte_bounds(np.float64())

np.info(1, output=FILE)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\shape.py ===
from typing import Any, NamedTuple, cast

import numpy as np


# Subtype of tuple[int, int]
class XYGrid(NamedTuple):
    x_axis: int
    y_axis: int

# Test variance of _ShapeT_co
def accepts_2d(a: np.ndarray[tuple[int, int], Any]) -> None:
    return None


accepts_2d(np.empty(XYGrid(2, 2)))
accepts_2d(np.zeros(XYGrid(2, 2), dtype=int))
accepts_2d(np.ones(XYGrid(2, 2), dtype=int))
accepts_2d(np.full(XYGrid(2, 2), fill_value=5, dtype=int))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_warnings.py ===
import pytest

import pandas._testing as tm


class TestCategoricalWarnings:
    def test_tab_complete_warning(self, ip):
        # https://github.com/pandas-dev/pandas/issues/16409
        pytest.importorskip("IPython", minversion="6.0.0")
        from IPython.core.completer import provisionalcompleter

        code = "import pandas as pd; c = pd.Categorical([])"
        ip.run_cell(code)

        # GH 31324 newer jedi version raises Deprecation warning;
        #  appears resolved 2021-02-02
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                list(ip.Completer.completions("c.", 1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_indexing.py ===
import pandas as pd
import pandas._testing as tm


def test_array_setitem_nullable_boolean_mask():
    # GH 31446
    ser = pd.Series([1, 2], dtype="Int64")
    result = ser.where(ser > 1)
    expected = pd.Series([pd.NA, 2], dtype="Int64")
    tm.assert_series_equal(result, expected)


def test_array_setitem():
    # GH 31446
    arr = pd.Series([1, 2], dtype="Int64").array
    arr[arr > 1] = 1

    expected = pd.array([1, 1], dtype="Int64")
    tm.assert_extension_array_equal(arr, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\index.py ===
"""
Tests for Indexes backed by arbitrary ExtensionArrays.
"""
import pandas as pd


class BaseIndexTests:
    """Tests for Index object backed by an ExtensionArray"""

    def test_index_from_array(self, data):
        idx = pd.Index(data)
        assert data.dtype == idx.dtype

    def test_index_from_listlike_with_dtype(self, data):
        idx = pd.Index(data, dtype=data.dtype)
        assert idx.dtype == data.dtype

        idx = pd.Index(list(data), dtype=data.dtype)
        assert idx.dtype == data.dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_type_D.py ===
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix



def test_type_D():
    c = CartanType("D4")
    m = Matrix(4, 4, [2, -1, 0, 0, -1, 2, -1, -1, 0, -1, 2, 0, 0, -1, 0, 2])
    assert c.cartan_matrix() == m
    assert c.basis() == 6
    assert c.lie_algebra() == "so(8)"
    assert c.roots() == 24
    assert c.simple_root(3) == [0, 0, 1, -1]
    diag = "    3\n    0\n    |\n    |\n0---0---0\n1   2   4"
    assert diag == c.dynkin_diagram()
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 1, 0, 0],
            3: [1, 0, -1, 0], 4: [1, 0, 1, 0], 5: [1, 0, 0, -1], 6: [1, 0, 0, 1],
            7: [0, 1, -1, 0], 8: [0, 1, 1, 0], 9: [0, 1, 0, -1], 10: [0, 1, 0, 1],
            11: [0, 0, 1, -1], 12: [0, 0, 1, 1]}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_quoted_character.py ===
"""See https://github.com/numpy/numpy/pull/10676.

"""
import sys

import pytest

from . import util


class TestQuotedCharacter(util.F2PyTest):
    sources = [util.getpath("tests", "src", "quoted_character", "foo.f")]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    @pytest.mark.slow
    def test_quoted_character(self):
        assert self.module.foo() == (b"'", b'"', b";", b"!", b"(", b")")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\test_numeric.py ===
import numpy as np
from numpy.testing import assert_equal


class TestDot:
    def test_matscalar(self):
        b1 = np.matrix(np.ones((3, 3), dtype=complex))
        assert_equal(b1 * 1.0, b1)


def test_diagonal():
    b1 = np.matrix([[1, 2], [3, 4]])
    diag_b1 = np.matrix([[1, 4]])
    array_b1 = np.array([1, 4])

    assert_equal(b1.diagonal(), diag_b1)
    assert_equal(np.diagonal(b1), array_b1)
    assert_equal(np.diag(b1), array_b1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\lib_version.py ===
from numpy.lib import NumpyVersion

version = NumpyVersion("1.8.0")

version.vstring
version.version
version.major
version.minor
version.bugfix
version.pre_release
version.is_devversion

version == version
version != version
version < "1.8.0"
version <= version
version > version
version >= "1.8.0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\construction\test_extract_array.py ===
from pandas import Index
import pandas._testing as tm
from pandas.core.construction import extract_array


def test_extract_array_rangeindex():
    ri = Index(range(5))

    expected = ri._values
    res = extract_array(ri, extract_numpy=True, extract_range=True)
    tm.assert_numpy_array_equal(res, expected)
    res = extract_array(ri, extract_numpy=False, extract_range=True)
    tm.assert_numpy_array_equal(res, expected)

    res = extract_array(ri, extract_numpy=True, extract_range=False)
    tm.assert_index_equal(res, ri)
    res = extract_array(ri, extract_numpy=False, extract_range=False)
    tm.assert_index_equal(res, ri)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_setops.py ===
import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    Index,
)
import pandas._testing as tm


@pytest.mark.parametrize("na_value", [None, np.nan])
def test_difference_with_na(na_value):
    # GH 57318
    ci = CategoricalIndex(["a", "b", "c", None])
    other = Index(["c", na_value])
    result = ci.difference(other)
    expected = CategoricalIndex(["a", "b"], categories=["a", "b", "c"])
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_to_series.py ===
import numpy as np

from pandas import (
    DatetimeIndex,
    Series,
)
import pandas._testing as tm


class TestToSeries:
    def test_to_series(self):
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")

        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        result = idx.to_series(index=[0, 1])
        assert expected.dtype == idx.dtype
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_insert.py ===
import numpy as np
import pytest

from pandas import (
    NaT,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestInsert:
    @pytest.mark.parametrize("na", [np.nan, NaT, None])
    def test_insert(self, na):
        # GH#18295 (test missing)
        expected = PeriodIndex(["2017Q1", NaT, "2017Q2", "2017Q3", "2017Q4"], freq="Q")
        result = period_range("2017Q1", periods=4, freq="Q").insert(1, na)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_root_system.py ===
from sympy.liealgebras.root_system import RootSystem
from sympy.liealgebras.type_a import TypeA
from sympy.matrices import Matrix

def test_root_system():
    c = RootSystem("A3")
    assert c.cartan_type == TypeA(3)
    assert c.simple_roots() ==  {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}
    assert c.root_space() == "alpha[1] + alpha[2] + alpha[3]"
    assert c.cartan_matrix() == Matrix([[ 2, -1,  0], [-1,  2, -1], [ 0, -1,  2]])
    assert c.dynkin_diagram() == "0---0---0\n1   2   3"
    assert c.add_simple_roots(1, 2) == [1, 0, -1, 0]
    assert c.all_roots() == {1: [1, -1, 0, 0], 2: [1, 0, -1, 0],
            3: [1, 0, 0, -1], 4: [0, 1, -1, 0], 5: [0, 1, 0, -1],
            6: [0, 0, 1, -1], 7: [-1, 1, 0, 0], 8: [-1, 0, 1, 0],
            9: [-1, 0, 0, 1], 10: [0, -1, 1, 0],
            11: [0, -1, 0, 1], 12: [0, 0, -1, 1]}
    assert c.add_as_roots([1, 0, -1, 0], [0, 0, 1, -1]) == [1, 0, 0, -1]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_gtk.py ===
from sympy.functions.elementary.trigonometric import sin
from sympy.printing.gtk import print_gtk
from sympy.testing.pytest import XFAIL, raises

# this test fails if python-lxml isn't installed. We don't want to depend on
# anything with SymPy


@XFAIL
def test_1():
    from sympy.abc import x
    print_gtk(x**2, start_viewer=False)
    print_gtk(x**2 + sin(x)/4, start_viewer=False)


def test_settings():
    from sympy.abc import x
    raises(TypeError, lambda: print_gtk(x, method="garbage"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_f2cmap.py ===
import numpy as np

from . import util


class TestF2Cmap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_gh15095(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\test_multiarray.py ===
import numpy as np
from numpy.testing import assert_, assert_array_equal, assert_equal


class TestView:
    def test_type(self):
        x = np.array([1, 2, 3])
        assert_(isinstance(x.view(np.matrix), np.matrix))

    def test_keywords(self):
        x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])
        # We must be specific about the endianness here:
        y = x.view(dtype='<i2', type=np.matrix)
        assert_array_equal(y, [[513]])

        assert_(isinstance(y, np.matrix))
        assert_equal(y.dtype, np.dtype('<i2'))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\numerictypes.py ===
import numpy as np

np.isdtype(np.float64, (np.int64, np.float64))
np.isdtype(np.int64, "signed integer")

np.issubdtype("S1", np.bytes_)
np.issubdtype(np.float64, np.float32)

np.ScalarType
np.ScalarType[0]
np.ScalarType[3]
np.ScalarType[8]
np.ScalarType[10]

np.typecodes["Character"]
np.typecodes["Complex"]
np.typecodes["All"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_combine.py ===
from pandas import Series
import pandas._testing as tm


class TestCombine:
    def test_combine_scalar(self):
        # GH#21248
        # Note - combine() with another Series is tested elsewhere because
        # it is used when testing operators
        ser = Series([i * 10 for i in range(5)])
        result = ser.combine(3, lambda x, y: x + y)
        expected = Series([i * 10 + 3 for i in range(5)])
        tm.assert_series_equal(result, expected)

        result = ser.combine(22, lambda x, y: min(x, y))
        expected = Series([min(i * 10, 22) for i in range(5)])
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_type_A.py ===
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

def test_type_A():
    c = CartanType("A3")
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -1, 0, -1, 2])
    assert m == c.cartan_matrix()
    assert c.basis() == 8
    assert c.roots() == 12
    assert c.dimension() == 4
    assert c.simple_root(1) == [1, -1, 0, 0]
    assert c.highest_root() == [1, 0, 0, -1]
    assert c.lie_algebra() == "su(4)"
    diag = "0---0---0\n1   2   3"
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() == {1: [1, -1, 0, 0], 2: [1, 0, -1, 0],
            3: [1, 0, 0, -1], 4: [0, 1, -1, 0], 5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_type_B.py ===
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

def test_type_B():
    c = CartanType("B3")
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -2, 0, -1, 2])
    assert m == c.cartan_matrix()
    assert c.dimension() == 3
    assert c.roots() == 18
    assert c.simple_root(3) == [0, 0, 1]
    assert c.basis() == 3
    assert c.lie_algebra() == "so(6)"
    diag = "0---0=>=0\n1   2   3"
    assert c.dynkin_diagram() == diag
    assert c.positive_roots() ==  {1: [1, -1, 0], 2: [1, 1, 0], 3: [1, 0, -1],
            4: [1, 0, 1], 5: [0, 1, -1], 6: [0, 1, 1], 7: [1, 0, 0],
            8: [0, 1, 0], 9: [0, 0, 1]}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\__init__.py ===
import pytest

from numpy.testing import IS_EDITABLE, IS_WASM

if IS_WASM:
    pytest.skip(
        "WASM/Pyodide does not use or support Fortran",
        allow_module_level=True
    )


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\ufuncs.py ===
import numpy as np

np.sin(1)
np.sin([1, 2, 3])
np.sin(1, out=np.empty(1))
np.matmul(np.ones((2, 2, 2)), np.ones((2, 2, 2)), axes=[(0, 1), (0, 1), (0, 1)])
np.sin(1, signature="D->D")
# NOTE: `np.generic` subclasses are not guaranteed to support addition;
# re-enable this we can infer the exact return type of `np.sin(...)`.
#
# np.sin(1) + np.sin(1)
np.sin.types[0]
np.sin.__name__
np.sin.__doc__

np.abs(np.array([1]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_pyinstaller\tests\__init__.py ===
import pytest

from numpy.testing import IS_EDITABLE, IS_WASM

if IS_WASM:
    pytest.skip(
        "WASM/Pyodide does not use or support Fortran",
        allow_module_level=True
    )


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_iterrows.py ===
from pandas import (
    DataFrame,
    Timedelta,
)


def test_no_overflow_of_freq_and_time_in_dataframe():
    # GH 35665
    df = DataFrame(
        {
            "some_string": ["2222Y3"],
            "time": [Timedelta("0 days 00:00:00.990000")],
        }
    )
    for _, row in df.iterrows():
        assert row.dtype == "object"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_type_G.py ===
# coding=utf-8
from sympy.liealgebras.cartan_type import CartanType
from sympy.matrices import Matrix

def test_type_G():
    c = CartanType("G2")
    m = Matrix(2, 2, [2, -1, -3, 2])
    assert c.cartan_matrix() == m
    assert c.simple_root(2) == [1, -2, 1]
    assert c.basis() == 14
    assert c.roots() == 12
    assert c.dimension() == 3
    diag = "0≡<≡0\n1   2"
    assert diag == c.dynkin_diagram()
    assert c.positive_roots() == {1: [0, 1, -1], 2: [1, -2, 1], 3: [1, -1, 0],
            4: [1, 0, 1], 5: [1, 1, -2], 6: [2, -1, -1]}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_latex_deps.py ===
from sympy.external import import_module
from sympy.testing.pytest import ignore_warnings, raises

antlr4 = import_module("antlr4", warn_not_installed=False)

# disable tests if antlr4-python3-runtime is not present
if antlr4:
    disabled = True


def test_no_import():
    from sympy.parsing.latex import parse_latex

    with ignore_warnings(UserWarning):
        with raises(ImportError):
            parse_latex('1 + 1')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_value_attrspec.py ===
import pytest

from . import util


class TestValueAttr(util.F2PyTest):
    sources = [util.getpath("tests", "src", "value_attrspec", "gh21665.f90")]

    # gh-21665
    @pytest.mark.slow
    def test_gh21665(self):
        inp = 2
        out = self.module.fortfuncs.square(inp)
        exp_out = 4
        assert out == exp_out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\object\test_astype.py ===
import pytest

from pandas import (
    Index,
    NaT,
)


def test_astype_invalid_nas_to_tdt64_raises():
    # GH#45722 don't cast np.datetime64 NaTs to timedelta64 NaT
    idx = Index([NaT.asm8] * 2, dtype=object)

    msg = r"Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
    with pytest.raises(TypeError, match=msg):
        idx.astype("m8[ns]")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_group_constructs.py ===
from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.named_groups import CyclicGroup, DihedralGroup


def test_direct_product_n():
    C = CyclicGroup(4)
    D = DihedralGroup(4)
    G = DirectProduct(C, C, C)
    assert G.order() == 64
    assert G.degree == 12
    assert len(G.orbits()) == 3
    assert G.is_abelian is True
    H = DirectProduct(D, C)
    assert H.order() == 32
    assert H.is_abelian is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_str.py ===
from mpmath import nstr, matrix, inf

def test_nstr():
    m = matrix([[0.75, 0.190940654, -0.0299195971],
                [0.190940654, 0.65625, 0.205663228],
                [-0.0299195971, 0.205663228, 0.64453125e-20]])
    assert nstr(m, 4, min_fixed=-inf) == \
    '''[    0.75  0.1909                    -0.02992]
[  0.1909  0.6563                      0.2057]
[-0.02992  0.2057  0.000000000000000000006445]'''
    assert nstr(m, 4) == \
    '''[    0.75  0.1909   -0.02992]
[  0.1909  0.6563     0.2057]
[-0.02992  0.2057  6.445e-21]'''

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_util.py ===
import numpy as np

from pandas import DataFrame
from pandas.tests.copy_view.util import get_array


def test_get_array_numpy():
    df = DataFrame({"a": [1, 2, 3]})
    assert np.shares_memory(get_array(df, "a"), get_array(df, "a"))


def test_get_array_masked():
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    assert np.shares_memory(get_array(df, "a"), get_array(df, "a"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_dict_compat.py ===
import numpy as np

from pandas.core.dtypes.cast import dict_compat

from pandas import Timestamp


def test_dict_compat():
    data_datetime64 = {np.datetime64("1990-03-15"): 1, np.datetime64("2015-03-15"): 2}
    data_unchanged = {1: 2, 3: 4, 5: 6}
    expected = {Timestamp("1990-3-15"): 1, Timestamp("2015-03-15"): 2}
    assert dict_compat(data_datetime64) == expected
    assert dict_compat(expected) == expected
    assert dict_compat(data_unchanged) == data_unchanged

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_ops.py ===
from pandas import (
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexOps:
    def test_infer_freq(self, freq_sample):
        # GH#11018
        idx = timedelta_range("1", freq=freq_sample, periods=10)
        result = TimedeltaIndex(idx.asi8, freq="infer")
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_gbq.py ===
import pandas as pd
import pandas._testing as tm


def test_read_gbq_deprecated():
    with tm.assert_produces_warning(FutureWarning):
        with tm.external_error_raised(Exception):
            pd.read_gbq("fake")


def test_to_gbq_deprecated():
    with tm.assert_produces_warning(FutureWarning):
        with tm.external_error_raised(Exception):
            pd.DataFrame(range(1)).to_gbq("fake")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_pytables_missing.py ===
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


@td.skip_if_installed("tables")
def test_pytables_raises():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(ImportError, match="tables"):
        with tm.ensure_clean("foo.h5") as path:
            df.to_hdf(path, key="df")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_abstract_nodes.py ===
from sympy.core.symbol import symbols
from sympy.codegen.abstract_nodes import List


def test_List():
    l = List(2, 3, 4)
    assert l == List(2, 3, 4)
    assert str(l) == "[2, 3, 4]"
    x, y, z = symbols('x y z')
    l = List(x**2,y**3,z**4)
    # contrary to python's built-in list, we can call e.g. "replace" on List.
    m = l.replace(lambda arg: arg.is_Pow and arg.exp>2, lambda p: p.base-p.exp)
    assert m == [x**2, y-3, z-4]
    hash(m)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_cxxnodes.py ===
from sympy.core.symbol import Symbol
from sympy.codegen.ast import Type
from sympy.codegen.cxxnodes import using
from sympy.printing.codeprinter import cxxcode

x = Symbol('x')

def test_using():
    v = Type('std::vector')
    u1 = using(v)
    assert cxxcode(u1) == 'using std::vector'

    u2 = using(v, 'vec')
    assert cxxcode(u2) == 'using vec = std::vector'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_rules.py ===
from sympy.core.rules import Transform

from sympy.testing.pytest import raises


def test_Transform():
    add1 = Transform(lambda x: x + 1, lambda x: x % 2 == 1)
    assert add1[1] == 2
    assert (1 in add1) is True
    assert add1.get(1) == 2

    raises(KeyError, lambda: add1[2])
    assert (2 in add1) is False
    assert add1.get(2) is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_indexing.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("na", [None, np.nan, pd.NA])
def test_setitem_missing_values(na):
    arr = pd.array([True, False, None], dtype="boolean")
    expected = pd.array([True, None, None], dtype="boolean")
    arr[1] = na
    tm.assert_extension_array_equal(arr, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_repr.py ===
import pandas as pd


def test_repr():
    df = pd.DataFrame({"A": pd.array([True, False, None], dtype="boolean")})
    expected = "       A\n0   True\n1  False\n2   <NA>"
    assert repr(df) == expected

    expected = "0     True\n1    False\n2     <NA>\nName: A, dtype: boolean"
    assert repr(df.A) == expected

    expected = "<BooleanArray>\n[True, False, <NA>]\nLength: 3, dtype: boolean"
    assert repr(df.A.array) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\interval\test_formats.py ===
from pandas.core.arrays import IntervalArray


def test_repr():
    # GH#25022
    arr = IntervalArray.from_tuples([(0, 1), (1, 2)])
    result = repr(arr)
    expected = (
        "<IntervalArray>\n"
        "[(0, 1], (1, 2]]\n"
        "Length: 2, dtype: interval[int64, right]"
    )
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\test_where.py ===
import numpy as np

from pandas import Index
import pandas._testing as tm


class TestWhere:
    def test_where_intlike_str_doesnt_cast_ints(self):
        idx = Index(range(3))
        mask = np.array([True, False, True])
        res = idx.where(mask, "2")
        expected = Index([0, "2", 2])
        tm.assert_index_equal(res, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_npfuncs.py ===
import numpy as np

from pandas import date_range
import pandas._testing as tm


class TestSplit:
    def test_split_non_utc(self):
        # GH#14042
        indices = date_range("2016-01-01 00:00:00+0200", freq="s", periods=10)
        result = np.split(indices, indices_or_sections=[])[0]
        expected = indices._with_freq(None)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_pickle.py ===
import pytest

from pandas import IntervalIndex
import pandas._testing as tm


class TestPickle:
    @pytest.mark.parametrize("closed", ["left", "right", "both"])
    def test_pickle_round_trip_closed(self, closed):
        # https://github.com/pandas-dev/pandas/issues/35658
        idx = IntervalIndex.from_tuples([(1, 2), (2, 3)], closed=closed)
        result = tm.round_trip_pickle(idx)
        tm.assert_index_equal(result, idx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_pop.py ===
from pandas import Series
import pandas._testing as tm


def test_pop():
    # GH#6600
    ser = Series([0, 4, 0], index=["A", "B", "C"], name=4)

    result = ser.pop("B")
    assert result == 4

    expected = Series([0, 0], index=["A", "C"], name=4)
    tm.assert_series_equal(ser, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_pynodes.py ===
from sympy.core.symbol import symbols
from sympy.codegen.pynodes import List


def test_List():
    l = List(2, 3, 4)
    assert l == List(2, 3, 4)
    assert str(l) == "[2, 3, 4]"
    x, y, z = symbols('x y z')
    l = List(x**2,y**3,z**4)
    # contrary to python's built-in list, we can call e.g. "replace" on List.
    m = l.replace(lambda arg: arg.is_Pow and arg.exp>2, lambda p: p.base-p.exp)
    assert m == [x**2, y-3, z-4]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_lineintegrals.py ===
from sympy.core.numbers import E
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry.curve import Curve
from sympy.integrals.integrals import line_integrate

s, t, x, y, z = symbols('s,t,x,y,z')


def test_lineintegral():
    c = Curve([E**t + 1, E**t - 1], (t, 0, log(2)))
    assert line_integrate(x + y, c, [x, y]) == 3*sqrt(2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_constants.py ===
from sympy.core.numbers import Float

from sympy.physics.quantum.constants import hbar


def test_hbar():
    assert hbar.is_commutative is True
    assert hbar.is_real is True
    assert hbar.is_positive is True
    assert hbar.is_negative is False
    assert hbar.is_irrational is True

    assert hbar.evalf() == Float(1.05457162e-34)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\test_printing.py ===
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
from sympy import I

def test_printing_TensMul():
    R3 = TensorIndexType('R3', dim=3)
    p, q = tensor_indices("p q", R3)
    K = TensorHead("K", [R3])

    assert repr(2*K(p)) == "2*K(p)"
    assert repr(-K(p)) == "-K(p)"
    assert repr(-2*K(p)*K(q)) == "-2*K(p)*K(q)"
    assert repr(-I*K(p)) == "-I*K(p)"
    assert repr(I*K(p)) == "I*K(p)"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_deprecated.py ===
from sympy.testing.pytest import warns_deprecated_sympy

# See https://github.com/sympy/sympy/pull/18095

def test_deprecated_utilities():
    with warns_deprecated_sympy():
        import sympy.utilities.pytest  # noqa:F401
    with warns_deprecated_sympy():
        import sympy.utilities.runtests  # noqa:F401
    with warns_deprecated_sympy():
        import sympy.utilities.randtest  # noqa:F401
    with warns_deprecated_sympy():
        import sympy.utilities.tmpfiles  # noqa:F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_contains.py ===
import numpy as np

import pandas as pd


def test_contains_nan():
    # GH#52840
    arr = pd.array(range(5)) / 0

    assert np.isnan(arr._data[0])
    assert not arr.isna()[0]
    assert np.nan in arr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_numba.py ===
import pytest

import pandas.util._test_decorators as td

from pandas import option_context


@td.skip_if_installed("numba")
def test_numba_not_installed_option_context():
    with pytest.raises(ImportError, match="Missing optional"):
        with option_context("compute.use_numba", True):
            pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_gmpy.py ===
from sympy.external.gmpy import LONG_MAX, iroot
from sympy.testing.pytest import raises


def test_iroot():
    assert iroot(2, LONG_MAX) == (1, False)
    assert iroot(2, LONG_MAX + 1) == (1, False)
    for x in range(3):
        assert iroot(x, 1) == (x, True)
    raises(ValueError, lambda: iroot(-1, 1))
    raises(ValueError, lambda: iroot(0, 0))
    raises(ValueError, lambda: iroot(0, -1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_cartan_type.py ===
from sympy.liealgebras.cartan_type import CartanType, Standard_Cartan

def test_Standard_Cartan():
    c = CartanType("A4")
    assert c.rank() == 4
    assert c.series == "A"
    m = Standard_Cartan("A", 2)
    assert m.rank() == 2
    assert m.series == "A"
    b = CartanType("B12")
    assert b.rank() == 12
    assert b.series == "B"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_exceptions.py ===
from sympy.testing.pytest import raises
from sympy.utilities.exceptions import sympy_deprecation_warning

# Only test exceptions here because the other cases are tested in the
# warns_deprecated_sympy tests
def test_sympy_deprecation_warning():
    raises(TypeError, lambda: sympy_deprecation_warning('test',
                                                        deprecated_since_version=1.10,
                                                        active_deprecations_target='active-deprecations'))

    raises(ValueError, lambda: sympy_deprecation_warning('test',
                                                            deprecated_since_version="1.10", active_deprecations_target='(active-deprecations)='))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\test_pickle.py ===
from pandas import Index
import pandas._testing as tm


def test_pickle_preserves_object_dtype():
    # GH#43188, GH#43155 don't infer numeric dtype
    index = Index([1, 2, 3], dtype=object)

    result = tm.round_trip_pickle(index)
    assert result.dtype == object
    tm.assert_index_equal(index, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_pickle.py ===
from pandas import timedelta_range
import pandas._testing as tm


class TestPickle:
    def test_pickle_after_set_freq(self):
        tdi = timedelta_range("1 day", periods=4, freq="s")
        tdi = tdi._with_freq(None)

        res = tm.round_trip_pickle(tdi)
        tm.assert_index_equal(res, tdi)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\interval\test_formats.py ===
from pandas import Interval


def test_interval_repr():
    interval = Interval(0, 1)
    assert repr(interval) == "Interval(0, 1, closed='right')"
    assert str(interval) == "(0, 1]"

    interval_left = Interval(0, 1, closed="left")
    assert repr(interval_left) == "Interval(0, 1, closed='left')"
    assert str(interval_left) == "[0, 1)"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_source.py ===
from sympy.utilities.source import get_mod_func, get_class


def test_get_mod_func():
    assert get_mod_func(
        'sympy.core.basic.Basic') == ('sympy.core.basic', 'Basic')


def test_get_class():
    _basic = get_class('sympy.core.basic.Basic')
    assert _basic.__name__ == 'Basic'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test__all__.py ===

import collections

import numpy as np


def test_no_duplicates_in_np__all__():
    # Regression test for gh-10198.
    dups = {k: v for k, v in collections.Counter(np.__all__).items() if v > 1}
    assert len(dups) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_pickle.py ===
import pytest

from pandas import MultiIndex


def test_pickle_compat_construction():
    # this is testing for pickle compat
    # need an object to create with
    with pytest.raises(TypeError, match="Must pass both levels and codes"):
        MultiIndex()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\interactive\tests\test_interactive.py ===
from sympy.interactive.session import int_to_Integer


def test_int_to_Integer():
    assert int_to_Integer("1 + 2.2 + 0x3 + 40") == \
        'Integer (1 )+2.2 +Integer (0x3 )+Integer (40 )'
    assert int_to_Integer("0b101") == 'Integer (0b101 )'
    assert int_to_Integer("ab1 + 1 + '1 + 2'") == "ab1 +Integer (1 )+'1 + 2'"
    assert int_to_Integer("(2 + \n3)") == '(Integer (2 )+\nInteger (3 ))'
    assert int_to_Integer("2 + 2.0 + 2j + 2e-10") == 'Integer (2 )+2.0 +2j +2e-10 '

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_cartan_matrix.py ===
from sympy.liealgebras.cartan_matrix import CartanMatrix
from sympy.matrices import Matrix

def test_CartanMatrix():
    c = CartanMatrix("A3")
    m = Matrix(3, 3, [2, -1, 0, -1, 2, -1, 0, -1, 2])
    assert c == m
    a = CartanMatrix(["G",2])
    mt = Matrix(2, 2, [2, -1, -3, 2])
    assert a == mt

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_timeutils.py ===
"""Tests for simple tools for timing functions' execution. """

from sympy.utilities.timeutils import timed

def test_timed():
    result = timed(lambda: 1 + 1, limit=100000)
    assert result[0] == 100000 and result[3] == "ns", str(result)

    result = timed("1 + 1", limit=100000)
    assert result[0] == 100000 and result[3] == "ns"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\common.py ===
from typing import Any

from pandas import Index


def allow_na_ops(obj: Any) -> bool:
    """Whether to skip test cases including NaN"""
    is_bool_index = isinstance(obj, Index) and obj.inferred_type == "boolean"
    return not is_bool_index and obj._can_hold_na

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\conftest.py ===
import pytest


@pytest.fixture(params=["split", "records", "index", "columns", "values"])
def orient(request):
    """
    Fixture for orients excluding the table format.
    """
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\conftest.py ===
import uuid

import pytest


@pytest.fixture
def setup_path():
    """Fixture for setup path"""
    return f"tmp.__{uuid.uuid4()}__.h5"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\accessors\test_sparse_accessor.py ===
from pandas import Series


class TestSparseAccessor:
    def test_sparse_accessor_updates_on_inplace(self):
        ser = Series([1, 1, 2, 3], dtype="Sparse[int]")
        return_value = ser.drop([0, 1], inplace=True)
        assert return_value is None
        assert ser.sparse.density == 1.0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_dynkin_diagram.py ===
from sympy.liealgebras.dynkin_diagram import DynkinDiagram

def test_DynkinDiagram():
    c = DynkinDiagram("A3")
    diag = "0---0---0\n1   2   3"
    assert c == diag
    ct = DynkinDiagram(["B", 3])
    diag2 = "0---0=>=0\n1   2   3"
    assert ct == diag2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\decimal\__init__.py ===
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)

__all__ = ["DecimalArray", "DecimalDtype", "to_decimal", "make_data"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_head_tail.py ===
import pandas._testing as tm


def test_head_tail(string_series):
    tm.assert_series_equal(string_series.head(), string_series[:5])
    tm.assert_series_equal(string_series.head(0), string_series[0:0])
    tm.assert_series_equal(string_series.tail(), string_series[-5:])
    tm.assert_series_equal(string_series.tail(0), string_series[0:0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_mpmath.py ===
from mpmath.libmp import *
from mpmath import *

def test_newstyle_classes():
    for cls in [mp, fp, iv, mpf, mpc]:
        for s in cls.__class__.__mro__:
            assert isinstance(s, type)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\common.py ===
from pandas.core.groupby.base import transformation_kernels

# There is no Series.cumcount or DataFrame.cumcount
series_transform_kernels = [
    x for x in sorted(transformation_kernels) if x != "cumcount"
]
frame_transform_kernels = [x for x in sorted(transformation_kernels) if x != "cumcount"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\json\__init__.py ===
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)

__all__ = ["JSONArray", "JSONDtype", "make_data"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\list\__init__.py ===
from pandas.tests.extension.list.array import (
    ListArray,
    ListDtype,
    make_data,
)

__all__ = ["ListArray", "ListDtype", "make_data"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\__init__.py ===
"""
Test files dedicated to individual (stand-alone) DataFrame methods

Ideally these files/tests should correspond 1-to-1 with tests.series.methods

These may also present opportunities for sharing/de-duplicating test code.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\conftest.py ===
import pytest


@pytest.fixture(params=[True, False])
def sort(request):
    """Boolean sort keyword for concat and DataFrame.append."""
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_dtypes.py ===
import numpy as np


class TestSeriesDtypes:
    def test_dtype(self, datetime_series):
        assert datetime_series.dtype == np.dtype("float64")
        assert datetime_series.dtypes == np.dtype("float64")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\__init__.py ===
"""
Test files dedicated to individual (stand-alone) Series methods

Ideally these files/tests should correspond 1-to-1 with tests.frame.methods

These may also present opportunities for sharing/de-duplicating test code.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_pyutils.py ===
from sympy.codegen.ast import Print
from sympy.codegen.pyutils import render_as_module

def test_standard():
    ast = Print('x y'.split(), r"coordinate: %12.5g %12.5g\n")
    assert render_as_module(ast, standard='python3') == \
        '\n\nprint("coordinate: %12.5g %12.5g\\n" % (x, y), end="")'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\simple_py3.py ===
import numpy as np

array = np.array([1, 2])

# The @ operator is not in python 2
array @ array

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\warnings_and_errors.py ===
import numpy.exceptions as ex

ex.AxisError("test")
ex.AxisError(1, ndim=2)
ex.AxisError(1, ndim=2, msg_prefix="error")
ex.AxisError(1, ndim=2, msg_prefix=None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\array_with_attr\__init__.py ===
from pandas.tests.extension.array_with_attr.array import (
    FloatAttrArray,
    FloatAttrDtype,
)

__all__ = ["FloatAttrArray", "FloatAttrDtype"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\date\__init__.py ===
from pandas.tests.extension.date.array import (
    DateArray,
    DateDtype,
)

__all__ = ["DateArray", "DateDtype"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_compatibility.py ===
from sympy.testing.pytest import warns_deprecated_sympy

def test_compatibility_submodule():
    # Test the sympy.core.compatibility deprecation warning
    with warns_deprecated_sympy():
        import sympy.core.compatibility # noqa:F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_method.py ===
from sympy.physics.mechanics.method import _Methods
from sympy.testing.pytest import raises

def test_method():
    raises(TypeError, lambda: _Methods())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\tests\test_deprecated.py ===
from sympy.testing.pytest import warns_deprecated_sympy

def test_deprecated_testing_randtest():
    with warns_deprecated_sympy():
        import sympy.testing.randtest  # noqa:F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\nditer.py ===
import numpy as np

arr = np.array([1])
np.nditer([arr, None])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reductions\__init__.py ===
"""
Tests for reductions where we want to test for matching behavior across
Array, Index, Series, and DataFrame methods.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_xxe.py ===
# A test file for XXE injection
# Username: Test
# Password: Test

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\base.py ===
class BaseExtensionTests:
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\tests\__init__.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\fft\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\linalg\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\data\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\testing\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\api\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\datetimes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\interval\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\masked\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\numpy_\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\period\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\string_\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\timedeltas\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\computation\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\config\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\construction\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\index\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\constructors\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\aggregate\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\transform\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\numeric\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\object\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\ranges\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\string\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\methods\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\interval\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\interchange\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\internals\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\excel\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\dtypes\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\usecols\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\sas\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\xml\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\libs\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\frame\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\resample\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\merge\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\interval\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\period\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\methods\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\accessors\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tools\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\frequencies\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\holiday\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\moments\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\algebras\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\categories\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\crypto\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\diffgeom\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\discrete\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\combinatorial\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\interactive\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\biomechanics\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\control\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\hep\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\pretty\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sandbox\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\diophantine\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\sampling\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\branch\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\unify\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\_compilation\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\__init__.py ===