
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\cachecontrol\caches\__init__.py ===
# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0

from pip._vendor.cachecontrol.caches.file_cache import FileCache, SeparateBodyFileCache
from pip._vendor.cachecontrol.caches.redis_cache import RedisCache

__all__ = ["FileCache", "SeparateBodyFileCache", "RedisCache"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\tomli\__init__.py ===
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Taneli Hukkinen
# Licensed to PSF under a Contributor Agreement.

__all__ = ("loads", "load", "TOMLDecodeError")
__version__ = "2.2.1"  # DO NOT EDIT THIS LINE MANUALLY. LET bump2version UTILITY DO IT

from ._parser import TOMLDecodeError, load, loads

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\cutils.py ===
from sympy.printing.c import C99CodePrinter

def render_as_source_file(content, Printer=C99CodePrinter, settings=None):
    """ Renders a C source file (with required #include statements) """
    printer = Printer(settings or {})
    code_str = printer.doprint(content)
    includes = '\n'.join(['#include <%s>' % h for h in printer.headers])
    return includes + '\n\n' + code_str

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\__init__.py ===
from .products import product, Product
from .summations import summation, Sum

__all__ = [
    'product', 'Product',

    'summation', 'Sum',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\benchmarks\bench_special.py ===
from sympy.core.symbol import symbols
from sympy.functions.special.spherical_harmonics import Ynm

x, y = symbols('x,y')


def timeit_Ynm_xy():
    Ynm(1, 1, x, y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\interactive\__init__.py ===
"""Helper module for setting up interactive SymPy sessions. """

from .printing import init_printing
from .session import init_session
from .traversal import interactive_traversal


__all__ = ['init_printing', 'init_session', 'interactive_traversal']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_mathematica.py ===
from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer, Tuple,
                        Derivative, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.integrals import Integral
from sympy.concrete import Sum
from sympy.functions import (exp, sin, cos, fresnelc, fresnels, conjugate, Max,
                             Min, gamma, polygamma, loggamma, erf, erfi, erfc,
                             erf2, expint, erfinv, erfcinv, Ei, Si, Ci, li,
                             Shi, Chi, uppergamma, beta, subfactorial, erf2inv,
                             factorial, factorial2, catalan, RisingFactorial,
                             FallingFactorial, harmonic, atan2, sec, acsc,
                             hermite, laguerre, assoc_laguerre, jacobi,
                             gegenbauer, chebyshevt, chebyshevu, legendre,
                             assoc_legendre, Li, LambertW)

from sympy.printing.mathematica import mathematica_code as mcode

x, y, z, w = symbols('x,y,z,w')
f = Function('f')


def test_Integer():
    assert mcode(Integer(67)) == "67"
    assert mcode(Integer(-1)) == "-1"


def test_Rational():
    assert mcode(Rational(3, 7)) == "3/7"
    assert mcode(Rational(18, 9)) == "2"
    assert mcode(Rational(3, -7)) == "-3/7"
    assert mcode(Rational(-3, -7)) == "3/7"
    assert mcode(x + Rational(3, 7)) == "x + 3/7"
    assert mcode(Rational(3, 7)*x) == "(3/7)*x"


def test_Relational():
    assert mcode(Eq(x, y)) == "x == y"
    assert mcode(Ne(x, y)) == "x != y"
    assert mcode(Le(x, y)) == "x <= y"
    assert mcode(Lt(x, y)) == "x < y"
    assert mcode(Gt(x, y)) == "x > y"
    assert mcode(Ge(x, y)) == "x >= y"


def test_Function():
    assert mcode(f(x, y, z)) == "f[x, y, z]"
    assert mcode(sin(x) ** cos(x)) == "Sin[x]^Cos[x]"
    assert mcode(sec(x) * acsc(x)) == "ArcCsc[x]*Sec[x]"
    assert mcode(atan2(y, x)) == "ArcTan[x, y]"
    assert mcode(conjugate(x)) == "Conjugate[x]"
    assert mcode(Max(x, y, z)*Min(y, z)) == "Max[x, y, z]*Min[y, z]"
    assert mcode(fresnelc(x)) == "FresnelC[x]"
    assert mcode(fresnels(x)) == "FresnelS[x]"
    assert mcode(gamma(x)) == "Gamma[x]"
    assert mcode(uppergamma(x, y)) == "Gamma[x, y]"
    assert mcode(polygamma(x, y)) == "PolyGamma[x, y]"
    assert mcode(loggamma(x)) == "LogGamma[x]"
    assert mcode(erf(x)) == "Erf[x]"
    assert mcode(erfc(x)) == "Erfc[x]"
    assert mcode(erfi(x)) == "Erfi[x]"
    assert mcode(erf2(x, y)) == "Erf[x, y]"
    assert mcode(expint(x, y)) == "ExpIntegralE[x, y]"
    assert mcode(erfcinv(x)) == "InverseErfc[x]"
    assert mcode(erfinv(x)) == "InverseErf[x]"
    assert mcode(erf2inv(x, y)) == "InverseErf[x, y]"
    assert mcode(Ei(x)) == "ExpIntegralEi[x]"
    assert mcode(Ci(x)) == "CosIntegral[x]"
    assert mcode(li(x)) == "LogIntegral[x]"
    assert mcode(Si(x)) == "SinIntegral[x]"
    assert mcode(Shi(x)) == "SinhIntegral[x]"
    assert mcode(Chi(x)) == "CoshIntegral[x]"
    assert mcode(beta(x, y)) == "Beta[x, y]"
    assert mcode(factorial(x)) == "Factorial[x]"
    assert mcode(factorial2(x)) == "Factorial2[x]"
    assert mcode(subfactorial(x)) == "Subfactorial[x]"
    assert mcode(FallingFactorial(x, y)) == "FactorialPower[x, y]"
    assert mcode(RisingFactorial(x, y)) == "Pochhammer[x, y]"
    assert mcode(catalan(x)) == "CatalanNumber[x]"
    assert mcode(harmonic(x)) == "HarmonicNumber[x]"
    assert mcode(harmonic(x, y)) == "HarmonicNumber[x, y]"
    assert mcode(Li(x)) == "LogIntegral[x] - LogIntegral[2]"
    assert mcode(LambertW(x)) == "ProductLog[x]"
    assert mcode(LambertW(x, -1)) == "ProductLog[-1, x]"
    assert mcode(LambertW(x, y)) == "ProductLog[y, x]"


def test_special_polynomials():
    assert mcode(hermite(x, y)) == "HermiteH[x, y]"
    assert mcode(laguerre(x, y)) == "LaguerreL[x, y]"
    assert mcode(assoc_laguerre(x, y, z)) == "LaguerreL[x, y, z]"
    assert mcode(jacobi(x, y, z, w)) == "JacobiP[x, y, z, w]"
    assert mcode(gegenbauer(x, y, z)) == "GegenbauerC[x, y, z]"
    assert mcode(chebyshevt(x, y)) == "ChebyshevT[x, y]"
    assert mcode(chebyshevu(x, y)) == "ChebyshevU[x, y]"
    assert mcode(legendre(x, y)) == "LegendreP[x, y]"
    assert mcode(assoc_legendre(x, y, z)) == "LegendreP[x, y, z]"


def test_Pow():
    assert mcode(x**3) == "x^3"
    assert mcode(x**(y**3)) == "x^(y^3)"
    assert mcode(1/(f(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*f[x])^(-x + y^x)/(x^2 + y)"
    assert mcode(x**-1.0) == 'x^(-1.0)'
    assert mcode(x**Rational(2, 3)) == 'x^(2/3)'


def test_Mul():
    A, B, C, D = symbols('A B C D', commutative=False)
    assert mcode(x*y*z) == "x*y*z"
    assert mcode(x*y*A) == "x*y*A"
    assert mcode(x*y*A*B) == "x*y*A**B"
    assert mcode(x*y*A*B*C) == "x*y*A**B**C"
    assert mcode(x*A*B*(C + D)*A*y) == "x*y*A**B**(C + D)**A"


def test_constants():
    assert mcode(S.Zero) == "0"
    assert mcode(S.One) == "1"
    assert mcode(S.NegativeOne) == "-1"
    assert mcode(S.Half) == "1/2"
    assert mcode(S.ImaginaryUnit) == "I"

    assert mcode(oo) == "Infinity"
    assert mcode(S.NegativeInfinity) == "-Infinity"
    assert mcode(S.ComplexInfinity) == "ComplexInfinity"
    assert mcode(S.NaN) == "Indeterminate"

    assert mcode(S.Exp1) == "E"
    assert mcode(pi) == "Pi"
    assert mcode(S.GoldenRatio) == "GoldenRatio"
    assert mcode(S.TribonacciConstant) == \
        "(1/3 + (1/3)*(19 - 3*33^(1/2))^(1/3) + " \
        "(1/3)*(3*33^(1/2) + 19)^(1/3))"
    assert mcode(2*S.TribonacciConstant) == \
        "2*(1/3 + (1/3)*(19 - 3*33^(1/2))^(1/3) + " \
        "(1/3)*(3*33^(1/2) + 19)^(1/3))"
    assert mcode(S.EulerGamma) == "EulerGamma"
    assert mcode(S.Catalan) == "Catalan"


def test_containers():
    assert mcode([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
        "{1, 2, 3, {4, 5, {6, 7}}, 8, {9, 10}, 11}"
    assert mcode((1, 2, (3, 4))) == "{1, 2, {3, 4}}"
    assert mcode([1]) == "{1}"
    assert mcode((1,)) == "{1}"
    assert mcode(Tuple(*[1, 2, 3])) == "{1, 2, 3}"


def test_matrices():
    from sympy.matrices import MutableDenseMatrix, MutableSparseMatrix, \
        ImmutableDenseMatrix, ImmutableSparseMatrix
    A = MutableDenseMatrix(
        [[1, -1, 0, 0],
         [0, 1, -1, 0],
         [0, 0, 1, -1],
         [0, 0, 0, 1]]
    )
    B = MutableSparseMatrix(A)
    C = ImmutableDenseMatrix(A)
    D = ImmutableSparseMatrix(A)

    assert mcode(C) == mcode(A) == \
        "{{1, -1, 0, 0}, " \
        "{0, 1, -1, 0}, " \
        "{0, 0, 1, -1}, " \
        "{0, 0, 0, 1}}"

    assert mcode(D) == mcode(B) == \
        "SparseArray[{" \
        "{1, 1} -> 1, {1, 2} -> -1, {2, 2} -> 1, {2, 3} -> -1, " \
        "{3, 3} -> 1, {3, 4} -> -1, {4, 4} -> 1" \
        "}, {4, 4}]"

    # Trivial cases of matrices
    assert mcode(MutableDenseMatrix(0, 0, [])) == '{}'
    assert mcode(MutableSparseMatrix(0, 0, [])) == 'SparseArray[{}, {0, 0}]'
    assert mcode(MutableDenseMatrix(0, 3, [])) == '{}'
    assert mcode(MutableSparseMatrix(0, 3, [])) == 'SparseArray[{}, {0, 3}]'
    assert mcode(MutableDenseMatrix(3, 0, [])) == '{{}, {}, {}}'
    assert mcode(MutableSparseMatrix(3, 0, [])) == 'SparseArray[{}, {3, 0}]'

def test_NDArray():
    from sympy.tensor.array import (
        MutableDenseNDimArray, ImmutableDenseNDimArray,
        MutableSparseNDimArray, ImmutableSparseNDimArray)

    example = MutableDenseNDimArray(
        [[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12]],
         [[13, 14, 15, 16],
          [17, 18, 19, 20],
          [21, 22, 23, 24]]]
    )

    assert mcode(example) == \
    "{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, " \
    "{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}"

    example = ImmutableDenseNDimArray(example)

    assert mcode(example) == \
    "{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, " \
    "{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}"

    example = MutableSparseNDimArray(example)

    assert mcode(example) == \
    "SparseArray[{" \
        "{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, " \
        "{1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, " \
        "{1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, " \
        "{1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, " \
        "{2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, " \
        "{2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, " \
        "{2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, " \
        "{2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24" \
        "}, {2, 3, 4}]"

    example = ImmutableSparseNDimArray(example)

    assert mcode(example) == \
    "SparseArray[{" \
        "{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, " \
        "{1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, " \
        "{1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, " \
        "{1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, " \
        "{2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, " \
        "{2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, " \
        "{2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, " \
        "{2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24" \
        "}, {2, 3, 4}]"


def test_Integral():
    assert mcode(Integral(sin(sin(x)), x)) == "Hold[Integrate[Sin[Sin[x]], x]]"
    assert mcode(Integral(exp(-x**2 - y**2),
                          (x, -oo, oo),
                          (y, -oo, oo))) == \
        "Hold[Integrate[Exp[-x^2 - y^2], {x, -Infinity, Infinity}, " \
        "{y, -Infinity, Infinity}]]"


def test_Derivative():
    assert mcode(Derivative(sin(x), x)) == "Hold[D[Sin[x], x]]"
    assert mcode(Derivative(x, x)) == "Hold[D[x, x]]"
    assert mcode(Derivative(sin(x)*y**4, x, 2)) == "Hold[D[y^4*Sin[x], {x, 2}]]"
    assert mcode(Derivative(sin(x)*y**4, x, y, x)) == "Hold[D[y^4*Sin[x], x, y, x]]"
    assert mcode(Derivative(sin(x)*y**4, x, y, 3, x)) == "Hold[D[y^4*Sin[x], x, {y, 3}, x]]"


def test_Sum():
    assert mcode(Sum(sin(x), (x, 0, 10))) == "Hold[Sum[Sin[x], {x, 0, 10}]]"
    assert mcode(Sum(exp(-x**2 - y**2),
                     (x, -oo, oo),
                     (y, -oo, oo))) == \
        "Hold[Sum[Exp[-x^2 - y^2], {x, -Infinity, Infinity}, " \
        "{y, -Infinity, Infinity}]]"


def test_comment():
    from sympy.printing.mathematica import MCodePrinter
    assert MCodePrinter()._get_comment("Hello World") == \
        "(* Hello World *)"


def test_userfuncs():
    # Dictionary mutation test
    some_function = symbols("some_function", cls=Function)
    my_user_functions = {"some_function": "SomeFunction"}
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeFunction[z]'
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeFunction[z]'

    # List argument test
    my_user_functions = \
        {"some_function": [(lambda x: True, "SomeOtherFunction")]}
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeOtherFunction[z]'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sandbox\__init__.py ===
"""
Sandbox module of SymPy.

This module contains experimental code, use at your own risk!

There is no warranty that this code will still be located here in future
versions of SymPy.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\matrices.py ===
def allclose(A, B, rtol=1e-05, atol=1e-08):
    if len(A) != len(B):
        return False

    for x, y in zip(A, B):
        if abs(x-y) > atol + rtol * max(abs(x), abs(y)):
            return False
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_trio.py ===
def test_trio_import() -> None:
    import sys

    for module in list(sys.modules.keys()):
        if module.startswith("trio"):
            del sys.modules[module]

    import trio  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_feather.py ===
""" test feather-format compat """
import numpy as np
import pytest

from pandas.compat.pyarrow import (
    pa_version_under18p0,
    pa_version_under19p0,
)

import pandas as pd
import pandas._testing as tm

from pandas.io.feather_format import read_feather, to_feather  # isort:skip

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


pa = pytest.importorskip("pyarrow")


@pytest.mark.single_cpu
class TestFeather:
    def check_error_on_write(self, df, exc, err_msg):
        # check that we are raising the exception
        # on writing

        with pytest.raises(exc, match=err_msg):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_external_error_on_write(self, df):
        # check that we are raising the exception
        # on writing

        with tm.external_error_raised(Exception):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_round_trip(self, df, expected=None, write_kwargs={}, **read_kwargs):
        if expected is None:
            expected = df.copy()

        with tm.ensure_clean() as path:
            to_feather(df, path, **write_kwargs)

            result = read_feather(path, **read_kwargs)

            tm.assert_frame_equal(result, expected)

    def test_error(self):
        msg = "feather only support IO with DataFrames"
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            self.check_error_on_write(obj, ValueError, msg)

    def test_basic(self):
        df = pd.DataFrame(
            {
                "string": list("abc"),
                "int": list(range(1, 4)),
                "uint": np.arange(3, 6).astype("u1"),
                "float": np.arange(4.0, 7.0, dtype="float64"),
                "float_with_null": [1.0, np.nan, 3],
                "bool": [True, False, True],
                "bool_with_null": [True, np.nan, False],
                "cat": pd.Categorical(list("abc")),
                "dt": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3)), freq=None
                ),
                "dttz": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, tz="US/Eastern")),
                    freq=None,
                ),
                "dt_with_null": [
                    pd.Timestamp("20130101"),
                    pd.NaT,
                    pd.Timestamp("20130103"),
                ],
                "dtns": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, freq="ns")), freq=None
                ),
            }
        )
        df["periods"] = pd.period_range("2013", freq="M", periods=3)
        df["timedeltas"] = pd.timedelta_range("1 day", periods=3)
        df["intervals"] = pd.interval_range(0, 3, 3)

        assert df.dttz.dtype.tz.zone == "US/Eastern"

        expected = df.copy()
        expected.loc[1, "bool_with_null"] = None
        self.check_round_trip(df, expected=expected)

    def test_duplicate_columns(self):
        # https://github.com/wesm/feather/issues/53
        # not currently able to handle duplicate columns
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
        self.check_external_error_on_write(df)

    def test_read_columns(self):
        # GH 24025
        df = pd.DataFrame(
            {
                "col1": list("abc"),
                "col2": list(range(1, 4)),
                "col3": list("xyz"),
                "col4": list(range(4, 7)),
            }
        )
        columns = ["col1", "col3"]
        self.check_round_trip(df, expected=df[columns], columns=columns)

    def test_read_columns_different_order(self):
        # GH 33878
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [True, False]})
        expected = df[["B", "A"]]
        self.check_round_trip(df, expected, columns=["B", "A"])

    def test_unsupported_other(self):
        # mixed python objects
        df = pd.DataFrame({"a": ["a", 1, 2.0]})
        self.check_external_error_on_write(df)

    def test_rw_use_threads(self):
        df = pd.DataFrame({"A": np.arange(100000)})
        self.check_round_trip(df, use_threads=True)
        self.check_round_trip(df, use_threads=False)

    def test_path_pathlib(self):
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD")),
            index=pd.Index([f"i-{i}" for i in range(30)]),
        ).reset_index()
        result = tm.round_trip_pathlib(df.to_feather, read_feather)
        tm.assert_frame_equal(df, result)

    def test_path_localpath(self):
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD")),
            index=pd.Index([f"i-{i}" for i in range(30)]),
        ).reset_index()
        result = tm.round_trip_localpath(df.to_feather, read_feather)
        tm.assert_frame_equal(df, result)

    def test_passthrough_keywords(self):
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD")),
            index=pd.Index([f"i-{i}" for i in range(30)]),
        ).reset_index()
        self.check_round_trip(df, write_kwargs={"version": 1})

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_http_path(self, feather_file, httpserver):
        # GH 29055
        expected = read_feather(feather_file)
        with open(feather_file, "rb") as f:
            httpserver.serve_content(content=f.read())
            res = read_feather(httpserver.url)
        tm.assert_frame_equal(expected, res)

    def test_read_feather_dtype_backend(
        self, string_storage, dtype_backend, using_infer_string
    ):
        # GH#50765
        df = pd.DataFrame(
            {
                "a": pd.Series([1, np.nan, 3], dtype="Int64"),
                "b": pd.Series([1, 2, 3], dtype="Int64"),
                "c": pd.Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": pd.Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

        with tm.ensure_clean() as path:
            to_feather(df, path)
            with pd.option_context("mode.string_storage", string_storage):
                result = read_feather(path, dtype_backend=dtype_backend)

        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            if using_infer_string:
                string_dtype = pd.ArrowDtype(pa.large_string())
            else:
                string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)

        expected = pd.DataFrame(
            {
                "a": pd.Series([1, np.nan, 3], dtype="Int64"),
                "b": pd.Series([1, 2, 3], dtype="Int64"),
                "c": pd.Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": pd.Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": pd.Series([True, False, pd.NA], dtype="boolean"),
                "f": pd.Series([True, False, True], dtype="boolean"),
                "g": pd.Series(["a", "b", "c"], dtype=string_dtype),
                "h": pd.Series(["a", "b", None], dtype=string_dtype),
            }
        )

        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            expected = pd.DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        if using_infer_string:
            expected.columns = expected.columns.astype(
                pd.StringDtype(string_storage, na_value=np.nan)
            )
        tm.assert_frame_equal(result, expected)

    def test_int_columns_and_index(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([3, 4, 5], name="test"))
        self.check_round_trip(df)

    def test_invalid_dtype_backend(self):
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        df = pd.DataFrame({"int": list(range(1, 4))})
        with tm.ensure_clean("tmp.feather") as path:
            df.to_feather(path)
            with pytest.raises(ValueError, match=msg):
                read_feather(path, dtype_backend="numpy")

    def test_string_inference(self, tmp_path, using_infer_string):
        # GH#54431
        path = tmp_path / "test_string_inference.p"
        df = pd.DataFrame(data={"a": ["x", "y"]})
        df.to_feather(path)
        with pd.option_context("future.infer_string", True):
            result = read_feather(path)
        dtype = pd.StringDtype(na_value=np.nan)
        expected = pd.DataFrame(
            data={"a": ["x", "y"]}, dtype=pd.StringDtype(na_value=np.nan)
        )
        expected = pd.DataFrame(
            data={"a": ["x", "y"]},
            dtype=dtype,
            columns=pd.Index(
                ["a"],
                dtype=object
                if pa_version_under19p0 and not using_infer_string
                else dtype,
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(pa_version_under18p0, reason="not supported before 18.0")
    def test_string_inference_string_view_type(self, tmp_path):
        # GH#54798
        import pyarrow as pa
        from pyarrow import feather

        path = tmp_path / "string_view.parquet"
        table = pa.table({"a": pa.array([None, "b", "c"], pa.string_view())})
        feather.write_feather(table, path)

        with pd.option_context("future.infer_string", True):
            result = read_feather(path)

            expected = pd.DataFrame(
                data={"a": [None, "b", "c"]}, dtype=pd.StringDtype(na_value=np.nan)
            )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_zeta_functions.py ===
from sympy.concrete.summations import Sum
from sympy.core.function import expand_func
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, riemann_xi, stieltjes, zeta)
from sympy.series.order import O
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
                      random_complex_number as randcplx, verify_numerically)

x = Symbol('x')
a = Symbol('a')
b = Symbol('b', negative=True)
z = Symbol('z')
s = Symbol('s')


def test_zeta_eval():

    assert zeta(nan) is nan
    assert zeta(x, nan) is nan

    assert zeta(0) == Rational(-1, 2)
    assert zeta(0, x) == S.Half - x
    assert zeta(0, b) == S.Half - b

    assert zeta(1) is zoo
    assert zeta(1, 2) is zoo
    assert zeta(1, -7) is zoo
    assert zeta(1, x) is zoo

    assert zeta(2, 1) == pi**2/6
    assert zeta(3, 1) == zeta(3)

    assert zeta(2) == pi**2/6
    assert zeta(4) == pi**4/90
    assert zeta(6) == pi**6/945

    assert zeta(4, 3) == pi**4/90 - Rational(17, 16)
    assert zeta(7, 4) == zeta(7) - Rational(282251, 279936)
    assert zeta(S.Half, 2).func == zeta
    assert expand_func(zeta(S.Half, 2)) == zeta(S.Half) - 1
    assert zeta(x, 3).func == zeta
    assert expand_func(zeta(x, 3)) == zeta(x) - 1 - 1/2**x

    assert zeta(2, 0) is nan
    assert zeta(3, -1) is nan
    assert zeta(4, -2) is nan

    assert zeta(oo) == 1

    assert zeta(-1) == Rational(-1, 12)
    assert zeta(-2) == 0
    assert zeta(-3) == Rational(1, 120)
    assert zeta(-4) == 0
    assert zeta(-5) == Rational(-1, 252)

    assert zeta(-1, 3) == Rational(-37, 12)
    assert zeta(-1, 7) == Rational(-253, 12)
    assert zeta(-1, -4) == Rational(-121, 12)
    assert zeta(-1, -9) == Rational(-541, 12)

    assert zeta(-4, 3) == -17
    assert zeta(-4, -8) == 8772

    assert zeta(0, 1) == Rational(-1, 2)
    assert zeta(0, -1) == Rational(3, 2)

    assert zeta(0, 2) == Rational(-3, 2)
    assert zeta(0, -2) == Rational(5, 2)

    assert zeta(
        3).evalf(20).epsilon_eq(Float("1.2020569031595942854", 20), 1e-19)


def test_zeta_series():
    assert zeta(x, a).series(a, z, 2) == \
        zeta(x, z) - x*(a-z)*zeta(x+1, z) + O((a-z)**2, (a, z))


def test_dirichlet_eta_eval():
    assert dirichlet_eta(0) == S.Half
    assert dirichlet_eta(-1) == Rational(1, 4)
    assert dirichlet_eta(1) == log(2)
    assert dirichlet_eta(1, S.Half).simplify() == pi/2
    assert dirichlet_eta(1, 2) == 1 - log(2)
    assert dirichlet_eta(2) == pi**2/12
    assert dirichlet_eta(4) == pi**4*Rational(7, 720)
    assert str(dirichlet_eta(I).evalf(n=10)) == '0.5325931818 + 0.2293848577*I'
    assert str(dirichlet_eta(I, I).evalf(n=10)) == '3.462349253 + 0.220285771*I'


def test_riemann_xi_eval():
    assert riemann_xi(2) == pi/6
    assert riemann_xi(0) == Rational(1, 2)
    assert riemann_xi(1) == Rational(1, 2)
    assert riemann_xi(3).rewrite(zeta) == 3*zeta(3)/(2*pi)
    assert riemann_xi(4) == pi**2/15


def test_rewriting():
    from sympy.functions.elementary.piecewise import Piecewise
    assert isinstance(dirichlet_eta(x).rewrite(zeta), Piecewise)
    assert isinstance(dirichlet_eta(x).rewrite(genocchi), Piecewise)
    assert zeta(x).rewrite(dirichlet_eta) == dirichlet_eta(x)/(1 - 2**(1 - x))
    assert zeta(x).rewrite(dirichlet_eta, a=2) == zeta(x)
    assert verify_numerically(dirichlet_eta(x), dirichlet_eta(x).rewrite(zeta), x)
    assert verify_numerically(dirichlet_eta(x), dirichlet_eta(x).rewrite(genocchi), x)
    assert verify_numerically(zeta(x), zeta(x).rewrite(dirichlet_eta), x)

    assert zeta(x, a).rewrite(lerchphi) == lerchphi(1, x, a)
    assert polylog(s, z).rewrite(lerchphi) == lerchphi(z, s, 1)*z

    assert lerchphi(1, x, a).rewrite(zeta) == zeta(x, a)
    assert z*lerchphi(z, s, 1).rewrite(polylog) == polylog(s, z)


def test_derivatives():
    from sympy.core.function import Derivative
    assert zeta(x, a).diff(x) == Derivative(zeta(x, a), x)
    assert zeta(x, a).diff(a) == -x*zeta(x + 1, a)
    assert lerchphi(
        z, s, a).diff(z) == (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
    assert lerchphi(z, s, a).diff(a) == -s*lerchphi(z, s + 1, a)
    assert polylog(s, z).diff(z) == polylog(s - 1, z)/z

    b = randcplx()
    c = randcplx()
    assert td(zeta(b, x), x)
    assert td(polylog(b, z), z)
    assert td(lerchphi(c, b, x), x)
    assert td(lerchphi(x, b, c), x)
    raises(ArgumentIndexError, lambda: lerchphi(c, b, x).fdiff(2))
    raises(ArgumentIndexError, lambda: lerchphi(c, b, x).fdiff(4))
    raises(ArgumentIndexError, lambda: polylog(b, z).fdiff(1))
    raises(ArgumentIndexError, lambda: polylog(b, z).fdiff(3))


def myexpand(func, target):
    expanded = expand_func(func)
    if target is not None:
        return expanded == target
    if expanded == func:  # it didn't expand
        return False

    # check to see that the expanded and original evaluate to the same value
    subs = {}
    for a in func.free_symbols:
        subs[a] = randcplx()
    return abs(func.subs(subs).n()
               - expanded.replace(exp_polar, exp).subs(subs).n()) < 1e-10


def test_polylog_expansion():
    assert polylog(s, 0) == 0
    assert polylog(s, 1) == zeta(s)
    assert polylog(s, -1) == -dirichlet_eta(s)
    assert polylog(s, exp_polar(I*pi*Rational(4, 3))) == polylog(s, exp(I*pi*Rational(4, 3)))
    assert polylog(s, exp_polar(I*pi)/3) == polylog(s, exp(I*pi)/3)

    assert myexpand(polylog(1, z), -log(1 - z))
    assert myexpand(polylog(0, z), z/(1 - z))
    assert myexpand(polylog(-1, z), z/(1 - z)**2)
    assert ((1-z)**3 * expand_func(polylog(-2, z))).simplify() == z*(1 + z)
    assert myexpand(polylog(-5, z), None)


def test_polylog_series():
    assert polylog(1, z).series(z, n=5) == z + z**2/2 + z**3/3 + z**4/4 + O(z**5)
    assert polylog(1, sqrt(z)).series(z, n=3) == z/2 + z**2/4 + sqrt(z)\
        + z**(S(3)/2)/3 + z**(S(5)/2)/5 + O(z**3)

    # https://github.com/sympy/sympy/issues/9497
    assert polylog(S(3)/2, -z).series(z, 0, 5) == -z + sqrt(2)*z**2/4\
        - sqrt(3)*z**3/9 + z**4/8 + O(z**5)


def test_issue_8404():
    i = Symbol('i', integer=True)
    assert Abs(Sum(1/(3*i + 1)**2, (i, 0, S.Infinity)).doit().n(4)
        - 1.122) < 0.001


def test_polylog_values():
    assert polylog(2, 2) == pi**2/4 - I*pi*log(2)
    assert polylog(2, S.Half) == pi**2/12 - log(2)**2/2
    for z in [S.Half, 2, (sqrt(5)-1)/2, -(sqrt(5)-1)/2, -(sqrt(5)+1)/2, (3-sqrt(5))/2]:
        assert Abs(polylog(2, z).evalf() - polylog(2, z, evaluate=False).evalf()) < 1e-15
    z = Symbol("z")
    for s in [-1, 0]:
        for _ in range(10):
            assert verify_numerically(polylog(s, z), polylog(s, z, evaluate=False),
                                      z, a=-3, b=-2, c=S.Half, d=2)
            assert verify_numerically(polylog(s, z), polylog(s, z, evaluate=False),
                                      z, a=2, b=-2, c=5, d=2)

    from sympy.integrals.integrals import Integral
    assert polylog(0, Integral(1, (x, 0, 1))) == -S.Half


def test_lerchphi_expansion():
    assert myexpand(lerchphi(1, s, a), zeta(s, a))
    assert myexpand(lerchphi(z, s, 1), polylog(s, z)/z)

    # direct summation
    assert myexpand(lerchphi(z, -1, a), a/(1 - z) + z/(1 - z)**2)
    assert myexpand(lerchphi(z, -3, a), None)
    # polylog reduction
    assert myexpand(lerchphi(z, s, S.Half),
                    2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z)
                              - polylog(s, polar_lift(-1)*sqrt(z))/sqrt(z)))
    assert myexpand(lerchphi(z, s, 2), -1/z + polylog(s, z)/z**2)
    assert myexpand(lerchphi(z, s, Rational(3, 2)), None)
    assert myexpand(lerchphi(z, s, Rational(7, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-1, 3)), None)
    assert myexpand(lerchphi(z, s, Rational(-5, 2)), None)

    # hurwitz zeta reduction
    assert myexpand(lerchphi(-1, s, a),
                    2**(-s)*zeta(s, a/2) - 2**(-s)*zeta(s, (a + 1)/2))
    assert myexpand(lerchphi(I, s, a), None)
    assert myexpand(lerchphi(-I, s, a), None)
    assert myexpand(lerchphi(exp(I*pi*Rational(2, 5)), s, a), None)


def test_stieltjes():
    assert isinstance(stieltjes(x), stieltjes)
    assert isinstance(stieltjes(x, a), stieltjes)

    # Zero'th constant EulerGamma
    assert stieltjes(0) == S.EulerGamma
    assert stieltjes(0, 1) == S.EulerGamma

    # Not defined
    assert stieltjes(nan) is nan
    assert stieltjes(0, nan) is nan
    assert stieltjes(-1) is S.ComplexInfinity
    assert stieltjes(1.5) is S.ComplexInfinity
    assert stieltjes(z, 0) is S.ComplexInfinity
    assert stieltjes(z, -1) is S.ComplexInfinity


def test_stieltjes_evalf():
    assert abs(stieltjes(0).evalf() - 0.577215664) < 1E-9
    assert abs(stieltjes(0, 0.5).evalf() - 1.963510026) < 1E-9
    assert abs(stieltjes(1, 2).evalf() + 0.072815845) < 1E-9


def test_issue_10475():
    a = Symbol('a', extended_real=True)
    b = Symbol('b', extended_positive=True)
    s = Symbol('s', zero=False)

    assert zeta(2 + I).is_finite
    assert zeta(1).is_finite is False
    assert zeta(x).is_finite is None
    assert zeta(x + I).is_finite is None
    assert zeta(a).is_finite is None
    assert zeta(b).is_finite is None
    assert zeta(-b).is_finite is True
    assert zeta(b**2 - 2*b + 1).is_finite is None
    assert zeta(a + I).is_finite is True
    assert zeta(b + 1).is_finite is True
    assert zeta(s + 1).is_finite is True


def test_issue_14177():
    n = Symbol('n', nonnegative=True, integer=True)

    assert zeta(-n).rewrite(bernoulli) == bernoulli(n+1) / (-n-1)
    assert zeta(-n, a).rewrite(bernoulli) == bernoulli(n+1, a) / (-n-1)
    z2n = -(2*I*pi)**(2*n)*bernoulli(2*n) / (2*factorial(2*n))
    assert zeta(2*n).rewrite(bernoulli) == z2n
    assert expand_func(zeta(s, n+1)) == zeta(s) - harmonic(n, s)
    assert expand_func(zeta(-b, -n)) is nan
    assert expand_func(zeta(-b, n)) == zeta(-b, n)

    n = Symbol('n')

    assert zeta(2*n) == zeta(2*n) # As sign of z (= 2*n) is not determined

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_constructors.py ===
import numpy as np
import pytest

from pandas._libs.sparse import IntIndex

import pandas as pd
from pandas import (
    SparseDtype,
    isna,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


class TestConstructors:
    def test_constructor_dtype(self):
        arr = SparseArray([np.nan, 1, 2, np.nan])
        assert arr.dtype == SparseDtype(np.float64, np.nan)
        assert arr.dtype.subtype == np.float64
        assert np.isnan(arr.fill_value)

        arr = SparseArray([np.nan, 1, 2, np.nan], fill_value=0)
        assert arr.dtype == SparseDtype(np.float64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], dtype=np.float64)
        assert arr.dtype == SparseDtype(np.float64, np.nan)
        assert np.isnan(arr.fill_value)

        arr = SparseArray([0, 1, 2, 4], dtype=np.int64)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], fill_value=0, dtype=np.int64)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], dtype=None)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

        arr = SparseArray([0, 1, 2, 4], fill_value=0, dtype=None)
        assert arr.dtype == SparseDtype(np.int64, 0)
        assert arr.fill_value == 0

    def test_constructor_dtype_str(self):
        result = SparseArray([1, 2, 3], dtype="int")
        expected = SparseArray([1, 2, 3], dtype=int)
        tm.assert_sp_array_equal(result, expected)

    def test_constructor_sparse_dtype(self):
        result = SparseArray([1, 0, 0, 1], dtype=SparseDtype("int64", -1))
        expected = SparseArray([1, 0, 0, 1], fill_value=-1, dtype=np.int64)
        tm.assert_sp_array_equal(result, expected)
        assert result.sp_values.dtype == np.dtype("int64")

    def test_constructor_sparse_dtype_str(self):
        result = SparseArray([1, 0, 0, 1], dtype="Sparse[int32]")
        expected = SparseArray([1, 0, 0, 1], dtype=np.int32)
        tm.assert_sp_array_equal(result, expected)
        assert result.sp_values.dtype == np.dtype("int32")

    def test_constructor_object_dtype(self):
        # GH#11856
        arr = SparseArray(["A", "A", np.nan, "B"], dtype=object)
        assert arr.dtype == SparseDtype(object)
        assert np.isnan(arr.fill_value)

        arr = SparseArray(["A", "A", np.nan, "B"], dtype=object, fill_value="A")
        assert arr.dtype == SparseDtype(object, "A")
        assert arr.fill_value == "A"

    def test_constructor_object_dtype_bool_fill(self):
        # GH#17574
        data = [False, 0, 100.0, 0.0]
        arr = SparseArray(data, dtype=object, fill_value=False)
        assert arr.dtype == SparseDtype(object, False)
        assert arr.fill_value is False
        arr_expected = np.array(data, dtype=object)
        it = (type(x) == type(y) and x == y for x, y in zip(arr, arr_expected))
        assert np.fromiter(it, dtype=np.bool_).all()

    @pytest.mark.parametrize("dtype", [SparseDtype(int, 0), int])
    def test_constructor_na_dtype(self, dtype):
        with pytest.raises(ValueError, match="Cannot convert"):
            SparseArray([0, 1, np.nan], dtype=dtype)

    def test_constructor_warns_when_losing_timezone(self):
        # GH#32501 warn when losing timezone information
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")

        expected = SparseArray(np.asarray(dti, dtype="datetime64[ns]"))

        with tm.assert_produces_warning(UserWarning):
            result = SparseArray(dti)

        tm.assert_sp_array_equal(result, expected)

        with tm.assert_produces_warning(UserWarning):
            result = SparseArray(pd.Series(dti))

        tm.assert_sp_array_equal(result, expected)

    def test_constructor_spindex_dtype(self):
        arr = SparseArray(data=[1, 2], sparse_index=IntIndex(4, [1, 2]))
        # TODO: actionable?
        # XXX: Behavior change: specifying SparseIndex no longer changes the
        # fill_value
        expected = SparseArray([0, 1, 2, 0], kind="integer")
        tm.assert_sp_array_equal(arr, expected)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        arr = SparseArray(
            data=[1, 2, 3],
            sparse_index=IntIndex(4, [1, 2, 3]),
            dtype=np.int64,
            fill_value=0,
        )
        exp = SparseArray([0, 1, 2, 3], dtype=np.int64, fill_value=0)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        arr = SparseArray(
            data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=np.int64
        )
        exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=np.int64)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        arr = SparseArray(
            data=[1, 2, 3],
            sparse_index=IntIndex(4, [1, 2, 3]),
            dtype=None,
            fill_value=0,
        )
        exp = SparseArray([0, 1, 2, 3], dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

    @pytest.mark.parametrize("sparse_index", [None, IntIndex(1, [0])])
    def test_constructor_spindex_dtype_scalar(self, sparse_index):
        # scalar input
        msg = "Constructing SparseArray with scalar data is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            arr = SparseArray(data=1, sparse_index=sparse_index, dtype=None)
        exp = SparseArray([1], dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

        with tm.assert_produces_warning(FutureWarning, match=msg):
            arr = SparseArray(data=1, sparse_index=IntIndex(1, [0]), dtype=None)
        exp = SparseArray([1], dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

    def test_constructor_spindex_dtype_scalar_broadcasts(self):
        arr = SparseArray(
            data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=None
        )
        exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=None)
        tm.assert_sp_array_equal(arr, exp)
        assert arr.dtype == SparseDtype(np.int64)
        assert arr.fill_value == 0

    @pytest.mark.parametrize(
        "data, fill_value",
        [
            (np.array([1, 2]), 0),
            (np.array([1.0, 2.0]), np.nan),
            ([True, False], False),
            ([pd.Timestamp("2017-01-01")], pd.NaT),
        ],
    )
    def test_constructor_inferred_fill_value(self, data, fill_value):
        result = SparseArray(data).fill_value

        if isna(fill_value):
            assert isna(result)
        else:
            assert result == fill_value

    @pytest.mark.parametrize("format", ["coo", "csc", "csr"])
    @pytest.mark.parametrize("size", [0, 10])
    def test_from_spmatrix(self, size, format):
        sp_sparse = pytest.importorskip("scipy.sparse")

        mat = sp_sparse.random(size, 1, density=0.5, format=format)
        result = SparseArray.from_spmatrix(mat)

        result = np.asarray(result)
        expected = mat.toarray().ravel()
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("format", ["coo", "csc", "csr"])
    def test_from_spmatrix_including_explicit_zero(self, format):
        sp_sparse = pytest.importorskip("scipy.sparse")

        mat = sp_sparse.random(10, 1, density=0.5, format=format)
        mat.data[0] = 0
        result = SparseArray.from_spmatrix(mat)

        result = np.asarray(result)
        expected = mat.toarray().ravel()
        tm.assert_numpy_array_equal(result, expected)

    def test_from_spmatrix_raises(self):
        sp_sparse = pytest.importorskip("scipy.sparse")

        mat = sp_sparse.eye(5, 4, format="csc")

        with pytest.raises(ValueError, match="not '4'"):
            SparseArray.from_spmatrix(mat)

    def test_constructor_from_too_large_array(self):
        with pytest.raises(TypeError, match="expected dimension <= 1 data"):
            SparseArray(np.arange(10).reshape((2, 5)))

    def test_constructor_from_sparse(self):
        zarr = SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)
        res = SparseArray(zarr)
        assert res.fill_value == 0
        tm.assert_almost_equal(res.sp_values, zarr.sp_values)

    def test_constructor_copy(self):
        arr_data = np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 6])
        arr = SparseArray(arr_data)

        cp = SparseArray(arr, copy=True)
        cp.sp_values[:3] = 0
        assert not (arr.sp_values[:3] == 0).any()

        not_copy = SparseArray(arr)
        not_copy.sp_values[:3] = 0
        assert (arr.sp_values[:3] == 0).all()

    def test_constructor_bool(self):
        # GH#10648
        data = np.array([False, False, True, True, False, False])
        arr = SparseArray(data, fill_value=False, dtype=bool)

        assert arr.dtype == SparseDtype(bool)
        tm.assert_numpy_array_equal(arr.sp_values, np.array([True, True]))
        # Behavior change: np.asarray densifies.
        # tm.assert_numpy_array_equal(arr.sp_values, np.asarray(arr))
        tm.assert_numpy_array_equal(arr.sp_index.indices, np.array([2, 3], np.int32))

        dense = arr.to_dense()
        assert dense.dtype == bool
        tm.assert_numpy_array_equal(dense, data)

    def test_constructor_bool_fill_value(self):
        arr = SparseArray([True, False, True], dtype=None)
        assert arr.dtype == SparseDtype(np.bool_)
        assert not arr.fill_value

        arr = SparseArray([True, False, True], dtype=np.bool_)
        assert arr.dtype == SparseDtype(np.bool_)
        assert not arr.fill_value

        arr = SparseArray([True, False, True], dtype=np.bool_, fill_value=True)
        assert arr.dtype == SparseDtype(np.bool_, True)
        assert arr.fill_value

    def test_constructor_float32(self):
        # GH#10648
        data = np.array([1.0, np.nan, 3], dtype=np.float32)
        arr = SparseArray(data, dtype=np.float32)

        assert arr.dtype == SparseDtype(np.float32)
        tm.assert_numpy_array_equal(arr.sp_values, np.array([1, 3], dtype=np.float32))
        # Behavior change: np.asarray densifies.
        # tm.assert_numpy_array_equal(arr.sp_values, np.asarray(arr))
        tm.assert_numpy_array_equal(
            arr.sp_index.indices, np.array([0, 2], dtype=np.int32)
        )

        dense = arr.to_dense()
        assert dense.dtype == np.float32
        tm.assert_numpy_array_equal(dense, data)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_generate.py ===
from bisect import bisect, bisect_left

from sympy.functions.combinatorial.numbers import mobius, totient
from sympy.ntheory.generate import (sieve, Sieve)

from sympy.ntheory import isprime, randprime, nextprime, prevprime, \
    primerange, primepi, prime, primorial, composite, compositepi
from sympy.ntheory.generate import cycle_length, _primepi
from sympy.ntheory.primetest import mr
from sympy.testing.pytest import raises

def test_prime():
    assert prime(1) == 2
    assert prime(2) == 3
    assert prime(5) == 11
    assert prime(11) == 31
    assert prime(57) == 269
    assert prime(296) == 1949
    assert prime(559) == 4051
    assert prime(3000) == 27449
    assert prime(4096) == 38873
    assert prime(9096) == 94321
    assert prime(25023) == 287341
    assert prime(10000000) == 179424673 # issue #20951
    assert prime(99999999) == 2038074739
    raises(ValueError, lambda: prime(0))
    sieve.extend(3000)
    assert prime(401) == 2749
    raises(ValueError, lambda: prime(-1))


def test__primepi():
    assert _primepi(-1) == 0
    assert _primepi(1) == 0
    assert _primepi(2) == 1
    assert _primepi(5) == 3
    assert _primepi(11) == 5
    assert _primepi(57) == 16
    assert _primepi(296) == 62
    assert _primepi(559) == 102
    assert _primepi(3000) == 430
    assert _primepi(4096) == 564
    assert _primepi(9096) == 1128
    assert _primepi(25023) == 2763
    assert _primepi(10**8) == 5761455
    assert _primepi(253425253) == 13856396
    assert _primepi(8769575643) == 401464322
    sieve.extend(3000)
    assert _primepi(2000) == 303


def test_composite():
    from sympy.ntheory.generate import sieve
    sieve._reset()
    assert composite(1) == 4
    assert composite(2) == 6
    assert composite(5) == 10
    assert composite(11) == 20
    assert composite(41) == 58
    assert composite(57) == 80
    assert composite(296) == 370
    assert composite(559) == 684
    assert composite(3000) == 3488
    assert composite(4096) == 4736
    assert composite(9096) == 10368
    assert composite(25023) == 28088
    sieve.extend(3000)
    assert composite(1957) == 2300
    assert composite(2568) == 2998
    raises(ValueError, lambda: composite(0))


def test_compositepi():
    assert compositepi(1) == 0
    assert compositepi(2) == 0
    assert compositepi(5) == 1
    assert compositepi(11) == 5
    assert compositepi(57) == 40
    assert compositepi(296) == 233
    assert compositepi(559) == 456
    assert compositepi(3000) == 2569
    assert compositepi(4096) == 3531
    assert compositepi(9096) == 7967
    assert compositepi(25023) == 22259
    assert compositepi(10**8) == 94238544
    assert compositepi(253425253) == 239568856
    assert compositepi(8769575643) == 8368111320
    sieve.extend(3000)
    assert compositepi(2321) == 1976


def test_generate():
    from sympy.ntheory.generate import sieve
    sieve._reset()
    assert nextprime(-4) == 2
    assert nextprime(2) == 3
    assert nextprime(5) == 7
    assert nextprime(12) == 13
    assert prevprime(3) == 2
    assert prevprime(7) == 5
    assert prevprime(13) == 11
    assert prevprime(19) == 17
    assert prevprime(20) == 19

    sieve.extend_to_no(9)
    assert sieve._list[-1] == 23

    assert sieve._list[-1] < 31
    assert 31 in sieve

    assert nextprime(90) == 97
    assert nextprime(10**40) == (10**40 + 121)
    primelist = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                 79, 83, 89, 97, 101, 103, 107, 109, 113,
                 127, 131, 137, 139, 149, 151, 157, 163,
                 167, 173, 179, 181, 191, 193, 197, 199,
                 211, 223, 227, 229, 233, 239, 241, 251,
                 257, 263, 269, 271, 277, 281, 283, 293]
    for i in range(len(primelist) - 2):
        for j in range(2, len(primelist) - i):
            assert nextprime(primelist[i], j) == primelist[i + j]
            if 3 < i:
                assert nextprime(primelist[i] - 1, j) == primelist[i + j - 1]
    raises(ValueError, lambda: nextprime(2, 0))
    raises(ValueError, lambda: nextprime(2, -1))
    assert prevprime(97) == 89
    assert prevprime(10**40) == (10**40 - 17)

    raises(ValueError, lambda: Sieve(0))
    raises(ValueError, lambda: Sieve(-1))
    for sieve_interval in [1, 10, 11, 1_000_000]:
        s = Sieve(sieve_interval=sieve_interval)
        for head in range(s._list[-1] + 1, (s._list[-1] + 1)**2, 2):
            for tail in range(head + 1, (s._list[-1] + 1)**2):
                A = list(s._primerange(head, tail))
                B = primelist[bisect(primelist, head):bisect_left(primelist, tail)]
                assert A == B
        for k in range(s._list[-1], primelist[-1] - 1, 2):
            s = Sieve(sieve_interval=sieve_interval)
            s.extend(k)
            assert list(s._list) == primelist[:bisect(primelist, k)]
            s.extend(primelist[-1])
            assert list(s._list) == primelist

    assert list(sieve.primerange(10, 1)) == []
    assert list(sieve.primerange(5, 9)) == [5, 7]
    sieve._reset(prime=True)
    assert list(sieve.primerange(2, 13)) == [2, 3, 5, 7, 11]
    assert list(sieve.primerange(13)) == [2, 3, 5, 7, 11]
    assert list(sieve.primerange(8)) == [2, 3, 5, 7]
    assert list(sieve.primerange(-2)) == []
    assert list(sieve.primerange(29)) == [2, 3, 5, 7, 11, 13, 17, 19, 23]
    assert list(sieve.primerange(34)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    assert list(sieve.totientrange(5, 15)) == [4, 2, 6, 4, 6, 4, 10, 4, 12, 6]
    sieve._reset(totient=True)
    assert list(sieve.totientrange(3, 13)) == [2, 2, 4, 2, 6, 4, 6, 4, 10, 4]
    assert list(sieve.totientrange(900, 1000)) == [totient(x) for x in range(900, 1000)]
    assert list(sieve.totientrange(0, 1)) == []
    assert list(sieve.totientrange(1, 2)) == [1]

    assert list(sieve.mobiusrange(5, 15)) == [-1, 1, -1, 0, 0, 1, -1, 0, -1, 1]
    sieve._reset(mobius=True)
    assert list(sieve.mobiusrange(3, 13)) == [-1, 0, -1, 1, -1, 0, 0, 1, -1, 0]
    assert list(sieve.mobiusrange(1050, 1100)) == [mobius(x) for x in range(1050, 1100)]
    assert list(sieve.mobiusrange(0, 1)) == []
    assert list(sieve.mobiusrange(1, 2)) == [1]

    assert list(primerange(10, 1)) == []
    assert list(primerange(2, 7)) == [2, 3, 5]
    assert list(primerange(2, 10)) == [2, 3, 5, 7]
    assert list(primerange(1050, 1100)) == [1051, 1061,
        1063, 1069, 1087, 1091, 1093, 1097]
    s = Sieve()
    for i in range(30, 2350, 376):
        for j in range(2, 5096, 1139):
            A = list(s.primerange(i, i + j))
            B = list(primerange(i, i + j))
            assert A == B
    s = Sieve()
    sieve._reset(prime=True)
    sieve.extend(13)
    for i in range(200):
        for j in range(i, 200):
            A = list(s.primerange(i, j))
            B = list(primerange(i, j))
            assert A == B
    sieve.extend(1000)
    for a, b in [(901, 1103), # a < 1000 < b < 1000**2
                 (806, 1002007), # a < 1000 < 1000**2 < b
                 (2000, 30001), # 1000 < a < b < 1000**2
                 (100005, 1010001), # 1000 < a < 1000**2 < b
                 (1003003, 1005000), # 1000**2 < a < b
                 ]:
        assert list(primerange(a, b)) == list(s.primerange(a, b))
    sieve._reset(prime=True)
    sieve.extend(100000)
    assert len(sieve._list) == len(set(sieve._list))
    s = Sieve()
    assert s[10] == 29

    assert nextprime(2, 2) == 5

    raises(ValueError, lambda: totient(0))

    raises(ValueError, lambda: primorial(0))

    assert mr(1, [2]) is False

    func = lambda i: (i**2 + 1) % 51
    assert next(cycle_length(func, 4)) == (6, 3)
    assert list(cycle_length(func, 4, values=True)) == \
        [4, 17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]
    assert next(cycle_length(func, 4, nmax=5)) == (5, None)
    assert list(cycle_length(func, 4, nmax=5, values=True)) == \
        [4, 17, 35, 2, 5]
    sieve.extend(3000)
    assert nextprime(2968) == 2969
    assert prevprime(2930) == 2927
    raises(ValueError, lambda: prevprime(1))
    raises(ValueError, lambda: prevprime(-4))


def test_randprime():
    assert randprime(10, 1) is None
    assert randprime(3, -3) is None
    assert randprime(2, 3) == 2
    assert randprime(1, 3) == 2
    assert randprime(3, 5) == 3
    raises(ValueError, lambda: randprime(-12, -2))
    raises(ValueError, lambda: randprime(-10, 0))
    raises(ValueError, lambda: randprime(20, 22))
    raises(ValueError, lambda: randprime(0, 2))
    raises(ValueError, lambda: randprime(1, 2))
    for a in [100, 300, 500, 250000]:
        for b in [100, 300, 500, 250000]:
            p = randprime(a, a + b)
            assert a <= p < (a + b) and isprime(p)


def test_primorial():
    assert primorial(1) == 2
    assert primorial(1, nth=0) == 1
    assert primorial(2) == 6
    assert primorial(2, nth=0) == 2
    assert primorial(4, nth=0) == 6


def test_search():
    assert 2 in sieve
    assert 2.1 not in sieve
    assert 1 not in sieve
    assert 2**1000 not in sieve
    raises(ValueError, lambda: sieve.search(1))


def test_sieve_slice():
    assert sieve[5] == 11
    assert list(sieve[5:10]) == [sieve[x] for x in range(5, 10)]
    assert list(sieve[5:10:2]) == [sieve[x] for x in range(5, 10, 2)]
    assert list(sieve[1:5]) == [2, 3, 5, 7]
    raises(IndexError, lambda: sieve[:5])
    raises(IndexError, lambda: sieve[0])
    raises(IndexError, lambda: sieve[0:5])

def test_sieve_iter():
    values = []
    for value in sieve:
        if value > 7:
            break
        values.append(value)
    assert values == list(sieve[1:5])


def test_sieve_repr():
    assert "sieve" in repr(sieve)
    assert "prime" in repr(sieve)


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert primepi(0) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\datetimes\test_constructors.py ===
import numpy as np
import pytest

from pandas._libs import iNaT

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestDatetimeArrayConstructor:
    def test_from_sequence_invalid_type(self):
        mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
        with pytest.raises(TypeError, match="Cannot create a DatetimeArray"):
            DatetimeArray._from_sequence(mi, dtype="M8[ns]")

    def test_only_1dim_accepted(self):
        arr = np.array([0, 1, 2, 3], dtype="M8[h]").astype("M8[ns]")

        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Only 1-dimensional"):
                # 3-dim, we allow 2D to sneak in for ops purposes GH#29853
                DatetimeArray(arr.reshape(2, 2, 1))

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Only 1-dimensional"):
                # 0-dim
                DatetimeArray(arr[[0]].squeeze())

    def test_freq_validation(self):
        # GH#24623 check that invalid instances cannot be created with the
        #  public constructor
        arr = np.arange(5, dtype=np.int64) * 3600 * 10**9

        msg = (
            "Inferred frequency h from passed values does not "
            "conform to passed frequency W-SUN"
        )
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, freq="W")

    @pytest.mark.parametrize(
        "meth",
        [
            DatetimeArray._from_sequence,
            pd.to_datetime,
            pd.DatetimeIndex,
        ],
    )
    def test_mixing_naive_tzaware_raises(self, meth):
        # GH#24569
        arr = np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")])

        msg = (
            "Cannot mix tz-aware with tz-naive values|"
            "Tz-aware datetime.datetime cannot be converted "
            "to datetime64 unless utc=True"
        )

        for obj in [arr, arr[::-1]]:
            # check that we raise regardless of whether naive is found
            #  before aware or vice-versa
            with pytest.raises(ValueError, match=msg):
                meth(obj)

    def test_from_pandas_array(self):
        arr = pd.array(np.arange(5, dtype=np.int64)) * 3600 * 10**9

        result = DatetimeArray._from_sequence(arr, dtype="M8[ns]")._with_freq("infer")

        expected = pd.date_range("1970-01-01", periods=5, freq="h")._data
        tm.assert_datetime_array_equal(result, expected)

    def test_mismatched_timezone_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            arr = DatetimeArray(
                np.array(["2000-01-01T06:00:00"], dtype="M8[ns]"),
                dtype=DatetimeTZDtype(tz="US/Central"),
            )
        dtype = DatetimeTZDtype(tz="US/Eastern")
        msg = r"dtype=datetime64\[ns.*\] does not match data dtype datetime64\[ns.*\]"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr, dtype=dtype)

        # also with mismatched tzawareness
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr, dtype=np.dtype("M8[ns]"))
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(TypeError, match=msg):
                DatetimeArray(arr.tz_localize(None), dtype=arr.dtype)

    def test_non_array_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="list"):
                DatetimeArray([1, 2, 3])

    def test_bool_dtype_raises(self):
        arr = np.array([1, 2, 3], dtype="bool")

        depr_msg = "DatetimeArray.__init__ is deprecated"
        msg = "Unexpected value for 'dtype': 'bool'. Must be"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr)

        msg = r"dtype bool cannot be converted to datetime64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            DatetimeArray._from_sequence(arr, dtype="M8[ns]")

        with pytest.raises(TypeError, match=msg):
            pd.DatetimeIndex(arr)

        with pytest.raises(TypeError, match=msg):
            pd.to_datetime(arr)

    def test_incorrect_dtype_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="category")

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="m8[s]")

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="M8[D]")

    def test_mismatched_values_dtype_units(self):
        arr = np.array([1, 2, 3], dtype="M8[s]")
        dtype = np.dtype("M8[ns]")
        msg = "Values resolution does not match dtype."
        depr_msg = "DatetimeArray.__init__ is deprecated"

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, dtype=dtype)

        dtype2 = DatetimeTZDtype(tz="UTC", unit="ns")
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                DatetimeArray(arr, dtype=dtype2)

    def test_freq_infer_raises(self):
        depr_msg = "DatetimeArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Frequency inference"):
                DatetimeArray(np.array([1, 2, 3], dtype="i8"), freq="infer")

    def test_copy(self):
        data = np.array([1, 2, 3], dtype="M8[ns]")
        arr = DatetimeArray._from_sequence(data, copy=False)
        assert arr._ndarray is data

        arr = DatetimeArray._from_sequence(data, copy=True)
        assert arr._ndarray is not data

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_numpy_datetime_unit(self, unit):
        data = np.array([1, 2, 3], dtype=f"M8[{unit}]")
        arr = DatetimeArray._from_sequence(data)
        assert arr.unit == unit
        assert arr[0].unit == unit


class TestSequenceToDT64NS:
    def test_tz_dtype_mismatch_raises(self):
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(TypeError, match="data is already tz-aware"):
            DatetimeArray._from_sequence(arr, dtype=DatetimeTZDtype(tz="UTC"))

    def test_tz_dtype_matches(self):
        dtype = DatetimeTZDtype(tz="US/Central")
        arr = DatetimeArray._from_sequence(["2000"], dtype=dtype)
        result = DatetimeArray._from_sequence(arr, dtype=dtype)
        tm.assert_equal(arr, result)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_2d(self, order):
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        arr = np.array(dti, dtype=object).reshape(3, 2)
        if order == "F":
            arr = arr.T

        res = DatetimeArray._from_sequence(arr, dtype=dti.dtype)
        expected = DatetimeArray._from_sequence(arr.ravel(), dtype=dti.dtype).reshape(
            arr.shape
        )
        tm.assert_datetime_array_equal(res, expected)


# ----------------------------------------------------------------------------
# Arrow interaction


EXTREME_VALUES = [0, 123456789, None, iNaT, 2**63 - 1, -(2**63) + 1]
FINE_TO_COARSE_SAFE = [123_000_000_000, None, -123_000_000_000]
COARSE_TO_FINE_SAFE = [123, None, -123]


@pytest.mark.parametrize(
    ("pa_unit", "pd_unit", "pa_tz", "pd_tz", "data"),
    [
        ("s", "s", "UTC", "UTC", EXTREME_VALUES),
        ("ms", "ms", "UTC", "Europe/Berlin", EXTREME_VALUES),
        ("us", "us", "US/Eastern", "UTC", EXTREME_VALUES),
        ("ns", "ns", "US/Central", "Asia/Kolkata", EXTREME_VALUES),
        ("ns", "s", "UTC", "UTC", FINE_TO_COARSE_SAFE),
        ("us", "ms", "UTC", "Europe/Berlin", FINE_TO_COARSE_SAFE),
        ("ms", "us", "US/Eastern", "UTC", COARSE_TO_FINE_SAFE),
        ("s", "ns", "US/Central", "Asia/Kolkata", COARSE_TO_FINE_SAFE),
    ],
)
def test_from_arrow_with_different_units_and_timezones_with(
    pa_unit, pd_unit, pa_tz, pd_tz, data
):
    pa = pytest.importorskip("pyarrow")

    pa_type = pa.timestamp(pa_unit, tz=pa_tz)
    arr = pa.array(data, type=pa_type)
    dtype = DatetimeTZDtype(unit=pd_unit, tz=pd_tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(data, dtype=f"M8[{pa_unit}, UTC]").astype(
        dtype, copy=False
    )
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    ("unit", "tz"),
    [
        ("s", "UTC"),
        ("ms", "Europe/Berlin"),
        ("us", "US/Eastern"),
        ("ns", "Asia/Kolkata"),
        ("ns", "UTC"),
    ],
)
def test_from_arrow_from_empty(unit, tz):
    pa = pytest.importorskip("pyarrow")

    data = []
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit=unit, tz=tz)

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(np.array(data, dtype=f"datetime64[{unit}]"))
    expected = expected.tz_localize(tz=tz)
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)


def test_from_arrow_from_integers():
    pa = pytest.importorskip("pyarrow")

    data = [0, 123456789, None, 2**63 - 1, iNaT, -123456789]
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit="ns", tz="UTC")

    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(np.array(data, dtype="datetime64[ns]"))
    expected = expected.tz_localize("UTC")
    tm.assert_extension_array_equal(result, expected)

    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_equivalence.py ===
import numpy as np
import pytest

from pandas.core.dtypes.common import is_any_real_numeric_dtype

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


def test_equals(idx):
    assert idx.equals(idx)
    assert idx.equals(idx.copy())
    assert idx.equals(idx.astype(object))
    assert idx.equals(idx.to_flat_index())
    assert idx.equals(idx.to_flat_index().astype("category"))

    assert not idx.equals(list(idx))
    assert not idx.equals(np.array(idx))

    same_values = Index(idx, dtype=object)
    assert idx.equals(same_values)
    assert same_values.equals(idx)

    if idx.nlevels == 1:
        # do not test MultiIndex
        assert not idx.equals(Series(idx))


def test_equals_op(idx):
    # GH9947, GH10637
    index_a = idx

    n = len(index_a)
    index_b = index_a[0:-1]
    index_c = index_a[0:-1].append(index_a[-2:-1])
    index_d = index_a[0:1]
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == index_b
    expected1 = np.array([True] * n)
    expected2 = np.array([True] * (n - 1) + [False])
    tm.assert_numpy_array_equal(index_a == index_a, expected1)
    tm.assert_numpy_array_equal(index_a == index_c, expected2)

    # test comparisons with numpy arrays
    array_a = np.array(index_a)
    array_b = np.array(index_a[0:-1])
    array_c = np.array(index_a[0:-1].append(index_a[-2:-1]))
    array_d = np.array(index_a[0:1])
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == array_b
    tm.assert_numpy_array_equal(index_a == array_a, expected1)
    tm.assert_numpy_array_equal(index_a == array_c, expected2)

    # test comparisons with Series
    series_a = Series(array_a)
    series_b = Series(array_b)
    series_c = Series(array_c)
    series_d = Series(array_d)
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == series_b

    tm.assert_numpy_array_equal(index_a == series_a, expected1)
    tm.assert_numpy_array_equal(index_a == series_c, expected2)

    # cases where length is 1 for one of them
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == index_d
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == series_d
    with pytest.raises(ValueError, match="Lengths must match"):
        index_a == array_d
    msg = "Can only compare identically-labeled Series objects"
    with pytest.raises(ValueError, match=msg):
        series_a == series_d
    with pytest.raises(ValueError, match="Lengths must match"):
        series_a == array_d

    # comparing with a scalar should broadcast; note that we are excluding
    # MultiIndex because in this case each item in the index is a tuple of
    # length 2, and therefore is considered an array of length 2 in the
    # comparison instead of a scalar
    if not isinstance(index_a, MultiIndex):
        expected3 = np.array([False] * (len(index_a) - 2) + [True, False])
        # assuming the 2nd to last item is unique in the data
        item = index_a[-2]
        tm.assert_numpy_array_equal(index_a == item, expected3)
        tm.assert_series_equal(series_a == item, Series(expected3))


def test_compare_tuple():
    # GH#21517
    mi = MultiIndex.from_product([[1, 2]] * 2)

    all_false = np.array([False, False, False, False])

    result = mi == mi[0]
    expected = np.array([True, False, False, False])
    tm.assert_numpy_array_equal(result, expected)

    result = mi != mi[0]
    tm.assert_numpy_array_equal(result, ~expected)

    result = mi < mi[0]
    tm.assert_numpy_array_equal(result, all_false)

    result = mi <= mi[0]
    tm.assert_numpy_array_equal(result, expected)

    result = mi > mi[0]
    tm.assert_numpy_array_equal(result, ~expected)

    result = mi >= mi[0]
    tm.assert_numpy_array_equal(result, ~all_false)


def test_compare_tuple_strs():
    # GH#34180

    mi = MultiIndex.from_tuples([("a", "b"), ("b", "c"), ("c", "a")])

    result = mi == ("c", "a")
    expected = np.array([False, False, True])
    tm.assert_numpy_array_equal(result, expected)

    result = mi == ("c",)
    expected = np.array([False, False, False])
    tm.assert_numpy_array_equal(result, expected)


def test_equals_multi(idx):
    assert idx.equals(idx)
    assert not idx.equals(idx.values)
    assert idx.equals(Index(idx.values))

    assert idx.equal_levels(idx)
    assert not idx.equals(idx[:-1])
    assert not idx.equals(idx[-1])

    # different number of levels
    index = MultiIndex(
        levels=[Index(list(range(4))), Index(list(range(4))), Index(list(range(4)))],
        codes=[
            np.array([0, 0, 1, 2, 2, 2, 3, 3]),
            np.array([0, 1, 0, 0, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        ],
    )

    index2 = MultiIndex(levels=index.levels[:-1], codes=index.codes[:-1])
    assert not index.equals(index2)
    assert not index.equal_levels(index2)

    # levels are different
    major_axis = Index(list(range(4)))
    minor_axis = Index(list(range(2)))

    major_codes = np.array([0, 0, 1, 2, 2, 3])
    minor_codes = np.array([0, 1, 0, 0, 1, 0])

    index = MultiIndex(
        levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
    )
    assert not idx.equals(index)
    assert not idx.equal_levels(index)

    # some of the labels are different
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 2, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])

    index = MultiIndex(
        levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
    )
    assert not idx.equals(index)


def test_identical(idx):
    mi = idx.copy()
    mi2 = idx.copy()
    assert mi.identical(mi2)

    mi = mi.set_names(["new1", "new2"])
    assert mi.equals(mi2)
    assert not mi.identical(mi2)

    mi2 = mi2.set_names(["new1", "new2"])
    assert mi.identical(mi2)

    mi4 = Index(mi.tolist(), tupleize_cols=False)
    assert not mi.identical(mi4)
    assert mi.equals(mi4)


def test_equals_operator(idx):
    # GH9785
    assert (idx == idx).all()


def test_equals_missing_values():
    # make sure take is not using -1
    i = MultiIndex.from_tuples([(0, pd.NaT), (0, pd.Timestamp("20130101"))])
    result = i[0:1].equals(i[0])
    assert not result
    result = i[1:2].equals(i[1])
    assert not result


def test_equals_missing_values_differently_sorted():
    # GH#38439
    mi1 = MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    mi2 = MultiIndex.from_tuples([(np.nan, np.nan), (81.0, np.nan)])
    assert not mi1.equals(mi2)

    mi2 = MultiIndex.from_tuples([(81.0, np.nan), (np.nan, np.nan)])
    assert mi1.equals(mi2)


def test_is_():
    mi = MultiIndex.from_tuples(zip(range(10), range(10)))
    assert mi.is_(mi)
    assert mi.is_(mi.view())
    assert mi.is_(mi.view().view().view().view())
    mi2 = mi.view()
    # names are metadata, they don't change id
    mi2.names = ["A", "B"]
    assert mi2.is_(mi)
    assert mi.is_(mi2)

    assert not mi.is_(mi.set_names(["C", "D"]))
    # levels are inherent properties, they change identity
    mi3 = mi2.set_levels([list(range(10)), list(range(10))])
    assert not mi3.is_(mi2)
    # shouldn't change
    assert mi2.is_(mi)
    mi4 = mi3.view()

    # GH 17464 - Remove duplicate MultiIndex levels
    mi4 = mi4.set_levels([list(range(10)), list(range(10))])
    assert not mi4.is_(mi3)
    mi5 = mi.view()
    mi5 = mi5.set_levels(mi5.levels)
    assert not mi5.is_(mi)


def test_is_all_dates(idx):
    assert not idx._is_all_dates


def test_is_numeric(idx):
    # MultiIndex is never numeric
    assert not is_any_real_numeric_dtype(idx)


def test_multiindex_compare():
    # GH 21149
    # Ensure comparison operations for MultiIndex with nlevels == 1
    # behave consistently with those for MultiIndex with nlevels > 1

    midx = MultiIndex.from_product([[0, 1]])

    # Equality self-test: MultiIndex object vs self
    expected = Series([True, True])
    result = Series(midx == midx)
    tm.assert_series_equal(result, expected)

    # Greater than comparison: MultiIndex object vs self
    expected = Series([False, False])
    result = Series(midx > midx)
    tm.assert_series_equal(result, expected)


def test_equals_ea_int_regular_int():
    # GH#46026
    mi1 = MultiIndex.from_arrays([Index([1, 2], dtype="Int64"), [3, 4]])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]])
    assert not mi1.equals(mi2)
    assert not mi2.equals(mi1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\tests\test_dispatcher.py ===
from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
                                         MethodDispatcher, halt_ordering,
                                         restart_ordering,
                                         ambiguity_register_error_ignore_dup)
from sympy.testing.pytest import raises, warns


def identity(x):
    return x


def inc(x):
    return x + 1


def dec(x):
    return x - 1


def test_dispatcher():
    f = Dispatcher('f')
    f.add((int,), inc)
    f.add((float,), dec)

    with warns(DeprecationWarning, test_stacklevel=False):
        assert f.resolve((int,)) == inc
    assert f.dispatch(int) is inc

    assert f(1) == 2
    assert f(1.0) == 0.0


def test_union_types():
    f = Dispatcher('f')
    f.register((int, float))(inc)

    assert f(1) == 2
    assert f(1.0) == 2.0


def test_dispatcher_as_decorator():
    f = Dispatcher('f')

    @f.register(int)
    def inc(x): # noqa:F811
        return x + 1

    @f.register(float) # noqa:F811
    def inc(x): # noqa:F811
        return x - 1

    assert f(1) == 2
    assert f(1.0) == 0.0


def test_register_instance_method():

    class Test:
        __init__ = MethodDispatcher('f')

        @__init__.register(list)
        def _init_list(self, data):
            self.data = data

        @__init__.register(object)
        def _init_obj(self, datum):
            self.data = [datum]

    a = Test(3)
    b = Test([3])
    assert a.data == b.data


def test_on_ambiguity():
    f = Dispatcher('f')

    def identity(x): return x

    ambiguities = [False]

    def on_ambiguity(dispatcher, amb):
        ambiguities[0] = True

    f.add((object, object), identity, on_ambiguity=on_ambiguity)
    assert not ambiguities[0]
    f.add((object, float), identity, on_ambiguity=on_ambiguity)
    assert not ambiguities[0]
    f.add((float, object), identity, on_ambiguity=on_ambiguity)
    assert ambiguities[0]


def test_raise_error_on_non_class():
    f = Dispatcher('f')
    assert raises(TypeError, lambda: f.add((1,), inc))


def test_docstring():

    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x + y

    def three(x, y):
        return x + y

    master_doc = 'Doc of the multimethod itself'

    f = Dispatcher('f', doc=master_doc)
    f.add((object, object), one)
    f.add((int, int), two)
    f.add((float, float), three)

    assert one.__doc__.strip() in f.__doc__
    assert two.__doc__.strip() in f.__doc__
    assert f.__doc__.find(one.__doc__.strip()) < \
        f.__doc__.find(two.__doc__.strip())
    assert 'object, object' in f.__doc__
    assert master_doc in f.__doc__


def test_help():
    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x + y

    def three(x, y):
        """ Docstring number three """
        return x + y

    master_doc = 'Doc of the multimethod itself'

    f = Dispatcher('f', doc=master_doc)
    f.add((object, object), one)
    f.add((int, int), two)
    f.add((float, float), three)

    assert f._help(1, 1) == two.__doc__
    assert f._help(1.0, 2.0) == three.__doc__


def test_source():
    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x - y

    master_doc = 'Doc of the multimethod itself'

    f = Dispatcher('f', doc=master_doc)
    f.add((int, int), one)
    f.add((float, float), two)

    assert 'x + y' in f._source(1, 1)
    assert 'x - y' in f._source(1.0, 1.0)


def test_source_raises_on_missing_function():
    f = Dispatcher('f')

    assert raises(TypeError, lambda: f.source(1))


def test_halt_method_resolution():
    g = [0]

    def on_ambiguity(a, b):
        g[0] += 1

    f = Dispatcher('f')

    halt_ordering()

    def func(*args):
        pass

    f.add((int, object), func)
    f.add((object, int), func)

    assert g == [0]

    restart_ordering(on_ambiguity=on_ambiguity)

    assert g == [1]

    assert set(f.ordering) == {(int, object), (object, int)}


def test_no_implementations():
    f = Dispatcher('f')
    assert raises(NotImplementedError, lambda: f('hello'))


def test_register_stacking():
    f = Dispatcher('f')

    @f.register(list)
    @f.register(tuple)
    def rev(x):
        return x[::-1]

    assert f((1, 2, 3)) == (3, 2, 1)
    assert f([1, 2, 3]) == [3, 2, 1]

    assert raises(NotImplementedError, lambda: f('hello'))
    assert rev('hello') == 'olleh'


def test_dispatch_method():
    f = Dispatcher('f')

    @f.register(list)
    def rev(x):
        return x[::-1]

    @f.register(int, int)
    def add(x, y):
        return x + y

    class MyList(list):
        pass

    assert f.dispatch(list) is rev
    assert f.dispatch(MyList) is rev
    assert f.dispatch(int, int) is add


def test_not_implemented():
    f = Dispatcher('f')

    @f.register(object)
    def _(x):
        return 'default'

    @f.register(int)
    def _(x):
        if x % 2 == 0:
            return 'even'
        else:
            raise MDNotImplementedError()

    assert f('hello') == 'default'  # default behavior
    assert f(2) == 'even'          # specialized behavior
    assert f(3) == 'default'       # fall bac to default behavior
    assert raises(NotImplementedError, lambda: f(1, 2))


def test_not_implemented_error():
    f = Dispatcher('f')

    @f.register(float)
    def _(a):
        raise MDNotImplementedError()

    assert raises(NotImplementedError, lambda: f(1.0))

def test_ambiguity_register_error_ignore_dup():
    f = Dispatcher('f')

    class A:
        pass
    class B(A):
        pass
    class C(A):
        pass

    # suppress warning for registering ambiguous signal
    f.add((A, B), lambda x,y: None, ambiguity_register_error_ignore_dup)
    f.add((B, A), lambda x,y: None, ambiguity_register_error_ignore_dup)
    f.add((A, C), lambda x,y: None, ambiguity_register_error_ignore_dup)
    f.add((C, A), lambda x,y: None, ambiguity_register_error_ignore_dup)

    # raises error if ambiguous signal is passed
    assert raises(NotImplementedError, lambda: f(B(), C()))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_tz_convert.py ===
from datetime import datetime

import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import timezones

from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Timestamp,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestTZConvert:
    def test_tz_convert_nat(self):
        # GH#5546
        dates = [NaT]
        idx = DatetimeIndex(dates)
        idx = idx.tz_localize("US/Pacific")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Pacific"))
        idx = idx.tz_convert("US/Eastern")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Eastern"))
        idx = idx.tz_convert("UTC")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="UTC"))

        dates = ["2010-12-01 00:00", "2010-12-02 00:00", NaT]
        idx = DatetimeIndex(dates)
        idx = idx.tz_localize("US/Pacific")
        tm.assert_index_equal(idx, DatetimeIndex(dates, tz="US/Pacific"))
        idx = idx.tz_convert("US/Eastern")
        expected = ["2010-12-01 03:00", "2010-12-02 03:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))

        idx = idx + offsets.Hour(5)
        expected = ["2010-12-01 08:00", "2010-12-02 08:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))
        idx = idx.tz_convert("US/Pacific")
        expected = ["2010-12-01 05:00", "2010-12-02 05:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Pacific"))

        idx = idx + np.timedelta64(3, "h")
        expected = ["2010-12-01 08:00", "2010-12-02 08:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Pacific"))

        idx = idx.tz_convert("US/Eastern")
        expected = ["2010-12-01 11:00", "2010-12-02 11:00", NaT]
        tm.assert_index_equal(idx, DatetimeIndex(expected, tz="US/Eastern"))

    @pytest.mark.parametrize("prefix", ["", "dateutil/"])
    def test_dti_tz_convert_compat_timestamp(self, prefix):
        strdates = ["1/1/2012", "3/1/2012", "4/1/2012"]
        idx = DatetimeIndex(strdates, tz=prefix + "US/Eastern")

        conv = idx[0].tz_convert(prefix + "US/Pacific")
        expected = idx.tz_convert(prefix + "US/Pacific")[0]

        assert conv == expected

    def test_dti_tz_convert_hour_overflow_dst(self):
        # Regression test for GH#13306

        # sorted case US/Eastern -> UTC
        ts = ["2008-05-12 09:50:00", "2008-12-12 09:50:35", "2009-05-12 09:50:32"]
        tt = DatetimeIndex(ts).tz_localize("US/Eastern")
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # sorted case UTC -> US/Eastern
        ts = ["2008-05-12 13:50:00", "2008-12-12 14:50:35", "2009-05-12 13:50:32"]
        tt = DatetimeIndex(ts).tz_localize("UTC")
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case US/Eastern -> UTC
        ts = ["2008-05-12 09:50:00", "2008-12-12 09:50:35", "2008-05-12 09:50:32"]
        tt = DatetimeIndex(ts).tz_localize("US/Eastern")
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case UTC -> US/Eastern
        ts = ["2008-05-12 13:50:00", "2008-12-12 14:50:35", "2008-05-12 13:50:32"]
        tt = DatetimeIndex(ts).tz_localize("UTC")
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_tz_convert_hour_overflow_dst_timestamps(self, tz):
        # Regression test for GH#13306

        # sorted case US/Eastern -> UTC
        ts = [
            Timestamp("2008-05-12 09:50:00", tz=tz),
            Timestamp("2008-12-12 09:50:35", tz=tz),
            Timestamp("2009-05-12 09:50:32", tz=tz),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # sorted case UTC -> US/Eastern
        ts = [
            Timestamp("2008-05-12 13:50:00", tz="UTC"),
            Timestamp("2008-12-12 14:50:35", tz="UTC"),
            Timestamp("2009-05-12 13:50:32", tz="UTC"),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case US/Eastern -> UTC
        ts = [
            Timestamp("2008-05-12 09:50:00", tz=tz),
            Timestamp("2008-12-12 09:50:35", tz=tz),
            Timestamp("2008-05-12 09:50:32", tz=tz),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("UTC")
        expected = Index([13, 14, 13], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

        # unsorted case UTC -> US/Eastern
        ts = [
            Timestamp("2008-05-12 13:50:00", tz="UTC"),
            Timestamp("2008-12-12 14:50:35", tz="UTC"),
            Timestamp("2008-05-12 13:50:32", tz="UTC"),
        ]
        tt = DatetimeIndex(ts)
        ut = tt.tz_convert("US/Eastern")
        expected = Index([9, 9, 9], dtype=np.int32)
        tm.assert_index_equal(ut.hour, expected)

    @pytest.mark.parametrize("freq, n", [("h", 1), ("min", 60), ("s", 3600)])
    def test_dti_tz_convert_trans_pos_plus_1__bug(self, freq, n):
        # Regression test for tslib.tz_convert(vals, tz1, tz2).
        # See GH#4496 for details.
        idx = date_range(datetime(2011, 3, 26, 23), datetime(2011, 3, 27, 1), freq=freq)
        idx = idx.tz_localize("UTC")
        idx = idx.tz_convert("Europe/Moscow")

        expected = np.repeat(np.array([3, 4, 5]), np.array([n, n, 1]))
        tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

    def test_dti_tz_convert_dst(self):
        for freq, n in [("h", 1), ("min", 60), ("s", 3600)]:
            # Start DST
            idx = date_range(
                "2014-03-08 23:00", "2014-03-09 09:00", freq=freq, tz="UTC"
            )
            idx = idx.tz_convert("US/Eastern")
            expected = np.repeat(
                np.array([18, 19, 20, 21, 22, 23, 0, 1, 3, 4, 5]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            idx = date_range(
                "2014-03-08 18:00", "2014-03-09 05:00", freq=freq, tz="US/Eastern"
            )
            idx = idx.tz_convert("UTC")
            expected = np.repeat(
                np.array([23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            # End DST
            idx = date_range(
                "2014-11-01 23:00", "2014-11-02 09:00", freq=freq, tz="UTC"
            )
            idx = idx.tz_convert("US/Eastern")
            expected = np.repeat(
                np.array([19, 20, 21, 22, 23, 0, 1, 1, 2, 3, 4]),
                np.array([n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

            idx = date_range(
                "2014-11-01 18:00", "2014-11-02 05:00", freq=freq, tz="US/Eastern"
            )
            idx = idx.tz_convert("UTC")
            expected = np.repeat(
                np.array([22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                np.array([n, n, n, n, n, n, n, n, n, n, n, n, 1]),
            )
            tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))

        # daily
        # Start DST
        idx = date_range("2014-03-08 00:00", "2014-03-09 00:00", freq="D", tz="UTC")
        idx = idx.tz_convert("US/Eastern")
        tm.assert_index_equal(idx.hour, Index([19, 19], dtype=np.int32))

        idx = date_range(
            "2014-03-08 00:00", "2014-03-09 00:00", freq="D", tz="US/Eastern"
        )
        idx = idx.tz_convert("UTC")
        tm.assert_index_equal(idx.hour, Index([5, 5], dtype=np.int32))

        # End DST
        idx = date_range("2014-11-01 00:00", "2014-11-02 00:00", freq="D", tz="UTC")
        idx = idx.tz_convert("US/Eastern")
        tm.assert_index_equal(idx.hour, Index([20, 20], dtype=np.int32))

        idx = date_range(
            "2014-11-01 00:00", "2014-11-02 000:00", freq="D", tz="US/Eastern"
        )
        idx = idx.tz_convert("UTC")
        tm.assert_index_equal(idx.hour, Index([4, 4], dtype=np.int32))

    def test_tz_convert_roundtrip(self, tz_aware_fixture):
        tz = tz_aware_fixture
        idx1 = date_range(start="2014-01-01", end="2014-12-31", freq="ME", tz="UTC")
        exp1 = date_range(start="2014-01-01", end="2014-12-31", freq="ME")

        idx2 = date_range(start="2014-01-01", end="2014-12-31", freq="D", tz="UTC")
        exp2 = date_range(start="2014-01-01", end="2014-12-31", freq="D")

        idx3 = date_range(start="2014-01-01", end="2014-03-01", freq="h", tz="UTC")
        exp3 = date_range(start="2014-01-01", end="2014-03-01", freq="h")

        idx4 = date_range(start="2014-08-01", end="2014-10-31", freq="min", tz="UTC")
        exp4 = date_range(start="2014-08-01", end="2014-10-31", freq="min")

        for idx, expected in [(idx1, exp1), (idx2, exp2), (idx3, exp3), (idx4, exp4)]:
            converted = idx.tz_convert(tz)
            reset = converted.tz_convert(None)
            tm.assert_index_equal(reset, expected)
            assert reset.tzinfo is None
            expected = converted.tz_convert("UTC").tz_localize(None)
            expected = expected._with_freq("infer")
            tm.assert_index_equal(reset, expected)

    def test_dti_tz_convert_tzlocal(self):
        # GH#13583
        # tz_convert doesn't affect to internal
        dti = date_range(start="2001-01-01", end="2001-03-01", tz="UTC")
        dti2 = dti.tz_convert(dateutil.tz.tzlocal())
        tm.assert_numpy_array_equal(dti2.asi8, dti.asi8)

        dti = date_range(start="2001-01-01", end="2001-03-01", tz=dateutil.tz.tzlocal())
        dti2 = dti.tz_convert(None)
        tm.assert_numpy_array_equal(dti2.asi8, dti.asi8)

    @pytest.mark.parametrize(
        "tz",
        [
            "US/Eastern",
            "dateutil/US/Eastern",
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
        ],
    )
    def test_dti_tz_convert_utc_to_local_no_modify(self, tz):
        rng = date_range("3/11/2012", "3/12/2012", freq="h", tz="utc")
        rng_eastern = rng.tz_convert(tz)

        # Values are unmodified
        tm.assert_numpy_array_equal(rng.asi8, rng_eastern.asi8)

        assert timezones.tz_compare(rng_eastern.tz, timezones.maybe_get_tz(tz))

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_tz_convert_unsorted(self, tzstr):
        dr = date_range("2012-03-09", freq="h", periods=100, tz="utc")
        dr = dr.tz_convert(tzstr)

        result = dr[::-1].hour
        exp = dr.hour[::-1]
        tm.assert_almost_equal(result, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_matrices.py ===
from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
        OneMatrix, Trace, MatrixSlice, Determinant, BlockMatrix, BlockDiagMatrix)
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL

X = MatrixSymbol('X', 2, 2)
Y = MatrixSymbol('Y', 2, 3)
Z = MatrixSymbol('Z', 2, 2)
A1x1 = MatrixSymbol('A1x1', 1, 1)
B1x1 = MatrixSymbol('B1x1', 1, 1)
C0x0 = MatrixSymbol('C0x0', 0, 0)
V1 = MatrixSymbol('V1', 2, 1)
V2 = MatrixSymbol('V2', 2, 1)

def test_square():
    assert ask(Q.square(X))
    assert not ask(Q.square(Y))
    assert ask(Q.square(Y*Y.T))

def test_invertible():
    assert ask(Q.invertible(X), Q.invertible(X))
    assert ask(Q.invertible(Y)) is False
    assert ask(Q.invertible(X*Y), Q.invertible(X)) is False
    assert ask(Q.invertible(X*Z), Q.invertible(X)) is None
    assert ask(Q.invertible(X*Z), Q.invertible(X) & Q.invertible(Z)) is True
    assert ask(Q.invertible(X.T)) is None
    assert ask(Q.invertible(X.T), Q.invertible(X)) is True
    assert ask(Q.invertible(X.I)) is True
    assert ask(Q.invertible(Identity(3))) is True
    assert ask(Q.invertible(ZeroMatrix(3, 3))) is False
    assert ask(Q.invertible(OneMatrix(1, 1))) is True
    assert ask(Q.invertible(OneMatrix(3, 3))) is False
    assert ask(Q.invertible(X), Q.fullrank(X) & Q.square(X))

def test_singular():
    assert ask(Q.singular(X)) is None
    assert ask(Q.singular(X), Q.invertible(X)) is False
    assert ask(Q.singular(X), ~Q.invertible(X)) is True

@XFAIL
def test_invertible_fullrank():
    assert ask(Q.invertible(X), Q.fullrank(X)) is True


def test_invertible_BlockMatrix():
    assert ask(Q.invertible(BlockMatrix([Identity(3)]))) == True
    assert ask(Q.invertible(BlockMatrix([ZeroMatrix(3, 3)]))) == False

    X = Matrix([[1, 2, 3], [3, 5, 4]])
    Y = Matrix([[4, 2, 7], [2, 3, 5]])
    # non-invertible A block
    assert ask(Q.invertible(BlockMatrix([
        [Matrix.ones(3, 3), Y.T],
        [X, Matrix.eye(2)],
    ]))) == True
    # non-invertible B block
    assert ask(Q.invertible(BlockMatrix([
        [Y.T, Matrix.ones(3, 3)],
        [Matrix.eye(2), X],
    ]))) == True
    # non-invertible C block
    assert ask(Q.invertible(BlockMatrix([
        [X, Matrix.eye(2)],
        [Matrix.ones(3, 3), Y.T],
    ]))) == True
    # non-invertible D block
    assert ask(Q.invertible(BlockMatrix([
        [Matrix.eye(2), X],
        [Y.T, Matrix.ones(3, 3)],
    ]))) == True


def test_invertible_BlockDiagMatrix():
    assert ask(Q.invertible(BlockDiagMatrix(Identity(3), Identity(5)))) == True
    assert ask(Q.invertible(BlockDiagMatrix(ZeroMatrix(3, 3), Identity(5)))) == False
    assert ask(Q.invertible(BlockDiagMatrix(Identity(3), OneMatrix(5, 5)))) == False


def test_symmetric():
    assert ask(Q.symmetric(X), Q.symmetric(X))
    assert ask(Q.symmetric(X*Z), Q.symmetric(X)) is None
    assert ask(Q.symmetric(X*Z), Q.symmetric(X) & Q.symmetric(Z)) is True
    assert ask(Q.symmetric(X + Z), Q.symmetric(X) & Q.symmetric(Z)) is True
    assert ask(Q.symmetric(Y)) is False
    assert ask(Q.symmetric(Y*Y.T)) is True
    assert ask(Q.symmetric(Y.T*X*Y)) is None
    assert ask(Q.symmetric(Y.T*X*Y), Q.symmetric(X)) is True
    assert ask(Q.symmetric(X**10), Q.symmetric(X)) is True
    assert ask(Q.symmetric(A1x1)) is True
    assert ask(Q.symmetric(A1x1 + B1x1)) is True
    assert ask(Q.symmetric(A1x1 * B1x1)) is True
    assert ask(Q.symmetric(V1.T*V1)) is True
    assert ask(Q.symmetric(V1.T*(V1 + V2))) is True
    assert ask(Q.symmetric(V1.T*(V1 + V2) + A1x1)) is True
    assert ask(Q.symmetric(MatrixSlice(Y, (0, 1), (1, 2)))) is True
    assert ask(Q.symmetric(Identity(3))) is True
    assert ask(Q.symmetric(ZeroMatrix(3, 3))) is True
    assert ask(Q.symmetric(OneMatrix(3, 3))) is True

def _test_orthogonal_unitary(predicate):
    assert ask(predicate(X), predicate(X))
    assert ask(predicate(X.T), predicate(X)) is True
    assert ask(predicate(X.I), predicate(X)) is True
    assert ask(predicate(X**2), predicate(X))
    assert ask(predicate(Y)) is False
    assert ask(predicate(X)) is None
    assert ask(predicate(X), ~Q.invertible(X)) is False
    assert ask(predicate(X*Z*X), predicate(X) & predicate(Z)) is True
    assert ask(predicate(Identity(3))) is True
    assert ask(predicate(ZeroMatrix(3, 3))) is False
    assert ask(Q.invertible(X), predicate(X))
    assert not ask(predicate(X + Z), predicate(X) & predicate(Z))

def test_orthogonal():
    _test_orthogonal_unitary(Q.orthogonal)

def test_unitary():
    _test_orthogonal_unitary(Q.unitary)
    assert ask(Q.unitary(X), Q.orthogonal(X))

def test_fullrank():
    assert ask(Q.fullrank(X), Q.fullrank(X))
    assert ask(Q.fullrank(X**2), Q.fullrank(X))
    assert ask(Q.fullrank(X.T), Q.fullrank(X)) is True
    assert ask(Q.fullrank(X)) is None
    assert ask(Q.fullrank(Y)) is None
    assert ask(Q.fullrank(X*Z), Q.fullrank(X) & Q.fullrank(Z)) is True
    assert ask(Q.fullrank(Identity(3))) is True
    assert ask(Q.fullrank(ZeroMatrix(3, 3))) is False
    assert ask(Q.fullrank(OneMatrix(1, 1))) is True
    assert ask(Q.fullrank(OneMatrix(3, 3))) is False
    assert ask(Q.invertible(X), ~Q.fullrank(X)) == False


def test_positive_definite():
    assert ask(Q.positive_definite(X), Q.positive_definite(X))
    assert ask(Q.positive_definite(X.T), Q.positive_definite(X)) is True
    assert ask(Q.positive_definite(X.I), Q.positive_definite(X)) is True
    assert ask(Q.positive_definite(Y)) is False
    assert ask(Q.positive_definite(X)) is None
    assert ask(Q.positive_definite(X**3), Q.positive_definite(X))
    assert ask(Q.positive_definite(X*Z*X),
            Q.positive_definite(X) & Q.positive_definite(Z)) is True
    assert ask(Q.positive_definite(X), Q.orthogonal(X))
    assert ask(Q.positive_definite(Y.T*X*Y),
            Q.positive_definite(X) & Q.fullrank(Y)) is True
    assert not ask(Q.positive_definite(Y.T*X*Y), Q.positive_definite(X))
    assert ask(Q.positive_definite(Identity(3))) is True
    assert ask(Q.positive_definite(ZeroMatrix(3, 3))) is False
    assert ask(Q.positive_definite(OneMatrix(1, 1))) is True
    assert ask(Q.positive_definite(OneMatrix(3, 3))) is False
    assert ask(Q.positive_definite(X + Z), Q.positive_definite(X) &
            Q.positive_definite(Z)) is True
    assert not ask(Q.positive_definite(-X), Q.positive_definite(X))
    assert ask(Q.positive(X[1, 1]), Q.positive_definite(X))

def test_triangular():
    assert ask(Q.upper_triangular(X + Z.T + Identity(2)), Q.upper_triangular(X) &
            Q.lower_triangular(Z)) is True
    assert ask(Q.upper_triangular(X*Z.T), Q.upper_triangular(X) &
            Q.lower_triangular(Z)) is True
    assert ask(Q.lower_triangular(Identity(3))) is True
    assert ask(Q.lower_triangular(ZeroMatrix(3, 3))) is True
    assert ask(Q.upper_triangular(ZeroMatrix(3, 3))) is True
    assert ask(Q.lower_triangular(OneMatrix(1, 1))) is True
    assert ask(Q.upper_triangular(OneMatrix(1, 1))) is True
    assert ask(Q.lower_triangular(OneMatrix(3, 3))) is False
    assert ask(Q.upper_triangular(OneMatrix(3, 3))) is False
    assert ask(Q.triangular(X), Q.unit_triangular(X))
    assert ask(Q.upper_triangular(X**3), Q.upper_triangular(X))
    assert ask(Q.lower_triangular(X**3), Q.lower_triangular(X))


def test_diagonal():
    assert ask(Q.diagonal(X + Z.T + Identity(2)), Q.diagonal(X) &
               Q.diagonal(Z)) is True
    assert ask(Q.diagonal(ZeroMatrix(3, 3)))
    assert ask(Q.diagonal(OneMatrix(1, 1))) is True
    assert ask(Q.diagonal(OneMatrix(3, 3))) is False
    assert ask(Q.lower_triangular(X) & Q.upper_triangular(X), Q.diagonal(X))
    assert ask(Q.diagonal(X), Q.lower_triangular(X) & Q.upper_triangular(X))
    assert ask(Q.symmetric(X), Q.diagonal(X))
    assert ask(Q.triangular(X), Q.diagonal(X))
    assert ask(Q.diagonal(C0x0))
    assert ask(Q.diagonal(A1x1))
    assert ask(Q.diagonal(A1x1 + B1x1))
    assert ask(Q.diagonal(A1x1*B1x1))
    assert ask(Q.diagonal(V1.T*V2))
    assert ask(Q.diagonal(V1.T*(X + Z)*V1))
    assert ask(Q.diagonal(MatrixSlice(Y, (0, 1), (1, 2)))) is True
    assert ask(Q.diagonal(V1.T*(V1 + V2))) is True
    assert ask(Q.diagonal(X**3), Q.diagonal(X))
    assert ask(Q.diagonal(Identity(3)))
    assert ask(Q.diagonal(DiagMatrix(V1)))
    assert ask(Q.diagonal(DiagonalMatrix(X)))


def test_non_atoms():
    assert ask(Q.real(Trace(X)), Q.positive(Trace(X)))

@XFAIL
def test_non_trivial_implies():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    assert ask(Q.lower_triangular(X+Y), Q.lower_triangular(X) &
               Q.lower_triangular(Y)) is True
    assert ask(Q.triangular(X), Q.lower_triangular(X)) is True
    assert ask(Q.triangular(X+Y), Q.lower_triangular(X) &
               Q.lower_triangular(Y)) is True

def test_MatrixSlice():
    X = MatrixSymbol('X', 4, 4)
    B = MatrixSlice(X, (1, 3), (1, 3))
    C = MatrixSlice(X, (0, 3), (1, 3))
    assert ask(Q.symmetric(B), Q.symmetric(X))
    assert ask(Q.invertible(B), Q.invertible(X))
    assert ask(Q.diagonal(B), Q.diagonal(X))
    assert ask(Q.orthogonal(B), Q.orthogonal(X))
    assert ask(Q.upper_triangular(B), Q.upper_triangular(X))

    assert not ask(Q.symmetric(C), Q.symmetric(X))
    assert not ask(Q.invertible(C), Q.invertible(X))
    assert not ask(Q.diagonal(C), Q.diagonal(X))
    assert not ask(Q.orthogonal(C), Q.orthogonal(X))
    assert not ask(Q.upper_triangular(C), Q.upper_triangular(X))

def test_det_trace_positive():
    X = MatrixSymbol('X', 4, 4)
    assert ask(Q.positive(Trace(X)), Q.positive_definite(X))
    assert ask(Q.positive(Determinant(X)), Q.positive_definite(X))

def test_field_assumptions():
    X = MatrixSymbol('X', 4, 4)
    Y = MatrixSymbol('Y', 4, 4)
    assert ask(Q.real_elements(X), Q.real_elements(X))
    assert not ask(Q.integer_elements(X), Q.real_elements(X))
    assert ask(Q.complex_elements(X), Q.real_elements(X))
    assert ask(Q.complex_elements(X**2), Q.real_elements(X))
    assert ask(Q.real_elements(X**2), Q.integer_elements(X))
    assert ask(Q.real_elements(X+Y), Q.real_elements(X)) is None
    assert ask(Q.real_elements(X+Y), Q.real_elements(X) & Q.real_elements(Y))
    from sympy.matrices.expressions.hadamard import HadamardProduct
    assert ask(Q.real_elements(HadamardProduct(X, Y)),
                    Q.real_elements(X) & Q.real_elements(Y))
    assert ask(Q.complex_elements(X+Y), Q.real_elements(X) & Q.complex_elements(Y))

    assert ask(Q.real_elements(X.T), Q.real_elements(X))
    assert ask(Q.real_elements(X.I), Q.real_elements(X) & Q.invertible(X))
    assert ask(Q.real_elements(Trace(X)), Q.real_elements(X))
    assert ask(Q.integer_elements(Determinant(X)), Q.integer_elements(X))
    assert not ask(Q.integer_elements(X.I), Q.integer_elements(X))
    alpha = Symbol('alpha')
    assert ask(Q.real_elements(alpha*X), Q.real_elements(X) & Q.real(alpha))
    assert ask(Q.real_elements(LofLU(X)), Q.real_elements(X))
    e = Symbol('e', integer=True, negative=True)
    assert ask(Q.real_elements(X**e), Q.real_elements(X) & Q.invertible(X))
    assert ask(Q.real_elements(X**e), Q.real_elements(X)) is None

def test_matrix_element_sets():
    X = MatrixSymbol('X', 4, 4)
    assert ask(Q.real(X[1, 2]), Q.real_elements(X))
    assert ask(Q.integer(X[1, 2]), Q.integer_elements(X))
    assert ask(Q.complex(X[1, 2]), Q.complex_elements(X))
    assert ask(Q.integer_elements(Identity(3)))
    assert ask(Q.integer_elements(ZeroMatrix(3, 3)))
    assert ask(Q.integer_elements(OneMatrix(3, 3)))
    from sympy.matrices.expressions.fourier import DFT
    assert ask(Q.complex_elements(DFT(3)))


def test_matrix_element_sets_slices_blocks():
    X = MatrixSymbol('X', 4, 4)
    assert ask(Q.integer_elements(X[:, 3]), Q.integer_elements(X))
    assert ask(Q.integer_elements(BlockMatrix([[X], [X]])),
                        Q.integer_elements(X))

def test_matrix_element_sets_determinant_trace():
    assert ask(Q.integer(Determinant(X)), Q.integer_elements(X))
    assert ask(Q.integer(Trace(X)), Q.integer_elements(X))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\string_\test_string_arrow.py ===
import pickle
import re

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
    StringArray,
    StringDtype,
)
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,
    ArrowStringArrayNumpySemantics,
)


def test_eq_all_na():
    pytest.importorskip("pyarrow")
    a = pd.array([pd.NA, pd.NA], dtype=StringDtype("pyarrow"))
    result = a == a
    expected = pd.array([pd.NA, pd.NA], dtype="boolean[pyarrow]")
    tm.assert_extension_array_equal(result, expected)


def test_config(string_storage, using_infer_string):
    # with the default string_storage setting
    # always "python" at the moment
    assert StringDtype().storage == "python"

    with pd.option_context("string_storage", string_storage):
        assert StringDtype().storage == string_storage
        result = pd.array(["a", "b"])
        assert result.dtype.storage == string_storage

    # pd.array(..) by default always returns the NA-variant
    dtype = StringDtype(string_storage, na_value=pd.NA)
    expected = dtype.construct_array_type()._from_sequence(["a", "b"], dtype=dtype)
    tm.assert_equal(result, expected)


def test_config_bad_storage_raises():
    msg = re.escape("Value must be one of python|pyarrow")
    with pytest.raises(ValueError, match=msg):
        pd.options.mode.string_storage = "foo"


@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize("array_lib", ["numpy", "pyarrow"])
def test_constructor_not_string_type_raises(array_lib, chunked):
    pa = pytest.importorskip("pyarrow")

    array_lib = pa if array_lib == "pyarrow" else np

    arr = array_lib.array([1, 2, 3])
    if chunked:
        if array_lib is np:
            pytest.skip("chunked not applicable to numpy array")
        arr = pa.chunked_array(arr)
    if array_lib is np:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"
    else:
        msg = re.escape(
            "ArrowStringArray requires a PyArrow (chunked) array of large_string type"
        )
    with pytest.raises(ValueError, match=msg):
        ArrowStringArray(arr)


@pytest.mark.parametrize("chunked", [True, False])
def test_constructor_not_string_type_value_dictionary_raises(chunked):
    pa = pytest.importorskip("pyarrow")

    arr = pa.array([1, 2, 3], pa.dictionary(pa.int32(), pa.int32()))
    if chunked:
        arr = pa.chunked_array(arr)

    msg = re.escape(
        "ArrowStringArray requires a PyArrow (chunked) array of large_string type"
    )
    with pytest.raises(ValueError, match=msg):
        ArrowStringArray(arr)


@pytest.mark.parametrize("string_type", ["string", "large_string"])
@pytest.mark.parametrize("chunked", [True, False])
def test_constructor_valid_string_type_value_dictionary(string_type, chunked):
    pa = pytest.importorskip("pyarrow")

    arr = pa.array(["1", "2", "3"], getattr(pa, string_type)()).dictionary_encode()
    if chunked:
        arr = pa.chunked_array(arr)

    arr = ArrowStringArray(arr)
    # dictionary type get converted to dense large string array
    assert pa.types.is_large_string(arr._pa_array.type)


@pytest.mark.parametrize("chunked", [True, False])
def test_constructor_valid_string_view(chunked):
    # requires pyarrow>=18 for casting string_view to string
    pa = pytest.importorskip("pyarrow", minversion="18")

    arr = pa.array(["1", "2", "3"], pa.string_view())
    if chunked:
        arr = pa.chunked_array(arr)

    arr = ArrowStringArray(arr)
    # dictionary type get converted to dense large string array
    assert pa.types.is_large_string(arr._pa_array.type)


def test_constructor_from_list():
    # GH#27673
    pytest.importorskip("pyarrow")
    result = pd.Series(["E"], dtype=StringDtype(storage="pyarrow"))
    assert isinstance(result.dtype, StringDtype)
    assert result.dtype.storage == "pyarrow"


def test_from_sequence_wrong_dtype_raises(using_infer_string):
    pytest.importorskip("pyarrow")
    with pd.option_context("string_storage", "python"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string")

    with pd.option_context("string_storage", "pyarrow"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string")

    with pytest.raises(AssertionError, match=None):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string[python]")

    ArrowStringArray._from_sequence(["a", None, "c"], dtype="string[pyarrow]")

    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            with pd.option_context("string_storage", "python"):
                ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    with pd.option_context("string_storage", "pyarrow"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            ArrowStringArray._from_sequence(
                ["a", None, "c"], dtype=StringDtype("python")
            )

    ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype("pyarrow"))

    with pd.option_context("string_storage", "python"):
        StringArray._from_sequence(["a", None, "c"], dtype="string")

    with pd.option_context("string_storage", "pyarrow"):
        StringArray._from_sequence(["a", None, "c"], dtype="string")

    StringArray._from_sequence(["a", None, "c"], dtype="string[python]")

    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(["a", None, "c"], dtype="string[pyarrow]")

    if not using_infer_string:
        with pd.option_context("string_storage", "python"):
            StringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            with pd.option_context("string_storage", "pyarrow"):
                StringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    StringArray._from_sequence(["a", None, "c"], dtype=StringDtype("python"))

    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(["a", None, "c"], dtype=StringDtype("pyarrow"))


@td.skip_if_installed("pyarrow")
def test_pyarrow_not_installed_raises():
    msg = re.escape("pyarrow>=10.0.1 is required for PyArrow backed")

    with pytest.raises(ImportError, match=msg):
        StringDtype(storage="pyarrow")

    with pytest.raises(ImportError, match=msg):
        ArrowStringArray([])

    with pytest.raises(ImportError, match=msg):
        ArrowStringArrayNumpySemantics([])

    with pytest.raises(ImportError, match=msg):
        ArrowStringArray._from_sequence(["a", None, "b"])


@pytest.mark.parametrize("multiple_chunks", [False, True])
@pytest.mark.parametrize(
    "key, value, expected",
    [
        (-1, "XX", ["a", "b", "c", "d", "XX"]),
        (1, "XX", ["a", "XX", "c", "d", "e"]),
        (1, None, ["a", None, "c", "d", "e"]),
        (1, pd.NA, ["a", None, "c", "d", "e"]),
        ([1, 3], "XX", ["a", "XX", "c", "XX", "e"]),
        ([1, 3], ["XX", "YY"], ["a", "XX", "c", "YY", "e"]),
        ([1, 3], ["XX", None], ["a", "XX", "c", None, "e"]),
        ([1, 3], ["XX", pd.NA], ["a", "XX", "c", None, "e"]),
        ([0, -1], ["XX", "YY"], ["XX", "b", "c", "d", "YY"]),
        ([-1, 0], ["XX", "YY"], ["YY", "b", "c", "d", "XX"]),
        (slice(3, None), "XX", ["a", "b", "c", "XX", "XX"]),
        (slice(2, 4), ["XX", "YY"], ["a", "b", "XX", "YY", "e"]),
        (slice(3, 1, -1), ["XX", "YY"], ["a", "b", "YY", "XX", "e"]),
        (slice(None), "XX", ["XX", "XX", "XX", "XX", "XX"]),
        ([False, True, False, True, False], ["XX", "YY"], ["a", "XX", "c", "YY", "e"]),
    ],
)
def test_setitem(multiple_chunks, key, value, expected):
    pa = pytest.importorskip("pyarrow")

    result = pa.array(list("abcde"))
    expected = pa.array(expected)

    if multiple_chunks:
        result = pa.chunked_array([result[:3], result[3:]])
        expected = pa.chunked_array([expected[:3], expected[3:]])

    result = ArrowStringArray(result)
    expected = ArrowStringArray(expected)

    result[key] = value
    tm.assert_equal(result, expected)


def test_setitem_invalid_indexer_raises():
    pa = pytest.importorskip("pyarrow")

    arr = ArrowStringArray(pa.array(list("abcde")))

    with pytest.raises(IndexError, match=None):
        arr[5] = "foo"

    with pytest.raises(IndexError, match=None):
        arr[-6] = "foo"

    with pytest.raises(IndexError, match=None):
        arr[[0, 5]] = "foo"

    with pytest.raises(IndexError, match=None):
        arr[[0, -6]] = "foo"

    with pytest.raises(IndexError, match=None):
        arr[[True, True, False]] = "foo"

    with pytest.raises(ValueError, match=None):
        arr[[0, 1]] = ["foo", "bar", "baz"]


@pytest.mark.parametrize("na_value", [pd.NA, np.nan])
def test_pickle_roundtrip(na_value):
    # GH 42600
    pytest.importorskip("pyarrow")
    dtype = StringDtype("pyarrow", na_value=na_value)
    expected = pd.Series(range(10), dtype=dtype)
    expected_sliced = expected.head(2)
    full_pickled = pickle.dumps(expected)
    sliced_pickled = pickle.dumps(expected_sliced)

    assert len(full_pickled) > len(sliced_pickled)

    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)

    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)


def test_string_dtype_error_message():
    # GH#55051
    pytest.importorskip("pyarrow")
    msg = "Storage must be 'python' or 'pyarrow'."
    with pytest.raises(ValueError, match=msg):
        StringDtype("bla")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_values.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestDataFrameValues:
    @td.skip_array_manager_invalid_test
    def test_values(self, float_frame, using_copy_on_write):
        if using_copy_on_write:
            with pytest.raises(ValueError, match="read-only"):
                float_frame.values[:, 0] = 5.0
            assert (float_frame.values[:, 0] != 5).all()
        else:
            float_frame.values[:, 0] = 5.0
            assert (float_frame.values[:, 0] == 5).all()

    def test_more_values(self, float_string_frame):
        values = float_string_frame.values
        assert values.shape[1] == len(float_string_frame.columns)

    def test_values_mixed_dtypes(self, float_frame, float_string_frame):
        frame = float_frame
        arr = frame.values

        frame_cols = frame.columns
        for i, row in enumerate(arr):
            for j, value in enumerate(row):
                col = frame_cols[j]
                if np.isnan(value):
                    assert np.isnan(frame[col].iloc[i])
                else:
                    assert value == frame[col].iloc[i]

        # mixed type
        arr = float_string_frame[["foo", "A"]].values
        assert arr[0, 0] == "bar"

        df = DataFrame({"complex": [1j, 2j, 3j], "real": [1, 2, 3]})
        arr = df.values
        assert arr[0, 0] == 1j

    def test_values_duplicates(self):
        df = DataFrame(
            [[1, 2, "a", "b"], [1, 2, "a", "b"]], columns=["one", "one", "two", "two"]
        )

        result = df.values
        expected = np.array([[1, 2, "a", "b"], [1, 2, "a", "b"]], dtype=object)

        tm.assert_numpy_array_equal(result, expected)

    def test_values_with_duplicate_columns(self):
        df = DataFrame([[1, 2.5], [3, 4.5]], index=[1, 2], columns=["x", "x"])
        result = df.values
        expected = np.array([[1, 2.5], [3, 4.5]])
        assert (result == expected).all().all()

    @pytest.mark.parametrize("constructor", [date_range, period_range])
    def test_values_casts_datetimelike_to_object(self, constructor):
        series = Series(constructor("2000-01-01", periods=10, freq="D"))

        expected = series.astype("object")

        df = DataFrame(
            {"a": series, "b": np.random.default_rng(2).standard_normal(len(series))}
        )

        result = df.values.squeeze()
        assert (result[:, 0] == expected.values).all()

        df = DataFrame({"a": series, "b": ["foo"] * len(series)})

        result = df.values.squeeze()
        assert (result[:, 0] == expected.values).all()

    def test_frame_values_with_tz(self):
        tz = "US/Central"
        df = DataFrame({"A": date_range("2000", periods=4, tz=tz)})
        result = df.values
        expected = np.array(
            [
                [Timestamp("2000-01-01", tz=tz)],
                [Timestamp("2000-01-02", tz=tz)],
                [Timestamp("2000-01-03", tz=tz)],
                [Timestamp("2000-01-04", tz=tz)],
            ]
        )
        tm.assert_numpy_array_equal(result, expected)

        # two columns, homogeneous

        df["B"] = df["A"]
        result = df.values
        expected = np.concatenate([expected, expected], axis=1)
        tm.assert_numpy_array_equal(result, expected)

        # three columns, heterogeneous
        est = "US/Eastern"
        df["C"] = df["A"].dt.tz_convert(est)

        new = np.array(
            [
                [Timestamp("2000-01-01T01:00:00", tz=est)],
                [Timestamp("2000-01-02T01:00:00", tz=est)],
                [Timestamp("2000-01-03T01:00:00", tz=est)],
                [Timestamp("2000-01-04T01:00:00", tz=est)],
            ]
        )
        expected = np.concatenate([expected, new], axis=1)
        result = df.values
        tm.assert_numpy_array_equal(result, expected)

    def test_interleave_with_tzaware(self, timezone_frame):
        # interleave with object
        result = timezone_frame.assign(D="foo").values
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
                ["foo", "foo", "foo"],
            ],
            dtype=object,
        ).T
        tm.assert_numpy_array_equal(result, expected)

        # interleave with only datetime64[ns]
        result = timezone_frame.values
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        tm.assert_numpy_array_equal(result, expected)

    def test_values_interleave_non_unique_cols(self):
        df = DataFrame(
            [[Timestamp("20130101"), 3.5], [Timestamp("20130102"), 4.5]],
            columns=["x", "x"],
            index=[1, 2],
        )

        df_unique = df.copy()
        df_unique.columns = ["x", "y"]
        assert df_unique.values.shape == df.values.shape
        tm.assert_numpy_array_equal(df_unique.values[0], df.values[0])
        tm.assert_numpy_array_equal(df_unique.values[1], df.values[1])

    def test_values_numeric_cols(self, float_frame):
        float_frame["foo"] = "bar"

        values = float_frame[["A", "B", "C", "D"]].values
        assert values.dtype == np.float64

    def test_values_lcd(self, mixed_float_frame, mixed_int_frame):
        # mixed lcd
        values = mixed_float_frame[["A", "B", "C", "D"]].values
        assert values.dtype == np.float64

        values = mixed_float_frame[["A", "B", "C"]].values
        assert values.dtype == np.float32

        values = mixed_float_frame[["C"]].values
        assert values.dtype == np.float16

        # GH#10364
        # B uint64 forces float because there are other signed int types
        values = mixed_int_frame[["A", "B", "C", "D"]].values
        assert values.dtype == np.float64

        values = mixed_int_frame[["A", "D"]].values
        assert values.dtype == np.int64

        # B uint64 forces float because there are other signed int types
        values = mixed_int_frame[["A", "B", "C"]].values
        assert values.dtype == np.float64

        # as B and C are both unsigned, no forcing to float is needed
        values = mixed_int_frame[["B", "C"]].values
        assert values.dtype == np.uint64

        values = mixed_int_frame[["A", "C"]].values
        assert values.dtype == np.int32

        values = mixed_int_frame[["C", "D"]].values
        assert values.dtype == np.int64

        values = mixed_int_frame[["A"]].values
        assert values.dtype == np.int32

        values = mixed_int_frame[["C"]].values
        assert values.dtype == np.uint8


class TestPrivateValues:
    @td.skip_array_manager_invalid_test
    def test_private_values_dt64tz(self, using_copy_on_write):
        dta = date_range("2000", periods=4, tz="US/Central")._data.reshape(-1, 1)

        df = DataFrame(dta, columns=["A"])
        tm.assert_equal(df._values, dta)

        if using_copy_on_write:
            assert not np.shares_memory(df._values._ndarray, dta._ndarray)
        else:
            # we have a view
            assert np.shares_memory(df._values._ndarray, dta._ndarray)

        # TimedeltaArray
        tda = dta - dta
        df2 = df - df
        tm.assert_equal(df2._values, tda)

    @td.skip_array_manager_invalid_test
    def test_private_values_dt64tz_multicol(self, using_copy_on_write):
        dta = date_range("2000", periods=8, tz="US/Central")._data.reshape(-1, 2)

        df = DataFrame(dta, columns=["A", "B"])
        tm.assert_equal(df._values, dta)

        if using_copy_on_write:
            assert not np.shares_memory(df._values._ndarray, dta._ndarray)
        else:
            # we have a view
            assert np.shares_memory(df._values._ndarray, dta._ndarray)

        # TimedeltaArray
        tda = dta - dta
        df2 = df - df
        tm.assert_equal(df2._values, tda)

    def test_private_values_dt64_multiblock(self):
        dta = date_range("2000", periods=8)._data

        df = DataFrame({"A": dta[:4]}, copy=False)
        df["B"] = dta[4:]

        assert len(df._mgr.arrays) == 2

        result = df._values
        expected = dta.reshape(2, 4).T
        tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_determinant.py ===
import random
import pytest
from sympy.core.numbers import I
from sympy.core.numbers import Rational
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.matrices import Matrix, eye, ones
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.functions.combinatorial.factorials import factorial, subfactorial


@pytest.mark.parametrize("method", [
    # Evaluating these directly because they are never reached via M.det()
    Matrix._eval_det_bareiss, Matrix._eval_det_berkowitz,
    Matrix._eval_det_bird, Matrix._eval_det_laplace, Matrix._eval_det_lu
])
@pytest.mark.parametrize("M, sol", [
    (Matrix(), 1),
    (Matrix([[0]]), 0),
    (Matrix([[5]]), 5),
])
def test_eval_determinant(method, M, sol):
    assert method(M) == sol


@pytest.mark.parametrize("method", [
    "domain-ge", "bareiss", "berkowitz", "bird", "laplace", "lu"])
@pytest.mark.parametrize("M, sol", [
    (Matrix(( (-3,  2),
              ( 8, -5) )), -1),
    (Matrix(( (x,   1),
              (y, 2*y) )), 2*x*y - y),
    (Matrix(( (1, 1, 1),
              (1, 2, 3),
              (1, 3, 6) )), 1),
    (Matrix(( ( 3, -2,  0, 5),
              (-2,  1, -2, 2),
              ( 0, -2,  5, 0),
              ( 5,  0,  3, 4) )), -289),
    (Matrix(( ( 1,  2,  3,  4),
              ( 5,  6,  7,  8),
              ( 9, 10, 11, 12),
              (13, 14, 15, 16) )), 0),
    (Matrix(( (3, 2, 0, 0, 0),
              (0, 3, 2, 0, 0),
              (0, 0, 3, 2, 0),
              (0, 0, 0, 3, 2),
              (2, 0, 0, 0, 3) )), 275),
    (Matrix(( ( 3,  0,  0, 0),
              (-2,  1,  0, 0),
              ( 0, -2,  5, 0),
              ( 5,  0,  3, 4) )), 60),
    (Matrix(( ( 1,  0,  0,  0),
              ( 5,  0,  0,  0),
              ( 9, 10, 11, 0),
              (13, 14, 15, 16) )), 0),
    (Matrix(( (3, 2, 0, 0, 0),
              (0, 3, 2, 0, 0),
              (0, 0, 3, 2, 0),
              (0, 0, 0, 3, 2),
              (0, 0, 0, 0, 3) )), 243),
    (Matrix(( (1, 0,  1,  2, 12),
              (2, 0,  1,  1,  4),
              (2, 1,  1, -1,  3),
              (3, 2, -1,  1,  8),
              (1, 1,  1,  0,  6) )), -55),
    (Matrix(( (-5,  2,  3,  4,  5),
              ( 1, -4,  3,  4,  5),
              ( 1,  2, -3,  4,  5),
              ( 1,  2,  3, -2,  5),
              ( 1,  2,  3,  4, -1) )), 11664),
    (Matrix(( ( 2,  7, -1, 3, 2),
              ( 0,  0,  1, 0, 1),
              (-2,  0,  7, 0, 2),
              (-3, -2,  4, 5, 3),
              ( 1,  0,  0, 0, 1) )), 123),
    (Matrix(( (x, y, z),
              (1, 0, 0),
              (y, z, x) )), z**2 - x*y),
])
def test_determinant(method, M, sol):
    assert M.det(method=method) == sol


def test_issue_13835():
    a = symbols('a')
    M = lambda n: Matrix([[i + a*j for i in range(n)]
                          for j in range(n)])
    assert M(5).det() == 0
    assert M(6).det() == 0
    assert M(7).det() == 0


def test_issue_14517():
    M = Matrix([
        [   0, 10*I,    10*I,       0],
        [10*I,    0,       0,    10*I],
        [10*I,    0, 5 + 2*I,    10*I],
        [   0, 10*I,    10*I, 5 + 2*I]])
    ev = M.eigenvals()
    # test one random eigenvalue, the computation is a little slow
    test_ev = random.choice(list(ev.keys()))
    assert (M - test_ev*eye(4)).det() == 0


@pytest.mark.parametrize("method", [
    "bareis", "det_lu", "det_LU", "Bareis", "BAREISS", "BERKOWITZ", "LU"])
@pytest.mark.parametrize("M, sol", [
    (Matrix(( ( 3, -2,  0, 5),
              (-2,  1, -2, 2),
              ( 0, -2,  5, 0),
              ( 5,  0,  3, 4) )), -289),
    (Matrix(( (-5,  2,  3,  4,  5),
              ( 1, -4,  3,  4,  5),
              ( 1,  2, -3,  4,  5),
              ( 1,  2,  3, -2,  5),
              ( 1,  2,  3,  4, -1) )), 11664),
])
def test_legacy_det(method, M, sol):
    # Minimal support for legacy keys for 'method' in det()
    # Partially copied from test_determinant()
    assert M.det(method=method) == sol


def eye_Determinant(n):
    return Matrix(n, n, lambda i, j: int(i == j))

def zeros_Determinant(n):
    return Matrix(n, n, lambda i, j: 0)

def test_det():
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    raises(NonSquareMatrixError, lambda: a.det())

    z = zeros_Determinant(2)
    ey = eye_Determinant(2)
    assert z.det() == 0
    assert ey.det() == 1

    x = Symbol('x')
    a = Matrix(0, 0, [])
    b = Matrix(1, 1, [5])
    c = Matrix(2, 2, [1, 2, 3, 4])
    d = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 8])
    e = Matrix(4, 4,
        [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])
    from sympy.abc import i, j, k, l, m, n
    f = Matrix(3, 3, [i, l, m, 0, j, n, 0, 0, k])
    g = Matrix(3, 3, [i, 0, 0, l, j, 0, m, n, k])
    h = Matrix(3, 3, [x**3, 0, 0, i, x**-1, 0, j, k, x**-2])
    # the method keyword for `det` doesn't kick in until 4x4 matrices,
    # so there is no need to test all methods on smaller ones

    assert a.det() == 1
    assert b.det() == 5
    assert c.det() == -2
    assert d.det() == 3
    assert e.det() == 4*x - 24
    assert e.det(method="domain-ge") == 4*x - 24
    assert e.det(method='bareiss') == 4*x - 24
    assert e.det(method='berkowitz') == 4*x - 24
    assert f.det() == i*j*k
    assert g.det() == i*j*k
    assert h.det() == 1
    raises(ValueError, lambda: e.det(iszerofunc="test"))

def test_permanent():
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert M.per() == 450
    for i in range(1, 12):
        assert ones(i, i).per() == ones(i, i).T.per() == factorial(i)
        assert (ones(i, i)-eye(i)).per() == (ones(i, i)-eye(i)).T.per() == subfactorial(i)

    a1, a2, a3, a4, a5 = symbols('a_1 a_2 a_3 a_4 a_5')
    M = Matrix([a1, a2, a3, a4, a5])
    assert M.per() == M.T.per() == a1 + a2 + a3 + a4 + a5

def test_adjugate():
    x = Symbol('x')
    e = Matrix(4, 4,
        [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])

    adj = Matrix([
        [   4,         -8,         4,         0],
        [  76, -14*x - 68,  14*x - 8, -4*x + 24],
        [-122, 17*x + 142, -21*x + 4,  8*x - 48],
        [  48,  -4*x - 72,       8*x, -4*x + 24]])
    assert e.adjugate() == adj
    assert e.adjugate(method='bareiss') == adj
    assert e.adjugate(method='berkowitz') == adj
    assert e.adjugate(method='bird') == adj
    assert e.adjugate(method='laplace') == adj

    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    raises(NonSquareMatrixError, lambda: a.adjugate())

def test_util():
    R = Rational

    v1 = Matrix(1, 3, [1, 2, 3])
    v2 = Matrix(1, 3, [3, 4, 5])
    assert v1.norm() == sqrt(14)
    assert v1.project(v2) == Matrix(1, 3, [R(39)/25, R(52)/25, R(13)/5])
    assert Matrix.zeros(1, 2) == Matrix(1, 2, [0, 0])
    assert ones(1, 2) == Matrix(1, 2, [1, 1])
    assert v1.copy() == v1
    # cofactor
    assert eye(3) == eye(3).cofactor_matrix()
    test = Matrix([[1, 3, 2], [2, 6, 3], [2, 3, 6]])
    assert test.cofactor_matrix() == \
        Matrix([[27, -6, -6], [-12, 2, 3], [-3, 1, 0]])
    test = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert test.cofactor_matrix() == \
        Matrix([[-3, 6, -3], [6, -12, 6], [-3, 6, -3]])

def test_cofactor_and_minors():
    x = Symbol('x')
    e = Matrix(4, 4,
        [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])

    m = Matrix([
        [ x,  1,  3],
        [ 2,  9, 11],
        [12, 13, 14]])
    cm = Matrix([
        [ 4,         76,       -122,        48],
        [-8, -14*x - 68, 17*x + 142, -4*x - 72],
        [ 4,   14*x - 8,  -21*x + 4,       8*x],
        [ 0,  -4*x + 24,   8*x - 48, -4*x + 24]])
    sub = Matrix([
            [x, 1,  2],
            [4, 5,  6],
            [2, 9, 10]])

    assert e.minor_submatrix(1, 2) == m
    assert e.minor_submatrix(-1, -1) == sub
    assert e.minor(1, 2) == -17*x - 142
    assert e.cofactor(1, 2) == 17*x + 142
    assert e.cofactor_matrix() == cm
    assert e.cofactor_matrix(method="bareiss") == cm
    assert e.cofactor_matrix(method="berkowitz") == cm
    assert e.cofactor_matrix(method="bird") == cm
    assert e.cofactor_matrix(method="laplace") == cm

    raises(ValueError, lambda: e.cofactor(4, 5))
    raises(ValueError, lambda: e.minor(4, 5))
    raises(ValueError, lambda: e.minor_submatrix(4, 5))

    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    assert a.minor_submatrix(0, 0) == Matrix([[5, 6]])

    raises(ValueError, lambda:
        Matrix(0, 0, []).minor_submatrix(0, 0))
    raises(NonSquareMatrixError, lambda: a.cofactor(0, 0))
    raises(NonSquareMatrixError, lambda: a.minor(0, 0))
    raises(NonSquareMatrixError, lambda: a.cofactor_matrix())

def test_charpoly():
    x, y = Symbol('x'), Symbol('y')
    z, t = Symbol('z'), Symbol('t')

    from sympy.abc import a,b,c

    m = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert eye_Determinant(3).charpoly(x) == Poly((x - 1)**3, x)
    assert eye_Determinant(3).charpoly(y) == Poly((y - 1)**3, y)
    assert m.charpoly() == Poly(x**3 - 15*x**2 - 18*x, x)
    raises(NonSquareMatrixError, lambda: Matrix([[1], [2]]).charpoly())
    n = Matrix(4, 4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert n.charpoly() == Poly(x**4, x)

    n = Matrix(4, 4, [45, 0, 0, 0, 0, 23, 0, 0, 0, 0, 87, 0, 0, 0, 0, 12])
    assert n.charpoly() == Poly(x**4 - 167*x**3 + 8811*x**2 - 173457*x + 1080540, x)

    n = Matrix(3, 3, [x, 0, 0, a, y, 0, b, c, z])
    assert n.charpoly() == Poly(t**3 - (x+y+z)*t**2 + t*(x*y+y*z+x*z) - x*y*z, t)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_mathematica.py ===
from sympy import sin, Function, symbols, Dummy, Lambda, cos
from sympy.parsing.mathematica import parse_mathematica, MathematicaParser
from sympy.core.sympify import sympify
from sympy.abc import n, w, x, y, z
from sympy.testing.pytest import raises


def test_mathematica():
    d = {
        '- 6x': '-6*x',
        'Sin[x]^2': 'sin(x)**2',
        '2(x-1)': '2*(x-1)',
        '3y+8': '3*y+8',
        'ArcSin[2x+9(4-x)^2]/x': 'asin(2*x+9*(4-x)**2)/x',
        'x+y': 'x+y',
        '355/113': '355/113',
        '2.718281828': '2.718281828',
        'Cos(1/2 * π)': 'Cos(π/2)',
        'Sin[12]': 'sin(12)',
        'Exp[Log[4]]': 'exp(log(4))',
        '(x+1)(x+3)': '(x+1)*(x+3)',
        'Cos[ArcCos[3.6]]': 'cos(acos(3.6))',
        'Cos[x]==Sin[y]': 'Eq(cos(x), sin(y))',
        '2*Sin[x+y]': '2*sin(x+y)',
        'Sin[x]+Cos[y]': 'sin(x)+cos(y)',
        'Sin[Cos[x]]': 'sin(cos(x))',
        '2*Sqrt[x+y]': '2*sqrt(x+y)',   # Test case from the issue 4259
        '+Sqrt[2]': 'sqrt(2)',
        '-Sqrt[2]': '-sqrt(2)',
        '-1/Sqrt[2]': '-1/sqrt(2)',
        '-(1/Sqrt[3])': '-(1/sqrt(3))',
        '1/(2*Sqrt[5])': '1/(2*sqrt(5))',
        'Mod[5,3]': 'Mod(5,3)',
        '-Mod[5,3]': '-Mod(5,3)',
        '(x+1)y': '(x+1)*y',
        'x(y+1)': 'x*(y+1)',
        'Sin[x]Cos[y]': 'sin(x)*cos(y)',
        'Sin[x]^2Cos[y]^2': 'sin(x)**2*cos(y)**2',
        'Cos[x]^2(1 - Cos[y]^2)': 'cos(x)**2*(1-cos(y)**2)',
        'x y': 'x*y',
        'x  y': 'x*y',
        '2 x': '2*x',
        'x 8': 'x*8',
        '2 8': '2*8',
        '4.x': '4.*x',
        '4. 3': '4.*3',
        '4. 3.': '4.*3.',
        '1 2 3': '1*2*3',
        ' -  2 *  Sqrt[  2 3 *   ( 1   +  5 ) ]  ': '-2*sqrt(2*3*(1+5))',
        'Log[2,4]': 'log(4,2)',
        'Log[Log[2,4],4]': 'log(4,log(4,2))',
        'Exp[Sqrt[2]^2Log[2, 8]]': 'exp(sqrt(2)**2*log(8,2))',
        'ArcSin[Cos[0]]': 'asin(cos(0))',
        'Log2[16]': 'log(16,2)',
        'Max[1,-2,3,-4]': 'Max(1,-2,3,-4)',
        'Min[1,-2,3]': 'Min(1,-2,3)',
        'Exp[I Pi/2]': 'exp(I*pi/2)',
        'ArcTan[x,y]': 'atan2(y,x)',
        'Pochhammer[x,y]': 'rf(x,y)',
        'ExpIntegralEi[x]': 'Ei(x)',
        'SinIntegral[x]': 'Si(x)',
        'CosIntegral[x]': 'Ci(x)',
        'AiryAi[x]': 'airyai(x)',
        'AiryAiPrime[5]': 'airyaiprime(5)',
        'AiryBi[x]': 'airybi(x)',
        'AiryBiPrime[7]': 'airybiprime(7)',
        'LogIntegral[4]': ' li(4)',
        'PrimePi[7]': 'primepi(7)',
        'Prime[5]': 'prime(5)',
        'PrimeQ[5]': 'isprime(5)',
        'Rational[2,19]': 'Rational(2,19)',    # test case for issue 25716
        }

    for e in d:
        assert parse_mathematica(e) == sympify(d[e])

    # The parsed form of this expression should not evaluate the Lambda object:
    assert parse_mathematica("Sin[#]^2 + Cos[#]^2 &[x]") == sin(x)**2 + cos(x)**2

    d1, d2, d3 = symbols("d1:4", cls=Dummy)
    assert parse_mathematica("Sin[#] + Cos[#3] &").dummy_eq(Lambda((d1, d2, d3), sin(d1) + cos(d3)))
    assert parse_mathematica("Sin[#^2] &").dummy_eq(Lambda(d1, sin(d1**2)))
    assert parse_mathematica("Function[x, x^3]") == Lambda(x, x**3)
    assert parse_mathematica("Function[{x, y}, x^2 + y^2]") == Lambda((x, y), x**2 + y**2)


def test_parser_mathematica_tokenizer():
    parser = MathematicaParser()

    chain = lambda expr: parser._from_tokens_to_fullformlist(parser._from_mathematica_to_tokens(expr))

    # Basic patterns
    assert chain("x") == "x"
    assert chain("42") == "42"
    assert chain(".2") == ".2"
    assert chain("+x") == "x"
    assert chain("-1") == "-1"
    assert chain("- 3") == "-3"
    assert chain("α") == "α"
    assert chain("+Sin[x]") == ["Sin", "x"]
    assert chain("-Sin[x]") == ["Times", "-1", ["Sin", "x"]]
    assert chain("x(a+1)") == ["Times", "x", ["Plus", "a", "1"]]
    assert chain("(x)") == "x"
    assert chain("(+x)") == "x"
    assert chain("-a") == ["Times", "-1", "a"]
    assert chain("(-x)") == ["Times", "-1", "x"]
    assert chain("(x + y)") == ["Plus", "x", "y"]
    assert chain("3 + 4") == ["Plus", "3", "4"]
    assert chain("a - 3") == ["Plus", "a", "-3"]
    assert chain("a - b") == ["Plus", "a", ["Times", "-1", "b"]]
    assert chain("7 * 8") == ["Times", "7", "8"]
    assert chain("a + b*c") == ["Plus", "a", ["Times", "b", "c"]]
    assert chain("a + b* c* d + 2 * e") == ["Plus", "a", ["Times", "b", "c", "d"], ["Times", "2", "e"]]
    assert chain("a / b") == ["Times", "a", ["Power", "b", "-1"]]

    # Missing asterisk (*) patterns:
    assert chain("x y") == ["Times", "x", "y"]
    assert chain("3 4") == ["Times", "3", "4"]
    assert chain("a[b] c") == ["Times", ["a", "b"], "c"]
    assert chain("(x) (y)") == ["Times", "x", "y"]
    assert chain("3 (a)") == ["Times", "3", "a"]
    assert chain("(a) b") == ["Times", "a", "b"]
    assert chain("4.2") == "4.2"
    assert chain("4 2") == ["Times", "4", "2"]
    assert chain("4  2") == ["Times", "4", "2"]
    assert chain("3 . 4") == ["Dot", "3", "4"]
    assert chain("4. 2") == ["Times", "4.", "2"]
    assert chain("x.y") == ["Dot", "x", "y"]
    assert chain("4.y") == ["Times", "4.", "y"]
    assert chain("4 .y") == ["Dot", "4", "y"]
    assert chain("x.4") == ["Times", "x", ".4"]
    assert chain("x0.3") == ["Times", "x0", ".3"]
    assert chain("x. 4") == ["Dot", "x", "4"]

    # Comments
    assert chain("a (* +b *) + c") == ["Plus", "a", "c"]
    assert chain("a (* + b *) + (**)c (* +d *) + e") == ["Plus", "a", "c", "e"]
    assert chain("""a + (*
    + b
    *) c + (* d
    *) e
    """) == ["Plus", "a", "c", "e"]

    # Operators couples + and -, * and / are mutually associative:
    # (i.e. expression gets flattened when mixing these operators)
    assert chain("a*b/c") == ["Times", "a", "b", ["Power", "c", "-1"]]
    assert chain("a/b*c") == ["Times", "a", ["Power", "b", "-1"], "c"]
    assert chain("a+b-c") == ["Plus", "a", "b", ["Times", "-1", "c"]]
    assert chain("a-b+c") == ["Plus", "a", ["Times", "-1", "b"], "c"]
    assert chain("-a + b -c ") == ["Plus", ["Times", "-1", "a"], "b", ["Times", "-1", "c"]]
    assert chain("a/b/c*d") == ["Times", "a", ["Power", "b", "-1"], ["Power", "c", "-1"], "d"]
    assert chain("a/b/c") == ["Times", "a", ["Power", "b", "-1"], ["Power", "c", "-1"]]
    assert chain("a-b-c") == ["Plus", "a", ["Times", "-1", "b"], ["Times", "-1", "c"]]
    assert chain("1/a") == ["Times", "1", ["Power", "a", "-1"]]
    assert chain("1/a/b") == ["Times", "1", ["Power", "a", "-1"], ["Power", "b", "-1"]]
    assert chain("-1/a*b") == ["Times", "-1", ["Power", "a", "-1"], "b"]

    # Enclosures of various kinds, i.e. ( )  [ ]  [[ ]]  { }
    assert chain("(a + b) + c") == ["Plus", ["Plus", "a", "b"], "c"]
    assert chain(" a + (b + c) + d ") == ["Plus", "a", ["Plus", "b", "c"], "d"]
    assert chain("a * (b + c)") == ["Times", "a", ["Plus", "b", "c"]]
    assert chain("a b (c d)") == ["Times", "a", "b", ["Times", "c", "d"]]
    assert chain("{a, b, 2, c}") == ["List", "a", "b", "2", "c"]
    assert chain("{a, {b, c}}") == ["List", "a", ["List", "b", "c"]]
    assert chain("{{a}}") == ["List", ["List", "a"]]
    assert chain("a[b, c]") == ["a", "b", "c"]
    assert chain("a[[b, c]]") == ["Part", "a", "b", "c"]
    assert chain("a[b[c]]") == ["a", ["b", "c"]]
    assert chain("a[[b, c[[d, {e,f}]]]]") == ["Part", "a", "b", ["Part", "c", "d", ["List", "e", "f"]]]
    assert chain("a[b[[c,d]]]") == ["a", ["Part", "b", "c", "d"]]
    assert chain("a[[b[c]]]") == ["Part", "a", ["b", "c"]]
    assert chain("a[[b[[c]]]]") == ["Part", "a", ["Part", "b", "c"]]
    assert chain("a[[b[c[[d]]]]]") == ["Part", "a", ["b", ["Part", "c", "d"]]]
    assert chain("a[b[[c[d]]]]") == ["a", ["Part", "b", ["c", "d"]]]
    assert chain("x[[a+1, b+2, c+3]]") == ["Part", "x", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]
    assert chain("x[a+1, b+2, c+3]") == ["x", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]
    assert chain("{a+1, b+2, c+3}") == ["List", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]

    # Flat operator:
    assert chain("a*b*c*d*e") == ["Times", "a", "b", "c", "d", "e"]
    assert chain("a +b + c+ d+e") == ["Plus", "a", "b", "c", "d", "e"]

    # Right priority operator:
    assert chain("a^b") == ["Power", "a", "b"]
    assert chain("a^b^c") == ["Power", "a", ["Power", "b", "c"]]
    assert chain("a^b^c^d") == ["Power", "a", ["Power", "b", ["Power", "c", "d"]]]

    # Left priority operator:
    assert chain("a/.b") == ["ReplaceAll", "a", "b"]
    assert chain("a/.b/.c/.d") == ["ReplaceAll", ["ReplaceAll", ["ReplaceAll", "a", "b"], "c"], "d"]

    assert chain("a//b") == ["a", "b"]
    assert chain("a//b//c") == [["a", "b"], "c"]
    assert chain("a//b//c//d") == [[["a", "b"], "c"], "d"]

    # Compound expressions
    assert chain("a;b") == ["CompoundExpression", "a", "b"]
    assert chain("a;") == ["CompoundExpression", "a", "Null"]
    assert chain("a;b;") == ["CompoundExpression", "a", "b", "Null"]
    assert chain("a[b;c]") == ["a", ["CompoundExpression", "b", "c"]]
    assert chain("a[b,c;d,e]") == ["a", "b", ["CompoundExpression", "c", "d"], "e"]
    assert chain("a[b,c;,d]") == ["a", "b", ["CompoundExpression", "c", "Null"], "d"]

    # New lines
    assert chain("a\nb\n") == ["CompoundExpression", "a", "b"]
    assert chain("a\n\nb\n (c \nd)  \n") == ["CompoundExpression", "a", "b", ["Times", "c", "d"]]
    assert chain("\na; b\nc") == ["CompoundExpression", "a", "b", "c"]
    assert chain("a + \nb\n") == ["Plus", "a", "b"]
    assert chain("a\nb; c; d\n e; (f \n g); h + \n i") == ["CompoundExpression", "a", "b", "c", "d", "e", ["Times", "f", "g"], ["Plus", "h", "i"]]
    assert chain("\n{\na\nb; c; d\n e (f \n g); h + \n i\n\n}\n") == ["List", ["CompoundExpression", ["Times", "a", "b"], "c", ["Times", "d", "e", ["Times", "f", "g"]], ["Plus", "h", "i"]]]

    # Patterns
    assert chain("y_") == ["Pattern", "y", ["Blank"]]
    assert chain("y_.") == ["Optional", ["Pattern", "y", ["Blank"]]]
    assert chain("y__") == ["Pattern", "y", ["BlankSequence"]]
    assert chain("y___") == ["Pattern", "y", ["BlankNullSequence"]]
    assert chain("a[b_.,c_]") == ["a", ["Optional", ["Pattern", "b", ["Blank"]]], ["Pattern", "c", ["Blank"]]]
    assert chain("b_. c") == ["Times", ["Optional", ["Pattern", "b", ["Blank"]]], "c"]

    # Slots for lambda functions
    assert chain("#") == ["Slot", "1"]
    assert chain("#3") == ["Slot", "3"]
    assert chain("#n") == ["Slot", "n"]
    assert chain("##") == ["SlotSequence", "1"]
    assert chain("##a") == ["SlotSequence", "a"]

    # Lambda functions
    assert chain("x&") == ["Function", "x"]
    assert chain("#&") == ["Function", ["Slot", "1"]]
    assert chain("#+3&") == ["Function", ["Plus", ["Slot", "1"], "3"]]
    assert chain("#1 + #2&") == ["Function", ["Plus", ["Slot", "1"], ["Slot", "2"]]]
    assert chain("# + #&") == ["Function", ["Plus", ["Slot", "1"], ["Slot", "1"]]]
    assert chain("#&[x]") == [["Function", ["Slot", "1"]], "x"]
    assert chain("#1 + #2 & [x, y]") == [["Function", ["Plus", ["Slot", "1"], ["Slot", "2"]]], "x", "y"]
    assert chain("#1^2#2^3&") == ["Function", ["Times", ["Power", ["Slot", "1"], "2"], ["Power", ["Slot", "2"], "3"]]]

    # Strings inside Mathematica expressions:
    assert chain('"abc"') == ["_Str", "abc"]
    assert chain('"a\\"b"') == ["_Str", 'a"b']
    # This expression does not make sense mathematically, it's just testing the parser:
    assert chain('x + "abc" ^ 3') == ["Plus", "x", ["Power", ["_Str", "abc"], "3"]]
    assert chain('"a (* b *) c"') == ["_Str", "a (* b *) c"]
    assert chain('"a" (* b *) ') == ["_Str", "a"]
    assert chain('"a [ b] "') == ["_Str", "a [ b] "]
    raises(SyntaxError, lambda: chain('"'))
    raises(SyntaxError, lambda: chain('"\\"'))
    raises(SyntaxError, lambda: chain('"abc'))
    raises(SyntaxError, lambda: chain('"abc\\"def'))

    # Invalid expressions:
    raises(SyntaxError, lambda: chain("(,"))
    raises(SyntaxError, lambda: chain("()"))
    raises(SyntaxError, lambda: chain("a (* b"))


def test_parser_mathematica_exp_alt():
    parser = MathematicaParser()

    convert_chain2 = lambda expr: parser._from_fullformlist_to_fullformsympy(parser._from_fullform_to_fullformlist(expr))
    convert_chain3 = lambda expr: parser._from_fullformsympy_to_sympy(convert_chain2(expr))

    Sin, Times, Plus, Power = symbols("Sin Times Plus Power", cls=Function)

    full_form1 = "Sin[Times[x, y]]"
    full_form2 = "Plus[Times[x, y], z]"
    full_form3 = "Sin[Times[x, Plus[y, z], Power[w, n]]]]"
    full_form4 = "Rational[Rational[x, y], z]"

    assert parser._from_fullform_to_fullformlist(full_form1) == ["Sin", ["Times", "x", "y"]]
    assert parser._from_fullform_to_fullformlist(full_form2) == ["Plus", ["Times", "x", "y"], "z"]
    assert parser._from_fullform_to_fullformlist(full_form3) == ["Sin", ["Times", "x", ["Plus", "y", "z"], ["Power", "w", "n"]]]
    assert parser._from_fullform_to_fullformlist(full_form4) == ["Rational", ["Rational", "x", "y"], "z"]

    assert convert_chain2(full_form1) == Sin(Times(x, y))
    assert convert_chain2(full_form2) == Plus(Times(x, y), z)
    assert convert_chain2(full_form3) == Sin(Times(x, Plus(y, z), Power(w, n)))

    assert convert_chain3(full_form1) == sin(x*y)
    assert convert_chain3(full_form2) == x*y + z
    assert convert_chain3(full_form3) == sin(x*(y + z)*w**n)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\interactive\tests\test_ipython.py ===
"""Tests of tools for setting up interactive IPython sessions. """

from sympy.interactive.session import (init_ipython_session,
    enable_automatic_symbols, enable_automatic_int_sympification)

from sympy.core import Symbol, Rational, Integer
from sympy.external import import_module
from sympy.testing.pytest import raises

# TODO: The code below could be made more granular with something like:
#
# @requires('IPython', version=">=1.0")
# def test_automatic_symbols(ipython):

ipython = import_module("IPython", min_module_version="1.0")

if not ipython:
    #bin/test will not execute any tests now
    disabled = True

# WARNING: These tests will modify the existing IPython environment. IPython
# uses a single instance for its interpreter, so there is no way to isolate
# the test from another IPython session. It also means that if this test is
# run twice in the same Python session it will fail. This isn't usually a
# problem because the test suite is run in a subprocess by default, but if the
# tests are run with subprocess=False it can pollute the current IPython
# session. See the discussion in issue #15149.

def test_automatic_symbols():
    # NOTE: Because of the way the hook works, you have to use run_cell(code,
    # True).  This means that the code must have no Out, or it will be printed
    # during the tests.
    app = init_ipython_session()
    app.run_cell("from sympy import *")

    enable_automatic_symbols(app)

    symbol = "verylongsymbolname"
    assert symbol not in app.user_ns
    app.run_cell("a = %s" % symbol, True)
    assert symbol not in app.user_ns
    app.run_cell("a = type(%s)" % symbol, True)
    assert app.user_ns['a'] == Symbol
    app.run_cell("%s = Symbol('%s')" % (symbol, symbol), True)
    assert symbol in app.user_ns

    # Check that built-in names aren't overridden
    app.run_cell("a = all == __builtin__.all", True)
    assert "all" not in app.user_ns
    assert app.user_ns['a'] is True

    # Check that SymPy names aren't overridden
    app.run_cell("import sympy")
    app.run_cell("a = factorial == sympy.factorial", True)
    assert app.user_ns['a'] is True


def test_int_to_Integer():
    # XXX: Warning, don't test with == here.  0.5 == Rational(1, 2) is True!
    app = init_ipython_session()
    app.run_cell("from sympy import Integer")
    app.run_cell("a = 1")
    assert isinstance(app.user_ns['a'], int)

    enable_automatic_int_sympification(app)
    app.run_cell("a = 1/2")
    assert isinstance(app.user_ns['a'], Rational)
    app.run_cell("a = 1")
    assert isinstance(app.user_ns['a'], Integer)
    app.run_cell("a = int(1)")
    assert isinstance(app.user_ns['a'], int)
    app.run_cell("a = (1/\n2)")
    assert app.user_ns['a'] == Rational(1, 2)
    # TODO: How can we test that the output of a SyntaxError is the original
    # input, not the transformed input?


def test_ipythonprinting():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import Symbol")

    # Printing without printing extension
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] == "pi"
        assert app.user_ns['a2']['text/plain'] == "pi**2"
    else:
        assert app.user_ns['a'][0]['text/plain'] == "pi"
        assert app.user_ns['a2'][0]['text/plain'] == "pi**2"

    # Load printing extension
    app.run_cell("from sympy import init_printing")
    app.run_cell("init_printing()")
    # Printing with printing extension
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] in ('\N{GREEK SMALL LETTER PI}', 'pi')
        assert app.user_ns['a2']['text/plain'] in (' 2\n\N{GREEK SMALL LETTER PI} ', '  2\npi ')
    else:
        assert app.user_ns['a'][0]['text/plain'] in ('\N{GREEK SMALL LETTER PI}', 'pi')
        assert app.user_ns['a2'][0]['text/plain'] in (' 2\n\N{GREEK SMALL LETTER PI} ', '  2\npi ')


def test_print_builtin_option():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import Symbol")
    app.run_cell("from sympy import init_printing")

    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        text = app.user_ns['a'][0]['text/plain']
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    # XXX: How can we make this ignore the terminal width? This test fails if
    # the terminal is too narrow.
    assert text in ("{pi: 3.14, n_i: 3}",
                    '{n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3, \N{GREEK SMALL LETTER PI}: 3.14}',
                    "{n_i: 3, pi: 3.14}",
                    '{\N{GREEK SMALL LETTER PI}: 3.14, n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3}')

    # If we enable the default printing, then the dictionary's should render
    # as a LaTeX version of the whole dict: ${\pi: 3.14, n_i: 3}$
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("init_printing(use_latex=True)")
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        latex = app.user_ns['a']['text/latex']
    else:
        text = app.user_ns['a'][0]['text/plain']
        latex = app.user_ns['a'][0]['text/latex']
    assert text in ("{pi: 3.14, n_i: 3}",
                    '{n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3, \N{GREEK SMALL LETTER PI}: 3.14}',
                    "{n_i: 3, pi: 3.14}",
                    '{\N{GREEK SMALL LETTER PI}: 3.14, n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3}')
    assert latex == r'$\displaystyle \left\{ n_{i} : 3, \  \pi : 3.14\right\}$'

    # Objects with an _latex overload should also be handled by our tuple
    # printer.
    app.run_cell("""\
    class WithOverload:
        def _latex(self, printer):
            return r"\\LaTeX"
    """)
    app.run_cell("a = format((WithOverload(),))")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        latex = app.user_ns['a']['text/latex']
    else:
        latex = app.user_ns['a'][0]['text/latex']
    assert latex == r'$\displaystyle \left( \LaTeX,\right)$'

    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("init_printing(use_latex=True, print_builtin=False)")
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        text = app.user_ns['a'][0]['text/plain']
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    # Note : In Python 3 we have one text type: str which holds Unicode data
    # and two byte types bytes and bytearray.
    # Python 3.3.3 + IPython 0.13.2 gives: '{n_i: 3, pi: 3.14}'
    # Python 3.3.3 + IPython 1.1.0 gives: '{n_i: 3, pi: 3.14}'
    assert text in ("{pi: 3.14, n_i: 3}", "{n_i: 3, pi: 3.14}")


def test_builtin_containers():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("from sympy import init_printing, Matrix")
    app.run_cell('init_printing(use_latex=True, use_unicode=False)')

    # Make sure containers that shouldn't pretty print don't.
    app.run_cell('a = format((True, False))')
    app.run_cell('import sys')
    app.run_cell('b = format(sys.flags)')
    app.run_cell('c = format((Matrix([1, 2]),))')
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] ==  '(True, False)'
        assert 'text/latex' not in app.user_ns['a']
        assert app.user_ns['b']['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b']
        assert app.user_ns['c']['text/plain'] == \
"""\
 [1]  \n\
([ ],)
 [2]  \
"""
        assert app.user_ns['c']['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'
    else:
        assert app.user_ns['a'][0]['text/plain'] ==  '(True, False)'
        assert 'text/latex' not in app.user_ns['a'][0]
        assert app.user_ns['b'][0]['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b'][0]
        assert app.user_ns['c'][0]['text/plain'] == \
"""\
 [1]  \n\
([ ],)
 [2]  \
"""
        assert app.user_ns['c'][0]['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'

def test_matplotlib_bad_latex():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("import IPython")
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import init_printing, Matrix")
    app.run_cell("init_printing(use_latex='matplotlib')")

    # The png formatter is not enabled by default in this context
    app.run_cell("inst.display_formatter.formatters['image/png'].enabled = True")

    # Make sure no warnings are raised by IPython
    app.run_cell("import warnings")
    # IPython.core.formatters.FormatterWarning was introduced in IPython 2.0
    if int(ipython.__version__.split(".")[0]) < 2:
        app.run_cell("warnings.simplefilter('error')")
    else:
        app.run_cell("warnings.simplefilter('error', IPython.core.formatters.FormatterWarning)")

    # This should not raise an exception
    app.run_cell("a = format(Matrix([1, 2, 3]))")

    # issue 9799
    app.run_cell("from sympy import Piecewise, Symbol, Eq")
    app.run_cell("x = Symbol('x'); pw = format(Piecewise((1, Eq(x, 0)), (0, True)))")


def test_override_repr_latex():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("import IPython")
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("from sympy import init_printing")
    app.run_cell("from sympy import Symbol")
    app.run_cell("init_printing(use_latex=True)")
    app.run_cell("""\
    class SymbolWithOverload(Symbol):
        def _repr_latex_(self):
            return r"Hello " + super()._repr_latex_() + " world"
    """)
    app.run_cell("a = format(SymbolWithOverload('s'))")

    if int(ipython.__version__.split(".")[0]) < 1:
        latex = app.user_ns['a']['text/latex']
    else:
        latex = app.user_ns['a'][0]['text/latex']
    assert latex == r'Hello $\displaystyle s$ world'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_failing_integrals.py ===
# A collection of failing integrals from the issues.

from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sech, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, cos, sin, tan)
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.fu import fu


from sympy.testing.pytest import XFAIL, slow, tooslow

from sympy.abc import x, k, c, y, b, h, a, m, z, n, t


@tooslow
@XFAIL
def test_issue_3880():
    # integrate_hyperexponential(Poly(t*2*(1 - t0**2)*t0*(x**3 + x**2), t), Poly((1 + t0**2)**2*2*(x**2 + x + 1), t), [Poly(1, x), Poly(1 + t0**2, t0), Poly(t, t)], [x, t0, t], [exp, tan])
    assert not integrate(exp(x)*cos(2*x)*sin(2*x) * (x**3 + x**2)/(2*(x**2 + x + 1)), x).has(Integral)


def test_issue_4212_real():
    xr = symbols('xr', real=True)
    negabsx = Piecewise((-xr, xr < 0), (xr, True))
    assert integrate(sign(xr), xr) == negabsx


@XFAIL
def test_issue_4212():
    # XXX: Maybe this should be expected to fail without real assumptions on x.
    # As a complex function sign(x) is not analytic and so there is no complex
    # function whose complex derivative is sign(x). With real assumptions this
    # works (see test_issue_4212_real above).
    assert not integrate(sign(x), x).has(Integral)


def test_issue_4511():
    # This works, but gives a slightly over-complicated answer.
    f = integrate(cos(x)**2 / (1 - sin(x)), x)
    assert fu(f) == x - cos(x) - 1
    assert f == ((x*tan(x/2)**2 + x - 2)/(tan(x/2)**2 + 1)).expand()


def test_integrate_DiracDelta_no_meijerg():
    assert integrate(integrate(integrate(
        DiracDelta(x - y - z), (z, 0, oo)), (y, 0, 1), meijerg=False), (x, 0, 1)) == S.Half


@XFAIL
def test_integrate_DiracDelta_fails():
    # issue 6427
    # works without meijerg. See test_integrate_DiracDelta_no_meijerg above.
    assert integrate(integrate(integrate(
        DiracDelta(x - y - z), (z, 0, oo)), (y, 0, 1)), (x, 0, 1)) == S.Half


@XFAIL
@slow
def test_issue_4525():
    # Warning: takes a long time
    assert not integrate((x**m * (1 - x)**n * (a + b*x + c*x**2))/(1 + x**2), (x, 0, 1)).has(Integral)


@XFAIL
@tooslow
def test_issue_4540():
    # Note, this integral is probably nonelementary
    assert not integrate(
        (sin(1/x) - x*exp(x)) /
        ((-sin(1/x) + x*exp(x))*x + x*sin(1/x)), x).has(Integral)


@XFAIL
@slow
def test_issue_4891():
    # Requires the hypergeometric function.
    assert not integrate(cos(x)**y, x).has(Integral)


@XFAIL
@slow
def test_issue_1796a():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), x).has(Integral)


@XFAIL
def test_issue_4895b():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), (x, -oo, 0)).has(Integral)


@XFAIL
def test_issue_4895c():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), (x, -oo, oo)).has(Integral)


@XFAIL
def test_issue_4895d():
    assert not integrate(exp(2*b*x)*exp(-a*x**2), (x, 0, oo)).has(Integral)


@XFAIL
@slow
def test_issue_4941():
    assert not integrate(sqrt(1 + sinh(x/20)**2), (x, -25, 25)).has(Integral)


@XFAIL
def test_issue_4992():
    # Nonelementary integral.  Requires hypergeometric/Meijer-G handling.
    assert not integrate(log(x) * x**(k - 1) * exp(-x) / gamma(k), (x, 0, oo)).has(Integral)


@XFAIL
def test_issue_16396a():
    i = integrate(1/(1+sqrt(tan(x))), (x, pi/3, pi/6))
    assert not i.has(Integral)


@XFAIL
def test_issue_16396b():
    i = integrate(x*sin(x)/(1+cos(x)**2), (x, 0, pi))
    assert not i.has(Integral)


@XFAIL
def test_issue_16046():
    assert integrate(exp(exp(I*x)), [x, 0, 2*pi]) == 2*pi


@XFAIL
def test_issue_15925a():
    assert not integrate(sqrt((1+sin(x))**2+(cos(x))**2), (x, -pi/2, pi/2)).has(Integral)


def test_issue_15925b():
    f = sqrt((-12*cos(x)**2*sin(x))**2+(12*cos(x)*sin(x)**2)**2)
    assert integrate(f, (x, 0, pi/6)) == Rational(3, 2)


@XFAIL
def test_issue_15925b_manual():
    assert not integrate(sqrt((-12*cos(x)**2*sin(x))**2+(12*cos(x)*sin(x)**2)**2),
                         (x, 0, pi/6), manual=True).has(Integral)


@XFAIL
@tooslow
def test_issue_15227():
    i = integrate(log(1-x)*log((1+x)**2)/x, (x, 0, 1))
    assert not i.has(Integral)
    # assert i == -5*zeta(3)/4


@XFAIL
@slow
def test_issue_14716():
    i = integrate(log(x + 5)*cos(pi*x),(x, S.Half, 1))
    assert not i.has(Integral)
    # Mathematica can not solve it either, but
    # integrate(log(x + 5)*cos(pi*x),(x, S.Half, 1)).transform(x, y - 5).doit()
    # works
    # assert i == -log(Rational(11, 2))/pi - Si(pi*Rational(11, 2))/pi + Si(6*pi)/pi


@XFAIL
def test_issue_14709a():
    i = integrate(x*acos(1 - 2*x/h), (x, 0, h))
    assert not i.has(Integral)
    # assert i == 5*h**2*pi/16


@slow
@XFAIL
def test_issue_14398():
    assert not integrate(exp(x**2)*cos(x), x).has(Integral)


@XFAIL
def test_issue_14074():
    i = integrate(log(sin(x)), (x, 0, pi/2))
    assert not i.has(Integral)
    # assert i == -pi*log(2)/2


@XFAIL
@slow
def test_issue_14078b():
    i = integrate((atan(4*x)-atan(2*x))/x, (x, 0, oo))
    assert not i.has(Integral)
    # assert i == pi*log(2)/2


@XFAIL
def test_issue_13792():
    i =  integrate(log(1/x) / (1 - x), (x, 0, 1))
    assert not i.has(Integral)
    # assert i in [polylog(2, -exp_polar(I*pi)), pi**2/6]


@XFAIL
def test_issue_11845a():
    assert not integrate(exp(y - x**3), (x, 0, 1)).has(Integral)


@XFAIL
def test_issue_11845b():
    assert not integrate(exp(-y - x**3), (x, 0, 1)).has(Integral)


@XFAIL
def test_issue_11813():
    assert not integrate((a - x)**Rational(-1, 2)*x, (x, 0, a)).has(Integral)


@XFAIL
def test_issue_11254c():
    assert not integrate(sech(x)**2, (x, 0, 1)).has(Integral)


@XFAIL
def test_issue_10584():
    assert not integrate(sqrt(x**2 + 1/x**2), x).has(Integral)


@XFAIL
def test_issue_9101():
    assert not integrate(log(x + sqrt(x**2 + y**2 + z**2)), z).has(Integral)


@XFAIL
def test_issue_7147():
    assert not integrate(x/sqrt(a*x**2 + b*x + c)**3, x).has(Integral)


@XFAIL
def test_issue_7109():
    assert not integrate(sqrt(a**2/(a**2 - x**2)), x).has(Integral)


@XFAIL
def test_integrate_Piecewise_rational_over_reals():
    f = Piecewise(
        (0,                                              t - 478.515625*pi <  0),
        (13.2075145209219*pi/(0.000871222*t + 0.995)**2, t - 478.515625*pi >= 0))

    assert abs((integrate(f, (t, 0, oo)) - 15235.9375*pi).evalf()) <= 1e-7


@XFAIL
def test_issue_4311_slow():
    # Not slow when bypassing heurish
    assert not integrate(x*abs(9-x**2), x).has(Integral)

@XFAIL
def test_issue_20370():
    a = symbols('a', positive=True)
    assert integrate((1 + a * cos(x))**-1, (x, 0, 2 * pi)) == (2 * pi / sqrt(1 - a**2))


@XFAIL
def test_polylog():
    # log(1/x)*log(x+1)-polylog(2, -x)
    assert not integrate(log(1/x)/(x + 1), x).has(Integral)


@XFAIL
def test_polylog_manual():
    # Make sure _parts_rule does not go into an infinite loop here
    assert not integrate(log(1/x)/(x + 1), x, manual=True).has(Integral)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reductions\test_stat_reductions.py ===
"""
Tests for statistical reductions of 2nd moment or higher: var, skew, kurt, ...
"""
import inspect

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDatetimeLikeStatReductions:
    @pytest.mark.parametrize("box", [Series, pd.Index, pd.array])
    def test_dt64_mean(self, tz_naive_fixture, box):
        tz = tz_naive_fixture

        dti = date_range("2001-01-01", periods=11, tz=tz)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        dtarr = dti._data

        obj = box(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06", tz=tz)
        assert obj.mean(skipna=False) == pd.Timestamp("2001-01-06", tz=tz)

        # dtarr[-2] will be the first date 2001-01-1
        dtarr[-2] = pd.NaT

        obj = box(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06 07:12:00", tz=tz)
        assert obj.mean(skipna=False) is pd.NaT

    @pytest.mark.parametrize("box", [Series, pd.Index, pd.array])
    @pytest.mark.parametrize("freq", ["s", "h", "D", "W", "B"])
    def test_period_mean(self, box, freq):
        # GH#24757
        dti = date_range("2001-01-01", periods=11)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])

        warn = FutureWarning if freq == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            parr = dti._data.to_period(freq)
        obj = box(parr)
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

        # parr[-2] will be the first date 2001-01-1
        parr[-2] = pd.NaT

        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

    @pytest.mark.parametrize("box", [Series, pd.Index, pd.array])
    def test_td64_mean(self, box):
        m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], "m8[D]")
        tdi = pd.TimedeltaIndex(m8values).as_unit("ns")

        tdarr = tdi._data
        obj = box(tdarr, copy=False)

        result = obj.mean()
        expected = np.array(tdarr).mean()
        assert result == expected

        tdarr[0] = pd.NaT
        assert obj.mean(skipna=False) is pd.NaT

        result2 = obj.mean(skipna=True)
        assert result2 == tdi[1:].mean()

        # exact equality fails by 1 nanosecond
        assert result2.round("us") == (result * 11.0 / 10).round("us")


class TestSeriesStatReductions:
    # Note: the name TestSeriesStatReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def _check_stat_op(
        self, name, alternate, string_series_, check_objects=False, check_allna=False
    ):
        with pd.option_context("use_bottleneck", False):
            f = getattr(Series, name)

            # add some NaNs
            string_series_[5:15] = np.nan

            # mean, idxmax, idxmin, min, and max are valid for dates
            if name not in ["max", "min", "mean", "median", "std"]:
                ds = Series(date_range("1/1/2001", periods=10))
                msg = f"does not support reduction '{name}'"
                with pytest.raises(TypeError, match=msg):
                    f(ds)

            # skipna or no
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))

            # check the result is correct
            nona = string_series_.dropna()
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))

            allna = string_series_ * np.nan

            if check_allna:
                assert np.isnan(f(allna))

            # dtype=object with None, it works!
            s = Series([1, 2, 3, None, 5])
            f(s)

            # GH#2888
            items = [0]
            items.extend(range(2**40, 2**40 + 1000))
            s = Series(items, dtype="int64")
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))

            # check date range
            if check_objects:
                s = Series(pd.bdate_range("1/1/2000", periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp

            # check on string data
            if name not in ["sum", "min", "max"]:
                with pytest.raises(TypeError, match=None):
                    f(Series(list("abc")))

            # Invalid axis.
            msg = "No axis named 1 for object type Series"
            with pytest.raises(ValueError, match=msg):
                f(string_series_, axis=1)

            if "numeric_only" in inspect.getfullargspec(f).args:
                # only the index is string; dtype is float
                f(string_series_, numeric_only=True)

    def test_sum(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("sum", np.sum, string_series, check_allna=False)

    def test_mean(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("mean", np.mean, string_series)

    def test_median(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("median", np.median, string_series)

        # test with integers, test failure
        int_ts = Series(np.ones(10, dtype=int), index=range(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())

    def test_prod(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("prod", np.prod, string_series)

    def test_min(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("min", np.min, string_series, check_objects=True)

    def test_max(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("max", np.max, string_series, check_objects=True)

    def test_var_std(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1)
        self._check_stat_op("std", alt, string_series)

        alt = lambda x: np.var(x, ddof=1)
        self._check_stat_op("var", alt, string_series)

        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        assert pd.isna(result)

        result = s.std(ddof=1)
        assert pd.isna(result)

    def test_sem(self):
        string_series = Series(range(20), dtype=np.float64, name="series")
        datetime_series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        self._check_stat_op("sem", alt, string_series)

        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values, ddof=4) / np.sqrt(
            len(datetime_series.values)
        )
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        assert pd.isna(result)

    def test_skew(self):
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.skew(x, bias=False)
        self._check_stat_op("skew", alt, string_series)

        # test corner cases, skew() returns NaN unless there's at least 3
        # values
        min_N = 3
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.skew())
                assert np.isnan(df.skew()).all()
            else:
                assert 0 == s.skew()
                assert isinstance(s.skew(), np.float64)  # GH53482
                assert (df.skew() == 0).all()

    def test_kurt(self):
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.kurtosis(x, bias=False)
        self._check_stat_op("kurt", alt, string_series)

    def test_kurt_corner(self):
        # test corner cases, kurt() returns NaN unless there's at least 4
        # values
        min_N = 4
        for i in range(1, min_N + 1):
            s = Series(np.ones(i))
            df = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.kurt())
                assert np.isnan(df.kurt()).all()
            else:
                assert 0 == s.kurt()
                assert isinstance(s.kurt(), np.float64)  # GH53482
                assert (df.kurt() == 0).all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_vector.py ===
from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises

A = ReferenceFrame('A')


def test_free_dynamicsymbols():
    A, B, C, D = symbols('A, B, C, D', cls=ReferenceFrame)
    a, b, c, d, e, f = dynamicsymbols('a, b, c, d, e, f')
    B.orient_axis(A, a, A.x)
    C.orient_axis(B, b, B.y)
    D.orient_axis(C, c, C.x)

    v = d*D.x + e*D.y + f*D.z

    assert set(ordered(v.free_dynamicsymbols(A))) == {a, b, c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(B))) == {b, c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(C))) == {c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(D))) == {d, e, f}


def test_Vector():
    assert A.x != A.y
    assert A.y != A.z
    assert A.z != A.x

    assert A.x + 0 == A.x

    v1 = x*A.x + y*A.y + z*A.z
    v2 = x**2*A.x + y**2*A.y + z**2*A.z
    v3 = v1 + v2
    v4 = v1 - v2

    assert isinstance(v1, Vector)
    assert dot(v1, A.x) == x
    assert dot(v1, A.y) == y
    assert dot(v1, A.z) == z

    assert isinstance(v2, Vector)
    assert dot(v2, A.x) == x**2
    assert dot(v2, A.y) == y**2
    assert dot(v2, A.z) == z**2

    assert isinstance(v3, Vector)
    # We probably shouldn't be using simplify in dot...
    assert dot(v3, A.x) == x**2 + x
    assert dot(v3, A.y) == y**2 + y
    assert dot(v3, A.z) == z**2 + z

    assert isinstance(v4, Vector)
    # We probably shouldn't be using simplify in dot...
    assert dot(v4, A.x) == x - x**2
    assert dot(v4, A.y) == y - y**2
    assert dot(v4, A.z) == z - z**2

    assert v1.to_matrix(A) == Matrix([[x], [y], [z]])
    q = symbols('q')
    B = A.orientnew('B', 'Axis', (q, A.x))
    assert v1.to_matrix(B) == Matrix([[x],
                                      [ y * cos(q) + z * sin(q)],
                                      [-y * sin(q) + z * cos(q)]])

    #Test the separate method
    B = ReferenceFrame('B')
    v5 = x*A.x + y*A.y + z*B.z
    assert Vector(0).separate() == {}
    assert v1.separate() == {A: v1}
    assert v5.separate() == {A: x*A.x + y*A.y, B: z*B.z}

    #Test the free_symbols property
    v6 = x*A.x + y*A.y + z*A.z
    assert v6.free_symbols(A) == {x,y,z}

    raises(TypeError, lambda: v3.applyfunc(v1))


def test_Vector_diffs():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    q1dd, q2dd, q3dd, q4dd = dynamicsymbols('q1 q2 q3 q4', 2)
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q3, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    v1 = q2 * A.x + q3 * N.y
    v2 = q3 * B.x + v1
    v3 = v1.dt(B)
    v4 = v2.dt(B)
    v5 = q1*A.x + q2*A.y + q3*A.z

    assert v1.dt(N) == q2d * A.x + q2 * q3d * A.y + q3d * N.y
    assert v1.dt(A) == q2d * A.x + q3 * q3d * N.x + q3d * N.y
    assert v1.dt(B) == (q2d * A.x + q3 * q3d * N.x + q3d *
                        N.y - q3 * cos(q3) * q2d * N.z)
    assert v2.dt(N) == (q2d * A.x + (q2 + q3) * q3d * A.y + q3d * B.x + q3d *
                        N.y)
    assert v2.dt(A) == q2d * A.x + q3d * B.x + q3 * q3d * N.x + q3d * N.y
    assert v2.dt(B) == (q2d * A.x + q3d * B.x + q3 * q3d * N.x + q3d * N.y -
                        q3 * cos(q3) * q2d * N.z)
    assert v3.dt(N) == (q2dd * A.x + q2d * q3d * A.y + (q3d**2 + q3 * q3dd) *
                        N.x + q3dd * N.y + (q3 * sin(q3) * q2d * q3d -
                        cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    assert v3.dt(A) == (q2dd * A.x + (2 * q3d**2 + q3 * q3dd) * N.x + (q3dd -
                        q3 * q3d**2) * N.y + (q3 * sin(q3) * q2d * q3d -
                        cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    assert (v3.dt(B) - (q2dd*A.x - q3*cos(q3)*q2d**2*A.y + (2*q3d**2 +
        q3*q3dd)*N.x + (q3dd - q3*q3d**2)*N.y + (2*q3*sin(q3)*q2d*q3d -
        2*cos(q3)*q2d*q3d - q3*cos(q3)*q2dd)*N.z)).express(B).simplify() == 0
    assert v4.dt(N) == (q2dd * A.x + q3d * (q2d + q3d) * A.y + q3dd * B.x +
                        (q3d**2 + q3 * q3dd) * N.x + q3dd * N.y + (q3 *
                        sin(q3) * q2d * q3d - cos(q3) * q2d * q3d - q3 *
                        cos(q3) * q2dd) * N.z)
    assert v4.dt(A) == (q2dd * A.x + q3dd * B.x + (2 * q3d**2 + q3 * q3dd) *
                        N.x + (q3dd - q3 * q3d**2) * N.y + (q3 * sin(q3) *
                        q2d * q3d - cos(q3) * q2d * q3d - q3 * cos(q3) *
                        q2dd) * N.z)
    assert (v4.dt(B) - (q2dd*A.x - q3*cos(q3)*q2d**2*A.y + q3dd*B.x +
                        (2*q3d**2 + q3*q3dd)*N.x + (q3dd - q3*q3d**2)*N.y +
                        (2*q3*sin(q3)*q2d*q3d - 2*cos(q3)*q2d*q3d -
                         q3*cos(q3)*q2dd)*N.z)).express(B).simplify() == 0
    assert v5.dt(B) == q1d*A.x + (q3*q2d + q2d)*A.y + (-q2*q2d + q3d)*A.z
    assert v5.dt(A) == q1d*A.x + q2d*A.y + q3d*A.z
    assert v5.dt(N) == (-q2*q3d + q1d)*A.x + (q1*q3d + q2d)*A.y + q3d*A.z
    assert v3.diff(q1d, N) == 0
    assert v3.diff(q2d, N) == A.x - q3 * cos(q3) * N.z
    assert v3.diff(q3d, N) == q3 * N.x + N.y
    assert v3.diff(q1d, A) == 0
    assert v3.diff(q2d, A) == A.x - q3 * cos(q3) * N.z
    assert v3.diff(q3d, A) == q3 * N.x + N.y
    assert v3.diff(q1d, B) == 0
    assert v3.diff(q2d, B) == A.x - q3 * cos(q3) * N.z
    assert v3.diff(q3d, B) == q3 * N.x + N.y
    assert v4.diff(q1d, N) == 0
    assert v4.diff(q2d, N) == A.x - q3 * cos(q3) * N.z
    assert v4.diff(q3d, N) == B.x + q3 * N.x + N.y
    assert v4.diff(q1d, A) == 0
    assert v4.diff(q2d, A) == A.x - q3 * cos(q3) * N.z
    assert v4.diff(q3d, A) == B.x + q3 * N.x + N.y
    assert v4.diff(q1d, B) == 0
    assert v4.diff(q2d, B) == A.x - q3 * cos(q3) * N.z
    assert v4.diff(q3d, B) == B.x + q3 * N.x + N.y

    # diff() should only express vector components in the derivative frame if
    # the orientation of the component's frame depends on the variable
    v6 = q2**2*N.y + q2**2*A.y + q2**2*B.y
    # already expressed in N
    n_measy = 2*q2
    # A_C_N does not depend on q2, so don't express in N
    a_measy = 2*q2
    # B_C_N depends on q2, so express in N
    b_measx = (q2**2*B.y).dot(N.x).diff(q2)
    b_measy = (q2**2*B.y).dot(N.y).diff(q2)
    b_measz = (q2**2*B.y).dot(N.z).diff(q2)
    n_comp, a_comp = v6.diff(q2, N).args
    assert len(v6.diff(q2, N).args) == 2  # only N and A parts
    assert n_comp[1] == N
    assert a_comp[1] == A
    assert n_comp[0] == Matrix([b_measx, b_measy + n_measy, b_measz])
    assert a_comp[0] == Matrix([0, a_measy, 0])


def test_vector_var_in_dcm():

    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')

    v = u1 * u2 * A.x + u3 * N.y + u4**2 * N.z

    assert v.diff(u1, N, var_in_dcm=False) == u2 * A.x
    assert v.diff(u1, A, var_in_dcm=False) == u2 * A.x
    assert v.diff(u3, N, var_in_dcm=False) == N.y
    assert v.diff(u3, A, var_in_dcm=False) == N.y
    assert v.diff(u3, B, var_in_dcm=False) == N.y
    assert v.diff(u4, N, var_in_dcm=False) == 2 * u4 * N.z

    raises(ValueError, lambda: v.diff(u1, N))


def test_vector_simplify():
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    N = ReferenceFrame('N')

    test1 = (1 / x + 1 / y) * N.x
    assert (test1 & N.x) != (x + y) / (x * y)
    test1 = test1.simplify()
    assert (test1 & N.x) == (x + y) / (x * y)

    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * N.x
    test2 = test2.simplify()
    assert (test2 & N.x) == (A**2 * s**4 / (4 * pi * k * m**3))

    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * N.x
    test3 = test3.simplify()
    assert (test3 & N.x) == 0

    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * N.x
    test4 = test4.simplify()
    assert (test4 & N.x) == -2 * y


def test_vector_evalf():
    a, b = symbols('a b')
    v = pi * A.x
    assert v.evalf(2) == Float('3.1416', 2) * A.x
    v = pi * A.x + 5 * a * A.y - b * A.z
    assert v.evalf(3) == Float('3.1416', 3) * A.x + Float('5', 3) * a * A.y - b * A.z
    assert v.evalf(5, subs={a: 1.234, b:5.8973}) == Float('3.1415926536', 5) * A.x + Float('6.17', 5) * A.y - Float('5.8973', 5) * A.z


def test_vector_angle():
    A = ReferenceFrame('A')
    v1 = A.x + A.y
    v2 = A.z
    assert v1.angle_between(v2) == pi/2
    B = ReferenceFrame('B')
    B.orient_axis(A, A.x, pi)
    v3 = A.x
    v4 = B.x
    assert v3.angle_between(v4) == 0


def test_vector_xreplace():
    x, y, z = symbols('x y z')
    v = x**2 * A.x + x*y * A.y + x*y*z * A.z
    assert v.xreplace({x : cos(x)}) == cos(x)**2 * A.x + y*cos(x) * A.y + y*z*cos(x) * A.z
    assert v.xreplace({x*y : pi}) == x**2 * A.x + pi * A.y + x*y*z * A.z
    assert v.xreplace({x*y*z : 1}) == x**2*A.x + x*y*A.y + A.z
    assert v.xreplace({x:1, z:0}) == A.x + y * A.y
    raises(TypeError, lambda: v.xreplace())
    raises(TypeError, lambda: v.xreplace([x, y]))

def test_issue_23366():
    u1 = dynamicsymbols('u1')
    N = ReferenceFrame('N')
    N_v_A = u1*N.x
    raises(VectorTypeError, lambda: N_v_A.diff(N, u1))


def test_vector_outer():
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    N = ReferenceFrame('N')
    v1 = a*N.x + b*N.y + c*N.z
    v2 = d*N.x + e*N.y + f*N.z
    v1v2 = Matrix([[a*d, a*e, a*f],
                   [b*d, b*e, b*f],
                   [c*d, c*e, c*f]])
    assert v1.outer(v2).to_matrix(N) == v1v2
    assert (v1 | v2).to_matrix(N) == v1v2
    v2v1 = Matrix([[d*a, d*b, d*c],
                   [e*a, e*b, e*c],
                   [f*a, f*b, f*c]])
    assert v2.outer(v1).to_matrix(N) == v2v1
    assert (v2 | v1).to_matrix(N) == v2v1


def test_overloaded_operators():
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    N = ReferenceFrame('N')
    v1 = a*N.x + b*N.y + c*N.z
    v2 = d*N.x + e*N.y + f*N.z

    assert v1 + v2 == v2 + v1
    assert v1 - v2 == -v2 + v1
    assert v1 & v2 == v2 & v1
    assert v1 ^ v2 == v1.cross(v2)
    assert v2 ^ v1 == v2.cross(v1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_string.py ===
"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
from __future__ import annotations

import string
from typing import cast

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW

from pandas.core.dtypes.base import StorageExtensionDtype

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base


def maybe_split_array(arr, chunked):
    if not chunked:
        return arr
    elif arr.dtype.storage != "pyarrow":
        return arr

    pa = pytest.importorskip("pyarrow")

    arrow_array = arr._pa_array
    split = len(arrow_array) // 2
    arrow_array = pa.chunked_array(
        [*arrow_array[:split].chunks, *arrow_array[split:].chunks]
    )
    assert arrow_array.num_chunks == 2
    return type(arr)(arrow_array)


@pytest.fixture(params=[True, False])
def chunked(request):
    return request.param


@pytest.fixture
def dtype(string_dtype_arguments):
    storage, na_value = string_dtype_arguments
    return StringDtype(storage=storage, na_value=na_value)


@pytest.fixture
def data(dtype, chunked):
    strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
    while strings[0] == strings[1]:
        strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)

    arr = dtype.construct_array_type()._from_sequence(strings, dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_missing(dtype, chunked):
    """Length 2 array with [NA, Valid]"""
    arr = dtype.construct_array_type()._from_sequence([pd.NA, "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_for_sorting(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(["B", "C", "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_missing_for_sorting(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(["B", pd.NA, "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


@pytest.fixture
def data_for_grouping(dtype, chunked):
    arr = dtype.construct_array_type()._from_sequence(
        ["B", "B", pd.NA, pd.NA, "A", "A", "B", "C"], dtype=dtype
    )
    return maybe_split_array(arr, chunked)


class TestStringArray(base.ExtensionTests):
    def test_eq_with_str(self, dtype):
        super().test_eq_with_str(dtype)

        if dtype.na_value is pd.NA:
            # only the NA-variant supports parametrized string alias
            assert dtype == f"string[{dtype.storage}]"
        elif dtype.storage == "pyarrow":
            with tm.assert_produces_warning(FutureWarning):
                assert dtype == "string[pyarrow_numpy]"

    def test_is_not_string_type(self, dtype):
        # Different from BaseDtypeTests.test_is_not_string_type
        # because StringDtype is a string type
        assert is_string_dtype(dtype)

    def test_is_dtype_from_name(self, dtype, using_infer_string):
        if dtype.na_value is np.nan and not using_infer_string:
            result = type(dtype).is_dtype(dtype.name)
            assert result is False
        else:
            super().test_is_dtype_from_name(dtype)

    def test_construct_from_string_own_name(self, dtype, using_infer_string):
        if dtype.na_value is np.nan and not using_infer_string:
            with pytest.raises(TypeError, match="Cannot construct a 'StringDtype'"):
                dtype.construct_from_string(dtype.name)
        else:
            super().test_construct_from_string_own_name(dtype)

    def test_view(self, data):
        if data.dtype.storage == "pyarrow":
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_view(data)

    def test_from_dtype(self, data):
        # base test uses string representation of dtype
        pass

    def test_transpose(self, data):
        if data.dtype.storage == "pyarrow":
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_transpose(data)

    def test_setitem_preserves_views(self, data):
        if data.dtype.storage == "pyarrow":
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        super().test_setitem_preserves_views(data)

    def test_dropna_array(self, data_missing):
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

        result = data.fillna(method="backfill")
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        if op_name in [
            "__mod__",
            "__rmod__",
            "__divmod__",
            "__rdivmod__",
            "__pow__",
            "__rpow__",
        ]:
            return TypeError
        elif op_name in ["__mul__", "__rmul__"]:
            # Can only multiply strings by integers
            return TypeError
        elif op_name in [
            "__truediv__",
            "__rtruediv__",
            "__floordiv__",
            "__rfloordiv__",
            "__sub__",
            "__rsub__",
        ]:
            return TypeError

        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return (
            op_name in ["min", "max", "sum"]
            or ser.dtype.na_value is np.nan  # type: ignore[union-attr]
            and op_name in ("any", "all")
        )

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        assert isinstance(ser.dtype, StorageExtensionDtype)
        return op_name in ["cummin", "cummax", "cumsum"]

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        dtype = cast(StringDtype, tm.get_dtype(obj))
        if op_name in ["__add__", "__radd__"]:
            cast_to = dtype
        elif dtype.na_value is np.nan:
            cast_to = np.bool_  # type: ignore[assignment]
        elif dtype.storage == "pyarrow":
            cast_to = "boolean[pyarrow]"  # type: ignore[assignment]
        else:
            cast_to = "boolean"  # type: ignore[assignment]
        return pointwise_result.astype(cast_to)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, "abc")

    def test_combine_add(self, data_repeated, using_infer_string, request):
        dtype = next(data_repeated(1)).dtype
        if using_infer_string and (
            (dtype.na_value is pd.NA) and dtype.storage == "python"
        ):
            mark = pytest.mark.xfail(
                reason="The pointwise operation result will be inferred to "
                "string[nan, pyarrow], which does not match the input dtype"
            )
            request.applymarker(mark)
        super().test_combine_add(data_repeated)

    def test_arith_series_with_array(
        self, data, all_arithmetic_operators, using_infer_string, request
    ):
        dtype = data.dtype
        if (
            using_infer_string
            and all_arithmetic_operators == "__radd__"
            and (
                (dtype.na_value is pd.NA) or (dtype.storage == "python" and HAS_PYARROW)
            )
        ):
            mark = pytest.mark.xfail(
                reason="The pointwise operation result will be inferred to "
                "string[nan, pyarrow], which does not match the input dtype"
            )
            request.applymarker(mark)
        super().test_arith_series_with_array(data, all_arithmetic_operators)


class Test2DCompat(base.Dim2CompatTests):
    @pytest.fixture(autouse=True)
    def arrow_not_supported(self, data):
        if isinstance(data, ArrowStringArray):
            pytest.skip(reason="2D support not implemented for ArrowStringArray")


def test_searchsorted_with_na_raises(data_for_sorting, as_series):
    # GH50447
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]
    arr[-1] = pd.NA

    if as_series:
        arr = pd.Series(arr)

    msg = (
        "searchsorted requires array to be sorted, "
        "which is impossible with NAs present."
    )
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\json\array.py ===
"""
Test extension array for storing nested data in a pandas container.

The JSONArray stores lists of dictionaries. The storage mechanism is a list,
not an ndarray.

Note
----
We currently store lists of UserDicts. Pandas has a few places
internally that specifically check for dicts, and does non-scalar things
in that case. We *want* the dictionaries to be treated as scalars, so we
hack around pandas by using UserDicts.
"""
from __future__ import annotations

from collections import (
    UserDict,
    abc,
)
import itertools
import numbers
import string
import sys
from typing import (
    TYPE_CHECKING,
    Any,
)
import warnings

import numpy as np

from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_list_like,
    pandas_dtype,
)

import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)
from pandas.core.indexers import unpack_tuple_and_ellipses

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pandas._typing import type_t


class JSONDtype(ExtensionDtype):
    type = abc.Mapping
    name = "json"
    na_value: Mapping[str, Any] = UserDict()

    @classmethod
    def construct_array_type(cls) -> type_t[JSONArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return JSONArray


class JSONArray(ExtensionArray):
    dtype = JSONDtype()
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False) -> None:
        for val in values:
            if not isinstance(val, self.dtype.type):
                raise TypeError("All values must be of type " + str(self.dtype.type))
        self.data = values

        # Some aliases for common attribute names to ensure pandas supports
        # these
        self._items = self._data = self.data
        # those aliases are currently not working due to assumptions
        # in internal code (GH-20735)
        # self._values = self.values = self.data

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls([UserDict(x) for x in values if x != ()])

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        if isinstance(item, numbers.Integral):
            return self.data[item]
        elif isinstance(item, slice) and item == slice(None):
            # Make sure we get a view
            return type(self)(self.data)
        elif isinstance(item, slice):
            # slice
            return type(self)(self.data[item])
        elif not is_list_like(item):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        else:
            item = pd.api.indexers.check_array_indexer(self, item)
            if is_bool_dtype(item.dtype):
                return type(self)._from_sequence(
                    [x for x, m in zip(self, item) if m], dtype=self.dtype
                )
            # integer
            return type(self)([self.data[i] for i in item])

    def __setitem__(self, key, value) -> None:
        if isinstance(key, numbers.Integral):
            self.data[key] = value
        else:
            if not isinstance(value, (type(self), abc.Sequence)):
                # broadcast value
                value = itertools.cycle([value])

            if isinstance(key, np.ndarray) and key.dtype == "bool":
                # masking
                for i, (k, v) in enumerate(zip(key, value)):
                    if k:
                        assert isinstance(v, self.dtype.type)
                        self.data[i] = v
            else:
                for k, v in zip(key, value):
                    assert isinstance(v, self.dtype.type)
                    self.data[k] = v

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return NotImplemented

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            warnings.warn(
                "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                "changed and passing 'copy=False' raises an error when returning "
                "a zero-copy NumPy array is not possible. pandas will follow "
                "this behavior starting with pandas 3.0.\nThis conversion to "
                "NumPy requires a copy, but 'copy=False' was passed. Consider "
                "using 'np.asarray(..)' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        if dtype is None:
            dtype = object
        if dtype == object:
            # on py38 builds it looks like numpy is inferring to a non-1D array
            return construct_1d_object_array_from_listlike(list(self))
        if copy is None:
            # Note: branch avoids `copy=None` for NumPy 1.x support
            return np.asarray(self.data, dtype=dtype)
        return np.asarray(self.data, dtype=dtype, copy=copy)

    @property
    def nbytes(self) -> int:
        return sys.getsizeof(self.data)

    def isna(self):
        return np.array([x == self.dtype.na_value for x in self.data], dtype=bool)

    def take(self, indexer, allow_fill=False, fill_value=None):
        # re-implement here, since NumPy has trouble setting
        # sized objects like UserDicts into scalar slots of
        # an ndarary.
        indexer = np.asarray(indexer)
        msg = (
            "Index is out of bounds or cannot do a "
            "non-empty take from an empty array."
        )

        if allow_fill:
            if fill_value is None:
                fill_value = self.dtype.na_value
            # bounds check
            if (indexer < -1).any():
                raise ValueError
            try:
                output = [
                    self.data[loc] if loc != -1 else fill_value for loc in indexer
                ]
            except IndexError as err:
                raise IndexError(msg) from err
        else:
            try:
                output = [self.data[loc] for loc in indexer]
            except IndexError as err:
                raise IndexError(msg) from err

        return type(self)._from_sequence(output, dtype=self.dtype)

    def copy(self):
        return type(self)(self.data[:])

    def astype(self, dtype, copy=True):
        # NumPy has issues when all the dicts are the same length.
        # np.array([UserDict(...), UserDict(...)]) fails,
        # but np.array([{...}, {...}]) works, so cast.
        from pandas.core.arrays.string_ import StringDtype

        dtype = pandas_dtype(dtype)
        # needed to add this check for the Series constructor
        if isinstance(dtype, type(self.dtype)) and dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, StringDtype):
            arr_cls = dtype.construct_array_type()
            return arr_cls._from_sequence(self, dtype=dtype, copy=False)
        elif not copy:
            return np.asarray([dict(x) for x in self], dtype=dtype)
        else:
            return np.array([dict(x) for x in self], dtype=dtype, copy=copy)

    def unique(self):
        # Parent method doesn't work since np.array will try to infer
        # a 2-dim object.
        return type(self)([dict(x) for x in {tuple(d.items()) for d in self.data}])

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = list(itertools.chain.from_iterable(x.data for x in to_concat))
        return cls(data)

    def _values_for_factorize(self):
        frozen = self._values_for_argsort()
        if len(frozen) == 0:
            # factorize_array expects 1-d array, this is a len-0 2-d array.
            frozen = frozen.ravel()
        return frozen, ()

    def _values_for_argsort(self):
        # Bypass NumPy's shape inference to get a (N,) array of tuples.
        frozen = [tuple(x.items()) for x in self]
        return construct_1d_object_array_from_listlike(frozen)

    def _pad_or_backfill(self, *, method, limit=None, copy=True):
        # GH#56616 - test EA method without limit_area argument
        return super()._pad_or_backfill(method=method, limit=limit, copy=copy)


def make_data():
    # TODO: Use a regular dict. See _NDFrameIndexer._setitem_with_indexer
    rng = np.random.default_rng(2)
    return [
        UserDict(
            [
                (rng.choice(list(string.ascii_letters)), rng.integers(0, 100))
                for _ in range(rng.integers(0, 10))
            ]
        )
        for _ in range(100)
    ]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\fromnumeric.py ===
"""Tests for :mod:`numpy._core.fromnumeric`."""

import numpy as np

A = np.array(True, ndmin=2, dtype=bool)
B = np.array(1.0, ndmin=2, dtype=np.float32)
A.setflags(write=False)
B.setflags(write=False)

a = np.bool(True)
b = np.float32(1.0)
c = 1.0
d = np.array(1.0, dtype=np.float32)  # writeable

np.take(a, 0)
np.take(b, 0)
np.take(c, 0)
np.take(A, 0)
np.take(B, 0)
np.take(A, [0])
np.take(B, [0])

np.reshape(a, 1)
np.reshape(b, 1)
np.reshape(c, 1)
np.reshape(A, 1)
np.reshape(B, 1)

np.choose(a, [True, True])
np.choose(A, [1.0, 1.0])

np.repeat(a, 1)
np.repeat(b, 1)
np.repeat(c, 1)
np.repeat(A, 1)
np.repeat(B, 1)

np.swapaxes(A, 0, 0)
np.swapaxes(B, 0, 0)

np.transpose(a)
np.transpose(b)
np.transpose(c)
np.transpose(A)
np.transpose(B)

np.partition(a, 0, axis=None)
np.partition(b, 0, axis=None)
np.partition(c, 0, axis=None)
np.partition(A, 0)
np.partition(B, 0)

np.argpartition(a, 0)
np.argpartition(b, 0)
np.argpartition(c, 0)
np.argpartition(A, 0)
np.argpartition(B, 0)

np.sort(A, 0)
np.sort(B, 0)

np.argsort(A, 0)
np.argsort(B, 0)

np.argmax(A)
np.argmax(B)
np.argmax(A, axis=0)
np.argmax(B, axis=0)

np.argmin(A)
np.argmin(B)
np.argmin(A, axis=0)
np.argmin(B, axis=0)

np.searchsorted(A[0], 0)
np.searchsorted(B[0], 0)
np.searchsorted(A[0], [0])
np.searchsorted(B[0], [0])

np.resize(a, (5, 5))
np.resize(b, (5, 5))
np.resize(c, (5, 5))
np.resize(A, (5, 5))
np.resize(B, (5, 5))

np.squeeze(a)
np.squeeze(b)
np.squeeze(c)
np.squeeze(A)
np.squeeze(B)

np.diagonal(A)
np.diagonal(B)

np.trace(A)
np.trace(B)

np.ravel(a)
np.ravel(b)
np.ravel(c)
np.ravel(A)
np.ravel(B)

np.nonzero(A)
np.nonzero(B)

np.shape(a)
np.shape(b)
np.shape(c)
np.shape(A)
np.shape(B)

np.compress([True], a)
np.compress([True], b)
np.compress([True], c)
np.compress([True], A)
np.compress([True], B)

np.clip(a, 0, 1.0)
np.clip(b, -1, 1)
np.clip(a, 0, None)
np.clip(b, None, 1)
np.clip(c, 0, 1)
np.clip(A, 0, 1)
np.clip(B, 0, 1)
np.clip(B, [0, 1], [1, 2])

np.sum(a)
np.sum(b)
np.sum(c)
np.sum(A)
np.sum(B)
np.sum(A, axis=0)
np.sum(B, axis=0)

np.all(a)
np.all(b)
np.all(c)
np.all(A)
np.all(B)
np.all(A, axis=0)
np.all(B, axis=0)
np.all(A, keepdims=True)
np.all(B, keepdims=True)

np.any(a)
np.any(b)
np.any(c)
np.any(A)
np.any(B)
np.any(A, axis=0)
np.any(B, axis=0)
np.any(A, keepdims=True)
np.any(B, keepdims=True)

np.cumsum(a)
np.cumsum(b)
np.cumsum(c)
np.cumsum(A)
np.cumsum(B)

np.cumulative_sum(a)
np.cumulative_sum(b)
np.cumulative_sum(c)
np.cumulative_sum(A, axis=0)
np.cumulative_sum(B, axis=0)

np.ptp(b)
np.ptp(c)
np.ptp(B)
np.ptp(B, axis=0)
np.ptp(B, keepdims=True)

np.amax(a)
np.amax(b)
np.amax(c)
np.amax(A)
np.amax(B)
np.amax(A, axis=0)
np.amax(B, axis=0)
np.amax(A, keepdims=True)
np.amax(B, keepdims=True)

np.amin(a)
np.amin(b)
np.amin(c)
np.amin(A)
np.amin(B)
np.amin(A, axis=0)
np.amin(B, axis=0)
np.amin(A, keepdims=True)
np.amin(B, keepdims=True)

np.prod(a)
np.prod(b)
np.prod(c)
np.prod(A)
np.prod(B)
np.prod(a, dtype=None)
np.prod(A, dtype=None)
np.prod(A, axis=0)
np.prod(B, axis=0)
np.prod(A, keepdims=True)
np.prod(B, keepdims=True)
np.prod(b, out=d)
np.prod(B, out=d)

np.cumprod(a)
np.cumprod(b)
np.cumprod(c)
np.cumprod(A)
np.cumprod(B)

np.cumulative_prod(a)
np.cumulative_prod(b)
np.cumulative_prod(c)
np.cumulative_prod(A, axis=0)
np.cumulative_prod(B, axis=0)

np.ndim(a)
np.ndim(b)
np.ndim(c)
np.ndim(A)
np.ndim(B)

np.size(a)
np.size(b)
np.size(c)
np.size(A)
np.size(B)

np.around(a)
np.around(b)
np.around(c)
np.around(A)
np.around(B)

np.mean(a)
np.mean(b)
np.mean(c)
np.mean(A)
np.mean(B)
np.mean(A, axis=0)
np.mean(B, axis=0)
np.mean(A, keepdims=True)
np.mean(B, keepdims=True)
np.mean(b, out=d)
np.mean(B, out=d)

np.std(a)
np.std(b)
np.std(c)
np.std(A)
np.std(B)
np.std(A, axis=0)
np.std(B, axis=0)
np.std(A, keepdims=True)
np.std(B, keepdims=True)
np.std(b, out=d)
np.std(B, out=d)

np.var(a)
np.var(b)
np.var(c)
np.var(A)
np.var(B)
np.var(A, axis=0)
np.var(B, axis=0)
np.var(A, keepdims=True)
np.var(B, keepdims=True)
np.var(b, out=d)
np.var(B, out=d)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\frame\test_frame_legend.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    date_range,
)
from pandas.tests.plotting.common import (
    _check_legend_labels,
    _check_legend_marker,
    _check_text_labels,
)
from pandas.util.version import Version

mpl = pytest.importorskip("matplotlib")


class TestFrameLegend:
    @pytest.mark.xfail(
        reason=(
            "Open bug in matplotlib "
            "https://github.com/matplotlib/matplotlib/issues/11357"
        )
    )
    def test_mixed_yerr(self):
        # https://github.com/pandas-dev/pandas/issues/39522
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        df = DataFrame([{"x": 1, "a": 1, "b": 1}, {"x": 2, "a": 2, "b": 3}])

        ax = df.plot("x", "a", c="orange", yerr=0.1, label="orange")
        df.plot("x", "b", c="blue", yerr=None, ax=ax, label="blue")

        legend = ax.get_legend()
        if Version(mpl.__version__) < Version("3.7"):
            result_handles = legend.legendHandles
        else:
            result_handles = legend.legend_handles

        assert isinstance(result_handles[0], LineCollection)
        assert isinstance(result_handles[1], Line2D)

    def test_legend_false(self):
        # https://github.com/pandas-dev/pandas/issues/40044
        df = DataFrame({"a": [1, 1], "b": [2, 3]})
        df2 = DataFrame({"d": [2.5, 2.5]})

        ax = df.plot(legend=True, color={"a": "blue", "b": "green"}, secondary_y="b")
        df2.plot(legend=True, color={"d": "red"}, ax=ax)
        legend = ax.get_legend()
        if Version(mpl.__version__) < Version("3.7"):
            handles = legend.legendHandles
        else:
            handles = legend.legend_handles
        result = [handle.get_color() for handle in handles]
        expected = ["blue", "green", "red"]
        assert result == expected

    @pytest.mark.parametrize("kind", ["line", "bar", "barh", "kde", "area", "hist"])
    def test_df_legend_labels(self, kind):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=["a", "b", "c"])
        df2 = DataFrame(
            np.random.default_rng(2).random((3, 3)), columns=["d", "e", "f"]
        )
        df3 = DataFrame(
            np.random.default_rng(2).random((3, 3)), columns=["g", "h", "i"]
        )
        df4 = DataFrame(
            np.random.default_rng(2).random((3, 3)), columns=["j", "k", "l"]
        )

        ax = df.plot(kind=kind, legend=True)
        _check_legend_labels(ax, labels=df.columns)

        ax = df2.plot(kind=kind, legend=False, ax=ax)
        _check_legend_labels(ax, labels=df.columns)

        ax = df3.plot(kind=kind, legend=True, ax=ax)
        _check_legend_labels(ax, labels=df.columns.union(df3.columns))

        ax = df4.plot(kind=kind, legend="reverse", ax=ax)
        expected = list(df.columns.union(df3.columns)) + list(reversed(df4.columns))
        _check_legend_labels(ax, labels=expected)

    def test_df_legend_labels_secondary_y(self):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=["a", "b", "c"])
        df2 = DataFrame(
            np.random.default_rng(2).random((3, 3)), columns=["d", "e", "f"]
        )
        df3 = DataFrame(
            np.random.default_rng(2).random((3, 3)), columns=["g", "h", "i"]
        )
        # Secondary Y
        ax = df.plot(legend=True, secondary_y="b")
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df2.plot(legend=False, ax=ax)
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df3.plot(kind="bar", legend=True, secondary_y="h", ax=ax)
        _check_legend_labels(ax, labels=["a", "b (right)", "c", "g", "h (right)", "i"])

    def test_df_legend_labels_time_series(self):
        # Time Series
        pytest.importorskip("scipy")
        ind = date_range("1/1/2014", periods=3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["a", "b", "c"],
            index=ind,
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["d", "e", "f"],
            index=ind,
        )
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["g", "h", "i"],
            index=ind,
        )
        ax = df.plot(legend=True, secondary_y="b")
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df2.plot(legend=False, ax=ax)
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        ax = df3.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=["a", "b (right)", "c", "g", "h", "i"])

    def test_df_legend_labels_time_series_scatter(self):
        # Time Series
        pytest.importorskip("scipy")
        ind = date_range("1/1/2014", periods=3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["a", "b", "c"],
            index=ind,
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["d", "e", "f"],
            index=ind,
        )
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["g", "h", "i"],
            index=ind,
        )
        # scatter
        ax = df.plot.scatter(x="a", y="b", label="data1")
        _check_legend_labels(ax, labels=["data1"])
        ax = df2.plot.scatter(x="d", y="e", legend=False, label="data2", ax=ax)
        _check_legend_labels(ax, labels=["data1"])
        ax = df3.plot.scatter(x="g", y="h", label="data3", ax=ax)
        _check_legend_labels(ax, labels=["data1", "data3"])

    def test_df_legend_labels_time_series_no_mutate(self):
        pytest.importorskip("scipy")
        ind = date_range("1/1/2014", periods=3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["a", "b", "c"],
            index=ind,
        )
        # ensure label args pass through and
        # index name does not mutate
        # column names don't mutate
        df5 = df.set_index("a")
        ax = df5.plot(y="b")
        _check_legend_labels(ax, labels=["b"])
        ax = df5.plot(y="b", label="LABEL_b")
        _check_legend_labels(ax, labels=["LABEL_b"])
        _check_text_labels(ax.xaxis.get_label(), "a")
        ax = df5.plot(y="c", label="LABEL_c", ax=ax)
        _check_legend_labels(ax, labels=["LABEL_b", "LABEL_c"])
        assert df5.columns.tolist() == ["b", "c"]

    def test_missing_marker_multi_plots_on_same_ax(self):
        # GH 18222
        df = DataFrame(data=[[1, 1, 1, 1], [2, 2, 4, 8]], columns=["x", "r", "g", "b"])
        _, ax = mpl.pyplot.subplots(nrows=1, ncols=3)
        # Left plot
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[0])
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[0])
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[0])
        _check_legend_labels(ax[0], labels=["r", "g", "b"])
        _check_legend_marker(ax[0], expected_markers=["o", "x", "o"])
        # Center plot
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[1])
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[1])
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[1])
        _check_legend_labels(ax[1], labels=["b", "r", "g"])
        _check_legend_marker(ax[1], expected_markers=["o", "o", "x"])
        # Right plot
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[2])
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[2])
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[2])
        _check_legend_labels(ax[2], labels=["g", "b", "r"])
        _check_legend_marker(ax[2], expected_markers=["x", "o", "o"])

    def test_legend_name(self):
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=[np.array(["a", "a", "b", "b"]), np.array(["x", "y", "x", "y"])],
        )
        multi.columns.names = ["group", "individual"]

        ax = multi.plot()
        leg_title = ax.legend_.get_title()
        _check_text_labels(leg_title, "group,individual")

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(legend=True, ax=ax)
        leg_title = ax.legend_.get_title()
        _check_text_labels(leg_title, "group,individual")

        df.columns.name = "new"
        ax = df.plot(legend=False, ax=ax)
        leg_title = ax.legend_.get_title()
        _check_text_labels(leg_title, "group,individual")

        ax = df.plot(legend=True, ax=ax)
        leg_title = ax.legend_.get_title()
        _check_text_labels(leg_title, "new")

    @pytest.mark.parametrize(
        "kind",
        [
            "line",
            "bar",
            "barh",
            pytest.param("kde", marks=td.skip_if_no("scipy")),
            "area",
            "hist",
        ],
    )
    def test_no_legend(self, kind):
        df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=["a", "b", "c"])
        ax = df.plot(kind=kind, legend=False)
        _check_legend_labels(ax, visible=False)

    def test_missing_markers_legend(self):
        # 14958
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)), columns=["A", "B", "C"]
        )
        ax = df.plot(y=["A"], marker="x", linestyle="solid")
        df.plot(y=["B"], marker="o", linestyle="dotted", ax=ax)
        df.plot(y=["C"], marker="<", linestyle="dotted", ax=ax)

        _check_legend_labels(ax, labels=["A", "B", "C"])
        _check_legend_marker(ax, expected_markers=["x", "o", "<"])

    def test_missing_markers_legend_using_style(self):
        # 14563
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6],
                "B": [2, 4, 1, 3, 2, 4],
                "C": [3, 3, 2, 6, 4, 2],
                "X": [1, 2, 3, 4, 5, 6],
            }
        )

        _, ax = mpl.pyplot.subplots()
        for kind in "ABC":
            df.plot("X", kind, label=kind, ax=ax, style=".")

        _check_legend_labels(ax, labels=["A", "B", "C"])
        _check_legend_marker(ax, expected_markers=[".", ".", "."])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_categorical.py ===
from datetime import datetime

import numpy as np

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestCategoricalConcat:
    def test_categorical_concat(self, sort):
        # See GH 10177
        df1 = DataFrame(
            np.arange(18, dtype="int64").reshape(6, 3), columns=["a", "b", "c"]
        )

        df2 = DataFrame(np.arange(14, dtype="int64").reshape(7, 2), columns=["a", "c"])

        cat_values = ["one", "one", "two", "one", "two", "two", "one"]
        df2["h"] = Series(Categorical(cat_values))

        res = pd.concat((df1, df2), axis=0, ignore_index=True, sort=sort)
        exp = DataFrame(
            {
                "a": [0, 3, 6, 9, 12, 15, 0, 2, 4, 6, 8, 10, 12],
                "b": [
                    1,
                    4,
                    7,
                    10,
                    13,
                    16,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "c": [2, 5, 8, 11, 14, 17, 1, 3, 5, 7, 9, 11, 13],
                "h": [None] * 6 + cat_values,
            }
        )
        exp["h"] = exp["h"].astype(df2["h"].dtype)
        tm.assert_frame_equal(res, exp)

    def test_categorical_concat_dtypes(self, using_infer_string):
        # GH8143
        index = ["cat", "obj", "num"]
        cat = Categorical(["a", "b", "c"])
        obj = Series(["a", "b", "c"])
        num = Series([1, 2, 3])
        df = pd.concat([Series(cat), obj, num], axis=1, keys=index)

        result = df.dtypes == (object if not using_infer_string else "str")
        expected = Series([False, True, False], index=index)
        tm.assert_series_equal(result, expected)

        result = df.dtypes == "int64"
        expected = Series([False, False, True], index=index)
        tm.assert_series_equal(result, expected)

        result = df.dtypes == "category"
        expected = Series([True, False, False], index=index)
        tm.assert_series_equal(result, expected)

    def test_concat_categoricalindex(self):
        # GH 16111, categories that aren't lexsorted
        categories = [9, 0, 1, 2, 3]

        a = Series(1, index=pd.CategoricalIndex([9, 0], categories=categories))
        b = Series(2, index=pd.CategoricalIndex([0, 1], categories=categories))
        c = Series(3, index=pd.CategoricalIndex([1, 2], categories=categories))

        result = pd.concat([a, b, c], axis=1)

        exp_idx = pd.CategoricalIndex([9, 0, 1, 2], categories=categories)
        exp = DataFrame(
            {
                0: [1, 1, np.nan, np.nan],
                1: [np.nan, 2, 2, np.nan],
                2: [np.nan, np.nan, 3, 3],
            },
            columns=[0, 1, 2],
            index=exp_idx,
        )
        tm.assert_frame_equal(result, exp)

    def test_categorical_concat_preserve(self):
        # GH 8641  series concat not preserving category dtype
        # GH 13524 can concat different categories
        s = Series(list("abc"), dtype="category")
        s2 = Series(list("abd"), dtype="category")

        exp = Series(list("abcabd"))
        res = pd.concat([s, s2], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series(list("abcabc"), dtype="category")
        res = pd.concat([s, s], ignore_index=True)
        tm.assert_series_equal(res, exp)

        exp = Series(list("abcabc"), index=[0, 1, 2, 0, 1, 2], dtype="category")
        res = pd.concat([s, s])
        tm.assert_series_equal(res, exp)

        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        df2 = DataFrame({"A": a, "B": b.astype(CategoricalDtype(list("cab")))})
        res = pd.concat([df2, df2])
        exp = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        )
        tm.assert_frame_equal(res, exp)

    def test_categorical_index_preserver(self):
        a = Series(np.arange(6, dtype="int64"))
        b = Series(list("aabbca"))

        df2 = DataFrame(
            {"A": a, "B": b.astype(CategoricalDtype(list("cab")))}
        ).set_index("B")
        result = pd.concat([df2, df2])
        expected = DataFrame(
            {
                "A": pd.concat([a, a]),
                "B": pd.concat([b, b]).astype(CategoricalDtype(list("cab"))),
            }
        ).set_index("B")
        tm.assert_frame_equal(result, expected)

        # wrong categories -> uses concat_compat, which casts to object
        df3 = DataFrame(
            {"A": a, "B": Categorical(b, categories=list("abe"))}
        ).set_index("B")
        result = pd.concat([df2, df3])
        expected = pd.concat(
            [
                df2.set_axis(df2.index.astype(object), axis=0),
                df3.set_axis(df3.index.astype(object), axis=0),
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_categorical_tz(self):
        # GH-23816
        a = Series(pd.date_range("2017-01-01", periods=2, tz="US/Pacific"))
        b = Series(["a", "b"], dtype="category")
        result = pd.concat([a, b], ignore_index=True)
        expected = Series(
            [
                pd.Timestamp("2017-01-01", tz="US/Pacific"),
                pd.Timestamp("2017-01-02", tz="US/Pacific"),
                "a",
                "b",
            ]
        )
        tm.assert_series_equal(result, expected)

    def test_concat_categorical_datetime(self):
        # GH-39443
        df1 = DataFrame(
            {"x": Series(datetime(2021, 1, 1), index=[0], dtype="category")}
        )
        df2 = DataFrame(
            {"x": Series(datetime(2021, 1, 2), index=[1], dtype="category")}
        )

        result = pd.concat([df1, df2])
        expected = DataFrame(
            {"x": Series([datetime(2021, 1, 1), datetime(2021, 1, 2)])}
        )

        tm.assert_equal(result, expected)

    def test_concat_categorical_unchanged(self):
        # GH-12007
        # test fix for when concat on categorical and float
        # coerces dtype categorical -> float
        df = DataFrame(Series(["a", "b", "c"], dtype="category", name="A"))
        ser = Series([0, 1, 2], index=[0, 1, 3], name="B")
        result = pd.concat([df, ser], axis=1)
        expected = DataFrame(
            {
                "A": Series(["a", "b", "c", np.nan], dtype="category"),
                "B": Series([0, 1, np.nan, 2], dtype="float"),
            }
        )
        tm.assert_equal(result, expected)

    def test_categorical_concat_gh7864(self):
        # GH 7864
        # make sure ordering is preserved
        df = DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": list("abbaae")})
        df["grade"] = Categorical(df["raw_grade"])
        df["grade"].cat.set_categories(["e", "a", "b"])

        df1 = df[0:3]
        df2 = df[3:]

        tm.assert_index_equal(df["grade"].cat.categories, df1["grade"].cat.categories)
        tm.assert_index_equal(df["grade"].cat.categories, df2["grade"].cat.categories)

        dfx = pd.concat([df1, df2])
        tm.assert_index_equal(df["grade"].cat.categories, dfx["grade"].cat.categories)

        dfa = df1._append(df2)
        tm.assert_index_equal(df["grade"].cat.categories, dfa["grade"].cat.categories)

    def test_categorical_index_upcast(self):
        # GH 17629
        # test upcasting to object when concatenating on categorical indexes
        # with non-identical categories

        a = DataFrame({"foo": [1, 2]}, index=Categorical(["foo", "bar"]))
        b = DataFrame({"foo": [4, 3]}, index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = DataFrame({"foo": [1, 2, 4, 3]}, index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

        a = Series([1, 2], index=Categorical(["foo", "bar"]))
        b = Series([4, 3], index=Categorical(["baz", "bar"]))

        res = pd.concat([a, b])
        exp = Series([1, 2, 4, 3], index=["foo", "bar", "baz", "bar"])

        tm.assert_equal(res, exp)

    def test_categorical_missing_from_one_frame(self):
        # GH 25412
        df1 = DataFrame({"f1": [1, 2, 3]})
        df2 = DataFrame({"f1": [2, 3, 1], "f2": Series([4, 4, 4]).astype("category")})
        result = pd.concat([df1, df2], sort=True)
        dtype = CategoricalDtype([4])
        expected = DataFrame(
            {
                "f1": [1, 2, 3, 2, 3, 1],
                "f2": Categorical.from_codes([-1, -1, -1, 0, 0, 0], dtype=dtype),
            },
            index=[0, 1, 2, 0, 1, 2],
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_categorical_same_categories_different_order(self):
        # https://github.com/pandas-dev/pandas/issues/24845

        c1 = pd.CategoricalIndex(["a", "a"], categories=["a", "b"], ordered=False)
        c2 = pd.CategoricalIndex(["b", "b"], categories=["b", "a"], ordered=False)
        c3 = pd.CategoricalIndex(
            ["a", "a", "b", "b"], categories=["a", "b"], ordered=False
        )

        df1 = DataFrame({"A": [1, 2]}, index=c1)
        df2 = DataFrame({"A": [3, 4]}, index=c2)

        result = pd.concat((df1, df2))
        expected = DataFrame({"A": [1, 2, 3, 4]}, index=c3)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_value_counts.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    Series,
)
import pandas._testing as tm


class TestSeriesValueCounts:
    def test_value_counts_datetime(self, unit):
        # most dtypes are tested in tests/base
        values = [
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 10:00"),
            pd.Timestamp("2011-01-01 11:00"),
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 11:00"),
        ]

        exp_idx = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 11:00", "2011-01-01 10:00"],
            name="xxx",
        ).as_unit(unit)
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx").dt.as_unit(unit)
        tm.assert_series_equal(ser.value_counts(), exp)
        # check DatetimeIndex outputs the same result
        idx = pd.DatetimeIndex(values, name="xxx").as_unit(unit)
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_datetime_tz(self, unit):
        values = [
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 10:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 11:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 11:00", tz="US/Eastern"),
        ]

        exp_idx = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 11:00", "2011-01-01 10:00"],
            tz="US/Eastern",
            name="xxx",
        ).as_unit(unit)
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx").dt.as_unit(unit)
        tm.assert_series_equal(ser.value_counts(), exp)
        idx = pd.DatetimeIndex(values, name="xxx").as_unit(unit)
        tm.assert_series_equal(idx.value_counts(), exp)

        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_period(self):
        values = [
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-02", freq="M"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-03", freq="M"),
        ]

        exp_idx = pd.PeriodIndex(
            ["2011-01", "2011-03", "2011-02"], freq="M", name="xxx"
        )
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx")
        tm.assert_series_equal(ser.value_counts(), exp)
        # check DatetimeIndex outputs the same result
        idx = pd.PeriodIndex(values, name="xxx")
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical_ordered(self):
        # most dtypes are tested in tests/base
        values = Categorical([1, 2, 3, 1, 1, 3], ordered=True)

        exp_idx = CategoricalIndex(
            [1, 3, 2], categories=[1, 2, 3], ordered=True, name="xxx"
        )
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx")
        tm.assert_series_equal(ser.value_counts(), exp)
        # check CategoricalIndex outputs the same result
        idx = CategoricalIndex(values, name="xxx")
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical_not_ordered(self):
        values = Categorical([1, 2, 3, 1, 1, 3], ordered=False)

        exp_idx = CategoricalIndex(
            [1, 3, 2], categories=[1, 2, 3], ordered=False, name="xxx"
        )
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        ser = Series(values, name="xxx")
        tm.assert_series_equal(ser.value_counts(), exp)
        # check CategoricalIndex outputs the same result
        idx = CategoricalIndex(values, name="xxx")
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical(self):
        # GH#12835
        cats = Categorical(list("abcccb"), categories=list("cabd"))
        ser = Series(cats, name="xxx")
        res = ser.value_counts(sort=False)

        exp_index = CategoricalIndex(
            list("cabd"), categories=cats.categories, name="xxx"
        )
        exp = Series([3, 1, 2, 0], name="count", index=exp_index)
        tm.assert_series_equal(res, exp)

        res = ser.value_counts(sort=True)

        exp_index = CategoricalIndex(
            list("cbad"), categories=cats.categories, name="xxx"
        )
        exp = Series([3, 2, 1, 0], name="count", index=exp_index)
        tm.assert_series_equal(res, exp)

        # check object dtype handles the Series.name as the same
        # (tested in tests/base)
        ser = Series(["a", "b", "c", "c", "c", "b"], name="xxx")
        res = ser.value_counts()
        exp = Series([3, 2, 1], name="count", index=Index(["c", "b", "a"], name="xxx"))
        tm.assert_series_equal(res, exp)

    def test_value_counts_categorical_with_nan(self):
        # see GH#9443

        # sanity check
        ser = Series(["a", "b", "a"], dtype="category")
        exp = Series([2, 1], index=CategoricalIndex(["a", "b"]), name="count")

        res = ser.value_counts(dropna=True)
        tm.assert_series_equal(res, exp)

        res = ser.value_counts(dropna=True)
        tm.assert_series_equal(res, exp)

        # same Series via two different constructions --> same behaviour
        series = [
            Series(["a", "b", None, "a", None, None], dtype="category"),
            Series(
                Categorical(["a", "b", None, "a", None, None], categories=["a", "b"])
            ),
        ]

        for ser in series:
            # None is a NaN value, so we exclude its count here
            exp = Series([2, 1], index=CategoricalIndex(["a", "b"]), name="count")
            res = ser.value_counts(dropna=True)
            tm.assert_series_equal(res, exp)

            # we don't exclude the count of None and sort by counts
            exp = Series(
                [3, 2, 1], index=CategoricalIndex([np.nan, "a", "b"]), name="count"
            )
            res = ser.value_counts(dropna=False)
            tm.assert_series_equal(res, exp)

            # When we aren't sorting by counts, and np.nan isn't a
            # category, it should be last.
            exp = Series(
                [2, 1, 3], index=CategoricalIndex(["a", "b", np.nan]), name="count"
            )
            res = ser.value_counts(dropna=False, sort=False)
            tm.assert_series_equal(res, exp)

    @pytest.mark.parametrize(
        "ser, dropna, exp",
        [
            (
                Series([False, True, True, pd.NA]),
                False,
                Series([2, 1, 1], index=[True, False, pd.NA], name="count"),
            ),
            (
                Series([False, True, True, pd.NA]),
                True,
                Series([2, 1], index=Index([True, False], dtype=object), name="count"),
            ),
            (
                Series(range(3), index=[True, False, np.nan]).index,
                False,
                Series([1, 1, 1], index=[True, False, np.nan], name="count"),
            ),
        ],
    )
    def test_value_counts_bool_with_nan(self, ser, dropna, exp):
        # GH32146
        out = ser.value_counts(dropna=dropna)
        tm.assert_series_equal(out, exp)

    @pytest.mark.parametrize(
        "input_array,expected",
        [
            (
                [1 + 1j, 1 + 1j, 1, 3j, 3j, 3j],
                Series(
                    [3, 2, 1],
                    index=Index([3j, 1 + 1j, 1], dtype=np.complex128),
                    name="count",
                ),
            ),
            (
                np.array([1 + 1j, 1 + 1j, 1, 3j, 3j, 3j], dtype=np.complex64),
                Series(
                    [3, 2, 1],
                    index=Index([3j, 1 + 1j, 1], dtype=np.complex64),
                    name="count",
                ),
            ),
        ],
    )
    def test_value_counts_complex_numbers(self, input_array, expected):
        # GH 17927
        result = Series(input_array).value_counts()
        tm.assert_series_equal(result, expected)

    def test_value_counts_masked(self):
        # GH#54984
        dtype = "Int64"
        ser = Series([1, 2, None, 2, None, 3], dtype=dtype)
        result = ser.value_counts(dropna=False)
        expected = Series(
            [2, 2, 1, 1],
            index=Index([2, None, 1, 3], dtype=dtype),
            dtype=dtype,
            name="count",
        )
        tm.assert_series_equal(result, expected)

        result = ser.value_counts(dropna=True)
        expected = Series(
            [2, 1, 1], index=Index([2, 1, 3], dtype=dtype), dtype=dtype, name="count"
        )
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_partial.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    date_range,
)
import pandas._testing as tm


class TestMultiIndexPartial:
    def test_getitem_partial_int(self):
        # GH 12416
        # with single item
        l1 = [10, 20]
        l2 = ["a", "b"]
        df = DataFrame(index=range(2), columns=MultiIndex.from_product([l1, l2]))
        expected = DataFrame(index=range(2), columns=l2)
        result = df[20]
        tm.assert_frame_equal(result, expected)

        # with list
        expected = DataFrame(
            index=range(2), columns=MultiIndex.from_product([l1[1:], l2])
        )
        result = df[[20]]
        tm.assert_frame_equal(result, expected)

        # missing item:
        with pytest.raises(KeyError, match="1"):
            df[1]
        with pytest.raises(KeyError, match=r"'\[1\] not in index'"):
            df[[1]]

    def test_series_slice_partial(self):
        pass

    def test_xs_partial(
        self,
        multiindex_dataframe_random_data,
        multiindex_year_month_day_dataframe_random_data,
    ):
        frame = multiindex_dataframe_random_data
        ymd = multiindex_year_month_day_dataframe_random_data
        result = frame.xs("foo")
        result2 = frame.loc["foo"]
        expected = frame.T["foo"].T
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, result2)

        result = ymd.xs((2000, 4))
        expected = ymd.loc[2000, 4]
        tm.assert_frame_equal(result, expected)

        # ex from #1796
        index = MultiIndex(
            levels=[["foo", "bar"], ["one", "two"], [-1, 1]],
            codes=[
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=index,
            columns=list("abcd"),
        )

        result = df.xs(("foo", "one"))
        expected = df.loc["foo", "one"]
        tm.assert_frame_equal(result, expected)

    def test_getitem_partial(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data
        ymd = ymd.T
        result = ymd[2000, 2]

        expected = ymd.reindex(columns=ymd.columns[ymd.columns.codes[1] == 1])
        expected.columns = expected.columns.droplevel(0).droplevel(0)
        tm.assert_frame_equal(result, expected)

    def test_fancy_slice_partial(
        self,
        multiindex_dataframe_random_data,
        multiindex_year_month_day_dataframe_random_data,
    ):
        frame = multiindex_dataframe_random_data
        result = frame.loc["bar":"baz"]
        expected = frame[3:7]
        tm.assert_frame_equal(result, expected)

        ymd = multiindex_year_month_day_dataframe_random_data
        result = ymd.loc[(2000, 2):(2000, 4)]
        lev = ymd.index.codes[1]
        expected = ymd[(lev >= 1) & (lev <= 3)]
        tm.assert_frame_equal(result, expected)

    def test_getitem_partial_column_select(self):
        idx = MultiIndex(
            codes=[[0, 0, 0], [0, 1, 1], [1, 0, 1]],
            levels=[["a", "b"], ["x", "y"], ["p", "q"]],
        )
        df = DataFrame(np.random.default_rng(2).random((3, 2)), index=idx)

        result = df.loc[("a", "y"), :]
        expected = df.loc[("a", "y")]
        tm.assert_frame_equal(result, expected)

        result = df.loc[("a", "y"), [1, 0]]
        expected = df.loc[("a", "y")][[1, 0]]
        tm.assert_frame_equal(result, expected)

        with pytest.raises(KeyError, match=r"\('a', 'foo'\)"):
            df.loc[("a", "foo"), :]

    # TODO(ArrayManager) rewrite test to not use .values
    # exp.loc[2000, 4].values[:] select multiple columns -> .values is not a view
    @td.skip_array_manager_invalid_test
    def test_partial_set(
        self,
        multiindex_year_month_day_dataframe_random_data,
        using_copy_on_write,
        warn_copy_on_write,
    ):
        # GH #397
        ymd = multiindex_year_month_day_dataframe_random_data
        df = ymd.copy()
        exp = ymd.copy()
        df.loc[2000, 4] = 0
        exp.iloc[65:85] = 0
        tm.assert_frame_equal(df, exp)

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                df["A"].loc[2000, 4] = 1
            df.loc[(2000, 4), "A"] = 1
        else:
            with tm.raises_chained_assignment_error():
                df["A"].loc[2000, 4] = 1
        exp.iloc[65:85, 0] = 1
        tm.assert_frame_equal(df, exp)

        df.loc[2000] = 5
        exp.iloc[:100] = 5
        tm.assert_frame_equal(df, exp)

        # this works...for now
        with tm.raises_chained_assignment_error():
            df["A"].iloc[14] = 5
        if using_copy_on_write:
            assert df["A"].iloc[14] == exp["A"].iloc[14]
        else:
            assert df["A"].iloc[14] == 5

    @pytest.mark.parametrize("dtype", [int, float])
    def test_getitem_intkey_leading_level(
        self, multiindex_year_month_day_dataframe_random_data, dtype
    ):
        # GH#33355 dont fall-back to positional when leading level is int
        ymd = multiindex_year_month_day_dataframe_random_data
        levels = ymd.index.levels
        ymd.index = ymd.index.set_levels([levels[0].astype(dtype)] + levels[1:])
        ser = ymd["A"]
        mi = ser.index
        assert isinstance(mi, MultiIndex)
        if dtype is int:
            assert mi.levels[0].dtype == np.dtype(int)
        else:
            assert mi.levels[0].dtype == np.float64

        assert 14 not in mi.levels[0]
        assert not mi.levels[0]._should_fallback_to_positional
        assert not mi._should_fallback_to_positional

        with pytest.raises(KeyError, match="14"):
            ser[14]

    # ---------------------------------------------------------------------

    def test_setitem_multiple_partial(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        expected = frame.copy()
        result = frame.copy()
        result.loc[["foo", "bar"]] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_frame_equal(result, expected)

        expected = frame.copy()
        result = frame.copy()
        result.loc["foo":"bar"] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_frame_equal(result, expected)

        expected = frame["A"].copy()
        result = frame["A"].copy()
        result.loc[["foo", "bar"]] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_series_equal(result, expected)

        expected = frame["A"].copy()
        result = frame["A"].copy()
        result.loc["foo":"bar"] = 0
        expected.loc["foo"] = 0
        expected.loc["bar"] = 0
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer, exp_idx, exp_values",
        [
            (
                slice("2019-2", None),
                DatetimeIndex(["2019-02-01"], dtype="M8[ns]"),
                [2, 3],
            ),
            (
                slice(None, "2019-2"),
                date_range("2019", periods=2, freq="MS"),
                [0, 1, 2, 3],
            ),
        ],
    )
    def test_partial_getitem_loc_datetime(self, indexer, exp_idx, exp_values):
        # GH: 25165
        date_idx = date_range("2019", periods=2, freq="MS")
        df = DataFrame(
            list(range(4)),
            index=MultiIndex.from_product([date_idx, [0, 1]], names=["x", "y"]),
        )
        expected = DataFrame(
            exp_values,
            index=MultiIndex.from_product([exp_idx, [0, 1]], names=["x", "y"]),
        )
        result = df[indexer]
        tm.assert_frame_equal(result, expected)
        result = df.loc[indexer]
        tm.assert_frame_equal(result, expected)

        result = df.loc(axis=0)[indexer]
        tm.assert_frame_equal(result, expected)

        result = df.loc[indexer, :]
        tm.assert_frame_equal(result, expected)

        df2 = df.swaplevel(0, 1).sort_index()
        expected = expected.swaplevel(0, 1).sort_index()

        result = df2.loc[:, indexer, :]
        tm.assert_frame_equal(result, expected)


def test_loc_getitem_partial_both_axis():
    # gh-12660
    iterables = [["a", "b"], [2, 1]]
    columns = MultiIndex.from_product(iterables, names=["col1", "col2"])
    rows = MultiIndex.from_product(iterables, names=["row1", "row2"])
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)), index=rows, columns=columns
    )
    expected = df.iloc[:2, 2:].droplevel("row1").droplevel("col1", axis=1)
    result = df.loc["a", "b"]
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_operator.py ===
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.physics.quantum.operator import (Operator, UnitaryOperator,
                                            HermitianOperator, OuterProduct,
                                            DifferentialOperator,
                                            IdentityOperator)
from sympy.physics.quantum.state import Ket, Bra, Wavefunction
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import JzKet, JzBra
from sympy.physics.quantum.trace import Tr
from sympy.matrices import eye

from sympy.testing.pytest import warns_deprecated_sympy


class CustomKet(Ket):
    @classmethod
    def default_args(self):
        return ("t",)


class CustomOp(HermitianOperator):
    @classmethod
    def default_args(self):
        return ("T",)

t_ket = CustomKet()
t_op = CustomOp()


def test_operator():
    A = Operator('A')
    B = Operator('B')
    C = Operator('C')

    assert isinstance(A, Operator)
    assert isinstance(A, QExpr)

    assert A.label == (Symbol('A'),)
    assert A.is_commutative is False
    assert A.hilbert_space == HilbertSpace()

    assert A*B != B*A

    assert (A*(B + C)).expand() == A*B + A*C
    assert ((A + B)**2).expand() == A**2 + A*B + B*A + B**2

    assert t_op.label[0] == Symbol(t_op.default_args()[0])

    assert Operator() == Operator("O")
    with warns_deprecated_sympy():
        assert A*IdentityOperator() == A


def test_operator_inv():
    A = Operator('A')
    assert A*A.inv() == 1
    assert A.inv()*A == 1


def test_hermitian():
    H = HermitianOperator('H')

    assert isinstance(H, HermitianOperator)
    assert isinstance(H, Operator)

    assert Dagger(H) == H
    assert H.inv() != H
    assert H.is_commutative is False
    assert Dagger(H).is_commutative is False


def test_unitary():
    U = UnitaryOperator('U')

    assert isinstance(U, UnitaryOperator)
    assert isinstance(U, Operator)

    assert U.inv() == Dagger(U)
    assert U*Dagger(U) == 1
    assert Dagger(U)*U == 1
    assert U.is_commutative is False
    assert Dagger(U).is_commutative is False


def test_identity():
    with warns_deprecated_sympy():
        I = IdentityOperator()
        O = Operator('O')
        x = Symbol("x")
        three = sympify(3)

        assert isinstance(I, IdentityOperator)
        assert isinstance(I, Operator)

        assert I * O == O
        assert O * I == O
        assert I * Dagger(O) == Dagger(O)
        assert Dagger(O) * I == Dagger(O)
        assert isinstance(I * I, IdentityOperator)
        assert three * I == three
        assert I * x == x
        assert I.inv() == I
        assert Dagger(I) == I
        assert qapply(I * O) == O
        assert qapply(O * I) == O

        for n in [2, 3, 5]:
            assert represent(IdentityOperator(n)) == eye(n)


def test_outer_product():
    k = Ket('k')
    b = Bra('b')
    op = OuterProduct(k, b)

    assert isinstance(op, OuterProduct)
    assert isinstance(op, Operator)

    assert op.ket == k
    assert op.bra == b
    assert op.label == (k, b)
    assert op.is_commutative is False

    op = k*b

    assert isinstance(op, OuterProduct)
    assert isinstance(op, Operator)

    assert op.ket == k
    assert op.bra == b
    assert op.label == (k, b)
    assert op.is_commutative is False

    op = 2*k*b

    assert op == Mul(Integer(2), k, b)

    op = 2*(k*b)

    assert op == Mul(Integer(2), OuterProduct(k, b))

    assert Dagger(k*b) == OuterProduct(Dagger(b), Dagger(k))
    assert Dagger(k*b).is_commutative is False

    #test the _eval_trace
    assert Tr(OuterProduct(JzKet(1, 1), JzBra(1, 1))).doit() == 1

    # test scaled kets and bras
    assert OuterProduct(2 * k, b) == 2 * OuterProduct(k, b)
    assert OuterProduct(k, 2 * b) == 2 * OuterProduct(k, b)

    # test sums of kets and bras
    k1, k2 = Ket('k1'), Ket('k2')
    b1, b2 = Bra('b1'), Bra('b2')
    assert (OuterProduct(k1 + k2, b1) ==
            OuterProduct(k1, b1) + OuterProduct(k2, b1))
    assert (OuterProduct(k1, b1 + b2) ==
            OuterProduct(k1, b1) + OuterProduct(k1, b2))
    assert (OuterProduct(1 * k1 + 2 * k2, 3 * b1 + 4 * b2) ==
            3 * OuterProduct(k1, b1) +
            4 * OuterProduct(k1, b2) +
            6 * OuterProduct(k2, b1) +
            8 * OuterProduct(k2, b2))


def test_operator_dagger():
    A = Operator('A')
    B = Operator('B')
    assert Dagger(A*B) == Dagger(B)*Dagger(A)
    assert Dagger(A + B) == Dagger(A) + Dagger(B)
    assert Dagger(A**2) == Dagger(A)**2


def test_differential_operator():
    x = Symbol('x')
    f = Function('f')
    d = DifferentialOperator(Derivative(f(x), x), f(x))
    g = Wavefunction(x**2, x)
    assert qapply(d*g) == Wavefunction(2*x, x)
    assert d.expr == Derivative(f(x), x)
    assert d.function == f(x)
    assert d.variables == (x,)
    assert diff(d, x) == DifferentialOperator(Derivative(f(x), x, 2), f(x))

    d = DifferentialOperator(Derivative(f(x), x, 2), f(x))
    g = Wavefunction(x**3, x)
    assert qapply(d*g) == Wavefunction(6*x, x)
    assert d.expr == Derivative(f(x), x, 2)
    assert d.function == f(x)
    assert d.variables == (x,)
    assert diff(d, x) == DifferentialOperator(Derivative(f(x), x, 3), f(x))

    d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
    assert d.expr == 1/x*Derivative(f(x), x)
    assert d.function == f(x)
    assert d.variables == (x,)
    assert diff(d, x) == \
        DifferentialOperator(Derivative(1/x*Derivative(f(x), x), x), f(x))
    assert qapply(d*g) == Wavefunction(3*x, x)

    # 2D cartesian Laplacian
    y = Symbol('y')
    d = DifferentialOperator(Derivative(f(x, y), x, 2) +
                             Derivative(f(x, y), y, 2), f(x, y))
    w = Wavefunction(x**3*y**2 + y**3*x**2, x, y)
    assert d.expr == Derivative(f(x, y), x, 2) + Derivative(f(x, y), y, 2)
    assert d.function == f(x, y)
    assert d.variables == (x, y)
    assert diff(d, x) == \
        DifferentialOperator(Derivative(d.expr, x), f(x, y))
    assert diff(d, y) == \
        DifferentialOperator(Derivative(d.expr, y), f(x, y))
    assert qapply(d*w) == Wavefunction(2*x**3 + 6*x*y**2 + 6*x**2*y + 2*y**3,
                                       x, y)

    # 2D polar Laplacian (th = theta)
    r, th = symbols('r th')
    d = DifferentialOperator(1/r*Derivative(r*Derivative(f(r, th), r), r) +
                             1/(r**2)*Derivative(f(r, th), th, 2), f(r, th))
    w = Wavefunction(r**2*sin(th), r, (th, 0, pi))
    assert d.expr == \
        1/r*Derivative(r*Derivative(f(r, th), r), r) + \
        1/(r**2)*Derivative(f(r, th), th, 2)
    assert d.function == f(r, th)
    assert d.variables == (r, th)
    assert diff(d, r) == \
        DifferentialOperator(Derivative(d.expr, r), f(r, th))
    assert diff(d, th) == \
        DifferentialOperator(Derivative(d.expr, th), f(r, th))
    assert qapply(d*w) == Wavefunction(3*sin(th), r, (th, 0, pi))


def test_eval_power():
    from sympy.core import Pow
    from sympy.core.expr import unchanged
    O = Operator('O')
    U = UnitaryOperator('U')
    H = HermitianOperator('H')
    assert O**-1 == O.inv() # same as doc test
    assert U**-1 == U.inv()
    assert H**-1 == H.inv()
    x = symbols("x", commutative = True)
    assert unchanged(Pow, H, x) # verify Pow(H,x)=="X^n"
    assert H**x == Pow(H, x)
    assert Pow(H,x) == Pow(H, x, evaluate=False) # Just check
    from sympy.physics.quantum.gate import XGate
    X = XGate(0) # is hermitian and unitary
    assert unchanged(Pow, X, x) # verify Pow(X,x)=="X^x"
    assert X**x == Pow(X, x)
    assert Pow(X, x, evaluate=False) == Pow(X, x) # Just check
    n = symbols("n", integer=True, even=True)
    assert X**n == 1
    n = symbols("n", integer=True, odd=True)
    assert X**n == X
    n = symbols("n", integer=True)
    assert unchanged(Pow, X, n) # verify Pow(X,n)=="X^n"
    assert X**n == Pow(X, n)
    assert Pow(X, n, evaluate=False)==Pow(X, n) # Just check
    assert X**4 == 1
    assert X**7 == X

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_monomials.py ===
"""Tests for tools and arithmetics for monomials of distributed polynomials. """

from sympy.polys.monomials import (
    itermonomials, monomial_count,
    monomial_mul, monomial_div,
    monomial_gcd, monomial_lcm,
    monomial_max, monomial_min,
    monomial_divides, monomial_pow,
    Monomial,
)

from sympy.polys.polyerrors import ExactQuotientFailed

from sympy.abc import a, b, c, x, y, z
from sympy.core import S, symbols
from sympy.testing.pytest import raises

def test_monomials():

    # total_degree tests
    assert set(itermonomials([], 0)) == {S.One}
    assert set(itermonomials([], 1)) == {S.One}
    assert set(itermonomials([], 2)) == {S.One}

    assert set(itermonomials([], 0, 0)) == {S.One}
    assert set(itermonomials([], 1, 0)) == {S.One}
    assert set(itermonomials([], 2, 0)) == {S.One}

    raises(StopIteration, lambda: next(itermonomials([], 0, 1)))
    raises(StopIteration, lambda: next(itermonomials([], 0, 2)))
    raises(StopIteration, lambda: next(itermonomials([], 0, 3)))

    assert set(itermonomials([], 0, 1)) == set()
    assert set(itermonomials([], 0, 2)) == set()
    assert set(itermonomials([], 0, 3)) == set()

    raises(ValueError, lambda: set(itermonomials([], -1)))
    raises(ValueError, lambda: set(itermonomials([x], -1)))
    raises(ValueError, lambda: set(itermonomials([x, y], -1)))

    assert set(itermonomials([x], 0)) == {S.One}
    assert set(itermonomials([x], 1)) == {S.One, x}
    assert set(itermonomials([x], 2)) == {S.One, x, x**2}
    assert set(itermonomials([x], 3)) == {S.One, x, x**2, x**3}

    assert set(itermonomials([x, y], 0)) == {S.One}
    assert set(itermonomials([x, y], 1)) == {S.One, x, y}
    assert set(itermonomials([x, y], 2)) == {S.One, x, y, x**2, y**2, x*y}
    assert set(itermonomials([x, y], 3)) == \
            {S.One, x, y, x**2, x**3, y**2, y**3, x*y, x*y**2, y*x**2}

    i, j, k = symbols('i j k', commutative=False)
    assert set(itermonomials([i, j, k], 0)) == {S.One}
    assert set(itermonomials([i, j, k], 1)) == {S.One, i, j, k}
    assert set(itermonomials([i, j, k], 2)) == \
           {S.One, i, j, k, i**2, j**2, k**2, i*j, i*k, j*i, j*k, k*i, k*j}

    assert set(itermonomials([i, j, k], 3)) == \
            {S.One, i, j, k, i**2, j**2, k**2, i*j, i*k, j*i, j*k, k*i, k*j,
                    i**3, j**3, k**3,
                    i**2 * j, i**2 * k, j * i**2, k * i**2,
                    j**2 * i, j**2 * k, i * j**2, k * j**2,
                    k**2 * i, k**2 * j, i * k**2, j * k**2,
                    i*j*i, i*k*i, j*i*j, j*k*j, k*i*k, k*j*k,
                    i*j*k, i*k*j, j*i*k, j*k*i, k*i*j, k*j*i,
            }

    assert set(itermonomials([x, i, j], 0)) == {S.One}
    assert set(itermonomials([x, i, j], 1)) == {S.One, x, i, j}
    assert set(itermonomials([x, i, j], 2)) == {S.One, x, i, j, x*i, x*j, i*j, j*i, x**2, i**2, j**2}
    assert set(itermonomials([x, i, j], 3)) == \
            {S.One, x, i, j, x*i, x*j, i*j, j*i, x**2, i**2, j**2,
                            x**3, i**3, j**3,
                            x**2 * i, x**2 * j,
                            x * i**2, j * i**2, i**2 * j, i*j*i,
                            x * j**2, i * j**2, j**2 * i, j*i*j,
                            x * i * j, x * j * i
            }

    # degree_list tests
    assert set(itermonomials([], [])) == {S.One}

    raises(ValueError, lambda: set(itermonomials([], [0])))
    raises(ValueError, lambda: set(itermonomials([], [1])))
    raises(ValueError, lambda: set(itermonomials([], [2])))

    raises(ValueError, lambda: set(itermonomials([x], [1], [])))
    raises(ValueError, lambda: set(itermonomials([x], [1, 2], [])))
    raises(ValueError, lambda: set(itermonomials([x], [1, 2, 3], [])))

    raises(ValueError, lambda: set(itermonomials([x], [], [1])))
    raises(ValueError, lambda: set(itermonomials([x], [], [1, 2])))
    raises(ValueError, lambda: set(itermonomials([x], [], [1, 2, 3])))

    raises(ValueError, lambda: set(itermonomials([x, y], [1, 2], [1, 2, 3])))
    raises(ValueError, lambda: set(itermonomials([x, y, z], [1, 2, 3], [0, 1])))

    raises(ValueError, lambda: set(itermonomials([x], [1], [-1])))
    raises(ValueError, lambda: set(itermonomials([x, y], [1, 2], [1, -1])))

    raises(ValueError, lambda: set(itermonomials([], [], 1)))
    raises(ValueError, lambda: set(itermonomials([], [], 2)))
    raises(ValueError, lambda: set(itermonomials([], [], 3)))

    raises(ValueError, lambda: set(itermonomials([x, y], [0, 1], [1, 2])))
    raises(ValueError, lambda: set(itermonomials([x, y, z], [0, 0, 3], [0, 1, 2])))

    assert set(itermonomials([x], [0])) == {S.One}
    assert set(itermonomials([x], [1])) == {S.One, x}
    assert set(itermonomials([x], [2])) == {S.One, x, x**2}
    assert set(itermonomials([x], [3])) == {S.One, x, x**2, x**3}

    assert set(itermonomials([x], [3], [1])) == {x, x**3, x**2}
    assert set(itermonomials([x], [3], [2])) == {x**3, x**2}

    assert set(itermonomials([x, y], 3, 3)) == {x**3, x**2*y, x*y**2, y**3}
    assert set(itermonomials([x, y], 3, 2)) == {x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3}

    assert set(itermonomials([x, y], [0, 0])) == {S.One}
    assert set(itermonomials([x, y], [0, 1])) == {S.One, y}
    assert set(itermonomials([x, y], [0, 2])) == {S.One, y, y**2}
    assert set(itermonomials([x, y], [0, 2], [0, 1])) == {y, y**2}
    assert set(itermonomials([x, y], [0, 2], [0, 2])) == {y**2}

    assert set(itermonomials([x, y], [1, 0])) == {S.One, x}
    assert set(itermonomials([x, y], [1, 1])) == {S.One, x, y, x*y}
    assert set(itermonomials([x, y], [1, 2])) == {S.One, x, y, x*y, y**2, x*y**2}
    assert set(itermonomials([x, y], [1, 2], [1, 1])) == {x*y, x*y**2}
    assert set(itermonomials([x, y], [1, 2], [1, 2])) == {x*y**2}

    assert set(itermonomials([x, y], [2, 0])) == {S.One, x, x**2}
    assert set(itermonomials([x, y], [2, 1])) == {S.One, x, y, x*y, x**2, x**2*y}
    assert set(itermonomials([x, y], [2, 2])) == \
            {S.One, y**2, x*y**2, x, x*y, x**2, x**2*y**2, y, x**2*y}

    i, j, k = symbols('i j k', commutative=False)
    assert set(itermonomials([i, j, k], 2, 2)) == \
            {k*i, i**2, i*j, j*k, j*i, k**2, j**2, k*j, i*k}
    assert set(itermonomials([i, j, k], 3, 2)) == \
            {j*k**2, i*k**2, k*i*j, k*i**2, k**2, j*k*j, k*j**2, i*k*i, i*j,
                    j**2*k, i**2*j, j*i*k, j**3, i**3, k*j*i, j*k*i, j*i,
                    k**2*j, j*i**2, k*j, k*j*k, i*j*i, j*i*j, i*j**2, j**2,
                    k*i*k, i**2, j*k, i*k, i*k*j, k**3, i**2*k, j**2*i, k**2*i,
                    i*j*k, k*i
            }
    assert set(itermonomials([i, j, k], [0, 0, 0])) == {S.One}
    assert set(itermonomials([i, j, k], [0, 0, 1])) == {1, k}
    assert set(itermonomials([i, j, k], [0, 1, 0])) == {1, j}
    assert set(itermonomials([i, j, k], [1, 0, 0])) == {i, 1}
    assert set(itermonomials([i, j, k], [0, 0, 2])) == {k**2, 1, k}
    assert set(itermonomials([i, j, k], [0, 2, 0])) == {1, j, j**2}
    assert set(itermonomials([i, j, k], [2, 0, 0])) == {i, 1, i**2}
    assert set(itermonomials([i, j, k], [1, 1, 1])) == {1, k, j, j*k, i*k, i, i*j, i*j*k}
    assert set(itermonomials([i, j, k], [2, 2, 2])) == \
            {1, k, i**2*k**2, j*k, j**2, i, i*k, j*k**2, i*j**2*k**2,
                    i**2*j, i**2*j**2, k**2, j**2*k, i*j**2*k,
                    j**2*k**2, i*j, i**2*k, i**2*j**2*k, j, i**2*j*k,
                    i*j**2, i*k**2, i*j*k, i**2*j**2*k**2, i*j*k**2, i**2, i**2*j*k**2
            }

    assert set(itermonomials([x, j, k], [0, 0, 0])) == {S.One}
    assert set(itermonomials([x, j, k], [0, 0, 1])) == {1, k}
    assert set(itermonomials([x, j, k], [0, 1, 0])) == {1, j}
    assert set(itermonomials([x, j, k], [1, 0, 0])) == {x, 1}
    assert set(itermonomials([x, j, k], [0, 0, 2])) == {k**2, 1, k}
    assert set(itermonomials([x, j, k], [0, 2, 0])) == {1, j, j**2}
    assert set(itermonomials([x, j, k], [2, 0, 0])) == {x, 1, x**2}
    assert set(itermonomials([x, j, k], [1, 1, 1])) == {1, k, j, j*k, x*k, x, x*j, x*j*k}
    assert set(itermonomials([x, j, k], [2, 2, 2])) == \
            {1, k, x**2*k**2, j*k, j**2, x, x*k, j*k**2, x*j**2*k**2,
                    x**2*j, x**2*j**2, k**2, j**2*k, x*j**2*k,
                    j**2*k**2, x*j, x**2*k, x**2*j**2*k, j, x**2*j*k,
                    x*j**2, x*k**2, x*j*k, x**2*j**2*k**2, x*j*k**2, x**2, x**2*j*k**2
            }

def test_monomial_count():
    assert monomial_count(2, 2) == 6
    assert monomial_count(2, 3) == 10

def test_monomial_mul():
    assert monomial_mul((3, 4, 1), (1, 2, 0)) == (4, 6, 1)

def test_monomial_div():
    assert monomial_div((3, 4, 1), (1, 2, 0)) == (2, 2, 1)

def test_monomial_gcd():
    assert monomial_gcd((3, 4, 1), (1, 2, 0)) == (1, 2, 0)

def test_monomial_lcm():
    assert monomial_lcm((3, 4, 1), (1, 2, 0)) == (3, 4, 1)

def test_monomial_max():
    assert monomial_max((3, 4, 5), (0, 5, 1), (6, 3, 9)) == (6, 5, 9)

def test_monomial_pow():
    assert monomial_pow((1, 2, 3), 3) == (3, 6, 9)

def test_monomial_min():
    assert monomial_min((3, 4, 5), (0, 5, 1), (6, 3, 9)) == (0, 3, 1)

def test_monomial_divides():
    assert monomial_divides((1, 2, 3), (4, 5, 6)) is True
    assert monomial_divides((1, 2, 3), (0, 5, 6)) is False

def test_Monomial():
    m = Monomial((3, 4, 1), (x, y, z))
    n = Monomial((1, 2, 0), (x, y, z))

    assert m.as_expr() == x**3*y**4*z
    assert n.as_expr() == x**1*y**2

    assert m.as_expr(a, b, c) == a**3*b**4*c
    assert n.as_expr(a, b, c) == a**1*b**2

    assert m.exponents == (3, 4, 1)
    assert m.gens == (x, y, z)

    assert n.exponents == (1, 2, 0)
    assert n.gens == (x, y, z)

    assert m == (3, 4, 1)
    assert n != (3, 4, 1)
    assert m != (1, 2, 0)
    assert n == (1, 2, 0)
    assert (m == 1) is False

    assert m[0] == m[-3] == 3
    assert m[1] == m[-2] == 4
    assert m[2] == m[-1] == 1

    assert n[0] == n[-3] == 1
    assert n[1] == n[-2] == 2
    assert n[2] == n[-1] == 0

    assert m[:2] == (3, 4)
    assert n[:2] == (1, 2)

    assert m*n == Monomial((4, 6, 1))
    assert m/n == Monomial((2, 2, 1))

    assert m*(1, 2, 0) == Monomial((4, 6, 1))
    assert m/(1, 2, 0) == Monomial((2, 2, 1))

    assert m.gcd(n) == Monomial((1, 2, 0))
    assert m.lcm(n) == Monomial((3, 4, 1))

    assert m.gcd((1, 2, 0)) == Monomial((1, 2, 0))
    assert m.lcm((1, 2, 0)) == Monomial((3, 4, 1))

    assert m**0 == Monomial((0, 0, 0))
    assert m**1 == m
    assert m**2 == Monomial((6, 8, 2))
    assert m**3 == Monomial((9, 12, 3))
    _a = Monomial((0, 0, 0))
    for n in range(10):
        assert _a == m**n
        _a *= m

    raises(ExactQuotientFailed, lambda: m/Monomial((5, 2, 0)))

    mm = Monomial((1, 2, 3))
    raises(ValueError, lambda: mm.as_expr())
    assert str(mm) == 'Monomial((1, 2, 3))'
    assert str(m) == 'x**3*y**4*z**1'
    raises(NotImplementedError, lambda: m*1)
    raises(NotImplementedError, lambda: m/1)
    raises(ValueError, lambda: m**-1)
    raises(TypeError, lambda: m.gcd(3))
    raises(TypeError, lambda: m.lcm(3))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_join.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Interval,
    MultiIndex,
    Series,
    StringDtype,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "other", [Index(["three", "one", "two"]), Index(["one"]), Index(["one", "three"])]
)
def test_join_level(idx, other, join_type):
    join_index, lidx, ridx = other.join(
        idx, how=join_type, level="second", return_indexers=True
    )

    exp_level = other.join(idx.levels[1], how=join_type)
    assert join_index.levels[0].equals(idx.levels[0])
    assert join_index.levels[1].equals(exp_level)

    # pare down levels
    mask = np.array([x[1] in exp_level for x in idx], dtype=bool)
    exp_values = idx.values[mask]
    tm.assert_numpy_array_equal(join_index.values, exp_values)

    if join_type in ("outer", "inner"):
        join_index2, ridx2, lidx2 = idx.join(
            other, how=join_type, level="second", return_indexers=True
        )

        assert join_index.equals(join_index2)
        tm.assert_numpy_array_equal(lidx, lidx2)
        tm.assert_numpy_array_equal(ridx, ridx2)
        tm.assert_numpy_array_equal(join_index2.values, exp_values)


def test_join_level_corner_case(idx):
    # some corner cases
    index = Index(["three", "one", "two"])
    result = index.join(idx, level="second")
    assert isinstance(result, MultiIndex)

    with pytest.raises(TypeError, match="Join.*MultiIndex.*ambiguous"):
        idx.join(idx, level=1)


def test_join_self(idx, join_type):
    result = idx.join(idx, how=join_type)
    expected = idx
    if join_type == "outer":
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)


def test_join_multi():
    # GH 10665
    midx = MultiIndex.from_product([np.arange(4), np.arange(4)], names=["a", "b"])
    idx = Index([1, 2, 5], name="b")

    # inner
    jidx, lidx, ridx = midx.join(idx, how="inner", return_indexers=True)
    exp_idx = MultiIndex.from_product([np.arange(4), [1, 2]], names=["a", "b"])
    exp_lidx = np.array([1, 2, 5, 6, 9, 10, 13, 14], dtype=np.intp)
    exp_ridx = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.intp)
    tm.assert_index_equal(jidx, exp_idx)
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)
    # flip
    jidx, ridx, lidx = idx.join(midx, how="inner", return_indexers=True)
    tm.assert_index_equal(jidx, exp_idx)
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)

    # keep MultiIndex
    jidx, lidx, ridx = midx.join(idx, how="left", return_indexers=True)
    exp_ridx = np.array(
        [-1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1], dtype=np.intp
    )
    tm.assert_index_equal(jidx, midx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)
    # flip
    jidx, ridx, lidx = idx.join(midx, how="right", return_indexers=True)
    tm.assert_index_equal(jidx, midx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_join_multi_wrong_order():
    # GH 25760
    # GH 28956

    midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["b", "a"])

    join_idx, lidx, ridx = midx1.join(midx2, return_indexers=True)

    exp_ridx = np.array([-1, -1, -1, -1], dtype=np.intp)

    tm.assert_index_equal(midx1, join_idx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_join_multi_return_indexers():
    # GH 34074

    midx1 = MultiIndex.from_product([[1, 2], [3, 4], [5, 6]], names=["a", "b", "c"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])

    result = midx1.join(midx2, return_indexers=False)
    tm.assert_index_equal(result, midx1)


def test_join_overlapping_interval_level():
    # GH 44096
    idx_1 = MultiIndex.from_tuples(
        [
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (1, Interval(2.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),  # interval limit is here at 3.0, not at 2.0
            (2, Interval(3.0, 5.0)),
        ],
        names=["num", "interval"],
    )

    idx_2 = MultiIndex.from_tuples(
        [
            (1, Interval(2.0, 5.0)),
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (2, Interval(3.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),
        ],
        names=["num", "interval"],
    )

    expected = MultiIndex.from_tuples(
        [
            (1, Interval(0.0, 1.0)),
            (1, Interval(1.0, 2.0)),
            (1, Interval(2.0, 5.0)),
            (2, Interval(0.0, 1.0)),
            (2, Interval(1.0, 3.0)),
            (2, Interval(3.0, 5.0)),
        ],
        names=["num", "interval"],
    )
    result = idx_1.join(idx_2, how="outer")

    tm.assert_index_equal(result, expected)


def test_join_midx_ea():
    # GH#49277
    midx = MultiIndex.from_arrays(
        [Series([1, 1, 3], dtype="Int64"), Series([1, 2, 3], dtype="Int64")],
        names=["a", "b"],
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1], dtype="Int64"), Series([3], dtype="Int64")], names=["a", "c"]
    )
    result = midx.join(midx2, how="inner")
    expected = MultiIndex.from_arrays(
        [
            Series([1, 1], dtype="Int64"),
            Series([1, 2], dtype="Int64"),
            Series([3, 3], dtype="Int64"),
        ],
        names=["a", "b", "c"],
    )
    tm.assert_index_equal(result, expected)


def test_join_midx_string():
    # GH#49277
    midx = MultiIndex.from_arrays(
        [
            Series(["a", "a", "c"], dtype=StringDtype()),
            Series(["a", "b", "c"], dtype=StringDtype()),
        ],
        names=["a", "b"],
    )
    midx2 = MultiIndex.from_arrays(
        [Series(["a"], dtype=StringDtype()), Series(["c"], dtype=StringDtype())],
        names=["a", "c"],
    )
    result = midx.join(midx2, how="inner")
    expected = MultiIndex.from_arrays(
        [
            Series(["a", "a"], dtype=StringDtype()),
            Series(["a", "b"], dtype=StringDtype()),
            Series(["c", "c"], dtype=StringDtype()),
        ],
        names=["a", "b", "c"],
    )
    tm.assert_index_equal(result, expected)


def test_join_multi_with_nan():
    # GH29252
    df1 = DataFrame(
        data={"col1": [1.1, 1.2]},
        index=MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"]),
    )
    df2 = DataFrame(
        data={"col2": [2.1, 2.2]},
        index=MultiIndex.from_product([["A"], [np.nan, 2.0]], names=["id1", "id2"]),
    )
    result = df1.join(df2)
    expected = DataFrame(
        data={"col1": [1.1, 1.2], "col2": [np.nan, 2.2]},
        index=MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val", [0, 5])
def test_join_dtypes(any_numeric_ea_dtype, val):
    # GH#49830
    midx = MultiIndex.from_arrays([Series([1, 2], dtype=any_numeric_ea_dtype), [3, 4]])
    midx2 = MultiIndex.from_arrays(
        [Series([1, val, val], dtype=any_numeric_ea_dtype), [3, 4, 4]]
    )
    result = midx.join(midx2, how="outer")
    expected = MultiIndex.from_arrays(
        [Series([val, val, 1, 2], dtype=any_numeric_ea_dtype), [4, 4, 3, 4]]
    ).sort_values()
    tm.assert_index_equal(result, expected)


def test_join_dtypes_all_nan(any_numeric_ea_dtype):
    # GH#49830
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [np.nan, np.nan]]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 0, 0], dtype=any_numeric_ea_dtype), [np.nan, np.nan, np.nan]]
    )
    result = midx.join(midx2, how="outer")
    expected = MultiIndex.from_arrays(
        [
            Series([0, 0, 1, 2], dtype=any_numeric_ea_dtype),
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    tm.assert_index_equal(result, expected)


def test_join_index_levels():
    # GH#53093
    midx = midx = MultiIndex.from_tuples([("a", "2019-02-01"), ("a", "2019-02-01")])
    midx2 = MultiIndex.from_tuples([("a", "2019-01-31")])
    result = midx.join(midx2, how="outer")
    expected = MultiIndex.from_tuples(
        [("a", "2019-01-31"), ("a", "2019-02-01"), ("a", "2019-02-01")]
    )
    tm.assert_index_equal(result.levels[1], expected.levels[1])
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_common.py ===
from datetime import datetime

from dateutil.tz.tz import tzlocal
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timestamp,
)
from pandas.compat import (
    IS64,
    is_platform_windows,
)

from pandas.tseries.offsets import (
    FY5253,
    BDay,
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BusinessHour,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CDay,
    CustomBusinessHour,
    DateOffset,
    FY5253Quarter,
    LastWeekOfMonth,
    MonthBegin,
    MonthEnd,
    QuarterEnd,
    SemiMonthBegin,
    SemiMonthEnd,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)


def _get_offset(klass, value=1, normalize=False):
    # create instance from offset class
    if klass is FY5253:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is FY5253Quarter:
        klass = klass(
            n=value,
            startingMonth=1,
            weekday=1,
            qtr_with_extra_week=1,
            variation="last",
            normalize=normalize,
        )
    elif klass is LastWeekOfMonth:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is WeekOfMonth:
        klass = klass(n=value, week=1, weekday=5, normalize=normalize)
    elif klass is Week:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is DateOffset:
        klass = klass(days=value, normalize=normalize)
    else:
        klass = klass(value, normalize=normalize)
    return klass


@pytest.fixture(
    params=[
        BDay,
        BusinessHour,
        BMonthEnd,
        BMonthBegin,
        BQuarterEnd,
        BQuarterBegin,
        BYearEnd,
        BYearBegin,
        CDay,
        CustomBusinessHour,
        CBMonthEnd,
        CBMonthBegin,
        MonthEnd,
        MonthBegin,
        SemiMonthBegin,
        SemiMonthEnd,
        QuarterEnd,
        LastWeekOfMonth,
        WeekOfMonth,
        Week,
        YearBegin,
        YearEnd,
        FY5253,
        FY5253Quarter,
        DateOffset,
    ]
)
def _offset(request):
    return request.param


@pytest.fixture
def dt(_offset):
    if _offset in (CBMonthBegin, CBMonthEnd, BDay):
        return Timestamp(2008, 1, 1)
    elif _offset is (CustomBusinessHour, BusinessHour):
        return Timestamp(2014, 7, 1, 10, 00)
    return Timestamp(2008, 1, 2)


def test_apply_out_of_range(request, tz_naive_fixture, _offset):
    tz = tz_naive_fixture

    # try to create an out-of-bounds result timestamp; if we can't create
    # the offset skip
    try:
        if _offset in (BusinessHour, CustomBusinessHour):
            # Using 10000 in BusinessHour fails in tz check because of DST
            # difference
            offset = _get_offset(_offset, value=100000)
        else:
            offset = _get_offset(_offset, value=10000)

        result = Timestamp("20080101") + offset
        assert isinstance(result, datetime)
        assert result.tzinfo is None

        # Check tz is preserved
        t = Timestamp("20080101", tz=tz)
        result = t + offset
        assert isinstance(result, datetime)
        if tz is not None:
            assert t.tzinfo is not None

        if isinstance(tz, tzlocal) and not IS64 and _offset is not DateOffset:
            # If we hit OutOfBoundsDatetime on non-64 bit machines
            # we'll drop out of the try clause before the next test
            request.applymarker(
                pytest.mark.xfail(reason="OverflowError inside tzlocal past 2038")
            )
        elif (
            isinstance(tz, tzlocal)
            and is_platform_windows()
            and _offset in (QuarterEnd, BQuarterBegin, BQuarterEnd)
        ):
            request.applymarker(
                pytest.mark.xfail(reason="After GH#49737 t.tzinfo is None on CI")
            )
        assert str(t.tzinfo) == str(result.tzinfo)

    except OutOfBoundsDatetime:
        pass
    except (ValueError, KeyError):
        # we are creating an invalid offset
        # so ignore
        pass


def test_offsets_compare_equal(_offset):
    # root cause of GH#456: __ne__ was not implemented
    offset1 = _offset()
    offset2 = _offset()
    assert not offset1 != offset2
    assert offset1 == offset2


@pytest.mark.parametrize(
    "date, offset2",
    [
        [Timestamp(2008, 1, 1), BDay(2)],
        [Timestamp(2014, 7, 1, 10, 00), BusinessHour(n=3)],
        [
            Timestamp(2014, 7, 1, 10),
            CustomBusinessHour(
                holidays=["2014-06-27", Timestamp(2014, 6, 30), Timestamp("2014-07-02")]
            ),
        ],
        [Timestamp(2008, 1, 2), SemiMonthEnd(2)],
        [Timestamp(2008, 1, 2), SemiMonthBegin(2)],
        [Timestamp(2008, 1, 2), Week(2)],
        [Timestamp(2008, 1, 2), WeekOfMonth(2)],
        [Timestamp(2008, 1, 2), LastWeekOfMonth(2)],
    ],
)
def test_rsub(date, offset2):
    assert date - offset2 == (-offset2)._apply(date)


@pytest.mark.parametrize(
    "date, offset2",
    [
        [Timestamp(2008, 1, 1), BDay(2)],
        [Timestamp(2014, 7, 1, 10, 00), BusinessHour(n=3)],
        [
            Timestamp(2014, 7, 1, 10),
            CustomBusinessHour(
                holidays=["2014-06-27", Timestamp(2014, 6, 30), Timestamp("2014-07-02")]
            ),
        ],
        [Timestamp(2008, 1, 2), SemiMonthEnd(2)],
        [Timestamp(2008, 1, 2), SemiMonthBegin(2)],
        [Timestamp(2008, 1, 2), Week(2)],
        [Timestamp(2008, 1, 2), WeekOfMonth(2)],
        [Timestamp(2008, 1, 2), LastWeekOfMonth(2)],
    ],
)
def test_radd(date, offset2):
    assert date + offset2 == offset2 + date


@pytest.mark.parametrize(
    "date, offset_box, offset2",
    [
        [Timestamp(2008, 1, 1), BDay, BDay(2)],
        [Timestamp(2008, 1, 2), SemiMonthEnd, SemiMonthEnd(2)],
        [Timestamp(2008, 1, 2), SemiMonthBegin, SemiMonthBegin(2)],
        [Timestamp(2008, 1, 2), Week, Week(2)],
        [Timestamp(2008, 1, 2), WeekOfMonth, WeekOfMonth(2)],
        [Timestamp(2008, 1, 2), LastWeekOfMonth, LastWeekOfMonth(2)],
    ],
)
def test_sub(date, offset_box, offset2):
    off = offset2
    msg = "Cannot subtract datetime from offset"
    with pytest.raises(TypeError, match=msg):
        off - date

    assert 2 * off - off == off
    assert date - offset2 == date + offset_box(-2)
    assert date - offset2 == date - (2 * off - off)


@pytest.mark.parametrize(
    "offset_box, offset1",
    [
        [BDay, BDay()],
        [LastWeekOfMonth, LastWeekOfMonth()],
        [WeekOfMonth, WeekOfMonth()],
        [Week, Week()],
        [SemiMonthBegin, SemiMonthBegin()],
        [SemiMonthEnd, SemiMonthEnd()],
        [CustomBusinessHour, CustomBusinessHour(weekmask="Tue Wed Thu Fri")],
        [BusinessHour, BusinessHour()],
    ],
)
def test_Mult1(offset_box, offset1):
    dt = Timestamp(2008, 1, 2)
    assert dt + 10 * offset1 == dt + offset_box(10)
    assert dt + 5 * offset1 == dt + offset_box(5)


def test_compare_str(_offset):
    # GH#23524
    # comparing to strings that cannot be cast to DateOffsets should
    #  not raise for __eq__ or __ne__
    off = _get_offset(_offset)

    assert not off == "infer"
    assert off != "foo"
    # Note: inequalities are only implemented for Tick subclasses;
    #  tests for this are in test_ticks

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_plane.py ===
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.geometry import Line, Point, Ray, Segment, Point3D, Line3D, Ray3D, Segment3D, Plane, Circle
from sympy.geometry.util import are_coplanar
from sympy.testing.pytest import raises


def test_plane():
    x, y, z, u, v = symbols('x y z u v', real=True)
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    p3 = Point3D(1, 2, 3)
    pl3 = Plane(p1, p2, p3)
    pl4 = Plane(p1, normal_vector=(1, 1, 1))
    pl4b = Plane(p1, p2)
    pl5 = Plane(p3, normal_vector=(1, 2, 3))
    pl6 = Plane(Point3D(2, 3, 7), normal_vector=(2, 2, 2))
    pl7 = Plane(Point3D(1, -5, -6), normal_vector=(1, -2, 1))
    pl8 = Plane(p1, normal_vector=(0, 0, 1))
    pl9 = Plane(p1, normal_vector=(0, 12, 0))
    pl10 = Plane(p1, normal_vector=(-2, 0, 0))
    pl11 = Plane(p2, normal_vector=(0, 0, 1))
    l1 = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    l2 = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    l3 = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))

    raises(ValueError, lambda: Plane(p1, p1, p1))

    assert Plane(p1, p2, p3) != Plane(p1, p3, p2)
    assert Plane(p1, p2, p3).is_coplanar(Plane(p1, p3, p2))
    assert Plane(p1, p2, p3).is_coplanar(p1)
    assert Plane(p1, p2, p3).is_coplanar(Circle(p1, 1)) is False
    assert Plane(p1, normal_vector=(0, 0, 1)).is_coplanar(Circle(p1, 1))

    assert pl3 == Plane(Point3D(0, 0, 0), normal_vector=(1, -2, 1))
    assert pl3 != pl4
    assert pl4 == pl4b
    assert pl5 == Plane(Point3D(1, 2, 3), normal_vector=(1, 2, 3))

    assert pl5.equation(x, y, z) == x + 2*y + 3*z - 14
    assert pl3.equation(x, y, z) == x - 2*y + z

    assert pl3.p1 == p1
    assert pl4.p1 == p1
    assert pl5.p1 == p3

    assert pl4.normal_vector == (1, 1, 1)
    assert pl5.normal_vector == (1, 2, 3)

    assert p1 in pl3
    assert p1 in pl4
    assert p3 in pl5

    assert pl3.projection(Point(0, 0)) == p1
    p = pl3.projection(Point3D(1, 1, 0))
    assert p == Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6))
    assert p in pl3

    l = pl3.projection_line(Line(Point(0, 0), Point(1, 1)))
    assert l == Line3D(Point3D(0, 0, 0), Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6)))
    assert l in pl3
    # get a segment that does not intersect the plane which is also
    # parallel to pl3's normal veector
    t = Dummy()
    r = pl3.random_point()
    a = pl3.perpendicular_line(r).arbitrary_point(t)
    s = Segment3D(a.subs(t, 1), a.subs(t, 2))
    assert s.p1 not in pl3 and s.p2 not in pl3
    assert pl3.projection_line(s).equals(r)
    assert pl3.projection_line(Segment(Point(1, 0), Point(1, 1))) == \
               Segment3D(Point3D(Rational(5, 6), Rational(1, 3), Rational(-1, 6)), Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6)))
    assert pl6.projection_line(Ray(Point(1, 0), Point(1, 1))) == \
               Ray3D(Point3D(Rational(14, 3), Rational(11, 3), Rational(11, 3)), Point3D(Rational(13, 3), Rational(13, 3), Rational(10, 3)))
    assert pl3.perpendicular_line(r.args) == pl3.perpendicular_line(r)

    assert pl3.is_parallel(pl6) is False
    assert pl4.is_parallel(pl6)
    assert pl3.is_parallel(Line(p1, p2))
    assert pl6.is_parallel(l1) is False

    assert pl3.is_perpendicular(pl6)
    assert pl4.is_perpendicular(pl7)
    assert pl6.is_perpendicular(pl7)
    assert pl6.is_perpendicular(pl4) is False
    assert pl6.is_perpendicular(l1) is False
    assert pl6.is_perpendicular(Line((0, 0, 0), (1, 1, 1)))
    assert pl6.is_perpendicular((1, 1)) is False

    assert pl6.distance(pl6.arbitrary_point(u, v)) == 0
    assert pl7.distance(pl7.arbitrary_point(u, v)) == 0
    assert pl6.distance(pl6.arbitrary_point(t)) == 0
    assert pl7.distance(pl7.arbitrary_point(t)) == 0
    assert pl6.p1.distance(pl6.arbitrary_point(t)).simplify() == 1
    assert pl7.p1.distance(pl7.arbitrary_point(t)).simplify() == 1
    assert pl3.arbitrary_point(t) == Point3D(-sqrt(30)*sin(t)/30 + \
        2*sqrt(5)*cos(t)/5, sqrt(30)*sin(t)/15 + sqrt(5)*cos(t)/5, sqrt(30)*sin(t)/6)
    assert pl3.arbitrary_point(u, v) == Point3D(2*u - v, u + 2*v, 5*v)

    assert pl7.distance(Point3D(1, 3, 5)) == 5*sqrt(6)/6
    assert pl6.distance(Point3D(0, 0, 0)) == 4*sqrt(3)
    assert pl6.distance(pl6.p1) == 0
    assert pl7.distance(pl6) == 0
    assert pl7.distance(l1) == 0
    assert pl6.distance(Segment3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == \
        pl6.distance(Point3D(1, 3, 4)) == 4*sqrt(3)/3
    assert pl6.distance(Segment3D(Point3D(1, 3, 4), Point3D(0, 3, 7))) == \
        pl6.distance(Point3D(0, 3, 7)) == 2*sqrt(3)/3
    assert pl6.distance(Segment3D(Point3D(0, 3, 7), Point3D(-1, 3, 10))) == 0
    assert pl6.distance(Segment3D(Point3D(-1, 3, 10), Point3D(-2, 3, 13))) == 0
    assert pl6.distance(Segment3D(Point3D(-2, 3, 13), Point3D(-3, 3, 16))) == \
        pl6.distance(Point3D(-2, 3, 13)) == 2*sqrt(3)/3
    assert pl6.distance(Plane(Point3D(5, 5, 5), normal_vector=(8, 8, 8))) == sqrt(3)
    assert pl6.distance(Ray3D(Point3D(1, 3, 4), direction_ratio=[1, 0, -3])) == 4*sqrt(3)/3
    assert pl6.distance(Ray3D(Point3D(2, 3, 1), direction_ratio=[-1, 0, 3])) == 0


    assert pl6.angle_between(pl3) == pi/2
    assert pl6.angle_between(pl6) == 0
    assert pl6.angle_between(pl4) == 0
    assert pl7.angle_between(Line3D(Point3D(2, 3, 5), Point3D(2, 4, 6))) == \
        -asin(sqrt(3)/6)
    assert pl6.angle_between(Ray3D(Point3D(2, 4, 1), Point3D(6, 5, 3))) == \
        asin(sqrt(7)/3)
    assert pl7.angle_between(Segment3D(Point3D(5, 6, 1), Point3D(1, 2, 4))) == \
        asin(7*sqrt(246)/246)

    assert are_coplanar(l1, l2, l3) is False
    assert are_coplanar(l1) is False
    assert are_coplanar(Point3D(2, 7, 2), Point3D(0, 0, 2),
        Point3D(1, 1, 2), Point3D(1, 2, 2))
    assert are_coplanar(Plane(p1, p2, p3), Plane(p1, p3, p2))
    assert Plane.are_concurrent(pl3, pl4, pl5) is False
    assert Plane.are_concurrent(pl6) is False
    raises(ValueError, lambda: Plane.are_concurrent(Point3D(0, 0, 0)))
    raises(ValueError, lambda: Plane((1, 2, 3), normal_vector=(0, 0, 0)))

    assert pl3.parallel_plane(Point3D(1, 2, 5)) == Plane(Point3D(1, 2, 5), \
                                                      normal_vector=(1, -2, 1))

    # perpendicular_plane
    p = Plane((0, 0, 0), (1, 0, 0))
    # default
    assert p.perpendicular_plane() == Plane(Point3D(0, 0, 0), (0, 1, 0))
    # 1 pt
    assert p.perpendicular_plane(Point3D(1, 0, 1)) == \
        Plane(Point3D(1, 0, 1), (0, 1, 0))
    # pts as tuples
    assert p.perpendicular_plane((1, 0, 1), (1, 1, 1)) == \
        Plane(Point3D(1, 0, 1), (0, 0, -1))
    # more than two planes
    raises(ValueError, lambda: p.perpendicular_plane((1, 0, 1), (1, 1, 1), (1, 1, 0)))

    a, b = Point3D(0, 0, 0), Point3D(0, 1, 0)
    Z = (0, 0, 1)
    p = Plane(a, normal_vector=Z)
    # case 4
    assert p.perpendicular_plane(a, b) == Plane(a, (1, 0, 0))
    n = Point3D(*Z)
    # case 1
    assert p.perpendicular_plane(a, n) == Plane(a, (-1, 0, 0))
    # case 2
    assert Plane(a, normal_vector=b.args).perpendicular_plane(a, a + b) == \
        Plane(Point3D(0, 0, 0), (1, 0, 0))
    # case 1&3
    assert Plane(b, normal_vector=Z).perpendicular_plane(b, b + n) == \
        Plane(Point3D(0, 1, 0), (-1, 0, 0))
    # case 2&3
    assert Plane(b, normal_vector=b.args).perpendicular_plane(n, n + b) == \
        Plane(Point3D(0, 0, 1), (1, 0, 0))

    p = Plane(a, normal_vector=(0, 0, 1))
    assert p.perpendicular_plane() == Plane(a, normal_vector=(1, 0, 0))

    assert pl6.intersection(pl6) == [pl6]
    assert pl4.intersection(pl4.p1) == [pl4.p1]
    assert pl3.intersection(pl6) == [
        Line3D(Point3D(8, 4, 0), Point3D(2, 4, 6))]
    assert pl3.intersection(Line3D(Point3D(1,2,4), Point3D(4,4,2))) == [
        Point3D(2, Rational(8, 3), Rational(10, 3))]
    assert pl3.intersection(Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
        ) == [Line3D(Point3D(-24, -12, 0), Point3D(-25, -13, -1))]
    assert pl6.intersection(Ray3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == [
        Point3D(-1, 3, 10)]
    assert pl6.intersection(Segment3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == []
    assert pl7.intersection(Line(Point(2, 3), Point(4, 2))) == [
        Point3D(Rational(13, 2), Rational(3, 4), 0)]
    r = Ray(Point(2, 3), Point(4, 2))
    assert Plane((1,2,0), normal_vector=(0,0,1)).intersection(r) == [
        Ray3D(Point(2, 3), Point(4, 2))]
    assert pl9.intersection(pl8) == [Line3D(Point3D(0, 0, 0), Point3D(12, 0, 0))]
    assert pl10.intersection(pl11) == [Line3D(Point3D(0, 0, 1), Point3D(0, 2, 1))]
    assert pl4.intersection(pl8) == [Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]
    assert pl11.intersection(pl8) == []
    assert pl9.intersection(pl11) == [Line3D(Point3D(0, 0, 1), Point3D(12, 0, 1))]
    assert pl9.intersection(pl4) == [Line3D(Point3D(0, 0, 0), Point3D(12, 0, -12))]
    assert pl3.random_point() in pl3
    assert pl3.random_point(seed=1) in pl3

    # test geometrical entity using equals
    assert pl4.intersection(pl4.p1)[0].equals(pl4.p1)
    assert pl3.intersection(pl6)[0].equals(Line3D(Point3D(8, 4, 0), Point3D(2, 4, 6)))
    pl8 = Plane((1, 2, 0), normal_vector=(0, 0, 1))
    assert pl8.intersection(Line3D(p1, (1, 12, 0)))[0].equals(Line((0, 0, 0), (0.1, 1.2, 0)))
    assert pl8.intersection(Ray3D(p1, (1, 12, 0)))[0].equals(Ray((0, 0, 0), (1, 12, 0)))
    assert pl8.intersection(Segment3D(p1, (21, 1, 0)))[0].equals(Segment3D(p1, (21, 1, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(0, 0, 112)))[0].equals(pl8)
    assert pl8.intersection(Plane(p1, normal_vector=(0, 12, 0)))[0].equals(
        Line3D(p1, direction_ratio=(112 * pi, 0, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(11, 0, 1)))[0].equals(
        Line3D(p1, direction_ratio=(0, -11, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(1, 0, 11)))[0].equals(
        Line3D(p1, direction_ratio=(0, 11, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(-1, -1, -11)))[0].equals(
        Line3D(p1, direction_ratio=(1, -1, 0)))
    assert pl3.random_point() in pl3
    assert len(pl8.intersection(Ray3D(Point3D(0, 2, 3), Point3D(1, 0, 3)))) == 0
    # check if two plane are equals
    assert pl6.intersection(pl6)[0].equals(pl6)
    assert pl8.equals(Plane(p1, normal_vector=(0, 12, 0))) is False
    assert pl8.equals(pl8)
    assert pl8.equals(Plane(p1, normal_vector=(0, 0, -12)))
    assert pl8.equals(Plane(p1, normal_vector=(0, 0, -12*sqrt(3))))
    assert pl8.equals(p1) is False

    # issue 8570
    l2 = Line3D(Point3D(Rational(50000004459633, 5000000000000),
                        Rational(-891926590718643, 1000000000000000),
                        Rational(231800966893633, 100000000000000)),
                Point3D(Rational(50000004459633, 50000000000000),
                        Rational(-222981647679771, 250000000000000),
                        Rational(231800966893633, 100000000000000)))

    p2 = Plane(Point3D(Rational(402775636372767, 100000000000000),
                       Rational(-97224357654973, 100000000000000),
                       Rational(216793600814789, 100000000000000)),
               (-S('9.00000087501922'), -S('4.81170658872543e-13'),
                S('0.0')))

    assert str([i.n(2) for i in p2.intersection(l2)]) == \
           '[Point3D(4.0, -0.89, 2.3)]'


def test_dimension_normalization():
    A = Plane(Point3D(1, 1, 2), normal_vector=(1, 1, 1))
    b = Point(1, 1)
    assert A.projection(b) == Point(Rational(5, 3), Rational(5, 3), Rational(2, 3))

    a, b = Point(0, 0), Point3D(0, 1)
    Z = (0, 0, 1)
    p = Plane(a, normal_vector=Z)
    assert p.perpendicular_plane(a, b) == Plane(Point3D(0, 0, 0), (1, 0, 0))
    assert Plane((1, 2, 1), (2, 1, 0), (3, 1, 2)
        ).intersection((2, 1)) == [Point(2, 1, 0)]


def test_parameter_value():
    t, u, v = symbols("t, u v")
    p1, p2, p3 = Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)
    p = Plane(p1, p2, p3)
    assert p.parameter_value((0, -3, 2), t) == {t: asin(2*sqrt(13)/13)}
    assert p.parameter_value((0, -3, 2), u, v) == {u: 3, v: 2}
    assert p.parameter_value(p1, t) == p1
    raises(ValueError, lambda: p.parameter_value((1, 0, 0), t))
    raises(ValueError, lambda: p.parameter_value(Line(Point(0, 0), Point(1, 1)), t))
    raises(ValueError, lambda: p.parameter_value((0, -3, 2), t, 1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_common.py ===
import collections
from functools import partial
import string
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version


def test_get_callable_name():
    getname = com.get_callable_name

    def fn(x):
        return x

    lambda_ = lambda x: x
    part1 = partial(fn)
    part2 = partial(part1)

    class somecall:
        def __call__(self):
            # This shouldn't actually get called below; somecall.__init__
            #  should.
            raise NotImplementedError

    assert getname(fn) == "fn"
    assert getname(lambda_)
    assert getname(part1) == "fn"
    assert getname(part2) == "fn"
    assert getname(somecall()) == "somecall"
    assert getname(1) is None


def test_any_none():
    assert com.any_none(1, 2, 3, None)
    assert not com.any_none(1, 2, 3, 4)


def test_all_not_none():
    assert com.all_not_none(1, 2, 3, 4)
    assert not com.all_not_none(1, 2, 3, None)
    assert not com.all_not_none(None, None, None, None)


def test_random_state():
    # Check with seed
    state = com.random_state(5)
    assert state.uniform() == np.random.RandomState(5).uniform()

    # Check with random state object
    state2 = np.random.RandomState(10)
    assert com.random_state(state2).uniform() == np.random.RandomState(10).uniform()

    # check with no arg random state
    assert com.random_state() is np.random

    # check array-like
    # GH32503
    state_arr_like = np.random.default_rng(None).integers(
        0, 2**31, size=624, dtype="uint32"
    )
    assert (
        com.random_state(state_arr_like).uniform()
        == np.random.RandomState(state_arr_like).uniform()
    )

    # Check BitGenerators
    # GH32503
    assert (
        com.random_state(np.random.MT19937(3)).uniform()
        == np.random.RandomState(np.random.MT19937(3)).uniform()
    )
    assert (
        com.random_state(np.random.PCG64(11)).uniform()
        == np.random.RandomState(np.random.PCG64(11)).uniform()
    )

    # Error for floats or strings
    msg = (
        "random_state must be an integer, array-like, a BitGenerator, Generator, "
        "a numpy RandomState, or None"
    )
    with pytest.raises(ValueError, match=msg):
        com.random_state("test")

    with pytest.raises(ValueError, match=msg):
        com.random_state(5.5)


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (Series([1], name="x"), Series([2], name="x"), "x"),
        (Series([1], name="x"), Series([2], name="y"), None),
        (Series([1]), Series([2], name="x"), None),
        (Series([1], name="x"), Series([2]), None),
        (Series([1], name="x"), [2], "x"),
        ([1], Series([2], name="y"), "y"),
        # matching NAs
        (Series([1], name=np.nan), pd.Index([], name=np.nan), np.nan),
        (Series([1], name=np.nan), pd.Index([], name=pd.NaT), None),
        (Series([1], name=pd.NA), pd.Index([], name=pd.NA), pd.NA),
        # tuple name GH#39757
        (
            Series([1], name=np.int64(1)),
            pd.Index([], name=(np.int64(1), np.int64(2))),
            None,
        ),
        (
            Series([1], name=(np.int64(1), np.int64(2))),
            pd.Index([], name=(np.int64(1), np.int64(2))),
            (np.int64(1), np.int64(2)),
        ),
        pytest.param(
            Series([1], name=(np.float64("nan"), np.int64(2))),
            pd.Index([], name=(np.float64("nan"), np.int64(2))),
            (np.float64("nan"), np.int64(2)),
            marks=pytest.mark.xfail(
                reason="Not checking for matching NAs inside tuples."
            ),
        ),
    ],
)
def test_maybe_match_name(left, right, expected):
    res = ops.common._maybe_match_name(left, right)
    assert res is expected or res == expected


def test_standardize_mapping():
    # No uninitialized defaultdicts
    msg = r"to_dict\(\) only accepts initialized defaultdicts"
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(collections.defaultdict)

    # No non-mapping subtypes, instance
    msg = "unsupported type: <class 'list'>"
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping([])

    # No non-mapping subtypes, class
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(list)

    fill = {"bad": "data"}
    assert com.standardize_mapping(fill) == dict

    # Convert instance to type
    assert com.standardize_mapping({}) == dict

    dd = collections.defaultdict(list)
    assert isinstance(com.standardize_mapping(dd), partial)


def test_git_version():
    # GH 21295
    git_version = pd.__git_version__
    assert len(git_version) == 40
    assert all(c in string.hexdigits for c in git_version)


def test_version_tag():
    version = Version(pd.__version__)
    try:
        version > Version("0.0.1")
    except TypeError:
        raise ValueError(
            "No git tags exist, please sync tags between upstream and your repo"
        )


@pytest.mark.parametrize(
    "obj", [(obj,) for obj in pd.__dict__.values() if callable(obj)]
)
def test_serializable(obj):
    # GH 35611
    unpickled = tm.round_trip_pickle(obj)
    assert type(obj) == type(unpickled)


class TestIsBoolIndexer:
    def test_non_bool_array_with_na(self):
        # in particular, this should not raise
        arr = np.array(["A", "B", np.nan], dtype=object)
        assert not com.is_bool_indexer(arr)

    def test_list_subclass(self):
        # GH#42433

        class MyList(list):
            pass

        val = MyList(["a"])

        assert not com.is_bool_indexer(val)

        val = MyList([True])
        assert com.is_bool_indexer(val)

    def test_frozenlist(self):
        # GH#42461
        data = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data=data)

        frozen = df.index.names[1:]
        assert not com.is_bool_indexer(frozen)

        result = df[frozen]
        expected = df[[]]
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("with_exception", [True, False])
def test_temp_setattr(with_exception):
    # GH#45954
    ser = Series(dtype=object)
    ser.name = "first"
    # Raise a ValueError in either case to satisfy pytest.raises
    match = "Inside exception raised" if with_exception else "Outside exception raised"
    with pytest.raises(ValueError, match=match):
        with com.temp_setattr(ser, "name", "second"):
            assert ser.name == "second"
            if with_exception:
                raise ValueError("Inside exception raised")
        raise ValueError("Outside exception raised")
    assert ser.name == "first"


@pytest.mark.single_cpu
def test_str_size():
    # GH#21758
    a = "a"
    expected = sys.getsizeof(a)
    pyexe = sys.executable.replace("\\", "/")
    call = [
        pyexe,
        "-c",
        "a='a';import sys;sys.getsizeof(a);import pandas;print(sys.getsizeof(a));",
    ]
    result = subprocess.check_output(call).decode()[-4:-1].strip("\n")
    assert int(result) == int(expected)


@pytest.mark.single_cpu
def test_bz2_missing_import():
    # Check whether bz2 missing import is handled correctly (issue #53857)
    code = """
        import sys
        sys.modules['bz2'] = None
        import pytest
        import pandas as pd
        from pandas.compat import get_bz2_file
        msg = 'bz2 module not available.'
        with pytest.raises(RuntimeError, match=msg):
            get_bz2_file()
    """
    code = textwrap.dedent(code)
    call = [sys.executable, "-c", code]
    subprocess.check_output(call)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\test_setops.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Series,
)
import pandas._testing as tm
from pandas.core.algorithms import safe_sort


def equal_contents(arr1, arr2) -> bool:
    """
    Checks if the set of unique elements of arr1 and arr2 are equivalent.
    """
    return frozenset(arr1) == frozenset(arr2)


class TestIndexSetOps:
    @pytest.mark.parametrize(
        "method", ["union", "intersection", "difference", "symmetric_difference"]
    )
    def test_setops_sort_validation(self, method):
        idx1 = Index(["a", "b"])
        idx2 = Index(["b", "c"])

        with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
            getattr(idx1, method)(idx2, sort=2)

        # sort=True is supported as of GH#??
        getattr(idx1, method)(idx2, sort=True)

    def test_setops_preserve_object_dtype(self):
        idx = Index([1, 2, 3], dtype=object)
        result = idx.intersection(idx[1:])
        expected = idx[1:]
        tm.assert_index_equal(result, expected)

        # if other is not monotonic increasing, intersection goes through
        #  a different route
        result = idx.intersection(idx[1:][::-1])
        tm.assert_index_equal(result, expected)

        result = idx._union(idx[1:], sort=None)
        expected = idx
        tm.assert_numpy_array_equal(result, expected.values)

        result = idx.union(idx[1:], sort=None)
        tm.assert_index_equal(result, expected)

        # if other is not monotonic increasing, _union goes through
        #  a different route
        result = idx._union(idx[1:][::-1], sort=None)
        tm.assert_numpy_array_equal(result, expected.values)

        result = idx.union(idx[1:][::-1], sort=None)
        tm.assert_index_equal(result, expected)

    def test_union_base(self):
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[3:]
        second = index[:5]

        result = first.union(second)

        expected = Index([0, 1, 2, "a", "b", "c"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    def test_union_different_type_base(self, klass):
        # GH 10149
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[3:]
        second = index[:5]

        result = first.union(klass(second.values))

        assert equal_contents(result, index)

    def test_union_sort_other_incomparable(self):
        # https://github.com/pandas-dev/pandas/issues/24959
        idx = Index([1, pd.Timestamp("2000")])
        # default (sort=None)
        with tm.assert_produces_warning(RuntimeWarning):
            result = idx.union(idx[:1])

        tm.assert_index_equal(result, idx)

        # sort=None
        with tm.assert_produces_warning(RuntimeWarning):
            result = idx.union(idx[:1], sort=None)
        tm.assert_index_equal(result, idx)

        # sort=False
        result = idx.union(idx[:1], sort=False)
        tm.assert_index_equal(result, idx)

    def test_union_sort_other_incomparable_true(self):
        idx = Index([1, pd.Timestamp("2000")])
        with pytest.raises(TypeError, match=".*"):
            idx.union(idx[:1], sort=True)

    def test_intersection_equal_sort_true(self):
        idx = Index(["c", "a", "b"])
        sorted_ = Index(["a", "b", "c"])
        tm.assert_index_equal(idx.intersection(idx, sort=True), sorted_)

    def test_intersection_base(self, sort):
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:5]
        second = index[:3]

        expected = Index([0, 1, "a"]) if sort is None else Index([0, "a", 1])
        result = first.intersection(second, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [np.array, Series, list])
    def test_intersection_different_type_base(self, klass, sort):
        # GH 10149
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:5]
        second = index[:3]

        result = first.intersection(klass(second.values), sort=sort)
        assert equal_contents(result, second)

    def test_intersection_nosort(self):
        result = Index(["c", "b", "a"]).intersection(["b", "a"])
        expected = Index(["b", "a"])
        tm.assert_index_equal(result, expected)

    def test_intersection_equal_sort(self):
        idx = Index(["c", "a", "b"])
        tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
        tm.assert_index_equal(idx.intersection(idx, sort=None), idx)

    def test_intersection_str_dates(self, sort):
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        i1 = Index(dt_dates, dtype=object)
        i2 = Index(["aa"], dtype=object)
        result = i2.intersection(i1, sort=sort)

        assert len(result) == 0

    @pytest.mark.parametrize(
        "index2,expected_arr",
        [(Index(["B", "D"]), ["B"]), (Index(["B", "D", "A"]), ["A", "B"])],
    )
    def test_intersection_non_monotonic_non_unique(self, index2, expected_arr, sort):
        # non-monotonic non-unique
        index1 = Index(["A", "B", "A", "C"])
        expected = Index(expected_arr)
        result = index1.intersection(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_difference_base(self, sort):
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:4]
        second = index[3:]

        result = first.difference(second, sort)
        expected = Index([0, "a", 1])
        if sort is None:
            expected = Index(safe_sort(expected))
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self):
        # (same results for py2 and py3 but sortedness not tested elsewhere)
        index = Index([0, "a", 1, "b", 2, "c"])
        first = index[:4]
        second = index[3:]

        result = first.symmetric_difference(second)
        expected = Index([0, 1, 2, "a", "c"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "method,expected,sort",
        [
            (
                "intersection",
                np.array(
                    [(1, "A"), (2, "A"), (1, "B"), (2, "B")],
                    dtype=[("num", int), ("let", "S1")],
                ),
                False,
            ),
            (
                "intersection",
                np.array(
                    [(1, "A"), (1, "B"), (2, "A"), (2, "B")],
                    dtype=[("num", int), ("let", "S1")],
                ),
                None,
            ),
            (
                "union",
                np.array(
                    [(1, "A"), (1, "B"), (1, "C"), (2, "A"), (2, "B"), (2, "C")],
                    dtype=[("num", int), ("let", "S1")],
                ),
                None,
            ),
        ],
    )
    def test_tuple_union_bug(self, method, expected, sort):
        index1 = Index(
            np.array(
                [(1, "A"), (2, "A"), (1, "B"), (2, "B")],
                dtype=[("num", int), ("let", "S1")],
            )
        )
        index2 = Index(
            np.array(
                [(1, "A"), (2, "A"), (1, "B"), (2, "B"), (1, "C"), (2, "C")],
                dtype=[("num", int), ("let", "S1")],
            )
        )

        result = getattr(index1, method)(index2, sort=sort)
        assert result.ndim == 1

        expected = Index(expected)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("first_list", [["b", "a"], []])
    @pytest.mark.parametrize("second_list", [["a", "b"], []])
    @pytest.mark.parametrize(
        "first_name, second_name, expected_name",
        [("A", "B", None), (None, "B", None), ("A", None, None)],
    )
    def test_union_name_preservation(
        self, first_list, second_list, first_name, second_name, expected_name, sort
    ):
        expected_dtype = object if not first_list or not second_list else "str"
        first = Index(first_list, name=first_name)
        second = Index(second_list, name=second_name)
        union = first.union(second, sort=sort)

        vals = set(first_list).union(second_list)

        if sort is None and len(first_list) > 0 and len(second_list) > 0:
            expected = Index(sorted(vals), name=expected_name)
            tm.assert_index_equal(union, expected)
        else:
            expected = Index(vals, name=expected_name, dtype=expected_dtype)
            tm.assert_index_equal(union.sort_values(), expected.sort_values())

    @pytest.mark.parametrize(
        "diff_type, expected",
        [["difference", [1, "B"]], ["symmetric_difference", [1, 2, "B", "C"]]],
    )
    def test_difference_object_type(self, diff_type, expected):
        # GH 13432
        idx1 = Index([0, 1, "A", "B"])
        idx2 = Index([0, 2, "A", "C"])
        result = getattr(idx1, diff_type)(idx2)
        expected = Index(expected)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_drop_duplicates.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, False, False, True, True, False])),
        ("last", Series([False, True, True, False, False, False, False])),
        (False, Series([False, True, True, False, True, True, False])),
    ],
)
def test_drop_duplicates(any_numpy_dtype, keep, expected):
    tc = Series([1, 0, 3, 5, 3, 0, 4], dtype=np.dtype(any_numpy_dtype))

    if tc.dtype == "bool":
        pytest.skip("tested separately in test_drop_duplicates_bool")

    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    tm.assert_series_equal(tc.drop_duplicates(keep=keep), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(keep=keep, inplace=True)
    assert return_value is None
    tm.assert_series_equal(sc, tc[~expected])


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, True])),
        ("last", Series([True, True, False, False])),
        (False, Series([True, True, True, True])),
    ],
)
def test_drop_duplicates_bool(keep, expected):
    tc = Series([True, False, True, False])

    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    tm.assert_series_equal(tc.drop_duplicates(keep=keep), tc[~expected])
    sc = tc.copy()
    return_value = sc.drop_duplicates(keep=keep, inplace=True)
    tm.assert_series_equal(sc, tc[~expected])
    assert return_value is None


@pytest.mark.parametrize("values", [[], list(range(5))])
def test_drop_duplicates_no_duplicates(any_numpy_dtype, keep, values):
    tc = Series(values, dtype=np.dtype(any_numpy_dtype))
    expected = Series([False] * len(tc), dtype="bool")

    if tc.dtype == "bool":
        # 0 -> False and 1-> True
        # any other value would be duplicated
        tc = tc[:2]
        expected = expected[:2]

    tm.assert_series_equal(tc.duplicated(keep=keep), expected)

    result_dropped = tc.drop_duplicates(keep=keep)
    tm.assert_series_equal(result_dropped, tc)

    # validate shallow copy
    assert result_dropped is not tc


class TestSeriesDropDuplicates:
    @pytest.fixture(
        params=["int_", "uint", "float64", "str_", "timedelta64[h]", "datetime64[D]"]
    )
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def cat_series_unused_category(self, dtype, ordered):
        # Test case 1
        cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))

        input1 = np.array([1, 2, 3, 3], dtype=np.dtype(dtype))
        cat = Categorical(input1, categories=cat_array, ordered=ordered)
        tc1 = Series(cat)
        return tc1

    def test_drop_duplicates_categorical_non_bool(self, cat_series_unused_category):
        tc1 = cat_series_unused_category

        expected = Series([False, False, False, True])

        result = tc1.duplicated()
        tm.assert_series_equal(result, expected)

        result = tc1.drop_duplicates()
        tm.assert_series_equal(result, tc1[~expected])

        sc = tc1.copy()
        return_value = sc.drop_duplicates(inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc1[~expected])

    def test_drop_duplicates_categorical_non_bool_keeplast(
        self, cat_series_unused_category
    ):
        tc1 = cat_series_unused_category

        expected = Series([False, False, True, False])

        result = tc1.duplicated(keep="last")
        tm.assert_series_equal(result, expected)

        result = tc1.drop_duplicates(keep="last")
        tm.assert_series_equal(result, tc1[~expected])

        sc = tc1.copy()
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc1[~expected])

    def test_drop_duplicates_categorical_non_bool_keepfalse(
        self, cat_series_unused_category
    ):
        tc1 = cat_series_unused_category

        expected = Series([False, False, True, True])

        result = tc1.duplicated(keep=False)
        tm.assert_series_equal(result, expected)

        result = tc1.drop_duplicates(keep=False)
        tm.assert_series_equal(result, tc1[~expected])

        sc = tc1.copy()
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc1[~expected])

    @pytest.fixture
    def cat_series(self, dtype, ordered):
        # no unused categories, unlike cat_series_unused_category
        cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))

        input2 = np.array([1, 2, 3, 5, 3, 2, 4], dtype=np.dtype(dtype))
        cat = Categorical(input2, categories=cat_array, ordered=ordered)
        tc2 = Series(cat)
        return tc2

    def test_drop_duplicates_categorical_non_bool2(self, cat_series):
        tc2 = cat_series

        expected = Series([False, False, False, False, True, True, False])

        result = tc2.duplicated()
        tm.assert_series_equal(result, expected)

        result = tc2.drop_duplicates()
        tm.assert_series_equal(result, tc2[~expected])

        sc = tc2.copy()
        return_value = sc.drop_duplicates(inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc2[~expected])

    def test_drop_duplicates_categorical_non_bool2_keeplast(self, cat_series):
        tc2 = cat_series

        expected = Series([False, True, True, False, False, False, False])

        result = tc2.duplicated(keep="last")
        tm.assert_series_equal(result, expected)

        result = tc2.drop_duplicates(keep="last")
        tm.assert_series_equal(result, tc2[~expected])

        sc = tc2.copy()
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc2[~expected])

    def test_drop_duplicates_categorical_non_bool2_keepfalse(self, cat_series):
        tc2 = cat_series

        expected = Series([False, True, True, False, True, True, False])

        result = tc2.duplicated(keep=False)
        tm.assert_series_equal(result, expected)

        result = tc2.drop_duplicates(keep=False)
        tm.assert_series_equal(result, tc2[~expected])

        sc = tc2.copy()
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc2[~expected])

    def test_drop_duplicates_categorical_bool(self, ordered):
        tc = Series(
            Categorical(
                [True, False, True, False], categories=[True, False], ordered=ordered
            )
        )

        expected = Series([False, False, True, True])
        tm.assert_series_equal(tc.duplicated(), expected)
        tm.assert_series_equal(tc.drop_duplicates(), tc[~expected])
        sc = tc.copy()
        return_value = sc.drop_duplicates(inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc[~expected])

        expected = Series([True, True, False, False])
        tm.assert_series_equal(tc.duplicated(keep="last"), expected)
        tm.assert_series_equal(tc.drop_duplicates(keep="last"), tc[~expected])
        sc = tc.copy()
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc[~expected])

        expected = Series([True, True, True, True])
        tm.assert_series_equal(tc.duplicated(keep=False), expected)
        tm.assert_series_equal(tc.drop_duplicates(keep=False), tc[~expected])
        sc = tc.copy()
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(sc, tc[~expected])

    def test_drop_duplicates_categorical_bool_na(self, nulls_fixture):
        # GH#44351
        ser = Series(
            Categorical(
                [True, False, True, False, nulls_fixture],
                categories=[True, False],
                ordered=True,
            )
        )
        result = ser.drop_duplicates()
        expected = Series(
            Categorical([True, False, np.nan], categories=[True, False], ordered=True),
            index=[0, 1, 4],
        )
        tm.assert_series_equal(result, expected)

    def test_drop_duplicates_ignore_index(self):
        # GH#48304
        ser = Series([1, 2, 2, 3])
        result = ser.drop_duplicates(ignore_index=True)
        expected = Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_duplicated_arrow_dtype(self):
        pytest.importorskip("pyarrow")
        ser = Series([True, False, None, False], dtype="bool[pyarrow]")
        result = ser.drop_duplicates()
        expected = Series([True, False, None], dtype="bool[pyarrow]")
        tm.assert_series_equal(result, expected)

    def test_drop_duplicates_arrow_strings(self):
        # GH#54904
        pa = pytest.importorskip("pyarrow")
        ser = Series(["a", "a"], dtype=pd.ArrowDtype(pa.string()))
        result = ser.drop_duplicates()
        expecetd = Series(["a"], dtype=pd.ArrowDtype(pa.string()))
        tm.assert_series_equal(result, expecetd)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_api.py ===
"""
Tests of the groupby API, including internal consistency and with other pandas objects.

Tests in this file should only check the existence, names, and arguments of groupby
methods. It should not test the results of any groupby operation.
"""

import inspect

import pytest

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.groupby.base import (
    groupby_other_methods,
    reduction_kernels,
    transformation_kernels,
)
from pandas.core.groupby.generic import (
    DataFrameGroupBy,
    SeriesGroupBy,
)


def test_tab_completion(multiindex_dataframe_random_data):
    grp = multiindex_dataframe_random_data.groupby(level="second")
    results = {v for v in dir(grp) if not v.startswith("_")}
    expected = {
        "A",
        "B",
        "C",
        "agg",
        "aggregate",
        "apply",
        "boxplot",
        "filter",
        "first",
        "get_group",
        "groups",
        "hist",
        "indices",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "ngroups",
        "nth",
        "ohlc",
        "plot",
        "prod",
        "size",
        "std",
        "sum",
        "transform",
        "var",
        "sem",
        "count",
        "nunique",
        "head",
        "describe",
        "cummax",
        "quantile",
        "rank",
        "cumprod",
        "tail",
        "resample",
        "cummin",
        "fillna",
        "cumsum",
        "cumcount",
        "ngroup",
        "all",
        "shift",
        "skew",
        "take",
        "pct_change",
        "any",
        "corr",
        "corrwith",
        "cov",
        "dtypes",
        "ndim",
        "diff",
        "idxmax",
        "idxmin",
        "ffill",
        "bfill",
        "rolling",
        "expanding",
        "pipe",
        "sample",
        "ewm",
        "value_counts",
    }
    assert results == expected


def test_all_methods_categorized(multiindex_dataframe_random_data):
    grp = multiindex_dataframe_random_data.groupby(
        multiindex_dataframe_random_data.iloc[:, 0]
    )
    names = {_ for _ in dir(grp) if not _.startswith("_")} - set(
        multiindex_dataframe_random_data.columns
    )
    new_names = set(names)
    new_names -= reduction_kernels
    new_names -= transformation_kernels
    new_names -= groupby_other_methods

    assert not reduction_kernels & transformation_kernels
    assert not reduction_kernels & groupby_other_methods
    assert not transformation_kernels & groupby_other_methods

    # new public method?
    if new_names:
        msg = f"""
There are uncategorized methods defined on the Grouper class:
{new_names}.

Was a new method recently added?

Every public method On Grouper must appear in exactly one the
following three lists defined in pandas.core.groupby.base:
- `reduction_kernels`
- `transformation_kernels`
- `groupby_other_methods`
see the comments in pandas/core/groupby/base.py for guidance on
how to fix this test.
        """
        raise AssertionError(msg)

    # removed a public method?
    all_categorized = reduction_kernels | transformation_kernels | groupby_other_methods
    if names != all_categorized:
        msg = f"""
Some methods which are supposed to be on the Grouper class
are missing:
{all_categorized - names}.

They're still defined in one of the lists that live in pandas/core/groupby/base.py.
If you removed a method, you should update them
"""
        raise AssertionError(msg)


def test_frame_consistency(groupby_func):
    # GH#48028
    if groupby_func in ("first", "last"):
        msg = "first and last are entirely different between frame and groupby"
        pytest.skip(reason=msg)

    if groupby_func in ("cumcount", "ngroup"):
        assert not hasattr(DataFrame, groupby_func)
        return

    frame_method = getattr(DataFrame, groupby_func)
    gb_method = getattr(DataFrameGroupBy, groupby_func)
    result = set(inspect.signature(gb_method).parameters)
    if groupby_func == "size":
        # "size" is a method on GroupBy but property on DataFrame:
        expected = {"self"}
    else:
        expected = set(inspect.signature(frame_method).parameters)

    # Exclude certain arguments from result and expected depending on the operation
    # Some of these may be purposeful inconsistencies between the APIs
    exclude_expected, exclude_result = set(), set()
    if groupby_func in ("any", "all"):
        exclude_expected = {"kwargs", "bool_only", "axis"}
    elif groupby_func in ("count",):
        exclude_expected = {"numeric_only", "axis"}
    elif groupby_func in ("nunique",):
        exclude_expected = {"axis"}
    elif groupby_func in ("max", "min"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"min_count", "engine", "engine_kwargs"}
    elif groupby_func in ("mean", "std", "sum", "var"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"engine", "engine_kwargs"}
    elif groupby_func in ("median", "prod", "sem"):
        exclude_expected = {"axis", "kwargs", "skipna"}
    elif groupby_func in ("backfill", "bfill", "ffill", "pad"):
        exclude_expected = {"downcast", "inplace", "axis", "limit_area"}
    elif groupby_func in ("cummax", "cummin"):
        exclude_expected = {"skipna", "args"}
        exclude_result = {"numeric_only"}
    elif groupby_func in ("cumprod", "cumsum"):
        exclude_expected = {"skipna"}
    elif groupby_func in ("pct_change",):
        exclude_expected = {"kwargs"}
        exclude_result = {"axis"}
    elif groupby_func in ("rank",):
        exclude_expected = {"numeric_only"}
    elif groupby_func in ("quantile",):
        exclude_expected = {"method", "axis"}

    # Ensure excluded arguments are actually in the signatures
    assert result & exclude_result == exclude_result
    assert expected & exclude_expected == exclude_expected

    result -= exclude_result
    expected -= exclude_expected
    assert result == expected


def test_series_consistency(request, groupby_func):
    # GH#48028
    if groupby_func in ("first", "last"):
        pytest.skip("first and last are entirely different between Series and groupby")

    if groupby_func in ("cumcount", "corrwith", "ngroup"):
        assert not hasattr(Series, groupby_func)
        return

    series_method = getattr(Series, groupby_func)
    gb_method = getattr(SeriesGroupBy, groupby_func)
    result = set(inspect.signature(gb_method).parameters)
    if groupby_func == "size":
        # "size" is a method on GroupBy but property on Series
        expected = {"self"}
    else:
        expected = set(inspect.signature(series_method).parameters)

    # Exclude certain arguments from result and expected depending on the operation
    # Some of these may be purposeful inconsistencies between the APIs
    exclude_expected, exclude_result = set(), set()
    if groupby_func in ("any", "all"):
        exclude_expected = {"kwargs", "bool_only", "axis"}
    elif groupby_func in ("diff",):
        exclude_result = {"axis"}
    elif groupby_func in ("max", "min"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"min_count", "engine", "engine_kwargs"}
    elif groupby_func in ("mean", "std", "sum", "var"):
        exclude_expected = {"axis", "kwargs", "skipna"}
        exclude_result = {"engine", "engine_kwargs"}
    elif groupby_func in ("median", "prod", "sem"):
        exclude_expected = {"axis", "kwargs", "skipna"}
    elif groupby_func in ("backfill", "bfill", "ffill", "pad"):
        exclude_expected = {"downcast", "inplace", "axis", "limit_area"}
    elif groupby_func in ("cummax", "cummin"):
        exclude_expected = {"skipna", "args"}
        exclude_result = {"numeric_only"}
    elif groupby_func in ("cumprod", "cumsum"):
        exclude_expected = {"skipna"}
    elif groupby_func in ("pct_change",):
        exclude_expected = {"kwargs"}
        exclude_result = {"axis"}
    elif groupby_func in ("rank",):
        exclude_expected = {"numeric_only"}
    elif groupby_func in ("idxmin", "idxmax"):
        exclude_expected = {"args", "kwargs"}
    elif groupby_func in ("quantile",):
        exclude_result = {"numeric_only"}

    # Ensure excluded arguments are actually in the signatures
    assert result & exclude_result == exclude_result
    assert expected & exclude_expected == exclude_expected

    result -= exclude_result
    expected -= exclude_expected
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_insert.py ===
from datetime import datetime

import numpy as np
import pytest
import pytz

from pandas import (
    NA,
    DatetimeIndex,
    Index,
    NaT,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestInsert:
    @pytest.mark.parametrize("null", [None, np.nan, np.datetime64("NaT"), NaT, NA])
    @pytest.mark.parametrize("tz", [None, "UTC", "US/Eastern"])
    def test_insert_nat(self, tz, null):
        # GH#16537, GH#18295 (test missing)

        idx = DatetimeIndex(["2017-01-01"], tz=tz)
        expected = DatetimeIndex(["NaT", "2017-01-01"], tz=tz)
        if tz is not None and isinstance(null, np.datetime64):
            expected = Index([null, idx[0]], dtype=object)

        res = idx.insert(0, null)
        tm.assert_index_equal(res, expected)

    @pytest.mark.parametrize("tz", [None, "UTC", "US/Eastern"])
    def test_insert_invalid_na(self, tz):
        idx = DatetimeIndex(["2017-01-01"], tz=tz)

        item = np.timedelta64("NaT")
        result = idx.insert(0, item)
        expected = Index([item] + list(idx), dtype=object)
        tm.assert_index_equal(result, expected)

    def test_insert_empty_preserves_freq(self, tz_naive_fixture):
        # GH#33573
        tz = tz_naive_fixture
        dti = DatetimeIndex([], tz=tz, freq="D")
        item = Timestamp("2017-04-05").tz_localize(tz)

        result = dti.insert(0, item)
        assert result.freq == dti.freq

        # But not when we insert an item that doesn't conform to freq
        dti = DatetimeIndex([], tz=tz, freq="W-THU")
        result = dti.insert(0, item)
        assert result.freq is None

    def test_insert(self, unit):
        idx = DatetimeIndex(
            ["2000-01-04", "2000-01-01", "2000-01-02"], name="idx"
        ).as_unit(unit)

        result = idx.insert(2, datetime(2000, 1, 5))
        exp = DatetimeIndex(
            ["2000-01-04", "2000-01-01", "2000-01-05", "2000-01-02"], name="idx"
        ).as_unit(unit)
        tm.assert_index_equal(result, exp)

        # insertion of non-datetime should coerce to object index
        result = idx.insert(1, "inserted")
        expected = Index(
            [
                datetime(2000, 1, 4),
                "inserted",
                datetime(2000, 1, 1),
                datetime(2000, 1, 2),
            ],
            name="idx",
        )
        assert not isinstance(result, DatetimeIndex)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

    def test_insert2(self, unit):
        idx = date_range("1/1/2000", periods=3, freq="ME", name="idx", unit=unit)

        # preserve freq
        expected_0 = DatetimeIndex(
            ["1999-12-31", "2000-01-31", "2000-02-29", "2000-03-31"],
            name="idx",
            freq="ME",
        ).as_unit(unit)
        expected_3 = DatetimeIndex(
            ["2000-01-31", "2000-02-29", "2000-03-31", "2000-04-30"],
            name="idx",
            freq="ME",
        ).as_unit(unit)

        # reset freq to None
        expected_1_nofreq = DatetimeIndex(
            ["2000-01-31", "2000-01-31", "2000-02-29", "2000-03-31"],
            name="idx",
            freq=None,
        ).as_unit(unit)
        expected_3_nofreq = DatetimeIndex(
            ["2000-01-31", "2000-02-29", "2000-03-31", "2000-01-02"],
            name="idx",
            freq=None,
        ).as_unit(unit)

        cases = [
            (0, datetime(1999, 12, 31), expected_0),
            (-3, datetime(1999, 12, 31), expected_0),
            (3, datetime(2000, 4, 30), expected_3),
            (1, datetime(2000, 1, 31), expected_1_nofreq),
            (3, datetime(2000, 1, 2), expected_3_nofreq),
        ]

        for n, d, expected in cases:
            result = idx.insert(n, d)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

    def test_insert3(self, unit):
        idx = date_range("1/1/2000", periods=3, freq="ME", name="idx", unit=unit)

        # reset freq to None
        result = idx.insert(3, datetime(2000, 1, 2))
        expected = DatetimeIndex(
            ["2000-01-31", "2000-02-29", "2000-03-31", "2000-01-02"],
            name="idx",
            freq=None,
        ).as_unit(unit)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq is None

    def test_insert4(self, unit):
        for tz in ["US/Pacific", "Asia/Singapore"]:
            idx = date_range(
                "1/1/2000 09:00", periods=6, freq="h", tz=tz, name="idx", unit=unit
            )
            # preserve freq
            expected = date_range(
                "1/1/2000 09:00", periods=7, freq="h", tz=tz, name="idx", unit=unit
            )
            for d in [
                Timestamp("2000-01-01 15:00", tz=tz),
                pytz.timezone(tz).localize(datetime(2000, 1, 1, 15)),
            ]:
                result = idx.insert(6, d)
                tm.assert_index_equal(result, expected)
                assert result.name == expected.name
                assert result.freq == expected.freq
                assert result.tz == expected.tz

            expected = DatetimeIndex(
                [
                    "2000-01-01 09:00",
                    "2000-01-01 10:00",
                    "2000-01-01 11:00",
                    "2000-01-01 12:00",
                    "2000-01-01 13:00",
                    "2000-01-01 14:00",
                    "2000-01-01 10:00",
                ],
                name="idx",
                tz=tz,
                freq=None,
            ).as_unit(unit)
            # reset freq to None
            for d in [
                Timestamp("2000-01-01 10:00", tz=tz),
                pytz.timezone(tz).localize(datetime(2000, 1, 1, 10)),
            ]:
                result = idx.insert(6, d)
                tm.assert_index_equal(result, expected)
                assert result.name == expected.name
                assert result.tz == expected.tz
                assert result.freq is None

    # TODO: also changes DataFrame.__setitem__ with expansion
    def test_insert_mismatched_tzawareness(self):
        # see GH#7299
        idx = date_range("1/1/2000", periods=3, freq="D", tz="Asia/Tokyo", name="idx")

        # mismatched tz-awareness
        item = Timestamp("2000-01-04")
        result = idx.insert(3, item)
        expected = Index(
            list(idx[:3]) + [item] + list(idx[3:]), dtype=object, name="idx"
        )
        tm.assert_index_equal(result, expected)

        # mismatched tz-awareness
        item = datetime(2000, 1, 4)
        result = idx.insert(3, item)
        expected = Index(
            list(idx[:3]) + [item] + list(idx[3:]), dtype=object, name="idx"
        )
        tm.assert_index_equal(result, expected)

    # TODO: also changes DataFrame.__setitem__ with expansion
    def test_insert_mismatched_tz(self):
        # see GH#7299
        # pre-2.0 with mismatched tzs we would cast to object
        idx = date_range("1/1/2000", periods=3, freq="D", tz="Asia/Tokyo", name="idx")

        # mismatched tz -> cast to object (could reasonably cast to same tz or UTC)
        item = Timestamp("2000-01-04", tz="US/Eastern")
        result = idx.insert(3, item)
        expected = Index(
            list(idx[:3]) + [item.tz_convert(idx.tz)] + list(idx[3:]),
            name="idx",
        )
        assert expected.dtype == idx.dtype
        tm.assert_index_equal(result, expected)

        item = datetime(2000, 1, 4, tzinfo=pytz.timezone("US/Eastern"))
        result = idx.insert(3, item)
        expected = Index(
            list(idx[:3]) + [item.astimezone(idx.tzinfo)] + list(idx[3:]),
            name="idx",
        )
        assert expected.dtype == idx.dtype
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "item", [0, np.int64(0), np.float64(0), np.array(0), np.timedelta64(456)]
    )
    def test_insert_mismatched_types_raises(self, tz_aware_fixture, item):
        # GH#33703 dont cast these to dt64
        tz = tz_aware_fixture
        dti = date_range("2019-11-04", periods=9, freq="-1D", name=9, tz=tz)

        result = dti.insert(1, item)

        if isinstance(item, np.ndarray):
            assert item.item() == 0
            expected = Index([dti[0], 0] + list(dti[1:]), dtype=object, name=9)
        else:
            expected = Index([dti[0], item] + list(dti[1:]), dtype=object, name=9)

        tm.assert_index_equal(result, expected)

    def test_insert_castable_str(self, tz_aware_fixture):
        # GH#33703
        tz = tz_aware_fixture
        dti = date_range("2019-11-04", periods=3, freq="-1D", name=9, tz=tz)

        value = "2019-11-05"
        result = dti.insert(0, value)

        ts = Timestamp(value).tz_localize(tz)
        expected = DatetimeIndex([ts] + list(dti), dtype=dti.dtype, name=9)
        tm.assert_index_equal(result, expected)

    def test_insert_non_castable_str(self, tz_aware_fixture):
        # GH#33703
        tz = tz_aware_fixture
        dti = date_range("2019-11-04", periods=3, freq="-1D", name=9, tz=tz)

        value = "foo"
        result = dti.insert(0, value)

        expected = Index(["foo"] + list(dti), dtype=object, name=9)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_html5lib.py ===
"""Tests to ensure that the html5lib tree builder generates good trees."""

import pytest
import warnings

from bs4 import BeautifulSoup
from bs4.filter import SoupStrainer
from . import (
    HTML5LIB_PRESENT,
    HTML5TreeBuilderSmokeTest,
)


@pytest.mark.skipif(
    not HTML5LIB_PRESENT,
    reason="html5lib seems not to be present, not testing its tree builder.",
)
class TestHTML5LibBuilder(HTML5TreeBuilderSmokeTest):
    """See ``HTML5TreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        from bs4.builder import HTML5TreeBuilder

        return HTML5TreeBuilder

    def test_soupstrainer(self):
        # The html5lib tree builder does not support parse_only.
        strainer = SoupStrainer("b")
        markup = "<p>A <b>bold</b> statement.</p>"
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup(markup, "html5lib", parse_only=strainer)
        assert soup.decode() == self.document_for(markup)

        [warning] = w
        assert warning.filename == __file__
        assert "the html5lib tree builder doesn't support parse_only" in str(
            warning.message
        )

    def test_correctly_nested_tables(self):
        """html5lib inserts <tbody> tags where other parsers don't."""
        markup = (
            '<table id="1">'
            "<tr>"
            "<td>Here's another table:"
            '<table id="2">'
            "<tr><td>foo</td></tr>"
            "</table></td>"
        )

        self.assert_soup(
            markup,
            '<table id="1"><tbody><tr><td>Here\'s another table:'
            '<table id="2"><tbody><tr><td>foo</td></tr></tbody></table>'
            "</td></tr></tbody></table>",
        )

        self.assert_soup(
            "<table><thead><tr><td>Foo</td></tr></thead>"
            "<tbody><tr><td>Bar</td></tr></tbody>"
            "<tfoot><tr><td>Baz</td></tr></tfoot></table>"
        )

    def test_xml_declaration_followed_by_doctype(self):
        markup = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html>
  <head>
  </head>
  <body>
   <p>foo</p>
  </body>
</html>"""
        soup = self.soup(markup)
        # Verify that we can reach the <p> tag; this means the tree is connected.
        assert b"<p>foo</p>" == soup.p.encode()

    def test_reparented_markup(self):
        markup = "<p><em>foo</p>\n<p>bar<a></a></em></p>"
        soup = self.soup(markup)
        assert (
            "<body><p><em>foo</em></p><em>\n</em><p><em>bar<a></a></em></p></body>"
            == soup.body.decode()
        )
        assert 2 == len(soup.find_all("p"))

    def test_reparented_markup_ends_with_whitespace(self):
        markup = "<p><em>foo</p>\n<p>bar<a></a></em></p>\n"
        soup = self.soup(markup)
        assert (
            "<body><p><em>foo</em></p><em>\n</em><p><em>bar<a></a></em></p>\n</body>"
            == soup.body.decode()
        )
        assert 2 == len(soup.find_all("p"))

    def test_reparented_markup_containing_identical_whitespace_nodes(self):
        """Verify that we keep the two whitespace nodes in this
        document distinct when reparenting the adjacent <tbody> tags.
        """
        markup = "<table> <tbody><tbody><ims></tbody> </table>"
        soup = self.soup(markup)
        space1, space2 = soup.find_all(string=" ")
        tbody1, tbody2 = soup.find_all("tbody")
        assert space1.next_element is tbody1
        assert tbody2.next_element is space2

    def test_reparented_markup_containing_children(self):
        markup = (
            "<div><a>aftermath<p><noscript>target</noscript>aftermath</a></p></div>"
        )
        soup = self.soup(markup)
        noscript = soup.noscript
        assert "target" == noscript.next_element
        target = soup.find(string="target")

        # The 'aftermath' string was duplicated; we want the second one.
        final_aftermath = soup.find_all(string="aftermath")[-1]

        # The <noscript> tag was moved beneath a copy of the <a> tag,
        # but the 'target' string within is still connected to the
        # (second) 'aftermath' string.
        assert final_aftermath == target.next_element
        assert target == final_aftermath.previous_element

    def test_processing_instruction(self):
        """Processing instructions become comments."""
        markup = b"""<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert str(soup).startswith("<!--?PITarget PIContent?-->")

    def test_cloned_multivalue_node(self):
        markup = b"""<a class="my_class"><p></a>"""
        soup = self.soup(markup)
        a1, a2 = soup.find_all("a")
        assert a1 == a2
        assert a1 is not a2

    def test_foster_parenting(self):
        markup = b"""<table><td></tbody>A"""
        soup = self.soup(markup)
        assert (
            "<body>A<table><tbody><tr><td></td></tr></tbody></table></body>"
            == soup.body.decode()
        )

    def test_extraction(self):
        """
        Test that extraction does not destroy the tree.

        https://bugs.launchpad.net/beautifulsoup/+bug/1782928
        """

        markup = """
<html><head></head>
<style>
</style><script></script><body><p>hello</p></body></html>
"""
        soup = self.soup(markup)
        [s.extract() for s in soup("script")]
        [s.extract() for s in soup("style")]

        assert len(soup.find_all("p")) == 1

    def test_empty_comment(self):
        """
        Test that empty comment does not break structure.

        https://bugs.launchpad.net/beautifulsoup/+bug/1806598
        """

        markup = """
<html>
<body>
<form>
<!----><input type="text">
</form>
</body>
</html>
"""
        soup = self.soup(markup)
        inputs = []
        for form in soup.find_all("form"):
            inputs.extend(form.find_all("input"))
        assert len(inputs) == 1

    def test_tracking_line_numbers(self):
        # The html.parser TreeBuilder keeps track of line number and
        # position of each element.
        markup = "\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>"
        soup = self.soup(markup)
        assert 2 == soup.p.sourceline
        assert 5 == soup.p.sourcepos
        assert "sourceline" == soup.p.find("sourceline").name

        # You can deactivate this behavior.
        soup = self.soup(markup, store_line_numbers=False)
        assert None is soup.p.sourceline
        assert None is soup.p.sourcepos

    def test_special_string_containers(self):
        # The html5lib tree builder doesn't support this standard feature,
        # because there's no way of knowing, when a string is created,
        # where in the tree it will eventually end up.
        pass

    def test_html5_attributes(self):
        # The html5lib TreeBuilder can convert any entity named in
        # the HTML5 spec to a sequence of Unicode characters, and
        # convert those Unicode characters to a (potentially
        # different) named entity on the way out.
        #
        # This is a copy of the same test from
        # HTMLParserTreeBuilderSmokeTest.  It's not in the superclass
        # because the lxml HTML TreeBuilder _doesn't_ work this way.
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

    @pytest.mark.parametrize(
        "name,value",
        [("document_declared_encoding", "utf8"), ("exclude_encodings", ["utf8"])],
    )
    def test_prepare_markup_warnings(self, name, value):
        # html5lib doesn't support a couple of the common arguments to
        # prepare_markup.
        builder = self.default_builder()
        kwargs = {name: value}
        with warnings.catch_warnings(record=True) as w:
            list(builder.prepare_markup("a", **kwargs))
        [warning] = w
        msg = str(warning.message)
        assert (
            msg
            == f"You provided a value for {name}, but the html5lib tree builder doesn't support {name}."
        )

    def test_doctype_filtered(self):
        # Since the html5lib parser doesn't support parse_only, this standard
        # smoke-test test can't be run.
        pass