
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_view.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    Series,
    array,
    date_range,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Series.view is deprecated and will be removed in a future version.:FutureWarning"  # noqa: E501
)


class TestView:
    def test_view_i8_to_datetimelike(self):
        dti = date_range("2000", periods=4, tz="US/Central")
        ser = Series(dti.asi8)

        result = ser.view(dti.dtype)
        tm.assert_datetime_array_equal(result._values, dti._data._with_freq(None))

        pi = dti.tz_localize(None).to_period("D")
        ser = Series(pi.asi8)
        result = ser.view(pi.dtype)
        tm.assert_period_array_equal(result._values, pi._data)

    def test_view_tz(self):
        # GH#24024
        ser = Series(date_range("2000", periods=4, tz="US/Central"))
        result = ser.view("i8")
        expected = Series(
            [
                946706400000000000,
                946792800000000000,
                946879200000000000,
                946965600000000000,
            ]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "first", ["m8[ns]", "M8[ns]", "M8[ns, US/Central]", "period[D]"]
    )
    @pytest.mark.parametrize(
        "second", ["m8[ns]", "M8[ns]", "M8[ns, US/Central]", "period[D]"]
    )
    @pytest.mark.parametrize("box", [Series, Index, array])
    def test_view_between_datetimelike(self, first, second, box):
        dti = date_range("2016-01-01", periods=3)

        orig = box(dti)
        obj = orig.view(first)
        assert obj.dtype == first
        tm.assert_numpy_array_equal(np.asarray(obj.view("i8")), dti.asi8)

        res = obj.view(second)
        assert res.dtype == second
        tm.assert_numpy_array_equal(np.asarray(obj.view("i8")), dti.asi8)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_random.py ===
import random
from sympy.core.random import random as rand, seed, shuffle, _assumptions_shuffle
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.trigonometric import sin, acos
from sympy.abc import x


def test_random():
    random.seed(42)
    a = random.random()
    random.seed(42)
    Symbol('z').is_finite
    b = random.random()
    assert a == b

    got = set()
    for i in range(2):
        random.seed(28)
        m0, m1 = symbols('m_0 m_1', real=True)
        _ = acos(-m0/m1)
        got.add(random.uniform(0,1))
    assert len(got) == 1

    random.seed(10)
    y = 0
    for i in range(4):
        y += sin(random.uniform(-10,10) * x)
    random.seed(10)
    z = 0
    for i in range(4):
        z += sin(random.uniform(-10,10) * x)
    assert y == z


def test_seed():
    assert rand() < 1
    seed(1)
    a = rand()
    b = rand()
    seed(1)
    c = rand()
    d = rand()
    assert a == c
    if not c == d:
        assert a != b
    else:
        assert a == b

    abc = 'abc'
    first = list(abc)
    second = list(abc)
    third = list(abc)

    seed(123)
    shuffle(first)

    seed(123)
    shuffle(second)
    _assumptions_shuffle(third)

    assert first == second == third

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\tests\test_arch.py ===
from sympy.physics.continuum_mechanics.arch import Arch
from sympy import Symbol, simplify

x = Symbol('x')
t = Symbol('t')

def test_arch_init():
    a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
    assert a.get_loads == {'distributed': {}, 'concentrated': {}}
    assert a.reaction_force == {Symbol('R_A_x'):0, Symbol('R_A_y'):0, Symbol('R_B_x'):0, Symbol('R_B_y'):0}
    assert a.supports == {'left':'hinge', 'right':'hinge'}
    assert a.left_support == (0,0)
    assert a.right_support == (10,0)
    assert a.get_shape_eqn == 5 - ((x-5)**2)/5

    a = Arch((0,0),(10,1),crown_x=6)
    a.change_support_type(left_support='roller')
    a.add_member(0.5)
    assert a.supports == {'left':'roller', 'right':'hinge'}
    assert simplify(a.get_shape_eqn) == simplify(9/5 - (x - 6)**2/20)

def test_arch_support():
    a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
    a.apply_load(-1,'C',8,150,angle=270)
    a.apply_load(0,'D',start=20,end=40,mag=-4)
    a.solve()
    assert abs(a.reaction_force[Symbol("R_A_x")] - 83.33333333333333) < 10e-12
    assert abs(a.reaction_force[Symbol("R_B_y")] - 90.00000000000000) < 10e-12
    assert abs(a.reaction_force[Symbol("R_B_x")] + 83.33333333333333) < 10e-12
    assert abs(a.reaction_force[Symbol("R_A_y")] - 140.00000000000000) < 10e-12

def test_arch_member():
    a = Arch((0,0),(40,0),crown_x=20,crown_y=15)
    a.change_support_type(right_support='roller')
    a.add_member(0)
    a.apply_load(-1,'D',start=12,mag=3,angle=270)
    a.apply_load(-1,'E',start=6,mag=4,angle=270)
    a.apply_load(-1,'C',start=30,mag=5,angle=270)
    a.solve()
    assert a.reaction_force[Symbol("R_A_x")] == 0
    assert abs(a.reaction_force[Symbol("R_A_y")] - 6.750000000000000) < 10e-12
    assert a.reaction_force[Symbol("R_B_x")] == 0
    assert abs(a.reaction_force[Symbol("R_B_y")] - 5.250000000000000) < 10e-12

def test_symbol_magnitude():
    a = Arch((0,0),(16,0),crown_x=8,crown_y=5)
    a.apply_load(0,'C',start=3,end=5,mag=t)
    a.solve()
    assert a.reaction_force[Symbol("R_A_x")] == -(4*t)/5
    assert a.reaction_force[Symbol("R_A_y")] == -(3*t)/2
    assert a.reaction_force[Symbol("R_B_x")] == (4*t)/5
    assert a.reaction_force[Symbol("R_B_y")] == -t/2
    assert a.bending_moment_at(4) == -5*t/2

def test_forces():
    a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
    a.apply_load(-1,'C',8,150,angle=270)
    a.apply_load(0,'D',start=20,end=40,mag=-4)
    a.solve()
    assert abs(a.axial_force_at(7.999999999999999)-149.430523405935) < 1e-12
    assert abs(a.shear_force_at(7.999999999999999)-64.9227473161196) < 1e-12

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_convert_array_to_indexed.py ===
from sympy import Sum, Dummy, sin
from sympy.tensor.array.expressions import ArraySymbol, ArrayTensorProduct, ArrayContraction, PermuteDims, \
    ArrayDiagonal, ArrayAdd, OneArray, ZeroArray, convert_indexed_to_array, ArrayElementwiseApplyFunc, Reshape
from sympy.tensor.array.expressions.from_array_to_indexed import convert_array_to_indexed

from sympy.abc import i, j, k, l, m, n, o


def test_convert_array_to_indexed_main():
    A = ArraySymbol("A", (3, 3, 3))
    B = ArraySymbol("B", (3, 3))
    C = ArraySymbol("C", (3, 3))

    d_ = Dummy("d_")

    assert convert_array_to_indexed(A, [i, j, k]) == A[i, j, k]

    expr = ArrayTensorProduct(A, B, C)
    conv = convert_array_to_indexed(expr, [i,j,k,l,m,n,o])
    assert conv == A[i,j,k]*B[l,m]*C[n,o]
    assert convert_indexed_to_array(conv, [i,j,k,l,m,n,o]) == expr

    expr = ArrayContraction(A, (0, 2))
    assert convert_array_to_indexed(expr, [i]).dummy_eq(Sum(A[d_, i, d_], (d_, 0, 2)))

    expr = ArrayDiagonal(A, (0, 2))
    assert convert_array_to_indexed(expr, [i, j]) == A[j, i, j]

    expr = PermuteDims(A, [1, 2, 0])
    conv = convert_array_to_indexed(expr, [i, j, k])
    assert conv == A[k, i, j]
    assert convert_indexed_to_array(conv, [i, j, k]) == expr

    expr = ArrayAdd(B, C, PermuteDims(C, [1, 0]))
    conv = convert_array_to_indexed(expr, [i, j])
    assert conv == B[i, j] + C[i, j] + C[j, i]
    assert convert_indexed_to_array(conv, [i, j]) == expr

    expr = ArrayElementwiseApplyFunc(sin, A)
    conv = convert_array_to_indexed(expr, [i, j, k])
    assert conv == sin(A[i, j, k])
    assert convert_indexed_to_array(conv, [i, j, k]).dummy_eq(expr)

    assert convert_array_to_indexed(OneArray(3, 3), [i, j]) == 1
    assert convert_array_to_indexed(ZeroArray(3, 3), [i, j]) == 0

    expr = Reshape(A, (27,))
    assert convert_array_to_indexed(expr, [i]) == A[i // 9, i // 3 % 3, i % 3]

    X = ArraySymbol("X", (2, 3, 4, 5, 6))
    expr = Reshape(X, (2*3*4*5*6,))
    assert convert_array_to_indexed(expr, [i]) == X[i // 360, i // 120 % 3, i // 30 % 4, i // 6 % 5, i % 6]

    expr = Reshape(X, (4, 9, 2, 2, 5))
    one_index = 180*i + 20*j + 10*k + 5*l + m
    expected = X[one_index // (3*4*5*6), one_index // (4*5*6) % 3, one_index // (5*6) % 4, one_index // 6 % 5, one_index % 6]
    assert convert_array_to_indexed(expr, [i, j, k, l, m]) == expected

    X = ArraySymbol("X", (2*3*5,))
    expr = Reshape(X, (2, 3, 5))
    assert convert_array_to_indexed(expr, [i, j, k]) == X[15*i + 5*j + k]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\index_tricks.py ===
from __future__ import annotations
from typing import Any
import numpy as np

AR_LIKE_b = [[True, True], [True, True]]
AR_LIKE_i = [[1, 2], [3, 4]]
AR_LIKE_f = [[1.0, 2.0], [3.0, 4.0]]
AR_LIKE_U = [["1", "2"], ["3", "4"]]

AR_i8: np.ndarray[Any, np.dtype[np.int64]] = np.array(AR_LIKE_i, dtype=np.int64)

np.ndenumerate(AR_i8)
np.ndenumerate(AR_LIKE_f)
np.ndenumerate(AR_LIKE_U)

next(np.ndenumerate(AR_i8))
next(np.ndenumerate(AR_LIKE_f))
next(np.ndenumerate(AR_LIKE_U))

iter(np.ndenumerate(AR_i8))
iter(np.ndenumerate(AR_LIKE_f))
iter(np.ndenumerate(AR_LIKE_U))

iter(np.ndindex(1, 2, 3))
next(np.ndindex(1, 2, 3))

np.unravel_index([22, 41, 37], (7, 6))
np.unravel_index([31, 41, 13], (7, 6), order='F')
np.unravel_index(1621, (6, 7, 8, 9))

np.ravel_multi_index(AR_LIKE_i, (7, 6))
np.ravel_multi_index(AR_LIKE_i, (7, 6), order='F')
np.ravel_multi_index(AR_LIKE_i, (4, 6), mode='clip')
np.ravel_multi_index(AR_LIKE_i, (4, 4), mode=('clip', 'wrap'))
np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9))

np.mgrid[1:1:2]
np.mgrid[1:1:2, None:10]

np.ogrid[1:1:2]
np.ogrid[1:1:2, None:10]

np.index_exp[0:1]
np.index_exp[0:1, None:3]
np.index_exp[0, 0:1, ..., [0, 1, 3]]

np.s_[0:1]
np.s_[0:1, None:3]
np.s_[0, 0:1, ..., [0, 1, 3]]

np.ix_(AR_LIKE_b[0])
np.ix_(AR_LIKE_i[0], AR_LIKE_f[0])
np.ix_(AR_i8[0])

np.fill_diagonal(AR_i8, 5)

np.diag_indices(4)
np.diag_indices(2, 3)

np.diag_indices_from(AR_i8)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_comparison.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.tests.arrays.masked_shared import ComparisonOps


@pytest.fixture
def data():
    """Fixture returning boolean array with valid and missing data"""
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.fixture
def dtype():
    """Fixture returning BooleanDtype"""
    return pd.BooleanDtype()


class TestComparisonOps(ComparisonOps):
    def test_compare_scalar(self, data, comparison_op):
        self._compare_other(data, comparison_op, True)

    def test_compare_array(self, data, comparison_op):
        other = pd.array([True] * len(data), dtype="boolean")
        self._compare_other(data, comparison_op, other)
        other = np.array([True] * len(data))
        self._compare_other(data, comparison_op, other)
        other = pd.Series([True] * len(data))
        self._compare_other(data, comparison_op, other)

    @pytest.mark.parametrize("other", [True, False, pd.NA])
    def test_scalar(self, other, comparison_op, dtype):
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_array(self, comparison_op):
        op = comparison_op
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")

        result = op(a, b)

        values = op(a._data, b._data)
        mask = a._mask | b._mask
        expected = BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = None
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\masked\test_indexing.py ===
import re

import numpy as np
import pytest

import pandas as pd


class TestSetitemValidation:
    def _check_setitem_invalid(self, arr, invalid):
        msg = f"Invalid value '{invalid!s}' for dtype '{arr.dtype}'"
        msg = re.escape(msg)
        with pytest.raises(TypeError, match=msg):
            arr[0] = invalid

        with pytest.raises(TypeError, match=msg):
            arr[:] = invalid

        with pytest.raises(TypeError, match=msg):
            arr[[0]] = invalid

        # FIXME: don't leave commented-out
        # with pytest.raises(TypeError):
        #    arr[[0]] = [invalid]

        # with pytest.raises(TypeError):
        #    arr[[0]] = np.array([invalid], dtype=object)

        # Series non-coercion, behavior subject to change
        ser = pd.Series(arr)
        with pytest.raises(TypeError, match=msg):
            ser[0] = invalid
            # TODO: so, so many other variants of this...

    _invalid_scalars = [
        1 + 2j,
        "True",
        "1",
        "1.0",
        pd.NaT,
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ]

    @pytest.mark.parametrize(
        "invalid", _invalid_scalars + [1, 1.0, np.int64(1), np.float64(1)]
    )
    def test_setitem_validation_scalar_bool(self, invalid):
        arr = pd.array([True, False, None], dtype="boolean")
        self._check_setitem_invalid(arr, invalid)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True, 1.5, np.float64(1.5)])
    def test_setitem_validation_scalar_int(self, invalid, any_int_ea_dtype):
        arr = pd.array([1, 2, None], dtype=any_int_ea_dtype)
        self._check_setitem_invalid(arr, invalid)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True])
    def test_setitem_validation_scalar_float(self, invalid, float_ea_dtype):
        arr = pd.array([1, 2, None], dtype=float_ea_dtype)
        self._check_setitem_invalid(arr, invalid)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\test_fillna.py ===
"""
Though Index.fillna and Series.fillna has separate impl,
test here to confirm these works as the same
"""

import numpy as np
import pytest

from pandas import MultiIndex
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops


def test_fillna(index_or_series_obj):
    # GH 11343
    obj = index_or_series_obj

    if isinstance(obj, MultiIndex):
        msg = "isna is not defined for MultiIndex"
        with pytest.raises(NotImplementedError, match=msg):
            obj.fillna(0)
        return

    # values will not be changed
    fill_value = obj.values[0] if len(obj) > 0 else 0
    result = obj.fillna(fill_value)

    tm.assert_equal(obj, result)

    # check shallow_copied
    assert obj is not result


@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_fillna_null(null_obj, index_or_series_obj):
    # GH 11343
    obj = index_or_series_obj
    klass = type(obj)

    if not allow_na_ops(obj):
        pytest.skip(f"{klass} doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(obj, MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    values = obj._values
    fill_value = values[0]
    expected = values.copy()
    values[0:2] = null_obj
    expected[0:2] = fill_value

    expected = klass(expected)
    obj = klass(values)

    result = obj.fillna(fill_value)
    tm.assert_equal(result, expected)

    # check shallow_copied
    assert obj is not result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_delitem.py ===
import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
)


class TestDataFrameDelItem:
    def test_delitem(self, float_frame):
        del float_frame["A"]
        assert "A" not in float_frame

    def test_delitem_multiindex(self):
        midx = MultiIndex.from_product([["A", "B"], [1, 2]])
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=midx)
        assert len(df.columns) == 4
        assert ("A",) in df.columns
        assert "A" in df.columns

        result = df["A"]
        assert isinstance(result, DataFrame)
        del df["A"]

        assert len(df.columns) == 2

        # A still in the levels, BUT get a KeyError if trying
        # to delete
        assert ("A",) not in df.columns
        with pytest.raises(KeyError, match=re.escape("('A',)")):
            del df[("A",)]

        # behavior of dropped/deleted MultiIndex levels changed from
        # GH 2770 to GH 19027: MultiIndex no longer '.__contains__'
        # levels which are dropped/deleted
        assert "A" not in df.columns
        with pytest.raises(KeyError, match=re.escape("('A',)")):
            del df["A"]

    def test_delitem_corner(self, float_frame):
        f = float_frame.copy()
        del f["D"]
        assert len(f.columns) == 3
        with pytest.raises(KeyError, match=r"^'D'$"):
            del f["D"]
        del f["B"]
        assert len(f.columns) == 2

    def test_delitem_col_still_multiindex(self):
        arrays = [["a", "b", "c", "top"], ["", "", "", "OD"], ["", "", "", "wx"]]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)

        df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), columns=index)
        del df[("a", "", "")]
        assert isinstance(df.columns, MultiIndex)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_common.py ===
import pytest

from pandas import DataFrame
from pandas.tests.plotting.common import (
    _check_plot_works,
    _check_ticks_props,
    _gen_two_subplots,
)

plt = pytest.importorskip("matplotlib.pyplot")


class TestCommon:
    def test__check_ticks_props(self):
        # GH 34768
        df = DataFrame({"b": [0, 1, 0], "a": [1, 2, 3]})
        ax = _check_plot_works(df.plot, rot=30)
        ax.yaxis.set_tick_params(rotation=30)
        msg = "expected 0.00000 but got "
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, xrot=0)
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, xlabelsize=0)
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, yrot=0)
        with pytest.raises(AssertionError, match=msg):
            _check_ticks_props(ax, ylabelsize=0)

    def test__gen_two_subplots_with_ax(self):
        fig = plt.gcf()
        gen = _gen_two_subplots(f=lambda **kwargs: None, fig=fig, ax="test")
        # On the first yield, no subplot should be added since ax was passed
        next(gen)
        assert fig.get_axes() == []
        # On the second, the one axis should match fig.subplot(2, 1, 2)
        next(gen)
        axes = fig.get_axes()
        assert len(axes) == 1
        subplot_geometry = list(axes[0].get_subplotspec().get_geometry()[:-1])
        subplot_geometry[-1] += 1
        assert subplot_geometry == [2, 1, 2]

    def test_colorbar_layout(self):
        fig = plt.figure()

        axes = fig.subplot_mosaic(
            """
            AB
            CC
            """
        )

        x = [1, 2, 3]
        y = [1, 2, 3]

        cs0 = axes["A"].scatter(x, y)
        axes["B"].scatter(x, y)

        fig.colorbar(cs0, ax=[axes["A"], axes["B"]], location="right")
        DataFrame(x).plot(ax=axes["C"])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_offsets_properties.py ===
"""
Behavioral based tests for offsets and date_range.

This file is adapted from https://github.com/pandas-dev/pandas/pull/18761 -
which was more ambitious but less idiomatic in its use of Hypothesis.

You may wish to consult the previous version for inspiration on further
tests, or when trying to pin down the bugs exposed by the tests below.
"""
from hypothesis import (
    assume,
    given,
)
import pytest
import pytz

import pandas as pd
from pandas._testing._hypothesis import (
    DATETIME_JAN_1_1900_OPTIONAL_TZ,
    YQM_OFFSET,
)

# ----------------------------------------------------------------
# Offset-specific behaviour tests


@pytest.mark.arm_slow
@given(DATETIME_JAN_1_1900_OPTIONAL_TZ, YQM_OFFSET)
def test_on_offset_implementations(dt, offset):
    assume(not offset.normalize)
    # check that the class-specific implementations of is_on_offset match
    # the general case definition:
    #   (dt + offset) - offset == dt
    try:
        compare = (dt + offset) - offset
    except (pytz.NonExistentTimeError, pytz.AmbiguousTimeError):
        # When dt + offset does not exist or is DST-ambiguous, assume(False) to
        # indicate to hypothesis that this is not a valid test case
        # DST-ambiguous example (GH41906):
        # dt = datetime.datetime(1900, 1, 1, tzinfo=pytz.timezone('Africa/Kinshasa'))
        # offset = MonthBegin(66)
        assume(False)

    assert offset.is_on_offset(dt) == (compare == dt)


@given(YQM_OFFSET)
def test_shift_across_dst(offset):
    # GH#18319 check that 1) timezone is correctly normalized and
    # 2) that hour is not incorrectly changed by this normalization
    assume(not offset.normalize)

    # Note that dti includes a transition across DST boundary
    dti = pd.date_range(
        start="2017-10-30 12:00:00", end="2017-11-06", freq="D", tz="US/Eastern"
    )
    assert (dti.hour == 12).all()  # we haven't screwed up yet

    res = dti + offset
    assert (res.hour == 12).all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_error_prop.py ===
from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.stats.error_prop import variance_prop
from sympy.stats.symbolic_probability import (RandomSymbol, Variance,
        Covariance)


def test_variance_prop():
    x, y, z = symbols('x y z')
    phi, t = consts = symbols('phi t')
    a = RandomSymbol(x)
    var_x = Variance(a)
    var_y = Variance(RandomSymbol(y))
    var_z = Variance(RandomSymbol(z))
    f = Function('f')(x)
    cases = {
        x + y: var_x + var_y,
        a + y: var_x + var_y,
        x + y + z: var_x + var_y + var_z,
        2*x: 4*var_x,
        x*y: var_x*y**2 + var_y*x**2,
        1/x: var_x/x**4,
        x/y: (var_x*y**2 + var_y*x**2)/y**4,
        exp(x): var_x*exp(2*x),
        exp(2*x): 4*var_x*exp(4*x),
        exp(-x*t): t**2*var_x*exp(-2*t*x),
        f: Variance(f),
        }
    for inp, out in cases.items():
        obs = variance_prop(inp, consts=consts)
        assert out == obs

def test_variance_prop_with_covar():
    x, y, z = symbols('x y z')
    phi, t = consts = symbols('phi t')
    a = RandomSymbol(x)
    var_x = Variance(a)
    b = RandomSymbol(y)
    var_y = Variance(b)
    c = RandomSymbol(z)
    var_z = Variance(c)
    covar_x_y = Covariance(a, b)
    covar_x_z = Covariance(a, c)
    covar_y_z = Covariance(b, c)
    cases = {
        x + y: var_x + var_y + 2*covar_x_y,
        a + y: var_x + var_y + 2*covar_x_y,
        x + y + z: var_x + var_y + var_z + \
                   2*covar_x_y + 2*covar_x_z + 2*covar_y_z,
        2*x: 4*var_x,
        x*y: var_x*y**2 + var_y*x**2 + 2*covar_x_y/(x*y),
        1/x: var_x/x**4,
        exp(x): var_x*exp(2*x),
        exp(2*x): 4*var_x*exp(4*x),
        exp(-x*t): t**2*var_x*exp(-2*t*x),
        }
    for inp, out in cases.items():
        obs = variance_prop(inp, consts=consts, include_covar=True)
        assert out == obs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_generator.py ===

from greenlet import greenlet

from . import TestCase

class genlet(greenlet):
    parent = None
    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds

    def run(self):
        fn, = self.fn
        fn(*self.args, **self.kwds)

    def __iter__(self):
        return self

    def __next__(self):
        self.parent = greenlet.getcurrent()
        result = self.switch()
        if self:
            return result

        raise StopIteration

    next = __next__


def Yield(value):
    g = greenlet.getcurrent()
    while not isinstance(g, genlet):
        if g is None:
            raise RuntimeError('yield outside a genlet')
        g = g.parent
    g.parent.switch(value)


def generator(func):
    class Generator(genlet):
        fn = (func,)
    return Generator

# ____________________________________________________________


class GeneratorTests(TestCase):
    def test_generator(self):
        seen = []

        def g(n):
            for i in range(n):
                seen.append(i)
                Yield(i)
        g = generator(g)
        for _ in range(3):
            for j in g(5):
                seen.append(j)
        self.assertEqual(seen, 3 * [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_matlib.py ===
import numpy as np
import numpy.matlib
from numpy.testing import assert_, assert_array_equal


def test_empty():
    x = numpy.matlib.empty((2,))
    assert_(isinstance(x, np.matrix))
    assert_(x.shape, (1, 2))

def test_ones():
    assert_array_equal(numpy.matlib.ones((2, 3)),
                       np.matrix([[ 1.,  1.,  1.],
                                  [ 1.,  1.,  1.]]))

    assert_array_equal(numpy.matlib.ones(2), np.matrix([[ 1.,  1.]]))

def test_zeros():
    assert_array_equal(numpy.matlib.zeros((2, 3)),
                       np.matrix([[ 0.,  0.,  0.],
                                  [ 0.,  0.,  0.]]))

    assert_array_equal(numpy.matlib.zeros(2), np.matrix([[0.,  0.]]))

def test_identity():
    x = numpy.matlib.identity(2, dtype=int)
    assert_array_equal(x, np.matrix([[1, 0], [0, 1]]))

def test_eye():
    xc = numpy.matlib.eye(3, k=1, dtype=int)
    assert_array_equal(xc, np.matrix([[ 0,  1,  0],
                                      [ 0,  0,  1],
                                      [ 0,  0,  0]]))
    assert xc.flags.c_contiguous
    assert not xc.flags.f_contiguous

    xf = numpy.matlib.eye(3, 4, dtype=int, order='F')
    assert_array_equal(xf, np.matrix([[ 1,  0,  0,  0],
                                      [ 0,  1,  0,  0],
                                      [ 0,  0,  1,  0]]))
    assert not xf.flags.c_contiguous
    assert xf.flags.f_contiguous

def test_rand():
    x = numpy.matlib.rand(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

def test_randn():
    x = np.matlib.randn(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

def test_repmat():
    a1 = np.arange(4)
    x = numpy.matlib.repmat(a1, 2, 2)
    y = np.array([[0, 1, 2, 3, 0, 1, 2, 3],
                  [0, 1, 2, 3, 0, 1, 2, 3]])
    assert_array_equal(x, y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_astype.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_astype(using_infer_string):
    # with missing values
    arr = pd.array([True, False, None], dtype="boolean")

    with pytest.raises(ValueError, match="cannot convert NA to integer"):
        arr.astype("int64")

    with pytest.raises(ValueError, match="cannot convert float NaN to"):
        arr.astype("bool")

    result = arr.astype("float64")
    expected = np.array([1, 0, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("str")
    if using_infer_string:
        expected = pd.array(
            ["True", "False", None], dtype=pd.StringDtype(na_value=np.nan)
        )
        tm.assert_extension_array_equal(result, expected)
    else:
        expected = np.array(["True", "False", "<NA>"], dtype=f"{tm.ENDIAN}U5")
        tm.assert_numpy_array_equal(result, expected)

    # no missing values
    arr = pd.array([True, False, True], dtype="boolean")
    result = arr.astype("int64")
    expected = np.array([1, 0, 1], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("bool")
    expected = np.array([True, False, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_to_boolean_array():
    # astype to BooleanArray
    arr = pd.array([True, False, None], dtype="boolean")

    result = arr.astype("boolean")
    tm.assert_extension_array_equal(result, arr)
    result = arr.astype(pd.BooleanDtype())
    tm.assert_extension_array_equal(result, arr)


def test_astype_to_integer_array():
    # astype to IntegerArray
    arr = pd.array([True, False, None], dtype="boolean")

    result = arr.astype("Int64")
    expected = pd.array([1, 0, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_item.py ===
"""
Series.item method, mainly testing that we get python scalars as opposed to
numpy scalars.
"""
import pytest

from pandas import (
    Series,
    Timedelta,
    Timestamp,
    date_range,
)


class TestItem:
    def test_item(self):
        # We are testing that we get python scalars as opposed to numpy scalars
        ser = Series([1])
        result = ser.item()
        assert result == 1
        assert result == ser.iloc[0]
        assert isinstance(result, int)  # i.e. not np.int64

        ser = Series([0.5], index=[3])
        result = ser.item()
        assert isinstance(result, float)
        assert result == 0.5

        ser = Series([1, 2])
        msg = "can only convert an array of size 1"
        with pytest.raises(ValueError, match=msg):
            ser.item()

        dti = date_range("2016-01-01", periods=2)
        with pytest.raises(ValueError, match=msg):
            dti.item()
        with pytest.raises(ValueError, match=msg):
            Series(dti).item()

        val = dti[:1].item()
        assert isinstance(val, Timestamp)
        val = Series(dti)[:1].item()
        assert isinstance(val, Timestamp)

        tdi = dti - dti
        with pytest.raises(ValueError, match=msg):
            tdi.item()
        with pytest.raises(ValueError, match=msg):
            Series(tdi).item()

        val = tdi[:1].item()
        assert isinstance(val, Timedelta)
        val = Series(tdi)[:1].item()
        assert isinstance(val, Timedelta)

        # Case where ser[0] would not work
        ser = Series(dti, index=[5, 6])
        val = ser.iloc[:1].item()
        assert val == dti[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\discrete\tests\test_recurrences.py ===
from sympy.core.numbers import Rational
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.core import S, symbols
from sympy.testing.pytest import raises
from sympy.discrete.recurrences import linrec

def test_linrec():
    assert linrec(coeffs=[1, 1], init=[1, 1], n=20) == 10946
    assert linrec(coeffs=[1, 2, 3, 4, 5], init=[1, 1, 0, 2], n=10) == 1040
    assert linrec(coeffs=[0, 0, 11, 13], init=[23, 27], n=25) == 59628567384
    assert linrec(coeffs=[0, 0, 1, 1, 2], init=[1, 5, 3], n=15) == 165
    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=70) == \
        56889923441670659718376223533331214868804815612050381493741233489928913241
    assert linrec(coeffs=[0]*55 + [1, 1, 2, 3], init=[0]*50 + [1, 2, 3], n=4000) == \
        702633573874937994980598979769135096432444135301118916539

    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=10**4)
    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=10**5)

    assert all(linrec(coeffs=[1, 1], init=[0, 1], n=n) == fibonacci(n)
                                                    for n in range(95, 115))

    assert all(linrec(coeffs=[1, 1], init=[1, 1], n=n) == fibonacci(n + 1)
                                                    for n in range(595, 615))

    a = [S.Half, Rational(3, 4), Rational(5, 6), 7, Rational(8, 9), Rational(3, 5)]
    b = [1, 2, 8, Rational(5, 7), Rational(3, 7), Rational(2, 9), 6]
    x, y, z = symbols('x y z')

    assert linrec(coeffs=a[:5], init=b[:4], n=80) == \
        Rational(1726244235456268979436592226626304376013002142588105090705187189,
            1960143456748895967474334873705475211264)

    assert linrec(coeffs=a[:4], init=b[:4], n=50) == \
        Rational(368949940033050147080268092104304441, 504857282956046106624)

    assert linrec(coeffs=a[3:], init=b[:3], n=35) == \
        Rational(97409272177295731943657945116791049305244422833125109,
            814315512679031689453125)

    assert linrec(coeffs=[0]*60 + [Rational(2, 3), Rational(4, 5)], init=b, n=3000) == \
        Rational(26777668739896791448594650497024, 48084516708184142230517578125)

    raises(TypeError, lambda: linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4, 5], n=1))
    raises(TypeError, lambda: linrec(coeffs=a[:4], init=b[:5], n=10000))
    raises(ValueError, lambda: linrec(coeffs=a[:4], init=b[:4], n=-10000))
    raises(TypeError, lambda: linrec(x, b, n=10000))
    raises(TypeError, lambda: linrec(a, y, n=10000))

    assert linrec(coeffs=[x, y, z], init=[1, 1, 1], n=4) == \
        x**2  + x*y + x*z + y + z
    assert linrec(coeffs=[1, 2, 1], init=[x, y, z], n=20) == \
        269542*x + 664575*y + 578949*z
    assert linrec(coeffs=[0, 3, 1, 2], init=[x, y], n=30) == \
        58516436*x + 56372788*y
    assert linrec(coeffs=[0]*50 + [1, 2, 3], init=[x, y, z], n=1000) == \
        11477135884896*x + 25999077948732*y + 41975630244216*z
    assert linrec(coeffs=[], init=[1, 1], n=20) == 0
    assert linrec(coeffs=[x, y, z], init=[1, 2, 3], n=2) == 3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_decompogen.py ===
from sympy.solvers.decompogen import decompogen, compogen
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import XFAIL, raises

x, y = symbols('x y')


def test_decompogen():
    assert decompogen(sin(cos(x)), x) == [sin(x), cos(x)]
    assert decompogen(sin(x)**2 + sin(x) + 1, x) == [x**2 + x + 1, sin(x)]
    assert decompogen(sqrt(6*x**2 - 5), x) == [sqrt(x), 6*x**2 - 5]
    assert decompogen(sin(sqrt(cos(x**2 + 1))), x) == [sin(x), sqrt(x), cos(x), x**2 + 1]
    assert decompogen(Abs(cos(x)**2 + 3*cos(x) - 4), x) == [Abs(x), x**2 + 3*x - 4, cos(x)]
    assert decompogen(sin(x)**2 + sin(x) - sqrt(3)/2, x) == [x**2 + x - sqrt(3)/2, sin(x)]
    assert decompogen(Abs(cos(y)**2 + 3*cos(x) - 4), x) == [Abs(x), 3*x + cos(y)**2 - 4, cos(x)]
    assert decompogen(x, y) == [x]
    assert decompogen(1, x) == [1]
    assert decompogen(Max(3, x), x) == [Max(3, x)]
    raises(TypeError, lambda: decompogen(x < 5, x))
    u = 2*x + 3
    assert decompogen(Max(sqrt(u),(u)**2), x) == [Max(sqrt(x), x**2), u]
    assert decompogen(Max(u, u**2, y), x) == [Max(x, x**2, y), u]
    assert decompogen(Max(sin(x), u), x) == [Max(2*x + 3, sin(x))]


def test_decompogen_poly():
    assert decompogen(x**4 + 2*x**2 + 1, x) == [x**2 + 2*x + 1, x**2]
    assert decompogen(x**4 + 2*x**3 - x - 1, x) == [x**2 - x - 1, x**2 + x]


@XFAIL
def test_decompogen_fails():
    A = lambda x: x**2 + 2*x + 3
    B = lambda x: 4*x**2 + 5*x + 6
    assert decompogen(A(x*exp(x)), x) == [x**2 + 2*x + 3, x*exp(x)]
    assert decompogen(A(B(x)), x) == [x**2 + 2*x + 3, 4*x**2 + 5*x + 6]
    assert decompogen(A(1/x + 1/x**2), x) == [x**2 + 2*x + 3, 1/x + 1/x**2]
    assert decompogen(A(1/x + 2/(x + 1)), x) == [x**2 + 2*x + 3, 1/x + 2/(x + 1)]


def test_compogen():
    assert compogen([sin(x), cos(x)], x) == sin(cos(x))
    assert compogen([x**2 + x + 1, sin(x)], x) == sin(x)**2 + sin(x) + 1
    assert compogen([sqrt(x), 6*x**2 - 5], x) == sqrt(6*x**2 - 5)
    assert compogen([sin(x), sqrt(x), cos(x), x**2 + 1], x) == sin(sqrt(
                                                                cos(x**2 + 1)))
    assert compogen([Abs(x), x**2 + 3*x - 4, cos(x)], x) == Abs(cos(x)**2 +
                                                                3*cos(x) - 4)
    assert compogen([x**2 + x - sqrt(3)/2, sin(x)], x) == (sin(x)**2 + sin(x) -
                                                           sqrt(3)/2)
    assert compogen([Abs(x), 3*x + cos(y)**2 - 4, cos(x)], x) == \
        Abs(3*cos(x) + cos(y)**2 - 4)
    assert compogen([x**2 + 2*x + 1, x**2], x) == x**4 + 2*x**2 + 1
    # the result is in unsimplified form
    assert compogen([x**2 - x - 1, x**2 + x], x) == -x**2 - x + (x**2 + x)**2 - 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_is_homogeneous_dtype.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    Categorical,
    DataFrame,
)

# _is_homogeneous_type always returns True for ArrayManager
pytestmark = td.skip_array_manager_invalid_test


@pytest.mark.parametrize(
    "data, expected",
    [
        # empty
        (DataFrame(), True),
        # multi-same
        (DataFrame({"A": [1, 2], "B": [1, 2]}), True),
        # multi-object
        (
            DataFrame(
                {
                    "A": np.array([1, 2], dtype=object),
                    "B": np.array(["a", "b"], dtype=object),
                },
                dtype="object",
            ),
            True,
        ),
        # multi-extension
        (
            DataFrame({"A": Categorical(["a", "b"]), "B": Categorical(["a", "b"])}),
            True,
        ),
        # differ types
        (DataFrame({"A": [1, 2], "B": [1.0, 2.0]}), False),
        # differ sizes
        (
            DataFrame(
                {
                    "A": np.array([1, 2], dtype=np.int32),
                    "B": np.array([1, 2], dtype=np.int64),
                }
            ),
            False,
        ),
        # multi-extension differ
        (
            DataFrame({"A": Categorical(["a", "b"]), "B": Categorical(["b", "c"])}),
            False,
        ),
    ],
)
def test_is_homogeneous_type(data, expected):
    assert data._is_homogeneous_type is expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_join.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    DataFrame,
    Index,
    PeriodIndex,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestJoin:
    def test_join_outer_indexer(self):
        pi = period_range("1/1/2000", "1/20/2000", freq="D")

        result = pi._outer_indexer(pi)
        tm.assert_extension_array_equal(result[0], pi._values)
        tm.assert_numpy_array_equal(result[1], np.arange(len(pi), dtype=np.intp))
        tm.assert_numpy_array_equal(result[2], np.arange(len(pi), dtype=np.intp))

    def test_joins(self, join_type):
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        joined = index.join(index[:-5], how=join_type)

        assert isinstance(joined, PeriodIndex)
        assert joined.freq == index.freq

    def test_join_self(self, join_type):
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        res = index.join(index, how=join_type)
        assert index is res

    def test_join_does_not_recur(self):
        df = DataFrame(
            np.ones((3, 2)),
            index=date_range("2020-01-01", periods=3),
            columns=period_range("2020-01-01", periods=2),
        )
        ser = df.iloc[:2, 0]

        res = ser.index.join(df.columns, how="outer")
        expected = Index(
            [ser.index[0], ser.index[1], df.columns[0], df.columns[1]], object
        )
        tm.assert_index_equal(res, expected)

    def test_join_mismatched_freq_raises(self):
        index = period_range("1/1/2000", "1/20/2000", freq="D")
        index3 = period_range("1/1/2000", "1/20/2000", freq="2D")
        msg = r".*Input has different freq=2D from Period\(freq=D\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            index.join(index3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\holiday\test_federal.py ===
from datetime import datetime

from pandas import DatetimeIndex
import pandas._testing as tm

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    USFederalHolidayCalendar,
    USMartinLutherKingJr,
    USMemorialDay,
)


def test_no_mlk_before_1986():
    # see gh-10278
    class MLKCalendar(AbstractHolidayCalendar):
        rules = [USMartinLutherKingJr]

    holidays = MLKCalendar().holidays(start="1984", end="1988").to_pydatetime().tolist()

    # Testing to make sure holiday is not incorrectly observed before 1986.
    assert holidays == [datetime(1986, 1, 20, 0, 0), datetime(1987, 1, 19, 0, 0)]


def test_memorial_day():
    class MemorialDay(AbstractHolidayCalendar):
        rules = [USMemorialDay]

    holidays = MemorialDay().holidays(start="1971", end="1980").to_pydatetime().tolist()

    # Fixes 5/31 error and checked manually against Wikipedia.
    assert holidays == [
        datetime(1971, 5, 31, 0, 0),
        datetime(1972, 5, 29, 0, 0),
        datetime(1973, 5, 28, 0, 0),
        datetime(1974, 5, 27, 0, 0),
        datetime(1975, 5, 26, 0, 0),
        datetime(1976, 5, 31, 0, 0),
        datetime(1977, 5, 30, 0, 0),
        datetime(1978, 5, 29, 0, 0),
        datetime(1979, 5, 28, 0, 0),
    ]


def test_federal_holiday_inconsistent_returntype():
    # GH 49075 test case
    # Instantiate two calendars to rule out _cache
    cal1 = USFederalHolidayCalendar()
    cal2 = USFederalHolidayCalendar()

    results_2018 = cal1.holidays(start=datetime(2018, 8, 1), end=datetime(2018, 8, 31))
    results_2019 = cal2.holidays(start=datetime(2019, 8, 1), end=datetime(2019, 8, 31))
    expected_results = DatetimeIndex([], dtype="datetime64[ns]", freq=None)

    # Check against expected results to ensure both date
    # ranges generate expected results as per GH49075 submission
    tm.assert_index_equal(results_2018, expected_results)
    tm.assert_index_equal(results_2019, expected_results)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_util.py ===
import os

import pytest

from pandas import (
    array,
    compat,
)
import pandas._testing as tm


def test_numpy_err_state_is_default():
    expected = {"over": "warn", "divide": "warn", "invalid": "warn", "under": "ignore"}
    import numpy as np

    # The error state should be unchanged after that import.
    assert np.geterr() == expected


def test_convert_rows_list_to_csv_str():
    rows_list = ["aaa", "bbb", "ccc"]
    ret = tm.convert_rows_list_to_csv_str(rows_list)

    if compat.is_platform_windows():
        expected = "aaa\r\nbbb\r\nccc\r\n"
    else:
        expected = "aaa\nbbb\nccc\n"

    assert ret == expected


@pytest.mark.parametrize("strict_data_files", [True, False])
def test_datapath_missing(datapath):
    with pytest.raises(ValueError, match="Could not find file"):
        datapath("not_a_file")


def test_datapath(datapath):
    args = ("io", "data", "csv", "iris.csv")

    result = datapath(*args)
    expected = os.path.join(os.path.dirname(os.path.dirname(__file__)), *args)

    assert result == expected


def test_external_error_raised():
    with tm.external_error_raised(TypeError):
        raise TypeError("Should not check this error message, so it will pass")


def test_is_sorted():
    arr = array([1, 2, 3], dtype="Int64")
    tm.assert_is_sorted(arr)

    arr = array([4, 2, 3], dtype="Int64")
    with pytest.raises(AssertionError, match="ExtensionArray are different"):
        tm.assert_is_sorted(arr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_applications.py ===
# This file contains tests that exercise multiple AST nodes

import tempfile

from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, skip_under_pyodide
from sympy.codegen.ast import (
    FunctionDefinition, FunctionPrototype, Variable, Pointer, real, Assignment,
    integer, CodeBlock, While
)
from sympy.codegen.cnodes import void, PreIncrement
from sympy.codegen.cutils import render_as_source_file

cython = import_module('cython')
np = import_module('numpy')

def _mk_func1():
    declars = n, inp, out = Variable('n', integer), Pointer('inp', real), Pointer('out', real)
    i = Variable('i', integer)
    whl = While(i<n, [Assignment(out[i], inp[i]), PreIncrement(i)])
    body = CodeBlock(i.as_Declaration(value=0), whl)
    return FunctionDefinition(void, 'our_test_function', declars, body)


def _render_compile_import(funcdef, build_dir):
    code_str = render_as_source_file(funcdef, settings={"contract": False})
    declar = ccode(FunctionPrototype.from_FunctionDefinition(funcdef))
    return compile_link_import_strings([
        ('our_test_func.c', code_str),
        ('_our_test_func.pyx', ("#cython: language_level={}\n".format("3") +
                                "cdef extern {declar}\n"
                                "def _{fname}({typ}[:] inp, {typ}[:] out):\n"
                                "    {fname}(inp.size, &inp[0], &out[0])").format(
                                    declar=declar, fname=funcdef.name, typ='double'
                                ))
    ], build_dir=build_dir)


@may_xfail
@skip_under_pyodide("Emscripten does not support process spawning")
def test_copying_function():
    if not np:
        skip("numpy not installed.")
    if not has_c():
        skip("No C compiler found.")
    if not cython:
        skip("Cython not found.")

    info = None
    with tempfile.TemporaryDirectory() as folder:
        mod, info = _render_compile_import(_mk_func1(), build_dir=folder)
        inp = np.arange(10.0)
        out = np.empty_like(inp)
        mod._our_test_function(inp, out)
        assert np.allclose(inp, out)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_matadd.py ===
from sympy.matrices.expressions import MatrixSymbol, MatAdd, MatPow, MatMul
from sympy.matrices.expressions.special import GenericZeroMatrix, ZeroMatrix
from sympy.matrices.exceptions import ShapeError
from sympy.matrices import eye, ImmutableMatrix
from sympy.core import Add, Basic, S
from sympy.core.add import add
from sympy.testing.pytest import XFAIL, raises

X = MatrixSymbol('X', 2, 2)
Y = MatrixSymbol('Y', 2, 2)

def test_evaluate():
    assert MatAdd(X, X, evaluate=True) == add(X, X, evaluate=True) == MatAdd(X, X).doit()

def test_sort_key():
    assert MatAdd(Y, X).doit().args == add(Y, X).doit().args == (X, Y)


def test_matadd_sympify():
    assert isinstance(MatAdd(eye(1), eye(1)).args[0], Basic)
    assert isinstance(add(eye(1), eye(1)).args[0], Basic)


def test_matadd_of_matrices():
    assert MatAdd(eye(2), 4*eye(2), eye(2)).doit() == ImmutableMatrix(6*eye(2))
    assert add(eye(2), 4*eye(2), eye(2)).doit() == ImmutableMatrix(6*eye(2))


def test_doit_args():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix([[2, 3], [4, 5]])
    assert MatAdd(A, MatPow(B, 2)).doit() == A + B**2
    assert MatAdd(A, MatMul(A, B)).doit() == A + A*B
    assert (MatAdd(A, X, MatMul(A, B), Y, MatAdd(2*A, B)).doit() ==
    add(A, X, MatMul(A, B), Y, add(2*A, B)).doit() ==
    MatAdd(3*A + A*B + B, X, Y))


def test_generic_identity():
    assert MatAdd.identity == GenericZeroMatrix()
    assert MatAdd.identity != S.Zero


def test_zero_matrix_add():
    assert Add(ZeroMatrix(2, 2), ZeroMatrix(2, 2)) == ZeroMatrix(2, 2)

@XFAIL
def test_matrix_Add_with_scalar():
    raises(TypeError, lambda: Add(0, ZeroMatrix(2, 2)))


def test_shape_error():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 3)
    raises(ShapeError, lambda: MatAdd(A, B))

    A = MatrixSymbol('A', 3, 2)
    raises(ShapeError, lambda: MatAdd(A, B))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\tests\isatty_test.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
import sys
from unittest import TestCase, main

from ..ansitowin32 import StreamWrapper, AnsiToWin32
from .utils import pycharm, replace_by, replace_original_by, StreamTTY, StreamNonTTY


def is_a_tty(stream):
    return StreamWrapper(stream, None).isatty()

class IsattyTest(TestCase):

    def test_TTY(self):
        tty = StreamTTY()
        self.assertTrue(is_a_tty(tty))
        with pycharm():
            self.assertTrue(is_a_tty(tty))

    def test_nonTTY(self):
        non_tty = StreamNonTTY()
        self.assertFalse(is_a_tty(non_tty))
        with pycharm():
            self.assertFalse(is_a_tty(non_tty))

    def test_withPycharm(self):
        with pycharm():
            self.assertTrue(is_a_tty(sys.stderr))
            self.assertTrue(is_a_tty(sys.stdout))

    def test_withPycharmTTYOverride(self):
        tty = StreamTTY()
        with pycharm(), replace_by(tty):
            self.assertTrue(is_a_tty(tty))

    def test_withPycharmNonTTYOverride(self):
        non_tty = StreamNonTTY()
        with pycharm(), replace_by(non_tty):
            self.assertFalse(is_a_tty(non_tty))

    def test_withPycharmNoneOverride(self):
        with pycharm():
            with replace_by(None), replace_original_by(None):
                self.assertFalse(is_a_tty(None))
                self.assertFalse(is_a_tty(StreamNonTTY()))
                self.assertTrue(is_a_tty(StreamTTY()))

    def test_withPycharmStreamWrapped(self):
        with pycharm():
            self.assertTrue(AnsiToWin32(StreamTTY()).stream.isatty())
            self.assertFalse(AnsiToWin32(StreamNonTTY()).stream.isatty())
            self.assertTrue(AnsiToWin32(sys.stdout).stream.isatty())
            self.assertTrue(AnsiToWin32(sys.stderr).stream.isatty())


if __name__ == '__main__':
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\dtype.py ===
import numpy as np

dtype_obj = np.dtype(np.str_)
void_dtype_obj = np.dtype([("f0", np.float64), ("f1", np.float32)])

np.dtype(dtype=np.int64)
np.dtype(int)
np.dtype("int")
np.dtype(None)

np.dtype((int, 2))
np.dtype((int, (1,)))

np.dtype({"names": ["a", "b"], "formats": [int, float]})
np.dtype({"names": ["a"], "formats": [int], "titles": [object]})
np.dtype({"names": ["a"], "formats": [int], "titles": [object()]})

np.dtype([("name", np.str_, 16), ("grades", np.float64, (2,)), ("age", "int32")])

np.dtype(
    {
        "names": ["a", "b"],
        "formats": [int, float],
        "itemsize": 9,
        "aligned": False,
        "titles": ["x", "y"],
        "offsets": [0, 1],
    }
)

np.dtype((np.float64, float))


class Test:
    dtype = np.dtype(float)


np.dtype(Test())

# Methods and attributes
dtype_obj.base
dtype_obj.subdtype
dtype_obj.newbyteorder()
dtype_obj.type
dtype_obj.name
dtype_obj.names

dtype_obj * 0
dtype_obj * 2

0 * dtype_obj
2 * dtype_obj

void_dtype_obj["f0"]
void_dtype_obj[0]
void_dtype_obj[["f0", "f1"]]
void_dtype_obj[["f0"]]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_head_tail.py ===
import numpy as np

from pandas import DataFrame
import pandas._testing as tm


def test_head_tail_generic(index, frame_or_series):
    # GH#5370

    ndim = 2 if frame_or_series is DataFrame else 1
    shape = (len(index),) * ndim
    vals = np.random.default_rng(2).standard_normal(shape)
    obj = frame_or_series(vals, index=index)

    tm.assert_equal(obj.head(), obj.iloc[:5])
    tm.assert_equal(obj.tail(), obj.iloc[-5:])

    # 0-len
    tm.assert_equal(obj.head(0), obj.iloc[0:0])
    tm.assert_equal(obj.tail(0), obj.iloc[0:0])

    # bounded
    tm.assert_equal(obj.head(len(obj) + 1), obj)
    tm.assert_equal(obj.tail(len(obj) + 1), obj)

    # neg index
    tm.assert_equal(obj.head(-3), obj.head(len(index) - 3))
    tm.assert_equal(obj.tail(-3), obj.tail(len(index) - 3))


def test_head_tail(float_frame):
    tm.assert_frame_equal(float_frame.head(), float_frame[:5])
    tm.assert_frame_equal(float_frame.tail(), float_frame[-5:])

    tm.assert_frame_equal(float_frame.head(0), float_frame[0:0])
    tm.assert_frame_equal(float_frame.tail(0), float_frame[0:0])

    tm.assert_frame_equal(float_frame.head(-1), float_frame[:-1])
    tm.assert_frame_equal(float_frame.tail(-1), float_frame[1:])
    tm.assert_frame_equal(float_frame.head(1), float_frame[:1])
    tm.assert_frame_equal(float_frame.tail(1), float_frame[-1:])
    # with a float index
    df = float_frame.copy()
    df.index = np.arange(len(float_frame)) + 0.1
    tm.assert_frame_equal(df.head(), df.iloc[:5])
    tm.assert_frame_equal(df.tail(), df.iloc[-5:])
    tm.assert_frame_equal(df.head(0), df[0:0])
    tm.assert_frame_equal(df.tail(0), df[0:0])
    tm.assert_frame_equal(df.head(-1), df.iloc[:-1])
    tm.assert_frame_equal(df.tail(-1), df.iloc[1:])


def test_head_tail_empty():
    # test empty dataframe
    empty_df = DataFrame()
    tm.assert_frame_equal(empty_df.tail(), empty_df)
    tm.assert_frame_equal(empty_df.head(), empty_df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_index.py ===
"""
Tests for offset behavior with indices.
"""
import pytest

from pandas import (
    Series,
    date_range,
)

from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)


@pytest.mark.parametrize("n", [-2, 1])
@pytest.mark.parametrize(
    "cls",
    [
        MonthBegin,
        MonthEnd,
        BMonthBegin,
        BMonthEnd,
        QuarterBegin,
        QuarterEnd,
        BQuarterBegin,
        BQuarterEnd,
        YearBegin,
        YearEnd,
        BYearBegin,
        BYearEnd,
    ],
)
def test_apply_index(cls, n):
    offset = cls(n=n)
    rng = date_range(start="1/1/2000", periods=100000, freq="min")
    ser = Series(rng)

    res = rng + offset
    assert res.freq is None  # not retained
    assert res[0] == rng[0] + offset
    assert res[-1] == rng[-1] + offset
    res2 = ser + offset
    # apply_index is only for indexes, not series, so no res2_v2
    assert res2.iloc[0] == ser.iloc[0] + offset
    assert res2.iloc[-1] == ser.iloc[-1] + offset

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_resolution.py ===
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (
    Resolution,
    get_resolution,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit

import pandas._testing as tm


def test_get_resolution_nano():
    # don't return the fallback RESO_DAY
    arr = np.array([1], dtype=np.int64)
    res = get_resolution(arr)
    assert res == Resolution.RESO_NS


def test_get_resolution_non_nano_data():
    arr = np.array([1], dtype=np.int64)
    res = get_resolution(arr, None, NpyDatetimeUnit.NPY_FR_us.value)
    assert res == Resolution.RESO_US

    res = get_resolution(arr, pytz.UTC, NpyDatetimeUnit.NPY_FR_us.value)
    assert res == Resolution.RESO_US


@pytest.mark.parametrize(
    "freqstr,expected",
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
        ("ns", "nanosecond"),
    ],
)
def test_get_attrname_from_abbrev(freqstr, expected):
    reso = Resolution.get_reso_from_freqstr(freqstr)
    assert reso.attr_abbrev == freqstr
    assert reso.attrname == expected


@pytest.mark.parametrize("freq", ["A", "H", "T", "S", "L", "U", "N"])
def test_units_A_H_T_S_L_U_N_deprecated_from_attrname_to_abbrevs(freq):
    # GH#52536
    msg = f"'{freq}' is deprecated and will be removed in a future version."

    with tm.assert_produces_warning(FutureWarning, match=msg):
        Resolution.get_reso_from_freqstr(freq)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_kind.py ===
from sympy.core.add import Add
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.numbers import pi, zoo, I, AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.core.function import Derivative
from sympy.matrices import (Matrix, SparseMatrix, ImmutableMatrix,
    ImmutableSparseMatrix, MatrixSymbol, MatrixKind, MatMul)

comm_x = Symbol('x')
noncomm_x = Symbol('x', commutative=False)

def test_NumberKind():
    assert S.One.kind is NumberKind
    assert pi.kind is NumberKind
    assert S.NaN.kind is NumberKind
    assert zoo.kind is NumberKind
    assert I.kind is NumberKind
    assert AlgebraicNumber(1).kind is NumberKind

def test_Add_kind():
    assert Add(2, 3, evaluate=False).kind is NumberKind
    assert Add(2,comm_x).kind is NumberKind
    assert Add(2,noncomm_x).kind is UndefinedKind

def test_mul_kind():
    assert Mul(2,comm_x, evaluate=False).kind is NumberKind
    assert Mul(2,3, evaluate=False).kind is NumberKind
    assert Mul(noncomm_x,2, evaluate=False).kind is UndefinedKind
    assert Mul(2,noncomm_x, evaluate=False).kind is UndefinedKind

def test_Symbol_kind():
    assert comm_x.kind is NumberKind
    assert noncomm_x.kind is UndefinedKind

def test_Integral_kind():
    A = MatrixSymbol('A', 2,2)
    assert Integral(comm_x, comm_x).kind is NumberKind
    assert Integral(A, comm_x).kind is MatrixKind(NumberKind)

def test_Derivative_kind():
    A = MatrixSymbol('A', 2,2)
    assert Derivative(comm_x, comm_x).kind is NumberKind
    assert Derivative(A, comm_x).kind is MatrixKind(NumberKind)

def test_Matrix_kind():
    classes = (Matrix, SparseMatrix, ImmutableMatrix, ImmutableSparseMatrix)
    for cls in classes:
        m = cls.zeros(3, 2)
        assert m.kind is MatrixKind(NumberKind)

def test_MatMul_kind():
    M = Matrix([[1,2],[3,4]])
    assert MatMul(2, M).kind is MatrixKind(NumberKind)
    assert MatMul(comm_x, M).kind is MatrixKind(NumberKind)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\tests\test_polarization.py ===
from sympy.physics.optics.polarization import (jones_vector, stokes_vector,
    jones_2_stokes, linear_polarizer, phase_retarder, half_wave_retarder,
    quarter_wave_retarder, transmissive_filter, reflective_filter,
    mueller_matrix, polarizing_beam_splitter)
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix


def test_polarization():
    assert jones_vector(0, 0) == Matrix([1, 0])
    assert jones_vector(pi/2, 0) == Matrix([0, 1])
    #################################################################
    assert stokes_vector(0, 0) == Matrix([1, 1, 0, 0])
    assert stokes_vector(pi/2, 0) == Matrix([1, -1, 0, 0])
    #################################################################
    H = jones_vector(0, 0)
    V = jones_vector(pi/2, 0)
    D = jones_vector(pi/4, 0)
    A = jones_vector(-pi/4, 0)
    R = jones_vector(0, pi/4)
    L = jones_vector(0, -pi/4)

    res = [Matrix([1, 1, 0, 0]),
           Matrix([1, -1, 0, 0]),
           Matrix([1, 0, 1, 0]),
           Matrix([1, 0, -1, 0]),
           Matrix([1, 0, 0, 1]),
           Matrix([1, 0, 0, -1])]

    assert [jones_2_stokes(e) for e in [H, V, D, A, R, L]] == res
    #################################################################
    assert linear_polarizer(0) == Matrix([[1, 0], [0, 0]])
    #################################################################
    delta = symbols("delta", real=True)
    res = Matrix([[exp(-I*delta/2), 0], [0, exp(I*delta/2)]])
    assert phase_retarder(0, delta) == res
    #################################################################
    assert half_wave_retarder(0) == Matrix([[-I, 0], [0, I]])
    #################################################################
    res = Matrix([[exp(-I*pi/4), 0], [0, I*exp(-I*pi/4)]])
    assert quarter_wave_retarder(0) == res
    #################################################################
    assert transmissive_filter(1) == Matrix([[1, 0], [0, 1]])
    #################################################################
    assert reflective_filter(1) == Matrix([[1, 0], [0, -1]])

    res = Matrix([[S(1)/2, S(1)/2, 0, 0],
                  [S(1)/2, S(1)/2, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    assert mueller_matrix(linear_polarizer(0)) == res
    #################################################################
    res = Matrix([[1, 0, 0, 0], [0, 0, 0, -I], [0, 0, 1, 0], [0, -I, 0, 0]])
    assert polarizing_beam_splitter() == res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_paulialgebra.py ===
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.physics.paulialgebra import Pauli
from sympy.testing.pytest import XFAIL
from sympy.physics.quantum import TensorProduct

sigma1 = Pauli(1)
sigma2 = Pauli(2)
sigma3 = Pauli(3)

tau1 = symbols("tau1", commutative = False)


def test_Pauli():

    assert sigma1 == sigma1
    assert sigma1 != sigma2

    assert sigma1*sigma2 == I*sigma3
    assert sigma3*sigma1 == I*sigma2
    assert sigma2*sigma3 == I*sigma1

    assert sigma1*sigma1 == 1
    assert sigma2*sigma2 == 1
    assert sigma3*sigma3 == 1

    assert sigma1**0 == 1
    assert sigma1**1 == sigma1
    assert sigma1**2 == 1
    assert sigma1**3 == sigma1
    assert sigma1**4 == 1

    assert sigma3**2 == 1

    assert sigma1*2*sigma1 == 2


def test_evaluate_pauli_product():
    from sympy.physics.paulialgebra import evaluate_pauli_product

    assert evaluate_pauli_product(I*sigma2*sigma3) == -sigma1

    # Check issue 6471
    assert evaluate_pauli_product(-I*4*sigma1*sigma2) == 4*sigma3

    assert evaluate_pauli_product(
        1 + I*sigma1*sigma2*sigma1*sigma2 + \
        I*sigma1*sigma2*tau1*sigma1*sigma3 + \
        ((tau1**2).subs(tau1, I*sigma1)) + \
        sigma3*((tau1**2).subs(tau1, I*sigma1)) + \
        TensorProduct(I*sigma1*sigma2*sigma1*sigma2, 1)
    ) == 1 -I + I*sigma3*tau1*sigma2 - 1 - sigma3 - I*TensorProduct(1,1)


@XFAIL
def test_Pauli_should_work():
    assert sigma1*sigma3*sigma1 == -sigma3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\test_functions.py ===
from sympy.tensor.functions import TensorProduct
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.array import Array
from sympy.abc import x, y, z
from sympy.abc import i, j, k, l


A = MatrixSymbol("A", 3, 3)
B = MatrixSymbol("B", 3, 3)
C = MatrixSymbol("C", 3, 3)


def test_TensorProduct_construction():
    assert TensorProduct(3, 4) == 12
    assert isinstance(TensorProduct(A, A), TensorProduct)

    expr = TensorProduct(TensorProduct(x, y), z)
    assert expr == x*y*z

    expr = TensorProduct(TensorProduct(A, B), C)
    assert expr == TensorProduct(A, B, C)

    expr = TensorProduct(Matrix.eye(2), Array([[0, -1], [1, 0]]))
    assert expr == Array([
        [
            [[0, -1], [1, 0]],
            [[0, 0], [0, 0]]
        ],
        [
            [[0, 0], [0, 0]],
            [[0, -1], [1, 0]]
        ]
    ])


def test_TensorProduct_shape():

    expr = TensorProduct(3, 4, evaluate=False)
    assert expr.shape == ()
    assert expr.rank() == 0

    expr = TensorProduct(Array([1, 2]), Array([x, y]), evaluate=False)
    assert expr.shape == (2, 2)
    assert expr.rank() == 2
    expr = TensorProduct(expr, expr, evaluate=False)
    assert expr.shape == (2, 2, 2, 2)
    assert expr.rank() == 4

    expr = TensorProduct(Matrix.eye(2), Array([[0, -1], [1, 0]]), evaluate=False)
    assert expr.shape == (2, 2, 2, 2)
    assert expr.rank() == 4


def test_TensorProduct_getitem():
    expr = TensorProduct(A, B)
    assert expr[i, j, k, l] == A[i, j]*B[k, l]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_isoc.py ===
import pytest

import numpy as np
from numpy.testing import assert_allclose

from . import util


class TestISOC(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
    ]

    # gh-24553
    @pytest.mark.slow
    def test_c_double(self):
        out = self.module.coddity.c_add(1, 2)
        exp_out = 3
        assert out == exp_out

    # gh-9693
    def test_bindc_function(self):
        out = self.module.coddity.wat(1, 20)
        exp_out = 8
        assert out == exp_out

    # gh-25207
    def test_bindc_kinds(self):
        out = self.module.coddity.c_add_int64(1, 20)
        exp_out = 21
        assert out == exp_out

    # gh-25207
    def test_bindc_add_arr(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        out = self.module.coddity.add_arr(a, b)
        exp_out = a * 2
        assert_allclose(out, exp_out)


def test_process_f2cmap_dict():
    from numpy.f2py.auxfuncs import process_f2cmap_dict

    f2cmap_all = {"integer": {"8": "rubbish_type"}}
    new_map = {"INTEGER": {"4": "int"}}
    c2py_map = {"int": "int", "rubbish_type": "long"}

    exp_map, exp_maptyp = ({"integer": {"8": "rubbish_type", "4": "int"}}, ["int"])

    # Call the function
    res_map, res_maptyp = process_f2cmap_dict(f2cmap_all, new_map, c2py_map)

    # Assert the result is as expected
    assert res_map == exp_map
    assert res_maptyp == exp_maptyp

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\test_transpose.py ===
import numpy as np
import pytest

from pandas import (
    CategoricalDtype,
    DataFrame,
)
import pandas._testing as tm


def test_transpose(index_or_series_obj):
    obj = index_or_series_obj
    tm.assert_equal(obj.transpose(), obj)


def test_transpose_non_default_axes(index_or_series_obj):
    msg = "the 'axes' parameter is not supported"
    obj = index_or_series_obj
    with pytest.raises(ValueError, match=msg):
        obj.transpose(1)
    with pytest.raises(ValueError, match=msg):
        obj.transpose(axes=1)


def test_numpy_transpose(index_or_series_obj):
    msg = "the 'axes' parameter is not supported"
    obj = index_or_series_obj
    tm.assert_equal(np.transpose(obj), obj)

    with pytest.raises(ValueError, match=msg):
        np.transpose(obj, axes=1)


@pytest.mark.parametrize(
    "data, transposed_data, index, columns, dtype",
    [
        ([[1], [2]], [[1, 2]], ["a", "a"], ["b"], int),
        ([[1], [2]], [[1, 2]], ["a", "a"], ["b"], CategoricalDtype([1, 2])),
        ([[1, 2]], [[1], [2]], ["b"], ["a", "a"], int),
        ([[1, 2]], [[1], [2]], ["b"], ["a", "a"], CategoricalDtype([1, 2])),
        ([[1, 2], [3, 4]], [[1, 3], [2, 4]], ["a", "a"], ["b", "b"], int),
        (
            [[1, 2], [3, 4]],
            [[1, 3], [2, 4]],
            ["a", "a"],
            ["b", "b"],
            CategoricalDtype([1, 2, 3, 4]),
        ),
    ],
)
def test_duplicate_labels(data, transposed_data, index, columns, dtype):
    # GH 42380
    df = DataFrame(data, index=index, columns=columns, dtype=dtype)
    result = df.T
    expected = DataFrame(transposed_data, index=columns, columns=index, dtype=dtype)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_arithmetic.py ===
# Arithmetic tests specific to DatetimeIndex are generally about `freq`
#  rentention or inference.  Other arithmetic tests belong in
#  tests/arithmetic/test_datetime64.py
import pytest

from pandas import (
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
    timedelta_range,
)
import pandas._testing as tm


class TestDatetimeIndexArithmetic:
    def test_add_timedelta_preserves_freq(self):
        # GH#37295 should hold for any DTI with freq=None or Tick freq
        tz = "Canada/Eastern"
        dti = date_range(
            start=Timestamp("2019-03-26 00:00:00-0400", tz=tz),
            end=Timestamp("2020-10-17 00:00:00-0400", tz=tz),
            freq="D",
        )
        result = dti + Timedelta(days=1)
        assert result.freq == dti.freq

    def test_sub_datetime_preserves_freq(self, tz_naive_fixture):
        # GH#48818
        dti = date_range("2016-01-01", periods=12, tz=tz_naive_fixture)

        res = dti - dti[0]
        expected = timedelta_range("0 Days", "11 Days")
        tm.assert_index_equal(res, expected)
        assert res.freq == expected.freq

    @pytest.mark.xfail(
        reason="The inherited freq is incorrect bc dti.freq is incorrect "
        "https://github.com/pandas-dev/pandas/pull/48818/files#r982793461"
    )
    def test_sub_datetime_preserves_freq_across_dst(self):
        # GH#48818
        ts = Timestamp("2016-03-11", tz="US/Pacific")
        dti = date_range(ts, periods=4)

        res = dti - dti[0]
        expected = TimedeltaIndex(
            [
                Timedelta(days=0),
                Timedelta(days=1),
                Timedelta(days=2),
                Timedelta(days=2, hours=23),
            ]
        )
        tm.assert_index_equal(res, expected)
        assert res.freq == expected.freq

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_ops.py ===
from datetime import datetime

import pytest

from pandas import (
    DatetimeIndex,
    Index,
    bdate_range,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndexOps:
    def test_infer_freq(self, freq_sample):
        # GH 11018
        idx = date_range("2011-01-01 09:00:00", freq=freq_sample, periods=10)
        result = DatetimeIndex(idx.asi8, freq="infer")
        tm.assert_index_equal(idx, result)
        assert result.freq == freq_sample


@pytest.mark.parametrize("freq", ["B", "C"])
class TestBusinessDatetimeIndex:
    @pytest.fixture
    def rng(self, freq):
        START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)
        return bdate_range(START, END, freq=freq)

    def test_comparison(self, rng):
        d = rng[10]

        comp = rng > d
        assert comp[11]
        assert not comp[9]

    def test_copy(self, rng):
        cp = rng.copy()
        tm.assert_index_equal(cp, rng)

    def test_identical(self, rng):
        t1 = rng.copy()
        t2 = rng.copy()
        assert t1.identical(t2)

        # name
        t1 = t1.rename("foo")
        assert t1.equals(t2)
        assert not t1.identical(t2)
        t2 = t2.rename("foo")
        assert t1.identical(t2)

        # freq
        t2v = Index(t2.values)
        assert t1.equals(t2v)
        assert not t1.identical(t2v)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_reindex.py ===
from datetime import timedelta

import numpy as np

from pandas import (
    DatetimeIndex,
    date_range,
)
import pandas._testing as tm


class TestDatetimeIndexReindex:
    def test_reindex_preserves_tz_if_target_is_empty_list_or_array(self):
        # GH#7774
        index = date_range("2013-01-01", periods=3, tz="US/Eastern")
        assert str(index.reindex([])[0].tz) == "US/Eastern"
        assert str(index.reindex(np.array([]))[0].tz) == "US/Eastern"

    def test_reindex_with_same_tz_nearest(self):
        # GH#32740
        rng_a = date_range("2010-01-01", "2010-01-02", periods=24, tz="utc")
        rng_b = date_range("2010-01-01", "2010-01-02", periods=23, tz="utc")
        result1, result2 = rng_a.reindex(
            rng_b, method="nearest", tolerance=timedelta(seconds=20)
        )
        expected_list1 = [
            "2010-01-01 00:00:00",
            "2010-01-01 01:05:27.272727272",
            "2010-01-01 02:10:54.545454545",
            "2010-01-01 03:16:21.818181818",
            "2010-01-01 04:21:49.090909090",
            "2010-01-01 05:27:16.363636363",
            "2010-01-01 06:32:43.636363636",
            "2010-01-01 07:38:10.909090909",
            "2010-01-01 08:43:38.181818181",
            "2010-01-01 09:49:05.454545454",
            "2010-01-01 10:54:32.727272727",
            "2010-01-01 12:00:00",
            "2010-01-01 13:05:27.272727272",
            "2010-01-01 14:10:54.545454545",
            "2010-01-01 15:16:21.818181818",
            "2010-01-01 16:21:49.090909090",
            "2010-01-01 17:27:16.363636363",
            "2010-01-01 18:32:43.636363636",
            "2010-01-01 19:38:10.909090909",
            "2010-01-01 20:43:38.181818181",
            "2010-01-01 21:49:05.454545454",
            "2010-01-01 22:54:32.727272727",
            "2010-01-02 00:00:00",
        ]
        expected1 = DatetimeIndex(
            expected_list1, dtype="datetime64[ns, UTC]", freq=None
        )
        expected2 = np.array([0] + [-1] * 21 + [23], dtype=np.dtype("intp"))
        tm.assert_index_equal(result1, expected1)
        tm.assert_numpy_array_equal(result2, expected2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\conftest.py ===
import gc

import numpy as np
import pytest

from pandas import (
    DataFrame,
    to_datetime,
)


@pytest.fixture(autouse=True)
def mpl_cleanup():
    # matplotlib/testing/decorators.py#L24
    # 1) Resets units registry
    # 2) Resets rc_context
    # 3) Closes all figures
    mpl = pytest.importorskip("matplotlib")
    mpl_units = pytest.importorskip("matplotlib.units")
    plt = pytest.importorskip("matplotlib.pyplot")
    orig_units_registry = mpl_units.registry.copy()
    with mpl.rc_context():
        mpl.use("template")
        yield
    mpl_units.registry.clear()
    mpl_units.registry.update(orig_units_registry)
    plt.close("all")
    # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.6.0.html#garbage-collection-is-no-longer-run-on-figure-close  # noqa: E501
    gc.collect(1)


@pytest.fixture
def hist_df():
    n = 50
    rng = np.random.default_rng(10)
    gender = rng.choice(["Male", "Female"], size=n)
    classroom = rng.choice(["A", "B", "C"], size=n)

    hist_df = DataFrame(
        {
            "gender": gender,
            "classroom": classroom,
            "height": rng.normal(66, 4, size=n),
            "weight": rng.normal(161, 32, size=n),
            "category": rng.integers(4, size=n),
            "datetime": to_datetime(
                rng.integers(
                    812419200000000000,
                    819331200000000000,
                    size=n,
                    dtype=np.int64,
                )
            ),
        }
    )
    return hist_df

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_infer_objects.py ===
import numpy as np

from pandas import (
    Series,
    interval_range,
)
import pandas._testing as tm


class TestInferObjects:
    def test_copy(self, index_or_series):
        # GH#50096
        # case where we don't need to do inference because it is already non-object
        obj = index_or_series(np.array([1, 2, 3], dtype="int64"))

        result = obj.infer_objects(copy=False)
        assert tm.shares_memory(result, obj)

        # case where we try to do inference but can't do better than object
        obj2 = index_or_series(np.array(["foo", 2], dtype=object))
        result2 = obj2.infer_objects(copy=False)
        assert tm.shares_memory(result2, obj2)

    def test_infer_objects_series(self, index_or_series):
        # GH#11221
        actual = index_or_series(np.array([1, 2, 3], dtype="O")).infer_objects()
        expected = index_or_series([1, 2, 3])
        tm.assert_equal(actual, expected)

        actual = index_or_series(np.array([1, 2, 3, None], dtype="O")).infer_objects()
        expected = index_or_series([1.0, 2.0, 3.0, np.nan])
        tm.assert_equal(actual, expected)

        # only soft conversions, unconvertible pass thru unchanged

        obj = index_or_series(np.array([1, 2, 3, None, "a"], dtype="O"))
        actual = obj.infer_objects()
        expected = index_or_series([1, 2, 3, None, "a"], dtype=object)

        assert actual.dtype == "object"
        tm.assert_equal(actual, expected)

    def test_infer_objects_interval(self, index_or_series):
        # GH#50090
        ii = interval_range(1, 10)
        obj = index_or_series(ii)

        result = obj.astype(object).infer_objects()
        tm.assert_equal(result, obj)

    def test_infer_objects_bytes(self):
        # GH#49650
        ser = Series([b"a"], dtype="bytes")
        expected = ser.copy()
        result = ser.infer_objects()
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_anticommutator.py ===
from sympy.core.numbers import Integer
from sympy.core.symbol import symbols

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.anticommutator import AntiCommutator as AComm
from sympy.physics.quantum.operator import Operator


a, b, c = symbols('a,b,c')
A, B, C, D = symbols('A,B,C,D', commutative=False)


def test_anticommutator():
    ac = AComm(A, B)
    assert isinstance(ac, AComm)
    assert ac.is_commutative is False
    assert ac.subs(A, C) == AComm(C, B)


def test_commutator_identities():
    assert AComm(a*A, b*B) == a*b*AComm(A, B)
    assert AComm(A, A) == 2*A**2
    assert AComm(A, B) == AComm(B, A)
    assert AComm(a, b) == 2*a*b
    assert AComm(A, B).doit() == A*B + B*A


def test_anticommutator_dagger():
    assert Dagger(AComm(A, B)) == AComm(Dagger(A), Dagger(B))


class Foo(Operator):

    def _eval_anticommutator_Bar(self, bar):
        return Integer(0)


class Bar(Operator):
    pass


class Tam(Operator):

    def _eval_anticommutator_Foo(self, foo):
        return Integer(1)


def test_eval_commutator():
    F = Foo('F')
    B = Bar('B')
    T = Tam('T')
    assert AComm(F, B).doit() == 0
    assert AComm(B, F).doit() == 0
    assert AComm(F, T).doit() == 1
    assert AComm(T, F).doit() == 1
    assert AComm(B, T).doit() == B*T + T*B

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_cupy.py ===
from sympy.concrete.summations import Sum
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, i, a, b
from sympy.codegen.numpy_nodes import logaddexp
from sympy.printing.numpy import CuPyPrinter, _cupy_known_constants, _cupy_known_functions

from sympy.testing.pytest import skip, raises
from sympy.external import import_module

cp = import_module('cupy')

def test_cupy_print():
    prntr = CuPyPrinter()
    assert prntr.doprint(logaddexp(a, b)) == 'cupy.logaddexp(a, b)'
    assert prntr.doprint(sqrt(x)) == 'cupy.sqrt(x)'
    assert prntr.doprint(log(x)) == 'cupy.log(x)'
    assert prntr.doprint("acos(x)") == 'cupy.arccos(x)'
    assert prntr.doprint("exp(x)") == 'cupy.exp(x)'
    assert prntr.doprint("Abs(x)") == 'abs(x)'

def test_not_cupy_print():
    prntr = CuPyPrinter()
    with raises(NotImplementedError):
        prntr.doprint("abcd(x)")

def test_cupy_sum():
    if not cp:
        skip("CuPy not installed")

    s = Sum(x ** i, (i, a, b))
    f = lambdify((a, b, x), s, 'cupy')

    a_, b_ = 0, 10
    x_ = cp.linspace(-1, +1, 10)
    assert cp.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))

    s = Sum(i * x, (i, a, b))
    f = lambdify((a, b, x), s, 'numpy')

    a_, b_ = 0, 10
    x_ = cp.linspace(-1, +1, 10)
    assert cp.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))

def test_cupy_known_funcs_consts():
    assert _cupy_known_constants['NaN'] == 'cupy.nan'
    assert _cupy_known_constants['EulerGamma'] == 'cupy.euler_gamma'

    assert _cupy_known_functions['acos'] == 'cupy.arccos'
    assert _cupy_known_functions['log'] == 'cupy.log'

def test_cupy_print_methods():
    prntr = CuPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\fail_switch_three_greenlets2.py ===
"""
Like fail_switch_three_greenlets, but the call into g1_run would actually be
valid.
"""
import greenlet

g1 = None
g2 = None

switch_to_g2 = True

results = []

def tracefunc(*args):
    results.append(('trace', args[0]))
    print('TRACE', *args)
    global switch_to_g2
    if switch_to_g2:
        switch_to_g2 = False
        g2.switch('g2 from tracefunc')
    print('\tLEAVE TRACE', *args)

def g1_run(arg):
    results.append(('g1 arg', arg))
    print('In g1_run')
    from_parent = greenlet.getcurrent().parent.switch('from g1_run')
    results.append(('g1 from parent', from_parent))
    return 'g1 done'

def g2_run(arg):
    #g1.switch()
    results.append(('g2 arg', arg))
    parent = greenlet.getcurrent().parent.switch('from g2_run')
    global switch_to_g2
    switch_to_g2 = False
    results.append(('g2 from parent', parent))
    return 'g2 done'


greenlet.settrace(tracefunc)

g1 = greenlet.greenlet(g1_run)
g2 = greenlet.greenlet(g2_run)

x = g1.switch('g1 from main')
results.append(('main g1', x))
print('Back in main', x)
x = g1.switch('g2 from main')
results.append(('main g2', x))
print('back in amain again', x)
x = g1.switch('g1 from main 2')
results.append(('main g1.2', x))
x = g2.switch()
results.append(('main g2.2', x))
print("RESULTS:", results)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_return_integer.py ===
import pytest

from numpy import array

from . import util


@pytest.mark.slow
class TestReturnInteger(util.F2PyTest):
    def check_function(self, t, tname):
        assert t(123) == 123
        assert t(123.6) == 123
        assert t("123") == 123
        assert t(-123) == -123
        assert t([123]) == 123
        assert t((123, )) == 123
        assert t(array(123)) == 123
        assert t(array(123, "b")) == 123
        assert t(array(123, "h")) == 123
        assert t(array(123, "i")) == 123
        assert t(array(123, "l")) == 123
        assert t(array(123, "B")) == 123
        assert t(array(123, "f")) == 123
        assert t(array(123, "d")) == 123

        # pytest.raises(ValueError, t, array([123],'S3'))
        pytest.raises(ValueError, t, "abc")

        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        pytest.raises(Exception, t, t)
        pytest.raises(Exception, t, {})

        if tname in ["t8", "s8"]:
            pytest.raises(OverflowError, t, 100000000000000000000000)
            pytest.raises(OverflowError, t, 10000000011111111111111.23)


class TestFReturnInteger(TestReturnInteger):
    sources = [
        util.getpath("tests", "src", "return_integer", "foo77.f"),
        util.getpath("tests", "src", "return_integer", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name",
                             ["t0", "t1", "t2", "t4", "t8", "s0", "s1", "s2", "s4", "s8"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name",
                             ["t0", "t1", "t2", "t4", "t8", "s0", "s1", "s2", "s4", "s8"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_integer, name),
                            name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_construct_from_scalar.py ===
import numpy as np
import pytest

from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas import (
    Categorical,
    Timedelta,
)
import pandas._testing as tm


def test_cast_1d_array_like_from_scalar_categorical():
    # see gh-19565
    #
    # Categorical result from scalar did not maintain
    # categories and ordering of the passed dtype.
    cats = ["a", "b", "c"]
    cat_type = CategoricalDtype(categories=cats, ordered=False)
    expected = Categorical(["a", "a"], categories=cats)

    result = construct_1d_arraylike_from_scalar("a", len(expected), cat_type)
    tm.assert_categorical_equal(result, expected)


def test_cast_1d_array_like_from_timestamp(fixed_now_ts):
    # check we dont lose nanoseconds
    ts = fixed_now_ts + Timedelta(1)
    res = construct_1d_arraylike_from_scalar(ts, 2, np.dtype("M8[ns]"))
    assert res[0] == ts


def test_cast_1d_array_like_from_timedelta():
    # check we dont lose nanoseconds
    td = Timedelta(1)
    res = construct_1d_arraylike_from_scalar(td, 2, np.dtype("m8[ns]"))
    assert res[0] == td


def test_cast_1d_array_like_mismatched_datetimelike():
    td = np.timedelta64("NaT", "ns")
    dt = np.datetime64("NaT", "ns")

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(td, 2, dt.dtype)

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(np.timedelta64(4, "ns"), 2, dt.dtype)

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(dt, 2, td.dtype)

    with pytest.raises(TypeError, match="Cannot cast"):
        construct_1d_arraylike_from_scalar(np.datetime64(4, "ns"), 2, td.dtype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\sas\test_byteswap.py ===
from hypothesis import (
    assume,
    example,
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs.byteswap import (
    read_double_with_byteswap,
    read_float_with_byteswap,
    read_uint16_with_byteswap,
    read_uint32_with_byteswap,
    read_uint64_with_byteswap,
)

import pandas._testing as tm


@given(read_offset=st.integers(0, 11), number=st.integers(min_value=0))
@example(number=2**16, read_offset=0)
@example(number=2**32, read_offset=0)
@example(number=2**64, read_offset=0)
@pytest.mark.parametrize("int_type", [np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("should_byteswap", [True, False])
def test_int_byteswap(read_offset, number, int_type, should_byteswap):
    assume(number < 2 ** (8 * int_type(0).itemsize))
    _test(number, int_type, read_offset, should_byteswap)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@given(read_offset=st.integers(0, 11), number=st.floats())
@pytest.mark.parametrize("float_type", [np.float32, np.float64])
@pytest.mark.parametrize("should_byteswap", [True, False])
def test_float_byteswap(read_offset, number, float_type, should_byteswap):
    _test(number, float_type, read_offset, should_byteswap)


def _test(number, number_type, read_offset, should_byteswap):
    number = number_type(number)
    data = np.random.default_rng(2).integers(0, 256, size=20, dtype="uint8")
    data[read_offset : read_offset + number.itemsize] = number[None].view("uint8")
    swap_func = {
        np.float32: read_float_with_byteswap,
        np.float64: read_double_with_byteswap,
        np.uint16: read_uint16_with_byteswap,
        np.uint32: read_uint32_with_byteswap,
        np.uint64: read_uint64_with_byteswap,
    }[type(number)]
    output_number = number_type(swap_func(bytes(data), read_offset, should_byteswap))
    if should_byteswap:
        tm.assert_equal(output_number, number.byteswap())
    else:
        tm.assert_equal(output_number, number)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_schur_number.py ===
from sympy.core import S, Rational
from sympy.combinatorics.schur_number import schur_partition, SchurNumber
from sympy.core.random import _randint
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols


def _sum_free_test(subset):
    """
    Checks if subset is sum-free(There are no x,y,z in the subset such that
    x + y = z)
    """
    for i in subset:
        for j in subset:
            assert (i + j in subset) is False


def test_schur_partition():
    raises(ValueError, lambda: schur_partition(S.Infinity))
    raises(ValueError, lambda: schur_partition(-1))
    raises(ValueError, lambda: schur_partition(0))
    assert schur_partition(2) == [[1, 2]]

    random_number_generator = _randint(1000)
    for _ in range(5):
        n = random_number_generator(1, 1000)
        result = schur_partition(n)
        t = 0
        numbers = []
        for item in result:
            _sum_free_test(item)
            """
            Checks if the occurrence of all numbers is exactly one
            """
            t += len(item)
            for l in item:
                assert (l in numbers) is False
                numbers.append(l)
        assert n == t

    x = symbols("x")
    raises(ValueError, lambda: schur_partition(x))

def test_schur_number():
    first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}
    for k in first_known_schur_numbers:
        assert SchurNumber(k) == first_known_schur_numbers[k]

    assert SchurNumber(S.Infinity) == S.Infinity
    assert SchurNumber(0) == 0
    raises(ValueError, lambda: SchurNumber(0.5))

    n = symbols("n")
    assert SchurNumber(n).lower_bound() == 3**n/2 - Rational(1, 2)
    assert SchurNumber(8).lower_bound() == 5039

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_testutil.py ===
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup,\
    CyclicGroup
from sympy.combinatorics.testutil import _verify_bsgs, _cmp_perm_lists,\
    _naive_list_centralizer, _verify_centralizer,\
    _verify_normal_closure
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.random import shuffle


def test_cmp_perm_lists():
    S = SymmetricGroup(4)
    els = list(S.generate_dimino())
    other = els.copy()
    shuffle(other)
    assert _cmp_perm_lists(els, other) is True


def test_naive_list_centralizer():
    # verified by GAP
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    assert _naive_list_centralizer(S, S) == [Permutation([0, 1, 2])]
    assert PermutationGroup(_naive_list_centralizer(S, A)).is_subgroup(A)


def test_verify_bsgs():
    S = SymmetricGroup(5)
    S.schreier_sims()
    base = S.base
    strong_gens = S.strong_gens
    assert _verify_bsgs(S, base, strong_gens) is True
    assert _verify_bsgs(S, base[:-1], strong_gens) is False
    assert _verify_bsgs(S, base, S.generators) is False


def test_verify_centralizer():
    # verified by GAP
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert _verify_centralizer(S, S, centr=triv)
    assert _verify_centralizer(S, A, centr=A)


def test_verify_normal_closure():
    # verified by GAP
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    assert _verify_normal_closure(S, A, closure=A)
    S = SymmetricGroup(5)
    A = AlternatingGroup(5)
    C = CyclicGroup(5)
    assert _verify_normal_closure(S, A, closure=A)
    assert _verify_normal_closure(S, C, closure=A)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_digits.py ===
from sympy.ntheory import count_digits, digits, is_palindromic
from sympy.core.intfunc import num_digits

from sympy.testing.pytest import raises


def test_num_digits():
    # depending on whether one rounds up or down or uses log or log10,
    # one or more of these will fail if you don't check for the off-by
    # one condition
    assert num_digits(2, 2) == 2
    assert num_digits(2**48 - 1, 2) == 48
    assert num_digits(1000, 10) == 4
    assert num_digits(125, 5) == 4
    assert num_digits(100, 16) == 2
    assert num_digits(-1000, 10) == 4
    # if changes are made to the function, this structured test over
    # this range will expose problems
    for base in range(2, 100):
        for e in range(1, 100):
            n = base**e
            assert num_digits(n, base) == e + 1
            assert num_digits(n + 1, base) == e + 1
            assert num_digits(n - 1, base) == e


def test_digits():
    assert all(digits(n, 2)[1:] == [int(d) for d in format(n, 'b')]
                for n in range(20))
    assert all(digits(n, 8)[1:] == [int(d) for d in format(n, 'o')]
                for n in range(20))
    assert all(digits(n, 16)[1:] == [int(d, 16) for d in format(n, 'x')]
                for n in range(20))
    assert digits(2345, 34) == [34, 2, 0, 33]
    assert digits(384753, 71) == [71, 1, 5, 23, 4]
    assert digits(93409, 10) == [10, 9, 3, 4, 0, 9]
    assert digits(-92838, 11) == [-11, 6, 3, 8, 2, 9]
    assert digits(35, 10) == [10, 3, 5]
    assert digits(35, 10, 3) == [10, 0, 3, 5]
    assert digits(-35, 10, 4) == [-10, 0, 0, 3, 5]
    raises(ValueError, lambda: digits(2, 2, 1))


def test_count_digits():
    assert count_digits(55, 2) == {1: 5, 0: 1}
    assert count_digits(55, 10) == {5: 2}
    n = count_digits(123)
    assert n[4] == 0 and type(n[4]) is int


def test_is_palindromic():
    assert is_palindromic(-11)
    assert is_palindromic(11)
    assert is_palindromic(0o121, 8)
    assert not is_palindromic(123)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\test_unit_system_cgs_gauss.py ===
from sympy.concrete.tests.test_sums_products import NS

from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import convert_to, coulomb_constant, elementary_charge, gravitational_constant, planck
from sympy.physics.units.definitions.unit_definitions import angstrom, statcoulomb, coulomb, second, gram, centimeter, erg, \
    newton, joule, dyne, speed_of_light, meter, farad, henry, statvolt, volt, ohm
from sympy.physics.units.systems import SI
from sympy.physics.units.systems.cgs import cgs_gauss


def test_conversion_to_from_si():
    assert convert_to(statcoulomb, coulomb, cgs_gauss) == coulomb/2997924580
    assert convert_to(coulomb, statcoulomb, cgs_gauss) == 2997924580*statcoulomb
    assert convert_to(statcoulomb, sqrt(gram*centimeter**3)/second, cgs_gauss) == centimeter**(S(3)/2)*sqrt(gram)/second
    assert convert_to(coulomb, sqrt(gram*centimeter**3)/second, cgs_gauss) == 2997924580*centimeter**(S(3)/2)*sqrt(gram)/second

    # SI units have an additional base unit, no conversion in case of electromagnetism:
    assert convert_to(coulomb, statcoulomb, SI) == coulomb
    assert convert_to(statcoulomb, coulomb, SI) == statcoulomb

    # SI without electromagnetism:
    assert convert_to(erg, joule, SI) == joule/10**7
    assert convert_to(erg, joule, cgs_gauss) == joule/10**7
    assert convert_to(joule, erg, SI) == 10**7*erg
    assert convert_to(joule, erg, cgs_gauss) == 10**7*erg


    assert convert_to(dyne, newton, SI) == newton/10**5
    assert convert_to(dyne, newton, cgs_gauss) == newton/10**5
    assert convert_to(newton, dyne, SI) == 10**5*dyne
    assert convert_to(newton, dyne, cgs_gauss) == 10**5*dyne


def test_cgs_gauss_convert_constants():

    assert convert_to(speed_of_light, centimeter/second, cgs_gauss) == 29979245800*centimeter/second

    assert convert_to(coulomb_constant, 1, cgs_gauss) == 1
    assert convert_to(coulomb_constant, newton*meter**2/coulomb**2, cgs_gauss) == 22468879468420441*meter**2*newton/(2500000*coulomb**2)
    assert convert_to(coulomb_constant, newton*meter**2/coulomb**2, SI) == 22468879468420441*meter**2*newton/(2500000*coulomb**2)
    assert convert_to(coulomb_constant, dyne*centimeter**2/statcoulomb**2, cgs_gauss) == centimeter**2*dyne/statcoulomb**2
    assert convert_to(coulomb_constant, 1, SI) == coulomb_constant
    assert NS(convert_to(coulomb_constant, newton*meter**2/coulomb**2, SI)) == '8987551787.36818*meter**2*newton/coulomb**2'

    assert convert_to(elementary_charge, statcoulomb, cgs_gauss)
    assert convert_to(angstrom, centimeter, cgs_gauss) == 1*centimeter/10**8
    assert convert_to(gravitational_constant, dyne*centimeter**2/gram**2, cgs_gauss)
    assert NS(convert_to(planck, erg*second, cgs_gauss)) == '6.62607015e-27*erg*second'

    spc = 25000*second/(22468879468420441*centimeter)
    assert convert_to(ohm, second/centimeter, cgs_gauss) == spc
    assert convert_to(henry, second**2/centimeter, cgs_gauss) == spc*second
    assert convert_to(volt, statvolt, cgs_gauss) == 10**6*statvolt/299792458
    assert convert_to(farad, centimeter, cgs_gauss) == 299792458**2*centimeter/10**5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_aseries.py ===
from sympy.core.function import PoleError
from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.abc import x

from sympy.testing.pytest import raises

def test_simple():
    # Gruntz' theses pp. 91 to 96
    # 6.6
    e = sin(1/x + exp(-x)) - sin(1/x)
    assert e.aseries(x) == (1/(24*x**4) - 1/(2*x**2) + 1 + O(x**(-6), (x, oo)))*exp(-x)

    e = exp(x) * (exp(1/x + exp(-x)) - exp(1/x))
    assert e.aseries(x, n=4) == 1/(6*x**3) + 1/(2*x**2) + 1/x + 1 + O(x**(-4), (x, oo))

    e = exp(exp(x) / (1 - 1/x))
    assert e.aseries(x) == exp(exp(x) / (1 - 1/x))

    # The implementation of bound in aseries is incorrect currently. This test
    # should be commented out when that is fixed.
    # assert e.aseries(x, bound=3) == exp(exp(x) / x**2)*exp(exp(x) / x)*exp(-exp(x) + exp(x)/(1 - 1/x) - \
    #         exp(x) / x - exp(x) / x**2) * exp(exp(x))

    e = exp(sin(1/x + exp(-exp(x)))) - exp(sin(1/x))
    assert e.aseries(x, n=4) == (-1/(2*x**3) + 1/x + 1 + O(x**(-4), (x, oo)))*exp(-exp(x))

    e3 = lambda x:exp(exp(exp(x)))
    e = e3(x)/e3(x - 1/e3(x))
    assert e.aseries(x, n=3) == 1 + exp(2*x + 2*exp(x))*exp(-2*exp(exp(x)))/2\
            - exp(2*x + exp(x))*exp(-2*exp(exp(x)))/2 - exp(x + exp(x))*exp(-2*exp(exp(x)))/2\
            + exp(x + exp(x))*exp(-exp(exp(x))) + O(exp(-3*exp(exp(x))), (x, oo))

    e = exp(exp(x)) * (exp(sin(1/x + 1/exp(exp(x)))) - exp(sin(1/x)))
    assert e.aseries(x, n=4) == -1/(2*x**3) + 1/x + 1 + O(x**(-4), (x, oo))

    n = Symbol('n', integer=True)
    e = (sqrt(n)*log(n)**2*exp(sqrt(log(n))*log(log(n))**2*exp(sqrt(log(log(n)))*log(log(log(n)))**3)))/n
    assert e.aseries(n) == \
            exp(exp(sqrt(log(log(n)))*log(log(log(n)))**3)*sqrt(log(n))*log(log(n))**2)*log(n)**2/sqrt(n)


def test_hierarchical():
    e = sin(1/x + exp(-x))
    assert e.aseries(x, n=3, hir=True) == -exp(-2*x)*sin(1/x)/2 + \
            exp(-x)*cos(1/x) + sin(1/x) + O(exp(-3*x), (x, oo))

    e = sin(x) * cos(exp(-x))
    assert e.aseries(x, hir=True) == exp(-4*x)*sin(x)/24 - \
            exp(-2*x)*sin(x)/2 + sin(x) + O(exp(-6*x), (x, oo))
    raises(PoleError, lambda: e.aseries(x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_numpy_version.py ===
"""
Check the numpy version is valid.

Note that a development version is marked by the presence of 'dev0' or '+'
in the version string, all else is treated as a release. The version string
itself is set from the output of ``git describe`` which relies on tags.

Examples
--------

Valid Development: 1.22.0.dev0 1.22.0.dev0+5-g7999db4df2 1.22.0+5-g7999db4df2
Valid Release: 1.21.0.rc1, 1.21.0.b1, 1.21.0
Invalid: 1.22.0.dev, 1.22.0.dev0-5-g7999db4dfB, 1.21.0.d1, 1.21.a

Note that a release is determined by the version string, which in turn
is controlled by the result of the ``git describe`` command.
"""
import re

import numpy as np
from numpy.testing import assert_


def test_valid_numpy_version():
    # Verify that the numpy version is a valid one (no .post suffix or other
    # nonsense).  See gh-6431 for an issue caused by an invalid version.
    version_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+(a[0-9]|b[0-9]|rc[0-9])?"
    dev_suffix = r"(\.dev[0-9]+(\+git[0-9]+\.[0-9a-f]+)?)?"
    res = re.match(version_pattern + dev_suffix + '$', np.__version__)

    assert_(res is not None, np.__version__)


def test_short_version():
    # Check numpy.short_version actually exists
    if np.version.release:
        assert_(np.__version__ == np.version.short_version,
                "short_version mismatch in release version")
    else:
        assert_(np.__version__.split("+")[0] == np.version.short_version,
                "short_version mismatch in development version")


def test_version_module():
    contents = {s for s in dir(np.version) if not s.startswith('_')}
    expected = {
        'full_version',
        'git_revision',
        'release',
        'short_version',
        'version',
    }

    assert contents == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_abc.py ===
import numbers

import numpy as np
from numpy._core.numerictypes import sctypes
from numpy.testing import assert_


class TestABC:
    def test_abstract(self):
        assert_(issubclass(np.number, numbers.Number))

        assert_(issubclass(np.inexact, numbers.Complex))
        assert_(issubclass(np.complexfloating, numbers.Complex))
        assert_(issubclass(np.floating, numbers.Real))

        assert_(issubclass(np.integer, numbers.Integral))
        assert_(issubclass(np.signedinteger, numbers.Integral))
        assert_(issubclass(np.unsignedinteger, numbers.Integral))

    def test_floats(self):
        for t in sctypes['float']:
            assert_(isinstance(t(), numbers.Real),
                    f"{t.__name__} is not instance of Real")
            assert_(issubclass(t, numbers.Real),
                    f"{t.__name__} is not subclass of Real")
            assert_(not isinstance(t(), numbers.Rational),
                    f"{t.__name__} is instance of Rational")
            assert_(not issubclass(t, numbers.Rational),
                    f"{t.__name__} is subclass of Rational")

    def test_complex(self):
        for t in sctypes['complex']:
            assert_(isinstance(t(), numbers.Complex),
                    f"{t.__name__} is not instance of Complex")
            assert_(issubclass(t, numbers.Complex),
                    f"{t.__name__} is not subclass of Complex")
            assert_(not isinstance(t(), numbers.Real),
                    f"{t.__name__} is instance of Real")
            assert_(not issubclass(t, numbers.Real),
                    f"{t.__name__} is subclass of Real")

    def test_int(self):
        for t in sctypes['int']:
            assert_(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            assert_(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")

    def test_uint(self):
        for t in sctypes['uint']:
            assert_(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            assert_(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_fillna.py ===
import numpy as np
import pytest

from pandas import CategoricalIndex
import pandas._testing as tm


class TestFillNA:
    def test_fillna_categorical(self):
        # GH#11343
        idx = CategoricalIndex([1.0, np.nan, 3.0, 1.0], name="x")
        # fill by value in categories
        exp = CategoricalIndex([1.0, 1.0, 3.0, 1.0], name="x")
        tm.assert_index_equal(idx.fillna(1.0), exp)

        cat = idx._data

        # fill by value not in categories raises TypeError on EA, casts on CI
        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            cat.fillna(2.0)

        result = idx.fillna(2.0)
        expected = idx.astype(object).fillna(2.0)
        tm.assert_index_equal(result, expected)

    def test_fillna_copies_with_no_nas(self):
        # Nothing to fill, should still get a copy for the Categorical method,
        #  but OK to get a view on CategoricalIndex method
        ci = CategoricalIndex([0, 1, 1])
        result = ci.fillna(0)
        assert result is not ci
        assert tm.shares_memory(result, ci)

        # But at the EA level we always get a copy.
        cat = ci._data
        result = cat.fillna(0)
        assert result._ndarray is not cat._ndarray
        assert result._ndarray.base is None
        assert not tm.shares_memory(result, cat)

    def test_fillna_validates_with_no_nas(self):
        # We validate the fill value even if fillna is a no-op
        ci = CategoricalIndex([2, 3, 3])
        cat = ci._data

        msg = "Cannot setitem on a Categorical with a new category"
        res = ci.fillna(False)
        # nothing to fill, so we dont cast
        tm.assert_index_equal(res, ci)

        # Same check directly on the Categorical
        with pytest.raises(TypeError, match=msg):
            cat.fillna(False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_invalid.py ===
from io import StringIO

import numpy as np
import pytest

from pandas import (
    DataFrame,
    concat,
    read_csv,
)
import pandas._testing as tm


class TestInvalidConcat:
    @pytest.mark.parametrize("obj", [1, {}, [1, 2], (1, 2)])
    def test_concat_invalid(self, obj):
        # trying to concat a ndframe with a non-ndframe
        df1 = DataFrame(range(2))
        msg = (
            f"cannot concatenate object of type '{type(obj)}'; "
            "only Series and DataFrame objs are valid"
        )
        with pytest.raises(TypeError, match=msg):
            concat([df1, obj])

    def test_concat_invalid_first_argument(self):
        df1 = DataFrame(range(2))
        msg = (
            "first argument must be an iterable of pandas "
            'objects, you passed an object of type "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            concat(df1)

    def test_concat_generator_obj(self):
        # generator ok though
        concat(DataFrame(np.random.default_rng(2).random((5, 5))) for _ in range(3))

    def test_concat_textreader_obj(self):
        # text reader ok
        # GH6583
        data = """index,A,B,C,D
                  foo,2,3,4,5
                  bar,7,8,9,10
                  baz,12,13,14,15
                  qux,12,13,14,15
                  foo2,12,13,14,15
                  bar2,12,13,14,15
               """

        with read_csv(StringIO(data), chunksize=1) as reader:
            result = concat(reader, ignore_index=True)
        expected = read_csv(StringIO(data))
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_truediv.py ===
#this module tests that SymPy works with true division turned on

from sympy.core.numbers import (Float, Rational)
from sympy.core.symbol import Symbol


def test_truediv():
    assert 1/2 != 0
    assert Rational(1)/2 != 0


def dotest(s):
    x = Symbol("x")
    y = Symbol("y")
    l = [
        Rational(2),
        Float("1.3"),
        x,
        y,
        pow(x, y)*y,
        5,
        5.5
    ]
    for x in l:
        for y in l:
            s(x, y)
    return True


def test_basic():
    def s(a, b):
        x = a
        x = +a
        x = -a
        x = a + b
        x = a - b
        x = a*b
        x = a/b
        x = a**b
        del x
    assert dotest(s)


def test_ibasic():
    def s(a, b):
        x = a
        x += b
        x = a
        x -= b
        x = a
        x *= b
        x = a
        x /= b
    assert dotest(s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_funcmatrix.py ===
from sympy.core import symbols, Lambda
from sympy.core.sympify import SympifyError
from sympy.functions import KroneckerDelta
from sympy.matrices import Matrix
from sympy.matrices.expressions import FunctionMatrix, MatrixExpr, Identity
from sympy.testing.pytest import raises


def test_funcmatrix_creation():
    i, j, k = symbols('i j k')
    assert FunctionMatrix(2, 2, Lambda((i, j), 0))
    assert FunctionMatrix(0, 0, Lambda((i, j), 0))

    raises(ValueError, lambda: FunctionMatrix(-1, 0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(2.0, 0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(2j, 0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(0, -1, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(0, 2.0, Lambda((i, j), 0)))
    raises(ValueError, lambda: FunctionMatrix(0, 2j, Lambda((i, j), 0)))

    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda(i, 0)))
    raises(SympifyError, lambda: FunctionMatrix(2, 2, lambda i, j: 0))
    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda((i,), 0)))
    raises(ValueError, lambda: FunctionMatrix(2, 2, Lambda((i, j, k), 0)))
    raises(ValueError, lambda: FunctionMatrix(2, 2, i+j))
    assert FunctionMatrix(2, 2, "lambda i, j: 0") == \
        FunctionMatrix(2, 2, Lambda((i, j), 0))

    m = FunctionMatrix(2, 2, KroneckerDelta)
    assert m.as_explicit() == Identity(2).as_explicit()
    assert m.args[2].dummy_eq(Lambda((i, j), KroneckerDelta(i, j)))

    n = symbols('n')
    assert FunctionMatrix(n, n, Lambda((i, j), 0))
    n = symbols('n', integer=False)
    raises(ValueError, lambda: FunctionMatrix(n, n, Lambda((i, j), 0)))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: FunctionMatrix(n, n, Lambda((i, j), 0)))


def test_funcmatrix():
    i, j = symbols('i,j')
    X = FunctionMatrix(3, 3, Lambda((i, j), i - j))
    assert X[1, 1] == 0
    assert X[1, 2] == -1
    assert X.shape == (3, 3)
    assert X.rows == X.cols == 3
    assert Matrix(X) == Matrix(3, 3, lambda i, j: i - j)
    assert isinstance(X*X + X, MatrixExpr)


def test_replace_issue():
    X = FunctionMatrix(3, 3, KroneckerDelta)
    assert X.replace(lambda x: True, lambda x: x) == X

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_function.py ===
""" Unit tests for Hyper_Function"""
from sympy.core import symbols, Dummy, Tuple, S, Rational
from sympy.functions import hyper

from sympy.simplify.hyperexpand import Hyper_Function

def test_attrs():
    a, b = symbols('a, b', cls=Dummy)
    f = Hyper_Function([2, a], [b])
    assert f.ap == Tuple(2, a)
    assert f.bq == Tuple(b)
    assert f.args == (Tuple(2, a), Tuple(b))
    assert f.sizes == (2, 1)

def test_call():
    a, b, x = symbols('a, b, x', cls=Dummy)
    f = Hyper_Function([2, a], [b])
    assert f(x) == hyper([2, a], [b], x)

def test_has():
    a, b, c = symbols('a, b, c', cls=Dummy)
    f = Hyper_Function([2, -a], [b])
    assert f.has(a)
    assert f.has(Tuple(b))
    assert not f.has(c)

def test_eq():
    assert Hyper_Function([1], []) == Hyper_Function([1], [])
    assert (Hyper_Function([1], []) != Hyper_Function([1], [])) is False
    assert Hyper_Function([1], []) != Hyper_Function([2], [])
    assert Hyper_Function([1], []) != Hyper_Function([1, 2], [])
    assert Hyper_Function([1], []) != Hyper_Function([1], [2])

def test_gamma():
    assert Hyper_Function([2, 3], [-1]).gamma == 0
    assert Hyper_Function([-2, -3], [-1]).gamma == 2
    n = Dummy(integer=True)
    assert Hyper_Function([-1, n, 1], []).gamma == 1
    assert Hyper_Function([-1, -n, 1], []).gamma == 1
    p = Dummy(integer=True, positive=True)
    assert Hyper_Function([-1, p, 1], []).gamma == 1
    assert Hyper_Function([-1, -p, 1], []).gamma == 2

def test_suitable_origin():
    assert Hyper_Function((S.Half,), (Rational(3, 2),))._is_suitable_origin() is True
    assert Hyper_Function((S.Half,), (S.Half,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (Rational(-1, 2),))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (0,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half,), (-1, 1,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 0), (1,))._is_suitable_origin() is False
    assert Hyper_Function((S.Half, 1),
            (2, Rational(-2, 3)))._is_suitable_origin() is True
    assert Hyper_Function((S.Half, 1),
            (2, Rational(-2, 3), Rational(3, 2)))._is_suitable_origin() is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_summation.py ===
from mpmath import *

def test_sumem():
    mp.dps = 15
    assert sumem(lambda k: 1/k**2.5, [50, 100]).ae(0.0012524505324784962)
    assert sumem(lambda k: k**4 + 3*k + 1, [10, 100]).ae(2050333103)

def test_nsum():
    mp.dps = 15
    assert nsum(lambda x: x**2, [1, 3]) == 14
    assert nsum(lambda k: 1/factorial(k), [0, inf]).ae(e)
    assert nsum(lambda k: (-1)**(k+1) / k, [1, inf]).ae(log(2))
    assert nsum(lambda k: (-1)**(k+1) / k**2, [1, inf]).ae(pi**2 / 12)
    assert nsum(lambda k: (-1)**k / log(k), [2, inf]).ae(0.9242998972229388)
    assert nsum(lambda k: 1/k**2, [1, inf]).ae(pi**2 / 6)
    assert nsum(lambda k: 2**k/fac(k), [0, inf]).ae(exp(2))
    assert nsum(lambda k: 1/k**2, [4, inf], method='e').ae(0.2838229557371153)
    assert abs(fp.nsum(lambda k: 1/k**4, [1, fp.inf]) - 1.082323233711138) < 1e-5
    assert abs(fp.nsum(lambda k: 1/k**4, [1, fp.inf], method='e') - 1.082323233711138) < 1e-4

def test_nprod():
    mp.dps = 15
    assert nprod(lambda k: exp(1/k**2), [1,inf], method='r').ae(exp(pi**2/6))
    assert nprod(lambda x: x**2, [1, 3]) == 36

def test_fsum():
    mp.dps = 15
    assert fsum([]) == 0
    assert fsum([-4]) == -4
    assert fsum([2,3]) == 5
    assert fsum([1e-100,1]) == 1
    assert fsum([1,1e-100]) == 1
    assert fsum([1e100,1]) == 1e100
    assert fsum([1,1e100]) == 1e100
    assert fsum([1e-100,0]) == 1e-100
    assert fsum([1e-100,1e100,1e-100]) == 1e100
    assert fsum([2,1+1j,1]) == 4+1j
    assert fsum([2,inf,3]) == inf
    assert fsum([2,-1], absolute=1) == 3
    assert fsum([2,-1], squared=1) == 5
    assert fsum([1,1+j], squared=1) == 1+2j
    assert fsum([1,3+4j], absolute=1) == 6
    assert fsum([1,2+3j], absolute=1, squared=1) == 14
    assert isnan(fsum([inf,-inf]))
    assert fsum([inf,-inf], absolute=1) == inf
    assert fsum([inf,-inf], squared=1) == inf
    assert fsum([inf,-inf], absolute=1, squared=1) == inf
    assert iv.fsum([1,mpi(2,3)]) == mpi(3,4)

def test_fprod():
    mp.dps = 15
    assert fprod([]) == 1
    assert fprod([2,3]) == 6

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_kind.py ===
import platform
import sys

import pytest

from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
)
from numpy.f2py.crackfortran import (
    _selected_real_kind_func as selected_real_kind,
)

from . import util


class TestKind(util.F2PyTest):
    sources = [util.getpath("tests", "src", "kind", "foo.f90")]

    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1,
                        reason="Fails for 32 bit machines")
    def test_int(self):
        """Test `int` kind_func for integers up to 10**40."""
        selectedintkind = self.module.selectedintkind

        for i in range(40):
            assert selectedintkind(i) == selected_int_kind(
                i
            ), f"selectedintkind({i}): expected {selected_int_kind(i)!r} but got {selectedintkind(i)!r}"

    def test_real(self):
        """
        Test (processor-dependent) `real` kind_func for real numbers
        of up to 31 digits precision (extended/quadruple).
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"

    @pytest.mark.xfail(platform.machine().lower().startswith("ppc"),
                       reason="Some PowerPC may not support full IEEE 754 precision")
    def test_quad_precision(self):
        """
        Test kind_func for quadruple precision [`real(16)`] of 32+ digits .
        """
        selectedrealkind = self.module.selectedrealkind

        for i in range(32, 40):
            assert selectedrealkind(i) == selected_real_kind(
                i
            ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_to_numpy.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm


class TestToNumpy:
    def test_to_numpy(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})
        expected = np.array([[1, 3], [2, 4.5]])
        result = df.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_dtype(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})
        expected = np.array([[1, 3], [2, 4]], dtype="int64")
        result = df.to_numpy(dtype="int64")
        tm.assert_numpy_array_equal(result, expected)

    @td.skip_array_manager_invalid_test
    def test_to_numpy_copy(self, using_copy_on_write):
        arr = np.random.default_rng(2).standard_normal((4, 3))
        df = DataFrame(arr)
        if using_copy_on_write:
            assert df.values.base is not arr
            assert df.to_numpy(copy=False).base is df.values.base
        else:
            assert df.values.base is arr
            assert df.to_numpy(copy=False).base is arr
        assert df.to_numpy(copy=True).base is not arr

        # we still don't want a copy when na_value=np.nan is passed,
        #  and that can be respected because we are already numpy-float
        if using_copy_on_write:
            assert df.to_numpy(copy=False).base is df.values.base
        else:
            assert df.to_numpy(copy=False, na_value=np.nan).base is arr

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_to_numpy_mixed_dtype_to_str(self):
        # https://github.com/pandas-dev/pandas/issues/35455
        df = DataFrame([[Timestamp("2020-01-01 00:00:00"), 100.0]])
        result = df.to_numpy(dtype=str)
        expected = np.array([["2020-01-01 00:00:00", "100.0"]], dtype=str)
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\test_nat.py ===
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    NaT,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm


class NATests:
    def test_nat(self, index_without_na):
        empty_index = index_without_na[:0]

        index_with_na = index_without_na.copy(deep=True)
        index_with_na._data[1] = NaT

        assert empty_index._na_value is NaT
        assert index_with_na._na_value is NaT
        assert index_without_na._na_value is NaT

        idx = index_without_na
        assert idx._can_hold_na

        tm.assert_numpy_array_equal(idx._isnan, np.array([False, False]))
        assert idx.hasnans is False

        idx = index_with_na
        assert idx._can_hold_na

        tm.assert_numpy_array_equal(idx._isnan, np.array([False, True]))
        assert idx.hasnans is True


class TestDatetimeIndexNA(NATests):
    @pytest.fixture
    def index_without_na(self, tz_naive_fixture):
        tz = tz_naive_fixture
        return DatetimeIndex(["2011-01-01", "2011-01-02"], tz=tz)


class TestTimedeltaIndexNA(NATests):
    @pytest.fixture
    def index_without_na(self):
        return TimedeltaIndex(["1 days", "2 days"])


class TestPeriodIndexNA(NATests):
    @pytest.fixture
    def index_without_na(self):
        return PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_iat.py ===
import numpy as np

from pandas import (
    DataFrame,
    Series,
    period_range,
)
import pandas._testing as tm


def test_iat(float_frame):
    for i, row in enumerate(float_frame.index):
        for j, col in enumerate(float_frame.columns):
            result = float_frame.iat[i, j]
            expected = float_frame.at[row, col]
            assert result == expected


def test_iat_duplicate_columns():
    # https://github.com/pandas-dev/pandas/issues/11754
    df = DataFrame([[1, 2]], columns=["x", "x"])
    assert df.iat[0, 0] == 1


def test_iat_getitem_series_with_period_index():
    # GH#4390, iat incorrectly indexing
    index = period_range("1/1/2001", periods=10)
    ser = Series(np.random.default_rng(2).standard_normal(10), index=index)
    expected = ser[index[0]]
    result = ser.iat[0]
    assert expected == result


def test_iat_setitem_item_cache_cleared(
    indexer_ial, using_copy_on_write, warn_copy_on_write
):
    # GH#45684
    data = {"x": np.arange(8, dtype=np.int64), "y": np.int64(0)}
    df = DataFrame(data).copy()
    ser = df["y"]

    # previously this iat setting would split the block and fail to clear
    #  the item_cache.
    with tm.assert_cow_warning(warn_copy_on_write):
        indexer_ial(df)[7, 0] = 9999

    with tm.assert_cow_warning(warn_copy_on_write):
        indexer_ial(df)[7, 1] = 1234

    assert df.iat[7, 1] == 1234
    if not using_copy_on_write:
        assert ser.iloc[-1] == 1234
    assert df.iloc[-1, -1] == 1234

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\test_get_dummies.py ===
import numpy as np

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)


def test_get_dummies(any_string_dtype):
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)
    result = s.str.get_dummies("|")
    expected = DataFrame([[1, 1, 0], [1, 0, 1], [0, 0, 0]], columns=list("abc"))
    tm.assert_frame_equal(result, expected)

    s = Series(["a;b", "a", 7], dtype=any_string_dtype)
    result = s.str.get_dummies(";")
    expected = DataFrame([[0, 1, 1], [0, 1, 0], [1, 0, 0]], columns=list("7ab"))
    tm.assert_frame_equal(result, expected)


def test_get_dummies_index():
    # GH9980, GH8028
    idx = Index(["a|b", "a|c", "b|c"])
    result = idx.str.get_dummies("|")

    expected = MultiIndex.from_tuples(
        [(1, 1, 0), (1, 0, 1), (0, 1, 1)], names=("a", "b", "c")
    )
    tm.assert_index_equal(result, expected)


def test_get_dummies_with_name_dummy(any_string_dtype):
    # GH 12180
    # Dummies named 'name' should work as expected
    s = Series(["a", "b,name", "b"], dtype=any_string_dtype)
    result = s.str.get_dummies(",")
    expected = DataFrame([[1, 0, 0], [0, 1, 1], [0, 1, 0]], columns=["a", "b", "name"])
    tm.assert_frame_equal(result, expected)


def test_get_dummies_with_name_dummy_index():
    # GH 12180
    # Dummies named 'name' should work as expected
    idx = Index(["a|b", "name|c", "b|name"])
    result = idx.str.get_dummies("|")

    expected = MultiIndex.from_tuples(
        [(1, 1, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1)], names=("a", "b", "c", "name")
    )
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_approximations.py ===
import math
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.codegen.rewriting import optimize
from sympy.codegen.approximations import SumApprox, SeriesApprox


def test_SumApprox_trivial():
    x = symbols('x')
    expr1 = 1 + x
    sum_approx = SumApprox(bounds={x: (-1e-20, 1e-20)}, reltol=1e-16)
    apx1 = optimize(expr1, [sum_approx])
    assert apx1 - 1 == 0


def test_SumApprox_monotone_terms():
    x, y, z = symbols('x y z')
    expr1 = exp(z)*(x**2 + y**2 + 1)
    bnds1 = {x: (0, 1e-3), y: (100, 1000)}
    sum_approx_m2 = SumApprox(bounds=bnds1, reltol=1e-2)
    sum_approx_m5 = SumApprox(bounds=bnds1, reltol=1e-5)
    sum_approx_m11 = SumApprox(bounds=bnds1, reltol=1e-11)
    assert (optimize(expr1, [sum_approx_m2])/exp(z) - (y**2)).simplify() == 0
    assert (optimize(expr1, [sum_approx_m5])/exp(z) - (y**2 + 1)).simplify() == 0
    assert (optimize(expr1, [sum_approx_m11])/exp(z) - (y**2 + 1 + x**2)).simplify() == 0


def test_SeriesApprox_trivial():
    x, z = symbols('x z')
    for factor in [1, exp(z)]:
        x = symbols('x')
        expr1 = exp(x)*factor
        bnds1 = {x: (-1, 1)}
        series_approx_50 = SeriesApprox(bounds=bnds1, reltol=0.50)
        series_approx_10 = SeriesApprox(bounds=bnds1, reltol=0.10)
        series_approx_05 = SeriesApprox(bounds=bnds1, reltol=0.05)
        c = (bnds1[x][1] + bnds1[x][0])/2  # 0.0
        f0 = math.exp(c)  # 1.0

        ref_50 = f0 + x + x**2/2
        ref_10 = f0 + x + x**2/2 + x**3/6
        ref_05 = f0 + x + x**2/2 + x**3/6 + x**4/24

        res_50 = optimize(expr1, [series_approx_50])
        res_10 = optimize(expr1, [series_approx_10])
        res_05 = optimize(expr1, [series_approx_05])

        assert (res_50/factor - ref_50).simplify() == 0
        assert (res_10/factor - ref_10).simplify() == 0
        assert (res_05/factor - ref_05).simplify() == 0

        max_ord3 = SeriesApprox(bounds=bnds1, reltol=0.05, max_order=3)
        assert optimize(expr1, [max_ord3]) == expr1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\branch\tests\test_traverse.py ===
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.strategies.branch.traverse import top_down, sall
from sympy.strategies.branch.core import do_one, identity


def inc(x):
    if isinstance(x, Integer):
        yield x + 1


def test_top_down_easy():
    expr = Basic(S(1), S(2))
    expected = Basic(S(2), S(3))
    brl = top_down(inc)

    assert set(brl(expr)) == {expected}


def test_top_down_big_tree():
    expr = Basic(S(1), Basic(S(2)), Basic(S(3), Basic(S(4)), S(5)))
    expected = Basic(S(2), Basic(S(3)), Basic(S(4), Basic(S(5)), S(6)))
    brl = top_down(inc)

    assert set(brl(expr)) == {expected}


def test_top_down_harder_function():
    def split5(x):
        if x == 5:
            yield x - 1
            yield x + 1

    expr = Basic(Basic(S(5), S(6)), S(1))
    expected = {Basic(Basic(S(4), S(6)), S(1)), Basic(Basic(S(6), S(6)), S(1))}
    brl = top_down(split5)

    assert set(brl(expr)) == expected


def test_sall():
    expr = Basic(S(1), S(2))
    expected = Basic(S(2), S(3))
    brl = sall(inc)

    assert list(brl(expr)) == [expected]

    expr = Basic(S(1), S(2), Basic(S(3), S(4)))
    expected = Basic(S(2), S(3), Basic(S(3), S(4)))
    brl = sall(do_one(inc, identity))

    assert list(brl(expr)) == [expected]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_tools.py ===
import numpy as np
import pytest

from pandas import (
    Period,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestPeriodRepresentation:
    """
    Wish to match NumPy units
    """

    @pytest.mark.parametrize(
        "freq, base_date",
        [
            ("W-THU", "1970-01-01"),
            ("D", "1970-01-01"),
            ("B", "1970-01-01"),
            ("h", "1970-01-01"),
            ("min", "1970-01-01"),
            ("s", "1970-01-01"),
            ("ms", "1970-01-01"),
            ("us", "1970-01-01"),
            ("ns", "1970-01-01"),
            ("M", "1970-01"),
            ("Y", 1970),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_freq(self, freq, base_date):
        rng = period_range(start=base_date, periods=10, freq=freq)
        exp = np.arange(10, dtype=np.int64)

        tm.assert_numpy_array_equal(rng.asi8, exp)


class TestPeriodIndexConversion:
    def test_tolist(self):
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        rs = index.tolist()
        for x in rs:
            assert isinstance(x, Period)

        recon = PeriodIndex(rs)
        tm.assert_index_equal(index, recon)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_subclass.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm

from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)

pytest.importorskip("tables")


class TestHDFStoreSubclass:
    # GH 33748
    def test_supported_for_subclass_dataframe(self, tmp_path):
        data = {"a": [1, 2], "b": [3, 4]}
        sdf = tm.SubclassedDataFrame(data, dtype=np.intp)

        expected = DataFrame(data, dtype=np.intp)

        path = tmp_path / "temp.h5"
        sdf.to_hdf(path, key="df")
        result = read_hdf(path, "df")
        tm.assert_frame_equal(result, expected)

        path = tmp_path / "temp.h5"
        with HDFStore(path) as store:
            store.put("df", sdf)
        result = read_hdf(path, "df")
        tm.assert_frame_equal(result, expected)

    def test_supported_for_subclass_series(self, tmp_path):
        data = [1, 2, 3]
        sser = tm.SubclassedSeries(data, dtype=np.intp)

        expected = Series(data, dtype=np.intp)

        path = tmp_path / "temp.h5"
        sser.to_hdf(path, key="ser")
        result = read_hdf(path, "ser")
        tm.assert_series_equal(result, expected)

        path = tmp_path / "temp.h5"
        with HDFStore(path) as store:
            store.put("ser", sser)
        result = read_hdf(path, "ser")
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_qft.py ===
from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix

from sympy.physics.quantum.qft import QFT, IQFT, RkGate
from sympy.physics.quantum.gate import (ZGate, SwapGate, HadamardGate, CGate,
                                        PhaseGate, TGate)
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent

from sympy.functions.elementary.complexes import sign


def test_RkGate():
    x = Symbol('x')
    assert RkGate(1, x).k == x
    assert RkGate(1, x).targets == (1,)
    assert RkGate(1, 1) == ZGate(1)
    assert RkGate(2, 2) == PhaseGate(2)
    assert RkGate(3, 3) == TGate(3)

    assert represent(
        RkGate(0, x), nqubits=1) == Matrix([[1, 0], [0, exp(sign(x)*2*pi*I/(2**abs(x)))]])


def test_quantum_fourier():
    assert QFT(0, 3).decompose() == \
        SwapGate(0, 2)*HadamardGate(0)*CGate((0,), PhaseGate(1)) * \
        HadamardGate(1)*CGate((0,), TGate(2))*CGate((1,), PhaseGate(2)) * \
        HadamardGate(2)

    assert IQFT(0, 3).decompose() == \
        HadamardGate(2)*CGate((1,), RkGate(2, -2))*CGate((0,), RkGate(2, -3)) * \
        HadamardGate(1)*CGate((0,), RkGate(1, -2))*HadamardGate(0)*SwapGate(0, 2)

    assert represent(QFT(0, 3), nqubits=3) == \
        Matrix([[exp(2*pi*I/8)**(i*j % 8)/sqrt(8) for i in range(8)] for j in range(8)])

    assert QFT(0, 4).decompose()  # non-trivial decomposition
    assert qapply(QFT(0, 3).decompose()*Qubit(0, 0, 0)).expand() == qapply(
        HadamardGate(0)*HadamardGate(1)*HadamardGate(2)*Qubit(0, 0, 0)
    ).expand()


def test_qft_represent():
    c = QFT(0, 3)
    a = represent(c, nqubits=3)
    b = represent(c.decompose(), nqubits=3)
    assert a.evalf(n=10) == b.evalf(n=10)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\tests\test_quotientring.py ===
"""Tests for quotient rings."""

from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y

from sympy.polys.polyerrors import NotReversible

from sympy.testing.pytest import raises


def test_QuotientRingElement():
    R = QQ.old_poly_ring(x)/[x**10]
    X = R.convert(x)

    assert X*(X + 1) == R.convert(x**2 + x)
    assert X*x == R.convert(x**2)
    assert x*X == R.convert(x**2)
    assert X + x == R.convert(2*x)
    assert x + X == 2*X
    assert X**2 == R.convert(x**2)
    assert 1/(1 - X) == R.convert(sum(x**i for i in range(10)))
    assert X**10 == R.zero
    assert X != x

    raises(NotReversible, lambda: 1/X)


def test_QuotientRing():
    I = QQ.old_poly_ring(x).ideal(x**2 + 1)
    R = QQ.old_poly_ring(x)/I

    assert R == QQ.old_poly_ring(x)/[x**2 + 1]
    assert R == QQ.old_poly_ring(x)/QQ.old_poly_ring(x).ideal(x**2 + 1)
    assert R != QQ.old_poly_ring(x)

    assert R.convert(1)/x == -x + I
    assert -1 + I == x**2 + I
    assert R.convert(ZZ(1), ZZ) == 1 + I
    assert R.convert(R.convert(x), R) == R.convert(x)

    X = R.convert(x)
    Y = QQ.old_poly_ring(x).convert(x)
    assert -1 + I == X**2 + I
    assert -1 + I == Y**2 + I
    assert R.to_sympy(X) == x

    raises(ValueError, lambda: QQ.old_poly_ring(x)/QQ.old_poly_ring(x, y).ideal(x))

    R = QQ.old_poly_ring(x, order="ilex")
    I = R.ideal(x)
    assert R.convert(1) + I == (R/I).convert(1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\tests\test_contains.py ===
from sympy.core.expr import unchanged
from sympy.core.numbers import oo
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.testing.pytest import raises


def test_contains_basic():
    raises(TypeError, lambda: Contains(S.Integers, 1))
    assert Contains(2, S.Integers) is S.true
    assert Contains(-2, S.Naturals) is S.false

    i = Symbol('i', integer=True)
    assert Contains(i, S.Naturals) == Contains(i, S.Naturals, evaluate=False)


def test_issue_6194():
    x = Symbol('x')
    assert unchanged(Contains, x, Interval(0, 1))
    assert Interval(0, 1).contains(x) == (S.Zero <= x) & (x <= 1)
    assert Contains(x, FiniteSet(0)) != S.false
    assert Contains(x, Interval(1, 1)) != S.false
    assert Contains(x, S.Integers) != S.false


def test_issue_10326():
    assert Contains(oo, Interval(-oo, oo)) == False
    assert Contains(-oo, Interval(-oo, oo)) == False


def test_binary_symbols():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert Contains(x, FiniteSet(y, Eq(z, True))
        ).binary_symbols == {y, z}


def test_as_set():
    x = Symbol('x')
    y = Symbol('y')
    assert Contains(x, FiniteSet(y)).as_set() == FiniteSet(y)
    assert Contains(x, S.Integers).as_set() == S.Integers
    assert Contains(x, S.Reals).as_set() == S.Reals


def test_type_error():
    # Pass in a parameter not of type "set"
    raises(TypeError, lambda: Contains(2, None))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\tests\test_array_derivatives.py ===
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.array.ndim_array import NDimArray
from sympy.matrices.matrixbase import MatrixBase
from sympy.tensor.array.array_derivatives import ArrayDerivative

x, y, z, t = symbols("x y z t")

m = Matrix([[x, y], [z, t]])

M = MatrixSymbol("M", 3, 2)
N = MatrixSymbol("N", 4, 3)


def test_array_derivative_construction():

    d = ArrayDerivative(x, m, evaluate=False)
    assert d.shape == (2, 2)
    expr = d.doit()
    assert isinstance(expr, MatrixBase)
    assert expr.shape == (2, 2)

    d = ArrayDerivative(m, m, evaluate=False)
    assert d.shape == (2, 2, 2, 2)
    expr = d.doit()
    assert isinstance(expr, NDimArray)
    assert expr.shape == (2, 2, 2, 2)

    d = ArrayDerivative(m, x, evaluate=False)
    assert d.shape == (2, 2)
    expr = d.doit()
    assert isinstance(expr, MatrixBase)
    assert expr.shape == (2, 2)

    d = ArrayDerivative(M, N, evaluate=False)
    assert d.shape == (4, 3, 3, 2)
    expr = d.doit()
    assert isinstance(expr, ArrayDerivative)
    assert expr.shape == (4, 3, 3, 2)

    d = ArrayDerivative(M, (N, 2), evaluate=False)
    assert d.shape == (4, 3, 4, 3, 3, 2)
    expr = d.doit()
    assert isinstance(expr, ArrayDerivative)
    assert expr.shape == (4, 3, 4, 3, 3, 2)

    d = ArrayDerivative(M.as_explicit(), (N.as_explicit(), 2), evaluate=False)
    assert d.doit().shape == (4, 3, 4, 3, 3, 2)
    expr = d.doit()
    assert isinstance(expr, NDimArray)
    assert expr.shape == (4, 3, 4, 3, 3, 2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\literal.py ===
from __future__ import annotations

from typing import Any, TYPE_CHECKING
from functools import partial

import pytest
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

AR = np.array(0)
AR.setflags(write=False)

KACF = frozenset({None, "K", "A", "C", "F"})
ACF = frozenset({None, "A", "C", "F"})
CF = frozenset({None, "C", "F"})

order_list: list[tuple[frozenset[str | None], Callable[..., Any]]] = [
    (KACF, AR.tobytes),
    (KACF, partial(AR.astype, int)),
    (KACF, AR.copy),
    (ACF, partial(AR.reshape, 1)),
    (KACF, AR.flatten),
    (KACF, AR.ravel),
    (KACF, partial(np.array, 1)),
    # NOTE: __call__ is needed due to mypy bugs (#17620, #17631)
    (KACF, partial(np.ndarray.__call__, 1)),
    (CF, partial(np.zeros.__call__, 1)),
    (CF, partial(np.ones.__call__, 1)),
    (CF, partial(np.empty.__call__, 1)),
    (CF, partial(np.full, 1, 1)),
    (KACF, partial(np.zeros_like, AR)),
    (KACF, partial(np.ones_like, AR)),
    (KACF, partial(np.empty_like, AR)),
    (KACF, partial(np.full_like, AR, 1)),
    (KACF, partial(np.add.__call__, 1, 1)),  # i.e. np.ufunc.__call__
    (ACF, partial(np.reshape, AR, 1)),
    (KACF, partial(np.ravel, AR)),
    (KACF, partial(np.asarray, 1)),
    (KACF, partial(np.asanyarray, 1)),
]

for order_set, func in order_list:
    for order in order_set:
        func(order=order)

    invalid_orders = KACF - order_set
    for order in invalid_orders:
        with pytest.raises(ValueError):
            func(order=order)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\test_concat.py ===
import pytest

import pandas.core.dtypes.concat as _concat

import pandas as pd
from pandas import Series
import pandas._testing as tm


def test_concat_mismatched_categoricals_with_empty():
    # concat_compat behavior on series._values should match pd.concat on series
    ser1 = Series(["a", "b", "c"], dtype="category")
    ser2 = Series([], dtype="category")

    msg = "The behavior of array concatenation with empty entries is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = _concat.concat_compat([ser1._values, ser2._values])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = pd.concat([ser1, ser2])._values
    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize("copy", [True, False])
def test_concat_single_dataframe_tz_aware(copy):
    # https://github.com/pandas-dev/pandas/issues/25257
    df = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2020-04-08 09:00:00.709949+0000", tz="UTC")]}
    )
    expected = df.copy()
    result = pd.concat([df], copy=copy)
    tm.assert_frame_equal(result, expected)


def test_concat_periodarray_2d():
    pi = pd.period_range("2016-01-01", periods=36, freq="D")
    arr = pi._data.reshape(6, 6)

    result = _concat.concat_compat([arr[:2], arr[2:]], axis=0)
    tm.assert_period_array_equal(result, arr)

    result = _concat.concat_compat([arr[:, :2], arr[:, 2:]], axis=1)
    tm.assert_period_array_equal(result, arr)

    msg = (
        "all the input array dimensions.* for the concatenation axis must match exactly"
    )
    with pytest.raises(ValueError, match=msg):
        _concat.concat_compat([arr[:, :2], arr[:, 2:]], axis=0)

    with pytest.raises(ValueError, match=msg):
        _concat.concat_compat([arr[:2], arr[2:]], axis=1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_to_pydatetime.py ===
from datetime import (
    datetime,
    timezone,
)

import dateutil.parser
import dateutil.tz
from dateutil.tz import tzlocal
import numpy as np

from pandas import (
    DatetimeIndex,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.tests.indexes.datetimes.test_timezones import FixedOffset

fixed_off = FixedOffset(-420, "-07:00")


class TestToPyDatetime:
    def test_dti_to_pydatetime(self):
        dt = dateutil.parser.parse("2012-06-13T01:39:00Z")
        dt = dt.replace(tzinfo=tzlocal())

        arr = np.array([dt], dtype=object)

        result = to_datetime(arr, utc=True)
        assert result.tz is timezone.utc

        rng = date_range("2012-11-03 03:00", "2012-11-05 03:00", tz=tzlocal())
        arr = rng.to_pydatetime()
        result = to_datetime(arr, utc=True)
        assert result.tz is timezone.utc

    def test_dti_to_pydatetime_fizedtz(self):
        dates = np.array(
            [
                datetime(2000, 1, 1, tzinfo=fixed_off),
                datetime(2000, 1, 2, tzinfo=fixed_off),
                datetime(2000, 1, 3, tzinfo=fixed_off),
            ]
        )
        dti = DatetimeIndex(dates)

        result = dti.to_pydatetime()
        tm.assert_numpy_array_equal(dates, result)

        result = dti._mpl_repr()
        tm.assert_numpy_array_equal(dates, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_arithmetic.py ===
# Arithmetic tests for TimedeltaIndex are generally about the result's `freq` attribute.
# Other cases can be shared in tests.arithmetic.test_timedelta64
import numpy as np

from pandas import (
    NaT,
    Timedelta,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexArithmetic:
    def test_arithmetic_zero_freq(self):
        # GH#51575 don't get a .freq with freq.n = 0
        tdi = timedelta_range(0, periods=100, freq="ns")
        result = tdi / 2
        assert result.freq is None
        expected = tdi[:50].repeat(2)
        tm.assert_index_equal(result, expected)

        result2 = tdi // 2
        assert result2.freq is None
        expected2 = expected
        tm.assert_index_equal(result2, expected2)

        result3 = tdi * 0
        assert result3.freq is None
        expected3 = tdi[:1].repeat(100)
        tm.assert_index_equal(result3, expected3)

    def test_tdi_division(self, index_or_series):
        # doc example

        scalar = Timedelta(days=31)
        td = index_or_series(
            [scalar, scalar, scalar + Timedelta(minutes=5, seconds=3), NaT],
            dtype="m8[ns]",
        )

        result = td / np.timedelta64(1, "D")
        expected = index_or_series(
            [31, 31, (31 * 86400 + 5 * 60 + 3) / 86400.0, np.nan]
        )
        tm.assert_equal(result, expected)

        result = td / np.timedelta64(1, "s")
        expected = index_or_series(
            [31 * 86400, 31 * 86400, 31 * 86400 + 5 * 60 + 3, np.nan]
        )
        tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\interval\test_constructors.py ===
import pytest

from pandas import (
    Interval,
    Period,
    Timestamp,
)


class TestIntervalConstructors:
    @pytest.mark.parametrize(
        "left, right",
        [
            ("a", "z"),
            (("a", "b"), ("c", "d")),
            (list("AB"), list("ab")),
            (Interval(0, 1), Interval(1, 2)),
            (Period("2018Q1", freq="Q"), Period("2018Q1", freq="Q")),
        ],
    )
    def test_construct_errors(self, left, right):
        # GH#23013
        msg = "Only numeric, Timestamp and Timedelta endpoints are allowed"
        with pytest.raises(ValueError, match=msg):
            Interval(left, right)

    def test_constructor_errors(self):
        msg = "invalid option for 'closed': foo"
        with pytest.raises(ValueError, match=msg):
            Interval(0, 1, closed="foo")

        msg = "left side of interval must be <= right side"
        with pytest.raises(ValueError, match=msg):
            Interval(1, 0)

    @pytest.mark.parametrize(
        "tz_left, tz_right", [(None, "UTC"), ("UTC", None), ("UTC", "US/Eastern")]
    )
    def test_constructor_errors_tz(self, tz_left, tz_right):
        # GH#18538
        left = Timestamp("2017-01-01", tz=tz_left)
        right = Timestamp("2017-01-02", tz=tz_right)

        if tz_left is None or tz_right is None:
            error = TypeError
            msg = "Cannot compare tz-naive and tz-aware timestamps"
        else:
            error = ValueError
            msg = "left and right must have the same time zone"
        with pytest.raises(error, match=msg):
            Interval(left, right)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_tz_convert.py ===
import dateutil
import pytest

from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td

from pandas import Timestamp


class TestTimestampTZConvert:
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_astimezone(self, tzstr):
        # astimezone is an alias for tz_convert, so keep it with
        # the tz_convert tests
        utcdate = Timestamp("3/11/2012 22:00", tz="UTC")
        expected = utcdate.tz_convert(tzstr)
        result = utcdate.astimezone(tzstr)
        assert expected == result
        assert isinstance(result, Timestamp)

    @pytest.mark.parametrize(
        "stamp",
        [
            "2014-02-01 09:00",
            "2014-07-08 09:00",
            "2014-11-01 17:00",
            "2014-11-05 00:00",
        ],
    )
    def test_tz_convert_roundtrip(self, stamp, tz_aware_fixture):
        tz = tz_aware_fixture

        ts = Timestamp(stamp, tz="UTC")
        converted = ts.tz_convert(tz)

        reset = converted.tz_convert(None)
        assert reset == Timestamp(stamp)
        assert reset.tzinfo is None
        assert reset == converted.tz_convert("UTC").tz_localize(None)

    @td.skip_if_windows
    def test_tz_convert_utc_with_system_utc(self):
        # from system utc to real utc
        ts = Timestamp("2001-01-05 11:56", tz=timezones.maybe_get_tz("dateutil/UTC"))
        # check that the time hasn't changed.
        assert ts == ts.tz_convert(dateutil.tz.tzutc())

        # from system utc to real utc
        ts = Timestamp("2001-01-05 11:56", tz=timezones.maybe_get_tz("dateutil/UTC"))
        # check that the time hasn't changed.
        assert ts == ts.tz_convert(dateutil.tz.tzutc())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_assumed_shape.py ===
import os
import tempfile

import pytest

from . import util


class TestAssumedShapeSumExample(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "assumed_shape", "foo_free.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_use.f90"),
        util.getpath("tests", "src", "assumed_shape", "precision.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_mod.f90"),
        util.getpath("tests", "src", "assumed_shape", ".f2py_f2cmap"),
    ]

    @pytest.mark.slow
    def test_all(self):
        r = self.module.fsum([1, 2])
        assert r == 3
        r = self.module.sum([1, 2])
        assert r == 3
        r = self.module.sum_with_use([1, 2])
        assert r == 3

        r = self.module.mod.sum([1, 2])
        assert r == 3
        r = self.module.mod.fsum([1, 2])
        assert r == 3


class TestF2cmapOption(TestAssumedShapeSumExample):
    def setup_method(self):
        # Use a custom file name for .f2py_f2cmap
        self.sources = list(self.sources)
        f2cmap_src = self.sources.pop(-1)

        self.f2cmap_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f2cmap_src, "rb") as f:
            self.f2cmap_file.write(f.read())
        self.f2cmap_file.close()

        self.sources.append(self.f2cmap_file.name)
        self.options = ["--f2cmap", self.f2cmap_file.name]

        super().setup_method()

    def teardown_method(self):
        os.unlink(self.f2cmap_file.name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_isetitem.py ===
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameSetItem:
    def test_isetitem_ea_df(self):
        # GH#49922
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        rhs = DataFrame([[11, 12], [13, 14]], dtype="Int64")

        df.isetitem([0, 1], rhs)
        expected = DataFrame(
            {
                0: Series([11, 13], dtype="Int64"),
                1: Series([12, 14], dtype="Int64"),
                2: [3, 6],
            }
        )
        tm.assert_frame_equal(df, expected)

    def test_isetitem_ea_df_scalar_indexer(self):
        # GH#49922
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        rhs = DataFrame([[11], [13]], dtype="Int64")

        df.isetitem(2, rhs)
        expected = DataFrame(
            {
                0: [1, 4],
                1: [2, 5],
                2: Series([11, 13], dtype="Int64"),
            }
        )
        tm.assert_frame_equal(df, expected)

    def test_isetitem_dimension_mismatch(self):
        # GH#51701
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        value = df.copy()
        with pytest.raises(ValueError, match="Got 2 positions but value has 3 columns"):
            df.isetitem([1, 2], value)

        value = df.copy()
        with pytest.raises(ValueError, match="Got 2 positions but value has 1 columns"):
            df.isetitem([1, 2], value[["a"]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_datetime.py ===
from datetime import datetime

import numpy as np

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    period_range,
    to_datetime,
)
import pandas._testing as tm


def test_multiindex_period_datetime():
    # GH4861, using datetime in period of multiindex raises exception

    idx1 = Index(["a", "a", "a", "b", "b"])
    idx2 = period_range("2012-01", periods=len(idx1), freq="M")
    s = Series(np.random.default_rng(2).standard_normal(len(idx1)), [idx1, idx2])

    # try Period as index
    expected = s.iloc[0]
    result = s.loc["a", Period("2012-01")]
    assert result == expected

    # try datetime as index
    result = s.loc["a", datetime(2012, 1, 1)]
    assert result == expected


def test_multiindex_datetime_columns():
    # GH35015, using datetime as column indices raises exception

    mi = MultiIndex.from_tuples(
        [(to_datetime("02/29/2020"), to_datetime("03/01/2020"))], names=["a", "b"]
    )

    df = DataFrame([], columns=mi)

    expected_df = DataFrame(
        [],
        columns=MultiIndex.from_arrays(
            [[to_datetime("02/29/2020")], [to_datetime("03/01/2020")]], names=["a", "b"]
        ),
    )

    tm.assert_frame_equal(df, expected_df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\common.py ===
from collections.abc import Generator
from contextlib import contextmanager
import pathlib
import tempfile

import pytest

from pandas.io.pytables import HDFStore

tables = pytest.importorskip("tables")
# set these parameters so we don't have file sharing
tables.parameters.MAX_NUMEXPR_THREADS = 1
tables.parameters.MAX_BLOSC_THREADS = 1
tables.parameters.MAX_THREADS = 1


def safe_close(store):
    try:
        if store is not None:
            store.close()
    except OSError:
        pass


# contextmanager to ensure the file cleanup
@contextmanager
def ensure_clean_store(
    path, mode="a", complevel=None, complib=None, fletcher32=False
) -> Generator[HDFStore, None, None]:
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname, path)
        with HDFStore(
            tmp_path,
            mode=mode,
            complevel=complevel,
            complib=complib,
            fletcher32=fletcher32,
        ) as store:
            yield store


def _maybe_remove(store, key):
    """
    For tests using tables, try removing the table to be sure there is
    no content from previous tests using the same table name.
    """
    try:
        store.remove(key)
    except (ValueError, KeyError):
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_unary.py ===
import pytest

from pandas import Series
import pandas._testing as tm


class TestSeriesUnaryOps:
    # __neg__, __pos__, __invert__

    def test_neg(self):
        ser = Series(range(5), dtype="float64", name="series")
        tm.assert_series_equal(-ser, -1 * ser)

    def test_invert(self):
        ser = Series(range(5), dtype="float64", name="series")
        tm.assert_series_equal(-(ser < 0), ~(ser < 0))

    @pytest.mark.parametrize(
        "source, neg_target, abs_target",
        [
            ([1, 2, 3], [-1, -2, -3], [1, 2, 3]),
            ([1, 2, None], [-1, -2, None], [1, 2, None]),
        ],
    )
    def test_all_numeric_unary_operators(
        self, any_numeric_ea_dtype, source, neg_target, abs_target
    ):
        # GH38794
        dtype = any_numeric_ea_dtype
        ser = Series(source, dtype=dtype)
        neg_result, pos_result, abs_result = -ser, +ser, abs(ser)
        if dtype.startswith("U"):
            neg_target = -Series(source, dtype=dtype)
        else:
            neg_target = Series(neg_target, dtype=dtype)

        abs_target = Series(abs_target, dtype=dtype)

        tm.assert_series_equal(neg_result, neg_target)
        tm.assert_series_equal(pos_result, ser)
        tm.assert_series_equal(abs_result, abs_target)

    @pytest.mark.parametrize("op", ["__neg__", "__abs__"])
    def test_unary_float_op_mask(self, float_ea_dtype, op):
        dtype = float_ea_dtype
        ser = Series([1.1, 2.2, 3.3], dtype=dtype)
        result = getattr(ser, op)()
        target = result.copy(deep=True)
        ser[0] = None
        tm.assert_series_equal(result, target)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_take.py ===
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm


def test_take_validate_axis():
    # GH#51022
    ser = Series([-1, 5, 6, 2, 4])

    msg = "No axis named foo for object type Series"
    with pytest.raises(ValueError, match=msg):
        ser.take([1, 2], axis="foo")


def test_take():
    ser = Series([-1, 5, 6, 2, 4])

    actual = ser.take([1, 3, 4])
    expected = Series([5, 2, 4], index=[1, 3, 4])
    tm.assert_series_equal(actual, expected)

    actual = ser.take([-1, 3, 4])
    expected = Series([4, 2, 4], index=[4, 3, 4])
    tm.assert_series_equal(actual, expected)

    msg = "indices are out-of-bounds"
    with pytest.raises(IndexError, match=msg):
        ser.take([1, 10])
    with pytest.raises(IndexError, match=msg):
        ser.take([2, 5])


def test_take_categorical():
    # https://github.com/pandas-dev/pandas/issues/20664
    ser = Series(pd.Categorical(["a", "b", "c"]))
    result = ser.take([-2, -2, 0])
    expected = Series(
        pd.Categorical(["b", "b", "a"], categories=["a", "b", "c"]), index=[1, 1, 0]
    )
    tm.assert_series_equal(result, expected)


def test_take_slice_raises():
    ser = Series([-1, 5, 6, 2, 4])

    msg = "Series.take requires a sequence of integers, not slice"
    with pytest.raises(TypeError, match=msg):
        ser.take(slice(0, 3, 1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_sathandlers.py ===
from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.logic.boolalg import (And, Or)

from sympy.assumptions.sathandlers import (ClassFactRegistry, allargs,
    anyarg, exactlyonearg,)

x, y, z = symbols('x y z')


def test_class_handler_registry():
    my_handler_registry = ClassFactRegistry()

    # The predicate doesn't matter here, so just pass
    @my_handler_registry.register(Mul)
    def fact1(expr):
        pass
    @my_handler_registry.multiregister(Expr)
    def fact2(expr):
        pass

    assert my_handler_registry[Basic] == (frozenset(), frozenset())
    assert my_handler_registry[Expr] == (frozenset(), frozenset({fact2}))
    assert my_handler_registry[Mul] == (frozenset({fact1}), frozenset({fact2}))


def test_allargs():
    assert allargs(x, Q.zero(x), x*y) == And(Q.zero(x), Q.zero(y))
    assert allargs(x, Q.positive(x) | Q.negative(x), x*y) == And(Q.positive(x) | Q.negative(x), Q.positive(y) | Q.negative(y))


def test_anyarg():
    assert anyarg(x, Q.zero(x), x*y) == Or(Q.zero(x), Q.zero(y))
    assert anyarg(x, Q.positive(x) & Q.negative(x), x*y) == \
        Or(Q.positive(x) & Q.negative(x), Q.positive(y) & Q.negative(y))


def test_exactlyonearg():
    assert exactlyonearg(x, Q.zero(x), x*y) == \
        Or(Q.zero(x) & ~Q.zero(y), Q.zero(y) & ~Q.zero(x))
    assert exactlyonearg(x, Q.zero(x), x*y*z) == \
        Or(Q.zero(x) & ~Q.zero(y) & ~Q.zero(z), Q.zero(y)
        & ~Q.zero(x) & ~Q.zero(z), Q.zero(z) & ~Q.zero(x) & ~Q.zero(y))
    assert exactlyonearg(x, Q.positive(x) | Q.negative(x), x*y) == \
        Or((Q.positive(x) | Q.negative(x)) &
        ~(Q.positive(y) | Q.negative(y)), (Q.positive(y) | Q.negative(y)) &
        ~(Q.positive(x) | Q.negative(x)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_matrix_nodes.py ===
from sympy.core.symbol import symbols
from sympy.core.function import Function
from sympy.matrices.dense import Matrix
from sympy.matrices.dense import zeros
from sympy.simplify.simplify import simplify
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.utilities.lambdify import lambdify
from sympy.printing.numpy import NumPyPrinter
from sympy.testing.pytest import skip
from sympy.external import import_module


def test_matrix_solve_issue_24862():
    A = Matrix(3, 3, symbols('a:9'))
    b = Matrix(3, 1, symbols('b:3'))
    hash(MatrixSolve(A, b))


def test_matrix_solve_derivative_exact():
    q = symbols('q')
    a11, a12, a21, a22, b1, b2 = (
        f(q) for f in symbols('a11 a12 a21 a22 b1 b2', cls=Function))
    A = Matrix([[a11, a12], [a21, a22]])
    b = Matrix([b1, b2])
    x_lu = A.LUsolve(b)
    dxdq_lu = A.LUsolve(b.diff(q) - A.diff(q) * A.LUsolve(b))
    assert simplify(x_lu.diff(q) - dxdq_lu) == zeros(2, 1)
    # dxdq_ms is the MatrixSolve equivalent of dxdq_lu
    dxdq_ms = MatrixSolve(A, b.diff(q) - A.diff(q) * MatrixSolve(A, b))
    assert MatrixSolve(A, b).diff(q) == dxdq_ms


def test_matrix_solve_derivative_numpy():
    np = import_module('numpy')
    if not np:
        skip("numpy not installed.")
    q = symbols('q')
    a11, a12, a21, a22, b1, b2 = (
        f(q) for f in symbols('a11 a12 a21 a22 b1 b2', cls=Function))
    A = Matrix([[a11, a12], [a21, a22]])
    b = Matrix([b1, b2])
    dx_lu = A.LUsolve(b).diff(q)
    subs = {a11.diff(q): 0.2, a12.diff(q): 0.3, a21.diff(q): 0.1,
            a22.diff(q): 0.5, b1.diff(q): 0.4, b2.diff(q): 0.9,
            a11: 1.3, a12: 0.5, a21: 1.2, a22: 4, b1: 6.2, b2: 3.5}
    p, p_vals = zip(*subs.items())
    dx_sm = MatrixSolve(A, b).diff(q)
    np.testing.assert_allclose(
        lambdify(p, dx_sm, printer=NumPyPrinter)(*p_vals),
        lambdify(p, dx_lu, printer=NumPyPrinter)(*p_vals))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_maxima.py ===
from sympy.parsing.maxima import parse_maxima
from sympy.core.numbers import (E, Rational, oo)
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.abc import x

n = Symbol('n', integer=True)


def test_parser():
    assert Abs(parse_maxima('float(1/3)') - 0.333333333) < 10**(-5)
    assert parse_maxima('13^26') == 91733330193268616658399616009
    assert parse_maxima('sin(%pi/2) + cos(%pi/3)') == Rational(3, 2)
    assert parse_maxima('log(%e)') == 1


def test_injection():
    parse_maxima('c: x+1', globals=globals())
    # c created by parse_maxima
    assert c == x + 1 # noqa:F821

    parse_maxima('g: sqrt(81)', globals=globals())
    # g created by parse_maxima
    assert g == 9 # noqa:F821


def test_maxima_functions():
    assert parse_maxima('expand( (x+1)^2)') == x**2 + 2*x + 1
    assert parse_maxima('factor( x**2 + 2*x + 1)') == (x + 1)**2
    assert parse_maxima('2*cos(x)^2 + sin(x)^2') == 2*cos(x)**2 + sin(x)**2
    assert parse_maxima('trigexpand(sin(2*x)+cos(2*x))') == \
        -1 + 2*cos(x)**2 + 2*cos(x)*sin(x)
    assert parse_maxima('solve(x^2-4,x)') == [-2, 2]
    assert parse_maxima('limit((1+1/x)^x,x,inf)') == E
    assert parse_maxima('limit(sqrt(-x)/x,x,0,minus)') is -oo
    assert parse_maxima('diff(x^x, x)') == x**x*(1 + log(x))
    assert parse_maxima('sum(k, k, 1, n)', name_dict={
        "n": Symbol('n', integer=True),
        "k": Symbol('k', integer=True)
    }) == (n**2 + n)/2
    assert parse_maxima('product(k, k, 1, n)', name_dict={
        "n": Symbol('n', integer=True),
        "k": Symbol('k', integer=True)
    }) == factorial(n)
    assert parse_maxima('ratsimp((x^2-1)/(x+1))') == x - 1
    assert Abs( parse_maxima(
        'float(sec(%pi/3) + csc(%pi/3))') - 3.154700538379252) < 10**(-5)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_boson.py ===
from math import prod

from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Dagger, Commutator, qapply
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.boson import (
    BosonFockKet, BosonFockBra, BosonCoherentKet, BosonCoherentBra)


def test_bosonoperator():
    a = BosonOp('a')
    b = BosonOp('b')

    assert isinstance(a, BosonOp)
    assert isinstance(Dagger(a), BosonOp)

    assert a.is_annihilation
    assert not Dagger(a).is_annihilation

    assert BosonOp("a") == BosonOp("a", True)
    assert BosonOp("a") != BosonOp("c")
    assert BosonOp("a", True) != BosonOp("a", False)

    assert Commutator(a, Dagger(a)).doit() == 1

    assert Commutator(a, Dagger(b)).doit() == a * Dagger(b) - Dagger(b) * a

    assert Dagger(exp(a)) == exp(Dagger(a))


def test_boson_states():
    a = BosonOp("a")

    # Fock states
    n = 3
    assert (BosonFockBra(0) * BosonFockKet(1)).doit() == 0
    assert (BosonFockBra(1) * BosonFockKet(1)).doit() == 1
    assert qapply(BosonFockBra(n) * Dagger(a)**n * BosonFockKet(0)) \
        == sqrt(prod(range(1, n+1)))

    # Coherent states
    alpha1, alpha2 = 1.2, 4.3
    assert (BosonCoherentBra(alpha1) * BosonCoherentKet(alpha1)).doit() == 1
    assert (BosonCoherentBra(alpha2) * BosonCoherentKet(alpha2)).doit() == 1
    assert abs((BosonCoherentBra(alpha1) * BosonCoherentKet(alpha2)).doit() -
               exp((alpha1 - alpha2) ** 2 * Rational(-1, 2))) < 1e-12
    assert qapply(a * BosonCoherentKet(alpha1)) == \
        alpha1 * BosonCoherentKet(alpha1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_operatorordering.py ===
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import (normal_order,
                                                 normal_ordered_form)


def test_normal_order():
    a = BosonOp('a')

    c = FermionOp('c')

    assert normal_order(a * Dagger(a)) == Dagger(a) * a
    assert normal_order(Dagger(a) * a) == Dagger(a) * a
    assert normal_order(a * Dagger(a) ** 2) == Dagger(a) ** 2 * a

    assert normal_order(c * Dagger(c)) == - Dagger(c) * c
    assert normal_order(Dagger(c) * c) == Dagger(c) * c
    assert normal_order(c * Dagger(c) ** 2) == Dagger(c) ** 2 * c


def test_normal_ordered_form():
    a = BosonOp('a')
    b = BosonOp('b')

    c = FermionOp('c')
    d = FermionOp('d')

    assert normal_ordered_form(Dagger(a) * a) == Dagger(a) * a
    assert normal_ordered_form(a * Dagger(a)) == 1 + Dagger(a) * a
    assert normal_ordered_form(a ** 2 * Dagger(a)) == \
        2 * a + Dagger(a) * a ** 2
    assert normal_ordered_form(a ** 3 * Dagger(a)) == \
        3 * a ** 2 + Dagger(a) * a ** 3

    assert normal_ordered_form(Dagger(c) * c) == Dagger(c) * c
    assert normal_ordered_form(c * Dagger(c)) == 1 - Dagger(c) * c
    assert normal_ordered_form(c ** 2 * Dagger(c)) == Dagger(c) * c ** 2
    assert normal_ordered_form(c ** 3 * Dagger(c)) == \
        c ** 2 - Dagger(c) * c ** 3

    assert normal_ordered_form(a * Dagger(b), True) == Dagger(b) * a
    assert normal_ordered_form(Dagger(a) * b, True) == Dagger(a) * b
    assert normal_ordered_form(b * a, True) == a * b
    assert normal_ordered_form(Dagger(b) * Dagger(a), True) == Dagger(a) * Dagger(b)

    assert normal_ordered_form(c * Dagger(d), True) == -Dagger(d) * c
    assert normal_ordered_form(Dagger(c) * d, True) == Dagger(c) * d
    assert normal_ordered_form(d * c, True) == -c * d
    assert normal_ordered_form(Dagger(d) * Dagger(c), True) == -Dagger(c) * Dagger(d)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_qho_1d.py ===
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.abc import omega, m, x
from sympy.physics.qho_1d import psi_n, E_n, coherent_state
from sympy.physics.quantum.constants import hbar

nu = m * omega / hbar


def test_wavefunction():
    Psi = {
        0: (nu/pi)**Rational(1, 4) * exp(-nu * x**2 /2),
        1: (nu/pi)**Rational(1, 4) * sqrt(2*nu) * x * exp(-nu * x**2 /2),
        2: (nu/pi)**Rational(1, 4) * (2 * nu * x**2 - 1)/sqrt(2) * exp(-nu * x**2 /2),
        3: (nu/pi)**Rational(1, 4) * sqrt(nu/3) * (2 * nu * x**3 - 3 * x) * exp(-nu * x**2 /2)
    }
    for n in Psi:
        assert simplify(psi_n(n, x, m, omega) - Psi[n]) == 0


def test_norm(n=1):
    # Maximum "n" which is tested:
    for i in range(n + 1):
        assert integrate(psi_n(i, x, 1, 1)**2, (x, -oo, oo)) == 1


def test_orthogonality(n=1):
    # Maximum "n" which is tested:
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            assert integrate(
                psi_n(i, x, 1, 1)*psi_n(j, x, 1, 1), (x, -oo, oo)) == 0


def test_energies(n=1):
    # Maximum "n" which is tested:
    for i in range(n + 1):
        assert E_n(i, omega) == hbar * omega * (i + S.Half)

def test_coherent_state(n=10):
    # Maximum "n" which is tested:
    # test whether coherent state is the eigenstate of annihilation operator
    alpha = Symbol("alpha")
    for i in range(n + 1):
        assert simplify(sqrt(n + 1) * coherent_state(n + 1, alpha)) == simplify(alpha * coherent_state(n, alpha))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\tests\utils.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
from contextlib import contextmanager
from io import StringIO
import sys
import os


class StreamTTY(StringIO):
    def isatty(self):
        return True

class StreamNonTTY(StringIO):
    def isatty(self):
        return False

@contextmanager
def osname(name):
    orig = os.name
    os.name = name
    yield
    os.name = orig

@contextmanager
def replace_by(stream):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = stream
    sys.stderr = stream
    yield
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

@contextmanager
def replace_original_by(stream):
    orig_stdout = sys.__stdout__
    orig_stderr = sys.__stderr__
    sys.__stdout__ = stream
    sys.__stderr__ = stream
    yield
    sys.__stdout__ = orig_stdout
    sys.__stderr__ = orig_stderr

@contextmanager
def pycharm():
    os.environ["PYCHARM_HOSTED"] = "1"
    non_tty = StreamNonTTY()
    with replace_by(non_tty), replace_original_by(non_tty):
        yield
    del os.environ["PYCHARM_HOSTED"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_scripts.py ===
""" Test scripts

Test that we can run executable scripts that have been installed with numpy.
"""
import os
import subprocess
import sys
from os.path import dirname, isfile
from os.path import join as pathjoin

import pytest

import numpy as np
from numpy.testing import IS_WASM, assert_equal

is_inplace = isfile(pathjoin(dirname(np.__file__), '..', 'setup.py'))


def find_f2py_commands():
    if sys.platform == 'win32':
        exe_dir = dirname(sys.executable)
        if exe_dir.endswith('Scripts'):  # virtualenv
            return [os.path.join(exe_dir, 'f2py')]
        else:
            return [os.path.join(exe_dir, "Scripts", 'f2py')]
    else:
        # Three scripts are installed in Unix-like systems:
        # 'f2py', 'f2py{major}', and 'f2py{major.minor}'. For example,
        # if installed with python3.9 the scripts would be named
        # 'f2py', 'f2py3', and 'f2py3.9'.
        version = sys.version_info
        major = str(version.major)
        minor = str(version.minor)
        return ['f2py', 'f2py' + major, 'f2py' + major + '.' + minor]


@pytest.mark.skipif(is_inplace, reason="Cannot test f2py command inplace")
@pytest.mark.xfail(reason="Test is unreliable")
@pytest.mark.parametrize('f2py_cmd', find_f2py_commands())
def test_f2py(f2py_cmd):
    # test that we can run f2py script
    stdout = subprocess.check_output([f2py_cmd, '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
def test_pep338():
    stdout = subprocess.check_output([sys.executable, '-mnumpy.f2py', '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_cpu_dispatcher.py ===
from numpy._core._multiarray_umath import (
    __cpu_baseline__,
    __cpu_dispatch__,
    __cpu_features__,
)

from numpy._core import _umath_tests
from numpy.testing import assert_equal


def test_dispatcher():
    """
    Testing the utilities of the CPU dispatcher
    """
    targets = (
        "SSE2", "SSE41", "AVX2",
        "VSX", "VSX2", "VSX3",
        "NEON", "ASIMD", "ASIMDHP",
        "VX", "VXE", "LSX"
    )
    highest_sfx = ""  # no suffix for the baseline
    all_sfx = []
    for feature in reversed(targets):
        # skip baseline features, by the default `CCompilerOpt` do not generate separated objects
        # for the baseline,  just one object combined all of them via 'baseline' option
        # within the configuration statements.
        if feature in __cpu_baseline__:
            continue
        # check compiler and running machine support
        if feature not in __cpu_dispatch__ or not __cpu_features__[feature]:
            continue

        if not highest_sfx:
            highest_sfx = "_" + feature
        all_sfx.append("func" + "_" + feature)

    test = _umath_tests.test_dispatch()
    assert_equal(test["func"], "func" + highest_sfx)
    assert_equal(test["var"], "var" + highest_sfx)

    if highest_sfx:
        assert_equal(test["func_xb"], "func" + highest_sfx)
        assert_equal(test["var_xb"], "var" + highest_sfx)
    else:
        assert_equal(test["func_xb"], "nobase")
        assert_equal(test["var_xb"], "nobase")

    all_sfx.append("func")  # add the baseline
    assert_equal(test["all"], all_sfx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_add_prefix_suffix.py ===
import pytest

from pandas import Index
import pandas._testing as tm


def test_add_prefix_suffix(float_frame):
    with_prefix = float_frame.add_prefix("foo#")
    expected = Index([f"foo#{c}" for c in float_frame.columns])
    tm.assert_index_equal(with_prefix.columns, expected)

    with_suffix = float_frame.add_suffix("#foo")
    expected = Index([f"{c}#foo" for c in float_frame.columns])
    tm.assert_index_equal(with_suffix.columns, expected)

    with_pct_prefix = float_frame.add_prefix("%")
    expected = Index([f"%{c}" for c in float_frame.columns])
    tm.assert_index_equal(with_pct_prefix.columns, expected)

    with_pct_suffix = float_frame.add_suffix("%")
    expected = Index([f"{c}%" for c in float_frame.columns])
    tm.assert_index_equal(with_pct_suffix.columns, expected)


def test_add_prefix_suffix_axis(float_frame):
    # GH 47819
    with_prefix = float_frame.add_prefix("foo#", axis=0)
    expected = Index([f"foo#{c}" for c in float_frame.index])
    tm.assert_index_equal(with_prefix.index, expected)

    with_prefix = float_frame.add_prefix("foo#", axis=1)
    expected = Index([f"foo#{c}" for c in float_frame.columns])
    tm.assert_index_equal(with_prefix.columns, expected)

    with_pct_suffix = float_frame.add_suffix("#foo", axis=0)
    expected = Index([f"{c}#foo" for c in float_frame.index])
    tm.assert_index_equal(with_pct_suffix.index, expected)

    with_pct_suffix = float_frame.add_suffix("#foo", axis=1)
    expected = Index([f"{c}#foo" for c in float_frame.columns])
    tm.assert_index_equal(with_pct_suffix.columns, expected)


def test_add_prefix_suffix_invalid_axis(float_frame):
    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        float_frame.add_prefix("foo#", axis=2)

    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        float_frame.add_suffix("foo#", axis=2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_to_numpy.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    NA,
    Series,
    Timedelta,
)
import pandas._testing as tm


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_to_numpy_na_value(dtype):
    # GH#48951
    ser = Series([1, 2, NA, 4])
    result = ser.to_numpy(dtype=dtype, na_value=0)
    expected = np.array([1, 2, 0, 4], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_cast_before_setting_na():
    # GH#50600
    ser = Series([1])
    result = ser.to_numpy(dtype=np.float64, na_value=np.nan)
    expected = np.array([1.0])
    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_to_numpy_arrow_dtype_given():
    # GH#57121
    ser = Series([1, NA], dtype="int64[pyarrow]")
    result = ser.to_numpy(dtype="float64")
    expected = np.array([1.0, np.nan])
    tm.assert_numpy_array_equal(result, expected)


def test_astype_ea_int_to_td_ts():
    # GH#57093
    ser = Series([1, None], dtype="Int64")
    result = ser.astype("m8[ns]")
    expected = Series([1, Timedelta("nat")], dtype="m8[ns]")
    tm.assert_series_equal(result, expected)

    result = ser.astype("M8[ns]")
    expected = Series([1, Timedelta("nat")], dtype="M8[ns]")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_rewriting.py ===
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.free_groups import free_group
from sympy.testing.pytest import raises


def test_rewriting():
    F, a, b = free_group("a, b")
    G = FpGroup(F, [a*b*a**-1*b**-1])
    a, b = G.generators
    R = G._rewriting_system
    assert R.is_confluent

    assert G.reduce(b**-1*a) == a*b**-1
    assert G.reduce(b**3*a**4*b**-2*a) == a**5*b
    assert G.equals(b**2*a**-1*b, b**4*a**-1*b**-1)

    assert R.reduce_using_automaton(b*a*a**2*b**-1) == a**3
    assert R.reduce_using_automaton(b**3*a**4*b**-2*a) == a**5*b
    assert R.reduce_using_automaton(b**-1*a) == a*b**-1

    G = FpGroup(F, [a**3, b**3, (a*b)**2])
    R = G._rewriting_system
    R.make_confluent()
    # R._is_confluent should be set to True after
    # a successful run of make_confluent
    assert R.is_confluent
    # but also the system should actually be confluent
    assert R._check_confluence()
    assert G.reduce(b*a**-1*b**-1*a**3*b**4*a**-1*b**-15) == a**-1*b**-1
    # check for automaton reduction
    assert R.reduce_using_automaton(b*a**-1*b**-1*a**3*b**4*a**-1*b**-15) == a**-1*b**-1

    G = FpGroup(F, [a**2, b**3, (a*b)**4])
    R = G._rewriting_system
    assert G.reduce(a**2*b**-2*a**2*b) == b**-1
    assert R.reduce_using_automaton(a**2*b**-2*a**2*b) == b**-1
    assert G.reduce(a**3*b**-2*a**2*b) == a**-1*b**-1
    assert R.reduce_using_automaton(a**3*b**-2*a**2*b) == a**-1*b**-1
    # Check after adding a rule
    R.add_rule(a**2, b)
    assert R.reduce_using_automaton(a**2*b**-2*a**2*b) == b**-1
    assert R.reduce_using_automaton(a**4*b**-2*a**2*b**3) == b

    R.set_max(15)
    raises(RuntimeError, lambda:  R.add_rule(a**-3, b))
    R.set_max(20)
    R.add_rule(a**-3, b)

    assert R.add_rule(a, a) == set()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_egyptian_fraction.py ===
from sympy.core.numbers import Rational
from sympy.ntheory.egyptian_fraction import egyptian_fraction
from sympy.core.add import Add
from sympy.testing.pytest import raises
from sympy.core.random import random_complex_number


def test_egyptian_fraction():
    def test_equality(r, alg="Greedy"):
        return r == Add(*[Rational(1, i) for i in egyptian_fraction(r, alg)])

    r = random_complex_number(a=0, c=1, b=0, d=0, rational=True)
    assert test_equality(r)

    assert egyptian_fraction(Rational(4, 17)) == [5, 29, 1233, 3039345]
    assert egyptian_fraction(Rational(7, 13), "Greedy") == [2, 26]
    assert egyptian_fraction(Rational(23, 101), "Greedy") == \
        [5, 37, 1438, 2985448, 40108045937720]
    assert egyptian_fraction(Rational(18, 23), "Takenouchi") == \
        [2, 6, 12, 35, 276, 2415]
    assert egyptian_fraction(Rational(5, 6), "Graham Jewett") == \
        [6, 7, 8, 9, 10, 42, 43, 44, 45, 56, 57, 58, 72, 73, 90, 1806, 1807,
         1808, 1892, 1893, 1980, 3192, 3193, 3306, 5256, 3263442, 3263443,
         3267056, 3581556, 10192056, 10650056950806]
    assert egyptian_fraction(Rational(5, 6), "Golomb") == [2, 6, 12, 20, 30]
    assert egyptian_fraction(Rational(5, 121), "Golomb") == [25, 1225, 3577, 7081, 11737]
    raises(ValueError, lambda: egyptian_fraction(Rational(-4, 9)))
    assert egyptian_fraction(Rational(8, 3), "Golomb") == [1, 2, 3, 4, 5, 6, 7,
                                                           14, 574, 2788, 6460,
                                                           11590, 33062, 113820]
    assert egyptian_fraction(Rational(355, 113)) == [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                     10, 11, 12, 27, 744, 893588,
                                                     1251493536607,
                                                     20361068938197002344405230]


def test_input():
    r = (2,3), Rational(2, 3), (Rational(2), Rational(3))
    for m in ["Greedy", "Graham Jewett", "Takenouchi", "Golomb"]:
        for i in r:
            d = egyptian_fraction(i, m)
            assert all(i.is_Integer for i in d)
            if m == "Graham Jewett":
                assert d == [3, 4, 12]
            else:
                assert d == [2, 6]
    # check prefix
    d = egyptian_fraction(Rational(5, 3))
    assert d == [1, 2, 6] and all(i.is_Integer for i in d)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_return_character.py ===
import platform

import pytest

from numpy import array

from . import util

IS_S390X = platform.machine() == "s390x"


@pytest.mark.slow
class TestReturnCharacter(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t1", "s0", "s1"]:
            assert t("23") == b"2"
            r = t("ab")
            assert r == b"a"
            r = t(array("ab"))
            assert r == b"a"
            r = t(array(77, "u1"))
            assert r == b"M"
        elif tname in ["ts", "ss"]:
            assert t(23) == b"23"
            assert t("123456789abcdef") == b"123456789a"
        elif tname in ["t5", "s5"]:
            assert t(23) == b"23"
            assert t("ab") == b"ab"
            assert t("123456789abcdef") == b"12345"
        else:
            raise NotImplementedError


class TestFReturnCharacter(TestReturnCharacter):
    sources = [
        util.getpath("tests", "src", "return_character", "foo77.f"),
        util.getpath("tests", "src", "return_character", "foo90.f90"),
    ]

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize("name", ["t0", "t1", "t5", "s0", "s1", "s5", "ss"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.xfail(IS_S390X, reason="callback returns ' '")
    @pytest.mark.parametrize("name", ["t0", "t1", "t5", "ts", "s0", "s1", "s5", "ss"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_char, name), name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_configtool.py ===
import importlib
import importlib.metadata
import os
import pathlib
import subprocess

import pytest

import numpy as np
import numpy._core.include
import numpy._core.lib.pkgconfig
from numpy.testing import IS_EDITABLE, IS_INSTALLED, IS_WASM, NUMPY_ROOT

INCLUDE_DIR = NUMPY_ROOT / '_core' / 'include'
PKG_CONFIG_DIR = NUMPY_ROOT / '_core' / 'lib' / 'pkgconfig'


@pytest.mark.skipif(not IS_INSTALLED, reason="`numpy-config` not expected to be installed")
@pytest.mark.skipif(IS_WASM, reason="wasm interpreter cannot start subprocess")
class TestNumpyConfig:
    def check_numpyconfig(self, arg):
        p = subprocess.run(['numpy-config', arg], capture_output=True, text=True)
        p.check_returncode()
        return p.stdout.strip()

    def test_configtool_version(self):
        stdout = self.check_numpyconfig('--version')
        assert stdout == np.__version__

    def test_configtool_cflags(self):
        stdout = self.check_numpyconfig('--cflags')
        assert f'-I{os.fspath(INCLUDE_DIR)}' in stdout

    def test_configtool_pkgconfigdir(self):
        stdout = self.check_numpyconfig('--pkgconfigdir')
        assert pathlib.Path(stdout) == PKG_CONFIG_DIR


@pytest.mark.skipif(not IS_INSTALLED, reason="numpy must be installed to check its entrypoints")
def test_pkg_config_entrypoint():
    (entrypoint,) = importlib.metadata.entry_points(group='pkg_config', name='numpy')
    assert entrypoint.value == numpy._core.lib.pkgconfig.__name__


@pytest.mark.skipif(not IS_INSTALLED, reason="numpy.pc is only available when numpy is installed")
@pytest.mark.skipif(IS_EDITABLE, reason="editable installs don't have a numpy.pc")
def test_pkg_config_config_exists():
    assert PKG_CONFIG_DIR.joinpath('numpy.pc').is_file()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_flags.py ===
import pytest

import pandas as pd


class TestFlags:
    def test_equality(self):
        a = pd.DataFrame().set_flags(allows_duplicate_labels=True).flags
        b = pd.DataFrame().set_flags(allows_duplicate_labels=False).flags

        assert a == a
        assert b == b
        assert a != b
        assert a != 2

    def test_set(self):
        df = pd.DataFrame().set_flags(allows_duplicate_labels=True)
        a = df.flags
        a.allows_duplicate_labels = False
        assert a.allows_duplicate_labels is False
        a["allows_duplicate_labels"] = True
        assert a.allows_duplicate_labels is True

    def test_repr(self):
        a = repr(pd.DataFrame({"A"}).set_flags(allows_duplicate_labels=True).flags)
        assert a == "<Flags(allows_duplicate_labels=True)>"
        a = repr(pd.DataFrame({"A"}).set_flags(allows_duplicate_labels=False).flags)
        assert a == "<Flags(allows_duplicate_labels=False)>"

    def test_obj_ref(self):
        df = pd.DataFrame()
        flags = df.flags
        del df
        with pytest.raises(ValueError, match="object has been deleted"):
            flags.allows_duplicate_labels = True

    def test_getitem(self):
        df = pd.DataFrame()
        flags = df.flags
        assert flags["allows_duplicate_labels"] is True
        flags["allows_duplicate_labels"] = False
        assert flags["allows_duplicate_labels"] is False

        with pytest.raises(KeyError, match="a"):
            flags["a"]

        with pytest.raises(ValueError, match="a"):
            flags["a"] = 10

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\conftest.py ===
import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


@pytest.fixture(params=[Float32Dtype, Float64Dtype])
def dtype(request):
    """Parametrized fixture returning a float 'dtype'"""
    return request.param()


@pytest.fixture
def data(dtype):
    """Fixture returning 'data' array according to parametrized float 'dtype'"""
    return pd.array(
        list(np.arange(0.1, 0.9, 0.1))
        + [pd.NA]
        + list(np.arange(1, 9.8, 0.1))
        + [pd.NA]
        + [9.9, 10.0],
        dtype=dtype,
    )


@pytest.fixture
def data_missing(dtype):
    """
    Fixture returning array with missing data according to parametrized float
    'dtype'.
    """
    return pd.array([np.nan, 0.1], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' float arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_companion.py ===
from sympy.core.expr import unchanged
from sympy.core.symbol import Symbol, symbols
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.companion import CompanionMatrix
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises


def test_creation():
    x = Symbol('x')
    y = Symbol('y')
    raises(ValueError, lambda: CompanionMatrix(1))
    raises(ValueError, lambda: CompanionMatrix(Poly([1], x)))
    raises(ValueError, lambda: CompanionMatrix(Poly([2, 1], x)))
    raises(ValueError, lambda: CompanionMatrix(Poly(x*y, [x, y])))
    assert unchanged(CompanionMatrix, Poly([1, 2, 3], x))


def test_shape():
    c0, c1, c2 = symbols('c0:3')
    x = Symbol('x')
    assert CompanionMatrix(Poly([1, c0], x)).shape == (1, 1)
    assert CompanionMatrix(Poly([1, c1, c0], x)).shape == (2, 2)
    assert CompanionMatrix(Poly([1, c2, c1, c0], x)).shape == (3, 3)


def test_entry():
    c0, c1, c2 = symbols('c0:3')
    x = Symbol('x')
    A = CompanionMatrix(Poly([1, c2, c1, c0], x))
    assert A[0, 0] == 0
    assert A[1, 0] == 1
    assert A[1, 1] == 0
    assert A[2, 1] == 1
    assert A[0, 2] == -c0
    assert A[1, 2] == -c1
    assert A[2, 2] == -c2


def test_as_explicit():
    c0, c1, c2 = symbols('c0:3')
    x = Symbol('x')
    assert CompanionMatrix(Poly([1, c0], x)).as_explicit() == \
        ImmutableDenseMatrix([-c0])
    assert CompanionMatrix(Poly([1, c1, c0], x)).as_explicit() == \
        ImmutableDenseMatrix([[0, -c0], [1, -c1]])
    assert CompanionMatrix(Poly([1, c2, c1, c0], x)).as_explicit() == \
        ImmutableDenseMatrix([[0, 0, -c0], [1, 0, -c1], [0, 1, -c2]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_multinomial.py ===
from sympy.ntheory.multinomial import (binomial_coefficients, binomial_coefficients_list, multinomial_coefficients)
from sympy.ntheory.multinomial import multinomial_coefficients_iterator


def test_binomial_coefficients_list():
    assert binomial_coefficients_list(0) == [1]
    assert binomial_coefficients_list(1) == [1, 1]
    assert binomial_coefficients_list(2) == [1, 2, 1]
    assert binomial_coefficients_list(3) == [1, 3, 3, 1]
    assert binomial_coefficients_list(4) == [1, 4, 6, 4, 1]
    assert binomial_coefficients_list(5) == [1, 5, 10, 10, 5, 1]
    assert binomial_coefficients_list(6) == [1, 6, 15, 20, 15, 6, 1]


def test_binomial_coefficients():
    for n in range(15):
        c = binomial_coefficients(n)
        l = [c[k] for k in sorted(c)]
        assert l == binomial_coefficients_list(n)


def test_multinomial_coefficients():
    assert multinomial_coefficients(1, 1) == {(1,): 1}
    assert multinomial_coefficients(1, 2) == {(2,): 1}
    assert multinomial_coefficients(1, 3) == {(3,): 1}
    assert multinomial_coefficients(2, 0) == {(0, 0): 1}
    assert multinomial_coefficients(2, 1) == {(0, 1): 1, (1, 0): 1}
    assert multinomial_coefficients(2, 2) == {(2, 0): 1, (0, 2): 1, (1, 1): 2}
    assert multinomial_coefficients(2, 3) == {(3, 0): 1, (1, 2): 3, (0, 3): 1,
            (2, 1): 3}
    assert multinomial_coefficients(3, 1) == {(1, 0, 0): 1, (0, 1, 0): 1,
            (0, 0, 1): 1}
    assert multinomial_coefficients(3, 2) == {(0, 1, 1): 2, (0, 0, 2): 1,
            (1, 1, 0): 2, (0, 2, 0): 1, (1, 0, 1): 2, (2, 0, 0): 1}
    mc = multinomial_coefficients(3, 3)
    assert mc == {(2, 1, 0): 3, (0, 3, 0): 1,
            (1, 0, 2): 3, (0, 2, 1): 3, (0, 1, 2): 3, (3, 0, 0): 1,
            (2, 0, 1): 3, (1, 2, 0): 3, (1, 1, 1): 6, (0, 0, 3): 1}
    assert dict(multinomial_coefficients_iterator(2, 0)) == {(0, 0): 1}
    assert dict(
        multinomial_coefficients_iterator(2, 1)) == {(0, 1): 1, (1, 0): 1}
    assert dict(multinomial_coefficients_iterator(2, 2)) == \
        {(2, 0): 1, (0, 2): 1, (1, 1): 2}
    assert dict(multinomial_coefficients_iterator(3, 3)) == mc
    it = multinomial_coefficients_iterator(7, 2)
    assert [next(it) for i in range(4)] == \
        [((2, 0, 0, 0, 0, 0, 0), 1), ((1, 1, 0, 0, 0, 0, 0), 2),
      ((0, 2, 0, 0, 0, 0, 0), 1), ((1, 0, 1, 0, 0, 0, 0), 2)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\biomechanics\tests\test_mixin.py ===
"""Tests for the ``sympy.physics.biomechanics._mixin.py`` module."""

import pytest

from sympy.physics.biomechanics._mixin import _NamedMixin


class TestNamedMixin:

    @staticmethod
    def test_subclass():

        class Subclass(_NamedMixin):

            def __init__(self, name):
                self.name = name

        instance = Subclass('name')
        assert instance.name == 'name'

    @pytest.fixture(autouse=True)
    def _named_mixin_fixture(self):

        class Subclass(_NamedMixin):

            def __init__(self, name):
                self.name = name

        self.Subclass = Subclass

    @pytest.mark.parametrize('name', ['a', 'name', 'long_name'])
    def test_valid_name_argument(self, name):
        instance = self.Subclass(name)
        assert instance.name == name

    @pytest.mark.parametrize('invalid_name', [0, 0.0, None, False])
    def test_invalid_name_argument_not_str(self, invalid_name):
        with pytest.raises(TypeError):
            _ = self.Subclass(invalid_name)

    def test_invalid_name_argument_zero_length_str(self):
        with pytest.raises(ValueError):
            _ = self.Subclass('')

    def test_name_attribute_is_immutable(self):
        instance = self.Subclass('name')
        with pytest.raises(AttributeError):
            instance.name = 'new_name'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\tests\test_medium.py ===
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.optics import Medium
from sympy.abc import epsilon, mu, n
from sympy.physics.units import speed_of_light, u0, e0, m, kg, s, A

from sympy.testing.pytest import raises

c = speed_of_light.convert_to(m/s)
e0 = e0.convert_to(A**2*s**4/(kg*m**3))
u0 = u0.convert_to(m*kg/(A**2*s**2))


def test_medium():
    m1 = Medium('m1')
    assert m1.intrinsic_impedance == sqrt(u0/e0)
    assert m1.speed == 1/sqrt(e0*u0)
    assert m1.refractive_index == c*sqrt(e0*u0)
    assert m1.permittivity == e0
    assert m1.permeability == u0
    m2 = Medium('m2', epsilon, mu)
    assert m2.intrinsic_impedance == sqrt(mu/epsilon)
    assert m2.speed == 1/sqrt(epsilon*mu)
    assert m2.refractive_index == c*sqrt(epsilon*mu)
    assert m2.permittivity == epsilon
    assert m2.permeability == mu
    # Increasing electric permittivity and magnetic permeability
    # by small amount from its value in vacuum.
    m3 = Medium('m3', 9.0*10**(-12)*s**4*A**2/(m**3*kg), 1.45*10**(-6)*kg*m/(A**2*s**2))
    assert m3.refractive_index > m1.refractive_index
    assert m3 != m1
    # Decreasing electric permittivity and magnetic permeability
    # by small amount from its value in vacuum.
    m4 = Medium('m4', 7.0*10**(-12)*s**4*A**2/(m**3*kg), 1.15*10**(-6)*kg*m/(A**2*s**2))
    assert m4.refractive_index < m1.refractive_index
    m5 = Medium('m5', permittivity=710*10**(-12)*s**4*A**2/(m**3*kg), n=1.33)
    assert abs(m5.intrinsic_impedance - 6.24845417765552*kg*m**2/(A**2*s**3)) \
                < 1e-12*kg*m**2/(A**2*s**3)
    assert abs(m5.speed - 225407863.157895*m/s) < 1e-6*m/s
    assert abs(m5.refractive_index - 1.33000000000000) < 1e-12
    assert abs(m5.permittivity - 7.1e-10*A**2*s**4/(kg*m**3)) \
                < 1e-20*A**2*s**4/(kg*m**3)
    assert abs(m5.permeability - 2.77206575232851e-8*kg*m/(A**2*s**2)) \
                < 1e-20*kg*m/(A**2*s**2)
    m6 = Medium('m6', None, mu, n)
    assert m6.permittivity == n**2/(c**2*mu)
    # test for equality of refractive indices
    assert Medium('m7').refractive_index == Medium('m8', e0, u0).refractive_index
    raises(ValueError, lambda:Medium('m9', e0, u0, 2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\fail_clearing_run_switches.py ===
# -*- coding: utf-8 -*-
"""
If we have a run callable passed to the constructor or set as an
attribute, but we don't actually use that (because ``__getattribute__``
or the like interferes), then when we clear callable before beginning
to run, there's an opportunity for Python code to run.

"""
import greenlet

g = None
main = greenlet.getcurrent()

results = []

class RunCallable:

    def __del__(self):
        results.append(('RunCallable', '__del__'))
        main.switch('from RunCallable')


class G(greenlet.greenlet):

    def __getattribute__(self, name):
        if name == 'run':
            results.append(('G.__getattribute__', 'run'))
            return run_func
        return object.__getattribute__(self, name)


def run_func():
    results.append(('run_func', 'enter'))


g = G(RunCallable())
# Try to start G. It will get to the point where it deletes
# its run callable C++ variable in inner_bootstrap. That triggers
# the __del__ method, which switches back to main before g
# actually even starts running.
x = g.switch()
results.append(('main: g.switch()', x))
# In the C++ code, this results in g->g_switch() appearing to return, even though
# it has yet to run.
print('In main with', x, flush=True)
g.switch()
print('RESULTS', results)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\ndarray_shape_manipulation.py ===
import numpy as np

nd1 = np.array([[1, 2], [3, 4]])

# reshape
nd1.reshape(4)
nd1.reshape(2, 2)
nd1.reshape((2, 2))

nd1.reshape((2, 2), order="C")
nd1.reshape(4, order="C")

# resize
nd1.resize()
nd1.resize(4)
nd1.resize(2, 2)
nd1.resize((2, 2))

nd1.resize((2, 2), refcheck=True)
nd1.resize(4, refcheck=True)

nd2 = np.array([[1, 2], [3, 4]])

# transpose
nd2.transpose()
nd2.transpose(1, 0)
nd2.transpose((1, 0))

# swapaxes
nd2.swapaxes(0, 1)

# flatten
nd2.flatten()
nd2.flatten("C")

# ravel
nd2.ravel()
nd2.ravel("C")

# squeeze
nd2.squeeze()

nd3 = np.array([[1, 2]])
nd3.squeeze(0)

nd4 = np.array([[[1, 2]]])
nd4.squeeze((0, 1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\ufunclike.py ===
from __future__ import annotations
from typing import Any
import numpy as np


class Object:
    def __ceil__(self) -> Object:
        return self

    def __floor__(self) -> Object:
        return self

    def __ge__(self, value: object) -> bool:
        return True

    def __array__(self, dtype: np.typing.DTypeLike | None = None,
                  copy: bool | None = None) -> np.ndarray[Any, np.dtype[np.object_]]:
        ret = np.empty((), dtype=object)
        ret[()] = self
        return ret


AR_LIKE_b = [True, True, False]
AR_LIKE_u = [np.uint32(1), np.uint32(2), np.uint32(3)]
AR_LIKE_i = [1, 2, 3]
AR_LIKE_f = [1.0, 2.0, 3.0]
AR_LIKE_O = [Object(), Object(), Object()]
AR_U: np.ndarray[Any, np.dtype[np.str_]] = np.zeros(3, dtype="U5")

np.fix(AR_LIKE_b)
np.fix(AR_LIKE_u)
np.fix(AR_LIKE_i)
np.fix(AR_LIKE_f)
np.fix(AR_LIKE_O)
np.fix(AR_LIKE_f, out=AR_U)

np.isposinf(AR_LIKE_b)
np.isposinf(AR_LIKE_u)
np.isposinf(AR_LIKE_i)
np.isposinf(AR_LIKE_f)
np.isposinf(AR_LIKE_f, out=AR_U)

np.isneginf(AR_LIKE_b)
np.isneginf(AR_LIKE_u)
np.isneginf(AR_LIKE_i)
np.isneginf(AR_LIKE_f)
np.isneginf(AR_LIKE_f, out=AR_U)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_repr.py ===
import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.floating import (
    Float32Dtype,
    Float64Dtype,
)


def test_dtypes(dtype):
    # smoke tests on auto dtype construction

    np.dtype(dtype.type).kind == "f"
    assert dtype.name is not None


@pytest.mark.parametrize(
    "dtype, expected",
    [(Float32Dtype(), "Float32Dtype()"), (Float64Dtype(), "Float64Dtype()")],
)
def test_repr_dtype(dtype, expected):
    assert repr(dtype) == expected


def test_repr_array():
    result = repr(pd.array([1.0, None, 3.0]))
    expected = "<FloatingArray>\n[1.0, <NA>, 3.0]\nLength: 3, dtype: Float64"
    assert result == expected


def test_repr_array_long():
    data = pd.array([1.0, 2.0, None] * 1000)
    expected = """<FloatingArray>
[ 1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,
 ...
 <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>]
Length: 3000, dtype: Float64"""
    result = repr(data)
    assert result == expected


def test_frame_repr(data_missing):
    df = pd.DataFrame({"A": data_missing})
    result = repr(df)
    expected = "      A\n0  <NA>\n1   0.1"
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_arrow_interface.py ===
import ctypes

import pytest

import pandas.util._test_decorators as td

import pandas as pd

pa = pytest.importorskip("pyarrow")


@td.skip_if_no("pyarrow", min_version="14.0")
def test_dataframe_arrow_interface(using_infer_string):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    capsule = df.__arrow_c_stream__()
    assert (
        ctypes.pythonapi.PyCapsule_IsValid(
            ctypes.py_object(capsule), b"arrow_array_stream"
        )
        == 1
    )

    table = pa.table(df)
    string_type = pa.large_string() if using_infer_string else pa.string()
    expected = pa.table({"a": [1, 2, 3], "b": pa.array(["a", "b", "c"], string_type)})
    assert table.equals(expected)

    schema = pa.schema([("a", pa.int8()), ("b", pa.string())])
    table = pa.table(df, schema=schema)
    expected = expected.cast(schema)
    assert table.equals(expected)


@td.skip_if_no("pyarrow", min_version="15.0")
def test_dataframe_to_arrow(using_infer_string):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})

    table = pa.RecordBatchReader.from_stream(df).read_all()
    string_type = pa.large_string() if using_infer_string else pa.string()
    expected = pa.table({"a": [1, 2, 3], "b": pa.array(["a", "b", "c"], string_type)})
    assert table.equals(expected)

    schema = pa.schema([("a", pa.int8()), ("b", pa.string())])
    table = pa.RecordBatchReader.from_stream(df, schema=schema).read_all()
    expected = expected.cast(schema)
    assert table.equals(expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_combine.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestCombine:
    @pytest.mark.parametrize(
        "data",
        [
            pd.date_range("2000", periods=4),
            pd.date_range("2000", periods=4, tz="US/Central"),
            pd.period_range("2000", periods=4),
            pd.timedelta_range(0, periods=4),
        ],
    )
    def test_combine_datetlike_udf(self, data):
        # GH#23079
        df = pd.DataFrame({"A": data})
        other = df.copy()
        df.iloc[1, 0] = None

        def combiner(a, b):
            return b

        result = df.combine(other, combiner)
        tm.assert_frame_equal(result, other)

    def test_combine_generic(self, float_frame):
        df1 = float_frame
        df2 = float_frame.loc[float_frame.index[:-5], ["A", "B", "C"]]

        combined = df1.combine(df2, np.add)
        combined2 = df2.combine(df1, np.add)
        assert combined["D"].isna().all()
        assert combined2["D"].isna().all()

        chunk = combined.loc[combined.index[:-5], ["A", "B", "C"]]
        chunk2 = combined2.loc[combined2.index[:-5], ["A", "B", "C"]]

        exp = (
            float_frame.loc[float_frame.index[:-5], ["A", "B", "C"]].reindex_like(chunk)
            * 2
        )
        tm.assert_frame_equal(chunk, exp)
        tm.assert_frame_equal(chunk2, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_map.py ===
import pytest

from pandas import (
    DatetimeIndex,
    Index,
    MultiIndex,
    Period,
    date_range,
)
import pandas._testing as tm


class TestMap:
    def test_map(self):
        rng = date_range("1/1/2000", periods=10)

        f = lambda x: x.strftime("%Y%m%d")
        result = rng.map(f)
        exp = Index([f(x) for x in rng])
        tm.assert_index_equal(result, exp)

    def test_map_fallthrough(self, capsys):
        # GH#22067, check we don't get warnings about silently ignored errors
        dti = date_range("2017-01-01", "2018-01-01", freq="B")

        dti.map(lambda x: Period(year=x.year, month=x.month, freq="M"))

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_map_bug_1677(self):
        index = DatetimeIndex(["2012-04-25 09:30:00.393000"])
        f = index.asof

        result = index.map(f)
        expected = Index([f(index[0])])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("name", [None, "name"])
    def test_index_map(self, name):
        # see GH#20990
        count = 6
        index = date_range("2018-01-01", periods=count, freq="ME", name=name).map(
            lambda x: (x.year, x.month)
        )
        exp_index = MultiIndex.from_product(((2018,), range(1, 7)), names=[name, name])
        tm.assert_index_equal(index, exp_index)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_snap.py ===
import pytest

from pandas import (
    DatetimeIndex,
    date_range,
)
import pandas._testing as tm


@pytest.mark.parametrize("tz", [None, "Asia/Shanghai", "Europe/Berlin"])
@pytest.mark.parametrize("name", [None, "my_dti"])
@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_dti_snap(name, tz, unit):
    dti = DatetimeIndex(
        [
            "1/1/2002",
            "1/2/2002",
            "1/3/2002",
            "1/4/2002",
            "1/5/2002",
            "1/6/2002",
            "1/7/2002",
        ],
        name=name,
        tz=tz,
        freq="D",
    )
    dti = dti.as_unit(unit)

    result = dti.snap(freq="W-MON")
    expected = date_range("12/31/2001", "1/7/2002", name=name, tz=tz, freq="w-mon")
    expected = expected.repeat([3, 4])
    expected = expected.as_unit(unit)
    tm.assert_index_equal(result, expected)
    assert result.tz == expected.tz
    assert result.freq is None
    assert expected.freq is None

    result = dti.snap(freq="B")

    expected = date_range("1/1/2002", "1/7/2002", name=name, tz=tz, freq="b")
    expected = expected.repeat([1, 1, 1, 2, 2])
    expected = expected.as_unit(unit)
    tm.assert_index_equal(result, expected)
    assert result.tz == expected.tz
    assert result.freq is None
    assert expected.freq is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_join.py ===
import numpy as np

from pandas import (
    DataFrame,
    Index,
    Timedelta,
    timedelta_range,
)
import pandas._testing as tm


class TestJoin:
    def test_append_join_nondatetimeindex(self):
        rng = timedelta_range("1 days", periods=10)
        idx = Index(["a", "b", "c", "d"])

        result = rng.append(idx)
        assert isinstance(result[0], Timedelta)

        # it works
        rng.join(idx, how="outer")

    def test_join_self(self, join_type):
        index = timedelta_range("1 day", periods=10)
        joined = index.join(index, how=join_type)
        tm.assert_index_equal(index, joined)

    def test_does_not_convert_mixed_integer(self):
        df = DataFrame(np.ones((5, 5)), columns=timedelta_range("1 day", periods=5))

        cols = df.columns.join(df.index, how="outer")
        joined = cols.join(df.columns)
        assert cols.dtype == np.dtype("O")
        assert cols.dtype == joined.dtype
        tm.assert_index_equal(cols, joined)

    def test_join_preserves_freq(self):
        # GH#32157
        tdi = timedelta_range("1 day", periods=10)
        result = tdi[:5].join(tdi[5:], how="outer")
        assert result.freq == tdi.freq
        tm.assert_index_equal(result, tdi)

        result = tdi[:5].join(tdi[6:], how="outer")
        assert result.freq is None
        expected = tdi.delete(5)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_rename_axis.py ===
import pytest

from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestSeriesRenameAxis:
    def test_rename_axis_mapper(self):
        # GH 19978
        mi = MultiIndex.from_product([["a", "b", "c"], [1, 2]], names=["ll", "nn"])
        ser = Series(list(range(len(mi))), index=mi)

        result = ser.rename_axis(index={"ll": "foo"})
        assert result.index.names == ["foo", "nn"]

        result = ser.rename_axis(index=str.upper, axis=0)
        assert result.index.names == ["LL", "NN"]

        result = ser.rename_axis(index=["foo", "goo"])
        assert result.index.names == ["foo", "goo"]

        with pytest.raises(TypeError, match="unexpected"):
            ser.rename_axis(columns="wrong")

    def test_rename_axis_inplace(self, datetime_series):
        # GH 15704
        expected = datetime_series.rename_axis("foo")
        result = datetime_series
        no_return = result.rename_axis("foo", inplace=True)

        assert no_return is None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("kwargs", [{"mapper": None}, {"index": None}, {}])
    def test_rename_axis_none(self, kwargs):
        # GH 25034
        index = Index(list("abc"), name="foo")
        ser = Series([1, 2, 3], index=index)

        result = ser.rename_axis(**kwargs)
        expected_index = index.rename(None) if kwargs else index
        expected = Series([1, 2, 3], index=expected_index)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_numpy_config.py ===
"""
Check the numpy config is valid.
"""
from unittest.mock import patch

import pytest

import numpy as np

pytestmark = pytest.mark.skipif(
    not hasattr(np.__config__, "_built_with_meson"),
    reason="Requires Meson builds",
)


class TestNumPyConfigs:
    REQUIRED_CONFIG_KEYS = [
        "Compilers",
        "Machine Information",
        "Python Information",
    ]

    @patch("numpy.__config__._check_pyyaml")
    def test_pyyaml_not_found(self, mock_yaml_importer):
        mock_yaml_importer.side_effect = ModuleNotFoundError()
        with pytest.warns(UserWarning):
            np.show_config()

    def test_dict_mode(self):
        config = np.show_config(mode="dicts")

        assert isinstance(config, dict)
        assert all(key in config for key in self.REQUIRED_CONFIG_KEYS), (
            "Required key missing,"
            " see index of `False` with `REQUIRED_CONFIG_KEYS`"
        )

    def test_invalid_mode(self):
        with pytest.raises(AttributeError):
            np.show_config(mode="foo")

    def test_warn_to_add_tests(self):
        assert len(np.__config__.DisplayModes) == 2, (
            "New mode detected,"
            " please add UT if applicable and increment this count"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_protocols.py ===
import warnings

import pytest

import numpy as np


@pytest.mark.filterwarnings("error")
def test_getattr_warning():
    # issue gh-14735: make sure we clear only getattr errors, and let warnings
    # through
    class Wrapper:
        def __init__(self, array):
            self.array = array

        def __len__(self):
            return len(self.array)

        def __getitem__(self, item):
            return type(self)(self.array[item])

        def __getattr__(self, name):
            if name.startswith("__array_"):
                warnings.warn("object got converted", UserWarning, stacklevel=1)

            return getattr(self.array, name)

        def __repr__(self):
            return f"<Wrapper({self.array})>"

    array = Wrapper(np.arange(10))
    with pytest.raises(UserWarning, match="object got converted"):
        np.asarray(array)


def test_array_called():
    class Wrapper:
        val = '0' * 100

        def __array__(self, dtype=None, copy=None):
            return np.array([self.val], dtype=dtype, copy=copy)

    wrapped = Wrapper()
    arr = np.array(wrapped, dtype=str)
    assert arr.dtype == 'U100'
    assert arr[0] == Wrapper.val

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\test_is_monotonic.py ===
from pandas import (
    Index,
    NaT,
    date_range,
)


def test_is_monotonic_with_nat():
    # GH#31437
    # PeriodIndex.is_monotonic_increasing should behave analogously to DatetimeIndex,
    #  in particular never be monotonic when we have NaT
    dti = date_range("2016-01-01", periods=3)
    pi = dti.to_period("D")
    tdi = Index(dti.view("timedelta64[ns]"))

    for obj in [pi, pi._engine, dti, dti._engine, tdi, tdi._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert obj.is_monotonic_increasing
        assert obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique

    dti1 = dti.insert(0, NaT)
    pi1 = dti1.to_period("D")
    tdi1 = Index(dti1.view("timedelta64[ns]"))

    for obj in [pi1, pi1._engine, dti1, dti1._engine, tdi1, tdi1._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique

    dti2 = dti.insert(3, NaT)
    pi2 = dti2.to_period("h")
    tdi2 = Index(dti2.view("timedelta64[ns]"))

    for obj in [pi2, pi2._engine, dti2, dti2._engine, tdi2, tdi2._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_lexsort.py ===
from pandas import MultiIndex


class TestIsLexsorted:
    def test_is_lexsorted(self):
        levels = [[0, 1], [0, 1, 2]]

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
        )
        assert index._is_lexsorted()

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]]
        )
        assert not index._is_lexsorted()

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]]
        )
        assert not index._is_lexsorted()
        assert index._lexsort_depth == 0


class TestLexsortDepth:
    def test_lexsort_depth(self):
        # Test that lexsort_depth return the correct sortorder
        # when it was given to the MultiIndex const.
        # GH#28518

        levels = [[0, 1], [0, 1, 2]]

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2
        )
        assert index._lexsort_depth == 2

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=1
        )
        assert index._lexsort_depth == 1

        index = MultiIndex(
            levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=0
        )
        assert index._lexsort_depth == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_npfuncs.py ===
"""
Tests for np.foo applied to Series, not necessarily ufuncs.
"""

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import Series
import pandas._testing as tm


class TestPtp:
    def test_ptp(self):
        # GH#21614
        N = 1000
        arr = np.random.default_rng(2).standard_normal(N)
        ser = Series(arr)
        assert np.ptp(ser) == np.ptp(arr)


def test_numpy_unique(datetime_series):
    # it works!
    np.unique(datetime_series)


@pytest.mark.parametrize("index", [["a", "b", "c", "d", "e"], None])
def test_numpy_argwhere(index):
    # GH#35331

    s = Series(range(5), index=index, dtype=np.int64)

    result = np.argwhere(s > 2).astype(np.int64)
    expected = np.array([[3], [4]], dtype=np.int64)

    tm.assert_numpy_array_equal(result, expected)


@td.skip_if_no("pyarrow")
def test_log_arrow_backed_missing_value():
    # GH#56285
    ser = Series([1, 2, None], dtype="float64[pyarrow]")
    result = np.log(ser)
    expected = np.log(Series([1, 2, None], dtype="float64"))
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_lagrange2.py ===
from sympy import symbols
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.mechanics import ReferenceFrame, Point, Particle
from sympy.physics.mechanics import LagrangesMethod, Lagrangian

### This test asserts that a system with more than one external forces
### is accurately formed with Lagrange method (see issue #8626)

def test_lagrange_2forces():
    ### Equations for two damped springs in series with two forces

    ### generalized coordinates
    q1, q2 = dynamicsymbols('q1, q2')
    ### generalized speeds
    q1d, q2d = dynamicsymbols('q1, q2', 1)

    ### Mass, spring strength, friction coefficient
    m, k, nu = symbols('m, k, nu')

    N = ReferenceFrame('N')
    O = Point('O')

    ### Two points
    P1 = O.locatenew('P1', q1 * N.x)
    P1.set_vel(N, q1d * N.x)
    P2 = O.locatenew('P1', q2 * N.x)
    P2.set_vel(N, q2d * N.x)

    pP1 = Particle('pP1', P1, m)
    pP1.potential_energy = k * q1**2 / 2

    pP2 = Particle('pP2', P2, m)
    pP2.potential_energy = k * (q1 - q2)**2 / 2

    #### Friction forces
    forcelist = [(P1, - nu * q1d * N.x),
                 (P2, - nu * q2d * N.x)]
    lag = Lagrangian(N, pP1, pP2)

    l_method = LagrangesMethod(lag, (q1, q2), forcelist=forcelist, frame=N)
    l_method.form_lagranges_equations()

    eq1 = l_method.eom[0]
    assert eq1.diff(q1d) == nu
    eq2 = l_method.eom[1]
    assert eq2.diff(q2d) == nu

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_size.py ===
import pytest

import numpy as np

from . import util


class TestSizeSumExample(util.F2PyTest):
    sources = [util.getpath("tests", "src", "size", "foo.f90")]

    @pytest.mark.slow
    def test_all(self):
        r = self.module.foo([[]])
        assert r == [0]

        r = self.module.foo([[1, 2]])
        assert r == [3]

        r = self.module.foo([[1, 2], [3, 4]])
        assert np.allclose(r, [3, 7])

        r = self.module.foo([[1, 2], [3, 4], [5, 6]])
        assert np.allclose(r, [3, 7, 11])

    @pytest.mark.slow
    def test_transpose(self):
        r = self.module.trans([[]])
        assert np.allclose(r.T, np.array([[]]))

        r = self.module.trans([[1, 2]])
        assert np.allclose(r, [[1.], [2.]])

        r = self.module.trans([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(r, [[1, 4], [2, 5], [3, 6]])

    @pytest.mark.slow
    def test_flatten(self):
        r = self.module.flatten([[]])
        assert np.allclose(r, [])

        r = self.module.flatten([[1, 2]])
        assert np.allclose(r, [1, 2])

        r = self.module.flatten([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(r, [1, 2, 3, 4, 5, 6])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\modules.py ===
import numpy as np
from numpy import f2py

np.char
np.ctypeslib
np.emath
np.fft
np.lib
np.linalg
np.ma
np.matrixlib
np.polynomial
np.random
np.rec
np.strings
np.testing
np.version

np.lib.format
np.lib.mixins
np.lib.scimath
np.lib.stride_tricks
np.lib.array_utils
np.ma.extras
np.polynomial.chebyshev
np.polynomial.hermite
np.polynomial.hermite_e
np.polynomial.laguerre
np.polynomial.legendre
np.polynomial.polynomial

np.__path__
np.__version__

np.__all__
np.char.__all__
np.ctypeslib.__all__
np.emath.__all__
np.lib.__all__
np.ma.__all__
np.random.__all__
np.rec.__all__
np.strings.__all__
np.testing.__all__
f2py.__all__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\test_indexing.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
)
import pandas._testing as tm

dtlike_dtypes = [
    np.dtype("timedelta64[ns]"),
    np.dtype("datetime64[ns]"),
    pd.DatetimeTZDtype("ns", "Asia/Tokyo"),
    pd.PeriodDtype("ns"),
]


@pytest.mark.parametrize("ldtype", dtlike_dtypes)
@pytest.mark.parametrize("rdtype", dtlike_dtypes)
def test_get_indexer_non_unique_wrong_dtype(ldtype, rdtype):
    vals = np.tile(3600 * 10**9 * np.arange(3, dtype=np.int64), 2)

    def construct(dtype):
        if dtype is dtlike_dtypes[-1]:
            # PeriodArray will try to cast ints to strings
            return DatetimeIndex(vals).astype(dtype)
        return Index(vals, dtype=dtype)

    left = construct(ldtype)
    right = construct(rdtype)

    result = left.get_indexer_non_unique(right)

    if ldtype is rdtype:
        ex1 = np.array([0, 3, 1, 4, 2, 5] * 2, dtype=np.intp)
        ex2 = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(result[0], ex1)
        tm.assert_numpy_array_equal(result[1], ex2)

    else:
        no_matches = np.array([-1] * 6, dtype=np.intp)
        missing = np.arange(6, dtype=np.intp)
        tm.assert_numpy_array_equal(result[0], no_matches)
        tm.assert_numpy_array_equal(result[1], missing)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_pickle.py ===
import pytest

from pandas import (
    NaT,
    date_range,
    to_datetime,
)
import pandas._testing as tm


class TestPickle:
    def test_pickle(self):
        # GH#4606
        idx = to_datetime(["2013-01-01", NaT, "2014-01-06"])
        idx_p = tm.round_trip_pickle(idx)
        assert idx_p[0] == idx[0]
        assert idx_p[1] is NaT
        assert idx_p[2] == idx[2]

    def test_pickle_dont_infer_freq(self):
        # GH#11002
        # don't infer freq
        idx = date_range("1750-1-1", "2050-1-1", freq="7D")
        idx_p = tm.round_trip_pickle(idx)
        tm.assert_index_equal(idx, idx_p)

    def test_pickle_after_set_freq(self):
        dti = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        dti = dti._with_freq(None)

        res = tm.round_trip_pickle(dti)
        tm.assert_index_equal(res, dti)

    def test_roundtrip_pickle_with_tz(self):
        # GH#8367
        # round-trip of timezone
        index = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        unpickled = tm.round_trip_pickle(index)
        tm.assert_index_equal(index, unpickled)

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_pickle_unpickle(self, freq):
        rng = date_range("2009-01-01", "2010-01-01", freq=freq)
        unpickled = tm.round_trip_pickle(rng)
        assert unpickled.freq == freq

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_to_julian_date.py ===
import numpy as np

from pandas import (
    Index,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestDateTimeIndexToJulianDate:
    def test_1700(self):
        dr = date_range(start=Timestamp("1710-10-01"), periods=5, freq="D")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_2000(self):
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="D")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_hour(self):
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="h")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_minute(self):
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="min")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

    def test_second(self):
        dr = date_range(start=Timestamp("2000-02-27"), periods=5, freq="s")
        r1 = Index([x.to_julian_date() for x in dr])
        r2 = dr.to_julian_date()
        assert isinstance(r2, Index) and r2.dtype == np.float64
        tm.assert_index_equal(r1, r2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_set_value.py ===
from datetime import datetime

import numpy as np

from pandas import (
    DatetimeIndex,
    Series,
)
import pandas._testing as tm


def test_series_set_value():
    # GH#1561

    dates = [datetime(2001, 1, 1), datetime(2001, 1, 2)]
    index = DatetimeIndex(dates)

    s = Series(dtype=object)
    s._set_value(dates[0], 1.0)
    s._set_value(dates[1], np.nan)

    expected = Series([1.0, np.nan], index=index)

    tm.assert_series_equal(s, expected)


def test_set_value_dt64(datetime_series):
    idx = datetime_series.index[10]
    res = datetime_series._set_value(idx, 0)
    assert res is None
    assert datetime_series[idx] == 0


def test_set_value_str_index(string_series):
    # equiv
    ser = string_series.copy()
    res = ser._set_value("foobar", 0)
    assert res is None
    assert ser.index[-1] == "foobar"
    assert ser["foobar"] == 0

    ser2 = string_series.copy()
    ser2.loc["foobar"] = 0
    assert ser2.index[-1] == "foobar"
    assert ser2["foobar"] == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\fail_switch_three_greenlets.py ===
"""
Uses a trace function to switch greenlets at unexpected times.

In the trace function, we switch from the current greenlet to another
greenlet, which switches
"""
import greenlet

g1 = None
g2 = None

switch_to_g2 = False

def tracefunc(*args):
    print('TRACE', *args)
    global switch_to_g2
    if switch_to_g2:
        switch_to_g2 = False
        g2.switch()
    print('\tLEAVE TRACE', *args)

def g1_run():
    print('In g1_run')
    global switch_to_g2
    switch_to_g2 = True
    from_parent = greenlet.getcurrent().parent.switch()
    print('Return to g1_run')
    print('From parent', from_parent)

def g2_run():
    #g1.switch()
    greenlet.getcurrent().parent.switch()

greenlet.settrace(tracefunc)

g1 = greenlet.greenlet(g1_run)
g2 = greenlet.greenlet(g2_run)

# This switch didn't actually finish!
# And if it did, it would raise TypeError
# because g1_run() doesn't take any arguments.
g1.switch(1)
print('Back in main')
g1.switch(2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\datetimes\test_cumulative.py ===
import pytest

import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestAccumulator:
    def test_accumulators_freq(self):
        # GH#50297
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            dtype="M8[ns]",
        )._with_freq("infer")
        result = arr._accumulate("cummin")
        expected = DatetimeArray._from_sequence(["2000-01-01"] * 3, dtype="M8[ns]")
        tm.assert_datetime_array_equal(result, expected)

        result = arr._accumulate("cummax")
        expected = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            dtype="M8[ns]",
        )
        tm.assert_datetime_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["cumsum", "cumprod"])
    def test_accumulators_disallowed(self, func):
        # GH#50297
        arr = DatetimeArray._from_sequence(
            [
                "2000-01-01",
                "2000-01-02",
            ],
            dtype="M8[ns]",
        )._with_freq("infer")
        with pytest.raises(TypeError, match=f"Accumulation {func}"):
            arr._accumulate(func)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_join.py ===
import pytest

from pandas import (
    IntervalIndex,
    MultiIndex,
    RangeIndex,
)
import pandas._testing as tm


@pytest.fixture
def range_index():
    return RangeIndex(3, name="range_index")


@pytest.fixture
def interval_index():
    return IntervalIndex.from_tuples(
        [(0.0, 1.0), (1.0, 2.0), (1.5, 2.5)], name="interval_index"
    )


def test_join_overlapping_in_mi_to_same_intervalindex(range_index, interval_index):
    #  GH-45661
    multi_index = MultiIndex.from_product([interval_index, range_index])
    result = multi_index.join(interval_index)

    tm.assert_index_equal(result, multi_index)


def test_join_overlapping_to_multiindex_with_same_interval(range_index, interval_index):
    #  GH-45661
    multi_index = MultiIndex.from_product([interval_index, range_index])
    result = interval_index.join(multi_index)

    tm.assert_index_equal(result, multi_index)


def test_join_overlapping_interval_to_another_intervalindex(interval_index):
    #  GH-45661
    flipped_interval_index = interval_index[::-1]
    result = interval_index.join(flipped_interval_index)

    tm.assert_index_equal(result, interval_index)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_exceptions.py ===
import pytest

jinja2 = pytest.importorskip("jinja2")

from pandas import (
    DataFrame,
    MultiIndex,
)

from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    return DataFrame(
        data=[[0, -0.609], [1, -1.228]],
        columns=["A", "B"],
        index=["x", "y"],
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)


def test_concat_bad_columns(styler):
    msg = "`other.data` must have same columns as `Styler.data"
    with pytest.raises(ValueError, match=msg):
        styler.concat(DataFrame([[1, 2]]).style)


def test_concat_bad_type(styler):
    msg = "`other` must be of type `Styler`"
    with pytest.raises(TypeError, match=msg):
        styler.concat(DataFrame([[1, 2]]))


def test_concat_bad_index_levels(styler, df):
    df = df.copy()
    df.index = MultiIndex.from_tuples([(0, 0), (1, 1)])
    msg = "number of index levels must be same in `other`"
    with pytest.raises(ValueError, match=msg):
        styler.concat(df.style)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_scipy_nodes.py ===
from itertools import product
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos
from sympy.core.numbers import pi
from sympy.codegen.scipy_nodes import cosm1, powm1

x, y, z = symbols('x y z')


def test_cosm1():
    cm1_xy = cosm1(x*y)
    ref_xy = cos(x*y) - 1
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert (
            cm1_xy.diff(wrt, deriv_order) -
            ref_xy.diff(wrt, deriv_order)
        ).rewrite(cos).simplify() == 0

    expr_minus2 = cosm1(pi)
    assert expr_minus2.rewrite(cos) == -2
    assert cosm1(3.14).simplify() == cosm1(3.14)  # cannot simplify with 3.14
    assert cosm1(pi/2).simplify() == -1
    assert (1/cos(x) - 1 + cosm1(x)/cos(x)).simplify() == 0


def test_powm1():
    cases = {
            powm1(x, y): x**y - 1,
            powm1(x*y, z): (x*y)**z - 1,
            powm1(x, y*z): x**(y*z)-1,
            powm1(x*y*z, x*y*z): (x*y*z)**(x*y*z)-1
    }
    for pm1_e, ref_e in cases.items():
        for wrt, deriv_order in product([x, y, z], range(3)):
            der = pm1_e.diff(wrt, deriv_order)
            ref = ref_e.diff(wrt, deriv_order)
            delta = (der - ref).rewrite(Pow)
            assert delta.simplify() == 0

    eulers_constant_m1 = powm1(x, 1/log(x))
    assert eulers_constant_m1.rewrite(Pow) == exp(1) - 1
    assert eulers_constant_m1.simplify() == exp(1) - 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_fourier.py ===
from sympy.assumptions.ask import (Q, ask)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.core.symbol import symbols
from sympy.matrices.expressions.fourier import DFT, IDFT
from sympy.matrices import det, Matrix, Identity
from sympy.testing.pytest import raises


def test_dft_creation():
    assert DFT(2)
    assert DFT(0)
    raises(ValueError, lambda: DFT(-1))
    raises(ValueError, lambda: DFT(2.0))
    raises(ValueError, lambda: DFT(2 + 1j))

    n = symbols('n')
    assert DFT(n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: DFT(n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: DFT(n))


def test_dft():
    n, i, j = symbols('n i j')
    assert DFT(4).shape == (4, 4)
    assert ask(Q.unitary(DFT(4)))
    assert Abs(simplify(det(Matrix(DFT(4))))) == 1
    assert DFT(n)*IDFT(n) == Identity(n)
    assert DFT(n)[i, j] == exp(-2*S.Pi*I/n)**(i*j) / sqrt(n)


def test_dft2():
    assert DFT(1).as_explicit() == Matrix([[1]])
    assert DFT(2).as_explicit() == 1/sqrt(2)*Matrix([[1,1],[1,-1]])
    assert DFT(4).as_explicit() == Matrix([[S.Half,  S.Half,  S.Half, S.Half],
                                           [S.Half, -I/2, Rational(-1,2),  I/2],
                                           [S.Half, Rational(-1,2),  S.Half, Rational(-1,2)],
                                           [S.Half,  I/2, Rational(-1,2), -I/2]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_pyf_src.py ===
# This test is ported from numpy.distutils
from numpy.f2py._src_pyf import process_str
from numpy.testing import assert_equal

pyf_src = """
python module foo
    <_rd=real,double precision>
    interface
        subroutine <s,d>foosub(tol)
            <_rd>, intent(in,out) :: tol
        end subroutine <s,d>foosub
    end interface
end python module foo
"""

expected_pyf = """
python module foo
    interface
        subroutine sfoosub(tol)
            real, intent(in,out) :: tol
        end subroutine sfoosub
        subroutine dfoosub(tol)
            double precision, intent(in,out) :: tol
        end subroutine dfoosub
    end interface
end python module foo
"""


def normalize_whitespace(s):
    """
    Remove leading and trailing whitespace, and convert internal
    stretches of whitespace to a single space.
    """
    return ' '.join(s.split())


def test_from_template():
    """Regression test for gh-10712."""
    pyf = process_str(pyf_src)
    normalized_pyf = normalize_whitespace(pyf)
    normalized_expected_pyf = normalize_whitespace(expected_pyf)
    assert_equal(normalized_pyf, normalized_expected_pyf)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\array_like.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy._typing import NDArray, ArrayLike, _SupportsArray

x1: ArrayLike = True
x2: ArrayLike = 5
x3: ArrayLike = 1.0
x4: ArrayLike = 1 + 1j
x5: ArrayLike = np.int8(1)
x6: ArrayLike = np.float64(1)
x7: ArrayLike = np.complex128(1)
x8: ArrayLike = np.array([1, 2, 3])
x9: ArrayLike = [1, 2, 3]
x10: ArrayLike = (1, 2, 3)
x11: ArrayLike = "foo"
x12: ArrayLike = memoryview(b'foo')


class A:
    def __array__(self, dtype: np.dtype | None = None) -> NDArray[np.float64]:
        return np.array([1.0, 2.0, 3.0])


x13: ArrayLike = A()

scalar: _SupportsArray[np.dtype[np.int64]] = np.int64(1)
scalar.__array__()
array: _SupportsArray[np.dtype[np.int_]] = np.array(1)
array.__array__()

a: _SupportsArray[np.dtype[np.float64]] = A()
a.__array__()
a.__array__()

# Escape hatch for when you mean to make something like an object
# array.
object_array_scalar: object = (i for i in range(10))
np.array(object_array_scalar)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_s3.py ===
from io import BytesIO

import pytest

from pandas import read_csv


def test_streaming_s3_objects():
    # GH17135
    # botocore gained iteration support in 1.10.47, can now be used in read_*
    pytest.importorskip("botocore", minversion="1.10.47")
    from botocore.response import StreamingBody

    data = [b"foo,bar,baz\n1,2,3\n4,5,6\n", b"just,the,header\n"]
    for el in data:
        body = StreamingBody(BytesIO(el), content_length=len(el))
        read_csv(body)


@pytest.mark.single_cpu
def test_read_without_creds_from_pub_bucket(s3_public_bucket_with_data, s3so):
    # GH 34626
    pytest.importorskip("s3fs")
    result = read_csv(
        f"s3://{s3_public_bucket_with_data.name}/tips.csv",
        nrows=3,
        storage_options=s3so,
    )
    assert len(result) == 3


@pytest.mark.single_cpu
def test_read_with_creds_from_pub_bucket(s3_public_bucket_with_data, s3so):
    # Ensure we can read from a public bucket with credentials
    # GH 34626
    pytest.importorskip("s3fs")
    df = read_csv(
        f"s3://{s3_public_bucket_with_data.name}/tips.csv",
        nrows=5,
        header=None,
        storage_options=s3so,
    )
    assert len(df) == 5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_operators.py ===
from sympy.vector import CoordSys3D, Gradient, Divergence, Curl, VectorZero, Laplacian
from sympy.printing.repr import srepr

R = CoordSys3D('R')
s1 = R.x*R.y*R.z  # type: ignore
s2 = R.x + 3*R.y**2  # type: ignore
s3 = R.x**2 + R.y**2 + R.z**2  # type: ignore
v1 = R.x*R.i + R.z*R.z*R.j  # type: ignore
v2 = R.x*R.i + R.y*R.j + R.z*R.k  # type: ignore
v3 = R.x**2*R.i + R.y**2*R.j + R.z**2*R.k  # type: ignore


def test_Gradient():
    assert Gradient(s1) == Gradient(R.x*R.y*R.z)
    assert Gradient(s2) == Gradient(R.x + 3*R.y**2)
    assert Gradient(s1).doit() == R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    assert Gradient(s2).doit() == R.i + 6*R.y*R.j


def test_Divergence():
    assert Divergence(v1) == Divergence(R.x*R.i + R.z*R.z*R.j)
    assert Divergence(v2) == Divergence(R.x*R.i + R.y*R.j + R.z*R.k)
    assert Divergence(v1).doit() == 1
    assert Divergence(v2).doit() == 3
    # issue 22384
    Rc = CoordSys3D('R', transformation='cylindrical')
    assert Divergence(Rc.i).doit() == 1/Rc.r


def test_Curl():
    assert Curl(v1) == Curl(R.x*R.i + R.z*R.z*R.j)
    assert Curl(v2) == Curl(R.x*R.i + R.y*R.j + R.z*R.k)
    assert Curl(v1).doit() == (-2*R.z)*R.i
    assert Curl(v2).doit() == VectorZero()


def test_Laplacian():
    assert Laplacian(s3) == Laplacian(R.x**2 + R.y**2 + R.z**2)
    assert Laplacian(v3) == Laplacian(R.x**2*R.i + R.y**2*R.j + R.z**2*R.k)
    assert Laplacian(s3).doit() == 6
    assert Laplacian(v3).doit() == 2*R.i + 2*R.j + 2*R.k
    assert srepr(Laplacian(s3)) == \
            'Laplacian(Add(Pow(R.x, Integer(2)), Pow(R.y, Integer(2)), Pow(R.z, Integer(2))))'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\period\test_reductions.py ===
import pytest

import pandas as pd
from pandas.core.arrays import period_array


class TestReductions:
    def test_min_max(self):
        arr = period_array(
            [
                "2000-01-03",
                "2000-01-03",
                "NaT",
                "2000-01-02",
                "2000-01-05",
                "2000-01-04",
            ],
            freq="D",
        )

        result = arr.min()
        expected = pd.Period("2000-01-02", freq="D")
        assert result == expected

        result = arr.max()
        expected = pd.Period("2000-01-05", freq="D")
        assert result == expected

        result = arr.min(skipna=False)
        assert result is pd.NaT

        result = arr.max(skipna=False)
        assert result is pd.NaT

    @pytest.mark.parametrize("skipna", [True, False])
    def test_min_max_empty(self, skipna):
        arr = period_array([], freq="D")
        result = arr.min(skipna=skipna)
        assert result is pd.NaT

        result = arr.max(skipna=skipna)
        assert result is pd.NaT

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_infer_objects.py ===
from datetime import datetime

from pandas import DataFrame
import pandas._testing as tm


class TestInferObjects:
    def test_infer_objects(self):
        # GH#11221
        df = DataFrame(
            {
                "a": ["a", 1, 2, 3],
                "b": ["b", 2.0, 3.0, 4.1],
                "c": [
                    "c",
                    datetime(2016, 1, 1),
                    datetime(2016, 1, 2),
                    datetime(2016, 1, 3),
                ],
                "d": [1, 2, 3, "d"],
            },
            columns=["a", "b", "c", "d"],
        )
        df = df.iloc[1:].infer_objects()

        assert df["a"].dtype == "int64"
        assert df["b"].dtype == "float64"
        assert df["c"].dtype == "M8[ns]"
        assert df["d"].dtype == "object"

        expected = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [2.0, 3.0, 4.1],
                "c": [datetime(2016, 1, 1), datetime(2016, 1, 2), datetime(2016, 1, 3)],
                "d": [2, 3, "d"],
            },
            columns=["a", "b", "c", "d"],
        )
        # reconstruct frame to verify inference is same
        result = df.reset_index(drop=True)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_monotonic.py ===
from pandas import (
    Period,
    PeriodIndex,
)


def test_is_monotonic_increasing():
    # GH#17717
    p0 = Period("2017-09-01")
    p1 = Period("2017-09-02")
    p2 = Period("2017-09-03")

    idx_inc0 = PeriodIndex([p0, p1, p2])
    idx_inc1 = PeriodIndex([p0, p1, p1])
    idx_dec0 = PeriodIndex([p2, p1, p0])
    idx_dec1 = PeriodIndex([p2, p1, p1])
    idx = PeriodIndex([p1, p2, p0])

    assert idx_inc0.is_monotonic_increasing is True
    assert idx_inc1.is_monotonic_increasing is True
    assert idx_dec0.is_monotonic_increasing is False
    assert idx_dec1.is_monotonic_increasing is False
    assert idx.is_monotonic_increasing is False


def test_is_monotonic_decreasing():
    # GH#17717
    p0 = Period("2017-09-01")
    p1 = Period("2017-09-02")
    p2 = Period("2017-09-03")

    idx_inc0 = PeriodIndex([p0, p1, p2])
    idx_inc1 = PeriodIndex([p0, p1, p1])
    idx_dec0 = PeriodIndex([p2, p1, p0])
    idx_dec1 = PeriodIndex([p2, p1, p1])
    idx = PeriodIndex([p1, p2, p0])

    assert idx_inc0.is_monotonic_decreasing is False
    assert idx_inc1.is_monotonic_decreasing is False
    assert idx_dec0.is_monotonic_decreasing is True
    assert idx_dec1.is_monotonic_decreasing is True
    assert idx.is_monotonic_decreasing is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_sets.py ===
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.sets import MatrixSet
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.testing.pytest import raises
from sympy.sets.sets import SetKind
from sympy.matrices.kind import MatrixKind
from sympy.core.kind import NumberKind


def test_MatrixSet():
    n, m = symbols('n m', integer=True)
    A = MatrixSymbol('A', n, m)
    C = MatrixSymbol('C', n, n)

    M = MatrixSet(2, 2, set=S.Reals)
    assert M.shape == (2, 2)
    assert M.set == S.Reals
    X = Matrix([[1, 2], [3, 4]])
    assert X in M
    X = ZeroMatrix(2, 2)
    assert X in M
    raises(TypeError, lambda: A in M)
    raises(TypeError, lambda: 1 in M)
    M = MatrixSet(n, m, set=S.Reals)
    assert A in M
    raises(TypeError, lambda: C in M)
    raises(TypeError, lambda: X in M)
    M = MatrixSet(2, 2, set={1, 2, 3})
    X = Matrix([[1, 2], [3, 4]])
    Y = Matrix([[1, 2]])
    assert (X in M) == S.false
    assert (Y in M) == S.false
    raises(ValueError, lambda: MatrixSet(2, -2, S.Reals))
    raises(ValueError, lambda: MatrixSet(2.4, -1, S.Reals))
    raises(TypeError, lambda: MatrixSet(2, 2, (1, 2, 3)))


def test_SetKind_MatrixSet():
    assert MatrixSet(2, 2, set=S.Reals).kind is SetKind(MatrixKind(NumberKind))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\branch\tests\test_tools.py ===
from sympy.strategies.branch.tools import canon
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.singleton import S


def posdec(x):
    if isinstance(x, Integer) and x > 0:
        yield x - 1
    else:
        yield x


def branch5(x):
    if isinstance(x, Integer):
        if 0 < x < 5:
            yield x - 1
        elif 5 < x < 10:
            yield x + 1
        elif x == 5:
            yield x + 1
            yield x - 1
        else:
            yield x


def test_zero_ints():
    expr = Basic(S(2), Basic(S(5), S(3)), S(8))
    expected = {Basic(S(0), Basic(S(0), S(0)), S(0))}

    brl = canon(posdec)
    assert set(brl(expr)) == expected


def test_split5():
    expr = Basic(S(2), Basic(S(5), S(3)), S(8))
    expected = {
        Basic(S(0), Basic(S(0), S(0)), S(10)),
        Basic(S(0), Basic(S(10), S(0)), S(10))}

    brl = canon(branch5)
    assert set(brl(expr)) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\tests\test_module_imports.py ===
"""
Checks that SymPy does not contain indirect imports.

An indirect import is importing a symbol from a module that itself imported the
symbol from elsewhere. Such a constellation makes it harder to diagnose
inter-module dependencies and import order problems, and is therefore strongly
discouraged.

(Indirect imports from end-user code is fine and in fact a best practice.)

Implementation note: Forcing Python into actually unloading already-imported
submodules is a tricky and partly undocumented process. To avoid these issues,
the actual diagnostic code is in bin/diagnose_imports, which is run as a
separate, pristine Python process.
"""

import subprocess
import sys
from os.path import abspath, dirname, join, normpath
import inspect

from sympy.testing.pytest import XFAIL

@XFAIL
def test_module_imports_are_direct():
    my_filename = abspath(inspect.getfile(inspect.currentframe()))
    my_dirname = dirname(my_filename)
    diagnose_imports_filename = join(my_dirname, 'diagnose_imports.py')
    diagnose_imports_filename = normpath(diagnose_imports_filename)

    process = subprocess.Popen(
        [
            sys.executable,
            normpath(diagnose_imports_filename),
            '--problems',
            '--by-importer'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=-1)
    output, _ = process.communicate()
    assert output == '', "There are import problems:\n" + output.decode()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\fail_switch_two_greenlets.py ===
"""
Uses a trace function to switch greenlets at unexpected times.

In the trace function, we switch from the current greenlet to another
greenlet, which switches
"""
import greenlet

g1 = None
g2 = None

switch_to_g2 = False

def tracefunc(*args):
    print('TRACE', *args)
    global switch_to_g2
    if switch_to_g2:
        switch_to_g2 = False
        g2.switch()
    print('\tLEAVE TRACE', *args)

def g1_run():
    print('In g1_run')
    global switch_to_g2
    switch_to_g2 = True
    greenlet.getcurrent().parent.switch()
    print('Return to g1_run')
    print('Falling off end of g1_run')

def g2_run():
    g1.switch()
    print('Falling off end of g2')

greenlet.settrace(tracefunc)

g1 = greenlet.greenlet(g1_run)
g2 = greenlet.greenlet(g2_run)

g1.switch()
print('Falling off end of main')
g2.switch()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_version.py ===
#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
from unittest import TestCase as NonLeakingTestCase

import greenlet

# No reason to run this multiple times under leakchecks,
# it doesn't do anything.
class VersionTests(NonLeakingTestCase):
    def test_version(self):
        def find_dominating_file(name):
            if os.path.exists(name):
                return name

            tried = []
            here = os.path.abspath(os.path.dirname(__file__))
            for i in range(10):
                up = ['..'] * i
                path = [here] + up + [name]
                fname = os.path.join(*path)
                fname = os.path.abspath(fname)
                tried.append(fname)
                if os.path.exists(fname):
                    return fname
            raise AssertionError("Could not find file " + name + "; checked " + str(tried))

        try:
            setup_py = find_dominating_file('setup.py')
        except AssertionError as e:
            self.skipTest("Unable to find setup.py; must be out of tree. " + str(e))


        invoke_setup = "%s %s --version" % (sys.executable, setup_py)
        with os.popen(invoke_setup) as f:
            sversion = f.read().strip()

        self.assertEqual(sversion, greenlet.__version__)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\numpy_\test_indexing.py ===
import numpy as np

from pandas.core.dtypes.common import is_scalar

import pandas as pd
import pandas._testing as tm


class TestSearchsorted:
    def test_searchsorted_string(self, string_dtype):
        arr = pd.array(["a", "b", "c"], dtype=string_dtype)

        result = arr.searchsorted("a", side="left")
        assert is_scalar(result)
        assert result == 0

        result = arr.searchsorted("a", side="right")
        assert is_scalar(result)
        assert result == 1

    def test_searchsorted_numeric_dtypes_scalar(self, any_real_numpy_dtype):
        arr = pd.array([1, 3, 90], dtype=any_real_numpy_dtype)
        result = arr.searchsorted(30)
        assert is_scalar(result)
        assert result == 2

        result = arr.searchsorted([30])
        expected = np.array([2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_searchsorted_numeric_dtypes_vector(self, any_real_numpy_dtype):
        arr = pd.array([1, 3, 90], dtype=any_real_numpy_dtype)
        result = arr.searchsorted([2, 30])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_searchsorted_sorter(self, any_real_numpy_dtype):
        arr = pd.array([3, 1, 2], dtype=any_real_numpy_dtype)
        result = arr.searchsorted([0, 3], sorter=np.argsort(arr))
        expected = np.array([0, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\printing.py ===
import io

import pytest

import pandas as pd


class BasePrintingTests:
    """Tests checking the formatting of your EA when printed."""

    @pytest.mark.parametrize("size", ["big", "small"])
    def test_array_repr(self, data, size):
        if size == "small":
            data = data[:5]
        else:
            data = type(data)._concat_same_type([data] * 5)

        result = repr(data)
        assert type(data).__name__ in result
        assert f"Length: {len(data)}" in result
        assert str(data.dtype) in result
        if size == "big":
            assert "..." in result

    def test_array_repr_unicode(self, data):
        result = str(data)
        assert isinstance(result, str)

    def test_series_repr(self, data):
        ser = pd.Series(data)
        assert data.dtype.name in repr(ser)

    def test_dataframe_repr(self, data):
        df = pd.DataFrame({"A": data})
        repr(df)

    def test_dtype_name_in_info(self, data):
        buf = io.StringIO()
        pd.DataFrame({"A": data}).info(buf=buf)
        result = buf.getvalue()
        assert data.dtype.name in result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_validate.py ===
import pytest

from pandas.core.frame import DataFrame


@pytest.fixture
def dataframe():
    return DataFrame({"a": [1, 2], "b": [3, 4]})


class TestDataFrameValidate:
    """Tests for error handling related to data types of method arguments."""

    @pytest.mark.parametrize(
        "func",
        [
            "query",
            "eval",
            "set_index",
            "reset_index",
            "dropna",
            "drop_duplicates",
            "sort_values",
        ],
    )
    @pytest.mark.parametrize("inplace", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, dataframe, func, inplace):
        msg = 'For argument "inplace" expected type bool'
        kwargs = {"inplace": inplace}

        if func == "query":
            kwargs["expr"] = "a > b"
        elif func == "eval":
            kwargs["expr"] = "a + b"
        elif func == "set_index":
            kwargs["keys"] = ["a"]
        elif func == "sort_values":
            kwargs["by"] = ["a"]

        with pytest.raises(ValueError, match=msg):
            getattr(dataframe, func)(**kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\conftest.py ===
import numpy as np
import pytest

from pandas import (
    Series,
    array,
)


@pytest.fixture(params=[None, False])
def sort(request):
    """
    Valid values for the 'sort' parameter used in the Index
    setops methods (intersection, union, etc.)

    Caution:
        Don't confuse this one with the "sort" fixture used
        for DataFrame.append or concat. That one has
        parameters [True, False].

        We can't combine them as sort=True is not permitted
        in the Index setops methods.
    """
    return request.param


@pytest.fixture(params=["D", "3D", "-3D", "h", "2h", "-2h", "min", "2min", "s", "-3s"])
def freq_sample(request):
    """
    Valid values for 'freq' parameter used to create date_range and
    timedelta_range..
    """
    return request.param


@pytest.fixture(params=[list, tuple, np.array, array, Series])
def listlike_box(request):
    """
    Types that may be passed as the indexer to searchsorted.
    """
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_factorize.py ===
import numpy as np

from pandas import PeriodIndex
import pandas._testing as tm


class TestFactorize:
    def test_factorize_period(self):
        idx1 = PeriodIndex(
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"],
            freq="M",
        )

        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")

        arr, idx = idx1.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

        arr, idx = idx1.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

    def test_factorize_period_nonmonotonic(self):
        idx2 = PeriodIndex(
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"],
            freq="M",
        )
        exp_idx = PeriodIndex(["2014-01", "2014-02", "2014-03"], freq="M")

        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.intp)
        arr, idx = idx2.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

        exp_arr = np.array([0, 0, 1, 2, 0, 2], dtype=np.intp)
        exp_idx = PeriodIndex(["2014-03", "2014-02", "2014-01"], freq="M")
        arr, idx = idx2.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_fillna.py ===
from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
)
import pandas._testing as tm


class TestFillNA:
    def test_fillna_period(self):
        # GH#11343
        idx = PeriodIndex(["2011-01-01 09:00", NaT, "2011-01-01 11:00"], freq="h")

        exp = PeriodIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"], freq="h"
        )
        result = idx.fillna(Period("2011-01-01 10:00", freq="h"))
        tm.assert_index_equal(result, exp)

        exp = Index(
            [
                Period("2011-01-01 09:00", freq="h"),
                "x",
                Period("2011-01-01 11:00", freq="h"),
            ],
            dtype=object,
        )
        result = idx.fillna("x")
        tm.assert_index_equal(result, exp)

        exp = Index(
            [
                Period("2011-01-01 09:00", freq="h"),
                Period("2011-01-01", freq="D"),
                Period("2011-01-01 11:00", freq="h"),
            ],
            dtype=object,
        )
        result = idx.fillna(Period("2011-01-01", freq="D"))
        tm.assert_index_equal(result, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_add_prefix_suffix.py ===
import pytest

from pandas import Index
import pandas._testing as tm


def test_add_prefix_suffix(string_series):
    with_prefix = string_series.add_prefix("foo#")
    expected = Index([f"foo#{c}" for c in string_series.index])
    tm.assert_index_equal(with_prefix.index, expected)

    with_suffix = string_series.add_suffix("#foo")
    expected = Index([f"{c}#foo" for c in string_series.index])
    tm.assert_index_equal(with_suffix.index, expected)

    with_pct_prefix = string_series.add_prefix("%")
    expected = Index([f"%{c}" for c in string_series.index])
    tm.assert_index_equal(with_pct_prefix.index, expected)

    with_pct_suffix = string_series.add_suffix("%")
    expected = Index([f"{c}%" for c in string_series.index])
    tm.assert_index_equal(with_pct_suffix.index, expected)


def test_add_prefix_suffix_axis(string_series):
    # GH 47819
    with_prefix = string_series.add_prefix("foo#", axis=0)
    expected = Index([f"foo#{c}" for c in string_series.index])
    tm.assert_index_equal(with_prefix.index, expected)

    with_pct_suffix = string_series.add_suffix("#foo", axis=0)
    expected = Index([f"{c}#foo" for c in string_series.index])
    tm.assert_index_equal(with_pct_suffix.index, expected)


def test_add_prefix_suffix_invalid_axis(string_series):
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        string_series.add_prefix("foo#", axis=1)

    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        string_series.add_suffix("foo#", axis=1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_reindex_like.py ===
from datetime import datetime

import numpy as np

from pandas import Series
import pandas._testing as tm


def test_reindex_like(datetime_series):
    other = datetime_series[::2]
    tm.assert_series_equal(
        datetime_series.reindex(other.index), datetime_series.reindex_like(other)
    )

    # GH#7179
    day1 = datetime(2013, 3, 5)
    day2 = datetime(2013, 5, 5)
    day3 = datetime(2014, 3, 5)

    series1 = Series([5, None, None], [day1, day2, day3])
    series2 = Series([None, None], [day1, day3])

    result = series1.reindex_like(series2, method="pad")
    expected = Series([5, np.nan], index=[day1, day3])
    tm.assert_series_equal(result, expected)


def test_reindex_like_nearest():
    ser = Series(np.arange(10, dtype="int64"))

    target = [0.1, 0.9, 1.5, 2.0]
    other = ser.reindex(target, method="nearest")
    expected = Series(np.around(target).astype("int64"), target)

    result = ser.reindex_like(other, method="nearest")
    tm.assert_series_equal(expected, result)

    result = ser.reindex_like(other, method="nearest", tolerance=1)
    tm.assert_series_equal(expected, result)
    result = ser.reindex_like(other, method="nearest", tolerance=[1, 2, 3, 4])
    tm.assert_series_equal(expected, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\tests\test_recurrence.py ===
from sympy.holonomic.recurrence import RecurrenceOperators, RecurrenceOperator
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ


def test_RecurrenceOperator():
    n = symbols('n', integer=True)
    R, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    assert Sn*n == (n + 1)*Sn
    assert Sn*n**2 == (n**2+1+2*n)*Sn
    assert Sn**2*n**2 == (n**2 + 4*n + 4)*Sn**2
    p = (Sn**3*n**2 + Sn*n)**2
    q = (n**2 + 3*n + 2)*Sn**2 + (2*n**3 + 19*n**2 + 57*n + 52)*Sn**4 + (n**4 + 18*n**3 + \
        117*n**2 + 324*n + 324)*Sn**6
    assert p == q


def test_RecurrenceOperatorEqPoly():
    n = symbols('n', integer=True)
    R, Sn = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    rr = RecurrenceOperator([n**2, 0, 0], R)
    rr2 = RecurrenceOperator([n**2, 1, n], R)
    assert not rr == rr2

    # polynomial comparison issue, see https://github.com/sympy/sympy/pull/15799
    # should work once that is solved
    # d = rr.listofpoly[0]
    # assert rr == d

    d2 = rr2.listofpoly[0]
    assert not rr2 == d2


def test_RecurrenceOperatorPow():
    n = symbols('n', integer=True)
    R, _ = RecurrenceOperators(QQ.old_poly_ring(n), 'Sn')
    rr = RecurrenceOperator([n**2, 0, 0], R)
    a = RecurrenceOperator([R.base.one], R)
    for m in range(10):
        assert a == rr**m
        a *= rr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_linearity_of_velocity_constraints.py ===
from sympy import symbols, sin, cos
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
            KanesMethod)
from sympy.testing import pytest
from sympy.solvers.solveset import NonlinearError

def test_linearity_of_motion_constraints():
    # Test that an error is raised by KanesMethod if nonlinear velocity
    # constraints are supplied.
    # It is a simple pendulum.
    t = dynamicsymbols._t
    N, A = ReferenceFrame('N'), ReferenceFrame('A')
    O, P = Point('O'), Point('P')
    O.set_vel(N, 0)

    l = symbols('l')
    q, x, y, u, ux, uy = dynamicsymbols('q x y u ux uy')

    A.orient_axis(N, q, N.z)
    A.set_ang_vel(N, u * N.z)
    P.set_pos(O, -l * A.y)
    P.v2pt_theory(O, N, A)

    kd = [u - q.diff(t), ux - x.diff(t), uy - y.diff(t)]
    config_constr = [x - l * sin(q), y - l * cos(q)]

    q_ind = [q]
    q_dep = [x, y]
    u_ind = [u]
    u_dep = [ux, uy]

    # Make sure an error is raised if nonlinear velocity constraints are
    # supplied.
    speed_constr = [ux - l * q.diff(t) * cos(q), sin(uy) +
        l * q.diff(t) * sin(q)]

    with pytest.raises(NonlinearError):
        KanesMethod(N, q_ind=q_ind, q_dependent=q_dep, u_ind=u_ind,
            u_dependent=u_dep, kd_eqs=kd,
            configuration_constraints=config_constr,
            velocity_constraints=speed_constr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_pring.py ===
from sympy.physics.pring import wavefunction, energy
from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.abc import m, x, r
from sympy.physics.quantum.constants import hbar


def test_wavefunction():
    Psi = {
        0: (1/sqrt(2 * pi)),
        1: (1/sqrt(2 * pi)) * exp(I * x),
        2: (1/sqrt(2 * pi)) * exp(2 * I * x),
        3: (1/sqrt(2 * pi)) * exp(3 * I * x)
    }
    for n in Psi:
        assert simplify(wavefunction(n, x) - Psi[n]) == 0


def test_norm(n=1):
    # Maximum "n" which is tested:
    for i in range(n + 1):
        assert integrate(
            wavefunction(i, x) * wavefunction(-i, x), (x, 0, 2 * pi)) == 1


def test_orthogonality(n=1):
    # Maximum "n" which is tested:
    for i in range(n + 1):
        for j in range(i+1, n+1):
            assert integrate(
                wavefunction(i, x) * wavefunction(j, x), (x, 0, 2 * pi)) == 0


def test_energy(n=1):
    # Maximum "n" which is tested:
    for i in range(n+1):
        assert simplify(
            energy(i, m, r) - ((i**2 * hbar**2) / (2 * m * r**2))) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\test_arrayobject.py ===
import pytest

import numpy as np
from numpy.ma import masked_array
from numpy.testing import assert_array_equal


def test_matrix_transpose_raises_error_for_1d():
    msg = "matrix transpose with ndim < 2 is undefined"
    ma_arr = masked_array(data=[1, 2, 3, 4, 5, 6],
                          mask=[1, 0, 1, 1, 1, 0])
    with pytest.raises(ValueError, match=msg):
        ma_arr.mT


def test_matrix_transpose_equals_transpose_2d():
    ma_arr = masked_array(data=[[1, 2, 3], [4, 5, 6]],
                          mask=[[1, 0, 1], [1, 1, 0]])
    assert_array_equal(ma_arr.T, ma_arr.mT)


ARRAY_SHAPES_TO_TEST = (
    (5, 2),
    (5, 2, 3),
    (5, 2, 3, 4),
)


@pytest.mark.parametrize("shape", ARRAY_SHAPES_TO_TEST)
def test_matrix_transpose_equals_swapaxes(shape):
    num_of_axes = len(shape)
    vec = np.arange(shape[-1])
    arr = np.broadcast_to(vec, shape)

    rng = np.random.default_rng(42)
    mask = rng.choice([0, 1], size=shape)
    ma_arr = masked_array(data=arr, mask=mask)

    tgt = np.swapaxes(arr, num_of_axes - 2, num_of_axes - 1)
    assert_array_equal(tgt, ma_arr.mT)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_maybe_box_native.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas.core.dtypes.cast import maybe_box_native

from pandas import (
    Interval,
    Period,
    Timedelta,
    Timestamp,
)


@pytest.mark.parametrize(
    "obj,expected_dtype",
    [
        (b"\x00\x10", bytes),
        (int(4), int),
        (np.uint(4), int),
        (np.int32(-4), int),
        (np.uint8(4), int),
        (float(454.98), float),
        (np.float16(0.4), float),
        (np.float64(1.4), float),
        (np.bool_(False), bool),
        (datetime(2005, 2, 25), datetime),
        (np.datetime64("2005-02-25"), Timestamp),
        (Timestamp("2005-02-25"), Timestamp),
        (np.timedelta64(1, "D"), Timedelta),
        (Timedelta(1, "D"), Timedelta),
        (Interval(0, 1), Interval),
        (Period("4Q2005"), Period),
    ],
)
def test_maybe_box_native(obj, expected_dtype):
    boxed_obj = maybe_box_native(obj)
    result_dtype = type(boxed_obj)
    assert result_dtype is expected_dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\accumulate.py ===
import pytest

import pandas as pd
import pandas._testing as tm


class BaseAccumulateTests:
    """
    Accumulation specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # Do we expect this accumulation to be supported for this dtype?
        # We default to assuming "no"; subclass authors should override here.
        return False

    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool):
        try:
            alt = ser.astype("float64")
        except (TypeError, ValueError):
            # e.g. Period can't be cast to float64 (TypeError)
            #      String can't be cast to float64 (ValueError)
            alt = ser.astype(object)

        result = getattr(ser, op_name)(skipna=skipna)
        expected = getattr(alt, op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna):
        op_name = all_numeric_accumulations
        ser = pd.Series(data)

        if self._supports_accumulation(ser, op_name):
            self.check_accumulate(ser, op_name, skipna)
        else:
            with pytest.raises((NotImplementedError, TypeError)):
                # TODO: require TypeError for things that will _never_ work?
                getattr(ser, op_name)(skipna=skipna)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_subclass.py ===
"""
Tests involving custom Index subclasses
"""
import numpy as np

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


class CustomIndex(Index):
    def __new__(cls, data, name=None):
        # assert that this index class cannot hold strings
        if any(isinstance(val, str) for val in data):
            raise TypeError("CustomIndex cannot hold strings")

        if name is None and hasattr(data, "name"):
            name = data.name
        data = np.array(data, dtype="O")

        return cls._simple_new(data, name)


def test_insert_fallback_to_base_index():
    # https://github.com/pandas-dev/pandas/issues/47071

    idx = CustomIndex([1, 2, 3])
    result = idx.insert(0, "string")
    expected = Index(["string", 1, 2, 3], dtype=object)
    tm.assert_index_equal(result, expected)

    df = DataFrame(
        np.random.default_rng(2).standard_normal((2, 3)),
        columns=idx,
        index=Index([1, 2], name="string"),
    )
    result = df.reset_index()
    tm.assert_index_equal(result.columns, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\methods\test_factorize.py ===
import numpy as np

from pandas import (
    TimedeltaIndex,
    factorize,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexFactorize:
    def test_factorize(self):
        idx1 = TimedeltaIndex(["1 day", "1 day", "2 day", "2 day", "3 day", "3 day"])

        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
        exp_idx = TimedeltaIndex(["1 day", "2 day", "3 day"])

        arr, idx = idx1.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
        assert idx.freq == exp_idx.freq

        arr, idx = idx1.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
        assert idx.freq == exp_idx.freq

    def test_factorize_preserves_freq(self):
        # GH#38120 freq should be preserved
        idx3 = timedelta_range("1 day", periods=4, freq="s")
        exp_arr = np.array([0, 1, 2, 3], dtype=np.intp)
        arr, idx = idx3.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, idx3)
        assert idx.freq == idx3.freq

        arr, idx = factorize(idx3)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, idx3)
        assert idx.freq == idx3.freq

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\common.py ===
""" common utilities """
from __future__ import annotations

from typing import (
    Any,
    Literal,
)


def _mklbl(prefix: str, n: int):
    return [f"{prefix}{i}" for i in range(n)]


def check_indexing_smoketest_or_raises(
    obj,
    method: Literal["iloc", "loc"],
    key: Any,
    axes: Literal[0, 1] | None = None,
    fails=None,
) -> None:
    if axes is None:
        axes_list = [0, 1]
    else:
        assert axes in [0, 1]
        axes_list = [axes]

    for ax in axes_list:
        if ax < obj.ndim:
            # create a tuple accessor
            new_axes = [slice(None)] * obj.ndim
            new_axes[ax] = key
            axified = tuple(new_axes)
            try:
                getattr(obj, method).__getitem__(axified)
            except (IndexError, TypeError, KeyError) as detail:
                # if we are in fails, the ok, otherwise raise it
                if fails is not None:
                    if isinstance(detail, fails):
                        return
                raise

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_is_unique.py ===
import numpy as np
import pytest

from pandas import Series


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.random.default_rng(2).integers(0, 10, size=1000), False),
        (np.arange(1000), True),
        ([], True),
        ([np.nan], True),
        (["foo", "bar", np.nan], True),
        (["foo", "foo", np.nan], False),
        (["foo", "bar", np.nan, np.nan], False),
    ],
)
def test_is_unique(data, expected):
    # GH#11946 / GH#25180
    ser = Series(data)
    assert ser.is_unique is expected


def test_is_unique_class_ne(capsys):
    # GH#20661
    class Foo:
        def __init__(self, val) -> None:
            self._value = val

        def __ne__(self, other):
            raise Exception("NEQ not supported")

    with capsys.disabled():
        li = [Foo(i) for i in range(5)]
        ser = Series(li, index=list(range(5)))

    ser.is_unique
    captured = capsys.readouterr()
    assert len(captured.err) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_repeat.py ===
import numpy as np
import pytest

from pandas import (
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestRepeat:
    def test_repeat(self):
        ser = Series(np.random.default_rng(2).standard_normal(3), index=["a", "b", "c"])

        reps = ser.repeat(5)
        exp = Series(ser.values.repeat(5), index=ser.index.values.repeat(5))
        tm.assert_series_equal(reps, exp)

        to_rep = [2, 3, 4]
        reps = ser.repeat(to_rep)
        exp = Series(ser.values.repeat(to_rep), index=ser.index.values.repeat(to_rep))
        tm.assert_series_equal(reps, exp)

    def test_numpy_repeat(self):
        ser = Series(np.arange(3), name="x")
        expected = Series(
            ser.values.repeat(2), name="x", index=ser.index.values.repeat(2)
        )
        tm.assert_series_equal(np.repeat(ser, 2), expected)

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(ser, 2, axis=0)

    def test_repeat_with_multiindex(self):
        # GH#9361, fixed by  GH#7891
        m_idx = MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6), (7, 8)])
        data = ["a", "b", "c", "d"]
        m_df = Series(data, index=m_idx)
        assert m_df.repeat(3).shape == (3 * len(data),)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_fields.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import fields

import pandas._testing as tm


@pytest.fixture
def dtindex():
    dtindex = np.arange(5, dtype=np.int64) * 10**9 * 3600 * 24 * 32
    dtindex.flags.writeable = False
    return dtindex


def test_get_date_name_field_readonly(dtindex):
    # https://github.com/vaexio/vaex/issues/357
    #  fields functions shouldn't raise when we pass read-only data
    result = fields.get_date_name_field(dtindex, "month_name")
    expected = np.array(["January", "February", "March", "April", "May"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_get_date_field_readonly(dtindex):
    result = fields.get_date_field(dtindex, "Y")
    expected = np.array([1970, 1970, 1970, 1970, 1970], dtype=np.int32)
    tm.assert_numpy_array_equal(result, expected)


def test_get_start_end_field_readonly(dtindex):
    result = fields.get_start_end_field(dtindex, "is_month_start", None)
    expected = np.array([True, False, False, False, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)


def test_get_timedelta_field_readonly(dtindex):
    # treat dtindex as timedeltas for this next one
    result = fields.get_timedelta_field(dtindex, "seconds")
    expected = np.array([0] * 5, dtype=np.int32)
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_validate_inclusive.py ===
import numpy as np
import pytest

from pandas.util._validators import validate_inclusive

import pandas as pd


@pytest.mark.parametrize(
    "invalid_inclusive",
    (
        "ccc",
        2,
        object(),
        None,
        np.nan,
        pd.NA,
        pd.DataFrame(),
    ),
)
def test_invalid_inclusive(invalid_inclusive):
    with pytest.raises(
        ValueError,
        match="Inclusive has to be either 'both', 'neither', 'left' or 'right'",
    ):
        validate_inclusive(invalid_inclusive)


@pytest.mark.parametrize(
    "valid_inclusive, expected_tuple",
    (
        ("left", (True, False)),
        ("right", (False, True)),
        ("both", (True, True)),
        ("neither", (False, False)),
    ),
)
def test_valid_inclusive(valid_inclusive, expected_tuple):
    resultant_tuple = validate_inclusive(valid_inclusive)
    assert expected_tuple == resultant_tuple

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_importtools.py ===
from sympy.external import import_module
from sympy.testing.pytest import warns

# fixes issue that arose in addressing issue 6533
def test_no_stdlib_collections():
    '''
    make sure we get the right collections when it is not part of a
    larger list
    '''
    import collections
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['cm', 'collections']},
        min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        assert collections != matplotlib.collections

def test_no_stdlib_collections2():
    '''
    make sure we get the right collections when it is not part of a
    larger list
    '''
    import collections
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['collections']},
        min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        assert collections != matplotlib.collections

def test_no_stdlib_collections3():
    '''make sure we get the right collections with no catch'''
    import collections
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['cm', 'collections']},
        min_module_version='1.1.0')
    if matplotlib:
        assert collections != matplotlib.collections

def test_min_module_version_python3_basestring_error():
    with warns(UserWarning):
        import_module('mpmath', min_module_version='1000.0.1')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\examples\cython\setup.py ===
"""
Provide python-space access to the functions exposed in numpy/__init__.pxd
for testing.
"""

import os
from distutils.core import setup

import Cython
from Cython.Build import cythonize
from setuptools.extension import Extension

import numpy as np
from numpy._utils import _pep440

macros = [
    ("NPY_NO_DEPRECATED_API", 0),
    # Require 1.25+ to test datetime additions
    ("NPY_TARGET_VERSION", "NPY_2_0_API_VERSION"),
]

checks = Extension(
    "checks",
    sources=[os.path.join('.', "checks.pyx")],
    include_dirs=[np.get_include()],
    define_macros=macros,
)

extensions = [checks]

compiler_directives = {}
if _pep440.parse(Cython.__version__) >= _pep440.parse("3.1.0a0"):
    compiler_directives['freethreading_compatible'] = True

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives)
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_series_apply_relabeling.py ===
import pandas as pd
import pandas._testing as tm


def test_relabel_no_duplicated_method():
    # this is to test there is no duplicated method used in agg
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4]})

    result = df["A"].agg(foo="sum")
    expected = df["A"].agg({"foo": "sum"})
    tm.assert_series_equal(result, expected)

    result = df["B"].agg(foo="min", bar="max")
    expected = df["B"].agg({"foo": "min", "bar": "max"})
    tm.assert_series_equal(result, expected)

    msg = "using Series.[sum|min|max]"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df["B"].agg(foo=sum, bar=min, cat="max")
    msg = "using Series.[sum|min|max]"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df["B"].agg({"foo": sum, "bar": min, "cat": "max"})
    tm.assert_series_equal(result, expected)


def test_relabel_duplicated_method():
    # this is to test with nested renaming, duplicated method can be used
    # if they are assigned with different new names
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4]})

    result = df["A"].agg(foo="sum", bar="sum")
    expected = pd.Series([6, 6], index=["foo", "bar"], name="A")
    tm.assert_series_equal(result, expected)

    msg = "using Series.min"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df["B"].agg(foo=min, bar="min")
    expected = pd.Series([1, 1], index=["foo", "bar"], name="B")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\test_array_ops.py ===
import operator

import numpy as np
import pytest

import pandas._testing as tm
from pandas.core.ops.array_ops import (
    comparison_op,
    na_logical_op,
)


def test_na_logical_op_2d():
    left = np.arange(8).reshape(4, 2)
    right = left.astype(object)
    right[0, 0] = np.nan

    # Check that we fall back to the vec_binop branch
    with pytest.raises(TypeError, match="unsupported operand type"):
        operator.or_(left, right)

    result = na_logical_op(left, right, operator.or_)
    expected = right
    tm.assert_numpy_array_equal(result, expected)


def test_object_comparison_2d():
    left = np.arange(9).reshape(3, 3).astype(object)
    right = left.T

    result = comparison_op(left, right, operator.eq)
    expected = np.eye(3).astype(bool)
    tm.assert_numpy_array_equal(result, expected)

    # Ensure that cython doesn't raise on non-writeable arg, which
    #  we can get from np.broadcast_to
    right.flags.writeable = False
    result = comparison_op(left, right, operator.ne)
    tm.assert_numpy_array_equal(result, ~expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_comparison.py ===
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.arrays.masked_shared import (
    ComparisonOps,
    NumericOps,
)


class TestComparisonOps(NumericOps, ComparisonOps):
    @pytest.mark.parametrize("other", [True, False, pd.NA, -1, 0, 1])
    def test_scalar(self, other, comparison_op, dtype):
        ComparisonOps.test_scalar(self, other, comparison_op, dtype)

    def test_compare_to_int(self, dtype, comparison_op):
        # GH 28930
        op_name = f"__{comparison_op.__name__}__"
        s1 = pd.Series([1, None, 3], dtype=dtype)
        s2 = pd.Series([1, None, 3], dtype="float")

        method = getattr(s1, op_name)
        result = method(2)

        method = getattr(s2, op_name)
        expected = method(2).astype("boolean")
        expected[s2.isna()] = pd.NA

        tm.assert_series_equal(result, expected)


def test_equals():
    # GH-30652
    # equals is generally tested in /tests/extension/base/methods, but this
    # specifically tests that two arrays of the same class but different dtype
    # do not evaluate equal
    a1 = pd.array([1, 2, None], dtype="Int64")
    a2 = pd.array([1, 2, None], dtype="Int32")
    assert a1.equals(a2) is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\io.py ===
from io import StringIO

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray


class BaseParsingTests:
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        if isinstance(data.dtype, pd.CategoricalDtype):
            # in parsers.pyx _convert_with_dtype there is special-casing for
            #  Categorical that pre-empts _from_sequence_of_strings
            pass
        elif isinstance(data.dtype, pd.core.dtypes.dtypes.NumpyEADtype):
            # These get unwrapped internally so are treated as numpy dtypes
            #  in the parsers.pyx code
            pass
        elif (
            type(data)._from_sequence_of_strings.__func__
            is ExtensionArray._from_sequence_of_strings.__func__
        ):
            # i.e. the EA hasn't overridden _from_sequence_of_strings
            mark = pytest.mark.xfail(
                reason="_from_sequence_of_strings not implemented",
                raises=NotImplementedError,
            )
            request.node.add_marker(mark)

        df = pd.DataFrame({"with_dtype": pd.Series(data, dtype=str(data.dtype))})
        csv_output = df.to_csv(index=False, na_rep=np.nan)
        result = pd.read_csv(
            StringIO(csv_output), dtype={"with_dtype": str(data.dtype)}, engine=engine
        )
        expected = df
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_count.py ===
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameCount:
    def test_count(self):
        # corner case
        frame = DataFrame()
        ct1 = frame.count(1)
        assert isinstance(ct1, Series)

        ct2 = frame.count(0)
        assert isinstance(ct2, Series)

        # GH#423
        df = DataFrame(index=range(10))
        result = df.count(1)
        expected = Series(0, index=df.index)
        tm.assert_series_equal(result, expected)

        df = DataFrame(columns=range(10))
        result = df.count(0)
        expected = Series(0, index=df.columns)
        tm.assert_series_equal(result, expected)

        df = DataFrame()
        result = df.count()
        expected = Series(dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_count_objects(self, float_string_frame):
        dm = DataFrame(float_string_frame._series)
        df = DataFrame(float_string_frame._series)

        tm.assert_series_equal(dm.count(), df.count())
        tm.assert_series_equal(dm.count(1), df.count(1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_pipe.py ===
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestPipe:
    def test_pipe(self, frame_or_series):
        obj = DataFrame({"A": [1, 2, 3]})
        expected = DataFrame({"A": [1, 4, 9]})
        if frame_or_series is Series:
            obj = obj["A"]
            expected = expected["A"]

        f = lambda x, y: x**y
        result = obj.pipe(f, 2)
        tm.assert_equal(result, expected)

    def test_pipe_tuple(self, frame_or_series):
        obj = DataFrame({"A": [1, 2, 3]})
        obj = tm.get_obj(obj, frame_or_series)

        f = lambda x, y: y
        result = obj.pipe((f, "y"), 0)
        tm.assert_equal(result, obj)

    def test_pipe_tuple_error(self, frame_or_series):
        obj = DataFrame({"A": [1, 2, 3]})
        obj = tm.get_obj(obj, frame_or_series)

        f = lambda x, y: y

        msg = "y is both the pipe target and a keyword argument"

        with pytest.raises(ValueError, match=msg):
            obj.pipe((f, "y"), x=1, y=0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_reindex_like.py ===
import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestDataFrameReindexLike:
    def test_reindex_like(self, float_frame):
        other = float_frame.reindex(index=float_frame.index[:10], columns=["C", "B"])

        tm.assert_frame_equal(other, float_frame.reindex_like(other))

    @pytest.mark.parametrize(
        "method,expected_values",
        [
            ("nearest", [0, 1, 1, 2]),
            ("pad", [np.nan, 0, 1, 1]),
            ("backfill", [0, 1, 2, 2]),
        ],
    )
    def test_reindex_like_methods(self, method, expected_values):
        df = DataFrame({"x": list(range(5))})

        result = df.reindex_like(df, method=method, tolerance=0)
        tm.assert_frame_equal(df, result)
        result = df.reindex_like(df, method=method, tolerance=[0, 0, 0, 0])
        tm.assert_frame_equal(df, result)

    def test_reindex_like_subclass(self):
        # https://github.com/pandas-dev/pandas/issues/31925
        class MyDataFrame(DataFrame):
            pass

        expected = DataFrame()
        df = MyDataFrame()
        result = df.reindex_like(expected)

        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_rewrite_warning.py ===
import warnings

import pytest

from pandas.util._exceptions import rewrite_warning

import pandas._testing as tm


@pytest.mark.parametrize(
    "target_category, target_message, hit",
    [
        (FutureWarning, "Target message", True),
        (FutureWarning, "Target", True),
        (FutureWarning, "get mess", True),
        (FutureWarning, "Missed message", False),
        (DeprecationWarning, "Target message", False),
    ],
)
@pytest.mark.parametrize(
    "new_category",
    [
        None,
        DeprecationWarning,
    ],
)
def test_rewrite_warning(target_category, target_message, hit, new_category):
    new_message = "Rewritten message"
    if hit:
        expected_category = new_category if new_category else target_category
        expected_message = new_message
    else:
        expected_category = FutureWarning
        expected_message = "Target message"
    with tm.assert_produces_warning(expected_category, match=expected_message):
        with rewrite_warning(
            target_message, target_category, new_message, new_category
        ):
            warnings.warn(message="Target message", category=FutureWarning)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_context.py ===
from sympy.assumptions import ask, Q
from sympy.assumptions.assume import assuming, global_assumptions
from sympy.abc import x, y

def test_assuming():
    with assuming(Q.integer(x)):
        assert ask(Q.integer(x))
    assert not ask(Q.integer(x))

def test_assuming_nested():
    assert not ask(Q.integer(x))
    assert not ask(Q.integer(y))
    with assuming(Q.integer(x)):
        assert ask(Q.integer(x))
        assert not ask(Q.integer(y))
        with assuming(Q.integer(y)):
            assert ask(Q.integer(x))
            assert ask(Q.integer(y))
        assert ask(Q.integer(x))
        assert not ask(Q.integer(y))
    assert not ask(Q.integer(x))
    assert not ask(Q.integer(y))

def test_finally():
    try:
        with assuming(Q.integer(x)):
            1/0
    except ZeroDivisionError:
        pass
    assert not ask(Q.integer(x))

def test_remove_safe():
    global_assumptions.add(Q.integer(x))
    with assuming():
        assert ask(Q.integer(x))
        global_assumptions.remove(Q.integer(x))
        assert not ask(Q.integer(x))
    assert ask(Q.integer(x))
    global_assumptions.clear() # for the benefit of other tests

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_wrapper.py ===
from sympy.assumptions.ask import Q
from sympy.assumptions.wrapper import (AssumptionsWrapper, is_infinite,
    is_extended_real)
from sympy.core.symbol import Symbol
from sympy.core.assumptions import _assume_defined


def test_all_predicates():
    for fact in _assume_defined:
        method_name = f'_eval_is_{fact}'
        assert hasattr(AssumptionsWrapper, method_name)


def test_AssumptionsWrapper():
    x = Symbol('x', positive=True)
    y = Symbol('y')
    assert AssumptionsWrapper(x).is_positive
    assert AssumptionsWrapper(y).is_positive is None
    assert AssumptionsWrapper(y, Q.positive(y)).is_positive


def test_is_infinite():
    x = Symbol('x', infinite=True)
    y = Symbol('y', infinite=False)
    z = Symbol('z')
    assert is_infinite(x)
    assert not is_infinite(y)
    assert is_infinite(z) is None
    assert is_infinite(z, Q.infinite(z))


def test_is_extended_real():
    x = Symbol('x', extended_real=True)
    y = Symbol('y', extended_real=False)
    z = Symbol('z')
    assert is_extended_real(x)
    assert not is_extended_real(y)
    assert is_extended_real(z) is None
    assert is_extended_real(z, Q.extended_real(z))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_injections.py ===
"""Tests for functions that inject symbols into the global namespace. """

from sympy.polys.rings import vring
from sympy.polys.fields import vfield
from sympy.polys.domains import QQ

def test_vring():
    ns = {'vring':vring, 'QQ':QQ}
    exec('R = vring("r", QQ)', ns)
    exec('assert r == R.gens[0]', ns)

    exec('R = vring("rb rbb rcc rzz _rx", QQ)', ns)
    exec('assert rb == R.gens[0]', ns)
    exec('assert rbb == R.gens[1]', ns)
    exec('assert rcc == R.gens[2]', ns)
    exec('assert rzz == R.gens[3]', ns)
    exec('assert _rx == R.gens[4]', ns)

    exec('R = vring(["rd", "re", "rfg"], QQ)', ns)
    exec('assert rd == R.gens[0]', ns)
    exec('assert re == R.gens[1]', ns)
    exec('assert rfg == R.gens[2]', ns)

def test_vfield():
    ns = {'vfield':vfield, 'QQ':QQ}
    exec('F = vfield("f", QQ)', ns)
    exec('assert f == F.gens[0]', ns)

    exec('F = vfield("fb fbb fcc fzz _fx", QQ)', ns)
    exec('assert fb == F.gens[0]', ns)
    exec('assert fbb == F.gens[1]', ns)
    exec('assert fcc == F.gens[2]', ns)
    exec('assert fzz == F.gens[3]', ns)
    exec('assert _fx == F.gens[4]', ns)

    exec('F = vfield(["fd", "fe", "ffg"], QQ)', ns)
    exec('assert fd == F.gens[0]', ns)
    exec('assert fe == F.gens[1]', ns)
    exec('assert ffg == F.gens[2]', ns)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_lazyloading.py ===
import sys
from importlib.util import LazyLoader, find_spec, module_from_spec

import pytest


# Warning raised by _reload_guard() in numpy/__init__.py
@pytest.mark.filterwarnings("ignore:The NumPy module was reloaded")
def test_lazy_load():
    # gh-22045. lazyload doesn't import submodule names into the namespace
    # muck with sys.modules to test the importing system
    old_numpy = sys.modules.pop("numpy")

    numpy_modules = {}
    for mod_name, mod in list(sys.modules.items()):
        if mod_name[:6] == "numpy.":
            numpy_modules[mod_name] = mod
            sys.modules.pop(mod_name)

    try:
        # create lazy load of numpy as np
        spec = find_spec("numpy")
        module = module_from_spec(spec)
        sys.modules["numpy"] = module
        loader = LazyLoader(spec.loader)
        loader.exec_module(module)
        np = module

        # test a subpackage import
        from numpy.lib import recfunctions  # noqa: F401

        # test triggering the import of the package
        np.ndarray

    finally:
        if old_numpy:
            sys.modules["numpy"] = old_numpy
            sys.modules.update(numpy_modules)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_scalar_compat.py ===
"""Tests for PeriodIndex behaving like a vectorized Period scalar"""

import pytest

from pandas import (
    Timedelta,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestPeriodIndexOps:
    def test_start_time(self):
        # GH#17157
        index = period_range(freq="M", start="2016-01-01", end="2016-05-31")
        expected_index = date_range("2016-01-01", end="2016-05-31", freq="MS")
        tm.assert_index_equal(index.start_time, expected_index)

    def test_end_time(self):
        # GH#17157
        index = period_range(freq="M", start="2016-01-01", end="2016-05-31")
        expected_index = date_range("2016-01-01", end="2016-05-31", freq="ME")
        expected_index += Timedelta(1, "D") - Timedelta(1, "ns")
        tm.assert_index_equal(index.end_time, expected_index)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_end_time_business_friday(self):
        # GH#34449
        pi = period_range("1990-01-05", freq="B", periods=1)
        result = pi.end_time

        dti = date_range("1990-01-05", freq="D", periods=1)._with_freq(None)
        expected = dti + Timedelta(days=1, nanoseconds=-1)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\xml\conftest.py ===
from pathlib import Path

import pytest


@pytest.fixture
def xml_data_path():
    return Path(__file__).parent.parent / "data" / "xml"


@pytest.fixture
def xml_books(xml_data_path, datapath):
    return datapath(xml_data_path / "books.xml")


@pytest.fixture
def xml_doc_ch_utf(xml_data_path, datapath):
    return datapath(xml_data_path / "doc_ch_utf.xml")


@pytest.fixture
def xml_baby_names(xml_data_path, datapath):
    return datapath(xml_data_path / "baby_names.xml")


@pytest.fixture
def kml_cta_rail_lines(xml_data_path, datapath):
    return datapath(xml_data_path / "cta_rail_lines.kml")


@pytest.fixture
def xsl_flatten_doc(xml_data_path, datapath):
    return datapath(xml_data_path / "flatten_doc.xsl")


@pytest.fixture
def xsl_row_field_output(xml_data_path, datapath):
    return datapath(xml_data_path / "row_field_output.xsl")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_get_numeric_data.py ===
from pandas import (
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


class TestGetNumericData:
    def test_get_numeric_data_preserve_dtype(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # get the numeric data
        obj = Series([1, 2, 3])
        result = obj._get_numeric_data()
        tm.assert_series_equal(result, obj)

        # returned object is a shallow copy
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[0] = 0
        if using_copy_on_write:
            assert obj.iloc[0] == 1
        else:
            assert obj.iloc[0] == 0

        obj = Series([1, "2", 3.0])
        result = obj._get_numeric_data()
        expected = Series([], dtype=object, index=Index([], dtype=object))
        tm.assert_series_equal(result, expected)

        obj = Series([True, False, True])
        result = obj._get_numeric_data()
        tm.assert_series_equal(result, obj)

        obj = Series(date_range("20130101", periods=3))
        result = obj._get_numeric_data()
        expected = Series([], dtype="M8[ns]", index=Index([], dtype=object))
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_to_dict.py ===
import collections

import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


class TestSeriesToDict:
    @pytest.mark.parametrize(
        "mapping", (dict, collections.defaultdict(list), collections.OrderedDict)
    )
    def test_to_dict(self, mapping, datetime_series):
        # GH#16122
        result = Series(datetime_series.to_dict(into=mapping), name="ts")
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)

        from_method = Series(datetime_series.to_dict(into=collections.Counter))
        from_constructor = Series(collections.Counter(datetime_series.items()))
        tm.assert_series_equal(from_method, from_constructor)

    @pytest.mark.parametrize(
        "input",
        (
            {"a": np.int64(64), "b": 10},
            {"a": np.int64(64), "b": 10, "c": "ABC"},
            {"a": np.uint64(64), "b": 10, "c": "ABC"},
        ),
    )
    def test_to_dict_return_types(self, input):
        # GH25969

        d = Series(input).to_dict()
        assert isinstance(d["a"], int)
        assert isinstance(d["b"], int)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_geometrysets.py ===
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.geometry import Circle, Line, Point, Polygon, Segment
from sympy.sets import FiniteSet, Union, Intersection, EmptySet


def test_booleans():
    """ test basic unions and intersections """
    half = S.Half

    p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
    p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
    l1 = Line(Point(0,0), Point(1,1))
    l2 = Line(Point(half, half), Point(5,5))
    l3 = Line(p2, p3)
    l4 = Line(p3, p4)
    poly1 = Polygon(p1, p2, p3, p4)
    poly2 = Polygon(p5, p6, p7)
    poly3 = Polygon(p1, p2, p5)
    assert Union(l1, l2).equals(l1)
    assert Intersection(l1, l2).equals(l1)
    assert Intersection(l1, l4) == FiniteSet(Point(1,1))
    assert Intersection(Union(l1, l4), l3) == FiniteSet(Point(Rational(-1, 3), Rational(-1, 3)), Point(5, 1))
    assert Intersection(l1, FiniteSet(Point(7,-7))) == EmptySet
    assert Intersection(Circle(Point(0,0), 3), Line(p1,p2)) == FiniteSet(Point(-3,0), Point(3,0))
    assert Intersection(l1, FiniteSet(p1)) == FiniteSet(p1)
    assert Union(l1, FiniteSet(p1)) == l1

    fs = FiniteSet(Point(Rational(1, 3), 1), Point(Rational(2, 3), 0), Point(Rational(9, 5), Rational(1, 5)), Point(Rational(7, 3), 1))
    # test the intersection of polygons
    assert Intersection(poly1, poly2) == fs
    # make sure if we union polygons with subsets, the subsets go away
    assert Union(poly1, poly2, fs) == Union(poly1, poly2)
    # make sure that if we union with a FiniteSet that isn't a subset,
    # that the points in the intersection stop being listed
    assert Union(poly1, FiniteSet(Point(0,0), Point(3,5))) == Union(poly1, FiniteSet(Point(3,5)))
    # intersect two polygons that share an edge
    assert Intersection(poly1, poly3) == Union(FiniteSet(Point(Rational(3, 2), 1), Point(2, 1)), Segment(Point(0, 0), Point(1, 0)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_preview.py ===
# -*- coding: utf-8 -*-

from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.preview import preview

from io import BytesIO


def test_preview():
    x = Symbol('x')
    obj = BytesIO()
    try:
        preview(x, output='png', viewer='BytesIO', outputbuffer=obj)
    except RuntimeError:
        pass  # latex not installed on CI server


def test_preview_unicode_symbol():
    # issue 9107
    a = Symbol('α')
    obj = BytesIO()
    try:
        preview(a, output='png', viewer='BytesIO', outputbuffer=obj)
    except RuntimeError:
        pass  # latex not installed on CI server


def test_preview_latex_construct_in_expr():
    # see PR 9801
    x = Symbol('x')
    pw = Piecewise((1, Eq(x, 0)), (0, True))
    obj = BytesIO()
    try:
        preview(pw, output='png', viewer='BytesIO', outputbuffer=obj)
    except RuntimeError:
        pass  # latex not installed on CI server

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\arrayprint.py ===
import numpy as np

AR = np.arange(10)
AR.setflags(write=False)

with np.printoptions():
    np.set_printoptions(
        precision=1,
        threshold=2,
        edgeitems=3,
        linewidth=4,
        suppress=False,
        nanstr="Bob",
        infstr="Bill",
        formatter={},
        sign="+",
        floatmode="unique",
    )
    np.get_printoptions()
    str(AR)

    np.array2string(
        AR,
        max_line_width=5,
        precision=2,
        suppress_small=True,
        separator=";",
        prefix="test",
        threshold=5,
        floatmode="fixed",
        suffix="?",
        legacy="1.13",
    )
    np.format_float_scientific(1, precision=5)
    np.format_float_positional(1, trim="k")
    np.array_repr(AR)
    np.array_str(AR)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_swapaxes.py ===
import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestSwapAxes:
    def test_swapaxes(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_frame_equal(df.T, df.swapaxes(0, 1))
            tm.assert_frame_equal(df.T, df.swapaxes(1, 0))

    def test_swapaxes_noop(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_frame_equal(df, df.swapaxes(0, 0))

    def test_swapaxes_invalid_axis(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            msg = "No axis named 2 for object type DataFrame"
            with pytest.raises(ValueError, match=msg):
                df.swapaxes(2, 5)

    def test_round_empty_not_input(self):
        # GH#51032
        df = DataFrame({"a": [1, 2]})
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.swapaxes("index", "index")
        tm.assert_frame_equal(df, result)
        assert df is not result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\common.py ===
"""
Assertion helpers and base class for offsets tests
"""
from __future__ import annotations


def assert_offset_equal(offset, base, expected):
    actual = offset + base
    actual_swapped = base + offset
    actual_apply = offset._apply(base)
    try:
        assert actual == expected
        assert actual_swapped == expected
        assert actual_apply == expected
    except AssertionError as err:
        raise AssertionError(
            f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
            f"\nAt Date: {base}"
        ) from err


def assert_is_on_offset(offset, date, expected):
    actual = offset.is_on_offset(date)
    assert actual == expected, (
        f"\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})"
        f"\nAt Date: {date}"
    )


class WeekDay:
    MON = 0
    TUE = 1
    WED = 2
    THU = 3
    FRI = 4
    SAT = 5
    SUN = 6

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\einsumfunc.py ===
from __future__ import annotations

from typing import Any

import numpy as np

AR_LIKE_b = [True, True, True]
AR_LIKE_u = [np.uint32(1), np.uint32(2), np.uint32(3)]
AR_LIKE_i = [1, 2, 3]
AR_LIKE_f = [1.0, 2.0, 3.0]
AR_LIKE_c = [1j, 2j, 3j]
AR_LIKE_U = ["1", "2", "3"]

OUT_f: np.ndarray[Any, np.dtype[np.float64]] = np.empty(3, dtype=np.float64)
OUT_c: np.ndarray[Any, np.dtype[np.complex128]] = np.empty(3, dtype=np.complex128)

np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_b)
np.einsum("i,i->i", AR_LIKE_u, AR_LIKE_u)
np.einsum("i,i->i", AR_LIKE_i, AR_LIKE_i)
np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f)
np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c)
np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_i)
np.einsum("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)

np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, dtype="c16")
np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe")
np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, out=OUT_c)
np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=int, casting="unsafe", out=OUT_f)

np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_b)
np.einsum_path("i,i->i", AR_LIKE_u, AR_LIKE_u)
np.einsum_path("i,i->i", AR_LIKE_i, AR_LIKE_i)
np.einsum_path("i,i->i", AR_LIKE_f, AR_LIKE_f)
np.einsum_path("i,i->i", AR_LIKE_c, AR_LIKE_c)
np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_i)
np.einsum_path("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\cast\test_construct_ndarray.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.construction import sanitize_array


@pytest.mark.parametrize(
    "values, dtype, expected",
    [
        ([1, 2, 3], None, np.array([1, 2, 3], dtype=np.int64)),
        (np.array([1, 2, 3]), None, np.array([1, 2, 3])),
        (["1", "2", None], None, np.array(["1", "2", None])),
        (["1", "2", None], np.dtype("str"), np.array(["1", "2", None])),
        ([1, 2, None], np.dtype("str"), np.array(["1", "2", None])),
    ],
)
def test_construct_1d_ndarray_preserving_na(
    values, dtype, expected, using_infer_string
):
    result = sanitize_array(values, index=None, dtype=dtype)
    if using_infer_string and expected.dtype == object and dtype is None:
        tm.assert_extension_array_equal(result, pd.array(expected, dtype="str"))
    else:
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]"])
def test_construct_1d_ndarray_preserving_na_datetimelike(dtype):
    arr = np.arange(5, dtype=np.int64).view(dtype)
    expected = np.array(list(arr), dtype=object)
    assert all(isinstance(x, type(arr[0])) for x in expected)

    result = sanitize_array(arr, index=None, dtype=np.dtype(object))
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_droplevel.py ===
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm


class TestDropLevel:
    def test_droplevel(self, frame_or_series):
        # GH#20342
        cols = MultiIndex.from_tuples(
            [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
        )
        mi = MultiIndex.from_tuples([(1, 2), (5, 6), (9, 10)], names=["a", "b"])
        df = DataFrame([[3, 4], [7, 8], [11, 12]], index=mi, columns=cols)
        if frame_or_series is not DataFrame:
            df = df.iloc[:, 0]

        # test that dropping of a level in index works
        expected = df.reset_index("a", drop=True)
        result = df.droplevel("a", axis="index")
        tm.assert_equal(result, expected)

        if frame_or_series is DataFrame:
            # test that dropping of a level in columns works
            expected = df.copy()
            expected.columns = Index(["c", "d"], name="level_1")
            result = df.droplevel("level_2", axis="columns")
            tm.assert_equal(result, expected)
        else:
            # test that droplevel raises ValueError on axis != 0
            with pytest.raises(ValueError, match="No axis named columns"):
                df.droplevel(1, axis="columns")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_swaplevel.py ===
import pytest

from pandas import DataFrame
import pandas._testing as tm


class TestSwaplevel:
    def test_swaplevel(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        swapped = frame["A"].swaplevel()
        swapped2 = frame["A"].swaplevel(0)
        swapped3 = frame["A"].swaplevel(0, 1)
        swapped4 = frame["A"].swaplevel("first", "second")
        assert not swapped.index.equals(frame.index)
        tm.assert_series_equal(swapped, swapped2)
        tm.assert_series_equal(swapped, swapped3)
        tm.assert_series_equal(swapped, swapped4)

        back = swapped.swaplevel()
        back2 = swapped.swaplevel(0)
        back3 = swapped.swaplevel(0, 1)
        back4 = swapped.swaplevel("second", "first")
        assert back.index.equals(frame.index)
        tm.assert_series_equal(back, back2)
        tm.assert_series_equal(back, back3)
        tm.assert_series_equal(back, back4)

        ft = frame.T
        swapped = ft.swaplevel("first", "second", axis=1)
        exp = frame.swaplevel("first", "second").T
        tm.assert_frame_equal(swapped, exp)

        msg = "Can only swap levels on a hierarchical axis."
        with pytest.raises(TypeError, match=msg):
            DataFrame(range(3)).swaplevel()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_equals.py ===
import numpy as np

from pandas import (
    IntervalIndex,
    date_range,
)


class TestEquals:
    def test_equals(self, closed):
        expected = IntervalIndex.from_breaks(np.arange(5), closed=closed)
        assert expected.equals(expected)
        assert expected.equals(expected.copy())

        assert not expected.equals(expected.astype(object))
        assert not expected.equals(np.array(expected))
        assert not expected.equals(list(expected))

        assert not expected.equals([1, 2])
        assert not expected.equals(np.array([1, 2]))
        assert not expected.equals(date_range("20130101", periods=2))

        expected_name1 = IntervalIndex.from_breaks(
            np.arange(5), closed=closed, name="foo"
        )
        expected_name2 = IntervalIndex.from_breaks(
            np.arange(5), closed=closed, name="bar"
        )
        assert expected.equals(expected_name1)
        assert expected_name1.equals(expected_name2)

        for other_closed in {"left", "right", "both", "neither"} - {closed}:
            expected_other_closed = IntervalIndex.from_breaks(
                np.arange(5), closed=other_closed
            )
            assert not expected.equals(expected_other_closed)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_concatenate_chunks.py ===
import numpy as np
import pytest

from pandas.errors import DtypeWarning

import pandas._testing as tm
from pandas.core.arrays import ArrowExtensionArray

from pandas.io.parsers.c_parser_wrapper import _concatenate_chunks


def test_concatenate_chunks_pyarrow():
    # GH#51876
    pa = pytest.importorskip("pyarrow")
    chunks = [
        {0: ArrowExtensionArray(pa.array([1.5, 2.5]))},
        {0: ArrowExtensionArray(pa.array([1, 2]))},
    ]
    result = _concatenate_chunks(chunks)
    expected = ArrowExtensionArray(pa.array([1.5, 2.5, 1.0, 2.0]))
    tm.assert_extension_array_equal(result[0], expected)


def test_concatenate_chunks_pyarrow_strings():
    # GH#51876
    pa = pytest.importorskip("pyarrow")
    chunks = [
        {0: ArrowExtensionArray(pa.array([1.5, 2.5]))},
        {0: ArrowExtensionArray(pa.array(["a", "b"]))},
    ]
    with tm.assert_produces_warning(DtypeWarning, match="have mixed types"):
        result = _concatenate_chunks(chunks)
    expected = np.concatenate(
        [np.array([1.5, 2.5], dtype=object), np.array(["a", "b"])]
    )
    tm.assert_numpy_array_equal(result[0], expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_tolist.py ===
import pytest

import pandas.util._test_decorators as td

from pandas import (
    Interval,
    Period,
    Series,
    Timedelta,
    Timestamp,
)


@pytest.mark.parametrize(
    "values, dtype, expected_dtype",
    (
        ([1], "int64", int),
        ([1], "Int64", int),
        ([1.0], "float64", float),
        ([1.0], "Float64", float),
        (["abc"], "object", str),
        (["abc"], "string", str),
        ([Interval(1, 3)], "interval", Interval),
        ([Period("2000-01-01", "D")], "period[D]", Period),
        ([Timedelta(days=1)], "timedelta64[ns]", Timedelta),
        ([Timestamp("2000-01-01")], "datetime64[ns]", Timestamp),
        pytest.param([1], "int64[pyarrow]", int, marks=td.skip_if_no("pyarrow")),
        pytest.param([1.0], "float64[pyarrow]", float, marks=td.skip_if_no("pyarrow")),
        pytest.param(["abc"], "string[pyarrow]", str, marks=td.skip_if_no("pyarrow")),
    ),
)
def test_tolist_scalar_dtype(values, dtype, expected_dtype):
    # GH49890
    ser = Series(values, dtype=dtype)
    result_dtype = type(ser.tolist()[0])
    assert result_dtype == expected_dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_hypothesis.py ===
from hypothesis import given
from hypothesis import strategies as st
from sympy.abc import x
from sympy.polys.polytools import Poly


def polys(*, nonzero=False, domain="ZZ"):
    # This is a simple strategy, but sufficient the tests below
    elems = {"ZZ": st.integers(), "QQ": st.fractions()}
    coeff_st = st.lists(elems[domain])
    if nonzero:
        coeff_st = coeff_st.filter(any)
    return st.builds(Poly, coeff_st, st.just(x), domain=st.just(domain))


@given(f=polys(), g=polys(), r=polys())
def test_gcd_hypothesis(f, g, r):
    gcd_1 = f.gcd(g)
    gcd_2 = g.gcd(f)
    assert gcd_1 == gcd_2

    # multiply by r
    gcd_3 = g.gcd(f + r * g)
    assert gcd_1 == gcd_3


@given(f_z=polys(), g_z=polys(nonzero=True))
def test_poly_hypothesis_integers(f_z, g_z):
    remainder_z = f_z.rem(g_z)
    assert g_z.degree() >= remainder_z.degree() or remainder_z.degree() == 0


@given(f_q=polys(domain="QQ"), g_q=polys(nonzero=True, domain="QQ"))
def test_poly_hypothesis_rationals(f_q, g_q):
    remainder_q = f_q.rem(g_q)
    assert g_q.degree() >= remainder_q.degree() or remainder_q.degree() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_weakref.py ===
import gc
import weakref


import greenlet
from . import TestCase

class WeakRefTests(TestCase):
    def test_dead_weakref(self):
        def _dead_greenlet():
            g = greenlet.greenlet(lambda: None)
            g.switch()
            return g
        o = weakref.ref(_dead_greenlet())
        gc.collect()
        self.assertEqual(o(), None)

    def test_inactive_weakref(self):
        o = weakref.ref(greenlet.greenlet())
        gc.collect()
        self.assertEqual(o(), None)

    def test_dealloc_weakref(self):
        seen = []
        def worker():
            try:
                greenlet.getcurrent().parent.switch()
            finally:
                seen.append(g())
        g = greenlet.greenlet(worker)
        g.switch()
        g2 = greenlet.greenlet(lambda: None, g)
        g = weakref.ref(g2)
        g2 = None
        self.assertEqual(seen, [None])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_mixed.py ===
import textwrap

import pytest

from numpy.testing import IS_PYPY

from . import util


class TestMixed(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "mixed", "foo.f"),
        util.getpath("tests", "src", "mixed", "foo_fixed.f90"),
        util.getpath("tests", "src", "mixed", "foo_free.f90"),
    ]

    @pytest.mark.slow
    def test_all(self):
        assert self.module.bar11() == 11
        assert self.module.foo_fixed.bar12() == 12
        assert self.module.foo_free.bar13() == 13

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = bar11()

        Wrapper for ``bar11``.

        Returns
        -------
        a : int
        """)
        assert self.module.bar11.__doc__ == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_hashtable.py ===
import random

import pytest
from numpy._core._multiarray_tests import identityhash_tester


@pytest.mark.parametrize("key_length", [1, 3, 6])
@pytest.mark.parametrize("length", [1, 16, 2000])
def test_identity_hashtable(key_length, length):
    # use a 30 object pool for everything (duplicates will happen)
    pool = [object() for i in range(20)]
    keys_vals = []
    for i in range(length):
        keys = tuple(random.choices(pool, k=key_length))
        keys_vals.append((keys, random.choice(pool)))

    dictionary = dict(keys_vals)

    # add a random item at the end:
    keys_vals.append(random.choice(keys_vals))
    # the expected one could be different with duplicates:
    expected = dictionary[keys_vals[-1][0]]

    res = identityhash_tester(key_length, keys_vals, replace=True)
    assert res is expected

    if length == 1:
        return

    # add a new item with a key that is already used and a new value, this
    # should error if replace is False, see gh-26690
    new_key = (keys_vals[1][0], object())
    keys_vals[0] = new_key
    with pytest.raises(RuntimeError):
        identityhash_tester(key_length, keys_vals)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_pyinstaller\tests\test_pyinstaller.py ===
import subprocess
from pathlib import Path

import pytest


# PyInstaller has been very unproactive about replacing 'imp' with 'importlib'.
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
# It also leaks io.BytesIO()s.
@pytest.mark.filterwarnings('ignore::ResourceWarning')
@pytest.mark.parametrize("mode", ["--onedir", "--onefile"])
@pytest.mark.slow
def test_pyinstaller(mode, tmp_path):
    """Compile and run pyinstaller-smoke.py using PyInstaller."""

    pyinstaller_cli = pytest.importorskip("PyInstaller.__main__").run

    source = Path(__file__).with_name("pyinstaller-smoke.py").resolve()
    args = [
        # Place all generated files in ``tmp_path``.
        '--workpath', str(tmp_path / "build"),
        '--distpath', str(tmp_path / "dist"),
        '--specpath', str(tmp_path),
        mode,
        str(source),
    ]
    pyinstaller_cli(args)

    if mode == "--onefile":
        exe = tmp_path / "dist" / source.stem
    else:
        exe = tmp_path / "dist" / source.stem / source.stem

    p = subprocess.run([str(exe)], check=True, stdout=subprocess.PIPE)
    assert p.stdout.strip() == b"I made it!"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_iteration.py ===
class TestIteration:
    def test_keys(self, datetime_series):
        assert datetime_series.keys() is datetime_series.index

    def test_iter_datetimes(self, datetime_series):
        for i, val in enumerate(datetime_series):
            # pylint: disable-next=unnecessary-list-index-lookup
            assert val == datetime_series.iloc[i]

    def test_iter_strings(self, string_series):
        for i, val in enumerate(string_series):
            # pylint: disable-next=unnecessary-list-index-lookup
            assert val == string_series.iloc[i]

    def test_iteritems_datetimes(self, datetime_series):
        for idx, val in datetime_series.items():
            assert val == datetime_series[idx]

    def test_iteritems_strings(self, string_series):
        for idx, val in string_series.items():
            assert val == string_series[idx]

        # assert is lazy (generators don't define reverse, lists do)
        assert not hasattr(string_series.items(), "reverse")

    def test_items_datetimes(self, datetime_series):
        for idx, val in datetime_series.items():
            assert val == datetime_series[idx]

    def test_items_strings(self, string_series):
        for idx, val in string_series.items():
            assert val == string_series[idx]

        # assert is lazy (generators don't define reverse, lists do)
        assert not hasattr(string_series.items(), "reverse")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_isna.py ===
"""
We also test Series.notna in this file.
"""
import numpy as np

from pandas import (
    Period,
    Series,
)
import pandas._testing as tm


class TestIsna:
    def test_isna_period_dtype(self):
        # GH#13737
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])

        expected = Series([False, True])

        result = ser.isna()
        tm.assert_series_equal(result, expected)

        result = ser.notna()
        tm.assert_series_equal(result, ~expected)

    def test_isna(self):
        ser = Series([0, 5.4, 3, np.nan, -0.001])
        expected = Series([False, False, False, True, False])
        tm.assert_series_equal(ser.isna(), expected)
        tm.assert_series_equal(ser.notna(), ~expected)

        ser = Series(["hi", "", np.nan])
        expected = Series([False, False, True])
        tm.assert_series_equal(ser.isna(), expected)
        tm.assert_series_equal(ser.notna(), ~expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_assumptions_2.py ===
"""
rename this to test_assumptions.py when the old assumptions system is deleted
"""
from sympy.abc import x, y
from sympy.assumptions.assume import global_assumptions
from sympy.assumptions.ask import Q
from sympy.printing import pretty


def test_equal():
    """Test for equality"""
    assert Q.positive(x) == Q.positive(x)
    assert Q.positive(x) != ~Q.positive(x)
    assert ~Q.positive(x) == ~Q.positive(x)


def test_pretty():
    assert pretty(Q.positive(x)) == "Q.positive(x)"
    assert pretty(
        {Q.positive, Q.integer}) == "{Q.integer, Q.positive}"


def test_global():
    """Test for global assumptions"""
    global_assumptions.add(x > 0)
    assert (x > 0) in global_assumptions
    global_assumptions.remove(x > 0)
    assert not (x > 0) in global_assumptions
    # same with multiple of assumptions
    global_assumptions.add(x > 0, y > 0)
    assert (x > 0) in global_assumptions
    assert (y > 0) in global_assumptions
    global_assumptions.clear()
    assert not (x > 0) in global_assumptions
    assert not (y > 0) in global_assumptions

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\tests\test_scipy.py ===
# This testfile tests SymPy <-> SciPy compatibility

# Don't test any SymPy features here. Just pure interaction with SciPy.
# Always write regular SymPy tests for anything, that can be tested in pure
# Python (without scipy). Here we test everything, that a user may need when
# using SymPy with SciPy

from sympy.external import import_module

scipy = import_module('scipy')
if not scipy:
    #bin/test will not execute any tests now
    disabled = True

from sympy.functions.special.bessel import jn_zeros


def eq(a, b, tol=1e-6):
    for x, y in zip(a, b):
        if not (abs(x - y) < tol):
            return False
    return True


def test_jn_zeros():
    assert eq(jn_zeros(0, 4, method="scipy"),
            [3.141592, 6.283185, 9.424777, 12.566370])
    assert eq(jn_zeros(1, 4, method="scipy"),
            [4.493409, 7.725251, 10.904121, 14.066193])
    assert eq(jn_zeros(2, 4, method="scipy"),
            [5.763459, 9.095011, 12.322940, 15.514603])
    assert eq(jn_zeros(3, 4, method="scipy"),
            [6.987932, 10.417118, 13.698023, 16.923621])
    assert eq(jn_zeros(4, 4, method="scipy"),
            [8.182561, 11.704907, 15.039664, 18.301255])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\tests\test_weyl_group.py ===
from sympy.liealgebras.weyl_group import WeylGroup
from sympy.matrices import Matrix

def test_weyl_group():
    c = WeylGroup("A3")
    assert c.matrix_form('r1*r2') == Matrix([[0, 0, 1, 0], [1, 0, 0, 0],
        [0, 1, 0, 0], [0, 0, 0, 1]])
    assert c.generators() == ['r1', 'r2', 'r3']
    assert c.group_order() == 24.0
    assert c.group_name() == "S4: the symmetric group acting on 4 elements."
    assert c.coxeter_diagram() == "0---0---0\n1   2   3"
    assert c.element_order('r1*r2*r3') == 4
    assert c.element_order('r1*r3*r2*r3') == 3
    d = WeylGroup("B5")
    assert d.group_order() == 3840
    assert d.element_order('r1*r2*r4*r5') == 12
    assert d.matrix_form('r2*r3') ==  Matrix([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    assert d.element_order('r1*r2*r1*r3*r5') == 6
    e = WeylGroup("D5")
    assert e.element_order('r2*r3*r5') == 4
    assert e.matrix_form('r2*r3*r5') == Matrix([[1, 0, 0, 0, 0], [0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0]])
    f = WeylGroup("G2")
    assert f.element_order('r1*r2*r1*r2') == 3
    assert f.element_order('r2*r1*r1*r2') == 1

    assert f.matrix_form('r1*r2*r1*r2') == Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    g = WeylGroup("F4")
    assert g.matrix_form('r2*r3') == Matrix([[1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, -1], [0, 0, 1, 0]])

    assert g.element_order('r2*r3') == 4
    h = WeylGroup("E6")
    assert h.group_order() == 51840