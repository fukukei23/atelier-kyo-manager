
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_scalar_compat.py ===
"""
Tests for TimedeltaIndex methods behaving like their Timedelta counterparts
"""

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG

from pandas import (
    Index,
    Series,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestVectorizedTimedelta:
    def test_tdi_total_seconds(self):
        # GH#10939
        # test index
        rng = timedelta_range("1 days, 10:11:12.100123456", periods=2, freq="s")
        expt = [
            1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9,
            1 * 86400 + 10 * 3600 + 11 * 60 + 13 + 100123456.0 / 1e9,
        ]
        tm.assert_almost_equal(rng.total_seconds(), Index(expt))

        # test Series
        ser = Series(rng)
        s_expt = Series(expt, index=[0, 1])
        tm.assert_series_equal(ser.dt.total_seconds(), s_expt)

        # with nat
        ser[1] = np.nan
        s_expt = Series(
            [1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9, np.nan],
            index=[0, 1],
        )
        tm.assert_series_equal(ser.dt.total_seconds(), s_expt)

    def test_tdi_total_seconds_all_nat(self):
        # with both nat
        ser = Series([np.nan, np.nan], dtype="timedelta64[ns]")
        result = ser.dt.total_seconds()
        expected = Series([np.nan, np.nan])
        tm.assert_series_equal(result, expected)

    def test_tdi_round(self):
        td = timedelta_range(start="16801 days", periods=5, freq="30Min")
        elt = td[1]

        expected_rng = TimedeltaIndex(
            [
                Timedelta("16801 days 00:00:00"),
                Timedelta("16801 days 00:00:00"),
                Timedelta("16801 days 01:00:00"),
                Timedelta("16801 days 02:00:00"),
                Timedelta("16801 days 02:00:00"),
            ]
        )
        expected_elt = expected_rng[1]

        tm.assert_index_equal(td.round(freq="h"), expected_rng)
        assert elt.round(freq="h") == expected_elt

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            td.round(freq="foo")
        with pytest.raises(ValueError, match=msg):
            elt.round(freq="foo")

        msg = "<MonthEnd> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            td.round(freq="ME")
        with pytest.raises(ValueError, match=msg):
            elt.round(freq="ME")

    @pytest.mark.parametrize(
        "freq,msg",
        [
            ("YE", "<YearEnd: month=12> is a non-fixed frequency"),
            ("ME", "<MonthEnd> is a non-fixed frequency"),
            ("foobar", "Invalid frequency: foobar"),
        ],
    )
    def test_tdi_round_invalid(self, freq, msg):
        t1 = timedelta_range("1 days", periods=3, freq="1 min 2 s 3 us")

        with pytest.raises(ValueError, match=msg):
            t1.round(freq)
        with pytest.raises(ValueError, match=msg):
            # Same test for TimedeltaArray
            t1._data.round(freq)

    # TODO: de-duplicate with test_tdi_round
    def test_round(self):
        t1 = timedelta_range("1 days", periods=3, freq="1 min 2 s 3 us")
        t2 = -1 * t1
        t1a = timedelta_range("1 days", periods=3, freq="1 min 2 s")
        t1c = TimedeltaIndex(np.array([1, 1, 1], "m8[D]")).as_unit("ns")

        # note that negative times round DOWN! so don't give whole numbers
        for freq, s1, s2 in [
            ("ns", t1, t2),
            ("us", t1, t2),
            (
                "ms",
                t1a,
                TimedeltaIndex(
                    ["-1 days +00:00:00", "-2 days +23:58:58", "-2 days +23:57:56"]
                ),
            ),
            (
                "s",
                t1a,
                TimedeltaIndex(
                    ["-1 days +00:00:00", "-2 days +23:58:58", "-2 days +23:57:56"]
                ),
            ),
            ("12min", t1c, TimedeltaIndex(["-1 days", "-1 days", "-1 days"])),
            ("h", t1c, TimedeltaIndex(["-1 days", "-1 days", "-1 days"])),
            ("d", t1c, -1 * t1c),
        ]:
            r1 = t1.round(freq)
            tm.assert_index_equal(r1, s1)
            r2 = t2.round(freq)
            tm.assert_index_equal(r2, s2)

    def test_components(self):
        rng = timedelta_range("1 days, 10:11:12", periods=2, freq="s")
        rng.components

        # with nat
        s = Series(rng)
        s[1] = np.nan

        result = s.dt.components
        assert not result.iloc[0].isna().all()
        assert result.iloc[1].isna().all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_tensorproduct.py ===
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.core.expr import unchanged
from sympy.matrices import Matrix, SparseMatrix, ImmutableMatrix
from sympy.testing.pytest import warns_deprecated_sympy

from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import TensorProduct as TP
from sympy.physics.quantum.tensorproduct import tensor_product_simp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr

A = Operator('A')
B = Operator('B')
C = Operator('C')
D = Operator('D')
x = symbols('x')
y = symbols('y', integer=True, positive=True)

mat1 = Matrix([[1, 2*I], [1 + I, 3]])
mat2 = Matrix([[2*I, 3], [4*I, 2]])


def test_sparse_matrices():
    spm = SparseMatrix.diag(1, 0)
    assert unchanged(TensorProduct, spm, spm)


def test_tensor_product_dagger():
    assert Dagger(TensorProduct(I*A, B)) == \
        -I*TensorProduct(Dagger(A), Dagger(B))
    assert Dagger(TensorProduct(mat1, mat2)) == \
        TensorProduct(Dagger(mat1), Dagger(mat2))


def test_tensor_product_abstract():

    assert TP(x*A, 2*B) == x*2*TP(A, B)
    assert TP(A, B) != TP(B, A)
    assert TP(A, B).is_commutative is False
    assert isinstance(TP(A, B), TP)
    assert TP(A, B).subs(A, C) == TP(C, B)


def test_tensor_product_expand():
    assert TP(A + B, B + C).expand(tensorproduct=True) == \
        TP(A, B) + TP(A, C) + TP(B, B) + TP(B, C)
    #Tests for fix of issue #24142
    assert TP(A-B, B-A).expand(tensorproduct=True) == \
        TP(A, B) - TP(A, A) - TP(B, B) + TP(B, A)
    assert TP(2*A + B, A + B).expand(tensorproduct=True) == \
        2 * TP(A, A) + 2 * TP(A, B) + TP(B, A) + TP(B, B)
    assert TP(2 * A * B + A, A + B).expand(tensorproduct=True) == \
        2 * TP(A*B, A) + 2 * TP(A*B, B) + TP(A, A) + TP(A, B)


def test_tensor_product_commutator():
    assert TP(Comm(A, B), C).doit().expand(tensorproduct=True) == \
        TP(A*B, C) - TP(B*A, C)
    assert Comm(TP(A, B), TP(B, C)).doit() == \
        TP(A, B)*TP(B, C) - TP(B, C)*TP(A, B)


def test_tensor_product_simp():
    with warns_deprecated_sympy():
        assert tensor_product_simp(TP(A, B)*TP(B, C)) == TP(A*B, B*C)
        # tests for Pow-expressions
        assert TP(A, B)**y == TP(A**y, B**y)
        assert tensor_product_simp(TP(A, B)**y) == TP(A**y, B**y)
        assert tensor_product_simp(x*TP(A, B)**2) == x*TP(A**2,B**2)
        assert tensor_product_simp(x*(TP(A, B)**2)*TP(C,D)) == x*TP(A**2*C,B**2*D)
        assert tensor_product_simp(TP(A,B)-TP(C,D)**y) == TP(A,B)-TP(C**y,D**y)


def test_issue_5923():
    # most of the issue regarding sympification of args has been handled
    # and is tested internally by the use of args_cnc through the quantum
    # module, but the following is a test from the issue that used to raise.
    assert TensorProduct(1, Qubit('1')*Qubit('1').dual) == \
        TensorProduct(1, OuterProduct(Qubit(1), QubitBra(1)))


def test_eval_trace():
    # This test includes tests with dependencies between TensorProducts
    #and density operators. Since, the test is more to test the behavior of
    #TensorProducts it remains here

    # Density with simple tensor products as args
    t = TensorProduct(A, B)
    d = Density([t, 1.0])
    tr = Tr(d)
    assert tr.doit() == 1.0*Tr(A*Dagger(A))*Tr(B*Dagger(B))

    ## partial trace with simple tensor products as args
    t = TensorProduct(A, B, C)
    d = Density([t, 1.0])
    tr = Tr(d, [1])
    assert tr.doit() == 1.0*A*Dagger(A)*Tr(B*Dagger(B))*C*Dagger(C)

    tr = Tr(d, [0, 2])
    assert tr.doit() == 1.0*Tr(A*Dagger(A))*B*Dagger(B)*Tr(C*Dagger(C))

    # Density with multiple Tensorproducts as states
    t2 = TensorProduct(A, B)
    t3 = TensorProduct(C, D)

    d = Density([t2, 0.5], [t3, 0.5])
    t = Tr(d)
    assert t.doit() == (0.5*Tr(A*Dagger(A))*Tr(B*Dagger(B)) +
                        0.5*Tr(C*Dagger(C))*Tr(D*Dagger(D)))

    t = Tr(d, [0])
    assert t.doit() == (0.5*Tr(A*Dagger(A))*B*Dagger(B) +
                        0.5*Tr(C*Dagger(C))*D*Dagger(D))

    #Density with mixed states
    d = Density([t2 + t3, 1.0])
    t = Tr(d)
    assert t.doit() == ( 1.0*Tr(A*Dagger(A))*Tr(B*Dagger(B)) +
                        1.0*Tr(A*Dagger(C))*Tr(B*Dagger(D)) +
                        1.0*Tr(C*Dagger(A))*Tr(D*Dagger(B)) +
                        1.0*Tr(C*Dagger(C))*Tr(D*Dagger(D)))

    t = Tr(d, [1] )
    assert t.doit() == ( 1.0*A*Dagger(A)*Tr(B*Dagger(B)) +
                        1.0*A*Dagger(C)*Tr(B*Dagger(D)) +
                        1.0*C*Dagger(A)*Tr(D*Dagger(B)) +
                        1.0*C*Dagger(C)*Tr(D*Dagger(D)))


def test_pr24993():
    from sympy.matrices.expressions.kronecker import matrix_kronecker_product
    from sympy.physics.quantum.matrixutils    import matrix_tensor_product
    X = Matrix([[0, 1], [1, 0]])
    Xi = ImmutableMatrix(X)
    assert TensorProduct(Xi, Xi) == TensorProduct(X, X)
    assert TensorProduct(Xi, Xi) == matrix_tensor_product(X, X)
    assert TensorProduct(Xi, Xi) == matrix_kronecker_product(X, X)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_delete.py ===
import pytest

from pandas import (
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDelete:
    def test_delete(self, unit):
        idx = date_range(
            start="2000-01-01", periods=5, freq="ME", name="idx", unit=unit
        )

        # preserve freq
        expected_0 = date_range(
            start="2000-02-01", periods=4, freq="ME", name="idx", unit=unit
        )
        expected_4 = date_range(
            start="2000-01-01", periods=4, freq="ME", name="idx", unit=unit
        )

        # reset freq to None
        expected_1 = DatetimeIndex(
            ["2000-01-31", "2000-03-31", "2000-04-30", "2000-05-31"],
            freq=None,
            name="idx",
        ).as_unit(unit)

        cases = {
            0: expected_0,
            -5: expected_0,
            -1: expected_4,
            4: expected_4,
            1: expected_1,
        }
        for n, expected in cases.items():
            result = idx.delete(n)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

        with pytest.raises((IndexError, ValueError), match="out of bounds"):
            # either depending on numpy version
            idx.delete(5)

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Pacific"])
    def test_delete2(self, tz):
        idx = date_range(
            start="2000-01-01 09:00", periods=10, freq="h", name="idx", tz=tz
        )

        expected = date_range(
            start="2000-01-01 10:00", periods=9, freq="h", name="idx", tz=tz
        )
        result = idx.delete(0)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freqstr == "h"
        assert result.tz == expected.tz

        expected = date_range(
            start="2000-01-01 09:00", periods=9, freq="h", name="idx", tz=tz
        )
        result = idx.delete(-1)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freqstr == "h"
        assert result.tz == expected.tz

    def test_delete_slice(self, unit):
        idx = date_range(
            start="2000-01-01", periods=10, freq="D", name="idx", unit=unit
        )

        # preserve freq
        expected_0_2 = date_range(
            start="2000-01-04", periods=7, freq="D", name="idx", unit=unit
        )
        expected_7_9 = date_range(
            start="2000-01-01", periods=7, freq="D", name="idx", unit=unit
        )

        # reset freq to None
        expected_3_5 = DatetimeIndex(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
                "2000-01-07",
                "2000-01-08",
                "2000-01-09",
                "2000-01-10",
            ],
            freq=None,
            name="idx",
        ).as_unit(unit)

        cases = {
            (0, 1, 2): expected_0_2,
            (7, 8, 9): expected_7_9,
            (3, 4, 5): expected_3_5,
        }
        for n, expected in cases.items():
            result = idx.delete(n)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

            result = idx.delete(slice(n[0], n[-1] + 1))
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

    # TODO: belongs in Series.drop tests?
    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Pacific"])
    def test_delete_slice2(self, tz, unit):
        dti = date_range(
            "2000-01-01 09:00", periods=10, freq="h", name="idx", tz=tz, unit=unit
        )
        ts = Series(
            1,
            index=dti,
        )
        # preserve freq
        result = ts.drop(ts.index[:5]).index
        expected = dti[5:]
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == expected.freq
        assert result.tz == expected.tz

        # reset freq to None
        result = ts.drop(ts.index[[1, 3, 5, 7, 9]]).index
        expected = dti[::2]._with_freq(None)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == expected.freq
        assert result.tz == expected.tz

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_compare.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("align_axis", [0, 1, "index", "columns"])
def test_compare_axis(align_axis):
    # GH#30429
    s1 = pd.Series(["a", "b", "c"])
    s2 = pd.Series(["x", "b", "z"])

    result = s1.compare(s2, align_axis=align_axis)

    if align_axis in (1, "columns"):
        indices = pd.Index([0, 2])
        columns = pd.Index(["self", "other"])
        expected = pd.DataFrame(
            [["a", "x"], ["c", "z"]], index=indices, columns=columns
        )
        tm.assert_frame_equal(result, expected)
    else:
        indices = pd.MultiIndex.from_product([[0, 2], ["self", "other"]])
        expected = pd.Series(["a", "x", "c", "z"], index=indices)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "keep_shape, keep_equal",
    [
        (True, False),
        (False, True),
        (True, True),
        # False, False case is already covered in test_compare_axis
    ],
)
def test_compare_various_formats(keep_shape, keep_equal):
    s1 = pd.Series(["a", "b", "c"])
    s2 = pd.Series(["x", "b", "z"])

    result = s1.compare(s2, keep_shape=keep_shape, keep_equal=keep_equal)

    if keep_shape:
        indices = pd.Index([0, 1, 2])
        columns = pd.Index(["self", "other"])
        if keep_equal:
            expected = pd.DataFrame(
                [["a", "x"], ["b", "b"], ["c", "z"]], index=indices, columns=columns
            )
        else:
            expected = pd.DataFrame(
                [["a", "x"], [np.nan, np.nan], ["c", "z"]],
                index=indices,
                columns=columns,
            )
    else:
        indices = pd.Index([0, 2])
        columns = pd.Index(["self", "other"])
        expected = pd.DataFrame(
            [["a", "x"], ["c", "z"]], index=indices, columns=columns
        )
    tm.assert_frame_equal(result, expected)


def test_compare_with_equal_nulls():
    # We want to make sure two NaNs are considered the same
    # and dropped where applicable
    s1 = pd.Series(["a", "b", np.nan])
    s2 = pd.Series(["x", "b", np.nan])

    result = s1.compare(s2)
    expected = pd.DataFrame([["a", "x"]], columns=["self", "other"])
    tm.assert_frame_equal(result, expected)


def test_compare_with_non_equal_nulls():
    # We want to make sure the relevant NaNs do not get dropped
    s1 = pd.Series(["a", "b", "c"])
    s2 = pd.Series(["x", "b", np.nan])

    result = s1.compare(s2, align_axis=0)

    indices = pd.MultiIndex.from_product([[0, 2], ["self", "other"]])
    expected = pd.Series(["a", "x", "c", np.nan], index=indices)
    tm.assert_series_equal(result, expected)


def test_compare_multi_index():
    index = pd.MultiIndex.from_arrays([[0, 0, 1], [0, 1, 2]])
    s1 = pd.Series(["a", "b", "c"], index=index)
    s2 = pd.Series(["x", "b", "z"], index=index)

    result = s1.compare(s2, align_axis=0)

    indices = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], [0, 0, 2, 2], ["self", "other", "self", "other"]]
    )
    expected = pd.Series(["a", "x", "c", "z"], index=indices)
    tm.assert_series_equal(result, expected)


def test_compare_unaligned_objects():
    # test Series with different indices
    msg = "Can only compare identically-labeled Series objects"
    with pytest.raises(ValueError, match=msg):
        ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
        ser2 = pd.Series([1, 2, 3], index=["a", "b", "d"])
        ser1.compare(ser2)

    # test Series with different lengths
    msg = "Can only compare identically-labeled Series objects"
    with pytest.raises(ValueError, match=msg):
        ser1 = pd.Series([1, 2, 3])
        ser2 = pd.Series([1, 2, 3, 4])
        ser1.compare(ser2)


def test_compare_datetime64_and_string():
    # Issue https://github.com/pandas-dev/pandas/issues/45506
    # Catch OverflowError when comparing datetime64 and string
    data = [
        {"a": "2015-07-01", "b": "08335394550"},
        {"a": "2015-07-02", "b": "+49 (0) 0345 300033"},
        {"a": "2015-07-03", "b": "+49(0)2598 04457"},
        {"a": "2015-07-04", "b": "0741470003"},
        {"a": "2015-07-05", "b": "04181 83668"},
    ]
    dtypes = {"a": "datetime64[ns]", "b": "string"}
    df = pd.DataFrame(data=data).astype(dtypes)

    result_eq1 = df["a"].eq(df["b"])
    result_eq2 = df["a"] == df["b"]
    result_neq = df["a"] != df["b"]

    expected_eq = pd.Series([False] * 5)  # For .eq and ==
    expected_neq = pd.Series([True] * 5)  # For !=

    tm.assert_series_equal(result_eq1, expected_eq)
    tm.assert_series_equal(result_eq2, expected_eq)
    tm.assert_series_equal(result_neq, expected_neq)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_deprecate_nonkeyword_arguments.py ===
"""
Tests for the `deprecate_nonkeyword_arguments` decorator
"""

import inspect

from pandas.util._decorators import deprecate_nonkeyword_arguments

import pandas._testing as tm


@deprecate_nonkeyword_arguments(
    version="1.1", allowed_args=["a", "b"], name="f_add_inputs"
)
def f(a, b=0, c=0, d=0):
    return a + b + c + d


def test_f_signature():
    assert str(inspect.signature(f)) == "(a, b=0, *, c=0, d=0)"


def test_one_argument():
    with tm.assert_produces_warning(None):
        assert f(19) == 19


def test_one_and_one_arguments():
    with tm.assert_produces_warning(None):
        assert f(19, d=6) == 25


def test_two_arguments():
    with tm.assert_produces_warning(None):
        assert f(1, 5) == 6


def test_two_and_two_arguments():
    with tm.assert_produces_warning(None):
        assert f(1, 3, c=3, d=5) == 12


def test_three_arguments():
    with tm.assert_produces_warning(FutureWarning):
        assert f(6, 3, 3) == 12


def test_four_arguments():
    with tm.assert_produces_warning(FutureWarning):
        assert f(1, 2, 3, 4) == 10


def test_three_arguments_with_name_in_warning():
    msg = (
        "Starting with pandas version 1.1 all arguments of f_add_inputs "
        "except for the arguments 'a' and 'b' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert f(6, 3, 3) == 12


@deprecate_nonkeyword_arguments(version="1.1")
def g(a, b=0, c=0, d=0):
    with tm.assert_produces_warning(None):
        return a + b + c + d


def test_g_signature():
    assert str(inspect.signature(g)) == "(a, *, b=0, c=0, d=0)"


def test_one_and_three_arguments_default_allowed_args():
    with tm.assert_produces_warning(None):
        assert g(1, b=3, c=3, d=5) == 12


def test_three_arguments_default_allowed_args():
    with tm.assert_produces_warning(FutureWarning):
        assert g(6, 3, 3) == 12


def test_three_positional_argument_with_warning_message_analysis():
    msg = (
        "Starting with pandas version 1.1 all arguments of g "
        "except for the argument 'a' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert g(6, 3, 3) == 12


@deprecate_nonkeyword_arguments(version="1.1")
def h(a=0, b=0, c=0, d=0):
    return a + b + c + d


def test_h_signature():
    assert str(inspect.signature(h)) == "(*, a=0, b=0, c=0, d=0)"


def test_all_keyword_arguments():
    with tm.assert_produces_warning(None):
        assert h(a=1, b=2) == 3


def test_one_positional_argument():
    with tm.assert_produces_warning(FutureWarning):
        assert h(23) == 23


def test_one_positional_argument_with_warning_message_analysis():
    msg = "Starting with pandas version 1.1 all arguments of h will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert h(19) == 19


@deprecate_nonkeyword_arguments(version="1.1")
def i(a=0, /, b=0, *, c=0, d=0):
    return a + b + c + d


def test_i_signature():
    assert str(inspect.signature(i)) == "(*, a=0, b=0, c=0, d=0)"


class Foo:
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "bar"])
    def baz(self, bar=None, foobar=None):  # pylint: disable=disallowed-name
        ...


def test_foo_signature():
    assert str(inspect.signature(Foo.baz)) == "(self, bar=None, *, foobar=None)"


def test_class():
    msg = (
        r"In a future version of pandas all arguments of Foo\.baz "
        r"except for the argument \'bar\' will be keyword-only"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        Foo().baz("qux", "quox")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_hadamard.py ===
from sympy.matrices.dense import Matrix, eye
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
from sympy.core import symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy

from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions import (HadamardProduct, hadamard_product, HadamardPower, hadamard_power)

n, m, k = symbols('n,m,k')
Z = MatrixSymbol('Z', n, n)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', n, m)
C = MatrixSymbol('C', m, k)


def test_HadamardProduct():
    assert HadamardProduct(A, B, A).shape == A.shape

    raises(TypeError,  lambda: HadamardProduct(A, n))
    raises(TypeError,  lambda: HadamardProduct(A, 1))

    assert HadamardProduct(A, 2*B, -A)[1, 1] == \
            -2 * A[1, 1] * B[1, 1] * A[1, 1]

    mix = HadamardProduct(Z*A, B)*C
    assert mix.shape == (n, k)

    assert set(HadamardProduct(A, B, A).T.args) == {A.T, A.T, B.T}


def test_HadamardProduct_isnt_commutative():
    assert HadamardProduct(A, B) != HadamardProduct(B, A)


def test_mixed_indexing():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    Z = MatrixSymbol('Z', 2, 2)

    assert (X*HadamardProduct(Y, Z))[0, 0] == \
            X[0, 0]*Y[0, 0]*Z[0, 0] + X[0, 1]*Y[1, 0]*Z[1, 0]


def test_canonicalize():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    with warns_deprecated_sympy():
        expr = HadamardProduct(X, check=False)
    assert isinstance(expr, HadamardProduct)
    expr2 = expr.doit() # unpack is called
    assert isinstance(expr2, MatrixSymbol)
    Z = ZeroMatrix(2, 2)
    U = OneMatrix(2, 2)
    assert HadamardProduct(Z, X).doit() == Z
    assert HadamardProduct(U, X, X, U).doit() == HadamardPower(X, 2)
    assert HadamardProduct(X, U, Y).doit() == HadamardProduct(X, Y)
    assert HadamardProduct(X, Z, U, Y).doit() == Z


def test_hadamard():
    m, n, p = symbols('m, n, p', integer=True)
    A = MatrixSymbol('A', m, n)
    B = MatrixSymbol('B', m, n)
    X = MatrixSymbol('X', m, m)
    I = Identity(m)

    raises(TypeError, lambda: hadamard_product())
    assert hadamard_product(A) == A
    assert isinstance(hadamard_product(A, B), HadamardProduct)
    assert hadamard_product(A, B).doit() == hadamard_product(A, B)
    assert hadamard_product(X, I) == HadamardProduct(I, X)
    assert isinstance(hadamard_product(X, I), HadamardProduct)

    a = MatrixSymbol("a", k, 1)
    expr = MatAdd(ZeroMatrix(k, 1), OneMatrix(k, 1))
    expr = HadamardProduct(expr, a)
    assert expr.doit() == a

    raises(ValueError, lambda: HadamardProduct())


def test_hadamard_product_with_explicit_mat():
    A = MatrixSymbol("A", 3, 3).as_explicit()
    B = MatrixSymbol("B", 3, 3).as_explicit()
    X = MatrixSymbol("X", 3, 3)
    expr = hadamard_product(A, B)
    ret = Matrix([i*j for i, j in zip(A, B)]).reshape(3, 3)
    assert expr == ret
    expr = hadamard_product(A, X, B)
    assert expr == HadamardProduct(ret, X)
    expr = hadamard_product(eye(3), A)
    assert expr == Matrix([[A[0, 0], 0, 0], [0, A[1, 1], 0], [0, 0, A[2, 2]]])
    expr = hadamard_product(eye(3), eye(3))
    assert expr == eye(3)


def test_hadamard_power():
    m, n, p = symbols('m, n, p', integer=True)
    A = MatrixSymbol('A', m, n)

    assert hadamard_power(A, 1) == A
    assert isinstance(hadamard_power(A, 2), HadamardPower)
    assert hadamard_power(A, n).T == hadamard_power(A.T, n)
    assert hadamard_power(A, n)[0, 0] == A[0, 0]**n
    assert hadamard_power(m, n) == m**n
    raises(ValueError, lambda: hadamard_power(A, A))


def test_hadamard_power_explicit():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    a, b = symbols('a b')

    assert HadamardPower(a, b) == a**b

    assert HadamardPower(a, B).as_explicit() == \
        Matrix([
            [a**B[0, 0], a**B[0, 1]],
            [a**B[1, 0], a**B[1, 1]]])

    assert HadamardPower(A, b).as_explicit() == \
        Matrix([
            [A[0, 0]**b, A[0, 1]**b],
            [A[1, 0]**b, A[1, 1]**b]])

    assert HadamardPower(A, B).as_explicit() == \
        Matrix([
            [A[0, 0]**B[0, 0], A[0, 1]**B[0, 1]],
            [A[1, 0]**B[1, 0], A[1, 1]**B[1, 1]]])


def test_shape_error():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 3)
    raises(ShapeError, lambda: HadamardProduct(A, B))
    raises(ShapeError, lambda: HadamardPower(A, B))
    A = MatrixSymbol('A', 3, 2)
    raises(ShapeError, lambda: HadamardProduct(A, B))
    raises(ShapeError, lambda: HadamardPower(A, B))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\tests\test_powerset.py ===
from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL


def test_powerset_creation():
    assert unchanged(PowerSet, FiniteSet(1, 2))
    assert unchanged(PowerSet, S.EmptySet)
    raises(ValueError, lambda: PowerSet(123))
    assert unchanged(PowerSet, S.Reals)
    assert unchanged(PowerSet, S.Integers)


def test_powerset_rewrite_FiniteSet():
    assert PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet) == \
        FiniteSet(S.EmptySet, FiniteSet(1), FiniteSet(2), FiniteSet(1, 2))
    assert PowerSet(S.EmptySet).rewrite(FiniteSet) == FiniteSet(S.EmptySet)
    assert PowerSet(S.Naturals).rewrite(FiniteSet) == PowerSet(S.Naturals)


def test_finiteset_rewrite_powerset():
    assert FiniteSet(S.EmptySet).rewrite(PowerSet) == PowerSet(S.EmptySet)
    assert FiniteSet(
        S.EmptySet, FiniteSet(1),
        FiniteSet(2), FiniteSet(1, 2)).rewrite(PowerSet) == \
            PowerSet(FiniteSet(1, 2))
    assert FiniteSet(1, 2, 3).rewrite(PowerSet) == FiniteSet(1, 2, 3)


def test_powerset__contains__():
    subset_series = [
        S.EmptySet,
        FiniteSet(1, 2),
        S.Naturals,
        S.Naturals0,
        S.Integers,
        S.Rationals,
        S.Reals,
        S.Complexes]

    l = len(subset_series)
    for i in range(l):
        for j in range(l):
            if i <= j:
                assert subset_series[i] in \
                    PowerSet(subset_series[j], evaluate=False)
            else:
                assert subset_series[i] not in \
                    PowerSet(subset_series[j], evaluate=False)


@XFAIL
def test_failing_powerset__contains__():
    # XXX These are failing when evaluate=True,
    # but using unevaluated PowerSet works fine.
    assert FiniteSet(1, 2) not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)


def test_powerset__len__():
    A = PowerSet(S.EmptySet, evaluate=False)
    assert len(A) == 1
    A = PowerSet(A, evaluate=False)
    assert len(A) == 2
    A = PowerSet(A, evaluate=False)
    assert len(A) == 4
    A = PowerSet(A, evaluate=False)
    assert len(A) == 16


def test_powerset__iter__():
    a = PowerSet(FiniteSet(1, 2)).__iter__()
    assert next(a) == S.EmptySet
    assert next(a) == FiniteSet(1)
    assert next(a) == FiniteSet(2)
    assert next(a) == FiniteSet(1, 2)

    a = PowerSet(S.Naturals).__iter__()
    assert next(a) == S.EmptySet
    assert next(a) == FiniteSet(1)
    assert next(a) == FiniteSet(2)
    assert next(a) == FiniteSet(1, 2)
    assert next(a) == FiniteSet(3)
    assert next(a) == FiniteSet(1, 3)
    assert next(a) == FiniteSet(2, 3)
    assert next(a) == FiniteSet(1, 2, 3)


def test_powerset_contains():
    A = PowerSet(FiniteSet(1), evaluate=False)
    assert A.contains(2) == Contains(2, A)

    x = Symbol('x')

    A = PowerSet(FiniteSet(x), evaluate=False)
    assert A.contains(FiniteSet(1)) == Contains(FiniteSet(1), A)


def test_powerset_method():
    # EmptySet
    A = FiniteSet()
    pset = A.powerset()
    assert len(pset) == 1
    assert pset ==  FiniteSet(S.EmptySet)

    # FiniteSets
    A = FiniteSet(1, 2)
    pset = A.powerset()
    assert len(pset) == 2**len(A)
    assert pset == FiniteSet(FiniteSet(), FiniteSet(1),
                             FiniteSet(2), A)
    # Not finite sets
    A = Interval(0, 1)
    assert A.powerset() == PowerSet(A)

def test_is_subset():
    # covers line 101-102
    # initialize powerset(1), which is a subset of powerset(1,2)
    subset = PowerSet(FiniteSet(1))
    pset = PowerSet(FiniteSet(1, 2))
    bad_set = PowerSet(FiniteSet(2, 3))
    # assert "subset" is subset of pset == True
    assert subset.is_subset(pset)
    # assert "bad_set" is subset of pset == False
    assert not pset.is_subset(bad_set)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_non_unique.py ===
from textwrap import dedent

import pytest

from pandas import (
    DataFrame,
    IndexSlice,
)

pytest.importorskip("jinja2")

from pandas.io.formats.style import Styler


@pytest.fixture
def df():
    return DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["i", "j", "j"],
        columns=["c", "d", "d"],
        dtype=float,
    )


@pytest.fixture
def styler(df):
    return Styler(df, uuid_len=0)


def test_format_non_unique(df):
    # GH 41269

    # test dict
    html = df.style.format({"d": "{:.1f}"}).to_html()
    for val in ["1.000000<", "4.000000<", "7.000000<"]:
        assert val in html
    for val in ["2.0<", "3.0<", "5.0<", "6.0<", "8.0<", "9.0<"]:
        assert val in html

    # test subset
    html = df.style.format(precision=1, subset=IndexSlice["j", "d"]).to_html()
    for val in ["1.000000<", "4.000000<", "7.000000<", "2.000000<", "3.000000<"]:
        assert val in html
    for val in ["5.0<", "6.0<", "8.0<", "9.0<"]:
        assert val in html


@pytest.mark.parametrize("func", ["apply", "map"])
def test_apply_map_non_unique_raises(df, func):
    # GH 41269
    if func == "apply":
        op = lambda s: ["color: red;"] * len(s)
    else:
        op = lambda v: "color: red;"

    with pytest.raises(KeyError, match="`Styler.apply` and `.map` are not"):
        getattr(df.style, func)(op)._compute()


def test_table_styles_dict_non_unique_index(styler):
    styles = styler.set_table_styles(
        {"j": [{"selector": "td", "props": "a: v;"}]}, axis=1
    ).table_styles
    assert styles == [
        {"selector": "td.row1", "props": [("a", "v")]},
        {"selector": "td.row2", "props": [("a", "v")]},
    ]


def test_table_styles_dict_non_unique_columns(styler):
    styles = styler.set_table_styles(
        {"d": [{"selector": "td", "props": "a: v;"}]}, axis=0
    ).table_styles
    assert styles == [
        {"selector": "td.col1", "props": [("a", "v")]},
        {"selector": "td.col2", "props": [("a", "v")]},
    ]


def test_tooltips_non_unique_raises(styler):
    # ttips has unique keys
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "b"])
    styler.set_tooltips(ttips=ttips)  # OK

    # ttips has non-unique columns
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "c"], index=["a", "b"])
    with pytest.raises(KeyError, match="Tooltips render only if `ttips` has unique"):
        styler.set_tooltips(ttips=ttips)

    # ttips has non-unique index
    ttips = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "a"])
    with pytest.raises(KeyError, match="Tooltips render only if `ttips` has unique"):
        styler.set_tooltips(ttips=ttips)


def test_set_td_classes_non_unique_raises(styler):
    # classes has unique keys
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "b"])
    styler.set_td_classes(classes=classes)  # OK

    # classes has non-unique columns
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "c"], index=["a", "b"])
    with pytest.raises(KeyError, match="Classes render only if `classes` has unique"):
        styler.set_td_classes(classes=classes)

    # classes has non-unique index
    classes = DataFrame([["1", "2"], ["3", "4"]], columns=["c", "d"], index=["a", "a"])
    with pytest.raises(KeyError, match="Classes render only if `classes` has unique"):
        styler.set_td_classes(classes=classes)


def test_hide_columns_non_unique(styler):
    ctx = styler.hide(["d"], axis="columns")._translate(True, True)

    assert ctx["head"][0][1]["display_value"] == "c"
    assert ctx["head"][0][1]["is_visible"] is True

    assert ctx["head"][0][2]["display_value"] == "d"
    assert ctx["head"][0][2]["is_visible"] is False

    assert ctx["head"][0][3]["display_value"] == "d"
    assert ctx["head"][0][3]["is_visible"] is False

    assert ctx["body"][0][1]["is_visible"] is True
    assert ctx["body"][0][2]["is_visible"] is False
    assert ctx["body"][0][3]["is_visible"] is False


def test_latex_non_unique(styler):
    result = styler.to_latex()
    assert result == dedent(
        """\
        \\begin{tabular}{lrrr}
         & c & d & d \\\\
        i & 1.000000 & 2.000000 & 3.000000 \\\\
        j & 4.000000 & 5.000000 & 6.000000 \\\\
        j & 7.000000 & 8.000000 & 9.000000 \\\\
        \\end{tabular}
    """
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_noncommutative.py ===
"""Tests for noncommutative symbols and expressions."""

from sympy.core.function import expand
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import (cancel, factor)
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.radsimp import (collect, radsimp, rcollect)
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import (posify, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.abc import x, y, z
from sympy.testing.pytest import XFAIL

A, B, C = symbols("A B C", commutative=False)
X = symbols("X", commutative=False, hermitian=True)
Y = symbols("Y", commutative=False, antihermitian=True)


def test_adjoint():
    assert adjoint(A).is_commutative is False
    assert adjoint(A*A) == adjoint(A)**2
    assert adjoint(A*B) == adjoint(B)*adjoint(A)
    assert adjoint(A*B**2) == adjoint(B)**2*adjoint(A)
    assert adjoint(A*B - B*A) == adjoint(B)*adjoint(A) - adjoint(A)*adjoint(B)
    assert adjoint(A + I*B) == adjoint(A) - I*adjoint(B)

    assert adjoint(X) == X
    assert adjoint(-I*X) == I*X
    assert adjoint(Y) == -Y
    assert adjoint(-I*Y) == -I*Y

    assert adjoint(X) == conjugate(transpose(X))
    assert adjoint(Y) == conjugate(transpose(Y))
    assert adjoint(X) == transpose(conjugate(X))
    assert adjoint(Y) == transpose(conjugate(Y))


def test_cancel():
    assert cancel(A*B - B*A) == A*B - B*A
    assert cancel(A*B*(x - 1)) == A*B*(x - 1)
    assert cancel(A*B*(x**2 - 1)/(x + 1)) == A*B*(x - 1)
    assert cancel(A*B*(x**2 - 1)/(x + 1) - B*A*(x - 1)) == A*B*(x - 1) + (1 - x)*B*A


@XFAIL
def test_collect():
    assert collect(A*B - B*A, A) == A*B - B*A
    assert collect(A*B - B*A, B) == A*B - B*A
    assert collect(A*B - B*A, x) == A*B - B*A


def test_combsimp():
    assert combsimp(A*B - B*A) == A*B - B*A


def test_gammasimp():
    assert gammasimp(A*B - B*A) == A*B - B*A


def test_conjugate():
    assert conjugate(A).is_commutative is False
    assert (A*A).conjugate() == conjugate(A)**2
    assert (A*B).conjugate() == conjugate(A)*conjugate(B)
    assert (A*B**2).conjugate() == conjugate(A)*conjugate(B)**2
    assert (A*B - B*A).conjugate() == \
        conjugate(A)*conjugate(B) - conjugate(B)*conjugate(A)
    assert (A*B).conjugate() - (B*A).conjugate() == \
        conjugate(A)*conjugate(B) - conjugate(B)*conjugate(A)
    assert (A + I*B).conjugate() == conjugate(A) - I*conjugate(B)


def test_expand():
    assert expand((A*B)**2) == A*B*A*B
    assert expand(A*B - B*A) == A*B - B*A
    assert expand((A*B/A)**2) == A*B*B/A
    assert expand(B*A*(A + B)*B) == B*A**2*B + B*A*B**2
    assert expand(B*A*(A + C)*B) == B*A**2*B + B*A*C*B


def test_factor():
    assert factor(A*B - B*A) == A*B - B*A


def test_posify():
    assert posify(A)[0].is_commutative is False
    for q in (A*B/A, (A*B/A)**2, (A*B)**2, A*B - B*A):
        p = posify(q)
        assert p[0].subs(p[1]) == q


def test_radsimp():
    assert radsimp(A*B - B*A) == A*B - B*A


@XFAIL
def test_ratsimp():
    assert ratsimp(A*B - B*A) == A*B - B*A


@XFAIL
def test_rcollect():
    assert rcollect(A*B - B*A, A) == A*B - B*A
    assert rcollect(A*B - B*A, B) == A*B - B*A
    assert rcollect(A*B - B*A, x) == A*B - B*A


def test_simplify():
    assert simplify(A*B - B*A) == A*B - B*A


def test_subs():
    assert (x*y*A).subs(x*y, z) == A*z
    assert (x*A*B).subs(x*A, C) == C*B
    assert (x*A*x*x).subs(x**2*A, C) == x*C
    assert (x*A*x*B).subs(x**2*A, C) == C*B
    assert (A**2*B**2).subs(A*B**2, C) == A*C
    assert (A*A*A + A*B*A).subs(A*A*A, C) == C + A*B*A


def test_transpose():
    assert transpose(A).is_commutative is False
    assert transpose(A*A) == transpose(A)**2
    assert transpose(A*B) == transpose(B)*transpose(A)
    assert transpose(A*B**2) == transpose(B)**2*transpose(A)
    assert transpose(A*B - B*A) == \
        transpose(B)*transpose(A) - transpose(A)*transpose(B)
    assert transpose(A + I*B) == transpose(A) + I*transpose(B)

    assert transpose(X) == conjugate(X)
    assert transpose(-I*X) == -I*conjugate(X)
    assert transpose(Y) == -conjugate(Y)
    assert transpose(-I*Y) == I*conjugate(Y)


def test_trigsimp():
    assert trigsimp(A*sin(x)**2 + A*cos(x)**2) == A

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_builder_registry.py ===
"""Tests of the builder registry."""

import pytest
import warnings
from typing import Type

from bs4 import BeautifulSoup
from bs4.builder import (
    builder_registry as registry,
    TreeBuilder,
    TreeBuilderRegistry,
)
from bs4.builder._htmlparser import HTMLParserTreeBuilder

from . import (
    HTML5LIB_PRESENT,
    LXML_PRESENT,
)

if HTML5LIB_PRESENT:
    from bs4.builder._html5lib import HTML5TreeBuilder

if LXML_PRESENT:
    from bs4.builder._lxml import (
        LXMLTreeBuilderForXML,
        LXMLTreeBuilder,
    )


# TODO: Split out the lxml and html5lib tests into their own classes
# and gate with pytest.mark.skipIf.
class TestBuiltInRegistry(object):
    """Test the built-in registry with the default builders registered."""

    def test_combination(self):
        assert registry.lookup("strict", "html") == HTMLParserTreeBuilder
        if LXML_PRESENT:
            assert registry.lookup("fast", "html") == LXMLTreeBuilder
            assert registry.lookup("permissive", "xml") == LXMLTreeBuilderForXML
        if HTML5LIB_PRESENT:
            assert registry.lookup("html5lib", "html") == HTML5TreeBuilder

    def test_lookup_by_markup_type(self):
        if LXML_PRESENT:
            assert registry.lookup("html") == LXMLTreeBuilder
            assert registry.lookup("xml") == LXMLTreeBuilderForXML
        else:
            assert registry.lookup("xml") is None
            if HTML5LIB_PRESENT:
                assert registry.lookup("html") == HTML5TreeBuilder
            else:
                assert registry.lookup("html") == HTMLParserTreeBuilder

    def test_named_library(self):
        if LXML_PRESENT:
            assert registry.lookup("lxml", "xml") == LXMLTreeBuilderForXML
            assert registry.lookup("lxml", "html") == LXMLTreeBuilder
        if HTML5LIB_PRESENT:
            assert registry.lookup("html5lib") == HTML5TreeBuilder

        assert registry.lookup("html.parser") == HTMLParserTreeBuilder

    def test_beautifulsoup_constructor_does_lookup(self):
        with warnings.catch_warnings(record=True):
            # This will create a warning about not explicitly
            # specifying a parser, but we'll ignore it.

            # You can pass in a string.
            BeautifulSoup("", features="html")
            # Or a list of strings.
            BeautifulSoup("", features=["html", "fast"])
            pass

        # You'll get an exception if BS can't find an appropriate
        # builder.
        with pytest.raises(ValueError):
            BeautifulSoup("", features="no-such-feature")


class TestRegistry(object):
    """Test the TreeBuilderRegistry class in general."""

    def setup_method(self):
        self.registry = TreeBuilderRegistry()

    def builder_for_features(self, *feature_list: str) -> Type[TreeBuilder]:
        cls = type(
            "Builder_" + "_".join(feature_list), (object,), {"features": feature_list}
        )

        self.registry.register(cls)
        return cls

    def test_register_with_no_features(self):
        builder = self.builder_for_features()

        # Since the builder advertises no features, you can't find it
        # by looking up features.
        assert self.registry.lookup("foo") is None

        # But you can find it by doing a lookup with no features, if
        # this happens to be the only registered builder.
        assert self.registry.lookup() == builder

    def test_register_with_features_makes_lookup_succeed(self):
        builder = self.builder_for_features("foo", "bar")
        assert self.registry.lookup("foo") is builder
        assert self.registry.lookup("bar") is builder

    def test_lookup_fails_when_no_builder_implements_feature(self):
        assert self.registry.lookup("baz") is None

    def test_lookup_gets_most_recent_registration_when_no_feature_specified(self):
        self.builder_for_features("foo")
        builder2 = self.builder_for_features("bar")
        assert self.registry.lookup() == builder2

    def test_lookup_fails_when_no_tree_builders_registered(self):
        assert self.registry.lookup() is None

    def test_lookup_gets_most_recent_builder_supporting_all_features(self):
        self.builder_for_features("foo")
        self.builder_for_features("bar")
        has_both_early = self.builder_for_features("foo", "bar", "baz")
        has_both_late = self.builder_for_features("foo", "bar", "quux")
        self.builder_for_features("bar")
        self.builder_for_features("foo")

        # There are two builders featuring 'foo' and 'bar', but
        # the one that also features 'quux' was registered later.
        assert self.registry.lookup("foo", "bar") == has_both_late

        # There is only one builder featuring 'foo', 'bar', and 'baz'.
        assert self.registry.lookup("foo", "bar", "baz") == has_both_early

    def test_lookup_fails_when_cannot_reconcile_requested_features(self):
        self.builder_for_features("foo", "bar")
        self.builder_for_features("foo", "baz")
        assert self.registry.lookup("bar", "baz") is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\conftest.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import Index


@pytest.fixture(params=[1, np.array(1, dtype=np.int64)])
def one(request):
    """
    Several variants of integer value 1. The zero-dim integer array
    behaves like an integer.

    This fixture can be used to check that datetimelike indexes handle
    addition and subtraction of integers and zero-dimensional arrays
    of integers.

    Examples
    --------
    dti = pd.date_range('2016-01-01', periods=2, freq='h')
    dti
    DatetimeIndex(['2016-01-01 00:00:00', '2016-01-01 01:00:00'],
    dtype='datetime64[ns]', freq='h')
    dti + one
    DatetimeIndex(['2016-01-01 01:00:00', '2016-01-01 02:00:00'],
    dtype='datetime64[ns]', freq='h')
    """
    return request.param


zeros = [
    box_cls([0] * 5, dtype=dtype)
    for box_cls in [Index, np.array, pd.array]
    for dtype in [np.int64, np.uint64, np.float64]
]
zeros.extend([box_cls([-0.0] * 5, dtype=np.float64) for box_cls in [Index, np.array]])
zeros.extend([np.array(0, dtype=dtype) for dtype in [np.int64, np.uint64, np.float64]])
zeros.extend([np.array(-0.0, dtype=np.float64)])
zeros.extend([0, 0.0, -0.0])


@pytest.fixture(params=zeros)
def zero(request):
    """
    Several types of scalar zeros and length 5 vectors of zeros.

    This fixture can be used to check that numeric-dtype indexes handle
    division by any zero numeric-dtype.

    Uses vector of length 5 for broadcasting with `numeric_idx` fixture,
    which creates numeric-dtype vectors also of length 5.

    Examples
    --------
    arr = RangeIndex(5)
    arr / zeros
    Index([nan, inf, inf, inf, inf], dtype='float64')
    """
    return request.param


# ------------------------------------------------------------------
# Scalar Fixtures


@pytest.fixture(
    params=[
        pd.Timedelta("10m7s").to_pytimedelta(),
        pd.Timedelta("10m7s"),
        pd.Timedelta("10m7s").to_timedelta64(),
    ],
    ids=lambda x: type(x).__name__,
)
def scalar_td(request):
    """
    Several variants of Timedelta scalars representing 10 minutes and 7 seconds.
    """
    return request.param


@pytest.fixture(
    params=[
        pd.offsets.Day(3),
        pd.offsets.Hour(72),
        pd.Timedelta(days=3).to_pytimedelta(),
        pd.Timedelta("72:00:00"),
        np.timedelta64(3, "D"),
        np.timedelta64(72, "h"),
    ],
    ids=lambda x: type(x).__name__,
)
def three_days(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 3-day timedelta
    """
    return request.param


@pytest.fixture(
    params=[
        pd.offsets.Hour(2),
        pd.offsets.Minute(120),
        pd.Timedelta(hours=2).to_pytimedelta(),
        pd.Timedelta(seconds=2 * 3600),
        np.timedelta64(2, "h"),
        np.timedelta64(120, "m"),
    ],
    ids=lambda x: type(x).__name__,
)
def two_hours(request):
    """
    Several timedelta-like and DateOffset objects that each represent
    a 2-hour timedelta
    """
    return request.param


_common_mismatch = [
    pd.offsets.YearBegin(2),
    pd.offsets.MonthBegin(1),
    pd.offsets.Minute(),
]


@pytest.fixture(
    params=[
        np.timedelta64(4, "h"),
        pd.Timedelta(hours=23).to_pytimedelta(),
        pd.Timedelta("23:00:00"),
    ]
    + _common_mismatch
)
def not_daily(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Daily frequencies.
    """
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_dtypes.py ===
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas import (
    Categorical,
    CategoricalIndex,
    Index,
    IntervalIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestCategoricalDtypes:
    def test_categories_match_up_to_permutation(self):
        # test dtype comparisons between cats

        c1 = Categorical(list("aabca"), categories=list("abc"), ordered=False)
        c2 = Categorical(list("aabca"), categories=list("cab"), ordered=False)
        c3 = Categorical(list("aabca"), categories=list("cab"), ordered=True)
        assert c1._categories_match_up_to_permutation(c1)
        assert c2._categories_match_up_to_permutation(c2)
        assert c3._categories_match_up_to_permutation(c3)
        assert c1._categories_match_up_to_permutation(c2)
        assert not c1._categories_match_up_to_permutation(c3)
        assert not c1._categories_match_up_to_permutation(Index(list("aabca")))
        assert not c1._categories_match_up_to_permutation(c1.astype(object))
        assert c1._categories_match_up_to_permutation(CategoricalIndex(c1))
        assert c1._categories_match_up_to_permutation(
            CategoricalIndex(c1, categories=list("cab"))
        )
        assert not c1._categories_match_up_to_permutation(
            CategoricalIndex(c1, ordered=True)
        )

        # GH 16659
        s1 = Series(c1)
        s2 = Series(c2)
        s3 = Series(c3)
        assert c1._categories_match_up_to_permutation(s1)
        assert c2._categories_match_up_to_permutation(s2)
        assert c3._categories_match_up_to_permutation(s3)
        assert c1._categories_match_up_to_permutation(s2)
        assert not c1._categories_match_up_to_permutation(s3)
        assert not c1._categories_match_up_to_permutation(s1.astype(object))

    def test_set_dtype_same(self):
        c = Categorical(["a", "b", "c"])
        result = c._set_dtype(CategoricalDtype(["a", "b", "c"]))
        tm.assert_categorical_equal(result, c)

    def test_set_dtype_new_categories(self):
        c = Categorical(["a", "b", "c"])
        result = c._set_dtype(CategoricalDtype(list("abcd")))
        tm.assert_numpy_array_equal(result.codes, c.codes)
        tm.assert_index_equal(result.dtype.categories, Index(list("abcd")))

    @pytest.mark.parametrize(
        "values, categories, new_categories",
        [
            # No NaNs, same cats, same order
            (["a", "b", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["a", "b", "a"], ["a", "b"], ["b", "a"]),
            # Same, unsorted
            (["b", "a", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["b", "a", "a"], ["a", "b"], ["b", "a"]),
            # NaNs
            (["a", "b", "c"], ["a", "b"], ["a", "b"]),
            (["a", "b", "c"], ["a", "b"], ["b", "a"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            # Introduce NaNs
            (["a", "b", "c"], ["a", "b"], ["a"]),
            (["a", "b", "c"], ["a", "b"], ["b"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            # No overlap
            (["a", "b", "c"], ["a", "b"], ["d", "e"]),
        ],
    )
    @pytest.mark.parametrize("ordered", [True, False])
    def test_set_dtype_many(self, values, categories, new_categories, ordered):
        c = Categorical(values, categories)
        expected = Categorical(values, new_categories, ordered)
        result = c._set_dtype(expected.dtype)
        tm.assert_categorical_equal(result, expected)

    def test_set_dtype_no_overlap(self):
        c = Categorical(["a", "b", "c"], ["d", "e"])
        result = c._set_dtype(CategoricalDtype(["a", "b"]))
        expected = Categorical([None, None, None], categories=["a", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_codes_dtypes(self):
        # GH 8453
        result = Categorical(["foo", "bar", "baz"])
        assert result.codes.dtype == "int8"

        result = Categorical([f"foo{i:05d}" for i in range(400)])
        assert result.codes.dtype == "int16"

        result = Categorical([f"foo{i:05d}" for i in range(40000)])
        assert result.codes.dtype == "int32"

        # adding cats
        result = Categorical(["foo", "bar", "baz"])
        assert result.codes.dtype == "int8"
        result = result.add_categories([f"foo{i:05d}" for i in range(400)])
        assert result.codes.dtype == "int16"

        # removing cats
        result = result.remove_categories([f"foo{i:05d}" for i in range(300)])
        assert result.codes.dtype == "int8"

    def test_iter_python_types(self):
        # GH-19909
        cat = Categorical([1, 2])
        assert isinstance(next(iter(cat)), int)
        assert isinstance(cat.tolist()[0], int)

    def test_iter_python_types_datetime(self):
        cat = Categorical([Timestamp("2017-01-01"), Timestamp("2017-01-02")])
        assert isinstance(next(iter(cat)), Timestamp)
        assert isinstance(cat.tolist()[0], Timestamp)

    def test_interval_index_category(self):
        # GH 38316
        index = IntervalIndex.from_breaks(np.arange(3, dtype="uint64"))

        result = CategoricalIndex(index).dtype.categories
        expected = IntervalIndex.from_arrays(
            [0, 1], [1, 2], dtype="interval[uint64, right]"
        )
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_update.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    CategoricalDtype,
    DataFrame,
    NaT,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestUpdate:
    def test_update(self, using_copy_on_write):
        s = Series([1.5, np.nan, 3.0, 4.0, np.nan])
        s2 = Series([np.nan, 3.5, np.nan, 5.0])
        s.update(s2)

        expected = Series([1.5, 3.5, 3.0, 5.0, np.nan])
        tm.assert_series_equal(s, expected)

        # GH 3217
        df = DataFrame([{"a": 1}, {"a": 3, "b": 2}])
        df["c"] = np.nan
        # Cast to object to avoid upcast when setting "foo"
        df["c"] = df["c"].astype(object)
        df_orig = df.copy()

        if using_copy_on_write:
            with tm.raises_chained_assignment_error():
                df["c"].update(Series(["foo"], index=[0]))
            expected = df_orig
        else:
            with tm.assert_produces_warning(FutureWarning, match="inplace method"):
                df["c"].update(Series(["foo"], index=[0]))
            expected = DataFrame(
                [[1, np.nan, "foo"], [3, 2.0, np.nan]], columns=["a", "b", "c"]
            )
            expected["c"] = expected["c"].astype(object)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "other, dtype, expected, warn",
        [
            # other is int
            ([61, 63], "int32", Series([10, 61, 12], dtype="int32"), None),
            ([61, 63], "int64", Series([10, 61, 12]), None),
            ([61, 63], float, Series([10.0, 61.0, 12.0]), None),
            ([61, 63], object, Series([10, 61, 12], dtype=object), None),
            # other is float, but can be cast to int
            ([61.0, 63.0], "int32", Series([10, 61, 12], dtype="int32"), None),
            ([61.0, 63.0], "int64", Series([10, 61, 12]), None),
            ([61.0, 63.0], float, Series([10.0, 61.0, 12.0]), None),
            ([61.0, 63.0], object, Series([10, 61.0, 12], dtype=object), None),
            # others is float, cannot be cast to int
            ([61.1, 63.1], "int32", Series([10.0, 61.1, 12.0]), FutureWarning),
            ([61.1, 63.1], "int64", Series([10.0, 61.1, 12.0]), FutureWarning),
            ([61.1, 63.1], float, Series([10.0, 61.1, 12.0]), None),
            ([61.1, 63.1], object, Series([10, 61.1, 12], dtype=object), None),
            # other is object, cannot be cast
            ([(61,), (63,)], "int32", Series([10, (61,), 12]), FutureWarning),
            ([(61,), (63,)], "int64", Series([10, (61,), 12]), FutureWarning),
            ([(61,), (63,)], float, Series([10.0, (61,), 12.0]), FutureWarning),
            ([(61,), (63,)], object, Series([10, (61,), 12]), None),
        ],
    )
    def test_update_dtypes(self, other, dtype, expected, warn):
        ser = Series([10, 11, 12], dtype=dtype)
        other = Series(other, index=[1, 3])
        with tm.assert_produces_warning(warn, match="item of incompatible dtype"):
            ser.update(other)

        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "series, other, expected",
        [
            # update by key
            (
                Series({"a": 1, "b": 2, "c": 3, "d": 4}),
                {"b": 5, "c": np.nan},
                Series({"a": 1, "b": 5, "c": 3, "d": 4}),
            ),
            # update by position
            (Series([1, 2, 3, 4]), [np.nan, 5, 1], Series([1, 5, 1, 4])),
        ],
    )
    def test_update_from_non_series(self, series, other, expected):
        # GH 33215
        series.update(other)
        tm.assert_series_equal(series, expected)

    @pytest.mark.parametrize(
        "data, other, expected, dtype",
        [
            (["a", None], [None, "b"], ["a", "b"], "string[python]"),
            pytest.param(
                ["a", None],
                [None, "b"],
                ["a", "b"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            ([1, None], [None, 2], [1, 2], "Int64"),
            ([True, None], [None, False], [True, False], "boolean"),
            (
                ["a", None],
                [None, "b"],
                ["a", "b"],
                CategoricalDtype(categories=["a", "b"]),
            ),
            (
                [Timestamp(year=2020, month=1, day=1, tz="Europe/London"), NaT],
                [NaT, Timestamp(year=2020, month=1, day=1, tz="Europe/London")],
                [Timestamp(year=2020, month=1, day=1, tz="Europe/London")] * 2,
                "datetime64[ns, Europe/London]",
            ),
        ],
    )
    def test_update_extension_array_series(self, data, other, expected, dtype):
        result = Series(data, dtype=dtype)
        other = Series(other, dtype=dtype)
        expected = Series(expected, dtype=dtype)

        result.update(other)
        tm.assert_series_equal(result, expected)

    def test_update_with_categorical_type(self):
        # GH 25744
        dtype = CategoricalDtype(["a", "b", "c", "d"])
        s1 = Series(["a", "b", "c"], index=[1, 2, 3], dtype=dtype)
        s2 = Series(["b", "a"], index=[1, 2], dtype=dtype)
        s1.update(s2)
        result = s1
        expected = Series(["b", "a", "c"], index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_pythonrational.py ===
"""Tests for PythonRational type. """

from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises

def test_PythonRational__init__():
    assert QQ(0).numerator == 0
    assert QQ(0).denominator == 1
    assert QQ(0, 1).numerator == 0
    assert QQ(0, 1).denominator == 1
    assert QQ(0, -1).numerator == 0
    assert QQ(0, -1).denominator == 1

    assert QQ(1).numerator == 1
    assert QQ(1).denominator == 1
    assert QQ(1, 1).numerator == 1
    assert QQ(1, 1).denominator == 1
    assert QQ(-1, -1).numerator == 1
    assert QQ(-1, -1).denominator == 1

    assert QQ(-1).numerator == -1
    assert QQ(-1).denominator == 1
    assert QQ(-1, 1).numerator == -1
    assert QQ(-1, 1).denominator == 1
    assert QQ( 1, -1).numerator == -1
    assert QQ( 1, -1).denominator == 1

    assert QQ(1, 2).numerator == 1
    assert QQ(1, 2).denominator == 2
    assert QQ(3, 4).numerator == 3
    assert QQ(3, 4).denominator == 4

    assert QQ(2, 2).numerator == 1
    assert QQ(2, 2).denominator == 1
    assert QQ(2, 4).numerator == 1
    assert QQ(2, 4).denominator == 2

def test_PythonRational__hash__():
    assert hash(QQ(0)) == hash(0)
    assert hash(QQ(1)) == hash(1)
    assert hash(QQ(117)) == hash(117)

def test_PythonRational__int__():
    assert int(QQ(-1, 4)) == 0
    assert int(QQ( 1, 4)) == 0
    assert int(QQ(-5, 4)) == -1
    assert int(QQ( 5, 4)) == 1

def test_PythonRational__float__():
    assert float(QQ(-1, 2)) == -0.5
    assert float(QQ( 1, 2)) == 0.5

def test_PythonRational__abs__():
    assert abs(QQ(-1, 2)) == QQ(1, 2)
    assert abs(QQ( 1, 2)) == QQ(1, 2)

def test_PythonRational__pos__():
    assert +QQ(-1, 2) == QQ(-1, 2)
    assert +QQ( 1, 2) == QQ( 1, 2)

def test_PythonRational__neg__():
    assert -QQ(-1, 2) == QQ( 1, 2)
    assert -QQ( 1, 2) == QQ(-1, 2)

def test_PythonRational__add__():
    assert QQ(-1, 2) + QQ( 1, 2) == QQ(0)
    assert QQ( 1, 2) + QQ(-1, 2) == QQ(0)

    assert QQ(1, 2) + QQ(1, 2) == QQ(1)
    assert QQ(1, 2) + QQ(3, 2) == QQ(2)
    assert QQ(3, 2) + QQ(1, 2) == QQ(2)
    assert QQ(3, 2) + QQ(3, 2) == QQ(3)

    assert 1 + QQ(1, 2) == QQ(3, 2)
    assert QQ(1, 2) + 1 == QQ(3, 2)

def test_PythonRational__sub__():
    assert QQ(-1, 2) - QQ( 1, 2) == QQ(-1)
    assert QQ( 1, 2) - QQ(-1, 2) == QQ( 1)

    assert QQ(1, 2) - QQ(1, 2) == QQ( 0)
    assert QQ(1, 2) - QQ(3, 2) == QQ(-1)
    assert QQ(3, 2) - QQ(1, 2) == QQ( 1)
    assert QQ(3, 2) - QQ(3, 2) == QQ( 0)

    assert 1 - QQ(1, 2) == QQ( 1, 2)
    assert QQ(1, 2) - 1 == QQ(-1, 2)

def test_PythonRational__mul__():
    assert QQ(-1, 2) * QQ( 1, 2) == QQ(-1, 4)
    assert QQ( 1, 2) * QQ(-1, 2) == QQ(-1, 4)

    assert QQ(1, 2) * QQ(1, 2) == QQ(1, 4)
    assert QQ(1, 2) * QQ(3, 2) == QQ(3, 4)
    assert QQ(3, 2) * QQ(1, 2) == QQ(3, 4)
    assert QQ(3, 2) * QQ(3, 2) == QQ(9, 4)

    assert 2 * QQ(1, 2) == QQ(1)
    assert QQ(1, 2) * 2 == QQ(1)

def test_PythonRational__truediv__():
    assert QQ(-1, 2) / QQ( 1, 2) == QQ(-1)
    assert QQ( 1, 2) / QQ(-1, 2) == QQ(-1)

    assert QQ(1, 2) / QQ(1, 2) == QQ(1)
    assert QQ(1, 2) / QQ(3, 2) == QQ(1, 3)
    assert QQ(3, 2) / QQ(1, 2) == QQ(3)
    assert QQ(3, 2) / QQ(3, 2) == QQ(1)

    assert 2 / QQ(1, 2) == QQ(4)
    assert QQ(1, 2) / 2 == QQ(1, 4)

    raises(ZeroDivisionError, lambda: QQ(1, 2) / QQ(0))
    raises(ZeroDivisionError, lambda: QQ(1, 2) / 0)

def test_PythonRational__pow__():
    assert QQ(1)**10 == QQ(1)
    assert QQ(2)**10 == QQ(1024)

    assert QQ(1)**(-10) == QQ(1)
    assert QQ(2)**(-10) == QQ(1, 1024)

def test_PythonRational__eq__():
    assert (QQ(1, 2) == QQ(1, 2)) is True
    assert (QQ(1, 2) != QQ(1, 2)) is False

    assert (QQ(1, 2) == QQ(1, 3)) is False
    assert (QQ(1, 2) != QQ(1, 3)) is True

def test_PythonRational__lt_le_gt_ge__():
    assert (QQ(1, 2) < QQ(1, 4)) is False
    assert (QQ(1, 2) <= QQ(1, 4)) is False
    assert (QQ(1, 2) > QQ(1, 4)) is True
    assert (QQ(1, 2) >= QQ(1, 4)) is True

    assert (QQ(1, 4) < QQ(1, 2)) is True
    assert (QQ(1, 4) <= QQ(1, 2)) is True
    assert (QQ(1, 4) > QQ(1, 2)) is False
    assert (QQ(1, 4) >= QQ(1, 2)) is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_numeric.py ===
from sympy.core.function import nfloat
from sympy.core.numbers import (Float, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from mpmath import mnorm, mpf
from sympy.solvers import nsolve
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.decorator import conserve_mpmath_dps

@XFAIL
def test_nsolve_fail():
    x = symbols('x')
    # Sometimes it is better to use the numerator (issue 4829)
    # but sometimes it is not (issue 11768) so leave this to
    # the discretion of the user
    ans = nsolve(x**2/(1 - x)/(1 - 2*x)**2 - 100, x, 0)
    assert ans > 0.46 and ans < 0.47


def test_nsolve_denominator():
    x = symbols('x')
    # Test that nsolve uses the full expression (numerator and denominator).
    ans = nsolve((x**2 + 3*x + 2)/(x + 2), -2.1)
    # The root -2 was divided out, so make sure we don't find it.
    assert ans == -1.0

def test_nsolve():
    # onedimensional
    x = Symbol('x')
    assert nsolve(sin(x), 2) - pi.evalf() < 1e-15
    assert nsolve(Eq(2*x, 2), x, -10) == nsolve(2*x - 2, -10)
    # Testing checks on number of inputs
    raises(TypeError, lambda: nsolve(Eq(2*x, 2)))
    raises(TypeError, lambda: nsolve(Eq(2*x, 2), x, 1, 2))
    # multidimensional
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    f1 = 3 * x1**2 - 2 * x2**2 - 1
    f2 = x1**2 - 2 * x1 + x2**2 + 2 * x2 - 8
    f = Matrix((f1, f2)).T
    F = lambdify((x1, x2), f.T, modules='mpmath')
    for x0 in [(-1, 1), (1, -2), (4, 4), (-4, -4)]:
        x = nsolve(f, (x1, x2), x0, tol=1.e-8)
        assert mnorm(F(*x), 1) <= 1.e-10
    # The Chinese mathematician Zhu Shijie was the very first to solve this
    # nonlinear system 700 years ago (z was added to make it 3-dimensional)
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    f1 = -x + 2*y
    f2 = (x**2 + x*(y**2 - 2) - 4*y) / (x + 4)
    f3 = sqrt(x**2 + y**2)*z
    f = Matrix((f1, f2, f3)).T
    F = lambdify((x, y, z), f.T, modules='mpmath')

    def getroot(x0):
        root = nsolve(f, (x, y, z), x0)
        assert mnorm(F(*root), 1) <= 1.e-8
        return root
    assert list(map(round, getroot((1, 1, 1)))) == [2, 1, 0]
    assert nsolve([Eq(
        f1, 0), Eq(f2, 0), Eq(f3, 0)], [x, y, z], (1, 1, 1))  # just see that it works
    a = Symbol('a')
    assert abs(nsolve(1/(0.001 + a)**3 - 6/(0.9 - a)**3, a, 0.3) -
        mpf('0.31883011387318591')) < 1e-15


def test_issue_6408():
    x = Symbol('x')
    assert nsolve(Piecewise((x, x < 1), (x**2, True)), x, 2) == 0


def test_issue_6408_integral():
    x, y = symbols('x y')
    assert nsolve(Integral(x*y, (x, 0, 5)), y, 2) == 0


@conserve_mpmath_dps
def test_increased_dps():
    # Issue 8564
    import mpmath
    mpmath.mp.dps = 128
    x = Symbol('x')
    e1 = x**2 - pi
    q = nsolve(e1, x, 3.0)

    assert abs(sqrt(pi).evalf(128) - q) < 1e-128

def test_nsolve_precision():
    x, y = symbols('x y')
    sol = nsolve(x**2 - pi, x, 3, prec=128)
    assert abs(sqrt(pi).evalf(128) - sol) < 1e-128
    assert isinstance(sol, Float)

    sols = nsolve((y**2 - x, x**2 - pi), (x, y), (3, 3), prec=128)
    assert isinstance(sols, Matrix)
    assert sols.shape == (2, 1)
    assert abs(sqrt(pi).evalf(128) - sols[0]) < 1e-128
    assert abs(sqrt(sqrt(pi)).evalf(128) - sols[1]) < 1e-128
    assert all(isinstance(i, Float) for i in sols)

def test_nsolve_complex():
    x, y = symbols('x y')

    assert nsolve(x**2 + 2, 1j) == sqrt(2.)*I
    assert nsolve(x**2 + 2, I) == sqrt(2.)*I

    assert nsolve([x**2 + 2, y**2 + 2], [x, y], [I, I]) == Matrix([sqrt(2.)*I, sqrt(2.)*I])
    assert nsolve([x**2 + 2, y**2 + 2], [x, y], [I, I]) == Matrix([sqrt(2.)*I, sqrt(2.)*I])

def test_nsolve_dict_kwarg():
    x, y = symbols('x y')
    # one variable
    assert nsolve(x**2 - 2, 1, dict = True) == \
        [{x: sqrt(2.)}]
    # one variable with complex solution
    assert nsolve(x**2 + 2, I, dict = True) == \
        [{x: sqrt(2.)*I}]
    # two variables
    assert nsolve([x**2 + y**2 - 5, x**2 - y**2 + 1], [x, y], [1, 1], dict = True) == \
        [{x: sqrt(2.), y: sqrt(3.)}]

def test_nsolve_rational():
    x = symbols('x')
    assert nsolve(x - Rational(1, 3), 0, prec=100) == Rational(1, 3).evalf(100)


def test_issue_14950():
    x = Matrix(symbols('t s'))
    x0 = Matrix([17, 23])
    eqn = x + x0
    assert nsolve(eqn, x, x0) == nfloat(-x0)
    assert nsolve(eqn.T, x.T, x0.T) == nfloat(-x0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_element.py ===
"""Tests of classes in element.py.

The really big classes -- Tag, PageElement, and NavigableString --
are tested in separate files.
"""

import pytest
from bs4.element import (
    HTMLAttributeDict,
    XMLAttributeDict,
    CharsetMetaAttributeValue,
    ContentMetaAttributeValue,
    NamespacedAttribute,
    ResultSet,
)

class TestNamedspacedAttribute:
    def test_name_may_be_none_or_missing(self):
        a = NamespacedAttribute("xmlns", None)
        assert a == "xmlns"

        a = NamespacedAttribute("xmlns", "")
        assert a == "xmlns"

        a = NamespacedAttribute("xmlns")
        assert a == "xmlns"

    def test_namespace_may_be_none_or_missing(self):
        a = NamespacedAttribute(None, "tag")
        assert a == "tag"

        a = NamespacedAttribute("", "tag")
        assert a == "tag"

    def test_attribute_is_equivalent_to_colon_separated_string(self):
        a = NamespacedAttribute("a", "b")
        assert "a:b" == a

    def test_attributes_are_equivalent_if_prefix_and_name_identical(self):
        a = NamespacedAttribute("a", "b", "c")
        b = NamespacedAttribute("a", "b", "c")
        assert a == b

        # The actual namespace is not considered.
        c = NamespacedAttribute("a", "b", None)
        assert a == c

        # But name and prefix are important.
        d = NamespacedAttribute("a", "z", "c")
        assert a != d

        e = NamespacedAttribute("z", "b", "c")
        assert a != e


class TestAttributeValueWithCharsetSubstitution:
    """Certain attributes are designed to have the charset of the
    final document substituted into their value.
    """

    def test_charset_meta_attribute_value(self):
        # The value of a CharsetMetaAttributeValue is whatever
        # encoding the string is in.
        value = CharsetMetaAttributeValue("euc-jp")
        assert "euc-jp" == value
        assert "euc-jp" == value.original_value
        assert "utf8" == value.substitute_encoding("utf8")
        assert "ascii" == value.substitute_encoding("ascii")

        # If the target encoding is a Python internal encoding,
        # no encoding will be mentioned in the output HTML.
        assert "" == value.substitute_encoding("palmos")

    def test_content_meta_attribute_value(self):
        value = ContentMetaAttributeValue("text/html; charset=euc-jp")
        assert "text/html; charset=euc-jp" == value
        assert "text/html; charset=euc-jp" == value.original_value
        assert "text/html; charset=utf8" == value.substitute_encoding("utf8")
        assert "text/html; charset=ascii" == value.substitute_encoding("ascii")

        # If the target encoding is a Python internal encoding, the
        # charset argument will be omitted altogether.
        assert "text/html" == value.substitute_encoding("palmos")


class TestAttributeDicts:
    def test_xml_attribute_value_handling(self):
        # Verify that attribute values are processed according to the
        # XML spec's rules.
        d = XMLAttributeDict()
        d["v"] = 100
        assert d["v"] == "100"
        d["v"] = 100.123
        assert d["v"] == "100.123"

        # This preserves Beautiful Soup's old behavior in the absence of
        # guidance from the spec.
        d["v"] = False
        assert d["v"] is False

        d["v"] = True
        assert d["v"] is True

        d["v"] = None
        assert d["v"] == ""

    def test_html_attribute_value_handling(self):
        # Verify that attribute values are processed according to the
        # HTML spec's rules.
        d = HTMLAttributeDict()
        d["v"] = 100
        assert d["v"] == "100"
        d["v"] = 100.123
        assert d["v"] == "100.123"

        d["v"] = False
        assert "v" not in d

        d["v"] = None
        assert "v" not in d

        d["v"] = True
        assert d["v"] == "v"

        attribute = NamespacedAttribute("prefix", "name", "namespace")
        d[attribute] = True
        assert d[attribute] == "name"


class TestResultSet:
    def test_getattr_exception(self):
        rs = ResultSet(None)
        with pytest.raises(AttributeError) as e:
            rs.name
        assert (
            """ResultSet object has no attribute "name". You're probably treating a list of elements like a single element. Did you call find_all() when you meant to call find()?"""
            == str(e.value)
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\array_constructors.py ===
from typing import Any

import numpy as np
import numpy.typing as npt

class Index:
    def __index__(self) -> int:
        return 0


class SubClass(npt.NDArray[np.float64]):
    pass


def func(i: int, j: int, **kwargs: Any) -> SubClass:
    return B


i8 = np.int64(1)

A = np.array([1])
B = A.view(SubClass).copy()
B_stack = np.array([[1], [1]]).view(SubClass)
C = [1]

np.ndarray(Index())
np.ndarray([Index()])

np.array(1, dtype=float)
np.array(1, copy=None)
np.array(1, order='F')
np.array(1, order=None)
np.array(1, subok=True)
np.array(1, ndmin=3)
np.array(1, str, copy=True, order='C', subok=False, ndmin=2)

np.asarray(A)
np.asarray(B)
np.asarray(C)

np.asanyarray(A)
np.asanyarray(B)
np.asanyarray(B, dtype=int)
np.asanyarray(C)

np.ascontiguousarray(A)
np.ascontiguousarray(B)
np.ascontiguousarray(C)

np.asfortranarray(A)
np.asfortranarray(B)
np.asfortranarray(C)

np.require(A)
np.require(B)
np.require(B, dtype=int)
np.require(B, requirements=None)
np.require(B, requirements="E")
np.require(B, requirements=["ENSUREARRAY"])
np.require(B, requirements={"F", "E"})
np.require(B, requirements=["C", "OWNDATA"])
np.require(B, requirements="W")
np.require(B, requirements="A")
np.require(C)

np.linspace(0, 2)
np.linspace(0.5, [0, 1, 2])
np.linspace([0, 1, 2], 3)
np.linspace(0j, 2)
np.linspace(0, 2, num=10)
np.linspace(0, 2, endpoint=True)
np.linspace(0, 2, retstep=True)
np.linspace(0j, 2j, retstep=True)
np.linspace(0, 2, dtype=bool)
np.linspace([0, 1], [2, 3], axis=Index())

np.logspace(0, 2, base=2)
np.logspace(0, 2, base=2)
np.logspace(0, 2, base=[1j, 2j], num=2)

np.geomspace(1, 2)

np.zeros_like(A)
np.zeros_like(C)
np.zeros_like(B)
np.zeros_like(B, dtype=np.int64)

np.ones_like(A)
np.ones_like(C)
np.ones_like(B)
np.ones_like(B, dtype=np.int64)

np.empty_like(A)
np.empty_like(C)
np.empty_like(B)
np.empty_like(B, dtype=np.int64)

np.full_like(A, i8)
np.full_like(C, i8)
np.full_like(B, i8)
np.full_like(B, i8, dtype=np.int64)

np.ones(1)
np.ones([1, 1, 1])

np.full(1, i8)
np.full([1, 1, 1], i8)

np.indices([1, 2, 3])
np.indices([1, 2, 3], sparse=True)

np.fromfunction(func, (3, 5))

np.identity(10)

np.atleast_1d(C)
np.atleast_1d(A)
np.atleast_1d(C, C)
np.atleast_1d(C, A)
np.atleast_1d(A, A)

np.atleast_2d(C)

np.atleast_3d(C)

np.vstack([C, C])
np.vstack([C, A])
np.vstack([A, A])

np.hstack([C, C])

np.stack([C, C])
np.stack([C, C], axis=0)
np.stack([C, C], out=B_stack)

np.block([[C, C], [C, C]])
np.block(A)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\list\array.py ===
"""
Test extension array for storing nested data in a pandas container.

The ListArray stores an ndarray of lists.
"""
from __future__ import annotations

import numbers
import string
from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.base import ExtensionDtype

import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.arrays import ExtensionArray

if TYPE_CHECKING:
    from pandas._typing import type_t


class ListDtype(ExtensionDtype):
    type = list
    name = "list"
    na_value = np.nan

    @classmethod
    def construct_array_type(cls) -> type_t[ListArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return ListArray


class ListArray(ExtensionArray):
    dtype = ListDtype()
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False) -> None:
        if not isinstance(values, np.ndarray):
            raise TypeError("Need to pass a numpy array as values")
        for val in values:
            if not isinstance(val, self.dtype.type) and not pd.isna(val):
                raise TypeError("All values must be of type " + str(self.dtype.type))
        self.data = values

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        data = np.empty(len(scalars), dtype=object)
        data[:] = scalars
        return cls(data)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        else:
            # slice, list-like, mask
            return type(self)(self.data[item])

    def __len__(self) -> int:
        return len(self.data)

    def isna(self):
        return np.array(
            [not isinstance(x, list) and np.isnan(x) for x in self.data], dtype=bool
        )

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

        return self._from_sequence(output)

    def copy(self):
        return type(self)(self.data[:])

    def astype(self, dtype, copy=True):
        if isinstance(dtype, type(self.dtype)) and dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif is_string_dtype(dtype) and not is_object_dtype(dtype):
            # numpy has problems with astype(str) for nested elements
            return np.array([str(x) for x in self.data], dtype=dtype)
        elif not copy:
            return np.asarray(self.data, dtype=dtype)
        else:
            return np.array(self.data, dtype=dtype, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([x.data for x in to_concat])
        return cls(data)


def make_data():
    # TODO: Use a regular dict. See _NDFrameIndexer._setitem_with_indexer
    rng = np.random.default_rng(2)
    data = np.empty(100, dtype=object)
    data[:] = [
        [rng.choice(list(string.ascii_letters)) for _ in range(rng.integers(0, 10))]
        for _ in range(100)
    ]
    return data

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\ranges\test_indexing.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm


class TestGetIndexer:
    def test_get_indexer(self):
        index = RangeIndex(start=0, stop=20, step=2)
        target = RangeIndex(10)
        indexer = index.get_indexer(target)
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_pad(self):
        index = RangeIndex(start=0, stop=20, step=2)
        target = RangeIndex(10)
        indexer = index.get_indexer(target, method="pad")
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_backfill(self):
        index = RangeIndex(start=0, stop=20, step=2)
        target = RangeIndex(10)
        indexer = index.get_indexer(target, method="backfill")
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, expected)

    def test_get_indexer_limit(self):
        # GH#28631
        idx = RangeIndex(4)
        target = RangeIndex(6)
        result = idx.get_indexer(target, method="pad", limit=1)
        expected = np.array([0, 1, 2, 3, 3, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("stop", [0, -1, -2])
    def test_get_indexer_decreasing(self, stop):
        # GH#28678
        index = RangeIndex(7, stop, -3)
        result = index.get_indexer(range(9))
        expected = np.array([-1, 2, -1, -1, 1, -1, -1, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)


class TestTake:
    def test_take_preserve_name(self):
        index = RangeIndex(1, 5, name="foo")
        taken = index.take([3, 0, 1])
        assert index.name == taken.name

    def test_take_fill_value(self):
        # GH#12631
        idx = RangeIndex(1, 4, name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = Index([2, 1, 3], dtype=np.int64, name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index([2, 1, 3], dtype=np.int64, name="xxx")
        tm.assert_index_equal(result, expected)

        msg = "Unable to fill values because RangeIndex cannot contain NA"
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

    def test_take_raises_index_error(self):
        idx = RangeIndex(1, 4, name="xxx")

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

        msg = "index -4 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -4]))

        # no errors
        result = idx.take(np.array([1, -3]))
        expected = Index([2, 1], dtype=np.int64, name="xxx")
        tm.assert_index_equal(result, expected)

    def test_take_accepts_empty_array(self):
        idx = RangeIndex(1, 4, name="foo")
        result = idx.take(np.array([]))
        expected = Index([], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

        # empty index
        idx = RangeIndex(0, name="foo")
        result = idx.take(np.array([]))
        expected = Index([], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

    def test_take_accepts_non_int64_array(self):
        idx = RangeIndex(1, 4, name="foo")
        result = idx.take(np.array([2, 1], dtype=np.uint32))
        expected = Index([3, 2], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

    def test_take_when_index_has_step(self):
        idx = RangeIndex(1, 11, 3, name="foo")  # [1, 4, 7, 10]
        result = idx.take(np.array([1, 0, -1, -4]))
        expected = Index([4, 1, 10, 1], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)

    def test_take_when_index_has_negative_step(self):
        idx = RangeIndex(11, -4, -2, name="foo")  # [11, 9, 7, 5, 3, 1, -1, -3]
        result = idx.take(np.array([1, 0, -1, -8]))
        expected = Index([9, 11, -3, 11], dtype=np.int64, name="foo")
        tm.assert_index_equal(result, expected)


class TestWhere:
    def test_where_putmask_range_cast(self):
        # GH#43240
        idx = RangeIndex(0, 5, name="test")

        mask = np.array([True, True, False, False, False])
        result = idx.putmask(mask, 10)
        expected = Index([10, 10, 2, 3, 4], dtype=np.int64, name="test")
        tm.assert_index_equal(result, expected)

        result = idx.where(~mask, 10)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_trig.py ===
from mpmath import *
from mpmath.libmp import *

def test_trig_misc_hard():
    mp.prec = 53
    # Worst-case input for an IEEE double, from a paper by Kahan
    x = ldexp(6381956970095103,797)
    assert cos(x) == mpf('-4.6871659242546277e-19')
    assert sin(x) == 1

    mp.prec = 150
    a = mpf(10**50)
    mp.prec = 53
    assert sin(a).ae(-0.7896724934293100827)
    assert cos(a).ae(-0.6135286082336635622)

    # Check relative accuracy close to x = zero
    assert sin(1e-100) == 1e-100  # when rounding to nearest
    assert sin(1e-6).ae(9.999999999998333e-007, rel_eps=2e-15, abs_eps=0)
    assert sin(1e-6j).ae(1.0000000000001666e-006j, rel_eps=2e-15, abs_eps=0)
    assert sin(-1e-6j).ae(-1.0000000000001666e-006j, rel_eps=2e-15, abs_eps=0)
    assert cos(1e-100) == 1
    assert cos(1e-6).ae(0.9999999999995)
    assert cos(-1e-6j).ae(1.0000000000005)
    assert tan(1e-100) == 1e-100
    assert tan(1e-6).ae(1.0000000000003335e-006, rel_eps=2e-15, abs_eps=0)
    assert tan(1e-6j).ae(9.9999999999966644e-007j, rel_eps=2e-15, abs_eps=0)
    assert tan(-1e-6j).ae(-9.9999999999966644e-007j, rel_eps=2e-15, abs_eps=0)

def test_trig_near_zero():
    mp.dps = 15

    for r in [round_nearest, round_down, round_up, round_floor, round_ceiling]:
        assert sin(0, rounding=r) == 0
        assert cos(0, rounding=r) == 1

    a = mpf('1e-100')
    b = mpf('-1e-100')

    assert sin(a, rounding=round_nearest) == a
    assert sin(a, rounding=round_down) < a
    assert sin(a, rounding=round_floor) < a
    assert sin(a, rounding=round_up) >= a
    assert sin(a, rounding=round_ceiling) >= a
    assert sin(b, rounding=round_nearest) == b
    assert sin(b, rounding=round_down) > b
    assert sin(b, rounding=round_floor) <= b
    assert sin(b, rounding=round_up) <= b
    assert sin(b, rounding=round_ceiling) > b

    assert cos(a, rounding=round_nearest) == 1
    assert cos(a, rounding=round_down) < 1
    assert cos(a, rounding=round_floor) < 1
    assert cos(a, rounding=round_up) == 1
    assert cos(a, rounding=round_ceiling) == 1
    assert cos(b, rounding=round_nearest) == 1
    assert cos(b, rounding=round_down) < 1
    assert cos(b, rounding=round_floor) < 1
    assert cos(b, rounding=round_up) == 1
    assert cos(b, rounding=round_ceiling) == 1


def test_trig_near_n_pi():

    mp.dps = 15
    a = [n*pi for n in [1, 2, 6, 11, 100, 1001, 10000, 100001]]
    mp.dps = 135
    a.append(10**100 * pi)
    mp.dps = 15

    assert sin(a[0]) == mpf('1.2246467991473531772e-16')
    assert sin(a[1]) == mpf('-2.4492935982947063545e-16')
    assert sin(a[2]) == mpf('-7.3478807948841190634e-16')
    assert sin(a[3]) == mpf('4.8998251578625894243e-15')
    assert sin(a[4]) == mpf('1.9643867237284719452e-15')
    assert sin(a[5]) == mpf('-8.8632615209684813458e-15')
    assert sin(a[6]) == mpf('-4.8568235395684898392e-13')
    assert sin(a[7]) == mpf('3.9087342299491231029e-11')
    assert sin(a[8]) == mpf('-1.369235466754566993528e-36')

    r = round_nearest
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) == 1

    r = round_up
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) == 1

    r = round_down
    assert cos(a[0], rounding=r) > -1
    assert cos(a[1], rounding=r) < 1
    assert cos(a[2], rounding=r) < 1
    assert cos(a[3], rounding=r) > -1
    assert cos(a[4], rounding=r) < 1
    assert cos(a[5], rounding=r) > -1
    assert cos(a[6], rounding=r) < 1
    assert cos(a[7], rounding=r) > -1
    assert cos(a[8], rounding=r) < 1

    r = round_floor
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) < 1
    assert cos(a[2], rounding=r) < 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) < 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) < 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) < 1

    r = round_ceiling
    assert cos(a[0], rounding=r) > -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) > -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) > -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) > -1
    assert cos(a[8], rounding=r) == 1

    mp.dps = 15

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_immutable.py ===
from itertools import product

from sympy.core.relational import (Equality, Unequality)
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices import SparseMatrix
from sympy.matrices.immutable import \
    ImmutableDenseMatrix, ImmutableSparseMatrix
from sympy.abc import x, y
from sympy.testing.pytest import raises

IM = ImmutableDenseMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ISM = ImmutableSparseMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ieye = ImmutableDenseMatrix(eye(3))


def test_creation():
    assert IM.shape == ISM.shape == (3, 3)
    assert IM[1, 2] == ISM[1, 2] == 6
    assert IM[2, 2] == ISM[2, 2] == 9


def test_immutability():
    with raises(TypeError):
        IM[2, 2] = 5
    with raises(TypeError):
        ISM[2, 2] = 5


def test_slicing():
    assert IM[1, :] == ImmutableDenseMatrix([[4, 5, 6]])
    assert IM[:2, :2] == ImmutableDenseMatrix([[1, 2], [4, 5]])
    assert ISM[1, :] == ImmutableSparseMatrix([[4, 5, 6]])
    assert ISM[:2, :2] == ImmutableSparseMatrix([[1, 2], [4, 5]])


def test_subs():
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix([[1, 2], [x, 4]])
    C = ImmutableMatrix([[-x, x*y], [-(x + y), y**2]])
    assert B.subs(x, 3) == A
    assert (x*B).subs(x, 3) == 3*A
    assert (x*eye(2) + B).subs(x, 3) == 3*eye(2) + A
    assert C.subs([[x, -1], [y, -2]]) == A
    assert C.subs([(x, -1), (y, -2)]) == A
    assert C.subs({x: -1, y: -2}) == A
    assert C.subs({x: y - 1, y: x - 1}, simultaneous=True) == \
        ImmutableMatrix([[1 - y, (x - 1)*(y - 1)], [2 - x - y, (x - 1)**2]])


def test_as_immutable():
    data = [[1, 2], [3, 4]]
    X = Matrix(data)
    assert sympify(X) == X.as_immutable() == ImmutableMatrix(data)

    data = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
    X = SparseMatrix(2, 2, data)
    assert sympify(X) == X.as_immutable() == ImmutableSparseMatrix(2, 2, data)


def test_function_return_types():
    # Lets ensure that decompositions of immutable matrices remain immutable
    # I.e. do MatrixBase methods return the correct class?
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[1], [0]])
    q, r = X.QRdecomposition()
    assert (type(q), type(r)) == (ImmutableMatrix, ImmutableMatrix)

    assert type(X.LUsolve(Y)) == ImmutableMatrix
    assert type(X.QRsolve(Y)) == ImmutableMatrix

    X = ImmutableMatrix([[5, 2], [2, 7]])
    assert X.T == X
    assert X.is_symmetric
    assert type(X.cholesky()) == ImmutableMatrix
    L, D = X.LDLdecomposition()
    assert (type(L), type(D)) == (ImmutableMatrix, ImmutableMatrix)

    X = ImmutableMatrix([[1, 2], [2, 1]])
    assert X.is_diagonalizable()
    assert X.det() == -3
    assert X.norm(2) == 3

    assert type(X.eigenvects()[0][2][0]) == ImmutableMatrix

    assert type(zeros(3, 3).as_immutable().nullspace()[0]) == ImmutableMatrix

    X = ImmutableMatrix([[1, 0], [2, 1]])
    assert type(X.lower_triangular_solve(Y)) == ImmutableMatrix
    assert type(X.T.upper_triangular_solve(Y)) == ImmutableMatrix

    assert type(X.minor_submatrix(0, 0)) == ImmutableMatrix

# issue 6279
# https://github.com/sympy/sympy/issues/6279
# Test that Immutable _op_ Immutable => Immutable and not MatExpr


def test_immutable_evaluation():
    X = ImmutableMatrix(eye(3))
    A = ImmutableMatrix(3, 3, range(9))
    assert isinstance(X + A, ImmutableMatrix)
    assert isinstance(X * A, ImmutableMatrix)
    assert isinstance(X * 2, ImmutableMatrix)
    assert isinstance(2 * X, ImmutableMatrix)
    assert isinstance(A**2, ImmutableMatrix)


def test_deterimant():
    assert ImmutableMatrix(4, 4, lambda i, j: i + j).det() == 0


def test_Equality():
    assert Equality(IM, IM) is S.true
    assert Unequality(IM, IM) is S.false
    assert Equality(IM, IM.subs(1, 2)) is S.false
    assert Unequality(IM, IM.subs(1, 2)) is S.true
    assert Equality(IM, 2) is S.false
    assert Unequality(IM, 2) is S.true
    M = ImmutableMatrix([x, y])
    assert Equality(M, IM) is S.false
    assert Unequality(M, IM) is S.true
    assert Equality(M, M.subs(x, 2)).subs(x, 2) is S.true
    assert Unequality(M, M.subs(x, 2)).subs(x, 2) is S.false
    assert Equality(M, M.subs(x, 2)).subs(x, 3) is S.false
    assert Unequality(M, M.subs(x, 2)).subs(x, 3) is S.true


def test_integrate():
    intIM = integrate(IM, x)
    assert intIM.shape == IM.shape
    assert all(intIM[i, j] == (1 + j + 3*i)*x for i, j in
                product(range(3), range(3)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_matrixutils.py ===
from sympy.core.random import randint

from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, ones, zeros)

from sympy.physics.quantum.matrixutils import (
    to_sympy, to_numpy, to_scipy_sparse, matrix_tensor_product,
    matrix_to_zero, matrix_zeros, numpy_ndarray, scipy_sparse_matrix
)

from sympy.external import import_module
from sympy.testing.pytest import skip

m = Matrix([[1, 2], [3, 4]])


def test_sympy_to_sympy():
    assert to_sympy(m) == m


def test_matrix_to_zero():
    assert matrix_to_zero(m) == m
    assert matrix_to_zero(Matrix([[0, 0], [0, 0]])) == Integer(0)

np = import_module('numpy')


def test_to_numpy():
    if not np:
        skip("numpy not installed.")

    result = np.array([[1, 2], [3, 4]], dtype='complex')
    assert (to_numpy(m) == result).all()


def test_matrix_tensor_product():
    if not np:
        skip("numpy not installed.")

    l1 = zeros(4)
    for i in range(16):
        l1[i] = 2**i
    l2 = zeros(4)
    for i in range(16):
        l2[i] = i
    l3 = zeros(2)
    for i in range(4):
        l3[i] = i
    vec = Matrix([1, 2, 3])

    #test for Matrix known 4x4 matrices
    numpyl1 = np.array(l1.tolist())
    numpyl2 = np.array(l2.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, l2]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [l2, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()

    #test for other known matrix of different dimensions
    numpyl2 = np.array(l3.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, l3]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [l3, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()

    #test for non square matrix
    numpyl2 = np.array(vec.tolist())
    numpy_product = np.kron(numpyl1, numpyl2)
    args = [l1, vec]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()
    numpy_product = np.kron(numpyl2, numpyl1)
    args = [vec, l1]
    sympy_product = matrix_tensor_product(*args)
    assert numpy_product.tolist() == sympy_product.tolist()

    #test for random matrix with random values that are floats
    random_matrix1 = np.random.rand(randint(1, 5), randint(1, 5))
    random_matrix2 = np.random.rand(randint(1, 5), randint(1, 5))
    numpy_product = np.kron(random_matrix1, random_matrix2)
    args = [Matrix(random_matrix1.tolist()), Matrix(random_matrix2.tolist())]
    sympy_product = matrix_tensor_product(*args)
    assert not (sympy_product - Matrix(numpy_product.tolist())).tolist() > \
        (ones(sympy_product.rows, sympy_product.cols)*epsilon).tolist()

    #test for three matrix kronecker
    sympy_product = matrix_tensor_product(l1, vec, l2)

    numpy_product = np.kron(l1, np.kron(vec, l2))
    assert numpy_product.tolist() == sympy_product.tolist()


scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def test_to_scipy_sparse():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")
    else:
        sparse = scipy.sparse

    result = sparse.csr_matrix([[1, 2], [3, 4]], dtype='complex')
    assert np.linalg.norm((to_scipy_sparse(m) - result).todense()) == 0.0

epsilon = .000001


def test_matrix_zeros_sympy():
    sym = matrix_zeros(4, 4, format='sympy')
    assert isinstance(sym, Matrix)

def test_matrix_zeros_numpy():
    if not np:
        skip("numpy not installed.")

    num = matrix_zeros(4, 4, format='numpy')
    assert isinstance(num, numpy_ndarray)

def test_matrix_zeros_scipy():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")

    sci = matrix_zeros(4, 4, format='scipy.sparse')
    assert isinstance(sci, scipy_sparse_matrix)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_astype.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_astype():
    # with missing values
    arr = pd.array([0.1, 0.2, None], dtype="Float64")

    with pytest.raises(ValueError, match="cannot convert NA to integer"):
        arr.astype("int64")

    with pytest.raises(ValueError, match="cannot convert float NaN to bool"):
        arr.astype("bool")

    result = arr.astype("float64")
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    # no missing values
    arr = pd.array([0.0, 1.0, 0.5], dtype="Float64")
    result = arr.astype("int64")
    expected = np.array([0, 1, 0], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("bool")
    expected = np.array([False, True, True], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_to_floating_array():
    # astype to FloatingArray
    arr = pd.array([0.0, 1.0, None], dtype="Float64")

    result = arr.astype("Float64")
    tm.assert_extension_array_equal(result, arr)
    result = arr.astype(pd.Float64Dtype())
    tm.assert_extension_array_equal(result, arr)
    result = arr.astype("Float32")
    expected = pd.array([0.0, 1.0, None], dtype="Float32")
    tm.assert_extension_array_equal(result, expected)


def test_astype_to_boolean_array():
    # astype to BooleanArray
    arr = pd.array([0.0, 1.0, None], dtype="Float64")

    result = arr.astype("boolean")
    expected = pd.array([False, True, None], dtype="boolean")
    tm.assert_extension_array_equal(result, expected)
    result = arr.astype(pd.BooleanDtype())
    tm.assert_extension_array_equal(result, expected)


def test_astype_to_integer_array():
    # astype to IntegerArray
    arr = pd.array([0.0, 1.5, None], dtype="Float64")

    result = arr.astype("Int64")
    expected = pd.array([0, 1, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_astype_str(using_infer_string):
    a = pd.array([0.1, 0.2, None], dtype="Float64")

    if using_infer_string:
        expected = pd.array(["0.1", "0.2", None], dtype=pd.StringDtype(na_value=np.nan))

        tm.assert_extension_array_equal(a.astype(str), expected)
        tm.assert_extension_array_equal(a.astype("str"), expected)
    else:
        expected = np.array(["0.1", "0.2", "<NA>"], dtype="U32")

        tm.assert_numpy_array_equal(a.astype(str), expected)
        tm.assert_numpy_array_equal(a.astype("str"), expected)


def test_astype_copy():
    arr = pd.array([0.1, 0.2, None], dtype="Float64")
    orig = pd.array([0.1, 0.2, None], dtype="Float64")

    # copy=True -> ensure both data and mask are actual copies
    result = arr.astype("Float64", copy=True)
    assert result is not arr
    assert not tm.shares_memory(result, arr)
    result[0] = 10
    tm.assert_extension_array_equal(arr, orig)
    result[0] = pd.NA
    tm.assert_extension_array_equal(arr, orig)

    # copy=False
    result = arr.astype("Float64", copy=False)
    assert result is arr
    assert np.shares_memory(result._data, arr._data)
    assert np.shares_memory(result._mask, arr._mask)
    result[0] = 10
    assert arr[0] == 10
    result[0] = pd.NA
    assert arr[0] is pd.NA

    # astype to different dtype -> always needs a copy -> even with copy=False
    # we need to ensure that also the mask is actually copied
    arr = pd.array([0.1, 0.2, None], dtype="Float64")
    orig = pd.array([0.1, 0.2, None], dtype="Float64")

    result = arr.astype("Float32", copy=False)
    assert not tm.shares_memory(result, arr)
    result[0] = 10
    tm.assert_extension_array_equal(arr, orig)
    result[0] = pd.NA
    tm.assert_extension_array_equal(arr, orig)


def test_astype_object(dtype):
    arr = pd.array([1.0, pd.NA], dtype=dtype)

    result = arr.astype(object)
    expected = np.array([1.0, pd.NA], dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    # check exact element types
    assert isinstance(result[0], float)
    assert result[1] is pd.NA


def test_Float64_conversion():
    # GH#40729
    testseries = pd.Series(["1", "2", "3", "4"], dtype="object")
    result = testseries.astype(pd.Float64Dtype())

    expected = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=pd.Float64Dtype())

    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_groupby_subclass.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)


@pytest.mark.parametrize(
    "obj",
    [
        tm.SubclassedDataFrame({"A": np.arange(0, 10)}),
        tm.SubclassedSeries(np.arange(0, 10), name="A"),
    ],
)
def test_groupby_preserves_subclass(obj, groupby_func):
    # GH28330 -- preserve subclass through groupby operations

    if isinstance(obj, Series) and groupby_func in {"corrwith"}:
        pytest.skip(f"Not applicable for Series and {groupby_func}")

    grouped = obj.groupby(np.arange(0, 10))

    # Groups should preserve subclass type
    assert isinstance(grouped.get_group(0), type(obj))

    args = get_groupby_method_args(groupby_func, obj)

    warn = FutureWarning if groupby_func == "fillna" else None
    msg = f"{type(grouped).__name__}.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=msg, raise_on_extra_warnings=False):
        result1 = getattr(grouped, groupby_func)(*args)
    with tm.assert_produces_warning(warn, match=msg, raise_on_extra_warnings=False):
        result2 = grouped.agg(groupby_func, *args)

    # Reduction or transformation kernels should preserve type
    slices = {"ngroup", "cumcount", "size"}
    if isinstance(obj, DataFrame) and groupby_func in slices:
        assert isinstance(result1, tm.SubclassedSeries)
    else:
        assert isinstance(result1, type(obj))

    # Confirm .agg() groupby operations return same results
    if isinstance(result1, DataFrame):
        tm.assert_frame_equal(result1, result2)
    else:
        tm.assert_series_equal(result1, result2)


def test_groupby_preserves_metadata():
    # GH-37343
    custom_df = tm.SubclassedDataFrame({"a": [1, 2, 3], "b": [1, 1, 2], "c": [7, 8, 9]})
    assert "testattr" in custom_df._metadata
    custom_df.testattr = "hello"
    for _, group_df in custom_df.groupby("c"):
        assert group_df.testattr == "hello"

    # GH-45314
    def func(group):
        assert isinstance(group, tm.SubclassedDataFrame)
        assert hasattr(group, "testattr")
        assert group.testattr == "hello"
        return group.testattr

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(
        FutureWarning,
        match=msg,
        raise_on_extra_warnings=False,
        check_stacklevel=False,
    ):
        result = custom_df.groupby("c").apply(func)
    expected = tm.SubclassedSeries(["hello"] * 3, index=Index([7, 8, 9], name="c"))
    tm.assert_series_equal(result, expected)

    result = custom_df.groupby("c").apply(func, include_groups=False)
    tm.assert_series_equal(result, expected)

    # https://github.com/pandas-dev/pandas/pull/56761
    result = custom_df.groupby("c")[["a", "b"]].apply(func)
    tm.assert_series_equal(result, expected)

    def func2(group):
        assert isinstance(group, tm.SubclassedSeries)
        assert hasattr(group, "testattr")
        return group.testattr

    custom_series = tm.SubclassedSeries([1, 2, 3])
    custom_series.testattr = "hello"
    result = custom_series.groupby(custom_df["c"]).apply(func2)
    tm.assert_series_equal(result, expected)
    result = custom_series.groupby(custom_df["c"]).agg(func2)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("obj", [DataFrame, tm.SubclassedDataFrame])
def test_groupby_resample_preserves_subclass(obj):
    # GH28330 -- preserve subclass through groupby.resample()

    df = obj(
        {
            "Buyer": Series("Carl Carl Carl Carl Joe Carl".split(), dtype=object),
            "Quantity": [18, 3, 5, 1, 9, 3],
            "Date": [
                datetime(2013, 9, 1, 13, 0),
                datetime(2013, 9, 1, 13, 5),
                datetime(2013, 10, 1, 20, 0),
                datetime(2013, 10, 3, 10, 0),
                datetime(2013, 12, 2, 12, 0),
                datetime(2013, 9, 2, 14, 0),
            ],
        }
    )
    df = df.set_index("Date")

    # Confirm groupby.resample() preserves dataframe type
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(
        FutureWarning,
        match=msg,
        raise_on_extra_warnings=False,
        check_stacklevel=False,
    ):
        result = df.groupby("Buyer").resample("5D").sum()
    assert isinstance(result, obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_random_matrix.py ===
from sympy.concrete.products import Product
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.stats import (GaussianUnitaryEnsemble as GUE, density,
                         GaussianOrthogonalEnsemble as GOE,
                         GaussianSymplecticEnsemble as GSE,
                         joint_eigen_distribution,
                         CircularUnitaryEnsemble as CUE,
                         CircularOrthogonalEnsemble as COE,
                         CircularSymplecticEnsemble as CSE,
                         JointEigenDistribution,
                         level_spacing_distribution,
                         Normal, Beta)
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.random_matrix_models import GaussianEnsemble, RandomMatrixPSpace
from sympy.testing.pytest import raises

def test_GaussianEnsemble():
    G = GaussianEnsemble('G', 3)
    assert density(G) == G.pspace.model
    raises(ValueError, lambda: GaussianEnsemble('G', 3.5))

def test_GaussianUnitaryEnsemble():
    H = RandomMatrixSymbol('H', 3, 3)
    G = GUE('U', 3)
    assert density(G)(H) == sqrt(2)*exp(-3*Trace(H**2)/2)/(4*pi**Rational(9, 2))
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    l = IndexedBase('l')
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            27*sqrt(6)*exp(-3*(l[1]**2)/2 - 3*(l[2]**2)/2 - 3*(l[3]**2)/2)*
            Product(Abs(l[i] - l[j])**2, (j, i + 1, 3), (i, 1, 2))/(16*pi**Rational(3, 2))))
    s = Dummy('s')
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, 32*s**2*exp(-4*s**2/pi)/pi**2))


def test_GaussianOrthogonalEnsemble():
    H = RandomMatrixSymbol('H', 3, 3)
    _H = MatrixSymbol('_H', 3, 3)
    G = GOE('O', 3)
    assert density(G)(H) == exp(-3*Trace(H**2)/4)/Integral(exp(-3*Trace(_H**2)/4), _H)
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    l = IndexedBase('l')
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            9*sqrt(2)*exp(-3*l[1]**2/2 - 3*l[2]**2/2 - 3*l[3]**2/2)*
            Product(Abs(l[i] - l[j]), (j, i + 1, 3), (i, 1, 2))/(32*pi)))
    s = Dummy('s')
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, s*pi*exp(-s**2*pi/4)/2))

def test_GaussianSymplecticEnsemble():
    H = RandomMatrixSymbol('H', 3, 3)
    _H = MatrixSymbol('_H', 3, 3)
    G = GSE('O', 3)
    assert density(G)(H) == exp(-3*Trace(H**2))/Integral(exp(-3*Trace(_H**2)), _H)
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    l = IndexedBase('l')
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            162*sqrt(3)*exp(-3*l[1]**2/2 - 3*l[2]**2/2 - 3*l[3]**2/2)*
            Product(Abs(l[i] - l[j])**4, (j, i + 1, 3), (i, 1, 2))/(5*pi**Rational(3, 2))))
    s = Dummy('s')
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, S(262144)*s**4*exp(-64*s**2/(9*pi))/(729*pi**3)))

def test_CircularUnitaryEnsemble():
    CU = CUE('U', 3)
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    t = IndexedBase('t')
    assert joint_eigen_distribution(CU).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            Product(Abs(exp(I*t[j]) - exp(I*t[k]))**2,
            (j, k + 1, 3), (k, 1, 2))/(48*pi**3))
    )

def test_CircularOrthogonalEnsemble():
    CO = COE('U', 3)
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    t = IndexedBase('t')
    assert joint_eigen_distribution(CO).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            Product(Abs(exp(I*t[j]) - exp(I*t[k])),
            (j, k + 1, 3), (k, 1, 2))/(48*pi**2))
    )

def test_CircularSymplecticEnsemble():
    CS = CSE('U', 3)
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    t = IndexedBase('t')
    assert joint_eigen_distribution(CS).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            Product(Abs(exp(I*t[j]) - exp(I*t[k]))**4,
            (j, k + 1, 3), (k, 1, 2))/(720*pi**3))
    )

def test_JointEigenDistribution():
    A = Matrix([[Normal('A00', 0, 1), Normal('A01', 1, 1)],
                [Beta('A10', 1, 1), Beta('A11', 1, 1)]])
    assert JointEigenDistribution(A) == \
    JointDistributionHandmade(-sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)/2 +
    A[0, 0]/2 + A[1, 1]/2, sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)/2 + A[0, 0]/2 + A[1, 1]/2)
    raises(ValueError, lambda: JointEigenDistribution(Matrix([[1, 0], [2, 1]])))

def test_issue_19841():
    G1 = GUE('U', 2)
    G2 = G1.xreplace({2: 2})
    assert G1.args == G2.args

    X = MatrixSymbol('X', 2, 2)
    G = GSE('U', 2)
    h_pspace = RandomMatrixPSpace('P', model=density(G))
    H = RandomMatrixSymbol('H', 2, 2, pspace=h_pspace)
    H2 = RandomMatrixSymbol('H', 2, 2, pspace=None)
    assert H.doit() == H

    assert (2*H).xreplace({H: X}) == 2*X
    assert (2*H).xreplace({H2: X}) == 2*H
    assert (2*H2).xreplace({H: X}) == 2*H2
    assert (2*H2).xreplace({H2: X}) == 2*X

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_arithmetic.py ===
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.fixture
def data():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.fixture
def left_array():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")


@pytest.fixture
def right_array():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True, False, None] * 3, dtype="boolean")


# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opname, exp",
    [
        ("add", [True, True, None, True, False, None, None, None, None]),
        ("mul", [True, False, None, False, False, None, None, None, None]),
    ],
    ids=["add", "mul"],
)
def test_add_mul(left_array, right_array, opname, exp):
    op = getattr(operator, opname)
    result = op(left_array, right_array)
    expected = pd.array(exp, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_sub(left_array, right_array):
    msg = (
        r"numpy boolean subtract, the `-` operator, is (?:deprecated|not supported), "
        r"use the bitwise_xor, the `\^` operator, or the logical_xor function instead\."
    )
    with pytest.raises(TypeError, match=msg):
        left_array - right_array


def test_div(left_array, right_array):
    msg = "operator '.*' not implemented for bool dtypes"
    with pytest.raises(NotImplementedError, match=msg):
        # check that we are matching the non-masked Series behavior
        pd.Series(left_array._data) / pd.Series(right_array._data)

    with pytest.raises(NotImplementedError, match=msg):
        left_array / right_array


@pytest.mark.parametrize(
    "opname",
    [
        "floordiv",
        "mod",
        "pow",
    ],
)
def test_op_int8(left_array, right_array, opname):
    op = getattr(operator, opname)
    if opname != "mod":
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            result = op(left_array, right_array)
        return
    result = op(left_array, right_array)
    expected = op(left_array.astype("Int8"), right_array.astype("Int8"))
    tm.assert_extension_array_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_values(data, all_arithmetic_operators):
    # invalid ops
    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)

    # invalid scalars
    msg = (
        "did not contain a loop with signature matching types|"
        "BooleanArray cannot perform the operation|"
        "not supported for the input types, and the inputs could not be safely coerced "
        "to any supported types according to the casting rule ''safe''|"
        "not supported for dtype"
    )
    with pytest.raises(TypeError, match=msg):
        ops("foo")
    msg = "|".join(
        [
            r"unsupported operand type\(s\) for",
            "Concatenation operation is not implemented for NumPy arrays",
            "has no kernel",
            "not supported for dtype",
        ]
    )
    with pytest.raises(TypeError, match=msg):
        ops(pd.Timestamp("20180101"))

    # invalid array-likes
    if op not in ("__mul__", "__rmul__"):
        # TODO(extension) numpy's mul with object array sees booleans as numbers
        msg = "|".join(
            [
                r"unsupported operand type\(s\) for",
                "can only concatenate str",
                "not all arguments converted during string formatting",
                "has no kernel",
                "not implemented",
                "not supported for dtype",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            ops(pd.Series("foo", index=s.index))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_iterator.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import pytest

from pandas import (
    DataFrame,
    concat,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


def test_iterator(all_parsers):
    # see gh-6607
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0}

    expected = parser.read_csv(StringIO(data), **kwargs)

    if parser.engine == "pyarrow":
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), iterator=True, **kwargs)
        return

    with parser.read_csv(StringIO(data), iterator=True, **kwargs) as reader:
        first_chunk = reader.read(3)
        tm.assert_frame_equal(first_chunk, expected[:3])

        last_chunk = reader.read(5)
    tm.assert_frame_equal(last_chunk, expected[3:])


def test_iterator2(all_parsers):
    parser = all_parsers
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""

    if parser.engine == "pyarrow":
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), iterator=True)
        return

    with parser.read_csv(StringIO(data), iterator=True) as reader:
        result = list(reader)

    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(result[0], expected)


def test_iterator_stop_on_chunksize(all_parsers):
    # gh-3967: stopping iteration when chunksize is specified
    parser = all_parsers
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), chunksize=1)
        return

    with parser.read_csv(StringIO(data), chunksize=1) as reader:
        result = list(reader)

    assert len(result) == 3
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(concat(result), expected)


@pytest.mark.parametrize(
    "kwargs", [{"iterator": True, "chunksize": 1}, {"iterator": True}, {"chunksize": 1}]
)
def test_iterator_skipfooter_errors(all_parsers, kwargs):
    msg = "'skipfooter' not supported for iteration"
    parser = all_parsers
    data = "a\n1\n2"

    if parser.engine == "pyarrow":
        msg = (
            "The '(chunksize|iterator)' option is not supported with the "
            "'pyarrow' engine"
        )

    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), skipfooter=1, **kwargs) as _:
            pass


def test_iteration_open_handle(all_parsers):
    parser = all_parsers
    kwargs = {"header": None}

    with tm.ensure_clean() as path:
        with open(path, "w", encoding="utf-8") as f:
            f.write("AAA\nBBB\nCCC\nDDD\nEEE\nFFF\nGGG")

        with open(path, encoding="utf-8") as f:
            for line in f:
                if "CCC" in line:
                    break

            result = parser.read_csv(f, **kwargs)
            expected = DataFrame({0: ["DDD", "EEE", "FFF", "GGG"]})
            tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_bbp_pi.py ===
from sympy.core.random import randint

from sympy.ntheory.bbp_pi import pi_hex_digits
from sympy.testing.pytest import raises


# http://www.herongyang.com/Cryptography/Blowfish-First-8366-Hex-Digits-of-PI.html
# There are actually 8336 listed there; with the prepended 3 there are 8337
# below
dig=''.join('''
3243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d013
77be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b54709179216d5d98979fb1bd1310ba698dfb5
ac2ffd72dbd01adfb7b8e1afed6a267e96ba7c9045f12c7f9924a19947b3916cf70801f2e2858efc
16636920d871574e69a458fea3f4933d7e0d95748f728eb658718bcd5882154aee7b54a41dc25a59
b59c30d5392af26013c5d1b023286085f0ca417918b8db38ef8e79dcb0603a180e6c9e0e8bb01e8a
3ed71577c1bd314b2778af2fda55605c60e65525f3aa55ab945748986263e8144055ca396a2aab10
b6b4cc5c341141e8cea15486af7c72e993b3ee1411636fbc2a2ba9c55d741831f6ce5c3e169b8793
1eafd6ba336c24cf5c7a325381289586773b8f48986b4bb9afc4bfe81b6628219361d809ccfb21a9
91487cac605dec8032ef845d5de98575b1dc262302eb651b8823893e81d396acc50f6d6ff383f442
392e0b4482a484200469c8f04a9e1f9b5e21c66842f6e96c9a670c9c61abd388f06a51a0d2d8542f
68960fa728ab5133a36eef0b6c137a3be4ba3bf0507efb2a98a1f1651d39af017666ca593e82430e
888cee8619456f9fb47d84a5c33b8b5ebee06f75d885c12073401a449f56c16aa64ed3aa62363f77
061bfedf72429b023d37d0d724d00a1248db0fead349f1c09b075372c980991b7b25d479d8f6e8de
f7e3fe501ab6794c3b976ce0bd04c006bac1a94fb6409f60c45e5c9ec2196a246368fb6faf3e6c53
b51339b2eb3b52ec6f6dfc511f9b30952ccc814544af5ebd09bee3d004de334afd660f2807192e4b
b3c0cba85745c8740fd20b5f39b9d3fbdb5579c0bd1a60320ad6a100c6402c7279679f25fefb1fa3
cc8ea5e9f8db3222f83c7516dffd616b152f501ec8ad0552ab323db5fafd23876053317b483e00df
829e5c57bbca6f8ca01a87562edf1769dbd542a8f6287effc3ac6732c68c4f5573695b27b0bbca58
c8e1ffa35db8f011a010fa3d98fd2183b84afcb56c2dd1d35b9a53e479b6f84565d28e49bc4bfb97
90e1ddf2daa4cb7e3362fb1341cee4c6e8ef20cada36774c01d07e9efe2bf11fb495dbda4dae9091
98eaad8e716b93d5a0d08ed1d0afc725e08e3c5b2f8e7594b78ff6e2fbf2122b648888b812900df0
1c4fad5ea0688fc31cd1cff191b3a8c1ad2f2f2218be0e1777ea752dfe8b021fa1e5a0cc0fb56f74
e818acf3d6ce89e299b4a84fe0fd13e0b77cc43b81d2ada8d9165fa2668095770593cc7314211a14
77e6ad206577b5fa86c75442f5fb9d35cfebcdaf0c7b3e89a0d6411bd3ae1e7e4900250e2d2071b3
5e226800bb57b8e0af2464369bf009b91e5563911d59dfa6aa78c14389d95a537f207d5ba202e5b9
c5832603766295cfa911c819684e734a41b3472dca7b14a94a1b5100529a532915d60f573fbc9bc6
e42b60a47681e6740008ba6fb5571be91ff296ec6b2a0dd915b6636521e7b9f9b6ff34052ec58556
6453b02d5da99f8fa108ba47996e85076a4b7a70e9b5b32944db75092ec4192623ad6ea6b049a7df
7d9cee60b88fedb266ecaa8c71699a17ff5664526cc2b19ee1193602a575094c29a0591340e4183a
3e3f54989a5b429d656b8fe4d699f73fd6a1d29c07efe830f54d2d38e6f0255dc14cdd20868470eb
266382e9c6021ecc5e09686b3f3ebaefc93c9718146b6a70a1687f358452a0e286b79c5305aa5007
373e07841c7fdeae5c8e7d44ec5716f2b8b03ada37f0500c0df01c1f040200b3ffae0cf51a3cb574
b225837a58dc0921bdd19113f97ca92ff69432477322f547013ae5e58137c2dadcc8b576349af3dd
a7a94461460fd0030eecc8c73ea4751e41e238cd993bea0e2f3280bba1183eb3314e548b384f6db9
086f420d03f60a04bf2cb8129024977c795679b072bcaf89afde9a771fd9930810b38bae12dccf3f
2e5512721f2e6b7124501adde69f84cd877a5847187408da17bc9f9abce94b7d8cec7aec3adb851d
fa63094366c464c3d2ef1c18473215d908dd433b3724c2ba1612a14d432a65c45150940002133ae4
dd71dff89e10314e5581ac77d65f11199b043556f1d7a3c76b3c11183b5924a509f28fe6ed97f1fb
fa9ebabf2c1e153c6e86e34570eae96fb1860e5e0a5a3e2ab3771fe71c4e3d06fa2965dcb999e71d
0f803e89d65266c8252e4cc9789c10b36ac6150eba94e2ea78a5fc3c531e0a2df4f2f74ea7361d2b
3d1939260f19c279605223a708f71312b6ebadfe6eeac31f66e3bc4595a67bc883b17f37d1018cff
28c332ddefbe6c5aa56558218568ab9802eecea50fdb2f953b2aef7dad5b6e2f841521b628290761
70ecdd4775619f151013cca830eb61bd960334fe1eaa0363cfb5735c904c70a239d59e9e0bcbaade
14eecc86bc60622ca79cab5cabb2f3846e648b1eaf19bdf0caa02369b9655abb5040685a323c2ab4
b3319ee9d5c021b8f79b540b19875fa09995f7997e623d7da8f837889a97e32d7711ed935f166812
810e358829c7e61fd696dedfa17858ba9957f584a51b2272639b83c3ff1ac24696cdb30aeb532e30
548fd948e46dbc312858ebf2ef34c6ffeafe28ed61ee7c3c735d4a14d9e864b7e342105d14203e13
e045eee2b6a3aaabeadb6c4f15facb4fd0c742f442ef6abbb5654f3b1d41cd2105d81e799e86854d
c7e44b476a3d816250cf62a1f25b8d2646fc8883a0c1c7b6a37f1524c369cb749247848a0b5692b2
85095bbf00ad19489d1462b17423820e0058428d2a0c55f5ea1dadf43e233f70613372f0928d937e
41d65fecf16c223bdb7cde3759cbee74604085f2a7ce77326ea607808419f8509ee8efd85561d997
35a969a7aac50c06c25a04abfc800bcadc9e447a2ec3453484fdd567050e1e9ec9db73dbd3105588
cd675fda79e3674340c5c43465713e38d83d28f89ef16dff20153e21e78fb03d4ae6e39f2bdb83ad
f7e93d5a68948140f7f64c261c94692934411520f77602d4f7bcf46b2ed4a20068d40824713320f4
6a43b7d4b7500061af1e39f62e9724454614214f74bf8b88404d95fc1d96b591af70f4ddd366a02f
45bfbc09ec03bd97857fac6dd031cb850496eb27b355fd3941da2547e6abca0a9a28507825530429
f40a2c86dae9b66dfb68dc1462d7486900680ec0a427a18dee4f3ffea2e887ad8cb58ce0067af4d6
b6aace1e7cd3375fecce78a399406b2a4220fe9e35d9f385b9ee39d7ab3b124e8b1dc9faf74b6d18
5626a36631eae397b23a6efa74dd5b43326841e7f7ca7820fbfb0af54ed8feb397454056acba4895
2755533a3a20838d87fe6ba9b7d096954b55a867bca1159a58cca9296399e1db33a62a4a563f3125
f95ef47e1c9029317cfdf8e80204272f7080bb155c05282ce395c11548e4c66d2248c1133fc70f86
dc07f9c9ee41041f0f404779a45d886e17325f51ebd59bc0d1f2bcc18f41113564257b7834602a9c
60dff8e8a31f636c1b0e12b4c202e1329eaf664fd1cad181156b2395e0333e92e13b240b62eebeb9
2285b2a20ee6ba0d99de720c8c2da2f728d012784595b794fd647d0862e7ccf5f05449a36f877d48
fac39dfd27f33e8d1e0a476341992eff743a6f6eabf4f8fd37a812dc60a1ebddf8991be14cdb6e6b
0dc67b55106d672c372765d43bdcd0e804f1290dc7cc00ffa3b5390f92690fed0b667b9ffbcedb7d
9ca091cf0bd9155ea3bb132f88515bad247b9479bf763bd6eb37392eb3cc1159798026e297f42e31
2d6842ada7c66a2b3b12754ccc782ef11c6a124237b79251e706a1bbe64bfb63501a6b101811caed
fa3d25bdd8e2e1c3c9444216590a121386d90cec6ed5abea2a64af674eda86a85fbebfe98864e4c3
fe9dbc8057f0f7c08660787bf86003604dd1fd8346f6381fb07745ae04d736fccc83426b33f01eab
71b08041873c005e5f77a057bebde8ae2455464299bf582e614e58f48ff2ddfda2f474ef388789bd
c25366f9c3c8b38e74b475f25546fcd9b97aeb26618b1ddf84846a0e79915f95e2466e598e20b457
708cd55591c902de4cb90bace1bb8205d011a862487574a99eb77f19b6e0a9dc09662d09a1c43246
33e85a1f0209f0be8c4a99a0251d6efe101ab93d1d0ba5a4dfa186f20f2868f169dcb7da83573906
fea1e2ce9b4fcd7f5250115e01a70683faa002b5c40de6d0279af88c27773f8641c3604c0661a806
b5f0177a28c0f586e0006058aa30dc7d6211e69ed72338ea6353c2dd94c2c21634bbcbee5690bcb6
deebfc7da1ce591d766f05e4094b7c018839720a3d7c927c2486e3725f724d9db91ac15bb4d39eb8
fced54557808fca5b5d83d7cd34dad0fc41e50ef5eb161e6f8a28514d96c51133c6fd5c7e756e14e
c4362abfceddc6c837d79a323492638212670efa8e406000e03a39ce37d3faf5cfabc277375ac52d
1b5cb0679e4fa33742d382274099bc9bbed5118e9dbf0f7315d62d1c7ec700c47bb78c1b6b21a190
45b26eb1be6a366eb45748ab2fbc946e79c6a376d26549c2c8530ff8ee468dde7dd5730a1d4cd04d
c62939bbdba9ba4650ac9526e8be5ee304a1fad5f06a2d519a63ef8ce29a86ee22c089c2b843242e
f6a51e03aa9cf2d0a483c061ba9be96a4d8fe51550ba645bd62826a2f9a73a3ae14ba99586ef5562
e9c72fefd3f752f7da3f046f6977fa0a5980e4a91587b086019b09e6ad3b3ee593e990fd5a9e34d7
972cf0b7d9022b8b5196d5ac3a017da67dd1cf3ed67c7d2d281f9f25cfadf2b89b5ad6b4725a88f5
4ce029ac71e019a5e647b0acfded93fa9be8d3c48d283b57ccf8d5662979132e28785f0191ed7560
55f7960e44e3d35e8c15056dd488f46dba03a161250564f0bdc3eb9e153c9057a297271aeca93a07
2a1b3f6d9b1e6321f5f59c66fb26dcf3197533d928b155fdf5035634828aba3cbb28517711c20ad9
f8abcc5167ccad925f4de817513830dc8e379d58629320f991ea7a90c2fb3e7bce5121ce64774fbe
32a8b6e37ec3293d4648de53696413e680a2ae0810dd6db22469852dfd09072166b39a460a6445c0
dd586cdecf1c20c8ae5bbef7dd1b588d40ccd2017f6bb4e3bbdda26a7e3a59ff453e350a44bcb4cd
d572eacea8fa6484bb8d6612aebf3c6f47d29be463542f5d9eaec2771bf64e6370740e0d8de75b13
57f8721671af537d5d4040cb084eb4e2cc34d2466a0115af84e1b0042895983a1d06b89fb4ce6ea0
486f3f3b823520ab82011a1d4b277227f8611560b1e7933fdcbb3a792b344525bda08839e151ce79
4b2f32c9b7a01fbac9e01cc87ebcc7d1f6cf0111c3a1e8aac71a908749d44fbd9ad0dadecbd50ada
380339c32ac69136678df9317ce0b12b4ff79e59b743f5bb3af2d519ff27d9459cbf97222c15e6fc
2a0f91fc719b941525fae59361ceb69cebc2a8645912baa8d1b6c1075ee3056a0c10d25065cb03a4
42e0ec6e0e1698db3b4c98a0be3278e9649f1f9532e0d392dfd3a0342b8971f21e1b0a74414ba334
8cc5be7120c37632d8df359f8d9b992f2ee60b6f470fe3f11de54cda541edad891ce6279cfcd3e7e
6f1618b166fd2c1d05848fd2c5f6fb2299f523f357a632762393a8353156cccd02acf081625a75eb
b56e16369788d273ccde96629281b949d04c50901b71c65614e6c6c7bd327a140a45e1d006c3f27b
9ac9aa53fd62a80f00bb25bfe235bdd2f671126905b2040222b6cbcf7ccd769c2b53113ec01640e3
d338abbd602547adf0ba38209cf746ce7677afa1c52075606085cbfe4e8ae88dd87aaaf9b04cf9aa
7e1948c25c02fb8a8c01c36ae4d6ebe1f990d4f869a65cdea03f09252dc208e69fb74e6132ce77e2
5b578fdfe33ac372e6'''.split())


def test_hex_pi_nth_digits():
    assert pi_hex_digits(0) == '3243f6a8885a30'
    assert pi_hex_digits(1) ==  '243f6a8885a308'
    assert pi_hex_digits(10000) == '68ac8fcfb8016c'
    assert pi_hex_digits(13) == '08d313198a2e03'
    assert pi_hex_digits(0, 3) == '324'
    assert pi_hex_digits(0, 0) == ''
    raises(ValueError, lambda: pi_hex_digits(-1))
    raises(ValueError, lambda: pi_hex_digits(0, -1))
    raises(ValueError, lambda: pi_hex_digits(3.14))

    # this will pick a random segment to compute every time
    # it is run. If it ever fails, there is an error in the
    # computation.
    n = randint(0, len(dig))
    prec = randint(0, len(dig) - n)
    assert pi_hex_digits(n, prec) == dig[n: n + prec]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_dot.py ===
from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
        dotedges, dotprint)
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x


def test_purestr():
    assert purestr(Symbol('x')) == "Symbol('x')"
    assert purestr(Basic(S(1), S(2))) == "Basic(Integer(1), Integer(2))"
    assert purestr(Float(2)) == "Float('2.0', precision=53)"

    assert purestr(Symbol('x'), with_args=True) == ("Symbol('x')", ())
    assert purestr(Basic(S(1), S(2)), with_args=True) == \
            ('Basic(Integer(1), Integer(2))', ('Integer(1)', 'Integer(2)'))
    assert purestr(Float(2), with_args=True) == \
        ("Float('2.0', precision=53)", ())


def test_styleof():
    styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
              (Expr,  {'color': 'black'})]
    assert styleof(Basic(S(1)), styles) == {'color': 'blue', 'shape': 'ellipse'}

    assert styleof(x + 1, styles) == {'color': 'black', 'shape': 'ellipse'}


def test_attrprint():
    assert attrprint({'color': 'blue', 'shape': 'ellipse'}) == \
           '"color"="blue", "shape"="ellipse"'

def test_dotnode():

    assert dotnode(x, repeat=False) == \
        '"Symbol(\'x\')" ["color"="black", "label"="x", "shape"="ellipse"];'
    assert dotnode(x+2, repeat=False) == \
        '"Add(Integer(2), Symbol(\'x\'))" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];', \
        dotnode(x+2,repeat=0)

    assert dotnode(x + x**2, repeat=False) == \
        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];'
    assert dotnode(x + x**2, repeat=True) == \
        '"Add(Symbol(\'x\'), Pow(Symbol(\'x\'), Integer(2)))_()" ' \
        '["color"="black", "label"="Add", "shape"="ellipse"];'

def test_dotedges():
    assert sorted(dotedges(x+2, repeat=False)) == [
        '"Add(Integer(2), Symbol(\'x\'))" -> "Integer(2)";',
        '"Add(Integer(2), Symbol(\'x\'))" -> "Symbol(\'x\')";'
    ]
    assert sorted(dotedges(x + 2, repeat=True)) == [
        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Integer(2)_(0,)";',
        '"Add(Integer(2), Symbol(\'x\'))_()" -> "Symbol(\'x\')_(1,)";'
    ]

def test_dotprint():
    text = dotprint(x+2, repeat=False)
    assert all(e in text for e in dotedges(x+2, repeat=False))
    assert all(
        n in text for n in [dotnode(expr, repeat=False)
        for expr in (x, Integer(2), x+2)])
    assert 'digraph' in text

    text = dotprint(x+x**2, repeat=False)
    assert all(e in text for e in dotedges(x+x**2, repeat=False))
    assert all(
        n in text for n in [dotnode(expr, repeat=False)
        for expr in (x, Integer(2), x**2)])
    assert 'digraph' in text

    text = dotprint(x+x**2, repeat=True)
    assert all(e in text for e in dotedges(x+x**2, repeat=True))
    assert all(
        n in text for n in [dotnode(expr, pos=())
        for expr in [x + x**2]])

    text = dotprint(x**x, repeat=True)
    assert all(e in text for e in dotedges(x**x, repeat=True))
    assert all(
        n in text for n in [dotnode(x, pos=(0,)), dotnode(x, pos=(1,))])
    assert 'digraph' in text

def test_dotprint_depth():
    text = dotprint(3*x+2, depth=1)
    assert dotnode(3*x+2) in text
    assert dotnode(x) not in text
    text = dotprint(3*x+2)
    assert "depth" not in text

def test_Matrix_and_non_basics():
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    n = Symbol('n')
    assert dotprint(MatrixSymbol('X', n, n)) == \
"""digraph{

# Graph style
"ordering"="out"
"rankdir"="TD"

#########
# Nodes #
#########

"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" ["color"="black", "label"="MatrixSymbol", "shape"="ellipse"];
"Str('X')_(0,)" ["color"="blue", "label"="X", "shape"="ellipse"];
"Symbol('n')_(1,)" ["color"="black", "label"="n", "shape"="ellipse"];
"Symbol('n')_(2,)" ["color"="black", "label"="n", "shape"="ellipse"];

#########
# Edges #
#########

"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Str('X')_(0,)";
"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(1,)";
"MatrixSymbol(Str('X'), Symbol('n'), Symbol('n'))_()" -> "Symbol('n')_(2,)";
}"""


def test_labelfunc():
    text = dotprint(x + 2, labelfunc=srepr)
    assert "Symbol('x')" in text
    assert "Integer(2)" in text


def test_commutative():
    x, y = symbols('x y', commutative=False)
    assert dotprint(x + y) == dotprint(y + x)
    assert dotprint(x*y) != dotprint(y*x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_dyadic.py ===
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector import (CoordSys3D, Vector, Dyadic,
                          DyadicAdd, DyadicMul, DyadicZero,
                          BaseDyadic, express)


A = CoordSys3D('A')


def test_dyadic():
    a, b = symbols('a, b')
    assert Dyadic.zero != 0
    assert isinstance(Dyadic.zero, DyadicZero)
    assert BaseDyadic(A.i, A.j) != BaseDyadic(A.j, A.i)
    assert (BaseDyadic(Vector.zero, A.i) ==
            BaseDyadic(A.i, Vector.zero) == Dyadic.zero)

    d1 = A.i | A.i
    d2 = A.j | A.j
    d3 = A.i | A.j

    assert isinstance(d1, BaseDyadic)
    d_mul = a*d1
    assert isinstance(d_mul, DyadicMul)
    assert d_mul.base_dyadic == d1
    assert d_mul.measure_number == a
    assert isinstance(a*d1 + b*d3, DyadicAdd)
    assert d1 == A.i.outer(A.i)
    assert d3 == A.i.outer(A.j)
    v1 = a*A.i - A.k
    v2 = A.i + b*A.j
    assert v1 | v2 == v1.outer(v2) == a * (A.i|A.i) + (a*b) * (A.i|A.j) +\
           - (A.k|A.i) - b * (A.k|A.j)
    assert d1 * 0 == Dyadic.zero
    assert d1 != Dyadic.zero
    assert d1 * 2 == 2 * (A.i | A.i)
    assert d1 / 2. == 0.5 * d1

    assert d1.dot(0 * d1) == Vector.zero
    assert d1 & d2 == Dyadic.zero
    assert d1.dot(A.i) == A.i == d1 & A.i

    assert d1.cross(Vector.zero) == Dyadic.zero
    assert d1.cross(A.i) == Dyadic.zero
    assert d1 ^ A.j == d1.cross(A.j)
    assert d1.cross(A.k) == - A.i | A.j
    assert d2.cross(A.i) == - A.j | A.k == d2 ^ A.i

    assert A.i ^ d1 == Dyadic.zero
    assert A.j.cross(d1) == - A.k | A.i == A.j ^ d1
    assert Vector.zero.cross(d1) == Dyadic.zero
    assert A.k ^ d1 == A.j | A.i
    assert A.i.dot(d1) == A.i & d1 == A.i
    assert A.j.dot(d1) == Vector.zero
    assert Vector.zero.dot(d1) == Vector.zero
    assert A.j & d2 == A.j

    assert d1.dot(d3) == d1 & d3 == A.i | A.j == d3
    assert d3 & d1 == Dyadic.zero

    q = symbols('q')
    B = A.orient_new_axis('B', q, A.k)
    assert express(d1, B) == express(d1, B, B)

    expr1 = ((cos(q)**2) * (B.i | B.i) + (-sin(q) * cos(q)) *
            (B.i | B.j) + (-sin(q) * cos(q)) * (B.j | B.i) + (sin(q)**2) *
            (B.j | B.j))
    assert (express(d1, B) - expr1).simplify() == Dyadic.zero

    expr2 = (cos(q)) * (B.i | A.i) + (-sin(q)) * (B.j | A.i)
    assert (express(d1, B, A) - expr2).simplify() == Dyadic.zero

    expr3 = (cos(q)) * (A.i | B.i) + (-sin(q)) * (A.i | B.j)
    assert (express(d1, A, B) - expr3).simplify() == Dyadic.zero

    assert d1.to_matrix(A) == Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert d1.to_matrix(A, B) == Matrix([[cos(q), -sin(q), 0],
                                         [0, 0, 0],
                                         [0, 0, 0]])
    assert d3.to_matrix(A) == Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    v1 = a * A.i + b * A.j + c * A.k
    v2 = d * A.i + e * A.j + f * A.k
    d4 = v1.outer(v2)
    assert d4.to_matrix(A) == Matrix([[a * d, a * e, a * f],
                                      [b * d, b * e, b * f],
                                      [c * d, c * e, c * f]])
    d5 = v1.outer(v1)
    C = A.orient_new_axis('C', q, A.i)
    for expected, actual in zip(C.rotation_matrix(A) * d5.to_matrix(A) * \
                               C.rotation_matrix(A).T, d5.to_matrix(C)):
        assert (expected - actual).simplify() == 0


def test_dyadic_simplify():
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    N = CoordSys3D('N')

    dy = N.i | N.i
    test1 = (1 / x + 1 / y) * dy
    assert (N.i & test1 & N.i) != (x + y) / (x * y)
    test1 = test1.simplify()
    assert test1.simplify() == simplify(test1)
    assert (N.i & test1 & N.i) == (x + y) / (x * y)

    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * dy
    test2 = test2.simplify()
    assert (N.i & test2 & N.i) == (A**2 * s**4 / (4 * pi * k * m**3))

    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * dy
    test3 = test3.simplify()
    assert (N.i & test3 & N.i) == 0

    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * dy
    test4 = test4.simplify()
    assert (N.i & test4 & N.i) == -2 * y


def test_dyadic_srepr():
    from sympy.printing.repr import srepr
    N = CoordSys3D('N')

    dy = N.i | N.j
    res = "BaseDyadic(CoordSys3D(Str('N'), Tuple(ImmutableDenseMatrix([["\
        "Integer(1), Integer(0), Integer(0)], [Integer(0), Integer(1), "\
        "Integer(0)], [Integer(0), Integer(0), Integer(1)]]), "\
        "VectorZero())).i, CoordSys3D(Str('N'), Tuple(ImmutableDenseMatrix("\
        "[[Integer(1), Integer(0), Integer(0)], [Integer(0), Integer(1), "\
        "Integer(0)], [Integer(0), Integer(0), Integer(1)]]), VectorZero())).j)"
    assert srepr(dy) == res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_astype.py ===
import numpy as np
import pytest

from pandas._libs.sparse import IntIndex

from pandas import (
    SparseDtype,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


class TestAstype:
    def test_astype(self):
        # float -> float
        arr = SparseArray([None, None, 0, 2])
        result = arr.astype("Sparse[float32]")
        expected = SparseArray([None, None, 0, 2], dtype=np.dtype("float32"))
        tm.assert_sp_array_equal(result, expected)

        dtype = SparseDtype("float64", fill_value=0)
        result = arr.astype(dtype)
        expected = SparseArray._simple_new(
            np.array([0.0, 2.0], dtype=dtype.subtype), IntIndex(4, [2, 3]), dtype
        )
        tm.assert_sp_array_equal(result, expected)

        dtype = SparseDtype("int64", 0)
        result = arr.astype(dtype)
        expected = SparseArray._simple_new(
            np.array([0, 2], dtype=np.int64), IntIndex(4, [2, 3]), dtype
        )
        tm.assert_sp_array_equal(result, expected)

        arr = SparseArray([0, np.nan, 0, 1], fill_value=0)
        with pytest.raises(ValueError, match="NA"):
            arr.astype("Sparse[i8]")

    def test_astype_bool(self):
        a = SparseArray([1, 0, 0, 1], dtype=SparseDtype(int, 0))
        result = a.astype(bool)
        expected = np.array([1, 0, 0, 1], dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

        # update fill value
        result = a.astype(SparseDtype(bool, False))
        expected = SparseArray(
            [True, False, False, True], dtype=SparseDtype(bool, False)
        )
        tm.assert_sp_array_equal(result, expected)

    def test_astype_all(self, any_real_numpy_dtype):
        vals = np.array([1, 2, 3])
        arr = SparseArray(vals, fill_value=1)
        typ = np.dtype(any_real_numpy_dtype)
        res = arr.astype(typ)
        tm.assert_numpy_array_equal(res, vals.astype(any_real_numpy_dtype))

    @pytest.mark.parametrize(
        "arr, dtype, expected",
        [
            (
                SparseArray([0, 1]),
                "float",
                SparseArray([0.0, 1.0], dtype=SparseDtype(float, 0.0)),
            ),
            (SparseArray([0, 1]), bool, SparseArray([False, True])),
            (
                SparseArray([0, 1], fill_value=1),
                bool,
                SparseArray([False, True], dtype=SparseDtype(bool, True)),
            ),
            pytest.param(
                SparseArray([0, 1]),
                "datetime64[ns]",
                SparseArray(
                    np.array([0, 1], dtype="datetime64[ns]"),
                    dtype=SparseDtype("datetime64[ns]", Timestamp("1970")),
                ),
            ),
            (
                SparseArray([0, 1, 10]),
                np.str_,
                SparseArray(["0", "1", "10"], dtype=SparseDtype(np.str_, "0")),
            ),
            (SparseArray(["10", "20"]), float, SparseArray([10.0, 20.0])),
            (
                SparseArray([0, 1, 0]),
                object,
                SparseArray([0, 1, 0], dtype=SparseDtype(object, 0)),
            ),
        ],
    )
    def test_astype_more(self, arr, dtype, expected):
        result = arr.astype(arr.dtype.update_dtype(dtype))
        tm.assert_sp_array_equal(result, expected)

    def test_astype_nan_raises(self):
        arr = SparseArray([1.0, np.nan])
        with pytest.raises(ValueError, match="Cannot convert non-finite"):
            arr.astype(int)

    def test_astype_copy_false(self):
        # GH#34456 bug caused by using .view instead of .astype in astype_nansafe
        arr = SparseArray([1, 2, 3])

        dtype = SparseDtype(float, 0)

        result = arr.astype(dtype, copy=False)
        expected = SparseArray([1.0, 2.0, 3.0], fill_value=0.0)
        tm.assert_sp_array_equal(result, expected)

    def test_astype_dt64_to_int64(self):
        # GH#49631 match non-sparse behavior
        values = np.array(["NaT", "2016-01-02", "2016-01-03"], dtype="M8[ns]")

        arr = SparseArray(values)
        result = arr.astype("int64")
        expected = values.astype("int64")
        tm.assert_numpy_array_equal(result, expected)

        # we should also be able to cast to equivalent Sparse[int64]
        dtype_int64 = SparseDtype("int64", np.iinfo(np.int64).min)
        result2 = arr.astype(dtype_int64)
        tm.assert_numpy_array_equal(result2.to_numpy(), expected)

        # GH#50087 we should match the non-sparse behavior regardless of
        #  if we have a fill_value other than NaT
        dtype = SparseDtype("datetime64[ns]", values[1])
        arr3 = SparseArray(values, dtype=dtype)
        result3 = arr3.astype("int64")
        tm.assert_numpy_array_equal(result3, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_fieldfunctions.py ===
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.physics.vector import ReferenceFrame, Vector, Point, \
     dynamicsymbols
from sympy.physics.vector.fieldfunctions import divergence, \
     gradient, curl, is_conservative, is_solenoidal, \
     scalar_potential, scalar_potential_difference
from sympy.testing.pytest import raises

R = ReferenceFrame('R')
q = dynamicsymbols('q')
P = R.orientnew('P', 'Axis', [q, R.z])


def test_curl():
    assert curl(Vector(0), R) == Vector(0)
    assert curl(R.x, R) == Vector(0)
    assert curl(2*R[1]**2*R.y, R) == Vector(0)
    assert curl(R[0]*R[1]*R.z, R) == R[0]*R.x - R[1]*R.y
    assert curl(R[0]*R[1]*R[2] * (R.x+R.y+R.z), R) == \
           (-R[0]*R[1] + R[0]*R[2])*R.x + (R[0]*R[1] - R[1]*R[2])*R.y + \
           (-R[0]*R[2] + R[1]*R[2])*R.z
    assert curl(2*R[0]**2*R.y, R) == 4*R[0]*R.z
    assert curl(P[0]**2*R.x + P.y, R) == \
           - 2*(R[0]*cos(q) + R[1]*sin(q))*sin(q)*R.z
    assert curl(P[0]*R.y, P) == cos(q)*P.z


def test_divergence():
    assert divergence(Vector(0), R) is S.Zero
    assert divergence(R.x, R) is S.Zero
    assert divergence(R[0]**2*R.x, R) == 2*R[0]
    assert divergence(R[0]*R[1]*R[2] * (R.x+R.y+R.z), R) == \
           R[0]*R[1] + R[0]*R[2] + R[1]*R[2]
    assert divergence((1/(R[0]*R[1]*R[2])) * (R.x+R.y+R.z), R) == \
           -1/(R[0]*R[1]*R[2]**2) - 1/(R[0]*R[1]**2*R[2]) - \
           1/(R[0]**2*R[1]*R[2])
    v = P[0]*P.x + P[1]*P.y + P[2]*P.z
    assert divergence(v, P) == 3
    assert divergence(v, R).simplify() == 3
    assert divergence(P[0]*R.x + R[0]*P.x, R) == 2*cos(q)


def test_gradient():
    a = Symbol('a')
    assert gradient(0, R) == Vector(0)
    assert gradient(R[0], R) == R.x
    assert gradient(R[0]*R[1]*R[2], R) == \
           R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z
    assert gradient(2*R[0]**2, R) == 4*R[0]*R.x
    assert gradient(a*sin(R[1])/R[0], R) == \
           - a*sin(R[1])/R[0]**2*R.x + a*cos(R[1])/R[0]*R.y
    assert gradient(P[0]*P[1], R) == \
           ((-R[0]*sin(q) + R[1]*cos(q))*cos(q) - (R[0]*cos(q) + R[1]*sin(q))*sin(q))*R.x + \
           ((-R[0]*sin(q) + R[1]*cos(q))*sin(q) + (R[0]*cos(q) + R[1]*sin(q))*cos(q))*R.y
    assert gradient(P[0]*R[2], P) == P[2]*P.x + P[0]*P.z


scalar_field = 2*R[0]**2*R[1]*R[2]
grad_field = gradient(scalar_field, R)
vector_field = R[1]**2*R.x + 3*R[0]*R.y + 5*R[1]*R[2]*R.z
curl_field = curl(vector_field, R)


def test_conservative():
    assert is_conservative(0) is True
    assert is_conservative(R.x) is True
    assert is_conservative(2 * R.x + 3 * R.y + 4 * R.z) is True
    assert is_conservative(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z) is \
           True
    assert is_conservative(R[0] * R.y) is False
    assert is_conservative(grad_field) is True
    assert is_conservative(curl_field) is False
    assert is_conservative(4*R[0]*R[1]*R[2]*R.x + 2*R[0]**2*R[2]*R.y) is \
                           False
    assert is_conservative(R[2]*P.x + P[0]*R.z) is True


def test_solenoidal():
    assert is_solenoidal(0) is True
    assert is_solenoidal(R.x) is True
    assert is_solenoidal(2 * R.x + 3 * R.y + 4 * R.z) is True
    assert is_solenoidal(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z) is \
           True
    assert is_solenoidal(R[1] * R.y) is False
    assert is_solenoidal(grad_field) is False
    assert is_solenoidal(curl_field) is True
    assert is_solenoidal((-2*R[1] + 3)*R.z) is True
    assert is_solenoidal(cos(q)*R.x + sin(q)*R.y + cos(q)*P.z) is True
    assert is_solenoidal(R[2]*P.x + P[0]*R.z) is True


def test_scalar_potential():
    assert scalar_potential(0, R) == 0
    assert scalar_potential(R.x, R) == R[0]
    assert scalar_potential(R.y, R) == R[1]
    assert scalar_potential(R.z, R) == R[2]
    assert scalar_potential(R[1]*R[2]*R.x + R[0]*R[2]*R.y + \
                            R[0]*R[1]*R.z, R) == R[0]*R[1]*R[2]
    assert scalar_potential(grad_field, R) == scalar_field
    assert scalar_potential(R[2]*P.x + P[0]*R.z, R) == \
           R[0]*R[2]*cos(q) + R[1]*R[2]*sin(q)
    assert scalar_potential(R[2]*P.x + P[0]*R.z, P) == P[0]*P[2]
    raises(ValueError, lambda: scalar_potential(R[0] * R.y, R))


def test_scalar_potential_difference():
    origin = Point('O')
    point1 = origin.locatenew('P1', 1*R.x + 2*R.y + 3*R.z)
    point2 = origin.locatenew('P2', 4*R.x + 5*R.y + 6*R.z)
    genericpointR = origin.locatenew('RP', R[0]*R.x + R[1]*R.y + R[2]*R.z)
    genericpointP = origin.locatenew('PP', P[0]*P.x + P[1]*P.y + P[2]*P.z)
    assert scalar_potential_difference(S.Zero, R, point1, point2, \
                                       origin) == 0
    assert scalar_potential_difference(scalar_field, R, origin, \
                                       genericpointR, origin) == \
                                       scalar_field
    assert scalar_potential_difference(grad_field, R, origin, \
                                       genericpointR, origin) == \
                                       scalar_field
    assert scalar_potential_difference(grad_field, R, point1, point2,
                                       origin) == 948
    assert scalar_potential_difference(R[1]*R[2]*R.x + R[0]*R[2]*R.y + \
                                       R[0]*R[1]*R.z, R, point1,
                                       genericpointR, origin) == \
                                       R[0]*R[1]*R[2] - 6
    potential_diff_P = 2*P[2]*(P[0]*sin(q) + P[1]*cos(q))*\
                       (P[0]*cos(q) - P[1]*sin(q))**2
    assert scalar_potential_difference(grad_field, P, origin, \
                                       genericpointP, \
                                       origin).simplify() == \
                                       potential_diff_P

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_to_numpy.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy(box):
    con = pd.Series if box else pd.array

    # default (with or without missing values) -> object dtype
    arr = con([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy()
    expected = np.array([0.1, 0.2, 0.3], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([0.1, 0.2, None], dtype="Float64")
    result = arr.to_numpy()
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_float(box):
    con = pd.Series if box else pd.array

    # no missing values -> can convert to float, otherwise raises
    arr = con([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy(dtype="float64")
    expected = np.array([0.1, 0.2, 0.3], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([0.1, 0.2, None], dtype="Float64")
    result = arr.to_numpy(dtype="float64")
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="float64", na_value=np.nan)
    expected = np.array([0.1, 0.2, np.nan], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_int(box):
    con = pd.Series if box else pd.array

    # no missing values -> can convert to int, otherwise raises
    arr = con([1.0, 2.0, 3.0], dtype="Float64")
    result = arr.to_numpy(dtype="int64")
    expected = np.array([1, 2, 3], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    arr = con([1.0, 2.0, None], dtype="Float64")
    with pytest.raises(ValueError, match="cannot convert to 'int64'-dtype"):
        result = arr.to_numpy(dtype="int64")

    # automatic casting (floors the values)
    arr = con([0.1, 0.9, 1.1], dtype="Float64")
    result = arr.to_numpy(dtype="int64")
    expected = np.array([0, 0, 1], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_na_value(box):
    con = pd.Series if box else pd.array

    arr = con([0.0, 1.0, None], dtype="Float64")
    result = arr.to_numpy(dtype=object, na_value=None)
    expected = np.array([0.0, 1.0, None], dtype="object")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype=bool, na_value=False)
    expected = np.array([False, True, False], dtype="bool")
    tm.assert_numpy_array_equal(result, expected)

    result = arr.to_numpy(dtype="int64", na_value=-99)
    expected = np.array([0, 1, -99], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_na_value_with_nan():
    # array with both NaN and NA -> only fill NA with `na_value`
    arr = FloatingArray(np.array([0.0, np.nan, 0.0]), np.array([False, False, True]))
    result = arr.to_numpy(dtype="float64", na_value=-1)
    expected = np.array([0.0, np.nan, -1.0], dtype="float64")
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64", "bool"])
@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_dtype(box, dtype):
    con = pd.Series if box else pd.array
    arr = con([0.0, 1.0], dtype="Float64")

    result = arr.to_numpy(dtype=dtype)
    expected = np.array([0, 1], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_na_raises(box, dtype):
    con = pd.Series if box else pd.array
    arr = con([0.0, 1.0, None], dtype="Float64")
    with pytest.raises(ValueError, match=dtype):
        arr.to_numpy(dtype=dtype)


@pytest.mark.parametrize("box", [True, False], ids=["series", "array"])
def test_to_numpy_string(box, dtype):
    con = pd.Series if box else pd.array
    arr = con([0.0, 1.0, None], dtype="Float64")

    result = arr.to_numpy(dtype="str")
    expected = np.array([0.0, 1.0, pd.NA], dtype=f"{tm.ENDIAN}U32")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_copy():
    # to_numpy can be zero-copy if no missing values
    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy(dtype="float64")
    result[0] = 10
    tm.assert_extension_array_equal(arr, pd.array([10, 0.2, 0.3], dtype="Float64"))

    arr = pd.array([0.1, 0.2, 0.3], dtype="Float64")
    result = arr.to_numpy(dtype="float64", copy=True)
    result[0] = 10
    tm.assert_extension_array_equal(arr, pd.array([0.1, 0.2, 0.3], dtype="Float64"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_at_time.py ===
from datetime import time

import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import timezones

from pandas import (
    DataFrame,
    date_range,
)
import pandas._testing as tm


class TestAtTime:
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_localized_at_time(self, tzstr, frame_or_series):
        tz = timezones.maybe_get_tz(tzstr)

        rng = date_range("4/16/2012", "5/1/2012", freq="h")
        ts = frame_or_series(
            np.random.default_rng(2).standard_normal(len(rng)), index=rng
        )

        ts_local = ts.tz_localize(tzstr)

        result = ts_local.at_time(time(10, 0))
        expected = ts.at_time(time(10, 0)).tz_localize(tzstr)
        tm.assert_equal(result, expected)
        assert timezones.tz_compare(result.index.tz, tz)

    def test_at_time(self, frame_or_series):
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)
        rs = ts.at_time(rng[1])
        assert (rs.index.hour == rng[1].hour).all()
        assert (rs.index.minute == rng[1].minute).all()
        assert (rs.index.second == rng[1].second).all()

        result = ts.at_time("9:30")
        expected = ts.at_time(time(9, 30))
        tm.assert_equal(result, expected)

    def test_at_time_midnight(self, frame_or_series):
        # midnight, everything
        rng = date_range("1/1/2000", "1/31/2000")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)

        result = ts.at_time(time(0, 0))
        tm.assert_equal(result, ts)

    def test_at_time_nonexistent(self, frame_or_series):
        # time doesn't exist
        rng = date_range("1/1/2012", freq="23Min", periods=384)
        ts = DataFrame(np.random.default_rng(2).standard_normal(len(rng)), rng)
        ts = tm.get_obj(ts, frame_or_series)
        rs = ts.at_time("16:00")
        assert len(rs) == 0

    @pytest.mark.parametrize(
        "hour", ["1:00", "1:00AM", time(1), time(1, tzinfo=pytz.UTC)]
    )
    def test_at_time_errors(self, hour):
        # GH#24043
        dti = date_range("2018", periods=3, freq="h")
        df = DataFrame(list(range(len(dti))), index=dti)
        if getattr(hour, "tzinfo", None) is None:
            result = df.at_time(hour)
            expected = df.iloc[1:2]
            tm.assert_frame_equal(result, expected)
        else:
            with pytest.raises(ValueError, match="Index must be timezone"):
                df.at_time(hour)

    def test_at_time_tz(self):
        # GH#24043
        dti = date_range("2018", periods=3, freq="h", tz="US/Pacific")
        df = DataFrame(list(range(len(dti))), index=dti)
        result = df.at_time(time(4, tzinfo=pytz.timezone("US/Eastern")))
        expected = df.iloc[1:2]
        tm.assert_frame_equal(result, expected)

    def test_at_time_raises(self, frame_or_series):
        # GH#20725
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        obj = tm.get_obj(obj, frame_or_series)
        msg = "Index must be DatetimeIndex"
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.at_time("00:00")

    @pytest.mark.parametrize("axis", ["index", "columns", 0, 1])
    def test_at_time_axis(self, axis):
        # issue 8839
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
        ts.index, ts.columns = rng, rng

        indices = rng[(rng.hour == 9) & (rng.minute == 30) & (rng.second == 0)]

        if axis in ["index", 0]:
            expected = ts.loc[indices, :]
        elif axis in ["columns", 1]:
            expected = ts.loc[:, indices]

        result = ts.at_time("9:30", axis=axis)

        # Without clearing freq, result has freq 1440T and expected 5T
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        tm.assert_frame_equal(result, expected)

    def test_at_time_datetimeindex(self):
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        akey = time(12, 0, 0)
        ainds = [24, 72, 120, 168]

        result = df.at_time(akey)
        expected = df.loc[akey]
        expected2 = df.iloc[ainds]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, expected2)
        assert len(result) == 4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\conftest.py ===
import pytest

from pandas import Series
from pandas.core.strings.accessor import StringMethods

_any_string_method = [
    ("cat", (), {"sep": ","}),
    ("cat", (Series(list("zyx")),), {"sep": ",", "join": "left"}),
    ("center", (10,), {}),
    ("contains", ("a",), {}),
    ("count", ("a",), {}),
    ("decode", ("UTF-8",), {}),
    ("encode", ("UTF-8",), {}),
    ("endswith", ("a",), {}),
    ("endswith", ((),), {}),
    ("endswith", (("a",),), {}),
    ("endswith", (("a", "b"),), {}),
    ("endswith", (("a", "MISSING"),), {}),
    ("endswith", ("a",), {"na": True}),
    ("endswith", ("a",), {"na": False}),
    ("extract", ("([a-z]*)",), {"expand": False}),
    ("extract", ("([a-z]*)",), {"expand": True}),
    ("extractall", ("([a-z]*)",), {}),
    ("find", ("a",), {}),
    ("findall", ("a",), {}),
    ("get", (0,), {}),
    # because "index" (and "rindex") fail intentionally
    # if the string is not found, search only for empty string
    ("index", ("",), {}),
    ("join", (",",), {}),
    ("ljust", (10,), {}),
    ("match", ("a",), {}),
    ("fullmatch", ("a",), {}),
    ("normalize", ("NFC",), {}),
    ("pad", (10,), {}),
    ("partition", (" ",), {"expand": False}),
    ("partition", (" ",), {"expand": True}),
    ("repeat", (3,), {}),
    ("replace", ("a", "z"), {}),
    ("rfind", ("a",), {}),
    ("rindex", ("",), {}),
    ("rjust", (10,), {}),
    ("rpartition", (" ",), {"expand": False}),
    ("rpartition", (" ",), {"expand": True}),
    ("slice", (0, 1), {}),
    ("slice_replace", (0, 1, "z"), {}),
    ("split", (" ",), {"expand": False}),
    ("split", (" ",), {"expand": True}),
    ("startswith", ("a",), {}),
    ("startswith", (("a",),), {}),
    ("startswith", (("a", "b"),), {}),
    ("startswith", (("a", "MISSING"),), {}),
    ("startswith", ((),), {}),
    ("startswith", ("a",), {"na": True}),
    ("startswith", ("a",), {"na": False}),
    ("removeprefix", ("a",), {}),
    ("removesuffix", ("a",), {}),
    # translating unicode points of "a" to "d"
    ("translate", ({97: 100},), {}),
    ("wrap", (2,), {}),
    ("zfill", (10,), {}),
] + list(
    zip(
        [
            # methods without positional arguments: zip with empty tuple and empty dict
            "capitalize",
            "cat",
            "get_dummies",
            "isalnum",
            "isalpha",
            "isdecimal",
            "isdigit",
            "islower",
            "isnumeric",
            "isspace",
            "istitle",
            "isupper",
            "len",
            "lower",
            "lstrip",
            "partition",
            "rpartition",
            "rsplit",
            "rstrip",
            "slice",
            "slice_replace",
            "split",
            "strip",
            "swapcase",
            "title",
            "upper",
            "casefold",
        ],
        [()] * 100,
        [{}] * 100,
    )
)
ids, _, _ = zip(*_any_string_method)  # use method name as fixture-id
missing_methods = {f for f in dir(StringMethods) if not f.startswith("_")} - set(ids)

# test that the above list captures all methods of StringMethods
assert not missing_methods


@pytest.fixture(params=_any_string_method, ids=ids)
def any_string_method(request):
    """
    Fixture for all public methods of `StringMethods`

    This fixture returns a tuple of the method name and sample arguments
    necessary to call the method.

    Returns
    -------
    method_name : str
        The name of the method in `StringMethods`
    args : tuple
        Sample values for the positional arguments
    kwargs : dict
        Sample values for the keyword arguments

    Examples
    --------
    >>> def test_something(any_string_method):
    ...     s = Series(['a', 'b', np.nan, 'd'])
    ...
    ...     method_name, args, kwargs = any_string_method
    ...     method = getattr(s.str, method_name)
    ...     # will not raise
    ...     method(*args, **kwargs)
    """
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_sparsetools.py ===
from sympy.matrices.sparsetools import _doktocsr, _csrtodok, banded
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices import SparseMatrix
from sympy.testing.pytest import raises


def test_doktocsr():
    a = SparseMatrix([[1, 2, 0, 0], [0, 3, 9, 0], [0, 1, 4, 0]])
    b = SparseMatrix(4, 6, [10, 20, 0, 0, 0, 0, 0, 30, 0, 40, 0, 0, 0, 0, 50,
        60, 70, 0, 0, 0, 0, 0, 0, 80])
    c = SparseMatrix(4, 4, [0, 0, 0, 0, 0, 12, 0, 2, 15, 0, 12, 0, 0, 0, 0, 4])
    d = SparseMatrix(10, 10, {(1, 1): 12, (3, 5): 7, (7, 8): 12})
    e = SparseMatrix([[0, 0, 0], [1, 0, 2], [3, 0, 0]])
    f = SparseMatrix(7, 8, {(2, 3): 5, (4, 5):12})
    assert _doktocsr(a) == [[1, 2, 3, 9, 1, 4], [0, 1, 1, 2, 1, 2],
        [0, 2, 4, 6], [3, 4]]
    assert _doktocsr(b) == [[10, 20, 30, 40, 50, 60, 70, 80],
        [0, 1, 1, 3, 2, 3, 4, 5], [0, 2, 4, 7, 8], [4, 6]]
    assert _doktocsr(c) == [[12, 2, 15, 12, 4], [1, 3, 0, 2, 3],
        [0, 0, 2, 4, 5], [4, 4]]
    assert _doktocsr(d) == [[12, 7, 12], [1, 5, 8],
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], [10, 10]]
    assert _doktocsr(e) == [[1, 2, 3], [0, 2, 0], [0, 0, 2, 3], [3, 3]]
    assert _doktocsr(f) == [[5, 12], [3, 5], [0, 0, 0, 1, 1, 2, 2, 2], [7, 8]]


def test_csrtodok():
    h = [[5, 7, 5], [2, 1, 3], [0, 1, 1, 3], [3, 4]]
    g = [[12, 5, 4], [2, 4, 2], [0, 1, 2, 3], [3, 7]]
    i = [[1, 3, 12], [0, 2, 4], [0, 2, 3], [2, 5]]
    j = [[11, 15, 12, 15], [2, 4, 1, 2], [0, 1, 1, 2, 3, 4], [5, 8]]
    k = [[1, 3], [2, 1], [0, 1, 1, 2], [3, 3]]
    m = _csrtodok(h)
    assert isinstance(m, SparseMatrix)
    assert m == SparseMatrix(3, 4,
        {(0, 2): 5, (2, 1): 7, (2, 3): 5})
    assert _csrtodok(g) == SparseMatrix(3, 7,
        {(0, 2): 12, (1, 4): 5, (2, 2): 4})
    assert _csrtodok(i) == SparseMatrix([[1, 0, 3, 0, 0], [0, 0, 0, 0, 12]])
    assert _csrtodok(j) == SparseMatrix(5, 8,
        {(0, 2): 11, (2, 4): 15, (3, 1): 12, (4, 2): 15})
    assert _csrtodok(k) == SparseMatrix(3, 3, {(0, 2): 1, (2, 1): 3})


def test_banded():
    raises(TypeError, lambda: banded())
    raises(TypeError, lambda: banded(1))
    raises(TypeError, lambda: banded(1, 2))
    raises(TypeError, lambda: banded(1, 2, 3))
    raises(TypeError, lambda: banded(1, 2, 3, 4))
    raises(ValueError, lambda: banded({0: (1, 2)}, rows=1))
    raises(ValueError, lambda: banded({0: (1, 2)}, cols=1))
    raises(ValueError, lambda: banded(1, {0: (1, 2)}))
    raises(ValueError, lambda: banded(2, 1, {0: (1, 2)}))
    raises(ValueError, lambda: banded(1, 2, {0: (1, 2)}))

    assert isinstance(banded(2, 4, {}), SparseMatrix)
    assert banded(2, 4, {}) == zeros(2, 4)
    assert banded({0: 0, 1: 0}) == zeros(0)
    assert banded({0: Matrix([1, 2])}) == Matrix([1, 2])
    assert banded({1: [1, 2, 3, 0], -1: [4, 5, 6]}) == \
        banded({1: (1, 2, 3), -1: (4, 5, 6)}) == \
        Matrix([
        [0, 1, 0, 0],
        [4, 0, 2, 0],
        [0, 5, 0, 3],
        [0, 0, 6, 0]])
    assert banded(3, 4, {-1: 1, 0: 2, 1: 3}) == \
        Matrix([
        [2, 3, 0, 0],
        [1, 2, 3, 0],
        [0, 1, 2, 3]])
    s = lambda d: (1 + d)**2
    assert banded(5, {0: s, 2: s}) == \
        Matrix([
        [1, 0, 1,  0,  0],
        [0, 4, 0,  4,  0],
        [0, 0, 9,  0,  9],
        [0, 0, 0, 16,  0],
        [0, 0, 0,  0, 25]])
    assert banded(2, {0: 1}) == \
        Matrix([
        [1, 0],
        [0, 1]])
    assert banded(2, 3, {0: 1}) == \
        Matrix([
        [1, 0, 0],
        [0, 1, 0]])
    vert = Matrix([1, 2, 3])
    assert banded({0: vert}, cols=3) == \
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [3, 2, 1],
        [0, 3, 2],
        [0, 0, 3]])
    assert banded(4, {0: ones(2)}) == \
        Matrix([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]])
    raises(ValueError, lambda: banded({0: 2, 1: ones(2)}, rows=5))
    assert banded({0: 2, 2: (ones(2),)*3}) == \
        Matrix([
        [2, 0, 1, 1, 0, 0, 0, 0],
        [0, 2, 1, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 1, 1, 0, 0],
        [0, 0, 0, 2, 1, 1, 0, 0],
        [0, 0, 0, 0, 2, 0, 1, 1],
        [0, 0, 0, 0, 0, 2, 1, 1]])
    raises(ValueError, lambda: banded({0: (2,)*5, 1: (ones(2),)*3}))
    u2 = Matrix([[1, 1], [0, 1]])
    assert banded({0: (2,)*5, 1: (u2,)*3}) == \
        Matrix([
        [2, 1, 1, 0, 0, 0, 0],
        [0, 2, 1, 0, 0, 0, 0],
        [0, 0, 2, 1, 1, 0, 0],
        [0, 0, 0, 2, 1, 0, 0],
        [0, 0, 0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0, 0, 1]])
    assert banded({0:(0, ones(2)), 2: 2}) == \
        Matrix([
        [0, 0, 2],
        [0, 1, 1],
        [0, 1, 1]])
    raises(ValueError, lambda: banded({0: (0, ones(2)), 1: 2}))
    assert banded({0: 1}, cols=3) == banded({0: 1}, rows=3) == eye(3)
    assert banded({1: 1}, rows=3) == Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\tests\winterm_test.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
import sys
from unittest import TestCase, main, skipUnless

try:
    from unittest.mock import Mock, patch
except ImportError:
    from mock import Mock, patch

from ..winterm import WinColor, WinStyle, WinTerm


class WinTermTest(TestCase):

    @patch('colorama.winterm.win32')
    def testInit(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 7 + 6 * 16 + 8
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()
        self.assertEqual(term._fore, 7)
        self.assertEqual(term._back, 6)
        self.assertEqual(term._style, 8)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testGetAttrs(self):
        term = WinTerm()

        term._fore = 0
        term._back = 0
        term._style = 0
        self.assertEqual(term.get_attrs(), 0)

        term._fore = WinColor.YELLOW
        self.assertEqual(term.get_attrs(), WinColor.YELLOW)

        term._back = WinColor.MAGENTA
        self.assertEqual(
            term.get_attrs(),
            WinColor.YELLOW + WinColor.MAGENTA * 16)

        term._style = WinStyle.BRIGHT
        self.assertEqual(
            term.get_attrs(),
            WinColor.YELLOW + WinColor.MAGENTA * 16 + WinStyle.BRIGHT)

    @patch('colorama.winterm.win32')
    def testResetAll(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 1 + 2 * 16 + 8
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()

        term.set_console = Mock()
        term._fore = -1
        term._back = -1
        term._style = -1

        term.reset_all()

        self.assertEqual(term._fore, 1)
        self.assertEqual(term._back, 2)
        self.assertEqual(term._style, 8)
        self.assertEqual(term.set_console.called, True)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testFore(self):
        term = WinTerm()
        term.set_console = Mock()
        term._fore = 0

        term.fore(5)

        self.assertEqual(term._fore, 5)
        self.assertEqual(term.set_console.called, True)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testBack(self):
        term = WinTerm()
        term.set_console = Mock()
        term._back = 0

        term.back(5)

        self.assertEqual(term._back, 5)
        self.assertEqual(term.set_console.called, True)

    @skipUnless(sys.platform.startswith("win"), "requires Windows")
    def testStyle(self):
        term = WinTerm()
        term.set_console = Mock()
        term._style = 0

        term.style(22)

        self.assertEqual(term._style, 22)
        self.assertEqual(term.set_console.called, True)

    @patch('colorama.winterm.win32')
    def testSetConsole(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 0
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()
        term.windll = Mock()

        term.set_console()

        self.assertEqual(
            mockWin32.SetConsoleTextAttribute.call_args,
            ((mockWin32.STDOUT, term.get_attrs()), {})
        )

    @patch('colorama.winterm.win32')
    def testSetConsoleOnStderr(self, mockWin32):
        mockAttr = Mock()
        mockAttr.wAttributes = 0
        mockWin32.GetConsoleScreenBufferInfo.return_value = mockAttr
        term = WinTerm()
        term.windll = Mock()

        term.set_console(on_stderr=True)

        self.assertEqual(
            mockWin32.SetConsoleTextAttribute.call_args,
            ((mockWin32.STDERR, term.get_attrs()), {})
        )


if __name__ == '__main__':
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\bitwise_ops.py ===
import numpy as np

i8 = np.int64(1)
u8 = np.uint64(1)

i4 = np.int32(1)
u4 = np.uint32(1)

b_ = np.bool(1)

b = bool(1)
i = int(1)

AR = np.array([0, 1, 2], dtype=np.int32)
AR.setflags(write=False)


i8 << i8
i8 >> i8
i8 | i8
i8 ^ i8
i8 & i8

i << AR
i >> AR
i | AR
i ^ AR
i & AR

i8 << AR
i8 >> AR
i8 | AR
i8 ^ AR
i8 & AR

i4 << i4
i4 >> i4
i4 | i4
i4 ^ i4
i4 & i4

i8 << i4
i8 >> i4
i8 | i4
i8 ^ i4
i8 & i4

i8 << i
i8 >> i
i8 | i
i8 ^ i
i8 & i

i8 << b_
i8 >> b_
i8 | b_
i8 ^ b_
i8 & b_

i8 << b
i8 >> b
i8 | b
i8 ^ b
i8 & b

u8 << u8
u8 >> u8
u8 | u8
u8 ^ u8
u8 & u8

u4 << u4
u4 >> u4
u4 | u4
u4 ^ u4
u4 & u4

u4 << i4
u4 >> i4
u4 | i4
u4 ^ i4
u4 & i4

u4 << i
u4 >> i
u4 | i
u4 ^ i
u4 & i

u8 << b_
u8 >> b_
u8 | b_
u8 ^ b_
u8 & b_

u8 << b
u8 >> b
u8 | b
u8 ^ b
u8 & b

b_ << b_
b_ >> b_
b_ | b_
b_ ^ b_
b_ & b_

b_ << AR
b_ >> AR
b_ | AR
b_ ^ AR
b_ & AR

b_ << b
b_ >> b
b_ | b
b_ ^ b
b_ & b

b_ << i
b_ >> i
b_ | i
b_ ^ i
b_ & i

~i8
~i4
~u8
~u4
~b_
~AR

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_errstate.py ===
import sysconfig

import pytest

import numpy as np
from numpy.testing import IS_WASM, assert_raises

# The floating point emulation on ARM EABI systems lacking a hardware FPU is
# known to be buggy. This is an attempt to identify these hosts. It may not
# catch all possible cases, but it catches the known cases of gh-413 and
# gh-15562.
hosttype = sysconfig.get_config_var('HOST_GNU_TYPE')
arm_softfloat = False if hosttype is None else hosttype.endswith('gnueabi')

class TestErrstate:
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat,
                        reason='platform/cpu issue with FPU (gh-413,-15562)')
    def test_invalid(self):
        with np.errstate(all='raise', under='ignore'):
            a = -np.arange(3)
            # This should work
            with np.errstate(invalid='ignore'):
                np.sqrt(a)
            # While this should fail!
            with assert_raises(FloatingPointError):
                np.sqrt(a)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat,
                        reason='platform/cpu issue with FPU (gh-15562)')
    def test_divide(self):
        with np.errstate(all='raise', under='ignore'):
            a = -np.arange(3)
            # This should work
            with np.errstate(divide='ignore'):
                a // 0
            # While this should fail!
            with assert_raises(FloatingPointError):
                a // 0
            # As should this, see gh-15562
            with assert_raises(FloatingPointError):
                a // a

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat,
                        reason='platform/cpu issue with FPU (gh-15562)')
    def test_errcall(self):
        count = 0

        def foo(*args):
            nonlocal count
            count += 1

        olderrcall = np.geterrcall()
        with np.errstate(call=foo):
            assert np.geterrcall() is foo
            with np.errstate(call=None):
                assert np.geterrcall() is None
        assert np.geterrcall() is olderrcall
        assert count == 0

        with np.errstate(call=foo, invalid="call"):
            np.array(np.inf) - np.array(np.inf)

        assert count == 1

    def test_errstate_decorator(self):
        @np.errstate(all='ignore')
        def foo():
            a = -np.arange(3)
            a // 0

        foo()

    def test_errstate_enter_once(self):
        errstate = np.errstate(invalid="warn")
        with errstate:
            pass

        # The errstate context cannot be entered twice as that would not be
        # thread-safe
        with pytest.raises(TypeError,
                match="Cannot enter `np.errstate` twice"):
            with errstate:
                pass

    @pytest.mark.skipif(IS_WASM, reason="wasm doesn't support asyncio")
    def test_asyncio_safe(self):
        # asyncio may not always work, lets assume its fine if missing
        # Pyodide/wasm doesn't support it.  If this test makes problems,
        # it should just be skipped liberally (or run differently).
        asyncio = pytest.importorskip("asyncio")

        @np.errstate(invalid="ignore")
        def decorated():
            # Decorated non-async function (it is not safe to decorate an
            # async one)
            assert np.geterr()["invalid"] == "ignore"

        async def func1():
            decorated()
            await asyncio.sleep(0.1)
            decorated()

        async def func2():
            with np.errstate(invalid="raise"):
                assert np.geterr()["invalid"] == "raise"
                await asyncio.sleep(0.125)
                assert np.geterr()["invalid"] == "raise"

        # for good sport, a third one with yet another state:
        async def func3():
            with np.errstate(invalid="print"):
                assert np.geterr()["invalid"] == "print"
                await asyncio.sleep(0.11)
                assert np.geterr()["invalid"] == "print"

        async def main():
            # simply run all three function multiple times:
            await asyncio.gather(
                    func1(), func2(), func3(), func1(), func2(), func3(),
                    func1(), func2(), func3(), func1(), func2(), func3())

        loop = asyncio.new_event_loop()
        with np.errstate(invalid="warn"):
            asyncio.run(main())
            assert np.geterr()["invalid"] == "warn"

        assert np.geterr()["invalid"] == "warn"  # the default
        loop.close()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\__init__.py ===
"""
Base test suite for extension arrays.

These tests are intended for third-party libraries to subclass to validate
that their extension arrays and dtypes satisfy the interface. Moving or
renaming the tests should not be done lightly.

Libraries are expected to implement a few pytest fixtures to provide data
for the tests. The fixtures may be located in either

* The same module as your test class.
* A ``conftest.py`` in the same directory as your test class.

The full list of fixtures may be found in the ``conftest.py`` next to this
file.

.. code-block:: python

   import pytest
   from pandas.tests.extension.base import BaseDtypeTests


   @pytest.fixture
   def dtype():
       return MyDtype()


   class TestMyDtype(BaseDtypeTests):
       pass


Your class ``TestDtype`` will inherit all the tests defined on
``BaseDtypeTests``. pytest's fixture discover will supply your ``dtype``
wherever the test requires it. You're free to implement additional tests.

"""
from pandas.tests.extension.base.accumulate import BaseAccumulateTests
from pandas.tests.extension.base.casting import BaseCastingTests
from pandas.tests.extension.base.constructors import BaseConstructorsTests
from pandas.tests.extension.base.dim2 import (  # noqa: F401
    Dim2CompatTests,
    NDArrayBacked2DTests,
)
from pandas.tests.extension.base.dtype import BaseDtypeTests
from pandas.tests.extension.base.getitem import BaseGetitemTests
from pandas.tests.extension.base.groupby import BaseGroupbyTests
from pandas.tests.extension.base.index import BaseIndexTests
from pandas.tests.extension.base.interface import BaseInterfaceTests
from pandas.tests.extension.base.io import BaseParsingTests
from pandas.tests.extension.base.methods import BaseMethodsTests
from pandas.tests.extension.base.missing import BaseMissingTests
from pandas.tests.extension.base.ops import (  # noqa: F401
    BaseArithmeticOpsTests,
    BaseComparisonOpsTests,
    BaseOpsUtil,
    BaseUnaryOpsTests,
)
from pandas.tests.extension.base.printing import BasePrintingTests
from pandas.tests.extension.base.reduce import BaseReduceTests
from pandas.tests.extension.base.reshaping import BaseReshapingTests
from pandas.tests.extension.base.setitem import BaseSetitemTests


# One test class that you can inherit as an alternative to inheriting all the
# test classes above.
# Note 1) this excludes Dim2CompatTests and NDArrayBacked2DTests.
# Note 2) this uses BaseReduceTests and and _not_ BaseBooleanReduceTests,
#  BaseNoReduceTests, or BaseNumericReduceTests
class ExtensionTests(
    BaseAccumulateTests,
    BaseCastingTests,
    BaseConstructorsTests,
    BaseDtypeTests,
    BaseGetitemTests,
    BaseGroupbyTests,
    BaseIndexTests,
    BaseInterfaceTests,
    BaseParsingTests,
    BaseMethodsTests,
    BaseMissingTests,
    BaseArithmeticOpsTests,
    BaseComparisonOpsTests,
    BaseUnaryOpsTests,
    BasePrintingTests,
    BaseReduceTests,
    BaseReshapingTests,
    BaseSetitemTests,
    Dim2CompatTests,
):
    pass


def __getattr__(name: str):
    import warnings

    if name == "BaseNoReduceTests":
        warnings.warn(
            "BaseNoReduceTests is deprecated and will be removed in a "
            "future version. Use BaseReduceTests and override "
            "`_supports_reduction` instead.",
            FutureWarning,
        )
        from pandas.tests.extension.base.reduce import BaseNoReduceTests

        return BaseNoReduceTests

    elif name == "BaseNumericReduceTests":
        warnings.warn(
            "BaseNumericReduceTests is deprecated and will be removed in a "
            "future version. Use BaseReduceTests and override "
            "`_supports_reduction` instead.",
            FutureWarning,
        )
        from pandas.tests.extension.base.reduce import BaseNumericReduceTests

        return BaseNumericReduceTests

    elif name == "BaseBooleanReduceTests":
        warnings.warn(
            "BaseBooleanReduceTests is deprecated and will be removed in a "
            "future version. Use BaseReduceTests and override "
            "`_supports_reduction` instead.",
            FutureWarning,
        )
        from pandas.tests.extension.base.reduce import BaseBooleanReduceTests

        return BaseBooleanReduceTests

    raise AttributeError(
        f"module 'pandas.tests.extension.base' has no attribute '{name}'"
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_tz_convert.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


class TestTZConvert:
    def test_tz_convert(self, frame_or_series):
        rng = date_range("1/1/2011", periods=200, freq="D", tz="US/Eastern")

        obj = DataFrame({"a": 1}, index=rng)
        obj = tm.get_obj(obj, frame_or_series)

        result = obj.tz_convert("Europe/Berlin")
        expected = DataFrame({"a": 1}, rng.tz_convert("Europe/Berlin"))
        expected = tm.get_obj(expected, frame_or_series)

        assert result.index.tz.zone == "Europe/Berlin"
        tm.assert_equal(result, expected)

    def test_tz_convert_axis1(self):
        rng = date_range("1/1/2011", periods=200, freq="D", tz="US/Eastern")

        obj = DataFrame({"a": 1}, index=rng)

        obj = obj.T
        result = obj.tz_convert("Europe/Berlin", axis=1)
        assert result.columns.tz.zone == "Europe/Berlin"

        expected = DataFrame({"a": 1}, rng.tz_convert("Europe/Berlin"))

        tm.assert_equal(result, expected.T)

    def test_tz_convert_naive(self, frame_or_series):
        # can't convert tz-naive
        rng = date_range("1/1/2011", periods=200, freq="D")
        ts = Series(1, index=rng)
        ts = frame_or_series(ts)

        with pytest.raises(TypeError, match="Cannot convert tz-naive"):
            ts.tz_convert("US/Eastern")

    @pytest.mark.parametrize("fn", ["tz_localize", "tz_convert"])
    def test_tz_convert_and_localize(self, fn):
        l0 = date_range("20140701", periods=5, freq="D")
        l1 = date_range("20140701", periods=5, freq="D")

        int_idx = Index(range(5))

        if fn == "tz_convert":
            l0 = l0.tz_localize("UTC")
            l1 = l1.tz_localize("UTC")

        for idx in [l0, l1]:
            l0_expected = getattr(idx, fn)("US/Pacific")
            l1_expected = getattr(idx, fn)("US/Pacific")

            df1 = DataFrame(np.ones(5), index=l0)
            df1 = getattr(df1, fn)("US/Pacific")
            tm.assert_index_equal(df1.index, l0_expected)

            # MultiIndex
            # GH7846
            df2 = DataFrame(np.ones(5), MultiIndex.from_arrays([l0, l1]))

            # freq is not preserved in MultiIndex construction
            l1_expected = l1_expected._with_freq(None)
            l0_expected = l0_expected._with_freq(None)
            l1 = l1._with_freq(None)
            l0 = l0._with_freq(None)

            df3 = getattr(df2, fn)("US/Pacific", level=0)
            assert not df3.index.levels[0].equals(l0)
            tm.assert_index_equal(df3.index.levels[0], l0_expected)
            tm.assert_index_equal(df3.index.levels[1], l1)
            assert not df3.index.levels[1].equals(l1_expected)

            df3 = getattr(df2, fn)("US/Pacific", level=1)
            tm.assert_index_equal(df3.index.levels[0], l0)
            assert not df3.index.levels[0].equals(l0_expected)
            tm.assert_index_equal(df3.index.levels[1], l1_expected)
            assert not df3.index.levels[1].equals(l1)

            df4 = DataFrame(np.ones(5), MultiIndex.from_arrays([int_idx, l0]))

            # TODO: untested
            getattr(df4, fn)("US/Pacific", level=1)

            tm.assert_index_equal(df3.index.levels[0], l0)
            assert not df3.index.levels[0].equals(l0_expected)
            tm.assert_index_equal(df3.index.levels[1], l1_expected)
            assert not df3.index.levels[1].equals(l1)

        # Bad Inputs

        # Not DatetimeIndex / PeriodIndex
        with pytest.raises(TypeError, match="DatetimeIndex"):
            df = DataFrame(index=int_idx)
            getattr(df, fn)("US/Pacific")

        # Not DatetimeIndex / PeriodIndex
        with pytest.raises(TypeError, match="DatetimeIndex"):
            df = DataFrame(np.ones(5), MultiIndex.from_arrays([int_idx, l0]))
            getattr(df, fn)("US/Pacific", level=0)

        # Invalid level
        with pytest.raises(ValueError, match="not valid"):
            df = DataFrame(index=l0)
            getattr(df, fn)("US/Pacific", level=1)

    @pytest.mark.parametrize("copy", [True, False])
    def test_tz_convert_copy_inplace_mutate(self, copy, frame_or_series):
        # GH#6326
        obj = frame_or_series(
            np.arange(0, 5),
            index=date_range("20131027", periods=5, freq="h", tz="Europe/Berlin"),
        )
        orig = obj.copy()
        result = obj.tz_convert("UTC", copy=copy)
        expected = frame_or_series(np.arange(0, 5), index=obj.index.tz_convert("UTC"))
        tm.assert_equal(result, expected)
        tm.assert_equal(obj, orig)
        assert result.index is not obj.index
        assert result is not obj

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\tests\test_ideals.py ===
"""Test ideals.py code."""

from sympy.polys import QQ, ilex
from sympy.abc import x, y, z
from sympy.testing.pytest import raises


def test_ideal_operations():
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x)
    J = R.ideal(y)
    S = R.ideal(x*y)
    T = R.ideal(x, y)

    assert not (I == J)
    assert I == I

    assert I.union(J) == T
    assert I + J == T
    assert I + T == T

    assert not I.subset(T)
    assert T.subset(I)

    assert I.product(J) == S
    assert I*J == S
    assert x*J == S
    assert I*y == S
    assert R.convert(x)*J == S
    assert I*R.convert(y) == S

    assert not I.is_zero()
    assert not J.is_whole_ring()

    assert R.ideal(x**2 + 1, x).is_whole_ring()
    assert R.ideal() == R.ideal(0)
    assert R.ideal().is_zero()

    assert T.contains(x*y)
    assert T.subset([x, y])

    assert T.in_terms_of_generators(x) == [R(1), R(0)]

    assert T**0 == R.ideal(1)
    assert T**1 == T
    assert T**2 == R.ideal(x**2, y**2, x*y)
    assert I**5 == R.ideal(x**5)


def test_exceptions():
    I = QQ.old_poly_ring(x).ideal(x)
    J = QQ.old_poly_ring(y).ideal(1)
    raises(ValueError, lambda: I.union(x))
    raises(ValueError, lambda: I + J)
    raises(ValueError, lambda: I * J)
    raises(ValueError, lambda: I.union(J))
    assert (I == J) is False
    assert I != J


def test_nontriv_global():
    R = QQ.old_poly_ring(x, y, z)

    def contains(I, f):
        return R.ideal(*I).contains(f)

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4 + y**3 + 2*z*y*x)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y*z)
    assert contains([x, 1 + x + y, 5 - 7*y], 1)
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)

    # compare local order
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_nontriv_local():
    R = QQ.old_poly_ring(x, y, z, order=ilex)

    def contains(I, f):
        return R.ideal(*I).contains(f)

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_intersection():
    R = QQ.old_poly_ring(x, y, z)
    # SCA, example 1.8.11
    assert R.ideal(x, y).intersect(R.ideal(y**2, z)) == R.ideal(y**2, y*z, x*z)

    assert R.ideal(x, y).intersect(R.ideal()).is_zero()

    R = QQ.old_poly_ring(x, y, z, order="ilex")
    assert R.ideal(x, y).intersect(R.ideal(y**2 + y**2*z, z + z*x**3*y)) == \
        R.ideal(y**2, y*z, x*z)


def test_quotient():
    # SCA, example 1.8.13
    R = QQ.old_poly_ring(x, y, z)
    assert R.ideal(x, y).quotient(R.ideal(y**2, z)) == R.ideal(x, y)


def test_reduction():
    from sympy.polys.distributedmodules import sdm_nf_buchberger_reduced
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x**5, y)
    e = R.convert(x**3 + y**2)
    assert I.reduce_element(e) == e
    assert I.reduce_element(e, NF=sdm_nf_buchberger_reduced) == R.convert(x**3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\period\test_arrow_compat.py ===
import pytest

from pandas.compat.pyarrow import pa_version_under10p1

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    PeriodArray,
    period_array,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


pa = pytest.importorskip("pyarrow")


def test_arrow_extension_type():
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    p1 = ArrowPeriodType("D")
    p2 = ArrowPeriodType("D")
    p3 = ArrowPeriodType("M")

    assert p1.freq == "D"
    assert p1 == p2
    assert p1 != p3
    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)


@pytest.mark.xfail(not pa_version_under10p1, reason="Wrong behavior with pyarrow 10")
@pytest.mark.parametrize(
    "data, freq",
    [
        (pd.date_range("2017", periods=3), "D"),
        (pd.date_range("2017", periods=3, freq="YE"), "Y-DEC"),
    ],
)
def test_arrow_array(data, freq):
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    periods = period_array(data, freq=freq)
    result = pa.array(periods)
    assert isinstance(result.type, ArrowPeriodType)
    assert result.type.freq == freq
    expected = pa.array(periods.asi8, type="int64")
    assert result.storage.equals(expected)

    # convert to its storage type
    result = pa.array(periods, type=pa.int64())
    assert result.equals(expected)

    # unsupported conversions
    msg = "Not supported to convert PeriodArray to 'double' type"
    with pytest.raises(TypeError, match=msg):
        pa.array(periods, type="float64")

    with pytest.raises(TypeError, match="different 'freq'"):
        pa.array(periods, type=ArrowPeriodType("T"))


def test_arrow_array_missing():
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    arr = PeriodArray([1, 2, 3], dtype="period[D]")
    arr[1] = pd.NaT

    result = pa.array(arr)
    assert isinstance(result.type, ArrowPeriodType)
    assert result.type.freq == "D"
    expected = pa.array([1, None, 3], type="int64")
    assert result.storage.equals(expected)


def test_arrow_table_roundtrip():
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    arr = PeriodArray([1, 2, 3], dtype="period[D]")
    arr[1] = pd.NaT
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    assert isinstance(table.field("a").type, ArrowPeriodType)
    result = table.to_pandas()
    assert isinstance(result["a"].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)

    table2 = pa.concat_tables([table, table])
    result = table2.to_pandas()
    expected = pd.concat([df, df], ignore_index=True)
    tm.assert_frame_equal(result, expected)


def test_arrow_load_from_zero_chunks():
    # GH-41040

    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    arr = PeriodArray([], dtype="period[D]")
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    assert isinstance(table.field("a").type, ArrowPeriodType)
    table = pa.table(
        [pa.chunked_array([], type=table.column(0).type)], schema=table.schema
    )

    result = table.to_pandas()
    assert isinstance(result["a"].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)


def test_arrow_table_roundtrip_without_metadata():
    arr = PeriodArray([1, 2, 3], dtype="period[h]")
    arr[1] = pd.NaT
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    # remove the metadata
    table = table.replace_schema_metadata()
    assert table.schema.metadata is None

    result = table.to_pandas()
    assert isinstance(result["a"].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\dtypes\test_generic.py ===
import re

import numpy as np
import pytest

from pandas.core.dtypes import generic as gt

import pandas as pd
import pandas._testing as tm


class TestABCClasses:
    tuples = [[1, 2, 2], ["red", "blue", "red"]]
    multi_index = pd.MultiIndex.from_arrays(tuples, names=("number", "color"))
    datetime_index = pd.to_datetime(["2000/1/1", "2010/1/1"])
    timedelta_index = pd.to_timedelta(np.arange(5), unit="s")
    period_index = pd.period_range("2000/1/1", "2010/1/1/", freq="M")
    categorical = pd.Categorical([1, 2, 3], categories=[2, 3, 1])
    categorical_df = pd.DataFrame({"values": [1, 2, 3]}, index=categorical)
    df = pd.DataFrame({"names": ["a", "b", "c"]}, index=multi_index)
    sparse_array = pd.arrays.SparseArray(np.random.default_rng(2).standard_normal(10))

    datetime_array = pd.core.arrays.DatetimeArray._from_sequence(datetime_index)
    timedelta_array = pd.core.arrays.TimedeltaArray._from_sequence(timedelta_index)

    abc_pairs = [
        ("ABCMultiIndex", multi_index),
        ("ABCDatetimeIndex", datetime_index),
        ("ABCRangeIndex", pd.RangeIndex(3)),
        ("ABCTimedeltaIndex", timedelta_index),
        ("ABCIntervalIndex", pd.interval_range(start=0, end=3)),
        (
            "ABCPeriodArray",
            pd.arrays.PeriodArray([2000, 2001, 2002], dtype="period[D]"),
        ),
        ("ABCNumpyExtensionArray", pd.arrays.NumpyExtensionArray(np.array([0, 1, 2]))),
        ("ABCPeriodIndex", period_index),
        ("ABCCategoricalIndex", categorical_df.index),
        ("ABCSeries", pd.Series([1, 2, 3])),
        ("ABCDataFrame", df),
        ("ABCCategorical", categorical),
        ("ABCDatetimeArray", datetime_array),
        ("ABCTimedeltaArray", timedelta_array),
    ]

    @pytest.mark.parametrize("abctype1, inst", abc_pairs)
    @pytest.mark.parametrize("abctype2, _", abc_pairs)
    def test_abc_pairs_instance_check(self, abctype1, abctype2, inst, _):
        # GH 38588, 46719
        if abctype1 == abctype2:
            assert isinstance(inst, getattr(gt, abctype2))
            assert not isinstance(type(inst), getattr(gt, abctype2))
        else:
            assert not isinstance(inst, getattr(gt, abctype2))

    @pytest.mark.parametrize("abctype1, inst", abc_pairs)
    @pytest.mark.parametrize("abctype2, _", abc_pairs)
    def test_abc_pairs_subclass_check(self, abctype1, abctype2, inst, _):
        # GH 38588, 46719
        if abctype1 == abctype2:
            assert issubclass(type(inst), getattr(gt, abctype2))

            with pytest.raises(
                TypeError, match=re.escape("issubclass() arg 1 must be a class")
            ):
                issubclass(inst, getattr(gt, abctype2))
        else:
            assert not issubclass(type(inst), getattr(gt, abctype2))

    abc_subclasses = {
        "ABCIndex": [
            abctype
            for abctype, _ in abc_pairs
            if "Index" in abctype and abctype != "ABCIndex"
        ],
        "ABCNDFrame": ["ABCSeries", "ABCDataFrame"],
        "ABCExtensionArray": [
            "ABCCategorical",
            "ABCDatetimeArray",
            "ABCPeriodArray",
            "ABCTimedeltaArray",
        ],
    }

    @pytest.mark.parametrize("parent, subs", abc_subclasses.items())
    @pytest.mark.parametrize("abctype, inst", abc_pairs)
    def test_abc_hierarchy(self, parent, subs, abctype, inst):
        # GH 38588
        if abctype in subs:
            assert isinstance(inst, getattr(gt, parent))
        else:
            assert not isinstance(inst, getattr(gt, parent))

    @pytest.mark.parametrize("abctype", [e for e in gt.__dict__ if e.startswith("ABC")])
    def test_abc_coverage(self, abctype):
        # GH 38588
        assert (
            abctype in (e for e, _ in self.abc_pairs) or abctype in self.abc_subclasses
        )


def test_setattr_warnings():
    # GH7175 - GOTCHA: You can't use dot notation to add a column...
    d = {
        "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
        "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
    }
    df = pd.DataFrame(d)

    with tm.assert_produces_warning(None):
        #  successfully add new column
        #  this should not raise a warning
        df["three"] = df.two + 1
        assert df.three.sum() > df.two.sum()

    with tm.assert_produces_warning(None):
        #  successfully modify column in place
        #  this should not raise a warning
        df.one += 1
        assert df.one.iloc[0] == 2

    with tm.assert_produces_warning(None):
        #  successfully add an attribute to a series
        #  this should not raise a warning
        df.two.not_an_index = [1, 2]

    with tm.assert_produces_warning(UserWarning):
        #  warn when setting column to nonexistent name
        df.four = df.two + 2
        assert df.four.sum() > df.two.sum()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\test_to_xarray.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm

pytest.importorskip("xarray")


class TestDataFrameToXArray:
    @pytest.fixture
    def df(self):
        return DataFrame(
            {
                "a": list("abcd"),
                "b": list(range(1, 5)),
                "c": np.arange(3, 7).astype("u1"),
                "d": np.arange(4.0, 8.0, dtype="float64"),
                "e": [True, False, True, False],
                "f": Categorical(list("abcd")),
                "g": date_range("20130101", periods=4),
                "h": date_range("20130101", periods=4, tz="US/Eastern"),
            }
        )

    def test_to_xarray_index_types(self, index_flat, df, using_infer_string):
        index = index_flat
        # MultiIndex is tested in test_to_xarray_with_multiindex
        if len(index) == 0:
            pytest.skip("Test doesn't make sense for empty index")

        from xarray import Dataset

        df.index = index[:4]
        df.index.name = "foo"
        df.columns.name = "bar"
        result = df.to_xarray()
        assert result.sizes["foo"] == 4
        assert len(result.coords) == 1
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, Dataset)

        # idempotency
        # datetimes w/tz are preserved
        # column names are lost
        expected = df.copy()
        expected["f"] = expected["f"].astype(
            object if not using_infer_string else "str"
        )
        expected.columns.name = None
        tm.assert_frame_equal(result.to_dataframe(), expected)

    def test_to_xarray_empty(self, df):
        from xarray import Dataset

        df.index.name = "foo"
        result = df[0:0].to_xarray()
        assert result.sizes["foo"] == 0
        assert isinstance(result, Dataset)

    def test_to_xarray_with_multiindex(self, df, using_infer_string):
        from xarray import Dataset

        # MultiIndex
        df.index = MultiIndex.from_product([["a"], range(4)], names=["one", "two"])
        result = df.to_xarray()
        assert result.sizes["one"] == 1
        assert result.sizes["two"] == 4
        assert len(result.coords) == 2
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ["one", "two"])
        assert isinstance(result, Dataset)

        result = result.to_dataframe()
        expected = df.copy()
        expected["f"] = expected["f"].astype(
            object if not using_infer_string else "str"
        )
        expected.columns.name = None
        tm.assert_frame_equal(result, expected)


class TestSeriesToXArray:
    def test_to_xarray_index_types(self, index_flat):
        index = index_flat
        # MultiIndex is tested in test_to_xarray_with_multiindex

        from xarray import DataArray

        ser = Series(range(len(index)), index=index, dtype="int64")
        ser.index.name = "foo"
        result = ser.to_xarray()
        repr(result)
        assert len(result) == len(index)
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, DataArray)

        # idempotency
        tm.assert_series_equal(result.to_series(), ser)

    def test_to_xarray_empty(self):
        from xarray import DataArray

        ser = Series([], dtype=object)
        ser.index.name = "foo"
        result = ser.to_xarray()
        assert len(result) == 0
        assert len(result.coords) == 1
        tm.assert_almost_equal(list(result.coords.keys()), ["foo"])
        assert isinstance(result, DataArray)

    def test_to_xarray_with_multiindex(self):
        from xarray import DataArray

        mi = MultiIndex.from_product([["a", "b"], range(3)], names=["one", "two"])
        ser = Series(range(6), dtype="int64", index=mi)
        result = ser.to_xarray()
        assert len(result) == 2
        tm.assert_almost_equal(list(result.coords.keys()), ["one", "two"])
        assert isinstance(result, DataArray)
        res = result.to_series()
        tm.assert_series_equal(res, ser)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_compression.py ===
from io import (
    BytesIO,
    StringIO,
)

import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


def test_compression_roundtrip(compression):
    df = pd.DataFrame(
        [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        index=["A", "B"],
        columns=["X", "Y", "Z"],
    )

    with tm.ensure_clean() as path:
        df.to_json(path, compression=compression)
        tm.assert_frame_equal(df, pd.read_json(path, compression=compression))

        # explicitly ensure file was compressed.
        with tm.decompress_file(path, compression) as fh:
            result = fh.read().decode("utf8")
            data = StringIO(result)
        tm.assert_frame_equal(df, pd.read_json(data))


def test_read_zipped_json(datapath):
    uncompressed_path = datapath("io", "json", "data", "tsframe_v012.json")
    uncompressed_df = pd.read_json(uncompressed_path)

    compressed_path = datapath("io", "json", "data", "tsframe_v012.json.zip")
    compressed_df = pd.read_json(compressed_path, compression="zip")

    tm.assert_frame_equal(uncompressed_df, compressed_df)


@td.skip_if_not_us_locale
@pytest.mark.single_cpu
def test_with_s3_url(compression, s3_public_bucket, s3so):
    # Bucket created in tests/io/conftest.py
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))

    with tm.ensure_clean() as path:
        df.to_json(path, compression=compression)
        with open(path, "rb") as f:
            s3_public_bucket.put_object(Key="test-1", Body=f)

    roundtripped_df = pd.read_json(
        f"s3://{s3_public_bucket.name}/test-1",
        compression=compression,
        storage_options=s3so,
    )
    tm.assert_frame_equal(df, roundtripped_df)


def test_lines_with_compression(compression):
    with tm.ensure_clean() as path:
        df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
        df.to_json(path, orient="records", lines=True, compression=compression)
        roundtripped_df = pd.read_json(path, lines=True, compression=compression)
        tm.assert_frame_equal(df, roundtripped_df)


def test_chunksize_with_compression(compression):
    with tm.ensure_clean() as path:
        df = pd.read_json(StringIO('{"a": ["foo", "bar", "baz"], "b": [4, 5, 6]}'))
        df.to_json(path, orient="records", lines=True, compression=compression)

        with pd.read_json(
            path, lines=True, chunksize=1, compression=compression
        ) as res:
            roundtripped_df = pd.concat(res)
        tm.assert_frame_equal(df, roundtripped_df)


def test_write_unsupported_compression_type():
    df = pd.read_json(StringIO('{"a": [1, 2, 3], "b": [4, 5, 6]}'))
    with tm.ensure_clean() as path:
        msg = "Unrecognized compression type: unsupported"
        with pytest.raises(ValueError, match=msg):
            df.to_json(path, compression="unsupported")


def test_read_unsupported_compression_type():
    with tm.ensure_clean() as path:
        msg = "Unrecognized compression type: unsupported"
        with pytest.raises(ValueError, match=msg):
            pd.read_json(path, compression="unsupported")


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("to_infer", [True, False])
@pytest.mark.parametrize("read_infer", [True, False])
def test_to_json_compression(
    compression_only, read_infer, to_infer, compression_to_extension, infer_string
):
    with pd.option_context("future.infer_string", infer_string):
        # see gh-15008
        compression = compression_only

        # We'll complete file extension subsequently.
        filename = "test."
        filename += compression_to_extension[compression]

        df = pd.DataFrame({"A": [1]})

        to_compression = "infer" if to_infer else compression
        read_compression = "infer" if read_infer else compression

        with tm.ensure_clean(filename) as path:
            df.to_json(path, compression=to_compression)
            result = pd.read_json(path, compression=read_compression)
            tm.assert_frame_equal(result, df)


def test_to_json_compression_mode(compression):
    # GH 39985 (read_json does not support user-provided binary files)
    expected = pd.DataFrame({"A": [1]})

    with BytesIO() as buffer:
        expected.to_json(buffer, compression=compression)
        # df = pd.read_json(buffer, compression=compression)
        # tm.assert_frame_equal(expected, df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_parameter.py ===
import pytest

import numpy as np

from . import util


class TestParameters(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [
        util.getpath("tests", "src", "parameter", "constant_real.f90"),
        util.getpath("tests", "src", "parameter", "constant_integer.f90"),
        util.getpath("tests", "src", "parameter", "constant_both.f90"),
        util.getpath("tests", "src", "parameter", "constant_compound.f90"),
        util.getpath("tests", "src", "parameter", "constant_non_compound.f90"),
        util.getpath("tests", "src", "parameter", "constant_array.f90"),
    ]

    @pytest.mark.slow
    def test_constant_real_single(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo_single, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo_single(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    @pytest.mark.slow
    def test_constant_real_double(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo_double, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_double(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    @pytest.mark.slow
    def test_constant_compound_int(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int32)[::2]
        pytest.raises(ValueError, self.module.foo_compound_int, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int32)
        self.module.foo_compound_int(x)
        assert np.allclose(x, [0 + 1 + 2 * 6, 1, 2])

    @pytest.mark.slow
    def test_constant_non_compound_int(self):
        # check values
        x = np.arange(4, dtype=np.int32)
        self.module.foo_non_compound_int(x)
        assert np.allclose(x, [0 + 1 + 2 + 3 * 4, 1, 2, 3])

    @pytest.mark.slow
    def test_constant_integer_int(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int32)[::2]
        pytest.raises(ValueError, self.module.foo_int, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int32)
        self.module.foo_int(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    @pytest.mark.slow
    def test_constant_integer_long(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.int64)[::2]
        pytest.raises(ValueError, self.module.foo_long, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.int64)
        self.module.foo_long(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    @pytest.mark.slow
    def test_constant_both(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo(x)
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    @pytest.mark.slow
    def test_constant_no(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo_no, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_no(x)
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    @pytest.mark.slow
    def test_constant_sum(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo_sum, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float64)
        self.module.foo_sum(x)
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    def test_constant_array(self):
        x = np.arange(3, dtype=np.float64)
        y = np.arange(5, dtype=np.float64)
        z = self.module.foo_array(x, y)
        assert np.allclose(x, [0.0, 1. / 10, 2. / 10])
        assert np.allclose(y, [0.0, 1. * 10, 2. * 10, 3. * 10, 4. * 10])
        assert np.allclose(z, 19.0)

    def test_constant_array_any_index(self):
        x = np.arange(6, dtype=np.float64)
        y = self.module.foo_array_any_index(x)
        assert np.allclose(y, x.reshape((2, 3), order='F'))

    def test_constant_array_delims(self):
        x = self.module.foo_array_delims()
        assert x == 9

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_numba.py ===
import numpy as np
import pytest

from pandas.compat import is_platform_arm
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm
from pandas.util.version import Version

pytestmark = [td.skip_if_no("numba"), pytest.mark.single_cpu, pytest.mark.skipif()]

numba = pytest.importorskip("numba")
pytestmark.append(
    pytest.mark.skipif(
        Version(numba.__version__) == Version("0.61") and is_platform_arm(),
        reason=f"Segfaults on ARM platforms with numba {numba.__version__}",
    )
)


@pytest.fixture(params=[0, 1])
def apply_axis(request):
    return request.param


def test_numba_vs_python_noop(float_frame, apply_axis):
    func = lambda x: x
    result = float_frame.apply(func, engine="numba", axis=apply_axis)
    expected = float_frame.apply(func, engine="python", axis=apply_axis)
    tm.assert_frame_equal(result, expected)


def test_numba_vs_python_string_index():
    # GH#56189
    df = DataFrame(
        1,
        index=Index(["a", "b"], dtype=pd.StringDtype(na_value=np.nan)),
        columns=Index(["x", "y"], dtype=pd.StringDtype(na_value=np.nan)),
    )
    func = lambda x: x
    result = df.apply(func, engine="numba", axis=0)
    expected = df.apply(func, engine="python", axis=0)
    tm.assert_frame_equal(
        result, expected, check_column_type=False, check_index_type=False
    )


def test_numba_vs_python_indexing():
    frame = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7.0, 8.0, 9.0]},
        index=Index(["A", "B", "C"]),
    )
    row_func = lambda x: x["c"]
    result = frame.apply(row_func, engine="numba", axis=1)
    expected = frame.apply(row_func, engine="python", axis=1)
    tm.assert_series_equal(result, expected)

    col_func = lambda x: x["A"]
    result = frame.apply(col_func, engine="numba", axis=0)
    expected = frame.apply(col_func, engine="python", axis=0)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "reduction",
    [lambda x: x.mean(), lambda x: x.min(), lambda x: x.max(), lambda x: x.sum()],
)
def test_numba_vs_python_reductions(reduction, apply_axis):
    df = DataFrame(np.ones((4, 4), dtype=np.float64))
    result = df.apply(reduction, engine="numba", axis=apply_axis)
    expected = df.apply(reduction, engine="python", axis=apply_axis)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("colnames", [[1, 2, 3], [1.0, 2.0, 3.0]])
def test_numba_numeric_colnames(colnames):
    # Check that numeric column names lower properly and can be indxed on
    df = DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64), columns=colnames
    )
    first_col = colnames[0]
    f = lambda x: x[first_col]  # Get the first column
    result = df.apply(f, engine="numba", axis=1)
    expected = df.apply(f, engine="python", axis=1)
    tm.assert_series_equal(result, expected)


def test_numba_parallel_unsupported(float_frame):
    f = lambda x: x
    with pytest.raises(
        NotImplementedError,
        match="Parallel apply is not supported when raw=False and engine='numba'",
    ):
        float_frame.apply(f, engine="numba", engine_kwargs={"parallel": True})


def test_numba_nonunique_unsupported(apply_axis):
    f = lambda x: x
    df = DataFrame({"a": [1, 2]}, index=Index(["a", "a"]))
    with pytest.raises(
        NotImplementedError,
        match="The index/columns must be unique when raw=False and engine='numba'",
    ):
        df.apply(f, engine="numba", axis=apply_axis)


def test_numba_unsupported_dtypes(apply_axis):
    pytest.importorskip("pyarrow")
    f = lambda x: x
    df = DataFrame({"a": [1, 2], "b": ["a", "b"], "c": [4, 5]})
    df["c"] = df["c"].astype("double[pyarrow]")

    with pytest.raises(
        ValueError,
        match="Column b must have a numeric dtype. Found 'object|str' instead",
    ):
        df.apply(f, engine="numba", axis=apply_axis)

    with pytest.raises(
        ValueError,
        match="Column c is backed by an extension array, "
        "which is not supported by the numba engine.",
    ):
        df["c"].to_frame().apply(f, engine="numba", axis=apply_axis)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_printing.py ===
# Note! This file is aimed specifically at pandas.io.formats.printing utility
# functions, not the general printing of pandas objects.
import string

import pandas._config.config as cf

from pandas.io.formats import printing


def test_adjoin():
    data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
    expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

    adjoined = printing.adjoin(2, *data)

    assert adjoined == expected


class TestPPrintThing:
    def test_repr_binary_type(self):
        letters = string.ascii_letters
        try:
            raw = bytes(letters, encoding=cf.get_option("display.encoding"))
        except TypeError:
            raw = bytes(letters)
        b = str(raw.decode("utf-8"))
        res = printing.pprint_thing(b, quote_strings=True)
        assert res == repr(b)
        res = printing.pprint_thing(b, quote_strings=False)
        assert res == b

    def test_repr_obeys_max_seq_limit(self):
        with cf.option_context("display.max_seq_items", 2000):
            assert len(printing.pprint_thing(list(range(1000)))) > 1000

        with cf.option_context("display.max_seq_items", 5):
            assert len(printing.pprint_thing(list(range(1000)))) < 100

        with cf.option_context("display.max_seq_items", 1):
            assert len(printing.pprint_thing(list(range(1000)))) < 9

    def test_repr_set(self):
        assert printing.pprint_thing({1}) == "{1}"


class TestFormatBase:
    def test_adjoin(self):
        data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
        expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

        adjoined = printing.adjoin(2, *data)

        assert adjoined == expected

    def test_adjoin_unicode(self):
        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "hhh", "いいい"]]
        expected = "あ  dd  ggg\nb  ええ  hhh\nc  ff  いいい"
        adjoined = printing.adjoin(2, *data)
        assert adjoined == expected

        adj = printing._EastAsianTextAdjustment()

        expected = """あ  dd    ggg
b   ええ  hhh
c   ff    いいい"""

        adjoined = adj.adjoin(2, *data)
        assert adjoined == expected
        cols = adjoined.split("\n")
        assert adj.len(cols[0]) == 13
        assert adj.len(cols[1]) == 13
        assert adj.len(cols[2]) == 16

        expected = """あ       dd         ggg
b        ええ       hhh
c        ff         いいい"""

        adjoined = adj.adjoin(7, *data)
        assert adjoined == expected
        cols = adjoined.split("\n")
        assert adj.len(cols[0]) == 23
        assert adj.len(cols[1]) == 23
        assert adj.len(cols[2]) == 26

    def test_justify(self):
        adj = printing._EastAsianTextAdjustment()

        def just(x, *args, **kwargs):
            # wrapper to test single str
            return adj.justify([x], *args, **kwargs)[0]

        assert just("abc", 5, mode="left") == "abc  "
        assert just("abc", 5, mode="center") == " abc "
        assert just("abc", 5, mode="right") == "  abc"
        assert just("abc", 5, mode="left") == "abc  "
        assert just("abc", 5, mode="center") == " abc "
        assert just("abc", 5, mode="right") == "  abc"

        assert just("パンダ", 5, mode="left") == "パンダ"
        assert just("パンダ", 5, mode="center") == "パンダ"
        assert just("パンダ", 5, mode="right") == "パンダ"

        assert just("パンダ", 10, mode="left") == "パンダ    "
        assert just("パンダ", 10, mode="center") == "  パンダ  "
        assert just("パンダ", 10, mode="right") == "    パンダ"

    def test_east_asian_len(self):
        adj = printing._EastAsianTextAdjustment()

        assert adj.len("abc") == 3
        assert adj.len("abc") == 3

        assert adj.len("パンダ") == 6
        assert adj.len("ﾊﾟﾝﾀﾞ") == 5
        assert adj.len("パンダpanda") == 11
        assert adj.len("ﾊﾟﾝﾀﾞpanda") == 10

    def test_ambiguous_width(self):
        adj = printing._EastAsianTextAdjustment()
        assert adj.len("¡¡ab") == 4

        with cf.option_context("display.unicode.ambiguous_as_wide", True):
            adj = printing._EastAsianTextAdjustment()
            assert adj.len("¡¡ab") == 6

        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "¡¡ab", "いいい"]]
        expected = "あ  dd    ggg \nb   ええ  ¡¡ab\nc   ff    いいい"
        adjoined = adj.adjoin(2, *data)
        assert adjoined == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\accessors\test_list_accessor.py ===
import re

import pytest

from pandas import (
    ArrowDtype,
    Series,
)
import pandas._testing as tm

pa = pytest.importorskip("pyarrow")

from pandas.compat import pa_version_under11p0


@pytest.mark.parametrize(
    "list_dtype",
    (
        pa.list_(pa.int64()),
        pa.list_(pa.int64(), list_size=3),
        pa.large_list(pa.int64()),
    ),
)
def test_list_getitem(list_dtype):
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(list_dtype),
    )
    actual = ser.list[1]
    expected = Series([2, None, None], dtype="int64[pyarrow]")
    tm.assert_series_equal(actual, expected)


def test_list_getitem_slice():
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    if pa_version_under11p0:
        with pytest.raises(
            NotImplementedError, match="List slice not supported by pyarrow "
        ):
            ser.list[1:None:None]
    else:
        actual = ser.list[1:None:None]
        expected = Series(
            [[2, 3], [None, 5], None], dtype=ArrowDtype(pa.list_(pa.int64()))
        )
        tm.assert_series_equal(actual, expected)


def test_list_len():
    ser = Series(
        [[1, 2, 3], [4, None], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    actual = ser.list.len()
    expected = Series([3, 2, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(actual, expected)


def test_list_flatten():
    ser = Series(
        [[1, 2, 3], [4, None], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    actual = ser.list.flatten()
    expected = Series([1, 2, 3, 4, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(actual, expected)


def test_list_getitem_slice_invalid():
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    if pa_version_under11p0:
        with pytest.raises(
            NotImplementedError, match="List slice not supported by pyarrow "
        ):
            ser.list[1:None:0]
    else:
        with pytest.raises(pa.lib.ArrowInvalid, match=re.escape("`step` must be >= 1")):
            ser.list[1:None:0]


def test_list_accessor_non_list_dtype():
    ser = Series(
        [1, 2, 4],
        dtype=ArrowDtype(pa.int64()),
    )
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "Can only use the '.list' accessor with 'list[pyarrow]' dtype, "
            "not int64[pyarrow]."
        ),
    ):
        ser.list[1:None:0]


@pytest.mark.parametrize(
    "list_dtype",
    (
        pa.list_(pa.int64()),
        pa.list_(pa.int64(), list_size=3),
        pa.large_list(pa.int64()),
    ),
)
def test_list_getitem_invalid_index(list_dtype):
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(list_dtype),
    )
    with pytest.raises(pa.lib.ArrowInvalid, match="Index -1 is out of bounds"):
        ser.list[-1]
    with pytest.raises(pa.lib.ArrowInvalid, match="Index 5 is out of bounds"):
        ser.list[5]
    with pytest.raises(ValueError, match="key must be an int or slice, got str"):
        ser.list["abc"]


def test_list_accessor_not_iterable():
    ser = Series(
        [[1, 2, 3], [4, None], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    with pytest.raises(TypeError, match="'ListAccessor' object is not iterable"):
        iter(ser.list)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_singularity_functions.py ===
from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Float, I, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.series.order import O


from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises

x, y, a, n = symbols('x y a n')


def test_fdiff():
    assert SingularityFunction(x, 4, 5).fdiff() == 5*SingularityFunction(x, 4, 4)
    assert SingularityFunction(x, 4, -1).fdiff() == SingularityFunction(x, 4, -2)
    assert SingularityFunction(x, 4, -2).fdiff() == SingularityFunction(x, 4, -3)
    assert SingularityFunction(x, 4, -3).fdiff() == SingularityFunction(x, 4, -4)
    assert SingularityFunction(x, 4, 0).fdiff() == SingularityFunction(x, 4, -1)

    assert SingularityFunction(y, 6, 2).diff(y) == 2*SingularityFunction(y, 6, 1)
    assert SingularityFunction(y, -4, -1).diff(y) == SingularityFunction(y, -4, -2)
    assert SingularityFunction(y, 4, 0).diff(y) == SingularityFunction(y, 4, -1)
    assert SingularityFunction(y, 4, 0).diff(y, 2) == SingularityFunction(y, 4, -2)

    n = Symbol('n', positive=True)
    assert SingularityFunction(x, a, n).fdiff() == n*SingularityFunction(x, a, n - 1)
    assert SingularityFunction(y, a, n).diff(y) == n*SingularityFunction(y, a, n - 1)

    expr_in = 4*SingularityFunction(x, a, n) + 3*SingularityFunction(x, a, -1) + -10*SingularityFunction(x, a, 0)
    expr_out = n*4*SingularityFunction(x, a, n - 1) + 3*SingularityFunction(x, a, -2) - 10*SingularityFunction(x, a, -1)
    assert diff(expr_in, x) == expr_out

    assert SingularityFunction(x, -10, 5).diff(evaluate=False) == (
        Derivative(SingularityFunction(x, -10, 5), x))

    raises(ArgumentIndexError, lambda: SingularityFunction(x, 4, 5).fdiff(2))


def test_eval():
    assert SingularityFunction(x, a, n).func == SingularityFunction
    assert unchanged(SingularityFunction, x, 5, n)
    assert SingularityFunction(5, 3, 2) == 4
    assert SingularityFunction(3, 5, 1) == 0
    assert SingularityFunction(3, 3, 0) == 1
    assert SingularityFunction(3, 3, 1) == 0
    assert SingularityFunction(Symbol('z', zero=True), 0, 1) == 0  # like sin(z) == 0
    assert SingularityFunction(4, 4, -1) is oo
    assert SingularityFunction(4, 2, -1) == 0
    assert SingularityFunction(4, 7, -1) == 0
    assert SingularityFunction(5, 6, -2) == 0
    assert SingularityFunction(4, 2, -2) == 0
    assert SingularityFunction(4, 4, -2) is oo
    assert SingularityFunction(4, 2, -3) == 0
    assert SingularityFunction(8, 8, -3) is oo
    assert SingularityFunction(4, 2, -4) == 0
    assert SingularityFunction(8, 8, -4) is oo
    assert (SingularityFunction(6.1, 4, 5)).evalf(5) == Float('40.841', '5')
    assert SingularityFunction(6.1, pi, 2) == (-pi + 6.1)**2
    assert SingularityFunction(x, a, nan) is nan
    assert SingularityFunction(x, nan, 1) is nan
    assert SingularityFunction(nan, a, n) is nan

    raises(ValueError, lambda: SingularityFunction(x, a, I))
    raises(ValueError, lambda: SingularityFunction(2*I, I, n))
    raises(ValueError, lambda: SingularityFunction(x, a, -5))


def test_leading_term():
    l = Symbol('l', positive=True)
    assert SingularityFunction(x, 3, 2).as_leading_term(x) == 0
    assert SingularityFunction(x, -2, 1).as_leading_term(x) == 2
    assert SingularityFunction(x, 0, 0).as_leading_term(x) == 1
    assert SingularityFunction(x, 0, 0).as_leading_term(x, cdir=-1) == 0
    assert SingularityFunction(x, 0, -1).as_leading_term(x) == 0
    assert SingularityFunction(x, 0, -2).as_leading_term(x) == 0
    assert SingularityFunction(x, 0, -3).as_leading_term(x) == 0
    assert SingularityFunction(x, 0, -4).as_leading_term(x) == 0
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).as_leading_term(x) == -x/2


def test_series():
    l = Symbol('l', positive=True)
    assert SingularityFunction(x, -3, 2).series(x) == x**2 + 6*x + 9
    assert SingularityFunction(x, -2, 1).series(x) == x + 2
    assert SingularityFunction(x, 0, 0).series(x) == 1
    assert SingularityFunction(x, 0, 0).series(x, dir='-') == 0
    assert SingularityFunction(x, 0, -1).series(x) == 0
    assert SingularityFunction(x, 0, -2).series(x) == 0
    assert SingularityFunction(x, 0, -3).series(x) == 0
    assert SingularityFunction(x, 0, -4).series(x) == 0
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).nseries(x) == -x/2 + O(x**6)


def test_rewrite():
    assert SingularityFunction(x, 4, 5).rewrite(Piecewise) == (
        Piecewise(((x - 4)**5, x - 4 >= 0), (0, True)))
    assert SingularityFunction(x, -10, 0).rewrite(Piecewise) == (
        Piecewise((1, x + 10 >= 0), (0, True)))
    assert SingularityFunction(x, 2, -1).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x - 2, 0)), (0, True)))
    assert SingularityFunction(x, 0, -2).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x, 0)), (0, True)))

    n = Symbol('n', nonnegative=True)
    p = SingularityFunction(x, a, n).rewrite(Piecewise)
    assert p == (
        Piecewise(((x - a)**n, x - a >= 0), (0, True)))
    assert p.subs(x, a).subs(n, 0) == 1

    expr_in = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    expr_out = (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    assert expr_in.rewrite(Heaviside) == expr_out
    assert expr_in.rewrite(DiracDelta) == expr_out
    assert expr_in.rewrite('HeavisideDiracDelta') == expr_out

    expr_in = SingularityFunction(x, a, n) + SingularityFunction(x, a, -1) - SingularityFunction(x, a, -2)
    expr_out = (x - a)**n*Heaviside(x - a, 1) + DiracDelta(x - a) + DiracDelta(a - x, 1)
    assert expr_in.rewrite(Heaviside) == expr_out
    assert expr_in.rewrite(DiracDelta) == expr_out
    assert expr_in.rewrite('HeavisideDiracDelta') == expr_out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_decorator.py ===
from functools import wraps

from sympy.utilities.decorator import threaded, xthreaded, memoize_property, deprecated
from sympy.testing.pytest import warns_deprecated_sympy

from sympy.core.basic import Basic
from sympy.core.relational import Eq
from sympy.matrices.dense import Matrix

from sympy.abc import x, y


def test_threaded():
    @threaded
    def function(expr, *args):
        return 2*expr + sum(args)

    assert function(Matrix([[x, y], [1, x]]), 1, 2) == \
        Matrix([[2*x + 3, 2*y + 3], [5, 2*x + 3]])

    assert function(Eq(x, y), 1, 2) == Eq(2*x + 3, 2*y + 3)

    assert function([x, y], 1, 2) == [2*x + 3, 2*y + 3]
    assert function((x, y), 1, 2) == (2*x + 3, 2*y + 3)

    assert function({x, y}, 1, 2) == {2*x + 3, 2*y + 3}

    @threaded
    def function(expr, n):
        return expr**n

    assert function(x + y, 2) == x**2 + y**2
    assert function(x, 2) == x**2


def test_xthreaded():
    @xthreaded
    def function(expr, n):
        return expr**n

    assert function(x + y, 2) == (x + y)**2


def test_wraps():
    def my_func(x):
        """My function. """

    my_func.is_my_func = True

    new_my_func = threaded(my_func)
    new_my_func = wraps(my_func)(new_my_func)

    assert new_my_func.__name__ == 'my_func'
    assert new_my_func.__doc__ == 'My function. '
    assert hasattr(new_my_func, 'is_my_func')
    assert new_my_func.is_my_func is True


def test_memoize_property():
    class TestMemoize(Basic):
        @memoize_property
        def prop(self):
            return Basic()

    member = TestMemoize()
    obj1 = member.prop
    obj2 = member.prop
    assert obj1 is obj2

def test_deprecated():
    @deprecated('deprecated_function is deprecated',
                deprecated_since_version='1.10',
                # This is the target at the top of the file, which will never
                # go away.
                active_deprecations_target='active-deprecations')
    def deprecated_function(x):
        return x

    with warns_deprecated_sympy():
        assert deprecated_function(1) == 1

    @deprecated('deprecated_class is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    class deprecated_class:
        pass

    with warns_deprecated_sympy():
        assert isinstance(deprecated_class(), deprecated_class)

    # Ensure the class decorator works even when the class never returns
    # itself
    @deprecated('deprecated_class_new is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    class deprecated_class_new:
        def __new__(cls, arg):
            return arg

    with warns_deprecated_sympy():
        assert deprecated_class_new(1) == 1

    @deprecated('deprecated_class_init is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    class deprecated_class_init:
        def __init__(self, arg):
            self.arg = 1

    with warns_deprecated_sympy():
        assert deprecated_class_init(1).arg == 1

    @deprecated('deprecated_class_new_init is deprecated',
                deprecated_since_version='1.10',
                active_deprecations_target='active-deprecations')
    class deprecated_class_new_init:
        def __new__(cls, arg):
            if arg == 0:
                return arg
            return object.__new__(cls)

        def __init__(self, arg):
            self.arg = 1

    with warns_deprecated_sympy():
        assert deprecated_class_new_init(0) == 0

    with warns_deprecated_sympy():
        assert deprecated_class_new_init(1).arg == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_throw.py ===
import sys


from greenlet import greenlet
from . import TestCase

def switch(*args):
    return greenlet.getcurrent().parent.switch(*args)


class ThrowTests(TestCase):
    def test_class(self):
        def f():
            try:
                switch("ok")
            except RuntimeError:
                switch("ok")
                return
            switch("fail")
        g = greenlet(f)
        res = g.switch()
        self.assertEqual(res, "ok")
        res = g.throw(RuntimeError)
        self.assertEqual(res, "ok")

    def test_val(self):
        def f():
            try:
                switch("ok")
            except RuntimeError:
                val = sys.exc_info()[1]
                if str(val) == "ciao":
                    switch("ok")
                    return
            switch("fail")

        g = greenlet(f)
        res = g.switch()
        self.assertEqual(res, "ok")
        res = g.throw(RuntimeError("ciao"))
        self.assertEqual(res, "ok")

        g = greenlet(f)
        res = g.switch()
        self.assertEqual(res, "ok")
        res = g.throw(RuntimeError, "ciao")
        self.assertEqual(res, "ok")

    def test_kill(self):
        def f():
            switch("ok")
            switch("fail")
        g = greenlet(f)
        res = g.switch()
        self.assertEqual(res, "ok")
        res = g.throw()
        self.assertTrue(isinstance(res, greenlet.GreenletExit))
        self.assertTrue(g.dead)
        res = g.throw()    # immediately eaten by the already-dead greenlet
        self.assertTrue(isinstance(res, greenlet.GreenletExit))

    def test_throw_goes_to_original_parent(self):
        main = greenlet.getcurrent()

        def f1():
            try:
                main.switch("f1 ready to catch")
            except IndexError:
                return "caught"
            return "normal exit"

        def f2():
            main.switch("from f2")

        g1 = greenlet(f1)
        g2 = greenlet(f2, parent=g1)
        with self.assertRaises(IndexError):
            g2.throw(IndexError)
        self.assertTrue(g2.dead)
        self.assertTrue(g1.dead)

        g1 = greenlet(f1)
        g2 = greenlet(f2, parent=g1)
        res = g1.switch()
        self.assertEqual(res, "f1 ready to catch")
        res = g2.throw(IndexError)
        self.assertEqual(res, "caught")
        self.assertTrue(g2.dead)
        self.assertTrue(g1.dead)

        g1 = greenlet(f1)
        g2 = greenlet(f2, parent=g1)
        res = g1.switch()
        self.assertEqual(res, "f1 ready to catch")
        res = g2.switch()
        self.assertEqual(res, "from f2")
        res = g2.throw(IndexError)
        self.assertEqual(res, "caught")
        self.assertTrue(g2.dead)
        self.assertTrue(g1.dead)

    def test_non_traceback_param(self):
        with self.assertRaises(TypeError) as exc:
            greenlet.getcurrent().throw(
                Exception,
                Exception(),
                self
            )
        self.assertEqual(str(exc.exception),
                         "throw() third argument must be a traceback object")

    def test_instance_of_wrong_type(self):
        with self.assertRaises(TypeError) as exc:
            greenlet.getcurrent().throw(
                Exception(),
                BaseException()
            )

        self.assertEqual(str(exc.exception),
                         "instance exception may not have a separate value")

    def test_not_throwable(self):
        with self.assertRaises(TypeError) as exc:
            greenlet.getcurrent().throw(
                "abc"
            )
        self.assertEqual(str(exc.exception),
                         "exceptions must be classes, or instances, not str")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_polyutils.py ===
"""Tests for polyutils module.

"""
import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)


class TestMisc:

    def test_trimseq(self):
        tgt = [1]
        for num_trailing_zeros in range(5):
            res = pu.trimseq([1] + [0] * num_trailing_zeros)
            assert_equal(res, tgt)

    def test_trimseq_empty_input(self):
        for empty_seq in [[], np.array([], dtype=np.int32)]:
            assert_equal(pu.trimseq(empty_seq), empty_seq)

    def test_as_series(self):
        # check exceptions
        assert_raises(ValueError, pu.as_series, [[]])
        assert_raises(ValueError, pu.as_series, [[[1, 2]]])
        assert_raises(ValueError, pu.as_series, [[1], ['a']])
        # check common types
        types = ['i', 'd', 'O']
        for i in range(len(types)):
            for j in range(i):
                ci = np.ones(1, types[i])
                cj = np.ones(1, types[j])
                [resi, resj] = pu.as_series([ci, cj])
                assert_(resi.dtype.char == resj.dtype.char)
                assert_(resj.dtype.char == types[i])

    def test_trimcoef(self):
        coef = [2, -1, 1, 0]
        # Test exceptions
        assert_raises(ValueError, pu.trimcoef, coef, -1)
        # Test results
        assert_equal(pu.trimcoef(coef), coef[:-1])
        assert_equal(pu.trimcoef(coef, 1), coef[:-3])
        assert_equal(pu.trimcoef(coef, 2), [0])

    def test_vander_nd_exception(self):
        # n_dims != len(points)
        assert_raises(ValueError, pu._vander_nd, (), (1, 2, 3), [90])
        # n_dims != len(degrees)
        assert_raises(ValueError, pu._vander_nd, (), (), [90.65])
        # n_dims == 0
        assert_raises(ValueError, pu._vander_nd, (), (), [])

    def test_div_zerodiv(self):
        # c2[-1] == 0
        assert_raises(ZeroDivisionError, pu._div, pu._div, (1, 2, 3), [0])

    def test_pow_too_large(self):
        # power > maxpower
        assert_raises(ValueError, pu._pow, (), [1, 2, 3], 5, 4)

class TestDomain:

    def test_getdomain(self):
        # test for real values
        x = [1, 10, 3, -1]
        tgt = [-1, 10]
        res = pu.getdomain(x)
        assert_almost_equal(res, tgt)

        # test for complex values
        x = [1 + 1j, 1 - 1j, 0, 2]
        tgt = [-1j, 2 + 1j]
        res = pu.getdomain(x)
        assert_almost_equal(res, tgt)

    def test_mapdomain(self):
        # test for real values
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = dom2
        res = pu.mapdomain(dom1, dom1, dom2)
        assert_almost_equal(res, tgt)

        # test for complex values
        dom1 = [0 - 1j, 2 + 1j]
        dom2 = [-2, 2]
        tgt = dom2
        x = dom1
        res = pu.mapdomain(x, dom1, dom2)
        assert_almost_equal(res, tgt)

        # test for multidimensional arrays
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = np.array([dom2, dom2])
        x = np.array([dom1, dom1])
        res = pu.mapdomain(x, dom1, dom2)
        assert_almost_equal(res, tgt)

        # test that subtypes are preserved.
        class MyNDArray(np.ndarray):
            pass

        dom1 = [0, 4]
        dom2 = [1, 3]
        x = np.array([dom1, dom1]).view(MyNDArray)
        res = pu.mapdomain(x, dom1, dom2)
        assert_(isinstance(res, MyNDArray))

    def test_mapparms(self):
        # test for real values
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = [1, .5]
        res = pu. mapparms(dom1, dom2)
        assert_almost_equal(res, tgt)

        # test for complex values
        dom1 = [0 - 1j, 2 + 1j]
        dom2 = [-2, 2]
        tgt = [-1 + 1j, 1 - 1j]
        res = pu.mapparms(dom1, dom2)
        assert_almost_equal(res, tgt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_sorting.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    Index,
)
import pandas._testing as tm


class TestCategoricalSort:
    def test_argsort(self):
        c = Categorical([5, 3, 1, 4, 2], ordered=True)

        expected = np.array([2, 4, 1, 3, 0])
        tm.assert_numpy_array_equal(
            c.argsort(ascending=True), expected, check_dtype=False
        )

        expected = expected[::-1]
        tm.assert_numpy_array_equal(
            c.argsort(ascending=False), expected, check_dtype=False
        )

    def test_numpy_argsort(self):
        c = Categorical([5, 3, 1, 4, 2], ordered=True)

        expected = np.array([2, 4, 1, 3, 0])
        tm.assert_numpy_array_equal(np.argsort(c), expected, check_dtype=False)

        tm.assert_numpy_array_equal(
            np.argsort(c, kind="mergesort"), expected, check_dtype=False
        )

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(c, axis=0)

        msg = "the 'order' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(c, order="C")

    def test_sort_values(self):
        # unordered cats are sortable
        cat = Categorical(["a", "b", "b", "a"], ordered=False)
        cat.sort_values()

        cat = Categorical(["a", "c", "b", "d"], ordered=True)

        # sort_values
        res = cat.sort_values()
        exp = np.array(["a", "b", "c", "d"], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        cat = Categorical(
            ["a", "c", "b", "d"], categories=["a", "b", "c", "d"], ordered=True
        )
        res = cat.sort_values()
        exp = np.array(["a", "b", "c", "d"], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        res = cat.sort_values(ascending=False)
        exp = np.array(["d", "c", "b", "a"], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        # sort (inplace order)
        cat1 = cat.copy()
        orig_codes = cat1._codes
        cat1.sort_values(inplace=True)
        assert cat1._codes is orig_codes
        exp = np.array(["a", "b", "c", "d"], dtype=object)
        tm.assert_numpy_array_equal(cat1.__array__(), exp)
        tm.assert_index_equal(res.categories, cat.categories)

        # reverse
        cat = Categorical(["a", "c", "c", "b", "d"], ordered=True)
        res = cat.sort_values(ascending=False)
        exp_val = np.array(["d", "c", "c", "b", "a"], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)

    def test_sort_values_na_position(self):
        # see gh-12882
        cat = Categorical([5, 2, np.nan, 2, np.nan], ordered=True)
        exp_categories = Index([2, 5])

        exp = np.array([2.0, 2.0, 5.0, np.nan, np.nan])
        res = cat.sort_values()  # default arguments
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([np.nan, np.nan, 2.0, 2.0, 5.0])
        res = cat.sort_values(ascending=True, na_position="first")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([np.nan, np.nan, 5.0, 2.0, 2.0])
        res = cat.sort_values(ascending=False, na_position="first")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([2.0, 2.0, 5.0, np.nan, np.nan])
        res = cat.sort_values(ascending=True, na_position="last")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        exp = np.array([5.0, 2.0, 2.0, np.nan, np.nan])
        res = cat.sort_values(ascending=False, na_position="last")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        cat = Categorical(["a", "c", "b", "d", np.nan], ordered=True)
        res = cat.sort_values(ascending=False, na_position="last")
        exp_val = np.array(["d", "c", "b", "a", np.nan], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)

        cat = Categorical(["a", "c", "b", "d", np.nan], ordered=True)
        res = cat.sort_values(ascending=False, na_position="first")
        exp_val = np.array([np.nan, "d", "c", "b", "a"], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_pct_change.py ===
import numpy as np
import pytest

from pandas import (
    Series,
    date_range,
)
import pandas._testing as tm


class TestSeriesPctChange:
    def test_pct_change(self, datetime_series):
        msg = (
            "The 'fill_method' keyword being not None and the 'limit' keyword in "
            "Series.pct_change are deprecated"
        )

        rs = datetime_series.pct_change(fill_method=None)
        tm.assert_series_equal(rs, datetime_series / datetime_series.shift(1) - 1)

        rs = datetime_series.pct_change(2)
        filled = datetime_series.ffill()
        tm.assert_series_equal(rs, filled / filled.shift(2) - 1)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs = datetime_series.pct_change(fill_method="bfill", limit=1)
        filled = datetime_series.bfill(limit=1)
        tm.assert_series_equal(rs, filled / filled.shift(1) - 1)

        rs = datetime_series.pct_change(freq="5D")
        filled = datetime_series.ffill()
        tm.assert_series_equal(
            rs, (filled / filled.shift(freq="5D") - 1).reindex_like(filled)
        )

    def test_pct_change_with_duplicate_axis(self):
        # GH#28664
        common_idx = date_range("2019-11-14", periods=5, freq="D")
        result = Series(range(5), common_idx).pct_change(freq="B")

        # the reason that the expected should be like this is documented at PR 28681
        expected = Series([np.nan, np.inf, np.nan, np.nan, 3.0], common_idx)

        tm.assert_series_equal(result, expected)

    def test_pct_change_shift_over_nas(self):
        s = Series([1.0, 1.5, np.nan, 2.5, 3.0])

        msg = "The default fill_method='pad' in Series.pct_change is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            chg = s.pct_change()

        expected = Series([np.nan, 0.5, 0.0, 2.5 / 1.5 - 1, 0.2])
        tm.assert_series_equal(chg, expected)

    @pytest.mark.parametrize(
        "freq, periods, fill_method, limit",
        [
            ("5B", 5, None, None),
            ("3B", 3, None, None),
            ("3B", 3, "bfill", None),
            ("7B", 7, "pad", 1),
            ("7B", 7, "bfill", 3),
            ("14B", 14, None, None),
        ],
    )
    def test_pct_change_periods_freq(
        self, freq, periods, fill_method, limit, datetime_series
    ):
        msg = (
            "The 'fill_method' keyword being not None and the 'limit' keyword in "
            "Series.pct_change are deprecated"
        )

        # GH#7292
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_freq = datetime_series.pct_change(
                freq=freq, fill_method=fill_method, limit=limit
            )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_periods = datetime_series.pct_change(
                periods, fill_method=fill_method, limit=limit
            )
        tm.assert_series_equal(rs_freq, rs_periods)

        empty_ts = Series(index=datetime_series.index, dtype=object)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_freq = empty_ts.pct_change(
                freq=freq, fill_method=fill_method, limit=limit
            )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rs_periods = empty_ts.pct_change(
                periods, fill_method=fill_method, limit=limit
            )
        tm.assert_series_equal(rs_freq, rs_periods)


@pytest.mark.parametrize("fill_method", ["pad", "ffill", None])
def test_pct_change_with_duplicated_indices(fill_method):
    # GH30463
    s = Series([np.nan, 1, 2, 3, 9, 18], index=["a", "b"] * 3)

    warn = None if fill_method is None else FutureWarning
    msg = (
        "The 'fill_method' keyword being not None and the 'limit' keyword in "
        "Series.pct_change are deprecated"
    )
    with tm.assert_produces_warning(warn, match=msg):
        result = s.pct_change(fill_method=fill_method)

    expected = Series([np.nan, np.nan, 1.0, 0.5, 2.0, 1.0], index=["a", "b"] * 3)
    tm.assert_series_equal(result, expected)


def test_pct_change_no_warning_na_beginning():
    # GH#54981
    ser = Series([None, None, 1, 2, 3])
    result = ser.pct_change()
    expected = Series([np.nan, np.nan, np.nan, 1, 0.5])
    tm.assert_series_equal(result, expected)


def test_pct_change_empty():
    # GH 57056
    ser = Series([], dtype="float64")
    expected = ser.copy()
    result = ser.pct_change(periods=0)
    tm.assert_series_equal(expected, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_kane5.py ===
from sympy import (zeros, Matrix, symbols, lambdify, sqrt, pi,
                                simplify)
from sympy.physics.mechanics import (dynamicsymbols, cross, inertia, RigidBody,
                                     ReferenceFrame, KanesMethod)


def _create_rolling_disc():
    # Define symbols and coordinates
    t = dynamicsymbols._t
    q1, q2, q3, q4, q5, u1, u2, u3, u4, u5 = dynamicsymbols('q1:6 u1:6')
    g, r, m = symbols('g r m')
    # Define bodies and frames
    ground = RigidBody('ground')
    disc = RigidBody('disk', mass=m)
    disc.inertia = (m * r ** 2 / 4 * inertia(disc.frame, 1, 2, 1),
                    disc.masscenter)
    ground.masscenter.set_vel(ground.frame, 0)
    disc.masscenter.set_vel(disc.frame, 0)
    int_frame = ReferenceFrame('int_frame')
    # Orient frames
    int_frame.orient_body_fixed(ground.frame, (q1, q2, 0), 'zxy')
    disc.frame.orient_axis(int_frame, int_frame.y, q3)
    g_w_d = disc.frame.ang_vel_in(ground.frame)
    disc.frame.set_ang_vel(ground.frame,
                           u1 * disc.x + u2 * disc.y + u3 * disc.z)
    # Define points
    cp = ground.masscenter.locatenew('contact_point',
                                     q4 * ground.x + q5 * ground.y)
    cp.set_vel(ground.frame, u4 * ground.x + u5 * ground.y)
    disc.masscenter.set_pos(cp, r * int_frame.z)
    disc.masscenter.set_vel(ground.frame, cross(
        disc.frame.ang_vel_in(ground.frame), disc.masscenter.pos_from(cp)))
    # Define kinematic differential equations
    kdes = [g_w_d.dot(disc.x) - u1, g_w_d.dot(disc.y) - u2,
            g_w_d.dot(disc.z) - u3, q4.diff(t) - u4, q5.diff(t) - u5]
    # Define nonholonomic constraints
    v0 = cp.vel(ground.frame) + cross(
        disc.frame.ang_vel_in(int_frame), cp.pos_from(disc.masscenter))
    fnh = [v0.dot(ground.x), v0.dot(ground.y)]
    # Define loads
    loads = [(disc.masscenter, -disc.mass * g * ground.z)]
    bodies = [disc]
    return {
        'frame': ground.frame,
        'q_ind': [q1, q2, q3, q4, q5],
        'u_ind': [u1, u2, u3],
        'u_dep': [u4, u5],
        'kdes': kdes,
        'fnh': fnh,
        'bodies': bodies,
        'loads': loads
    }


def _verify_rolling_disc_numerically(kane, all_zero=False):
    q, u, p = dynamicsymbols('q1:6'), dynamicsymbols('u1:6'), symbols('g r m')
    eval_sys = lambdify((q, u, p), (kane.mass_matrix_full, kane.forcing_full),
                        cse=True)
    solve_sys = lambda q, u, p: Matrix.LUsolve(
        *(Matrix(mat) for mat in eval_sys(q, u, p)))
    solve_u_dep = lambdify((q, u[:3], p), kane._Ars * Matrix(u[:3]), cse=True)
    eps = 1e-10
    p_vals = (9.81, 0.26, 3.43)
    # First numeric test
    q_vals = (0.3, 0.1, 1.97, -0.35, 2.27)
    u_vals = [-0.2, 1.3, 0.15]
    u_vals.extend(solve_u_dep(q_vals, u_vals, p_vals)[:2, 0])
    expected = Matrix([
        0.126603940595934, 0.215942571601660, 1.28736069604936,
        0.319764288376543, 0.0989146857254898, -0.925848952664489,
        -0.0181350656532944, 2.91695398184589, -0.00992793421754526,
        0.0412861634829171])
    assert all(abs(x) < eps for x in
               (solve_sys(q_vals, u_vals, p_vals) - expected))
    # Second numeric test
    q_vals = (3.97, -0.28, 8.2, -0.35, 2.27)
    u_vals = [-0.25, -2.2, 0.62]
    u_vals.extend(solve_u_dep(q_vals, u_vals, p_vals)[:2, 0])
    expected = Matrix([
        0.0259159090798597, 0.668041660387416, -2.19283799213811,
        0.385441810852219, 0.420109283790573, 1.45030568179066,
        -0.0110924422400793, -8.35617840186040, -0.154098542632173,
        -0.146102664410010])
    assert all(abs(x) < eps for x in
               (solve_sys(q_vals, u_vals, p_vals) - expected))
    if all_zero:
        q_vals = (0, 0, 0, 0, 0)
        u_vals = (0, 0, 0, 0, 0)
        assert solve_sys(q_vals, u_vals, p_vals) == zeros(10, 1)


def test_kane_rolling_disc_lu():
    props = _create_rolling_disc()
    kane = KanesMethod(props['frame'], props['q_ind'], props['u_ind'],
                       props['kdes'], u_dependent=props['u_dep'],
                       velocity_constraints=props['fnh'],
                       bodies=props['bodies'], forcelist=props['loads'],
                       explicit_kinematics=False, constraint_solver='LU')
    kane.kanes_equations()
    _verify_rolling_disc_numerically(kane)


def test_kane_rolling_disc_kdes_callable():
    props = _create_rolling_disc()
    kane = KanesMethod(
        props['frame'], props['q_ind'], props['u_ind'], props['kdes'],
        u_dependent=props['u_dep'], velocity_constraints=props['fnh'],
        bodies=props['bodies'], forcelist=props['loads'],
        explicit_kinematics=False,
        kd_eqs_solver=lambda A, b: simplify(A.LUsolve(b)))
    q, u, p = dynamicsymbols('q1:6'), dynamicsymbols('u1:6'), symbols('g r m')
    qd = dynamicsymbols('q1:6', 1)
    eval_kdes = lambdify((q, qd, u, p), tuple(kane.kindiffdict().items()))
    eps = 1e-10
    # Test with only zeros. If 'LU' would be used this would result in nan.
    p_vals = (9.81, 0.25, 3.5)
    zero_vals = (0, 0, 0, 0, 0)
    assert all(abs(qdi - fui) < eps for qdi, fui in
               eval_kdes(zero_vals, zero_vals, zero_vals, p_vals))
    # Test with some arbitrary values
    q_vals = tuple(map(float, (pi / 6, pi / 3, pi / 2, 0.42, 0.62)))
    qd_vals = tuple(map(float, (4, 1 / 3, 4 - 2 * sqrt(3),
                                0.25 * (2 * sqrt(3) - 3),
                                0.25 * (2 - sqrt(3)))))
    u_vals = tuple(map(float, (-2, 4, 1 / 3, 0.25 * (-3 + 2 * sqrt(3)),
                               0.25 * (-sqrt(3) + 2))))
    assert all(abs(qdi - fui) < eps for qdi, fui in
               eval_kdes(q_vals, qd_vals, u_vals, p_vals))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_precedence.py ===
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative, Function
from sympy.core.numbers import Integer, Rational, Float, oo
from sympy.core.relational import Rel
from sympy.core.symbol import symbols
from sympy.functions import sin
from sympy.integrals.integrals import Integral
from sympy.series.order import Order

from sympy.printing.precedence import precedence, PRECEDENCE

x, y = symbols("x,y")


def test_Add():
    assert precedence(x + y) == PRECEDENCE["Add"]
    assert precedence(x*y + 1) == PRECEDENCE["Add"]


def test_Function():
    assert precedence(sin(x)) == PRECEDENCE["Func"]

def test_Derivative():
    assert precedence(Derivative(x, y)) == PRECEDENCE["Atom"]

def test_Integral():
    assert precedence(Integral(x, y)) == PRECEDENCE["Atom"]


def test_Mul():
    assert precedence(x*y) == PRECEDENCE["Mul"]
    assert precedence(-x*y) == PRECEDENCE["Add"]


def test_Number():
    assert precedence(Integer(0)) == PRECEDENCE["Atom"]
    assert precedence(Integer(1)) == PRECEDENCE["Atom"]
    assert precedence(Integer(-1)) == PRECEDENCE["Add"]
    assert precedence(Integer(10)) == PRECEDENCE["Atom"]
    assert precedence(Rational(5, 2)) == PRECEDENCE["Mul"]
    assert precedence(Rational(-5, 2)) == PRECEDENCE["Add"]
    assert precedence(Float(5)) == PRECEDENCE["Atom"]
    assert precedence(Float(-5)) == PRECEDENCE["Add"]
    assert precedence(oo) == PRECEDENCE["Atom"]
    assert precedence(-oo) == PRECEDENCE["Add"]


def test_Order():
    assert precedence(Order(x)) == PRECEDENCE["Atom"]


def test_Pow():
    assert precedence(x**y) == PRECEDENCE["Pow"]
    assert precedence(-x**y) == PRECEDENCE["Add"]
    assert precedence(x**-y) == PRECEDENCE["Pow"]


def test_Product():
    assert precedence(Product(x, (x, y, y + 1))) == PRECEDENCE["Atom"]


def test_Relational():
    assert precedence(Rel(x + y, y, "<")) == PRECEDENCE["Relational"]


def test_Sum():
    assert precedence(Sum(x, (x, y, y + 1))) == PRECEDENCE["Atom"]


def test_Symbol():
    assert precedence(x) == PRECEDENCE["Atom"]


def test_And_Or():
    # precedence relations between logical operators, ...
    assert precedence(x & y) > precedence(x | y)
    assert precedence(~y) > precedence(x & y)
    # ... and with other operators (cfr. other programming languages)
    assert precedence(x + y) > precedence(x | y)
    assert precedence(x + y) > precedence(x & y)
    assert precedence(x*y) > precedence(x | y)
    assert precedence(x*y) > precedence(x & y)
    assert precedence(~y) > precedence(x*y)
    assert precedence(~y) > precedence(x - y)
    # double checks
    assert precedence(x & y) == PRECEDENCE["And"]
    assert precedence(x | y) == PRECEDENCE["Or"]
    assert precedence(~y) == PRECEDENCE["Not"]


def test_custom_function_precedence_comparison():
    """
    Test cases for custom functions with different precedence values,
    specifically handling:
    1. Functions with precedence < PRECEDENCE["Mul"] (50)
    2. Functions with precedence = Func (70)

    Key distinction:
    1. Lower precedence functions (45) need parentheses: -2*(x F y)
    2. Higher precedence functions (70) don't: -2*x F y
    """
    class LowPrecedenceF(Function):
        precedence = PRECEDENCE["Mul"] - 5
        def _sympystr(self, printer):
            return f"{printer._print(self.args[0])} F {printer._print(self.args[1])}"

    class HighPrecedenceF(Function):
        precedence = PRECEDENCE["Func"]
        def _sympystr(self, printer):
            return f"{printer._print(self.args[0])} F {printer._print(self.args[1])}"

    def test_low_precedence():
        expr1 = 2 * LowPrecedenceF(x, y)
        assert str(expr1) == "2*(x F y)"

        expr2 = -2 * LowPrecedenceF(x, y)
        assert str(expr2) == "-2*(x F y)"

    def test_high_precedence():
        expr1 = 2 * HighPrecedenceF(x, y)
        assert str(expr1) == "2*x F y"

        expr2 = -2 * HighPrecedenceF(x, y)
        assert str(expr2) == "-2*x F y"

    test_low_precedence()
    test_high_precedence()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_convert_matrix_to_array.py ===
from sympy import Lambda, KroneckerProduct
from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import Identity
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayContraction, \
    PermuteDims, ArrayDiagonal, ArrayElementwiseApplyFunc, _array_contraction, _array_tensor_product, Reshape
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

i, j, k, l, m, n = symbols("i j k l m n")

I = Identity(k)

M = MatrixSymbol("M", k, k)
N = MatrixSymbol("N", k, k)
P = MatrixSymbol("P", k, k)
Q = MatrixSymbol("Q", k, k)

A = MatrixSymbol("A", k, k)
B = MatrixSymbol("B", k, k)
C = MatrixSymbol("C", k, k)
D = MatrixSymbol("D", k, k)

X = MatrixSymbol("X", k, k)
Y = MatrixSymbol("Y", k, k)

a = MatrixSymbol("a", k, 1)
b = MatrixSymbol("b", k, 1)
c = MatrixSymbol("c", k, 1)
d = MatrixSymbol("d", k, 1)


def test_arrayexpr_convert_matrix_to_array():

    expr = M*N
    result = ArrayContraction(ArrayTensorProduct(M, N), (1, 2))
    assert convert_matrix_to_array(expr) == result

    expr = M*N*M
    result = _array_contraction(ArrayTensorProduct(M, N, M), (1, 2), (3, 4))
    assert convert_matrix_to_array(expr) == result

    expr = Transpose(M)
    assert convert_matrix_to_array(expr) == PermuteDims(M, [1, 0])

    expr = M*Transpose(N)
    assert convert_matrix_to_array(expr) == _array_contraction(_array_tensor_product(M, PermuteDims(N, [1, 0])), (1, 2))

    expr = 3*M*N
    res = convert_matrix_to_array(expr)
    rexpr = convert_array_to_matrix(res)
    assert expr == rexpr

    expr = 3*M + N*M.T*M + 4*k*N
    res = convert_matrix_to_array(expr)
    rexpr = convert_array_to_matrix(res)
    assert expr == rexpr

    expr = Inverse(M)*N
    rexpr = convert_array_to_matrix(convert_matrix_to_array(expr))
    assert expr == rexpr

    expr = M**2
    rexpr = convert_array_to_matrix(convert_matrix_to_array(expr))
    assert expr == rexpr

    expr = M*(2*N + 3*M)
    res = convert_matrix_to_array(expr)
    rexpr = convert_array_to_matrix(res)
    assert expr == rexpr

    expr = Trace(M)
    result = ArrayContraction(M, (0, 1))
    assert convert_matrix_to_array(expr) == result

    expr = 3*Trace(M)
    result = ArrayContraction(ArrayTensorProduct(3, M), (0, 1))
    assert convert_matrix_to_array(expr) == result

    expr = 3*Trace(Trace(M) * M)
    result = ArrayContraction(ArrayTensorProduct(3, M, M), (0, 1), (2, 3))
    assert convert_matrix_to_array(expr) == result

    expr = 3*Trace(M)**2
    result = ArrayContraction(ArrayTensorProduct(3, M, M), (0, 1), (2, 3))
    assert convert_matrix_to_array(expr) == result

    expr = HadamardProduct(M, N)
    result = ArrayDiagonal(ArrayTensorProduct(M, N), (0, 2), (1, 3))
    assert convert_matrix_to_array(expr) == result

    expr = HadamardProduct(M*N, N*M)
    result = ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, N, M), (1, 2), (5, 6)), (0, 2), (1, 3))
    assert convert_matrix_to_array(expr) == result

    expr = HadamardPower(M, 2)
    result = ArrayDiagonal(ArrayTensorProduct(M, M), (0, 2), (1, 3))
    assert convert_matrix_to_array(expr) == result

    expr = HadamardPower(M*N, 2)
    result = ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, M, N), (1, 2), (5, 6)), (0, 2), (1, 3))
    assert convert_matrix_to_array(expr) == result

    expr = HadamardPower(M, n)
    d0 = Dummy("d0")
    result = ArrayElementwiseApplyFunc(Lambda(d0, d0**n), M)
    assert convert_matrix_to_array(expr).dummy_eq(result)

    expr = M**2
    assert isinstance(expr, MatPow)
    assert convert_matrix_to_array(expr) == ArrayContraction(ArrayTensorProduct(M, M), (1, 2))

    expr = a.T*b
    cg = convert_matrix_to_array(expr)
    assert cg == ArrayContraction(ArrayTensorProduct(a, b), (0, 2))

    expr = KroneckerProduct(A, B)
    cg = convert_matrix_to_array(expr)
    assert cg == Reshape(PermuteDims(ArrayTensorProduct(A, B), [0, 2, 1, 3]), (k**2, k**2))

    expr = KroneckerProduct(A, B, C, D)
    cg = convert_matrix_to_array(expr)
    assert cg == Reshape(PermuteDims(ArrayTensorProduct(A, B, C, D), [0, 2, 4, 6, 1, 3, 5, 7]), (k**4, k**4))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attrs\converters.py ===
# SPDX-License-Identifier: MIT

from attr.converters import *  # noqa: F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attrs\exceptions.py ===
# SPDX-License-Identifier: MIT

from attr.exceptions import *  # noqa: F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attrs\filters.py ===
# SPDX-License-Identifier: MIT

from attr.filters import *  # noqa: F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attrs\setters.py ===
# SPDX-License-Identifier: MIT

from attr.setters import *  # noqa: F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attrs\validators.py ===
# SPDX-License-Identifier: MIT

from attr.validators import *  # noqa: F403

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\__main__.py ===
from .cli import main

main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_extending.py ===
import os
import shutil
import subprocess
import sys
import sysconfig
import warnings
from importlib.util import module_from_spec, spec_from_file_location

import pytest

import numpy as np
from numpy.testing import IS_EDITABLE, IS_WASM

try:
    import cffi
except ImportError:
    cffi = None

if sys.flags.optimize > 1:
    # no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1
    # cffi cannot succeed
    cffi = None

try:
    with warnings.catch_warnings(record=True) as w:
        # numba issue gh-4733
        warnings.filterwarnings('always', '', DeprecationWarning)
        import numba
except (ImportError, SystemError):
    # Certain numpy/numba versions trigger a SystemError due to a numba bug
    numba = None

try:
    import cython
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    from numpy._utils import _pep440
    # Note: keep in sync with the one in pyproject.toml
    required_version = '3.0.6'
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip the test
        cython = None


@pytest.mark.skipif(
    IS_EDITABLE,
    reason='Editable install cannot find .pxd headers'
)
@pytest.mark.skipif(
        sys.platform == "win32" and sys.maxsize < 2**32,
        reason="Failing in 32-bit Windows wheel build job, skip for now"
)
@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
@pytest.mark.skipif(cython is None, reason="requires cython")
@pytest.mark.skipif(sysconfig.get_platform() == 'win-arm64',
                    reason='Meson unable to find MSVC linker on win-arm64')
@pytest.mark.slow
def test_cython(tmp_path):
    import glob
    # build the examples in a temporary directory
    srcdir = os.path.join(os.path.dirname(__file__), '..')
    shutil.copytree(srcdir, tmp_path / 'random')
    build_dir = tmp_path / 'random' / '_examples' / 'cython'
    target_dir = build_dir / "build"
    os.makedirs(target_dir, exist_ok=True)
    # Ensure we use the correct Python interpreter even when `meson` is
    # installed in a different Python environment (see gh-24956)
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'\n")
        f.write(f"python3 = '{sys.executable}'")
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--vsenv", "--native-file", native_file,
                               str(build_dir)],
                              cwd=target_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup",
                               "--native-file", native_file, str(build_dir)],
                              cwd=target_dir
                              )
    subprocess.check_call(["meson", "compile", "-vv"], cwd=target_dir)

    # gh-16162: make sure numpy's __init__.pxd was used for cython
    # not really part of this test, but it is a convenient place to check

    g = glob.glob(str(target_dir / "*" / "extending.pyx.c"))
    with open(g[0]) as fid:
        txt_to_find = 'NumPy API declarations from "numpy/__init__'
        for line in fid:
            if txt_to_find in line:
                break
        else:
            assert False, f"Could not find '{txt_to_find}' in C file, wrong pxd used"
    # import without adding the directory to sys.path
    suffix = sysconfig.get_config_var('EXT_SUFFIX')

    def load(modname):
        so = (target_dir / modname).with_suffix(suffix)
        spec = spec_from_file_location(modname, so)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # test that the module can be imported
    load("extending")
    load("extending_cpp")
    # actually test the cython c-extension
    extending_distributions = load("extending_distributions")
    from numpy.random import PCG64
    values = extending_distributions.uniforms_ex(PCG64(0), 10, 'd')
    assert values.shape == (10,)
    assert values.dtype == np.float64

@pytest.mark.skipif(numba is None or cffi is None,
                    reason="requires numba and cffi")
def test_numba():
    from numpy.random._examples.numba import extending  # noqa: F401

@pytest.mark.skipif(cffi is None, reason="requires cffi")
def test_cffi():
    from numpy.random._examples.cffi import extending  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\fusions\__init__.py ===
from .fusion import Fusion  # noqa: F401
from .fusion_gelu import FusionGelu  # noqa: F401
from .fusion_layernorm import FusionLayerNormalization  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chartsheet\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

from .chartsheet import Chartsheet

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\formatting\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

from .rule import Rule

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\formula\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

from .tokenizer import Tokenizer

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\packaging\__init__.py ===
"""
Stuff related to Office OpenXML packaging: relationships, archive, content types.
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\external_link\__init__.py ===
# Copyright (c) 2010-2024 openpyxl

from .external import ExternalLink

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\sas\__init__.py ===
from pandas.io.sas.sasreader import read_sas

__all__ = ["read_sas"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\conftest.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)


@pytest.fixture
def series_ints():
    return Series(np.random.default_rng(2).random(4), index=np.arange(0, 8, 2))


@pytest.fixture
def frame_ints():
    return DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)),
        index=np.arange(0, 8, 2),
        columns=np.arange(0, 12, 3),
    )


@pytest.fixture
def series_uints():
    return Series(
        np.random.default_rng(2).random(4),
        index=Index(np.arange(0, 8, 2, dtype=np.uint64)),
    )


@pytest.fixture
def frame_uints():
    return DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)),
        index=Index(range(0, 8, 2), dtype=np.uint64),
        columns=Index(range(0, 12, 3), dtype=np.uint64),
    )


@pytest.fixture
def series_labels():
    return Series(np.random.default_rng(2).standard_normal(4), index=list("abcd"))


@pytest.fixture
def frame_labels():
    return DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)),
        index=list("abcd"),
        columns=list("ABCD"),
    )


@pytest.fixture
def series_ts():
    return Series(
        np.random.default_rng(2).standard_normal(4),
        index=date_range("20130101", periods=4),
    )


@pytest.fixture
def frame_ts():
    return DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)),
        index=date_range("20130101", periods=4),
    )


@pytest.fixture
def series_floats():
    return Series(
        np.random.default_rng(2).random(4),
        index=Index(range(0, 8, 2), dtype=np.float64),
    )


@pytest.fixture
def frame_floats():
    return DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)),
        index=Index(range(0, 8, 2), dtype=np.float64),
        columns=Index(range(0, 12, 3), dtype=np.float64),
    )


@pytest.fixture
def series_mixed():
    return Series(np.random.default_rng(2).standard_normal(4), index=[2, 4, "null", 8])


@pytest.fixture
def frame_mixed():
    return DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)), index=[2, 4, "null", 8]
    )


@pytest.fixture
def frame_empty():
    return DataFrame()


@pytest.fixture
def series_empty():
    return Series(dtype=object)


@pytest.fixture
def frame_multi():
    return DataFrame(
        np.random.default_rng(2).standard_normal((4, 4)),
        index=MultiIndex.from_product([[1, 2], [3, 4]]),
        columns=MultiIndex.from_product([[5, 6], [7, 8]]),
    )


@pytest.fixture
def series_multi():
    return Series(
        np.random.default_rng(2).random(4),
        index=MultiIndex.from_product([[1, 2], [3, 4]]),
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\__init__.py ===
"""Subpackage containing all of pip's command line interface related code"""

# This file intentionally does not import submodules

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sniffio\_version.py ===
# This file is imported from __init__.py and exec'd from setup.py

__version__ = "1.3.1"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\algebras\__init__.py ===
from .quaternion import Quaternion

__all__ = ["Quaternion",]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\__init__.py ===
from sympy.liealgebras.cartan_type import CartanType

__all__ = ['CartanType']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\utilities\__init__.py ===
from .dimacs import load_file

__all__ = ['load_file']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\backends\textbackend\__init__.py ===
from sympy.plotting.backends.textbackend.text import TextBackend

__all__ = ["TextBackend"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_gammasimp.py ===
from sympy.core.function import Function
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.simplify import simplify

from sympy.abc import x, y, n, k


def test_gammasimp():
    R = Rational

    # was part of test_combsimp_gamma() in test_combsimp.py
    assert gammasimp(gamma(x)) == gamma(x)
    assert gammasimp(gamma(x + 1)/x) == gamma(x)
    assert gammasimp(gamma(x)/(x - 1)) == gamma(x - 1)
    assert gammasimp(x*gamma(x)) == gamma(x + 1)
    assert gammasimp((x + 1)*gamma(x + 1)) == gamma(x + 2)
    assert gammasimp(gamma(x + y)*(x + y)) == gamma(x + y + 1)
    assert gammasimp(x/gamma(x + 1)) == 1/gamma(x)
    assert gammasimp((x + 1)**2/gamma(x + 2)) == (x + 1)/gamma(x + 1)
    assert gammasimp(x*gamma(x) + gamma(x + 3)/(x + 2)) == \
        (x + 2)*gamma(x + 1)

    assert gammasimp(gamma(2*x)*x) == gamma(2*x + 1)/2
    assert gammasimp(gamma(2*x)/(x - S.Half)) == 2*gamma(2*x - 1)

    assert gammasimp(gamma(x)*gamma(1 - x)) == pi/sin(pi*x)
    assert gammasimp(gamma(x)*gamma(-x)) == -pi/(x*sin(pi*x))
    assert gammasimp(1/gamma(x + 3)/gamma(1 - x)) == \
        sin(pi*x)/(pi*x*(x + 1)*(x + 2))

    assert gammasimp(factorial(n + 2)) == gamma(n + 3)
    assert gammasimp(binomial(n, k)) == \
        gamma(n + 1)/(gamma(k + 1)*gamma(-k + n + 1))

    assert powsimp(gammasimp(
        gamma(x)*gamma(x + S.Half)*gamma(y)/gamma(x + y))) == \
        2**(-2*x + 1)*sqrt(pi)*gamma(2*x)*gamma(y)/gamma(x + y)
    assert gammasimp(1/gamma(x)/gamma(x - Rational(1, 3))/gamma(x + Rational(1, 3))) == \
        3**(3*x - Rational(3, 2))/(2*pi*gamma(3*x - 1))
    assert simplify(
        gamma(S.Half + x/2)*gamma(1 + x/2)/gamma(1 + x)/sqrt(pi)*2**x) == 1
    assert gammasimp(gamma(Rational(-1, 4))*gamma(Rational(-3, 4))) == 16*sqrt(2)*pi/3

    assert powsimp(gammasimp(gamma(2*x)/gamma(x))) == \
        2**(2*x - 1)*gamma(x + S.Half)/sqrt(pi)

    # issue 6792
    e = (-gamma(k)*gamma(k + 2) + gamma(k + 1)**2)/gamma(k)**2
    assert gammasimp(e) == -k
    assert gammasimp(1/e) == -1/k
    e = (gamma(x) + gamma(x + 1))/gamma(x)
    assert gammasimp(e) == x + 1
    assert gammasimp(1/e) == 1/(x + 1)
    e = (gamma(x) + gamma(x + 2))*(gamma(x - 1) + gamma(x))/gamma(x)
    assert gammasimp(e) == (x**2 + x + 1)*gamma(x + 1)/(x - 1)
    e = (-gamma(k)*gamma(k + 2) + gamma(k + 1)**2)/gamma(k)**2
    assert gammasimp(e**2) == k**2
    assert gammasimp(e**2/gamma(k + 1)) == k/gamma(k)
    a = R(1, 2) + R(1, 3)
    b = a + R(1, 3)
    assert gammasimp(gamma(2*k)/gamma(k)*gamma(k + a)*gamma(k + b)
        ) == 3*2**(2*k + 1)*3**(-3*k - 2)*sqrt(pi)*gamma(3*k + R(3, 2))/2

    # issue 9699
    assert gammasimp((x + 1)*factorial(x)/gamma(y)) == gamma(x + 2)/gamma(y)
    assert gammasimp(rf(x + n, k)*binomial(n, k)).simplify() == Piecewise(
        (gamma(n + 1)*gamma(k + n + x)/(gamma(k + 1)*gamma(n + x)*gamma(-k + n + 1)), n > -x),
        ((-1)**k*gamma(n + 1)*gamma(-n - x + 1)/(gamma(k + 1)*gamma(-k + n + 1)*gamma(-k - n - x + 1)), True))

    A, B = symbols('A B', commutative=False)
    assert gammasimp(e*B*A) == gammasimp(e)*B*A

    # check iteration
    assert gammasimp(gamma(2*k)/gamma(k)*gamma(-k - R(1, 2))) == (
        -2**(2*k + 1)*sqrt(pi)/(2*((2*k + 1)*cos(pi*k))))
    assert gammasimp(
        gamma(k)*gamma(k + R(1, 3))*gamma(k + R(2, 3))/gamma(k*R(3, 2))) == (
        3*2**(3*k + 1)*3**(-3*k - S.Half)*sqrt(pi)*gamma(k*R(3, 2) + S.Half)/2)

    # issue 6153
    assert gammasimp(gamma(Rational(1, 4))/gamma(Rational(5, 4))) == 4

    # was part of test_combsimp() in test_combsimp.py
    assert gammasimp(binomial(n + 2, k + S.Half)) == gamma(n + 3)/ \
        (gamma(k + R(3, 2))*gamma(-k + n + R(5, 2)))
    assert gammasimp(binomial(n + 2, k + 2.0)) == \
        gamma(n + 3)/(gamma(k + 3.0)*gamma(-k + n + 1))

    # issue 11548
    assert gammasimp(binomial(0, x)) == sin(pi*x)/(pi*x)

    e = gamma(n + Rational(1, 3))*gamma(n + R(2, 3))
    assert gammasimp(e) == e
    assert gammasimp(gamma(4*n + S.Half)/gamma(2*n - R(3, 4))) == \
        2**(4*n - R(5, 2))*(8*n - 3)*gamma(2*n + R(3, 4))/sqrt(pi)

    i, m = symbols('i m', integer = True)
    e = gamma(exp(i))
    assert gammasimp(e) == e
    e = gamma(m + 3)
    assert gammasimp(e) == e
    e = gamma(m + 1)/(gamma(i + 1)*gamma(-i + m + 1))
    assert gammasimp(e) == e

    p = symbols("p", integer=True, positive=True)
    assert gammasimp(gamma(-p + 4)) == gamma(-p + 4)


def test_issue_22606():
    fx = Function('f')(x)
    eq = x + gamma(y)
    # seems like ans should be `eq`, not `(x*y + gamma(y + 1))/y`
    ans = gammasimp(eq)
    assert gammasimp(eq.subs(x, fx)).subs(fx, x) == ans
    assert gammasimp(eq.subs(x, cos(x))).subs(cos(x), x) == ans
    assert 1/gammasimp(1/eq) == ans
    assert gammasimp(fx.subs(x, eq)).args[0] == ans

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_version.py ===
# This file is imported from __init__.py and parsed by setuptools

__version__ = "0.30.0"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\__main__.py ===
from trio._repl import main

main(locals())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\wrappers\__init__.py ===
from .request import Request as Request
from .response import Response as Response
from .response import ResponseStream as ResponseStream

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wsproto\typing.py ===
from typing import List, Tuple

Headers = List[Tuple[bytes, bytes]]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\boolean\test_function.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize(
    "ufunc", [np.add, np.logical_or, np.logical_and, np.logical_xor]
)
def test_ufuncs_binary(ufunc):
    # two BooleanArrays
    a = pd.array([True, False, None], dtype="boolean")
    result = ufunc(a, a)
    expected = pd.array(ufunc(a._data, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s, a)
    expected = pd.Series(ufunc(a._data, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)

    # Boolean with numpy array
    arr = np.array([True, True, False])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a._data, arr), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    # BooleanArray with scalar
    result = ufunc(a, True)
    expected = pd.array(ufunc(a._data, True), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(True, a)
    expected = pd.array(ufunc(True, a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    # not handled types
    msg = r"operand type\(s\) all returned NotImplemented from __array_ufunc__"
    with pytest.raises(TypeError, match=msg):
        ufunc(a, "test")


@pytest.mark.parametrize("ufunc", [np.logical_not])
def test_ufuncs_unary(ufunc):
    a = pd.array([True, False, None], dtype="boolean")
    result = ufunc(a)
    expected = pd.array(ufunc(a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_extension_array_equal(result, expected)

    ser = pd.Series(a)
    result = ufunc(ser)
    expected = pd.Series(ufunc(a._data), dtype="boolean")
    expected[a._mask] = np.nan
    tm.assert_series_equal(result, expected)


def test_ufunc_numeric():
    # np.sqrt on np.bool_ returns float16, which we upcast to Float32
    #  bc we do not have Float16
    arr = pd.array([True, False, None], dtype="boolean")

    res = np.sqrt(arr)

    expected = pd.array([1, 0, None], dtype="Float32")
    tm.assert_extension_array_equal(res, expected)


@pytest.mark.parametrize("values", [[True, False], [True, None]])
def test_ufunc_reduce_raises(values):
    arr = pd.array(values, dtype="boolean")

    res = np.add.reduce(arr)
    if arr[-1] is pd.NA:
        expected = pd.NA
    else:
        expected = arr._data.sum()
    tm.assert_almost_equal(res, expected)


def test_value_counts_na():
    arr = pd.array([True, False, pd.NA], dtype="boolean")
    result = arr.value_counts(dropna=False)
    expected = pd.Series([1, 1, 1], index=arr, dtype="Int64", name="count")
    assert expected.index.dtype == arr.dtype
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([1, 1], index=arr[:-1], dtype="Int64", name="count")
    assert expected.index.dtype == arr.dtype
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize():
    ser = pd.Series([True, False, pd.NA], dtype="boolean")
    result = ser.value_counts(normalize=True)
    expected = pd.Series([1, 1], index=ser[:-1], dtype="Float64", name="proportion") / 2
    assert expected.index.dtype == "boolean"
    tm.assert_series_equal(result, expected)


def test_diff():
    a = pd.array(
        [True, True, False, False, True, None, True, None, False], dtype="boolean"
    )
    result = pd.core.algorithms.diff(a, 1)
    expected = pd.array(
        [None, False, True, False, True, None, None, None, None], dtype="boolean"
    )
    tm.assert_extension_array_equal(result, expected)

    ser = pd.Series(a)
    result = ser.diff()
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_extension_array_equal.py ===
import numpy as np
import pytest

from pandas import (
    Timestamp,
    array,
)
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # Default is check_exact=False
        {"check_exact": False},
        {"check_exact": True},
    ],
)
def test_assert_extension_array_equal_not_exact(kwargs):
    # see gh-23709
    arr1 = SparseArray([-0.17387645482451206, 0.3414148016424936])
    arr2 = SparseArray([-0.17387645482451206, 0.3414148016424937])

    if kwargs.get("check_exact", False):
        msg = """\
ExtensionArray are different

ExtensionArray values are different \\(50\\.0 %\\)
\\[left\\]:  \\[-0\\.17387645482.*, 0\\.341414801642.*\\]
\\[right\\]: \\[-0\\.17387645482.*, 0\\.341414801642.*\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)


@pytest.mark.parametrize("decimals", range(10))
def test_assert_extension_array_equal_less_precise(decimals):
    rtol = 0.5 * 10**-decimals
    arr1 = SparseArray([0.5, 0.123456])
    arr2 = SparseArray([0.5, 0.123457])

    if decimals >= 5:
        msg = """\
ExtensionArray are different

ExtensionArray values are different \\(50\\.0 %\\)
\\[left\\]:  \\[0\\.5, 0\\.123456\\]
\\[right\\]: \\[0\\.5, 0\\.123457\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)
    else:
        tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)


def test_assert_extension_array_equal_dtype_mismatch(check_dtype):
    end = 5
    kwargs = {"check_dtype": check_dtype}

    arr1 = SparseArray(np.arange(end, dtype="int64"))
    arr2 = SparseArray(np.arange(end, dtype="int32"))

    if check_dtype:
        msg = """\
ExtensionArray are different

Attribute "dtype" are different
\\[left\\]:  Sparse\\[int64, 0\\]
\\[right\\]: Sparse\\[int32, 0\\]"""

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)


def test_assert_extension_array_equal_missing_values():
    arr1 = SparseArray([np.nan, 1, 2, np.nan])
    arr2 = SparseArray([np.nan, 1, 2, 3])

    msg = """\
ExtensionArray NA mask are different

ExtensionArray NA mask values are different \\(25\\.0 %\\)
\\[left\\]:  \\[True, False, False, True\\]
\\[right\\]: \\[True, False, False, False\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(arr1, arr2)


@pytest.mark.parametrize("side", ["left", "right"])
def test_assert_extension_array_equal_non_extension_array(side):
    numpy_array = np.arange(5)
    extension_array = SparseArray(numpy_array)

    msg = f"{side} is not an ExtensionArray"
    args = (
        (numpy_array, extension_array)
        if side == "left"
        else (extension_array, numpy_array)
    )

    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(*args)


@pytest.mark.parametrize("right_dtype", ["Int32", "int64"])
def test_assert_extension_array_equal_ignore_dtype_mismatch(right_dtype):
    # https://github.com/pandas-dev/pandas/issues/35715
    left = array([1, 2, 3], dtype="Int64")
    right = array([1, 2, 3], dtype=right_dtype)
    tm.assert_extension_array_equal(left, right, check_dtype=False)


def test_assert_extension_array_equal_time_units():
    # https://github.com/pandas-dev/pandas/issues/55730
    timestamp = Timestamp("2023-11-04T12")
    naive = array([timestamp], dtype="datetime64[ns]")
    utc = array([timestamp], dtype="datetime64[ns, UTC]")

    tm.assert_extension_array_equal(naive, utc, check_dtype=False)
    tm.assert_extension_array_equal(utc, naive, check_dtype=False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_parameters.py ===
from sympy.abc import x, y
from sympy.core.parameters import evaluate
from sympy.core import Mul, Add, Pow, S
from sympy.core.numbers import oo
from sympy.functions.elementary.miscellaneous import sqrt

def test_add():
    with evaluate(False):
        p = oo - oo
        assert isinstance(p, Add) and p.args == (oo, -oo)
        p = 5 - oo
        assert isinstance(p, Add) and p.args == (-oo, 5)
        p = oo - 5
        assert isinstance(p, Add) and p.args == (oo, -5)
        p = oo + 5
        assert isinstance(p, Add) and p.args == (oo, 5)
        p = 5 + oo
        assert isinstance(p, Add) and p.args == (oo, 5)
        p = -oo + 5
        assert isinstance(p, Add) and p.args == (-oo, 5)
        p = -5 - oo
        assert isinstance(p, Add) and p.args == (-oo, -5)

    with evaluate(False):
        expr = x + x
        assert isinstance(expr, Add)
        assert expr.args == (x, x)

        with evaluate(True):
            assert (x + x).args == (2, x)

        assert (x + x).args == (x, x)

    assert isinstance(x + x, Mul)

    with evaluate(False):
        assert S.One + 1 == Add(1, 1)
        assert 1 + S.One == Add(1, 1)

        assert S(4) - 3 == Add(4, -3)
        assert -3 + S(4) == Add(4, -3)

        assert S(2) * 4 == Mul(2, 4)
        assert 4 * S(2) == Mul(2, 4)

        assert S(6) / 3 == Mul(6, Pow(3, -1))
        assert S.One / 3 * 6 == Mul(S.One / 3, 6)

        assert 9 ** S(2) == Pow(9, 2)
        assert S(2) ** 9 == Pow(2, 9)

        assert S(2) / 2 == Mul(2, Pow(2, -1))
        assert S.One / 2 * 2 == Mul(S.One / 2, 2)

        assert S(2) / 3 + 1 == Add(S(2) / 3, 1)
        assert 1 + S(2) / 3 == Add(1, S(2) / 3)

        assert S(4) / 7 - 3 == Add(S(4) / 7, -3)
        assert -3 + S(4) / 7 == Add(-3, S(4) / 7)

        assert S(2) / 4 * 4 == Mul(S(2) / 4, 4)
        assert 4 * (S(2) / 4) == Mul(4, S(2) / 4)

        assert S(6) / 3 == Mul(6, Pow(3, -1))
        assert S.One / 3 * 6 == Mul(S.One / 3, 6)

        assert S.One / 3 + sqrt(3) == Add(S.One / 3, sqrt(3))
        assert sqrt(3) + S.One / 3 == Add(sqrt(3), S.One / 3)

        assert S.One / 2 * 10.333 == Mul(S.One / 2, 10.333)
        assert 10.333 * (S.One / 2) == Mul(10.333, S.One / 2)

        assert sqrt(2) * sqrt(2) == Mul(sqrt(2), sqrt(2))

        assert S.One / 2 + x == Add(S.One / 2, x)
        assert x + S.One / 2 == Add(x, S.One / 2)

        assert S.One / x * x == Mul(S.One / x, x)
        assert x * (S.One / x) == Mul(x, Pow(x, -1))

        assert S.One / 3 == Pow(3, -1)
        assert S.One / x == Pow(x, -1)
        assert 1 / S(3) == Pow(3, -1)
        assert 1 / x == Pow(x, -1)

def test_nested():
    with evaluate(False):
        expr = (x + x) + (y + y)
        assert expr.args == ((x + x), (y + y))
        assert expr.args[0].args == (x, x)

def test_reentrantcy():
    with evaluate(False):
        expr = x + x
        assert expr.args == (x, x)
        with evaluate(True):
            expr = x + x
            assert expr.args == (2, x)
        expr = x + x
        assert expr.args == (x, x)

def test_reusability():
    f = evaluate(False)

    with f:
        expr = x + x
        assert expr.args == (x, x)

    expr = x + x
    assert expr.args == (2, x)

    with f:
        expr = x + x
        assert expr.args == (x, x)

    # Assure reentrancy with reusability
    ctx = evaluate(False)
    with ctx:
        expr = x + x
        assert expr.args == (x, x)
        with ctx:
            expr = x + x
            assert expr.args == (x, x)

    expr = x + x
    assert expr.args == (2, x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\tests\test_hydrogen.py ===
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.physics.hydrogen import R_nl, E_nl, E_nl_dirac, Psi_nlm
from sympy.testing.pytest import raises

n, r, Z = symbols('n r Z')


def feq(a, b, max_relative_error=1e-12, max_absolute_error=1e-12):
    a = float(a)
    b = float(b)
    # if the numbers are close enough (absolutely), then they are equal
    if abs(a - b) < max_absolute_error:
        return True
    # if not, they can still be equal if their relative error is small
    if abs(b) > abs(a):
        relative_error = abs((a - b)/b)
    else:
        relative_error = abs((a - b)/a)
    return relative_error <= max_relative_error


def test_wavefunction():
    a = 1/Z
    R = {
        (1, 0): 2*sqrt(1/a**3) * exp(-r/a),
        (2, 0): sqrt(1/(2*a**3)) * exp(-r/(2*a)) * (1 - r/(2*a)),
        (2, 1): S.Half * sqrt(1/(6*a**3)) * exp(-r/(2*a)) * r/a,
        (3, 0): Rational(2, 3) * sqrt(1/(3*a**3)) * exp(-r/(3*a)) *
        (1 - 2*r/(3*a) + Rational(2, 27) * (r/a)**2),
        (3, 1): Rational(4, 27) * sqrt(2/(3*a**3)) * exp(-r/(3*a)) *
        (1 - r/(6*a)) * r/a,
        (3, 2): Rational(2, 81) * sqrt(2/(15*a**3)) * exp(-r/(3*a)) * (r/a)**2,
        (4, 0): Rational(1, 4) * sqrt(1/a**3) * exp(-r/(4*a)) *
        (1 - 3*r/(4*a) + Rational(1, 8) * (r/a)**2 - Rational(1, 192) * (r/a)**3),
        (4, 1): Rational(1, 16) * sqrt(5/(3*a**3)) * exp(-r/(4*a)) *
        (1 - r/(4*a) + Rational(1, 80) * (r/a)**2) * (r/a),
        (4, 2): Rational(1, 64) * sqrt(1/(5*a**3)) * exp(-r/(4*a)) *
        (1 - r/(12*a)) * (r/a)**2,
        (4, 3): Rational(1, 768) * sqrt(1/(35*a**3)) * exp(-r/(4*a)) * (r/a)**3,
    }
    for n, l in R:
        assert simplify(R_nl(n, l, r, Z) - R[(n, l)]) == 0


def test_norm():
    # Maximum "n" which is tested:
    n_max = 2  # it works, but is slow, for n_max > 2
    for n in range(n_max + 1):
        for l in range(n):
            assert integrate(R_nl(n, l, r)**2 * r**2, (r, 0, oo)) == 1

def test_psi_nlm():
    r=S('r')
    phi=S('phi')
    theta=S('theta')
    assert (Psi_nlm(1, 0, 0, r, phi, theta) == exp(-r) / sqrt(pi))
    assert (Psi_nlm(2, 1, -1, r, phi, theta)) == S.Half * exp(-r / (2)) * r \
        * (sin(theta) * exp(-I * phi) / (4 * sqrt(pi)))
    assert (Psi_nlm(3, 2, 1, r, phi, theta, 2) == -sqrt(2) * sin(theta) \
         * exp(I * phi) * cos(theta) / (4 * sqrt(pi)) * S(2) / 81 \
        * sqrt(2 * 2 ** 3) * exp(-2 * r / (3)) * (r * 2) ** 2)

def test_hydrogen_energies():
    assert E_nl(n, Z) == -Z**2/(2*n**2)
    assert E_nl(n) == -1/(2*n**2)

    assert E_nl(1, 47) == -S(47)**2/(2*1**2)
    assert E_nl(2, 47) == -S(47)**2/(2*2**2)

    assert E_nl(1) == -S.One/(2*1**2)
    assert E_nl(2) == -S.One/(2*2**2)
    assert E_nl(3) == -S.One/(2*3**2)
    assert E_nl(4) == -S.One/(2*4**2)
    assert E_nl(100) == -S.One/(2*100**2)

    raises(ValueError, lambda: E_nl(0))


def test_hydrogen_energies_relat():
    # First test exact formulas for small "c" so that we get nice expressions:
    assert E_nl_dirac(2, 0, Z=1, c=1) == 1/sqrt(2) - 1
    assert simplify(E_nl_dirac(2, 0, Z=1, c=2) - ( (8*sqrt(3) + 16)
                / sqrt(16*sqrt(3) + 32) - 4)) == 0
    assert simplify(E_nl_dirac(2, 0, Z=1, c=3) - ( (54*sqrt(2) + 81)
                / sqrt(108*sqrt(2) + 162) - 9)) == 0

    # Now test for almost the correct speed of light, without floating point
    # numbers:
    assert simplify(E_nl_dirac(2, 0, Z=1, c=137) - ( (352275361 + 10285412 *
        sqrt(1173)) / sqrt(704550722 + 20570824 * sqrt(1173)) - 18769)) == 0
    assert simplify(E_nl_dirac(2, 0, Z=82, c=137) - ( (352275361 + 2571353 *
        sqrt(12045)) / sqrt(704550722 + 5142706*sqrt(12045)) - 18769)) == 0

    # Test using exact speed of light, and compare against the nonrelativistic
    # energies:
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l), E_nl(n), 1e-5, 1e-5)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False), E_nl(n), 1e-5, 1e-5)

    Z = 2
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l, Z=Z), E_nl(n, Z), 1e-4, 1e-4)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False, Z), E_nl(n, Z), 1e-4, 1e-4)

    Z = 3
    for n in range(1, 5):
        for l in range(n):
            assert feq(E_nl_dirac(n, l, Z=Z), E_nl(n, Z), 1e-3, 1e-3)
            if l > 0:
                assert feq(E_nl_dirac(n, l, False, Z), E_nl(n, Z), 1e-3, 1e-3)

    # Test the exceptions:
    raises(ValueError, lambda: E_nl_dirac(0, 0))
    raises(ValueError, lambda: E_nl_dirac(1, -1))
    raises(ValueError, lambda: E_nl_dirac(1, 0, False))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_polyfuncs.py ===
"""Tests for high-level polynomials manipulation functions. """

from sympy.polys.polyfuncs import (
    symmetrize, horner, interpolate, rational_interpolate, viete,
)

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
)

from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.testing.pytest import raises

from sympy.abc import a, b, c, d, e, x, y, z


def test_symmetrize():
    assert symmetrize(0, x, y, z) == (0, 0)
    assert symmetrize(1, x, y, z) == (1, 0)

    s1 = x + y + z
    s2 = x*y + x*z + y*z

    assert symmetrize(1) == (1, 0)
    assert symmetrize(1, formal=True) == (1, 0, [])

    assert symmetrize(x) == (x, 0)
    assert symmetrize(x + 1) == (x + 1, 0)

    assert symmetrize(x, x, y) == (x + y, -y)
    assert symmetrize(x + 1, x, y) == (x + y + 1, -y)

    assert symmetrize(x, x, y, z) == (s1, -y - z)
    assert symmetrize(x + 1, x, y, z) == (s1 + 1, -y - z)

    assert symmetrize(x**2, x, y, z) == (s1**2 - 2*s2, -y**2 - z**2)

    assert symmetrize(x**2 + y**2) == (-2*x*y + (x + y)**2, 0)
    assert symmetrize(x**2 - y**2) == (-2*x*y + (x + y)**2, -2*y**2)

    assert symmetrize(x**3 + y**2 + a*x**2 + b*y**3, x, y) == \
        (-3*x*y*(x + y) - 2*a*x*y + a*(x + y)**2 + (x + y)**3,
         y**2*(1 - a) + y**3*(b - 1))

    U = [u0, u1, u2] = symbols('u:3')

    assert symmetrize(x + 1, x, y, z, formal=True, symbols=U) == \
        (u0 + 1, -y - z, [(u0, x + y + z), (u1, x*y + x*z + y*z), (u2, x*y*z)])

    assert symmetrize([1, 2, 3]) == [(1, 0), (2, 0), (3, 0)]
    assert symmetrize([1, 2, 3], formal=True) == ([(1, 0), (2, 0), (3, 0)], [])

    assert symmetrize([x + y, x - y]) == [(x + y, 0), (x + y, -2*y)]


def test_horner():
    assert horner(0) == 0
    assert horner(1) == 1
    assert horner(x) == x

    assert horner(x + 1) == x + 1
    assert horner(x**2 + 1) == x**2 + 1
    assert horner(x**2 + x) == (x + 1)*x
    assert horner(x**2 + x + 1) == (x + 1)*x + 1

    assert horner(
        9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5) == (((9*x + 8)*x + 7)*x + 6)*x + 5
    assert horner(
        a*x**4 + b*x**3 + c*x**2 + d*x + e) == (((a*x + b)*x + c)*x + d)*x + e

    assert horner(4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y, wrt=x) == ((
        4*y + 2)*x*y + (2*y + 1)*y)*x
    assert horner(4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y, wrt=y) == ((
        4*x + 2)*y*x + (2*x + 1)*x)*y


def test_interpolate():
    assert interpolate([1, 4, 9, 16], x) == x**2
    assert interpolate([1, 4, 9, 25], x) == S(3)*x**3/2 - S(8)*x**2 + S(33)*x/2 - 9
    assert interpolate([(1, 1), (2, 4), (3, 9)], x) == x**2
    assert interpolate([(1, 2), (2, 5), (3, 10)], x) == 1 + x**2
    assert interpolate({1: 2, 2: 5, 3: 10}, x) == 1 + x**2
    assert interpolate({5: 2, 7: 5, 8: 10, 9: 13}, x) == \
        -S(13)*x**3/24 + S(12)*x**2 - S(2003)*x/24 + 187
    assert interpolate([(1, 3), (0, 6), (2, 5), (5, 7), (-2, 4)], x) == \
        S(-61)*x**4/280 + S(247)*x**3/210 + S(139)*x**2/280 - S(1871)*x/420 + 6
    assert interpolate((9, 4, 9), 3) == 9
    assert interpolate((1, 9, 16), 1) is S.One
    assert interpolate(((x, 1), (2, 3)), x) is S.One
    assert interpolate({x: 1, 2: 3}, x) is S.One
    assert interpolate(((2, x), (1, 3)), x) == x**2 - 4*x + 6


def test_rational_interpolate():
    x, y = symbols('x,y')
    xdata = [1, 2, 3, 4, 5, 6]
    ydata1 = [120, 150, 200, 255, 312, 370]
    ydata2 = [-210, -35, 105, 231, 350, 465]
    assert rational_interpolate(list(zip(xdata, ydata1)), 2) == (
      (60*x**2 + 60)/x )
    assert rational_interpolate(list(zip(xdata, ydata1)), 3) == (
      (60*x**2 + 60)/x )
    assert rational_interpolate(list(zip(xdata, ydata2)), 2, X=y) == (
      (105*y**2 - 525)/(y + 1) )
    xdata = list(range(1,11))
    ydata = [-1923885361858460, -5212158811973685, -9838050145867125,
      -15662936261217245, -22469424125057910, -30073793365223685,
      -38332297297028735, -47132954289530109, -56387719094026320,
      -66026548943876885]
    assert rational_interpolate(list(zip(xdata, ydata)), 5) == (
      (-12986226192544605*x**4 +
      8657484128363070*x**3 - 30301194449270745*x**2 + 4328742064181535*x
      - 4328742064181535)/(x**3 + 9*x**2 - 3*x + 11))


def test_viete():
    r1, r2 = symbols('r1, r2')

    assert viete(
        a*x**2 + b*x + c, [r1, r2], x) == [(r1 + r2, -b/a), (r1*r2, c/a)]

    raises(ValueError, lambda: viete(1, [], x))
    raises(ValueError, lambda: viete(x**2 + 1, [r1]))

    raises(MultivariatePolynomialError, lambda: viete(x + y, [r1]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_indexerrors.py ===
import numpy as np
from numpy.testing import (
    assert_raises,
    assert_raises_regex,
)


class TestIndexErrors:
    '''Tests to exercise indexerrors not covered by other tests.'''

    def test_arraytypes_fasttake(self):
        'take from a 0-length dimension'
        x = np.empty((2, 3, 0, 4))
        assert_raises(IndexError, x.take, [0], axis=2)
        assert_raises(IndexError, x.take, [1], axis=2)
        assert_raises(IndexError, x.take, [0], axis=2, mode='wrap')
        assert_raises(IndexError, x.take, [0], axis=2, mode='clip')

    def test_take_from_object(self):
        # Check exception taking from object array
        d = np.zeros(5, dtype=object)
        assert_raises(IndexError, d.take, [6])

        # Check exception taking from 0-d array
        d = np.zeros((5, 0), dtype=object)
        assert_raises(IndexError, d.take, [1], axis=1)
        assert_raises(IndexError, d.take, [0], axis=1)
        assert_raises(IndexError, d.take, [0])
        assert_raises(IndexError, d.take, [0], mode='wrap')
        assert_raises(IndexError, d.take, [0], mode='clip')

    def test_multiindex_exceptions(self):
        a = np.empty(5, dtype=object)
        assert_raises(IndexError, a.item, 20)
        a = np.empty((5, 0), dtype=object)
        assert_raises(IndexError, a.item, (0, 0))

    def test_put_exceptions(self):
        a = np.zeros((5, 5))
        assert_raises(IndexError, a.put, 100, 0)
        a = np.zeros((5, 5), dtype=object)
        assert_raises(IndexError, a.put, 100, 0)
        a = np.zeros((5, 5, 0))
        assert_raises(IndexError, a.put, 100, 0)
        a = np.zeros((5, 5, 0), dtype=object)
        assert_raises(IndexError, a.put, 100, 0)

    def test_iterators_exceptions(self):
        "cases in iterators.c"
        def assign(obj, ind, val):
            obj[ind] = val

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a[0, 5, None, 2])
        assert_raises(IndexError, lambda: a[0, 5, 0, 2])
        assert_raises(IndexError, lambda: assign(a, (0, 5, None, 2), 1))
        assert_raises(IndexError, lambda: assign(a, (0, 5, 0, 2),  1))

        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a[0, 0, None, 2])
        assert_raises(IndexError, lambda: assign(a, (0, 0, None, 2), 1))

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a.flat[10])
        assert_raises(IndexError, lambda: assign(a.flat, 10, 5))
        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a.flat[10])
        assert_raises(IndexError, lambda: assign(a.flat, 10, 5))

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a.flat[np.array(10)])
        assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))
        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a.flat[np.array(10)])
        assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))

        a = np.zeros([1, 2, 3])
        assert_raises(IndexError, lambda: a.flat[np.array([10])])
        assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))
        a = np.zeros([1, 0, 3])
        assert_raises(IndexError, lambda: a.flat[np.array([10])])
        assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))

    def test_mapping(self):
        "cases from mapping.c"

        def assign(obj, ind, val):
            obj[ind] = val

        a = np.zeros((0, 10))
        assert_raises(IndexError, lambda: a[12])

        a = np.zeros((3, 5))
        assert_raises(IndexError, lambda: a[(10, 20)])
        assert_raises(IndexError, lambda: assign(a, (10, 20), 1))
        a = np.zeros((3, 0))
        assert_raises(IndexError, lambda: a[(1, 0)])
        assert_raises(IndexError, lambda: assign(a, (1, 0), 1))

        a = np.zeros((10,))
        assert_raises(IndexError, lambda: assign(a, 10, 1))
        a = np.zeros((0,))
        assert_raises(IndexError, lambda: assign(a, 10, 1))

        a = np.zeros((3, 5))
        assert_raises(IndexError, lambda: a[(1, [1, 20])])
        assert_raises(IndexError, lambda: assign(a, (1, [1, 20]), 1))
        a = np.zeros((3, 0))
        assert_raises(IndexError, lambda: a[(1, [0, 1])])
        assert_raises(IndexError, lambda: assign(a, (1, [0, 1]), 1))

    def test_mapping_error_message(self):
        a = np.zeros((3, 5))
        index = (1, 2, 3, 4, 5)
        assert_raises_regex(
                IndexError,
                "too many indices for array: "
                "array is 2-dimensional, but 5 were indexed",
                lambda: a[index])

    def test_methods(self):
        "cases from methods.c"

        a = np.zeros((3, 3))
        assert_raises(IndexError, lambda: a.item(100))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_factorize.py ===
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    Index,
    date_range,
    factorize,
)
import pandas._testing as tm


class TestDatetimeIndexFactorize:
    def test_factorize(self):
        idx1 = DatetimeIndex(
            ["2014-01", "2014-01", "2014-02", "2014-02", "2014-03", "2014-03"]
        )

        exp_arr = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
        exp_idx = DatetimeIndex(["2014-01", "2014-02", "2014-03"])

        arr, idx = idx1.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
        assert idx.freq == exp_idx.freq

        arr, idx = idx1.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
        assert idx.freq == exp_idx.freq

        # tz must be preserved
        idx1 = idx1.tz_localize("Asia/Tokyo")
        exp_idx = exp_idx.tz_localize("Asia/Tokyo")

        arr, idx = idx1.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
        assert idx.freq == exp_idx.freq

        idx2 = DatetimeIndex(
            ["2014-03", "2014-03", "2014-02", "2014-01", "2014-03", "2014-01"]
        )

        exp_arr = np.array([2, 2, 1, 0, 2, 0], dtype=np.intp)
        exp_idx = DatetimeIndex(["2014-01", "2014-02", "2014-03"])
        arr, idx = idx2.factorize(sort=True)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
        assert idx.freq == exp_idx.freq

        exp_arr = np.array([0, 0, 1, 2, 0, 2], dtype=np.intp)
        exp_idx = DatetimeIndex(["2014-03", "2014-02", "2014-01"])
        arr, idx = idx2.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, exp_idx)
        assert idx.freq == exp_idx.freq

    def test_factorize_preserves_freq(self):
        # GH#38120 freq should be preserved
        idx3 = date_range("2000-01", periods=4, freq="ME", tz="Asia/Tokyo")
        exp_arr = np.array([0, 1, 2, 3], dtype=np.intp)

        arr, idx = idx3.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, idx3)
        assert idx.freq == idx3.freq

        arr, idx = factorize(idx3)
        tm.assert_numpy_array_equal(arr, exp_arr)
        tm.assert_index_equal(idx, idx3)
        assert idx.freq == idx3.freq

    def test_factorize_tz(self, tz_naive_fixture, index_or_series):
        tz = tz_naive_fixture
        # GH#13750
        base = date_range("2016-11-05", freq="h", periods=100, tz=tz)
        idx = base.repeat(5)

        exp_arr = np.arange(100, dtype=np.intp).repeat(5)

        obj = index_or_series(idx)

        arr, res = obj.factorize()
        tm.assert_numpy_array_equal(arr, exp_arr)
        expected = base._with_freq(None)
        tm.assert_index_equal(res, expected)
        assert res.freq == expected.freq

    def test_factorize_dst(self, index_or_series):
        # GH#13750
        idx = date_range("2016-11-06", freq="h", periods=12, tz="US/Eastern")
        obj = index_or_series(idx)

        arr, res = obj.factorize()
        tm.assert_numpy_array_equal(arr, np.arange(12, dtype=np.intp))
        tm.assert_index_equal(res, idx)
        if index_or_series is Index:
            assert res.freq == idx.freq

        idx = date_range("2016-06-13", freq="h", periods=12, tz="US/Eastern")
        obj = index_or_series(idx)

        arr, res = obj.factorize()
        tm.assert_numpy_array_equal(arr, np.arange(12, dtype=np.intp))
        tm.assert_index_equal(res, idx)
        if index_or_series is Index:
            assert res.freq == idx.freq

    @pytest.mark.parametrize("sort", [True, False])
    def test_factorize_no_freq_non_nano(self, tz_naive_fixture, sort):
        # GH#51978 case that does not go through the fastpath based on
        #  non-None freq
        tz = tz_naive_fixture
        idx = date_range("2016-11-06", freq="h", periods=5, tz=tz)[[0, 4, 1, 3, 2]]
        exp_codes, exp_uniques = idx.factorize(sort=sort)

        res_codes, res_uniques = idx.as_unit("s").factorize(sort=sort)

        tm.assert_numpy_array_equal(res_codes, exp_codes)
        tm.assert_index_equal(res_uniques, exp_uniques.as_unit("s"))

        res_codes, res_uniques = idx.as_unit("s").to_series().factorize(sort=sort)
        tm.assert_numpy_array_equal(res_codes, exp_codes)
        tm.assert_index_equal(res_uniques, exp_uniques.as_unit("s"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\test_abnf.py ===
# -*- coding: utf-8 -*-
#
import unittest

from websocket._abnf import ABNF, frame_buffer
from websocket._exceptions import WebSocketProtocolException

"""
test_abnf.py
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


class ABNFTest(unittest.TestCase):
    def test_init(self):
        a = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_PING)
        self.assertEqual(a.fin, 0)
        self.assertEqual(a.rsv1, 0)
        self.assertEqual(a.rsv2, 0)
        self.assertEqual(a.rsv3, 0)
        self.assertEqual(a.opcode, 9)
        self.assertEqual(a.data, "")
        a_bad = ABNF(0, 1, 0, 0, opcode=77)
        self.assertEqual(a_bad.rsv1, 1)
        self.assertEqual(a_bad.opcode, 77)

    def test_validate(self):
        a_invalid_ping = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_PING)
        self.assertRaises(
            WebSocketProtocolException,
            a_invalid_ping.validate,
            skip_utf8_validation=False,
        )
        a_bad_rsv_value = ABNF(0, 1, 0, 0, opcode=ABNF.OPCODE_TEXT)
        self.assertRaises(
            WebSocketProtocolException,
            a_bad_rsv_value.validate,
            skip_utf8_validation=False,
        )
        a_bad_opcode = ABNF(0, 0, 0, 0, opcode=77)
        self.assertRaises(
            WebSocketProtocolException,
            a_bad_opcode.validate,
            skip_utf8_validation=False,
        )
        a_bad_close_frame = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_CLOSE, data=b"\x01")
        self.assertRaises(
            WebSocketProtocolException,
            a_bad_close_frame.validate,
            skip_utf8_validation=False,
        )
        a_bad_close_frame_2 = ABNF(
            0, 0, 0, 0, opcode=ABNF.OPCODE_CLOSE, data=b"\x01\x8a\xaa\xff\xdd"
        )
        self.assertRaises(
            WebSocketProtocolException,
            a_bad_close_frame_2.validate,
            skip_utf8_validation=False,
        )
        a_bad_close_frame_3 = ABNF(
            0, 0, 0, 0, opcode=ABNF.OPCODE_CLOSE, data=b"\x03\xe7"
        )
        self.assertRaises(
            WebSocketProtocolException,
            a_bad_close_frame_3.validate,
            skip_utf8_validation=True,
        )

    def test_mask(self):
        abnf_none_data = ABNF(
            0, 0, 0, 0, opcode=ABNF.OPCODE_PING, mask_value=1, data=None
        )
        bytes_val = b"aaaa"
        self.assertEqual(abnf_none_data._get_masked(bytes_val), bytes_val)
        abnf_str_data = ABNF(
            0, 0, 0, 0, opcode=ABNF.OPCODE_PING, mask_value=1, data="a"
        )
        self.assertEqual(abnf_str_data._get_masked(bytes_val), b"aaaa\x00")

    def test_format(self):
        abnf_bad_rsv_bits = ABNF(2, 0, 0, 0, opcode=ABNF.OPCODE_TEXT)
        self.assertRaises(ValueError, abnf_bad_rsv_bits.format)
        abnf_bad_opcode = ABNF(0, 0, 0, 0, opcode=5)
        self.assertRaises(ValueError, abnf_bad_opcode.format)
        abnf_length_10 = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_TEXT, data="abcdefghij")
        self.assertEqual(b"\x01", abnf_length_10.format()[0].to_bytes(1, "big"))
        self.assertEqual(b"\x8a", abnf_length_10.format()[1].to_bytes(1, "big"))
        self.assertEqual("fin=0 opcode=1 data=abcdefghij", abnf_length_10.__str__())
        abnf_length_20 = ABNF(
            0, 0, 0, 0, opcode=ABNF.OPCODE_BINARY, data="abcdefghijabcdefghij"
        )
        self.assertEqual(b"\x02", abnf_length_20.format()[0].to_bytes(1, "big"))
        self.assertEqual(b"\x94", abnf_length_20.format()[1].to_bytes(1, "big"))
        abnf_no_mask = ABNF(
            0, 0, 0, 0, opcode=ABNF.OPCODE_TEXT, mask_value=0, data=b"\x01\x8a\xcc"
        )
        self.assertEqual(b"\x01\x03\x01\x8a\xcc", abnf_no_mask.format())

    def test_frame_buffer(self):
        fb = frame_buffer(0, True)
        self.assertEqual(fb.recv, 0)
        self.assertEqual(fb.skip_utf8_validation, True)
        fb.clear
        self.assertEqual(fb.header, None)
        self.assertEqual(fb.length, None)
        self.assertEqual(fb.mask_value, None)
        self.assertEqual(fb.has_mask(), False)


if __name__ == "__main__":
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_umath_accuracy.py ===
import os
import sys
from ctypes import POINTER, c_double, c_float, c_int, c_longlong, cast, pointer
from os import path

import pytest
from numpy._core._multiarray_umath import __cpu_features__

import numpy as np
from numpy.testing import assert_array_max_ulp
from numpy.testing._private.utils import _glibc_older_than

UNARY_UFUNCS = [obj for obj in np._core.umath.__dict__.values() if
        isinstance(obj, np.ufunc)]
UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]

# Remove functions that do not support `floats`
UNARY_OBJECT_UFUNCS.remove(np.invert)
UNARY_OBJECT_UFUNCS.remove(np.bitwise_count)

IS_AVX = __cpu_features__.get('AVX512F', False) or \
        (__cpu_features__.get('FMA3', False) and __cpu_features__.get('AVX2', False))

IS_AVX512FP16 = __cpu_features__.get('AVX512FP16', False)

# only run on linux with AVX, also avoid old glibc (numpy/numpy#20448).
runtest = (sys.platform.startswith('linux')
           and IS_AVX and not _glibc_older_than("2.17"))
platform_skip = pytest.mark.skipif(not runtest,
                                   reason="avoid testing inconsistent platform "
                                   "library implementations")

# convert string to hex function taken from:
# https://stackoverflow.com/questions/1592158/convert-hex-to-float #
def convert(s, datatype="np.float32"):
    i = int(s, 16)                   # convert from hex to a Python int
    if (datatype == "np.float64"):
        cp = pointer(c_longlong(i))           # make this into a c long long integer
        fp = cast(cp, POINTER(c_double))  # cast the int pointer to a double pointer
    else:
        cp = pointer(c_int(i))           # make this into a c integer
        fp = cast(cp, POINTER(c_float))  # cast the int pointer to a float pointer

    return fp.contents.value         # dereference the pointer, get the float


str_to_float = np.vectorize(convert)

class TestAccuracy:
    @platform_skip
    def test_validate_transcendentals(self):
        with np.errstate(all='ignore'):
            data_dir = path.join(path.dirname(__file__), 'data')
            files = os.listdir(data_dir)
            files = list(filter(lambda f: f.endswith('.csv'), files))
            for filename in files:
                filepath = path.join(data_dir, filename)
                with open(filepath) as fid:
                    file_without_comments = (
                        r for r in fid if r[0] not in ('$', '#')
                    )
                    data = np.genfromtxt(file_without_comments,
                                         dtype=('|S39', '|S39', '|S39', int),
                                         names=('type', 'input', 'output', 'ulperr'),
                                         delimiter=',',
                                         skip_header=1)
                    npname = path.splitext(filename)[0].split('-')[3]
                    npfunc = getattr(np, npname)
                    for datatype in np.unique(data['type']):
                        data_subset = data[data['type'] == datatype]
                        inval = np.array(str_to_float(data_subset['input'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                        outval = np.array(str_to_float(data_subset['output'].astype(str), data_subset['type'].astype(str)), dtype=eval(datatype))
                        perm = np.random.permutation(len(inval))
                        inval = inval[perm]
                        outval = outval[perm]
                        maxulperr = data_subset['ulperr'].max()
                        assert_array_max_ulp(npfunc(inval), outval, maxulperr)

    @pytest.mark.skipif(IS_AVX512FP16,
            reason="SVML FP16 have slightly higher ULP errors")
    @pytest.mark.parametrize("ufunc", UNARY_OBJECT_UFUNCS)
    def test_validate_fp16_transcendentals(self, ufunc):
        with np.errstate(all='ignore'):
            arr = np.arange(65536, dtype=np.int16)
            datafp16 = np.frombuffer(arr.tobytes(), dtype=np.float16)
            datafp32 = datafp16.astype(np.float32)
            assert_array_max_ulp(ufunc(datafp16), ufunc(datafp32),
                    maxulp=1, dtype=np.float16)

    @pytest.mark.skipif(not IS_AVX512FP16,
                               reason="lower ULP only apply for SVML FP16")
    def test_validate_svml_fp16(self):
        max_ulp_err = {
                "arccos": 2.54,
                "arccosh": 2.09,
                "arcsin": 3.06,
                "arcsinh": 1.51,
                "arctan": 2.61,
                "arctanh": 1.88,
                "cbrt": 1.57,
                "cos": 1.43,
                "cosh": 1.33,
                "exp2": 1.33,
                "exp": 1.27,
                "expm1": 0.53,
                "log": 1.80,
                "log10": 1.27,
                "log1p": 1.88,
                "log2": 1.80,
                "sin": 1.88,
                "sinh": 2.05,
                "tan": 2.26,
                "tanh": 3.00,
                }

        with np.errstate(all='ignore'):
            arr = np.arange(65536, dtype=np.int16)
            datafp16 = np.frombuffer(arr.tobytes(), dtype=np.float16)
            datafp32 = datafp16.astype(np.float32)
            for func in max_ulp_err:
                ufunc = getattr(np, func)
                ulp = np.ceil(max_ulp_err[func])
                assert_array_max_ulp(ufunc(datafp16), ufunc(datafp32),
                        maxulp=ulp, dtype=np.float16)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_get_level_values.py ===
import numpy as np

import pandas as pd
from pandas import (
    CategoricalIndex,
    Index,
    MultiIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestGetLevelValues:
    def test_get_level_values_box_datetime64(self):
        dates = date_range("1/1/2000", periods=4)
        levels = [dates, [0, 1]]
        codes = [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]]

        index = MultiIndex(levels=levels, codes=codes)

        assert isinstance(index.get_level_values(0)[0], Timestamp)


def test_get_level_values(idx):
    result = idx.get_level_values(0)
    expected = Index(["foo", "foo", "bar", "baz", "qux", "qux"], name="first")
    tm.assert_index_equal(result, expected)
    assert result.name == "first"

    result = idx.get_level_values("first")
    expected = idx.get_level_values(0)
    tm.assert_index_equal(result, expected)

    # GH 10460
    index = MultiIndex(
        levels=[CategoricalIndex(["A", "B"]), CategoricalIndex([1, 2, 3])],
        codes=[np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 0, 1, 2])],
    )

    exp = CategoricalIndex(["A", "A", "A", "B", "B", "B"])
    tm.assert_index_equal(index.get_level_values(0), exp)
    exp = CategoricalIndex([1, 2, 3, 1, 2, 3])
    tm.assert_index_equal(index.get_level_values(1), exp)


def test_get_level_values_all_na():
    # GH#17924 when level entirely consists of nan
    arrays = [[np.nan, np.nan, np.nan], ["a", np.nan, 1]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([np.nan, np.nan, np.nan], dtype=np.float64)
    tm.assert_index_equal(result, expected)

    result = index.get_level_values(1)
    expected = Index(["a", np.nan, 1], dtype=object)
    tm.assert_index_equal(result, expected)


def test_get_level_values_int_with_na():
    # GH#17924
    arrays = [["a", "b", "b"], [1, np.nan, 2]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = Index([1, np.nan, 2])
    tm.assert_index_equal(result, expected)

    arrays = [["a", "b", "b"], [np.nan, np.nan, 2]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = Index([np.nan, np.nan, 2])
    tm.assert_index_equal(result, expected)


def test_get_level_values_na():
    arrays = [[np.nan, np.nan, np.nan], ["a", np.nan, 1]]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([np.nan, np.nan, np.nan])
    tm.assert_index_equal(result, expected)

    result = index.get_level_values(1)
    expected = Index(["a", np.nan, 1])
    tm.assert_index_equal(result, expected)

    arrays = [["a", "b", "b"], pd.DatetimeIndex([0, 1, pd.NaT])]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(1)
    expected = pd.DatetimeIndex([0, 1, pd.NaT])
    tm.assert_index_equal(result, expected)

    arrays = [[], []]
    index = MultiIndex.from_arrays(arrays)
    result = index.get_level_values(0)
    expected = Index([], dtype=object)
    tm.assert_index_equal(result, expected)


def test_get_level_values_when_periods():
    # GH33131. See also discussion in GH32669.
    # This test can probably be removed when PeriodIndex._engine is removed.
    from pandas import (
        Period,
        PeriodIndex,
    )

    idx = MultiIndex.from_arrays(
        [PeriodIndex([Period("2019Q1"), Period("2019Q2")], name="b")]
    )
    idx2 = MultiIndex.from_arrays(
        [idx._get_level_values(level) for level in range(idx.nlevels)]
    )
    assert all(x.is_monotonic_increasing for x in idx2.levels)


def test_values_loses_freq_of_underlying_index():
    # GH#49054
    idx = pd.DatetimeIndex(date_range("20200101", periods=3, freq="BME"))
    expected = idx.copy(deep=True)
    idx2 = Index([1, 2, 3])
    midx = MultiIndex(levels=[idx, idx2], codes=[[0, 1, 2], [0, 1, 2]])
    midx.values
    assert idx.freq is not None
    tm.assert_index_equal(idx, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_orderings.py ===
"""Tests of monomial orderings. """

from sympy.polys.orderings import (
    monomial_key, lex, grlex, grevlex, ilex, igrlex,
    LexOrder, InverseOrder, ProductOrder, build_product_order,
)

from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises

def test_lex_order():
    assert lex((1, 2, 3)) == (1, 2, 3)
    assert str(lex) == 'lex'

    assert lex((1, 2, 3)) == lex((1, 2, 3))

    assert lex((2, 2, 3)) > lex((1, 2, 3))
    assert lex((1, 3, 3)) > lex((1, 2, 3))
    assert lex((1, 2, 4)) > lex((1, 2, 3))

    assert lex((0, 2, 3)) < lex((1, 2, 3))
    assert lex((1, 1, 3)) < lex((1, 2, 3))
    assert lex((1, 2, 2)) < lex((1, 2, 3))

    assert lex.is_global is True
    assert lex == LexOrder()
    assert lex != grlex

def test_grlex_order():
    assert grlex((1, 2, 3)) == (6, (1, 2, 3))
    assert str(grlex) == 'grlex'

    assert grlex((1, 2, 3)) == grlex((1, 2, 3))

    assert grlex((2, 2, 3)) > grlex((1, 2, 3))
    assert grlex((1, 3, 3)) > grlex((1, 2, 3))
    assert grlex((1, 2, 4)) > grlex((1, 2, 3))

    assert grlex((0, 2, 3)) < grlex((1, 2, 3))
    assert grlex((1, 1, 3)) < grlex((1, 2, 3))
    assert grlex((1, 2, 2)) < grlex((1, 2, 3))

    assert grlex((2, 2, 3)) > grlex((1, 2, 4))
    assert grlex((1, 3, 3)) > grlex((1, 2, 4))

    assert grlex((0, 2, 3)) < grlex((1, 2, 2))
    assert grlex((1, 1, 3)) < grlex((1, 2, 2))

    assert grlex((0, 1, 1)) > grlex((0, 0, 2))
    assert grlex((0, 3, 1)) < grlex((2, 2, 1))

    assert grlex.is_global is True

def test_grevlex_order():
    assert grevlex((1, 2, 3)) == (6, (-3, -2, -1))
    assert str(grevlex) == 'grevlex'

    assert grevlex((1, 2, 3)) == grevlex((1, 2, 3))

    assert grevlex((2, 2, 3)) > grevlex((1, 2, 3))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 3))
    assert grevlex((1, 2, 4)) > grevlex((1, 2, 3))

    assert grevlex((0, 2, 3)) < grevlex((1, 2, 3))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 3))
    assert grevlex((1, 2, 2)) < grevlex((1, 2, 3))

    assert grevlex((2, 2, 3)) > grevlex((1, 2, 4))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 4))

    assert grevlex((0, 2, 3)) < grevlex((1, 2, 2))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 2))

    assert grevlex((0, 1, 1)) > grevlex((0, 0, 2))
    assert grevlex((0, 3, 1)) < grevlex((2, 2, 1))

    assert grevlex.is_global is True

def test_InverseOrder():
    ilex = InverseOrder(lex)
    igrlex = InverseOrder(grlex)

    assert ilex((1, 2, 3)) > ilex((2, 0, 3))
    assert igrlex((1, 2, 3)) < igrlex((0, 2, 3))
    assert str(ilex) == "ilex"
    assert str(igrlex) == "igrlex"
    assert ilex.is_global is False
    assert igrlex.is_global is False
    assert ilex != igrlex
    assert ilex == InverseOrder(LexOrder())

def test_ProductOrder():
    P = ProductOrder((grlex, lambda m: m[:2]), (grlex, lambda m: m[2:]))
    assert P((1, 3, 3, 4, 5)) > P((2, 1, 5, 5, 5))
    assert str(P) == "ProductOrder(grlex, grlex)"
    assert P.is_global is True
    assert ProductOrder((grlex, None), (ilex, None)).is_global is None
    assert ProductOrder((igrlex, None), (ilex, None)).is_global is False

def test_monomial_key():
    assert monomial_key() == lex

    assert monomial_key('lex') == lex
    assert monomial_key('grlex') == grlex
    assert monomial_key('grevlex') == grevlex

    raises(ValueError, lambda: monomial_key('foo'))
    raises(ValueError, lambda: monomial_key(1))

    M = [x, x**2*z**2, x*y, x**2, S.One, y**2, x**3, y, z, x*y**2*z, x**2*y**2]
    assert sorted(M, key=monomial_key('lex', [z, y, x])) == \
        [S.One, x, x**2, x**3, y, x*y, y**2, x**2*y**2, z, x*y**2*z, x**2*z**2]
    assert sorted(M, key=monomial_key('grlex', [z, y, x])) == \
        [S.One, x, y, z, x**2, x*y, y**2, x**3, x**2*y**2, x*y**2*z, x**2*z**2]
    assert sorted(M, key=monomial_key('grevlex', [z, y, x])) == \
        [S.One, x, y, z, x**2, x*y, y**2, x**3, x**2*y**2, x**2*z**2, x*y**2*z]

def test_build_product_order():
    assert build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])((4, 5, 6, 7)) == \
        ((9, (4, 5)), (13, (6, 7)))

    assert build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t]) == \
        build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_reduction.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    array,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", np.int64(3)],
        ["prod", np.int64(2)],
        ["min", np.int64(1)],
        ["max", np.int64(2)],
        ["mean", np.float64(1.5)],
        ["median", np.float64(1.5)],
        ["var", np.float64(0.5)],
        ["std", np.float64(0.5**0.5)],
        ["skew", pd.NA],
        ["kurt", pd.NA],
        ["any", True],
        ["all", True],
    ],
)
def test_series_reductions(op, expected):
    ser = Series([1, 2], dtype="Int64")
    result = getattr(ser, op)()
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", Series([3], index=["a"], dtype="Int64")],
        ["prod", Series([2], index=["a"], dtype="Int64")],
        ["min", Series([1], index=["a"], dtype="Int64")],
        ["max", Series([2], index=["a"], dtype="Int64")],
        ["mean", Series([1.5], index=["a"], dtype="Float64")],
        ["median", Series([1.5], index=["a"], dtype="Float64")],
        ["var", Series([0.5], index=["a"], dtype="Float64")],
        ["std", Series([0.5**0.5], index=["a"], dtype="Float64")],
        ["skew", Series([pd.NA], index=["a"], dtype="Float64")],
        ["kurt", Series([pd.NA], index=["a"], dtype="Float64")],
        ["any", Series([True], index=["a"], dtype="boolean")],
        ["all", Series([True], index=["a"], dtype="boolean")],
    ],
)
def test_dataframe_reductions(op, expected):
    df = DataFrame({"a": array([1, 2], dtype="Int64")})
    result = getattr(df, op)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", array([1, 3], dtype="Int64")],
        ["prod", array([1, 3], dtype="Int64")],
        ["min", array([1, 3], dtype="Int64")],
        ["max", array([1, 3], dtype="Int64")],
        ["mean", array([1, 3], dtype="Float64")],
        ["median", array([1, 3], dtype="Float64")],
        ["var", array([pd.NA], dtype="Float64")],
        ["std", array([pd.NA], dtype="Float64")],
        ["skew", array([pd.NA], dtype="Float64")],
        ["any", array([True, True], dtype="boolean")],
        ["all", array([True, True], dtype="boolean")],
    ],
)
def test_groupby_reductions(op, expected):
    df = DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": array([1, None, 3], dtype="Int64"),
        }
    )
    result = getattr(df.groupby("A"), op)()
    expected = DataFrame(expected, index=pd.Index(["a", "b"], name="A"), columns=["B"])

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", Series([4, 4], index=["B", "C"], dtype="Float64")],
        ["prod", Series([3, 3], index=["B", "C"], dtype="Float64")],
        ["min", Series([1, 1], index=["B", "C"], dtype="Float64")],
        ["max", Series([3, 3], index=["B", "C"], dtype="Float64")],
        ["mean", Series([2, 2], index=["B", "C"], dtype="Float64")],
        ["median", Series([2, 2], index=["B", "C"], dtype="Float64")],
        ["var", Series([2, 2], index=["B", "C"], dtype="Float64")],
        ["std", Series([2**0.5, 2**0.5], index=["B", "C"], dtype="Float64")],
        ["skew", Series([pd.NA, pd.NA], index=["B", "C"], dtype="Float64")],
        ["kurt", Series([pd.NA, pd.NA], index=["B", "C"], dtype="Float64")],
        ["any", Series([True, True, True], index=["A", "B", "C"], dtype="boolean")],
        ["all", Series([True, True, True], index=["A", "B", "C"], dtype="boolean")],
    ],
)
def test_mixed_reductions(op, expected):
    df = DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": array([1, None, 3], dtype="Int64"),
        }
    )

    # series
    result = getattr(df.C, op)()
    tm.assert_equal(result, expected["C"])

    # frame
    if op in ["any", "all"]:
        result = getattr(df, op)()
    else:
        result = getattr(df, op)(numeric_only=True)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_interval.py ===
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

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import IntervalDtype

from pandas import Interval
from pandas.core.arrays import IntervalArray
from pandas.tests.extension import base

if TYPE_CHECKING:
    import pandas as pd


def make_data():
    N = 100
    left_array = np.random.default_rng(2).uniform(size=N).cumsum()
    right_array = left_array + np.random.default_rng(2).uniform(size=N)
    return [Interval(left, right) for left, right in zip(left_array, right_array)]


@pytest.fixture
def dtype():
    return IntervalDtype()


@pytest.fixture
def data():
    """Length-100 PeriodArray for semantics test."""
    return IntervalArray(make_data())


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    return IntervalArray.from_tuples([None, (0, 1)])


@pytest.fixture
def data_for_twos():
    pytest.skip("Interval is not a numeric dtype")


@pytest.fixture
def data_for_sorting():
    return IntervalArray.from_tuples([(1, 2), (2, 3), (0, 1)])


@pytest.fixture
def data_missing_for_sorting():
    return IntervalArray.from_tuples([(1, 2), None, (0, 1)])


@pytest.fixture
def data_for_grouping():
    a = (0, 1)
    b = (1, 2)
    c = (2, 3)
    return IntervalArray.from_tuples([b, b, None, None, a, a, b, c])


class TestIntervalArray(base.ExtensionTests):
    divmod_exc = TypeError

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return op_name in ["min", "max"]

    @pytest.mark.xfail(
        reason="Raises with incorrect message bc it disallows *all* listlikes "
        "instead of just wrong-length listlikes"
    )
    def test_fillna_length_mismatch(self, data_missing):
        super().test_fillna_length_mismatch(data_missing)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_hash_pandas_object(self, data):
        super().test_hash_pandas_object(data)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_hash_pandas_object_works(self, data, as_frame):
        super().test_hash_pandas_object_works(data, as_frame)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        super().test_EA_types(engine, data, request)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_astype_str(self, data):
        super().test_astype_str(data)


# TODO: either belongs in tests.arrays.interval or move into base tests.
def test_fillna_non_scalar_raises(data_missing):
    msg = "can only insert Interval objects and NA into an IntervalArray"
    with pytest.raises(TypeError, match=msg):
        data_missing.fillna([1, 1])