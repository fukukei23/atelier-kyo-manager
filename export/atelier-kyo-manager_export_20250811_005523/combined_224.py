
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_system.py ===
from sympy import symbols, Matrix, atan, zeros
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, Particle, Point,
                                     ReferenceFrame, SymbolicSystem)
from sympy.testing.pytest import raises

# This class is going to be tested using a simple pendulum set up in x and y
# coordinates
x, y, u, v, lam = dynamicsymbols('x y u v lambda')
m, l, g = symbols('m l g')

# Set up the different forms the equations can take
#       [1] Explicit form where the kinematics and dynamics are combined
#           x' = F(x, t, r, p)
#
#       [2] Implicit form where the kinematics and dynamics are combined
#           M(x, p) x' = F(x, t, r, p)
#
#       [3] Implicit form where the kinematics and dynamics are separate
#           M(q, p) u' = F(q, u, t, r, p)
#           q' = G(q, u, t, r, p)
dyn_implicit_mat = Matrix([[1, 0, -x/m],
                           [0, 1, -y/m],
                           [0, 0, l**2/m]])

dyn_implicit_rhs = Matrix([0, 0, u**2 + v**2 - g*y])

comb_implicit_mat = Matrix([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, -x/m],
                            [0, 0, 0, 1, -y/m],
                            [0, 0, 0, 0, l**2/m]])

comb_implicit_rhs = Matrix([u, v, 0, 0, u**2 + v**2 - g*y])

kin_explicit_rhs = Matrix([u, v])

comb_explicit_rhs = comb_implicit_mat.LUsolve(comb_implicit_rhs)

# Set up a body and load to pass into the system
theta = atan(x/y)
N = ReferenceFrame('N')
A = N.orientnew('A', 'Axis', [theta, N.z])
O = Point('O')
P = O.locatenew('P', l * A.x)

Pa = Particle('Pa', P, m)

bodies = [Pa]
loads = [(P, g * m * N.x)]

# Set up some output equations to be given to SymbolicSystem
# Change to make these fit the pendulum
PE = symbols("PE")
out_eqns = {PE: m*g*(l+y)}

# Set up remaining arguments that can be passed to SymbolicSystem
alg_con = [2]
alg_con_full = [4]
coordinates = (x, y, lam)
speeds = (u, v)
states = (x, y, u, v, lam)
coord_idxs = (0, 1)
speed_idxs = (2, 3)


def test_form_1():
    symsystem1 = SymbolicSystem(states, comb_explicit_rhs,
                                alg_con=alg_con_full, output_eqns=out_eqns,
                                coord_idxs=coord_idxs, speed_idxs=speed_idxs,
                                bodies=bodies, loads=loads)

    assert symsystem1.coordinates == Matrix([x, y])
    assert symsystem1.speeds == Matrix([u, v])
    assert symsystem1.states == Matrix([x, y, u, v, lam])

    assert symsystem1.alg_con == [4]

    inter = comb_explicit_rhs
    assert simplify(symsystem1.comb_explicit_rhs - inter) == zeros(5, 1)

    assert set(symsystem1.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem1.dynamic_symbols()) == tuple
    assert set(symsystem1.constant_symbols()) == {l, g, m}
    assert type(symsystem1.constant_symbols()) == tuple

    assert symsystem1.output_eqns == out_eqns

    assert symsystem1.bodies == (Pa,)
    assert symsystem1.loads == ((P, g * m * N.x),)


def test_form_2():
    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds,
                                mass_matrix=comb_implicit_mat,
                                alg_con=alg_con_full, output_eqns=out_eqns,
                                bodies=bodies, loads=loads)

    assert symsystem2.coordinates == Matrix([x, y, lam])
    assert symsystem2.speeds == Matrix([u, v])
    assert symsystem2.states == Matrix([x, y, lam, u, v])

    assert symsystem2.alg_con == [4]

    inter = comb_implicit_rhs
    assert simplify(symsystem2.comb_implicit_rhs - inter) == zeros(5, 1)
    assert simplify(symsystem2.comb_implicit_mat-comb_implicit_mat) == zeros(5)

    assert set(symsystem2.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem2.dynamic_symbols()) == tuple
    assert set(symsystem2.constant_symbols()) == {l, g, m}
    assert type(symsystem2.constant_symbols()) == tuple

    inter = comb_explicit_rhs
    symsystem2.compute_explicit_form()
    assert simplify(symsystem2.comb_explicit_rhs - inter) == zeros(5, 1)


    assert symsystem2.output_eqns == out_eqns

    assert symsystem2.bodies == (Pa,)
    assert symsystem2.loads == ((P, g * m * N.x),)


def test_form_3():
    symsystem3 = SymbolicSystem(states, dyn_implicit_rhs,
                                mass_matrix=dyn_implicit_mat,
                                coordinate_derivatives=kin_explicit_rhs,
                                alg_con=alg_con, coord_idxs=coord_idxs,
                                speed_idxs=speed_idxs, bodies=bodies,
                                loads=loads)

    assert symsystem3.coordinates == Matrix([x, y])
    assert symsystem3.speeds == Matrix([u, v])
    assert symsystem3.states == Matrix([x, y, u, v, lam])

    assert symsystem3.alg_con == [4]

    inter1 = kin_explicit_rhs
    inter2 = dyn_implicit_rhs
    assert simplify(symsystem3.kin_explicit_rhs - inter1) == zeros(2, 1)
    assert simplify(symsystem3.dyn_implicit_mat - dyn_implicit_mat) == zeros(3)
    assert simplify(symsystem3.dyn_implicit_rhs - inter2) == zeros(3, 1)

    inter = comb_implicit_rhs
    assert simplify(symsystem3.comb_implicit_rhs - inter) == zeros(5, 1)
    assert simplify(symsystem3.comb_implicit_mat-comb_implicit_mat) == zeros(5)

    inter = comb_explicit_rhs
    symsystem3.compute_explicit_form()
    assert simplify(symsystem3.comb_explicit_rhs - inter) == zeros(5, 1)

    assert set(symsystem3.dynamic_symbols()) == {y, v, lam, u, x}
    assert type(symsystem3.dynamic_symbols()) == tuple
    assert set(symsystem3.constant_symbols()) == {l, g, m}
    assert type(symsystem3.constant_symbols()) == tuple

    assert symsystem3.output_eqns == {}

    assert symsystem3.bodies == (Pa,)
    assert symsystem3.loads == ((P, g * m * N.x),)


def test_property_attributes():
    symsystem = SymbolicSystem(states, comb_explicit_rhs,
                               alg_con=alg_con_full, output_eqns=out_eqns,
                               coord_idxs=coord_idxs, speed_idxs=speed_idxs,
                               bodies=bodies, loads=loads)

    with raises(AttributeError):
        symsystem.bodies = 42
    with raises(AttributeError):
        symsystem.coordinates = 42
    with raises(AttributeError):
        symsystem.dyn_implicit_rhs = 42
    with raises(AttributeError):
        symsystem.comb_implicit_rhs = 42
    with raises(AttributeError):
        symsystem.loads = 42
    with raises(AttributeError):
        symsystem.dyn_implicit_mat = 42
    with raises(AttributeError):
        symsystem.comb_implicit_mat = 42
    with raises(AttributeError):
        symsystem.kin_explicit_rhs = 42
    with raises(AttributeError):
        symsystem.comb_explicit_rhs = 42
    with raises(AttributeError):
        symsystem.speeds = 42
    with raises(AttributeError):
        symsystem.states = 42
    with raises(AttributeError):
        symsystem.alg_con = 42


def test_not_specified_errors():
    """This test will cover errors that arise from trying to access attributes
    that were not specified upon object creation or were specified on creation
    and the user tries to recalculate them."""
    # Trying to access form 2 when form 1 given
    # Trying to access form 3 when form 2 given

    symsystem1 = SymbolicSystem(states, comb_explicit_rhs)

    with raises(AttributeError):
        symsystem1.comb_implicit_mat
    with raises(AttributeError):
        symsystem1.comb_implicit_rhs
    with raises(AttributeError):
        symsystem1.dyn_implicit_mat
    with raises(AttributeError):
        symsystem1.dyn_implicit_rhs
    with raises(AttributeError):
        symsystem1.kin_explicit_rhs
    with raises(AttributeError):
        symsystem1.compute_explicit_form()

    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds,
                                mass_matrix=comb_implicit_mat)

    with raises(AttributeError):
        symsystem2.dyn_implicit_mat
    with raises(AttributeError):
        symsystem2.dyn_implicit_rhs
    with raises(AttributeError):
        symsystem2.kin_explicit_rhs

    # Attribute error when trying to access coordinates and speeds when only the
    # states were given.
    with raises(AttributeError):
        symsystem1.coordinates
    with raises(AttributeError):
        symsystem1.speeds

    # Attribute error when trying to access bodies and loads when they are not
    # given
    with raises(AttributeError):
        symsystem1.bodies
    with raises(AttributeError):
        symsystem1.loads

    # Attribute error when trying to access comb_explicit_rhs before it was
    # calculated
    with raises(AttributeError):
        symsystem2.comb_explicit_rhs

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\merge\test_merge_ordered.py ===
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    merge_ordered,
)
import pandas._testing as tm


@pytest.fixture
def left():
    return DataFrame({"key": ["a", "c", "e"], "lvalue": [1, 2.0, 3]})


@pytest.fixture
def right():
    return DataFrame({"key": ["b", "c", "d", "f"], "rvalue": [1, 2, 3.0, 4]})


class TestMergeOrdered:
    def test_basic(self, left, right):
        result = merge_ordered(left, right, on="key")
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1, np.nan, 2, np.nan, 3, np.nan],
                "rvalue": [np.nan, 1, 2, 3, np.nan, 4],
            }
        )

        tm.assert_frame_equal(result, expected)

    def test_ffill(self, left, right):
        result = merge_ordered(left, right, on="key", fill_method="ffill")
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1.0, 1, 2, 2, 3, 3.0],
                "rvalue": [np.nan, 1, 2, 3, 3, 4],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_multigroup(self, left, right):
        left = pd.concat([left, left], ignore_index=True)

        left["group"] = ["a"] * 3 + ["b"] * 3

        result = merge_ordered(
            left, right, on="key", left_by="group", fill_method="ffill"
        )
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"] * 2,
                "lvalue": [1.0, 1, 2, 2, 3, 3.0] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3, 4] * 2,
            }
        )
        expected["group"] = ["a"] * 6 + ["b"] * 6

        tm.assert_frame_equal(result, expected.loc[:, result.columns])

        result2 = merge_ordered(
            right, left, on="key", right_by="group", fill_method="ffill"
        )
        tm.assert_frame_equal(result, result2.loc[:, result.columns])

        result = merge_ordered(left, right, on="key", left_by="group")
        assert result["group"].notna().all()

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_merge_type(self, left, right):
        class NotADataFrame(DataFrame):
            @property
            def _constructor(self):
                return NotADataFrame

        nad = NotADataFrame(left)
        result = nad.merge(right, on="key")

        assert isinstance(result, NotADataFrame)

    @pytest.mark.parametrize(
        "df_seq, pattern",
        [
            ((), "[Nn]o objects"),
            ([], "[Nn]o objects"),
            ({}, "[Nn]o objects"),
            ([None], "objects.*None"),
            ([None, None], "objects.*None"),
        ],
    )
    def test_empty_sequence_concat(self, df_seq, pattern):
        # GH 9157
        with pytest.raises(ValueError, match=pattern):
            pd.concat(df_seq)

    @pytest.mark.parametrize(
        "arg", [[DataFrame()], [None, DataFrame()], [DataFrame(), None]]
    )
    def test_empty_sequence_concat_ok(self, arg):
        pd.concat(arg)

    def test_doc_example(self):
        left = DataFrame(
            {
                "group": list("aaabbb"),
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3] * 2,
            }
        )

        right = DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})

        result = merge_ordered(left, right, fill_method="ffill", left_by="group")

        expected = DataFrame(
            {
                "group": list("aaaaabbbbb"),
                "key": ["a", "b", "c", "d", "e"] * 2,
                "lvalue": [1, 1, 2, 2, 3] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3] * 2,
            }
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left, right, on, left_by, right_by, expected",
        [
            (
                DataFrame({"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]}),
                DataFrame({"T": [2], "E": [1]}),
                ["T"],
                ["G", "H"],
                None,
                DataFrame(
                    {
                        "G": ["g"] * 3,
                        "H": ["h"] * 3,
                        "T": [1, 2, 3],
                        "E": [np.nan, 1.0, np.nan],
                    }
                ),
            ),
            (
                DataFrame({"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]}),
                DataFrame({"T": [2], "E": [1]}),
                "T",
                ["G", "H"],
                None,
                DataFrame(
                    {
                        "G": ["g"] * 3,
                        "H": ["h"] * 3,
                        "T": [1, 2, 3],
                        "E": [np.nan, 1.0, np.nan],
                    }
                ),
            ),
            (
                DataFrame({"T": [2], "E": [1]}),
                DataFrame({"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]}),
                ["T"],
                None,
                ["G", "H"],
                DataFrame(
                    {
                        "T": [1, 2, 3],
                        "E": [np.nan, 1.0, np.nan],
                        "G": ["g"] * 3,
                        "H": ["h"] * 3,
                    }
                ),
            ),
        ],
    )
    def test_list_type_by(self, left, right, on, left_by, right_by, expected):
        # GH 35269
        result = merge_ordered(
            left=left,
            right=right,
            on=on,
            left_by=left_by,
            right_by=right_by,
        )

        tm.assert_frame_equal(result, expected)

    def test_left_by_length_equals_to_right_shape0(self):
        # GH 38166
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        right = DataFrame([[2, 1]], columns=list("ET"))
        result = merge_ordered(left, right, on="E", left_by=["G", "H"])
        expected = DataFrame(
            {"G": ["g"] * 3, "H": ["h"] * 3, "E": [1, 2, 3], "T": [np.nan, 1.0, np.nan]}
        )

        tm.assert_frame_equal(result, expected)

    def test_elements_not_in_by_but_in_df(self):
        # GH 38167
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        right = DataFrame([[2, 1]], columns=list("ET"))
        msg = r"\{'h'\} not found in left columns"
        with pytest.raises(KeyError, match=msg):
            merge_ordered(left, right, on="E", left_by=["G", "h"])

    @pytest.mark.parametrize("invalid_method", ["linear", "carrot"])
    def test_ffill_validate_fill_method(self, left, right, invalid_method):
        # GH 55884
        with pytest.raises(
            ValueError, match=re.escape("fill_method must be 'ffill' or None")
        ):
            merge_ordered(left, right, on="key", fill_method=invalid_method)

    def test_ffill_left_merge(self):
        # GH 57010
        df1 = DataFrame(
            {
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3, 1, 2, 3],
                "group": ["a", "a", "a", "b", "b", "b"],
            }
        )
        df2 = DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
        result = merge_ordered(
            df1, df2, fill_method="ffill", left_by="group", how="left"
        )
        expected = DataFrame(
            {
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3, 1, 2, 3],
                "group": ["a", "a", "a", "b", "b", "b"],
                "rvalue": [np.nan, 2.0, 2.0, np.nan, 2.0, 2.0],
            }
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\moments\test_moments_consistency_rolling.py ===
import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


def no_nans(x):
    return x.notna().all().all()


def all_na(x):
    return x.isnull().all().all()


@pytest.fixture(params=[(1, 0), (5, 1)])
def rolling_consistency_cases(request):
    """window, min_periods"""
    return request.param


@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_rolling_apply_consistency_sum(
    request, all_data, rolling_consistency_cases, center, f
):
    window, min_periods = rolling_consistency_cases

    if f is np.sum:
        if not no_nans(all_data) and not (
            all_na(all_data) and not all_data.empty and min_periods > 0
        ):
            request.applymarker(
                pytest.mark.xfail(reason="np.sum has different behavior with NaNs")
            )
    rolling_f_result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).sum()
    rolling_apply_f_result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).apply(func=f, raw=True)
    tm.assert_equal(rolling_f_result, rolling_apply_f_result)


@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var(all_data, rolling_consistency_cases, center, ddof):
    window, min_periods = rolling_consistency_cases

    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )
    assert not (var_x < 0).any().any()

    if ddof == 0:
        # check that biased var(x) == mean(x^2) - mean(x)^2
        mean_x = all_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        mean_x2 = (
            (all_data * all_data)
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var_constant(
    consistent_data, rolling_consistency_cases, center, ddof
):
    window, min_periods = rolling_consistency_cases

    count_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).count()
    var_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)

    # check that variance of constant series is identically 0
    assert not (var_x > 0).any().any()
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if ddof == 1:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)


@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_var_std_cov(
    all_data, rolling_consistency_cases, center, ddof
):
    window, min_periods = rolling_consistency_cases

    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )
    assert not (var_x < 0).any().any()

    std_x = all_data.rolling(window=window, min_periods=min_periods, center=center).std(
        ddof=ddof
    )
    assert not (std_x < 0).any().any()

    # check that var(x) == std(x)^2
    tm.assert_equal(var_x, std_x * std_x)

    cov_x_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).cov(all_data, ddof=ddof)
    assert not (cov_x_x < 0).any().any()

    # check that var(x) == cov(x, x)
    tm.assert_equal(var_x, cov_x_x)


@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_series_cov_corr(
    series_data, rolling_consistency_cases, center, ddof
):
    window, min_periods = rolling_consistency_cases

    var_x_plus_y = (
        (series_data + series_data)
        .rolling(window=window, min_periods=min_periods, center=center)
        .var(ddof=ddof)
    )
    var_x = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)
    var_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)
    cov_x_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).cov(series_data, ddof=ddof)
    # check that cov(x, y) == (var(x+y) - var(x) -
    # var(y)) / 2
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

    # check that corr(x, y) == cov(x, y) / (std(x) *
    # std(y))
    corr_x_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).corr(series_data)
    std_x = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).std(ddof=ddof)
    std_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).std(ddof=ddof)
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

    if ddof == 0:
        # check that biased cov(x, y) == mean(x*y) -
        # mean(x)*mean(y)
        mean_x = series_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        mean_y = series_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        mean_x_times_y = (
            (series_data * series_data)
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))


def test_rolling_consistency_mean(all_data, rolling_consistency_cases, center):
    window, min_periods = rolling_consistency_cases

    result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).mean()
    expected = (
        all_data.rolling(window=window, min_periods=min_periods, center=center)
        .sum()
        .divide(
            all_data.rolling(
                window=window, min_periods=min_periods, center=center
            ).count()
        )
    )
    tm.assert_equal(result, expected.astype("float64"))


def test_rolling_consistency_constant(
    consistent_data, rolling_consistency_cases, center
):
    window, min_periods = rolling_consistency_cases

    count_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).count()
    mean_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).mean()
    # check that correlation of a series with itself is either 1 or NaN
    corr_x_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).corr(consistent_data)

    exp = (
        consistent_data.max()
        if isinstance(consistent_data, Series)
        else consistent_data.max().max()
    )

    # check mean of constant series
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = exp
    tm.assert_equal(mean_x, expected)

    # check correlation of constant series with itself is NaN
    expected[:] = np.nan
    tm.assert_equal(corr_x_x, expected)


def test_rolling_consistency_var_debiasing_factors(
    all_data, rolling_consistency_cases, center
):
    window, min_periods = rolling_consistency_cases

    # check variance debiasing factors
    var_unbiased_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var()
    var_biased_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=0)
    var_debiasing_factors_x = (
        all_data.rolling(window=window, min_periods=min_periods, center=center)
        .count()
        .divide(
            (
                all_data.rolling(
                    window=window, min_periods=min_periods, center=center
                ).count()
                - 1.0
            ).replace(0.0, np.nan)
        )
    )
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\moments\test_moments_consistency_ewm.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    concat,
)
import pandas._testing as tm


def create_mock_weights(obj, com, adjust, ignore_na):
    if isinstance(obj, DataFrame):
        if not len(obj.columns):
            return DataFrame(index=obj.index, columns=obj.columns)
        w = concat(
            [
                create_mock_series_weights(
                    obj.iloc[:, i], com=com, adjust=adjust, ignore_na=ignore_na
                )
                for i in range(len(obj.columns))
            ],
            axis=1,
        )
        w.index = obj.index
        w.columns = obj.columns
        return w
    else:
        return create_mock_series_weights(obj, com, adjust, ignore_na)


def create_mock_series_weights(s, com, adjust, ignore_na):
    w = Series(np.nan, index=s.index, name=s.name)
    alpha = 1.0 / (1.0 + com)
    if adjust:
        count = 0
        for i in range(len(s)):
            if s.iat[i] == s.iat[i]:
                w.iat[i] = pow(1.0 / (1.0 - alpha), count)
                count += 1
            elif not ignore_na:
                count += 1
    else:
        sum_wts = 0.0
        prev_i = -1
        count = 0
        for i in range(len(s)):
            if s.iat[i] == s.iat[i]:
                if prev_i == -1:
                    w.iat[i] = 1.0
                else:
                    w.iat[i] = alpha * sum_wts / pow(1.0 - alpha, count - prev_i)
                sum_wts += w.iat[i]
                prev_i = count
                count += 1
            elif not ignore_na:
                count += 1
    return w


def test_ewm_consistency_mean(all_data, adjust, ignore_na, min_periods):
    com = 3.0

    result = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).mean()
    weights = create_mock_weights(all_data, com=com, adjust=adjust, ignore_na=ignore_na)
    expected = all_data.multiply(weights).cumsum().divide(weights.cumsum()).ffill()
    expected[
        all_data.expanding().count() < (max(min_periods, 1) if min_periods else 1)
    ] = np.nan
    tm.assert_equal(result, expected.astype("float64"))


def test_ewm_consistency_consistent(consistent_data, adjust, ignore_na, min_periods):
    com = 3.0

    count_x = consistent_data.expanding().count()
    mean_x = consistent_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).mean()
    # check that correlation of a series with itself is either 1 or NaN
    corr_x_x = consistent_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).corr(consistent_data)
    exp = (
        consistent_data.max()
        if isinstance(consistent_data, Series)
        else consistent_data.max().max()
    )

    # check mean of constant series
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = exp
    tm.assert_equal(mean_x, expected)

    # check correlation of constant series with itself is NaN
    expected[:] = np.nan
    tm.assert_equal(corr_x_x, expected)


def test_ewm_consistency_var_debiasing_factors(
    all_data, adjust, ignore_na, min_periods
):
    com = 3.0

    # check variance debiasing factors
    var_unbiased_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=False)
    var_biased_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=True)

    weights = create_mock_weights(all_data, com=com, adjust=adjust, ignore_na=ignore_na)
    cum_sum = weights.cumsum().ffill()
    cum_sum_sq = (weights * weights).cumsum().ffill()
    numerator = cum_sum * cum_sum
    denominator = numerator - cum_sum_sq
    denominator[denominator <= 0.0] = np.nan
    var_debiasing_factors_x = numerator / denominator

    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)


@pytest.mark.parametrize("bias", [True, False])
def test_moments_consistency_var(all_data, adjust, ignore_na, min_periods, bias):
    com = 3.0

    mean_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).mean()
    var_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)
    assert not (var_x < 0).any().any()

    if bias:
        # check that biased var(x) == mean(x^2) - mean(x)^2
        mean_x2 = (
            (all_data * all_data)
            .ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na)
            .mean()
        )
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


@pytest.mark.parametrize("bias", [True, False])
def test_moments_consistency_var_constant(
    consistent_data, adjust, ignore_na, min_periods, bias
):
    com = 3.0
    count_x = consistent_data.expanding(min_periods=min_periods).count()
    var_x = consistent_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)

    # check that variance of constant series is identically 0
    assert not (var_x > 0).any().any()
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if not bias:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)


@pytest.mark.parametrize("bias", [True, False])
def test_ewm_consistency_std(all_data, adjust, ignore_na, min_periods, bias):
    com = 3.0
    var_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)
    assert not (var_x < 0).any().any()

    std_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).std(bias=bias)
    assert not (std_x < 0).any().any()

    # check that var(x) == std(x)^2
    tm.assert_equal(var_x, std_x * std_x)

    cov_x_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).cov(all_data, bias=bias)
    assert not (cov_x_x < 0).any().any()

    # check that var(x) == cov(x, x)
    tm.assert_equal(var_x, cov_x_x)


@pytest.mark.parametrize("bias", [True, False])
def test_ewm_consistency_series_cov_corr(
    series_data, adjust, ignore_na, min_periods, bias
):
    com = 3.0

    var_x_plus_y = (
        (series_data + series_data)
        .ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na)
        .var(bias=bias)
    )
    var_x = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)
    var_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)
    cov_x_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).cov(series_data, bias=bias)
    # check that cov(x, y) == (var(x+y) - var(x) -
    # var(y)) / 2
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

    # check that corr(x, y) == cov(x, y) / (std(x) *
    # std(y))
    corr_x_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).corr(series_data)
    std_x = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).std(bias=bias)
    std_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).std(bias=bias)
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

    if bias:
        # check that biased cov(x, y) == mean(x*y) -
        # mean(x)*mean(y)
        mean_x = series_data.ewm(
            com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
        ).mean()
        mean_y = series_data.ewm(
            com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
        ).mean()
        mean_x_times_y = (
            (series_data * series_data)
            .ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na)
            .mean()
        )
        tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_tag.py ===
import warnings
from bs4.element import (
    Comment,
    NavigableString,
)
from . import SoupTest


class TestTag(SoupTest):
    """Test various methods of Tag which aren't so complicated they
    need their own classes.
    """

    def test__should_pretty_print(self):
        # Test the rules about when a tag should be pretty-printed.
        tag = self.soup("").new_tag("a_tag")

        # No list of whitespace-preserving tags -> pretty-print
        tag._preserve_whitespace_tags = None
        assert True is tag._should_pretty_print(0)

        # List exists but tag is not on the list -> pretty-print
        tag.preserve_whitespace_tags = ["some_other_tag"]
        assert True is tag._should_pretty_print(1)

        # Indent level is None -> don't pretty-print
        assert False is tag._should_pretty_print(None)

        # Tag is on the whitespace-preserving list -> don't pretty-print
        tag.preserve_whitespace_tags = ["some_other_tag", "a_tag"]
        assert False is tag._should_pretty_print(1)

    def test_len(self):
        """The length of a Tag is its number of children."""
        soup = self.soup("<top>1<b>2</b>3</top>")

        # The BeautifulSoup object itself contains one element: the
        # <top> tag.
        assert len(soup.contents) == 1
        assert len(soup) == 1

        # The <top> tag contains three elements: the text node "1", the
        # <b> tag, and the text node "3".
        assert len(soup.top) == 3
        assert len(soup.top.contents) == 3

    def test_member_access_invokes_find(self):
        """Accessing a Python member .foo invokes find('foo')"""
        soup = self.soup("<b><i></i></b>")
        assert soup.b == soup.find("b")
        assert soup.b.i == soup.find("b").find("i")
        assert soup.a is None

    def test_deprecated_member_access(self):
        soup = self.soup("<b><i></i></b>")
        with warnings.catch_warnings(record=True) as w:
            tag = soup.bTag
        assert soup.b == tag
        assert (
            '.bTag is deprecated, use .find("b") instead. If you really were looking for a tag called bTag, use .find("bTag")'
            == str(w[0].message)
        )

    def test_has_attr(self):
        """has_attr() checks for the presence of an attribute.

        Please note note: has_attr() is different from
        __in__. has_attr() checks the tag's attributes and __in__
        checks the tag's chidlren.
        """
        soup = self.soup("<foo attr='bar'>")
        assert soup.foo.has_attr("attr")
        assert not soup.foo.has_attr("attr2")

    def test_attributes_come_out_in_alphabetical_order(self):
        markup = '<b a="1" z="5" m="3" f="2" y="4"></b>'
        self.assertSoupEquals(markup, '<b a="1" f="2" m="3" y="4" z="5"></b>')

    def test_string(self):
        # A Tag that contains only a text node makes that node
        # available as .string.
        soup = self.soup("<b>foo</b>")
        assert soup.b.string == "foo"

    def test_empty_tag_has_no_string(self):
        # A Tag with no children has no .stirng.
        soup = self.soup("<b></b>")
        assert soup.b.string is None

    def test_tag_with_multiple_children_has_no_string(self):
        # A Tag with no children has no .string.
        soup = self.soup("<a>foo<b></b><b></b></b>")
        assert soup.b.string is None

        soup = self.soup("<a>foo<b></b>bar</b>")
        assert soup.b.string is None

        # Even if all the children are strings, due to trickery,
        # it won't work--but this would be a good optimization.
        soup = self.soup("<a>foo</b>")
        soup.a.insert(1, "bar")
        assert soup.a.string is None

    def test_tag_with_recursive_string_has_string(self):
        # A Tag with a single child which has a .string inherits that
        # .string.
        soup = self.soup("<a><b>foo</b></a>")
        assert soup.a.string == "foo"
        assert soup.string == "foo"

    def test_lack_of_string(self):
        """Only a Tag containing a single text node has a .string."""
        soup = self.soup("<b>f<i>e</i>o</b>")
        assert soup.b.string is None

        soup = self.soup("<b></b>")
        assert soup.b.string is None

    def test_all_text(self):
        """Tag.text and Tag.get_text(sep=u"") -> all child text, concatenated"""
        soup = self.soup("<a>a<b>r</b>   <r> t </r></a>")
        assert soup.a.text == "ar  t "
        assert soup.a.get_text(strip=True) == "art"
        assert soup.a.get_text(",") == "a,r, , t "
        assert soup.a.get_text(",", strip=True) == "a,r,t"

    def test_get_text_ignores_special_string_containers(self):
        soup = self.soup("foo<!--IGNORE-->bar")
        assert soup.get_text() == "foobar"

        assert soup.get_text(types=(NavigableString, Comment)) == "fooIGNOREbar"
        assert soup.get_text(types=None) == "fooIGNOREbar"

        soup = self.soup("foo<style>CSS</style><script>Javascript</script>bar")
        assert soup.get_text() == "foobar"

    def test_all_strings_ignores_special_string_containers(self):
        soup = self.soup("foo<!--IGNORE-->bar")
        assert ["foo", "bar"] == list(soup.strings)

        soup = self.soup("foo<style>CSS</style><script>Javascript</script>bar")
        assert ["foo", "bar"] == list(soup.strings)

    def test_string_methods_inside_special_string_container_tags(self):
        # Strings inside tags like <script> are generally ignored by
        # methods like get_text, because they're not what humans
        # consider 'text'. But if you call get_text on the <script>
        # tag itself, those strings _are_ considered to be 'text',
        # because there's nothing else you might be looking for.

        style = self.soup("<div>a<style>Some CSS</style></div>")
        template = self.soup(
            "<div>a<template><p>Templated <b>text</b>.</p><!--With a comment.--></template></div>"
        )
        script = self.soup("<div>a<script><!--a comment-->Some text</script></div>")

        assert style.div.get_text() == "a"
        assert list(style.div.strings) == ["a"]
        assert style.div.style.get_text() == "Some CSS"
        assert list(style.div.style.strings) == ["Some CSS"]

        # The comment is not picked up here. That's because it was
        # parsed into a Comment object, which is not considered
        # interesting by template.strings.
        assert template.div.get_text() == "a"
        assert list(template.div.strings) == ["a"]
        assert template.div.template.get_text() == "Templated text."
        assert list(template.div.template.strings) == ["Templated ", "text", "."]

        # The comment is included here, because it didn't get parsed
        # into a Comment object--it's part of the Script string.
        assert script.div.get_text() == "a"
        assert list(script.div.strings) == ["a"]
        assert script.div.script.get_text() == "<!--a comment-->Some text"
        assert list(script.div.script.strings) == ["<!--a comment-->Some text"]


class TestMultiValuedAttributes(SoupTest):
    """Test the behavior of multi-valued attributes like 'class'.

    The values of such attributes are always presented as lists.
    """

    def test_single_value_becomes_list(self):
        soup = self.soup("<a class='foo'>")
        assert ["foo"] == soup.a["class"]

    def test_multiple_values_becomes_list(self):
        soup = self.soup("<a class='foo bar'>")
        assert ["foo", "bar"] == soup.a["class"]

    def test_multiple_values_separated_by_weird_whitespace(self):
        soup = self.soup("<a class='foo\tbar\nbaz'>")
        assert ["foo", "bar", "baz"] == soup.a["class"]

    def test_attributes_joined_into_string_on_output(self):
        soup = self.soup("<a class='foo\tbar'>")
        assert b'<a class="foo bar"></a>' == soup.a.encode()

    def test_get_attribute_list(self):
        soup = self.soup("<a id='abc def'>")
        assert ["abc def"] == soup.a.get_attribute_list("id")
        assert [] == soup.a.get_attribute_list("no such attribute")

    def test_accept_charset(self):
        soup = self.soup('<form accept-charset="ISO-8859-1 UTF-8">')
        assert ["ISO-8859-1", "UTF-8"] == soup.form["accept-charset"]

    def test_cdata_attribute_applying_only_to_one_tag(self):
        data = '<a accept-charset="ISO-8859-1 UTF-8"></a>'
        soup = self.soup(data)
        # We saw in another test that accept-charset is a cdata-list
        # attribute for the <form> tag. But it's not a cdata-list
        # attribute for any other tag.
        assert "ISO-8859-1 UTF-8" == soup.a["accept-charset"]

    def test_customization(self):
        # It's possible to change which attributes of which tags
        # are treated as multi-valued attributes.
        #
        # Here, 'id' is a multi-valued attribute and 'class' is not.
        #
        # TODO: This code is in the builder and should be tested there.
        soup = self.soup(
            '<a class="foo" id="bar">', multi_valued_attributes={"*": "id"}
        )
        assert soup.a["class"] == "foo"
        assert soup.a["id"] == ["bar"]

    def test_hidden_tag_is_invisible(self):
        # Setting .hidden on a tag makes it invisible in output, but
        # leaves its contents visible.
        #
        # This is not a documented or supported feature of Beautiful
        # Soup (e.g. NavigableString doesn't support .hidden even
        # though it could), but some people use it and it's not
        # hurting anything to verify that it keeps working.
        #
        soup = self.soup('<div id="1"><span id="2">a string</span></div>')
        soup.span.hidden = True
        assert '<div id="1">a string</div>' == str(soup.div)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_period_range.py ===
import numpy as np
import pytest

from pandas import (
    NaT,
    Period,
    PeriodIndex,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestPeriodRangeKeywords:
    def test_required_arguments(self):
        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range("2011-1-1", "2012-1-1", "B")

    def test_required_arguments2(self):
        start = Period("02-Apr-2005", "D")
        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start=start)

    def test_required_arguments3(self):
        # not enough params
        msg = (
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1")

        with pytest.raises(ValueError, match=msg):
            period_range(end="2017Q1")

        with pytest.raises(ValueError, match=msg):
            period_range(periods=5)

        with pytest.raises(ValueError, match=msg):
            period_range()

    def test_required_arguments_too_many(self):
        msg = (
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end="2018Q1", periods=8, freq="Q")

    def test_start_end_non_nat(self):
        # start/end NaT
        msg = "start and end must not be NaT"
        with pytest.raises(ValueError, match=msg):
            period_range(start=NaT, end="2018Q1")
        with pytest.raises(ValueError, match=msg):
            period_range(start=NaT, end="2018Q1", freq="Q")

        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end=NaT)
        with pytest.raises(ValueError, match=msg):
            period_range(start="2017Q1", end=NaT, freq="Q")

    def test_periods_requires_integer(self):
        # invalid periods param
        msg = "periods must be a number, got foo"
        with pytest.raises(TypeError, match=msg):
            period_range(start="2017Q1", periods="foo")


class TestPeriodRange:
    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("D", "D"),
            ("W", "W"),
            ("QE", "Q"),
            ("YE", "Y"),
        ],
    )
    def test_construction_from_string(self, freq_offset, freq_period):
        # non-empty
        expected = date_range(
            start="2017-01-01", periods=5, freq=freq_offset, name="foo"
        ).to_period()
        start, end = str(expected[0]), str(expected[-1])

        result = period_range(start=start, end=end, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=start, periods=5, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=5, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq=freq_period, name="foo")

        result = period_range(start=start, periods=0, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq=freq_period, name="foo")
        tm.assert_index_equal(result, expected)

    def test_construction_from_string_monthly(self):
        # non-empty
        expected = date_range(
            start="2017-01-01", periods=5, freq="ME", name="foo"
        ).to_period()
        start, end = str(expected[0]), str(expected[-1])

        result = period_range(start=start, end=end, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=start, periods=5, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=5, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq="M", name="foo")

        result = period_range(start=start, periods=0, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

    def test_construction_from_period(self):
        # upsampling
        start, end = Period("2017Q1", freq="Q"), Period("2018Q1", freq="Q")
        expected = date_range(
            start="2017-03-31", end="2018-03-31", freq="ME", name="foo"
        ).to_period()
        result = period_range(start=start, end=end, freq="M", name="foo")
        tm.assert_index_equal(result, expected)

        # downsampling
        start = Period("2017-1", freq="M")
        end = Period("2019-12", freq="M")
        expected = date_range(
            start="2017-01-31", end="2019-12-31", freq="QE", name="foo"
        ).to_period()
        result = period_range(start=start, end=end, freq="Q", name="foo")
        tm.assert_index_equal(result, expected)

        # test for issue # 21793
        start = Period("2017Q1", freq="Q")
        end = Period("2018Q1", freq="Q")
        idx = period_range(start=start, end=end, freq="Q", name="foo")
        result = idx == idx.values
        expected = np.array([True, True, True, True, True])
        tm.assert_numpy_array_equal(result, expected)

        # empty
        expected = PeriodIndex([], freq="W", name="foo")

        result = period_range(start=start, periods=0, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(end=end, periods=0, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

        result = period_range(start=end, end=start, freq="W", name="foo")
        tm.assert_index_equal(result, expected)

    def test_mismatched_start_end_freq_raises(self):
        depr_msg = "Period with BDay freq is deprecated"
        msg = "'w' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            end_w = Period("2006-12-31", "1w")

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            start_b = Period("02-Apr-2005", "B")
            end_b = Period("2005-05-01", "B")

        msg = "start and end must have same freq"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                period_range(start=start_b, end=end_w)

        # without mismatch we are OK
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            period_range(start=start_b, end=end_b)


class TestPeriodRangeDisallowedFreqs:
    def test_constructor_U(self):
        # U was used as undefined period
        with pytest.raises(ValueError, match="Invalid frequency: X"):
            period_range("2007-1-1", periods=500, freq="X")

    @pytest.mark.parametrize(
        "freq,freq_depr",
        [
            ("2Y", "2A"),
            ("2Y", "2a"),
            ("2Y-AUG", "2A-AUG"),
            ("2Y-AUG", "2A-aug"),
        ],
    )
    def test_a_deprecated_from_time_series(self, freq, freq_depr):
        # GH#52536
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version. Please use '{freq[1:]}' instead."

        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range(freq=freq_depr, start="1/1/2001", end="12/1/2009")

    @pytest.mark.parametrize("freq_depr", ["2H", "2MIN", "2S", "2US", "2NS"])
    def test_uppercase_freq_deprecated_from_time_series(self, freq_depr):
        # GH#52536, GH#54939
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version. Please use '{freq_depr.lower()[1:]}' instead."

        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range("2020-01-01 00:00:00 00:00", periods=2, freq=freq_depr)

    @pytest.mark.parametrize("freq_depr", ["2m", "2q-sep", "2y", "2w"])
    def test_lowercase_freq_deprecated_from_time_series(self, freq_depr):
        # GH#52536, GH#54939
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
        f"future version. Please use '{freq_depr.upper()[1:]}' instead."

        with tm.assert_produces_warning(FutureWarning, match=msg):
            period_range(freq=freq_depr, start="1/1/2001", end="12/1/2009")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_produces_warning.py ===
""""
Test module for testing ``pandas._testing.assert_produces_warning``.
"""
import warnings

import pytest

from pandas.errors import (
    DtypeWarning,
    PerformanceWarning,
)

import pandas._testing as tm


@pytest.fixture(
    params=[
        RuntimeWarning,
        ResourceWarning,
        UserWarning,
        FutureWarning,
        DeprecationWarning,
        PerformanceWarning,
        DtypeWarning,
    ],
)
def category(request):
    """
    Return unique warning.

    Useful for testing behavior of tm.assert_produces_warning with various categories.
    """
    return request.param


@pytest.fixture(
    params=[
        (RuntimeWarning, UserWarning),
        (UserWarning, FutureWarning),
        (FutureWarning, RuntimeWarning),
        (DeprecationWarning, PerformanceWarning),
        (PerformanceWarning, FutureWarning),
        (DtypeWarning, DeprecationWarning),
        (ResourceWarning, DeprecationWarning),
        (FutureWarning, DeprecationWarning),
    ],
    ids=lambda x: type(x).__name__,
)
def pair_different_warnings(request):
    """
    Return pair or different warnings.

    Useful for testing how several different warnings are handled
    in tm.assert_produces_warning.
    """
    return request.param


def f():
    warnings.warn("f1", FutureWarning)
    warnings.warn("f2", RuntimeWarning)


@pytest.mark.filterwarnings("ignore:f1:FutureWarning")
def test_assert_produces_warning_honors_filter():
    # Raise by default.
    msg = r"Caused unexpected warning\(s\)"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            f()

    with tm.assert_produces_warning(RuntimeWarning, raise_on_extra_warnings=False):
        f()


@pytest.mark.parametrize(
    "message, match",
    [
        ("", None),
        ("", ""),
        ("Warning message", r".*"),
        ("Warning message", "War"),
        ("Warning message", r"[Ww]arning"),
        ("Warning message", "age"),
        ("Warning message", r"age$"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"^Mes.*\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}\s\S+"),
        ("Message, which we do not match", None),
    ],
)
def test_catch_warning_category_and_match(category, message, match):
    with tm.assert_produces_warning(category, match=match):
        warnings.warn(message, category)


def test_fail_to_match_runtime_warning():
    category = RuntimeWarning
    match = "Did not see this warning"
    unmatched = (
        r"Did not see warning 'RuntimeWarning' matching 'Did not see this warning'. "
        r"The emitted warning messages are "
        r"\[RuntimeWarning\('This is not a match.'\), "
        r"RuntimeWarning\('Another unmatched warning.'\)\]"
    )
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn("This is not a match.", category)
            warnings.warn("Another unmatched warning.", category)


def test_fail_to_match_future_warning():
    category = FutureWarning
    match = "Warning"
    unmatched = (
        r"Did not see warning 'FutureWarning' matching 'Warning'. "
        r"The emitted warning messages are "
        r"\[FutureWarning\('This is not a match.'\), "
        r"FutureWarning\('Another unmatched warning.'\)\]"
    )
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn("This is not a match.", category)
            warnings.warn("Another unmatched warning.", category)


def test_fail_to_match_resource_warning():
    category = ResourceWarning
    match = r"\d+"
    unmatched = (
        r"Did not see warning 'ResourceWarning' matching '\\d\+'. "
        r"The emitted warning messages are "
        r"\[ResourceWarning\('This is not a match.'\), "
        r"ResourceWarning\('Another unmatched warning.'\)\]"
    )
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn("This is not a match.", category)
            warnings.warn("Another unmatched warning.", category)


def test_fail_to_catch_actual_warning(pair_different_warnings):
    expected_category, actual_category = pair_different_warnings
    match = "Did not see expected warning of class"
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn("warning message", actual_category)


def test_ignore_extra_warning(pair_different_warnings):
    expected_category, extra_category = pair_different_warnings
    with tm.assert_produces_warning(expected_category, raise_on_extra_warnings=False):
        warnings.warn("Expected warning", expected_category)
        warnings.warn("Unexpected warning OK", extra_category)


def test_raise_on_extra_warning(pair_different_warnings):
    expected_category, extra_category = pair_different_warnings
    match = r"Caused unexpected warning\(s\)"
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn("Expected warning", expected_category)
            warnings.warn("Unexpected warning NOT OK", extra_category)


def test_same_category_different_messages_first_match():
    category = UserWarning
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Match this", category)
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)


def test_same_category_different_messages_last_match():
    category = DeprecationWarning
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)
        warnings.warn("Match this", category)


def test_match_multiple_warnings():
    # https://github.com/pandas-dev/pandas/issues/47829
    category = (FutureWarning, UserWarning)
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Match this", FutureWarning)
        warnings.warn("Match this too", UserWarning)


def test_right_category_wrong_match_raises(pair_different_warnings):
    target_category, other_category = pair_different_warnings
    with pytest.raises(AssertionError, match="Did not see warning.*matching"):
        with tm.assert_produces_warning(target_category, match=r"^Match this"):
            warnings.warn("Do not match it", target_category)
            warnings.warn("Match this", other_category)


@pytest.mark.parametrize("false_or_none", [False, None])
class TestFalseOrNoneExpectedWarning:
    def test_raise_on_warning(self, false_or_none):
        msg = r"Caused unexpected warning\(s\)"
        with pytest.raises(AssertionError, match=msg):
            with tm.assert_produces_warning(false_or_none):
                f()

    def test_no_raise_without_warning(self, false_or_none):
        with tm.assert_produces_warning(false_or_none):
            pass

    def test_no_raise_with_false_raise_on_extra(self, false_or_none):
        with tm.assert_produces_warning(false_or_none, raise_on_extra_warnings=False):
            f()


def test_raises_during_exception():
    msg = "Did not see expected warning of class 'UserWarning'"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            raise ValueError

    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(UserWarning):
            warnings.warn("FutureWarning", FutureWarning)
            raise IndexError

    msg = "Caused unexpected warning"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(None):
            warnings.warn("FutureWarning", FutureWarning)
            raise SystemError


def test_passes_during_exception():
    with pytest.raises(SyntaxError, match="Error"):
        with tm.assert_produces_warning(None):
            raise SyntaxError("Error")

    with pytest.raises(ValueError, match="Error"):
        with tm.assert_produces_warning(FutureWarning, match="FutureWarning"):
            warnings.warn("FutureWarning", FutureWarning)
            raise ValueError("Error")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\__init__.py ===
# -*- coding: utf-8 -*-
"""
Tests for greenlet.

"""
import os
import sys
import unittest

from gc import collect
from gc import get_objects
from threading import active_count as active_thread_count
from time import sleep
from time import time

import psutil

from greenlet import greenlet as RawGreenlet
from greenlet import getcurrent

from greenlet._greenlet import get_pending_cleanup_count
from greenlet._greenlet import get_total_main_greenlets

from . import leakcheck

PY312 = sys.version_info[:2] >= (3, 12)
PY313 = sys.version_info[:2] >= (3, 13)
# XXX: First tested on 3.14a7. Revisit all uses of this on later versions to ensure they
# are still valid.
PY314 = sys.version_info[:2] >= (3, 14)

WIN = sys.platform.startswith("win")
RUNNING_ON_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS')
RUNNING_ON_TRAVIS = os.environ.get('TRAVIS') or RUNNING_ON_GITHUB_ACTIONS
RUNNING_ON_APPVEYOR = os.environ.get('APPVEYOR')
RUNNING_ON_CI = RUNNING_ON_TRAVIS or RUNNING_ON_APPVEYOR
RUNNING_ON_MANYLINUX = os.environ.get('GREENLET_MANYLINUX')

class TestCaseMetaClass(type):
    # wrap each test method with
    # a) leak checks
    def __new__(cls, classname, bases, classDict):
        # pylint and pep8 fight over what this should be called (mcs or cls).
        # pylint gets it right, but we can't scope disable pep8, so we go with
        # its convention.
        # pylint: disable=bad-mcs-classmethod-argument
        check_totalrefcount = True

        # Python 3: must copy, we mutate the classDict. Interestingly enough,
        # it doesn't actually error out, but under 3.6 we wind up wrapping
        # and re-wrapping the same items over and over and over.
        for key, value in list(classDict.items()):
            if key.startswith('test') and callable(value):
                classDict.pop(key)
                if check_totalrefcount:
                    value = leakcheck.wrap_refcount(value)
                classDict[key] = value
        return type.__new__(cls, classname, bases, classDict)


class TestCase(unittest.TestCase, metaclass=TestCaseMetaClass):

    cleanup_attempt_sleep_duration = 0.001
    cleanup_max_sleep_seconds = 1

    def wait_for_pending_cleanups(self,
                                  initial_active_threads=None,
                                  initial_main_greenlets=None):
        initial_active_threads = initial_active_threads or self.threads_before_test
        initial_main_greenlets = initial_main_greenlets or self.main_greenlets_before_test
        sleep_time = self.cleanup_attempt_sleep_duration
        # NOTE: This is racy! A Python-level thread object may be dead
        # and gone, but the C thread may not yet have fired its
        # destructors and added to the queue. There's no particular
        # way to know that's about to happen. We try to watch the
        # Python threads to make sure they, at least, have gone away.
        # Counting the main greenlets, which we can easily do deterministically,
        # also helps.

        # Always sleep at least once to let other threads run
        sleep(sleep_time)
        quit_after = time() + self.cleanup_max_sleep_seconds
        # TODO: We could add an API that calls us back when a particular main greenlet is deleted?
        # It would have to drop the GIL
        while (
                get_pending_cleanup_count()
                or active_thread_count() > initial_active_threads
                or (not self.expect_greenlet_leak
                    and get_total_main_greenlets() > initial_main_greenlets)):
            sleep(sleep_time)
            if time() > quit_after:
                print("Time limit exceeded.")
                print("Threads: Waiting for only", initial_active_threads,
                      "-->", active_thread_count())
                print("MGlets : Waiting for only", initial_main_greenlets,
                      "-->", get_total_main_greenlets())
                break
        collect()

    def count_objects(self, kind=list, exact_kind=True):
        # pylint:disable=unidiomatic-typecheck
        # Collect the garbage.
        for _ in range(3):
            collect()
        if exact_kind:
            return sum(
                1
                for x in get_objects()
                if type(x) is kind
            )
        # instances
        return sum(
            1
            for x in get_objects()
            if isinstance(x, kind)
        )

    greenlets_before_test = 0
    threads_before_test = 0
    main_greenlets_before_test = 0
    expect_greenlet_leak = False

    def count_greenlets(self):
        """
        Find all the greenlets and subclasses tracked by the GC.
        """
        return self.count_objects(RawGreenlet, False)

    def setUp(self):
        # Ensure the main greenlet exists, otherwise the first test
        # gets a false positive leak
        super().setUp()
        getcurrent()
        self.threads_before_test = active_thread_count()
        self.main_greenlets_before_test = get_total_main_greenlets()
        self.wait_for_pending_cleanups(self.threads_before_test, self.main_greenlets_before_test)
        self.greenlets_before_test = self.count_greenlets()

    def tearDown(self):
        if getattr(self, 'skipTearDown', False):
            return

        self.wait_for_pending_cleanups(self.threads_before_test, self.main_greenlets_before_test)
        super().tearDown()

    def get_expected_returncodes_for_aborted_process(self):
        import signal
        # The child should be aborted in an unusual way. On POSIX
        # platforms, this is done with abort() and signal.SIGABRT,
        # which is reflected in a negative return value; however, on
        # Windows, even though we observe the child print "Fatal
        # Python error: Aborted" and in older versions of the C
        # runtime "This application has requested the Runtime to
        # terminate it in an unusual way," it always has an exit code
        # of 3. This is interesting because 3 is the error code for
        # ERROR_PATH_NOT_FOUND; BUT: the C runtime abort() function
        # also uses this code.
        #
        # If we link to the static C library on Windows, the error
        # code changes to '0xc0000409' (hex(3221226505)), which
        # apparently is STATUS_STACK_BUFFER_OVERRUN; but "What this
        # means is that nowadays when you get a
        # STATUS_STACK_BUFFER_OVERRUN, it doesn’t actually mean that
        # there is a stack buffer overrun. It just means that the
        # application decided to terminate itself with great haste."
        #
        #
        # On windows, we've also seen '0xc0000005' (hex(3221225477)).
        # That's "Access Violation"
        #
        # See
        # https://devblogs.microsoft.com/oldnewthing/20110519-00/?p=10623
        # and
        # https://docs.microsoft.com/en-us/previous-versions/k089yyh0(v=vs.140)?redirectedfrom=MSDN
        # and
        # https://devblogs.microsoft.com/oldnewthing/20190108-00/?p=100655
        expected_exit = (
            -signal.SIGABRT,
            # But beginning on Python 3.11, the faulthandler
            # that prints the C backtraces sometimes segfaults after
            # reporting the exception but before printing the stack.
            # This has only been seen on linux/gcc.
            -signal.SIGSEGV,
        ) if not WIN else (
            3,
            0xc0000409,
            0xc0000005,
        )
        return expected_exit

    def get_process_uss(self):
        """
        Return the current process's USS in bytes.

        uss is available on Linux, macOS, Windows. Also known as
        "Unique Set Size", this is the memory which is unique to a
        process and which would be freed if the process was terminated
        right now.

        If this is not supported by ``psutil``, this raises the
        :exc:`unittest.SkipTest` exception.
        """
        try:
            return psutil.Process().memory_full_info().uss
        except AttributeError as e:
            raise unittest.SkipTest("uss not supported") from e

    def run_script(self, script_name, show_output=True):
        import subprocess
        script = os.path.join(
            os.path.dirname(__file__),
            script_name,
        )

        try:
            return subprocess.check_output([sys.executable, script],
                                           encoding='utf-8',
                                           stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            if show_output:
                print('-----')
                print('Failed to run script', script)
                print('~~~~~')
                print(ex.output)
                print('------')
            raise


    def assertScriptRaises(self, script_name, exitcodes=None):
        import subprocess
        with self.assertRaises(subprocess.CalledProcessError) as exc:
            output = self.run_script(script_name, show_output=False)
            __traceback_info__ = output
            # We're going to fail the assertion if we get here, at least
            # preserve the output in the traceback.

        if exitcodes is None:
            exitcodes = self.get_expected_returncodes_for_aborted_process()
        self.assertIn(exc.exception.returncode, exitcodes)
        return exc.exception

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\test_masked_matrix.py ===
import pickle

import numpy as np
from numpy.ma.core import (
    MaskedArray,
    MaskType,
    add,
    allequal,
    divide,
    getmask,
    hypot,
    log,
    masked,
    masked_array,
    masked_values,
    nomask,
)
from numpy.ma.extras import mr_
from numpy.ma.testutils import assert_, assert_array_equal, assert_equal, assert_raises


class MMatrix(MaskedArray, np.matrix,):

    def __new__(cls, data, mask=nomask):
        mat = np.matrix(data)
        _data = MaskedArray.__new__(cls, data=mat, mask=mask)
        return _data

    def __array_finalize__(self, obj):
        np.matrix.__array_finalize__(self, obj)
        MaskedArray.__array_finalize__(self, obj)

    @property
    def _series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view


class TestMaskedMatrix:
    def test_matrix_indexing(self):
        # Tests conversions and indexing
        x1 = np.matrix([[1, 2, 3], [4, 3, 2]])
        x2 = masked_array(x1, mask=[[1, 0, 0], [0, 1, 0]])
        x3 = masked_array(x1, mask=[[0, 1, 0], [1, 0, 0]])
        x4 = masked_array(x1)
        # test conversion to strings
        str(x2)  # raises?
        repr(x2)  # raises?
        # tests of indexing
        assert_(type(x2[1, 0]) is type(x1[1, 0]))
        assert_(x1[1, 0] == x2[1, 0])
        assert_(x2[1, 1] is masked)
        assert_equal(x1[0, 2], x2[0, 2])
        assert_equal(x1[0, 1:], x2[0, 1:])
        assert_equal(x1[:, 2], x2[:, 2])
        assert_equal(x1[:], x2[:])
        assert_equal(x1[1:], x3[1:])
        x1[0, 2] = 9
        x2[0, 2] = 9
        assert_equal(x1, x2)
        x1[0, 1:] = 99
        x2[0, 1:] = 99
        assert_equal(x1, x2)
        x2[0, 1] = masked
        assert_equal(x1, x2)
        x2[0, 1:] = masked
        assert_equal(x1, x2)
        x2[0, :] = x1[0, :]
        x2[0, 1] = masked
        assert_(allequal(getmask(x2), np.array([[0, 1, 0], [0, 1, 0]])))
        x3[1, :] = masked_array([1, 2, 3], [1, 1, 0])
        assert_(allequal(getmask(x3)[1], masked_array([1, 1, 0])))
        assert_(allequal(getmask(x3[1]), masked_array([1, 1, 0])))
        x4[1, :] = masked_array([1, 2, 3], [1, 1, 0])
        assert_(allequal(getmask(x4[1]), masked_array([1, 1, 0])))
        assert_(allequal(x4[1], masked_array([1, 2, 3])))
        x1 = np.matrix(np.arange(5) * 1.0)
        x2 = masked_values(x1, 3.0)
        assert_equal(x1, x2)
        assert_(allequal(masked_array([0, 0, 0, 1, 0], dtype=MaskType),
                         x2.mask))
        assert_equal(3.0, x2.fill_value)

    def test_pickling_subbaseclass(self):
        # Test pickling w/ a subclass of ndarray
        a = masked_array(np.matrix(list(range(10))), mask=[1, 0, 1, 0, 0] * 2)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            assert_equal(a_pickled._mask, a._mask)
            assert_equal(a_pickled, a)
            assert_(isinstance(a_pickled._data, np.matrix))

    def test_count_mean_with_matrix(self):
        m = masked_array(np.matrix([[1, 2], [3, 4]]), mask=np.zeros((2, 2)))

        assert_equal(m.count(axis=0).shape, (1, 2))
        assert_equal(m.count(axis=1).shape, (2, 1))

        # Make sure broadcasting inside mean and var work
        assert_equal(m.mean(axis=0), [[2., 3.]])
        assert_equal(m.mean(axis=1), [[1.5], [3.5]])

    def test_flat(self):
        # Test that flat can return items even for matrices [#4585, #4615]
        # test simple access
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        assert_equal(test.flat[1], 2)
        assert_equal(test.flat[2], masked)
        assert_(np.all(test.flat[0:2] == test[0, 0:2]))
        # Test flat on masked_matrices
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        test.flat = masked_array([3, 2, 1], mask=[1, 0, 0])
        control = masked_array(np.matrix([[3, 2, 1]]), mask=[1, 0, 0])
        assert_equal(test, control)
        # Test setting
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        testflat = test.flat
        testflat[:] = testflat[[2, 1, 0]]
        assert_equal(test, control)
        testflat[0] = 9
        # test that matrices keep the correct shape (#4615)
        a = masked_array(np.matrix(np.eye(2)), mask=0)
        b = a.flat
        b01 = b[:2]
        assert_equal(b01.data, np.array([[1., 0.]]))
        assert_equal(b01.mask, np.array([[False, False]]))

    def test_allany_onmatrices(self):
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        X = np.matrix(x)
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool)
        mX = masked_array(X, mask=m)
        mXbig = (mX > 0.5)
        mXsmall = (mX < 0.5)

        assert_(not mXbig.all())
        assert_(mXbig.any())
        assert_equal(mXbig.all(0), np.matrix([False, False, True]))
        assert_equal(mXbig.all(1), np.matrix([False, False, True]).T)
        assert_equal(mXbig.any(0), np.matrix([False, False, True]))
        assert_equal(mXbig.any(1), np.matrix([True, True, True]).T)

        assert_(not mXsmall.all())
        assert_(mXsmall.any())
        assert_equal(mXsmall.all(0), np.matrix([True, True, False]))
        assert_equal(mXsmall.all(1), np.matrix([False, False, False]).T)
        assert_equal(mXsmall.any(0), np.matrix([True, True, False]))
        assert_equal(mXsmall.any(1), np.matrix([True, True, False]).T)

    def test_compressed(self):
        a = masked_array(np.matrix([1, 2, 3, 4]), mask=[0, 0, 0, 0])
        b = a.compressed()
        assert_equal(b, a)
        assert_(isinstance(b, np.matrix))
        a[0, 0] = masked
        b = a.compressed()
        assert_equal(b, [[2, 3, 4]])

    def test_ravel(self):
        a = masked_array(np.matrix([1, 2, 3, 4, 5]), mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        assert_equal(aravel.shape, (1, 5))
        assert_equal(aravel._mask.shape, a.shape)

    def test_view(self):
        # Test view w/ flexible dtype
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        a = masked_array(iterator, dtype=[('a', float), ('b', float)])
        a.mask[0] = (1, 0)
        test = a.view((float, 2), np.matrix)
        assert_equal(test, data)
        assert_(isinstance(test, np.matrix))
        assert_(not isinstance(test, MaskedArray))


class TestSubclassing:
    # Test suite for masked subclasses of ndarray.

    def setup_method(self):
        x = np.arange(5, dtype='float')
        mx = MMatrix(x, mask=[0, 1, 0, 0, 0])
        self.data = (x, mx)

    def test_maskedarray_subclassing(self):
        # Tests subclassing MaskedArray
        (x, mx) = self.data
        assert_(isinstance(mx._data, np.matrix))

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (x, mx) = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(log(mx), MMatrix))
            assert_equal(log(x), np.log(x))

    def test_masked_binary_operations(self):
        # Tests masked_binary_operation
        (x, mx) = self.data
        # Result should be a MMatrix
        assert_(isinstance(add(mx, mx), MMatrix))
        assert_(isinstance(add(mx, x), MMatrix))
        # Result should work
        assert_equal(add(mx, x), mx + x)
        assert_(isinstance(add(mx, mx)._data, np.matrix))
        with assert_raises(TypeError):
            add.outer(mx, mx)
        assert_(isinstance(hypot(mx, mx), MMatrix))
        assert_(isinstance(hypot(mx, x), MMatrix))

    def test_masked_binary_operations2(self):
        # Tests domained_masked_binary_operation
        (x, mx) = self.data
        xmx = masked_array(mx.data.__array__(), mask=mx.mask)
        assert_(isinstance(divide(mx, mx), MMatrix))
        assert_(isinstance(divide(mx, x), MMatrix))
        assert_equal(divide(mx, mx), divide(xmx, xmx))

class TestConcatenator:
    # Tests for mr_, the equivalent of r_ for masked arrays.

    def test_matrix_builder(self):
        assert_raises(np.ma.MAError, lambda: mr_['1, 2; 3, 4'])

    def test_matrix(self):
        # Test consistency with unmasked version.  If we ever deprecate
        # matrix, this test should either still pass, or both actual and
        # expected should fail to be build.
        actual = mr_['r', 1, 2, 3]
        expected = np.ma.array(np.r_['r', 1, 2, 3])
        assert_array_equal(actual, expected)

        # outer type is masked array, inner type is matrix
        assert_equal(type(actual), type(expected))
        assert_equal(type(actual.data), type(expected.data))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\floating\test_arithmetic.py ===
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray

# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opname, exp",
    [
        ("add", [1.1, 2.2, None, None, 5.5]),
        ("mul", [0.1, 0.4, None, None, 2.5]),
        ("sub", [0.9, 1.8, None, None, 4.5]),
        ("truediv", [10.0, 10.0, None, None, 10.0]),
        ("floordiv", [9.0, 9.0, None, None, 10.0]),
        ("mod", [0.1, 0.2, None, None, 0.0]),
    ],
    ids=["add", "mul", "sub", "div", "floordiv", "mod"],
)
def test_array_op(dtype, opname, exp):
    a = pd.array([1.0, 2.0, None, 4.0, 5.0], dtype=dtype)
    b = pd.array([0.1, 0.2, 0.3, None, 0.5], dtype=dtype)

    op = getattr(operator, opname)

    result = op(a, b)
    expected = pd.array(exp, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("zero, negative", [(0, False), (0.0, False), (-0.0, True)])
def test_divide_by_zero(dtype, zero, negative):
    # TODO pending NA/NaN discussion
    # https://github.com/pandas-dev/pandas/issues/32265/
    a = pd.array([0, 1, -1, None], dtype=dtype)
    result = a / zero
    expected = FloatingArray(
        np.array([np.nan, np.inf, -np.inf, np.nan], dtype=dtype.numpy_dtype),
        np.array([False, False, False, True]),
    )
    if negative:
        expected *= -1
    tm.assert_extension_array_equal(result, expected)


def test_pow_scalar(dtype):
    a = pd.array([-1, 0, 1, None, 2], dtype=dtype)
    result = a**0
    expected = pd.array([1, 1, 1, 1, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = a**1
    expected = pd.array([-1, 0, 1, None, 2], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = a**pd.NA
    expected = pd.array([None, None, 1, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = a**np.nan
    # TODO np.nan should be converted to pd.NA / missing before operation?
    expected = FloatingArray(
        np.array([np.nan, np.nan, 1, np.nan, np.nan], dtype=dtype.numpy_dtype),
        mask=a._mask,
    )
    tm.assert_extension_array_equal(result, expected)

    # reversed
    a = a[1:]  # Can't raise integers to negative powers.

    result = 0**a
    expected = pd.array([1, 0, None, 0], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = 1**a
    expected = pd.array([1, 1, 1, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = pd.NA**a
    expected = pd.array([1, None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = np.nan**a
    expected = FloatingArray(
        np.array([1, np.nan, np.nan, np.nan], dtype=dtype.numpy_dtype), mask=a._mask
    )
    tm.assert_extension_array_equal(result, expected)


def test_pow_array(dtype):
    a = pd.array([0, 0, 0, 1, 1, 1, None, None, None], dtype=dtype)
    b = pd.array([0, 1, None, 0, 1, None, 0, 1, None], dtype=dtype)
    result = a**b
    expected = pd.array([1, 0, None, 1, 1, 1, 1, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_rpow_one_to_na():
    # https://github.com/pandas-dev/pandas/issues/22022
    # https://github.com/pandas-dev/pandas/issues/29997
    arr = pd.array([np.nan, np.nan], dtype="Float64")
    result = np.array([1.0, 2.0]) ** arr
    expected = pd.array([1.0, np.nan], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("other", [0, 0.5])
def test_arith_zero_dim_ndarray(other):
    arr = pd.array([1, None, 2], dtype="Float64")
    result = arr + np.array(other)
    expected = arr + other
    tm.assert_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_values(data, all_arithmetic_operators):
    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)

    # invalid scalars
    msg = "|".join(
        [
            r"can only perform ops with numeric values",
            r"FloatingArray cannot perform the operation mod",
            "unsupported operand type",
            "not all arguments converted during string formatting",
            "can't multiply sequence by non-int of type 'float'",
            "ufunc 'subtract' cannot use operands with types dtype",
            r"can only concatenate str \(not \"float\"\) to str",
            "ufunc '.*' not supported for the input types, and the inputs could not",
            "ufunc '.*' did not contain a loop with signature matching types",
            "Concatenation operation is not implemented for NumPy arrays",
            "has no kernel",
            "not implemented",
            "not supported for dtype",
            "Can only string multiply by an integer",
        ]
    )
    with pytest.raises(TypeError, match=msg):
        ops("foo")
    with pytest.raises(TypeError, match=msg):
        ops(pd.Timestamp("20180101"))

    # invalid array-likes
    with pytest.raises(TypeError, match=msg):
        ops(pd.Series("foo", index=s.index))

    msg = "|".join(
        [
            "can only perform ops with numeric values",
            "cannot perform .* with this index type: DatetimeArray",
            "Addition/subtraction of integers and integer-arrays "
            "with DatetimeArray is no longer supported. *",
            "unsupported operand type",
            "not all arguments converted during string formatting",
            "can't multiply sequence by non-int of type 'float'",
            "ufunc 'subtract' cannot use operands with types dtype",
            (
                "ufunc 'add' cannot use operands with types "
                rf"dtype\('{tm.ENDIAN}M8\[ns\]'\)"
            ),
            r"ufunc 'add' cannot use operands with types dtype\('float\d{2}'\)",
            "cannot subtract DatetimeArray from ndarray",
            "has no kernel",
            "not implemented",
            "not supported for dtype",
        ]
    )
    with pytest.raises(TypeError, match=msg):
        ops(pd.Series(pd.date_range("20180101", periods=len(s))))


# Various
# -----------------------------------------------------------------------------


def test_cross_type_arithmetic():
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, np.nan], dtype="Float64"),
            "B": pd.array([1, np.nan, 3], dtype="Float32"),
            "C": np.array([1, 2, 3], dtype="float64"),
        }
    )

    result = df.A + df.C
    expected = pd.Series([2, 4, np.nan], dtype="Float64")
    tm.assert_series_equal(result, expected)

    result = (df.A + df.C) * 3 == 12
    expected = pd.Series([False, True, None], dtype="boolean")
    tm.assert_series_equal(result, expected)

    result = df.A + df.B
    expected = pd.Series([2, np.nan, np.nan], dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "source, neg_target, abs_target",
    [
        ([1.1, 2.2, 3.3], [-1.1, -2.2, -3.3], [1.1, 2.2, 3.3]),
        ([1.1, 2.2, None], [-1.1, -2.2, None], [1.1, 2.2, None]),
        ([-1.1, 0.0, 1.1], [1.1, 0.0, -1.1], [1.1, 0.0, 1.1]),
    ],
)
def test_unary_float_operators(float_ea_dtype, source, neg_target, abs_target):
    # GH38794
    dtype = float_ea_dtype
    arr = pd.array(source, dtype=dtype)
    neg_result, pos_result, abs_result = -arr, +arr, abs(arr)
    neg_target = pd.array(neg_target, dtype=dtype)
    abs_target = pd.array(abs_target, dtype=dtype)

    tm.assert_extension_array_equal(neg_result, neg_target)
    tm.assert_extension_array_equal(pos_result, arr)
    assert not tm.shares_memory(pos_result, arr)
    tm.assert_extension_array_equal(abs_result, abs_target)


def test_bitwise(dtype):
    left = pd.array([1, None, 3, 4], dtype=dtype)
    right = pd.array([None, 3, 5, 4], dtype=dtype)

    with pytest.raises(TypeError, match="unsupported operand type"):
        left | right
    with pytest.raises(TypeError, match="unsupported operand type"):
        left & right
    with pytest.raises(TypeError, match="unsupported operand type"):
        left ^ right

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_pde.py ===
from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
    pdsolve, classify_pde, checkpdesol)
from sympy.testing.pytest import raises


a, b, c, x, y = symbols('a b c x y')

def test_pde_separate_add():
    x, y, z, t = symbols("x,y,z,t")
    F, T, X, Y, Z, u = map(Function, 'FTXYZu')

    eq = Eq(D(u(x, t), x), D(u(x, t), t)*exp(u(x, t)))
    res = pde_separate_add(eq, u(x, t), [X(x), T(t)])
    assert res == [D(X(x), x)*exp(-X(x)), D(T(t), t)*exp(T(t))]


def test_pde_separate():
    x, y, z, t = symbols("x,y,z,t")
    F, T, X, Y, Z, u = map(Function, 'FTXYZu')

    eq = Eq(D(u(x, t), x), D(u(x, t), t)*exp(u(x, t)))
    raises(ValueError, lambda: pde_separate(eq, u(x, t), [X(x), T(t)], 'div'))


def test_pde_separate_mul():
    x, y, z, t = symbols("x,y,z,t")
    c = Symbol("C", real=True)
    Phi = Function('Phi')
    F, R, T, X, Y, Z, u = map(Function, 'FRTXYZu')
    r, theta, z = symbols('r,theta,z')

    # Something simple :)
    eq = Eq(D(F(x, y, z), x) + D(F(x, y, z), y) + D(F(x, y, z), z), 0)

    # Duplicate arguments in functions
    raises(
        ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), u(z, z)]))
    # Wrong number of arguments
    raises(ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), Y(y)]))
    # Wrong variables: [x, y] -> [x, z]
    raises(
        ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(t), Y(x, y)]))

    assert pde_separate_mul(eq, F(x, y, z), [Y(y), u(x, z)]) == \
        [D(Y(y), y)/Y(y), -D(u(x, z), x)/u(x, z) - D(u(x, z), z)/u(x, z)]
    assert pde_separate_mul(eq, F(x, y, z), [X(x), Y(y), Z(z)]) == \
        [D(X(x), x)/X(x), -D(Z(z), z)/Z(z) - D(Y(y), y)/Y(y)]

    # wave equation
    wave = Eq(D(u(x, t), t, t), c**2*D(u(x, t), x, x))
    res = pde_separate_mul(wave, u(x, t), [X(x), T(t)])
    assert res == [D(X(x), x, x)/X(x), D(T(t), t, t)/(c**2*T(t))]

    # Laplace equation in cylindrical coords
    eq = Eq(1/r * D(Phi(r, theta, z), r) + D(Phi(r, theta, z), r, 2) +
            1/r**2 * D(Phi(r, theta, z), theta, 2) + D(Phi(r, theta, z), z, 2), 0)
    # Separate z
    res = pde_separate_mul(eq, Phi(r, theta, z), [Z(z), u(theta, r)])
    assert res == [D(Z(z), z, z)/Z(z),
            -D(u(theta, r), r, r)/u(theta, r) -
        D(u(theta, r), r)/(r*u(theta, r)) -
        D(u(theta, r), theta, theta)/(r**2*u(theta, r))]
    # Lets use the result to create a new equation...
    eq = Eq(res[1], c)
    # ...and separate theta...
    res = pde_separate_mul(eq, u(theta, r), [T(theta), R(r)])
    assert res == [D(T(theta), theta, theta)/T(theta),
            -r*D(R(r), r)/R(r) - r**2*D(R(r), r, r)/R(r) - c*r**2]
    # ...or r...
    res = pde_separate_mul(eq, u(theta, r), [R(r), T(theta)])
    assert res == [r*D(R(r), r)/R(r) + r**2*D(R(r), r, r)/R(r) + c*r**2,
            -D(T(theta), theta, theta)/T(theta)]


def test_issue_11726():
    x, t = symbols("x t")
    f  = symbols("f", cls=Function)
    X, T = symbols("X T", cls=Function)

    u = f(x, t)
    eq = u.diff(x, 2) - u.diff(t, 2)
    res = pde_separate(eq, u, [T(x), X(t)])
    assert res == [D(T(x), x, x)/T(x),D(X(t), t, t)/X(t)]


def test_pde_classify():
    # When more number of hints are added, add tests for classifying here.
    f = Function('f')
    eq1 = a*f(x,y) + b*f(x,y).diff(x) + c*f(x,y).diff(y)
    eq2 = 3*f(x,y) + 2*f(x,y).diff(x) + f(x,y).diff(y)
    eq3 = a*f(x,y) + b*f(x,y).diff(x) + 2*f(x,y).diff(y)
    eq4 = x*f(x,y) + f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq5 = x**2*f(x,y) + x*f(x,y).diff(x) + x*y*f(x,y).diff(y)
    eq6 = y*x**2*f(x,y) + y*f(x,y).diff(x) + f(x,y).diff(y)
    for eq in [eq1, eq2, eq3]:
        assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    for eq in [eq4, eq5, eq6]:
        assert classify_pde(eq) == ('1st_linear_variable_coeff',)


def test_checkpdesol():
    f, F = map(Function, ['f', 'F'])
    eq1 = a*f(x,y) + b*f(x,y).diff(x) + c*f(x,y).diff(y)
    eq2 = 3*f(x,y) + 2*f(x,y).diff(x) + f(x,y).diff(y)
    eq3 = a*f(x,y) + b*f(x,y).diff(x) + 2*f(x,y).diff(y)
    for eq in [eq1, eq2, eq3]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    eq4 = x*f(x,y) + f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq5 = 2*f(x,y) + 1*f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq6 = f(x,y) + 1*f(x,y).diff(x) + 3*f(x,y).diff(y)
    assert checkpdesol(eq4, [pdsolve(eq5), pdsolve(eq6)]) == [
        (False, (x - 2)*F(3*x - y)*exp(-x/S(5) - 3*y/S(5))),
         (False, (x - 1)*F(3*x - y)*exp(-x/S(10) - 3*y/S(10)))]
    for eq in [eq4, eq5, eq6]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    sol = pdsolve(eq4)
    sol4 = Eq(sol.lhs - sol.rhs, 0)
    raises(NotImplementedError, lambda:
        checkpdesol(eq4, sol4, solve_for_func=False))


def test_solvefun():
    f, F, G, H = map(Function, ['f', 'F', 'G', 'H'])
    eq1 = f(x,y) + f(x,y).diff(x) + f(x,y).diff(y)
    assert pdsolve(eq1) == Eq(f(x, y), F(x - y)*exp(-x/2 - y/2))
    assert pdsolve(eq1, solvefun=G) == Eq(f(x, y), G(x - y)*exp(-x/2 - y/2))
    assert pdsolve(eq1, solvefun=H) == Eq(f(x, y), H(x - y)*exp(-x/2 - y/2))


def test_pde_1st_linear_constant_coeff_homogeneous():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = 2*u + u.diff(x) + u.diff(y)
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(x - y)*exp(-x - y))
    assert checkpdesol(eq, sol)[0]

    eq = 4 + (3*u.diff(x)/u) + (2*u.diff(y)/u)
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(2*x - 3*y)*exp(-S(12)*x/13 - S(8)*y/13))
    assert checkpdesol(eq, sol)[0]

    eq = u + (6*u.diff(x)) + (7*u.diff(y))
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(7*x - 6*y)*exp(-6*x/S(85) - 7*y/S(85)))
    assert checkpdesol(eq, sol)[0]

    eq = a*u + b*u.diff(x) + c*u.diff(y)
    sol = pdsolve(eq)
    assert checkpdesol(eq, sol)[0]


def test_pde_1st_linear_constant_coeff():
    f, F = map(Function, ['f', 'F'])
    u = f(x,y)
    eq = -2*u.diff(x) + 4*u.diff(y) + 5*u - exp(x + 3*y)
    sol = pdsolve(eq)
    assert sol == Eq(f(x,y),
    (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]

    eq = (u.diff(x)/u) + (u.diff(y)/u) + 1 - (exp(x + y)/u)
    sol = pdsolve(eq)
    assert sol == Eq(f(x, y), F(x - y)*exp(-x/2 - y/2) + exp(x + y)/3)
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]

    eq = 2*u + -u.diff(x) + 3*u.diff(y) + sin(x)
    sol = pdsolve(eq)
    assert sol == Eq(f(x, y),
         F(3*x + y)*exp(x/5 - 3*y/5) - 2*sin(x)/5 - cos(x)/5)
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]

    eq = u + u.diff(x) + u.diff(y) + x*y
    sol = pdsolve(eq)
    assert sol.expand() == Eq(f(x, y),
        x + y + (x - y)**2/4 - (x + y)**2/4 + F(x - y)*exp(-x/2 - y/2) - 2).expand()
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]
    eq = u + u.diff(x) + u.diff(y) + log(x)
    assert classify_pde(eq) == ('1st_linear_constant_coeff',
    '1st_linear_constant_coeff_Integral')


def test_pdsolve_all():
    f, F = map(Function, ['f', 'F'])
    u = f(x,y)
    eq = u + u.diff(x) + u.diff(y) + x**2*y
    sol = pdsolve(eq, hint = 'all')
    keys = ['1st_linear_constant_coeff',
        '1st_linear_constant_coeff_Integral', 'default', 'order']
    assert sorted(sol.keys()) == keys
    assert sol['order'] == 1
    assert sol['default'] == '1st_linear_constant_coeff'
    assert sol['1st_linear_constant_coeff'].expand() == Eq(f(x, y),
        -x**2*y + x**2 + 2*x*y - 4*x - 2*y + F(x - y)*exp(-x/2 - y/2) + 6).expand()


def test_pdsolve_variable_coeff():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2
    sol = pdsolve(eq, hint="1st_linear_variable_coeff")
    assert sol == Eq(u, F(x*y)*exp(y**2/2) + 1)
    assert checkpdesol(eq, sol)[0]

    eq = x**2*u + x*u.diff(x) + x*y*u.diff(y)
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(y*exp(-x))*exp(-x**2/2))
    assert checkpdesol(eq, sol)[0]

    eq = y*x**2*u + y*u.diff(x) + u.diff(y)
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(-2*x + y**2)*exp(-x**3/3))
    assert checkpdesol(eq, sol)[0]

    eq = exp(x)**2*(u.diff(x)) + y
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, y*exp(-2*x)/2 + F(y))
    assert checkpdesol(eq, sol)[0]

    eq = exp(2*x)*(u.diff(y)) + y*u - u
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(x)*exp(-y*(y - 2)*exp(-2*x)/2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\indexing\test_get.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


def test_get():
    # GH 6383
    s = Series(
        np.array(
            [
                43,
                48,
                60,
                48,
                50,
                51,
                50,
                45,
                57,
                48,
                56,
                45,
                51,
                39,
                55,
                43,
                54,
                52,
                51,
                54,
            ]
        )
    )

    result = s.get(25, 0)
    expected = 0
    assert result == expected

    s = Series(
        np.array(
            [
                43,
                48,
                60,
                48,
                50,
                51,
                50,
                45,
                57,
                48,
                56,
                45,
                51,
                39,
                55,
                43,
                54,
                52,
                51,
                54,
            ]
        ),
        index=Index(
            [
                25.0,
                36.0,
                49.0,
                64.0,
                81.0,
                100.0,
                121.0,
                144.0,
                169.0,
                196.0,
                1225.0,
                1296.0,
                1369.0,
                1444.0,
                1521.0,
                1600.0,
                1681.0,
                1764.0,
                1849.0,
                1936.0,
            ],
            dtype=np.float64,
        ),
    )

    result = s.get(25, 0)
    expected = 43
    assert result == expected

    # GH 7407
    # with a boolean accessor
    df = pd.DataFrame({"i": [0] * 3, "b": [False] * 3})
    vc = df.i.value_counts()
    result = vc.get(99, default="Missing")
    assert result == "Missing"

    vc = df.b.value_counts()
    result = vc.get(False, default="Missing")
    assert result == 3

    result = vc.get(True, default="Missing")
    assert result == "Missing"


def test_get_nan(float_numpy_dtype):
    # GH 8569
    s = Index(range(10), dtype=float_numpy_dtype).to_series()
    assert s.get(np.nan) is None
    assert s.get(np.nan, default="Missing") == "Missing"


def test_get_nan_multiple(float_numpy_dtype):
    # GH 8569
    # ensure that fixing "test_get_nan" above hasn't broken get
    # with multiple elements
    s = Index(range(10), dtype=float_numpy_dtype).to_series()

    idx = [2, 30]
    assert s.get(idx) is None

    idx = [2, np.nan]
    assert s.get(idx) is None

    # GH 17295 - all missing keys
    idx = [20, 30]
    assert s.get(idx) is None

    idx = [np.nan, np.nan]
    assert s.get(idx) is None


def test_get_with_default():
    # GH#7725
    d0 = ["a", "b", "c", "d"]
    d1 = np.arange(4, dtype="int64")

    for data, index in ((d0, d1), (d1, d0)):
        s = Series(data, index=index)
        for i, d in zip(index, data):
            assert s.get(i) == d
            assert s.get(i, d) == d
            assert s.get(i, "z") == d

            assert s.get("e", "z") == "z"
            assert s.get("e", "e") == "e"

            msg = "Series.__getitem__ treating keys as positions is deprecated"
            warn = None
            if index is d0:
                warn = FutureWarning
            with tm.assert_produces_warning(warn, match=msg):
                assert s.get(10, "z") == "z"
                assert s.get(10, 10) == 10


@pytest.mark.parametrize(
    "arr",
    [
        np.random.default_rng(2).standard_normal(10),
        DatetimeIndex(date_range("2020-01-01", periods=10), name="a").tz_localize(
            tz="US/Eastern"
        ),
    ],
)
def test_get_with_ea(arr):
    # GH#21260
    ser = Series(arr, index=[2 * i for i in range(len(arr))])
    assert ser.get(4) == ser.iloc[2]

    result = ser.get([4, 6])
    expected = ser.iloc[[2, 3]]
    tm.assert_series_equal(result, expected)

    result = ser.get(slice(2))
    expected = ser.iloc[[0, 1]]
    tm.assert_series_equal(result, expected)

    assert ser.get(-1) is None
    assert ser.get(ser.index.max() + 1) is None

    ser = Series(arr[:6], index=list("abcdef"))
    assert ser.get("c") == ser.iloc[2]

    result = ser.get(slice("b", "d"))
    expected = ser.iloc[[1, 2, 3]]
    tm.assert_series_equal(result, expected)

    result = ser.get("Z")
    assert result is None

    msg = "Series.__getitem__ treating keys as positions is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ser.get(4) == ser.iloc[4]
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ser.get(-1) == ser.iloc[-1]
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ser.get(len(ser)) is None

    # GH#21257
    ser = Series(arr)
    ser2 = ser[::2]
    assert ser2.get(1) is None


def test_getitem_get(string_series, object_series):
    msg = "Series.__getitem__ treating keys as positions is deprecated"

    for obj in [string_series, object_series]:
        idx = obj.index[5]

        assert obj[idx] == obj.get(idx)
        assert obj[idx] == obj.iloc[5]

    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert string_series.get(-1) == string_series.get(string_series.index[-1])
    assert string_series.iloc[5] == string_series.get(string_series.index[5])


def test_get_none():
    # GH#5652
    s1 = Series(dtype=object)
    s2 = Series(dtype=object, index=list("abc"))
    for s in [s1, s2]:
        result = s.get(None)
        assert result is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_business_day.py ===
"""
Tests for offsets.BDay
"""
from __future__ import annotations

from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs.offsets import (
    ApplyTypeError,
    BDay,
    BMonthEnd,
)

from pandas import (
    DatetimeIndex,
    Timedelta,
    _testing as tm,
)
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)

from pandas.tseries import offsets


@pytest.fixture
def dt():
    return datetime(2008, 1, 1)


@pytest.fixture
def _offset():
    return BDay


@pytest.fixture
def offset(_offset):
    return _offset()


@pytest.fixture
def offset2(_offset):
    return _offset(2)


class TestBusinessDay:
    def test_different_normalize_equals(self, _offset, offset2):
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = _offset()
        offset2 = _offset(normalize=True)
        assert offset != offset2

    def test_repr(self, offset, offset2):
        assert repr(offset) == "<BusinessDay>"
        assert repr(offset2) == "<2 * BusinessDays>"

        expected = "<BusinessDay: offset=datetime.timedelta(days=1)>"
        assert repr(offset + timedelta(1)) == expected

    def test_with_offset(self, dt, offset):
        offset = offset + timedelta(hours=2)

        assert (dt + offset) == datetime(2008, 1, 2, 2)

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta(hours=2),
            Timedelta(hours=2).to_pytimedelta(),
            Timedelta(hours=2).to_timedelta64(),
        ],
        ids=lambda x: type(x),
    )
    def test_with_offset_index(self, td, dt, offset):
        dti = DatetimeIndex([dt])
        expected = DatetimeIndex([datetime(2008, 1, 2, 2)])

        result = dti + (td + offset)
        tm.assert_index_equal(result, expected)

        result = dti + (offset + td)
        tm.assert_index_equal(result, expected)

    def test_eq(self, offset2):
        assert offset2 == offset2

    def test_hash(self, offset2):
        assert hash(offset2) == hash(offset2)

    def test_add_datetime(self, dt, offset2):
        assert offset2 + dt == datetime(2008, 1, 3)
        assert offset2 + np.datetime64("2008-01-01 00:00:00") == datetime(2008, 1, 3)

    def testRollback1(self, dt, _offset):
        assert _offset(10).rollback(dt) == dt

    def testRollback2(self, _offset):
        assert _offset(10).rollback(datetime(2008, 1, 5)) == datetime(2008, 1, 4)

    def testRollforward1(self, dt, _offset):
        assert _offset(10).rollforward(dt) == dt

    def testRollforward2(self, _offset):
        assert _offset(10).rollforward(datetime(2008, 1, 5)) == datetime(2008, 1, 7)

    def test_roll_date_object(self, offset):
        dt = date(2012, 9, 15)

        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 14)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 17)

        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)

    @pytest.mark.parametrize(
        "dt, expected",
        [
            (datetime(2008, 1, 1), True),
            (datetime(2008, 1, 5), False),
        ],
    )
    def test_is_on_offset(self, offset, dt, expected):
        assert_is_on_offset(offset, dt, expected)

    apply_cases: list[tuple[int, dict[datetime, datetime]]] = [
        (
            1,
            {
                datetime(2008, 1, 1): datetime(2008, 1, 2),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 8),
            },
        ),
        (
            2,
            {
                datetime(2008, 1, 1): datetime(2008, 1, 3),
                datetime(2008, 1, 4): datetime(2008, 1, 8),
                datetime(2008, 1, 5): datetime(2008, 1, 8),
                datetime(2008, 1, 6): datetime(2008, 1, 8),
                datetime(2008, 1, 7): datetime(2008, 1, 9),
            },
        ),
        (
            -1,
            {
                datetime(2008, 1, 1): datetime(2007, 12, 31),
                datetime(2008, 1, 4): datetime(2008, 1, 3),
                datetime(2008, 1, 5): datetime(2008, 1, 4),
                datetime(2008, 1, 6): datetime(2008, 1, 4),
                datetime(2008, 1, 7): datetime(2008, 1, 4),
                datetime(2008, 1, 8): datetime(2008, 1, 7),
            },
        ),
        (
            -2,
            {
                datetime(2008, 1, 1): datetime(2007, 12, 28),
                datetime(2008, 1, 4): datetime(2008, 1, 2),
                datetime(2008, 1, 5): datetime(2008, 1, 3),
                datetime(2008, 1, 6): datetime(2008, 1, 3),
                datetime(2008, 1, 7): datetime(2008, 1, 3),
                datetime(2008, 1, 8): datetime(2008, 1, 4),
                datetime(2008, 1, 9): datetime(2008, 1, 7),
            },
        ),
        (
            0,
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 4): datetime(2008, 1, 4),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 7),
            },
        ),
    ]

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case, _offset):
        n, cases = case
        offset = _offset(n)
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    def test_apply_large_n(self, _offset):
        dt = datetime(2012, 10, 23)

        result = dt + _offset(10)
        assert result == datetime(2012, 11, 6)

        result = dt + _offset(100) - _offset(100)
        assert result == dt

        off = _offset() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 12, 23)
        assert rs == xp

        st = datetime(2011, 12, 18)
        rs = st + off
        xp = datetime(2011, 12, 26)
        assert rs == xp

        off = _offset() * 10
        rs = datetime(2014, 1, 5) + off  # see #5890
        xp = datetime(2014, 1, 17)
        assert rs == xp

    def test_apply_corner(self, _offset):
        if _offset is BDay:
            msg = "Only know how to combine business day with datetime or timedelta"
        else:
            msg = (
                "Only know how to combine trading day "
                "with datetime, datetime64 or timedelta"
            )
        with pytest.raises(ApplyTypeError, match=msg):
            _offset()._apply(BMonthEnd())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_constructor.py ===
"""Tests for tools for constructing domains for expressions. """

from sympy.testing.pytest import tooslow

from sympy.polys.constructor import construct_domain
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX
from sympy.polys.domains.realfield import RealField
from sympy.polys.domains.complexfield import ComplexField

from sympy.core import (Catalan, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy import rootof

from sympy.abc import x, y


def test_construct_domain():

    assert construct_domain([1, 2, 3]) == (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    assert construct_domain([1, 2, 3], field=True) == (QQ, [QQ(1), QQ(2), QQ(3)])

    assert construct_domain([S.One, S(2), S(3)]) == (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    assert construct_domain([S.One, S(2), S(3)], field=True) == (QQ, [QQ(1), QQ(2), QQ(3)])

    assert construct_domain([S.Half, S(2)]) == (QQ, [QQ(1, 2), QQ(2)])
    result = construct_domain([3.14, 1, S.Half])
    assert isinstance(result[0], RealField)
    assert result[1] == [RR(3.14), RR(1.0), RR(0.5)]

    result = construct_domain([3.14, I, S.Half])
    assert isinstance(result[0], ComplexField)
    assert result[1] == [CC(3.14), CC(1.0j), CC(0.5)]

    assert construct_domain([1.0+I]) == (CC, [CC(1.0, 1.0)])
    assert construct_domain([2.0+3.0*I]) == (CC, [CC(2.0, 3.0)])

    assert construct_domain([1, I]) == (ZZ_I, [ZZ_I(1, 0), ZZ_I(0, 1)])
    assert construct_domain([1, I/2]) == (QQ_I, [QQ_I(1, 0), QQ_I(0, S.Half)])

    assert construct_domain([3.14, sqrt(2)], extension=None) == (EX, [EX(3.14), EX(sqrt(2))])
    assert construct_domain([3.14, sqrt(2)], extension=True) == (EX, [EX(3.14), EX(sqrt(2))])

    assert construct_domain([1, sqrt(2)], extension=None) == (EX, [EX(1), EX(sqrt(2))])

    assert construct_domain([x, sqrt(x)]) == (EX, [EX(x), EX(sqrt(x))])
    assert construct_domain([x, sqrt(x), sqrt(y)]) == (EX, [EX(x), EX(sqrt(x)), EX(sqrt(y))])

    alg = QQ.algebraic_field(sqrt(2))

    assert construct_domain([7, S.Half, sqrt(2)], extension=True) == \
        (alg, [alg.convert(7), alg.convert(S.Half), alg.convert(sqrt(2))])

    alg = QQ.algebraic_field(sqrt(2) + sqrt(3))

    assert construct_domain([7, sqrt(2), sqrt(3)], extension=True) == \
        (alg, [alg.convert(7), alg.convert(sqrt(2)), alg.convert(sqrt(3))])

    dom = ZZ[x]

    assert construct_domain([2*x, 3]) == \
        (dom, [dom.convert(2*x), dom.convert(3)])

    dom = ZZ[x, y]

    assert construct_domain([2*x, 3*y]) == \
        (dom, [dom.convert(2*x), dom.convert(3*y)])

    dom = QQ[x]

    assert construct_domain([x/2, 3]) == \
        (dom, [dom.convert(x/2), dom.convert(3)])

    dom = QQ[x, y]

    assert construct_domain([x/2, 3*y]) == \
        (dom, [dom.convert(x/2), dom.convert(3*y)])

    dom = ZZ_I[x]

    assert construct_domain([2*x, I]) == \
        (dom, [dom.convert(2*x), dom.convert(I)])

    dom = ZZ_I[x, y]

    assert construct_domain([2*x, I*y]) == \
        (dom, [dom.convert(2*x), dom.convert(I*y)])

    dom = QQ_I[x]

    assert construct_domain([x/2, I]) == \
        (dom, [dom.convert(x/2), dom.convert(I)])

    dom = QQ_I[x, y]

    assert construct_domain([x/2, I*y]) == \
        (dom, [dom.convert(x/2), dom.convert(I*y)])

    dom = RR[x]

    assert construct_domain([x/2, 3.5]) == \
        (dom, [dom.convert(x/2), dom.convert(3.5)])

    dom = RR[x, y]

    assert construct_domain([x/2, 3.5*y]) == \
        (dom, [dom.convert(x/2), dom.convert(3.5*y)])

    dom = CC[x]

    assert construct_domain([I*x/2, 3.5]) == \
        (dom, [dom.convert(I*x/2), dom.convert(3.5)])

    dom = CC[x, y]

    assert construct_domain([I*x/2, 3.5*y]) == \
        (dom, [dom.convert(I*x/2), dom.convert(3.5*y)])

    dom = CC[x]

    assert construct_domain([x/2, I*3.5]) == \
        (dom, [dom.convert(x/2), dom.convert(I*3.5)])

    dom = CC[x, y]

    assert construct_domain([x/2, I*3.5*y]) == \
        (dom, [dom.convert(x/2), dom.convert(I*3.5*y)])

    dom = ZZ.frac_field(x)

    assert construct_domain([2/x, 3]) == \
        (dom, [dom.convert(2/x), dom.convert(3)])

    dom = ZZ.frac_field(x, y)

    assert construct_domain([2/x, 3*y]) == \
        (dom, [dom.convert(2/x), dom.convert(3*y)])

    dom = RR.frac_field(x)

    assert construct_domain([2/x, 3.5]) == \
        (dom, [dom.convert(2/x), dom.convert(3.5)])

    dom = RR.frac_field(x, y)

    assert construct_domain([2/x, 3.5*y]) == \
        (dom, [dom.convert(2/x), dom.convert(3.5*y)])

    dom = RealField(prec=336)[x]

    assert construct_domain([pi.evalf(100)*x]) == \
        (dom, [dom.convert(pi.evalf(100)*x)])

    assert construct_domain(2) == (ZZ, ZZ(2))
    assert construct_domain(S(2)/3) == (QQ, QQ(2, 3))
    assert construct_domain(Rational(2, 3)) == (QQ, QQ(2, 3))

    assert construct_domain({}) == (ZZ, {})


def test_complex_exponential():
    w = exp(-I*2*pi/3, evaluate=False)
    alg = QQ.algebraic_field(w)
    assert construct_domain([w**2, w, 1], extension=True) == (
        alg,
        [alg.convert(w**2),
         alg.convert(w),
         alg.convert(1)]
    )


def test_rootof():
    r1 = rootof(x**3 + x + 1, 0)
    r2 = rootof(x**3 + x + 1, 1)
    K1 = QQ.algebraic_field(r1)
    K2 = QQ.algebraic_field(r2)
    assert construct_domain([r1]) == (EX, [EX(r1)])
    assert construct_domain([r2]) == (EX, [EX(r2)])
    assert construct_domain([r1, r2]) == (EX, [EX(r1), EX(r2)])

    assert construct_domain([r1], extension=True) == (
            K1, [K1.from_sympy(r1)])
    assert construct_domain([r2], extension=True) == (
            K2, [K2.from_sympy(r2)])


@tooslow
def test_rootof_primitive_element():
    r1 = rootof(x**3 + x + 1, 0)
    r2 = rootof(x**3 + x + 1, 1)
    K12 = QQ.algebraic_field(r1 + r2)
    assert construct_domain([r1, r2], extension=True) == (
            K12, [K12.from_sympy(r1), K12.from_sympy(r2)])


def test_composite_option():
    assert construct_domain({(1,): sin(y)}, composite=False) == \
        (EX, {(1,): EX(sin(y))})

    assert construct_domain({(1,): y}, composite=False) == \
        (EX, {(1,): EX(y)})

    assert construct_domain({(1, 1): 1}, composite=False) == \
        (ZZ, {(1, 1): 1})

    assert construct_domain({(1, 0): y}, composite=False) == \
        (EX, {(1, 0): EX(y)})


def test_precision():
    f1 = Float("1.01")
    f2 = Float("1.0000000000000000000001")
    for u in [1, 1e-2, 1e-6, 1e-13, 1e-14, 1e-16, 1e-20, 1e-100, 1e-300,
            f1, f2]:
        result = construct_domain([u])
        v = float(result[1][0])
        assert abs(u - v) / u < 1e-14  # Test relative accuracy

    result = construct_domain([f1])
    y = result[1][0]
    assert y-1 > 1e-50

    result = construct_domain([f2])
    y = result[1][0]
    assert y-1 > 1e-50


def test_issue_11538():
    for n in [E, pi, Catalan]:
        assert construct_domain(n)[0] == ZZ[n]
        assert construct_domain(x + n)[0] == ZZ[x, n]
    assert construct_domain(GoldenRatio)[0] == EX
    assert construct_domain(x + GoldenRatio)[0] == EX

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_multiindex.py ===
import numpy as np
import pytest

import pandas._libs.index as libindex
from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import (
    CategoricalDtype,
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype


class TestMultiIndexBasic:
    def test_multiindex_perf_warn(self):
        df = DataFrame(
            {
                "jim": [0, 0, 1, 1],
                "joe": ["x", "x", "z", "y"],
                "jolie": np.random.default_rng(2).random(4),
            }
        ).set_index(["jim", "joe"])

        with tm.assert_produces_warning(PerformanceWarning):
            df.loc[(1, "z")]

        df = df.iloc[[2, 1, 3, 0]]
        with tm.assert_produces_warning(PerformanceWarning):
            df.loc[(0,)]

    @pytest.mark.parametrize("offset", [-5, 5])
    def test_indexing_over_hashtable_size_cutoff(self, monkeypatch, offset):
        size_cutoff = 20
        n = size_cutoff + offset

        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            s = Series(np.arange(n), MultiIndex.from_arrays((["a"] * n, np.arange(n))))

            # hai it works!
            assert s[("a", 5)] == 5
            assert s[("a", 6)] == 6
            assert s[("a", 7)] == 7

    def test_multi_nan_indexing(self):
        # GH 3588
        df = DataFrame(
            {
                "a": ["R1", "R2", np.nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            }
        )
        result = df.set_index(["a", "b"], drop=False)
        expected = DataFrame(
            {
                "a": ["R1", "R2", np.nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            },
            index=[
                Index(["R1", "R2", np.nan, "R4"], name="a"),
                Index(["C1", "C2", "C3", "C4"], name="b"),
            ],
        )
        tm.assert_frame_equal(result, expected)

    def test_exclusive_nat_column_indexing(self):
        # GH 38025
        # test multi indexing when one column exclusively contains NaT values
        df = DataFrame(
            {
                "a": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            }
        )
        df = df.set_index(["a", "b"])
        expected = DataFrame(
            {
                "c": [10, 15, np.nan, 20],
            },
            index=[
                Index([pd.NaT, pd.NaT, pd.NaT, pd.NaT], name="a"),
                Index(["C1", "C2", "C3", "C4"], name="b"),
            ],
        )
        tm.assert_frame_equal(df, expected)

    def test_nested_tuples_duplicates(self):
        # GH#30892

        dti = pd.to_datetime(["20190101", "20190101", "20190102"])
        idx = Index(["a", "a", "c"])
        mi = MultiIndex.from_arrays([dti, idx], names=["index1", "index2"])

        df = DataFrame({"c1": [1, 2, 3], "c2": [np.nan, np.nan, np.nan]}, index=mi)

        expected = DataFrame({"c1": df["c1"], "c2": [1.0, 1.0, np.nan]}, index=mi)

        df2 = df.copy(deep=True)
        df2.loc[(dti[0], "a"), "c2"] = 1.0
        tm.assert_frame_equal(df2, expected)

        df3 = df.copy(deep=True)
        df3.loc[[(dti[0], "a")], "c2"] = 1.0
        tm.assert_frame_equal(df3, expected)

    def test_multiindex_with_datatime_level_preserves_freq(self):
        # https://github.com/pandas-dev/pandas/issues/35563
        idx = Index(range(2), name="A")
        dti = pd.date_range("2020-01-01", periods=7, freq="D", name="B")
        mi = MultiIndex.from_product([idx, dti])
        df = DataFrame(np.random.default_rng(2).standard_normal((14, 2)), index=mi)
        result = df.loc[0].index
        tm.assert_index_equal(result, dti)
        assert result.freq == dti.freq

    def test_multiindex_complex(self):
        # GH#42145
        complex_data = [1 + 2j, 4 - 3j, 10 - 1j]
        non_complex_data = [3, 4, 5]
        result = DataFrame(
            {
                "x": complex_data,
                "y": non_complex_data,
                "z": non_complex_data,
            }
        )
        result.set_index(["x", "y"], inplace=True)
        expected = DataFrame(
            {"z": non_complex_data},
            index=MultiIndex.from_arrays(
                [complex_data, non_complex_data],
                names=("x", "y"),
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_rename_multiindex_with_duplicates(self):
        # GH 38015
        mi = MultiIndex.from_tuples([("A", "cat"), ("B", "cat"), ("B", "cat")])
        df = DataFrame(index=mi)
        df = df.rename(index={"A": "Apple"}, level=0)

        mi2 = MultiIndex.from_tuples([("Apple", "cat"), ("B", "cat"), ("B", "cat")])
        expected = DataFrame(index=mi2)
        tm.assert_frame_equal(df, expected)

    def test_series_align_multiindex_with_nan_overlap_only(self):
        # GH 38439
        mi1 = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        mi2 = MultiIndex.from_arrays([[np.nan, 82.0], [np.nan, np.nan]])
        ser1 = Series([1, 2], index=mi1)
        ser2 = Series([1, 2], index=mi2)
        result1, result2 = ser1.align(ser2)

        mi = MultiIndex.from_arrays([[81.0, 82.0, np.nan], [np.nan, np.nan, np.nan]])
        expected1 = Series([1.0, np.nan, 2.0], index=mi)
        expected2 = Series([np.nan, 2.0, 1.0], index=mi)

        tm.assert_series_equal(result1, expected1)
        tm.assert_series_equal(result2, expected2)

    def test_series_align_multiindex_with_nan(self):
        # GH 38439
        mi1 = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        mi2 = MultiIndex.from_arrays([[np.nan, 81.0], [np.nan, np.nan]])
        ser1 = Series([1, 2], index=mi1)
        ser2 = Series([1, 2], index=mi2)
        result1, result2 = ser1.align(ser2)

        mi = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        expected1 = Series([1, 2], index=mi)
        expected2 = Series([2, 1], index=mi)

        tm.assert_series_equal(result1, expected1)
        tm.assert_series_equal(result2, expected2)

    def test_nunique_smoke(self):
        # GH 34019
        n = DataFrame([[1, 2], [1, 2]]).set_index([0, 1]).index.nunique()
        assert n == 1

    def test_multiindex_repeated_keys(self):
        # GH19414
        tm.assert_series_equal(
            Series([1, 2], MultiIndex.from_arrays([["a", "b"]])).loc[
                ["a", "a", "b", "b"]
            ],
            Series([1, 1, 2, 2], MultiIndex.from_arrays([["a", "a", "b", "b"]])),
        )

    def test_multiindex_with_na_missing_key(self):
        # GH46173
        df = DataFrame.from_dict(
            {
                ("foo",): [1, 2, 3],
                ("bar",): [5, 6, 7],
                (None,): [8, 9, 0],
            }
        )
        with pytest.raises(KeyError, match="missing_key"):
            df[[("missing_key",)]]

    def test_multiindex_dtype_preservation(self):
        # GH51261
        columns = MultiIndex.from_tuples([("A", "B")], names=["lvl1", "lvl2"])
        df = DataFrame(["value"], columns=columns).astype("category")
        df_no_multiindex = df["A"]
        assert isinstance(df_no_multiindex["B"].dtype, CategoricalDtype)

        # geopandas 1763 analogue
        df = DataFrame(
            [[1, 0], [0, 1]],
            columns=[
                ["foo", "foo"],
                ["location", "location"],
                ["x", "y"],
            ],
        ).assign(bools=Series([True, False], dtype="boolean"))
        assert isinstance(df["bools"].dtype, BooleanDtype)

    def test_multiindex_from_tuples_with_nan(self):
        # GH#23578
        result = MultiIndex.from_tuples([("a", "b", "c"), np.nan, ("d", "", "")])
        expected = MultiIndex.from_tuples(
            [("a", "b", "c"), (np.nan, np.nan, np.nan), ("d", "", "")]
        )
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_primetest.py ===
from math import gcd

from sympy.ntheory.generate import Sieve, sieve
from sympy.ntheory.primetest import (mr, _lucas_extrastrong_params, is_lucas_prp, is_square,
                                     is_strong_lucas_prp, is_extra_strong_lucas_prp,
                                     proth_test, isprime, is_euler_pseudoprime,
                                     is_gaussian_prime, is_fermat_pseudoprime, is_euler_jacobi_pseudoprime,
                                     MERSENNE_PRIME_EXPONENTS, _lucas_lehmer_primality_test,
                                     is_mersenne_prime)

from sympy.testing.pytest import slow, raises
from sympy.core.numbers import I, Float


def test_is_fermat_pseudoprime():
    assert is_fermat_pseudoprime(5, 1)
    assert is_fermat_pseudoprime(9, 1)


def test_euler_pseudoprimes():
    assert is_euler_pseudoprime(13, 1)
    assert is_euler_pseudoprime(15, 1)
    assert is_euler_pseudoprime(17, 6)
    assert is_euler_pseudoprime(101, 7)
    assert is_euler_pseudoprime(1009, 10)
    assert is_euler_pseudoprime(11287, 41)

    raises(ValueError, lambda: is_euler_pseudoprime(0, 4))
    raises(ValueError, lambda: is_euler_pseudoprime(3, 0))
    raises(ValueError, lambda: is_euler_pseudoprime(15, 6))

    # A006970
    euler_prp = [341, 561, 1105, 1729, 1905, 2047, 2465, 3277,
                 4033, 4681, 5461, 6601, 8321, 8481, 10261, 10585]
    for p in euler_prp:
        assert is_euler_pseudoprime(p, 2)

    # A048950
    euler_prp = [121, 703, 1729, 1891, 2821, 3281, 7381, 8401, 8911, 10585,
                 12403, 15457, 15841, 16531, 18721, 19345, 23521, 24661, 28009]
    for p in euler_prp:
        assert is_euler_pseudoprime(p, 3)

    # A033181
    absolute_euler_prp = [1729, 2465, 15841, 41041, 46657, 75361,
                          162401, 172081, 399001, 449065, 488881]
    for p in absolute_euler_prp:
        for a in range(2, p):
            if gcd(a, p) != 1:
                continue
            assert is_euler_pseudoprime(p, a)


def test_is_euler_jacobi_pseudoprime():
    assert is_euler_jacobi_pseudoprime(11, 1)
    assert is_euler_jacobi_pseudoprime(15, 1)


def test_lucas_extrastrong_params():
    assert _lucas_extrastrong_params(3) == (5, 3, 1)
    assert _lucas_extrastrong_params(5) == (12, 4, 1)
    assert _lucas_extrastrong_params(7) == (5, 3, 1)
    assert _lucas_extrastrong_params(9) == (0, 0, 0)
    assert _lucas_extrastrong_params(11) == (21, 5, 1)
    assert _lucas_extrastrong_params(59) == (32, 6, 1)
    assert _lucas_extrastrong_params(479) == (117, 11, 1)


def test_is_extra_strong_lucas_prp():
    assert is_extra_strong_lucas_prp(4) == False
    assert is_extra_strong_lucas_prp(989) == True
    assert is_extra_strong_lucas_prp(10877) == True
    assert is_extra_strong_lucas_prp(9) == False
    assert is_extra_strong_lucas_prp(16) == False
    assert is_extra_strong_lucas_prp(169) == False

@slow
def test_prps():
    oddcomposites = [n for n in range(1, 10**5) if
        n % 2 and not isprime(n)]
    # A checksum would be better.
    assert sum(oddcomposites) == 2045603465
    assert [n for n in oddcomposites if mr(n, [2])] == [
        2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141,
        52633, 65281, 74665, 80581, 85489, 88357, 90751]
    assert [n for n in oddcomposites if mr(n, [3])] == [
        121, 703, 1891, 3281, 8401, 8911, 10585, 12403, 16531,
        18721, 19345, 23521, 31621, 44287, 47197, 55969, 63139,
        74593, 79003, 82513, 87913, 88573, 97567]
    assert [n for n in oddcomposites if mr(n, [325])] == [
        9, 25, 27, 49, 65, 81, 325, 341, 343, 697, 1141, 2059,
        2149, 3097, 3537, 4033, 4681, 4941, 5833, 6517, 7987, 8911,
        12403, 12913, 15043, 16021, 20017, 22261, 23221, 24649,
        24929, 31841, 35371, 38503, 43213, 44173, 47197, 50041,
        55909, 56033, 58969, 59089, 61337, 65441, 68823, 72641,
        76793, 78409, 85879]
    assert not any(mr(n, [9345883071009581737]) for n in oddcomposites)
    assert [n for n in oddcomposites if is_lucas_prp(n)] == [
        323, 377, 1159, 1829, 3827, 5459, 5777, 9071, 9179, 10877,
        11419, 11663, 13919, 14839, 16109, 16211, 18407, 18971,
        19043, 22499, 23407, 24569, 25199, 25877, 26069, 27323,
        32759, 34943, 35207, 39059, 39203, 39689, 40309, 44099,
        46979, 47879, 50183, 51983, 53663, 56279, 58519, 60377,
        63881, 69509, 72389, 73919, 75077, 77219, 79547, 79799,
        82983, 84419, 86063, 90287, 94667, 97019, 97439]
    assert [n for n in oddcomposites if is_strong_lucas_prp(n)] == [
        5459, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309,
        58519, 75077, 97439]
    assert [n for n in oddcomposites if is_extra_strong_lucas_prp(n)
            ] == [
        989, 3239, 5777, 10877, 27971, 29681, 30739, 31631, 39059,
        72389, 73919, 75077]


def test_proth_test():
    # Proth number
    A080075 = [3, 5, 9, 13, 17, 25, 33, 41, 49, 57, 65,
               81, 97, 113, 129, 145, 161, 177, 193]
    # Proth prime
    A080076 = [3, 5, 13, 17, 41, 97, 113, 193]

    for n in range(200):
        if n in A080075:
            assert proth_test(n) == (n in A080076)
        else:
            raises(ValueError, lambda: proth_test(n))


def test_lucas_lehmer_primality_test():
    for p in sieve.primerange(3, 100):
        assert _lucas_lehmer_primality_test(p) == (p in MERSENNE_PRIME_EXPONENTS)


def test_is_mersenne_prime():
    assert is_mersenne_prime(-3) is False
    assert is_mersenne_prime(3) is True
    assert is_mersenne_prime(10) is False
    assert is_mersenne_prime(127) is True
    assert is_mersenne_prime(511) is False
    assert is_mersenne_prime(131071) is True
    assert is_mersenne_prime(2147483647) is True


def test_isprime():
    s = Sieve()
    s.extend(100000)
    ps = set(s.primerange(2, 100001))
    for n in range(100001):
        # if (n in ps) != isprime(n): print n
        assert (n in ps) == isprime(n)
    assert isprime(179424673)
    assert isprime(20678048681)
    assert isprime(1968188556461)
    assert isprime(2614941710599)
    assert isprime(65635624165761929287)
    assert isprime(1162566711635022452267983)
    assert isprime(77123077103005189615466924501)
    assert isprime(3991617775553178702574451996736229)
    assert isprime(273952953553395851092382714516720001799)
    assert isprime(int('''
531137992816767098689588206552468627329593117727031923199444138200403\
559860852242739162502265229285668889329486246501015346579337652707239\
409519978766587351943831270835393219031728127'''))

    # Some Mersenne primes
    assert isprime(2**61 - 1)
    assert isprime(2**89 - 1)
    assert isprime(2**607 - 1)
    # (but not all Mersenne's are primes
    assert not isprime(2**601 - 1)

    # pseudoprimes
    #-------------
    # to some small bases
    assert not isprime(2152302898747)
    assert not isprime(3474749660383)
    assert not isprime(341550071728321)
    assert not isprime(3825123056546413051)
    # passes the base set [2, 3, 7, 61, 24251]
    assert not isprime(9188353522314541)
    # large examples
    assert not isprime(877777777777777777777777)
    # conjectured psi_12 given at http://mathworld.wolfram.com/StrongPseudoprime.html
    assert not isprime(318665857834031151167461)
    # conjectured psi_17 given at http://mathworld.wolfram.com/StrongPseudoprime.html
    assert not isprime(564132928021909221014087501701)
    # Arnault's 1993 number; a factor of it is
    # 400958216639499605418306452084546853005188166041132508774506\
    # 204738003217070119624271622319159721973358216316508535816696\
    # 9145233813917169287527980445796800452592031836601
    assert not isprime(int('''
803837457453639491257079614341942108138837688287558145837488917522297\
427376533365218650233616396004545791504202360320876656996676098728404\
396540823292873879185086916685732826776177102938969773947016708230428\
687109997439976544144845341155872450633409279022275296229414984230688\
1685404326457534018329786111298960644845216191652872597534901'''))
    # Arnault's 1995 number; can be factored as
    # p1*(313*(p1 - 1) + 1)*(353*(p1 - 1) + 1) where p1 is
    # 296744956686855105501541746429053327307719917998530433509950\
    # 755312768387531717701995942385964281211880336647542183455624\
    # 93168782883
    assert not isprime(int('''
288714823805077121267142959713039399197760945927972270092651602419743\
230379915273311632898314463922594197780311092934965557841894944174093\
380561511397999942154241693397290542371100275104208013496673175515285\
922696291677532547504444585610194940420003990443211677661994962953925\
045269871932907037356403227370127845389912612030924484149472897688540\
6024976768122077071687938121709811322297802059565867'''))
    sieve.extend(3000)
    assert isprime(2819)
    assert not isprime(2931)
    raises(ValueError, lambda: isprime(2.0))
    raises(ValueError, lambda: isprime(Float(2)))


def test_is_square():
    assert [i for i in range(25) if is_square(i)] == [0, 1, 4, 9, 16]

    # issue #17044
    assert not is_square(60 ** 3)
    assert not is_square(60 ** 5)
    assert not is_square(84 ** 7)
    assert not is_square(105 ** 9)
    assert not is_square(120 ** 3)

def test_is_gaussianprime():
    assert is_gaussian_prime(7*I)
    assert is_gaussian_prime(7)
    assert is_gaussian_prime(2 + 3*I)
    assert not is_gaussian_prime(2 + 2*I)


def test_issue_27145():
    #https://github.com/sympy/sympy/issues/27145
    assert [mr(i,[2,3,5,7]) for i in (1, 2, 6)] == [False, True, False]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\tests\test_dimacs.py ===
"""Various tests on satisfiability using dimacs cnf file syntax
You can find lots of cnf files in
ftp://dimacs.rutgers.edu/pub/challenge/satisfiability/benchmarks/cnf/
"""

from sympy.logic.utilities.dimacs import load
from sympy.logic.algorithms.dpll import dpll_satisfiable


def test_f1():
    assert bool(dpll_satisfiable(load(f1)))


def test_f2():
    assert bool(dpll_satisfiable(load(f2)))


def test_f3():
    assert bool(dpll_satisfiable(load(f3)))


def test_f4():
    assert not bool(dpll_satisfiable(load(f4)))


def test_f5():
    assert bool(dpll_satisfiable(load(f5)))

f1 = """c  simple example
c Resolution: SATISFIABLE
c
p cnf 3 2
1 -3 0
2 3 -1 0
"""


f2 = """c  an example from Quinn's text, 16 variables and 18 clauses.
c Resolution: SATISFIABLE
c
p cnf 16 18
  1    2  0
 -2   -4  0
  3    4  0
 -4   -5  0
  5   -6  0
  6   -7  0
  6    7  0
  7  -16  0
  8   -9  0
 -8  -14  0
  9   10  0
  9  -10  0
-10  -11  0
 10   12  0
 11   12  0
 13   14  0
 14  -15  0
 15   16  0
"""

f3 = """c
p cnf 6 9
-1 0
-3 0
2 -1 0
2 -4 0
5 -4 0
-1 -3 0
-4 -6 0
1 3 -2 0
4 6 -2 -5 0
"""

f4 = """c
c file:   hole6.cnf [http://people.sc.fsu.edu/~jburkardt/data/cnf/hole6.cnf]
c
c SOURCE: John Hooker (jh38+@andrew.cmu.edu)
c
c DESCRIPTION: Pigeon hole problem of placing n (for file 'holen.cnf') pigeons
c              in n+1 holes without placing 2 pigeons in the same hole
c
c NOTE: Part of the collection at the Forschungsinstitut fuer
c       anwendungsorientierte Wissensverarbeitung in Ulm Germany.
c
c NOTE: Not satisfiable
c
p cnf 42 133
-1     -7    0
-1     -13   0
-1     -19   0
-1     -25   0
-1     -31   0
-1     -37   0
-7     -13   0
-7     -19   0
-7     -25   0
-7     -31   0
-7     -37   0
-13    -19   0
-13    -25   0
-13    -31   0
-13    -37   0
-19    -25   0
-19    -31   0
-19    -37   0
-25    -31   0
-25    -37   0
-31    -37   0
-2     -8    0
-2     -14   0
-2     -20   0
-2     -26   0
-2     -32   0
-2     -38   0
-8     -14   0
-8     -20   0
-8     -26   0
-8     -32   0
-8     -38   0
-14    -20   0
-14    -26   0
-14    -32   0
-14    -38   0
-20    -26   0
-20    -32   0
-20    -38   0
-26    -32   0
-26    -38   0
-32    -38   0
-3     -9    0
-3     -15   0
-3     -21   0
-3     -27   0
-3     -33   0
-3     -39   0
-9     -15   0
-9     -21   0
-9     -27   0
-9     -33   0
-9     -39   0
-15    -21   0
-15    -27   0
-15    -33   0
-15    -39   0
-21    -27   0
-21    -33   0
-21    -39   0
-27    -33   0
-27    -39   0
-33    -39   0
-4     -10   0
-4     -16   0
-4     -22   0
-4     -28   0
-4     -34   0
-4     -40   0
-10    -16   0
-10    -22   0
-10    -28   0
-10    -34   0
-10    -40   0
-16    -22   0
-16    -28   0
-16    -34   0
-16    -40   0
-22    -28   0
-22    -34   0
-22    -40   0
-28    -34   0
-28    -40   0
-34    -40   0
-5     -11   0
-5     -17   0
-5     -23   0
-5     -29   0
-5     -35   0
-5     -41   0
-11    -17   0
-11    -23   0
-11    -29   0
-11    -35   0
-11    -41   0
-17    -23   0
-17    -29   0
-17    -35   0
-17    -41   0
-23    -29   0
-23    -35   0
-23    -41   0
-29    -35   0
-29    -41   0
-35    -41   0
-6     -12   0
-6     -18   0
-6     -24   0
-6     -30   0
-6     -36   0
-6     -42   0
-12    -18   0
-12    -24   0
-12    -30   0
-12    -36   0
-12    -42   0
-18    -24   0
-18    -30   0
-18    -36   0
-18    -42   0
-24    -30   0
-24    -36   0
-24    -42   0
-30    -36   0
-30    -42   0
-36    -42   0
 6      5      4      3      2      1    0
 12     11     10     9      8      7    0
 18     17     16     15     14     13   0
 24     23     22     21     20     19   0
 30     29     28     27     26     25   0
 36     35     34     33     32     31   0
 42     41     40     39     38     37   0
"""

f5 = """c  simple example requiring variable selection
c
c NOTE: Satisfiable
c
p cnf 5 5
1 2 3 0
1 -2 3 0
4 5 -3 0
1 -4 -3 0
-1 -5 0
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_convert.py ===
import random
from mpmath import *
from mpmath.libmp import *


def test_basic_string():
    """
    Test basic string conversion
    """
    mp.dps = 15
    assert mpf('3') == mpf('3.0') == mpf('0003.') == mpf('0.03e2') == mpf(3.0)
    assert mpf('30') == mpf('30.0') == mpf('00030.') == mpf(30.0)
    for i in range(10):
        for j in range(10):
            assert mpf('%ie%i' % (i,j)) == i * 10**j
    assert str(mpf('25000.0')) == '25000.0'
    assert str(mpf('2500.0')) == '2500.0'
    assert str(mpf('250.0')) == '250.0'
    assert str(mpf('25.0')) == '25.0'
    assert str(mpf('2.5')) == '2.5'
    assert str(mpf('0.25')) == '0.25'
    assert str(mpf('0.025')) == '0.025'
    assert str(mpf('0.0025')) == '0.0025'
    assert str(mpf('0.00025')) == '0.00025'
    assert str(mpf('0.000025')) == '2.5e-5'
    assert str(mpf(0)) == '0.0'
    assert str(mpf('2.5e1000000000000000000000')) == '2.5e+1000000000000000000000'
    assert str(mpf('2.6e-1000000000000000000000')) == '2.6e-1000000000000000000000'
    assert str(mpf(1.23402834e-15)) == '1.23402834e-15'
    assert str(mpf(-1.23402834e-15)) == '-1.23402834e-15'
    assert str(mpf(-1.2344e-15)) == '-1.2344e-15'
    assert repr(mpf(-1.2344e-15)) == "mpf('-1.2343999999999999e-15')"
    assert str(mpf("2163048125L")) == '2163048125.0'
    assert str(mpf("-2163048125l")) == '-2163048125.0'
    assert str(mpf("-2163048125L/1088391168")) == '-1.98738118113799'
    assert str(mpf("2163048125/1088391168l")) == '1.98738118113799'

def test_pretty():
    mp.pretty = True
    assert repr(mpf(2.5)) == '2.5'
    assert repr(mpc(2.5,3.5)) == '(2.5 + 3.5j)'
    mp.pretty = False
    iv.pretty = True
    assert repr(mpi(2.5,3.5)) == '[2.5, 3.5]'
    iv.pretty = False

def test_str_whitespace():
    assert mpf('1.26 ') == 1.26

def test_unicode():
    mp.dps = 15
    try:
        unicode = unicode
    except NameError:
        unicode = str
    assert mpf(unicode('2.76')) == 2.76
    assert mpf(unicode('inf')) == inf

def test_str_format():
    assert to_str(from_float(0.1),15,strip_zeros=False) == '0.100000000000000'
    assert to_str(from_float(0.0),15,show_zero_exponent=True) == '0.0e+0'
    assert to_str(from_float(0.0),0,show_zero_exponent=True) == '.0e+0'
    assert to_str(from_float(0.0),0,show_zero_exponent=False) == '.0'
    assert to_str(from_float(0.0),1,show_zero_exponent=True) == '0.0e+0'
    assert to_str(from_float(0.0),1,show_zero_exponent=False) == '0.0'
    assert to_str(from_float(1.23),3,show_zero_exponent=True) == '1.23e+0'
    assert to_str(from_float(1.23456789000000e-2),15,strip_zeros=False,min_fixed=0,max_fixed=0) == '1.23456789000000e-2'
    assert to_str(from_float(1.23456789000000e+2),15,strip_zeros=False,min_fixed=0,max_fixed=0) == '1.23456789000000e+2'
    assert to_str(from_float(2.1287e14), 15, max_fixed=1000) == '212870000000000.0'
    assert to_str(from_float(2.1287e15), 15, max_fixed=1000) == '2128700000000000.0'
    assert to_str(from_float(2.1287e16), 15, max_fixed=1000) == '21287000000000000.0'
    assert to_str(from_float(2.1287e30), 15, max_fixed=1000) == '2128700000000000000000000000000.0'

def test_tight_string_conversion():
    mp.dps = 15
    # In an old version, '0.5' wasn't recognized as representing
    # an exact binary number and was erroneously rounded up or down
    assert from_str('0.5', 10, round_floor) == fhalf
    assert from_str('0.5', 10, round_ceiling) == fhalf

def test_eval_repr_invariant():
    """Test that eval(repr(x)) == x"""
    random.seed(123)
    for dps in [10, 15, 20, 50, 100]:
        mp.dps = dps
        for i in range(1000):
            a = mpf(random.random())**0.5 * 10**random.randint(-100, 100)
            assert eval(repr(a)) == a
    mp.dps = 15

def test_str_bugs():
    mp.dps = 15
    # Decimal rounding used to give the wrong exponent in some cases
    assert str(mpf('1e600')) == '1.0e+600'
    assert str(mpf('1e10000')) == '1.0e+10000'

def test_str_prec0():
    assert to_str(from_float(1.234), 0) == '.0e+0'
    assert to_str(from_float(1e-15), 0) == '.0e-15'
    assert to_str(from_float(1e+15), 0) == '.0e+15'
    assert to_str(from_float(-1e-15), 0) == '-.0e-15'
    assert to_str(from_float(-1e+15), 0) == '-.0e+15'

def test_convert_rational():
    mp.dps = 15
    assert from_rational(30, 5, 53, round_nearest) == (0, 3, 1, 2)
    assert from_rational(-7, 4, 53, round_nearest) == (1, 7, -2, 3)
    assert to_rational((0, 1, -1, 1)) == (1, 2)

def test_custom_class():
    class mympf:
        @property
        def _mpf_(self):
            return mpf(3.5)._mpf_
    class mympc:
        @property
        def _mpc_(self):
            return mpf(3.5)._mpf_, mpf(2.5)._mpf_
    assert mpf(2) + mympf() == 5.5
    assert mympf() + mpf(2) == 5.5
    assert mpf(mympf()) == 3.5
    assert mympc() + mpc(2) == mpc(5.5, 2.5)
    assert mpc(2) + mympc() == mpc(5.5, 2.5)
    assert mpc(mympc()) == (3.5+2.5j)

def test_conversion_methods():
    class SomethingRandom:
        pass
    class SomethingReal:
        def _mpmath_(self, prec, rounding):
            return mp.make_mpf(from_str('1.3', prec, rounding))
    class SomethingComplex:
        def _mpmath_(self, prec, rounding):
            return mp.make_mpc((from_str('1.3', prec, rounding), \
                from_str('1.7', prec, rounding)))
    x = mpf(3)
    z = mpc(3)
    a = SomethingRandom()
    y = SomethingReal()
    w = SomethingComplex()
    for d in [15, 45]:
        mp.dps = d
        assert (x+y).ae(mpf('4.3'))
        assert (y+x).ae(mpf('4.3'))
        assert (x+w).ae(mpc('4.3', '1.7'))
        assert (w+x).ae(mpc('4.3', '1.7'))
        assert (z+y).ae(mpc('4.3'))
        assert (y+z).ae(mpc('4.3'))
        assert (z+w).ae(mpc('4.3', '1.7'))
        assert (w+z).ae(mpc('4.3', '1.7'))
        x-y; y-x; x-w; w-x; z-y; y-z; z-w; w-z
        x*y; y*x; x*w; w*x; z*y; y*z; z*w; w*z
        x/y; y/x; x/w; w/x; z/y; y/z; z/w; w/z
        x**y; y**x; x**w; w**x; z**y; y**z; z**w; w**z
        x==y; y==x; x==w; w==x; z==y; y==z; z==w; w==z
    mp.dps = 15
    assert x.__add__(a) is NotImplemented
    assert x.__radd__(a) is NotImplemented
    assert x.__lt__(a) is NotImplemented
    assert x.__gt__(a) is NotImplemented
    assert x.__le__(a) is NotImplemented
    assert x.__ge__(a) is NotImplemented
    assert x.__eq__(a) is NotImplemented
    assert x.__ne__(a) is NotImplemented
    # implementation detail
    if hasattr(x, "__cmp__"):
        assert x.__cmp__(a) is NotImplemented
    assert x.__sub__(a) is NotImplemented
    assert x.__rsub__(a) is NotImplemented
    assert x.__mul__(a) is NotImplemented
    assert x.__rmul__(a) is NotImplemented
    assert x.__div__(a) is NotImplemented
    assert x.__rdiv__(a) is NotImplemented
    assert x.__mod__(a) is NotImplemented
    assert x.__rmod__(a) is NotImplemented
    assert x.__pow__(a) is NotImplemented
    assert x.__rpow__(a) is NotImplemented
    assert z.__add__(a) is NotImplemented
    assert z.__radd__(a) is NotImplemented
    assert z.__eq__(a) is NotImplemented
    assert z.__ne__(a) is NotImplemented
    assert z.__sub__(a) is NotImplemented
    assert z.__rsub__(a) is NotImplemented
    assert z.__mul__(a) is NotImplemented
    assert z.__rmul__(a) is NotImplemented
    assert z.__div__(a) is NotImplemented
    assert z.__rdiv__(a) is NotImplemented
    assert z.__pow__(a) is NotImplemented
    assert z.__rpow__(a) is NotImplemented

def test_mpmathify():
    assert mpmathify('1/2') == 0.5
    assert mpmathify('(1.0+1.0j)') == mpc(1, 1)
    assert mpmathify('(1.2e-10 - 3.4e5j)') == mpc('1.2e-10', '-3.4e5')
    assert mpmathify('1j') == mpc(1j)

def test_issue548():
    try:
        # This expression is invalid, but may trigger the ReDOS vulnerability
        # in the regular expression for parsing complex numbers.
        mpmathify('(' + '1' * 5000 + '!j')
    except:
        return
    # The expression is invalid and should raise an exception.
    assert False

def test_compatibility():
    try:
        import numpy as np
        from fractions import Fraction
        from decimal import Decimal
        import decimal
    except ImportError:
        return
    # numpy types
    for nptype in np.core.numerictypes.typeDict.values():
        if issubclass(nptype, np.complexfloating):
            x = nptype(complex(0.5, -0.5))
        elif issubclass(nptype, np.floating):
            x = nptype(0.5)
        elif issubclass(nptype, np.integer):
            x = nptype(2)
        # Handle the weird types
        try: diff = np.abs(type(np.sqrt(x))(sqrt(x)) - np.sqrt(x))
        except: continue
        assert diff < 2.0**-53
    #Fraction and Decimal
    oldprec = mp.prec
    mp.prec = 1000
    decimal.getcontext().prec = mp.dps
    assert sqrt(Fraction(2, 3)).ae(sqrt(mpf('2/3')))
    assert sqrt(Decimal(2)/Decimal(3)).ae(sqrt(mpf('2/3')))
    mp.prec = oldprec

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\interval\test_interval.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


@pytest.fixture(
    params=[
        (Index([0, 2, 4]), Index([1, 3, 5])),
        (Index([0.0, 1.0, 2.0]), Index([1.0, 2.0, 3.0])),
        (timedelta_range("0 days", periods=3), timedelta_range("1 day", periods=3)),
        (date_range("20170101", periods=3), date_range("20170102", periods=3)),
        (
            date_range("20170101", periods=3, tz="US/Eastern"),
            date_range("20170102", periods=3, tz="US/Eastern"),
        ),
    ],
    ids=lambda x: str(x[0].dtype),
)
def left_right_dtypes(request):
    """
    Fixture for building an IntervalArray from various dtypes
    """
    return request.param


class TestAttributes:
    @pytest.mark.parametrize(
        "left, right",
        [
            (0, 1),
            (Timedelta("0 days"), Timedelta("1 day")),
            (Timestamp("2018-01-01"), Timestamp("2018-01-02")),
            (
                Timestamp("2018-01-01", tz="US/Eastern"),
                Timestamp("2018-01-02", tz="US/Eastern"),
            ),
        ],
    )
    @pytest.mark.parametrize("constructor", [IntervalArray, IntervalIndex])
    def test_is_empty(self, constructor, left, right, closed):
        # GH27219
        tuples = [(left, left), (left, right), np.nan]
        expected = np.array([closed != "both", False, False])
        result = constructor.from_tuples(tuples, closed=closed).is_empty
        tm.assert_numpy_array_equal(result, expected)


class TestMethods:
    @pytest.mark.parametrize("new_closed", ["left", "right", "both", "neither"])
    def test_set_closed(self, closed, new_closed):
        # GH 21670
        array = IntervalArray.from_breaks(range(10), closed=closed)
        result = array.set_closed(new_closed)
        expected = IntervalArray.from_breaks(range(10), closed=new_closed)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            Interval(0, 1, closed="right"),
            IntervalArray.from_breaks([1, 2, 3, 4], closed="right"),
        ],
    )
    def test_where_raises(self, other):
        # GH#45768 The IntervalArray methods raises; the Series method coerces
        ser = pd.Series(IntervalArray.from_breaks([1, 2, 3, 4], closed="left"))
        mask = np.array([True, False, True])
        match = "'value.closed' is 'right', expected 'left'."
        with pytest.raises(ValueError, match=match):
            ser.array._where(mask, other)

        res = ser.where(mask, other=other)
        expected = ser.astype(object).where(mask, other)
        tm.assert_series_equal(res, expected)

    def test_shift(self):
        # https://github.com/pandas-dev/pandas/issues/31495, GH#22428, GH#31502
        a = IntervalArray.from_breaks([1, 2, 3])
        result = a.shift()
        # int -> float
        expected = IntervalArray.from_tuples([(np.nan, np.nan), (1.0, 2.0)])
        tm.assert_interval_array_equal(result, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            a.shift(1, fill_value=pd.NaT)

    def test_shift_datetime(self):
        # GH#31502, GH#31504
        a = IntervalArray.from_breaks(date_range("2000", periods=4))
        result = a.shift(2)
        expected = a.take([-1, -1, 0], allow_fill=True)
        tm.assert_interval_array_equal(result, expected)

        result = a.shift(-1)
        expected = a.take([1, 2, -1], allow_fill=True)
        tm.assert_interval_array_equal(result, expected)

        msg = "can only insert Interval objects and NA into an IntervalArray"
        with pytest.raises(TypeError, match=msg):
            a.shift(1, fill_value=np.timedelta64("NaT", "ns"))


class TestSetitem:
    def test_set_na(self, left_right_dtypes):
        left, right = left_right_dtypes
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        result = IntervalArray.from_arrays(left, right)

        if result.dtype.subtype.kind not in ["m", "M"]:
            msg = "'value' should be an interval type, got <.*NaTType'> instead."
            with pytest.raises(TypeError, match=msg):
                result[0] = pd.NaT
        if result.dtype.subtype.kind in ["i", "u"]:
            msg = "Cannot set float NaN to integer-backed IntervalArray"
            # GH#45484 TypeError, not ValueError, matches what we get with
            # non-NA un-holdable value.
            with pytest.raises(TypeError, match=msg):
                result[0] = np.nan
            return

        result[0] = np.nan

        expected_left = Index([left._na_value] + list(left[1:]))
        expected_right = Index([right._na_value] + list(right[1:]))
        expected = IntervalArray.from_arrays(expected_left, expected_right)

        tm.assert_extension_array_equal(result, expected)

    def test_setitem_mismatched_closed(self):
        arr = IntervalArray.from_breaks(range(4))
        orig = arr.copy()
        other = arr.set_closed("both")

        msg = "'value.closed' is 'both', expected 'right'"
        with pytest.raises(ValueError, match=msg):
            arr[0] = other[0]
        with pytest.raises(ValueError, match=msg):
            arr[:1] = other[:1]
        with pytest.raises(ValueError, match=msg):
            arr[:0] = other[:0]
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1]
        with pytest.raises(ValueError, match=msg):
            arr[:] = list(other[::-1])
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype(object)
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype("category")

        # empty list should be no-op
        arr[:0] = []
        tm.assert_interval_array_equal(arr, orig)


class TestReductions:
    def test_min_max_invalid_axis(self, left_right_dtypes):
        left, right = left_right_dtypes
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        arr = IntervalArray.from_arrays(left, right)

        msg = "`axis` must be fewer than the number of dimensions"
        for axis in [-2, 1]:
            with pytest.raises(ValueError, match=msg):
                arr.min(axis=axis)
            with pytest.raises(ValueError, match=msg):
                arr.max(axis=axis)

        msg = "'>=' not supported between"
        with pytest.raises(TypeError, match=msg):
            arr.min(axis="foo")
        with pytest.raises(TypeError, match=msg):
            arr.max(axis="foo")

    def test_min_max(self, left_right_dtypes, index_or_series_or_array):
        # GH#44746
        left, right = left_right_dtypes
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        arr = IntervalArray.from_arrays(left, right)

        # The expected results below are only valid if monotonic
        assert left.is_monotonic_increasing
        assert Index(arr).is_monotonic_increasing

        MIN = arr[0]
        MAX = arr[-1]

        indexer = np.arange(len(arr))
        np.random.default_rng(2).shuffle(indexer)
        arr = arr.take(indexer)

        arr_na = arr.insert(2, np.nan)

        arr = index_or_series_or_array(arr)
        arr_na = index_or_series_or_array(arr_na)

        for skipna in [True, False]:
            res = arr.min(skipna=skipna)
            assert res == MIN
            assert type(res) == type(MIN)

            res = arr.max(skipna=skipna)
            assert res == MAX
            assert type(res) == type(MAX)

        res = arr_na.min(skipna=False)
        assert np.isnan(res)
        res = arr_na.max(skipna=False)
        assert np.isnan(res)

        res = arr_na.min(skipna=True)
        assert res == MIN
        assert type(res) == type(MIN)
        res = arr_na.max(skipna=True)
        assert res == MAX
        assert type(res) == type(MAX)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_period.py ===
import numpy as np
import pytest

from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
    Series,
    date_range,
    offsets,
    period_range,
)
import pandas._testing as tm


class TestPeriodIndex:
    def test_view_asi8(self):
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.view("i8"), exp)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([492, -9223372036854775808], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.view("i8"), exp)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        exp = np.array([14975, -9223372036854775808], dtype=np.int64)
        idx = PeriodIndex(["2011-01-01", NaT], freq="D")
        tm.assert_numpy_array_equal(idx.view("i8"), exp)
        tm.assert_numpy_array_equal(idx.asi8, exp)

    def test_values(self):
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=object)
        tm.assert_numpy_array_equal(idx.values, exp)
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)

        exp = np.array([], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([Period("2011-01", freq="M"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.values, exp)
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)
        exp = np.array([492, -9223372036854775808], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        idx = PeriodIndex(["2011-01-01", NaT], freq="D")

        exp = np.array([Period("2011-01-01", freq="D"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.values, exp)
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)
        exp = np.array([14975, -9223372036854775808], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.asi8, exp)

    @pytest.mark.parametrize(
        "field",
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "weekofyear",
            "week",
            "dayofweek",
            "day_of_week",
            "dayofyear",
            "day_of_year",
            "quarter",
            "qyear",
            "days_in_month",
        ],
    )
    @pytest.mark.parametrize(
        "periodindex",
        [
            period_range(freq="Y", start="1/1/2001", end="12/1/2005"),
            period_range(freq="Q", start="1/1/2001", end="12/1/2002"),
            period_range(freq="M", start="1/1/2001", end="1/1/2002"),
            period_range(freq="D", start="12/1/2001", end="6/1/2001"),
            period_range(freq="h", start="12/31/2001", end="1/1/2002 23:00"),
            period_range(freq="Min", start="12/31/2001", end="1/1/2002 00:20"),
            period_range(
                freq="s", start="12/31/2001 00:00:00", end="12/31/2001 00:05:00"
            ),
            period_range(end=Period("2006-12-31", "W"), periods=10),
        ],
    )
    def test_fields(self, periodindex, field):
        periods = list(periodindex)
        ser = Series(periodindex)

        field_idx = getattr(periodindex, field)
        assert len(periodindex) == len(field_idx)
        for x, val in zip(periods, field_idx):
            assert getattr(x, field) == val

        if len(ser) == 0:
            return

        field_s = getattr(ser.dt, field)
        assert len(periodindex) == len(field_s)
        for x, val in zip(periods, field_s):
            assert getattr(x, field) == val

    def test_is_(self):
        create_index = lambda: period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        index = create_index()
        assert index.is_(index)
        assert not index.is_(create_index())
        assert index.is_(index.view())
        assert index.is_(index.view().view().view().view().view())
        assert index.view().is_(index)
        ind2 = index.view()
        index.name = "Apple"
        assert ind2.is_(index)
        assert not index.is_(index[:])
        assert not index.is_(index.asfreq("M"))
        assert not index.is_(index.asfreq("Y"))

        assert not index.is_(index - 2)
        assert not index.is_(index - 0)

    def test_index_unique(self):
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2009], freq="Y-JUN")
        expected = PeriodIndex([2000, 2007, 2009], freq="Y-JUN")
        tm.assert_index_equal(idx.unique(), expected)
        assert idx.nunique() == 3

    def test_pindex_fieldaccessor_nat(self):
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2012-03", "2012-04"], freq="D", name="name"
        )

        exp = Index([2011, 2011, -1, 2012, 2012], dtype=np.int64, name="name")
        tm.assert_index_equal(idx.year, exp)
        exp = Index([1, 2, -1, 3, 4], dtype=np.int64, name="name")
        tm.assert_index_equal(idx.month, exp)

    def test_pindex_multiples(self):
        expected = PeriodIndex(
            ["2011-01", "2011-03", "2011-05", "2011-07", "2011-09", "2011-11"],
            freq="2M",
        )

        pi = period_range(start="1/1/11", end="12/31/11", freq="2M")
        tm.assert_index_equal(pi, expected)
        assert pi.freq == offsets.MonthEnd(2)
        assert pi.freqstr == "2M"

        pi = period_range(start="1/1/11", periods=6, freq="2M")
        tm.assert_index_equal(pi, expected)
        assert pi.freq == offsets.MonthEnd(2)
        assert pi.freqstr == "2M"

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings("ignore:Period with BDay freq:FutureWarning")
    def test_iteration(self):
        index = period_range(start="1/1/10", periods=4, freq="B")

        result = list(index)
        assert isinstance(result[0], Period)
        assert result[0].freq == index.freq

    def test_with_multi_index(self):
        # #1705
        index = date_range("1/1/2012", periods=4, freq="12h")
        index_as_arrays = [index.to_period(freq="D"), index.hour]

        s = Series([0, 1, 2, 3], index_as_arrays)

        assert isinstance(s.index.levels[0], PeriodIndex)

        assert isinstance(s.index.values[0][0], Period)

    def test_map(self):
        # test_map_dictlike generally tests

        index = PeriodIndex([2005, 2007, 2009], freq="Y")
        result = index.map(lambda x: x.ordinal)
        exp = Index([x.ordinal for x in index])
        tm.assert_index_equal(result, exp)


def test_maybe_convert_timedelta():
    pi = PeriodIndex(["2000", "2001"], freq="D")
    offset = offsets.Day(2)
    assert pi._maybe_convert_timedelta(offset) == 2
    assert pi._maybe_convert_timedelta(2) == 2

    offset = offsets.BusinessDay()
    msg = r"Input has different freq=B from PeriodIndex\(freq=D\)"
    with pytest.raises(ValueError, match=msg):
        pi._maybe_convert_timedelta(offset)


@pytest.mark.parametrize("array", [True, False])
def test_dunder_array(array):
    obj = PeriodIndex(["2000-01-01", "2001-01-01"], freq="D")
    if array:
        obj = obj._data

    expected = np.array([obj[0], obj[1]], dtype=object)
    result = np.array(obj)
    tm.assert_numpy_array_equal(result, expected)

    result = np.asarray(obj)
    tm.assert_numpy_array_equal(result, expected)

    expected = obj.asi8
    for dtype in ["i8", "int64", np.int64]:
        result = np.array(obj, dtype=dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(obj, dtype=dtype)
        tm.assert_numpy_array_equal(result, expected)

    for dtype in ["float64", "int32", "uint64"]:
        msg = "argument must be"
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=dtype)
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=getattr(np, dtype))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\common\test_ints.py ===
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_int_conversion(all_parsers):
    data = """A,B
1.0,1
2.0,2
3.0,3
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))

    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            "A,B\nTrue,1\nFalse,2\nTrue,3",
            {},
            DataFrame([[True, 1], [False, 2], [True, 3]], columns=["A", "B"]),
        ),
        (
            "A,B\nYES,1\nno,2\nyes,3\nNo,3\nYes,3",
            {"true_values": ["yes", "Yes", "YES"], "false_values": ["no", "NO", "No"]},
            DataFrame(
                [[True, 1], [False, 2], [True, 3], [False, 3], [True, 3]],
                columns=["A", "B"],
            ),
        ),
        (
            "A,B\nTRUE,1\nFALSE,2\nTRUE,3",
            {},
            DataFrame([[True, 1], [False, 2], [True, 3]], columns=["A", "B"]),
        ),
        (
            "A,B\nfoo,bar\nbar,foo",
            {"true_values": ["foo"], "false_values": ["bar"]},
            DataFrame([[True, False], [False, True]], columns=["A", "B"]),
        ),
    ],
)
def test_parse_bool(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)


def test_parse_integers_above_fp_precision(all_parsers):
    data = """Numbers
17007000002000191
17007000002000191
17007000002000191
17007000002000191
17007000002000192
17007000002000192
17007000002000192
17007000002000192
17007000002000192
17007000002000194"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        {
            "Numbers": [
                17007000002000191,
                17007000002000191,
                17007000002000191,
                17007000002000191,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000192,
                17007000002000194,
            ]
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("sep", [" ", r"\s+"])
def test_integer_overflow_bug(all_parsers, sep):
    # see gh-2601
    data = "65248E10 11\n55555E55 22\n"
    parser = all_parsers
    if parser.engine == "pyarrow" and sep != " ":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, sep=sep)
        return

    result = parser.read_csv(StringIO(data), header=None, sep=sep)
    expected = DataFrame([[6.5248e14, 11], [5.5555e59, 22]])
    tm.assert_frame_equal(result, expected)


def test_int64_min_issues(all_parsers):
    # see gh-2599
    parser = all_parsers
    data = "A,B\n0,0\n0,"
    result = parser.read_csv(StringIO(data))

    expected = DataFrame({"A": [0, 0], "B": [0, np.nan]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conv", [None, np.int64, np.uint64])
def test_int64_overflow(all_parsers, conv, request):
    data = """ID
00013007854817840016671868
00013007854817840016749251
00013007854817840016754630
00013007854817840016781876
00013007854817840017028824
00013007854817840017963235
00013007854817840018860166"""
    parser = all_parsers

    if conv is None:
        # 13007854817840016671868 > UINT64_MAX, so this
        # will overflow and return object as the dtype.
        if parser.engine == "pyarrow":
            mark = pytest.mark.xfail(reason="parses to float64")
            request.applymarker(mark)

        result = parser.read_csv(StringIO(data))
        expected = DataFrame(
            [
                "00013007854817840016671868",
                "00013007854817840016749251",
                "00013007854817840016754630",
                "00013007854817840016781876",
                "00013007854817840017028824",
                "00013007854817840017963235",
                "00013007854817840018860166",
            ],
            columns=["ID"],
        )
        tm.assert_frame_equal(result, expected)
    else:
        # 13007854817840016671868 > UINT64_MAX, so attempts
        # to cast to either int64 or uint64 will result in
        # an OverflowError being raised.
        msg = "|".join(
            [
                "Python int too large to convert to C long",
                "long too big to convert",
                "int too big to convert",
            ]
        )
        err = OverflowError
        if parser.engine == "pyarrow":
            err = ValueError
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"

        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), converters={"ID": conv})


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "val", [np.iinfo(np.uint64).max, np.iinfo(np.int64).max, np.iinfo(np.int64).min]
)
def test_int64_uint64_range(all_parsers, val):
    # These numbers fall right inside the int64-uint64
    # range, so they should be parsed as string.
    parser = all_parsers
    result = parser.read_csv(StringIO(str(val)), header=None)

    expected = DataFrame([val])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
@pytest.mark.parametrize(
    "val", [np.iinfo(np.uint64).max + 1, np.iinfo(np.int64).min - 1]
)
def test_outside_int64_uint64_range(all_parsers, val):
    # These numbers fall just outside the int64-uint64
    # range, so they should be parsed as string.
    parser = all_parsers
    result = parser.read_csv(StringIO(str(val)), header=None)

    expected = DataFrame([str(val)])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # gets float64 dtype instead of object
@pytest.mark.parametrize("exp_data", [[str(-1), str(2**63)], [str(2**63), str(-1)]])
def test_numeric_range_too_wide(all_parsers, exp_data):
    # No numerical dtype can hold both negative and uint64
    # values, so they should be cast as string.
    parser = all_parsers
    data = "\n".join(exp_data)
    expected = DataFrame(exp_data)

    result = parser.read_csv(StringIO(data), header=None)
    tm.assert_frame_equal(result, expected)


def test_integer_precision(all_parsers):
    # Gh 7072
    s = """1,1;0;0;0;1;1;3844;3844;3844;1;1;1;1;1;1;0;0;1;1;0;0,,,4321583677327450765
5,1;0;0;0;1;1;843;843;843;1;1;1;1;1;1;0;0;1;1;0;0,64.0,;,4321113141090630389"""
    parser = all_parsers
    result = parser.read_csv(StringIO(s), header=None)[4]
    expected = Series([4321583677327450765, 4321113141090630389], name=4)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\conftest.py ===
import operator

import pytest

from pandas._config.config import _get_option

from pandas import (
    Series,
    options,
)


@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    raise NotImplementedError


@pytest.fixture
def data():
    """
    Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    raise NotImplementedError


@pytest.fixture
def data_for_twos(dtype):
    """
    Length-100 array in which all the elements are two.

    Call pytest.skip in your fixture if the dtype does not support divmod.
    """
    if not (dtype._is_numeric or dtype.kind == "m"):
        # Object-dtypes may want to allow this, but for the most part
        #  only numeric and timedelta-like dtypes will need to implement this.
        pytest.skip(f"{dtype} is not a numeric dtype")

    raise NotImplementedError


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    raise NotImplementedError


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C

    For boolean dtypes (for which there are only 2 values available),
    set B=C=True
    """
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting():
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError


@pytest.fixture
def na_cmp():
    """
    Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.

    By default, uses ``operator.is_``
    """
    return operator.is_


@pytest.fixture
def na_value(dtype):
    """
    The scalar missing value for this type. Default dtype.na_value.

    TODO: can be removed in 3.x (see https://github.com/pandas-dev/pandas/pull/54930)
    """
    return dtype.na_value


@pytest.fixture
def data_for_grouping():
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing.

    If a dtype has _is_boolean = True, i.e. only 2 unique non-NA entries,
    then set C=B.
    """
    raise NotImplementedError


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


@pytest.fixture
def invalid_scalar(data):
    """
    A scalar that *cannot* be held by this ExtensionArray.

    The default should work for most subclasses, but is not guaranteed.

    If the array can hold any item (i.e. object dtype), then use pytest.skip.
    """
    return object.__new__(object)


@pytest.fixture
def using_copy_on_write() -> bool:
    """
    Fixture to check if Copy-on-Write is enabled.
    """
    return (
        options.mode.copy_on_write is True
        and _get_option("mode.data_manager", silent=True) == "block"
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_dataframe.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    concat,
)
import pandas._testing as tm


class TestDataFrameConcat:
    def test_concat_multiple_frames_dtypes(self):
        # GH#2759
        df1 = DataFrame(data=np.ones((10, 2)), columns=["foo", "bar"], dtype=np.float64)
        df2 = DataFrame(data=np.ones((10, 2)), dtype=np.float32)
        results = concat((df1, df2), axis=1).dtypes
        expected = Series(
            [np.dtype("float64")] * 2 + [np.dtype("float32")] * 2,
            index=["foo", "bar", 0, 1],
        )
        tm.assert_series_equal(results, expected)

    def test_concat_tuple_keys(self):
        # GH#14438
        df1 = DataFrame(np.ones((2, 2)), columns=list("AB"))
        df2 = DataFrame(np.ones((3, 2)) * 2, columns=list("AB"))
        results = concat((df1, df2), keys=[("bee", "bah"), ("bee", "boo")])
        expected = DataFrame(
            {
                "A": {
                    ("bee", "bah", 0): 1.0,
                    ("bee", "bah", 1): 1.0,
                    ("bee", "boo", 0): 2.0,
                    ("bee", "boo", 1): 2.0,
                    ("bee", "boo", 2): 2.0,
                },
                "B": {
                    ("bee", "bah", 0): 1.0,
                    ("bee", "bah", 1): 1.0,
                    ("bee", "boo", 0): 2.0,
                    ("bee", "boo", 1): 2.0,
                    ("bee", "boo", 2): 2.0,
                },
            }
        )
        tm.assert_frame_equal(results, expected)

    def test_concat_named_keys(self):
        # GH#14252
        df = DataFrame({"foo": [1, 2], "bar": [0.1, 0.2]})
        index = Index(["a", "b"], name="baz")
        concatted_named_from_keys = concat([df, df], keys=index)
        expected_named = DataFrame(
            {"foo": [1, 2, 1, 2], "bar": [0.1, 0.2, 0.1, 0.2]},
            index=pd.MultiIndex.from_product((["a", "b"], [0, 1]), names=["baz", None]),
        )
        tm.assert_frame_equal(concatted_named_from_keys, expected_named)

        index_no_name = Index(["a", "b"], name=None)
        concatted_named_from_names = concat([df, df], keys=index_no_name, names=["baz"])
        tm.assert_frame_equal(concatted_named_from_names, expected_named)

        concatted_unnamed = concat([df, df], keys=index_no_name)
        expected_unnamed = DataFrame(
            {"foo": [1, 2, 1, 2], "bar": [0.1, 0.2, 0.1, 0.2]},
            index=pd.MultiIndex.from_product((["a", "b"], [0, 1]), names=[None, None]),
        )
        tm.assert_frame_equal(concatted_unnamed, expected_unnamed)

    def test_concat_axis_parameter(self):
        # GH#14369
        df1 = DataFrame({"A": [0.1, 0.2]}, index=range(2))
        df2 = DataFrame({"A": [0.3, 0.4]}, index=range(2))

        # Index/row/0 DataFrame
        expected_index = DataFrame({"A": [0.1, 0.2, 0.3, 0.4]}, index=[0, 1, 0, 1])

        concatted_index = concat([df1, df2], axis="index")
        tm.assert_frame_equal(concatted_index, expected_index)

        concatted_row = concat([df1, df2], axis="rows")
        tm.assert_frame_equal(concatted_row, expected_index)

        concatted_0 = concat([df1, df2], axis=0)
        tm.assert_frame_equal(concatted_0, expected_index)

        # Columns/1 DataFrame
        expected_columns = DataFrame(
            [[0.1, 0.3], [0.2, 0.4]], index=[0, 1], columns=["A", "A"]
        )

        concatted_columns = concat([df1, df2], axis="columns")
        tm.assert_frame_equal(concatted_columns, expected_columns)

        concatted_1 = concat([df1, df2], axis=1)
        tm.assert_frame_equal(concatted_1, expected_columns)

        series1 = Series([0.1, 0.2])
        series2 = Series([0.3, 0.4])

        # Index/row/0 Series
        expected_index_series = Series([0.1, 0.2, 0.3, 0.4], index=[0, 1, 0, 1])

        concatted_index_series = concat([series1, series2], axis="index")
        tm.assert_series_equal(concatted_index_series, expected_index_series)

        concatted_row_series = concat([series1, series2], axis="rows")
        tm.assert_series_equal(concatted_row_series, expected_index_series)

        concatted_0_series = concat([series1, series2], axis=0)
        tm.assert_series_equal(concatted_0_series, expected_index_series)

        # Columns/1 Series
        expected_columns_series = DataFrame(
            [[0.1, 0.3], [0.2, 0.4]], index=[0, 1], columns=[0, 1]
        )

        concatted_columns_series = concat([series1, series2], axis="columns")
        tm.assert_frame_equal(concatted_columns_series, expected_columns_series)

        concatted_1_series = concat([series1, series2], axis=1)
        tm.assert_frame_equal(concatted_1_series, expected_columns_series)

        # Testing ValueError
        with pytest.raises(ValueError, match="No axis named"):
            concat([series1, series2], axis="something")

    def test_concat_numerical_names(self):
        # GH#15262, GH#12223
        df = DataFrame(
            {"col": range(9)},
            dtype="int32",
            index=(
                pd.MultiIndex.from_product(
                    [["A0", "A1", "A2"], ["B0", "B1", "B2"]], names=[1, 2]
                )
            ),
        )
        result = concat((df.iloc[:2, :], df.iloc[-2:, :]))
        expected = DataFrame(
            {"col": [0, 1, 7, 8]},
            dtype="int32",
            index=pd.MultiIndex.from_tuples(
                [("A0", "B0"), ("A0", "B1"), ("A2", "B1"), ("A2", "B2")], names=[1, 2]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_astype_dup_col(self):
        # GH#23049
        df = DataFrame([{"a": "b"}])
        df = concat([df, df], axis=1)

        result = df.astype("category")
        expected = DataFrame(
            np.array(["b", "b"]).reshape(1, 2), columns=["a", "a"]
        ).astype("category")
        tm.assert_frame_equal(result, expected)

    def test_concat_dataframe_keys_bug(self, sort):
        t1 = DataFrame(
            {"value": Series([1, 2, 3], index=Index(["a", "b", "c"], name="id"))}
        )
        t2 = DataFrame({"value": Series([7, 8], index=Index(["a", "b"], name="id"))})

        # it works
        result = concat([t1, t2], axis=1, keys=["t1", "t2"], sort=sort)
        assert list(result.columns) == [("t1", "value"), ("t2", "value")]

    def test_concat_bool_with_int(self):
        # GH#42092 we may want to change this to return object, but that
        #  would need a deprecation
        df1 = DataFrame(Series([True, False, True, True], dtype="bool"))
        df2 = DataFrame(Series([1, 0, 1], dtype="int64"))

        result = concat([df1, df2])
        expected = concat([df1.astype("int64"), df2])
        tm.assert_frame_equal(result, expected)

    def test_concat_duplicates_in_index_with_keys(self):
        # GH#42651
        index = [1, 1, 3]
        data = [1, 2, 3]

        df = DataFrame(data=data, index=index)
        result = concat([df], keys=["A"], names=["ID", "date"])
        mi = pd.MultiIndex.from_product([["A"], index], names=["ID", "date"])
        expected = DataFrame(data=data, index=mi)
        tm.assert_frame_equal(result, expected)
        tm.assert_index_equal(result.index.levels[1], Index([1, 3], name="date"))

    @pytest.mark.parametrize("ignore_index", [True, False])
    @pytest.mark.parametrize("order", ["C", "F"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_concat_copies(self, axis, order, ignore_index, using_copy_on_write):
        # based on asv ConcatDataFrames
        df = DataFrame(np.zeros((10, 5), dtype=np.float32, order=order))

        res = concat([df] * 5, axis=axis, ignore_index=ignore_index, copy=True)

        if not using_copy_on_write:
            for arr in res._iter_column_arrays():
                for arr2 in df._iter_column_arrays():
                    assert not np.shares_memory(arr, arr2)

    def test_outer_sort_columns(self):
        # GH#47127
        df1 = DataFrame({"A": [0], "B": [1], 0: 1})
        df2 = DataFrame({"A": [100]})
        result = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected = DataFrame({0: [1.0, np.nan], "A": [0, 100], "B": [1.0, np.nan]})
        tm.assert_frame_equal(result, expected)

    def test_inner_sort_columns(self):
        # GH#47127
        df1 = DataFrame({"A": [0], "B": [1], 0: 1})
        df2 = DataFrame({"A": [100], 0: 2})
        result = concat([df1, df2], ignore_index=True, join="inner", sort=True)
        expected = DataFrame({0: [1, 2], "A": [0, 100]})
        tm.assert_frame_equal(result, expected)

    def test_sort_columns_one_df(self):
        # GH#47127
        df1 = DataFrame({"A": [100], 0: 2})
        result = concat([df1], ignore_index=True, join="inner", sort=True)
        expected = DataFrame({0: [2], "A": [100]})
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\interval\test_interval_new.py ===
import re

import numpy as np
import pytest

from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Series,
)
import pandas._testing as tm


class TestIntervalIndex:
    @pytest.fixture
    def series_with_interval_index(self):
        return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))

    def test_loc_with_interval(self, series_with_interval_index, indexer_sl):
        # loc with single label / list of labels:
        #   - Intervals: only exact matches
        #   - scalars: those that contain it

        ser = series_with_interval_index.copy()

        expected = 0
        result = indexer_sl(ser)[Interval(0, 1)]
        assert result == expected

        expected = ser.iloc[3:5]
        result = indexer_sl(ser)[[Interval(3, 4), Interval(4, 5)]]
        tm.assert_series_equal(expected, result)

        # missing or not exact
        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='left')")):
            indexer_sl(ser)[Interval(3, 5, closed="left")]

        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='right')")):
            indexer_sl(ser)[Interval(3, 5)]

        with pytest.raises(
            KeyError, match=re.escape("Interval(-2, 0, closed='right')")
        ):
            indexer_sl(ser)[Interval(-2, 0)]

        with pytest.raises(KeyError, match=re.escape("Interval(5, 6, closed='right')")):
            indexer_sl(ser)[Interval(5, 6)]

    def test_loc_with_scalar(self, series_with_interval_index, indexer_sl):
        # loc with single label / list of labels:
        #   - Intervals: only exact matches
        #   - scalars: those that contain it

        ser = series_with_interval_index.copy()

        assert indexer_sl(ser)[1] == 0
        assert indexer_sl(ser)[1.5] == 1
        assert indexer_sl(ser)[2] == 1

        expected = ser.iloc[1:4]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2.5, 3.5]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[2, 3, 4]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 3, 4]])

        expected = ser.iloc[[1, 1, 2, 1]]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2, 2.5, 1.5]])

        expected = ser.iloc[2:5]
        tm.assert_series_equal(expected, indexer_sl(ser)[ser >= 2])

    def test_loc_with_slices(self, series_with_interval_index, indexer_sl):
        # loc with slices:
        #   - Interval objects: only works with exact matches
        #   - scalars: only works for non-overlapping, monotonic intervals,
        #     and start/stop select location based on the interval that
        #     contains them:
        #    (slice_loc(start, stop) == (idx.get_loc(start), idx.get_loc(stop))

        ser = series_with_interval_index.copy()

        # slice of interval

        expected = ser.iloc[:3]
        result = indexer_sl(ser)[Interval(0, 1) : Interval(2, 3)]
        tm.assert_series_equal(expected, result)

        expected = ser.iloc[3:]
        result = indexer_sl(ser)[Interval(3, 4) :]
        tm.assert_series_equal(expected, result)

        msg = "Interval objects are not currently supported"
        with pytest.raises(NotImplementedError, match=msg):
            indexer_sl(ser)[Interval(3, 6) :]

        with pytest.raises(NotImplementedError, match=msg):
            indexer_sl(ser)[Interval(3, 4, closed="left") :]

    def test_slice_step_ne1(self, series_with_interval_index):
        # GH#31658 slice of scalar with step != 1
        ser = series_with_interval_index.copy()
        expected = ser.iloc[0:4:2]

        result = ser[0:4:2]
        tm.assert_series_equal(result, expected)

        result2 = ser[0:4][::2]
        tm.assert_series_equal(result2, expected)

    def test_slice_float_start_stop(self, series_with_interval_index):
        # GH#31658 slicing with integers is positional, with floats is not
        #  supported
        ser = series_with_interval_index.copy()

        msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
        with pytest.raises(ValueError, match=msg):
            ser[1.5:9.5:2]

    def test_slice_interval_step(self, series_with_interval_index):
        # GH#31658 allows for integer step!=1, not Interval step
        ser = series_with_interval_index.copy()
        msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
        with pytest.raises(ValueError, match=msg):
            ser[0 : 4 : Interval(0, 1)]

    def test_loc_with_overlap(self, indexer_sl):
        idx = IntervalIndex.from_tuples([(1, 5), (3, 7)])
        ser = Series(range(len(idx)), index=idx)

        # scalar
        expected = ser
        result = indexer_sl(ser)[4]
        tm.assert_series_equal(expected, result)

        result = indexer_sl(ser)[[4]]
        tm.assert_series_equal(expected, result)

        # interval
        expected = 0
        result = indexer_sl(ser)[Interval(1, 5)]
        assert expected == result

        expected = ser
        result = indexer_sl(ser)[[Interval(1, 5), Interval(3, 7)]]
        tm.assert_series_equal(expected, result)

        with pytest.raises(KeyError, match=re.escape("Interval(3, 5, closed='right')")):
            indexer_sl(ser)[Interval(3, 5)]

        msg = (
            r"None of \[IntervalIndex\(\[\(3, 5\]\], "
            r"dtype='interval\[int64, right\]'\)\] are in the \[index\]"
        )
        with pytest.raises(KeyError, match=msg):
            indexer_sl(ser)[[Interval(3, 5)]]

        # slices with interval (only exact matches)
        expected = ser
        result = indexer_sl(ser)[Interval(1, 5) : Interval(3, 7)]
        tm.assert_series_equal(expected, result)

        msg = (
            "'can only get slices from an IntervalIndex if bounds are "
            "non-overlapping and all monotonic increasing or decreasing'"
        )
        with pytest.raises(KeyError, match=msg):
            indexer_sl(ser)[Interval(1, 6) : Interval(3, 8)]

        if indexer_sl is tm.loc:
            # slices with scalar raise for overlapping intervals
            # TODO KeyError is the appropriate error?
            with pytest.raises(KeyError, match=msg):
                ser.loc[1:4]

    def test_non_unique(self, indexer_sl):
        idx = IntervalIndex.from_tuples([(1, 3), (3, 7)])
        ser = Series(range(len(idx)), index=idx)

        result = indexer_sl(ser)[Interval(1, 3)]
        assert result == 0

        result = indexer_sl(ser)[[Interval(1, 3)]]
        expected = ser.iloc[0:1]
        tm.assert_series_equal(expected, result)

    def test_non_unique_moar(self, indexer_sl):
        idx = IntervalIndex.from_tuples([(1, 3), (1, 3), (3, 7)])
        ser = Series(range(len(idx)), index=idx)

        expected = ser.iloc[[0, 1]]
        result = indexer_sl(ser)[Interval(1, 3)]
        tm.assert_series_equal(expected, result)

        expected = ser
        result = indexer_sl(ser)[Interval(1, 3) :]
        tm.assert_series_equal(expected, result)

        expected = ser.iloc[[0, 1]]
        result = indexer_sl(ser)[[Interval(1, 3)]]
        tm.assert_series_equal(expected, result)

    def test_loc_getitem_missing_key_error_message(
        self, frame_or_series, series_with_interval_index
    ):
        # GH#27365
        ser = series_with_interval_index.copy()
        obj = frame_or_series(ser)
        with pytest.raises(KeyError, match=r"\[6\]"):
            obj.loc[[4, 5, 6]]


@pytest.mark.parametrize(
    "intervals",
    [
        ([Interval(-np.inf, 0.0), Interval(0.0, 1.0)]),
        ([Interval(-np.inf, -2.0), Interval(-2.0, -1.0)]),
        ([Interval(-1.0, 0.0), Interval(0.0, np.inf)]),
        ([Interval(1.0, 2.0), Interval(2.0, np.inf)]),
    ],
)
def test_repeating_interval_index_with_infs(intervals):
    # GH 46658

    interval_index = Index(intervals * 51)

    expected = np.arange(1, 102, 2, dtype=np.intp)
    result = interval_index.get_indexer_for([intervals[1]])

    tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_gcs.py ===
from io import BytesIO
import os
import pathlib
import tarfile
import zipfile

import numpy as np
import pytest

from pandas.compat.pyarrow import pa_version_under17p0

from pandas import (
    DataFrame,
    Index,
    date_range,
    read_csv,
    read_excel,
    read_json,
    read_parquet,
)
import pandas._testing as tm
from pandas.util import _test_decorators as td

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def gcs_buffer():
    """Emulate GCS using a binary buffer."""
    pytest.importorskip("gcsfs")
    fsspec = pytest.importorskip("fsspec")

    gcs_buffer = BytesIO()
    gcs_buffer.close = lambda: True

    class MockGCSFileSystem(fsspec.AbstractFileSystem):
        @staticmethod
        def open(*args, **kwargs):
            gcs_buffer.seek(0)
            return gcs_buffer

        def ls(self, path, **kwargs):
            # needed for pyarrow
            return [{"name": path, "type": "file"}]

    # Overwrites the default implementation from gcsfs to our mock class
    fsspec.register_implementation("gs", MockGCSFileSystem, clobber=True)

    return gcs_buffer


# Patches pyarrow; other processes should not pick up change
@pytest.mark.single_cpu
@pytest.mark.parametrize("format", ["csv", "json", "parquet", "excel", "markdown"])
def test_to_read_gcs(gcs_buffer, format, monkeypatch, capsys, request):
    """
    Test that many to/read functions support GCS.

    GH 33987
    """

    df1 = DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )

    path = f"gs://test/test.{format}"

    if format == "csv":
        df1.to_csv(path, index=True)
        df2 = read_csv(path, parse_dates=["dt"], index_col=0)
    elif format == "excel":
        path = "gs://test/test.xlsx"
        df1.to_excel(path)
        df2 = read_excel(path, parse_dates=["dt"], index_col=0)
    elif format == "json":
        df1.to_json(path)
        df2 = read_json(path, convert_dates=["dt"])
    elif format == "parquet":
        pytest.importorskip("pyarrow")
        pa_fs = pytest.importorskip("pyarrow.fs")

        class MockFileSystem(pa_fs.FileSystem):
            @staticmethod
            def from_uri(path):
                print("Using pyarrow filesystem")
                to_local = pathlib.Path(path.replace("gs://", "")).absolute().as_uri()
                return pa_fs.LocalFileSystem(to_local)

        request.applymarker(
            pytest.mark.xfail(
                not pa_version_under17p0,
                raises=TypeError,
                reason="pyarrow 17 broke the mocked filesystem",
            )
        )
        with monkeypatch.context() as m:
            m.setattr(pa_fs, "FileSystem", MockFileSystem)
            df1.to_parquet(path)
            df2 = read_parquet(path)
        captured = capsys.readouterr()
        assert captured.out == "Using pyarrow filesystem\nUsing pyarrow filesystem\n"
    elif format == "markdown":
        pytest.importorskip("tabulate")
        df1.to_markdown(path)
        df2 = df1

    tm.assert_frame_equal(df1, df2)


def assert_equal_zip_safe(result: bytes, expected: bytes, compression: str):
    """
    For zip compression, only compare the CRC-32 checksum of the file contents
    to avoid checking the time-dependent last-modified timestamp which
    in some CI builds is off-by-one

    See https://en.wikipedia.org/wiki/ZIP_(file_format)#File_headers
    """
    if compression == "zip":
        # Only compare the CRC checksum of the file contents
        with zipfile.ZipFile(BytesIO(result)) as exp, zipfile.ZipFile(
            BytesIO(expected)
        ) as res:
            for res_info, exp_info in zip(res.infolist(), exp.infolist()):
                assert res_info.CRC == exp_info.CRC
    elif compression == "tar":
        with tarfile.open(fileobj=BytesIO(result)) as tar_exp, tarfile.open(
            fileobj=BytesIO(expected)
        ) as tar_res:
            for tar_res_info, tar_exp_info in zip(
                tar_res.getmembers(), tar_exp.getmembers()
            ):
                actual_file = tar_res.extractfile(tar_res_info)
                expected_file = tar_exp.extractfile(tar_exp_info)
                assert (actual_file is None) == (expected_file is None)
                if actual_file is not None and expected_file is not None:
                    assert actual_file.read() == expected_file.read()
    else:
        assert result == expected


@pytest.mark.parametrize("encoding", ["utf-8", "cp1251"])
def test_to_csv_compression_encoding_gcs(
    gcs_buffer, compression_only, encoding, compression_to_extension
):
    """
    Compression and encoding should with GCS.

    GH 35677 (to_csv, compression), GH 26124 (to_csv, encoding), and
    GH 32392 (read_csv, encoding)
    """
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    # reference of compressed and encoded file
    compression = {"method": compression_only}
    if compression_only == "gzip":
        compression["mtime"] = 1  # be reproducible
    buffer = BytesIO()
    df.to_csv(buffer, compression=compression, encoding=encoding, mode="wb")

    # write compressed file with explicit compression
    path_gcs = "gs://test/test.csv"
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    res = gcs_buffer.getvalue()
    expected = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)

    read_df = read_csv(
        path_gcs, index_col=0, compression=compression_only, encoding=encoding
    )
    tm.assert_frame_equal(df, read_df)

    # write compressed file with implicit compression
    file_ext = compression_to_extension[compression_only]
    compression["method"] = "infer"
    path_gcs += f".{file_ext}"
    df.to_csv(path_gcs, compression=compression, encoding=encoding)

    res = gcs_buffer.getvalue()
    expected = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)

    read_df = read_csv(path_gcs, index_col=0, compression="infer", encoding=encoding)
    tm.assert_frame_equal(df, read_df)


def test_to_parquet_gcs_new_file(monkeypatch, tmpdir):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip("fastparquet")
    pytest.importorskip("gcsfs")

    from fsspec import AbstractFileSystem

    df1 = DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )

    class MockGCSFileSystem(AbstractFileSystem):
        def open(self, path, mode="r", *args):
            if "w" not in mode:
                raise FileNotFoundError
            return open(os.path.join(tmpdir, "test.parquet"), mode, encoding="utf-8")

    monkeypatch.setattr("gcsfs.GCSFileSystem", MockGCSFileSystem)
    df1.to_parquet(
        "gs://test/test.csv", index=True, engine="fastparquet", compression=None
    )


@td.skip_if_installed("gcsfs")
def test_gcs_not_present_exception():
    with tm.external_error_raised(ImportError):
        read_csv("gs://test/test.csv")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_special.py ===
from sympy.core.add import Add
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.core.relational import Eq
from sympy.concrete.summations import Sum
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import (
    ZeroMatrix, GenericZeroMatrix, Identity, GenericIdentity, OneMatrix)
from sympy.matrices.expressions.matmul import MatMul
from sympy.testing.pytest import raises


def test_zero_matrix_creation():
    assert unchanged(ZeroMatrix, 2, 2)
    assert unchanged(ZeroMatrix, 0, 0)
    raises(ValueError, lambda: ZeroMatrix(-1, 2))
    raises(ValueError, lambda: ZeroMatrix(2.0, 2))
    raises(ValueError, lambda: ZeroMatrix(2j, 2))
    raises(ValueError, lambda: ZeroMatrix(2, -1))
    raises(ValueError, lambda: ZeroMatrix(2, 2.0))
    raises(ValueError, lambda: ZeroMatrix(2, 2j))

    n = symbols('n')
    assert unchanged(ZeroMatrix, n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: ZeroMatrix(n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: ZeroMatrix(n, n))


def test_generic_zero_matrix():
    z = GenericZeroMatrix()
    n = symbols('n', integer=True)
    A = MatrixSymbol("A", n, n)

    assert z == z
    assert z != A
    assert A != z

    assert z.is_ZeroMatrix

    raises(TypeError, lambda: z.shape)
    raises(TypeError, lambda: z.rows)
    raises(TypeError, lambda: z.cols)

    assert MatAdd() == z
    assert MatAdd(z, A) == MatAdd(A)
    # Make sure it is hashable
    hash(z)


def test_identity_matrix_creation():
    assert Identity(2)
    assert Identity(0)
    raises(ValueError, lambda: Identity(-1))
    raises(ValueError, lambda: Identity(2.0))
    raises(ValueError, lambda: Identity(2j))

    n = symbols('n')
    assert Identity(n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: Identity(n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: Identity(n))


def test_generic_identity():
    I = GenericIdentity()
    n = symbols('n', integer=True)
    A = MatrixSymbol("A", n, n)

    assert I == I
    assert I != A
    assert A != I

    assert I.is_Identity
    assert I**-1 == I

    raises(TypeError, lambda: I.shape)
    raises(TypeError, lambda: I.rows)
    raises(TypeError, lambda: I.cols)

    assert MatMul() == I
    assert MatMul(I, A) == MatMul(A)
    # Make sure it is hashable
    hash(I)


def test_one_matrix_creation():
    assert OneMatrix(2, 2)
    assert OneMatrix(0, 0)
    assert Eq(OneMatrix(1, 1), Identity(1))
    raises(ValueError, lambda: OneMatrix(-1, 2))
    raises(ValueError, lambda: OneMatrix(2.0, 2))
    raises(ValueError, lambda: OneMatrix(2j, 2))
    raises(ValueError, lambda: OneMatrix(2, -1))
    raises(ValueError, lambda: OneMatrix(2, 2.0))
    raises(ValueError, lambda: OneMatrix(2, 2j))

    n = symbols('n')
    assert OneMatrix(n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: OneMatrix(n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: OneMatrix(n, n))


def test_ZeroMatrix():
    n, m = symbols('n m', integer=True)
    A = MatrixSymbol('A', n, m)
    Z = ZeroMatrix(n, m)

    assert A + Z == A
    assert A*Z.T == ZeroMatrix(n, n)
    assert Z*A.T == ZeroMatrix(n, n)
    assert A - A == ZeroMatrix(*A.shape)

    assert Z

    assert Z.transpose() == ZeroMatrix(m, n)
    assert Z.conjugate() == Z
    assert Z.adjoint() == ZeroMatrix(m, n)
    assert re(Z) == Z
    assert im(Z) == Z

    assert ZeroMatrix(n, n)**0 == Identity(n)
    assert ZeroMatrix(3, 3).as_explicit() == ImmutableDenseMatrix.zeros(3, 3)


def test_ZeroMatrix_doit():
    n = symbols('n', integer=True)
    Znn = ZeroMatrix(Add(n, n, evaluate=False), n)
    assert isinstance(Znn.rows, Add)
    assert Znn.doit() == ZeroMatrix(2*n, n)
    assert isinstance(Znn.doit().rows, Mul)


def test_OneMatrix():
    n, m = symbols('n m', integer=True)
    A = MatrixSymbol('A', n, m)
    U = OneMatrix(n, m)

    assert U.shape == (n, m)
    assert isinstance(A + U, Add)
    assert U.transpose() == OneMatrix(m, n)
    assert U.conjugate() == U
    assert U.adjoint() == OneMatrix(m, n)
    assert re(U) == U
    assert im(U) == ZeroMatrix(n, m)

    assert OneMatrix(n, n) ** 0 == Identity(n)

    U = OneMatrix(n, n)
    assert U[1, 2] == 1

    U = OneMatrix(2, 3)
    assert U.as_explicit() == ImmutableDenseMatrix.ones(2, 3)


def test_OneMatrix_doit():
    n = symbols('n', integer=True)
    Unn = OneMatrix(Add(n, n, evaluate=False), n)
    assert isinstance(Unn.rows, Add)
    assert Unn.doit() == OneMatrix(2 * n, n)
    assert isinstance(Unn.doit().rows, Mul)


def test_OneMatrix_mul():
    n, m, k = symbols('n m k', integer=True)
    w = MatrixSymbol('w', n, 1)
    assert OneMatrix(n, m) * OneMatrix(m, k) == OneMatrix(n, k) * m
    assert w * OneMatrix(1, 1) == w
    assert OneMatrix(1, 1) * w.T == w.T


def test_Identity():
    n, m = symbols('n m', integer=True)
    A = MatrixSymbol('A', n, m)
    i, j = symbols('i j')

    In = Identity(n)
    Im = Identity(m)

    assert A*Im == A
    assert In*A == A

    assert In.transpose() == In
    assert In.inverse() == In
    assert In.conjugate() == In
    assert In.adjoint() == In
    assert re(In) == In
    assert im(In) == ZeroMatrix(n, n)

    assert In[i, j] != 0
    assert Sum(In[i, j], (i, 0, n-1), (j, 0, n-1)).subs(n,3).doit() == 3
    assert Sum(Sum(In[i, j], (i, 0, n-1)), (j, 0, n-1)).subs(n,3).doit() == 3

    # If range exceeds the limit `(0, n-1)`, do not remove `Piecewise`:
    expr = Sum(In[i, j], (i, 0, n-1))
    assert expr.doit() == 1
    expr = Sum(In[i, j], (i, 0, n-2))
    assert expr.doit().dummy_eq(
        Piecewise(
            (1, (j >= 0) & (j <= n-2)),
            (0, True)
        )
    )
    expr = Sum(In[i, j], (i, 1, n-1))
    assert expr.doit().dummy_eq(
        Piecewise(
            (1, (j >= 1) & (j <= n-1)),
            (0, True)
        )
    )
    assert Identity(3).as_explicit() == ImmutableDenseMatrix.eye(3)


def test_Identity_doit():
    n = symbols('n', integer=True)
    Inn = Identity(Add(n, n, evaluate=False))
    assert isinstance(Inn.rows, Add)
    assert Inn.doit() == Identity(2*n)
    assert isinstance(Inn.doit().rows, Mul)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_between_time.py ===
from datetime import (
    datetime,
    time,
)

import numpy as np
import pytest

from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestBetweenTime:
    @td.skip_if_not_us_locale
    def test_between_time_formats(self, frame_or_series):
        # GH#11818
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)

        strings = [
            ("2:00", "2:30"),
            ("0200", "0230"),
            ("2:00am", "2:30am"),
            ("0200am", "0230am"),
            ("2:00:00", "2:30:00"),
            ("020000", "023000"),
            ("2:00:00am", "2:30:00am"),
            ("020000am", "023000am"),
        ]
        expected_length = 28

        for time_string in strings:
            assert len(ts.between_time(*time_string)) == expected_length

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_localized_between_time(self, tzstr, frame_or_series):
        tz = timezones.maybe_get_tz(tzstr)

        rng = date_range("4/16/2012", "5/1/2012", freq="h")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        if frame_or_series is DataFrame:
            ts = ts.to_frame()

        ts_local = ts.tz_localize(tzstr)

        t1, t2 = time(10, 0), time(11, 0)
        result = ts_local.between_time(t1, t2)
        expected = ts.between_time(t1, t2).tz_localize(tzstr)
        tm.assert_equal(result, expected)
        assert timezones.tz_compare(result.index.tz, tz)

    def test_between_time_types(self, frame_or_series):
        # GH11818
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        obj = DataFrame({"A": 0}, index=rng)
        obj = tm.get_obj(obj, frame_or_series)

        msg = r"Cannot convert arg \[datetime\.datetime\(2010, 1, 2, 1, 0\)\] to a time"
        with pytest.raises(ValueError, match=msg):
            obj.between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))

    def test_between_time(self, inclusive_endpoints_fixture, frame_or_series):
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)

        stime = time(0, 0)
        etime = time(1, 0)
        inclusive = inclusive_endpoints_fixture

        filtered = ts.between_time(stime, etime, inclusive=inclusive)
        exp_len = 13 * 4 + 1

        if inclusive in ["right", "neither"]:
            exp_len -= 5
        if inclusive in ["left", "neither"]:
            exp_len -= 4

        assert len(filtered) == exp_len
        for rs in filtered.index:
            t = rs.time()
            if inclusive in ["left", "both"]:
                assert t >= stime
            else:
                assert t > stime

            if inclusive in ["right", "both"]:
                assert t <= etime
            else:
                assert t < etime

        result = ts.between_time("00:00", "01:00")
        expected = ts.between_time(stime, etime)
        tm.assert_equal(result, expected)

        # across midnight
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )
        ts = tm.get_obj(ts, frame_or_series)
        stime = time(22, 0)
        etime = time(9, 0)

        filtered = ts.between_time(stime, etime, inclusive=inclusive)
        exp_len = (12 * 11 + 1) * 4 + 1
        if inclusive in ["right", "neither"]:
            exp_len -= 4
        if inclusive in ["left", "neither"]:
            exp_len -= 4

        assert len(filtered) == exp_len
        for rs in filtered.index:
            t = rs.time()
            if inclusive in ["left", "both"]:
                assert (t >= stime) or (t <= etime)
            else:
                assert (t > stime) or (t <= etime)

            if inclusive in ["right", "both"]:
                assert (t <= etime) or (t >= stime)
            else:
                assert (t < etime) or (t >= stime)

    def test_between_time_raises(self, frame_or_series):
        # GH#20725
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        obj = tm.get_obj(obj, frame_or_series)

        msg = "Index must be DatetimeIndex"
        with pytest.raises(TypeError, match=msg):  # index is not a DatetimeIndex
            obj.between_time(start_time="00:00", end_time="12:00")

    def test_between_time_axis(self, frame_or_series):
        # GH#8839
        rng = date_range("1/1/2000", periods=100, freq="10min")
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        if frame_or_series is DataFrame:
            ts = ts.to_frame()

        stime, etime = ("08:00:00", "09:00:00")
        expected_length = 7

        assert len(ts.between_time(stime, etime)) == expected_length
        assert len(ts.between_time(stime, etime, axis=0)) == expected_length
        msg = f"No axis named {ts.ndim} for object type {type(ts).__name__}"
        with pytest.raises(ValueError, match=msg):
            ts.between_time(stime, etime, axis=ts.ndim)

    def test_between_time_axis_aliases(self, axis):
        # GH#8839
        rng = date_range("1/1/2000", periods=100, freq="10min")
        ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
        stime, etime = ("08:00:00", "09:00:00")
        exp_len = 7

        if axis in ["index", 0]:
            ts.index = rng
            assert len(ts.between_time(stime, etime)) == exp_len
            assert len(ts.between_time(stime, etime, axis=0)) == exp_len

        if axis in ["columns", 1]:
            ts.columns = rng
            selected = ts.between_time(stime, etime, axis=1).columns
            assert len(selected) == exp_len

    def test_between_time_axis_raises(self, axis):
        # issue 8839
        rng = date_range("1/1/2000", periods=100, freq="10min")
        mask = np.arange(0, len(rng))
        rand_data = np.random.default_rng(2).standard_normal((len(rng), len(rng)))
        ts = DataFrame(rand_data, index=rng, columns=rng)
        stime, etime = ("08:00:00", "09:00:00")

        msg = "Index must be DatetimeIndex"
        if axis in ["columns", 1]:
            ts.index = mask
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime)
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime, axis=0)

        if axis in ["index", 0]:
            ts.columns = mask
            with pytest.raises(TypeError, match=msg):
                ts.between_time(stime, etime, axis=1)

    def test_between_time_datetimeindex(self):
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        bkey = slice(time(13, 0, 0), time(14, 0, 0))
        binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]

        result = df.between_time(bkey.start, bkey.stop)
        expected = df.loc[bkey]
        expected2 = df.iloc[binds]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result, expected2)
        assert len(result) == 12

    def test_between_time_incorrect_arg_inclusive(self):
        # GH40245
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng
        )

        stime = time(0, 0)
        etime = time(1, 0)
        inclusive = "bad_string"
        msg = "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        with pytest.raises(ValueError, match=msg):
            ts.between_time(stime, etime, inclusive=inclusive)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_isin.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestDataFrameIsIn:
    def test_isin(self):
        # GH#4211
        df = DataFrame(
            {
                "vals": [1, 2, 3, 4],
                "ids": ["a", "b", "f", "n"],
                "ids2": ["a", "n", "c", "n"],
            },
            index=["foo", "bar", "baz", "qux"],
        )
        other = ["a", "b", "c"]

        result = df.isin(other)
        expected = DataFrame([df.loc[s].isin(other) for s in df.index])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # GH#16991
        df = DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})
        expected = DataFrame(False, df.index, df.columns)

        result = df.isin(empty)
        tm.assert_frame_equal(result, expected)

    def test_isin_dict(self):
        df = DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})
        d = {"A": ["a"]}

        expected = DataFrame(False, df.index, df.columns)
        expected.loc[0, "A"] = True

        result = df.isin(d)
        tm.assert_frame_equal(result, expected)

        # non unique columns
        df = DataFrame({"A": ["a", "b", "c"], "B": ["a", "e", "f"]})
        df.columns = ["A", "A"]
        expected = DataFrame(False, df.index, df.columns)
        expected.loc[0, "A"] = True
        result = df.isin(d)
        tm.assert_frame_equal(result, expected)

    def test_isin_with_string_scalar(self):
        # GH#4763
        df = DataFrame(
            {
                "vals": [1, 2, 3, 4],
                "ids": ["a", "b", "f", "n"],
                "ids2": ["a", "n", "c", "n"],
            },
            index=["foo", "bar", "baz", "qux"],
        )
        msg = (
            r"only list-like or dict-like objects are allowed "
            r"to be passed to DataFrame.isin\(\), you passed a 'str'"
        )
        with pytest.raises(TypeError, match=msg):
            df.isin("a")

        with pytest.raises(TypeError, match=msg):
            df.isin("aaa")

    def test_isin_df(self):
        df1 = DataFrame({"A": [1, 2, 3, 4], "B": [2, np.nan, 4, 4]})
        df2 = DataFrame({"A": [0, 2, 12, 4], "B": [2, np.nan, 4, 5]})
        expected = DataFrame(False, df1.index, df1.columns)
        result = df1.isin(df2)
        expected.loc[[1, 3], "A"] = True
        expected.loc[[0, 2], "B"] = True
        tm.assert_frame_equal(result, expected)

        # partial overlapping columns
        df2.columns = ["A", "C"]
        result = df1.isin(df2)
        expected["B"] = False
        tm.assert_frame_equal(result, expected)

    def test_isin_tuples(self):
        # GH#16394
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "f"]})
        df["C"] = list(zip(df["A"], df["B"]))
        result = df["C"].isin([(1, "a")])
        tm.assert_series_equal(result, Series([True, False, False], name="C"))

    def test_isin_df_dupe_values(self):
        df1 = DataFrame({"A": [1, 2, 3, 4], "B": [2, np.nan, 4, 4]})
        # just cols duped
        df2 = DataFrame([[0, 2], [12, 4], [2, np.nan], [4, 5]], columns=["B", "B"])
        msg = r"cannot compute isin with a duplicate axis\."
        with pytest.raises(ValueError, match=msg):
            df1.isin(df2)

        # just index duped
        df2 = DataFrame(
            [[0, 2], [12, 4], [2, np.nan], [4, 5]],
            columns=["A", "B"],
            index=[0, 0, 1, 1],
        )
        with pytest.raises(ValueError, match=msg):
            df1.isin(df2)

        # cols and index:
        df2.columns = ["B", "B"]
        with pytest.raises(ValueError, match=msg):
            df1.isin(df2)

    def test_isin_dupe_self(self):
        other = DataFrame({"A": [1, 0, 1, 0], "B": [1, 1, 0, 0]})
        df = DataFrame([[1, 1], [1, 0], [0, 0]], columns=["A", "A"])
        result = df.isin(other)
        expected = DataFrame(False, index=df.index, columns=df.columns)
        expected.loc[0] = True
        expected.iloc[1, 1] = True
        tm.assert_frame_equal(result, expected)

    def test_isin_against_series(self):
        df = DataFrame(
            {"A": [1, 2, 3, 4], "B": [2, np.nan, 4, 4]}, index=["a", "b", "c", "d"]
        )
        s = Series([1, 3, 11, 4], index=["a", "b", "c", "d"])
        expected = DataFrame(False, index=df.index, columns=df.columns)
        expected.loc["a", "A"] = True
        expected.loc["d"] = True
        result = df.isin(s)
        tm.assert_frame_equal(result, expected)

    def test_isin_multiIndex(self):
        idx = MultiIndex.from_tuples(
            [
                (0, "a", "foo"),
                (0, "a", "bar"),
                (0, "b", "bar"),
                (0, "b", "baz"),
                (2, "a", "foo"),
                (2, "a", "bar"),
                (2, "c", "bar"),
                (2, "c", "baz"),
                (1, "b", "foo"),
                (1, "b", "bar"),
                (1, "c", "bar"),
                (1, "c", "baz"),
            ]
        )
        df1 = DataFrame({"A": np.ones(12), "B": np.zeros(12)}, index=idx)
        df2 = DataFrame(
            {
                "A": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                "B": [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        # against regular index
        expected = DataFrame(False, index=df1.index, columns=df1.columns)
        result = df1.isin(df2)
        tm.assert_frame_equal(result, expected)

        df2.index = idx
        expected = df2.values.astype(bool)
        expected[:, 1] = ~expected[:, 1]
        expected = DataFrame(expected, columns=["A", "B"], index=idx)

        result = df1.isin(df2)
        tm.assert_frame_equal(result, expected)

    def test_isin_empty_datetimelike(self):
        # GH#15473
        df1_ts = DataFrame({"date": pd.to_datetime(["2014-01-01", "2014-01-02"])})
        df1_td = DataFrame({"date": [pd.Timedelta(1, "s"), pd.Timedelta(2, "s")]})
        df2 = DataFrame({"date": []})
        df3 = DataFrame()

        expected = DataFrame({"date": [False, False]})

        result = df1_ts.isin(df2)
        tm.assert_frame_equal(result, expected)
        result = df1_ts.isin(df3)
        tm.assert_frame_equal(result, expected)

        result = df1_td.isin(df2)
        tm.assert_frame_equal(result, expected)
        result = df1_td.isin(df3)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            DataFrame({"a": [1, 2, 3]}, dtype="category"),
            Series([1, 2, 3], dtype="category"),
        ],
    )
    def test_isin_category_frame(self, values):
        # GH#34256
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = DataFrame({"a": [True, True, True], "b": [False, False, False]})

        result = df.isin(values)
        tm.assert_frame_equal(result, expected)

    def test_isin_read_only(self):
        # https://github.com/pandas-dev/pandas/issues/37174
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        df = DataFrame([1, 2, 3])
        result = df.isin(arr)
        expected = DataFrame([True, True, True])
        tm.assert_frame_equal(result, expected)

    def test_isin_not_lossy(self):
        # GH 53514
        val = 1666880195890293744
        df = DataFrame({"a": [val], "b": [1.0]})
        result = df.isin([val])
        expected = DataFrame({"a": [True], "b": [False]})
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_comment.py ===
"""
Tests that comments are properly handled during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm


@pytest.mark.parametrize("na_values", [None, ["NaN"]])
def test_comment(all_parsers, na_values):
    parser = all_parsers
    data = """A,B,C
1,2.,4.#hello world
5.,NaN,10.0
"""
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", na_values=na_values)
        return
    result = parser.read_csv(StringIO(data), comment="#", na_values=na_values)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "read_kwargs", [{}, {"lineterminator": "*"}, {"delim_whitespace": True}]
)
def test_line_comment(all_parsers, read_kwargs, request):
    parser = all_parsers
    data = """# empty
A,B,C
1,2.,4.#hello world
#ignore this line
5.,NaN,10.0
"""
    warn = None
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"

    if read_kwargs.get("delim_whitespace"):
        data = data.replace(",", " ")
        warn = FutureWarning
    elif read_kwargs.get("lineterminator"):
        data = data.replace("\n", read_kwargs.get("lineterminator"))

    read_kwargs["comment"] = "#"
    if parser.engine == "pyarrow":
        if "lineterminator" in read_kwargs:
            msg = (
                "The 'lineterminator' option is not supported with the 'pyarrow' engine"
            )
        else:
            msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                warn, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(StringIO(data), **read_kwargs)
        return
    elif parser.engine == "python" and read_kwargs.get("lineterminator"):
        msg = r"Custom line terminators not supported in python parser \(yet\)"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(
                warn, match=depr_msg, check_stacklevel=False
            ):
                parser.read_csv(StringIO(data), **read_kwargs)
        return

    with tm.assert_produces_warning(warn, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), **read_kwargs)

    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows(all_parsers):
    parser = all_parsers
    data = """# empty
random line
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # This should ignore the first four lines (including comments).
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", skiprows=4)
        return

    result = parser.read_csv(StringIO(data), comment="#", skiprows=4)
    tm.assert_frame_equal(result, expected)


def test_comment_header(all_parsers):
    parser = all_parsers
    data = """# empty
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Header should begin at the second non-comment line.
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", header=1)
        return
    result = parser.read_csv(StringIO(data), comment="#", header=1)
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows_header(all_parsers):
    parser = all_parsers
    data = """# empty
# second empty line
# third empty line
X,Y,Z
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Skiprows should skip the first 4 lines (including comments),
    # while header should start from the second non-commented line,
    # starting with line 5.
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
        return

    result = parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("comment_char", ["#", "~", "&", "^", "*", "@"])
def test_custom_comment_char(all_parsers, comment_char):
    parser = all_parsers
    data = "a,b,c\n1,2,3#ignore this!\n4,5,6#ignorethistoo"

    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data.replace("#", comment_char)), comment=comment_char
            )
        return
    result = parser.read_csv(
        StringIO(data.replace("#", comment_char)), comment=comment_char
    )

    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("header", ["infer", None])
def test_comment_first_line(all_parsers, header):
    # see gh-4623
    parser = all_parsers
    data = "# notes\na,b,c\n# more notes\n1,2,3"

    if header is None:
        expected = DataFrame({0: ["a", "1"], 1: ["b", "2"], 2: ["c", "3"]})
    else:
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", header=header)
        return
    result = parser.read_csv(StringIO(data), comment="#", header=header)
    tm.assert_frame_equal(result, expected)


def test_comment_char_in_default_value(all_parsers, request):
    # GH#34002
    if all_parsers.engine == "c":
        reason = "see gh-34002: works on the python engine but not the c engine"
        # NA value containing comment char is interpreted as comment
        request.applymarker(pytest.mark.xfail(reason=reason, raises=AssertionError))
    parser = all_parsers

    data = (
        "# this is a comment\n"
        "col1,col2,col3,col4\n"
        "1,2,3,4#inline comment\n"
        "4,5#,6,10\n"
        "7,8,#N/A,11\n"
    )
    if parser.engine == "pyarrow":
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment="#", na_values="#N/A")
        return
    result = parser.read_csv(StringIO(data), comment="#", na_values="#N/A")
    expected = DataFrame(
        {
            "col1": [1, 4, 7],
            "col2": [2, 5, 8],
            "col3": [3.0, np.nan, np.nan],
            "col4": [4.0, np.nan, 11.0],
        }
    )
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_rolling_skew_kurt.py ===
from functools import partial

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    concat,
    isna,
    notna,
)
import pandas._testing as tm

from pandas.tseries import offsets


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_series(series, sp_func, roll_func):
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    result = getattr(series.rolling(50), roll_func)()
    assert isinstance(result, Series)
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[-50:]))


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_frame(raw, frame, sp_func, roll_func):
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    result = getattr(frame.rolling(50), roll_func)()
    assert isinstance(result, DataFrame)
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[-50:, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_time_rule_series(series, sp_func, roll_func):
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    win = 25
    ser = series[::2].resample("B").mean()
    series_result = getattr(ser.rolling(window=win, min_periods=10), roll_func)()
    last_date = series_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_series = series[::2].truncate(prev_date, last_date)
    tm.assert_almost_equal(series_result.iloc[-1], compare_func(trunc_series))


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_time_rule_frame(raw, frame, sp_func, roll_func):
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    win = 25
    frm = frame[::2].resample("B").mean()
    frame_result = getattr(frm.rolling(window=win, min_periods=10), roll_func)()
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_nans(sp_func, roll_func):
    sp_stats = pytest.importorskip("scipy.stats")

    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = getattr(obj.rolling(50, min_periods=30), roll_func)()
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))

    # min_periods is working correctly
    result = getattr(obj.rolling(20, min_periods=15), roll_func)()
    assert isna(result.iloc[23])
    assert not isna(result.iloc[24])

    assert not isna(result.iloc[-6])
    assert isna(result.iloc[-5])

    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    result = getattr(obj2.rolling(10, min_periods=5), roll_func)()
    assert isna(result.iloc[3])
    assert notna(result.iloc[4])

    result0 = getattr(obj.rolling(20, min_periods=0), roll_func)()
    result1 = getattr(obj.rolling(20, min_periods=1), roll_func)()
    tm.assert_almost_equal(result0, result1)


@pytest.mark.parametrize("minp", [0, 99, 100])
@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_min_periods(series, minp, roll_func, step):
    result = getattr(
        series.rolling(len(series) + 1, min_periods=minp, step=step), roll_func
    )()
    expected = getattr(
        series.rolling(len(series), min_periods=minp, step=step), roll_func
    )()
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_center(roll_func):
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    result = getattr(obj.rolling(20, center=True), roll_func)()
    expected = (
        getattr(concat([obj, Series([np.nan] * 9)]).rolling(20), roll_func)()
        .iloc[9:]
        .reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_center_reindex_series(series, roll_func):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    series_xp = (
        getattr(
            series.reindex(list(series.index) + s).rolling(window=25),
            roll_func,
        )()
        .shift(-12)
        .reindex(series.index)
    )
    series_rs = getattr(series.rolling(window=25, center=True), roll_func)()
    tm.assert_series_equal(series_xp, series_rs)


@pytest.mark.slow
@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
def test_center_reindex_frame(frame, roll_func):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    frame_xp = (
        getattr(
            frame.reindex(list(frame.index) + s).rolling(window=25),
            roll_func,
        )()
        .shift(-12)
        .reindex(frame.index)
    )
    frame_rs = getattr(frame.rolling(window=25, center=True), roll_func)()
    tm.assert_frame_equal(frame_xp, frame_rs)


def test_rolling_skew_edge_cases(step):
    expected = Series([np.nan] * 4 + [0.0])[::step]
    # yields all NaN (0 variance)
    d = Series([1] * 5)
    x = d.rolling(window=5, step=step).skew()
    # index 4 should be 0 as it contains 5 same obs
    tm.assert_series_equal(expected, x)

    expected = Series([np.nan] * 5)[::step]
    # yields all NaN (window too small)
    d = Series(np.random.default_rng(2).standard_normal(5))
    x = d.rolling(window=2, step=step).skew()
    tm.assert_series_equal(expected, x)

    # yields [NaN, NaN, NaN, 0.177994, 1.548824]
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    expected = Series([np.nan, np.nan, np.nan, 0.177994, 1.548824])[::step]
    x = d.rolling(window=4, step=step).skew()
    tm.assert_series_equal(expected, x)


def test_rolling_kurt_edge_cases(step):
    expected = Series([np.nan] * 4 + [-3.0])[::step]

    # yields all NaN (0 variance)
    d = Series([1] * 5)
    x = d.rolling(window=5, step=step).kurt()
    tm.assert_series_equal(expected, x)

    # yields all NaN (window too small)
    expected = Series([np.nan] * 5)[::step]
    d = Series(np.random.default_rng(2).standard_normal(5))
    x = d.rolling(window=3, step=step).kurt()
    tm.assert_series_equal(expected, x)

    # yields [NaN, NaN, NaN, 1.224307, 2.671499]
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    expected = Series([np.nan, np.nan, np.nan, 1.224307, 2.671499])[::step]
    x = d.rolling(window=4, step=step).kurt()
    tm.assert_series_equal(expected, x)


def test_rolling_skew_eq_value_fperr(step):
    # #18804 all rolling skew for all equal values should return Nan
    # #46717 update: all equal values should return 0 instead of NaN
    a = Series([1.1] * 15).rolling(window=10, step=step).skew()
    assert (a[a.index >= 9] == 0).all()
    assert a[a.index < 9].isna().all()


def test_rolling_kurt_eq_value_fperr(step):
    # #18804 all rolling kurt for all equal values should return Nan
    # #46717 update: all equal values should return -3 instead of NaN
    a = Series([1.1] * 15).rolling(window=10, step=step).kurt()
    assert (a[a.index >= 9] == -3).all()
    assert a[a.index < 9].isna().all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_refine.py ===
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.expr import Expr
from sympy.core.numbers import (I, Rational, nan, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, atan2)
from sympy.abc import w, x, y, z
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.matexpr import MatrixSymbol


def test_Abs():
    assert refine(Abs(x), Q.positive(x)) == x
    assert refine(1 + Abs(x), Q.positive(x)) == 1 + x
    assert refine(Abs(x), Q.negative(x)) == -x
    assert refine(1 + Abs(x), Q.negative(x)) == 1 - x

    assert refine(Abs(x**2)) != x**2
    assert refine(Abs(x**2), Q.real(x)) == x**2


def test_pow1():
    assert refine((-1)**x, Q.even(x)) == 1
    assert refine((-1)**x, Q.odd(x)) == -1
    assert refine((-2)**x, Q.even(x)) == 2**x

    # nested powers
    assert refine(sqrt(x**2)) != Abs(x)
    assert refine(sqrt(x**2), Q.complex(x)) != Abs(x)
    assert refine(sqrt(x**2), Q.real(x)) == Abs(x)
    assert refine(sqrt(x**2), Q.positive(x)) == x
    assert refine((x**3)**Rational(1, 3)) != x

    assert refine((x**3)**Rational(1, 3), Q.real(x)) != x
    assert refine((x**3)**Rational(1, 3), Q.positive(x)) == x

    assert refine(sqrt(1/x), Q.real(x)) != 1/sqrt(x)
    assert refine(sqrt(1/x), Q.positive(x)) == 1/sqrt(x)

    # powers of (-1)
    assert refine((-1)**(x + y), Q.even(x)) == (-1)**y
    assert refine((-1)**(x + y + z), Q.odd(x) & Q.odd(z)) == (-1)**y
    assert refine((-1)**(x + y + 1), Q.odd(x)) == (-1)**y
    assert refine((-1)**(x + y + 2), Q.odd(x)) == (-1)**(y + 1)
    assert refine((-1)**(x + 3)) == (-1)**(x + 1)

    # continuation
    assert refine((-1)**((-1)**x/2 - S.Half), Q.integer(x)) == (-1)**x
    assert refine((-1)**((-1)**x/2 + S.Half), Q.integer(x)) == (-1)**(x + 1)
    assert refine((-1)**((-1)**x/2 + 5*S.Half), Q.integer(x)) == (-1)**(x + 1)


def test_pow2():
    assert refine((-1)**((-1)**x/2 - 7*S.Half), Q.integer(x)) == (-1)**(x + 1)
    assert refine((-1)**((-1)**x/2 - 9*S.Half), Q.integer(x)) == (-1)**x

    # powers of Abs
    assert refine(Abs(x)**2, Q.real(x)) == x**2
    assert refine(Abs(x)**3, Q.real(x)) == Abs(x)**3
    assert refine(Abs(x)**2) == Abs(x)**2


def test_exp():
    x = Symbol('x', integer=True)
    assert refine(exp(pi*I*2*x)) == 1
    assert refine(exp(pi*I*2*(x + S.Half))) == -1
    assert refine(exp(pi*I*2*(x + Rational(1, 4)))) == I
    assert refine(exp(pi*I*2*(x + Rational(3, 4)))) == -I


def test_Piecewise():
    assert refine(Piecewise((1, x < 0), (3, True)), (x < 0)) == 1
    assert refine(Piecewise((1, x < 0), (3, True)), ~(x < 0)) == 3
    assert refine(Piecewise((1, x < 0), (3, True)), (y < 0)) == \
        Piecewise((1, x < 0), (3, True))
    assert refine(Piecewise((1, x > 0), (3, True)), (x > 0)) == 1
    assert refine(Piecewise((1, x > 0), (3, True)), ~(x > 0)) == 3
    assert refine(Piecewise((1, x > 0), (3, True)), (y > 0)) == \
        Piecewise((1, x > 0), (3, True))
    assert refine(Piecewise((1, x <= 0), (3, True)), (x <= 0)) == 1
    assert refine(Piecewise((1, x <= 0), (3, True)), ~(x <= 0)) == 3
    assert refine(Piecewise((1, x <= 0), (3, True)), (y <= 0)) == \
        Piecewise((1, x <= 0), (3, True))
    assert refine(Piecewise((1, x >= 0), (3, True)), (x >= 0)) == 1
    assert refine(Piecewise((1, x >= 0), (3, True)), ~(x >= 0)) == 3
    assert refine(Piecewise((1, x >= 0), (3, True)), (y >= 0)) == \
        Piecewise((1, x >= 0), (3, True))
    assert refine(Piecewise((1, Eq(x, 0)), (3, True)), (Eq(x, 0)))\
        == 1
    assert refine(Piecewise((1, Eq(x, 0)), (3, True)), (Eq(0, x)))\
        == 1
    assert refine(Piecewise((1, Eq(x, 0)), (3, True)), ~(Eq(x, 0)))\
        == 3
    assert refine(Piecewise((1, Eq(x, 0)), (3, True)), ~(Eq(0, x)))\
        == 3
    assert refine(Piecewise((1, Eq(x, 0)), (3, True)), (Eq(y, 0)))\
        == Piecewise((1, Eq(x, 0)), (3, True))
    assert refine(Piecewise((1, Ne(x, 0)), (3, True)), (Ne(x, 0)))\
        == 1
    assert refine(Piecewise((1, Ne(x, 0)), (3, True)), ~(Ne(x, 0)))\
        == 3
    assert refine(Piecewise((1, Ne(x, 0)), (3, True)), (Ne(y, 0)))\
        == Piecewise((1, Ne(x, 0)), (3, True))


def test_atan2():
    assert refine(atan2(y, x), Q.real(y) & Q.positive(x)) == atan(y/x)
    assert refine(atan2(y, x), Q.negative(y) & Q.positive(x)) == atan(y/x)
    assert refine(atan2(y, x), Q.negative(y) & Q.negative(x)) == atan(y/x) - pi
    assert refine(atan2(y, x), Q.positive(y) & Q.negative(x)) == atan(y/x) + pi
    assert refine(atan2(y, x), Q.zero(y) & Q.negative(x)) == pi
    assert refine(atan2(y, x), Q.positive(y) & Q.zero(x)) == pi/2
    assert refine(atan2(y, x), Q.negative(y) & Q.zero(x)) == -pi/2
    assert refine(atan2(y, x), Q.zero(y) & Q.zero(x)) is nan


def test_re():
    assert refine(re(x), Q.real(x)) == x
    assert refine(re(x), Q.imaginary(x)) is S.Zero
    assert refine(re(x+y), Q.real(x) & Q.real(y)) == x + y
    assert refine(re(x+y), Q.real(x) & Q.imaginary(y)) == x
    assert refine(re(x*y), Q.real(x) & Q.real(y)) == x * y
    assert refine(re(x*y), Q.real(x) & Q.imaginary(y)) == 0
    assert refine(re(x*y*z), Q.real(x) & Q.real(y) & Q.real(z)) == x * y * z


def test_im():
    assert refine(im(x), Q.imaginary(x)) == -I*x
    assert refine(im(x), Q.real(x)) is S.Zero
    assert refine(im(x+y), Q.imaginary(x) & Q.imaginary(y)) == -I*x - I*y
    assert refine(im(x+y), Q.real(x) & Q.imaginary(y)) == -I*y
    assert refine(im(x*y), Q.imaginary(x) & Q.real(y)) == -I*x*y
    assert refine(im(x*y), Q.imaginary(x) & Q.imaginary(y)) == 0
    assert refine(im(1/x), Q.imaginary(x)) == -I/x
    assert refine(im(x*y*z), Q.imaginary(x) & Q.imaginary(y)
        & Q.imaginary(z)) == -I*x*y*z


def test_complex():
    assert refine(re(1/(x + I*y)), Q.real(x) & Q.real(y)) == \
        x/(x**2 + y**2)
    assert refine(im(1/(x + I*y)), Q.real(x) & Q.real(y)) == \
        -y/(x**2 + y**2)
    assert refine(re((w + I*x) * (y + I*z)), Q.real(w) & Q.real(x) & Q.real(y)
        & Q.real(z)) == w*y - x*z
    assert refine(im((w + I*x) * (y + I*z)), Q.real(w) & Q.real(x) & Q.real(y)
        & Q.real(z)) == w*z + x*y


def test_sign():
    x = Symbol('x', real = True)
    assert refine(sign(x), Q.positive(x)) == 1
    assert refine(sign(x), Q.negative(x)) == -1
    assert refine(sign(x), Q.zero(x)) == 0
    assert refine(sign(x), True) == sign(x)
    assert refine(sign(Abs(x)), Q.nonzero(x)) == 1

    x = Symbol('x', imaginary=True)
    assert refine(sign(x), Q.positive(im(x))) == S.ImaginaryUnit
    assert refine(sign(x), Q.negative(im(x))) == -S.ImaginaryUnit
    assert refine(sign(x), True) == sign(x)

    x = Symbol('x', complex=True)
    assert refine(sign(x), Q.zero(x)) == 0

def test_arg():
    x = Symbol('x', complex = True)
    assert refine(arg(x), Q.positive(x)) == 0
    assert refine(arg(x), Q.negative(x)) == pi

def test_func_args():
    class MyClass(Expr):
        # A class with nontrivial .func

        def __init__(self, *args):
            self.my_member = ""

        @property
        def func(self):
            def my_func(*args):
                obj = MyClass(*args)
                obj.my_member = self.my_member
                return obj
            return my_func

    x = MyClass()
    x.my_member = "A very important value"
    assert x.my_member == refine(x).my_member

def test_issue_refine_9384():
    assert refine(Piecewise((1, x < 0), (0, True)), Q.positive(x)) == 0
    assert refine(Piecewise((1, x < 0), (0, True)), Q.negative(x)) == 1
    assert refine(Piecewise((1, x > 0), (0, True)), Q.positive(x)) == 1
    assert refine(Piecewise((1, x > 0), (0, True)), Q.negative(x)) == 0


def test_eval_refine():
    class MockExpr(Expr):
        def _eval_refine(self, assumptions):
            return True

    mock_obj = MockExpr()
    assert refine(mock_obj)

def test_refine_issue_12724():
    expr1 = refine(Abs(x * y), Q.positive(x))
    expr2 = refine(Abs(x * y * z), Q.positive(x))
    assert expr1 == x * Abs(y)
    assert expr2 == x * Abs(y * z)
    y1 = Symbol('y1', real = True)
    expr3 = refine(Abs(x * y1**2 * z), Q.positive(x))
    assert expr3 == x * y1**2 * Abs(z)


def test_matrixelement():
    x = MatrixSymbol('x', 3, 3)
    i = Symbol('i', positive = True)
    j = Symbol('j', positive = True)
    assert refine(x[0, 1], Q.symmetric(x)) == x[0, 1]
    assert refine(x[1, 0], Q.symmetric(x)) == x[0, 1]
    assert refine(x[i, j], Q.symmetric(x)) == x[j, i]
    assert refine(x[j, i], Q.symmetric(x)) == x[j, i]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\test_index_methods.py ===
from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException

from sympy.tensor.index_methods import (get_contraction_structure, get_indices)


def test_trivial_indices():
    x, y = symbols('x y')
    assert get_indices(x) == (set(), {})
    assert get_indices(x*y) == (set(), {})
    assert get_indices(x + y) == (set(), {})
    assert get_indices(x**y) == (set(), {})


def test_get_indices_Indexed():
    x = IndexedBase('x')
    i, j = Idx('i'), Idx('j')
    assert get_indices(x[i, j]) == ({i, j}, {})
    assert get_indices(x[j, i]) == ({j, i}, {})


def test_get_indices_Idx():
    f = Function('f')
    i, j = Idx('i'), Idx('j')
    assert get_indices(f(i)*j) == ({i, j}, {})
    assert get_indices(f(j, i)) == ({j, i}, {})
    assert get_indices(f(i)*i) == (set(), {})


def test_get_indices_mul():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j = Idx('i'), Idx('j')
    assert get_indices(x[j]*y[i]) == ({i, j}, {})
    assert get_indices(x[i]*y[j]) == ({i, j}, {})


def test_get_indices_exceptions():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j = Idx('i'), Idx('j')
    raises(IndexConformanceException, lambda: get_indices(x[i] + y[j]))


def test_scalar_broadcast():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j = Idx('i'), Idx('j')
    assert get_indices(x[i] + y[i, i]) == ({i}, {})
    assert get_indices(x[i] + y[j, j]) == ({i}, {})


def test_get_indices_add():
    x = IndexedBase('x')
    y = IndexedBase('y')
    A = IndexedBase('A')
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    assert get_indices(x[i] + 2*y[i]) == ({i}, {})
    assert get_indices(y[i] + 2*A[i, j]*x[j]) == ({i}, {})
    assert get_indices(y[i] + 2*(x[i] + A[i, j]*x[j])) == ({i}, {})
    assert get_indices(y[i] + x[i]*(A[j, j] + 1)) == ({i}, {})
    assert get_indices(
        y[i] + x[i]*x[j]*(y[j] + A[j, k]*x[k])) == ({i}, {})


def test_get_indices_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    A = IndexedBase('A')
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    assert get_indices(Pow(x[i], y[j])) == ({i, j}, {})
    assert get_indices(Pow(x[i, k], y[j, k])) == ({i, j, k}, {})
    assert get_indices(Pow(A[i, k], y[k] + A[k, j]*x[j])) == ({i, k}, {})
    assert get_indices(Pow(2, x[i])) == get_indices(exp(x[i]))

    # test of a design decision, this may change:
    assert get_indices(Pow(x[i], 2)) == ({i}, {})


def test_get_contraction_structure_basic():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j = Idx('i'), Idx('j')
    assert get_contraction_structure(x[i]*y[j]) == {None: {x[i]*y[j]}}
    assert get_contraction_structure(x[i] + y[j]) == {None: {x[i], y[j]}}
    assert get_contraction_structure(x[i]*y[i]) == {(i,): {x[i]*y[i]}}
    assert get_contraction_structure(
        1 + x[i]*y[i]) == {None: {S.One}, (i,): {x[i]*y[i]}}
    assert get_contraction_structure(x[i]**y[i]) == {None: {x[i]**y[i]}}


def test_get_contraction_structure_complex():
    x = IndexedBase('x')
    y = IndexedBase('y')
    A = IndexedBase('A')
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    expr1 = y[i] + A[i, j]*x[j]
    d1 = {None: {y[i]}, (j,): {A[i, j]*x[j]}}
    assert get_contraction_structure(expr1) == d1
    expr2 = expr1*A[k, i] + x[k]
    d2 = {None: {x[k]}, (i,): {expr1*A[k, i]}, expr1*A[k, i]: [d1]}
    assert get_contraction_structure(expr2) == d2


def test_contraction_structure_simple_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    ii_jj = x[i, i]**y[j, j]
    assert get_contraction_structure(ii_jj) == {
        None: {ii_jj},
        ii_jj: [
            {(i,): {x[i, i]}},
            {(j,): {y[j, j]}}
        ]
    }

    ii_jk = x[i, i]**y[j, k]
    assert get_contraction_structure(ii_jk) == {
        None: {x[i, i]**y[j, k]},
        x[i, i]**y[j, k]: [
            {(i,): {x[i, i]}}
        ]
    }


def test_contraction_structure_Mul_and_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j, k = Idx('i'), Idx('j'), Idx('k')

    i_ji = x[i]**(y[j]*x[i])
    assert get_contraction_structure(i_ji) == {None: {i_ji}}
    ij_i = (x[i]*y[j])**(y[i])
    assert get_contraction_structure(ij_i) == {None: {ij_i}}
    j_ij_i = x[j]*(x[i]*y[j])**(y[i])
    assert get_contraction_structure(j_ij_i) == {(j,): {j_ij_i}}
    j_i_ji = x[j]*x[i]**(y[j]*x[i])
    assert get_contraction_structure(j_i_ji) == {(j,): {j_i_ji}}
    ij_exp_kki = x[i]*y[j]*exp(y[i]*y[k, k])
    result = get_contraction_structure(ij_exp_kki)
    expected = {
        (i,): {ij_exp_kki},
        ij_exp_kki: [{
                     None: {exp(y[i]*y[k, k])},
                exp(y[i]*y[k, k]): [{
                    None: {y[i]*y[k, k]},
                    y[i]*y[k, k]: [{(k,): {y[k, k]}}]
                }]}
        ]
    }
    assert result == expected


def test_contraction_structure_Add_in_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    s_ii_jj_s = (1 + x[i, i])**(1 + y[j, j])
    expected = {
        None: {s_ii_jj_s},
        s_ii_jj_s: [
            {None: {S.One}, (i,): {x[i, i]}},
            {None: {S.One}, (j,): {y[j, j]}}
        ]
    }
    result = get_contraction_structure(s_ii_jj_s)
    assert result == expected

    s_ii_jk_s = (1 + x[i, i]) ** (1 + y[j, k])
    expected_2 = {
        None: {(x[i, i] + 1)**(y[j, k] + 1)},
        s_ii_jk_s: [
            {None: {S.One}, (i,): {x[i, i]}}
        ]
    }
    result_2 = get_contraction_structure(s_ii_jk_s)
    assert result_2 == expected_2


def test_contraction_structure_Pow_in_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i, j, k = Idx('i'), Idx('j'), Idx('k')
    ii_jj_kk = x[i, i]**y[j, j]**z[k, k]
    expected = {
        None: {ii_jj_kk},
        ii_jj_kk: [
            {(i,): {x[i, i]}},
            {
                None: {y[j, j]**z[k, k]},
                y[j, j]**z[k, k]: [
                    {(j,): {y[j, j]}},
                    {(k,): {z[k, k]}}
                ]
            }
        ]
    }
    assert get_contraction_structure(ii_jj_kk) == expected


def test_ufunc_support():
    f = Function('f')
    g = Function('g')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j = Idx('i'), Idx('j')
    a = symbols('a')

    assert get_indices(f(x[i])) == ({i}, {})
    assert get_indices(f(x[i], y[j])) == ({i, j}, {})
    assert get_indices(f(y[i])*g(x[i])) == (set(), {})
    assert get_indices(f(a, x[i])) == ({i}, {})
    assert get_indices(f(a, y[i], x[j])*g(x[i])) == ({j}, {})
    assert get_indices(g(f(x[i]))) == ({i}, {})

    assert get_contraction_structure(f(x[i])) == {None: {f(x[i])}}
    assert get_contraction_structure(
        f(y[i])*g(x[i])) == {(i,): {f(y[i])*g(x[i])}}
    assert get_contraction_structure(
        f(y[i])*g(f(x[i]))) == {(i,): {f(y[i])*g(f(x[i]))}}
    assert get_contraction_structure(
        f(x[j], y[i])*g(x[i])) == {(i,): {f(x[j], y[i])*g(x[i])}}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\constructors\test_from_dict.py ===
from collections import OrderedDict

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
)
import pandas._testing as tm


class TestFromDict:
    # Note: these tests are specific to the from_dict method, not for
    #  passing dictionaries to DataFrame.__init__

    def test_constructor_list_of_odicts(self):
        data = [
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4], ["d", 6]]),
            OrderedDict([["a", 1.5], ["b", 3], ["d", 6]]),
            OrderedDict([["a", 1.5], ["d", 6]]),
            OrderedDict(),
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4]]),
            OrderedDict([["b", 3], ["c", 4], ["d", 6]]),
        ]

        result = DataFrame(data)
        expected = DataFrame.from_dict(
            dict(zip(range(len(data)), data)), orient="index"
        )
        tm.assert_frame_equal(result, expected.reindex(result.index))

    def test_constructor_single_row(self):
        data = [OrderedDict([["a", 1.5], ["b", 3], ["c", 4], ["d", 6]])]

        result = DataFrame(data)
        expected = DataFrame.from_dict(dict(zip([0], data)), orient="index").reindex(
            result.index
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="columns inferring logic broken")
    def test_constructor_list_of_series(self):
        data = [
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 4.0]]),
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 6.0]]),
        ]
        sdict = OrderedDict(zip(["x", "y"], data))
        idx = Index(["a", "b", "c"])

        # all named
        data2 = [
            Series([1.5, 3, 4], idx, dtype="O", name="x"),
            Series([1.5, 3, 6], idx, name="y"),
        ]
        result = DataFrame(data2)
        expected = DataFrame.from_dict(sdict, orient="index")
        tm.assert_frame_equal(result, expected)

        # some unnamed
        data2 = [
            Series([1.5, 3, 4], idx, dtype="O", name="x"),
            Series([1.5, 3, 6], idx),
        ]
        result = DataFrame(data2)

        sdict = OrderedDict(zip(["x", "Unnamed 0"], data))
        expected = DataFrame.from_dict(sdict, orient="index")
        tm.assert_frame_equal(result, expected)

        # none named
        data = [
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4], ["d", 6]]),
            OrderedDict([["a", 1.5], ["b", 3], ["d", 6]]),
            OrderedDict([["a", 1.5], ["d", 6]]),
            OrderedDict(),
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4]]),
            OrderedDict([["b", 3], ["c", 4], ["d", 6]]),
        ]
        data = [Series(d) for d in data]

        result = DataFrame(data)
        sdict = OrderedDict(zip(range(len(data)), data))
        expected = DataFrame.from_dict(sdict, orient="index")
        tm.assert_frame_equal(result, expected.reindex(result.index))

        result2 = DataFrame(data, index=np.arange(6, dtype=np.int64))
        tm.assert_frame_equal(result, result2)

        result = DataFrame([Series(dtype=object)])
        expected = DataFrame(index=[0])
        tm.assert_frame_equal(result, expected)

        data = [
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 4.0]]),
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 6.0]]),
        ]
        sdict = OrderedDict(zip(range(len(data)), data))

        idx = Index(["a", "b", "c"])
        data2 = [Series([1.5, 3, 4], idx, dtype="O"), Series([1.5, 3, 6], idx)]
        result = DataFrame(data2)
        expected = DataFrame.from_dict(sdict, orient="index")
        tm.assert_frame_equal(result, expected)

    def test_constructor_orient(self, float_string_frame):
        data_dict = float_string_frame.T._series
        recons = DataFrame.from_dict(data_dict, orient="index")
        expected = float_string_frame.reindex(index=recons.index)
        tm.assert_frame_equal(recons, expected)

        # dict of sequence
        a = {"hi": [32, 3, 3], "there": [3, 5, 3]}
        rs = DataFrame.from_dict(a, orient="index")
        xp = DataFrame.from_dict(a).T.reindex(list(a.keys()))
        tm.assert_frame_equal(rs, xp)

    def test_constructor_from_ordered_dict(self):
        # GH#8425
        a = OrderedDict(
            [
                ("one", OrderedDict([("col_a", "foo1"), ("col_b", "bar1")])),
                ("two", OrderedDict([("col_a", "foo2"), ("col_b", "bar2")])),
                ("three", OrderedDict([("col_a", "foo3"), ("col_b", "bar3")])),
            ]
        )
        expected = DataFrame.from_dict(a, orient="columns").T
        result = DataFrame.from_dict(a, orient="index")
        tm.assert_frame_equal(result, expected)

    def test_from_dict_columns_parameter(self):
        # GH#18529
        # Test new columns parameter for from_dict that was added to make
        # from_items(..., orient='index', columns=[...]) easier to replicate
        result = DataFrame.from_dict(
            OrderedDict([("A", [1, 2]), ("B", [4, 5])]),
            orient="index",
            columns=["one", "two"],
        )
        expected = DataFrame([[1, 2], [4, 5]], index=["A", "B"], columns=["one", "two"])
        tm.assert_frame_equal(result, expected)

        msg = "cannot use columns parameter with orient='columns'"
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_dict(
                {"A": [1, 2], "B": [4, 5]},
                orient="columns",
                columns=["one", "two"],
            )
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_dict({"A": [1, 2], "B": [4, 5]}, columns=["one", "two"])

    @pytest.mark.parametrize(
        "data_dict, orient, expected",
        [
            ({}, "index", RangeIndex(0)),
            (
                [{("a",): 1}, {("a",): 2}],
                "columns",
                Index([("a",)], tupleize_cols=False),
            ),
            (
                [OrderedDict([(("a",), 1), (("b",), 2)])],
                "columns",
                Index([("a",), ("b",)], tupleize_cols=False),
            ),
            ([{("a", "b"): 1}], "columns", Index([("a", "b")], tupleize_cols=False)),
        ],
    )
    def test_constructor_from_dict_tuples(self, data_dict, orient, expected):
        # GH#16769
        df = DataFrame.from_dict(data_dict, orient)
        result = df.columns
        tm.assert_index_equal(result, expected)

    def test_frame_dict_constructor_empty_series(self):
        s1 = Series(
            [1, 2, 3, 4], index=MultiIndex.from_tuples([(1, 2), (1, 3), (2, 2), (2, 4)])
        )
        s2 = Series(
            [1, 2, 3, 4], index=MultiIndex.from_tuples([(1, 2), (1, 3), (3, 2), (3, 4)])
        )
        s3 = Series(dtype=object)

        # it works!
        DataFrame({"foo": s1, "bar": s2, "baz": s3})
        DataFrame.from_dict({"foo": s1, "baz": s3, "bar": s2})

    def test_from_dict_scalars_requires_index(self):
        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_dict(OrderedDict([("b", 8), ("a", 5), ("a", 6)]))

    def test_from_dict_orient_invalid(self):
        msg = (
            "Expected 'index', 'columns' or 'tight' for orient parameter. "
            "Got 'abc' instead"
        )
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_dict({"foo": 1, "baz": 3, "bar": 2}, orient="abc")

    def test_from_dict_order_with_single_column(self):
        data = {
            "alpha": {
                "value2": 123,
                "value1": 532,
                "animal": 222,
                "plant": False,
                "name": "test",
            }
        }
        result = DataFrame.from_dict(
            data,
            orient="columns",
        )
        expected = DataFrame(
            [[123], [532], [222], [False], ["test"]],
            index=["value2", "value1", "animal", "plant", "name"],
            columns=["alpha"],
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_unsupported.py ===
"""
Tests that features that are currently unsupported in
either the Python or C parser are actually enforced
and are clearly communicated to the user.

Ultimately, the goal is to remove test cases from this
test suite as new feature support is added to the parsers.
"""
from io import StringIO
import os
from pathlib import Path

import pytest

from pandas.errors import ParserError

import pandas._testing as tm

from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture(params=["python", "python-fwf"], ids=lambda val: val)
def python_engine(request):
    return request.param


class TestUnsupportedFeatures:
    def test_mangle_dupe_cols_false(self):
        # see gh-12935
        data = "a b c\n1 2 3"

        for engine in ("c", "python"):
            with pytest.raises(TypeError, match="unexpected keyword"):
                read_csv(StringIO(data), engine=engine, mangle_dupe_cols=True)

    def test_c_engine(self):
        # see gh-6607
        data = "a b c\n1 2 3"
        msg = "does not support"

        depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"

        # specify C engine with unsupported options (raise)
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                read_csv(StringIO(data), engine="c", sep=None, delim_whitespace=False)
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep=r"\s")
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep="\t", quotechar=chr(128))
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", skipfooter=1)

        # specify C-unsupported options without python-unsupported options
        with tm.assert_produces_warning((parsers.ParserWarning, FutureWarning)):
            read_csv(StringIO(data), sep=None, delim_whitespace=False)
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep=r"\s")
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep="\t", quotechar=chr(128))
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), skipfooter=1)

        text = """                      A       B       C       D        E
one two three   four
a   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640
a   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744
x   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838"""
        msg = "Error tokenizing data"

        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), sep="\\s+")
        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), engine="c", sep="\\s+")

        msg = "Only length-1 thousands markers supported"
        data = """A|B|C
1|2,334|5
10|13|10.
"""
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands=",,")
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands="")

        msg = "Only length-1 line terminators supported"
        data = "a,b,c~~1,2,3~~4,5,6"
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), lineterminator="~~")

    def test_python_engine(self, python_engine):
        from pandas.io.parsers.readers import _python_unsupported as py_unsupported

        data = """1,2,3,,
1,2,3,4,
1,2,3,4,5
1,2,,,
1,2,3,4,"""

        for default in py_unsupported:
            msg = (
                f"The {repr(default)} option is not "
                f"supported with the {repr(python_engine)} engine"
            )

            kwargs = {default: object()}
            with pytest.raises(ValueError, match=msg):
                read_csv(StringIO(data), engine=python_engine, **kwargs)

    def test_python_engine_file_no_iter(self, python_engine):
        # see gh-16530
        class NoNextBuffer:
            def __init__(self, csv_data) -> None:
                self.data = csv_data

            def __next__(self):
                return self.data.__next__()

            def read(self):
                return self.data

            def readline(self):
                return self.data

        data = "a\n1"
        msg = "'NoNextBuffer' object is not iterable|argument 1 must be an iterator"

        with pytest.raises(TypeError, match=msg):
            read_csv(NoNextBuffer(data), engine=python_engine)

    def test_pyarrow_engine(self):
        from pandas.io.parsers.readers import _pyarrow_unsupported as pa_unsupported

        data = """1,2,3,,
        1,2,3,4,
        1,2,3,4,5
        1,2,,,
        1,2,3,4,"""

        for default in pa_unsupported:
            msg = (
                f"The {repr(default)} option is not "
                f"supported with the 'pyarrow' engine"
            )
            kwargs = {default: object()}
            default_needs_bool = {"warn_bad_lines", "error_bad_lines"}
            if default == "dialect":
                kwargs[default] = "excel"  # test a random dialect
            elif default in default_needs_bool:
                kwargs[default] = True
            elif default == "on_bad_lines":
                kwargs[default] = "warn"

            warn = None
            depr_msg = None
            if "delim_whitespace" in kwargs:
                depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
                warn = FutureWarning
            if "verbose" in kwargs:
                depr_msg = "The 'verbose' keyword in pd.read_csv is deprecated"
                warn = FutureWarning

            with pytest.raises(ValueError, match=msg):
                with tm.assert_produces_warning(warn, match=depr_msg):
                    read_csv(StringIO(data), engine="pyarrow", **kwargs)

    def test_on_bad_lines_callable_python_or_pyarrow(self, all_parsers):
        # GH 5686
        # GH 54643
        sio = StringIO("a,b\n1,2")
        bad_lines_func = lambda x: x
        parser = all_parsers
        if all_parsers.engine not in ["python", "pyarrow"]:
            msg = (
                "on_bad_line can only be a callable "
                "function if engine='python' or 'pyarrow'"
            )
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(sio, on_bad_lines=bad_lines_func)
        else:
            parser.read_csv(sio, on_bad_lines=bad_lines_func)


def test_close_file_handle_on_invalid_usecols(all_parsers):
    # GH 45384
    parser = all_parsers

    error = ValueError
    if parser.engine == "pyarrow":
        # Raises pyarrow.lib.ArrowKeyError
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    with tm.ensure_clean("test.csv") as fname:
        Path(fname).write_text("col1,col2\na,b\n1,2", encoding="utf-8")
        with tm.assert_produces_warning(False):
            with pytest.raises(error, match="col3"):
                parser.read_csv(fname, usecols=["col1", "col2", "col3"])
        # unlink fails on windows if file handles still point to it
        os.unlink(fname)


def test_invalid_file_inputs(request, all_parsers):
    # GH#45957
    parser = all_parsers
    if parser.engine == "python":
        request.applymarker(
            pytest.mark.xfail(reason=f"{parser.engine} engine supports lists.")
        )

    with pytest.raises(ValueError, match="Invalid"):
        parser.read_csv([])


def test_invalid_dtype_backend(all_parsers):
    parser = all_parsers
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        parser.read_csv("test", dtype_backend="numpy")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_free_groups.py ===
from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo

F, x, y, z = free_group("x, y, z")


def test_FreeGroup__init__():
    x, y, z = map(Symbol, "xyz")

    assert len(FreeGroup("x, y, z").generators) == 3
    assert len(FreeGroup(x).generators) == 1
    assert len(FreeGroup(("x", "y", "z"))) == 3
    assert len(FreeGroup((x, y, z)).generators) == 3


def test_FreeGroup__getnewargs__():
    x, y, z = map(Symbol, "xyz")
    assert FreeGroup("x, y, z").__getnewargs__() == ((x, y, z),)


def test_free_group():
    G, a, b, c = free_group("a, b, c")
    assert F.generators == (x, y, z)
    assert x*z**2 in F
    assert x in F
    assert y*z**-1 in F
    assert (y*z)**0 in F
    assert a not in F
    assert a**0 not in F
    assert len(F) == 3
    assert str(F) == '<free group on the generators (x, y, z)>'
    assert not F == G
    assert F.order() is oo
    assert F.is_abelian == False
    assert F.center() == {F.identity}

    (e,) = free_group("")
    assert e.order() == 1
    assert e.generators == ()
    assert e.elements == {e.identity}
    assert e.is_abelian == True


def test_FreeGroup__hash__():
    assert hash(F)


def test_FreeGroup__eq__():
    assert free_group("x, y, z")[0] == free_group("x, y, z")[0]
    assert free_group("x, y, z")[0] is free_group("x, y, z")[0]

    assert free_group("x, y, z")[0] != free_group("a, x, y")[0]
    assert free_group("x, y, z")[0] is not free_group("a, x, y")[0]

    assert free_group("x, y")[0] != free_group("x, y, z")[0]
    assert free_group("x, y")[0] is not free_group("x, y, z")[0]

    assert free_group("x, y, z")[0] != free_group("x, y")[0]
    assert free_group("x, y, z")[0] is not free_group("x, y")[0]


def test_FreeGroup__getitem__():
    assert F[0:] == FreeGroup("x, y, z")
    assert F[1:] == FreeGroup("y, z")
    assert F[2:] == FreeGroup("z")


def test_FreeGroupElm__hash__():
    assert hash(x*y*z)


def test_FreeGroupElm_copy():
    f = x*y*z**3
    g = f.copy()
    h = x*y*z**7

    assert f == g
    assert f != h


def test_FreeGroupElm_inverse():
    assert x.inverse() == x**-1
    assert (x*y).inverse() == y**-1*x**-1
    assert (y*x*y**-1).inverse() == y*x**-1*y**-1
    assert (y**2*x**-1).inverse() == x*y**-2


def test_FreeGroupElm_type_error():
    raises(TypeError, lambda: 2/x)
    raises(TypeError, lambda: x**2 + y**2)
    raises(TypeError, lambda: x/2)


def test_FreeGroupElm_methods():
    assert (x**0).order() == 1
    assert (y**2).order() is oo
    assert (x**-1*y).commutator(x) == y**-1*x**-1*y*x
    assert len(x**2*y**-1) == 3
    assert len(x**-1*y**3*z) == 5


def test_FreeGroupElm_eliminate_word():
    w = x**5*y*x**2*y**-4*x
    assert w.eliminate_word( x, x**2 ) == x**10*y*x**4*y**-4*x**2
    w3 = x**2*y**3*x**-1*y
    assert w3.eliminate_word(x, x**2) == x**4*y**3*x**-2*y
    assert w3.eliminate_word(x, y) == y**5
    assert w3.eliminate_word(x, y**4) == y**8
    assert w3.eliminate_word(y, x**-1) == x**-3
    assert w3.eliminate_word(x, y*z) == y*z*y*z*y**3*z**-1
    assert (y**-3).eliminate_word(y, x**-1*z**-1) == z*x*z*x*z*x
    #assert w3.eliminate_word(x, y*x) == y*x*y*x**2*y*x*y*x*y*x*z**3
    #assert w3.eliminate_word(x, x*y) == x*y*x**2*y*x*y*x*y*x*y*z**3


def test_FreeGroupElm_array_form():
    assert (x*z).array_form == ((Symbol('x'), 1), (Symbol('z'), 1))
    assert (x**2*z*y*x**-2).array_form == \
        ((Symbol('x'), 2), (Symbol('z'), 1), (Symbol('y'), 1), (Symbol('x'), -2))
    assert (x**-2*y**-1).array_form == ((Symbol('x'), -2), (Symbol('y'), -1))


def test_FreeGroupElm_letter_form():
    assert (x**3).letter_form == (Symbol('x'), Symbol('x'), Symbol('x'))
    assert (x**2*z**-2*x).letter_form == \
        (Symbol('x'), Symbol('x'), -Symbol('z'), -Symbol('z'), Symbol('x'))


def test_FreeGroupElm_ext_rep():
    assert (x**2*z**-2*x).ext_rep == \
        (Symbol('x'), 2, Symbol('z'), -2, Symbol('x'), 1)
    assert (x**-2*y**-1).ext_rep == (Symbol('x'), -2, Symbol('y'), -1)
    assert (x*z).ext_rep == (Symbol('x'), 1, Symbol('z'), 1)


def test_FreeGroupElm__mul__pow__():
    x1 = x.group.dtype(((Symbol('x'), 1),))
    assert x**2 == x1*x

    assert (x**2*y*x**-2)**4 == x**2*y**4*x**-2
    assert (x**2)**2 == x**4
    assert (x**-1)**-1 == x
    assert (x**-1)**0 == F.identity
    assert (y**2)**-2 == y**-4

    assert x**2*x**-1 == x
    assert x**2*y**2*y**-1 == x**2*y
    assert x*x**-1 == F.identity

    assert x/x == F.identity
    assert x/x**2 == x**-1
    assert (x**2*y)/(x**2*y**-1) == x**2*y**2*x**-2
    assert (x**2*y)/(y**-1*x**2) == x**2*y*x**-2*y

    assert x*(x**-1*y*z*y**-1) == y*z*y**-1
    assert x**2*(x**-2*y**-1*z**2*y) == y**-1*z**2*y

    a = F.identity
    for n in range(10):
        assert a == x**n
        assert a**-1 == x**-n
        a *= x


def test_FreeGroupElm__len__():
    assert len(x**5*y*x**2*y**-4*x) == 13
    assert len(x**17) == 17
    assert len(y**0) == 0


def test_FreeGroupElm_comparison():
    assert not (x*y == y*x)
    assert x**0 == y**0

    assert x**2 < y**3
    assert not x**3 < y**2
    assert x*y < x**2*y
    assert x**2*y**2 < y**4
    assert not y**4 < y**-4
    assert not y**4 < x**-4
    assert y**-2 < y**2

    assert x**2 <= y**2
    assert x**2 <= x**2

    assert not y*z > z*y
    assert x > x**-1

    assert not x**2 >= y**2


def test_FreeGroupElm_syllables():
    w = x**5*y*x**2*y**-4*x
    assert w.number_syllables() == 5
    assert w.exponent_syllable(2) == 2
    assert w.generator_syllable(3) == Symbol('y')
    assert w.sub_syllables(1, 2) == y
    assert w.sub_syllables(3, 3) == F.identity


def test_FreeGroup_exponents():
    w1 = x**2*y**3
    assert w1.exponent_sum(x) == 2
    assert w1.exponent_sum(x**-1) == -2
    assert w1.generator_count(x) == 2

    w2 = x**2*y**4*x**-3
    assert w2.exponent_sum(x) == -1
    assert w2.generator_count(x) == 5


def test_FreeGroup_generators():
    assert (x**2*y**4*z**-1).contains_generators() == {x, y, z}
    assert (x**-1*y**3).contains_generators() == {x, y}


def test_FreeGroupElm_words():
    w = x**5*y*x**2*y**-4*x
    assert w.subword(2, 6) == x**3*y
    assert w.subword(3, 2) == F.identity
    assert w.subword(6, 10) == x**2*y**-2

    assert w.substituted_word(0, 7, y**-1) == y**-1*x*y**-4*x
    assert w.substituted_word(0, 7, y**2*x) == y**2*x**2*y**-4*x

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_complex.py ===
from sympy.core.function import expand_complex
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)

def test_complex():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    e = (a + I*b)*(a - I*b)
    assert e.expand() == a**2 + b**2
    assert sqrt(I) == Pow(I, S.Half)


def test_conjugate():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    c = Symbol("c", imaginary=True)
    d = Symbol("d", imaginary=True)
    x = Symbol('x')
    z = a + I*b + c + I*d
    zc = a - I*b - c + I*d
    assert conjugate(z) == zc
    assert conjugate(exp(z)) == exp(zc)
    assert conjugate(exp(I*x)) == exp(-I*conjugate(x))
    assert conjugate(z**5) == zc**5
    assert conjugate(abs(x)) == abs(x)
    assert conjugate(sign(z)) == sign(zc)
    assert conjugate(sin(z)) == sin(zc)
    assert conjugate(cos(z)) == cos(zc)
    assert conjugate(tan(z)) == tan(zc)
    assert conjugate(cot(z)) == cot(zc)
    assert conjugate(sinh(z)) == sinh(zc)
    assert conjugate(cosh(z)) == cosh(zc)
    assert conjugate(tanh(z)) == tanh(zc)
    assert conjugate(coth(z)) == coth(zc)


def test_abs1():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    assert abs(a) == Abs(a)
    assert abs(-a) == abs(a)
    assert abs(a + I*b) == sqrt(a**2 + b**2)


def test_abs2():
    a = Symbol("a", real=False)
    b = Symbol("b", real=False)
    assert abs(a) != a
    assert abs(-a) != a
    assert abs(a + I*b) != sqrt(a**2 + b**2)


def test_evalc():
    x = Symbol("x", real=True)
    y = Symbol("y", real=True)
    z = Symbol("z")
    assert ((x + I*y)**2).expand(complex=True) == x**2 + 2*I*x*y - y**2
    assert expand_complex(z**(2*I)) == (re((re(z) + I*im(z))**(2*I)) +
        I*im((re(z) + I*im(z))**(2*I)))
    assert expand_complex(
        z**(2*I), deep=False) == I*im(z**(2*I)) + re(z**(2*I))

    assert exp(I*x) != cos(x) + I*sin(x)
    assert exp(I*x).expand(complex=True) == cos(x) + I*sin(x)
    assert exp(I*x + y).expand(complex=True) == exp(y)*cos(x) + I*sin(x)*exp(y)

    assert sin(I*x).expand(complex=True) == I * sinh(x)
    assert sin(x + I*y).expand(complex=True) == sin(x)*cosh(y) + \
        I * sinh(y) * cos(x)

    assert cos(I*x).expand(complex=True) == cosh(x)
    assert cos(x + I*y).expand(complex=True) == cos(x)*cosh(y) - \
        I * sinh(y) * sin(x)

    assert tan(I*x).expand(complex=True) == tanh(x) * I
    assert tan(x + I*y).expand(complex=True) == (
        sin(2*x)/(cos(2*x) + cosh(2*y)) +
        I*sinh(2*y)/(cos(2*x) + cosh(2*y)))

    assert sinh(I*x).expand(complex=True) == I * sin(x)
    assert sinh(x + I*y).expand(complex=True) == sinh(x)*cos(y) + \
        I * sin(y) * cosh(x)

    assert cosh(I*x).expand(complex=True) == cos(x)
    assert cosh(x + I*y).expand(complex=True) == cosh(x)*cos(y) + \
        I * sin(y) * sinh(x)

    assert tanh(I*x).expand(complex=True) == tan(x) * I
    assert tanh(x + I*y).expand(complex=True) == (
        (sinh(x)*cosh(x) + I*cos(y)*sin(y)) /
        (sinh(x)**2 + cos(y)**2)).expand()


def test_pythoncomplex():
    x = Symbol("x")
    assert 4j*x != 4*x*I
    assert 4j*x == 4.0*x*I
    assert 4.1j*x != 4*x*I


def test_rootcomplex():
    R = Rational
    assert ((+1 + I)**R(1, 2)).expand(
        complex=True) == 2**R(1, 4)*cos(  pi/8) + 2**R(1, 4)*sin(  pi/8)*I
    assert ((-1 - I)**R(1, 2)).expand(
        complex=True) == 2**R(1, 4)*cos(3*pi/8) - 2**R(1, 4)*sin(3*pi/8)*I
    assert (sqrt(-10)*I).as_real_imag() == (-sqrt(10), 0)


def test_expand_inverse():
    assert (1/(1 + I)).expand(complex=True) == (1 - I)/2
    assert ((1 + 2*I)**(-2)).expand(complex=True) == (-3 - 4*I)/25
    assert ((1 + I)**(-8)).expand(complex=True) == Rational(1, 16)


def test_expand_complex():
    assert ((2 + 3*I)**10).expand(complex=True) == -341525 - 145668*I
    # the following two tests are to ensure the SymPy uses an efficient
    # algorithm for calculating powers of complex numbers. They should execute
    # in something like 0.01s.
    assert ((2 + 3*I)**1000).expand(complex=True) == \
        -81079464736246615951519029367296227340216902563389546989376269312984127074385455204551402940331021387412262494620336565547972162814110386834027871072723273110439771695255662375718498785908345629702081336606863762777939617745464755635193139022811989314881997210583159045854968310911252660312523907616129080027594310008539817935736331124833163907518549408018652090650537035647520296539436440394920287688149200763245475036722326561143851304795139005599209239350981457301460233967137708519975586996623552182807311159141501424576682074392689622074945519232029999 + \
        46938745946789557590804551905243206242164799136976022474337918748798900569942573265747576032611189047943842446167719177749107138603040963603119861476016947257034472364028585381714774667326478071264878108114128915685688115488744955550920239128462489496563930809677159214598114273887061533057125164518549173898349061972857446844052995037423459472376202251620778517659247970283904820245958198842631651569984310559418135975795868314764489884749573052997832686979294085577689571149679540256349988338406458116270429842222666345146926395233040564229555893248370000*I
    assert ((2 + 3*I/4)**1000).expand(complex=True) == \
        Integer(1)*37079892761199059751745775382463070250205990218394308874593455293485167797989691280095867197640410033222367257278387021789651672598831503296531725827158233077451476545928116965316544607115843772405184272449644892857783761260737279675075819921259597776770965829089907990486964515784097181964312256560561065607846661496055417619388874421218472707497847700629822858068783288579581649321248495739224020822198695759609598745114438265083593711851665996586461937988748911532242908776883696631067311443171682974330675406616373422505939887984366289623091300746049101284856530270685577940283077888955692921951247230006346681086274961362500646889925803654263491848309446197554307105991537357310209426736453173441104334496173618419659521888945605315751089087820455852582920963561495787655250624781448951403353654348109893478206364632640344111022531861683064175862889459084900614967785405977231549003280842218501570429860550379522498497412180001/114813069527425452423283320117768198402231770208869520047764273682576626139237031385665948631650626991844596463898746277344711896086305533142593135616665318539129989145312280000688779148240044871428926990063486244781615463646388363947317026040466353970904996558162398808944629605623311649536164221970332681344168908984458505602379484807914058900934776500429002716706625830522008132236281291761267883317206598995396418127021779858404042159853183251540889433902091920554957783589672039160081957216630582755380425583726015528348786419432054508915275783882625175435528800822842770817965453762184851149029376 + \
        I*421638390580169706973991429333213477486930178424989246669892530737775352519112934278994501272111385966211392610029433824534634841747911783746811994443436271013377059560245191441549885048056920190833693041257216263519792201852046825443439142932464031501882145407459174948712992271510309541474392303461939389368955986650538525895866713074543004916049550090364398070215427272240155060576252568700906004691224321432509053286859100920489253598392100207663785243368195857086816912514025693453058403158416856847185079684216151337200057494966741268925263085619240941610301610538225414050394612058339070756009433535451561664522479191267503989904464718368605684297071150902631208673621618217106272361061676184840810762902463998065947687814692402219182668782278472952758690939877465065070481351343206840649517150634973307937551168752642148704904383991876969408056379195860410677814566225456558230131911142229028179902418223009651437985670625/1793954211366022694113801876840128100034871409513586250746316776290259783425578615401030447369541046747571819748417910583511123376348523955353017744010395602173906080395504375010762174191250701116076984219741972574712741619474818186676828531882286780795390571221287481389759837587864244524002565968286448146002639202882164150037179450123657170327105882819203167448541028601906377066191895183769810676831353109303069033234715310287563158747705988305326397404720186258671215368588625611876280581509852855552819149745718992630449787803625851701801184123166018366180137512856918294030710215034138299203584
    assert ((2 + 3*I)**-1000).expand(complex=True) == \
        Integer(1)*-81079464736246615951519029367296227340216902563389546989376269312984127074385455204551402940331021387412262494620336565547972162814110386834027871072723273110439771695255662375718498785908345629702081336606863762777939617745464755635193139022811989314881997210583159045854968310911252660312523907616129080027594310008539817935736331124833163907518549408018652090650537035647520296539436440394920287688149200763245475036722326561143851304795139005599209239350981457301460233967137708519975586996623552182807311159141501424576682074392689622074945519232029999/8777125472973511649630750050295188683351430110097915876250894978429797369155961290321829625004920141758416719066805645579710744290541337680113772670040386863849283653078324415471816788604945889094925784900885812724984087843737442111926413818245854362613018058774368703971604921858023116665586358870612944209398056562604561248859926344335598822815885851096698226775053153403320782439987679978321289537645645163767251396759519805603090332694449553371530571613352311006350058217982509738362083094920649452123351717366337410243853659113315547584871655479914439219520157174729130746351059075207407866012574386726064196992865627149566238044625779078186624347183905913357718850537058578084932880569701242598663149911276357125355850792073635533676541250531086757377369962506979378337216411188347761901006460813413505861461267545723590468627854202034450569581626648934062198718362303420281555886394558137408159453103395918783625713213314350531051312551733021627153081075080140680608080529736975658786227362251632725009435866547613598753584705455955419696609282059191031962604169242974038517575645939316377801594539335940001 - Integer(1)*46938745946789557590804551905243206242164799136976022474337918748798900569942573265747576032611189047943842446167719177749107138603040963603119861476016947257034472364028585381714774667326478071264878108114128915685688115488744955550920239128462489496563930809677159214598114273887061533057125164518549173898349061972857446844052995037423459472376202251620778517659247970283904820245958198842631651569984310559418135975795868314764489884749573052997832686979294085577689571149679540256349988338406458116270429842222666345146926395233040564229555893248370000*I/8777125472973511649630750050295188683351430110097915876250894978429797369155961290321829625004920141758416719066805645579710744290541337680113772670040386863849283653078324415471816788604945889094925784900885812724984087843737442111926413818245854362613018058774368703971604921858023116665586358870612944209398056562604561248859926344335598822815885851096698226775053153403320782439987679978321289537645645163767251396759519805603090332694449553371530571613352311006350058217982509738362083094920649452123351717366337410243853659113315547584871655479914439219520157174729130746351059075207407866012574386726064196992865627149566238044625779078186624347183905913357718850537058578084932880569701242598663149911276357125355850792073635533676541250531086757377369962506979378337216411188347761901006460813413505861461267545723590468627854202034450569581626648934062198718362303420281555886394558137408159453103395918783625713213314350531051312551733021627153081075080140680608080529736975658786227362251632725009435866547613598753584705455955419696609282059191031962604169242974038517575645939316377801594539335940001
    assert ((2 + 3*I/4)**-1000).expand(complex=True) == \
        Integer(1)*4257256305661027385394552848555894604806501409793288342610746813288539790051927148781268212212078237301273165351052934681382567968787279534591114913777456610214738290619922068269909423637926549603264174216950025398244509039145410016404821694746262142525173737175066432954496592560621330313807235750500564940782099283410261748370262433487444897446779072067625787246390824312580440138770014838135245148574339248259670887549732495841810961088930810608893772914812838358159009303794863047635845688453859317690488124382253918725010358589723156019888846606295866740117645571396817375322724096486161308083462637370825829567578309445855481578518239186117686659177284332344643124760453112513611749309168470605289172320376911472635805822082051716625171429727162039621902266619821870482519063133136820085579315127038372190224739238686708451840610064871885616258831386810233957438253532027049148030157164346719204500373766157143311767338973363806106967439378604898250533766359989107510507493549529158818602327525235240510049484816090584478644771183158342479140194633579061295740839490629457435283873180259847394582069479062820225159699506175855369539201399183443253793905149785994830358114153241481884290274629611529758663543080724574566578220908907477622643689220814376054314972190402285121776593824615083669045183404206291739005554569305329760211752815718335731118664756831942466773261465213581616104242113894521054475516019456867271362053692785300826523328020796670205463390909136593859765912483565093461468865534470710132881677639651348709376/2103100954337624833663208713697737151593634525061637972297915388685604042449504336765884978184588688426595940401280828953096857809292320006227881797146858511436638446932833617514351442216409828605662238790280753075176269765767010004889778647709740770757817960711900340755635772183674511158570690702969774966791073165467918123298694584729211212414462628433370481195120564586361368504153395406845170075275051749019600057116719726628746724489572189061061036426955163696859127711110719502594479795200686212257570291758725259007379710596548777812659422174199194837355646482046783616494013289495563083118517507178847555801163089723056310287760875135196081975602765511153122381201303871673391366630940702817360340900568748719988954847590748960761446218262344767250783946365392689256634180417145926390656439421745644011831124277463643383712803287985472471755648426749842410972650924240795946699346613614779460399530274263580007672855851663196114585312432954432654691485867618908420370875753749297487803461900447407917655296784879220450937110470920633595689721819488638484547259978337741496090602390463594556401615298457456112485536498177883358587125449801777718900375736758266215245325999241624148841915093787519330809347240990363802360596034171167818310322276373120180985148650099673289383722502488957717848531612020897298448601714154586319660314294591620415272119454982220034319689607295960162971300417552364254983071768070124456169427638371140064235083443242844616326538396503937972586505546495649094344512270582463639152160238137952390380581401171977159154009407415523525096743009110916334144716516647041176989758534635251844947906038080852185583742296318878233394998111078843229681030277039104786225656992262073797524057992347971177720807155842376332851559276430280477639539393920006008737472164850104411971830120295750221200029811143140323763349636629725073624360001 - Integer(1)*3098214262599218784594285246258841485430681674561917573155883806818465520660668045042109232930382494608383663464454841313154390741655282039877410154577448327874989496074260116195788919037407420625081798124301494353693248757853222257918294662198297114746822817460991242508743651430439120439020484502408313310689912381846149597061657483084652685283853595100434135149479564507015504022249330340259111426799121454516345905101620532787348293877485702600390665276070250119465888154331218827342488849948540687659846652377277250614246402784754153678374932540789808703029043827352976139228402417432199779415751301480406673762521987999573209628597459357964214510139892316208670927074795773830798600837815329291912002136924506221066071242281626618211060464126372574400100990746934953437169840312584285942093951405864225230033279614235191326102697164613004299868695519642598882914862568516635347204441042798206770888274175592401790040170576311989738272102077819127459014286741435419468254146418098278519775722104890854275995510700298782146199325790002255362719776098816136732897323406228294203133323296591166026338391813696715894870956511298793595675308998014158717167429941371979636895553724830981754579086664608880698350866487717403917070872269853194118364230971216854931998642990452908852258008095741042117326241406479532880476938937997238098399302185675832474590293188864060116934035867037219176916416481757918864533515526389079998129329045569609325290897577497835388451456680707076072624629697883854217331728051953671643278797380171857920000*I/2103100954337624833663208713697737151593634525061637972297915388685604042449504336765884978184588688426595940401280828953096857809292320006227881797146858511436638446932833617514351442216409828605662238790280753075176269765767010004889778647709740770757817960711900340755635772183674511158570690702969774966791073165467918123298694584729211212414462628433370481195120564586361368504153395406845170075275051749019600057116719726628746724489572189061061036426955163696859127711110719502594479795200686212257570291758725259007379710596548777812659422174199194837355646482046783616494013289495563083118517507178847555801163089723056310287760875135196081975602765511153122381201303871673391366630940702817360340900568748719988954847590748960761446218262344767250783946365392689256634180417145926390656439421745644011831124277463643383712803287985472471755648426749842410972650924240795946699346613614779460399530274263580007672855851663196114585312432954432654691485867618908420370875753749297487803461900447407917655296784879220450937110470920633595689721819488638484547259978337741496090602390463594556401615298457456112485536498177883358587125449801777718900375736758266215245325999241624148841915093787519330809347240990363802360596034171167818310322276373120180985148650099673289383722502488957717848531612020897298448601714154586319660314294591620415272119454982220034319689607295960162971300417552364254983071768070124456169427638371140064235083443242844616326538396503937972586505546495649094344512270582463639152160238137952390380581401171977159154009407415523525096743009110916334144716516647041176989758534635251844947906038080852185583742296318878233394998111078843229681030277039104786225656992262073797524057992347971177720807155842376332851559276430280477639539393920006008737472164850104411971830120295750221200029811143140323763349636629725073624360001

    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    assert exp(a*(2 + I*b)).expand(complex=True) == \
        I*exp(2*a)*sin(a*b) + exp(2*a)*cos(a*b)


def test_expand():
    f = (16 - 2*sqrt(29))**2
    assert f.expand() == 372 - 64*sqrt(29)
    f = (Integer(1)/2 + I/2)**10
    assert f.expand() == I/32
    f = (Integer(1)/2 + I)**10
    assert f.expand() == Integer(237)/1024 - 779*I/256


def test_re_im1652():
    x = Symbol('x')
    assert re(x) == re(conjugate(x))
    assert im(x) == - im(conjugate(x))
    assert im(x)*re(conjugate(x)) + im(conjugate(x)) * re(x) == 0


def test_issue_5084():
    x = Symbol('x')
    assert ((x + x*I)/(1 + I)).as_real_imag() == (re((x + I*x)/(1 + I)
            ), im((x + I*x)/(1 + I)))


def test_issue_5236():
    assert (cos(1 + I)**3).as_real_imag() == (-3*sin(1)**2*sinh(1)**2*cos(1)*cosh(1) +
        cos(1)**3*cosh(1)**3, -3*cos(1)**2*cosh(1)**2*sin(1)*sinh(1) + sin(1)**3*sinh(1)**3)


def test_real_imag():
    x, y, z = symbols('x, y, z')
    X, Y, Z = symbols('X, Y, Z', commutative=False)
    a = Symbol('a', real=True)
    assert (2*a*x).as_real_imag() == (2*a*re(x), 2*a*im(x))

    # issue 5395:
    assert (x*x.conjugate()).as_real_imag() == (Abs(x)**2, 0)
    assert im(x*x.conjugate()) == 0
    assert im(x*y.conjugate()*z*y) == im(x*z)*Abs(y)**2
    assert im(x*y.conjugate()*x*y) == im(x**2)*Abs(y)**2
    assert im(Z*y.conjugate()*X*y) == im(Z*X)*Abs(y)**2
    assert im(X*X.conjugate()) == im(X*X.conjugate(), evaluate=False)
    assert (sin(x)*sin(x).conjugate()).as_real_imag() == \
        (Abs(sin(x))**2, 0)

    # issue 6573:
    assert (x**2).as_real_imag() == (re(x)**2 - im(x)**2, 2*re(x)*im(x))

    # issue 6428:
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    assert (i*r*x).as_real_imag() == (I*i*r*im(x), -I*i*r*re(x))
    assert (i*r*x*(y + 2)).as_real_imag() == (
        I*i*r*(re(y) + 2)*im(x) + I*i*r*re(x)*im(y),
        -I*i*r*(re(y) + 2)*re(x) + I*i*r*im(x)*im(y))

    # issue 7106:
    assert ((1 + I)/(1 - I)).as_real_imag() == (0, 1)
    assert ((1 + 2*I)*(1 + 3*I)).as_real_imag() == (-5, 5)


def test_pow_issue_1724():
    e = ((S.NegativeOne)**(S.One/3))
    assert e.conjugate().n() == e.n().conjugate()
    e = S('-2/3 - (-29/54 + sqrt(93)/18)**(1/3) - 1/(9*(-29/54 + sqrt(93)/18)**(1/3))')
    assert e.conjugate().n() == e.n().conjugate()
    e = 2**I
    assert e.conjugate().n() == e.n().conjugate()


def test_issue_5429():
    assert sqrt(I).conjugate() != sqrt(I)

def test_issue_4124():
    from sympy.core.numbers import oo
    assert expand_complex(I*oo) == oo*I

def test_issue_11518():
    x = Symbol("x", real=True)
    y = Symbol("y", real=True)
    r = sqrt(x**2 + y**2)
    assert conjugate(r) == r
    s = abs(x + I * y)
    assert conjugate(s) == r

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_round.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameRound:
    def test_round(self):
        # GH#2665

        # Test that rounding an empty DataFrame does nothing
        df = DataFrame()
        tm.assert_frame_equal(df, df.round())

        # Here's the test frame we'll be working with
        df = DataFrame({"col1": [1.123, 2.123, 3.123], "col2": [1.234, 2.234, 3.234]})

        # Default round to integer (i.e. decimals=0)
        expected_rounded = DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [1.0, 2.0, 3.0]})
        tm.assert_frame_equal(df.round(), expected_rounded)

        # Round with an integer
        decimals = 2
        expected_rounded = DataFrame(
            {"col1": [1.12, 2.12, 3.12], "col2": [1.23, 2.23, 3.23]}
        )
        tm.assert_frame_equal(df.round(decimals), expected_rounded)

        # This should also work with np.round (since np.round dispatches to
        # df.round)
        tm.assert_frame_equal(np.round(df, decimals), expected_rounded)

        # Round with a list
        round_list = [1, 2]
        msg = "decimals must be an integer, a dict-like or a Series"
        with pytest.raises(TypeError, match=msg):
            df.round(round_list)

        # Round with a dictionary
        expected_rounded = DataFrame(
            {"col1": [1.1, 2.1, 3.1], "col2": [1.23, 2.23, 3.23]}
        )
        round_dict = {"col1": 1, "col2": 2}
        tm.assert_frame_equal(df.round(round_dict), expected_rounded)

        # Incomplete dict
        expected_partially_rounded = DataFrame(
            {"col1": [1.123, 2.123, 3.123], "col2": [1.2, 2.2, 3.2]}
        )
        partial_round_dict = {"col2": 1}
        tm.assert_frame_equal(df.round(partial_round_dict), expected_partially_rounded)

        # Dict with unknown elements
        wrong_round_dict = {"col3": 2, "col2": 1}
        tm.assert_frame_equal(df.round(wrong_round_dict), expected_partially_rounded)

        # float input to `decimals`
        non_int_round_dict = {"col1": 1, "col2": 0.5}
        msg = "Values in decimals must be integers"
        with pytest.raises(TypeError, match=msg):
            df.round(non_int_round_dict)

        # String input
        non_int_round_dict = {"col1": 1, "col2": "foo"}
        with pytest.raises(TypeError, match=msg):
            df.round(non_int_round_dict)

        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError, match=msg):
            df.round(non_int_round_Series)

        # List input
        non_int_round_dict = {"col1": 1, "col2": [1, 2]}
        with pytest.raises(TypeError, match=msg):
            df.round(non_int_round_dict)

        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError, match=msg):
            df.round(non_int_round_Series)

        # Non integer Series inputs
        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError, match=msg):
            df.round(non_int_round_Series)

        non_int_round_Series = Series(non_int_round_dict)
        with pytest.raises(TypeError, match=msg):
            df.round(non_int_round_Series)

        # Negative numbers
        negative_round_dict = {"col1": -1, "col2": -2}
        big_df = df * 100
        expected_neg_rounded = DataFrame(
            {"col1": [110.0, 210, 310], "col2": [100.0, 200, 300]}
        )
        tm.assert_frame_equal(big_df.round(negative_round_dict), expected_neg_rounded)

        # nan in Series round
        nan_round_Series = Series({"col1": np.nan, "col2": 1})

        with pytest.raises(TypeError, match=msg):
            df.round(nan_round_Series)

        # Make sure this doesn't break existing Series.round
        tm.assert_series_equal(df["col1"].round(1), expected_rounded["col1"])

        # named columns
        # GH#11986
        decimals = 2
        expected_rounded = DataFrame(
            {"col1": [1.12, 2.12, 3.12], "col2": [1.23, 2.23, 3.23]}
        )
        df.columns.name = "cols"
        expected_rounded.columns.name = "cols"
        tm.assert_frame_equal(df.round(decimals), expected_rounded)

        # interaction of named columns & series
        tm.assert_series_equal(df["col1"].round(decimals), expected_rounded["col1"])
        tm.assert_series_equal(df.round(decimals)["col1"], expected_rounded["col1"])

    def test_round_numpy(self):
        # GH#12600
        df = DataFrame([[1.53, 1.36], [0.06, 7.01]])
        out = np.round(df, decimals=0)
        expected = DataFrame([[2.0, 1.0], [0.0, 7.0]])
        tm.assert_frame_equal(out, expected)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.round(df, decimals=0, out=df)

    def test_round_numpy_with_nan(self):
        # See GH#14197
        df = Series([1.53, np.nan, 0.06]).to_frame()
        with tm.assert_produces_warning(None):
            result = df.round()
        expected = Series([2.0, np.nan, 0.0]).to_frame()
        tm.assert_frame_equal(result, expected)

    def test_round_mixed_type(self):
        # GH#11885
        df = DataFrame(
            {
                "col1": [1.1, 2.2, 3.3, 4.4],
                "col2": ["1", "a", "c", "f"],
                "col3": date_range("20111111", periods=4),
            }
        )
        round_0 = DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": ["1", "a", "c", "f"],
                "col3": date_range("20111111", periods=4),
            }
        )
        tm.assert_frame_equal(df.round(), round_0)
        tm.assert_frame_equal(df.round(1), df)
        tm.assert_frame_equal(df.round({"col1": 1}), df)
        tm.assert_frame_equal(df.round({"col1": 0}), round_0)
        tm.assert_frame_equal(df.round({"col1": 0, "col2": 1}), round_0)
        tm.assert_frame_equal(df.round({"col3": 1}), df)

    def test_round_with_duplicate_columns(self):
        # GH#11611

        df = DataFrame(
            np.random.default_rng(2).random([3, 3]),
            columns=["A", "B", "C"],
            index=["first", "second", "third"],
        )

        dfs = pd.concat((df, df), axis=1)
        rounded = dfs.round()
        tm.assert_index_equal(rounded.index, dfs.index)

        decimals = Series([1, 0, 2], index=["A", "B", "A"])
        msg = "Index of decimals must be unique"
        with pytest.raises(ValueError, match=msg):
            df.round(decimals)

    def test_round_builtin(self):
        # GH#11763
        # Here's the test frame we'll be working with
        df = DataFrame({"col1": [1.123, 2.123, 3.123], "col2": [1.234, 2.234, 3.234]})

        # Default round to integer (i.e. decimals=0)
        expected_rounded = DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [1.0, 2.0, 3.0]})
        tm.assert_frame_equal(round(df), expected_rounded)

    def test_round_nonunique_categorical(self):
        # See GH#21809
        idx = pd.CategoricalIndex(["low"] * 3 + ["hi"] * 3)
        df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=list("abc"))

        expected = df.round(3)
        expected.index = idx

        df_categorical = df.copy().set_index(idx)
        assert df_categorical.shape == (6, 3)
        result = df_categorical.round(3)
        assert result.shape == (6, 3)

        tm.assert_frame_equal(result, expected)

    def test_round_interval_category_columns(self):
        # GH#30063
        columns = pd.CategoricalIndex(pd.interval_range(0, 2))
        df = DataFrame([[0.66, 1.1], [0.3, 0.25]], columns=columns)

        result = df.round()
        expected = DataFrame([[1.0, 1.0], [0.0, 0.0]], columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_round_empty_not_input(self):
        # GH#51032
        df = DataFrame()
        result = df.round()
        tm.assert_frame_equal(df, result)
        assert df is not result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\methods\test_to_period.py ===
import dateutil.tz
from dateutil.tz import tzlocal
import pytest
import pytz

from pandas._libs.tslibs.ccalendar import MONTHS
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

from pandas import (
    DatetimeIndex,
    Period,
    PeriodIndex,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestToPeriod:
    def test_dti_to_period(self):
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")
        pi1 = dti.to_period()
        pi2 = dti.to_period(freq="D")
        pi3 = dti.to_period(freq="3D")

        assert pi1[0] == Period("Jan 2005", freq="M")
        assert pi2[0] == Period("1/31/2005", freq="D")
        assert pi3[0] == Period("1/31/2005", freq="3D")

        assert pi1[-1] == Period("Nov 2005", freq="M")
        assert pi2[-1] == Period("11/30/2005", freq="D")
        assert pi3[-1], Period("11/30/2005", freq="3D")

        tm.assert_index_equal(pi1, period_range("1/1/2005", "11/1/2005", freq="M"))
        tm.assert_index_equal(
            pi2, period_range("1/1/2005", "11/1/2005", freq="M").asfreq("D")
        )
        tm.assert_index_equal(
            pi3, period_range("1/1/2005", "11/1/2005", freq="M").asfreq("3D")
        )

    @pytest.mark.parametrize("month", MONTHS)
    def test_to_period_quarterly(self, month):
        # make sure we can make the round trip
        freq = f"Q-{month}"
        rng = period_range("1989Q3", "1991Q3", freq=freq)
        stamps = rng.to_timestamp()
        result = stamps.to_period(freq)
        tm.assert_index_equal(rng, result)

    @pytest.mark.parametrize("off", ["BQE", "QS", "BQS"])
    def test_to_period_quarterlyish(self, off):
        rng = date_range("01-Jan-2012", periods=8, freq=off)
        prng = rng.to_period()
        assert prng.freq == "QE-DEC"

    @pytest.mark.parametrize("off", ["BYE", "YS", "BYS"])
    def test_to_period_annualish(self, off):
        rng = date_range("01-Jan-2012", periods=8, freq=off)
        prng = rng.to_period()
        assert prng.freq == "YE-DEC"

    def test_to_period_monthish(self):
        offsets = ["MS", "BME"]
        for off in offsets:
            rng = date_range("01-Jan-2012", periods=8, freq=off)
            prng = rng.to_period()
            assert prng.freqstr == "M"

        rng = date_range("01-Jan-2012", periods=8, freq="ME")
        prng = rng.to_period()
        assert prng.freqstr == "M"

        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            date_range("01-Jan-2012", periods=8, freq="EOM")

    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("2ME", "2M"),
            (MonthEnd(2), MonthEnd(2)),
        ],
    )
    def test_dti_to_period_2monthish(self, freq_offset, freq_period):
        dti = date_range("2020-01-01", periods=3, freq=freq_offset)
        pi = dti.to_period()

        tm.assert_index_equal(pi, period_range("2020-01", "2020-05", freq=freq_period))

    @pytest.mark.parametrize(
        "freq, freq_depr",
        [
            ("2ME", "2M"),
            ("2QE", "2Q"),
            ("2QE-SEP", "2Q-SEP"),
            ("1YE", "1Y"),
            ("2YE-MAR", "2Y-MAR"),
            ("1YE", "1A"),
            ("2YE-MAR", "2A-MAR"),
        ],
    )
    def test_to_period_frequency_M_Q_Y_A_deprecated(self, freq, freq_depr):
        # GH#9586
        msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
        f"in a future version, please use '{freq[1:]}' instead."

        rng = date_range("01-Jan-2012", periods=8, freq=freq)
        prng = rng.to_period()
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert prng.freq == freq_depr

    def test_to_period_infer(self):
        # https://github.com/pandas-dev/pandas/issues/33358
        rng = date_range(
            start="2019-12-22 06:40:00+00:00",
            end="2019-12-22 08:45:00+00:00",
            freq="5min",
        )

        with tm.assert_produces_warning(UserWarning):
            pi1 = rng.to_period("5min")

        with tm.assert_produces_warning(UserWarning):
            pi2 = rng.to_period()

        tm.assert_index_equal(pi1, pi2)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_period_dt64_round_trip(self):
        dti = date_range("1/1/2000", "1/7/2002", freq="B")
        pi = dti.to_period()
        tm.assert_index_equal(pi.to_timestamp(), dti)

        dti = date_range("1/1/2000", "1/7/2002", freq="B")
        pi = dti.to_period(freq="h")
        tm.assert_index_equal(pi.to_timestamp(), dti)

    def test_to_period_millisecond(self):
        index = DatetimeIndex(
            [
                Timestamp("2007-01-01 10:11:12.123456Z"),
                Timestamp("2007-01-01 10:11:13.789123Z"),
            ]
        )

        with tm.assert_produces_warning(UserWarning):
            # warning that timezone info will be lost
            period = index.to_period(freq="ms")
        assert 2 == len(period)
        assert period[0] == Period("2007-01-01 10:11:12.123Z", "ms")
        assert period[1] == Period("2007-01-01 10:11:13.789Z", "ms")

    def test_to_period_microsecond(self):
        index = DatetimeIndex(
            [
                Timestamp("2007-01-01 10:11:12.123456Z"),
                Timestamp("2007-01-01 10:11:13.789123Z"),
            ]
        )

        with tm.assert_produces_warning(UserWarning):
            # warning that timezone info will be lost
            period = index.to_period(freq="us")
        assert 2 == len(period)
        assert period[0] == Period("2007-01-01 10:11:12.123456Z", "us")
        assert period[1] == Period("2007-01-01 10:11:13.789123Z", "us")

    @pytest.mark.parametrize(
        "tz",
        ["US/Eastern", pytz.utc, tzlocal(), "dateutil/US/Eastern", dateutil.tz.tzutc()],
    )
    def test_to_period_tz(self, tz):
        ts = date_range("1/1/2000", "2/1/2000", tz=tz)

        with tm.assert_produces_warning(UserWarning):
            # GH#21333 warning that timezone info will be lost
            # filter warning about freq deprecation

            result = ts.to_period()[0]
            expected = ts[0].to_period(ts.freq)

        assert result == expected

        expected = date_range("1/1/2000", "2/1/2000").to_period()

        with tm.assert_produces_warning(UserWarning):
            # GH#21333 warning that timezone info will be lost
            result = ts.to_period(ts.freq)

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", ["Etc/GMT-1", "Etc/GMT+1"])
    def test_to_period_tz_utc_offset_consistency(self, tz):
        # GH#22905
        ts = date_range("1/1/2000", "2/1/2000", tz="Etc/GMT-1")
        with tm.assert_produces_warning(UserWarning):
            result = ts.to_period()[0]
            expected = ts[0].to_period(ts.freq)
            assert result == expected

    def test_to_period_nofreq(self):
        idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"])
        msg = "You must pass a freq argument as current index has none."
        with pytest.raises(ValueError, match=msg):
            idx.to_period()

        idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"], freq="infer")
        assert idx.freqstr == "D"
        expected = PeriodIndex(["2000-01-01", "2000-01-02", "2000-01-03"], freq="D")
        tm.assert_index_equal(idx.to_period(), expected)

        # GH#7606
        idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03"])
        assert idx.freqstr is None
        tm.assert_index_equal(idx.to_period(), expected)

    @pytest.mark.parametrize("freq", ["2BMS", "1SME-15"])
    def test_to_period_offsets_not_supported(self, freq):
        # GH#56243
        msg = f"{freq[1:]} is not supported as period frequency"
        ts = date_range("1/1/2012", periods=4, freq=freq)
        with pytest.raises(ValueError, match=msg):
            ts.to_period()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\interval\test_interval.py ===
import numpy as np
import pytest

from pandas._libs import index as libindex

import pandas as pd
from pandas import (
    DataFrame,
    IntervalIndex,
    Series,
)
import pandas._testing as tm


class TestIntervalIndex:
    @pytest.fixture
    def series_with_interval_index(self):
        return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))

    def test_getitem_with_scalar(self, series_with_interval_index, indexer_sl):
        ser = series_with_interval_index.copy()

        expected = ser.iloc[:3]
        tm.assert_series_equal(expected, indexer_sl(ser)[:3])
        tm.assert_series_equal(expected, indexer_sl(ser)[:2.5])
        tm.assert_series_equal(expected, indexer_sl(ser)[0.1:2.5])
        if indexer_sl is tm.loc:
            tm.assert_series_equal(expected, ser.loc[-1:3])

        expected = ser.iloc[1:4]
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2.5, 3.5]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[2, 3, 4]])
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 3, 4]])

        expected = ser.iloc[2:5]
        tm.assert_series_equal(expected, indexer_sl(ser)[ser >= 2])

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_getitem_nonoverlapping_monotonic(self, direction, closed, indexer_sl):
        tpls = [(0, 1), (2, 3), (4, 5)]
        if direction == "decreasing":
            tpls = tpls[::-1]

        idx = IntervalIndex.from_tuples(tpls, closed=closed)
        ser = Series(list("abc"), idx)

        for key, expected in zip(idx.left, ser):
            if idx.closed_left:
                assert indexer_sl(ser)[key] == expected
            else:
                with pytest.raises(KeyError, match=str(key)):
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.right, ser):
            if idx.closed_right:
                assert indexer_sl(ser)[key] == expected
            else:
                with pytest.raises(KeyError, match=str(key)):
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.mid, ser):
            assert indexer_sl(ser)[key] == expected

    def test_getitem_non_matching(self, series_with_interval_index, indexer_sl):
        ser = series_with_interval_index.copy()

        # this is a departure from our current
        # indexing scheme, but simpler
        with pytest.raises(KeyError, match=r"\[-1\] not in index"):
            indexer_sl(ser)[[-1, 3, 4, 5]]

        with pytest.raises(KeyError, match=r"\[-1\] not in index"):
            indexer_sl(ser)[[-1, 3]]

    def test_loc_getitem_large_series(self, monkeypatch):
        size_cutoff = 20
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            ser = Series(
                np.arange(size_cutoff),
                index=IntervalIndex.from_breaks(np.arange(size_cutoff + 1)),
            )

            result1 = ser.loc[:8]
            result2 = ser.loc[0:8]
            result3 = ser.loc[0:8:1]
        tm.assert_series_equal(result1, result2)
        tm.assert_series_equal(result1, result3)

    def test_loc_getitem_frame(self):
        # CategoricalIndex with IntervalIndex categories
        df = DataFrame({"A": range(10)})
        ser = pd.cut(df.A, 5)
        df["B"] = ser
        df = df.set_index("B")

        result = df.loc[4]
        expected = df.iloc[4:6]
        tm.assert_frame_equal(result, expected)

        with pytest.raises(KeyError, match="10"):
            df.loc[10]

        # single list-like
        result = df.loc[[4]]
        expected = df.iloc[4:6]
        tm.assert_frame_equal(result, expected)

        # non-unique
        result = df.loc[[4, 5]]
        expected = df.take([4, 5, 4, 5])
        tm.assert_frame_equal(result, expected)

        msg = (
            r"None of \[Index\(\[10\], dtype='object', name='B'\)\] "
            r"are in the \[index\]"
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[10]]

        # partial missing
        with pytest.raises(KeyError, match=r"\[10\] not in index"):
            df.loc[[10, 4]]

    def test_getitem_interval_with_nans(self, frame_or_series, indexer_sl):
        # GH#41831

        index = IntervalIndex([np.nan, np.nan])
        key = index[:-1]

        obj = frame_or_series(range(2), index=index)
        if frame_or_series is DataFrame and indexer_sl is tm.setitem:
            obj = obj.T

        result = indexer_sl(obj)[key]
        expected = obj

        tm.assert_equal(result, expected)

    def test_setitem_interval_with_slice(self):
        # GH#54722
        ii = IntervalIndex.from_breaks(range(4, 15))
        ser = Series(range(10), index=ii)

        orig = ser.copy()

        # This should be a no-op (used to raise)
        ser.loc[1:3] = 20
        tm.assert_series_equal(ser, orig)

        ser.loc[6:8] = 19
        orig.iloc[1:4] = 19
        tm.assert_series_equal(ser, orig)

        ser2 = Series(range(5), index=ii[::2])
        orig2 = ser2.copy()

        # this used to raise
        ser2.loc[6:8] = 22  # <- raises on main, sets on branch
        orig2.iloc[1] = 22
        tm.assert_series_equal(ser2, orig2)

        ser2.loc[5:7] = 21
        orig2.iloc[:2] = 21
        tm.assert_series_equal(ser2, orig2)


class TestIntervalIndexInsideMultiIndex:
    def test_mi_intervalindex_slicing_with_scalar(self):
        # GH#27456
        ii = IntervalIndex.from_arrays(
            [0, 1, 10, 11, 0, 1, 10, 11], [1, 2, 11, 12, 1, 2, 11, 12], name="MP"
        )
        idx = pd.MultiIndex.from_arrays(
            [
                pd.Index(["FC", "FC", "FC", "FC", "OWNER", "OWNER", "OWNER", "OWNER"]),
                pd.Index(
                    ["RID1", "RID1", "RID2", "RID2", "RID1", "RID1", "RID2", "RID2"]
                ),
                ii,
            ]
        )

        idx.names = ["Item", "RID", "MP"]
        df = DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8]})
        df.index = idx

        query_df = DataFrame(
            {
                "Item": ["FC", "OWNER", "FC", "OWNER", "OWNER"],
                "RID": ["RID1", "RID1", "RID1", "RID2", "RID2"],
                "MP": [0.2, 1.5, 1.6, 11.1, 10.9],
            }
        )

        query_df = query_df.sort_index()

        idx = pd.MultiIndex.from_arrays([query_df.Item, query_df.RID, query_df.MP])
        query_df.index = idx
        result = df.value.loc[query_df.index]

        # the IntervalIndex level is indexed with floats, which map to
        #  the intervals containing them.  Matching the behavior we would get
        #  with _only_ an IntervalIndex, we get an IntervalIndex level back.
        sliced_level = ii.take([0, 1, 1, 3, 2])
        expected_index = pd.MultiIndex.from_arrays(
            [idx.get_level_values(0), idx.get_level_values(1), sliced_level]
        )
        expected = Series([1, 6, 2, 8, 7], index=expected_index, name="value")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "base",
        [101, 1010],
    )
    def test_reindex_behavior_with_interval_index(self, base):
        # GH 51826

        ser = Series(
            range(base),
            index=IntervalIndex.from_arrays(range(base), range(1, base + 1)),
        )
        expected_result = Series([np.nan, 0], index=[np.nan, 1.0], dtype=float)
        result = ser.reindex(index=[np.nan, 1.0])
        tm.assert_series_equal(result, expected_result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\conftest.py ===
import shlex
import subprocess
import time
import uuid

import pytest

from pandas.compat import (
    is_ci_environment,
    is_platform_arm,
    is_platform_mac,
    is_platform_windows,
)
import pandas.util._test_decorators as td

import pandas.io.common as icom
from pandas.io.parsers import read_csv


@pytest.fixture
def compression_to_extension():
    return {value: key for key, value in icom.extension_to_compression.items()}


@pytest.fixture
def tips_file(datapath):
    """Path to the tips dataset"""
    return datapath("io", "data", "csv", "tips.csv")


@pytest.fixture
def jsonl_file(datapath):
    """Path to a JSONL dataset"""
    return datapath("io", "parser", "data", "items.jsonl")


@pytest.fixture
def salaries_table(datapath):
    """DataFrame with the salaries dataset"""
    return read_csv(datapath("io", "parser", "data", "salaries.csv"), sep="\t")


@pytest.fixture
def feather_file(datapath):
    return datapath("io", "data", "feather", "feather-0_3_1.feather")


@pytest.fixture
def xml_file(datapath):
    return datapath("io", "data", "xml", "books.xml")


@pytest.fixture
def s3_base(worker_id, monkeypatch):
    """
    Fixture for mocking S3 interaction.

    Sets up moto server in separate process locally
    Return url for motoserver/moto CI service
    """
    pytest.importorskip("s3fs")
    pytest.importorskip("boto3")

    # temporary workaround as moto fails for botocore >= 1.11 otherwise,
    # see https://github.com/spulec/moto/issues/1924 & 1952
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "foobar_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "foobar_secret")
    if is_ci_environment():
        if is_platform_arm() or is_platform_mac() or is_platform_windows():
            # NOT RUN on Windows/macOS, only Ubuntu
            # - subprocess in CI can cause timeouts
            # - GitHub Actions do not support
            #   container services for the above OSs
            pytest.skip(
                "S3 tests do not have a corresponding service on "
                "Windows or macOS platforms"
            )
        else:
            # set in .github/workflows/unit-tests.yml
            yield "http://localhost:5000"
    else:
        requests = pytest.importorskip("requests")
        pytest.importorskip("moto")
        pytest.importorskip("flask")  # server mode needs flask too

        # Launching moto in server mode, i.e., as a separate process
        # with an S3 endpoint on localhost

        worker_id = "5" if worker_id == "master" else worker_id.lstrip("gw")
        endpoint_port = f"555{worker_id}"
        endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

        # pipe to null to avoid logging in terminal
        with subprocess.Popen(
            shlex.split(f"moto_server s3 -p {endpoint_port}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as proc:
            timeout = 5
            while timeout > 0:
                try:
                    # OK to go once server is accepting connections
                    r = requests.get(endpoint_uri)
                    if r.ok:
                        break
                except Exception:
                    pass
                timeout -= 0.1
                time.sleep(0.1)
            yield endpoint_uri

            proc.terminate()


@pytest.fixture
def s3so(s3_base):
    return {"client_kwargs": {"endpoint_url": s3_base}}


@pytest.fixture
def s3_resource(s3_base):
    import boto3

    s3 = boto3.resource("s3", endpoint_url=s3_base)
    return s3


@pytest.fixture
def s3_public_bucket(s3_resource):
    bucket = s3_resource.Bucket(f"pandas-test-{uuid.uuid4()}")
    bucket.create()
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_public_bucket_with_data(
    s3_public_bucket, tips_file, jsonl_file, feather_file, xml_file
):
    """
    The following datasets
    are loaded.

    - tips.csv
    - tips.csv.gz
    - tips.csv.bz2
    - items.jsonl
    """
    test_s3_files = [
        ("tips#1.csv", tips_file),
        ("tips.csv", tips_file),
        ("tips.csv.gz", tips_file + ".gz"),
        ("tips.csv.bz2", tips_file + ".bz2"),
        ("items.jsonl", jsonl_file),
        ("simple_dataset.feather", feather_file),
        ("books.xml", xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, "rb") as f:
            s3_public_bucket.put_object(Key=s3_key, Body=f)
    return s3_public_bucket


@pytest.fixture
def s3_private_bucket(s3_resource):
    bucket = s3_resource.Bucket(f"cant_get_it-{uuid.uuid4()}")
    bucket.create(ACL="private")
    yield bucket
    bucket.objects.delete()
    bucket.delete()


@pytest.fixture
def s3_private_bucket_with_data(
    s3_private_bucket, tips_file, jsonl_file, feather_file, xml_file
):
    """
    The following datasets
    are loaded.

    - tips.csv
    - tips.csv.gz
    - tips.csv.bz2
    - items.jsonl
    """
    test_s3_files = [
        ("tips#1.csv", tips_file),
        ("tips.csv", tips_file),
        ("tips.csv.gz", tips_file + ".gz"),
        ("tips.csv.bz2", tips_file + ".bz2"),
        ("items.jsonl", jsonl_file),
        ("simple_dataset.feather", feather_file),
        ("books.xml", xml_file),
    ]
    for s3_key, file_name in test_s3_files:
        with open(file_name, "rb") as f:
            s3_private_bucket.put_object(Key=s3_key, Body=f)
    return s3_private_bucket


_compression_formats_params = [
    (".no_compress", None),
    ("", None),
    (".gz", "gzip"),
    (".GZ", "gzip"),
    (".bz2", "bz2"),
    (".BZ2", "bz2"),
    (".zip", "zip"),
    (".ZIP", "zip"),
    (".xz", "xz"),
    (".XZ", "xz"),
    pytest.param((".zst", "zstd"), marks=td.skip_if_no("zstandard")),
    pytest.param((".ZST", "zstd"), marks=td.skip_if_no("zstandard")),
]


@pytest.fixture(params=_compression_formats_params[1:])
def compression_format(request):
    return request.param


@pytest.fixture(params=_compression_formats_params)
def compression_ext(request):
    return request.param[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_reset_index.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    date_range,
    option_context,
)
import pandas._testing as tm


class TestResetIndex:
    def test_reset_index_dti_round_trip(self):
        dti = date_range(start="1/1/2001", end="6/1/2001", freq="D")._with_freq(None)
        d1 = DataFrame({"v": np.random.default_rng(2).random(len(dti))}, index=dti)
        d2 = d1.reset_index()
        assert d2.dtypes.iloc[0] == np.dtype("M8[ns]")
        d3 = d2.set_index("index")
        tm.assert_frame_equal(d1, d3, check_names=False)

        # GH#2329
        stamp = datetime(2012, 11, 22)
        df = DataFrame([[stamp, 12.1]], columns=["Date", "Value"])
        df = df.set_index("Date")

        assert df.index[0] == stamp
        assert df.reset_index()["Date"].iloc[0] == stamp

    def test_reset_index(self):
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )[:5]
        ser = df.stack(future_stack=True)
        ser.index.names = ["hash", "category"]

        ser.name = "value"
        df = ser.reset_index()
        assert "value" in df

        df = ser.reset_index(name="value2")
        assert "value2" in df

        # check inplace
        s = ser.reset_index(drop=True)
        s2 = ser
        return_value = s2.reset_index(drop=True, inplace=True)
        assert return_value is None
        tm.assert_series_equal(s, s2)

        # level
        index = MultiIndex(
            levels=[["bar"], ["one", "two", "three"], [0, 1]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
        )
        s = Series(np.random.default_rng(2).standard_normal(6), index=index)
        rs = s.reset_index(level=1)
        assert len(rs.columns) == 2

        rs = s.reset_index(level=[0, 2], drop=True)
        tm.assert_index_equal(rs.index, Index(index.get_level_values(1)))
        assert isinstance(rs, Series)

    def test_reset_index_name(self):
        s = Series([1, 2, 3], index=Index(range(3), name="x"))
        assert s.reset_index().index.name is None
        assert s.reset_index(drop=True).index.name is None

    def test_reset_index_level(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        for levels in ["A", "B"], [0, 1]:
            # With MultiIndex
            s = df.set_index(["A", "B"])["C"]

            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df.set_index("B"))

            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df.set_index("B"))

            result = s.reset_index(level=levels)
            tm.assert_frame_equal(result, df)

            result = df.set_index(["A", "B"]).reset_index(level=levels, drop=True)
            tm.assert_frame_equal(result, df[["C"]])

            with pytest.raises(KeyError, match="Level E "):
                s.reset_index(level=["A", "E"])

            # With single-level Index
            s = df.set_index("A")["B"]

            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df[["A", "B"]])

            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df[["A", "B"]])

            result = s.reset_index(level=levels[0], drop=True)
            tm.assert_series_equal(result, df["B"])

            with pytest.raises(IndexError, match="Too many levels"):
                s.reset_index(level=[0, 1, 2])

        # Check that .reset_index([],drop=True) doesn't fail
        result = Series(range(4)).reset_index([], drop=True)
        expected = Series(range(4))
        tm.assert_series_equal(result, expected)

    def test_reset_index_range(self):
        # GH 12071
        s = Series(range(2), name="A", dtype="int64")
        series_result = s.reset_index()
        assert isinstance(series_result.index, RangeIndex)
        series_expected = DataFrame(
            [[0, 0], [1, 1]], columns=["index", "A"], index=RangeIndex(stop=2)
        )
        tm.assert_frame_equal(series_result, series_expected)

    def test_reset_index_drop_errors(self):
        #  GH 20925

        # KeyError raised for series index when passed level name is missing
        s = Series(range(4))
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong", drop=True)
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong")

        # KeyError raised for series when level to be dropped is missing
        s = Series(range(4), index=MultiIndex.from_product([[1, 2]] * 2))
        with pytest.raises(KeyError, match="not found"):
            s.reset_index("wrong", drop=True)

    def test_reset_index_with_drop(self):
        arrays = [
            ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = zip(*arrays)
        index = MultiIndex.from_tuples(tuples)
        data = np.random.default_rng(2).standard_normal(8)
        ser = Series(data, index=index)
        ser.iloc[3] = np.nan

        deleveled = ser.reset_index()
        assert isinstance(deleveled, DataFrame)
        assert len(deleveled.columns) == len(ser.index.levels) + 1
        assert deleveled.index.name == ser.index.name

        deleveled = ser.reset_index(drop=True)
        assert isinstance(deleveled, Series)
        assert deleveled.index.name == ser.index.name

    def test_reset_index_inplace_and_drop_ignore_name(self):
        # GH#44575
        ser = Series(range(2), name="old")
        ser.reset_index(name="new", drop=True, inplace=True)
        expected = Series(range(2), name="old")
        tm.assert_series_equal(ser, expected)

    def test_reset_index_drop_infer_string(self):
        # GH#56160
        pytest.importorskip("pyarrow")
        ser = Series(["a", "b", "c"], dtype=object)
        with option_context("future.infer_string", True):
            result = ser.reset_index(drop=True)
        tm.assert_series_equal(result, ser)


@pytest.mark.parametrize(
    "array, dtype",
    [
        (["a", "b"], object),
        (
            pd.period_range("12-1-2000", periods=2, freq="Q-DEC"),
            pd.PeriodDtype(freq="Q-DEC"),
        ),
    ],
)
def test_reset_index_dtypes_on_empty_series_with_multiindex(
    array, dtype, using_infer_string
):
    # GH 19602 - Preserve dtype on empty Series with MultiIndex
    idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
    result = Series(dtype=object, index=idx)[:0].reset_index().dtypes
    exp = "str" if using_infer_string else object
    expected = Series(
        {
            "level_0": np.int64,
            "level_1": np.float64,
            "level_2": exp if dtype == object else dtype,
            0: object,
        }
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "names, expected_names",
    [
        (["A", "A"], ["A", "A"]),
        (["level_1", None], ["level_1", "level_1"]),
    ],
)
@pytest.mark.parametrize("allow_duplicates", [False, True])
def test_column_name_duplicates(names, expected_names, allow_duplicates):
    # GH#44755 reset_index with duplicate column labels
    s = Series([1], index=MultiIndex.from_arrays([[1], [1]], names=names))
    if allow_duplicates:
        result = s.reset_index(allow_duplicates=True)
        expected = DataFrame([[1, 1, 1]], columns=expected_names + [0])
        tm.assert_frame_equal(result, expected)
    else:
        with pytest.raises(ValueError, match="cannot insert"):
            s.reset_index()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\torture.py ===
"""
Torture tests for asymptotics and high precision evaluation of
special functions.

(Other torture tests may also be placed here.)

Running this file (gmpy recommended!) takes several CPU minutes.
With Python 2.6+, multiprocessing is used automatically to run tests
in parallel if many cores are available. (A single test may take between
a second and several minutes; possibly more.)

The idea:

* We evaluate functions at positive, negative, imaginary, 45- and 135-degree
  complex values with magnitudes between 10^-20 to 10^20, at precisions between
  5 and 150 digits (we can go even higher for fast functions).

* Comparing the result from two different precision levels provides
  a strong consistency check (particularly for functions that use
  different algorithms at different precision levels).

* That the computation finishes at all (without failure), within reasonable
  time, provides a check that evaluation works at all: that the code runs,
  that it doesn't get stuck in an infinite loop, and that it doesn't use
  some extremely slowly algorithm where it could use a faster one.

TODO:

* Speed up those functions that take long to finish!
* Generalize to test more cases; more options.
* Implement a timeout mechanism.
* Some functions are notably absent, including the following:
  * inverse trigonometric functions (some become inaccurate for complex arguments)
  * ci, si (not implemented properly for large complex arguments)
  * zeta functions (need to modify test not to try too large imaginary values)
  * and others...

"""


import sys, os
from timeit import default_timer as clock

if "-nogmpy" in sys.argv:
    sys.argv.remove('-nogmpy')
    os.environ['MPMATH_NOGMPY'] = 'Y'

filt = ''
if not sys.argv[-1].endswith(".py"):
    filt = sys.argv[-1]

from mpmath import *
from mpmath.libmp.backend import exec_

def test_asymp(f, maxdps=150, verbose=False, huge_range=False):
    dps = [5,15,25,50,90,150,500,1500,5000,10000]
    dps = [p for p in dps if p <= maxdps]
    def check(x,y,p,inpt):
        if abs(x-y)/abs(y) < workprec(20)(power)(10, -p+1):
            return
        print()
        print("Error!")
        print("Input:", inpt)
        print("dps =", p)
        print("Result 1:", x)
        print("Result 2:", y)
        print("Absolute error:", abs(x-y))
        print("Relative error:", abs(x-y)/abs(y))
        raise AssertionError
    exponents = range(-20,20)
    if huge_range:
        exponents += [-1000, -100, -50, 50, 100, 1000]
    for n in exponents:
        if verbose:
            sys.stdout.write(". ")
        mp.dps = 25
        xpos = mpf(10)**n / 1.1287
        xneg = -xpos
        ximag = xpos*j
        xcomplex1 = xpos*(1+j)
        xcomplex2 = xpos*(-1+j)
        for i in range(len(dps)):
            if verbose:
                print("Testing dps = %s" % dps[i])
            mp.dps = dps[i]
            new = f(xpos), f(xneg), f(ximag), f(xcomplex1), f(xcomplex2)
            if i != 0:
                p = dps[i-1]
                check(prev[0], new[0], p, xpos)
                check(prev[1], new[1], p, xneg)
                check(prev[2], new[2], p, ximag)
                check(prev[3], new[3], p, xcomplex1)
                check(prev[4], new[4], p, xcomplex2)
            prev = new
    if verbose:
        print()

a1, a2, a3, a4, a5 = 1.5, -2.25, 3.125, 4, 2

def test_bernoulli_huge():
    p, q = bernfrac(9000)
    assert p % 10**10 == 9636701091
    assert q == 4091851784687571609141381951327092757255270
    mp.dps = 15
    assert str(bernoulli(10**100)) == '-2.58183325604736e+987675256497386331227838638980680030172857347883537824464410652557820800494271520411283004120790908623'
    mp.dps = 50
    assert str(bernoulli(10**100)) == '-2.5818332560473632073252488656039475548106223822913e+987675256497386331227838638980680030172857347883537824464410652557820800494271520411283004120790908623'
    mp.dps = 15

cases = """\
test_bernoulli_huge()
test_asymp(lambda z: +pi, maxdps=10000)
test_asymp(lambda z: +e, maxdps=10000)
test_asymp(lambda z: +ln2, maxdps=10000)
test_asymp(lambda z: +ln10, maxdps=10000)
test_asymp(lambda z: +phi, maxdps=10000)
test_asymp(lambda z: +catalan, maxdps=5000)
test_asymp(lambda z: +euler, maxdps=5000)
test_asymp(lambda z: +glaisher, maxdps=1000)
test_asymp(lambda z: +khinchin, maxdps=1000)
test_asymp(lambda z: +twinprime, maxdps=150)
test_asymp(lambda z: stieltjes(2), maxdps=150)
test_asymp(lambda z: +mertens, maxdps=150)
test_asymp(lambda z: +apery, maxdps=5000)
test_asymp(sqrt, maxdps=10000, huge_range=True)
test_asymp(cbrt, maxdps=5000, huge_range=True)
test_asymp(lambda z: root(z,4), maxdps=5000, huge_range=True)
test_asymp(lambda z: root(z,-5), maxdps=5000, huge_range=True)
test_asymp(exp, maxdps=5000, huge_range=True)
test_asymp(expm1, maxdps=1500)
test_asymp(ln, maxdps=5000, huge_range=True)
test_asymp(cosh, maxdps=5000)
test_asymp(sinh, maxdps=5000)
test_asymp(tanh, maxdps=1500)
test_asymp(sin, maxdps=5000, huge_range=True)
test_asymp(cos, maxdps=5000, huge_range=True)
test_asymp(tan, maxdps=1500)
test_asymp(agm, maxdps=1500, huge_range=True)
test_asymp(ellipk, maxdps=1500)
test_asymp(ellipe, maxdps=1500)
test_asymp(lambertw, huge_range=True)
test_asymp(lambda z: lambertw(z,-1))
test_asymp(lambda z: lambertw(z,1))
test_asymp(lambda z: lambertw(z,4))
test_asymp(gamma)
test_asymp(loggamma)  # huge_range=True ?
test_asymp(ei)
test_asymp(e1)
test_asymp(li, huge_range=True)
test_asymp(ci)
test_asymp(si)
test_asymp(chi)
test_asymp(shi)
test_asymp(erf)
test_asymp(erfc)
test_asymp(erfi)
test_asymp(lambda z: besselj(2, z))
test_asymp(lambda z: bessely(2, z))
test_asymp(lambda z: besseli(2, z))
test_asymp(lambda z: besselk(2, z))
test_asymp(lambda z: besselj(-2.25, z))
test_asymp(lambda z: bessely(-2.25, z))
test_asymp(lambda z: besseli(-2.25, z))
test_asymp(lambda z: besselk(-2.25, z))
test_asymp(airyai)
test_asymp(airybi)
test_asymp(lambda z: hyp0f1(a1, z))
test_asymp(lambda z: hyp1f1(a1, a2, z))
test_asymp(lambda z: hyp1f2(a1, a2, a3, z))
test_asymp(lambda z: hyp2f0(a1, a2, z))
test_asymp(lambda z: hyperu(a1, a2, z))
test_asymp(lambda z: hyp2f1(a1, a2, a3, z))
test_asymp(lambda z: hyp2f2(a1, a2, a3, a4, z))
test_asymp(lambda z: hyp2f3(a1, a2, a3, a4, a5, z))
test_asymp(lambda z: coulombf(a1, a2, z))
test_asymp(lambda z: coulombg(a1, a2, z))
test_asymp(lambda z: polylog(2,z))
test_asymp(lambda z: polylog(3,z))
test_asymp(lambda z: polylog(-2,z))
test_asymp(lambda z: expint(4, z))
test_asymp(lambda z: expint(-4, z))
test_asymp(lambda z: expint(2.25, z))
test_asymp(lambda z: gammainc(2.5, z, 5))
test_asymp(lambda z: gammainc(2.5, 5, z))
test_asymp(lambda z: hermite(3, z))
test_asymp(lambda z: hermite(2.5, z))
test_asymp(lambda z: legendre(3, z))
test_asymp(lambda z: legendre(4, z))
test_asymp(lambda z: legendre(2.5, z))
test_asymp(lambda z: legenp(a1, a2, z))
test_asymp(lambda z: legenq(a1, a2, z), maxdps=90)   # abnormally slow
test_asymp(lambda z: jtheta(1, z, 0.5))
test_asymp(lambda z: jtheta(2, z, 0.5))
test_asymp(lambda z: jtheta(3, z, 0.5))
test_asymp(lambda z: jtheta(4, z, 0.5))
test_asymp(lambda z: jtheta(1, z, 0.5, 1))
test_asymp(lambda z: jtheta(2, z, 0.5, 1))
test_asymp(lambda z: jtheta(3, z, 0.5, 1))
test_asymp(lambda z: jtheta(4, z, 0.5, 1))
test_asymp(barnesg, maxdps=90)
"""

def testit(line):
    if filt in line:
        print(line)
        t1 = clock()
        exec_(line, globals(), locals())
        t2 = clock()
        elapsed = t2-t1
        print("Time:", elapsed, "for", line, "(OK)")

if __name__ == '__main__':
    try:
        from multiprocessing import Pool
        mapf = Pool(None).map
        print("Running tests with multiprocessing")
    except ImportError:
        print("Not using multiprocessing")
        mapf = map
    t1 = clock()
    tasks = cases.splitlines()
    mapf(testit, tasks)
    t2 = clock()
    print("Cumulative wall time:", t2-t1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\sparse\test_dtype.py ===
import re
import warnings

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype


@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        ("int", 0),
        ("float", np.nan),
        ("bool", False),
        ("object", np.nan),
        ("datetime64[ns]", np.datetime64("NaT", "ns")),
        ("timedelta64[ns]", np.timedelta64("NaT", "ns")),
    ],
)
def test_inferred_dtype(dtype, fill_value):
    sparse_dtype = SparseDtype(dtype)
    result = sparse_dtype.fill_value
    if pd.isna(fill_value):
        assert pd.isna(result) and type(result) == type(fill_value)
    else:
        assert result == fill_value


def test_from_sparse_dtype():
    dtype = SparseDtype("float", 0)
    result = SparseDtype(dtype)
    assert result.fill_value == 0


def test_from_sparse_dtype_fill_value():
    dtype = SparseDtype("int", 1)
    result = SparseDtype(dtype, fill_value=2)
    expected = SparseDtype("int", 2)
    assert result == expected


@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        ("int", None),
        ("float", None),
        ("bool", None),
        ("object", None),
        ("datetime64[ns]", None),
        ("timedelta64[ns]", None),
        ("int", np.nan),
        ("float", 0),
    ],
)
def test_equal(dtype, fill_value):
    a = SparseDtype(dtype, fill_value)
    b = SparseDtype(dtype, fill_value)
    assert a == b
    assert b == a


def test_nans_equal():
    a = SparseDtype(float, float("nan"))
    b = SparseDtype(float, np.nan)
    assert a == b
    assert b == a


with warnings.catch_warnings():
    msg = "Allowing arbitrary scalar fill_value in SparseDtype is deprecated"
    warnings.filterwarnings("ignore", msg, category=FutureWarning)

    tups = [
        (SparseDtype("float64"), SparseDtype("float32")),
        (SparseDtype("float64"), SparseDtype("float64", 0)),
        (SparseDtype("float64"), SparseDtype("datetime64[ns]", np.nan)),
        (SparseDtype(int, pd.NaT), SparseDtype(float, pd.NaT)),
        (SparseDtype("float64"), np.dtype("float64")),
    ]


@pytest.mark.parametrize(
    "a, b",
    tups,
)
def test_not_equal(a, b):
    assert a != b


def test_construct_from_string_raises():
    with pytest.raises(
        TypeError, match="Cannot construct a 'SparseDtype' from 'not a dtype'"
    ):
        SparseDtype.construct_from_string("not a dtype")


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (SparseDtype(int), True),
        (SparseDtype(float), True),
        (SparseDtype(bool), True),
        (SparseDtype(object), False),
        (SparseDtype(str), False),
    ],
)
def test_is_numeric(dtype, expected):
    assert dtype._is_numeric is expected


def test_str_uses_object():
    result = SparseDtype(str).subtype
    assert result == np.dtype("object")


@pytest.mark.parametrize(
    "string, expected",
    [
        ("Sparse[float64]", SparseDtype(np.dtype("float64"))),
        ("Sparse[float32]", SparseDtype(np.dtype("float32"))),
        ("Sparse[int]", SparseDtype(np.dtype("int"))),
        ("Sparse[str]", SparseDtype(np.dtype("str"))),
        ("Sparse[datetime64[ns]]", SparseDtype(np.dtype("datetime64[ns]"))),
        ("Sparse", SparseDtype(np.dtype("float"), np.nan)),
    ],
)
def test_construct_from_string(string, expected):
    result = SparseDtype.construct_from_string(string)
    assert result == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (SparseDtype(float, 0.0), SparseDtype(np.dtype("float"), 0.0), True),
        (SparseDtype(int, 0), SparseDtype(int, 0), True),
        (SparseDtype(float, float("nan")), SparseDtype(float, np.nan), True),
        (SparseDtype(float, 0), SparseDtype(float, np.nan), False),
        (SparseDtype(int, 0.0), SparseDtype(float, 0.0), False),
    ],
)
def test_hash_equal(a, b, expected):
    result = a == b
    assert result is expected

    result = hash(a) == hash(b)
    assert result is expected


@pytest.mark.parametrize(
    "string, expected",
    [
        ("Sparse[int]", "int"),
        ("Sparse[int, 0]", "int"),
        ("Sparse[int64]", "int64"),
        ("Sparse[int64, 0]", "int64"),
        ("Sparse[datetime64[ns], 0]", "datetime64[ns]"),
    ],
)
def test_parse_subtype(string, expected):
    subtype, _ = SparseDtype._parse_subtype(string)
    assert subtype == expected


@pytest.mark.parametrize(
    "string", ["Sparse[int, 1]", "Sparse[float, 0.0]", "Sparse[bool, True]"]
)
def test_construct_from_string_fill_value_raises(string):
    with pytest.raises(TypeError, match="fill_value in the string is not"):
        SparseDtype.construct_from_string(string)


@pytest.mark.parametrize(
    "original, dtype, expected",
    [
        (SparseDtype(int, 0), float, SparseDtype(float, 0.0)),
        (SparseDtype(int, 1), float, SparseDtype(float, 1.0)),
        (SparseDtype(int, 1), np.str_, SparseDtype(object, "1")),
        (SparseDtype(float, 1.5), int, SparseDtype(int, 1)),
    ],
)
def test_update_dtype(original, dtype, expected):
    result = original.update_dtype(dtype)
    assert result == expected


@pytest.mark.parametrize(
    "original, dtype, expected_error_msg",
    [
        (
            SparseDtype(float, np.nan),
            int,
            re.escape("Cannot convert non-finite values (NA or inf) to integer"),
        ),
        (
            SparseDtype(str, "abc"),
            int,
            r"invalid literal for int\(\) with base 10: ('abc'|np\.str_\('abc'\))",
        ),
    ],
)
def test_update_dtype_raises(original, dtype, expected_error_msg):
    with pytest.raises(ValueError, match=expected_error_msg):
        original.update_dtype(dtype)


def test_repr():
    # GH-34352
    result = str(SparseDtype("int64", fill_value=0))
    expected = "Sparse[int64, 0]"
    assert result == expected

    result = str(SparseDtype(object, fill_value="0"))
    expected = "Sparse[object, '0']"
    assert result == expected


def test_sparse_dtype_subtype_must_be_numpy_dtype():
    # GH#53160
    msg = "SparseDtype subtype must be a numpy dtype"
    with pytest.raises(TypeError, match=msg):
        SparseDtype("category", fill_value="c")