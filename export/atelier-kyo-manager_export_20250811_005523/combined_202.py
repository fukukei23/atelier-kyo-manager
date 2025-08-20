
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_ewm.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm


def test_doc_string():
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df
    df.ewm(com=0.5).mean()


def test_constructor(frame_or_series):
    c = frame_or_series(range(5)).ewm

    # valid
    c(com=0.5)
    c(span=1.5)
    c(alpha=0.5)
    c(halflife=0.75)
    c(com=0.5, span=None)
    c(alpha=0.5, com=None)
    c(halflife=0.75, alpha=None)

    # not valid: mutually exclusive
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        c(com=0.5, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        c(span=1.5, halflife=0.75)
    with pytest.raises(ValueError, match=msg):
        c(alpha=0.5, span=1.5)

    # not valid: com < 0
    msg = "comass must satisfy: comass >= 0"
    with pytest.raises(ValueError, match=msg):
        c(com=-0.5)

    # not valid: span < 1
    msg = "span must satisfy: span >= 1"
    with pytest.raises(ValueError, match=msg):
        c(span=0.5)

    # not valid: halflife <= 0
    msg = "halflife must satisfy: halflife > 0"
    with pytest.raises(ValueError, match=msg):
        c(halflife=0)

    # not valid: alpha <= 0 or alpha > 1
    msg = "alpha must satisfy: 0 < alpha <= 1"
    for alpha in (-0.5, 1.5):
        with pytest.raises(ValueError, match=msg):
            c(alpha=alpha)


def test_ewma_times_not_datetime_type():
    msg = r"times must be datetime64 dtype."
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(times=np.arange(5))


def test_ewma_times_not_same_length():
    msg = "times must be the same length as the object."
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(times=np.arange(4).astype("datetime64[ns]"))


def test_ewma_halflife_not_correct_type():
    msg = "halflife must be a timedelta convertible object"
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(halflife=1, times=np.arange(5).astype("datetime64[ns]"))


def test_ewma_halflife_without_times(halflife_with_times):
    msg = "halflife can only be a timedelta convertible argument if times is not None."
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(halflife=halflife_with_times)


@pytest.mark.parametrize(
    "times",
    [
        np.arange(10).astype("datetime64[D]").astype("datetime64[ns]"),
        date_range("2000", freq="D", periods=10),
        date_range("2000", freq="D", periods=10).tz_localize("UTC"),
    ],
)
@pytest.mark.parametrize("min_periods", [0, 2])
def test_ewma_with_times_equal_spacing(halflife_with_times, times, min_periods):
    halflife = halflife_with_times
    data = np.arange(10.0)
    data[::2] = np.nan
    df = DataFrame({"A": data})
    result = df.ewm(halflife=halflife, min_periods=min_periods, times=times).mean()
    expected = df.ewm(halflife=1.0, min_periods=min_periods).mean()
    tm.assert_frame_equal(result, expected)


def test_ewma_with_times_variable_spacing(tz_aware_fixture, unit):
    tz = tz_aware_fixture
    halflife = "23 days"
    times = (
        DatetimeIndex(["2020-01-01", "2020-01-10T00:04:05", "2020-02-23T05:00:23"])
        .tz_localize(tz)
        .as_unit(unit)
    )
    data = np.arange(3)
    df = DataFrame(data)
    result = df.ewm(halflife=halflife, times=times).mean()
    expected = DataFrame([0.0, 0.5674161888241773, 1.545239952073459])
    tm.assert_frame_equal(result, expected)


def test_ewm_with_nat_raises(halflife_with_times):
    # GH#38535
    ser = Series(range(1))
    times = DatetimeIndex(["NaT"])
    with pytest.raises(ValueError, match="Cannot convert NaT values to integer"):
        ser.ewm(com=0.1, halflife=halflife_with_times, times=times)


def test_ewm_with_times_getitem(halflife_with_times):
    # GH 40164
    halflife = halflife_with_times
    data = np.arange(10.0)
    data[::2] = np.nan
    times = date_range("2000", freq="D", periods=10)
    df = DataFrame({"A": data, "B": data})
    result = df.ewm(halflife=halflife, times=times)["A"].mean()
    expected = df.ewm(halflife=1.0)["A"].mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("arg", ["com", "halflife", "span", "alpha"])
def test_ewm_getitem_attributes_retained(arg, adjust, ignore_na):
    # GH 40164
    kwargs = {arg: 1, "adjust": adjust, "ignore_na": ignore_na}
    ewm = DataFrame({"A": range(1), "B": range(1)}).ewm(**kwargs)
    expected = {attr: getattr(ewm, attr) for attr in ewm._attributes}
    ewm_slice = ewm["A"]
    result = {attr: getattr(ewm, attr) for attr in ewm_slice._attributes}
    assert result == expected


def test_ewma_times_adjust_false_raises():
    # GH 40098
    with pytest.raises(
        NotImplementedError, match="times is not supported with adjust=False."
    ):
        Series(range(1)).ewm(
            0.1, adjust=False, times=date_range("2000", freq="D", periods=1)
        )


@pytest.mark.parametrize(
    "func, expected",
    [
        [
            "mean",
            DataFrame(
                {
                    0: range(5),
                    1: range(4, 9),
                    2: [7.428571, 9, 10.571429, 12.142857, 13.714286],
                },
                dtype=float,
            ),
        ],
        [
            "std",
            DataFrame(
                {
                    0: [np.nan] * 5,
                    1: [4.242641] * 5,
                    2: [4.6291, 5.196152, 5.781745, 6.380775, 6.989788],
                }
            ),
        ],
        [
            "var",
            DataFrame(
                {
                    0: [np.nan] * 5,
                    1: [18.0] * 5,
                    2: [21.428571, 27, 33.428571, 40.714286, 48.857143],
                }
            ),
        ],
    ],
)
def test_float_dtype_ewma(func, expected, float_numpy_dtype):
    # GH#42452

    df = DataFrame(
        {0: range(5), 1: range(6, 11), 2: range(10, 20, 2)}, dtype=float_numpy_dtype
    )
    msg = "Support for axis=1 in DataFrame.ewm is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        e = df.ewm(alpha=0.5, axis=1)
    result = getattr(e, func)()

    tm.assert_frame_equal(result, expected)


def test_times_string_col_raises():
    # GH 43265
    df = DataFrame(
        {"A": np.arange(10.0), "time_col": date_range("2000", freq="D", periods=10)}
    )
    with pytest.raises(ValueError, match="times must be datetime64"):
        df.ewm(halflife="1 day", min_periods=0, times="time_col")


def test_ewm_sum_adjust_false_notimplemented():
    data = Series(range(1)).ewm(com=1, adjust=False)
    with pytest.raises(NotImplementedError, match="sum is not"):
        data.sum()


@pytest.mark.parametrize(
    "expected_data, ignore",
    [[[10.0, 5.0, 2.5, 11.25], False], [[10.0, 5.0, 5.0, 12.5], True]],
)
def test_ewm_sum(expected_data, ignore):
    # xref from Numbagg tests
    # https://github.com/numbagg/numbagg/blob/v0.2.1/numbagg/test/test_moving.py#L50
    data = Series([10, 0, np.nan, 10])
    result = data.ewm(alpha=0.5, ignore_na=ignore).sum()
    expected = Series(expected_data)
    tm.assert_series_equal(result, expected)


def test_ewma_adjust():
    vals = Series(np.zeros(1000))
    vals[5] = 1
    result = vals.ewm(span=100, adjust=False).mean().sum()
    assert np.abs(result - 1) < 1e-2


def test_ewma_cases(adjust, ignore_na):
    # try adjust/ignore_na args matrix

    s = Series([1.0, 2.0, 4.0, 8.0])

    if adjust:
        expected = Series([1.0, 1.6, 2.736842, 4.923077])
    else:
        expected = Series([1.0, 1.333333, 2.222222, 4.148148])

    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()
    tm.assert_series_equal(result, expected)


def test_ewma_nan_handling():
    s = Series([1.0] + [np.nan] * 5 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([1.0] * len(s)))

    s = Series([np.nan] * 2 + [1.0] + [np.nan] * 2 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([np.nan] * 2 + [1.0] * 4))


@pytest.mark.parametrize(
    "s, adjust, ignore_na, w",
    [
        (
            Series([np.nan, 1.0, 101.0]),
            True,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), 1.0],
        ),
        (
            Series([np.nan, 1.0, 101.0]),
            True,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), 1.0],
        ),
        (
            Series([np.nan, 1.0, 101.0]),
            False,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([np.nan, 1.0, 101.0]),
            False,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            True,
            False,
            [(1.0 - (1.0 / (1.0 + 2.0))) ** 2, np.nan, 1.0],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            True,
            True,
            [(1.0 - (1.0 / (1.0 + 2.0))), np.nan, 1.0],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            False,
            False,
            [(1.0 - (1.0 / (1.0 + 2.0))) ** 2, np.nan, (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            False,
            True,
            [(1.0 - (1.0 / (1.0 + 2.0))), np.nan, (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            True,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))) ** 3, np.nan, np.nan, 1.0, np.nan],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            True,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), np.nan, np.nan, 1.0, np.nan],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            False,
            False,
            [
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                np.nan,
                (1.0 / (1.0 + 2.0)),
                np.nan,
            ],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            False,
            True,
            [
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                np.nan,
                np.nan,
                (1.0 / (1.0 + 2.0)),
                np.nan,
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            True,
            False,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                1.0,
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            True,
            True,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 2,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                1.0,
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            False,
            False,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) * (1.0 / (1.0 + 2.0)),
                (1.0 / (1.0 + 2.0))
                * ((1.0 - (1.0 / (1.0 + 2.0))) ** 2 + (1.0 / (1.0 + 2.0))),
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            False,
            True,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 2,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) * (1.0 / (1.0 + 2.0)),
                (1.0 / (1.0 + 2.0)),
            ],
        ),
    ],
)
def test_ewma_nan_handling_cases(s, adjust, ignore_na, w):
    # GH 7603
    expected = (s.multiply(w).cumsum() / Series(w).cumsum()).ffill()
    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()

    tm.assert_series_equal(result, expected)
    if ignore_na is False:
        # check that ignore_na defaults to False
        result = s.ewm(com=2.0, adjust=adjust).mean()
        tm.assert_series_equal(result, expected)


def test_ewm_alpha():
    # GH 10789
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan

    s = Series(arr)
    a = s.ewm(alpha=0.61722699889169674).mean()
    b = s.ewm(com=0.62014947789973052).mean()
    c = s.ewm(span=2.240298955799461).mean()
    d = s.ewm(halflife=0.721792864318).mean()
    tm.assert_series_equal(a, b)
    tm.assert_series_equal(a, c)
    tm.assert_series_equal(a, d)


def test_ewm_domain_checks():
    # GH 12492
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan

    s = Series(arr)
    msg = "comass must satisfy: comass >= 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=-0.1)
    s.ewm(com=0.0)
    s.ewm(com=0.1)

    msg = "span must satisfy: span >= 1"
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.9)
    s.ewm(span=1.0)
    s.ewm(span=1.1)

    msg = "halflife must satisfy: halflife > 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=0.0)
    s.ewm(halflife=0.1)

    msg = "alpha must satisfy: 0 < alpha <= 1"
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=0.0)
    s.ewm(alpha=0.1)
    s.ewm(alpha=1.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=1.1)


@pytest.mark.parametrize("method", ["mean", "std", "var"])
def test_ew_empty_series(method):
    vals = Series([], dtype=np.float64)

    ewm = vals.ewm(3)
    result = getattr(ewm, method)()
    tm.assert_almost_equal(result, vals)


@pytest.mark.parametrize("min_periods", [0, 1])
@pytest.mark.parametrize("name", ["mean", "var", "std"])
def test_ew_min_periods(min_periods, name):
    # excluding NaNs correctly
    arr = np.random.default_rng(2).standard_normal(50)
    arr[:10] = np.nan
    arr[-10:] = np.nan
    s = Series(arr)

    # check min_periods
    # GH 7898
    result = getattr(s.ewm(com=50, min_periods=2), name)()
    assert result[:11].isna().all()
    assert not result[11:].isna().any()

    result = getattr(s.ewm(com=50, min_periods=min_periods), name)()
    if name == "mean":
        assert result[:10].isna().all()
        assert not result[10:].isna().any()
    else:
        # ewm.std, ewm.var (with bias=False) require at least
        # two values
        assert result[:11].isna().all()
        assert not result[11:].isna().any()

    # check series of length 0
    result = getattr(Series(dtype=object).ewm(com=50, min_periods=min_periods), name)()
    tm.assert_series_equal(result, Series(dtype="float64"))

    # check series of length 1
    result = getattr(Series([1.0]).ewm(50, min_periods=min_periods), name)()
    if name == "mean":
        tm.assert_series_equal(result, Series([1.0]))
    else:
        # ewm.std, ewm.var with bias=False require at least
        # two values
        tm.assert_series_equal(result, Series([np.nan]))

    # pass in ints
    result2 = getattr(Series(np.arange(50)).ewm(span=10), name)()
    assert result2.dtype == np.float64


@pytest.mark.parametrize("name", ["cov", "corr"])
def test_ewm_corr_cov(name):
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)

    A[:10] = np.nan
    B.iloc[-10:] = np.nan

    result = getattr(A.ewm(com=20, min_periods=5), name)(B)
    assert np.isnan(result.values[:14]).all()
    assert not np.isnan(result.values[14:]).any()


@pytest.mark.parametrize("min_periods", [0, 1, 2])
@pytest.mark.parametrize("name", ["cov", "corr"])
def test_ewm_corr_cov_min_periods(name, min_periods):
    # GH 7898
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)

    A[:10] = np.nan
    B.iloc[-10:] = np.nan

    result = getattr(A.ewm(com=20, min_periods=min_periods), name)(B)
    # binary functions (ewmcov, ewmcorr) with bias=False require at
    # least two values
    assert np.isnan(result.values[:11]).all()
    assert not np.isnan(result.values[11:]).any()

    # check series of length 0
    empty = Series([], dtype=np.float64)
    result = getattr(empty.ewm(com=50, min_periods=min_periods), name)(empty)
    tm.assert_series_equal(result, empty)

    # check series of length 1
    result = getattr(Series([1.0]).ewm(com=50, min_periods=min_periods), name)(
        Series([1.0])
    )
    tm.assert_series_equal(result, Series([np.nan]))


@pytest.mark.parametrize("name", ["cov", "corr"])
def test_different_input_array_raise_exception(name):
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    A[:10] = np.nan

    msg = "other must be a DataFrame or Series"
    # exception raised is Exception
    with pytest.raises(ValueError, match=msg):
        getattr(A.ewm(com=20, min_periods=5), name)(
            np.random.default_rng(2).standard_normal(50)
        )


@pytest.mark.parametrize("name", ["var", "std", "mean"])
def test_ewma_series(series, name):
    series_result = getattr(series.ewm(com=10), name)()
    assert isinstance(series_result, Series)


@pytest.mark.parametrize("name", ["var", "std", "mean"])
def test_ewma_frame(frame, name):
    frame_result = getattr(frame.ewm(com=10), name)()
    assert isinstance(frame_result, DataFrame)


def test_ewma_span_com_args(series):
    A = series.ewm(com=9.5).mean()
    B = series.ewm(span=20).mean()
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, span=20)

    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        series.ewm().mean()


def test_ewma_halflife_arg(series):
    A = series.ewm(com=13.932726172912965).mean()
    B = series.ewm(halflife=10.0).mean()
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        series.ewm(span=20, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, span=20, halflife=50)
    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        series.ewm()


def test_ewm_alpha_arg(series):
    # GH 10789
    s = series
    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        s.ewm()

    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=10.0, alpha=0.5)


@pytest.mark.parametrize("func", ["cov", "corr"])
def test_ewm_pairwise_cov_corr(func, frame):
    result = getattr(frame.ewm(span=10, min_periods=5), func)()
    result = result.loc[(slice(None), 1), 5]
    result.index = result.index.droplevel(1)
    expected = getattr(frame[1].ewm(span=10, min_periods=5), func)(frame[5])
    tm.assert_series_equal(result, expected, check_names=False)


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # GH#46560
    kernel = arithmetic_win_operators
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    ewm = df.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel, None)
    if op is not None:
        result = op(numeric_only=numeric_only)

        columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        assert list(expected.columns) == columns

        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # GH#46560
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    arg = (df,) if use_arg else ()
    ewm = df.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel)
    result = op(*arg, numeric_only=numeric_only)

    # Compare result to op using float dtypes, dropping c when numeric_only is True
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    ewm2 = df2.ewm(span=2, min_periods=1)
    op2 = getattr(ewm2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # GH#46560
    kernel = arithmetic_win_operators
    ser = Series([1], dtype=dtype)
    ewm = ser.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel, None)
    if op is None:
        # Nothing to test
        pytest.skip("No op to test")
    if numeric_only and dtype is object:
        msg = f"ExponentialMovingWindow.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        result = op(numeric_only=numeric_only)
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # GH#46560
    ser = Series([1, 2, 3], dtype=dtype)
    arg = (ser,) if use_arg else ()
    ewm = ser.ewm(span=2, min_periods=1)
    op = getattr(ewm, kernel)
    if numeric_only and dtype is object:
        msg = f"ExponentialMovingWindow.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result = op(*arg, numeric_only=numeric_only)

        ser2 = ser.astype(float)
        arg2 = (ser2,) if use_arg else ()
        ewm2 = ser2.ewm(span=2, min_periods=1)
        op2 = getattr(ewm2, kernel)
        expected = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\test_extract.py ===
from datetime import datetime
import re

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import ArrowDtype

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)


def test_extract_expand_kwarg_wrong_type_raises(any_string_dtype):
    # TODO: should this raise TypeError
    values = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    with pytest.raises(ValueError, match="expand must be True or False"):
        values.str.extract(".*(BAD[_]+).*(BAD)", expand=None)


def test_extract_expand_kwarg(any_string_dtype):
    s = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    expected = DataFrame(["BAD__", np.nan, np.nan], dtype=any_string_dtype)

    result = s.str.extract(".*(BAD[_]+).*")
    tm.assert_frame_equal(result, expected)

    result = s.str.extract(".*(BAD[_]+).*", expand=True)
    tm.assert_frame_equal(result, expected)

    expected = DataFrame(
        [["BAD__", "BAD"], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    result = s.str.extract(".*(BAD[_]+).*(BAD)", expand=False)
    tm.assert_frame_equal(result, expected)


def test_extract_expand_False_mixed_object():
    ser = Series(
        ["aBAD_BAD", np.nan, "BAD_b_BAD", True, datetime.today(), "foo", None, 1, 2.0]
    )

    # two groups
    result = ser.str.extract(".*(BAD[_]+).*(BAD)", expand=False)
    er = [np.nan, np.nan]  # empty row
    expected = DataFrame(
        [["BAD_", "BAD"], er, ["BAD_", "BAD"], er, er, er, er, er, er], dtype=object
    )
    tm.assert_frame_equal(result, expected)

    # single group
    result = ser.str.extract(".*(BAD[_]+).*BAD", expand=False)
    expected = Series(
        ["BAD_", np.nan, "BAD_", np.nan, np.nan, np.nan, None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_extract_expand_index_raises():
    # GH9980
    # Index only works with one regex group since
    # multi-group would expand to a frame
    idx = Index(["A1", "A2", "A3", "A4", "B5"])
    msg = "only one regex group is supported with Index"
    with pytest.raises(ValueError, match=msg):
        idx.str.extract("([AB])([123])", expand=False)


def test_extract_expand_no_capture_groups_raises(index_or_series, any_string_dtype):
    s_or_idx = index_or_series(["A1", "B2", "C3"], dtype=any_string_dtype)
    msg = "pattern contains no capture groups"

    # no groups
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("[ABC][123]", expand=False)

    # only non-capturing groups
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("(?:[AB]).*", expand=False)


def test_extract_expand_single_capture_group(index_or_series, any_string_dtype):
    # single group renames series/index properly
    s_or_idx = index_or_series(["A1", "A2"], dtype=any_string_dtype)
    result = s_or_idx.str.extract(r"(?P<uno>A)\d", expand=False)

    expected = index_or_series(["A", "A"], name="uno", dtype=any_string_dtype)
    if index_or_series == Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)


def test_extract_expand_capture_groups(any_string_dtype):
    s = Series(["A1", "B2", "C3"], dtype=any_string_dtype)
    # one group, no matches
    result = s.str.extract("(_)", expand=False)
    expected = Series([np.nan, np.nan, np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two groups, no matches
    result = s.str.extract("(_)(_)", expand=False)
    expected = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one group, some matches
    result = s.str.extract("([AB])[123]", expand=False)
    expected = Series(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two groups, some matches
    result = s.str.extract("([AB])([123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one named group
    result = s.str.extract("(?P<letter>[AB])", expand=False)
    expected = Series(["A", "B", np.nan], name="letter", dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two named groups
    result = s.str.extract("(?P<letter>[AB])(?P<number>[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # mix named and unnamed groups
    result = s.str.extract("([AB])(?P<number>[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=[0, "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group, one non-capturing group
    result = s.str.extract("([AB])(?:[123])", expand=False)
    expected = Series(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    # two normal groups, one non-capturing group
    s = Series(["A11", "B22", "C33"], dtype=any_string_dtype)
    result = s.str.extract("([AB])([123])(?:[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one optional group followed by one normal group
    s = Series(["A1", "B2", "3"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[AB])?(?P<number>[123])", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, "3"]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group followed by one optional group
    s = Series(["A1", "B2", "C"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[ABC])(?P<number>[123])?", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_expand_capture_groups_index(index, any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/6348
    # not passing index to the extractor
    data = ["A1", "B2", "C"]

    if len(index) == 0:
        pytest.skip("Test requires len(index) > 0")
    while len(index) < len(data):
        index = index.repeat(2)

    index = index[: len(data)]
    ser = Series(data, index=index, dtype=any_string_dtype)

    result = ser.str.extract(r"(\d)", expand=False)
    expected = Series(["1", "2", np.nan], index=index, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.extract(r"(?P<letter>\D)(?P<number>\d)?", expand=False)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        index=index,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_single_series_name_is_preserved(any_string_dtype):
    s = Series(["a3", "b3", "c2"], name="bob", dtype=any_string_dtype)
    result = s.str.extract(r"(?P<sue>[a-z])", expand=False)
    expected = Series(["a", "b", "c"], name="sue", dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_extract_expand_True(any_string_dtype):
    # Contains tests like those in test_match and some others.
    s = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)

    result = s.str.extract(".*(BAD[_]+).*(BAD)", expand=True)
    expected = DataFrame(
        [["BAD__", "BAD"], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_extract_expand_True_mixed_object():
    er = [np.nan, np.nan]  # empty row
    mixed = Series(
        [
            "aBAD_BAD",
            np.nan,
            "BAD_b_BAD",
            True,
            datetime.today(),
            "foo",
            None,
            1,
            2.0,
        ]
    )

    result = mixed.str.extract(".*(BAD[_]+).*(BAD)", expand=True)
    expected = DataFrame(
        [["BAD_", "BAD"], er, ["BAD_", "BAD"], er, er, er, er, er, er], dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_extract_expand_True_single_capture_group_raises(
    index_or_series, any_string_dtype
):
    # these should work for both Series and Index
    # no groups
    s_or_idx = index_or_series(["A1", "B2", "C3"], dtype=any_string_dtype)
    msg = "pattern contains no capture groups"
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("[ABC][123]", expand=True)

    # only non-capturing groups
    with pytest.raises(ValueError, match=msg):
        s_or_idx.str.extract("(?:[AB]).*", expand=True)


def test_extract_expand_True_single_capture_group(index_or_series, any_string_dtype):
    # single group renames series/index properly
    s_or_idx = index_or_series(["A1", "A2"], dtype=any_string_dtype)
    result = s_or_idx.str.extract(r"(?P<uno>A)\d", expand=True)
    expected = DataFrame({"uno": ["A", "A"]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("name", [None, "series_name"])
def test_extract_series(name, any_string_dtype):
    # extract should give the same result whether or not the series has a name.
    s = Series(["A1", "B2", "C3"], name=name, dtype=any_string_dtype)

    # one group, no matches
    result = s.str.extract("(_)", expand=True)
    expected = DataFrame([np.nan, np.nan, np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two groups, no matches
    result = s.str.extract("(_)(_)", expand=True)
    expected = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one group, some matches
    result = s.str.extract("([AB])[123]", expand=True)
    expected = DataFrame(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two groups, some matches
    result = s.str.extract("([AB])([123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one named group
    result = s.str.extract("(?P<letter>[AB])", expand=True)
    expected = DataFrame({"letter": ["A", "B", np.nan]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two named groups
    result = s.str.extract("(?P<letter>[AB])(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # mix named and unnamed groups
    result = s.str.extract("([AB])(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]],
        columns=[0, "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group, one non-capturing group
    result = s.str.extract("([AB])(?:[123])", expand=True)
    expected = DataFrame(["A", "B", np.nan], dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)


def test_extract_optional_groups(any_string_dtype):
    # two normal groups, one non-capturing group
    s = Series(["A11", "B22", "C33"], dtype=any_string_dtype)
    result = s.str.extract("([AB])([123])(?:[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one optional group followed by one normal group
    s = Series(["A1", "B2", "3"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[AB])?(?P<number>[123])", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], [np.nan, "3"]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # one normal group followed by one optional group
    s = Series(["A1", "B2", "C"], dtype=any_string_dtype)
    result = s.str.extract("(?P<letter>[ABC])(?P<number>[123])?", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_dataframe_capture_groups_index(index, any_string_dtype):
    # GH6348
    # not passing index to the extractor

    data = ["A1", "B2", "C"]

    if len(index) < len(data):
        pytest.skip(f"Index needs more than {len(data)} values")

    index = index[: len(data)]
    s = Series(data, index=index, dtype=any_string_dtype)

    result = s.str.extract(r"(\d)", expand=True)
    expected = DataFrame(["1", "2", np.nan], index=index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    result = s.str.extract(r"(?P<letter>\D)(?P<number>\d)?", expand=True)
    expected = DataFrame(
        [["A", "1"], ["B", "2"], ["C", np.nan]],
        columns=["letter", "number"],
        index=index,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extract_single_group_returns_frame(any_string_dtype):
    # GH11386 extract should always return DataFrame, even when
    # there is only one group. Prior to v0.18.0, extract returned
    # Series when there was only one group in the regex.
    s = Series(["a3", "b3", "c2"], name="series_name", dtype=any_string_dtype)
    result = s.str.extract(r"(?P<letter>[a-z])", expand=True)
    expected = DataFrame({"letter": ["a", "b", "c"]}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)


def test_extractall(any_string_dtype):
    data = [
        "dave@google.com",
        "tdhock5@gmail.com",
        "maudelaperriere@gmail.com",
        "rob@gmail.com some text steve@gmail.com",
        "a@b.com some text c@d.com and e@f.com",
        np.nan,
        "",
    ]
    expected_tuples = [
        ("dave", "google", "com"),
        ("tdhock5", "gmail", "com"),
        ("maudelaperriere", "gmail", "com"),
        ("rob", "gmail", "com"),
        ("steve", "gmail", "com"),
        ("a", "b", "com"),
        ("c", "d", "com"),
        ("e", "f", "com"),
    ]
    pat = r"""
    (?P<user>[a-z0-9]+)
    @
    (?P<domain>[a-z]+)
    \.
    (?P<tld>[a-z]{2,4})
    """
    expected_columns = ["user", "domain", "tld"]
    s = Series(data, dtype=any_string_dtype)
    # extractall should return a DataFrame with one row for each match, indexed by the
    # subject from which the match came.
    expected_index = MultiIndex.from_tuples(
        [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)],
        names=(None, "match"),
    )
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)

    # The index of the input Series should be used to construct the index of the output
    # DataFrame:
    mi = MultiIndex.from_tuples(
        [
            ("single", "Dave"),
            ("single", "Toby"),
            ("single", "Maude"),
            ("multiple", "robAndSteve"),
            ("multiple", "abcdef"),
            ("none", "missing"),
            ("none", "empty"),
        ]
    )
    s = Series(data, index=mi, dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples(
        [
            ("single", "Dave", 0),
            ("single", "Toby", 0),
            ("single", "Maude", 0),
            ("multiple", "robAndSteve", 0),
            ("multiple", "robAndSteve", 1),
            ("multiple", "abcdef", 0),
            ("multiple", "abcdef", 1),
            ("multiple", "abcdef", 2),
        ],
        names=(None, None, "match"),
    )
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)

    # MultiIndexed subject with names.
    s = Series(data, index=mi, dtype=any_string_dtype)
    s.index.names = ("matches", "description")
    expected_index.names = ("matches", "description", "match")
    expected = DataFrame(
        expected_tuples, expected_index, expected_columns, dtype=any_string_dtype
    )
    result = s.str.extractall(pat, flags=re.VERBOSE)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "pat,expected_names",
    [
        # optional groups.
        ("(?P<letter>[AB])?(?P<number>[123])", ["letter", "number"]),
        # only one of two groups has a name.
        ("([AB])?(?P<number>[123])", [0, "number"]),
    ],
)
def test_extractall_column_names(pat, expected_names, any_string_dtype):
    s = Series(["", "A1", "32"], dtype=any_string_dtype)

    result = s.str.extractall(pat)
    expected = DataFrame(
        [("A", "1"), (np.nan, "3"), (np.nan, "2")],
        index=MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)], names=(None, "match")),
        columns=expected_names,
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_single_group(any_string_dtype):
    s = Series(["a3", "b3", "d4c2"], name="series_name", dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples(
        [(0, 0), (1, 0), (2, 0), (2, 1)], names=(None, "match")
    )

    # extractall(one named group) returns DataFrame with one named column.
    result = s.str.extractall(r"(?P<letter>[a-z])")
    expected = DataFrame(
        {"letter": ["a", "b", "d", "c"]}, index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # extractall(one un-named group) returns DataFrame with one un-named column.
    result = s.str.extractall(r"([a-z])")
    expected = DataFrame(
        ["a", "b", "d", "c"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_single_group_with_quantifier(any_string_dtype):
    # GH#13382
    # extractall(one un-named group with quantifier) returns DataFrame with one un-named
    # column.
    s = Series(["ab3", "abc3", "d4cd2"], name="series_name", dtype=any_string_dtype)
    result = s.str.extractall(r"([a-z]+)")
    expected = DataFrame(
        ["ab", "abc", "d", "cd"],
        index=MultiIndex.from_tuples(
            [(0, 0), (1, 0), (2, 0), (2, 1)], names=(None, "match")
        ),
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, names",
    [
        ([], (None,)),
        ([], ("i1",)),
        ([], (None, "i2")),
        ([], ("i1", "i2")),
        (["a3", "b3", "d4c2"], (None,)),
        (["a3", "b3", "d4c2"], ("i1", "i2")),
        (["a3", "b3", "d4c2"], (None, "i2")),
        (["a3", "b3", "d4c2"], ("i1", "i2")),
    ],
)
def test_extractall_no_matches(data, names, any_string_dtype):
    # GH19075 extractall with no matches should return a valid MultiIndex
    n = len(data)
    if len(names) == 1:
        index = Index(range(n), name=names[0])
    else:
        tuples = (tuple([i] * (n - 1)) for i in range(n))
        index = MultiIndex.from_tuples(tuples, names=names)
    s = Series(data, name="series_name", index=index, dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples([], names=(names + ("match",)))

    # one un-named group.
    result = s.str.extractall("(z)")
    expected = DataFrame(columns=[0], index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # two un-named groups.
    result = s.str.extractall("(z)(z)")
    expected = DataFrame(columns=[0, 1], index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)

    # one named group.
    result = s.str.extractall("(?P<first>z)")
    expected = DataFrame(
        columns=["first"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # two named groups.
    result = s.str.extractall("(?P<first>z)(?P<second>z)")
    expected = DataFrame(
        columns=["first", "second"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # one named, one un-named.
    result = s.str.extractall("(z)(?P<second>z)")
    expected = DataFrame(
        columns=[0, "second"], index=expected_index, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_stringindex(any_string_dtype):
    s = Series(["a1a2", "b1", "c1"], name="xxx", dtype=any_string_dtype)
    result = s.str.extractall(r"[ab](?P<digit>\d)")
    expected = DataFrame(
        {"digit": ["1", "2", "1"]},
        index=MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0)], names=[None, "match"]),
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    # index should return the same result as the default index without name thus
    # index.name doesn't affect to the result
    if any_string_dtype == "object":
        for idx in [
            Index(["a1a2", "b1", "c1"], dtype=object),
            Index(["a1a2", "b1", "c1"], name="xxx", dtype=object),
        ]:
            result = idx.str.extractall(r"[ab](?P<digit>\d)")
            tm.assert_frame_equal(result, expected)

    s = Series(
        ["a1a2", "b1", "c1"],
        name="s_name",
        index=Index(["XX", "yy", "zz"], name="idx_name"),
        dtype=any_string_dtype,
    )
    result = s.str.extractall(r"[ab](?P<digit>\d)")
    expected = DataFrame(
        {"digit": ["1", "2", "1"]},
        index=MultiIndex.from_tuples(
            [("XX", 0), ("XX", 1), ("yy", 0)], names=["idx_name", "match"]
        ),
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_extractall_no_capture_groups_raises(any_string_dtype):
    # Does not make sense to use extractall with a regex that has no capture groups.
    # (it returns DataFrame with one column for each capture group)
    s = Series(["a3", "b3", "d4c2"], name="series_name", dtype=any_string_dtype)
    with pytest.raises(ValueError, match="no capture groups"):
        s.str.extractall(r"[a-z]")


def test_extract_index_one_two_groups():
    s = Series(["a3", "b3", "d4c2"], index=["A3", "B3", "D4"], name="series_name")
    r = s.index.str.extract(r"([A-Z])", expand=True)
    e = DataFrame(["A", "B", "D"])
    tm.assert_frame_equal(r, e)

    # Prior to v0.18.0, index.str.extract(regex with one group)
    # returned Index. With more than one group, extract raised an
    # error (GH9980). Now extract always returns DataFrame.
    r = s.index.str.extract(r"(?P<letter>[A-Z])(?P<digit>[0-9])", expand=True)
    e_list = [("A", "3"), ("B", "3"), ("D", "4")]
    e = DataFrame(e_list, columns=["letter", "digit"])
    tm.assert_frame_equal(r, e)


def test_extractall_same_as_extract(any_string_dtype):
    s = Series(["a3", "b3", "c2"], name="series_name", dtype=any_string_dtype)

    pattern_two_noname = r"([a-z])([0-9])"
    extract_two_noname = s.str.extract(pattern_two_noname, expand=True)
    has_multi_index = s.str.extractall(pattern_two_noname)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_noname, no_multi_index)

    pattern_two_named = r"(?P<letter>[a-z])(?P<digit>[0-9])"
    extract_two_named = s.str.extract(pattern_two_named, expand=True)
    has_multi_index = s.str.extractall(pattern_two_named)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_named, no_multi_index)

    pattern_one_named = r"(?P<group_name>[a-z])"
    extract_one_named = s.str.extract(pattern_one_named, expand=True)
    has_multi_index = s.str.extractall(pattern_one_named)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_named, no_multi_index)

    pattern_one_noname = r"([a-z])"
    extract_one_noname = s.str.extract(pattern_one_noname, expand=True)
    has_multi_index = s.str.extractall(pattern_one_noname)
    no_multi_index = has_multi_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_noname, no_multi_index)


def test_extractall_same_as_extract_subject_index(any_string_dtype):
    # same as above tests, but s has an MultiIndex.
    mi = MultiIndex.from_tuples(
        [("A", "first"), ("B", "second"), ("C", "third")],
        names=("capital", "ordinal"),
    )
    s = Series(["a3", "b3", "c2"], index=mi, name="series_name", dtype=any_string_dtype)

    pattern_two_noname = r"([a-z])([0-9])"
    extract_two_noname = s.str.extract(pattern_two_noname, expand=True)
    has_match_index = s.str.extractall(pattern_two_noname)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_noname, no_match_index)

    pattern_two_named = r"(?P<letter>[a-z])(?P<digit>[0-9])"
    extract_two_named = s.str.extract(pattern_two_named, expand=True)
    has_match_index = s.str.extractall(pattern_two_named)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_two_named, no_match_index)

    pattern_one_named = r"(?P<group_name>[a-z])"
    extract_one_named = s.str.extract(pattern_one_named, expand=True)
    has_match_index = s.str.extractall(pattern_one_named)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_named, no_match_index)

    pattern_one_noname = r"([a-z])"
    extract_one_noname = s.str.extract(pattern_one_noname, expand=True)
    has_match_index = s.str.extractall(pattern_one_noname)
    no_match_index = has_match_index.xs(0, level="match")
    tm.assert_frame_equal(extract_one_noname, no_match_index)


def test_extractall_preserves_dtype():
    # Ensure that when extractall is called on a series with specific dtypes set, that
    # the dtype is preserved in the resulting DataFrame's column.
    pa = pytest.importorskip("pyarrow")

    result = Series(["abc", "ab"], dtype=ArrowDtype(pa.string())).str.extractall("(ab)")
    assert result.dtypes[0] == "string[pyarrow]"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_expanding.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    isna,
    notna,
)
import pandas._testing as tm


def test_doc_string():
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df
    df.expanding(2).sum()


def test_constructor(frame_or_series):
    # GH 12669

    c = frame_or_series(range(5)).expanding

    # valid
    c(min_periods=1)


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_constructor_invalid(frame_or_series, w):
    # not valid

    c = frame_or_series(range(5)).expanding
    msg = "min_periods must be an integer"
    with pytest.raises(ValueError, match=msg):
        c(min_periods=w)


@pytest.mark.parametrize(
    "expander",
    [
        1,
        pytest.param(
            "ls",
            marks=pytest.mark.xfail(
                reason="GH#16425 expanding with offset not supported"
            ),
        ),
    ],
)
def test_empty_df_expanding(expander):
    # GH 15819 Verifies that datetime and integer expanding windows can be
    # applied to empty DataFrames

    expected = DataFrame()
    result = DataFrame().expanding(expander).sum()
    tm.assert_frame_equal(result, expected)

    # Verifies that datetime and integer expanding windows can be applied
    # to empty DataFrames with datetime index
    expected = DataFrame(index=DatetimeIndex([]))
    result = DataFrame(index=DatetimeIndex([])).expanding(expander).sum()
    tm.assert_frame_equal(result, expected)


def test_missing_minp_zero():
    # https://github.com/pandas-dev/pandas/pull/18921
    # minp=0
    x = Series([np.nan])
    result = x.expanding(min_periods=0).sum()
    expected = Series([0.0])
    tm.assert_series_equal(result, expected)

    # minp=1
    result = x.expanding(min_periods=1).sum()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_expanding_axis(axis_frame):
    # see gh-23372.
    df = DataFrame(np.ones((10, 20)))
    axis = df._get_axis_number(axis_frame)

    if axis == 0:
        msg = "The 'axis' keyword in DataFrame.expanding is deprecated"
        expected = DataFrame(
            {i: [np.nan] * 2 + [float(j) for j in range(3, 11)] for i in range(20)}
        )
    else:
        # axis == 1
        msg = "Support for axis=1 in DataFrame.expanding is deprecated"
        expected = DataFrame([[np.nan] * 2 + [float(i) for i in range(3, 21)]] * 10)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.expanding(3, axis=axis_frame).sum()
    tm.assert_frame_equal(result, expected)


def test_expanding_count_with_min_periods(frame_or_series):
    # GH 26996
    result = frame_or_series(range(5)).expanding(min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 4.0, 5.0])
    tm.assert_equal(result, expected)


def test_expanding_count_default_min_periods_with_null_values(frame_or_series):
    # GH 26996
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]

    result = frame_or_series(values).expanding().count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)


def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series):
    # GH 25857
    result = frame_or_series(range(5)).expanding(min_periods=6).count()
    expected = frame_or_series([np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "df,expected,min_periods",
    [
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            3,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            2,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            1,
        ),
        (DataFrame({"A": [1], "B": [4]}), [], 2),
        (DataFrame(), [({}, [])], 1),
        (
            DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}),
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            3,
        ),
        (
            DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}),
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            2,
        ),
        (
            DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}),
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            1,
        ),
    ],
)
def test_iter_expanding_dataframe(df, expected, min_periods):
    # GH 11704
    expected = [DataFrame(values, index=index) for (values, index) in expected]

    for expected, actual in zip(expected, df.expanding(min_periods)):
        tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "ser,expected,min_periods",
    [
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3),
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 2),
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 1),
        (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2),
        (Series([np.nan, 2]), [([np.nan], [0]), ([np.nan, 2], [0, 1])], 2),
        (Series([], dtype="int64"), [], 2),
    ],
)
def test_iter_expanding_series(ser, expected, min_periods):
    # GH 11704
    expected = [Series(values, index=index) for (values, index) in expected]

    for expected, actual in zip(expected, ser.expanding(min_periods)):
        tm.assert_series_equal(actual, expected)


def test_center_invalid():
    # GH 20647
    df = DataFrame()
    with pytest.raises(TypeError, match=".* got an unexpected keyword"):
        df.expanding(center=True)


def test_expanding_sem(frame_or_series):
    # GH: 26476
    obj = frame_or_series([0, 1, 2])
    result = obj.expanding().sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan] + [0.707107] * 2)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_expanding_skew_kurt_numerical_stability(method):
    # GH: 6929
    s = Series(np.random.default_rng(2).random(10))
    expected = getattr(s.expanding(3), method)()
    s = s + 5000
    result = getattr(s.expanding(3), method)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("window", [1, 3, 10, 20])
@pytest.mark.parametrize("method", ["min", "max", "average"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("test_data", ["default", "duplicates", "nans"])
def test_rank(window, method, pct, ascending, test_data):
    length = 20
    if test_data == "default":
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == "duplicates":
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == "nans":
        ser = Series(
            data=np.random.default_rng(2).choice(
                [1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length
            )
        )

    expected = ser.expanding(window).apply(
        lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1]
    )
    result = ser.expanding(window).rank(method=method, pct=pct, ascending=ascending)

    tm.assert_series_equal(result, expected)


def test_expanding_corr(series):
    A = series.dropna()
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]

    result = A.expanding().corr(B)

    rolling_result = A.rolling(window=len(A), min_periods=1).corr(B)

    tm.assert_almost_equal(rolling_result, result)


def test_expanding_count(series):
    result = series.expanding(min_periods=0).count()
    tm.assert_almost_equal(
        result, series.rolling(window=len(series), min_periods=0).count()
    )


def test_expanding_quantile(series):
    result = series.expanding().quantile(0.5)

    rolling_result = series.rolling(window=len(series), min_periods=1).quantile(0.5)

    tm.assert_almost_equal(result, rolling_result)


def test_expanding_cov(series):
    A = series
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]

    result = A.expanding().cov(B)

    rolling_result = A.rolling(window=len(A), min_periods=1).cov(B)

    tm.assert_almost_equal(rolling_result, result)


def test_expanding_cov_pairwise(frame):
    result = frame.expanding().cov()

    rolling_result = frame.rolling(window=len(frame), min_periods=1).cov()

    tm.assert_frame_equal(result, rolling_result)


def test_expanding_corr_pairwise(frame):
    result = frame.expanding().corr()

    rolling_result = frame.rolling(window=len(frame), min_periods=1).corr()
    tm.assert_frame_equal(result, rolling_result)


@pytest.mark.parametrize(
    "func,static_comp",
    [
        ("sum", np.sum),
        ("mean", lambda x: np.mean(x, axis=0)),
        ("max", lambda x: np.max(x, axis=0)),
        ("min", lambda x: np.min(x, axis=0)),
    ],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_func(func, static_comp, frame_or_series):
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))

    msg = "The 'axis' keyword in (Series|DataFrame).expanding is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        obj = data.expanding(min_periods=1, axis=0)
    result = getattr(obj, func)()
    assert isinstance(result, frame_or_series)

    msg = "The behavior of DataFrame.sum with axis=None is deprecated"
    warn = None
    if frame_or_series is DataFrame and static_comp is np.sum:
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        expected = static_comp(data[:11])
    if frame_or_series is Series:
        tm.assert_almost_equal(result[10], expected)
    else:
        tm.assert_series_equal(result.iloc[10], expected, check_names=False)


@pytest.mark.parametrize(
    "func,static_comp",
    [("sum", np.sum), ("mean", np.mean), ("max", np.max), ("min", np.min)],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_min_periods(func, static_comp):
    ser = Series(np.random.default_rng(2).standard_normal(50))

    msg = "The 'axis' keyword in Series.expanding is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser.expanding(min_periods=30, axis=0), func)()
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))

    # min_periods is working correctly
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser.expanding(min_periods=15, axis=0), func)()
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])

    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser2.expanding(min_periods=5, axis=0), func)()
    assert isna(result[3])
    assert notna(result[4])

    # min_periods=0
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result0 = getattr(ser.expanding(min_periods=0, axis=0), func)()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result1 = getattr(ser.expanding(min_periods=1, axis=0), func)()
    tm.assert_almost_equal(result0, result1)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser.expanding(min_periods=1, axis=0), func)()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))


def test_expanding_apply(engine_and_raw, frame_or_series):
    engine, raw = engine_and_raw
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    result = data.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isinstance(result, frame_or_series)

    if frame_or_series is Series:
        tm.assert_almost_equal(result[9], np.mean(data[:11], axis=0))
    else:
        tm.assert_series_equal(
            result.iloc[9], np.mean(data[:11], axis=0), check_names=False
        )


def test_expanding_min_periods_apply(engine_and_raw):
    engine, raw = engine_and_raw
    ser = Series(np.random.default_rng(2).standard_normal(50))

    result = ser.expanding(min_periods=30).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))

    # min_periods is working correctly
    result = ser.expanding(min_periods=15).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])

    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    result = ser2.expanding(min_periods=5).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isna(result[3])
    assert notna(result[4])

    # min_periods=0
    result0 = ser.expanding(min_periods=0).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    result1 = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    tm.assert_almost_equal(result0, result1)

    result = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.expanding(min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.expanding(min_periods=5).corr(x, pairwise=True)),
    ],
)
def test_moment_functions_zero_length_pairwise(f):
    df1 = DataFrame()
    df2 = DataFrame(columns=Index(["a"], name="foo"), index=Index([], name="bar"))
    df2["a"] = df2["a"].astype("float64")

    df1_expected = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    df2_expected = DataFrame(
        index=MultiIndex.from_product([df2.index, df2.columns], names=["bar", "foo"]),
        columns=Index(["a"], name="foo"),
        dtype="float64",
    )

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.expanding().count(),
        lambda x: x.expanding(min_periods=5).cov(x, pairwise=False),
        lambda x: x.expanding(min_periods=5).corr(x, pairwise=False),
        lambda x: x.expanding(min_periods=5).max(),
        lambda x: x.expanding(min_periods=5).min(),
        lambda x: x.expanding(min_periods=5).sum(),
        lambda x: x.expanding(min_periods=5).mean(),
        lambda x: x.expanding(min_periods=5).std(),
        lambda x: x.expanding(min_periods=5).var(),
        lambda x: x.expanding(min_periods=5).skew(),
        lambda x: x.expanding(min_periods=5).kurt(),
        lambda x: x.expanding(min_periods=5).quantile(0.5),
        lambda x: x.expanding(min_periods=5).median(),
        lambda x: x.expanding(min_periods=5).apply(sum, raw=False),
        lambda x: x.expanding(min_periods=5).apply(sum, raw=True),
    ],
)
def test_moment_functions_zero_length(f):
    # GH 8056
    s = Series(dtype=np.float64)
    s_expected = s
    df1 = DataFrame()
    df1_expected = df1
    df2 = DataFrame(columns=["a"])
    df2["a"] = df2["a"].astype("float64")
    df2_expected = df2

    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)


def test_expanding_apply_empty_series(engine_and_raw):
    engine, raw = engine_and_raw
    ser = Series([], dtype=np.float64)
    tm.assert_series_equal(
        ser, ser.expanding().apply(lambda x: x.mean(), raw=raw, engine=engine)
    )


def test_expanding_apply_min_periods_0(engine_and_raw):
    # GH 8080
    engine, raw = engine_and_raw
    s = Series([None, None, None])
    result = s.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    expected = Series([1.0, 2.0, 3.0])
    tm.assert_series_equal(result, expected)


def test_expanding_cov_diff_index():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.expanding().cov(s2)
    expected = Series([None, None, 2.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().cov(s2a)
    tm.assert_series_equal(result, expected)

    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().cov(s2)
    expected = Series([None, None, None, 4.5])
    tm.assert_series_equal(result, expected)


def test_expanding_corr_diff_index():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.expanding().corr(s2)
    expected = Series([None, None, 1.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().corr(s2a)
    tm.assert_series_equal(result, expected)

    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().corr(s2)
    expected = Series([None, None, None, 1.0])
    tm.assert_series_equal(result, expected)


def test_expanding_cov_pairwise_diff_length():
    # GH 7512
    df1 = DataFrame([[1, 5], [3, 2], [3, 9]], columns=Index(["A", "B"], name="foo"))
    df1a = DataFrame(
        [[1, 5], [3, 9]], index=[0, 2], columns=Index(["A", "B"], name="foo")
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]], columns=Index(["X", "Y"], name="foo")
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=[0, 2], columns=Index(["X", "Y"], name="foo")
    )
    # TODO: xref gh-15826
    # .loc is not preserving the names
    result1 = df1.expanding().cov(df2, pairwise=True).loc[2]
    result2 = df1.expanding().cov(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().cov(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().cov(df2a, pairwise=True).loc[2]
    expected = DataFrame(
        [[-3.0, -6.0], [-5.0, -10.0]],
        columns=Index(["A", "B"], name="foo"),
        index=Index(["X", "Y"], name="foo"),
    )
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


def test_expanding_corr_pairwise_diff_length():
    # GH 7512
    df1 = DataFrame(
        [[1, 2], [3, 2], [3, 4]], columns=["A", "B"], index=Index(range(3), name="bar")
    )
    df1a = DataFrame(
        [[1, 2], [3, 4]], index=Index([0, 2], name="bar"), columns=["A", "B"]
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]],
        columns=["X", "Y"],
        index=Index(range(3), name="bar"),
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=Index([0, 2], name="bar"), columns=["X", "Y"]
    )
    result1 = df1.expanding().corr(df2, pairwise=True).loc[2]
    result2 = df1.expanding().corr(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().corr(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().corr(df2a, pairwise=True).loc[2]
    expected = DataFrame(
        [[-1.0, -1.0], [-1.0, -1.0]], columns=["A", "B"], index=Index(["X", "Y"])
    )
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


def test_expanding_apply_args_kwargs(engine_and_raw):
    def mean_w_arg(x, const):
        return np.mean(x) + const

    engine, raw = engine_and_raw

    df = DataFrame(np.random.default_rng(2).random((20, 3)))

    expected = df.expanding().apply(np.mean, engine=engine, raw=raw) + 20.0

    result = df.expanding().apply(mean_w_arg, engine=engine, raw=raw, args=(20,))
    tm.assert_frame_equal(result, expected)

    result = df.expanding().apply(mean_w_arg, raw=raw, kwargs={"const": 20})
    tm.assert_frame_equal(result, expected)


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # GH#46560
    kernel = arithmetic_win_operators
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    expanding = df.expanding()
    op = getattr(expanding, kernel, None)
    if op is not None:
        result = op(numeric_only=numeric_only)

        columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        assert list(expected.columns) == columns

        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # GH#46560
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    arg = (df,) if use_arg else ()
    expanding = df.expanding()
    op = getattr(expanding, kernel)
    result = op(*arg, numeric_only=numeric_only)

    # Compare result to op using float dtypes, dropping c when numeric_only is True
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    expanding2 = df2.expanding()
    op2 = getattr(expanding2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # GH#46560
    kernel = arithmetic_win_operators
    ser = Series([1], dtype=dtype)
    expanding = ser.expanding()
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        msg = f"Expanding.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        result = op(numeric_only=numeric_only)
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # GH#46560
    ser = Series([1, 2, 3], dtype=dtype)
    arg = (ser,) if use_arg else ()
    expanding = ser.expanding()
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        msg = f"Expanding.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result = op(*arg, numeric_only=numeric_only)

        ser2 = ser.astype(float)
        arg2 = (ser2,) if use_arg else ()
        expanding2 = ser2.expanding()
        op2 = getattr(expanding2, kernel)
        expected = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)


def test_keyword_quantile_deprecated():
    # GH #52550
    ser = Series([1, 2, 3, 4])
    with tm.assert_produces_warning(FutureWarning):
        ser.expanding().quantile(quantile=0.5)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_pickling.py ===
import inspect
import copy
import pickle

from sympy.physics.units import meter

from sympy.testing.pytest import XFAIL, raises, ignore_warnings

from sympy.core.basic import Atom, Basic
from sympy.core.singleton import SingletonRegistry
from sympy.core.symbol import Str, Dummy, Symbol, Wild
from sympy.core.numbers import (E, I, pi, oo, zoo, nan, Integer,
        Rational, Float, AlgebraicNumber)
from sympy.core.relational import (Equality, GreaterThan, LessThan, Relational,
        StrictGreaterThan, StrictLessThan, Unequality)
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import Derivative, Function, FunctionClass, Lambda, \
    WildFunction
from sympy.sets.sets import Interval
from sympy.core.multidimensional import vectorize

from sympy.external.gmpy import gmpy as _gmpy
from sympy.utilities.exceptions import SymPyDeprecationWarning

from sympy.core.singleton import S
from sympy.core.symbol import symbols

from sympy.external import import_module
cloudpickle = import_module('cloudpickle')


not_equal_attrs = {
    '_assumptions',  # This is a local cache that isn't automatically filled on creation
    '_mhash',   # Cached after __hash__ is called but set to None after creation
}


deprecated_attrs = {
    'is_EmptySet',  # Deprecated from SymPy 1.5. This can be removed when is_EmptySet is removed.
    'expr_free_symbols',  # Deprecated from SymPy 1.9. This can be removed when exr_free_symbols is removed.
}

dont_check_attrs = {
    '_sage_',  # Fails because Sage is not installed
}


def check(a, exclude=[], check_attr=True, deprecated=()):
    """ Check that pickling and copying round-trips.
    """
    # Pickling with protocols 0 and 1 is disabled for Basic instances:
    if isinstance(a, Basic):
        for protocol in [0, 1]:
            raises(NotImplementedError, lambda: pickle.dumps(a, protocol))

    protocols = [2, copy.copy, copy.deepcopy, 3, 4]
    if cloudpickle:
        protocols.extend([cloudpickle])

    for protocol in protocols:
        if protocol in exclude:
            continue

        if callable(protocol):
            if isinstance(a, type):
                # Classes can't be copied, but that's okay.
                continue
            b = protocol(a)
        elif inspect.ismodule(protocol):
            b = protocol.loads(protocol.dumps(a))
        else:
            b = pickle.loads(pickle.dumps(a, protocol))

        d1 = dir(a)
        d2 = dir(b)
        assert set(d1) == set(d2)

        if not check_attr:
            continue

        def c(a, b, d):
            for i in d:
                if i in dont_check_attrs:
                    continue
                elif i in not_equal_attrs:
                    if hasattr(a, i):
                        assert hasattr(b, i), i
                elif i in deprecated_attrs or i in deprecated:
                    with ignore_warnings(SymPyDeprecationWarning):
                        assert getattr(a, i) == getattr(b, i), i
                elif not hasattr(a, i):
                    continue
                else:
                    attr = getattr(a, i)
                    if not hasattr(attr, "__call__"):
                        assert hasattr(b, i), i
                        assert getattr(b, i) == attr, "%s != %s, protocol: %s" % (getattr(b, i), attr, protocol)

        c(a, b, d1)
        c(b, a, d2)



#================== core =========================


def test_core_basic():
    for c in (Atom, Atom(), Basic, Basic(), SingletonRegistry, S):
        check(c)

def test_core_Str():
    check(Str('x'))

def test_core_symbol():
    # make the Symbol a unique name that doesn't class with any other
    # testing variable in this file since after this test the symbol
    # having the same name will be cached as noncommutative
    for c in (Dummy, Dummy("x", commutative=False), Symbol,
            Symbol("_issue_3130", commutative=False), Wild, Wild("x")):
        check(c)


def test_core_numbers():
    for c in (Integer(2), Rational(2, 3), Float("1.2")):
        check(c)
    for c in (AlgebraicNumber, AlgebraicNumber(sqrt(3))):
        check(c, check_attr=False)


def test_core_float_copy():
    # See gh-7457
    y = Symbol("x") + 1.0
    check(y)  # does not raise TypeError ("argument is not an mpz")


def test_core_relational():
    x = Symbol("x")
    y = Symbol("y")
    for c in (Equality, Equality(x, y), GreaterThan, GreaterThan(x, y),
              LessThan, LessThan(x, y), Relational, Relational(x, y),
              StrictGreaterThan, StrictGreaterThan(x, y), StrictLessThan,
              StrictLessThan(x, y), Unequality, Unequality(x, y)):
        check(c)


def test_core_add():
    x = Symbol("x")
    for c in (Add, Add(x, 4)):
        check(c)


def test_core_mul():
    x = Symbol("x")
    for c in (Mul, Mul(x, 4)):
        check(c)


def test_core_power():
    x = Symbol("x")
    for c in (Pow, Pow(x, 4)):
        check(c)


def test_core_function():
    x = Symbol("x")
    for f in (Derivative, Derivative(x), Function, FunctionClass, Lambda,
              WildFunction):
        check(f)


def test_core_undefinedfunctions():
    f = Function("f")
    check(f)


def test_core_appliedundef():
    x = Symbol("_long_unique_name_1")
    f = Function("_long_unique_name_2")
    check(f(x))


def test_core_interval():
    for c in (Interval, Interval(0, 2)):
        check(c)


def test_core_multidimensional():
    for c in (vectorize, vectorize(0)):
        check(c)


def test_Singletons():
    protocols = [0, 1, 2, 3, 4]
    copiers = [copy.copy, copy.deepcopy]
    copiers += [lambda x: pickle.loads(pickle.dumps(x, proto))
            for proto in protocols]
    if cloudpickle:
        copiers += [lambda x: cloudpickle.loads(cloudpickle.dumps(x))]

    for obj in (Integer(-1), Integer(0), Integer(1), Rational(1, 2), pi, E, I,
            oo, -oo, zoo, nan, S.GoldenRatio, S.TribonacciConstant,
            S.EulerGamma, S.Catalan, S.EmptySet, S.IdentityFunction):
        for func in copiers:
            assert func(obj) is obj

#================== combinatorics ===================
from sympy.combinatorics.free_groups import FreeGroup

def test_free_group():
    check(FreeGroup("x, y, z"), check_attr=False)

#================== functions ===================
from sympy.functions import (Piecewise, lowergamma, acosh, chebyshevu,
        chebyshevt, ln, chebyshevt_root, legendre, Heaviside, bernoulli, coth,
        tanh, assoc_legendre, sign, arg, asin, DiracDelta, re, rf, Abs,
        uppergamma, binomial, sinh, cos, cot, acos, acot, gamma, bell,
        hermite, harmonic, LambertW, zeta, log, factorial, asinh, acoth, cosh,
        dirichlet_eta, Eijk, loggamma, erf, ceiling, im, fibonacci,
        tribonacci, conjugate, tan, chebyshevu_root, floor, atanh, sqrt, sin,
        atan, ff, lucas, atan2, polygamma, exp)


def test_functions():
    one_var = (acosh, ln, Heaviside, factorial, bernoulli, coth, tanh,
            sign, arg, asin, DiracDelta, re, Abs, sinh, cos, cot, acos, acot,
            gamma, bell, harmonic, LambertW, zeta, log, factorial, asinh,
            acoth, cosh, dirichlet_eta, loggamma, erf, ceiling, im, fibonacci,
            tribonacci, conjugate, tan, floor, atanh, sin, atan, lucas, exp)
    two_var = (rf, ff, lowergamma, chebyshevu, chebyshevt, binomial,
            atan2, polygamma, hermite, legendre, uppergamma)
    x, y, z = symbols("x,y,z")
    others = (chebyshevt_root, chebyshevu_root, Eijk(x, y, z),
            Piecewise( (0, x < -1), (x**2, x <= 1), (x**3, True)),
            assoc_legendre)
    for cls in one_var:
        check(cls)
        c = cls(x)
        check(c)
    for cls in two_var:
        check(cls)
        c = cls(x, y)
        check(c)
    for cls in others:
        check(cls)

#================== geometry ====================
from sympy.geometry.entity import GeometryEntity
from sympy.geometry.point import Point
from sympy.geometry.ellipse import Circle, Ellipse
from sympy.geometry.line import Line, LinearEntity, Ray, Segment
from sympy.geometry.polygon import Polygon, RegularPolygon, Triangle


def test_geometry():
    p1 = Point(1, 2)
    p2 = Point(2, 3)
    p3 = Point(0, 0)
    p4 = Point(0, 1)
    for c in (
        GeometryEntity, GeometryEntity(), Point, p1, Circle, Circle(p1, 2),
        Ellipse, Ellipse(p1, 3, 4), Line, Line(p1, p2), LinearEntity,
        LinearEntity(p1, p2), Ray, Ray(p1, p2), Segment, Segment(p1, p2),
        Polygon, Polygon(p1, p2, p3, p4), RegularPolygon,
            RegularPolygon(p1, 4, 5), Triangle, Triangle(p1, p2, p3)):
        check(c, check_attr=False)

#================== integrals ====================
from sympy.integrals.integrals import Integral


def test_integrals():
    x = Symbol("x")
    for c in (Integral, Integral(x)):
        check(c)

#==================== logic =====================
from sympy.core.logic import Logic


def test_logic():
    for c in (Logic, Logic(1)):
        check(c)

#================== matrices ====================
from sympy.matrices import Matrix, SparseMatrix


def test_matrices():
    for c in (Matrix, Matrix([1, 2, 3]), SparseMatrix, SparseMatrix([[1, 2], [3, 4]])):
        check(c, deprecated=['_smat', '_mat'])

#================== ntheory =====================
from sympy.ntheory.generate import Sieve


def test_ntheory():
    for c in (Sieve, Sieve()):
        check(c)

#================== physics =====================
from sympy.physics.paulialgebra import Pauli
from sympy.physics.units import Unit


def test_physics():
    for c in (Unit, meter, Pauli, Pauli(1)):
        check(c)

#================== plotting ====================
# XXX: These tests are not complete, so XFAIL them


@XFAIL
def test_plotting():
    from sympy.plotting.pygletplot.color_scheme import ColorGradient, ColorScheme
    from sympy.plotting.pygletplot.managed_window import ManagedWindow
    from sympy.plotting.plot import Plot, ScreenShot
    from sympy.plotting.pygletplot.plot_axes import PlotAxes, PlotAxesBase, PlotAxesFrame, PlotAxesOrdinate
    from sympy.plotting.pygletplot.plot_camera import PlotCamera
    from sympy.plotting.pygletplot.plot_controller import PlotController
    from sympy.plotting.pygletplot.plot_curve import PlotCurve
    from sympy.plotting.pygletplot.plot_interval import PlotInterval
    from sympy.plotting.pygletplot.plot_mode import PlotMode
    from sympy.plotting.pygletplot.plot_modes import Cartesian2D, Cartesian3D, Cylindrical, \
        ParametricCurve2D, ParametricCurve3D, ParametricSurface, Polar, Spherical
    from sympy.plotting.pygletplot.plot_object import PlotObject
    from sympy.plotting.pygletplot.plot_surface import PlotSurface
    from sympy.plotting.pygletplot.plot_window import PlotWindow
    for c in (
        ColorGradient, ColorGradient(0.2, 0.4), ColorScheme, ManagedWindow,
        ManagedWindow, Plot, ScreenShot, PlotAxes, PlotAxesBase,
        PlotAxesFrame, PlotAxesOrdinate, PlotCamera, PlotController,
        PlotCurve, PlotInterval, PlotMode, Cartesian2D, Cartesian3D,
        Cylindrical, ParametricCurve2D, ParametricCurve3D,
        ParametricSurface, Polar, Spherical, PlotObject, PlotSurface,
            PlotWindow):
        check(c)


@XFAIL
def test_plotting2():
    #from sympy.plotting.color_scheme import ColorGradient
    from sympy.plotting.pygletplot.color_scheme import ColorScheme
    #from sympy.plotting.managed_window import ManagedWindow
    from sympy.plotting.plot import Plot
    #from sympy.plotting.plot import ScreenShot
    from sympy.plotting.pygletplot.plot_axes import PlotAxes
    #from sympy.plotting.plot_axes import PlotAxesBase, PlotAxesFrame, PlotAxesOrdinate
    #from sympy.plotting.plot_camera import PlotCamera
    #from sympy.plotting.plot_controller import PlotController
    #from sympy.plotting.plot_curve import PlotCurve
    #from sympy.plotting.plot_interval import PlotInterval
    #from sympy.plotting.plot_mode import PlotMode
    #from sympy.plotting.plot_modes import Cartesian2D, Cartesian3D, Cylindrical, \
    #    ParametricCurve2D, ParametricCurve3D, ParametricSurface, Polar, Spherical
    #from sympy.plotting.plot_object import PlotObject
    #from sympy.plotting.plot_surface import PlotSurface
    # from sympy.plotting.plot_window import PlotWindow
    check(ColorScheme("rainbow"))
    check(Plot(1, visible=False))
    check(PlotAxes())

#================== polys =======================
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.orderings import lex
from sympy.polys.polytools import Poly

def test_pickling_polys_polytools():
    from sympy.polys.polytools import PurePoly
    # from sympy.polys.polytools import GroebnerBasis
    x = Symbol('x')

    for c in (Poly, Poly(x, x)):
        check(c)

    for c in (PurePoly, PurePoly(x)):
        check(c)

    # TODO: fix pickling of Options class (see GroebnerBasis._options)
    # for c in (GroebnerBasis, GroebnerBasis([x**2 - 1], x, order=lex)):
    #     check(c)

def test_pickling_polys_polyclasses():
    from sympy.polys.polyclasses import DMP, DMF, ANP

    for c in (DMP, DMP([[ZZ(1)], [ZZ(2)], [ZZ(3)]], ZZ)):
        check(c, deprecated=['rep'])
    for c in (DMF, DMF(([ZZ(1), ZZ(2)], [ZZ(1), ZZ(3)]), ZZ)):
        check(c)
    for c in (ANP, ANP([QQ(1), QQ(2)], [QQ(1), QQ(2), QQ(3)], QQ)):
        check(c)

@XFAIL
def test_pickling_polys_rings():
    # NOTE: can't use protocols < 2 because we have to execute __new__ to
    # make sure caching of rings works properly.

    from sympy.polys.rings import PolyRing

    ring = PolyRing("x,y,z", ZZ, lex)

    for c in (PolyRing, ring):
        check(c, exclude=[0, 1])

    for c in (ring.dtype, ring.one):
        check(c, exclude=[0, 1], check_attr=False) # TODO: Py3k

def test_pickling_polys_fields():
    pass
    # NOTE: can't use protocols < 2 because we have to execute __new__ to
    # make sure caching of fields works properly.

    # from sympy.polys.fields import FracField

    # field = FracField("x,y,z", ZZ, lex)

    # TODO: AssertionError: assert id(obj) not in self.memo
    # for c in (FracField, field):
    #     check(c, exclude=[0, 1])

    # TODO: AssertionError: assert id(obj) not in self.memo
    # for c in (field.dtype, field.one):
    #     check(c, exclude=[0, 1])

def test_pickling_polys_elements():
    from sympy.polys.domains.pythonrational import PythonRational
    #from sympy.polys.domains.pythonfinitefield import PythonFiniteField
    #from sympy.polys.domains.mpelements import MPContext

    for c in (PythonRational, PythonRational(1, 7)):
        check(c)

    #gf = PythonFiniteField(17)

    # TODO: fix pickling of ModularInteger
    # for c in (gf.dtype, gf(5)):
    #     check(c)

    #mp = MPContext()

    # TODO: fix pickling of RealElement
    # for c in (mp.mpf, mp.mpf(1.0)):
    #     check(c)

    # TODO: fix pickling of ComplexElement
    # for c in (mp.mpc, mp.mpc(1.0, -1.5)):
    #     check(c)

def test_pickling_polys_domains():
    # from sympy.polys.domains.pythonfinitefield import PythonFiniteField
    from sympy.polys.domains.pythonintegerring import PythonIntegerRing
    from sympy.polys.domains.pythonrationalfield import PythonRationalField

    # TODO: fix pickling of ModularInteger
    # for c in (PythonFiniteField, PythonFiniteField(17)):
    #     check(c)

    for c in (PythonIntegerRing, PythonIntegerRing()):
        check(c, check_attr=False)

    for c in (PythonRationalField, PythonRationalField()):
        check(c, check_attr=False)

    if _gmpy is not None:
        # from sympy.polys.domains.gmpyfinitefield import GMPYFiniteField
        from sympy.polys.domains.gmpyintegerring import GMPYIntegerRing
        from sympy.polys.domains.gmpyrationalfield import GMPYRationalField

        # TODO: fix pickling of ModularInteger
        # for c in (GMPYFiniteField, GMPYFiniteField(17)):
        #     check(c)

        for c in (GMPYIntegerRing, GMPYIntegerRing()):
            check(c, check_attr=False)

        for c in (GMPYRationalField, GMPYRationalField()):
            check(c, check_attr=False)

    #from sympy.polys.domains.realfield import RealField
    #from sympy.polys.domains.complexfield import ComplexField
    from sympy.polys.domains.algebraicfield import AlgebraicField
    #from sympy.polys.domains.polynomialring import PolynomialRing
    #from sympy.polys.domains.fractionfield import FractionField
    from sympy.polys.domains.expressiondomain import ExpressionDomain

    # TODO: fix pickling of RealElement
    # for c in (RealField, RealField(100)):
    #     check(c)

    # TODO: fix pickling of ComplexElement
    # for c in (ComplexField, ComplexField(100)):
    #     check(c)

    for c in (AlgebraicField, AlgebraicField(QQ, sqrt(3))):
        check(c, check_attr=False)

    # TODO: AssertionError
    # for c in (PolynomialRing, PolynomialRing(ZZ, "x,y,z")):
    #     check(c)

    # TODO: AttributeError: 'PolyElement' object has no attribute 'ring'
    # for c in (FractionField, FractionField(ZZ, "x,y,z")):
    #     check(c)

    for c in (ExpressionDomain, ExpressionDomain()):
        check(c, check_attr=False)


def test_pickling_polys_orderings():
    from sympy.polys.orderings import (LexOrder, GradedLexOrder,
        ReversedGradedLexOrder, InverseOrder)
    # from sympy.polys.orderings import ProductOrder

    for c in (LexOrder, LexOrder()):
        check(c)

    for c in (GradedLexOrder, GradedLexOrder()):
        check(c)

    for c in (ReversedGradedLexOrder, ReversedGradedLexOrder()):
        check(c)

    # TODO: Argh, Python is so naive. No lambdas nor inner function support in
    # pickling module. Maybe someone could figure out what to do with this.
    #
    # for c in (ProductOrder, ProductOrder((LexOrder(),       lambda m: m[:2]),
    #                                      (GradedLexOrder(), lambda m: m[2:]))):
    #     check(c)

    for c in (InverseOrder, InverseOrder(LexOrder())):
        check(c)

def test_pickling_polys_monomials():
    from sympy.polys.monomials import MonomialOps, Monomial
    x, y, z = symbols("x,y,z")

    for c in (MonomialOps, MonomialOps(3)):
        check(c)

    for c in (Monomial, Monomial((1, 2, 3), (x, y, z))):
        check(c)

def test_pickling_polys_errors():
    from sympy.polys.polyerrors import (HeuristicGCDFailed,
        HomomorphismFailed, IsomorphismFailed, ExtraneousFactors,
        EvaluationFailed, RefinementFailed, CoercionFailed, NotInvertible,
        NotReversible, NotAlgebraic, DomainError, PolynomialError,
        UnificationFailed, GeneratorsError, GeneratorsNeeded,
        UnivariatePolynomialError, MultivariatePolynomialError, OptionError,
        FlagError)
    # from sympy.polys.polyerrors import (ExactQuotientFailed,
    #         OperationNotSupported, ComputationFailed, PolificationFailed)

    # x = Symbol('x')

    # TODO: TypeError: __init__() takes at least 3 arguments (1 given)
    # for c in (ExactQuotientFailed, ExactQuotientFailed(x, 3*x, ZZ)):
    #    check(c)

    # TODO: TypeError: can't pickle instancemethod objects
    # for c in (OperationNotSupported, OperationNotSupported(Poly(x), Poly.gcd)):
    #    check(c)

    for c in (HeuristicGCDFailed, HeuristicGCDFailed()):
        check(c)

    for c in (HomomorphismFailed, HomomorphismFailed()):
        check(c)

    for c in (IsomorphismFailed, IsomorphismFailed()):
        check(c)

    for c in (ExtraneousFactors, ExtraneousFactors()):
        check(c)

    for c in (EvaluationFailed, EvaluationFailed()):
        check(c)

    for c in (RefinementFailed, RefinementFailed()):
        check(c)

    for c in (CoercionFailed, CoercionFailed()):
        check(c)

    for c in (NotInvertible, NotInvertible()):
        check(c)

    for c in (NotReversible, NotReversible()):
        check(c)

    for c in (NotAlgebraic, NotAlgebraic()):
        check(c)

    for c in (DomainError, DomainError()):
        check(c)

    for c in (PolynomialError, PolynomialError()):
        check(c)

    for c in (UnificationFailed, UnificationFailed()):
        check(c)

    for c in (GeneratorsError, GeneratorsError()):
        check(c)

    for c in (GeneratorsNeeded, GeneratorsNeeded()):
        check(c)

    # TODO: PicklingError: Can't pickle <function <lambda> at 0x38578c0>: it's not found as __main__.<lambda>
    # for c in (ComputationFailed, ComputationFailed(lambda t: t, 3, None)):
    #    check(c)

    for c in (UnivariatePolynomialError, UnivariatePolynomialError()):
        check(c)

    for c in (MultivariatePolynomialError, MultivariatePolynomialError()):
        check(c)

    # TODO: TypeError: __init__() takes at least 3 arguments (1 given)
    # for c in (PolificationFailed, PolificationFailed({}, x, x, False)):
    #    check(c)

    for c in (OptionError, OptionError()):
        check(c)

    for c in (FlagError, FlagError()):
        check(c)

#def test_pickling_polys_options():
    #from sympy.polys.polyoptions import Options

    # TODO: fix pickling of `symbols' flag
    # for c in (Options, Options((), dict(domain='ZZ', polys=False))):
    #    check(c)

# TODO: def test_pickling_polys_rootisolation():
#    RealInterval
#    ComplexInterval

def test_pickling_polys_rootoftools():
    from sympy.polys.rootoftools import CRootOf, RootSum

    x = Symbol('x')
    f = x**3 + x + 3

    for c in (CRootOf, CRootOf(f, 0)):
        check(c)

    for c in (RootSum, RootSum(f, exp)):
        check(c)

#================== printing ====================
from sympy.printing.latex import LatexPrinter
from sympy.printing.mathml import MathMLContentPrinter, MathMLPresentationPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.printer import Printer
from sympy.printing.python import PythonPrinter


def test_printing():
    for c in (LatexPrinter, LatexPrinter(), MathMLContentPrinter,
              MathMLPresentationPrinter, PrettyPrinter, prettyForm, stringPict,
              stringPict("a"), Printer, Printer(), PythonPrinter,
              PythonPrinter()):
        check(c)


@XFAIL
def test_printing1():
    check(MathMLContentPrinter())


@XFAIL
def test_printing2():
    check(MathMLPresentationPrinter())


@XFAIL
def test_printing3():
    check(PrettyPrinter())

#================== series ======================
from sympy.series.limits import Limit
from sympy.series.order import Order


def test_series():
    e = Symbol("e")
    x = Symbol("x")
    for c in (Limit, Limit(e, x, 1), Order, Order(e)):
        check(c)

#================== concrete ==================
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum


def test_concrete():
    x = Symbol("x")
    for c in (Product, Product(x, (x, 2, 4)), Sum, Sum(x, (x, 2, 4))):
        check(c)

def test_deprecation_warning():
    w = SymPyDeprecationWarning("message", deprecated_since_version='1.0', active_deprecations_target="active-deprecations")
    check(w)

def test_issue_18438():
    assert pickle.loads(pickle.dumps(S.Half)) == S.Half


#================= old pickles =================
def test_unpickle_from_older_versions():
    data = (
        b'\x80\x04\x95^\x00\x00\x00\x00\x00\x00\x00\x8c\x10sympy.core.power'
        b'\x94\x8c\x03Pow\x94\x93\x94\x8c\x12sympy.core.numbers\x94\x8c'
        b'\x07Integer\x94\x93\x94K\x02\x85\x94R\x94}\x94bh\x03\x8c\x04Half'
        b'\x94\x93\x94)R\x94}\x94b\x86\x94R\x94}\x94b.'
    )
    assert pickle.loads(data) == sqrt(2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_rank.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    NaT,
    Series,
    concat,
)
import pandas._testing as tm


def test_rank_unordered_categorical_typeerror():
    # GH#51034 should be TypeError, not NotImplementedError
    cat = pd.Categorical([], ordered=False)
    ser = Series(cat)
    df = ser.to_frame()

    msg = "Cannot perform rank with non-ordered Categorical"

    gb = ser.groupby(cat, observed=False)
    with pytest.raises(TypeError, match=msg):
        gb.rank()

    gb2 = df.groupby(cat, observed=False)
    with pytest.raises(TypeError, match=msg):
        gb2.rank()


def test_rank_apply():
    lev1 = np.array(["a" * 10] * 100, dtype=object)
    lev2 = np.array(["b" * 10] * 130, dtype=object)
    lab1 = np.random.default_rng(2).integers(0, 100, size=500, dtype=int)
    lab2 = np.random.default_rng(2).integers(0, 130, size=500, dtype=int)

    df = DataFrame(
        {
            "value": np.random.default_rng(2).standard_normal(500),
            "key1": lev1.take(lab1),
            "key2": lev2.take(lab2),
        }
    )

    result = df.groupby(["key1", "key2"]).value.rank()

    expected = [piece.value.rank() for key, piece in df.groupby(["key1", "key2"])]
    expected = concat(expected, axis=0)
    expected = expected.reindex(result.index)
    tm.assert_series_equal(result, expected)

    result = df.groupby(["key1", "key2"]).value.rank(pct=True)

    expected = [
        piece.value.rank(pct=True) for key, piece in df.groupby(["key1", "key2"])
    ]
    expected = concat(expected, axis=0)
    expected = expected.reindex(result.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("grps", [["qux"], ["qux", "quux"]])
@pytest.mark.parametrize(
    "vals",
    [
        np.array([2, 2, 8, 2, 6], dtype=dtype)
        for dtype in ["i8", "i4", "i2", "i1", "u8", "u4", "u2", "u1", "f8", "f4", "f2"]
    ]
    + [
        [
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-08"),
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-06"),
        ],
        [
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-08", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-06", tz="US/Pacific"),
        ],
        [
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-08") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-06") - pd.Timestamp(0),
        ],
        [
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-08").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-06").to_period("D"),
        ],
    ],
    ids=lambda x: type(x[0]),
)
@pytest.mark.parametrize(
    "ties_method,ascending,pct,exp",
    [
        ("average", True, False, [2.0, 2.0, 5.0, 2.0, 4.0]),
        ("average", True, True, [0.4, 0.4, 1.0, 0.4, 0.8]),
        ("average", False, False, [4.0, 4.0, 1.0, 4.0, 2.0]),
        ("average", False, True, [0.8, 0.8, 0.2, 0.8, 0.4]),
        ("min", True, False, [1.0, 1.0, 5.0, 1.0, 4.0]),
        ("min", True, True, [0.2, 0.2, 1.0, 0.2, 0.8]),
        ("min", False, False, [3.0, 3.0, 1.0, 3.0, 2.0]),
        ("min", False, True, [0.6, 0.6, 0.2, 0.6, 0.4]),
        ("max", True, False, [3.0, 3.0, 5.0, 3.0, 4.0]),
        ("max", True, True, [0.6, 0.6, 1.0, 0.6, 0.8]),
        ("max", False, False, [5.0, 5.0, 1.0, 5.0, 2.0]),
        ("max", False, True, [1.0, 1.0, 0.2, 1.0, 0.4]),
        ("first", True, False, [1.0, 2.0, 5.0, 3.0, 4.0]),
        ("first", True, True, [0.2, 0.4, 1.0, 0.6, 0.8]),
        ("first", False, False, [3.0, 4.0, 1.0, 5.0, 2.0]),
        ("first", False, True, [0.6, 0.8, 0.2, 1.0, 0.4]),
        ("dense", True, False, [1.0, 1.0, 3.0, 1.0, 2.0]),
        ("dense", True, True, [1.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]),
        ("dense", False, False, [3.0, 3.0, 1.0, 3.0, 2.0]),
        ("dense", False, True, [3.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 2.0 / 3.0]),
    ],
)
def test_rank_args(grps, vals, ties_method, ascending, pct, exp):
    key = np.repeat(grps, len(vals))

    orig_vals = vals
    vals = list(vals) * len(grps)
    if isinstance(orig_vals, np.ndarray):
        vals = np.array(vals, dtype=orig_vals.dtype)

    df = DataFrame({"key": key, "val": vals})
    result = df.groupby("key").rank(method=ties_method, ascending=ascending, pct=pct)

    exp_df = DataFrame(exp * len(grps), columns=["val"])
    tm.assert_frame_equal(result, exp_df)


@pytest.mark.parametrize("grps", [["qux"], ["qux", "quux"]])
@pytest.mark.parametrize(
    "vals", [[-np.inf, -np.inf, np.nan, 1.0, np.nan, np.inf, np.inf]]
)
@pytest.mark.parametrize(
    "ties_method,ascending,na_option,exp",
    [
        ("average", True, "keep", [1.5, 1.5, np.nan, 3, np.nan, 4.5, 4.5]),
        ("average", True, "top", [3.5, 3.5, 1.5, 5.0, 1.5, 6.5, 6.5]),
        ("average", True, "bottom", [1.5, 1.5, 6.5, 3.0, 6.5, 4.5, 4.5]),
        ("average", False, "keep", [4.5, 4.5, np.nan, 3, np.nan, 1.5, 1.5]),
        ("average", False, "top", [6.5, 6.5, 1.5, 5.0, 1.5, 3.5, 3.5]),
        ("average", False, "bottom", [4.5, 4.5, 6.5, 3.0, 6.5, 1.5, 1.5]),
        ("min", True, "keep", [1.0, 1.0, np.nan, 3.0, np.nan, 4.0, 4.0]),
        ("min", True, "top", [3.0, 3.0, 1.0, 5.0, 1.0, 6.0, 6.0]),
        ("min", True, "bottom", [1.0, 1.0, 6.0, 3.0, 6.0, 4.0, 4.0]),
        ("min", False, "keep", [4.0, 4.0, np.nan, 3.0, np.nan, 1.0, 1.0]),
        ("min", False, "top", [6.0, 6.0, 1.0, 5.0, 1.0, 3.0, 3.0]),
        ("min", False, "bottom", [4.0, 4.0, 6.0, 3.0, 6.0, 1.0, 1.0]),
        ("max", True, "keep", [2.0, 2.0, np.nan, 3.0, np.nan, 5.0, 5.0]),
        ("max", True, "top", [4.0, 4.0, 2.0, 5.0, 2.0, 7.0, 7.0]),
        ("max", True, "bottom", [2.0, 2.0, 7.0, 3.0, 7.0, 5.0, 5.0]),
        ("max", False, "keep", [5.0, 5.0, np.nan, 3.0, np.nan, 2.0, 2.0]),
        ("max", False, "top", [7.0, 7.0, 2.0, 5.0, 2.0, 4.0, 4.0]),
        ("max", False, "bottom", [5.0, 5.0, 7.0, 3.0, 7.0, 2.0, 2.0]),
        ("first", True, "keep", [1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0]),
        ("first", True, "top", [3.0, 4.0, 1.0, 5.0, 2.0, 6.0, 7.0]),
        ("first", True, "bottom", [1.0, 2.0, 6.0, 3.0, 7.0, 4.0, 5.0]),
        ("first", False, "keep", [4.0, 5.0, np.nan, 3.0, np.nan, 1.0, 2.0]),
        ("first", False, "top", [6.0, 7.0, 1.0, 5.0, 2.0, 3.0, 4.0]),
        ("first", False, "bottom", [4.0, 5.0, 6.0, 3.0, 7.0, 1.0, 2.0]),
        ("dense", True, "keep", [1.0, 1.0, np.nan, 2.0, np.nan, 3.0, 3.0]),
        ("dense", True, "top", [2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 4.0]),
        ("dense", True, "bottom", [1.0, 1.0, 4.0, 2.0, 4.0, 3.0, 3.0]),
        ("dense", False, "keep", [3.0, 3.0, np.nan, 2.0, np.nan, 1.0, 1.0]),
        ("dense", False, "top", [4.0, 4.0, 1.0, 3.0, 1.0, 2.0, 2.0]),
        ("dense", False, "bottom", [3.0, 3.0, 4.0, 2.0, 4.0, 1.0, 1.0]),
    ],
)
def test_infs_n_nans(grps, vals, ties_method, ascending, na_option, exp):
    # GH 20561
    key = np.repeat(grps, len(vals))
    vals = vals * len(grps)
    df = DataFrame({"key": key, "val": vals})
    result = df.groupby("key").rank(
        method=ties_method, ascending=ascending, na_option=na_option
    )
    exp_df = DataFrame(exp * len(grps), columns=["val"])
    tm.assert_frame_equal(result, exp_df)


@pytest.mark.parametrize("grps", [["qux"], ["qux", "quux"]])
@pytest.mark.parametrize(
    "vals",
    [
        np.array([2, 2, np.nan, 8, 2, 6, np.nan, np.nan], dtype=dtype)
        for dtype in ["f8", "f4", "f2"]
    ]
    + [
        [
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-02"),
            np.nan,
            pd.Timestamp("2018-01-08"),
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-06"),
            np.nan,
            np.nan,
        ],
        [
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            np.nan,
            pd.Timestamp("2018-01-08", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-06", tz="US/Pacific"),
            np.nan,
            np.nan,
        ],
        [
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            np.nan,
            pd.Timestamp("2018-01-08") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-06") - pd.Timestamp(0),
            np.nan,
            np.nan,
        ],
        [
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            np.nan,
            pd.Timestamp("2018-01-08").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-06").to_period("D"),
            np.nan,
            np.nan,
        ],
    ],
    ids=lambda x: type(x[0]),
)
@pytest.mark.parametrize(
    "ties_method,ascending,na_option,pct,exp",
    [
        (
            "average",
            True,
            "keep",
            False,
            [2.0, 2.0, np.nan, 5.0, 2.0, 4.0, np.nan, np.nan],
        ),
        (
            "average",
            True,
            "keep",
            True,
            [0.4, 0.4, np.nan, 1.0, 0.4, 0.8, np.nan, np.nan],
        ),
        (
            "average",
            False,
            "keep",
            False,
            [4.0, 4.0, np.nan, 1.0, 4.0, 2.0, np.nan, np.nan],
        ),
        (
            "average",
            False,
            "keep",
            True,
            [0.8, 0.8, np.nan, 0.2, 0.8, 0.4, np.nan, np.nan],
        ),
        ("min", True, "keep", False, [1.0, 1.0, np.nan, 5.0, 1.0, 4.0, np.nan, np.nan]),
        ("min", True, "keep", True, [0.2, 0.2, np.nan, 1.0, 0.2, 0.8, np.nan, np.nan]),
        (
            "min",
            False,
            "keep",
            False,
            [3.0, 3.0, np.nan, 1.0, 3.0, 2.0, np.nan, np.nan],
        ),
        ("min", False, "keep", True, [0.6, 0.6, np.nan, 0.2, 0.6, 0.4, np.nan, np.nan]),
        ("max", True, "keep", False, [3.0, 3.0, np.nan, 5.0, 3.0, 4.0, np.nan, np.nan]),
        ("max", True, "keep", True, [0.6, 0.6, np.nan, 1.0, 0.6, 0.8, np.nan, np.nan]),
        (
            "max",
            False,
            "keep",
            False,
            [5.0, 5.0, np.nan, 1.0, 5.0, 2.0, np.nan, np.nan],
        ),
        ("max", False, "keep", True, [1.0, 1.0, np.nan, 0.2, 1.0, 0.4, np.nan, np.nan]),
        (
            "first",
            True,
            "keep",
            False,
            [1.0, 2.0, np.nan, 5.0, 3.0, 4.0, np.nan, np.nan],
        ),
        (
            "first",
            True,
            "keep",
            True,
            [0.2, 0.4, np.nan, 1.0, 0.6, 0.8, np.nan, np.nan],
        ),
        (
            "first",
            False,
            "keep",
            False,
            [3.0, 4.0, np.nan, 1.0, 5.0, 2.0, np.nan, np.nan],
        ),
        (
            "first",
            False,
            "keep",
            True,
            [0.6, 0.8, np.nan, 0.2, 1.0, 0.4, np.nan, np.nan],
        ),
        (
            "dense",
            True,
            "keep",
            False,
            [1.0, 1.0, np.nan, 3.0, 1.0, 2.0, np.nan, np.nan],
        ),
        (
            "dense",
            True,
            "keep",
            True,
            [
                1.0 / 3.0,
                1.0 / 3.0,
                np.nan,
                3.0 / 3.0,
                1.0 / 3.0,
                2.0 / 3.0,
                np.nan,
                np.nan,
            ],
        ),
        (
            "dense",
            False,
            "keep",
            False,
            [3.0, 3.0, np.nan, 1.0, 3.0, 2.0, np.nan, np.nan],
        ),
        (
            "dense",
            False,
            "keep",
            True,
            [
                3.0 / 3.0,
                3.0 / 3.0,
                np.nan,
                1.0 / 3.0,
                3.0 / 3.0,
                2.0 / 3.0,
                np.nan,
                np.nan,
            ],
        ),
        ("average", True, "bottom", False, [2.0, 2.0, 7.0, 5.0, 2.0, 4.0, 7.0, 7.0]),
        (
            "average",
            True,
            "bottom",
            True,
            [0.25, 0.25, 0.875, 0.625, 0.25, 0.5, 0.875, 0.875],
        ),
        ("average", False, "bottom", False, [4.0, 4.0, 7.0, 1.0, 4.0, 2.0, 7.0, 7.0]),
        (
            "average",
            False,
            "bottom",
            True,
            [0.5, 0.5, 0.875, 0.125, 0.5, 0.25, 0.875, 0.875],
        ),
        ("min", True, "bottom", False, [1.0, 1.0, 6.0, 5.0, 1.0, 4.0, 6.0, 6.0]),
        (
            "min",
            True,
            "bottom",
            True,
            [0.125, 0.125, 0.75, 0.625, 0.125, 0.5, 0.75, 0.75],
        ),
        ("min", False, "bottom", False, [3.0, 3.0, 6.0, 1.0, 3.0, 2.0, 6.0, 6.0]),
        (
            "min",
            False,
            "bottom",
            True,
            [0.375, 0.375, 0.75, 0.125, 0.375, 0.25, 0.75, 0.75],
        ),
        ("max", True, "bottom", False, [3.0, 3.0, 8.0, 5.0, 3.0, 4.0, 8.0, 8.0]),
        ("max", True, "bottom", True, [0.375, 0.375, 1.0, 0.625, 0.375, 0.5, 1.0, 1.0]),
        ("max", False, "bottom", False, [5.0, 5.0, 8.0, 1.0, 5.0, 2.0, 8.0, 8.0]),
        (
            "max",
            False,
            "bottom",
            True,
            [0.625, 0.625, 1.0, 0.125, 0.625, 0.25, 1.0, 1.0],
        ),
        ("first", True, "bottom", False, [1.0, 2.0, 6.0, 5.0, 3.0, 4.0, 7.0, 8.0]),
        (
            "first",
            True,
            "bottom",
            True,
            [0.125, 0.25, 0.75, 0.625, 0.375, 0.5, 0.875, 1.0],
        ),
        ("first", False, "bottom", False, [3.0, 4.0, 6.0, 1.0, 5.0, 2.0, 7.0, 8.0]),
        (
            "first",
            False,
            "bottom",
            True,
            [0.375, 0.5, 0.75, 0.125, 0.625, 0.25, 0.875, 1.0],
        ),
        ("dense", True, "bottom", False, [1.0, 1.0, 4.0, 3.0, 1.0, 2.0, 4.0, 4.0]),
        ("dense", True, "bottom", True, [0.25, 0.25, 1.0, 0.75, 0.25, 0.5, 1.0, 1.0]),
        ("dense", False, "bottom", False, [3.0, 3.0, 4.0, 1.0, 3.0, 2.0, 4.0, 4.0]),
        ("dense", False, "bottom", True, [0.75, 0.75, 1.0, 0.25, 0.75, 0.5, 1.0, 1.0]),
    ],
)
def test_rank_args_missing(grps, vals, ties_method, ascending, na_option, pct, exp):
    key = np.repeat(grps, len(vals))

    orig_vals = vals
    vals = list(vals) * len(grps)
    if isinstance(orig_vals, np.ndarray):
        vals = np.array(vals, dtype=orig_vals.dtype)

    df = DataFrame({"key": key, "val": vals})
    result = df.groupby("key").rank(
        method=ties_method, ascending=ascending, na_option=na_option, pct=pct
    )

    exp_df = DataFrame(exp * len(grps), columns=["val"])
    tm.assert_frame_equal(result, exp_df)


@pytest.mark.parametrize(
    "pct,exp", [(False, [3.0, 3.0, 3.0, 3.0, 3.0]), (True, [0.6, 0.6, 0.6, 0.6, 0.6])]
)
def test_rank_resets_each_group(pct, exp):
    df = DataFrame(
        {"key": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"], "val": [1] * 10}
    )
    result = df.groupby("key").rank(pct=pct)
    exp_df = DataFrame(exp * 2, columns=["val"])
    tm.assert_frame_equal(result, exp_df)


@pytest.mark.parametrize(
    "dtype", ["int64", "int32", "uint64", "uint32", "float64", "float32"]
)
@pytest.mark.parametrize("upper", [True, False])
def test_rank_avg_even_vals(dtype, upper):
    if upper:
        # use IntegerDtype/FloatingDtype
        dtype = dtype[0].upper() + dtype[1:]
        dtype = dtype.replace("Ui", "UI")
    df = DataFrame({"key": ["a"] * 4, "val": [1] * 4})
    df["val"] = df["val"].astype(dtype)
    assert df["val"].dtype == dtype

    result = df.groupby("key").rank()
    exp_df = DataFrame([2.5, 2.5, 2.5, 2.5], columns=["val"])
    if upper:
        exp_df = exp_df.astype("Float64")
    tm.assert_frame_equal(result, exp_df)


@pytest.mark.parametrize("ties_method", ["average", "min", "max", "first", "dense"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize(
    "vals", [["bar", "bar", "foo", "bar", "baz"], ["bar", np.nan, "foo", np.nan, "baz"]]
)
def test_rank_object_dtype(ties_method, ascending, na_option, pct, vals):
    df = DataFrame({"key": ["foo"] * 5, "val": vals})
    mask = df["val"].isna()

    gb = df.groupby("key")
    res = gb.rank(method=ties_method, ascending=ascending, na_option=na_option, pct=pct)

    # construct our expected by using numeric values with the same ordering
    if mask.any():
        df2 = DataFrame({"key": ["foo"] * 5, "val": [0, np.nan, 2, np.nan, 1]})
    else:
        df2 = DataFrame({"key": ["foo"] * 5, "val": [0, 0, 2, 0, 1]})

    gb2 = df2.groupby("key")
    alt = gb2.rank(
        method=ties_method, ascending=ascending, na_option=na_option, pct=pct
    )

    tm.assert_frame_equal(res, alt)


@pytest.mark.parametrize("na_option", [True, "bad", 1])
@pytest.mark.parametrize("ties_method", ["average", "min", "max", "first", "dense"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize(
    "vals",
    [
        ["bar", "bar", "foo", "bar", "baz"],
        ["bar", np.nan, "foo", np.nan, "baz"],
        [1, np.nan, 2, np.nan, 3],
    ],
)
def test_rank_naoption_raises(ties_method, ascending, na_option, pct, vals):
    df = DataFrame({"key": ["foo"] * 5, "val": vals})
    msg = "na_option must be one of 'keep', 'top', or 'bottom'"

    with pytest.raises(ValueError, match=msg):
        df.groupby("key").rank(
            method=ties_method, ascending=ascending, na_option=na_option, pct=pct
        )


def test_rank_empty_group():
    # see gh-22519
    column = "A"
    df = DataFrame({"A": [0, 1, 0], "B": [1.0, np.nan, 2.0]})

    result = df.groupby(column).B.rank(pct=True)
    expected = Series([0.5, np.nan, 1.0], name="B")
    tm.assert_series_equal(result, expected)

    result = df.groupby(column).rank(pct=True)
    expected = DataFrame({"B": [0.5, np.nan, 1.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "input_key,input_value,output_value",
    [
        ([1, 2], [1, 1], [1.0, 1.0]),
        ([1, 1, 2, 2], [1, 2, 1, 2], [0.5, 1.0, 0.5, 1.0]),
        ([1, 1, 2, 2], [1, 2, 1, np.nan], [0.5, 1.0, 1.0, np.nan]),
        ([1, 1, 2], [1, 2, np.nan], [0.5, 1.0, np.nan]),
    ],
)
def test_rank_zero_div(input_key, input_value, output_value):
    # GH 23666
    df = DataFrame({"A": input_key, "B": input_value})

    result = df.groupby("A").rank(method="dense", pct=True)
    expected = DataFrame({"B": output_value})
    tm.assert_frame_equal(result, expected)


def test_rank_min_int():
    # GH-32859
    df = DataFrame(
        {
            "grp": [1, 1, 2],
            "int_col": [
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
                np.iinfo(np.int64).min,
            ],
            "datetimelike": [NaT, datetime(2001, 1, 1), NaT],
        }
    )

    result = df.groupby("grp").rank()
    expected = DataFrame(
        {"int_col": [1.0, 2.0, 1.0], "datetimelike": [np.nan, 1.0, np.nan]}
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("use_nan", [True, False])
def test_rank_pct_equal_values_on_group_transition(use_nan):
    # GH#40518
    fill_value = np.nan if use_nan else 3
    df = DataFrame(
        [
            [-1, 1],
            [-1, 2],
            [1, fill_value],
            [-1, fill_value],
        ],
        columns=["group", "val"],
    )
    result = df.groupby(["group"])["val"].rank(
        method="dense",
        pct=True,
    )
    if use_nan:
        expected = Series([0.5, 1, np.nan, np.nan], name="val")
    else:
        expected = Series([1 / 3, 2 / 3, 1, 1], name="val")

    tm.assert_series_equal(result, expected)


def test_rank_multiindex():
    # GH27721
    df = concat(
        {
            "a": DataFrame({"col1": [3, 4], "col2": [1, 2]}),
            "b": DataFrame({"col3": [5, 6], "col4": [7, 8]}),
        },
        axis=1,
    )

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=0, axis=1)
    msg = "DataFrameGroupBy.rank with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = gb.rank(axis=1)

    expected = concat(
        [
            df["a"].rank(axis=1),
            df["b"].rank(axis=1),
        ],
        axis=1,
        keys=["a", "b"],
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_axis0_rank_axis1():
    # GH#41320
    df = DataFrame(
        {0: [1, 3, 5, 7], 1: [2, 4, 6, 8], 2: [1.5, 3.5, 5.5, 7.5]},
        index=["a", "a", "b", "b"],
    )
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=0, axis=0)

    msg = "DataFrameGroupBy.rank with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = gb.rank(axis=1)

    # This should match what we get when "manually" operating group-by-group
    expected = concat([df.loc["a"].rank(axis=1), df.loc["b"].rank(axis=1)], axis=0)
    tm.assert_frame_equal(res, expected)

    # check that we haven't accidentally written a case that coincidentally
    # matches rank(axis=0)
    msg = "The 'axis' keyword in DataFrameGroupBy.rank"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        alt = gb.rank(axis=0)
    assert not alt.equals(expected)


def test_groupby_axis0_cummax_axis1():
    # case where groupby axis is 0 and axis keyword in transform is 1

    # df has mixed dtype -> multiple blocks
    df = DataFrame(
        {0: [1, 3, 5, 7], 1: [2, 4, 6, 8], 2: [1.5, 3.5, 5.5, 7.5]},
        index=["a", "a", "b", "b"],
    )
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=0, axis=0)

    msg = "DataFrameGroupBy.cummax with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        cmax = gb.cummax(axis=1)
    expected = df[[0, 1]].astype(np.float64)
    expected[2] = expected[1]
    tm.assert_frame_equal(cmax, expected)


def test_non_unique_index():
    # GH 16577
    df = DataFrame(
        {"A": [1.0, 2.0, 3.0, np.nan], "value": 1.0},
        index=[pd.Timestamp("20170101", tz="US/Eastern")] * 4,
    )
    result = df.groupby([df.index, "A"]).value.rank(ascending=True, pct=True)
    expected = Series(
        [1.0, 1.0, 1.0, np.nan],
        index=[pd.Timestamp("20170101", tz="US/Eastern")] * 4,
        name="value",
    )
    tm.assert_series_equal(result, expected)


def test_rank_categorical():
    cat = pd.Categorical(["a", "a", "b", np.nan, "c", "b"], ordered=True)
    cat2 = pd.Categorical([1, 2, 3, np.nan, 4, 5], ordered=True)

    df = DataFrame({"col1": [0, 1, 0, 1, 0, 1], "col2": cat, "col3": cat2})

    gb = df.groupby("col1")

    res = gb.rank()

    expected = df.astype(object).groupby("col1").rank()
    tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize("na_option", ["top", "bottom"])
def test_groupby_op_with_nullables(na_option):
    # GH 54206
    df = DataFrame({"x": [None]}, dtype="Float64")
    result = df.groupby("x", dropna=False)["x"].rank(method="min", na_option=na_option)
    expected = Series([1.0], dtype="Float64", name=result.name)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\methods.py ===
import inspect
import operator

import numpy as np
import pytest

from pandas._typing import Dtype

from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort


class BaseMethodsTests:
    """Various Series and DataFrame methods."""

    def test_hash_pandas_object(self, data):
        # _hash_pandas_object should return a uint64 ndarray of the same length
        # as the data
        from pandas.core.util.hashing import _default_hash_key

        res = data._hash_pandas_object(
            encoding="utf-8", hash_key=_default_hash_key, categorize=False
        )
        assert res.dtype == np.uint64
        assert res.shape == data.shape

    def test_value_counts_default_dropna(self, data):
        # make sure we have consistent default dropna kwarg
        if not hasattr(data, "value_counts"):
            pytest.skip(f"value_counts is not implemented for {type(data)}")
        sig = inspect.signature(data.value_counts)
        kwarg = sig.parameters["dropna"]
        assert kwarg.default is True

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna):
        all_data = all_data[:10]
        if dropna:
            other = all_data[~all_data.isna()]
        else:
            other = all_data

        result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
        expected = pd.Series(other).value_counts(dropna=dropna).sort_index()

        tm.assert_series_equal(result, expected)

    def test_value_counts_with_normalize(self, data):
        # GH 33172
        data = data[:10].unique()
        values = np.array(data[~data.isna()])
        ser = pd.Series(data, dtype=data.dtype)

        result = ser.value_counts(normalize=True).sort_index()

        if not isinstance(data, pd.Categorical):
            expected = pd.Series(
                [1 / len(values)] * len(values), index=result.index, name="proportion"
            )
        else:
            expected = pd.Series(0.0, index=result.index, name="proportion")
            expected[result > 0] = 1 / len(values)

        if isinstance(data.dtype, pd.StringDtype) and data.dtype.na_value is np.nan:
            # TODO: avoid special-casing
            expected = expected.astype("float64")
        elif getattr(data.dtype, "storage", "") == "pyarrow" or isinstance(
            data.dtype, pd.ArrowDtype
        ):
            # TODO: avoid special-casing
            expected = expected.astype("double[pyarrow]")
        elif na_value_for_dtype(data.dtype) is pd.NA:
            # TODO(GH#44692): avoid special-casing
            expected = expected.astype("Float64")

        tm.assert_series_equal(result, expected)

    def test_count(self, data_missing):
        df = pd.DataFrame({"A": data_missing})
        result = df.count(axis="columns")
        expected = pd.Series([0, 1])
        tm.assert_series_equal(result, expected)

    def test_series_count(self, data_missing):
        # GH#26835
        ser = pd.Series(data_missing)
        result = ser.count()
        expected = 1
        assert result == expected

    def test_apply_simple_series(self, data):
        result = pd.Series(data).apply(id)
        assert isinstance(result, pd.Series)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action):
        result = data_missing.map(lambda x: x, na_action=na_action)
        expected = data_missing.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort(self, data_for_sorting):
        result = pd.Series(data_for_sorting).argsort()
        # argsort result gets passed to take, so should be np.intp
        expected = pd.Series(np.array([2, 0, 1], dtype=np.intp))
        tm.assert_series_equal(result, expected)

    def test_argsort_missing_array(self, data_missing_for_sorting):
        result = data_missing_for_sorting.argsort()
        # argsort result gets passed to take, so should be np.intp
        expected = np.array([2, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort_missing(self, data_missing_for_sorting):
        msg = "The behavior of Series.argsort in the presence of NA values"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pd.Series(data_missing_for_sorting).argsort()
        expected = pd.Series(np.array([1, -1, 0], dtype=np.intp))
        tm.assert_series_equal(result, expected)

    def test_argmin_argmax(self, data_for_sorting, data_missing_for_sorting, na_value):
        # GH 24382
        is_bool = data_for_sorting.dtype._is_boolean

        exp_argmax = 1
        exp_argmax_repeated = 3
        if is_bool:
            # See data_for_sorting docstring
            exp_argmax = 0
            exp_argmax_repeated = 1

        # data_for_sorting -> [B, C, A] with A < B < C
        assert data_for_sorting.argmax() == exp_argmax
        assert data_for_sorting.argmin() == 2

        # with repeated values -> first occurrence
        data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
        assert data.argmax() == exp_argmax_repeated
        assert data.argmin() == 0

        # with missing values
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        assert data_missing_for_sorting.argmax() == 0
        assert data_missing_for_sorting.argmin() == 2

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_empty_array(self, method, data):
        # GH 24382
        err_msg = "attempt to get"
        with pytest.raises(ValueError, match=err_msg):
            getattr(data[:0], method)()

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(self, method, data, na_value):
        # all missing with skipna=True is the same as empty
        err_msg = "attempt to get"
        data_na = type(data)._from_sequence([na_value, na_value], dtype=data.dtype)
        with pytest.raises(ValueError, match=err_msg):
            getattr(data_na, method)()

    @pytest.mark.parametrize(
        "op_name, skipna, expected",
        [
            ("idxmax", True, 0),
            ("idxmin", True, 2),
            ("argmax", True, 0),
            ("argmin", True, 2),
            ("idxmax", False, np.nan),
            ("idxmin", False, np.nan),
            ("argmax", False, -1),
            ("argmin", False, -1),
        ],
    )
    def test_argreduce_series(
        self, data_missing_for_sorting, op_name, skipna, expected
    ):
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        warn = None
        msg = "The behavior of Series.argmax/argmin"
        if op_name.startswith("arg") and expected == -1:
            warn = FutureWarning
        if op_name.startswith("idx") and np.isnan(expected):
            warn = FutureWarning
            msg = f"The behavior of Series.{op_name}"
        ser = pd.Series(data_missing_for_sorting)
        with tm.assert_produces_warning(warn, match=msg):
            result = getattr(ser, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting):
        # GH#38733
        data = data_missing_for_sorting

        with pytest.raises(NotImplementedError, match=""):
            data.argmin(skipna=False)

        with pytest.raises(NotImplementedError, match=""):
            data.argmax(skipna=False)

    @pytest.mark.parametrize(
        "na_position, expected",
        [
            ("last", np.array([2, 0, 1], dtype=np.dtype("intp"))),
            ("first", np.array([1, 2, 0], dtype=np.dtype("intp"))),
        ],
    )
    def test_nargsort(self, data_missing_for_sorting, na_position, expected):
        # GH 25439
        result = nargsort(data_missing_for_sorting, na_position=na_position)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        ser = pd.Series(data_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        expected = ser.iloc[[2, 0, 1]]
        if not ascending:
            # GH 35922. Expect stable sort
            if ser.nunique() == 2:
                expected = ser.iloc[[0, 1, 2]]
            else:
                expected = ser.iloc[[1, 0, 2]]

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self, data_missing_for_sorting, ascending, sort_by_key
    ):
        ser = pd.Series(data_missing_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        if ascending:
            expected = ser.iloc[[2, 0, 1]]
        else:
            expected = ser.iloc[[0, 2, 1]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending):
        df = pd.DataFrame({"A": [1, 2, 1], "B": data_for_sorting})
        result = df.sort_values(["A", "B"])
        expected = pd.DataFrame(
            {"A": [1, 1, 2], "B": data_for_sorting.take([2, 0, 1])}, index=[2, 0, 1]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("keep", ["first", "last", False])
    def test_duplicated(self, data, keep):
        arr = data.take([0, 1, 0, 1])
        result = arr.duplicated(keep=keep)
        if keep == "first":
            expected = np.array([False, False, True, True])
        elif keep == "last":
            expected = np.array([True, True, False, False])
        else:
            expected = np.array([True, True, True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        duplicated = box(data._from_sequence([data[0], data[0]], dtype=data.dtype))

        result = method(duplicated)

        assert len(result) == 1
        assert isinstance(result, type(data))
        assert result[0] == duplicated[0]

    def test_factorize(self, data_for_grouping):
        codes, uniques = pd.factorize(data_for_grouping, use_na_sentinel=True)

        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 0], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4])
        else:
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 2], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4, 7])

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_extension_array_equal(uniques, expected_uniques)

    def test_factorize_equivalence(self, data_for_grouping):
        codes_1, uniques_1 = pd.factorize(data_for_grouping, use_na_sentinel=True)
        codes_2, uniques_2 = data_for_grouping.factorize(use_na_sentinel=True)

        tm.assert_numpy_array_equal(codes_1, codes_2)
        tm.assert_extension_array_equal(uniques_1, uniques_2)
        assert len(uniques_1) == len(pd.unique(uniques_1))
        assert uniques_1.dtype == data_for_grouping.dtype

    def test_factorize_empty(self, data):
        codes, uniques = pd.factorize(data[:0])
        expected_codes = np.array([], dtype=np.intp)
        expected_uniques = type(data)._from_sequence([], dtype=data[:0].dtype)

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_extension_array_equal(uniques, expected_uniques)

    def test_fillna_copy_frame(self, data_missing):
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr})
        df_orig = df.copy()

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        result.iloc[0, 0] = filled_val

        tm.assert_frame_equal(df, df_orig)

    def test_fillna_copy_series(self, data_missing):
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr, copy=False)
        ser_orig = ser.copy()

        filled_val = ser[0]
        result = ser.fillna(filled_val)
        result.iloc[0] = filled_val

        tm.assert_series_equal(ser, ser_orig)

    def test_fillna_length_mismatch(self, data_missing):
        msg = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg):
            data_missing.fillna(data_missing.take([1]))

    # Subclasses can override if we expect e.g Sparse[bool], boolean, pyarrow[bool]
    _combine_le_expected_dtype: Dtype = NumpyEADtype("bool")

    def test_combine_le(self, data_repeated):
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            pd.array(
                [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))],
                dtype=self._combine_le_expected_dtype,
            )
        )
        tm.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            pd.array(
                [a <= val for a in list(orig_data1)],
                dtype=self._combine_le_expected_dtype,
            )
        )
        tm.assert_series_equal(result, expected)

    def test_combine_add(self, data_repeated):
        # GH 20825
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)

        # Check if the operation is supported pointwise for our scalars. If not,
        #  we will expect Series.combine to raise as well.
        try:
            with np.errstate(over="ignore"):
                expected = pd.Series(
                    orig_data1._from_sequence(
                        [a + b for (a, b) in zip(list(orig_data1), list(orig_data2))]
                    )
                )
        except TypeError:
            # If the operation is not supported pointwise for our scalars,
            #  then Series.combine should also raise
            with pytest.raises(TypeError):
                s1.combine(s2, lambda x1, x2: x1 + x2)
            return

        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        tm.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        expected = pd.Series(
            orig_data1._from_sequence([a + val for a in list(orig_data1)])
        )
        tm.assert_series_equal(result, expected)

    def test_combine_first(self, data):
        # https://github.com/pandas-dev/pandas/issues/24147
        a = pd.Series(data[:3])
        b = pd.Series(data[2:5], index=[2, 3, 4])
        result = a.combine_first(b)
        expected = pd.Series(data[:5])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("frame", [True, False])
    @pytest.mark.parametrize(
        "periods, indices",
        [(-2, [2, 3, 4, -1, -1]), (0, [0, 1, 2, 3, 4]), (2, [-1, -1, 0, 1, 2])],
    )
    def test_container_shift(self, data, frame, periods, indices):
        # https://github.com/pandas-dev/pandas/issues/22386
        subset = data[:5]
        data = pd.Series(subset, name="A")
        expected = pd.Series(subset.take(indices, allow_fill=True), name="A")

        if frame:
            result = data.to_frame(name="A").assign(B=1).shift(periods)
            expected = pd.concat(
                [expected, pd.Series([1] * 5, name="B").shift(periods)], axis=1
            )
            compare = tm.assert_frame_equal
        else:
            result = data.shift(periods)
            compare = tm.assert_series_equal

        compare(result, expected)

    def test_shift_0_periods(self, data):
        # GH#33856 shifting with periods=0 should return a copy, not same obj
        result = data.shift(0)
        assert data[0] != data[1]  # otherwise below is invalid
        data[0] = data[1]
        assert result[0] != result[1]  # i.e. not the same object/view

    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods):
        data = data[:5]
        if is_bool_dtype(data.dtype):
            op = operator.xor
        else:
            op = operator.sub
        try:
            # does this array implement ops?
            op(data, data)
        except Exception:
            pytest.skip(f"{type(data)} does not support diff")
        s = pd.Series(data)
        result = s.diff(periods)
        expected = pd.Series(op(data, data.shift(periods)))
        tm.assert_series_equal(result, expected)

        df = pd.DataFrame({"A": data, "B": [1.0] * 5})
        result = df.diff(periods)
        if periods == 1:
            b = [np.nan, 0, 0, 0, 0]
        else:
            b = [0, 0, 0, np.nan, np.nan]
        expected = pd.DataFrame({"A": expected, "B": b})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, indices",
        [[-4, [-1, -1]], [-1, [1, -1]], [0, [0, 1]], [1, [-1, 0]], [4, [-1, -1]]],
    )
    def test_shift_non_empty_array(self, data, periods, indices):
        # https://github.com/pandas-dev/pandas/issues/23911
        subset = data[:2]
        result = subset.shift(periods)
        expected = subset.take(indices, allow_fill=True)
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("periods", [-4, -1, 0, 1, 4])
    def test_shift_empty_array(self, data, periods):
        # https://github.com/pandas-dev/pandas/issues/23911
        empty = data[:0]
        result = empty.shift(periods)
        expected = empty
        tm.assert_extension_array_equal(result, expected)

    def test_shift_zero_copies(self, data):
        # GH#31502
        result = data.shift(0)
        assert result is not data

        result = data[:0].shift(2)
        assert result is not data

    def test_shift_fill_value(self, data):
        arr = data[:4]
        fill_value = data[0]
        result = arr.shift(1, fill_value=fill_value)
        expected = data.take([0, 0, 1, 2])
        tm.assert_extension_array_equal(result, expected)

        result = arr.shift(-2, fill_value=fill_value)
        expected = data.take([2, 3, 0, 0])
        tm.assert_extension_array_equal(result, expected)

    def test_not_hashable(self, data):
        # We are in general mutable, so not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(data)

    def test_hash_pandas_object_works(self, data, as_frame):
        # https://github.com/pandas-dev/pandas/issues/23066
        data = pd.Series(data)
        if as_frame:
            data = data.to_frame()
        a = pd.util.hash_pandas_object(data)
        b = pd.util.hash_pandas_object(data)
        tm.assert_equal(a, b)

    def test_searchsorted(self, data_for_sorting, as_series):
        if data_for_sorting.dtype._is_boolean:
            return self._test_searchsorted_bool_dtypes(data_for_sorting, as_series)

        b, c, a = data_for_sorting
        arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]

        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        assert arr.searchsorted(c) == 2
        assert arr.searchsorted(c, side="right") == 3

        result = arr.searchsorted(arr.take([0, 2]))
        expected = np.array([0, 2], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)

        # sorter
        sorter = np.array([1, 2, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    def _test_searchsorted_bool_dtypes(self, data_for_sorting, as_series):
        # We call this from test_searchsorted in cases where we have a
        #  boolean-like dtype. The non-bool test assumes we have more than 2
        #  unique values.
        dtype = data_for_sorting.dtype
        data_for_sorting = pd.array([True, False], dtype=dtype)
        b, a = data_for_sorting
        arr = type(data_for_sorting)._from_sequence([a, b])

        if as_series:
            arr = pd.Series(arr)
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        result = arr.searchsorted(arr.take([0, 1]))
        expected = np.array([0, 1], dtype=np.intp)

        tm.assert_numpy_array_equal(result, expected)

        # sorter
        sorter = np.array([1, 0])
        assert data_for_sorting.searchsorted(a, sorter=sorter) == 0

    def test_where_series(self, data, na_value, as_frame):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        orig = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))
        ser = orig.copy()
        cond = np.array([True, True, False, False])

        if as_frame:
            ser = ser.to_frame(name="a")
            cond = cond.reshape(-1, 1)

        result = ser.where(cond)
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=data.dtype)
        )

        if as_frame:
            expected = expected.to_frame(name="a")
        tm.assert_equal(result, expected)

        ser.mask(~cond, inplace=True)
        tm.assert_equal(ser, expected)

        # array other
        ser = orig.copy()
        if as_frame:
            ser = ser.to_frame(name="a")
        cond = np.array([True, False, True, True])
        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        if as_frame:
            other = pd.DataFrame({"a": other})
            cond = pd.DataFrame({"a": cond})
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        if as_frame:
            expected = expected.to_frame(name="a")
        tm.assert_equal(result, expected)

        ser.mask(~cond, other, inplace=True)
        tm.assert_equal(ser, expected)

    @pytest.mark.parametrize("repeats", [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy):
        arr = type(data)._from_sequence(data[:3], dtype=data.dtype)
        if as_series:
            arr = pd.Series(arr)

        result = np.repeat(arr, repeats) if use_numpy else arr.repeat(repeats)

        repeats = [repeats] * 3 if isinstance(repeats, int) else repeats
        expected = [x for x, n in zip(arr, repeats) for _ in range(n)]
        expected = type(data)._from_sequence(expected, dtype=data.dtype)
        if as_series:
            expected = pd.Series(expected, index=arr.index.repeat(repeats))

        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "repeats, kwargs, error, msg",
        [
            (2, {"axis": 1}, ValueError, "axis"),
            (-1, {}, ValueError, "negative"),
            ([1, 2], {}, ValueError, "shape"),
            (2, {"foo": "bar"}, TypeError, "'foo'"),
        ],
    )
    def test_repeat_raises(self, data, repeats, kwargs, error, msg, use_numpy):
        with pytest.raises(error, match=msg):
            if use_numpy:
                np.repeat(data, repeats, **kwargs)
            else:
                data.repeat(repeats, **kwargs)

    def test_delete(self, data):
        result = data.delete(0)
        expected = data[1:]
        tm.assert_extension_array_equal(result, expected)

        result = data.delete([1, 3])
        expected = data._concat_same_type([data[[0]], data[[2]], data[4:]])
        tm.assert_extension_array_equal(result, expected)

    def test_insert(self, data):
        # insert at the beginning
        result = data[1:].insert(0, data[0])
        tm.assert_extension_array_equal(result, data)

        result = data[1:].insert(-len(data[1:]), data[0])
        tm.assert_extension_array_equal(result, data)

        # insert at the middle
        result = data[:-1].insert(4, data[-1])

        taker = np.arange(len(data))
        taker[5:] = taker[4:-1]
        taker[4] = len(data) - 1
        expected = data.take(taker)
        tm.assert_extension_array_equal(result, expected)

    def test_insert_invalid(self, data, invalid_scalar):
        item = invalid_scalar

        with pytest.raises((TypeError, ValueError)):
            data.insert(0, item)

        with pytest.raises((TypeError, ValueError)):
            data.insert(4, item)

        with pytest.raises((TypeError, ValueError)):
            data.insert(len(data) - 1, item)

    def test_insert_invalid_loc(self, data):
        ub = len(data)

        with pytest.raises(IndexError):
            data.insert(ub + 1, data[0])

        with pytest.raises(IndexError):
            data.insert(-ub - 1, data[0])

        with pytest.raises(TypeError):
            # we expect TypeError here instead of IndexError to match np.insert
            data.insert(1.5, data[0])

    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        data2 = type(data)._from_sequence([data[0]] * len(data), dtype=data.dtype)
        data_na = type(data)._from_sequence([na_value] * len(data), dtype=data.dtype)

        data = tm.box_expected(data, box, transpose=False)
        data2 = tm.box_expected(data2, box, transpose=False)
        data_na = tm.box_expected(data_na, box, transpose=False)

        # we are asserting with `is True/False` explicitly, to test that the
        # result is an actual Python bool, and not something "truthy"

        assert data.equals(data) is True
        assert data.equals(data.copy()) is True

        # unequal other data
        assert data.equals(data2) is False
        assert data.equals(data_na) is False

        # different length
        assert data[:2].equals(data[:3]) is False

        # empty are equal
        assert data[:0].equals(data[:0]) is True

        # other types
        assert data.equals(None) is False
        assert data[[0]].equals(data[0]) is False

    def test_equals_same_data_different_object(self, data):
        # https://github.com/pandas-dev/pandas/issues/34660
        assert pd.Series(data).equals(pd.Series(data))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_misc.py ===
""" Test cases for misc plot functions """
import os

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
    interval_range,
    period_range,
    plotting,
    read_csv,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_colors,
    _check_legend_labels,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
)

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
cm = pytest.importorskip("matplotlib.cm")


@pytest.fixture
def iris(datapath) -> DataFrame:
    """
    The iris dataset as a DataFrame.
    """
    return read_csv(datapath("io", "data", "csv", "iris.csv"))


@td.skip_if_installed("matplotlib")
def test_import_error_message():
    # GH-19810
    df = DataFrame({"A": [1, 2]})

    with pytest.raises(ImportError, match="matplotlib is required for plotting"):
        df.plot()


def test_get_accessor_args():
    func = plotting._core.PlotAccessor._get_call_args

    msg = "Called plot accessor for type list, expected Series or DataFrame"
    with pytest.raises(TypeError, match=msg):
        func(backend_name="", data=[], args=[], kwargs={})

    msg = "should not be called with positional arguments"
    with pytest.raises(TypeError, match=msg):
        func(backend_name="", data=Series(dtype=object), args=["line", None], kwargs={})

    x, y, kind, kwargs = func(
        backend_name="",
        data=DataFrame(),
        args=["x"],
        kwargs={"y": "y", "kind": "bar", "grid": False},
    )
    assert x == "x"
    assert y == "y"
    assert kind == "bar"
    assert kwargs == {"grid": False}

    x, y, kind, kwargs = func(
        backend_name="pandas.plotting._matplotlib",
        data=Series(dtype=object),
        args=[],
        kwargs={},
    )
    assert x is None
    assert y is None
    assert kind == "line"
    assert len(kwargs) == 24


@pytest.mark.parametrize("kind", plotting.PlotAccessor._all_kinds)
@pytest.mark.parametrize(
    "data", [DataFrame(np.arange(15).reshape(5, 3)), Series(range(5))]
)
@pytest.mark.parametrize(
    "index",
    [
        Index(range(5)),
        date_range("2020-01-01", periods=5),
        period_range("2020-01-01", periods=5),
    ],
)
def test_savefig(kind, data, index):
    fig, ax = plt.subplots()
    data.index = index
    kwargs = {}
    if kind in ["hexbin", "scatter", "pie"]:
        if isinstance(data, Series):
            pytest.skip(f"{kind} not supported with Series")
        kwargs = {"x": 0, "y": 1}
    data.plot(kind=kind, ax=ax, **kwargs)
    fig.savefig(os.devnull)


class TestSeriesPlots:
    def test_autocorrelation_plot(self):
        from pandas.plotting import autocorrelation_plot

        ser = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        # Ensure no UserWarning when making plot
        with tm.assert_produces_warning(None):
            _check_plot_works(autocorrelation_plot, series=ser)
            _check_plot_works(autocorrelation_plot, series=ser.values)

            ax = autocorrelation_plot(ser, label="Test")
        _check_legend_labels(ax, labels=["Test"])

    @pytest.mark.parametrize("kwargs", [{}, {"lag": 5}])
    def test_lag_plot(self, kwargs):
        from pandas.plotting import lag_plot

        ser = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        _check_plot_works(lag_plot, series=ser, **kwargs)

    def test_bootstrap_plot(self):
        from pandas.plotting import bootstrap_plot

        ser = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        _check_plot_works(bootstrap_plot, series=ser, size=10)


class TestDataFramePlots:
    @pytest.mark.parametrize("pass_axis", [False, True])
    def test_scatter_matrix_axis(self, pass_axis):
        pytest.importorskip("scipy")
        scatter_matrix = plotting.scatter_matrix

        ax = None
        if pass_axis:
            _, ax = mpl.pyplot.subplots(3, 3)

        df = DataFrame(np.random.default_rng(2).standard_normal((100, 3)))

        # we are plotting multiples on a sub-plot
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(
                scatter_matrix,
                frame=df,
                range_padding=0.1,
                ax=ax,
            )
        axes0_labels = axes[0][0].yaxis.get_majorticklabels()
        # GH 5662
        expected = ["-2", "0", "2"]
        _check_text_labels(axes0_labels, expected)
        _check_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    @pytest.mark.parametrize("pass_axis", [False, True])
    def test_scatter_matrix_axis_smaller(self, pass_axis):
        pytest.importorskip("scipy")
        scatter_matrix = plotting.scatter_matrix

        ax = None
        if pass_axis:
            _, ax = mpl.pyplot.subplots(3, 3)

        df = DataFrame(np.random.default_rng(11).standard_normal((100, 3)))
        df[0] = (df[0] - 2) / 3

        # we are plotting multiples on a sub-plot
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(
                scatter_matrix,
                frame=df,
                range_padding=0.1,
                ax=ax,
            )
        axes0_labels = axes[0][0].yaxis.get_majorticklabels()
        expected = ["-1.0", "-0.5", "0.0"]
        _check_text_labels(axes0_labels, expected)
        _check_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    @pytest.mark.slow
    def test_andrews_curves_no_warning(self, iris):
        from pandas.plotting import andrews_curves

        df = iris
        # Ensure no UserWarning when making plot
        with tm.assert_produces_warning(None):
            _check_plot_works(andrews_curves, frame=df, class_column="Name")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "linecolors",
        [
            ("#556270", "#4ECDC4", "#C7F464"),
            ["dodgerblue", "aquamarine", "seagreen"],
        ],
    )
    @pytest.mark.parametrize(
        "df",
        [
            "iris",
            DataFrame(
                {
                    "A": np.random.default_rng(2).standard_normal(10),
                    "B": np.random.default_rng(2).standard_normal(10),
                    "C": np.random.default_rng(2).standard_normal(10),
                    "Name": ["A"] * 10,
                }
            ),
        ],
    )
    def test_andrews_curves_linecolors(self, request, df, linecolors):
        from pandas.plotting import andrews_curves

        if isinstance(df, str):
            df = request.getfixturevalue(df)
        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", color=linecolors
        )
        _check_colors(
            ax.get_lines()[:10], linecolors=linecolors, mapping=df["Name"][:10]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "df",
        [
            "iris",
            DataFrame(
                {
                    "A": np.random.default_rng(2).standard_normal(10),
                    "B": np.random.default_rng(2).standard_normal(10),
                    "C": np.random.default_rng(2).standard_normal(10),
                    "Name": ["A"] * 10,
                }
            ),
        ],
    )
    def test_andrews_curves_cmap(self, request, df):
        from pandas.plotting import andrews_curves

        if isinstance(df, str):
            df = request.getfixturevalue(df)
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        ax = _check_plot_works(
            andrews_curves, frame=df, class_column="Name", color=cmaps
        )
        _check_colors(ax.get_lines()[:10], linecolors=cmaps, mapping=df["Name"][:10])

    @pytest.mark.slow
    def test_andrews_curves_handle(self):
        from pandas.plotting import andrews_curves

        colors = ["b", "g", "r"]
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "Name": colors})
        ax = andrews_curves(df, "Name", color=colors)
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, linecolors=colors)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "color",
        [("#556270", "#4ECDC4", "#C7F464"), ["dodgerblue", "aquamarine", "seagreen"]],
    )
    def test_parallel_coordinates_colors(self, iris, color):
        from pandas.plotting import parallel_coordinates

        df = iris

        ax = _check_plot_works(
            parallel_coordinates, frame=df, class_column="Name", color=color
        )
        _check_colors(ax.get_lines()[:10], linecolors=color, mapping=df["Name"][:10])

    @pytest.mark.slow
    def test_parallel_coordinates_cmap(self, iris):
        from matplotlib import cm

        from pandas.plotting import parallel_coordinates

        df = iris

        ax = _check_plot_works(
            parallel_coordinates, frame=df, class_column="Name", colormap=cm.jet
        )
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        _check_colors(ax.get_lines()[:10], linecolors=cmaps, mapping=df["Name"][:10])

    @pytest.mark.slow
    def test_parallel_coordinates_line_diff(self, iris):
        from pandas.plotting import parallel_coordinates

        df = iris

        ax = _check_plot_works(parallel_coordinates, frame=df, class_column="Name")
        nlines = len(ax.get_lines())
        nxticks = len(ax.xaxis.get_ticklabels())

        ax = _check_plot_works(
            parallel_coordinates, frame=df, class_column="Name", axvlines=False
        )
        assert len(ax.get_lines()) == (nlines - nxticks)

    @pytest.mark.slow
    def test_parallel_coordinates_handles(self, iris):
        from pandas.plotting import parallel_coordinates

        df = iris
        colors = ["b", "g", "r"]
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "Name": colors})
        ax = parallel_coordinates(df, "Name", color=colors)
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, linecolors=colors)

    # not sure if this is indicative of a problem
    @pytest.mark.filterwarnings("ignore:Attempting to set:UserWarning")
    def test_parallel_coordinates_with_sorted_labels(self):
        """For #15908"""
        from pandas.plotting import parallel_coordinates

        df = DataFrame(
            {
                "feat": list(range(30)),
                "class": [2 for _ in range(10)]
                + [3 for _ in range(10)]
                + [1 for _ in range(10)],
            }
        )
        ax = parallel_coordinates(df, "class", sort_labels=True)
        polylines, labels = ax.get_legend_handles_labels()
        color_label_tuples = zip(
            [polyline.get_color() for polyline in polylines], labels
        )
        ordered_color_label_tuples = sorted(color_label_tuples, key=lambda x: x[1])
        prev_next_tupels = zip(
            list(ordered_color_label_tuples[0:-1]), list(ordered_color_label_tuples[1:])
        )
        for prev, nxt in prev_next_tupels:
            # labels and colors are ordered strictly increasing
            assert prev[1] < nxt[1] and prev[0] < nxt[0]

    def test_radviz_no_warning(self, iris):
        from pandas.plotting import radviz

        df = iris
        # Ensure no UserWarning when making plot
        with tm.assert_produces_warning(None):
            _check_plot_works(radviz, frame=df, class_column="Name")

    @pytest.mark.parametrize(
        "color",
        [("#556270", "#4ECDC4", "#C7F464"), ["dodgerblue", "aquamarine", "seagreen"]],
    )
    def test_radviz_color(self, iris, color):
        from pandas.plotting import radviz

        df = iris
        ax = _check_plot_works(radviz, frame=df, class_column="Name", color=color)
        # skip Circle drawn as ticks
        patches = [p for p in ax.patches[:20] if p.get_label() != ""]
        _check_colors(patches[:10], facecolors=color, mapping=df["Name"][:10])

    def test_radviz_color_cmap(self, iris):
        from matplotlib import cm

        from pandas.plotting import radviz

        df = iris
        ax = _check_plot_works(radviz, frame=df, class_column="Name", colormap=cm.jet)
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        patches = [p for p in ax.patches[:20] if p.get_label() != ""]
        _check_colors(patches, facecolors=cmaps, mapping=df["Name"][:10])

    def test_radviz_colors_handles(self):
        from pandas.plotting import radviz

        colors = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        df = DataFrame(
            {"A": [1, 2, 3], "B": [2, 1, 3], "C": [3, 2, 1], "Name": ["b", "g", "r"]}
        )
        ax = radviz(df, "Name", color=colors)
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=colors)

    def test_subplot_titles(self, iris):
        df = iris.drop("Name", axis=1).head()
        # Use the column names as the subplot titles
        title = list(df.columns)

        # Case len(title) == len(df)
        plot = df.plot(subplots=True, title=title)
        assert [p.get_title() for p in plot] == title

    def test_subplot_titles_too_much(self, iris):
        df = iris.drop("Name", axis=1).head()
        # Use the column names as the subplot titles
        title = list(df.columns)
        # Case len(title) > len(df)
        msg = (
            "The length of `title` must equal the number of columns if "
            "using `title` of type `list` and `subplots=True`"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, title=title + ["kittens > puppies"])

    def test_subplot_titles_too_little(self, iris):
        df = iris.drop("Name", axis=1).head()
        # Use the column names as the subplot titles
        title = list(df.columns)
        msg = (
            "The length of `title` must equal the number of columns if "
            "using `title` of type `list` and `subplots=True`"
        )
        # Case len(title) < len(df)
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, title=title[:2])

    def test_subplot_titles_subplots_false(self, iris):
        df = iris.drop("Name", axis=1).head()
        # Use the column names as the subplot titles
        title = list(df.columns)
        # Case subplots=False and title is of type list
        msg = (
            "Using `title` of type `list` is not supported unless "
            "`subplots=True` is passed"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=False, title=title)

    def test_subplot_titles_numeric_square_layout(self, iris):
        df = iris.drop("Name", axis=1).head()
        # Use the column names as the subplot titles
        title = list(df.columns)
        # Case df with 3 numeric columns but layout of (2,2)
        plot = df.drop("SepalWidth", axis=1).plot(
            subplots=True, layout=(2, 2), title=title[:-1]
        )
        title_list = [ax.get_title() for sublist in plot for ax in sublist]
        assert title_list == title[:3] + [""]

    def test_get_standard_colors_random_seed(self):
        # GH17525
        df = DataFrame(np.zeros((10, 10)))

        # Make sure that the random seed isn't reset by get_standard_colors
        plotting.parallel_coordinates(df, 0)
        rand1 = np.random.default_rng(None).random()
        plotting.parallel_coordinates(df, 0)
        rand2 = np.random.default_rng(None).random()
        assert rand1 != rand2

    def test_get_standard_colors_consistency(self):
        # GH17525
        # Make sure it produces the same colors every time it's called
        from pandas.plotting._matplotlib.style import get_standard_colors

        color1 = get_standard_colors(1, color_type="random")
        color2 = get_standard_colors(1, color_type="random")
        assert color1 == color2

    def test_get_standard_colors_default_num_colors(self):
        from pandas.plotting._matplotlib.style import get_standard_colors

        # Make sure the default color_types returns the specified amount
        color1 = get_standard_colors(1, color_type="default")
        color2 = get_standard_colors(9, color_type="default")
        color3 = get_standard_colors(20, color_type="default")
        assert len(color1) == 1
        assert len(color2) == 9
        assert len(color3) == 20

    def test_plot_single_color(self):
        # Example from #20585. All 3 bars should have the same color
        df = DataFrame(
            {
                "account-start": ["2017-02-03", "2017-03-03", "2017-01-01"],
                "client": ["Alice Anders", "Bob Baker", "Charlie Chaplin"],
                "balance": [-1432.32, 10.43, 30000.00],
                "db-id": [1234, 2424, 251],
                "proxy-id": [525, 1525, 2542],
                "rank": [52, 525, 32],
            }
        )
        ax = df.client.value_counts().plot.bar()
        colors = [rect.get_facecolor() for rect in ax.get_children()[0:3]]
        assert all(color == colors[0] for color in colors)

    def test_get_standard_colors_no_appending(self):
        # GH20726

        # Make sure not to add more colors so that matplotlib can cycle
        # correctly.
        from matplotlib import cm

        from pandas.plotting._matplotlib.style import get_standard_colors

        color_before = cm.gnuplot(range(5))
        color_after = get_standard_colors(1, color=color_before)
        assert len(color_after) == len(color_before)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((48, 4)), columns=list("ABCD")
        )

        color_list = cm.gnuplot(np.linspace(0, 1, 16))
        p = df.A.plot.bar(figsize=(16, 7), color=color_list)
        assert p.patches[1].get_facecolor() == p.patches[17].get_facecolor()

    @pytest.mark.parametrize("kind", ["bar", "line"])
    def test_dictionary_color(self, kind):
        # issue-8193
        # Test plot color dictionary format
        data_files = ["a", "b"]

        expected = [(0.5, 0.24, 0.6), (0.3, 0.7, 0.7)]

        df1 = DataFrame(np.random.default_rng(2).random((2, 2)), columns=data_files)
        dic_color = {"b": (0.3, 0.7, 0.7), "a": (0.5, 0.24, 0.6)}

        ax = df1.plot(kind=kind, color=dic_color)
        if kind == "bar":
            colors = [rect.get_facecolor()[0:-1] for rect in ax.get_children()[0:3:2]]
        else:
            colors = [rect.get_color() for rect in ax.get_lines()[0:2]]
        assert all(color == expected[index] for index, color in enumerate(colors))

    def test_bar_plot(self):
        # GH38947
        # Test bar plot with string and int index
        from matplotlib.text import Text

        expected = [Text(0, 0, "0"), Text(1, 0, "Total")]

        df = DataFrame(
            {
                "a": [1, 2],
            },
            index=Index([0, "Total"]),
        )
        plot_bar = df.plot.bar()
        assert all(
            (a.get_text() == b.get_text())
            for a, b in zip(plot_bar.get_xticklabels(), expected)
        )

    def test_barh_plot_labels_mixed_integer_string(self):
        # GH39126
        # Test barh plot with string and integer at the same column
        from matplotlib.text import Text

        df = DataFrame([{"word": 1, "value": 0}, {"word": "knowledge", "value": 2}])
        plot_barh = df.plot.barh(x="word", legend=None)
        expected_yticklabels = [Text(0, 0, "1"), Text(0, 1, "knowledge")]
        assert all(
            actual.get_text() == expected.get_text()
            for actual, expected in zip(
                plot_barh.get_yticklabels(), expected_yticklabels
            )
        )

    def test_has_externally_shared_axis_x_axis(self):
        # GH33819
        # Test _has_externally_shared_axis() works for x-axis
        func = plotting._matplotlib.tools._has_externally_shared_axis

        fig = mpl.pyplot.figure()
        plots = fig.subplots(2, 4)

        # Create *externally* shared axes for first and third columns
        plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
        plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])

        # Create *internally* shared axes for second and third columns
        plots[0][1].twinx()
        plots[0][2].twinx()

        # First  column is only externally shared
        # Second column is only internally shared
        # Third  column is both
        # Fourth column is neither
        assert func(plots[0][0], "x")
        assert not func(plots[0][1], "x")
        assert func(plots[0][2], "x")
        assert not func(plots[0][3], "x")

    def test_has_externally_shared_axis_y_axis(self):
        # GH33819
        # Test _has_externally_shared_axis() works for y-axis
        func = plotting._matplotlib.tools._has_externally_shared_axis

        fig = mpl.pyplot.figure()
        plots = fig.subplots(4, 2)

        # Create *externally* shared axes for first and third rows
        plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])
        plots[2][0] = fig.add_subplot(325, sharey=plots[2][1])

        # Create *internally* shared axes for second and third rows
        plots[1][0].twiny()
        plots[2][0].twiny()

        # First  row is only externally shared
        # Second row is only internally shared
        # Third  row is both
        # Fourth row is neither
        assert func(plots[0][0], "y")
        assert not func(plots[1][0], "y")
        assert func(plots[2][0], "y")
        assert not func(plots[3][0], "y")

    def test_has_externally_shared_axis_invalid_compare_axis(self):
        # GH33819
        # Test _has_externally_shared_axis() raises an exception when
        # passed an invalid value as compare_axis parameter
        func = plotting._matplotlib.tools._has_externally_shared_axis

        fig = mpl.pyplot.figure()
        plots = fig.subplots(4, 2)

        # Create arbitrary axes
        plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])

        # Check that an invalid compare_axis value triggers the expected exception
        msg = "needs 'x' or 'y' as a second parameter"
        with pytest.raises(ValueError, match=msg):
            func(plots[0][0], "z")

    def test_externally_shared_axes(self):
        # Example from GH33819
        # Create data
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(1000),
                "b": np.random.default_rng(2).standard_normal(1000),
            }
        )

        # Create figure
        fig = mpl.pyplot.figure()
        plots = fig.subplots(2, 3)

        # Create *externally* shared axes
        plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
        # note: no plots[0][1] that's the twin only case
        plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])

        # Create *internally* shared axes
        # note: no plots[0][0] that's the external only case
        twin_ax1 = plots[0][1].twinx()
        twin_ax2 = plots[0][2].twinx()

        # Plot data to primary axes
        df["a"].plot(ax=plots[0][0], title="External share only").set_xlabel(
            "this label should never be visible"
        )
        df["a"].plot(ax=plots[1][0])

        df["a"].plot(ax=plots[0][1], title="Internal share (twin) only").set_xlabel(
            "this label should always be visible"
        )
        df["a"].plot(ax=plots[1][1])

        df["a"].plot(ax=plots[0][2], title="Both").set_xlabel(
            "this label should never be visible"
        )
        df["a"].plot(ax=plots[1][2])

        # Plot data to twinned axes
        df["b"].plot(ax=twin_ax1, color="green")
        df["b"].plot(ax=twin_ax2, color="yellow")

        assert not plots[0][0].xaxis.get_label().get_visible()
        assert plots[0][1].xaxis.get_label().get_visible()
        assert not plots[0][2].xaxis.get_label().get_visible()

    def test_plot_bar_axis_units_timestamp_conversion(self):
        # GH 38736
        # Ensure string x-axis from the second plot will not be converted to datetime
        # due to axis data from first plot
        df = DataFrame(
            [1.0],
            index=[Timestamp("2022-02-22 22:22:22")],
        )
        _check_plot_works(df.plot)
        s = Series({"A": 1.0})
        _check_plot_works(s.plot.bar)

    def test_bar_plt_xaxis_intervalrange(self):
        # GH 38969
        # Ensure IntervalIndex x-axis produces a bar plot as expected
        from matplotlib.text import Text

        expected = [Text(0, 0, "([0, 1],)"), Text(1, 0, "([1, 2],)")]
        s = Series(
            [1, 2],
            index=[interval_range(0, 2, closed="both")],
        )
        _check_plot_works(s.plot.bar)
        assert all(
            (a.get_text() == b.get_text())
            for a, b in zip(s.plot.bar().get_xticklabels(), expected)
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_indexing.py ===
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.compat.numpy import np_long

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Timestamp,
    bdate_range,
    date_range,
    notna,
)
import pandas._testing as tm

from pandas.tseries.frequencies import to_offset

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestGetItem:
    def test_getitem_slice_keeps_name(self):
        # GH4226
        st = Timestamp("2013-07-01 00:00:00", tz="America/Los_Angeles")
        et = Timestamp("2013-07-02 00:00:00", tz="America/Los_Angeles")
        dr = date_range(st, et, freq="h", name="timebucket")
        assert dr[1:].name == dr.name

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo"])
    def test_getitem(self, tz):
        idx = date_range("2011-01-01", "2011-01-31", freq="D", tz=tz, name="idx")

        result = idx[0]
        assert result == Timestamp("2011-01-01", tz=idx.tz)

        result = idx[0:5]
        expected = date_range(
            "2011-01-01", "2011-01-05", freq="D", tz=idx.tz, name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        result = idx[0:10:2]
        expected = date_range(
            "2011-01-01", "2011-01-09", freq="2D", tz=idx.tz, name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        result = idx[-20:-5:3]
        expected = date_range(
            "2011-01-12", "2011-01-24", freq="3D", tz=idx.tz, name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        result = idx[4::-1]
        expected = DatetimeIndex(
            ["2011-01-05", "2011-01-04", "2011-01-03", "2011-01-02", "2011-01-01"],
            dtype=idx.dtype,
            freq="-1D",
            name="idx",
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_dti_business_getitem(self, freq):
        rng = bdate_range(START, END, freq=freq)
        smaller = rng[:5]
        exp = DatetimeIndex(rng.view(np.ndarray)[:5], freq=freq)
        tm.assert_index_equal(smaller, exp)
        assert smaller.freq == exp.freq
        assert smaller.freq == rng.freq

        sliced = rng[::5]
        assert sliced.freq == to_offset(freq) * 5

        fancy_indexed = rng[[4, 3, 2, 1, 0]]
        assert len(fancy_indexed) == 5
        assert isinstance(fancy_indexed, DatetimeIndex)
        assert fancy_indexed.freq is None

        # 32-bit vs. 64-bit platforms
        assert rng[4] == rng[np_long(4)]

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_dti_business_getitem_matplotlib_hackaround(self, freq):
        rng = bdate_range(START, END, freq=freq)
        with pytest.raises(ValueError, match="Multi-dimensional indexing"):
            # GH#30588 multi-dimensional indexing deprecated
            rng[:, None]

    def test_getitem_int_list(self):
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="ME")
        dti2 = dti[[1, 3, 5]]

        v1 = dti2[0]
        v2 = dti2[1]
        v3 = dti2[2]

        assert v1 == Timestamp("2/28/2005")
        assert v2 == Timestamp("4/30/2005")
        assert v3 == Timestamp("6/30/2005")

        # getitem with non-slice drops freq
        assert dti2.freq is None


class TestWhere:
    def test_where_doesnt_retain_freq(self):
        dti = date_range("20130101", periods=3, freq="D", name="idx")
        cond = [True, True, False]
        expected = DatetimeIndex([dti[0], dti[1], dti[0]], freq=None, name="idx")

        result = dti.where(cond, dti[::-1])
        tm.assert_index_equal(result, expected)

    def test_where_other(self):
        # other is ndarray or Index
        i = date_range("20130101", periods=3, tz="US/Eastern")

        for arr in [np.nan, pd.NaT]:
            result = i.where(notna(i), other=arr)
            expected = i
            tm.assert_index_equal(result, expected)

        i2 = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2), i2)
        tm.assert_index_equal(result, i2)

        i2 = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2), i2._values)
        tm.assert_index_equal(result, i2)

    def test_where_invalid_dtypes(self):
        dti = date_range("20130101", periods=3, tz="US/Eastern")

        tail = dti[2:].tolist()
        i2 = Index([pd.NaT, pd.NaT] + tail)

        mask = notna(i2)

        # passing tz-naive ndarray to tzaware DTI
        result = dti.where(mask, i2.values)
        expected = Index([pd.NaT.asm8, pd.NaT.asm8] + tail, dtype=object)
        tm.assert_index_equal(result, expected)

        # passing tz-aware DTI to tznaive DTI
        naive = dti.tz_localize(None)
        result = naive.where(mask, i2)
        expected = Index([i2[0], i2[1]] + naive[2:].tolist(), dtype=object)
        tm.assert_index_equal(result, expected)

        pi = i2.tz_localize(None).to_period("D")
        result = dti.where(mask, pi)
        expected = Index([pi[0], pi[1]] + tail, dtype=object)
        tm.assert_index_equal(result, expected)

        tda = i2.asi8.view("timedelta64[ns]")
        result = dti.where(mask, tda)
        expected = Index([tda[0], tda[1]] + tail, dtype=object)
        assert isinstance(expected[0], np.timedelta64)
        tm.assert_index_equal(result, expected)

        result = dti.where(mask, i2.asi8)
        expected = Index([pd.NaT._value, pd.NaT._value] + tail, dtype=object)
        assert isinstance(expected[0], int)
        tm.assert_index_equal(result, expected)

        # non-matching scalar
        td = pd.Timedelta(days=4)
        result = dti.where(mask, td)
        expected = Index([td, td] + tail, dtype=object)
        assert expected[0] is td
        tm.assert_index_equal(result, expected)

    def test_where_mismatched_nat(self, tz_aware_fixture):
        tz = tz_aware_fixture
        dti = date_range("2013-01-01", periods=3, tz=tz)
        cond = np.array([True, False, True])

        tdnat = np.timedelta64("NaT", "ns")
        expected = Index([dti[0], tdnat, dti[2]], dtype=object)
        assert expected[1] is tdnat

        result = dti.where(cond, tdnat)
        tm.assert_index_equal(result, expected)

    def test_where_tz(self):
        i = date_range("20130101", periods=3, tz="US/Eastern")
        result = i.where(notna(i))
        expected = i
        tm.assert_index_equal(result, expected)

        i2 = i.copy()
        i2 = Index([pd.NaT, pd.NaT] + i[2:].tolist())
        result = i.where(notna(i2))
        expected = i2
        tm.assert_index_equal(result, expected)


class TestTake:
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_take_dont_lose_meta(self, tzstr):
        rng = date_range("1/1/2000", periods=20, tz=tzstr)

        result = rng.take(range(5))
        assert result.tz == rng.tz
        assert result.freq == rng.freq

    def test_take_nan_first_datetime(self):
        index = DatetimeIndex([pd.NaT, Timestamp("20130101"), Timestamp("20130102")])
        result = index.take([-1, 0, 1])
        expected = DatetimeIndex([index[-1], index[0], index[1]])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo"])
    def test_take(self, tz):
        # GH#10295
        idx = date_range("2011-01-01", "2011-01-31", freq="D", name="idx", tz=tz)

        result = idx.take([0])
        assert result == Timestamp("2011-01-01", tz=idx.tz)

        result = idx.take([0, 1, 2])
        expected = date_range(
            "2011-01-01", "2011-01-03", freq="D", tz=idx.tz, name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        result = idx.take([0, 2, 4])
        expected = date_range(
            "2011-01-01", "2011-01-05", freq="2D", tz=idx.tz, name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        result = idx.take([7, 4, 1])
        expected = date_range(
            "2011-01-08", "2011-01-02", freq="-3D", tz=idx.tz, name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

        result = idx.take([3, 2, 5])
        expected = DatetimeIndex(
            ["2011-01-04", "2011-01-03", "2011-01-06"],
            dtype=idx.dtype,
            freq=None,
            name="idx",
        )
        tm.assert_index_equal(result, expected)
        assert result.freq is None

        result = idx.take([-3, 2, 5])
        expected = DatetimeIndex(
            ["2011-01-29", "2011-01-03", "2011-01-06"],
            dtype=idx.dtype,
            freq=None,
            name="idx",
        )
        tm.assert_index_equal(result, expected)
        assert result.freq is None

    def test_take_invalid_kwargs(self):
        idx = date_range("2011-01-01", "2011-01-31", freq="D", name="idx")
        indices = [1, 6, 5, 9, 10, 13, 15, 3]

        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            idx.take(indices, foo=2)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, out=indices)

        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, mode="clip")

    # TODO: This method came from test_datetime; de-dup with version above
    @pytest.mark.parametrize("tz", [None, "US/Eastern", "Asia/Tokyo"])
    def test_take2(self, tz):
        dates = [
            datetime(2010, 1, 1, 14),
            datetime(2010, 1, 1, 15),
            datetime(2010, 1, 1, 17),
            datetime(2010, 1, 1, 21),
        ]

        idx = date_range(
            start="2010-01-01 09:00",
            end="2010-02-01 09:00",
            freq="h",
            tz=tz,
            name="idx",
        )
        expected = DatetimeIndex(dates, freq=None, name="idx", dtype=idx.dtype)

        taken1 = idx.take([5, 6, 8, 12])
        taken2 = idx[[5, 6, 8, 12]]

        for taken in [taken1, taken2]:
            tm.assert_index_equal(taken, expected)
            assert isinstance(taken, DatetimeIndex)
            assert taken.freq is None
            assert taken.tz == expected.tz
            assert taken.name == expected.name

    def test_take_fill_value(self):
        # GH#12631
        idx = DatetimeIndex(["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = DatetimeIndex(["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = DatetimeIndex(["2011-02-01", "2011-01-01", "NaT"], name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = DatetimeIndex(["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx")
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "out of bounds"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    def test_take_fill_value_with_timezone(self):
        idx = DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx", tz="US/Eastern"
        )
        result = idx.take(np.array([1, 0, -1]))
        expected = DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", tz="US/Eastern"
        )
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = DatetimeIndex(
            ["2011-02-01", "2011-01-01", "NaT"], name="xxx", tz="US/Eastern"
        )
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx", tz="US/Eastern"
        )
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "out of bounds"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))


class TestGetLoc:
    def test_get_loc_key_unit_mismatch(self):
        idx = date_range("2000-01-01", periods=3)
        key = idx[1].as_unit("ms")
        loc = idx.get_loc(key)
        assert loc == 1
        assert key in idx

    def test_get_loc_key_unit_mismatch_not_castable(self):
        dta = date_range("2000-01-01", periods=3)._data.astype("M8[s]")
        dti = DatetimeIndex(dta)
        key = dta[0].as_unit("ns") + pd.Timedelta(1)

        with pytest.raises(
            KeyError, match=r"Timestamp\('2000-01-01 00:00:00.000000001'\)"
        ):
            dti.get_loc(key)

        assert key not in dti

    def test_get_loc_time_obj(self):
        # time indexing
        idx = date_range("2000-01-01", periods=24, freq="h")

        result = idx.get_loc(time(12))
        expected = np.array([12])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

        result = idx.get_loc(time(12, 30))
        expected = np.array([])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("offset", [-10, 10])
    def test_get_loc_time_obj2(self, monkeypatch, offset):
        # GH#8667
        size_cutoff = 50
        n = size_cutoff + offset
        key = time(15, 11, 30)
        start = key.hour * 3600 + key.minute * 60 + key.second
        step = 24 * 3600

        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            idx = date_range("2014-11-26", periods=n, freq="s")
            ts = pd.Series(np.random.default_rng(2).standard_normal(n), index=idx)
            locs = np.arange(start, n, step, dtype=np.intp)

            result = ts.index.get_loc(key)
            tm.assert_numpy_array_equal(result, locs)
            tm.assert_series_equal(ts[key], ts.iloc[locs])

            left, right = ts.copy(), ts.copy()
            left[key] *= -10
            right.iloc[locs] *= -10
            tm.assert_series_equal(left, right)

    def test_get_loc_time_nat(self):
        # GH#35114
        # Case where key's total microseconds happens to match iNaT % 1e6 // 1000
        tic = time(minute=12, second=43, microsecond=145224)
        dti = DatetimeIndex([pd.NaT])

        loc = dti.get_loc(tic)
        expected = np.array([], dtype=np.intp)
        tm.assert_numpy_array_equal(loc, expected)

    def test_get_loc_nat(self):
        # GH#20464
        index = DatetimeIndex(["1/3/2000", "NaT"])
        assert index.get_loc(pd.NaT) == 1

        assert index.get_loc(None) == 1

        assert index.get_loc(np.nan) == 1

        assert index.get_loc(pd.NA) == 1

        assert index.get_loc(np.datetime64("NaT")) == 1

        with pytest.raises(KeyError, match="NaT"):
            index.get_loc(np.timedelta64("NaT"))

    @pytest.mark.parametrize("key", [pd.Timedelta(0), pd.Timedelta(1), timedelta(0)])
    def test_get_loc_timedelta_invalid_key(self, key):
        # GH#20464
        dti = date_range("1970-01-01", periods=10)
        msg = "Cannot index DatetimeIndex with [Tt]imedelta"
        with pytest.raises(TypeError, match=msg):
            dti.get_loc(key)

    def test_get_loc_reasonable_key_error(self):
        # GH#1062
        index = DatetimeIndex(["1/3/2000"])
        with pytest.raises(KeyError, match="2000"):
            index.get_loc("1/1/2000")

    def test_get_loc_year_str(self):
        rng = date_range("1/1/2000", "1/1/2010")

        result = rng.get_loc("2009")
        expected = slice(3288, 3653)
        assert result == expected


class TestContains:
    def test_dti_contains_with_duplicates(self):
        d = datetime(2011, 12, 5, 20, 30)
        ix = DatetimeIndex([d, d])
        assert d in ix

    @pytest.mark.parametrize(
        "vals",
        [
            [0, 1, 0],
            [0, 0, -1],
            [0, -1, -1],
            ["2015", "2015", "2016"],
            ["2015", "2015", "2014"],
        ],
    )
    def test_contains_nonunique(self, vals):
        # GH#9512
        idx = DatetimeIndex(vals)
        assert idx[0] in idx


class TestGetIndexer:
    def test_get_indexer_date_objs(self):
        rng = date_range("1/1/2000", periods=20)

        result = rng.get_indexer(rng.map(lambda x: x.date()))
        expected = rng.get_indexer(rng)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer(self):
        idx = date_range("2000-01-01", periods=3)
        exp = np.array([0, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(idx.get_indexer(idx), exp)

        target = idx[0] + pd.to_timedelta(["-1 hour", "12 hours", "1 day 1 hour"])
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest", tolerance=pd.Timedelta("1 hour")),
            np.array([0, -1, 1], dtype=np.intp),
        )
        tol_raw = [
            pd.Timedelta("1 hour"),
            pd.Timedelta("1 hour"),
            pd.Timedelta("1 hour").to_timedelta64(),
        ]
        tm.assert_numpy_array_equal(
            idx.get_indexer(
                target, "nearest", tolerance=[np.timedelta64(x) for x in tol_raw]
            ),
            np.array([0, -1, 1], dtype=np.intp),
        )
        tol_bad = [
            pd.Timedelta("2 hour").to_timedelta64(),
            pd.Timedelta("1 hour").to_timedelta64(),
            "foo",
        ]
        msg = "Could not convert 'foo' to NumPy timedelta"
        with pytest.raises(ValueError, match=msg):
            idx.get_indexer(target, "nearest", tolerance=tol_bad)
        with pytest.raises(ValueError, match="abbreviation w/o a number"):
            idx.get_indexer(idx[[0]], method="nearest", tolerance="foo")

    @pytest.mark.parametrize(
        "target",
        [
            [date(2020, 1, 1), Timestamp("2020-01-02")],
            [Timestamp("2020-01-01"), date(2020, 1, 2)],
        ],
    )
    def test_get_indexer_mixed_dtypes(self, target):
        # https://github.com/pandas-dev/pandas/issues/33741
        values = DatetimeIndex([Timestamp("2020-01-01"), Timestamp("2020-01-02")])
        result = values.get_indexer(target)
        expected = np.array([0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "target, positions",
        [
            ([date(9999, 1, 1), Timestamp("2020-01-01")], [-1, 0]),
            ([Timestamp("2020-01-01"), date(9999, 1, 1)], [0, -1]),
            ([date(9999, 1, 1), date(9999, 1, 1)], [-1, -1]),
        ],
    )
    def test_get_indexer_out_of_bounds_date(self, target, positions):
        values = DatetimeIndex([Timestamp("2020-01-01"), Timestamp("2020-01-02")])

        result = values.get_indexer(target)
        expected = np.array(positions, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_pad_requires_monotonicity(self):
        rng = date_range("1/1/2000", "3/1/2000", freq="B")

        # neither monotonic increasing or decreasing
        rng2 = rng[[1, 0, 2]]

        msg = "index must be monotonic increasing or decreasing"
        with pytest.raises(ValueError, match=msg):
            rng2.get_indexer(rng, method="pad")


class TestMaybeCastSliceBound:
    def test_maybe_cast_slice_bounds_empty(self):
        # GH#14354
        empty_idx = date_range(freq="1h", periods=0, end="2015")

        right = empty_idx._maybe_cast_slice_bound("2015-01-02", "right")
        exp = Timestamp("2015-01-02 23:59:59.999999999")
        assert right == exp

        left = empty_idx._maybe_cast_slice_bound("2015-01-02", "left")
        exp = Timestamp("2015-01-02 00:00:00")
        assert left == exp

    def test_maybe_cast_slice_duplicate_monotonic(self):
        # https://github.com/pandas-dev/pandas/issues/16515
        idx = DatetimeIndex(["2017", "2017"])
        result = idx._maybe_cast_slice_bound("2017-01-01", "left")
        expected = Timestamp("2017-01-01")
        assert result == expected


class TestGetSliceBounds:
    @pytest.mark.parametrize("box", [date, datetime, Timestamp])
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
    def test_get_slice_bounds_datetime_within(
        self, box, side, expected, tz_aware_fixture
    ):
        # GH 35690
        tz = tz_aware_fixture
        index = bdate_range("2000-01-03", "2000-02-11").tz_localize(tz)
        key = box(year=2000, month=1, day=7)

        if tz is not None:
            with pytest.raises(TypeError, match="Cannot compare tz-naive"):
                # GH#36148 we require tzawareness-compat as of 2.0
                index.get_slice_bound(key, side=side)
        else:
            result = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize("box", [datetime, Timestamp])
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("year, expected", [(1999, 0), (2020, 30)])
    def test_get_slice_bounds_datetime_outside(
        self, box, side, year, expected, tz_aware_fixture
    ):
        # GH 35690
        tz = tz_aware_fixture
        index = bdate_range("2000-01-03", "2000-02-11").tz_localize(tz)
        key = box(year=year, month=1, day=7)

        if tz is not None:
            with pytest.raises(TypeError, match="Cannot compare tz-naive"):
                # GH#36148 we require tzawareness-compat as of 2.0
                index.get_slice_bound(key, side=side)
        else:
            result = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize("box", [datetime, Timestamp])
    def test_slice_datetime_locs(self, box, tz_aware_fixture):
        # GH 34077
        tz = tz_aware_fixture
        index = DatetimeIndex(["2010-01-01", "2010-01-03"]).tz_localize(tz)
        key = box(2010, 1, 1)

        if tz is not None:
            with pytest.raises(TypeError, match="Cannot compare tz-naive"):
                # GH#36148 we require tzawareness-compat as of 2.0
                index.slice_locs(key, box(2010, 1, 2))
        else:
            result = index.slice_locs(key, box(2010, 1, 2))
            expected = (0, 1)
            assert result == expected


class TestIndexerBetweenTime:
    def test_indexer_between_time(self):
        # GH#11818
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        msg = r"Cannot convert arg \[datetime\.datetime\(2010, 1, 2, 1, 0\)\] to a time"
        with pytest.raises(ValueError, match=msg):
            rng.indexer_between_time(datetime(2010, 1, 2, 1), datetime(2010, 1, 2, 5))

    @pytest.mark.parametrize("unit", ["us", "ms", "s"])
    def test_indexer_between_time_non_nano(self, unit):
        # For simple cases like this, the non-nano indexer_between_time
        #  should match the nano result

        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        arr_nano = rng._data._ndarray

        arr = arr_nano.astype(f"M8[{unit}]")

        dta = type(rng._data)._simple_new(arr, dtype=arr.dtype)
        dti = DatetimeIndex(dta)
        assert dti.dtype == arr.dtype

        tic = time(1, 25)
        toc = time(2, 29)

        result = dti.indexer_between_time(tic, toc)
        expected = rng.indexer_between_time(tic, toc)
        tm.assert_numpy_array_equal(result, expected)

        # case with non-zero micros in arguments
        tic = time(1, 25, 0, 45678)
        toc = time(2, 29, 0, 1234)

        result = dti.indexer_between_time(tic, toc)
        expected = rng.indexer_between_time(tic, toc)
        tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\resample\test_resampler_grouper.py ===
from textwrap import dedent

import numpy as np
import pytest

from pandas.compat import is_platform_windows

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range


@pytest.fixture
def test_frame():
    return DataFrame(
        {"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)},
        index=date_range("1/1/2000", freq="s", periods=40),
    )


def test_tab_complete_ipython6_warning(ip):
    from IPython.core.completer import provisionalcompleter

    code = dedent(
        """\
    import numpy as np
    from pandas import Series, date_range
    data = np.arange(10, dtype=np.float64)
    index = date_range("2020-01-01", periods=len(data))
    s = Series(data, index=index)
    rs = s.resample("D")
    """
    )
    ip.run_cell(code)

    # GH 31324 newer jedi version raises Deprecation warning;
    #  appears resolved 2021-02-02
    with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
        with provisionalcompleter("ignore"):
            list(ip.Completer.completions("rs.", 1))


def test_deferred_with_groupby():
    # GH 12486
    # support deferred resample ops with groupby
    data = [
        ["2010-01-01", "A", 2],
        ["2010-01-02", "A", 3],
        ["2010-01-05", "A", 8],
        ["2010-01-10", "A", 7],
        ["2010-01-13", "A", 3],
        ["2010-01-01", "B", 5],
        ["2010-01-03", "B", 2],
        ["2010-01-04", "B", 1],
        ["2010-01-11", "B", 7],
        ["2010-01-14", "B", 3],
    ]

    df = DataFrame(data, columns=["date", "id", "score"])
    df.date = pd.to_datetime(df.date)

    def f_0(x):
        return x.set_index("date").resample("D").asfreq()

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby("id").apply(f_0)
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.set_index("date").groupby("id").resample("D").asfreq()
    tm.assert_frame_equal(result, expected)

    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W"),
            "group": [1, 1, 2, 2],
            "val": [5, 6, 7, 8],
        }
    ).set_index("date")

    def f_1(x):
        return x.resample("1D").ffill()

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby("group").apply(f_1)
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("group").resample("1D").ffill()
    tm.assert_frame_equal(result, expected)


def test_getitem(test_frame):
    g = test_frame.groupby("A")

    expected = g.B.apply(lambda x: x.resample("2s").mean())

    result = g.resample("2s").B.mean()
    tm.assert_series_equal(result, expected)

    result = g.B.resample("2s").mean()
    tm.assert_series_equal(result, expected)

    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = g.resample("2s").mean().B
    tm.assert_series_equal(result, expected)


def test_getitem_multiple():
    # GH 13174
    # multiple calls after selection causing an issue with aliasing
    data = [{"id": 1, "buyer": "A"}, {"id": 2, "buyer": "B"}]
    df = DataFrame(data, index=date_range("2016-01-01", periods=2))
    r = df.groupby("id").resample("1D")
    result = r["buyer"].count()

    exp_mi = pd.MultiIndex.from_arrays([[1, 2], df.index], names=("id", None))
    expected = Series(
        [1, 1],
        index=exp_mi,
        name="buyer",
    )
    tm.assert_series_equal(result, expected)

    result = r["buyer"].count()
    tm.assert_series_equal(result, expected)


def test_groupby_resample_on_api_with_getitem():
    # GH 17813
    df = DataFrame(
        {"id": list("aabbb"), "date": date_range("1-1-2016", periods=5), "data": 1}
    )
    exp = df.set_index("date").groupby("id").resample("2D")["data"].sum()
    result = df.groupby("id").resample("2D", on="date")["data"].sum()
    tm.assert_series_equal(result, exp)


def test_groupby_with_origin():
    # GH 31809

    freq = "1399min"  # prime number that is smaller than 24h
    start, end = "1/1/2000 00:00:00", "1/31/2000 00:00"
    middle = "1/15/2000 00:00:00"

    rng = date_range(start, end, freq="1231min")  # prime number
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts2 = ts[middle:end]

    # proves that grouper without a fixed origin does not work
    # when dealing with unusual frequencies
    simple_grouper = pd.Grouper(freq=freq)
    count_ts = ts.groupby(simple_grouper).agg("count")
    count_ts = count_ts[middle:end]
    count_ts2 = ts2.groupby(simple_grouper).agg("count")
    with pytest.raises(AssertionError, match="Index are different"):
        tm.assert_index_equal(count_ts.index, count_ts2.index)

    # test origin on 1970-01-01 00:00:00
    origin = Timestamp(0)
    adjusted_grouper = pd.Grouper(freq=freq, origin=origin)
    adjusted_count_ts = ts.groupby(adjusted_grouper).agg("count")
    adjusted_count_ts = adjusted_count_ts[middle:end]
    adjusted_count_ts2 = ts2.groupby(adjusted_grouper).agg("count")
    tm.assert_series_equal(adjusted_count_ts, adjusted_count_ts2)

    # test origin on 2049-10-18 20:00:00
    origin_future = Timestamp(0) + pd.Timedelta("1399min") * 30_000
    adjusted_grouper2 = pd.Grouper(freq=freq, origin=origin_future)
    adjusted2_count_ts = ts.groupby(adjusted_grouper2).agg("count")
    adjusted2_count_ts = adjusted2_count_ts[middle:end]
    adjusted2_count_ts2 = ts2.groupby(adjusted_grouper2).agg("count")
    tm.assert_series_equal(adjusted2_count_ts, adjusted2_count_ts2)

    # both grouper use an adjusted timestamp that is a multiple of 1399 min
    # they should be equals even if the adjusted_timestamp is in the future
    tm.assert_series_equal(adjusted_count_ts, adjusted2_count_ts2)


def test_nearest():
    # GH 17496
    # Resample nearest
    index = date_range("1/1/2000", periods=3, freq="min")
    result = Series(range(3), index=index).resample("20s").nearest()

    expected = Series(
        [0, 0, 1, 1, 1, 2, 2],
        index=pd.DatetimeIndex(
            [
                "2000-01-01 00:00:00",
                "2000-01-01 00:00:20",
                "2000-01-01 00:00:40",
                "2000-01-01 00:01:00",
                "2000-01-01 00:01:20",
                "2000-01-01 00:01:40",
                "2000-01-01 00:02:00",
            ],
            dtype="datetime64[ns]",
            freq="20s",
        ),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "f",
    [
        "first",
        "last",
        "median",
        "sem",
        "sum",
        "mean",
        "min",
        "max",
        "size",
        "count",
        "nearest",
        "bfill",
        "ffill",
        "asfreq",
        "ohlc",
    ],
)
def test_methods(f, test_frame):
    g = test_frame.groupby("A")
    r = g.resample("2s")

    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(r, f)()
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)())
    tm.assert_equal(result, expected)


def test_methods_nunique(test_frame):
    # series only
    g = test_frame.groupby("A")
    r = g.resample("2s")
    result = r.B.nunique()
    expected = g.B.apply(lambda x: x.resample("2s").nunique())
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("f", ["std", "var"])
def test_methods_std_var(f, test_frame):
    g = test_frame.groupby("A")
    r = g.resample("2s")
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(r, f)(ddof=1)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)(ddof=1))
    tm.assert_frame_equal(result, expected)


def test_apply(test_frame):
    g = test_frame.groupby("A")
    r = g.resample("2s")

    # reduction
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = g.resample("2s").sum()

    def f_0(x):
        return x.resample("2s").sum()

    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = r.apply(f_0)
    tm.assert_frame_equal(result, expected)

    def f_1(x):
        return x.resample("2s").apply(lambda y: y.sum())

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = g.apply(f_1)
    # y.sum() results in int64 instead of int32 on 32-bit architectures
    expected = expected.astype("int64")
    tm.assert_frame_equal(result, expected)


def test_apply_with_mutated_index():
    # GH 15169
    index = date_range("1-1-2015", "12-31-15", freq="D")
    df = DataFrame(
        data={"col1": np.random.default_rng(2).random(len(index))}, index=index
    )

    def f(x):
        s = Series([1, 2], index=["a", "b"])
        return s

    expected = df.groupby(pd.Grouper(freq="ME")).apply(f)

    result = df.resample("ME").apply(f)
    tm.assert_frame_equal(result, expected)

    # A case for series
    expected = df["col1"].groupby(pd.Grouper(freq="ME"), group_keys=False).apply(f)
    result = df["col1"].resample("ME").apply(f)
    tm.assert_series_equal(result, expected)


def test_apply_columns_multilevel():
    # GH 16231
    cols = pd.MultiIndex.from_tuples([("A", "a", "", "one"), ("B", "b", "i", "two")])
    ind = date_range(start="2017-01-01", freq="15Min", periods=8)
    df = DataFrame(np.array([0] * 16).reshape(8, 2), index=ind, columns=cols)
    agg_dict = {col: (np.sum if col[3] == "one" else np.mean) for col in df.columns}
    result = df.resample("h").apply(lambda x: agg_dict[x.name](x))
    expected = DataFrame(
        2 * [[0, 0.0]],
        index=date_range(start="2017-01-01", freq="1h", periods=2),
        columns=pd.MultiIndex.from_tuples(
            [("A", "a", "", "one"), ("B", "b", "i", "two")]
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_apply_non_naive_index():
    def weighted_quantile(series, weights, q):
        series = series.sort_values()
        cumsum = weights.reindex(series.index).fillna(0).cumsum()
        cutoff = cumsum.iloc[-1] * q
        return series[cumsum >= cutoff].iloc[0]

    times = date_range("2017-6-23 18:00", periods=8, freq="15min", tz="UTC")
    data = Series([1.0, 1, 1, 1, 1, 2, 2, 0], index=times)
    weights = Series([160.0, 91, 65, 43, 24, 10, 1, 0], index=times)

    result = data.resample("D").apply(weighted_quantile, weights=weights, q=0.5)
    ind = date_range(
        "2017-06-23 00:00:00+00:00", "2017-06-23 00:00:00+00:00", freq="D", tz="UTC"
    )
    expected = Series([1.0], index=ind)
    tm.assert_series_equal(result, expected)


def test_resample_groupby_with_label(unit):
    # GH 13235
    index = date_range("2000-01-01", freq="2D", periods=5, unit=unit)
    df = DataFrame(index=index, data={"col0": [0, 0, 1, 1, 2], "col1": [1, 1, 1, 1, 1]})
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("col0").resample("1W", label="left").sum()

    mi = [
        np.array([0, 0, 1, 2], dtype=np.int64),
        np.array(
            ["1999-12-26", "2000-01-02", "2000-01-02", "2000-01-02"],
            dtype=f"M8[{unit}]",
        ),
    ]
    mindex = pd.MultiIndex.from_arrays(mi, names=["col0", None])
    expected = DataFrame(
        data={"col0": [0, 0, 2, 2], "col1": [1, 1, 2, 1]}, index=mindex
    )

    tm.assert_frame_equal(result, expected)


def test_consistency_with_window(test_frame):
    # consistent return values with window
    df = test_frame
    expected = Index([1, 2, 3], name="A")
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").resample("2s").mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)

    result = df.groupby("A").rolling(20).mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)


def test_median_duplicate_columns():
    # GH 14233

    df = DataFrame(
        np.random.default_rng(2).standard_normal((20, 3)),
        columns=list("aaa"),
        index=date_range("2012-01-01", periods=20, freq="s"),
    )
    df2 = df.copy()
    df2.columns = ["a", "b", "c"]
    expected = df2.resample("5s").median()
    result = df.resample("5s").median()
    expected.columns = result.columns
    tm.assert_frame_equal(result, expected)


def test_apply_to_one_column_of_df():
    # GH: 36951
    df = DataFrame(
        {"col": range(10), "col1": range(10, 20)},
        index=date_range("2012-01-01", periods=10, freq="20min"),
    )

    # access "col" via getattr -> make sure we handle AttributeError
    result = df.resample("h").apply(lambda group: group.col.sum())
    expected = Series(
        [3, 12, 21, 9], index=date_range("2012-01-01", periods=4, freq="h")
    )
    tm.assert_series_equal(result, expected)

    # access "col" via _getitem__ -> make sure we handle KeyErrpr
    result = df.resample("h").apply(lambda group: group["col"].sum())
    tm.assert_series_equal(result, expected)


def test_resample_groupby_agg():
    # GH: 33548
    df = DataFrame(
        {
            "cat": [
                "cat_1",
                "cat_1",
                "cat_2",
                "cat_1",
                "cat_2",
                "cat_1",
                "cat_2",
                "cat_1",
            ],
            "num": [5, 20, 22, 3, 4, 30, 10, 50],
            "date": [
                "2019-2-1",
                "2018-02-03",
                "2020-3-11",
                "2019-2-2",
                "2019-2-2",
                "2018-12-4",
                "2020-3-11",
                "2020-12-12",
            ],
        }
    )
    df["date"] = pd.to_datetime(df["date"])

    resampled = df.groupby("cat").resample("YE", on="date")
    expected = resampled[["num"]].sum()
    result = resampled.agg({"num": "sum"})

    tm.assert_frame_equal(result, expected)


def test_resample_groupby_agg_listlike():
    # GH 42905
    ts = Timestamp("2021-02-28 00:00:00")
    df = DataFrame({"class": ["beta"], "value": [69]}, index=Index([ts], name="date"))
    resampled = df.groupby("class").resample("ME")["value"]
    result = resampled.agg(["sum", "size"])
    expected = DataFrame(
        [[69, 1]],
        index=pd.MultiIndex.from_tuples([("beta", ts)], names=["class", "date"]),
        columns=["sum", "size"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_empty(keys):
    # GH 26411
    df = DataFrame([], columns=["a", "b"], index=TimedeltaIndex([]))
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(keys).resample(rule=pd.to_timedelta("00:00:01")).mean()
    expected = (
        DataFrame(columns=["a", "b"])
        .set_index(keys, drop=False)
        .set_index(TimedeltaIndex([]), append=True)
    )
    if len(keys) == 1:
        expected.index.name = keys[0]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("consolidate", [True, False])
def test_resample_groupby_agg_object_dtype_all_nan(consolidate):
    # https://github.com/pandas-dev/pandas/issues/39329

    dates = date_range("2020-01-01", periods=15, freq="D")
    df1 = DataFrame({"key": "A", "date": dates, "col1": range(15), "col_object": "val"})
    df2 = DataFrame({"key": "B", "date": dates, "col1": range(15)})
    df = pd.concat([df1, df2], ignore_index=True)
    if consolidate:
        df = df._consolidate()

    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(["key"]).resample("W", on="date").min()
    idx = pd.MultiIndex.from_arrays(
        [
            ["A"] * 3 + ["B"] * 3,
            pd.to_datetime(["2020-01-05", "2020-01-12", "2020-01-19"] * 2).as_unit(
                "ns"
            ),
        ],
        names=["key", "date"],
    )
    expected = DataFrame(
        {
            "key": ["A"] * 3 + ["B"] * 3,
            "col1": [0, 5, 12] * 2,
            "col_object": ["val"] * 3 + [np.nan] * 3,
        },
        index=idx,
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_resample_with_list_of_keys():
    # GH 47362
    df = DataFrame(
        data={
            "date": date_range(start="2016-01-01", periods=8),
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "val": [1, 7, 5, 2, 3, 10, 5, 1],
        }
    )
    result = df.groupby("group").resample("2D", on="date")[["val"]].mean()

    mi_exp = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], df["date"]._values[::2]], names=["group", "date"]
    )
    expected = DataFrame(
        data={
            "val": [4.0, 3.5, 6.5, 3.0],
        },
        index=mi_exp,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_resample_no_index(keys):
    # GH 47705
    df = DataFrame([], columns=["a", "b", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(keys).resample(rule=pd.to_timedelta("00:00:01")).mean()
    expected = DataFrame(columns=["a", "b", "date"]).set_index(keys, drop=False)
    expected["date"] = pd.to_datetime(expected["date"])
    expected = expected.set_index("date", append=True, drop=True)
    if len(keys) == 1:
        expected.index.name = keys[0]

    tm.assert_frame_equal(result, expected)


def test_resample_no_columns():
    # GH#52484
    df = DataFrame(
        index=Index(
            pd.to_datetime(
                ["2018-01-01 00:00:00", "2018-01-01 12:00:00", "2018-01-02 00:00:00"]
            ),
            name="date",
        )
    )
    result = df.groupby([0, 0, 1]).resample(rule=pd.to_timedelta("06:00:00")).mean()
    index = pd.to_datetime(
        [
            "2018-01-01 00:00:00",
            "2018-01-01 06:00:00",
            "2018-01-01 12:00:00",
            "2018-01-02 00:00:00",
        ]
    )
    expected = DataFrame(
        index=pd.MultiIndex(
            levels=[np.array([0, 1], dtype=np.intp), index],
            codes=[[0, 0, 0, 1], [0, 1, 2, 3]],
            names=[None, "date"],
        )
    )

    # GH#52710 - Index comes out as 32-bit on 64-bit Windows
    tm.assert_frame_equal(result, expected, check_index_type=not is_platform_windows())


def test_groupby_resample_size_all_index_same():
    # GH 46826
    df = DataFrame(
        {"A": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3, "B": np.arange(12)},
        index=date_range("31/12/2000 18:00", freq="h", periods=12),
    )
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").resample("D").size()

    mi_exp = pd.MultiIndex.from_arrays(
        [
            [1, 1, 2, 2],
            pd.DatetimeIndex(["2000-12-31", "2001-01-01"] * 2, dtype="M8[ns]"),
        ],
        names=["A", None],
    )
    expected = Series(
        3,
        index=mi_exp,
    )
    tm.assert_series_equal(result, expected)


def test_groupby_resample_on_index_with_list_of_keys():
    # GH 50840
    df = DataFrame(
        data={
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "val": [3, 1, 4, 1, 5, 9, 2, 6],
        },
        index=date_range(start="2016-01-01", periods=8, name="date"),
    )
    result = df.groupby("group").resample("2D")[["val"]].mean()

    mi_exp = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], df.index[::2]], names=["group", "date"]
    )
    expected = DataFrame(
        data={
            "val": [2.0, 2.5, 7.0, 4.0],
        },
        index=mi_exp,
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_resample_on_index_with_list_of_keys_multi_columns():
    # GH 50876
    df = DataFrame(
        data={
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "first_val": [3, 1, 4, 1, 5, 9, 2, 6],
            "second_val": [2, 7, 1, 8, 2, 8, 1, 8],
            "third_val": [1, 4, 1, 4, 2, 1, 3, 5],
        },
        index=date_range(start="2016-01-01", periods=8, name="date"),
    )
    result = df.groupby("group").resample("2D")[["first_val", "second_val"]].mean()

    mi_exp = pd.MultiIndex.from_arrays(
        [[0, 0, 1, 1], df.index[::2]], names=["group", "date"]
    )
    expected = DataFrame(
        data={
            "first_val": [2.0, 2.5, 7.0, 4.0],
            "second_val": [4.5, 4.5, 5.0, 4.5],
        },
        index=mi_exp,
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_resample_on_index_with_list_of_keys_missing_column():
    # GH 50876
    df = DataFrame(
        data={
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "val": [3, 1, 4, 1, 5, 9, 2, 6],
        },
        index=Series(
            date_range(start="2016-01-01", periods=8),
            name="date",
        ),
    )
    gb = df.groupby("group")
    rs = gb.resample("2D")
    with pytest.raises(KeyError, match="Columns not found"):
        rs[["val_not_in_dataframe"]]


@pytest.mark.parametrize("kind", ["datetime", "period"])
def test_groupby_resample_kind(kind):
    # GH 24103
    df = DataFrame(
        {
            "datetime": pd.to_datetime(
                ["20181101 1100", "20181101 1200", "20181102 1300", "20181102 1400"]
            ),
            "group": ["A", "B", "A", "B"],
            "value": [1, 2, 3, 4],
        }
    )
    df = df.set_index("datetime")
    result = df.groupby("group")["value"].resample("D", kind=kind).last()

    dt_level = pd.DatetimeIndex(["2018-11-01", "2018-11-02"])
    if kind == "period":
        dt_level = dt_level.to_period(freq="D")
    expected_index = pd.MultiIndex.from_product(
        [["A", "B"], dt_level],
        names=["group", "datetime"],
    )
    expected = Series([1, 3, 2, 4], index=expected_index, name="value")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_timeseries_window.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm

from pandas.tseries import offsets


@pytest.fixture
def regular():
    return DataFrame(
        {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
    ).set_index("A")


@pytest.fixture
def ragged():
    df = DataFrame({"B": range(5)})
    df.index = [
        Timestamp("20130101 09:00:00"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:03"),
        Timestamp("20130101 09:00:05"),
        Timestamp("20130101 09:00:06"),
    ]
    return df


class TestRollingTS:
    # rolling time-series friendly
    # xref GH13327

    def test_doc_string(self):
        df = DataFrame(
            {"B": [0, 1, 2, np.nan, 4]},
            index=[
                Timestamp("20130101 09:00:00"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:05"),
                Timestamp("20130101 09:00:06"),
            ],
        )
        df
        df.rolling("2s").sum()

    def test_invalid_window_non_int(self, regular):
        # not a valid freq
        msg = "passed window foobar is not compatible with a datetimelike index"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="foobar")
        # not a datetimelike index
        msg = "window must be an integer"
        with pytest.raises(ValueError, match=msg):
            regular.reset_index().rolling(window="foobar")

    @pytest.mark.parametrize("freq", ["2MS", offsets.MonthBegin(2)])
    def test_invalid_window_nonfixed(self, freq, regular):
        # non-fixed freqs
        msg = "\\<2 \\* MonthBegins\\> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window=freq)

    @pytest.mark.parametrize("freq", ["1D", offsets.Day(2), "2ms"])
    def test_valid_window(self, freq, regular):
        regular.rolling(window=freq)

    @pytest.mark.parametrize("minp", [1.0, "foo", np.array([1, 2, 3])])
    def test_invalid_minp(self, minp, regular):
        # non-integer min_periods
        msg = (
            r"local variable 'minp' referenced before assignment|"
            "min_periods must be an integer"
        )
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="1D", min_periods=minp)

    def test_on(self, regular):
        df = regular

        # not a valid column
        msg = (
            r"invalid on specified as foobar, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2s", on="foobar")

        # column is valid
        df = df.copy()
        df["C"] = date_range("20130101", periods=len(df))
        df.rolling(window="2d", on="C").sum()

        # invalid columns
        msg = "window must be an integer"
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2d", on="B")

        # ok even though on non-selected
        df.rolling(window="2d", on="C").B.sum()

    def test_monotonic_on(self):
        # on/index must be monotonic
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )

        assert df.A.is_monotonic_increasing
        df.rolling("2s", on="A").sum()

        df = df.set_index("A")
        assert df.index.is_monotonic_increasing
        df.rolling("2s").sum()

    def test_non_monotonic_on(self):
        # GH 19248
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )
        df = df.set_index("A")
        non_monotonic_index = df.index.to_list()
        non_monotonic_index[0] = non_monotonic_index[3]
        df.index = non_monotonic_index

        assert not df.index.is_monotonic_increasing

        msg = "index values must be monotonic"
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s").sum()

        df = df.reset_index()

        msg = (
            r"invalid on specified as A, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s", on="A").sum()

    def test_frame_on(self):
        df = DataFrame(
            {"B": range(5), "C": date_range("20130101 09:00:00", periods=5, freq="3s")}
        )

        df["A"] = [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130101 09:00:02"),
            Timestamp("20130101 09:00:03"),
            Timestamp("20130101 09:00:05"),
            Timestamp("20130101 09:00:06"),
        ]

        # we are doing simulating using 'on'
        expected = df.set_index("A").rolling("2s").B.sum().reset_index(drop=True)

        result = df.rolling("2s", on="A").B.sum()
        tm.assert_series_equal(result, expected)

        # test as a frame
        # we should be ignoring the 'on' as an aggregation column
        # note that the expected is setting, computing, and resetting
        # so the columns need to be switched compared
        # to the actual result where they are ordered as in the
        # original
        expected = (
            df.set_index("A").rolling("2s")[["B"]].sum().reset_index()[["B", "A"]]
        )

        result = df.rolling("2s", on="A")[["B"]].sum()
        tm.assert_frame_equal(result, expected)

    def test_frame_on2(self, unit):
        # using multiple aggregation columns
        dti = DatetimeIndex(
            [
                Timestamp("20130101 09:00:00"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:05"),
                Timestamp("20130101 09:00:06"),
            ]
        ).as_unit(unit)
        df = DataFrame(
            {
                "A": [0, 1, 2, 3, 4],
                "B": [0, 1, 2, np.nan, 4],
                "C": dti,
            },
            columns=["A", "C", "B"],
        )

        expected1 = DataFrame(
            {"A": [0.0, 1, 3, 3, 7], "B": [0, 1, 3, np.nan, 4], "C": df["C"]},
            columns=["A", "C", "B"],
        )

        result = df.rolling("2s", on="C").sum()
        expected = expected1
        tm.assert_frame_equal(result, expected)

        expected = Series([0, 1, 3, np.nan, 4], name="B")
        result = df.rolling("2s", on="C").B.sum()
        tm.assert_series_equal(result, expected)

        expected = expected1[["A", "B", "C"]]
        result = df.rolling("2s", on="C")[["A", "B", "C"]].sum()
        tm.assert_frame_equal(result, expected)

    def test_basic_regular(self, regular):
        df = regular.copy()

        df.index = date_range("20130101", periods=5, freq="D")
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="1D").sum()
        tm.assert_frame_equal(result, expected)

        df.index = date_range("20130101", periods=5, freq="2D")
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(window=1).sum()
        result = df.rolling(window="2D").sum()
        tm.assert_frame_equal(result, expected)

    def test_min_periods(self, regular):
        # compare for min_periods
        df = regular

        # these slightly different
        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_closed(self, regular, unit):
        # xref GH13965

        dti = DatetimeIndex(
            [
                Timestamp("20130101 09:00:01"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:04"),
                Timestamp("20130101 09:00:06"),
            ]
        ).as_unit(unit)

        df = DataFrame(
            {"A": [1] * 5},
            index=dti,
        )

        # closed must be 'right', 'left', 'both', 'neither'
        msg = "closed must be 'right', 'left', 'both' or 'neither'"
        with pytest.raises(ValueError, match=msg):
            regular.rolling(window="2s", closed="blabla")

        expected = df.copy()
        expected["A"] = [1.0, 2, 2, 2, 1]
        result = df.rolling("2s", closed="right").sum()
        tm.assert_frame_equal(result, expected)

        # default should be 'right'
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [1.0, 2, 3, 3, 2]
        result = df.rolling("2s", closed="both").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 2, 2, 1]
        result = df.rolling("2s", closed="left").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 1, 1, np.nan]
        result = df.rolling("2s", closed="neither").sum()
        tm.assert_frame_equal(result, expected)

    def test_ragged_sum(self, ragged):
        df = ragged
        result = df.rolling(window="1s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 3, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=2).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, np.nan, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s").sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="4s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="4s", min_periods=3).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 10]
        tm.assert_frame_equal(result, expected)

    def test_ragged_mean(self, ragged):
        df = ragged
        result = df.rolling(window="1s", min_periods=1).mean()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).mean()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_median(self, ragged):
        df = ragged
        result = df.rolling(window="1s", min_periods=1).median()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).median()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_quantile(self, ragged):
        df = ragged
        result = df.rolling(window="1s", min_periods=1).quantile(0.5)
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).quantile(0.5)
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_std(self, ragged):
        df = ragged
        result = df.rolling(window="1s", min_periods=1).std(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="1s", min_periods=1).std(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).std(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] + [0.5] * 4
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).std(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan, 0.707107, 1.0, 1.0, 1.290994]
        tm.assert_frame_equal(result, expected)

    def test_ragged_var(self, ragged):
        df = ragged
        result = df.rolling(window="1s", min_periods=1).var(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="1s", min_periods=1).var(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).var(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] + [0.25] * 4
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).var(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan, 0.5, 1.0, 1.0, 1 + 2 / 3.0]
        tm.assert_frame_equal(result, expected)

    def test_ragged_skew(self, ragged):
        df = ragged
        result = df.rolling(window="3s", min_periods=1).skew()
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).skew()
        expected = df.copy()
        expected["B"] = [np.nan] * 2 + [0.0, 0.0, 0.0]
        tm.assert_frame_equal(result, expected)

    def test_ragged_kurt(self, ragged):
        df = ragged
        result = df.rolling(window="3s", min_periods=1).kurt()
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).kurt()
        expected = df.copy()
        expected["B"] = [np.nan] * 4 + [-1.2]
        tm.assert_frame_equal(result, expected)

    def test_ragged_count(self, ragged):
        df = ragged
        result = df.rolling(window="1s", min_periods=1).count()
        expected = df.copy()
        expected["B"] = [1.0, 1, 1, 1, 1]
        tm.assert_frame_equal(result, expected)

        df = ragged
        result = df.rolling(window="1s").count()
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).count()
        expected = df.copy()
        expected["B"] = [1.0, 1, 2, 1, 2]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=2).count()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 2, np.nan, 2]
        tm.assert_frame_equal(result, expected)

    def test_regular_min(self):
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [0.0, 1, 2, 3, 4]}
        ).set_index("A")
        result = df.rolling("1s").min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [5, 4, 3, 4, 5]}
        ).set_index("A")

        tm.assert_frame_equal(result, expected)
        result = df.rolling("2s").min()
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling("5s").min()
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 3]
        tm.assert_frame_equal(result, expected)

    def test_ragged_min(self, ragged):
        df = ragged

        result = df.rolling(window="1s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1, 3, 3]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 0, 0, 1, 1]
        tm.assert_frame_equal(result, expected)

    def test_perf_min(self):
        N = 10000

        dfp = DataFrame(
            {"B": np.random.default_rng(2).standard_normal(N)},
            index=date_range("20130101", periods=N, freq="s"),
        )
        expected = dfp.rolling(2, min_periods=1).min()
        result = dfp.rolling("2s").min()
        assert ((result - expected) < 0.01).all().all()

        expected = dfp.rolling(200, min_periods=1).min()
        result = dfp.rolling("200s").min()
        assert ((result - expected) < 0.01).all().all()

    def test_ragged_max(self, ragged):
        df = ragged

        result = df.rolling(window="1s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, op, result_data",
        [
            ("ms", "min", [0.0] * 10),
            ("ms", "mean", [0.0] * 9 + [2.0 / 9]),
            ("ms", "max", [0.0] * 9 + [2.0]),
            ("s", "min", [0.0] * 10),
            ("s", "mean", [0.0] * 9 + [2.0 / 9]),
            ("s", "max", [0.0] * 9 + [2.0]),
            ("min", "min", [0.0] * 10),
            ("min", "mean", [0.0] * 9 + [2.0 / 9]),
            ("min", "max", [0.0] * 9 + [2.0]),
            ("h", "min", [0.0] * 10),
            ("h", "mean", [0.0] * 9 + [2.0 / 9]),
            ("h", "max", [0.0] * 9 + [2.0]),
            ("D", "min", [0.0] * 10),
            ("D", "mean", [0.0] * 9 + [2.0 / 9]),
            ("D", "max", [0.0] * 9 + [2.0]),
        ],
    )
    def test_freqs_ops(self, freq, op, result_data):
        # GH 21096
        index = date_range(start="2018-1-1 01:00:00", freq=f"1{freq}", periods=10)
        # Explicit cast to float to avoid implicit cast when setting nan
        s = Series(data=0, index=index, dtype="float")
        s.iloc[1] = np.nan
        s.iloc[-1] = 2
        result = getattr(s.rolling(window=f"10{freq}"), op)()
        expected = Series(data=result_data, index=index)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            "count",
            "median",
            "std",
            "var",
            "kurt",
            "skew",
            "min",
            "max",
        ],
    )
    def test_all(self, f, regular):
        # simple comparison of integer vs time-based windowing
        df = regular * 2
        er = df.rolling(window=1)
        r = df.rolling(window="1s")

        result = getattr(r, f)()
        expected = getattr(er, f)()
        tm.assert_frame_equal(result, expected)

        result = r.quantile(0.5)
        expected = er.quantile(0.5)
        tm.assert_frame_equal(result, expected)

    def test_all2(self, arithmetic_win_operators):
        f = arithmetic_win_operators
        # more sophisticated comparison of integer vs.
        # time-based windowing
        df = DataFrame(
            {"B": np.arange(50)}, index=date_range("20130101", periods=50, freq="h")
        )
        # in-range data
        dft = df.between_time("09:00", "16:00")

        r = dft.rolling(window="5h")

        result = getattr(r, f)()

        # we need to roll the days separately
        # to compare with a time-based roll
        # finally groupby-apply will return a multi-index
        # so we need to drop the day
        def agg_by_day(x):
            x = x.between_time("09:00", "16:00")
            return getattr(x.rolling(5, min_periods=1), f)()

        expected = (
            df.groupby(df.index.day).apply(agg_by_day).reset_index(level=0, drop=True)
        )

        tm.assert_frame_equal(result, expected)

    def test_rolling_cov_offset(self):
        # GH16058

        idx = date_range("2017-01-01", periods=24, freq="1h")
        ss = Series(np.arange(len(idx)), index=idx)

        result = ss.rolling("2h").cov()
        expected = Series([np.nan] + [0.5] * (len(idx) - 1), index=idx)
        tm.assert_series_equal(result, expected)

        expected2 = ss.rolling(2, min_periods=1).cov()
        tm.assert_series_equal(result, expected2)

        result = ss.rolling("3h").cov()
        expected = Series([np.nan, 0.5] + [1.0] * (len(idx) - 2), index=idx)
        tm.assert_series_equal(result, expected)

        expected2 = ss.rolling(3, min_periods=1).cov()
        tm.assert_series_equal(result, expected2)

    def test_rolling_on_decreasing_index(self, unit):
        # GH-19248, GH-32385
        index = DatetimeIndex(
            [
                Timestamp("20190101 09:00:30"),
                Timestamp("20190101 09:00:27"),
                Timestamp("20190101 09:00:20"),
                Timestamp("20190101 09:00:18"),
                Timestamp("20190101 09:00:10"),
            ]
        ).as_unit(unit)

        df = DataFrame({"column": [3, 4, 4, 5, 6]}, index=index)
        result = df.rolling("5s").min()
        expected = DataFrame({"column": [3.0, 3.0, 4.0, 4.0, 6.0]}, index=index)
        tm.assert_frame_equal(result, expected)

    def test_rolling_on_empty(self):
        # GH-32385
        df = DataFrame({"column": []}, index=[])
        result = df.rolling("5s").min()
        expected = DataFrame({"column": []}, index=[])
        tm.assert_frame_equal(result, expected)

    def test_rolling_on_multi_index_level(self):
        # GH-15584
        df = DataFrame(
            {"column": range(6)},
            index=MultiIndex.from_product(
                [date_range("20190101", periods=3), range(2)], names=["date", "seq"]
            ),
        )
        result = df.rolling("10d", on=df.index.get_level_values("date")).sum()
        expected = DataFrame(
            {"column": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]}, index=df.index
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("msg, axis", [["column", 1], ["index", 0]])
def test_nat_axis_error(msg, axis):
    idx = [Timestamp("2020"), NaT]
    kwargs = {"columns" if axis == 1 else "index": idx}
    df = DataFrame(np.eye(2), **kwargs)
    warn_msg = "The 'axis' keyword in DataFrame.rolling is deprecated"
    if axis == 1:
        warn_msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with pytest.raises(ValueError, match=f"{msg} values must not have NaT"):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            df.rolling("D", axis=axis).mean()


@td.skip_if_no("pyarrow")
def test_arrow_datetime_axis():
    # GH 55849
    expected = Series(
        np.arange(5, dtype=np.float64),
        index=Index(
            date_range("2020-01-01", periods=5), dtype="timestamp[ns][pyarrow]"
        ),
    )
    result = expected.rolling("1D").sum()
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_manual.py ===
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.function import (Derivative, Function, diff, expand)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, csch, cosh, coth, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, cos, cot, csc, sec, sin, tan)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.polynomials import (assoc_laguerre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.zeta_functions import polylog
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import And
from sympy.integrals.manualintegrate import (manualintegrate, find_substitutions,
    _parts_rule, integral_steps, manual_subs)
from sympy.testing.pytest import raises, slow

x, y, z, u, n, a, b, c, d, e = symbols('x y z u n a b c d e')
f = Function('f')


def assert_is_integral_of(f: Expr, F: Expr):
    assert manualintegrate(f, x) == F
    assert F.diff(x).equals(f)


def test_find_substitutions():
    assert find_substitutions((cot(x)**2 + 1)**2*csc(x)**2*cot(x)**2, x, u) == \
        [(cot(x), 1, -u**6 - 2*u**4 - u**2)]
    assert find_substitutions((sec(x)**2 + tan(x) * sec(x)) / (sec(x) + tan(x)),
                              x, u) == [(sec(x) + tan(x), 1, 1/u)]
    assert (-x**2, Rational(-1, 2), exp(u)) in find_substitutions(x * exp(-x**2), x, u)
    assert not find_substitutions(Derivative(f(x), x)**2, x, u)


def test_manualintegrate_polynomials():
    assert manualintegrate(y, x) == x*y
    assert manualintegrate(exp(2), x) == x * exp(2)
    assert manualintegrate(x**2, x) == x**3 / 3
    assert manualintegrate(3 * x**2 + 4 * x**3, x) == x**3 + x**4

    assert manualintegrate((x + 2)**3, x) == (x + 2)**4 / 4
    assert manualintegrate((3*x + 4)**2, x) == (3*x + 4)**3 / 9

    assert manualintegrate((u + 2)**3, u) == (u + 2)**4 / 4
    assert manualintegrate((3*u + 4)**2, u) == (3*u + 4)**3 / 9


def test_manualintegrate_exponentials():
    assert manualintegrate(exp(2*x), x) == exp(2*x) / 2
    assert manualintegrate(2**x, x) == (2 ** x) / log(2)
    assert_is_integral_of(1/sqrt(1-exp(2*x)),
                          log(sqrt(1 - exp(2*x)) - 1)/2 - log(sqrt(1 - exp(2*x)) + 1)/2)

    assert manualintegrate(1 / x, x) == log(x)
    assert manualintegrate(1 / (2*x + 3), x) == log(2*x + 3) / 2
    assert manualintegrate(log(x)**2 / x, x) == log(x)**3 / 3

    assert_is_integral_of(x**x*(log(x)+1), x**x)


def test_manualintegrate_parts():
    assert manualintegrate(exp(x) * sin(x), x) == \
        (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    assert manualintegrate(2*x*cos(x), x) == 2*x*sin(x) + 2*cos(x)
    assert manualintegrate(x * log(x), x) == x**2*log(x)/2 - x**2/4
    assert manualintegrate(log(x), x) == x * log(x) - x
    assert manualintegrate((3*x**2 + 5) * exp(x), x) == \
        3*x**2*exp(x) - 6*x*exp(x) + 11*exp(x)
    assert manualintegrate(atan(x), x) == x*atan(x) - log(x**2 + 1)/2

    # Make sure _parts_rule doesn't pick u = constant but can pick dv =
    # constant if necessary, e.g. for integrate(atan(x))
    assert _parts_rule(cos(x), x) == None
    assert _parts_rule(exp(x), x) == None
    assert _parts_rule(x**2, x) == None
    result = _parts_rule(atan(x), x)
    assert result[0] == atan(x) and result[1] == 1


def test_manualintegrate_trigonometry():
    assert manualintegrate(sin(x), x) == -cos(x)
    assert manualintegrate(tan(x), x) == -log(cos(x))

    assert manualintegrate(sec(x), x) == log(sec(x) + tan(x))
    assert manualintegrate(csc(x), x) == -log(csc(x) + cot(x))

    assert manualintegrate(sin(x) * cos(x), x) in [sin(x) ** 2 / 2, -cos(x)**2 / 2]
    assert manualintegrate(-sec(x) * tan(x), x) == -sec(x)
    assert manualintegrate(csc(x) * cot(x), x) == -csc(x)
    assert manualintegrate(sec(x)**2, x) == tan(x)
    assert manualintegrate(csc(x)**2, x) == -cot(x)

    assert manualintegrate(x * sec(x**2), x) == log(tan(x**2) + sec(x**2))/2
    assert manualintegrate(cos(x)*csc(sin(x)), x) == -log(cot(sin(x)) + csc(sin(x)))
    assert manualintegrate(cos(3*x)*sec(x), x) == -x + sin(2*x)
    assert manualintegrate(sin(3*x)*sec(x), x) == \
        -3*log(cos(x)) + 2*log(cos(x)**2) - 2*cos(x)**2

    assert_is_integral_of(sinh(2*x), cosh(2*x)/2)
    assert_is_integral_of(x*cosh(x**2), sinh(x**2)/2)
    assert_is_integral_of(tanh(x), log(cosh(x)))
    assert_is_integral_of(coth(x), log(sinh(x)))
    f, F = sech(x), 2*atan(tanh(x/2))
    assert manualintegrate(f, x) == F
    assert (F.diff(x) - f).rewrite(exp).simplify() == 0  # todo: equals returns None
    f, F = csch(x), log(tanh(x/2))
    assert manualintegrate(f, x) == F
    assert (F.diff(x) - f).rewrite(exp).simplify() == 0


@slow
def test_manualintegrate_trigpowers():
    assert manualintegrate(sin(x)**2 * cos(x), x) == sin(x)**3 / 3
    assert manualintegrate(sin(x)**2 * cos(x) **2, x) == \
        x / 8 - sin(4*x) / 32
    assert manualintegrate(sin(x) * cos(x)**3, x) == -cos(x)**4 / 4
    assert manualintegrate(sin(x)**3 * cos(x)**2, x) == \
        cos(x)**5 / 5 - cos(x)**3 / 3

    assert manualintegrate(tan(x)**3 * sec(x), x) == sec(x)**3/3 - sec(x)
    assert manualintegrate(tan(x) * sec(x) **2, x) == sec(x)**2/2

    assert manualintegrate(cot(x)**5 * csc(x), x) == \
        -csc(x)**5/5 + 2*csc(x)**3/3 - csc(x)
    assert manualintegrate(cot(x)**2 * csc(x)**6, x) == \
        -cot(x)**7/7 - 2*cot(x)**5/5 - cot(x)**3/3


@slow
def test_manualintegrate_inversetrig():
    # atan
    assert manualintegrate(exp(x) / (1 + exp(2*x)), x) == atan(exp(x))
    assert manualintegrate(1 / (4 + 9 * x**2), x) == atan(3 * x/2) / 6
    assert manualintegrate(1 / (16 + 16 * x**2), x) == atan(x) / 16
    assert manualintegrate(1 / (4 + x**2), x) == atan(x / 2) / 2
    assert manualintegrate(1 / (1 + 4 * x**2), x) == atan(2*x) / 2
    ra = Symbol('a', real=True)
    rb = Symbol('b', real=True)
    assert manualintegrate(1/(ra + rb*x**2), x) == \
        Piecewise((atan(x/sqrt(ra/rb))/(rb*sqrt(ra/rb)), ra/rb > 0),
                  ((log(x - sqrt(-ra/rb)) - log(x + sqrt(-ra/rb)))/(2*sqrt(rb)*sqrt(-ra)), True))
    assert manualintegrate(1/(4 + rb*x**2), x) == \
        Piecewise((atan(x/(2*sqrt(1/rb)))/(2*rb*sqrt(1/rb)), 1/rb > 0),
                  (-I*(log(x - 2*sqrt(-1/rb)) - log(x + 2*sqrt(-1/rb)))/(4*sqrt(rb)), True))
    assert manualintegrate(1/(ra + 4*x**2), x) == \
        Piecewise((atan(2*x/sqrt(ra))/(2*sqrt(ra)), ra > 0),
                  ((log(x - sqrt(-ra)/2) - log(x + sqrt(-ra)/2))/(4*sqrt(-ra)), True))
    assert manualintegrate(1/(4 + 4*x**2), x) == atan(x) / 4

    assert manualintegrate(1/(a + b*x**2), x) == Piecewise((atan(x/sqrt(a/b))/(b*sqrt(a/b)), Ne(a, 0)),
                                                           (-1/(b*x), True))

    # asin
    assert manualintegrate(1/sqrt(1-x**2), x) == asin(x)
    assert manualintegrate(1/sqrt(4-4*x**2), x) == asin(x)/2
    assert manualintegrate(3/sqrt(1-9*x**2), x) == asin(3*x)
    assert manualintegrate(1/sqrt(4-9*x**2), x) == asin(x*Rational(3, 2))/3

    # asinh
    assert manualintegrate(1/sqrt(x**2 + 1), x) == \
        asinh(x)
    assert manualintegrate(1/sqrt(x**2 + 4), x) == \
        asinh(x/2)
    assert manualintegrate(1/sqrt(4*x**2 + 4), x) == \
        asinh(x)/2
    assert manualintegrate(1/sqrt(4*x**2 + 1), x) == \
        asinh(2*x)/2
    assert manualintegrate(1/sqrt(ra*x**2 + 1), x) == \
        Piecewise((asin(x*sqrt(-ra))/sqrt(-ra), ra < 0), (asinh(sqrt(ra)*x)/sqrt(ra), ra > 0), (x, True))
    assert manualintegrate(1/sqrt(ra + x**2), x) == \
        Piecewise((asinh(x*sqrt(1/ra)), ra > 0), (log(2*x + 2*sqrt(ra + x**2)), True))

    # log
    assert manualintegrate(1/sqrt(x**2 - 1), x) == log(2*x + 2*sqrt(x**2 - 1))
    assert manualintegrate(1/sqrt(x**2 - 4), x) == log(2*x + 2*sqrt(x**2 - 4))
    assert manualintegrate(1/sqrt(4*x**2 - 4), x) == log(8*x + 4*sqrt(4*x**2 - 4))/2
    assert manualintegrate(1/sqrt(9*x**2 - 1), x) == log(18*x + 6*sqrt(9*x**2 - 1))/3
    assert manualintegrate(1/sqrt(ra*x**2 - 4), x) == \
           Piecewise((log(2*sqrt(ra)*sqrt(ra*x**2 - 4) + 2*ra*x)/sqrt(ra), Ne(ra, 0)), (-I*x/2, True))
    assert manualintegrate(1/sqrt(-ra + 4*x**2), x) == \
        Piecewise((asinh(2*x*sqrt(-1/ra))/2, ra < 0), (log(8*x + 4*sqrt(-ra + 4*x**2))/2, True))

    # From https://www.wikiwand.com/en/List_of_integrals_of_inverse_trigonometric_functions
    # asin
    assert manualintegrate(asin(x), x) == x*asin(x) + sqrt(1 - x**2)
    assert manualintegrate(asin(a*x), x) == Piecewise(((a*x*asin(a*x) + sqrt(-a**2*x**2 + 1))/a, Ne(a, 0)), (0, True))
    assert manualintegrate(x*asin(a*x), x) == \
           -a*Piecewise((-x*sqrt(-a**2*x**2 + 1)/(2*a**2) +
                         log(-2*a**2*x + 2*sqrt(-a**2)*sqrt(-a**2*x**2 + 1))/(2*a**2*sqrt(-a**2)), Ne(a**2, 0)),
                        (x**3/3, True))/2 + x**2*asin(a*x)/2
    # acos
    assert manualintegrate(acos(x), x) == x*acos(x) - sqrt(1 - x**2)
    assert manualintegrate(acos(a*x), x) == Piecewise(((a*x*acos(a*x) - sqrt(-a**2*x**2 + 1))/a, Ne(a, 0)), (pi*x/2, True))
    assert manualintegrate(x*acos(a*x), x) == \
           a*Piecewise((-x*sqrt(-a**2*x**2 + 1)/(2*a**2) +
                        log(-2*a**2*x + 2*sqrt(-a**2)*sqrt(-a**2*x**2 + 1))/(2*a**2*sqrt(-a**2)), Ne(a**2, 0)),
                       (x**3/3, True))/2 + x**2*acos(a*x)/2
    # atan
    assert manualintegrate(atan(x), x) == x*atan(x) - log(x**2 + 1)/2
    assert manualintegrate(atan(a*x), x) == Piecewise(((a*x*atan(a*x) - log(a**2*x**2 + 1)/2)/a, Ne(a, 0)), (0, True))
    assert manualintegrate(x*atan(a*x), x) == -a*(x/a**2 - atan(x/sqrt(a**(-2)))/(a**4*sqrt(a**(-2))))/2 + x**2*atan(a*x)/2
    # acsc
    assert manualintegrate(acsc(x), x) == x*acsc(x) + Integral(1/(x*sqrt(1 - 1/x**2)), x)
    assert manualintegrate(acsc(a*x), x) == x*acsc(a*x) + Integral(1/(x*sqrt(1 - 1/(a**2*x**2))), x)/a
    assert manualintegrate(x*acsc(a*x), x) == x**2*acsc(a*x)/2 + Integral(1/sqrt(1 - 1/(a**2*x**2)), x)/(2*a)
    # asec
    assert manualintegrate(asec(x), x) == x*asec(x) - Integral(1/(x*sqrt(1 - 1/x**2)), x)
    assert manualintegrate(asec(a*x), x) == x*asec(a*x) - Integral(1/(x*sqrt(1 - 1/(a**2*x**2))), x)/a
    assert manualintegrate(x*asec(a*x), x) == x**2*asec(a*x)/2 - Integral(1/sqrt(1 - 1/(a**2*x**2)), x)/(2*a)
    # acot
    assert manualintegrate(acot(x), x) == x*acot(x) + log(x**2 + 1)/2
    assert manualintegrate(acot(a*x), x) == Piecewise(((a*x*acot(a*x) + log(a**2*x**2 + 1)/2)/a, Ne(a, 0)), (pi*x/2, True))
    assert manualintegrate(x*acot(a*x), x) == a*(x/a**2 - atan(x/sqrt(a**(-2)))/(a**4*sqrt(a**(-2))))/2 + x**2*acot(a*x)/2

    # piecewise
    assert manualintegrate(1/sqrt(ra-rb*x**2), x) == \
        Piecewise((asin(x*sqrt(rb/ra))/sqrt(rb), And(-rb < 0, ra > 0)),
                  (asinh(x*sqrt(-rb/ra))/sqrt(-rb), And(-rb > 0, ra > 0)),
                  (log(-2*rb*x + 2*sqrt(-rb)*sqrt(ra - rb*x**2))/sqrt(-rb), Ne(rb, 0)),
                  (x/sqrt(ra), True))
    assert manualintegrate(1/sqrt(ra + rb*x**2), x) == \
        Piecewise((asin(x*sqrt(-rb/ra))/sqrt(-rb), And(ra > 0, rb < 0)),
                  (asinh(x*sqrt(rb/ra))/sqrt(rb), And(ra > 0, rb > 0)),
                  (log(2*sqrt(rb)*sqrt(ra + rb*x**2) + 2*rb*x)/sqrt(rb), Ne(rb, 0)),
                  (x/sqrt(ra), True))


def test_manualintegrate_trig_substitution():
    assert manualintegrate(sqrt(16*x**2 - 9)/x, x) == \
        Piecewise((sqrt(16*x**2 - 9) - 3*acos(3/(4*x)),
                   And(x < Rational(3, 4), x > Rational(-3, 4))))
    assert manualintegrate(1/(x**4 * sqrt(25-x**2)), x) == \
        Piecewise((-sqrt(-x**2/25 + 1)/(125*x) -
                   (-x**2/25 + 1)**(3*S.Half)/(15*x**3), And(x < 5, x > -5)))
    assert manualintegrate(x**7/(49*x**2 + 1)**(3 * S.Half), x) == \
        ((49*x**2 + 1)**(5*S.Half)/28824005 -
         (49*x**2 + 1)**(3*S.Half)/5764801 +
         3*sqrt(49*x**2 + 1)/5764801 + 1/(5764801*sqrt(49*x**2 + 1)))

def test_manualintegrate_trivial_substitution():
    assert manualintegrate((exp(x) - exp(-x))/x, x) == -Ei(-x) + Ei(x)
    f = Function('f')
    assert manualintegrate((f(x) - f(-x))/x, x) == \
        -Integral(f(-x)/x, x) + Integral(f(x)/x, x)


def test_manualintegrate_rational():
    assert manualintegrate(1/(4 - x**2), x) == -log(x - 2)/4 + log(x + 2)/4
    assert manualintegrate(1/(-1 + x**2), x) == log(x - 1)/2 - log(x + 1)/2


def test_manualintegrate_special():
    f, F = 4*exp(-x**2/3), 2*sqrt(3)*sqrt(pi)*erf(sqrt(3)*x/3)
    assert_is_integral_of(f, F)
    f, F = 3*exp(4*x**2), 3*sqrt(pi)*erfi(2*x)/4
    assert_is_integral_of(f, F)
    f, F = x**Rational(1, 3)*exp(-x/8), -16*uppergamma(Rational(4, 3), x/8)
    assert_is_integral_of(f, F)
    f, F = exp(2*x)/x, Ei(2*x)
    assert_is_integral_of(f, F)
    f, F = exp(1 + 2*x - x**2), sqrt(pi)*exp(2)*erf(x - 1)/2
    assert_is_integral_of(f, F)
    f = sin(x**2 + 4*x + 1)
    F = (sqrt(2)*sqrt(pi)*(-sin(3)*fresnelc(sqrt(2)*(2*x + 4)/(2*sqrt(pi))) +
        cos(3)*fresnels(sqrt(2)*(2*x + 4)/(2*sqrt(pi))))/2)
    assert_is_integral_of(f, F)
    f, F = cos(4*x**2), sqrt(2)*sqrt(pi)*fresnelc(2*sqrt(2)*x/sqrt(pi))/4
    assert_is_integral_of(f, F)
    f, F = sin(3*x + 2)/x, sin(2)*Ci(3*x) + cos(2)*Si(3*x)
    assert_is_integral_of(f, F)
    f, F = sinh(3*x - 2)/x, -sinh(2)*Chi(3*x) + cosh(2)*Shi(3*x)
    assert_is_integral_of(f, F)
    f, F = 5*cos(2*x - 3)/x, 5*cos(3)*Ci(2*x) + 5*sin(3)*Si(2*x)
    assert_is_integral_of(f, F)
    f, F = cosh(x/2)/x, Chi(x/2)
    assert_is_integral_of(f, F)
    f, F = cos(x**2)/x, Ci(x**2)/2
    assert_is_integral_of(f, F)
    f, F = 1/log(2*x + 1), li(2*x + 1)/2
    assert_is_integral_of(f, F)
    f, F = polylog(2, 5*x)/x, polylog(3, 5*x)
    assert_is_integral_of(f, F)
    f, F = 5/sqrt(3 - 2*sin(x)**2), 5*sqrt(3)*elliptic_f(x, Rational(2, 3))/3
    assert_is_integral_of(f, F)
    f, F = sqrt(4 + 9*sin(x)**2), 2*elliptic_e(x, Rational(-9, 4))
    assert_is_integral_of(f, F)


def test_manualintegrate_derivative():
    assert manualintegrate(pi * Derivative(x**2 + 2*x + 3), x) == \
        pi * (x**2 + 2*x + 3)
    assert manualintegrate(Derivative(x**2 + 2*x + 3, y), x) == \
        Integral(Derivative(x**2 + 2*x + 3, y))
    assert manualintegrate(Derivative(sin(x), x, x, x, y), x) == \
        Derivative(sin(x), x, x, y)


def test_manualintegrate_Heaviside():
    assert_is_integral_of(DiracDelta(3*x+2), Heaviside(3*x+2)/3)
    assert_is_integral_of(DiracDelta(3*x, 0), Heaviside(3*x)/3)
    assert manualintegrate(DiracDelta(a+b*x, 1), x) == \
        Piecewise((DiracDelta(a + b*x)/b, Ne(b, 0)), (x*DiracDelta(a, 1), True))
    assert_is_integral_of(DiracDelta(x/3-1, 2), 3*DiracDelta(x/3-1, 1))
    assert manualintegrate(Heaviside(x), x) == x*Heaviside(x)
    assert manualintegrate(x*Heaviside(2), x) == x**2/2
    assert manualintegrate(x*Heaviside(-2), x) == 0
    assert manualintegrate(x*Heaviside( x), x) == x**2*Heaviside( x)/2
    assert manualintegrate(x*Heaviside(-x), x) == x**2*Heaviside(-x)/2
    assert manualintegrate(Heaviside(2*x + 4), x) == (x+2)*Heaviside(2*x + 4)
    assert manualintegrate(x*Heaviside(x), x) == x**2*Heaviside(x)/2
    assert manualintegrate(Heaviside(x + 1)*Heaviside(1 - x)*x**2, x) == \
        ((x**3/3 + Rational(1, 3))*Heaviside(x + 1) - Rational(2, 3))*Heaviside(-x + 1)

    y = Symbol('y')
    assert manualintegrate(sin(7 + x)*Heaviside(3*x - 7), x) == \
            (- cos(x + 7) + cos(Rational(28, 3)))*Heaviside(3*x - S(7))

    assert manualintegrate(sin(y + x)*Heaviside(3*x - y), x) == \
            (cos(y*Rational(4, 3)) - cos(x + y))*Heaviside(3*x - y)


def test_manualintegrate_orthogonal_poly():
    n = symbols('n')
    a, b = 7, Rational(5, 3)
    polys = [jacobi(n, a, b, x), gegenbauer(n, a, x), chebyshevt(n, x),
        chebyshevu(n, x), legendre(n, x), hermite(n, x), laguerre(n, x),
        assoc_laguerre(n, a, x)]
    for p in polys:
        integral = manualintegrate(p, x)
        for deg in [-2, -1, 0, 1, 3, 5, 8]:
            # some accept negative "degree", some do not
            try:
                p_subbed = p.subs(n, deg)
            except ValueError:
                continue
            assert (integral.subs(n, deg).diff(x) - p_subbed).expand() == 0

        # can also integrate simple expressions with these polynomials
        q = x*p.subs(x, 2*x + 1)
        integral = manualintegrate(q, x)
        for deg in [2, 4, 7]:
            assert (integral.subs(n, deg).diff(x) - q.subs(n, deg)).expand() == 0

        # cannot integrate with respect to any other parameter
        t = symbols('t')
        for i in range(len(p.args) - 1):
            new_args = list(p.args)
            new_args[i] = t
            assert isinstance(manualintegrate(p.func(*new_args), t), Integral)


@slow
def test_issue_6799():
    r, x, phi = map(Symbol, 'r x phi'.split())
    n = Symbol('n', integer=True, positive=True)

    integrand = (cos(n*(x-phi))*cos(n*x))
    limits = (x, -pi, pi)
    assert manualintegrate(integrand, x) == \
        ((n*x/2 + sin(2*n*x)/4)*cos(n*phi) - sin(n*phi)*cos(n*x)**2/2)/n
    assert r * integrate(integrand, limits).trigsimp() / pi == r * cos(n * phi)
    assert not integrate(integrand, limits).has(Dummy)


def test_issue_12251():
    assert manualintegrate(x**y, x) == Piecewise(
        (x**(y + 1)/(y + 1), Ne(y, -1)), (log(x), True))


def test_issue_3796():
    assert manualintegrate(diff(exp(x + x**2)), x) == exp(x + x**2)
    assert integrate(x * exp(x**4), x, risch=False) == -I*sqrt(pi)*erf(I*x**2)/4


def test_manual_true():
    assert integrate(exp(x) * sin(x), x, manual=True) == \
        (exp(x) * sin(x)) / 2 - (exp(x) * cos(x)) / 2
    assert integrate(sin(x) * cos(x), x, manual=True) in \
        [sin(x) ** 2 / 2, -cos(x)**2 / 2]


def test_issue_6746():
    y = Symbol('y')
    n = Symbol('n')
    assert manualintegrate(y**x, x) == Piecewise(
        (y**x/log(y), Ne(log(y), 0)), (x, True))
    assert manualintegrate(y**(n*x), x) == Piecewise(
        (Piecewise(
            (y**(n*x)/log(y), Ne(log(y), 0)),
            (n*x, True)
        )/n, Ne(n, 0)),
        (x, True))
    assert manualintegrate(exp(n*x), x) == Piecewise(
        (exp(n*x)/n, Ne(n, 0)), (x, True))

    y = Symbol('y', positive=True)
    assert manualintegrate((y + 1)**x, x) == (y + 1)**x/log(y + 1)
    y = Symbol('y', zero=True)
    assert manualintegrate((y + 1)**x, x) == x
    y = Symbol('y')
    n = Symbol('n', nonzero=True)
    assert manualintegrate(y**(n*x), x) == Piecewise(
        (y**(n*x)/log(y), Ne(log(y), 0)), (n*x, True))/n
    y = Symbol('y', positive=True)
    assert manualintegrate((y + 1)**(n*x), x) == \
        (y + 1)**(n*x)/(n*log(y + 1))
    a = Symbol('a', negative=True)
    b = Symbol('b')
    assert manualintegrate(1/(a + b*x**2), x) == atan(x/sqrt(a/b))/(b*sqrt(a/b))
    b = Symbol('b', negative=True)
    assert manualintegrate(1/(a + b*x**2), x) == \
        atan(x/(sqrt(-a)*sqrt(-1/b)))/(b*sqrt(-a)*sqrt(-1/b))
    assert manualintegrate(1/((x**a + y**b + 4)*sqrt(a*x**2 + 1)), x) == \
        y**(-b)*Integral(x**(-a)/(y**(-b)*sqrt(a*x**2 + 1) +
        x**(-a)*sqrt(a*x**2 + 1) + 4*x**(-a)*y**(-b)*sqrt(a*x**2 + 1)), x)
    assert manualintegrate(1/((x**2 + 4)*sqrt(4*x**2 + 1)), x) == \
        Integral(1/((x**2 + 4)*sqrt(4*x**2 + 1)), x)
    assert manualintegrate(1/(x - a**x + x*b**2), x) == \
        Integral(1/(-a**x + b**2*x + x), x)


@slow
def test_issue_2850():
    assert manualintegrate(asin(x)*log(x), x) == -x*asin(x) - sqrt(-x**2 + 1) \
            + (x*asin(x) + sqrt(-x**2 + 1))*log(x) - Integral(sqrt(-x**2 + 1)/x, x)
    assert manualintegrate(acos(x)*log(x), x) == -x*acos(x) + sqrt(-x**2 + 1) + \
        (x*acos(x) - sqrt(-x**2 + 1))*log(x) + Integral(sqrt(-x**2 + 1)/x, x)
    assert manualintegrate(atan(x)*log(x), x) == -x*atan(x) + (x*atan(x) - \
            log(x**2 + 1)/2)*log(x) + log(x**2 + 1)/2 + Integral(log(x**2 + 1)/x, x)/2


def test_issue_9462():
    assert manualintegrate(sin(2*x)*exp(x), x) == exp(x)*sin(2*x)/5 - 2*exp(x)*cos(2*x)/5
    assert not integral_steps(sin(2*x)*exp(x), x).contains_dont_know()
    assert manualintegrate((x - 3) / (x**2 - 2*x + 2)**2, x) == \
                           Integral(x/(x**4 - 4*x**3 + 8*x**2 - 8*x + 4), x) \
                           - 3*Integral(1/(x**4 - 4*x**3 + 8*x**2 - 8*x + 4), x)


def test_cyclic_parts():
    f = cos(x)*exp(x/4)
    F = 16*exp(x/4)*sin(x)/17 + 4*exp(x/4)*cos(x)/17
    assert manualintegrate(f, x) == F and F.diff(x) == f
    f = x*cos(x)*exp(x/4)
    F = (x*(16*exp(x/4)*sin(x)/17 + 4*exp(x/4)*cos(x)/17) -
        128*exp(x/4)*sin(x)/289 + 240*exp(x/4)*cos(x)/289)
    assert manualintegrate(f, x) == F and F.diff(x) == f


@slow
def test_issue_10847_slow():
    assert manualintegrate((4*x**4 + 4*x**3 + 16*x**2 + 12*x + 8)
                           / (x**6 + 2*x**5 + 3*x**4 + 4*x**3 + 3*x**2 + 2*x + 1), x) == \
                           2*x/(x**2 + 1) + 3*atan(x) - 1/(x**2 + 1) - 3/(x + 1)


@slow
def test_issue_10847():

    assert manualintegrate(x**2 / (x**2 - c), x) == \
        c*Piecewise((atan(x/sqrt(-c))/sqrt(-c), Ne(c, 0)), (-1/x, True)) + x

    rc = Symbol('c', real=True)
    assert manualintegrate(x**2 / (x**2 - rc), x) == \
        rc*Piecewise((atan(x/sqrt(-rc))/sqrt(-rc), rc < 0),
                     ((log(-sqrt(rc) + x) - log(sqrt(rc) + x))/(2*sqrt(rc)), True)) + x

    assert manualintegrate(sqrt(x - y) * log(z / x), x) == \
        4*y**2*Piecewise((atan(sqrt(x - y)/sqrt(y))/sqrt(y), Ne(y, 0)),
                         (-1/sqrt(x - y), True))/3 - 4*y*sqrt(x - y)/3 + \
        2*(x - y)**Rational(3, 2)*log(z/x)/3 + 4*(x - y)**Rational(3, 2)/9
    ry = Symbol('y', real=True)
    rz = Symbol('z', real=True)
    assert manualintegrate(sqrt(x - ry) * log(rz / x), x) == \
        4*ry**2*Piecewise((atan(sqrt(x - ry)/sqrt(ry))/sqrt(ry), ry > 0),
                         ((log(-sqrt(-ry) + sqrt(x - ry)) - log(sqrt(-ry) + sqrt(x - ry)))/(2*sqrt(-ry)), True))/3 \
                         - 4*ry*sqrt(x - ry)/3 + 2*(x - ry)**Rational(3, 2)*log(rz/x)/3 \
                         + 4*(x - ry)**Rational(3, 2)/9

    assert manualintegrate(sqrt(x) * log(x), x) == 2*x**Rational(3, 2)*log(x)/3 - 4*x**Rational(3, 2)/9

    result = manualintegrate(sqrt(a*x + b) / x, x)
    assert result == Piecewise((-2*b*Piecewise(
        (-atan(sqrt(a*x + b)/sqrt(-b))/sqrt(-b), Ne(b, 0)),
        (1/sqrt(a*x + b), True)) + 2*sqrt(a*x + b), Ne(a, 0)),
        (sqrt(b)*log(x), True))
    assert piecewise_fold(result) == Piecewise(
        (2*b*atan(sqrt(a*x + b)/sqrt(-b))/sqrt(-b) + 2*sqrt(a*x + b), Ne(a, 0) & Ne(b, 0)),
        (-2*b/sqrt(a*x + b) + 2*sqrt(a*x + b), Ne(a, 0)),
        (sqrt(b)*log(x), True))

    ra = Symbol('a', real=True)
    rb = Symbol('b', real=True)
    assert manualintegrate(sqrt(ra*x + rb) / x, x) == \
        Piecewise(
            (-2*rb*Piecewise(
                (-atan(sqrt(ra*x + rb)/sqrt(-rb))/sqrt(-rb), rb < 0),
                (-I*(log(-sqrt(rb) + sqrt(ra*x + rb)) - log(sqrt(rb) + sqrt(ra*x + rb)))/(2*sqrt(-rb)), True)) +
             2*sqrt(ra*x + rb), Ne(ra, 0)),
            (sqrt(rb)*log(x), True))

    assert expand(manualintegrate(sqrt(ra*x + rb) / (x + rc), x)) == \
           Piecewise((-2*ra*rc*Piecewise((atan(sqrt(ra*x + rb)/sqrt(ra*rc - rb))/sqrt(ra*rc - rb), ra*rc - rb > 0),
                                       (log(-sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)) -
                                        log(sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)), True)) +
                      2*rb*Piecewise((atan(sqrt(ra*x + rb)/sqrt(ra*rc - rb))/sqrt(ra*rc - rb), ra*rc - rb > 0),
                                    (log(-sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)) -
                                     log(sqrt(-ra*rc + rb) + sqrt(ra*x + rb))/(2*sqrt(-ra*rc + rb)), True)) +
                      2*sqrt(ra*x + rb), Ne(ra, 0)), (sqrt(rb)*log(rc + x), True))

    assert manualintegrate(sqrt(2*x + 3) / (x + 1), x) == 2*sqrt(2*x + 3) - log(sqrt(2*x + 3) + 1) + log(sqrt(2*x + 3) - 1)
    assert manualintegrate(sqrt(2*x + 3) / 2 * x, x) == (2*x + 3)**Rational(5, 2)/20 - (2*x + 3)**Rational(3, 2)/4
    assert manualintegrate(x**Rational(3,2) * log(x), x) == 2*x**Rational(5,2)*log(x)/5 - 4*x**Rational(5,2)/25
    assert manualintegrate(x**(-3) * log(x), x) == -log(x)/(2*x**2) - 1/(4*x**2)
    assert manualintegrate(log(y)/(y**2*(1 - 1/y)), y) == \
        log(y)*log(-1 + 1/y) - Integral(log(-1 + 1/y)/y, y)


def test_issue_12899():
    assert manualintegrate(f(x,y).diff(x),y) == Integral(Derivative(f(x,y),x),y)
    assert manualintegrate(f(x,y).diff(y).diff(x),y) == Derivative(f(x,y),x)


def test_constant_independent_of_symbol():
    assert manualintegrate(Integral(y, (x, 1, 2)), x) == \
        x*Integral(y, (x, 1, 2))


def test_issue_12641():
    assert manualintegrate(sin(2*x), x) == -cos(2*x)/2
    assert manualintegrate(cos(x)*sin(2*x), x) == -2*cos(x)**3/3
    assert manualintegrate((sin(2*x)*cos(x))/(1 + cos(x)), x) == \
        -2*log(cos(x) + 1) - cos(x)**2 + 2*cos(x)


@slow
def test_issue_13297():
    assert manualintegrate(sin(x) * cos(x)**5, x) == -cos(x)**6 / 6


def test_issue_14470():
    assert_is_integral_of(1/(x*sqrt(x + 1)), log(sqrt(x + 1) - 1) - log(sqrt(x + 1) + 1))


@slow
def test_issue_9858():
    assert manualintegrate(exp(x)*cos(exp(x)), x) == sin(exp(x))
    assert manualintegrate(exp(2*x)*cos(exp(x)), x) == \
        exp(x)*sin(exp(x)) + cos(exp(x))
    res = manualintegrate(exp(10*x)*sin(exp(x)), x)
    assert not res.has(Integral)
    assert res.diff(x) == exp(10*x)*sin(exp(x))
    # an example with many similar integrations by parts
    assert manualintegrate(sum(x*exp(k*x) for k in range(1, 8)), x) == (
        x*exp(7*x)/7 + x*exp(6*x)/6 + x*exp(5*x)/5 + x*exp(4*x)/4 +
        x*exp(3*x)/3 + x*exp(2*x)/2 + x*exp(x) - exp(7*x)/49 -exp(6*x)/36 -
        exp(5*x)/25 - exp(4*x)/16 - exp(3*x)/9 - exp(2*x)/4 - exp(x))


def test_issue_8520():
    assert manualintegrate(x/(x**4 + 1), x) == atan(x**2)/2
    assert manualintegrate(x**2/(x**6 + 25), x) == atan(x**3/5)/15
    f = x/(9*x**4 + 4)**2
    assert manualintegrate(f, x).diff(x).factor() == f


def test_manual_subs():
    x, y = symbols('x y')
    expr = log(x) + exp(x)
    # if log(x) is y, then exp(y) is x
    assert manual_subs(expr, log(x), y) == y + exp(exp(y))
    # if exp(x) is y, then log(y) need not be x
    assert manual_subs(expr, exp(x), y) == log(x) + y

    raises(ValueError, lambda: manual_subs(expr, x))
    raises(ValueError, lambda: manual_subs(expr, exp(x), x, y))


@slow
def test_issue_15471():
    f = log(x)*cos(log(x))/x**Rational(3, 4)
    F = -128*x**Rational(1, 4)*sin(log(x))/289 + 240*x**Rational(1, 4)*cos(log(x))/289 + (16*x**Rational(1, 4)*sin(log(x))/17 + 4*x**Rational(1, 4)*cos(log(x))/17)*log(x)
    assert_is_integral_of(f, F)


def test_quadratic_denom():
    f = (5*x + 2)/(3*x**2 - 2*x + 8)
    assert manualintegrate(f, x) == 5*log(3*x**2 - 2*x + 8)/6 + 11*sqrt(23)*atan(3*sqrt(23)*(x - Rational(1, 3))/23)/69
    g = 3/(2*x**2 + 3*x + 1)
    assert manualintegrate(g, x) == 3*log(4*x + 2) - 3*log(4*x + 4)

def test_issue_22757():
    assert manualintegrate(sin(x), y) == y * sin(x)


def test_issue_23348():
    steps = integral_steps(tan(x), x)
    constant_times_step = steps.substep.substep
    assert constant_times_step.integrand == constant_times_step.constant * constant_times_step.other


def test_issue_23566():
    i = Integral(1/sqrt(x**2 - 1), (x, -2, -1)).doit(manual=True)
    assert i == -log(4 - 2*sqrt(3)) + log(2)
    assert str(i.n()) == '1.31695789692482'


def test_issue_25093():
    ap = Symbol('ap', positive=True)
    an = Symbol('an', negative=True)
    assert manualintegrate(exp(a*x**2 + b), x) == sqrt(pi)*exp(b)*erfi(sqrt(a)*x)/(2*sqrt(a))
    assert manualintegrate(exp(ap*x**2 + b), x) == sqrt(pi)*exp(b)*erfi(sqrt(ap)*x)/(2*sqrt(ap))
    assert manualintegrate(exp(an*x**2 + b), x) == -sqrt(pi)*exp(b)*erf(an*x/sqrt(-an))/(2*sqrt(-an))
    assert manualintegrate(sin(a*x**2 + b), x) == (
        sqrt(2)*sqrt(pi)*(sin(b)*fresnelc(sqrt(2)*sqrt(a)*x/sqrt(pi))
        + cos(b)*fresnels(sqrt(2)*sqrt(a)*x/sqrt(pi)))/(2*sqrt(a)))
    assert manualintegrate(cos(a*x**2 + b), x) == (
        sqrt(2)*sqrt(pi)*(-sin(b)*fresnels(sqrt(2)*sqrt(a)*x/sqrt(pi))
        + cos(b)*fresnelc(sqrt(2)*sqrt(a)*x/sqrt(pi)))/(2*sqrt(a)))


def test_nested_pow():
    assert_is_integral_of(sqrt(x**2), x*sqrt(x**2)/2)
    assert_is_integral_of(sqrt(x**(S(5)/3)), 6*x*sqrt(x**(S(5)/3))/11)
    assert_is_integral_of(1/sqrt(x**2), x*log(x)/sqrt(x**2))
    assert_is_integral_of(x*sqrt(x**(-4)), x**2*sqrt(x**-4)*log(x))
    f = (c*(a+b*x)**d)**e
    F1 = (c*(a + b*x)**d)**e*(a/b + x)/(d*e + 1)
    F2 = (c*(a + b*x)**d)**e*(a/b + x)*log(a/b + x)
    assert manualintegrate(f, x) == \
        Piecewise((Piecewise((F1, Ne(d*e, -1)), (F2, True)), Ne(b, 0)), (x*(a**d*c)**e, True))
    assert F1.diff(x).equals(f)
    assert F2.diff(x).subs(d*e, -1).equals(f)


def test_manualintegrate_sqrt_linear():
    assert_is_integral_of((5*x**3+4)/sqrt(2+3*x),
                          10*(3*x + 2)**(S(7)/2)/567 - 4*(3*x + 2)**(S(5)/2)/27 +
                          40*(3*x + 2)**(S(3)/2)/81 + 136*sqrt(3*x + 2)/81)
    assert manualintegrate(x/sqrt(a+b*x)**3, x) == \
        Piecewise((Mul(2, b**-2, a/sqrt(a + b*x) + sqrt(a + b*x)), Ne(b, 0)), (x**2/(2*a**(S(3)/2)), True))
    assert_is_integral_of((sqrt(3*x+3)+1)/((2*x+2)**(1/S(3))+1),
                          3*sqrt(6)*(2*x + 2)**(S(7)/6)/14 - 3*sqrt(6)*(2*x + 2)**(S(5)/6)/10 -
                          3*sqrt(6)*(2*x + 2)**(S.One/6)/2 + 3*(2*x + 2)**(S(2)/3)/4 - 3*(2*x + 2)**(S.One/3)/2 +
                          sqrt(6)*sqrt(2*x + 2)/2 + 3*log((2*x + 2)**(S.One/3) + 1)/2 +
                          3*sqrt(6)*atan((2*x + 2)**(S.One/6))/2)
    assert_is_integral_of(sqrt(x+sqrt(x)),
                          2*sqrt(sqrt(x) + x)*(sqrt(x)/12 + x/3 - S(1)/8) + log(2*sqrt(x) + 2*sqrt(sqrt(x) + x) + 1)/8)
    assert_is_integral_of(sqrt(2*x+3+sqrt(4*x+5))**3,
                          sqrt(2*x + sqrt(4*x + 5) + 3) *
                          (9*x/10 + 11*(4*x + 5)**(S(3)/2)/40 + sqrt(4*x + 5)/40 + (4*x + 5)**2/10 + S(11)/10)/2)


def test_manualintegrate_sqrt_quadratic():
    assert_is_integral_of(1/sqrt((x - I)**2-1), log(2*x + 2*sqrt(x**2 - 2*I*x - 2) - 2*I))
    assert_is_integral_of(1/sqrt(3*x**2+4*x+5), sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/3)
    assert_is_integral_of(1/sqrt(-3*x**2+4*x+5), sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/3)
    assert_is_integral_of(1/sqrt(3*x**2+4*x-5), sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/3)
    assert_is_integral_of(1/sqrt(4*x**2-4*x+1), (x - S.Half)*log(x - S.Half)/(2*sqrt((x - S.Half)**2)))
    assert manualintegrate(1/sqrt(a+b*x+c*x**2), x) == \
        Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(c, 0) & Ne(a - b**2/(4*c), 0)),
                  ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), Ne(c, 0)),
                  (2*sqrt(a + b*x)/b, Ne(b, 0)), (x/sqrt(a), True))

    assert_is_integral_of((7*x+6)/sqrt(3*x**2+4*x+5),
                          7*sqrt(3*x**2 + 4*x + 5)/3 + 4*sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/9)
    assert_is_integral_of((7*x+6)/sqrt(-3*x**2+4*x+5),
                          -7*sqrt(-3*x**2 + 4*x + 5)/3 + 32*sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/9)
    assert_is_integral_of((7*x+6)/sqrt(3*x**2+4*x-5),
                          7*sqrt(3*x**2 + 4*x - 5)/3 + 4*sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/9)
    assert manualintegrate((d+e*x)/sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((-b*e/(2*c) + d) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   e*sqrt(a + b*x + c*x**2)/c, Ne(c, 0)),
                  ((2*d*sqrt(a + b*x) + 2*e*(-a*sqrt(a + b*x) + (a + b*x)**(S(3)/2)/3)/b)/b, Ne(b, 0)),
                  ((d*x + e*x**2/2)/sqrt(a), True))

    assert manualintegrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2), x) == \
        sqrt(x**2 - 3*x + 2)*(x**2 + 13*x/4 + S(101)/8) + 135*log(2*x + 2*sqrt(x**2 - 3*x + 2) - 3)/16

    assert_is_integral_of(sqrt(53225*x**2-66732*x+23013),
                          (x/2 - S(16683)/53225)*sqrt(53225*x**2 - 66732*x + 23013) +
                          111576969*sqrt(2129)*asinh(53225*x/10563 - S(11122)/3521)/1133160250)
    assert manualintegrate(sqrt(a+c*x**2), x) == \
        Piecewise((a*Piecewise((log(2*sqrt(c)*sqrt(a + c*x**2) + 2*c*x)/sqrt(c), Ne(a, 0)),
                               (x*log(x)/sqrt(c*x**2), True))/2 + x*sqrt(a + c*x**2)/2, Ne(c, 0)),
                  (sqrt(a)*x, True))
    assert manualintegrate(sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((a/2 - b**2/(8*c)) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   (b/(4*c) + x/2)*sqrt(a + b*x + c*x**2), Ne(c, 0)),
                  (2*(a + b*x)**(S(3)/2)/(3*b), Ne(b, 0)),
                  (sqrt(a)*x, True))

    assert_is_integral_of(x*sqrt(x**2+2*x+4),
                          (x**2/3 + x/6 + S(5)/6)*sqrt(x**2 + 2*x + 4) - 3*asinh(sqrt(3)*(x + 1)/3)/2)


def test_mul_pow_derivative():
    assert_is_integral_of(x*sec(x)*tan(x), x*sec(x) - log(tan(x) + sec(x)))
    assert_is_integral_of(x*sec(x)**2, x*tan(x) + log(cos(x)))
    assert_is_integral_of(x**3*Derivative(f(x), (x, 4)),
                          x**3*Derivative(f(x), (x, 3)) - 3*x**2*Derivative(f(x), (x, 2)) +
                          6*x*Derivative(f(x), x) - 6*f(x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_densetools.py ===
"""Tests for dense recursive polynomials' tools. """

from sympy.polys.densebasic import (
    dup_normal, dmp_normal,
    dup_from_raw_dict,
    dmp_convert, dmp_swap,
)

from sympy.polys.densearith import dmp_mul_ground

from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,
    dup_integrate, dmp_integrate, dmp_integrate_in,
    dup_diff, dmp_diff, dmp_diff_in,
    dup_eval, dmp_eval, dmp_eval_in,
    dmp_eval_tail, dmp_diff_eval_in,
    dup_trunc, dmp_trunc, dmp_ground_trunc,
    dup_monic, dmp_ground_monic,
    dup_content, dmp_ground_content,
    dup_primitive, dmp_ground_primitive,
    dup_extract, dmp_ground_extract,
    dup_real_imag,
    dup_mirror, dup_scale, dup_shift, dmp_shift,
    dup_transform,
    dup_compose, dmp_compose,
    dup_decompose,
    dmp_lift,
    dup_sign_variations,
    dup_revert, dmp_revert,
)
from sympy.polys.polyclasses import ANP

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    ExactQuotientFailed,
    NotReversible,
    DomainError,
)

from sympy.polys.specialpolys import f_polys

from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, EX, RR
from sympy.polys.rings import ring

from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.trigonometric import sin

from sympy.abc import x
from sympy.testing.pytest import raises

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]

def test_dup_integrate():
    assert dup_integrate([], 1, QQ) == []
    assert dup_integrate([], 2, QQ) == []

    assert dup_integrate([QQ(1)], 1, QQ) == [QQ(1), QQ(0)]
    assert dup_integrate([QQ(1)], 2, QQ) == [QQ(1, 2), QQ(0), QQ(0)]

    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 0, QQ) == \
        [QQ(1), QQ(2), QQ(3)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 1, QQ) == \
        [QQ(1, 3), QQ(1), QQ(3), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 2, QQ) == \
        [QQ(1, 12), QQ(1, 3), QQ(3, 2), QQ(0), QQ(0)]
    assert dup_integrate([QQ(1), QQ(2), QQ(3)], 3, QQ) == \
        [QQ(1, 60), QQ(1, 12), QQ(1, 2), QQ(0), QQ(0), QQ(0)]

    assert dup_integrate(dup_from_raw_dict({29: QQ(17)}, QQ), 3, QQ) == \
        dup_from_raw_dict({32: QQ(17, 29760)}, QQ)

    assert dup_integrate(dup_from_raw_dict({29: QQ(17), 5: QQ(1, 2)}, QQ), 3, QQ) == \
        dup_from_raw_dict({32: QQ(17, 29760), 8: QQ(1, 672)}, QQ)


def test_dmp_integrate():
    assert dmp_integrate([QQ(1)], 2, 0, QQ) == [QQ(1, 2), QQ(0), QQ(0)]

    assert dmp_integrate([[[]]], 1, 2, QQ) == [[[]]]
    assert dmp_integrate([[[]]], 2, 2, QQ) == [[[]]]

    assert dmp_integrate([[[QQ(1)]]], 1, 2, QQ) == [[[QQ(1)]], [[]]]
    assert dmp_integrate([[[QQ(1)]]], 2, 2, QQ) == [[[QQ(1, 2)]], [[]], [[]]]

    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 0, 1, QQ) == \
        [[QQ(1)], [QQ(2)], [QQ(3)]]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 1, 1, QQ) == \
        [[QQ(1, 3)], [QQ(1)], [QQ(3)], []]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 2, 1, QQ) == \
        [[QQ(1, 12)], [QQ(1, 3)], [QQ(3, 2)], [], []]
    assert dmp_integrate([[QQ(1)], [QQ(2)], [QQ(3)]], 3, 1, QQ) == \
        [[QQ(1, 60)], [QQ(1, 12)], [QQ(1, 2)], [], [], []]


def test_dmp_integrate_in():
    f = dmp_convert(f_6, 3, ZZ, QQ)

    assert dmp_integrate_in(f, 2, 1, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 2, 3, QQ), 0, 1, 3, QQ)
    assert dmp_integrate_in(f, 3, 1, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 1, 3, QQ), 3, 3, QQ), 0, 1, 3, QQ)
    assert dmp_integrate_in(f, 2, 2, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 2, 3, QQ), 0, 2, 3, QQ)
    assert dmp_integrate_in(f, 3, 2, 3, QQ) == \
        dmp_swap(
            dmp_integrate(dmp_swap(f, 0, 2, 3, QQ), 3, 3, QQ), 0, 2, 3, QQ)

    raises(IndexError, lambda: dmp_integrate_in(f, 1, -1, 3, QQ))
    raises(IndexError, lambda: dmp_integrate_in(f, 1, 4, 3, QQ))


def test_dup_diff():
    assert dup_diff([], 1, ZZ) == []
    assert dup_diff([7], 1, ZZ) == []
    assert dup_diff([2, 7], 1, ZZ) == [2]
    assert dup_diff([1, 2, 1], 1, ZZ) == [2, 2]
    assert dup_diff([1, 2, 3, 4], 1, ZZ) == [3, 4, 3]
    assert dup_diff([1, -1, 0, 0, 2], 1, ZZ) == [4, -3, 0, 0]

    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], ZZ)

    assert dup_diff(f, 0, ZZ) == f
    assert dup_diff(f, 1, ZZ) == [170, 306, 448, -2415, 138, 380, 0, 0, 24, 3]
    assert dup_diff(f, 2, ZZ) == dup_diff(dup_diff(f, 1, ZZ), 1, ZZ)
    assert dup_diff(
        f, 3, ZZ) == dup_diff(dup_diff(dup_diff(f, 1, ZZ), 1, ZZ), 1, ZZ)

    K = FF(3)
    f = dup_normal([17, 34, 56, -345, 23, 76, 0, 0, 12, 3, 7], K)

    assert dup_diff(f, 1, K) == dup_normal([2, 0, 1, 0, 0, 2, 0, 0, 0, 0], K)
    assert dup_diff(f, 2, K) == dup_normal([1, 0, 0, 2, 0, 0, 0], K)
    assert dup_diff(f, 3, K) == dup_normal([], K)

    assert dup_diff(f, 0, K) == f
    assert dup_diff(f, 2, K) == dup_diff(dup_diff(f, 1, K), 1, K)
    assert dup_diff(
        f, 3, K) == dup_diff(dup_diff(dup_diff(f, 1, K), 1, K), 1, K)


def test_dmp_diff():
    assert dmp_diff([], 1, 0, ZZ) == []
    assert dmp_diff([[]], 1, 1, ZZ) == [[]]
    assert dmp_diff([[[]]], 1, 2, ZZ) == [[[]]]

    assert dmp_diff([[[1], [2]]], 1, 2, ZZ) == [[[]]]

    assert dmp_diff([[[1]], [[]]], 1, 2, ZZ) == [[[1]]]
    assert dmp_diff([[[3]], [[1]], [[]]], 1, 2, ZZ) == [[[6]], [[1]]]

    assert dmp_diff([1, -1, 0, 0, 2], 1, 0, ZZ) == \
        dup_diff([1, -1, 0, 0, 2], 1, ZZ)

    assert dmp_diff(f_6, 0, 3, ZZ) == f_6
    assert dmp_diff(f_6, 1, 3, ZZ) == [[[[8460]], [[]]],
                       [[[135, 0, 0], [], [], [-135, 0, 0]]],
                       [[[]]],
                       [[[-423]], [[-47]], [[]], [[141], [], [94, 0], []], [[]]]]
    assert dmp_diff(
        f_6, 2, 3, ZZ) == dmp_diff(dmp_diff(f_6, 1, 3, ZZ), 1, 3, ZZ)
    assert dmp_diff(f_6, 3, 3, ZZ) == dmp_diff(
        dmp_diff(dmp_diff(f_6, 1, 3, ZZ), 1, 3, ZZ), 1, 3, ZZ)

    K = FF(23)
    F_6 = dmp_normal(f_6, 3, K)

    assert dmp_diff(F_6, 0, 3, K) == F_6
    assert dmp_diff(F_6, 1, 3, K) == dmp_diff(F_6, 1, 3, K)
    assert dmp_diff(F_6, 2, 3, K) == dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K)
    assert dmp_diff(F_6, 3, 3, K) == dmp_diff(
        dmp_diff(dmp_diff(F_6, 1, 3, K), 1, 3, K), 1, 3, K)


def test_dmp_diff_in():
    assert dmp_diff_in(f_6, 2, 1, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 1, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 3, 3, ZZ), 0, 1, 3, ZZ)
    assert dmp_diff_in(f_6, 2, 2, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 2, 3, ZZ), 0, 2, 3, ZZ)
    assert dmp_diff_in(f_6, 3, 2, 3, ZZ) == \
        dmp_swap(dmp_diff(dmp_swap(f_6, 0, 2, 3, ZZ), 3, 3, ZZ), 0, 2, 3, ZZ)

    raises(IndexError, lambda: dmp_diff_in(f_6, 1, -1, 3, ZZ))
    raises(IndexError, lambda: dmp_diff_in(f_6, 1, 4, 3, ZZ))

def test_dup_eval():
    assert dup_eval([], 7, ZZ) == 0
    assert dup_eval([1, 2], 0, ZZ) == 2
    assert dup_eval([1, 2, 3], 7, ZZ) == 66


def test_dmp_eval():
    assert dmp_eval([], 3, 0, ZZ) == 0

    assert dmp_eval([[]], 3, 1, ZZ) == []
    assert dmp_eval([[[]]], 3, 2, ZZ) == [[]]

    assert dmp_eval([[1, 2]], 0, 1, ZZ) == [1, 2]

    assert dmp_eval([[[1]]], 3, 2, ZZ) == [[1]]
    assert dmp_eval([[[1, 2]]], 3, 2, ZZ) == [[1, 2]]

    assert dmp_eval([[3, 2], [1, 2]], 3, 1, ZZ) == [10, 8]
    assert dmp_eval([[[3, 2]], [[1, 2]]], 3, 2, ZZ) == [[10, 8]]


def test_dmp_eval_in():
    assert dmp_eval_in(
        f_6, -2, 1, 3, ZZ) == dmp_eval(dmp_swap(f_6, 0, 1, 3, ZZ), -2, 3, ZZ)
    assert dmp_eval_in(
        f_6, 7, 1, 3, ZZ) == dmp_eval(dmp_swap(f_6, 0, 1, 3, ZZ), 7, 3, ZZ)
    assert dmp_eval_in(f_6, -2, 2, 3, ZZ) == dmp_swap(
        dmp_eval(dmp_swap(f_6, 0, 2, 3, ZZ), -2, 3, ZZ), 0, 1, 2, ZZ)
    assert dmp_eval_in(f_6, 7, 2, 3, ZZ) == dmp_swap(
        dmp_eval(dmp_swap(f_6, 0, 2, 3, ZZ), 7, 3, ZZ), 0, 1, 2, ZZ)

    f = [[[int(45)]], [[]], [[]], [[int(-9)], [-1], [], [int(3), int(0), int(10), int(0)]]]

    assert dmp_eval_in(f, -2, 2, 2, ZZ) == \
        [[45], [], [], [-9, -1, 0, -44]]

    raises(IndexError, lambda: dmp_eval_in(f_6, ZZ(1), -1, 3, ZZ))
    raises(IndexError, lambda: dmp_eval_in(f_6, ZZ(1), 4, 3, ZZ))


def test_dmp_eval_tail():
    assert dmp_eval_tail([[]], [1], 1, ZZ) == []
    assert dmp_eval_tail([[[]]], [1], 2, ZZ) == [[]]
    assert dmp_eval_tail([[[]]], [1, 2], 2, ZZ) == []

    assert dmp_eval_tail(f_0, [], 2, ZZ) == f_0

    assert dmp_eval_tail(f_0, [1, -17, 8], 2, ZZ) == 84496
    assert dmp_eval_tail(f_0, [-17, 8], 2, ZZ) == [-1409, 3, 85902]
    assert dmp_eval_tail(f_0, [8], 2, ZZ) == [[83, 2], [3], [302, 81, 1]]

    assert dmp_eval_tail(f_1, [-17, 8], 2, ZZ) == [-136, 15699, 9166, -27144]

    assert dmp_eval_tail(
        f_2, [-12, 3], 2, ZZ) == [-1377, 0, -702, -1224, 0, -624]
    assert dmp_eval_tail(
        f_3, [-12, 3], 2, ZZ) == [144, 82, -5181, -28872, -14868, -540]

    assert dmp_eval_tail(
        f_4, [25, -1], 2, ZZ) == [152587890625, 9765625, -59605407714843750,
        -3839159765625, -1562475, 9536712644531250, 610349546750, -4, 24414375000, 1562520]
    assert dmp_eval_tail(f_5, [25, -1], 2, ZZ) == [-1, -78, -2028, -17576]

    assert dmp_eval_tail(f_6, [0, 2, 4], 3, ZZ) == [5040, 0, 0, 4480]


def test_dmp_diff_eval_in():
    assert dmp_diff_eval_in(f_6, 2, 7, 1, 3, ZZ) == \
        dmp_eval(dmp_diff(dmp_swap(f_6, 0, 1, 3, ZZ), 2, 3, ZZ), 7, 3, ZZ)

    assert dmp_diff_eval_in(f_6, 2, 7, 0, 3, ZZ) == \
        dmp_eval(dmp_diff(f_6, 2, 3, ZZ), 7, 3, ZZ)

    raises(IndexError, lambda: dmp_diff_eval_in(f_6, 1, ZZ(1), 4, 3, ZZ))


def test_dup_revert():
    f = [-QQ(1, 720), QQ(0), QQ(1, 24), QQ(0), -QQ(1, 2), QQ(0), QQ(1)]
    g = [QQ(61, 720), QQ(0), QQ(5, 24), QQ(0), QQ(1, 2), QQ(0), QQ(1)]

    assert dup_revert(f, 8, QQ) == g

    raises(NotReversible, lambda: dup_revert([QQ(1), QQ(0)], 3, QQ))


def test_dmp_revert():
    f = [-QQ(1, 720), QQ(0), QQ(1, 24), QQ(0), -QQ(1, 2), QQ(0), QQ(1)]
    g = [QQ(61, 720), QQ(0), QQ(5, 24), QQ(0), QQ(1, 2), QQ(0), QQ(1)]

    assert dmp_revert(f, 8, 0, QQ) == g

    raises(MultivariatePolynomialError, lambda: dmp_revert([[1]], 2, 1, QQ))


def test_dup_trunc():
    assert dup_trunc([1, 2, 3, 4, 5, 6], ZZ(3), ZZ) == [1, -1, 0, 1, -1, 0]
    assert dup_trunc([6, 5, 4, 3, 2, 1], ZZ(3), ZZ) == [-1, 1, 0, -1, 1]

    R = ZZ_I
    assert dup_trunc([R(3), R(4), R(5)], R(3), R) == [R(1), R(-1)]

    K = FF(5)
    assert dup_trunc([K(3), K(4), K(5)], K(3), K) == [K(1), K(0)]


def test_dmp_trunc():
    assert dmp_trunc([[]], [1, 2], 2, ZZ) == [[]]
    assert dmp_trunc([[1, 2], [1, 4, 1], [1]], [1, 2], 1, ZZ) == [[-3], [1]]


def test_dmp_ground_trunc():
    assert dmp_ground_trunc(f_0, ZZ(3), 2, ZZ) == \
        dmp_normal(
            [[[1, -1, 0], [-1]], [[]], [[1, -1, 0], [1, -1, 1], [1]]], 2, ZZ)


def test_dup_monic():
    assert dup_monic([3, 6, 9], ZZ) == [1, 2, 3]

    raises(ExactQuotientFailed, lambda: dup_monic([3, 4, 5], ZZ))

    assert dup_monic([], QQ) == []
    assert dup_monic([QQ(1)], QQ) == [QQ(1)]
    assert dup_monic([QQ(7), QQ(1), QQ(21)], QQ) == [QQ(1), QQ(1, 7), QQ(3)]


def test_dmp_ground_monic():
    assert dmp_ground_monic([3, 6, 9], 0, ZZ) == [1, 2, 3]

    assert dmp_ground_monic([[3], [6], [9]], 1, ZZ) == [[1], [2], [3]]

    raises(
        ExactQuotientFailed, lambda: dmp_ground_monic([[3], [4], [5]], 1, ZZ))

    assert dmp_ground_monic([[]], 1, QQ) == [[]]
    assert dmp_ground_monic([[QQ(1)]], 1, QQ) == [[QQ(1)]]
    assert dmp_ground_monic(
        [[QQ(7)], [QQ(1)], [QQ(21)]], 1, QQ) == [[QQ(1)], [QQ(1, 7)], [QQ(3)]]


def test_dup_content():
    assert dup_content([], ZZ) == ZZ(0)
    assert dup_content([1], ZZ) == ZZ(1)
    assert dup_content([-1], ZZ) == ZZ(1)
    assert dup_content([1, 1], ZZ) == ZZ(1)
    assert dup_content([2, 2], ZZ) == ZZ(2)
    assert dup_content([1, 2, 1], ZZ) == ZZ(1)
    assert dup_content([2, 4, 2], ZZ) == ZZ(2)

    assert dup_content([QQ(2, 3), QQ(4, 9)], QQ) == QQ(2, 9)
    assert dup_content([QQ(2, 3), QQ(4, 5)], QQ) == QQ(2, 15)


def test_dmp_ground_content():
    assert dmp_ground_content([[]], 1, ZZ) == ZZ(0)
    assert dmp_ground_content([[]], 1, QQ) == QQ(0)
    assert dmp_ground_content([[1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[-1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[1], [1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[2], [2]], 1, ZZ) == ZZ(2)
    assert dmp_ground_content([[1], [2], [1]], 1, ZZ) == ZZ(1)
    assert dmp_ground_content([[2], [4], [2]], 1, ZZ) == ZZ(2)

    assert dmp_ground_content([[QQ(2, 3)], [QQ(4, 9)]], 1, QQ) == QQ(2, 9)
    assert dmp_ground_content([[QQ(2, 3)], [QQ(4, 5)]], 1, QQ) == QQ(2, 15)

    assert dmp_ground_content(f_0, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_0, ZZ(2), 2, ZZ), 2, ZZ) == ZZ(2)

    assert dmp_ground_content(f_1, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_1, ZZ(3), 2, ZZ), 2, ZZ) == ZZ(3)

    assert dmp_ground_content(f_2, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_2, ZZ(4), 2, ZZ), 2, ZZ) == ZZ(4)

    assert dmp_ground_content(f_3, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_3, ZZ(5), 2, ZZ), 2, ZZ) == ZZ(5)

    assert dmp_ground_content(f_4, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_4, ZZ(6), 2, ZZ), 2, ZZ) == ZZ(6)

    assert dmp_ground_content(f_5, 2, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_5, ZZ(7), 2, ZZ), 2, ZZ) == ZZ(7)

    assert dmp_ground_content(f_6, 3, ZZ) == ZZ(1)
    assert dmp_ground_content(
        dmp_mul_ground(f_6, ZZ(8), 3, ZZ), 3, ZZ) == ZZ(8)


def test_dup_primitive():
    assert dup_primitive([], ZZ) == (ZZ(0), [])
    assert dup_primitive([ZZ(1)], ZZ) == (ZZ(1), [ZZ(1)])
    assert dup_primitive([ZZ(1), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(1)])
    assert dup_primitive([ZZ(2), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(1)])
    assert dup_primitive(
        [ZZ(1), ZZ(2), ZZ(1)], ZZ) == (ZZ(1), [ZZ(1), ZZ(2), ZZ(1)])
    assert dup_primitive(
        [ZZ(2), ZZ(4), ZZ(2)], ZZ) == (ZZ(2), [ZZ(1), ZZ(2), ZZ(1)])

    assert dup_primitive([], QQ) == (QQ(0), [])
    assert dup_primitive([QQ(1)], QQ) == (QQ(1), [QQ(1)])
    assert dup_primitive([QQ(1), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(1)])
    assert dup_primitive([QQ(2), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(1)])
    assert dup_primitive(
        [QQ(1), QQ(2), QQ(1)], QQ) == (QQ(1), [QQ(1), QQ(2), QQ(1)])
    assert dup_primitive(
        [QQ(2), QQ(4), QQ(2)], QQ) == (QQ(2), [QQ(1), QQ(2), QQ(1)])

    assert dup_primitive(
        [QQ(2, 3), QQ(4, 9)], QQ) == (QQ(2, 9), [QQ(3), QQ(2)])
    assert dup_primitive(
        [QQ(2, 3), QQ(4, 5)], QQ) == (QQ(2, 15), [QQ(5), QQ(6)])


def test_dmp_ground_primitive():
    assert dmp_ground_primitive([ZZ(1)], 0, ZZ) == (ZZ(1), [ZZ(1)])

    assert dmp_ground_primitive([[]], 1, ZZ) == (ZZ(0), [[]])

    assert dmp_ground_primitive(f_0, 2, ZZ) == (ZZ(1), f_0)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_0, ZZ(2), 2, ZZ), 2, ZZ) == (ZZ(2), f_0)

    assert dmp_ground_primitive(f_1, 2, ZZ) == (ZZ(1), f_1)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_1, ZZ(3), 2, ZZ), 2, ZZ) == (ZZ(3), f_1)

    assert dmp_ground_primitive(f_2, 2, ZZ) == (ZZ(1), f_2)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_2, ZZ(4), 2, ZZ), 2, ZZ) == (ZZ(4), f_2)

    assert dmp_ground_primitive(f_3, 2, ZZ) == (ZZ(1), f_3)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_3, ZZ(5), 2, ZZ), 2, ZZ) == (ZZ(5), f_3)

    assert dmp_ground_primitive(f_4, 2, ZZ) == (ZZ(1), f_4)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_4, ZZ(6), 2, ZZ), 2, ZZ) == (ZZ(6), f_4)

    assert dmp_ground_primitive(f_5, 2, ZZ) == (ZZ(1), f_5)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_5, ZZ(7), 2, ZZ), 2, ZZ) == (ZZ(7), f_5)

    assert dmp_ground_primitive(f_6, 3, ZZ) == (ZZ(1), f_6)
    assert dmp_ground_primitive(
        dmp_mul_ground(f_6, ZZ(8), 3, ZZ), 3, ZZ) == (ZZ(8), f_6)

    assert dmp_ground_primitive([[ZZ(2)]], 1, ZZ) == (ZZ(2), [[ZZ(1)]])
    assert dmp_ground_primitive([[QQ(2)]], 1, QQ) == (QQ(2), [[QQ(1)]])

    assert dmp_ground_primitive(
        [[QQ(2, 3)], [QQ(4, 9)]], 1, QQ) == (QQ(2, 9), [[QQ(3)], [QQ(2)]])
    assert dmp_ground_primitive(
        [[QQ(2, 3)], [QQ(4, 5)]], 1, QQ) == (QQ(2, 15), [[QQ(5)], [QQ(6)]])


def test_dup_extract():
    f = dup_normal([2930944, 0, 2198208, 0, 549552, 0, 45796], ZZ)
    g = dup_normal([17585664, 0, 8792832, 0, 1099104, 0], ZZ)

    F = dup_normal([64, 0, 48, 0, 12, 0, 1], ZZ)
    G = dup_normal([384, 0, 192, 0, 24, 0], ZZ)

    assert dup_extract(f, g, ZZ) == (45796, F, G)


def test_dmp_ground_extract():
    f = dmp_normal(
        [[2930944], [], [2198208], [], [549552], [], [45796]], 1, ZZ)
    g = dmp_normal([[17585664], [], [8792832], [], [1099104], []], 1, ZZ)

    F = dmp_normal([[64], [], [48], [], [12], [], [1]], 1, ZZ)
    G = dmp_normal([[384], [], [192], [], [24], []], 1, ZZ)

    assert dmp_ground_extract(f, g, 1, ZZ) == (45796, F, G)


def test_dup_real_imag():
    assert dup_real_imag([], ZZ) == ([[]], [[]])
    assert dup_real_imag([1], ZZ) == ([[1]], [[]])

    assert dup_real_imag([1, 1], ZZ) == ([[1], [1]], [[1, 0]])
    assert dup_real_imag([1, 2], ZZ) == ([[1], [2]], [[1, 0]])

    assert dup_real_imag(
        [1, 2, 3], ZZ) == ([[1], [2], [-1, 0, 3]], [[2, 0], [2, 0]])

    assert dup_real_imag([ZZ(1), ZZ(0), ZZ(1), ZZ(3)], ZZ) == (
        [[ZZ(1)], [], [ZZ(-3), ZZ(0), ZZ(1)], [ZZ(3)]],
        [[ZZ(3), ZZ(0)], [], [ZZ(-1), ZZ(0), ZZ(1), ZZ(0)]]
    )

    raises(DomainError, lambda: dup_real_imag([EX(1), EX(2)], EX))



def test_dup_mirror():
    assert dup_mirror([], ZZ) == []
    assert dup_mirror([1], ZZ) == [1]

    assert dup_mirror([1, 2, 3, 4, 5], ZZ) == [1, -2, 3, -4, 5]
    assert dup_mirror([1, 2, 3, 4, 5, 6], ZZ) == [-1, 2, -3, 4, -5, 6]


def test_dup_scale():
    assert dup_scale([], -1, ZZ) == []
    assert dup_scale([1], -1, ZZ) == [1]

    assert dup_scale([1, 2, 3, 4, 5], -1, ZZ) == [1, -2, 3, -4, 5]
    assert dup_scale([1, 2, 3, 4, 5], -7, ZZ) == [2401, -686, 147, -28, 5]


def test_dup_shift():
    assert dup_shift([], 1, ZZ) == []
    assert dup_shift([1], 1, ZZ) == [1]

    assert dup_shift([1, 2, 3, 4, 5], 1, ZZ) == [1, 6, 15, 20, 15]
    assert dup_shift([1, 2, 3, 4, 5], 7, ZZ) == [1, 30, 339, 1712, 3267]


def test_dmp_shift():
    assert dmp_shift([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == [ZZ(1), ZZ(3)]

    assert dmp_shift([[]], [ZZ(1), ZZ(2)], 1, ZZ) == [[]]

    xy = [[ZZ(1), ZZ(0)], []]               # x*y
    x1y2 = [[ZZ(1), ZZ(2)], [ZZ(1), ZZ(2)]] # (x+1)*(y+2)
    assert dmp_shift(xy, [ZZ(1), ZZ(2)], 1, ZZ) == x1y2


def test_dup_transform():
    assert dup_transform([], [], [1, 1], ZZ) == []
    assert dup_transform([], [1], [1, 1], ZZ) == []
    assert dup_transform([], [1, 2], [1, 1], ZZ) == []

    assert dup_transform([6, -5, 4, -3, 17], [1, -3, 4], [2, -3], ZZ) == \
        [6, -82, 541, -2205, 6277, -12723, 17191, -13603, 4773]


def test_dup_compose():
    assert dup_compose([], [], ZZ) == []
    assert dup_compose([], [1], ZZ) == []
    assert dup_compose([], [1, 2], ZZ) == []

    assert dup_compose([1], [], ZZ) == [1]

    assert dup_compose([1, 2, 0], [], ZZ) == []
    assert dup_compose([1, 2, 1], [], ZZ) == [1]

    assert dup_compose([1, 2, 1], [1], ZZ) == [4]
    assert dup_compose([1, 2, 1], [7], ZZ) == [64]

    assert dup_compose([1, 2, 1], [1, -1], ZZ) == [1, 0, 0]
    assert dup_compose([1, 2, 1], [1, 1], ZZ) == [1, 4, 4]
    assert dup_compose([1, 2, 1], [1, 2, 1], ZZ) == [1, 4, 8, 8, 4]


def test_dmp_compose():
    assert dmp_compose([1, 2, 1], [1, 2, 1], 0, ZZ) == [1, 4, 8, 8, 4]

    assert dmp_compose([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_compose([[[]]], [[[1]]], 2, ZZ) == [[[]]]
    assert dmp_compose([[[]]], [[[1]], [[2]]], 2, ZZ) == [[[]]]

    assert dmp_compose([[[1]]], [], 2, ZZ) == [[[1]]]

    assert dmp_compose([[1], [2], [ ]], [[]], 1, ZZ) == [[]]
    assert dmp_compose([[1], [2], [1]], [[]], 1, ZZ) == [[1]]

    assert dmp_compose([[1], [2], [1]], [[1]], 1, ZZ) == [[4]]
    assert dmp_compose([[1], [2], [1]], [[7]], 1, ZZ) == [[64]]

    assert dmp_compose([[1], [2], [1]], [[1], [-1]], 1, ZZ) == [[1], [ ], [ ]]
    assert dmp_compose([[1], [2], [1]], [[1], [ 1]], 1, ZZ) == [[1], [4], [4]]

    assert dmp_compose(
        [[1], [2], [1]], [[1], [2], [1]], 1, ZZ) == [[1], [4], [8], [8], [4]]


def test_dup_decompose():
    assert dup_decompose([1], ZZ) == [[1]]

    assert dup_decompose([1, 0], ZZ) == [[1, 0]]
    assert dup_decompose([1, 0, 0, 0], ZZ) == [[1, 0, 0, 0]]

    assert dup_decompose([1, 0, 0, 0, 0], ZZ) == [[1, 0, 0], [1, 0, 0]]
    assert dup_decompose(
        [1, 0, 0, 0, 0, 0, 0], ZZ) == [[1, 0, 0, 0], [1, 0, 0]]

    assert dup_decompose([7, 0, 0, 0, 1], ZZ) == [[7, 0, 1], [1, 0, 0]]
    assert dup_decompose([4, 0, 3, 0, 2], ZZ) == [[4, 3, 2], [1, 0, 0]]

    f = [1, 0, 20, 0, 150, 0, 500, 0, 625, -2, 0, -10, 9]

    assert dup_decompose(f, ZZ) == [[1, 0, 0, -2, 9], [1, 0, 5, 0]]

    f = [2, 0, 40, 0, 300, 0, 1000, 0, 1250, -4, 0, -20, 18]

    assert dup_decompose(f, ZZ) == [[2, 0, 0, -4, 18], [1, 0, 5, 0]]

    f = [1, 0, 20, -8, 150, -120, 524, -600, 865, -1034, 600, -170, 29]

    assert dup_decompose(f, ZZ) == [[1, -8, 24, -34, 29], [1, 0, 5, 0]]

    R, t = ring("t", ZZ)
    f = [6*t**2 - 42,
         48*t**2 + 96,
         144*t**2 + 648*t + 288,
         624*t**2 + 864*t + 384,
         108*t**3 + 312*t**2 + 432*t + 192]

    assert dup_decompose(f, R.to_domain()) == [f]


def test_dmp_lift():
    q = [QQ(1, 1), QQ(0, 1), QQ(1, 1)]

    f_a = [ANP([QQ(1, 1)], q, QQ), ANP([], q, QQ), ANP([], q, QQ),
         ANP([QQ(1, 1), QQ(0, 1)], q, QQ), ANP([QQ(17, 1), QQ(0, 1)], q, QQ)]

    f_lift = QQ.map([1, 0, 0, 0, 0, 0, 1, 34, 289])

    assert dmp_lift(f_a, 0, QQ.algebraic_field(I)) == f_lift

    f_g = [QQ_I(1), QQ_I(0), QQ_I(0), QQ_I(0, 1), QQ_I(0, 17)]

    assert dmp_lift(f_g, 0, QQ_I) == f_lift

    raises(DomainError, lambda: dmp_lift([EX(1), EX(2)], 0, EX))


def test_dup_sign_variations():
    assert dup_sign_variations([], ZZ) == 0
    assert dup_sign_variations([1, 0], ZZ) == 0
    assert dup_sign_variations([1, 0, 2], ZZ) == 0
    assert dup_sign_variations([1, 0, 3, 0], ZZ) == 0
    assert dup_sign_variations([1, 0, 4, 0, 5], ZZ) == 0

    assert dup_sign_variations([-1, 0, 2], ZZ) == 1
    assert dup_sign_variations([-1, 0, 3, 0], ZZ) == 1
    assert dup_sign_variations([-1, 0, 4, 0, 5], ZZ) == 1

    assert dup_sign_variations([-1, -4, -5], ZZ) == 0
    assert dup_sign_variations([ 1, -4, -5], ZZ) == 1
    assert dup_sign_variations([ 1, 4, -5], ZZ) == 1
    assert dup_sign_variations([ 1, -4, 5], ZZ) == 2
    assert dup_sign_variations([-1, 4, -5], ZZ) == 2
    assert dup_sign_variations([-1, 4, 5], ZZ) == 1
    assert dup_sign_variations([-1, -4, 5], ZZ) == 1
    assert dup_sign_variations([ 1, 4, 5], ZZ) == 0

    assert dup_sign_variations([-1, 0, -4, 0, -5], ZZ) == 0
    assert dup_sign_variations([ 1, 0, -4, 0, -5], ZZ) == 1
    assert dup_sign_variations([ 1, 0, 4, 0, -5], ZZ) == 1
    assert dup_sign_variations([ 1, 0, -4, 0, 5], ZZ) == 2
    assert dup_sign_variations([-1, 0, 4, 0, -5], ZZ) == 2
    assert dup_sign_variations([-1, 0, 4, 0, 5], ZZ) == 1
    assert dup_sign_variations([-1, 0, -4, 0, 5], ZZ) == 1
    assert dup_sign_variations([ 1, 0, 4, 0, 5], ZZ) == 0


def test_dup_clear_denoms():
    assert dup_clear_denoms([], QQ, ZZ) == (ZZ(1), [])

    assert dup_clear_denoms([QQ(1)], QQ, ZZ) == (ZZ(1), [QQ(1)])
    assert dup_clear_denoms([QQ(7)], QQ, ZZ) == (ZZ(1), [QQ(7)])

    assert dup_clear_denoms([QQ(7, 3)], QQ) == (ZZ(3), [QQ(7)])
    assert dup_clear_denoms([QQ(7, 3)], QQ, ZZ) == (ZZ(3), [QQ(7)])

    assert dup_clear_denoms(
        [QQ(3), QQ(1), QQ(0)], QQ, ZZ) == (ZZ(1), [QQ(3), QQ(1), QQ(0)])
    assert dup_clear_denoms(
        [QQ(1), QQ(1, 2), QQ(0)], QQ, ZZ) == (ZZ(2), [QQ(2), QQ(1), QQ(0)])

    assert dup_clear_denoms([QQ(3), QQ(
        1), QQ(0)], QQ, ZZ, convert=True) == (ZZ(1), [ZZ(3), ZZ(1), ZZ(0)])
    assert dup_clear_denoms([QQ(1), QQ(
        1, 2), QQ(0)], QQ, ZZ, convert=True) == (ZZ(2), [ZZ(2), ZZ(1), ZZ(0)])

    assert dup_clear_denoms(
        [EX(S(3)/2), EX(S(9)/4)], EX) == (EX(4), [EX(6), EX(9)])

    assert dup_clear_denoms([EX(7)], EX) == (EX(1), [EX(7)])
    assert dup_clear_denoms([EX(sin(x)/x), EX(0)], EX) == (EX(x), [EX(sin(x)), EX(0)])

    F = RR.frac_field(x)
    result = dup_clear_denoms([F(8.48717/(8.0089*x + 2.83)), F(0.0)], F)
    assert str(result) == "(x + 0.353356890459364, [1.05971731448763, 0.0])"

def test_dmp_clear_denoms():
    assert dmp_clear_denoms([[]], 1, QQ, ZZ) == (ZZ(1), [[]])

    assert dmp_clear_denoms([[QQ(1)]], 1, QQ, ZZ) == (ZZ(1), [[QQ(1)]])
    assert dmp_clear_denoms([[QQ(7)]], 1, QQ, ZZ) == (ZZ(1), [[QQ(7)]])

    assert dmp_clear_denoms([[QQ(7, 3)]], 1, QQ) == (ZZ(3), [[QQ(7)]])
    assert dmp_clear_denoms([[QQ(7, 3)]], 1, QQ, ZZ) == (ZZ(3), [[QQ(7)]])

    assert dmp_clear_denoms(
        [[QQ(3)], [QQ(1)], []], 1, QQ, ZZ) == (ZZ(1), [[QQ(3)], [QQ(1)], []])
    assert dmp_clear_denoms([[QQ(
        1)], [QQ(1, 2)], []], 1, QQ, ZZ) == (ZZ(2), [[QQ(2)], [QQ(1)], []])

    assert dmp_clear_denoms([QQ(3), QQ(
        1), QQ(0)], 0, QQ, ZZ, convert=True) == (ZZ(1), [ZZ(3), ZZ(1), ZZ(0)])
    assert dmp_clear_denoms([QQ(1), QQ(1, 2), QQ(
        0)], 0, QQ, ZZ, convert=True) == (ZZ(2), [ZZ(2), ZZ(1), ZZ(0)])

    assert dmp_clear_denoms([[QQ(3)], [QQ(
        1)], []], 1, QQ, ZZ, convert=True) == (ZZ(1), [[QQ(3)], [QQ(1)], []])
    assert dmp_clear_denoms([[QQ(1)], [QQ(1, 2)], []], 1, QQ, ZZ,
                            convert=True) == (ZZ(2), [[QQ(2)], [QQ(1)], []])

    assert dmp_clear_denoms(
        [[EX(S(3)/2)], [EX(S(9)/4)]], 1, EX) == (EX(4), [[EX(6)], [EX(9)]])
    assert dmp_clear_denoms([[EX(7)]], 1, EX) == (EX(1), [[EX(7)]])
    assert dmp_clear_denoms([[EX(sin(x)/x), EX(0)]], 1, EX) == (EX(x), [[EX(sin(x)), EX(0)]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_eigen.py ===
from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices import eye, Matrix
from sympy.core.singleton import S
from sympy.testing.pytest import raises, XFAIL
from sympy.matrices.exceptions import NonSquareMatrixError, MatrixError
from sympy.matrices.expressions.fourier import DFT
from sympy.simplify.simplify import simplify
from sympy.matrices.immutable import ImmutableMatrix
from sympy.testing.pytest import slow
from sympy.testing.matrices import allclose


def test_eigen():
    R = Rational
    M = Matrix.eye(3)
    assert M.eigenvals(multiple=False) == {S.One: 3}
    assert M.eigenvals(multiple=True) == [1, 1, 1]

    assert M.eigenvects() == (
        [(1, 3, [Matrix([1, 0, 0]),
                 Matrix([0, 1, 0]),
                 Matrix([0, 0, 1])])])

    assert M.left_eigenvects() == (
        [(1, 3, [Matrix([[1, 0, 0]]),
                 Matrix([[0, 1, 0]]),
                 Matrix([[0, 0, 1]])])])

    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])

    assert M.eigenvals() == {2*S.One: 1, -S.One: 1, S.Zero: 1}

    assert M.eigenvects() == (
        [
            (-1, 1, [Matrix([-1, 1, 0])]),
            ( 0, 1, [Matrix([0, -1, 1])]),
            ( 2, 1, [Matrix([R(2, 3), R(1, 3), 1])])
        ])

    assert M.left_eigenvects() == (
        [
            (-1, 1, [Matrix([[-2, 1, 1]])]),
            (0, 1, [Matrix([[-1, -1, 1]])]),
            (2, 1, [Matrix([[1, 1, 1]])])
        ])

    a = Symbol('a')
    M = Matrix([[a, 0],
                [0, 1]])

    assert M.eigenvals() == {a: 1, S.One: 1}

    M = Matrix([[1, -1],
                [1,  3]])
    assert M.eigenvects() == ([(2, 2, [Matrix(2, 1, [-1, 1])])])
    assert M.left_eigenvects() == ([(2, 2, [Matrix([[1, 1]])])])

    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = R(15, 2)
    b = 3*33**R(1, 2)
    c = R(13, 2)
    d = (R(33, 8) + 3*b/8)
    e = (R(33, 8) - 3*b/8)

    def NS(e, n):
        return str(N(e, n))
    r = [
        (a - b/2, 1, [Matrix([(12 + 24/(c - b/2))/((c - b/2)*e) + 3/(c - b/2),
                              (6 + 12/(c - b/2))/e, 1])]),
        (      0, 1, [Matrix([1, -2, 1])]),
        (a + b/2, 1, [Matrix([(12 + 24/(c + b/2))/((c + b/2)*d) + 3/(c + b/2),
                              (6 + 12/(c + b/2))/d, 1])]),
    ]
    r1 = [(NS(r[i][0], 2), NS(r[i][1], 2),
        [NS(j, 2) for j in r[i][2][0]]) for i in range(len(r))]
    r = M.eigenvects()
    r2 = [(NS(r[i][0], 2), NS(r[i][1], 2),
        [NS(j, 2) for j in r[i][2][0]]) for i in range(len(r))]
    assert sorted(r1) == sorted(r2)

    eps = Symbol('eps', real=True)

    M = Matrix([[abs(eps), I*eps    ],
                [-I*eps,   abs(eps) ]])

    assert M.eigenvects() == (
        [
            ( 0, 1, [Matrix([[-I*eps/abs(eps)], [1]])]),
            ( 2*abs(eps), 1, [ Matrix([[I*eps/abs(eps)], [1]]) ] ),
        ])

    assert M.left_eigenvects() == (
        [
            (0, 1, [Matrix([[I*eps/Abs(eps), 1]])]),
            (2*Abs(eps), 1, [Matrix([[-I*eps/Abs(eps), 1]])])
        ])

    M = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    M._eigenvects = M.eigenvects(simplify=False)
    assert max(i.q for i in M._eigenvects[0][2][0]) > 1
    M._eigenvects = M.eigenvects(simplify=True)
    assert max(i.q for i in M._eigenvects[0][2][0]) == 1

    M = Matrix([[Rational(1, 4), 1], [1, 1]])
    assert M.eigenvects() == [
        (Rational(5, 8) - sqrt(73)/8, 1, [Matrix([[-sqrt(73)/8 - Rational(3, 8)], [1]])]),
        (Rational(5, 8) + sqrt(73)/8, 1, [Matrix([[Rational(-3, 8) + sqrt(73)/8], [1]])])]

    # issue 10719
    assert Matrix([]).eigenvals() == {}
    assert Matrix([]).eigenvals(multiple=True) == []
    assert Matrix([]).eigenvects() == []

    # issue 15119
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2], [0, 4], [0, 0]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0], [3, 4], [5, 6]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2, 3], [0, 5, 6]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0, 0], [4, 5, 0]]).eigenvals())
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 2, 3], [0, 5, 6]]).eigenvals(
               error_when_incomplete = False))
    raises(NonSquareMatrixError,
           lambda: Matrix([[1, 0, 0], [4, 5, 0]]).eigenvals(
               error_when_incomplete = False))

    m = Matrix([[1, 2], [3, 4]])
    assert isinstance(m.eigenvals(simplify=True, multiple=False), dict)
    assert isinstance(m.eigenvals(simplify=True, multiple=True), list)
    assert isinstance(m.eigenvals(simplify=lambda x: x, multiple=False), dict)
    assert isinstance(m.eigenvals(simplify=lambda x: x, multiple=True), list)


def test_float_eigenvals():
    m = Matrix([[1, .6, .6], [.6, .9, .9], [.9, .6, .6]])
    evals = [
        Rational(5, 4) - sqrt(385)/20,
        sqrt(385)/20 + Rational(5, 4),
        S.Zero]

    n_evals = m.eigenvals(rational=True, multiple=True)
    n_evals = sorted(n_evals)
    s_evals = [x.evalf() for x in evals]
    s_evals = sorted(s_evals)

    for x, y in zip(n_evals, s_evals):
        assert abs(x-y) < 10**-9


@XFAIL
def test_eigen_vects():
    m = Matrix(2, 2, [1, 0, 0, I])
    raises(NotImplementedError, lambda: m.is_diagonalizable(True))
    # !!! bug because of eigenvects() or roots(x**2 + (-1 - I)*x + I, x)
    # see issue 5292
    assert not m.is_diagonalizable(True)
    raises(MatrixError, lambda: m.diagonalize(True))
    (P, D) = m.diagonalize(True)

def test_issue_8240():
    # Eigenvalues of large triangular matrices
    x, y = symbols('x y')
    n = 200

    diagonal_variables = [Symbol('x%s' % i) for i in range(n)]
    M = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        M[i][i] = diagonal_variables[i]
    M = Matrix(M)

    eigenvals = M.eigenvals()
    assert len(eigenvals) == n
    for i in range(n):
        assert eigenvals[diagonal_variables[i]] == 1

    eigenvals = M.eigenvals(multiple=True)
    assert set(eigenvals) == set(diagonal_variables)

    # with multiplicity
    M = Matrix([[x, 0, 0], [1, y, 0], [2, 3, x]])
    eigenvals = M.eigenvals()
    assert eigenvals == {x: 2, y: 1}

    eigenvals = M.eigenvals(multiple=True)
    assert len(eigenvals) == 3
    assert eigenvals.count(x) == 2
    assert eigenvals.count(y) == 1


def test_eigenvals():
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    assert M.eigenvals() == {2*S.One: 1, -S.One: 1, S.Zero: 1}

    m = Matrix([
        [3,  0,  0, 0, -3],
        [0, -3, -3, 0,  3],
        [0,  3,  0, 3,  0],
        [0,  0,  3, 0,  3],
        [3,  0,  0, 3,  0]])

    # XXX Used dry-run test because arbitrary symbol that appears in
    # CRootOf may not be unique.
    assert m.eigenvals()


def test_eigenvects():
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    vecs = M.eigenvects()
    for val, mult, vec_list in vecs:
        assert len(vec_list) == 1
        assert M*vec_list[0] == val*vec_list[0]


def test_left_eigenvects():
    M = Matrix([[0, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
    vecs = M.left_eigenvects()
    for val, mult, vec_list in vecs:
        assert len(vec_list) == 1
        assert vec_list[0]*M == val*vec_list[0]


@slow
def test_bidiagonalize():
    M = Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    assert M.bidiagonalize() == M
    assert M.bidiagonalize(upper=False) == M
    assert M.bidiagonalize() == M
    assert M.bidiagonal_decomposition() == (M, M, M)
    assert M.bidiagonal_decomposition(upper=False) == (M, M, M)
    assert M.bidiagonalize() == M

    import random
    #Real Tests
    for real_test in range(2):
        test_values = []
        row = 2
        col = 2
        for _ in range(row * col):
            value = random.randint(-1000000000, 1000000000)
            test_values = test_values + [value]
        # L     -> Lower Bidiagonalization
        # M     -> Mutable Matrix
        # N     -> Immutable Matrix
        # 0     -> Bidiagonalized form
        # 1,2,3 -> Bidiagonal_decomposition matrices
        # 4     -> Product of 1 2 3
        M = Matrix(row, col, test_values)
        N = ImmutableMatrix(M)

        N1, N2, N3 = N.bidiagonal_decomposition()
        M1, M2, M3 = M.bidiagonal_decomposition()
        M0 = M.bidiagonalize()
        N0 = N.bidiagonalize()

        N4 = N1 * N2 * N3
        M4 = M1 * M2 * M3

        N2.simplify()
        N4.simplify()
        N0.simplify()

        M0.simplify()
        M2.simplify()
        M4.simplify()

        LM0 = M.bidiagonalize(upper=False)
        LM1, LM2, LM3 = M.bidiagonal_decomposition(upper=False)
        LN0 = N.bidiagonalize(upper=False)
        LN1, LN2, LN3 = N.bidiagonal_decomposition(upper=False)

        LN4 = LN1 * LN2 * LN3
        LM4 = LM1 * LM2 * LM3

        LN2.simplify()
        LN4.simplify()
        LN0.simplify()

        LM0.simplify()
        LM2.simplify()
        LM4.simplify()

        assert M == M4
        assert M2 == M0
        assert N == N4
        assert N2 == N0
        assert M == LM4
        assert LM2 == LM0
        assert N == LN4
        assert LN2 == LN0

    #Complex Tests
    for complex_test in range(2):
        test_values = []
        size = 2
        for _ in range(size * size):
            real = random.randint(-1000000000, 1000000000)
            comp = random.randint(-1000000000, 1000000000)
            value = real + comp * I
            test_values = test_values + [value]
        M = Matrix(size, size, test_values)
        N = ImmutableMatrix(M)
        # L     -> Lower Bidiagonalization
        # M     -> Mutable Matrix
        # N     -> Immutable Matrix
        # 0     -> Bidiagonalized form
        # 1,2,3 -> Bidiagonal_decomposition matrices
        # 4     -> Product of 1 2 3
        N1, N2, N3 = N.bidiagonal_decomposition()
        M1, M2, M3 = M.bidiagonal_decomposition()
        M0 = M.bidiagonalize()
        N0 = N.bidiagonalize()

        N4 = N1 * N2 * N3
        M4 = M1 * M2 * M3

        N2.simplify()
        N4.simplify()
        N0.simplify()

        M0.simplify()
        M2.simplify()
        M4.simplify()

        LM0 = M.bidiagonalize(upper=False)
        LM1, LM2, LM3 = M.bidiagonal_decomposition(upper=False)
        LN0 = N.bidiagonalize(upper=False)
        LN1, LN2, LN3 = N.bidiagonal_decomposition(upper=False)

        LN4 = LN1 * LN2 * LN3
        LM4 = LM1 * LM2 * LM3

        LN2.simplify()
        LN4.simplify()
        LN0.simplify()

        LM0.simplify()
        LM2.simplify()
        LM4.simplify()

        assert M == M4
        assert M2 == M0
        assert N == N4
        assert N2 == N0
        assert M == LM4
        assert LM2 == LM0
        assert N == LN4
        assert LN2 == LN0

    M = Matrix(18, 8, range(1, 145))
    M = M.applyfunc(lambda i: Float(i))
    assert M.bidiagonal_decomposition()[1] == M.bidiagonalize()
    assert M.bidiagonal_decomposition(upper=False)[1] == M.bidiagonalize(upper=False)
    a, b, c = M.bidiagonal_decomposition()
    diff = a * b * c - M
    assert abs(max(diff)) < 10**-12


def test_diagonalize():
    m = Matrix(2, 2, [0, -1, 1, 0])
    raises(MatrixError, lambda: m.diagonalize(reals_only=True))
    P, D = m.diagonalize()
    assert D.is_diagonal()
    assert D == Matrix([
                 [-I, 0],
                 [ 0, I]])

    # make sure we use floats out if floats are passed in
    m = Matrix(2, 2, [0, .5, .5, 0])
    P, D = m.diagonalize()
    assert all(isinstance(e, Float) for e in D.values())
    assert all(isinstance(e, Float) for e in P.values())

    _, D2 = m.diagonalize(reals_only=True)
    assert D == D2

    m = Matrix(
        [[0, 1, 0, 0], [1, 0, 0, 0.002], [0.002, 0, 0, 1], [0, 0, 1, 0]])
    P, D = m.diagonalize()
    assert allclose(P*D, m*P)


def test_is_diagonalizable():
    a, b, c = symbols('a b c')
    m = Matrix(2, 2, [a, c, c, b])
    assert m.is_symmetric()
    assert m.is_diagonalizable()
    assert not Matrix(2, 2, [1, 1, 0, 1]).is_diagonalizable()

    m = Matrix(2, 2, [0, -1, 1, 0])
    assert m.is_diagonalizable()
    assert not m.is_diagonalizable(reals_only=True)


def test_jordan_form():
    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    raises(NonSquareMatrixError, lambda: m.jordan_form())

    # the next two tests test the cases where the old
    # algorithm failed due to the fact that the block structure can
    # *NOT* be determined  from algebraic and geometric multiplicity alone
    # This can be seen most easily when one lets compute the J.c.f. of a matrix that
    # is in J.c.f already.
    m = Matrix(4, 4, [2, 1, 0, 0,
                    0, 2, 1, 0,
                    0, 0, 2, 0,
                    0, 0, 0, 2
    ])
    P, J = m.jordan_form()
    assert m == J

    m = Matrix(4, 4, [2, 1, 0, 0,
                    0, 2, 0, 0,
                    0, 0, 2, 1,
                    0, 0, 0, 2
    ])
    P, J = m.jordan_form()
    assert m == J

    A = Matrix([[ 2,  4,  1,  0],
                [-4,  2,  0,  1],
                [ 0,  0,  2,  4],
                [ 0,  0, -4,  2]])
    P, J = A.jordan_form()
    assert simplify(P*J*P.inv()) == A

    assert Matrix(1, 1, [1]).jordan_form() == (Matrix([1]), Matrix([1]))
    assert Matrix(1, 1, [1]).jordan_form(calc_transform=False) == Matrix([1])

    # If we have eigenvalues in CRootOf form, raise errors
    m = Matrix([[3, 0, 0, 0, -3], [0, -3, -3, 0, 3], [0, 3, 0, 3, 0], [0, 0, 3, 0, 3], [3, 0, 0, 3, 0]])
    raises(MatrixError, lambda: m.jordan_form())

    # make sure that if the input has floats, the output does too
    m = Matrix([
        [                0.6875, 0.125 + 0.1875*sqrt(3)],
        [0.125 + 0.1875*sqrt(3),                 0.3125]])
    P, J = m.jordan_form()
    assert all(isinstance(x, Float) or x == 0 for x in P)
    assert all(isinstance(x, Float) or x == 0 for x in J)


def test_singular_values():
    x = Symbol('x', real=True)

    A = Matrix([[0, 1*I], [2, 0]])
    # if singular values can be sorted, they should be in decreasing order
    assert A.singular_values() == [2, 1]

    A = eye(3)
    A[1, 1] = x
    A[2, 2] = 5
    vals = A.singular_values()
    # since Abs(x) cannot be sorted, test set equality
    assert set(vals) == {5, 1, Abs(x)}

    A = Matrix([[sin(x), cos(x)], [-cos(x), sin(x)]])
    vals = [sv.trigsimp() for sv in A.singular_values()]
    assert vals == [S.One, S.One]

    A = Matrix([
        [2, 4],
        [1, 3],
        [0, 0],
        [0, 0]
        ])
    assert A.singular_values() == \
        [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
    assert A.T.singular_values() == \
        [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221)), 0, 0]

def test___eq__():
    assert (Matrix(
        [[0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]]) == {}) is False


def test_definite():
    # Examples from Gilbert Strang, "Introduction to Linear Algebra"
    # Positive definite matrices
    m = Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[5, 4], [4, 5]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Positive semidefinite matrices
    m = Matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[1, 2], [2, 4]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Examples from Mathematica documentation
    # Non-hermitian positive definite matrices
    m = Matrix([[2, 3], [4, 8]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Hermetian matrices
    m = Matrix([[1, 2*I], [-I, 4]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    # Symbolic matrices examples
    a = Symbol('a', positive=True)
    b = Symbol('b', negative=True)
    m = Matrix([[a, 0, 0], [0, a, 0], [0, 0, a]])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == False

    m = Matrix([[b, 0, 0], [0, b, 0], [0, 0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == True
    assert m.is_negative_semidefinite == True
    assert m.is_indefinite == False

    m = Matrix([[a, 0], [0, b]])
    assert m.is_positive_definite == False
    assert m.is_positive_semidefinite == False
    assert m.is_negative_definite == False
    assert m.is_negative_semidefinite == False
    assert m.is_indefinite == True

    m = Matrix([
        [0.0228202735623867, 0.00518748979085398,
         -0.0743036351048907, -0.00709135324903921],
        [0.00518748979085398, 0.0349045359786350,
         0.0830317991056637, 0.00233147902806909],
        [-0.0743036351048907, 0.0830317991056637,
         1.15859676366277, 0.340359081555988],
        [-0.00709135324903921, 0.00233147902806909,
         0.340359081555988, 0.928147644848199]
    ])
    assert m.is_positive_definite == True
    assert m.is_positive_semidefinite == True
    assert m.is_indefinite == False

    # test for issue 19547: https://github.com/sympy/sympy/issues/19547
    m = Matrix([
        [0, 0, 0],
        [0, 1, 2],
        [0, 2, 1]
    ])
    assert not m.is_positive_definite
    assert not m.is_positive_semidefinite


def test_positive_semidefinite_cholesky():
    from sympy.matrices.eigen import _is_positive_semidefinite_cholesky

    m = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[0, 0, 0], [0, 5, -10*I], [0, 10*I, 5]])
    assert _is_positive_semidefinite_cholesky(m) == False
    m = Matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    assert _is_positive_semidefinite_cholesky(m) == False
    m = Matrix([[0, 1], [1, 0]])
    assert _is_positive_semidefinite_cholesky(m) == False

    # https://www.value-at-risk.net/cholesky-factorization/
    m = Matrix([[4, -2, -6], [-2, 10, 9], [-6, 9, 14]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[9, -3, 3], [-3, 2, 1], [3, 1, 6]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[4, -2, 2], [-2, 1, -1], [2, -1, 5]])
    assert _is_positive_semidefinite_cholesky(m) == True
    m = Matrix([[1, 2, -1], [2, 5, 1], [-1, 1, 9]])
    assert _is_positive_semidefinite_cholesky(m) == False


def test_issue_20582():
    A = Matrix([
        [5, -5, -3, 2, -7],
        [-2, -5, 0, 2, 1],
        [-2, -7, -5, -2, -6],
        [7, 10, 3, 9, -2],
        [4, -10, 3, -8, -4]
    ])
    # XXX Used dry-run test because arbitrary symbol that appears in
    # CRootOf may not be unique.
    assert A.eigenvects()

def test_issue_19210():
    t = Symbol('t')
    H = Matrix([[3, 0, 0, 0], [0, 1 , 2, 0], [0, 2, 2, 0], [0, 0, 0, 4]])
    A = (-I * H * t).jordan_form()
    assert A == (Matrix([
                    [0, 1,                  0,                0],
                    [0, 0, -4/(-1 + sqrt(17)), 4/(1 + sqrt(17))],
                    [0, 0,                  1,                1],
                    [1, 0,                  0,                0]]), Matrix([
                    [-4*I*t,      0,                         0,                         0],
                    [     0, -3*I*t,                         0,                         0],
                    [     0,      0, t*(-3*I/2 + sqrt(17)*I/2),                         0],
                    [     0,      0,                         0, t*(-sqrt(17)*I/2 - 3*I/2)]]))


def test_issue_20275():
    # XXX We use complex expansions because complex exponentials are not
    # recognized by polys.domains
    A = DFT(3).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[1 - sqrt(3)], [1], [1]])]
    )
    assert eigenvects[1] == (
        1, 1,
        [Matrix([[1 + sqrt(3)], [1], [1]])]
    )
    assert eigenvects[2] == (
        -I, 1,
        [Matrix([[0], [-1], [1]])]
    )

    A = DFT(4).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[-1], [1], [1], [1]])]
    )
    assert eigenvects[1] == (
        1, 2,
        [Matrix([[1], [0], [1], [0]]), Matrix([[2], [1], [0], [1]])]
    )
    assert eigenvects[2] == (
        -I, 1,
        [Matrix([[0], [-1], [0], [1]])]
    )

    # XXX We skip test for some parts of eigenvectors which are very
    # complicated and fragile under expression tree changes
    A = DFT(5).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (
        -1, 1,
        [Matrix([[1 - sqrt(5)], [1], [1], [1], [1]])]
    )
    assert eigenvects[1] == (
        1, 2,
        [Matrix([[S(1)/2 + sqrt(5)/2], [0], [1], [1], [0]]),
         Matrix([[S(1)/2 + sqrt(5)/2], [1], [0], [0], [1]])]
    )


def test_issue_20752():
    b = symbols('b', nonzero=True)
    m = Matrix([[0, 0, 0], [0, b, 0], [0, 0, b]])
    assert m.is_positive_semidefinite is None


def test_issue_25282():
    dd = sd = [0] * 11 + [1]
    ds = [2, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    ss = ds.copy()
    ss[8] = 2

    def rotate(x, i):
        return x[i:] + x[:i]

    mat = []
    for i in range(12):
        mat.append(rotate(ss, i) + rotate(sd, i))
    for i in range(12):
        mat.append(rotate(ds, i) + rotate(dd, i))

    assert sum(Matrix(mat).eigenvals().values()) == 24

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_euclidtools.py ===
"""Tests for Euclidean algorithms, GCDs, LCMs and polynomial remainder sequences. """

from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR

from sympy.polys.specialpolys import (
    f_polys,
    dmp_fateman_poly_F_1,
    dmp_fateman_poly_F_2,
    dmp_fateman_poly_F_3)

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()

def test_dup_gcdex():
    R, x = ring("x", QQ)

    f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    g = x**3 + x**2 - 4*x - 4

    s = -QQ(1,5)*x + QQ(3,5)
    t = QQ(1,5)*x**2 - QQ(6,5)*x + 2
    h = x + 1

    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)

    f = x**4 + 4*x**3 - x + 1
    g = x**3 - x + 1

    s, t, h = R.dup_gcdex(f, g)
    S, T, H = R.dup_gcdex(g, f)

    assert R.dup_add(R.dup_mul(s, f),
                     R.dup_mul(t, g)) == h
    assert R.dup_add(R.dup_mul(S, g),
                     R.dup_mul(T, f)) == H

    f = 2*x
    g = x**2 - 16

    s = QQ(1,32)*x
    t = -QQ(1,16)
    h = 1

    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)


def test_dup_invert():
    R, x = ring("x", QQ)
    assert R.dup_invert(2*x, x**2 - 16) == QQ(1,32)*x


def test_dup_euclidean_prs():
    R, x = ring("x", QQ)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    assert R.dup_euclidean_prs(f, g) == [
        f,
        g,
        -QQ(5,9)*x**4 + QQ(1,9)*x**2 - QQ(1,3),
        -QQ(117,25)*x**2 - 9*x + QQ(441,25),
        QQ(233150,19773)*x - QQ(102500,6591),
        -QQ(1288744821,543589225)]


def test_dup_primitive_prs():
    R, x = ring("x", ZZ)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    assert R.dup_primitive_prs(f, g) == [
        f,
        g,
        -5*x**4 + x**2 - 3,
        13*x**2 + 25*x - 49,
        4663*x - 6150,
        1]


def test_dup_subresultants():
    R, x = ring("x", ZZ)

    assert R.dup_resultant(0, 0) == 0

    assert R.dup_resultant(1, 0) == 0
    assert R.dup_resultant(0, 1) == 0

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    a = 15*x**4 - 3*x**2 + 9
    b = 65*x**2 + 125*x - 245
    c = 9326*x - 12300
    d = 260708

    assert R.dup_subresultants(f, g) == [f, g, a, b, c, d]
    assert R.dup_resultant(f, g) == R.dup_LC(d)

    f = x**2 - 2*x + 1
    g = x**2 - 1

    a = 2*x - 2

    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 0

    f = x**2 + 1
    g = x**2 - 1

    a = -2

    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 4

    f = x**2 - 1
    g = x**3 - x**2 + 2

    assert R.dup_resultant(f, g) == 0

    f = 3*x**3 - x
    g = 5*x**2 + 1

    assert R.dup_resultant(f, g) == 64

    f = x**2 - 2*x + 7
    g = x**3 - x + 5

    assert R.dup_resultant(f, g) == 265

    f = x**3 - 6*x**2 + 11*x - 6
    g = x**3 - 15*x**2 + 74*x - 120

    assert R.dup_resultant(f, g) == -8640

    f = x**3 - 6*x**2 + 11*x - 6
    g = x**3 - 10*x**2 + 29*x - 20

    assert R.dup_resultant(f, g) == 0

    f = x**3 - 1
    g = x**3 + 2*x**2 + 2*x - 1

    assert R.dup_resultant(f, g) == 16

    f = x**8 - 2
    g = x - 1

    assert R.dup_resultant(f, g) == -1


def test_dmp_subresultants():
    R, x, y = ring("x,y", ZZ)

    assert R.dmp_resultant(0, 0) == 0
    assert R.dmp_prs_resultant(0, 0)[0] == 0
    assert R.dmp_zz_collins_resultant(0, 0) == 0
    assert R.dmp_qq_collins_resultant(0, 0) == 0

    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0

    assert R.dmp_resultant(0, 1) == 0
    assert R.dmp_prs_resultant(0, 1)[0] == 0
    assert R.dmp_zz_collins_resultant(0, 1) == 0
    assert R.dmp_qq_collins_resultant(0, 1) == 0

    f = 3*x**2*y - y**3 - 4
    g = x**2 + x*y**3 - 9

    a = 3*x*y**4 + y**3 - 27*y + 4
    b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    r = R.dmp_LC(b)

    assert R.dmp_subresultants(f, g) == [f, g, a, b]

    assert R.dmp_resultant(f, g) == r
    assert R.dmp_prs_resultant(f, g)[0] == r
    assert R.dmp_zz_collins_resultant(f, g) == r
    assert R.dmp_qq_collins_resultant(f, g) == r

    f = -x**3 + 5
    g = 3*x**2*y + x**2

    a = 45*y**2 + 30*y + 5
    b = 675*y**3 + 675*y**2 + 225*y + 25

    r = R.dmp_LC(b)

    assert R.dmp_subresultants(f, g) == [f, g, a]
    assert R.dmp_resultant(f, g) == r
    assert R.dmp_prs_resultant(f, g)[0] == r
    assert R.dmp_zz_collins_resultant(f, g) == r
    assert R.dmp_qq_collins_resultant(f, g) == r

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f = 6*x**2 - 3*x*y - 2*x*z + y*z
    g = x**2 - x*u - x*v + u*v

    r = y**2*z**2 - 3*y**2*z*u - 3*y**2*z*v + 9*y**2*u*v - 2*y*z**2*u \
      - 2*y*z**2*v + 6*y*z*u**2 + 12*y*z*u*v + 6*y*z*v**2 - 18*y*u**2*v \
      - 18*y*u*v**2 + 4*z**2*u*v - 12*z*u**2*v - 12*z*u*v**2 + 36*u**2*v**2

    assert R.dmp_zz_collins_resultant(f, g) == r.drop(x)

    R, x, y, z, u, v = ring("x,y,z,u,v", QQ)

    f = x**2 - QQ(1,2)*x*y - QQ(1,3)*x*z + QQ(1,6)*y*z
    g = x**2 - x*u - x*v + u*v

    r = QQ(1,36)*y**2*z**2 - QQ(1,12)*y**2*z*u - QQ(1,12)*y**2*z*v + QQ(1,4)*y**2*u*v \
      - QQ(1,18)*y*z**2*u - QQ(1,18)*y*z**2*v + QQ(1,6)*y*z*u**2 + QQ(1,3)*y*z*u*v \
      + QQ(1,6)*y*z*v**2 - QQ(1,2)*y*u**2*v - QQ(1,2)*y*u*v**2 + QQ(1,9)*z**2*u*v \
      - QQ(1,3)*z*u**2*v - QQ(1,3)*z*u*v**2 + u**2*v**2

    assert R.dmp_qq_collins_resultant(f, g) == r.drop(x)

    Rt, t = ring("t", ZZ)
    Rx, x = ring("x", Rt)

    f = x**6 - 5*x**4 + 5*x**2 + 4
    g = -6*t*x**5 + x**4 + 20*t*x**3 - 3*x**2 - 10*t*x + 6

    assert Rx.dup_resultant(f, g) == 2930944*t**6 + 2198208*t**4 + 549552*t**2 + 45796


def test_dup_discriminant():
    R, x = ring("x", ZZ)

    assert R.dup_discriminant(0) == 0
    assert R.dup_discriminant(x) == 1

    assert R.dup_discriminant(x**3 + 3*x**2 + 9*x - 13) == -11664
    assert R.dup_discriminant(5*x**5 + x**3 + 2) == 31252160
    assert R.dup_discriminant(x**4 + 2*x**3 + 6*x**2 - 22*x + 13) == 0
    assert R.dup_discriminant(12*x**7 + 15*x**4 + 30*x**3 + x**2 + 1) == -220289699947514112


def test_dmp_discriminant():
    R, x = ring("x", ZZ)

    assert R.dmp_discriminant(0) == 0

    R, x, y = ring("x,y", ZZ)

    assert R.dmp_discriminant(0) == 0
    assert R.dmp_discriminant(y) == 0

    assert R.dmp_discriminant(x**3 + 3*x**2 + 9*x - 13) == -11664
    assert R.dmp_discriminant(5*x**5 + x**3 + 2) == 31252160
    assert R.dmp_discriminant(x**4 + 2*x**3 + 6*x**2 - 22*x + 13) == 0
    assert R.dmp_discriminant(12*x**7 + 15*x**4 + 30*x**3 + x**2 + 1) == -220289699947514112

    assert R.dmp_discriminant(x**2*y + 2*y) == (-8*y**2).drop(x)
    assert R.dmp_discriminant(x*y**2 + 2*x) == 1

    R, x, y, z = ring("x,y,z", ZZ)
    assert R.dmp_discriminant(x*y + z) == 1

    R, x, y, z, u = ring("x,y,z,u", ZZ)
    assert R.dmp_discriminant(x**2*y + x*z + u) == (-4*y*u + z**2).drop(x)

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)
    assert R.dmp_discriminant(x**3*y + x**2*z + x*u + v) == \
        (-27*y**2*v**2 + 18*y*z*u*v - 4*y*u**3 - 4*z**3*v + z**2*u**2).drop(x)


def test_dup_gcd():
    R, x = ring("x", ZZ)

    f, g = 0, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (0, 0, 0)

    f, g = 2, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, 0)

    f, g = -2, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, -1, 0)

    f, g = 0, -2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 0, -1)

    f, g = 0, 2*x + 4
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2*x + 4, 0, 1)

    f, g = 2*x + 4, 0
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2*x + 4, 1, 0)

    f, g = 2, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, 1)

    f, g = -2, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, -1, 1)

    f, g = 2, -2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, -1)

    f, g = -2, -2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, -1, -1)

    f, g = x**2 + 2*x + 1, 1
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 1)

    f, g = x**2 + 2*x + 1, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 2)

    f, g = 2*x**2 + 4*x + 2, 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, x**2 + 2*x + 1, 1)

    f, g = 2, 2*x**2 + 4*x + 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, x**2 + 2*x + 1)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (x + 1, 1, 2*x + 2)

    f, g = x - 31, x
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (1, f, g)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_rr_prs_gcd(f, g) == (h, cff, cfg)

    f = x**4 - 4
    g = x**4 + 4*x**2 + 4

    h = x**2 + 2

    cff = x**2 - 2
    cfg = x**2 + 2

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_rr_prs_gcd(f, g) == (h, cff, cfg)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_rr_prs_gcd(f, g) == (h, cff, cfg)

    R, x = ring("x", QQ)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert R.dup_qq_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dup_ff_prs_gcd(f, g) == (h, cff, cfg)

    R, x = ring("x", ZZ)

    f = - 352518131239247345597970242177235495263669787845475025293906825864749649589178600387510272*x**49 \
        + 46818041807522713962450042363465092040687472354933295397472942006618953623327997952*x**42 \
        + 378182690892293941192071663536490788434899030680411695933646320291525827756032*x**35 \
        + 112806468807371824947796775491032386836656074179286744191026149539708928*x**28 \
        - 12278371209708240950316872681744825481125965781519138077173235712*x**21 \
        + 289127344604779611146960547954288113529690984687482920704*x**14 \
        + 19007977035740498977629742919480623972236450681*x**7 \
        + 311973482284542371301330321821976049

    g =   365431878023781158602430064717380211405897160759702125019136*x**21 \
        + 197599133478719444145775798221171663643171734081650688*x**14 \
        - 9504116979659010018253915765478924103928886144*x**7 \
        - 311973482284542371301330321821976049

    assert R.dup_zz_heu_gcd(f, R.dup_diff(f, 1))[0] == g
    assert R.dup_rr_prs_gcd(f, R.dup_diff(f, 1))[0] == g

    R, x = ring("x", QQ)

    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)

    h = x + 1

    assert R.dup_qq_heu_gcd(f, g) == (h, g, QQ(1,2))
    assert R.dup_ff_prs_gcd(f, g) == (h, g, QQ(1,2))

    R, x = ring("x", ZZ)

    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760

    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1

    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)


def test_dmp_gcd():
    R, x, y = ring("x,y", ZZ)

    f, g = 0, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (0, 0, 0)

    f, g = 2, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, 0)

    f, g = -2, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, 0)

    f, g = 0, -2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 0, -1)

    f, g = 0, 2*x + 4
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2*x + 4, 0, 1)

    f, g = 2*x + 4, 0
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2*x + 4, 1, 0)

    f, g = 2, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, 1)

    f, g = -2, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, 1)

    f, g = 2, -2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, -1)

    f, g = -2, -2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, -1)

    f, g = x**2 + 2*x + 1, 1
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 1)

    f, g = x**2 + 2*x + 1, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 2)

    f, g = 2*x**2 + 4*x + 2, 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, x**2 + 2*x + 1, 1)

    f, g = 2, 2*x**2 + 4*x + 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, x**2 + 2*x + 1)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (x + 1, 1, 2*x + 2)

    R, x, y, z, u = ring("x,y,z,u", ZZ)

    f, g = u**2 + 2*u + 1, 2*u + 2
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (u + 1, u + 1, 2)

    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1

    assert R.dmp_zz_heu_gcd(f, g) == (h, cff, cfg)
    assert R.dmp_rr_prs_gcd(f, g) == (h, cff, cfg)

    assert R.dmp_zz_heu_gcd(g, f) == (h, cfg, cff)
    assert R.dmp_rr_prs_gcd(g, f) == (h, cfg, cff)

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(4, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v, a, b = ring("x,y,z,u,v,a,b", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(6, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(8, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_2(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_3(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = map(R.from_dense, dmp_fateman_poly_F_3(4, ZZ))
    H, cff, cfg = R.dmp_inner_gcd(f, g)

    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    R, x, y = ring("x,y", QQ)

    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)

    h = x + 1

    assert R.dmp_qq_heu_gcd(f, g) == (h, g, QQ(1,2))
    assert R.dmp_ff_prs_gcd(f, g) == (h, g, QQ(1,2))

    R, x, y = ring("x,y", RR)

    f = 2.1*x*y**2 - 2.2*x*y + 2.1*x
    g = 1.0*x**3

    assert R.dmp_ff_prs_gcd(f, g) == \
        (1.0*x, 2.1*y**2 - 2.2*y + 2.1, 1.0*x**2)


def test_dup_lcm():
    R, x = ring("x", ZZ)

    assert R.dup_lcm(2, 6) == 6

    assert R.dup_lcm(2*x**3, 6*x) == 6*x**3
    assert R.dup_lcm(2*x**3, 3*x) == 6*x**3

    assert R.dup_lcm(x**2 + x, x) == x**2 + x
    assert R.dup_lcm(x**2 + x, 2*x) == 2*x**2 + 2*x
    assert R.dup_lcm(x**2 + 2*x, x) == x**2 + 2*x
    assert R.dup_lcm(2*x**2 + x, x) == 2*x**2 + x
    assert R.dup_lcm(2*x**2 + x, 2*x) == 4*x**2 + 2*x


def test_dmp_lcm():
    R, x, y = ring("x,y", ZZ)

    assert R.dmp_lcm(2, 6) == 6
    assert R.dmp_lcm(x, y) == x*y

    assert R.dmp_lcm(2*x**3, 6*x*y**2) == 6*x**3*y**2
    assert R.dmp_lcm(2*x**3, 3*x*y**2) == 6*x**3*y**2

    assert R.dmp_lcm(x**2*y, x*y**2) == x**2*y**2

    f = 2*x*y**5 - 3*x*y**4 - 2*x*y**3 + 3*x*y**2
    g = y**5 - 2*y**3 + y
    h = 2*x*y**7 - 3*x*y**6 - 4*x*y**5 + 6*x*y**4 + 2*x*y**3 - 3*x*y**2

    assert R.dmp_lcm(f, g) == h

    f = x**3 - 3*x**2*y - 9*x*y**2 - 5*y**3
    g = x**4 + 6*x**3*y + 12*x**2*y**2 + 10*x*y**3 + 3*y**4
    h = x**5 + x**4*y - 18*x**3*y**2 - 50*x**2*y**3 - 47*x*y**4 - 15*y**5

    assert R.dmp_lcm(f, g) == h


def test_dmp_content():
    R, x,y = ring("x,y", ZZ)

    assert R.dmp_content(-2) == 2

    f, g, F = 3*y**2 + 2*y + 1, 1, 0

    for i in range(0, 5):
        g *= f
        F += x**i*g

    assert R.dmp_content(F) == f.drop(x)

    R, x,y,z = ring("x,y,z", ZZ)

    assert R.dmp_content(f_4) == 1
    assert R.dmp_content(f_5) == 1

    R, x,y,z,t = ring("x,y,z,t", ZZ)
    assert R.dmp_content(f_6) == 1


def test_dmp_primitive():
    R, x,y = ring("x,y", ZZ)

    assert R.dmp_primitive(0) == (0, 0)
    assert R.dmp_primitive(1) == (1, 1)

    f, g, F = 3*y**2 + 2*y + 1, 1, 0

    for i in range(0, 5):
        g *= f
        F += x**i*g

    assert R.dmp_primitive(F) == (f.drop(x), F / f)

    R, x,y,z = ring("x,y,z", ZZ)

    cont, f = R.dmp_primitive(f_4)
    assert cont == 1 and f == f_4
    cont, f = R.dmp_primitive(f_5)
    assert cont == 1 and f == f_5

    R, x,y,z,t = ring("x,y,z,t", ZZ)

    cont, f = R.dmp_primitive(f_6)
    assert cont == 1 and f == f_6


def test_dup_cancel():
    R, x = ring("x", ZZ)

    f = 2*x**2 - 2
    g = x**2 - 2*x + 1

    p = 2*x + 2
    q = x - 1

    assert R.dup_cancel(f, g) == (p, q)
    assert R.dup_cancel(f, g, include=False) == (1, 1, p, q)

    f = -x - 2
    g = 3*x - 4

    F = x + 2
    G = -3*x + 4

    assert R.dup_cancel(f, g) == (f, g)
    assert R.dup_cancel(F, G) == (f, g)

    assert R.dup_cancel(0, 0) == (0, 0)
    assert R.dup_cancel(0, 0, include=False) == (1, 1, 0, 0)

    assert R.dup_cancel(x, 0) == (1, 0)
    assert R.dup_cancel(x, 0, include=False) == (1, 1, 1, 0)

    assert R.dup_cancel(0, x) == (0, 1)
    assert R.dup_cancel(0, x, include=False) == (1, 1, 0, 1)

    f = 0
    g = x
    one = 1

    assert R.dup_cancel(f, g, include=True) == (f, one)


def test_dmp_cancel():
    R, x, y = ring("x,y", ZZ)

    f = 2*x**2 - 2
    g = x**2 - 2*x + 1

    p = 2*x + 2
    q = x - 1

    assert R.dmp_cancel(f, g) == (p, q)
    assert R.dmp_cancel(f, g, include=False) == (1, 1, p, q)

    assert R.dmp_cancel(0, 0) == (0, 0)
    assert R.dmp_cancel(0, 0, include=False) == (1, 1, 0, 0)

    assert R.dmp_cancel(y, 0) == (1, 0)
    assert R.dmp_cancel(y, 0, include=False) == (1, 1, 1, 0)

    assert R.dmp_cancel(0, y) == (0, 1)
    assert R.dmp_cancel(0, y, include=False) == (1, 1, 0, 1)