
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_style.py ===
import pytest

from pandas import Series

pytest.importorskip("matplotlib")
from pandas.plotting._matplotlib.style import get_standard_colors


class TestGetStandardColors:
    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (3, ["red", "green", "blue"]),
            (5, ["red", "green", "blue", "red", "green"]),
            (7, ["red", "green", "blue", "red", "green", "blue", "red"]),
            (2, ["red", "green"]),
            (1, ["red"]),
        ],
    )
    def test_default_colors_named_from_prop_cycle(self, num_colors, expected):
        import matplotlib as mpl
        from matplotlib.pyplot import cycler

        mpl_params = {
            "axes.prop_cycle": cycler(color=["red", "green", "blue"]),
        }
        with mpl.rc_context(rc=mpl_params):
            result = get_standard_colors(num_colors=num_colors)
            assert result == expected

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, ["b"]),
            (3, ["b", "g", "r"]),
            (4, ["b", "g", "r", "y"]),
            (5, ["b", "g", "r", "y", "b"]),
            (7, ["b", "g", "r", "y", "b", "g", "r"]),
        ],
    )
    def test_default_colors_named_from_prop_cycle_string(self, num_colors, expected):
        import matplotlib as mpl
        from matplotlib.pyplot import cycler

        mpl_params = {
            "axes.prop_cycle": cycler(color="bgry"),
        }
        with mpl.rc_context(rc=mpl_params):
            result = get_standard_colors(num_colors=num_colors)
            assert result == expected

    @pytest.mark.parametrize(
        "num_colors, expected_name",
        [
            (1, ["C0"]),
            (3, ["C0", "C1", "C2"]),
            (
                12,
                [
                    "C0",
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                    "C7",
                    "C8",
                    "C9",
                    "C0",
                    "C1",
                ],
            ),
        ],
    )
    def test_default_colors_named_undefined_prop_cycle(self, num_colors, expected_name):
        import matplotlib as mpl
        import matplotlib.colors as mcolors

        with mpl.rc_context(rc={}):
            expected = [mcolors.to_hex(x) for x in expected_name]
            result = get_standard_colors(num_colors=num_colors)
            assert result == expected

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, ["red", "green", (0.1, 0.2, 0.3)]),
            (2, ["red", "green", (0.1, 0.2, 0.3)]),
            (3, ["red", "green", (0.1, 0.2, 0.3)]),
            (4, ["red", "green", (0.1, 0.2, 0.3), "red"]),
        ],
    )
    def test_user_input_color_sequence(self, num_colors, expected):
        color = ["red", "green", (0.1, 0.2, 0.3)]
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, ["r", "g", "b", "k"]),
            (2, ["r", "g", "b", "k"]),
            (3, ["r", "g", "b", "k"]),
            (4, ["r", "g", "b", "k"]),
            (5, ["r", "g", "b", "k", "r"]),
            (6, ["r", "g", "b", "k", "r", "g"]),
        ],
    )
    def test_user_input_color_string(self, num_colors, expected):
        color = "rgbk"
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize(
        "num_colors, expected",
        [
            (1, [(0.1, 0.2, 0.3)]),
            (2, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]),
            (3, [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]),
        ],
    )
    def test_user_input_color_floats(self, num_colors, expected):
        color = (0.1, 0.2, 0.3)
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize(
        "color, num_colors, expected",
        [
            ("Crimson", 1, ["Crimson"]),
            ("DodgerBlue", 2, ["DodgerBlue", "DodgerBlue"]),
            ("firebrick", 3, ["firebrick", "firebrick", "firebrick"]),
        ],
    )
    def test_user_input_named_color_string(self, color, num_colors, expected):
        result = get_standard_colors(color=color, num_colors=num_colors)
        assert result == expected

    @pytest.mark.parametrize("color", ["", [], (), Series([], dtype="object")])
    def test_empty_color_raises(self, color):
        with pytest.raises(ValueError, match="Invalid color argument"):
            get_standard_colors(color=color, num_colors=1)

    @pytest.mark.parametrize(
        "color",
        [
            "bad_color",
            ("red", "green", "bad_color"),
            (0.1,),
            (0.1, 0.2),
            (0.1, 0.2, 0.3, 0.4, 0.5),  # must be either 3 or 4 floats
        ],
    )
    def test_bad_color_raises(self, color):
        with pytest.raises(ValueError, match="Invalid color"):
            get_standard_colors(color=color, num_colors=5)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_power.py ===
from mpmath import *
from mpmath.libmp import *

import random

def test_fractional_pow():
    mp.dps = 15
    assert mpf(16) ** 2.5 == 1024
    assert mpf(64) ** 0.5 == 8
    assert mpf(64) ** -0.5 == 0.125
    assert mpf(16) ** -2.5 == 0.0009765625
    assert (mpf(10) ** 0.5).ae(3.1622776601683791)
    assert (mpf(10) ** 2.5).ae(316.2277660168379)
    assert (mpf(10) ** -0.5).ae(0.31622776601683794)
    assert (mpf(10) ** -2.5).ae(0.0031622776601683794)
    assert (mpf(10) ** 0.3).ae(1.9952623149688795)
    assert (mpf(10) ** -0.3).ae(0.50118723362727224)

def test_pow_integer_direction():
    """
    Test that inexact integer powers are rounded in the right
    direction.
    """
    random.seed(1234)
    for prec in [10, 53, 200]:
        for i in range(50):
            a = random.randint(1<<(prec-1), 1<<prec)
            b = random.randint(2, 100)
            ab = a**b
            # note: could actually be exact, but that's very unlikely!
            assert to_int(mpf_pow(from_int(a), from_int(b), prec, round_down)) < ab
            assert to_int(mpf_pow(from_int(a), from_int(b), prec, round_up)) > ab


def test_pow_epsilon_rounding():
    """
    Stress test directed rounding for powers with integer exponents.
    Basically, we look at the following cases:

    >>> 1.0001 ** -5 # doctest: +SKIP
    0.99950014996500702
    >>> 0.9999 ** -5 # doctest: +SKIP
    1.000500150035007
    >>> (-1.0001) ** -5 # doctest: +SKIP
    -0.99950014996500702
    >>> (-0.9999) ** -5 # doctest: +SKIP
    -1.000500150035007

    >>> 1.0001 ** -6 # doctest: +SKIP
    0.99940020994401269
    >>> 0.9999 ** -6 # doctest: +SKIP
    1.0006002100560125
    >>> (-1.0001) ** -6 # doctest: +SKIP
    0.99940020994401269
    >>> (-0.9999) ** -6 # doctest: +SKIP
    1.0006002100560125

    etc.

    We run the tests with values a very small epsilon away from 1:
    small enough that the result is indistinguishable from 1 when
    rounded to nearest at the output precision. We check that the
    result is not erroneously rounded to 1 in cases where the
    rounding should be done strictly away from 1.
    """

    def powr(x, n, r):
        return make_mpf(mpf_pow_int(x._mpf_, n, mp.prec, r))

    for (inprec, outprec) in [(100, 20), (5000, 3000)]:

        mp.prec = inprec

        pos10001 = mpf(1) + mpf(2)**(-inprec+5)
        pos09999 = mpf(1) - mpf(2)**(-inprec+5)
        neg10001 = -pos10001
        neg09999 = -pos09999

        mp.prec = outprec
        r = round_up
        assert powr(pos10001, 5, r) > 1
        assert powr(pos09999, 5, r) == 1
        assert powr(neg10001, 5, r) < -1
        assert powr(neg09999, 5, r) == -1
        assert powr(pos10001, 6, r) > 1
        assert powr(pos09999, 6, r) == 1
        assert powr(neg10001, 6, r) > 1
        assert powr(neg09999, 6, r) == 1

        assert powr(pos10001, -5, r) == 1
        assert powr(pos09999, -5, r) > 1
        assert powr(neg10001, -5, r) == -1
        assert powr(neg09999, -5, r) < -1
        assert powr(pos10001, -6, r) == 1
        assert powr(pos09999, -6, r) > 1
        assert powr(neg10001, -6, r) == 1
        assert powr(neg09999, -6, r) > 1

        r = round_down
        assert powr(pos10001, 5, r) == 1
        assert powr(pos09999, 5, r) < 1
        assert powr(neg10001, 5, r) == -1
        assert powr(neg09999, 5, r) > -1
        assert powr(pos10001, 6, r) == 1
        assert powr(pos09999, 6, r) < 1
        assert powr(neg10001, 6, r) == 1
        assert powr(neg09999, 6, r) < 1

        assert powr(pos10001, -5, r) < 1
        assert powr(pos09999, -5, r) == 1
        assert powr(neg10001, -5, r) > -1
        assert powr(neg09999, -5, r) == -1
        assert powr(pos10001, -6, r) < 1
        assert powr(pos09999, -6, r) == 1
        assert powr(neg10001, -6, r) < 1
        assert powr(neg09999, -6, r) == 1

        r = round_ceiling
        assert powr(pos10001, 5, r) > 1
        assert powr(pos09999, 5, r) == 1
        assert powr(neg10001, 5, r) == -1
        assert powr(neg09999, 5, r) > -1
        assert powr(pos10001, 6, r) > 1
        assert powr(pos09999, 6, r) == 1
        assert powr(neg10001, 6, r) > 1
        assert powr(neg09999, 6, r) == 1

        assert powr(pos10001, -5, r) == 1
        assert powr(pos09999, -5, r) > 1
        assert powr(neg10001, -5, r) > -1
        assert powr(neg09999, -5, r) == -1
        assert powr(pos10001, -6, r) == 1
        assert powr(pos09999, -6, r) > 1
        assert powr(neg10001, -6, r) == 1
        assert powr(neg09999, -6, r) > 1

        r = round_floor
        assert powr(pos10001, 5, r) == 1
        assert powr(pos09999, 5, r) < 1
        assert powr(neg10001, 5, r) < -1
        assert powr(neg09999, 5, r) == -1
        assert powr(pos10001, 6, r) == 1
        assert powr(pos09999, 6, r) < 1
        assert powr(neg10001, 6, r) == 1
        assert powr(neg09999, 6, r) < 1

        assert powr(pos10001, -5, r) < 1
        assert powr(pos09999, -5, r) == 1
        assert powr(neg10001, -5, r) == -1
        assert powr(neg09999, -5, r) < -1
        assert powr(pos10001, -6, r) < 1
        assert powr(pos09999, -6, r) == 1
        assert powr(neg10001, -6, r) < 1
        assert powr(neg09999, -6, r) == 1

    mp.dps = 15

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\period\test_constructors.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    PeriodArray,
    period_array,
)


@pytest.mark.parametrize(
    "data, freq, expected",
    [
        ([pd.Period("2017", "D")], None, [17167]),
        ([pd.Period("2017", "D")], "D", [17167]),
        ([2017], "D", [17167]),
        (["2017"], "D", [17167]),
        ([pd.Period("2017", "D")], pd.tseries.offsets.Day(), [17167]),
        ([pd.Period("2017", "D"), None], None, [17167, iNaT]),
        (pd.Series(pd.date_range("2017", periods=3)), None, [17167, 17168, 17169]),
        (pd.date_range("2017", periods=3), None, [17167, 17168, 17169]),
        (pd.period_range("2017", periods=4, freq="Q"), None, [188, 189, 190, 191]),
    ],
)
def test_period_array_ok(data, freq, expected):
    result = period_array(data, freq=freq).asi8
    expected = np.asarray(expected, dtype=np.int64)
    tm.assert_numpy_array_equal(result, expected)


def test_period_array_readonly_object():
    # https://github.com/pandas-dev/pandas/issues/25403
    pa = period_array([pd.Period("2019-01-01")])
    arr = np.asarray(pa, dtype="object")
    arr.setflags(write=False)

    result = period_array(arr)
    tm.assert_period_array_equal(result, pa)

    result = pd.Series(arr)
    tm.assert_series_equal(result, pd.Series(pa))

    result = pd.DataFrame({"A": arr})
    tm.assert_frame_equal(result, pd.DataFrame({"A": pa}))


def test_from_datetime64_freq_changes():
    # https://github.com/pandas-dev/pandas/issues/23438
    arr = pd.date_range("2017", periods=3, freq="D")
    result = PeriodArray._from_datetime64(arr, freq="M")
    expected = period_array(["2017-01-01", "2017-01-01", "2017-01-01"], freq="M")
    tm.assert_period_array_equal(result, expected)


@pytest.mark.parametrize("freq", ["2M", MonthEnd(2)])
def test_from_datetime64_freq_2M(freq):
    arr = np.array(
        ["2020-01-01T00:00:00", "2020-01-02T00:00:00"], dtype="datetime64[ns]"
    )
    result = PeriodArray._from_datetime64(arr, freq)
    expected = period_array(["2020-01", "2020-01"], freq=freq)
    tm.assert_period_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, freq, msg",
    [
        (
            [pd.Period("2017", "D"), pd.Period("2017", "Y")],
            None,
            "Input has different freq",
        ),
        ([pd.Period("2017", "D")], "Y", "Input has different freq"),
    ],
)
def test_period_array_raises(data, freq, msg):
    with pytest.raises(IncompatibleFrequency, match=msg):
        period_array(data, freq)


def test_period_array_non_period_series_raies():
    ser = pd.Series([1, 2, 3])
    with pytest.raises(TypeError, match="dtype"):
        PeriodArray(ser, dtype="period[D]")


def test_period_array_freq_mismatch():
    arr = period_array(["2000", "2001"], freq="D")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        PeriodArray(arr, dtype="period[M]")

    dtype = pd.PeriodDtype(pd.tseries.offsets.MonthEnd())
    with pytest.raises(IncompatibleFrequency, match="freq"):
        PeriodArray(arr, dtype=dtype)


def test_from_sequence_disallows_i8():
    arr = period_array(["2000", "2001"], freq="D")

    msg = str(arr[0].ordinal)
    with pytest.raises(TypeError, match=msg):
        PeriodArray._from_sequence(arr.asi8, dtype=arr.dtype)

    with pytest.raises(TypeError, match=msg):
        PeriodArray._from_sequence(list(arr.asi8), dtype=arr.dtype)


def test_from_td64nat_sequence_raises():
    # GH#44507
    td = pd.NaT.to_numpy("m8[ns]")

    dtype = pd.period_range("2005-01-01", periods=3, freq="D").dtype

    arr = np.array([None], dtype=object)
    arr[0] = td

    msg = "Value must be Period, string, integer, or datetime"
    with pytest.raises(ValueError, match=msg):
        PeriodArray._from_sequence(arr, dtype=dtype)

    with pytest.raises(ValueError, match=msg):
        pd.PeriodIndex(arr, dtype=dtype)
    with pytest.raises(ValueError, match=msg):
        pd.Index(arr, dtype=dtype)
    with pytest.raises(ValueError, match=msg):
        pd.array(arr, dtype=dtype)
    with pytest.raises(ValueError, match=msg):
        pd.Series(arr, dtype=dtype)
    with pytest.raises(ValueError, match=msg):
        pd.DataFrame(arr, dtype=dtype)


def test_freq_deprecated():
    # GH#52462
    data = np.arange(5).astype(np.int64)
    msg = "The 'freq' keyword in the PeriodArray constructor is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = PeriodArray(data, freq="M")

    expected = PeriodArray(data, dtype="period[M]")
    tm.assert_equal(res, expected)


def test_period_array_from_datetime64():
    arr = np.array(
        ["2020-01-01T00:00:00", "2020-02-02T00:00:00"], dtype="datetime64[ns]"
    )
    result = PeriodArray._from_datetime64(arr, freq=MonthEnd(2))

    expected = period_array(["2020-01-01", "2020-02-01"], freq=MonthEnd(2))
    tm.assert_period_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\config\test_localization.py ===
import codecs
import locale
import os

import pytest

from pandas._config.localization import (
    can_set_locale,
    get_locales,
    set_locale,
)

from pandas.compat import ISMUSL

import pandas as pd

_all_locales = get_locales()
_current_locale = locale.setlocale(locale.LC_ALL)  # getlocale() is wrong, see GH#46595

# Don't run any of these tests if we have no locales.
pytestmark = pytest.mark.skipif(not _all_locales, reason="Need locales")

_skip_if_only_one_locale = pytest.mark.skipif(
    len(_all_locales) <= 1, reason="Need multiple locales for meaningful test"
)


def _get_current_locale(lc_var: int = locale.LC_ALL) -> str:
    # getlocale is not always compliant with setlocale, use setlocale. GH#46595
    return locale.setlocale(lc_var)


@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_current_locale(lc_var):
    # Can set the current locale
    before_locale = _get_current_locale(lc_var)
    assert can_set_locale(before_locale, lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_locale_valid_set(lc_var):
    # Can set the default locale.
    before_locale = _get_current_locale(lc_var)
    assert can_set_locale("", lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize(
    "lc_var",
    (
        locale.LC_ALL,
        locale.LC_CTYPE,
        pytest.param(
            locale.LC_TIME,
            marks=pytest.mark.skipif(
                ISMUSL, reason="MUSL allows setting invalid LC_TIME."
            ),
        ),
    ),
)
def test_can_set_locale_invalid_set(lc_var):
    # Cannot set an invalid locale.
    before_locale = _get_current_locale(lc_var)
    assert not can_set_locale("non-existent_locale", lc_var=lc_var)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


@pytest.mark.parametrize(
    "lang,enc",
    [
        ("it_CH", "UTF-8"),
        ("en_US", "ascii"),
        ("zh_CN", "GB2312"),
        ("it_IT", "ISO-8859-1"),
    ],
)
@pytest.mark.parametrize("lc_var", (locale.LC_ALL, locale.LC_CTYPE, locale.LC_TIME))
def test_can_set_locale_no_leak(lang, enc, lc_var):
    # Test that can_set_locale does not leak even when returning False. See GH#46595
    before_locale = _get_current_locale(lc_var)
    can_set_locale((lang, enc), locale.LC_ALL)
    after_locale = _get_current_locale(lc_var)
    assert before_locale == after_locale


def test_can_set_locale_invalid_get(monkeypatch):
    # see GH#22129
    # In some cases, an invalid locale can be set,
    #  but a subsequent getlocale() raises a ValueError.

    def mock_get_locale():
        raise ValueError()

    with monkeypatch.context() as m:
        m.setattr(locale, "getlocale", mock_get_locale)
        assert not can_set_locale("")


def test_get_locales_at_least_one():
    # see GH#9744
    assert len(_all_locales) > 0


@_skip_if_only_one_locale
def test_get_locales_prefix():
    first_locale = _all_locales[0]
    assert len(get_locales(prefix=first_locale[:2])) > 0


@_skip_if_only_one_locale
@pytest.mark.parametrize(
    "lang,enc",
    [
        ("it_CH", "UTF-8"),
        ("en_US", "ascii"),
        ("zh_CN", "GB2312"),
        ("it_IT", "ISO-8859-1"),
    ],
)
def test_set_locale(lang, enc):
    before_locale = _get_current_locale()

    enc = codecs.lookup(enc).name
    new_locale = lang, enc

    if not can_set_locale(new_locale):
        msg = "unsupported locale setting"

        with pytest.raises(locale.Error, match=msg):
            with set_locale(new_locale):
                pass
    else:
        with set_locale(new_locale) as normalized_locale:
            new_lang, new_enc = normalized_locale.split(".")
            new_enc = codecs.lookup(enc).name

            normalized_locale = new_lang, new_enc
            assert normalized_locale == new_locale

    # Once we exit the "with" statement, locale should be back to what it was.
    after_locale = _get_current_locale()
    assert before_locale == after_locale


def test_encoding_detected():
    system_locale = os.environ.get("LC_ALL")
    system_encoding = system_locale.split(".")[-1] if system_locale else "utf-8"

    assert (
        codecs.lookup(pd.options.display.encoding).name
        == codecs.lookup(system_encoding).name
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_setitem.py ===
import numpy as np

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

# -----------------------------------------------------------------------------
# Copy/view behaviour for the values that are set in a DataFrame


def test_set_column_with_array():
    # Case: setting an array as a new column (df[col] = arr) copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = np.array([1, 2, 3], dtype="int64")

    df["c"] = arr

    # the array data is copied
    assert not np.shares_memory(get_array(df, "c"), arr)
    # and thus modifying the array does not modify the DataFrame
    arr[0] = 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_series(using_copy_on_write):
    # Case: setting a series as a new column (df[col] = s) copies that data
    # (with delayed copy with CoW)
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = Series([1, 2, 3])

    df["c"] = ser

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "c"), get_array(ser))
    else:
        # the series data is copied
        assert not np.shares_memory(get_array(df, "c"), get_array(ser))

    # and modifying the series does not modify the DataFrame
    ser.iloc[0] = 0
    assert ser.iloc[0] == 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_index(using_copy_on_write):
    # Case: setting an index as a new column (df[col] = idx) copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    idx = Index([1, 2, 3])

    df["c"] = idx

    # the index data is copied
    assert not np.shares_memory(get_array(df, "c"), idx.values)

    idx = RangeIndex(1, 4)
    arr = idx.values

    df["d"] = idx

    assert not np.shares_memory(get_array(df, "d"), arr)


def test_set_columns_with_dataframe(using_copy_on_write):
    # Case: setting a DataFrame as new columns copies that data
    # (with delayed copy with CoW)
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})

    df[["c", "d"]] = df2

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    else:
        # the data is copied
        assert not np.shares_memory(get_array(df, "c"), get_array(df2, "c"))

    # and modifying the set DataFrame does not modify the original DataFrame
    df2.iloc[0, 0] = 0
    tm.assert_series_equal(df["c"], Series([7, 8, 9], name="c"))


def test_setitem_series_no_copy(using_copy_on_write):
    # Case: setting a Series as column into a DataFrame can delay copying that data
    df = DataFrame({"a": [1, 2, 3]})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    # adding a new column
    df["b"] = rhs
    if using_copy_on_write:
        assert np.shares_memory(get_array(rhs), get_array(df, "b"))

    df.iloc[0, 1] = 100
    tm.assert_series_equal(rhs, rhs_orig)


def test_setitem_series_no_copy_single_block(using_copy_on_write):
    # Overwriting an existing column that is a single block
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    df["a"] = rhs
    if using_copy_on_write:
        assert np.shares_memory(get_array(rhs), get_array(df, "a"))

    df.iloc[0, 0] = 100
    tm.assert_series_equal(rhs, rhs_orig)


def test_setitem_series_no_copy_split_block(using_copy_on_write):
    # Overwriting an existing column that is part of a larger block
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    rhs = Series([4, 5, 6])
    rhs_orig = rhs.copy()

    df["b"] = rhs
    if using_copy_on_write:
        assert np.shares_memory(get_array(rhs), get_array(df, "b"))

    df.iloc[0, 1] = 100
    tm.assert_series_equal(rhs, rhs_orig)


def test_setitem_series_column_midx_broadcasting(using_copy_on_write):
    # Setting a Series to multiple columns will repeat the data
    # (currently copying the data eagerly)
    df = DataFrame(
        [[1, 2, 3], [3, 4, 5]],
        columns=MultiIndex.from_arrays([["a", "a", "b"], [1, 2, 3]]),
    )
    rhs = Series([10, 11])
    df["a"] = rhs
    assert not np.shares_memory(get_array(rhs), df._get_column_array(0))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)


def test_set_column_with_inplace_operator(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # this should not raise any warning
    with tm.assert_produces_warning(None):
        df["a"] += 1

    # when it is not in a chain, then it should produce a warning
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = df["a"]
    with tm.assert_cow_warning(warn_copy_on_write):
        ser += 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_astype.py ===
import numpy as np
import pytest

from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    NaT,
    Period,
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestPeriodIndexAsType:
    @pytest.mark.parametrize("dtype", [float, "timedelta64", "timedelta64[ns]"])
    def test_astype_raises(self, dtype):
        # GH#13149, GH#13209
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.nan], freq="D")
        msg = "Cannot cast PeriodIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            idx.astype(dtype)

    def test_astype_conversion(self, using_infer_string):
        # GH#13149, GH#13209
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.nan], freq="D", name="idx")

        result = idx.astype(object)
        expected = Index(
            [Period("2016-05-16", freq="D")] + [Period(NaT, freq="D")] * 3,
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(np.int64)
        expected = Index(
            [16937] + [-9223372036854775808] * 3, dtype=np.int64, name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.astype(str)
        if using_infer_string:
            expected = Index(
                [str(x) if x is not NaT else None for x in idx], name="idx", dtype="str"
            )
        else:
            expected = Index([str(x) for x in idx], name="idx", dtype=object)
        tm.assert_index_equal(result, expected)

        idx = period_range("1990", "2009", freq="Y", name="idx")
        result = idx.astype("i8")
        tm.assert_index_equal(result, Index(idx.asi8, name="idx"))
        tm.assert_numpy_array_equal(result.values, idx.asi8)

    def test_astype_uint(self):
        arr = period_range("2000", periods=2, name="idx")

        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint64")
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype("uint32")

    def test_astype_object(self):
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([Period("2011-01", freq="M"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        exp = np.array([Period("2011-01-01", freq="D"), NaT], dtype=object)
        idx = PeriodIndex(["2011-01-01", NaT], freq="D")
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

    # TODO: de-duplicate this version (from test_ops) with the one above
    # (from test_period)
    def test_astype_object2(self):
        idx = period_range(start="2013-01-01", periods=4, freq="M", name="idx")
        expected_list = [
            Period("2013-01-31", freq="M"),
            Period("2013-02-28", freq="M"),
            Period("2013-03-31", freq="M"),
            Period("2013-04-30", freq="M"),
        ]
        expected = Index(expected_list, dtype=object, name="idx")
        result = idx.astype(object)
        assert isinstance(result, Index)
        assert result.dtype == object
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert idx.tolist() == expected_list

        idx = PeriodIndex(
            ["2013-01-01", "2013-01-02", "NaT", "2013-01-04"], freq="D", name="idx"
        )
        expected_list = [
            Period("2013-01-01", freq="D"),
            Period("2013-01-02", freq="D"),
            Period("NaT", freq="D"),
            Period("2013-01-04", freq="D"),
        ]
        expected = Index(expected_list, dtype=object, name="idx")
        result = idx.astype(object)
        assert isinstance(result, Index)
        assert result.dtype == object
        tm.assert_index_equal(result, expected)
        for i in [0, 1, 3]:
            assert result[i] == expected[i]
        assert result[2] is NaT
        assert result.name == expected.name

        result_list = idx.tolist()
        for i in [0, 1, 3]:
            assert result_list[i] == expected_list[i]
        assert result_list[2] is NaT

    def test_astype_category(self):
        obj = period_range("2000", periods=2, name="idx")
        result = obj.astype("category")
        expected = CategoricalIndex(
            [Period("2000-01-01", freq="D"), Period("2000-01-02", freq="D")], name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = obj._data.astype("category")
        expected = expected.values
        tm.assert_categorical_equal(result, expected)

    def test_astype_array_fallback(self):
        obj = period_range("2000", periods=2, name="idx")
        result = obj.astype(bool)
        expected = Index(np.array([True, True]), name="idx")
        tm.assert_index_equal(result, expected)

        result = obj._data.astype(bool)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

    def test_period_astype_to_timestamp(self, unit):
        # GH#55958
        pi = PeriodIndex(["2011-01", "2011-02", "2011-03"], freq="M")

        exp = DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01"], tz="US/Eastern"
        ).as_unit(unit)
        res = pi.astype(f"datetime64[{unit}, US/Eastern]")
        tm.assert_index_equal(res, exp)
        assert res.freq == exp.freq

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_diagonal.py ===
from sympy.matrices.expressions import MatrixSymbol
from sympy.matrices.expressions.diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector
from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import Identity
from sympy.testing.pytest import raises


n = Symbol('n')
m = Symbol('m')


def test_DiagonalMatrix():
    x = MatrixSymbol('x', n, m)
    D = DiagonalMatrix(x)
    assert D.diagonal_length is None
    assert D.shape == (n, m)

    x = MatrixSymbol('x', n, n)
    D = DiagonalMatrix(x)
    assert D.diagonal_length == n
    assert D.shape == (n, n)
    assert D[1, 2] == 0
    assert D[1, 1] == x[1, 1]
    i = Symbol('i')
    j = Symbol('j')
    x = MatrixSymbol('x', 3, 3)
    ij = DiagonalMatrix(x)[i, j]
    assert ij != 0
    assert ij.subs({i:0, j:0}) == x[0, 0]
    assert ij.subs({i:0, j:1}) == 0
    assert ij.subs({i:1, j:1}) == x[1, 1]
    assert ask(Q.diagonal(D))  # affirm that D is diagonal

    x = MatrixSymbol('x', n, 3)
    D = DiagonalMatrix(x)
    assert D.diagonal_length == 3
    assert D.shape == (n, 3)
    assert D[2, m] == KroneckerDelta(2, m)*x[2, m]
    assert D[3, m] == 0
    raises(IndexError, lambda: D[m, 3])

    x = MatrixSymbol('x', 3, n)
    D = DiagonalMatrix(x)
    assert D.diagonal_length == 3
    assert D.shape == (3, n)
    assert D[m, 2] == KroneckerDelta(m, 2)*x[m, 2]
    assert D[m, 3] == 0
    raises(IndexError, lambda: D[3, m])

    x = MatrixSymbol('x', n, m)
    D = DiagonalMatrix(x)
    assert D.diagonal_length is None
    assert D.shape == (n, m)
    assert D[m, 4] != 0

    x = MatrixSymbol('x', 3, 4)
    assert [DiagonalMatrix(x)[i] for i in range(12)] == [
        x[0, 0], 0, 0, 0, 0, x[1, 1], 0, 0, 0, 0, x[2, 2], 0]

    # shape is retained, issue 12427
    assert (
        DiagonalMatrix(MatrixSymbol('x', 3, 4))*
        DiagonalMatrix(MatrixSymbol('x', 4, 2))).shape == (3, 2)


def test_DiagonalOf():
    x = MatrixSymbol('x', n, n)
    d = DiagonalOf(x)
    assert d.shape == (n, 1)
    assert d.diagonal_length == n
    assert d[2, 0] == d[2] == x[2, 2]

    x = MatrixSymbol('x', n, m)
    d = DiagonalOf(x)
    assert d.shape == (None, 1)
    assert d.diagonal_length is None
    assert d[2, 0] == d[2] == x[2, 2]

    d = DiagonalOf(MatrixSymbol('x', 4, 3))
    assert d.shape == (3, 1)
    d = DiagonalOf(MatrixSymbol('x', n, 3))
    assert d.shape == (3, 1)
    d = DiagonalOf(MatrixSymbol('x', 3, n))
    assert d.shape == (3, 1)
    x = MatrixSymbol('x', n, m)
    assert [DiagonalOf(x)[i] for i in range(4)] ==[
        x[0, 0], x[1, 1], x[2, 2], x[3, 3]]


def test_DiagMatrix():
    x = MatrixSymbol('x', n, 1)
    d = DiagMatrix(x)
    assert d.shape == (n, n)
    assert d[0, 1] == 0
    assert d[0, 0] == x[0, 0]

    a = MatrixSymbol('a', 1, 1)
    d = diagonalize_vector(a)
    assert isinstance(d, MatrixSymbol)
    assert a == d
    assert diagonalize_vector(Identity(3)) == Identity(3)
    assert DiagMatrix(Identity(3)).doit() == Identity(3)
    assert isinstance(DiagMatrix(Identity(3)), DiagMatrix)

    # A diagonal matrix is equal to its transpose:
    assert DiagMatrix(x).T == DiagMatrix(x)
    assert diagonalize_vector(x.T) == DiagMatrix(x)

    dx = DiagMatrix(x)
    assert dx[0, 0] == x[0, 0]
    assert dx[1, 1] == x[1, 0]
    assert dx[0, 1] == 0
    assert dx[0, m] == x[0, 0]*KroneckerDelta(0, m)

    z = MatrixSymbol('z', 1, n)
    dz = DiagMatrix(z)
    assert dz[0, 0] == z[0, 0]
    assert dz[1, 1] == z[0, 1]
    assert dz[0, 1] == 0
    assert dz[0, m] == z[0, m]*KroneckerDelta(0, m)

    v = MatrixSymbol('v', 3, 1)
    dv = DiagMatrix(v)
    assert dv.as_explicit() == Matrix([
        [v[0, 0], 0, 0],
        [0, v[1, 0], 0],
        [0, 0, v[2, 0]],
    ])

    v = MatrixSymbol('v', 1, 3)
    dv = DiagMatrix(v)
    assert dv.as_explicit() == Matrix([
        [v[0, 0], 0, 0],
        [0, v[0, 1], 0],
        [0, 0, v[0, 2]],
    ])

    dv = DiagMatrix(3*v)
    assert dv.args == (3*v,)
    assert dv.doit() == 3*DiagMatrix(v)
    assert isinstance(dv.doit(), MatMul)

    a = MatrixSymbol("a", 3, 1).as_explicit()
    expr = DiagMatrix(a)
    result = Matrix([
        [a[0, 0], 0, 0],
        [0, a[1, 0], 0],
        [0, 0, a[2, 0]],
    ])
    assert expr.doit() == result
    expr = DiagMatrix(a.T)
    assert expr.doit() == result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_normalforms.py ===
from sympy.testing.pytest import raises

from sympy.core.symbol import Symbol
from sympy.polys.matrices.normalforms import (
    invariant_factors,
    smith_normal_form,
    smith_normal_decomp,
    is_smith_normal_form,
    hermite_normal_form,
    _hermite_normal_form,
    _hermite_normal_form_modulo_D
)
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.exceptions import DMDomainError, DMShapeError


def test_is_smith_normal_form():

    snf_examples = [
        DM([[0, 0], [0, 0]], ZZ),
        DM([[1, 0], [0, 0]], ZZ),
        DM([[1, 0], [0, 1]], ZZ),
        DM([[1, 0], [0, 2]], ZZ),
    ]

    non_snf_examples = [
        DM([[0, 1], [0, 0]], ZZ),
        DM([[0, 0], [0, 1]], ZZ),
        DM([[2, 0], [0, 3]], ZZ),
    ]

    for m in snf_examples:
        assert is_smith_normal_form(m) is True

    for m in non_snf_examples:
        assert is_smith_normal_form(m) is False


def test_smith_normal():

    m = DM([
        [12, 6, 4, 8],
        [3, 9, 6, 12],
        [2, 16, 14, 28],
        [20, 10, 10, 20]], ZZ)

    smf = DM([
        [1, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 30, 0],
        [0, 0, 0, 0]], ZZ)

    s = DM([
        [0, 1, -1, 0],
        [1, -4, 0, 0],
        [0, -2, 3, 0],
        [-2, 2, -1, 1]], ZZ)

    t = DM([
        [1, 1, 10, 0],
        [0, -1, -2, 0],
        [0, 1, 3, -2],
        [0, 0, 0, 1]], ZZ)

    assert smith_normal_form(m).to_dense() == smf
    assert smith_normal_decomp(m) == (smf, s, t)
    assert is_smith_normal_form(smf)
    assert smf == s * m * t

    m00 = DomainMatrix.zeros((0, 0), ZZ).to_dense()
    m01 = DomainMatrix.zeros((0, 1), ZZ).to_dense()
    m10 = DomainMatrix.zeros((1, 0), ZZ).to_dense()
    i11 = DM([[1]], ZZ)

    assert smith_normal_form(m00) == m00.to_sparse()
    assert smith_normal_form(m01) == m01.to_sparse()
    assert smith_normal_form(m10) == m10.to_sparse()
    assert smith_normal_form(i11) == i11.to_sparse()

    assert smith_normal_decomp(m00) == (m00, m00, m00)
    assert smith_normal_decomp(m01) == (m01, m00, i11)
    assert smith_normal_decomp(m10) == (m10, i11, m00)
    assert smith_normal_decomp(i11) == (i11, i11, i11)

    x = Symbol('x')
    m = DM([[x-1,  1, -1],
            [  0,  x, -1],
            [  0, -1,  x]], QQ[x])
    dx = m.domain.gens[0]
    assert invariant_factors(m) == (1, dx-1, dx**2-1)

    zr = DomainMatrix([], (0, 2), ZZ)
    zc = DomainMatrix([[], []], (2, 0), ZZ)
    assert smith_normal_form(zr).to_dense() == zr
    assert smith_normal_form(zc).to_dense() == zc

    assert smith_normal_form(DM([[2, 4]], ZZ)).to_dense() == DM([[2, 0]], ZZ)
    assert smith_normal_form(DM([[0, -2]], ZZ)).to_dense() == DM([[2, 0]], ZZ)
    assert smith_normal_form(DM([[0], [-2]], ZZ)).to_dense() == DM([[2], [0]], ZZ)

    assert smith_normal_decomp(DM([[0, -2]], ZZ)) == (
        DM([[2, 0]], ZZ), DM([[-1]], ZZ), DM([[0, 1], [1, 0]], ZZ)
    )
    assert smith_normal_decomp(DM([[0], [-2]], ZZ)) == (
        DM([[2], [0]], ZZ), DM([[0, -1], [1, 0]], ZZ), DM([[1]], ZZ)
    )

    m =   DM([[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0]], ZZ)
    snf = DM([[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 0, 0]], ZZ)
    s = DM([[1, 0, 1], [2, 0, 3], [0, 1, 0]], ZZ)
    t = DM([[1, -2, 0, 0], [0, 0, 0, 1], [-1, 3, 0, 0], [0, 0, 1, 0]], ZZ)

    assert smith_normal_form(m).to_dense() == snf
    assert smith_normal_decomp(m) == (snf, s, t)
    assert is_smith_normal_form(snf)
    assert snf == s * m * t

    raises(ValueError, lambda: smith_normal_form(DM([[1]], ZZ[x])))


def test_hermite_normal():
    m = DM([[2, 7, 17, 29, 41], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]], ZZ)
    hnf = DM([[1, 0, 0], [0, 2, 1], [0, 0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf
    assert hermite_normal_form(m, D=ZZ(2)) == hnf
    assert hermite_normal_form(m, D=ZZ(2), check_rank=True) == hnf

    m = m.transpose()
    hnf = DM([[37, 0, 19], [222, -6, 113], [48, 0, 25], [0, 2, 1], [0, 0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf
    raises(DMShapeError, lambda: _hermite_normal_form_modulo_D(m, ZZ(96)))
    raises(DMDomainError, lambda: _hermite_normal_form_modulo_D(m, QQ(96)))

    m = DM([[8, 28, 68, 116, 164], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]], ZZ)
    hnf = DM([[4, 0, 0], [0, 2, 1], [0, 0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf
    assert hermite_normal_form(m, D=ZZ(8)) == hnf
    assert hermite_normal_form(m, D=ZZ(8), check_rank=True) == hnf

    m = DM([[10, 8, 6, 30, 2], [45, 36, 27, 18, 9], [5, 4, 3, 2, 1]], ZZ)
    hnf = DM([[26, 2], [0, 9], [0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf

    m = DM([[2, 7], [0, 0], [0, 0]], ZZ)
    hnf = DM([[1], [0], [0]], ZZ)
    assert hermite_normal_form(m) == hnf

    m = DM([[-2, 1], [0, 1]], ZZ)
    hnf = DM([[2, 1], [0, 1]], ZZ)
    assert hermite_normal_form(m) == hnf

    m = DomainMatrix([[QQ(1)]], (1, 1), QQ)
    raises(DMDomainError, lambda: hermite_normal_form(m))
    raises(DMDomainError, lambda: _hermite_normal_form(m))
    raises(DMDomainError, lambda: _hermite_normal_form_modulo_D(m, ZZ(1)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_navigablestring.py ===
import pytest

from bs4.element import (
    CData,
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    RubyParenthesisString,
    RubyTextString,
    Script,
    Stylesheet,
    TemplateString,
)

from . import SoupTest


class TestNavigableString(SoupTest):
    def test_text_acquisition_methods(self):
        # These methods are intended for use against Tag, but they
        # work on NavigableString as well,

        s = NavigableString("fee ")
        cdata = CData("fie ")
        comment = Comment("foe ")

        assert "fee " == s.get_text()
        assert "fee " == s.string
        assert "fee" == s.get_text(strip=True)
        assert ["fee "] == list(s.strings)
        assert ["fee"] == list(s.stripped_strings)
        assert ["fee "] == list(s._all_strings())

        assert "fie " == cdata.get_text()
        assert "fie " == cdata.string
        assert "fie" == cdata.get_text(strip=True)
        assert ["fie "] == list(cdata.strings)
        assert ["fie"] == list(cdata.stripped_strings)
        assert ["fie "] == list(cdata._all_strings())

        # Since a Comment isn't normally considered 'text',
        # these methods generally do nothing.
        assert "" == comment.get_text()
        assert [] == list(comment.strings)
        assert [] == list(comment.stripped_strings)
        assert [] == list(comment._all_strings())

        # Unless you specifically say that comments are okay.
        assert "foe" == comment.get_text(strip=True, types=Comment)
        assert "foe " == comment.get_text(types=(Comment, NavigableString))

    def test_string_has_immutable_name_property(self):
        # string.name is defined as None and can't be modified
        string = self.soup("s").string
        assert None is string.name
        with pytest.raises(AttributeError):
            string.name = "foo"

    def test_string_detects_attribute_access_attempt(self):
        string = self.soup("the string").string

        # Try to access an HTML attribute of a NavigableString and get a helpful exception.
        with pytest.raises(TypeError) as e:
            string["attr"]
        assert str(e.value) == "string indices must be integers, not 'str'. Are you treating a NavigableString like a Tag?"

        # Normal string access works.
        assert string[2] == "e"
        assert string[2:5] == "e s"

class TestNavigableStringSubclasses(SoupTest):
    def test_cdata(self):
        # None of the current builders turn CDATA sections into CData
        # objects, but you can create them manually.
        soup = self.soup("")
        cdata = CData("foo")
        soup.insert(1, cdata)
        assert str(soup) == "<![CDATA[foo]]>"
        assert soup.find(string="foo") == "foo"
        assert soup.contents[0] == "foo"

    def test_cdata_is_never_formatted(self):
        """Text inside a CData object is passed into the formatter.

        But the return value is ignored.
        """

        self.count = 0

        def increment(*args):
            self.count += 1
            return "BITTER FAILURE"

        soup = self.soup("")
        cdata = CData("<><><>")
        soup.insert(1, cdata)
        assert b"<![CDATA[<><><>]]>" == soup.encode(formatter=increment)
        assert 1 == self.count

    def test_doctype_ends_in_newline(self):
        # Unlike other NavigableString subclasses, a DOCTYPE always ends
        # in a newline.
        doctype = Doctype("foo")
        soup = self.soup("")
        soup.insert(1, doctype)
        assert soup.encode() == b"<!DOCTYPE foo>\n"

    def test_declaration(self):
        d = Declaration("foo")
        assert "<?foo?>" == d.output_ready()

    def test_default_string_containers(self):
        # In some cases, we use different NavigableString subclasses for
        # the same text in different tags.
        soup = self.soup("<div>text</div><script>text</script><style>text</style>")
        assert [NavigableString, Script, Stylesheet] == [
            x.__class__ for x in soup.find_all(string=True)
        ]

        # The TemplateString is a little unusual because it's generally found
        # _inside_ children of a <template> element, not a direct child of the
        # <template> element.
        soup = self.soup(
            "<template>Some text<p>In a tag</p></template>Some text outside"
        )
        assert all(
            isinstance(x, TemplateString)
            for x in soup.template._all_strings(types=None)
        )

        # Once the <template> tag closed, we went back to using
        # NavigableString.
        outside = soup.template.next_sibling
        assert isinstance(outside, NavigableString)
        assert not isinstance(outside, TemplateString)

        # The TemplateString is also unusual because it can contain
        # NavigableString subclasses of _other_ types, such as
        # Comment.
        markup = b"<template>Some text<p>In a tag</p><!--with a comment--></template>"
        soup = self.soup(markup)
        assert markup == soup.template.encode("utf8")

    def test_ruby_strings(self):
        markup = "<ruby>漢 <rp>(</rp><rt>kan</rt><rp>)</rp> 字 <rp>(</rp><rt>ji</rt><rp>)</rp></ruby>"
        soup = self.soup(markup)
        assert isinstance(soup.rp.string, RubyParenthesisString)
        assert isinstance(soup.rt.string, RubyTextString)

        # Just as a demo, here's what this means for get_text usage.
        assert "漢字" == soup.get_text(strip=True)
        assert "漢(kan)字(ji)" == soup.get_text(
            strip=True, types=(NavigableString, RubyTextString, RubyParenthesisString)
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\common.py ===
"""
Assertion helpers for arithmetic tests.
"""
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    array,
)
import pandas._testing as tm
from pandas.core.arrays import (
    BooleanArray,
    NumpyExtensionArray,
)


def assert_cannot_add(left, right, msg="cannot add"):
    """
    Helper to assert that left and right cannot be added.

    Parameters
    ----------
    left : object
    right : object
    msg : str, default "cannot add"
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left


def assert_invalid_addsub_type(left, right, msg=None):
    """
    Helper to assert that left and right can be neither added nor subtracted.

    Parameters
    ----------
    left : object
    right : object
    msg : str or None, default None
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left
    with pytest.raises(TypeError, match=msg):
        left - right
    with pytest.raises(TypeError, match=msg):
        right - left


def get_upcast_box(left, right, is_cmp: bool = False):
    """
    Get the box to use for 'expected' in an arithmetic or comparison operation.

    Parameters
    left : Any
    right : Any
    is_cmp : bool, default False
        Whether the operation is a comparison method.
    """

    if isinstance(left, DataFrame) or isinstance(right, DataFrame):
        return DataFrame
    if isinstance(left, Series) or isinstance(right, Series):
        if is_cmp and isinstance(left, Index):
            # Index does not defer for comparisons
            return np.array
        return Series
    if isinstance(left, Index) or isinstance(right, Index):
        if is_cmp:
            return np.array
        return Index
    return tm.to_array


def assert_invalid_comparison(left, right, box):
    """
    Assert that comparison operations with mismatched types behave correctly.

    Parameters
    ----------
    left : np.ndarray, ExtensionArray, Index, or Series
    right : object
    box : {pd.DataFrame, pd.Series, pd.Index, pd.array, tm.to_array}
    """
    # Not for tznaive-tzaware comparison

    # Note: not quite the same as how we do this for tm.box_expected
    xbox = box if box not in [Index, array] else np.array

    def xbox2(x):
        # Eventually we'd like this to be tighter, but for now we'll
        #  just exclude NumpyExtensionArray[bool]
        if isinstance(x, NumpyExtensionArray):
            return x._ndarray
        if isinstance(x, BooleanArray):
            # NB: we are assuming no pd.NAs for now
            return x.astype(bool)
        return x

    # rev_box: box to use for reversed comparisons
    rev_box = xbox
    if isinstance(right, Index) and isinstance(left, Series):
        rev_box = np.array

    result = xbox2(left == right)
    expected = xbox(np.zeros(result.shape, dtype=np.bool_))

    tm.assert_equal(result, expected)

    result = xbox2(right == left)
    tm.assert_equal(result, rev_box(expected))

    result = xbox2(left != right)
    tm.assert_equal(result, ~expected)

    result = xbox2(right != left)
    tm.assert_equal(result, rev_box(~expected))

    msg = "|".join(
        [
            "Invalid comparison between",
            "Cannot compare type",
            "not supported between",
            "invalid type promotion",
            (
                # GH#36706 npdev 1.20.0 2020-09-28
                r"The DTypes <class 'numpy.dtype\[datetime64\]'> and "
                r"<class 'numpy.dtype\[int64\]'> do not have a common DType. "
                "For example they cannot be stored in a single array unless the "
                "dtype is `object`."
            ),
        ]
    )
    with pytest.raises(TypeError, match=msg):
        left < right
    with pytest.raises(TypeError, match=msg):
        left <= right
    with pytest.raises(TypeError, match=msg):
        left > right
    with pytest.raises(TypeError, match=msg):
        left >= right
    with pytest.raises(TypeError, match=msg):
        right < left
    with pytest.raises(TypeError, match=msg):
        right <= left
    with pytest.raises(TypeError, match=msg):
        right > left
    with pytest.raises(TypeError, match=msg):
        right >= left

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_astype.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DatetimeIndex,
    Interval,
    NaT,
    Period,
    Timestamp,
    array,
    to_datetime,
)
import pandas._testing as tm


class TestAstype:
    @pytest.mark.parametrize("cls", [Categorical, CategoricalIndex])
    @pytest.mark.parametrize("values", [[1, np.nan], [Timestamp("2000"), NaT]])
    def test_astype_nan_to_int(self, cls, values):
        # GH#28406
        obj = cls(values)

        msg = "Cannot (cast|convert)"
        with pytest.raises((ValueError, TypeError), match=msg):
            obj.astype(int)

    @pytest.mark.parametrize(
        "expected",
        [
            array(["2019", "2020"], dtype="datetime64[ns, UTC]"),
            array([0, 0], dtype="timedelta64[ns]"),
            array([Period("2019"), Period("2020")], dtype="period[Y-DEC]"),
            array([Interval(0, 1), Interval(1, 2)], dtype="interval"),
            array([1, np.nan], dtype="Int64"),
        ],
    )
    def test_astype_category_to_extension_dtype(self, expected):
        # GH#28668
        result = expected.astype("category").astype(expected.dtype)

        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (
                "datetime64[ns]",
                np.array(["2015-01-01T00:00:00.000000000"], dtype="datetime64[ns]"),
            ),
            (
                "datetime64[ns, MET]",
                DatetimeIndex([Timestamp("2015-01-01 00:00:00+0100", tz="MET")]).array,
            ),
        ],
    )
    def test_astype_to_datetime64(self, dtype, expected):
        # GH#28448
        result = Categorical(["2015-01-01"]).astype(dtype)
        assert result == expected

    def test_astype_str_int_categories_to_nullable_int(self):
        # GH#39616
        dtype = CategoricalDtype([str(i) for i in range(5)])
        codes = np.random.default_rng(2).integers(5, size=20)
        arr = Categorical.from_codes(codes, dtype=dtype)

        res = arr.astype("Int64")
        expected = array(codes, dtype="Int64")
        tm.assert_extension_array_equal(res, expected)

    def test_astype_str_int_categories_to_nullable_float(self):
        # GH#39616
        dtype = CategoricalDtype([str(i / 2) for i in range(5)])
        codes = np.random.default_rng(2).integers(5, size=20)
        arr = Categorical.from_codes(codes, dtype=dtype)

        res = arr.astype("Float64")
        expected = array(codes, dtype="Float64") / 2
        tm.assert_extension_array_equal(res, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_astype(self, ordered):
        # string
        cat = Categorical(list("abbaaccc"), ordered=ordered)
        result = cat.astype(object)
        expected = np.array(cat)
        tm.assert_numpy_array_equal(result, expected)

        msg = r"Cannot cast object|str dtype to float64"
        with pytest.raises(ValueError, match=msg):
            cat.astype(float)

        # numeric
        cat = Categorical([0, 1, 2, 2, 1, 0, 1, 0, 2], ordered=ordered)
        result = cat.astype(object)
        expected = np.array(cat, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = cat.astype(int)
        expected = np.array(cat, dtype="int")
        tm.assert_numpy_array_equal(result, expected)

        result = cat.astype(float)
        expected = np.array(cat, dtype=float)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype_ordered", [True, False])
    @pytest.mark.parametrize("cat_ordered", [True, False])
    def test_astype_category(self, dtype_ordered, cat_ordered):
        # GH#10696/GH#18593
        data = list("abcaacbab")
        cat = Categorical(data, categories=list("bac"), ordered=cat_ordered)

        # standard categories
        dtype = CategoricalDtype(ordered=dtype_ordered)
        result = cat.astype(dtype)
        expected = Categorical(data, categories=cat.categories, ordered=dtype_ordered)
        tm.assert_categorical_equal(result, expected)

        # non-standard categories
        dtype = CategoricalDtype(list("adc"), dtype_ordered)
        result = cat.astype(dtype)
        expected = Categorical(data, dtype=dtype)
        tm.assert_categorical_equal(result, expected)

        if dtype_ordered is False:
            # dtype='category' can't specify ordered, so only test once
            result = cat.astype("category")
            expected = cat
            tm.assert_categorical_equal(result, expected)

    def test_astype_object_datetime_categories(self):
        # GH#40754
        cat = Categorical(to_datetime(["2021-03-27", NaT]))
        result = cat.astype(object)
        expected = np.array([Timestamp("2021-03-27 00:00:00"), NaT], dtype="object")
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_object_timestamp_categories(self):
        # GH#18024
        cat = Categorical([Timestamp("2014-01-01")])
        result = cat.astype(object)
        expected = np.array([Timestamp("2014-01-01 00:00:00")], dtype="object")
        tm.assert_numpy_array_equal(result, expected)

    def test_astype_category_readonly_mask_values(self):
        # GH#53658
        arr = array([0, 1, 2], dtype="Int64")
        arr._mask.flags["WRITEABLE"] = False
        result = arr.astype("category")
        expected = array([0, 1, 2], dtype="Int64").astype("category")
        tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_dot.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class DotSharedTests:
    @pytest.fixture
    def obj(self):
        raise NotImplementedError

    @pytest.fixture
    def other(self) -> DataFrame:
        """
        other is a DataFrame that is indexed so that obj.dot(other) is valid
        """
        raise NotImplementedError

    @pytest.fixture
    def expected(self, obj, other) -> DataFrame:
        """
        The expected result of obj.dot(other)
        """
        raise NotImplementedError

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        raise NotImplementedError

    def test_dot_equiv_values_dot(self, obj, other, expected):
        # `expected` is constructed from obj.values.dot(other.values)
        result = obj.dot(other)
        tm.assert_equal(result, expected)

    def test_dot_2d_ndarray(self, obj, other, expected):
        # Check ndarray argument; in this case we get matching values,
        #  but index/columns may not match
        result = obj.dot(other.values)
        assert np.all(result == expected.values)

    def test_dot_1d_ndarray(self, obj, expected):
        # can pass correct-length array
        row = obj.iloc[0] if obj.ndim == 2 else obj

        result = obj.dot(row.values)
        expected = obj.dot(row)
        self.reduced_dim_assert(result, expected)

    def test_dot_series(self, obj, other, expected):
        # Check series argument
        result = obj.dot(other["1"])
        self.reduced_dim_assert(result, expected["1"])

    def test_dot_series_alignment(self, obj, other, expected):
        result = obj.dot(other.iloc[::-1]["1"])
        self.reduced_dim_assert(result, expected["1"])

    def test_dot_aligns(self, obj, other, expected):
        # Check index alignment
        other2 = other.iloc[::-1]
        result = obj.dot(other2)
        tm.assert_equal(result, expected)

    def test_dot_shape_mismatch(self, obj):
        msg = "Dot product shape mismatch"
        # exception raised is of type Exception
        with pytest.raises(Exception, match=msg):
            obj.dot(obj.values[:3])

    def test_dot_misaligned(self, obj, other):
        msg = "matrices are not aligned"
        with pytest.raises(ValueError, match=msg):
            obj.dot(other.T)


class TestSeriesDot(DotSharedTests):
    @pytest.fixture
    def obj(self):
        return Series(
            np.random.default_rng(2).standard_normal(4), index=["p", "q", "r", "s"]
        )

    @pytest.fixture
    def other(self):
        return DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["1", "2", "3"],
            columns=["p", "q", "r", "s"],
        ).T

    @pytest.fixture
    def expected(self, obj, other):
        return Series(np.dot(obj.values, other.values), index=other.columns)

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        tm.assert_almost_equal(result, expected)


class TestDataFrameDot(DotSharedTests):
    @pytest.fixture
    def obj(self):
        return DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["a", "b", "c"],
            columns=["p", "q", "r", "s"],
        )

    @pytest.fixture
    def other(self):
        return DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            index=["p", "q", "r", "s"],
            columns=["1", "2"],
        )

    @pytest.fixture
    def expected(self, obj, other):
        return DataFrame(
            np.dot(obj.values, other.values), index=obj.index, columns=other.columns
        )

    @classmethod
    def reduced_dim_assert(cls, result, expected):
        """
        Assertion about results with 1 fewer dimension that self.obj
        """
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name is None


@pytest.mark.parametrize(
    "dtype,exp_dtype",
    [("Float32", "Float64"), ("Int16", "Int32"), ("float[pyarrow]", "double[pyarrow]")],
)
def test_arrow_dtype(dtype, exp_dtype):
    pytest.importorskip("pyarrow")

    cols = ["a", "b"]
    df_a = DataFrame([[1, 2], [3, 4], [5, 6]], columns=cols, dtype="int32")
    df_b = DataFrame([[1, 0], [0, 1]], index=cols, dtype=dtype)
    result = df_a.dot(df_b)
    expected = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=exp_dtype)

    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_groupby.py ===
""" Test cases for GroupBy.plot """


import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_legend_labels,
)

pytest.importorskip("matplotlib")


class TestDataFrameGroupByPlots:
    def test_series_groupby_plotting_nominally_works(self):
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)

        weight.groupby(gender).plot()

    def test_series_groupby_plotting_nominally_works_hist(self):
        n = 10
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        height.groupby(gender).hist()

    def test_series_groupby_plotting_nominally_works_alpha(self):
        n = 10
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        # Regression test for GH8733
        height.groupby(gender).plot(alpha=0.5)

    def test_plotting_with_float_index_works(self):
        # GH 7025
        df = DataFrame(
            {
                "def": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "val": np.random.default_rng(2).standard_normal(9),
            },
            index=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        )

        df.groupby("def")["val"].plot()

    def test_plotting_with_float_index_works_apply(self):
        # GH 7025
        df = DataFrame(
            {
                "def": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "val": np.random.default_rng(2).standard_normal(9),
            },
            index=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        )
        df.groupby("def")["val"].apply(lambda x: x.plot())

    def test_hist_single_row(self):
        # GH10214
        bins = np.arange(80, 100 + 2, 1)
        df = DataFrame({"Name": ["AAA", "BBB"], "ByCol": [1, 2], "Mark": [85, 89]})
        df["Mark"].hist(by=df["ByCol"], bins=bins)

    def test_hist_single_row_single_bycol(self):
        # GH10214
        bins = np.arange(80, 100 + 2, 1)
        df = DataFrame({"Name": ["AAA"], "ByCol": [1], "Mark": [85]})
        df["Mark"].hist(by=df["ByCol"], bins=bins)

    def test_plot_submethod_works(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})
        df.groupby("z").plot.scatter("x", "y")

    def test_plot_submethod_works_line(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})
        df.groupby("z")["x"].plot.line()

    def test_plot_kwargs(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})

        res = df.groupby("z").plot(kind="scatter", x="x", y="y")
        # check that a scatter plot is effectively plotted: the axes should
        # contain a PathCollection from the scatter plot (GH11805)
        assert len(res["a"].collections) == 1

    def test_plot_kwargs_scatter(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})
        res = df.groupby("z").plot.scatter(x="x", y="y")
        assert len(res["a"].collections) == 1

    @pytest.mark.parametrize("column, expected_axes_num", [(None, 2), ("b", 1)])
    def test_groupby_hist_frame_with_legend(self, column, expected_axes_num):
        # GH 6279 - DataFrameGroupBy histogram can have a legend
        expected_layout = (1, expected_axes_num)
        expected_labels = column or [["a"], ["b"]]

        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        for axes in g.hist(legend=True, column=column):
            _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
            for ax, expected_label in zip(axes[0], expected_labels):
                _check_legend_labels(ax, expected_label)

    @pytest.mark.parametrize("column", [None, "b"])
    def test_groupby_hist_frame_with_legend_raises(self, column):
        # GH 6279 - DataFrameGroupBy histogram with legend and label raises
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            g.hist(legend=True, column=column, label="d")

    def test_groupby_hist_series_with_legend(self):
        # GH 6279 - SeriesGroupBy histogram can have a legend
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        for ax in g["a"].hist(legend=True):
            _check_axes_shape(ax, axes_num=1, layout=(1, 1))
            _check_legend_labels(ax, ["1", "2"])

    def test_groupby_hist_series_with_legend_raises(self):
        # GH 6279 - SeriesGroupBy histogram with legend and label raises
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            g.hist(legend=True, label="d")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_count_ops.py ===
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import (Derivative, Function, count_ops)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Rel)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Equivalent, ITE, Implies, Nand,
    Nor, Not, Or, Xor)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.core.containers import Tuple

x, y, z = symbols('x,y,z')
a, b, c = symbols('a,b,c')

def test_count_ops_non_visual():
    def count(val):
        return count_ops(val, visual=False)
    assert count(x) == 0
    assert count(x) is not S.Zero
    assert count(x + y) == 1
    assert count(x + y) is not S.One
    assert count(x + y*x + 2*y) == 4
    assert count({x + y: x}) == 1
    assert count({x + y: S(2) + x}) is not S.One
    assert count(x < y) == 1
    assert count(Or(x,y)) == 1
    assert count(And(x,y)) == 1
    assert count(Not(x)) == 1
    assert count(Nor(x,y)) == 2
    assert count(Nand(x,y)) == 2
    assert count(Xor(x,y)) == 1
    assert count(Implies(x,y)) == 1
    assert count(Equivalent(x,y)) == 1
    assert count(ITE(x,y,z)) == 1
    assert count(ITE(True,x,y)) == 0


def test_count_ops_visual():
    ADD, MUL, POW, SIN, COS, EXP, AND, D, G, M = symbols(
        'Add Mul Pow sin cos exp And Derivative Integral Sum'.upper())
    DIV, SUB, NEG = symbols('DIV SUB NEG')
    LT, LE, GT, GE, EQ, NE = symbols('LT LE GT GE EQ NE')
    NOT, OR, AND, XOR, IMPLIES, EQUIVALENT, _ITE, BASIC, TUPLE = symbols(
        'Not Or And Xor Implies Equivalent ITE Basic Tuple'.upper())

    def count(val):
        return count_ops(val, visual=True)

    assert count(7) is S.Zero
    assert count(S(7)) is S.Zero
    assert count(-1) == NEG
    assert count(-2) == NEG
    assert count(S(2)/3) == DIV
    assert count(Rational(2, 3)) == DIV
    assert count(pi/3) == DIV
    assert count(-pi/3) == DIV + NEG
    assert count(I - 1) == SUB
    assert count(1 - I) == SUB
    assert count(1 - 2*I) == SUB + MUL

    assert count(x) is S.Zero
    assert count(-x) == NEG
    assert count(-2*x/3) == NEG + DIV + MUL
    assert count(Rational(-2, 3)*x) == NEG + DIV + MUL
    assert count(1/x) == DIV
    assert count(1/(x*y)) == DIV + MUL
    assert count(-1/x) == NEG + DIV
    assert count(-2/x) == NEG + DIV
    assert count(x/y) == DIV
    assert count(-x/y) == NEG + DIV

    assert count(x**2) == POW
    assert count(-x**2) == POW + NEG
    assert count(-2*x**2) == POW + MUL + NEG

    assert count(x + pi/3) == ADD + DIV
    assert count(x + S.One/3) == ADD + DIV
    assert count(x + Rational(1, 3)) == ADD + DIV
    assert count(x + y) == ADD
    assert count(x - y) == SUB
    assert count(y - x) == SUB
    assert count(-1/(x - y)) == DIV + NEG + SUB
    assert count(-1/(y - x)) == DIV + NEG + SUB
    assert count(1 + x**y) == ADD + POW
    assert count(1 + x + y) == 2*ADD
    assert count(1 + x + y + z) == 3*ADD
    assert count(1 + x**y + 2*x*y + y**2) == 3*ADD + 2*POW + 2*MUL
    assert count(2*z + y + x + 1) == 3*ADD + MUL
    assert count(2*z + y**17 + x + 1) == 3*ADD + MUL + POW
    assert count(2*z + y**17 + x + sin(x)) == 3*ADD + POW + MUL + SIN
    assert count(2*z + y**17 + x + sin(x**2)) == 3*ADD + MUL + 2*POW + SIN
    assert count(2*z + y**17 + x + sin(
        x**2) + exp(cos(x))) == 4*ADD + MUL + 2*POW + EXP + COS + SIN

    assert count(Derivative(x, x)) == D
    assert count(Integral(x, x) + 2*x/(1 + x)) == G + DIV + MUL + 2*ADD
    assert count(Sum(x, (x, 1, x + 1)) + 2*x/(1 + x)) == M + DIV + MUL + 3*ADD
    assert count(Basic()) is S.Zero

    assert count({x + 1: sin(x)}) == ADD + SIN
    assert count([x + 1, sin(x) + y, None]) == ADD + SIN + ADD
    assert count({x + 1: sin(x), y: cos(x) + 1}) == SIN + COS + 2*ADD
    assert count({}) is S.Zero
    assert count([x + 1, sin(x)*y, None]) == SIN + ADD + MUL
    assert count([]) is S.Zero

    assert count(Basic()) == 0
    assert count(Basic(Basic(),Basic(x,x+y))) == ADD + 2*BASIC
    assert count(Basic(x, x + y)) == ADD + BASIC
    assert [count(Rel(x, y, op)) for op in '< <= > >= == <> !='.split()
        ] == [LT, LE, GT, GE, EQ, NE, NE]
    assert count(Or(x, y)) == OR
    assert count(And(x, y)) == AND
    assert count(Or(x, Or(y, And(z, a)))) == AND + OR
    assert count(Nor(x, y)) == NOT + OR
    assert count(Nand(x, y)) == NOT + AND
    assert count(Xor(x, y)) == XOR
    assert count(Implies(x, y)) == IMPLIES
    assert count(Equivalent(x, y)) == EQUIVALENT
    assert count(ITE(x, y, z)) == _ITE
    assert count([Or(x, y), And(x, y), Basic(x + y)]
        ) == ADD + AND + BASIC + OR

    assert count(Basic(Tuple(x))) == BASIC + TUPLE
    #It checks that TUPLE is counted as an operation.

    assert count(Eq(x + y, S(2))) == ADD + EQ


def test_issue_9324():
    def count(val):
        return count_ops(val, visual=False)

    M = MatrixSymbol('M', 10, 10)
    assert count(M[0, 0]) == 0
    assert count(2 * M[0, 0] + M[5, 7]) == 2
    P = MatrixSymbol('P', 3, 3)
    Q = MatrixSymbol('Q', 3, 3)
    assert count(P + Q) == 1
    m = Symbol('m', integer=True)
    n = Symbol('n', integer=True)
    M = MatrixSymbol('M', m + n, m * m)
    assert count(M[0, 1]) == 2


def test_issue_21532():
    f = Function('f')
    g = Function('g')
    FUNC_F, FUNC_G = symbols('FUNC_F, FUNC_G')
    assert f(x).count_ops(visual=True) == FUNC_F
    assert g(x).count_ops(visual=True) == FUNC_G

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_casting_floatingpoint_errors.py ===
import pytest
from pytest import param

import numpy as np
from numpy.testing import IS_WASM


def values_and_dtypes():
    """
    Generate value+dtype pairs that generate floating point errors during
    casts.  The invalid casts to integers will generate "invalid" value
    warnings, the float casts all generate "overflow".

    (The Python int/float paths don't need to get tested in all the same
    situations, but it does not hurt.)
    """
    # Casting to float16:
    yield param(70000, "float16", id="int-to-f2")
    yield param("70000", "float16", id="str-to-f2")
    yield param(70000.0, "float16", id="float-to-f2")
    yield param(np.longdouble(70000.), "float16", id="longdouble-to-f2")
    yield param(np.float64(70000.), "float16", id="double-to-f2")
    yield param(np.float32(70000.), "float16", id="float-to-f2")
    # Casting to float32:
    yield param(10**100, "float32", id="int-to-f4")
    yield param(1e100, "float32", id="float-to-f2")
    yield param(np.longdouble(1e300), "float32", id="longdouble-to-f2")
    yield param(np.float64(1e300), "float32", id="double-to-f2")
    # Casting to float64:
    # If longdouble is double-double, its max can be rounded down to the double
    # max.  So we correct the double spacing (a bit weird, admittedly):
    max_ld = np.finfo(np.longdouble).max
    spacing = np.spacing(np.nextafter(np.finfo("f8").max, 0))
    if max_ld - spacing > np.finfo("f8").max:
        yield param(np.finfo(np.longdouble).max, "float64",
                    id="longdouble-to-f8")

    # Cast to complex32:
    yield param(2e300, "complex64", id="float-to-c8")
    yield param(2e300 + 0j, "complex64", id="complex-to-c8")
    yield param(2e300j, "complex64", id="complex-to-c8")
    yield param(np.longdouble(2e300), "complex64", id="longdouble-to-c8")

    # Invalid float to integer casts:
    with np.errstate(over="ignore"):
        for to_dt in np.typecodes["AllInteger"]:
            for value in [np.inf, np.nan]:
                for from_dt in np.typecodes["AllFloat"]:
                    from_dt = np.dtype(from_dt)
                    from_val = from_dt.type(value)

                    yield param(from_val, to_dt, id=f"{from_val}-to-{to_dt}")


def check_operations(dtype, value):
    """
    There are many dedicated paths in NumPy which cast and should check for
    floating point errors which occurred during those casts.
    """
    if dtype.kind != 'i':
        # These assignments use the stricter setitem logic:
        def assignment():
            arr = np.empty(3, dtype=dtype)
            arr[0] = value

        yield assignment

        def fill():
            arr = np.empty(3, dtype=dtype)
            arr.fill(value)

        yield fill

    def copyto_scalar():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting="unsafe")

    yield copyto_scalar

    def copyto():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting="unsafe")

    yield copyto

    def copyto_scalar_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting="unsafe",
                  where=[True, False, True])

    yield copyto_scalar_masked

    def copyto_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting="unsafe",
                  where=[True, False, True])

    yield copyto_masked

    def direct_cast():
        np.array([value, value, value]).astype(dtype)

    yield direct_cast

    def direct_cast_nd_strided():
        arr = np.full((5, 5, 5), fill_value=value)[:, ::2, :]
        arr.astype(dtype)

    yield direct_cast_nd_strided

    def boolean_array_assignment():
        arr = np.empty(3, dtype=dtype)
        arr[[True, False, True]] = np.array([value, value])

    yield boolean_array_assignment

    def integer_array_assignment():
        arr = np.empty(3, dtype=dtype)
        values = np.array([value, value])

        arr[[0, 1]] = values

    yield integer_array_assignment

    def integer_array_assignment_with_subspace():
        arr = np.empty((5, 3), dtype=dtype)
        values = np.array([value, value, value])

        arr[[0, 2]] = values

    yield integer_array_assignment_with_subspace

    def flat_assignment():
        arr = np.empty((3,), dtype=dtype)
        values = np.array([value, value, value])
        arr.flat[:] = values

    yield flat_assignment

@pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
@pytest.mark.parametrize(["value", "dtype"], values_and_dtypes())
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_floatingpoint_errors_casting(dtype, value):
    dtype = np.dtype(dtype)
    for operation in check_operations(dtype, value):
        dtype = np.dtype(dtype)

        match = "invalid" if dtype.kind in 'iu' else "overflow"
        with pytest.warns(RuntimeWarning, match=match):
            operation()

        with np.errstate(all="raise"):
            with pytest.raises(FloatingPointError, match=match):
                operation()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\masked_shared.py ===
"""
Tests shared by MaskedArray subclasses.
"""
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil


class ComparisonOps(BaseOpsUtil):
    def _compare_other(self, data, op, other):
        # array
        result = pd.Series(op(data, other))
        expected = pd.Series(op(data._data, other), dtype="boolean")

        # fill the nan locations
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)

        # series
        ser = pd.Series(data)
        result = op(ser, other)

        # Set nullable dtype here to avoid upcasting when setting to pd.NA below
        expected = op(pd.Series(data._data), other).astype("boolean")

        # fill the nan locations
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)

    # subclass will override to parametrize 'other'
    def test_scalar(self, other, comparison_op, dtype):
        op = comparison_op
        left = pd.array([1, 0, None], dtype=dtype)

        result = op(left, other)

        if other is pd.NA:
            expected = pd.array([None, None, None], dtype="boolean")
        else:
            values = op(left._data, other)
            expected = pd.arrays.BooleanArray(values, left._mask, copy=True)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(left, pd.array([1, 0, None], dtype=dtype))


class NumericOps:
    # Shared by IntegerArray and FloatingArray, not BooleanArray

    def test_searchsorted_nan(self, dtype):
        # The base class casts to object dtype, for which searchsorted returns
        #  0 from the left and 10 from the right.
        arr = pd.array(range(10), dtype=dtype)

        assert arr.searchsorted(np.nan, side="left") == 10
        assert arr.searchsorted(np.nan, side="right") == 10

    def test_no_shared_mask(self, data):
        result = data + 1
        assert not tm.shares_memory(result, data)

    def test_array(self, comparison_op, dtype):
        op = comparison_op

        left = pd.array([0, 1, 2, None, None, None], dtype=dtype)
        right = pd.array([0, 1, None, 0, 1, None], dtype=dtype)

        result = op(left, right)
        values = op(left._data, right._data)
        mask = left._mask | right._mask

        expected = pd.arrays.BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(
            left, pd.array([0, 1, 2, None, None, None], dtype=dtype)
        )
        tm.assert_extension_array_equal(
            right, pd.array([0, 1, None, 0, 1, None], dtype=dtype)
        )

    def test_compare_with_booleanarray(self, comparison_op, dtype):
        op = comparison_op

        left = pd.array([True, False, None] * 3, dtype="boolean")
        right = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype=dtype)
        other = pd.array([False] * 3 + [True] * 3 + [None] * 3, dtype="boolean")

        expected = op(left, other)
        result = op(left, right)
        tm.assert_extension_array_equal(result, expected)

        # reversed op
        expected = op(other, left)
        result = op(right, left)
        tm.assert_extension_array_equal(result, expected)

    def test_compare_to_string(self, dtype):
        # GH#28930
        ser = pd.Series([1, None], dtype=dtype)
        result = ser == "a"
        expected = pd.Series([False, pd.NA], dtype="boolean")

        tm.assert_series_equal(result, expected)

    def test_ufunc_with_out(self, dtype):
        arr = pd.array([1, 2, 3], dtype=dtype)
        arr2 = pd.array([1, 2, pd.NA], dtype=dtype)

        mask = arr == arr
        mask2 = arr2 == arr2

        result = np.zeros(3, dtype=bool)
        result |= mask
        # If MaskedArray.__array_ufunc__ handled "out" appropriately,
        #  `result` should still be an ndarray.
        assert isinstance(result, np.ndarray)
        assert result.all()

        # result |= mask worked because mask could be cast losslessly to
        #  boolean ndarray. mask2 can't, so this raises
        result = np.zeros(3, dtype=bool)
        msg = "Specify an appropriate 'na_value' for this dtype"
        with pytest.raises(ValueError, match=msg):
            result |= mask2

        # addition
        res = np.add(arr, arr2)
        expected = pd.array([2, 4, pd.NA], dtype=dtype)
        tm.assert_extension_array_equal(res, expected)

        # when passing out=arr, we will modify 'arr' inplace.
        res = np.add(arr, arr2, out=arr)
        assert res is arr
        tm.assert_extension_array_equal(res, expected)
        tm.assert_extension_array_equal(arr, expected)

    def test_mul_td64_array(self, dtype):
        # GH#45622
        arr = pd.array([1, 2, pd.NA], dtype=dtype)
        other = np.arange(3, dtype=np.int64).view("m8[ns]")

        result = arr * other
        expected = pd.array([pd.Timedelta(0), pd.Timedelta(2), pd.NaT])
        tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_map.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.fixture(params=[None, "ignore"])
def na_action(request):
    return request.param


@pytest.mark.parametrize(
    "data, categories",
    [
        (list("abcbca"), list("cab")),
        (pd.interval_range(0, 3).repeat(3), pd.interval_range(0, 3)),
    ],
    ids=["string", "interval"],
)
def test_map_str(data, categories, ordered, na_action):
    # GH 31202 - override base class since we want to maintain categorical/ordered
    cat = Categorical(data, categories=categories, ordered=ordered)
    result = cat.map(str, na_action=na_action)
    expected = Categorical(
        map(str, data), categories=map(str, categories), ordered=ordered
    )
    tm.assert_categorical_equal(result, expected)


def test_map(na_action):
    cat = Categorical(list("ABABC"), categories=list("CBA"), ordered=True)
    result = cat.map(lambda x: x.lower(), na_action=na_action)
    exp = Categorical(list("ababc"), categories=list("cba"), ordered=True)
    tm.assert_categorical_equal(result, exp)

    cat = Categorical(list("ABABC"), categories=list("BAC"), ordered=False)
    result = cat.map(lambda x: x.lower(), na_action=na_action)
    exp = Categorical(list("ababc"), categories=list("bac"), ordered=False)
    tm.assert_categorical_equal(result, exp)

    # GH 12766: Return an index not an array
    result = cat.map(lambda x: 1, na_action=na_action)
    exp = Index(np.array([1] * 5, dtype=np.int64))
    tm.assert_index_equal(result, exp)

    # change categories dtype
    cat = Categorical(list("ABABC"), categories=list("BAC"), ordered=False)

    def f(x):
        return {"A": 10, "B": 20, "C": 30}.get(x)

    result = cat.map(f, na_action=na_action)
    exp = Categorical([10, 20, 10, 20, 30], categories=[20, 10, 30], ordered=False)
    tm.assert_categorical_equal(result, exp)

    mapper = Series([10, 20, 30], index=["A", "B", "C"])
    result = cat.map(mapper, na_action=na_action)
    tm.assert_categorical_equal(result, exp)

    result = cat.map({"A": 10, "B": 20, "C": 30}, na_action=na_action)
    tm.assert_categorical_equal(result, exp)


@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, Index([False, False, True])),
        ([1, 2, np.nan], pd.isna, Index([False, False, True])),
        ([1, 1, np.nan], {1: False}, Categorical([False, False, np.nan])),
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),
        (
            [1, 1, np.nan],
            Series([False, False]),
            Categorical([False, False, np.nan]),
        ),
        (
            [1, 2, np.nan],
            Series([False] * 3),
            Index([False, False, np.nan]),
        ),
    ),
)
def test_map_with_nan_none(data, f, expected):  # GH 24241
    values = Categorical(data)
    result = values.map(f, na_action=None)
    if isinstance(expected, Categorical):
        tm.assert_categorical_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, Categorical([False, False, np.nan])),
        ([1, 2, np.nan], pd.isna, Index([False, False, np.nan])),
        ([1, 1, np.nan], {1: False}, Categorical([False, False, np.nan])),
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),
        (
            [1, 1, np.nan],
            Series([False, False]),
            Categorical([False, False, np.nan]),
        ),
        (
            [1, 2, np.nan],
            Series([False, False, False]),
            Index([False, False, np.nan]),
        ),
    ),
)
def test_map_with_nan_ignore(data, f, expected):  # GH 24241
    values = Categorical(data)
    result = values.map(f, na_action="ignore")
    if data[1] == 1:
        tm.assert_categorical_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)


def test_map_with_dict_or_series(na_action):
    orig_values = ["a", "B", 1, "a"]
    new_values = ["one", 2, 3.0, "one"]
    cat = Categorical(orig_values)

    mapper = Series(new_values[:-1], index=orig_values[:-1])
    result = cat.map(mapper, na_action=na_action)

    # Order of categories in result can be different
    expected = Categorical(new_values, categories=[3.0, 2, "one"])
    tm.assert_categorical_equal(result, expected)

    mapper = dict(zip(orig_values[:-1], new_values[:-1]))
    result = cat.map(mapper, na_action=na_action)
    # Order of categories in result can be different
    tm.assert_categorical_equal(result, expected)


def test_map_na_action_no_default_deprecated():
    # GH51645
    cat = Categorical(["a", "b", "c"])
    msg = (
        "The default value of 'ignore' for the `na_action` parameter in "
        "pandas.Categorical.map is deprecated and will be "
        "changed to 'None' in a future version. Please set na_action to the "
        "desired value to avoid seeing this warning"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        cat.map(lambda x: x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_internals.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


@td.skip_array_manager_invalid_test
def test_consolidate(using_copy_on_write):
    # create unconsolidated DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df["c"] = [4, 5, 6]

    # take a viewing subset
    subset = df[:]

    # each block of subset references a block of df
    assert all(blk.refs.has_reference() for blk in subset._mgr.blocks)

    # consolidate the two int64 blocks
    subset._consolidate_inplace()

    # the float64 block still references the parent one because it still a view
    assert subset._mgr.blocks[0].refs.has_reference()
    # equivalent of assert np.shares_memory(df["b"].values, subset["b"].values)
    # but avoids caching df["b"]
    assert np.shares_memory(get_array(df, "b"), get_array(subset, "b"))

    # the new consolidated int64 block does not reference another
    assert not subset._mgr.blocks[1].refs.has_reference()

    # the parent dataframe now also only is linked for the float column
    assert not df._mgr.blocks[0].refs.has_reference()
    assert df._mgr.blocks[1].refs.has_reference()
    assert not df._mgr.blocks[2].refs.has_reference()

    # and modifying subset still doesn't modify parent
    if using_copy_on_write:
        subset.iloc[0, 1] = 0.0
        assert not df._mgr.blocks[1].refs.has_reference()
        assert df.loc[0, "b"] == 0.1


@pytest.mark.single_cpu
@td.skip_array_manager_invalid_test
def test_switch_options():
    # ensure we can switch the value of the option within one session
    # (assuming data is constructed after switching)

    # using the option_context to ensure we set back to global option value
    # after running the test
    with pd.option_context("mode.copy_on_write", False):
        df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        # df updated with CoW disabled
        assert df.iloc[0, 0] == 0

        pd.options.mode.copy_on_write = True
        df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        # df not updated with CoW enabled
        assert df.iloc[0, 0] == 1

        pd.options.mode.copy_on_write = False
        df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        # df updated with CoW disabled
        assert df.iloc[0, 0] == 0


@td.skip_array_manager_invalid_test
@pytest.mark.parametrize("dtype", [np.intp, np.int8])
@pytest.mark.parametrize(
    "locs, arr",
    [
        ([0], np.array([-1, -2, -3])),
        ([1], np.array([-1, -2, -3])),
        ([5], np.array([-1, -2, -3])),
        ([0, 1], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
        ([0, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
        ([0, 1, 2], np.array([[-1, -2, -3], [-4, -5, -6], [-4, -5, -6]]).T),
        ([1, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
        ([1, 3], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
        ([1, 3], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
    ],
)
def test_iset_splits_blocks_inplace(using_copy_on_write, locs, arr, dtype):
    # Nothing currently calls iset with
    # more than 1 loc with inplace=True (only happens with inplace=False)
    # but ensure that it works
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12],
            "e": [13, 14, 15],
            "f": Series(["a", "b", "c"], dtype=object),
        },
    )
    arr = arr.astype(dtype)
    df_orig = df.copy()
    df2 = df.copy(deep=None)  # Trigger a CoW (if enabled, otherwise makes copy)
    df2._mgr.iset(locs, arr, inplace=True)

    tm.assert_frame_equal(df, df_orig)

    if using_copy_on_write:
        for i, col in enumerate(df.columns):
            if i not in locs:
                assert np.shares_memory(get_array(df, col), get_array(df2, col))
    else:
        for col in df.columns:
            assert not np.shares_memory(get_array(df, col), get_array(df2, col))


def test_exponential_backoff():
    # GH#55518
    df = DataFrame({"a": [1, 2, 3]})
    for i in range(490):
        df.copy(deep=False)

    assert len(df._mgr.blocks[0].refs.referenced_blocks) == 491

    df = DataFrame({"a": [1, 2, 3]})
    dfs = [df.copy(deep=False) for i in range(510)]

    for i in range(20):
        df.copy(deep=False)
    assert len(df._mgr.blocks[0].refs.referenced_blocks) == 531
    assert df._mgr.blocks[0].refs.clear_counter == 1000

    for i in range(500):
        df.copy(deep=False)

    # Don't reduce since we still have over 500 objects alive
    assert df._mgr.blocks[0].refs.clear_counter == 1000

    dfs = dfs[:300]
    for i in range(500):
        df.copy(deep=False)

    # Reduce since there are less than 500 objects alive
    assert df._mgr.blocks[0].refs.clear_counter == 500

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_to_timestamp.py ===
from datetime import timedelta

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    PeriodIndex,
    Series,
    Timedelta,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm


def _get_with_delta(delta, freq="YE-DEC"):
    return date_range(
        to_datetime("1/1/2001") + delta,
        to_datetime("12/31/2009") + delta,
        freq=freq,
    )


class TestToTimestamp:
    def test_to_timestamp(self, frame_or_series):
        K = 5
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), K)),
            index=index,
            columns=["A", "B", "C", "D", "E"],
        )
        obj["mix"] = "a"
        obj = tm.get_obj(obj, frame_or_series)

        exp_index = date_range("1/1/2001", end="12/31/2009", freq="YE-DEC")
        exp_index = exp_index + Timedelta(1, "D") - Timedelta(1, "ns")
        result = obj.to_timestamp("D", "end")
        tm.assert_index_equal(result.index, exp_index)
        tm.assert_numpy_array_equal(result.values, obj.values)
        if frame_or_series is Series:
            assert result.name == "A"

        exp_index = date_range("1/1/2001", end="1/1/2009", freq="YS-JAN")
        result = obj.to_timestamp("D", "start")
        tm.assert_index_equal(result.index, exp_index)

        result = obj.to_timestamp(how="start")
        tm.assert_index_equal(result.index, exp_index)

        delta = timedelta(hours=23)
        result = obj.to_timestamp("H", "end")
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "h") - Timedelta(1, "ns")
        tm.assert_index_equal(result.index, exp_index)

        delta = timedelta(hours=23, minutes=59)
        result = obj.to_timestamp("T", "end")
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "m") - Timedelta(1, "ns")
        tm.assert_index_equal(result.index, exp_index)

        result = obj.to_timestamp("S", "end")
        delta = timedelta(hours=23, minutes=59, seconds=59)
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "s") - Timedelta(1, "ns")
        tm.assert_index_equal(result.index, exp_index)

    def test_to_timestamp_columns(self):
        K = 5
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), K)),
            index=index,
            columns=["A", "B", "C", "D", "E"],
        )
        df["mix"] = "a"

        # columns
        df = df.T

        exp_index = date_range("1/1/2001", end="12/31/2009", freq="YE-DEC")
        exp_index = exp_index + Timedelta(1, "D") - Timedelta(1, "ns")
        result = df.to_timestamp("D", "end", axis=1)
        tm.assert_index_equal(result.columns, exp_index)
        tm.assert_numpy_array_equal(result.values, df.values)

        exp_index = date_range("1/1/2001", end="1/1/2009", freq="YS-JAN")
        result = df.to_timestamp("D", "start", axis=1)
        tm.assert_index_equal(result.columns, exp_index)

        delta = timedelta(hours=23)
        result = df.to_timestamp("H", "end", axis=1)
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "h") - Timedelta(1, "ns")
        tm.assert_index_equal(result.columns, exp_index)

        delta = timedelta(hours=23, minutes=59)
        result = df.to_timestamp("min", "end", axis=1)
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "m") - Timedelta(1, "ns")
        tm.assert_index_equal(result.columns, exp_index)

        result = df.to_timestamp("S", "end", axis=1)
        delta = timedelta(hours=23, minutes=59, seconds=59)
        exp_index = _get_with_delta(delta)
        exp_index = exp_index + Timedelta(1, "s") - Timedelta(1, "ns")
        tm.assert_index_equal(result.columns, exp_index)

        result1 = df.to_timestamp("5min", axis=1)
        result2 = df.to_timestamp("min", axis=1)
        expected = date_range("2001-01-01", "2009-01-01", freq="YS")
        assert isinstance(result1.columns, DatetimeIndex)
        assert isinstance(result2.columns, DatetimeIndex)
        tm.assert_numpy_array_equal(result1.columns.asi8, expected.asi8)
        tm.assert_numpy_array_equal(result2.columns.asi8, expected.asi8)
        # PeriodIndex.to_timestamp always use 'infer'
        assert result1.columns.freqstr == "YS-JAN"
        assert result2.columns.freqstr == "YS-JAN"

    def test_to_timestamp_invalid_axis(self):
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )

        # invalid axis
        with pytest.raises(ValueError, match="axis"):
            obj.to_timestamp(axis=2)

    def test_to_timestamp_hourly(self, frame_or_series):
        index = period_range(freq="h", start="1/1/2001", end="1/2/2001")
        obj = Series(1, index=index, name="foo")
        if frame_or_series is not Series:
            obj = obj.to_frame()

        exp_index = date_range("1/1/2001 00:59:59", end="1/2/2001 00:59:59", freq="h")
        result = obj.to_timestamp(how="end")
        exp_index = exp_index + Timedelta(1, "s") - Timedelta(1, "ns")
        tm.assert_index_equal(result.index, exp_index)
        if frame_or_series is Series:
            assert result.name == "foo"

    def test_to_timestamp_raises(self, index, frame_or_series):
        # GH#33327
        obj = frame_or_series(index=index, dtype=object)

        if not isinstance(index, PeriodIndex):
            msg = f"unsupported Type {type(index).__name__}"
            with pytest.raises(TypeError, match=msg):
                obj.to_timestamp()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_truncate.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameTruncate:
    def test_truncate(self, datetime_frame, frame_or_series):
        ts = datetime_frame[::3]
        ts = tm.get_obj(ts, frame_or_series)

        start, end = datetime_frame.index[3], datetime_frame.index[6]

        start_missing = datetime_frame.index[2]
        end_missing = datetime_frame.index[7]

        # neither specified
        truncated = ts.truncate()
        tm.assert_equal(truncated, ts)

        # both specified
        expected = ts[1:3]

        truncated = ts.truncate(start, end)
        tm.assert_equal(truncated, expected)

        truncated = ts.truncate(start_missing, end_missing)
        tm.assert_equal(truncated, expected)

        # start specified
        expected = ts[1:]

        truncated = ts.truncate(before=start)
        tm.assert_equal(truncated, expected)

        truncated = ts.truncate(before=start_missing)
        tm.assert_equal(truncated, expected)

        # end specified
        expected = ts[:3]

        truncated = ts.truncate(after=end)
        tm.assert_equal(truncated, expected)

        truncated = ts.truncate(after=end_missing)
        tm.assert_equal(truncated, expected)

        # corner case, empty series/frame returned
        truncated = ts.truncate(after=ts.index[0] - ts.index.freq)
        assert len(truncated) == 0

        truncated = ts.truncate(before=ts.index[-1] + ts.index.freq)
        assert len(truncated) == 0

        msg = "Truncate: 2000-01-06 00:00:00 must be after 2000-05-16 00:00:00"
        with pytest.raises(ValueError, match=msg):
            ts.truncate(
                before=ts.index[-1] - ts.index.freq, after=ts.index[0] + ts.index.freq
            )

    def test_truncate_nonsortedindex(self, frame_or_series):
        # GH#17935

        obj = DataFrame({"A": ["a", "b", "c", "d", "e"]}, index=[5, 3, 2, 9, 0])
        obj = tm.get_obj(obj, frame_or_series)

        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            obj.truncate(before=3, after=9)

    def test_sort_values_nonsortedindex(self):
        rng = date_range("2011-01-01", "2012-01-01", freq="W")
        ts = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(len(rng)),
                "B": np.random.default_rng(2).standard_normal(len(rng)),
            },
            index=rng,
        )

        decreasing = ts.sort_values("A", ascending=False)

        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            decreasing.truncate(before="2011-11", after="2011-12")

    def test_truncate_nonsortedindex_axis1(self):
        # GH#17935

        df = DataFrame(
            {
                3: np.random.default_rng(2).standard_normal(5),
                20: np.random.default_rng(2).standard_normal(5),
                2: np.random.default_rng(2).standard_normal(5),
                0: np.random.default_rng(2).standard_normal(5),
            },
            columns=[3, 20, 2, 0],
        )
        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            df.truncate(before=2, after=20, axis=1)

    @pytest.mark.parametrize(
        "before, after, indices",
        [(1, 2, [2, 1]), (None, 2, [2, 1, 0]), (1, None, [3, 2, 1])],
    )
    @pytest.mark.parametrize("dtyp", [*tm.ALL_REAL_NUMPY_DTYPES, "datetime64[ns]"])
    def test_truncate_decreasing_index(
        self, before, after, indices, dtyp, frame_or_series
    ):
        # https://github.com/pandas-dev/pandas/issues/33756
        idx = Index([3, 2, 1, 0], dtype=dtyp)
        if isinstance(idx, DatetimeIndex):
            before = pd.Timestamp(before) if before is not None else None
            after = pd.Timestamp(after) if after is not None else None
            indices = [pd.Timestamp(i) for i in indices]
        values = frame_or_series(range(len(idx)), index=idx)
        result = values.truncate(before=before, after=after)
        expected = values.loc[indices]
        tm.assert_equal(result, expected)

    def test_truncate_multiindex(self, frame_or_series):
        # GH 34564
        mi = pd.MultiIndex.from_product([[1, 2, 3, 4], ["A", "B"]], names=["L1", "L2"])
        s1 = DataFrame(range(mi.shape[0]), index=mi, columns=["col"])
        s1 = tm.get_obj(s1, frame_or_series)

        result = s1.truncate(before=2, after=3)

        df = DataFrame.from_dict(
            {"L1": [2, 2, 3, 3], "L2": ["A", "B", "A", "B"], "col": [2, 3, 4, 5]}
        )
        expected = df.set_index(["L1", "L2"])
        expected = tm.get_obj(expected, frame_or_series)

        tm.assert_equal(result, expected)

    def test_truncate_index_only_one_unique_value(self, frame_or_series):
        # GH 42365
        obj = Series(0, index=date_range("2021-06-30", "2021-06-30")).repeat(5)
        if frame_or_series is DataFrame:
            obj = obj.to_frame(name="a")

        truncated = obj.truncate("2021-06-28", "2021-07-01")

        tm.assert_equal(truncated, obj)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_sample.py ===
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize("n, frac", [(2, None), (None, 0.2)])
def test_groupby_sample_balanced_groups_shape(n, frac):
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(n=n, frac=frac)
    values = [1] * 2 + [2] * 2
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=n, frac=frac)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_unbalanced_groups_shape():
    values = [1] * 10 + [2] * 20
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(n=5)
    values = [1] * 5 + [2] * 5
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=5)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_index_value_spans_groups():
    values = [1] * 3 + [2] * 3
    df = DataFrame({"a": values, "b": values}, index=[1, 2, 2, 2, 2, 2])

    result = df.groupby("a").sample(n=2)
    values = [1] * 2 + [2] * 2
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=2)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_n_and_frac_raises():
    df = DataFrame({"a": [1, 2], "b": [1, 2]})
    msg = "Please enter a value for `frac` OR `n`, not both"

    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(n=1, frac=1.0)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(n=1, frac=1.0)


def test_groupby_sample_frac_gt_one_without_replacement_raises():
    df = DataFrame({"a": [1, 2], "b": [1, 2]})
    msg = "Replace has to be set to `True` when upsampling the population `frac` > 1."

    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(frac=1.5, replace=False)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(frac=1.5, replace=False)


@pytest.mark.parametrize("n", [-1, 1.5])
def test_groupby_sample_invalid_n_raises(n):
    df = DataFrame({"a": [1, 2], "b": [1, 2]})

    if n < 0:
        msg = "A negative number of rows requested. Please provide `n` >= 0."
    else:
        msg = "Only integers accepted as `n` values"

    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(n=n)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(n=n)


def test_groupby_sample_oversample():
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(frac=2.0, replace=True)
    values = [1] * 20 + [2] * 20
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(frac=2.0, replace=True)
    expected = Series(values, name="b", index=result.index)
    tm.assert_series_equal(result, expected)


def test_groupby_sample_without_n_or_frac():
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values})

    result = df.groupby("a").sample(n=None, frac=None)
    expected = DataFrame({"a": [1, 2], "b": [1, 2]}, index=result.index)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=None, frac=None)
    expected = Series([1, 2], name="b", index=result.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index, expected_index",
    [(["w", "x", "y", "z"], ["w", "w", "y", "y"]), ([3, 4, 5, 6], [3, 3, 5, 5])],
)
def test_groupby_sample_with_weights(index, expected_index):
    # GH 39927 - tests for integer index needed
    values = [1] * 2 + [2] * 2
    df = DataFrame({"a": values, "b": values}, index=Index(index))

    result = df.groupby("a").sample(n=2, replace=True, weights=[1, 0, 1, 0])
    expected = DataFrame({"a": values, "b": values}, index=Index(expected_index))
    tm.assert_frame_equal(result, expected)

    result = df.groupby("a")["b"].sample(n=2, replace=True, weights=[1, 0, 1, 0])
    expected = Series(values, name="b", index=Index(expected_index))
    tm.assert_series_equal(result, expected)


def test_groupby_sample_with_selections():
    # GH 39928
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values, "c": values})

    result = df.groupby("a")[["b", "c"]].sample(n=None, frac=None)
    expected = DataFrame({"b": [1, 2], "c": [1, 2]}, index=result.index)
    tm.assert_frame_equal(result, expected)


def test_groupby_sample_with_empty_inputs():
    # GH48459
    df = DataFrame({"a": [], "b": []})
    groupby_df = df.groupby("a")

    result = groupby_df.sample()
    expected = df
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\discrete\tests\test_transforms.py ===
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Symbol, symbols, I, Rational
from sympy.discrete import (fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from sympy.testing.pytest import raises


def test_fft_ifft():
    assert all(tf(ls) == ls for tf in (fft, ifft)
                            for ls in ([], [Rational(5, 3)]))

    ls = list(range(6))
    fls = [15, -7*sqrt(2)/2 - 4 - sqrt(2)*I/2 + 2*I, 2 + 3*I,
             -4 + 7*sqrt(2)/2 - 2*I - sqrt(2)*I/2, -3,
             -4 + 7*sqrt(2)/2 + sqrt(2)*I/2 + 2*I,
              2 - 3*I, -7*sqrt(2)/2 - 4 - 2*I + sqrt(2)*I/2]

    assert fft(ls) == fls
    assert ifft(fls) == ls + [S.Zero]*2

    ls = [1 + 2*I, 3 + 4*I, 5 + 6*I]
    ifls = [Rational(9, 4) + 3*I, I*Rational(-7, 4), Rational(3, 4) + I, -2 - I/4]

    assert ifft(ls) == ifls
    assert fft(ifls) == ls + [S.Zero]

    x = Symbol('x', real=True)
    raises(TypeError, lambda: fft(x))
    raises(ValueError, lambda: ifft([x, 2*x, 3*x**2, 4*x**3]))


def test_ntt_intt():
    # prime moduli of the form (m*2**k + 1), sequence length
    # should be a divisor of 2**k
    p = 7*17*2**23 + 1
    q = 2*500000003 + 1 # only for sequences of length 1 or 2
    r = 2*3*5*7 # composite modulus

    assert all(tf(ls, p) == ls for tf in (ntt, intt)
                                for ls in ([], [5]))

    ls = list(range(6))
    nls = [15, 801133602, 738493201, 334102277, 998244350, 849020224,
            259751156, 12232587]

    assert ntt(ls, p) == nls
    assert intt(nls, p) == ls + [0]*2

    ls = [1 + 2*I, 3 + 4*I, 5 + 6*I]
    x = Symbol('x', integer=True)

    raises(TypeError, lambda: ntt(x, p))
    raises(ValueError, lambda: intt([x, 2*x, 3*x**2, 4*x**3], p))
    raises(ValueError, lambda: intt(ls, p))
    raises(ValueError, lambda: ntt([1.2, 2.1, 3.5], p))
    raises(ValueError, lambda: ntt([3, 5, 6], q))
    raises(ValueError, lambda: ntt([4, 5, 7], r))
    raises(ValueError, lambda: ntt([1.0, 2.0, 3.0], p))


def test_fwht_ifwht():
    assert all(tf(ls) == ls for tf in (fwht, ifwht) \
                        for ls in ([], [Rational(7, 4)]))

    ls = [213, 321, 43235, 5325, 312, 53]
    fls = [49459, 38061, -47661, -37759, 48729, 37543, -48391, -38277]

    assert fwht(ls) == fls
    assert ifwht(fls) == ls + [S.Zero]*2

    ls = [S.Half + 2*I, Rational(3, 7) + 4*I, Rational(5, 6) + 6*I, Rational(7, 3), Rational(9, 4)]
    ifls = [Rational(533, 672) + I*3/2, Rational(23, 224) + I/2, Rational(1, 672), Rational(107, 224) - I,
        Rational(155, 672) + I*3/2, Rational(-103, 224) + I/2, Rational(-377, 672), Rational(-19, 224) - I]

    assert ifwht(ls) == ifls
    assert fwht(ifls) == ls + [S.Zero]*3

    x, y = symbols('x y')

    raises(TypeError, lambda: fwht(x))

    ls = [x, 2*x, 3*x**2, 4*x**3]
    ifls = [x**3 + 3*x**2/4 + x*Rational(3, 4),
        -x**3 + 3*x**2/4 - x/4,
        -x**3 - 3*x**2/4 + x*Rational(3, 4),
        x**3 - 3*x**2/4 - x/4]

    assert ifwht(ls) == ifls
    assert fwht(ifls) == ls

    ls = [x, y, x**2, y**2, x*y]
    fls = [x**2 + x*y + x + y**2 + y,
        x**2 + x*y + x - y**2 - y,
        -x**2 + x*y + x - y**2 + y,
        -x**2 + x*y + x + y**2 - y,
        x**2 - x*y + x + y**2 + y,
        x**2 - x*y + x - y**2 - y,
        -x**2 - x*y + x - y**2 + y,
        -x**2 - x*y + x + y**2 - y]

    assert fwht(ls) == fls
    assert ifwht(fls) == ls + [S.Zero]*3

    ls = list(range(6))

    assert fwht(ls) == [x*8 for x in ifwht(ls)]


def test_mobius_transform():
    assert all(tf(ls, subset=subset) == ls
                for ls in ([], [Rational(7, 4)]) for subset in (True, False)
                for tf in (mobius_transform, inverse_mobius_transform))

    w, x, y, z = symbols('w x y z')

    assert mobius_transform([x, y]) == [x, x + y]
    assert inverse_mobius_transform([x, x + y]) == [x, y]
    assert mobius_transform([x, y], subset=False) == [x + y, y]
    assert inverse_mobius_transform([x + y, y], subset=False) == [x, y]

    assert mobius_transform([w, x, y, z]) == [w, w + x, w + y, w + x + y + z]
    assert inverse_mobius_transform([w, w + x, w + y, w + x + y + z]) == \
            [w, x, y, z]
    assert mobius_transform([w, x, y, z], subset=False) == \
            [w + x + y + z, x + z, y + z, z]
    assert inverse_mobius_transform([w + x + y + z, x + z, y + z, z], subset=False) == \
            [w, x, y, z]

    ls = [Rational(2, 3), Rational(6, 7), Rational(5, 8), 9, Rational(5, 3) + 7*I]
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168),
            Rational(7, 3) + 7*I, Rational(67, 21) + 7*I, Rational(71, 24) + 7*I,
            Rational(2153, 168) + 7*I]

    assert mobius_transform(ls) == mls
    assert inverse_mobius_transform(mls) == ls + [S.Zero]*3

    mls = [Rational(2153, 168) + 7*I, Rational(69, 7), Rational(77, 8), 9, Rational(5, 3) + 7*I, 0, 0, 0]

    assert mobius_transform(ls, subset=False) == mls
    assert inverse_mobius_transform(mls, subset=False) == ls + [S.Zero]*3

    ls = ls[:-1]
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168)]

    assert mobius_transform(ls) == mls
    assert inverse_mobius_transform(mls) == ls

    mls = [Rational(1873, 168), Rational(69, 7), Rational(77, 8), 9]

    assert mobius_transform(ls, subset=False) == mls
    assert inverse_mobius_transform(mls, subset=False) == ls

    raises(TypeError, lambda: mobius_transform(x, subset=True))
    raises(TypeError, lambda: inverse_mobius_transform(y, subset=False))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_levin.py ===
#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpmath import mp
from mpmath import libmp

xrange = libmp.backend.xrange

# Attention:
#   These tests run with 15-20 decimal digits precision. For higher precision the
#   working precision must be raised.

def test_levin_0():
    mp.dps = 17
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method = "levin", variant = "u")
        S, s, n = [], 0, 1
        while 1:
            s += mp.one / (n * n)
            n += 1
            S.append(s)
            v, e = L.update_psum(S)
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(eps))
    err = abs(v - mp.pi ** 2 / 6)
    assert err < eps
    w = mp.nsum(lambda n: 1/(n * n), [1, mp.inf], method = "levin", levin_variant = "u")
    err = abs(v - w)
    assert err < eps

def test_levin_1():
    mp.dps = 17
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method = "levin", variant = "v")
        A, n = [], 1
        while 1:
            s = mp.mpf(n) ** (2 + 3j)
            n += 1
            A.append(s)
            v, e = L.update(A)
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(eps))
    err = abs(v - mp.zeta(-2-3j))
    assert err < eps
    w = mp.nsum(lambda n: n ** (2 + 3j), [1, mp.inf], method = "levin", levin_variant = "v")
    err = abs(v - w)
    assert err < eps

def test_levin_2():
    # [2] A. Sidi - "Pratical Extrapolation Methods" p.373
    mp.dps = 17
    z=mp.mpf(10)
    eps = mp.mpf(mp.eps)
    with mp.extraprec(2 * mp.prec):
        L = mp.levin(method = "sidi", variant = "t")
        n = 0
        while 1:
            s = (-1)**n * mp.fac(n) * z ** (-n)
            v, e = L.step(s)
            n += 1
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(eps))
    exact = mp.quad(lambda x: mp.exp(-x)/(1+x/z),[0,mp.inf])
    # there is also a symbolic expression for the integral:
    #   exact = z * mp.exp(z) * mp.expint(1,z)
    err = abs(v - exact)
    assert err < eps
    w = mp.nsum(lambda n: (-1) ** n * mp.fac(n) * z ** (-n), [0, mp.inf], method = "sidi", levin_variant = "t")
    assert err < eps

def test_levin_3():
    mp.dps = 17
    z=mp.mpf(2)
    eps = mp.mpf(mp.eps)
    with mp.extraprec(7*mp.prec):  # we need copious amount of precision to sum this highly divergent series
        L = mp.levin(method = "levin", variant = "t")
        n, s = 0, 0
        while 1:
            s += (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n))
            n += 1
            v, e = L.step_psum(s)
            if e < eps:
                break
            if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.8 * mp.log(eps))
    exact = mp.quad(lambda x: mp.exp( -x * x / 2 - z * x ** 4), [0,mp.inf]) * 2 / mp.sqrt(2 * mp.pi)
    # there is also a symbolic expression for the integral:
    #   exact = mp.exp(mp.one / (32 * z)) * mp.besselk(mp.one / 4, mp.one / (32 * z)) / (4 * mp.sqrt(z * mp.pi))
    err = abs(v - exact)
    assert err < eps
    w = mp.nsum(lambda n: (-z)**n * mp.fac(4 * n) / (mp.fac(n) * mp.fac(2 * n) * (4 ** n)), [0, mp.inf], method = "levin", levin_variant = "t", workprec = 8*mp.prec, steps = [2] + [1 for x in xrange(1000)])
    err = abs(v - w)
    assert err < eps

def test_levin_nsum():
    mp.dps = 17

    with mp.extraprec(mp.prec):
        z = mp.mpf(10) ** (-10)
        a = mp.nsum(lambda n: n**(-(1+z)), [1, mp.inf], method = "l") - 1 / z
        assert abs(a - mp.euler) < 1e-10

    eps = mp.exp(0.8 * mp.log(mp.eps))

    a = mp.nsum(lambda n: (-1)**(n-1) / n, [1, mp.inf], method = "sidi")
    assert abs(a - mp.log(2)) < eps

    z = 2 + 1j
    f = lambda n: mp.rf(2 / mp.mpf(3), n) * mp.rf(4 / mp.mpf(3), n) * z**n / (mp.rf(1 / mp.mpf(3), n) * mp.fac(n))
    v = mp.nsum(f, [0, mp.inf], method = "levin", steps = [10 for x in xrange(1000)])
    exact = mp.hyp2f1(2 / mp.mpf(3), 4 / mp.mpf(3), 1 / mp.mpf(3), z)
    assert abs(exact - v) < eps

def test_cohen_alt_0():
    mp.dps = 17
    AC = mp.cohen_alt()
    S, s, n = [], 0, 1
    while 1:
        s += -((-1) ** n) * mp.one / (n * n)
        n += 1
        S.append(s)
        v, e = AC.update_psum(S)
        if e < mp.eps:
            break
        if n > 1000: raise RuntimeError("iteration limit exceeded")
    eps = mp.exp(0.9 * mp.log(mp.eps))
    err = abs(v - mp.pi ** 2 / 12)
    assert err < eps

def test_cohen_alt_1():
    mp.dps = 17
    A = []
    AC = mp.cohen_alt()
    n = 1
    while 1:
        A.append( mp.loggamma(1 + mp.one / (2 * n - 1)))
        A.append(-mp.loggamma(1 + mp.one / (2 * n)))
        n += 1
        v, e = AC.update(A)
        if e < mp.eps:
            break
        if n > 1000: raise RuntimeError("iteration limit exceeded")
    v = mp.exp(v)
    err = abs(v - 1.06215090557106)
    assert err < 1e-12

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_scalarbuffer.py ===
"""
Test scalar buffer interface adheres to PEP 3118
"""
import pytest
from numpy._core._multiarray_tests import get_buffer_info
from numpy._core._rational_tests import rational

import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises

# PEP3118 format strings for native (standard alignment and byteorder) types
scalars_and_codes = [
    (np.bool, '?'),
    (np.byte, 'b'),
    (np.short, 'h'),
    (np.intc, 'i'),
    (np.long, 'l'),
    (np.longlong, 'q'),
    (np.ubyte, 'B'),
    (np.ushort, 'H'),
    (np.uintc, 'I'),
    (np.ulong, 'L'),
    (np.ulonglong, 'Q'),
    (np.half, 'e'),
    (np.single, 'f'),
    (np.double, 'd'),
    (np.longdouble, 'g'),
    (np.csingle, 'Zf'),
    (np.cdouble, 'Zd'),
    (np.clongdouble, 'Zg'),
]
scalars_only, codes_only = zip(*scalars_and_codes)


class TestScalarPEP3118:

    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_match_array(self, scalar):
        x = scalar()
        a = np.array([], dtype=np.dtype(scalar))
        mv_x = memoryview(x)
        mv_a = memoryview(a)
        assert_equal(mv_x.format, mv_a.format)

    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_dim(self, scalar):
        x = scalar()
        mv_x = memoryview(x)
        assert_equal(mv_x.itemsize, np.dtype(scalar).itemsize)
        assert_equal(mv_x.ndim, 0)
        assert_equal(mv_x.shape, ())
        assert_equal(mv_x.strides, ())
        assert_equal(mv_x.suboffsets, ())

    @pytest.mark.parametrize('scalar, code', scalars_and_codes, ids=codes_only)
    def test_scalar_code_and_properties(self, scalar, code):
        x = scalar()
        expected = {'strides': (), 'itemsize': x.dtype.itemsize, 'ndim': 0,
                        'shape': (), 'format': code, 'readonly': True}

        mv_x = memoryview(x)
        assert self._as_dict(mv_x) == expected

    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_buffers_readonly(self, scalar):
        x = scalar()
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(x, ["WRITABLE"])

    def test_void_scalar_structured_data(self):
        dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
        x = np.array(('ndarray_scalar', (1.2, 3.0)), dtype=dt)[()]
        assert_(isinstance(x, np.void))
        mv_x = memoryview(x)
        expected_size = 16 * np.dtype((np.str_, 1)).itemsize
        expected_size += 2 * np.dtype(np.float64).itemsize
        assert_equal(mv_x.itemsize, expected_size)
        assert_equal(mv_x.ndim, 0)
        assert_equal(mv_x.shape, ())
        assert_equal(mv_x.strides, ())
        assert_equal(mv_x.suboffsets, ())

        # check scalar format string against ndarray format string
        a = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        assert_(isinstance(a, np.ndarray))
        mv_a = memoryview(a)
        assert_equal(mv_x.itemsize, mv_a.itemsize)
        assert_equal(mv_x.format, mv_a.format)

        # Check that we do not allow writeable buffer export (technically
        # we could allow it sometimes here...)
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(x, ["WRITABLE"])

    def _as_dict(self, m):
        return {'strides': m.strides, 'shape': m.shape, 'itemsize': m.itemsize,
                    'ndim': m.ndim, 'format': m.format, 'readonly': m.readonly}

    def test_datetime_memoryview(self):
        # gh-11656
        # Values verified with v1.13.3, shape is not () as in test_scalar_dim

        dt1 = np.datetime64('2016-01-01')
        dt2 = np.datetime64('2017-01-01')
        expected = {'strides': (1,), 'itemsize': 1, 'ndim': 1, 'shape': (8,),
                        'format': 'B', 'readonly': True}
        v = memoryview(dt1)
        assert self._as_dict(v) == expected

        v = memoryview(dt2 - dt1)
        assert self._as_dict(v) == expected

        dt = np.dtype([('a', 'uint16'), ('b', 'M8[s]')])
        a = np.empty(1, dt)
        # Fails to create a PEP 3118 valid buffer
        assert_raises((ValueError, BufferError), memoryview, a[0])

        # Check that we do not allow writeable buffer export
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(dt1, ["WRITABLE"])

    @pytest.mark.parametrize('s', [
        pytest.param("\x32\x32", id="ascii"),
        pytest.param("\uFE0F\uFE0F", id="basic multilingual"),
        pytest.param("\U0001f4bb\U0001f4bb", id="non-BMP"),
    ])
    def test_str_ucs4(self, s):
        s = np.str_(s)  # only our subclass implements the buffer protocol

        # all the same, characters always encode as ucs4
        expected = {'strides': (), 'itemsize': 8, 'ndim': 0, 'shape': (), 'format': '2w',
                        'readonly': True}

        v = memoryview(s)
        assert self._as_dict(v) == expected

        # integers of the paltform-appropriate endianness
        code_points = np.frombuffer(v, dtype='i4')

        assert_equal(code_points, [ord(c) for c in s])

        # Check that we do not allow writeable buffer export
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(s, ["WRITABLE"])

    def test_user_scalar_fails_buffer(self):
        r = rational(1)
        with assert_raises(TypeError):
            memoryview(r)

        # Check that we do not allow writeable buffer export
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(r, ["WRITABLE"])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\reduce.py ===
from typing import final

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype


class BaseReduceTests:
    """
    Reduction specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # Specify if we expect this reduction to succeed.
        return False

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # We perform the same operation on the np.float64 data and check
        #  that the results match. Override if you need to cast to something
        #  other than float64.
        res_op = getattr(ser, op_name)

        try:
            alt = ser.astype("float64")
        except (TypeError, ValueError):
            # e.g. Interval can't cast (TypeError), StringArray can't cast
            #  (ValueError), so let's cast to object and do
            #  the reduction pointwise
            alt = ser.astype(object)

        exp_op = getattr(alt, op_name)
        if op_name == "count":
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # Find the expected dtype when the given reduction is done on a DataFrame
        # column with this array.  The default assumes float64-like behavior,
        # i.e. retains the dtype.
        return arr.dtype

    # We anticipate that authors should not need to override check_reduce_frame,
    #  but should be able to do any necessary overriding in
    #  _get_expected_reduction_dtype. If you have a use case where this
    #  does not hold, please let us know at github.com/pandas-dev/pandas/issues.
    @final
    def check_reduce_frame(self, ser: pd.Series, op_name: str, skipna: bool):
        # Check that the 2D reduction done in a DataFrame reduction "looks like"
        # a wrapped version of the 1D reduction done by Series.
        arr = ser.array
        df = pd.DataFrame({"a": arr})

        kwargs = {"ddof": 1} if op_name in ["var", "std"] else {}

        cmp_dtype = self._get_expected_reduction_dtype(arr, op_name, skipna)

        # The DataFrame method just calls arr._reduce with keepdims=True,
        #  so this first check is perfunctory.
        result1 = arr._reduce(op_name, skipna=skipna, keepdims=True, **kwargs)
        result2 = getattr(df, op_name)(skipna=skipna, **kwargs).array
        tm.assert_extension_array_equal(result1, result2)

        # Check that the 2D reduction looks like a wrapped version of the
        #  1D reduction
        if not skipna and ser.isna().any():
            expected = pd.array([pd.NA], dtype=cmp_dtype)
        else:
            exp_value = getattr(ser.dropna(), op_name)()
            expected = pd.array([exp_value], dtype=cmp_dtype)

        tm.assert_extension_array_equal(result1, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        op_name = all_boolean_reductions
        ser = pd.Series(data)

        if not self._supports_reduction(ser, op_name):
            # TODO: the message being checked here isn't actually checking anything
            msg = (
                "[Cc]annot perform|Categorical is not ordered for operation|"
                "does not support reduction|"
            )

            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)

        else:
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions
        ser = pd.Series(data)

        if not self._supports_reduction(ser, op_name):
            # TODO: the message being checked here isn't actually checking anything
            msg = (
                "[Cc]annot perform|Categorical is not ordered for operation|"
                "does not support reduction|"
            )

            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)

        else:
            # min/max with empty produce numpy warnings
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        op_name = all_numeric_reductions
        ser = pd.Series(data)
        if not is_numeric_dtype(ser.dtype):
            pytest.skip(f"{ser.dtype} is not numeric dtype")

        if op_name in ["count", "kurt", "sem"]:
            pytest.skip(f"{op_name} not an array method")

        if not self._supports_reduction(ser, op_name):
            pytest.skip(f"Reduction {op_name} not supported for this dtype")

        self.check_reduce_frame(ser, op_name, skipna)


# TODO(3.0): remove BaseNoReduceTests, BaseNumericReduceTests,
#  BaseBooleanReduceTests
class BaseNoReduceTests(BaseReduceTests):
    """we don't define any reductions"""


class BaseNumericReduceTests(BaseReduceTests):
    # For backward compatibility only, this only runs the numeric reductions
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ["any", "all"]:
            pytest.skip("These are tested in BaseBooleanReduceTests")
        return True


class BaseBooleanReduceTests(BaseReduceTests):
    # For backward compatibility only, this only runs the numeric reductions
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name not in ["any", "all"]:
            pytest.skip("These are tested in BaseNumericReduceTests")
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_filter.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm


class TestDataFrameFilter:
    def test_filter(self, float_frame, float_string_frame):
        # Items
        filtered = float_frame.filter(["A", "B", "E"])
        assert len(filtered.columns) == 2
        assert "E" not in filtered

        filtered = float_frame.filter(["A", "B", "E"], axis="columns")
        assert len(filtered.columns) == 2
        assert "E" not in filtered

        # Other axis
        idx = float_frame.index[0:4]
        filtered = float_frame.filter(idx, axis="index")
        expected = float_frame.reindex(index=idx)
        tm.assert_frame_equal(filtered, expected)

        # like
        fcopy = float_frame.copy()
        fcopy["AA"] = 1

        filtered = fcopy.filter(like="A")
        assert len(filtered.columns) == 2
        assert "AA" in filtered

        # like with ints in column names
        df = DataFrame(0.0, index=[0, 1, 2], columns=[0, 1, "_A", "_B"])
        filtered = df.filter(like="_")
        assert len(filtered.columns) == 2

        # regex with ints in column names
        # from PR #10384
        df = DataFrame(0.0, index=[0, 1, 2], columns=["A1", 1, "B", 2, "C"])
        expected = DataFrame(
            0.0, index=[0, 1, 2], columns=pd.Index([1, 2], dtype=object)
        )
        filtered = df.filter(regex="^[0-9]+$")
        tm.assert_frame_equal(filtered, expected)

        expected = DataFrame(0.0, index=[0, 1, 2], columns=[0, "0", 1, "1"])
        # shouldn't remove anything
        filtered = expected.filter(regex="^[0-9]+$")
        tm.assert_frame_equal(filtered, expected)

        # pass in None
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter()
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter(items=None)
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter(axis=1)

        # test mutually exclusive arguments
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$", like="bbi")
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$", axis=1)
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$")
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], like="bbi", axis=0)
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], like="bbi")

        # objects
        filtered = float_string_frame.filter(like="foo")
        assert "foo" in filtered

        # unicode columns, won't ascii-encode
        df = float_frame.rename(columns={"B": "\u2202"})
        filtered = df.filter(like="C")
        assert "C" in filtered

    def test_filter_regex_search(self, float_frame):
        fcopy = float_frame.copy()
        fcopy["AA"] = 1

        # regex
        filtered = fcopy.filter(regex="[A]+")
        assert len(filtered.columns) == 2
        assert "AA" in filtered

        # doesn't have to be at beginning
        df = DataFrame(
            {"aBBa": [1, 2], "BBaBB": [1, 2], "aCCa": [1, 2], "aCCaBB": [1, 2]}
        )

        result = df.filter(regex="BB")
        exp = df[[x for x in df.columns if "BB" in x]]
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("a", DataFrame({"a": [1, 2]})),
            ("a", DataFrame({"a": [1, 2]})),
            ("あ", DataFrame({"あ": [3, 4]})),
        ],
    )
    def test_filter_unicode(self, name, expected):
        # GH13101
        df = DataFrame({"a": [1, 2], "あ": [3, 4]})

        tm.assert_frame_equal(df.filter(like=name), expected)
        tm.assert_frame_equal(df.filter(regex=name), expected)

    @pytest.mark.parametrize("name", ["a", "a"])
    def test_filter_bytestring(self, name):
        # GH13101
        df = DataFrame({b"a": [1, 2], b"b": [3, 4]})
        expected = DataFrame({b"a": [1, 2]})

        tm.assert_frame_equal(df.filter(like=name), expected)
        tm.assert_frame_equal(df.filter(regex=name), expected)

    def test_filter_corner(self):
        empty = DataFrame()

        result = empty.filter([])
        tm.assert_frame_equal(result, empty)

        result = empty.filter(like="foo")
        tm.assert_frame_equal(result, empty)

    def test_filter_regex_non_string(self):
        # GH#5798 trying to filter on non-string columns should drop,
        #  not raise
        df = DataFrame(np.random.default_rng(2).random((3, 2)), columns=["STRING", 123])
        result = df.filter(regex="STRING")
        expected = df[["STRING"]]
        tm.assert_frame_equal(result, expected)

    def test_filter_keep_order(self):
        # GH#54980
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.filter(items=["B", "A"])
        expected = df[["B", "A"]]
        tm.assert_frame_equal(result, expected)

    def test_filter_different_dtype(self):
        # GH#54980
        df = DataFrame({1: [1, 2, 3], 2: [4, 5, 6]})
        result = df.filter(items=["B", "A"])
        expected = df[[]]
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_sorted.py ===
import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    MultiIndex,
    Series,
    array,
)
import pandas._testing as tm


class TestMultiIndexSorted:
    def test_getitem_multilevel_index_tuple_not_sorted(self):
        index_columns = list("abc")
        df = DataFrame(
            [[0, 1, 0, "x"], [0, 0, 1, "y"]], columns=index_columns + ["data"]
        )
        df = df.set_index(index_columns)
        query_index = df.index[:1]
        rs = df.loc[query_index, "data"]

        xp_idx = MultiIndex.from_tuples([(0, 1, 0)], names=["a", "b", "c"])
        xp = Series(["x"], index=xp_idx, name="data")
        tm.assert_series_equal(rs, xp)

    def test_getitem_slice_not_sorted(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        df = frame.sort_index(level=1).T

        # buglet with int typechecking
        result = df.iloc[:, : np.int32(3)]
        expected = df.reindex(columns=df.columns[:3])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("key", [None, lambda x: x])
    def test_frame_getitem_not_sorted2(self, key):
        # 13431
        df = DataFrame(
            {
                "col1": ["b", "d", "b", "a"],
                "col2": [3, 1, 1, 2],
                "data": ["one", "two", "three", "four"],
            }
        )

        df2 = df.set_index(["col1", "col2"])
        df2_original = df2.copy()

        df2.index = df2.index.set_levels(["b", "d", "a"], level="col1")
        df2.index = df2.index.set_codes([0, 1, 0, 2], level="col1")
        assert not df2.index.is_monotonic_increasing

        assert df2_original.index.equals(df2.index)
        expected = df2.sort_index(key=key)
        assert expected.index.is_monotonic_increasing

        result = df2.sort_index(level=0, key=key)
        assert result.index.is_monotonic_increasing
        tm.assert_frame_equal(result, expected)

    def test_sort_values_key(self):
        arrays = [
            ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = zip(*arrays)
        index = MultiIndex.from_tuples(tuples)
        index = index.sort_values(  # sort by third letter
            key=lambda x: x.map(lambda entry: entry[2])
        )
        result = DataFrame(range(8), index=index)

        arrays = [
            ["foo", "foo", "bar", "bar", "qux", "qux", "baz", "baz"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = zip(*arrays)
        index = MultiIndex.from_tuples(tuples)
        expected = DataFrame(range(8), index=index)

        tm.assert_frame_equal(result, expected)

    def test_argsort_with_na(self):
        # GH48495
        arrays = [
            array([2, NA, 1], dtype="Int64"),
            array([1, 2, 3], dtype="Int64"),
        ]
        index = MultiIndex.from_arrays(arrays)
        result = index.argsort()
        expected = np.array([2, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_sort_values_with_na(self):
        # GH48495
        arrays = [
            array([2, NA, 1], dtype="Int64"),
            array([1, 2, 3], dtype="Int64"),
        ]
        index = MultiIndex.from_arrays(arrays)
        index = index.sort_values()
        result = DataFrame(range(3), index=index)

        arrays = [
            array([1, 2, NA], dtype="Int64"),
            array([3, 1, 2], dtype="Int64"),
        ]
        index = MultiIndex.from_arrays(arrays)
        expected = DataFrame(range(3), index=index)

        tm.assert_frame_equal(result, expected)

    def test_frame_getitem_not_sorted(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        df = frame.T
        df["foo", "four"] = "foo"

        arrays = [np.array(x) for x in zip(*df.columns.values)]

        result = df["foo"]
        result2 = df.loc[:, "foo"]
        expected = df.reindex(columns=df.columns[arrays[0] == "foo"])
        expected.columns = expected.columns.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        df = df.T
        result = df.xs("foo")
        result2 = df.loc["foo"]
        expected = df.reindex(df.index[arrays[0] == "foo"])
        expected.index = expected.index.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_series_getitem_not_sorted(self):
        arrays = [
            ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = zip(*arrays)
        index = MultiIndex.from_tuples(tuples)
        s = Series(np.random.default_rng(2).standard_normal(8), index=index)

        arrays = [np.array(x) for x in zip(*index.values)]

        result = s["qux"]
        result2 = s.loc["qux"]
        expected = s[arrays[0] == "qux"]
        expected.index = expected.index.droplevel(0)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_domainscalar.py ===
from sympy.testing.pytest import raises

from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix


def test_DomainScalar___new__():
    raises(TypeError, lambda: DomainScalar(ZZ(1), QQ))
    raises(TypeError, lambda: DomainScalar(ZZ(1), 1))


def test_DomainScalar_new():
    A = DomainScalar(ZZ(1), ZZ)
    B = A.new(ZZ(4), ZZ)
    assert B == DomainScalar(ZZ(4), ZZ)


def test_DomainScalar_repr():
    A = DomainScalar(ZZ(1), ZZ)
    assert repr(A) in {'1', 'mpz(1)'}


def test_DomainScalar_from_sympy():
    expr = S(1)
    B = DomainScalar.from_sympy(expr)
    assert B == DomainScalar(ZZ(1), ZZ)


def test_DomainScalar_to_sympy():
    B = DomainScalar(ZZ(1), ZZ)
    expr = B.to_sympy()
    assert expr.is_Integer and expr == 1


def test_DomainScalar_to_domain():
    A = DomainScalar(ZZ(1), ZZ)
    B = A.to_domain(QQ)
    assert B == DomainScalar(QQ(1), QQ)


def test_DomainScalar_convert_to():
    A = DomainScalar(ZZ(1), ZZ)
    B = A.convert_to(QQ)
    assert B == DomainScalar(QQ(1), QQ)


def test_DomainScalar_unify():
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    A, B = A.unify(B)
    assert A.domain == B.domain == QQ


def test_DomainScalar_add():
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A + B == DomainScalar(QQ(3), QQ)

    raises(TypeError, lambda: A + 1.5)

def test_DomainScalar_sub():
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A - B == DomainScalar(QQ(-1), QQ)

    raises(TypeError, lambda: A - 1.5)

def test_DomainScalar_mul():
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    dm = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A * B == DomainScalar(QQ(2), QQ)
    assert A * dm == dm
    assert B * 2 == DomainScalar(QQ(4), QQ)

    raises(TypeError, lambda: A * 1.5)


def test_DomainScalar_floordiv():
    A = DomainScalar(ZZ(-5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A // B == DomainScalar(QQ(-5, 2), QQ)
    C = DomainScalar(ZZ(2), ZZ)
    assert A // C == DomainScalar(ZZ(-3), ZZ)

    raises(TypeError, lambda: A // 1.5)


def test_DomainScalar_mod():
    A = DomainScalar(ZZ(5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A % B == DomainScalar(QQ(0), QQ)
    C = DomainScalar(ZZ(2), ZZ)
    assert A % C == DomainScalar(ZZ(1), ZZ)

    raises(TypeError, lambda: A % 1.5)


def test_DomainScalar_divmod():
    A = DomainScalar(ZZ(5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert divmod(A, B) == (DomainScalar(QQ(5, 2), QQ), DomainScalar(QQ(0), QQ))
    C = DomainScalar(ZZ(2), ZZ)
    assert divmod(A, C) == (DomainScalar(ZZ(2), ZZ), DomainScalar(ZZ(1), ZZ))

    raises(TypeError, lambda: divmod(A, 1.5))


def test_DomainScalar_pow():
    A = DomainScalar(ZZ(-5), ZZ)
    B = A**(2)
    assert B == DomainScalar(ZZ(25), ZZ)

    raises(TypeError, lambda: A**(1.5))


def test_DomainScalar_pos():
    A = DomainScalar(QQ(2), QQ)
    B = DomainScalar(QQ(2), QQ)
    assert  +A == B


def test_DomainScalar_neg():
    A = DomainScalar(QQ(2), QQ)
    B = DomainScalar(QQ(-2), QQ)
    assert  -A == B


def test_DomainScalar_eq():
    A = DomainScalar(QQ(2), QQ)
    assert A == A
    B = DomainScalar(ZZ(-5), ZZ)
    assert A != B
    C = DomainScalar(ZZ(2), ZZ)
    assert A != C
    D = [1]
    assert A != D


def test_DomainScalar_isZero():
    A = DomainScalar(ZZ(0), ZZ)
    assert A.is_zero() == True
    B = DomainScalar(ZZ(1), ZZ)
    assert B.is_zero() == False


def test_DomainScalar_isOne():
    A = DomainScalar(ZZ(1), ZZ)
    assert A.is_one() == True
    B = DomainScalar(ZZ(0), ZZ)
    assert B.is_one() == False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_regression.py ===
import sys

import numpy as np
from numpy import random
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_raises,
)


class TestRegression:

    def test_VonMises_range(self):
        # Make sure generated random variables are in [-pi, pi].
        # Regression test for ticket #986.
        for mu in np.linspace(-7., 7., 5):
            r = random.mtrand.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        # Test for ticket #921
        assert_(np.all(np.random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(np.random.hypergeometric(18, 3, 11, size=10) > 0))

        # Test for ticket #5623
        args = [
            (2**20 - 2, 2**20 - 2, 2**20 - 2),  # Check for 32-bit systems
        ]
        is_64bits = sys.maxsize > 2**32
        if is_64bits and sys.platform != 'win32':
            # Check for 64-bit systems
            args.append((2**40 - 2, 2**40 - 2, 2**40 - 2))
        for arg in args:
            assert_(np.random.hypergeometric(*arg) > 0)

    def test_logseries_convergence(self):
        # Test for ticket #923
        N = 1000
        np.random.seed(0)
        rvsn = np.random.logseries(0.8, size=N)
        # these two frequency counts should be close to theoretical
        # numbers with this large sample
        # theoretical large N result is 0.49706795
        freq = np.sum(rvsn == 1) / N
        msg = f'Frequency was {freq:f}, should be > 0.45'
        assert_(freq > 0.45, msg)
        # theoretical large N result is 0.19882718
        freq = np.sum(rvsn == 2) / N
        msg = f'Frequency was {freq:f}, should be < 0.23'
        assert_(freq < 0.23, msg)

    def test_shuffle_mixed_dimension(self):
        # Test for trac ticket #2074
        for t in [[1, 2, 3, None],
                  [(1, 1), (2, 2), (3, 3), None],
                  [1, (2, 2), (3, 3), None],
                  [(1, 1), 2, 3, None]]:
            np.random.seed(12345)
            shuffled = list(t)
            random.shuffle(shuffled)
            expected = np.array([t[0], t[3], t[1], t[2]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    def test_call_within_randomstate(self):
        # Check that custom RandomState does not call into global state
        m = np.random.RandomState()
        res = np.array([0, 8, 7, 2, 1, 9, 4, 7, 0, 3])
        for i in range(3):
            np.random.seed(i)
            m.seed(4321)
            # If m.state is not honored, the result will change
            assert_array_equal(m.choice(10, size=10, p=np.ones(10) / 10.), res)

    def test_multivariate_normal_size_types(self):
        # Test for multivariate_normal issue with 'size' argument.
        # Check that the multivariate_normal size argument can be a
        # numpy integer.
        np.random.multivariate_normal([0], [[0]], size=1)
        np.random.multivariate_normal([0], [[0]], size=np.int_(1))
        np.random.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # Test that beta with small a and b parameters does not produce
        # NaNs due to roundoff errors causing 0 / 0, gh-5851
        np.random.seed(1234567890)
        x = np.random.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in np.random.beta')

    def test_choice_sum_of_probs_tolerance(self):
        # The sum of probs should be 1.0 with some tolerance.
        # For low precision dtypes the tolerance was too tight.
        # See numpy github issue 6123.
        np.random.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = np.random.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, np.random.choice, a, p=probs * 0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # Test that permuting an array of different length strings
        # will not cause a segfault on garbage collection
        # Tests gh-7710
        np.random.seed(1234)

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            np.random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # Test that permuting an array of objects will not cause
        # a segfault on garbage collection.
        # See gh-7719
        np.random.seed(1234)
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            np.random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        class N(np.ndarray):
            pass

        np.random.seed(1)
        orig = np.arange(3).view(N)
        perm = np.random.permutation(orig)
        assert_array_equal(perm, np.array([0, 2, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        np.random.seed(1)
        m = M()
        perm = np.random.permutation(m)
        assert_array_equal(perm, np.array([2, 1, 4, 0, 3]))
        assert_array_equal(m.__array__(), np.arange(5))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_mask.py ===
"""
Tests for DataFrame.mask; tests DataFrame.where as a side-effect.
"""

import numpy as np

from pandas import (
    NA,
    DataFrame,
    Float64Dtype,
    Series,
    StringDtype,
    Timedelta,
    isna,
)
import pandas._testing as tm


class TestDataFrameMask:
    def test_mask(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        cond = df > 0

        rs = df.where(cond, np.nan)
        tm.assert_frame_equal(rs, df.mask(df <= 0))
        tm.assert_frame_equal(rs, df.mask(~cond))

        other = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        rs = df.where(cond, other)
        tm.assert_frame_equal(rs, df.mask(df <= 0, other))
        tm.assert_frame_equal(rs, df.mask(~cond, other))

    def test_mask2(self):
        # see GH#21891
        df = DataFrame([1, 2])
        res = df.mask([[True], [False]])

        exp = DataFrame([np.nan, 2])
        tm.assert_frame_equal(res, exp)

    def test_mask_inplace(self):
        # GH#8801
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        cond = df > 0

        rdf = df.copy()

        return_value = rdf.where(cond, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(rdf, df.where(cond))
        tm.assert_frame_equal(rdf, df.mask(~cond))

        rdf = df.copy()
        return_value = rdf.where(cond, -df, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(rdf, df.where(cond, -df))
        tm.assert_frame_equal(rdf, df.mask(~cond, -df))

    def test_mask_edge_case_1xN_frame(self):
        # GH#4071
        df = DataFrame([[1, 2]])
        res = df.mask(DataFrame([[True, False]]))
        expec = DataFrame([[np.nan, 2]])
        tm.assert_frame_equal(res, expec)

    def test_mask_callable(self):
        # GH#12533
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = df.mask(lambda x: x > 4, lambda x: x + 1)
        exp = DataFrame([[1, 2, 3], [4, 6, 7], [8, 9, 10]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.mask(df > 4, df + 1))

        # return ndarray and scalar
        result = df.mask(lambda x: (x % 2 == 0).values, lambda x: 99)
        exp = DataFrame([[1, 99, 3], [99, 5, 99], [7, 99, 9]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.mask(df % 2 == 0, 99))

        # chain
        result = (df + 2).mask(lambda x: x > 8, lambda x: x + 10)
        exp = DataFrame([[3, 4, 5], [6, 7, 8], [19, 20, 21]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, (df + 2).mask((df + 2) > 8, (df + 2) + 10))

    def test_mask_dtype_bool_conversion(self):
        # GH#3733
        df = DataFrame(data=np.random.default_rng(2).standard_normal((100, 50)))
        df = df.where(df > 0)  # create nans
        bools = df > 0
        mask = isna(df)
        expected = bools.astype(object).mask(mask)
        result = bools.mask(mask)
        tm.assert_frame_equal(result, expected)


def test_mask_stringdtype(frame_or_series):
    # GH 40824
    obj = DataFrame(
        {"A": ["foo", "bar", "baz", NA]},
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    filtered_obj = DataFrame(
        {"A": ["this", "that"]}, index=["id2", "id3"], dtype=StringDtype()
    )
    expected = DataFrame(
        {"A": [NA, "this", "that", NA]},
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    if frame_or_series is Series:
        obj = obj["A"]
        filtered_obj = filtered_obj["A"]
        expected = expected["A"]

    filter_ser = Series([False, True, True, False])
    result = obj.mask(filter_ser, filtered_obj)

    tm.assert_equal(result, expected)


def test_mask_where_dtype_timedelta():
    # https://github.com/pandas-dev/pandas/issues/39548
    df = DataFrame([Timedelta(i, unit="d") for i in range(5)])

    expected = DataFrame(np.full(5, np.nan, dtype="timedelta64[ns]"))
    tm.assert_frame_equal(df.mask(df.notna()), expected)

    expected = DataFrame(
        [np.nan, np.nan, np.nan, Timedelta("3 day"), Timedelta("4 day")]
    )
    tm.assert_frame_equal(df.where(df > Timedelta(2, unit="d")), expected)


def test_mask_return_dtype():
    # GH#50488
    ser = Series([0.0, 1.0, 2.0, 3.0], dtype=Float64Dtype())
    cond = ~ser.isna()
    other = Series([True, False, True, False])
    excepted = Series([1.0, 0.0, 1.0, 0.0], dtype=ser.dtype)
    result = ser.mask(cond, other)
    tm.assert_series_equal(result, excepted)


def test_mask_inplace_no_other():
    # GH#51685
    df = DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    cond = DataFrame({"a": [True, False], "b": [False, True]})
    df.mask(cond, inplace=True)
    expected = DataFrame({"a": [np.nan, 2], "b": ["x", np.nan]})
    tm.assert_frame_equal(df, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_qapply.py ===
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import H, XGate, IdentityGate
from sympy.physics.quantum.operator import Operator, IdentityOperator
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.spin import Jx, Jy, Jz, Jplus, Jminus, J2, JzKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.state import Ket
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.boson import BosonOp, BosonFockKet, BosonFockBra
from sympy.testing.pytest import warns_deprecated_sympy


j, jp, m, mp = symbols("j j' m m'")

z = JzKet(1, 0)
po = JzKet(1, 1)
mo = JzKet(1, -1)

A = Operator('A')


class Foo(Operator):
    def _apply_operator_JzKet(self, ket, **options):
        return ket


def test_basic():
    assert qapply(Jz*po) == hbar*po
    assert qapply(Jx*z) == hbar*po/sqrt(2) + hbar*mo/sqrt(2)
    assert qapply((Jplus + Jminus)*z/sqrt(2)) == hbar*po + hbar*mo
    assert qapply(Jz*(po + mo)) == hbar*po - hbar*mo
    assert qapply(Jz*po + Jz*mo) == hbar*po - hbar*mo
    assert qapply(Jminus*Jminus*po) == 2*hbar**2*mo
    assert qapply(Jplus**2*mo) == 2*hbar**2*po
    assert qapply(Jplus**2*Jminus**2*po) == 4*hbar**4*po


def test_extra():
    extra = z.dual*A*z
    assert qapply(Jz*po*extra) == hbar*po*extra
    assert qapply(Jx*z*extra) == (hbar*po/sqrt(2) + hbar*mo/sqrt(2))*extra
    assert qapply(
        (Jplus + Jminus)*z/sqrt(2)*extra) == hbar*po*extra + hbar*mo*extra
    assert qapply(Jz*(po + mo)*extra) == hbar*po*extra - hbar*mo*extra
    assert qapply(Jz*po*extra + Jz*mo*extra) == hbar*po*extra - hbar*mo*extra
    assert qapply(Jminus*Jminus*po*extra) == 2*hbar**2*mo*extra
    assert qapply(Jplus**2*mo*extra) == 2*hbar**2*po*extra
    assert qapply(Jplus**2*Jminus**2*po*extra) == 4*hbar**4*po*extra


def test_innerproduct():
    assert qapply(po.dual*Jz*po, ip_doit=False) == hbar*(po.dual*po)
    assert qapply(po.dual*Jz*po) == hbar


def test_zero():
    assert qapply(0) == 0
    assert qapply(Integer(0)) == 0


def test_commutator():
    assert qapply(Commutator(Jx, Jy)*Jz*po) == I*hbar**3*po
    assert qapply(Commutator(J2, Jz)*Jz*po) == 0
    assert qapply(Commutator(Jz, Foo('F'))*po) == 0
    assert qapply(Commutator(Foo('F'), Jz)*po) == 0


def test_anticommutator():
    assert qapply(AntiCommutator(Jz, Foo('F'))*po) == 2*hbar*po
    assert qapply(AntiCommutator(Foo('F'), Jz)*po) == 2*hbar*po


def test_outerproduct():
    e = Jz*(mo*po.dual)*Jz*po
    assert qapply(e) == -hbar**2*mo
    assert qapply(e, ip_doit=False) == -hbar**2*(po.dual*po)*mo
    assert qapply(e).doit() == -hbar**2*mo


def test_tensorproduct():
    a = BosonOp("a")
    b = BosonOp("b")
    ket1 = TensorProduct(BosonFockKet(1), BosonFockKet(2))
    ket2 = TensorProduct(BosonFockKet(0), BosonFockKet(0))
    ket3 = TensorProduct(BosonFockKet(0), BosonFockKet(2))
    bra1 = TensorProduct(BosonFockBra(0), BosonFockBra(0))
    bra2 = TensorProduct(BosonFockBra(1), BosonFockBra(2))
    assert qapply(TensorProduct(a, b ** 2) * ket1) == sqrt(2) * ket2
    assert qapply(TensorProduct(a, Dagger(b) * b) * ket1) == 2 * ket3
    assert qapply(bra1 * TensorProduct(a, b * b),
                  dagger=True) == sqrt(2) * bra2
    assert qapply(bra2 * ket1).doit() == S.One
    assert qapply(TensorProduct(a, b * b) * ket1) == sqrt(2) * ket2
    assert qapply(Dagger(TensorProduct(a, b * b) * ket1),
                  dagger=True) == sqrt(2) * Dagger(ket2)


def test_dagger():
    lhs = Dagger(Qubit(0))*Dagger(H(0))
    rhs = Dagger(Qubit(1))/sqrt(2) + Dagger(Qubit(0))/sqrt(2)
    assert qapply(lhs, dagger=True) == rhs


def test_issue_6073():
    x, y = symbols('x y', commutative=False)
    A = Ket(x, y)
    B = Operator('B')
    assert qapply(A) == A
    assert qapply(A.dual*B) == A.dual*B


def test_density():
    d = Density([Jz*mo, 0.5], [Jz*po, 0.5])
    assert qapply(d) == Density([-hbar*mo, 0.5], [hbar*po, 0.5])


def test_issue3044():
    expr1 = TensorProduct(Jz*JzKet(S(2),S.NegativeOne)/sqrt(2), Jz*JzKet(S.Half,S.Half))
    result = Mul(S.NegativeOne, Rational(1, 4), 2**S.Half, hbar**2)
    result *= TensorProduct(JzKet(2,-1), JzKet(S.Half,S.Half))
    assert qapply(expr1) == result


# Issue 24158: Tests whether qapply incorrectly evaluates some ket*op as op*ket
def test_issue24158_ket_times_op():
    P = BosonFockKet(0) * BosonOp("a") # undefined term
    # Does lhs._apply_operator_BosonOp(rhs) still evaluate ket*op as op*ket?
    assert qapply(P) == P   # qapply(P) -> BosonOp("a")*BosonFockKet(0) = 0 before fix
    P = Qubit(1) * XGate(0) # undefined term
    # Does rhs._apply_operator_Qubit(lhs) still evaluate ket*op as op*ket?
    assert qapply(P) == P   # qapply(P) -> Qubit(0) before fix
    P1 = Mul(QubitBra(0), Mul(QubitBra(0), Qubit(0)), XGate(0)) # legal expr <0| * (<1|*|1>) * X
    assert qapply(P1) == QubitBra(0) * XGate(0)     # qapply(P1) -> 0 before fix
    P1 = qapply(P1, dagger = True)  # unsatisfactorily -> <0|*X(0), expect <1| since dagger=True
    assert qapply(P1, dagger = True) == QubitBra(1) # qapply(P1, dagger=True) -> 0 before fix
    P2 = QubitBra(0) * (QubitBra(0) * Qubit(0)) * XGate(0) # 'forgot' to set brackets
    P2 = qapply(P2, dagger = True) # unsatisfactorily -> <0|*X(0), expect <1| since dagger=True
    assert P2 == QubitBra(1) # qapply(P1) -> 0 before fix
    # Pull Request 24237: IdentityOperator from the right without dagger=True option
    with warns_deprecated_sympy():
        assert qapply(QubitBra(1)*IdentityOperator()) == QubitBra(1)
        assert qapply(IdentityGate(0)*(Qubit(0) + Qubit(1))) == Qubit(0) + Qubit(1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_heuristicgcd.py ===
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ
from sympy.polys.heuristicgcd import heugcd


def test_heugcd_univariate_integers():
    R, x = ring("x", ZZ)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6

    h = x**2 + 3*x + 2

    cff = x**2 + 5*x + 4
    cfg = x + 3

    assert heugcd(f, g) == (h, cff, cfg)

    f = x**4 - 4
    g = x**4 + 4*x**2 + 4

    h = x**2 + 2

    cff = x**2 - 2
    cfg = x**2 + 2

    assert heugcd(f, g) == (h, cff, cfg)

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    h = 1

    cff = f
    cfg = g

    assert heugcd(f, g) == (h, cff, cfg)

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

    # TODO: assert heugcd(f, f.diff(x))[0] == g

    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760

    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1

    assert heugcd(f, g) == (h, cff, cfg)

def test_heugcd_multivariate_integers():
    R, x, y = ring("x,y", ZZ)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert heugcd(f, g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert heugcd(f, g) == (x + 1, 1, 2*x + 2)

    R, x, y, z, u = ring("x,y,z,u", ZZ)

    f, g = u**2 + 2*u + 1, 2*u + 2
    assert heugcd(f, g) == (u + 1, u + 1, 2)

    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1

    assert heugcd(f, g) == (h, cff, cfg)
    assert heugcd(g, f) == (h, cfg, cff)

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b = ring("x,y,z,u,v,a,b", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)

    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z = ring("x,y,z", ZZ)

    f, g, h = R.fateman_poly_F_2()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g

    R, x, y, z, t = ring("x,y,z,t", ZZ)

    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = heugcd(f, g)

    assert H == h and H*cff == f and H*cfg == g


def test_issue_10996():
    R, x, y, z = ring("x,y,z", ZZ)

    f = 12*x**6*y**7*z**3 - 3*x**4*y**9*z**3 + 12*x**3*y**5*z**4
    g = -48*x**7*y**8*z**3 + 12*x**5*y**10*z**3 - 48*x**5*y**7*z**2 + \
    36*x**4*y**7*z - 48*x**4*y**6*z**4 + 12*x**3*y**9*z**2 - 48*x**3*y**4 \
    - 9*x**2*y**9*z - 48*x**2*y**5*z**3 + 12*x*y**6 + 36*x*y**5*z**2 - 48*y**2*z

    H, cff, cfg = heugcd(f, g)

    assert H == 12*x**3*y**4 - 3*x*y**6 + 12*y**2*z
    assert H*cff == f and H*cfg == g


def test_issue_25793():
    R, x = ring("x", ZZ)
    f = x - 4851  # failure starts for values more than 4850
    g = f*(2*x + 1)
    H, cff, cfg = R.dup_zz_heu_gcd(f, g)
    assert H == f
    # needs a test for dmp, too, that fails in master before this change

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_specialpolys.py ===
"""Tests for functions for generating interesting polynomials. """

from sympy.core.add import Add
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import prime
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import permute_signs
from sympy.testing.pytest import raises

from sympy.polys.specialpolys import (
    swinnerton_dyer_poly,
    cyclotomic_poly,
    symmetric_poly,
    random_poly,
    interpolating_poly,
    fateman_poly_F_1,
    dmp_fateman_poly_F_1,
    fateman_poly_F_2,
    dmp_fateman_poly_F_2,
    fateman_poly_F_3,
    dmp_fateman_poly_F_3,
)

from sympy.abc import x, y, z


def test_swinnerton_dyer_poly():
    raises(ValueError, lambda: swinnerton_dyer_poly(0, x))

    assert swinnerton_dyer_poly(1, x, polys=True) == Poly(x**2 - 2)

    assert swinnerton_dyer_poly(1, x) == x**2 - 2
    assert swinnerton_dyer_poly(2, x) == x**4 - 10*x**2 + 1
    assert swinnerton_dyer_poly(
        3, x) == x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576
    # we only need to check that the polys arg works but
    # we may as well test that the roots are correct
    p = [sqrt(prime(i)) for i in range(1, 5)]
    assert str([i.n(3) for i in
        swinnerton_dyer_poly(4, polys=True).all_roots()]
        ) == str(sorted([Add(*i).n(3) for i in permute_signs(p)]))


def test_cyclotomic_poly():
    raises(ValueError, lambda: cyclotomic_poly(0, x))

    assert cyclotomic_poly(1, x, polys=True) == Poly(x - 1)

    assert cyclotomic_poly(1, x) == x - 1
    assert cyclotomic_poly(2, x) == x + 1
    assert cyclotomic_poly(3, x) == x**2 + x + 1
    assert cyclotomic_poly(4, x) == x**2 + 1
    assert cyclotomic_poly(5, x) == x**4 + x**3 + x**2 + x + 1
    assert cyclotomic_poly(6, x) == x**2 - x + 1


def test_symmetric_poly():
    raises(ValueError, lambda: symmetric_poly(-1, x, y, z))
    raises(ValueError, lambda: symmetric_poly(5, x, y, z))

    assert symmetric_poly(1, x, y, z, polys=True) == Poly(x + y + z)
    assert symmetric_poly(1, (x, y, z), polys=True) == Poly(x + y + z)

    assert symmetric_poly(0, x, y, z) == 1
    assert symmetric_poly(1, x, y, z) == x + y + z
    assert symmetric_poly(2, x, y, z) == x*y + x*z + y*z
    assert symmetric_poly(3, x, y, z) == x*y*z


def test_random_poly():
    poly = random_poly(x, 10, -100, 100, polys=False)

    assert Poly(poly).degree() == 10
    assert all(-100 <= coeff <= 100 for coeff in Poly(poly).coeffs()) is True

    poly = random_poly(x, 10, -100, 100, polys=True)

    assert poly.degree() == 10
    assert all(-100 <= coeff <= 100 for coeff in poly.coeffs()) is True


def test_interpolating_poly():
    x0, x1, x2, x3, y0, y1, y2, y3 = symbols('x:4, y:4')

    assert interpolating_poly(0, x) == 0
    assert interpolating_poly(1, x) == y0

    assert interpolating_poly(2, x) == \
        y0*(x - x1)/(x0 - x1) + y1*(x - x0)/(x1 - x0)

    assert interpolating_poly(3, x) == \
        y0*(x - x1)*(x - x2)/((x0 - x1)*(x0 - x2)) + \
        y1*(x - x0)*(x - x2)/((x1 - x0)*(x1 - x2)) + \
        y2*(x - x0)*(x - x1)/((x2 - x0)*(x2 - x1))

    assert interpolating_poly(4, x) == \
        y0*(x - x1)*(x - x2)*(x - x3)/((x0 - x1)*(x0 - x2)*(x0 - x3)) + \
        y1*(x - x0)*(x - x2)*(x - x3)/((x1 - x0)*(x1 - x2)*(x1 - x3)) + \
        y2*(x - x0)*(x - x1)*(x - x3)/((x2 - x0)*(x2 - x1)*(x2 - x3)) + \
        y3*(x - x0)*(x - x1)*(x - x2)/((x3 - x0)*(x3 - x1)*(x3 - x2))

    raises(ValueError, lambda:
        interpolating_poly(2, x, (x, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, x, (x + y, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, x + y, (x, 2), (1, 3)))
    raises(ValueError, lambda:
        interpolating_poly(2, 3, (4, 5), (6, 7)))
    raises(ValueError, lambda:
        interpolating_poly(2, 3, (4, 5), (6, 7, 8)))
    assert interpolating_poly(0, x, (1, 2), (3, 4)) == 0
    assert interpolating_poly(1, x, (1, 2), (3, 4)) == 3
    assert interpolating_poly(2, x, (1, 2), (3, 4)) == x + 2


def test_fateman_poly_F_1():
    f, g, h = fateman_poly_F_1(1)
    F, G, H = dmp_fateman_poly_F_1(1, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    f, g, h = fateman_poly_F_1(3)
    F, G, H = dmp_fateman_poly_F_1(3, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]


def test_fateman_poly_F_2():
    f, g, h = fateman_poly_F_2(1)
    F, G, H = dmp_fateman_poly_F_2(1, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    f, g, h = fateman_poly_F_2(3)
    F, G, H = dmp_fateman_poly_F_2(3, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]


def test_fateman_poly_F_3():
    f, g, h = fateman_poly_F_3(1)
    F, G, H = dmp_fateman_poly_F_3(1, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

    f, g, h = fateman_poly_F_3(3)
    F, G, H = dmp_fateman_poly_F_3(3, ZZ)

    assert [ t.rep.to_list() for t in [f, g, h] ] == [F, G, H]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\tests\test_lie_group.py ===
from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, sin, tan)

from sympy.solvers.ode import (classify_ode, checkinfsol, dsolve, infinitesimals)

from sympy.solvers.ode.subscheck import checkodesol

from sympy.testing.pytest import XFAIL


C1 = Symbol('C1')
x, y = symbols("x y")
f = Function('f')
xi = Function('xi')
eta = Function('eta')


def test_heuristic1():
    a, b, c, a4, a3, a2, a1, a0 = symbols("a b c a4 a3 a2 a1 a0")
    df = f(x).diff(x)
    eq = Eq(df, x**2*f(x))
    eq1 = f(x).diff(x) + a*f(x) - c*exp(b*x)
    eq2 = f(x).diff(x) + 2*x*f(x) - x*exp(-x**2)
    eq3 = (1 + 2*x)*df + 2 - 4*exp(-f(x))
    eq4 = f(x).diff(x) - (a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)**Rational(-1, 2)
    eq5 = x**2*df - f(x) + x**2*exp(x - (1/x))
    eqlist = [eq, eq1, eq2, eq3, eq4, eq5]

    i = infinitesimals(eq, hint='abaco1_simple')
    assert i == [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0},
        {eta(x, f(x)): f(x), xi(x, f(x)): 0},
        {eta(x, f(x)): 0, xi(x, f(x)): x**(-2)}]
    i1 = infinitesimals(eq1, hint='abaco1_simple')
    assert i1 == [{eta(x, f(x)): exp(-a*x), xi(x, f(x)): 0}]
    i2 = infinitesimals(eq2, hint='abaco1_simple')
    assert i2 == [{eta(x, f(x)): exp(-x**2), xi(x, f(x)): 0}]
    i3 = infinitesimals(eq3, hint='abaco1_simple')
    assert i3 == [{eta(x, f(x)): 0, xi(x, f(x)): 2*x + 1},
        {eta(x, f(x)): 0, xi(x, f(x)): 1/(exp(f(x)) - 2)}]
    i4 = infinitesimals(eq4, hint='abaco1_simple')
    assert i4 == [{eta(x, f(x)): 1, xi(x, f(x)): 0},
        {eta(x, f(x)): 0,
        xi(x, f(x)): sqrt(a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)}]
    i5 = infinitesimals(eq5, hint='abaco1_simple')
    assert i5 == [{xi(x, f(x)): 0, eta(x, f(x)): exp(-1/x)}]

    ilist = [i, i1, i2, i3, i4, i5]
    for eq, i in (zip(eqlist, ilist)):
        check = checkinfsol(eq, i)
        assert check[0]

    # This ODE can be solved by the Lie Group method, when there are
    # better assumptions
    eq6 = df - (f(x)/x)*(x*log(x**2/f(x)) + 2)
    i = infinitesimals(eq6, hint='abaco1_product')
    assert i == [{eta(x, f(x)): f(x)*exp(-x), xi(x, f(x)): 0}]
    assert checkinfsol(eq6, i)[0]

    eq7 = x*(f(x).diff(x)) + 1 - f(x)**2
    i = infinitesimals(eq7, hint='chi')
    assert checkinfsol(eq7, i)[0]


def test_heuristic3():
    a, b = symbols("a b")
    df = f(x).diff(x)

    eq = x**2*df + x*f(x) + f(x)**2 + x**2
    i = infinitesimals(eq, hint='bivariate')
    assert i == [{eta(x, f(x)): f(x), xi(x, f(x)): x}]
    assert checkinfsol(eq, i)[0]

    eq = x**2*(-f(x)**2 + df)- a*x**2*f(x) + 2 - a*x
    i = infinitesimals(eq, hint='bivariate')
    assert checkinfsol(eq, i)[0]


def test_heuristic_function_sum():
    eq = f(x).diff(x) - (3*(1 + x**2/f(x)**2)*atan(f(x)/x) + (1 - 2*f(x))/x +
       (1 - 3*f(x))*(x/f(x)**2))
    i = infinitesimals(eq, hint='function_sum')
    assert i == [{eta(x, f(x)): f(x)**(-2) + x**(-2), xi(x, f(x)): 0}]
    assert checkinfsol(eq, i)[0]


def test_heuristic_abaco2_similar():
    a, b = symbols("a b")
    F = Function('F')
    eq = f(x).diff(x) - F(a*x + b*f(x))
    i = infinitesimals(eq, hint='abaco2_similar')
    assert i == [{eta(x, f(x)): -a/b, xi(x, f(x)): 1}]
    assert checkinfsol(eq, i)[0]

    eq = f(x).diff(x) - (f(x)**2 / (sin(f(x) - x) - x**2 + 2*x*f(x)))
    i = infinitesimals(eq, hint='abaco2_similar')
    assert i == [{eta(x, f(x)): f(x)**2, xi(x, f(x)): f(x)**2}]
    assert checkinfsol(eq, i)[0]


def test_heuristic_abaco2_unique_unknown():

    a, b = symbols("a b")
    F = Function('F')
    eq = f(x).diff(x) - x**(a - 1)*(f(x)**(1 - b))*F(x**a/a + f(x)**b/b)
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert i == [{eta(x, f(x)): -f(x)*f(x)**(-b), xi(x, f(x)): x*x**(-a)}]
    assert checkinfsol(eq, i)[0]

    eq = f(x).diff(x) + tan(F(x**2 + f(x)**2) + atan(x/f(x)))
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert i == [{eta(x, f(x)): x, xi(x, f(x)): -f(x)}]
    assert checkinfsol(eq, i)[0]

    eq = (x*f(x).diff(x) + f(x) + 2*x)**2 -4*x*f(x) -4*x**2 -4*a
    i = infinitesimals(eq, hint='abaco2_unique_unknown')
    assert checkinfsol(eq, i)[0]


def test_heuristic_linear():
    a, b, m, n = symbols("a b m n")

    eq = x**(n*(m + 1) - m)*(f(x).diff(x)) - a*f(x)**n -b*x**(n*(m + 1))
    i = infinitesimals(eq, hint='linear')
    assert checkinfsol(eq, i)[0]


@XFAIL
def test_kamke():
    a, b, alpha, c = symbols("a b alpha c")
    eq = x**2*(a*f(x)**2+(f(x).diff(x))) + b*x**alpha + c
    i = infinitesimals(eq, hint='sum_function')  # XFAIL
    assert checkinfsol(eq, i)[0]


def test_user_infinitesimals():
    x = Symbol("x") # assuming x is real generates an error
    eq = x*(f(x).diff(x)) + 1 - f(x)**2
    sol = Eq(f(x), (C1 + x**2)/(C1 - x**2))
    infinitesimals = {'xi':sqrt(f(x) - 1)/sqrt(f(x) + 1), 'eta':0}
    assert dsolve(eq, hint='lie_group', **infinitesimals) == sol
    assert checkodesol(eq, sol) == (True, 0)


@XFAIL
def test_lie_group_issue15219():
    eqn = exp(f(x).diff(x)-f(x))
    assert 'lie_group' not in classify_ode(eqn, f(x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_dtypes.py ===
from datetime import timedelta

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    option_context,
)
import pandas._testing as tm


class TestDataFrameDataTypes:
    def test_empty_frame_dtypes(self):
        empty_df = DataFrame()
        tm.assert_series_equal(empty_df.dtypes, Series(dtype=object))

        nocols_df = DataFrame(index=[1, 2, 3])
        tm.assert_series_equal(nocols_df.dtypes, Series(dtype=object))

        norows_df = DataFrame(columns=list("abc"))
        tm.assert_series_equal(norows_df.dtypes, Series(object, index=list("abc")))

        norows_int_df = DataFrame(columns=list("abc")).astype(np.int32)
        tm.assert_series_equal(
            norows_int_df.dtypes, Series(np.dtype("int32"), index=list("abc"))
        )

        df = DataFrame({"a": 1, "b": True, "c": 1.0}, index=[1, 2, 3])
        ex_dtypes = Series({"a": np.int64, "b": np.bool_, "c": np.float64})
        tm.assert_series_equal(df.dtypes, ex_dtypes)

        # same but for empty slice of df
        tm.assert_series_equal(df[:0].dtypes, ex_dtypes)

    def test_datetime_with_tz_dtypes(self):
        tzframe = DataFrame(
            {
                "A": date_range("20130101", periods=3),
                "B": date_range("20130101", periods=3, tz="US/Eastern"),
                "C": date_range("20130101", periods=3, tz="CET"),
            }
        )
        tzframe.iloc[1, 1] = pd.NaT
        tzframe.iloc[1, 2] = pd.NaT
        result = tzframe.dtypes.sort_index()
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype("ns", "US/Eastern"),
                DatetimeTZDtype("ns", "CET"),
            ],
            ["A", "B", "C"],
        )

        tm.assert_series_equal(result, expected)

    def test_dtypes_are_correct_after_column_slice(self):
        # GH6525
        df = DataFrame(index=range(5), columns=list("abc"), dtype=np.float64)
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float64, "b": np.float64, "c": np.float64}),
        )
        tm.assert_series_equal(df.iloc[:, 2:].dtypes, Series({"c": np.float64}))
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float64, "b": np.float64, "c": np.float64}),
        )

    @pytest.mark.parametrize(
        "data",
        [pd.NA, True],
    )
    def test_dtypes_are_correct_after_groupby_last(self, data):
        # GH46409
        df = DataFrame(
            {"id": [1, 2, 3, 4], "test": [True, pd.NA, data, False]}
        ).convert_dtypes()
        result = df.groupby("id").last().test
        expected = df.set_index("id").test
        assert result.dtype == pd.BooleanDtype()
        tm.assert_series_equal(expected, result)

    def test_dtypes_gh8722(self, float_string_frame):
        float_string_frame["bool"] = float_string_frame["A"] > 0
        result = float_string_frame.dtypes
        expected = Series(
            {k: v.dtype for k, v in float_string_frame.items()}, index=result.index
        )
        tm.assert_series_equal(result, expected)

        # compat, GH 8722
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with option_context("use_inf_as_na", True):
                df = DataFrame([[1]])
                result = df.dtypes
                tm.assert_series_equal(result, Series({0: np.dtype("int64")}))

    def test_dtypes_timedeltas(self):
        df = DataFrame(
            {
                "A": Series(date_range("2012-1-1", periods=3, freq="D")),
                "B": Series([timedelta(days=i) for i in range(3)]),
            }
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("datetime64[ns]"), np.dtype("timedelta64[ns]")], index=list("AB")
        )
        tm.assert_series_equal(result, expected)

        df["C"] = df["A"] + df["B"]
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[ns]"),
                np.dtype("datetime64[ns]"),
            ],
            index=list("ABC"),
        )
        tm.assert_series_equal(result, expected)

        # mixed int types
        df["D"] = 1
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[ns]"),
                np.dtype("datetime64[ns]"),
                np.dtype("int64"),
            ],
            index=list("ABCD"),
        )
        tm.assert_series_equal(result, expected)

    def test_frame_apply_np_array_return_type(self, using_infer_string):
        # GH 35517
        df = DataFrame([["foo"]])
        result = df.apply(lambda col: np.array("bar"))
        expected = Series(np.array("bar"))
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_kronecker.py ===
from sympy.core.mod import Mod
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import (Matrix, eye)
from sympy.matrices import MatrixSymbol, Identity
from sympy.matrices.expressions import det, trace

from sympy.matrices.expressions.kronecker import (KroneckerProduct,
                                                  kronecker_product,
                                                  combine_kronecker)


mat1 = Matrix([[1, 2 * I], [1 + I, 3]])
mat2 = Matrix([[2 * I, 3], [4 * I, 2]])

i, j, k, n, m, o, p, x = symbols('i,j,k,n,m,o,p,x')
Z = MatrixSymbol('Z', n, n)
W = MatrixSymbol('W', m, m)
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', n, m)
C = MatrixSymbol('C', m, k)


def test_KroneckerProduct():
    assert isinstance(KroneckerProduct(A, B), KroneckerProduct)
    assert KroneckerProduct(A, B).subs(A, C) == KroneckerProduct(C, B)
    assert KroneckerProduct(A, C).shape == (n*m, m*k)
    assert (KroneckerProduct(A, C) + KroneckerProduct(-A, C)).is_ZeroMatrix
    assert (KroneckerProduct(W, Z) * KroneckerProduct(W.I, Z.I)).is_Identity


def test_KroneckerProduct_identity():
    assert KroneckerProduct(Identity(m), Identity(n)) == Identity(m*n)
    assert KroneckerProduct(eye(2), eye(3)) == eye(6)


def test_KroneckerProduct_explicit():
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    kp = KroneckerProduct(X, Y)
    assert kp.shape == (4, 4)
    assert kp.as_explicit() == Matrix(
        [
            [X[0, 0]*Y[0, 0], X[0, 0]*Y[0, 1], X[0, 1]*Y[0, 0], X[0, 1]*Y[0, 1]],
            [X[0, 0]*Y[1, 0], X[0, 0]*Y[1, 1], X[0, 1]*Y[1, 0], X[0, 1]*Y[1, 1]],
            [X[1, 0]*Y[0, 0], X[1, 0]*Y[0, 1], X[1, 1]*Y[0, 0], X[1, 1]*Y[0, 1]],
            [X[1, 0]*Y[1, 0], X[1, 0]*Y[1, 1], X[1, 1]*Y[1, 0], X[1, 1]*Y[1, 1]]
        ]
    )


def test_tensor_product_adjoint():
    assert KroneckerProduct(I*A, B).adjoint() == \
        -I*KroneckerProduct(A.adjoint(), B.adjoint())
    assert KroneckerProduct(mat1, mat2).adjoint() == \
        kronecker_product(mat1.adjoint(), mat2.adjoint())


def test_tensor_product_conjugate():
    assert KroneckerProduct(I*A, B).conjugate() == \
        -I*KroneckerProduct(A.conjugate(), B.conjugate())
    assert KroneckerProduct(mat1, mat2).conjugate() == \
        kronecker_product(mat1.conjugate(), mat2.conjugate())


def test_tensor_product_transpose():
    assert KroneckerProduct(I*A, B).transpose() == \
        I*KroneckerProduct(A.transpose(), B.transpose())
    assert KroneckerProduct(mat1, mat2).transpose() == \
        kronecker_product(mat1.transpose(), mat2.transpose())


def test_KroneckerProduct_is_associative():
    assert kronecker_product(A, kronecker_product(
        B, C)) == kronecker_product(kronecker_product(A, B), C)
    assert kronecker_product(A, kronecker_product(
        B, C)) == KroneckerProduct(A, B, C)


def test_KroneckerProduct_is_bilinear():
    assert kronecker_product(x*A, B) == x*kronecker_product(A, B)
    assert kronecker_product(A, x*B) == x*kronecker_product(A, B)


def test_KroneckerProduct_determinant():
    kp = kronecker_product(W, Z)
    assert det(kp) == det(W)**n * det(Z)**m


def test_KroneckerProduct_trace():
    kp = kronecker_product(W, Z)
    assert trace(kp) == trace(W)*trace(Z)


def test_KroneckerProduct_isnt_commutative():
    assert KroneckerProduct(A, B) != KroneckerProduct(B, A)
    assert KroneckerProduct(A, B).is_commutative is False


def test_KroneckerProduct_extracts_commutative_part():
    assert kronecker_product(x * A, 2 * B) == x * \
        2 * KroneckerProduct(A, B)


def test_KroneckerProduct_inverse():
    kp = kronecker_product(W, Z)
    assert kp.inverse() == kronecker_product(W.inverse(), Z.inverse())


def test_KroneckerProduct_combine_add():
    kp1 = kronecker_product(A, B)
    kp2 = kronecker_product(C, W)
    assert combine_kronecker(kp1*kp2) == kronecker_product(A*C, B*W)


def test_KroneckerProduct_combine_mul():
    X = MatrixSymbol('X', m, n)
    Y = MatrixSymbol('Y', m, n)
    kp1 = kronecker_product(A, X)
    kp2 = kronecker_product(B, Y)
    assert combine_kronecker(kp1+kp2) == kronecker_product(A+B, X+Y)


def test_KroneckerProduct_combine_pow():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    assert combine_kronecker(KroneckerProduct(
        X, Y)**x) == KroneckerProduct(X**x, Y**x)
    assert combine_kronecker(x * KroneckerProduct(X, Y)
                             ** 2) == x * KroneckerProduct(X**2, Y**2)
    assert combine_kronecker(
        x * (KroneckerProduct(X, Y)**2) * KroneckerProduct(A, B)) == x * KroneckerProduct(X**2 * A, Y**2 * B)
    # cannot simplify because of non-square arguments to kronecker product:
    assert combine_kronecker(KroneckerProduct(A, B.T) ** m) == KroneckerProduct(A, B.T) ** m


def test_KroneckerProduct_expand():
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)

    assert KroneckerProduct(X + Y, Y + Z).expand(kroneckerproduct=True) == \
        KroneckerProduct(X, Y) + KroneckerProduct(X, Z) + \
        KroneckerProduct(Y, Y) + KroneckerProduct(Y, Z)

def test_KroneckerProduct_entry():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', o, p)

    assert KroneckerProduct(A, B)._entry(i, j) == A[Mod(floor(i/o), n), Mod(floor(j/p), m)]*B[Mod(i, o), Mod(j, p)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\test_dimensions.py ===
from sympy.physics.units.systems.si import dimsys_SI

from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos)
from sympy.physics.units.dimensions import Dimension
from sympy.physics.units.definitions.dimension_definitions import (
    length, time, mass, force, pressure, angle
)
from sympy.physics.units import foot
from sympy.testing.pytest import raises


def test_Dimension_definition():
    assert dimsys_SI.get_dimensional_dependencies(length) == {length: 1}
    assert length.name == Symbol("length")
    assert length.symbol == Symbol("L")

    halflength = sqrt(length)
    assert dimsys_SI.get_dimensional_dependencies(halflength) == {length: S.Half}


def test_Dimension_error_definition():
    # tuple with more or less than two entries
    raises(TypeError, lambda: Dimension(("length", 1, 2)))
    raises(TypeError, lambda: Dimension(["length"]))

    # non-number power
    raises(TypeError, lambda: Dimension({"length": "a"}))

    # non-number with named argument
    raises(TypeError, lambda: Dimension({"length": (1, 2)}))

    # symbol should by Symbol or str
    raises(AssertionError, lambda: Dimension("length", symbol=1))


def test_str():
    assert str(Dimension("length")) == "Dimension(length)"
    assert str(Dimension("length", "L")) == "Dimension(length, L)"


def test_Dimension_properties():
    assert dimsys_SI.is_dimensionless(length) is False
    assert dimsys_SI.is_dimensionless(length/length) is True
    assert dimsys_SI.is_dimensionless(Dimension("undefined")) is False

    assert length.has_integer_powers(dimsys_SI) is True
    assert (length**(-1)).has_integer_powers(dimsys_SI) is True
    assert (length**1.5).has_integer_powers(dimsys_SI) is False


def test_Dimension_add_sub():
    assert length + length == length
    assert length - length == length
    assert -length == length

    raises(TypeError, lambda: length + foot)
    raises(TypeError, lambda: foot + length)
    raises(TypeError, lambda: length - foot)
    raises(TypeError, lambda: foot - length)

    # issue 14547 - only raise error for dimensional args; allow
    # others to pass
    x = Symbol('x')
    e = length + x
    assert e == x + length and e.is_Add and set(e.args) == {length, x}
    e = length + 1
    assert e == 1 + length == 1 - length and e.is_Add and set(e.args) == {length, 1}

    assert dimsys_SI.get_dimensional_dependencies(mass * length / time**2 + force) == \
            {length: 1, mass: 1, time: -2}
    assert dimsys_SI.get_dimensional_dependencies(mass * length / time**2 + force -
                                                   pressure * length**2) == \
            {length: 1, mass: 1, time: -2}

    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(mass * length / time**2 + pressure))

def test_Dimension_mul_div_exp():
    assert 2*length == length*2 == length/2 == length
    assert 2/length == 1/length
    x = Symbol('x')
    m = x*length
    assert m == length*x and m.is_Mul and set(m.args) == {x, length}
    d = x/length
    assert d == x*length**-1 and d.is_Mul and set(d.args) == {x, 1/length}
    d = length/x
    assert d == length*x**-1 and d.is_Mul and set(d.args) == {1/x, length}

    velo = length / time

    assert (length * length) == length ** 2

    assert dimsys_SI.get_dimensional_dependencies(length * length) == {length: 2}
    assert dimsys_SI.get_dimensional_dependencies(length ** 2) == {length: 2}
    assert dimsys_SI.get_dimensional_dependencies(length * time) == {length: 1, time: 1}
    assert dimsys_SI.get_dimensional_dependencies(velo) == {length: 1, time: -1}
    assert dimsys_SI.get_dimensional_dependencies(velo ** 2) == {length: 2, time: -2}

    assert dimsys_SI.get_dimensional_dependencies(length / length) == {}
    assert dimsys_SI.get_dimensional_dependencies(velo / length * time) == {}
    assert dimsys_SI.get_dimensional_dependencies(length ** -1) == {length: -1}
    assert dimsys_SI.get_dimensional_dependencies(velo ** -1.5) == {length: -1.5, time: 1.5}

    length_a = length**"a"
    assert dimsys_SI.get_dimensional_dependencies(length_a) == {length: Symbol("a")}

    assert dimsys_SI.get_dimensional_dependencies(length**pi) == {length: pi}
    assert dimsys_SI.get_dimensional_dependencies(length**(length/length)) == {length: Dimension(1)}

    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(length**length))

    assert length != 1
    assert length / length != 1

    length_0 = length ** 0
    assert dimsys_SI.get_dimensional_dependencies(length_0) == {}

    # issue 18738
    a = Symbol('a')
    b = Symbol('b')
    c = sqrt(a**2 + b**2)
    c_dim = c.subs({a: length, b: length})
    assert dimsys_SI.equivalent_dims(c_dim, length)

def test_Dimension_functions():
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(cos(length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(acos(angle)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(atan2(length, time)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(100, length)))
    raises(TypeError, lambda: dimsys_SI.get_dimensional_dependencies(log(length, 10)))

    assert dimsys_SI.get_dimensional_dependencies(pi) == {}

    assert dimsys_SI.get_dimensional_dependencies(cos(1)) == {}
    assert dimsys_SI.get_dimensional_dependencies(cos(angle)) == {}

    assert dimsys_SI.get_dimensional_dependencies(atan2(length, length)) == {}

    assert dimsys_SI.get_dimensional_dependencies(log(length / length, length / length)) == {}

    assert dimsys_SI.get_dimensional_dependencies(Abs(length)) == {length: 1}
    assert dimsys_SI.get_dimensional_dependencies(Abs(length / length)) == {}

    assert dimsys_SI.get_dimensional_dependencies(sqrt(-1)) == {}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\tests\test_interval_membership.py ===
from sympy.core.symbol import Symbol
from sympy.plotting.intervalmath import interval
from sympy.plotting.intervalmath.interval_membership import intervalMembership
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.testing.pytest import raises


def test_creation():
    assert intervalMembership(True, True)
    raises(TypeError, lambda: intervalMembership(True))
    raises(TypeError, lambda: intervalMembership(True, True, True))


def test_getitem():
    a = intervalMembership(True, False)
    assert a[0] is True
    assert a[1] is False
    raises(IndexError, lambda: a[2])


def test_str():
    a = intervalMembership(True, False)
    assert str(a) == 'intervalMembership(True, False)'
    assert repr(a) == 'intervalMembership(True, False)'


def test_equivalence():
    a = intervalMembership(True, True)
    b = intervalMembership(True, False)
    assert (a == b) is False
    assert (a != b) is True

    a = intervalMembership(True, False)
    b = intervalMembership(True, False)
    assert (a == b) is True
    assert (a != b) is False


def test_not():
    x = Symbol('x')

    r1 = x > -1
    r2 = x <= -1

    i = interval

    f1 = experimental_lambdify((x,), r1)
    f2 = experimental_lambdify((x,), r2)

    tt = i(-0.1, 0.1, is_valid=True)
    tn = i(-0.1, 0.1, is_valid=None)
    tf = i(-0.1, 0.1, is_valid=False)

    assert f1(tt) == ~f2(tt)
    assert f1(tn) == ~f2(tn)
    assert f1(tf) == ~f2(tf)

    nt = i(0.9, 1.1, is_valid=True)
    nn = i(0.9, 1.1, is_valid=None)
    nf = i(0.9, 1.1, is_valid=False)

    assert f1(nt) == ~f2(nt)
    assert f1(nn) == ~f2(nn)
    assert f1(nf) == ~f2(nf)

    ft = i(1.9, 2.1, is_valid=True)
    fn = i(1.9, 2.1, is_valid=None)
    ff = i(1.9, 2.1, is_valid=False)

    assert f1(ft) == ~f2(ft)
    assert f1(fn) == ~f2(fn)
    assert f1(ff) == ~f2(ff)


def test_boolean():
    # There can be 9*9 test cases in full mapping of the cartesian product.
    # But we only consider 3*3 cases for simplicity.
    s = [
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(True, True)
    ]

    # Reduced tests for 'And'
    a1 = [
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(None, None),
        intervalMembership(False, False),
        intervalMembership(None, None),
        intervalMembership(True, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        for j in range(len(s)):
            assert s[i] & s[j] == next(a1_iter)

    # Reduced tests for 'Or'
    a1 = [
        intervalMembership(False, False),
        intervalMembership(None, False),
        intervalMembership(True, False),
        intervalMembership(None, False),
        intervalMembership(None, None),
        intervalMembership(True, None),
        intervalMembership(True, False),
        intervalMembership(True, None),
        intervalMembership(True, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        for j in range(len(s)):
            assert s[i] | s[j] == next(a1_iter)

    # Reduced tests for 'Xor'
    a1 = [
        intervalMembership(False, False),
        intervalMembership(None, False),
        intervalMembership(True, False),
        intervalMembership(None, False),
        intervalMembership(None, None),
        intervalMembership(None, None),
        intervalMembership(True, False),
        intervalMembership(None, None),
        intervalMembership(False, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        for j in range(len(s)):
            assert s[i] ^ s[j] == next(a1_iter)

    # Reduced tests for 'Not'
    a1 = [
        intervalMembership(True, False),
        intervalMembership(None, None),
        intervalMembership(False, True)
    ]
    a1_iter = iter(a1)
    for i in range(len(s)):
        assert ~s[i] == next(a1_iter)


def test_boolean_errors():
    a = intervalMembership(True, True)
    raises(ValueError, lambda: a & 1)
    raises(ValueError, lambda: a | 1)
    raises(ValueError, lambda: a ^ 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\typing\tests\data\pass\mod.py ===
import numpy as np

f8 = np.float64(1)
i8 = np.int64(1)
u8 = np.uint64(1)

f4 = np.float32(1)
i4 = np.int32(1)
u4 = np.uint32(1)

td = np.timedelta64(1, "D")
b_ = np.bool(1)

b = bool(1)
f = float(1)
i = int(1)

AR = np.array([1], dtype=np.bool)
AR.setflags(write=False)

AR2 = np.array([1], dtype=np.timedelta64)
AR2.setflags(write=False)

# Time structures

td % td
td % AR2
AR2 % td

divmod(td, td)
divmod(td, AR2)
divmod(AR2, td)

# Bool

b_ % b
b_ % i
b_ % f
b_ % b_
b_ % i8
b_ % u8
b_ % f8
b_ % AR

divmod(b_, b)
divmod(b_, i)
divmod(b_, f)
divmod(b_, b_)
divmod(b_, i8)
divmod(b_, u8)
divmod(b_, f8)
divmod(b_, AR)

b % b_
i % b_
f % b_
b_ % b_
i8 % b_
u8 % b_
f8 % b_
AR % b_

divmod(b, b_)
divmod(i, b_)
divmod(f, b_)
divmod(b_, b_)
divmod(i8, b_)
divmod(u8, b_)
divmod(f8, b_)
divmod(AR, b_)

# int

i8 % b
i8 % i
i8 % f
i8 % i8
i8 % f8
i4 % i8
i4 % f8
i4 % i4
i4 % f4
i8 % AR

divmod(i8, b)
divmod(i8, i)
divmod(i8, f)
divmod(i8, i8)
divmod(i8, f8)
divmod(i8, i4)
divmod(i8, f4)
divmod(i4, i4)
divmod(i4, f4)
divmod(i8, AR)

b % i8
i % i8
f % i8
i8 % i8
f8 % i8
i8 % i4
f8 % i4
i4 % i4
f4 % i4
AR % i8

divmod(b, i8)
divmod(i, i8)
divmod(f, i8)
divmod(i8, i8)
divmod(f8, i8)
divmod(i4, i8)
divmod(f4, i8)
divmod(i4, i4)
divmod(f4, i4)
divmod(AR, i8)

# float

f8 % b
f8 % i
f8 % f
i8 % f4
f4 % f4
f8 % AR

divmod(f8, b)
divmod(f8, i)
divmod(f8, f)
divmod(f8, f8)
divmod(f8, f4)
divmod(f4, f4)
divmod(f8, AR)

b % f8
i % f8
f % f8
f8 % f8
f8 % f8
f4 % f4
AR % f8

divmod(b, f8)
divmod(i, f8)
divmod(f, f8)
divmod(f8, f8)
divmod(f4, f8)
divmod(f4, f4)
divmod(AR, f8)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_join.py ===
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    BDay,
    BMonthEnd,
)


class TestJoin:
    def test_does_not_convert_mixed_integer(self):
        df = DataFrame(np.ones((3, 2)), columns=date_range("2020-01-01", periods=2))
        cols = df.columns.join(df.index, how="outer")
        joined = cols.join(df.columns)
        assert cols.dtype == np.dtype("O")
        assert cols.dtype == joined.dtype
        tm.assert_numpy_array_equal(cols.values, joined.values)

    def test_join_self(self, join_type):
        index = date_range("1/1/2000", periods=10)
        joined = index.join(index, how=join_type)
        assert index is joined

    def test_join_with_period_index(self, join_type):
        df = DataFrame(
            np.ones((10, 2)),
            index=date_range("2020-01-01", periods=10),
            columns=period_range("2020-01-01", periods=2),
        )
        s = df.iloc[:5, 0]

        expected = df.columns.astype("O").join(s.index, how=join_type)
        result = df.columns.join(s.index, how=join_type)
        tm.assert_index_equal(expected, result)

    def test_join_object_index(self):
        rng = date_range("1/1/2000", periods=10)
        idx = Index(["a", "b", "c", "d"])

        result = rng.join(idx, how="outer")
        assert isinstance(result[0], Timestamp)

    def test_join_utc_convert(self, join_type):
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        left = rng.tz_convert("US/Eastern")
        right = rng.tz_convert("Europe/Berlin")

        result = left.join(left[:-5], how=join_type)
        assert isinstance(result, DatetimeIndex)
        assert result.tz == left.tz

        result = left.join(right[:-5], how=join_type)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is timezone.utc

    def test_datetimeindex_union_join_empty(self, sort):
        dti = date_range(start="1/1/2001", end="2/1/2001", freq="D")
        empty = Index([])

        result = dti.union(empty, sort=sort)
        expected = dti.astype("O")
        tm.assert_index_equal(result, expected)

        result = dti.join(empty)
        assert isinstance(result, DatetimeIndex)
        tm.assert_index_equal(result, dti)

    def test_join_nonunique(self):
        idx1 = to_datetime(["2012-11-06 16:00:11.477563", "2012-11-06 16:00:11.477563"])
        idx2 = to_datetime(["2012-11-06 15:11:09.006507", "2012-11-06 15:11:09.006507"])
        rs = idx1.join(idx2, how="outer")
        assert rs.is_monotonic_increasing

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_outer_join(self, freq):
        # should just behave as union
        start, end = datetime(2009, 1, 1), datetime(2010, 1, 1)
        rng = date_range(start=start, end=end, freq=freq)

        # overlapping
        left = rng[:10]
        right = rng[5:10]

        the_join = left.join(right, how="outer")
        assert isinstance(the_join, DatetimeIndex)

        # non-overlapping, gap in middle
        left = rng[:5]
        right = rng[10:]

        the_join = left.join(right, how="outer")
        assert isinstance(the_join, DatetimeIndex)
        assert the_join.freq is None

        # non-overlapping, no gap
        left = rng[:5]
        right = rng[5:10]

        the_join = left.join(right, how="outer")
        assert isinstance(the_join, DatetimeIndex)

        # overlapping, but different offset
        other = date_range(start, end, freq=BMonthEnd())

        the_join = rng.join(other, how="outer")
        assert isinstance(the_join, DatetimeIndex)
        assert the_join.freq is None

    def test_naive_aware_conflicts(self):
        start, end = datetime(2009, 1, 1), datetime(2010, 1, 1)
        naive = date_range(start, end, freq=BDay(), tz=None)
        aware = date_range(start, end, freq=BDay(), tz="Asia/Hong_Kong")

        msg = "tz-naive.*tz-aware"
        with pytest.raises(TypeError, match=msg):
            naive.join(aware)

        with pytest.raises(TypeError, match=msg):
            aware.join(naive)

    @pytest.mark.parametrize("tz", [None, "US/Pacific"])
    def test_join_preserves_freq(self, tz):
        # GH#32157
        dti = date_range("2016-01-01", periods=10, tz=tz)
        result = dti[:5].join(dti[5:], how="outer")
        assert result.freq == dti.freq
        tm.assert_index_equal(result, dti)

        result = dti[:5].join(dti[6:], how="outer")
        assert result.freq is None
        expected = dti.delete(5)
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_combine_first.py ===
from datetime import datetime

import numpy as np

import pandas as pd
from pandas import (
    Period,
    Series,
    date_range,
    period_range,
    to_datetime,
)
import pandas._testing as tm


class TestCombineFirst:
    def test_combine_first_period_datetime(self):
        # GH#3367
        didx = date_range(start="1950-01-31", end="1950-07-31", freq="ME")
        pidx = period_range(start=Period("1950-1"), end=Period("1950-7"), freq="M")
        # check to be consistent with DatetimeIndex
        for idx in [didx, pidx]:
            a = Series([1, np.nan, np.nan, 4, 5, np.nan, 7], index=idx)
            b = Series([9, 9, 9, 9, 9, 9, 9], index=idx)

            result = a.combine_first(b)
            expected = Series([1, 9, 9, 4, 5, 9, 7], index=idx, dtype=np.float64)
            tm.assert_series_equal(result, expected)

    def test_combine_first_name(self, datetime_series):
        result = datetime_series.combine_first(datetime_series[:5])
        assert result.name == datetime_series.name

    def test_combine_first(self):
        values = np.arange(20, dtype=np.float64)
        series = Series(values, index=np.arange(20, dtype=np.int64))

        series_copy = series * 2
        series_copy[::2] = np.nan

        # nothing used from the input
        combined = series.combine_first(series_copy)

        tm.assert_series_equal(combined, series)

        # Holes filled from input
        combined = series_copy.combine_first(series)
        assert np.isfinite(combined).all()

        tm.assert_series_equal(combined[::2], series[::2])
        tm.assert_series_equal(combined[1::2], series_copy[1::2])

        # mixed types
        index = pd.Index([str(i) for i in range(20)])
        floats = Series(np.random.default_rng(2).standard_normal(20), index=index)
        strings = Series([str(i) for i in range(10)], index=index[::2], dtype=object)

        combined = strings.combine_first(floats)

        tm.assert_series_equal(strings, combined.loc[index[::2]])
        tm.assert_series_equal(floats[1::2].astype(object), combined.loc[index[1::2]])

        # corner case
        ser = Series([1.0, 2, 3], index=[0, 1, 2])
        empty = Series([], index=[], dtype=object)
        msg = "The behavior of array concatenation with empty entries is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.combine_first(empty)
        ser.index = ser.index.astype("O")
        tm.assert_series_equal(ser, result)

    def test_combine_first_dt64(self, unit):
        s0 = to_datetime(Series(["2010", np.nan])).dt.as_unit(unit)
        s1 = to_datetime(Series([np.nan, "2011"])).dt.as_unit(unit)
        rs = s0.combine_first(s1)
        xp = to_datetime(Series(["2010", "2011"])).dt.as_unit(unit)
        tm.assert_series_equal(rs, xp)

        s0 = to_datetime(Series(["2010", np.nan])).dt.as_unit(unit)
        s1 = Series([np.nan, "2011"])
        rs = s0.combine_first(s1)

        xp = Series([datetime(2010, 1, 1), "2011"], dtype="datetime64[ns]")

        tm.assert_series_equal(rs, xp)

    def test_combine_first_dt_tz_values(self, tz_naive_fixture):
        ser1 = Series(
            pd.DatetimeIndex(["20150101", "20150102", "20150103"], tz=tz_naive_fixture),
            name="ser1",
        )
        ser2 = Series(
            pd.DatetimeIndex(["20160514", "20160515", "20160516"], tz=tz_naive_fixture),
            index=[2, 3, 4],
            name="ser2",
        )
        result = ser1.combine_first(ser2)
        exp_vals = pd.DatetimeIndex(
            ["20150101", "20150102", "20150103", "20160515", "20160516"],
            tz=tz_naive_fixture,
        )
        exp = Series(exp_vals, name="ser1")
        tm.assert_series_equal(exp, result)

    def test_combine_first_timezone_series_with_empty_series(self):
        # GH 41800
        time_index = date_range(
            datetime(2021, 1, 1, 1),
            datetime(2021, 1, 1, 10),
            freq="h",
            tz="Europe/Rome",
        )
        s1 = Series(range(10), index=time_index)
        s2 = Series(index=time_index)
        msg = "The behavior of array concatenation with empty entries is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s1.combine_first(s2)
        tm.assert_series_equal(result, s1)

    def test_combine_first_preserves_dtype(self):
        # GH51764
        s1 = Series([1666880195890293744, 1666880195890293837])
        s2 = Series([1, 2, 3])
        result = s1.combine_first(s2)
        expected = Series([1666880195890293744, 1666880195890293837, 3])
        tm.assert_series_equal(result, expected)

    def test_combine_mixed_timezone(self):
        # GH 26283
        uniform_tz = Series({pd.Timestamp("2019-05-01", tz="UTC"): 1.0})
        multi_tz = Series(
            {
                pd.Timestamp("2019-05-01 01:00:00+0100", tz="Europe/London"): 2.0,
                pd.Timestamp("2019-05-02", tz="UTC"): 3.0,
            }
        )

        result = uniform_tz.combine_first(multi_tz)
        expected = Series(
            [1.0, 3.0],
            index=pd.Index(
                [
                    pd.Timestamp("2019-05-01 00:00:00+00:00", tz="UTC"),
                    pd.Timestamp("2019-05-02 00:00:00+00:00", tz="UTC"),
                ],
                dtype="object",
            ),
        )
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_timedeltas.py ===
import re

import numpy as np
import pytest

from pandas._libs.tslibs.timedeltas import (
    array_to_timedelta64,
    delta_to_nanoseconds,
    ints_to_pytimedelta,
)

from pandas import (
    Timedelta,
    offsets,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "obj,expected",
    [
        (np.timedelta64(14, "D"), 14 * 24 * 3600 * 1e9),
        (Timedelta(minutes=-7), -7 * 60 * 1e9),
        (Timedelta(minutes=-7).to_pytimedelta(), -7 * 60 * 1e9),
        (Timedelta(seconds=1234e-9), 1234),  # GH43764, GH40946
        (
            Timedelta(seconds=1e-9, milliseconds=1e-5, microseconds=1e-1),
            111,
        ),  # GH43764
        (
            Timedelta(days=1, seconds=1e-9, milliseconds=1e-5, microseconds=1e-1),
            24 * 3600e9 + 111,
        ),  # GH43764
        (offsets.Nano(125), 125),
    ],
)
def test_delta_to_nanoseconds(obj, expected):
    result = delta_to_nanoseconds(obj)
    assert result == expected


def test_delta_to_nanoseconds_error():
    obj = np.array([123456789], dtype="m8[ns]")

    with pytest.raises(TypeError, match="<class 'numpy.ndarray'>"):
        delta_to_nanoseconds(obj)

    with pytest.raises(TypeError, match="float"):
        delta_to_nanoseconds(1.5)
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(1)
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(np.int64(2))
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(np.int32(3))


def test_delta_to_nanoseconds_td64_MY_raises():
    msg = (
        "delta_to_nanoseconds does not support Y or M units, "
        "as their duration in nanoseconds is ambiguous"
    )

    td = np.timedelta64(1234, "Y")

    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)

    td = np.timedelta64(1234, "M")

    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)


@pytest.mark.parametrize("unit", ["Y", "M"])
def test_unsupported_td64_unit_raises(unit):
    # GH 52806
    with pytest.raises(
        ValueError,
        match=f"Unit {unit} is not supported. "
        "Only unambiguous timedelta values durations are supported. "
        "Allowed units are 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'",
    ):
        Timedelta(np.timedelta64(1, unit))


def test_huge_nanoseconds_overflow():
    # GH 32402
    assert delta_to_nanoseconds(Timedelta(1e10)) == 1e10
    assert delta_to_nanoseconds(Timedelta(nanoseconds=1e10)) == 1e10


@pytest.mark.parametrize(
    "kwargs", [{"Seconds": 1}, {"seconds": 1, "Nanoseconds": 1}, {"Foo": 2}]
)
def test_kwarg_assertion(kwargs):
    err_message = (
        "cannot construct a Timedelta from the passed arguments, "
        "allowed keywords are "
        "[weeks, days, hours, minutes, seconds, "
        "milliseconds, microseconds, nanoseconds]"
    )

    with pytest.raises(ValueError, match=re.escape(err_message)):
        Timedelta(**kwargs)


class TestArrayToTimedelta64:
    def test_array_to_timedelta64_string_with_unit_2d_raises(self):
        # check the 'unit is not None and errors != "coerce"' path
        #  in array_to_timedelta64 raises correctly with 2D values
        values = np.array([["1", 2], [3, "4"]], dtype=object)
        with pytest.raises(ValueError, match="unit must not be specified"):
            array_to_timedelta64(values, unit="s")

    def test_array_to_timedelta64_non_object_raises(self):
        # check we raise, not segfault
        values = np.arange(5)

        msg = "'values' must have object dtype"
        with pytest.raises(TypeError, match=msg):
            array_to_timedelta64(values)


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_ints_to_pytimedelta(unit):
    # tests for non-nanosecond cases
    arr = np.arange(6, dtype=np.int64).view(f"m8[{unit}]")

    res = ints_to_pytimedelta(arr, box=False)
    # For non-nanosecond, .astype(object) gives pytimedelta objects
    #  instead of integers
    expected = arr.astype(object)
    tm.assert_numpy_array_equal(res, expected)

    res = ints_to_pytimedelta(arr, box=True)
    expected = np.array([Timedelta(x) for x in arr], dtype=object)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize("unit", ["Y", "M", "ps", "fs", "as"])
def test_ints_to_pytimedelta_unsupported(unit):
    arr = np.arange(6, dtype=np.int64).view(f"m8[{unit}]")

    with pytest.raises(NotImplementedError, match=r"\d{1,2}"):
        ints_to_pytimedelta(arr, box=False)
    msg = "Only resolutions 's', 'ms', 'us', 'ns' are supported"
    with pytest.raises(NotImplementedError, match=msg):
        ints_to_pytimedelta(arr, box=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_partial_indexing.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    IndexSlice,
    MultiIndex,
    date_range,
)
import pandas._testing as tm


@pytest.fixture
def df():
    #                        c1
    # 2016-01-01 00:00:00 a   0
    #                     b   1
    #                     c   2
    # 2016-01-01 12:00:00 a   3
    #                     b   4
    #                     c   5
    # 2016-01-02 00:00:00 a   6
    #                     b   7
    #                     c   8
    # 2016-01-02 12:00:00 a   9
    #                     b  10
    #                     c  11
    # 2016-01-03 00:00:00 a  12
    #                     b  13
    #                     c  14
    dr = date_range("2016-01-01", "2016-01-03", freq="12h")
    abc = ["a", "b", "c"]
    mi = MultiIndex.from_product([dr, abc])
    frame = DataFrame({"c1": range(15)}, index=mi)
    return frame


def test_partial_string_matching_single_index(df):
    # partial string matching on a single index
    for df_swap in [df.swaplevel(), df.swaplevel(0), df.swaplevel(0, 1)]:
        df_swap = df_swap.sort_index()
        just_a = df_swap.loc["a"]
        result = just_a.loc["2016-01-01"]
        expected = df.loc[IndexSlice[:, "a"], :].iloc[0:2]
        expected.index = expected.index.droplevel(1)
        tm.assert_frame_equal(result, expected)


def test_get_loc_partial_timestamp_multiindex(df):
    mi = df.index
    key = ("2016-01-01", "a")
    loc = mi.get_loc(key)

    expected = np.zeros(len(mi), dtype=bool)
    expected[[0, 3]] = True
    tm.assert_numpy_array_equal(loc, expected)

    key2 = ("2016-01-02", "a")
    loc2 = mi.get_loc(key2)
    expected2 = np.zeros(len(mi), dtype=bool)
    expected2[[6, 9]] = True
    tm.assert_numpy_array_equal(loc2, expected2)

    key3 = ("2016-01", "a")
    loc3 = mi.get_loc(key3)
    expected3 = np.zeros(len(mi), dtype=bool)
    expected3[mi.get_level_values(1).get_loc("a")] = True
    tm.assert_numpy_array_equal(loc3, expected3)

    key4 = ("2016", "a")
    loc4 = mi.get_loc(key4)
    expected4 = expected3
    tm.assert_numpy_array_equal(loc4, expected4)

    # non-monotonic
    taker = np.arange(len(mi), dtype=np.intp)
    taker[::2] = taker[::-2]
    mi2 = mi.take(taker)
    loc5 = mi2.get_loc(key)
    expected5 = np.zeros(len(mi2), dtype=bool)
    expected5[[3, 14]] = True
    tm.assert_numpy_array_equal(loc5, expected5)


def test_partial_string_timestamp_multiindex(df):
    # GH10331
    df_swap = df.swaplevel(0, 1).sort_index()
    SLC = IndexSlice

    # indexing with IndexSlice
    result = df.loc[SLC["2016-01-01":"2016-02-01", :], :]
    expected = df
    tm.assert_frame_equal(result, expected)

    # match on secondary index
    result = df_swap.loc[SLC[:, "2016-01-01":"2016-01-01"], :]
    expected = df_swap.iloc[[0, 1, 5, 6, 10, 11]]
    tm.assert_frame_equal(result, expected)

    # partial string match on year only
    result = df.loc["2016"]
    expected = df
    tm.assert_frame_equal(result, expected)

    # partial string match on date
    result = df.loc["2016-01-01"]
    expected = df.iloc[0:6]
    tm.assert_frame_equal(result, expected)

    # partial string match on date and hour, from middle
    result = df.loc["2016-01-02 12"]
    # hourly resolution, same as index.levels[0], so we are _not_ slicing on
    #  that level, so that level gets dropped
    expected = df.iloc[9:12].droplevel(0)
    tm.assert_frame_equal(result, expected)

    # partial string match on secondary index
    result = df_swap.loc[SLC[:, "2016-01-02"], :]
    expected = df_swap.iloc[[2, 3, 7, 8, 12, 13]]
    tm.assert_frame_equal(result, expected)

    # tuple selector with partial string match on date
    # "2016-01-01" has daily resolution, so _is_ a slice on the first level.
    result = df.loc[("2016-01-01", "a"), :]
    expected = df.iloc[[0, 3]]
    expected = df.iloc[[0, 3]].droplevel(1)
    tm.assert_frame_equal(result, expected)

    # Slicing date on first level should break (of course) bc the DTI is the
    #  second level on df_swap
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df_swap.loc["2016-01-01"]


def test_partial_string_timestamp_multiindex_str_key_raises(df):
    # Even though this syntax works on a single index, this is somewhat
    # ambiguous and we don't want to extend this behavior forward to work
    # in multi-indexes. This would amount to selecting a scalar from a
    # column.
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df["2016-01-01"]


def test_partial_string_timestamp_multiindex_daily_resolution(df):
    # GH12685 (partial string with daily resolution or below)
    result = df.loc[IndexSlice["2013-03":"2013-03", :], :]
    expected = df.iloc[118:180]
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_case_when.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    array as pd_array,
    date_range,
)
import pandas._testing as tm


@pytest.fixture
def df():
    """
    base dataframe for testing
    """
    return DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_case_when_caselist_is_not_a_list(df):
    """
    Raise ValueError if caselist is not a list.
    """
    msg = "The caselist argument should be a list; "
    msg += "instead got.+"
    with pytest.raises(TypeError, match=msg):  # GH39154
        df["a"].case_when(caselist=())


def test_case_when_no_caselist(df):
    """
    Raise ValueError if no caselist is provided.
    """
    msg = "provide at least one boolean condition, "
    msg += "with a corresponding replacement."
    with pytest.raises(ValueError, match=msg):  # GH39154
        df["a"].case_when([])


def test_case_when_odd_caselist(df):
    """
    Raise ValueError if no of caselist is odd.
    """
    msg = "Argument 0 must have length 2; "
    msg += "a condition and replacement; instead got length 3."

    with pytest.raises(ValueError, match=msg):
        df["a"].case_when([(df["a"].eq(1), 1, df.a.gt(1))])


def test_case_when_raise_error_from_mask(df):
    """
    Raise Error from within Series.mask
    """
    msg = "Failed to apply condition0 and replacement0."
    with pytest.raises(ValueError, match=msg):
        df["a"].case_when([(df["a"].eq(1), [1, 2])])


def test_case_when_single_condition(df):
    """
    Test output on a single condition.
    """
    result = Series([np.nan, np.nan, np.nan]).case_when([(df.a.eq(1), 1)])
    expected = Series([1, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions(df):
    """
    Test output when booleans are derived from a computation
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [(df.a.eq(1), 1), (Series([False, True, False]), 2)]
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions_replacement_list(df):
    """
    Test output when replacement is a list
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [([True, False, False], 1), (df["a"].gt(1) & df["b"].eq(5), [1, 2, 3])]
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions_replacement_extension_dtype(df):
    """
    Test output when replacement has an extension dtype
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [
            ([True, False, False], 1),
            (df["a"].gt(1) & df["b"].eq(5), pd_array([1, 2, 3], dtype="Int64")),
        ],
    )
    expected = Series([1, 2, np.nan], dtype="Float64")
    tm.assert_series_equal(result, expected)


def test_case_when_multiple_conditions_replacement_series(df):
    """
    Test output when replacement is a Series
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [
            (np.array([True, False, False]), 1),
            (df["a"].gt(1) & df["b"].eq(5), Series([1, 2, 3])),
        ],
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)


def test_case_when_non_range_index():
    """
    Test output if index is not RangeIndex
    """
    rng = np.random.default_rng(seed=123)
    dates = date_range("1/1/2000", periods=8)
    df = DataFrame(
        rng.standard_normal(size=(8, 4)), index=dates, columns=["A", "B", "C", "D"]
    )
    result = Series(5, index=df.index, name="A").case_when([(df.A.gt(0), df.B)])
    expected = df.A.mask(df.A.gt(0), df.B).where(df.A.gt(0), 5)
    tm.assert_series_equal(result, expected)


def test_case_when_callable():
    """
    Test output on a callable
    """
    # https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html
    x = np.linspace(-2.5, 2.5, 6)
    ser = Series(x)
    result = ser.case_when(
        caselist=[
            (lambda df: df < 0, lambda df: -df),
            (lambda df: df >= 0, lambda df: df),
        ]
    )
    expected = np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    tm.assert_series_equal(result, Series(expected))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_misc.py ===
from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os

from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
                                  skip_under_pyodide)
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
                                  strlines, as_int, find_executable)


def test_translate():
    abc = 'abc'
    assert translate(abc, None, 'a') == 'bc'
    assert translate(abc, None, '') == 'abc'
    assert translate(abc, {'a': 'x'}, 'c') == 'xb'
    assert translate(abc, {'a': 'bc'}, 'c') == 'bcb'
    assert translate(abc, {'ab': 'x'}, 'c') == 'x'
    assert translate(abc, {'ab': ''}, 'c') == ''
    assert translate(abc, {'bc': 'x'}, 'c') == 'ab'
    assert translate(abc, {'abc': 'x', 'a': 'y'}) == 'x'
    u = chr(4096)
    assert translate(abc, 'a', 'x', u) == 'xbc'
    assert (u in translate(abc, 'a', u, u)) is True


def test_replace():
    assert replace('abc', ('a', 'b')) == 'bbc'
    assert replace('abc', {'a': 'Aa'}) == 'Aabc'
    assert replace('abc', ('a', 'b'), ('c', 'C')) == 'bbC'


def test_ordinal():
    assert ordinal(-1) == '-1st'
    assert ordinal(0) == '0th'
    assert ordinal(1) == '1st'
    assert ordinal(2) == '2nd'
    assert ordinal(3) == '3rd'
    assert all(ordinal(i).endswith('th') for i in range(4, 21))
    assert ordinal(100) == '100th'
    assert ordinal(101) == '101st'
    assert ordinal(102) == '102nd'
    assert ordinal(103) == '103rd'
    assert ordinal(104) == '104th'
    assert ordinal(200) == '200th'
    assert all(ordinal(i) == str(i) + 'th' for i in range(-220, -203))


def test_rawlines():
    assert rawlines('a a\na') == "dedent('''\\\n    a a\n    a''')"
    assert rawlines('a a') == "'a a'"
    assert rawlines(strlines('\\le"ft')) == (
        '(\n'
        "    '(\\n'\n"
        '    \'r\\\'\\\\le"ft\\\'\\n\'\n'
        "    ')'\n"
        ')')


def test_strlines():
    q = 'this quote (") is in the middle'
    # the following assert rhs was prepared with
    # print(rawlines(strlines(q, 10)))
    assert strlines(q, 10) == dedent('''\
        (
        'this quo'
        'te (") i'
        's in the'
        ' middle'
        )''')
    assert q == (
        'this quo'
        'te (") i'
        's in the'
        ' middle'
        )
    q = "this quote (') is in the middle"
    assert strlines(q, 20) == dedent('''\
        (
        "this quote (') is "
        "in the middle"
        )''')
    assert strlines('\\left') == (
        '(\n'
        "r'\\left'\n"
        ')')
    assert strlines('\\left', short=True) == r"r'\left'"
    assert strlines('\\le"ft') == (
        '(\n'
        'r\'\\le"ft\'\n'
        ')')
    q = 'this\nother line'
    assert strlines(q) == rawlines(q)


def test_translate_args():
    try:
        translate(None, None, None, 'not_none')
    except ValueError:
        pass # Exception raised successfully
    else:
        assert False

    assert translate('s', None, None, None) == 's'

    try:
        translate('s', 'a', 'bc')
    except ValueError:
        pass # Exception raised successfully
    else:
        assert False


@skip_under_pyodide("Cannot create subprocess under pyodide.")
def test_debug_output():
    env = os.environ.copy()
    env['SYMPY_DEBUG'] = 'True'
    cmd = 'from sympy import *; x = Symbol("x"); print(integrate((1-cos(x))/x, x))'
    cmdline = [sys.executable, '-c', cmd]
    proc = Popen(cmdline, env=env, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode('ascii') # utf-8?
    err = err.decode('ascii')
    expected = 'substituted: -x*(1 - cos(x)), u: 1/x, u_var: _u'
    assert expected in err, err


def test_as_int():
    raises(ValueError, lambda : as_int(True))
    raises(ValueError, lambda : as_int(1.1))
    raises(ValueError, lambda : as_int([]))
    raises(ValueError, lambda : as_int(S.NaN))
    raises(ValueError, lambda : as_int(S.Infinity))
    raises(ValueError, lambda : as_int(S.NegativeInfinity))
    raises(ValueError, lambda : as_int(S.ComplexInfinity))
    # for the following, limited precision makes int(arg) == arg
    # but the int value is not necessarily what a user might have
    # expected; Q.prime is more nuanced in its response for
    # expressions which might be complex representations of an
    # integer. This is not -- by design -- as_ints role.
    raises(ValueError, lambda : as_int(1e23))
    raises(ValueError, lambda : as_int(S('1.'+'0'*20+'1')))
    assert as_int(True, strict=False) == 1

def test_deprecated_find_executable():
    with warns_deprecated_sympy():
        find_executable('python')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_clip.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    isna,
    notna,
)
import pandas._testing as tm


class TestSeriesClip:
    def test_clip(self, datetime_series):
        val = datetime_series.median()

        assert datetime_series.clip(lower=val).min() == val
        assert datetime_series.clip(upper=val).max() == val

        result = datetime_series.clip(-0.5, 0.5)
        expected = np.clip(datetime_series, -0.5, 0.5)
        tm.assert_series_equal(result, expected)
        assert isinstance(expected, Series)

    def test_clip_types_and_nulls(self):
        sers = [
            Series([np.nan, 1.0, 2.0, 3.0]),
            Series([None, "a", "b", "c"]),
            Series(pd.to_datetime([np.nan, 1, 2, 3], unit="D")),
        ]

        for s in sers:
            thresh = s[2]
            lower = s.clip(lower=thresh)
            upper = s.clip(upper=thresh)
            assert lower[notna(lower)].min() == thresh
            assert upper[notna(upper)].max() == thresh
            assert list(isna(s)) == list(isna(lower))
            assert list(isna(s)) == list(isna(upper))

    def test_series_clipping_with_na_values(self, any_numeric_ea_dtype, nulls_fixture):
        # Ensure that clipping method can handle NA values with out failing
        # GH#40581

        if nulls_fixture is pd.NaT:
            # constructor will raise, see
            #  test_constructor_mismatched_null_nullable_dtype
            pytest.skip("See test_constructor_mismatched_null_nullable_dtype")

        ser = Series([nulls_fixture, 1.0, 3.0], dtype=any_numeric_ea_dtype)
        s_clipped_upper = ser.clip(upper=2.0)
        s_clipped_lower = ser.clip(lower=2.0)

        expected_upper = Series([nulls_fixture, 1.0, 2.0], dtype=any_numeric_ea_dtype)
        expected_lower = Series([nulls_fixture, 2.0, 3.0], dtype=any_numeric_ea_dtype)

        tm.assert_series_equal(s_clipped_upper, expected_upper)
        tm.assert_series_equal(s_clipped_lower, expected_lower)

    def test_clip_with_na_args(self):
        """Should process np.nan argument as None"""
        # GH#17276
        s = Series([1, 2, 3])

        tm.assert_series_equal(s.clip(np.nan), Series([1, 2, 3]))
        tm.assert_series_equal(s.clip(upper=np.nan, lower=np.nan), Series([1, 2, 3]))

        # GH#19992
        msg = "Downcasting behavior in Series and DataFrame methods 'where'"
        # TODO: avoid this warning here?  seems like we should never be upcasting
        #  in the first place?
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = s.clip(lower=[0, 4, np.nan])
        tm.assert_series_equal(res, Series([1, 4, 3]))
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = s.clip(upper=[1, np.nan, 1])
        tm.assert_series_equal(res, Series([1, 2, 1]))

        # GH#40420
        s = Series([1, 2, 3])
        result = s.clip(0, [np.nan, np.nan, np.nan])
        tm.assert_series_equal(s, result)

    def test_clip_against_series(self):
        # GH#6966

        s = Series([1.0, 1.0, 4.0])

        lower = Series([1.0, 2.0, 3.0])
        upper = Series([1.5, 2.5, 3.5])

        tm.assert_series_equal(s.clip(lower, upper), Series([1.0, 2.0, 3.5]))
        tm.assert_series_equal(s.clip(1.5, upper), Series([1.5, 1.5, 3.5]))

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("upper", [[1, 2, 3], np.asarray([1, 2, 3])])
    def test_clip_against_list_like(self, inplace, upper):
        # GH#15390
        original = Series([5, 6, 7])
        result = original.clip(upper=upper, inplace=inplace)
        expected = Series([1, 2, 3])

        if inplace:
            result = original
        tm.assert_series_equal(result, expected, check_exact=True)

    def test_clip_with_datetimes(self):
        # GH#11838
        # naive and tz-aware datetimes

        t = Timestamp("2015-12-01 09:30:30")
        s = Series([Timestamp("2015-12-01 09:30:00"), Timestamp("2015-12-01 09:31:00")])
        result = s.clip(upper=t)
        expected = Series(
            [Timestamp("2015-12-01 09:30:00"), Timestamp("2015-12-01 09:30:30")]
        )
        tm.assert_series_equal(result, expected)

        t = Timestamp("2015-12-01 09:30:30", tz="US/Eastern")
        s = Series(
            [
                Timestamp("2015-12-01 09:30:00", tz="US/Eastern"),
                Timestamp("2015-12-01 09:31:00", tz="US/Eastern"),
            ]
        )
        result = s.clip(upper=t)
        expected = Series(
            [
                Timestamp("2015-12-01 09:30:00", tz="US/Eastern"),
                Timestamp("2015-12-01 09:30:30", tz="US/Eastern"),
            ]
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [object, "M8[us]"])
    def test_clip_with_timestamps_and_oob_datetimes(self, dtype):
        # GH-42794
        ser = Series([datetime(1, 1, 1), datetime(9999, 9, 9)], dtype=dtype)

        result = ser.clip(lower=Timestamp.min, upper=Timestamp.max)
        expected = Series([Timestamp.min, Timestamp.max], dtype=dtype)

        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\conftest.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    bdate_range,
)


@pytest.fixture(params=[True, False])
def raw(request):
    """raw keyword argument for rolling.apply"""
    return request.param


@pytest.fixture(
    params=[
        "sum",
        "mean",
        "median",
        "max",
        "min",
        "var",
        "std",
        "kurt",
        "skew",
        "count",
        "sem",
    ]
)
def arithmetic_win_operators(request):
    return request.param


@pytest.fixture(params=[True, False])
def center(request):
    return request.param


@pytest.fixture(params=[None, 1])
def min_periods(request):
    return request.param


@pytest.fixture(params=[True, False])
def parallel(request):
    """parallel keyword argument for numba.jit"""
    return request.param


# Can parameterize nogil & nopython over True | False, but limiting per
# https://github.com/pandas-dev/pandas/pull/41971#issuecomment-860607472


@pytest.fixture(params=[False])
def nogil(request):
    """nogil keyword argument for numba.jit"""
    return request.param


@pytest.fixture(params=[True])
def nopython(request):
    """nopython keyword argument for numba.jit"""
    return request.param


@pytest.fixture(params=[True, False])
def adjust(request):
    """adjust keyword argument for ewm"""
    return request.param


@pytest.fixture(params=[True, False])
def ignore_na(request):
    """ignore_na keyword argument for ewm"""
    return request.param


@pytest.fixture(params=[True, False])
def numeric_only(request):
    """numeric_only keyword argument"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("numba", marks=[td.skip_if_no("numba"), pytest.mark.single_cpu]),
        "cython",
    ]
)
def engine(request):
    """engine keyword argument for rolling.apply"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            ("numba", True), marks=[td.skip_if_no("numba"), pytest.mark.single_cpu]
        ),
        ("cython", True),
        ("cython", False),
    ]
)
def engine_and_raw(request):
    """engine and raw keyword arguments for rolling.apply"""
    return request.param


@pytest.fixture(params=["1 day", timedelta(days=1), np.timedelta64(1, "D")])
def halflife_with_times(request):
    """Halflife argument for EWM when times is specified."""
    return request.param


@pytest.fixture
def series():
    """Make mocked series as fixture."""
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan
    series = Series(arr, index=bdate_range(datetime(2009, 1, 1), periods=100))
    return series


@pytest.fixture
def frame():
    """Make mocked frame as fixture."""
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 10)),
        index=bdate_range(datetime(2009, 1, 1), periods=100),
    )


@pytest.fixture(params=[None, 1, 2, 5, 10])
def step(request):
    """step keyword argument for rolling window operations."""
    return request.param

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\tests\test_plot_implicit.py ===
from sympy.core.numbers import (I, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import (And, Or)
from sympy.plotting.plot_implicit import plot_implicit
from sympy.plotting.plot import unset_show
from tempfile import NamedTemporaryFile, mkdtemp
from sympy.testing.pytest import skip, warns, XFAIL
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager

import os

#Set plots not to show
unset_show()

def tmp_file(dir=None, name=''):
    return NamedTemporaryFile(
    suffix='.png', dir=dir, delete=False).name

def plot_and_save(expr, *args, name='', dir=None, **kwargs):
    p = plot_implicit(expr, *args, **kwargs)
    p.save(tmp_file(dir=dir, name=name))
    # Close the plot to avoid a warning from matplotlib
    p._backend.close()

def plot_implicit_tests(name):
    temp_dir = mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    x = Symbol('x')
    y = Symbol('y')
    #implicit plot tests
    plot_and_save(Eq(y, cos(x)), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), (x, -5, 5),
            (y, -4, 4), name=name, dir=temp_dir)
    plot_and_save(y > 1 / x, (x, -5, 5),
            (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y < 1 / tan(x), (x, -5, 5),
            (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y >= 2 * sin(x) * cos(x), (x, -5, 5),
            (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y <= x**2, (x, -3, 3),
            (y, -1, 5), name=name, dir=temp_dir)

    #Test all input args for plot_implicit
    plot_and_save(Eq(y**2, x**3 - x), dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), adaptive=False, dir=temp_dir)
    plot_and_save(Eq(y**2, x**3 - x), adaptive=False, n=500, dir=temp_dir)
    plot_and_save(y > x, (x, -5, 5), dir=temp_dir)
    plot_and_save(And(y > exp(x), y > x + 2), dir=temp_dir)
    plot_and_save(Or(y > x, y > -x), dir=temp_dir)
    plot_and_save(x**2 - 1, (x, -5, 5), dir=temp_dir)
    plot_and_save(x**2 - 1, dir=temp_dir)
    plot_and_save(y > x, depth=-5, dir=temp_dir)
    plot_and_save(y > x, depth=5, dir=temp_dir)
    plot_and_save(y > cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(y < cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(And(y > cos(x), Or(y > x, Eq(y, x))), dir=temp_dir)
    plot_and_save(y - cos(pi / x), dir=temp_dir)

    plot_and_save(x**2 - 1, title='An implicit plot', dir=temp_dir)

@XFAIL
def test_no_adaptive_meshing():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        try:
            temp_dir = mkdtemp()
            TmpFileManager.tmp_folder(temp_dir)
            x = Symbol('x')
            y = Symbol('y')
            # Test plots which cannot be rendered using the adaptive algorithm

            # This works, but it triggers a deprecation warning from sympify(). The
            # code needs to be updated to detect if interval math is supported without
            # relying on random AttributeErrors.
            with warns(UserWarning, match="Adaptive meshing could not be applied"):
                plot_and_save(Eq(y, re(cos(x) + I*sin(x))), name='test', dir=temp_dir)
        finally:
            TmpFileManager.cleanup()
    else:
        skip("Matplotlib not the default backend")
def test_line_color():
    x, y = symbols('x, y')
    p = plot_implicit(x**2 + y**2 - 1, line_color="green", show=False)
    assert p._series[0].line_color == "green"
    p = plot_implicit(x**2 + y**2 - 1, line_color='r', show=False)
    assert p._series[0].line_color == "r"

def test_matplotlib():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        try:
            plot_implicit_tests('test')
            test_line_color()
        finally:
            TmpFileManager.cleanup()
    else:
        skip("Matplotlib not the default backend")


def test_region_and():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if not matplotlib:
        skip("Matplotlib not the default backend")

    from matplotlib.testing.compare import compare_images
    test_directory = os.path.dirname(os.path.abspath(__file__))

    try:
        temp_dir = mkdtemp()
        TmpFileManager.tmp_folder(temp_dir)

        x, y = symbols('x y')

        r1 = (x - 1)**2 + y**2 < 2
        r2 = (x + 1)**2 + y**2 < 2

        test_filename = tmp_file(dir=temp_dir, name="test_region_and")
        cmp_filename = os.path.join(test_directory, "test_region_and.png")
        p = plot_implicit(r1 & r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_or")
        cmp_filename = os.path.join(test_directory, "test_region_or.png")
        p = plot_implicit(r1 | r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_not")
        cmp_filename = os.path.join(test_directory, "test_region_not.png")
        p = plot_implicit(~r1, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)

        test_filename = tmp_file(dir=temp_dir, name="test_region_xor")
        cmp_filename = os.path.join(test_directory, "test_region_xor.png")
        p = plot_implicit(r1 ^ r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
    finally:
        TmpFileManager.cleanup()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\methods\test_insert.py ===
from datetime import timedelta

import numpy as np
import pytest

from pandas._libs import lib

import pandas as pd
from pandas import (
    Index,
    Timedelta,
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm


class TestTimedeltaIndexInsert:
    def test_insert(self):
        idx = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        result = idx.insert(2, timedelta(days=5))
        exp = TimedeltaIndex(["4day", "1day", "5day", "2day"], name="idx")
        tm.assert_index_equal(result, exp)

        # insertion of non-datetime should coerce to object index
        result = idx.insert(1, "inserted")
        expected = Index(
            [Timedelta("4day"), "inserted", Timedelta("1day"), Timedelta("2day")],
            name="idx",
        )
        assert not isinstance(result, TimedeltaIndex)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

        idx = timedelta_range("1day 00:00:01", periods=3, freq="s", name="idx")

        # preserve freq
        expected_0 = TimedeltaIndex(
            ["1day", "1day 00:00:01", "1day 00:00:02", "1day 00:00:03"],
            name="idx",
            freq="s",
        )
        expected_3 = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:02", "1day 00:00:03", "1day 00:00:04"],
            name="idx",
            freq="s",
        )

        # reset freq to None
        expected_1_nofreq = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:01", "1day 00:00:02", "1day 00:00:03"],
            name="idx",
            freq=None,
        )
        expected_3_nofreq = TimedeltaIndex(
            ["1day 00:00:01", "1day 00:00:02", "1day 00:00:03", "1day 00:00:05"],
            name="idx",
            freq=None,
        )

        cases = [
            (0, Timedelta("1day"), expected_0),
            (-3, Timedelta("1day"), expected_0),
            (3, Timedelta("1day 00:00:04"), expected_3),
            (1, Timedelta("1day 00:00:01"), expected_1_nofreq),
            (3, Timedelta("1day 00:00:05"), expected_3_nofreq),
        ]

        for n, d, expected in cases:
            result = idx.insert(n, d)
            tm.assert_index_equal(result, expected)
            assert result.name == expected.name
            assert result.freq == expected.freq

    @pytest.mark.parametrize(
        "null", [None, np.nan, np.timedelta64("NaT"), pd.NaT, pd.NA]
    )
    def test_insert_nat(self, null):
        # GH 18295 (test missing)
        idx = timedelta_range("1day", "3day")
        result = idx.insert(1, null)
        expected = TimedeltaIndex(["1day", pd.NaT, "2day", "3day"])
        tm.assert_index_equal(result, expected)

    def test_insert_invalid_na(self):
        idx = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        item = np.datetime64("NaT")
        result = idx.insert(0, item)

        expected = Index([item] + list(idx), dtype=object, name="idx")
        tm.assert_index_equal(result, expected)

        # Also works if we pass a different dt64nat object
        item2 = np.datetime64("NaT")
        result = idx.insert(0, item2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "item", [0, np.int64(0), np.float64(0), np.array(0), np.datetime64(456, "us")]
    )
    def test_insert_mismatched_types_raises(self, item):
        # GH#33703 dont cast these to td64
        tdi = TimedeltaIndex(["4day", "1day", "2day"], name="idx")

        result = tdi.insert(1, item)

        expected = Index(
            [tdi[0], lib.item_from_zerodim(item)] + list(tdi[1:]),
            dtype=object,
            name="idx",
        )
        tm.assert_index_equal(result, expected)

    def test_insert_castable_str(self):
        idx = timedelta_range("1day", "3day")

        result = idx.insert(0, "1 Day")

        expected = TimedeltaIndex([idx[0]] + list(idx))
        tm.assert_index_equal(result, expected)

    def test_insert_non_castable_str(self):
        idx = timedelta_range("1day", "3day")

        result = idx.insert(0, "foo")

        expected = Index(["foo"] + list(idx), dtype=object)
        tm.assert_index_equal(result, expected)

    def test_insert_empty(self):
        # Corner case inserting with length zero doesn't raise IndexError
        # GH#33573 for freq preservation
        idx = timedelta_range("1 Day", periods=3)
        td = idx[0]

        result = idx[:0].insert(0, td)
        assert result.freq == "D"

        with pytest.raises(IndexError, match="loc must be an integer between"):
            result = idx[:0].insert(1, td)

        with pytest.raises(IndexError, match="loc must be an integer between"):
            result = idx[:0].insert(-1, td)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_equals.py ===
from contextlib import nullcontext
import copy

import numpy as np
import pytest

from pandas._libs.missing import is_matching_na
from pandas.compat.numpy import np_version_gte1p25

from pandas.core.dtypes.common import is_float

from pandas import (
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "arr, idx",
    [
        ([1, 2, 3, 4], [0, 2, 1, 3]),
        ([1, np.nan, 3, np.nan], [0, 2, 1, 3]),
        (
            [1, np.nan, 3, np.nan],
            MultiIndex.from_tuples([(0, "a"), (1, "b"), (2, "c"), (3, "c")]),
        ),
    ],
)
def test_equals(arr, idx):
    s1 = Series(arr, index=idx)
    s2 = s1.copy()
    assert s1.equals(s2)

    s1[1] = 9
    assert not s1.equals(s2)


@pytest.mark.parametrize(
    "val", [1, 1.1, 1 + 1j, True, "abc", [1, 2], (1, 2), {1, 2}, {"a": 1}, None]
)
def test_equals_list_array(val):
    # GH20676 Verify equals operator for list of Numpy arrays
    arr = np.array([1, 2])
    s1 = Series([arr, arr])
    s2 = s1.copy()
    assert s1.equals(s2)

    s1[1] = val

    cm = (
        tm.assert_produces_warning(FutureWarning, check_stacklevel=False)
        if isinstance(val, str) and not np_version_gte1p25
        else nullcontext()
    )
    with cm:
        assert not s1.equals(s2)


def test_equals_false_negative():
    # GH8437 Verify false negative behavior of equals function for dtype object
    arr = [False, np.nan]
    s1 = Series(arr)
    s2 = s1.copy()
    s3 = Series(index=range(2), dtype=object)
    s4 = s3.copy()
    s5 = s3.copy()
    s6 = s3.copy()

    s3[:-1] = s4[:-1] = s5[0] = s6[0] = False
    assert s1.equals(s1)
    assert s1.equals(s2)
    assert s1.equals(s3)
    assert s1.equals(s4)
    assert s1.equals(s5)
    assert s5.equals(s6)


def test_equals_matching_nas():
    # matching but not identical NAs
    left = Series([np.datetime64("NaT")], dtype=object)
    right = Series([np.datetime64("NaT")], dtype=object)
    assert left.equals(right)
    with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
        assert Index(left).equals(Index(right))
    assert left.array.equals(right.array)

    left = Series([np.timedelta64("NaT")], dtype=object)
    right = Series([np.timedelta64("NaT")], dtype=object)
    assert left.equals(right)
    with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
        assert Index(left).equals(Index(right))
    assert left.array.equals(right.array)

    left = Series([np.float64("NaN")], dtype=object)
    right = Series([np.float64("NaN")], dtype=object)
    assert left.equals(right)
    assert Index(left, dtype=left.dtype).equals(Index(right, dtype=right.dtype))
    assert left.array.equals(right.array)


def test_equals_mismatched_nas(nulls_fixture, nulls_fixture2):
    # GH#39650
    left = nulls_fixture
    right = nulls_fixture2
    if hasattr(right, "copy"):
        right = right.copy()
    else:
        right = copy.copy(right)

    ser = Series([left], dtype=object)
    ser2 = Series([right], dtype=object)

    if is_matching_na(left, right):
        assert ser.equals(ser2)
    elif (left is None and is_float(right)) or (right is None and is_float(left)):
        assert ser.equals(ser2)
    else:
        assert not ser.equals(ser2)


def test_equals_none_vs_nan():
    # GH#39650
    ser = Series([1, None], dtype=object)
    ser2 = Series([1, np.nan], dtype=object)

    assert ser.equals(ser2)
    assert Index(ser, dtype=ser.dtype).equals(Index(ser2, dtype=ser2.dtype))
    assert ser.array.equals(ser2.array)


def test_equals_None_vs_float():
    # GH#44190
    left = Series([-np.inf, np.nan, -1.0, 0.0, 1.0, 10 / 3, np.inf], dtype=object)
    right = Series([None] * len(left))

    # these series were found to be equal due to a bug, check that they are correctly
    # found to not equal
    assert not left.equals(right)
    assert not right.equals(left)
    assert not left.to_frame().equals(right.to_frame())
    assert not right.to_frame().equals(left.to_frame())
    assert not Index(left, dtype="object").equals(Index(right, dtype="object"))
    assert not Index(right, dtype="object").equals(Index(left, dtype="object"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_priority.py ===
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.integers import floor


class Higher(Integer):
    '''
    Integer of value 1 and _op_priority 20

    Operations handled by this class return 1 and reverse operations return 2
    '''

    _op_priority = 20.0
    result: Expr = S.One

    def __new__(cls):
        obj = Expr.__new__(cls)
        obj.p = 1
        return obj

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return self.result

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return 2*self.result

    @call_highest_priority('__radd__')
    def __add__(self, other):
        return self.result

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return 2*self.result

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self.result

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return 2*self.result

    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return self.result

    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return 2*self.result

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self.result

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return 2*self.result

    @call_highest_priority('__rmod__')
    def __mod__(self, other):
        return self.result

    @call_highest_priority('__mod__')
    def __rmod__(self, other):
        return 2*self.result

    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other):
        return self.result

    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other):
        return 2*self.result


class Lower(Higher):
    '''
    Integer of value -1 and _op_priority 5

    Operations handled by this class return -1 and reverse operations return -2
    '''

    _op_priority = 5.0
    result: Expr = S.NegativeOne

    def __new__(cls):
        obj = Expr.__new__(cls)
        obj.p = -1
        return obj


x = Symbol('x')
h = Higher()
l = Lower()


def test_mul():
    assert h*l == h*x == 1
    assert l*h == x*h == 2
    assert x*l == l*x == -x


def test_add():
    assert h + l == h + x == 1
    assert l + h == x + h == 2
    assert x + l == l + x == x - 1


def test_sub():
    assert h - l == h - x == 1
    assert l - h == x - h == 2
    assert x - l == -(l - x) == x + 1


def test_pow():
    assert h**l == h**x == 1
    assert l**h == x**h == 2
    assert (x**l).args == (1/x).args and (x**l).is_Pow
    assert (l**x).args == ((-1)**x).args and (l**x).is_Pow


def test_div():
    assert h/l == h/x == 1
    assert l/h == x/h == 2
    assert x/l == 1/(l/x) == -x


def test_mod():
    assert h%l == h%x == 1
    assert l%h == x%h == 2
    assert x%l == Mod(x, -1)
    assert l%x == Mod(-1, x)


def test_floordiv():
    assert h//l == h//x == 1
    assert l//h == x//h == 2
    assert x//l == floor(-x)
    assert l//x == floor(-1/x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\diffgeom\tests\test_function_diffgeom_book.py ===
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r
from sympy.diffgeom import intcurve_series, Differential, WedgeProduct
from sympy.core import symbols, Function, Derivative
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin, cos
from sympy.matrices import Matrix

# Most of the functionality is covered in the
# test_functional_diffgeom_ch* tests which are based on the
# example from the paper of Sussman and Wisdom.
# If they do not cover something, additional tests are added in other test
# functions.

# From "Functional Differential Geometry" as of 2011
# by Sussman and Wisdom.


def test_functional_diffgeom_ch2():
    x0, y0, r0, theta0 = symbols('x0, y0, r0, theta0', real=True)
    x, y = symbols('x, y', real=True)
    f = Function('f')

    assert (R2_p.point_to_coords(R2_r.point([x0, y0])) ==
           Matrix([sqrt(x0**2 + y0**2), atan2(y0, x0)]))
    assert (R2_r.point_to_coords(R2_p.point([r0, theta0])) ==
           Matrix([r0*cos(theta0), r0*sin(theta0)]))

    assert R2_p.jacobian(R2_r, [r0, theta0]) == Matrix(
        [[cos(theta0), -r0*sin(theta0)], [sin(theta0), r0*cos(theta0)]])

    field = f(R2.x, R2.y)
    p1_in_rect = R2_r.point([x0, y0])
    p1_in_polar = R2_p.point([sqrt(x0**2 + y0**2), atan2(y0, x0)])
    assert field.rcall(p1_in_rect) == f(x0, y0)
    assert field.rcall(p1_in_polar) == f(x0, y0)

    p_r = R2_r.point([x0, y0])
    p_p = R2_p.point([r0, theta0])
    assert R2.x(p_r) == x0
    assert R2.x(p_p) == r0*cos(theta0)
    assert R2.r(p_p) == r0
    assert R2.r(p_r) == sqrt(x0**2 + y0**2)
    assert R2.theta(p_r) == atan2(y0, x0)

    h = R2.x*R2.r**2 + R2.y**3
    assert h.rcall(p_r) == x0*(x0**2 + y0**2) + y0**3
    assert h.rcall(p_p) == r0**3*sin(theta0)**3 + r0**3*cos(theta0)


def test_functional_diffgeom_ch3():
    x0, y0 = symbols('x0, y0', real=True)
    x, y, t = symbols('x, y, t', real=True)
    f = Function('f')
    b1 = Function('b1')
    b2 = Function('b2')
    p_r = R2_r.point([x0, y0])

    s_field = f(R2.x, R2.y)
    v_field = b1(R2.x)*R2.e_x + b2(R2.y)*R2.e_y
    assert v_field.rcall(s_field).rcall(p_r).doit() == b1(
        x0)*Derivative(f(x0, y0), x0) + b2(y0)*Derivative(f(x0, y0), y0)

    assert R2.e_x(R2.r**2).rcall(p_r) == 2*x0
    v = R2.e_x + 2*R2.e_y
    s = R2.r**2 + 3*R2.x
    assert v.rcall(s).rcall(p_r).doit() == 2*x0 + 4*y0 + 3

    circ = -R2.y*R2.e_x + R2.x*R2.e_y
    series = intcurve_series(circ, t, R2_r.point([1, 0]), coeffs=True)
    series_x, series_y = zip(*series)
    assert all(
        term == cos(t).taylor_term(i, t) for i, term in enumerate(series_x))
    assert all(
        term == sin(t).taylor_term(i, t) for i, term in enumerate(series_y))


def test_functional_diffgeom_ch4():
    x0, y0, theta0 = symbols('x0, y0, theta0', real=True)
    x, y, r, theta = symbols('x, y, r, theta', real=True)
    r0 = symbols('r0', positive=True)
    f = Function('f')
    b1 = Function('b1')
    b2 = Function('b2')
    p_r = R2_r.point([x0, y0])
    p_p = R2_p.point([r0, theta0])

    f_field = b1(R2.x, R2.y)*R2.dx + b2(R2.x, R2.y)*R2.dy
    assert f_field.rcall(R2.e_x).rcall(p_r) == b1(x0, y0)
    assert f_field.rcall(R2.e_y).rcall(p_r) == b2(x0, y0)

    s_field_r = f(R2.x, R2.y)
    df = Differential(s_field_r)
    assert df(R2.e_x).rcall(p_r).doit() == Derivative(f(x0, y0), x0)
    assert df(R2.e_y).rcall(p_r).doit() == Derivative(f(x0, y0), y0)

    s_field_p = f(R2.r, R2.theta)
    df = Differential(s_field_p)
    assert trigsimp(df(R2.e_x).rcall(p_p).doit()) == (
        cos(theta0)*Derivative(f(r0, theta0), r0) -
        sin(theta0)*Derivative(f(r0, theta0), theta0)/r0)
    assert trigsimp(df(R2.e_y).rcall(p_p).doit()) == (
        sin(theta0)*Derivative(f(r0, theta0), r0) +
        cos(theta0)*Derivative(f(r0, theta0), theta0)/r0)

    assert R2.dx(R2.e_x).rcall(p_r) == 1
    assert R2.dx(R2.e_x) == 1
    assert R2.dx(R2.e_y).rcall(p_r) == 0
    assert R2.dx(R2.e_y) == 0

    circ = -R2.y*R2.e_x + R2.x*R2.e_y
    assert R2.dx(circ).rcall(p_r).doit() == -y0
    assert R2.dy(circ).rcall(p_r) == x0
    assert R2.dr(circ).rcall(p_r) == 0
    assert simplify(R2.dtheta(circ).rcall(p_r)) == 1

    assert (circ - R2.e_theta).rcall(s_field_r).rcall(p_r) == 0


def test_functional_diffgeom_ch6():
    u0, u1, u2, v0, v1, v2, w0, w1, w2 = symbols('u0:3, v0:3, w0:3', real=True)

    u = u0*R2.e_x + u1*R2.e_y
    v = v0*R2.e_x + v1*R2.e_y
    wp = WedgeProduct(R2.dx, R2.dy)
    assert wp(u, v) == u0*v1 - u1*v0

    u = u0*R3_r.e_x + u1*R3_r.e_y + u2*R3_r.e_z
    v = v0*R3_r.e_x + v1*R3_r.e_y + v2*R3_r.e_z
    w = w0*R3_r.e_x + w1*R3_r.e_y + w2*R3_r.e_z
    wp = WedgeProduct(R3_r.dx, R3_r.dy, R3_r.dz)
    assert wp(
        u, v, w) == Matrix(3, 3, [u0, u1, u2, v0, v1, v2, w0, w1, w2]).det()

    a, b, c = symbols('a, b, c', cls=Function)
    a_f = a(R3_r.x, R3_r.y, R3_r.z)
    b_f = b(R3_r.x, R3_r.y, R3_r.z)
    c_f = c(R3_r.x, R3_r.y, R3_r.z)
    theta = a_f*R3_r.dx + b_f*R3_r.dy + c_f*R3_r.dz
    dtheta = Differential(theta)
    da = Differential(a_f)
    db = Differential(b_f)
    dc = Differential(c_f)
    expr = dtheta - WedgeProduct(
        da, R3_r.dx) - WedgeProduct(db, R3_r.dy) - WedgeProduct(dc, R3_r.dz)
    assert expr.rcall(R3_r.e_x, R3_r.e_y) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_tensor_functions.py ===
from sympy.core.relational import Ne
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.tensor_functions import (Eijk, KroneckerDelta, LeviCivita)

from sympy.physics.secondquant import evaluate_deltas, F

x, y = symbols('x y')


def test_levicivita():
    assert Eijk(1, 2, 3) == LeviCivita(1, 2, 3)
    assert LeviCivita(1, 2, 3) == 1
    assert LeviCivita(int(1), int(2), int(3)) == 1
    assert LeviCivita(1, 3, 2) == -1
    assert LeviCivita(1, 2, 2) == 0
    i, j, k = symbols('i j k')
    assert LeviCivita(i, j, k) == LeviCivita(i, j, k, evaluate=False)
    assert LeviCivita(i, j, i) == 0
    assert LeviCivita(1, i, i) == 0
    assert LeviCivita(i, j, k).doit() == (j - i)*(k - i)*(k - j)/2
    assert LeviCivita(1, 2, 3, 1) == 0
    assert LeviCivita(4, 5, 1, 2, 3) == 1
    assert LeviCivita(4, 5, 2, 1, 3) == -1

    assert LeviCivita(i, j, k).is_integer is True

    assert adjoint(LeviCivita(i, j, k)) == LeviCivita(i, j, k)
    assert conjugate(LeviCivita(i, j, k)) == LeviCivita(i, j, k)
    assert transpose(LeviCivita(i, j, k)) == LeviCivita(i, j, k)


def test_kronecker_delta():
    i, j = symbols('i j')
    k = Symbol('k', nonzero=True)
    assert KroneckerDelta(1, 1) == 1
    assert KroneckerDelta(1, 2) == 0
    assert KroneckerDelta(k, 0) == 0
    assert KroneckerDelta(x, x) == 1
    assert KroneckerDelta(x**2 - y**2, x**2 - y**2) == 1
    assert KroneckerDelta(i, i) == 1
    assert KroneckerDelta(i, i + 1) == 0
    assert KroneckerDelta(0, 0) == 1
    assert KroneckerDelta(0, 1) == 0
    assert KroneckerDelta(i + k, i) == 0
    assert KroneckerDelta(i + k, i + k) == 1
    assert KroneckerDelta(i + k, i + 1 + k) == 0
    assert KroneckerDelta(i, j).subs({"i": 1, "j": 0}) == 0
    assert KroneckerDelta(i, j).subs({"i": 3, "j": 3}) == 1

    assert KroneckerDelta(i, j)**0 == 1
    for n in range(1, 10):
        assert KroneckerDelta(i, j)**n == KroneckerDelta(i, j)
        assert KroneckerDelta(i, j)**-n == 1/KroneckerDelta(i, j)

    assert KroneckerDelta(i, j).is_integer is True

    assert adjoint(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    assert conjugate(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    assert transpose(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    # to test if canonical
    assert (KroneckerDelta(i, j) == KroneckerDelta(j, i)) == True

    assert KroneckerDelta(i, j).rewrite(Piecewise) == Piecewise((0, Ne(i, j)), (1, True))

    # Tests with range:
    assert KroneckerDelta(i, j, (0, i)).args == (i, j, (0, i))
    assert KroneckerDelta(i, j, (-j, i)).delta_range == (-j, i)

    # If index is out of range, return zero:
    assert KroneckerDelta(i, j, (0, i-1)) == 0
    assert KroneckerDelta(-1, j, (0, i-1)) == 0
    assert KroneckerDelta(j, -1, (0, i-1)) == 0
    assert KroneckerDelta(j, i, (0, i-1)) == 0


def test_kronecker_delta_secondquant():
    """secondquant-specific methods"""
    D = KroneckerDelta
    i, j, v, w = symbols('i j v w', below_fermi=True, cls=Dummy)
    a, b, t, u = symbols('a b t u', above_fermi=True, cls=Dummy)
    p, q, r, s = symbols('p q r s', cls=Dummy)

    assert D(i, a) == 0
    assert D(i, t) == 0

    assert D(i, j).is_above_fermi is False
    assert D(a, b).is_above_fermi is True
    assert D(p, q).is_above_fermi is True
    assert D(i, q).is_above_fermi is False
    assert D(q, i).is_above_fermi is False
    assert D(q, v).is_above_fermi is False
    assert D(a, q).is_above_fermi is True

    assert D(i, j).is_below_fermi is True
    assert D(a, b).is_below_fermi is False
    assert D(p, q).is_below_fermi is True
    assert D(p, j).is_below_fermi is True
    assert D(q, b).is_below_fermi is False

    assert D(i, j).is_only_above_fermi is False
    assert D(a, b).is_only_above_fermi is True
    assert D(p, q).is_only_above_fermi is False
    assert D(i, q).is_only_above_fermi is False
    assert D(q, i).is_only_above_fermi is False
    assert D(a, q).is_only_above_fermi is True

    assert D(i, j).is_only_below_fermi is True
    assert D(a, b).is_only_below_fermi is False
    assert D(p, q).is_only_below_fermi is False
    assert D(p, j).is_only_below_fermi is True
    assert D(q, b).is_only_below_fermi is False

    assert not D(i, q).indices_contain_equal_information
    assert not D(a, q).indices_contain_equal_information
    assert D(p, q).indices_contain_equal_information
    assert D(a, b).indices_contain_equal_information
    assert D(i, j).indices_contain_equal_information

    assert D(q, b).preferred_index == b
    assert D(q, b).killable_index == q
    assert D(q, t).preferred_index == t
    assert D(q, t).killable_index == q
    assert D(q, i).preferred_index == i
    assert D(q, i).killable_index == q
    assert D(q, v).preferred_index == v
    assert D(q, v).killable_index == q
    assert D(q, p).preferred_index == p
    assert D(q, p).killable_index == q

    EV = evaluate_deltas
    assert EV(D(a, q)*F(q)) == F(a)
    assert EV(D(i, q)*F(q)) == F(i)
    assert EV(D(a, q)*F(a)) == D(a, q)*F(a)
    assert EV(D(i, q)*F(i)) == D(i, q)*F(i)
    assert EV(D(a, b)*F(a)) == F(b)
    assert EV(D(a, b)*F(b)) == F(a)
    assert EV(D(i, j)*F(i)) == F(j)
    assert EV(D(i, j)*F(j)) == F(i)
    assert EV(D(p, q)*F(q)) == F(p)
    assert EV(D(p, q)*F(p)) == F(q)
    assert EV(D(p, j)*D(p, i)*F(i)) == F(j)
    assert EV(D(p, j)*D(p, i)*F(j)) == F(i)
    assert EV(D(p, q)*D(p, i))*F(i) == D(q, i)*F(i)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_lll.py ===
from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices import DM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMRankError, DMValueError, DMShapeError, DMDomainError
from sympy.polys.matrices.lll import _ddm_lll, ddm_lll, ddm_lll_transform
from sympy.testing.pytest import raises


def test_lll():
    normal_test_data = [
        (
            DM([[1, 0, 0, 0, -20160],
                [0, 1, 0, 0, 33768],
                [0, 0, 1, 0, 39578],
                [0, 0, 0, 1, 47757]], ZZ),
            DM([[10, -3, -2, 8, -4],
                [3, -9, 8, 1, -11],
                [-3, 13, -9, -3, -9],
                [-12, -7, -11, 9, -1]], ZZ)
        ),
        (
            DM([[20, 52, 3456],
                [14, 31, -1],
                [34, -442, 0]], ZZ),
            DM([[14, 31, -1],
                [188, -101, -11],
                [236, 13, 3443]], ZZ)
        ),
        (
            DM([[34, -1, -86, 12],
                [-54, 34, 55, 678],
                [23, 3498, 234, 6783],
                [87, 49, 665, 11]], ZZ),
            DM([[34, -1, -86, 12],
                [291, 43, 149, 83],
                [-54, 34, 55, 678],
                [-189, 3077, -184, -223]], ZZ)
        )
    ]
    delta = QQ(5, 6)
    for basis_dm, reduced_dm in normal_test_data:
        reduced = _ddm_lll(basis_dm.rep.to_ddm(), delta=delta)[0]
        assert reduced == reduced_dm.rep.to_ddm()

        reduced = ddm_lll(basis_dm.rep.to_ddm(), delta=delta)
        assert reduced == reduced_dm.rep.to_ddm()

        reduced, transform = _ddm_lll(basis_dm.rep.to_ddm(), delta=delta, return_transform=True)
        assert reduced == reduced_dm.rep.to_ddm()
        assert transform.matmul(basis_dm.rep.to_ddm()) == reduced_dm.rep.to_ddm()

        reduced, transform = ddm_lll_transform(basis_dm.rep.to_ddm(), delta=delta)
        assert reduced == reduced_dm.rep.to_ddm()
        assert transform.matmul(basis_dm.rep.to_ddm()) == reduced_dm.rep.to_ddm()

        reduced = basis_dm.rep.lll(delta=delta)
        assert reduced == reduced_dm.rep

        reduced, transform = basis_dm.rep.lll_transform(delta=delta)
        assert reduced == reduced_dm.rep
        assert transform.matmul(basis_dm.rep) == reduced_dm.rep

        reduced = basis_dm.rep.to_sdm().lll(delta=delta)
        assert reduced == reduced_dm.rep.to_sdm()

        reduced, transform = basis_dm.rep.to_sdm().lll_transform(delta=delta)
        assert reduced == reduced_dm.rep.to_sdm()
        assert transform.matmul(basis_dm.rep.to_sdm()) == reduced_dm.rep.to_sdm()

        reduced = basis_dm.lll(delta=delta)
        assert reduced == reduced_dm

        reduced, transform = basis_dm.lll_transform(delta=delta)
        assert reduced == reduced_dm
        assert transform.matmul(basis_dm) == reduced_dm


def test_lll_linear_dependent():
    linear_dependent_test_data = [
        DM([[0, -1, -2, -3],
            [1, 0, -1, -2],
            [2, 1, 0, -1],
            [3, 2, 1, 0]], ZZ),
        DM([[1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 2, 3, 6]], ZZ),
        DM([[3, -5, 1],
            [4, 6, 0],
            [10, -4, 2]], ZZ)
    ]
    for not_basis in linear_dependent_test_data:
        raises(DMRankError, lambda: _ddm_lll(not_basis.rep.to_ddm()))
        raises(DMRankError, lambda: ddm_lll(not_basis.rep.to_ddm()))
        raises(DMRankError, lambda: not_basis.rep.lll())
        raises(DMRankError, lambda: not_basis.rep.to_sdm().lll())
        raises(DMRankError, lambda: not_basis.lll())
        raises(DMRankError, lambda: _ddm_lll(not_basis.rep.to_ddm(), return_transform=True))
        raises(DMRankError, lambda: ddm_lll_transform(not_basis.rep.to_ddm()))
        raises(DMRankError, lambda: not_basis.rep.lll_transform())
        raises(DMRankError, lambda: not_basis.rep.to_sdm().lll_transform())
        raises(DMRankError, lambda: not_basis.lll_transform())


def test_lll_wrong_delta():
    dummy_matrix = DomainMatrix.ones((3, 3), ZZ)
    for wrong_delta in [QQ(-1, 4), QQ(0, 1), QQ(1, 4), QQ(1, 1), QQ(100, 1)]:
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: ddm_lll(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.lll(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.lll(delta=wrong_delta))
        raises(DMValueError, lambda: _ddm_lll(dummy_matrix.rep, delta=wrong_delta, return_transform=True))
        raises(DMValueError, lambda: ddm_lll_transform(dummy_matrix.rep, delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.lll_transform(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.rep.to_sdm().lll_transform(delta=wrong_delta))
        raises(DMValueError, lambda: dummy_matrix.lll_transform(delta=wrong_delta))


def test_lll_wrong_shape():
    wrong_shape_matrix = DomainMatrix.ones((4, 3), ZZ)
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: ddm_lll(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll())
    raises(DMShapeError, lambda: _ddm_lll(wrong_shape_matrix.rep, return_transform=True))
    raises(DMShapeError, lambda: ddm_lll_transform(wrong_shape_matrix.rep))
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.rep.to_sdm().lll_transform())
    raises(DMShapeError, lambda: wrong_shape_matrix.lll_transform())


def test_lll_wrong_domain():
    wrong_domain_matrix = DomainMatrix.ones((3, 3), QQ)
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: ddm_lll(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll())
    raises(DMDomainError, lambda: _ddm_lll(wrong_domain_matrix.rep, return_transform=True))
    raises(DMDomainError, lambda: ddm_lll_transform(wrong_domain_matrix.rep))
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.rep.to_sdm().lll_transform())
    raises(DMDomainError, lambda: wrong_domain_matrix.lll_transform())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_datetime.py ===
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
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tests.extension import base


@pytest.fixture(params=["US/Central"])
def dtype(request):
    return DatetimeTZDtype(unit="ns", tz=request.param)


@pytest.fixture
def data(dtype):
    data = DatetimeArray._from_sequence(
        pd.date_range("2000", periods=100, tz=dtype.tz), dtype=dtype
    )
    return data


@pytest.fixture
def data_missing(dtype):
    return DatetimeArray._from_sequence(
        np.array(["NaT", "2000-01-01"], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def data_for_sorting(dtype):
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    return DatetimeArray._from_sequence(
        np.array([b, c, a], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def data_missing_for_sorting(dtype):
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    return DatetimeArray._from_sequence(
        np.array([b, "NaT", a], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def data_for_grouping(dtype):
    """
    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    na = "NaT"
    return DatetimeArray._from_sequence(
        np.array([b, b, na, na, a, a, b, c], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def na_cmp():
    def cmp(a, b):
        return a is pd.NaT and a is b

    return cmp


# ----------------------------------------------------------------------------
class TestDatetimeArray(base.ExtensionTests):
    def _get_expected_exception(self, op_name, obj, other):
        if op_name in ["__sub__", "__rsub__"]:
            return None
        return super()._get_expected_exception(op_name, obj, other)

    def _supports_accumulation(self, ser, op_name: str) -> bool:
        return op_name in ["cummin", "cummax"]

    def _supports_reduction(self, obj, op_name: str) -> bool:
        return op_name in ["min", "max", "median", "mean", "std", "any", "all"]

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        meth = all_boolean_reductions
        msg = f"'{meth}' with datetime64 dtypes is deprecated and will raise in"
        with tm.assert_produces_warning(
            FutureWarning, match=msg, check_stacklevel=False
        ):
            super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    def test_series_constructor(self, data):
        # Series construction drops any .freq attr
        data = data._with_freq(None)
        super().test_series_constructor(data)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        result = data.map(lambda x: x, na_action=na_action)
        tm.assert_extension_array_equal(result, data)

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        if op_name in ["median", "mean", "std"]:
            alt = ser.astype("int64")

            res_op = getattr(ser, op_name)
            exp_op = getattr(alt, op_name)
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
            if op_name in ["mean", "median"]:
                # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype"
                # has no attribute "tz"
                tz = ser.dtype.tz  # type: ignore[union-attr]
                expected = pd.Timestamp(expected, tz=tz)
            else:
                expected = pd.Timedelta(expected)
            tm.assert_almost_equal(result, expected)

        else:
            return super().check_reduce(ser, op_name, skipna)


class Test2DCompat(base.NDArrayBacked2DTests):
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_map.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    Index,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "data, categories",
    [
        (list("abcbca"), list("cab")),
        (pd.interval_range(0, 3).repeat(3), pd.interval_range(0, 3)),
    ],
    ids=["string", "interval"],
)
def test_map_str(data, categories, ordered):
    # GH 31202 - override base class since we want to maintain categorical/ordered
    index = CategoricalIndex(data, categories=categories, ordered=ordered)
    result = index.map(str)
    expected = CategoricalIndex(
        map(str, data), categories=map(str, categories), ordered=ordered
    )
    tm.assert_index_equal(result, expected)


def test_map():
    ci = CategoricalIndex(list("ABABC"), categories=list("CBA"), ordered=True)
    result = ci.map(lambda x: x.lower())
    exp = CategoricalIndex(list("ababc"), categories=list("cba"), ordered=True)
    tm.assert_index_equal(result, exp)

    ci = CategoricalIndex(
        list("ABABC"), categories=list("BAC"), ordered=False, name="XXX"
    )
    result = ci.map(lambda x: x.lower())
    exp = CategoricalIndex(
        list("ababc"), categories=list("bac"), ordered=False, name="XXX"
    )
    tm.assert_index_equal(result, exp)

    # GH 12766: Return an index not an array
    tm.assert_index_equal(
        ci.map(lambda x: 1), Index(np.array([1] * 5, dtype=np.int64), name="XXX")
    )

    # change categories dtype
    ci = CategoricalIndex(list("ABABC"), categories=list("BAC"), ordered=False)

    def f(x):
        return {"A": 10, "B": 20, "C": 30}.get(x)

    result = ci.map(f)
    exp = CategoricalIndex([10, 20, 10, 20, 30], categories=[20, 10, 30], ordered=False)
    tm.assert_index_equal(result, exp)

    result = ci.map(Series([10, 20, 30], index=["A", "B", "C"]))
    tm.assert_index_equal(result, exp)

    result = ci.map({"A": 10, "B": 20, "C": 30})
    tm.assert_index_equal(result, exp)


def test_map_with_categorical_series():
    # GH 12756
    a = Index([1, 2, 3, 4])
    b = Series(["even", "odd", "even", "odd"], dtype="category")
    c = Series(["even", "odd", "even", "odd"])

    exp = CategoricalIndex(["odd", "even", "odd", np.nan])
    tm.assert_index_equal(a.map(b), exp)
    exp = Index(["odd", "even", "odd", np.nan])
    tm.assert_index_equal(a.map(c), exp)


@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, CategoricalIndex([False, False, np.nan])),
        ([1, 2, np.nan], pd.isna, Index([False, False, np.nan])),
        ([1, 1, np.nan], {1: False}, CategoricalIndex([False, False, np.nan])),
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),
        (
            [1, 1, np.nan],
            Series([False, False]),
            CategoricalIndex([False, False, np.nan]),
        ),
        (
            [1, 2, np.nan],
            Series([False, False, False]),
            Index([False, False, np.nan]),
        ),
    ),
)
def test_map_with_nan_ignore(data, f, expected):  # GH 24241
    values = CategoricalIndex(data)
    result = values.map(f, na_action="ignore")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, Index([False, False, True])),
        ([1, 2, np.nan], pd.isna, Index([False, False, True])),
        ([1, 1, np.nan], {1: False}, CategoricalIndex([False, False, np.nan])),
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),
        (
            [1, 1, np.nan],
            Series([False, False]),
            CategoricalIndex([False, False, np.nan]),
        ),
        (
            [1, 2, np.nan],
            Series([False, False, False]),
            Index([False, False, np.nan]),
        ),
    ),
)
def test_map_with_nan_none(data, f, expected):  # GH 24241
    values = CategoricalIndex(data)
    result = values.map(f, na_action=None)
    tm.assert_index_equal(result, expected)


def test_map_with_dict_or_series():
    orig_values = ["a", "B", 1, "a"]
    new_values = ["one", 2, 3.0, "one"]
    cur_index = CategoricalIndex(orig_values, name="XXX")
    expected = CategoricalIndex(new_values, name="XXX", categories=[3.0, 2, "one"])

    mapper = Series(new_values[:-1], index=orig_values[:-1])
    result = cur_index.map(mapper)
    # Order of categories in result can be different
    tm.assert_index_equal(result, expected)

    mapper = dict(zip(orig_values[:-1], new_values[:-1]))
    result = cur_index.map(mapper)
    # Order of categories in result can be different
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\moments\test_moments_consistency_expanding.py ===
import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


def no_nans(x):
    return x.notna().all().all()


def all_na(x):
    return x.isnull().all().all()


@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_expanding_apply_consistency_sum_nans(request, all_data, min_periods, f):
    if f is np.sum:
        if not no_nans(all_data) and not (
            all_na(all_data) and not all_data.empty and min_periods > 0
        ):
            request.applymarker(
                pytest.mark.xfail(reason="np.sum has different behavior with NaNs")
            )
    expanding_f_result = all_data.expanding(min_periods=min_periods).sum()
    expanding_apply_f_result = all_data.expanding(min_periods=min_periods).apply(
        func=f, raw=True
    )
    tm.assert_equal(expanding_f_result, expanding_apply_f_result)


@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var(all_data, min_periods, ddof):
    var_x = all_data.expanding(min_periods=min_periods).var(ddof=ddof)
    assert not (var_x < 0).any().any()

    if ddof == 0:
        # check that biased var(x) == mean(x^2) - mean(x)^2
        mean_x2 = (all_data * all_data).expanding(min_periods=min_periods).mean()
        mean_x = all_data.expanding(min_periods=min_periods).mean()
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var_constant(consistent_data, min_periods, ddof):
    count_x = consistent_data.expanding(min_periods=min_periods).count()
    var_x = consistent_data.expanding(min_periods=min_periods).var(ddof=ddof)

    # check that variance of constant series is identically 0
    assert not (var_x > 0).any().any()
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if ddof == 1:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)


@pytest.mark.parametrize("ddof", [0, 1])
def test_expanding_consistency_var_std_cov(all_data, min_periods, ddof):
    var_x = all_data.expanding(min_periods=min_periods).var(ddof=ddof)
    assert not (var_x < 0).any().any()

    std_x = all_data.expanding(min_periods=min_periods).std(ddof=ddof)
    assert not (std_x < 0).any().any()

    # check that var(x) == std(x)^2
    tm.assert_equal(var_x, std_x * std_x)

    cov_x_x = all_data.expanding(min_periods=min_periods).cov(all_data, ddof=ddof)
    assert not (cov_x_x < 0).any().any()

    # check that var(x) == cov(x, x)
    tm.assert_equal(var_x, cov_x_x)


@pytest.mark.parametrize("ddof", [0, 1])
def test_expanding_consistency_series_cov_corr(series_data, min_periods, ddof):
    var_x_plus_y = (
        (series_data + series_data).expanding(min_periods=min_periods).var(ddof=ddof)
    )
    var_x = series_data.expanding(min_periods=min_periods).var(ddof=ddof)
    var_y = series_data.expanding(min_periods=min_periods).var(ddof=ddof)
    cov_x_y = series_data.expanding(min_periods=min_periods).cov(series_data, ddof=ddof)
    # check that cov(x, y) == (var(x+y) - var(x) -
    # var(y)) / 2
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

    # check that corr(x, y) == cov(x, y) / (std(x) *
    # std(y))
    corr_x_y = series_data.expanding(min_periods=min_periods).corr(series_data)
    std_x = series_data.expanding(min_periods=min_periods).std(ddof=ddof)
    std_y = series_data.expanding(min_periods=min_periods).std(ddof=ddof)
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

    if ddof == 0:
        # check that biased cov(x, y) == mean(x*y) -
        # mean(x)*mean(y)
        mean_x = series_data.expanding(min_periods=min_periods).mean()
        mean_y = series_data.expanding(min_periods=min_periods).mean()
        mean_x_times_y = (
            (series_data * series_data).expanding(min_periods=min_periods).mean()
        )
        tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))


def test_expanding_consistency_mean(all_data, min_periods):
    result = all_data.expanding(min_periods=min_periods).mean()
    expected = (
        all_data.expanding(min_periods=min_periods).sum()
        / all_data.expanding(min_periods=min_periods).count()
    )
    tm.assert_equal(result, expected.astype("float64"))


def test_expanding_consistency_constant(consistent_data, min_periods):
    count_x = consistent_data.expanding().count()
    mean_x = consistent_data.expanding(min_periods=min_periods).mean()
    # check that correlation of a series with itself is either 1 or NaN
    corr_x_x = consistent_data.expanding(min_periods=min_periods).corr(consistent_data)

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


def test_expanding_consistency_var_debiasing_factors(all_data, min_periods):
    # check variance debiasing factors
    var_unbiased_x = all_data.expanding(min_periods=min_periods).var()
    var_biased_x = all_data.expanding(min_periods=min_periods).var(ddof=0)
    var_debiasing_factors_x = all_data.expanding().count() / (
        all_data.expanding().count() - 1.0
    ).replace(0.0, np.nan)
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_division.py ===
from mpmath.libmp import *
from mpmath import mpf, mp

from random import randint, choice, seed

all_modes = [round_floor, round_ceiling, round_down, round_up, round_nearest]

fb = from_bstr
fi = from_int
ff = from_float


def test_div_1_3():
    a = fi(1)
    b = fi(3)
    c = fi(-1)

    # floor rounds down, ceiling rounds up
    assert mpf_div(a, b, 7, round_floor)   == fb('0.01010101')
    assert mpf_div(a, b, 7, round_ceiling) == fb('0.01010110')
    assert mpf_div(a, b, 7, round_down)    == fb('0.01010101')
    assert mpf_div(a, b, 7, round_up)      == fb('0.01010110')
    assert mpf_div(a, b, 7, round_nearest) == fb('0.01010101')

    # floor rounds up, ceiling rounds down
    assert mpf_div(c, b, 7, round_floor)   == fb('-0.01010110')
    assert mpf_div(c, b, 7, round_ceiling) == fb('-0.01010101')
    assert mpf_div(c, b, 7, round_down)    == fb('-0.01010101')
    assert mpf_div(c, b, 7, round_up)      == fb('-0.01010110')
    assert mpf_div(c, b, 7, round_nearest) == fb('-0.01010101')

def test_mpf_divi_1_3():
    a = 1
    b = fi(3)
    c = -1
    assert mpf_rdiv_int(a, b, 7, round_floor)   == fb('0.01010101')
    assert mpf_rdiv_int(a, b, 7, round_ceiling) == fb('0.01010110')
    assert mpf_rdiv_int(a, b, 7, round_down)    == fb('0.01010101')
    assert mpf_rdiv_int(a, b, 7, round_up)      == fb('0.01010110')
    assert mpf_rdiv_int(a, b, 7, round_nearest) == fb('0.01010101')
    assert mpf_rdiv_int(c, b, 7, round_floor)   == fb('-0.01010110')
    assert mpf_rdiv_int(c, b, 7, round_ceiling) == fb('-0.01010101')
    assert mpf_rdiv_int(c, b, 7, round_down)    == fb('-0.01010101')
    assert mpf_rdiv_int(c, b, 7, round_up)      == fb('-0.01010110')
    assert mpf_rdiv_int(c, b, 7, round_nearest) == fb('-0.01010101')


def test_div_300():

    q = fi(1000000)
    a = fi(300499999)    # a/q is a little less than a half-integer
    b = fi(300500000)    # b/q exactly a half-integer
    c = fi(300500001)    # c/q is a little more than a half-integer

    # Check nearest integer rounding (prec=9 as 2**8 < 300 < 2**9)

    assert mpf_div(a, q, 9, round_down) == fi(300)
    assert mpf_div(b, q, 9, round_down) == fi(300)
    assert mpf_div(c, q, 9, round_down) == fi(300)
    assert mpf_div(a, q, 9, round_up) == fi(301)
    assert mpf_div(b, q, 9, round_up) == fi(301)
    assert mpf_div(c, q, 9, round_up) == fi(301)

    # Nearest even integer is down
    assert mpf_div(a, q, 9, round_nearest) == fi(300)
    assert mpf_div(b, q, 9, round_nearest) == fi(300)
    assert mpf_div(c, q, 9, round_nearest) == fi(301)

    # Nearest even integer is up
    a = fi(301499999)
    b = fi(301500000)
    c = fi(301500001)
    assert mpf_div(a, q, 9, round_nearest) == fi(301)
    assert mpf_div(b, q, 9, round_nearest) == fi(302)
    assert mpf_div(c, q, 9, round_nearest) == fi(302)


def test_tight_integer_division():
    # Test that integer division at tightest possible precision is exact
    N = 100
    seed(1)
    for i in range(N):
        a = choice([1, -1]) * randint(1, 1<<randint(10, 100))
        b = choice([1, -1]) * randint(1, 1<<randint(10, 100))
        p = a * b
        width = bitcount(abs(b)) - trailing(b)
        a = fi(a); b = fi(b); p = fi(p)
        for mode in all_modes:
            assert mpf_div(p, a, width, mode) == b


def test_epsilon_rounding():
    # Verify that mpf_div uses infinite precision; this result will
    # appear to be exactly 0.101 to a near-sighted algorithm

    a = fb('0.101' + ('0'*200) + '1')
    b = fb('1.10101')
    c = mpf_mul(a, b, 250, round_floor) # exact
    assert mpf_div(c, b, bitcount(a[1]), round_floor) == a # exact

    assert mpf_div(c, b, 2, round_down) == fb('0.10')
    assert mpf_div(c, b, 3, round_down) == fb('0.101')
    assert mpf_div(c, b, 2, round_up) == fb('0.11')
    assert mpf_div(c, b, 3, round_up) == fb('0.110')
    assert mpf_div(c, b, 2, round_floor) == fb('0.10')
    assert mpf_div(c, b, 3, round_floor) == fb('0.101')
    assert mpf_div(c, b, 2, round_ceiling) == fb('0.11')
    assert mpf_div(c, b, 3, round_ceiling) == fb('0.110')

    # The same for negative numbers
    a = fb('-0.101' + ('0'*200) + '1')
    b = fb('1.10101')
    c = mpf_mul(a, b, 250, round_floor)
    assert mpf_div(c, b, bitcount(a[1]), round_floor) == a

    assert mpf_div(c, b, 2, round_down) == fb('-0.10')
    assert mpf_div(c, b, 3, round_up) == fb('-0.110')

    # Floor goes up, ceiling goes down
    assert mpf_div(c, b, 2, round_floor) == fb('-0.11')
    assert mpf_div(c, b, 3, round_floor) == fb('-0.110')
    assert mpf_div(c, b, 2, round_ceiling) == fb('-0.10')
    assert mpf_div(c, b, 3, round_ceiling) == fb('-0.101')


def test_mod():
    mp.dps = 15
    assert mpf(234) % 1 == 0
    assert mpf(-3) % 256 == 253
    assert mpf(0.25) % 23490.5 == 0.25
    assert mpf(0.25) % -23490.5 == -23490.25
    assert mpf(-0.25) % 23490.5 == 23490.25
    assert mpf(-0.25) % -23490.5 == -0.25
    # Check that these cases are handled efficiently
    assert mpf('1e10000000000') % 1 == 0
    assert mpf('1.23e-1000000000') % 1 == mpf('1.23e-1000000000')
    # test __rmod__
    assert 3 % mpf('1.75') == 1.25

def test_div_negative_rnd_bug():
    mp.dps = 15
    assert (-3) / mpf('0.1531879017645047') == mpf('-19.583791966887116')
    assert mpf('-2.6342475750861301') / mpf('0.35126216427941814') == mpf('-7.4993775104985909')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_first_and_last.py ===
"""
Note: includes tests for `last`
"""
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    bdate_range,
    date_range,
)
import pandas._testing as tm

deprecated_msg = "first is deprecated"
last_deprecated_msg = "last is deprecated"


class TestFirst:
    def test_first_subset(self, frame_or_series):
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((100, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100, freq="12h"),
        )
        ts = tm.get_obj(ts, frame_or_series)
        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = ts.first("10d")
            assert len(result) == 20

        ts = DataFrame(
            np.random.default_rng(2).standard_normal((100, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100, freq="D"),
        )
        ts = tm.get_obj(ts, frame_or_series)
        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = ts.first("10d")
            assert len(result) == 10

        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = ts.first("3ME")
            expected = ts[:"3/31/2000"]
            tm.assert_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = ts.first("21D")
            expected = ts[:21]
            tm.assert_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = ts[:0].first("3ME")
            tm.assert_equal(result, ts[:0])

    def test_first_last_raises(self, frame_or_series):
        # GH#20725
        obj = DataFrame([[1, 2, 3], [4, 5, 6]])
        obj = tm.get_obj(obj, frame_or_series)

        msg = "'first' only supports a DatetimeIndex index"
        with tm.assert_produces_warning(
            FutureWarning, match=deprecated_msg
        ), pytest.raises(
            TypeError, match=msg
        ):  # index is not a DatetimeIndex
            obj.first("1D")

        msg = "'last' only supports a DatetimeIndex index"
        with tm.assert_produces_warning(
            FutureWarning, match=last_deprecated_msg
        ), pytest.raises(
            TypeError, match=msg
        ):  # index is not a DatetimeIndex
            obj.last("1D")

    def test_last_subset(self, frame_or_series):
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((100, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100, freq="12h"),
        )
        ts = tm.get_obj(ts, frame_or_series)
        with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
            result = ts.last("10d")
        assert len(result) == 20

        ts = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=30, freq="D"),
        )
        ts = tm.get_obj(ts, frame_or_series)
        with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
            result = ts.last("10d")
        assert len(result) == 10

        with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
            result = ts.last("21D")
        expected = ts["2000-01-10":]
        tm.assert_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
            result = ts.last("21D")
        expected = ts[-21:]
        tm.assert_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
            result = ts[:0].last("3ME")
        tm.assert_equal(result, ts[:0])

    @pytest.mark.parametrize("start, periods", [("2010-03-31", 1), ("2010-03-30", 2)])
    def test_first_with_first_day_last_of_month(self, frame_or_series, start, periods):
        # GH#29623
        x = frame_or_series([1] * 100, index=bdate_range(start, periods=100))
        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = x.first("1ME")
        expected = frame_or_series(
            [1] * periods, index=bdate_range(start, periods=periods)
        )
        tm.assert_equal(result, expected)

    def test_first_with_first_day_end_of_frq_n_greater_one(self, frame_or_series):
        # GH#29623
        x = frame_or_series([1] * 100, index=bdate_range("2010-03-31", periods=100))
        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = x.first("2ME")
        expected = frame_or_series(
            [1] * 23, index=bdate_range("2010-03-31", "2010-04-30")
        )
        tm.assert_equal(result, expected)

    def test_empty_not_input(self):
        # GH#51032
        df = DataFrame(index=pd.DatetimeIndex([]))
        with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
            result = df.last(offset=1)

        with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
            result = df.first(offset=1)

        tm.assert_frame_equal(df, result)
        assert df is not result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_set_axis.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class SharedSetAxisTests:
    @pytest.fixture
    def obj(self):
        raise NotImplementedError("Implemented by subclasses")

    def test_set_axis(self, obj):
        # GH14636; this tests setting index for both Series and DataFrame
        new_index = list("abcd")[: len(obj)]
        expected = obj.copy()
        expected.index = new_index
        result = obj.set_axis(new_index, axis=0)
        tm.assert_equal(expected, result)

    def test_set_axis_copy(self, obj, using_copy_on_write):
        # Test copy keyword GH#47932
        new_index = list("abcd")[: len(obj)]

        orig = obj.iloc[:]
        expected = obj.copy()
        expected.index = new_index

        result = obj.set_axis(new_index, axis=0, copy=True)
        tm.assert_equal(expected, result)
        assert result is not obj
        # check we DID make a copy
        if not using_copy_on_write:
            if obj.ndim == 1:
                assert not tm.shares_memory(result, obj)
            else:
                assert not any(
                    tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                    for i in range(obj.shape[1])
                )

        result = obj.set_axis(new_index, axis=0, copy=False)
        tm.assert_equal(expected, result)
        assert result is not obj
        # check we did NOT make a copy
        if obj.ndim == 1:
            assert tm.shares_memory(result, obj)
        else:
            assert all(
                tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                for i in range(obj.shape[1])
            )

        # copy defaults to True
        result = obj.set_axis(new_index, axis=0)
        tm.assert_equal(expected, result)
        assert result is not obj
        if using_copy_on_write:
            # check we DID NOT make a copy
            if obj.ndim == 1:
                assert tm.shares_memory(result, obj)
            else:
                assert any(
                    tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                    for i in range(obj.shape[1])
                )
        # check we DID make a copy
        elif obj.ndim == 1:
            assert not tm.shares_memory(result, obj)
        else:
            assert not any(
                tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                for i in range(obj.shape[1])
            )

        res = obj.set_axis(new_index, copy=False)
        tm.assert_equal(expected, res)
        # check we did NOT make a copy
        if res.ndim == 1:
            assert tm.shares_memory(res, orig)
        else:
            assert all(
                tm.shares_memory(res.iloc[:, i], orig.iloc[:, i])
                for i in range(res.shape[1])
            )

    def test_set_axis_unnamed_kwarg_warns(self, obj):
        # omitting the "axis" parameter
        new_index = list("abcd")[: len(obj)]

        expected = obj.copy()
        expected.index = new_index

        result = obj.set_axis(new_index)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("axis", [3, "foo"])
    def test_set_axis_invalid_axis_name(self, axis, obj):
        # wrong values for the "axis" parameter
        with pytest.raises(ValueError, match="No axis named"):
            obj.set_axis(list("abc"), axis=axis)

    def test_set_axis_setattr_index_not_collection(self, obj):
        # wrong type
        msg = (
            r"Index\(\.\.\.\) must be called with a collection of some "
            r"kind, None was passed"
        )
        with pytest.raises(TypeError, match=msg):
            obj.index = None

    def test_set_axis_setattr_index_wrong_length(self, obj):
        # wrong length
        msg = (
            f"Length mismatch: Expected axis has {len(obj)} elements, "
            f"new values have {len(obj)-1} elements"
        )
        with pytest.raises(ValueError, match=msg):
            obj.index = np.arange(len(obj) - 1)

        if obj.ndim == 2:
            with pytest.raises(ValueError, match="Length mismatch"):
                obj.columns = obj.columns[::2]


class TestDataFrameSetAxis(SharedSetAxisTests):
    @pytest.fixture
    def obj(self):
        df = DataFrame(
            {"A": [1.1, 2.2, 3.3], "B": [5.0, 6.1, 7.2], "C": [4.4, 5.5, 6.6]},
            index=[2010, 2011, 2012],
        )
        return df


class TestSeriesSetAxis(SharedSetAxisTests):
    @pytest.fixture
    def obj(self):
        ser = Series(np.arange(4), index=[1, 3, 5, 7], dtype="int64")
        return ser

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\resample\conftest.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)

# The various methods we support
downsample_methods = [
    "min",
    "max",
    "first",
    "last",
    "sum",
    "mean",
    "sem",
    "median",
    "prod",
    "var",
    "std",
    "ohlc",
    "quantile",
]
upsample_methods = ["count", "size"]
series_methods = ["nunique"]
resample_methods = downsample_methods + upsample_methods + series_methods


@pytest.fixture(params=downsample_methods)
def downsample_method(request):
    """Fixture for parametrization of Grouper downsample methods."""
    return request.param


@pytest.fixture(params=resample_methods)
def resample_method(request):
    """Fixture for parametrization of Grouper resample methods."""
    return request.param


@pytest.fixture
def _index_start():
    """Fixture for parametrization of index, series and frame."""
    return datetime(2005, 1, 1)


@pytest.fixture
def _index_end():
    """Fixture for parametrization of index, series and frame."""
    return datetime(2005, 1, 10)


@pytest.fixture
def _index_freq():
    """Fixture for parametrization of index, series and frame."""
    return "D"


@pytest.fixture
def _index_name():
    """Fixture for parametrization of index, series and frame."""
    return None


@pytest.fixture
def index(_index_factory, _index_start, _index_end, _index_freq, _index_name):
    """
    Fixture for parametrization of date_range, period_range and
    timedelta_range indexes
    """
    return _index_factory(_index_start, _index_end, freq=_index_freq, name=_index_name)


@pytest.fixture
def _static_values(index):
    """
    Fixture for parametrization of values used in parametrization of
    Series and DataFrames with date_range, period_range and
    timedelta_range indexes
    """
    return np.arange(len(index))


@pytest.fixture
def _series_name():
    """
    Fixture for parametrization of Series name for Series used with
    date_range, period_range and timedelta_range indexes
    """
    return None


@pytest.fixture
def series(index, _series_name, _static_values):
    """
    Fixture for parametrization of Series with date_range, period_range and
    timedelta_range indexes
    """
    return Series(_static_values, index=index, name=_series_name)


@pytest.fixture
def empty_series_dti(series):
    """
    Fixture for parametrization of empty Series with date_range,
    period_range and timedelta_range indexes
    """
    return series[:0]


@pytest.fixture
def frame(index, _series_name, _static_values):
    """
    Fixture for parametrization of DataFrame with date_range, period_range
    and timedelta_range indexes
    """
    # _series_name is intentionally unused
    return DataFrame({"value": _static_values}, index=index)


@pytest.fixture
def empty_frame_dti(series):
    """
    Fixture for parametrization of empty DataFrame with date_range,
    period_range and timedelta_range indexes
    """
    index = series.index[:0]
    return DataFrame(index=index)


@pytest.fixture
def series_and_frame(frame_or_series, series, frame):
    """
    Fixture for parametrization of Series and DataFrame with date_range,
    period_range and timedelta_range indexes
    """
    if frame_or_series == Series:
        return series
    if frame_or_series == DataFrame:
        return frame

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_parabola.py ===
from sympy.core.numbers import (Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry.ellipse import (Circle, Ellipse)
from sympy.geometry.line import (Line, Ray2D, Segment2D)
from sympy.geometry.parabola import Parabola
from sympy.geometry.point import (Point, Point2D)
from sympy.testing.pytest import raises

from sympy.abc import x, y

def test_parabola_geom():
    a, b = symbols('a b')
    p1 = Point(0, 0)
    p2 = Point(3, 7)
    p3 = Point(0, 4)
    p4 = Point(6, 0)
    p5 = Point(a, a)
    d1 = Line(Point(4, 0), Point(4, 9))
    d2 = Line(Point(7, 6), Point(3, 6))
    d3 = Line(Point(4, 0), slope=oo)
    d4 = Line(Point(7, 6), slope=0)
    d5 = Line(Point(b, a), slope=oo)
    d6 = Line(Point(a, b), slope=0)

    half = S.Half

    pa1 = Parabola(None, d2)
    pa2 = Parabola(directrix=d1)
    pa3 = Parabola(p1, d1)
    pa4 = Parabola(p2, d2)
    pa5 = Parabola(p2, d4)
    pa6 = Parabola(p3, d2)
    pa7 = Parabola(p2, d1)
    pa8 = Parabola(p4, d1)
    pa9 = Parabola(p4, d3)
    pa10 = Parabola(p5, d5)
    pa11 = Parabola(p5, d6)
    d = Line(Point(3, 7), Point(2, 9))
    pa12 = Parabola(Point(7, 8), d)
    pa12r = Parabola(Point(7, 8).reflect(d), d)

    raises(ValueError, lambda:
           Parabola(Point(7, 8, 9), Line(Point(6, 7), Point(7, 7))))
    raises(ValueError, lambda:
           Parabola(Point(0, 2), Line(Point(7, 2), Point(6, 2))))
    raises(ValueError, lambda: Parabola(Point(7, 8), Point(3, 8)))

    # Basic Stuff
    assert pa1.focus == Point(0, 0)
    assert pa1.ambient_dimension == S(2)
    assert pa2 == pa3
    assert pa4 != pa7
    assert pa6 != pa7
    assert pa6.focus == Point2D(0, 4)
    assert pa6.focal_length == 1
    assert pa6.p_parameter == -1
    assert pa6.vertex == Point2D(0, 5)
    assert pa6.eccentricity == 1
    assert pa7.focus == Point2D(3, 7)
    assert pa7.focal_length == half
    assert pa7.p_parameter == -half
    assert pa7.vertex == Point2D(7*half, 7)
    assert pa4.focal_length == half
    assert pa4.p_parameter == half
    assert pa4.vertex == Point2D(3, 13*half)
    assert pa8.focal_length == 1
    assert pa8.p_parameter == 1
    assert pa8.vertex == Point2D(5, 0)
    assert pa4.focal_length == pa5.focal_length
    assert pa4.p_parameter == pa5.p_parameter
    assert pa4.vertex == pa5.vertex
    assert pa4.equation() == pa5.equation()
    assert pa8.focal_length == pa9.focal_length
    assert pa8.p_parameter == pa9.p_parameter
    assert pa8.vertex == pa9.vertex
    assert pa8.equation() == pa9.equation()
    assert pa10.focal_length == pa11.focal_length == sqrt((a - b) ** 2) / 2 # if a, b real == abs(a - b)/2
    assert pa11.vertex == Point(*pa10.vertex[::-1]) == Point(a,
                            a - sqrt((a - b)**2)*sign(a - b)/2) # change axis x->y, y->x on pa10
    aos = pa12.axis_of_symmetry
    assert aos == Line(Point(7, 8), Point(5, 7))
    assert pa12.directrix == Line(Point(3, 7), Point(2, 9))
    assert pa12.directrix.angle_between(aos) == S.Pi/2
    assert pa12.eccentricity == 1
    assert pa12.equation(x, y) == (x - 7)**2 + (y - 8)**2 - (-2*x - y + 13)**2/5
    assert pa12.focal_length == 9*sqrt(5)/10
    assert pa12.focus == Point(7, 8)
    assert pa12.p_parameter == 9*sqrt(5)/10
    assert pa12.vertex == Point2D(S(26)/5, S(71)/10)
    assert pa12r.focal_length == 9*sqrt(5)/10
    assert pa12r.focus == Point(-S(1)/5, S(22)/5)
    assert pa12r.p_parameter == -9*sqrt(5)/10
    assert pa12r.vertex == Point(S(8)/5, S(53)/10)


def test_parabola_intersection():
    l1 = Line(Point(1, -2), Point(-1,-2))
    l2 = Line(Point(1, 2), Point(-1,2))
    l3 = Line(Point(1, 0), Point(-1,0))

    p1 = Point(0,0)
    p2 = Point(0, -2)
    p3 = Point(120, -12)
    parabola1 = Parabola(p1, l1)

    # parabola with parabola
    assert parabola1.intersection(parabola1) == [parabola1]
    assert parabola1.intersection(Parabola(p1, l2)) == [Point2D(-2, 0), Point2D(2, 0)]
    assert parabola1.intersection(Parabola(p2, l3)) == [Point2D(0, -1)]
    assert parabola1.intersection(Parabola(Point(16, 0), l1)) == [Point2D(8, 15)]
    assert parabola1.intersection(Parabola(Point(0, 16), l1)) == [Point2D(-6, 8), Point2D(6, 8)]
    assert parabola1.intersection(Parabola(p3, l3)) == []
    # parabola with point
    assert parabola1.intersection(p1) == []
    assert parabola1.intersection(Point2D(0, -1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Point2D(4, 3)) == [Point2D(4, 3)]
    # parabola with line
    assert parabola1.intersection(Line(Point2D(-7, 3), Point(12, 3))) == [Point2D(-4, 3), Point2D(4, 3)]
    assert parabola1.intersection(Line(Point(-4, -1), Point(4, -1))) == [Point(0, -1)]
    assert parabola1.intersection(Line(Point(2, 0), Point(0, -2))) == [Point2D(2, 0)]
    raises(TypeError, lambda: parabola1.intersection(Line(Point(0, 0, 0), Point(1, 1, 1))))
    # parabola with segment
    assert parabola1.intersection(Segment2D((-4, -5), (4, 3))) == [Point2D(0, -1), Point2D(4, 3)]
    assert parabola1.intersection(Segment2D((0, -5), (0, 6))) == [Point2D(0, -1)]
    assert parabola1.intersection(Segment2D((-12, -65), (14, -68))) == []
    # parabola with ray
    assert parabola1.intersection(Ray2D((-4, -5), (4, 3))) == [Point2D(0, -1), Point2D(4, 3)]
    assert parabola1.intersection(Ray2D((0, 7), (1, 14))) == [Point2D(14 + 2*sqrt(57), 105 + 14*sqrt(57))]
    assert parabola1.intersection(Ray2D((0, 7), (0, 14))) == []
    # parabola with ellipse/circle
    assert parabola1.intersection(Circle(p1, 2)) == [Point2D(-2, 0), Point2D(2, 0)]
    assert parabola1.intersection(Circle(p2, 1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Ellipse(p2, 2, 1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Ellipse(Point(0, 19), 5, 7)) == []
    assert parabola1.intersection(Ellipse((0, 3), 12, 4)) == [
           Point2D(0, -1),
           Point2D(-4*sqrt(17)/3, Rational(59, 9)),
           Point2D(4*sqrt(17)/3, Rational(59, 9))]
    # parabola with unsupported type
    raises(TypeError, lambda: parabola1.intersection(2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\test_galoisgroups.py ===
"""Tests for computing Galois groups. """

from sympy.abc import x
from sympy.combinatorics.galois import (
    S1TransitiveSubgroups, S2TransitiveSubgroups, S3TransitiveSubgroups,
    S4TransitiveSubgroups, S5TransitiveSubgroups, S6TransitiveSubgroups,
)
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.numberfields.galoisgroups import (
    tschirnhausen_transformation,
    galois_group,
    _galois_group_degree_4_root_approx,
    _galois_group_degree_5_hybrid,
)
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises


def test_tschirnhausen_transformation():
    for T in [
        Poly(x**2 - 2),
        Poly(x**2 + x + 1),
        Poly(x**4 + 1),
        Poly(x**4 - x**3 + x**2 - x + 1),
    ]:
        _, U = tschirnhausen_transformation(T)
        assert U.degree() == T.degree()
        assert U.is_monic
        assert U.is_irreducible
        K = QQ.alg_field_from_poly(T)
        L = QQ.alg_field_from_poly(U)
        assert field_isomorphism(K.ext, L.ext) is not None


# Test polys are from:
# Cohen, H. *A Course in Computational Algebraic Number Theory*.
test_polys_by_deg = {
    # Degree 1
    1: [
        (x, S1TransitiveSubgroups.S1, True)
    ],
    # Degree 2
    2: [
        (x**2 + x + 1, S2TransitiveSubgroups.S2, False)
    ],
    # Degree 3
    3: [
        (x**3 + x**2 - 2*x - 1, S3TransitiveSubgroups.A3, True),
        (x**3 + 2, S3TransitiveSubgroups.S3, False),
    ],
    # Degree 4
    4: [
        (x**4 + x**3 + x**2 + x + 1, S4TransitiveSubgroups.C4, False),
        (x**4 + 1, S4TransitiveSubgroups.V, True),
        (x**4 - 2, S4TransitiveSubgroups.D4, False),
        (x**4 + 8*x + 12, S4TransitiveSubgroups.A4, True),
        (x**4 + x + 1, S4TransitiveSubgroups.S4, False),
    ],
    # Degree 5
    5: [
        (x**5 + x**4 - 4*x**3 - 3*x**2 + 3*x + 1, S5TransitiveSubgroups.C5, True),
        (x**5 - 5*x + 12, S5TransitiveSubgroups.D5, True),
        (x**5 + 2, S5TransitiveSubgroups.M20, False),
        (x**5 + 20*x + 16, S5TransitiveSubgroups.A5, True),
        (x**5 - x + 1, S5TransitiveSubgroups.S5, False),
    ],
    # Degree 6
    6: [
        (x**6 + x**5 + x**4 + x**3 + x**2 + x + 1, S6TransitiveSubgroups.C6, False),
        (x**6 + 108, S6TransitiveSubgroups.S3, False),
        (x**6 + 2, S6TransitiveSubgroups.D6, False),
        (x**6 - 3*x**2 - 1, S6TransitiveSubgroups.A4, True),
        (x**6 + 3*x**3 + 3, S6TransitiveSubgroups.G18, False),
        (x**6 - 3*x**2 + 1, S6TransitiveSubgroups.A4xC2, False),
        (x**6 - 4*x**2 - 1, S6TransitiveSubgroups.S4p, True),
        (x**6 - 3*x**5 + 6*x**4 - 7*x**3 + 2*x**2 + x - 4, S6TransitiveSubgroups.S4m, False),
        (x**6 + 2*x**3 - 2, S6TransitiveSubgroups.G36m, False),
        (x**6 + 2*x**2 + 2, S6TransitiveSubgroups.S4xC2, False),
        (x**6 + 10*x**5 + 55*x**4 + 140*x**3 + 175*x**2 + 170*x + 25, S6TransitiveSubgroups.PSL2F5, True),
        (x**6 + 10*x**5 + 55*x**4 + 140*x**3 + 175*x**2 - 3019*x + 25, S6TransitiveSubgroups.PGL2F5, False),
        (x**6 + 6*x**4 + 2*x**3 + 9*x**2 + 6*x - 4, S6TransitiveSubgroups.G36p, True),
        (x**6 + 2*x**4 + 2*x**3 + x**2 + 2*x + 2, S6TransitiveSubgroups.G72, False),
        (x**6 + 24*x - 20, S6TransitiveSubgroups.A6, True),
        (x**6 + x + 1, S6TransitiveSubgroups.S6, False),
    ],
}


def test_galois_group():
    """
    Try all the test polys.
    """
    for deg in range(1, 7):
        polys = test_polys_by_deg[deg]
        for T, G, alt in polys:
            assert galois_group(T, by_name=True) == (G, alt)


def test_galois_group_degree_out_of_bounds():
    raises(ValueError, lambda: galois_group(Poly(0, x)))
    raises(ValueError, lambda: galois_group(Poly(1, x)))
    raises(ValueError, lambda: galois_group(Poly(x ** 7 + 1)))


def test_galois_group_not_by_name():
    """
    Check at least one polynomial of each supported degree, to see that
    conversion from name to group works.
    """
    for deg in range(1, 7):
        T, G_name, _ = test_polys_by_deg[deg][0]
        G, _ = galois_group(T)
        assert G == G_name.get_perm_group()


def test_galois_group_not_monic_over_ZZ():
    """
    Check that we can work with polys that are not monic over ZZ.
    """
    for deg in range(1, 7):
        T, G, alt = test_polys_by_deg[deg][0]
        assert galois_group(T/2, by_name=True) == (G, alt)


def test__galois_group_degree_4_root_approx():
    for T, G, alt in test_polys_by_deg[4]:
        assert _galois_group_degree_4_root_approx(Poly(T)) == (G, alt)


def test__galois_group_degree_5_hybrid():
    for T, G, alt in test_polys_by_deg[5]:
        assert _galois_group_degree_5_hybrid(Poly(T)) == (G, alt)


def test_AlgebraicField_galois_group():
    k = QQ.alg_field_from_poly(Poly(x**4 + 1))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.V

    k = QQ.alg_field_from_poly(Poly(x**4 - 2))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.D4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_demidovich.py ===
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit

# Numbers listed with the tests refer to problem numbers in the book
# "Anti-demidovich, problemas resueltos, Ed. URSS"

x = Symbol("x")


def test_leadterm():
    assert (3 + 2*x**(log(3)/log(2) - 1)).leadterm(x) == (3, 0)


def root3(x):
    return root(x, 3)


def root4(x):
    return root(x, 4)


def test_Limits_simple_0():
    assert limit((2**(x + 1) + 3**(x + 1))/(2**x + 3**x), x, oo) == 3  # 175


def test_Limits_simple_1():
    assert limit((x + 1)*(x + 2)*(x + 3)/x**3, x, oo) == 1  # 172
    assert limit(sqrt(x + 1) - sqrt(x), x, oo) == 0  # 179
    assert limit((2*x - 3)*(3*x + 5)*(4*x - 6)/(3*x**3 + x - 1), x, oo) == 8  # Primjer 1
    assert limit(x/root3(x**3 + 10), x, oo) == 1  # Primjer 2
    assert limit((x + 1)**2/(x**2 + 1), x, oo) == 1  # 181


def test_Limits_simple_2():
    assert limit(1000*x/(x**2 - 1), x, oo) == 0  # 182
    assert limit((x**2 - 5*x + 1)/(3*x + 7), x, oo) is oo  # 183
    assert limit((2*x**2 - x + 3)/(x**3 - 8*x + 5), x, oo) == 0  # 184
    assert limit((2*x**2 - 3*x - 4)/sqrt(x**4 + 1), x, oo) == 2  # 186
    assert limit((2*x + 3)/(x + root3(x)), x, oo) == 2  # 187
    assert limit(x**2/(10 + x*sqrt(x)), x, oo) is oo  # 188
    assert limit(root3(x**2 + 1)/(x + 1), x, oo) == 0  # 189
    assert limit(sqrt(x)/sqrt(x + sqrt(x + sqrt(x))), x, oo) == 1  # 190


def test_Limits_simple_3a():
    a = Symbol('a')
    #issue 3513
    assert together(limit((x**2 - (a + 1)*x + a)/(x**3 - a**3), x, a)) == \
        (a - 1)/(3*a**2)  # 196


def test_Limits_simple_3b():
    h = Symbol("h")
    assert limit(((x + h)**3 - x**3)/h, h, 0) == 3*x**2  # 197
    assert limit((1/(1 - x) - 3/(1 - x**3)), x, 1) == -1  # 198
    assert limit((sqrt(1 + x) - 1)/(root3(1 + x) - 1), x, 0) == Rational(3)/2  # Primer 4
    assert limit((sqrt(x) - 1)/(x - 1), x, 1) == Rational(1)/2  # 199
    assert limit((sqrt(x) - 8)/(root3(x) - 4), x, 64) == 3  # 200
    assert limit((root3(x) - 1)/(root4(x) - 1), x, 1) == Rational(4)/3  # 201
    assert limit(
        (root3(x**2) - 2*root3(x) + 1)/(x - 1)**2, x, 1) == Rational(1)/9  # 202


def test_Limits_simple_4a():
    a = Symbol('a')
    assert limit((sqrt(x) - sqrt(a))/(x - a), x, a) == 1/(2*sqrt(a))  # Primer 5
    assert limit((sqrt(x) - 1)/(root3(x) - 1), x, 1) == Rational(3, 2)  # 205
    assert limit((sqrt(1 + x) - sqrt(1 - x))/x, x, 0) == 1  # 207
    assert limit(sqrt(x**2 - 5*x + 6) - x, x, oo) == Rational(-5, 2)  # 213


def test_limits_simple_4aa():
    assert limit(x*(sqrt(x**2 + 1) - x), x, oo) == Rational(1)/2  # 214


def test_Limits_simple_4b():
    #issue 3511
    assert limit(x - root3(x**3 - 1), x, oo) == 0  # 215


def test_Limits_simple_4c():
    assert limit(log(1 + exp(x))/x, x, -oo) == 0  # 267a
    assert limit(log(1 + exp(x))/x, x, oo) == 1  # 267b


def test_bounded():
    assert limit(sin(x)/x, x, oo) == 0  # 216b
    assert limit(x*sin(1/x), x, 0) == 0  # 227a


def test_f1a():
    #issue 3508:
    assert limit((sin(2*x)/x)**(1 + x), x, 0) == 2  # Primer 7


def test_f1a2():
    #issue 3509:
    assert limit(((x - 1)/(x + 1))**x, x, oo) == exp(-2)  # Primer 9


def test_f1b():
    m = Symbol("m")
    n = Symbol("n")
    h = Symbol("h")
    a = Symbol("a")
    assert limit(sin(x)/x, x, 2) == sin(2)/2  # 216a
    assert limit(sin(3*x)/x, x, 0) == 3  # 217
    assert limit(sin(5*x)/sin(2*x), x, 0) == Rational(5, 2)  # 218
    assert limit(sin(pi*x)/sin(3*pi*x), x, 0) == Rational(1, 3)  # 219
    assert limit(x*sin(pi/x), x, oo) == pi  # 220
    assert limit((1 - cos(x))/x**2, x, 0) == S.Half  # 221
    assert limit(x*sin(1/x), x, oo) == 1  # 227b
    assert limit((cos(m*x) - cos(n*x))/x**2, x, 0) == -m**2/2 + n**2/2  # 232
    assert limit((tan(x) - sin(x))/x**3, x, 0) == S.Half  # 233
    assert limit((x - sin(2*x))/(x + sin(3*x)), x, 0) == -Rational(1, 4)  # 237
    assert limit((1 - sqrt(cos(x)))/x**2, x, 0) == Rational(1, 4)  # 239
    assert limit((sqrt(1 + sin(x)) - sqrt(1 - sin(x)))/x, x, 0) == 1  # 240

    assert limit((1 + h/x)**x, x, oo) == exp(h)  # Primer 9
    assert limit((sin(x) - sin(a))/(x - a), x, a) == cos(a)  # 222, *176
    assert limit((cos(x) - cos(a))/(x - a), x, a) == -sin(a)  # 223
    assert limit((sin(x + h) - sin(x))/h, h, 0) == cos(x)  # 225


def test_f2a():
    assert limit(((x + 1)/(2*x + 1))**(x**2), x, oo) == 0  # Primer 8


def test_f2():
    assert limit((sqrt(
        cos(x)) - root3(cos(x)))/(sin(x)**2), x, 0) == -Rational(1, 12)  # *184


def test_f3():
    a = Symbol('a')
    #issue 3504
    assert limit(asin(a*x)/x, x, 0) == a

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\constructors.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock


class BaseConstructorsTests:
    def test_from_sequence_from_cls(self, data):
        result = type(data)._from_sequence(data, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

        data = data[:0]
        result = type(data)._from_sequence(data, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

    def test_array_from_scalars(self, data):
        scalars = [data[0], data[1], data[2]]
        result = data._from_sequence(scalars, dtype=data.dtype)
        assert isinstance(result, type(data))

    def test_series_constructor(self, data):
        result = pd.Series(data, copy=False)
        assert result.dtype == data.dtype
        assert len(result) == len(data)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert result._mgr.array is data

        # Series[EA] is unboxed / boxed correctly
        result2 = pd.Series(result)
        assert result2.dtype == data.dtype
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result2._mgr.blocks[0], EABackedBlock)

    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        result = pd.Series(index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

        # GH 33559 - empty index
        result = pd.Series(index=[], dtype=dtype)
        expected = pd.Series([], index=pd.Index([], dtype="object"), dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        result = pd.Series(na_value, index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_series_constructor_scalar_with_index(self, data, dtype):
        scalar = data[0]
        result = pd.Series(scalar, index=[1, 2, 3], dtype=dtype)
        expected = pd.Series([scalar] * 3, index=[1, 2, 3], dtype=dtype)
        tm.assert_series_equal(result, expected)

        result = pd.Series(scalar, index=["foo"], dtype=dtype)
        expected = pd.Series([scalar], index=["foo"], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("from_series", [True, False])
    def test_dataframe_constructor_from_dict(self, data, from_series):
        if from_series:
            data = pd.Series(data)
        result = pd.DataFrame({"A": data})
        assert result.dtypes["A"] == data.dtype
        assert result.shape == (len(data), 1)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert isinstance(result._mgr.arrays[0], ExtensionArray)

    def test_dataframe_from_series(self, data):
        result = pd.DataFrame(pd.Series(data))
        assert result.dtypes[0] == data.dtype
        assert result.shape == (len(data), 1)
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        assert isinstance(result._mgr.arrays[0], ExtensionArray)

    def test_series_given_mismatched_index_raises(self, data):
        msg = r"Length of values \(3\) does not match length of index \(5\)"
        with pytest.raises(ValueError, match=msg):
            pd.Series(data[:3], index=[0, 1, 2, 3, 4])

    def test_from_dtype(self, data):
        # construct from our dtype & string dtype
        dtype = data.dtype

        expected = pd.Series(data)
        result = pd.Series(list(data), dtype=dtype)
        tm.assert_series_equal(result, expected)

        result = pd.Series(list(data), dtype=str(dtype))
        tm.assert_series_equal(result, expected)

        # gh-30280

        expected = pd.DataFrame(data).astype(dtype)
        result = pd.DataFrame(list(data), dtype=dtype)
        tm.assert_frame_equal(result, expected)

        result = pd.DataFrame(list(data), dtype=str(dtype))
        tm.assert_frame_equal(result, expected)

    def test_pandas_array(self, data):
        # pd.array(extension_array) should be idempotent...
        result = pd.array(data)
        tm.assert_extension_array_equal(result, data)

    def test_pandas_array_dtype(self, data):
        # ... but specifying dtype will override idempotency
        result = pd.array(data, dtype=np.dtype(object))
        expected = pd.arrays.NumpyExtensionArray(np.asarray(data, dtype=object))
        tm.assert_equal(result, expected)

    def test_construct_empty_dataframe(self, dtype):
        # GH 33623
        result = pd.DataFrame(columns=["a"], dtype=dtype)
        expected = pd.DataFrame(
            {"a": pd.array([], dtype=dtype)}, index=pd.RangeIndex(0)
        )
        tm.assert_frame_equal(result, expected)

    def test_empty(self, dtype):
        cls = dtype.construct_array_type()
        result = cls._empty((4,), dtype=dtype)
        assert isinstance(result, cls)
        assert result.dtype == dtype
        assert result.shape == (4,)

        # GH#19600 method on ExtensionDtype
        result2 = dtype.empty((4,))
        assert isinstance(result2, cls)
        assert result2.dtype == dtype
        assert result2.shape == (4,)

        result2 = dtype.empty(4)
        assert isinstance(result2, cls)
        assert result2.dtype == dtype
        assert result2.shape == (4,)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_constructors.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
)
import pandas._testing as tm


class TestCategoricalIndexConstructors:
    def test_construction_disallows_scalar(self):
        msg = "must be called with a collection of some kind"
        with pytest.raises(TypeError, match=msg):
            CategoricalIndex(data=1, categories=list("abcd"), ordered=False)
        with pytest.raises(TypeError, match=msg):
            CategoricalIndex(categories=list("abcd"), ordered=False)

    def test_construction(self):
        ci = CategoricalIndex(list("aabbca"), categories=list("abcd"), ordered=False)
        categories = ci.categories

        result = Index(ci)
        tm.assert_index_equal(result, ci, exact=True)
        assert not result.ordered

        result = Index(ci.values)
        tm.assert_index_equal(result, ci, exact=True)
        assert not result.ordered

        # empty
        result = CategoricalIndex([], categories=categories)
        tm.assert_index_equal(result.categories, Index(categories))
        tm.assert_numpy_array_equal(result.codes, np.array([], dtype="int8"))
        assert not result.ordered

        # passing categories
        result = CategoricalIndex(list("aabbca"), categories=categories)
        tm.assert_index_equal(result.categories, Index(categories))
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )

        c = Categorical(list("aabbca"))
        result = CategoricalIndex(c)
        tm.assert_index_equal(result.categories, Index(list("abc")))
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )
        assert not result.ordered

        result = CategoricalIndex(c, categories=categories)
        tm.assert_index_equal(result.categories, Index(categories))
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )
        assert not result.ordered

        ci = CategoricalIndex(c, categories=list("abcd"))
        result = CategoricalIndex(ci)
        tm.assert_index_equal(result.categories, Index(categories))
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )
        assert not result.ordered

        result = CategoricalIndex(ci, categories=list("ab"))
        tm.assert_index_equal(result.categories, Index(list("ab")))
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, -1, 0], dtype="int8")
        )
        assert not result.ordered

        result = CategoricalIndex(ci, categories=list("ab"), ordered=True)
        tm.assert_index_equal(result.categories, Index(list("ab")))
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, -1, 0], dtype="int8")
        )
        assert result.ordered

        result = CategoricalIndex(ci, categories=list("ab"), ordered=True)
        expected = CategoricalIndex(
            ci, categories=list("ab"), ordered=True, dtype="category"
        )
        tm.assert_index_equal(result, expected, exact=True)

        # turn me to an Index
        result = Index(np.array(ci))
        assert isinstance(result, Index)
        assert not isinstance(result, CategoricalIndex)

    def test_construction_with_dtype(self):
        # specify dtype
        ci = CategoricalIndex(list("aabbca"), categories=list("abc"), ordered=False)

        result = Index(np.array(ci), dtype="category")
        tm.assert_index_equal(result, ci, exact=True)

        result = Index(np.array(ci).tolist(), dtype="category")
        tm.assert_index_equal(result, ci, exact=True)

        # these are generally only equal when the categories are reordered
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)

        result = Index(np.array(ci), dtype="category").reorder_categories(ci.categories)
        tm.assert_index_equal(result, ci, exact=True)

        # make sure indexes are handled
        idx = Index(range(3))
        expected = CategoricalIndex([0, 1, 2], categories=idx, ordered=True)
        result = CategoricalIndex(idx, categories=idx, ordered=True)
        tm.assert_index_equal(result, expected, exact=True)

    def test_construction_empty_with_bool_categories(self):
        # see GH#22702
        cat = CategoricalIndex([], categories=[True, False])
        categories = sorted(cat.categories.tolist())
        assert categories == [False, True]

    def test_construction_with_categorical_dtype(self):
        # construction with CategoricalDtype
        # GH#18109
        data, cats, ordered = "a a b b".split(), "c b a".split(), True
        dtype = CategoricalDtype(categories=cats, ordered=ordered)

        result = CategoricalIndex(data, dtype=dtype)
        expected = CategoricalIndex(data, categories=cats, ordered=ordered)
        tm.assert_index_equal(result, expected, exact=True)

        # GH#19032
        result = Index(data, dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

        # error when combining categories/ordered and dtype kwargs
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."
        with pytest.raises(ValueError, match=msg):
            CategoricalIndex(data, categories=cats, dtype=dtype)

        with pytest.raises(ValueError, match=msg):
            CategoricalIndex(data, ordered=ordered, dtype=dtype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_to_timestamp.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestToTimestamp:
    def test_to_timestamp_non_contiguous(self):
        # GH#44100
        dti = date_range("2021-10-18", periods=9, freq="D")
        pi = dti.to_period()

        result = pi[::2].to_timestamp()
        expected = dti[::2]
        tm.assert_index_equal(result, expected)

        result = pi._data[::2].to_timestamp()
        expected = dti._data[::2]
        # TODO: can we get the freq to round-trip?
        tm.assert_datetime_array_equal(result, expected, check_freq=False)

        result = pi[::-1].to_timestamp()
        expected = dti[::-1]
        tm.assert_index_equal(result, expected)

        result = pi._data[::-1].to_timestamp()
        expected = dti._data[::-1]
        tm.assert_datetime_array_equal(result, expected, check_freq=False)

        result = pi[::2][::-1].to_timestamp()
        expected = dti[::2][::-1]
        tm.assert_index_equal(result, expected)

        result = pi._data[::2][::-1].to_timestamp()
        expected = dti._data[::2][::-1]
        tm.assert_datetime_array_equal(result, expected, check_freq=False)

    def test_to_timestamp_freq(self):
        idx = period_range("2017", periods=12, freq="Y-DEC")
        result = idx.to_timestamp()
        expected = date_range("2017", periods=12, freq="YS-JAN")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_nat(self):
        # GH#7228
        index = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="M", name="idx")

        result = index.to_timestamp("D")
        expected = DatetimeIndex(
            [NaT, datetime(2011, 1, 1), datetime(2011, 2, 1)],
            dtype="M8[ns]",
            name="idx",
        )
        tm.assert_index_equal(result, expected)
        assert result.name == "idx"

        result2 = result.to_period(freq="M")
        tm.assert_index_equal(result2, index)
        assert result2.name == "idx"

        result3 = result.to_period(freq="3M")
        exp = PeriodIndex(["NaT", "2011-01", "2011-02"], freq="3M", name="idx")
        tm.assert_index_equal(result3, exp)
        assert result3.freqstr == "3M"

        msg = "Frequency must be positive, because it represents span: -2Y"
        with pytest.raises(ValueError, match=msg):
            result.to_period(freq="-2Y")

    def test_to_timestamp_preserve_name(self):
        index = period_range(freq="Y", start="1/1/2001", end="12/1/2009", name="foo")
        assert index.name == "foo"

        conv = index.to_timestamp("D")
        assert conv.name == "foo"

    def test_to_timestamp_quarterly_bug(self):
        years = np.arange(1960, 2000).repeat(4)
        quarters = np.tile(list(range(1, 5)), 40)

        pindex = PeriodIndex.from_fields(year=years, quarter=quarters)

        stamps = pindex.to_timestamp("D", "end")
        expected = DatetimeIndex([x.to_timestamp("D", "end") for x in pindex])
        tm.assert_index_equal(stamps, expected)
        assert stamps.freq == expected.freq

    def test_to_timestamp_pi_mult(self):
        idx = PeriodIndex(["2011-01", "NaT", "2011-02"], freq="2M", name="idx")

        result = idx.to_timestamp()
        expected = DatetimeIndex(
            ["2011-01-01", "NaT", "2011-02-01"], dtype="M8[ns]", name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E")
        expected = DatetimeIndex(
            ["2011-02-28", "NaT", "2011-03-31"], dtype="M8[ns]", name="idx"
        )
        expected = expected + Timedelta(1, "D") - Timedelta(1, "ns")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_pi_combined(self):
        idx = period_range(start="2011", periods=2, freq="1D1h", name="idx")

        result = idx.to_timestamp()
        expected = DatetimeIndex(
            ["2011-01-01 00:00", "2011-01-02 01:00"], dtype="M8[ns]", name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E")
        expected = DatetimeIndex(
            ["2011-01-02 00:59:59", "2011-01-03 01:59:59"], name="idx", dtype="M8[ns]"
        )
        expected = expected + Timedelta(1, "s") - Timedelta(1, "ns")
        tm.assert_index_equal(result, expected)

        result = idx.to_timestamp(how="E", freq="h")
        expected = DatetimeIndex(
            ["2011-01-02 00:00", "2011-01-03 01:00"], dtype="M8[ns]", name="idx"
        )
        expected = expected + Timedelta(1, "h") - Timedelta(1, "ns")
        tm.assert_index_equal(result, expected)

    def test_to_timestamp_1703(self):
        index = period_range("1/1/2012", periods=4, freq="D")

        result = index.to_timestamp()
        assert result[0] == Timestamp("1/1/2012")