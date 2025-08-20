
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\dtype.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
    infer_dtype,
    is_object_dtype,
    is_string_dtype,
)


class BaseDtypeTests:
    """Base class for ExtensionDtype classes"""

    def test_name(self, dtype):
        assert isinstance(dtype.name, str)

    def test_kind(self, dtype):
        valid = set("biufcmMOSUV")
        assert dtype.kind in valid

    def test_is_dtype_from_name(self, dtype):
        result = type(dtype).is_dtype(dtype.name)
        assert result is True

    def test_is_dtype_unboxes_dtype(self, data, dtype):
        assert dtype.is_dtype(data) is True

    def test_is_dtype_from_self(self, dtype):
        result = type(dtype).is_dtype(dtype)
        assert result is True

    def test_is_dtype_other_input(self, dtype):
        assert dtype.is_dtype([1, 2, 3]) is False

    def test_is_not_string_type(self, dtype):
        assert not is_string_dtype(dtype)

    def test_is_not_object_type(self, dtype):
        assert not is_object_dtype(dtype)

    def test_eq_with_str(self, dtype):
        assert dtype == dtype.name
        assert dtype != dtype.name + "-suffix"

    def test_eq_with_numpy_object(self, dtype):
        assert dtype != np.dtype("object")

    def test_eq_with_self(self, dtype):
        assert dtype == dtype
        assert dtype != object()

    def test_array_type(self, data, dtype):
        assert dtype.construct_array_type() is type(data)

    def test_check_dtype(self, data):
        dtype = data.dtype

        # check equivalency for using .dtypes
        df = pd.DataFrame(
            {
                "A": pd.Series(data, dtype=dtype),
                "B": data,
                "C": pd.Series(["foo"] * len(data), dtype=object),
                "D": 1,
            }
        )
        result = df.dtypes == str(dtype)
        assert np.dtype("int64") != "Int64"

        expected = pd.Series([True, True, False, False], index=list("ABCD"))

        tm.assert_series_equal(result, expected)

        expected = pd.Series([True, True, False, False], index=list("ABCD"))
        result = df.dtypes.apply(str) == str(dtype)
        tm.assert_series_equal(result, expected)

    def test_hashable(self, dtype):
        hash(dtype)  # no error

    def test_str(self, dtype):
        assert str(dtype) == dtype.name

    def test_eq(self, dtype):
        assert dtype == dtype.name
        assert dtype != "anonther_type"

    def test_construct_from_string_own_name(self, dtype):
        result = dtype.construct_from_string(dtype.name)
        assert type(result) is type(dtype)

        # check OK as classmethod
        result = type(dtype).construct_from_string(dtype.name)
        assert type(result) is type(dtype)

    def test_construct_from_string_another_type_raises(self, dtype):
        msg = f"Cannot construct a '{type(dtype).__name__}' from 'another_type'"
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")

    def test_construct_from_string_wrong_type_raises(self, dtype):
        with pytest.raises(
            TypeError,
            match="'construct_from_string' expects a string, got <class 'int'>",
        ):
            type(dtype).construct_from_string(0)

    def test_get_common_dtype(self, dtype):
        # in practice we will not typically call this with a 1-length list
        # (we shortcut to just use that dtype as the common dtype), but
        # still testing as good practice to have this working (and it is the
        # only case we can test in general)
        assert dtype._get_common_dtype([dtype]) == dtype

    @pytest.mark.parametrize("skipna", [True, False])
    def test_infer_dtype(self, data, data_missing, skipna):
        # only testing that this works without raising an error
        res = infer_dtype(data, skipna=skipna)
        assert isinstance(res, str)
        res = infer_dtype(data_missing, skipna=skipna)
        assert isinstance(res, str)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_tz_localize.py ===
from datetime import timezone

import pytest
import pytz

from pandas._libs.tslibs import timezones

from pandas import (
    DatetimeIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestTZLocalize:
    def test_series_tz_localize_ambiguous_bool(self):
        # make sure that we are correctly accepting bool values as ambiguous

        # GH#14402
        ts = Timestamp("2015-11-01 01:00:03")
        expected0 = Timestamp("2015-11-01 01:00:03-0500", tz="US/Central")
        expected1 = Timestamp("2015-11-01 01:00:03-0600", tz="US/Central")

        ser = Series([ts])
        expected0 = Series([expected0])
        expected1 = Series([expected1])

        with tm.external_error_raised(pytz.AmbiguousTimeError):
            ser.dt.tz_localize("US/Central")

        result = ser.dt.tz_localize("US/Central", ambiguous=True)
        tm.assert_series_equal(result, expected0)

        result = ser.dt.tz_localize("US/Central", ambiguous=[True])
        tm.assert_series_equal(result, expected0)

        result = ser.dt.tz_localize("US/Central", ambiguous=False)
        tm.assert_series_equal(result, expected1)

        result = ser.dt.tz_localize("US/Central", ambiguous=[False])
        tm.assert_series_equal(result, expected1)

    def test_series_tz_localize_matching_index(self):
        # Matching the index of the result with that of the original series
        # GH 43080
        dt_series = Series(
            date_range(start="2021-01-01T02:00:00", periods=5, freq="1D"),
            index=[2, 6, 7, 8, 11],
            dtype="category",
        )
        result = dt_series.dt.tz_localize("Europe/Berlin")
        expected = Series(
            date_range(
                start="2021-01-01T02:00:00", periods=5, freq="1D", tz="Europe/Berlin"
            ),
            index=[2, 6, 7, 8, 11],
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "method, exp",
        [
            ["shift_forward", "2015-03-29 03:00:00"],
            ["shift_backward", "2015-03-29 01:59:59.999999999"],
            ["NaT", NaT],
            ["raise", None],
            ["foo", "invalid"],
        ],
    )
    def test_tz_localize_nonexistent(self, warsaw, method, exp, unit):
        # GH 8917
        tz = warsaw
        n = 60
        dti = date_range(start="2015-03-29 02:00:00", periods=n, freq="min", unit=unit)
        ser = Series(1, index=dti)
        df = ser.to_frame()

        if method == "raise":
            with tm.external_error_raised(pytz.NonExistentTimeError):
                dti.tz_localize(tz, nonexistent=method)
            with tm.external_error_raised(pytz.NonExistentTimeError):
                ser.tz_localize(tz, nonexistent=method)
            with tm.external_error_raised(pytz.NonExistentTimeError):
                df.tz_localize(tz, nonexistent=method)

        elif exp == "invalid":
            msg = (
                "The nonexistent argument must be one of "
                "'raise', 'NaT', 'shift_forward', 'shift_backward' "
                "or a timedelta object"
            )
            with pytest.raises(ValueError, match=msg):
                dti.tz_localize(tz, nonexistent=method)
            with pytest.raises(ValueError, match=msg):
                ser.tz_localize(tz, nonexistent=method)
            with pytest.raises(ValueError, match=msg):
                df.tz_localize(tz, nonexistent=method)

        else:
            result = ser.tz_localize(tz, nonexistent=method)
            expected = Series(1, index=DatetimeIndex([exp] * n, tz=tz).as_unit(unit))
            tm.assert_series_equal(result, expected)

            result = df.tz_localize(tz, nonexistent=method)
            expected = expected.to_frame()
            tm.assert_frame_equal(result, expected)

            res_index = dti.tz_localize(tz, nonexistent=method)
            tm.assert_index_equal(res_index, expected.index)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_series_tz_localize_empty(self, tzstr):
        # GH#2248
        ser = Series(dtype=object)

        ser2 = ser.tz_localize("utc")
        assert ser2.index.tz == timezone.utc

        ser2 = ser.tz_localize(tzstr)
        timezones.tz_compare(ser2.index.tz, timezones.maybe_get_tz(tzstr))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_period.py ===
import numpy as np
import pytest

from pandas._libs.tslibs import (
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.period import (
    extract_ordinals,
    get_period_field_arr,
    period_asfreq,
    period_ordinal,
)

import pandas._testing as tm


def get_freq_code(freqstr: str) -> int:
    off = to_offset(freqstr, is_period=True)
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    code = off._period_dtype_code  # type: ignore[attr-defined]
    return code


@pytest.mark.parametrize(
    "freq1,freq2,expected",
    [
        ("D", "h", 24),
        ("D", "min", 1440),
        ("D", "s", 86400),
        ("D", "ms", 86400000),
        ("D", "us", 86400000000),
        ("D", "ns", 86400000000000),
        ("h", "min", 60),
        ("h", "s", 3600),
        ("h", "ms", 3600000),
        ("h", "us", 3600000000),
        ("h", "ns", 3600000000000),
        ("min", "s", 60),
        ("min", "ms", 60000),
        ("min", "us", 60000000),
        ("min", "ns", 60000000000),
        ("s", "ms", 1000),
        ("s", "us", 1000000),
        ("s", "ns", 1000000000),
        ("ms", "us", 1000),
        ("ms", "ns", 1000000),
        ("us", "ns", 1000),
    ],
)
def test_intra_day_conversion_factors(freq1, freq2, expected):
    assert (
        period_asfreq(1, get_freq_code(freq1), get_freq_code(freq2), False) == expected
    )


@pytest.mark.parametrize(
    "freq,expected", [("Y", 0), ("M", 0), ("W", 1), ("D", 0), ("B", 0)]
)
def test_period_ordinal_start_values(freq, expected):
    # information for Jan. 1, 1970.
    assert period_ordinal(1970, 1, 1, 0, 0, 0, 0, 0, get_freq_code(freq)) == expected


@pytest.mark.parametrize(
    "dt,expected",
    [
        ((1970, 1, 4, 0, 0, 0, 0, 0), 1),
        ((1970, 1, 5, 0, 0, 0, 0, 0), 2),
        ((2013, 10, 6, 0, 0, 0, 0, 0), 2284),
        ((2013, 10, 7, 0, 0, 0, 0, 0), 2285),
    ],
)
def test_period_ordinal_week(dt, expected):
    args = dt + (get_freq_code("W"),)
    assert period_ordinal(*args) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        # Thursday (Oct. 3, 2013).
        (3, 11415),
        # Friday (Oct. 4, 2013).
        (4, 11416),
        # Saturday (Oct. 5, 2013).
        (5, 11417),
        # Sunday (Oct. 6, 2013).
        (6, 11417),
        # Monday (Oct. 7, 2013).
        (7, 11417),
        # Tuesday (Oct. 8, 2013).
        (8, 11418),
    ],
)
def test_period_ordinal_business_day(day, expected):
    # 5000 is PeriodDtypeCode for BusinessDay
    args = (2013, 10, day, 0, 0, 0, 0, 0, 5000)
    assert period_ordinal(*args) == expected


class TestExtractOrdinals:
    def test_extract_ordinals_raises(self):
        # with non-object, make sure we raise TypeError, not segfault
        arr = np.arange(5)
        freq = to_offset("D")
        with pytest.raises(TypeError, match="values must be object-dtype"):
            extract_ordinals(arr, freq)

    def test_extract_ordinals_2d(self):
        freq = to_offset("D")
        arr = np.empty(10, dtype=object)
        arr[:] = iNaT

        res = extract_ordinals(arr, freq)
        res2 = extract_ordinals(arr.reshape(5, 2), freq)
        tm.assert_numpy_array_equal(res, res2.reshape(-1))


def test_get_period_field_array_raises_on_out_of_range():
    msg = "Buffer dtype mismatch, expected 'const int64_t' but got 'double'"
    with pytest.raises(ValueError, match=msg):
        get_period_field_arr(-1, np.empty(1), 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_dyadic.py ===
from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, outer
from sympy.physics.vector.dyadic import _check_dyadic
from sympy.testing.pytest import raises

A = ReferenceFrame('A')


def test_dyadic():
    d1 = A.x | A.x
    d2 = A.y | A.y
    d3 = A.x | A.y
    assert d1 * 0 == 0
    assert d1 != 0
    assert d1 * 2 == 2 * A.x | A.x
    assert d1 / 2. == 0.5 * d1
    assert d1 & (0 * d1) == 0
    assert d1 & d2 == 0
    assert d1 & A.x == A.x
    assert d1 ^ A.x == 0
    assert d1 ^ A.y == A.x | A.z
    assert d1 ^ A.z == - A.x | A.y
    assert d2 ^ A.x == - A.y | A.z
    assert A.x ^ d1 == 0
    assert A.y ^ d1 == - A.z | A.x
    assert A.z ^ d1 == A.y | A.x
    assert A.x & d1 == A.x
    assert A.y & d1 == 0
    assert A.y & d2 == A.y
    assert d1 & d3 == A.x | A.y
    assert d3 & d1 == 0
    assert d1.dt(A) == 0
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    B = A.orientnew('B', 'Axis', [q, A.z])
    assert d1.express(B) == d1.express(B, B)
    assert d1.express(B) == ((cos(q)**2) * (B.x | B.x) + (-sin(q) * cos(q)) *
            (B.x | B.y) + (-sin(q) * cos(q)) * (B.y | B.x) + (sin(q)**2) *
            (B.y | B.y))
    assert d1.express(B, A) == (cos(q)) * (B.x | A.x) + (-sin(q)) * (B.y | A.x)
    assert d1.express(A, B) == (cos(q)) * (A.x | B.x) + (-sin(q)) * (A.x | B.y)
    assert d1.dt(B) == (-qd) * (A.y | A.x) + (-qd) * (A.x | A.y)

    assert d1.to_matrix(A) == Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert d1.to_matrix(A, B) == Matrix([[cos(q), -sin(q), 0],
                                         [0, 0, 0],
                                         [0, 0, 0]])
    assert d3.to_matrix(A) == Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    v1 = a * A.x + b * A.y + c * A.z
    v2 = d * A.x + e * A.y + f * A.z
    d4 = v1.outer(v2)
    assert d4.to_matrix(A) == Matrix([[a * d, a * e, a * f],
                                      [b * d, b * e, b * f],
                                      [c * d, c * e, c * f]])
    d5 = v1.outer(v1)
    C = A.orientnew('C', 'Axis', [q, A.x])
    for expected, actual in zip(C.dcm(A) * d5.to_matrix(A) * C.dcm(A).T,
                                d5.to_matrix(C)):
        assert (expected - actual).simplify() == 0

    raises(TypeError, lambda: d1.applyfunc(0))


def test_dyadic_simplify():
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    N = ReferenceFrame('N')

    dy = N.x | N.x
    test1 = (1 / x + 1 / y) * dy
    assert (N.x & test1 & N.x) != (x + y) / (x * y)
    test1 = test1.simplify()
    assert (N.x & test1 & N.x) == (x + y) / (x * y)

    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * dy
    test2 = test2.simplify()
    assert (N.x & test2 & N.x) == (A**2 * s**4 / (4 * pi * k * m**3))

    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * dy
    test3 = test3.simplify()
    assert (N.x & test3 & N.x) == 0

    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * dy
    test4 = test4.simplify()
    assert (N.x & test4 & N.x) == -2 * y


def test_dyadic_subs():
    N = ReferenceFrame('N')
    s = symbols('s')
    a = s*(N.x | N.x)
    assert a.subs({s: 2}) == 2*(N.x | N.x)


def test_check_dyadic():
    raises(TypeError, lambda: _check_dyadic(0))


def test_dyadic_evalf():
    N = ReferenceFrame('N')
    a = pi * (N.x | N.x)
    assert a.evalf(3) == Float('3.1416', 3) * (N.x | N.x)
    s = symbols('s')
    a = 5 * s * pi* (N.x | N.x)
    assert a.evalf(2) == Float('5', 2) * Float('3.1416', 2) * s * (N.x | N.x)
    assert a.evalf(9, subs={s: 5.124}) == Float('80.48760378', 9) * (N.x | N.x)


def test_dyadic_xreplace():
    x, y, z = symbols('x y z')
    N = ReferenceFrame('N')
    D = outer(N.x, N.x)
    v = x*y * D
    assert v.xreplace({x : cos(x)}) == cos(x)*y * D
    assert v.xreplace({x*y : pi}) == pi * D
    v = (x*y)**z * D
    assert v.xreplace({(x*y)**z : 1}) == D
    assert v.xreplace({x:1, z:0}) == D
    raises(TypeError, lambda: v.xreplace())
    raises(TypeError, lambda: v.xreplace([x, y]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\test_cookiejar.py ===
import unittest

from websocket._cookiejar import SimpleCookieJar

"""
test_cookiejar.py
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


class CookieJarTest(unittest.TestCase):
    def test_add(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.add("")
        self.assertFalse(
            cookie_jar.jar, "Cookie with no domain should not be added to the jar"
        )

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b")
        self.assertFalse(
            cookie_jar.jar, "Cookie with no domain should not be added to the jar"
        )

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; domain=.abc")
        self.assertTrue(".abc" in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; domain=abc")
        self.assertTrue(".abc" in cookie_jar.jar)
        self.assertTrue("abc" not in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEqual(cookie_jar.get(None), "")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d; e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=.abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d; e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.add("a=b; c=d; domain=abc")
        cookie_jar.add("e=f; domain=xyz")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("xyz"), "e=f")
        self.assertEqual(cookie_jar.get("something"), "")

    def test_set(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b")
        self.assertFalse(
            cookie_jar.jar, "Cookie with no domain should not be added to the jar"
        )

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; domain=.abc")
        self.assertTrue(".abc" in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; domain=abc")
        self.assertTrue(".abc" in cookie_jar.jar)
        self.assertTrue("abc" not in cookie_jar.jar)

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=abc")
        self.assertEqual(cookie_jar.get("abc"), "e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=.abc")
        self.assertEqual(cookie_jar.get("abc"), "e=f")

        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc")
        cookie_jar.set("e=f; domain=xyz")
        self.assertEqual(cookie_jar.get("abc"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("xyz"), "e=f")
        self.assertEqual(cookie_jar.get("something"), "")

    def test_get(self):
        cookie_jar = SimpleCookieJar()
        cookie_jar.set("a=b; c=d; domain=abc.com")
        self.assertEqual(cookie_jar.get("abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("x.abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("abc.com.es"), "")
        self.assertEqual(cookie_jar.get("xabc.com"), "")

        cookie_jar.set("a=b; c=d; domain=.abc.com")
        self.assertEqual(cookie_jar.get("abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("x.abc.com"), "a=b; c=d")
        self.assertEqual(cookie_jar.get("abc.com.es"), "")
        self.assertEqual(cookie_jar.get("xabc.com"), "")


if __name__ == "__main__":
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_size.py ===
import numpy as np
import pytest

from pandas.core.dtypes.common import is_integer_dtype

from pandas import (
    DataFrame,
    Index,
    PeriodIndex,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize("by", ["A", "B", ["A", "B"]])
def test_size(df, by):
    grouped = df.groupby(by=by)
    result = grouped.size()
    for key, group in grouped:
        assert result[key] == len(group)


@pytest.mark.parametrize(
    "by",
    [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [0, None, None, None],
        pytest.param([None, None, None, None], marks=pytest.mark.xfail),
    ],
)
def test_size_axis_1(df, axis_1, by, sort, dropna):
    # GH#45715
    counts = {key: sum(value == key for value in by) for key in dict.fromkeys(by)}
    if dropna:
        counts = {key: value for key, value in counts.items() if key is not None}
    expected = Series(counts, dtype="int64")
    if sort:
        expected = expected.sort_index()
    if is_integer_dtype(expected.index.dtype) and not any(x is None for x in by):
        expected.index = expected.index.astype(int)

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = df.groupby(by=by, axis=axis_1, sort=sort, dropna=dropna)
    result = grouped.size()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("by", ["A", "B", ["A", "B"]])
@pytest.mark.parametrize("sort", [True, False])
def test_size_sort(sort, by):
    df = DataFrame(np.random.default_rng(2).choice(20, (1000, 3)), columns=list("ABC"))
    left = df.groupby(by=by, sort=sort).size()
    right = df.groupby(by=by, sort=sort)["C"].apply(lambda a: a.shape[0])
    tm.assert_series_equal(left, right, check_names=False)


def test_size_series_dataframe():
    # https://github.com/pandas-dev/pandas/issues/11699
    df = DataFrame(columns=["A", "B"])
    out = Series(dtype="int64", index=Index([], name="A"))
    tm.assert_series_equal(df.groupby("A").size(), out)


def test_size_groupby_all_null():
    # https://github.com/pandas-dev/pandas/issues/23050
    # Assert no 'Value Error : Length of passed values is 2, index implies 0'
    df = DataFrame({"A": [None, None]})  # all-null groups
    result = df.groupby("A").size()
    expected = Series(dtype="int64", index=Index([], name="A"))
    tm.assert_series_equal(result, expected)


def test_size_period_index():
    # https://github.com/pandas-dev/pandas/issues/34010
    ser = Series([1], index=PeriodIndex(["2000"], name="A", freq="D"))
    grp = ser.groupby(level="A")
    result = grp.size()
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize("as_index", [True, False])
def test_size_on_categorical(as_index):
    df = DataFrame([[1, 1], [2, 2]], columns=["A", "B"])
    df["A"] = df["A"].astype("category")
    result = df.groupby(["A", "B"], as_index=as_index, observed=False).size()

    expected = DataFrame(
        [[1, 1, 1], [1, 2, 0], [2, 1, 0], [2, 2, 1]], columns=["A", "B", "size"]
    )
    expected["A"] = expected["A"].astype("category")
    if as_index:
        expected = expected.set_index(["A", "B"])["size"].rename(None)

    tm.assert_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_size_series_masked_type_returns_Int64(dtype):
    # GH 54132
    ser = Series([1, 1, 1], index=["a", "a", "b"], dtype=dtype)
    result = ser.groupby(level=0).size()
    expected = Series([2, 1], dtype="Int64", index=["a", "b"])
    tm.assert_series_equal(result, expected)


def test_size_strings(any_string_dtype, using_infer_string):
    # GH#55627
    dtype = any_string_dtype
    df = DataFrame({"a": ["a", "a", "b"], "b": "a"}, dtype=dtype)
    result = df.groupby("a")["b"].size()
    exp_dtype = "Int64" if dtype == "string[pyarrow]" else "int64"
    exp_index_dtype = "str" if using_infer_string and dtype == "object" else dtype
    expected = Series(
        [2, 1],
        index=Index(["a", "b"], name="a", dtype=exp_index_dtype),
        name="b",
        dtype=exp_dtype,
    )
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_compat.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm


def test_numeric_compat(idx):
    with pytest.raises(TypeError, match="cannot perform __mul__"):
        idx * 1

    with pytest.raises(TypeError, match="cannot perform __rmul__"):
        1 * idx

    div_err = "cannot perform __truediv__"
    with pytest.raises(TypeError, match=div_err):
        idx / 1

    div_err = div_err.replace(" __", " __r")
    with pytest.raises(TypeError, match=div_err):
        1 / idx

    with pytest.raises(TypeError, match="cannot perform __floordiv__"):
        idx // 1

    with pytest.raises(TypeError, match="cannot perform __rfloordiv__"):
        1 // idx


@pytest.mark.parametrize("method", ["all", "any", "__invert__"])
def test_logical_compat(idx, method):
    msg = f"cannot perform {method}"

    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)()


def test_inplace_mutation_resets_values():
    levels = [["a", "b", "c"], [4]]
    levels2 = [[1, 2, 3], ["a"]]
    codes = [[0, 1, 0, 2, 2, 0], [0, 0, 0, 0, 0, 0]]

    mi1 = MultiIndex(levels=levels, codes=codes)
    mi2 = MultiIndex(levels=levels2, codes=codes)

    # instantiating MultiIndex should not access/cache _.values
    assert "_values" not in mi1._cache
    assert "_values" not in mi2._cache

    vals = mi1.values.copy()
    vals2 = mi2.values.copy()

    # accessing .values should cache ._values
    assert mi1._values is mi1._cache["_values"]
    assert mi1.values is mi1._cache["_values"]
    assert isinstance(mi1._cache["_values"], np.ndarray)

    # Make sure level setting works
    new_vals = mi1.set_levels(levels2).values
    tm.assert_almost_equal(vals2, new_vals)

    #  Doesn't drop _values from _cache [implementation detail]
    tm.assert_almost_equal(mi1._cache["_values"], vals)

    # ...and values is still same too
    tm.assert_almost_equal(mi1.values, vals)

    # Make sure label setting works too
    codes2 = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    exp_values = np.empty((6,), dtype=object)
    exp_values[:] = [(1, "a")] * 6

    # Must be 1d array of tuples
    assert exp_values.shape == (6,)

    new_mi = mi2.set_codes(codes2)
    assert "_values" not in new_mi._cache
    new_values = new_mi.values
    assert "_values" in new_mi._cache

    # Shouldn't change cache
    tm.assert_almost_equal(mi2._cache["_values"], vals2)

    # Should have correct values
    tm.assert_almost_equal(exp_values, new_values)


def test_boxable_categorical_values():
    cat = pd.Categorical(pd.date_range("2012-01-01", periods=3, freq="h"))
    result = MultiIndex.from_product([["a", "b", "c"], cat]).values
    expected = pd.Series(
        [
            ("a", pd.Timestamp("2012-01-01 00:00:00")),
            ("a", pd.Timestamp("2012-01-01 01:00:00")),
            ("a", pd.Timestamp("2012-01-01 02:00:00")),
            ("b", pd.Timestamp("2012-01-01 00:00:00")),
            ("b", pd.Timestamp("2012-01-01 01:00:00")),
            ("b", pd.Timestamp("2012-01-01 02:00:00")),
            ("c", pd.Timestamp("2012-01-01 00:00:00")),
            ("c", pd.Timestamp("2012-01-01 01:00:00")),
            ("c", pd.Timestamp("2012-01-01 02:00:00")),
        ]
    ).values
    tm.assert_numpy_array_equal(result, expected)
    result = pd.DataFrame({"a": ["a", "b", "c"], "b": cat, "c": np.array(cat)}).values
    expected = pd.DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": [
                pd.Timestamp("2012-01-01 00:00:00"),
                pd.Timestamp("2012-01-01 01:00:00"),
                pd.Timestamp("2012-01-01 02:00:00"),
            ],
            "c": [
                pd.Timestamp("2012-01-01 00:00:00"),
                pd.Timestamp("2012-01-01 01:00:00"),
                pd.Timestamp("2012-01-01 02:00:00"),
            ],
        }
    ).values
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\methods\test_shift.py ===
import numpy as np
import pytest

from pandas import (
    PeriodIndex,
    period_range,
)
import pandas._testing as tm


class TestPeriodIndexShift:
    # ---------------------------------------------------------------
    # PeriodIndex.shift is used by __add__ and __sub__

    def test_pi_shift_ndarray(self):
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="M", name="idx"
        )
        result = idx.shift(np.array([1, 2, 3, 4]))
        expected = PeriodIndex(
            ["2011-02", "2011-04", "NaT", "2011-08"], freq="M", name="idx"
        )
        tm.assert_index_equal(result, expected)

        result = idx.shift(np.array([1, -2, 3, -4]))
        expected = PeriodIndex(
            ["2011-02", "2010-12", "NaT", "2010-12"], freq="M", name="idx"
        )
        tm.assert_index_equal(result, expected)

    def test_shift(self):
        pi1 = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="Y", start="1/1/2002", end="12/1/2010")

        tm.assert_index_equal(pi1.shift(0), pi1)

        assert len(pi1) == len(pi2)
        tm.assert_index_equal(pi1.shift(1), pi2)

        pi1 = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="Y", start="1/1/2000", end="12/1/2008")
        assert len(pi1) == len(pi2)
        tm.assert_index_equal(pi1.shift(-1), pi2)

        pi1 = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="M", start="2/1/2001", end="1/1/2010")
        assert len(pi1) == len(pi2)
        tm.assert_index_equal(pi1.shift(1), pi2)

        pi1 = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="M", start="12/1/2000", end="11/1/2009")
        assert len(pi1) == len(pi2)
        tm.assert_index_equal(pi1.shift(-1), pi2)

        pi1 = period_range(freq="D", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="D", start="1/2/2001", end="12/2/2009")
        assert len(pi1) == len(pi2)
        tm.assert_index_equal(pi1.shift(1), pi2)

        pi1 = period_range(freq="D", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="D", start="12/31/2000", end="11/30/2009")
        assert len(pi1) == len(pi2)
        tm.assert_index_equal(pi1.shift(-1), pi2)

    def test_shift_corner_cases(self):
        # GH#9903
        idx = PeriodIndex([], name="xxx", freq="h")

        msg = "`freq` argument is not supported for PeriodIndex.shift"
        with pytest.raises(TypeError, match=msg):
            # period shift doesn't accept freq
            idx.shift(1, freq="h")

        tm.assert_index_equal(idx.shift(0), idx)
        tm.assert_index_equal(idx.shift(3), idx)

        idx = PeriodIndex(
            ["2011-01-01 10:00", "2011-01-01 11:00", "2011-01-01 12:00"],
            name="xxx",
            freq="h",
        )
        tm.assert_index_equal(idx.shift(0), idx)
        exp = PeriodIndex(
            ["2011-01-01 13:00", "2011-01-01 14:00", "2011-01-01 15:00"],
            name="xxx",
            freq="h",
        )
        tm.assert_index_equal(idx.shift(3), exp)
        exp = PeriodIndex(
            ["2011-01-01 07:00", "2011-01-01 08:00", "2011-01-01 09:00"],
            name="xxx",
            freq="h",
        )
        tm.assert_index_equal(idx.shift(-3), exp)

    def test_shift_nat(self):
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="M", name="idx"
        )
        result = idx.shift(1)
        expected = PeriodIndex(
            ["2011-02", "2011-03", "NaT", "2011-05"], freq="M", name="idx"
        )
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

    def test_shift_gh8083(self):
        # test shift for PeriodIndex
        # GH#8083
        drange = period_range("20130101", periods=5, freq="D")
        result = drange.shift(1)
        expected = PeriodIndex(
            ["2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05", "2013-01-06"],
            freq="D",
        )
        tm.assert_index_equal(result, expected)

    def test_shift_periods(self):
        # GH #22458 : argument 'n' was deprecated in favor of 'periods'
        idx = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        tm.assert_index_equal(idx.shift(periods=0), idx)
        tm.assert_index_equal(idx.shift(0), idx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\tests\test_singularities.py ===
from sympy.core.numbers import (I, Rational, pi, oo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import sec, csc
from sympy.functions.elementary.hyperbolic import (coth, sech,
                                                   atanh, asech, acoth, acsch)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.calculus.singularities import (
    singularities,
    is_increasing,
    is_strictly_increasing,
    is_decreasing,
    is_strictly_decreasing,
    is_monotonic
)
from sympy.sets import Interval, FiniteSet, Union, ImageSet
from sympy.testing.pytest import raises
from sympy.abc import x, y


def test_singularities():
    x = Symbol('x')
    assert singularities(x**2, x) == S.EmptySet
    assert singularities(x/(x**2 + 3*x + 2), x) == FiniteSet(-2, -1)
    assert singularities(1/(x**2 + 1), x) == FiniteSet(I, -I)
    assert singularities(x/(x**3 + 1), x) == \
        FiniteSet(-1, (1 - sqrt(3) * I) / 2, (1 + sqrt(3) * I) / 2)
    assert singularities(1/(y**2 + 2*I*y + 1), y) == \
        FiniteSet(-I + sqrt(2)*I, -I - sqrt(2)*I)
    _n = Dummy('n')
    assert singularities(sech(x), x).dummy_eq(Union(
        ImageSet(Lambda(_n, 2*_n*I*pi + I*pi/2), S.Integers),
        ImageSet(Lambda(_n, 2*_n*I*pi + 3*I*pi/2), S.Integers)))
    assert singularities(coth(x), x).dummy_eq(Union(
        ImageSet(Lambda(_n, 2*_n*I*pi + I*pi), S.Integers),
        ImageSet(Lambda(_n, 2*_n*I*pi), S.Integers)))
    assert singularities(atanh(x), x) == FiniteSet(-1, 1)
    assert singularities(acoth(x), x) == FiniteSet(-1, 1)
    assert singularities(asech(x), x) == FiniteSet(0)
    assert singularities(acsch(x), x) == FiniteSet(0)

    x = Symbol('x', real=True)
    assert singularities(1/(x**2 + 1), x) == S.EmptySet
    assert singularities(exp(1/x), x, S.Reals) == FiniteSet(0)
    assert singularities(exp(1/x), x, Interval(1, 2)) == S.EmptySet
    assert singularities(log((x - 2)**2), x, Interval(1, 3)) == FiniteSet(2)
    raises(NotImplementedError, lambda: singularities(x**-oo, x))
    assert singularities(sec(x), x, Interval(0, 3*pi)) == FiniteSet(
        pi/2, 3*pi/2, 5*pi/2)
    assert singularities(csc(x), x, Interval(0, 3*pi)) == FiniteSet(
        0, pi, 2*pi, 3*pi)


def test_is_increasing():
    """Test whether is_increasing returns correct value."""
    a = Symbol('a', negative=True)

    assert is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    assert is_increasing(-x**2, Interval(-oo, 0))
    assert not is_increasing(-x**2, Interval(0, oo))
    assert not is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    assert is_increasing(x**2 + y, Interval(1, oo), x)
    assert is_increasing(-x**2*a, Interval(1, oo), x)
    assert is_increasing(1)

    assert is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3)) is False


def test_is_strictly_increasing():
    """Test whether is_strictly_increasing returns correct value."""
    assert is_strictly_increasing(
        4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    assert is_strictly_increasing(
        4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    assert not is_strictly_increasing(
        4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))
    assert not is_strictly_increasing(-x**2, Interval(0, oo))
    assert not is_strictly_decreasing(1)

    assert is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3)) is False


def test_is_decreasing():
    """Test whether is_decreasing returns correct value."""
    b = Symbol('b', positive=True)

    assert is_decreasing(1/(x**2 - 3*x), Interval.open(Rational(3,2), 3))
    assert is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    assert is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    assert not is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, Rational(3, 2)))
    assert not is_decreasing(-x**2, Interval(-oo, 0))
    assert not is_decreasing(-x**2*b, Interval(-oo, 0), x)


def test_is_strictly_decreasing():
    """Test whether is_strictly_decreasing returns correct value."""
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    assert not is_strictly_decreasing(
        1/(x**2 - 3*x), Interval.Ropen(-oo, Rational(3, 2)))
    assert not is_strictly_decreasing(-x**2, Interval(-oo, 0))
    assert not is_strictly_decreasing(1)
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.open(Rational(3,2), 3))
    assert is_strictly_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))


def test_is_monotonic():
    """Test whether is_monotonic returns correct value."""
    assert is_monotonic(1/(x**2 - 3*x), Interval.open(Rational(3,2), 3))
    assert is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    assert is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    assert is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    assert not is_monotonic(-x**2, S.Reals)
    assert is_monotonic(x**2 + y + 1, Interval(1, 2), x)
    raises(NotImplementedError, lambda: is_monotonic(x**2 + y + 1))


def test_issue_23401():
    x = Symbol('x')
    expr = (x + 1)/(-1.0e-3*x**2 + 0.1*x + 0.1)
    assert is_increasing(expr, Interval(1,2), x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\test_unique.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_unique(index_or_series_obj):
    obj = index_or_series_obj
    obj = np.repeat(obj, range(1, len(obj) + 1))
    result = obj.unique()

    # dict.fromkeys preserves the order
    unique_values = list(dict.fromkeys(obj.values))
    if isinstance(obj, pd.MultiIndex):
        expected = pd.MultiIndex.from_tuples(unique_values)
        expected.names = obj.names
        tm.assert_index_equal(result, expected, exact=True)
    elif isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values)
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_unique_null(null_obj, index_or_series_obj):
    obj = index_or_series_obj

    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(obj, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    values = obj._values
    values[0:2] = null_obj

    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)
    result = obj.unique()

    unique_values_raw = dict.fromkeys(obj.values)
    # because np.nan == np.nan is False, but None == None is True
    # np.nan would be duplicated, whereas None wouldn't
    unique_values_not_null = [val for val in unique_values_raw if not pd.isnull(val)]
    unique_values = [null_obj] + unique_values_not_null

    if isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            result = result.normalize()
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values, dtype=obj.dtype)
        tm.assert_numpy_array_equal(result, expected)


def test_nunique(index_or_series_obj):
    obj = index_or_series_obj
    obj = np.repeat(obj, range(1, len(obj) + 1))
    expected = len(obj.unique())
    assert obj.nunique(dropna=False) == expected


@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_nunique_null(null_obj, index_or_series_obj):
    obj = index_or_series_obj

    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif isinstance(obj, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    values = obj._values
    values[0:2] = null_obj

    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)

    if isinstance(obj, pd.CategoricalIndex):
        assert obj.nunique() == len(obj.categories)
        assert obj.nunique(dropna=False) == len(obj.categories) + 1
    else:
        num_unique_values = len(obj.unique())
        assert obj.nunique() == max(0, num_unique_values - 1)
        assert obj.nunique(dropna=False) == max(0, num_unique_values)


@pytest.mark.single_cpu
def test_unique_bad_unicode(index_or_series):
    # regression test for #34550
    uval = "\ud83d"  # smiley emoji

    obj = index_or_series([uval] * 2, dtype=object)
    result = obj.unique()

    if isinstance(obj, pd.Index):
        expected = pd.Index(["\ud83d"], dtype=object)
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(["\ud83d"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("dropna", [True, False])
def test_nunique_dropna(dropna):
    # GH37566
    ser = pd.Series(["yes", "yes", pd.NA, np.nan, None, pd.NaT])
    res = ser.nunique(dropna)
    assert res == 1 if dropna else 5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\indexing\test_insert.py ===
"""
test_insert is specifically for the DataFrame.insert method; not to be
confused with tests with "insert" in their names that are really testing
__setitem__.
"""
import numpy as np
import pytest

from pandas.errors import PerformanceWarning

from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


class TestDataFrameInsert:
    def test_insert(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=np.arange(5),
            columns=["c", "b", "a"],
        )

        df.insert(0, "foo", df["a"])
        tm.assert_index_equal(df.columns, Index(["foo", "c", "b", "a"]))
        tm.assert_series_equal(df["a"], df["foo"], check_names=False)

        df.insert(2, "bar", df["c"])
        tm.assert_index_equal(df.columns, Index(["foo", "c", "bar", "b", "a"]))
        tm.assert_almost_equal(df["c"], df["bar"], check_names=False)

        with pytest.raises(ValueError, match="already exists"):
            df.insert(1, "a", df["b"])

        msg = "cannot insert c, already exists"
        with pytest.raises(ValueError, match=msg):
            df.insert(1, "c", df["b"])

        df.columns.name = "some_name"
        # preserve columns name field
        df.insert(0, "baz", df["c"])
        assert df.columns.name == "some_name"

    def test_insert_column_bug_4032(self):
        # GH#4032, inserting a column and renaming causing errors
        df = DataFrame({"b": [1.1, 2.2]})

        df = df.rename(columns={})
        df.insert(0, "a", [1, 2])
        result = df.rename(columns={})

        expected = DataFrame([[1, 1.1], [2, 2.2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

        df.insert(0, "c", [1.3, 2.3])
        result = df.rename(columns={})

        expected = DataFrame([[1.3, 1, 1.1], [2.3, 2, 2.2]], columns=["c", "a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_insert_with_columns_dups(self):
        # GH#14291
        df = DataFrame()
        df.insert(0, "A", ["g", "h", "i"], allow_duplicates=True)
        df.insert(0, "A", ["d", "e", "f"], allow_duplicates=True)
        df.insert(0, "A", ["a", "b", "c"], allow_duplicates=True)
        exp = DataFrame(
            [["a", "d", "g"], ["b", "e", "h"], ["c", "f", "i"]],
            columns=Index(["A", "A", "A"], dtype=object),
        )
        tm.assert_frame_equal(df, exp)

    def test_insert_item_cache(self, using_array_manager, using_copy_on_write):
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
        ser = df[0]

        if using_array_manager:
            expected_warning = None
        else:
            # with BlockManager warn about high fragmentation of single dtype
            expected_warning = PerformanceWarning

        with tm.assert_produces_warning(expected_warning):
            for n in range(100):
                df[n + 3] = df[1] * n

        if using_copy_on_write:
            ser.iloc[0] = 99
            assert df.iloc[0, 0] == df[0][0]
            assert df.iloc[0, 0] != 99
        else:
            ser.values[0] = 99
            assert df.iloc[0, 0] == df[0][0]
            assert df.iloc[0, 0] == 99

    def test_insert_EA_no_warning(self):
        # PerformanceWarning about fragmented frame should not be raised when
        # using EAs (https://github.com/pandas-dev/pandas/issues/44098)
        df = DataFrame(
            np.random.default_rng(2).integers(0, 100, size=(3, 100)), dtype="Int64"
        )
        with tm.assert_produces_warning(None):
            df["a"] = np.array([1, 2, 3])

    def test_insert_frame(self):
        # GH#42403
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]})

        msg = (
            "Expected a one-dimensional object, got a DataFrame with 2 columns instead."
        )
        with pytest.raises(ValueError, match=msg):
            df.insert(1, "newcol", df)

    def test_insert_int64_loc(self):
        # GH#53193
        df = DataFrame({"a": [1, 2]})
        df.insert(np.int64(0), "b", 0)
        tm.assert_frame_equal(df, DataFrame({"b": [0, 0], "a": [1, 2]}))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\categorical\test_formats.py ===
"""
Tests for CategoricalIndex.__repr__ and related methods.
"""
import pytest

from pandas._config import using_string_dtype
import pandas._config.config as cf

from pandas import CategoricalIndex
import pandas._testing as tm


class TestCategoricalIndexRepr:
    def test_format_different_scalar_lengths(self):
        # GH#35439
        idx = CategoricalIndex(["aaaaaaaaa", "b"])
        expected = ["aaaaaaaaa", "b"]
        msg = r"CategoricalIndex\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert idx.format() == expected

    @pytest.mark.xfail(using_string_dtype(), reason="repr different")
    def test_string_categorical_index_repr(self):
        # short
        idx = CategoricalIndex(["a", "bb", "ccc"])
        expected = """CategoricalIndex(['a', 'bb', 'ccc'], categories=['a', 'bb', 'ccc'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

        # multiple lines
        idx = CategoricalIndex(["a", "bb", "ccc"] * 10)
        expected = """CategoricalIndex(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',
                  'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb',
                  'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],
                 categories=['a', 'bb', 'ccc'], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(idx) == expected

        # truncated
        idx = CategoricalIndex(["a", "bb", "ccc"] * 100)
        expected = """CategoricalIndex(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',
                  ...
                  'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],
                 categories=['a', 'bb', 'ccc'], ordered=False, dtype='category', length=300)"""  # noqa: E501

        assert repr(idx) == expected

        # larger categories
        idx = CategoricalIndex(list("abcdefghijklmmo"))
        expected = """CategoricalIndex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                  'm', 'm', 'o'],
                 categories=['a', 'b', 'c', 'd', ..., 'k', 'l', 'm', 'o'], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(idx) == expected

        # short
        idx = CategoricalIndex(["あ", "いい", "ううう"])
        expected = """CategoricalIndex(['あ', 'いい', 'ううう'], categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501
        assert repr(idx) == expected

        # multiple lines
        idx = CategoricalIndex(["あ", "いい", "ううう"] * 10)
        expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ',
                  'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(idx) == expected

        # truncated
        idx = CategoricalIndex(["あ", "いい", "ううう"] * 100)
        expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ',
                  ...
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category', length=300)"""  # noqa: E501

        assert repr(idx) == expected

        # larger categories
        idx = CategoricalIndex(list("あいうえおかきくけこさしすせそ"))
        expected = """CategoricalIndex(['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し',
                  'す', 'せ', 'そ'],
                 categories=['あ', 'い', 'う', 'え', ..., 'し', 'す', 'せ', 'そ'], ordered=False, dtype='category')"""  # noqa: E501

        assert repr(idx) == expected

        # Enable Unicode option -----------------------------------------
        with cf.option_context("display.unicode.east_asian_width", True):
            # short
            idx = CategoricalIndex(["あ", "いい", "ううう"])
            expected = """CategoricalIndex(['あ', 'いい', 'ううう'], categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501
            assert repr(idx) == expected

            # multiple lines
            idx = CategoricalIndex(["あ", "いい", "ううう"] * 10)
            expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',
                  'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category')"""  # noqa: E501

            assert repr(idx) == expected

            # truncated
            idx = CategoricalIndex(["あ", "いい", "ううう"] * 100)
            expected = """CategoricalIndex(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい',
                  'ううう', 'あ',
                  ...
                  'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',
                  'あ', 'いい', 'ううう'],
                 categories=['あ', 'いい', 'ううう'], ordered=False, dtype='category', length=300)"""  # noqa: E501

            assert repr(idx) == expected

            # larger categories
            idx = CategoricalIndex(list("あいうえおかきくけこさしすせそ"))
            expected = """CategoricalIndex(['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
                  'さ', 'し', 'す', 'せ', 'そ'],
                 categories=['あ', 'い', 'う', 'え', ..., 'し', 'す', 'せ', 'そ'], ordered=False, dtype='category')"""  # noqa: E501

            assert repr(idx) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_util.py ===
from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup,\
    AlternatingGroup
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.util import _check_cycles_alt_sym, _strip,\
    _distribute_gens_by_base, _strong_gens_from_distr,\
    _orbits_transversals_from_bsgs, _handle_precomputed_bsgs, _base_ordering,\
    _remove_gens
from sympy.combinatorics.testutil import _verify_bsgs


def test_check_cycles_alt_sym():
    perm1 = Permutation([[0, 1, 2, 3, 4, 5, 6], [7], [8], [9]])
    perm2 = Permutation([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9]])
    perm3 = Permutation([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert _check_cycles_alt_sym(perm1) is True
    assert _check_cycles_alt_sym(perm2) is False
    assert _check_cycles_alt_sym(perm3) is False


def test_strip():
    D = DihedralGroup(5)
    D.schreier_sims()
    member = Permutation([4, 0, 1, 2, 3])
    not_member1 = Permutation([0, 1, 4, 3, 2])
    not_member2 = Permutation([3, 1, 4, 2, 0])
    identity = Permutation([0, 1, 2, 3, 4])
    res1 = _strip(member, D.base, D.basic_orbits, D.basic_transversals)
    res2 = _strip(not_member1, D.base, D.basic_orbits, D.basic_transversals)
    res3 = _strip(not_member2, D.base, D.basic_orbits, D.basic_transversals)
    assert res1[0] == identity
    assert res1[1] == len(D.base) + 1
    assert res2[0] == not_member1
    assert res2[1] == len(D.base) + 1
    assert res3[0] != identity
    assert res3[1] == 2


def test_distribute_gens_by_base():
    base = [0, 1, 2]
    gens = [Permutation([0, 1, 2, 3]), Permutation([0, 1, 3, 2]),
           Permutation([0, 2, 3, 1]), Permutation([3, 2, 1, 0])]
    assert _distribute_gens_by_base(base, gens) == [gens,
                                                   [Permutation([0, 1, 2, 3]),
                                                   Permutation([0, 1, 3, 2]),
                                                   Permutation([0, 2, 3, 1])],
                                                   [Permutation([0, 1, 2, 3]),
                                                   Permutation([0, 1, 3, 2])]]


def test_strong_gens_from_distr():
    strong_gens_distr = [[Permutation([0, 2, 1]), Permutation([1, 2, 0]),
                  Permutation([1, 0, 2])], [Permutation([0, 2, 1])]]
    assert _strong_gens_from_distr(strong_gens_distr) == \
        [Permutation([0, 2, 1]),
         Permutation([1, 2, 0]),
         Permutation([1, 0, 2])]


def test_orbits_transversals_from_bsgs():
    S = SymmetricGroup(4)
    S.schreier_sims()
    base = S.base
    strong_gens = S.strong_gens
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    result = _orbits_transversals_from_bsgs(base, strong_gens_distr)
    orbits = result[0]
    transversals = result[1]
    base_len = len(base)
    for i in range(base_len):
        for el in orbits[i]:
            assert transversals[i][el](base[i]) == el
            for j in range(i):
                assert transversals[i][el](base[j]) == base[j]
    order = 1
    for i in range(base_len):
        order *= len(orbits[i])
    assert S.order() == order


def test_handle_precomputed_bsgs():
    A = AlternatingGroup(5)
    A.schreier_sims()
    base = A.base
    strong_gens = A.strong_gens
    result = _handle_precomputed_bsgs(base, strong_gens)
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    assert strong_gens_distr == result[2]
    transversals = result[0]
    orbits = result[1]
    base_len = len(base)
    for i in range(base_len):
        for el in orbits[i]:
            assert transversals[i][el](base[i]) == el
            for j in range(i):
                assert transversals[i][el](base[j]) == base[j]
    order = 1
    for i in range(base_len):
        order *= len(orbits[i])
    assert A.order() == order


def test_base_ordering():
    base = [2, 4, 5]
    degree = 7
    assert _base_ordering(base, degree) == [3, 4, 0, 5, 1, 2, 6]


def test_remove_gens():
    S = SymmetricGroup(10)
    base, strong_gens = S.schreier_sims_incremental()
    new_gens = _remove_gens(base, strong_gens)
    assert _verify_bsgs(S, base, new_gens) is True
    A = AlternatingGroup(7)
    base, strong_gens = A.schreier_sims_incremental()
    new_gens = _remove_gens(base, strong_gens)
    assert _verify_bsgs(A, base, new_gens) is True
    D = DihedralGroup(2)
    base, strong_gens = D.schreier_sims_incremental()
    new_gens = _remove_gens(base, strong_gens)
    assert _verify_bsgs(D, base, new_gens) is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_curve.py ===
from sympy.core.containers import Tuple
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Curve, Line, Point, Ellipse, Ray, Segment, Circle, Polygon, RegularPolygon
from sympy.testing.pytest import raises, slow


def test_curve():
    x = Symbol('x', real=True)
    s = Symbol('s')
    z = Symbol('z')

    # this curve is independent of the indicated parameter
    c = Curve([2*s, s**2], (z, 0, 2))

    assert c.parameter == z
    assert c.functions == (2*s, s**2)
    assert c.arbitrary_point() == Point(2*s, s**2)
    assert c.arbitrary_point(z) == Point(2*s, s**2)

    # this is how it is normally used
    c = Curve([2*s, s**2], (s, 0, 2))

    assert c.parameter == s
    assert c.functions == (2*s, s**2)
    t = Symbol('t')
    # the t returned as assumptions
    assert c.arbitrary_point() != Point(2*t, t**2)
    t = Symbol('t', real=True)
    # now t has the same assumptions so the test passes
    assert c.arbitrary_point() == Point(2*t, t**2)
    assert c.arbitrary_point(z) == Point(2*z, z**2)
    assert c.arbitrary_point(c.parameter) == Point(2*s, s**2)
    assert c.arbitrary_point(None) == Point(2*s, s**2)
    assert c.plot_interval() == [t, 0, 2]
    assert c.plot_interval(z) == [z, 0, 2]

    assert Curve([x, x], (x, 0, 1)).rotate(pi/2) == Curve([-x, x], (x, 0, 1))
    assert Curve([x, x], (x, 0, 1)).rotate(pi/2, (1, 2)).scale(2, 3).translate(
        1, 3).arbitrary_point(s) == \
        Line((0, 0), (1, 1)).rotate(pi/2, (1, 2)).scale(2, 3).translate(
            1, 3).arbitrary_point(s) == \
        Point(-2*s + 7, 3*s + 6)

    raises(ValueError, lambda: Curve((s), (s, 1, 2)))
    raises(ValueError, lambda: Curve((x, x * 2), (1, x)))

    raises(ValueError, lambda: Curve((s, s + t), (s, 1, 2)).arbitrary_point())
    raises(ValueError, lambda: Curve((s, s + t), (t, 1, 2)).arbitrary_point(s))


@slow
def test_free_symbols():
    a, b, c, d, e, f, s = symbols('a:f,s')
    assert Point(a, b).free_symbols == {a, b}
    assert Line((a, b), (c, d)).free_symbols == {a, b, c, d}
    assert Ray((a, b), (c, d)).free_symbols == {a, b, c, d}
    assert Ray((a, b), angle=c).free_symbols == {a, b, c}
    assert Segment((a, b), (c, d)).free_symbols == {a, b, c, d}
    assert Line((a, b), slope=c).free_symbols == {a, b, c}
    assert Curve((a*s, b*s), (s, c, d)).free_symbols == {a, b, c, d}
    assert Ellipse((a, b), c, d).free_symbols == {a, b, c, d}
    assert Ellipse((a, b), c, eccentricity=d).free_symbols == \
        {a, b, c, d}
    assert Ellipse((a, b), vradius=c, eccentricity=d).free_symbols == \
        {a, b, c, d}
    assert Circle((a, b), c).free_symbols == {a, b, c}
    assert Circle((a, b), (c, d), (e, f)).free_symbols == \
        {e, d, c, b, f, a}
    assert Polygon((a, b), (c, d), (e, f)).free_symbols == \
        {e, b, d, f, a, c}
    assert RegularPolygon((a, b), c, d, e).free_symbols == {e, a, b, c, d}


def test_transform():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    c = Curve((x, x**2), (x, 0, 1))
    cout = Curve((2*x - 4, 3*x**2 - 10), (x, 0, 1))
    pts = [Point(0, 0), Point(S.Half, Rational(1, 4)), Point(1, 1)]
    pts_out = [Point(-4, -10), Point(-3, Rational(-37, 4)), Point(-2, -7)]

    assert c.scale(2, 3, (4, 5)) == cout
    assert [c.subs(x, xi/2) for xi in Tuple(0, 1, 2)] == pts
    assert [cout.subs(x, xi/2) for xi in Tuple(0, 1, 2)] == pts_out
    assert Curve((x + y, 3*x), (x, 0, 1)).subs(y, S.Half) == \
        Curve((x + S.Half, 3*x), (x, 0, 1))
    assert Curve((x, 3*x), (x, 0, 1)).translate(4, 5) == \
        Curve((x + 4, 3*x + 5), (x, 0, 1))


def test_length():
    t = Symbol('t', real=True)

    c1 = Curve((t, 0), (t, 0, 1))
    assert c1.length == 1

    c2 = Curve((t, t), (t, 0, 1))
    assert c2.length == sqrt(2)

    c3 = Curve((t ** 2, t), (t, 2, 5))
    assert c3.length == -sqrt(17) - asinh(4) / 4 + asinh(10) / 4 + 5 * sqrt(101) / 2


def test_parameter_value():
    t = Symbol('t')
    C = Curve([2*t, t**2], (t, 0, 2))
    assert C.parameter_value((2, 1), t) == {t: 1}
    raises(ValueError, lambda: C.parameter_value((2, 0), t))


def test_issue_17997():
    t, s = symbols('t s')
    c = Curve((t, t**2), (t, 0, 10))
    p = Curve([2*s, s**2], (s, 0, 2))
    assert c(2) == Point(2, 4)
    assert p(1) == Point(2, 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_entity.py ===
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.geometry import (Circle, Ellipse, Point, Line, Parabola,
    Polygon, Ray, RegularPolygon, Segment, Triangle, Plane, Curve)
from sympy.geometry.entity import scale, GeometryEntity
from sympy.testing.pytest import raises


def test_entity():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)

    assert GeometryEntity(x, y) in GeometryEntity(x, y)
    raises(NotImplementedError, lambda: Point(0, 0) in GeometryEntity(x, y))

    assert GeometryEntity(x, y) == GeometryEntity(x, y)
    assert GeometryEntity(x, y).equals(GeometryEntity(x, y))

    c = Circle((0, 0), 5)
    assert GeometryEntity.encloses(c, Point(0, 0))
    assert GeometryEntity.encloses(c, Segment((0, 0), (1, 1)))
    assert GeometryEntity.encloses(c, Line((0, 0), (1, 1))) is False
    assert GeometryEntity.encloses(c, Circle((0, 0), 4))
    assert GeometryEntity.encloses(c, Polygon(Point(0, 0), Point(1, 0), Point(0, 1)))
    assert GeometryEntity.encloses(c, RegularPolygon(Point(8, 8), 1, 3)) is False


def test_svg():
    a = Symbol('a')
    b = Symbol('b')
    d = Symbol('d')

    entity = Circle(Point(a, b), d)
    assert entity._repr_svg_() is None

    entity = Circle(Point(0, 0), S.Infinity)
    assert entity._repr_svg_() is None


def test_subs():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    p = Point(x, 2)
    q = Point(1, 1)
    r = Point(3, 4)
    for o in [p,
              Segment(p, q),
              Ray(p, q),
              Line(p, q),
              Triangle(p, q, r),
              RegularPolygon(p, 3, 6),
              Polygon(p, q, r, Point(5, 4)),
              Circle(p, 3),
              Ellipse(p, 3, 4)]:
        assert 'y' in str(o.subs(x, y))
    assert p.subs({x: 1}) == Point(1, 2)
    assert Point(1, 2).subs(Point(1, 2), Point(3, 4)) == Point(3, 4)
    assert Point(1, 2).subs((1, 2), Point(3, 4)) == Point(3, 4)
    assert Point(1, 2).subs(Point(1, 2), Point(3, 4)) == Point(3, 4)
    assert Point(1, 2).subs({(1, 2)}) == Point(2, 2)
    raises(ValueError, lambda: Point(1, 2).subs(1))
    raises(TypeError, lambda: Point(1, 1).subs((Point(1, 1), Point(1,
           2)), 1, 2))


def test_transform():
    assert scale(1, 2, (3, 4)).tolist() == \
        [[1, 0, 0], [0, 2, 0], [0, -4, 1]]


def test_reflect_entity_overrides():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    b = Symbol('b')
    m = Symbol('m')
    l = Line((0, b), slope=m)
    p = Point(x, y)
    r = p.reflect(l)
    c = Circle((x, y), 3)
    cr = c.reflect(l)
    assert cr == Circle(r, -3)
    assert c.area == -cr.area

    pent = RegularPolygon((1, 2), 1, 5)
    slope = S.ComplexInfinity
    while slope is S.ComplexInfinity:
        slope = Rational(*(x._random()/2).as_real_imag())
    l = Line(pent.vertices[1], slope=slope)
    rpent = pent.reflect(l)
    assert rpent.center == pent.center.reflect(l)
    rvert = [i.reflect(l) for i in pent.vertices]
    for v in rpent.vertices:
        for i in range(len(rvert)):
            ri = rvert[i]
            if ri.equals(v):
                rvert.remove(ri)
                break
    assert not rvert
    assert pent.area.equals(-rpent.area)


def test_geometry_EvalfMixin():
    x = pi
    t = Symbol('t')
    for g in [
            Point(x, x),
            Plane(Point(0, x, 0), (0, 0, x)),
            Curve((x*t, x), (t, 0, x)),
            Ellipse((x, x), x, -x),
            Circle((x, x), x),
            Line((0, x), (x, 0)),
            Segment((0, x), (x, 0)),
            Ray((0, x), (x, 0)),
            Parabola((0, x), Line((-x, 0), (x, 0))),
            Polygon((0, 0), (0, x), (x, 0), (x, x)),
            RegularPolygon((0, x), x, 4, x),
            Triangle((0, 0), (x, 0), (x, x)),
            ]:
        assert str(g).replace('pi', '3.1') == str(g.n(2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_period.py ===
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

from pandas._libs import (
    Period,
    iNaT,
)
from pandas.compat import is_platform_windows
from pandas.compat.numpy import np_version_gte1p24

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas._testing as tm
from pandas.core.arrays import PeriodArray
from pandas.tests.extension import base

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture(params=["D", "2D"])
def dtype(request):
    return PeriodDtype(freq=request.param)


@pytest.fixture
def data(dtype):
    return PeriodArray(np.arange(1970, 2070), dtype=dtype)


@pytest.fixture
def data_for_sorting(dtype):
    return PeriodArray([2018, 2019, 2017], dtype=dtype)


@pytest.fixture
def data_missing(dtype):
    return PeriodArray([iNaT, 2017], dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    return PeriodArray([2018, iNaT, 2017], dtype=dtype)


@pytest.fixture
def data_for_grouping(dtype):
    B = 2018
    NA = iNaT
    A = 2017
    C = 2019
    return PeriodArray([B, B, NA, NA, A, A, B, C], dtype=dtype)


class TestPeriodArray(base.ExtensionTests):
    def _get_expected_exception(self, op_name, obj, other):
        if op_name in ("__sub__", "__rsub__"):
            return None
        return super()._get_expected_exception(op_name, obj, other)

    def _supports_accumulation(self, ser, op_name: str) -> bool:
        return op_name in ["cummin", "cummax"]

    def _supports_reduction(self, obj, op_name: str) -> bool:
        return op_name in ["min", "max", "median"]

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        if op_name == "median":
            res_op = getattr(ser, op_name)

            alt = ser.astype("int64")

            exp_op = getattr(alt, op_name)
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
            # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
            # attribute "freq"
            freq = ser.dtype.freq  # type: ignore[union-attr]
            expected = Period._from_ordinal(int(expected), freq=freq)
            tm.assert_almost_equal(result, expected)

        else:
            return super().check_reduce(ser, op_name, skipna)

    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods):
        if is_platform_windows() and np_version_gte1p24:
            with tm.assert_produces_warning(RuntimeWarning, check_stacklevel=False):
                super().test_diff(data, periods)
        else:
            super().test_diff(data, periods)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        result = data.map(lambda x: x, na_action=na_action)
        tm.assert_extension_array_equal(result, data)


class Test2DCompat(base.NDArrayBacked2DTests):
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_formats.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestIntervalIndexRendering:
    # TODO: this is a test for DataFrame/Series, not IntervalIndex
    @pytest.mark.parametrize(
        "constructor,expected",
        [
            (
                Series,
                (
                    "(0.0, 1.0]    a\n"
                    "NaN           b\n"
                    "(2.0, 3.0]    c\n"
                    "dtype: object"
                ),
            ),
            (DataFrame, ("            0\n(0.0, 1.0]  a\nNaN         b\n(2.0, 3.0]  c")),
        ],
    )
    def test_repr_missing(self, constructor, expected, using_infer_string, request):
        # GH 25984
        if using_infer_string and constructor is Series:
            request.applymarker(pytest.mark.xfail(reason="repr different"))
        index = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        obj = constructor(list("abc"), index=index)
        result = repr(obj)
        assert result == expected

    def test_repr_floats(self):
        # GH 32553

        markers = Series(
            [1, 2],
            index=IntervalIndex(
                [
                    Interval(left, right)
                    for left, right in zip(
                        Index([329.973, 345.137], dtype="float64"),
                        Index([345.137, 360.191], dtype="float64"),
                    )
                ]
            ),
        )
        result = str(markers)
        expected = "(329.973, 345.137]    1\n(345.137, 360.191]    2\ndtype: int64"
        assert result == expected

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    @pytest.mark.parametrize(
        "tuples, closed, expected_data",
        [
            ([(0, 1), (1, 2), (2, 3)], "left", ["[0, 1)", "[1, 2)", "[2, 3)"]),
            (
                [(0.5, 1.0), np.nan, (2.0, 3.0)],
                "right",
                ["(0.5, 1.0]", "NaN", "(2.0, 3.0]"],
            ),
            (
                [
                    (Timestamp("20180101"), Timestamp("20180102")),
                    np.nan,
                    ((Timestamp("20180102"), Timestamp("20180103"))),
                ],
                "both",
                [
                    "[2018-01-01 00:00:00, 2018-01-02 00:00:00]",
                    "NaN",
                    "[2018-01-02 00:00:00, 2018-01-03 00:00:00]",
                ],
            ),
            (
                [
                    (Timedelta("0 days"), Timedelta("1 days")),
                    (Timedelta("1 days"), Timedelta("2 days")),
                    np.nan,
                ],
                "neither",
                [
                    "(0 days 00:00:00, 1 days 00:00:00)",
                    "(1 days 00:00:00, 2 days 00:00:00)",
                    "NaN",
                ],
            ),
        ],
    )
    def test_get_values_for_csv(self, tuples, closed, expected_data):
        # GH 28210
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index._get_values_for_csv(na_rep="NaN")
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)

    def test_timestamp_with_timezone(self, unit):
        # GH 55035
        left = DatetimeIndex(["2020-01-01"], dtype=f"M8[{unit}, UTC]")
        right = DatetimeIndex(["2020-01-02"], dtype=f"M8[{unit}, UTC]")
        index = IntervalIndex.from_arrays(left, right)
        result = repr(index)
        expected = (
            "IntervalIndex([(2020-01-01 00:00:00+00:00, 2020-01-02 00:00:00+00:00]], "
            f"dtype='interval[datetime64[{unit}, UTC], right]')"
        )
        assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\holiday\test_calendar.py ===
from datetime import datetime

import pytest

from pandas import (
    DatetimeIndex,
    offsets,
    to_datetime,
)
import pandas._testing as tm

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    Timestamp,
    USFederalHolidayCalendar,
    USLaborDay,
    USThanksgivingDay,
    get_calendar,
)


@pytest.mark.parametrize(
    "transform", [lambda x: x, lambda x: x.strftime("%Y-%m-%d"), lambda x: Timestamp(x)]
)
def test_calendar(transform):
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2012, 12, 31)

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(transform(start_date), transform(end_date))

    expected = [
        datetime(2012, 1, 2),
        datetime(2012, 1, 16),
        datetime(2012, 2, 20),
        datetime(2012, 5, 28),
        datetime(2012, 7, 4),
        datetime(2012, 9, 3),
        datetime(2012, 10, 8),
        datetime(2012, 11, 12),
        datetime(2012, 11, 22),
        datetime(2012, 12, 25),
    ]

    assert list(holidays.to_pydatetime()) == expected


def test_calendar_caching():
    # see gh-9552.

    class TestCalendar(AbstractHolidayCalendar):
        def __init__(self, name=None, rules=None) -> None:
            super().__init__(name=name, rules=rules)

    jan1 = TestCalendar(rules=[Holiday("jan1", year=2015, month=1, day=1)])
    jan2 = TestCalendar(rules=[Holiday("jan2", year=2015, month=1, day=2)])

    # Getting holidays for Jan 1 should not alter results for Jan 2.
    expected = DatetimeIndex(["01-Jan-2015"]).as_unit("ns")
    tm.assert_index_equal(jan1.holidays(), expected)

    expected2 = DatetimeIndex(["02-Jan-2015"]).as_unit("ns")
    tm.assert_index_equal(jan2.holidays(), expected2)


def test_calendar_observance_dates():
    # see gh-11477
    us_fed_cal = get_calendar("USFederalHolidayCalendar")
    holidays0 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 3)
    )  # <-- same start and end dates
    holidays1 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 6)
    )  # <-- different start and end dates
    holidays2 = us_fed_cal.holidays(
        datetime(2015, 7, 3), datetime(2015, 7, 3)
    )  # <-- same start and end dates

    # These should all produce the same result.
    #
    # In addition, calling with different start and end
    # dates should not alter the output if we call the
    # function again with the same start and end date.
    tm.assert_index_equal(holidays0, holidays1)
    tm.assert_index_equal(holidays0, holidays2)


def test_rule_from_name():
    us_fed_cal = get_calendar("USFederalHolidayCalendar")
    assert us_fed_cal.rule_from_name("Thanksgiving Day") == USThanksgivingDay


def test_calendar_2031():
    # See gh-27790
    #
    # Labor Day 2031 is on September 1. Saturday before is August 30.
    # Next working day after August 30 ought to be Tuesday, September 2.

    class testCalendar(AbstractHolidayCalendar):
        rules = [USLaborDay]

    cal = testCalendar()
    workDay = offsets.CustomBusinessDay(calendar=cal)
    Sat_before_Labor_Day_2031 = to_datetime("2031-08-30")
    next_working_day = Sat_before_Labor_Day_2031 + 0 * workDay
    assert next_working_day == to_datetime("2031-09-02")


def test_no_holidays_calendar():
    # Test for issue #31415

    class NoHolidaysCalendar(AbstractHolidayCalendar):
        pass

    cal = NoHolidaysCalendar()
    holidays = cal.holidays(Timestamp("01-Jan-2020"), Timestamp("01-Jan-2021"))
    empty_index = DatetimeIndex([])  # Type is DatetimeIndex since return_name=False
    tm.assert_index_equal(holidays, empty_index)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_parse_iso8601.py ===
from datetime import datetime

import pytest

from pandas._libs import tslib

from pandas import Timestamp


@pytest.mark.parametrize(
    "date_str, exp",
    [
        ("2011-01-02", datetime(2011, 1, 2)),
        ("2011-1-2", datetime(2011, 1, 2)),
        ("2011-01", datetime(2011, 1, 1)),
        ("2011-1", datetime(2011, 1, 1)),
        ("2011 01 02", datetime(2011, 1, 2)),
        ("2011.01.02", datetime(2011, 1, 2)),
        ("2011/01/02", datetime(2011, 1, 2)),
        ("2011\\01\\02", datetime(2011, 1, 2)),
        ("2013-01-01 05:30:00", datetime(2013, 1, 1, 5, 30)),
        ("2013-1-1 5:30:00", datetime(2013, 1, 1, 5, 30)),
        ("2013-1-1 5:30:00+01:00", Timestamp(2013, 1, 1, 5, 30, tz="UTC+01:00")),
    ],
)
def test_parsers_iso8601(date_str, exp):
    # see gh-12060
    #
    # Test only the ISO parser - flexibility to
    # different separators and leading zero's.
    actual = tslib._test_parse_iso8601(date_str)
    assert actual == exp


@pytest.mark.parametrize(
    "date_str",
    [
        "2011-01/02",
        "2011=11=11",
        "201401",
        "201111",
        "200101",
        # Mixed separated and unseparated.
        "2005-0101",
        "200501-01",
        "20010101 12:3456",
        "20010101 1234:56",
        # HHMMSS must have two digits in
        # each component if unseparated.
        "20010101 1",
        "20010101 123",
        "20010101 12345",
        "20010101 12345Z",
    ],
)
def test_parsers_iso8601_invalid(date_str):
    msg = f'Error parsing datetime string "{date_str}"'

    with pytest.raises(ValueError, match=msg):
        tslib._test_parse_iso8601(date_str)


def test_parsers_iso8601_invalid_offset_invalid():
    date_str = "2001-01-01 12-34-56"
    msg = f'Timezone hours offset out of range in datetime string "{date_str}"'

    with pytest.raises(ValueError, match=msg):
        tslib._test_parse_iso8601(date_str)


def test_parsers_iso8601_leading_space():
    # GH#25895 make sure isoparser doesn't overflow with long input
    date_str, expected = ("2013-1-1 5:30:00", datetime(2013, 1, 1, 5, 30))
    actual = tslib._test_parse_iso8601(" " * 200 + date_str)
    assert actual == expected


@pytest.mark.parametrize(
    "date_str, timespec, exp",
    [
        ("2023-01-01 00:00:00", "auto", "2023-01-01T00:00:00"),
        ("2023-01-01 00:00:00", "seconds", "2023-01-01T00:00:00"),
        ("2023-01-01 00:00:00", "milliseconds", "2023-01-01T00:00:00.000"),
        ("2023-01-01 00:00:00", "microseconds", "2023-01-01T00:00:00.000000"),
        ("2023-01-01 00:00:00", "nanoseconds", "2023-01-01T00:00:00.000000000"),
        ("2023-01-01 00:00:00.001", "auto", "2023-01-01T00:00:00.001000"),
        ("2023-01-01 00:00:00.001", "seconds", "2023-01-01T00:00:00"),
        ("2023-01-01 00:00:00.001", "milliseconds", "2023-01-01T00:00:00.001"),
        ("2023-01-01 00:00:00.001", "microseconds", "2023-01-01T00:00:00.001000"),
        ("2023-01-01 00:00:00.001", "nanoseconds", "2023-01-01T00:00:00.001000000"),
        ("2023-01-01 00:00:00.000001", "auto", "2023-01-01T00:00:00.000001"),
        ("2023-01-01 00:00:00.000001", "seconds", "2023-01-01T00:00:00"),
        ("2023-01-01 00:00:00.000001", "milliseconds", "2023-01-01T00:00:00.000"),
        ("2023-01-01 00:00:00.000001", "microseconds", "2023-01-01T00:00:00.000001"),
        ("2023-01-01 00:00:00.000001", "nanoseconds", "2023-01-01T00:00:00.000001000"),
        ("2023-01-01 00:00:00.000000001", "auto", "2023-01-01T00:00:00.000000001"),
        ("2023-01-01 00:00:00.000000001", "seconds", "2023-01-01T00:00:00"),
        ("2023-01-01 00:00:00.000000001", "milliseconds", "2023-01-01T00:00:00.000"),
        ("2023-01-01 00:00:00.000000001", "microseconds", "2023-01-01T00:00:00.000000"),
        (
            "2023-01-01 00:00:00.000000001",
            "nanoseconds",
            "2023-01-01T00:00:00.000000001",
        ),
        ("2023-01-01 00:00:00.000001001", "auto", "2023-01-01T00:00:00.000001001"),
        ("2023-01-01 00:00:00.000001001", "seconds", "2023-01-01T00:00:00"),
        ("2023-01-01 00:00:00.000001001", "milliseconds", "2023-01-01T00:00:00.000"),
        ("2023-01-01 00:00:00.000001001", "microseconds", "2023-01-01T00:00:00.000001"),
        (
            "2023-01-01 00:00:00.000001001",
            "nanoseconds",
            "2023-01-01T00:00:00.000001001",
        ),
    ],
)
def test_iso8601_formatter(date_str: str, timespec: str, exp: str):
    # GH#53020
    ts = Timestamp(date_str)
    assert ts.isoformat(timespec=timespec) == exp

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_traversal.py ===
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import symbols
from sympy.core.singleton import S
from sympy.core.function import expand, Function
from sympy.core.numbers import I
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor
from sympy.core.traversal import preorder_traversal, use, postorder_traversal, iterargs, iterfreeargs
from sympy.functions.elementary.piecewise import ExprCondPair, Piecewise
from sympy.testing.pytest import warns_deprecated_sympy
from sympy.utilities.iterables import capture

b1 = Basic()
b2 = Basic(b1)
b3 = Basic(b2)
b21 = Basic(b2, b1)


def test_preorder_traversal():
    expr = Basic(b21, b3)
    assert list(
        preorder_traversal(expr)) == [expr, b21, b2, b1, b1, b3, b2, b1]
    assert list(preorder_traversal(('abc', ('d', 'ef')))) == [
        ('abc', ('d', 'ef')), 'abc', ('d', 'ef'), 'd', 'ef']

    result = []
    pt = preorder_traversal(expr)
    for i in pt:
        result.append(i)
        if i == b2:
            pt.skip()
    assert result == [expr, b21, b2, b1, b3, b2]

    w, x, y, z = symbols('w:z')
    expr = z + w*(x + y)
    assert list(preorder_traversal([expr], keys=default_sort_key)) == \
        [[w*(x + y) + z], w*(x + y) + z, z, w*(x + y), w, x + y, x, y]
    assert list(preorder_traversal((x + y)*z, keys=True)) == \
        [z*(x + y), z, x + y, x, y]


def test_use():
    x, y = symbols('x y')

    assert use(0, expand) == 0

    f = (x + y)**2*x + 1

    assert use(f, expand, level=0) == x**3 + 2*x**2*y + x*y**2 + + 1
    assert use(f, expand, level=1) == x**3 + 2*x**2*y + x*y**2 + + 1
    assert use(f, expand, level=2) == 1 + x*(2*x*y + x**2 + y**2)
    assert use(f, expand, level=3) == (x + y)**2*x + 1

    f = (x**2 + 1)**2 - 1
    kwargs = {'gaussian': True}

    assert use(f, factor, level=0, kwargs=kwargs) == x**2*(x**2 + 2)
    assert use(f, factor, level=1, kwargs=kwargs) == (x + I)**2*(x - I)**2 - 1
    assert use(f, factor, level=2, kwargs=kwargs) == (x + I)**2*(x - I)**2 - 1
    assert use(f, factor, level=3, kwargs=kwargs) == (x**2 + 1)**2 - 1


def test_postorder_traversal():
    x, y, z, w = symbols('x y z w')
    expr = z + w*(x + y)
    expected = [z, w, x, y, x + y, w*(x + y), w*(x + y) + z]
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    assert list(postorder_traversal(expr, keys=True)) == expected

    expr = Piecewise((x, x < 1), (x**2, True))
    expected = [
        x, 1, x, x < 1, ExprCondPair(x, x < 1),
        2, x, x**2, S.true,
        ExprCondPair(x**2, True), Piecewise((x, x < 1), (x**2, True))
    ]
    assert list(postorder_traversal(expr, keys=default_sort_key)) == expected
    assert list(postorder_traversal(
        [expr], keys=default_sort_key)) == expected + [[expr]]

    assert list(postorder_traversal(Integral(x**2, (x, 0, 1)),
        keys=default_sort_key)) == [
            2, x, x**2, 0, 1, x, Tuple(x, 0, 1),
            Integral(x**2, Tuple(x, 0, 1))
        ]
    assert list(postorder_traversal(('abc', ('d', 'ef')))) == [
        'abc', 'd', 'ef', ('d', 'ef'), ('abc', ('d', 'ef'))]


def test_iterargs():
    f = Function('f')
    x = symbols('x')
    assert list(iterfreeargs(Integral(f(x), (f(x), 1)))) == [
        Integral(f(x), (f(x), 1)), 1]
    assert list(iterargs(Integral(f(x), (f(x), 1)))) == [
        Integral(f(x), (f(x), 1)), f(x), (f(x), 1), x, f(x), 1, x]

def test_deprecated_imports():
    x = symbols('x')

    with warns_deprecated_sympy():
        from sympy.core.basic import preorder_traversal
        preorder_traversal(x)
    with warns_deprecated_sympy():
        from sympy.simplify.simplify import bottom_up
        bottom_up(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.simplify.simplify import walk
        walk(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.simplify.traversaltools import use
        use(x, lambda x: x)
    with warns_deprecated_sympy():
        from sympy.utilities.iterables import postorder_traversal
        postorder_traversal(x)
    with warns_deprecated_sympy():
        from sympy.utilities.iterables import interactive_traversal
        capture(lambda: interactive_traversal(x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_indexing_slow.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


@pytest.fixture
def m():
    return 5


@pytest.fixture
def n():
    return 100


@pytest.fixture
def cols():
    return ["jim", "joe", "jolie", "joline", "jolia"]


@pytest.fixture
def vals(n):
    vals = [
        np.random.default_rng(2).integers(0, 10, n),
        np.random.default_rng(2).choice(list("abcdefghij"), n),
        np.random.default_rng(2).choice(
            pd.date_range("20141009", periods=10).tolist(), n
        ),
        np.random.default_rng(2).choice(list("ZYXWVUTSRQ"), n),
        np.random.default_rng(2).standard_normal(n),
    ]
    vals = list(map(tuple, zip(*vals)))
    return vals


@pytest.fixture
def keys(n, m, vals):
    # bunch of keys for testing
    keys = [
        np.random.default_rng(2).integers(0, 11, m),
        np.random.default_rng(2).choice(list("abcdefghijk"), m),
        np.random.default_rng(2).choice(
            pd.date_range("20141009", periods=11).tolist(), m
        ),
        np.random.default_rng(2).choice(list("ZYXWVUTSRQP"), m),
    ]
    keys = list(map(tuple, zip(*keys)))
    keys += [t[:-1] for t in vals[:: n // m]]
    return keys


# covers both unique index and non-unique index
@pytest.fixture
def df(vals, cols):
    return DataFrame(vals, columns=cols)


@pytest.fixture
def a(df):
    return pd.concat([df, df])


@pytest.fixture
def b(df, cols):
    return df.drop_duplicates(subset=cols[:-1])


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
@pytest.mark.parametrize("lexsort_depth", list(range(5)))
@pytest.mark.parametrize("frame_fixture", ["a", "b"])
def test_multiindex_get_loc(request, lexsort_depth, keys, frame_fixture, cols):
    # GH7724, GH2646

    frame = request.getfixturevalue(frame_fixture)
    if lexsort_depth == 0:
        df = frame.copy(deep=False)
    else:
        df = frame.sort_values(by=cols[:lexsort_depth])

    mi = df.set_index(cols[:-1])
    assert not mi.index._lexsort_depth < lexsort_depth
    for key in keys:
        mask = np.ones(len(df), dtype=bool)

        # test for all partials of this key
        for i, k in enumerate(key):
            mask &= df.iloc[:, i] == k

            if not mask.any():
                assert key[: i + 1] not in mi.index
                continue

            assert key[: i + 1] in mi.index
            right = df[mask].copy(deep=False)

            if i + 1 != len(key):  # partial key
                return_value = right.drop(cols[: i + 1], axis=1, inplace=True)
                assert return_value is None
                return_value = right.set_index(cols[i + 1 : -1], inplace=True)
                assert return_value is None
                tm.assert_frame_equal(mi.loc[key[: i + 1]], right)

            else:  # full key
                return_value = right.set_index(cols[:-1], inplace=True)
                assert return_value is None
                if len(right) == 1:  # single hit
                    right = Series(
                        right["jolia"].values, name=right.index[0], index=["jolia"]
                    )
                    tm.assert_series_equal(mi.loc[key[: i + 1]], right)
                else:  # multi hit
                    tm.assert_frame_equal(mi.loc[key[: i + 1]], right)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\concat\test_sort.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm


class TestConcatSort:
    def test_concat_sorts_columns(self, sort):
        # GH-4588
        df1 = DataFrame({"a": [1, 2], "b": [1, 2]}, columns=["b", "a"])
        df2 = DataFrame({"a": [3, 4], "c": [5, 6]})

        # for sort=True/None
        expected = DataFrame(
            {"a": [1, 2, 3, 4], "b": [1, 2, None, None], "c": [None, None, 5, 6]},
            columns=["a", "b", "c"],
        )

        if sort is False:
            expected = expected[["b", "a", "c"]]

        # default
        with tm.assert_produces_warning(None):
            result = pd.concat([df1, df2], ignore_index=True, sort=sort)
        tm.assert_frame_equal(result, expected)

    def test_concat_sorts_index(self, sort):
        df1 = DataFrame({"a": [1, 2, 3]}, index=["c", "a", "b"])
        df2 = DataFrame({"b": [1, 2]}, index=["a", "b"])

        # For True/None
        expected = DataFrame(
            {"a": [2, 3, 1], "b": [1, 2, None]},
            index=["a", "b", "c"],
            columns=["a", "b"],
        )
        if sort is False:
            expected = expected.loc[["c", "a", "b"]]

        # Warn and sort by default
        with tm.assert_produces_warning(None):
            result = pd.concat([df1, df2], axis=1, sort=sort)
        tm.assert_frame_equal(result, expected)

    def test_concat_inner_sort(self, sort):
        # https://github.com/pandas-dev/pandas/pull/20613
        df1 = DataFrame(
            {"a": [1, 2], "b": [1, 2], "c": [1, 2]}, columns=["b", "a", "c"]
        )
        df2 = DataFrame({"a": [1, 2], "b": [3, 4]}, index=[3, 4])

        with tm.assert_produces_warning(None):
            # unset sort should *not* warn for inner join
            # since that never sorted
            result = pd.concat([df1, df2], sort=sort, join="inner", ignore_index=True)

        expected = DataFrame({"b": [1, 2, 3, 4], "a": [1, 2, 1, 2]}, columns=["b", "a"])
        if sort is True:
            expected = expected[["a", "b"]]
        tm.assert_frame_equal(result, expected)

    def test_concat_aligned_sort(self):
        # GH-4588
        df = DataFrame({"c": [1, 2], "b": [3, 4], "a": [5, 6]}, columns=["c", "b", "a"])
        result = pd.concat([df, df], sort=True, ignore_index=True)
        expected = DataFrame(
            {"a": [5, 6, 5, 6], "b": [3, 4, 3, 4], "c": [1, 2, 1, 2]},
            columns=["a", "b", "c"],
        )
        tm.assert_frame_equal(result, expected)

        result = pd.concat(
            [df, df[["c", "b"]]], join="inner", sort=True, ignore_index=True
        )
        expected = expected[["b", "c"]]
        tm.assert_frame_equal(result, expected)

    def test_concat_aligned_sort_does_not_raise(self):
        # GH-4588
        # We catch TypeErrors from sorting internally and do not re-raise.
        df = DataFrame({1: [1, 2], "a": [3, 4]}, columns=[1, "a"])
        expected = DataFrame({1: [1, 2, 1, 2], "a": [3, 4, 3, 4]}, columns=[1, "a"])
        result = pd.concat([df, df], ignore_index=True, sort=True)
        tm.assert_frame_equal(result, expected)

    def test_concat_frame_with_sort_false(self):
        # GH 43375
        result = pd.concat(
            [DataFrame({i: i}, index=[i]) for i in range(2, 0, -1)], sort=False
        )
        expected = DataFrame([[2, np.nan], [np.nan, 1]], index=[2, 1], columns=[2, 1])

        tm.assert_frame_equal(result, expected)

        # GH 37937
        df1 = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[1, 2, 3])
        df2 = DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]}, index=[3, 1, 6])
        result = pd.concat([df2, df1], axis=1, sort=False)
        expected = DataFrame(
            [
                [7.0, 10.0, 3.0, 6.0],
                [8.0, 11.0, 1.0, 4.0],
                [9.0, 12.0, np.nan, np.nan],
                [np.nan, np.nan, 2.0, 5.0],
            ],
            index=[3, 1, 6, 2],
            columns=["c", "d", "a", "b"],
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_sort_none_raises(self):
        # GH#41518
        df = DataFrame({1: [1, 2], "a": [3, 4]})
        msg = "The 'sort' keyword only accepts boolean values; None was passed."
        with pytest.raises(ValueError, match=msg):
            pd.concat([df, df], sort=None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_partitions.py ===
from sympy.core.sorting import ordered, default_sort_key
from sympy.combinatorics.partitions import (Partition, IntegerPartition,
                                            RGS_enum, RGS_unrank, RGS_rank,
                                            random_integer_partition)
from sympy.testing.pytest import raises
from sympy.utilities.iterables import partitions
from sympy.sets.sets import Set, FiniteSet


def test_partition_constructor():
    raises(ValueError, lambda: Partition([1, 1, 2]))
    raises(ValueError, lambda: Partition([1, 2, 3], [2, 3, 4]))
    raises(ValueError, lambda: Partition(1, 2, 3))
    raises(ValueError, lambda: Partition(*list(range(3))))

    assert Partition([1, 2, 3], [4, 5]) == Partition([4, 5], [1, 2, 3])
    assert Partition({1, 2, 3}, {4, 5}) == Partition([1, 2, 3], [4, 5])

    a = FiniteSet(1, 2, 3)
    b = FiniteSet(4, 5)
    assert Partition(a, b) == Partition([1, 2, 3], [4, 5])
    assert Partition({a, b}) == Partition(FiniteSet(a, b))
    assert Partition({a, b}) != Partition(a, b)

def test_partition():
    from sympy.abc import x

    a = Partition([1, 2, 3], [4])
    b = Partition([1, 2], [3, 4])
    c = Partition([x])
    l = [a, b, c]
    l.sort(key=default_sort_key)
    assert l == [c, a, b]
    l.sort(key=lambda w: default_sort_key(w, order='rev-lex'))
    assert l == [c, a, b]

    assert (a == b) is False
    assert a <= b
    assert (a > b) is False
    assert a != b
    assert a < b

    assert (a + 2).partition == [[1, 2], [3, 4]]
    assert (b - 1).partition == [[1, 2, 4], [3]]

    assert (a - 1).partition == [[1, 2, 3, 4]]
    assert (a + 1).partition == [[1, 2, 4], [3]]
    assert (b + 1).partition == [[1, 2], [3], [4]]

    assert a.rank == 1
    assert b.rank == 3

    assert a.RGS == (0, 0, 0, 1)
    assert b.RGS == (0, 0, 1, 1)


def test_integer_partition():
    # no zeros in partition
    raises(ValueError, lambda: IntegerPartition(list(range(3))))
    # check fails since 1 + 2 != 100
    raises(ValueError, lambda: IntegerPartition(100, list(range(1, 3))))
    a = IntegerPartition(8, [1, 3, 4])
    b = a.next_lex()
    c = IntegerPartition([1, 3, 4])
    d = IntegerPartition(8, {1: 3, 3: 1, 2: 1})
    assert a == c
    assert a.integer == d.integer
    assert a.conjugate == [3, 2, 2, 1]
    assert (a == b) is False
    assert a <= b
    assert (a > b) is False
    assert a != b

    for i in range(1, 11):
        next = set()
        prev = set()
        a = IntegerPartition([i])
        ans = {IntegerPartition(p) for p in partitions(i)}
        n = len(ans)
        for j in range(n):
            next.add(a)
            a = a.next_lex()
            IntegerPartition(i, a.partition)  # check it by giving i
        for j in range(n):
            prev.add(a)
            a = a.prev_lex()
            IntegerPartition(i, a.partition)  # check it by giving i
        assert next == ans
        assert prev == ans

    assert IntegerPartition([1, 2, 3]).as_ferrers() == '###\n##\n#'
    assert IntegerPartition([1, 1, 3]).as_ferrers('o') == 'ooo\no\no'
    assert str(IntegerPartition([1, 1, 3])) == '[3, 1, 1]'
    assert IntegerPartition([1, 1, 3]).partition == [3, 1, 1]

    raises(ValueError, lambda: random_integer_partition(-1))
    assert random_integer_partition(1) == [1]
    assert random_integer_partition(10, seed=[1, 3, 2, 1, 5, 1]
            ) == [5, 2, 1, 1, 1]


def test_rgs():
    raises(ValueError, lambda: RGS_unrank(-1, 3))
    raises(ValueError, lambda: RGS_unrank(3, 0))
    raises(ValueError, lambda: RGS_unrank(10, 1))

    raises(ValueError, lambda: Partition.from_rgs(list(range(3)), list(range(2))))
    raises(ValueError, lambda: Partition.from_rgs(list(range(1, 3)), list(range(2))))
    assert RGS_enum(-1) == 0
    assert RGS_enum(1) == 1
    assert RGS_unrank(7, 5) == [0, 0, 1, 0, 2]
    assert RGS_unrank(23, 14) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2]
    assert RGS_rank(RGS_unrank(40, 100)) == 40

def test_ordered_partition_9608():
    a = Partition([1, 2, 3], [4])
    b = Partition([1, 2], [3, 4])
    assert list(ordered([a,b], Set._infimum_key))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_applyfunc.py ===
from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matmul import MatMul
from sympy.simplify.simplify import simplify


X = MatrixSymbol("X", 3, 3)
Y = MatrixSymbol("Y", 3, 3)

k = symbols("k")
Xk = MatrixSymbol("X", k, k)

Xd = X.as_explicit()

x, y, z, t = symbols("x y z t")


def test_applyfunc_matrix():
    x = Dummy('x')
    double = Lambda(x, x**2)

    expr = ElementwiseApplyFunction(double, Xd)
    assert isinstance(expr, ElementwiseApplyFunction)
    assert expr.doit() == Xd.applyfunc(lambda x: x**2)
    assert expr.shape == (3, 3)
    assert expr.func(*expr.args) == expr
    assert simplify(expr) == expr
    assert expr[0, 0] == double(Xd[0, 0])

    expr = ElementwiseApplyFunction(double, X)
    assert isinstance(expr, ElementwiseApplyFunction)
    assert isinstance(expr.doit(), ElementwiseApplyFunction)
    assert expr == X.applyfunc(double)
    assert expr.func(*expr.args) == expr

    expr = ElementwiseApplyFunction(exp, X*Y)
    assert expr.expr == X*Y
    assert expr.function.dummy_eq(Lambda(x, exp(x)))
    assert expr.dummy_eq((X*Y).applyfunc(exp))
    assert expr.func(*expr.args) == expr

    assert isinstance(X*expr, MatMul)
    assert (X*expr).shape == (3, 3)
    Z = MatrixSymbol("Z", 2, 3)
    assert (Z*expr).shape == (2, 3)

    expr = ElementwiseApplyFunction(exp, Z.T)*ElementwiseApplyFunction(exp, Z)
    assert expr.shape == (3, 3)
    expr = ElementwiseApplyFunction(exp, Z)*ElementwiseApplyFunction(exp, Z.T)
    assert expr.shape == (2, 2)

    M = Matrix([[x, y], [z, t]])
    expr = ElementwiseApplyFunction(sin, M)
    assert isinstance(expr, ElementwiseApplyFunction)
    assert expr.function.dummy_eq(Lambda(x, sin(x)))
    assert expr.expr == M
    assert expr.doit() == M.applyfunc(sin)
    assert expr.doit() == Matrix([[sin(x), sin(y)], [sin(z), sin(t)]])
    assert expr.func(*expr.args) == expr

    expr = ElementwiseApplyFunction(double, Xk)
    assert expr.doit() == expr
    assert expr.subs(k, 2).shape == (2, 2)
    assert (expr*expr).shape == (k, k)
    M = MatrixSymbol("M", k, t)
    expr2 = M.T*expr*M
    assert isinstance(expr2, MatMul)
    assert expr2.args[1] == expr
    assert expr2.shape == (t, t)
    expr3 = expr*M
    assert expr3.shape == (k, t)

    expr1 = ElementwiseApplyFunction(lambda x: x+1, Xk)
    expr2 = ElementwiseApplyFunction(lambda x: x, Xk)
    assert expr1 != expr2


def test_applyfunc_entry():

    af = X.applyfunc(sin)
    assert af[0, 0] == sin(X[0, 0])

    af = Xd.applyfunc(sin)
    assert af[0, 0] == sin(X[0, 0])


def test_applyfunc_as_explicit():

    af = X.applyfunc(sin)
    assert af.as_explicit() == Matrix([
        [sin(X[0, 0]), sin(X[0, 1]), sin(X[0, 2])],
        [sin(X[1, 0]), sin(X[1, 1]), sin(X[1, 2])],
        [sin(X[2, 0]), sin(X[2, 1]), sin(X[2, 2])],
    ])


def test_applyfunc_transpose():

    af = Xk.applyfunc(sin)
    assert af.T.dummy_eq(Xk.T.applyfunc(sin))


def test_applyfunc_shape_11_matrices():
    M = MatrixSymbol("M", 1, 1)

    double = Lambda(x, x*2)

    expr = M.applyfunc(sin)
    assert isinstance(expr, ElementwiseApplyFunction)

    expr = M.applyfunc(double)
    assert isinstance(expr, MatMul)
    assert expr == 2*M

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\tests\test_core.py ===
from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.basic import Basic
from sympy.strategies.core import (
    null_safe, exhaust, memoize, condition,
    chain, tryit, do_one, debug, switch, minimize)
from io import StringIO


def posdec(x: int) -> int:
    if x > 0:
        return x - 1
    return x


def inc(x: int) -> int:
    return x + 1


def dec(x: int) -> int:
    return x - 1


def test_null_safe():
    def rl(expr: int) -> int | None:
        if expr == 1:
            return 2
        return None

    safe_rl = null_safe(rl)
    assert rl(1) == safe_rl(1)
    assert rl(3) is None
    assert safe_rl(3) == 3


def test_exhaust():
    sink = exhaust(posdec)
    assert sink(5) == 0
    assert sink(10) == 0


def test_memoize():
    rl = memoize(posdec)
    assert rl(5) == posdec(5)
    assert rl(5) == posdec(5)
    assert rl(-2) == posdec(-2)


def test_condition():
    rl = condition(lambda x: x % 2 == 0, posdec)
    assert rl(5) == 5
    assert rl(4) == 3


def test_chain():
    rl = chain(posdec, posdec)
    assert rl(5) == 3
    assert rl(1) == 0


def test_tryit():
    def rl(expr: Basic) -> Basic:
        assert False

    safe_rl = tryit(rl, AssertionError)
    assert safe_rl(S(1)) == S(1)


def test_do_one():
    rl = do_one(posdec, posdec)
    assert rl(5) == 4

    def rl1(x: int) -> int:
        if x == 1:
            return 2
        return x

    def rl2(x: int) -> int:
        if x == 2:
            return 3
        return x

    rule = do_one(rl1, rl2)
    assert rule(1) == 2
    assert rule(rule(1)) == 3


def test_debug():
    file = StringIO()
    rl = debug(posdec, file)
    rl(5)
    log = file.getvalue()
    file.close()

    assert posdec.__name__ in log
    assert '5' in log
    assert '4' in log


def test_switch():
    def key(x: int) -> int:
        return x % 3

    rl = switch(key, {0: inc, 1: dec})
    assert rl(3) == 4
    assert rl(4) == 3
    assert rl(5) == 5


def test_minimize():
    def key(x: int) -> int:
        return -x

    rl = minimize(inc, dec)
    assert rl(4) == 3

    rl = minimize(inc, dec, objective=key)
    assert rl(4) == 5

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_duplicated.py ===
import re
import sys

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_duplicated_with_misspelled_column_name(subset):
    # GH 19730
    df = DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    msg = re.escape("Index(['a'], dtype=")

    with pytest.raises(KeyError, match=msg):
        df.duplicated(subset)


def test_duplicated_implemented_no_recursion():
    # gh-21524
    # Ensure duplicated isn't implemented using recursion that
    # can fail on wide frames
    df = DataFrame(np.random.default_rng(2).integers(0, 1000, (10, 1000)))
    rec_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        result = df.duplicated()
    finally:
        sys.setrecursionlimit(rec_limit)

    # Then duplicates produce the bool Series as a result and don't fail during
    # calculation. Actual values doesn't matter here, though usually it's all
    # False in this case
    assert isinstance(result, Series)
    assert result.dtype == np.bool_


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True])),
        ("last", Series([True, True, False, False, False])),
        (False, Series([True, True, True, False, True])),
    ],
)
def test_duplicated_keep(keep, expected):
    df = DataFrame({"A": [0, 1, 1, 2, 0], "B": ["a", "b", "b", "c", "a"]})

    result = df.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(reason="GH#21720; nan/None falsely considered equal")
@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, True, False, True])),
        ("last", Series([True, True, False, False, False])),
        (False, Series([True, True, True, False, True])),
    ],
)
def test_duplicated_nan_none(keep, expected):
    df = DataFrame({"C": [np.nan, 3, 3, None, np.nan], "x": 1}, dtype=object)

    result = df.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("subset", [None, ["A", "B"], "A"])
def test_duplicated_subset(subset, keep):
    df = DataFrame(
        {
            "A": [0, 1, 1, 2, 0],
            "B": ["a", "b", "b", "c", "a"],
            "C": [np.nan, 3, 3, None, np.nan],
        }
    )

    if subset is None:
        subset = list(df.columns)
    elif isinstance(subset, str):
        # need to have a DataFrame, not a Series
        # -> select columns with singleton list, not string
        subset = [subset]

    expected = df[subset].duplicated(keep=keep)
    result = df.duplicated(keep=keep, subset=subset)
    tm.assert_series_equal(result, expected)


def test_duplicated_on_empty_frame():
    # GH 25184

    df = DataFrame(columns=["a", "b"])
    dupes = df.duplicated("a")

    result = df[dupes]
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_frame_datetime64_duplicated():
    dates = date_range("2010-07-01", end="2010-08-05")

    tst = DataFrame({"symbol": "AAA", "date": dates})
    result = tst.duplicated(["date", "symbol"])
    assert (-result).all()

    tst = DataFrame({"date": dates})
    result = tst.date.duplicated()
    assert (-result).all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_dropna.py ===
import numpy as np
import pytest

from pandas import (
    DatetimeIndex,
    IntervalIndex,
    NaT,
    Period,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestDropna:
    def test_dropna_empty(self):
        ser = Series([], dtype=object)

        assert len(ser.dropna()) == 0
        return_value = ser.dropna(inplace=True)
        assert return_value is None
        assert len(ser) == 0

        # invalid axis
        msg = "No axis named 1 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser.dropna(axis=1)

    def test_dropna_preserve_name(self, datetime_series):
        datetime_series[:5] = np.nan
        result = datetime_series.dropna()
        assert result.name == datetime_series.name
        name = datetime_series.name
        ts = datetime_series.copy()
        return_value = ts.dropna(inplace=True)
        assert return_value is None
        assert ts.name == name

    def test_dropna_no_nan(self):
        for ser in [
            Series([1, 2, 3], name="x"),
            Series([False, True, False], name="x"),
        ]:
            result = ser.dropna()
            tm.assert_series_equal(result, ser)
            assert result is not ser

            s2 = ser.copy()
            return_value = s2.dropna(inplace=True)
            assert return_value is None
            tm.assert_series_equal(s2, ser)

    def test_dropna_intervals(self):
        ser = Series(
            [np.nan, 1, 2, 3],
            IntervalIndex.from_arrays([np.nan, 0, 1, 2], [np.nan, 1, 2, 3]),
        )

        result = ser.dropna()
        expected = ser.iloc[1:]
        tm.assert_series_equal(result, expected)

    def test_dropna_period_dtype(self):
        # GH#13737
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])
        result = ser.dropna()
        expected = Series([Period("2011-01", freq="M")])

        tm.assert_series_equal(result, expected)

    def test_datetime64_tz_dropna(self, unit):
        # DatetimeLikeBlock
        ser = Series(
            [
                Timestamp("2011-01-01 10:00"),
                NaT,
                Timestamp("2011-01-03 10:00"),
                NaT,
            ],
            dtype=f"M8[{unit}]",
        )
        result = ser.dropna()
        expected = Series(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-03 10:00")],
            index=[0, 2],
            dtype=f"M8[{unit}]",
        )
        tm.assert_series_equal(result, expected)

        # DatetimeTZBlock
        idx = DatetimeIndex(
            ["2011-01-01 10:00", NaT, "2011-01-03 10:00", NaT], tz="Asia/Tokyo"
        ).as_unit(unit)
        ser = Series(idx)
        assert ser.dtype == f"datetime64[{unit}, Asia/Tokyo]"
        result = ser.dropna()
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-03 10:00", tz="Asia/Tokyo"),
            ],
            index=[0, 2],
            dtype=f"datetime64[{unit}, Asia/Tokyo]",
        )
        assert result.dtype == f"datetime64[{unit}, Asia/Tokyo]"
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [1, 1.5])
    def test_dropna_ignore_index(self, val):
        # GH#31725
        ser = Series([1, 2, val], index=[3, 2, 1])
        result = ser.dropna(ignore_index=True)
        expected = Series([1, 2, val])
        tm.assert_series_equal(result, expected)

        ser.dropna(ignore_index=True, inplace=True)
        tm.assert_series_equal(ser, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_models.py ===
import sympy.physics.mechanics.models as models
from sympy import (cos, sin, Matrix, symbols, zeros)
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols)


def test_multi_mass_spring_damper_inputs():

    c0, k0, m0 = symbols("c0 k0 m0")
    g = symbols("g")
    v0, x0, f0 = dynamicsymbols("v0 x0 f0")

    kane1 = models.multi_mass_spring_damper(1)
    massmatrix1 = Matrix([[m0]])
    forcing1 = Matrix([[-c0*v0 - k0*x0]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == Matrix([0])
    assert simplify(forcing1 - kane1.forcing) == Matrix([0])

    kane2 = models.multi_mass_spring_damper(1, True)
    massmatrix2 = Matrix([[m0]])
    forcing2 = Matrix([[-c0*v0 + g*m0 - k0*x0]])
    assert simplify(massmatrix2 - kane2.mass_matrix) == Matrix([0])
    assert simplify(forcing2 - kane2.forcing) == Matrix([0])

    kane3 = models.multi_mass_spring_damper(1, True, True)
    massmatrix3 = Matrix([[m0]])
    forcing3 = Matrix([[-c0*v0 + g*m0 - k0*x0 + f0]])
    assert simplify(massmatrix3 - kane3.mass_matrix) == Matrix([0])
    assert simplify(forcing3 - kane3.forcing) == Matrix([0])

    kane4 = models.multi_mass_spring_damper(1, False, True)
    massmatrix4 = Matrix([[m0]])
    forcing4 = Matrix([[-c0*v0 - k0*x0 + f0]])
    assert simplify(massmatrix4 - kane4.mass_matrix) == Matrix([0])
    assert simplify(forcing4 - kane4.forcing) == Matrix([0])


def test_multi_mass_spring_damper_higher_order():
    c0, k0, m0 = symbols("c0 k0 m0")
    c1, k1, m1 = symbols("c1 k1 m1")
    c2, k2, m2 = symbols("c2 k2 m2")
    v0, x0 = dynamicsymbols("v0 x0")
    v1, x1 = dynamicsymbols("v1 x1")
    v2, x2 = dynamicsymbols("v2 x2")

    kane1 = models.multi_mass_spring_damper(3)
    massmatrix1 = Matrix([[m0 + m1 + m2, m1 + m2, m2],
                          [m1 + m2, m1 + m2, m2],
                          [m2, m2, m2]])
    forcing1 = Matrix([[-c0*v0 - k0*x0],
                       [-c1*v1 - k1*x1],
                       [-c2*v2 - k2*x2]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])


def test_n_link_pendulum_on_cart_inputs():
    l0, m0 = symbols("l0 m0")
    m1 = symbols("m1")
    g = symbols("g")
    q0, q1, F, T1 = dynamicsymbols("q0 q1 F T1")
    u0, u1 = dynamicsymbols("u0 u1")

    kane1 = models.n_link_pendulum_on_cart(1)
    massmatrix1 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing1 = Matrix([[-l0*m1*u1**2*sin(q1) + F], [g*l0*m1*sin(q1)]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(2)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0])

    kane2 = models.n_link_pendulum_on_cart(1, False)
    massmatrix2 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing2 = Matrix([[-l0*m1*u1**2*sin(q1)], [g*l0*m1*sin(q1)]])
    assert simplify(massmatrix2 - kane2.mass_matrix) == zeros(2)
    assert simplify(forcing2 - kane2.forcing) == Matrix([0, 0])

    kane3 = models.n_link_pendulum_on_cart(1, False, True)
    massmatrix3 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing3 = Matrix([[-l0*m1*u1**2*sin(q1)], [g*l0*m1*sin(q1) + T1]])
    assert simplify(massmatrix3 - kane3.mass_matrix) == zeros(2)
    assert simplify(forcing3 - kane3.forcing) == Matrix([0, 0])

    kane4 = models.n_link_pendulum_on_cart(1, True, False)
    massmatrix4 = Matrix([[m0 + m1, -l0*m1*cos(q1)],
                          [-l0*m1*cos(q1), l0**2*m1]])
    forcing4 = Matrix([[-l0*m1*u1**2*sin(q1) + F], [g*l0*m1*sin(q1)]])
    assert simplify(massmatrix4 - kane4.mass_matrix) == zeros(2)
    assert simplify(forcing4 - kane4.forcing) == Matrix([0, 0])


def test_n_link_pendulum_on_cart_higher_order():
    l0, m0 = symbols("l0 m0")
    l1, m1 = symbols("l1 m1")
    m2 = symbols("m2")
    g = symbols("g")
    q0, q1, q2 = dynamicsymbols("q0 q1 q2")
    u0, u1, u2 = dynamicsymbols("u0 u1 u2")
    F, T1 = dynamicsymbols("F T1")

    kane1 = models.n_link_pendulum_on_cart(2)
    massmatrix1 = Matrix([[m0 + m1 + m2, -l0*m1*cos(q1) - l0*m2*cos(q1),
                           -l1*m2*cos(q2)],
                          [-l0*m1*cos(q1) - l0*m2*cos(q1), l0**2*m1 + l0**2*m2,
                           l0*l1*m2*(sin(q1)*sin(q2) + cos(q1)*cos(q2))],
                          [-l1*m2*cos(q2),
                           l0*l1*m2*(sin(q1)*sin(q2) + cos(q1)*cos(q2)),
                           l1**2*m2]])
    forcing1 = Matrix([[-l0*m1*u1**2*sin(q1) - l0*m2*u1**2*sin(q1) -
                        l1*m2*u2**2*sin(q2) + F],
                       [g*l0*m1*sin(q1) + g*l0*m2*sin(q1) -
                        l0*l1*m2*(sin(q1)*cos(q2) - sin(q2)*cos(q1))*u2**2],
                       [g*l1*m2*sin(q2) - l0*l1*m2*(-sin(q1)*cos(q2) +
                                                    sin(q2)*cos(q1))*u1**2]])
    assert simplify(massmatrix1 - kane1.mass_matrix) == zeros(3)
    assert simplify(forcing1 - kane1.forcing) == Matrix([0, 0, 0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\branch\tests\test_core.py ===
from sympy.strategies.branch.core import (
    exhaust, debug, multiplex, condition, notempty, chain, onaction, sfilter,
    yieldify, do_one, identity)


def posdec(x):
    if x > 0:
        yield x - 1
    else:
        yield x


def branch5(x):
    if 0 < x < 5:
        yield x - 1
    elif 5 < x < 10:
        yield x + 1
    elif x == 5:
        yield x + 1
        yield x - 1
    else:
        yield x


def even(x):
    return x % 2 == 0


def inc(x):
    yield x + 1


def one_to_n(n):
    yield from range(n)


def test_exhaust():
    brl = exhaust(branch5)
    assert set(brl(3)) == {0}
    assert set(brl(7)) == {10}
    assert set(brl(5)) == {0, 10}


def test_debug():
    from io import StringIO
    file = StringIO()
    rl = debug(posdec, file)
    list(rl(5))
    log = file.getvalue()
    file.close()

    assert posdec.__name__ in log
    assert '5' in log
    assert '4' in log


def test_multiplex():
    brl = multiplex(posdec, branch5)
    assert set(brl(3)) == {2}
    assert set(brl(7)) == {6, 8}
    assert set(brl(5)) == {4, 6}


def test_condition():
    brl = condition(even, branch5)
    assert set(brl(4)) == set(branch5(4))
    assert set(brl(5)) == set()


def test_sfilter():
    brl = sfilter(even, one_to_n)
    assert set(brl(10)) == {0, 2, 4, 6, 8}


def test_notempty():
    def ident_if_even(x):
        if even(x):
            yield x

    brl = notempty(ident_if_even)
    assert set(brl(4)) == {4}
    assert set(brl(5)) == {5}


def test_chain():
    assert list(chain()(2)) == [2]  # identity
    assert list(chain(inc, inc)(2)) == [4]
    assert list(chain(branch5, inc)(4)) == [4]
    assert set(chain(branch5, inc)(5)) == {5, 7}
    assert list(chain(inc, branch5)(5)) == [7]


def test_onaction():
    L = []

    def record(fn, input, output):
        L.append((input, output))

    list(onaction(inc, record)(2))
    assert L == [(2, 3)]

    list(onaction(identity, record)(2))
    assert L == [(2, 3)]


def test_yieldify():
    yinc = yieldify(lambda x: x + 1)
    assert list(yinc(3)) == [4]


def test_do_one():
    def bad(expr):
        raise ValueError

    assert list(do_one(inc)(3)) == [4]
    assert list(do_one(inc, bad)(3)) == [4]
    assert list(do_one(inc, posdec)(3)) == [4]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_trace.py ===
from sympy.core import Lambda, S, symbols
from sympy.concrete import Sum
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices import eye, Matrix, ShapeError, ImmutableMatrix
from sympy.matrices.expressions import (
    Adjoint, Identity, FunctionMatrix, MatrixExpr, MatrixSymbol, Trace,
    ZeroMatrix, trace, MatPow, MatAdd, MatMul
)
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.abc import i


n = symbols('n', integer=True)
A = MatrixSymbol('A', n, n)
B = MatrixSymbol('B', n, n)
C = MatrixSymbol('C', 3, 4)


def test_Trace():
    assert isinstance(Trace(A), Trace)
    assert not isinstance(Trace(A), MatrixExpr)
    raises(ShapeError, lambda: Trace(C))
    assert trace(eye(3)) == 3
    assert trace(Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])) == 15

    assert adjoint(Trace(A)) == trace(Adjoint(A))
    assert conjugate(Trace(A)) == trace(Adjoint(A))
    assert transpose(Trace(A)) == Trace(A)

    _ = A / Trace(A)  # Make sure this is possible

    # Some easy simplifications
    assert trace(Identity(5)) == 5
    assert trace(ZeroMatrix(5, 5)) == 0
    assert trace(OneMatrix(1, 1)) == 1
    assert trace(OneMatrix(2, 2)) == 2
    assert trace(OneMatrix(n, n)) == n
    assert trace(2*A*B) == 2*Trace(A*B)
    assert trace(A.T) == trace(A)

    i, j = symbols('i j')
    F = FunctionMatrix(3, 3, Lambda((i, j), i + j))
    assert trace(F) == (0 + 0) + (1 + 1) + (2 + 2)

    raises(TypeError, lambda: Trace(S.One))

    assert Trace(A).arg is A

    assert str(trace(A)) == str(Trace(A).doit())

    assert Trace(A).is_commutative is True

def test_Trace_A_plus_B():
    assert trace(A + B) == Trace(A) + Trace(B)
    assert Trace(A + B).arg == MatAdd(A, B)
    assert Trace(A + B).doit() == Trace(A) + Trace(B)


def test_Trace_MatAdd_doit():
    # See issue #9028
    X = ImmutableMatrix([[1, 2, 3]]*3)
    Y = MatrixSymbol('Y', 3, 3)
    q = MatAdd(X, 2*X, Y, -3*Y)
    assert Trace(q).arg == q
    assert Trace(q).doit() == 18 - 2*Trace(Y)


def test_Trace_MatPow_doit():
    X = Matrix([[1, 2], [3, 4]])
    assert Trace(X).doit() == 5
    q = MatPow(X, 2)
    assert Trace(q).arg == q
    assert Trace(q).doit() == 29


def test_Trace_MutableMatrix_plus():
    # See issue #9043
    X = Matrix([[1, 2], [3, 4]])
    assert Trace(X) + Trace(X) == 2*Trace(X)


def test_Trace_doit_deep_False():
    X = Matrix([[1, 2], [3, 4]])
    q = MatPow(X, 2)
    assert Trace(q).doit(deep=False).arg == q
    q = MatAdd(X, 2*X)
    assert Trace(q).doit(deep=False).arg == q
    q = MatMul(X, 2*X)
    assert Trace(q).doit(deep=False).arg == q


def test_trace_constant_factor():
    # Issue 9052: gave 2*Trace(MatMul(A)) instead of 2*Trace(A)
    assert trace(2*A) == 2*Trace(A)
    X = ImmutableMatrix([[1, 2], [3, 4]])
    assert trace(MatMul(2, X)) == 10


def test_trace_rewrite():
    assert trace(A).rewrite(Sum) == Sum(A[i, i], (i, 0, n - 1))
    assert trace(eye(3)).rewrite(Sum) == 3


def test_trace_normalize():
    assert Trace(B*A) != Trace(A*B)
    assert Trace(B*A)._normalize() == Trace(A*B)
    assert Trace(B*A.T)._normalize() == Trace(A*B.T)


def test_trace_as_explicit():
    raises(ValueError, lambda: Trace(A).as_explicit())

    X = MatrixSymbol("X", 3, 3)
    assert Trace(X).as_explicit() == X[0, 0] + X[1, 1] + X[2, 2]
    assert Trace(eye(3)).as_explicit() == 3

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_conventions.py ===
# -*- coding: utf-8 -*-

from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.functions.special.bessel import besselj
from sympy.functions.special.polynomials import legendre
from sympy.functions.combinatorial.numbers import bell
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.testing.pytest import XFAIL

def test_super_sub():
    assert split_super_sub("beta_13_2") == ("beta", [], ["13", "2"])
    assert split_super_sub("beta_132_20") == ("beta", [], ["132", "20"])
    assert split_super_sub("beta_13") == ("beta", [], ["13"])
    assert split_super_sub("x_a_b") == ("x", [], ["a", "b"])
    assert split_super_sub("x_1_2_3") == ("x", [], ["1", "2", "3"])
    assert split_super_sub("x_a_b1") == ("x", [], ["a", "b1"])
    assert split_super_sub("x_a_1") == ("x", [], ["a", "1"])
    assert split_super_sub("x_1_a") == ("x", [], ["1", "a"])
    assert split_super_sub("x_1^aa") == ("x", ["aa"], ["1"])
    assert split_super_sub("x_1__aa") == ("x", ["aa"], ["1"])
    assert split_super_sub("x_11^a") == ("x", ["a"], ["11"])
    assert split_super_sub("x_11__a") == ("x", ["a"], ["11"])
    assert split_super_sub("x_a_b_c_d") == ("x", [], ["a", "b", "c", "d"])
    assert split_super_sub("x_a_b^c^d") == ("x", ["c", "d"], ["a", "b"])
    assert split_super_sub("x_a_b__c__d") == ("x", ["c", "d"], ["a", "b"])
    assert split_super_sub("x_a^b_c^d") == ("x", ["b", "d"], ["a", "c"])
    assert split_super_sub("x_a__b_c__d") == ("x", ["b", "d"], ["a", "c"])
    assert split_super_sub("x^a^b_c_d") == ("x", ["a", "b"], ["c", "d"])
    assert split_super_sub("x__a__b_c_d") == ("x", ["a", "b"], ["c", "d"])
    assert split_super_sub("x^a^b^c^d") == ("x", ["a", "b", "c", "d"], [])
    assert split_super_sub("x__a__b__c__d") == ("x", ["a", "b", "c", "d"], [])
    assert split_super_sub("alpha_11") == ("alpha", [], ["11"])
    assert split_super_sub("alpha_11_11") == ("alpha", [], ["11", "11"])
    assert split_super_sub("w1") == ("w", [], ["1"])
    assert split_super_sub("w𝟙") == ("w", [], ["𝟙"])
    assert split_super_sub("w11") == ("w", [], ["11"])
    assert split_super_sub("w𝟙𝟙") == ("w", [], ["𝟙𝟙"])
    assert split_super_sub("w𝟙2𝟙") == ("w", [], ["𝟙2𝟙"])
    assert split_super_sub("w1^a") == ("w", ["a"], ["1"])
    assert split_super_sub("ω1") == ("ω", [], ["1"])
    assert split_super_sub("ω11") == ("ω", [], ["11"])
    assert split_super_sub("ω1^a") == ("ω", ["a"], ["1"])
    assert split_super_sub("ω𝟙^α") == ("ω", ["α"], ["𝟙"])
    assert split_super_sub("ω𝟙2^3α") == ("ω", ["3α"], ["𝟙2"])
    assert split_super_sub("") == ("", [], [])


def test_requires_partial():
    x, y, z, t, nu = symbols('x y z t nu')
    n = symbols('n', integer=True)

    f = x * y
    assert requires_partial(Derivative(f, x)) is True
    assert requires_partial(Derivative(f, y)) is True

    ## integrating out one of the variables
    assert requires_partial(Derivative(Integral(exp(-x * y), (x, 0, oo)), y, evaluate=False)) is False

    ## bessel function with smooth parameter
    f = besselj(nu, x)
    assert requires_partial(Derivative(f, x)) is True
    assert requires_partial(Derivative(f, nu)) is True

    ## bessel function with integer parameter
    f = besselj(n, x)
    assert requires_partial(Derivative(f, x)) is False
    # this is not really valid (differentiating with respect to an integer)
    # but there's no reason to use the partial derivative symbol there. make
    # sure we don't throw an exception here, though
    assert requires_partial(Derivative(f, n)) is False

    ## bell polynomial
    f = bell(n, x)
    assert requires_partial(Derivative(f, x)) is False
    # again, invalid
    assert requires_partial(Derivative(f, n)) is False

    ## legendre polynomial
    f = legendre(0, x)
    assert requires_partial(Derivative(f, x)) is False

    f = legendre(n, x)
    assert requires_partial(Derivative(f, x)) is False
    # again, invalid
    assert requires_partial(Derivative(f, n)) is False

    f = x ** n
    assert requires_partial(Derivative(f, x)) is False

    assert requires_partial(Derivative(Integral((x*y) ** n * exp(-x * y), (x, 0, oo)), y, evaluate=False)) is False

    # parametric equation
    f = (exp(t), cos(t))
    g = sum(f)
    assert requires_partial(Derivative(g, t)) is False

    f = symbols('f', cls=Function)
    assert requires_partial(Derivative(f(x), x)) is False
    assert requires_partial(Derivative(f(x), y)) is False
    assert requires_partial(Derivative(f(x, y), x)) is True
    assert requires_partial(Derivative(f(x, y), y)) is True
    assert requires_partial(Derivative(f(x, y), z)) is True
    assert requires_partial(Derivative(f(x, y), x, y)) is True

@XFAIL
def test_requires_partial_unspecified_variables():
    x, y = symbols('x y')
    # function of unspecified variables
    f = symbols('f', cls=Function)
    assert requires_partial(Derivative(f, x)) is False
    assert requires_partial(Derivative(f, x, y)) is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\tests\test_extension_interface.py ===
from __future__ import print_function
from __future__ import absolute_import

import sys

import greenlet
from . import _test_extension
from . import TestCase

# pylint:disable=c-extension-no-member

class CAPITests(TestCase):
    def test_switch(self):
        self.assertEqual(
            50, _test_extension.test_switch(greenlet.greenlet(lambda: 50)))

    def test_switch_kwargs(self):
        def adder(x, y):
            return x * y
        g = greenlet.greenlet(adder)
        self.assertEqual(6, _test_extension.test_switch_kwargs(g, x=3, y=2))

    def test_setparent(self):
        # pylint:disable=disallowed-name
        def foo():
            def bar():
                greenlet.getcurrent().parent.switch()

                # This final switch should go back to the main greenlet, since
                # the test_setparent() function in the C extension should have
                # reparented this greenlet.
                greenlet.getcurrent().parent.switch()
                raise AssertionError("Should never have reached this code")
            child = greenlet.greenlet(bar)
            child.switch()
            greenlet.getcurrent().parent.switch(child)
            greenlet.getcurrent().parent.throw(
                AssertionError("Should never reach this code"))
        foo_child = greenlet.greenlet(foo).switch()
        self.assertEqual(None, _test_extension.test_setparent(foo_child))

    def test_getcurrent(self):
        _test_extension.test_getcurrent()

    def test_new_greenlet(self):
        self.assertEqual(-15, _test_extension.test_new_greenlet(lambda: -15))

    def test_raise_greenlet_dead(self):
        self.assertRaises(
            greenlet.GreenletExit, _test_extension.test_raise_dead_greenlet)

    def test_raise_greenlet_error(self):
        self.assertRaises(
            greenlet.error, _test_extension.test_raise_greenlet_error)

    def test_throw(self):
        seen = []

        def foo():         # pylint:disable=disallowed-name
            try:
                greenlet.getcurrent().parent.switch()
            except ValueError:
                seen.append(sys.exc_info()[1])
            except greenlet.GreenletExit:
                raise AssertionError
        g = greenlet.greenlet(foo)
        g.switch()
        _test_extension.test_throw(g)
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            isinstance(seen[0], ValueError),
            "ValueError was not raised in foo()")
        self.assertEqual(
            str(seen[0]),
            'take that sucka!',
            "message doesn't match")

    def test_non_traceback_param(self):
        with self.assertRaises(TypeError) as exc:
            _test_extension.test_throw_exact(
                greenlet.getcurrent(),
                Exception,
                Exception(),
                self
            )
        self.assertEqual(str(exc.exception),
                         "throw() third argument must be a traceback object")

    def test_instance_of_wrong_type(self):
        with self.assertRaises(TypeError) as exc:
            _test_extension.test_throw_exact(
                greenlet.getcurrent(),
                Exception(),
                BaseException(),
                None,
            )

        self.assertEqual(str(exc.exception),
                         "instance exception may not have a separate value")

    def test_not_throwable(self):
        with self.assertRaises(TypeError) as exc:
            _test_extension.test_throw_exact(
                greenlet.getcurrent(),
                "abc",
                None,
                None,
            )
        self.assertEqual(str(exc.exception),
                         "exceptions must be classes, or instances, not str")


if __name__ == '__main__':
    import unittest
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_nlargest_nsmallest.py ===
import numpy as np
import pytest

from pandas import (
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm


def test_nlargest():
    a = Series([1, 3, 5, 7, 2, 9, 0, 4, 6, 10])
    b = Series(list("a" * 5 + "b" * 5))
    gb = a.groupby(b)
    r = gb.nlargest(3)
    e = Series(
        [7, 5, 3, 10, 9, 6],
        index=MultiIndex.from_arrays([list("aaabbb"), [3, 2, 1, 9, 5, 8]]),
    )
    tm.assert_series_equal(r, e)

    a = Series([1, 1, 3, 2, 0, 3, 3, 2, 1, 0])
    gb = a.groupby(b)
    e = Series(
        [3, 2, 1, 3, 3, 2],
        index=MultiIndex.from_arrays([list("aaabbb"), [2, 3, 1, 6, 5, 7]]),
    )
    tm.assert_series_equal(gb.nlargest(3, keep="last"), e)


def test_nlargest_mi_grouper():
    # see gh-21411
    npr = np.random.default_rng(2)

    dts = date_range("20180101", periods=10)
    iterables = [dts, ["one", "two"]]

    idx = MultiIndex.from_product(iterables, names=["first", "second"])
    s = Series(npr.standard_normal(20), index=idx)

    result = s.groupby("first").nlargest(1)

    exp_idx = MultiIndex.from_tuples(
        [
            (dts[0], dts[0], "one"),
            (dts[1], dts[1], "one"),
            (dts[2], dts[2], "one"),
            (dts[3], dts[3], "two"),
            (dts[4], dts[4], "one"),
            (dts[5], dts[5], "one"),
            (dts[6], dts[6], "one"),
            (dts[7], dts[7], "one"),
            (dts[8], dts[8], "one"),
            (dts[9], dts[9], "one"),
        ],
        names=["first", "first", "second"],
    )

    exp_values = [
        0.18905338179353307,
        -0.41306354339189344,
        1.799707382720902,
        0.7738065867276614,
        0.28121066979764925,
        0.9775674511260357,
        -0.3288239040579627,
        0.45495807124085547,
        0.5452887139646817,
        0.12682784711186987,
    ]

    expected = Series(exp_values, index=exp_idx)
    tm.assert_series_equal(result, expected, check_exact=False, rtol=1e-3)


def test_nsmallest():
    a = Series([1, 3, 5, 7, 2, 9, 0, 4, 6, 10])
    b = Series(list("a" * 5 + "b" * 5))
    gb = a.groupby(b)
    r = gb.nsmallest(3)
    e = Series(
        [1, 2, 3, 0, 4, 6],
        index=MultiIndex.from_arrays([list("aaabbb"), [0, 4, 1, 6, 7, 8]]),
    )
    tm.assert_series_equal(r, e)

    a = Series([1, 1, 3, 2, 0, 3, 3, 2, 1, 0])
    gb = a.groupby(b)
    e = Series(
        [0, 1, 1, 0, 1, 2],
        index=MultiIndex.from_arrays([list("aaabbb"), [4, 1, 0, 9, 8, 7]]),
    )
    tm.assert_series_equal(gb.nsmallest(3, keep="last"), e)


@pytest.mark.parametrize(
    "data, groups",
    [([0, 1, 2, 3], [0, 0, 1, 1]), ([0], [0])],
)
@pytest.mark.parametrize("dtype", [None, *tm.ALL_INT_NUMPY_DTYPES])
@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_nlargest_and_smallest_noop(data, groups, dtype, method):
    # GH 15272, GH 16345, GH 29129
    # Test nlargest/smallest when it results in a noop,
    # i.e. input is sorted and group size <= n
    if dtype is not None:
        data = np.array(data, dtype=dtype)
    if method == "nlargest":
        data = list(reversed(data))
    ser = Series(data, name="a")
    result = getattr(ser.groupby(groups), method)(n=2)
    expidx = np.array(groups, dtype=int) if isinstance(groups, list) else groups
    expected = Series(data, index=MultiIndex.from_arrays([expidx, ser.index]), name="a")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_kane4.py ===
from sympy import (cos, sin, Matrix, symbols)
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
                                        KanesMethod, Particle)

def test_replace_qdots_in_force():
    # Test PR 16700 "Replaces qdots with us in force-list in kanes.py"
    # The new functionality allows one to specify forces in qdots which will
    # automatically be replaced with u:s which are defined by the kde supplied
    # to KanesMethod. The test case is the double pendulum with interacting
    # forces in the example of chapter 4.7 "CONTRIBUTING INTERACTION FORCES"
    # in Ref. [1]. Reference list at end test function.

    q1, q2 = dynamicsymbols('q1, q2')
    qd1, qd2 = dynamicsymbols('q1, q2', level=1)
    u1, u2 = dynamicsymbols('u1, u2')

    l, m = symbols('l, m')

    N = ReferenceFrame('N') # Inertial frame
    A = N.orientnew('A', 'Axis', (q1, N.z)) # Rod A frame
    B = A.orientnew('B', 'Axis', (q2, N.z)) # Rod B frame

    O = Point('O') # Origo
    O.set_vel(N, 0)

    P = O.locatenew('P', ( l * A.x )) # Point @ end of rod A
    P.v2pt_theory(O, N, A)

    Q = P.locatenew('Q', ( l * B.x )) # Point @ end of rod B
    Q.v2pt_theory(P, N, B)

    Ap = Particle('Ap', P, m)
    Bp = Particle('Bp', Q, m)

    # The forces are specified below. sigma is the torsional spring stiffness
    # and delta is the viscous damping coefficient acting between the two
    # bodies. Here, we specify the viscous damper as function of qdots prior
    # forming the kde. In more complex systems it not might be obvious which
    # kde is most efficient, why it is convenient to specify viscous forces in
    # qdots independently of the kde.
    sig, delta = symbols('sigma, delta')
    Ta = (sig * q2 + delta * qd2) * N.z
    forces = [(A, Ta), (B, -Ta)]

    # Try different kdes.
    kde1 = [u1 - qd1, u2 - qd2]
    kde2 = [u1 - qd1, u2 - (qd1 + qd2)]

    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    fr1, fstar1 = KM1.kanes_equations([Ap, Bp], forces)

    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    fr2, fstar2 = KM2.kanes_equations([Ap, Bp], forces)

    # Check EOM for KM2:
    # Mass and force matrix from p.6 in Ref. [2] with added forces from
    # example of chapter 4.7 in [1] and without gravity.
    forcing_matrix_expected = Matrix( [ [ m * l**2 * sin(q2) * u2**2 + sig * q2
                                        + delta * (u2 - u1)],
                                        [ m * l**2 * sin(q2) * -u1**2 - sig * q2
                                        - delta * (u2 - u1)] ] )
    mass_matrix_expected = Matrix( [ [ 2 * m * l**2, m * l**2 * cos(q2) ],
                                    [ m * l**2 * cos(q2), m * l**2 ] ] )

    assert (KM2.mass_matrix.expand() == mass_matrix_expected.expand())
    assert (KM2.forcing.expand() == forcing_matrix_expected.expand())

    # Check fr1 with reference fr_expected from [1] with u:s instead of qdots.
    fr1_expected = Matrix([ 0, -(sig*q2 + delta * u2) ])
    assert fr1.expand() == fr1_expected.expand()

    # Check fr2
    fr2_expected = Matrix([sig * q2 + delta * (u2 - u1),
                            - sig * q2 - delta * (u2 - u1)])
    assert fr2.expand() == fr2_expected.expand()

    # Specifying forces in u:s should stay the same:
    Ta = (sig * q2 + delta * u2) * N.z
    forces = [(A, Ta), (B, -Ta)]
    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    fr1, fstar1 = KM1.kanes_equations([Ap, Bp], forces)

    assert fr1.expand() == fr1_expected.expand()

    Ta = (sig * q2 + delta * (u2-u1)) * N.z
    forces = [(A, Ta), (B, -Ta)]
    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    fr2, fstar2 = KM2.kanes_equations([Ap, Bp], forces)

    assert fr2.expand() == fr2_expected.expand()

    # Test if we have a qubic qdot force:
    Ta = (sig * q2 + delta * qd2**3) * N.z
    forces = [(A, Ta), (B, -Ta)]

    KM1 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde1)
    fr1, fstar1 = KM1.kanes_equations([Ap, Bp], forces)

    fr1_cubic_expected = Matrix([ 0, -(sig*q2 + delta * u2**3) ])

    assert fr1.expand() == fr1_cubic_expected.expand()

    KM2 = KanesMethod(N, [q1, q2], [u1, u2], kd_eqs=kde2)
    fr2, fstar2 = KM2.kanes_equations([Ap, Bp], forces)

    fr2_cubic_expected = Matrix([sig * q2 + delta * (u2 - u1)**3,
                            - sig * q2 - delta * (u2 - u1)**3])

    assert fr2.expand() == fr2_cubic_expected.expand()

    # References:
    # [1] T.R. Kane, D. a Levinson, Dynamics Theory and Applications, 2005.
    # [2] Arun K Banerjee, Flexible Multibody Dynamics:Efficient Formulations
    #     and Applications, John Wiley and Sons, Ltd, 2016.
    #     doi:http://dx.doi.org/10.1002/9781119015635.

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_homomorphisms.py ===
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.homomorphisms import homomorphism, group_isomorphism, is_isomorphic
from sympy.combinatorics.free_groups import free_group
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.named_groups import AlternatingGroup, DihedralGroup, CyclicGroup
from sympy.testing.pytest import raises

def test_homomorphism():
    # FpGroup -> PermutationGroup
    F, a, b = free_group("a, b")
    G = FpGroup(F, [a**3, b**3, (a*b)**2])

    c = Permutation(3)(0, 1, 2)
    d = Permutation(3)(1, 2, 3)
    A = AlternatingGroup(4)
    T = homomorphism(G, A, [a, b], [c, d])
    assert T(a*b**2*a**-1) == c*d**2*c**-1
    assert T.is_isomorphism()
    assert T(T.invert(Permutation(3)(0, 2, 3))) == Permutation(3)(0, 2, 3)

    T = homomorphism(G, AlternatingGroup(4), G.generators)
    assert T.is_trivial()
    assert T.kernel().order() == G.order()

    E, e = free_group("e")
    G = FpGroup(E, [e**8])
    P = PermutationGroup([Permutation(0, 1, 2, 3), Permutation(0, 2)])
    T = homomorphism(G, P, [e], [Permutation(0, 1, 2, 3)])
    assert T.image().order() == 4
    assert T(T.invert(Permutation(0, 2)(1, 3))) == Permutation(0, 2)(1, 3)

    T = homomorphism(E, AlternatingGroup(4), E.generators, [c])
    assert T.invert(c**2) == e**-1 #order(c) == 3 so c**2 == c**-1

    # FreeGroup -> FreeGroup
    T = homomorphism(F, E, [a], [e])
    assert T(a**-2*b**4*a**2).is_identity

    # FreeGroup -> FpGroup
    G = FpGroup(F, [a*b*a**-1*b**-1])
    T = homomorphism(F, G, F.generators, G.generators)
    assert T.invert(a**-1*b**-1*a**2) == a*b**-1

    # PermutationGroup -> PermutationGroup
    D = DihedralGroup(8)
    p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    P = PermutationGroup(p)
    T = homomorphism(P, D, [p], [p])
    assert T.is_injective()
    assert not T.is_isomorphism()
    assert T.invert(p**3) == p**3

    T2 = homomorphism(F, P, [F.generators[0]], P.generators)
    T = T.compose(T2)
    assert T.domain == F
    assert T.codomain == D
    assert T(a*b) == p

    D3 = DihedralGroup(3)
    T = homomorphism(D3, D3, D3.generators, D3.generators)
    assert T.is_isomorphism()


def test_isomorphisms():

    F, a, b = free_group("a, b")
    E, c, d = free_group("c, d")
    # Infinite groups with differently ordered relators.
    G = FpGroup(F, [a**2, b**3])
    H = FpGroup(F, [b**3, a**2])
    assert is_isomorphic(G, H)

    # Trivial Case
    # FpGroup -> FpGroup
    H = FpGroup(F, [a**3, b**3, (a*b)**2])
    F, c, d = free_group("c, d")
    G = FpGroup(F, [c**3, d**3, (c*d)**2])
    check, T =  group_isomorphism(G, H)
    assert check
    assert T(c**3*d**2) == a**3*b**2

    # FpGroup -> PermutationGroup
    # FpGroup is converted to the equivalent isomorphic group.
    F, a, b = free_group("a, b")
    G = FpGroup(F, [a**3, b**3, (a*b)**2])
    H = AlternatingGroup(4)
    check, T = group_isomorphism(G, H)
    assert check
    assert T(b*a*b**-1*a**-1*b**-1) == Permutation(0, 2, 3)
    assert T(b*a*b*a**-1*b**-1) == Permutation(0, 3, 2)

    # PermutationGroup -> PermutationGroup
    D = DihedralGroup(8)
    p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    P = PermutationGroup(p)
    assert not is_isomorphic(D, P)

    A = CyclicGroup(5)
    B = CyclicGroup(7)
    assert not is_isomorphic(A, B)

    # Two groups of the same prime order are isomorphic to each other.
    G = FpGroup(F, [a, b**5])
    H = CyclicGroup(5)
    assert G.order() == H.order()
    assert is_isomorphic(G, H)


def test_check_homomorphism():
    a = Permutation(1,2,3,4)
    b = Permutation(1,3)
    G = PermutationGroup([a, b])
    raises(ValueError, lambda: homomorphism(G, G, [a], [a]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_special.py ===
from mpmath import *

def test_special():
    assert inf == inf
    assert inf != -inf
    assert -inf == -inf
    assert inf != nan
    assert nan != nan
    assert isnan(nan)
    assert --inf == inf
    assert abs(inf) == inf
    assert abs(-inf) == inf
    assert abs(nan) != abs(nan)

    assert isnan(inf - inf)
    assert isnan(inf + (-inf))
    assert isnan(-inf - (-inf))

    assert isnan(inf + nan)
    assert isnan(-inf + nan)

    assert mpf(2) + inf == inf
    assert 2 + inf == inf
    assert mpf(2) - inf == -inf
    assert 2 - inf == -inf

    assert inf > 3
    assert 3 < inf
    assert 3 > -inf
    assert -inf < 3
    assert inf > mpf(3)
    assert mpf(3) < inf
    assert mpf(3) > -inf
    assert -inf < mpf(3)

    assert not (nan < 3)
    assert not (nan > 3)

    assert isnan(inf * 0)
    assert isnan(-inf * 0)
    assert inf * 3 == inf
    assert inf * -3 == -inf
    assert -inf * 3 == -inf
    assert -inf * -3 == inf
    assert inf * inf == inf
    assert -inf * -inf == inf

    assert isnan(nan / 3)
    assert inf / -3 == -inf
    assert inf / 3 == inf
    assert 3 / inf == 0
    assert -3 / inf == 0
    assert 0 / inf == 0
    assert isnan(inf / inf)
    assert isnan(inf / -inf)
    assert isnan(inf / nan)

    assert mpf('inf') == mpf('+inf') == inf
    assert mpf('-inf') == -inf
    assert isnan(mpf('nan'))

    assert isinf(inf)
    assert isinf(-inf)
    assert not isinf(mpf(0))
    assert not isinf(nan)

def test_special_powers():
    assert inf**3 == inf
    assert isnan(inf**0)
    assert inf**-3 == 0
    assert (-inf)**2 == inf
    assert (-inf)**3 == -inf
    assert isnan((-inf)**0)
    assert (-inf)**-2 == 0
    assert (-inf)**-3 == 0
    assert isnan(nan**5)
    assert isnan(nan**0)

def test_functions_special():
    assert exp(inf) == inf
    assert exp(-inf) == 0
    assert isnan(exp(nan))
    assert log(inf) == inf
    assert isnan(log(nan))
    assert isnan(sin(inf))
    assert isnan(sin(nan))
    assert atan(inf).ae(pi/2)
    assert atan(-inf).ae(-pi/2)
    assert isnan(sqrt(nan))
    assert sqrt(inf) == inf

def test_convert_special():
    float_inf = 1e300 * 1e300
    float_ninf = -float_inf
    float_nan = float_inf/float_ninf
    assert mpf(3) * float_inf == inf
    assert mpf(3) * float_ninf == -inf
    assert isnan(mpf(3) * float_nan)
    assert not (mpf(3) < float_nan)
    assert not (mpf(3) > float_nan)
    assert not (mpf(3) <= float_nan)
    assert not (mpf(3) >= float_nan)
    assert float(mpf('1e1000')) == float_inf
    assert float(mpf('-1e1000')) == float_ninf
    assert float(mpf('1e100000000000000000')) == float_inf
    assert float(mpf('-1e100000000000000000')) == float_ninf
    assert float(mpf('1e-100000000000000000')) == 0.0

def test_div_bug():
    assert isnan(nan/1)
    assert isnan(nan/2)
    assert inf/2 == inf
    assert (-inf)/2 == -inf

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_array_api_info.py ===
import pytest

import numpy as np

info = np.__array_namespace_info__()


def test_capabilities():
    caps = info.capabilities()
    assert caps["boolean indexing"] is True
    assert caps["data-dependent shapes"] is True

    # This will be added in the 2024.12 release of the array API standard.

    # assert caps["max rank"] == 64
    # np.zeros((1,)*64)
    # with pytest.raises(ValueError):
    #     np.zeros((1,)*65)


def test_default_device():
    assert info.default_device() == "cpu" == np.asarray(0).device


def test_default_dtypes():
    dtypes = info.default_dtypes()
    assert dtypes["real floating"] == np.float64 == np.asarray(0.0).dtype
    assert dtypes["complex floating"] == np.complex128 == \
        np.asarray(0.0j).dtype
    assert dtypes["integral"] == np.intp == np.asarray(0).dtype
    assert dtypes["indexing"] == np.intp == np.argmax(np.zeros(10)).dtype

    with pytest.raises(ValueError, match="Device not understood"):
        info.default_dtypes(device="gpu")


def test_dtypes_all():
    dtypes = info.dtypes()
    assert dtypes == {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }


dtype_categories = {
    "bool": {"bool": np.bool_},
    "signed integer": {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
    },
    "unsigned integer": {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    },
    "integral": ("signed integer", "unsigned integer"),
    "real floating": {"float32": np.float32, "float64": np.float64},
    "complex floating": {"complex64": np.complex64, "complex128":
                         np.complex128},
    "numeric": ("integral", "real floating", "complex floating"),
}


@pytest.mark.parametrize("kind", dtype_categories)
def test_dtypes_kind(kind):
    expected = dtype_categories[kind]
    if isinstance(expected, tuple):
        assert info.dtypes(kind=kind) == info.dtypes(kind=expected)
    else:
        assert info.dtypes(kind=kind) == expected


def test_dtypes_tuple():
    dtypes = info.dtypes(kind=("bool", "integral"))
    assert dtypes == {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }


def test_dtypes_invalid_kind():
    with pytest.raises(ValueError, match="unsupported kind"):
        info.dtypes(kind="invalid")


def test_dtypes_invalid_device():
    with pytest.raises(ValueError, match="Device not understood"):
        info.dtypes(device="gpu")


def test_devices():
    assert info.devices() == ["cpu"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_frame_apply_relabeling.py ===
import numpy as np
import pytest

from pandas.compat.numpy import np_version_gte1p25

import pandas as pd
import pandas._testing as tm


def test_agg_relabel():
    # GH 26513
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})

    # simplest case with one column, one func
    result = df.agg(foo=("B", "sum"))
    expected = pd.DataFrame({"B": [10]}, index=pd.Index(["foo"]))
    tm.assert_frame_equal(result, expected)

    # test on same column with different methods
    result = df.agg(foo=("B", "sum"), bar=("B", "min"))
    expected = pd.DataFrame({"B": [10, 1]}, index=pd.Index(["foo", "bar"]))

    tm.assert_frame_equal(result, expected)


def test_agg_relabel_multi_columns_multi_methods():
    # GH 26513, test on multiple columns with multiple methods
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})
    result = df.agg(
        foo=("A", "sum"),
        bar=("B", "mean"),
        cat=("A", "min"),
        dat=("B", "max"),
        f=("A", "max"),
        g=("C", "min"),
    )
    expected = pd.DataFrame(
        {
            "A": [6.0, np.nan, 1.0, np.nan, 2.0, np.nan],
            "B": [np.nan, 2.5, np.nan, 4.0, np.nan, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 3.0],
        },
        index=pd.Index(["foo", "bar", "cat", "dat", "f", "g"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(np_version_gte1p25, reason="name of min now equals name of np.min")
def test_agg_relabel_partial_functions():
    # GH 26513, test on partial, functools or more complex cases
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})
    msg = "using Series.[mean|min]"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.agg(foo=("A", np.mean), bar=("A", "mean"), cat=("A", min))
    expected = pd.DataFrame(
        {"A": [1.5, 1.5, 1.0]}, index=pd.Index(["foo", "bar", "cat"])
    )
    tm.assert_frame_equal(result, expected)

    msg = "using Series.[mean|min|max|sum]"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.agg(
            foo=("A", min),
            bar=("A", np.min),
            cat=("B", max),
            dat=("C", "min"),
            f=("B", np.sum),
            kk=("B", lambda x: min(x)),
        )
    expected = pd.DataFrame(
        {
            "A": [1.0, 1.0, np.nan, np.nan, np.nan, np.nan],
            "B": [np.nan, np.nan, 4.0, np.nan, 10.0, 1.0],
            "C": [np.nan, np.nan, np.nan, 3.0, np.nan, np.nan],
        },
        index=pd.Index(["foo", "bar", "cat", "dat", "f", "kk"]),
    )
    tm.assert_frame_equal(result, expected)


def test_agg_namedtuple():
    # GH 26513
    df = pd.DataFrame({"A": [0, 1], "B": [1, 2]})
    result = df.agg(
        foo=pd.NamedAgg("B", "sum"),
        bar=pd.NamedAgg("B", "min"),
        cat=pd.NamedAgg(column="B", aggfunc="count"),
        fft=pd.NamedAgg("B", aggfunc="max"),
    )

    expected = pd.DataFrame(
        {"B": [3, 1, 2, 2]}, index=pd.Index(["foo", "bar", "cat", "fft"])
    )
    tm.assert_frame_equal(result, expected)

    result = df.agg(
        foo=pd.NamedAgg("A", "min"),
        bar=pd.NamedAgg(column="B", aggfunc="max"),
        cat=pd.NamedAgg(column="A", aggfunc="max"),
    )
    expected = pd.DataFrame(
        {"A": [0.0, np.nan, 1.0], "B": [np.nan, 2.0, np.nan]},
        index=pd.Index(["foo", "bar", "cat"]),
    )
    tm.assert_frame_equal(result, expected)


def test_reconstruct_func():
    # GH 28472, test to ensure reconstruct_func isn't moved;
    # This method is used by other libraries (e.g. dask)
    result = pd.core.apply.reconstruct_func("min")
    expected = (False, "min", None, None)
    tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_frozen.py ===
import re

import pytest

from pandas.core.indexes.frozen import FrozenList


@pytest.fixture
def lst():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def container(lst):
    return FrozenList(lst)


@pytest.fixture
def unicode_container():
    return FrozenList(["\u05d0", "\u05d1", "c"])


class TestFrozenList:
    def check_mutable_error(self, *args, **kwargs):
        # Pass whatever function you normally would to pytest.raises
        # (after the Exception kind).
        mutable_regex = re.compile("does not support mutable operations")
        msg = "'(_s)?re.(SRE_)?Pattern' object is not callable"
        with pytest.raises(TypeError, match=msg):
            mutable_regex(*args, **kwargs)

    def test_no_mutable_funcs(self, container):
        def setitem():
            container[0] = 5

        self.check_mutable_error(setitem)

        def setslice():
            container[1:2] = 3

        self.check_mutable_error(setslice)

        def delitem():
            del container[0]

        self.check_mutable_error(delitem)

        def delslice():
            del container[0:3]

        self.check_mutable_error(delslice)

        mutable_methods = ("extend", "pop", "remove", "insert")

        for meth in mutable_methods:
            self.check_mutable_error(getattr(container, meth))

    def test_slicing_maintains_type(self, container, lst):
        result = container[1:2]
        expected = lst[1:2]
        self.check_result(result, expected)

    def check_result(self, result, expected):
        assert isinstance(result, FrozenList)
        assert result == expected

    def test_string_methods_dont_fail(self, container):
        repr(container)
        str(container)
        bytes(container)

    def test_tricky_container(self, unicode_container):
        repr(unicode_container)
        str(unicode_container)

    def test_add(self, container, lst):
        result = container + (1, 2, 3)
        expected = FrozenList(lst + [1, 2, 3])
        self.check_result(result, expected)

        result = (1, 2, 3) + container
        expected = FrozenList([1, 2, 3] + lst)
        self.check_result(result, expected)

    def test_iadd(self, container, lst):
        q = r = container

        q += [5]
        self.check_result(q, lst + [5])

        # Other shouldn't be mutated.
        self.check_result(r, lst)

    def test_union(self, container, lst):
        result = container.union((1, 2, 3))
        expected = FrozenList(lst + [1, 2, 3])
        self.check_result(result, expected)

    def test_difference(self, container):
        result = container.difference([2])
        expected = FrozenList([1, 3, 4, 5])
        self.check_result(result, expected)

    def test_difference_dupe(self):
        result = FrozenList([1, 2, 3, 2]).difference([2])
        expected = FrozenList([1, 3])
        self.check_result(result, expected)

    def test_tricky_container_to_bytes_raises(self, unicode_container):
        # GH 26447
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(unicode_container)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_domains.py ===
# Test Matrix/DomainMatrix interaction.


from sympy import GF, ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix, DM

from sympy import (
    Matrix,
    MutableMatrix,
    ImmutableMatrix,
    SparseMatrix,
    MutableDenseMatrix,
    ImmutableDenseMatrix,
    MutableSparseMatrix,
    ImmutableSparseMatrix,
)
from sympy import symbols, S, sqrt

from sympy.testing.pytest import raises


x, y = symbols('x y')


MATRIX_TYPES = (
    Matrix,
    MutableMatrix,
    ImmutableMatrix,
    SparseMatrix,
    MutableDenseMatrix,
    ImmutableDenseMatrix,
    MutableSparseMatrix,
    ImmutableSparseMatrix,
)
IMMUTABLE = (
    ImmutableMatrix,
    ImmutableDenseMatrix,
    ImmutableSparseMatrix,
)


def DMs(items, domain):
    return DM(items, domain).to_sparse()


def test_Matrix_rep_domain():

    for Mat in MATRIX_TYPES:

        M = Mat([[1, 2], [3, 4]])
        assert M._rep == DMs([[1, 2], [3, 4]], ZZ)
        assert (M / 2)._rep == DMs([[(1,2), 1], [(3,2), 2]], QQ)
        if not isinstance(M, IMMUTABLE):
            M[0, 0] = x
            assert M._rep == DMs([[x, 2], [3, 4]], EXRAW)

        M = Mat([[S(1)/2, 2], [3, 4]])
        assert M._rep == DMs([[(1,2), 2], [3, 4]], QQ)
        if not isinstance(M, IMMUTABLE):
            M[0, 0] = x
            assert M._rep == DMs([[x, 2], [3, 4]], EXRAW)

        dM = DMs([[1, 2], [3, 4]], ZZ)
        assert Mat._fromrep(dM)._rep == dM

    # XXX: This is not intended. Perhaps it should be coerced to EXRAW?
    # The private _fromrep method is never called like this but perhaps it
    # should be guarded.
    #
    # It is not clear how to integrate domains other than ZZ, QQ and EXRAW with
    # the rest of Matrix or if the public type for this needs to be something
    # different from Matrix somehow.
    K = QQ.algebraic_field(sqrt(2))
    dM = DM([[1, 2], [3, 4]], K)
    assert Mat._fromrep(dM)._rep.domain == K


def test_Matrix_to_DM():

    M = Matrix([[1, 2], [3, 4]])
    assert M.to_DM() == DMs([[1, 2], [3, 4]], ZZ)
    assert M.to_DM() is not M._rep
    assert M.to_DM(field=True) == DMs([[1, 2], [3, 4]], QQ)
    assert M.to_DM(domain=QQ) == DMs([[1, 2], [3, 4]], QQ)
    assert M.to_DM(domain=QQ[x]) == DMs([[1, 2], [3, 4]], QQ[x])
    assert M.to_DM(domain=GF(3)) == DMs([[1, 2], [0, 1]], GF(3))

    M = Matrix([[1, 2], [3, 4]])
    M[0, 0] = x
    assert M._rep.domain == EXRAW
    M[0, 0] = 1
    assert M.to_DM() == DMs([[1, 2], [3, 4]], ZZ)

    M = Matrix([[S(1)/2, 2], [3, 4]])
    assert M.to_DM() == DMs([[QQ(1,2), 2], [3, 4]], QQ)

    M = Matrix([[x, 2], [3, 4]])
    assert M.to_DM() == DMs([[x, 2], [3, 4]], ZZ[x])
    assert M.to_DM(field=True) == DMs([[x, 2], [3, 4]], ZZ.frac_field(x))

    M = Matrix([[1/x, 2], [3, 4]])
    assert M.to_DM() == DMs([[1/x, 2], [3, 4]], ZZ.frac_field(x))

    M = Matrix([[1, sqrt(2)], [3, 4]])
    K = QQ.algebraic_field(sqrt(2))
    sqrt2 = K.from_sympy(sqrt(2)) # XXX: Maybe K(sqrt(2)) should work
    M_K = DomainMatrix([[K(1), sqrt2], [K(3), K(4)]], (2, 2), K)
    assert M.to_DM() == DMs([[1, sqrt(2)], [3, 4]], EXRAW)
    assert M.to_DM(extension=True) == M_K.to_sparse()

    # Options cannot be used with the domain parameter
    M = Matrix([[1, 2], [3, 4]])
    raises(TypeError, lambda: M.to_DM(domain=QQ, field=True))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_cartesian.py ===
"""Tests for cartesian.py"""

from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL

from sympy.physics.quantum import qapply, represent, L2, Dagger
from sympy.physics.quantum import Commutator, hbar
from sympy.physics.quantum.cartesian import (
    XOp, YOp, ZOp, PxOp, X, Y, Z, Px, XKet, XBra, PxKet, PxBra,
    PositionKet3D, PositionBra3D
)
from sympy.physics.quantum.operator import DifferentialOperator

x, y, z, x_1, x_2, x_3, y_1, z_1 = symbols('x,y,z,x_1,x_2,x_3,y_1,z_1')
px, py, px_1, px_2 = symbols('px py px_1 px_2')


def test_x():
    assert X.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert Commutator(X, Px).doit() == I*hbar
    assert qapply(X*XKet(x)) == x*XKet(x)
    assert XKet(x).dual_class() == XBra
    assert XBra(x).dual_class() == XKet
    assert (Dagger(XKet(y))*XKet(x)).doit() == DiracDelta(x - y)
    assert (PxBra(px)*XKet(x)).doit() == \
        exp(-I*x*px/hbar)/sqrt(2*pi*hbar)
    assert represent(XKet(x)) == DiracDelta(x - x_1)
    assert represent(XBra(x)) == DiracDelta(-x + x_1)
    assert XBra(x).position == x
    assert represent(XOp()*XKet()) == x*DiracDelta(x - x_2)
    assert represent(XBra("y")*XKet()) == DiracDelta(x - y)
    assert represent(
        XKet()*XBra()) == DiracDelta(x - x_2) * DiracDelta(x_1 - x)

    rep_p = represent(XOp(), basis=PxOp)
    assert rep_p == hbar*I*DiracDelta(px_1 - px_2)*DifferentialOperator(px_1)
    assert rep_p == represent(XOp(), basis=PxOp())
    assert rep_p == represent(XOp(), basis=PxKet)
    assert rep_p == represent(XOp(), basis=PxKet())

    assert represent(XOp()*PxKet(), basis=PxKet) == \
        hbar*I*DiracDelta(px - px_2)*DifferentialOperator(px)


@XFAIL
def _text_x_broken():
    # represent has some broken logic that is relying in particular
    # forms of input, rather than a full and proper handling of
    # all valid quantum expressions. Marking this test as XFAIL until
    # we can refactor represent.
    assert represent(XOp()*XKet()*XBra('y')) == \
        x*DiracDelta(x - x_3)*DiracDelta(x_1 - y)


def test_p():
    assert Px.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert qapply(Px*PxKet(px)) == px*PxKet(px)
    assert PxKet(px).dual_class() == PxBra
    assert PxBra(x).dual_class() == PxKet
    assert (Dagger(PxKet(py))*PxKet(px)).doit() == DiracDelta(px - py)
    assert (XBra(x)*PxKet(px)).doit() == \
        exp(I*x*px/hbar)/sqrt(2*pi*hbar)
    assert represent(PxKet(px)) == DiracDelta(px - px_1)

    rep_x = represent(PxOp(), basis=XOp)
    assert rep_x == -hbar*I*DiracDelta(x_1 - x_2)*DifferentialOperator(x_1)
    assert rep_x == represent(PxOp(), basis=XOp())
    assert rep_x == represent(PxOp(), basis=XKet)
    assert rep_x == represent(PxOp(), basis=XKet())

    assert represent(PxOp()*XKet(), basis=XKet) == \
        -hbar*I*DiracDelta(x - x_2)*DifferentialOperator(x)
    assert represent(XBra("y")*PxOp()*XKet(), basis=XKet) == \
        -hbar*I*DiracDelta(x - y)*DifferentialOperator(x)


def test_3dpos():
    assert Y.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))
    assert Z.hilbert_space == L2(Interval(S.NegativeInfinity, S.Infinity))

    test_ket = PositionKet3D(x, y, z)
    assert qapply(X*test_ket) == x*test_ket
    assert qapply(Y*test_ket) == y*test_ket
    assert qapply(Z*test_ket) == z*test_ket
    assert qapply(X*Y*test_ket) == x*y*test_ket
    assert qapply(X*Y*Z*test_ket) == x*y*z*test_ket
    assert qapply(Y*Z*test_ket) == y*z*test_ket

    assert PositionKet3D() == test_ket
    assert YOp() == Y
    assert ZOp() == Z

    assert PositionKet3D.dual_class() == PositionBra3D
    assert PositionBra3D.dual_class() == PositionKet3D

    other_ket = PositionKet3D(x_1, y_1, z_1)
    assert (Dagger(other_ket)*test_ket).doit() == \
        DiracDelta(x - x_1)*DiracDelta(y - y_1)*DiracDelta(z - z_1)

    assert test_ket.position_x == x
    assert test_ket.position_y == y
    assert test_ket.position_z == z
    assert other_ket.position_x == x_1
    assert other_ket.position_y == y_1
    assert other_ket.position_z == z_1

    # TODO: Add tests for representations

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\tests\test_homomorphisms.py ===
"""Tests for homomorphisms."""

from sympy.core.singleton import S
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
from sympy.polys.agca import homomorphism
from sympy.testing.pytest import raises


def test_printing():
    R = QQ.old_poly_ring(x)

    assert str(homomorphism(R.free_module(1), R.free_module(1), [0])) == \
        'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1'
    assert str(homomorphism(R.free_module(2), R.free_module(2), [0, 0])) == \
        'Matrix([                       \n[0, 0], : QQ[x]**2 -> QQ[x]**2\n[0, 0]])                       '
    assert str(homomorphism(R.free_module(1), R.free_module(1) / [[x]], [0])) == \
        'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1/<[x]>'
    assert str(R.free_module(0).identity_hom()) == 'Matrix(0, 0, []) : QQ[x]**0 -> QQ[x]**0'

def test_operations():
    F = QQ.old_poly_ring(x).free_module(2)
    G = QQ.old_poly_ring(x).free_module(3)
    f = F.identity_hom()
    g = homomorphism(F, F, [0, [1, x]])
    h = homomorphism(F, F, [[1, 0], 0])
    i = homomorphism(F, G, [[1, 0, 0], [0, 1, 0]])

    assert f == f
    assert f != g
    assert f != i
    assert (f != F.identity_hom()) is False
    assert 2*f == f*2 == homomorphism(F, F, [[2, 0], [0, 2]])
    assert f/2 == homomorphism(F, F, [[S.Half, 0], [0, S.Half]])
    assert f + g == homomorphism(F, F, [[1, 0], [1, x + 1]])
    assert f - g == homomorphism(F, F, [[1, 0], [-1, 1 - x]])
    assert f*g == g == g*f
    assert h*g == homomorphism(F, F, [0, [1, 0]])
    assert g*h == homomorphism(F, F, [0, 0])
    assert i*f == i
    assert f([1, 2]) == [1, 2]
    assert g([1, 2]) == [2, 2*x]

    assert i.restrict_domain(F.submodule([x, x]))([x, x]) == i([x, x])
    h1 = h.quotient_domain(F.submodule([0, 1]))
    assert h1([1, 0]) == h([1, 0])
    assert h1.restrict_domain(h1.domain.submodule([x, 0]))([x, 0]) == h([x, 0])

    raises(TypeError, lambda: f/g)
    raises(TypeError, lambda: f + 1)
    raises(TypeError, lambda: f + i)
    raises(TypeError, lambda: f - 1)
    raises(TypeError, lambda: f*i)


def test_creation():
    F = QQ.old_poly_ring(x).free_module(3)
    G = QQ.old_poly_ring(x).free_module(2)
    SM = F.submodule([1, 1, 1])
    Q = F / SM
    SQ = Q.submodule([1, 0, 0])

    matrix = [[1, 0], [0, 1], [-1, -1]]
    h = homomorphism(F, G, matrix)
    h2 = homomorphism(Q, G, matrix)
    assert h.quotient_domain(SM) == h2
    raises(ValueError, lambda: h.quotient_domain(F.submodule([1, 0, 0])))
    assert h2.restrict_domain(SQ) == homomorphism(SQ, G, matrix)
    raises(ValueError, lambda: h.restrict_domain(G))
    raises(ValueError, lambda: h.restrict_codomain(G.submodule([1, 0])))
    raises(ValueError, lambda: h.quotient_codomain(F))

    im = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for M in [F, SM, Q, SQ]:
        assert M.identity_hom() == homomorphism(M, M, im)
    assert SM.inclusion_hom() == homomorphism(SM, F, im)
    assert SQ.inclusion_hom() == homomorphism(SQ, Q, im)
    assert Q.quotient_hom() == homomorphism(F, Q, im)
    assert SQ.quotient_hom() == homomorphism(SQ.base, SQ, im)

    class conv:
        def convert(x, y=None):
            return x

    class dummy:
        container = conv()

        def submodule(*args):
            return None
    raises(TypeError, lambda: homomorphism(dummy(), G, matrix))
    raises(TypeError, lambda: homomorphism(F, dummy(), matrix))
    raises(
        ValueError, lambda: homomorphism(QQ.old_poly_ring(x, y).free_module(3), G, matrix))
    raises(ValueError, lambda: homomorphism(F, G, [0, 0]))


def test_properties():
    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    h = homomorphism(F, F, [[x, 0], [y, 0]])
    assert h.kernel() == F.submodule([-y, x])
    assert h.image() == F.submodule([x, 0], [y, 0])
    assert not h.is_injective()
    assert not h.is_surjective()
    assert h.restrict_codomain(h.image()).is_surjective()
    assert h.restrict_domain(F.submodule([1, 0])).is_injective()
    assert h.quotient_domain(
        h.kernel()).restrict_codomain(h.image()).is_isomorphism()

    R2 = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y))) / [x**2 + 1]
    F = R2.free_module(2)
    h = homomorphism(F, F, [[x, 0], [y, y + 1]])
    assert h.is_isomorphism()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\test_utilities.py ===
from sympy.abc import x
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.utilities import (
    AlgIntPowers, coeff_search, extract_fundamental_discriminant,
    isolate, supplement_a_subspace,
)
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.testing.pytest import raises


def test_AlgIntPowers_01():
    T = Poly(cyclotomic_poly(5))
    zeta_pow = AlgIntPowers(T)
    raises(ValueError, lambda: zeta_pow[-1])
    for e in range(10):
        a = e % 5
        if a < 4:
            c = zeta_pow[e]
            assert c[a] == 1 and all(c[i] == 0 for i in range(4) if i != a)
        else:
            assert zeta_pow[e] == [-1] * 4


def test_AlgIntPowers_02():
    T = Poly(x**3 + 2*x**2 + 3*x + 4)
    m = 7
    theta_pow = AlgIntPowers(T, m)
    for e in range(10):
        computed = theta_pow[e]
        coeffs = (Poly(x)**e % T + Poly(x**3)).rep.to_list()[1:]
        expected = [c % m for c in reversed(coeffs)]
        assert computed == expected


def test_coeff_search():
    C = []
    search = coeff_search(2, 1)
    for i, c in enumerate(search):
        C.append(c)
        if i == 12:
            break
    assert C == [[1, 1], [1, 0], [1, -1], [0, 1], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2], [1, 2], [1, -2], [0, 2], [3, 3]]


def test_extract_fundamental_discriminant():
    # To extract, integer must be 0 or 1 mod 4.
    raises(ValueError, lambda: extract_fundamental_discriminant(2))
    raises(ValueError, lambda: extract_fundamental_discriminant(3))
    # Try many cases, of different forms:
    cases = (
        (0, {}, {0: 1}),
        (1, {}, {}),
        (8, {2: 3}, {}),
        (-8, {2: 3, -1: 1}, {}),
        (12, {2: 2, 3: 1}, {}),
        (36, {}, {2: 1, 3: 1}),
        (45, {5: 1}, {3: 1}),
        (48, {2: 2, 3: 1}, {2: 1}),
        (1125, {5: 1}, {3: 1, 5: 1}),
    )
    for a, D_expected, F_expected in cases:
        D, F = extract_fundamental_discriminant(a)
        assert D == D_expected
        assert F == F_expected


def test_supplement_a_subspace_1():
    M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()

    # First supplement over QQ:
    B = supplement_a_subspace(M)
    assert B[:, :2] == M
    assert B[:, 2] == DomainMatrix.eye(3, QQ).to_dense()[:, 0]

    # Now supplement over FF(7):
    M = M.convert_to(FF(7))
    B = supplement_a_subspace(M)
    assert B[:, :2] == M
    # When we work mod 7, first col of M goes to [1, 0, 0],
    # so the supplementary vector cannot equal this, as it did
    # when we worked over QQ. Instead, we get the second std basis vector:
    assert B[:, 2] == DomainMatrix.eye(3, FF(7)).to_dense()[:, 1]


def test_supplement_a_subspace_2():
    M = DM([[1, 0, 0], [2, 0, 0]], QQ).transpose()
    with raises(DMRankError):
        supplement_a_subspace(M)


def test_IntervalPrinter():
    ip = IntervalPrinter()
    assert ip.doprint(x**Rational(1, 3)) == "x**(mpi('1/3'))"
    assert ip.doprint(sqrt(x)) == "x**(mpi('1/2'))"


def test_isolate():
    assert isolate(1) == (1, 1)
    assert isolate(S.Half) == (S.Half, S.Half)

    assert isolate(sqrt(2)) == (1, 2)
    assert isolate(-sqrt(2)) == (-2, -1)

    assert isolate(sqrt(2), eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert isolate(-sqrt(2), eps=Rational(1, 100)) == (Rational(-17, 12), Rational(-24, 17))

    raises(NotImplementedError, lambda: isolate(I))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_errors.py ===
import pytest

from pandas.errors import (
    AbstractMethodError,
    UndefinedVariableError,
)

import pandas as pd


@pytest.mark.parametrize(
    "exc",
    [
        "AttributeConflictWarning",
        "CSSWarning",
        "CategoricalConversionWarning",
        "ClosedFileError",
        "DataError",
        "DatabaseError",
        "DtypeWarning",
        "EmptyDataError",
        "IncompatibilityWarning",
        "IndexingError",
        "InvalidColumnName",
        "InvalidComparison",
        "InvalidVersion",
        "LossySetitemError",
        "MergeError",
        "NoBufferPresent",
        "NumExprClobberingError",
        "NumbaUtilError",
        "OptionError",
        "OutOfBoundsDatetime",
        "ParserError",
        "ParserWarning",
        "PerformanceWarning",
        "PossibleDataLossError",
        "PossiblePrecisionLoss",
        "PyperclipException",
        "SettingWithCopyError",
        "SettingWithCopyWarning",
        "SpecificationError",
        "UnsortedIndexError",
        "UnsupportedFunctionCall",
        "ValueLabelTypeMismatch",
    ],
)
def test_exception_importable(exc):
    from pandas import errors

    err = getattr(errors, exc)
    assert err is not None

    # check that we can raise on them

    msg = "^$"

    with pytest.raises(err, match=msg):
        raise err()


def test_catch_oob():
    from pandas import errors

    msg = "Cannot cast 1500-01-01 00:00:00 to unit='ns' without overflow"
    with pytest.raises(errors.OutOfBoundsDatetime, match=msg):
        pd.Timestamp("15000101").as_unit("ns")


@pytest.mark.parametrize(
    "is_local",
    [
        True,
        False,
    ],
)
def test_catch_undefined_variable_error(is_local):
    variable_name = "x"
    if is_local:
        msg = f"local variable '{variable_name}' is not defined"
    else:
        msg = f"name '{variable_name}' is not defined"

    with pytest.raises(UndefinedVariableError, match=msg):
        raise UndefinedVariableError(variable_name, is_local)


class Foo:
    @classmethod
    def classmethod(cls):
        raise AbstractMethodError(cls, methodtype="classmethod")

    @property
    def property(self):
        raise AbstractMethodError(self, methodtype="property")

    def method(self):
        raise AbstractMethodError(self)


def test_AbstractMethodError_classmethod():
    xpr = "This classmethod must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo.classmethod()

    xpr = "This property must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().property

    xpr = "This method must be defined in the concrete class Foo"
    with pytest.raises(AbstractMethodError, match=xpr):
        Foo().method()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_online.py ===
import numpy as np
import pytest

from pandas.compat import is_platform_arm

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.util.version import Version

pytestmark = [pytest.mark.single_cpu]

numba = pytest.importorskip("numba")
pytestmark.append(
    pytest.mark.skipif(
        Version(numba.__version__) == Version("0.61") and is_platform_arm(),
        reason=f"Segfaults on ARM platforms with numba {numba.__version__}",
    )
)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestEWM:
    def test_invalid_update(self):
        df = DataFrame({"a": range(5), "b": range(5)})
        online_ewm = df.head(2).ewm(0.5).online()
        with pytest.raises(
            ValueError,
            match="Must call mean with update=None first before passing update",
        ):
            online_ewm.mean(update=df.head(1))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "obj", [DataFrame({"a": range(5), "b": range(5)}), Series(range(5), name="foo")]
    )
    def test_online_vs_non_online_mean(
        self, obj, nogil, parallel, nopython, adjust, ignore_na
    ):
        expected = obj.ewm(0.5, adjust=adjust, ignore_na=ignore_na).mean()
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        online_ewm = (
            obj.head(2)
            .ewm(0.5, adjust=adjust, ignore_na=ignore_na)
            .online(engine_kwargs=engine_kwargs)
        )
        # Test resetting once
        for _ in range(2):
            result = online_ewm.mean()
            tm.assert_equal(result, expected.head(2))

            result = online_ewm.mean(update=obj.tail(3))
            tm.assert_equal(result, expected.tail(3))

            online_ewm.reset()

    @pytest.mark.xfail(raises=NotImplementedError)
    @pytest.mark.parametrize(
        "obj", [DataFrame({"a": range(5), "b": range(5)}), Series(range(5), name="foo")]
    )
    def test_update_times_mean(
        self, obj, nogil, parallel, nopython, adjust, ignore_na, halflife_with_times
    ):
        times = Series(
            np.array(
                ["2020-01-01", "2020-01-05", "2020-01-07", "2020-01-17", "2020-01-21"],
                dtype="datetime64[ns]",
            )
        )
        expected = obj.ewm(
            0.5,
            adjust=adjust,
            ignore_na=ignore_na,
            times=times,
            halflife=halflife_with_times,
        ).mean()

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        online_ewm = (
            obj.head(2)
            .ewm(
                0.5,
                adjust=adjust,
                ignore_na=ignore_na,
                times=times.head(2),
                halflife=halflife_with_times,
            )
            .online(engine_kwargs=engine_kwargs)
        )
        # Test resetting once
        for _ in range(2):
            result = online_ewm.mean()
            tm.assert_equal(result, expected.head(2))

            result = online_ewm.mean(update=obj.tail(3), update_times=times.tail(3))
            tm.assert_equal(result, expected.tail(3))

            online_ewm.reset()

    @pytest.mark.parametrize("method", ["aggregate", "std", "corr", "cov", "var"])
    def test_ewm_notimplementederror_raises(self, method):
        ser = Series(range(10))
        kwargs = {}
        if method == "aggregate":
            kwargs["func"] = lambda x: x

        with pytest.raises(NotImplementedError, match=".* is not implemented."):
            getattr(ser.ewm(1).online(), method)(**kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_cnodes.py ===
from sympy.core.symbol import symbols
from sympy.printing.codeprinter import ccode
from sympy.codegen.ast import Declaration, Variable, float64, int64, String, CodeBlock
from sympy.codegen.cnodes import (
    alignof, CommaOperator, goto, Label, PreDecrement, PostDecrement, PreIncrement, PostIncrement,
    sizeof, union, struct
)

x, y = symbols('x y')


def test_alignof():
    ax = alignof(x)
    assert ccode(ax) == 'alignof(x)'
    assert ax.func(*ax.args) == ax


def test_CommaOperator():
    expr = CommaOperator(PreIncrement(x), 2*x)
    assert ccode(expr) == '(++(x), 2*x)'
    assert expr.func(*expr.args) == expr


def test_goto_Label():
    s = 'early_exit'
    g = goto(s)
    assert g.func(*g.args) == g
    assert g != goto('foobar')
    assert ccode(g) == 'goto early_exit'

    l1 = Label(s)
    assert ccode(l1) == 'early_exit:'
    assert l1 == Label('early_exit')
    assert l1 != Label('foobar')

    body = [PreIncrement(x)]
    l2 = Label(s, body)
    assert l2.name == String("early_exit")
    assert l2.body == CodeBlock(PreIncrement(x))
    assert ccode(l2) == ("early_exit:\n"
        "++(x);")

    body = [PreIncrement(x), PreDecrement(y)]
    l2 = Label(s, body)
    assert l2.name == String("early_exit")
    assert l2.body == CodeBlock(PreIncrement(x), PreDecrement(y))
    assert ccode(l2) == ("early_exit:\n"
        "{\n   ++(x);\n   --(y);\n}")


def test_PreDecrement():
    p = PreDecrement(x)
    assert p.func(*p.args) == p
    assert ccode(p) == '--(x)'


def test_PostDecrement():
    p = PostDecrement(x)
    assert p.func(*p.args) == p
    assert ccode(p) == '(x)--'


def test_PreIncrement():
    p = PreIncrement(x)
    assert p.func(*p.args) == p
    assert ccode(p) == '++(x)'


def test_PostIncrement():
    p = PostIncrement(x)
    assert p.func(*p.args) == p
    assert ccode(p) == '(x)++'


def test_sizeof():
    typename = 'unsigned int'
    sz = sizeof(typename)
    assert ccode(sz) == 'sizeof(%s)' % typename
    assert sz.func(*sz.args) == sz
    assert not sz.is_Atom
    assert sz.atoms() == {String('unsigned int'), String('sizeof')}


def test_struct():
    vx, vy = Variable(x, type=float64), Variable(y, type=float64)
    s = struct('vec2', [vx, vy])
    assert s.func(*s.args) == s
    assert s == struct('vec2', (vx, vy))
    assert s != struct('vec2', (vy, vx))
    assert str(s.name) == 'vec2'
    assert len(s.declarations) == 2
    assert all(isinstance(arg, Declaration) for arg in s.declarations)
    assert ccode(s) == (
        "struct vec2 {\n"
        "   double x;\n"
        "   double y;\n"
        "}")


def test_union():
    vx, vy = Variable(x, type=float64), Variable(y, type=int64)
    u = union('dualuse', [vx, vy])
    assert u.func(*u.args) == u
    assert u == union('dualuse', (vx, vy))
    assert str(u.name) == 'dualuse'
    assert len(u.declarations) == 2
    assert all(isinstance(arg, Declaration) for arg in u.declarations)
    assert ccode(u) == (
        "union dualuse {\n"
        "   double x;\n"
        "   int64_t y;\n"
        "}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_linsolve.py ===
#
# test_linsolve.py
#
#    Test the internal implementation of linsolve.
#

from sympy.testing.pytest import raises

from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z

from sympy.polys.matrices.linsolve import _linsolve
from sympy.polys.solvers import PolyNonlinearError


def test__linsolve():
    assert _linsolve([], [x]) == {x:x}
    assert _linsolve([S.Zero], [x]) == {x:x}
    assert _linsolve([x-1,x-2], [x]) is None
    assert _linsolve([x-1], [x]) == {x:1}
    assert _linsolve([x-1, y], [x, y]) == {x:1, y:S.Zero}
    assert _linsolve([2*I], [x]) is None
    raises(PolyNonlinearError, lambda: _linsolve([x*(1 + x)], [x]))


def test__linsolve_float():

    # This should give the exact answer:
    eqs = [
        y - x,
        y - 0.0216 * x
    ]
    # Should _linsolve return floats here?
    sol = {x:0, y:0}
    assert _linsolve(eqs, (x, y)) == sol

    # Other cases should be close to eps

    def all_close(sol1, sol2, eps=1e-15):
        close = lambda a, b: abs(a - b) < eps
        assert sol1.keys() == sol2.keys()
        return all(close(sol1[s], sol2[s]) for s in sol1)

    eqs = [
        0.8*x +         0.8*z + 0.2,
        0.9*x + 0.7*y + 0.2*z + 0.9,
        0.7*x + 0.2*y + 0.2*z + 0.5
    ]
    sol_exact = {x:-29/42, y:-11/21, z:37/84}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        0.9*x + 0.3*y + 0.4*z + 0.6,
        0.6*x + 0.9*y + 0.1*z + 0.7,
        0.4*x + 0.6*y + 0.9*z + 0.5
    ]
    sol_exact = {x:-88/175, y:-46/105, z:-1/25}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        0.4*x + 0.3*y + 0.6*z + 0.7,
        0.4*x + 0.3*y + 0.9*z + 0.9,
        0.7*x + 0.9*y,
    ]
    sol_exact = {x:-9/5, y:7/5, z:-2/3}
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    eqs = [
        x*(0.7 + 0.6*I) + y*(0.4 + 0.7*I) + z*(0.9 + 0.1*I) + 0.5,
        0.2*I*x + 0.2*I*y + z*(0.9 + 0.2*I) + 0.1,
        x*(0.9 + 0.7*I) + y*(0.9 + 0.7*I) + z*(0.9 + 0.4*I) + 0.4,
    ]
    sol_exact = {
        x:-6157/7995 - 411/5330*I,
        y:8519/15990 + 1784/7995*I,
        z:-34/533 + 107/1599*I,
    }
    sol_linsolve = _linsolve(eqs, [x,y,z])
    assert all_close(sol_exact, sol_linsolve)

    # XXX: This system for x and y over RR(z) is problematic.
    #
    # eqs = [
    #     x*(0.2*z + 0.9) + y*(0.5*z + 0.8) + 0.6,
    #     0.1*x*z + y*(0.1*z + 0.6) + 0.9,
    # ]
    #
    # linsolve(eqs, [x, y])
    # The solution for x comes out as
    #
    #       -3.9e-5*z**2 - 3.6e-5*z - 8.67361737988404e-20
    #  x =  ----------------------------------------------
    #           3.0e-6*z**3 - 1.3e-5*z**2 - 5.4e-5*z
    #
    # The 8e-20 in the numerator should be zero which would allow z to cancel
    # from top and bottom. It should be possible to avoid this somehow because
    # the inverse of the matrix only has a quadratic factor (the determinant)
    # in the denominator.


def test__linsolve_deprecated():
    raises(PolyNonlinearError, lambda:
        _linsolve([Eq(x**2, x**2 + y)], [x, y]))
    raises(PolyNonlinearError, lambda:
        _linsolve([(x + y)**2 - x**2], [x]))
    raises(PolyNonlinearError, lambda:
        _linsolve([Eq((x + y)**2, x**2)], [x]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_solvers.py ===
"""Tests for low-level linear systems solver. """

from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix


def test_solve_lin_sys_2x2_one():
    domain, x1,x2 = ring("x1,x2", QQ)
    eqs = [x1 + x2 - 5,
           2*x1 - x2]
    sol = {x1: QQ(5, 3), x2: QQ(10, 3)}
    _sol = solve_lin_sys(eqs, domain)
    assert _sol == sol and all(s.ring == domain for s in _sol)

def test_solve_lin_sys_2x4_none():
    domain, x1,x2 = ring("x1,x2", QQ)
    eqs = [x1 - 1,
           x1 - x2,
           x1 - 2*x2,
           x2 - 1]
    assert solve_lin_sys(eqs, domain) is None


def test_solve_lin_sys_3x4_one():
    domain, x1,x2,x3 = ring("x1,x2,x3", QQ)
    eqs = [x1 + 2*x2 + 3*x3,
           2*x1 - x2 + x3,
           3*x1 + x2 + x3,
           5*x2 + 2*x3]
    sol = {x1: 0, x2: 0, x3: 0}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_3x3_inf():
    domain, x1,x2,x3 = ring("x1,x2,x3", QQ)
    eqs = [x1 - x2 + 2*x3 - 1,
           2*x1 + x2 + x3 - 8,
           x1 + x2 - 5]
    sol = {x1: -x3 + 3, x2: x3 + 2}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_3x4_none():
    domain, x1,x2,x3,x4 = ring("x1,x2,x3,x4", QQ)
    eqs = [2*x1 + x2 + 7*x3 - 7*x4 - 2,
           -3*x1 + 4*x2 - 5*x3 - 6*x4 - 3,
           x1 + x2 + 4*x3 - 5*x4 - 2]
    assert solve_lin_sys(eqs, domain) is None


def test_solve_lin_sys_4x7_inf():
    domain, x1,x2,x3,x4,x5,x6,x7 = ring("x1,x2,x3,x4,x5,x6,x7", QQ)
    eqs = [x1 + 4*x2 - x4 + 7*x6 - 9*x7 - 3,
           2*x1 + 8*x2 - x3 + 3*x4 + 9*x5 - 13*x6 + 7*x7 - 9,
           2*x3 - 3*x4 - 4*x5 + 12*x6 - 8*x7 - 1,
           -x1 - 4*x2 + 2*x3 + 4*x4 + 8*x5 - 31*x6 + 37*x7 - 4]
    sol = {x1: 4 - 4*x2 - 2*x5 - x6 + 3*x7,
           x3: 2 - x5 + 3*x6 - 5*x7,
           x4: 1 - 2*x5 + 6*x6 - 6*x7}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_5x5_inf():
    domain, x1,x2,x3,x4,x5 = ring("x1,x2,x3,x4,x5", QQ)
    eqs = [x1 - x2 - 2*x3 + x4 + 11*x5 - 13,
           x1 - x2 + x3 + x4 + 5*x5 - 16,
           2*x1 - 2*x2 + x4 + 10*x5 - 21,
           2*x1 - 2*x2 - x3 + 3*x4 + 20*x5 - 38,
           2*x1 - 2*x2 + x3 + x4 + 8*x5 - 22]
    sol = {x1: 6 + x2 - 3*x5,
           x3: 1 + 2*x5,
           x4: 9 - 4*x5}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_6x6_1():
    ground, d,r,e,g,i,j,l,o,m,p,q = field("d,r,e,g,i,j,l,o,m,p,q", ZZ)
    domain, c,f,h,k,n,b = ring("c,f,h,k,n,b", ground)

    eqs = [b + q/d - c/d, c*(1/d + 1/e + 1/g) - f/g - q/d, f*(1/g + 1/i + 1/j) - c/g - h/i, h*(1/i + 1/l + 1/m) - f/i - k/m, k*(1/m + 1/o + 1/p) - h/m - n/p, n/p - k/p]
    sol = {
         b: (e*i*l*q + e*i*m*q + e*i*o*q + e*j*l*q + e*j*m*q + e*j*o*q + e*l*m*q + e*l*o*q + g*i*l*q + g*i*m*q + g*i*o*q + g*j*l*q + g*j*m*q + g*j*o*q + g*l*m*q + g*l*o*q + i*j*l*q + i*j*m*q + i*j*o*q + j*l*m*q + j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         c: (-e*g*i*l*q - e*g*i*m*q - e*g*i*o*q - e*g*j*l*q - e*g*j*m*q - e*g*j*o*q - e*g*l*m*q - e*g*l*o*q - e*i*j*l*q - e*i*j*m*q - e*i*j*o*q - e*j*l*m*q - e*j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         f: (-e*i*j*l*q - e*i*j*m*q - e*i*j*o*q - e*j*l*m*q - e*j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         h: (-e*j*l*m*q - e*j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         k: e*j*l*o*q/(d*e*i*l + d*e*i*m + d*e*i*o + d*e*j*l + d*e*j*m + d*e*j*o + d*e*l*m + d*e*l*o + d*g*i*l + d*g*i*m + d*g*i*o + d*g*j*l + d*g*j*m + d*g*j*o + d*g*l*m + d*g*l*o + d*i*j*l + d*i*j*m + d*i*j*o + d*j*l*m + d*j*l*o + e*g*i*l + e*g*i*m + e*g*i*o + e*g*j*l + e*g*j*m + e*g*j*o + e*g*l*m + e*g*l*o + e*i*j*l + e*i*j*m + e*i*j*o + e*j*l*m + e*j*l*o),
         n: e*j*l*o*q/(d*e*i*l + d*e*i*m + d*e*i*o + d*e*j*l + d*e*j*m + d*e*j*o + d*e*l*m + d*e*l*o + d*g*i*l + d*g*i*m + d*g*i*o + d*g*j*l + d*g*j*m + d*g*j*o + d*g*l*m + d*g*l*o + d*i*j*l + d*i*j*m + d*i*j*o + d*j*l*m + d*j*l*o + e*g*i*l + e*g*i*m + e*g*i*o + e*g*j*l + e*g*j*m + e*g*j*o + e*g*l*m + e*g*l*o + e*i*j*l + e*i*j*m + e*i*j*o + e*j*l*m + e*j*l*o),
    }

    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_6x6_2():
    ground, d,r,e,g,i,j,l,o,m,p,q = field("d,r,e,g,i,j,l,o,m,p,q", ZZ)
    domain, c,f,h,k,n,b = ring("c,f,h,k,n,b", ground)

    eqs = [b + r/d - c/d, c*(1/d + 1/e + 1/g) - f/g - r/d, f*(1/g + 1/i + 1/j) - c/g - h/i, h*(1/i + 1/l + 1/m) - f/i - k/m, k*(1/m + 1/o + 1/p) - h/m - n/p, n*(1/p + 1/q) - k/p]
    sol = {
        b: -((l*q*e*o + l*q*g*o + i*m*q*e + i*l*q*e + i*l*p*e + i*j*o*q + j*e*o*q + g*j*o*q + i*e*o*q + g*i*o*q + e*l*o*p + e*l*m*p + e*l*m*o + e*i*o*p + e*i*m*p + e*i*m*o + e*i*l*o + j*e*o*p + j*e*m*q + j*e*m*p + j*e*m*o + j*l*m*q + j*l*m*p + j*l*m*o + i*j*m*p + i*j*m*o + i*j*l*q + i*j*l*o + i*j*m*q + j*l*o*p + j*e*l*o + g*j*o*p + g*j*m*q + g*j*m*p + i*j*l*p + i*j*o*p + j*e*l*q + j*e*l*p + j*l*o*q + g*j*m*o + g*j*l*q + g*j*l*p + g*j*l*o + g*l*o*p + g*l*m*p + g*l*m*o + g*i*m*o + g*i*o*p + g*i*m*q + g*i*m*p + g*i*l*q + g*i*l*p + g*i*l*o + l*m*q*e + l*m*q*g)*r)/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        c: (r*e*(l*q*g*o + i*j*o*q + g*j*o*q + g*i*o*q + j*l*m*q + j*l*m*p + j*l*m*o + i*j*m*p + i*j*m*o + i*j*l*q + i*j*l*o + i*j*m*q + j*l*o*p + g*j*o*p + g*j*m*q + g*j*m*p + i*j*l*p + i*j*o*p + j*l*o*q + g*j*m*o + g*j*l*q + g*j*l*p + g*j*l*o + g*l*o*p + g*l*m*p + g*l*m*o + g*i*m*o + g*i*o*p + g*i*m*q + g*i*m*p + g*i*l*q + g*i*l*p + g*i*l*o + l*m*q*g))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        f: (r*e*j*(l*q*o + l*o*p + l*m*q + l*m*p + l*m*o + i*o*q + i*o*p + i*m*q + i*m*p + i*m*o + i*l*q + i*l*p + i*l*o))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        h: (j*e*r*l*(o*q + o*p + m*q + m*p + m*o))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        k: (j*e*r*o*l*(q + p))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        n: (j*e*r*o*q*l)/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
    }

    assert solve_lin_sys(eqs, domain) == sol

def test_eqs_to_matrix():
    domain, x1,x2 = ring("x1,x2", QQ)
    eqs_coeff = [{x1: QQ(1), x2: QQ(1)}, {x1: QQ(2), x2: QQ(-1)}]
    eqs_rhs = [QQ(-5), QQ(0)]
    M = eqs_to_matrix(eqs_coeff, eqs_rhs, [x1, x2], QQ)
    assert M.to_Matrix() == Matrix([[1, 1, 5], [2, -1, 0]])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_replace.py ===
import pytest

import pandas as pd
from pandas import Categorical
import pandas._testing as tm


@pytest.mark.parametrize(
    "to_replace,value,expected,flip_categories",
    [
        # one-to-one
        (1, 2, [2, 2, 3], False),
        (1, 4, [4, 2, 3], False),
        (4, 1, [1, 2, 3], False),
        (5, 6, [1, 2, 3], False),
        # many-to-one
        ([1], 2, [2, 2, 3], False),
        ([1, 2], 3, [3, 3, 3], False),
        ([1, 2], 4, [4, 4, 3], False),
        ((1, 2, 4), 5, [5, 5, 3], False),
        ((5, 6), 2, [1, 2, 3], False),
        ([1], [2], [2, 2, 3], False),
        ([1, 4], [5, 2], [5, 2, 3], False),
        # GH49404: overlap between to_replace and value
        ([1, 2, 3], [2, 3, 4], [2, 3, 4], False),
        # GH50872, GH46884: replace with null
        (1, None, [None, 2, 3], False),
        (1, pd.NA, [None, 2, 3], False),
        # check_categorical sorts categories, which crashes on mixed dtypes
        (3, "4", [1, 2, "4"], False),
        ([1, 2, "3"], "5", ["5", "5", 3], True),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:.*with CategoricalDtype is deprecated:FutureWarning"
)
def test_replace_categorical_series(to_replace, value, expected, flip_categories):
    # GH 31720

    ser = pd.Series([1, 2, 3], dtype="category")
    result = ser.replace(to_replace, value)
    expected = pd.Series(expected, dtype="category")
    ser.replace(to_replace, value, inplace=True)

    if flip_categories:
        expected = expected.cat.set_categories(expected.cat.categories[::-1])

    tm.assert_series_equal(expected, result, check_category_order=False)
    tm.assert_series_equal(expected, ser, check_category_order=False)


@pytest.mark.parametrize(
    "to_replace, value, result, expected_error_msg",
    [
        ("b", "c", ["a", "c"], "Categorical.categories are different"),
        ("c", "d", ["a", "b"], None),
        # https://github.com/pandas-dev/pandas/issues/33288
        ("a", "a", ["a", "b"], None),
        ("b", None, ["a", None], "Categorical.categories length are different"),
    ],
)
def test_replace_categorical(to_replace, value, result, expected_error_msg):
    # GH#26988
    cat = Categorical(["a", "b"])
    expected = Categorical(result)
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    warn = FutureWarning if expected_error_msg is not None else None
    with tm.assert_produces_warning(warn, match=msg):
        result = pd.Series(cat, copy=False).replace(to_replace, value)._values

    tm.assert_categorical_equal(result, expected)
    if to_replace == "b":  # the "c" test is supposed to be unchanged
        with pytest.raises(AssertionError, match=expected_error_msg):
            # ensure non-inplace call does not affect original
            tm.assert_categorical_equal(cat, expected)

    ser = pd.Series(cat, copy=False)
    with tm.assert_produces_warning(warn, match=msg):
        ser.replace(to_replace, value, inplace=True)
    tm.assert_categorical_equal(cat, expected)


def test_replace_categorical_ea_dtype():
    # GH49404
    cat = Categorical(pd.array(["a", "b"], dtype="string"))
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pd.Series(cat).replace(["a", "b"], ["c", pd.NA])._values
    expected = Categorical(pd.array(["c", pd.NA], dtype="string"))
    tm.assert_categorical_equal(result, expected)


def test_replace_maintain_ordering():
    # GH51016
    dtype = pd.CategoricalDtype([0, 1, 2], ordered=True)
    ser = pd.Series([0, 1, 2], dtype=dtype)
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.replace(0, 2)
    expected_dtype = pd.CategoricalDtype([1, 2], ordered=True)
    expected = pd.Series([2, 1, 2], dtype=expected_dtype)
    tm.assert_series_equal(expected, result, check_category_order=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_rename_axis.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm


class TestDataFrameRenameAxis:
    def test_rename_axis_inplace(self, float_frame):
        # GH#15704
        expected = float_frame.rename_axis("foo")
        result = float_frame.copy()
        return_value = no_return = result.rename_axis("foo", inplace=True)
        assert return_value is None

        assert no_return is None
        tm.assert_frame_equal(result, expected)

        expected = float_frame.rename_axis("bar", axis=1)
        result = float_frame.copy()
        return_value = no_return = result.rename_axis("bar", axis=1, inplace=True)
        assert return_value is None

        assert no_return is None
        tm.assert_frame_equal(result, expected)

    def test_rename_axis_raises(self):
        # GH#17833
        df = DataFrame({"A": [1, 2], "B": [1, 2]})
        with pytest.raises(ValueError, match="Use `.rename`"):
            df.rename_axis(id, axis=0)

        with pytest.raises(ValueError, match="Use `.rename`"):
            df.rename_axis({0: 10, 1: 20}, axis=0)

        with pytest.raises(ValueError, match="Use `.rename`"):
            df.rename_axis(id, axis=1)

        with pytest.raises(ValueError, match="Use `.rename`"):
            df["A"].rename_axis(id)

    def test_rename_axis_mapper(self):
        # GH#19978
        mi = MultiIndex.from_product([["a", "b", "c"], [1, 2]], names=["ll", "nn"])
        df = DataFrame(
            {"x": list(range(len(mi))), "y": [i * 10 for i in range(len(mi))]}, index=mi
        )

        # Test for rename of the Index object of columns
        result = df.rename_axis("cols", axis=1)
        tm.assert_index_equal(result.columns, Index(["x", "y"], name="cols"))

        # Test for rename of the Index object of columns using dict
        result = result.rename_axis(columns={"cols": "new"}, axis=1)
        tm.assert_index_equal(result.columns, Index(["x", "y"], name="new"))

        # Test for renaming index using dict
        result = df.rename_axis(index={"ll": "foo"})
        assert result.index.names == ["foo", "nn"]

        # Test for renaming index using a function
        result = df.rename_axis(index=str.upper, axis=0)
        assert result.index.names == ["LL", "NN"]

        # Test for renaming index providing complete list
        result = df.rename_axis(index=["foo", "goo"])
        assert result.index.names == ["foo", "goo"]

        # Test for changing index and columns at same time
        sdf = df.reset_index().set_index("nn").drop(columns=["ll", "y"])
        result = sdf.rename_axis(index="foo", columns="meh")
        assert result.index.name == "foo"
        assert result.columns.name == "meh"

        # Test different error cases
        with pytest.raises(TypeError, match="Must pass"):
            df.rename_axis(index="wrong")

        with pytest.raises(ValueError, match="Length of names"):
            df.rename_axis(index=["wrong"])

        with pytest.raises(TypeError, match="bogus"):
            df.rename_axis(bogus=None)

    @pytest.mark.parametrize(
        "kwargs, rename_index, rename_columns",
        [
            ({"mapper": None, "axis": 0}, True, False),
            ({"mapper": None, "axis": 1}, False, True),
            ({"index": None}, True, False),
            ({"columns": None}, False, True),
            ({"index": None, "columns": None}, True, True),
            ({}, False, False),
        ],
    )
    def test_rename_axis_none(self, kwargs, rename_index, rename_columns):
        # GH 25034
        index = Index(list("abc"), name="foo")
        columns = Index(["col1", "col2"], name="bar")
        data = np.arange(6).reshape(3, 2)
        df = DataFrame(data, index, columns)

        result = df.rename_axis(**kwargs)
        expected_index = index.rename(None) if rename_index else index
        expected_columns = columns.rename(None) if rename_columns else columns
        expected = DataFrame(data, expected_index, expected_columns)
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_missing.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm


def test_fillna(idx):
    # GH 11343
    msg = "isna is not defined for MultiIndex"
    with pytest.raises(NotImplementedError, match=msg):
        idx.fillna(idx[0])


def test_dropna():
    # GH 6194
    idx = MultiIndex.from_arrays(
        [
            [1, np.nan, 3, np.nan, 5],
            [1, 2, np.nan, np.nan, 5],
            ["a", "b", "c", np.nan, "e"],
        ]
    )

    exp = MultiIndex.from_arrays([[1, 5], [1, 5], ["a", "e"]])
    tm.assert_index_equal(idx.dropna(), exp)
    tm.assert_index_equal(idx.dropna(how="any"), exp)

    exp = MultiIndex.from_arrays(
        [[1, np.nan, 3, 5], [1, 2, np.nan, 5], ["a", "b", "c", "e"]]
    )
    tm.assert_index_equal(idx.dropna(how="all"), exp)

    msg = "invalid how option: xxx"
    with pytest.raises(ValueError, match=msg):
        idx.dropna(how="xxx")

    # GH26408
    # test if missing values are dropped for multiindex constructed
    # from codes and values
    idx = MultiIndex(
        levels=[[np.nan, None, pd.NaT, "128", 2], [np.nan, None, pd.NaT, "128", 2]],
        codes=[[0, -1, 1, 2, 3, 4], [0, -1, 3, 3, 3, 4]],
    )
    expected = MultiIndex.from_arrays([["128", 2], ["128", 2]])
    tm.assert_index_equal(idx.dropna(), expected)
    tm.assert_index_equal(idx.dropna(how="any"), expected)

    expected = MultiIndex.from_arrays(
        [[np.nan, np.nan, "128", 2], ["128", "128", "128", 2]]
    )
    tm.assert_index_equal(idx.dropna(how="all"), expected)


def test_nulls(idx):
    # this is really a smoke test for the methods
    # as these are adequately tested for function elsewhere

    msg = "isna is not defined for MultiIndex"
    with pytest.raises(NotImplementedError, match=msg):
        idx.isna()


@pytest.mark.xfail(reason="isna is not defined for MultiIndex")
def test_hasnans_isnans(idx):
    # GH 11343, added tests for hasnans / isnans
    index = idx.copy()

    # cases in indices doesn't include NaN
    expected = np.array([False] * len(index), dtype=bool)
    tm.assert_numpy_array_equal(index._isnan, expected)
    assert index.hasnans is False

    index = idx.copy()
    values = index.values
    values[1] = np.nan

    index = type(idx)(values)

    expected = np.array([False] * len(index), dtype=bool)
    expected[1] = True
    tm.assert_numpy_array_equal(index._isnan, expected)
    assert index.hasnans is True


def test_nan_stays_float():
    # GH 7031
    idx0 = MultiIndex(levels=[["A", "B"], []], codes=[[1, 0], [-1, -1]], names=[0, 1])
    idx1 = MultiIndex(levels=[["C"], ["D"]], codes=[[0], [0]], names=[0, 1])
    idxm = idx0.join(idx1, how="outer")
    assert pd.isna(idx0.get_level_values(1)).all()
    # the following failed in 0.14.1
    assert pd.isna(idxm.get_level_values(1)[:-1]).all()

    df0 = pd.DataFrame([[1, 2]], index=idx0)
    df1 = pd.DataFrame([[3, 4]], index=idx1)
    dfm = df0 - df1
    assert pd.isna(df0.index.get_level_values(1)).all()
    # the following failed in 0.14.1
    assert pd.isna(dfm.index.get_level_values(1)[:-1]).all()


def test_tuples_have_na():
    index = MultiIndex(
        levels=[[1, 0], [0, 1, 2, 3]],
        codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
    )

    assert pd.isna(index[4][0])
    assert pd.isna(index.values[4][0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\merge\test_merge_cross.py ===
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.core.reshape.merge import (
    MergeError,
    merge,
)


@pytest.mark.parametrize(
    ("input_col", "output_cols"), [("b", ["a", "b"]), ("a", ["a_x", "a_y"])]
)
def test_merge_cross(input_col, output_cols):
    # GH#5401
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({input_col: [3, 4]})
    left_copy = left.copy()
    right_copy = right.copy()
    result = merge(left, right, how="cross")
    expected = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(left, left_copy)
    tm.assert_frame_equal(right, right_copy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True},
        {"right_index": True},
        {"on": "a"},
        {"left_on": "a"},
        {"right_on": "b"},
    ],
)
def test_merge_cross_error_reporting(kwargs):
    # GH#5401
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({"b": [3, 4]})
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or "
        "left_index=True"
    )
    with pytest.raises(MergeError, match=msg):
        merge(left, right, how="cross", **kwargs)


def test_merge_cross_mixed_dtypes():
    # GH#5401
    left = DataFrame(["a", "b", "c"], columns=["A"])
    right = DataFrame(range(2), columns=["B"])
    result = merge(left, right, how="cross")
    expected = DataFrame({"A": ["a", "a", "b", "b", "c", "c"], "B": [0, 1, 0, 1, 0, 1]})
    tm.assert_frame_equal(result, expected)


def test_merge_cross_more_than_one_column():
    # GH#5401
    left = DataFrame({"A": list("ab"), "B": [2, 1]})
    right = DataFrame({"C": range(2), "D": range(4, 6)})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "A": ["a", "a", "b", "b"],
            "B": [2, 2, 1, 1],
            "C": [0, 1, 0, 1],
            "D": [4, 5, 4, 5],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_merge_cross_null_values(nulls_fixture):
    # GH#5401
    left = DataFrame({"a": [1, nulls_fixture]})
    right = DataFrame({"b": ["a", "b"], "c": [1.0, 2.0]})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "a": [1, 1, nulls_fixture, nulls_fixture],
            "b": ["a", "b", "a", "b"],
            "c": [1.0, 2.0, 1.0, 2.0],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_join_cross_error_reporting():
    # GH#5401
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({"a": [3, 4]})
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or "
        "left_index=True"
    )
    with pytest.raises(MergeError, match=msg):
        left.join(right, how="cross", on="a")


def test_merge_cross_series():
    # GH#54055
    ls = Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left")
    rs = Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right")
    res = merge(ls, rs, how="cross")

    expected = merge(ls.to_frame(), rs.to_frame(), how="cross")
    tm.assert_frame_equal(res, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\test_string_array.py ===
import numpy as np
import pytest

from pandas._libs import lib

from pandas import (
    NA,
    DataFrame,
    Series,
    _testing as tm,
    option_context,
)


def test_string_array(nullable_string_dtype, any_string_method):
    method_name, args, kwargs = any_string_method

    data = ["a", "bb", np.nan, "ccc"]
    a = Series(data, dtype=object)
    b = Series(data, dtype=nullable_string_dtype)

    if method_name == "decode":
        with pytest.raises(TypeError, match="a bytes-like object is required"):
            getattr(b.str, method_name)(*args, **kwargs)
        return

    expected = getattr(a.str, method_name)(*args, **kwargs)
    result = getattr(b.str, method_name)(*args, **kwargs)

    if isinstance(expected, Series):
        if expected.dtype == "object" and lib.is_string_array(
            expected.dropna().values,
        ):
            assert result.dtype == nullable_string_dtype
            result = result.astype(object)

        elif expected.dtype == "object" and lib.is_bool_array(
            expected.values, skipna=True
        ):
            assert result.dtype == "boolean"
            expected = expected.astype("boolean")

        elif expected.dtype == "bool":
            assert result.dtype == "boolean"
            result = result.astype("bool")

        elif expected.dtype == "float" and expected.isna().any():
            assert result.dtype == "Int64"
            result = result.astype("float")

        if expected.dtype == object:
            # GH#18463
            expected[expected.isna()] = NA

    elif isinstance(expected, DataFrame):
        columns = expected.select_dtypes(include="object").columns
        assert all(result[columns].dtypes == nullable_string_dtype)
        result[columns] = result[columns].astype(object)
        with option_context("future.no_silent_downcasting", True):
            expected[columns] = expected[columns].fillna(NA)  # GH#18463

    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("count", [2, None]),
        ("find", [0, None]),
        ("index", [0, None]),
        ("rindex", [2, None]),
    ],
)
def test_string_array_numeric_integer_array(nullable_string_dtype, method, expected):
    s = Series(["aba", None], dtype=nullable_string_dtype)
    result = getattr(s.str, method)("a")
    expected = Series(expected, dtype="Int64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("isdigit", [False, None, True]),
        ("isalpha", [True, None, False]),
        ("isalnum", [True, None, True]),
        ("isnumeric", [False, None, True]),
    ],
)
def test_string_array_boolean_array(nullable_string_dtype, method, expected):
    s = Series(["a", None, "1"], dtype=nullable_string_dtype)
    result = getattr(s.str, method)()
    expected = Series(expected, dtype="boolean")
    tm.assert_series_equal(result, expected)


def test_string_array_extract(nullable_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/30969
    # Only expand=False & multiple groups was failing

    a = Series(["a1", "b2", "cc"], dtype=nullable_string_dtype)
    b = Series(["a1", "b2", "cc"], dtype="object")
    pat = r"(\w)(\d)"

    result = a.str.extract(pat, expand=False)
    expected = b.str.extract(pat, expand=False)
    expected = expected.fillna(NA)  # GH#18463
    assert all(result.dtypes == nullable_string_dtype)

    result = result.astype(object)
    tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_cython_aggregations.py ===
from functools import partial
import sys

import numpy as np
import pytest

import pandas._libs.window.aggregations as window_aggregations

from pandas import Series
import pandas._testing as tm


def _get_rolling_aggregations():
    # list pairs of name and function
    # each function has this signature:
    # (const float64_t[:] values, ndarray[int64_t] start,
    #  ndarray[int64_t] end, int64_t minp) -> np.ndarray
    named_roll_aggs = (
        [
            ("roll_sum", window_aggregations.roll_sum),
            ("roll_mean", window_aggregations.roll_mean),
        ]
        + [
            (f"roll_var({ddof})", partial(window_aggregations.roll_var, ddof=ddof))
            for ddof in [0, 1]
        ]
        + [
            ("roll_skew", window_aggregations.roll_skew),
            ("roll_kurt", window_aggregations.roll_kurt),
            ("roll_median_c", window_aggregations.roll_median_c),
            ("roll_max", window_aggregations.roll_max),
            ("roll_min", window_aggregations.roll_min),
        ]
        + [
            (
                f"roll_quantile({quantile},{interpolation})",
                partial(
                    window_aggregations.roll_quantile,
                    quantile=quantile,
                    interpolation=interpolation,
                ),
            )
            for quantile in [0.0001, 0.5, 0.9999]
            for interpolation in window_aggregations.interpolation_types
        ]
        + [
            (
                f"roll_rank({percentile},{method},{ascending})",
                partial(
                    window_aggregations.roll_rank,
                    percentile=percentile,
                    method=method,
                    ascending=ascending,
                ),
            )
            for percentile in [True, False]
            for method in window_aggregations.rolling_rank_tiebreakers.keys()
            for ascending in [True, False]
        ]
    )
    # unzip to a list of 2 tuples, names and functions
    unzipped = list(zip(*named_roll_aggs))
    return {"ids": unzipped[0], "params": unzipped[1]}


_rolling_aggregations = _get_rolling_aggregations()


@pytest.fixture(
    params=_rolling_aggregations["params"], ids=_rolling_aggregations["ids"]
)
def rolling_aggregation(request):
    """Make a rolling aggregation function as fixture."""
    return request.param


def test_rolling_aggregation_boundary_consistency(rolling_aggregation):
    # GH-45647
    minp, step, width, size, selection = 0, 1, 3, 11, [2, 7]
    values = np.arange(1, 1 + size, dtype=np.float64)
    end = np.arange(width, size, step, dtype=np.int64)
    start = end - width
    selarr = np.array(selection, dtype=np.int32)
    result = Series(rolling_aggregation(values, start[selarr], end[selarr], minp))
    expected = Series(rolling_aggregation(values, start, end, minp)[selarr])
    tm.assert_equal(expected, result)


def test_rolling_aggregation_with_unused_elements(rolling_aggregation):
    # GH-45647
    minp, width = 0, 5  # width at least 4 for kurt
    size = 2 * width + 5
    values = np.arange(1, size + 1, dtype=np.float64)
    values[width : width + 2] = sys.float_info.min
    values[width + 2] = np.nan
    values[width + 3 : width + 5] = sys.float_info.max
    start = np.array([0, size - width], dtype=np.int64)
    end = np.array([width, size], dtype=np.int64)
    loc = np.array(
        [j for i in range(len(start)) for j in range(start[i], end[i])],
        dtype=np.int32,
    )
    result = Series(rolling_aggregation(values, start, end, minp))
    compact_values = np.array(values[loc], dtype=np.float64)
    compact_start = np.arange(0, len(start) * width, width, dtype=np.int64)
    compact_end = compact_start + width
    expected = Series(
        rolling_aggregation(compact_values, compact_start, compact_end, minp)
    )
    assert np.isfinite(expected.values).all(), "Not all expected values are finite"
    tm.assert_equal(expected, result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_normalforms.py ===
from sympy.testing.pytest import warns_deprecated_sympy

from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly
from sympy.matrices import Matrix, randMatrix
from sympy.matrices.normalforms import (
    invariant_factors,
    smith_normal_form,
    smith_normal_decomp,
    hermite_normal_form,
    is_smith_normal_form,
)
from sympy.polys.domains import ZZ, QQ
from sympy.core.numbers import Integer

import random


def test_smith_normal():
    m = Matrix([[12,6,4,8],[3,9,6,12],[2,16,14,28],[20,10,10,20]])
    smf = Matrix([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, 30, 0], [0, 0, 0, 0]])
    assert smith_normal_form(m) == smf

    a, s, t = smith_normal_decomp(m)
    assert a == s * m * t

    x = Symbol('x')
    with warns_deprecated_sympy():
        m = Matrix([[Poly(x-1), Poly(1, x),Poly(-1,x)],
                    [0, Poly(x), Poly(-1,x)],
                    [Poly(0,x),Poly(-1,x),Poly(x)]])
    invs = 1, x - 1, x**2 - 1
    assert invariant_factors(m, domain=QQ[x]) == invs

    m = Matrix([[2, 4]])
    smf = Matrix([[2, 0]])
    assert smith_normal_form(m) == smf

    prng = random.Random(0)
    for i in range(6):
        for j in range(6):
            for _ in range(10 if i*j else 1):
                m = randMatrix(i, j, max=5, percent=50, prng=prng)
                a, s, t = smith_normal_decomp(m)
                assert a == s * m * t
                assert is_smith_normal_form(a)
                s.inv().to_DM(ZZ)
                t.inv().to_DM(ZZ)

                a, s, t = smith_normal_decomp(m, QQ)
                assert a == s * m * t
                assert is_smith_normal_form(a)
                s.inv()
                t.inv()


def test_smith_normal_deprecated():
    from sympy.polys.solvers import RawMatrix as Matrix

    with warns_deprecated_sympy():
        m = Matrix([[12, 6, 4,8],[3,9,6,12],[2,16,14,28],[20,10,10,20]])
    setattr(m, 'ring', ZZ)
    with warns_deprecated_sympy():
        smf = Matrix([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, 30, 0], [0, 0, 0, 0]])
    assert smith_normal_form(m) == smf

    x = Symbol('x')
    with warns_deprecated_sympy():
        m = Matrix([[Poly(x-1), Poly(1, x),Poly(-1,x)],
                    [0, Poly(x), Poly(-1,x)],
                    [Poly(0,x),Poly(-1,x),Poly(x)]])
    setattr(m, 'ring', QQ[x])
    invs = (Poly(1, x, domain='QQ'), Poly(x - 1, domain='QQ'), Poly(x**2 - 1, domain='QQ'))
    assert invariant_factors(m) == invs

    with warns_deprecated_sympy():
        m = Matrix([[2, 4]])
    setattr(m, 'ring', ZZ)
    with warns_deprecated_sympy():
        smf = Matrix([[2, 0]])
    assert smith_normal_form(m) == smf


def test_hermite_normal():
    m = Matrix([[2, 7, 17, 29, 41], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]])
    hnf = Matrix([[1, 0, 0], [0, 2, 1], [0, 0, 1]])
    assert hermite_normal_form(m) == hnf

    tr_hnf = Matrix([[37, 0, 19], [222, -6, 113], [48, 0, 25], [0, 2, 1], [0, 0, 1]])
    assert hermite_normal_form(m.transpose()) == tr_hnf

    m = Matrix([[8, 28, 68, 116, 164], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]])
    hnf = Matrix([[4, 0, 0], [0, 2, 1], [0, 0, 1]])
    assert hermite_normal_form(m) == hnf
    assert hermite_normal_form(m, D=8) == hnf
    assert hermite_normal_form(m, D=ZZ(8)) == hnf
    assert hermite_normal_form(m, D=Integer(8)) == hnf

    m = Matrix([[10, 8, 6, 30, 2], [45, 36, 27, 18, 9], [5, 4, 3, 2, 1]])
    hnf = Matrix([[26, 2], [0, 9], [0, 1]])
    assert hermite_normal_form(m) == hnf

    m = Matrix([[2, 7], [0, 0], [0, 0]])
    hnf = Matrix([[1], [0], [0]])
    assert hermite_normal_form(m) == hnf


def test_issue_23410():
    A = Matrix([[1, 12], [0, 8], [0, 5]])
    H = Matrix([[1, 0], [0, 8], [0, 5]])
    assert hermite_normal_form(A) == H

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tslibs\test_strptime.py ===
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.strptime import array_strptime

from pandas import (
    NaT,
    Timestamp,
)
import pandas._testing as tm

creso_infer = NpyDatetimeUnit.NPY_FR_GENERIC.value


class TestArrayStrptimeResolutionInference:
    def test_array_strptime_resolution_all_nat(self):
        arr = np.array([NaT, np.nan], dtype=object)

        fmt = "%Y-%m-%d %H:%M:%S"
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        assert res.dtype == "M8[s]"

        res, _ = array_strptime(arr, fmt=fmt, utc=True, creso=creso_infer)
        assert res.dtype == "M8[s]"

    @pytest.mark.parametrize("tz", [None, timezone.utc])
    def test_array_strptime_resolution_inference_homogeneous_strings(self, tz):
        dt = datetime(2016, 1, 2, 3, 4, 5, 678900, tzinfo=tz)

        fmt = "%Y-%m-%d %H:%M:%S"
        dtstr = dt.strftime(fmt)
        arr = np.array([dtstr] * 3, dtype=object)
        expected = np.array([dt.replace(tzinfo=None)] * 3, dtype="M8[s]")

        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        tm.assert_numpy_array_equal(res, expected)

        fmt = "%Y-%m-%d %H:%M:%S.%f"
        dtstr = dt.strftime(fmt)
        arr = np.array([dtstr] * 3, dtype=object)
        expected = np.array([dt.replace(tzinfo=None)] * 3, dtype="M8[us]")

        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        tm.assert_numpy_array_equal(res, expected)

        fmt = "ISO8601"
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize("tz", [None, timezone.utc])
    def test_array_strptime_resolution_mixed(self, tz):
        dt = datetime(2016, 1, 2, 3, 4, 5, 678900, tzinfo=tz)

        ts = Timestamp(dt).as_unit("ns")

        arr = np.array([dt, ts], dtype=object)
        expected = np.array(
            [Timestamp(dt).as_unit("ns").asm8, ts.asm8],
            dtype="M8[ns]",
        )

        fmt = "%Y-%m-%d %H:%M:%S"
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        tm.assert_numpy_array_equal(res, expected)

        fmt = "ISO8601"
        res, _ = array_strptime(arr, fmt=fmt, utc=False, creso=creso_infer)
        tm.assert_numpy_array_equal(res, expected)

    def test_array_strptime_resolution_todaynow(self):
        # specifically case where today/now is the *first* item
        vals = np.array(["today", np.datetime64("2017-01-01", "us")], dtype=object)

        now = Timestamp("now").asm8
        res, _ = array_strptime(vals, fmt="%Y-%m-%d", utc=False, creso=creso_infer)
        res2, _ = array_strptime(
            vals[::-1], fmt="%Y-%m-%d", utc=False, creso=creso_infer
        )

        # 1s is an arbitrary cutoff for call overhead; in local testing the
        #  actual difference is about 250us
        tolerance = np.timedelta64(1, "s")

        assert res.dtype == "M8[us]"
        assert abs(res[0] - now) < tolerance
        assert res[1] == vals[1]

        assert res2.dtype == "M8[us]"
        assert abs(res2[1] - now) < tolerance * 2
        assert res2[0] == vals[1]

    def test_array_strptime_str_outside_nano_range(self):
        vals = np.array(["2401-09-15"], dtype=object)
        expected = np.array(["2401-09-15"], dtype="M8[s]")
        fmt = "ISO8601"
        res, _ = array_strptime(vals, fmt=fmt, creso=creso_infer)
        tm.assert_numpy_array_equal(res, expected)

        # non-iso -> different path
        vals2 = np.array(["Sep 15, 2401"], dtype=object)
        expected2 = np.array(["2401-09-15"], dtype="M8[s]")
        fmt2 = "%b %d, %Y"
        res2, _ = array_strptime(vals2, fmt=fmt2, creso=creso_infer)
        tm.assert_numpy_array_equal(res2, expected2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_group_numbers.py ===
from sympy.combinatorics.group_numbers import (is_nilpotent_number,
    is_abelian_number, is_cyclic_number, _holder_formula, groups_count)
from sympy.ntheory.factor_ import factorint
from sympy.ntheory.generate import prime
from sympy.testing.pytest import raises
from sympy import randprime


def test_is_nilpotent_number():
    assert is_nilpotent_number(21) == False
    assert is_nilpotent_number(randprime(1, 30)**12) == True
    raises(ValueError, lambda: is_nilpotent_number(-5))

    A056867	= [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19,
               23, 25, 27, 29, 31, 32, 33, 35, 37, 41, 43, 45,
               47, 49, 51, 53, 59, 61, 64, 65, 67, 69, 71, 73,
               77, 79, 81, 83, 85, 87, 89, 91, 95, 97, 99]
    for n in range(1, 100):
        assert is_nilpotent_number(n) == (n in A056867)


def test_is_abelian_number():
    assert is_abelian_number(4) == True
    assert is_abelian_number(randprime(1, 2000)**2) == True
    assert is_abelian_number(randprime(1000, 100000)) == True
    assert is_abelian_number(60) == False
    assert is_abelian_number(24) == False
    raises(ValueError, lambda: is_abelian_number(-5))

    A051532 = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 23, 25,
               29, 31, 33, 35, 37, 41, 43, 45, 47, 49, 51, 53,
               59, 61, 65, 67, 69, 71, 73, 77, 79, 83, 85, 87,
               89, 91, 95, 97, 99]
    for n in range(1, 100):
        assert is_abelian_number(n) == (n in A051532)


A003277 = [1, 2, 3, 5, 7, 11, 13, 15, 17, 19, 23, 29,
           31, 33, 35, 37, 41, 43, 47, 51, 53, 59, 61,
           65, 67, 69, 71, 73, 77, 79, 83, 85, 87, 89,
           91, 95, 97]


def test_is_cyclic_number():
    assert is_cyclic_number(15) == True
    assert is_cyclic_number(randprime(1, 2000)**2) == False
    assert is_cyclic_number(randprime(1000, 100000)) == True
    assert is_cyclic_number(4) == False
    raises(ValueError, lambda: is_cyclic_number(-5))

    for n in range(1, 100):
        assert is_cyclic_number(n) == (n in A003277)


def test_holder_formula():
    # semiprime
    assert _holder_formula({3, 5}) == 1
    assert _holder_formula({5, 11}) == 2
    # n in A003277 is always 1
    for n in A003277:
        assert _holder_formula(set(factorint(n).keys())) == 1
    # otherwise
    assert _holder_formula({2, 3, 5, 7}) == 12


def test_groups_count():
    A000001 = [0, 1, 1, 1, 2, 1, 2, 1, 5, 2, 2, 1, 5, 1,
               2, 1, 14, 1, 5, 1, 5, 2, 2, 1, 15, 2, 2,
               5, 4, 1, 4, 1, 51, 1, 2, 1, 14, 1, 2, 2,
               14, 1, 6, 1, 4, 2, 2, 1, 52, 2, 5, 1, 5,
               1, 15, 2, 13, 2, 2, 1, 13, 1, 2, 4, 267,
               1, 4, 1, 5, 1, 4, 1, 50, 1, 2, 3, 4, 1,
               6, 1, 52, 15, 2, 1, 15, 1, 2, 1, 12, 1,
               10, 1, 4, 2]
    for n in range(1, len(A000001)):
        try:
            assert groups_count(n) == A000001[n]
        except ValueError:
            pass

    A000679 = [1, 1, 2, 5, 14, 51, 267, 2328, 56092, 10494213, 49487367289]
    for e in range(1, len(A000679)):
        assert groups_count(2**e) == A000679[e]

    A090091 = [1, 1, 2, 5, 15, 67, 504, 9310, 1396077, 5937876645]
    for e in range(1, len(A090091)):
        assert groups_count(3**e) == A090091[e]

    A090130 = [1, 1, 2, 5, 15, 77, 684, 34297]
    for e in range(1, len(A090130)):
        assert groups_count(5**e) == A090130[e]

    A090140 = [1, 1, 2, 5, 15, 83, 860, 113147]
    for e in range(1, len(A090140)):
        assert groups_count(7**e) == A090140[e]

    A232105 = [51, 67, 77, 83, 87, 97, 101, 107, 111, 125, 131,
               145, 149, 155, 159, 173, 183, 193, 203, 207, 217]
    for i in range(len(A232105)):
        assert groups_count(prime(i+1)**5) == A232105[i]

    A232106 = [267, 504, 684, 860, 1192, 1476, 1944, 2264, 2876,
               4068, 4540, 6012, 7064, 7664, 8852, 10908, 13136]
    for i in range(len(A232106)):
        assert groups_count(prime(i+1)**6) == A232106[i]

    A232107 = [2328, 9310, 34297, 113147, 750735, 1600573,
               5546909, 9380741, 23316851, 71271069, 98488755]
    for i in range(len(A232107)):
        assert groups_count(prime(i+1)**7) == A232107[i]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_operations.py ===
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.operations import AssocOp, LatticeOp
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
from sympy.core.add import Add, add
from sympy.core.mul import Mul, mul

# create the simplest possible Lattice class


class join(LatticeOp):
    zero = Integer(0)
    identity = Integer(1)


def test_lattice_simple():
    assert join(join(2, 3), 4) == join(2, join(3, 4))
    assert join(2, 3) == join(3, 2)
    assert join(0, 2) == 0
    assert join(1, 2) == 2
    assert join(2, 2) == 2

    assert join(join(2, 3), 4) == join(2, 3, 4)
    assert join() == 1
    assert join(4) == 4
    assert join(1, 4, 2, 3, 1, 3, 2) == join(2, 3, 4)


def test_lattice_shortcircuit():
    raises(SympifyError, lambda: join(object))
    assert join(0, object) == 0


def test_lattice_print():
    assert str(join(5, 4, 3, 2)) == 'join(2, 3, 4, 5)'


def test_lattice_make_args():
    assert join.make_args(join(2, 3, 4)) == {S(2), S(3), S(4)}
    assert join.make_args(0) == {0}
    assert list(join.make_args(0))[0] is S.Zero
    assert Add.make_args(0)[0] is S.Zero


def test_issue_14025():
    a, b, c, d = symbols('a,b,c,d', commutative=False)
    assert Mul(a, b, c).has(c*b) == False
    assert Mul(a, b, c).has(b*c) == True
    assert Mul(a, b, c, d).has(b*c*d) == True


def test_AssocOp_flatten():
    a, b, c, d = symbols('a,b,c,d')

    class MyAssoc(AssocOp):
        identity = S.One

    assert MyAssoc(a, MyAssoc(b, c)).args == \
        MyAssoc(MyAssoc(a, b), c).args == \
        MyAssoc(MyAssoc(a, b, c)).args == \
        MyAssoc(a, b, c).args == \
            (a, b, c)
    u = MyAssoc(b, c)
    v = MyAssoc(u, d, evaluate=False)
    assert v.args == (u, d)
    # like Add, any unevaluated outer call will flatten inner args
    assert MyAssoc(a, v).args == (a, b, c, d)


def test_add_dispatcher():

    class NewBase(Expr):
        @property
        def _add_handler(self):
            return NewAdd
    class NewAdd(NewBase, Add):
        pass
    add.register_handlerclass((Add, NewAdd), NewAdd)

    a, b = Symbol('a'), NewBase()

    # Add called as fallback
    assert add(1, 2) == Add(1, 2)
    assert add(a, a) == Add(a, a)

    # selection by registered priority
    assert add(a,b,a) == NewAdd(2*a, b)


def test_mul_dispatcher():

    class NewBase(Expr):
        @property
        def _mul_handler(self):
            return NewMul
    class NewMul(NewBase, Mul):
        pass
    mul.register_handlerclass((Mul, NewMul), NewMul)

    a, b = Symbol('a'), NewBase()

    # Mul called as fallback
    assert mul(1, 2) == Mul(1, 2)
    assert mul(a, a) == Mul(a, a)

    # selection by registered priority
    assert mul(a,b,a) == NewMul(a**2, b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_qs.py ===
from __future__ import annotations

import math
from sympy.core.random import _randint
from sympy.ntheory import qs, qs_factor
from sympy.ntheory.qs import SievePolynomial, _generate_factor_base, \
    _generate_polynomial, \
    _gen_sieve_array, _check_smoothness, _trial_division_stage, _find_factor
from sympy.testing.pytest import slow


@slow
def test_qs_1():
    assert qs(10009202107, 100, 10000) == {100043, 100049}
    assert qs(211107295182713951054568361, 1000, 10000) == \
        {13791315212531, 15307263442931}
    assert qs(980835832582657*990377764891511, 2000, 10000) == \
        {980835832582657, 990377764891511}
    assert qs(18640889198609*20991129234731, 1000, 50000) == \
        {18640889198609, 20991129234731}


def test_qs_2() -> None:
    n = 10009202107
    M = 50
    sieve_poly = SievePolynomial(10, 80, n)
    assert sieve_poly.eval_v(10) == sieve_poly.eval_u(10)**2 - n == -10009169707
    assert sieve_poly.eval_v(5) == sieve_poly.eval_u(5)**2 - n == -10009185207

    idx_1000, idx_5000, factor_base = _generate_factor_base(2000, n)
    assert idx_1000 == 82
    assert [factor_base[i].prime for i in range(15)] == \
        [2, 3, 7, 11, 17, 19, 29, 31, 43, 59, 61, 67, 71, 73, 79]
    assert [factor_base[i].tmem_p for i in range(15)] == \
        [1, 1, 3, 5, 3, 6, 6, 14, 1, 16, 24, 22, 18, 22, 15]
    assert [factor_base[i].log_p for i in range(5)] == \
        [710, 1125, 1993, 2455, 2901]

    it = _generate_polynomial(
        n, M, factor_base, idx_1000, idx_5000, _randint(0))
    g = next(it)
    assert g.a == 1133107
    assert g.b == 682543
    assert [factor_base[i].soln1 for i in range(15)] == \
        [0, 0, 3, 7, 13, 0, 8, 19, 9, 43, 27, 25, 63, 29, 19]
    assert [factor_base[i].soln2 for i in range(15)] == \
        [0, 1, 1, 3, 12, 16, 15, 6, 15, 1, 56, 55, 61, 58, 16]
    assert [factor_base[i].b_ainv for i in range(5)] == \
        [[0, 0], [0, 2], [3, 0], [3, 9], [13, 13]]

    g_1 = next(it)
    assert g_1.a == 1133107
    assert g_1.b == 136765

    sieve_array = _gen_sieve_array(M, factor_base)
    assert sieve_array[0:5] == [8424, 13603, 1835, 5335, 710]

    assert _check_smoothness(9645, factor_base) == (36028797018963972, 5)
    assert _check_smoothness(210313, factor_base) == (20992, 1)

    partial_relations: dict[int, tuple[int, int]] = {}
    smooth_relation, proper_factor = _trial_division_stage(
        n, M, factor_base, sieve_array, sieve_poly, partial_relations,
        ERROR_TERM=25*2**10)

    assert partial_relations == {
        8699: (440, -10009008507, 75557863761098695507973),
        166741: (490, -10008962007, 524341),
        131449: (530, -10008921207, 664613997892457936451903530140172325),
        6653: (550, -10008899607, 19342813113834066795307021)
    }
    assert [smooth_relation[i][0] for i in range(5)] == [
        -250, 1064469, 72819, 231957, 44167]
    assert [smooth_relation[i][1] for i in range(5)] == [
        -10009139607, 1133094251961, 5302606761, 53804049849, 1950723889]
    assert smooth_relation[0][2] == 89213869829863962596973701078031812362502145
    assert proper_factor == set()


def test_qs_3():
    N = 1817
    smooth_relations = [
        (2455024, 637, 8),
        (-27993000, 81536, 10),
        (11461840, 12544, 0),
        (149, 20384, 10),
        (-31138074, 19208, 2)
    ]
    assert next(_find_factor(N, smooth_relations, 4)) == 23


def test_qs_4():
    N = 10007**2 * 10009 * 10037**3 * 10039
    for factor in qs(N, 1000, 2000):
        assert N % factor == 0
        N //= factor


def test_qs_factor():
    assert qs_factor(1009 * 100003, 2000, 10000) == {1009: 1, 100003: 1}
    n = 1009**2 * 2003**2*30011*400009
    factors = qs_factor(n, 2000, 10000)
    assert len(factors) > 1
    assert math.prod(p**e for p, e in factors.items()) == n


def test_issue_27616():
    #https://github.com/sympy/sympy/issues/27616
    N = 9804659461513846513 + 1
    assert qs(N, 5000, 20000) is not None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_hilbert.py ===
from sympy.physics.quantum.hilbert import (
    HilbertSpace, ComplexSpace, L2, FockSpace, TensorProductHilbertSpace,
    DirectSumHilbertSpace, TensorPowerHilbertSpace
)

from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.printing.repr import srepr
from sympy.printing.str import sstr
from sympy.sets.sets import Interval


def test_hilbert_space():
    hs = HilbertSpace()
    assert isinstance(hs, HilbertSpace)
    assert sstr(hs) == 'H'
    assert srepr(hs) == 'HilbertSpace()'


def test_complex_space():
    c1 = ComplexSpace(2)
    assert isinstance(c1, ComplexSpace)
    assert c1.dimension == 2
    assert sstr(c1) == 'C(2)'
    assert srepr(c1) == 'ComplexSpace(Integer(2))'

    n = Symbol('n')
    c2 = ComplexSpace(n)
    assert isinstance(c2, ComplexSpace)
    assert c2.dimension == n
    assert sstr(c2) == 'C(n)'
    assert srepr(c2) == "ComplexSpace(Symbol('n'))"
    assert c2.subs(n, 2) == ComplexSpace(2)


def test_L2():
    b1 = L2(Interval(-oo, 1))
    assert isinstance(b1, L2)
    assert b1.dimension is oo
    assert b1.interval == Interval(-oo, 1)

    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    b2 = L2(Interval(x, y))
    assert b2.dimension is oo
    assert b2.interval == Interval(x, y)
    assert b2.subs(x, -1) == L2(Interval(-1, y))


def test_fock_space():
    f1 = FockSpace()
    f2 = FockSpace()
    assert isinstance(f1, FockSpace)
    assert f1.dimension is oo
    assert f1 == f2


def test_tensor_product():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)

    h = hs1*hs2
    assert isinstance(h, TensorProductHilbertSpace)
    assert h.dimension == 2*n
    assert h.spaces == (hs1, hs2)

    h = hs2*hs2
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs2
    assert h.exp == 2
    assert h.dimension == n**2

    f = FockSpace()
    h = hs1*hs2*f
    assert h.dimension is oo


def test_tensor_power():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)

    h = hs1**2
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs1
    assert h.exp == 2
    assert h.dimension == 4

    h = hs2**3
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs2
    assert h.exp == 3
    assert h.dimension == n**3


def test_direct_sum():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)

    h = hs1 + hs2
    assert isinstance(h, DirectSumHilbertSpace)
    assert h.dimension == 2 + n
    assert h.spaces == (hs1, hs2)

    f = FockSpace()
    h = hs1 + f + hs2
    assert h.dimension is oo
    assert h.spaces == (hs1, f, hs2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\tests\test_utils.py ===
from pytest import raises
from sympy import (
    symbols, Expr, Tuple, Integer, cos, solveset, FiniteSet, ImageSet)
from sympy.plotting.utils import (
    _create_ranges, _plot_sympify, extract_solution)
from sympy.physics.mechanics import ReferenceFrame, Vector as MechVector
from sympy.vector import CoordSys3D, Vector


def test_plot_sympify():
    x, y = symbols("x, y")

    # argument is already sympified
    args = x + y
    r = _plot_sympify(args)
    assert r == args

    # one argument needs to be sympified
    args = (x + y, 1)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Integer)

    # string and dict should not be sympified
    args = (x + y, (x, 0, 1), "str", 1, {1: 1, 2: 2.0})
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 5
    assert isinstance(r[0], Expr)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[2], str)
    assert isinstance(r[3], Integer)
    assert isinstance(r[4], dict) and isinstance(r[4][1], int) and isinstance(r[4][2], float)

    # nested arguments containing strings
    args = ((x + y, (y, 0, 1), "a"), (x + 1, (x, 0, 1), "$f_{1}$"))
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], Tuple)
    assert isinstance(r[0][1], Tuple)
    assert isinstance(r[0][1][1], Integer)
    assert isinstance(r[0][2], str)
    assert isinstance(r[1], Tuple)
    assert isinstance(r[1][1], Tuple)
    assert isinstance(r[1][1][1], Integer)
    assert isinstance(r[1][2], str)

    # vectors from sympy.physics.vectors module are not sympified
    # vectors from sympy.vectors are sympified
    # in both cases, no error should be raised
    R = ReferenceFrame("R")
    v1 = 2 * R.x + R.y
    C = CoordSys3D("C")
    v2 = 2 * C.i + C.j
    args = (v1, v2)
    r = _plot_sympify(args)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(v1, MechVector)
    assert isinstance(v2, Vector)


def test_create_ranges():
    x, y = symbols("x, y")

    # user don't provide any range -> return a default range
    r = _create_ranges({x}, [], 1)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 1
    assert isinstance(r[0], (Tuple, tuple))
    assert r[0] == (x, -10, 10)

    r = _create_ranges({x, y}, [], 2)
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, -10, 10) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, -10, 10)
    assert r[0] != r[1]

    # not enough ranges provided by the user -> create default ranges
    r = _create_ranges(
        {x, y},
        [
            (x, 0, 1),
        ],
        2,
    )
    assert isinstance(r, (list, tuple, Tuple)) and len(r) == 2
    assert isinstance(r[0], (Tuple, tuple))
    assert isinstance(r[1], (Tuple, tuple))
    assert r[0] == (x, 0, 1) or (y, -10, 10)
    assert r[1] == (y, -10, 10) or (x, 0, 1)
    assert r[0] != r[1]

    # too many free symbols
    raises(ValueError, lambda: _create_ranges({x, y}, [], 1))
    raises(ValueError, lambda: _create_ranges({x, y}, [(x, 0, 5), (y, 0, 1)], 1))


def test_extract_solution():
    x = symbols("x")

    sol = solveset(cos(10 * x))
    assert sol.has(ImageSet)
    res = extract_solution(sol)
    assert len(res) == 20
    assert isinstance(res, FiniteSet)

    res = extract_solution(sol, 20)
    assert len(res) == 40
    assert isinstance(res, FiniteSet)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_return_real.py ===
import platform

import pytest

from numpy import array
from numpy.testing import IS_64BIT

from . import util


@pytest.mark.slow
class TestReturnReal(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t4", "s0", "s4"]:
            err = 1e-5
        else:
            err = 0.0
        assert abs(t(234) - 234.0) <= err
        assert abs(t(234.6) - 234.6) <= err
        assert abs(t("234") - 234) <= err
        assert abs(t("234.6") - 234.6) <= err
        assert abs(t(-234) + 234) <= err
        assert abs(t([234]) - 234) <= err
        assert abs(t((234, )) - 234.0) <= err
        assert abs(t(array(234)) - 234.0) <= err
        assert abs(t(array(234).astype("b")) + 22) <= err
        assert abs(t(array(234, "h")) - 234.0) <= err
        assert abs(t(array(234, "i")) - 234.0) <= err
        assert abs(t(array(234, "l")) - 234.0) <= err
        assert abs(t(array(234, "B")) - 234.0) <= err
        assert abs(t(array(234, "f")) - 234.0) <= err
        assert abs(t(array(234, "d")) - 234.0) <= err
        if tname in ["t0", "t4", "s0", "s4"]:
            assert t(1e200) == t(1e300)  # inf

        # pytest.raises(ValueError, t, array([234], 'S1'))
        pytest.raises(ValueError, t, "abc")

        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        pytest.raises(Exception, t, t)
        pytest.raises(Exception, t, {})

        try:
            r = t(10**400)
            assert repr(r) in ["inf", "Infinity"]
        except OverflowError:
            pass


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Prone to error when run with numpy/f2py/tests on mac os, "
    "but not when run in isolation",
)
@pytest.mark.skipif(
    not IS_64BIT, reason="32-bit builds are buggy"
)
class TestCReturnReal(TestReturnReal):
    suffix = ".pyf"
    module_name = "c_ext_return_real"
    code = """
python module c_ext_return_real
usercode \'\'\'
float t4(float value) { return value; }
void s4(float *t4, float value) { *t4 = value; }
double t8(double value) { return value; }
void s8(double *t8, double value) { *t8 = value; }
\'\'\'
interface
  function t4(value)
    real*4 intent(c) :: t4,value
  end
  function t8(value)
    real*8 intent(c) :: t8,value
  end
  subroutine s4(t4,value)
    intent(c) s4
    real*4 intent(out) :: t4
    real*4 intent(c) :: value
  end
  subroutine s8(t8,value)
    intent(c) s8
    real*8 intent(out) :: t8
    real*8 intent(c) :: value
  end
end interface
end python module c_ext_return_real
    """

    @pytest.mark.parametrize("name", ["t4", "t8", "s4", "s8"])
    def test_all(self, name):
        self.check_function(getattr(self.module, name), name)


class TestFReturnReal(TestReturnReal):
    sources = [
        util.getpath("tests", "src", "return_real", "foo77.f"),
        util.getpath("tests", "src", "return_real", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name", ["t0", "t4", "t8", "td", "s0", "s4", "s8", "sd"])
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name", ["t0", "t4", "t8", "td", "s0", "s4", "s8", "sd"])
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_real, name), name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\test_formats.py ===
import pytest

from pandas import Timedelta


@pytest.mark.parametrize(
    "td, expected_repr",
    [
        (Timedelta(10, unit="d"), "Timedelta('10 days 00:00:00')"),
        (Timedelta(10, unit="s"), "Timedelta('0 days 00:00:10')"),
        (Timedelta(10, unit="ms"), "Timedelta('0 days 00:00:00.010000')"),
        (Timedelta(-10, unit="ms"), "Timedelta('-1 days +23:59:59.990000')"),
    ],
)
def test_repr(td, expected_repr):
    assert repr(td) == expected_repr


@pytest.mark.parametrize(
    "td, expected_iso",
    [
        (
            Timedelta(
                days=6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
            "P6DT0H50M3.010010012S",
        ),
        (Timedelta(days=4, hours=12, minutes=30, seconds=5), "P4DT12H30M5S"),
        (Timedelta(nanoseconds=123), "P0DT0H0M0.000000123S"),
        # trim nano
        (Timedelta(microseconds=10), "P0DT0H0M0.00001S"),
        # trim micro
        (Timedelta(milliseconds=1), "P0DT0H0M0.001S"),
        # don't strip every 0
        (Timedelta(minutes=1), "P0DT0H1M0S"),
    ],
)
def test_isoformat(td, expected_iso):
    assert td.isoformat() == expected_iso


class TestReprBase:
    def test_none(self):
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1s = Timedelta(1, unit="s")
        delta_500ms = Timedelta(500, unit="ms")

        drepr = lambda x: x._repr_base()
        assert drepr(delta_1d) == "1 days"
        assert drepr(-delta_1d) == "-1 days"
        assert drepr(delta_0d) == "0 days"
        assert drepr(delta_1s) == "0 days 00:00:01"
        assert drepr(delta_500ms) == "0 days 00:00:00.500000"
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"

    def test_sub_day(self):
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1s = Timedelta(1, unit="s")
        delta_500ms = Timedelta(500, unit="ms")

        drepr = lambda x: x._repr_base(format="sub_day")
        assert drepr(delta_1d) == "1 days"
        assert drepr(-delta_1d) == "-1 days"
        assert drepr(delta_0d) == "00:00:00"
        assert drepr(delta_1s) == "00:00:01"
        assert drepr(delta_500ms) == "00:00:00.500000"
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"

    def test_long(self):
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1s = Timedelta(1, unit="s")
        delta_500ms = Timedelta(500, unit="ms")

        drepr = lambda x: x._repr_base(format="long")
        assert drepr(delta_1d) == "1 days 00:00:00"
        assert drepr(-delta_1d) == "-1 days +00:00:00"
        assert drepr(delta_0d) == "0 days 00:00:00"
        assert drepr(delta_1s) == "0 days 00:00:01"
        assert drepr(delta_500ms) == "0 days 00:00:00.500000"
        assert drepr(delta_1d + delta_1s) == "1 days 00:00:01"
        assert drepr(-delta_1d + delta_1s) == "-1 days +00:00:01"
        assert drepr(delta_1d + delta_500ms) == "1 days 00:00:00.500000"
        assert drepr(-delta_1d + delta_500ms) == "-1 days +00:00:00.500000"

    def test_all(self):
        delta_1d = Timedelta(1, unit="D")
        delta_0d = Timedelta(0, unit="D")
        delta_1ns = Timedelta(1, unit="ns")

        drepr = lambda x: x._repr_base(format="all")
        assert drepr(delta_1d) == "1 days 00:00:00.000000000"
        assert drepr(-delta_1d) == "-1 days +00:00:00.000000000"
        assert drepr(delta_0d) == "0 days 00:00:00.000000000"
        assert drepr(delta_1ns) == "0 days 00:00:00.000000001"
        assert drepr(-delta_1d + delta_1ns) == "-1 days +00:00:00.000000001"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_subspaces.py ===
from sympy.matrices import Matrix
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.solvers import solve


def test_columnspace_one():
    m = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.columnspace()
    assert basis[0] == Matrix([1, -2, 0, 3])
    assert basis[1] == Matrix([2, -5, -3, 6])
    assert basis[2] == Matrix([2, -1, 4, -7])

    assert len(basis) == 3
    assert Matrix.hstack(m, *basis).columnspace() == basis


def test_rowspace():
    m = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.rowspace()
    assert basis[0] == Matrix([[1, 2, 0, 2, 5]])
    assert basis[1] == Matrix([[0, -1, 1, 3, 2]])
    assert basis[2] == Matrix([[0, 0, 0, 5, 5]])

    assert len(basis) == 3


def test_nullspace_one():
    m = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    basis = m.nullspace()
    assert basis[0] == Matrix([-2, 1, 1, 0, 0])
    assert basis[1] == Matrix([-1, -1, 0, -1, 1])
    # make sure the null space is really gets zeroed
    assert all(e.is_zero for e in m*basis[0])
    assert all(e.is_zero for e in m*basis[1])

def test_nullspace_second():
    # first test reduced row-ech form
    R = Rational

    M = Matrix([[5, 7, 2,  1],
                [1, 6, 2, -1]])
    out, tmp = M.rref()
    assert out == Matrix([[1, 0, -R(2)/23, R(13)/23],
                          [0, 1,  R(8)/23, R(-6)/23]])

    M = Matrix([[-5, -1,  4, -3, -1],
                [ 1, -1, -1,  1,  0],
                [-1,  0,  0,  0,  0],
                [ 4,  1, -4,  3,  1],
                [-2,  0,  2, -2, -1]])
    assert M*M.nullspace()[0] == Matrix(5, 1, [0]*5)

    M = Matrix([[ 1,  3, 0,  2,  6, 3, 1],
                [-2, -6, 0, -2, -8, 3, 1],
                [ 3,  9, 0,  0,  6, 6, 2],
                [-1, -3, 0,  1,  0, 9, 3]])
    out, tmp = M.rref()
    assert out == Matrix([[1, 3, 0, 0, 2, 0, 0],
                          [0, 0, 0, 1, 2, 0, 0],
                          [0, 0, 0, 0, 0, 1, R(1)/3],
                          [0, 0, 0, 0, 0, 0, 0]])

    # now check the vectors
    basis = M.nullspace()
    assert basis[0] == Matrix([-3, 1, 0, 0, 0, 0, 0])
    assert basis[1] == Matrix([0, 0, 1, 0, 0, 0, 0])
    assert basis[2] == Matrix([-2, 0, 0, -2, 1, 0, 0])
    assert basis[3] == Matrix([0, 0, 0, 0, 0, R(-1)/3, 1])

    # issue 4797; just see that we can do it when rows > cols
    M = Matrix([[1, 2], [2, 4], [3, 6]])
    assert M.nullspace()


def test_columnspace_second():
    M = Matrix([[ 1,  2,  0,  2,  5],
                [-2, -5,  1, -1, -8],
                [ 0, -3,  3,  4,  1],
                [ 3,  6,  0, -7,  2]])

    # now check the vectors
    basis = M.columnspace()
    assert basis[0] == Matrix([1, -2, 0, 3])
    assert basis[1] == Matrix([2, -5, -3, 6])
    assert basis[2] == Matrix([2, -1, 4, -7])

    #check by columnspace definition
    a, b, c, d, e = symbols('a b c d e')
    X = Matrix([a, b, c, d, e])
    for i in range(len(basis)):
        eq=M*X-basis[i]
        assert len(solve(eq, X)) != 0

    #check if rank-nullity theorem holds
    assert M.rank() == len(basis)
    assert len(M.nullspace()) + len(M.columnspace()) == M.cols

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_trace.py ===
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.trace import Tr
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_trace_new():
    a, b, c, d, Y = symbols('a b c d Y')
    A, B, C, D = symbols('A B C D', commutative=False)

    assert Tr(a + b) == a + b
    assert Tr(A + B) == Tr(A) + Tr(B)

    #check trace args not implicitly permuted
    assert Tr(C*D*A*B).args[0].args == (C, D, A, B)

    # check for mul and adds
    assert Tr((a*b) + ( c*d)) == (a*b) + (c*d)
    # Tr(scalar*A) = scalar*Tr(A)
    assert Tr(a*A) == a*Tr(A)
    assert Tr(a*A*B*b) == a*b*Tr(A*B)

    # since A is symbol and not commutative
    assert isinstance(Tr(A), Tr)

    #POW
    assert Tr(pow(a, b)) == a**b
    assert isinstance(Tr(pow(A, a)), Tr)

    #Matrix
    M = Matrix([[1, 1], [2, 2]])
    assert Tr(M) == 3

    ##test indices in different forms
    #no index
    t = Tr(A)
    assert t.args[1] == Tuple()

    #single index
    t = Tr(A, 0)
    assert t.args[1] == Tuple(0)

    #index in a list
    t = Tr(A, [0])
    assert t.args[1] == Tuple(0)

    t = Tr(A, [0, 1, 2])
    assert t.args[1] == Tuple(0, 1, 2)

    #index is tuple
    t = Tr(A, (0))
    assert t.args[1] == Tuple(0)

    t = Tr(A, (1, 2))
    assert t.args[1] == Tuple(1, 2)

    #trace indices test
    t = Tr((A + B), [2])
    assert t.args[0].args[1] == Tuple(2) and t.args[1].args[1] == Tuple(2)

    t = Tr(a*A, [2, 3])
    assert t.args[1].args[1] == Tuple(2, 3)

    #class with trace method defined
    #to simulate numpy objects
    class Foo:
        def trace(self):
            return 1
    assert Tr(Foo()) == 1

    #argument test
    # check for value error, when either/both arguments are not provided
    raises(ValueError, lambda: Tr())
    raises(ValueError, lambda: Tr(A, 1, 2))


def test_trace_doit():
    a, b, c, d = symbols('a b c d')
    A, B, C, D = symbols('A B C D', commutative=False)

    #TODO: needed while testing reduced density operations, etc.


def test_permute():
    A, B, C, D, E, F, G = symbols('A B C D E F G', commutative=False)
    t = Tr(A*B*C*D*E*F*G)

    assert t.permute(0).args[0].args == (A, B, C, D, E, F, G)
    assert t.permute(2).args[0].args == (F, G, A, B, C, D, E)
    assert t.permute(4).args[0].args == (D, E, F, G, A, B, C)
    assert t.permute(6).args[0].args == (B, C, D, E, F, G, A)
    assert t.permute(8).args[0].args == t.permute(1).args[0].args

    assert t.permute(-1).args[0].args == (B, C, D, E, F, G, A)
    assert t.permute(-3).args[0].args == (D, E, F, G, A, B, C)
    assert t.permute(-5).args[0].args == (F, G, A, B, C, D, E)
    assert t.permute(-8).args[0].args == t.permute(-1).args[0].args

    t = Tr((A + B)*(B*B)*C*D)
    assert t.permute(2).args[0].args == (C, D, (A + B), (B**2))

    t1 = Tr(A*B)
    t2 = t1.permute(1)
    assert id(t1) != id(t2) and t1 == t2

def test_deprecated_core_trace():
    with warns_deprecated_sympy():
        from sympy.core.trace import Tr # noqa:F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\sampling\tests\test_sample_discrete_rv.py ===
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.stats import (
    Geometric,
    Poisson,
    Zeta,
    sample,
    Skellam,
    Logarithmic,
    NegativeBinomial,
    YuleSimon,
    DiscreteRV,
)
from sympy.testing.pytest import skip, raises, slow


def test_sample_numpy():
    distribs_numpy = [
        Geometric('G', 0.5),
        Poisson('P', 1),
        Zeta('Z', 2)
    ]
    size = 3
    numpy = import_module('numpy')
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        for X in distribs_numpy:
            samps = sample(X, size=size, library='numpy')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
               lambda: sample(Skellam('S', 1, 1), library='numpy'))
    raises(NotImplementedError,
           lambda: Skellam('S', 1, 1).pspace.distribution.sample(library='tensorflow'))


def test_sample_scipy():
    p = S(2)/3
    x = Symbol('x', integer=True, positive=True)
    pdf = p*(1 - p)**(x - 1) # pdf of Geometric Distribution
    distribs_scipy = [
        DiscreteRV(x, pdf, set=S.Naturals),
        Geometric('G', 0.5),
        Logarithmic('L', 0.5),
        NegativeBinomial('N', 5, 0.4),
        Poisson('P', 1),
        Skellam('S', 1, 1),
        YuleSimon('Y', 1),
        Zeta('Z', 2)
    ]
    size = 3
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size, library='scipy')
            samps2 = sample(X, size=(2, 2), library='scipy')
            for sam in samps:
                assert sam in X.pspace.domain.set
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


def test_sample_pymc():
    distribs_pymc = [
        Geometric('G', 0.5),
        Poisson('P', 1),
        NegativeBinomial('N', 5, 0.4)
    ]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
               lambda: sample(Skellam('S', 1, 1), library='pymc'))

@slow
def test_sample_discrete():
    X = Geometric('X', S.Half)
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests')
    assert sample(X) in X.pspace.domain.set
    samps = sample(X, size=2) # This takes long time if ran without scipy
    for samp in samps:
        assert samp in X.pspace.domain.set

    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0, s1, s2 = [], [], []
                s0 = sample(X, size=10, library=lib, seed=0)
                s1 = sample(X, size=10, library=lib, seed=0)
                s2 = sample(X, size=10, library=lib, seed=1)
                assert all(s0 == s1)
                assert not all(s1 == s2)
        except NotImplementedError:
            continue

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_graph.py ===
from sympy.combinatorics import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.matrices.expressions import (
    PermutationMatrix, BlockDiagMatrix, BlockMatrix)


def test_connected_components():
    a, b, c, d, e, f, g, h, i, j, k, l, m = symbols('a:m')

    M = Matrix([
        [a, 0, 0, 0, b, 0, 0, 0, 0, 0, c, 0, 0],
        [0, d, 0, 0, 0, e, 0, 0, 0, 0, 0, f, 0],
        [0, 0, g, 0, 0, 0, h, 0, 0, 0, 0, 0, i],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [m, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, m, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, m, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [j, 0, 0, 0, k, 0, 0, 1, 0, 0, l, 0, 0],
        [0, j, 0, 0, 0, k, 0, 0, 1, 0, 0, l, 0],
        [0, 0, j, 0, 0, 0, k, 0, 0, 1, 0, 0, l],
        [0, 0, 0, 0, d, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, d, 0, 0, 0, 0, 0, 1]])
    cc = M.connected_components()
    assert cc == [[0, 4, 7, 10], [1, 5, 8, 11], [2, 6, 9, 12], [3]]

    P, B = M.connected_components_decomposition()
    p = Permutation([0, 4, 7, 10, 1, 5, 8, 11, 2, 6, 9, 12, 3])
    assert P == PermutationMatrix(p)

    B0 = Matrix([
        [a, b, 0, c],
        [m, 1, 0, 0],
        [j, k, 1, l],
        [0, d, 0, 1]])
    B1 = Matrix([
        [d, e, 0, f],
        [m, 1, 0, 0],
        [j, k, 1, l],
        [0, d, 0, 1]])
    B2 = Matrix([
        [g, h, 0, i],
        [m, 1, 0, 0],
        [j, k, 1, l],
        [0, d, 0, 1]])
    B3 = Matrix([[1]])
    assert B == BlockDiagMatrix(B0, B1, B2, B3)


def test_strongly_connected_components():
    M = Matrix([
        [11, 14, 10, 0, 15, 0],
        [0, 44, 0, 0, 45, 0],
        [1, 4, 0, 0, 5, 0],
        [0, 0, 0, 22, 0, 23],
        [0, 54, 0, 0, 55, 0],
        [0, 0, 0, 32, 0, 33]])
    scc = M.strongly_connected_components()
    assert scc == [[1, 4], [0, 2], [3, 5]]

    P, B = M.strongly_connected_components_decomposition()
    p = Permutation([1, 4, 0, 2, 3, 5])
    assert P == PermutationMatrix(p)
    assert B == BlockMatrix([
        [
            Matrix([[44, 45], [54, 55]]),
            Matrix.zeros(2, 2),
            Matrix.zeros(2, 2)
        ],
        [
            Matrix([[14, 15], [4, 5]]),
            Matrix([[11, 10], [1, 0]]),
            Matrix.zeros(2, 2)
        ],
        [
            Matrix.zeros(2, 2),
            Matrix.zeros(2, 2),
            Matrix([[22, 23], [32, 33]])
        ]
    ])
    P = P.as_explicit()
    B = B.as_explicit()
    assert P.T * B * P == M

    P, B = M.strongly_connected_components_decomposition(lower=False)
    p = Permutation([3, 5, 0, 2, 1, 4])
    assert P == PermutationMatrix(p)
    assert B == BlockMatrix([
        [
            Matrix([[22, 23], [32, 33]]),
            Matrix.zeros(2, 2),
            Matrix.zeros(2, 2)
        ],
        [
            Matrix.zeros(2, 2),
            Matrix([[11, 10], [1, 0]]),
            Matrix([[14, 15], [4, 5]])
        ],
        [
            Matrix.zeros(2, 2),
            Matrix.zeros(2, 2),
            Matrix([[44, 45], [54, 55]])
        ]
    ])
    P = P.as_explicit()
    B = B.as_explicit()
    assert P.T * B * P == M

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_core_functionalities.py ===
import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


def test_assigning_to_same_variable_removes_references(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]})
    df = df.reset_index()
    if using_copy_on_write:
        assert df._mgr._has_no_reference(1)
    arr = get_array(df, "a")
    df.iloc[0, 1] = 100  # Write into a

    assert np.shares_memory(arr, get_array(df, "a"))


def test_setitem_dont_track_unnecessary_references(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})

    df["b"] = 100
    arr = get_array(df, "a")
    # We split the block in setitem, if we are not careful the new blocks will
    # reference each other triggering a copy
    df.iloc[0, 0] = 100
    assert np.shares_memory(arr, get_array(df, "a"))


def test_setitem_with_view_copies(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})
    view = df[:]
    expected = df.copy()

    df["b"] = 100
    arr = get_array(df, "a")
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100  # Check that we correctly track reference
    if using_copy_on_write:
        assert not np.shares_memory(arr, get_array(df, "a"))
        tm.assert_frame_equal(view, expected)


def test_setitem_with_view_invalidated_does_not_copy(
    using_copy_on_write, warn_copy_on_write, request
):
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})
    view = df[:]

    df["b"] = 100
    arr = get_array(df, "a")
    view = None  # noqa: F841
    # TODO(CoW-warn) false positive? -> block gets split because of `df["b"] = 100`
    # which introduces additional refs, even when those of `view` go out of scopes
    with tm.assert_cow_warning(warn_copy_on_write):
        df.iloc[0, 0] = 100
    if using_copy_on_write:
        # Setitem split the block. Since the old block shared data with view
        # all the new blocks are referencing view and each other. When view
        # goes out of scope, they don't share data with any other block,
        # so we should not trigger a copy
        mark = pytest.mark.xfail(
            reason="blk.delete does not track references correctly"
        )
        request.applymarker(mark)
        assert np.shares_memory(arr, get_array(df, "a"))


def test_out_of_scope(using_copy_on_write):
    def func():
        df = DataFrame({"a": [1, 2], "b": 1.5, "c": 1})
        # create some subset
        result = df[["a", "b"]]
        return result

    result = func()
    if using_copy_on_write:
        assert not result._mgr.blocks[0].refs.has_reference()
        assert not result._mgr.blocks[1].refs.has_reference()


def test_delete(using_copy_on_write):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["a", "b", "c"]
    )
    del df["b"]
    if using_copy_on_write:
        assert not df._mgr.blocks[0].refs.has_reference()
        assert not df._mgr.blocks[1].refs.has_reference()

    df = df[["a"]]
    if using_copy_on_write:
        assert not df._mgr.blocks[0].refs.has_reference()


def test_delete_reference(using_copy_on_write):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["a", "b", "c"]
    )
    x = df[:]
    del df["b"]
    if using_copy_on_write:
        assert df._mgr.blocks[0].refs.has_reference()
        assert df._mgr.blocks[1].refs.has_reference()
        assert x._mgr.blocks[0].refs.has_reference()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_formats.py ===
import pytest

import pandas as pd
from pandas import (
    Series,
    TimedeltaIndex,
)


class TestTimedeltaIndexRendering:
    def test_repr_round_days_non_nano(self):
        # GH#55405
        # we should get "1 days", not "1 days 00:00:00" with non-nano
        tdi = TimedeltaIndex(["1 days"], freq="D").as_unit("s")
        result = repr(tdi)
        expected = "TimedeltaIndex(['1 days'], dtype='timedelta64[s]', freq='D')"
        assert result == expected

        result2 = repr(Series(tdi))
        expected2 = "0   1 days\ndtype: timedelta64[s]"
        assert result2 == expected2

    @pytest.mark.parametrize("method", ["__repr__", "__str__"])
    def test_representation(self, method):
        idx1 = TimedeltaIndex([], freq="D")
        idx2 = TimedeltaIndex(["1 days"], freq="D")
        idx3 = TimedeltaIndex(["1 days", "2 days"], freq="D")
        idx4 = TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")
        idx5 = TimedeltaIndex(["1 days 00:00:01", "2 days", "3 days"])

        exp1 = "TimedeltaIndex([], dtype='timedelta64[ns]', freq='D')"

        exp2 = "TimedeltaIndex(['1 days'], dtype='timedelta64[ns]', freq='D')"

        exp3 = "TimedeltaIndex(['1 days', '2 days'], dtype='timedelta64[ns]', freq='D')"

        exp4 = (
            "TimedeltaIndex(['1 days', '2 days', '3 days'], "
            "dtype='timedelta64[ns]', freq='D')"
        )

        exp5 = (
            "TimedeltaIndex(['1 days 00:00:01', '2 days 00:00:00', "
            "'3 days 00:00:00'], dtype='timedelta64[ns]', freq=None)"
        )

        with pd.option_context("display.width", 300):
            for idx, expected in zip(
                [idx1, idx2, idx3, idx4, idx5], [exp1, exp2, exp3, exp4, exp5]
            ):
                result = getattr(idx, method)()
                assert result == expected

    # TODO: this is a Series.__repr__ test
    def test_representation_to_series(self):
        idx1 = TimedeltaIndex([], freq="D")
        idx2 = TimedeltaIndex(["1 days"], freq="D")
        idx3 = TimedeltaIndex(["1 days", "2 days"], freq="D")
        idx4 = TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")
        idx5 = TimedeltaIndex(["1 days 00:00:01", "2 days", "3 days"])

        exp1 = """Series([], dtype: timedelta64[ns])"""

        exp2 = "0   1 days\ndtype: timedelta64[ns]"

        exp3 = "0   1 days\n1   2 days\ndtype: timedelta64[ns]"

        exp4 = "0   1 days\n1   2 days\n2   3 days\ndtype: timedelta64[ns]"

        exp5 = (
            "0   1 days 00:00:01\n"
            "1   2 days 00:00:00\n"
            "2   3 days 00:00:00\n"
            "dtype: timedelta64[ns]"
        )

        with pd.option_context("display.width", 300):
            for idx, expected in zip(
                [idx1, idx2, idx3, idx4, idx5], [exp1, exp2, exp3, exp4, exp5]
            ):
                result = repr(Series(idx))
                assert result == expected

    def test_summary(self):
        # GH#9116
        idx1 = TimedeltaIndex([], freq="D")
        idx2 = TimedeltaIndex(["1 days"], freq="D")
        idx3 = TimedeltaIndex(["1 days", "2 days"], freq="D")
        idx4 = TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")
        idx5 = TimedeltaIndex(["1 days 00:00:01", "2 days", "3 days"])

        exp1 = "TimedeltaIndex: 0 entries\nFreq: D"

        exp2 = "TimedeltaIndex: 1 entries, 1 days to 1 days\nFreq: D"

        exp3 = "TimedeltaIndex: 2 entries, 1 days to 2 days\nFreq: D"

        exp4 = "TimedeltaIndex: 3 entries, 1 days to 3 days\nFreq: D"

        exp5 = "TimedeltaIndex: 3 entries, 1 days 00:00:01 to 3 days 00:00:00"

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5], [exp1, exp2, exp3, exp4, exp5]
        ):
            result = idx._summary()
            assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\excel\test_odswriter.py ===
from datetime import (
    date,
    datetime,
)
import re

import pytest

from pandas.compat import is_platform_windows

import pandas as pd
import pandas._testing as tm

from pandas.io.excel import ExcelWriter

odf = pytest.importorskip("odf")

if is_platform_windows():
    pytestmark = pytest.mark.single_cpu


@pytest.fixture
def ext():
    return ".ods"


def test_write_append_mode_raises(ext):
    msg = "Append mode is not supported with odf!"

    with tm.ensure_clean(ext) as f:
        with pytest.raises(ValueError, match=msg):
            ExcelWriter(f, engine="odf", mode="a")


@pytest.mark.parametrize("engine_kwargs", [None, {"kwarg": 1}])
def test_engine_kwargs(ext, engine_kwargs):
    # GH 42286
    # GH 43445
    # test for error: OpenDocumentSpreadsheet does not accept any arguments
    with tm.ensure_clean(ext) as f:
        if engine_kwargs is not None:
            error = re.escape(
                "OpenDocumentSpreadsheet() got an unexpected keyword argument 'kwarg'"
            )
            with pytest.raises(
                TypeError,
                match=error,
            ):
                ExcelWriter(f, engine="odf", engine_kwargs=engine_kwargs)
        else:
            with ExcelWriter(f, engine="odf", engine_kwargs=engine_kwargs) as _:
                pass


def test_book_and_sheets_consistent(ext):
    # GH#45687 - Ensure sheets is updated if user modifies book
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f) as writer:
            assert writer.sheets == {}
            table = odf.table.Table(name="test_name")
            writer.book.spreadsheet.addElement(table)
            assert writer.sheets == {"test_name": table}


@pytest.mark.parametrize(
    ["value", "cell_value_type", "cell_value_attribute", "cell_value"],
    argvalues=[
        (True, "boolean", "boolean-value", "true"),
        ("test string", "string", "string-value", "test string"),
        (1, "float", "value", "1"),
        (1.5, "float", "value", "1.5"),
        (
            datetime(2010, 10, 10, 10, 10, 10),
            "date",
            "date-value",
            "2010-10-10T10:10:10",
        ),
        (date(2010, 10, 10), "date", "date-value", "2010-10-10"),
    ],
)
def test_cell_value_type(ext, value, cell_value_type, cell_value_attribute, cell_value):
    # GH#54994 ODS: cell attributes should follow specification
    # http://docs.oasis-open.org/office/v1.2/os/OpenDocument-v1.2-os-part1.html#refTable13
    from odf.namespaces import OFFICENS
    from odf.table import (
        TableCell,
        TableRow,
    )

    table_cell_name = TableCell().qname

    with tm.ensure_clean(ext) as f:
        pd.DataFrame([[value]]).to_excel(f, header=False, index=False)

        with pd.ExcelFile(f) as wb:
            sheet = wb._reader.get_sheet_by_index(0)
            sheet_rows = sheet.getElementsByType(TableRow)
            sheet_cells = [
                x
                for x in sheet_rows[0].childNodes
                if hasattr(x, "qname") and x.qname == table_cell_name
            ]

            cell = sheet_cells[0]
            assert cell.attributes.get((OFFICENS, "value-type")) == cell_value_type
            assert cell.attributes.get((OFFICENS, cell_value_attribute)) == cell_value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_to_markdown.py ===
from io import (
    BytesIO,
    StringIO,
)

import pytest

import pandas as pd
import pandas._testing as tm

pytest.importorskip("tabulate")


def test_simple():
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


def test_empty_frame():
    buf = StringIO()
    df = pd.DataFrame({"id": [], "first_name": [], "last_name": []}).set_index("id")
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == (
        "| id   | first_name   | last_name   |\n"
        "|------|--------------|-------------|"
    )


def test_other_tablefmt():
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, tablefmt="jira")
    result = buf.getvalue()
    assert result == "||    ||   0 ||\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"


def test_other_headers():
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, headers=["foo", "bar"])
    result = buf.getvalue()
    assert result == (
        "|   foo |   bar |\n|------:|------:|\n|     0 "
        "|     1 |\n|     1 |     2 |\n|     2 |     3 |"
    )


def test_series():
    buf = StringIO()
    s = pd.Series([1, 2, 3], name="foo")
    s.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == (
        "|    |   foo |\n|---:|------:|\n|  0 |     1 "
        "|\n|  1 |     2 |\n|  2 |     3 |"
    )


def test_no_buf():
    df = pd.DataFrame([1, 2, 3])
    result = df.to_markdown()
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


@pytest.mark.parametrize("index", [True, False])
def test_index(index):
    # GH 32667

    df = pd.DataFrame([1, 2, 3])

    result = df.to_markdown(index=index)

    if index:
        expected = (
            "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
        )
    else:
        expected = "|   0 |\n|----:|\n|   1 |\n|   2 |\n|   3 |"
    assert result == expected


def test_showindex_disallowed_in_kwargs():
    # GH 32667; disallowing showindex in kwargs enforced in 2.0
    df = pd.DataFrame([1, 2, 3])
    with pytest.raises(ValueError, match="Pass 'index' instead of 'showindex"):
        df.to_markdown(index=True, showindex=True)


def test_markdown_pos_args_deprecatation():
    # GH-54229
    df = pd.DataFrame({"a": [1, 2, 3]})
    msg = (
        r"Starting with pandas version 3.0 all arguments of to_markdown except for the "
        r"argument 'buf' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buffer = BytesIO()
        df.to_markdown(buffer, "grid")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\tests\test_integrals.py ===
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.integrals import ParametricIntegral, vector_integrate
from sympy.vector.parametricregion import ParametricRegion
from sympy.vector.implicitregion import ImplicitRegion
from sympy.abc import x, y, z, u, v, r, t, theta, phi
from sympy.geometry import Point, Segment, Curve, Circle, Polygon, Plane

C = CoordSys3D('C')

def test_parametric_lineintegrals():
    halfcircle = ParametricRegion((4*cos(theta), 4*sin(theta)), (theta, -pi/2, pi/2))
    assert ParametricIntegral(C.x*C.y**4, halfcircle) == S(8192)/5

    curve = ParametricRegion((t, t**2, t**3), (t, 0, 1))
    field1 = 8*C.x**2*C.y*C.z*C.i + 5*C.z*C.j - 4*C.x*C.y*C.k
    assert ParametricIntegral(field1, curve) == 1
    line = ParametricRegion((4*t - 1, 2 - 2*t, t), (t, 0, 1))
    assert ParametricIntegral(C.x*C.z*C.i - C.y*C.z*C.k, line) == 3

    assert ParametricIntegral(4*C.x**3, ParametricRegion((1, t), (t, 0, 2))) == 8

    helix = ParametricRegion((cos(t), sin(t), 3*t), (t, 0, 4*pi))
    assert ParametricIntegral(C.x*C.y*C.z, helix) == -3*sqrt(10)*pi

    field2 = C.y*C.i + C.z*C.j + C.z*C.k
    assert ParametricIntegral(field2, ParametricRegion((cos(t), sin(t), t**2), (t, 0, pi))) == -5*pi/2 + pi**4/2

def test_parametric_surfaceintegrals():

    semisphere = ParametricRegion((2*sin(phi)*cos(theta), 2*sin(phi)*sin(theta), 2*cos(phi)),\
                            (theta, 0, 2*pi), (phi, 0, pi/2))
    assert ParametricIntegral(C.z, semisphere) == 8*pi

    cylinder = ParametricRegion((sqrt(3)*cos(theta), sqrt(3)*sin(theta), z), (z, 0, 6), (theta, 0, 2*pi))
    assert ParametricIntegral(C.y, cylinder) == 0

    cone = ParametricRegion((v*cos(u), v*sin(u), v), (u, 0, 2*pi), (v, 0, 1))
    assert ParametricIntegral(C.x*C.i + C.y*C.j + C.z**4*C.k, cone) == pi/3

    triangle1 = ParametricRegion((x, y), (x, 0, 2), (y, 0, 10 - 5*x))
    triangle2 = ParametricRegion((x, y), (y, 0, 10 - 5*x), (x, 0, 2))
    assert ParametricIntegral(-15.6*C.y*C.k, triangle1) == ParametricIntegral(-15.6*C.y*C.k, triangle2)
    assert ParametricIntegral(C.z, triangle1) == 10*C.z

def test_parametric_volumeintegrals():

    cube = ParametricRegion((x, y, z), (x, 0, 1), (y, 0, 1), (z, 0, 1))
    assert ParametricIntegral(1, cube) == 1

    solidsphere1 = ParametricRegion((r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi)),\
                            (r, 0, 2), (theta, 0, 2*pi), (phi, 0, pi))
    solidsphere2 = ParametricRegion((r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi)),\
                            (r, 0, 2), (phi, 0, pi), (theta, 0, 2*pi))
    assert ParametricIntegral(C.x**2 + C.y**2, solidsphere1) == -256*pi/15
    assert ParametricIntegral(C.x**2 + C.y**2, solidsphere2) == 256*pi/15

    region_under_plane1 = ParametricRegion((x, y, z), (x, 0, 3), (y, 0, -2*x/3 + 2),\
                                    (z, 0, 6 - 2*x - 3*y))
    region_under_plane2 = ParametricRegion((x, y, z), (x, 0, 3), (z, 0, 6 - 2*x - 3*y),\
                                    (y, 0, -2*x/3 + 2))

    assert ParametricIntegral(C.x*C.i + C.j - 100*C.k, region_under_plane1) == \
        ParametricIntegral(C.x*C.i + C.j - 100*C.k, region_under_plane2)
    assert ParametricIntegral(2*C.x, region_under_plane2) == -9

def test_vector_integrate():
    halfdisc = ParametricRegion((r*cos(theta), r* sin(theta)), (r, -2, 2), (theta, 0, pi))
    assert vector_integrate(C.x**2, halfdisc) == 4*pi
    assert vector_integrate(C.x, ParametricRegion((t, t**2), (t, 2, 3))) == -17*sqrt(17)/12 + 37*sqrt(37)/12

    assert  vector_integrate(C.y**3*C.z, (C.x, 0, 3), (C.y, -1, 4)) == 765*C.z/4

    s1 = Segment(Point(0, 0), Point(0, 1))
    assert vector_integrate(-15*C.y, s1) == S(-15)/2
    s2 = Segment(Point(4, 3, 9), Point(1, 1, 7))
    assert vector_integrate(C.y*C.i, s2) == -6

    curve = Curve((sin(t), cos(t)), (t, 0, 2))
    assert vector_integrate(5*C.z, curve) == 10*C.z

    c1 = Circle(Point(2, 3), 6)
    assert vector_integrate(C.x*C.y, c1) == 72*pi
    c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
    assert vector_integrate(1, c2) == c2.circumference

    triangle = Polygon((0, 0), (1, 0), (1, 1))
    assert vector_integrate(C.x*C.i - 14*C.y*C.j, triangle) == 0
    p1, p2, p3, p4 = [(0, 0), (1, 0), (5, 1), (0, 1)]
    poly = Polygon(p1, p2, p3, p4)
    assert vector_integrate(-23*C.z, poly) == -161*C.z - 23*sqrt(17)*C.z

    point = Point(2, 3)
    assert vector_integrate(C.i*C.y, point) == ParametricIntegral(C.y*C.i, ParametricRegion((2, 3)))

    c3 = ImplicitRegion((x, y), x**2 + y**2 - 4)
    assert vector_integrate(45, c3) == 180*pi
    c4 = ImplicitRegion((x, y), (x - 3)**2 + (y - 4)**2 - 9)
    assert vector_integrate(1, c4) == 6*pi

    pl = Plane(Point(1, 1, 1), Point(2, 3, 4), Point(2, 2, 2))
    raises(ValueError, lambda: vector_integrate(C.x*C.z*C.i + C.k, pl))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matrixlib\tests\test_matrix_linalg.py ===
""" Test functions for linalg module using the matrix class."""
import numpy as np
from numpy.linalg.tests.test_linalg import (
    CondCases,
    DetCases,
    EigCases,
    EigvalsCases,
    InvCases,
    LinalgCase,
    LinalgTestCase,
    LstsqCases,
    PinvCases,
    SolveCases,
    SVDCases,
    _TestNorm2D,
    _TestNormDoubleBase,
    _TestNormInt64Base,
    _TestNormSingleBase,
    apply_tag,
)
from numpy.linalg.tests.test_linalg import TestQR as _TestQR

CASES = []

# square test cases
CASES += apply_tag('square', [
    LinalgCase("0x0_matrix",
               np.empty((0, 0), dtype=np.double).view(np.matrix),
               np.empty((0, 1), dtype=np.double).view(np.matrix),
               tags={'size-0'}),
    LinalgCase("matrix_b_only",
               np.array([[1., 2.], [3., 4.]]),
               np.matrix([2., 1.]).T),
    LinalgCase("matrix_a_and_b",
               np.matrix([[1., 2.], [3., 4.]]),
               np.matrix([2., 1.]).T),
])

# hermitian test-cases
CASES += apply_tag('hermitian', [
    LinalgCase("hmatrix_a_and_b",
               np.matrix([[1., 2.], [2., 1.]]),
               None),
])
# No need to make generalized or strided cases for matrices.


class MatrixTestCase(LinalgTestCase):
    TEST_CASES = CASES


class TestSolveMatrix(SolveCases, MatrixTestCase):
    pass


class TestInvMatrix(InvCases, MatrixTestCase):
    pass


class TestEigvalsMatrix(EigvalsCases, MatrixTestCase):
    pass


class TestEigMatrix(EigCases, MatrixTestCase):
    pass


class TestSVDMatrix(SVDCases, MatrixTestCase):
    pass


class TestCondMatrix(CondCases, MatrixTestCase):
    pass


class TestPinvMatrix(PinvCases, MatrixTestCase):
    pass


class TestDetMatrix(DetCases, MatrixTestCase):
    pass


class TestLstsqMatrix(LstsqCases, MatrixTestCase):
    pass


class _TestNorm2DMatrix(_TestNorm2D):
    array = np.matrix


class TestNormDoubleMatrix(_TestNorm2DMatrix, _TestNormDoubleBase):
    pass


class TestNormSingleMatrix(_TestNorm2DMatrix, _TestNormSingleBase):
    pass


class TestNormInt64Matrix(_TestNorm2DMatrix, _TestNormInt64Base):
    pass


class TestQRMatrix(_TestQR):
    array = np.matrix

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_scalarinherit.py ===
""" Test printing of scalar types.

"""
import pytest

import numpy as np
from numpy.testing import assert_, assert_raises


class A:
    pass
class B(A, np.float64):
    pass

class C(B):
    pass
class D(C, B):
    pass

class B0(np.float64, A):
    pass
class C0(B0):
    pass

class HasNew:
    def __new__(cls, *args, **kwargs):
        return cls, args, kwargs

class B1(np.float64, HasNew):
    pass


class TestInherit:
    def test_init(self):
        x = B(1.0)
        assert_(str(x) == '1.0')
        y = C(2.0)
        assert_(str(y) == '2.0')
        z = D(3.0)
        assert_(str(z) == '3.0')

    def test_init2(self):
        x = B0(1.0)
        assert_(str(x) == '1.0')
        y = C0(2.0)
        assert_(str(y) == '2.0')

    def test_gh_15395(self):
        # HasNew is the second base, so `np.float64` should have priority
        x = B1(1.0)
        assert_(str(x) == '1.0')

        # previously caused RecursionError!?
        with pytest.raises(TypeError):
            B1(1.0, 2.0)

    def test_int_repr(self):
        # Test that integer repr works correctly for subclasses (gh-27106)
        class my_int16(np.int16):
            pass

        s = repr(my_int16(3))
        assert s == "my_int16(3)"

class TestCharacter:
    def test_char_radd(self):
        # GH issue 9620, reached gentype_add and raise TypeError
        np_s = np.bytes_('abc')
        np_u = np.str_('abc')
        s = b'def'
        u = 'def'
        assert_(np_s.__radd__(np_s) is NotImplemented)
        assert_(np_s.__radd__(np_u) is NotImplemented)
        assert_(np_s.__radd__(s) is NotImplemented)
        assert_(np_s.__radd__(u) is NotImplemented)
        assert_(np_u.__radd__(np_s) is NotImplemented)
        assert_(np_u.__radd__(np_u) is NotImplemented)
        assert_(np_u.__radd__(s) is NotImplemented)
        assert_(np_u.__radd__(u) is NotImplemented)
        assert_(s + np_s == b'defabc')
        assert_(u + np_u == 'defabc')

        class MyStr(str, np.generic):
            # would segfault
            pass

        with assert_raises(TypeError):
            # Previously worked, but gave completely wrong result
            ret = s + MyStr('abc')

        class MyBytes(bytes, np.generic):
            # would segfault
            pass

        ret = s + MyBytes(b'abc')
        assert type(ret) is type(s)
        assert ret == b"defabc"

    def test_char_repeat(self):
        np_s = np.bytes_('abc')
        np_u = np.str_('abc')
        res_s = b'abc' * 5
        res_u = 'abc' * 5
        assert_(np_s * 5 == res_s)
        assert_(np_u * 5 == res_u)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_common.py ===
import numpy as np
import pytest

from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray


class DummyDtype(dtypes.ExtensionDtype):
    pass


class DummyArray(ExtensionArray):
    def __init__(self, data) -> None:
        self.data = data

    def __array__(self, dtype=None, copy=None):
        return self.data

    @property
    def dtype(self):
        return DummyDtype()

    def astype(self, dtype, copy=True):
        # we don't support anything but a single dtype
        if isinstance(dtype, DummyDtype):
            if copy:
                return type(self)(self.data)
            return self
        elif not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)


class TestExtensionArrayDtype:
    @pytest.mark.parametrize(
        "values",
        [
            pd.Categorical([]),
            pd.Categorical([]).dtype,
            pd.Series(pd.Categorical([])),
            DummyDtype(),
            DummyArray(np.array([1, 2])),
        ],
    )
    def test_is_extension_array_dtype(self, values):
        assert is_extension_array_dtype(values)

    @pytest.mark.parametrize("values", [np.array([]), pd.Series(np.array([]))])
    def test_is_not_extension_array_dtype(self, values):
        assert not is_extension_array_dtype(values)


def test_astype():
    arr = DummyArray(np.array([1, 2, 3]))
    expected = np.array([1, 2, 3], dtype=object)

    result = arr.astype(object)
    tm.assert_numpy_array_equal(result, expected)

    result = arr.astype("object")
    tm.assert_numpy_array_equal(result, expected)


def test_astype_no_copy():
    arr = DummyArray(np.array([1, 2, 3], dtype=np.int64))
    result = arr.astype(arr.dtype, copy=False)

    assert arr is result

    result = arr.astype(arr.dtype)
    assert arr is not result


@pytest.mark.parametrize("dtype", [dtypes.CategoricalDtype(), dtypes.IntervalDtype()])
def test_is_extension_array_dtype(dtype):
    assert isinstance(dtype, dtypes.ExtensionDtype)
    assert is_extension_array_dtype(dtype)


class CapturingStringArray(pd.arrays.StringArray):
    """Extend StringArray to capture arguments to __getitem__"""

    def __getitem__(self, item):
        self.last_item_arg = item
        return super().__getitem__(item)


def test_ellipsis_index():
    # GH#42430 1D slices over extension types turn into N-dimensional slices
    #  over ExtensionArrays
    df = pd.DataFrame(
        {"col1": CapturingStringArray(np.array(["hello", "world"], dtype=object))}
    )
    _ = df.iloc[:1]

    # String comparison because there's no native way to compare slices.
    # Before the fix for GH#42430, last_item_arg would get set to the 2D slice
    # (Ellipsis, slice(None, 1, None))
    out = df["col1"].array.last_item_arg
    assert str(out) == "slice(None, 1, None)"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_check_indexer.py ===
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer


@pytest.mark.parametrize(
    "indexer, expected",
    [
        # integer
        ([1, 2], np.array([1, 2], dtype=np.intp)),
        (np.array([1, 2], dtype="int64"), np.array([1, 2], dtype=np.intp)),
        (pd.array([1, 2], dtype="Int32"), np.array([1, 2], dtype=np.intp)),
        (pd.Index([1, 2]), np.array([1, 2], dtype=np.intp)),
        # boolean
        ([True, False, True], np.array([True, False, True], dtype=np.bool_)),
        (np.array([True, False, True]), np.array([True, False, True], dtype=np.bool_)),
        (
            pd.array([True, False, True], dtype="boolean"),
            np.array([True, False, True], dtype=np.bool_),
        ),
        # other
        ([], np.array([], dtype=np.intp)),
    ],
)
def test_valid_input(indexer, expected):
    arr = np.array([1, 2, 3])
    result = check_array_indexer(arr, indexer)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "indexer", [[True, False, None], pd.array([True, False, None], dtype="boolean")]
)
def test_boolean_na_returns_indexer(indexer):
    # https://github.com/pandas-dev/pandas/issues/31503
    arr = np.array([1, 2, 3])

    result = check_array_indexer(arr, indexer)
    expected = np.array([True, False, False], dtype=bool)

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "indexer",
    [
        [True, False],
        pd.array([True, False], dtype="boolean"),
        np.array([True, False], dtype=np.bool_),
    ],
)
def test_bool_raise_length(indexer):
    arr = np.array([1, 2, 3])

    msg = "Boolean index has wrong length"
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)


@pytest.mark.parametrize(
    "indexer", [[0, 1, None], pd.array([0, 1, pd.NA], dtype="Int64")]
)
def test_int_raise_missing_values(indexer):
    arr = np.array([1, 2, 3])

    msg = "Cannot index with an integer indexer containing NA values"
    with pytest.raises(ValueError, match=msg):
        check_array_indexer(arr, indexer)


@pytest.mark.parametrize(
    "indexer",
    [
        [0.0, 1.0],
        np.array([1.0, 2.0], dtype="float64"),
        np.array([True, False], dtype=object),
        pd.Index([True, False], dtype=object),
    ],
)
def test_raise_invalid_array_dtypes(indexer):
    arr = np.array([1, 2, 3])

    msg = "arrays used as indices must be of integer or boolean type"
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)


def test_raise_nullable_string_dtype(nullable_string_dtype):
    indexer = pd.array(["a", "b"], dtype=nullable_string_dtype)
    arr = np.array([1, 2, 3])

    msg = "arrays used as indices must be of integer or boolean type"
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)


@pytest.mark.parametrize("indexer", [None, Ellipsis, slice(0, 3), (None,)])
def test_pass_through_non_array_likes(indexer):
    arr = np.array([1, 2, 3])

    result = check_array_indexer(arr, indexer)
    assert result == indexer

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_missing.py ===
from datetime import timedelta

import numpy as np
import pytest

from pandas._libs import iNaT

import pandas as pd
from pandas import (
    Categorical,
    Index,
    NaT,
    Series,
    isna,
)
import pandas._testing as tm


class TestSeriesMissingData:
    def test_categorical_nan_handling(self):
        # NaNs are represented as -1 in labels
        s = Series(Categorical(["a", "b", np.nan, "a"]))
        tm.assert_index_equal(s.cat.categories, Index(["a", "b"]))
        tm.assert_numpy_array_equal(
            s.values.codes, np.array([0, 1, -1, 0], dtype=np.int8)
        )

    def test_isna_for_inf(self):
        s = Series(["a", np.inf, np.nan, pd.NA, 1.0])
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context("mode.use_inf_as_na", True):
                r = s.isna()
                dr = s.dropna()
        e = Series([False, True, True, True, False])
        de = Series(["a", 1.0], index=[0, 4])
        tm.assert_series_equal(r, e)
        tm.assert_series_equal(dr, de)

    def test_timedelta64_nan(self):
        td = Series([timedelta(days=i) for i in range(10)])

        # nan ops on timedeltas
        td1 = td.copy()
        td1[0] = np.nan
        assert isna(td1[0])
        assert td1[0]._value == iNaT
        td1[0] = td[0]
        assert not isna(td1[0])

        # GH#16674 iNaT is treated as an integer when given by the user
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            td1[1] = iNaT
        assert not isna(td1[1])
        assert td1.dtype == np.object_
        assert td1[1] == iNaT
        td1[1] = td[1]
        assert not isna(td1[1])

        td1[2] = NaT
        assert isna(td1[2])
        assert td1[2]._value == iNaT
        td1[2] = td[2]
        assert not isna(td1[2])

        # boolean setting
        # GH#2899 boolean setting
        td3 = np.timedelta64(timedelta(days=3))
        td7 = np.timedelta64(timedelta(days=7))
        td[(td > td3) & (td < td7)] = np.nan
        assert isna(td).sum() == 3

    @pytest.mark.xfail(
        reason="Chained inequality raises when trying to define 'selector'"
    )
    def test_logical_range_select(self, datetime_series):
        # NumPy limitation =(
        # https://github.com/pandas-dev/pandas/commit/9030dc021f07c76809848925cb34828f6c8484f3

        selector = -0.5 <= datetime_series <= 0.5
        expected = (datetime_series >= -0.5) & (datetime_series <= 0.5)
        tm.assert_series_equal(selector, expected)

    def test_valid(self, datetime_series):
        ts = datetime_series.copy()
        ts.index = ts.index._with_freq(None)
        ts[::2] = np.nan

        result = ts.dropna()
        assert len(result) == ts.count()
        tm.assert_series_equal(result, ts[1::2])
        tm.assert_series_equal(result, ts[pd.notna(ts)])


def test_hasnans_uncached_for_series():
    # GH#19700
    # set float64 dtype to avoid upcast when setting nan
    idx = Index([0, 1], dtype="float64")
    assert idx.hasnans is False
    assert "hasnans" in idx._cache
    ser = idx.to_series()
    assert ser.hasnans is False
    assert not hasattr(ser, "_cache")
    ser.iloc[-1] = np.nan
    assert ser.hasnans is True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\holiday\test_observance.py ===
from datetime import datetime

import pytest

from pandas.tseries.holiday import (
    after_nearest_workday,
    before_nearest_workday,
    nearest_workday,
    next_monday,
    next_monday_or_tuesday,
    next_workday,
    previous_friday,
    previous_workday,
    sunday_to_monday,
    weekend_to_monday,
)

_WEDNESDAY = datetime(2014, 4, 9)
_THURSDAY = datetime(2014, 4, 10)
_FRIDAY = datetime(2014, 4, 11)
_SATURDAY = datetime(2014, 4, 12)
_SUNDAY = datetime(2014, 4, 13)
_MONDAY = datetime(2014, 4, 14)
_TUESDAY = datetime(2014, 4, 15)
_NEXT_WEDNESDAY = datetime(2014, 4, 16)


@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_next_monday(day):
    assert next_monday(day) == _MONDAY


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_MONDAY, _TUESDAY)]
)
def test_next_monday_or_tuesday(day, expected):
    assert next_monday_or_tuesday(day) == expected


@pytest.mark.parametrize("day", [_SATURDAY, _SUNDAY])
def test_previous_friday(day):
    assert previous_friday(day) == _FRIDAY


def test_sunday_to_monday():
    assert sunday_to_monday(_SUNDAY) == _MONDAY


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _FRIDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)]
)
def test_nearest_workday(day, expected):
    assert nearest_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)]
)
def test_weekend_to_monday(day, expected):
    assert weekend_to_monday(day) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        (_WEDNESDAY, _THURSDAY),
        (_THURSDAY, _FRIDAY),
        (_SATURDAY, _MONDAY),
        (_SUNDAY, _MONDAY),
        (_MONDAY, _TUESDAY),
        (_TUESDAY, _NEXT_WEDNESDAY),  # WED is same week as TUE
    ],
)
def test_next_workday(day, expected):
    assert next_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _FRIDAY), (_SUNDAY, _FRIDAY), (_TUESDAY, _MONDAY)]
)
def test_previous_workday(day, expected):
    assert previous_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        (_THURSDAY, _WEDNESDAY),
        (_FRIDAY, _THURSDAY),
        (_SATURDAY, _THURSDAY),
        (_SUNDAY, _FRIDAY),
        (_MONDAY, _FRIDAY),  # last week Friday
        (_TUESDAY, _MONDAY),
        (_NEXT_WEDNESDAY, _TUESDAY),  # WED is same week as TUE
    ],
)
def test_before_nearest_workday(day, expected):
    assert before_nearest_workday(day) == expected


@pytest.mark.parametrize(
    "day,expected", [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_FRIDAY, _MONDAY)]
)
def test_after_nearest_workday(day, expected):
    assert after_nearest_workday(day) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_generators.py ===
from sympy.combinatorics.generators import symmetric, cyclic, alternating, \
    dihedral, rubik
from sympy.combinatorics.permutations import Permutation
from sympy.testing.pytest import raises

def test_generators():

    assert list(cyclic(6)) == [
        Permutation([0, 1, 2, 3, 4, 5]),
        Permutation([1, 2, 3, 4, 5, 0]),
        Permutation([2, 3, 4, 5, 0, 1]),
        Permutation([3, 4, 5, 0, 1, 2]),
        Permutation([4, 5, 0, 1, 2, 3]),
        Permutation([5, 0, 1, 2, 3, 4])]

    assert list(cyclic(10)) == [
        Permutation([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        Permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        Permutation([2, 3, 4, 5, 6, 7, 8, 9, 0, 1]),
        Permutation([3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
        Permutation([4, 5, 6, 7, 8, 9, 0, 1, 2, 3]),
        Permutation([5, 6, 7, 8, 9, 0, 1, 2, 3, 4]),
        Permutation([6, 7, 8, 9, 0, 1, 2, 3, 4, 5]),
        Permutation([7, 8, 9, 0, 1, 2, 3, 4, 5, 6]),
        Permutation([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]),
        Permutation([9, 0, 1, 2, 3, 4, 5, 6, 7, 8])]

    assert list(alternating(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 2, 1, 0])]

    assert list(symmetric(3)) == [
        Permutation([0, 1, 2]),
        Permutation([0, 2, 1]),
        Permutation([1, 0, 2]),
        Permutation([1, 2, 0]),
        Permutation([2, 0, 1]),
        Permutation([2, 1, 0])]

    assert list(symmetric(4)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([0, 1, 3, 2]),
        Permutation([0, 2, 1, 3]),
        Permutation([0, 2, 3, 1]),
        Permutation([0, 3, 1, 2]),
        Permutation([0, 3, 2, 1]),
        Permutation([1, 0, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([1, 2, 0, 3]),
        Permutation([1, 2, 3, 0]),
        Permutation([1, 3, 0, 2]),
        Permutation([1, 3, 2, 0]),
        Permutation([2, 0, 1, 3]),
        Permutation([2, 0, 3, 1]),
        Permutation([2, 1, 0, 3]),
        Permutation([2, 1, 3, 0]),
        Permutation([2, 3, 0, 1]),
        Permutation([2, 3, 1, 0]),
        Permutation([3, 0, 1, 2]),
        Permutation([3, 0, 2, 1]),
        Permutation([3, 1, 0, 2]),
        Permutation([3, 1, 2, 0]),
        Permutation([3, 2, 0, 1]),
        Permutation([3, 2, 1, 0])]

    assert list(dihedral(1)) == [
        Permutation([0, 1]), Permutation([1, 0])]

    assert list(dihedral(2)) == [
        Permutation([0, 1, 2, 3]),
        Permutation([1, 0, 3, 2]),
        Permutation([2, 3, 0, 1]),
        Permutation([3, 2, 1, 0])]

    assert list(dihedral(3)) == [
        Permutation([0, 1, 2]),
        Permutation([2, 1, 0]),
        Permutation([1, 2, 0]),
        Permutation([0, 2, 1]),
        Permutation([2, 0, 1]),
        Permutation([1, 0, 2])]

    assert list(dihedral(5)) == [
        Permutation([0, 1, 2, 3, 4]),
        Permutation([4, 3, 2, 1, 0]),
        Permutation([1, 2, 3, 4, 0]),
        Permutation([0, 4, 3, 2, 1]),
        Permutation([2, 3, 4, 0, 1]),
        Permutation([1, 0, 4, 3, 2]),
        Permutation([3, 4, 0, 1, 2]),
        Permutation([2, 1, 0, 4, 3]),
        Permutation([4, 0, 1, 2, 3]),
        Permutation([3, 2, 1, 0, 4])]

    raises(ValueError, lambda: rubik(1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_polyhedron.py ===
from sympy.core.symbol import symbols
from sympy.sets.sets import FiniteSet
from sympy.combinatorics.polyhedron import (Polyhedron,
    tetrahedron, cube as square, octahedron, dodecahedron, icosahedron,
    cube_faces)
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.testing.pytest import raises

rmul = Permutation.rmul


def test_polyhedron():
    raises(ValueError, lambda: Polyhedron(list('ab'),
        pgroup=[Permutation([0])]))
    pgroup = [Permutation([[0, 7, 2, 5], [6, 1, 4, 3]]),
              Permutation([[0, 7, 1, 6], [5, 2, 4, 3]]),
              Permutation([[3, 6, 0, 5], [4, 1, 7, 2]]),
              Permutation([[7, 4, 5], [1, 3, 0], [2], [6]]),
              Permutation([[1, 3, 2], [7, 6, 5], [4], [0]]),
              Permutation([[4, 7, 6], [2, 0, 3], [1], [5]]),
              Permutation([[1, 2, 0], [4, 5, 6], [3], [7]]),
              Permutation([[4, 2], [0, 6], [3, 7], [1, 5]]),
              Permutation([[3, 5], [7, 1], [2, 6], [0, 4]]),
              Permutation([[2, 5], [1, 6], [0, 4], [3, 7]]),
              Permutation([[4, 3], [7, 0], [5, 1], [6, 2]]),
              Permutation([[4, 1], [0, 5], [6, 2], [7, 3]]),
              Permutation([[7, 2], [3, 6], [0, 4], [1, 5]]),
              Permutation([0, 1, 2, 3, 4, 5, 6, 7])]
    corners = tuple(symbols('A:H'))
    faces = cube_faces
    cube = Polyhedron(corners, faces, pgroup)

    assert cube.edges == FiniteSet(*(
        (0, 1), (6, 7), (1, 2), (5, 6), (0, 3), (2, 3),
        (4, 7), (4, 5), (3, 7), (1, 5), (0, 4), (2, 6)))

    for i in range(3):  # add 180 degree face rotations
        cube.rotate(cube.pgroup[i]**2)

    assert cube.corners == corners

    for i in range(3, 7):  # add 240 degree axial corner rotations
        cube.rotate(cube.pgroup[i]**2)

    assert cube.corners == corners
    cube.rotate(1)
    raises(ValueError, lambda: cube.rotate(Permutation([0, 1])))
    assert cube.corners != corners
    assert cube.array_form == [7, 6, 4, 5, 3, 2, 0, 1]
    assert cube.cyclic_form == [[0, 7, 1, 6], [2, 4, 3, 5]]
    cube.reset()
    assert cube.corners == corners

    def check(h, size, rpt, target):

        assert len(h.faces) + len(h.vertices) - len(h.edges) == 2
        assert h.size == size

        got = set()
        for p in h.pgroup:
            # make sure it restores original
            P = h.copy()
            hit = P.corners
            for i in range(rpt):
                P.rotate(p)
                if P.corners == hit:
                    break
            else:
                print('error in permutation', p.array_form)
            for i in range(rpt):
                P.rotate(p)
                got.add(tuple(P.corners))
                c = P.corners
                f = [[c[i] for i in f] for f in P.faces]
                assert h.faces == Polyhedron(c, f).faces
        assert len(got) == target
        assert PermutationGroup([Permutation(g) for g in got]).is_group

    for h, size, rpt, target in zip(
        (tetrahedron, square, octahedron, dodecahedron, icosahedron),
        (4, 8, 6, 20, 12),
        (3, 4, 4, 5, 5),
            (12, 24, 24, 60, 60)):
        check(h, size, rpt, target)


def test_pgroups():
    from sympy.combinatorics.polyhedron import (cube, tetrahedron_faces,
            octahedron_faces, dodecahedron_faces, icosahedron_faces)
    from sympy.combinatorics.polyhedron import _pgroup_calcs
    (tetrahedron2, cube2, octahedron2, dodecahedron2, icosahedron2,
     tetrahedron_faces2, cube_faces2, octahedron_faces2,
     dodecahedron_faces2, icosahedron_faces2) = _pgroup_calcs()

    assert tetrahedron == tetrahedron2
    assert cube == cube2
    assert octahedron == octahedron2
    assert dodecahedron == dodecahedron2
    assert icosahedron == icosahedron2
    assert sorted(map(sorted, tetrahedron_faces)) == sorted(map(sorted, tetrahedron_faces2))
    assert sorted(cube_faces) == sorted(cube_faces2)
    assert sorted(octahedron_faces) == sorted(octahedron_faces2)
    assert sorted(dodecahedron_faces) == sorted(dodecahedron_faces2)
    assert sorted(icosahedron_faces) == sorted(icosahedron_faces2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_get_numeric_data.py ===
import numpy as np

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


class TestGetNumericData:
    def test_get_numeric_data_preserve_dtype(self):
        # get the numeric data
        obj = DataFrame({"A": [1, "2", 3.0]}, columns=Index(["A"], dtype="object"))
        result = obj._get_numeric_data()
        expected = DataFrame(dtype=object, index=pd.RangeIndex(3), columns=[])
        tm.assert_frame_equal(result, expected)

    def test_get_numeric_data(self, using_infer_string):
        datetime64name = np.dtype("M8[s]").name
        objectname = np.dtype(np.object_).name

        df = DataFrame(
            {"a": 1.0, "b": 2, "c": "foo", "f": Timestamp("20010102")},
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype("float64"),
                np.dtype("int64"),
                np.dtype(objectname)
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan),
                np.dtype(datetime64name),
            ],
            index=["a", "b", "c", "f"],
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                "d": np.array([1.0] * 10, dtype="float32"),
                "e": np.array([1] * 10, dtype="int32"),
                "f": np.array([1] * 10, dtype="int16"),
                "g": Timestamp("20010102"),
            },
            index=np.arange(10),
        )

        result = df._get_numeric_data()
        expected = df.loc[:, ["a", "b", "d", "e", "f"]]
        tm.assert_frame_equal(result, expected)

        only_obj = df.loc[:, ["c", "g"]]
        result = only_obj._get_numeric_data()
        expected = df.loc[:, []]
        tm.assert_frame_equal(result, expected)

        df = DataFrame.from_dict({"a": [1, 2], "b": ["foo", "bar"], "c": [np.pi, np.e]})
        result = df._get_numeric_data()
        expected = DataFrame.from_dict({"a": [1, 2], "c": [np.pi, np.e]})
        tm.assert_frame_equal(result, expected)

        df = result.copy()
        result = df._get_numeric_data()
        expected = df
        tm.assert_frame_equal(result, expected)

    def test_get_numeric_data_mixed_dtype(self):
        # numeric and object columns

        df = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [True, False, True],
                "c": ["foo", "bar", "baz"],
                "d": [None, None, None],
                "e": [3.14, 0.577, 2.773],
            }
        )
        result = df._get_numeric_data()
        tm.assert_index_equal(result.columns, Index(["a", "b", "e"]))

    def test_get_numeric_data_extension_dtype(self):
        # GH#22290
        df = DataFrame(
            {
                "A": pd.array([-10, np.nan, 0, 10, 20, 30], dtype="Int64"),
                "B": Categorical(list("abcabc")),
                "C": pd.array([0, 1, 2, 3, np.nan, 5], dtype="UInt8"),
                "D": IntervalArray.from_breaks(range(7)),
            }
        )
        result = df._get_numeric_data()
        expected = df.loc[:, ["A", "C"]]
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\base_class\test_indexing.py ===
import numpy as np
import pytest

from pandas._libs import index as libindex

import pandas as pd
from pandas import (
    Index,
    NaT,
)
import pandas._testing as tm


class TestGetSliceBounds:
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
    def test_get_slice_bounds_within(self, side, expected):
        index = Index(list("abcdef"))
        result = index.get_slice_bound("e", side=side)
        assert result == expected

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize(
        "data, bound, expected", [(list("abcdef"), "x", 6), (list("bcdefg"), "a", 0)]
    )
    def test_get_slice_bounds_outside(self, side, expected, data, bound):
        index = Index(data)
        result = index.get_slice_bound(bound, side=side)
        assert result == expected

    def test_get_slice_bounds_invalid_side(self):
        with pytest.raises(ValueError, match="Invalid value for side kwarg"):
            Index([]).get_slice_bound("a", side="middle")


class TestGetIndexerNonUnique:
    def test_get_indexer_non_unique_dtype_mismatch(self):
        # GH#25459
        indexes, missing = Index(["A", "B"]).get_indexer_non_unique(Index([0]))
        tm.assert_numpy_array_equal(np.array([-1], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), missing)

    @pytest.mark.parametrize(
        "idx_values,idx_non_unique",
        [
            ([np.nan, 100, 200, 100], [np.nan, 100]),
            ([np.nan, 100.0, 200.0, 100.0], [np.nan, 100.0]),
        ],
    )
    def test_get_indexer_non_unique_int_index(self, idx_values, idx_non_unique):
        indexes, missing = Index(idx_values).get_indexer_non_unique(Index([np.nan]))
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)

        indexes, missing = Index(idx_values).get_indexer_non_unique(
            Index(idx_non_unique)
        )
        tm.assert_numpy_array_equal(np.array([0, 1, 3], dtype=np.intp), indexes)
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)


class TestGetLoc:
    @pytest.mark.slow  # to_flat_index takes a while
    def test_get_loc_tuple_monotonic_above_size_cutoff(self, monkeypatch):
        # Go through the libindex path for which using
        # _bin_search vs ndarray.searchsorted makes a difference

        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 100)
            lev = list("ABCD")
            dti = pd.date_range("2016-01-01", periods=10)

            mi = pd.MultiIndex.from_product([lev, range(5), dti])
            oidx = mi.to_flat_index()

            loc = len(oidx) // 2
            tup = oidx[loc]

            res = oidx.get_loc(tup)
        assert res == loc

    def test_get_loc_nan_object_dtype_nonmonotonic_nonunique(self):
        # case that goes through _maybe_get_bool_indexer
        idx = Index(["foo", np.nan, None, "foo", 1.0, None], dtype=object)

        # we dont raise KeyError on nan
        res = idx.get_loc(np.nan)
        assert res == 1

        # we only match on None, not on np.nan
        res = idx.get_loc(None)
        expected = np.array([False, False, True, False, False, True])
        tm.assert_numpy_array_equal(res, expected)

        # we don't match at all on mismatched NA
        with pytest.raises(KeyError, match="NaT"):
            idx.get_loc(NaT)


def test_getitem_boolean_ea_indexer():
    # GH#45806
    ser = pd.Series([True, False, pd.NA], dtype="boolean")
    result = ser.index[ser]
    expected = Index([0])
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\_compilation\tests\test_compilation.py ===
import shutil
import os
import subprocess
import tempfile
from sympy.external import import_module
from sympy.testing.pytest import skip, skip_under_pyodide

from sympy.utilities._compilation.compilation import compile_link_import_py_ext, compile_link_import_strings, compile_sources, get_abspath

numpy = import_module('numpy')
cython = import_module('cython')

_sources1 = [
    ('sigmoid.c', r"""
#include <math.h>

void sigmoid(int n, const double * const restrict in,
             double * const restrict out, double lim){
    for (int i=0; i<n; ++i){
        const double x = in[i];
        out[i] = x*pow(pow(x/lim, 8)+1, -1./8.);
    }
}
"""),
    ('_sigmoid.pyx', r"""
import numpy as np
cimport numpy as cnp

cdef extern void c_sigmoid "sigmoid" (int, const double * const,
                                      double * const, double)

def sigmoid(double [:] inp, double lim=350.0):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(
        inp.size, dtype=np.float64)
    c_sigmoid(inp.size, &inp[0], &out[0], lim)
    return out
""")
]


def npy(data, lim=350.0):
    return data/((data/lim)**8+1)**(1/8.)


def test_compile_link_import_strings():
    if not numpy:
        skip("numpy not installed.")
    if not cython:
        skip("cython not installed.")

    from sympy.utilities._compilation import has_c
    if not has_c():
        skip("No C compiler found.")

    compile_kw = {"std": 'c99', "include_dirs": [numpy.get_include()]}
    info = None
    try:
        mod, info = compile_link_import_strings(_sources1, compile_kwargs=compile_kw)
        data = numpy.random.random(1024*1024*8)  # 64 MB of RAM needed..
        res_mod = mod.sigmoid(data)
        res_npy = npy(data)
        assert numpy.allclose(res_mod, res_npy)
    finally:
        if info and info['build_dir']:
            shutil.rmtree(info['build_dir'])


@skip_under_pyodide("Emscripten does not support subprocesses")
def test_compile_sources():
    tmpdir = tempfile.mkdtemp()

    from sympy.utilities._compilation import has_c
    if not has_c():
        skip("No C compiler found.")

    build_dir = str(tmpdir)
    _handle, file_path = tempfile.mkstemp('.c', dir=build_dir)
    with open(file_path, 'wt') as ofh:
        ofh.write("""
        int foo(int bar) {
            return 2*bar;
        }
        """)
    obj, = compile_sources([file_path], cwd=build_dir)
    obj_path = get_abspath(obj, cwd=build_dir)
    assert os.path.exists(obj_path)
    try:
        _ = subprocess.check_output(["nm", "--help"])
    except subprocess.CalledProcessError:
        pass  # we cannot test contents of object file
    else:
        nm_out = subprocess.check_output(["nm", obj_path])
        assert 'foo' in nm_out.decode('utf-8')

    if not cython:
        return  # the final (optional) part of the test below requires Cython.

    _handle, pyx_path = tempfile.mkstemp('.pyx', dir=build_dir)
    with open(pyx_path, 'wt') as ofh:
        ofh.write(("cdef extern int foo(int)\n"
                   "def _foo(arg):\n"
                   "    return foo(arg)"))
    mod = compile_link_import_py_ext([pyx_path], extra_objs=[obj_path], build_dir=build_dir)
    assert mod._foo(21) == 42

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_simd_module.py ===
import pytest

from numpy._core._simd import targets

"""
This testing unit only for checking the sanity of common functionality,
therefore all we need is just to take one submodule that represents any
of enabled SIMD extensions to run the test on it and the second submodule
required to run only one check related to the possibility of mixing
the data types among each submodule.
"""
npyvs = [npyv_mod for npyv_mod in targets.values() if npyv_mod and npyv_mod.simd]
npyv, npyv2 = (npyvs + [None, None])[:2]

unsigned_sfx = ["u8", "u16", "u32", "u64"]
signed_sfx = ["s8", "s16", "s32", "s64"]
fp_sfx = []
if npyv and npyv.simd_f32:
    fp_sfx.append("f32")
if npyv and npyv.simd_f64:
    fp_sfx.append("f64")

int_sfx = unsigned_sfx + signed_sfx
all_sfx = unsigned_sfx + int_sfx

@pytest.mark.skipif(not npyv, reason="could not find any SIMD extension with NPYV support")
class Test_SIMD_MODULE:

    @pytest.mark.parametrize('sfx', all_sfx)
    def test_num_lanes(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        vector = getattr(npyv, "setall_" + sfx)(1)
        assert len(vector) == nlanes

    @pytest.mark.parametrize('sfx', all_sfx)
    def test_type_name(self, sfx):
        vector = getattr(npyv, "setall_" + sfx)(1)
        assert vector.__name__ == "npyv_" + sfx

    def test_raises(self):
        a, b = [npyv.setall_u32(1)] * 2
        for sfx in all_sfx:
            vcb = lambda intrin: getattr(npyv, f"{intrin}_{sfx}")
            pytest.raises(TypeError, vcb("add"), a)
            pytest.raises(TypeError, vcb("add"), a, b, a)
            pytest.raises(TypeError, vcb("setall"))
            pytest.raises(TypeError, vcb("setall"), [1])
            pytest.raises(TypeError, vcb("load"), 1)
            pytest.raises(ValueError, vcb("load"), [1])
            pytest.raises(ValueError, vcb("store"), [1], getattr(npyv, f"reinterpret_{sfx}_u32")(a))

    @pytest.mark.skipif(not npyv2, reason=(
        "could not find a second SIMD extension with NPYV support"
    ))
    def test_nomix(self):
        # mix among submodules isn't allowed
        a = npyv.setall_u32(1)
        a2 = npyv2.setall_u32(1)
        pytest.raises(TypeError, npyv.add_u32, a2, a2)
        pytest.raises(TypeError, npyv2.add_u32, a, a)

    @pytest.mark.parametrize('sfx', unsigned_sfx)
    def test_unsigned_overflow(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        maxu = (1 << int(sfx[1:])) - 1
        maxu_72 = (1 << 72) - 1
        lane = getattr(npyv, "setall_" + sfx)(maxu_72)[0]
        assert lane == maxu
        lanes = getattr(npyv, "load_" + sfx)([maxu_72] * nlanes)
        assert lanes == [maxu] * nlanes
        lane = getattr(npyv, "setall_" + sfx)(-1)[0]
        assert lane == maxu
        lanes = getattr(npyv, "load_" + sfx)([-1] * nlanes)
        assert lanes == [maxu] * nlanes

    @pytest.mark.parametrize('sfx', signed_sfx)
    def test_signed_overflow(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        maxs_72 = (1 << 71) - 1
        lane = getattr(npyv, "setall_" + sfx)(maxs_72)[0]
        assert lane == -1
        lanes = getattr(npyv, "load_" + sfx)([maxs_72] * nlanes)
        assert lanes == [-1] * nlanes
        mins_72 = -1 << 71
        lane = getattr(npyv, "setall_" + sfx)(mins_72)[0]
        assert lane == 0
        lanes = getattr(npyv, "load_" + sfx)([mins_72] * nlanes)
        assert lanes == [0] * nlanes

    def test_truncate_f32(self):
        if not npyv.simd_f32:
            pytest.skip("F32 isn't support by the SIMD extension")
        f32 = npyv.setall_f32(0.1)[0]
        assert f32 != 0.1
        assert round(f32, 1) == 0.1

    def test_compare(self):
        data_range = range(npyv.nlanes_u32)
        vdata = npyv.load_u32(data_range)
        assert vdata == list(data_range)
        assert vdata == tuple(data_range)
        for i in data_range:
            assert vdata[i] == data_range[i]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_register_accessor.py ===
from collections.abc import Generator
import contextlib

import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core import accessor


def test_dirname_mixin() -> None:
    # GH37173

    class X(accessor.DirNamesMixin):
        x = 1
        y: int

        def __init__(self) -> None:
            self.z = 3

    result = [attr_name for attr_name in dir(X()) if not attr_name.startswith("_")]

    assert result == ["x", "z"]


@contextlib.contextmanager
def ensure_removed(obj, attr) -> Generator[None, None, None]:
    """Ensure that an attribute added to 'obj' during the test is
    removed when we're done
    """
    try:
        yield
    finally:
        try:
            delattr(obj, attr)
        except AttributeError:
            pass
        obj._accessors.discard(attr)


class MyAccessor:
    def __init__(self, obj) -> None:
        self.obj = obj
        self.item = "item"

    @property
    def prop(self):
        return self.item

    def method(self):
        return self.item


@pytest.mark.parametrize(
    "obj, registrar",
    [
        (pd.Series, pd.api.extensions.register_series_accessor),
        (pd.DataFrame, pd.api.extensions.register_dataframe_accessor),
        (pd.Index, pd.api.extensions.register_index_accessor),
    ],
)
def test_register(obj, registrar):
    with ensure_removed(obj, "mine"):
        before = set(dir(obj))
        registrar("mine")(MyAccessor)
        o = obj([]) if obj is not pd.Series else obj([], dtype=object)
        assert o.mine.prop == "item"
        after = set(dir(obj))
        assert (before ^ after) == {"mine"}
        assert "mine" in obj._accessors


def test_accessor_works():
    with ensure_removed(pd.Series, "mine"):
        pd.api.extensions.register_series_accessor("mine")(MyAccessor)

        s = pd.Series([1, 2])
        assert s.mine.obj is s

        assert s.mine.prop == "item"
        assert s.mine.method() == "item"


def test_overwrite_warns():
    match = r".*MyAccessor.*fake.*Series.*"
    with tm.assert_produces_warning(UserWarning, match=match):
        with ensure_removed(pd.Series, "fake"):
            setattr(pd.Series, "fake", 123)
            pd.api.extensions.register_series_accessor("fake")(MyAccessor)
            s = pd.Series([1, 2])
            assert s.fake.prop == "item"


def test_raises_attribute_error():
    with ensure_removed(pd.Series, "bad"):

        @pd.api.extensions.register_series_accessor("bad")
        class Bad:
            def __init__(self, data) -> None:
                raise AttributeError("whoops")

        with pytest.raises(AttributeError, match="whoops"):
            pd.Series([], dtype=object).bad

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\timedeltas\test_constructors.py ===
import numpy as np
import pytest

import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestTimedeltaArrayConstructor:
    def test_only_1dim_accepted(self):
        # GH#25282
        arr = np.array([0, 1, 2, 3], dtype="m8[h]").astype("m8[ns]")

        depr_msg = "TimedeltaArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Only 1-dimensional"):
                # 3-dim, we allow 2D to sneak in for ops purposes GH#29853
                TimedeltaArray(arr.reshape(2, 2, 1))

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="Only 1-dimensional"):
                # 0-dim
                TimedeltaArray(arr[[0]].squeeze())

    def test_freq_validation(self):
        # ensure that the public constructor cannot create an invalid instance
        arr = np.array([0, 0, 1], dtype=np.int64) * 3600 * 10**9

        msg = (
            "Inferred frequency None from passed values does not "
            "conform to passed frequency D"
        )
        depr_msg = "TimedeltaArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                TimedeltaArray(arr.view("timedelta64[ns]"), freq="D")

    def test_non_array_raises(self):
        depr_msg = "TimedeltaArray.__init__ is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match="list"):
                TimedeltaArray([1, 2, 3])

    def test_other_type_raises(self):
        msg = r"dtype bool cannot be converted to timedelta64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            TimedeltaArray._from_sequence(np.array([1, 2, 3], dtype="bool"))

    def test_incorrect_dtype_raises(self):
        msg = "dtype 'category' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype="category"
            )

        msg = "dtype 'int64' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("int64")
            )

        msg = r"dtype 'datetime64\[ns\]' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("M8[ns]")
            )

        msg = (
            r"dtype 'datetime64\[us, UTC\]' is invalid, should be np.timedelta64 dtype"
        )
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype="M8[us, UTC]"
            )

        msg = "Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence(
                np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("m8[Y]")
            )

    def test_mismatched_values_dtype_units(self):
        arr = np.array([1, 2, 3], dtype="m8[s]")
        dtype = np.dtype("m8[ns]")
        msg = r"Values resolution does not match dtype"
        depr_msg = "TimedeltaArray.__init__ is deprecated"

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            with pytest.raises(ValueError, match=msg):
                TimedeltaArray(arr, dtype=dtype)

    def test_copy(self):
        data = np.array([1, 2, 3], dtype="m8[ns]")
        arr = TimedeltaArray._from_sequence(data, copy=False)
        assert arr._ndarray is data

        arr = TimedeltaArray._from_sequence(data, copy=True)
        assert arr._ndarray is not data
        assert arr._ndarray.base is not data

    def test_from_sequence_dtype(self):
        msg = "dtype 'object' is invalid, should be np.timedelta64 dtype"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence([], dtype=object)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimelike_\test_value_counts.py ===
import numpy as np

from pandas import (
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestValueCounts:
    # GH#7735

    def test_value_counts_unique_datetimeindex(self, tz_naive_fixture):
        tz = tz_naive_fixture
        orig = date_range("2011-01-01 09:00", freq="h", periods=10, tz=tz)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_timedeltaindex(self):
        orig = timedelta_range("1 days 09:00:00", freq="h", periods=10)
        self._check_value_counts_with_repeats(orig)

    def test_value_counts_unique_periodindex(self):
        orig = period_range("2011-01-01 09:00", freq="h", periods=10)
        self._check_value_counts_with_repeats(orig)

    def _check_value_counts_with_repeats(self, orig):
        # create repeated values, 'n'th element is repeated by n+1 times
        idx = type(orig)(
            np.repeat(orig._values, range(1, len(orig) + 1)), dtype=orig.dtype
        )

        exp_idx = orig[::-1]
        if not isinstance(exp_idx, PeriodIndex):
            exp_idx = exp_idx._with_freq(None)
        expected = Series(range(10, 0, -1), index=exp_idx, dtype="int64", name="count")

        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)

        tm.assert_index_equal(idx.unique(), orig)

    def test_value_counts_unique_datetimeindex2(self, tz_naive_fixture):
        tz = tz_naive_fixture
        idx = DatetimeIndex(
            [
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 08:00",
                "2013-01-01 08:00",
                NaT,
            ],
            tz=tz,
        )
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_timedeltaindex2(self):
        idx = TimedeltaIndex(
            [
                "1 days 09:00:00",
                "1 days 09:00:00",
                "1 days 09:00:00",
                "1 days 08:00:00",
                "1 days 08:00:00",
                NaT,
            ]
        )
        self._check_value_counts_dropna(idx)

    def test_value_counts_unique_periodindex2(self):
        idx = PeriodIndex(
            [
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 09:00",
                "2013-01-01 08:00",
                "2013-01-01 08:00",
                NaT,
            ],
            freq="h",
        )
        self._check_value_counts_dropna(idx)

    def _check_value_counts_dropna(self, idx):
        exp_idx = idx[[2, 3]]
        expected = Series([3, 2], index=exp_idx, name="count")

        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(), expected)

        exp_idx = idx[[2, 3, -1]]
        expected = Series([3, 2, 1], index=exp_idx, name="count")

        for obj in [idx, Series(idx)]:
            tm.assert_series_equal(obj.value_counts(dropna=False), expected)

        tm.assert_index_equal(idx.unique(), exp_idx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_isin.py ===
import numpy as np
import pytest

from pandas import MultiIndex
import pandas._testing as tm


def test_isin_nan():
    idx = MultiIndex.from_arrays([["foo", "bar"], [1.0, np.nan]])
    tm.assert_numpy_array_equal(idx.isin([("bar", np.nan)]), np.array([False, True]))
    tm.assert_numpy_array_equal(
        idx.isin([("bar", float("nan"))]), np.array([False, True])
    )


def test_isin_missing(nulls_fixture):
    # GH48905
    mi1 = MultiIndex.from_tuples([(1, nulls_fixture)])
    mi2 = MultiIndex.from_tuples([(1, 1), (1, 2)])
    result = mi2.isin(mi1)
    expected = np.array([False, False])
    tm.assert_numpy_array_equal(result, expected)


def test_isin():
    values = [("foo", 2), ("bar", 3), ("quux", 4)]

    idx = MultiIndex.from_arrays([["qux", "baz", "foo", "bar"], np.arange(4)])
    result = idx.isin(values)
    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(result, expected)

    # empty, return dtype bool
    idx = MultiIndex.from_arrays([[], []])
    result = idx.isin(values)
    assert len(result) == 0
    assert result.dtype == np.bool_


def test_isin_level_kwarg():
    idx = MultiIndex.from_arrays([["qux", "baz", "foo", "bar"], np.arange(4)])

    vals_0 = ["foo", "bar", "quux"]
    vals_1 = [2, 3, 10]

    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=0))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level=-2))

    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=1))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level=-1))

    msg = "Too many levels: Index has only 2 levels, not 6"
    with pytest.raises(IndexError, match=msg):
        idx.isin(vals_0, level=5)
    msg = "Too many levels: Index has only 2 levels, -5 is not a valid level number"
    with pytest.raises(IndexError, match=msg):
        idx.isin(vals_0, level=-5)

    with pytest.raises(KeyError, match=r"'Level 1\.0 not found'"):
        idx.isin(vals_0, level=1.0)
    with pytest.raises(KeyError, match=r"'Level -1\.0 not found'"):
        idx.isin(vals_1, level=-1.0)
    with pytest.raises(KeyError, match="'Level A not found'"):
        idx.isin(vals_1, level="A")

    idx.names = ["A", "B"]
    tm.assert_numpy_array_equal(expected, idx.isin(vals_0, level="A"))
    tm.assert_numpy_array_equal(expected, idx.isin(vals_1, level="B"))

    with pytest.raises(KeyError, match="'Level C not found'"):
        idx.isin(vals_1, level="C")


@pytest.mark.parametrize(
    "labels,expected,level",
    [
        ([("b", np.nan)], np.array([False, False, True]), None),
        ([np.nan, "a"], np.array([True, True, False]), 0),
        (["d", np.nan], np.array([False, True, True]), 1),
    ],
)
def test_isin_multi_index_with_missing_value(labels, expected, level):
    # GH 19132
    midx = MultiIndex.from_arrays([[np.nan, "a", "b"], ["c", "d", np.nan]])
    result = midx.isin(labels, level=level)
    tm.assert_numpy_array_equal(result, expected)


def test_isin_empty():
    # GH#51599
    midx = MultiIndex.from_arrays([[1, 2], [3, 4]])
    result = midx.isin([])
    expected = np.array([False, False])
    tm.assert_numpy_array_equal(result, expected)


def test_isin_generator():
    # GH#52568
    midx = MultiIndex.from_tuples([(1, 2)])
    result = midx.isin(x for x in [(1, 2)])
    expected = np.array([True])
    tm.assert_numpy_array_equal(result, expected)