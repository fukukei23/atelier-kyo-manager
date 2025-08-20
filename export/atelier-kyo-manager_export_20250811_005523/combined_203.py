
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\test_nat.py ===
from datetime import (
    datetime,
    timedelta,
)
import operator

import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3

from pandas import (
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    isna,
    offsets,
)
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)


class TestNaTFormatting:
    def test_repr(self):
        assert repr(NaT) == "NaT"

    def test_str(self):
        assert str(NaT) == "NaT"

    def test_isoformat(self):
        assert NaT.isoformat() == "NaT"


@pytest.mark.parametrize(
    "nat,idx",
    [
        (Timestamp("NaT"), DatetimeArray),
        (Timedelta("NaT"), TimedeltaArray),
        (Period("NaT", freq="M"), PeriodArray),
    ],
)
def test_nat_fields(nat, idx):
    for field in idx._field_ops:
        # weekday is a property of DTI, but a method
        # on NaT/Timestamp for compat with datetime
        if field == "weekday":
            continue

        result = getattr(NaT, field)
        assert np.isnan(result)

        result = getattr(nat, field)
        assert np.isnan(result)

    for field in idx._bool_ops:
        result = getattr(NaT, field)
        assert result is False

        result = getattr(nat, field)
        assert result is False


def test_nat_vector_field_access():
    idx = DatetimeIndex(["1/1/2000", None, None, "1/4/2000"])

    for field in DatetimeArray._field_ops:
        # weekday is a property of DTI, but a method
        # on NaT/Timestamp for compat with datetime
        if field == "weekday":
            continue

        result = getattr(idx, field)
        expected = Index([getattr(x, field) for x in idx])
        tm.assert_index_equal(result, expected)

    ser = Series(idx)

    for field in DatetimeArray._field_ops:
        # weekday is a property of DTI, but a method
        # on NaT/Timestamp for compat with datetime
        if field == "weekday":
            continue

        result = getattr(ser.dt, field)
        expected = [getattr(x, field) for x in idx]
        tm.assert_series_equal(result, Series(expected))

    for field in DatetimeArray._bool_ops:
        result = getattr(ser.dt, field)
        expected = [getattr(x, field) for x in idx]
        tm.assert_series_equal(result, Series(expected))


@pytest.mark.parametrize("klass", [Timestamp, Timedelta, Period])
@pytest.mark.parametrize(
    "value", [None, np.nan, iNaT, float("nan"), NaT, "NaT", "nat", "", "NAT"]
)
def test_identity(klass, value):
    assert klass(value) is NaT


@pytest.mark.parametrize("klass", [Timestamp, Timedelta])
@pytest.mark.parametrize("method", ["round", "floor", "ceil"])
@pytest.mark.parametrize("freq", ["s", "5s", "min", "5min", "h", "5h"])
def test_round_nat(klass, method, freq):
    # see gh-14940
    ts = klass("nat")

    round_method = getattr(ts, method)
    assert round_method(freq) is ts


@pytest.mark.parametrize(
    "method",
    [
        "astimezone",
        "combine",
        "ctime",
        "dst",
        "fromordinal",
        "fromtimestamp",
        "fromisocalendar",
        "isocalendar",
        "strftime",
        "strptime",
        "time",
        "timestamp",
        "timetuple",
        "timetz",
        "toordinal",
        "tzname",
        "utcfromtimestamp",
        "utcnow",
        "utcoffset",
        "utctimetuple",
        "timestamp",
    ],
)
def test_nat_methods_raise(method):
    # see gh-9513, gh-17329
    msg = f"NaTType does not support {method}"

    with pytest.raises(ValueError, match=msg):
        getattr(NaT, method)()


@pytest.mark.parametrize("method", ["weekday", "isoweekday"])
def test_nat_methods_nan(method):
    # see gh-9513, gh-17329
    assert np.isnan(getattr(NaT, method)())


@pytest.mark.parametrize(
    "method", ["date", "now", "replace", "today", "tz_convert", "tz_localize"]
)
def test_nat_methods_nat(method):
    # see gh-8254, gh-9513, gh-17329
    assert getattr(NaT, method)() is NaT


@pytest.mark.parametrize(
    "get_nat", [lambda x: NaT, lambda x: Timedelta(x), lambda x: Timestamp(x)]
)
def test_nat_iso_format(get_nat):
    # see gh-12300
    assert get_nat("NaT").isoformat() == "NaT"
    assert get_nat("NaT").isoformat(timespec="nanoseconds") == "NaT"


@pytest.mark.parametrize(
    "klass,expected",
    [
        (Timestamp, ["normalize", "to_julian_date", "to_period", "unit"]),
        (
            Timedelta,
            [
                "components",
                "resolution_string",
                "to_pytimedelta",
                "to_timedelta64",
                "unit",
                "view",
            ],
        ),
    ],
)
def test_missing_public_nat_methods(klass, expected):
    # see gh-17327
    #
    # NaT should have *most* of the Timestamp and Timedelta methods.
    # Here, we check which public methods NaT does not have. We
    # ignore any missing private methods.
    nat_names = dir(NaT)
    klass_names = dir(klass)

    missing = [x for x in klass_names if x not in nat_names and not x.startswith("_")]
    missing.sort()

    assert missing == expected


def _get_overlap_public_nat_methods(klass, as_tuple=False):
    """
    Get overlapping public methods between NaT and another class.

    Parameters
    ----------
    klass : type
        The class to compare with NaT
    as_tuple : bool, default False
        Whether to return a list of tuples of the form (klass, method).

    Returns
    -------
    overlap : list
    """
    nat_names = dir(NaT)
    klass_names = dir(klass)

    overlap = [
        x
        for x in nat_names
        if x in klass_names and not x.startswith("_") and callable(getattr(klass, x))
    ]

    # Timestamp takes precedence over Timedelta in terms of overlap.
    if klass is Timedelta:
        ts_names = dir(Timestamp)
        overlap = [x for x in overlap if x not in ts_names]

    if as_tuple:
        overlap = [(klass, method) for method in overlap]

    overlap.sort()
    return overlap


@pytest.mark.parametrize(
    "klass,expected",
    [
        (
            Timestamp,
            [
                "as_unit",
                "astimezone",
                "ceil",
                "combine",
                "ctime",
                "date",
                "day_name",
                "dst",
                "floor",
                "fromisocalendar",
                "fromisoformat",
                "fromordinal",
                "fromtimestamp",
                "isocalendar",
                "isoformat",
                "isoweekday",
                "month_name",
                "now",
                "replace",
                "round",
                "strftime",
                "strptime",
                "time",
                "timestamp",
                "timetuple",
                "timetz",
                "to_datetime64",
                "to_numpy",
                "to_pydatetime",
                "today",
                "toordinal",
                "tz_convert",
                "tz_localize",
                "tzname",
                "utcfromtimestamp",
                "utcnow",
                "utcoffset",
                "utctimetuple",
                "weekday",
            ],
        ),
        (Timedelta, ["total_seconds"]),
    ],
)
def test_overlap_public_nat_methods(klass, expected):
    # see gh-17327
    #
    # NaT should have *most* of the Timestamp and Timedelta methods.
    # In case when Timestamp, Timedelta, and NaT are overlap, the overlap
    # is considered to be with Timestamp and NaT, not Timedelta.
    assert _get_overlap_public_nat_methods(klass) == expected


@pytest.mark.parametrize(
    "compare",
    (
        _get_overlap_public_nat_methods(Timestamp, True)
        + _get_overlap_public_nat_methods(Timedelta, True)
    ),
    ids=lambda x: f"{x[0].__name__}.{x[1]}",
)
def test_nat_doc_strings(compare):
    # see gh-17327
    #
    # The docstrings for overlapping methods should match.
    klass, method = compare
    klass_doc = getattr(klass, method).__doc__

    if klass == Timestamp and method == "isoformat":
        pytest.skip(
            "Ignore differences with Timestamp.isoformat() as they're intentional"
        )

    if method == "to_numpy":
        # GH#44460 can return either dt64 or td64 depending on dtype,
        #  different docstring is intentional
        pytest.skip(f"different docstring for {method} is intentional")

    nat_doc = getattr(NaT, method).__doc__
    assert klass_doc == nat_doc


_ops = {
    "left_plus_right": lambda a, b: a + b,
    "right_plus_left": lambda a, b: b + a,
    "left_minus_right": lambda a, b: a - b,
    "right_minus_left": lambda a, b: b - a,
    "left_times_right": lambda a, b: a * b,
    "right_times_left": lambda a, b: b * a,
    "left_div_right": lambda a, b: a / b,
    "right_div_left": lambda a, b: b / a,
}


@pytest.mark.parametrize("op_name", list(_ops.keys()))
@pytest.mark.parametrize(
    "value,val_type",
    [
        (2, "scalar"),
        (1.5, "floating"),
        (np.nan, "floating"),
        ("foo", "str"),
        (timedelta(3600), "timedelta"),
        (Timedelta("5s"), "timedelta"),
        (datetime(2014, 1, 1), "timestamp"),
        (Timestamp("2014-01-01"), "timestamp"),
        (Timestamp("2014-01-01", tz="UTC"), "timestamp"),
        (Timestamp("2014-01-01", tz="US/Eastern"), "timestamp"),
        (pytz.timezone("Asia/Tokyo").localize(datetime(2014, 1, 1)), "timestamp"),
    ],
)
def test_nat_arithmetic_scalar(op_name, value, val_type):
    # see gh-6873
    invalid_ops = {
        "scalar": {"right_div_left"},
        "floating": {
            "right_div_left",
            "left_minus_right",
            "right_minus_left",
            "left_plus_right",
            "right_plus_left",
        },
        "str": set(_ops.keys()),
        "timedelta": {"left_times_right", "right_times_left"},
        "timestamp": {
            "left_times_right",
            "right_times_left",
            "left_div_right",
            "right_div_left",
        },
    }

    op = _ops[op_name]

    if op_name in invalid_ops.get(val_type, set()):
        if (
            val_type == "timedelta"
            and "times" in op_name
            and isinstance(value, Timedelta)
        ):
            typs = "(Timedelta|NaTType)"
            msg = rf"unsupported operand type\(s\) for \*: '{typs}' and '{typs}'"
        elif val_type == "str":
            # un-specific check here because the message comes from str
            #  and varies by method
            msg = "|".join(
                [
                    "can only concatenate str",
                    "unsupported operand type",
                    "can't multiply sequence",
                    "Can't convert 'NaTType'",
                    "must be str, not NaTType",
                ]
            )
        else:
            msg = "unsupported operand type"

        with pytest.raises(TypeError, match=msg):
            op(NaT, value)
    else:
        if val_type == "timedelta" and "div" in op_name:
            expected = np.nan
        else:
            expected = NaT

        assert op(NaT, value) is expected


@pytest.mark.parametrize(
    "val,expected", [(np.nan, NaT), (NaT, np.nan), (np.timedelta64("NaT"), np.nan)]
)
def test_nat_rfloordiv_timedelta(val, expected):
    # see gh-#18846
    #
    # See also test_timedelta.TestTimedeltaArithmetic.test_floordiv
    td = Timedelta(hours=3, minutes=4)
    assert td // val is expected


@pytest.mark.parametrize(
    "op_name",
    ["left_plus_right", "right_plus_left", "left_minus_right", "right_minus_left"],
)
@pytest.mark.parametrize(
    "value",
    [
        DatetimeIndex(["2011-01-01", "2011-01-02"], name="x"),
        DatetimeIndex(["2011-01-01", "2011-01-02"], tz="US/Eastern", name="x"),
        DatetimeArray._from_sequence(["2011-01-01", "2011-01-02"], dtype="M8[ns]"),
        DatetimeArray._from_sequence(
            ["2011-01-01", "2011-01-02"], dtype=DatetimeTZDtype(tz="US/Pacific")
        ),
        TimedeltaIndex(["1 day", "2 day"], name="x"),
    ],
)
def test_nat_arithmetic_index(op_name, value):
    # see gh-11718
    exp_name = "x"
    exp_data = [NaT] * 2

    if value.dtype.kind == "M" and "plus" in op_name:
        expected = DatetimeIndex(exp_data, tz=value.tz, name=exp_name)
    else:
        expected = TimedeltaIndex(exp_data, name=exp_name)
    expected = expected.as_unit(value.unit)

    if not isinstance(value, Index):
        expected = expected.array

    op = _ops[op_name]
    result = op(NaT, value)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "op_name",
    ["left_plus_right", "right_plus_left", "left_minus_right", "right_minus_left"],
)
@pytest.mark.parametrize("box", [TimedeltaIndex, Series, TimedeltaArray._from_sequence])
def test_nat_arithmetic_td64_vector(op_name, box):
    # see gh-19124
    vec = box(["1 day", "2 day"], dtype="timedelta64[ns]")
    box_nat = box([NaT, NaT], dtype="timedelta64[ns]")
    tm.assert_equal(_ops[op_name](vec, NaT), box_nat)


@pytest.mark.parametrize(
    "dtype,op,out_dtype",
    [
        ("datetime64[ns]", operator.add, "datetime64[ns]"),
        ("datetime64[ns]", roperator.radd, "datetime64[ns]"),
        ("datetime64[ns]", operator.sub, "timedelta64[ns]"),
        ("datetime64[ns]", roperator.rsub, "timedelta64[ns]"),
        ("timedelta64[ns]", operator.add, "datetime64[ns]"),
        ("timedelta64[ns]", roperator.radd, "datetime64[ns]"),
        ("timedelta64[ns]", operator.sub, "datetime64[ns]"),
        ("timedelta64[ns]", roperator.rsub, "timedelta64[ns]"),
    ],
)
def test_nat_arithmetic_ndarray(dtype, op, out_dtype):
    other = np.arange(10).astype(dtype)
    result = op(NaT, other)

    expected = np.empty(other.shape, dtype=out_dtype)
    expected.fill("NaT")
    tm.assert_numpy_array_equal(result, expected)


def test_nat_pinned_docstrings():
    # see gh-17327
    assert NaT.ctime.__doc__ == Timestamp.ctime.__doc__


def test_to_numpy_alias():
    # GH 24653: alias .to_numpy() for scalars
    expected = NaT.to_datetime64()
    result = NaT.to_numpy()

    assert isna(expected) and isna(result)

    # GH#44460
    result = NaT.to_numpy("M8[s]")
    assert isinstance(result, np.datetime64)
    assert result.dtype == "M8[s]"

    result = NaT.to_numpy("m8[ns]")
    assert isinstance(result, np.timedelta64)
    assert result.dtype == "m8[ns]"

    result = NaT.to_numpy("m8[s]")
    assert isinstance(result, np.timedelta64)
    assert result.dtype == "m8[s]"

    with pytest.raises(ValueError, match="NaT.to_numpy dtype must be a "):
        NaT.to_numpy(np.int64)


@pytest.mark.parametrize(
    "other",
    [
        Timedelta(0),
        Timedelta(0).to_pytimedelta(),
        pytest.param(
            Timedelta(0).to_timedelta64(),
            marks=pytest.mark.xfail(
                not np_version_gte1p24p3,
                reason="td64 doesn't return NotImplemented, see numpy#17017",
                # When this xfail is fixed, test_nat_comparisons_numpy
                #  can be removed.
            ),
        ),
        Timestamp(0),
        Timestamp(0).to_pydatetime(),
        pytest.param(
            Timestamp(0).to_datetime64(),
            marks=pytest.mark.xfail(
                not np_version_gte1p24p3,
                reason="dt64 doesn't return NotImplemented, see numpy#17017",
            ),
        ),
        Timestamp(0).tz_localize("UTC"),
        NaT,
    ],
)
def test_nat_comparisons(compare_operators_no_eq_ne, other):
    # GH 26039
    opname = compare_operators_no_eq_ne

    assert getattr(NaT, opname)(other) is False

    op = getattr(operator, opname.strip("_"))
    assert op(NaT, other) is False
    assert op(other, NaT) is False


@pytest.mark.parametrize("other", [np.timedelta64(0, "ns"), np.datetime64("now", "ns")])
def test_nat_comparisons_numpy(other):
    # Once numpy#17017 is fixed and the xfailed cases in test_nat_comparisons
    #  pass, this test can be removed
    assert not NaT == other
    assert NaT != other
    assert not NaT < other
    assert not NaT > other
    assert not NaT <= other
    assert not NaT >= other


@pytest.mark.parametrize("other_and_type", [("foo", "str"), (2, "int"), (2.0, "float")])
@pytest.mark.parametrize(
    "symbol_and_op",
    [("<=", operator.le), ("<", operator.lt), (">=", operator.ge), (">", operator.gt)],
)
def test_nat_comparisons_invalid(other_and_type, symbol_and_op):
    # GH#35585
    other, other_type = other_and_type
    symbol, op = symbol_and_op

    assert not NaT == other
    assert not other == NaT

    assert NaT != other
    assert other != NaT

    msg = f"'{symbol}' not supported between instances of 'NaTType' and '{other_type}'"
    with pytest.raises(TypeError, match=msg):
        op(NaT, other)

    msg = f"'{symbol}' not supported between instances of '{other_type}' and 'NaTType'"
    with pytest.raises(TypeError, match=msg):
        op(other, NaT)


@pytest.mark.parametrize(
    "other",
    [
        np.array(["foo"] * 2, dtype=object),
        np.array([2, 3], dtype="int64"),
        np.array([2.0, 3.5], dtype="float64"),
    ],
    ids=["str", "int", "float"],
)
def test_nat_comparisons_invalid_ndarray(other):
    # GH#40722
    expected = np.array([False, False])
    result = NaT == other
    tm.assert_numpy_array_equal(result, expected)
    result = other == NaT
    tm.assert_numpy_array_equal(result, expected)

    expected = np.array([True, True])
    result = NaT != other
    tm.assert_numpy_array_equal(result, expected)
    result = other != NaT
    tm.assert_numpy_array_equal(result, expected)

    for symbol, op in [
        ("<=", operator.le),
        ("<", operator.lt),
        (">=", operator.ge),
        (">", operator.gt),
    ]:
        msg = f"'{symbol}' not supported between"

        with pytest.raises(TypeError, match=msg):
            op(NaT, other)

        if other.dtype == np.dtype("object"):
            # uses the reverse operator, so symbol changes
            msg = None
        with pytest.raises(TypeError, match=msg):
            op(other, NaT)


def test_compare_date(fixed_now_ts):
    # GH#39151 comparing NaT with date object is deprecated
    # See also: tests.scalar.timestamps.test_comparisons::test_compare_date

    dt = fixed_now_ts.to_pydatetime().date()

    msg = "Cannot compare NaT with datetime.date object"
    for left, right in [(NaT, dt), (dt, NaT)]:
        assert not left == right
        assert left != right

        with pytest.raises(TypeError, match=msg):
            left < right
        with pytest.raises(TypeError, match=msg):
            left <= right
        with pytest.raises(TypeError, match=msg):
            left > right
        with pytest.raises(TypeError, match=msg):
            left >= right


@pytest.mark.parametrize(
    "obj",
    [
        offsets.YearEnd(2),
        offsets.YearBegin(2),
        offsets.MonthBegin(1),
        offsets.MonthEnd(2),
        offsets.MonthEnd(12),
        offsets.Day(2),
        offsets.Day(5),
        offsets.Hour(24),
        offsets.Hour(3),
        offsets.Minute(),
        np.timedelta64(3, "h"),
        np.timedelta64(4, "h"),
        np.timedelta64(3200, "s"),
        np.timedelta64(3600, "s"),
        np.timedelta64(3600 * 24, "s"),
        np.timedelta64(2, "D"),
        np.timedelta64(365, "D"),
        timedelta(-2),
        timedelta(365),
        timedelta(minutes=120),
        timedelta(days=4, minutes=180),
        timedelta(hours=23),
        timedelta(hours=23, minutes=30),
        timedelta(hours=48),
    ],
)
def test_nat_addsub_tdlike_scalar(obj):
    assert NaT + obj is NaT
    assert obj + NaT is NaT
    assert NaT - obj is NaT


def test_pickle():
    # GH#4606
    p = tm.round_trip_pickle(NaT)
    assert p is NaT

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\_utils.py ===
import warnings


def _raise_warning(attr: str, submodule: str | None = None) -> None:
    new_module = "numpy._core"
    old_module = "numpy.core"
    if submodule is not None:
        new_module = f"{new_module}.{submodule}"
        old_module = f"{old_module}.{submodule}"
    warnings.warn(
        f"{old_module} is deprecated and has been renamed to {new_module}. "
        "The numpy._core namespace contains private NumPy internals and its "
        "use is discouraged, as NumPy internals can change without warning in "
        "any release. In practice, most real-world usage of numpy.core is to "
        "access functionality in the public NumPy API. If that is the case, "
        "use the public NumPy API. If not, you are using NumPy internals. "
        "If you would still like to access an internal attribute, "
        f"use {new_module}.{attr}.",
        DeprecationWarning,
        stacklevel=3
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\comments\author.py ===
# Copyright (c) 2010-2024 openpyxl


from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Sequence,
    Alias
)


class AuthorList(Serialisable):

    tagname = "authors"

    author = Sequence(expected_type=str)
    authors = Alias("author")

    def __init__(self,
                 author=(),
                ):
        self.author = author

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\distributions\__init__.py ===
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.distributions.sdist import SourceDistribution
from pip._internal.distributions.wheel import WheelDistribution
from pip._internal.req.req_install import InstallRequirement


def make_distribution_for_install_requirement(
    install_req: InstallRequirement,
) -> AbstractDistribution:
    """Returns a Distribution for the given InstallRequirement"""
    # Editable requirements will always be source distributions. They use the
    # legacy logic until we create a modern standard for them.
    if install_req.editable:
        return SourceDistribution(install_req)

    # If it's a wheel, it's a WheelDistribution
    if install_req.is_wheel:
        return WheelDistribution(install_req)

    # Otherwise, a SourceDistribution
    return SourceDistribution(install_req)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\platformdirs\version.py ===
# file generated by setuptools-scm
# don't change, don't track in version control

__all__ = ["__version__", "__version_tuple__", "version", "version_tuple"]

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Tuple
    from typing import Union

    VERSION_TUPLE = Tuple[Union[int, str], ...]
else:
    VERSION_TUPLE = object

version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE

__version__ = version = '4.3.7'
__version_tuple__ = version_tuple = (4, 3, 7)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\__init__.py ===
"""This is a subpackage because the directory is on sys.path for _in_process.py

The subpackage should stay as empty as possible to avoid shadowing modules that
the backend might import.
"""

import importlib.resources as resources

try:
    resources.files
except AttributeError:
    # Python 3.8 compatibility
    def _in_proc_script_path():
        return resources.path(__package__, "_in_process.py")

else:

    def _in_proc_script_path():
        return resources.as_file(
            resources.files(__package__).joinpath("_in_process.py")
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\support\ui.py ===
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

from .select import Select
from .wait import WebDriverWait

__all__ = ["Select", "WebDriverWait"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\asyncio\exc.py ===
# ext/asyncio/exc.py
# Copyright (C) 2020-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from ... import exc


class AsyncMethodRequired(exc.InvalidRequestError):
    """an API can't be used because its result would not be
    compatible with async"""


class AsyncContextNotStarted(exc.InvalidRequestError):
    """a startable context manager has not been started."""


class AsyncContextAlreadyStarted(exc.InvalidRequestError):
    """a startable context manager is already started."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\this.py ===
"""
The Zen of SymPy.
"""

s = """The Zen of SymPy

Unevaluated is better than evaluated.
The user interface matters.
Printing matters.
Pure Python can be fast enough.
If it's too slow, it's (probably) your fault.
Documentation matters.
Correctness is more important than speed.
Push it in now and improve upon it later.
Coverage by testing matters.
Smart tests are better than random tests.
But random tests sometimes find what your smartest test missed.
The Python way is probably the right way.
Community is more important than code."""

print(s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\core.py ===
""" The core's core. """
from __future__ import annotations


class Registry:
    """
    Base class for registry objects.

    Registries map a name to an object using attribute notation. Registry
    classes behave singletonically: all their instances share the same state,
    which is stored in the class object.

    All subclasses should set `__slots__ = ()`.
    """
    __slots__ = ()

    def __setattr__(self, name, obj):
        setattr(self.__class__, name, obj)

    def __delattr__(self, name):
        delattr(self.__class__, name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\benchmarks\bench_integrate.py ===
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate

x = Symbol('x')


def bench_integrate_sin():
    integrate(sin(x), x)


def bench_integrate_x1sin():
    integrate(x**1*sin(x), x)


def bench_integrate_x2sin():
    integrate(x**2*sin(x), x)


def bench_integrate_x3sin():
    integrate(x**3*sin(x), x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\benchmarks\bench_matrix.py ===
from sympy.core.numbers import Integer
from sympy.matrices.dense import (eye, zeros)

i3 = Integer(3)
M = eye(100)


def timeit_Matrix__getitem_ii():
    M[3, 3]


def timeit_Matrix__getitem_II():
    M[i3, i3]


def timeit_Matrix__getslice():
    M[:, :]


def timeit_Matrix_zeronm():
    zeros(100, 100)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio_websocket\__init__.py ===
# pylint: disable=useless-import-alias
from ._impl import (
    CloseReason as CloseReason,
    ConnectionClosed as ConnectionClosed,
    ConnectionRejected as ConnectionRejected,
    ConnectionTimeout as ConnectionTimeout,
    connect_websocket as connect_websocket,
    connect_websocket_url as connect_websocket_url,
    DisconnectionTimeout as DisconnectionTimeout,
    Endpoint as Endpoint,
    HandshakeError as HandshakeError,
    open_websocket as open_websocket,
    open_websocket_url as open_websocket_url,
    WebSocketConnection as WebSocketConnection,
    WebSocketRequest as WebSocketRequest,
    WebSocketServer as WebSocketServer,
    wrap_client_stream as wrap_client_stream,
    wrap_server_stream as wrap_server_stream,
    serve_websocket as serve_websocket,
)
from ._version import __version__ as __version__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\_version.py ===
# file generated by setuptools-scm
# don't change, don't track in version control

__all__ = ["__version__", "__version_tuple__", "version", "version_tuple"]

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Tuple
    from typing import Union

    VERSION_TUPLE = Tuple[Union[int, str], ...]
else:
    VERSION_TUPLE = object

version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE

__version__ = version = '2.4.0'
__version_tuple__ = version_tuple = (2, 4, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_partial.py ===
"""
test setting *parts* of objects both positionally and label based

TODO: these should be split among the indexer tests
"""

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Period,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm


class TestEmptyFrameSetitemExpansion:
    def test_empty_frame_setitem_index_name_retained(self):
        # GH#31368 empty frame has non-None index.name -> retained
        df = DataFrame({}, index=pd.RangeIndex(0, name="df_index"))
        series = Series(1.23, index=pd.RangeIndex(4, name="series_index"))

        df["series"] = series
        expected = DataFrame(
            {"series": [1.23] * 4},
            index=pd.RangeIndex(4, name="df_index"),
            columns=Index(["series"], dtype=object),
        )

        tm.assert_frame_equal(df, expected)

    def test_empty_frame_setitem_index_name_inherited(self):
        # GH#36527 empty frame has None index.name -> not retained
        df = DataFrame()
        series = Series(1.23, index=pd.RangeIndex(4, name="series_index"))
        df["series"] = series
        expected = DataFrame(
            {"series": [1.23] * 4},
            index=pd.RangeIndex(4, name="series_index"),
            columns=Index(["series"], dtype=object),
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_zerolen_series_columns_align(self):
        # columns will align
        df = DataFrame(columns=["A", "B"])
        df.loc[0] = Series(1, index=range(4))
        expected = DataFrame(columns=["A", "B"], index=[0], dtype=np.float64)
        tm.assert_frame_equal(df, expected)

        # columns will align
        df = DataFrame(columns=["A", "B"])
        df.loc[0] = Series(1, index=["B"])

        exp = DataFrame([[np.nan, 1]], columns=["A", "B"], index=[0], dtype="float64")
        tm.assert_frame_equal(df, exp)

    def test_loc_setitem_zerolen_list_length_must_match_columns(self):
        # list-like must conform
        df = DataFrame(columns=["A", "B"])

        msg = "cannot set a row with mismatched columns"
        with pytest.raises(ValueError, match=msg):
            df.loc[0] = [1, 2, 3]

        df = DataFrame(columns=["A", "B"])
        df.loc[3] = [6, 7]  # length matches len(df.columns) --> OK!

        exp = DataFrame([[6, 7]], index=[3], columns=["A", "B"], dtype=np.int64)
        tm.assert_frame_equal(df, exp)

    def test_partial_set_empty_frame(self):
        # partially set with an empty object
        # frame
        df = DataFrame()

        msg = "cannot set a frame with no defined columns"

        with pytest.raises(ValueError, match=msg):
            df.loc[1] = 1

        with pytest.raises(ValueError, match=msg):
            df.loc[1] = Series([1], index=["foo"])

        msg = "cannot set a frame with no defined index and a scalar"
        with pytest.raises(ValueError, match=msg):
            df.loc[:, 1] = 1

    def test_partial_set_empty_frame2(self):
        # these work as they don't really change
        # anything but the index
        # GH#5632
        expected = DataFrame(
            columns=Index(["foo"], dtype=object), index=Index([], dtype="object")
        )

        df = DataFrame(index=Index([], dtype="object"))
        df["foo"] = Series([], dtype="object")

        tm.assert_frame_equal(df, expected)

        df = DataFrame(index=Index([]))
        df["foo"] = Series(df.index)

        tm.assert_frame_equal(df, expected)

        df = DataFrame(index=Index([]))
        df["foo"] = df.index

        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame3(self):
        expected = DataFrame(
            columns=Index(["foo"], dtype=object), index=Index([], dtype="int64")
        )
        expected["foo"] = expected["foo"].astype("float64")

        df = DataFrame(index=Index([], dtype="int64"))
        df["foo"] = []

        tm.assert_frame_equal(df, expected)

        df = DataFrame(index=Index([], dtype="int64"))
        df["foo"] = Series(np.arange(len(df)), dtype="float64")

        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame4(self):
        df = DataFrame(index=Index([], dtype="int64"))
        df["foo"] = range(len(df))

        expected = DataFrame(
            columns=Index(["foo"], dtype=object), index=Index([], dtype="int64")
        )
        # range is int-dtype-like, so we get int64 dtype
        expected["foo"] = expected["foo"].astype("int64")
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame5(self):
        df = DataFrame()
        tm.assert_index_equal(df.columns, pd.RangeIndex(0))
        df2 = DataFrame()
        df2[1] = Series([1], index=["foo"])
        df.loc[:, 1] = Series([1], index=["foo"])
        tm.assert_frame_equal(df, DataFrame([[1]], index=["foo"], columns=[1]))
        tm.assert_frame_equal(df, df2)

    def test_partial_set_empty_frame_no_index(self):
        # no index to start
        expected = DataFrame({0: Series(1, index=range(4))}, columns=["A", "B", 0])

        df = DataFrame(columns=["A", "B"])
        df[0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)

        df = DataFrame(columns=["A", "B"])
        df.loc[:, 0] = Series(1, index=range(4))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_row(self):
        # GH#5720, GH#5744
        # don't create rows when empty
        expected = DataFrame(columns=["A", "B", "New"], index=Index([], dtype="int64"))
        expected["A"] = expected["A"].astype("int64")
        expected["B"] = expected["B"].astype("float64")
        expected["New"] = expected["New"].astype("float64")

        df = DataFrame({"A": [1, 2, 3], "B": [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        y["New"] = np.nan
        tm.assert_frame_equal(y, expected)

        expected = DataFrame(columns=["a", "b", "c c", "d"])
        expected["d"] = expected["d"].astype("int64")
        df = DataFrame(columns=["a", "b", "c c"])
        df["d"] = 3
        tm.assert_frame_equal(df, expected)
        tm.assert_series_equal(df["c c"], Series(name="c c", dtype=object))

        # reindex columns is ok
        df = DataFrame({"A": [1, 2, 3], "B": [1.2, 4.2, 5.2]})
        y = df[df.A > 5]
        result = y.reindex(columns=["A", "B", "C"])
        expected = DataFrame(columns=["A", "B", "C"])
        expected["A"] = expected["A"].astype("int64")
        expected["B"] = expected["B"].astype("float64")
        expected["C"] = expected["C"].astype("float64")
        tm.assert_frame_equal(result, expected)

    def test_partial_set_empty_frame_set_series(self):
        # GH#5756
        # setting with empty Series
        df = DataFrame(Series(dtype=object))
        expected = DataFrame({0: Series(dtype=object)})
        tm.assert_frame_equal(df, expected)

        df = DataFrame(Series(name="foo", dtype=object))
        expected = DataFrame({"foo": Series(dtype=object)})
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_copy_assignment(self):
        # GH#5932
        # copy on empty with assignment fails
        df = DataFrame(index=[0])
        df = df.copy()
        df["a"] = 0
        expected = DataFrame(0, index=[0], columns=Index(["a"], dtype=object))
        tm.assert_frame_equal(df, expected)

    def test_partial_set_empty_frame_empty_consistencies(self, using_infer_string):
        # GH#6171
        # consistency on empty frames
        df = DataFrame(columns=["x", "y"])
        df["x"] = [1, 2]
        expected = DataFrame({"x": [1, 2], "y": [np.nan, np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)

        df = DataFrame(columns=["x", "y"])
        df["x"] = ["1", "2"]
        expected = DataFrame(
            {
                "x": Series(
                    ["1", "2"],
                    dtype=object if not using_infer_string else "str",
                ),
                "y": Series([np.nan, np.nan], dtype=object),
            }
        )
        tm.assert_frame_equal(df, expected)

        df = DataFrame(columns=["x", "y"])
        df.loc[0, "x"] = 1
        expected = DataFrame({"x": [1], "y": [np.nan]})
        tm.assert_frame_equal(df, expected, check_dtype=False)


class TestPartialSetting:
    def test_partial_setting(self):
        # GH2578, allow ix and friends to partially set

        # series
        s_orig = Series([1, 2, 3])

        s = s_orig.copy()
        s[5] = 5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s.loc[5] = 5
        expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s[5] = 5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s.loc[5] = 5.0
        expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
        tm.assert_series_equal(s, expected)

        # iloc/iat raise
        s = s_orig.copy()

        msg = "iloc cannot enlarge its target object"
        with pytest.raises(IndexError, match=msg):
            s.iloc[3] = 5.0

        msg = "index 3 is out of bounds for axis 0 with size 3"
        with pytest.raises(IndexError, match=msg):
            s.iat[3] = 5.0

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_partial_setting_frame(self, using_array_manager):
        df_orig = DataFrame(
            np.arange(6).reshape(3, 2), columns=["A", "B"], dtype="int64"
        )

        # iloc/iat raise
        df = df_orig.copy()

        msg = "iloc cannot enlarge its target object"
        with pytest.raises(IndexError, match=msg):
            df.iloc[4, 2] = 5.0

        msg = "index 2 is out of bounds for axis 0 with size 2"
        if using_array_manager:
            msg = "list index out of range"
        with pytest.raises(IndexError, match=msg):
            df.iat[4, 2] = 5.0

        # row setting where it exists
        expected = DataFrame({"A": [0, 4, 4], "B": [1, 5, 5]})
        df = df_orig.copy()
        df.iloc[1] = df.iloc[2]
        tm.assert_frame_equal(df, expected)

        expected = DataFrame({"A": [0, 4, 4], "B": [1, 5, 5]})
        df = df_orig.copy()
        df.loc[1] = df.loc[2]
        tm.assert_frame_equal(df, expected)

        # like 2578, partial setting with dtype preservation
        expected = DataFrame({"A": [0, 2, 4, 4], "B": [1, 3, 5, 5]})
        df = df_orig.copy()
        df.loc[3] = df.loc[2]
        tm.assert_frame_equal(df, expected)

        # single dtype frame, overwrite
        expected = DataFrame({"A": [0, 2, 4], "B": [0, 2, 4]})
        df = df_orig.copy()
        df.loc[:, "B"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

        # mixed dtype frame, overwrite
        expected = DataFrame({"A": [0, 2, 4], "B": Series([0.0, 2.0, 4.0])})
        df = df_orig.copy()
        df["B"] = df["B"].astype(np.float64)
        # as of 2.0, df.loc[:, "B"] = ... attempts (and here succeeds) at
        #  setting inplace
        df.loc[:, "B"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

        # single dtype frame, partial setting
        expected = df_orig.copy()
        expected["C"] = df["A"]
        df = df_orig.copy()
        df.loc[:, "C"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

        # mixed frame, partial setting
        expected = df_orig.copy()
        expected["C"] = df["A"]
        df = df_orig.copy()
        df.loc[:, "C"] = df.loc[:, "A"]
        tm.assert_frame_equal(df, expected)

    def test_partial_setting2(self):
        # GH 8473
        dates = date_range("1/1/2000", periods=8)
        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=dates,
            columns=["A", "B", "C", "D"],
        )

        expected = pd.concat(
            [df_orig, DataFrame({"A": 7}, index=dates[-1:] + dates.freq)], sort=True
        )
        df = df_orig.copy()
        df.loc[dates[-1] + dates.freq, "A"] = 7
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.at[dates[-1] + dates.freq, "A"] = 7
        tm.assert_frame_equal(df, expected)

        exp_other = DataFrame({0: 7}, index=dates[-1:] + dates.freq)
        expected = pd.concat([df_orig, exp_other], axis=1)

        df = df_orig.copy()
        df.loc[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)
        df = df_orig.copy()
        df.at[dates[-1] + dates.freq, 0] = 7
        tm.assert_frame_equal(df, expected)

    def test_partial_setting_mixed_dtype(self):
        # in a mixed dtype environment, try to preserve dtypes
        # by appending
        df = DataFrame([[True, 1], [False, 2]], columns=["female", "fitness"])

        s = df.loc[1].copy()
        s.name = 2
        expected = pd.concat([df, DataFrame(s).T.infer_objects()])

        df.loc[2] = df.loc[1]
        tm.assert_frame_equal(df, expected)

    def test_series_partial_set(self):
        # partial set with new index
        # Regression from GH4825
        ser = Series([0.1, 0.2], index=[1, 2])

        # loc equiv to .reindex
        expected = Series([np.nan, 0.2, np.nan], index=[3, 2, 3])
        with pytest.raises(KeyError, match=r"not in index"):
            ser.loc[[3, 2, 3]]

        result = ser.reindex([3, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)

        expected = Series([np.nan, 0.2, np.nan, np.nan], index=[3, 2, 3, "x"])
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[3, 2, 3, "x"]]

        result = ser.reindex([3, 2, 3, "x"])
        tm.assert_series_equal(result, expected, check_index_type=True)

        expected = Series([0.2, 0.2, 0.1], index=[2, 2, 1])
        result = ser.loc[[2, 2, 1]]
        tm.assert_series_equal(result, expected, check_index_type=True)

        expected = Series([0.2, 0.2, np.nan, 0.1], index=[2, 2, "x", 1])
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[2, 2, "x", 1]]

        result = ser.reindex([2, 2, "x", 1])
        tm.assert_series_equal(result, expected, check_index_type=True)

        # raises as nothing is in the index
        msg = (
            rf"\"None of \[Index\(\[3, 3, 3\], dtype='{np.dtype(int)}'\)\] "
            r"are in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]

        expected = Series([0.2, 0.2, np.nan], index=[2, 2, 3])
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[2, 2, 3]]

        result = ser.reindex([2, 2, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3], index=[1, 2, 3])
        expected = Series([0.3, np.nan, np.nan], index=[3, 4, 4])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[3, 4, 4]]

        result = s.reindex([3, 4, 4])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([np.nan, 0.3, 0.3], index=[5, 3, 3])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[5, 3, 3]]

        result = s.reindex([5, 3, 3])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([np.nan, 0.4, 0.4], index=[5, 4, 4])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[5, 4, 4]]

        result = s.reindex([5, 4, 4])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[4, 5, 6, 7])
        expected = Series([0.4, np.nan, np.nan], index=[7, 2, 2])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[7, 2, 2]]

        result = s.reindex([7, 2, 2])
        tm.assert_series_equal(result, expected, check_index_type=True)

        s = Series([0.1, 0.2, 0.3, 0.4], index=[1, 2, 3, 4])
        expected = Series([0.4, np.nan, np.nan], index=[4, 5, 5])
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[4, 5, 5]]

        result = s.reindex([4, 5, 5])
        tm.assert_series_equal(result, expected, check_index_type=True)

        # iloc
        expected = Series([0.2, 0.2, 0.1, 0.1], index=[2, 2, 1, 1])
        result = ser.iloc[[1, 1, 0, 0]]
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_series_partial_set_with_name(self):
        # GH 11497

        idx = Index([1, 2], dtype="int64", name="idx")
        ser = Series([0.1, 0.2], index=idx, name="s")

        # loc
        with pytest.raises(KeyError, match=r"\[3\] not in index"):
            ser.loc[[3, 2, 3]]

        with pytest.raises(KeyError, match=r"not in index"):
            ser.loc[[3, 2, 3, "x"]]

        exp_idx = Index([2, 2, 1], dtype="int64", name="idx")
        expected = Series([0.2, 0.2, 0.1], index=exp_idx, name="s")
        result = ser.loc[[2, 2, 1]]
        tm.assert_series_equal(result, expected, check_index_type=True)

        with pytest.raises(KeyError, match=r"\['x'\] not in index"):
            ser.loc[[2, 2, "x", 1]]

        # raises as nothing is in the index
        msg = (
            rf"\"None of \[Index\(\[3, 3, 3\], dtype='{np.dtype(int)}', "
            r"name='idx'\)\] are in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            ser.loc[[3, 3, 3]]

        with pytest.raises(KeyError, match="not in index"):
            ser.loc[[2, 2, 3]]

        idx = Index([1, 2, 3], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3], index=idx, name="s").loc[[3, 4, 4]]

        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[5, 3, 3]]

        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[5, 4, 4]]

        idx = Index([4, 5, 6, 7], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[7, 2, 2]]

        idx = Index([1, 2, 3, 4], dtype="int64", name="idx")
        with pytest.raises(KeyError, match="not in index"):
            Series([0.1, 0.2, 0.3, 0.4], index=idx, name="s").loc[[4, 5, 5]]

        # iloc
        exp_idx = Index([2, 2, 1, 1], dtype="int64", name="idx")
        expected = Series([0.2, 0.2, 0.1, 0.1], index=exp_idx, name="s")
        result = ser.iloc[[1, 1, 0, 0]]
        tm.assert_series_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize("key", [100, 100.0])
    def test_setitem_with_expansion_numeric_into_datetimeindex(self, key):
        # GH#4940 inserting non-strings
        orig = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df = orig.copy()

        df.loc[key, :] = df.iloc[0]
        ex_index = Index(list(orig.index) + [key], dtype=object, name=orig.index.name)
        ex_data = np.concatenate([orig.values, df.iloc[[0]].values], axis=0)
        expected = DataFrame(ex_data, index=ex_index, columns=orig.columns)

        tm.assert_frame_equal(df, expected)

    def test_partial_set_invalid(self):
        # GH 4940
        # allow only setting of 'valid' values

        orig = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )

        # allow object conversion here
        df = orig.copy()
        df.loc["a", :] = df.iloc[0]
        ser = Series(df.iloc[0], name="a")
        exp = pd.concat([orig, DataFrame(ser).T.infer_objects()])
        tm.assert_frame_equal(df, exp)
        tm.assert_index_equal(df.index, Index(orig.index.tolist() + ["a"]))
        assert df.index.dtype == "object"

    @pytest.mark.parametrize(
        "idx,labels,expected_idx",
        [
            (
                period_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-08", "2000-01-12"],
                [
                    Period("2000-01-04", freq="D"),
                    Period("2000-01-08", freq="D"),
                    Period("2000-01-12", freq="D"),
                ],
            ),
            (
                date_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-08", "2000-01-12"],
                [
                    Timestamp("2000-01-04"),
                    Timestamp("2000-01-08"),
                    Timestamp("2000-01-12"),
                ],
            ),
            (
                pd.timedelta_range(start="1 day", periods=20),
                ["4D", "8D", "12D"],
                [pd.Timedelta("4 day"), pd.Timedelta("8 day"), pd.Timedelta("12 day")],
            ),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes(
        self, idx, labels, expected_idx, frame_or_series
    ):
        # GH 11278
        obj = frame_or_series(range(20), index=idx)

        expected_value = [3, 7, 11]
        expected = frame_or_series(expected_value, expected_idx)

        tm.assert_equal(expected, obj.loc[labels])
        if frame_or_series is Series:
            tm.assert_series_equal(expected, obj[labels])

    @pytest.mark.parametrize(
        "idx,labels",
        [
            (
                period_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-30"],
            ),
            (
                date_range(start="2000", periods=20, freq="D"),
                ["2000-01-04", "2000-01-30"],
            ),
            (pd.timedelta_range(start="1 day", periods=20), ["3 day", "30 day"]),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes_missing_value(
        self, idx, labels
    ):
        # GH 11278
        ser = Series(range(20), index=idx)
        df = DataFrame(range(20), index=idx)
        msg = r"not in index"

        with pytest.raises(KeyError, match=msg):
            ser.loc[labels]
        with pytest.raises(KeyError, match=msg):
            ser[labels]
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]

    @pytest.mark.parametrize(
        "idx,labels,msg",
        [
            (
                period_range(start="2000", periods=20, freq="D"),
                Index(["4D", "8D"], dtype=object),
                (
                    r"None of \[Index\(\['4D', '8D'\], dtype='object'\)\] "
                    r"are in the \[index\]"
                ),
            ),
            (
                date_range(start="2000", periods=20, freq="D"),
                Index(["4D", "8D"], dtype=object),
                (
                    r"None of \[Index\(\['4D', '8D'\], dtype='object'\)\] "
                    r"are in the \[index\]"
                ),
            ),
            (
                pd.timedelta_range(start="1 day", periods=20),
                Index(["2000-01-04", "2000-01-08"], dtype=object),
                (
                    r"None of \[Index\(\['2000-01-04', '2000-01-08'\], "
                    r"dtype='object'\)\] are in the \[index\]"
                ),
            ),
        ],
    )
    def test_loc_with_list_of_strings_representing_datetimes_not_matched_type(
        self, idx, labels, msg
    ):
        # GH 11278
        ser = Series(range(20), index=idx)
        df = DataFrame(range(20), index=idx)

        with pytest.raises(KeyError, match=msg):
            ser.loc[labels]
        with pytest.raises(KeyError, match=msg):
            ser[labels]
        with pytest.raises(KeyError, match=msg):
            df.loc[labels]


class TestStringSlicing:
    def test_slice_irregular_datetime_index_with_nan(self):
        # GH36953
        index = pd.to_datetime(["2012-01-01", "2012-01-02", "2012-01-03", None])
        df = DataFrame(range(len(index)), index=index)
        expected = DataFrame(range(len(index[:3])), index=index[:3])
        with pytest.raises(KeyError, match="non-existing keys is not allowed"):
            # Upper bound is not in index (which is unordered)
            # GH53983
            # GH37819
            df["2012-01-01":"2012-01-04"]
        # Need this precision for right bound since the right slice
        # bound is "rounded" up to the largest timepoint smaller than
        # the next "resolution"-step of the provided point.
        # e.g. 2012-01-03 is rounded up to 2012-01-04 - 1ns
        result = df["2012-01-01":"2012-01-03 00:00:00.000000000"]
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_factor_.py ===
from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial as fac
from sympy.core.numbers import Integer, Rational
from sympy.external.gmpy import gcd

from sympy.ntheory import (totient,
    factorint, primefactors, divisors, nextprime,
    pollard_rho, perfect_power, multiplicity, multiplicity_in_factorial,
    divisor_count, primorial, pollard_pm1, divisor_sigma,
    factorrat, reduced_totient)
from sympy.ntheory.factor_ import (smoothness, smoothness_p, proper_divisors,
    antidivisors, antidivisor_count, _divisor_sigma, core, udivisors, udivisor_sigma,
    udivisor_count, proper_divisor_count, primenu, primeomega,
    mersenne_prime_exponent, is_perfect, is_abundant,
    is_deficient, is_amicable, is_carmichael, find_carmichael_numbers_in_range,
    find_first_n_carmichaels, dra, drm, _perfect_power, factor_cache)

from sympy.testing.pytest import raises, slow

from sympy.utilities.iterables import capture


def fac_multiplicity(n, p):
    """Return the power of the prime number p in the
    factorization of n!"""
    if p > n:
        return 0
    if p > n//2:
        return 1
    q, m = n, 0
    while q >= p:
        q //= p
        m += q
    return m


def multiproduct(seq=(), start=1):
    """
    Return the product of a sequence of factors with multiplicities,
    times the value of the parameter ``start``. The input may be a
    sequence of (factor, exponent) pairs or a dict of such pairs.

        >>> multiproduct({3:7, 2:5}, 4) # = 3**7 * 2**5 * 4
        279936

    """
    if not seq:
        return start
    if isinstance(seq, dict):
        seq = iter(seq.items())
    units = start
    multi = []
    for base, exp in seq:
        if not exp:
            continue
        elif exp == 1:
            units *= base
        else:
            if exp % 2:
                units *= base
            multi.append((base, exp//2))
    return units * multiproduct(multi)**2


def test_multiplicity():
    for b in range(2, 20):
        for i in range(100):
            assert multiplicity(b, b**i) == i
            assert multiplicity(b, (b**i) * 23) == i
            assert multiplicity(b, (b**i) * 1000249) == i
    # Should be fast
    assert multiplicity(10, 10**10023) == 10023
    # Should exit quickly
    assert multiplicity(10**10, 10**10) == 1
    # Should raise errors for bad input
    raises(ValueError, lambda: multiplicity(1, 1))
    raises(ValueError, lambda: multiplicity(1, 2))
    raises(ValueError, lambda: multiplicity(1.3, 2))
    raises(ValueError, lambda: multiplicity(2, 0))
    raises(ValueError, lambda: multiplicity(1.3, 0))

    # handles Rationals
    assert multiplicity(10, Rational(30, 7)) == 1
    assert multiplicity(Rational(2, 7), Rational(4, 7)) == 1
    assert multiplicity(Rational(1, 7), Rational(3, 49)) == 2
    assert multiplicity(Rational(2, 7), Rational(7, 2)) == -1
    assert multiplicity(3, Rational(1, 9)) == -2


def test_multiplicity_in_factorial():
    n = fac(1000)
    for i in (2, 4, 6, 12, 30, 36, 48, 60, 72, 96):
        assert multiplicity(i, n) == multiplicity_in_factorial(i, 1000)


def test_private_perfect_power():
    assert _perfect_power(0) is False
    assert _perfect_power(1) is False
    assert _perfect_power(2) is False
    assert _perfect_power(3) is False
    for x in [2, 3, 5, 6, 7, 12, 15, 105, 100003]:
        for y in range(2, 100):
            assert _perfect_power(x**y) == (x, y)
            if x & 1:
                assert _perfect_power(x**y, next_p=3) == (x, y)
            if x == 100003:
                assert _perfect_power(x**y, next_p=100003) == (x, y)
            assert _perfect_power(101*x**y) == False
            # Catalan's conjecture
            if x**y not in [8, 9]:
                assert _perfect_power(x**y + 1) == False
                assert _perfect_power(x**y - 1) == False
    for x in range(1, 10):
        for y in range(1, 10):
            g = gcd(x, y)
            if g == 1:
                assert _perfect_power(5**x * 101**y) == False
            else:
                assert _perfect_power(5**x * 101**y) == (5**(x//g) * 101**(y//g), g)


def test_perfect_power():
    raises(ValueError, lambda: perfect_power(0.1))
    assert perfect_power(0) is False
    assert perfect_power(1) is False
    assert perfect_power(2) is False
    assert perfect_power(3) is False
    assert perfect_power(4) == (2, 2)
    assert perfect_power(14) is False
    assert perfect_power(25) == (5, 2)
    assert perfect_power(22) is False
    assert perfect_power(22, [2]) is False
    assert perfect_power(137**(3*5*13)) == (137, 3*5*13)
    assert perfect_power(137**(3*5*13) + 1) is False
    assert perfect_power(137**(3*5*13) - 1) is False
    assert perfect_power(103005006004**7) == (103005006004, 7)
    assert perfect_power(103005006004**7 + 1) is False
    assert perfect_power(103005006004**7 - 1) is False
    assert perfect_power(103005006004**12) == (103005006004, 12)
    assert perfect_power(103005006004**12 + 1) is False
    assert perfect_power(103005006004**12 - 1) is False
    assert perfect_power(2**10007) == (2, 10007)
    assert perfect_power(2**10007 + 1) is False
    assert perfect_power(2**10007 - 1) is False
    assert perfect_power((9**99 + 1)**60) == (9**99 + 1, 60)
    assert perfect_power((9**99 + 1)**60 + 1) is False
    assert perfect_power((9**99 + 1)**60 - 1) is False
    assert perfect_power((10**40000)**2, big=False) == (10**40000, 2)
    assert perfect_power(10**100000) == (10, 100000)
    assert perfect_power(10**100001) == (10, 100001)
    assert perfect_power(13**4, [3, 5]) is False
    assert perfect_power(3**4, [3, 10], factor=0) is False
    assert perfect_power(3**3*5**3) == (15, 3)
    assert perfect_power(2**3*5**5) is False
    assert perfect_power(2*13**4) is False
    assert perfect_power(2**5*3**3) is False
    t = 2**24
    for d in divisors(24):
        m = perfect_power(t*3**d)
        assert m and m[1] == d or d == 1
        m = perfect_power(t*3**d, big=False)
        assert m and m[1] == 2 or d == 1 or d == 3, (d, m)

    # negatives and non-integer rationals
    assert perfect_power(-4) is False
    assert perfect_power(-8) == (-2, 3)
    assert perfect_power(-S(1)/8) == (-S(1)/2, 3)
    assert perfect_power(S(1)/3) == False
    assert perfect_power(-5**15) == (-5, 15)
    assert perfect_power(-5**15, big=False) == (-3125, 3)
    assert perfect_power(-5**15, [15]) == (-5, 15)

    n = -3 ** 60
    assert perfect_power(n) == (-81, 15)
    assert perfect_power(n, big=False) == (-3486784401, 3)
    assert perfect_power(n, [3, 5], big=True) == (-531441, 5)
    assert perfect_power(n, [3, 5], big=False) == (-3486784401, 3)
    assert perfect_power(n, [2]) == False
    assert perfect_power(n, [2, 15]) == (-81, 15)
    assert perfect_power(n, [2, 13]) == False
    assert perfect_power(n, [17]) == False
    assert perfect_power(n, [3]) == (-3486784401, 3)
    assert perfect_power(n + 1) == False

    r = S(2) ** (2 * 5 * 7) / S(3) ** (2 * 7)
    assert perfect_power(r) == (S(32) / 3, 14)
    assert perfect_power(-r) == (-S(1024) / 9, 7)
    assert perfect_power(r, big=False) == (S(34359738368) / 2187, 2)
    assert perfect_power(r, [2, 5]) == (S(34359738368) / 2187, 2)
    assert perfect_power(r, [5, 7]) == (S(1024) / 9, 7)
    assert perfect_power(r, [5, 7], big=False) == (S(1024) / 9, 7)
    assert perfect_power(r, [2, 5, 7], big=False) == (S(34359738368) / 2187, 2)
    assert perfect_power(-r, [5, 7], big=False) == (-S(1024) / 9, 7)

    assert perfect_power(-S(1) / 8) == (-S(1) / 2, 3)

    assert perfect_power((-3)**60) == (3, 60)
    assert perfect_power((-3)**61) == (-3, 61)

    assert perfect_power(S(2 ** 9) / 3 ** 12) == (S(8)/81, 3)
    assert perfect_power(Rational(1, 2)**3) == (S.Half, 3)
    assert perfect_power(Rational(-3, 2)**3) == (-3*S.Half, 3)


def test_factor_cache():
    factor_cache.cache_clear()
    raises(ValueError, lambda: factor_cache.__setitem__(1, 5))
    raises(ValueError, lambda: factor_cache.__setitem__(10, 1))
    raises(ValueError, lambda: factor_cache.__setitem__(10, 10))
    raises(ValueError, lambda: factor_cache.__setitem__(10, 3))
    raises(ValueError, lambda: factor_cache.__setitem__(20, 4))
    factor_cache.maxsize = 3
    for i in range(2, 10):
        factor_cache[5*i] = 5
    assert len(factor_cache) == 3
    factor_cache.maxsize = 5
    for i in range(2, 10):
        factor_cache[5*i] = 5
    assert len(factor_cache) == 5
    factor_cache.maxsize = 2
    assert len(factor_cache) == 2
    factor_cache.maxsize =1000

    factor_cache.cache_clear()
    factor_cache[40] = 5
    assert factor_cache.get(40) == 5
    assert factor_cache.get(20) is None
    assert factor_cache[40] == 5
    raises(KeyError, lambda: factor_cache[10])
    del factor_cache[40]
    assert len(factor_cache) == 0
    raises(KeyError, lambda: factor_cache.__delitem__(40))
    factor_cache.add(100, [5, 2])
    assert len(factor_cache) == 2
    assert factor_cache[100] == 5

    for n in [1000000007, 10000019*20000003]:
        factorint(n)
        assert n in factor_cache

    # Restore the initial state
    factor_cache.cache_clear()
    factor_cache.maxsize = 1000


@slow
def test_factorint():
    assert primefactors(123456) == [2, 3, 643]
    assert factorint(0) == {0: 1}
    assert factorint(1) == {}
    assert factorint(-1) == {-1: 1}
    assert factorint(-2) == {-1: 1, 2: 1}
    assert factorint(-16) == {-1: 1, 2: 4}
    assert factorint(2) == {2: 1}
    assert factorint(126) == {2: 1, 3: 2, 7: 1}
    assert factorint(123456) == {2: 6, 3: 1, 643: 1}
    assert factorint(5951757) == {3: 1, 7: 1, 29: 2, 337: 1}
    assert factorint(64015937) == {7993: 1, 8009: 1}
    assert factorint(2**(2**6) + 1) == {274177: 1, 67280421310721: 1}
    #issue 19683
    assert factorint(10**38 - 1) == {3: 2, 11: 1, 909090909090909091: 1, 1111111111111111111: 1}
    #issue 17676
    assert factorint(28300421052393658575) == {3: 1, 5: 2, 11: 2, 43: 1, 2063: 2, 4127: 1, 4129: 1}
    assert factorint(2063**2 * 4127**1 * 4129**1) == {2063: 2, 4127: 1, 4129: 1}
    assert factorint(2347**2 * 7039**1 * 7043**1) == {2347: 2, 7039: 1, 7043: 1}

    assert factorint(0, multiple=True) == [0]
    assert factorint(1, multiple=True) == []
    assert factorint(-1, multiple=True) == [-1]
    assert factorint(-2, multiple=True) == [-1, 2]
    assert factorint(-16, multiple=True) == [-1, 2, 2, 2, 2]
    assert factorint(2, multiple=True) == [2]
    assert factorint(24, multiple=True) == [2, 2, 2, 3]
    assert factorint(126, multiple=True) == [2, 3, 3, 7]
    assert factorint(123456, multiple=True) == [2, 2, 2, 2, 2, 2, 3, 643]
    assert factorint(5951757, multiple=True) == [3, 7, 29, 29, 337]
    assert factorint(64015937, multiple=True) == [7993, 8009]
    assert factorint(2**(2**6) + 1, multiple=True) == [274177, 67280421310721]

    assert factorint(fac(1, evaluate=False)) == {}
    assert factorint(fac(7, evaluate=False)) == {2: 4, 3: 2, 5: 1, 7: 1}
    assert factorint(fac(15, evaluate=False)) == \
        {2: 11, 3: 6, 5: 3, 7: 2, 11: 1, 13: 1}
    assert factorint(fac(20, evaluate=False)) == \
        {2: 18, 3: 8, 5: 4, 7: 2, 11: 1, 13: 1, 17: 1, 19: 1}
    assert factorint(fac(23, evaluate=False)) == \
        {2: 19, 3: 9, 5: 4, 7: 3, 11: 2, 13: 1, 17: 1, 19: 1, 23: 1}

    assert multiproduct(factorint(fac(200))) == fac(200)
    assert multiproduct(factorint(fac(200, evaluate=False))) == fac(200)
    for b, e in factorint(fac(150)).items():
        assert e == fac_multiplicity(150, b)
    for b, e in factorint(fac(150, evaluate=False)).items():
        assert e == fac_multiplicity(150, b)
    assert factorint(103005006059**7) == {103005006059: 7}
    assert factorint(31337**191) == {31337: 191}
    assert factorint(2**1000 * 3**500 * 257**127 * 383**60) == \
        {2: 1000, 3: 500, 257: 127, 383: 60}
    assert len(factorint(fac(10000))) == 1229
    assert len(factorint(fac(10000, evaluate=False))) == 1229
    assert factorint(12932983746293756928584532764589230) == \
        {2: 1, 5: 1, 73: 1, 727719592270351: 1, 63564265087747: 1, 383: 1}
    assert factorint(727719592270351) == {727719592270351: 1}
    assert factorint(2**64 + 1, use_trial=False) == factorint(2**64 + 1)
    for n in range(60000):
        assert multiproduct(factorint(n)) == n
    assert pollard_rho(2**64 + 1, seed=1) == 274177
    assert pollard_rho(19, seed=1) is None
    assert factorint(3, limit=2) == {3: 1}
    assert factorint(12345) == {3: 1, 5: 1, 823: 1}
    assert factorint(
        12345, limit=3) == {4115: 1, 3: 1}  # the 5 is greater than the limit
    assert factorint(1, limit=1) == {}
    assert factorint(0, 3) == {0: 1}
    assert factorint(12, limit=1) == {12: 1}
    assert factorint(30, limit=2) == {2: 1, 15: 1}
    assert factorint(16, limit=2) == {2: 4}
    assert factorint(124, limit=3) == {2: 2, 31: 1}
    assert factorint(4*31**2, limit=3) == {2: 2, 31: 2}
    p1 = nextprime(2**32)
    p2 = nextprime(2**16)
    p3 = nextprime(p2)
    assert factorint(p1*p2*p3) == {p1: 1, p2: 1, p3: 1}
    assert factorint(13*17*19, limit=15) == {13: 1, 17*19: 1}
    assert factorint(1951*15013*15053, limit=2000) == {225990689: 1, 1951: 1}
    assert factorint(primorial(17) + 1, use_pm1=0) == \
        {int(19026377261): 1, 3467: 1, 277: 1, 105229: 1}
    # when prime b is closer than approx sqrt(8*p) to prime p then they are
    # "close" and have a trivial factorization
    a = nextprime(2**2**8)  # 78 digits
    b = nextprime(a + 2**2**4)
    assert 'Fermat' in capture(lambda: factorint(a*b, verbose=1))

    raises(ValueError, lambda: pollard_rho(4))
    raises(ValueError, lambda: pollard_pm1(3))
    raises(ValueError, lambda: pollard_pm1(10, B=2))
    # verbose coverage
    n = nextprime(2**16)*nextprime(2**17)*nextprime(1901)
    assert 'with primes' in capture(lambda: factorint(n, verbose=1))
    capture(lambda: factorint(nextprime(2**16)*1012, verbose=1))

    n = nextprime(2**17)
    capture(lambda: factorint(n**3, verbose=1))  # perfect power termination
    capture(lambda: factorint(2*n, verbose=1))  # factoring complete msg

    # exceed 1st
    n = nextprime(2**17)
    n *= nextprime(n)
    assert '1000' in capture(lambda: factorint(n, limit=1000, verbose=1))
    n *= nextprime(n)
    assert len(factorint(n)) == 3
    assert len(factorint(n, limit=p1)) == 3
    n *= nextprime(2*n)
    # exceed 2nd
    assert '2001' in capture(lambda: factorint(n, limit=2000, verbose=1))
    assert capture(
        lambda: factorint(n, limit=4000, verbose=1)).count('Pollard') == 2
    # non-prime pm1 result
    n = nextprime(8069)
    n *= nextprime(2*n)*nextprime(2*n, 2)
    capture(lambda: factorint(n, verbose=1))  # non-prime pm1 result
    # factor fermat composite
    p1 = nextprime(2**17)
    p2 = nextprime(2*p1)
    assert factorint((p1*p2**2)**3) == {p1: 3, p2: 6}
    # Test for non integer input
    raises(ValueError, lambda: factorint(4.5))
    # test dict/Dict input
    sans = '2**10*3**3'
    n = {4: 2, 12: 3}
    assert str(factorint(n)) == sans
    assert str(factorint(Dict(n))) == sans


def test_divisors_and_divisor_count():
    assert divisors(-1) == [1]
    assert divisors(0) == []
    assert divisors(1) == [1]
    assert divisors(2) == [1, 2]
    assert divisors(3) == [1, 3]
    assert divisors(17) == [1, 17]
    assert divisors(10) == [1, 2, 5, 10]
    assert divisors(100) == [1, 2, 4, 5, 10, 20, 25, 50, 100]
    assert divisors(101) == [1, 101]
    assert type(divisors(2, generator=True)) is not list

    assert divisor_count(0) == 0
    assert divisor_count(-1) == 1
    assert divisor_count(1) == 1
    assert divisor_count(6) == 4
    assert divisor_count(12) == 6

    assert divisor_count(180, 3) == divisor_count(180//3)
    assert divisor_count(2*3*5, 7) == 0


def test_proper_divisors_and_proper_divisor_count():
    assert proper_divisors(-1) == []
    assert proper_divisors(0) == []
    assert proper_divisors(1) == []
    assert proper_divisors(2) == [1]
    assert proper_divisors(3) == [1]
    assert proper_divisors(17) == [1]
    assert proper_divisors(10) == [1, 2, 5]
    assert proper_divisors(100) == [1, 2, 4, 5, 10, 20, 25, 50]
    assert proper_divisors(1000000007) == [1]
    assert type(proper_divisors(2, generator=True)) is not list

    assert proper_divisor_count(0) == 0
    assert proper_divisor_count(-1) == 0
    assert proper_divisor_count(1) == 0
    assert proper_divisor_count(36) == 8
    assert proper_divisor_count(2*3*5) == 7


def test_udivisors_and_udivisor_count():
    assert udivisors(-1) == [1]
    assert udivisors(0) == []
    assert udivisors(1) == [1]
    assert udivisors(2) == [1, 2]
    assert udivisors(3) == [1, 3]
    assert udivisors(17) == [1, 17]
    assert udivisors(10) == [1, 2, 5, 10]
    assert udivisors(100) == [1, 4, 25, 100]
    assert udivisors(101) == [1, 101]
    assert udivisors(1000) == [1, 8, 125, 1000]
    assert type(udivisors(2, generator=True)) is not list

    assert udivisor_count(0) == 0
    assert udivisor_count(-1) == 1
    assert udivisor_count(1) == 1
    assert udivisor_count(6) == 4
    assert udivisor_count(12) == 4

    assert udivisor_count(180) == 8
    assert udivisor_count(2*3*5*7) == 16


def test_issue_6981():
    S = set(divisors(4)).union(set(divisors(Integer(2))))
    assert S == {1,2,4}


def test_issue_4356():
    assert factorint(1030903) == {53: 2, 367: 1}


def test_divisors():
    assert divisors(28) == [1, 2, 4, 7, 14, 28]
    assert list(divisors(3*5*7, 1)) == [1, 3, 5, 15, 7, 21, 35, 105]
    assert divisors(0) == []


def test_divisor_count():
    assert divisor_count(0) == 0
    assert divisor_count(6) == 4


def test_proper_divisors():
    assert proper_divisors(-1) == []
    assert proper_divisors(28) == [1, 2, 4, 7, 14]
    assert list(proper_divisors(3*5*7, True)) == [1, 3, 5, 15, 7, 21, 35]


def test_proper_divisor_count():
    assert proper_divisor_count(6) == 3
    assert proper_divisor_count(108) == 11


def test_antidivisors():
    assert antidivisors(-1) == []
    assert antidivisors(-3) == [2]
    assert antidivisors(14) == [3, 4, 9]
    assert antidivisors(237) == [2, 5, 6, 11, 19, 25, 43, 95, 158]
    assert antidivisors(12345) == [2, 6, 7, 10, 30, 1646, 3527, 4938, 8230]
    assert antidivisors(393216) == [262144]
    assert sorted(x for x in antidivisors(3*5*7, 1)) == \
        [2, 6, 10, 11, 14, 19, 30, 42, 70]
    assert antidivisors(1) == []
    assert type(antidivisors(2, generator=True)) is not list

def test_antidivisor_count():
    assert antidivisor_count(0) == 0
    assert antidivisor_count(-1) == 0
    assert antidivisor_count(-4) == 1
    assert antidivisor_count(20) == 3
    assert antidivisor_count(25) == 5
    assert antidivisor_count(38) == 7
    assert antidivisor_count(180) == 6
    assert antidivisor_count(2*3*5) == 3


def test_smoothness_and_smoothness_p():
    assert smoothness(1) == (1, 1)
    assert smoothness(2**4*3**2) == (3, 16)

    assert smoothness_p(10431, m=1) == \
        (1, [(3, (2, 2, 4)), (19, (1, 5, 5)), (61, (1, 31, 31))])
    assert smoothness_p(10431) == \
        (-1, [(3, (2, 2, 2)), (19, (1, 3, 9)), (61, (1, 5, 5))])
    assert smoothness_p(10431, power=1) == \
        (-1, [(3, (2, 2, 2)), (61, (1, 5, 5)), (19, (1, 3, 9))])
    assert smoothness_p(21477639576571, visual=1) == \
        'p**i=4410317**1 has p-1 B=1787, B-pow=1787\n' + \
        'p**i=4869863**1 has p-1 B=2434931, B-pow=2434931'


def test_visual_factorint():
    assert factorint(1, visual=1) == 1
    forty2 = factorint(42, visual=True)
    assert type(forty2) == Mul
    assert str(forty2) == '2**1*3**1*7**1'
    assert factorint(1, visual=True) is S.One
    no = {"evaluate": False}
    assert factorint(42**2, visual=True) == Mul(Pow(2, 2, **no),
                                                Pow(3, 2, **no),
                                                Pow(7, 2, **no), **no)
    assert -1 in factorint(-42, visual=True).args


def test_factorrat():
    assert str(factorrat(S(12)/1, visual=True)) == '2**2*3**1'
    assert str(factorrat(Rational(1, 1), visual=True)) == '1'
    assert str(factorrat(S(25)/14, visual=True)) == '5**2/(2*7)'
    assert str(factorrat(Rational(25, 14), visual=True)) == '5**2/(2*7)'
    assert str(factorrat(S(-25)/14/9, visual=True)) == '-1*5**2/(2*3**2*7)'

    assert factorrat(S(12)/1, multiple=True) == [2, 2, 3]
    assert factorrat(Rational(1, 1), multiple=True) == []
    assert factorrat(S(25)/14, multiple=True) == [Rational(1, 7), S.Half, 5, 5]
    assert factorrat(Rational(25, 14), multiple=True) == [Rational(1, 7), S.Half, 5, 5]
    assert factorrat(Rational(12, 1), multiple=True) == [2, 2, 3]
    assert factorrat(S(-25)/14/9, multiple=True) == \
        [-1, Rational(1, 7), Rational(1, 3), Rational(1, 3), S.Half, 5, 5]


def test_visual_io():
    sm = smoothness_p
    fi = factorint
    # with smoothness_p
    n = 124
    d = fi(n)
    m = fi(d, visual=True)
    t = sm(n)
    s = sm(t)
    for th in [d, s, t, n, m]:
        assert sm(th, visual=True) == s
        assert sm(th, visual=1) == s
    for th in [d, s, t, n, m]:
        assert sm(th, visual=False) == t
    assert [sm(th, visual=None) for th in [d, s, t, n, m]] == [s, d, s, t, t]
    assert [sm(th, visual=2) for th in [d, s, t, n, m]] == [s, d, s, t, t]

    # with factorint
    for th in [d, m, n]:
        assert fi(th, visual=True) == m
        assert fi(th, visual=1) == m
    for th in [d, m, n]:
        assert fi(th, visual=False) == d
    assert [fi(th, visual=None) for th in [d, m, n]] == [m, d, d]
    assert [fi(th, visual=0) for th in [d, m, n]] == [m, d, d]

    # test reevaluation
    no = {"evaluate": False}
    assert sm({4: 2}, visual=False) == sm(16)
    assert sm(Mul(*[Pow(k, v, **no) for k, v in {4: 2, 2: 6}.items()], **no),
              visual=False) == sm(2**10)

    assert fi({4: 2}, visual=False) == fi(16)
    assert fi(Mul(*[Pow(k, v, **no) for k, v in {4: 2, 2: 6}.items()], **no),
              visual=False) == fi(2**10)


def test_core():
    assert core(35**13, 10) == 42875
    assert core(210**2) == 1
    assert core(7776, 3) == 36
    assert core(10**27, 22) == 10**5
    assert core(537824) == 14
    assert core(1, 6) == 1


def test__divisor_sigma():
    assert _divisor_sigma(23450) == 50592
    assert _divisor_sigma(23450, 0) == 24
    assert _divisor_sigma(23450, 1) == 50592
    assert _divisor_sigma(23450, 2) == 730747500
    assert _divisor_sigma(23450, 3) == 14666785333344
    A000005 = [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4, 5, 2, 6, 2, 6, 4,
               4, 2, 8, 3, 4, 4, 6, 2, 8, 2, 6, 4, 4, 4, 9, 2, 4, 4, 8, 2, 8]
    for n, val in enumerate(A000005, 1):
        assert _divisor_sigma(n, 0) == val
    A000203 = [1, 3, 4, 7, 6, 12, 8, 15, 13, 18, 12, 28, 14, 24, 24, 31, 18,
               39, 20, 42, 32, 36, 24, 60, 31, 42, 40, 56, 30, 72, 32, 63, 48]
    for n, val in enumerate(A000203, 1):
        assert _divisor_sigma(n, 1) == val
    A001157 = [1, 5, 10, 21, 26, 50, 50, 85, 91, 130, 122, 210, 170, 250, 260,
               341, 290, 455, 362, 546, 500, 610, 530, 850, 651, 850, 820, 1050]
    for n, val in enumerate(A001157, 1):
        assert _divisor_sigma(n, 2) == val


def test_mersenne_prime_exponent():
    assert mersenne_prime_exponent(1) == 2
    assert mersenne_prime_exponent(4) == 7
    assert mersenne_prime_exponent(10) == 89
    assert mersenne_prime_exponent(25) == 21701
    raises(ValueError, lambda: mersenne_prime_exponent(52))
    raises(ValueError, lambda: mersenne_prime_exponent(0))


def test_is_perfect():
    assert is_perfect(-6) is False
    assert is_perfect(6) is True
    assert is_perfect(15) is False
    assert is_perfect(28) is True
    assert is_perfect(400) is False
    assert is_perfect(496) is True
    assert is_perfect(8128) is True
    assert is_perfect(10000) is False


def test_is_abundant():
    assert is_abundant(10) is False
    assert is_abundant(12) is True
    assert is_abundant(18) is True
    assert is_abundant(21) is False
    assert is_abundant(945) is True


def test_is_deficient():
    assert is_deficient(10) is True
    assert is_deficient(22) is True
    assert is_deficient(56) is False
    assert is_deficient(20) is False
    assert is_deficient(36) is False


def test_is_amicable():
    assert is_amicable(173, 129) is False
    assert is_amicable(220, 284) is True
    assert is_amicable(8756, 8756) is False


def test_is_carmichael():
    A002997 = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841,
               29341, 41041, 46657, 52633, 62745, 63973, 75361, 101101]
    for n in range(1, 5000):
        assert is_carmichael(n) == (n in A002997)
    for n in A002997:
        assert is_carmichael(n)


def test_find_carmichael_numbers_in_range():
    assert find_carmichael_numbers_in_range(0, 561) == []
    assert find_carmichael_numbers_in_range(561, 562) == [561]
    assert find_carmichael_numbers_in_range(561, 1105) == find_carmichael_numbers_in_range(561, 562)
    raises(ValueError, lambda: find_carmichael_numbers_in_range(-2, 2))
    raises(ValueError, lambda: find_carmichael_numbers_in_range(22, 2))


def test_find_first_n_carmichaels():
    assert find_first_n_carmichaels(0) == []
    assert find_first_n_carmichaels(1) == [561]
    assert find_first_n_carmichaels(2) == [561, 1105]


def test_dra():
    assert dra(19, 12) == 8
    assert dra(2718, 10) == 9
    assert dra(0, 22) == 0
    assert dra(23456789, 10) == 8
    raises(ValueError, lambda: dra(24, -2))
    raises(ValueError, lambda: dra(24.2, 5))

def test_drm():
    assert drm(19, 12) == 7
    assert drm(2718, 10) == 2
    assert drm(0, 15) == 0
    assert drm(234161, 10) == 6
    raises(ValueError, lambda: drm(24, -2))
    raises(ValueError, lambda: drm(11.6, 9))


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert primenu(3) == 1
    with warns_deprecated_sympy():
        assert primeomega(3) == 1
    with warns_deprecated_sympy():
        assert totient(3) == 2
    with warns_deprecated_sympy():
        assert reduced_totient(3) == 2
    with warns_deprecated_sympy():
        assert divisor_sigma(3) == 4
    with warns_deprecated_sympy():
        assert udivisor_sigma(3) == 4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\apply\test_series_apply.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels


@pytest.fixture(params=[False, "compat"])
def by_row(request):
    return request.param


def test_series_map_box_timedelta(by_row):
    # GH#11349
    ser = Series(timedelta_range("1 day 1 s", periods=3, freq="h"))

    def f(x):
        return x.total_seconds() if by_row else x.dt.total_seconds()

    result = ser.apply(f, by_row=by_row)

    expected = ser.map(lambda x: x.total_seconds())
    tm.assert_series_equal(result, expected)

    expected = Series([86401.0, 90001.0, 93601.0])
    tm.assert_series_equal(result, expected)


def test_apply(datetime_series, by_row):
    result = datetime_series.apply(np.sqrt, by_row=by_row)
    with np.errstate(all="ignore"):
        expected = np.sqrt(datetime_series)
    tm.assert_series_equal(result, expected)

    # element-wise apply (ufunc)
    result = datetime_series.apply(np.exp, by_row=by_row)
    expected = np.exp(datetime_series)
    tm.assert_series_equal(result, expected)

    # empty series
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))
    rs = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)

    # check all metadata (GH 9322)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name

    # index but no data
    s = Series(index=[1, 2, 3], dtype=np.float64)
    rs = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)


def test_apply_map_same_length_inference_bug():
    s = Series([1, 2])

    def f(x):
        return (x, x + 1)

    result = s.apply(f, by_row="compat")
    expected = s.map(f)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("convert_dtype", [True, False])
def test_apply_convert_dtype_deprecated(convert_dtype):
    ser = Series(np.random.default_rng(2).standard_normal(10))

    def func(x):
        return x if x > 0 else np.nan

    with tm.assert_produces_warning(FutureWarning):
        ser.apply(func, convert_dtype=convert_dtype, by_row="compat")


def test_apply_args():
    s = Series(["foo,bar"])

    result = s.apply(str.split, args=(",",))
    assert result[0] == ["foo", "bar"]
    assert isinstance(result[0], list)


@pytest.mark.parametrize(
    "args, kwargs, increment",
    [((), {}, 0), ((), {"a": 1}, 1), ((2, 3), {}, 32), ((1,), {"c": 2}, 201)],
)
def test_agg_args(args, kwargs, increment):
    # GH 43357
    def f(x, a=0, b=0, c=0):
        return x + a + 10 * b + 100 * c

    s = Series([1, 2])
    msg = (
        "in Series.agg cannot aggregate and has been deprecated. "
        "Use Series.transform to keep behavior unchanged."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.agg(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated():
    # GH 53325
    s = Series([1, 2, 3])

    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c

    msg = "using .+ in Series.agg cannot aggregate and"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        s.agg(foo1, 0, 3, c=4)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        s.agg([foo1, foo2], 0, 3, c=4)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        s.agg({"a": foo1, "b": foo2}, 0, 3, c=4)


def test_series_apply_map_box_timestamps(by_row):
    # GH#2689, GH#2627
    ser = Series(date_range("1/1/2000", periods=10))

    def func(x):
        return (x.hour, x.day, x.month)

    if not by_row:
        msg = "Series' object has no attribute 'hour'"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(func, by_row=by_row)
        return

    result = ser.apply(func, by_row=by_row)
    expected = ser.map(func)
    tm.assert_series_equal(result, expected)


def test_apply_box_dt64():
    # ufunc will not be boxed. Same test cases as the test_map_box
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    ser = Series(vals, dtype="M8[ns]")
    assert ser.dtype == "datetime64[ns]"
    # boxed value must be Timestamp instance
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    tm.assert_series_equal(res, exp)


def test_apply_box_dt64tz():
    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    ser = Series(vals, dtype="M8[ns, US/Eastern]")
    assert ser.dtype == "datetime64[ns, US/Eastern]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    tm.assert_series_equal(res, exp)


def test_apply_box_td64():
    # timedelta
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    ser = Series(vals)
    assert ser.dtype == "timedelta64[ns]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.days}", by_row="compat")
    exp = Series(["Timedelta_1", "Timedelta_2"])
    tm.assert_series_equal(res, exp)


def test_apply_box_period():
    # period
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    ser = Series(vals)
    assert ser.dtype == "Period[M]"
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.freqstr}", by_row="compat")
    exp = Series(["Period_M", "Period_M"])
    tm.assert_series_equal(res, exp)


def test_apply_datetimetz(by_row):
    values = date_range("2011-01-01", "2011-01-02", freq="h").tz_localize("Asia/Tokyo")
    s = Series(values, name="XX")

    result = s.apply(lambda x: x + pd.offsets.Day(), by_row=by_row)
    exp_values = date_range("2011-01-02", "2011-01-03", freq="h").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    tm.assert_series_equal(result, exp)

    result = s.apply(lambda x: x.hour if by_row else x.dt.hour, by_row=by_row)
    exp = Series(list(range(24)) + [0], name="XX", dtype="int64" if by_row else "int32")
    tm.assert_series_equal(result, exp)

    # not vectorized
    def f(x):
        return str(x.tz) if by_row else str(x.dt.tz)

    result = s.apply(f, by_row=by_row)
    if by_row:
        exp = Series(["Asia/Tokyo"] * 25, name="XX")
        tm.assert_series_equal(result, exp)
    else:
        assert result == "Asia/Tokyo"


def test_apply_categorical(by_row, using_infer_string):
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    ser = Series(values, name="XX", index=list("abcdefg"))

    if not by_row:
        msg = "Series' object has no attribute 'lower"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(lambda x: x.lower(), by_row=by_row)
        assert ser.apply(lambda x: "A", by_row=by_row) == "A"
        return

    result = ser.apply(lambda x: x.lower(), by_row=by_row)

    # should be categorical dtype when the number of categories are
    # the same
    values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(values, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp.values)

    result = ser.apply(lambda x: "A")
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object if not using_infer_string else "str"


@pytest.mark.parametrize("series", [["1-1", "1-1", np.nan], ["1-1", "1-2", np.nan]])
def test_apply_categorical_with_nan_values(series, by_row):
    # GH 20714 bug fixed in: GH 24275
    s = Series(series, dtype="category")
    if not by_row:
        msg = "'Series' object has no attribute 'split'"
        with pytest.raises(AttributeError, match=msg):
            s.apply(lambda x: x.split("-")[0], by_row=by_row)
        return

    result = s.apply(lambda x: x.split("-")[0], by_row=by_row)
    result = result.astype(object)
    expected = Series(["1", "1", np.nan], dtype="category")
    expected = expected.astype(object)
    tm.assert_series_equal(result, expected)


def test_apply_empty_integer_series_with_datetime_index(by_row):
    # GH 21245
    s = Series([], index=date_range(start="2018-01-01", periods=0), dtype=int)
    result = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(result, s)


def test_apply_dataframe_iloc():
    uintDF = DataFrame(np.uint64([1, 2, 3, 4, 5]), columns=["Numbers"])
    indexDF = DataFrame([2, 3, 2, 1, 2], columns=["Indices"])

    def retrieve(targetRow, targetDF):
        val = targetDF["Numbers"].iloc[targetRow]
        return val

    result = indexDF["Indices"].apply(retrieve, args=(uintDF,))
    expected = Series([3, 4, 3, 2, 3], name="Indices", dtype="uint64")
    tm.assert_series_equal(result, expected)


def test_transform(string_series, by_row):
    # transforming functions

    with np.errstate(all="ignore"):
        f_sqrt = np.sqrt(string_series)
        f_abs = np.abs(string_series)

        # ufunc
        result = string_series.apply(np.sqrt, by_row=by_row)
        expected = f_sqrt.copy()
        tm.assert_series_equal(result, expected)

        # list-like
        result = string_series.apply([np.sqrt], by_row=by_row)
        expected = f_sqrt.to_frame().copy()
        expected.columns = ["sqrt"]
        tm.assert_frame_equal(result, expected)

        result = string_series.apply(["sqrt"], by_row=by_row)
        tm.assert_frame_equal(result, expected)

        # multiple items in list
        # these are in the order as if we are applying both functions per
        # series and then concatting
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["sqrt", "absolute"]
        result = string_series.apply([np.sqrt, np.abs], by_row=by_row)
        tm.assert_frame_equal(result, expected)

        # dict, provide renaming
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["foo", "bar"]
        expected = expected.unstack().rename("series")

        result = string_series.apply({"foo": np.sqrt, "bar": np.abs}, by_row=by_row)
        tm.assert_series_equal(result.reindex_like(expected), expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_partial_failure(op, request):
    # GH 35964
    if op in ("ffill", "bfill", "pad", "backfill", "shift"):
        request.applymarker(
            pytest.mark.xfail(reason=f"{op} is successful on any dtype")
        )

    # Using object makes most transform kernels fail
    ser = Series(3 * [object])

    if op in ("fillna", "ngroup"):
        error = ValueError
        msg = "Transform function failed"
    else:
        error = TypeError
        msg = "|".join(
            [
                "not supported between instances of 'type' and 'type'",
                "unsupported operand type",
            ]
        )

    with pytest.raises(error, match=msg):
        ser.transform([op, "shift"])

    with pytest.raises(error, match=msg):
        ser.transform({"A": op, "B": "shift"})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op], "B": ["shift"]})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op, "shift"], "B": [op]})


def test_transform_partial_failure_valueerror():
    # GH 40211
    def noop(x):
        return x

    def raising_op(_):
        raise ValueError

    ser = Series(3 * [object])
    msg = "Transform function failed"

    with pytest.raises(ValueError, match=msg):
        ser.transform([noop, raising_op])

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": raising_op, "B": noop})

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [raising_op], "B": [noop]})

    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [noop, raising_op], "B": [noop]})


def test_demo():
    # demonstration tests
    s = Series(range(6), dtype="int64", name="series")

    result = s.agg(["min", "max"])
    expected = Series([0, 5], index=["min", "max"], name="series")
    tm.assert_series_equal(result, expected)

    result = s.agg({"foo": "min"})
    expected = Series([0], index=["foo"], name="series")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [str, lambda x: str(x)])
def test_apply_map_evaluate_lambdas_the_same(string_series, func, by_row):
    # test that we are evaluating row-by-row first if by_row="compat"
    # else vectorized evaluation
    result = string_series.apply(func, by_row=by_row)

    if by_row:
        expected = string_series.map(func)
        tm.assert_series_equal(result, expected)
    else:
        assert result == str(string_series)


def test_agg_evaluate_lambdas(string_series):
    # GH53325
    # in the future, the result will be a Series class.

    with tm.assert_produces_warning(FutureWarning):
        result = string_series.agg(lambda x: type(x))
    assert isinstance(result, Series) and len(result) == len(string_series)

    with tm.assert_produces_warning(FutureWarning):
        result = string_series.agg(type)
    assert isinstance(result, Series) and len(result) == len(string_series)


@pytest.mark.parametrize("op_name", ["agg", "apply"])
def test_with_nested_series(datetime_series, op_name):
    # GH 2316
    # .agg with a reducer and a transform, what to do
    msg = "cannot aggregate"
    warning = FutureWarning if op_name == "agg" else None
    with tm.assert_produces_warning(warning, match=msg):
        # GH52123
        result = getattr(datetime_series, op_name)(
            lambda x: Series([x, x**2], index=["x", "x^2"])
        )
    expected = DataFrame({"x": datetime_series, "x^2": datetime_series**2})
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = datetime_series.agg(lambda x: Series([x, x**2], index=["x", "x^2"]))
    tm.assert_frame_equal(result, expected)


def test_replicate_describe(string_series):
    # this also tests a result set that is all scalars
    expected = string_series.describe()
    result = string_series.apply(
        {
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "25%": lambda x: x.quantile(0.25),
            "50%": "median",
            "75%": lambda x: x.quantile(0.75),
            "max": "max",
        },
    )
    tm.assert_series_equal(result, expected)


def test_reduce(string_series):
    # reductions with named functions
    result = string_series.agg(["sum", "mean"])
    expected = Series(
        [string_series.sum(), string_series.mean()],
        ["sum", "mean"],
        name=string_series.name,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "how, kwds",
    [("agg", {}), ("apply", {"by_row": "compat"}), ("apply", {"by_row": False})],
)
def test_non_callable_aggregates(how, kwds):
    # test agg using non-callable series attributes
    # GH 39116 - expand to apply
    s = Series([1, 2, None])

    # Calling agg w/ just a string arg same as calling s.arg
    result = getattr(s, how)("size", **kwds)
    expected = s.size
    assert result == expected

    # test when mixed w/ callable reducers
    result = getattr(s, how)(["size", "count", "mean"], **kwds)
    expected = Series({"size": 3.0, "count": 2.0, "mean": 1.5})
    tm.assert_series_equal(result, expected)

    result = getattr(s, how)({"size": "size", "count": "count", "mean": "mean"}, **kwds)
    tm.assert_series_equal(result, expected)


def test_series_apply_no_suffix_index(by_row):
    # GH36189
    s = Series([4] * 3)
    result = s.apply(["sum", lambda x: x.sum(), lambda x: x.sum()], by_row=by_row)
    expected = Series([12, 12, 12], index=["sum", "<lambda>", "<lambda>"])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dti,exp",
    [
        (
            Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])),
            DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype="int64"),
        ),
        (
            Series(
                np.arange(10, dtype=np.float64),
                index=date_range("2020-01-01", periods=10),
                name="ts",
            ),
            DataFrame(np.repeat([[1, 2]], 10, axis=0), dtype="int64"),
        ),
    ],
)
@pytest.mark.parametrize("aware", [True, False])
def test_apply_series_on_date_time_index_aware_series(dti, exp, aware):
    # GH 25959
    # Calling apply on a localized time series should not cause an error
    if aware:
        index = dti.tz_localize("UTC").index
    else:
        index = dti.index
    result = Series(index).apply(lambda x: Series([1, 2]))
    tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    "by_row, expected", [("compat", Series(np.ones(10), dtype="int64")), (False, 1)]
)
def test_apply_scalar_on_date_time_index_aware_series(by_row, expected):
    # GH 25959
    # Calling apply on a localized time series should not cause an error
    series = Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10, tz="UTC"),
    )
    result = Series(series.index).apply(lambda x: 1, by_row=by_row)
    tm.assert_equal(result, expected)


def test_apply_to_timedelta(by_row):
    list_of_valid_strings = ["00:00:01", "00:00:02"]
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)

    list_of_strings = ["00:00:01", np.nan, pd.NaT, pd.NaT]

    a = pd.to_timedelta(list_of_strings)
    ser = Series(list_of_strings)
    b = ser.apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sum], ["sum"]),
        ([np.sum, np.mean], ["sum", "mean"]),
        (np.array([np.sum]), ["sum"]),
        (np.array([np.sum, np.mean]), ["sum", "mean"]),
    ],
)
@pytest.mark.parametrize(
    "how, kwargs",
    [["agg", {}], ["apply", {"by_row": "compat"}], ["apply", {"by_row": False}]],
)
def test_apply_listlike_reducer(string_series, ops, names, how, kwargs):
    # GH 39140
    expected = Series({name: op(string_series) for name, op in zip(names, ops)})
    expected.name = "series"
    warn = FutureWarning if how == "agg" else None
    msg = f"using Series.[{'|'.join(names)}]"
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sum},
        {"A": np.sum, "B": np.mean},
        Series({"A": np.sum}),
        Series({"A": np.sum, "B": np.mean}),
    ],
)
@pytest.mark.parametrize(
    "how, kwargs",
    [["agg", {}], ["apply", {"by_row": "compat"}], ["apply", {"by_row": False}]],
)
def test_apply_dictlike_reducer(string_series, ops, how, kwargs, by_row):
    # GH 39140
    expected = Series({name: op(string_series) for name, op in ops.items()})
    expected.name = string_series.name
    warn = FutureWarning if how == "agg" else None
    msg = "using Series.[sum|mean]"
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
def test_apply_listlike_transformer(string_series, ops, names, by_row):
    # GH 39140
    with np.errstate(all="ignore"):
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result = string_series.apply(ops, by_row=by_row)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops, expected",
    [
        ([lambda x: x], DataFrame({"<lambda>": [1, 2, 3]})),
        ([lambda x: x.sum()], Series([6], index=["<lambda>"])),
    ],
)
def test_apply_listlike_lambda(ops, expected, by_row):
    # GH53400
    ser = Series([1, 2, 3])
    result = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sqrt},
        {"A": np.sqrt, "B": np.exp},
        Series({"A": np.sqrt}),
        Series({"A": np.sqrt, "B": np.exp}),
    ],
)
def test_apply_dictlike_transformer(string_series, ops, by_row):
    # GH 39140
    with np.errstate(all="ignore"):
        expected = concat({name: op(string_series) for name, op in ops.items()})
        expected.name = string_series.name
        result = string_series.apply(ops, by_row=by_row)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops, expected",
    [
        (
            {"a": lambda x: x},
            Series([1, 2, 3], index=MultiIndex.from_arrays([["a"] * 3, range(3)])),
        ),
        ({"a": lambda x: x.sum()}, Series([6], index=["a"])),
    ],
)
def test_apply_dictlike_lambda(ops, by_row, expected):
    # GH53400
    ser = Series([1, 2, 3])
    result = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


def test_apply_retains_column_name(by_row):
    # GH 16380
    df = DataFrame({"x": range(3)}, Index(range(3), name="x"))
    result = df.x.apply(lambda x: Series(range(x + 1), Index(range(x + 1), name="y")))
    expected = DataFrame(
        [[0.0, np.nan, np.nan], [0.0, 1.0, np.nan], [0.0, 1.0, 2.0]],
        columns=Index(range(3), name="y"),
        index=Index(range(3), name="x"),
    )
    tm.assert_frame_equal(result, expected)


def test_apply_type():
    # GH 46719
    s = Series([3, "string", float], index=["a", "b", "c"])
    result = s.apply(type)
    expected = Series([int, str, type], index=["a", "b", "c"])
    tm.assert_series_equal(result, expected)


def test_series_apply_unpack_nested_data():
    # GH#55189
    ser = Series([[1, 2, 3], [4, 5, 6, 7]])
    result = ser.apply(lambda x: Series(x))
    expected = DataFrame({0: [1.0, 4.0], 1: [2.0, 5.0], 2: [3.0, 6.0], 3: [np.nan, 7]})
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_gammazeta.py ===
from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int

def test_zeta_int_bug():
    assert mpf_zeta_int(0, 10) == from_float(-0.5)

def test_bernoulli():
    assert bernfrac(0) == (1,1)
    assert bernfrac(1) == (-1,2)
    assert bernfrac(2) == (1,6)
    assert bernfrac(3) == (0,1)
    assert bernfrac(4) == (-1,30)
    assert bernfrac(5) == (0,1)
    assert bernfrac(6) == (1,42)
    assert bernfrac(8) == (-1,30)
    assert bernfrac(10) == (5,66)
    assert bernfrac(12) == (-691,2730)
    assert bernfrac(18) == (43867,798)
    p, q = bernfrac(228)
    assert p % 10**10 == 164918161
    assert q == 625170
    p, q = bernfrac(1000)
    assert p % 10**10 == 7950421099
    assert q == 342999030
    mp.dps = 15
    assert bernoulli(0) == 1
    assert bernoulli(1) == -0.5
    assert bernoulli(2).ae(1./6)
    assert bernoulli(3) == 0
    assert bernoulli(4).ae(-1./30)
    assert bernoulli(5) == 0
    assert bernoulli(6).ae(1./42)
    assert str(bernoulli(10)) == '0.0757575757575758'
    assert str(bernoulli(234)) == '7.62772793964344e+267'
    assert str(bernoulli(10**5)) == '-5.82229431461335e+376755'
    assert str(bernoulli(10**8+2)) == '1.19570355039953e+676752584'

    mp.dps = 50
    assert str(bernoulli(10)) == '0.075757575757575757575757575757575757575757575757576'
    assert str(bernoulli(234)) == '7.6277279396434392486994969020496121553385863373331e+267'
    assert str(bernoulli(10**5)) == '-5.8222943146133508236497045360612887555320691004308e+376755'
    assert str(bernoulli(10**8+2)) == '1.1957035503995297272263047884604346914602088317782e+676752584'

    mp.dps = 1000
    assert bernoulli(10).ae(mpf(5)/66)

    mp.dps = 50000
    assert bernoulli(10).ae(mpf(5)/66)

    mp.dps = 15

def test_bernpoly_eulerpoly():
    mp.dps = 15
    assert bernpoly(0,-1).ae(1)
    assert bernpoly(0,0).ae(1)
    assert bernpoly(0,'1/2').ae(1)
    assert bernpoly(0,'3/4').ae(1)
    assert bernpoly(0,1).ae(1)
    assert bernpoly(0,2).ae(1)
    assert bernpoly(1,-1).ae('-3/2')
    assert bernpoly(1,0).ae('-1/2')
    assert bernpoly(1,'1/2').ae(0)
    assert bernpoly(1,'3/4').ae('1/4')
    assert bernpoly(1,1).ae('1/2')
    assert bernpoly(1,2).ae('3/2')
    assert bernpoly(2,-1).ae('13/6')
    assert bernpoly(2,0).ae('1/6')
    assert bernpoly(2,'1/2').ae('-1/12')
    assert bernpoly(2,'3/4').ae('-1/48')
    assert bernpoly(2,1).ae('1/6')
    assert bernpoly(2,2).ae('13/6')
    assert bernpoly(3,-1).ae(-3)
    assert bernpoly(3,0).ae(0)
    assert bernpoly(3,'1/2').ae(0)
    assert bernpoly(3,'3/4').ae('-3/64')
    assert bernpoly(3,1).ae(0)
    assert bernpoly(3,2).ae(3)
    assert bernpoly(4,-1).ae('119/30')
    assert bernpoly(4,0).ae('-1/30')
    assert bernpoly(4,'1/2').ae('7/240')
    assert bernpoly(4,'3/4').ae('7/3840')
    assert bernpoly(4,1).ae('-1/30')
    assert bernpoly(4,2).ae('119/30')
    assert bernpoly(5,-1).ae(-5)
    assert bernpoly(5,0).ae(0)
    assert bernpoly(5,'1/2').ae(0)
    assert bernpoly(5,'3/4').ae('25/1024')
    assert bernpoly(5,1).ae(0)
    assert bernpoly(5,2).ae(5)
    assert bernpoly(10,-1).ae('665/66')
    assert bernpoly(10,0).ae('5/66')
    assert bernpoly(10,'1/2').ae('-2555/33792')
    assert bernpoly(10,'3/4').ae('-2555/34603008')
    assert bernpoly(10,1).ae('5/66')
    assert bernpoly(10,2).ae('665/66')
    assert bernpoly(11,-1).ae(-11)
    assert bernpoly(11,0).ae(0)
    assert bernpoly(11,'1/2').ae(0)
    assert bernpoly(11,'3/4').ae('-555731/4194304')
    assert bernpoly(11,1).ae(0)
    assert bernpoly(11,2).ae(11)
    assert eulerpoly(0,-1).ae(1)
    assert eulerpoly(0,0).ae(1)
    assert eulerpoly(0,'1/2').ae(1)
    assert eulerpoly(0,'3/4').ae(1)
    assert eulerpoly(0,1).ae(1)
    assert eulerpoly(0,2).ae(1)
    assert eulerpoly(1,-1).ae('-3/2')
    assert eulerpoly(1,0).ae('-1/2')
    assert eulerpoly(1,'1/2').ae(0)
    assert eulerpoly(1,'3/4').ae('1/4')
    assert eulerpoly(1,1).ae('1/2')
    assert eulerpoly(1,2).ae('3/2')
    assert eulerpoly(2,-1).ae(2)
    assert eulerpoly(2,0).ae(0)
    assert eulerpoly(2,'1/2').ae('-1/4')
    assert eulerpoly(2,'3/4').ae('-3/16')
    assert eulerpoly(2,1).ae(0)
    assert eulerpoly(2,2).ae(2)
    assert eulerpoly(3,-1).ae('-9/4')
    assert eulerpoly(3,0).ae('1/4')
    assert eulerpoly(3,'1/2').ae(0)
    assert eulerpoly(3,'3/4').ae('-11/64')
    assert eulerpoly(3,1).ae('-1/4')
    assert eulerpoly(3,2).ae('9/4')
    assert eulerpoly(4,-1).ae(2)
    assert eulerpoly(4,0).ae(0)
    assert eulerpoly(4,'1/2').ae('5/16')
    assert eulerpoly(4,'3/4').ae('57/256')
    assert eulerpoly(4,1).ae(0)
    assert eulerpoly(4,2).ae(2)
    assert eulerpoly(5,-1).ae('-3/2')
    assert eulerpoly(5,0).ae('-1/2')
    assert eulerpoly(5,'1/2').ae(0)
    assert eulerpoly(5,'3/4').ae('361/1024')
    assert eulerpoly(5,1).ae('1/2')
    assert eulerpoly(5,2).ae('3/2')
    assert eulerpoly(10,-1).ae(2)
    assert eulerpoly(10,0).ae(0)
    assert eulerpoly(10,'1/2').ae('-50521/1024')
    assert eulerpoly(10,'3/4').ae('-36581523/1048576')
    assert eulerpoly(10,1).ae(0)
    assert eulerpoly(10,2).ae(2)
    assert eulerpoly(11,-1).ae('-699/4')
    assert eulerpoly(11,0).ae('691/4')
    assert eulerpoly(11,'1/2').ae(0)
    assert eulerpoly(11,'3/4').ae('-512343611/4194304')
    assert eulerpoly(11,1).ae('-691/4')
    assert eulerpoly(11,2).ae('699/4')
    # Potential accuracy issues
    assert bernpoly(10000,10000).ae('5.8196915936323387117e+39999')
    assert bernpoly(200,17.5).ae(3.8048418524583064909e244)
    assert eulerpoly(200,17.5).ae(-3.7309911582655785929e275)

def test_gamma():
    mp.dps = 15
    assert gamma(0.25).ae(3.6256099082219083119)
    assert gamma(0.0001).ae(9999.4228832316241908)
    assert gamma(300).ae('1.0201917073881354535e612')
    assert gamma(-0.5).ae(-3.5449077018110320546)
    assert gamma(-7.43).ae(0.00026524416464197007186)
    #assert gamma(Rational(1,2)) == gamma(0.5)
    #assert gamma(Rational(-7,3)).ae(gamma(mpf(-7)/3))
    assert gamma(1+1j).ae(0.49801566811835604271 - 0.15494982830181068512j)
    assert gamma(-1+0.01j).ae(-0.422733904013474115 + 99.985883082635367436j)
    assert gamma(20+30j).ae(-1453876687.5534810 + 1163777777.8031573j)
    # Should always give exact factorials when they can
    # be represented as mpfs under the current working precision
    fact = 1
    for i in range(1, 18):
        assert gamma(i) == fact
        fact *= i
    for dps in [170, 600]:
        fact = 1
        mp.dps = dps
        for i in range(1, 105):
            assert gamma(i) == fact
            fact *= i
    mp.dps = 100
    assert gamma(0.5).ae(sqrt(pi))
    mp.dps = 15
    assert factorial(0) == fac(0) == 1
    assert factorial(3) == 6
    assert isnan(gamma(nan))
    assert gamma(1100).ae('4.8579168073569433667e2866')
    assert rgamma(0) == 0
    assert rgamma(-1) == 0
    assert rgamma(2) == 1.0
    assert rgamma(3) == 0.5
    assert loggamma(2+8j).ae(-8.5205176753667636926 + 10.8569497125597429366j)
    assert loggamma('1e10000').ae('2.302485092994045684017991e10004')
    assert loggamma('1e10000j').ae(mpc('-1.570796326794896619231322e10000','2.302485092994045684017991e10004'))

def test_fac2():
    mp.dps = 15
    assert [fac2(n) for n in range(10)] == [1,1,2,3,8,15,48,105,384,945]
    assert fac2(-5).ae(1./3)
    assert fac2(-11).ae(-1./945)
    assert fac2(50).ae(5.20469842636666623e32)
    assert fac2(0.5+0.75j).ae(0.81546769394688069176-0.34901016085573266889j)
    assert fac2(inf) == inf
    assert isnan(fac2(-inf))

def test_gamma_quotients():
    mp.dps = 15
    h = 1e-8
    ep = 1e-4
    G = gamma
    assert gammaprod([-1],[-3,-4]) == 0
    assert gammaprod([-1,0],[-5]) == inf
    assert abs(gammaprod([-1],[-2]) - G(-1+h)/G(-2+h)) < 1e-4
    assert abs(gammaprod([-4,-3],[-2,0]) - G(-4+h)*G(-3+h)/G(-2+h)/G(0+h)) < 1e-4
    assert rf(3,0) == 1
    assert rf(2.5,1) == 2.5
    assert rf(-5,2) == 20
    assert rf(j,j).ae(gamma(2*j)/gamma(j))
    assert rf('-255.5815971722918','-0.5119253100282322').ae('-0.1952720278805729485')  # issue 421
    assert ff(-2,0) == 1
    assert ff(-2,1) == -2
    assert ff(4,3) == 24
    assert ff(3,4) == 0
    assert binomial(0,0) == 1
    assert binomial(1,0) == 1
    assert binomial(0,-1) == 0
    assert binomial(3,2) == 3
    assert binomial(5,2) == 10
    assert binomial(5,3) == 10
    assert binomial(5,5) == 1
    assert binomial(-1,0) == 1
    assert binomial(-2,-4) == 3
    assert binomial(4.5, 1.5) == 6.5625
    assert binomial(1100,1) == 1100
    assert binomial(1100,2) == 604450
    assert beta(1,1) == 1
    assert beta(0,0) == inf
    assert beta(3,0) == inf
    assert beta(-1,-1) == inf
    assert beta(1.5,1).ae(2/3.)
    assert beta(1.5,2.5).ae(pi/16)
    assert (10**15*beta(10,100)).ae(2.3455339739604649879)
    assert beta(inf,inf) == 0
    assert isnan(beta(-inf,inf))
    assert isnan(beta(-3,inf))
    assert isnan(beta(0,inf))
    assert beta(inf,0.5) == beta(0.5,inf) == 0
    assert beta(inf,-1.5) == inf
    assert beta(inf,-0.5) == -inf
    assert beta(1+2j,-1-j/2).ae(1.16396542451069943086+0.08511695947832914640j)
    assert beta(-0.5,0.5) == 0
    assert beta(-3,3).ae(-1/3.)
    assert beta('-255.5815971722918','-0.5119253100282322').ae('18.157330562703710339')  # issue 421

def test_zeta():
    mp.dps = 15
    assert zeta(2).ae(pi**2 / 6)
    assert zeta(2.0).ae(pi**2 / 6)
    assert zeta(mpc(2)).ae(pi**2 / 6)
    assert zeta(100).ae(1)
    assert zeta(0).ae(-0.5)
    assert zeta(0.5).ae(-1.46035450880958681)
    assert zeta(-1).ae(-mpf(1)/12)
    assert zeta(-2) == 0
    assert zeta(-3).ae(mpf(1)/120)
    assert zeta(-4) == 0
    assert zeta(-100) == 0
    assert isnan(zeta(nan))
    assert zeta(1e-30).ae(-0.5)
    assert zeta(-1e-30).ae(-0.5)
    # Zeros in the critical strip
    assert zeta(mpc(0.5, 14.1347251417346937904)).ae(0)
    assert zeta(mpc(0.5, 21.0220396387715549926)).ae(0)
    assert zeta(mpc(0.5, 25.0108575801456887632)).ae(0)
    assert zeta(mpc(1e-30,1e-40)).ae(-0.5)
    assert zeta(mpc(-1e-30,1e-40)).ae(-0.5)
    mp.dps = 50
    im = '236.5242296658162058024755079556629786895294952121891237'
    assert zeta(mpc(0.5, im)).ae(0, 1e-46)
    mp.dps = 15
    # Complex reflection formula
    assert (zeta(-60+3j) / 10**34).ae(8.6270183987866146+15.337398548226238j)
    # issue #358
    assert zeta(0,0.5) == 0
    assert zeta(0,0) == 0.5
    assert zeta(0,0.5,1).ae(-0.34657359027997265)
    # see issue #390
    assert zeta(-1.5,0.5j).ae(-0.13671400162512768475 + 0.11411333638426559139j)

def test_altzeta():
    mp.dps = 15
    assert altzeta(-2) == 0
    assert altzeta(-4) == 0
    assert altzeta(-100) == 0
    assert altzeta(0) == 0.5
    assert altzeta(-1) == 0.25
    assert altzeta(-3) == -0.125
    assert altzeta(-5) == 0.25
    assert altzeta(-21) == 1180529130.25
    assert altzeta(1).ae(log(2))
    assert altzeta(2).ae(pi**2/12)
    assert altzeta(10).ae(73*pi**10/6842880)
    assert altzeta(50) < 1
    assert altzeta(60, rounding='d') < 1
    assert altzeta(60, rounding='u') == 1
    assert altzeta(10000, rounding='d') < 1
    assert altzeta(10000, rounding='u') == 1
    assert altzeta(3+0j) == altzeta(3)
    s = 3+4j
    assert altzeta(s).ae((1-2**(1-s))*zeta(s))
    s = -3+4j
    assert altzeta(s).ae((1-2**(1-s))*zeta(s))
    assert altzeta(-100.5).ae(4.58595480083585913e+108)
    assert altzeta(1.3).ae(0.73821404216623045)
    assert altzeta(1e-30).ae(0.5)
    assert altzeta(-1e-30).ae(0.5)
    assert altzeta(mpc(1e-30,1e-40)).ae(0.5)
    assert altzeta(mpc(-1e-30,1e-40)).ae(0.5)

def test_zeta_huge():
    mp.dps = 15
    assert zeta(inf) == 1
    mp.dps = 50
    assert zeta(100).ae('1.0000000000000000000000000000007888609052210118073522')
    assert zeta(40*pi).ae('1.0000000000000000000000000000000000000148407238666182')
    mp.dps = 10000
    v = zeta(33000)
    mp.dps = 15
    assert str(v-1) == '1.02363019598118e-9934'
    assert zeta(pi*1000, rounding=round_up) > 1
    assert zeta(3000, rounding=round_up) > 1
    assert zeta(pi*1000) == 1
    assert zeta(3000) == 1

def test_zeta_negative():
    mp.dps = 150
    a = -pi*10**40
    mp.dps = 15
    assert str(zeta(a)) == '2.55880492708712e+1233536161668617575553892558646631323374078'
    mp.dps = 50
    assert str(zeta(a)) == '2.5588049270871154960875033337384432038436330847333e+1233536161668617575553892558646631323374078'
    mp.dps = 15

def test_polygamma():
    mp.dps = 15
    psi0 = lambda z: psi(0,z)
    psi1 = lambda z: psi(1,z)
    assert psi0(3) == psi(0,3) == digamma(3)
    #assert psi2(3) == psi(2,3) == tetragamma(3)
    #assert psi3(3) == psi(3,3) == pentagamma(3)
    assert psi0(pi).ae(0.97721330794200673)
    assert psi0(-pi).ae(7.8859523853854902)
    assert psi0(-pi+1).ae(7.5676424992016996)
    assert psi0(pi+j).ae(1.04224048313859376 + 0.35853686544063749j)
    assert psi0(-pi-j).ae(1.3404026194821986 - 2.8824392476809402j)
    assert findroot(psi0, 1).ae(1.4616321449683622)
    assert psi0(1e-10).ae(-10000000000.57722)
    assert psi0(1e-40).ae(-1.000000000000000e+40)
    assert psi0(1e-10+1e-10j).ae(-5000000000.577215 + 5000000000.000000j)
    assert psi0(1e-40+1e-40j).ae(-5.000000000000000e+39 + 5.000000000000000e+39j)
    assert psi0(inf) == inf
    assert psi1(inf) == 0
    assert psi(2,inf) == 0
    assert psi1(pi).ae(0.37424376965420049)
    assert psi1(-pi).ae(53.030438740085385)
    assert psi1(pi+j).ae(0.32935710377142464 - 0.12222163911221135j)
    assert psi1(-pi-j).ae(-0.30065008356019703 + 0.01149892486928227j)
    assert (10**6*psi(4,1+10*pi*j)).ae(-6.1491803479004446 - 0.3921316371664063j)
    assert psi0(1+10*pi*j).ae(3.4473994217222650 + 1.5548808324857071j)
    assert isnan(psi0(nan))
    assert isnan(psi0(-inf))
    assert psi0(-100.5).ae(4.615124601338064)
    assert psi0(3+0j).ae(psi0(3))
    assert psi0(-100+3j).ae(4.6106071768714086321+3.1117510556817394626j)
    assert isnan(psi(2,mpc(0,inf)))
    assert isnan(psi(2,mpc(0,nan)))
    assert isnan(psi(2,mpc(0,-inf)))
    assert isnan(psi(2,mpc(1,inf)))
    assert isnan(psi(2,mpc(1,nan)))
    assert isnan(psi(2,mpc(1,-inf)))
    assert isnan(psi(2,mpc(inf,inf)))
    assert isnan(psi(2,mpc(nan,nan)))
    assert isnan(psi(2,mpc(-inf,-inf)))
    mp.dps = 30
    # issue #534
    assert digamma(-0.75+1j).ae(mpc('0.46317279488182026118963809283042317', '2.4821070143037957102007677817351115'))
    mp.dps = 15

def test_polygamma_high_prec():
    mp.dps = 100
    assert str(psi(0,pi)) == "0.9772133079420067332920694864061823436408346099943256380095232865318105924777141317302075654362928734"
    assert str(psi(10,pi)) == "-12.98876181434889529310283769414222588307175962213707170773803550518307617769657562747174101900659238"

def test_polygamma_identities():
    mp.dps = 15
    psi0 = lambda z: psi(0,z)
    psi1 = lambda z: psi(1,z)
    psi2 = lambda z: psi(2,z)
    assert psi0(0.5).ae(-euler-2*log(2))
    assert psi0(1).ae(-euler)
    assert psi1(0.5).ae(0.5*pi**2)
    assert psi1(1).ae(pi**2/6)
    assert psi1(0.25).ae(pi**2 + 8*catalan)
    assert psi2(1).ae(-2*apery)
    mp.dps = 20
    u = -182*apery+4*sqrt(3)*pi**3
    mp.dps = 15
    assert psi(2,5/6.).ae(u)
    assert psi(3,0.5).ae(pi**4)

def test_foxtrot_identity():
    # A test of the complex digamma function.
    # See http://mathworld.wolfram.com/FoxTrotSeries.html and
    # http://mathworld.wolfram.com/DigammaFunction.html
    psi0 = lambda z: psi(0,z)
    mp.dps = 50
    a = (-1)**fraction(1,3)
    b = (-1)**fraction(2,3)
    x = -psi0(0.5*a) - psi0(-0.5*b) + psi0(0.5*(1+a)) + psi0(0.5*(1-b))
    y = 2*pi*sech(0.5*sqrt(3)*pi)
    assert x.ae(y)
    mp.dps = 15

def test_polygamma_high_order():
    mp.dps = 100
    assert str(psi(50, pi)) == "-1344100348958402765749252447726432491812.641985273160531055707095989227897753035823152397679626136483"
    assert str(psi(50, pi + 14*e)) == "-0.00000000000000000189793739550804321623512073101895801993019919886375952881053090844591920308111549337295143780341396"
    assert str(psi(50, pi + 14*e*j)) == ("(-0.0000000000000000522516941152169248975225472155683565752375889510631513244785"
        "9377385233700094871256507814151956624433 - 0.00000000000000001813157041407010184"
        "702414110218205348527862196327980417757665282244728963891298080199341480881811613j)")
    mp.dps = 15
    assert str(psi(50, pi)) == "-1.34410034895841e+39"
    assert str(psi(50, pi + 14*e)) == "-1.89793739550804e-18"
    assert str(psi(50, pi + 14*e*j)) == "(-5.2251694115217e-17 - 1.81315704140701e-17j)"

def test_harmonic():
    mp.dps = 15
    assert harmonic(0) == 0
    assert harmonic(1) == 1
    assert harmonic(2) == 1.5
    assert harmonic(3).ae(1. + 1./2 + 1./3)
    assert harmonic(10**10).ae(23.603066594891989701)
    assert harmonic(10**1000).ae(2303.162308658947)
    assert harmonic(0.5).ae(2-2*log(2))
    assert harmonic(inf) == inf
    assert harmonic(2+0j) == 1.5+0j
    assert harmonic(1+2j).ae(1.4918071802755104+0.92080728264223022j)

def test_gamma_huge_1():
    mp.dps = 500
    x = mpf(10**10) / 7
    mp.dps = 15
    assert str(gamma(x)) == "6.26075321389519e+12458010678"
    mp.dps = 50
    assert str(gamma(x)) == "6.2607532138951929201303779291707455874010420783933e+12458010678"
    mp.dps = 15

def test_gamma_huge_2():
    mp.dps = 500
    x = mpf(10**100) / 19
    mp.dps = 15
    assert str(gamma(x)) == (\
        "1.82341134776679e+5172997469323364168990133558175077136829182824042201886051511"
        "9656908623426021308685461258226190190661")
    mp.dps = 50
    assert str(gamma(x)) == (\
        "1.82341134776678875374414910350027596939980412984e+5172997469323364168990133558"
        "1750771368291828240422018860515119656908623426021308685461258226190190661")

def test_gamma_huge_3():
    mp.dps = 500
    x = 10**80 // 3 + 10**70*j / 7
    mp.dps = 15
    y = gamma(x)
    assert str(y.real) == (\
        "-6.82925203918106e+2636286142112569524501781477865238132302397236429627932441916"
        "056964386399485392600")
    assert str(y.imag) == (\
        "8.54647143678418e+26362861421125695245017814778652381323023972364296279324419160"
        "56964386399485392600")
    mp.dps = 50
    y = gamma(x)
    assert str(y.real) == (\
        "-6.8292520391810548460682736226799637356016538421817e+26362861421125695245017814"
        "77865238132302397236429627932441916056964386399485392600")
    assert str(y.imag) == (\
        "8.5464714367841748507479306948130687511711420234015e+263628614211256952450178147"
        "7865238132302397236429627932441916056964386399485392600")

def test_gamma_huge_4():
    x = 3200+11500j
    mp.dps = 15
    assert str(gamma(x)) == \
        "(8.95783268539713e+5164 - 1.94678798329735e+5164j)"
    mp.dps = 50
    assert str(gamma(x)) == (\
        "(8.9578326853971339570292952697675570822206567327092e+5164"
        " - 1.9467879832973509568895402139429643650329524144794e+51"
        "64j)")
    mp.dps = 15

def test_gamma_huge_5():
    mp.dps = 500
    x = 10**60 * j / 3
    mp.dps = 15
    y = gamma(x)
    assert str(y.real) == "-3.27753899634941e-227396058973640224580963937571892628368354580620654233316839"
    assert str(y.imag) == "-7.1519888950416e-227396058973640224580963937571892628368354580620654233316841"
    mp.dps = 50
    y = gamma(x)
    assert str(y.real) == (\
        "-3.2775389963494132168950056995974690946983219123935e-22739605897364022458096393"
        "7571892628368354580620654233316839")
    assert str(y.imag) == (\
        "-7.1519888950415979749736749222530209713136588885897e-22739605897364022458096393"
        "7571892628368354580620654233316841")
    mp.dps = 15

def test_gamma_huge_7():
    mp.dps = 100
    a = 3 + j/mpf(10)**1000
    mp.dps = 15
    y = gamma(a)
    assert str(y.real) == "2.0"
    # wrong
    #assert str(y.imag) == "2.16735365342606e-1000"
    assert str(y.imag) == "1.84556867019693e-1000"
    mp.dps = 50
    y = gamma(a)
    assert str(y.real) == "2.0"
    #assert str(y.imag) == "2.1673536534260596065418805612488708028522563689298e-1000"
    assert str(y.imag) ==  "1.8455686701969342787869758198351951379156813281202e-1000"

def test_stieltjes():
    mp.dps = 15
    assert stieltjes(0).ae(+euler)
    mp.dps = 25
    assert stieltjes(1).ae('-0.07281584548367672486058637587')
    assert stieltjes(2).ae('-0.009690363192872318484530386035')
    assert stieltjes(3).ae('0.002053834420303345866160046543')
    assert stieltjes(4).ae('0.002325370065467300057468170178')
    mp.dps = 15
    assert stieltjes(1).ae(-0.07281584548367672486058637587)
    assert stieltjes(2).ae(-0.009690363192872318484530386035)
    assert stieltjes(3).ae(0.002053834420303345866160046543)
    assert stieltjes(4).ae(0.0023253700654673000574681701775)

def test_barnesg():
    mp.dps = 15
    assert barnesg(0) == barnesg(-1) == 0
    assert [superfac(i) for i in range(8)] == [1, 1, 2, 12, 288, 34560, 24883200, 125411328000]
    assert str(superfac(1000)) == '3.24570818422368e+1177245'
    assert isnan(barnesg(nan))
    assert isnan(superfac(nan))
    assert isnan(hyperfac(nan))
    assert barnesg(inf) == inf
    assert superfac(inf) == inf
    assert hyperfac(inf) == inf
    assert isnan(superfac(-inf))
    assert barnesg(0.7).ae(0.8068722730141471)
    assert barnesg(2+3j).ae(-0.17810213864082169+0.04504542715447838j)
    assert [hyperfac(n) for n in range(7)] == [1, 1, 4, 108, 27648, 86400000, 4031078400000]
    assert [hyperfac(n) for n in range(0,-7,-1)] == [1,1,-1,-4,108,27648,-86400000]
    a = barnesg(-3+0j)
    assert a == 0 and isinstance(a, mpc)
    a = hyperfac(-3+0j)
    assert a == -4 and isinstance(a, mpc)

def test_polylog():
    mp.dps = 15
    zs = [mpmathify(z) for z in [0, 0.5, 0.99, 4, -0.5, -4, 1j, 3+4j]]
    for z in zs: assert polylog(1, z).ae(-log(1-z))
    for z in zs: assert polylog(0, z).ae(z/(1-z))
    for z in zs: assert polylog(-1, z).ae(z/(1-z)**2)
    for z in zs: assert polylog(-2, z).ae(z*(1+z)/(1-z)**3)
    for z in zs: assert polylog(-3, z).ae(z*(1+4*z+z**2)/(1-z)**4)
    assert polylog(3, 7).ae(5.3192579921456754382-5.9479244480803301023j)
    assert polylog(3, -7).ae(-4.5693548977219423182)
    assert polylog(2, 0.9).ae(1.2997147230049587252)
    assert polylog(2, -0.9).ae(-0.75216317921726162037)
    assert polylog(2, 0.9j).ae(-0.17177943786580149299+0.83598828572550503226j)
    assert polylog(2, 1.1).ae(1.9619991013055685931-0.2994257606855892575j)
    assert polylog(2, -1.1).ae(-0.89083809026228260587)
    assert polylog(2, 1.1*sqrt(j)).ae(0.58841571107611387722+1.09962542118827026011j)
    assert polylog(-2, 0.9).ae(1710)
    assert polylog(-2, -0.9).ae(-90/6859.)
    assert polylog(3, 0.9).ae(1.0496589501864398696)
    assert polylog(-3, 0.9).ae(48690)
    assert polylog(-3, -4).ae(-0.0064)
    assert polylog(0.5+j/3, 0.5+j/2).ae(0.31739144796565650535 + 0.99255390416556261437j)
    assert polylog(3+4j,1).ae(zeta(3+4j))
    assert polylog(3+4j,-1).ae(-altzeta(3+4j))
    # issue 390
    assert polylog(1.5, -48.910886523731889).ae(-6.272992229311817)
    assert polylog(1.5, 200).ae(-8.349608319033686529 - 8.159694826434266042j)
    assert polylog(-2+0j, -2).ae(mpf(1)/13.5)
    assert polylog(-2+0j, 1.25).ae(-180)

def test_bell_polyexp():
    mp.dps = 15
    # TODO: more tests for polyexp
    assert (polyexp(0,1e-10)*10**10).ae(1.00000000005)
    assert (polyexp(1,1e-10)*10**10).ae(1.0000000001)
    assert polyexp(5,3j).ae(-607.7044517476176454+519.962786482001476087j)
    assert polyexp(-1,3.5).ae(12.09537536175543444)
    # bell(0,x) = 1
    assert bell(0,0) == 1
    assert bell(0,1) == 1
    assert bell(0,2) == 1
    assert bell(0,inf) == 1
    assert bell(0,-inf) == 1
    assert isnan(bell(0,nan))
    # bell(1,x) = x
    assert bell(1,4) == 4
    assert bell(1,0) == 0
    assert bell(1,inf) == inf
    assert bell(1,-inf) == -inf
    assert isnan(bell(1,nan))
    # bell(2,x) = x*(1+x)
    assert bell(2,-1) == 0
    assert bell(2,0) == 0
    # large orders / arguments
    assert bell(10) == 115975
    assert bell(10,1) == 115975
    assert bell(10, -8) == 11054008
    assert bell(5,-50) == -253087550
    assert bell(50,-50).ae('3.4746902914629720259e74')
    mp.dps = 80
    assert bell(50,-50) == 347469029146297202586097646631767227177164818163463279814268368579055777450
    assert bell(40,50) == 5575520134721105844739265207408344706846955281965031698187656176321717550
    assert bell(74) == 5006908024247925379707076470957722220463116781409659160159536981161298714301202
    mp.dps = 15
    assert bell(10,20j) == 7504528595600+15649605360020j
    # continuity of the generalization
    assert bell(0.5,0).ae(sinc(pi*0.5))

def test_primezeta():
    mp.dps = 15
    assert primezeta(0.9).ae(1.8388316154446882243 + 3.1415926535897932385j)
    assert primezeta(4).ae(0.076993139764246844943)
    assert primezeta(1) == inf
    assert primezeta(inf) == 0
    assert isnan(primezeta(nan))

def test_rs_zeta():
    mp.dps = 15
    assert zeta(0.5+100000j).ae(1.0730320148577531321 + 5.7808485443635039843j)
    assert zeta(0.75+100000j).ae(1.837852337251873704 + 1.9988492668661145358j)
    assert zeta(0.5+1000000j, derivative=3).ae(1647.7744105852674733 - 1423.1270943036622097j)
    assert zeta(1+1000000j, derivative=3).ae(3.4085866124523582894 - 18.179184721525947301j)
    assert zeta(1+1000000j, derivative=1).ae(-0.10423479366985452134 - 0.74728992803359056244j)
    assert zeta(0.5-1000000j, derivative=1).ae(11.636804066002521459 + 17.127254072212996004j)
    # Additional sanity tests using fp arithmetic.
    # Some more high-precision tests are found in the docstrings
    def ae(x, y, tol=1e-6):
        return abs(x-y) < tol*abs(y)
    assert ae(fp.zeta(0.5-100000j), 1.0730320148577531321 - 5.7808485443635039843j)
    assert ae(fp.zeta(0.75-100000j), 1.837852337251873704 - 1.9988492668661145358j)
    assert ae(fp.zeta(0.5+1e6j), 0.076089069738227100006 + 2.8051021010192989554j)
    assert ae(fp.zeta(0.5+1e6j, derivative=1), 11.636804066002521459 - 17.127254072212996004j)
    assert ae(fp.zeta(1+1e6j), 0.94738726251047891048 + 0.59421999312091832833j)
    assert ae(fp.zeta(1+1e6j, derivative=1), -0.10423479366985452134 - 0.74728992803359056244j)
    assert ae(fp.zeta(0.5+100000j, derivative=1), 10.766962036817482375 - 30.92705282105996714j)
    assert ae(fp.zeta(0.5+100000j, derivative=2), -119.40515625740538429 + 217.14780631141830251j)
    assert ae(fp.zeta(0.5+100000j, derivative=3), 1129.7550282628460881 - 1685.4736895169690346j)
    assert ae(fp.zeta(0.5+100000j, derivative=4), -10407.160819314958615 + 13777.786698628045085j)
    assert ae(fp.zeta(0.75+100000j, derivative=1), -0.41742276699594321475 - 6.4453816275049955949j)
    assert ae(fp.zeta(0.75+100000j, derivative=2), -9.214314279161977266 + 35.07290795337967899j)
    assert ae(fp.zeta(0.75+100000j, derivative=3), 110.61331857820103469 - 236.87847130518129926j)
    assert ae(fp.zeta(0.75+100000j, derivative=4), -1054.334275898559401 + 1769.9177890161596383j)

def test_siegelz():
    mp.dps = 15
    assert siegelz(100000).ae(5.87959246868176504171)
    assert siegelz(100000, derivative=2).ae(-54.1172711010126452832)
    assert siegelz(100000, derivative=3).ae(-278.930831343966552538)
    assert siegelz(100000+j,derivative=1).ae(678.214511857070283307-379.742160779916375413j)



def test_zeta_near_1():
    # Test for a former bug in mpf_zeta and mpc_zeta
    mp.dps = 15
    s1 = fadd(1, '1e-10', exact=True)
    s2 = fadd(1, '-1e-10', exact=True)
    s3 = fadd(1, '1e-10j', exact=True)
    assert zeta(s1).ae(1.000000000057721566490881444e10)
    assert zeta(s2).ae(-9.99999999942278433510574872e9)
    z = zeta(s3)
    assert z.real.ae(0.57721566490153286060)
    assert z.imag.ae(-9.9999999999999999999927184e9)
    mp.dps = 30
    s1 = fadd(1, '1e-50', exact=True)
    s2 = fadd(1, '-1e-50', exact=True)
    s3 = fadd(1, '1e-50j', exact=True)
    assert zeta(s1).ae('1e50')
    assert zeta(s2).ae('-1e50')
    z = zeta(s3)
    assert z.real.ae('0.57721566490153286060651209008240243104215933593992')
    assert z.imag.ae('-1e50')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\test_constructors.py ===
from datetime import timedelta
from itertools import product

import numpy as np
import pytest

from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit

from pandas import (
    Index,
    NaT,
    Timedelta,
    TimedeltaIndex,
    offsets,
    to_timedelta,
)
import pandas._testing as tm


class TestTimedeltaConstructorUnitKeyword:
    @pytest.mark.parametrize("unit", ["Y", "y", "M"])
    def test_unit_m_y_raises(self, unit):
        msg = "Units 'M', 'Y', and 'y' are no longer supported"

        with pytest.raises(ValueError, match=msg):
            Timedelta(10, unit)

        with pytest.raises(ValueError, match=msg):
            to_timedelta(10, unit)

        with pytest.raises(ValueError, match=msg):
            to_timedelta([1, 2], unit)

    @pytest.mark.parametrize(
        "unit,unit_depr",
        [
            ("h", "H"),
            ("min", "T"),
            ("s", "S"),
            ("ms", "L"),
            ("ns", "N"),
            ("us", "U"),
        ],
    )
    def test_units_H_T_S_L_N_U_deprecated(self, unit, unit_depr):
        # GH#52536
        msg = f"'{unit_depr}' is deprecated and will be removed in a future version."

        expected = Timedelta(1, unit=unit)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = Timedelta(1, unit=unit_depr)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "unit, np_unit",
        [(value, "W") for value in ["W", "w"]]
        + [(value, "D") for value in ["D", "d", "days", "day", "Days", "Day"]]
        + [
            (value, "m")
            for value in [
                "m",
                "minute",
                "min",
                "minutes",
                "Minute",
                "Min",
                "Minutes",
            ]
        ]
        + [
            (value, "s")
            for value in [
                "s",
                "seconds",
                "sec",
                "second",
                "Seconds",
                "Sec",
                "Second",
            ]
        ]
        + [
            (value, "ms")
            for value in [
                "ms",
                "milliseconds",
                "millisecond",
                "milli",
                "millis",
                "MS",
                "Milliseconds",
                "Millisecond",
                "Milli",
                "Millis",
            ]
        ]
        + [
            (value, "us")
            for value in [
                "us",
                "microseconds",
                "microsecond",
                "micro",
                "micros",
                "u",
                "US",
                "Microseconds",
                "Microsecond",
                "Micro",
                "Micros",
                "U",
            ]
        ]
        + [
            (value, "ns")
            for value in [
                "ns",
                "nanoseconds",
                "nanosecond",
                "nano",
                "nanos",
                "n",
                "NS",
                "Nanoseconds",
                "Nanosecond",
                "Nano",
                "Nanos",
                "N",
            ]
        ],
    )
    @pytest.mark.parametrize("wrapper", [np.array, list, Index])
    def test_unit_parser(self, unit, np_unit, wrapper):
        # validate all units, GH 6855, GH 21762
        # array-likes
        expected = TimedeltaIndex(
            [np.timedelta64(i, np_unit) for i in np.arange(5).tolist()],
            dtype="m8[ns]",
        )
        # TODO(2.0): the desired output dtype may have non-nano resolution
        msg = f"'{unit}' is deprecated and will be removed in a future version."

        if (unit, np_unit) in (("u", "us"), ("U", "us"), ("n", "ns"), ("N", "ns")):
            warn = FutureWarning
        else:
            warn = FutureWarning
            msg = "The 'unit' keyword in TimedeltaIndex construction is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            result = to_timedelta(wrapper(range(5)), unit=unit)
            tm.assert_index_equal(result, expected)
            result = TimedeltaIndex(wrapper(range(5)), unit=unit)
            tm.assert_index_equal(result, expected)

            str_repr = [f"{x}{unit}" for x in np.arange(5)]
            result = to_timedelta(wrapper(str_repr))
            tm.assert_index_equal(result, expected)
            result = to_timedelta(wrapper(str_repr))
            tm.assert_index_equal(result, expected)

            # scalar
            expected = Timedelta(np.timedelta64(2, np_unit).astype("timedelta64[ns]"))
            result = to_timedelta(2, unit=unit)
            assert result == expected
            result = Timedelta(2, unit=unit)
            assert result == expected

            result = to_timedelta(f"2{unit}")
            assert result == expected
            result = Timedelta(f"2{unit}")
            assert result == expected


def test_construct_from_kwargs_overflow():
    # GH#55503
    msg = "seconds=86400000000000000000, milliseconds=0, microseconds=0, nanoseconds=0"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(days=10**6)
    msg = "seconds=60000000000000000000, milliseconds=0, microseconds=0, nanoseconds=0"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(minutes=10**9)


def test_construct_with_weeks_unit_overflow():
    # GH#47268 don't silently wrap around
    with pytest.raises(OutOfBoundsTimedelta, match="without overflow"):
        Timedelta(1000000000000000000, unit="W")

    with pytest.raises(OutOfBoundsTimedelta, match="without overflow"):
        Timedelta(1000000000000000000.0, unit="W")


def test_construct_from_td64_with_unit():
    # ignore the unit, as it may cause silently overflows leading to incorrect
    #  results, and in non-overflow cases is irrelevant GH#46827
    obj = np.timedelta64(123456789000000000, "h")

    with pytest.raises(OutOfBoundsTimedelta, match="123456789000000000 hours"):
        Timedelta(obj, unit="ps")

    with pytest.raises(OutOfBoundsTimedelta, match="123456789000000000 hours"):
        Timedelta(obj, unit="ns")

    with pytest.raises(OutOfBoundsTimedelta, match="123456789000000000 hours"):
        Timedelta(obj)


def test_from_td64_retain_resolution():
    # case where we retain millisecond resolution
    obj = np.timedelta64(12345, "ms")

    td = Timedelta(obj)
    assert td._value == obj.view("i8")
    assert td._creso == NpyDatetimeUnit.NPY_FR_ms.value

    # Case where we cast to nearest-supported reso
    obj2 = np.timedelta64(1234, "D")
    td2 = Timedelta(obj2)
    assert td2._creso == NpyDatetimeUnit.NPY_FR_s.value
    assert td2 == obj2
    assert td2.days == 1234

    # Case that _would_ overflow if we didn't support non-nano
    obj3 = np.timedelta64(1000000000000000000, "us")
    td3 = Timedelta(obj3)
    assert td3.total_seconds() == 1000000000000
    assert td3._creso == NpyDatetimeUnit.NPY_FR_us.value


def test_from_pytimedelta_us_reso():
    # pytimedelta has microsecond resolution, so Timedelta(pytd) inherits that
    td = timedelta(days=4, minutes=3)
    result = Timedelta(td)
    assert result.to_pytimedelta() == td
    assert result._creso == NpyDatetimeUnit.NPY_FR_us.value


def test_from_tick_reso():
    tick = offsets.Nano()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_ns.value

    tick = offsets.Micro()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_us.value

    tick = offsets.Milli()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_ms.value

    tick = offsets.Second()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value

    # everything above Second gets cast to the closest supported reso: second
    tick = offsets.Minute()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value

    tick = offsets.Hour()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value

    tick = offsets.Day()
    assert Timedelta(tick)._creso == NpyDatetimeUnit.NPY_FR_s.value


def test_construction():
    expected = np.timedelta64(10, "D").astype("m8[ns]").view("i8")
    assert Timedelta(10, unit="d")._value == expected
    assert Timedelta(10.0, unit="d")._value == expected
    assert Timedelta("10 days")._value == expected
    assert Timedelta(days=10)._value == expected
    assert Timedelta(days=10.0)._value == expected

    expected += np.timedelta64(10, "s").astype("m8[ns]").view("i8")
    assert Timedelta("10 days 00:00:10")._value == expected
    assert Timedelta(days=10, seconds=10)._value == expected
    assert Timedelta(days=10, milliseconds=10 * 1000)._value == expected
    assert Timedelta(days=10, microseconds=10 * 1000 * 1000)._value == expected

    # rounding cases
    assert Timedelta(82739999850000)._value == 82739999850000
    assert "0 days 22:58:59.999850" in str(Timedelta(82739999850000))
    assert Timedelta(123072001000000)._value == 123072001000000
    assert "1 days 10:11:12.001" in str(Timedelta(123072001000000))

    # string conversion with/without leading zero
    # GH#9570
    assert Timedelta("0:00:00") == timedelta(hours=0)
    assert Timedelta("00:00:00") == timedelta(hours=0)
    assert Timedelta("-1:00:00") == -timedelta(hours=1)
    assert Timedelta("-01:00:00") == -timedelta(hours=1)

    # more strings & abbrevs
    # GH#8190
    assert Timedelta("1 h") == timedelta(hours=1)
    assert Timedelta("1 hour") == timedelta(hours=1)
    assert Timedelta("1 hr") == timedelta(hours=1)
    assert Timedelta("1 hours") == timedelta(hours=1)
    assert Timedelta("-1 hours") == -timedelta(hours=1)
    assert Timedelta("1 m") == timedelta(minutes=1)
    assert Timedelta("1.5 m") == timedelta(seconds=90)
    assert Timedelta("1 minute") == timedelta(minutes=1)
    assert Timedelta("1 minutes") == timedelta(minutes=1)
    assert Timedelta("1 s") == timedelta(seconds=1)
    assert Timedelta("1 second") == timedelta(seconds=1)
    assert Timedelta("1 seconds") == timedelta(seconds=1)
    assert Timedelta("1 ms") == timedelta(milliseconds=1)
    assert Timedelta("1 milli") == timedelta(milliseconds=1)
    assert Timedelta("1 millisecond") == timedelta(milliseconds=1)
    assert Timedelta("1 us") == timedelta(microseconds=1)
    assert Timedelta("1 µs") == timedelta(microseconds=1)
    assert Timedelta("1 micros") == timedelta(microseconds=1)
    assert Timedelta("1 microsecond") == timedelta(microseconds=1)
    assert Timedelta("1.5 microsecond") == Timedelta("00:00:00.000001500")
    assert Timedelta("1 ns") == Timedelta("00:00:00.000000001")
    assert Timedelta("1 nano") == Timedelta("00:00:00.000000001")
    assert Timedelta("1 nanosecond") == Timedelta("00:00:00.000000001")

    # combos
    assert Timedelta("10 days 1 hour") == timedelta(days=10, hours=1)
    assert Timedelta("10 days 1 h") == timedelta(days=10, hours=1)
    assert Timedelta("10 days 1 h 1m 1s") == timedelta(
        days=10, hours=1, minutes=1, seconds=1
    )
    assert Timedelta("-10 days 1 h 1m 1s") == -timedelta(
        days=10, hours=1, minutes=1, seconds=1
    )
    assert Timedelta("-10 days 1 h 1m 1s") == -timedelta(
        days=10, hours=1, minutes=1, seconds=1
    )
    assert Timedelta("-10 days 1 h 1m 1s 3us") == -timedelta(
        days=10, hours=1, minutes=1, seconds=1, microseconds=3
    )
    assert Timedelta("-10 days 1 h 1.5m 1s 3us") == -timedelta(
        days=10, hours=1, minutes=1, seconds=31, microseconds=3
    )

    # Currently invalid as it has a - on the hh:mm:dd part
    # (only allowed on the days)
    msg = "only leading negative signs are allowed"
    with pytest.raises(ValueError, match=msg):
        Timedelta("-10 days -1 h 1.5m 1s 3us")

    # only leading neg signs are allowed
    with pytest.raises(ValueError, match=msg):
        Timedelta("10 days -1 h 1.5m 1s 3us")

    # no units specified
    msg = "no units specified"
    with pytest.raises(ValueError, match=msg):
        Timedelta("3.1415")

    # invalid construction
    msg = "cannot construct a Timedelta"
    with pytest.raises(ValueError, match=msg):
        Timedelta()

    msg = "unit abbreviation w/o a number"
    with pytest.raises(ValueError, match=msg):
        Timedelta("foo")

    msg = (
        "cannot construct a Timedelta from "
        "the passed arguments, allowed keywords are "
    )
    with pytest.raises(ValueError, match=msg):
        Timedelta(day=10)

    # floats
    expected = np.timedelta64(10, "s").astype("m8[ns]").view("i8") + np.timedelta64(
        500, "ms"
    ).astype("m8[ns]").view("i8")
    assert Timedelta(10.5, unit="s")._value == expected

    # offset
    assert to_timedelta(offsets.Hour(2)) == Timedelta(hours=2)
    assert Timedelta(offsets.Hour(2)) == Timedelta(hours=2)
    assert Timedelta(offsets.Second(2)) == Timedelta(seconds=2)

    # GH#11995: unicode
    expected = Timedelta("1h")
    result = Timedelta("1h")
    assert result == expected
    assert to_timedelta(offsets.Hour(2)) == Timedelta("0 days, 02:00:00")

    msg = "unit abbreviation w/o a number"
    with pytest.raises(ValueError, match=msg):
        Timedelta("foo bar")


@pytest.mark.parametrize(
    "item",
    list(
        {
            "days": "D",
            "seconds": "s",
            "microseconds": "us",
            "milliseconds": "ms",
            "minutes": "m",
            "hours": "h",
            "weeks": "W",
        }.items()
    ),
)
@pytest.mark.parametrize(
    "npdtype", [np.int64, np.int32, np.int16, np.float64, np.float32, np.float16]
)
def test_td_construction_with_np_dtypes(npdtype, item):
    # GH#8757: test construction with np dtypes
    pykwarg, npkwarg = item
    expected = np.timedelta64(1, npkwarg).astype("m8[ns]").view("i8")
    assert Timedelta(**{pykwarg: npdtype(1)})._value == expected


@pytest.mark.parametrize(
    "val",
    [
        "1s",
        "-1s",
        "1us",
        "-1us",
        "1 day",
        "-1 day",
        "-23:59:59.999999",
        "-1 days +23:59:59.999999",
        "-1ns",
        "1ns",
        "-23:59:59.999999999",
    ],
)
def test_td_from_repr_roundtrip(val):
    # round-trip both for string and value
    td = Timedelta(val)
    assert Timedelta(td._value) == td

    assert Timedelta(str(td)) == td
    assert Timedelta(td._repr_base(format="all")) == td
    assert Timedelta(td._repr_base()) == td


def test_overflow_on_construction():
    # GH#3374
    value = Timedelta("1day")._value * 20169940
    msg = "Cannot cast 1742682816000000000000 from ns to 'ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(value)

    # xref GH#17637
    msg = "Cannot cast 139993 from D to 'ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        Timedelta(7 * 19999, unit="D")

    # used to overflow before non-ns support
    td = Timedelta(timedelta(days=13 * 19999))
    assert td._creso == NpyDatetimeUnit.NPY_FR_us.value
    assert td.days == 13 * 19999


@pytest.mark.parametrize(
    "val, unit",
    [
        (15251, "W"),  # 1
        (106752, "D"),  # change from previous:
        (2562048, "h"),  # 0 hours
        (153722868, "m"),  # 13 minutes
        (9223372037, "s"),  # 44 seconds
    ],
)
def test_construction_out_of_bounds_td64ns(val, unit):
    # TODO: parametrize over units just above/below the implementation bounds
    #  once GH#38964 is resolved

    # Timedelta.max is just under 106752 days
    td64 = np.timedelta64(val, unit)
    assert td64.astype("m8[ns]").view("i8") < 0  # i.e. naive astype will be wrong

    td = Timedelta(td64)
    if unit != "M":
        # with unit="M" the conversion to "s" is poorly defined
        #  (and numpy issues DeprecationWarning)
        assert td.asm8 == td64
    assert td.asm8.dtype == "m8[s]"
    msg = r"Cannot cast 1067\d\d days .* to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td.as_unit("ns")

    # But just back in bounds and we are OK
    assert Timedelta(td64 - 1) == td64 - 1

    td64 *= -1
    assert td64.astype("m8[ns]").view("i8") > 0  # i.e. naive astype will be wrong

    td2 = Timedelta(td64)
    msg = r"Cannot cast -1067\d\d days .* to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td2.as_unit("ns")

    # But just back in bounds and we are OK
    assert Timedelta(td64 + 1) == td64 + 1


@pytest.mark.parametrize(
    "val, unit",
    [
        (15251 * 10**9, "W"),
        (106752 * 10**9, "D"),
        (2562048 * 10**9, "h"),
        (153722868 * 10**9, "m"),
    ],
)
def test_construction_out_of_bounds_td64s(val, unit):
    td64 = np.timedelta64(val, unit)
    with pytest.raises(OutOfBoundsTimedelta, match=str(td64)):
        Timedelta(td64)

    # But just back in bounds and we are OK
    assert Timedelta(td64 - 10**9) == td64 - 10**9


@pytest.mark.parametrize(
    "fmt,exp",
    [
        (
            "P6DT0H50M3.010010012S",
            Timedelta(
                days=6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
        ),
        (
            "P-6DT0H50M3.010010012S",
            Timedelta(
                days=-6,
                minutes=50,
                seconds=3,
                milliseconds=10,
                microseconds=10,
                nanoseconds=12,
            ),
        ),
        ("P4DT12H30M5S", Timedelta(days=4, hours=12, minutes=30, seconds=5)),
        ("P0DT0H0M0.000000123S", Timedelta(nanoseconds=123)),
        ("P0DT0H0M0.00001S", Timedelta(microseconds=10)),
        ("P0DT0H0M0.001S", Timedelta(milliseconds=1)),
        ("P0DT0H1M0S", Timedelta(minutes=1)),
        ("P1DT25H61M61S", Timedelta(days=1, hours=25, minutes=61, seconds=61)),
        ("PT1S", Timedelta(seconds=1)),
        ("PT0S", Timedelta(seconds=0)),
        ("P1WT0S", Timedelta(days=7, seconds=0)),
        ("P1D", Timedelta(days=1)),
        ("P1DT1H", Timedelta(days=1, hours=1)),
        ("P1W", Timedelta(days=7)),
        ("PT300S", Timedelta(seconds=300)),
        ("P1DT0H0M00000000000S", Timedelta(days=1)),
        ("PT-6H3M", Timedelta(hours=-6, minutes=3)),
        ("-PT6H3M", Timedelta(hours=-6, minutes=-3)),
        ("-PT-6H+3M", Timedelta(hours=6, minutes=-3)),
    ],
)
def test_iso_constructor(fmt, exp):
    assert Timedelta(fmt) == exp


@pytest.mark.parametrize(
    "fmt",
    [
        "PPPPPPPPPPPP",
        "PDTHMS",
        "P0DT999H999M999S",
        "P1DT0H0M0.0000000000000S",
        "P1DT0H0M0.S",
        "P",
        "-P",
    ],
)
def test_iso_constructor_raises(fmt):
    msg = f"Invalid ISO 8601 Duration format - {fmt}"
    with pytest.raises(ValueError, match=msg):
        Timedelta(fmt)


@pytest.mark.parametrize(
    "constructed_td, conversion",
    [
        (Timedelta(nanoseconds=100), "100ns"),
        (
            Timedelta(
                days=1,
                hours=1,
                minutes=1,
                weeks=1,
                seconds=1,
                milliseconds=1,
                microseconds=1,
                nanoseconds=1,
            ),
            694861001001001,
        ),
        (Timedelta(microseconds=1) + Timedelta(nanoseconds=1), "1us1ns"),
        (Timedelta(microseconds=1) - Timedelta(nanoseconds=1), "999ns"),
        (Timedelta(microseconds=1) + 5 * Timedelta(nanoseconds=-2), "990ns"),
    ],
)
def test_td_constructor_on_nanoseconds(constructed_td, conversion):
    # GH#9273
    assert constructed_td == Timedelta(conversion)


def test_td_constructor_value_error():
    msg = "Invalid type <class 'str'>. Must be int or float."
    with pytest.raises(TypeError, match=msg):
        Timedelta(nanoseconds="abc")


def test_timedelta_constructor_identity():
    # Test for #30543
    expected = Timedelta(np.timedelta64(1, "s"))
    result = Timedelta(expected)
    assert result is expected


def test_timedelta_pass_td_and_kwargs_raises():
    # don't silently ignore the kwargs GH#48898
    td = Timedelta(days=1)
    msg = (
        "Cannot pass both a Timedelta input and timedelta keyword arguments, "
        r"got \['days'\]"
    )
    with pytest.raises(ValueError, match=msg):
        Timedelta(td, days=2)


@pytest.mark.parametrize(
    "constructor, value, unit, expectation",
    [
        (Timedelta, "10s", "ms", (ValueError, "unit must not be specified")),
        (to_timedelta, "10s", "ms", (ValueError, "unit must not be specified")),
        (to_timedelta, ["1", 2, 3], "s", (ValueError, "unit must not be specified")),
    ],
)
def test_string_with_unit(constructor, value, unit, expectation):
    exp, match = expectation
    with pytest.raises(exp, match=match):
        _ = constructor(value, unit=unit)


@pytest.mark.parametrize(
    "value",
    [
        "".join(elements)
        for repetition in (1, 2)
        for elements in product("+-, ", repeat=repetition)
    ],
)
def test_string_without_numbers(value):
    # GH39710 Timedelta input string with only symbols and no digits raises an error
    msg = (
        "symbols w/o a number"
        if value != "--"
        else "only leading negative signs are allowed"
    )
    with pytest.raises(ValueError, match=msg):
        Timedelta(value)


def test_timedelta_new_npnat():
    # GH#48898
    nat = np.timedelta64("NaT", "h")
    assert Timedelta(nat) is NaT


def test_subclass_respected():
    # GH#49579
    class MyCustomTimedelta(Timedelta):
        pass

    td = MyCustomTimedelta("1 minute")
    assert isinstance(td, MyCustomTimedelta)


def test_non_nano_value():
    # https://github.com/pandas-dev/pandas/issues/49076
    result = Timedelta(10, unit="D").as_unit("s").value
    # `.value` shows nanoseconds, even though unit is 's'
    assert result == 864000000000000

    # out-of-nanoseconds-bounds `.value` raises informative message
    msg = (
        r"Cannot convert Timedelta to nanoseconds without overflow. "
        r"Use `.asm8.view\('i8'\)` to cast represent Timedelta in its "
        r"own unit \(here, s\).$"
    )
    td = Timedelta(1_000, "D").as_unit("s") * 1_000
    with pytest.raises(OverflowError, match=msg):
        td.value
    # check that the suggested workaround actually works
    result = td.asm8.view("i8")
    assert result == 86400000000

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_rootoftools.py ===
"""Tests for the implementation of RootOf class and related tools. """

from sympy.polys.polytools import Poly
import sympy.polys.rootoftools as rootoftools
from sympy.polys.rootoftools import (rootof, RootOf, CRootOf, RootSum,
    _pure_key_dict as D)

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    GeneratorsNeeded,
    PolynomialError,
)

from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import tan
from sympy.integrals.integrals import Integral
from sympy.polys.orthopolys import legendre_poly
from sympy.solvers.solvers import solve


from sympy.testing.pytest import raises, slow
from sympy.core.expr import unchanged

from sympy.abc import a, b, x, y, z, r


def test_CRootOf___new__():
    assert rootof(x, 0) == 0
    assert rootof(x, -1) == 0

    assert rootof(x, S.Zero) == 0

    assert rootof(x - 1, 0) == 1
    assert rootof(x - 1, -1) == 1

    assert rootof(x + 1, 0) == -1
    assert rootof(x + 1, -1) == -1

    assert rootof(x**2 + 2*x + 3, 0) == -1 - I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, 1) == -1 + I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, -1) == -1 + I*sqrt(2)
    assert rootof(x**2 + 2*x + 3, -2) == -1 - I*sqrt(2)

    r = rootof(x**2 + 2*x + 3, 0, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, 1, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, -1, radicals=False)
    assert isinstance(r, RootOf) is True

    r = rootof(x**2 + 2*x + 3, -2, radicals=False)
    assert isinstance(r, RootOf) is True

    assert rootof((x - 1)*(x + 1), 0, radicals=False) == -1
    assert rootof((x - 1)*(x + 1), 1, radicals=False) == 1
    assert rootof((x - 1)*(x + 1), -1, radicals=False) == 1
    assert rootof((x - 1)*(x + 1), -2, radicals=False) == -1

    assert rootof((x - 1)*(x + 1), 0, radicals=True) == -1
    assert rootof((x - 1)*(x + 1), 1, radicals=True) == 1
    assert rootof((x - 1)*(x + 1), -1, radicals=True) == 1
    assert rootof((x - 1)*(x + 1), -2, radicals=True) == -1

    assert rootof((x - 1)*(x**3 + x + 3), 0) == rootof(x**3 + x + 3, 0)
    assert rootof((x - 1)*(x**3 + x + 3), 1) == 1
    assert rootof((x - 1)*(x**3 + x + 3), 2) == rootof(x**3 + x + 3, 1)
    assert rootof((x - 1)*(x**3 + x + 3), 3) == rootof(x**3 + x + 3, 2)
    assert rootof((x - 1)*(x**3 + x + 3), -1) == rootof(x**3 + x + 3, 2)
    assert rootof((x - 1)*(x**3 + x + 3), -2) == rootof(x**3 + x + 3, 1)
    assert rootof((x - 1)*(x**3 + x + 3), -3) == 1
    assert rootof((x - 1)*(x**3 + x + 3), -4) == rootof(x**3 + x + 3, 0)

    assert rootof(x**4 + 3*x**3, 0) == -3
    assert rootof(x**4 + 3*x**3, 1) == 0
    assert rootof(x**4 + 3*x**3, 2) == 0
    assert rootof(x**4 + 3*x**3, 3) == 0

    raises(GeneratorsNeeded, lambda: rootof(0, 0))
    raises(GeneratorsNeeded, lambda: rootof(1, 0))

    raises(PolynomialError, lambda: rootof(Poly(0, x), 0))
    raises(PolynomialError, lambda: rootof(Poly(1, x), 0))
    raises(PolynomialError, lambda: rootof(x - y, 0))
    # issue 8617
    raises(PolynomialError, lambda: rootof(exp(x), 0))

    raises(NotImplementedError, lambda: rootof(x**3 - x + sqrt(2), 0))
    raises(NotImplementedError, lambda: rootof(x**3 - x + I, 0))

    raises(IndexError, lambda: rootof(x**2 - 1, -4))
    raises(IndexError, lambda: rootof(x**2 - 1, -3))
    raises(IndexError, lambda: rootof(x**2 - 1, 2))
    raises(IndexError, lambda: rootof(x**2 - 1, 3))
    raises(ValueError, lambda: rootof(x**2 - 1, x))

    assert rootof(Poly(x - y, x), 0) == y

    assert rootof(Poly(x**2 - y, x), 0) == -sqrt(y)
    assert rootof(Poly(x**2 - y, x), 1) == sqrt(y)

    assert rootof(Poly(x**3 - y, x), 0) == y**Rational(1, 3)

    assert rootof(y*x**3 + y*x + 2*y, x, 0) == -1
    raises(NotImplementedError, lambda: rootof(x**3 + x + 2*y, x, 0))

    assert rootof(x**3 + x + 1, 0).is_commutative is True


def test_CRootOf_attributes():
    r = rootof(x**3 + x + 3, 0)
    assert r.is_number
    assert r.free_symbols == set()
    # if the following assertion fails then multivariate polynomials
    # are apparently supported and the RootOf.free_symbols routine
    # should be changed to return whatever symbols would not be
    # the PurePoly dummy symbol
    raises(NotImplementedError, lambda: rootof(Poly(x**3 + y*x + 1, x), 0))


def test_CRootOf___eq__():
    assert (rootof(x**3 + x + 3, 0) == rootof(x**3 + x + 3, 0)) is True
    assert (rootof(x**3 + x + 3, 0) == rootof(x**3 + x + 3, 1)) is False
    assert (rootof(x**3 + x + 3, 1) == rootof(x**3 + x + 3, 1)) is True
    assert (rootof(x**3 + x + 3, 1) == rootof(x**3 + x + 3, 2)) is False
    assert (rootof(x**3 + x + 3, 2) == rootof(x**3 + x + 3, 2)) is True

    assert (rootof(x**3 + x + 3, 0) == rootof(y**3 + y + 3, 0)) is True
    assert (rootof(x**3 + x + 3, 0) == rootof(y**3 + y + 3, 1)) is False
    assert (rootof(x**3 + x + 3, 1) == rootof(y**3 + y + 3, 1)) is True
    assert (rootof(x**3 + x + 3, 1) == rootof(y**3 + y + 3, 2)) is False
    assert (rootof(x**3 + x + 3, 2) == rootof(y**3 + y + 3, 2)) is True


def test_CRootOf___eval_Eq__():
    f = Function('f')
    eq = x**3 + x + 3
    r = rootof(eq, 2)
    r1 = rootof(eq, 1)
    assert Eq(r, r1) is S.false
    assert Eq(r, r) is S.true
    assert unchanged(Eq, r, x)
    assert Eq(r, 0) is S.false
    assert Eq(r, S.Infinity) is S.false
    assert Eq(r, I) is S.false
    assert unchanged(Eq, r, f(0))
    sol = solve(eq)
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.false
    r = rootof(eq, 0)
    for s in sol:
        if s.is_real:
            assert Eq(r, s) is S.true
    eq = x**3 + x + 1
    sol = solve(eq)
    assert [Eq(rootof(eq, i), j) for i in range(3) for j in sol
        ].count(True) == 3
    assert Eq(rootof(eq, 0), 1 + S.ImaginaryUnit) == False


def test_CRootOf_is_real():
    assert rootof(x**3 + x + 3, 0).is_real is True
    assert rootof(x**3 + x + 3, 1).is_real is False
    assert rootof(x**3 + x + 3, 2).is_real is False


def test_CRootOf_is_complex():
    assert rootof(x**3 + x + 3, 0).is_complex is True


def test_CRootOf_is_algebraic():
    assert rootof(x**3 + x + 3, 0).is_algebraic is True
    assert rootof(x**3 + x + 3, 1).is_algebraic is True
    assert rootof(x**3 + x + 3, 2).is_algebraic is True


def test_CRootOf_subs():
    assert rootof(x**3 + x + 1, 0).subs(x, y) == rootof(y**3 + y + 1, 0)


def test_CRootOf_diff():
    assert rootof(x**3 + x + 1, 0).diff(x) == 0
    assert rootof(x**3 + x + 1, 0).diff(y) == 0

@slow
def test_CRootOf_evalf():
    real = rootof(x**3 + x + 3, 0).evalf(n=20)

    assert real.epsilon_eq(Float("-1.2134116627622296341"))

    re, im = rootof(x**3 + x + 3, 1).evalf(n=20).as_real_imag()

    assert re.epsilon_eq( Float("0.60670583138111481707"))
    assert im.epsilon_eq(-Float("1.45061224918844152650"))

    re, im = rootof(x**3 + x + 3, 2).evalf(n=20).as_real_imag()

    assert re.epsilon_eq(Float("0.60670583138111481707"))
    assert im.epsilon_eq(Float("1.45061224918844152650"))

    p = legendre_poly(4, x, polys=True)
    roots = [str(r.n(17)) for r in p.real_roots()]
    # magnitudes are given by
    # sqrt(3/S(7) - 2*sqrt(6/S(5))/7)
    #   and
    # sqrt(3/S(7) + 2*sqrt(6/S(5))/7)
    assert roots == [
            "-0.86113631159405258",
            "-0.33998104358485626",
             "0.33998104358485626",
             "0.86113631159405258",
             ]

    re = rootof(x**5 - 5*x + 12, 0).evalf(n=20)
    assert re.epsilon_eq(Float("-1.84208596619025438271"))

    re, im = rootof(x**5 - 5*x + 12, 1).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("-0.351854240827371999559"))
    assert im.epsilon_eq(Float("-1.709561043370328882010"))

    re, im = rootof(x**5 - 5*x + 12, 2).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("-0.351854240827371999559"))
    assert im.epsilon_eq(Float("+1.709561043370328882010"))

    re, im = rootof(x**5 - 5*x + 12, 3).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("+1.272897223922499190910"))
    assert im.epsilon_eq(Float("-0.719798681483861386681"))

    re, im = rootof(x**5 - 5*x + 12, 4).evalf(n=20).as_real_imag()
    assert re.epsilon_eq(Float("+1.272897223922499190910"))
    assert im.epsilon_eq(Float("+0.719798681483861386681"))

    # issue 6393
    assert str(rootof(x**5 + 2*x**4 + x**3 - 68719476736, 0).n(3)) == '147.'
    eq = (531441*x**11 + 3857868*x**10 + 13730229*x**9 + 32597882*x**8 +
        55077472*x**7 + 60452000*x**6 + 32172064*x**5 - 4383808*x**4 -
        11942912*x**3 - 1506304*x**2 + 1453312*x + 512)
    a, b = rootof(eq, 1).n(2).as_real_imag()
    c, d = rootof(eq, 2).n(2).as_real_imag()
    assert a == c
    assert b < d
    assert b == -d
    # issue 6451
    r = rootof(legendre_poly(64, x), 7)
    assert r.n(2) == r.n(100).n(2)
    # issue 9019
    r0 = rootof(x**2 + 1, 0, radicals=False)
    r1 = rootof(x**2 + 1, 1, radicals=False)
    assert r0.n(4) == Float(-1.0, 4) * I
    assert r1.n(4) == Float(1.0, 4) * I

    # make sure verification is used in case a max/min traps the "root"
    assert str(rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0).n(3)) == '-0.976'

    # watch out for UnboundLocalError
    c = CRootOf(90720*x**6 - 4032*x**4 + 84*x**2 - 1, 0)
    assert c._eval_evalf(2)  # doesn't fail

    # watch out for imaginary parts that don't want to evaluate
    assert str(RootOf(x**16 + 32*x**14 + 508*x**12 + 5440*x**10 +
        39510*x**8 + 204320*x**6 + 755548*x**4 + 1434496*x**2 +
        877969, 10).n(2)) == '-3.4*I'
    assert abs(RootOf(x**4 + 10*x**2 + 1, 0).n(2)) < 0.4

    # check reset and args
    r = [RootOf(x**3 + x + 3, i) for i in range(3)]
    r[0]._reset()
    for ri in r:
        i = ri._get_interval()
        ri.n(2)
        assert i != ri._get_interval()
        ri._reset()
        assert i == ri._get_interval()
        assert i == i.func(*i.args)


def test_issue_24978():
    # Irreducible poly with negative leading coeff is normalized
    # (factor of -1 is extracted), before being stored as CRootOf.poly.
    f = -x**2 + 2
    r = CRootOf(f, 0)
    assert r.poly.as_expr() == x**2 - 2
    # An action that prompts calculation of an interval puts r.poly in
    # the cache.
    r.n()
    assert r.poly in rootoftools._reals_cache


def test_CRootOf_evalf_caching_bug():
    r = rootof(x**5 - 5*x + 12, 1)
    r.n()
    a = r._get_interval()
    r = rootof(x**5 - 5*x + 12, 1)
    r.n()
    b = r._get_interval()
    assert a == b


def test_CRootOf_real_roots():
    assert Poly(x**5 + x + 1).real_roots() == [rootof(x**3 - x**2 + 1, 0)]
    assert Poly(x**5 + x + 1).real_roots(radicals=False) == [rootof(
        x**3 - x**2 + 1, 0)]

    # https://github.com/sympy/sympy/issues/20902
    p = Poly(-3*x**4 - 10*x**3 - 12*x**2 - 6*x - 1, x, domain='ZZ')
    assert CRootOf.real_roots(p) == [S(-1), S(-1), S(-1), S(-1)/3]

    # with real algebraic coefficients
    assert Poly(x**3 + sqrt(2)*x**2 - 1, x, extension=True).real_roots() == [
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 0)
    ]
    assert Poly(x**5 + sqrt(2) * x**3 - 1, x, extension=True).real_roots() == [
        rootof(x**10 - 2*x**6 - 2*x**5 + 1, 0)
    ]
    r = rootof(y**5 + y**3 - 1, 0)
    assert Poly(x**5 + r*x - 1, x, extension=True).real_roots() ==\
    [
        rootof(x**25 - 5*x**20 + x**17 + 10*x**15 - 3*x**12 -
               10*x**10 + 3*x**7 + 6*x**5 - x**2 - 1, 0)
    ]
    # roots with multiplicity
    assert Poly((x-1) * (x-sqrt(2))**2, x, extension=True).real_roots() ==\
    [
        S(1), sqrt(2), sqrt(2)
    ]


def test_CRootOf_all_roots():
    assert Poly(x**5 + x + 1).all_roots() == [
        rootof(x**3 - x**2 + 1, 0),
        Rational(-1, 2) - sqrt(3)*I/2,
        Rational(-1, 2) + sqrt(3)*I/2,
        rootof(x**3 - x**2 + 1, 1),
        rootof(x**3 - x**2 + 1, 2),
    ]

    assert Poly(x**5 + x + 1).all_roots(radicals=False) == [
        rootof(x**3 - x**2 + 1, 0),
        rootof(x**2 + x + 1, 0, radicals=False),
        rootof(x**2 + x + 1, 1, radicals=False),
        rootof(x**3 - x**2 + 1, 1),
        rootof(x**3 - x**2 + 1, 2),
    ]

    # with real algebraic coefficients
    assert Poly(x**3 + sqrt(2)*x**2 - 1, x, extension=True).all_roots() ==\
    [
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 0),
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 2),
        rootof(x**6 - 2*x**4 - 2*x**3 + 1, 3)
    ]
    # roots with multiplicity
    assert Poly((x-1) * (x-sqrt(2))**2 * (x-I) * (x+I), x, extension=True).all_roots() ==\
    [
        S(1), sqrt(2), sqrt(2), -I, I
    ]

    # imaginary algebraic coeffs (gaussian domain)
    assert Poly(x**2 - I/2, x, extension=True).all_roots() ==\
    [
        S(1)/2 + I/2,
        -S(1)/2 - I/2
    ]


def test_CRootOf_eval_rational():
    p = legendre_poly(4, x, polys=True)
    roots = [r.eval_rational(n=18) for r in p.real_roots()]
    for root in roots:
        assert isinstance(root, Rational)
    roots = [str(root.n(17)) for root in roots]
    assert roots == [
            "-0.86113631159405258",
            "-0.33998104358485626",
             "0.33998104358485626",
             "0.86113631159405258",
             ]


def test_CRootOf_lazy():
    # irreducible poly with both real and complex roots:
    f = Poly(x**3 + 2*x + 2)

    # real root:
    CRootOf.clear_cache()
    r = CRootOf(f, 0)
    # Not yet in cache, after construction:
    assert r.poly not in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache
    r.evalf()
    # In cache after evaluation:
    assert r.poly in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache

    # complex root:
    CRootOf.clear_cache()
    r = CRootOf(f, 1)
    # Not yet in cache, after construction:
    assert r.poly not in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache
    r.evalf()
    # In cache after evaluation:
    assert r.poly in rootoftools._reals_cache
    assert r.poly in rootoftools._complexes_cache

    # composite poly with both real and complex roots:
    f = Poly((x**2 - 2)*(x**2 + 1))

    # real root:
    CRootOf.clear_cache()
    r = CRootOf(f, 0)
    # In cache immediately after construction:
    assert r.poly in rootoftools._reals_cache
    assert r.poly not in rootoftools._complexes_cache

    # complex root:
    CRootOf.clear_cache()
    r = CRootOf(f, 2)
    # In cache immediately after construction:
    assert r.poly in rootoftools._reals_cache
    assert r.poly in rootoftools._complexes_cache


def test_RootSum___new__():
    f = x**3 + x + 3

    g = Lambda(r, log(r*x))
    s = RootSum(f, g)

    assert isinstance(s, RootSum) is True

    assert RootSum(f**2, g) == 2*RootSum(f, g)
    assert RootSum((x - 7)*f**3, g) == log(7*x) + 3*RootSum(f, g)

    # issue 5571
    assert hash(RootSum((x - 7)*f**3, g)) == hash(log(7*x) + 3*RootSum(f, g))

    raises(MultivariatePolynomialError, lambda: RootSum(x**3 + x + y))
    raises(ValueError, lambda: RootSum(x**2 + 3, lambda x: x))

    assert RootSum(f, exp) == RootSum(f, Lambda(x, exp(x)))
    assert RootSum(f, log) == RootSum(f, Lambda(x, log(x)))

    assert isinstance(RootSum(f, auto=False), RootSum) is True

    assert RootSum(f) == 0
    assert RootSum(f, Lambda(x, x)) == 0
    assert RootSum(f, Lambda(x, x**2)) == -2

    assert RootSum(f, Lambda(x, 1)) == 3
    assert RootSum(f, Lambda(x, 2)) == 6

    assert RootSum(f, auto=False).is_commutative is True

    assert RootSum(f, Lambda(x, 1/(x + x**2))) == Rational(11, 3)
    assert RootSum(f, Lambda(x, y/(x + x**2))) == Rational(11, 3)*y

    assert RootSum(x**2 - 1, Lambda(x, 3*x**2), x) == 6
    assert RootSum(x**2 - y, Lambda(x, 3*x**2), x) == 6*y

    assert RootSum(x**2 - 1, Lambda(x, z*x**2), x) == 2*z
    assert RootSum(x**2 - y, Lambda(x, z*x**2), x) == 2*z*y

    assert RootSum(
        x**2 - 1, Lambda(x, exp(x)), quadratic=True) == exp(-1) + exp(1)

    assert RootSum(x**3 + a*x + a**3, tan, x) == \
        RootSum(x**3 + x + 1, Lambda(x, tan(a*x)))
    assert RootSum(a**3*x**3 + a*x + 1, tan, x) == \
        RootSum(x**3 + x + 1, Lambda(x, tan(x/a)))


def test_RootSum_free_symbols():
    assert RootSum(x**3 + x + 3, Lambda(r, exp(r))).free_symbols == set()
    assert RootSum(x**3 + x + 3, Lambda(r, exp(a*r))).free_symbols == {a}
    assert RootSum(
        x**3 + x + y, Lambda(r, exp(a*r)), x).free_symbols == {a, y}


def test_RootSum___eq__():
    f = Lambda(x, exp(x))

    assert (RootSum(x**3 + x + 1, f) == RootSum(x**3 + x + 1, f)) is True
    assert (RootSum(x**3 + x + 1, f) == RootSum(y**3 + y + 1, f)) is True

    assert (RootSum(x**3 + x + 1, f) == RootSum(x**3 + x + 2, f)) is False
    assert (RootSum(x**3 + x + 1, f) == RootSum(y**3 + y + 2, f)) is False


def test_RootSum_doit():
    rs = RootSum(x**2 + 1, exp)

    assert isinstance(rs, RootSum) is True
    assert rs.doit() == exp(-I) + exp(I)

    rs = RootSum(x**2 + a, exp, x)

    assert isinstance(rs, RootSum) is True
    assert rs.doit() == exp(-sqrt(-a)) + exp(sqrt(-a))


def test_RootSum_evalf():
    rs = RootSum(x**2 + 1, exp)

    assert rs.evalf(n=20, chop=True).epsilon_eq(Float("1.0806046117362794348"))
    assert rs.evalf(n=15, chop=True).epsilon_eq(Float("1.08060461173628"))

    rs = RootSum(x**2 + a, exp, x)

    assert rs.evalf() == rs


def test_RootSum_diff():
    f = x**3 + x + 3

    g = Lambda(r, exp(r*x))
    h = Lambda(r, r*exp(r*x))

    assert RootSum(f, g).diff(x) == RootSum(f, h)


def test_RootSum_subs():
    f = x**3 + x + 3
    g = Lambda(r, exp(r*x))

    F = y**3 + y + 3
    G = Lambda(r, exp(r*y))

    assert RootSum(f, g).subs(y, 1) == RootSum(f, g)
    assert RootSum(f, g).subs(x, y) == RootSum(F, G)


def test_RootSum_rational():
    assert RootSum(
        z**5 - z + 1, Lambda(z, z/(x - z))) == (4*x - 5)/(x**5 - x + 1)

    f = 161*z**3 + 115*z**2 + 19*z + 1
    g = Lambda(z, z*log(
        -3381*z**4/4 - 3381*z**3/4 - 625*z**2/2 - z*Rational(125, 2) - 5 + exp(x)))

    assert RootSum(f, g).diff(x) == -(
        (5*exp(2*x) - 6*exp(x) + 4)*exp(x)/(exp(3*x) - exp(2*x) + 1))/7


def test_RootSum_independent():
    f = (x**3 - a)**2*(x**4 - b)**3

    g = Lambda(x, 5*tan(x) + 7)
    h = Lambda(x, tan(x))

    r0 = RootSum(x**3 - a, h, x)
    r1 = RootSum(x**4 - b, h, x)

    assert RootSum(f, g, x).as_ordered_terms() == [10*r0, 15*r1, 126]


def test_issue_7876():
    l1 = Poly(x**6 - x + 1, x).all_roots()
    l2 = [rootof(x**6 - x + 1, i) for i in range(6)]
    assert frozenset(l1) == frozenset(l2)


def test_issue_8316():
    f = Poly(7*x**8 - 9)
    assert len(f.all_roots()) == 8
    f = Poly(7*x**8 - 10)
    assert len(f.all_roots()) == 8


def test__imag_count():
    from sympy.polys.rootoftools import _imag_count_of_factor
    def imag_count(p):
        return sum(_imag_count_of_factor(f)*m for f, m in
        p.factor_list()[1])
    assert imag_count(Poly(x**6 + 10*x**2 + 1)) == 2
    assert imag_count(Poly(x**2)) == 0
    assert imag_count(Poly([1]*3 + [-1], x)) == 0
    assert imag_count(Poly(x**3 + 1)) == 0
    assert imag_count(Poly(x**2 + 1)) == 2
    assert imag_count(Poly(x**2 - 1)) == 0
    assert imag_count(Poly(x**4 - 1)) == 2
    assert imag_count(Poly(x**4 + 1)) == 0
    assert imag_count(Poly([1, 2, 3], x)) == 0
    assert imag_count(Poly(x**3 + x + 1)) == 0
    assert imag_count(Poly(x**4 + x + 1)) == 0
    def q(r1, r2, p):
        return Poly(((x - r1)*(x - r2)).subs(x, x**p), x)
    assert imag_count(q(-1, -2, 2)) == 4
    assert imag_count(q(-1, 2, 2)) == 2
    assert imag_count(q(1, 2, 2)) == 0
    assert imag_count(q(1, 2, 4)) == 4
    assert imag_count(q(-1, 2, 4)) == 2
    assert imag_count(q(-1, -2, 4)) == 0


def test_RootOf_is_imaginary():
    r = RootOf(x**4 + 4*x**2 + 1, 1)
    i = r._get_interval()
    assert r.is_imaginary and i.ax*i.bx <= 0


def test_is_disjoint():
    eq = x**3 + 5*x + 1
    ir = rootof(eq, 0)._get_interval()
    ii = rootof(eq, 1)._get_interval()
    assert ir.is_disjoint(ii)
    assert ii.is_disjoint(ir)


def test_pure_key_dict():
    p = D()
    assert (x in p) is False
    assert (1 in p) is False
    p[x] = 1
    assert x in p
    assert y in p
    assert p[y] == 1
    raises(KeyError, lambda: p[1])
    def dont(k):
        p[k] = 2
    raises(ValueError, lambda: dont(1))


@slow
def test_eval_approx_relative():
    CRootOf.clear_cache()
    t = [CRootOf(x**3 + 10*x + 1, i) for i in range(3)]
    assert [i.eval_rational(1e-1) for i in t] == [
        Rational(-21, 220), Rational(15, 256) - I*805/256,
        Rational(15, 256) + I*805/256]
    t[0]._reset()
    assert [i.eval_rational(1e-1, 1e-4) for i in t] == [
        Rational(-21, 220), Rational(3275, 65536) - I*414645/131072,
        Rational(3275, 65536) + I*414645/131072]
    assert S(t[0]._get_interval().dx) < 1e-1
    assert S(t[1]._get_interval().dx) < 1e-1
    assert S(t[1]._get_interval().dy) < 1e-4
    assert S(t[2]._get_interval().dx) < 1e-1
    assert S(t[2]._get_interval().dy) < 1e-4
    t[0]._reset()
    assert [i.eval_rational(1e-4, 1e-4) for i in t] == [
        Rational(-2001, 20020), Rational(6545, 131072) - I*414645/131072,
        Rational(6545, 131072) + I*414645/131072]
    assert S(t[0]._get_interval().dx) < 1e-4
    assert S(t[1]._get_interval().dx) < 1e-4
    assert S(t[1]._get_interval().dy) < 1e-4
    assert S(t[2]._get_interval().dx) < 1e-4
    assert S(t[2]._get_interval().dy) < 1e-4
    # in the following, the actual relative precision is
    # less than tested, but it should never be greater
    t[0]._reset()
    assert [i.eval_rational(n=2) for i in t] == [
        Rational(-202201, 2024022), Rational(104755, 2097152) - I*6634255/2097152,
        Rational(104755, 2097152) + I*6634255/2097152]
    assert abs(S(t[0]._get_interval().dx)/t[0]) < 1e-2
    assert abs(S(t[1]._get_interval().dx)/t[1]).n() < 1e-2
    assert abs(S(t[1]._get_interval().dy)/t[1]).n() < 1e-2
    assert abs(S(t[2]._get_interval().dx)/t[2]).n() < 1e-2
    assert abs(S(t[2]._get_interval().dy)/t[2]).n() < 1e-2
    t[0]._reset()
    assert [i.eval_rational(n=3) for i in t] == [
        Rational(-202201, 2024022), Rational(1676045, 33554432) - I*106148135/33554432,
        Rational(1676045, 33554432) + I*106148135/33554432]
    assert abs(S(t[0]._get_interval().dx)/t[0]) < 1e-3
    assert abs(S(t[1]._get_interval().dx)/t[1]).n() < 1e-3
    assert abs(S(t[1]._get_interval().dy)/t[1]).n() < 1e-3
    assert abs(S(t[2]._get_interval().dx)/t[2]).n() < 1e-3
    assert abs(S(t[2]._get_interval().dy)/t[2]).n() < 1e-3

    t[0]._reset()
    a = [i.eval_approx(2) for i in t]
    assert [str(i) for i in a] == [
        '-0.10', '0.05 - 3.2*I', '0.05 + 3.2*I']
    assert all(abs(((a[i] - t[i])/t[i]).n()) < 1e-2 for i in range(len(a)))


def test_issue_15920():
    r = rootof(x**5 - x + 1, 0)
    p = Integral(x, (x, 1, y))
    assert unchanged(Eq, r, p)


def test_issue_19113():
    eq = y**3 - y + 1
    # generator is a canonical x in RootOf
    assert str(Poly(eq).real_roots()) == '[CRootOf(x**3 - x + 1, 0)]'
    assert str(Poly(eq.subs(y, tan(y))).real_roots()
        ) == '[CRootOf(x**3 - x + 1, 0)]'
    assert str(Poly(eq.subs(y, tan(x))).real_roots()
        ) == '[CRootOf(x**3 - x + 1, 0)]'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_groupby_dropna.py ===
import numpy as np
import pytest

from pandas.compat.pyarrow import pa_version_under10p1

from pandas.core.dtypes.missing import na_value_for_dtype

import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [
        (
            True,
            [["A", "B"], ["B", "A"]],
            {"c": [13.0, 123.23], "d": [13.0, 123.0], "e": [13.0, 1.0]},
        ),
        (
            False,
            [["A", "B"], ["A", np.nan], ["B", "A"]],
            {
                "c": [13.0, 12.3, 123.23],
                "d": [13.0, 233.0, 123.0],
                "e": [13.0, 12.0, 1.0],
            },
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_nan_in_one_group(
    dropna, tuples, outputs, nulls_fixture
):
    # GH 3729 this is to test that NA is in one group
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", nulls_fixture, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        ["A", "B", 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    grouped = df.groupby(["a", "b"], dropna=dropna).sum()

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna:
        mi = mi.set_levels(["A", "B", np.nan], level="b")
    expected = pd.DataFrame(outputs, index=mi)

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [
        (
            True,
            [["A", "B"], ["B", "A"]],
            {"c": [12.0, 123.23], "d": [12.0, 123.0], "e": [12.0, 1.0]},
        ),
        (
            False,
            [["A", "B"], ["A", np.nan], ["B", "A"], [np.nan, "B"]],
            {
                "c": [12.0, 13.3, 123.23, 1.0],
                "d": [12.0, 234.0, 123.0, 1.0],
                "e": [12.0, 13.0, 1.0, 1.0],
            },
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_nan_in_two_groups(
    dropna, tuples, outputs, nulls_fixture, nulls_fixture2
):
    # GH 3729 this is to test that NA in different groups with different representations
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", nulls_fixture, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        [nulls_fixture2, "B", 1, 1, 1.0],
        ["A", nulls_fixture2, 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    grouped = df.groupby(["a", "b"], dropna=dropna).sum()

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna:
        mi = mi.set_levels([["A", "B", np.nan], ["A", "B", np.nan]])
    expected = pd.DataFrame(outputs, index=mi)

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, idx, outputs",
    [
        (True, ["A", "B"], {"b": [123.23, 13.0], "c": [123.0, 13.0], "d": [1.0, 13.0]}),
        (
            False,
            ["A", "B", np.nan],
            {
                "b": [123.23, 13.0, 12.3],
                "c": [123.0, 13.0, 233.0],
                "d": [1.0, 13.0, 12.0],
            },
        ),
    ],
)
def test_groupby_dropna_normal_index_dataframe(dropna, idx, outputs):
    # GH 3729
    df_list = [
        ["B", 12, 12, 12],
        [None, 12.3, 233.0, 12],
        ["A", 123.23, 123, 1],
        ["B", 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d"])
    grouped = df.groupby("a", dropna=dropna).sum()

    expected = pd.DataFrame(outputs, index=pd.Index(idx, name="a"))

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, idx, expected",
    [
        (True, ["a", "a", "b", np.nan], pd.Series([3, 3], index=["a", "b"])),
        (
            False,
            ["a", "a", "b", np.nan],
            pd.Series([3, 3, 3], index=["a", "b", np.nan]),
        ),
    ],
)
def test_groupby_dropna_series_level(dropna, idx, expected):
    ser = pd.Series([1, 2, 3, 3], index=idx)

    result = ser.groupby(level=0, dropna=dropna).sum()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, expected",
    [
        (True, pd.Series([210.0, 350.0], index=["a", "b"], name="Max Speed")),
        (
            False,
            pd.Series([210.0, 350.0, 20.0], index=["a", "b", np.nan], name="Max Speed"),
        ),
    ],
)
def test_groupby_dropna_series_by(dropna, expected):
    ser = pd.Series(
        [390.0, 350.0, 30.0, 20.0],
        index=["Falcon", "Falcon", "Parrot", "Parrot"],
        name="Max Speed",
    )

    result = ser.groupby(["a", "b", "a", np.nan], dropna=dropna).mean()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dropna", (False, True))
def test_grouper_dropna_propagation(dropna):
    # GH 36604
    df = pd.DataFrame({"A": [0, 0, 1, None], "B": [1, 2, 3, None]})
    gb = df.groupby("A", dropna=dropna)
    assert gb._grouper.dropna == dropna


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 4),
        list("abcd"),
        pd.MultiIndex.from_product([(1, 2), ("R", "B")], names=["num", "col"]),
    ],
)
def test_groupby_dataframe_slice_then_transform(dropna, index):
    # GH35014 & GH35612
    expected_data = {"B": [2, 2, 1, np.nan if dropna else 1]}

    df = pd.DataFrame({"A": [0, 0, 1, None], "B": [1, 2, 3, None]}, index=index)
    gb = df.groupby("A", dropna=dropna)

    result = gb.transform(len)
    expected = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)

    result = gb[["B"]].transform(len)
    expected = pd.DataFrame(expected_data, index=index)
    tm.assert_frame_equal(result, expected)

    result = gb["B"].transform(len)
    expected = pd.Series(expected_data["B"], index=index, name="B")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [
        (
            True,
            [["A", "B"], ["B", "A"]],
            {"c": [13.0, 123.23], "d": [12.0, 123.0], "e": [1.0, 1.0]},
        ),
        (
            False,
            [["A", "B"], ["A", np.nan], ["B", "A"]],
            {
                "c": [13.0, 12.3, 123.23],
                "d": [12.0, 233.0, 123.0],
                "e": [1.0, 12.0, 1.0],
            },
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_agg(dropna, tuples, outputs):
    # GH 3729
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", None, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        ["A", "B", 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    agg_dict = {"c": "sum", "d": "max", "e": "min"}
    grouped = df.groupby(["a", "b"], dropna=dropna).agg(agg_dict)

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna:
        mi = mi.set_levels(["A", "B", np.nan], level="b")
    expected = pd.DataFrame(outputs, index=mi)

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.arm_slow
@pytest.mark.parametrize(
    "datetime1, datetime2",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
        (pd.Timedelta("-2 days"), pd.Timedelta("-1 days")),
        (pd.Period("2020-01-01"), pd.Period("2020-02-01")),
    ],
)
@pytest.mark.parametrize("dropna, values", [(True, [12, 3]), (False, [12, 3, 6])])
def test_groupby_dropna_datetime_like_data(
    dropna, values, datetime1, datetime2, unique_nulls_fixture, unique_nulls_fixture2
):
    # 3729
    df = pd.DataFrame(
        {
            "values": [1, 2, 3, 4, 5, 6],
            "dt": [
                datetime1,
                unique_nulls_fixture,
                datetime2,
                unique_nulls_fixture2,
                datetime1,
                datetime1,
            ],
        }
    )

    if dropna:
        indexes = [datetime1, datetime2]
    else:
        indexes = [datetime1, datetime2, np.nan]

    grouped = df.groupby("dt", dropna=dropna).agg({"values": "sum"})
    expected = pd.DataFrame({"values": values}, index=pd.Index(indexes, name="dt"))

    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, data, selected_data, levels",
    [
        pytest.param(
            False,
            {"groups": ["a", "a", "b", np.nan], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            ["a", "b", np.nan],
            id="dropna_false_has_nan",
        ),
        pytest.param(
            True,
            {"groups": ["a", "a", "b", np.nan], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0]},
            None,
            id="dropna_true_has_nan",
        ),
        pytest.param(
            # no nan in "groups"; dropna=True|False should be same.
            False,
            {"groups": ["a", "a", "b", "c"], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            None,
            id="dropna_false_no_nan",
        ),
        pytest.param(
            # no nan in "groups"; dropna=True|False should be same.
            True,
            {"groups": ["a", "a", "b", "c"], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            None,
            id="dropna_true_no_nan",
        ),
    ],
)
def test_groupby_apply_with_dropna_for_multi_index(dropna, data, selected_data, levels):
    # GH 35889

    df = pd.DataFrame(data)
    gb = df.groupby("groups", dropna=dropna)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = gb.apply(lambda grp: pd.DataFrame({"values": range(len(grp))}))

    mi_tuples = tuple(zip(data["groups"], selected_data["values"]))
    mi = pd.MultiIndex.from_tuples(mi_tuples, names=["groups", None])
    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    if not dropna and levels:
        mi = mi.set_levels(levels, level="groups")

    expected = pd.DataFrame(selected_data, index=mi)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("input_index", [None, ["a"], ["a", "b"]])
@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
@pytest.mark.parametrize("series", [True, False])
def test_groupby_dropna_with_multiindex_input(input_index, keys, series):
    # GH#46783
    obj = pd.DataFrame(
        {
            "a": [1, np.nan],
            "b": [1, 1],
            "c": [2, 3],
        }
    )

    expected = obj.set_index(keys)
    if series:
        expected = expected["c"]
    elif input_index == ["a", "b"] and keys == ["a"]:
        # Column b should not be aggregated
        expected = expected[["c"]]

    if input_index is not None:
        obj = obj.set_index(input_index)
    gb = obj.groupby(keys, dropna=False)
    if series:
        gb = gb["c"]
    result = gb.sum()

    tm.assert_equal(result, expected)


def test_groupby_nan_included():
    # GH 35646
    data = {"group": ["g1", np.nan, "g1", "g2", np.nan], "B": [0, 1, 2, 3, 4]}
    df = pd.DataFrame(data)
    grouped = df.groupby("group", dropna=False)
    result = grouped.indices
    dtype = np.intp
    expected = {
        "g1": np.array([0, 2], dtype=dtype),
        "g2": np.array([3], dtype=dtype),
        np.nan: np.array([1, 4], dtype=dtype),
    }
    for result_values, expected_values in zip(result.values(), expected.values()):
        tm.assert_numpy_array_equal(result_values, expected_values)
    assert np.isnan(list(result.keys())[2])
    assert list(result.keys())[0:2] == ["g1", "g2"]


def test_groupby_drop_nan_with_multi_index():
    # GH 39895
    df = pd.DataFrame([[np.nan, 0, 1]], columns=["a", "b", "c"])
    df = df.set_index(["a", "b"])
    result = df.groupby(["a", "b"], dropna=False).first()
    expected = df
    tm.assert_frame_equal(result, expected)


# sequence_index enumerates all strings made up of x, y, z of length 4
@pytest.mark.parametrize("sequence_index", range(3**4))
@pytest.mark.parametrize(
    "dtype",
    [
        None,
        "UInt8",
        "Int8",
        "UInt16",
        "Int16",
        "UInt32",
        "Int32",
        "UInt64",
        "Int64",
        "Float32",
        "Int64",
        "Float64",
        "category",
        "string",
        pytest.param(
            "string[pyarrow]",
            marks=pytest.mark.skipif(
                pa_version_under10p1, reason="pyarrow is not installed"
            ),
        ),
        "datetime64[ns]",
        "period[d]",
        "Sparse[float]",
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_no_sort_keep_na(sequence_index, dtype, test_series, as_index):
    # GH#46584, GH#48794

    # Convert sequence_index into a string sequence, e.g. 5 becomes "xxyz"
    # This sequence is used for the grouper.
    sequence = "".join(
        [{0: "x", 1: "y", 2: "z"}[sequence_index // (3**k) % 3] for k in range(4)]
    )

    # Unique values to use for grouper, depends on dtype
    if dtype in ("string", "string[pyarrow]"):
        uniques = {"x": "x", "y": "y", "z": pd.NA}
    elif dtype in ("datetime64[ns]", "period[d]"):
        uniques = {"x": "2016-01-01", "y": "2017-01-01", "z": pd.NA}
    else:
        uniques = {"x": 1, "y": 2, "z": np.nan}

    df = pd.DataFrame(
        {
            "key": pd.Series([uniques[label] for label in sequence], dtype=dtype),
            "a": [0, 1, 2, 3],
        }
    )
    gb = df.groupby("key", dropna=False, sort=False, as_index=as_index, observed=False)
    if test_series:
        gb = gb["a"]
    result = gb.sum()

    # Manually compute the groupby sum, use the labels "x", "y", and "z" to avoid
    # issues with hashing np.nan
    summed = {}
    for idx, label in enumerate(sequence):
        summed[label] = summed.get(label, 0) + idx
    if dtype == "category":
        index = pd.CategoricalIndex(
            [uniques[e] for e in summed],
            df["key"].cat.categories,
            name="key",
        )
    elif isinstance(dtype, str) and dtype.startswith("Sparse"):
        index = pd.Index(
            pd.array([uniques[label] for label in summed], dtype=dtype), name="key"
        )
    else:
        index = pd.Index([uniques[label] for label in summed], dtype=dtype, name="key")
    expected = pd.Series(summed.values(), index=index, name="a", dtype=None)
    if not test_series:
        expected = expected.to_frame()
    if not as_index:
        expected = expected.reset_index()
        if dtype is not None and dtype.startswith("Sparse"):
            expected["key"] = expected["key"].astype(dtype)

    tm.assert_equal(result, expected)


@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize("dtype", [object, None])
def test_null_is_null_for_dtype(
    sort, dtype, nulls_fixture, nulls_fixture2, test_series
):
    # GH#48506 - groups should always result in using the null for the dtype
    df = pd.DataFrame({"a": [1, 2]})
    groups = pd.Series([nulls_fixture, nulls_fixture2], dtype=dtype)
    obj = df["a"] if test_series else df
    gb = obj.groupby(groups, dropna=False, sort=sort)
    result = gb.sum()
    index = pd.Index([na_value_for_dtype(groups.dtype)])
    expected = pd.DataFrame({"a": [3]}, index=index)
    if test_series:
        tm.assert_series_equal(result, expected["a"])
    else:
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
def test_categorical_reducers(reduction_func, observed, sort, as_index, index_kind):
    # Ensure there is at least one null value by appending to the end
    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(20)}
    )

    # Strategy: Compare to dropna=True by filling null values with a new code
    df_filled = df.copy()
    df_filled["x"] = pd.Categorical(values, categories=[1, 2, 3, 4]).fillna(4)

    if index_kind == "range":
        keys = ["x"]
    elif index_kind == "single":
        keys = ["x"]
        df = df.set_index("x")
        df_filled = df_filled.set_index("x")
    else:
        keys = ["x", "x2"]
        df["x2"] = df["x"]
        df = df.set_index(["x", "x2"])
        df_filled["x2"] = df_filled["x"]
        df_filled = df_filled.set_index(["x", "x2"])
    args = get_groupby_method_args(reduction_func, df)
    args_filled = get_groupby_method_args(reduction_func, df_filled)
    if reduction_func == "corrwith" and index_kind == "range":
        # Don't include the grouping columns so we can call reset_index
        args = (args[0].drop(columns=keys),)
        args_filled = (args_filled[0].drop(columns=keys),)

    gb_keepna = df.groupby(
        keys, dropna=False, observed=observed, sort=sort, as_index=as_index
    )

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(gb_keepna, reduction_func)(*args)
        return

    gb_filled = df_filled.groupby(keys, observed=observed, sort=sort, as_index=True)
    expected = getattr(gb_filled, reduction_func)(*args_filled).reset_index()
    expected["x"] = expected["x"].cat.remove_categories([4])
    if index_kind == "multi":
        expected["x2"] = expected["x2"].cat.remove_categories([4])
    if as_index:
        if index_kind == "multi":
            expected = expected.set_index(["x", "x2"])
        else:
            expected = expected.set_index("x")
    elif index_kind != "range" and reduction_func != "size":
        # size, unlike other methods, has the desired behavior in GH#49519
        expected = expected.drop(columns="x")
        if index_kind == "multi":
            expected = expected.drop(columns="x2")
    if reduction_func in ("idxmax", "idxmin") and index_kind != "range":
        # expected was computed with a RangeIndex; need to translate to index values
        values = expected["y"].values.tolist()
        if index_kind == "single":
            values = [np.nan if e == 4 else e for e in values]
            expected["y"] = pd.Categorical(values, categories=[1, 2, 3])
        else:
            values = [(np.nan, np.nan) if e == (4, 4) else e for e in values]
            expected["y"] = values
    if reduction_func == "size":
        # size, unlike other methods, has the desired behavior in GH#49519
        expected = expected.rename(columns={0: "size"})
        if as_index:
            expected = expected["size"].rename(None)

    if as_index or index_kind == "range" or reduction_func == "size":
        warn = None
    else:
        warn = FutureWarning
    msg = "A grouping .* was excluded from the result"
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(gb_keepna, reduction_func)(*args)

    # size will return a Series, others are DataFrame
    tm.assert_equal(result, expected)


def test_categorical_transformers(
    request, transformation_func, observed, sort, as_index
):
    # GH#36327
    if transformation_func == "fillna":
        msg = "GH#49651 fillna may incorrectly reorders results when dropna=False"
        request.applymarker(pytest.mark.xfail(reason=msg, strict=False))

    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(20)}
    )
    args = get_groupby_method_args(transformation_func, df)

    # Compute result for null group
    null_group_values = df[df["x"].isnull()]["y"]
    if transformation_func == "cumcount":
        null_group_data = list(range(len(null_group_values)))
    elif transformation_func == "ngroup":
        if sort:
            if observed:
                na_group = df["x"].nunique(dropna=False) - 1
            else:
                # TODO: Should this be 3?
                na_group = df["x"].nunique(dropna=False) - 1
        else:
            na_group = df.iloc[: null_group_values.index[0]]["x"].nunique()
        null_group_data = len(null_group_values) * [na_group]
    else:
        null_group_data = getattr(null_group_values, transformation_func)(*args)
    null_group_result = pd.DataFrame({"y": null_group_data})

    gb_keepna = df.groupby(
        "x", dropna=False, observed=observed, sort=sort, as_index=as_index
    )
    gb_dropna = df.groupby("x", dropna=True, observed=observed, sort=sort)

    msg = "The default fill_method='ffill' in DataFrameGroupBy.pct_change is deprecated"
    if transformation_func == "pct_change":
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = getattr(gb_keepna, "pct_change")(*args)
    else:
        result = getattr(gb_keepna, transformation_func)(*args)
    expected = getattr(gb_dropna, transformation_func)(*args)

    for iloc, value in zip(
        df[df["x"].isnull()].index.tolist(), null_group_result.values.ravel()
    ):
        if expected.ndim == 1:
            expected.iloc[iloc] = value
        else:
            expected.iloc[iloc, 0] = value
    if transformation_func == "ngroup":
        expected[df["x"].notnull() & expected.ge(na_group)] += 1
    if transformation_func not in ("rank", "diff", "pct_change", "shift"):
        expected = expected.astype("int64")

    tm.assert_equal(result, expected)


@pytest.mark.parametrize("method", ["head", "tail"])
def test_categorical_head_tail(method, observed, sort, as_index):
    # GH#36327
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(len(values))}
    )
    gb = df.groupby("x", dropna=False, observed=observed, sort=sort, as_index=as_index)
    result = getattr(gb, method)()

    if method == "tail":
        values = values[::-1]
    # Take the top 5 values from each group
    mask = (
        ((values == 1) & ((values == 1).cumsum() <= 5))
        | ((values == 2) & ((values == 2).cumsum() <= 5))
        # flake8 doesn't like the vectorized check for None, thinks we should use `is`
        | ((values == None) & ((values == None).cumsum() <= 5))  # noqa: E711
    )
    if method == "tail":
        mask = mask[::-1]
    expected = df[mask]

    tm.assert_frame_equal(result, expected)


def test_categorical_agg():
    # GH#36327
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(len(values))}
    )
    gb = df.groupby("x", dropna=False, observed=False)
    result = gb.agg(lambda x: x.sum())
    expected = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_categorical_transform():
    # GH#36327
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(len(values))}
    )
    gb = df.groupby("x", dropna=False, observed=False)
    result = gb.transform(lambda x: x.sum())
    expected = gb.transform("sum")
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_constructors.py ===
import numpy as np
import pytest

from pandas._libs.tslibs.period import IncompatibleFrequency

from pandas.core.dtypes.dtypes import PeriodDtype

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
from pandas.core.arrays import PeriodArray


class TestPeriodIndexDisallowedFreqs:
    @pytest.mark.parametrize(
        "freq,freq_depr",
        [
            ("2M", "2ME"),
            ("2Q-MAR", "2QE-MAR"),
            ("2Y-FEB", "2YE-FEB"),
            ("2M", "2me"),
            ("2Q-MAR", "2qe-MAR"),
            ("2Y-FEB", "2yE-feb"),
        ],
    )
    def test_period_index_offsets_frequency_error_message(self, freq, freq_depr):
        # GH#52064
        msg = f"for Period, please use '{freq[1:]}' instead of '{freq_depr[1:]}'"

        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01-01", "2020-01-02"], freq=freq_depr)

        with pytest.raises(ValueError, match=msg):
            period_range(start="2020-01-01", end="2020-01-02", freq=freq_depr)

    @pytest.mark.parametrize("freq_depr", ["2SME", "2sme", "2CBME", "2BYE", "2Bye"])
    def test_period_index_frequency_invalid_freq(self, freq_depr):
        # GH#9586
        msg = f"Invalid frequency: {freq_depr[1:]}"

        with pytest.raises(ValueError, match=msg):
            period_range("2020-01", "2020-05", freq=freq_depr)
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2020-01", "2020-05"], freq=freq_depr)

    @pytest.mark.parametrize("freq", ["2BQE-SEP", "2BYE-MAR", "2BME"])
    def test_period_index_from_datetime_index_invalid_freq(self, freq):
        # GH#56899
        msg = f"Invalid frequency: {freq[1:]}"

        rng = date_range("01-Jan-2012", periods=8, freq=freq)
        with pytest.raises(ValueError, match=msg):
            rng.to_period()


class TestPeriodIndex:
    def test_from_ordinals(self):
        Period(ordinal=-1000, freq="Y")
        Period(ordinal=0, freq="Y")

        msg = "The 'ordinal' keyword in PeriodIndex is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx1 = PeriodIndex(ordinal=[-1, 0, 1], freq="Y")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx2 = PeriodIndex(ordinal=np.array([-1, 0, 1]), freq="Y")
        tm.assert_index_equal(idx1, idx2)

        alt1 = PeriodIndex.from_ordinals([-1, 0, 1], freq="Y")
        tm.assert_index_equal(alt1, idx1)

        alt2 = PeriodIndex.from_ordinals(np.array([-1, 0, 1]), freq="Y")
        tm.assert_index_equal(alt2, idx2)

    def test_keyword_mismatch(self):
        # GH#55961 we should get exactly one of data/ordinals/**fields
        per = Period("2016-01-01", "D")
        depr_msg1 = "The 'ordinal' keyword in PeriodIndex is deprecated"
        depr_msg2 = "Constructing PeriodIndex from fields is deprecated"

        err_msg1 = "Cannot pass both data and ordinal"
        with pytest.raises(ValueError, match=err_msg1):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg1):
                PeriodIndex(data=[per], ordinal=[per.ordinal], freq=per.freq)

        err_msg2 = "Cannot pass both data and fields"
        with pytest.raises(ValueError, match=err_msg2):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg2):
                PeriodIndex(data=[per], year=[per.year], freq=per.freq)

        err_msg3 = "Cannot pass both ordinal and fields"
        with pytest.raises(ValueError, match=err_msg3):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg2):
                PeriodIndex(ordinal=[per.ordinal], year=[per.year], freq=per.freq)

    def test_construction_base_constructor(self):
        # GH 13664
        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")]
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        arr = [np.nan, NaT, Period("2011-03", freq="M")]
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="D")]
        tm.assert_index_equal(Index(arr), Index(arr, dtype=object))

        tm.assert_index_equal(Index(np.array(arr)), Index(np.array(arr), dtype=object))

    def test_base_constructor_with_period_dtype(self):
        dtype = PeriodDtype("D")
        values = ["2011-01-01", "2012-03-04", "2014-05-01"]
        result = Index(values, dtype=dtype)

        expected = PeriodIndex(values, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "values_constructor", [list, np.array, PeriodIndex, PeriodArray._from_sequence]
    )
    def test_index_object_dtype(self, values_constructor):
        # Index(periods, dtype=object) is an Index (not an PeriodIndex)
        periods = [
            Period("2011-01", freq="M"),
            NaT,
            Period("2011-03", freq="M"),
        ]
        values = values_constructor(periods)
        result = Index(values, dtype=object)

        assert type(result) is Index
        tm.assert_numpy_array_equal(result.values, np.array(values))

    def test_constructor_use_start_freq(self):
        # GH #1118
        msg1 = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg1):
            p = Period("4/2/2012", freq="B")
        msg2 = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            expected = period_range(start="4/2/2012", periods=10, freq="B")

        with tm.assert_produces_warning(FutureWarning, match=msg2):
            index = period_range(start=p, periods=10)
        tm.assert_index_equal(index, expected)

    def test_constructor_field_arrays(self):
        # GH #1264

        years = np.arange(1990, 2010).repeat(4)[2:-2]
        quarters = np.tile(np.arange(1, 5), 20)[2:-2]

        depr_msg = "Constructing PeriodIndex from fields is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            index = PeriodIndex(year=years, quarter=quarters, freq="Q-DEC")
        expected = period_range("1990Q3", "2009Q2", freq="Q-DEC")
        tm.assert_index_equal(index, expected)

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            index2 = PeriodIndex(year=years, quarter=quarters, freq="2Q-DEC")
        tm.assert_numpy_array_equal(index.asi8, index2.asi8)

        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            index = PeriodIndex(year=years, quarter=quarters)
        tm.assert_index_equal(index, expected)

        years = [2007, 2007, 2007]
        months = [1, 2]

        msg = "Mismatched Period array lengths"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                PeriodIndex(year=years, month=months, freq="M")
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                PeriodIndex(year=years, month=months, freq="2M")

        years = [2007, 2007, 2007]
        months = [1, 2, 3]
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            idx = PeriodIndex(year=years, month=months, freq="M")
        exp = period_range("2007-01", periods=3, freq="M")
        tm.assert_index_equal(idx, exp)

    def test_constructor_nano(self):
        idx = period_range(
            start=Period(ordinal=1, freq="ns"),
            end=Period(ordinal=4, freq="ns"),
            freq="ns",
        )
        exp = PeriodIndex(
            [
                Period(ordinal=1, freq="ns"),
                Period(ordinal=2, freq="ns"),
                Period(ordinal=3, freq="ns"),
                Period(ordinal=4, freq="ns"),
            ],
            freq="ns",
        )
        tm.assert_index_equal(idx, exp)

    def test_constructor_arrays_negative_year(self):
        years = np.arange(1960, 2000, dtype=np.int64).repeat(4)
        quarters = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), 40)

        msg = "Constructing PeriodIndex from fields is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            pindex = PeriodIndex(year=years, quarter=quarters)

        tm.assert_index_equal(pindex.year, Index(years))
        tm.assert_index_equal(pindex.quarter, Index(quarters))

        alt = PeriodIndex.from_fields(year=years, quarter=quarters)
        tm.assert_index_equal(alt, pindex)

    def test_constructor_invalid_quarters(self):
        depr_msg = "Constructing PeriodIndex from fields is deprecated"
        msg = "Quarter must be 1 <= q <= 4"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                PeriodIndex(
                    year=range(2000, 2004), quarter=list(range(4)), freq="Q-DEC"
                )

    def test_period_range_fractional_period(self):
        msg = "Non-integer 'periods' in pd.date_range, pd.timedelta_range"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = period_range("2007-01", periods=10.5, freq="M")
        exp = period_range("2007-01", periods=10, freq="M")
        tm.assert_index_equal(result, exp)

    def test_constructor_with_without_freq(self):
        # GH53687
        start = Period("2002-01-01 00:00", freq="30min")
        exp = period_range(start=start, periods=5, freq=start.freq)
        result = period_range(start=start, periods=5)
        tm.assert_index_equal(exp, result)

    def test_constructor_fromarraylike(self):
        idx = period_range("2007-01", periods=20, freq="M")

        # values is an array of Period, thus can retrieve freq
        tm.assert_index_equal(PeriodIndex(idx.values), idx)
        tm.assert_index_equal(PeriodIndex(list(idx.values)), idx)

        msg = "freq not specified and cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(idx.asi8)
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(list(idx.asi8))

        msg = "'Period' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(data=Period("2007", freq="Y"))

        result = PeriodIndex(iter(idx))
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx)
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx, freq="M")
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx, freq=offsets.MonthEnd())
        tm.assert_index_equal(result, idx)
        assert result.freq == "ME"

        result = PeriodIndex(idx, freq="2M")
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2ME"

        result = PeriodIndex(idx, freq=offsets.MonthEnd(2))
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2ME"

        result = PeriodIndex(idx, freq="D")
        exp = idx.asfreq("D", "e")
        tm.assert_index_equal(result, exp)

    def test_constructor_datetime64arr(self):
        vals = np.arange(100000, 100000 + 10000, 100, dtype=np.int64)
        vals = vals.view(np.dtype("M8[us]"))

        pi = PeriodIndex(vals, freq="D")

        expected = PeriodIndex(vals.astype("M8[ns]"), freq="D")
        tm.assert_index_equal(pi, expected)

    @pytest.mark.parametrize("box", [None, "series", "index"])
    def test_constructor_datetime64arr_ok(self, box):
        # https://github.com/pandas-dev/pandas/issues/23438
        data = date_range("2017", periods=4, freq="ME")
        if box is None:
            data = data._values
        elif box == "series":
            data = Series(data)

        result = PeriodIndex(data, freq="D")
        expected = PeriodIndex(
            ["2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30"], freq="D"
        )
        tm.assert_index_equal(result, expected)

    def test_constructor_dtype(self):
        # passing a dtype with a tz should localize
        idx = PeriodIndex(["2013-01", "2013-03"], dtype="period[M]")
        exp = PeriodIndex(["2013-01", "2013-03"], freq="M")
        tm.assert_index_equal(idx, exp)
        assert idx.dtype == "period[M]"

        idx = PeriodIndex(["2013-01-05", "2013-03-05"], dtype="period[3D]")
        exp = PeriodIndex(["2013-01-05", "2013-03-05"], freq="3D")
        tm.assert_index_equal(idx, exp)
        assert idx.dtype == "period[3D]"

        # if we already have a freq and its not the same, then asfreq
        # (not changed)
        idx = PeriodIndex(["2013-01-01", "2013-01-02"], freq="D")

        res = PeriodIndex(idx, dtype="period[M]")
        exp = PeriodIndex(["2013-01", "2013-01"], freq="M")
        tm.assert_index_equal(res, exp)
        assert res.dtype == "period[M]"

        res = PeriodIndex(idx, freq="M")
        tm.assert_index_equal(res, exp)
        assert res.dtype == "period[M]"

        msg = "specified freq and dtype are different"
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(["2011-01"], freq="M", dtype="period[D]")

    def test_constructor_empty(self):
        idx = PeriodIndex([], freq="M")
        assert isinstance(idx, PeriodIndex)
        assert len(idx) == 0
        assert idx.freq == "ME"

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([])

    def test_constructor_pi_nat(self):
        idx = PeriodIndex(
            [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")]
        )
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            np.array([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")])
        )
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
        )
        exp = PeriodIndex(["NaT", "NaT", "2011-01", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            np.array(
                [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
            )
        )
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex([NaT, NaT, "2011-01", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([NaT, NaT])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array([NaT, NaT]))

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(["NaT", "NaT"])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array(["NaT", "NaT"]))

    def test_constructor_incompat_freq(self):
        msg = "Input has different freq=D from PeriodIndex\\(freq=M\\)"

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")])

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")]
                )
            )

        # first element is NaT
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")])

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")]
                )
            )

    def test_constructor_mixed(self):
        idx = PeriodIndex(["2011-01", NaT, Period("2011-01", freq="M")])
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(["NaT", NaT, Period("2011-01", freq="M")])
        exp = PeriodIndex(["NaT", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex([Period("2011-01-01", freq="D"), NaT, "2012-01-01"])
        exp = PeriodIndex(["2011-01-01", "NaT", "2012-01-01"], freq="D")
        tm.assert_index_equal(idx, exp)

    @pytest.mark.parametrize("floats", [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_constructor_floats(self, floats):
        msg = "PeriodIndex does not allow floating point in construction"
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(floats)

    def test_constructor_year_and_quarter(self):
        year = Series([2001, 2002, 2003])
        quarter = year - 2000
        msg = "Constructing PeriodIndex from fields is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            idx = PeriodIndex(year=year, quarter=quarter)
        strs = [f"{t[0]:d}Q{t[1]:d}" for t in zip(quarter, year)]
        lops = list(map(Period, strs))
        p = PeriodIndex(lops)
        tm.assert_index_equal(p, idx)

    def test_constructor_freq_mult(self):
        # GH #7811
        pidx = period_range(start="2014-01", freq="2M", periods=4)
        expected = PeriodIndex(["2014-01", "2014-03", "2014-05", "2014-07"], freq="2M")
        tm.assert_index_equal(pidx, expected)

        pidx = period_range(start="2014-01-02", end="2014-01-15", freq="3D")
        expected = PeriodIndex(
            ["2014-01-02", "2014-01-05", "2014-01-08", "2014-01-11", "2014-01-14"],
            freq="3D",
        )
        tm.assert_index_equal(pidx, expected)

        pidx = period_range(end="2014-01-01 17:00", freq="4h", periods=3)
        expected = PeriodIndex(
            ["2014-01-01 09:00", "2014-01-01 13:00", "2014-01-01 17:00"], freq="4h"
        )
        tm.assert_index_equal(pidx, expected)

        msg = "Frequency must be positive, because it represents span: -1M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="-1M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="0M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            period_range("2011-01", periods=3, freq="0M")

    @pytest.mark.parametrize(
        "freq_offset, freq_period",
        [
            ("YE", "Y"),
            ("ME", "M"),
            ("D", "D"),
            ("min", "min"),
            ("s", "s"),
        ],
    )
    @pytest.mark.parametrize("mult", [1, 2, 3, 4, 5])
    def test_constructor_freq_mult_dti_compat(self, mult, freq_offset, freq_period):
        freqstr_offset = str(mult) + freq_offset
        freqstr_period = str(mult) + freq_period
        pidx = period_range(start="2014-04-01", freq=freqstr_period, periods=10)
        expected = date_range(
            start="2014-04-01", freq=freqstr_offset, periods=10
        ).to_period(freqstr_period)
        tm.assert_index_equal(pidx, expected)

    @pytest.mark.parametrize("mult", [1, 2, 3, 4, 5])
    def test_constructor_freq_mult_dti_compat_month(self, mult):
        pidx = period_range(start="2014-04-01", freq=f"{mult}M", periods=10)
        expected = date_range(
            start="2014-04-01", freq=f"{mult}ME", periods=10
        ).to_period(f"{mult}M")
        tm.assert_index_equal(pidx, expected)

    def test_constructor_freq_combined(self):
        for freq in ["1D1h", "1h1D"]:
            pidx = PeriodIndex(["2016-01-01", "2016-01-02"], freq=freq)
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 00:00"], freq="25h")
        for freq in ["1D1h", "1h1D"]:
            pidx = period_range(start="2016-01-01", periods=2, freq=freq)
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 01:00"], freq="25h")
            tm.assert_index_equal(pidx, expected)

    def test_period_range_length(self):
        pi = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 9

        pi = period_range(freq="Q", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 4 * 9

        pi = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 12 * 9

        pi = period_range(freq="D", start="1/1/2001", end="12/31/2009")
        assert len(pi) == 365 * 9 + 2

        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            pi = period_range(freq="B", start="1/1/2001", end="12/31/2009")
        assert len(pi) == 261 * 9

        pi = period_range(freq="h", start="1/1/2001", end="12/31/2001 23:00")
        assert len(pi) == 365 * 24

        pi = period_range(freq="Min", start="1/1/2001", end="1/1/2001 23:59")
        assert len(pi) == 24 * 60

        pi = period_range(freq="s", start="1/1/2001", end="1/1/2001 23:59:59")
        assert len(pi) == 24 * 60 * 60

        with tm.assert_produces_warning(FutureWarning, match=msg):
            start = Period("02-Apr-2005", "B")
            i1 = period_range(start=start, periods=20)
        assert len(i1) == 20
        assert i1.freq == start.freq
        assert i1[0] == start

        end_intv = Period("2006-12-31", "W")
        i1 = period_range(end=end_intv, periods=10)
        assert len(i1) == 10
        assert i1.freq == end_intv.freq
        assert i1[-1] == end_intv

        msg = "'w' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            end_intv = Period("2006-12-31", "1w")
        i2 = period_range(end=end_intv, periods=10)
        assert len(i1) == len(i2)
        assert (i1 == i2).all()
        assert i1.freq == i2.freq

    def test_infer_freq_from_first_element(self):
        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            start = Period("02-Apr-2005", "B")
            end_intv = Period("2005-05-01", "B")
            period_range(start=start, end=end_intv)

            # infer freq from first element
            i2 = PeriodIndex([end_intv, Period("2005-05-05", "B")])
        assert len(i2) == 2
        assert i2[0] == end_intv

        with tm.assert_produces_warning(FutureWarning, match=msg):
            i2 = PeriodIndex(np.array([end_intv, Period("2005-05-05", "B")]))
        assert len(i2) == 2
        assert i2[0] == end_intv

    def test_mixed_freq_raises(self):
        # Mixed freq should fail
        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            end_intv = Period("2005-05-01", "B")

        msg = "'w' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            vals = [end_intv, Period("2006-12-31", "w")]
        msg = r"Input has different freq=W-SUN from PeriodIndex\(freq=B\)"
        depr_msg = r"PeriodDtype\[B\] is deprecated"
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                PeriodIndex(vals)
        vals = np.array(vals)
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=depr_msg):
                PeriodIndex(vals)

    @pytest.mark.parametrize(
        "freq", ["M", "Q", "Y", "D", "B", "min", "s", "ms", "us", "ns", "h"]
    )
    @pytest.mark.filterwarnings(
        r"ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_recreate_from_data(self, freq):
        org = period_range(start="2001/04/01", freq=freq, periods=1)
        idx = PeriodIndex(org.values, freq=freq)
        tm.assert_index_equal(idx, org)

    def test_map_with_string_constructor(self):
        raw = [2005, 2007, 2009]
        index = PeriodIndex(raw, freq="Y")

        expected = Index([str(num) for num in raw])
        res = index.map(str)

        # should return an Index
        assert isinstance(res, Index)

        # preserve element types
        assert all(isinstance(resi, str) for resi in res)

        # lastly, values should compare equal
        tm.assert_index_equal(res, expected)


class TestSimpleNew:
    def test_constructor_simple_new(self):
        idx = period_range("2007-01", name="p", periods=2, freq="M")

        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

        msg = "Should be numpy array of type i8"
        with pytest.raises(AssertionError, match=msg):
            # Need ndarray, not int64 Index
            type(idx._data)._simple_new(Index(idx.asi8), dtype=idx.dtype)

        arr = type(idx._data)._simple_new(idx.asi8, dtype=idx.dtype)
        result = idx._simple_new(arr, name="p")
        tm.assert_index_equal(result, idx)

    def test_constructor_simple_new_empty(self):
        # GH13079
        idx = PeriodIndex([], freq="M", name="p")
        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

    @pytest.mark.parametrize("floats", [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_period_index_simple_new_disallows_floats(self, floats):
        with pytest.raises(AssertionError, match="<class "):
            PeriodIndex._simple_new(floats)


class TestShallowCopy:
    def test_shallow_copy_empty(self):
        # GH#13067
        idx = PeriodIndex([], freq="M")
        result = idx._view()
        expected = idx

        tm.assert_index_equal(result, expected)

    def test_shallow_copy_disallow_i8(self):
        # GH#24391
        pi = period_range("2018-01-01", periods=3, freq="2D")
        with pytest.raises(AssertionError, match="ndarray"):
            pi._shallow_copy(pi.asi8)

    def test_shallow_copy_requires_disallow_period_index(self):
        pi = period_range("2018-01-01", periods=3, freq="2D")
        with pytest.raises(AssertionError, match="PeriodIndex"):
            pi._shallow_copy(pi)


class TestSeriesPeriod:
    def test_constructor_cant_cast_period(self):
        msg = "Cannot cast PeriodIndex to dtype float64"
        with pytest.raises(TypeError, match=msg):
            Series(period_range("2000-01-01", periods=10, freq="D"), dtype=float)

    def test_constructor_cast_object(self):
        pi = period_range("1/1/2000", periods=10)
        ser = Series(pi, dtype=PeriodDtype("D"))
        exp = Series(pi)
        tm.assert_series_equal(ser, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\tests\test_pathway.py ===
"""Tests for the ``sympy.physics.mechanics.pathway.py`` module."""

import pytest

from sympy import (
    Rational,
    Symbol,
    cos,
    pi,
    sin,
    sqrt,
)
from sympy.physics.mechanics import (
    Force,
    LinearPathway,
    ObstacleSetPathway,
    PathwayBase,
    Point,
    ReferenceFrame,
    WrappingCylinder,
    WrappingGeometryBase,
    WrappingPathway,
    WrappingSphere,
    dynamicsymbols,
)
from sympy.simplify.simplify import simplify


def _simplify_loads(loads):
    return [
        load.__class__(load.location, load.vector.simplify())
        for load in loads
    ]


class TestLinearPathway:

    def test_is_pathway_base_subclass(self):
        assert issubclass(LinearPathway, PathwayBase)

    @staticmethod
    @pytest.mark.parametrize(
        'args, kwargs',
        [
            ((Point('pA'), Point('pB')), {}),
        ]
    )
    def test_valid_constructor(args, kwargs):
        pointA, pointB = args
        instance = LinearPathway(*args, **kwargs)
        assert isinstance(instance, LinearPathway)
        assert hasattr(instance, 'attachments')
        assert len(instance.attachments) == 2
        assert instance.attachments[0] is pointA
        assert instance.attachments[1] is pointB
        assert isinstance(instance.attachments[0], Point)
        assert instance.attachments[0].name == 'pA'
        assert isinstance(instance.attachments[1], Point)
        assert instance.attachments[1].name == 'pB'

    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (Point('pA'), ),
            (Point('pA'), Point('pB'), Point('pZ')),
        ]
    )
    def test_invalid_attachments_incorrect_number(attachments):
        with pytest.raises(ValueError):
            _ = LinearPathway(*attachments)

    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (None, Point('pB')),
            (Point('pA'), None),
        ]
    )
    def test_invalid_attachments_not_point(attachments):
        with pytest.raises(TypeError):
            _ = LinearPathway(*attachments)

    @pytest.fixture(autouse=True)
    def _linear_pathway_fixture(self):
        self.N = ReferenceFrame('N')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q1 = dynamicsymbols('q1')
        self.q2 = dynamicsymbols('q2')
        self.q3 = dynamicsymbols('q3')
        self.q1d = dynamicsymbols('q1', 1)
        self.q2d = dynamicsymbols('q2', 1)
        self.q3d = dynamicsymbols('q3', 1)
        self.F = Symbol('F')

    def test_properties_are_immutable(self):
        instance = LinearPathway(self.pA, self.pB)
        with pytest.raises(AttributeError):
            instance.attachments = None
        with pytest.raises(TypeError):
            instance.attachments[0] = None
        with pytest.raises(TypeError):
            instance.attachments[1] = None

    def test_repr(self):
        pathway = LinearPathway(self.pA, self.pB)
        expected = 'LinearPathway(pA, pB)'
        assert repr(pathway) == expected

    def test_static_pathway_length(self):
        self.pB.set_pos(self.pA, 2*self.N.x)
        assert self.pathway.length == 2

    def test_static_pathway_extension_velocity(self):
        self.pB.set_pos(self.pA, 2*self.N.x)
        assert self.pathway.extension_velocity == 0

    def test_static_pathway_to_loads(self):
        self.pB.set_pos(self.pA, 2*self.N.x)
        expected = [
            (self.pA, - self.F*self.N.x),
            (self.pB, self.F*self.N.x),
        ]
        assert self.pathway.to_loads(self.F) == expected

    def test_2D_pathway_length(self):
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        expected = 2*sqrt(self.q1**2)
        assert self.pathway.length == expected

    def test_2D_pathway_extension_velocity(self):
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        expected = 2*sqrt(self.q1**2)*self.q1d/self.q1
        assert self.pathway.extension_velocity == expected

    def test_2D_pathway_to_loads(self):
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        expected = [
            (self.pA, - self.F*(self.q1 / sqrt(self.q1**2))*self.N.x),
            (self.pB, self.F*(self.q1 / sqrt(self.q1**2))*self.N.x),
        ]
        assert self.pathway.to_loads(self.F) == expected

    def test_3D_pathway_length(self):
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        expected = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        assert simplify(self.pathway.length - expected) == 0

    def test_3D_pathway_extension_velocity(self):
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        length = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        expected = (
            self.q1*self.q1d/length
            + self.q2*self.q2d/length
            + 4*self.q3*self.q3d/length
        )
        assert simplify(self.pathway.extension_velocity - expected) == 0

    def test_3D_pathway_to_loads(self):
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        length = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        pO_force = (
            - self.F*self.q1*self.N.x/length
            + self.F*self.q2*self.N.y/length
            - 2*self.F*self.q3*self.N.z/length
        )
        pI_force = (
            self.F*self.q1*self.N.x/length
            - self.F*self.q2*self.N.y/length
            + 2*self.F*self.q3*self.N.z/length
        )
        expected = [
            (self.pA, pO_force),
            (self.pB, pI_force),
        ]
        assert self.pathway.to_loads(self.F) == expected


class TestObstacleSetPathway:

    def test_is_pathway_base_subclass(self):
        assert issubclass(ObstacleSetPathway, PathwayBase)

    @staticmethod
    @pytest.mark.parametrize(
        'num_attachments, attachments',
        [
            (3, [Point(name) for name in ('pO', 'pA', 'pI')]),
            (4, [Point(name) for name in ('pO', 'pA', 'pB', 'pI')]),
            (5, [Point(name) for name in ('pO', 'pA', 'pB', 'pC', 'pI')]),
            (6, [Point(name) for name in ('pO', 'pA', 'pB', 'pC', 'pD', 'pI')]),
        ]
    )
    def test_valid_constructor(num_attachments, attachments):
        instance = ObstacleSetPathway(*attachments)
        assert isinstance(instance, ObstacleSetPathway)
        assert hasattr(instance, 'attachments')
        assert len(instance.attachments) == num_attachments
        for attachment in instance.attachments:
            assert isinstance(attachment, Point)

    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [[Point('pO')], [Point('pO'), Point('pI')]],
    )
    def test_invalid_constructor_attachments_incorrect_number(attachments):
        with pytest.raises(ValueError):
            _ = ObstacleSetPathway(*attachments)

    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (None, Point('pA'), Point('pI')),
            (Point('pO'), None, Point('pI')),
            (Point('pO'), Point('pA'), None),
        ]
    )
    def test_invalid_constructor_attachments_not_point(attachments):
        with pytest.raises(TypeError):
            _ = WrappingPathway(*attachments)  # type: ignore

    def test_properties_are_immutable(self):
        pathway = ObstacleSetPathway(Point('pO'), Point('pA'), Point('pI'))
        with pytest.raises(AttributeError):
            pathway.attachments = None  # type: ignore
        with pytest.raises(TypeError):
            pathway.attachments[0] = None  # type: ignore
        with pytest.raises(TypeError):
            pathway.attachments[1] = None  # type: ignore
        with pytest.raises(TypeError):
            pathway.attachments[-1] = None  # type: ignore

    @staticmethod
    @pytest.mark.parametrize(
        'attachments, expected',
        [
            (
                [Point(name) for name in ('pO', 'pA', 'pI')],
                'ObstacleSetPathway(pO, pA, pI)'
            ),
            (
                [Point(name) for name in ('pO', 'pA', 'pB', 'pI')],
                'ObstacleSetPathway(pO, pA, pB, pI)'
            ),
            (
                [Point(name) for name in ('pO', 'pA', 'pB', 'pC', 'pI')],
                'ObstacleSetPathway(pO, pA, pB, pC, pI)'
            ),
        ]
    )
    def test_repr(attachments, expected):
        pathway = ObstacleSetPathway(*attachments)
        assert repr(pathway) == expected

    @pytest.fixture(autouse=True)
    def _obstacle_set_pathway_fixture(self):
        self.N = ReferenceFrame('N')
        self.pO = Point('pO')
        self.pI = Point('pI')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.q = dynamicsymbols('q')
        self.qd = dynamicsymbols('q', 1)
        self.F = Symbol('F')

    def test_static_pathway_length(self):
        self.pA.set_pos(self.pO, self.N.x)
        self.pB.set_pos(self.pO, self.N.y)
        self.pI.set_pos(self.pO, self.N.z)
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        assert pathway.length == 1 + 2 * sqrt(2)

    def test_static_pathway_extension_velocity(self):
        self.pA.set_pos(self.pO, self.N.x)
        self.pB.set_pos(self.pO, self.N.y)
        self.pI.set_pos(self.pO, self.N.z)
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        assert pathway.extension_velocity == 0

    def test_static_pathway_to_loads(self):
        self.pA.set_pos(self.pO, self.N.x)
        self.pB.set_pos(self.pO, self.N.y)
        self.pI.set_pos(self.pO, self.N.z)
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        expected = [
            Force(self.pO, -self.F * self.N.x),
            Force(self.pA, self.F * self.N.x),
            Force(self.pA, self.F * sqrt(2) / 2 * (self.N.x - self.N.y)),
            Force(self.pB, self.F * sqrt(2) / 2 * (self.N.y - self.N.x)),
            Force(self.pB, self.F * sqrt(2) / 2 * (self.N.y - self.N.z)),
            Force(self.pI, self.F * sqrt(2) / 2 * (self.N.z - self.N.y)),
        ]
        assert pathway.to_loads(self.F) == expected

    def test_2D_pathway_length(self):
        self.pA.set_pos(self.pO, -(self.N.x + self.N.y))
        self.pB.set_pos(
            self.pO, cos(self.q) * self.N.x - (sin(self.q) + 1) * self.N.y
        )
        self.pI.set_pos(
            self.pO, sin(self.q) * self.N.x + (cos(self.q) - 1) * self.N.y
        )
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        expected = 2 * sqrt(2) + sqrt(2 + 2*cos(self.q))
        assert (pathway.length - expected).simplify() == 0

    def test_2D_pathway_extension_velocity(self):
        self.pA.set_pos(self.pO, -(self.N.x + self.N.y))
        self.pB.set_pos(
            self.pO, cos(self.q) * self.N.x - (sin(self.q) + 1) * self.N.y
        )
        self.pI.set_pos(
            self.pO, sin(self.q) * self.N.x + (cos(self.q) - 1) * self.N.y
        )
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        expected = - (sqrt(2) * sin(self.q) * self.qd) / (2 * sqrt(cos(self.q) + 1))
        assert (pathway.extension_velocity - expected).simplify() == 0

    def test_2D_pathway_to_loads(self):
        self.pA.set_pos(self.pO, -(self.N.x + self.N.y))
        self.pB.set_pos(
            self.pO, cos(self.q) * self.N.x - (sin(self.q) + 1) * self.N.y
        )
        self.pI.set_pos(
            self.pO, sin(self.q) * self.N.x + (cos(self.q) - 1) * self.N.y
        )
        pathway = ObstacleSetPathway(self.pO, self.pA, self.pB, self.pI)
        pO_pA_force_vec = sqrt(2) / 2 * (self.N.x + self.N.y)
        pA_pB_force_vec = (
            - sqrt(2 * cos(self.q) + 2) / 2 * self.N.x
            + sqrt(2) * sin(self.q) / (2 * sqrt(cos(self.q) + 1)) * self.N.y
        )
        pB_pI_force_vec = cos(self.q + pi/4) * self.N.x - sin(self.q + pi/4) * self.N.y
        expected = [
            Force(self.pO, self.F * pO_pA_force_vec),
            Force(self.pA, -self.F * pO_pA_force_vec),
            Force(self.pA, self.F * pA_pB_force_vec),
            Force(self.pB, -self.F * pA_pB_force_vec),
            Force(self.pB, self.F * pB_pI_force_vec),
            Force(self.pI, -self.F * pB_pI_force_vec),
        ]
        assert _simplify_loads(pathway.to_loads(self.F)) == expected


class TestWrappingPathway:

    def test_is_pathway_base_subclass(self):
        assert issubclass(WrappingPathway, PathwayBase)

    @pytest.fixture(autouse=True)
    def _wrapping_pathway_fixture(self):
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.r = Symbol('r', positive=True)
        self.pO = Point('pO')
        self.N = ReferenceFrame('N')
        self.ax = self.N.z
        self.sphere = WrappingSphere(self.r, self.pO)
        self.cylinder = WrappingCylinder(self.r, self.pO, self.ax)
        self.pathway = WrappingPathway(self.pA, self.pB, self.cylinder)
        self.F = Symbol('F')

    def test_valid_constructor(self):
        instance = WrappingPathway(self.pA, self.pB, self.cylinder)
        assert isinstance(instance, WrappingPathway)
        assert hasattr(instance, 'attachments')
        assert len(instance.attachments) == 2
        assert isinstance(instance.attachments[0], Point)
        assert instance.attachments[0] == self.pA
        assert isinstance(instance.attachments[1], Point)
        assert instance.attachments[1] == self.pB
        assert hasattr(instance, 'geometry')
        assert isinstance(instance.geometry, WrappingGeometryBase)
        assert instance.geometry == self.cylinder

    @pytest.mark.parametrize(
        'attachments',
        [
            (Point('pA'), ),
            (Point('pA'), Point('pB'), Point('pZ')),
        ]
    )
    def test_invalid_constructor_attachments_incorrect_number(self, attachments):
        with pytest.raises(TypeError):
            _ = WrappingPathway(*attachments, self.cylinder)

    @staticmethod
    @pytest.mark.parametrize(
        'attachments',
        [
            (None, Point('pB')),
            (Point('pA'), None),
        ]
    )
    def test_invalid_constructor_attachments_not_point(attachments):
        with pytest.raises(TypeError):
            _ = WrappingPathway(*attachments)

    def test_invalid_constructor_geometry_is_not_supplied(self):
        with pytest.raises(TypeError):
            _ = WrappingPathway(self.pA, self.pB)

    @pytest.mark.parametrize(
        'geometry',
        [
            Symbol('r'),
            dynamicsymbols('q'),
            ReferenceFrame('N'),
            ReferenceFrame('N').x,
        ]
    )
    def test_invalid_geometry_not_geometry(self, geometry):
        with pytest.raises(TypeError):
            _ = WrappingPathway(self.pA, self.pB, geometry)

    def test_attachments_property_is_immutable(self):
        with pytest.raises(TypeError):
            self.pathway.attachments[0] = self.pB
        with pytest.raises(TypeError):
            self.pathway.attachments[1] = self.pA

    def test_geometry_property_is_immutable(self):
        with pytest.raises(AttributeError):
            self.pathway.geometry = None

    def test_repr(self):
        expected = (
            f'WrappingPathway(pA, pB, '
            f'geometry={self.cylinder!r})'
        )
        assert repr(self.pathway) == expected

    @staticmethod
    def _expand_pos_to_vec(pos, frame):
        return sum(mag*unit for (mag, unit) in zip(pos, frame))

    @pytest.mark.parametrize(
        'pA_vec, pB_vec, factor',
        [
            ((1, 0, 0), (0, 1, 0), pi/2),
            ((0, 1, 0), (sqrt(2)/2, -sqrt(2)/2, 0), 3*pi/4),
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 0), pi/3),
        ]
    )
    def test_static_pathway_on_sphere_length(self, pA_vec, pB_vec, factor):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.sphere)
        expected = factor*self.r
        assert simplify(pathway.length - expected) == 0

    @pytest.mark.parametrize(
        'pA_vec, pB_vec, factor',
        [
            ((1, 0, 0), (0, 1, 0), Rational(1, 2)*pi),
            ((1, 0, 0), (-1, 0, 0), pi),
            ((-1, 0, 0), (1, 0, 0), pi),
            ((0, 1, 0), (sqrt(2)/2, -sqrt(2)/2, 0), 5*pi/4),
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 0), pi/3),
            (
                (0, 1, 0),
                (sqrt(2)*Rational(1, 2), -sqrt(2)*Rational(1, 2), 1),
                sqrt(1 + (Rational(5, 4)*pi)**2),
            ),
            (
                (1, 0, 0),
                (Rational(1, 2), sqrt(3)*Rational(1, 2), 1),
                sqrt(1 + (Rational(1, 3)*pi)**2),
            ),
        ]
    )
    def test_static_pathway_on_cylinder_length(self, pA_vec, pB_vec, factor):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.cylinder)
        expected = factor*sqrt(self.r**2)
        assert simplify(pathway.length - expected) == 0

    @pytest.mark.parametrize(
        'pA_vec, pB_vec',
        [
            ((1, 0, 0), (0, 1, 0)),
            ((0, 1, 0), (sqrt(2)*Rational(1, 2), -sqrt(2)*Rational(1, 2), 0)),
            ((1, 0, 0), (Rational(1, 2), sqrt(3)*Rational(1, 2), 0)),
        ]
    )
    def test_static_pathway_on_sphere_extension_velocity(self, pA_vec, pB_vec):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.sphere)
        assert pathway.extension_velocity == 0

    @pytest.mark.parametrize(
        'pA_vec, pB_vec',
        [
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (-1, 0, 0)),
            ((-1, 0, 0), (1, 0, 0)),
            ((0, 1, 0), (sqrt(2)/2, -sqrt(2)/2, 0)),
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 0)),
            ((0, 1, 0), (sqrt(2)*Rational(1, 2), -sqrt(2)/2, 1)),
            ((1, 0, 0), (Rational(1, 2), sqrt(3)/2, 1)),
        ]
    )
    def test_static_pathway_on_cylinder_extension_velocity(self, pA_vec, pB_vec):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.cylinder)
        assert pathway.extension_velocity == 0

    @pytest.mark.parametrize(
        'pA_vec, pB_vec, pA_vec_expected, pB_vec_expected, pO_vec_expected',
        (
            ((1, 0, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (-1, -1, 0)),
            (
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 0),
                (1, 0, 0),
                (sqrt(2)/2, sqrt(2)/2, 0),
                (-1 - sqrt(2)/2, -sqrt(2)/2, 0)
            ),
            (
                (1, 0, 0),
                (Rational(1, 2), sqrt(3)/2, 0),
                (0, 1, 0),
                (sqrt(3)/2, -Rational(1, 2), 0),
                (-sqrt(3)/2, Rational(1, 2) - 1, 0),
            ),
        )
    )
    def test_static_pathway_on_sphere_to_loads(
        self,
        pA_vec,
        pB_vec,
        pA_vec_expected,
        pB_vec_expected,
        pO_vec_expected,
    ):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.sphere)

        pA_vec_expected = sum(
            mag*unit for (mag, unit) in zip(pA_vec_expected, self.N)
        )
        pB_vec_expected = sum(
            mag*unit for (mag, unit) in zip(pB_vec_expected, self.N)
        )
        pO_vec_expected = sum(
            mag*unit for (mag, unit) in zip(pO_vec_expected, self.N)
        )
        expected = [
            Force(self.pA, self.F*(self.r**3/sqrt(self.r**6))*pA_vec_expected),
            Force(self.pB, self.F*(self.r**3/sqrt(self.r**6))*pB_vec_expected),
            Force(self.pO, self.F*(self.r**3/sqrt(self.r**6))*pO_vec_expected),
        ]
        assert pathway.to_loads(self.F) == expected

    @pytest.mark.parametrize(
        'pA_vec, pB_vec, pA_vec_expected, pB_vec_expected, pO_vec_expected',
        (
            ((1, 0, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (-1, -1, 0)),
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 1, 0), (0, -2, 0)),
            ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -1, 0), (0, 2, 0)),
            (
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 0),
                (-1, 0, 0),
                (-sqrt(2)/2, -sqrt(2)/2, 0),
                (1 + sqrt(2)/2, sqrt(2)/2, 0)
            ),
            (
                (1, 0, 0),
                (Rational(1, 2), sqrt(3)/2, 0),
                (0, 1, 0),
                (sqrt(3)/2, -Rational(1, 2), 0),
                (-sqrt(3)/2, Rational(1, 2) - 1, 0),
            ),
            (
                (1, 0, 0),
                (sqrt(2)/2, sqrt(2)/2, 0),
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 0),
                (-sqrt(2)/2, sqrt(2)/2 - 1, 0),
            ),
            ((0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, -1), (0, 0, 0)),
            (
                (0, 1, 0),
                (sqrt(2)/2, -sqrt(2)/2, 1),
                (-5*pi/sqrt(16 + 25*pi**2), 0, 4/sqrt(16 + 25*pi**2)),
                (
                    -5*sqrt(2)*pi/(2*sqrt(16 + 25*pi**2)),
                    -5*sqrt(2)*pi/(2*sqrt(16 + 25*pi**2)),
                    -4/sqrt(16 + 25*pi**2),
                ),
                (
                    5*(sqrt(2) + 2)*pi/(2*sqrt(16 + 25*pi**2)),
                    5*sqrt(2)*pi/(2*sqrt(16 + 25*pi**2)),
                    0,
                ),
            ),
        )
    )
    def test_static_pathway_on_cylinder_to_loads(
        self,
        pA_vec,
        pB_vec,
        pA_vec_expected,
        pB_vec_expected,
        pO_vec_expected,
    ):
        pA_vec = self._expand_pos_to_vec(pA_vec, self.N)
        pB_vec = self._expand_pos_to_vec(pB_vec, self.N)
        self.pA.set_pos(self.pO, self.r*pA_vec)
        self.pB.set_pos(self.pO, self.r*pB_vec)
        pathway = WrappingPathway(self.pA, self.pB, self.cylinder)

        pA_force_expected = self.F*self._expand_pos_to_vec(pA_vec_expected,
                                                           self.N)
        pB_force_expected = self.F*self._expand_pos_to_vec(pB_vec_expected,
                                                           self.N)
        pO_force_expected = self.F*self._expand_pos_to_vec(pO_vec_expected,
                                                           self.N)
        expected = [
            Force(self.pA, pA_force_expected),
            Force(self.pB, pB_force_expected),
            Force(self.pO, pO_force_expected),
        ]
        assert _simplify_loads(pathway.to_loads(self.F)) == expected

    def test_2D_pathway_on_cylinder_length(self):
        q = dynamicsymbols('q')
        pA_pos = self.r*self.N.x
        pB_pos = self.r*(cos(q)*self.N.x + sin(q)*self.N.y)
        self.pA.set_pos(self.pO, pA_pos)
        self.pB.set_pos(self.pO, pB_pos)
        expected = self.r*sqrt(q**2)
        assert simplify(self.pathway.length - expected) == 0

    def test_2D_pathway_on_cylinder_extension_velocity(self):
        q = dynamicsymbols('q')
        qd = dynamicsymbols('q', 1)
        pA_pos = self.r*self.N.x
        pB_pos = self.r*(cos(q)*self.N.x + sin(q)*self.N.y)
        self.pA.set_pos(self.pO, pA_pos)
        self.pB.set_pos(self.pO, pB_pos)
        expected = self.r*(sqrt(q**2)/q)*qd
        assert simplify(self.pathway.extension_velocity - expected) == 0

    def test_2D_pathway_on_cylinder_to_loads(self):
        q = dynamicsymbols('q')
        pA_pos = self.r*self.N.x
        pB_pos = self.r*(cos(q)*self.N.x + sin(q)*self.N.y)
        self.pA.set_pos(self.pO, pA_pos)
        self.pB.set_pos(self.pO, pB_pos)

        pA_force = self.F*self.N.y
        pB_force = self.F*(sin(q)*self.N.x - cos(q)*self.N.y)
        pO_force = self.F*(-sin(q)*self.N.x + (cos(q) - 1)*self.N.y)
        expected = [
            Force(self.pA, pA_force),
            Force(self.pB, pB_force),
            Force(self.pO, pO_force),
        ]

        loads = _simplify_loads(self.pathway.to_loads(self.F))
        assert loads == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_floats.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    RangeIndex,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


def gen_obj(klass, index):
    if klass is Series:
        obj = Series(np.arange(len(index)), index=index)
    else:
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), len(index))),
            index=index,
            columns=index,
        )
    return obj


class TestFloatIndexers:
    def check(self, result, original, indexer, getitem):
        """
        comparator for results
        we need to take care if we are indexing on a
        Series or a frame
        """
        if isinstance(original, Series):
            expected = original.iloc[indexer]
        elif getitem:
            expected = original.iloc[:, indexer]
        else:
            expected = original.iloc[indexer]

        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde")),
            Index(list("abcde"), dtype="category"),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_scalar_non_numeric(self, index, frame_or_series, indexer_sl):
        # GH 4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors

        s = gen_obj(frame_or_series, index)

        # getting
        with pytest.raises(KeyError, match="^3.0$"):
            indexer_sl(s)[3.0]

        # contains
        assert 3.0 not in s

        s2 = s.copy()
        indexer_sl(s2)[3.0] = 10

        if indexer_sl is tm.setitem:
            assert 3.0 in s2.axes[-1]
        elif indexer_sl is tm.loc:
            assert 3.0 in s2.axes[0]
        else:
            assert 3.0 not in s2.axes[0]
            assert 3.0 not in s2.axes[-1]

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde")),
            Index(list("abcde"), dtype="category"),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_scalar_non_numeric_series_fallback(self, index):
        # fallsback to position selection, series only
        s = Series(np.arange(len(index)), index=index)

        msg = "Series.__getitem__ treating keys as positions is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            s[3]
        with pytest.raises(KeyError, match="^3.0$"):
            s[3.0]

    def test_scalar_with_mixed(self, indexer_sl):
        s2 = Series([1, 2, 3], index=["a", "b", "c"])
        s3 = Series([1, 2, 3], index=["a", "b", 1.5])

        # lookup in a pure string index with an invalid indexer

        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s2)[1.0]

        with pytest.raises(KeyError, match=r"^1\.0$"):
            indexer_sl(s2)[1.0]

        result = indexer_sl(s2)["b"]
        expected = 2
        assert result == expected

        # mixed index so we have label
        # indexing
        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s3)[1.0]

        if indexer_sl is not tm.loc:
            # __getitem__ falls back to positional
            msg = "Series.__getitem__ treating keys as positions is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                result = s3[1]
            expected = 2
            assert result == expected

        with pytest.raises(KeyError, match=r"^1\.0$"):
            indexer_sl(s3)[1.0]

        result = indexer_sl(s3)[1.5]
        expected = 3
        assert result == expected

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer(self, index, frame_or_series, indexer_sl):
        getitem = indexer_sl is not tm.loc

        # test how scalar float indexers work on int indexes

        # integer index
        i = index
        obj = gen_obj(frame_or_series, i)

        # coerce to equal int

        result = indexer_sl(obj)[3.0]
        self.check(result, obj, 3, getitem)

        if isinstance(obj, Series):

            def compare(x, y):
                assert x == y

            expected = 100
        else:
            compare = tm.assert_series_equal
            if getitem:
                expected = Series(100, index=range(len(obj)), name=3)
            else:
                expected = Series(100.0, index=range(len(obj)), name=3)

        s2 = obj.copy()
        indexer_sl(s2)[3.0] = 100

        result = indexer_sl(s2)[3.0]
        compare(result, expected)

        result = indexer_sl(s2)[3]
        compare(result, expected)

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer_contains_float(self, index, frame_or_series):
        # contains
        # integer index
        obj = gen_obj(frame_or_series, index)

        # coerce to equal int
        assert 3.0 in obj

    def test_scalar_float(self, frame_or_series):
        # scalar float indexers work on a float index
        index = Index(np.arange(5.0))
        s = gen_obj(frame_or_series, index)

        # assert all operations except for iloc are ok
        indexer = index[3]
        for idxr in [tm.loc, tm.setitem]:
            getitem = idxr is not tm.loc

            # getting
            result = idxr(s)[indexer]
            self.check(result, s, 3, getitem)

            # setting
            s2 = s.copy()

            result = idxr(s2)[indexer]
            self.check(result, s, 3, getitem)

            # random float is a KeyError
            with pytest.raises(KeyError, match=r"^3\.5$"):
                idxr(s)[3.5]

        # contains
        assert 3.0 in s

        # iloc succeeds with an integer
        expected = s.iloc[3]
        s2 = s.copy()

        s2.iloc[3] = expected
        result = s2.iloc[3]
        self.check(result, s, 3, False)

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde"), dtype=object),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_slice_non_numeric(self, index, idx, frame_or_series, indexer_sli):
        # GH 4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors

        s = gen_obj(frame_or_series, index)

        # getitem
        if indexer_sli is tm.iloc:
            msg = (
                "cannot do positional indexing "
                rf"on {type(index).__name__} with these indexers \[(3|4)\.0\] of "
                "type float"
            )
        else:
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers "
                r"\[(3|4)(\.0)?\] "
                r"of type (float|int)"
            )
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx]

        # setitem
        if indexer_sli is tm.iloc:
            # otherwise we keep the same message as above
            msg = "slice indices must be integers or None or have an __index__ method"
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx] = 0

    def test_slice_integer(self):
        # same as above, but for Integer based indexes
        # these coerce to a like integer
        # oob indicates if we are out of bounds
        # of positional indexing
        for index, oob in [
            (Index(np.arange(5, dtype=np.int64)), False),
            (RangeIndex(5), False),
            (Index(np.arange(5, dtype=np.int64) + 10), True),
        ]:
            # s is an in-range index
            s = Series(range(5), index=index)

            # getitem
            for idx in [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)]:
                result = s.loc[idx]

                # these are all label indexing
                # except getitem which is positional
                # empty
                if oob:
                    indexer = slice(0, 0)
                else:
                    indexer = slice(3, 5)
                self.check(result, s, indexer, False)

            # getitem out-of-bounds
            for idx in [slice(-6, 6), slice(-6.0, 6.0)]:
                result = s.loc[idx]

                # these are all label indexing
                # except getitem which is positional
                # empty
                if oob:
                    indexer = slice(0, 0)
                else:
                    indexer = slice(-6, 6)
                self.check(result, s, indexer, False)

            # positional indexing
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[-6\.0\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[slice(-6.0, 6.0)]

            # getitem odd floats
            for idx, res1 in [
                (slice(2.5, 4), slice(3, 5)),
                (slice(2, 3.5), slice(2, 4)),
                (slice(2.5, 3.5), slice(3, 4)),
            ]:
                result = s.loc[idx]
                if oob:
                    res = slice(0, 0)
                else:
                    res = res1

                self.check(result, s, res, False)

                # positional indexing
                msg = (
                    "cannot do slice indexing "
                    rf"on {type(index).__name__} with these indexers \[(2|3)\.5\] of "
                    "type float"
                )
                with pytest.raises(TypeError, match=msg):
                    s[idx]

    @pytest.mark.parametrize("idx", [slice(2, 4.0), slice(2.0, 4), slice(2.0, 4.0)])
    def test_integer_positional_indexing(self, idx):
        """make sure that we are raising on positional indexing
        w.r.t. an integer index
        """
        s = Series(range(2, 6), index=range(2, 6))

        result = s[2:4]
        expected = s.iloc[2:4]
        tm.assert_series_equal(result, expected)

        klass = RangeIndex
        msg = (
            "cannot do (slice|positional) indexing "
            rf"on {klass.__name__} with these indexers \[(2|4)\.0\] of "
            "type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[idx]
        with pytest.raises(TypeError, match=msg):
            s.iloc[idx]

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_slice_integer_frame_getitem(self, index):
        # similar to above, but on the getitem dim (of a DataFrame)
        s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)

        # getitem
        for idx in [slice(0.0, 1), slice(0, 1.0), slice(0.0, 1.0)]:
            result = s.loc[idx]
            indexer = slice(0, 2)
            self.check(result, s, indexer, False)

            # positional indexing
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[(0|1)\.0\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]

        # getitem out-of-bounds
        for idx in [slice(-10, 10), slice(-10.0, 10.0)]:
            result = s.loc[idx]
            self.check(result, s, slice(-10, 10), True)

        # positional indexing
        msg = (
            "cannot do slice indexing "
            rf"on {type(index).__name__} with these indexers \[-10\.0\] of "
            "type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[slice(-10.0, 10.0)]

        # getitem odd floats
        for idx, res in [
            (slice(0.5, 1), slice(1, 2)),
            (slice(0, 0.5), slice(0, 1)),
            (slice(0.5, 1.5), slice(1, 2)),
        ]:
            result = s.loc[idx]
            self.check(result, s, res, False)

            # positional indexing
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[0\.5\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_float_slice_getitem_with_integer_index_raises(self, idx, index):
        # similar to above, but on the getitem dim (of a DataFrame)
        s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)

        # setitem
        sc = s.copy()
        sc.loc[idx] = 0
        result = sc.loc[idx].values.ravel()
        assert (result == 0).all()

        # positional indexing
        msg = (
            "cannot do slice indexing "
            rf"on {type(index).__name__} with these indexers \[(3|4)\.0\] of "
            "type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[idx] = 0

        with pytest.raises(TypeError, match=msg):
            s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_slice_float(self, idx, frame_or_series, indexer_sl):
        # same as above, but for floats
        index = Index(np.arange(5.0)) + 0.1
        s = gen_obj(frame_or_series, index)

        expected = s.iloc[3:4]

        # getitem
        result = indexer_sl(s)[idx]
        assert isinstance(result, type(s))
        tm.assert_equal(result, expected)

        # setitem
        s2 = s.copy()
        indexer_sl(s2)[idx] = 0
        result = indexer_sl(s2)[idx].values.ravel()
        assert (result == 0).all()

    def test_floating_index_doc_example(self):
        index = Index([1.5, 2, 3, 4.5, 5])
        s = Series(range(5), index=index)
        assert s[3] == 2
        assert s.loc[3] == 2
        assert s.iloc[3] == 3

    def test_floating_misc(self, indexer_sl):
        # related 236
        # scalar/slicing of a float index
        s = Series(np.arange(5), index=np.arange(5) * 2.5, dtype=np.int64)

        # label based slicing
        result = indexer_sl(s)[1.0:3.0]
        expected = Series(1, index=[2.5])
        tm.assert_series_equal(result, expected)

        # exact indexing when found

        result = indexer_sl(s)[5.0]
        assert result == 2

        result = indexer_sl(s)[5]
        assert result == 2

        # value not found (and no fallbacking at all)

        # scalar integers
        with pytest.raises(KeyError, match=r"^4$"):
            indexer_sl(s)[4]

        # fancy floats/integers create the correct entry (as nan)
        # fancy tests
        expected = Series([2, 0], index=Index([5.0, 0.0], dtype=np.float64))
        for fancy_idx in [[5.0, 0.0], np.array([5.0, 0.0])]:  # float
            tm.assert_series_equal(indexer_sl(s)[fancy_idx], expected)

        expected = Series([2, 0], index=Index([5, 0], dtype="float64"))
        for fancy_idx in [[5, 0], np.array([5, 0])]:
            tm.assert_series_equal(indexer_sl(s)[fancy_idx], expected)

        warn = FutureWarning if indexer_sl is tm.setitem else None
        msg = r"The behavior of obj\[i:j\] with a float-dtype index"

        # all should return the same as we are slicing 'the same'
        with tm.assert_produces_warning(warn, match=msg):
            result1 = indexer_sl(s)[2:5]
        result2 = indexer_sl(s)[2.0:5.0]
        result3 = indexer_sl(s)[2.0:5]
        result4 = indexer_sl(s)[2.1:5]
        tm.assert_series_equal(result1, result2)
        tm.assert_series_equal(result1, result3)
        tm.assert_series_equal(result1, result4)

        expected = Series([1, 2], index=[2.5, 5.0])
        with tm.assert_produces_warning(warn, match=msg):
            result = indexer_sl(s)[2:5]

        tm.assert_series_equal(result, expected)

        # list selection
        result1 = indexer_sl(s)[[0.0, 5, 10]]
        result2 = s.iloc[[0, 2, 4]]
        tm.assert_series_equal(result1, result2)

        with pytest.raises(KeyError, match="not in index"):
            indexer_sl(s)[[1.6, 5, 10]]

        with pytest.raises(KeyError, match="not in index"):
            indexer_sl(s)[[0, 1, 2]]

        result = indexer_sl(s)[[2.5, 5]]
        tm.assert_series_equal(result, Series([1, 2], index=[2.5, 5.0]))

        result = indexer_sl(s)[[2.5]]
        tm.assert_series_equal(result, Series([1], index=[2.5]))

    def test_floatindex_slicing_bug(self, float_numpy_dtype):
        # GH 5557, related to slicing a float index
        dtype = float_numpy_dtype
        ser = {
            256: 2321.0,
            1: 78.0,
            2: 2716.0,
            3: 0.0,
            4: 369.0,
            5: 0.0,
            6: 269.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 3536.0,
            11: 0.0,
            12: 24.0,
            13: 0.0,
            14: 931.0,
            15: 0.0,
            16: 101.0,
            17: 78.0,
            18: 9643.0,
            19: 0.0,
            20: 0.0,
            21: 0.0,
            22: 63761.0,
            23: 0.0,
            24: 446.0,
            25: 0.0,
            26: 34773.0,
            27: 0.0,
            28: 729.0,
            29: 78.0,
            30: 0.0,
            31: 0.0,
            32: 3374.0,
            33: 0.0,
            34: 1391.0,
            35: 0.0,
            36: 361.0,
            37: 0.0,
            38: 61808.0,
            39: 0.0,
            40: 0.0,
            41: 0.0,
            42: 6677.0,
            43: 0.0,
            44: 802.0,
            45: 0.0,
            46: 2691.0,
            47: 0.0,
            48: 3582.0,
            49: 0.0,
            50: 734.0,
            51: 0.0,
            52: 627.0,
            53: 70.0,
            54: 2584.0,
            55: 0.0,
            56: 324.0,
            57: 0.0,
            58: 605.0,
            59: 0.0,
            60: 0.0,
            61: 0.0,
            62: 3989.0,
            63: 10.0,
            64: 42.0,
            65: 0.0,
            66: 904.0,
            67: 0.0,
            68: 88.0,
            69: 70.0,
            70: 8172.0,
            71: 0.0,
            72: 0.0,
            73: 0.0,
            74: 64902.0,
            75: 0.0,
            76: 347.0,
            77: 0.0,
            78: 36605.0,
            79: 0.0,
            80: 379.0,
            81: 70.0,
            82: 0.0,
            83: 0.0,
            84: 3001.0,
            85: 0.0,
            86: 1630.0,
            87: 7.0,
            88: 364.0,
            89: 0.0,
            90: 67404.0,
            91: 9.0,
            92: 0.0,
            93: 0.0,
            94: 7685.0,
            95: 0.0,
            96: 1017.0,
            97: 0.0,
            98: 2831.0,
            99: 0.0,
            100: 2963.0,
            101: 0.0,
            102: 854.0,
            103: 0.0,
            104: 0.0,
            105: 0.0,
            106: 0.0,
            107: 0.0,
            108: 0.0,
            109: 0.0,
            110: 0.0,
            111: 0.0,
            112: 0.0,
            113: 0.0,
            114: 0.0,
            115: 0.0,
            116: 0.0,
            117: 0.0,
            118: 0.0,
            119: 0.0,
            120: 0.0,
            121: 0.0,
            122: 0.0,
            123: 0.0,
            124: 0.0,
            125: 0.0,
            126: 67744.0,
            127: 22.0,
            128: 264.0,
            129: 0.0,
            260: 197.0,
            268: 0.0,
            265: 0.0,
            269: 0.0,
            261: 0.0,
            266: 1198.0,
            267: 0.0,
            262: 2629.0,
            258: 775.0,
            257: 0.0,
            263: 0.0,
            259: 0.0,
            264: 163.0,
            250: 10326.0,
            251: 0.0,
            252: 1228.0,
            253: 0.0,
            254: 2769.0,
            255: 0.0,
        }

        # smoke test for the repr
        s = Series(ser, dtype=dtype)
        result = s.value_counts()
        assert result.index.dtype == dtype
        str(result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\methods\test_astype.py ===
from datetime import (
    datetime,
    timedelta,
)
from importlib import reload
import string
import sys

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td

from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DatetimeTZDtype,
    Index,
    Interval,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    to_datetime,
)
import pandas._testing as tm


def rand_str(nchars: int) -> str:
    """
    Generate one random byte string.
    """
    RANDS_CHARS = np.array(
        list(string.ascii_letters + string.digits), dtype=(np.str_, 1)
    )
    return "".join(np.random.default_rng(2).choice(RANDS_CHARS, nchars))


class TestAstypeAPI:
    def test_astype_unitless_dt64_raises(self):
        # GH#47844
        ser = Series(["1970-01-01", "1970-01-01", "1970-01-01"], dtype="datetime64[ns]")
        df = ser.to_frame()

        msg = "Casting to unit-less dtype 'datetime64' is not supported"
        with pytest.raises(TypeError, match=msg):
            ser.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            df.astype(np.datetime64)
        with pytest.raises(TypeError, match=msg):
            ser.astype("datetime64")
        with pytest.raises(TypeError, match=msg):
            df.astype("datetime64")

    def test_arg_for_errors_in_astype(self):
        # see GH#14878
        ser = Series([1, 2, 3])

        msg = (
            r"Expected value of kwarg 'errors' to be one of \['raise', "
            r"'ignore'\]\. Supplied value is 'False'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.astype(np.float64, errors=False)

        ser.astype(np.int8, errors="raise")

    @pytest.mark.parametrize("dtype_class", [dict, Series])
    def test_astype_dict_like(self, dtype_class):
        # see GH#7271
        ser = Series(range(0, 10, 2), name="abc")

        dt1 = dtype_class({"abc": str})
        result = ser.astype(dt1)
        expected = Series(["0", "2", "4", "6", "8"], name="abc", dtype="str")
        tm.assert_series_equal(result, expected)

        dt2 = dtype_class({"abc": "float64"})
        result = ser.astype(dt2)
        expected = Series([0.0, 2.0, 4.0, 6.0, 8.0], dtype="float64", name="abc")
        tm.assert_series_equal(result, expected)

        dt3 = dtype_class({"abc": str, "def": str})
        msg = (
            "Only the Series name can be used for the key in Series dtype "
            r"mappings\."
        )
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt3)

        dt4 = dtype_class({0: str})
        with pytest.raises(KeyError, match=msg):
            ser.astype(dt4)

        # GH#16717
        # if dtypes provided is empty, it should error
        if dtype_class is Series:
            dt5 = dtype_class({}, dtype=object)
        else:
            dt5 = dtype_class({})

        with pytest.raises(KeyError, match=msg):
            ser.astype(dt5)


class TestAstype:
    @pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
    def test_astype_object_to_dt64_non_nano(self, tz):
        # GH#55756, GH#54620
        ts = Timestamp("2999-01-01")
        dtype = "M8[us]"
        if tz is not None:
            dtype = f"M8[us, {tz}]"
        vals = [ts, "2999-01-02 03:04:05.678910", 2500]
        ser = Series(vals, dtype=object)
        result = ser.astype(dtype)

        # The 2500 is interpreted as microseconds, consistent with what
        #  we would get if we created DatetimeIndexes from vals[:2] and vals[2:]
        #  and concated the results.
        pointwise = [
            vals[0].tz_localize(tz),
            Timestamp(vals[1], tz=tz),
            to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
        ]
        exp_vals = [x.as_unit("us").asm8 for x in pointwise]
        exp_arr = np.array(exp_vals, dtype="M8[us]")
        expected = Series(exp_arr, dtype="M8[us]")
        if tz is not None:
            expected = expected.dt.tz_localize("UTC").dt.tz_convert(tz)
        tm.assert_series_equal(result, expected)

    def test_astype_mixed_object_to_dt64tz(self):
        # pre-2.0 this raised ValueError bc of tz mismatch
        # xref GH#32581
        ts = Timestamp("2016-01-04 05:06:07", tz="US/Pacific")
        ts2 = ts.tz_convert("Asia/Tokyo")

        ser = Series([ts, ts2], dtype=object)
        res = ser.astype("datetime64[ns, Europe/Brussels]")
        expected = Series(
            [ts.tz_convert("Europe/Brussels"), ts2.tz_convert("Europe/Brussels")],
            dtype="datetime64[ns, Europe/Brussels]",
        )
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_astype_empty_constructor_equality(self, dtype):
        # see GH#15524

        if dtype not in (
            "S",
            "V",  # poor support (if any) currently
            "M",
            "m",  # Generic timestamps raise a ValueError. Already tested.
        ):
            init_empty = Series([], dtype=dtype)
            as_type_empty = Series([]).astype(dtype)
            tm.assert_series_equal(init_empty, as_type_empty)

    @pytest.mark.parametrize("dtype", [str, np.str_])
    @pytest.mark.parametrize(
        "series",
        [
            Series([string.digits * 10, rand_str(63), rand_str(64), rand_str(1000)]),
            Series([string.digits * 10, rand_str(63), rand_str(64), np.nan, 1.0]),
        ],
    )
    def test_astype_str_map(self, dtype, series, using_infer_string):
        # see GH#4405
        using_string_dtype = using_infer_string and dtype is str
        result = series.astype(dtype)
        if using_string_dtype:
            expected = series.map(lambda val: str(val) if val is not np.nan else np.nan)
        else:
            expected = series.map(str)
            if using_infer_string:
                expected = expected.astype(object)
        tm.assert_series_equal(result, expected)

    def test_astype_float_to_period(self):
        result = Series([np.nan]).astype("period[D]")
        expected = Series([NaT], dtype="period[D]")
        tm.assert_series_equal(result, expected)

    def test_astype_no_pandas_dtype(self):
        # https://github.com/pandas-dev/pandas/pull/24866
        ser = Series([1, 2], dtype="int64")
        # Don't have NumpyEADtype in the public API, so we use `.array.dtype`,
        # which is a NumpyEADtype.
        result = ser.astype(ser.array.dtype)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    def test_astype_generic_timestamp_no_frequency(self, dtype, request):
        # see GH#15524, GH#15987
        data = [1]
        ser = Series(data)

        if np.dtype(dtype).name not in ["timedelta64", "datetime64"]:
            mark = pytest.mark.xfail(reason="GH#33890 Is assigned ns unit")
            request.applymarker(mark)

        msg = (
            rf"The '{dtype.__name__}' dtype has no unit\. "
            rf"Please pass in '{dtype.__name__}\[ns\]' instead."
        )
        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)

    def test_astype_dt64_to_str(self):
        # GH#10442 : testing astype(str) is correct for Series/DatetimeIndex
        dti = date_range("2012-01-01", periods=3)
        result = Series(dti).astype(str)
        expected = Series(["2012-01-01", "2012-01-02", "2012-01-03"], dtype="str")
        tm.assert_series_equal(result, expected)

    def test_astype_dt64tz_to_str(self):
        # GH#10442 : testing astype(str) is correct for Series/DatetimeIndex
        dti_tz = date_range("2012-01-01", periods=3, tz="US/Eastern")
        result = Series(dti_tz).astype(str)
        expected = Series(
            [
                "2012-01-01 00:00:00-05:00",
                "2012-01-02 00:00:00-05:00",
                "2012-01-03 00:00:00-05:00",
            ],
            dtype="str",
        )
        tm.assert_series_equal(result, expected)

    def test_astype_datetime(self, unit):
        ser = Series(iNaT, dtype=f"M8[{unit}]", index=range(5))

        ser = ser.astype("O")
        assert ser.dtype == np.object_

        ser = Series([datetime(2001, 1, 2, 0, 0)])

        ser = ser.astype("O")
        assert ser.dtype == np.object_

        ser = Series(
            [datetime(2001, 1, 2, 0, 0) for i in range(3)], dtype=f"M8[{unit}]"
        )

        ser[1] = np.nan
        assert ser.dtype == f"M8[{unit}]"

        ser = ser.astype("O")
        assert ser.dtype == np.object_

    def test_astype_datetime64tz(self):
        ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))

        # astype
        result = ser.astype(object)
        expected = Series(ser.astype(object), dtype=object)
        tm.assert_series_equal(result, expected)

        result = Series(ser.values).dt.tz_localize("UTC").dt.tz_convert(ser.dt.tz)
        tm.assert_series_equal(result, ser)

        # astype - object, preserves on construction
        result = Series(ser.astype(object))
        expected = ser.astype(object)
        tm.assert_series_equal(result, expected)

        # astype - datetime64[ns, tz]
        msg = "Cannot use .astype to convert from timezone-naive"
        with pytest.raises(TypeError, match=msg):
            # dt64->dt64tz astype deprecated
            Series(ser.values).astype("datetime64[ns, US/Eastern]")

        with pytest.raises(TypeError, match=msg):
            # dt64->dt64tz astype deprecated
            Series(ser.values).astype(ser.dtype)

        result = ser.astype("datetime64[ns, CET]")
        expected = Series(date_range("20130101 06:00:00", periods=3, tz="CET"))
        tm.assert_series_equal(result, expected)

    def test_astype_str_cast_dt64(self):
        # see GH#9757
        ts = Series([Timestamp("2010-01-04 00:00:00")])
        res = ts.astype(str)

        expected = Series(["2010-01-04"], dtype="str")
        tm.assert_series_equal(res, expected)

        ts = Series([Timestamp("2010-01-04 00:00:00", tz="US/Eastern")])
        res = ts.astype(str)

        expected = Series(["2010-01-04 00:00:00-05:00"], dtype="str")
        tm.assert_series_equal(res, expected)

    def test_astype_str_cast_td64(self):
        # see GH#9757

        td = Series([Timedelta(1, unit="d")])
        ser = td.astype(str)

        expected = Series(["1 days"], dtype="str")
        tm.assert_series_equal(ser, expected)

    def test_dt64_series_astype_object(self):
        dt64ser = Series(date_range("20130101", periods=3))
        result = dt64ser.astype(object)
        assert isinstance(result.iloc[0], datetime)
        assert result.dtype == np.object_

    def test_td64_series_astype_object(self):
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="timedelta64[ns]")
        result = tdser.astype(object)
        assert isinstance(result.iloc[0], timedelta)
        assert result.dtype == np.object_

    @pytest.mark.parametrize(
        "data, dtype",
        [
            (["x", "y", "z"], "string[python]"),
            pytest.param(
                ["x", "y", "z"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            (["x", "y", "z"], "category"),
            (3 * [Timestamp("2020-01-01", tz="UTC")], None),
            (3 * [Interval(0, 1)], None),
        ],
    )
    @pytest.mark.parametrize("errors", ["raise", "ignore"])
    def test_astype_ignores_errors_for_extension_dtypes(self, data, dtype, errors):
        # https://github.com/pandas-dev/pandas/issues/35471
        ser = Series(data, dtype=dtype)
        if errors == "ignore":
            expected = ser
            result = ser.astype(float, errors="ignore")
            tm.assert_series_equal(result, expected)
        else:
            msg = "(Cannot cast)|(could not convert)"
            with pytest.raises((ValueError, TypeError), match=msg):
                ser.astype(float, errors=errors)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_astype_from_float_to_str(self, dtype):
        # https://github.com/pandas-dev/pandas/issues/36451
        ser = Series([0.1], dtype=dtype)
        result = ser.astype(str)
        expected = Series(["0.1"], dtype="str")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "value, string_value",
        [
            (None, "None"),
            (np.nan, "nan"),
            (NA, "<NA>"),
        ],
    )
    def test_astype_to_str_preserves_na(self, value, string_value, using_infer_string):
        # https://github.com/pandas-dev/pandas/issues/36904
        ser = Series(["a", "b", value], dtype=object)
        result = ser.astype(str)
        expected = Series(
            ["a", "b", None if using_infer_string else string_value], dtype="str"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "int32"])
    def test_astype(self, dtype):
        ser = Series(np.random.default_rng(2).standard_normal(5), name="foo")
        as_typed = ser.astype(dtype)

        assert as_typed.dtype == dtype
        assert as_typed.name == ser.name

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_astype_cast_nan_inf_int(self, dtype, value):
        # gh-14265: check NaN and inf raise error when converting to int
        msg = "Cannot convert non-finite values \\(NA or inf\\) to integer"
        ser = Series([value])

        with pytest.raises(ValueError, match=msg):
            ser.astype(dtype)

    @pytest.mark.parametrize("dtype", [int, np.int8, np.int64])
    def test_astype_cast_object_int_fail(self, dtype):
        arr = Series(["car", "house", "tree", "1"])
        msg = r"invalid literal for int\(\) with base 10: 'car'"
        with pytest.raises(ValueError, match=msg):
            arr.astype(dtype)

    def test_astype_float_to_uint_negatives_raise(
        self, float_numpy_dtype, any_unsigned_int_numpy_dtype
    ):
        # GH#45151 We don't cast negative numbers to nonsense values
        # TODO: same for EA float/uint dtypes, signed integers?
        arr = np.arange(5).astype(float_numpy_dtype) - 3  # includes negatives
        ser = Series(arr)

        msg = "Cannot losslessly cast from .* to .*"
        with pytest.raises(ValueError, match=msg):
            ser.astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            ser.to_frame().astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            # We currently catch and re-raise in Index.astype
            Index(ser).astype(any_unsigned_int_numpy_dtype)

        with pytest.raises(ValueError, match=msg):
            ser.array.astype(any_unsigned_int_numpy_dtype)

    def test_astype_cast_object_int(self):
        arr = Series(["1", "2", "3", "4"], dtype=object)
        result = arr.astype(int)

        tm.assert_series_equal(result, Series(np.arange(1, 5)))

    def test_astype_unicode(self, using_infer_string):
        # see GH#7758: A bit of magic is required to set
        # default encoding to utf-8
        digits = string.digits
        test_series = [
            Series([digits * 10, rand_str(63), rand_str(64), rand_str(1000)]),
            Series(["データーサイエンス、お前はもう死んでいる"]),
        ]

        former_encoding = None

        if sys.getdefaultencoding() == "utf-8":
            # GH#45326 as of 2.0 Series.astype matches Index.astype by handling
            #  bytes with obj.decode() instead of str(obj)
            item = "野菜食べないとやばい"
            ser = Series([item.encode()])
            result = ser.astype(np.str_)
            expected = Series([item], dtype=object)
            tm.assert_series_equal(result, expected)

        for ser in test_series:
            res = ser.astype(np.str_)
            expec = ser.map(str)
            if using_infer_string:
                expec = expec.astype(object)
            tm.assert_series_equal(res, expec)

        # Restore the former encoding
        if former_encoding is not None and former_encoding != "utf-8":
            reload(sys)
            sys.setdefaultencoding(former_encoding)

    def test_astype_bytes(self):
        # GH#39474
        result = Series(["foo", "bar", "baz"]).astype(bytes)
        assert result.dtypes == np.dtype("S3")

    def test_astype_nan_to_bool(self):
        # GH#43018
        ser = Series(np.nan, dtype="object")
        result = ser.astype("bool")
        expected = Series(True, dtype="bool")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        tm.ALL_INT_EA_DTYPES + tm.FLOAT_EA_DTYPES,
    )
    def test_astype_ea_to_datetimetzdtype(self, dtype):
        # GH37553
        ser = Series([4, 0, 9], dtype=dtype)
        result = ser.astype(DatetimeTZDtype(tz="US/Pacific"))

        expected = Series(
            {
                0: Timestamp("1969-12-31 16:00:00.000000004-08:00", tz="US/Pacific"),
                1: Timestamp("1969-12-31 16:00:00.000000000-08:00", tz="US/Pacific"),
                2: Timestamp("1969-12-31 16:00:00.000000009-08:00", tz="US/Pacific"),
            }
        )

        tm.assert_series_equal(result, expected)

    def test_astype_retain_attrs(self, any_numpy_dtype):
        # GH#44414
        ser = Series([0, 1, 2, 3])
        ser.attrs["Location"] = "Michigan"

        result = ser.astype(any_numpy_dtype).attrs
        expected = ser.attrs

        tm.assert_dict_equal(expected, result)


class TestAstypeString:
    @pytest.mark.parametrize(
        "data, dtype",
        [
            ([True, NA], "boolean"),
            (["A", NA], "category"),
            (["2020-10-10", "2020-10-10"], "datetime64[ns]"),
            (["2020-10-10", "2020-10-10", NaT], "datetime64[ns]"),
            (
                ["2012-01-01 00:00:00-05:00", NaT],
                "datetime64[ns, US/Eastern]",
            ),
            ([1, None], "UInt16"),
            (["1/1/2021", "2/1/2021"], "period[M]"),
            (["1/1/2021", "2/1/2021", NaT], "period[M]"),
            (["1 Day", "59 Days", NaT], "timedelta64[ns]"),
            # currently no way to parse IntervalArray from a list of strings
        ],
    )
    def test_astype_string_to_extension_dtype_roundtrip(
        self, data, dtype, request, nullable_string_dtype
    ):
        if dtype == "boolean":
            mark = pytest.mark.xfail(
                reason="TODO StringArray.astype() with missing values #GH40566"
            )
            request.applymarker(mark)
        # GH-40351
        ser = Series(data, dtype=dtype)

        # Note: just passing .astype(dtype) fails for dtype="category"
        #  with bc ser.dtype.categories will be object dtype whereas
        #  result.dtype.categories will have string dtype
        result = ser.astype(nullable_string_dtype).astype(ser.dtype)
        tm.assert_series_equal(result, ser)


class TestAstypeCategorical:
    def test_astype_categorical_to_other(self):
        cat = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)

        expected = ser
        tm.assert_series_equal(ser.astype("category"), expected)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), expected)
        msg = r"Cannot cast object|str dtype to float64"
        with pytest.raises(ValueError, match=msg):
            ser.astype("float64")

        cat = Series(Categorical(["a", "b", "b", "a", "a", "c", "c", "c"]))
        exp = Series(["a", "b", "b", "a", "a", "c", "c", "c"], dtype="str")
        tm.assert_series_equal(cat.astype("str"), exp)
        s2 = Series(Categorical(["1", "2", "3", "4"]))
        exp2 = Series([1, 2, 3, 4]).astype("int")
        tm.assert_series_equal(s2.astype("int"), exp2)

        # object don't sort correctly, so just compare that we have the same
        # values
        def cmp(a, b):
            tm.assert_almost_equal(np.sort(np.unique(a)), np.sort(np.unique(b)))

        expected = Series(np.array(ser.values), name="value_group")
        cmp(ser.astype("object"), expected)
        cmp(ser.astype(np.object_), expected)

        # array conversion
        tm.assert_almost_equal(np.array(ser), np.array(ser.values))

        tm.assert_series_equal(ser.astype("category"), ser)
        tm.assert_series_equal(ser.astype(CategoricalDtype()), ser)

        roundtrip_expected = ser.cat.set_categories(
            ser.cat.categories.sort_values()
        ).cat.remove_unused_categories()
        result = ser.astype("object").astype("category")
        tm.assert_series_equal(result, roundtrip_expected)
        result = ser.astype("object").astype(CategoricalDtype())
        tm.assert_series_equal(result, roundtrip_expected)

    def test_astype_categorical_invalid_conversions(self):
        # invalid conversion (these are NOT a dtype)
        cat = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])
        ser = Series(np.random.default_rng(2).integers(0, 10000, 100)).sort_values()
        ser = cut(ser, range(0, 10500, 500), right=False, labels=cat)

        msg = (
            "dtype '<class 'pandas.core.arrays.categorical.Categorical'>' "
            "not understood"
        )
        with pytest.raises(TypeError, match=msg):
            ser.astype(Categorical)
        with pytest.raises(TypeError, match=msg):
            ser.astype("object").astype(Categorical)

    def test_astype_categoricaldtype(self):
        ser = Series(["a", "b", "a"])
        result = ser.astype(CategoricalDtype(["a", "b"], ordered=True))
        expected = Series(Categorical(["a", "b", "a"], ordered=True))
        tm.assert_series_equal(result, expected)

        result = ser.astype(CategoricalDtype(["a", "b"], ordered=False))
        expected = Series(Categorical(["a", "b", "a"], ordered=False))
        tm.assert_series_equal(result, expected)

        result = ser.astype(CategoricalDtype(["a", "b", "c"], ordered=False))
        expected = Series(
            Categorical(["a", "b", "a"], categories=["a", "b", "c"], ordered=False)
        )
        tm.assert_series_equal(result, expected)
        tm.assert_index_equal(result.cat.categories, Index(["a", "b", "c"]))

    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("dtype_ordered", [True, False])
    @pytest.mark.parametrize("series_ordered", [True, False])
    def test_astype_categorical_to_categorical(
        self, name, dtype_ordered, series_ordered
    ):
        # GH#10696, GH#18593
        s_data = list("abcaacbab")
        s_dtype = CategoricalDtype(list("bac"), ordered=series_ordered)
        ser = Series(s_data, dtype=s_dtype, name=name)

        # unspecified categories
        dtype = CategoricalDtype(ordered=dtype_ordered)
        result = ser.astype(dtype)
        exp_dtype = CategoricalDtype(s_dtype.categories, dtype_ordered)
        expected = Series(s_data, name=name, dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

        # different categories
        dtype = CategoricalDtype(list("adc"), dtype_ordered)
        result = ser.astype(dtype)
        expected = Series(s_data, name=name, dtype=dtype)
        tm.assert_series_equal(result, expected)

        if dtype_ordered is False:
            # not specifying ordered, so only test once
            expected = ser
            result = ser.astype("category")
            tm.assert_series_equal(result, expected)

    def test_astype_bool_missing_to_categorical(self):
        # GH-19182
        ser = Series([True, False, np.nan])
        assert ser.dtypes == np.object_

        result = ser.astype(CategoricalDtype(categories=[True, False]))
        expected = Series(Categorical([True, False, np.nan], categories=[True, False]))
        tm.assert_series_equal(result, expected)

    def test_astype_categories_raises(self):
        # deprecated GH#17636, removed in GH#27141
        ser = Series(["a", "b", "a"])
        with pytest.raises(TypeError, match="got an unexpected"):
            ser.astype("category", categories=["a", "b"], ordered=True)

    @pytest.mark.parametrize("items", [["a", "b", "c", "a"], [1, 2, 3, 1]])
    def test_astype_from_categorical(self, items):
        ser = Series(items)
        exp = Series(Categorical(items))
        res = ser.astype("category")
        tm.assert_series_equal(res, exp)

    def test_astype_from_categorical_with_keywords(self):
        # with keywords
        lst = ["a", "b", "c", "a"]
        ser = Series(lst)
        exp = Series(Categorical(lst, ordered=True))
        res = ser.astype(CategoricalDtype(None, ordered=True))
        tm.assert_series_equal(res, exp)

        exp = Series(Categorical(lst, categories=list("abcdef"), ordered=True))
        res = ser.astype(CategoricalDtype(list("abcdef"), ordered=True))
        tm.assert_series_equal(res, exp)

    def test_astype_timedelta64_with_np_nan(self):
        # GH45798
        result = Series([Timedelta(1), np.nan], dtype="timedelta64[ns]")
        expected = Series([Timedelta(1), NaT], dtype="timedelta64[ns]")
        tm.assert_series_equal(result, expected)

    @td.skip_if_no("pyarrow")
    def test_astype_int_na_string(self):
        # GH#57418
        ser = Series([12, NA], dtype="Int64[pyarrow]")
        result = ser.astype("string[pyarrow]")
        expected = Series(["12", NA], dtype="string[pyarrow]")
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_convert_array_to_matrix.py ===
from sympy import Lambda, S, Dummy, KroneckerProduct
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.expressions.hadamard import HadamardProduct, HadamardPower
from sympy.matrices.expressions.special import (Identity, OneMatrix, ZeroMatrix)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.tensor.array.expressions.from_array_to_matrix import _support_function_tp1_recognize, \
    _array_diag2contr_diagmatrix, convert_array_to_matrix, _remove_trivial_dims, _array2matrix, \
    _combine_removed, identify_removable_identity_matrices, _array_contraction_to_diagonal_multiple_identity
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.diagonal import DiagMatrix, DiagonalMatrix
from sympy.matrices import Trace, MatMul, Transpose
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, \
    ArrayElement, ArraySymbol, ArrayElementwiseApplyFunc, _array_tensor_product, _array_contraction, \
    _array_diagonal, _permute_dims, PermuteDims, ArrayAdd, ArrayDiagonal, ArrayContraction, ArrayTensorProduct
from sympy.testing.pytest import raises


i, j, k, l, m, n = symbols("i j k l m n")

I = Identity(k)
I1 = Identity(1)

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

x = MatrixSymbol("x", k, 1)
y = MatrixSymbol("y", k, 1)


def test_arrayexpr_convert_array_to_matrix():

    cg = _array_contraction(_array_tensor_product(M), (0, 1))
    assert convert_array_to_matrix(cg) == Trace(M)

    cg = _array_contraction(_array_tensor_product(M, N), (0, 1), (2, 3))
    assert convert_array_to_matrix(cg) == Trace(M) * Trace(N)

    cg = _array_contraction(_array_tensor_product(M, N), (0, 3), (1, 2))
    assert convert_array_to_matrix(cg) == Trace(M * N)

    cg = _array_contraction(_array_tensor_product(M, N), (0, 2), (1, 3))
    assert convert_array_to_matrix(cg) == Trace(M * N.T)

    cg = convert_matrix_to_array(M * N * P)
    assert convert_array_to_matrix(cg) == M * N * P

    cg = convert_matrix_to_array(M * N.T * P)
    assert convert_array_to_matrix(cg) == M * N.T * P

    cg = _array_contraction(_array_tensor_product(M,N,P,Q), (1, 2), (5, 6))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * N, P * Q)

    cg = _array_contraction(_array_tensor_product(-2, M, N), (1, 2))
    assert convert_array_to_matrix(cg) == -2 * M * N

    a = MatrixSymbol("a", k, 1)
    b = MatrixSymbol("b", k, 1)
    c = MatrixSymbol("c", k, 1)
    cg = PermuteDims(
        _array_contraction(
            _array_tensor_product(
                a,
                ArrayAdd(
                    _array_tensor_product(b, c),
                    _array_tensor_product(c, b),
                )
            ), (2, 4)), [0, 1, 3, 2])
    assert convert_array_to_matrix(cg) == a * (b.T * c + c.T * b)

    za = ZeroArray(m, n)
    assert convert_array_to_matrix(za) == ZeroMatrix(m, n)

    cg = _array_tensor_product(3, M)
    assert convert_array_to_matrix(cg) == 3 * M

    # Partial conversion to matrix multiplication:
    expr = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 4, 6))
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(M.T*N, P, Q), (0, 2, 4))

    x = MatrixSymbol("x", k, 1)
    cg = PermuteDims(
        _array_contraction(_array_tensor_product(OneArray(1), x, OneArray(1), DiagMatrix(Identity(1))),
                                (0, 5)), Permutation(1, 2, 3))
    assert convert_array_to_matrix(cg) == x

    expr = ArrayAdd(M, PermuteDims(M, [1, 0]))
    assert convert_array_to_matrix(expr) == M + Transpose(M)


def test_arrayexpr_convert_array_to_matrix2():
    cg = _array_contraction(_array_tensor_product(M, N), (1, 3))
    assert convert_array_to_matrix(cg) == M * N.T

    cg = PermuteDims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    cg = _array_tensor_product(M, PermuteDims(N, Permutation([1, 0])))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    cg = _array_contraction(
        PermuteDims(
            _array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])),
        (1, 2), (3, 5)
    )
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)

    cg = _array_contraction(
        _array_tensor_product(M, N, P, PermuteDims(Q, Permutation([1, 0]))),
        (1, 5), (2, 3)
    )
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * P.T * Trace(N), Q.T)

    cg = _array_tensor_product(M, PermuteDims(N, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M, N.T)

    cg = _array_tensor_product(PermuteDims(M, [1, 0]), PermuteDims(N, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M.T, N.T)

    cg = _array_tensor_product(PermuteDims(N, [1, 0]), PermuteDims(M, [1, 0]))
    assert convert_array_to_matrix(cg) == _array_tensor_product(N.T, M.T)

    cg = _array_contraction(M, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, k)*M*OneMatrix(k, 1)

    cg = _array_contraction(x, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, k)*x

    Xm = MatrixSymbol("Xm", m, n)
    cg = _array_contraction(Xm, (0,), (1,))
    assert convert_array_to_matrix(cg) == OneMatrix(1, m)*Xm*OneMatrix(n, 1)


def test_arrayexpr_convert_array_to_diagonalized_vector():

    # Check matrix recognition over trivial dimensions:

    cg = _array_tensor_product(a, b)
    assert convert_array_to_matrix(cg) == a * b.T

    cg = _array_tensor_product(I1, a, b)
    assert convert_array_to_matrix(cg) == a * b.T

    # Recognize trace inside a tensor product:

    cg = _array_contraction(_array_tensor_product(A, B, C), (0, 3), (1, 2))
    assert convert_array_to_matrix(cg) == Trace(A * B) * C

    # Transform diagonal operator to contraction:

    cg = _array_diagonal(_array_tensor_product(A, a), (1, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (1, 3))
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(a, b), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _permute_dims(
        _array_contraction(_array_tensor_product(DiagMatrix(a), OneArray(1), b), (0, 3)), [1, 2, 0]
    )
    assert convert_array_to_matrix(cg) == b.T * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(A, a), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(A, OneArray(1), DiagMatrix(a)), (0, 3))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a)

    cg = _array_diagonal(_array_tensor_product(I, x, I1), (0, 2), (3, 5))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(I, OneArray(1), I1, DiagMatrix(x)), (0, 5))
    assert convert_array_to_matrix(cg) == DiagMatrix(x)

    cg = _array_diagonal(_array_tensor_product(I, x, A, B), (1, 2), (5, 6))
    assert _array_diag2contr_diagmatrix(cg) == _array_diagonal(_array_contraction(_array_tensor_product(I, OneArray(1), A, B, DiagMatrix(x)), (1, 7)), (5, 6))
    # TODO: this is returning a wrong result:
    # convert_array_to_matrix(cg)

    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3, 5))
    assert convert_array_to_matrix(cg) == a*b.T

    cg = _array_diagonal(_array_tensor_product(I1, a, b), (1, 3))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), a, b, I1), (2, 6))
    assert convert_array_to_matrix(cg) == a*b.T

    cg = _array_diagonal(_array_tensor_product(x, I1), (1, 2))
    assert isinstance(cg, ArrayDiagonal)
    assert cg.diagonal_indices == ((1, 2),)
    assert convert_array_to_matrix(cg) == x

    cg = _array_diagonal(_array_tensor_product(x, I), (0, 2))
    assert _array_diag2contr_diagmatrix(cg) == _array_contraction(_array_tensor_product(OneArray(1), I, DiagMatrix(x)), (1, 3))
    assert convert_array_to_matrix(cg).doit() == DiagMatrix(x)

    raises(ValueError, lambda: _array_diagonal(x, (1,)))

    # Ignore identity matrices with contractions:

    cg = _array_contraction(_array_tensor_product(I, A, I, I), (0, 2), (1, 3), (5, 7))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg) == Trace(A) * I

    cg = _array_contraction(_array_tensor_product(Trace(A) * I, I, I), (1, 5), (3, 4))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg).doit() == Trace(A) * I

    # Add DiagMatrix when required:

    cg = _array_contraction(_array_tensor_product(A, a), (1, 2))
    assert cg.split_multiple_contractions() == cg
    assert convert_array_to_matrix(cg) == A * a

    cg = _array_contraction(_array_tensor_product(A, a, B), (1, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (1, 2), (3, 5))
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * B

    cg = _array_contraction(_array_tensor_product(A, a, B), (0, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), B), (0, 2), (3, 5))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a) * B

    cg = _array_contraction(_array_tensor_product(A, a, b, a.T, B), (0, 2, 4, 7, 9))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1),
                                                DiagMatrix(b), OneArray(1), DiagMatrix(a), OneArray(1), B),
                                               (0, 2), (3, 5), (6, 9), (8, 12))
    assert convert_array_to_matrix(cg) == A.T * DiagMatrix(a) * DiagMatrix(b) * DiagMatrix(a) * B.T

    cg = _array_contraction(_array_tensor_product(I1, I1, I1), (1, 2, 4))
    assert cg.split_multiple_contractions() == _array_contraction(_array_tensor_product(I1, I1, OneArray(1), I1), (1, 2), (3, 5))
    assert convert_array_to_matrix(cg) == 1

    cg = _array_contraction(_array_tensor_product(I, I, I, I, A), (1, 2, 8), (5, 6, 9))
    assert convert_array_to_matrix(cg.split_multiple_contractions()).doit() == A

    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8))
    expected = _array_contraction(_array_tensor_product(A, DiagMatrix(a), OneArray(1), C, DiagMatrix(a), OneArray(1), B), (1, 3), (2, 5), (6, 7), (8, 10))
    assert cg.split_multiple_contractions() == expected
    assert convert_array_to_matrix(cg) == A * DiagMatrix(a) * C * DiagMatrix(a) * B

    cg = _array_contraction(_array_tensor_product(a, I1, b, I1, (a.T*b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    expected = _array_contraction(_array_tensor_product(a, I1, OneArray(1), b, I1, OneArray(1), (a.T*b).applyfunc(cos)),
                                (1, 3), (2, 10), (6, 8), (7, 11))
    assert cg.split_multiple_contractions().dummy_eq(expected)
    assert convert_array_to_matrix(cg).doit().dummy_eq(MatMul(a, (a.T * b).applyfunc(cos), b.T))


def test_arrayexpr_convert_array_contraction_tp_additions():
    a = ArrayAdd(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    )
    tp = _array_tensor_product(P, a, Q)
    expr = _array_contraction(tp, (3, 4))
    expected = _array_tensor_product(
        P,
        ArrayAdd(
            _array_contraction(_array_tensor_product(M, N), (1, 2)),
            _array_contraction(_array_tensor_product(N, M), (1, 2)),
        ),
        Q
    )
    assert expr == expected
    assert convert_array_to_matrix(expr) == _array_tensor_product(P, M * N + N * M, Q)

    expr = _array_contraction(tp, (1, 2), (3, 4), (5, 6))
    result = _array_contraction(
        _array_tensor_product(
            P,
            ArrayAdd(
                _array_contraction(_array_tensor_product(M, N), (1, 2)),
                _array_contraction(_array_tensor_product(N, M), (1, 2)),
            ),
            Q
        ), (1, 2), (3, 4))
    assert expr == result
    assert convert_array_to_matrix(expr) == P * (M * N + N * M) * Q


def test_arrayexpr_convert_array_to_implicit_matmul():
    # Trivial dimensions are suppressed, so the result can be expressed in matrix form:

    cg = _array_tensor_product(a, b)
    assert convert_array_to_matrix(cg) == a * b.T

    cg = _array_tensor_product(a, b, I)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a*b.T, I)

    cg = _array_tensor_product(I, a, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(I, a*b.T)

    cg = _array_tensor_product(a, I, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a, I, b)

    cg = _array_contraction(_array_tensor_product(I, I), (1, 2))
    assert convert_array_to_matrix(cg) == I

    cg = PermuteDims(_array_tensor_product(I, Identity(1)), [0, 2, 1, 3])
    assert convert_array_to_matrix(cg) == I


def test_arrayexpr_convert_array_to_matrix_remove_trivial_dims():

    # Tensor Product:
    assert _remove_trivial_dims(_array_tensor_product(a, b)) == (a * b.T, [1, 3])
    assert _remove_trivial_dims(_array_tensor_product(a.T, b)) == (a * b.T, [0, 3])
    assert _remove_trivial_dims(_array_tensor_product(a, b.T)) == (a * b.T, [1, 2])
    assert _remove_trivial_dims(_array_tensor_product(a.T, b.T)) == (a * b.T, [0, 2])

    assert _remove_trivial_dims(_array_tensor_product(I, a.T, b.T)) == (_array_tensor_product(I, a * b.T), [2, 4])
    assert _remove_trivial_dims(_array_tensor_product(a.T, I, b.T)) == (_array_tensor_product(a.T, I, b.T), [])

    assert _remove_trivial_dims(_array_tensor_product(a, I)) == (_array_tensor_product(a, I), [])
    assert _remove_trivial_dims(_array_tensor_product(I, a)) == (_array_tensor_product(I, a), [])

    assert _remove_trivial_dims(_array_tensor_product(a.T, b.T, c, d)) == (
        _array_tensor_product(a * b.T, c * d.T), [0, 2, 5, 7])
    assert _remove_trivial_dims(_array_tensor_product(a.T, I, b.T, c, d, I)) == (
        _array_tensor_product(a.T, I, b*c.T, d, I), [4, 7])

    # Addition:

    cg = ArrayAdd(_array_tensor_product(a, b), _array_tensor_product(c, d))
    assert _remove_trivial_dims(cg) == (a * b.T + c * d.T, [1, 3])

    # Permute Dims:

    cg = PermuteDims(_array_tensor_product(a, b), Permutation(3)(1, 2))
    assert _remove_trivial_dims(cg) == (a * b.T, [2, 3])

    cg = PermuteDims(_array_tensor_product(a, I, b), Permutation(5)(1, 2, 3, 4))
    assert _remove_trivial_dims(cg) == (cg, [])

    cg = PermuteDims(_array_tensor_product(I, b, a), Permutation(5)(1, 2, 4, 5, 3))
    assert _remove_trivial_dims(cg) == (PermuteDims(_array_tensor_product(I, b * a.T), [0, 2, 3, 1]), [4, 5])

    # Diagonal:

    cg = _array_diagonal(_array_tensor_product(M, a), (1, 2))
    assert _remove_trivial_dims(cg) == (cg, [])

    # Contraction:

    cg = _array_contraction(_array_tensor_product(M, a), (1, 2))
    assert _remove_trivial_dims(cg) == (cg, [])

    # A few more cases to test the removal and shift of nested removed axes
    # with array contractions and array diagonals:
    tp = _array_tensor_product(
        OneMatrix(1, 1),
        M,
        x,
        OneMatrix(1, 1),
        Identity(1),
    )

    expr = _array_contraction(tp, (1, 8))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 5, 6, 7]

    expr = _array_contraction(tp, (1, 8), (3, 4))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 3, 4, 5]

    expr = _array_diagonal(tp, (1, 8))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 5, 6, 7, 8]

    expr = _array_diagonal(tp, (1, 8), (3, 4))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [0, 3, 4, 5, 6]

    expr = _array_diagonal(_array_contraction(_array_tensor_product(A, x, I, I1), (1, 2, 5)), (1, 4))
    rexpr, removed = _remove_trivial_dims(expr)
    assert removed == [2, 3]

    cg = _array_diagonal(_array_tensor_product(PermuteDims(_array_tensor_product(x, I1), Permutation(1, 2, 3)), (x.T*x).applyfunc(sqrt)), (2, 4), (3, 5))
    rexpr, removed = _remove_trivial_dims(cg)
    assert removed == [1, 2]

    # Contractions with identity matrices need to be followed by a permutation
    # in order
    cg = _array_contraction(_array_tensor_product(A, B, C, M, I), (1, 8))
    ret, removed = _remove_trivial_dims(cg)
    assert ret == PermuteDims(_array_tensor_product(A, B, C, M), [0, 2, 3, 4, 5, 6, 7, 1])
    assert removed == []

    cg = _array_contraction(_array_tensor_product(A, B, C, M, I), (1, 8), (3, 4))
    ret, removed = _remove_trivial_dims(cg)
    assert ret == PermuteDims(_array_contraction(_array_tensor_product(A, B, C, M), (3, 4)), [0, 2, 3, 4, 5, 1])
    assert removed == []

    # Trivial matrices are sometimes inserted into MatMul expressions:

    cg = _array_tensor_product(b*b.T, a.T*a)
    ret, removed = _remove_trivial_dims(cg)
    assert ret == b*a.T*a*b.T
    assert removed == [2, 3]

    Xs = ArraySymbol("X", (3, 2, k))
    cg = _array_tensor_product(M, Xs, b.T*c, a*a.T, b*b.T, c.T*d)
    ret, removed = _remove_trivial_dims(cg)
    assert ret == _array_tensor_product(M, Xs, a*b.T*c*c.T*d*a.T, b*b.T)
    assert removed == [5, 6, 11, 12]

    cg = _array_diagonal(_array_tensor_product(I, I1, x), (1, 4), (3, 5))
    assert _remove_trivial_dims(cg) == (PermuteDims(_array_diagonal(_array_tensor_product(I, x), (1, 2)), Permutation(1, 2)), [1])

    expr = _array_diagonal(_array_tensor_product(x, I, y), (0, 2))
    assert _remove_trivial_dims(expr) == (PermuteDims(_array_tensor_product(DiagMatrix(x), y), [1, 2, 3, 0]), [0])

    expr = _array_diagonal(_array_tensor_product(x, I, y), (0, 2), (3, 4))
    assert _remove_trivial_dims(expr) == (expr, [])


def test_arrayexpr_convert_array_to_matrix_diag2contraction_diagmatrix():
    cg = _array_diagonal(_array_tensor_product(M, a), (1, 2))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(_array_tensor_product(M, OneArray(1), DiagMatrix(a)), (1, 3))

    raises(ValueError, lambda: _array_diagonal(_array_tensor_product(a, M), (1, 2)))

    cg = _array_diagonal(_array_tensor_product(a.T, M), (1, 2))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(_array_tensor_product(OneArray(1), M, DiagMatrix(a.T)), (1, 4))

    cg = _array_diagonal(_array_tensor_product(a.T, M, N, b.T), (1, 2), (4, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a.T), DiagMatrix(b.T)), (1, 7), (3, 9))

    cg = _array_diagonal(_array_tensor_product(a, M, N, b.T), (0, 2), (4, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a), DiagMatrix(b.T)), (1, 6), (3, 9))

    cg = _array_diagonal(_array_tensor_product(a, M, N, b.T), (0, 4), (3, 7))
    res = _array_diag2contr_diagmatrix(cg)
    assert res.shape == cg.shape
    assert res == _array_contraction(
        _array_tensor_product(OneArray(1), M, N, OneArray(1), DiagMatrix(a), DiagMatrix(b.T)), (3, 6), (2, 9))

    I1 = Identity(1)
    x = MatrixSymbol("x", k, 1)
    A = MatrixSymbol("A", k, k)
    cg = _array_diagonal(_array_tensor_product(x, A.T, I1), (0, 2))
    assert _array_diag2contr_diagmatrix(cg).shape == cg.shape
    assert _array2matrix(cg).shape == cg.shape


def test_arrayexpr_convert_array_to_matrix_support_function():

    assert _support_function_tp1_recognize([], [2 * k]) == 2 * k

    assert _support_function_tp1_recognize([(1, 2)], [A, 2 * k, B, 3]) == 6 * k * A * B

    assert _support_function_tp1_recognize([(0, 3), (1, 2)], [A, B]) == Trace(A * B)

    assert _support_function_tp1_recognize([(1, 2)], [A, B]) == A * B
    assert _support_function_tp1_recognize([(0, 2)], [A, B]) == A.T * B
    assert _support_function_tp1_recognize([(1, 3)], [A, B]) == A * B.T
    assert _support_function_tp1_recognize([(0, 3)], [A, B]) == A.T * B.T

    assert _support_function_tp1_recognize([(1, 2), (5, 6)], [A, B, C, D]) == _array_tensor_product(A * B, C * D)
    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(
        _array_tensor_product(A * C, B * D), [0, 2, 1, 3])

    assert _support_function_tp1_recognize([(0, 3), (1, 4)], [A, B, C]) == B * A * C

    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4), (7, 8)],
                                           [X, Y, A, B, C, D]) == X * Y * A * B * C * D

    assert _support_function_tp1_recognize([(9, 10), (1, 2), (5, 6), (3, 4)],
                                           [X, Y, A, B, C, D]) == _array_tensor_product(X * Y * A * B, C * D)

    assert _support_function_tp1_recognize([(1, 7), (3, 8), (4, 11)], [X, Y, A, B, C, D]) == PermuteDims(
        _array_tensor_product(X * B.T, Y * C, A.T * D.T), [0, 2, 4, 1, 3, 5]
    )

    assert _support_function_tp1_recognize([(0, 1), (3, 6), (5, 8)], [X, A, B, C, D]) == PermuteDims(
        _array_tensor_product(Trace(X) * A * C, B * D), [0, 2, 1, 3])

    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [A, A, B, C, D]) == A ** 2 * B * C * D
    assert _support_function_tp1_recognize([(1, 2), (3, 4), (5, 6), (7, 8)], [X, A, B, C, D]) == X * A * B * C * D

    assert _support_function_tp1_recognize([(1, 6), (3, 8), (5, 10)], [X, Y, A, B, C, D]) == PermuteDims(
        _array_tensor_product(X * B, Y * C, A * D), [0, 2, 4, 1, 3, 5]
    )

    assert _support_function_tp1_recognize([(1, 4), (3, 6)], [A, B, C, D]) == PermuteDims(
        _array_tensor_product(A * C, B * D), [0, 2, 1, 3])

    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C*X.T*B*A*D

    assert _support_function_tp1_recognize([(0, 4), (1, 7), (2, 5), (3, 8)], [X, A, B, C, D]) == C*X.T*B*A*D


def test_convert_array_to_hadamard_products():

    expr = HadamardProduct(M, N)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = HadamardProduct(M, N)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = Q*HadamardProduct(M, N)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = Q*HadamardProduct(M, N.T)*P
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == expr

    expr = HadamardProduct(M, N)*HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret

    expr = P.T*HadamardProduct(M, N)*HadamardProduct(Q, P)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert expr == ret

    # ArrayDiagonal should be converted
    cg = _array_diagonal(_array_tensor_product(M, N, Q), (1, 3), (0, 2, 4))
    ret = convert_array_to_matrix(cg)
    expected = PermuteDims(_array_diagonal(_array_tensor_product(HadamardProduct(M.T, N.T), Q), (1, 2)), [1, 0, 2])
    assert expected == ret

    # Special case that should return the same expression:
    cg = _array_diagonal(_array_tensor_product(HadamardProduct(M, N), Q), (0, 2))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    # Hadamard products with traces:

    expr = Trace(HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N.T))

    expr = Trace(A*HadamardProduct(M, N))
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M, N)*A)

    expr = Trace(HadamardProduct(A, M)*N)
    cg = convert_matrix_to_array(expr)
    ret = convert_array_to_matrix(cg)
    assert ret == Trace(HadamardProduct(M.T, N)*A)

    # These should not be converted into Hadamard products:

    cg = _array_diagonal(_array_tensor_product(M, N), (0, 1, 2, 3))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    cg = _array_diagonal(_array_tensor_product(A), (0, 1))
    ret = convert_array_to_matrix(cg)
    assert ret == cg

    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 2, 4), (1, 3, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, N, P)

    cg = _array_diagonal(_array_tensor_product(M, N, P), (0, 3, 4), (1, 2, 5))
    assert convert_array_to_matrix(cg) == HadamardProduct(M, P, N.T)

    cg = _array_diagonal(_array_tensor_product(I, I1, x), (1, 4), (3, 5))
    assert convert_array_to_matrix(cg) == DiagMatrix(x)


def test_identify_removable_identity_matrices():

    D = DiagonalMatrix(MatrixSymbol("D", k, k))

    cg = _array_contraction(_array_tensor_product(A, B, I), (1, 2, 4, 5))
    expected = _array_contraction(_array_tensor_product(A, B), (1, 2))
    assert identify_removable_identity_matrices(cg) == expected

    cg = _array_contraction(_array_tensor_product(A, B, C, I), (1, 3, 5, 6, 7))
    expected = _array_contraction(_array_tensor_product(A, B, C), (1, 3, 5))
    assert identify_removable_identity_matrices(cg) == expected

    # Tests with diagonal matrices:

    cg = _array_contraction(_array_tensor_product(A, B, D), (1, 2, 4, 5))
    ret = identify_removable_identity_matrices(cg)
    expected = _array_contraction(_array_tensor_product(A, B, D), (1, 4), (2, 5))
    assert ret == expected

    cg = _array_contraction(_array_tensor_product(A, B, D, M, N), (1, 2, 4, 5, 6, 8))
    ret = identify_removable_identity_matrices(cg)
    assert ret == cg


def test_combine_removed():

    assert _combine_removed(6, [0, 1, 2], [0, 1, 2]) == [0, 1, 2, 3, 4, 5]
    assert _combine_removed(8, [2, 5], [1, 3, 4]) == [1, 2, 4, 5, 6]
    assert _combine_removed(8, [7], []) == [7]


def test_array_contraction_to_diagonal_multiple_identities():

    expr = _array_contraction(_array_tensor_product(A, B, I, C), (1, 2, 4), (5, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(A, B, C), (1, 2, 4))

    expr = _array_contraction(_array_tensor_product(A, I, I), (1, 2, 4))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (A, [2])
    assert convert_array_to_matrix(expr) == A

    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 4), (3, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])

    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 3, 4, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])


def test_convert_array_element_to_matrix():

    expr = ArrayElement(M, (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M, i, j)

    expr = ArrayElement(_array_contraction(_array_tensor_product(M, N), (1, 3)), (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M*N.T, i, j)

    expr = ArrayElement(_array_tensor_product(M, N), (i, j, m, n))
    assert convert_array_to_matrix(expr) == expr


def test_convert_array_elementwise_function_to_matrix():

    d = Dummy("d")

    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x.T*y)
    assert convert_array_to_matrix(expr) == sin(x.T*y)

    expr = ArrayElementwiseApplyFunc(Lambda(d, d**2), x.T*y)
    assert convert_array_to_matrix(expr) == (x.T*y)**2

    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x)
    assert convert_array_to_matrix(expr).dummy_eq(x.applyfunc(sin))

    expr = ArrayElementwiseApplyFunc(Lambda(d, 1 / (2 * sqrt(d))), x)
    assert convert_array_to_matrix(expr) == S.Half * HadamardPower(x, -S.Half)


def test_array2matrix():
    # See issue https://github.com/sympy/sympy/pull/22877
    expr = PermuteDims(ArrayContraction(ArrayTensorProduct(x, I, I1, x), (0, 3), (1, 7)), Permutation(2, 3))
    expected = PermuteDims(ArrayTensorProduct(x*x.T, I1), Permutation(3)(1, 2))
    assert _array2matrix(expr) == expected


def test_recognize_broadcasting():
    expr = ArrayTensorProduct(x.T*x, A)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(x.T*x, A), [0, 1])

    expr = ArrayTensorProduct(A, x.T*x)
    assert _remove_trivial_dims(expr) == (KroneckerProduct(A, x.T*x), [2, 3])

    expr = ArrayTensorProduct(A, B, x.T*x, C)
    assert _remove_trivial_dims(expr) == (ArrayTensorProduct(A, KroneckerProduct(B, x.T*x), C), [4, 5])

    # Always prefer matrix multiplication to Kronecker product, if possible:
    expr = ArrayTensorProduct(a, b, x.T*x)
    assert _remove_trivial_dims(expr) == (a*x.T*x*b.T, [1, 3, 4, 5])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_win_type.py ===
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    Timedelta,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer


@pytest.fixture(
    params=[
        "triang",
        "blackman",
        "hamming",
        "bartlett",
        "bohman",
        "blackmanharris",
        "nuttall",
        "barthann",
    ]
)
def win_types(request):
    return request.param


@pytest.fixture(params=["kaiser", "gaussian", "general_gaussian", "exponential"])
def win_types_special(request):
    return request.param


def test_constructor(frame_or_series):
    # GH 12669
    pytest.importorskip("scipy")
    c = frame_or_series(range(5)).rolling

    # valid
    c(win_type="boxcar", window=2, min_periods=1)
    c(win_type="boxcar", window=2, min_periods=1, center=True)
    c(win_type="boxcar", window=2, min_periods=1, center=False)


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_invalid_constructor(frame_or_series, w):
    # not valid
    pytest.importorskip("scipy")
    c = frame_or_series(range(5)).rolling
    with pytest.raises(ValueError, match="min_periods must be an integer"):
        c(win_type="boxcar", window=2, min_periods=w)
    with pytest.raises(ValueError, match="center must be a boolean"):
        c(win_type="boxcar", window=2, min_periods=1, center=w)


@pytest.mark.parametrize("wt", ["foobar", 1])
def test_invalid_constructor_wintype(frame_or_series, wt):
    pytest.importorskip("scipy")
    c = frame_or_series(range(5)).rolling
    with pytest.raises(ValueError, match="Invalid win_type"):
        c(win_type=wt, window=2)


def test_constructor_with_win_type(frame_or_series, win_types):
    # GH 12669
    pytest.importorskip("scipy")
    c = frame_or_series(range(5)).rolling
    c(win_type=win_types, window=2)


@pytest.mark.parametrize("arg", ["median", "kurt", "skew"])
def test_agg_function_support(arg):
    pytest.importorskip("scipy")
    df = DataFrame({"A": np.arange(5)})
    roll = df.rolling(2, win_type="triang")

    msg = f"'{arg}' is not a valid function for 'Window' object"
    with pytest.raises(AttributeError, match=msg):
        roll.agg(arg)

    with pytest.raises(AttributeError, match=msg):
        roll.agg([arg])

    with pytest.raises(AttributeError, match=msg):
        roll.agg({"A": arg})


def test_invalid_scipy_arg():
    # This error is raised by scipy
    pytest.importorskip("scipy")
    msg = r"boxcar\(\) got an unexpected"
    with pytest.raises(TypeError, match=msg):
        Series(range(3)).rolling(1, win_type="boxcar").mean(foo="bar")


def test_constructor_with_win_type_invalid(frame_or_series):
    # GH 13383
    pytest.importorskip("scipy")
    c = frame_or_series(range(5)).rolling

    msg = "window must be an integer 0 or greater"

    with pytest.raises(ValueError, match=msg):
        c(-1, win_type="boxcar")


def test_window_with_args(step):
    # make sure that we are aggregating window functions correctly with arg
    pytest.importorskip("scipy")
    r = Series(np.random.default_rng(2).standard_normal(100)).rolling(
        window=10, min_periods=1, win_type="gaussian", step=step
    )
    expected = concat([r.mean(std=10), r.mean(std=0.01)], axis=1)
    expected.columns = ["<lambda>", "<lambda>"]
    result = r.aggregate([lambda x: x.mean(std=10), lambda x: x.mean(std=0.01)])
    tm.assert_frame_equal(result, expected)

    def a(x):
        return x.mean(std=10)

    def b(x):
        return x.mean(std=0.01)

    expected = concat([r.mean(std=10), r.mean(std=0.01)], axis=1)
    expected.columns = ["a", "b"]
    result = r.aggregate([a, b])
    tm.assert_frame_equal(result, expected)


def test_win_type_with_method_invalid():
    pytest.importorskip("scipy")
    with pytest.raises(
        NotImplementedError, match="'single' is the only supported method type."
    ):
        Series(range(1)).rolling(1, win_type="triang", method="table")


@pytest.mark.parametrize("arg", [2000000000, "2s", Timedelta("2s")])
def test_consistent_win_type_freq(arg):
    # GH 15969
    pytest.importorskip("scipy")
    s = Series(range(1))
    with pytest.raises(ValueError, match="Invalid win_type freq"):
        s.rolling(arg, win_type="freq")


def test_win_type_freq_return_none():
    # GH 48838
    freq_roll = Series(range(2), index=date_range("2020", periods=2)).rolling("2s")
    assert freq_roll.win_type is None


def test_win_type_not_implemented():
    pytest.importorskip("scipy")

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            return np.array([0, 1]), np.array([1, 2])

    df = DataFrame({"values": range(2)})
    indexer = CustomIndexer()
    with pytest.raises(NotImplementedError, match="BaseIndexer subclasses not"):
        df.rolling(indexer, win_type="boxcar")


def test_cmov_mean(step):
    # GH 8238
    pytest.importorskip("scipy")
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    result = Series(vals).rolling(5, center=True, step=step).mean()
    expected_values = [
        np.nan,
        np.nan,
        9.962,
        11.27,
        11.564,
        12.516,
        12.818,
        12.952,
        np.nan,
        np.nan,
    ]
    expected = Series(expected_values)[::step]
    tm.assert_series_equal(expected, result)


def test_cmov_window(step):
    # GH 8238
    pytest.importorskip("scipy")
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    result = Series(vals).rolling(5, win_type="boxcar", center=True, step=step).mean()
    expected_values = [
        np.nan,
        np.nan,
        9.962,
        11.27,
        11.564,
        12.516,
        12.818,
        12.952,
        np.nan,
        np.nan,
    ]
    expected = Series(expected_values)[::step]
    tm.assert_series_equal(expected, result)


def test_cmov_window_corner(step):
    # GH 8238
    # all nan
    pytest.importorskip("scipy")
    vals = Series([np.nan] * 10)
    result = vals.rolling(5, center=True, win_type="boxcar", step=step).mean()
    assert np.isnan(result).all()

    # empty
    vals = Series([], dtype=object)
    result = vals.rolling(5, center=True, win_type="boxcar", step=step).mean()
    assert len(result) == 0

    # shorter than window
    vals = Series(np.random.default_rng(2).standard_normal(5))
    result = vals.rolling(10, win_type="boxcar", step=step).mean()
    assert np.isnan(result).all()
    assert len(result) == len(range(0, 5, step or 1))


@pytest.mark.parametrize(
    "f,xp",
    [
        (
            "mean",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [9.252, 9.392],
                [8.644, 9.906],
                [8.87, 10.208],
                [6.81, 8.588],
                [7.792, 8.644],
                [9.05, 7.824],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
        (
            "std",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [3.789706, 4.068313],
                [3.429232, 3.237411],
                [3.589269, 3.220810],
                [3.405195, 2.380655],
                [3.281839, 2.369869],
                [3.676846, 1.801799],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
        (
            "var",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [14.36187, 16.55117],
                [11.75963, 10.48083],
                [12.88285, 10.37362],
                [11.59535, 5.66752],
                [10.77047, 5.61628],
                [13.51920, 3.24648],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
        (
            "sum",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [46.26, 46.96],
                [43.22, 49.53],
                [44.35, 51.04],
                [34.05, 42.94],
                [38.96, 43.22],
                [45.25, 39.12],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
    ],
)
def test_cmov_window_frame(f, xp, step):
    # Gh 8238
    pytest.importorskip("scipy")
    df = DataFrame(
        np.array(
            [
                [12.18, 3.64],
                [10.18, 9.16],
                [13.24, 14.61],
                [4.51, 8.11],
                [6.15, 11.44],
                [9.14, 6.21],
                [11.31, 10.67],
                [2.94, 6.51],
                [9.42, 8.39],
                [12.44, 7.34],
            ]
        )
    )
    xp = DataFrame(np.array(xp))[::step]

    roll = df.rolling(5, win_type="boxcar", center=True, step=step)
    rs = getattr(roll, f)()

    tm.assert_frame_equal(xp, rs)


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4, 5])
def test_cmov_window_na_min_periods(step, min_periods):
    pytest.importorskip("scipy")
    vals = Series(np.random.default_rng(2).standard_normal(10))
    vals[4] = np.nan
    vals[8] = np.nan

    xp = vals.rolling(5, min_periods=min_periods, center=True, step=step).mean()
    rs = vals.rolling(
        5, win_type="boxcar", min_periods=min_periods, center=True, step=step
    ).mean()
    tm.assert_series_equal(xp, rs)


def test_cmov_window_regular(win_types, step):
    # GH 8238
    pytest.importorskip("scipy")
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    xps = {
        "hamming": [
            np.nan,
            np.nan,
            8.71384,
            9.56348,
            12.38009,
            14.03687,
            13.8567,
            11.81473,
            np.nan,
            np.nan,
        ],
        "triang": [
            np.nan,
            np.nan,
            9.28667,
            10.34667,
            12.00556,
            13.33889,
            13.38,
            12.33667,
            np.nan,
            np.nan,
        ],
        "barthann": [
            np.nan,
            np.nan,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            14.0825,
            11.5675,
            np.nan,
            np.nan,
        ],
        "bohman": [
            np.nan,
            np.nan,
            7.61599,
            9.1764,
            12.83559,
            14.17267,
            14.65923,
            11.10401,
            np.nan,
            np.nan,
        ],
        "blackmanharris": [
            np.nan,
            np.nan,
            6.97691,
            9.16438,
            13.05052,
            14.02156,
            15.10512,
            10.74574,
            np.nan,
            np.nan,
        ],
        "nuttall": [
            np.nan,
            np.nan,
            7.04618,
            9.16786,
            13.02671,
            14.03559,
            15.05657,
            10.78514,
            np.nan,
            np.nan,
        ],
        "blackman": [
            np.nan,
            np.nan,
            7.73345,
            9.17869,
            12.79607,
            14.20036,
            14.57726,
            11.16988,
            np.nan,
            np.nan,
        ],
        "bartlett": [
            np.nan,
            np.nan,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            14.0825,
            11.5675,
            np.nan,
            np.nan,
        ],
    }

    xp = Series(xps[win_types])[::step]
    rs = Series(vals).rolling(5, win_type=win_types, center=True, step=step).mean()
    tm.assert_series_equal(xp, rs)


def test_cmov_window_regular_linear_range(win_types, step):
    # GH 8238
    pytest.importorskip("scipy")
    vals = np.array(range(10), dtype=float)
    xp = vals.copy()
    xp[:2] = np.nan
    xp[-2:] = np.nan
    xp = Series(xp)[::step]

    rs = Series(vals).rolling(5, win_type=win_types, center=True, step=step).mean()
    tm.assert_series_equal(xp, rs)


def test_cmov_window_regular_missing_data(win_types, step):
    # GH 8238
    pytest.importorskip("scipy")
    vals = np.array(
        [6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, np.nan, 10.63, 14.48]
    )
    xps = {
        "bartlett": [
            np.nan,
            np.nan,
            9.70333,
            10.5225,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            15.61667,
            13.655,
        ],
        "blackman": [
            np.nan,
            np.nan,
            9.04582,
            11.41536,
            7.73345,
            9.17869,
            12.79607,
            14.20036,
            15.8706,
            13.655,
        ],
        "barthann": [
            np.nan,
            np.nan,
            9.70333,
            10.5225,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            15.61667,
            13.655,
        ],
        "bohman": [
            np.nan,
            np.nan,
            8.9444,
            11.56327,
            7.61599,
            9.1764,
            12.83559,
            14.17267,
            15.90976,
            13.655,
        ],
        "hamming": [
            np.nan,
            np.nan,
            9.59321,
            10.29694,
            8.71384,
            9.56348,
            12.38009,
            14.20565,
            15.24694,
            13.69758,
        ],
        "nuttall": [
            np.nan,
            np.nan,
            8.47693,
            12.2821,
            7.04618,
            9.16786,
            13.02671,
            14.03673,
            16.08759,
            13.65553,
        ],
        "triang": [
            np.nan,
            np.nan,
            9.33167,
            9.76125,
            9.28667,
            10.34667,
            12.00556,
            13.82125,
            14.49429,
            13.765,
        ],
        "blackmanharris": [
            np.nan,
            np.nan,
            8.42526,
            12.36824,
            6.97691,
            9.16438,
            13.05052,
            14.02175,
            16.1098,
            13.65509,
        ],
    }

    xp = Series(xps[win_types])[::step]
    rs = Series(vals).rolling(5, win_type=win_types, min_periods=3, step=step).mean()
    tm.assert_series_equal(xp, rs)


def test_cmov_window_special(win_types_special, step):
    # GH 8238
    pytest.importorskip("scipy")
    kwds = {
        "kaiser": {"beta": 1.0},
        "gaussian": {"std": 1.0},
        "general_gaussian": {"p": 2.0, "sig": 2.0},
        "exponential": {"tau": 10},
    }

    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])

    xps = {
        "gaussian": [
            np.nan,
            np.nan,
            8.97297,
            9.76077,
            12.24763,
            13.89053,
            13.65671,
            12.01002,
            np.nan,
            np.nan,
        ],
        "general_gaussian": [
            np.nan,
            np.nan,
            9.85011,
            10.71589,
            11.73161,
            13.08516,
            12.95111,
            12.74577,
            np.nan,
            np.nan,
        ],
        "kaiser": [
            np.nan,
            np.nan,
            9.86851,
            11.02969,
            11.65161,
            12.75129,
            12.90702,
            12.83757,
            np.nan,
            np.nan,
        ],
        "exponential": [
            np.nan,
            np.nan,
            9.83364,
            11.10472,
            11.64551,
            12.66138,
            12.92379,
            12.83770,
            np.nan,
            np.nan,
        ],
    }

    xp = Series(xps[win_types_special])[::step]
    rs = (
        Series(vals)
        .rolling(5, win_type=win_types_special, center=True, step=step)
        .mean(**kwds[win_types_special])
    )
    tm.assert_series_equal(xp, rs)


def test_cmov_window_special_linear_range(win_types_special, step):
    # GH 8238
    pytest.importorskip("scipy")
    kwds = {
        "kaiser": {"beta": 1.0},
        "gaussian": {"std": 1.0},
        "general_gaussian": {"p": 2.0, "sig": 2.0},
        "slepian": {"width": 0.5},
        "exponential": {"tau": 10},
    }

    vals = np.array(range(10), dtype=float)
    xp = vals.copy()
    xp[:2] = np.nan
    xp[-2:] = np.nan
    xp = Series(xp)[::step]

    rs = (
        Series(vals)
        .rolling(5, win_type=win_types_special, center=True, step=step)
        .mean(**kwds[win_types_special])
    )
    tm.assert_series_equal(xp, rs)


def test_weighted_var_big_window_no_segfault(win_types, center):
    # GitHub Issue #46772
    pytest.importorskip("scipy")
    x = Series(0)
    result = x.rolling(window=16, center=center, win_type=win_types).var()
    expected = Series(np.nan)

    tm.assert_series_equal(result, expected)


def test_rolling_center_axis_1():
    pytest.importorskip("scipy")
    df = DataFrame(
        {"a": [1, 1, 0, 0, 0, 1], "b": [1, 0, 0, 1, 0, 0], "c": [1, 0, 0, 1, 0, 1]}
    )

    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=3, axis=1, win_type="boxcar", center=True).sum()

    expected = DataFrame(
        {"a": [np.nan] * 6, "b": [3.0, 1.0, 0.0, 2.0, 0.0, 2.0], "c": [np.nan] * 6}
    )

    tm.assert_frame_equal(result, expected, check_dtype=True)