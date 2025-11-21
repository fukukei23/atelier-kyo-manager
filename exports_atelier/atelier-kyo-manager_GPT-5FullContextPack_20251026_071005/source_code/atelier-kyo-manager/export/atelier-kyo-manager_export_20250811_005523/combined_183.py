
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\arrow\_arrow_utils.py ===
from __future__ import annotations

import numpy as np
import pyarrow


def pyarrow_array_to_numpy_and_mask(
    arr, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a primitive pyarrow.Array to a numpy array and boolean mask based
    on the buffers of the Array.

    At the moment pyarrow.BooleanArray is not supported.

    Parameters
    ----------
    arr : pyarrow.Array
    dtype : numpy.dtype

    Returns
    -------
    (data, mask)
        Tuple of two numpy arrays with the raw data (with specified dtype) and
        a boolean mask (validity mask, so False means missing)
    """
    dtype = np.dtype(dtype)

    if pyarrow.types.is_null(arr.type):
        # No initialization of data is needed since everything is null
        data = np.empty(len(arr), dtype=dtype)
        mask = np.zeros(len(arr), dtype=bool)
        return data, mask
    buflist = arr.buffers()
    # Since Arrow buffers might contain padding and the data might be offset,
    # the buffer gets sliced here before handing it to numpy.
    # See also https://github.com/pandas-dev/pandas/issues/40896
    offset = arr.offset * dtype.itemsize
    length = len(arr) * dtype.itemsize
    data_buf = buflist[1][offset : offset + length]
    data = np.frombuffer(data_buf, dtype=dtype)
    bitmask = buflist[0]
    if bitmask is not None:
        mask = pyarrow.BooleanArray.from_buffers(
            pyarrow.bool_(), len(arr), [None, bitmask], offset=arr.offset
        )
        mask = np.asarray(mask)
    else:
        mask = np.ones(len(arr), dtype=bool)
    return data, mask

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\transforms.py ===
"""
transforms.py is for shape-preserving functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,
        Scalar,
    )


def shift(
    values: np.ndarray, periods: int, axis: AxisInt, fill_value: Scalar
) -> np.ndarray:
    new_values = values

    if periods == 0 or values.size == 0:
        return new_values.copy()

    # make sure array sent to np.roll is c_contiguous
    f_ordered = values.flags.f_contiguous
    if f_ordered:
        new_values = new_values.T
        axis = new_values.ndim - axis - 1

    if new_values.size:
        new_values = np.roll(
            new_values,
            np.intp(periods),
            axis=axis,
        )

    axis_indexer = [slice(None)] * values.ndim
    if periods > 0:
        axis_indexer[axis] = slice(None, periods)
    else:
        axis_indexer[axis] = slice(periods, None)
    new_values[tuple(axis_indexer)] = fill_value

    # restore original order
    if f_ordered:
        new_values = new_values.T

    return new_values

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\__pip-runner__.py ===
"""Execute exactly this copy of pip, within a different environment.

This file is named as it is, to ensure that this module can't be imported via
an import statement.
"""

# /!\ This version compatibility check section must be Python 2 compatible. /!\

import sys

# Copied from pyproject.toml
PYTHON_REQUIRES = (3, 9)


def version_str(version):  # type: ignore
    return ".".join(str(v) for v in version)


if sys.version_info[:2] < PYTHON_REQUIRES:
    raise SystemExit(
        "This version of pip does not support python {} (requires >={}).".format(
            version_str(sys.version_info[:2]), version_str(PYTHON_REQUIRES)
        )
    )

# From here on, we can use Python 3 features, but the syntax must remain
# Python 2 compatible.

import runpy  # noqa: E402
from importlib.machinery import PathFinder  # noqa: E402
from os.path import dirname  # noqa: E402

PIP_SOURCES_ROOT = dirname(dirname(__file__))


class PipImportRedirectingFinder:
    @classmethod
    def find_spec(self, fullname, path=None, target=None):  # type: ignore
        if fullname != "pip":
            return None

        spec = PathFinder.find_spec(fullname, [PIP_SOURCES_ROOT], target)
        assert spec, (PIP_SOURCES_ROOT, fullname)
        return spec


sys.meta_path.insert(0, PipImportRedirectingFinder())

assert __name__ == "__main__", "Cannot run __pip-runner__.py as a non-main module"
runpy.run_module("pip", run_name="__main__", alter_sys=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\_internal_utils.py ===
"""
requests._internal_utils
~~~~~~~~~~~~~~

Provides utility functions that are consumed internally by Requests
which depend on extremely few external helpers (such as compat)
"""
import re

from .compat import builtin_str

_VALID_HEADER_NAME_RE_BYTE = re.compile(rb"^[^:\s][^:\r\n]*$")
_VALID_HEADER_NAME_RE_STR = re.compile(r"^[^:\s][^:\r\n]*$")
_VALID_HEADER_VALUE_RE_BYTE = re.compile(rb"^\S[^\r\n]*$|^$")
_VALID_HEADER_VALUE_RE_STR = re.compile(r"^\S[^\r\n]*$|^$")

_HEADER_VALIDATORS_STR = (_VALID_HEADER_NAME_RE_STR, _VALID_HEADER_VALUE_RE_STR)
_HEADER_VALIDATORS_BYTE = (_VALID_HEADER_NAME_RE_BYTE, _VALID_HEADER_VALUE_RE_BYTE)
HEADER_VALIDATORS = {
    bytes: _HEADER_VALIDATORS_BYTE,
    str: _HEADER_VALIDATORS_STR,
}


def to_native_string(string, encoding="ascii"):
    """Given a string object, regardless of type, returns a representation of
    that string in the native string type, encoding and decoding where
    necessary. This assumes ASCII unless told otherwise.
    """
    if isinstance(string, builtin_str):
        out = string
    else:
        out = string.decode(encoding)

    return out


def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
    assert isinstance(u_string, str)
    try:
        u_string.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\console\consolebase.py ===
class baseconsole(object):
    def __init__(self):
        pass

    def bell(self):
        raise NotImplementedError

    def pos(self, x=None, y=None):
        """Move or query the window cursor."""
        raise NotImplementedError

    def size(self):
        raise NotImplementedError

    def rectangle(self, rect, attr=None, fill=" "):
        """Fill Rectangle."""
        raise NotImplementedError

    def write_scrolling(self, text, attr=None):
        """write text at current cursor position while watching for scrolling.

        If the window scrolls because you are at the bottom of the screen
        buffer, all positions that you are storing will be shifted by the
        scroll amount. For example, I remember the cursor position of the
        prompt so that I can redraw the line but if the window scrolls,
        the remembered position is off.

        This variant of write tries to keep track of the cursor position
        so that it will know when the screen buffer is scrolled. It
        returns the number of lines that the buffer scrolled.

        """
        raise NotImplementedError

    def getkeypress(self):
        """Return next key press event from the queue, ignoring others."""
        raise NotImplementedError

    def write(self, text):
        raise NotImplementedError

    def page(self, attr=None, fill=" "):
        """Fill the entire screen."""
        raise NotImplementedError

    def isatty(self):
        return True

    def flush(self):
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\_internal_utils.py ===
"""
requests._internal_utils
~~~~~~~~~~~~~~

Provides utility functions that are consumed internally by Requests
which depend on extremely few external helpers (such as compat)
"""
import re

from .compat import builtin_str

_VALID_HEADER_NAME_RE_BYTE = re.compile(rb"^[^:\s][^:\r\n]*$")
_VALID_HEADER_NAME_RE_STR = re.compile(r"^[^:\s][^:\r\n]*$")
_VALID_HEADER_VALUE_RE_BYTE = re.compile(rb"^\S[^\r\n]*$|^$")
_VALID_HEADER_VALUE_RE_STR = re.compile(r"^\S[^\r\n]*$|^$")

_HEADER_VALIDATORS_STR = (_VALID_HEADER_NAME_RE_STR, _VALID_HEADER_VALUE_RE_STR)
_HEADER_VALIDATORS_BYTE = (_VALID_HEADER_NAME_RE_BYTE, _VALID_HEADER_VALUE_RE_BYTE)
HEADER_VALIDATORS = {
    bytes: _HEADER_VALIDATORS_BYTE,
    str: _HEADER_VALIDATORS_STR,
}


def to_native_string(string, encoding="ascii"):
    """Given a string object, regardless of type, returns a representation of
    that string in the native string type, encoding and decoding where
    necessary. This assumes ASCII unless told otherwise.
    """
    if isinstance(string, builtin_str):
        out = string
    else:
        out = string.decode(encoding)

    return out


def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
    assert isinstance(u_string, str)
    try:
        u_string.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\__init__.py ===
""" Rewrite Rules

DISCLAIMER: This module is experimental. The interface is subject to change.

A rule is a function that transforms one expression into another

    Rule :: Expr -> Expr

A strategy is a function that says how a rule should be applied to a syntax
tree. In general strategies take rules and produce a new rule

    Strategy :: [Rules], Other-stuff -> Rule

This allows developers to separate a mathematical transformation from the
algorithmic details of applying that transformation. The goal is to separate
the work of mathematical programming from algorithmic programming.

Submodules

strategies.rl         - some fundamental rules
strategies.core       - generic non-SymPy specific strategies
strategies.traverse   - strategies that traverse a SymPy tree
strategies.tools      - some conglomerate strategies that do depend on SymPy
"""

from . import rl
from . import traverse
from .rl import rm_id, unpack, flatten, sort, glom, distribute, rebuild
from .util import new
from .core import (
    condition, debug, chain, null_safe, do_one, exhaust, minimize, tryit)
from .tools import canon, typed
from . import branch

__all__ = [
    'rl',

    'traverse',

    'rm_id', 'unpack', 'flatten', 'sort', 'glom', 'distribute', 'rebuild',

    'new',

    'condition', 'debug', 'chain', 'null_safe', 'do_one', 'exhaust',
    'minimize', 'tryit',

    'canon', 'typed',

    'branch',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\__init__.py ===
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import (Vector, VectorAdd, VectorMul,
                                 BaseVector, VectorZero, Cross, Dot, cross, dot)
from sympy.vector.dyadic import (Dyadic, DyadicAdd, DyadicMul,
                                 BaseDyadic, DyadicZero)
from sympy.vector.scalar import BaseScalar
from sympy.vector.deloperator import Del
from sympy.vector.functions import (express, matrix_to_vector,
                                    laplacian, is_conservative,
                                    is_solenoidal, scalar_potential,
                                    directional_derivative,
                                    scalar_potential_difference)
from sympy.vector.point import Point
from sympy.vector.orienters import (AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)
from sympy.vector.operators import Gradient, Divergence, Curl, Laplacian, gradient, curl, divergence
from sympy.vector.implicitregion import ImplicitRegion
from sympy.vector.parametricregion import (ParametricRegion, parametric_region_list)
from sympy.vector.integrals import (ParametricIntegral, vector_integrate)
from sympy.vector.kind import VectorKind

__all__ = [
    'Vector', 'VectorAdd', 'VectorMul', 'BaseVector', 'VectorZero', 'Cross',
    'Dot', 'cross', 'dot',

    'VectorKind',

    'Dyadic', 'DyadicAdd', 'DyadicMul', 'BaseDyadic', 'DyadicZero',

    'BaseScalar',

    'Del',

    'CoordSys3D',

    'express', 'matrix_to_vector', 'laplacian', 'is_conservative',
    'is_solenoidal', 'scalar_potential', 'directional_derivative',
    'scalar_potential_difference',

    'Point',

    'AxisOrienter', 'BodyOrienter', 'SpaceOrienter', 'QuaternionOrienter',

    'Gradient', 'Divergence', 'Curl', 'Laplacian', 'gradient', 'curl',
    'divergence',

    'ParametricRegion', 'parametric_region_list', 'ImplicitRegion',

    'ParametricIntegral', 'vector_integrate',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\_trio_test.py ===
from __future__ import annotations

from functools import partial, wraps
from typing import TYPE_CHECKING, TypeVar

from .. import _core
from ..abc import Clock, Instrument

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from typing_extensions import ParamSpec

    ArgsT = ParamSpec("ArgsT")


RetT = TypeVar("RetT")


def trio_test(fn: Callable[ArgsT, Awaitable[RetT]]) -> Callable[ArgsT, RetT]:
    """Converts an async test function to be synchronous, running via Trio.

    Usage::

        @trio_test
        async def test_whatever():
            await ...

    If a pytest fixture is passed in that subclasses the :class:`~trio.abc.Clock` or
    :class:`~trio.abc.Instrument` ABCs, then those are passed to :meth:`trio.run()`.
    """

    @wraps(fn)
    def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
        __tracebackhide__ = True
        clocks = [c for c in kwargs.values() if isinstance(c, Clock)]
        if not clocks:
            clock = None
        elif len(clocks) == 1:
            clock = clocks[0]
        else:
            raise ValueError("too many clocks spoil the broth!")
        instruments = [i for i in kwargs.values() if isinstance(i, Instrument)]
        return _core.run(
            partial(fn, *args, **kwargs),
            clock=clock,
            instruments=instruments,
        )

    return wrapper

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_generated_instrumentation.py ===
# ***********************************************************
# ******* WARNING: AUTOGENERATED! ALL EDITS WILL BE LOST ******
# *************************************************************
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ki import enable_ki_protection
from ._run import GLOBAL_RUN_CONTEXT

if TYPE_CHECKING:
    from ._instrumentation import Instrument

__all__ = ["add_instrument", "remove_instrument"]


@enable_ki_protection
def add_instrument(instrument: Instrument) -> None:
    """Start instrumenting the current run loop with the given instrument.

    Args:
      instrument (trio.abc.Instrument): The instrument to activate.

    If ``instrument`` is already active, does nothing.

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.instruments.add_instrument(instrument)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def remove_instrument(instrument: Instrument) -> None:
    """Stop instrumenting the current run loop with the given instrument.

    Args:
      instrument (trio.abc.Instrument): The instrument to de-activate.

    Raises:
      KeyError: if the instrument is not currently active. This could
          occur either because you never added it, or because you added it
          and then it raised an unhandled exception and was automatically
          deactivated.

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.instruments.remove_instrument(instrument)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\opera.py ===
import os
from typing import Optional

from webdriver_manager.core.download_manager import DownloadManager
from webdriver_manager.core.driver_cache import DriverCacheManager
from webdriver_manager.core.manager import DriverManager
from webdriver_manager.core.os_manager import OperationSystemManager
from webdriver_manager.drivers.opera import OperaDriver


class OperaDriverManager(DriverManager):
    def __init__(
            self,
            version: Optional[str] = None,
            name: str = "operadriver",
            url: str = "https://github.com/operasoftware/operachromiumdriver/"
                       "releases/",
            latest_release_url: str = "https://api.github.com/repos/"
                                      "operasoftware/operachromiumdriver/releases/latest",
            opera_release_tag: str = "https://api.github.com/repos/"
                                     "operasoftware/operachromiumdriver/releases/tags/{0}",
            download_manager: Optional[DownloadManager] = None,
            cache_manager: Optional[DriverCacheManager] = None,
            os_system_manager: Optional[OperationSystemManager] = None
    ):
        super().__init__(
            download_manager=download_manager,
            cache_manager=cache_manager
        )

        self.driver = OperaDriver(
            name=name,
            driver_version=version,
            url=url,
            latest_release_url=latest_release_url,
            opera_release_tag=opera_release_tag,
            http_client=self.http_client,
            os_system_manager=os_system_manager
        )

    def install(self) -> str:
        driver_path = self._get_driver_binary_path(self.driver)
        if not os.path.isfile(driver_path):
            for name in os.listdir(driver_path):
                if "sha512_sum" in name:
                    os.remove(os.path.join(driver_path, name))
                    break
        driver_path = os.path.join(driver_path, os.listdir(driver_path)[0])
        os.chmod(driver_path, 0o755)
        return driver_path

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_apply.py ===
from datetime import (
    date,
    datetime,
)

import numpy as np
import pytest

from pandas._config import using_string_dtype

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    bdate_range,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


def test_apply_func_that_appends_group_to_list_without_copy():
    # GH: 17718

    df = DataFrame(1, index=list(range(10)) * 10, columns=[0]).reset_index()
    groups = []

    def store(group):
        groups.append(group)

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby("index").apply(store)
    expected_value = DataFrame(
        {"index": [0] * 10, 0: [1] * 10}, index=pd.RangeIndex(0, 100, 10)
    )

    tm.assert_frame_equal(groups[0], expected_value)


def test_apply_index_date(using_infer_string):
    # GH 5788
    ts = [
        "2011-05-16 00:00",
        "2011-05-16 01:00",
        "2011-05-16 02:00",
        "2011-05-16 03:00",
        "2011-05-17 02:00",
        "2011-05-17 03:00",
        "2011-05-17 04:00",
        "2011-05-17 05:00",
        "2011-05-18 02:00",
        "2011-05-18 03:00",
        "2011-05-18 04:00",
        "2011-05-18 05:00",
    ]
    df = DataFrame(
        {
            "value": [
                1.40893,
                1.40760,
                1.40750,
                1.40649,
                1.40893,
                1.40760,
                1.40750,
                1.40649,
                1.40893,
                1.40760,
                1.40750,
                1.40649,
            ],
        },
        index=Index(pd.to_datetime(ts), name="date_time"),
    )
    expected = df.groupby(df.index.date).idxmax()
    result = df.groupby(df.index.date).apply(lambda x: x.idxmax())
    tm.assert_frame_equal(result, expected)


def test_apply_index_date_object():
    # GH 5789
    # don't auto coerce dates
    ts = [
        "2011-05-16 00:00",
        "2011-05-16 01:00",
        "2011-05-16 02:00",
        "2011-05-16 03:00",
        "2011-05-17 02:00",
        "2011-05-17 03:00",
        "2011-05-17 04:00",
        "2011-05-17 05:00",
        "2011-05-18 02:00",
        "2011-05-18 03:00",
        "2011-05-18 04:00",
        "2011-05-18 05:00",
    ]
    df = DataFrame([row.split() for row in ts], columns=["date", "time"])
    df["value"] = [
        1.40893,
        1.40760,
        1.40750,
        1.40649,
        1.40893,
        1.40760,
        1.40750,
        1.40649,
        1.40893,
        1.40760,
        1.40750,
        1.40649,
    ]
    exp_idx = Index(["2011-05-16", "2011-05-17", "2011-05-18"], name="date")
    expected = Series(["00:00", "02:00", "02:00"], index=exp_idx)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("date", group_keys=False).apply(
            lambda x: x["time"][x["value"].idxmax()]
        )
    tm.assert_series_equal(result, expected)


def test_apply_trivial(using_infer_string):
    # GH 20066
    # trivial apply: ignore input and return a constant dataframe.
    df = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    dtype = "str" if using_infer_string else "object"
    expected = pd.concat([df.iloc[1:], df.iloc[1:]], axis=1, keys=["float64", dtype])

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby([str(x) for x in df.dtypes], axis=1)
    result = gb.apply(lambda x: df.iloc[1:])

    tm.assert_frame_equal(result, expected)


def test_apply_trivial_fail(using_infer_string):
    # GH 20066
    df = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    dtype = "str" if using_infer_string else "object"
    expected = pd.concat([df, df], axis=1, keys=["float64", dtype])
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby([str(x) for x in df.dtypes], axis=1, group_keys=True)
    result = gb.apply(lambda x: df)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "df, group_names",
    [
        (DataFrame({"a": [1, 1, 1, 2, 3], "b": ["a", "a", "a", "b", "c"]}), [1, 2, 3]),
        (DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}), [0, 1]),
        (DataFrame({"a": [1]}), [1]),
        (DataFrame({"a": [1, 1, 1, 2, 2, 1, 1, 2], "b": range(8)}), [1, 2]),
        (DataFrame({"a": [1, 2, 3, 1, 2, 3], "two": [4, 5, 6, 7, 8, 9]}), [1, 2, 3]),
        (
            DataFrame(
                {
                    "a": list("aaabbbcccc"),
                    "B": [3, 4, 3, 6, 5, 2, 1, 9, 5, 4],
                    "C": [4, 0, 2, 2, 2, 7, 8, 6, 2, 8],
                }
            ),
            ["a", "b", "c"],
        ),
        (DataFrame([[1, 2, 3], [2, 2, 3]], columns=["a", "b", "c"]), [1, 2]),
    ],
    ids=[
        "GH2936",
        "GH7739 & GH10519",
        "GH10519",
        "GH2656",
        "GH12155",
        "GH20084",
        "GH21417",
    ],
)
def test_group_apply_once_per_group(df, group_names):
    # GH2936, GH7739, GH10519, GH2656, GH12155, GH20084, GH21417

    # This test should ensure that a function is only evaluated
    # once per group. Previously the function has been evaluated twice
    # on the first group to check if the Cython index slider is safe to use
    # This test ensures that the side effect (append to list) is only triggered
    # once per group

    names = []
    # cannot parameterize over the functions since they need external
    # `names` to detect side effects

    def f_copy(group):
        # this takes the fast apply path
        names.append(group.name)
        return group.copy()

    def f_nocopy(group):
        # this takes the slow apply path
        names.append(group.name)
        return group

    def f_scalar(group):
        # GH7739, GH2656
        names.append(group.name)
        return 0

    def f_none(group):
        # GH10519, GH12155, GH21417
        names.append(group.name)

    def f_constant_df(group):
        # GH2936, GH20084
        names.append(group.name)
        return DataFrame({"a": [1], "b": [1]})

    for func in [f_copy, f_nocopy, f_scalar, f_none, f_constant_df]:
        del names[:]

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.groupby("a", group_keys=False).apply(func)
        assert names == group_names


def test_group_apply_once_per_group2(capsys):
    # GH: 31111
    # groupby-apply need to execute len(set(group_by_columns)) times

    expected = 2  # Number of times `apply` should call a function for the current test

    df = DataFrame(
        {
            "group_by_column": [0, 0, 0, 0, 1, 1, 1, 1],
            "test_column": ["0", "2", "4", "6", "8", "10", "12", "14"],
        },
        index=["0", "2", "4", "6", "8", "10", "12", "14"],
    )

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby("group_by_column", group_keys=False).apply(
            lambda df: print("function_called")
        )

    result = capsys.readouterr().out.count("function_called")
    # If `groupby` behaves unexpectedly, this test will break
    assert result == expected


def test_apply_fast_slow_identical():
    # GH 31613

    df = DataFrame({"A": [0, 0, 1], "b": range(3)})

    # For simple index structures we check for fast/slow apply using
    # an identity check on in/output
    def slow(group):
        return group

    def fast(group):
        return group.copy()

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        fast_df = df.groupby("A", group_keys=False).apply(fast)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        slow_df = df.groupby("A", group_keys=False).apply(slow)

    tm.assert_frame_equal(fast_df, slow_df)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x,
        lambda x: x[:],
        lambda x: x.copy(deep=False),
        lambda x: x.copy(deep=True),
    ],
)
def test_groupby_apply_identity_maybecopy_index_identical(func):
    # GH 14927
    # Whether the function returns a copy of the input data or not should not
    # have an impact on the index structure of the result since this is not
    # transparent to the user

    df = DataFrame({"g": [1, 2, 2, 2], "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("g", group_keys=False).apply(func)
    tm.assert_frame_equal(result, df)


def test_apply_with_mixed_dtype():
    # GH3480, apply with mixed dtype on axis=1 breaks in 0.11
    df = DataFrame(
        {
            "foo1": np.random.default_rng(2).standard_normal(6),
            "foo2": ["one", "two", "two", "three", "one", "two"],
        }
    )
    result = df.apply(lambda x: x, axis=1).dtypes
    expected = df.dtypes
    tm.assert_series_equal(result, expected)

    # GH 3610 incorrect dtype conversion with as_index=False
    df = DataFrame({"c1": [1, 2, 6, 6, 8]})
    df["c2"] = df.c1 / 2.0
    result1 = df.groupby("c2").mean().reset_index().c2
    result2 = df.groupby("c2", as_index=False).mean().c2
    tm.assert_series_equal(result1, result2)


def test_groupby_as_index_apply():
    # GH #4648 and #3417
    df = DataFrame(
        {
            "item_id": ["b", "b", "a", "c", "a", "b"],
            "user_id": [1, 2, 1, 1, 3, 1],
            "time": range(6),
        }
    )

    g_as = df.groupby("user_id", as_index=True)
    g_not_as = df.groupby("user_id", as_index=False)

    res_as = g_as.head(2).index
    res_not_as = g_not_as.head(2).index
    exp = Index([0, 1, 2, 4])
    tm.assert_index_equal(res_as, exp)
    tm.assert_index_equal(res_not_as, exp)

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res_as_apply = g_as.apply(lambda x: x.head(2)).index
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res_not_as_apply = g_not_as.apply(lambda x: x.head(2)).index

    # apply doesn't maintain the original ordering
    # changed in GH5610 as the as_index=False returns a MI here
    exp_not_as_apply = MultiIndex.from_tuples([(0, 0), (0, 2), (1, 1), (2, 4)])
    tp = [(1, 0), (1, 2), (2, 1), (3, 4)]
    exp_as_apply = MultiIndex.from_tuples(tp, names=["user_id", None])

    tm.assert_index_equal(res_as_apply, exp_as_apply)
    tm.assert_index_equal(res_not_as_apply, exp_not_as_apply)

    ind = Index(list("abcde"))
    df = DataFrame([[1, 2], [2, 3], [1, 4], [1, 5], [2, 6]], index=ind)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = df.groupby(0, as_index=False, group_keys=False).apply(lambda x: x).index
    tm.assert_index_equal(res, ind)


def test_apply_concat_preserve_names(three_group):
    grouped = three_group.groupby(["A", "B"])

    def desc(group):
        result = group.describe()
        result.index.name = "stat"
        return result

    def desc2(group):
        result = group.describe()
        result.index.name = "stat"
        result = result[: len(group)]
        # weirdo
        return result

    def desc3(group):
        result = group.describe()

        # names are different
        result.index.name = f"stat_{len(group):d}"

        result = result[: len(group)]
        # weirdo
        return result

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.apply(desc)
    assert result.index.names == ("A", "B", "stat")

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result2 = grouped.apply(desc2)
    assert result2.index.names == ("A", "B", "stat")

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result3 = grouped.apply(desc3)
    assert result3.index.names == ("A", "B", None)


def test_apply_series_to_frame():
    def f(piece):
        with np.errstate(invalid="ignore"):
            logged = np.log(piece)
        return DataFrame(
            {"value": piece, "demeaned": piece - piece.mean(), "logged": logged}
        )

    dr = bdate_range("1/1/2000", periods=100)
    ts = Series(np.random.default_rng(2).standard_normal(100), index=dr)

    grouped = ts.groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(f)

    assert isinstance(result, DataFrame)
    assert not hasattr(result, "name")  # GH49907
    tm.assert_index_equal(result.index, ts.index)


def test_apply_series_yield_constant(df):
    result = df.groupby(["A", "B"])["C"].apply(len)
    assert result.index.names[:2] == ("A", "B")


def test_apply_frame_yield_constant(df):
    # GH13568
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(["A", "B"]).apply(len)
    assert isinstance(result, Series)
    assert result.name is None

    result = df.groupby(["A", "B"])[["C", "D"]].apply(len)
    assert isinstance(result, Series)
    assert result.name is None


def test_apply_frame_to_series(df):
    grouped = df.groupby(["A", "B"])
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.apply(len)
    expected = grouped.count()["C"]
    tm.assert_index_equal(result.index, expected.index)
    tm.assert_numpy_array_equal(result.values, expected.values)


def test_apply_frame_not_as_index_column_name(df):
    # GH 35964 - path within _wrap_applied_output not hit by a test
    grouped = df.groupby(["A", "B"], as_index=False)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.apply(len)
    expected = grouped.count().rename(columns={"C": np.nan}).drop(columns="D")
    # TODO(GH#34306): Use assert_frame_equal when column name is not np.nan
    tm.assert_index_equal(result.index, expected.index)
    tm.assert_numpy_array_equal(result.values, expected.values)


def test_apply_frame_concat_series():
    def trans(group):
        return group.groupby("B")["C"].sum().sort_values().iloc[:2]

    def trans2(group):
        grouped = group.groupby(df.reindex(group.index)["B"])
        return grouped.sum().sort_values().iloc[:2]

    df = DataFrame(
        {
            "A": np.random.default_rng(2).integers(0, 5, 1000),
            "B": np.random.default_rng(2).integers(0, 5, 1000),
            "C": np.random.default_rng(2).standard_normal(1000),
        }
    )

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(trans)
    exp = df.groupby("A")["C"].apply(trans2)
    tm.assert_series_equal(result, exp, check_names=False)
    assert result.name == "C"


def test_apply_transform(ts):
    grouped = ts.groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x * 2)
    expected = grouped.transform(lambda x: x * 2)
    tm.assert_series_equal(result, expected)


def test_apply_multikey_corner(tsframe):
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])

    def f(group):
        return group.sort_values("A")[-5:]

    result = grouped.apply(f)
    for key, group in grouped:
        tm.assert_frame_equal(result.loc[key], f(group))


@pytest.mark.parametrize("group_keys", [True, False])
def test_apply_chunk_view(group_keys):
    # Low level tinkering could be unsafe, make sure not
    df = DataFrame({"key": [1, 1, 1, 2, 2, 2, 3, 3, 3], "value": range(9)})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("key", group_keys=group_keys).apply(lambda x: x.iloc[:2])
    expected = df.take([0, 1, 3, 4, 6, 7])
    if group_keys:
        expected.index = MultiIndex.from_arrays(
            [[1, 1, 2, 2, 3, 3], expected.index], names=["key", None]
        )

    tm.assert_frame_equal(result, expected)


def test_apply_no_name_column_conflict():
    df = DataFrame(
        {
            "name": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "name2": [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            "value": range(9, -1, -1),
        }
    )

    # it works! #2605
    grouped = df.groupby(["name", "name2"])
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped.apply(lambda x: x.sort_values("value", inplace=True))


def test_apply_typecast_fail():
    df = DataFrame(
        {
            "d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "c": np.tile(["a", "b", "c"], 2),
            "v": np.arange(1.0, 7.0),
        }
    )

    def f(group):
        v = group["v"]
        group["v2"] = (v - v.min()) / (v.max() - v.min())
        return group

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("d", group_keys=False).apply(f)

    expected = df.copy()
    expected["v2"] = np.tile([0.0, 0.5, 1], 2)

    tm.assert_frame_equal(result, expected)


def test_apply_multiindex_fail():
    index = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]])
    df = DataFrame(
        {
            "d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "c": np.tile(["a", "b", "c"], 2),
            "v": np.arange(1.0, 7.0),
        },
        index=index,
    )

    def f(group):
        v = group["v"]
        group["v2"] = (v - v.min()) / (v.max() - v.min())
        return group

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("d", group_keys=False).apply(f)

    expected = df.copy()
    expected["v2"] = np.tile([0.0, 0.5, 1], 2)

    tm.assert_frame_equal(result, expected)


def test_apply_corner(tsframe):
    result = tsframe.groupby(lambda x: x.year, group_keys=False).apply(lambda x: x * 2)
    expected = tsframe * 2
    tm.assert_frame_equal(result, expected)


def test_apply_without_copy():
    # GH 5545
    # returning a non-copy in an applied function fails

    data = DataFrame(
        {
            "id_field": [100, 100, 200, 300],
            "category": ["a", "b", "c", "c"],
            "value": [1, 2, 3, 4],
        }
    )

    def filt1(x):
        if x.shape[0] == 1:
            return x.copy()
        else:
            return x[x.category == "c"]

    def filt2(x):
        if x.shape[0] == 1:
            return x
        else:
            return x[x.category == "c"]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = data.groupby("id_field").apply(filt1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = data.groupby("id_field").apply(filt2)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("test_series", [True, False])
def test_apply_with_duplicated_non_sorted_axis(test_series):
    # GH 30667
    df = DataFrame(
        [["x", "p"], ["x", "p"], ["x", "o"]], columns=["X", "Y"], index=[1, 2, 2]
    )
    if test_series:
        ser = df.set_index("Y")["X"]
        result = ser.groupby(level=0, group_keys=False).apply(lambda x: x)

        # not expecting the order to remain the same for duplicated axis
        result = result.sort_index()
        expected = ser.sort_index()
        tm.assert_series_equal(result, expected)
    else:
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.groupby("Y", group_keys=False).apply(lambda x: x)

        # not expecting the order to remain the same for duplicated axis
        result = result.sort_values("Y")
        expected = df.sort_values("Y")
        tm.assert_frame_equal(result, expected)


def test_apply_reindex_values():
    # GH: 26209
    # reindexing from a single column of a groupby object with duplicate indices caused
    # a ValueError (cannot reindex from duplicate axis) in 0.24.2, the problem was
    # solved in #30679
    values = [1, 2, 3, 4]
    indices = [1, 1, 2, 2]
    df = DataFrame({"group": ["Group1", "Group2"] * 2, "value": values}, index=indices)
    expected = Series(values, index=indices, name="value")

    def reindex_helper(x):
        return x.reindex(np.arange(x.index.min(), x.index.max() + 1))

    # the following group by raised a ValueError
    result = df.groupby("group", group_keys=False).value.apply(reindex_helper)
    tm.assert_series_equal(expected, result)


def test_apply_corner_cases():
    # #535, can't use sliding iterator

    N = 1000
    labels = np.random.default_rng(2).integers(0, 100, size=N)
    df = DataFrame(
        {
            "key": labels,
            "value1": np.random.default_rng(2).standard_normal(N),
            "value2": ["foo", "bar", "baz", "qux"] * (N // 4),
        }
    )

    grouped = df.groupby("key", group_keys=False)

    def f(g):
        g["value3"] = g["value1"] * 2
        return g

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.apply(f)
    assert "value3" in result


def test_apply_numeric_coercion_when_datetime():
    # In the past, group-by/apply operations have been over-eager
    # in converting dtypes to numeric, in the presence of datetime
    # columns.  Various GH issues were filed, the reproductions
    # for which are here.

    # GH 15670
    df = DataFrame(
        {"Number": [1, 2], "Date": ["2017-03-02"] * 2, "Str": ["foo", "inf"]}
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby(["Number"]).apply(lambda x: x.iloc[0])
    df.Date = pd.to_datetime(df.Date)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(["Number"]).apply(lambda x: x.iloc[0])
    tm.assert_series_equal(result["Str"], expected["Str"])

    # GH 15421
    df = DataFrame(
        {"A": [10, 20, 30], "B": ["foo", "3", "4"], "T": [pd.Timestamp("12:31:22")] * 3}
    )

    def get_B(g):
        return g.iloc[0][["B"]]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(get_B)["B"]
    expected = df.B
    expected.index = df.A
    tm.assert_series_equal(result, expected)

    # GH 14423
    def predictions(tool):
        out = Series(index=["p1", "p2", "useTime"], dtype=object)
        if "step1" in list(tool.State):
            out["p1"] = str(tool[tool.State == "step1"].Machine.values[0])
        if "step2" in list(tool.State):
            out["p2"] = str(tool[tool.State == "step2"].Machine.values[0])
            out["useTime"] = str(tool[tool.State == "step2"].oTime.values[0])
        return out

    df1 = DataFrame(
        {
            "Key": ["B", "B", "A", "A"],
            "State": ["step1", "step2", "step1", "step2"],
            "oTime": ["", "2016-09-19 05:24:33", "", "2016-09-19 23:59:04"],
            "Machine": ["23", "36L", "36R", "36R"],
        }
    )
    df2 = df1.copy()
    df2.oTime = pd.to_datetime(df2.oTime)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df1.groupby("Key").apply(predictions).p1
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df2.groupby("Key").apply(predictions).p1
    tm.assert_series_equal(expected, result)


def test_apply_aggregating_timedelta_and_datetime():
    # Regression test for GH 15562
    # The following groupby caused ValueErrors and IndexErrors pre 0.20.0

    df = DataFrame(
        {
            "clientid": ["A", "B", "C"],
            "datetime": [np.datetime64("2017-02-01 00:00:00")] * 3,
        }
    )
    df["time_delta_zero"] = df.datetime - df.datetime
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("clientid").apply(
            lambda ddf: Series(
                {"clientid_age": ddf.time_delta_zero.min(), "date": ddf.datetime.min()}
            )
        )
    expected = DataFrame(
        {
            "clientid": ["A", "B", "C"],
            "clientid_age": [np.timedelta64(0, "D")] * 3,
            "date": [np.datetime64("2017-02-01 00:00:00")] * 3,
        }
    ).set_index("clientid")

    tm.assert_frame_equal(result, expected)


def test_apply_groupby_datetimeindex():
    # GH 26182
    # groupby apply failed on dataframe with DatetimeIndex

    data = [["A", 10], ["B", 20], ["B", 30], ["C", 40], ["C", 50]]
    df = DataFrame(
        data, columns=["Name", "Value"], index=pd.date_range("2020-09-01", "2020-09-05")
    )

    result = df.groupby("Name").sum()

    expected = DataFrame({"Name": ["A", "B", "C"], "Value": [10, 50, 90]})
    expected.set_index("Name", inplace=True)

    tm.assert_frame_equal(result, expected)


def test_time_field_bug():
    # Test a fix for the following error related to GH issue 11324 When
    # non-key fields in a group-by dataframe contained time-based fields
    # that were not returned by the apply function, an exception would be
    # raised.

    df = DataFrame({"a": 1, "b": [datetime.now() for nn in range(10)]})

    def func_with_no_date(batch):
        return Series({"c": 2})

    def func_with_date(batch):
        return Series({"b": datetime(2015, 1, 1), "c": 2})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        dfg_no_conversion = df.groupby(by=["a"]).apply(func_with_no_date)
    dfg_no_conversion_expected = DataFrame({"c": 2}, index=[1])
    dfg_no_conversion_expected.index.name = "a"

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        dfg_conversion = df.groupby(by=["a"]).apply(func_with_date)
    dfg_conversion_expected = DataFrame(
        {"b": pd.Timestamp(2015, 1, 1).as_unit("ns"), "c": 2}, index=[1]
    )
    dfg_conversion_expected.index.name = "a"

    tm.assert_frame_equal(dfg_no_conversion, dfg_no_conversion_expected)
    tm.assert_frame_equal(dfg_conversion, dfg_conversion_expected)


def test_gb_apply_list_of_unequal_len_arrays():
    # GH1738
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b", "b", "b", "a", "a", "a", "b", "b", "b"],
            "group2": ["c", "c", "d", "d", "d", "e", "c", "c", "d", "d", "d", "e"],
            "weight": [1.1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 1, 2],
            "value": [7.1, 8, 9, 10, 11, 12, 8, 7, 6, 5, 4, 3],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)

    def noddy(value, weight):
        out = np.array(value * weight).repeat(3)
        return out

    # the kernel function returns arrays of unequal length
    # pandas sniffs the first one, sees it's an array and not
    # a list, and assumed the rest are of equal length
    # and so tries a vstack

    # don't die
    df_grouped.apply(lambda x: noddy(x.value, x.weight))


def test_groupby_apply_all_none():
    # Tests to make sure no errors if apply function returns all None
    # values. Issue 9684.
    test_df = DataFrame({"groups": [0, 0, 1, 1], "random_vars": [8, 7, 4, 5]})

    def test_func(x):
        pass

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = test_df.groupby("groups").apply(test_func)
    expected = DataFrame()
    tm.assert_frame_equal(result, expected)


def test_groupby_apply_none_first():
    # GH 12824. Tests if apply returns None first.
    test_df1 = DataFrame({"groups": [1, 1, 1, 2], "vars": [0, 1, 2, 3]})
    test_df2 = DataFrame({"groups": [1, 2, 2, 2], "vars": [0, 1, 2, 3]})

    def test_func(x):
        if x.shape[0] < 2:
            return None
        return x.iloc[[0, -1]]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result1 = test_df1.groupby("groups").apply(test_func)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result2 = test_df2.groupby("groups").apply(test_func)
    index1 = MultiIndex.from_arrays([[1, 1], [0, 2]], names=["groups", None])
    index2 = MultiIndex.from_arrays([[2, 2], [1, 3]], names=["groups", None])
    expected1 = DataFrame({"groups": [1, 1], "vars": [0, 2]}, index=index1)
    expected2 = DataFrame({"groups": [2, 2], "vars": [1, 3]}, index=index2)
    tm.assert_frame_equal(result1, expected1)
    tm.assert_frame_equal(result2, expected2)


def test_groupby_apply_return_empty_chunk():
    # GH 22221: apply filter which returns some empty groups
    df = DataFrame({"value": [0, 1], "group": ["filled", "empty"]})
    groups = df.groupby("group")
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = groups.apply(lambda group: group[group.value != 1]["value"])
    expected = Series(
        [0],
        name="value",
        index=MultiIndex.from_product(
            [["empty", "filled"], [0]], names=["group", None]
        ).drop("empty"),
    )
    tm.assert_series_equal(result, expected)


def test_apply_with_mixed_types():
    # gh-20949
    df = DataFrame({"A": "a a b".split(), "B": [1, 2, 3], "C": [4, 6, 5]})
    g = df.groupby("A", group_keys=False)

    result = g.transform(lambda x: x / x.sum())
    expected = DataFrame({"B": [1 / 3.0, 2 / 3.0, 1], "C": [0.4, 0.6, 1.0]})
    tm.assert_frame_equal(result, expected)

    result = g.apply(lambda x: x / x.sum())
    tm.assert_frame_equal(result, expected)


def test_func_returns_object():
    # GH 28652
    df = DataFrame({"a": [1, 2]}, index=Index([1, 2]))
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("a").apply(lambda g: g.index)
    expected = Series([Index([1]), Index([2])], index=Index([1, 2], name="a"))

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "group_column_dtlike",
    [datetime.today(), datetime.today().date(), datetime.today().time()],
)
def test_apply_datetime_issue(group_column_dtlike):
    # GH-28247
    # groupby-apply throws an error if one of the columns in the DataFrame
    #   is a datetime object and the column labels are different from
    #   standard int values in range(len(num_columns))

    df = DataFrame({"a": ["foo"], "b": [group_column_dtlike]})
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("a").apply(lambda x: Series(["spam"], index=[42]))

    expected = DataFrame(["spam"], Index(["foo"], dtype="str", name="a"), columns=[42])
    tm.assert_frame_equal(result, expected)


def test_apply_series_return_dataframe_groups():
    # GH 10078
    tdf = DataFrame(
        {
            "day": {
                0: pd.Timestamp("2015-02-24 00:00:00"),
                1: pd.Timestamp("2015-02-24 00:00:00"),
                2: pd.Timestamp("2015-02-24 00:00:00"),
                3: pd.Timestamp("2015-02-24 00:00:00"),
                4: pd.Timestamp("2015-02-24 00:00:00"),
            },
            "userAgent": {
                0: "some UA string",
                1: "some UA string",
                2: "some UA string",
                3: "another UA string",
                4: "some UA string",
            },
            "userId": {
                0: "17661101",
                1: "17661101",
                2: "17661101",
                3: "17661101",
                4: "17661101",
            },
        }
    )

    def most_common_values(df):
        return Series({c: s.value_counts().index[0] for c, s in df.items()})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = tdf.groupby("day").apply(most_common_values)["userId"]
    expected = Series(
        ["17661101"], index=pd.DatetimeIndex(["2015-02-24"], name="day"), name="userId"
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("category", [False, True])
def test_apply_multi_level_name(category):
    # https://github.com/pandas-dev/pandas/issues/31068
    b = [1, 2] * 5
    if category:
        b = pd.Categorical(b, categories=[1, 2, 3])
        expected_index = pd.CategoricalIndex([1, 2, 3], categories=[1, 2, 3], name="B")
        expected_values = [20, 25, 0]
    else:
        expected_index = Index([1, 2], name="B")
        expected_values = [20, 25]
    expected = DataFrame(
        {"C": expected_values, "D": expected_values}, index=expected_index
    )

    df = DataFrame(
        {"A": np.arange(10), "B": b, "C": list(range(10)), "D": list(range(10))}
    ).set_index(["A", "B"])
    result = df.groupby("B", observed=False).apply(lambda x: x.sum())
    tm.assert_frame_equal(result, expected)
    assert df.index.names == ["A", "B"]


def test_groupby_apply_datetime_result_dtypes(using_infer_string):
    # GH 14849
    data = DataFrame.from_records(
        [
            (pd.Timestamp(2016, 1, 1), "red", "dark", 1, "8"),
            (pd.Timestamp(2015, 1, 1), "green", "stormy", 2, "9"),
            (pd.Timestamp(2014, 1, 1), "blue", "bright", 3, "10"),
            (pd.Timestamp(2013, 1, 1), "blue", "calm", 4, "potato"),
        ],
        columns=["observation", "color", "mood", "intensity", "score"],
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = data.groupby("color").apply(lambda g: g.iloc[0]).dtypes
    dtype = pd.StringDtype(na_value=np.nan) if using_infer_string else object
    expected = Series(
        [np.dtype("datetime64[ns]"), dtype, dtype, np.int64, dtype],
        index=["observation", "color", "mood", "intensity", "score"],
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        pd.CategoricalIndex(list("abc")),
        pd.interval_range(0, 3),
        pd.period_range("2020", periods=3, freq="D"),
        MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)]),
    ],
)
def test_apply_index_has_complex_internals(index):
    # GH 31248
    df = DataFrame({"group": [1, 1, 2], "value": [0, 1, 0]}, index=index)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("group", group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize(
    "function, expected_values",
    [
        (lambda x: x.index.to_list(), [[0, 1], [2, 3]]),
        (lambda x: set(x.index.to_list()), [{0, 1}, {2, 3}]),
        (lambda x: tuple(x.index.to_list()), [(0, 1), (2, 3)]),
        (
            lambda x: dict(enumerate(x.index.to_list())),
            [{0: 0, 1: 1}, {0: 2, 1: 3}],
        ),
        (
            lambda x: [{n: i} for (n, i) in enumerate(x.index.to_list())],
            [[{0: 0}, {1: 1}], [{0: 2}, {1: 3}]],
        ),
    ],
)
def test_apply_function_returns_non_pandas_non_scalar(function, expected_values):
    # GH 31441
    df = DataFrame(["A", "A", "B", "B"], columns=["groups"])
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("groups").apply(function)
    expected = Series(expected_values, index=Index(["A", "B"], name="groups"))
    tm.assert_series_equal(result, expected)


def test_apply_function_returns_numpy_array():
    # GH 31605
    def fct(group):
        return group["B"].values.flatten()

    df = DataFrame({"A": ["a", "a", "b", "none"], "B": [1, 2, 3, np.nan]})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A").apply(fct)
    expected = Series(
        [[1.0, 2.0], [3.0], [np.nan]], index=Index(["a", "b", "none"], name="A")
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("function", [lambda gr: gr.index, lambda gr: gr.index + 1 - 1])
def test_apply_function_index_return(function):
    # GH: 22541
    df = DataFrame([1, 2, 2, 2, 1, 2, 3, 1, 3, 1], columns=["id"])
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("id").apply(function)
    expected = Series(
        [Index([0, 4, 7, 9]), Index([1, 2, 3, 5]), Index([6, 8])],
        index=Index([1, 2, 3], name="id"),
    )
    tm.assert_series_equal(result, expected)


def test_apply_function_with_indexing_return_column():
    # GH#7002, GH#41480, GH#49256
    df = DataFrame(
        {
            "foo1": ["one", "two", "two", "three", "one", "two"],
            "foo2": [1, 2, 4, 4, 5, 6],
        }
    )
    result = df.groupby("foo1", as_index=False).apply(lambda x: x.mean())
    expected = DataFrame(
        {
            "foo1": ["one", "three", "two"],
            "foo2": [3.0, 4.0, 4.0],
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "udf",
    [(lambda x: x.copy()), (lambda x: x.copy().rename(lambda y: y + 1))],
)
@pytest.mark.parametrize("group_keys", [True, False])
def test_apply_result_type(group_keys, udf):
    # https://github.com/pandas-dev/pandas/issues/34809
    # We'd like to control whether the group keys end up in the index
    # regardless of whether the UDF happens to be a transform.
    df = DataFrame({"A": ["a", "b"], "B": [1, 2]})
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df_result = df.groupby("A", group_keys=group_keys).apply(udf)
    series_result = df.B.groupby(df.A, group_keys=group_keys).apply(udf)

    if group_keys:
        assert df_result.index.nlevels == 2
        assert series_result.index.nlevels == 2
    else:
        assert df_result.index.nlevels == 1
        assert series_result.index.nlevels == 1


def test_result_order_group_keys_false():
    # GH 34998
    # apply result order should not depend on whether index is the same or just equal
    df = DataFrame({"A": [2, 1, 2], "B": [1, 2, 3]})
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A", group_keys=False).apply(lambda x: x)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df.groupby("A", group_keys=False).apply(lambda x: x.copy())
    tm.assert_frame_equal(result, expected)


def test_apply_with_timezones_aware():
    # GH: 27212
    dates = ["2001-01-01"] * 2 + ["2001-01-02"] * 2 + ["2001-01-03"] * 2
    index_no_tz = pd.DatetimeIndex(dates)
    index_tz = pd.DatetimeIndex(dates, tz="UTC")
    df1 = DataFrame({"x": list(range(2)) * 3, "y": range(6), "t": index_no_tz})
    df2 = DataFrame({"x": list(range(2)) * 3, "y": range(6), "t": index_tz})

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result1 = df1.groupby("x", group_keys=False).apply(
            lambda df: df[["x", "y"]].copy()
        )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result2 = df2.groupby("x", group_keys=False).apply(
            lambda df: df[["x", "y"]].copy()
        )

    tm.assert_frame_equal(result1, result2)


def test_apply_is_unchanged_when_other_methods_are_called_first(reduction_func):
    # GH #34656
    # GH #34271
    df = DataFrame(
        {
            "a": [99, 99, 99, 88, 88, 88],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [10, 20, 30, 40, 50, 60],
        }
    )

    expected = DataFrame(
        {"b": [15, 6], "c": [150, 60]},
        index=Index([88, 99], name="a"),
    )

    # Check output when no other methods are called before .apply()
    grp = df.groupby(by="a")
    msg = "The behavior of DataFrame.sum with axis=None is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        result = grp.apply(sum, include_groups=False)
    tm.assert_frame_equal(result, expected)

    # Check output when another method is called before .apply()
    grp = df.groupby(by="a")
    args = get_groupby_method_args(reduction_func, df)
    _ = getattr(grp, reduction_func)(*args)
    with tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False):
        result = grp.apply(sum, include_groups=False)
    tm.assert_frame_equal(result, expected)


def test_apply_with_date_in_multiindex_does_not_convert_to_timestamp():
    # GH 29617

    df = DataFrame(
        {
            "A": ["a", "a", "a", "b"],
            "B": [
                date(2020, 1, 10),
                date(2020, 1, 10),
                date(2020, 2, 10),
                date(2020, 2, 10),
            ],
            "C": [1, 2, 3, 4],
        },
        index=Index([100, 101, 102, 103], name="idx"),
    )

    grp = df.groupby(["A", "B"])
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grp.apply(lambda x: x.head(1))

    expected = df.iloc[[0, 2, 3]]
    expected = expected.reset_index()
    expected.index = MultiIndex.from_frame(expected[["A", "B", "idx"]])
    expected = expected.drop(columns="idx")

    tm.assert_frame_equal(result, expected)
    for val in result.index.levels[1]:
        assert type(val) is date


def test_apply_by_cols_equals_apply_by_rows_transposed():
    # GH 16646
    # Operating on the columns, or transposing and operating on the rows
    # should give the same result. There was previously a bug where the
    # by_rows operation would work fine, but by_cols would throw a ValueError

    df = DataFrame(
        np.random.default_rng(2).random([6, 4]),
        columns=MultiIndex.from_product([["A", "B"], [1, 2]]),
    )

    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.T.groupby(axis=0, level=0)
    by_rows = gb.apply(lambda x: x.droplevel(axis=0, level=0))

    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb2 = df.groupby(axis=1, level=0)
    by_cols = gb2.apply(lambda x: x.droplevel(axis=1, level=0))

    tm.assert_frame_equal(by_cols, by_rows.T)
    tm.assert_frame_equal(by_cols, df)


@pytest.mark.parametrize("dropna", [True, False])
def test_apply_dropna_with_indexed_same(dropna):
    # GH 38227
    # GH#43205
    df = DataFrame(
        {
            "col": [1, 2, 3, 4, 5],
            "group": ["a", np.nan, np.nan, "b", "b"],
        },
        index=list("xxyxz"),
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("group", dropna=dropna, group_keys=False).apply(lambda x: x)
    expected = df.dropna() if dropna else df.iloc[[0, 3, 1, 2, 4]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "as_index, expected",
    [
        pytest.param(
            False,
            DataFrame(
                [[1, 1, 1], [2, 2, 1]], columns=Index(["a", "b", None], dtype=object)
            ),
            marks=pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)"),
        ),
        [
            True,
            Series(
                [1, 1], index=MultiIndex.from_tuples([(1, 1), (2, 2)], names=["a", "b"])
            ),
        ],
    ],
)
def test_apply_as_index_constant_lambda(as_index, expected):
    # GH 13217
    df = DataFrame({"a": [1, 1, 2, 2], "b": [1, 1, 2, 2], "c": [1, 1, 1, 1]})
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(["a", "b"], as_index=as_index).apply(lambda x: 1)
    tm.assert_equal(result, expected)


def test_sort_index_groups():
    # GH 20420
    df = DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 0], "C": [1, 1, 1, 2, 2]},
        index=range(5),
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("C").apply(lambda x: x.A.sort_index())
    expected = Series(
        range(1, 6),
        index=MultiIndex.from_tuples(
            [(1, 0), (1, 1), (1, 2), (2, 3), (2, 4)], names=["C", None]
        ),
        name="A",
    )
    tm.assert_series_equal(result, expected)


def test_positional_slice_groups_datetimelike():
    # GH 21651
    expected = DataFrame(
        {
            "date": pd.date_range("2010-01-01", freq="12h", periods=5),
            "vals": range(5),
            "let": list("abcde"),
        }
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = expected.groupby(
            [expected.let, expected.date.dt.date], group_keys=False
        ).apply(lambda x: x.iloc[0:])
    tm.assert_frame_equal(result, expected)


def test_groupby_apply_shape_cache_safety():
    # GH#42702 this fails if we cache_readonly Block.shape
    df = DataFrame({"A": ["a", "a", "b"], "B": [1, 2, 3], "C": [4, 6, 5]})
    gb = df.groupby("A")
    result = gb[["B", "C"]].apply(lambda x: x.astype(float).max() - x.min())

    expected = DataFrame(
        {"B": [1.0, 0.0], "C": [2.0, 0.0]}, index=Index(["a", "b"], name="A")
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_apply_to_series_name():
    # GH52444
    df = DataFrame.from_dict(
        {
            "a": ["a", "b", "a", "b"],
            "b1": ["aa", "ac", "ac", "ad"],
            "b2": ["aa", "aa", "aa", "ac"],
        }
    )
    grp = df.groupby("a")[["b1", "b2"]]
    result = grp.apply(lambda x: x.unstack().value_counts())

    expected_idx = MultiIndex.from_arrays(
        arrays=[["a", "a", "b", "b", "b"], ["aa", "ac", "ac", "ad", "aa"]],
        names=["a", None],
    )
    expected = Series([3, 1, 2, 1, 1], index=expected_idx, name="count")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dropna", [True, False])
def test_apply_na(dropna):
    # GH#28984
    df = DataFrame(
        {"grp": [1, 1, 2, 2], "y": [1, 0, 2, 5], "z": [1, 2, np.nan, np.nan]}
    )
    dfgrp = df.groupby("grp", dropna=dropna)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = dfgrp.apply(lambda grp_df: grp_df.nlargest(1, "z"))
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = dfgrp.apply(lambda x: x.sort_values("z", ascending=False).head(1))
    tm.assert_frame_equal(result, expected)


def test_apply_empty_string_nan_coerce_bug():
    # GH#24903
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = (
            DataFrame(
                {
                    "a": [1, 1, 2, 2],
                    "b": ["", "", "", ""],
                    "c": pd.to_datetime([1, 2, 3, 4], unit="s"),
                }
            )
            .groupby(["a", "b"])
            .apply(lambda df: df.iloc[-1])
        )
    expected = DataFrame(
        [[1, "", pd.to_datetime(2, unit="s")], [2, "", pd.to_datetime(4, unit="s")]],
        columns=["a", "b", "c"],
        index=MultiIndex.from_tuples([(1, ""), (2, "")], names=["a", "b"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_values", [[1, 2, 3], [1.0, 2.0, 3.0]])
def test_apply_index_key_error_bug(index_values):
    # GH 44310
    result = DataFrame(
        {
            "a": ["aa", "a2", "a3"],
            "b": [1, 2, 3],
        },
        index=Index(index_values),
    )
    expected = DataFrame(
        {
            "b_mean": [2.0, 3.0, 1.0],
        },
        index=Index(["a2", "a3", "aa"], name="a"),
    )
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = result.groupby("a").apply(
            lambda df: Series([df["b"].mean()], index=["b_mean"])
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "arg,idx",
    [
        [
            [
                1,
                2,
                3,
            ],
            [
                0.1,
                0.3,
                0.2,
            ],
        ],
        [
            [
                1,
                2,
                3,
            ],
            [
                0.1,
                0.2,
                0.3,
            ],
        ],
        [
            [
                1,
                4,
                3,
            ],
            [
                0.1,
                0.4,
                0.2,
            ],
        ],
    ],
)
def test_apply_nonmonotonic_float_index(arg, idx):
    # GH 34455
    expected = DataFrame({"col": arg}, index=idx)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = expected.groupby("col", group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("args, kwargs", [([True], {}), ([], {"numeric_only": True})])
def test_apply_str_with_args(df, args, kwargs):
    # GH#46479
    gb = df.groupby("A")
    result = gb.apply("sum", *args, **kwargs)
    expected = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("name", ["some_name", None])
def test_result_name_when_one_group(name):
    # GH 46369
    ser = Series([1, 2], name=name)
    result = ser.groupby(["a", "a"], group_keys=False).apply(lambda x: x)
    expected = Series([1, 2], name=name)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, op",
    [
        ("apply", lambda gb: gb.values[-1]),
        ("apply", lambda gb: gb["b"].iloc[0]),
        ("agg", "skew"),
        ("agg", "prod"),
        ("agg", "sum"),
    ],
)
def test_empty_df(method, op):
    # GH 47985
    empty_df = DataFrame({"a": [], "b": []})
    gb = empty_df.groupby("a", group_keys=True)
    group = getattr(gb, "b")

    result = getattr(group, method)(op)
    expected = Series(
        [], name="b", dtype="float64", index=Index([], dtype="float64", name="a")
    )

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("include_groups", [True, False])
def test_include_groups(include_groups):
    # GH#7155
    df = DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    gb = df.groupby("a")
    warn = FutureWarning if include_groups else None
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(warn, match=msg):
        result = gb.apply(lambda x: x.sum(), include_groups=include_groups)
    expected = DataFrame({"a": [2, 2], "b": [7, 5]}, index=Index([1, 2], name="a"))
    if not include_groups:
        expected = expected[["b"]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("f", [max, min, sum])
@pytest.mark.parametrize("keys", ["jim", ["jim", "joe"]])  # Single key  # Multi-key
def test_builtins_apply(keys, f):
    # see gh-8155
    rs = np.random.default_rng(2)
    df = DataFrame(rs.integers(1, 7, (10, 2)), columns=["jim", "joe"])
    df["jolie"] = rs.standard_normal(10)

    gb = df.groupby(keys)

    fname = f.__name__

    warn = None if f is not sum else FutureWarning
    msg = "The behavior of DataFrame.sum with axis=None is deprecated"
    with tm.assert_produces_warning(
        warn, match=msg, check_stacklevel=False, raise_on_extra_warnings=False
    ):
        # Also warns on deprecation GH#53425
        result = gb.apply(f)
    ngroups = len(df.drop_duplicates(subset=keys))

    assert_msg = f"invalid frame shape: {result.shape} (expected ({ngroups}, 3))"
    assert result.shape == (ngroups, 3), assert_msg

    npfunc = lambda x: getattr(np, fname)(x, axis=0)  # numpy's equivalent function
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = gb.apply(npfunc)
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected2 = gb.apply(lambda x: npfunc(x))
    tm.assert_frame_equal(result, expected2)

    if f != sum:
        expected = gb.agg(fname).reset_index()
        expected.set_index(keys, inplace=True, drop=False)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    tm.assert_series_equal(getattr(result, fname)(axis=0), getattr(df, fname)(axis=0))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\coloredlogs\demo.py ===
# Demonstration of the coloredlogs package.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: January 14, 2018
# URL: https://coloredlogs.readthedocs.io

"""A simple demonstration of the `coloredlogs` package."""

# Standard library modules.
import os
import time

# Modules included in our package.
import coloredlogs

# If my verbose logger is installed, we'll use that for the demo.
try:
    from verboselogs import VerboseLogger as getLogger
except ImportError:
    from logging import getLogger

# Initialize a logger for this module.
logger = getLogger(__name__)

DEMO_DELAY = float(os.environ.get('COLOREDLOGS_DEMO_DELAY', '1'))
"""The number of seconds between each message emitted by :func:`demonstrate_colored_logging()`."""


def demonstrate_colored_logging():
    """Interactively demonstrate the :mod:`coloredlogs` package."""
    # Determine the available logging levels and order them by numeric value.
    decorated_levels = []
    defined_levels = coloredlogs.find_defined_levels()
    normalizer = coloredlogs.NameNormalizer()
    for name, level in defined_levels.items():
        if name != 'NOTSET':
            item = (level, normalizer.normalize_name(name))
            if item not in decorated_levels:
                decorated_levels.append(item)
    ordered_levels = sorted(decorated_levels)
    # Initialize colored output to the terminal, default to the most
    # verbose logging level but enable the user the customize it.
    coloredlogs.install(level=os.environ.get('COLOREDLOGS_LOG_LEVEL', ordered_levels[0][1]))
    # Print some examples with different timestamps.
    for level, name in ordered_levels:
        log_method = getattr(logger, name, None)
        if log_method:
            log_method("message with level %s (%i)", name, level)
            time.sleep(DEMO_DELAY)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dotenv\__init__.py ===
from typing import Any, Optional

from .main import (dotenv_values, find_dotenv, get_key, load_dotenv, set_key,
                   unset_key)


def load_ipython_extension(ipython: Any) -> None:
    from .ipython import load_ipython_extension
    load_ipython_extension(ipython)


def get_cli_string(
    path: Optional[str] = None,
    action: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    quote: Optional[str] = None,
):
    """Returns a string suitable for running as a shell script.

    Useful for converting a arguments passed to a fabric task
    to be passed to a `local` or `run` command.
    """
    command = ['dotenv']
    if quote:
        command.append(f'-q {quote}')
    if path:
        command.append(f'-f {path}')
    if action:
        command.append(action)
        if key:
            command.append(key)
            if value:
                if ' ' in value:
                    command.append(f'"{value}"')
                else:
                    command.append(value)

    return ' '.join(command).strip()


__all__ = ['get_cli_string',
           'load_dotenv',
           'dotenv_values',
           'get_key',
           'set_key',
           'unset_key',
           'find_dotenv',
           'load_ipython_extension']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\pyext\cpp_message.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Protocol message implementation hooks for C++ implementation.

Contains helper functions used to create protocol message classes from
Descriptor objects at runtime backed by the protocol buffer C++ API.
"""

__author__ = 'tibell@google.com (Johan Tibell)'

from google.protobuf.internal import api_implementation


# pylint: disable=protected-access
_message = api_implementation._c_module
# TODO: Remove this import after fix api_implementation
if _message is None:
  from google.protobuf.pyext import _message


class GeneratedProtocolMessageType(_message.MessageMeta):

  """Metaclass for protocol message classes created at runtime from Descriptors.

  The protocol compiler currently uses this metaclass to create protocol
  message classes at runtime.  Clients can also manually create their own
  classes at runtime, as in this example:

  mydescriptor = Descriptor(.....)
  factory = symbol_database.Default()
  factory.pool.AddDescriptor(mydescriptor)
  MyProtoClass = message_factory.GetMessageClass(mydescriptor)
  myproto_instance = MyProtoClass()
  myproto.foo_field = 23
  ...

  The above example will not work for nested types. If you wish to include them,
  use reflection.MakeClass() instead of manually instantiating the class in
  order to create the appropriate class structure.
  """

  # Must be consistent with the protocol-compiler code in
  # proto2/compiler/internal/generator.*.
  _DESCRIPTOR_KEY = 'DESCRIPTOR'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\indexed_list.py ===
# Copyright (c) 2010-2024 openpyxl


class IndexedList(list):
    """
    List with optimised access by value
    Based on Alex Martelli's recipe

    http://code.activestate.com/recipes/52303-the-auxiliary-dictionary-idiom-for-sequences-with-/
    """

    _dict = {}

    def __init__(self, iterable=None):
        self.clean = True
        self._dict = {}
        if iterable is not None:
            self.clean = False
            for idx, val in enumerate(iterable):
                self._dict[val] = idx
                list.append(self, val)

    def _rebuild_dict(self):
        self._dict = {}
        idx = 0
        for value in self:
            if value not in self._dict:
                self._dict[value] = idx
                idx += 1
        self.clean = True

    def __contains__(self, value):
        if not self.clean:
            self._rebuild_dict()
        return value in self._dict

    def index(self, value):
        if value in self:
            return self._dict[value]
        raise ValueError

    def append(self, value):
        if value not in self._dict:
            self._dict[value] = len(self)
            list.append(self, value)

    def add(self, value):
        self.append(value)
        return self._dict[value]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\__init__.py ===
from __future__ import absolute_import

# For backwards compatibility, provide imports that used to be here.
from .connection import is_connection_dropped
from .request import SKIP_HEADER, SKIPPABLE_HEADERS, make_headers
from .response import is_fp_closed
from .retry import Retry
from .ssl_ import (
    ALPN_PROTOCOLS,
    HAS_SNI,
    IS_PYOPENSSL,
    IS_SECURETRANSPORT,
    PROTOCOL_TLS,
    SSLContext,
    assert_fingerprint,
    resolve_cert_reqs,
    resolve_ssl_version,
    ssl_wrap_socket,
)
from .timeout import Timeout, current_time
from .url import Url, get_host, parse_url, split_first
from .wait import wait_for_read, wait_for_write

__all__ = (
    "HAS_SNI",
    "IS_PYOPENSSL",
    "IS_SECURETRANSPORT",
    "SSLContext",
    "PROTOCOL_TLS",
    "ALPN_PROTOCOLS",
    "Retry",
    "Timeout",
    "Url",
    "assert_fingerprint",
    "current_time",
    "is_connection_dropped",
    "is_fp_closed",
    "get_host",
    "parse_url",
    "make_headers",
    "resolve_cert_reqs",
    "resolve_ssl_version",
    "split_first",
    "ssl_wrap_socket",
    "wait_for_read",
    "wait_for_write",
    "SKIP_HEADER",
    "SKIPPABLE_HEADERS",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\holonomicerrors.py ===
""" Common Exceptions for `holonomic` module. """

class BaseHolonomicError(Exception):

    def new(self, *args):
        raise NotImplementedError("abstract base class")

class NotPowerSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'A Power Series does not exists for '
        s += str(self.holonomic)
        s += ' about %s.' %self.x0
        return s

class NotHolonomicError(BaseHolonomicError):

    def __init__(self, m):
        self.m = m

    def __str__(self):
        return self.m

class SingularityError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = str(self.holonomic)
        s += ' has a singularity at %s.' %self.x0
        return s

class NotHyperSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'Power series expansion of '
        s += str(self.holonomic)
        s += ' about %s is not hypergeometric' %self.x0
        return s

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest8.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
c1, c2, c3 = _sm.symbols('c1 c2 c3', real=True)
a = _me.inertia(frame_a, 1, 1, 1)
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
body_r_cm = _me.Point('r_cm')
body_r_f = _me.ReferenceFrame('r_f')
body_r = _me.RigidBody('r', body_r_cm, body_r_f, _sm.symbols('m'), (_me.outer(body_r_f.x,body_r_f.x),body_r_cm))
frame_a.orient(body_r_f, 'DCM', _sm.Matrix([1,1,1,1,1,0,0,0,1]).reshape(3, 3))
point_o = _me.Point('o')
m1 = _sm.symbols('m1')
particle_p1.mass = m1
m2 = _sm.symbols('m2')
particle_p2.mass = m2
mr = _sm.symbols('mr')
body_r.mass = mr
i1 = _sm.symbols('i1')
i2 = _sm.symbols('i2')
i3 = _sm.symbols('i3')
body_r.inertia = (_me.inertia(body_r_f, i1, i2, i3, 0, 0, 0), body_r_cm)
point_o.set_pos(particle_p1.point, c1*frame_a.x)
point_o.set_pos(particle_p2.point, c2*frame_a.y)
point_o.set_pos(body_r_cm, c3*frame_a.z)
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a)
a = _me.inertia_of_point_mass(particle_p2.mass, particle_p2.point.pos_from(point_o), frame_a)
a = body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a) + _me.inertia_of_point_mass(particle_p2.mass, particle_p2.point.pos_from(point_o), frame_a) + body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a) + body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = body_r.inertia[0]
particle_p2.point.set_pos(particle_p1.point, c1*frame_a.x+c2*frame_a.y)
body_r_cm.set_pos(particle_p1.point, c3*frame_a.x)
body_r_cm.set_pos(particle_p2.point, c3*frame_a.y)
b = _me.functions.center_of_mass(point_o,particle_p1, particle_p2, body_r)
b = _me.functions.center_of_mass(point_o,particle_p1, body_r)
b = _me.functions.center_of_mass(particle_p1.point,particle_p1, particle_p2, body_r)
u1, u2, u3 = _me.dynamicsymbols('u1 u2 u3')
v = u1*frame_a.x+u2*frame_a.y+u3*frame_a.z
u = (v+c1*frame_a.x).normalize()
particle_p1.point.set_vel(frame_a, u1*frame_a.x)
a = particle_p1.point.partial_velocity(frame_a, u1)
m = particle_p1.mass+body_r.mass
m = particle_p2.mass
m = particle_p1.mass+particle_p2.mass+body_r.mass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_rings.py ===
"""Test sparse polynomials. """

from functools import reduce
from operator import add, mul

from sympy.polys.rings import ring, xring, sring, PolyRing, PolyElement
from sympy.polys.fields import field, FracField
from sympy.polys.densebasic import ninf
from sympy.polys.domains import ZZ, QQ, RR, FF, EX
from sympy.polys.orderings import lex, grlex
from sympy.polys.polyerrors import GeneratorsError, \
    ExactQuotientFailed, MultivariatePolynomialError, CoercionFailed

from sympy.testing.pytest import raises
from sympy.core import Symbol, symbols
from sympy.core.singleton import S
from sympy.core.numbers import pi
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt

def test_PolyRing___init__():
    x, y, z, t = map(Symbol, "xyzt")

    assert len(PolyRing("x,y,z", ZZ, lex).gens) == 3
    assert len(PolyRing(x, ZZ, lex).gens) == 1
    assert len(PolyRing(("x", "y", "z"), ZZ, lex).gens) == 3
    assert len(PolyRing((x, y, z), ZZ, lex).gens) == 3
    assert len(PolyRing("", ZZ, lex).gens) == 0
    assert len(PolyRing([], ZZ, lex).gens) == 0

    raises(GeneratorsError, lambda: PolyRing(0, ZZ, lex))

    assert PolyRing("x", ZZ[t], lex).domain == ZZ[t]
    assert PolyRing("x", 'ZZ[t]', lex).domain == ZZ[t]
    assert PolyRing("x", PolyRing("t", ZZ, lex), lex).domain == ZZ[t]

    raises(GeneratorsError, lambda: PolyRing("x", PolyRing("x", ZZ, lex), lex))

    _lex = Symbol("lex")
    assert PolyRing("x", ZZ, lex).order == lex
    assert PolyRing("x", ZZ, _lex).order == lex
    assert PolyRing("x", ZZ, 'lex').order == lex

    R1 = PolyRing("x,y", ZZ, lex)
    R2 = PolyRing("x,y", ZZ, lex)
    R3 = PolyRing("x,y,z", ZZ, lex)

    assert R1.x == R1.gens[0]
    assert R1.y == R1.gens[1]
    assert R1.x == R2.x
    assert R1.y == R2.y
    assert R1.x != R3.x
    assert R1.y != R3.y

def test_PolyRing___hash__():
    R, x, y, z = ring("x,y,z", QQ)
    assert hash(R)

def test_PolyRing___eq__():
    assert ring("x,y,z", QQ)[0] == ring("x,y,z", QQ)[0]
    assert ring("x,y,z", QQ)[0] != ring("x,y,z", ZZ)[0]
    assert ring("x,y,z", ZZ)[0] != ring("x,y,z", QQ)[0]
    assert ring("x,y,z", QQ)[0] != ring("x,y", QQ)[0]
    assert ring("x,y", QQ)[0] != ring("x,y,z", QQ)[0]

def test_PolyRing_ring_new():
    R, x, y, z = ring("x,y,z", QQ)

    assert R.ring_new(7) == R(7)
    assert R.ring_new(7*x*y*z) == 7*x*y*z

    f = x**2 + 2*x*y + 3*x + 4*z**2 + 5*z + 6

    assert R.ring_new([[[1]], [[2], [3]], [[4, 5, 6]]]) == f
    assert R.ring_new({(2, 0, 0): 1, (1, 1, 0): 2, (1, 0, 0): 3, (0, 0, 2): 4, (0, 0, 1): 5, (0, 0, 0): 6}) == f
    assert R.ring_new([((2, 0, 0), 1), ((1, 1, 0), 2), ((1, 0, 0), 3), ((0, 0, 2), 4), ((0, 0, 1), 5), ((0, 0, 0), 6)]) == f

    R, = ring("", QQ)
    assert R.ring_new([((), 7)]) == R(7)

def test_PolyRing_drop():
    R, x,y,z = ring("x,y,z", ZZ)

    assert R.drop(x) == PolyRing("y,z", ZZ, lex)
    assert R.drop(y) == PolyRing("x,z", ZZ, lex)
    assert R.drop(z) == PolyRing("x,y", ZZ, lex)

    assert R.drop(0) == PolyRing("y,z", ZZ, lex)
    assert R.drop(0).drop(0) == PolyRing("z", ZZ, lex)
    assert R.drop(0).drop(0).drop(0) == ZZ

    assert R.drop(1) == PolyRing("x,z", ZZ, lex)

    assert R.drop(2) == PolyRing("x,y", ZZ, lex)
    assert R.drop(2).drop(1) == PolyRing("x", ZZ, lex)
    assert R.drop(2).drop(1).drop(0) == ZZ

    raises(ValueError, lambda: R.drop(3))
    raises(ValueError, lambda: R.drop(x).drop(y))

def test_PolyRing___getitem__():
    R, x,y,z = ring("x,y,z", ZZ)

    assert R[0:] == PolyRing("x,y,z", ZZ, lex)
    assert R[1:] == PolyRing("y,z", ZZ, lex)
    assert R[2:] == PolyRing("z", ZZ, lex)
    assert R[3:] == ZZ

def test_PolyRing_is_():
    R = PolyRing("x", QQ, lex)

    assert R.is_univariate is True
    assert R.is_multivariate is False

    R = PolyRing("x,y,z", QQ, lex)

    assert R.is_univariate is False
    assert R.is_multivariate is True

    R = PolyRing("", QQ, lex)

    assert R.is_univariate is False
    assert R.is_multivariate is False

def test_PolyRing_add():
    R, x = ring("x", ZZ)
    F = [ x**2 + 2*i + 3 for i in range(4) ]

    assert R.add(F) == reduce(add, F) == 4*x**2 + 24

    R, = ring("", ZZ)

    assert R.add([2, 5, 7]) == 14

def test_PolyRing_mul():
    R, x = ring("x", ZZ)
    F = [ x**2 + 2*i + 3 for i in range(4) ]

    assert R.mul(F) == reduce(mul, F) == x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945

    R, = ring("", ZZ)

    assert R.mul([2, 3, 5]) == 30

def test_PolyRing_symmetric_poly():
    R, x, y, z, t = ring("x,y,z,t", ZZ)

    raises(ValueError, lambda: R.symmetric_poly(-1))
    raises(ValueError, lambda: R.symmetric_poly(5))

    assert R.symmetric_poly(0) == R.one
    assert R.symmetric_poly(1) == x + y + z + t
    assert R.symmetric_poly(2) == x*y + x*z + x*t + y*z + y*t + z*t
    assert R.symmetric_poly(3) == x*y*z + x*y*t + x*z*t + y*z*t
    assert R.symmetric_poly(4) == x*y*z*t

def test_sring():
    x, y, z, t = symbols("x,y,z,t")

    R = PolyRing("x,y,z", ZZ, lex)
    assert sring(x + 2*y + 3*z) == (R, R.x + 2*R.y + 3*R.z)

    R = PolyRing("x,y,z", QQ, lex)
    assert sring(x + 2*y + z/3) == (R, R.x + 2*R.y + R.z/3)
    assert sring([x, 2*y, z/3]) == (R, [R.x, 2*R.y, R.z/3])

    Rt = PolyRing("t", ZZ, lex)
    R = PolyRing("x,y,z", Rt, lex)
    assert sring(x + 2*t*y + 3*t**2*z, x, y, z) == (R, R.x + 2*Rt.t*R.y + 3*Rt.t**2*R.z)

    Rt = PolyRing("t", QQ, lex)
    R = PolyRing("x,y,z", Rt, lex)
    assert sring(x + t*y/2 + t**2*z/3, x, y, z) == (R, R.x + Rt.t*R.y/2 + Rt.t**2*R.z/3)

    Rt = FracField("t", ZZ, lex)
    R = PolyRing("x,y,z", Rt, lex)
    assert sring(x + 2*y/t + t**2*z/3, x, y, z) == (R, R.x + 2*R.y/Rt.t + Rt.t**2*R.z/3)

    r = sqrt(2) - sqrt(3)
    R, a = sring(r, extension=True)
    assert R.domain == QQ.algebraic_field(sqrt(2) + sqrt(3))
    assert R.gens == ()
    assert a == R.domain.from_sympy(r)

def test_PolyElement___hash__():
    R, x, y, z = ring("x,y,z", QQ)
    assert hash(x*y*z)

def test_PolyElement___eq__():
    R, x, y = ring("x,y", ZZ, lex)

    assert ((x*y + 5*x*y) == 6) == False
    assert ((x*y + 5*x*y) == 6*x*y) == True
    assert (6 == (x*y + 5*x*y)) == False
    assert (6*x*y == (x*y + 5*x*y)) == True

    assert ((x*y - x*y) == 0) == True
    assert (0 == (x*y - x*y)) == True

    assert ((x*y - x*y) == 1) == False
    assert (1 == (x*y - x*y)) == False

    assert ((x*y - x*y) == 1) == False
    assert (1 == (x*y - x*y)) == False

    assert ((x*y + 5*x*y) != 6) == True
    assert ((x*y + 5*x*y) != 6*x*y) == False
    assert (6 != (x*y + 5*x*y)) == True
    assert (6*x*y != (x*y + 5*x*y)) == False

    assert ((x*y - x*y) != 0) == False
    assert (0 != (x*y - x*y)) == False

    assert ((x*y - x*y) != 1) == True
    assert (1 != (x*y - x*y)) == True

    assert R.one == QQ(1, 1) == R.one
    assert R.one == 1 == R.one

    Rt, t = ring("t", ZZ)
    R, x, y = ring("x,y", Rt)

    assert (t**3*x/x == t**3) == True
    assert (t**3*x/x == t**4) == False

def test_PolyElement__lt_le_gt_ge__():
    R, x, y = ring("x,y", ZZ)

    assert R(1) < x < x**2 < x**3
    assert R(1) <= x <= x**2 <= x**3

    assert x**3 > x**2 > x > R(1)
    assert x**3 >= x**2 >= x >= R(1)

def test_PolyElement__str__():
    x, y = symbols('x, y')

    for dom in [ZZ, QQ, ZZ[x], ZZ[x,y], ZZ[x][y]]:
        R, t = ring('t', dom)
        assert str(2*t**2 + 1) == '2*t**2 + 1'

    for dom in [EX, EX[x]]:
        R, t = ring('t', dom)
        assert str(2*t**2 + 1) == 'EX(2)*t**2 + EX(1)'

def test_PolyElement_copy():
    R, x, y, z = ring("x,y,z", ZZ)

    f = x*y + 3*z
    g = f.copy()

    assert f == g
    g[(1, 1, 1)] = 7
    assert f != g

def test_PolyElement_as_expr():
    R, x, y, z = ring("x,y,z", ZZ)
    f = 3*x**2*y - x*y*z + 7*z**3 + 1

    X, Y, Z = R.symbols
    g = 3*X**2*Y - X*Y*Z + 7*Z**3 + 1

    assert f != g
    assert f.as_expr() == g

    U, V, W = symbols("u,v,w")
    g = 3*U**2*V - U*V*W + 7*W**3 + 1

    assert f != g
    assert f.as_expr(U, V, W) == g

    raises(ValueError, lambda: f.as_expr(X))

    R, = ring("", ZZ)
    assert R(3).as_expr() == 3

def test_PolyElement_from_expr():
    x, y, z = symbols("x,y,z")
    R, X, Y, Z = ring((x, y, z), ZZ)

    f = R.from_expr(1)
    assert f == 1 and R.is_element(f)

    f = R.from_expr(x)
    assert f == X and R.is_element(f)

    f = R.from_expr(x*y*z)
    assert f == X*Y*Z and R.is_element(f)

    f = R.from_expr(x*y*z + x*y + x)
    assert f == X*Y*Z + X*Y + X and R.is_element(f)

    f = R.from_expr(x**3*y*z + x**2*y**7 + 1)
    assert f == X**3*Y*Z + X**2*Y**7 + 1 and R.is_element(f)

    r, F = sring([exp(2)])
    f = r.from_expr(exp(2))
    assert f == F[0] and r.is_element(f)

    raises(ValueError, lambda: R.from_expr(1/x))
    raises(ValueError, lambda: R.from_expr(2**x))
    raises(ValueError, lambda: R.from_expr(7*x + sqrt(2)))

    R, = ring("", ZZ)
    f = R.from_expr(1)
    assert f == 1 and R.is_element(f)

def test_PolyElement_degree():
    R, x,y,z = ring("x,y,z", ZZ)

    assert ninf == float('-inf')

    assert R(0).degree() is ninf
    assert R(1).degree() == 0
    assert (x + 1).degree() == 1
    assert (2*y**3 + z).degree() == 0
    assert (x*y**3 + z).degree() == 1
    assert (x**5*y**3 + z).degree() == 5

    assert R(0).degree(x) is ninf
    assert R(1).degree(x) == 0
    assert (x + 1).degree(x) == 1
    assert (2*y**3 + z).degree(x) == 0
    assert (x*y**3 + z).degree(x) == 1
    assert (7*x**5*y**3 + z).degree(x) == 5

    assert R(0).degree(y) is ninf
    assert R(1).degree(y) == 0
    assert (x + 1).degree(y) == 0
    assert (2*y**3 + z).degree(y) == 3
    assert (x*y**3 + z).degree(y) == 3
    assert (7*x**5*y**3 + z).degree(y) == 3

    assert R(0).degree(z) is ninf
    assert R(1).degree(z) == 0
    assert (x + 1).degree(z) == 0
    assert (2*y**3 + z).degree(z) == 1
    assert (x*y**3 + z).degree(z) == 1
    assert (7*x**5*y**3 + z).degree(z) == 1

    R, = ring("", ZZ)
    assert R(0).degree() is ninf
    assert R(1).degree() == 0

def test_PolyElement_tail_degree():
    R, x,y,z = ring("x,y,z", ZZ)

    assert R(0).tail_degree() is ninf
    assert R(1).tail_degree() == 0
    assert (x + 1).tail_degree() == 0
    assert (2*y**3 + x**3*z).tail_degree() == 0
    assert (x*y**3 + x**3*z).tail_degree() == 1
    assert (x**5*y**3 + x**3*z).tail_degree() == 3

    assert R(0).tail_degree(x) is ninf
    assert R(1).tail_degree(x) == 0
    assert (x + 1).tail_degree(x) == 0
    assert (2*y**3 + x**3*z).tail_degree(x) == 0
    assert (x*y**3 + x**3*z).tail_degree(x) == 1
    assert (7*x**5*y**3 + x**3*z).tail_degree(x) == 3

    assert R(0).tail_degree(y) is ninf
    assert R(1).tail_degree(y) == 0
    assert (x + 1).tail_degree(y) == 0
    assert (2*y**3 + x**3*z).tail_degree(y) == 0
    assert (x*y**3 + x**3*z).tail_degree(y) == 0
    assert (7*x**5*y**3 + x**3*z).tail_degree(y) == 0

    assert R(0).tail_degree(z) is ninf
    assert R(1).tail_degree(z) == 0
    assert (x + 1).tail_degree(z) == 0
    assert (2*y**3 + x**3*z).tail_degree(z) == 0
    assert (x*y**3 + x**3*z).tail_degree(z) == 0
    assert (7*x**5*y**3 + x**3*z).tail_degree(z) == 0

    R, = ring("", ZZ)
    assert R(0).tail_degree() is ninf
    assert R(1).tail_degree() == 0

def test_PolyElement_degrees():
    R, x,y,z = ring("x,y,z", ZZ)

    assert R(0).degrees() == (ninf, ninf, ninf)
    assert R(1).degrees() == (0, 0, 0)
    assert (x**2*y + x**3*z**2).degrees() == (3, 1, 2)

def test_PolyElement_tail_degrees():
    R, x,y,z = ring("x,y,z", ZZ)

    assert R(0).tail_degrees() == (ninf, ninf, ninf)
    assert R(1).tail_degrees() == (0, 0, 0)
    assert (x**2*y + x**3*z**2).tail_degrees() == (2, 0, 0)

def test_PolyElement_coeff():
    R, x, y, z = ring("x,y,z", ZZ, lex)
    f = 3*x**2*y - x*y*z + 7*z**3 + 23

    assert f.coeff(1) == 23
    raises(ValueError, lambda: f.coeff(3))

    assert f.coeff(x) == 0
    assert f.coeff(y) == 0
    assert f.coeff(z) == 0

    assert f.coeff(x**2*y) == 3
    assert f.coeff(x*y*z) == -1
    assert f.coeff(z**3) == 7

    raises(ValueError, lambda: f.coeff(3*x**2*y))
    raises(ValueError, lambda: f.coeff(-x*y*z))
    raises(ValueError, lambda: f.coeff(7*z**3))

    R, = ring("", ZZ)
    assert R(3).coeff(1) == 3

def test_PolyElement_LC():
    R, x, y = ring("x,y", QQ, lex)
    assert R(0).LC == QQ(0)
    assert (QQ(1,2)*x).LC == QQ(1, 2)
    assert (QQ(1,4)*x*y + QQ(1,2)*x).LC == QQ(1, 4)

def test_PolyElement_LM():
    R, x, y = ring("x,y", QQ, lex)
    assert R(0).LM == (0, 0)
    assert (QQ(1,2)*x).LM == (1, 0)
    assert (QQ(1,4)*x*y + QQ(1,2)*x).LM == (1, 1)

def test_PolyElement_LT():
    R, x, y = ring("x,y", QQ, lex)
    assert R(0).LT == ((0, 0), QQ(0))
    assert (QQ(1,2)*x).LT == ((1, 0), QQ(1, 2))
    assert (QQ(1,4)*x*y + QQ(1,2)*x).LT == ((1, 1), QQ(1, 4))

    R, = ring("", ZZ)
    assert R(0).LT == ((), 0)
    assert R(1).LT == ((), 1)

def test_PolyElement_leading_monom():
    R, x, y = ring("x,y", QQ, lex)
    assert R(0).leading_monom() == 0
    assert (QQ(1,2)*x).leading_monom() == x
    assert (QQ(1,4)*x*y + QQ(1,2)*x).leading_monom() == x*y

def test_PolyElement_leading_term():
    R, x, y = ring("x,y", QQ, lex)
    assert R(0).leading_term() == 0
    assert (QQ(1,2)*x).leading_term() == QQ(1,2)*x
    assert (QQ(1,4)*x*y + QQ(1,2)*x).leading_term() == QQ(1,4)*x*y

def test_PolyElement_terms():
    R, x,y,z = ring("x,y,z", QQ)
    terms = (x**2/3 + y**3/4 + z**4/5).terms()
    assert terms == [((2,0,0), QQ(1,3)), ((0,3,0), QQ(1,4)), ((0,0,4), QQ(1,5))]

    R, x,y = ring("x,y", ZZ, lex)
    f = x*y**7 + 2*x**2*y**3

    assert f.terms() == f.terms(lex) == f.terms('lex') == [((2, 3), 2), ((1, 7), 1)]
    assert f.terms(grlex) == f.terms('grlex') == [((1, 7), 1), ((2, 3), 2)]

    R, x,y = ring("x,y", ZZ, grlex)
    f = x*y**7 + 2*x**2*y**3

    assert f.terms() == f.terms(grlex) == f.terms('grlex') == [((1, 7), 1), ((2, 3), 2)]
    assert f.terms(lex) == f.terms('lex') == [((2, 3), 2), ((1, 7), 1)]

    R, = ring("", ZZ)
    assert R(3).terms() == [((), 3)]

def test_PolyElement_monoms():
    R, x,y,z = ring("x,y,z", QQ)
    monoms = (x**2/3 + y**3/4 + z**4/5).monoms()
    assert monoms == [(2,0,0), (0,3,0), (0,0,4)]

    R, x,y = ring("x,y", ZZ, lex)
    f = x*y**7 + 2*x**2*y**3

    assert f.monoms() == f.monoms(lex) == f.monoms('lex') == [(2, 3), (1, 7)]
    assert f.monoms(grlex) == f.monoms('grlex') == [(1, 7), (2, 3)]

    R, x,y = ring("x,y", ZZ, grlex)
    f = x*y**7 + 2*x**2*y**3

    assert f.monoms() == f.monoms(grlex) == f.monoms('grlex') == [(1, 7), (2, 3)]
    assert f.monoms(lex) == f.monoms('lex') == [(2, 3), (1, 7)]

def test_PolyElement_coeffs():
    R, x,y,z = ring("x,y,z", QQ)
    coeffs = (x**2/3 + y**3/4 + z**4/5).coeffs()
    assert coeffs == [QQ(1,3), QQ(1,4), QQ(1,5)]

    R, x,y = ring("x,y", ZZ, lex)
    f = x*y**7 + 2*x**2*y**3

    assert f.coeffs() == f.coeffs(lex) == f.coeffs('lex') == [2, 1]
    assert f.coeffs(grlex) == f.coeffs('grlex') == [1, 2]

    R, x,y = ring("x,y", ZZ, grlex)
    f = x*y**7 + 2*x**2*y**3

    assert f.coeffs() == f.coeffs(grlex) == f.coeffs('grlex') == [1, 2]
    assert f.coeffs(lex) == f.coeffs('lex') == [2, 1]

def test_PolyElement___add__():
    Rt, t = ring("t", ZZ)
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)

    assert dict(x + 3*y) == {(1, 0, 0): 1, (0, 1, 0): 3}

    assert dict(u + x) == dict(x + u) == {(1, 0, 0): 1, (0, 0, 0): u}
    assert dict(u + x*y) == dict(x*y + u) == {(1, 1, 0): 1, (0, 0, 0): u}
    assert dict(u + x*y + z) == dict(x*y + z + u) == {(1, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): u}

    assert dict(u*x + x) == dict(x + u*x) == {(1, 0, 0): u + 1}
    assert dict(u*x + x*y) == dict(x*y + u*x) == {(1, 1, 0): 1, (1, 0, 0): u}
    assert dict(u*x + x*y + z) == dict(x*y + z + u*x) == {(1, 1, 0): 1, (0, 0, 1): 1, (1, 0, 0): u}

    raises(TypeError, lambda: t + x)
    raises(TypeError, lambda: x + t)
    raises(TypeError, lambda: t + u)
    raises(TypeError, lambda: u + t)

    Fuv, u,v = field("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Fuv)

    assert dict(u + x) == dict(x + u) == {(1, 0, 0): 1, (0, 0, 0): u}

    Rxyz, x,y,z = ring("x,y,z", EX)

    assert dict(EX(pi) + x*y*z) == dict(x*y*z + EX(pi)) == {(1, 1, 1): EX(1), (0, 0, 0): EX(pi)}

def test_PolyElement___sub__():
    Rt, t = ring("t", ZZ)
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)

    assert dict(x - 3*y) == {(1, 0, 0): 1, (0, 1, 0): -3}

    assert dict(-u + x) == dict(x - u) == {(1, 0, 0): 1, (0, 0, 0): -u}
    assert dict(-u + x*y) == dict(x*y - u) == {(1, 1, 0): 1, (0, 0, 0): -u}
    assert dict(-u + x*y + z) == dict(x*y + z - u) == {(1, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): -u}

    assert dict(-u*x + x) == dict(x - u*x) == {(1, 0, 0): -u + 1}
    assert dict(-u*x + x*y) == dict(x*y - u*x) == {(1, 1, 0): 1, (1, 0, 0): -u}
    assert dict(-u*x + x*y + z) == dict(x*y + z - u*x) == {(1, 1, 0): 1, (0, 0, 1): 1, (1, 0, 0): -u}

    raises(TypeError, lambda: t - x)
    raises(TypeError, lambda: x - t)
    raises(TypeError, lambda: t - u)
    raises(TypeError, lambda: u - t)

    Fuv, u,v = field("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Fuv)

    assert dict(-u + x) == dict(x - u) == {(1, 0, 0): 1, (0, 0, 0): -u}

    Rxyz, x,y,z = ring("x,y,z", EX)

    assert dict(-EX(pi) + x*y*z) == dict(x*y*z - EX(pi)) == {(1, 1, 1): EX(1), (0, 0, 0): -EX(pi)}

def test_PolyElement___mul__():
    Rt, t = ring("t", ZZ)
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)

    assert dict(u*x) == dict(x*u) == {(1, 0, 0): u}

    assert dict(2*u*x + z) == dict(x*2*u + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*2*x + z) == dict(2*x*u + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*x + z) == dict(x*2*u + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*x*2 + z) == dict(x*u*2 + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}

    assert dict(2*u*x*y + z) == dict(x*y*2*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*2*x*y + z) == dict(2*x*y*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*x*y + z) == dict(x*y*2*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*x*y*2 + z) == dict(x*y*u*2 + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}

    assert dict(2*u*y*x + z) == dict(y*x*2*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*2*y*x + z) == dict(2*y*x*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*y*x + z) == dict(y*x*2*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*y*x*2 + z) == dict(y*x*u*2 + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}

    assert dict(3*u*(x + y) + z) == dict((x + y)*3*u + z) == {(1, 0, 0): 3*u, (0, 1, 0): 3*u, (0, 0, 1): 1}

    raises(TypeError, lambda: t*x + z)
    raises(TypeError, lambda: x*t + z)
    raises(TypeError, lambda: t*u + z)
    raises(TypeError, lambda: u*t + z)

    Fuv, u,v = field("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Fuv)

    assert dict(u*x) == dict(x*u) == {(1, 0, 0): u}

    Rxyz, x,y,z = ring("x,y,z", EX)

    assert dict(EX(pi)*x*y*z) == dict(x*y*z*EX(pi)) == {(1, 1, 1): EX(pi)}

def test_PolyElement___truediv__():
    R, x,y,z = ring("x,y,z", ZZ)

    assert (2*x**2 - 4)/2 == x**2 - 2
    assert (2*x**2 - 3)/2 == x**2

    assert (x**2 - 1).quo(x) == x
    assert (x**2 - x).quo(x) == x - 1

    raises(ExactQuotientFailed, lambda: (x**2 - 1)/x)
    assert (x**2 - x)/x == x - 1
    raises(ExactQuotientFailed, lambda: (x**2 - 1)/(2*x))

    assert (x**2 - 1).quo(2*x) == 0
    assert (x**2 - x)/(x - 1) == (x**2 - x).quo(x - 1) == x


    R, x,y,z = ring("x,y,z", ZZ)
    assert len((x**2/3 + y**3/4 + z**4/5).terms()) == 0

    R, x,y,z = ring("x,y,z", QQ)
    assert len((x**2/3 + y**3/4 + z**4/5).terms()) == 3

    Rt, t = ring("t", ZZ)
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)

    assert dict((u**2*x + u)/u) == {(1, 0, 0): u, (0, 0, 0): 1}
    raises(ExactQuotientFailed, lambda: u/(u**2*x + u))

    raises(TypeError, lambda: t/x)
    raises(TypeError, lambda: x/t)
    raises(TypeError, lambda: t/u)
    raises(TypeError, lambda: u/t)

    R, x = ring("x", ZZ)
    f, g = x**2 + 2*x + 3, R(0)

    raises(ZeroDivisionError, lambda: f.div(g))
    raises(ZeroDivisionError, lambda: divmod(f, g))
    raises(ZeroDivisionError, lambda: f.rem(g))
    raises(ZeroDivisionError, lambda: f % g)
    raises(ZeroDivisionError, lambda: f.quo(g))
    raises(ZeroDivisionError, lambda: f / g)
    raises(ZeroDivisionError, lambda: f.exquo(g))

    R, x, y = ring("x,y", ZZ)
    f, g = x*y + 2*x + 3, R(0)

    raises(ZeroDivisionError, lambda: f.div(g))
    raises(ZeroDivisionError, lambda: divmod(f, g))
    raises(ZeroDivisionError, lambda: f.rem(g))
    raises(ZeroDivisionError, lambda: f % g)
    raises(ZeroDivisionError, lambda: f.quo(g))
    raises(ZeroDivisionError, lambda: f / g)
    raises(ZeroDivisionError, lambda: f.exquo(g))

    R, x = ring("x", ZZ)

    f, g = x**2 + 1, 2*x - 4
    q, r = R(0), x**2 + 1

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = 3*x**3 + x**2 + x + 5, 5*x**2 - 3*x + 1
    q, r = R(0), f

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1, x**2 + 2*x + 3
    q, r = 5*x**2 - 6*x, 20*x + 1

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = 5*x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x, x**4 + 2*x**3 + 9
    q, r = 5*x - 6, 15*x**3 + 2*x**2 - 44*x + 54

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    R, x = ring("x", QQ)

    f, g = x**2 + 1, 2*x - 4
    q, r = x/2 + 1, R(5)

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = 3*x**3 + x**2 + x + 5, 5*x**2 - 3*x + 1
    q, r = QQ(3, 5)*x + QQ(14, 25), QQ(52, 25)*x + QQ(111, 25)

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    R, x,y = ring("x,y", ZZ)

    f, g = x**2 - y**2, x - y
    q, r = x + y, R(0)

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    assert f.exquo(g) == f / g == q

    f, g = x**2 + y**2, x - y
    q, r = x + y, 2*y**2

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = x**2 + y**2, -x + y
    q, r = -x - y, 2*y**2

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = x**2 + y**2, 2*x - 2*y
    q, r = R(0), f

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    R, x,y = ring("x,y", QQ)

    f, g = x**2 - y**2, x - y
    q, r = x + y, R(0)

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    assert f.exquo(g) == f / g == q

    f, g = x**2 + y**2, x - y
    q, r = x + y, 2*y**2

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = x**2 + y**2, -x + y
    q, r = -x - y, 2*y**2

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    f, g = x**2 + y**2, 2*x - 2*y
    q, r = x/2 + y/2, 2*y**2

    assert f.div(g) == divmod(f, g) == (q, r)
    assert f.rem(g) == f % g == r
    assert f.quo(g) == q
    raises(ExactQuotientFailed, lambda: f / g)
    raises(ExactQuotientFailed, lambda: f.exquo(g))

def test_PolyElement___pow__():
    R, x = ring("x", ZZ, grlex)
    f = 2*x + 3

    assert f**0 == 1
    assert f**1 == f
    raises(ValueError, lambda: f**(-1))

    assert f**2 == f._pow_generic(2) == f._pow_multinomial(2) == 4*x**2 + 12*x + 9
    assert f**3 == f._pow_generic(3) == f._pow_multinomial(3) == 8*x**3 + 36*x**2 + 54*x + 27
    assert f**4 == f._pow_generic(4) == f._pow_multinomial(4) == 16*x**4 + 96*x**3 + 216*x**2 + 216*x + 81
    assert f**5 == f._pow_generic(5) == f._pow_multinomial(5) == 32*x**5 + 240*x**4 + 720*x**3 + 1080*x**2 + 810*x + 243

    R, x,y,z = ring("x,y,z", ZZ, grlex)
    f = x**3*y - 2*x*y**2 - 3*z + 1
    g = x**6*y**2 - 4*x**4*y**3 - 6*x**3*y*z + 2*x**3*y + 4*x**2*y**4 + 12*x*y**2*z - 4*x*y**2 + 9*z**2 - 6*z + 1

    assert f**2 == f._pow_generic(2) == f._pow_multinomial(2) == g

    R, t = ring("t", ZZ)
    f = -11200*t**4 - 2604*t**2 + 49
    g = 15735193600000000*t**16 + 14633730048000000*t**14 + 4828147466240000*t**12 \
      + 598976863027200*t**10 + 3130812416256*t**8 - 2620523775744*t**6 \
      + 92413760096*t**4 - 1225431984*t**2 + 5764801

    assert f**4 == f._pow_generic(4) == f._pow_multinomial(4) == g

def test_PolyElement_div():
    R, x = ring("x", ZZ, grlex)

    f = x**3 - 12*x**2 - 42
    g = x - 3

    q = x**2 - 9*x - 27
    r = -123

    assert f.div([g]) == ([q], r)

    R, x = ring("x", ZZ, grlex)
    f = x**2 + 2*x + 2
    assert f.div([R(1)]) == ([f], 0)

    R, x = ring("x", QQ, grlex)
    f = x**2 + 2*x + 2
    assert f.div([R(2)]) == ([QQ(1,2)*x**2 + x + 1], 0)

    R, x,y = ring("x,y", ZZ, grlex)
    f = 4*x**2*y - 2*x*y + 4*x - 2*y + 8

    assert f.div([R(2)]) == ([2*x**2*y - x*y + 2*x - y + 4], 0)
    assert f.div([2*y]) == ([2*x**2 - x - 1], 4*x + 8)

    f = x - 1
    g = y - 1

    assert f.div([g]) == ([0], f)

    f = x*y**2 + 1
    G = [x*y + 1, y + 1]

    Q = [y, -1]
    r = 2

    assert f.div(G) == (Q, r)

    f = x**2*y + x*y**2 + y**2
    G = [x*y - 1, y**2 - 1]

    Q = [x + y, 1]
    r = x + y + 1

    assert f.div(G) == (Q, r)

    G = [y**2 - 1, x*y - 1]

    Q = [x + 1, x]
    r = 2*x + 1

    assert f.div(G) == (Q, r)

    R, = ring("", ZZ)
    assert R(3).div(R(2)) == (0, 3)

    R, = ring("", QQ)
    assert R(3).div(R(2)) == (QQ(3, 2), 0)

def test_PolyElement_rem():
    R, x = ring("x", ZZ, grlex)

    f = x**3 - 12*x**2 - 42
    g = x - 3
    r = -123

    assert f.rem([g]) == f.div([g])[1] == r

    R, x,y = ring("x,y", ZZ, grlex)

    f = 4*x**2*y - 2*x*y + 4*x - 2*y + 8

    assert f.rem([R(2)]) == f.div([R(2)])[1] == 0
    assert f.rem([2*y]) == f.div([2*y])[1] == 4*x + 8

    f = x - 1
    g = y - 1

    assert f.rem([g]) == f.div([g])[1] == f

    f = x*y**2 + 1
    G = [x*y + 1, y + 1]
    r = 2

    assert f.rem(G) == f.div(G)[1] == r

    f = x**2*y + x*y**2 + y**2
    G = [x*y - 1, y**2 - 1]
    r = x + y + 1

    assert f.rem(G) == f.div(G)[1] == r

    G = [y**2 - 1, x*y - 1]
    r = 2*x + 1

    assert f.rem(G) == f.div(G)[1] == r

def test_PolyElement_deflate():
    R, x = ring("x", ZZ)

    assert (2*x**2).deflate(x**4 + 4*x**2 + 1) == ((2,), [2*x, x**2 + 4*x + 1])

    R, x,y = ring("x,y", ZZ)

    assert R(0).deflate(R(0)) == ((1, 1), [0, 0])
    assert R(1).deflate(R(0)) == ((1, 1), [1, 0])
    assert R(1).deflate(R(2)) == ((1, 1), [1, 2])
    assert R(1).deflate(2*y) == ((1, 1), [1, 2*y])
    assert (2*y).deflate(2*y) == ((1, 1), [2*y, 2*y])
    assert R(2).deflate(2*y**2) == ((1, 2), [2, 2*y])
    assert (2*y**2).deflate(2*y**2) == ((1, 2), [2*y, 2*y])

    f = x**4*y**2 + x**2*y + 1
    g = x**2*y**3 + x**2*y + 1

    assert f.deflate(g) == ((2, 1), [x**2*y**2 + x*y + 1, x*y**3 + x*y + 1])

def test_PolyElement_clear_denoms():
    R, x,y = ring("x,y", QQ)

    assert R(1).clear_denoms() == (ZZ(1), 1)
    assert R(7).clear_denoms() == (ZZ(1), 7)

    assert R(QQ(7,3)).clear_denoms() == (3, 7)
    assert R(QQ(7,3)).clear_denoms() == (3, 7)

    assert (3*x**2 + x).clear_denoms() == (1, 3*x**2 + x)
    assert (x**2 + QQ(1,2)*x).clear_denoms() == (2, 2*x**2 + x)

    rQQ, x,t = ring("x,t", QQ, lex)
    rZZ, X,T = ring("x,t", ZZ, lex)

    F = [x - QQ(17824537287975195925064602467992950991718052713078834557692023531499318507213727406844943097,413954288007559433755329699713866804710749652268151059918115348815925474842910720000)*t**7
           - QQ(4882321164854282623427463828745855894130208215961904469205260756604820743234704900167747753,12936071500236232304854053116058337647210926633379720622441104650497671088840960000)*t**6
           - QQ(36398103304520066098365558157422127347455927422509913596393052633155821154626830576085097433,25872143000472464609708106232116675294421853266759441244882209300995342177681920000)*t**5
           - QQ(168108082231614049052707339295479262031324376786405372698857619250210703675982492356828810819,58212321751063045371843239022262519412449169850208742800984970927239519899784320000)*t**4
           - QQ(5694176899498574510667890423110567593477487855183144378347226247962949388653159751849449037,1617008937529529038106756639507292205901365829172465077805138081312208886105120000)*t**3
           - QQ(154482622347268833757819824809033388503591365487934245386958884099214649755244381307907779,60637835157357338929003373981523457721301218593967440417692678049207833228942000)*t**2
           - QQ(2452813096069528207645703151222478123259511586701148682951852876484544822947007791153163,2425513406294293557160134959260938308852048743758697616707707121968313329157680)*t
           - QQ(34305265428126440542854669008203683099323146152358231964773310260498715579162112959703,202126117191191129763344579938411525737670728646558134725642260164026110763140),
         t**8 + QQ(693749860237914515552,67859264524169150569)*t**7
              + QQ(27761407182086143225024,610733380717522355121)*t**6
              + QQ(7785127652157884044288,67859264524169150569)*t**5
              + QQ(36567075214771261409792,203577793572507451707)*t**4
              + QQ(36336335165196147384320,203577793572507451707)*t**3
              + QQ(7452455676042754048000,67859264524169150569)*t**2
              + QQ(2593331082514399232000,67859264524169150569)*t
              + QQ(390399197427343360000,67859264524169150569)]

    G = [3725588592068034903797967297424801242396746870413359539263038139343329273586196480000*X -
         160420835591776763325581422211936558925462474417709511019228211783493866564923546661604487873*T**7 -
         1406108495478033395547109582678806497509499966197028487131115097902188374051595011248311352864*T**6 -
         5241326875850889518164640374668786338033653548841427557880599579174438246266263602956254030352*T**5 -
         10758917262823299139373269714910672770004760114329943852726887632013485035262879510837043892416*T**4 -
         13119383576444715672578819534846747735372132018341964647712009275306635391456880068261130581248*T**3 -
         9491412317016197146080450036267011389660653495578680036574753839055748080962214787557853941760*T**2 -
         3767520915562795326943800040277726397326609797172964377014046018280260848046603967211258368000*T -
         632314652371226552085897259159210286886724229880266931574701654721512325555116066073245696000,
         610733380717522355121*T**8 +
         6243748742141230639968*T**7 +
         27761407182086143225024*T**6 +
         70066148869420956398592*T**5 +
         109701225644313784229376*T**4 +
         109009005495588442152960*T**3 +
         67072101084384786432000*T**2 +
         23339979742629593088000*T +
         3513592776846090240000]

    assert [ f.clear_denoms()[1].set_ring(rZZ) for f in F ] == G

def test_PolyElement_cofactors():
    R, x, y = ring("x,y", ZZ)

    f, g = R(0), R(0)
    assert f.cofactors(g) == (0, 0, 0)

    f, g = R(2), R(0)
    assert f.cofactors(g) == (2, 1, 0)

    f, g = R(-2), R(0)
    assert f.cofactors(g) == (2, -1, 0)

    f, g = R(0), R(-2)
    assert f.cofactors(g) == (2, 0, -1)

    f, g = R(0), 2*x + 4
    assert f.cofactors(g) == (2*x + 4, 0, 1)

    f, g = 2*x + 4, R(0)
    assert f.cofactors(g) == (2*x + 4, 1, 0)

    f, g = R(2), R(2)
    assert f.cofactors(g) == (2, 1, 1)

    f, g = R(-2), R(2)
    assert f.cofactors(g) == (2, -1, 1)

    f, g = R(2), R(-2)
    assert f.cofactors(g) == (2, 1, -1)

    f, g = R(-2), R(-2)
    assert f.cofactors(g) == (2, -1, -1)

    f, g = x**2 + 2*x + 1, R(1)
    assert f.cofactors(g) == (1, x**2 + 2*x + 1, 1)

    f, g = x**2 + 2*x + 1, R(2)
    assert f.cofactors(g) == (1, x**2 + 2*x + 1, 2)

    f, g = 2*x**2 + 4*x + 2, R(2)
    assert f.cofactors(g) == (2, x**2 + 2*x + 1, 1)

    f, g = R(2), 2*x**2 + 4*x + 2
    assert f.cofactors(g) == (2, 1, x**2 + 2*x + 1)

    f, g = 2*x**2 + 4*x + 2, x + 1
    assert f.cofactors(g) == (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2
    assert f.cofactors(g) == (x + 1, 1, 2*x + 2)

    R, x, y, z, t = ring("x,y,z,t", ZZ)

    f, g = t**2 + 2*t + 1, 2*t + 2
    assert f.cofactors(g) == (t + 1, t + 1, 2)

    f, g = z**2*t**2 + 2*z**2*t + z**2 + z*t + z, t**2 + 2*t + 1
    h, cff, cfg = t + 1, z**2*t + z**2 + z, t + 1

    assert f.cofactors(g) == (h, cff, cfg)
    assert g.cofactors(f) == (h, cfg, cff)

    R, x, y = ring("x,y", QQ)

    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)

    h = x + 1

    assert f.cofactors(g) == (h, g, QQ(1,2))
    assert g.cofactors(f) == (h, QQ(1,2), g)

    R, x, y = ring("x,y", RR)

    f = 2.1*x*y**2 - 2.1*x*y + 2.1*x
    g = 2.1*x**3
    h = 1.0*x

    assert f.cofactors(g) == (h, f/h, g/h)
    assert g.cofactors(f) == (h, g/h, f/h)

def test_PolyElement_gcd():
    R, x, y = ring("x,y", QQ)

    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)

    assert f.gcd(g) == x + 1

def test_PolyElement_cancel():
    R, x, y = ring("x,y", ZZ)

    f = 2*x**3 + 4*x**2 + 2*x
    g = 3*x**2 + 3*x
    F = 2*x + 2
    G = 3

    assert f.cancel(g) == (F, G)

    assert (-f).cancel(g) == (-F, G)
    assert f.cancel(-g) == (-F, G)

    R, x, y = ring("x,y", QQ)

    f = QQ(1,2)*x**3 + x**2 + QQ(1,2)*x
    g = QQ(1,3)*x**2 + QQ(1,3)*x
    F = 3*x + 3
    G = 2

    assert f.cancel(g) == (F, G)

    assert (-f).cancel(g) == (-F, G)
    assert f.cancel(-g) == (-F, G)

    Fx, x = field("x", ZZ)
    Rt, t = ring("t", Fx)

    f = (-x**2 - 4)/4*t
    g = t**2 + (x**2 + 2)/2

    assert f.cancel(g) == ((-x**2 - 4)*t, 4*t**2 + 2*x**2 + 4)

def test_PolyElement_max_norm():
    R, x, y = ring("x,y", ZZ)

    assert R(0).max_norm() == 0
    assert R(1).max_norm() == 1

    assert (x**3 + 4*x**2 + 2*x + 3).max_norm() == 4

def test_PolyElement_l1_norm():
    R, x, y = ring("x,y", ZZ)

    assert R(0).l1_norm() == 0
    assert R(1).l1_norm() == 1

    assert (x**3 + 4*x**2 + 2*x + 3).l1_norm() == 10

def test_PolyElement_diff():
    R, X = xring("x:11", QQ)

    f = QQ(288,5)*X[0]**8*X[1]**6*X[4]**3*X[10]**2 + 8*X[0]**2*X[2]**3*X[4]**3 +2*X[0]**2 - 2*X[1]**2

    assert f.diff(X[0]) == QQ(2304,5)*X[0]**7*X[1]**6*X[4]**3*X[10]**2 + 16*X[0]*X[2]**3*X[4]**3 + 4*X[0]
    assert f.diff(X[4]) == QQ(864,5)*X[0]**8*X[1]**6*X[4]**2*X[10]**2 + 24*X[0]**2*X[2]**3*X[4]**2
    assert f.diff(X[10]) == QQ(576,5)*X[0]**8*X[1]**6*X[4]**3*X[10]

def test_PolyElement___call__():
    R, x = ring("x", ZZ)
    f = 3*x + 1

    assert f(0) == 1
    assert f(1) == 4

    raises(ValueError, lambda: f())
    raises(ValueError, lambda: f(0, 1))

    raises(CoercionFailed, lambda: f(QQ(1,7)))

    R, x,y = ring("x,y", ZZ)
    f = 3*x + y**2 + 1

    assert f(0, 0) == 1
    assert f(1, 7) == 53

    Ry = R.drop(x)

    assert f(0) == Ry.y**2 + 1
    assert f(1) == Ry.y**2 + 4

    raises(ValueError, lambda: f())
    raises(ValueError, lambda: f(0, 1, 2))

    raises(CoercionFailed, lambda: f(1, QQ(1,7)))
    raises(CoercionFailed, lambda: f(QQ(1,7), 1))
    raises(CoercionFailed, lambda: f(QQ(1,7), QQ(1,7)))

def test_PolyElement_evaluate():
    R, x = ring("x", ZZ)
    f = x**3 + 4*x**2 + 2*x + 3

    r = f.evaluate(x, 0)
    assert r == 3 and not isinstance(r, PolyElement)

    raises(CoercionFailed, lambda: f.evaluate(x, QQ(1,7)))

    R, x, y, z = ring("x,y,z", ZZ)
    f = (x*y)**3 + 4*(x*y)**2 + 2*x*y + 3

    r = f.evaluate(x, 0)
    assert r == 3 and R.drop(x).is_element(r)
    r = f.evaluate([(x, 0), (y, 0)])
    assert r == 3 and R.drop(x, y).is_element(r)
    r = f.evaluate(y, 0)
    assert r == 3 and R.drop(y).is_element(r)
    r = f.evaluate([(y, 0), (x, 0)])
    assert r == 3 and R.drop(y, x).is_element(r)

    r = f.evaluate([(x, 0), (y, 0), (z, 0)])
    assert r == 3 and not isinstance(r, PolyElement)

    raises(CoercionFailed, lambda: f.evaluate([(x, 1), (y, QQ(1,7))]))
    raises(CoercionFailed, lambda: f.evaluate([(x, QQ(1,7)), (y, 1)]))
    raises(CoercionFailed, lambda: f.evaluate([(x, QQ(1,7)), (y, QQ(1,7))]))

def test_PolyElement_subs():
    R, x = ring("x", ZZ)
    f = x**3 + 4*x**2 + 2*x + 3

    r = f.subs(x, 0)
    assert r == 3 and R.is_element(r)

    raises(CoercionFailed, lambda: f.subs(x, QQ(1,7)))

    R, x, y, z = ring("x,y,z", ZZ)
    f = x**3 + 4*x**2 + 2*x + 3

    r = f.subs(x, 0)
    assert r == 3 and R.is_element(r)
    r = f.subs([(x, 0), (y, 0)])
    assert r == 3 and R.is_element(r)

    raises(CoercionFailed, lambda: f.subs([(x, 1), (y, QQ(1,7))]))
    raises(CoercionFailed, lambda: f.subs([(x, QQ(1,7)), (y, 1)]))
    raises(CoercionFailed, lambda: f.subs([(x, QQ(1,7)), (y, QQ(1,7))]))

def test_PolyElement_symmetrize():
    R, x, y = ring("x,y", ZZ)

    # Homogeneous, symmetric
    f = x**2 + y**2
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f

    # Homogeneous, asymmetric
    f = x**2 - y**2
    sym, rem, m = f.symmetrize()
    assert rem != 0
    assert sym.compose(m) + rem == f

    # Inhomogeneous, symmetric
    f = x*y + 7
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f

    # Inhomogeneous, asymmetric
    f = y + 7
    sym, rem, m = f.symmetrize()
    assert rem != 0
    assert sym.compose(m) + rem == f

    # Constant
    f = R.from_expr(3)
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f

    # Constant constructed from sring
    R, f = sring(3)
    sym, rem, m = f.symmetrize()
    assert rem == 0
    assert sym.compose(m) + rem == f

def test_PolyElement_compose():
    R, x = ring("x", ZZ)
    f = x**3 + 4*x**2 + 2*x + 3

    r = f.compose(x, 0)
    assert r == 3 and R.is_element(r)

    assert f.compose(x, x) == f
    assert f.compose(x, x**2) == x**6 + 4*x**4 + 2*x**2 + 3

    raises(CoercionFailed, lambda: f.compose(x, QQ(1,7)))

    R, x, y, z = ring("x,y,z", ZZ)
    f = x**3 + 4*x**2 + 2*x + 3

    r = f.compose(x, 0)
    assert r == 3 and R.is_element(r)
    r = f.compose([(x, 0), (y, 0)])
    assert r == 3 and R.is_element(r)

    r = (x**3 + 4*x**2 + 2*x*y*z + 3).compose(x, y*z**2 - 1)
    q = (y*z**2 - 1)**3 + 4*(y*z**2 - 1)**2 + 2*(y*z**2 - 1)*y*z + 3
    assert r == q and R.is_element(r)

def test_PolyElement_is_():
    R, x,y,z = ring("x,y,z", QQ)

    assert (x - x).is_generator == False
    assert (x - x).is_ground == True
    assert (x - x).is_monomial == True
    assert (x - x).is_term == True

    assert (x - x + 1).is_generator == False
    assert (x - x + 1).is_ground == True
    assert (x - x + 1).is_monomial == True
    assert (x - x + 1).is_term == True

    assert x.is_generator == True
    assert x.is_ground == False
    assert x.is_monomial == True
    assert x.is_term == True

    assert (x*y).is_generator == False
    assert (x*y).is_ground == False
    assert (x*y).is_monomial == True
    assert (x*y).is_term == True

    assert (3*x).is_generator == False
    assert (3*x).is_ground == False
    assert (3*x).is_monomial == False
    assert (3*x).is_term == True

    assert (3*x + 1).is_generator == False
    assert (3*x + 1).is_ground == False
    assert (3*x + 1).is_monomial == False
    assert (3*x + 1).is_term == False

    assert R(0).is_zero is True
    assert R(1).is_zero is False

    assert R(0).is_one is False
    assert R(1).is_one is True

    assert (x - 1).is_monic is True
    assert (2*x - 1).is_monic is False

    assert (3*x + 2).is_primitive is True
    assert (4*x + 2).is_primitive is False

    assert (x + y + z + 1).is_linear is True
    assert (x*y*z + 1).is_linear is False

    assert (x*y + z + 1).is_quadratic is True
    assert (x*y*z + 1).is_quadratic is False

    assert (x - 1).is_squarefree is True
    assert ((x - 1)**2).is_squarefree is False

    assert (x**2 + x + 1).is_irreducible is True
    assert (x**2 + 2*x + 1).is_irreducible is False

    _, t = ring("t", FF(11))

    assert (7*t + 3).is_irreducible is True
    assert (7*t**2 + 3*t + 1).is_irreducible is False

    _, u = ring("u", ZZ)
    f = u**16 + u**14 - u**10 - u**8 - u**6 + u**2

    assert f.is_cyclotomic is False
    assert (f + 1).is_cyclotomic is True

    raises(MultivariatePolynomialError, lambda: x.is_cyclotomic)

    R, = ring("", ZZ)
    assert R(4).is_squarefree is True
    assert R(6).is_irreducible is True

def test_PolyElement_drop():
    R, x,y,z = ring("x,y,z", ZZ)

    assert R(1).drop(0).ring == PolyRing("y,z", ZZ, lex)
    assert R(1).drop(0).drop(0).ring == PolyRing("z", ZZ, lex)
    assert R.is_element(R(1).drop(0).drop(0).drop(0)) is False

    raises(ValueError, lambda: z.drop(0).drop(0).drop(0))
    raises(ValueError, lambda: x.drop(0))

def test_PolyElement_coeff_wrt():
    R, x, y, z = ring("x, y, z", ZZ)

    p = 4*x**3 + 5*y**2 + 6*y**2*z + 7
    assert p.coeff_wrt(1, 2) == 6*z + 5 # using generator index
    assert p.coeff_wrt(x, 3) == 4 # using generator

    p = 2*x**4 + 3*x*y**2*z + 10*y**2 + 10*x*z**2
    assert p.coeff_wrt(x, 1) == 3*y**2*z + 10*z**2
    assert p.coeff_wrt(y, 2) == 3*x*z + 10

    p = 4*x**2 + 2*x*y + 5
    assert p.coeff_wrt(z, 1) == R(0)
    assert p.coeff_wrt(y, 2) == R(0)

def test_PolyElement_prem():
    R, x, y = ring("x, y", ZZ)

    f, g = x**2 + x*y, 2*x + 2
    assert f.prem(g) == -4*y + 4 # first generator is chosen by default if it is not given

    f, g = x**2 + 1, 2*x - 4
    assert f.prem(g) == f.prem(g, x) == 20
    assert f.prem(g, 1) == R(0)

    f, g = x*y + 2*x + 1, x + y
    assert f.prem(g) == -y**2 - 2*y + 1
    assert f.prem(g, 1) == f.prem(g, y) == -x**2 + 2*x + 1

    raises(ZeroDivisionError, lambda: f.prem(R(0)))

def test_PolyElement_pdiv():
    R, x, y = ring("x,y", ZZ)

    f, g = x**4 + 5*x**3 + 7*x**2, 2*x**2 + 3
    assert f.pdiv(g) == f.pdiv(g, x) == (4*x**2 + 20*x + 22, -60*x - 66)

    f, g = x**2 - y**2, x - y
    assert f.pdiv(g) == f.pdiv(g, 0) == (x + y, 0)

    f, g = x*y + 2*x + 1, x + y
    assert f.pdiv(g) == (y + 2, -y**2 - 2*y + 1)
    assert f.pdiv(g, y) == f.pdiv(g, 1) == (x + 1, -x**2 + 2*x + 1)

    assert R(0).pdiv(g) == (0, 0)
    raises(ZeroDivisionError, lambda: f.prem(R(0)))

def test_PolyElement_pquo():
    R, x, y = ring("x, y", ZZ)

    f, g = x**4 - 4*x**2*y + 4*y**2, x**2 - 2*y
    assert f.pquo(g) == f.pquo(g, x) == x**2 - 2*y
    assert f.pquo(g, y) == 4*x**2 - 8*y + 4

    f, g = x**4 - y**4, x**2 - y**2
    assert f.pquo(g) == f.pquo(g, 0) == x**2 + y**2

def test_PolyElement_pexquo():
    R, x, y = ring("x, y", ZZ)

    f, g = x**2 - y**2, x - y
    assert f.pexquo(g) == f.pexquo(g, x) == x + y
    assert f.pexquo(g, y) == f.pexquo(g, 1) == x + y + 1

    f, g = x**2 + 3*x + 6, x + 2
    raises(ExactQuotientFailed, lambda: f.pexquo(g))

def test_PolyElement_gcdex():
    _, x = ring("x", QQ)

    f, g = 2*x, x**2 - 16
    s, t, h = x/32, -QQ(1, 16), 1

    assert f.half_gcdex(g) == (s, h)
    assert f.gcdex(g) == (s, t, h)

def test_PolyElement_subresultants():
    R, x, y = ring("x, y", ZZ)

    f, g = x**2*y + x*y, x + y # degree(f, x) > degree(g, x)
    h = y**3 - y**2
    assert f.subresultants(g) == [f, g, h] # first generator is chosen default

    # generator index or generator is given
    assert f.subresultants(g, 0) ==  f.subresultants(g, x) == [f, g, h]

    assert f.subresultants(g, y) == [x**2*y + x*y, x + y, x**3 + x**2]

    f, g = 2*x - y, x**2 + 2*y + x # degree(f, x) < degree(g, x)
    assert f.subresultants(g) == [x**2 + x + 2*y, 2*x - y, y**2 + 10*y]

    f, g = R(0), y**3 - y**2 # f = 0
    assert f.subresultants(g) == [y**3 - y**2, 1]

    f, g = x**2*y + x*y, R(0) # g = 0
    assert f.subresultants(g) == [x**2*y + x*y, 1]

    f, g = R(0), R(0) # f = 0 and g = 0
    assert f.subresultants(g) == [0, 0]

    f, g = x**2 + x, x**2 + x # f and g are same polynomial
    assert f.subresultants(g) == [x**2 + x, x**2 + x]

def test_PolyElement_resultant():
    _, x = ring("x", ZZ)
    f, g, h = x**2 - 2*x + 1, x**2 - 1, 0

    assert f.resultant(g) == h

def test_PolyElement_discriminant():
    _, x = ring("x", ZZ)
    f, g = x**3 + 3*x**2 + 9*x - 13, -11664

    assert f.discriminant() == g

    F, a, b, c = ring("a,b,c", ZZ)
    _, x = ring("x", F)

    f, g = a*x**2 + b*x + c, b**2 - 4*a*c

    assert f.discriminant() == g

def test_PolyElement_decompose():
    _, x = ring("x", ZZ)

    f = x**12 + 20*x**10 + 150*x**8 + 500*x**6 + 625*x**4 - 2*x**3 - 10*x + 9
    g = x**4 - 2*x + 9
    h = x**3 + 5*x

    assert g.compose(x, h) == f
    assert f.decompose() == [g, h]

def test_PolyElement_shift():
    _, x = ring("x", ZZ)
    assert (x**2 - 2*x + 1).shift(2) == x**2 + 2*x + 1
    assert (x**2 - 2*x + 1).shift_list([2]) == x**2 + 2*x + 1

    R, x, y = ring("x, y", ZZ)
    assert (x*y).shift_list([1, 2]) == (x+1)*(y+2)

    raises(MultivariatePolynomialError, lambda: (x*y).shift(1))

def test_PolyElement_sturm():
    F, t = field("t", ZZ)
    _, x = ring("x", F)

    f = 1024/(15625*t**8)*x**5 - 4096/(625*t**8)*x**4 + 32/(15625*t**4)*x**3 - 128/(625*t**4)*x**2 + F(1)/62500*x - F(1)/625

    assert f.sturm() == [
        x**3 - 100*x**2 + t**4/64*x - 25*t**4/16,
        3*x**2 - 200*x + t**4/64,
        (-t**4/96 + F(20000)/9)*x + 25*t**4/18,
        (-9*t**12 - 11520000*t**8 - 3686400000000*t**4)/(576*t**8 - 245760000*t**4 + 26214400000000),
    ]

def test_PolyElement_gff_list():
    _, x = ring("x", ZZ)

    f = x**5 + 2*x**4 - x**3 - 2*x**2
    assert f.gff_list() == [(x, 1), (x + 2, 4)]

    f = x*(x - 1)**3*(x - 2)**2*(x - 4)**2*(x - 5)
    assert f.gff_list() == [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]

def test_PolyElement_norm():
    k = QQ
    K = QQ.algebraic_field(sqrt(2))
    sqrt2 = K.unit
    _, X, Y = ring("x,y", k)
    _, x, y = ring("x,y", K)

    assert (x*y + sqrt2).norm() == X**2*Y**2 - 2

def test_PolyElement_sqf_norm():
    R, x = ring("x", QQ.algebraic_field(sqrt(3)))
    X = R.to_ground().x

    assert (x**2 - 2).sqf_norm() == ([1], x**2 - 2*sqrt(3)*x + 1, X**4 - 10*X**2 + 1)

    R, x = ring("x", QQ.algebraic_field(sqrt(2)))
    X = R.to_ground().x

    assert (x**2 - 3).sqf_norm() == ([1], x**2 - 2*sqrt(2)*x - 1, X**4 - 10*X**2 + 1)

def test_PolyElement_sqf_list():
    _, x = ring("x", ZZ)

    f = x**5 - x**3 - x**2 + 1
    g = x**3 + 2*x**2 + 2*x + 1
    h = x - 1
    p = x**4 + x**3 - x - 1

    assert f.sqf_part() == p
    assert f.sqf_list() == (1, [(g, 1), (h, 2)])

def test_issue_18894():
    items = [S(3)/16 + sqrt(3*sqrt(3) + 10)/8, S(1)/8 + 3*sqrt(3)/16, S(1)/8 + 3*sqrt(3)/16, -S(3)/16 + sqrt(3*sqrt(3) + 10)/8]
    R, a = sring(items, extension=True)
    assert R.domain == QQ.algebraic_field(sqrt(3)+sqrt(3*sqrt(3)+10))
    assert R.gens == ()
    result = []
    for item in items:
        result.append(R.domain.from_sympy(item))
    assert a == result

def test_PolyElement_factor_list():
    _, x = ring("x", ZZ)

    f = x**5 - x**3 - x**2 + 1

    u = x + 1
    v = x - 1
    w = x**2 + x + 1

    assert f.factor_list() == (1, [(u, 1), (v, 2), (w, 1)])


def test_issue_21410():
    R, x = ring('x', FF(2))
    p = x**6 + x**5 + x**4 + x**3 + 1
    assert p._pow_multinomial(4) == x**24 + x**20 + x**16 + x**12 + 1


def test_zero_polynomial_primitive():

    x = symbols('x')

    R = ZZ[x]
    zero_poly = R(0)
    cont, prim = zero_poly.primitive()
    assert cont == 0
    assert prim == zero_poly
    assert prim.is_primitive is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_style.py ===
import contextlib
import copy
import re
from textwrap import dedent

import numpy as np
import pytest

from pandas import (
    DataFrame,
    IndexSlice,
    MultiIndex,
    Series,
    option_context,
)
import pandas._testing as tm

jinja2 = pytest.importorskip("jinja2")
from pandas.io.formats.style import (  # isort:skip
    Styler,
)
from pandas.io.formats.style_render import (
    _get_level_lengths,
    _get_trimming_maximums,
    maybe_convert_css_to_tuples,
    non_reducing_slice,
)


@pytest.fixture
def mi_df():
    return DataFrame(
        [[1, 2], [3, 4]],
        index=MultiIndex.from_product([["i0"], ["i1_a", "i1_b"]]),
        columns=MultiIndex.from_product([["c0"], ["c1_a", "c1_b"]]),
        dtype=int,
    )


@pytest.fixture
def mi_styler(mi_df):
    return Styler(mi_df, uuid_len=0)


@pytest.fixture
def mi_styler_comp(mi_styler):
    # comprehensively add features to mi_styler
    mi_styler = mi_styler._copy(deepcopy=True)
    mi_styler.css = {**mi_styler.css, "row": "ROW", "col": "COL"}
    mi_styler.uuid_len = 5
    mi_styler.uuid = "abcde"
    mi_styler.set_caption("capt")
    mi_styler.set_table_styles([{"selector": "a", "props": "a:v;"}])
    mi_styler.hide(axis="columns")
    mi_styler.hide([("c0", "c1_a")], axis="columns", names=True)
    mi_styler.hide(axis="index")
    mi_styler.hide([("i0", "i1_a")], axis="index", names=True)
    mi_styler.set_table_attributes('class="box"')
    other = mi_styler.data.agg(["mean"])
    other.index = MultiIndex.from_product([[""], other.index])
    mi_styler.concat(other.style)
    mi_styler.format(na_rep="MISSING", precision=3)
    mi_styler.format_index(precision=2, axis=0)
    mi_styler.format_index(precision=4, axis=1)
    mi_styler.highlight_max(axis=None)
    mi_styler.map_index(lambda x: "color: white;", axis=0)
    mi_styler.map_index(lambda x: "color: black;", axis=1)
    mi_styler.set_td_classes(
        DataFrame(
            [["a", "b"], ["a", "c"]], index=mi_styler.index, columns=mi_styler.columns
        )
    )
    mi_styler.set_tooltips(
        DataFrame(
            [["a2", "b2"], ["a2", "c2"]],
            index=mi_styler.index,
            columns=mi_styler.columns,
        )
    )
    return mi_styler


@pytest.fixture
def blank_value():
    return "&nbsp;"


@pytest.fixture
def df():
    df = DataFrame({"A": [0, 1], "B": np.random.default_rng(2).standard_normal(2)})
    return df


@pytest.fixture
def styler(df):
    df = DataFrame({"A": [0, 1], "B": np.random.default_rng(2).standard_normal(2)})
    return Styler(df)


@pytest.mark.parametrize(
    "sparse_columns, exp_cols",
    [
        (
            True,
            [
                {"is_visible": True, "attributes": 'colspan="2"', "value": "c0"},
                {"is_visible": False, "attributes": "", "value": "c0"},
            ],
        ),
        (
            False,
            [
                {"is_visible": True, "attributes": "", "value": "c0"},
                {"is_visible": True, "attributes": "", "value": "c0"},
            ],
        ),
    ],
)
def test_mi_styler_sparsify_columns(mi_styler, sparse_columns, exp_cols):
    exp_l1_c0 = {"is_visible": True, "attributes": "", "display_value": "c1_a"}
    exp_l1_c1 = {"is_visible": True, "attributes": "", "display_value": "c1_b"}

    ctx = mi_styler._translate(True, sparse_columns)

    assert exp_cols[0].items() <= ctx["head"][0][2].items()
    assert exp_cols[1].items() <= ctx["head"][0][3].items()
    assert exp_l1_c0.items() <= ctx["head"][1][2].items()
    assert exp_l1_c1.items() <= ctx["head"][1][3].items()


@pytest.mark.parametrize(
    "sparse_index, exp_rows",
    [
        (
            True,
            [
                {"is_visible": True, "attributes": 'rowspan="2"', "value": "i0"},
                {"is_visible": False, "attributes": "", "value": "i0"},
            ],
        ),
        (
            False,
            [
                {"is_visible": True, "attributes": "", "value": "i0"},
                {"is_visible": True, "attributes": "", "value": "i0"},
            ],
        ),
    ],
)
def test_mi_styler_sparsify_index(mi_styler, sparse_index, exp_rows):
    exp_l1_r0 = {"is_visible": True, "attributes": "", "display_value": "i1_a"}
    exp_l1_r1 = {"is_visible": True, "attributes": "", "display_value": "i1_b"}

    ctx = mi_styler._translate(sparse_index, True)

    assert exp_rows[0].items() <= ctx["body"][0][0].items()
    assert exp_rows[1].items() <= ctx["body"][1][0].items()
    assert exp_l1_r0.items() <= ctx["body"][0][1].items()
    assert exp_l1_r1.items() <= ctx["body"][1][1].items()


def test_mi_styler_sparsify_options(mi_styler):
    with option_context("styler.sparse.index", False):
        html1 = mi_styler.to_html()
    with option_context("styler.sparse.index", True):
        html2 = mi_styler.to_html()

    assert html1 != html2

    with option_context("styler.sparse.columns", False):
        html1 = mi_styler.to_html()
    with option_context("styler.sparse.columns", True):
        html2 = mi_styler.to_html()

    assert html1 != html2


@pytest.mark.parametrize(
    "rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn",
    [
        (100, 100, 100, None, None, 12, 6),  # reduce to (12, 6) < 100 elements
        (1000, 3, 750, None, None, 250, 3),  # dynamically reduce rows to 250, keep cols
        (4, 1000, 500, None, None, 4, 125),  # dynamically reduce cols to 125, keep rows
        (1000, 3, 750, 10, None, 10, 3),  # overwrite above dynamics with max_row
        (4, 1000, 500, None, 5, 4, 5),  # overwrite above dynamics with max_col
        (100, 100, 700, 50, 50, 25, 25),  # rows cols below given maxes so < 700 elmts
    ],
)
def test_trimming_maximum(rn, cn, max_els, max_rows, max_cols, exp_rn, exp_cn):
    rn, cn = _get_trimming_maximums(
        rn, cn, max_els, max_rows, max_cols, scaling_factor=0.5
    )
    assert (rn, cn) == (exp_rn, exp_cn)


@pytest.mark.parametrize(
    "option, val",
    [
        ("styler.render.max_elements", 6),
        ("styler.render.max_rows", 3),
    ],
)
def test_render_trimming_rows(option, val):
    # test auto and specific trimming of rows
    df = DataFrame(np.arange(120).reshape(60, 2))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx["head"][0]) == 3  # index + 2 data cols
    assert len(ctx["body"]) == 4  # 3 data rows + trimming row
    assert len(ctx["body"][0]) == 3  # index + 2 data cols


@pytest.mark.parametrize(
    "option, val",
    [
        ("styler.render.max_elements", 6),
        ("styler.render.max_columns", 2),
    ],
)
def test_render_trimming_cols(option, val):
    # test auto and specific trimming of cols
    df = DataFrame(np.arange(30).reshape(3, 10))
    with option_context(option, val):
        ctx = df.style._translate(True, True)
    assert len(ctx["head"][0]) == 4  # index + 2 data cols + trimming col
    assert len(ctx["body"]) == 3  # 3 data rows
    assert len(ctx["body"][0]) == 4  # index + 2 data cols + trimming col


def test_render_trimming_mi():
    midx = MultiIndex.from_product([[1, 2], [1, 2, 3]])
    df = DataFrame(np.arange(36).reshape(6, 6), columns=midx, index=midx)
    with option_context("styler.render.max_elements", 4):
        ctx = df.style._translate(True, True)

    assert len(ctx["body"][0]) == 5  # 2 indexes + 2 data cols + trimming row
    assert {"attributes": 'rowspan="2"'}.items() <= ctx["body"][0][0].items()
    assert {"class": "data row0 col_trim"}.items() <= ctx["body"][0][4].items()
    assert {"class": "data row_trim col_trim"}.items() <= ctx["body"][2][4].items()
    assert len(ctx["body"]) == 3  # 2 data rows + trimming row


def test_render_empty_mi():
    # GH 43305
    df = DataFrame(index=MultiIndex.from_product([["A"], [0, 1]], names=[None, "one"]))
    expected = dedent(
        """\
    >
      <thead>
        <tr>
          <th class="index_name level0" >&nbsp;</th>
          <th class="index_name level1" >one</th>
        </tr>
      </thead>
    """
    )
    assert expected in df.style.to_html()


@pytest.mark.parametrize("comprehensive", [True, False])
@pytest.mark.parametrize("render", [True, False])
@pytest.mark.parametrize("deepcopy", [True, False])
def test_copy(comprehensive, render, deepcopy, mi_styler, mi_styler_comp):
    styler = mi_styler_comp if comprehensive else mi_styler
    styler.uuid_len = 5

    s2 = copy.deepcopy(styler) if deepcopy else copy.copy(styler)  # make copy and check
    assert s2 is not styler

    if render:
        styler.to_html()

    excl = [
        "cellstyle_map",  # render time vars..
        "cellstyle_map_columns",
        "cellstyle_map_index",
        "template_latex",  # render templates are class level
        "template_html",
        "template_html_style",
        "template_html_table",
    ]
    if not deepcopy:  # check memory locations are equal for all included attributes
        for attr in [a for a in styler.__dict__ if (not callable(a) and a not in excl)]:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))
    else:  # check memory locations are different for nested or mutable vars
        shallow = [
            "data",
            "columns",
            "index",
            "uuid_len",
            "uuid",
            "caption",
            "cell_ids",
            "hide_index_",
            "hide_columns_",
            "hide_index_names",
            "hide_column_names",
            "table_attributes",
        ]
        for attr in shallow:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))

        for attr in [
            a
            for a in styler.__dict__
            if (not callable(a) and a not in excl and a not in shallow)
        ]:
            if getattr(s2, attr) is None:
                assert id(getattr(s2, attr)) == id(getattr(styler, attr))
            else:
                assert id(getattr(s2, attr)) != id(getattr(styler, attr))


@pytest.mark.parametrize("deepcopy", [True, False])
def test_inherited_copy(mi_styler, deepcopy):
    # Ensure that the inherited class is preserved when a Styler object is copied.
    # GH 52728
    class CustomStyler(Styler):
        pass

    custom_styler = CustomStyler(mi_styler.data)
    custom_styler_copy = (
        copy.deepcopy(custom_styler) if deepcopy else copy.copy(custom_styler)
    )
    assert isinstance(custom_styler_copy, CustomStyler)


def test_clear(mi_styler_comp):
    # NOTE: if this test fails for new features then 'mi_styler_comp' should be updated
    # to ensure proper testing of the 'copy', 'clear', 'export' methods with new feature
    # GH 40675
    styler = mi_styler_comp
    styler._compute()  # execute applied methods

    clean_copy = Styler(styler.data, uuid=styler.uuid)

    excl = [
        "data",
        "index",
        "columns",
        "uuid",
        "uuid_len",  # uuid is set to be the same on styler and clean_copy
        "cell_ids",
        "cellstyle_map",  # execution time only
        "cellstyle_map_columns",  # execution time only
        "cellstyle_map_index",  # execution time only
        "template_latex",  # render templates are class level
        "template_html",
        "template_html_style",
        "template_html_table",
    ]
    # tests vars are not same vals on obj and clean copy before clear (except for excl)
    for attr in [a for a in styler.__dict__ if not (callable(a) or a in excl)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        if hasattr(res, "__iter__") and len(res) > 0:
            assert not all(res)  # some element in iterable differs
        elif hasattr(res, "__iter__") and len(res) == 0:
            pass  # empty array
        else:
            assert not res  # explicit var differs

    # test vars have same vales on obj and clean copy after clearing
    styler.clear()
    for attr in [a for a in styler.__dict__ if not callable(a)]:
        res = getattr(styler, attr) == getattr(clean_copy, attr)
        assert all(res) if hasattr(res, "__iter__") else res


def test_export(mi_styler_comp, mi_styler):
    exp_attrs = [
        "_todo",
        "hide_index_",
        "hide_index_names",
        "hide_columns_",
        "hide_column_names",
        "table_attributes",
        "table_styles",
        "css",
    ]
    for attr in exp_attrs:
        check = getattr(mi_styler, attr) == getattr(mi_styler_comp, attr)
        assert not (
            all(check) if (hasattr(check, "__iter__") and len(check) > 0) else check
        )

    export = mi_styler_comp.export()
    used = mi_styler.use(export)
    for attr in exp_attrs:
        check = getattr(used, attr) == getattr(mi_styler_comp, attr)
        assert all(check) if (hasattr(check, "__iter__") and len(check) > 0) else check

    used.to_html()


def test_hide_raises(mi_styler):
    msg = "`subset` and `level` cannot be passed simultaneously"
    with pytest.raises(ValueError, match=msg):
        mi_styler.hide(axis="index", subset="something", level="something else")

    msg = "`level` must be of type `int`, `str` or list of such"
    with pytest.raises(ValueError, match=msg):
        mi_styler.hide(axis="index", level={"bad": 1, "type": 2})


@pytest.mark.parametrize("level", [1, "one", [1], ["one"]])
def test_hide_index_level(mi_styler, level):
    mi_styler.index.names, mi_styler.columns.names = ["zero", "one"], ["zero", "one"]
    ctx = mi_styler.hide(axis="index", level=level)._translate(False, True)
    assert len(ctx["head"][0]) == 3
    assert len(ctx["head"][1]) == 3
    assert len(ctx["head"][2]) == 4
    assert ctx["head"][2][0]["is_visible"]
    assert not ctx["head"][2][1]["is_visible"]

    assert ctx["body"][0][0]["is_visible"]
    assert not ctx["body"][0][1]["is_visible"]
    assert ctx["body"][1][0]["is_visible"]
    assert not ctx["body"][1][1]["is_visible"]


@pytest.mark.parametrize("level", [1, "one", [1], ["one"]])
@pytest.mark.parametrize("names", [True, False])
def test_hide_columns_level(mi_styler, level, names):
    mi_styler.columns.names = ["zero", "one"]
    if names:
        mi_styler.index.names = ["zero", "one"]
    ctx = mi_styler.hide(axis="columns", level=level)._translate(True, False)
    assert len(ctx["head"]) == (2 if names else 1)


@pytest.mark.parametrize("method", ["map", "apply"])
@pytest.mark.parametrize("axis", ["index", "columns"])
def test_apply_map_header(method, axis):
    # GH 41893
    df = DataFrame({"A": [0, 0], "B": [1, 1]}, index=["C", "D"])
    func = {
        "apply": lambda s: ["attr: val" if ("A" in v or "C" in v) else "" for v in s],
        "map": lambda v: "attr: val" if ("A" in v or "C" in v) else "",
    }

    # test execution added to todo
    result = getattr(df.style, f"{method}_index")(func[method], axis=axis)
    assert len(result._todo) == 1
    assert len(getattr(result, f"ctx_{axis}")) == 0

    # test ctx object on compute
    result._compute()
    expected = {
        (0, 0): [("attr", "val")],
    }
    assert getattr(result, f"ctx_{axis}") == expected


@pytest.mark.parametrize("method", ["apply", "map"])
@pytest.mark.parametrize("axis", ["index", "columns"])
def test_apply_map_header_mi(mi_styler, method, axis):
    # GH 41893
    func = {
        "apply": lambda s: ["attr: val;" if "b" in v else "" for v in s],
        "map": lambda v: "attr: val" if "b" in v else "",
    }
    result = getattr(mi_styler, f"{method}_index")(func[method], axis=axis)._compute()
    expected = {(1, 1): [("attr", "val")]}
    assert getattr(result, f"ctx_{axis}") == expected


def test_apply_map_header_raises(mi_styler):
    # GH 41893
    with pytest.raises(ValueError, match="No axis named bad for object type DataFrame"):
        mi_styler.map_index(lambda v: "attr: val;", axis="bad")._compute()


class TestStyler:
    def test_init_non_pandas(self):
        msg = "``data`` must be a Series or DataFrame"
        with pytest.raises(TypeError, match=msg):
            Styler([1, 2, 3])

    def test_init_series(self):
        result = Styler(Series([1, 2]))
        assert result.data.ndim == 2

    def test_repr_html_ok(self, styler):
        styler._repr_html_()

    def test_repr_html_mathjax(self, styler):
        # gh-19824 / 41395
        assert "tex2jax_ignore" not in styler._repr_html_()

        with option_context("styler.html.mathjax", False):
            assert "tex2jax_ignore" in styler._repr_html_()

    def test_update_ctx(self, styler):
        styler._update_ctx(DataFrame({"A": ["color: red", "color: blue"]}))
        expected = {(0, 0): [("color", "red")], (1, 0): [("color", "blue")]}
        assert styler.ctx == expected

    def test_update_ctx_flatten_multi_and_trailing_semi(self, styler):
        attrs = DataFrame({"A": ["color: red; foo: bar", "color:blue ; foo: baz;"]})
        styler._update_ctx(attrs)
        expected = {
            (0, 0): [("color", "red"), ("foo", "bar")],
            (1, 0): [("color", "blue"), ("foo", "baz")],
        }
        assert styler.ctx == expected

    def test_render(self):
        df = DataFrame({"A": [0, 1]})
        style = lambda x: Series(["color: red", "color: blue"], name=x.name)
        s = Styler(df, uuid="AB").apply(style)
        s.to_html()
        # it worked?

    def test_multiple_render(self, df):
        # GH 39396
        s = Styler(df, uuid_len=0).map(lambda x: "color: red;", subset=["A"])
        s.to_html()  # do 2 renders to ensure css styles not duplicated
        assert (
            '<style type="text/css">\n#T__row0_col0, #T__row1_col0 {\n'
            "  color: red;\n}\n</style>" in s.to_html()
        )

    def test_render_empty_dfs(self):
        empty_df = DataFrame()
        es = Styler(empty_df)
        es.to_html()
        # An index but no columns
        DataFrame(columns=["a"]).style.to_html()
        # A column but no index
        DataFrame(index=["a"]).style.to_html()
        # No IndexError raised?

    def test_render_double(self):
        df = DataFrame({"A": [0, 1]})
        style = lambda x: Series(
            ["color: red; border: 1px", "color: blue; border: 2px"], name=x.name
        )
        s = Styler(df, uuid="AB").apply(style)
        s.to_html()
        # it worked?

    def test_set_properties(self):
        df = DataFrame({"A": [0, 1]})
        result = df.style.set_properties(color="white", size="10px")._compute().ctx
        # order is deterministic
        v = [("color", "white"), ("size", "10px")]
        expected = {(0, 0): v, (1, 0): v}
        assert result.keys() == expected.keys()
        for v1, v2 in zip(result.values(), expected.values()):
            assert sorted(v1) == sorted(v2)

    def test_set_properties_subset(self):
        df = DataFrame({"A": [0, 1]})
        result = (
            df.style.set_properties(subset=IndexSlice[0, "A"], color="white")
            ._compute()
            .ctx
        )
        expected = {(0, 0): [("color", "white")]}
        assert result == expected

    def test_empty_index_name_doesnt_display(self, blank_value):
        # https://github.com/pandas-dev/pandas/pull/12090#issuecomment-180695902
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = df.style._translate(True, True)
        assert len(result["head"]) == 1
        expected = {
            "class": "blank level0",
            "type": "th",
            "value": blank_value,
            "is_visible": True,
            "display_value": blank_value,
        }
        assert expected.items() <= result["head"][0][0].items()

    def test_index_name(self):
        # https://github.com/pandas-dev/pandas/issues/11655
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = df.set_index("A").style._translate(True, True)
        expected = {
            "class": "index_name level0",
            "type": "th",
            "value": "A",
            "is_visible": True,
            "display_value": "A",
        }
        assert expected.items() <= result["head"][1][0].items()

    def test_numeric_columns(self):
        # https://github.com/pandas-dev/pandas/issues/12125
        # smoke test for _translate
        df = DataFrame({0: [1, 2, 3]})
        df.style._translate(True, True)

    def test_apply_axis(self):
        df = DataFrame({"A": [0, 0], "B": [1, 1]})
        f = lambda x: [f"val: {x.max()}" for v in x]
        result = df.style.apply(f, axis=1)
        assert len(result._todo) == 1
        assert len(result.ctx) == 0
        result._compute()
        expected = {
            (0, 0): [("val", "1")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "1")],
            (1, 1): [("val", "1")],
        }
        assert result.ctx == expected

        result = df.style.apply(f, axis=0)
        expected = {
            (0, 0): [("val", "0")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "0")],
            (1, 1): [("val", "1")],
        }
        result._compute()
        assert result.ctx == expected
        result = df.style.apply(f)  # default
        result._compute()
        assert result.ctx == expected

    @pytest.mark.parametrize("axis", [0, 1])
    def test_apply_series_return(self, axis):
        # GH 42014
        df = DataFrame([[1, 2], [3, 4]], index=["X", "Y"], columns=["X", "Y"])

        # test Series return where len(Series) < df.index or df.columns but labels OK
        func = lambda s: Series(["color: red;"], index=["Y"])
        result = df.style.apply(func, axis=axis)._compute().ctx
        assert result[(1, 1)] == [("color", "red")]
        assert result[(1 - axis, axis)] == [("color", "red")]

        # test Series return where labels align but different order
        func = lambda s: Series(["color: red;", "color: blue;"], index=["Y", "X"])
        result = df.style.apply(func, axis=axis)._compute().ctx
        assert result[(0, 0)] == [("color", "blue")]
        assert result[(1, 1)] == [("color", "red")]
        assert result[(1 - axis, axis)] == [("color", "red")]
        assert result[(axis, 1 - axis)] == [("color", "blue")]

    @pytest.mark.parametrize("index", [False, True])
    @pytest.mark.parametrize("columns", [False, True])
    def test_apply_dataframe_return(self, index, columns):
        # GH 42014
        df = DataFrame([[1, 2], [3, 4]], index=["X", "Y"], columns=["X", "Y"])
        idxs = ["X", "Y"] if index else ["Y"]
        cols = ["X", "Y"] if columns else ["Y"]
        df_styles = DataFrame("color: red;", index=idxs, columns=cols)
        result = df.style.apply(lambda x: df_styles, axis=None)._compute().ctx

        assert result[(1, 1)] == [("color", "red")]  # (Y,Y) styles always present
        assert (result[(0, 1)] == [("color", "red")]) is index  # (X,Y) only if index
        assert (result[(1, 0)] == [("color", "red")]) is columns  # (Y,X) only if cols
        assert (result[(0, 0)] == [("color", "red")]) is (index and columns)  # (X,X)

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:],
            IndexSlice[:, ["A"]],
            IndexSlice[[1], :],
            IndexSlice[[1], ["A"]],
            IndexSlice[:2, ["A", "B"]],
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_apply_subset(self, slice_, axis, df):
        def h(x, color="bar"):
            return Series(f"color: {color}", index=x.index, name=x.name)

        result = df.style.apply(h, axis=axis, subset=slice_, color="baz")._compute().ctx
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:],
            IndexSlice[:, ["A"]],
            IndexSlice[[1], :],
            IndexSlice[[1], ["A"]],
            IndexSlice[:2, ["A", "B"]],
        ],
    )
    def test_map_subset(self, slice_, df):
        result = df.style.map(lambda x: "color:baz;", subset=slice_)._compute().ctx
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(df.index)
            for c, col in enumerate(df.columns)
            if row in df.loc[slice_].index and col in df.loc[slice_].columns
        }
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:, IndexSlice["x", "A"]],
            IndexSlice[:, IndexSlice[:, "A"]],
            IndexSlice[:, IndexSlice[:, ["A", "C"]]],  # missing col element
            IndexSlice[IndexSlice["a", 1], :],
            IndexSlice[IndexSlice[:, 1], :],
            IndexSlice[IndexSlice[:, [1, 3]], :],  # missing row element
            IndexSlice[:, ("x", "A")],
            IndexSlice[("a", 1), :],
        ],
    )
    def test_map_subset_multiindex(self, slice_):
        # GH 19861
        # edited for GH 33562
        if (
            isinstance(slice_[-1], tuple)
            and isinstance(slice_[-1][-1], list)
            and "C" in slice_[-1][-1]
        ):
            ctx = pytest.raises(KeyError, match="C")
        elif (
            isinstance(slice_[0], tuple)
            and isinstance(slice_[0][1], list)
            and 3 in slice_[0][1]
        ):
            ctx = pytest.raises(KeyError, match="3")
        else:
            ctx = contextlib.nullcontext()

        idx = MultiIndex.from_product([["a", "b"], [1, 2]])
        col = MultiIndex.from_product([["x", "y"], ["A", "B"]])
        df = DataFrame(np.random.default_rng(2).random((4, 4)), columns=col, index=idx)

        with ctx:
            df.style.map(lambda x: "color: red;", subset=slice_).to_html()

    def test_map_subset_multiindex_code(self):
        # https://github.com/pandas-dev/pandas/issues/25858
        # Checks styler.map works with multindex when codes are provided
        codes = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        columns = MultiIndex(
            levels=[["a", "b"], ["%", "#"]], codes=codes, names=["", ""]
        )
        df = DataFrame(
            [[1, -1, 1, 1], [-1, 1, 1, 1]], index=["hello", "world"], columns=columns
        )
        pct_subset = IndexSlice[:, IndexSlice[:, "%":"%"]]

        def color_negative_red(val):
            color = "red" if val < 0 else "black"
            return f"color: {color}"

        df.loc[pct_subset]
        df.style.map(color_negative_red, subset=pct_subset)

    @pytest.mark.parametrize(
        "stylefunc", ["background_gradient", "bar", "text_gradient"]
    )
    def test_subset_for_boolean_cols(self, stylefunc):
        # GH47838
        df = DataFrame(
            [
                [1, 2],
                [3, 4],
            ],
            columns=[False, True],
        )
        styled = getattr(df.style, stylefunc)()
        styled._compute()
        assert set(styled.ctx) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_empty(self):
        df = DataFrame({"A": [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("", "")]}

        result = s._translate(True, True)["cellstyle"]
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0"]},
            {"props": [("", "")], "selectors": ["row1_col0"]},
        ]
        assert result == expected

    def test_duplicate(self):
        df = DataFrame({"A": [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("color", "red")]}

        result = s._translate(True, True)["cellstyle"]
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0", "row1_col0"]}
        ]
        assert result == expected

    def test_init_with_na_rep(self):
        # GH 21527 28358
        df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

        ctx = Styler(df, na_rep="NA")._translate(True, True)
        assert ctx["body"][0][1]["display_value"] == "NA"
        assert ctx["body"][0][2]["display_value"] == "NA"

    def test_caption(self, df):
        styler = Styler(df, caption="foo")
        result = styler.to_html()
        assert all(["caption" in result, "foo" in result])

        styler = df.style
        result = styler.set_caption("baz")
        assert styler is result
        assert styler.caption == "baz"

    def test_uuid(self, df):
        styler = Styler(df, uuid="abc123")
        result = styler.to_html()
        assert "abc123" in result

        styler = df.style
        result = styler.set_uuid("aaa")
        assert result is styler
        assert result.uuid == "aaa"

    def test_unique_id(self):
        # See https://github.com/pandas-dev/pandas/issues/16780
        df = DataFrame({"a": [1, 3, 5, 6], "b": [2, 4, 12, 21]})
        result = df.style.to_html(uuid="test")
        assert "test" in result
        ids = re.findall('id="(.*?)"', result)
        assert np.unique(ids).size == len(ids)

    def test_table_styles(self, df):
        style = [{"selector": "th", "props": [("foo", "bar")]}]  # default format
        styler = Styler(df, table_styles=style)
        result = " ".join(styler.to_html().split())
        assert "th { foo: bar; }" in result

        styler = df.style
        result = styler.set_table_styles(style)
        assert styler is result
        assert styler.table_styles == style

        # GH 39563
        style = [{"selector": "th", "props": "foo:bar;"}]  # css string format
        styler = df.style.set_table_styles(style)
        result = " ".join(styler.to_html().split())
        assert "th { foo: bar; }" in result

    def test_table_styles_multiple(self, df):
        ctx = df.style.set_table_styles(
            [
                {"selector": "th,td", "props": "color:red;"},
                {"selector": "tr", "props": "color:green;"},
            ]
        )._translate(True, True)["table_styles"]
        assert ctx == [
            {"selector": "th", "props": [("color", "red")]},
            {"selector": "td", "props": [("color", "red")]},
            {"selector": "tr", "props": [("color", "green")]},
        ]

    def test_table_styles_dict_multiple_selectors(self, df):
        # GH 44011
        result = df.style.set_table_styles(
            {
                "B": [
                    {"selector": "th,td", "props": [("border-left", "2px solid black")]}
                ]
            }
        )._translate(True, True)["table_styles"]

        expected = [
            {"selector": "th.col1", "props": [("border-left", "2px solid black")]},
            {"selector": "td.col1", "props": [("border-left", "2px solid black")]},
        ]

        assert result == expected

    def test_maybe_convert_css_to_tuples(self):
        expected = [("a", "b"), ("c", "d e")]
        assert maybe_convert_css_to_tuples("a:b;c:d e;") == expected
        assert maybe_convert_css_to_tuples("a: b ;c:  d e  ") == expected
        expected = []
        assert maybe_convert_css_to_tuples("") == expected

    def test_maybe_convert_css_to_tuples_err(self):
        msg = "Styles supplied as string must follow CSS rule formats"
        with pytest.raises(ValueError, match=msg):
            maybe_convert_css_to_tuples("err")

    def test_table_attributes(self, df):
        attributes = 'class="foo" data-bar'
        styler = Styler(df, table_attributes=attributes)
        result = styler.to_html()
        assert 'class="foo" data-bar' in result

        result = df.style.set_table_attributes(attributes).to_html()
        assert 'class="foo" data-bar' in result

    def test_apply_none(self):
        def f(x):
            return DataFrame(
                np.where(x == x.max(), "color: red", ""),
                index=x.index,
                columns=x.columns,
            )

        result = DataFrame([[1, 2], [3, 4]]).style.apply(f, axis=None)._compute().ctx
        assert result[(1, 1)] == [("color", "red")]

    def test_trim(self, df):
        result = df.style.to_html()  # trim=True
        assert result.count("#") == 0

        result = df.style.highlight_max().to_html()
        assert result.count("#") == len(df.columns)

    def test_export(self, df, styler):
        f = lambda x: "color: red" if x > 0 else "color: blue"
        g = lambda x, z: f"color: {z}" if x > 0 else f"color: {z}"
        style1 = styler
        style1.map(f).map(g, z="b").highlight_max()._compute()  # = render
        result = style1.export()
        style2 = df.style
        style2.use(result)
        assert style1._todo == style2._todo
        style2.to_html()

    def test_bad_apply_shape(self):
        df = DataFrame([[1, 2], [3, 4]], index=["A", "B"], columns=["X", "Y"])

        msg = "resulted in the apply method collapsing to a Series."
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: "x")

        msg = "created invalid {} labels"
        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: [""])

        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: ["", "", "", ""])

        with pytest.raises(ValueError, match=msg.format("index")):
            df.style._apply(lambda x: Series(["a:v;", ""], index=["A", "C"]), axis=0)

        with pytest.raises(ValueError, match=msg.format("columns")):
            df.style._apply(lambda x: ["", "", ""], axis=1)

        with pytest.raises(ValueError, match=msg.format("columns")):
            df.style._apply(lambda x: Series(["a:v;", ""], index=["X", "Z"]), axis=1)

        msg = "returned ndarray with wrong shape"
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: np.array([[""], [""]]), axis=None)

    def test_apply_bad_return(self):
        def f(x):
            return ""

        df = DataFrame([[1, 2], [3, 4]])
        msg = (
            "must return a DataFrame or ndarray when passed to `Styler.apply` "
            "with axis=None"
        )
        with pytest.raises(TypeError, match=msg):
            df.style._apply(f, axis=None)

    @pytest.mark.parametrize("axis", ["index", "columns"])
    def test_apply_bad_labels(self, axis):
        def f(x):
            return DataFrame(**{axis: ["bad", "labels"]})

        df = DataFrame([[1, 2], [3, 4]])
        msg = f"created invalid {axis} labels."
        with pytest.raises(ValueError, match=msg):
            df.style._apply(f, axis=None)

    def test_get_level_lengths(self):
        index = MultiIndex.from_product([["a", "b"], [0, 1, 2]])
        expected = {
            (0, 0): 3,
            (0, 3): 3,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)

        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 1,
            (0, 5): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_get_level_lengths_un_sorted(self):
        index = MultiIndex.from_arrays([[1, 1, 2, 1], ["a", "b", "b", "d"]])
        expected = {
            (0, 0): 2,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)

        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_mi_sparse_index_names(self, blank_value):
        # Test the class names and displayed value are correct on rendering MI names
        df = DataFrame(
            {"A": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        result = df.style._translate(True, True)
        head = result["head"][1]
        expected = [
            {
                "class": "index_name level0",
                "display_value": "idx_level_0",
                "is_visible": True,
            },
            {
                "class": "index_name level1",
                "display_value": "idx_level_1",
                "is_visible": True,
            },
            {
                "class": "blank col0",
                "display_value": blank_value,
                "is_visible": True,
            },
        ]
        for i, expected_dict in enumerate(expected):
            assert expected_dict.items() <= head[i].items()

    def test_mi_sparse_column_names(self, blank_value):
        df = DataFrame(
            np.arange(16).reshape(4, 4),
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "a"], [0, 1, 1, 2]],
                names=["idx_level_0", "idx_level_1"],
            ),
            columns=MultiIndex.from_arrays(
                [["C1", "C1", "C2", "C2"], [1, 0, 1, 0]], names=["colnam_0", "colnam_1"]
            ),
        )
        result = Styler(df, cell_ids=False)._translate(True, True)

        for level in [0, 1]:
            head = result["head"][level]
            expected = [
                {
                    "class": "blank",
                    "display_value": blank_value,
                    "is_visible": True,
                },
                {
                    "class": f"index_name level{level}",
                    "display_value": f"colnam_{level}",
                    "is_visible": True,
                },
            ]
            for i, expected_dict in enumerate(expected):
                assert expected_dict.items() <= head[i].items()

    def test_hide_column_headers(self, df, styler):
        ctx = styler.hide(axis="columns")._translate(True, True)
        assert len(ctx["head"]) == 0  # no header entries with an unnamed index

        df.index.name = "some_name"
        ctx = df.style.hide(axis="columns")._translate(True, True)
        assert len(ctx["head"]) == 1
        # index names still visible, changed in #42101, reverted in 43404

    def test_hide_single_index(self, df):
        # GH 14194
        # single unnamed index
        ctx = df.style._translate(True, True)
        assert ctx["body"][0][0]["is_visible"]
        assert ctx["head"][0][0]["is_visible"]
        ctx2 = df.style.hide(axis="index")._translate(True, True)
        assert not ctx2["body"][0][0]["is_visible"]
        assert not ctx2["head"][0][0]["is_visible"]

        # single named index
        ctx3 = df.set_index("A").style._translate(True, True)
        assert ctx3["body"][0][0]["is_visible"]
        assert len(ctx3["head"]) == 2  # 2 header levels
        assert ctx3["head"][0][0]["is_visible"]

        ctx4 = df.set_index("A").style.hide(axis="index")._translate(True, True)
        assert not ctx4["body"][0][0]["is_visible"]
        assert len(ctx4["head"]) == 1  # only 1 header levels
        assert not ctx4["head"][0][0]["is_visible"]

    def test_hide_multiindex(self):
        # GH 14194
        df = DataFrame(
            {"A": [1, 2], "B": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        ctx1 = df.style._translate(True, True)
        # tests for 'a' and '0'
        assert ctx1["body"][0][0]["is_visible"]
        assert ctx1["body"][0][1]["is_visible"]
        # check for blank header rows
        assert len(ctx1["head"][0]) == 4  # two visible indexes and two data columns

        ctx2 = df.style.hide(axis="index")._translate(True, True)
        # tests for 'a' and '0'
        assert not ctx2["body"][0][0]["is_visible"]
        assert not ctx2["body"][0][1]["is_visible"]
        # check for blank header rows
        assert len(ctx2["head"][0]) == 3  # one hidden (col name) and two data columns
        assert not ctx2["head"][0][0]["is_visible"]

    def test_hide_columns_single_level(self, df):
        # GH 14194
        # test hiding single column
        ctx = df.style._translate(True, True)
        assert ctx["head"][0][1]["is_visible"]
        assert ctx["head"][0][1]["display_value"] == "A"
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][0][2]["display_value"] == "B"
        assert ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert ctx["body"][1][2]["is_visible"]  # col B, row 1

        ctx = df.style.hide("A", axis="columns")._translate(True, True)
        assert not ctx["head"][0][1]["is_visible"]
        assert not ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert ctx["body"][1][2]["is_visible"]  # col B, row 1

        # test hiding multiple columns
        ctx = df.style.hide(["A", "B"], axis="columns")._translate(True, True)
        assert not ctx["head"][0][1]["is_visible"]
        assert not ctx["head"][0][2]["is_visible"]
        assert not ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert not ctx["body"][1][2]["is_visible"]  # col B, row 1

    def test_hide_columns_index_mult_levels(self):
        # GH 14194
        # setup dataframe with multiple column levels and indices
        i1 = MultiIndex.from_arrays(
            [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
        )
        i2 = MultiIndex.from_arrays(
            [["b", "b"], [0, 1]], names=["col_level_0", "col_level_1"]
        )
        df = DataFrame([[1, 2], [3, 4]], index=i1, columns=i2)
        ctx = df.style._translate(True, True)
        # column headers
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][1][2]["is_visible"]
        assert ctx["head"][1][3]["display_value"] == "1"
        # indices
        assert ctx["body"][0][0]["is_visible"]
        # data
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == "3"
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == "4"

        # hide top column level, which hides both columns
        ctx = df.style.hide("b", axis="columns")._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][0][0]["is_visible"]  # index

        # hide first column only
        ctx = df.style.hide([("b", 0)], axis="columns")._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert ctx["head"][0][3]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == "4"

        # hide second column and index
        ctx = df.style.hide([("b", 1)], axis=1).hide(axis=0)._translate(True, True)
        assert not ctx["body"][0][0]["is_visible"]  # index
        assert len(ctx["head"][0]) == 3
        assert ctx["head"][0][1]["is_visible"]  # b
        assert ctx["head"][1][1]["is_visible"]  # 0
        assert not ctx["head"][1][2]["is_visible"]  # 1
        assert not ctx["body"][1][3]["is_visible"]  # 4
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == "3"

        # hide top row level, which hides both rows so body empty
        ctx = df.style.hide("a", axis="index")._translate(True, True)
        assert ctx["body"] == []

        # hide first row only
        ctx = df.style.hide(("a", 0), axis="index")._translate(True, True)
        for i in [0, 1, 2, 3]:
            assert "row1" in ctx["body"][0][i]["class"]  # row0 not included in body
            assert ctx["body"][0][i]["is_visible"]

    def test_pipe(self, df):
        def set_caption_from_template(styler, a, b):
            return styler.set_caption(f"Dataframe with a = {a} and b = {b}")

        styler = df.style.pipe(set_caption_from_template, "A", b="B")
        assert "Dataframe with a = A and b = B" in styler.to_html()

        # Test with an argument that is a (callable, keyword_name) pair.
        def f(a, b, styler):
            return (a, b, styler)

        styler = df.style
        result = styler.pipe((f, "styler"), a=1, b=2)
        assert result == (1, 2, styler)

    def test_no_cell_ids(self):
        # GH 35588
        # GH 35663
        df = DataFrame(data=[[0]])
        styler = Styler(df, uuid="_", cell_ids=False)
        styler.to_html()
        s = styler.to_html()  # render twice to ensure ctx is not updated
        assert s.find('<td class="data row0 col0" >') != -1

    @pytest.mark.parametrize(
        "classes",
        [
            DataFrame(
                data=[["", "test-class"], [np.nan, None]],
                columns=["A", "B"],
                index=["a", "b"],
            ),
            DataFrame(data=[["test-class"]], columns=["B"], index=["a"]),
            DataFrame(data=[["test-class", "unused"]], columns=["B", "C"], index=["a"]),
        ],
    )
    def test_set_data_classes(self, classes):
        # GH 36159
        df = DataFrame(data=[[0, 1], [2, 3]], columns=["A", "B"], index=["a", "b"])
        s = Styler(df, uuid_len=0, cell_ids=False).set_td_classes(classes).to_html()
        assert '<td class="data row0 col0" >0</td>' in s
        assert '<td class="data row0 col1 test-class" >1</td>' in s
        assert '<td class="data row1 col0" >2</td>' in s
        assert '<td class="data row1 col1" >3</td>' in s
        # GH 39317
        s = Styler(df, uuid_len=0, cell_ids=True).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0" >0</td>' in s
        assert '<td id="T__row0_col1" class="data row0 col1 test-class" >1</td>' in s
        assert '<td id="T__row1_col0" class="data row1 col0" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >3</td>' in s

    def test_set_data_classes_reindex(self):
        # GH 39317
        df = DataFrame(
            data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=[0, 1, 2], index=[0, 1, 2]
        )
        classes = DataFrame(
            data=[["mi", "ma"], ["mu", "mo"]],
            columns=[0, 2],
            index=[0, 2],
        )
        s = Styler(df, uuid_len=0).set_td_classes(classes).to_html()
        assert '<td id="T__row0_col0" class="data row0 col0 mi" >0</td>' in s
        assert '<td id="T__row0_col2" class="data row0 col2 ma" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >4</td>' in s
        assert '<td id="T__row2_col0" class="data row2 col0 mu" >6</td>' in s
        assert '<td id="T__row2_col2" class="data row2 col2 mo" >8</td>' in s

    def test_chaining_table_styles(self):
        # GH 35607
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        styler = df.style.set_table_styles(
            [{"selector": "", "props": [("background-color", "yellow")]}]
        ).set_table_styles(
            [{"selector": ".col0", "props": [("background-color", "blue")]}],
            overwrite=False,
        )
        assert len(styler.table_styles) == 2

    def test_column_and_row_styling(self):
        # GH 35607
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        s = Styler(df, uuid_len=0)
        s = s.set_table_styles({"A": [{"selector": "", "props": [("color", "blue")]}]})
        assert "#T_ .col0 {\n  color: blue;\n}" in s.to_html()
        s = s.set_table_styles(
            {0: [{"selector": "", "props": [("color", "blue")]}]}, axis=1
        )
        assert "#T_ .row0 {\n  color: blue;\n}" in s.to_html()

    @pytest.mark.parametrize("len_", [1, 5, 32, 33, 100])
    def test_uuid_len(self, len_):
        # GH 36345
        df = DataFrame(data=[["A"]])
        s = Styler(df, uuid_len=len_, cell_ids=False).to_html()
        strt = s.find('id="T_')
        end = s[strt + 6 :].find('"')
        if len_ > 32:
            assert end == 32
        else:
            assert end == len_

    @pytest.mark.parametrize("len_", [-2, "bad", None])
    def test_uuid_len_raises(self, len_):
        # GH 36345
        df = DataFrame(data=[["A"]])
        msg = "``uuid_len`` must be an integer in range \\[0, 32\\]."
        with pytest.raises(TypeError, match=msg):
            Styler(df, uuid_len=len_, cell_ids=False).to_html()

    @pytest.mark.parametrize(
        "slc",
        [
            IndexSlice[:, :],
            IndexSlice[:, 1],
            IndexSlice[1, :],
            IndexSlice[[1], [1]],
            IndexSlice[1, [1]],
            IndexSlice[[1], 1],
            IndexSlice[1],
            IndexSlice[1, 1],
            slice(None, None, None),
            [0, 1],
            np.array([0, 1]),
            Series([0, 1]),
        ],
    )
    def test_non_reducing_slice(self, slc):
        df = DataFrame([[0, 1], [2, 3]])

        tslice_ = non_reducing_slice(slc)
        assert isinstance(df.loc[tslice_], DataFrame)

    @pytest.mark.parametrize("box", [list, Series, np.array])
    def test_list_slice(self, box):
        # like dataframe getitem
        subset = box(["A"])

        df = DataFrame({"A": [1, 2], "B": [3, 4]}, index=["A", "B"])
        expected = IndexSlice[:, ["A"]]

        result = non_reducing_slice(subset)
        tm.assert_frame_equal(df.loc[result], df.loc[expected])

    def test_non_reducing_slice_on_multiindex(self):
        # GH 19861
        dic = {
            ("a", "d"): [1, 4],
            ("a", "c"): [2, 3],
            ("b", "c"): [3, 2],
            ("b", "d"): [4, 1],
        }
        df = DataFrame(dic, index=[0, 1])
        idx = IndexSlice
        slice_ = idx[:, idx["b", "d"]]
        tslice_ = non_reducing_slice(slice_)

        result = df.loc[tslice_]
        expected = DataFrame({("b", "d"): [4, 1]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "slice_",
        [
            IndexSlice[:, :],
            # check cols
            IndexSlice[:, IndexSlice[["a"]]],  # inferred deeper need list
            IndexSlice[:, IndexSlice[["a"], ["c"]]],  # inferred deeper need list
            IndexSlice[:, IndexSlice["a", "c", :]],
            IndexSlice[:, IndexSlice["a", :, "e"]],
            IndexSlice[:, IndexSlice[:, "c", "e"]],
            IndexSlice[:, IndexSlice["a", ["c", "d"], :]],  # check list
            IndexSlice[:, IndexSlice["a", ["c", "d", "-"], :]],  # don't allow missing
            IndexSlice[:, IndexSlice["a", ["c", "d", "-"], "e"]],  # no slice
            # check rows
            IndexSlice[IndexSlice[["U"]], :],  # inferred deeper need list
            IndexSlice[IndexSlice[["U"], ["W"]], :],  # inferred deeper need list
            IndexSlice[IndexSlice["U", "W", :], :],
            IndexSlice[IndexSlice["U", :, "Y"], :],
            IndexSlice[IndexSlice[:, "W", "Y"], :],
            IndexSlice[IndexSlice[:, "W", ["Y", "Z"]], :],  # check list
            IndexSlice[IndexSlice[:, "W", ["Y", "Z", "-"]], :],  # don't allow missing
            IndexSlice[IndexSlice["U", "W", ["Y", "Z", "-"]], :],  # no slice
            # check simultaneous
            IndexSlice[IndexSlice[:, "W", "Y"], IndexSlice["a", "c", :]],
        ],
    )
    def test_non_reducing_multi_slice_on_multiindex(self, slice_):
        # GH 33562
        cols = MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]])
        idxs = MultiIndex.from_product([["U", "V"], ["W", "X"], ["Y", "Z"]])
        df = DataFrame(np.arange(64).reshape(8, 8), columns=cols, index=idxs)

        for lvl in [0, 1]:
            key = slice_[lvl]
            if isinstance(key, tuple):
                for subkey in key:
                    if isinstance(subkey, list) and "-" in subkey:
                        # not present in the index level, raises KeyError since 2.0
                        with pytest.raises(KeyError, match="-"):
                            df.loc[slice_]
                        return

        expected = df.loc[slice_]
        result = df.loc[non_reducing_slice(slice_)]
        tm.assert_frame_equal(result, expected)


def test_hidden_index_names(mi_df):
    mi_df.index.names = ["Lev0", "Lev1"]
    mi_styler = mi_df.style
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 3  # 2 column index levels + 1 index names row

    mi_styler.hide(axis="index", names=True)
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 2  # index names row is unparsed
    for i in range(4):
        assert ctx["body"][0][i]["is_visible"]  # 2 index levels + 2 data values visible

    mi_styler.hide(axis="index", level=1)
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 2  # index names row is still hidden
    assert ctx["body"][0][0]["is_visible"] is True
    assert ctx["body"][0][1]["is_visible"] is False


def test_hidden_column_names(mi_df):
    mi_df.columns.names = ["Lev0", "Lev1"]
    mi_styler = mi_df.style
    ctx = mi_styler._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "Lev0"
    assert ctx["head"][1][1]["display_value"] == "Lev1"

    mi_styler.hide(names=True, axis="columns")
    ctx = mi_styler._translate(True, True)
    assert ctx["head"][0][1]["display_value"] == "&nbsp;"
    assert ctx["head"][1][1]["display_value"] == "&nbsp;"

    mi_styler.hide(level=0, axis="columns")
    ctx = mi_styler._translate(True, True)
    assert len(ctx["head"]) == 1  # no index names and only one visible column headers
    assert ctx["head"][0][1]["display_value"] == "&nbsp;"


@pytest.mark.parametrize("caption", [1, ("a", "b", "c"), (1, "s")])
def test_caption_raises(mi_styler, caption):
    msg = "`caption` must be either a string or 2-tuple of strings."
    with pytest.raises(ValueError, match=msg):
        mi_styler.set_caption(caption)


def test_hiding_headers_over_index_no_sparsify():
    # GH 43464
    midx = MultiIndex.from_product([[1, 2], ["a", "a", "b"]])
    df = DataFrame(9, index=midx, columns=[0])
    ctx = df.style._translate(False, False)
    assert len(ctx["body"]) == 6
    ctx = df.style.hide((1, "a"), axis=0)._translate(False, False)
    assert len(ctx["body"]) == 4
    assert "row2" in ctx["body"][0][0]["class"]


def test_hiding_headers_over_columns_no_sparsify():
    # GH 43464
    midx = MultiIndex.from_product([[1, 2], ["a", "a", "b"]])
    df = DataFrame(9, columns=midx, index=[0])
    ctx = df.style._translate(False, False)
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx["head"][ix[0]][ix[1]]["is_visible"] is True
    ctx = df.style.hide((1, "a"), axis="columns")._translate(False, False)
    for ix in [(0, 1), (0, 2), (1, 1), (1, 2)]:
        assert ctx["head"][ix[0]][ix[1]]["is_visible"] is False


def test_get_level_lengths_mi_hidden():
    # GH 43464
    index = MultiIndex.from_arrays([[1, 1, 1, 2, 2, 2], ["a", "a", "b", "a", "a", "b"]])
    expected = {
        (0, 2): 1,
        (0, 3): 1,
        (0, 4): 1,
        (0, 5): 1,
        (1, 2): 1,
        (1, 3): 1,
        (1, 4): 1,
        (1, 5): 1,
    }
    result = _get_level_lengths(
        index,
        sparsify=False,
        max_index=100,
        hidden_elements=[0, 1, 0, 1],  # hidden element can repeat if duplicated index
    )
    tm.assert_dict_equal(result, expected)


def test_row_trimming_hide_index():
    # gh 43703
    df = DataFrame([[1], [2], [3], [4], [5]])
    with option_context("styler.render.max_rows", 2):
        ctx = df.style.hide([0, 1], axis="index")._translate(True, True)
    assert len(ctx["body"]) == 3
    for r, val in enumerate(["3", "4", "..."]):
        assert ctx["body"][r][1]["display_value"] == val


def test_row_trimming_hide_index_mi():
    # gh 44247
    df = DataFrame([[1], [2], [3], [4], [5]])
    df.index = MultiIndex.from_product([[0], [0, 1, 2, 3, 4]])
    with option_context("styler.render.max_rows", 2):
        ctx = df.style.hide([(0, 0), (0, 1)], axis="index")._translate(True, True)
    assert len(ctx["body"]) == 3

    # level 0 index headers (sparsified)
    assert {"value": 0, "attributes": 'rowspan="2"', "is_visible": True}.items() <= ctx[
        "body"
    ][0][0].items()
    assert {"value": 0, "attributes": "", "is_visible": False}.items() <= ctx["body"][
        1
    ][0].items()
    assert {"value": "...", "is_visible": True}.items() <= ctx["body"][2][0].items()

    for r, val in enumerate(["2", "3", "..."]):
        assert ctx["body"][r][1]["display_value"] == val  # level 1 index headers
    for r, val in enumerate(["3", "4", "..."]):
        assert ctx["body"][r][2]["display_value"] == val  # data values


def test_col_trimming_hide_columns():
    # gh 44272
    df = DataFrame([[1, 2, 3, 4, 5]])
    with option_context("styler.render.max_columns", 2):
        ctx = df.style.hide([0, 1], axis="columns")._translate(True, True)

    assert len(ctx["head"][0]) == 6  # blank, [0, 1 (hidden)], [2 ,3 (visible)], + trim
    for c, vals in enumerate([(1, False), (2, True), (3, True), ("...", True)]):
        assert ctx["head"][0][c + 2]["value"] == vals[0]
        assert ctx["head"][0][c + 2]["is_visible"] == vals[1]

    assert len(ctx["body"][0]) == 6  # index + 2 hidden + 2 visible + trimming col


def test_no_empty_apply(mi_styler):
    # 45313
    mi_styler.apply(lambda s: ["a:v;"] * 2, subset=[False, False])
    mi_styler._compute()


@pytest.mark.parametrize("format", ["html", "latex", "string"])
def test_output_buffer(mi_styler, format):
    # gh 47053
    with tm.ensure_clean(f"delete_me.{format}") as f:
        getattr(mi_styler, f"to_{format}")(f)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_continuous_rv.py ===
from sympy.concrete.summations import Sum
from sympy.core.function import (Lambda, diff, expand_func)
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (E as e, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import (erf, erfc, erfi, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.utilities.lambdify import lambdify
from sympy.functions.special.error_functions import erfinv
from sympy.functions.special.hyper import meijerg
from sympy.sets.sets import FiniteSet, Complement, Intersection
from sympy.stats import (P, E, where, density, variance, covariance, skewness, kurtosis, median,
                         given, pspace, cdf, characteristic_function, moment_generating_function,
                         ContinuousRV, Arcsin, Benini, Beta, BetaNoncentral, BetaPrime,
                         Cauchy, Chi, ChiSquared, ChiNoncentral, Dagum, Davis, Erlang, ExGaussian,
                         Exponential, ExponentialPower, FDistribution, FisherZ, Frechet, Gamma,
                         GammaInverse, Gompertz, Gumbel, Kumaraswamy, Laplace, Levy, Logistic, LogCauchy,
                         LogLogistic, LogitNormal, LogNormal, Maxwell, Moyal, Nakagami, Normal, GaussianInverse,
                         Pareto, PowerFunction, QuadraticU, RaisedCosine, Rayleigh, Reciprocal, ShiftedGompertz, StudentT,
                         Trapezoidal, Triangular, Uniform, UniformSum, VonMises, Weibull, coskewness,
                         WignerSemicircle, Wald, correlation, moment, cmoment, smoment, quantile,
                         Lomax, BoundedPareto)

from sympy.stats.crv_types import NormalDistribution, ExponentialDistribution, ContinuousDistributionHandmade
from sympy.stats.joint_rv_types import MultivariateLaplaceDistribution, MultivariateNormalDistribution
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDomain
from sympy.stats.compound_rv import CompoundPSpace
from sympy.stats.symbolic_probability import Probability
from sympy.testing.pytest import raises, XFAIL, slow, ignore_warnings
from sympy.core.random import verify_numerically as tn

oo = S.Infinity

x, y, z = map(Symbol, 'xyz')

def test_single_normal():
    mu = Symbol('mu', real=True)
    sigma = Symbol('sigma', positive=True)
    X = Normal('x', 0, 1)
    Y = X*sigma + mu

    assert E(Y) == mu
    assert variance(Y) == sigma**2
    pdf = density(Y)
    x = Symbol('x', real=True)
    assert (pdf(x) ==
            2**S.Half*exp(-(x - mu)**2/(2*sigma**2))/(2*pi**S.Half*sigma))

    assert P(X**2 < 1) == erf(2**S.Half/2)
    ans = quantile(Y)(x)
    assert ans == Complement(Intersection(FiniteSet(
        sqrt(2)*sigma*(sqrt(2)*mu/(2*sigma)+ erfinv(2*x - 1))),
        Interval(-oo, oo)), FiniteSet(mu))
    assert E(X, Eq(X, mu)) == mu

    assert median(X) == FiniteSet(0)
    # issue 8248
    assert X.pspace.compute_expectation(1).doit() == 1


def test_conditional_1d():
    X = Normal('x', 0, 1)
    Y = given(X, X >= 0)
    z = Symbol('z')

    assert density(Y)(z) == 2 * density(X)(z)

    assert Y.pspace.domain.set == Interval(0, oo)
    assert E(Y) == sqrt(2) / sqrt(pi)

    assert E(X**2) == E(Y**2)


def test_ContinuousDomain():
    X = Normal('x', 0, 1)
    assert where(X**2 <= 1).set == Interval(-1, 1)
    assert where(X**2 <= 1).symbol == X.symbol
    assert where(And(X**2 <= 1, X >= 0)).set == Interval(0, 1)
    raises(ValueError, lambda: where(sin(X) > 1))

    Y = given(X, X >= 0)

    assert Y.pspace.domain.set == Interval(0, oo)


def test_multiple_normal():
    X, Y = Normal('x', 0, 1), Normal('y', 0, 1)
    p = Symbol("p", positive=True)

    assert E(X + Y) == 0
    assert variance(X + Y) == 2
    assert variance(X + X) == 4
    assert covariance(X, Y) == 0
    assert covariance(2*X + Y, -X) == -2*variance(X)
    assert skewness(X) == 0
    assert skewness(X + Y) == 0
    assert kurtosis(X) == 3
    assert kurtosis(X+Y) == 3
    assert correlation(X, Y) == 0
    assert correlation(X, X + Y) == correlation(X, X - Y)
    assert moment(X, 2) == 1
    assert cmoment(X, 3) == 0
    assert moment(X + Y, 4) == 12
    assert cmoment(X, 2) == variance(X)
    assert smoment(X*X, 2) == 1
    assert smoment(X + Y, 3) == skewness(X + Y)
    assert smoment(X + Y, 4) == kurtosis(X + Y)
    assert E(X, Eq(X + Y, 0)) == 0
    assert variance(X, Eq(X + Y, 0)) == S.Half
    assert quantile(X)(p) == sqrt(2)*erfinv(2*p - S.One)


def test_symbolic():
    mu1, mu2 = symbols('mu1 mu2', real=True)
    s1, s2 = symbols('sigma1 sigma2', positive=True)
    rate = Symbol('lambda', positive=True)
    X = Normal('x', mu1, s1)
    Y = Normal('y', mu2, s2)
    Z = Exponential('z', rate)
    a, b, c = symbols('a b c', real=True)

    assert E(X) == mu1
    assert E(X + Y) == mu1 + mu2
    assert E(a*X + b) == a*E(X) + b
    assert variance(X) == s1**2
    assert variance(X + a*Y + b) == variance(X) + a**2*variance(Y)

    assert E(Z) == 1/rate
    assert E(a*Z + b) == a*E(Z) + b
    assert E(X + a*Z + b) == mu1 + a/rate + b
    assert median(X) == FiniteSet(mu1)


def test_cdf():
    X = Normal('x', 0, 1)

    d = cdf(X)
    assert P(X < 1) == d(1).rewrite(erfc)
    assert d(0) == S.Half

    d = cdf(X, X > 0)  # given X>0
    assert d(0) == 0

    Y = Exponential('y', 10)
    d = cdf(Y)
    assert d(-5) == 0
    assert P(Y > 3) == 1 - d(3)

    raises(ValueError, lambda: cdf(X + Y))

    Z = Exponential('z', 1)
    f = cdf(Z)
    assert f(z) == Piecewise((1 - exp(-z), z >= 0), (0, True))


def test_characteristic_function():
    X = Uniform('x', 0, 1)

    cf = characteristic_function(X)
    assert cf(1) == -I*(-1 + exp(I))

    Y = Normal('y', 1, 1)
    cf = characteristic_function(Y)
    assert cf(0) == 1
    assert cf(1) == exp(I - S.Half)

    Z = Exponential('z', 5)
    cf = characteristic_function(Z)
    assert cf(0) == 1
    assert cf(1).expand() == Rational(25, 26) + I*5/26

    X = GaussianInverse('x', 1, 1)
    cf = characteristic_function(X)
    assert cf(0) == 1
    assert cf(1) == exp(1 - sqrt(1 - 2*I))

    X = ExGaussian('x', 0, 1, 1)
    cf = characteristic_function(X)
    assert cf(0) == 1
    assert cf(1) == (1 + I)*exp(Rational(-1, 2))/2

    L = Levy('x', 0, 1)
    cf = characteristic_function(L)
    assert cf(0) == 1
    assert cf(1) == exp(-sqrt(2)*sqrt(-I))


def test_moment_generating_function():
    t = symbols('t', positive=True)

    # Symbolic tests
    a, b, c = symbols('a b c')

    mgf = moment_generating_function(Beta('x', a, b))(t)
    assert mgf == hyper((a,), (a + b,), t)

    mgf = moment_generating_function(Chi('x', a))(t)
    assert mgf == sqrt(2)*t*gamma(a/2 + S.Half)*\
        hyper((a/2 + S.Half,), (Rational(3, 2),), t**2/2)/gamma(a/2) +\
        hyper((a/2,), (S.Half,), t**2/2)

    mgf = moment_generating_function(ChiSquared('x', a))(t)
    assert mgf == (1 - 2*t)**(-a/2)

    mgf = moment_generating_function(Erlang('x', a, b))(t)
    assert mgf == (1 - t/b)**(-a)

    mgf = moment_generating_function(ExGaussian("x", a, b, c))(t)
    assert mgf == exp(a*t + b**2*t**2/2)/(1 - t/c)

    mgf = moment_generating_function(Exponential('x', a))(t)
    assert mgf == a/(a - t)

    mgf = moment_generating_function(Gamma('x', a, b))(t)
    assert mgf == (-b*t + 1)**(-a)

    mgf = moment_generating_function(Gumbel('x', a, b))(t)
    assert mgf == exp(b*t)*gamma(-a*t + 1)

    mgf = moment_generating_function(Gompertz('x', a, b))(t)
    assert mgf == b*exp(b)*expint(t/a, b)

    mgf = moment_generating_function(Laplace('x', a, b))(t)
    assert mgf == exp(a*t)/(-b**2*t**2 + 1)

    mgf = moment_generating_function(Logistic('x', a, b))(t)
    assert mgf == exp(a*t)*beta(-b*t + 1, b*t + 1)

    mgf = moment_generating_function(Normal('x', a, b))(t)
    assert mgf == exp(a*t + b**2*t**2/2)

    mgf = moment_generating_function(Pareto('x', a, b))(t)
    assert mgf == b*(-a*t)**b*uppergamma(-b, -a*t)

    mgf = moment_generating_function(QuadraticU('x', a, b))(t)
    assert str(mgf) == ("(3*(t*(-4*b + (a + b)**2) + 4)*exp(b*t) - "
    "3*(t*(a**2 + 2*a*(b - 2) + b**2) + 4)*exp(a*t))/(t**2*(a - b)**3)")

    mgf = moment_generating_function(RaisedCosine('x', a, b))(t)
    assert mgf == pi**2*exp(a*t)*sinh(b*t)/(b*t*(b**2*t**2 + pi**2))

    mgf = moment_generating_function(Rayleigh('x', a))(t)
    assert mgf == sqrt(2)*sqrt(pi)*a*t*(erf(sqrt(2)*a*t/2) + 1)\
        *exp(a**2*t**2/2)/2 + 1

    mgf = moment_generating_function(Triangular('x', a, b, c))(t)
    assert str(mgf) == ("(-2*(-a + b)*exp(c*t) + 2*(-a + c)*exp(b*t) + "
    "2*(b - c)*exp(a*t))/(t**2*(-a + b)*(-a + c)*(b - c))")

    mgf = moment_generating_function(Uniform('x', a, b))(t)
    assert mgf == (-exp(a*t) + exp(b*t))/(t*(-a + b))

    mgf = moment_generating_function(UniformSum('x', a))(t)
    assert mgf == ((exp(t) - 1)/t)**a

    mgf = moment_generating_function(WignerSemicircle('x', a))(t)
    assert mgf == 2*besseli(1, a*t)/(a*t)

    # Numeric tests

    mgf = moment_generating_function(Beta('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 1) == hyper((2,), (3,), 1)/2

    mgf = moment_generating_function(Chi('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == sqrt(2)*hyper((1,), (Rational(3, 2),), S.Half
    )/sqrt(pi) + hyper((Rational(3, 2),), (Rational(3, 2),), S.Half) + 2*sqrt(2)*hyper((2,),
    (Rational(5, 2),), S.Half)/(3*sqrt(pi))

    mgf = moment_generating_function(ChiSquared('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == I

    mgf = moment_generating_function(Erlang('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(ExGaussian("x", 0, 1, 1))(t)
    assert mgf.diff(t).subs(t, 2) == -exp(2)

    mgf = moment_generating_function(Exponential('x', 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(Gamma('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(Gumbel('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == EulerGamma + 1

    mgf = moment_generating_function(Gompertz('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 1) == -e*meijerg(((), (1, 1)),
    ((0, 0, 0), ()), 1)

    mgf = moment_generating_function(Laplace('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == 1

    mgf = moment_generating_function(Logistic('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == beta(1, 1)

    mgf = moment_generating_function(Normal('x', 0, 1))(t)
    assert mgf.diff(t).subs(t, 1) == exp(S.Half)

    mgf = moment_generating_function(Pareto('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 0) == expint(1, 0)

    mgf = moment_generating_function(QuadraticU('x', 1, 2))(t)
    assert mgf.diff(t).subs(t, 1) == -12*e - 3*exp(2)

    mgf = moment_generating_function(RaisedCosine('x', 1, 1))(t)
    assert mgf.diff(t).subs(t, 1) == -2*e*pi**2*sinh(1)/\
    (1 + pi**2)**2 + e*pi**2*cosh(1)/(1 + pi**2)

    mgf = moment_generating_function(Rayleigh('x', 1))(t)
    assert mgf.diff(t).subs(t, 0) == sqrt(2)*sqrt(pi)/2

    mgf = moment_generating_function(Triangular('x', 1, 3, 2))(t)
    assert mgf.diff(t).subs(t, 1) == -e + exp(3)

    mgf = moment_generating_function(Uniform('x', 0, 1))(t)
    assert mgf.diff(t).subs(t, 1) == 1

    mgf = moment_generating_function(UniformSum('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == 1

    mgf = moment_generating_function(WignerSemicircle('x', 1))(t)
    assert mgf.diff(t).subs(t, 1) == -2*besseli(1, 1) + besseli(2, 1) +\
        besseli(0, 1)


def test_ContinuousRV():
    pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))  # Normal distribution
    # X and Y should be equivalent
    X = ContinuousRV(x, pdf, check=True)
    Y = Normal('y', 0, 1)

    assert variance(X) == variance(Y)
    assert P(X > 0) == P(Y > 0)
    Z = ContinuousRV(z, exp(-z), set=Interval(0, oo))
    assert Z.pspace.domain.set == Interval(0, oo)
    assert E(Z) == 1
    assert P(Z > 5) == exp(-5)
    raises(ValueError, lambda: ContinuousRV(z, exp(-z), set=Interval(0, 10), check=True))

    # the correct pdf for Gamma(k, theta) but the integral in `check`
    # integrates to something equivalent to 1 and not to 1 exactly
    _x, k, theta = symbols("x k theta", positive=True)
    pdf = 1/(gamma(k)*theta**k)*_x**(k-1)*exp(-_x/theta)
    X = ContinuousRV(_x, pdf, set=Interval(0, oo))
    Y = Gamma('y', k, theta)
    assert (E(X) - E(Y)).simplify() == 0
    assert (variance(X) - variance(Y)).simplify() == 0


def test_arcsin():

    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    X = Arcsin('x', a, b)
    assert density(X)(x) == 1/(pi*sqrt((-x + b)*(x - a)))
    assert cdf(X)(x) == Piecewise((0, a > x),
                            (2*asin(sqrt((-a + x)/(-a + b)))/pi, b >= x),
                            (1, True))
    assert pspace(X).domain.set == Interval(a, b)

def test_benini():
    alpha = Symbol("alpha", positive=True)
    beta = Symbol("beta", positive=True)
    sigma = Symbol("sigma", positive=True)
    X = Benini('x', alpha, beta, sigma)

    assert density(X)(x) == ((alpha/x + 2*beta*log(x/sigma)/x)
                          *exp(-alpha*log(x/sigma) - beta*log(x/sigma)**2))

    assert pspace(X).domain.set == Interval(sigma, oo)
    raises(NotImplementedError, lambda: moment_generating_function(X))
    alpha = Symbol("alpha", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    beta = Symbol("beta", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    alpha = Symbol("alpha", positive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

    beta = Symbol("beta", positive=True)
    sigma = Symbol("sigma", nonpositive=True)
    raises(ValueError, lambda: Benini('x', alpha, beta, sigma))

def test_beta():
    a, b = symbols('alpha beta', positive=True)
    B = Beta('x', a, b)

    assert pspace(B).domain.set == Interval(0, 1)
    assert characteristic_function(B)(x) == hyper((a,), (a + b,), I*x)
    assert density(B)(x) == x**(a - 1)*(1 - x)**(b - 1)/beta(a, b)

    assert simplify(E(B)) == a / (a + b)
    assert simplify(variance(B)) == a*b / (a**3 + 3*a**2*b + a**2 + 3*a*b**2 + 2*a*b + b**3 + b**2)

    # Full symbolic solution is too much, test with numeric version
    a, b = 1, 2
    B = Beta('x', a, b)
    assert expand_func(E(B)) == a / S(a + b)
    assert expand_func(variance(B)) == (a*b) / S((a + b)**2 * (a + b + 1))
    assert median(B) == FiniteSet(1 - 1/sqrt(2))

def test_beta_noncentral():
    a, b = symbols('a b', positive=True)
    c = Symbol('c', nonnegative=True)
    _k = Dummy('k')

    X = BetaNoncentral('x', a, b, c)

    assert pspace(X).domain.set == Interval(0, 1)

    dens = density(X)
    z = Symbol('z')

    res = Sum( z**(_k + a - 1)*(c/2)**_k*(1 - z)**(b - 1)*exp(-c/2)/
               (beta(_k + a, b)*factorial(_k)), (_k, 0, oo))
    assert dens(z).dummy_eq(res)

    # BetaCentral should not raise if the assumptions
    # on the symbols can not be determined
    a, b, c = symbols('a b c')
    assert BetaNoncentral('x', a, b, c)

    a = Symbol('a', positive=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

    a = Symbol('a', positive=True)
    b = Symbol('b', positive=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

    a = Symbol('a', positive=True)
    b = Symbol('b', positive=True)
    c = Symbol('c', nonnegative=False, real=True)
    raises(ValueError, lambda: BetaNoncentral('x', a, b, c))

def test_betaprime():
    alpha = Symbol("alpha", positive=True)

    betap = Symbol("beta", positive=True)

    X = BetaPrime('x', alpha, betap)
    assert density(X)(x) == x**(alpha - 1)*(x + 1)**(-alpha - betap)/beta(alpha, betap)

    alpha = Symbol("alpha", nonpositive=True)
    raises(ValueError, lambda: BetaPrime('x', alpha, betap))

    alpha = Symbol("alpha", positive=True)
    betap = Symbol("beta", nonpositive=True)
    raises(ValueError, lambda: BetaPrime('x', alpha, betap))
    X = BetaPrime('x', 1, 1)
    assert median(X) == FiniteSet(1)


def test_BoundedPareto():
    L, H = symbols('L, H', negative=True)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))
    L, H = symbols('L, H', real=False)
    raises(ValueError, lambda: BoundedPareto('X', 1, L, H))
    L, H = symbols('L, H', positive=True)
    raises(ValueError, lambda: BoundedPareto('X', -1, L, H))

    X = BoundedPareto('X', 2, L, H)
    assert X.pspace.domain.set == Interval(L, H)
    assert density(X)(x) == 2*L**2/(x**3*(1 - L**2/H**2))
    assert cdf(X)(x) == Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) \
                            + H**2/(H**2 - L**2), L <= x), (0, True))
    assert E(X).simplify() == 2*H*L/(H + L)
    X = BoundedPareto('X', 1, 2, 4)
    assert E(X).simplify() == log(16)
    assert median(X) == FiniteSet(Rational(8, 3))
    assert variance(X).simplify() == 8 - 16*log(2)**2


def test_cauchy():
    x0 = Symbol("x0", real=True)
    gamma = Symbol("gamma", positive=True)
    p = Symbol("p", positive=True)

    X = Cauchy('x', x0, gamma)
    # Tests the characteristic function
    assert characteristic_function(X)(x) == exp(-gamma*Abs(x) + I*x*x0)
    raises(NotImplementedError, lambda: moment_generating_function(X))
    assert density(X)(x) == 1/(pi*gamma*(1 + (x - x0)**2/gamma**2))
    assert diff(cdf(X)(x), x) == density(X)(x)
    assert quantile(X)(p) == gamma*tan(pi*(p - S.Half)) + x0

    x1 = Symbol("x1", real=False)
    raises(ValueError, lambda: Cauchy('x', x1, gamma))
    gamma = Symbol("gamma", nonpositive=True)
    raises(ValueError, lambda: Cauchy('x', x0, gamma))
    assert median(X) == FiniteSet(x0)

def test_chi():
    from sympy.core.numbers import I
    k = Symbol("k", integer=True)

    X = Chi('x', k)
    assert density(X)(x) == 2**(-k/2 + 1)*x**(k - 1)*exp(-x**2/2)/gamma(k/2)

    # Tests the characteristic function
    assert characteristic_function(X)(x) == sqrt(2)*I*x*gamma(k/2 + S(1)/2)*hyper((k/2 + S(1)/2,),
                                            (S(3)/2,), -x**2/2)/gamma(k/2) + hyper((k/2,), (S(1)/2,), -x**2/2)

    # Tests the moment generating function
    assert moment_generating_function(X)(x) == sqrt(2)*x*gamma(k/2 + S(1)/2)*hyper((k/2 + S(1)/2,),
                                                (S(3)/2,), x**2/2)/gamma(k/2) + hyper((k/2,), (S(1)/2,), x**2/2)

    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: Chi('x', k))

    k = Symbol("k", integer=False, positive=True)
    raises(ValueError, lambda: Chi('x', k))

def test_chi_noncentral():
    k = Symbol("k", integer=True)
    l = Symbol("l")

    X = ChiNoncentral("x", k, l)
    assert density(X)(x) == (x**k*l*(x*l)**(-k/2)*
                          exp(-x**2/2 - l**2/2)*besseli(k/2 - 1, x*l))

    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))

    k = Symbol("k", integer=True, positive=True)
    l = Symbol("l", nonpositive=True)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))

    k = Symbol("k", integer=False)
    l = Symbol("l", positive=True)
    raises(ValueError, lambda: ChiNoncentral('x', k, l))


def test_chi_squared():
    k = Symbol("k", integer=True)
    X = ChiSquared('x', k)

    # Tests the characteristic function
    assert characteristic_function(X)(x) == ((-2*I*x + 1)**(-k/2))

    assert density(X)(x) == 2**(-k/2)*x**(k/2 - 1)*exp(-x/2)/gamma(k/2)
    assert cdf(X)(x) == Piecewise((lowergamma(k/2, x/2)/gamma(k/2), x >= 0), (0, True))
    assert E(X) == k
    assert variance(X) == 2*k

    X = ChiSquared('x', 15)
    assert cdf(X)(3) == -14873*sqrt(6)*exp(Rational(-3, 2))/(5005*sqrt(pi)) + erf(sqrt(6)/2)

    k = Symbol("k", integer=True, positive=False)
    raises(ValueError, lambda: ChiSquared('x', k))

    k = Symbol("k", integer=False, positive=True)
    raises(ValueError, lambda: ChiSquared('x', k))


def test_dagum():
    p = Symbol("p", positive=True)
    b = Symbol("b", positive=True)
    a = Symbol("a", positive=True)

    X = Dagum('x', p, a, b)
    assert density(X)(x) == a*p*(x/b)**(a*p)*((x/b)**a + 1)**(-p - 1)/x
    assert cdf(X)(x) == Piecewise(((1 + (x/b)**(-a))**(-p), x >= 0),
                                    (0, True))

    p = Symbol("p", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))

    p = Symbol("p", positive=True)
    b = Symbol("b", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))

    b = Symbol("b", positive=True)
    a = Symbol("a", nonpositive=True)
    raises(ValueError, lambda: Dagum('x', p, a, b))
    X = Dagum('x', 1, 1, 1)
    assert median(X) == FiniteSet(1)

def test_davis():
    b = Symbol("b", positive=True)
    n = Symbol("n", positive=True)
    mu = Symbol("mu", positive=True)

    X = Davis('x', b, n, mu)
    dividend = b**n*(x - mu)**(-1-n)
    divisor = (exp(b/(x-mu))-1)*(gamma(n)*zeta(n))
    assert density(X)(x) == dividend/divisor


def test_erlang():
    k = Symbol("k", integer=True, positive=True)
    l = Symbol("l", positive=True)

    X = Erlang("x", k, l)
    assert density(X)(x) == x**(k - 1)*l**k*exp(-x*l)/gamma(k)
    assert cdf(X)(x) == Piecewise((lowergamma(k, l*x)/gamma(k), x > 0),
                               (0, True))


def test_exgaussian():
    m, z = symbols("m, z")
    s, l = symbols("s, l", positive=True)
    X = ExGaussian("x", m, s, l)

    assert density(X)(z) == l*exp(l*(l*s**2 + 2*m - 2*z)/2) *\
        erfc(sqrt(2)*(l*s**2 + m - z)/(2*s))/2

    # Note: actual_output simplifies to expected_output.
    # Ideally cdf(X)(z) would return expected_output
    # expected_output = (erf(sqrt(2)*(l*s**2 + m - z)/(2*s)) - 1)*exp(l*(l*s**2 + 2*m - 2*z)/2)/2 - erf(sqrt(2)*(m - z)/(2*s))/2 + S.Half
    u = l*(z - m)
    v = l*s
    GaussianCDF1 = cdf(Normal('x', 0, v))(u)
    GaussianCDF2 = cdf(Normal('x', v**2, v))(u)
    actual_output = GaussianCDF1 - exp(-u + (v**2/2) + log(GaussianCDF2))
    assert cdf(X)(z) == actual_output
    # assert simplify(actual_output) == expected_output

    assert variance(X).expand() == s**2 + l**(-2)

    assert skewness(X).expand() == 2/(l**3*s**2*sqrt(s**2 + l**(-2)) + l *
                                      sqrt(s**2 + l**(-2)))


@slow
def test_exponential():
    rate = Symbol('lambda', positive=True)
    X = Exponential('x', rate)
    p = Symbol("p", positive=True, real=True)

    assert E(X) == 1/rate
    assert variance(X) == 1/rate**2
    assert skewness(X) == 2
    assert skewness(X) == smoment(X, 3)
    assert kurtosis(X) == 9
    assert kurtosis(X) == smoment(X, 4)
    assert smoment(2*X, 4) == smoment(X, 4)
    assert moment(X, 3) == 3*2*1/rate**3
    assert P(X > 0) is S.One
    assert P(X > 1) == exp(-rate)
    assert P(X > 10) == exp(-10*rate)
    assert quantile(X)(p) == -log(1-p)/rate

    assert where(X <= 1).set == Interval(0, 1)
    Y = Exponential('y', 1)
    assert median(Y) == FiniteSet(log(2))
    #Test issue 9970
    z = Dummy('z')
    assert P(X > z) == exp(-z*rate)
    assert P(X < z) == 0
    #Test issue 10076 (Distribution with interval(0,oo))
    x = Symbol('x')
    _z = Dummy('_z')
    b = SingleContinuousPSpace(x, ExponentialDistribution(2))

    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        expected1 = Integral(2*exp(-2*_z), (_z, 3, oo))
        assert b.probability(x > 3, evaluate=False).rewrite(Integral).dummy_eq(expected1)

        expected2 = Integral(2*exp(-2*_z), (_z, 0, 4))
        assert b.probability(x < 4, evaluate=False).rewrite(Integral).dummy_eq(expected2)
    Y = Exponential('y', 2*rate)
    assert coskewness(X, X, X) == skewness(X)
    assert coskewness(X, Y + rate*X, Y + 2*rate*X) == \
                        4/(sqrt(1 + 1/(4*rate**2))*sqrt(4 + 1/(4*rate**2)))
    assert coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3) == \
                        sqrt(170)*Rational(9, 85)

def test_exponential_power():
    mu = Symbol('mu')
    z = Symbol('z')
    alpha = Symbol('alpha', positive=True)
    beta = Symbol('beta', positive=True)

    X = ExponentialPower('x', mu, alpha, beta)

    assert density(X)(z) == beta*exp(-(Abs(mu - z)/alpha)
                                     ** beta)/(2*alpha*gamma(1/beta))
    assert cdf(X)(z) == S.Half + lowergamma(1/beta,
                            (Abs(mu - z)/alpha)**beta)*sign(-mu + z)/\
                                (2*gamma(1/beta))


def test_f_distribution():
    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", positive=True)

    X = FDistribution("x", d1, d2)

    assert density(X)(x) == (d2**(d2/2)*sqrt((d1*x)**d1*(d1*x + d2)**(-d1 - d2))
                             /(x*beta(d1/2, d2/2)))

    raises(NotImplementedError, lambda: moment_generating_function(X))
    d1 = Symbol("d1", nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d1))

    d1 = Symbol("d1", positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d1))

    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", nonpositive=True)
    raises(ValueError, lambda: FDistribution('x', d1, d2))

    d2 = Symbol("d2", positive=True, integer=False)
    raises(ValueError, lambda: FDistribution('x', d1, d2))


def test_fisher_z():
    d1 = Symbol("d1", positive=True)
    d2 = Symbol("d2", positive=True)

    X = FisherZ("x", d1, d2)
    assert density(X)(x) == (2*d1**(d1/2)*d2**(d2/2)*(d1*exp(2*x) + d2)
                             **(-d1/2 - d2/2)*exp(d1*x)/beta(d1/2, d2/2))

def test_frechet():
    a = Symbol("a", positive=True)
    s = Symbol("s", positive=True)
    m = Symbol("m", real=True)

    X = Frechet("x", a, s=s, m=m)
    assert density(X)(x) == a*((x - m)/s)**(-a - 1)*exp(-((x - m)/s)**(-a))/s
    assert cdf(X)(x) == Piecewise((exp(-((-m + x)/s)**(-a)), m <= x), (0, True))

@slow
def test_gamma():
    k = Symbol("k", positive=True)
    theta = Symbol("theta", positive=True)

    X = Gamma('x', k, theta)

    # Tests characteristic function
    assert characteristic_function(X)(x) == ((-I*theta*x + 1)**(-k))

    assert density(X)(x) == x**(k - 1)*theta**(-k)*exp(-x/theta)/gamma(k)
    assert cdf(X, meijerg=True)(z) == Piecewise(
            (-k*lowergamma(k, 0)/gamma(k + 1) +
                k*lowergamma(k, z/theta)/gamma(k + 1), z >= 0),
            (0, True))

    # assert simplify(variance(X)) == k*theta**2  # handled numerically below
    assert E(X) == moment(X, 1)

    k, theta = symbols('k theta', positive=True)
    X = Gamma('x', k, theta)
    assert E(X) == k*theta
    assert variance(X) == k*theta**2
    assert skewness(X).expand() == 2/sqrt(k)
    assert kurtosis(X).expand() == 3 + 6/k

    Y = Gamma('y', 2*k, 3*theta)
    assert coskewness(X, theta*X + Y, k*X + Y).simplify() == \
        2*531441**(-k)*sqrt(k)*theta*(3*3**(12*k) - 2*531441**k) \
        /(sqrt(k**2 + 18)*sqrt(theta**2 + 18))

def test_gamma_inverse():
    a = Symbol("a", positive=True)
    b = Symbol("b", positive=True)
    X = GammaInverse("x", a, b)
    assert density(X)(x) == x**(-a - 1)*b**a*exp(-b/x)/gamma(a)
    assert cdf(X)(x) == Piecewise((uppergamma(a, b/x)/gamma(a), x > 0), (0, True))
    assert characteristic_function(X)(x) == 2 * (-I*b*x)**(a/2) \
            * besselk(a, 2*sqrt(b)*sqrt(-I*x))/gamma(a)
    raises(NotImplementedError, lambda: moment_generating_function(X))

def test_gompertz():
    b = Symbol("b", positive=True)
    eta = Symbol("eta", positive=True)

    X = Gompertz("x", b, eta)

    assert density(X)(x) == b*eta*exp(eta)*exp(b*x)*exp(-eta*exp(b*x))
    assert cdf(X)(x) == 1 - exp(eta)*exp(-eta*exp(b*x))
    assert diff(cdf(X)(x), x) == density(X)(x)


def test_gumbel():
    beta = Symbol("beta", positive=True)
    mu = Symbol("mu")
    x = Symbol("x")
    y = Symbol("y")
    X = Gumbel("x", beta, mu)
    Y = Gumbel("y", beta, mu, minimum=True)
    assert density(X)(x).expand() == \
    exp(mu/beta)*exp(-x/beta)*exp(-exp(mu/beta)*exp(-x/beta))/beta
    assert density(Y)(y).expand() == \
    exp(-mu/beta)*exp(y/beta)*exp(-exp(-mu/beta)*exp(y/beta))/beta
    assert cdf(X)(x).expand() == \
    exp(-exp(mu/beta)*exp(-x/beta))
    assert characteristic_function(X)(x) == exp(I*mu*x)*gamma(-I*beta*x + 1)

def test_kumaraswamy():
    a = Symbol("a", positive=True)
    b = Symbol("b", positive=True)

    X = Kumaraswamy("x", a, b)
    assert density(X)(x) == x**(a - 1)*a*b*(-x**a + 1)**(b - 1)
    assert cdf(X)(x) == Piecewise((0, x < 0),
                                (-(-x**a + 1)**b + 1, x <= 1),
                                (1, True))


def test_laplace():
    mu = Symbol("mu")
    b = Symbol("b", positive=True)

    X = Laplace('x', mu, b)

    #Tests characteristic_function
    assert characteristic_function(X)(x) == (exp(I*mu*x)/(b**2*x**2 + 1))

    assert density(X)(x) == exp(-Abs(x - mu)/b)/(2*b)
    assert cdf(X)(x) == Piecewise((exp((-mu + x)/b)/2, mu > x),
                            (-exp((mu - x)/b)/2 + 1, True))
    X = Laplace('x', [1, 2], [[1, 0], [0, 1]])
    assert isinstance(pspace(X).distribution, MultivariateLaplaceDistribution)

def test_levy():
    mu = Symbol("mu", real=True)
    c = Symbol("c", positive=True)

    X = Levy('x', mu, c)
    assert X.pspace.domain.set == Interval(mu, oo)
    assert density(X)(x) == sqrt(c/(2*pi))*exp(-c/(2*(x - mu)))/((x - mu)**(S.One + S.Half))
    assert cdf(X)(x) == erfc(sqrt(c/(2*(x - mu))))

    raises(NotImplementedError, lambda: moment_generating_function(X))
    mu = Symbol("mu", real=False)
    raises(ValueError, lambda: Levy('x',mu,c))

    c = Symbol("c", nonpositive=True)
    raises(ValueError, lambda: Levy('x',mu,c))

    mu = Symbol("mu", real=True)
    raises(ValueError, lambda: Levy('x',mu,c))

def test_logcauchy():
    mu = Symbol("mu", positive=True)
    sigma = Symbol("sigma", positive=True)

    X = LogCauchy("x", mu, sigma)

    assert density(X)(x) == sigma/(x*pi*(sigma**2 + (-mu + log(x))**2))
    assert cdf(X)(x) == atan((log(x) - mu)/sigma)/pi + S.Half


def test_logistic():
    mu = Symbol("mu", real=True)
    s = Symbol("s", positive=True)
    p = Symbol("p", positive=True)

    X = Logistic('x', mu, s)

    #Tests characteristics_function
    assert characteristic_function(X)(x) == \
           (Piecewise((pi*s*x*exp(I*mu*x)/sinh(pi*s*x), Ne(x, 0)), (1, True)))

    assert density(X)(x) == exp((-x + mu)/s)/(s*(exp((-x + mu)/s) + 1)**2)
    assert cdf(X)(x) == 1/(exp((mu - x)/s) + 1)
    assert quantile(X)(p) == mu - s*log(-S.One + 1/p)

def test_loglogistic():
    a, b = symbols('a b')
    assert LogLogistic('x', a, b)

    a = Symbol('a', negative=True)
    b = Symbol('b', positive=True)
    raises(ValueError, lambda: LogLogistic('x', a, b))

    a = Symbol('a', positive=True)
    b = Symbol('b', negative=True)
    raises(ValueError, lambda: LogLogistic('x', a, b))

    a, b, z, p = symbols('a b z p', positive=True)
    X = LogLogistic('x', a, b)
    assert density(X)(z) == b*(z/a)**(b - 1)/(a*((z/a)**b + 1)**2)
    assert cdf(X)(z) == 1/(1 + (z/a)**(-b))
    assert quantile(X)(p) == a*(p/(1 - p))**(1/b)

    # Expectation
    assert E(X) == Piecewise((S.NaN, b <= 1), (pi*a/(b*sin(pi/b)), True))
    b = symbols('b', prime=True) # b > 1
    X = LogLogistic('x', a, b)
    assert E(X) == pi*a/(b*sin(pi/b))
    X = LogLogistic('x', 1, 2)
    assert median(X) == FiniteSet(1)

def test_logitnormal():
    mu = Symbol('mu', real=True)
    s = Symbol('s', positive=True)
    X = LogitNormal('x', mu, s)
    x = Symbol('x')

    assert density(X)(x) == sqrt(2)*exp(-(-mu + log(x/(1 - x)))**2/(2*s**2))/(2*sqrt(pi)*s*x*(1 - x))
    assert cdf(X)(x) == erf(sqrt(2)*(-mu + log(x/(1 - x)))/(2*s))/2 + S(1)/2

def test_lognormal():
    mean = Symbol('mu', real=True)
    std = Symbol('sigma', positive=True)
    X = LogNormal('x', mean, std)
    # The sympy integrator can't do this too well
    #assert E(X) == exp(mean+std**2/2)
    #assert variance(X) == (exp(std**2)-1) * exp(2*mean + std**2)

    # The sympy integrator can't do this too well
    #assert E(X) ==
    raises(NotImplementedError, lambda: moment_generating_function(X))
    mu = Symbol("mu", real=True)
    sigma = Symbol("sigma", positive=True)

    X = LogNormal('x', mu, sigma)
    assert density(X)(x) == (sqrt(2)*exp(-(-mu + log(x))**2
                                    /(2*sigma**2))/(2*x*sqrt(pi)*sigma))
    # Tests cdf
    assert cdf(X)(x) == Piecewise(
                        (erf(sqrt(2)*(-mu + log(x))/(2*sigma))/2
                        + S(1)/2, x > 0), (0, True))

    X = LogNormal('x', 0, 1)  # Mean 0, standard deviation 1
    assert density(X)(x) == sqrt(2)*exp(-log(x)**2/2)/(2*x*sqrt(pi))


def test_Lomax():
    a, l = symbols('a, l', negative=True)
    raises(ValueError, lambda: Lomax('X', a, l))
    a, l = symbols('a, l', real=False)
    raises(ValueError, lambda: Lomax('X', a, l))

    a, l = symbols('a, l', positive=True)
    X = Lomax('X', a, l)
    assert X.pspace.domain.set == Interval(0, oo)
    assert density(X)(x) == a*(1 + x/l)**(-a - 1)/l
    assert cdf(X)(x) == Piecewise((1 - (1 + x/l)**(-a), x >= 0), (0, True))
    a = 3
    X = Lomax('X', a, l)
    assert E(X) == l/2
    assert median(X) == FiniteSet(l*(-1 + 2**Rational(1, 3)))
    assert variance(X) == 3*l**2/4


def test_maxwell():
    a = Symbol("a", positive=True)

    X = Maxwell('x', a)

    assert density(X)(x) == (sqrt(2)*x**2*exp(-x**2/(2*a**2))/
        (sqrt(pi)*a**3))
    assert E(X) == 2*sqrt(2)*a/sqrt(pi)
    assert variance(X) == -8*a**2/pi + 3*a**2
    assert cdf(X)(x) == erf(sqrt(2)*x/(2*a)) - sqrt(2)*x*exp(-x**2/(2*a**2))/(sqrt(pi)*a)
    assert diff(cdf(X)(x), x) == density(X)(x)


@slow
def test_Moyal():
    mu = Symbol('mu',real=False)
    sigma = Symbol('sigma', positive=True)
    raises(ValueError, lambda: Moyal('M',mu, sigma))

    mu = Symbol('mu', real=True)
    sigma = Symbol('sigma', negative=True)
    raises(ValueError, lambda: Moyal('M',mu, sigma))

    sigma = Symbol('sigma', positive=True)
    M = Moyal('M', mu, sigma)
    assert density(M)(z) == sqrt(2)*exp(-exp((mu - z)/sigma)/2
                        - (-mu + z)/(2*sigma))/(2*sqrt(pi)*sigma)
    assert cdf(M)(z).simplify() == 1 - erf(sqrt(2)*exp((mu - z)/(2*sigma))/2)
    assert characteristic_function(M)(z) == 2**(-I*sigma*z)*exp(I*mu*z) \
                        *gamma(-I*sigma*z + Rational(1, 2))/sqrt(pi)
    assert E(M) == mu + EulerGamma*sigma + sigma*log(2)
    assert moment_generating_function(M)(z) == 2**(-sigma*z)*exp(mu*z) \
                        *gamma(-sigma*z + Rational(1, 2))/sqrt(pi)


def test_nakagami():
    mu = Symbol("mu", positive=True)
    omega = Symbol("omega", positive=True)

    X = Nakagami('x', mu, omega)
    assert density(X)(x) == (2*x**(2*mu - 1)*mu**mu*omega**(-mu)
                                *exp(-x**2*mu/omega)/gamma(mu))
    assert simplify(E(X)) == (sqrt(mu)*sqrt(omega)
                                            *gamma(mu + S.Half)/gamma(mu + 1))
    assert simplify(variance(X)) == (
    omega - omega*gamma(mu + S.Half)**2/(gamma(mu)*gamma(mu + 1)))
    assert cdf(X)(x) == Piecewise(
                                (lowergamma(mu, mu*x**2/omega)/gamma(mu), x > 0),
                                (0, True))
    X = Nakagami('x', 1, 1)
    assert median(X) == FiniteSet(sqrt(log(2)))

def test_gaussian_inverse():
    # test for symbolic parameters
    a, b = symbols('a b')
    assert GaussianInverse('x', a, b)

    # Inverse Gaussian distribution is also known as Wald distribution
    # `GaussianInverse` can also be referred by the name `Wald`
    a, b, z = symbols('a b z')
    X = Wald('x', a, b)
    assert density(X)(z) == sqrt(2)*sqrt(b/z**3)*exp(-b*(-a + z)**2/(2*a**2*z))/(2*sqrt(pi))

    a, b = symbols('a b', positive=True)
    z = Symbol('z', positive=True)

    X = GaussianInverse('x', a, b)
    assert density(X)(z) == sqrt(2)*sqrt(b)*sqrt(z**(-3))*exp(-b*(-a + z)**2/(2*a**2*z))/(2*sqrt(pi))
    assert E(X) == a
    assert variance(X).expand() == a**3/b
    assert cdf(X)(z) == (S.Half - erf(sqrt(2)*sqrt(b)*(1 + z/a)/(2*sqrt(z)))/2)*exp(2*b/a) +\
         erf(sqrt(2)*sqrt(b)*(-1 + z/a)/(2*sqrt(z)))/2 + S.Half

    a = symbols('a', nonpositive=True)
    raises(ValueError, lambda: GaussianInverse('x', a, b))

    a = symbols('a', positive=True)
    b = symbols('b', nonpositive=True)
    raises(ValueError, lambda: GaussianInverse('x', a, b))

def test_pareto():
    xm, beta = symbols('xm beta', positive=True)
    alpha = beta + 5
    X = Pareto('x', xm, alpha)

    dens = density(X)

    #Tests cdf function
    assert cdf(X)(x) == \
           Piecewise((-x**(-beta - 5)*xm**(beta + 5) + 1, x >= xm), (0, True))

    #Tests characteristic_function
    assert characteristic_function(X)(x) == \
           ((-I*x*xm)**(beta + 5)*(beta + 5)*uppergamma(-beta - 5, -I*x*xm))

    assert dens(x) == x**(-(alpha + 1))*xm**(alpha)*(alpha)

    assert simplify(E(X)) == alpha*xm/(alpha-1)

    # computation of taylor series for MGF still too slow
    #assert simplify(variance(X)) == xm**2*alpha / ((alpha-1)**2*(alpha-2))


def test_pareto_numeric():
    xm, beta = 3, 2
    alpha = beta + 5
    X = Pareto('x', xm, alpha)

    assert E(X) == alpha*xm/S(alpha - 1)
    assert variance(X) == xm**2*alpha / S((alpha - 1)**2*(alpha - 2))
    assert median(X) == FiniteSet(3*2**Rational(1, 7))
    # Skewness tests too slow. Try shortcutting function?


def test_PowerFunction():
    alpha = Symbol("alpha", nonpositive=True)
    a, b = symbols('a, b', real=True)
    raises (ValueError, lambda: PowerFunction('x', alpha, a, b))

    a, b = symbols('a, b', real=False)
    raises (ValueError, lambda: PowerFunction('x', alpha, a, b))

    alpha = Symbol("alpha", positive=True)
    a, b = symbols('a, b', real=True)
    raises (ValueError, lambda: PowerFunction('x', alpha, 5, 2))

    X = PowerFunction('X', 2, a, b)
    assert density(X)(z) == (-2*a + 2*z)/(-a + b)**2
    assert cdf(X)(z) == Piecewise((a**2/(a**2 - 2*a*b + b**2) -
        2*a*z/(a**2 - 2*a*b + b**2) + z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))

    X = PowerFunction('X', 2, 0, 1)
    assert density(X)(z) == 2*z
    assert cdf(X)(z) == Piecewise((z**2, z >= 0), (0,True))
    assert E(X) == Rational(2,3)
    assert P(X < 0) == 0
    assert P(X < 1) == 1
    assert median(X) == FiniteSet(1/sqrt(2))

def test_raised_cosine():
    mu = Symbol("mu", real=True)
    s = Symbol("s", positive=True)

    X = RaisedCosine("x", mu, s)

    assert pspace(X).domain.set == Interval(mu - s, mu + s)
    #Tests characteristics_function
    assert characteristic_function(X)(x) == \
           Piecewise((exp(-I*pi*mu/s)/2, Eq(x, -pi/s)), (exp(I*pi*mu/s)/2, Eq(x, pi/s)), (pi**2*exp(I*mu*x)*sin(s*x)/(s*x*(-s**2*x**2 + pi**2)), True))

    assert density(X)(x) == (Piecewise(((cos(pi*(x - mu)/s) + 1)/(2*s),
                          And(x <= mu + s, mu - s <= x)), (0, True)))


def test_rayleigh():
    sigma = Symbol("sigma", positive=True)

    X = Rayleigh('x', sigma)

    #Tests characteristic_function
    assert characteristic_function(X)(x) == (-sqrt(2)*sqrt(pi)*sigma*x*(erfi(sqrt(2)*sigma*x/2) - I)*exp(-sigma**2*x**2/2)/2 + 1)

    assert density(X)(x) ==  x*exp(-x**2/(2*sigma**2))/sigma**2
    assert E(X) == sqrt(2)*sqrt(pi)*sigma/2
    assert variance(X) == -pi*sigma**2/2 + 2*sigma**2
    assert cdf(X)(x) == 1 - exp(-x**2/(2*sigma**2))
    assert diff(cdf(X)(x), x) == density(X)(x)

def test_reciprocal():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    X = Reciprocal('x', a, b)
    assert density(X)(x) == 1/(x*(-log(a) + log(b)))
    assert cdf(X)(x) == Piecewise((log(a)/(log(a) - log(b)) - log(x)/(log(a) - log(b)), a <= x), (0, True))
    X = Reciprocal('x', 5, 30)

    assert E(X) == 25/(log(30) - log(5))
    assert P(X < 4) == S.Zero
    assert P(X < 20) == log(20) / (log(30) - log(5)) - log(5) / (log(30) - log(5))
    assert cdf(X)(10) == log(10) / (log(30) - log(5)) - log(5) / (log(30) - log(5))

    a = symbols('a', nonpositive=True)
    raises(ValueError, lambda: Reciprocal('x', a, b))

    a = symbols('a', positive=True)
    b = symbols('b', positive=True)
    raises(ValueError, lambda: Reciprocal('x', a + b, a))

def test_shiftedgompertz():
    b = Symbol("b", positive=True)
    eta = Symbol("eta", positive=True)
    X = ShiftedGompertz("x", b, eta)
    assert density(X)(x) == b*(eta*(1 - exp(-b*x)) + 1)*exp(-b*x)*exp(-eta*exp(-b*x))


def test_studentt():
    nu = Symbol("nu", positive=True)

    X = StudentT('x', nu)
    assert density(X)(x) == (1 + x**2/nu)**(-nu/2 - S.Half)/(sqrt(nu)*beta(S.Half, nu/2))
    assert cdf(X)(x) == S.Half + x*gamma(nu/2 + S.Half)*hyper((S.Half, nu/2 + S.Half),
                                (Rational(3, 2),), -x**2/nu)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))
    raises(NotImplementedError, lambda: moment_generating_function(X))

def test_trapezoidal():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    c = Symbol("c", real=True)
    d = Symbol("d", real=True)

    X = Trapezoidal('x', a, b, c, d)
    assert density(X)(x) == Piecewise(((-2*a + 2*x)/((-a + b)*(-a - b + c + d)), (a <= x) & (x < b)),
                                      (2/(-a - b + c + d), (b <= x) & (x < c)),
                                      ((2*d - 2*x)/((-c + d)*(-a - b + c + d)), (c <= x) & (x <= d)),
                                      (0, True))

    X = Trapezoidal('x', 0, 1, 2, 3)
    assert E(X) == Rational(3, 2)
    assert variance(X) == Rational(5, 12)
    assert P(X < 2) == Rational(3, 4)
    assert median(X) == FiniteSet(Rational(3, 2))

def test_triangular():
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")

    X = Triangular('x', a, b, c)
    assert pspace(X).domain.set == Interval(a, b)
    assert str(density(X)(x)) == ("Piecewise(((-2*a + 2*x)/((-a + b)*(-a + c)), (a <= x) & (c > x)), "
    "(2/(-a + b), Eq(c, x)), ((2*b - 2*x)/((-a + b)*(b - c)), (b >= x) & (c < x)), (0, True))")

    #Tests moment_generating_function
    assert moment_generating_function(X)(x).expand() == \
    ((-2*(-a + b)*exp(c*x) + 2*(-a + c)*exp(b*x) + 2*(b - c)*exp(a*x))/(x**2*(-a + b)*(-a + c)*(b - c))).expand()
    assert str(characteristic_function(X)(x)) == \
    '(2*(-a + b)*exp(I*c*x) - 2*(-a + c)*exp(I*b*x) - 2*(b - c)*exp(I*a*x))/(x**2*(-a + b)*(-a + c)*(b - c))'

def test_quadratic_u():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)

    X = QuadraticU("x", a, b)
    Y = QuadraticU("x", 1, 2)

    assert pspace(X).domain.set == Interval(a, b)
    # Tests _moment_generating_function
    assert moment_generating_function(Y)(1)  == -15*exp(2) + 27*exp(1)
    assert moment_generating_function(Y)(2) == -9*exp(4)/2 + 21*exp(2)/2

    assert characteristic_function(Y)(1) == 3*I*(-1 + 4*I)*exp(I*exp(2*I))
    assert density(X)(x) == (Piecewise((12*(x - a/2 - b/2)**2/(-a + b)**3,
                          And(x <= b, a <= x)), (0, True)))


def test_uniform():
    l = Symbol('l', real=True)
    w = Symbol('w', positive=True)
    X = Uniform('x', l, l + w)

    assert E(X) == l + w/2
    assert variance(X).expand() == w**2/12

    # With numbers all is well
    X = Uniform('x', 3, 5)
    assert P(X < 3) == 0 and P(X > 5) == 0
    assert P(X < 4) == P(X > 4) == S.Half
    assert median(X) == FiniteSet(4)

    z = Symbol('z')
    p = density(X)(z)
    assert p.subs(z, 3.7) == S.Half
    assert p.subs(z, -1) == 0
    assert p.subs(z, 6) == 0

    c = cdf(X)
    assert c(2) == 0 and c(3) == 0
    assert c(Rational(7, 2)) == Rational(1, 4)
    assert c(5) == 1 and c(6) == 1


@XFAIL
@slow
def test_uniform_P():
    """ This stopped working because SingleContinuousPSpace.compute_density no
    longer calls integrate on a DiracDelta but rather just solves directly.
    integrate used to call UniformDistribution.expectation which special-cased
    subsed out the Min and Max terms that Uniform produces

    I decided to regress on this class for general cleanliness (and I suspect
    speed) of the algorithm.
    """
    l = Symbol('l', real=True)
    w = Symbol('w', positive=True)
    X = Uniform('x', l, l + w)
    assert P(X < l) == 0 and P(X > l + w) == 0


def test_uniformsum():
    n = Symbol("n", integer=True)
    _k = Dummy("k")
    x = Symbol("x")

    X = UniformSum('x', n)
    res = Sum((-1)**_k*(-_k + x)**(n - 1)*binomial(n, _k), (_k, 0, floor(x)))/factorial(n - 1)
    assert density(X)(x).dummy_eq(res)

    #Tests set functions
    assert X.pspace.domain.set == Interval(0, n)

    #Tests the characteristic_function
    assert characteristic_function(X)(x) == (-I*(exp(I*x) - 1)/x)**n

    #Tests the moment_generating_function
    assert moment_generating_function(X)(x) == ((exp(x) - 1)/x)**n


def test_von_mises():
    mu = Symbol("mu")
    k = Symbol("k", positive=True)

    X = VonMises("x", mu, k)
    assert density(X)(x) == exp(k*cos(x - mu))/(2*pi*besseli(0, k))


def test_weibull():
    a, b = symbols('a b', positive=True)
    # FIXME: simplify(E(X)) seems to hang without extended_positive=True
    # On a Linux machine this had a rapid memory leak...
    # a, b = symbols('a b', positive=True)
    X = Weibull('x', a, b)

    assert E(X).expand() == a * gamma(1 + 1/b)
    assert variance(X).expand() == (a**2 * gamma(1 + 2/b) - E(X)**2).expand()
    assert simplify(skewness(X)) == (2*gamma(1 + 1/b)**3 - 3*gamma(1 + 1/b)*gamma(1 + 2/b) + gamma(1 + 3/b))/(-gamma(1 + 1/b)**2 + gamma(1 + 2/b))**Rational(3, 2)
    assert simplify(kurtosis(X)) == (-3*gamma(1 + 1/b)**4 +\
        6*gamma(1 + 1/b)**2*gamma(1 + 2/b) - 4*gamma(1 + 1/b)*gamma(1 + 3/b) + gamma(1 + 4/b))/(gamma(1 + 1/b)**2 - gamma(1 + 2/b))**2

def test_weibull_numeric():
    # Test for integers and rationals
    a = 1
    bvals = [S.Half, 1, Rational(3, 2), 5]
    for b in bvals:
        X = Weibull('x', a, b)
        assert simplify(E(X)) == expand_func(a * gamma(1 + 1/S(b)))
        assert simplify(variance(X)) == simplify(
            a**2 * gamma(1 + 2/S(b)) - E(X)**2)
        # Not testing Skew... it's slow with int/frac values > 3/2


def test_wignersemicircle():
    R = Symbol("R", positive=True)

    X = WignerSemicircle('x', R)
    assert pspace(X).domain.set == Interval(-R, R)
    assert density(X)(x) == 2*sqrt(-x**2 + R**2)/(pi*R**2)
    assert E(X) == 0


    #Tests ChiNoncentralDistribution
    assert characteristic_function(X)(x) == \
           Piecewise((2*besselj(1, R*x)/(R*x), Ne(x, 0)), (1, True))


def test_input_value_assertions():
    a, b = symbols('a b')
    p, q = symbols('p q', positive=True)
    m, n = symbols('m n', positive=False, real=True)

    raises(ValueError, lambda: Normal('x', 3, 0))
    raises(ValueError, lambda: Normal('x', m, n))
    Normal('X', a, p)  # No error raised
    raises(ValueError, lambda: Exponential('x', m))
    Exponential('Ex', p)  # No error raised
    for fn in [Pareto, Weibull, Beta, Gamma]:
        raises(ValueError, lambda: fn('x', m, p))
        raises(ValueError, lambda: fn('x', p, n))
        fn('x', p, q)  # No error raised


def test_unevaluated():
    X = Normal('x', 0, 1)
    k = Dummy('k')
    expr1 = Integral(sqrt(2)*k*exp(-k**2/2)/(2*sqrt(pi)), (k, -oo, oo))
    expr2 = Integral(sqrt(2)*exp(-k**2/2)/(2*sqrt(pi)), (k, 0, oo))
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert E(X, evaluate=False).rewrite(Integral).dummy_eq(expr1)
        assert E(X + 1, evaluate=False).rewrite(Integral).dummy_eq(expr1 + 1)
        assert P(X > 0, evaluate=False).rewrite(Integral).dummy_eq(expr2)

    assert P(X > 0, X**2 < 1) == S.Half


def test_probability_unevaluated():
    T = Normal('T', 30, 3)
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert type(P(T > 33, evaluate=False)) == Probability


def test_density_unevaluated():
    X = Normal('X', 0, 1)
    Y = Normal('Y', 0, 2)
    assert isinstance(density(X+Y, evaluate=False)(z), Integral)


def test_NormalDistribution():
    nd = NormalDistribution(0, 1)
    x = Symbol('x')
    assert nd.cdf(x) == erf(sqrt(2)*x/2)/2 + S.Half
    assert nd.expectation(1, x) == 1
    assert nd.expectation(x, x) == 0
    assert nd.expectation(x**2, x) == 1
    #Test issue 10076
    a = SingleContinuousPSpace(x, NormalDistribution(2, 4))
    _z = Dummy('_z')

    expected1 = Integral(sqrt(2)*exp(-(_z - 2)**2/32)/(8*sqrt(pi)),(_z, -oo, 1))
    assert a.probability(x < 1, evaluate=False).dummy_eq(expected1) is True

    expected2 = Integral(sqrt(2)*exp(-(_z - 2)**2/32)/(8*sqrt(pi)),(_z, 1, oo))
    assert a.probability(x > 1, evaluate=False).dummy_eq(expected2) is True

    b = SingleContinuousPSpace(x, NormalDistribution(1, 9))

    expected3 = Integral(sqrt(2)*exp(-(_z - 1)**2/162)/(18*sqrt(pi)),(_z, 6, oo))
    assert b.probability(x > 6, evaluate=False).dummy_eq(expected3) is True

    expected4 = Integral(sqrt(2)*exp(-(_z - 1)**2/162)/(18*sqrt(pi)),(_z, -oo, 6))
    assert b.probability(x < 6, evaluate=False).dummy_eq(expected4) is True


def test_random_parameters():
    mu = Normal('mu', 2, 3)
    meas = Normal('T', mu, 1)
    assert density(meas, evaluate=False)(z)
    assert isinstance(pspace(meas), CompoundPSpace)
    X = Normal('x', [1, 2], [[1, 0], [0, 1]])
    assert isinstance(pspace(X).distribution, MultivariateNormalDistribution)
    assert density(meas)(z).simplify() == sqrt(5)*exp(-z**2/20 + z/5 - S(1)/5)/(10*sqrt(pi))


def test_random_parameters_given():
    mu = Normal('mu', 2, 3)
    meas = Normal('T', mu, 1)
    assert given(meas, Eq(mu, 5)) == Normal('T', 5, 1)


def test_conjugate_priors():
    mu = Normal('mu', 2, 3)
    x = Normal('x', mu, 1)
    assert isinstance(simplify(density(mu, Eq(x, y), evaluate=False)(z)),
            Mul)


def test_difficult_univariate():
    """ Since using solve in place of deltaintegrate we're able to perform
    substantially more complex density computations on single continuous random
    variables """
    x = Normal('x', 0, 1)
    assert density(x**3)
    assert density(exp(x**2))
    assert density(log(x))


def test_issue_10003():
    X = Exponential('x', 3)
    G = Gamma('g', 1, 2)
    assert P(X < -1) is S.Zero
    assert P(G < -1) is S.Zero


def test_precomputed_cdf():
    x = symbols("x", real=True)
    mu = symbols("mu", real=True)
    sigma, xm, alpha = symbols("sigma xm alpha", positive=True)
    n = symbols("n", integer=True, positive=True)
    distribs = [
            Normal("X", mu, sigma),
            Pareto("P", xm, alpha),
            ChiSquared("C", n),
            Exponential("E", sigma),
            # LogNormal("L", mu, sigma),
    ]
    for X in distribs:
        compdiff = cdf(X)(x) - simplify(X.pspace.density.compute_cdf()(x))
        compdiff = simplify(compdiff.rewrite(erfc))
        assert compdiff == 0


@slow
def test_precomputed_characteristic_functions():
    import mpmath

    def test_cf(dist, support_lower_limit, support_upper_limit):
        pdf = density(dist)
        t = Symbol('t')

        # first function is the hardcoded CF of the distribution
        cf1 = lambdify([t], characteristic_function(dist)(t), 'mpmath')

        # second function is the Fourier transform of the density function
        f = lambdify([x, t], pdf(x)*exp(I*x*t), 'mpmath')
        cf2 = lambda t: mpmath.quad(lambda x: f(x, t), [support_lower_limit, support_upper_limit], maxdegree=10)

        # compare the two functions at various points
        for test_point in [2, 5, 8, 11]:
            n1 = cf1(test_point)
            n2 = cf2(test_point)

            assert abs(re(n1) - re(n2)) < 1e-12
            assert abs(im(n1) - im(n2)) < 1e-12

    test_cf(Beta('b', 1, 2), 0, 1)
    test_cf(Chi('c', 3), 0, mpmath.inf)
    test_cf(ChiSquared('c', 2), 0, mpmath.inf)
    test_cf(Exponential('e', 6), 0, mpmath.inf)
    test_cf(Logistic('l', 1, 2), -mpmath.inf, mpmath.inf)
    test_cf(Normal('n', -1, 5), -mpmath.inf, mpmath.inf)
    test_cf(RaisedCosine('r', 3, 1), 2, 4)
    test_cf(Rayleigh('r', 0.5), 0, mpmath.inf)
    test_cf(Uniform('u', -1, 1), -1, 1)
    test_cf(WignerSemicircle('w', 3), -3, 3)


def test_long_precomputed_cdf():
    x = symbols("x", real=True)
    distribs = [
            Arcsin("A", -5, 9),
            Dagum("D", 4, 10, 3),
            Erlang("E", 14, 5),
            Frechet("F", 2, 6, -3),
            Gamma("G", 2, 7),
            GammaInverse("GI", 3, 5),
            Kumaraswamy("K", 6, 8),
            Laplace("LA", -5, 4),
            Logistic("L", -6, 7),
            Nakagami("N", 2, 7),
            StudentT("S", 4)
            ]
    for distr in distribs:
        for _ in range(5):
            assert tn(diff(cdf(distr)(x), x), density(distr)(x), x, a=0, b=0, c=1, d=0)

    US = UniformSum("US", 5)
    pdf01 = density(US)(x).subs(floor(x), 0).doit()   # pdf on (0, 1)
    cdf01 = cdf(US, evaluate=False)(x).subs(floor(x), 0).doit()   # cdf on (0, 1)
    assert tn(diff(cdf01, x), pdf01, x, a=0, b=0, c=1, d=0)


def test_issue_13324():
    X = Uniform('X', 0, 1)
    assert E(X, X > S.Half) == Rational(3, 4)
    assert E(X, X > 0) == S.Half

def test_issue_20756():
    X = Uniform('X', -1, +1)
    Y = Uniform('Y', -1, +1)
    assert E(X * Y) == S.Zero
    assert E(X * ((Y + 1) - 1)) == S.Zero
    assert E(Y * (X*(X + 1) - X*X)) == S.Zero

def test_FiniteSet_prob():
    E = Exponential('E', 3)
    N = Normal('N', 5, 7)
    assert P(Eq(E, 1)) is S.Zero
    assert P(Eq(N, 2)) is S.Zero
    assert P(Eq(N, x)) is S.Zero

def test_prob_neq():
    E = Exponential('E', 4)
    X = ChiSquared('X', 4)
    assert P(Ne(E, 2)) == 1
    assert P(Ne(X, 4)) == 1
    assert P(Ne(X, 4)) == 1
    assert P(Ne(X, 5)) == 1
    assert P(Ne(E, x)) == 1

def test_union():
    N = Normal('N', 3, 2)
    assert simplify(P(N**2 - N > 2)) == \
        -erf(sqrt(2))/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)
    assert simplify(P(N**2 - 4 > 0)) == \
        -erf(5*sqrt(2)/4)/2 - erfc(sqrt(2)/4)/2 + Rational(3, 2)

def test_Or():
    N = Normal('N', 0, 1)
    assert simplify(P(Or(N > 2, N < 1))) == \
        -erf(sqrt(2))/2 - erfc(sqrt(2)/2)/2 + Rational(3, 2)
    assert P(Or(N < 0, N < 1)) == P(N < 1)
    assert P(Or(N > 0, N < 0)) == 1


def test_conditional_eq():
    E = Exponential('E', 1)
    assert P(Eq(E, 1), Eq(E, 1)) == 1
    assert P(Eq(E, 1), Eq(E, 2)) == 0
    assert P(E > 1, Eq(E, 2)) == 1
    assert P(E < 1, Eq(E, 2)) == 0

def test_ContinuousDistributionHandmade():
    x = Symbol('x')
    z = Dummy('z')
    dens = Lambda(x, Piecewise((S.Half, (0<=x)&(x<1)), (0, (x>=1)&(x<2)),
        (S.Half, (x>=2)&(x<3)), (0, True)))
    dens = ContinuousDistributionHandmade(dens, set=Interval(0, 3))
    space = SingleContinuousPSpace(z, dens)
    assert dens.pdf == Lambda(x, Piecewise((S(1)/2, (x >= 0) & (x < 1)),
        (0, (x >= 1) & (x < 2)), (S(1)/2, (x >= 2) & (x < 3)), (0, True)))
    assert median(space.value) == Interval(1, 2)
    assert E(space.value) == Rational(3, 2)
    assert variance(space.value) == Rational(13, 12)


def test_issue_16318():
    # test compute_expectation function of the SingleContinuousDomain
    N = SingleContinuousDomain(x, Interval(0, 1))
    raises(ValueError, lambda: SingleContinuousDomain.compute_expectation(N, x+1, {x, y}))

def test_compute_density():
    X = Normal('X', 0, Symbol("sigma")**2)
    raises(ValueError, lambda: density(X**5 + X))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\__init__.py ===
"""
Charset-Normalizer
~~~~~~~~~~~~~~
The Real First Universal Charset Detector.
A library that helps you read text from an unknown charset encoding.
Motivated by chardet, This package is trying to resolve the issue by taking a new approach.
All IANA character set names for which the Python core library provides codecs are supported.

Basic usage:
   >>> from charset_normalizer import from_bytes
   >>> results = from_bytes('Bсеки човек има право на образование. Oбразованието!'.encode('utf_8'))
   >>> best_guess = results.best()
   >>> str(best_guess)
   'Bсеки човек има право на образование. Oбразованието!'

Others methods and usages are available - see the full documentation
at <https://github.com/Ousret/charset_normalizer>.
:copyright: (c) 2021 by Ahmed TAHRI
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import logging

from .api import from_bytes, from_fp, from_path, is_binary
from .legacy import detect
from .models import CharsetMatch, CharsetMatches
from .utils import set_logging_handler
from .version import VERSION, __version__

__all__ = (
    "from_fp",
    "from_path",
    "from_bytes",
    "is_binary",
    "detect",
    "CharsetMatch",
    "CharsetMatches",
    "__version__",
    "VERSION",
    "set_logging_handler",
)

# Attach a NullHandler to the top level logger by default
# https://docs.python.org/3.3/howto/logging.html#configuring-logging-for-a-library

logging.getLogger("charset_normalizer").addHandler(logging.NullHandler())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\defaults.py ===
import typing as t

from .filters import FILTERS as DEFAULT_FILTERS  # noqa: F401
from .tests import TESTS as DEFAULT_TESTS  # noqa: F401
from .utils import Cycler
from .utils import generate_lorem_ipsum
from .utils import Joiner
from .utils import Namespace

if t.TYPE_CHECKING:
    import typing_extensions as te

# defaults for the parser / lexer
BLOCK_START_STRING = "{%"
BLOCK_END_STRING = "%}"
VARIABLE_START_STRING = "{{"
VARIABLE_END_STRING = "}}"
COMMENT_START_STRING = "{#"
COMMENT_END_STRING = "#}"
LINE_STATEMENT_PREFIX: t.Optional[str] = None
LINE_COMMENT_PREFIX: t.Optional[str] = None
TRIM_BLOCKS = False
LSTRIP_BLOCKS = False
NEWLINE_SEQUENCE: "te.Literal['\\n', '\\r\\n', '\\r']" = "\n"
KEEP_TRAILING_NEWLINE = False

# default filters, tests and namespace

DEFAULT_NAMESPACE = {
    "range": range,
    "dict": dict,
    "lipsum": generate_lorem_ipsum,
    "cycler": Cycler,
    "joiner": Joiner,
    "namespace": Namespace,
}

# default policies
DEFAULT_POLICIES: t.Dict[str, t.Any] = {
    "compiler.ascii_str": True,
    "urlize.rel": "noopener",
    "urlize.target": None,
    "urlize.extra_schemes": None,
    "truncate.leeway": 5,
    "json.dumps_function": None,
    "json.dumps_kwargs": {"sort_keys": True},
    "ext.i18n.trimmed": False,
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\optimizer.py ===
"""The optimizer tries to constant fold expressions and modify the AST
in place so that it should be faster to evaluate.

Because the AST does not contain all the scoping information and the
compiler has to find that out, we cannot do all the optimizations we
want. For example, loop unrolling doesn't work because unrolled loops
would have a different scope. The solution would be a second syntax tree
that stored the scoping rules.
"""

import typing as t

from . import nodes
from .visitor import NodeTransformer

if t.TYPE_CHECKING:
    from .environment import Environment


def optimize(node: nodes.Node, environment: "Environment") -> nodes.Node:
    """The context hint can be used to perform an static optimization
    based on the context given."""
    optimizer = Optimizer(environment)
    return t.cast(nodes.Node, optimizer.visit(node))


class Optimizer(NodeTransformer):
    def __init__(self, environment: "t.Optional[Environment]") -> None:
        self.environment = environment

    def generic_visit(
        self, node: nodes.Node, *args: t.Any, **kwargs: t.Any
    ) -> nodes.Node:
        node = super().generic_visit(node, *args, **kwargs)

        # Do constant folding. Some other nodes besides Expr have
        # as_const, but folding them causes errors later on.
        if isinstance(node, nodes.Expr):
            try:
                return nodes.Const.from_untrusted(
                    node.as_const(args[0] if args else None),
                    lineno=node.lineno,
                    environment=self.environment,
                )
            except nodes.Impossible:
                pass

        return node

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\convert_npz_to_onnx_adapter.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script helps converting .npz files to .onnx_adapter files

import argparse
import os
import sys

import numpy as np

import onnxruntime as ort


def get_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--adapter_version", type=int, required=True)
    parser.add_argument("--model_version", type=int, required=True)
    return parser.parse_args()


def export_lora_parameters(
    npz_file_path: os.PathLike, adapter_version: int, model_version: int, output_file_path: os.PathLike
):
    """The function converts lora parameters in npz to onnx_adapter format"""
    adapter_format = ort.AdapterFormat()
    adapter_format.set_adapter_version(adapter_version)
    adapter_format.set_model_version(model_version)
    name_to_ort_value = {}
    with np.load(npz_file_path) as data:
        for name, np_arr in data.items():
            ort_value = ort.OrtValue.ortvalue_from_numpy(np_arr)
            name_to_ort_value[name] = ort_value

    adapter_format.set_parameters(name_to_ort_value)
    adapter_format.export_adapter(output_file_path)


def main() -> int:
    args = get_args()
    export_lora_parameters(args.npz_file_path, args.adapter_version, args.model_version, args.output_file_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\common.py ===
from __future__ import annotations

from functools import reduce

import numpy as np

from pandas._config import get_option


def ensure_decoded(s) -> str:
    """
    If we have bytes, decode them to unicode.
    """
    if isinstance(s, (np.bytes_, bytes)):
        s = s.decode(get_option("display.encoding"))
    return s


def result_type_many(*arrays_and_dtypes):
    """
    Wrapper around numpy.result_type which overcomes the NPY_MAXARGS (32)
    argument limit.
    """
    try:
        return np.result_type(*arrays_and_dtypes)
    except ValueError:
        # we have > NPY_MAXARGS terms in our expression
        return reduce(np.result_type, arrays_and_dtypes)
    except TypeError:
        from pandas.core.dtypes.cast import find_common_type
        from pandas.core.dtypes.common import is_extension_array_dtype

        arr_and_dtypes = list(arrays_and_dtypes)
        ea_dtypes, non_ea_dtypes = [], []
        for arr_or_dtype in arr_and_dtypes:
            if is_extension_array_dtype(arr_or_dtype):
                ea_dtypes.append(arr_or_dtype)
            else:
                non_ea_dtypes.append(arr_or_dtype)

        if non_ea_dtypes:
            try:
                np_dtype = np.result_type(*non_ea_dtypes)
            except ValueError:
                np_dtype = reduce(np.result_type, arrays_and_dtypes)
            return find_common_type(ea_dtypes + [np_dtype])

        return find_common_type(ea_dtypes)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\cachecontrol\caches\redis_cache.py ===
# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pip._vendor.cachecontrol.cache import BaseCache

if TYPE_CHECKING:
    from redis import Redis


class RedisCache(BaseCache):
    def __init__(self, conn: Redis[bytes]) -> None:
        self.conn = conn

    def get(self, key: str) -> bytes | None:
        return self.conn.get(key)

    def set(
        self, key: str, value: bytes, expires: int | datetime | None = None
    ) -> None:
        if not expires:
            self.conn.set(key, value)
        elif isinstance(expires, datetime):
            now_utc = datetime.now(timezone.utc)
            if expires.tzinfo is None:
                now_utc = now_utc.replace(tzinfo=None)
            delta = expires - now_utc
            self.conn.setex(key, int(delta.total_seconds()), value)
        else:
            self.conn.setex(key, expires, value)

    def delete(self, key: str) -> None:
        self.conn.delete(key)

    def clear(self) -> None:
        """Helper for clearing all the keys in a database. Use with
        caution!"""
        for key in self.conn.keys():
            self.conn.delete(key)

    def close(self) -> None:
        """Redis uses connection pooling, no need to close the connection."""
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\msgpack\exceptions.py ===
class UnpackException(Exception):
    """Base class for some exceptions raised while unpacking.

    NOTE: unpack may raise exception other than subclass of
    UnpackException.  If you want to catch all error, catch
    Exception instead.
    """


class BufferFull(UnpackException):
    pass


class OutOfData(UnpackException):
    pass


class FormatError(ValueError, UnpackException):
    """Invalid msgpack format"""


class StackError(ValueError, UnpackException):
    """Too nested"""


# Deprecated.  Use ValueError instead
UnpackValueError = ValueError


class ExtraData(UnpackValueError):
    """ExtraData is raised when there is trailing data.

    This exception is raised while only one-shot (not streaming)
    unpack.
    """

    def __init__(self, unpacked, extra):
        self.unpacked = unpacked
        self.extra = extra

    def __str__(self):
        return "unpack(b) received extra data."


# Deprecated.  Use Exception instead to catch all exception during packing.
PackException = Exception
PackValueError = ValueError
PackOverflowError = OverflowError

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\resolvelib\resolvers\criterion.py ===
from __future__ import annotations

from typing import Collection, Generic, Iterable, Iterator

from ..structs import CT, RT, RequirementInformation


class Criterion(Generic[RT, CT]):
    """Representation of possible resolution results of a package.

    This holds three attributes:

    * `information` is a collection of `RequirementInformation` pairs.
      Each pair is a requirement contributing to this criterion, and the
      candidate that provides the requirement.
    * `incompatibilities` is a collection of all known not-to-work candidates
      to exclude from consideration.
    * `candidates` is a collection containing all possible candidates deducted
      from the union of contributing requirements and known incompatibilities.
      It should never be empty, except when the criterion is an attribute of a
      raised `RequirementsConflicted` (in which case it is always empty).

    .. note::
        This class is intended to be externally immutable. **Do not** mutate
        any of its attribute containers.
    """

    def __init__(
        self,
        candidates: Iterable[CT],
        information: Collection[RequirementInformation[RT, CT]],
        incompatibilities: Collection[CT],
    ) -> None:
        self.candidates = candidates
        self.information = information
        self.incompatibilities = incompatibilities

    def __repr__(self) -> str:
        requirements = ", ".join(
            f"({req!r}, via={parent!r})" for req, parent in self.information
        )
        return f"Criterion({requirements})"

    def iter_requirement(self) -> Iterator[RT]:
        return (i.requirement for i in self.information)

    def iter_parent(self) -> Iterator[CT | None]:
        return (i.parent for i in self.information)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\key_input.py ===
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
from . import interaction
from .input_device import InputDevice
from .interaction import Interaction, Pause


class KeyInput(InputDevice):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.type = interaction.KEY

    def encode(self) -> dict:
        return {"type": self.type, "id": self.name, "actions": [acts.encode() for acts in self.actions]}

    def create_key_down(self, key) -> None:
        self.add_action(TypingInteraction(self, "keyDown", key))

    def create_key_up(self, key) -> None:
        self.add_action(TypingInteraction(self, "keyUp", key))

    def create_pause(self, pause_duration: float = 0) -> None:
        self.add_action(Pause(self, pause_duration))


class TypingInteraction(Interaction):
    def __init__(self, source, type_, key) -> None:
        super().__init__(source)
        self.type = type_
        self.key = key

    def encode(self) -> dict:
        return {"type": self.type, "value": self.key}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\schema.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Schema
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class Domain:
    '''
    Description of the protocol domain.
    '''
    #: Domain name.
    name: str

    #: Domain version.
    version: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['version'] = self.version
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            version=str(json['version']),
        )


def get_domains() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Domain]]:
    '''
    Returns supported domains.

    :returns: List of supported domains.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Schema.getDomains',
    }
    json = yield cmd_dict
    return [Domain.from_json(i) for i in json['domains']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\schema.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Schema
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class Domain:
    '''
    Description of the protocol domain.
    '''
    #: Domain name.
    name: str

    #: Domain version.
    version: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['version'] = self.version
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            version=str(json['version']),
        )


def get_domains() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Domain]]:
    '''
    Returns supported domains.

    :returns: List of supported domains.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Schema.getDomains',
    }
    json = yield cmd_dict
    return [Domain.from_json(i) for i in json['domains']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\schema.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Schema
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class Domain:
    '''
    Description of the protocol domain.
    '''
    #: Domain name.
    name: str

    #: Domain version.
    version: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['version'] = self.version
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            version=str(json['version']),
        )


def get_domains() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Domain]]:
    '''
    Returns supported domains.

    :returns: List of supported domains.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Schema.getDomains',
    }
    json = yield cmd_dict
    return [Domain.from_json(i) for i in json['domains']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\edge\options.py ===
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

from selenium.webdriver.chromium.options import ChromiumOptions
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


class Options(ChromiumOptions):
    KEY = "ms:edgeOptions"

    def __init__(self) -> None:
        super().__init__()
        self._use_webview = False

    @property
    def use_webview(self) -> bool:
        return self._use_webview

    @use_webview.setter
    def use_webview(self, value: bool) -> None:
        self._use_webview = bool(value)

    def to_capabilities(self) -> dict:
        """Creates a capabilities with all the options that have been set and
        :Returns: A dictionary with everything."""
        caps = super().to_capabilities()
        if self._use_webview:
            caps["browserName"] = "webview2"

        return caps

    @property
    def default_capabilities(self) -> dict:
        return DesiredCapabilities.EDGE.copy()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_subprocess_platform\kqueue.py ===
from __future__ import annotations

import select
import sys
from typing import TYPE_CHECKING

from .. import _core, _subprocess

assert (sys.platform != "win32" and sys.platform != "linux") or not TYPE_CHECKING


async def wait_child_exiting(process: _subprocess.Process) -> None:
    kqueue = _core.current_kqueue()
    try:
        from select import KQ_NOTE_EXIT
    except ImportError:  # pragma: no cover
        # pypy doesn't define KQ_NOTE_EXIT:
        # https://bitbucket.org/pypy/pypy/issues/2921/
        # I verified this value against both Darwin and FreeBSD
        KQ_NOTE_EXIT = 0x80000000

    def make_event(flags: int) -> select.kevent:
        return select.kevent(
            process.pid,
            filter=select.KQ_FILTER_PROC,
            flags=flags,
            fflags=KQ_NOTE_EXIT,
        )

    try:
        kqueue.control([make_event(select.KQ_EV_ADD | select.KQ_EV_ONESHOT)], 0)
    except ProcessLookupError:  # pragma: no cover
        # This can supposedly happen if the process is in the process
        # of exiting, and it can even be the case that kqueue says the
        # process doesn't exist before waitpid(WNOHANG) says it hasn't
        # exited yet. See the discussion in https://chromium.googlesource.com/
        # chromium/src/base/+/master/process/kill_mac.cc .
        # We haven't actually seen this error occur since we added
        # locking to prevent multiple calls to wait_child_exiting()
        # for the same process simultaneously, but given the explanation
        # in Chromium it seems we should still keep the check.
        return

    def abort(_: _core.RaiseCancelT) -> _core.Abort:
        kqueue.control([make_event(select.KQ_EV_DELETE)], 0)
        return _core.Abort.SUCCEEDED

    await _core.wait_kevent(process.pid, select.KQ_FILTER_PROC, abort)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_ssl_compat.py ===
"""
_ssl_compat.py
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
__all__ = [
    "HAVE_SSL",
    "ssl",
    "SSLError",
    "SSLEOFError",
    "SSLWantReadError",
    "SSLWantWriteError",
]

try:
    import ssl
    from ssl import SSLError, SSLEOFError, SSLWantReadError, SSLWantWriteError

    HAVE_SSL = True
except ImportError:
    # dummy class of SSLError for environment without ssl support
    class SSLError(Exception):
        pass

    class SSLEOFError(Exception):
        pass

    class SSLWantReadError(Exception):
        pass

    class SSLWantWriteError(Exception):
        pass

    ssl = None
    HAVE_SSL = False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\test_numeric.py ===
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for numeric dtypes
from __future__ import annotations

from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    array,
    date_range,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
)


@pytest.fixture(autouse=True, params=[0, 1000000], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield request.param


@pytest.fixture(params=[Index, Series, tm.to_array])
def box_pandas_1d_array(request):
    """
    Fixture to test behavior for Index, Series and tm.to_array classes
    """
    return request.param


@pytest.fixture(
    params=[
        # TODO: add more  dtypes here
        Index(np.arange(5, dtype="float64")),
        Index(np.arange(5, dtype="int64")),
        Index(np.arange(5, dtype="uint64")),
        RangeIndex(5),
    ],
    ids=lambda x: type(x).__name__,
)
def numeric_idx(request):
    """
    Several types of numeric-dtypes Index objects
    """
    return request.param


@pytest.fixture(
    params=[Index, Series, tm.to_array, np.array, list], ids=lambda x: x.__name__
)
def box_1d_array(request):
    """
    Fixture to test behavior for Index, Series, tm.to_array, numpy Array and list
    classes
    """
    return request.param


def adjust_negative_zero(zero, expected):
    """
    Helper to adjust the expected result if we are dividing by -0.0
    as opposed to 0.0
    """
    if np.signbit(np.array(zero)).any():
        # All entries in the `zero` fixture should be either
        #  all-negative or no-negative.
        assert np.signbit(np.array(zero)).all()

        expected *= -1

    return expected


def compare_op(series, other, op):
    left = np.abs(series) if op in (ops.rpow, operator.pow) else series
    right = np.abs(other) if op in (ops.rpow, operator.pow) else other

    cython_or_numpy = op(left, right)
    python = left.combine(right, op)
    if isinstance(other, Series) and not other.index.equals(series.index):
        python.index = python.index._with_freq(None)
    tm.assert_series_equal(cython_or_numpy, python)


# TODO: remove this kludge once mypy stops giving false positives here
# List comprehension has incompatible type List[PandasObject]; expected List[RangeIndex]
#  See GH#29725
_ldtypes = ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"]
lefts: list[Index | Series] = [RangeIndex(10, 40, 10)]
lefts.extend([Series([10, 20, 30], dtype=dtype) for dtype in _ldtypes])
lefts.extend([Index([10, 20, 30], dtype=dtype) for dtype in _ldtypes if dtype != "f2"])

# ------------------------------------------------------------------
# Comparisons


class TestNumericComparisons:
    def test_operator_series_comparison_zerorank(self):
        # GH#13006
        result = np.float64(0) > Series([1, 2, 3])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)
        result = Series([1, 2, 3]) < np.float64(0)
        expected = Series([1, 2, 3]) < 0.0
        tm.assert_series_equal(result, expected)
        result = np.array([0, 1, 2])[0] > Series([0, 1, 2])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_df_numeric_cmp_dt64_raises(self, box_with_array, fixed_now_ts):
        # GH#8932, GH#22163
        ts = fixed_now_ts
        obj = np.array(range(5))
        obj = tm.box_expected(obj, box_with_array)

        assert_invalid_comparison(obj, ts, box_with_array)

    def test_compare_invalid(self):
        # GH#8058
        # ops testing
        a = Series(np.random.default_rng(2).standard_normal(5), name=0)
        b = Series(np.random.default_rng(2).standard_normal(5))
        b.name = pd.Timestamp("2000-01-01")
        tm.assert_series_equal(a / b, 1 / (b / a))

    def test_numeric_cmp_string_numexpr_path(self, box_with_array, monkeypatch):
        # GH#36377, GH#35700
        box = box_with_array
        xbox = box if box is not Index else np.ndarray

        obj = Series(np.random.default_rng(2).standard_normal(51))
        obj = tm.box_expected(obj, box, transpose=False)
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 50)
            result = obj == "a"

        expected = Series(np.zeros(51, dtype=bool))
        expected = tm.box_expected(expected, xbox, transpose=False)
        tm.assert_equal(result, expected)

        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 50)
            result = obj != "a"
        tm.assert_equal(result, ~expected)

        msg = "Invalid comparison between dtype=float64 and str"
        with pytest.raises(TypeError, match=msg):
            obj < "a"


# ------------------------------------------------------------------
# Numeric dtypes Arithmetic with Datetime/Timedelta Scalar


class TestNumericArraylikeArithmeticWithDatetimeLike:
    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    def test_mul_td64arr(self, left, box_cls):
        # GH#22390
        right = np.array([1, 2, 3], dtype="m8[s]")
        right = box_cls(right)

        expected = TimedeltaIndex(["10s", "40s", "90s"], dtype=right.dtype)

        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        assert expected.dtype == right.dtype

        result = left * right
        tm.assert_equal(result, expected)

        result = right * left
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    def test_div_td64arr(self, left, box_cls):
        # GH#22390
        right = np.array([10, 40, 90], dtype="m8[s]")
        right = box_cls(right)

        expected = TimedeltaIndex(["1s", "2s", "3s"], dtype=right.dtype)
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        assert expected.dtype == right.dtype

        result = right / left
        tm.assert_equal(result, expected)

        result = right // left
        tm.assert_equal(result, expected)

        # (true_) needed for min-versions build 2022-12-26
        msg = "ufunc '(true_)?divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left / right

        msg = "ufunc 'floor_divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left // right

    # TODO: also test Tick objects;
    #  see test_numeric_arr_rdiv_tdscalar for note on these failing
    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            Timedelta(days=1).to_pytimedelta(),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[s]"),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[ms]"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_numeric_arr_mul_tdscalar(self, scalar_td, numeric_idx, box_with_array):
        # GH#19333
        box = box_with_array
        index = numeric_idx
        expected = TimedeltaIndex([Timedelta(days=n) for n in range(len(index))])
        if isinstance(scalar_td, np.timedelta64):
            dtype = scalar_td.dtype
            expected = expected.astype(dtype)
        elif type(scalar_td) is timedelta:
            expected = expected.astype("m8[us]")

        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)

        result = index * scalar_td
        tm.assert_equal(result, expected)

        commute = scalar_td * index
        tm.assert_equal(commute, expected)

    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            Timedelta(days=1).to_pytimedelta(),
        ],
        ids=lambda x: type(x).__name__,
    )
    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    def test_numeric_arr_mul_tdscalar_numexpr_path(
        self, dtype, scalar_td, box_with_array
    ):
        # GH#44772 for the float64 case
        box = box_with_array

        arr_i8 = np.arange(2 * 10**4).astype(np.int64, copy=False)
        arr = arr_i8.astype(dtype, copy=False)
        obj = tm.box_expected(arr, box, transpose=False)

        expected = arr_i8.view("timedelta64[D]").astype("timedelta64[ns]")
        if type(scalar_td) is timedelta:
            expected = expected.astype("timedelta64[us]")

        expected = tm.box_expected(expected, box, transpose=False)

        result = obj * scalar_td
        tm.assert_equal(result, expected)

        result = scalar_td * obj
        tm.assert_equal(result, expected)

    def test_numeric_arr_rdiv_tdscalar(self, three_days, numeric_idx, box_with_array):
        box = box_with_array

        index = numeric_idx[1:3]

        expected = TimedeltaIndex(["3 Days", "36 Hours"])
        if isinstance(three_days, np.timedelta64):
            dtype = three_days.dtype
            if dtype < np.dtype("m8[s]"):
                # i.e. resolution is lower -> use lowest supported resolution
                dtype = np.dtype("m8[s]")
            expected = expected.astype(dtype)
        elif type(three_days) is timedelta:
            expected = expected.astype("m8[us]")
        elif isinstance(
            three_days,
            (pd.offsets.Day, pd.offsets.Hour, pd.offsets.Minute, pd.offsets.Second),
        ):
            # closest reso is Second
            expected = expected.astype("m8[s]")

        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)

        result = three_days / index
        tm.assert_equal(result, expected)

        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            index / three_days

    @pytest.mark.parametrize(
        "other",
        [
            Timedelta(hours=31),
            Timedelta(hours=31).to_pytimedelta(),
            Timedelta(hours=31).to_timedelta64(),
            Timedelta(hours=31).to_timedelta64().astype("m8[h]"),
            np.timedelta64("NaT"),
            np.timedelta64("NaT", "D"),
            pd.offsets.Minute(3),
            pd.offsets.Second(0),
            # GH#28080 numeric+datetimelike should raise; Timestamp used
            #  to raise NullFrequencyError but that behavior was removed in 1.0
            pd.Timestamp("2021-01-01", tz="Asia/Tokyo"),
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-01").to_pydatetime(),
            pd.Timestamp("2021-01-01", tz="UTC").to_pydatetime(),
            pd.Timestamp("2021-01-01").to_datetime64(),
            np.datetime64("NaT", "ns"),
            pd.NaT,
        ],
        ids=repr,
    )
    def test_add_sub_datetimedeltalike_invalid(
        self, numeric_idx, other, box_with_array
    ):
        box = box_with_array

        left = tm.box_expected(numeric_idx, box)
        msg = "|".join(
            [
                "unsupported operand type",
                "Addition/subtraction of integers and integer-arrays",
                "Instead of adding/subtracting",
                "cannot use operands with types dtype",
                "Concatenation operation is not implemented for NumPy arrays",
                "Cannot (add|subtract) NaT (to|from) ndarray",
                # pd.array vs np.datetime64 case
                r"operand type\(s\) all returned NotImplemented from __array_ufunc__",
                "can only perform ops with numeric values",
                "cannot subtract DatetimeArray from ndarray",
                # pd.Timedelta(1) + Index([0, 1, 2])
                "Cannot add or subtract Timedelta from integers",
            ]
        )
        assert_invalid_addsub_type(left, other, msg)


# ------------------------------------------------------------------
# Arithmetic


class TestDivisionByZero:
    def test_div_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # We only adjust for Index, because Series does not yet apply
        #  the adjustment correctly.
        expected2 = adjust_negative_zero(zero, expected)

        result = idx / zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype("i8") / np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_floordiv_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # We only adjust for Index, because Series does not yet apply
        #  the adjustment correctly.
        expected2 = adjust_negative_zero(zero, expected)

        result = idx // zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype("i8") // np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_mod_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        result = idx % zero
        tm.assert_index_equal(result, expected)
        ser_compat = Series(idx).astype("i8") % np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(result))

    def test_divmod_zero(self, zero, numeric_idx):
        idx = numeric_idx

        exleft = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        exright = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        exleft = adjust_negative_zero(zero, exleft)

        result = divmod(idx, zero)
        tm.assert_index_equal(result[0], exleft)
        tm.assert_index_equal(result[1], exright)

    @pytest.mark.parametrize("op", [operator.truediv, operator.floordiv])
    def test_div_negative_zero(self, zero, numeric_idx, op):
        # Check that -1 / -0.0 returns np.inf, not -np.inf
        if numeric_idx.dtype == np.uint64:
            pytest.skip(f"Div by negative 0 not relevant for {numeric_idx.dtype}")
        idx = numeric_idx - 3

        expected = Index([-np.inf, -np.inf, -np.inf, np.nan, np.inf], dtype=np.float64)
        expected = adjust_negative_zero(zero, expected)

        result = op(idx, zero)
        tm.assert_index_equal(result, expected)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_div_ser(
        self,
        switch_numexpr_min_elements,
        dtype1,
        any_real_numpy_dtype,
    ):
        # no longer do integer div for any ops, but deal with the 0's
        dtype2 = any_real_numpy_dtype

        first = Series([3, 4, 5, 8], name="first").astype(dtype1)
        second = Series([0, 0, 0, 3], name="second").astype(dtype2)

        with np.errstate(all="ignore"):
            expected = Series(
                first.values.astype(np.float64) / second.values,
                dtype="float64",
                name=None,
            )
        expected.iloc[0:3] = np.inf
        if first.dtype == "int64" and second.dtype == "float32":
            # when using numexpr, the casting rules are slightly different
            # and int64/float32 combo results in float32 instead of float64
            if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                expected = expected.astype("float32")

        result = first / second
        tm.assert_series_equal(result, expected)
        assert not result.equals(second / first)

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_divmod_zero(self, dtype1, any_real_numpy_dtype):
        # GH#26987
        dtype2 = any_real_numpy_dtype
        left = Series([1, 1]).astype(dtype1)
        right = Series([0, 2]).astype(dtype2)

        # GH#27321 pandas convention is to set 1 // 0 to np.inf, as opposed
        #  to numpy which sets to np.nan; patch `expected[0]` below
        expected = left // right, left % right
        expected = list(expected)
        expected[0] = expected[0].astype(np.float64)
        expected[0][0] = np.inf
        result = divmod(left, right)

        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod case
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_ser_divmod_inf(self):
        left = Series([np.inf, 1.0])
        right = Series([np.inf, 2.0])

        expected = left // right, left % right
        result = divmod(left, right)

        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod case
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_rdiv_zero_compat(self):
        # GH#8674
        zero_array = np.array([0] * 5)
        data = np.random.default_rng(2).standard_normal(5)
        expected = Series([0.0] * 5)

        result = zero_array / Series(data)
        tm.assert_series_equal(result, expected)

        result = Series(zero_array) / data
        tm.assert_series_equal(result, expected)

        result = Series(zero_array) / Series(data)
        tm.assert_series_equal(result, expected)

    def test_div_zero_inf_signs(self):
        # GH#9144, inf signing
        ser = Series([-1, 0, 1], name="first")
        expected = Series([-np.inf, np.nan, np.inf], name="first")

        result = ser / 0
        tm.assert_series_equal(result, expected)

    def test_rdiv_zero(self):
        # GH#9144
        ser = Series([-1, 0, 1], name="first")
        expected = Series([0.0, np.nan, 0.0], name="first")

        result = 0 / ser
        tm.assert_series_equal(result, expected)

    def test_floordiv_div(self):
        # GH#9144
        ser = Series([-1, 0, 1], name="first")

        result = ser // 0
        expected = Series([-np.inf, np.nan, np.inf], name="first")
        tm.assert_series_equal(result, expected)

    def test_df_div_zero_df(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        result = df / df

        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({"first": first, "second": second})
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_array(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({"first": first, "second": second})

        with np.errstate(all="ignore"):
            arr = df.values.astype("float") / df.values
        result = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_int(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        result = df / 0
        expected = pd.DataFrame(np.inf, index=df.index, columns=df.columns)
        expected.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result, expected)

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values.astype("float64") / 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_div_zero_series_does_not_commute(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        ser = df[0]
        res = ser / df
        res2 = df / ser
        assert not res.fillna(0).equals(res2.fillna(0))

    # ------------------------------------------------------------------
    # Mod By Zero

    def test_df_mod_zero_df(self, using_array_manager):
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        # this is technically wrong, as the integer portion is coerced to float
        first = Series([0, 0, 0, 0])
        if not using_array_manager:
            # INFO(ArrayManager) BlockManager doesn't preserve dtype per column
            # while ArrayManager performs op column-wisedoes and thus preserves
            # dtype if possible
            first = first.astype("float64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})
        result = df % df
        tm.assert_frame_equal(result, expected)

        # GH#38939 If we dont pass copy=False, df is consolidated and
        #  result["first"] is float64 instead of int64
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]}, copy=False)
        first = Series([0, 0, 0, 0], dtype="int64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})
        result = df % df
        tm.assert_frame_equal(result, expected)

    def test_df_mod_zero_array(self):
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        # this is technically wrong, as the integer portion is coerced to float
        # ###
        first = Series([0, 0, 0, 0], dtype="float64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values % df.values
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns, dtype="float64")
        result2.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_int(self):
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        result = df % 0
        expected = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values.astype("float64") % 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_series_does_not_commute(self):
        # GH#3590, modulo as ints
        # not commutative with series
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        ser = df[0]
        res = ser % df
        res2 = df % ser
        assert not res.fillna(0).equals(res2.fillna(0))


class TestMultiplicationDivision:
    # __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__
    # for non-timestamp/timedelta/period dtypes

    def test_divide_decimal(self, box_with_array):
        # resolves issue GH#9787
        box = box_with_array
        ser = Series([Decimal(10)])
        expected = Series([Decimal(5)])

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = ser / Decimal(2)

        tm.assert_equal(result, expected)

        result = ser // Decimal(2)
        tm.assert_equal(result, expected)

    def test_div_equiv_binop(self):
        # Test Series.div as well as Series.__div__
        # float/integer issue
        # GH#7785
        first = Series([1, 0], name="first")
        second = Series([-0.01, -0.02], name="second")
        expected = Series([-0.01, -np.inf])

        result = second.div(first)
        tm.assert_series_equal(result, expected, check_names=False)

        result = second / first
        tm.assert_series_equal(result, expected)

    def test_div_int(self, numeric_idx):
        idx = numeric_idx
        result = idx / 1
        expected = idx.astype("float64")
        tm.assert_index_equal(result, expected)

        result = idx / 2
        expected = Index(idx.values / 2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.mul, ops.rmul, operator.floordiv])
    def test_mul_int_identity(self, op, numeric_idx, box_with_array):
        idx = numeric_idx
        idx = tm.box_expected(idx, box_with_array)

        result = op(idx, 1)
        tm.assert_equal(result, idx)

    def test_mul_int_array(self, numeric_idx):
        idx = numeric_idx
        didx = idx * idx

        result = idx * np.array(5, dtype="int64")
        tm.assert_index_equal(result, idx * 5)

        arr_dtype = "uint64" if idx.dtype == np.uint64 else "int64"
        result = idx * np.arange(5, dtype=arr_dtype)
        tm.assert_index_equal(result, didx)

    def test_mul_int_series(self, numeric_idx):
        idx = numeric_idx
        didx = idx * idx

        arr_dtype = "uint64" if idx.dtype == np.uint64 else "int64"
        result = idx * Series(np.arange(5, dtype=arr_dtype))
        tm.assert_series_equal(result, Series(didx))

    def test_mul_float_series(self, numeric_idx):
        idx = numeric_idx
        rng5 = np.arange(5, dtype="float64")

        result = idx * Series(rng5 + 0.1)
        expected = Series(rng5 * (rng5 + 0.1))
        tm.assert_series_equal(result, expected)

    def test_mul_index(self, numeric_idx):
        idx = numeric_idx

        result = idx * idx
        tm.assert_index_equal(result, idx**2)

    def test_mul_datelike_raises(self, numeric_idx):
        idx = numeric_idx
        msg = "cannot perform __rmul__ with this index type"
        with pytest.raises(TypeError, match=msg):
            idx * date_range("20130101", periods=5)

    def test_mul_size_mismatch_raises(self, numeric_idx):
        idx = numeric_idx
        msg = "operands could not be broadcast together"
        with pytest.raises(ValueError, match=msg):
            idx * idx[0:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    @pytest.mark.parametrize("op", [operator.pow, ops.rpow])
    def test_pow_float(self, op, numeric_idx, box_with_array):
        # test power calculations both ways, GH#14973
        box = box_with_array
        idx = numeric_idx
        expected = Index(op(idx.values, 2.0))

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        result = op(idx, 2.0)
        tm.assert_equal(result, expected)

    def test_modulo(self, numeric_idx, box_with_array):
        # GH#9244
        box = box_with_array
        idx = numeric_idx
        expected = Index(idx.values % 2)

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        result = idx % 2
        tm.assert_equal(result, expected)

    def test_divmod_scalar(self, numeric_idx):
        idx = numeric_idx

        result = divmod(idx, 2)
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, 2)

        expected = Index(div), Index(mod)
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    def test_divmod_ndarray(self, numeric_idx):
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        result = divmod(idx, other)
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        expected = Index(div), Index(mod)
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    def test_divmod_series(self, numeric_idx):
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        result = divmod(idx, Series(other))
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        expected = Series(div), Series(mod)
        for r, e in zip(result, expected):
            tm.assert_series_equal(r, e)

    @pytest.mark.parametrize("other", [np.nan, 7, -23, 2.718, -3.14, np.inf])
    def test_ops_np_scalar(self, other):
        vals = np.random.default_rng(2).standard_normal((5, 3))
        f = lambda x: pd.DataFrame(
            x, index=list("ABCDE"), columns=["jim", "joe", "jolie"]
        )

        df = f(vals)

        tm.assert_frame_equal(df / np.array(other), f(vals / other))
        tm.assert_frame_equal(np.array(other) * df, f(vals * other))
        tm.assert_frame_equal(df + np.array(other), f(vals + other))
        tm.assert_frame_equal(np.array(other) - df, f(other - vals))

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_operators_frame(self):
        # rpow does not work with DataFrame
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        ts.name = "ts"

        df = pd.DataFrame({"A": ts})

        tm.assert_series_equal(ts + ts, ts + df["A"], check_names=False)
        tm.assert_series_equal(ts**ts, ts ** df["A"], check_names=False)
        tm.assert_series_equal(ts < ts, ts < df["A"], check_names=False)
        tm.assert_series_equal(ts / ts, ts / df["A"], check_names=False)

    # TODO: this came from tests.series.test_analytics, needs cleanup and
    #  de-duplication with test_modulo above
    def test_modulo2(self):
        with np.errstate(all="ignore"):
            # GH#3590, modulo as ints
            p = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
            result = p["first"] % p["second"]
            expected = Series(p["first"].values % p["second"].values, dtype="float64")
            expected.iloc[0:3] = np.nan
            tm.assert_series_equal(result, expected)

            result = p["first"] % 0
            expected = Series(np.nan, index=p.index, name="first")
            tm.assert_series_equal(result, expected)

            p = p.astype("float64")
            result = p["first"] % p["second"]
            expected = Series(p["first"].values % p["second"].values)
            tm.assert_series_equal(result, expected)

            p = p.astype("float64")
            result = p["first"] % p["second"]
            result2 = p["second"] % p["first"]
            assert not result.equals(result2)

    def test_modulo_zero_int(self):
        # GH#9144
        with np.errstate(all="ignore"):
            s = Series([0, 1])

            result = s % 0
            expected = Series([np.nan, np.nan])
            tm.assert_series_equal(result, expected)

            result = 0 % s
            expected = Series([np.nan, 0.0])
            tm.assert_series_equal(result, expected)


class TestAdditionSubtraction:
    # __add__, __sub__, __radd__, __rsub__, __iadd__, __isub__
    # for non-timestamp/timedelta/period dtypes

    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2], index=list("ABD"), name="x"),
                Series([3.0, 4.0, np.nan, np.nan], index=list("ABCD"), name="x"),
            ),
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2, 2], index=list("ABCD"), name="x"),
                Series([3, 4, 5, np.nan], index=list("ABCD"), name="x"),
            ),
        ],
    )
    def test_add_series(self, first, second, expected):
        # GH#1134
        tm.assert_series_equal(first + second, expected)
        tm.assert_series_equal(second + first, expected)

    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),
                pd.DataFrame({"x": [2, 2, 2]}, index=list("ABD")),
                pd.DataFrame({"x": [3.0, 4.0, np.nan, np.nan]}, index=list("ABCD")),
            ),
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),
                pd.DataFrame({"x": [2, 2, 2, 2]}, index=list("ABCD")),
                pd.DataFrame({"x": [3, 4, 5, np.nan]}, index=list("ABCD")),
            ),
        ],
    )
    def test_add_frames(self, first, second, expected):
        # GH#1134
        tm.assert_frame_equal(first + second, expected)
        tm.assert_frame_equal(second + first, expected)

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_series_frame_radd_bug(self, fixed_now_ts):
        # GH#353
        vals = Series([str(i) for i in range(5)])
        result = "foo_" + vals
        expected = vals.map(lambda x: "foo_" + x)
        tm.assert_series_equal(result, expected)

        frame = pd.DataFrame({"vals": vals})
        result = "foo_" + frame
        expected = pd.DataFrame({"vals": vals.map(lambda x: "foo_" + x)})
        tm.assert_frame_equal(result, expected)

        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        # really raise this time
        fix_now = fixed_now_ts.to_pydatetime()
        msg = "|".join(
            [
                "unsupported operand type",
                # wrong error message, see https://github.com/numpy/numpy/issues/18832
                "Concatenation operation",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            fix_now + ts

        with pytest.raises(TypeError, match=msg):
            ts + fix_now

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_datetime64_with_index(self):
        # arithmetic integer ops with an index
        ser = Series(np.random.default_rng(2).standard_normal(5))
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)

        # GH#4629
        # arithmetic datetime64 ops with an index
        ser = Series(
            date_range("20130101", periods=5),
            index=date_range("20130101", periods=5),
        )
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)

        msg = "cannot subtract PeriodArray from DatetimeArray"
        with pytest.raises(TypeError, match=msg):
            # GH#18850
            result = ser - ser.index.to_period()

        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=date_range("20130101", periods=5),
        )
        df["date"] = pd.Timestamp("20130102")
        df["expected"] = df["date"] - df.index.to_series()
        df["result"] = df["date"] - df.index
        tm.assert_series_equal(df["result"], df["expected"], check_names=False)

    # TODO: taken from tests.frame.test_operators, needs cleanup
    def test_frame_operators(self, float_frame):
        frame = float_frame

        garbage = np.random.default_rng(2).random(4)
        colSeries = Series(garbage, index=np.array(frame.columns))

        idSum = frame + frame
        seriesSum = frame + colSeries

        for col, series in idSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] * 2
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)

        for col, series in seriesSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] + colSeries[col]
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)

    def test_frame_operators_col_align(self, float_frame):
        frame2 = pd.DataFrame(float_frame, columns=["D", "C", "B", "A"])
        added = frame2 + frame2
        expected = frame2 * 2
        tm.assert_frame_equal(added, expected)

    def test_frame_operators_none_to_nan(self):
        df = pd.DataFrame({"a": ["a", None, "b"]})
        tm.assert_frame_equal(df + df, pd.DataFrame({"a": ["aa", np.nan, "bb"]}))

    @pytest.mark.parametrize("dtype", ("float", "int64"))
    def test_frame_operators_empty_like(self, dtype):
        # Test for issue #10181
        frames = [
            pd.DataFrame(dtype=dtype),
            pd.DataFrame(columns=["A"], dtype=dtype),
            pd.DataFrame(index=[0], dtype=dtype),
        ]
        for df in frames:
            assert (df + df).equals(df)
            tm.assert_frame_equal(df + df, df)

    @pytest.mark.parametrize(
        "func",
        [lambda x: x * 2, lambda x: x[::2], lambda x: 5],
        ids=["multiply", "slice", "constant"],
    )
    def test_series_operators_arithmetic(self, all_arithmetic_functions, func):
        op = all_arithmetic_functions
        series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = func(series)
        compare_op(series, other, op)

    @pytest.mark.parametrize(
        "func", [lambda x: x + 1, lambda x: 5], ids=["add", "constant"]
    )
    def test_series_operators_compare(self, comparison_op, func):
        op = comparison_op
        series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = func(series)
        compare_op(series, other, op)

    @pytest.mark.parametrize(
        "func",
        [lambda x: x * 2, lambda x: x[::2], lambda x: 5],
        ids=["multiply", "slice", "constant"],
    )
    def test_divmod(self, func):
        series = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = func(series)
        results = divmod(series, other)
        if isinstance(other, abc.Iterable) and len(series) != len(other):
            # if the lengths don't match, this is the test where we use
            # `tser[::2]`. Pad every other value in `other_np` with nan.
            other_np = []
            for n in other:
                other_np.append(n)
                other_np.append(np.nan)
        else:
            other_np = other
        other_np = np.asarray(other_np)
        with np.errstate(all="ignore"):
            expecteds = divmod(series.values, np.asarray(other_np))

        for result, expected in zip(results, expecteds):
            # check the values, name, and index separately
            tm.assert_almost_equal(np.asarray(result), expected)

            assert result.name == series.name
            tm.assert_index_equal(result.index, series.index._with_freq(None))

    def test_series_divmod_zero(self):
        # Check that divmod uses pandas convention for division by zero,
        #  which does not match numpy.
        # pandas convention has
        #  1/0 == np.inf
        #  -1/0 == -np.inf
        #  1/-0.0 == -np.inf
        #  -1/-0.0 == np.inf
        tser = Series(
            np.arange(1, 11, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        other = tser * 0

        result = divmod(tser, other)
        exp1 = Series([np.inf] * len(tser), index=tser.index, name="ts")
        exp2 = Series([np.nan] * len(tser), index=tser.index, name="ts")
        tm.assert_series_equal(result[0], exp1)
        tm.assert_series_equal(result[1], exp2)


class TestUFuncCompat:
    # TODO: add more dtypes
    @pytest.mark.parametrize("holder", [Index, RangeIndex, Series])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_ufunc_compat(self, holder, dtype):
        box = Series if holder is Series else Index

        if holder is RangeIndex:
            if dtype != np.int64:
                pytest.skip(f"dtype {dtype} not relevant for RangeIndex")
            idx = RangeIndex(0, 5, name="foo")
        else:
            idx = holder(np.arange(5, dtype=dtype), name="foo")
        result = np.sin(idx)
        expected = box(np.sin(np.arange(5, dtype=dtype)), name="foo")
        tm.assert_equal(result, expected)

    # TODO: add more dtypes
    @pytest.mark.parametrize("holder", [Index, Series])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_ufunc_coercions(self, holder, dtype):
        idx = holder([1, 2, 3, 4, 5], dtype=dtype, name="x")
        box = Series if holder is Series else Index

        result = np.sqrt(idx)
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index(np.sqrt(np.array([1, 2, 3, 4, 5], dtype=np.float64)), name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = np.divide(idx, 2.0)
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        # _evaluate_numeric_binop
        result = idx + 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx - 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx * 1.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx / 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

    # TODO: add more dtypes
    @pytest.mark.parametrize("holder", [Index, Series])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    def test_ufunc_multiple_return_values(self, holder, dtype):
        obj = holder([1, 2, 3], dtype=dtype, name="x")
        box = Series if holder is Series else Index

        result = np.modf(obj)
        assert isinstance(result, tuple)
        exp1 = Index([0.0, 0.0, 0.0], dtype=np.float64, name="x")
        exp2 = Index([1.0, 2.0, 3.0], dtype=np.float64, name="x")
        tm.assert_equal(result[0], tm.box_expected(exp1, box))
        tm.assert_equal(result[1], tm.box_expected(exp2, box))

    def test_ufunc_at(self):
        s = Series([0, 1, 2], index=[1, 2, 3], name="x")
        np.add.at(s, [0, 2], 10)
        expected = Series([10, 1, 12], index=[1, 2, 3], name="x")
        tm.assert_series_equal(s, expected)


class TestObjectDtypeEquivalence:
    # Tests that arithmetic operations match operations executed elementwise

    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_nan(self, dtype, box_with_array):
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([np.nan, np.nan, np.nan], dtype=dtype)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = np.nan + ser
        tm.assert_equal(result, expected)

        result = ser + np.nan
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_int(self, dtype, box_with_array):
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([2, 3, 4], dtype=dtype)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = 1 + ser
        tm.assert_equal(result, expected)

        result = ser + 1
        tm.assert_equal(result, expected)

    # TODO: moved from tests.series.test_operators; needs cleanup
    @pytest.mark.parametrize(
        "op",
        [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
    )
    def test_operators_reverse_object(self, op):
        # GH#56
        arr = Series(
            np.random.default_rng(2).standard_normal(10),
            index=np.arange(10),
            dtype=object,
        )

        result = op(1.0, arr)
        expected = op(1.0, arr.astype(float))
        tm.assert_series_equal(result.astype(float), expected)


class TestNumericArithmeticUnsorted:
    # Tests in this class have been moved from type-specific test modules
    #  but not yet sorted, parametrized, and de-duplicated
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
        ],
    )
    @pytest.mark.parametrize(
        "idx1",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    @pytest.mark.parametrize(
        "idx2",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    def test_binops_index(self, op, idx1, idx2):
        idx1 = idx1._rename("foo")
        idx2 = idx2._rename("bar")
        result = op(idx1, idx2)
        expected = op(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        tm.assert_index_equal(result, expected, exact="equiv")

    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
        ],
    )
    @pytest.mark.parametrize(
        "idx",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    @pytest.mark.parametrize("scalar", [-1, 1, 2])
    def test_binops_index_scalar(self, op, idx, scalar):
        result = op(idx, scalar)
        expected = op(Index(idx.to_numpy()), scalar)
        tm.assert_index_equal(result, expected, exact="equiv")

    @pytest.mark.parametrize("idx1", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    @pytest.mark.parametrize("idx2", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    def test_binops_index_pow(self, idx1, idx2):
        # numpy does not allow powers of negative integers so test separately
        # https://github.com/numpy/numpy/pull/8127
        idx1 = idx1._rename("foo")
        idx2 = idx2._rename("bar")
        result = pow(idx1, idx2)
        expected = pow(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        tm.assert_index_equal(result, expected, exact="equiv")

    @pytest.mark.parametrize("idx", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    @pytest.mark.parametrize("scalar", [1, 2])
    def test_binops_index_scalar_pow(self, idx, scalar):
        # numpy does not allow powers of negative integers so test separately
        # https://github.com/numpy/numpy/pull/8127
        result = pow(idx, scalar)
        expected = pow(Index(idx.to_numpy()), scalar)
        tm.assert_index_equal(result, expected, exact="equiv")

    # TODO: divmod?
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
            operator.pow,
            operator.mod,
        ],
    )
    def test_arithmetic_with_frame_or_series(self, op):
        # check that we return NotImplemented when operating with Series
        # or DataFrame
        index = RangeIndex(5)
        other = Series(np.random.default_rng(2).standard_normal(5))

        expected = op(Series(index), other)
        result = op(index, other)
        tm.assert_series_equal(result, expected)

        other = pd.DataFrame(np.random.default_rng(2).standard_normal((2, 5)))
        expected = op(pd.DataFrame([index, index]), other)
        result = op(index, other)
        tm.assert_frame_equal(result, expected)

    def test_numeric_compat2(self):
        # validate that we are handling the RangeIndex overrides to numeric ops
        # and returning RangeIndex where possible

        idx = RangeIndex(0, 10, 2)

        result = idx * 2
        expected = RangeIndex(0, 20, 4)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx + 2
        expected = RangeIndex(2, 12, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx - 2
        expected = RangeIndex(-2, 8, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx / 2
        expected = RangeIndex(0, 5, 1).astype("float64")
        tm.assert_index_equal(result, expected, exact=True)

        result = idx / 4
        expected = RangeIndex(0, 10, 2) / 4
        tm.assert_index_equal(result, expected, exact=True)

        result = idx // 1
        expected = idx
        tm.assert_index_equal(result, expected, exact=True)

        # __mul__
        result = idx * idx
        expected = Index(idx.values * idx.values)
        tm.assert_index_equal(result, expected, exact=True)

        # __pow__
        idx = RangeIndex(0, 1000, 2)
        result = idx**2
        expected = Index(idx._values) ** 2
        tm.assert_index_equal(Index(result.values), expected, exact=True)

    @pytest.mark.parametrize(
        "idx, div, expected",
        [
            # TODO: add more dtypes
            (RangeIndex(0, 1000, 2), 2, RangeIndex(0, 500, 1)),
            (RangeIndex(-99, -201, -3), -3, RangeIndex(33, 67, 1)),
            (
                RangeIndex(0, 1000, 1),
                2,
                Index(RangeIndex(0, 1000, 1)._values) // 2,
            ),
            (
                RangeIndex(0, 100, 1),
                2.0,
                Index(RangeIndex(0, 100, 1)._values) // 2.0,
            ),
            (RangeIndex(0), 50, RangeIndex(0)),
            (RangeIndex(2, 4, 2), 3, RangeIndex(0, 1, 1)),
            (RangeIndex(-5, -10, -6), 4, RangeIndex(-2, -1, 1)),
            (RangeIndex(-100, -200, 3), 2, RangeIndex(0)),
        ],
    )
    def test_numeric_compat2_floordiv(self, idx, div, expected):
        # __floordiv__
        tm.assert_index_equal(idx // div, expected, exact=True)

    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("delta", [1, 0, -1])
    def test_addsub_arithmetic(self, dtype, delta):
        # GH#8142
        delta = dtype(delta)
        index = Index([10, 11, 12], dtype=dtype)
        result = index + delta
        expected = Index(index.values + delta, dtype=dtype)
        tm.assert_index_equal(result, expected)

        # this subtraction used to fail
        result = index - delta
        expected = Index(index.values - delta, dtype=dtype)
        tm.assert_index_equal(result, expected)

        tm.assert_index_equal(index + index, 2 * index)
        tm.assert_index_equal(index - index, 0 * index)
        assert not (index - index).empty

    def test_pow_nan_with_zero(self, box_with_array):
        left = Index([np.nan, np.nan, np.nan])
        right = Index([0, 0, 0])
        expected = Index([1.0, 1.0, 1.0])

        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = left**right
        tm.assert_equal(result, expected)


def test_fill_value_inf_masking():
    # GH #27464 make sure we mask 0/1 with Inf and not NaN
    df = pd.DataFrame({"A": [0, 1, 2], "B": [1.1, None, 1.1]})

    other = pd.DataFrame({"A": [1.1, 1.2, 1.3]}, index=[0, 2, 3])

    result = df.rfloordiv(other, fill_value=1)

    expected = pd.DataFrame(
        {"A": [np.inf, 1.0, 0.0, 1.0], "B": [0.0, np.nan, 0.0, np.nan]}
    )
    tm.assert_frame_equal(result, expected)


def test_dataframe_div_silenced():
    # GH#26793
    pdf1 = pd.DataFrame(
        {
            "A": np.arange(10),
            "B": [np.nan, 1, 2, 3, 4] * 2,
            "C": [np.nan] * 10,
            "D": np.arange(10),
        },
        index=list("abcdefghij"),
        columns=list("ABCD"),
    )
    pdf2 = pd.DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        index=list("abcdefghjk"),
        columns=list("ABCX"),
    )
    with tm.assert_produces_warning(None):
        pdf1.div(pdf2, fill_value=0)


@pytest.mark.parametrize(
    "data, expected_data",
    [([0, 1, 2], [0, 2, 4])],
)
def test_integer_array_add_list_like(
    box_pandas_1d_array, box_1d_array, data, expected_data
):
    # GH22606 Verify operators with IntegerArray and list-likes
    arr = array(data, dtype="Int64")
    container = box_pandas_1d_array(arr)
    left = container + box_1d_array(data)
    right = box_1d_array(data) + container

    if Series in [box_1d_array, box_pandas_1d_array]:
        cls = Series
    elif Index in [box_1d_array, box_pandas_1d_array]:
        cls = Index
    else:
        cls = array

    expected = cls(expected_data, dtype="Int64")

    tm.assert_equal(left, expected)
    tm.assert_equal(right, expected)


def test_sub_multiindex_swapped_levels():
    # GH 9952
    df = pd.DataFrame(
        {"a": np.random.default_rng(2).standard_normal(6)},
        index=pd.MultiIndex.from_product(
            [["a", "b"], [0, 1, 2]], names=["levA", "levB"]
        ),
    )
    df2 = df.copy()
    df2.index = df2.index.swaplevel(0, 1)
    result = df - df2
    expected = pd.DataFrame([0.0] * 6, columns=["a"], index=df.index)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("power", [1, 2, 5])
@pytest.mark.parametrize("string_size", [0, 1, 2, 5])
def test_empty_str_comparison(power, string_size):
    # GH 37348
    a = np.array(range(10**power))
    right = pd.DataFrame(a, dtype=np.int64)
    left = " " * string_size

    result = right == left
    expected = pd.DataFrame(np.zeros(right.shape, dtype=bool))
    tm.assert_frame_equal(result, expected)


def test_series_add_sub_with_UInt64():
    # GH 22023
    series1 = Series([1, 2, 3])
    series2 = Series([2, 1, 3], dtype="UInt64")

    result = series1 + series2
    expected = Series([3, 3, 6], dtype="Float64")
    tm.assert_series_equal(result, expected)

    result = series1 - series2
    expected = Series([-1, 1, 0], dtype="Float64")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_hyperbolic.py ===
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.function import (expand_mul, expand_trig)
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)
from sympy.series.order import O

from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError, PoleError
from sympy.testing.pytest import raises


def test_sinh():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert sinh(nan) is nan
    assert sinh(zoo) is nan

    assert sinh(oo) is oo
    assert sinh(-oo) is -oo

    assert sinh(0) == 0

    assert unchanged(sinh, 1)
    assert sinh(-1) == -sinh(1)

    assert unchanged(sinh, x)
    assert sinh(-x) == -sinh(x)

    assert unchanged(sinh, pi)
    assert sinh(-pi) == -sinh(pi)

    assert unchanged(sinh, 2**1024 * E)
    assert sinh(-2**1024 * E) == -sinh(2**1024 * E)

    assert sinh(pi*I) == 0
    assert sinh(-pi*I) == 0
    assert sinh(2*pi*I) == 0
    assert sinh(-2*pi*I) == 0
    assert sinh(-3*10**73*pi*I) == 0
    assert sinh(7*10**103*pi*I) == 0

    assert sinh(pi*I/2) == I
    assert sinh(-pi*I/2) == -I
    assert sinh(pi*I*Rational(5, 2)) == I
    assert sinh(pi*I*Rational(7, 2)) == -I

    assert sinh(pi*I/3) == S.Half*sqrt(3)*I
    assert sinh(pi*I*Rational(-2, 3)) == Rational(-1, 2)*sqrt(3)*I

    assert sinh(pi*I/4) == S.Half*sqrt(2)*I
    assert sinh(-pi*I/4) == Rational(-1, 2)*sqrt(2)*I
    assert sinh(pi*I*Rational(17, 4)) == S.Half*sqrt(2)*I
    assert sinh(pi*I*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)*I

    assert sinh(pi*I/6) == S.Half*I
    assert sinh(-pi*I/6) == Rational(-1, 2)*I
    assert sinh(pi*I*Rational(7, 6)) == Rational(-1, 2)*I
    assert sinh(pi*I*Rational(-5, 6)) == Rational(-1, 2)*I

    assert sinh(pi*I/105) == sin(pi/105)*I
    assert sinh(-pi*I/105) == -sin(pi/105)*I

    assert unchanged(sinh, 2 + 3*I)

    assert sinh(x*I) == sin(x)*I

    assert sinh(k*pi*I) == 0
    assert sinh(17*k*pi*I) == 0

    assert sinh(k*pi*I/2) == sin(k*pi/2)*I

    assert sinh(x).as_real_imag(deep=False) == (cos(im(x))*sinh(re(x)),
                sin(im(x))*cosh(re(x)))
    x = Symbol('x', extended_real=True)
    assert sinh(x).as_real_imag(deep=False) == (sinh(x), 0)

    x = Symbol('x', real=True)
    assert sinh(I*x).is_finite is True
    assert sinh(x).is_real is True
    assert sinh(I).is_real is False
    p = Symbol('p', positive=True)
    assert sinh(p).is_zero is False
    assert sinh(0, evaluate=False).is_zero is True
    assert sinh(2*pi*I, evaluate=False).is_zero is True


def test_sinh_series():
    x = Symbol('x')
    assert sinh(x).series(x, 0, 10) == \
        x + x**3/6 + x**5/120 + x**7/5040 + x**9/362880 + O(x**10)


def test_sinh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: sinh(x).fdiff(2))


def test_cosh():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert cosh(nan) is nan
    assert cosh(zoo) is nan

    assert cosh(oo) is oo
    assert cosh(-oo) is oo

    assert cosh(0) == 1

    assert unchanged(cosh, 1)
    assert cosh(-1) == cosh(1)

    assert unchanged(cosh, x)
    assert cosh(-x) == cosh(x)

    assert cosh(pi*I) == cos(pi)
    assert cosh(-pi*I) == cos(pi)

    assert unchanged(cosh, 2**1024 * E)
    assert cosh(-2**1024 * E) == cosh(2**1024 * E)

    assert cosh(pi*I/2) == 0
    assert cosh(-pi*I/2) == 0
    assert cosh((-3*10**73 + 1)*pi*I/2) == 0
    assert cosh((7*10**103 + 1)*pi*I/2) == 0

    assert cosh(pi*I) == -1
    assert cosh(-pi*I) == -1
    assert cosh(5*pi*I) == -1
    assert cosh(8*pi*I) == 1

    assert cosh(pi*I/3) == S.Half
    assert cosh(pi*I*Rational(-2, 3)) == Rational(-1, 2)

    assert cosh(pi*I/4) == S.Half*sqrt(2)
    assert cosh(-pi*I/4) == S.Half*sqrt(2)
    assert cosh(pi*I*Rational(11, 4)) == Rational(-1, 2)*sqrt(2)
    assert cosh(pi*I*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert cosh(pi*I/6) == S.Half*sqrt(3)
    assert cosh(-pi*I/6) == S.Half*sqrt(3)
    assert cosh(pi*I*Rational(7, 6)) == Rational(-1, 2)*sqrt(3)
    assert cosh(pi*I*Rational(-5, 6)) == Rational(-1, 2)*sqrt(3)

    assert cosh(pi*I/105) == cos(pi/105)
    assert cosh(-pi*I/105) == cos(pi/105)

    assert unchanged(cosh, 2 + 3*I)

    assert cosh(x*I) == cos(x)

    assert cosh(k*pi*I) == cos(k*pi)
    assert cosh(17*k*pi*I) == cos(17*k*pi)

    assert unchanged(cosh, k*pi)

    assert cosh(x).as_real_imag(deep=False) == (cos(im(x))*cosh(re(x)),
                sin(im(x))*sinh(re(x)))
    x = Symbol('x', extended_real=True)
    assert cosh(x).as_real_imag(deep=False) == (cosh(x), 0)

    x = Symbol('x', real=True)
    assert cosh(I*x).is_finite is True
    assert cosh(I*x).is_real is True
    assert cosh(I*2 + 1).is_real is False
    assert cosh(5*I*S.Pi/2, evaluate=False).is_zero is True
    assert cosh(x).is_zero is False


def test_cosh_series():
    x = Symbol('x')
    assert cosh(x).series(x, 0, 10) == \
        1 + x**2/2 + x**4/24 + x**6/720 + x**8/40320 + O(x**10)


def test_cosh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: cosh(x).fdiff(2))


def test_tanh():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert tanh(nan) is nan
    assert tanh(zoo) is nan

    assert tanh(oo) == 1
    assert tanh(-oo) == -1

    assert tanh(0) == 0

    assert unchanged(tanh, 1)
    assert tanh(-1) == -tanh(1)

    assert unchanged(tanh, x)
    assert tanh(-x) == -tanh(x)

    assert unchanged(tanh, pi)
    assert tanh(-pi) == -tanh(pi)

    assert unchanged(tanh, 2**1024 * E)
    assert tanh(-2**1024 * E) == -tanh(2**1024 * E)

    assert tanh(pi*I) == 0
    assert tanh(-pi*I) == 0
    assert tanh(2*pi*I) == 0
    assert tanh(-2*pi*I) == 0
    assert tanh(-3*10**73*pi*I) == 0
    assert tanh(7*10**103*pi*I) == 0

    assert tanh(pi*I/2) is zoo
    assert tanh(-pi*I/2) is zoo
    assert tanh(pi*I*Rational(5, 2)) is zoo
    assert tanh(pi*I*Rational(7, 2)) is zoo

    assert tanh(pi*I/3) == sqrt(3)*I
    assert tanh(pi*I*Rational(-2, 3)) == sqrt(3)*I

    assert tanh(pi*I/4) == I
    assert tanh(-pi*I/4) == -I
    assert tanh(pi*I*Rational(17, 4)) == I
    assert tanh(pi*I*Rational(-3, 4)) == I

    assert tanh(pi*I/6) == I/sqrt(3)
    assert tanh(-pi*I/6) == -I/sqrt(3)
    assert tanh(pi*I*Rational(7, 6)) == I/sqrt(3)
    assert tanh(pi*I*Rational(-5, 6)) == I/sqrt(3)

    assert tanh(pi*I/105) == tan(pi/105)*I
    assert tanh(-pi*I/105) == -tan(pi/105)*I

    assert unchanged(tanh, 2 + 3*I)

    assert tanh(x*I) == tan(x)*I

    assert tanh(k*pi*I) == 0
    assert tanh(17*k*pi*I) == 0

    assert tanh(k*pi*I/2) == tan(k*pi/2)*I

    assert tanh(x).as_real_imag(deep=False) == (sinh(re(x))*cosh(re(x))/(cos(im(x))**2
                                + sinh(re(x))**2),
                                sin(im(x))*cos(im(x))/(cos(im(x))**2 + sinh(re(x))**2))
    x = Symbol('x', extended_real=True)
    assert tanh(x).as_real_imag(deep=False) == (tanh(x), 0)
    assert tanh(I*pi/3 + 1).is_real is False
    assert tanh(x).is_real is True
    assert tanh(I*pi*x/2).is_real is None


def test_tanh_series():
    x = Symbol('x')
    assert tanh(x).series(x, 0, 10) == \
        x - x**3/3 + 2*x**5/15 - 17*x**7/315 + 62*x**9/2835 + O(x**10)


def test_tanh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: tanh(x).fdiff(2))


def test_coth():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert coth(nan) is nan
    assert coth(zoo) is nan

    assert coth(oo) == 1
    assert coth(-oo) == -1

    assert coth(0) is zoo
    assert unchanged(coth, 1)
    assert coth(-1) == -coth(1)

    assert unchanged(coth, x)
    assert coth(-x) == -coth(x)

    assert coth(pi*I) == -I*cot(pi)
    assert coth(-pi*I) == cot(pi)*I

    assert unchanged(coth, 2**1024 * E)
    assert coth(-2**1024 * E) == -coth(2**1024 * E)

    assert coth(pi*I) == -I*cot(pi)
    assert coth(-pi*I) == I*cot(pi)
    assert coth(2*pi*I) == -I*cot(2*pi)
    assert coth(-2*pi*I) == I*cot(2*pi)
    assert coth(-3*10**73*pi*I) == I*cot(3*10**73*pi)
    assert coth(7*10**103*pi*I) == -I*cot(7*10**103*pi)

    assert coth(pi*I/2) == 0
    assert coth(-pi*I/2) == 0
    assert coth(pi*I*Rational(5, 2)) == 0
    assert coth(pi*I*Rational(7, 2)) == 0

    assert coth(pi*I/3) == -I/sqrt(3)
    assert coth(pi*I*Rational(-2, 3)) == -I/sqrt(3)

    assert coth(pi*I/4) == -I
    assert coth(-pi*I/4) == I
    assert coth(pi*I*Rational(17, 4)) == -I
    assert coth(pi*I*Rational(-3, 4)) == -I

    assert coth(pi*I/6) == -sqrt(3)*I
    assert coth(-pi*I/6) == sqrt(3)*I
    assert coth(pi*I*Rational(7, 6)) == -sqrt(3)*I
    assert coth(pi*I*Rational(-5, 6)) == -sqrt(3)*I

    assert coth(pi*I/105) == -cot(pi/105)*I
    assert coth(-pi*I/105) == cot(pi/105)*I

    assert unchanged(coth, 2 + 3*I)

    assert coth(x*I) == -cot(x)*I

    assert coth(k*pi*I) == -cot(k*pi)*I
    assert coth(17*k*pi*I) == -cot(17*k*pi)*I

    assert coth(k*pi*I) == -cot(k*pi)*I

    assert coth(log(tan(2))) == coth(log(-tan(2)))
    assert coth(1 + I*pi/2) == tanh(1)

    assert coth(x).as_real_imag(deep=False) == (sinh(re(x))*cosh(re(x))/(sin(im(x))**2
                                + sinh(re(x))**2),
                                -sin(im(x))*cos(im(x))/(sin(im(x))**2 + sinh(re(x))**2))
    x = Symbol('x', extended_real=True)
    assert coth(x).as_real_imag(deep=False) == (coth(x), 0)

    assert expand_trig(coth(2*x)) == (coth(x)**2 + 1)/(2*coth(x))
    assert expand_trig(coth(3*x)) == (coth(x)**3 + 3*coth(x))/(1 + 3*coth(x)**2)

    assert expand_trig(coth(x + y)) == (1 + coth(x)*coth(y))/(coth(x) + coth(y))


def test_coth_series():
    x = Symbol('x')
    assert coth(x).series(x, 0, 8) == \
        1/x + x/3 - x**3/45 + 2*x**5/945 - x**7/4725 + O(x**8)


def test_coth_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: coth(x).fdiff(2))


def test_csch():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)
    n = Symbol('n', positive=True)

    assert csch(nan) is nan
    assert csch(zoo) is nan

    assert csch(oo) == 0
    assert csch(-oo) == 0

    assert csch(0) is zoo

    assert csch(-1) == -csch(1)

    assert csch(-x) == -csch(x)
    assert csch(-pi) == -csch(pi)
    assert csch(-2**1024 * E) == -csch(2**1024 * E)

    assert csch(pi*I) is zoo
    assert csch(-pi*I) is zoo
    assert csch(2*pi*I) is zoo
    assert csch(-2*pi*I) is zoo
    assert csch(-3*10**73*pi*I) is zoo
    assert csch(7*10**103*pi*I) is zoo

    assert csch(pi*I/2) == -I
    assert csch(-pi*I/2) == I
    assert csch(pi*I*Rational(5, 2)) == -I
    assert csch(pi*I*Rational(7, 2)) == I

    assert csch(pi*I/3) == -2/sqrt(3)*I
    assert csch(pi*I*Rational(-2, 3)) == 2/sqrt(3)*I

    assert csch(pi*I/4) == -sqrt(2)*I
    assert csch(-pi*I/4) == sqrt(2)*I
    assert csch(pi*I*Rational(7, 4)) == sqrt(2)*I
    assert csch(pi*I*Rational(-3, 4)) == sqrt(2)*I

    assert csch(pi*I/6) == -2*I
    assert csch(-pi*I/6) == 2*I
    assert csch(pi*I*Rational(7, 6)) == 2*I
    assert csch(pi*I*Rational(-7, 6)) == -2*I
    assert csch(pi*I*Rational(-5, 6)) == 2*I

    assert csch(pi*I/105) == -1/sin(pi/105)*I
    assert csch(-pi*I/105) == 1/sin(pi/105)*I

    assert csch(x*I) == -1/sin(x)*I

    assert csch(k*pi*I) is zoo
    assert csch(17*k*pi*I) is zoo

    assert csch(k*pi*I/2) == -1/sin(k*pi/2)*I

    assert csch(n).is_real is True

    assert expand_trig(csch(x + y)) == 1/(sinh(x)*cosh(y) + cosh(x)*sinh(y))


def test_csch_series():
    x = Symbol('x')
    assert csch(x).series(x, 0, 10) == \
       1/ x - x/6 + 7*x**3/360 - 31*x**5/15120 + 127*x**7/604800 \
          - 73*x**9/3421440 + O(x**10)


def test_csch_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: csch(x).fdiff(2))


def test_sech():
    x, y = symbols('x, y')

    k = Symbol('k', integer=True)
    n = Symbol('n', positive=True)

    assert sech(nan) is nan
    assert sech(zoo) is nan

    assert sech(oo) == 0
    assert sech(-oo) == 0

    assert sech(0) == 1

    assert sech(-1) == sech(1)
    assert sech(-x) == sech(x)

    assert sech(pi*I) == sec(pi)

    assert sech(-pi*I) == sec(pi)
    assert sech(-2**1024 * E) == sech(2**1024 * E)

    assert sech(pi*I/2) is zoo
    assert sech(-pi*I/2) is zoo
    assert sech((-3*10**73 + 1)*pi*I/2) is zoo
    assert sech((7*10**103 + 1)*pi*I/2) is zoo

    assert sech(pi*I) == -1
    assert sech(-pi*I) == -1
    assert sech(5*pi*I) == -1
    assert sech(8*pi*I) == 1

    assert sech(pi*I/3) == 2
    assert sech(pi*I*Rational(-2, 3)) == -2

    assert sech(pi*I/4) == sqrt(2)
    assert sech(-pi*I/4) == sqrt(2)
    assert sech(pi*I*Rational(5, 4)) == -sqrt(2)
    assert sech(pi*I*Rational(-5, 4)) == -sqrt(2)

    assert sech(pi*I/6) == 2/sqrt(3)
    assert sech(-pi*I/6) == 2/sqrt(3)
    assert sech(pi*I*Rational(7, 6)) == -2/sqrt(3)
    assert sech(pi*I*Rational(-5, 6)) == -2/sqrt(3)

    assert sech(pi*I/105) == 1/cos(pi/105)
    assert sech(-pi*I/105) == 1/cos(pi/105)

    assert sech(x*I) == 1/cos(x)

    assert sech(k*pi*I) == 1/cos(k*pi)
    assert sech(17*k*pi*I) == 1/cos(17*k*pi)

    assert sech(n).is_real is True

    assert expand_trig(sech(x + y)) == 1/(cosh(x)*cosh(y) + sinh(x)*sinh(y))


def test_sech_series():
    x = Symbol('x')
    assert sech(x).series(x, 0, 10) == \
        1 - x**2/2 + 5*x**4/24 - 61*x**6/720 + 277*x**8/8064 + O(x**10)


def test_sech_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: sech(x).fdiff(2))


def test_asinh():
    x, y = symbols('x,y')
    assert unchanged(asinh, x)
    assert asinh(-x) == -asinh(x)

    # at specific points
    assert asinh(nan) is nan
    assert asinh( 0) == 0
    assert asinh(+1) == log(sqrt(2) + 1)

    assert asinh(-1) == log(sqrt(2) - 1)
    assert asinh(I) == pi*I/2
    assert asinh(-I) == -pi*I/2
    assert asinh(I/2) == pi*I/6
    assert asinh(-I/2) == -pi*I/6

    # at infinites
    assert asinh(oo) is oo
    assert asinh(-oo) is -oo

    assert asinh(I*oo) is oo
    assert asinh(-I *oo) is -oo

    assert asinh(zoo) is zoo

    # properties
    assert asinh(I *(sqrt(3) - 1)/(2**Rational(3, 2))) == pi*I/12
    assert asinh(-I *(sqrt(3) - 1)/(2**Rational(3, 2))) == -pi*I/12

    assert asinh(I*(sqrt(5) - 1)/4) == pi*I/10
    assert asinh(-I*(sqrt(5) - 1)/4) == -pi*I/10

    assert asinh(I*(sqrt(5) + 1)/4) == pi*I*Rational(3, 10)
    assert asinh(-I*(sqrt(5) + 1)/4) == pi*I*Rational(-3, 10)

    # reality
    assert asinh(S(2)).is_real is True
    assert asinh(S(2)).is_finite is True
    assert asinh(S(-2)).is_real is True
    assert asinh(S(oo)).is_extended_real is True
    assert asinh(-S(oo)).is_real is False
    assert (asinh(2) - oo) == -oo
    assert asinh(symbols('y', real=True)).is_real is True

    # Symmetry
    assert asinh(Rational(-1, 2)) == -asinh(S.Half)

    # inverse composition
    assert unchanged(asinh, sinh(Symbol('v1')))

    assert asinh(sinh(0, evaluate=False)) == 0
    assert asinh(sinh(-3, evaluate=False)) == -3
    assert asinh(sinh(2, evaluate=False)) == 2
    assert asinh(sinh(I, evaluate=False)) == I
    assert asinh(sinh(-I, evaluate=False)) == -I
    assert asinh(sinh(5*I, evaluate=False)) == -2*I*pi + 5*I
    assert asinh(sinh(15 + 11*I)) == 15 - 4*I*pi + 11*I
    assert asinh(sinh(-73 + 97*I)) == 73 - 97*I + 31*I*pi
    assert asinh(sinh(-7 - 23*I)) == 7 - 7*I*pi + 23*I
    assert asinh(sinh(13 - 3*I)) == -13 - I*pi + 3*I
    p = Symbol('p', positive=True)
    assert asinh(p).is_zero is False
    assert asinh(sinh(0, evaluate=False), evaluate=False).is_zero is True


def test_asinh_rewrite():
    x = Symbol('x')
    assert asinh(x).rewrite(log) == log(x + sqrt(x**2 + 1))
    assert asinh(x).rewrite(atanh) == atanh(x/sqrt(1 + x**2))
    assert asinh(x).rewrite(asin) == -I*asin(I*x, evaluate=False)
    assert asinh(x*(1 + I)).rewrite(asin) == -I*asin(I*x*(1+I))
    assert asinh(x).rewrite(acos) == I*acos(I*x, evaluate=False) - I*pi/2


def test_asinh_leading_term():
    x = Symbol('x')
    assert asinh(x).as_leading_term(x, cdir=1) == x
    # Tests concerning branch points
    assert asinh(x + I).as_leading_term(x, cdir=1) == I*pi/2
    assert asinh(x - I).as_leading_term(x, cdir=1) == -I*pi/2
    assert asinh(1/x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert asinh(1/x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I*pi
    # Tests concerning points lying on branch cuts
    assert asinh(x + 2*I).as_leading_term(x, cdir=1) == I*asin(2)
    assert asinh(x + 2*I).as_leading_term(x, cdir=-1) == -I*asin(2) + I*pi
    assert asinh(x - 2*I).as_leading_term(x, cdir=1) == -I*pi + I*asin(2)
    assert asinh(x - 2*I).as_leading_term(x, cdir=-1) == -I*asin(2)
    # Tests concerning re(ndir) == 0
    assert asinh(2*I + I*x - x**2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) + I*pi/2
    assert asinh(2*I + I*x - x**2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) + I*pi/2


def test_asinh_series():
    x = Symbol('x')
    assert asinh(x).series(x, 0, 8) == \
        x - x**3/6 + 3*x**5/40 - 5*x**7/112 + O(x**8)
    t5 = asinh(x).taylor_term(5, x)
    assert t5 == 3*x**5/40
    assert asinh(x).taylor_term(7, x, t5, 0) == -5*x**7/112


def test_asinh_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert asinh(x + I)._eval_nseries(x, 4, None) == I*pi/2 - \
    sqrt(2)*sqrt(I)*I*sqrt(x) + sqrt(2)*sqrt(I)*x**(S(3)/2)/12 + 3*sqrt(2)*sqrt(I)*I*x**(S(5)/2)/160 - \
    5*sqrt(2)*sqrt(I)*x**(S(7)/2)/896 + O(x**4)
    assert asinh(x - I)._eval_nseries(x, 4, None) == -I*pi/2 + \
    sqrt(2)*I*sqrt(x)*sqrt(-I) + sqrt(2)*x**(S(3)/2)*sqrt(-I)/12 - \
    3*sqrt(2)*I*x**(S(5)/2)*sqrt(-I)/160 - 5*sqrt(2)*x**(S(7)/2)*sqrt(-I)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert asinh(x + 2*I)._eval_nseries(x, 4, None, cdir=1) == I*asin(2) - \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x + 2*I)._eval_nseries(x, 4, None, cdir=-1) == I*pi - I*asin(2) + \
    sqrt(3)*I*x/3 - sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x - 2*I)._eval_nseries(x, 4, None, cdir=1) == I*asin(2) - I*pi + \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x - 2*I)._eval_nseries(x, 4, None, cdir=-1) == -I*asin(2) - \
    sqrt(3)*I*x/3 - sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    # Tests concerning re(ndir) == 0
    assert asinh(2*I + I*x - x**2)._eval_nseries(x, 4, None) == I*pi/2 + log(2 - sqrt(3)) + \
    x*(-3 + 2*sqrt(3))/(-6 + 3*sqrt(3)) + x**2*(12 - 36*I + sqrt(3)*(-7 + 21*I))/(-63 + \
    36*sqrt(3)) + x**3*(-168 + sqrt(3)*(97 - 388*I) + 672*I)/(-1746 + 1008*sqrt(3)) + O(x**4)


def test_asinh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: asinh(x).fdiff(2))


def test_acosh():
    x = Symbol('x')

    assert unchanged(acosh, -x)

    #at specific points
    assert acosh(1) == 0
    assert acosh(-1) == pi*I
    assert acosh(0) == I*pi/2
    assert acosh(S.Half) == I*pi/3
    assert acosh(Rational(-1, 2)) == pi*I*Rational(2, 3)
    assert acosh(nan) is nan

    # at infinites
    assert acosh(oo) is oo
    assert acosh(-oo) is oo

    assert acosh(I*oo) == oo + I*pi/2
    assert acosh(-I*oo) == oo - I*pi/2

    assert acosh(zoo) is zoo

    assert acosh(I) == log(I*(1 + sqrt(2)))
    assert acosh(-I) == log(-I*(1 + sqrt(2)))
    assert acosh((sqrt(3) - 1)/(2*sqrt(2))) == pi*I*Rational(5, 12)
    assert acosh(-(sqrt(3) - 1)/(2*sqrt(2))) == pi*I*Rational(7, 12)
    assert acosh(sqrt(2)/2) == I*pi/4
    assert acosh(-sqrt(2)/2) == I*pi*Rational(3, 4)
    assert acosh(sqrt(3)/2) == I*pi/6
    assert acosh(-sqrt(3)/2) == I*pi*Rational(5, 6)
    assert acosh(sqrt(2 + sqrt(2))/2) == I*pi/8
    assert acosh(-sqrt(2 + sqrt(2))/2) == I*pi*Rational(7, 8)
    assert acosh(sqrt(2 - sqrt(2))/2) == I*pi*Rational(3, 8)
    assert acosh(-sqrt(2 - sqrt(2))/2) == I*pi*Rational(5, 8)
    assert acosh((1 + sqrt(3))/(2*sqrt(2))) == I*pi/12
    assert acosh(-(1 + sqrt(3))/(2*sqrt(2))) == I*pi*Rational(11, 12)
    assert acosh((sqrt(5) + 1)/4) == I*pi/5
    assert acosh(-(sqrt(5) + 1)/4) == I*pi*Rational(4, 5)

    assert str(acosh(5*I).n(6)) == '2.31244 + 1.5708*I'
    assert str(acosh(-5*I).n(6)) == '2.31244 - 1.5708*I'

    # inverse composition
    assert unchanged(acosh, Symbol('v1'))

    assert acosh(cosh(-3, evaluate=False)) == 3
    assert acosh(cosh(3, evaluate=False)) == 3
    assert acosh(cosh(0, evaluate=False)) == 0
    assert acosh(cosh(I, evaluate=False)) == I
    assert acosh(cosh(-I, evaluate=False)) == I
    assert acosh(cosh(7*I, evaluate=False)) == -2*I*pi + 7*I
    assert acosh(cosh(1 + I)) == 1 + I
    assert acosh(cosh(3 - 3*I)) == 3 - 3*I
    assert acosh(cosh(-3 + 2*I)) == 3 - 2*I
    assert acosh(cosh(-5 - 17*I)) == 5 - 6*I*pi + 17*I
    assert acosh(cosh(-21 + 11*I)) == 21 - 11*I + 4*I*pi
    assert acosh(cosh(cosh(1) + I)) == cosh(1) + I
    assert acosh(1, evaluate=False).is_zero is True

    # Reality
    assert acosh(S(2)).is_real is True
    assert acosh(S(2)).is_extended_real is True
    assert acosh(oo).is_extended_real is True
    assert acosh(S(2)).is_finite is True
    assert acosh(S(1) / 5).is_real is False
    assert (acosh(2) - oo) == -oo
    assert acosh(symbols('y', real=True)).is_real is None


def test_acosh_rewrite():
    x = Symbol('x')
    assert acosh(x).rewrite(log) == log(x + sqrt(x - 1)*sqrt(x + 1))
    assert acosh(x).rewrite(asin) == sqrt(x - 1)*(-asin(x) + pi/2)/sqrt(1 - x)
    assert acosh(x).rewrite(asinh) == sqrt(x - 1)*(I*asinh(I*x, evaluate=False) + pi/2)/sqrt(1 - x)
    assert acosh(x).rewrite(atanh) == \
        (sqrt(x - 1)*sqrt(x + 1)*atanh(sqrt(x**2 - 1)/x)/sqrt(x**2 - 1) +
         pi*sqrt(x - 1)*(-x*sqrt(x**(-2)) + 1)/(2*sqrt(1 - x)))
    x = Symbol('x', positive=True)
    assert acosh(x).rewrite(atanh) == \
        sqrt(x - 1)*sqrt(x + 1)*atanh(sqrt(x**2 - 1)/x)/sqrt(x**2 - 1)


def test_acosh_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert acosh(x).as_leading_term(x) == I*pi/2
    assert acosh(x + 1).as_leading_term(x) == sqrt(2)*sqrt(x)
    assert acosh(x - 1).as_leading_term(x) == I*pi
    assert acosh(1/x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert acosh(1/x).as_leading_term(x, cdir=-1) == -log(x) + log(2) + 2*I*pi
    # Tests concerning points lying on branch cuts
    assert acosh(I*x - 2).as_leading_term(x, cdir=1) == acosh(-2)
    assert acosh(-I*x - 2).as_leading_term(x, cdir=1) == -2*I*pi + acosh(-2)
    assert acosh(x**2 - I*x + S(1)/3).as_leading_term(x, cdir=1) == -acosh(S(1)/3)
    assert acosh(x**2 - I*x + S(1)/3).as_leading_term(x, cdir=-1) == acosh(S(1)/3)
    assert acosh(1/(I*x - 3)).as_leading_term(x, cdir=1) == -acosh(-S(1)/3)
    assert acosh(1/(I*x - 3)).as_leading_term(x, cdir=-1) == acosh(-S(1)/3)
    # Tests concerning im(ndir) == 0
    assert acosh(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == log(sqrt(3) + 2) - I*pi
    assert acosh(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == log(sqrt(3) + 2) - I*pi


def test_acosh_series():
    x = Symbol('x')
    assert acosh(x).series(x, 0, 8) == \
        -I*x + pi*I/2 - I*x**3/6 - 3*I*x**5/40 - 5*I*x**7/112 + O(x**8)
    t5 = acosh(x).taylor_term(5, x)
    assert t5 == - 3*I*x**5/40
    assert acosh(x).taylor_term(7, x, t5, 0) == - 5*I*x**7/112


def test_acosh_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert acosh(x + 1)._eval_nseries(x, 4, None) == sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/12 + 3*sqrt(2)*x**(S(5)/2)/160 - 5*sqrt(2)*x**(S(7)/2)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert acosh(x - 1)._eval_nseries(x, 4, None) == I*pi - \
    sqrt(2)*I*sqrt(x) - sqrt(2)*I*x**(S(3)/2)/12 - 3*sqrt(2)*I*x**(S(5)/2)/160 - \
    5*sqrt(2)*I*x**(S(7)/2)/896 + O(x**4)
    assert acosh(I*x - 2)._eval_nseries(x, 4, None, cdir=1) == acosh(-2) - \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acosh(-I*x - 2)._eval_nseries(x, 4, None, cdir=1) == acosh(-2) - \
    2*I*pi + sqrt(3)*I*x/3 + sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert acosh(1/(I*x - 3))._eval_nseries(x, 4, None, cdir=1) == -acosh(-S(1)/3) + \
    sqrt(2)*x/12 + 17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert acosh(1/(I*x - 3))._eval_nseries(x, 4, None, cdir=-1) == acosh(-S(1)/3) - \
    sqrt(2)*x/12 - 17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert acosh(-I*x**2 + x - 2)._eval_nseries(x, 4, None) == -I*pi + log(sqrt(3) + 2) + \
    x*(-2*sqrt(3) - 3)/(3*sqrt(3) + 6) + x**2*(-12 + 36*I + sqrt(3)*(-7 + 21*I))/(36*sqrt(3) + \
    63) + x**3*(-168 + 672*I + sqrt(3)*(-97 + 388*I))/(1008*sqrt(3) + 1746) + O(x**4)


def test_acosh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: acosh(x).fdiff(2))


def test_asech():
    x = Symbol('x')

    assert unchanged(asech, -x)

    # values at fixed points
    assert asech(1) == 0
    assert asech(-1) == pi*I
    assert asech(0) is oo
    assert asech(2) == I*pi/3
    assert asech(-2) == 2*I*pi / 3
    assert asech(nan) is nan

    # at infinites
    assert asech(oo) == I*pi/2
    assert asech(-oo) == I*pi/2
    assert asech(zoo) == I*AccumBounds(-pi/2, pi/2)

    assert asech(I) == log(1 + sqrt(2)) - I*pi/2
    assert asech(-I) == log(1 + sqrt(2)) + I*pi/2
    assert asech(sqrt(2) - sqrt(6)) == 11*I*pi / 12
    assert asech(sqrt(2 - 2/sqrt(5))) == I*pi / 10
    assert asech(-sqrt(2 - 2/sqrt(5))) == 9*I*pi / 10
    assert asech(2 / sqrt(2 + sqrt(2))) == I*pi / 8
    assert asech(-2 / sqrt(2 + sqrt(2))) == 7*I*pi / 8
    assert asech(sqrt(5) - 1) == I*pi / 5
    assert asech(1 - sqrt(5)) == 4*I*pi / 5
    assert asech(-sqrt(2*(2 + sqrt(2)))) == 5*I*pi / 8

    # properties
    # asech(x) == acosh(1/x)
    assert asech(sqrt(2)) == acosh(1/sqrt(2))
    assert asech(2/sqrt(3)) == acosh(sqrt(3)/2)
    assert asech(2/sqrt(2 + sqrt(2))) == acosh(sqrt(2 + sqrt(2))/2)
    assert asech(2) == acosh(S.Half)

    # reality
    assert asech(S(2)).is_real is False
    assert asech(-S(1) / 3).is_real is False
    assert asech(S(2) / 3).is_finite is True
    assert asech(S(0)).is_real is False
    assert asech(S(0)).is_extended_real is True
    assert asech(symbols('y', real=True)).is_real is None

    # asech(x) == I*acos(1/x)
    # (Note: the exact formula is asech(x) == +/- I*acos(1/x))
    assert asech(-sqrt(2)) == I*acos(-1/sqrt(2))
    assert asech(-2/sqrt(3)) == I*acos(-sqrt(3)/2)
    assert asech(-S(2)) == I*acos(Rational(-1, 2))
    assert asech(-2/sqrt(2)) == I*acos(-sqrt(2)/2)

    # sech(asech(x)) / x == 1
    assert expand_mul(sech(asech(sqrt(6) - sqrt(2))) / (sqrt(6) - sqrt(2))) == 1
    assert expand_mul(sech(asech(sqrt(6) + sqrt(2))) / (sqrt(6) + sqrt(2))) == 1
    assert (sech(asech(sqrt(2 + 2/sqrt(5)))) / (sqrt(2 + 2/sqrt(5)))).simplify() == 1
    assert (sech(asech(-sqrt(2 + 2/sqrt(5)))) / (-sqrt(2 + 2/sqrt(5)))).simplify() == 1
    assert (sech(asech(sqrt(2*(2 + sqrt(2))))) / (sqrt(2*(2 + sqrt(2))))).simplify() == 1
    assert expand_mul(sech(asech(1 + sqrt(5))) / (1 + sqrt(5))) == 1
    assert expand_mul(sech(asech(-1 - sqrt(5))) / (-1 - sqrt(5))) == 1
    assert expand_mul(sech(asech(-sqrt(6) - sqrt(2))) / (-sqrt(6) - sqrt(2))) == 1

    # numerical evaluation
    assert str(asech(5*I).n(6)) == '0.19869 - 1.5708*I'
    assert str(asech(-5*I).n(6)) == '0.19869 + 1.5708*I'


def test_asech_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert asech(x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert asech(x).as_leading_term(x, cdir=-1) == -log(x) + log(2) + 2*I*pi
    assert asech(x + 1).as_leading_term(x, cdir=1) == sqrt(2)*I*sqrt(x)
    assert asech(1/x).as_leading_term(x, cdir=1) == I*pi/2
    # Tests concerning points lying on branch cuts
    assert asech(x - 1).as_leading_term(x, cdir=1) == I*pi
    assert asech(I*x + 3).as_leading_term(x, cdir=1) == -asech(3)
    assert asech(-I*x + 3).as_leading_term(x, cdir=1) == asech(3)
    assert asech(I*x - 3).as_leading_term(x, cdir=1) == -asech(-3)
    assert asech(-I*x - 3).as_leading_term(x, cdir=1) == asech(-3)
    assert asech(I*x - S(1)/3).as_leading_term(x, cdir=1) == -2*I*pi + asech(-S(1)/3)
    assert asech(I*x - S(1)/3).as_leading_term(x, cdir=-1) == asech(-S(1)/3)
    # Tests concerning im(ndir) == 0
    assert asech(-I*x**2 + x - 3).as_leading_term(x, cdir=1) == log(-S(1)/3 + 2*sqrt(2)*I/3)
    assert asech(-I*x**2 + x - 3).as_leading_term(x, cdir=-1) == log(-S(1)/3 + 2*sqrt(2)*I/3)


def test_asech_series():
    x = Symbol('x')
    assert asech(x).series(x, 0, 9, cdir=1) == log(2) - log(x) - x**2/4 - 3*x**4/32 \
    - 5*x**6/96 - 35*x**8/1024 + O(x**9)
    assert asech(x).series(x, 0, 9, cdir=-1) == I*pi + log(2) - log(-x) - x**2/4 - \
    3*x**4/32 - 5*x**6/96 - 35*x**8/1024 + O(x**9)
    t6 = asech(x).taylor_term(6, x)
    assert t6 == -5*x**6/96
    assert asech(x).taylor_term(8, x, t6, 0) == -35*x**8/1024


def test_asech_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert asech(x + 1)._eval_nseries(x, 4, None) == sqrt(2)*sqrt(-x) + 5*sqrt(2)*(-x)**(S(3)/2)/12 + \
    43*sqrt(2)*(-x)**(S(5)/2)/160 + 177*sqrt(2)*(-x)**(S(7)/2)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert asech(x - 1)._eval_nseries(x, 4, None) == I*pi + sqrt(2)*sqrt(x) + \
    5*sqrt(2)*x**(S(3)/2)/12 + 43*sqrt(2)*x**(S(5)/2)/160 + 177*sqrt(2)*x**(S(7)/2)/896 + O(x**4)
    assert asech(I*x + 3)._eval_nseries(x, 4, None) == -asech(3) + sqrt(2)*x/12 - \
    17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(-I*x + 3)._eval_nseries(x, 4, None) == asech(3) + sqrt(2)*x/12 + \
    17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(I*x - 3)._eval_nseries(x, 4, None) == -asech(-3) - sqrt(2)*x/12 - \
    17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(-I*x - 3)._eval_nseries(x, 4, None) == asech(-3) - sqrt(2)*x/12 + \
    17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert asech(-I*x**2 + x - 2)._eval_nseries(x, 3, None) == 2*I*pi/3 + \
    x*(-sqrt(3) + 3*I)/(6*sqrt(3) + 6*I) + x**2*(36 + sqrt(3)*(7 - 12*I) + 21*I)/(72*sqrt(3) - \
    72*I) + O(x**3)


def test_asech_rewrite():
    x = Symbol('x')
    assert asech(x).rewrite(log) == log(1/x + sqrt(1/x - 1) * sqrt(1/x + 1))
    assert asech(x).rewrite(acosh) == acosh(1/x)
    assert asech(x).rewrite(asinh) == sqrt(-1 + 1/x)*(I*asinh(I/x, evaluate=False) + pi/2)/sqrt(1 - 1/x)
    assert asech(x).rewrite(atanh) == \
        sqrt(x + 1)*sqrt(1/(x + 1))*atanh(sqrt(1 - x**2)) + I*pi*(-sqrt(x)*sqrt(1/x) + 1 - I*sqrt(x**2)/(2*sqrt(-x**2)) - I*sqrt(-x)/(2*sqrt(x)))


def test_asech_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: asech(x).fdiff(2))


def test_acsch():
    x = Symbol('x')

    assert unchanged(acsch, x)
    assert acsch(-x) == -acsch(x)

    # values at fixed points
    assert acsch(1) == log(1 + sqrt(2))
    assert acsch(-1) == - log(1 + sqrt(2))
    assert acsch(0) is zoo
    assert acsch(2) == log((1+sqrt(5))/2)
    assert acsch(-2) == - log((1+sqrt(5))/2)

    assert acsch(I) == - I*pi/2
    assert acsch(-I) == I*pi/2
    assert acsch(-I*(sqrt(6) + sqrt(2))) == I*pi / 12
    assert acsch(I*(sqrt(2) + sqrt(6))) == -I*pi / 12
    assert acsch(-I*(1 + sqrt(5))) == I*pi / 10
    assert acsch(I*(1 + sqrt(5))) == -I*pi / 10
    assert acsch(-I*2 / sqrt(2 - sqrt(2))) == I*pi / 8
    assert acsch(I*2 / sqrt(2 - sqrt(2))) == -I*pi / 8
    assert acsch(-I*2) == I*pi / 6
    assert acsch(I*2) == -I*pi / 6
    assert acsch(-I*sqrt(2 + 2/sqrt(5))) == I*pi / 5
    assert acsch(I*sqrt(2 + 2/sqrt(5))) == -I*pi / 5
    assert acsch(-I*sqrt(2)) == I*pi / 4
    assert acsch(I*sqrt(2)) == -I*pi / 4
    assert acsch(-I*(sqrt(5)-1)) == 3*I*pi / 10
    assert acsch(I*(sqrt(5)-1)) == -3*I*pi / 10
    assert acsch(-I*2 / sqrt(3)) == I*pi / 3
    assert acsch(I*2 / sqrt(3)) == -I*pi / 3
    assert acsch(-I*2 / sqrt(2 + sqrt(2))) == 3*I*pi / 8
    assert acsch(I*2 / sqrt(2 + sqrt(2))) == -3*I*pi / 8
    assert acsch(-I*sqrt(2 - 2/sqrt(5))) == 2*I*pi / 5
    assert acsch(I*sqrt(2 - 2/sqrt(5))) == -2*I*pi / 5
    assert acsch(-I*(sqrt(6) - sqrt(2))) == 5*I*pi / 12
    assert acsch(I*(sqrt(6) - sqrt(2))) == -5*I*pi / 12
    assert acsch(nan) is nan

    # properties
    # acsch(x) == asinh(1/x)
    assert acsch(-I*sqrt(2)) == asinh(I/sqrt(2))
    assert acsch(-I*2 / sqrt(3)) == asinh(I*sqrt(3) / 2)

    # reality
    assert acsch(S(2)).is_real is True
    assert acsch(S(2)).is_finite is True
    assert acsch(S(-2)).is_real is True
    assert acsch(S(oo)).is_extended_real is True
    assert acsch(-S(oo)).is_real is True
    assert (acsch(2) - oo) == -oo
    assert acsch(symbols('y', extended_real=True)).is_extended_real is True

    # acsch(x) == -I*asin(I/x)
    assert acsch(-I*sqrt(2)) == -I*asin(-1/sqrt(2))
    assert acsch(-I*2 / sqrt(3)) == -I*asin(-sqrt(3)/2)

    # csch(acsch(x)) / x == 1
    assert expand_mul(csch(acsch(-I*(sqrt(6) + sqrt(2)))) / (-I*(sqrt(6) + sqrt(2)))) == 1
    assert expand_mul(csch(acsch(I*(1 + sqrt(5)))) / (I*(1 + sqrt(5)))) == 1
    assert (csch(acsch(I*sqrt(2 - 2/sqrt(5)))) / (I*sqrt(2 - 2/sqrt(5)))).simplify() == 1
    assert (csch(acsch(-I*sqrt(2 - 2/sqrt(5)))) / (-I*sqrt(2 - 2/sqrt(5)))).simplify() == 1

    # numerical evaluation
    assert str(acsch(5*I+1).n(6)) == '0.0391819 - 0.193363*I'
    assert str(acsch(-5*I+1).n(6)) == '0.0391819 + 0.193363*I'


def test_acsch_infinities():
    assert acsch(oo) == 0
    assert acsch(-oo) == 0
    assert acsch(zoo) == 0


def test_acsch_leading_term():
    x = Symbol('x')
    assert acsch(1/x).as_leading_term(x) == x
    # Tests concerning branch points
    assert acsch(x + I).as_leading_term(x) == -I*pi/2
    assert acsch(x - I).as_leading_term(x) == I*pi/2
    # Tests concerning points lying on branch cuts
    assert acsch(x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert acsch(x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I*pi
    assert acsch(x + I/2).as_leading_term(x, cdir=1) == -I*pi - acsch(I/2)
    assert acsch(x + I/2).as_leading_term(x, cdir=-1) == acsch(I/2)
    assert acsch(x - I/2).as_leading_term(x, cdir=1) == -acsch(I/2)
    assert acsch(x - I/2).as_leading_term(x, cdir=-1) == acsch(I/2) + I*pi
    # Tests concerning re(ndir) == 0
    assert acsch(I/2 + I*x - x**2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) - I*pi/2
    assert acsch(I/2 + I*x - x**2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) - I*pi/2


def test_acsch_series():
    x = Symbol('x')
    assert acsch(x).series(x, 0, 9) == log(2) - log(x) + x**2/4 - 3*x**4/32 \
    + 5*x**6/96 - 35*x**8/1024 + O(x**9)
    t4 = acsch(x).taylor_term(4, x)
    assert t4 == -3*x**4/32
    assert acsch(x).taylor_term(6, x, t4, 0) == 5*x**6/96


def test_acsch_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert acsch(x + I)._eval_nseries(x, 4, None) == -I*pi/2 + \
    sqrt(2)*I*sqrt(x)*sqrt(-I) - 5*x**(S(3)/2)*(1 - I)/12 - \
    43*sqrt(2)*I*x**(S(5)/2)*sqrt(-I)/160 + 177*x**(S(7)/2)*(1 - I)/896 + O(x**4)
    assert acsch(x - I)._eval_nseries(x, 4, None) == I*pi/2 - \
    sqrt(2)*sqrt(I)*I*sqrt(x) - 5*x**(S(3)/2)*(1 + I)/12 + \
    43*sqrt(2)*sqrt(I)*I*x**(S(5)/2)/160 + 177*x**(S(7)/2)*(1 + I)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert acsch(x + I/2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I/2) - \
    I*pi + 4*sqrt(3)*I*x/3 - 8*sqrt(3)*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsch(x + I/2)._eval_nseries(x, 4, None, cdir=-1) == acsch(I/2) - \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsch(x - I/2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I/2) - \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsch(x - I/2)._eval_nseries(x, 4, None, cdir=-1) == I*pi + \
    acsch(I/2) + 4*sqrt(3)*I*x/3 + 8*sqrt(3)*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # Tests concerning re(ndir) == 0
    assert acsch(I/2 + I*x - x**2)._eval_nseries(x, 4, None) == -I*pi/2 + \
    log(2 - sqrt(3)) + x*(12 - 8*sqrt(3))/(-6 + 3*sqrt(3)) + x**2*(-96 + \
    sqrt(3)*(56 - 84*I) + 144*I)/(-63 + 36*sqrt(3)) + x**3*(2688 - 2688*I + \
    sqrt(3)*(-1552 + 1552*I))/(-873 + 504*sqrt(3)) + O(x**4)


def test_acsch_rewrite():
    x = Symbol('x')
    assert acsch(x).rewrite(log) == log(1/x + sqrt(1/x**2 + 1))
    assert acsch(x).rewrite(asinh) == asinh(1/x)
    assert acsch(x).rewrite(atanh) == (sqrt(-x**2)*(-sqrt(-(x**2 + 1)**2)
                                                    *atanh(sqrt(x**2 + 1))/(x**2 + 1)
                                                    + pi/2)/x)


def test_acsch_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: acsch(x).fdiff(2))


def test_atanh():
    x = Symbol('x')

    # at specific points
    assert atanh(0) == 0
    assert atanh(I) == I*pi/4
    assert atanh(-I) == -I*pi/4
    assert atanh(1) is oo
    assert atanh(-1) is -oo
    assert atanh(nan) is nan

    # at infinites
    assert atanh(oo) == -I*pi/2
    assert atanh(-oo) == I*pi/2

    assert atanh(I*oo) == I*pi/2
    assert atanh(-I*oo) == -I*pi/2

    assert atanh(zoo) == I*AccumBounds(-pi/2, pi/2)

    # properties
    assert atanh(-x) == -atanh(x)

    # reality
    assert atanh(S(2)).is_real is False
    assert atanh(S(-1)/5).is_real is True
    assert atanh(symbols('y', extended_real=True)).is_real is None
    assert atanh(S(1)).is_real is False
    assert atanh(S(1)).is_extended_real is True
    assert atanh(S(-1)).is_real is False

    # special values
    assert atanh(I/sqrt(3)) == I*pi/6
    assert atanh(-I/sqrt(3)) == -I*pi/6
    assert atanh(I*sqrt(3)) == I*pi/3
    assert atanh(-I*sqrt(3)) == -I*pi/3
    assert atanh(I*(1 + sqrt(2))) == pi*I*Rational(3, 8)
    assert atanh(I*(sqrt(2) - 1)) == pi*I/8
    assert atanh(I*(1 - sqrt(2))) == -pi*I/8
    assert atanh(-I*(1 + sqrt(2))) == pi*I*Rational(-3, 8)
    assert atanh(I*sqrt(5 + 2*sqrt(5))) == I*pi*Rational(2, 5)
    assert atanh(-I*sqrt(5 + 2*sqrt(5))) == I*pi*Rational(-2, 5)
    assert atanh(I*(2 - sqrt(3))) == pi*I/12
    assert atanh(I*(sqrt(3) - 2)) == -pi*I/12
    assert atanh(oo) == -I*pi/2

    # Symmetry
    assert atanh(Rational(-1, 2)) == -atanh(S.Half)

    # inverse composition
    assert unchanged(atanh, tanh(Symbol('v1')))

    assert atanh(tanh(-5, evaluate=False)) == -5
    assert atanh(tanh(0, evaluate=False)) == 0
    assert atanh(tanh(7, evaluate=False)) == 7
    assert atanh(tanh(I, evaluate=False)) == I
    assert atanh(tanh(-I, evaluate=False)) == -I
    assert atanh(tanh(-11*I, evaluate=False)) == -11*I + 4*I*pi
    assert atanh(tanh(3 + I)) == 3 + I
    assert atanh(tanh(4 + 5*I)) == 4 - 2*I*pi + 5*I
    assert atanh(tanh(pi/2)) == pi/2
    assert atanh(tanh(pi)) == pi
    assert atanh(tanh(-3 + 7*I)) == -3 - 2*I*pi + 7*I
    assert atanh(tanh(9 - I*2/3)) == 9 - I*2/3
    assert atanh(tanh(-32 - 123*I)) == -32 - 123*I + 39*I*pi


def test_atanh_rewrite():
    x = Symbol('x')
    assert atanh(x).rewrite(log) == (log(1 + x) - log(1 - x)) / 2
    assert atanh(x).rewrite(asinh) == \
        pi*x/(2*sqrt(-x**2)) - sqrt(-x)*sqrt(1 - x**2)*sqrt(1/(x**2 - 1))*asinh(sqrt(1/(x**2 - 1)))/sqrt(x)


def test_atanh_leading_term():
    x = Symbol('x')
    assert atanh(x).as_leading_term(x) == x
    # Tests concerning branch points
    assert atanh(x + 1).as_leading_term(x, cdir=1) == -log(x)/2 + log(2)/2 - I*pi/2
    assert atanh(x + 1).as_leading_term(x, cdir=-1) == -log(x)/2 + log(2)/2 + I*pi/2
    assert atanh(x - 1).as_leading_term(x, cdir=1) == log(x)/2 - log(2)/2
    assert atanh(x - 1).as_leading_term(x, cdir=-1) == log(x)/2 - log(2)/2
    assert atanh(1/x).as_leading_term(x, cdir=1) == -I*pi/2
    assert atanh(1/x).as_leading_term(x, cdir=-1) == I*pi/2
    # Tests concerning points lying on branch cuts
    assert atanh(I*x + 2).as_leading_term(x, cdir=1) == atanh(2) + I*pi
    assert atanh(-I*x + 2).as_leading_term(x, cdir=1) == atanh(2)
    assert atanh(I*x - 2).as_leading_term(x, cdir=1) == -atanh(2)
    assert atanh(-I*x - 2).as_leading_term(x, cdir=1) == -I*pi - atanh(2)
    # Tests concerning im(ndir) == 0
    assert atanh(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == -log(3)/2 - I*pi/2
    assert atanh(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == -log(3)/2 - I*pi/2


def test_atanh_series():
    x = Symbol('x')
    assert atanh(x).series(x, 0, 10) == \
        x + x**3/3 + x**5/5 + x**7/7 + x**9/9 + O(x**10)


def test_atanh_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert atanh(x + 1)._eval_nseries(x, 4, None, cdir=1) == -I*pi/2 + log(2)/2 - \
    log(x)/2 + x/4 - x**2/16 + x**3/48 + O(x**4)
    assert atanh(x + 1)._eval_nseries(x, 4, None, cdir=-1) == I*pi/2 + log(2)/2 - \
    log(x)/2 + x/4 - x**2/16 + x**3/48 + O(x**4)
    assert atanh(x - 1)._eval_nseries(x, 4, None, cdir=1) == -log(2)/2 + log(x)/2 + \
    x/4 + x**2/16 + x**3/48 + O(x**4)
    assert atanh(x - 1)._eval_nseries(x, 4, None, cdir=-1) == -log(2)/2 + log(x)/2 + \
    x/4 + x**2/16 + x**3/48 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert atanh(I*x + 2)._eval_nseries(x, 4, None, cdir=1) == I*pi + atanh(2) - \
    I*x/3 - 2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x + 2)._eval_nseries(x, 4, None, cdir=-1) == atanh(2) - I*x/3 - \
    2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x - 2)._eval_nseries(x, 4, None, cdir=1) == -atanh(2) - I*x/3 + \
    2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x - 2)._eval_nseries(x, 4, None, cdir=-1) == -atanh(2) - I*pi - \
    I*x/3 + 2*x**2/9 + 13*I*x**3/81 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert atanh(-I*x**2 + x - 2)._eval_nseries(x, 4, None) == -I*pi/2 - log(3)/2 - x/3 + \
    x**2*(-S(1)/4 + I/2) + x**2*(S(1)/36 - I/6) + x**3*(-S(1)/6 + I/2) + x**3*(S(1)/162 - I/18) + O(x**4)


def test_atanh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: atanh(x).fdiff(2))


def test_acoth():
    x = Symbol('x')

    #at specific points
    assert acoth(0) == I*pi/2
    assert acoth(I) == -I*pi/4
    assert acoth(-I) == I*pi/4
    assert acoth(1) is oo
    assert acoth(-1) is -oo
    assert acoth(nan) is nan

    # at infinites
    assert acoth(oo) == 0
    assert acoth(-oo) == 0
    assert acoth(I*oo) == 0
    assert acoth(-I*oo) == 0
    assert acoth(zoo) == 0

    #properties
    assert acoth(-x) == -acoth(x)

    assert acoth(I/sqrt(3)) == -I*pi/3
    assert acoth(-I/sqrt(3)) == I*pi/3
    assert acoth(I*sqrt(3)) == -I*pi/6
    assert acoth(-I*sqrt(3)) == I*pi/6
    assert acoth(I*(1 + sqrt(2))) == -pi*I/8
    assert acoth(-I*(sqrt(2) + 1)) == pi*I/8
    assert acoth(I*(1 - sqrt(2))) == pi*I*Rational(3, 8)
    assert acoth(I*(sqrt(2) - 1)) == pi*I*Rational(-3, 8)
    assert acoth(I*sqrt(5 + 2*sqrt(5))) == -I*pi/10
    assert acoth(-I*sqrt(5 + 2*sqrt(5))) == I*pi/10
    assert acoth(I*(2 + sqrt(3))) == -pi*I/12
    assert acoth(-I*(2 + sqrt(3))) == pi*I/12
    assert acoth(I*(2 - sqrt(3))) == pi*I*Rational(-5, 12)
    assert acoth(I*(sqrt(3) - 2)) == pi*I*Rational(5, 12)

    # reality
    assert acoth(S(2)).is_real is True
    assert acoth(S(2)).is_finite is True
    assert acoth(S(2)).is_extended_real is True
    assert acoth(S(-2)).is_real is True
    assert acoth(S(1)).is_real is False
    assert acoth(S(1)).is_extended_real is True
    assert acoth(S(-1)).is_real is False
    assert acoth(symbols('y', real=True)).is_real is None

    # Symmetry
    assert acoth(Rational(-1, 2)) == -acoth(S.Half)


def test_acoth_rewrite():
    x = Symbol('x')
    assert acoth(x).rewrite(log) == (log(1 + 1/x) - log(1 - 1/x)) / 2
    assert acoth(x).rewrite(atanh) == atanh(1/x)
    assert acoth(x).rewrite(asinh) == \
        x*sqrt(x**(-2))*asinh(sqrt(1/(x**2 - 1))) + I*pi*(sqrt((x - 1)/x)*sqrt(x/(x - 1)) - sqrt(x/(x + 1))*sqrt(1 + 1/x))/2


def test_acoth_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert acoth(x + 1).as_leading_term(x, cdir=1) == -log(x)/2 + log(2)/2
    assert acoth(x + 1).as_leading_term(x, cdir=-1) == -log(x)/2 + log(2)/2
    assert acoth(x - 1).as_leading_term(x, cdir=1) == log(x)/2 - log(2)/2 + I*pi/2
    assert acoth(x - 1).as_leading_term(x, cdir=-1) == log(x)/2 - log(2)/2 - I*pi/2
    # Tests concerning points lying on branch cuts
    assert acoth(x).as_leading_term(x, cdir=-1) == I*pi/2
    assert acoth(x).as_leading_term(x, cdir=1) == -I*pi/2
    assert acoth(I*x + 1/2).as_leading_term(x, cdir=1) == acoth(1/2)
    assert acoth(-I*x + 1/2).as_leading_term(x, cdir=1) == acoth(1/2) + I*pi
    assert acoth(I*x - 1/2).as_leading_term(x, cdir=1) == -I*pi - acoth(1/2)
    assert acoth(-I*x - 1/2).as_leading_term(x, cdir=1) == -acoth(1/2)
    # Tests concerning im(ndir) == 0
    assert acoth(-I*x**2 - x - S(1)/2).as_leading_term(x, cdir=1) == -log(3)/2 + I*pi/2
    assert acoth(-I*x**2 - x - S(1)/2).as_leading_term(x, cdir=-1) == -log(3)/2 + I*pi/2


def test_acoth_series():
    x = Symbol('x')
    assert acoth(x).series(x, 0, 10) == \
        -I*pi/2 + x + x**3/3 + x**5/5 + x**7/7 + x**9/9 + O(x**10)


def test_acoth_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert acoth(x + 1)._eval_nseries(x, 4, None) == log(2)/2 - log(x)/2 + x/4 - \
    x**2/16 + x**3/48 + O(x**4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=1) == I*pi/2 - log(2)/2 + \
    log(x)/2 + x/4 + x**2/16 + x**3/48 + O(x**4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=-1) == -I*pi/2 - log(2)/2 + \
    log(x)/2 + x/4 + x**2/16 + x**3/48 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert acoth(I*x + S(1)/2)._eval_nseries(x, 4, None, cdir=1) == acoth(S(1)/2) + \
    4*I*x/3 - 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x + S(1)/2)._eval_nseries(x, 4, None, cdir=-1) == I*pi + \
    acoth(S(1)/2) + 4*I*x/3 - 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x - S(1)/2)._eval_nseries(x, 4, None, cdir=1) == -acoth(S(1)/2) - \
    I*pi + 4*I*x/3 + 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x - S(1)/2)._eval_nseries(x, 4, None, cdir=-1) == -acoth(S(1)/2) + \
    4*I*x/3 + 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert acoth(-I*x**2 - x - S(1)/2)._eval_nseries(x, 4, None) == I*pi/2 - log(3)/2 - \
    4*x/3 + x**2*(-S(8)/9 + 2*I/3) - 2*I*x**2 + x**3*(S(104)/81 - 16*I/9) - 8*x**3/3 + O(x**4)


def test_acoth_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: acoth(x).fdiff(2))


def test_inverses():
    x = Symbol('x')
    assert sinh(x).inverse() == asinh
    raises(AttributeError, lambda: cosh(x).inverse())
    assert tanh(x).inverse() == atanh
    assert coth(x).inverse() == acoth
    assert asinh(x).inverse() == sinh
    assert acosh(x).inverse() == cosh
    assert atanh(x).inverse() == tanh
    assert acoth(x).inverse() == coth
    assert asech(x).inverse() == sech
    assert acsch(x).inverse() == csch


def test_leading_term():
    x = Symbol('x')
    assert cosh(x).as_leading_term(x) == 1
    assert coth(x).as_leading_term(x) == 1/x
    for func in [sinh, tanh]:
        assert func(x).as_leading_term(x) == x
    for func in [sinh, cosh, tanh, coth]:
        for ar in (1/x, S.Half):
            eq = func(ar)
            assert eq.as_leading_term(x) == eq
    for func in [csch, sech]:
        eq = func(S.Half)
        assert eq.as_leading_term(x) == eq


def test_complex():
    a, b = symbols('a,b', real=True)
    z = a + b*I
    for func in [sinh, cosh, tanh, coth, sech, csch]:
        assert func(z).conjugate() == func(a - b*I)
    for deep in [True, False]:
        assert sinh(z).expand(
            complex=True, deep=deep) == sinh(a)*cos(b) + I*cosh(a)*sin(b)
        assert cosh(z).expand(
            complex=True, deep=deep) == cosh(a)*cos(b) + I*sinh(a)*sin(b)
        assert tanh(z).expand(complex=True, deep=deep) == sinh(a)*cosh(
            a)/(cos(b)**2 + sinh(a)**2) + I*sin(b)*cos(b)/(cos(b)**2 + sinh(a)**2)
        assert coth(z).expand(complex=True, deep=deep) == sinh(a)*cosh(
            a)/(sin(b)**2 + sinh(a)**2) - I*sin(b)*cos(b)/(sin(b)**2 + sinh(a)**2)
        assert csch(z).expand(complex=True, deep=deep) == cos(b) * sinh(a) / (sin(b)**2\
            *cosh(a)**2 + cos(b)**2 * sinh(a)**2) - I*sin(b) * cosh(a) / (sin(b)**2\
            *cosh(a)**2 + cos(b)**2 * sinh(a)**2)
        assert sech(z).expand(complex=True, deep=deep) == cos(b) * cosh(a) / (sin(b)**2\
            *sinh(a)**2 + cos(b)**2 * cosh(a)**2) - I*sin(b) * sinh(a) / (sin(b)**2\
            *sinh(a)**2 + cos(b)**2 * cosh(a)**2)


def test_complex_2899():
    a, b = symbols('a,b', real=True)
    for deep in [True, False]:
        for func in [sinh, cosh, tanh, coth]:
            assert func(a).expand(complex=True, deep=deep) == func(a)


def test_simplifications():
    x = Symbol('x')
    assert sinh(asinh(x)) == x
    assert sinh(acosh(x)) == sqrt(x - 1) * sqrt(x + 1)
    assert sinh(atanh(x)) == x/sqrt(1 - x**2)
    assert sinh(acoth(x)) == 1/(sqrt(x - 1) * sqrt(x + 1))

    assert cosh(asinh(x)) == sqrt(1 + x**2)
    assert cosh(acosh(x)) == x
    assert cosh(atanh(x)) == 1/sqrt(1 - x**2)
    assert cosh(acoth(x)) == x/(sqrt(x - 1) * sqrt(x + 1))

    assert tanh(asinh(x)) == x/sqrt(1 + x**2)
    assert tanh(acosh(x)) == sqrt(x - 1) * sqrt(x + 1) / x
    assert tanh(atanh(x)) == x
    assert tanh(acoth(x)) == 1/x

    assert coth(asinh(x)) == sqrt(1 + x**2)/x
    assert coth(acosh(x)) == x/(sqrt(x - 1) * sqrt(x + 1))
    assert coth(atanh(x)) == 1/x
    assert coth(acoth(x)) == x

    assert csch(asinh(x)) == 1/x
    assert csch(acosh(x)) == 1/(sqrt(x - 1) * sqrt(x + 1))
    assert csch(atanh(x)) == sqrt(1 - x**2)/x
    assert csch(acoth(x)) == sqrt(x - 1) * sqrt(x + 1)

    assert sech(asinh(x)) == 1/sqrt(1 + x**2)
    assert sech(acosh(x)) == 1/x
    assert sech(atanh(x)) == sqrt(1 - x**2)
    assert sech(acoth(x)) == sqrt(x - 1) * sqrt(x + 1)/x


def test_issue_4136():
    assert cosh(asinh(Integer(3)/2)) == sqrt(Integer(13)/4)


def test_sinh_rewrite():
    x = Symbol('x')
    assert sinh(x).rewrite(exp) == (exp(x) - exp(-x))/2 \
        == sinh(x).rewrite('tractable')
    assert sinh(x).rewrite(cosh) == -I*cosh(x + I*pi/2)
    tanh_half = tanh(S.Half*x)
    assert sinh(x).rewrite(tanh) == 2*tanh_half/(1 - tanh_half**2)
    coth_half = coth(S.Half*x)
    assert sinh(x).rewrite(coth) == 2*coth_half/(coth_half**2 - 1)


def test_cosh_rewrite():
    x = Symbol('x')
    assert cosh(x).rewrite(exp) == (exp(x) + exp(-x))/2 \
        == cosh(x).rewrite('tractable')
    assert cosh(x).rewrite(sinh) == -I*sinh(x + I*pi/2, evaluate=False)
    tanh_half = tanh(S.Half*x)**2
    assert cosh(x).rewrite(tanh) == (1 + tanh_half)/(1 - tanh_half)
    coth_half = coth(S.Half*x)**2
    assert cosh(x).rewrite(coth) == (coth_half + 1)/(coth_half - 1)


def test_tanh_rewrite():
    x = Symbol('x')
    assert tanh(x).rewrite(exp) == (exp(x) - exp(-x))/(exp(x) + exp(-x)) \
        == tanh(x).rewrite('tractable')
    assert tanh(x).rewrite(sinh) == I*sinh(x)/sinh(I*pi/2 - x, evaluate=False)
    assert tanh(x).rewrite(cosh) == I*cosh(I*pi/2 - x, evaluate=False)/cosh(x)
    assert tanh(x).rewrite(coth) == 1/coth(x)


def test_coth_rewrite():
    x = Symbol('x')
    assert coth(x).rewrite(exp) == (exp(x) + exp(-x))/(exp(x) - exp(-x)) \
        == coth(x).rewrite('tractable')
    assert coth(x).rewrite(sinh) == -I*sinh(I*pi/2 - x, evaluate=False)/sinh(x)
    assert coth(x).rewrite(cosh) == -I*cosh(x)/cosh(I*pi/2 - x, evaluate=False)
    assert coth(x).rewrite(tanh) == 1/tanh(x)


def test_csch_rewrite():
    x = Symbol('x')
    assert csch(x).rewrite(exp) == 1 / (exp(x)/2 - exp(-x)/2) \
        == csch(x).rewrite('tractable')
    assert csch(x).rewrite(cosh) == I/cosh(x + I*pi/2, evaluate=False)
    tanh_half = tanh(S.Half*x)
    assert csch(x).rewrite(tanh) == (1 - tanh_half**2)/(2*tanh_half)
    coth_half = coth(S.Half*x)
    assert csch(x).rewrite(coth) == (coth_half**2 - 1)/(2*coth_half)


def test_sech_rewrite():
    x = Symbol('x')
    assert sech(x).rewrite(exp) == 1 / (exp(x)/2 + exp(-x)/2) \
        == sech(x).rewrite('tractable')
    assert sech(x).rewrite(sinh) == I/sinh(x + I*pi/2, evaluate=False)
    tanh_half = tanh(S.Half*x)**2
    assert sech(x).rewrite(tanh) == (1 - tanh_half)/(1 + tanh_half)
    coth_half = coth(S.Half*x)**2
    assert sech(x).rewrite(coth) == (coth_half - 1)/(coth_half + 1)


def test_derivs():
    x = Symbol('x')
    assert coth(x).diff(x) == -sinh(x)**(-2)
    assert sinh(x).diff(x) == cosh(x)
    assert cosh(x).diff(x) == sinh(x)
    assert tanh(x).diff(x) == -tanh(x)**2 + 1
    assert csch(x).diff(x) == -coth(x)*csch(x)
    assert sech(x).diff(x) == -tanh(x)*sech(x)
    assert acoth(x).diff(x) == 1/(-x**2 + 1)
    assert asinh(x).diff(x) == 1/sqrt(x**2 + 1)
    assert acosh(x).diff(x) == 1/(sqrt(x - 1)*sqrt(x + 1))
    assert acosh(x).diff(x) == acosh(x).rewrite(log).diff(x).together()
    assert atanh(x).diff(x) == 1/(-x**2 + 1)
    assert asech(x).diff(x) == -1/(x*sqrt(1 - x**2))
    assert acsch(x).diff(x) == -1/(x**2*sqrt(1 + x**(-2)))


def test_sinh_expansion():
    x, y = symbols('x,y')
    assert sinh(x+y).expand(trig=True) == sinh(x)*cosh(y) + cosh(x)*sinh(y)
    assert sinh(2*x).expand(trig=True) == 2*sinh(x)*cosh(x)
    assert sinh(3*x).expand(trig=True).expand() == \
        sinh(x)**3 + 3*sinh(x)*cosh(x)**2


def test_cosh_expansion():
    x, y = symbols('x,y')
    assert cosh(x+y).expand(trig=True) == cosh(x)*cosh(y) + sinh(x)*sinh(y)
    assert cosh(2*x).expand(trig=True) == cosh(x)**2 + sinh(x)**2
    assert cosh(3*x).expand(trig=True).expand() == \
        3*sinh(x)**2*cosh(x) + cosh(x)**3

def test_cosh_positive():
    # See issue 11721
    # cosh(x) is positive for real values of x
    k = symbols('k', real=True)
    n = symbols('n', integer=True)

    assert cosh(k, evaluate=False).is_positive is True
    assert cosh(k + 2*n*pi*I, evaluate=False).is_positive is True
    assert cosh(I*pi/4, evaluate=False).is_positive is True
    assert cosh(3*I*pi/4, evaluate=False).is_positive is False

def test_cosh_nonnegative():
    k = symbols('k', real=True)
    n = symbols('n', integer=True)

    assert cosh(k, evaluate=False).is_nonnegative is True
    assert cosh(k + 2*n*pi*I, evaluate=False).is_nonnegative is True
    assert cosh(I*pi/4, evaluate=False).is_nonnegative is True
    assert cosh(3*I*pi/4, evaluate=False).is_nonnegative is False
    assert cosh(S.Zero, evaluate=False).is_nonnegative is True

def test_real_assumptions():
    z = Symbol('z', real=False)
    assert sinh(z).is_real is None
    assert cosh(z).is_real is None
    assert tanh(z).is_real is None
    assert sech(z).is_real is None
    assert csch(z).is_real is None
    assert coth(z).is_real is None

def test_sign_assumptions():
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    assert sinh(n).is_negative is True
    assert sinh(p).is_positive is True
    assert cosh(n).is_positive is True
    assert cosh(p).is_positive is True
    assert tanh(n).is_negative is True
    assert tanh(p).is_positive is True
    assert csch(n).is_negative is True
    assert csch(p).is_positive is True
    assert sech(n).is_positive is True
    assert sech(p).is_positive is True
    assert coth(n).is_negative is True
    assert coth(p).is_positive is True


def test_issue_25847():
    x = Symbol('x')

    #atanh
    assert atanh(sin(x)/x).as_leading_term(x) == atanh(sin(x)/x)
    raises(PoleError, lambda: atanh(exp(1/x)).as_leading_term(x))

    #asinh
    assert asinh(sin(x)/x).as_leading_term(x) == log(1 + sqrt(2))
    raises(PoleError, lambda: asinh(exp(1/x)).as_leading_term(x))

    #acosh
    assert acosh(sin(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: acosh(exp(1/x)).as_leading_term(x))

    #acoth
    assert acoth(sin(x)/x).as_leading_term(x) == acoth(sin(x)/x)
    raises(PoleError, lambda: acoth(exp(1/x)).as_leading_term(x))

    #asech
    assert asech(sinh(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: asech(exp(1/x)).as_leading_term(x))

    #acsch
    assert acsch(sin(x)/x).as_leading_term(x) == log(1 + sqrt(2))
    raises(PoleError, lambda: acsch(exp(1/x)).as_leading_term(x))


def test_issue_25175():
    x = Symbol('x')
    g1 = 2*acosh(1 + 2*x/3) - acosh(S(5)/3 - S(8)/3/(x + 4))
    g2 = 2*log(sqrt((x + 4)/3)*(sqrt(x + 3)+sqrt(x))**2/(2*sqrt(x + 3) + sqrt(x)))
    assert (g1 - g2).series(x) == O(x**6)