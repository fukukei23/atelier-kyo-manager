
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\attention.py ===
import onnx
from onnx import onnx_pb as onnx_proto  # noqa: F401

from ..quant_utils import attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase

"""
    Quantize Attention
"""


class AttentionQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def should_quantize(self):
        return self.quantizer.should_quantize_node(self.node)

    def quantize(self):
        """
        parameter node: Attention node.
        parameter new_nodes_list: List of new nodes created before processing this node.
        return: a list of nodes in topological order that represents quantized Attention node.
        """
        node = self.node
        assert node.op_type == "Attention"

        # TODO This is a temporary fix to stop exporting QAttention with qkv_hidden_sizes
        # attribute. This needs to be removed once the QAttention for varied q,k,v sizes
        # is implemented
        for attr in node.attribute:
            if attr.name == "qkv_hidden_sizes":
                return super().quantize()

        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0])

        (
            quantized_input_names_weight,
            zero_point_names_weight,
            scale_names_weight,
            nodes_weight,
        ) = self.quantizer.quantize_weight(node, [1], reduce_range=True, op_level_per_channel=True)
        quantized_input_names.extend(quantized_input_names_weight)
        zero_point_names.extend(zero_point_names_weight)
        scale_names.extend(scale_names_weight)
        nodes.extend(nodes_weight)

        if quantized_input_names is None:
            return super().quantize()

        qattention_name = "" if not node.name else node.name + "_quant"

        inputs = []
        inputs.extend(quantized_input_names)
        inputs.extend([node.input[2]])
        inputs.extend(scale_names)
        inputs.extend([node.input[3] if len(node.input) > 3 else ""])
        inputs.extend(zero_point_names)
        inputs.extend([node.input[4] if len(node.input) > 4 else ""])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qattention_node = onnx.helper.make_node("QAttention", inputs, node.output, qattention_name, **kwargs)
        nodes.append(qattention_node)

        self.quantizer.new_nodes += nodes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\make_dynamic_shape_fixed.py ===
#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import onnx

from .onnx_model_utils import fix_output_shapes, make_dim_param_fixed, make_input_shape_fixed


def make_dynamic_shape_fixed_helper():
    parser = argparse.ArgumentParser(
        f"{os.path.basename(__file__)}:{make_dynamic_shape_fixed_helper.__name__}",
        description="""
                                     Assign a fixed value to a dim_param or input shape
                                     Provide either dim_param and dim_value or input_name and input_shape.""",
    )

    parser.add_argument(
        "--dim_param", type=str, required=False, help="Symbolic parameter name. Provide dim_value if specified."
    )
    parser.add_argument(
        "--dim_value", type=int, required=False, help="Value to replace dim_param with in the model. Must be > 0."
    )
    parser.add_argument(
        "--input_name",
        type=str,
        required=False,
        help="Model input name to replace shape of. Provide input_shape if specified.",
    )
    parser.add_argument(
        "--input_shape",
        type=lambda x: [int(i) for i in x.split(",")],
        required=False,
        help="Shape to use for input_shape. Provide comma separated list for the shape. "
        "All values must be > 0. e.g. --input_shape 1,3,256,256",
    )

    parser.add_argument("input_model", type=pathlib.Path, help="Provide path to ONNX model to update.")
    parser.add_argument("output_model", type=pathlib.Path, help="Provide path to write updated ONNX model to.")

    args = parser.parse_args()

    if (
        (args.dim_param and args.input_name)
        or (not args.dim_param and not args.input_name)
        or (args.dim_param and (not args.dim_value or args.dim_value < 1))
        or (args.input_name and (not args.input_shape or any(value < 1 for value in args.input_shape)))
    ):
        print("Invalid usage.")
        parser.print_help()
        sys.exit(-1)

    model = onnx.load(str(args.input_model.resolve(strict=True)))

    if args.dim_param:
        make_dim_param_fixed(model.graph, args.dim_param, args.dim_value)
    else:
        make_input_shape_fixed(model.graph, args.input_name, args.input_shape)

    # update the output shapes to make them fixed if possible.
    fix_output_shapes(model)

    onnx.save(model, str(args.output_model.resolve()))


if __name__ == "__main__":
    make_dynamic_shape_fixed_helper()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\build\metadata_legacy.py ===
"""Metadata generation logic for legacy source distributions."""

import logging
import os

from pip._internal.build_env import BuildEnvironment
from pip._internal.cli.spinners import open_spinner
from pip._internal.exceptions import (
    InstallationError,
    InstallationSubprocessError,
    MetadataGenerationFailed,
)
from pip._internal.utils.setuptools_build import make_setuptools_egg_info_args
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory

logger = logging.getLogger(__name__)


def _find_egg_info(directory: str) -> str:
    """Find an .egg-info subdirectory in `directory`."""
    filenames = [f for f in os.listdir(directory) if f.endswith(".egg-info")]

    if not filenames:
        raise InstallationError(f"No .egg-info directory found in {directory}")

    if len(filenames) > 1:
        raise InstallationError(
            f"More than one .egg-info directory found in {directory}"
        )

    return os.path.join(directory, filenames[0])


def generate_metadata(
    build_env: BuildEnvironment,
    setup_py_path: str,
    source_dir: str,
    isolated: bool,
    details: str,
) -> str:
    """Generate metadata using setup.py-based defacto mechanisms.

    Returns the generated metadata directory.
    """
    logger.debug(
        "Running setup.py (path:%s) egg_info for package %s",
        setup_py_path,
        details,
    )

    egg_info_dir = TempDirectory(kind="pip-egg-info", globally_managed=True).path

    args = make_setuptools_egg_info_args(
        setup_py_path,
        egg_info_dir=egg_info_dir,
        no_user_config=isolated,
    )

    with build_env:
        with open_spinner("Preparing metadata (setup.py)") as spinner:
            try:
                call_subprocess(
                    args,
                    cwd=source_dir,
                    command_desc="python setup.py egg_info",
                    spinner=spinner,
                )
            except InstallationSubprocessError as error:
                raise MetadataGenerationFailed(package_details=details) from error

    # Return the .egg-info directory.
    return _find_egg_info(egg_info_dir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\cartan_type.py ===
from sympy.core import Atom, Basic


class CartanType_generator():
    """
    Constructor for actually creating things
    """
    def __call__(self, *args):
        c = args[0]
        if isinstance(c, list):
            letter, n = c[0], int(c[1])
        elif isinstance(c, str):
            letter, n = c[0], int(c[1:])
        else:
            raise TypeError("Argument must be a string (e.g. 'A3') or a list (e.g. ['A', 3])")

        if n < 0:
            raise ValueError("Lie algebra rank cannot be negative")
        if letter == "A":
            from . import type_a
            return type_a.TypeA(n)
        if letter == "B":
            from . import type_b
            return type_b.TypeB(n)

        if letter == "C":
            from . import type_c
            return type_c.TypeC(n)

        if letter == "D":
            from . import type_d
            return type_d.TypeD(n)

        if letter == "E":
            if n >= 6 and n <= 8:
                from . import type_e
                return type_e.TypeE(n)

        if letter == "F":
            if n == 4:
                from . import type_f
                return type_f.TypeF(n)

        if letter == "G":
            if n == 2:
                from . import type_g
                return type_g.TypeG(n)

CartanType = CartanType_generator()


class Standard_Cartan(Atom):
    """
    Concrete base class for Cartan types such as A4, etc
    """

    def __new__(cls, series, n):
        obj = Basic.__new__(cls)
        obj.n = n
        obj.series = series
        return obj

    def rank(self):
        """
        Returns the rank of the Lie algebra
        """
        return self.n

    def series(self):
        """
        Returns the type of the Lie algebra
        """
        return self.series

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\pythonrationalfield.py ===
"""Implementation of :class:`PythonRationalField` class. """


from sympy.polys.domains.groundtypes import PythonInteger, PythonRational, SymPyRational
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class PythonRationalField(RationalField):
    """Rational field based on :ref:`MPQ`.

    This will be used as :ref:`QQ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of :ref:`MPQ`.
    """

    dtype = PythonRational
    zero = dtype(0)
    one = dtype(1)
    alias = 'QQ_python'

    def __init__(self):
        pass

    def get_ring(self):
        """Returns ring associated with ``self``. """
        from sympy.polys.domains import PythonIntegerRing
        return PythonIntegerRing()

    def to_sympy(self, a):
        """Convert `a` to a SymPy object. """
        return SymPyRational(a.numerator, a.denominator)

    def from_sympy(self, a):
        """Convert SymPy's Rational to `dtype`. """
        if a.is_Rational:
            return PythonRational(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            p, q = RR.to_rational(a)
            return PythonRational(int(p), int(q))
        else:
            raise CoercionFailed("expected `Rational` object, got %s" % a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return PythonRational(a)

    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return a

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
        return PythonRational(PythonInteger(a))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
        return PythonRational(PythonInteger(a.numer()),
                              PythonInteger(a.denom()))

    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        p, q = K0.to_rational(a)
        return PythonRational(int(p), int(q))

    def numer(self, a):
        """Returns numerator of `a`. """
        return a.numerator

    def denom(self, a):
        """Returns denominator of `a`. """
        return a.denominator

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\residues.py ===
"""
This module implements the Residue function and related tools for working
with residues.
"""

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.utilities.timeutils import timethis


@timethis('residue')
def residue(expr, x, x0):
    """
    Finds the residue of ``expr`` at the point x=x0.

    The residue is defined as the coefficient of ``1/(x-x0)`` in the power series
    expansion about ``x=x0``.

    Examples
    ========

    >>> from sympy import Symbol, residue, sin
    >>> x = Symbol("x")
    >>> residue(1/x, x, 0)
    1
    >>> residue(1/x**2, x, 0)
    0
    >>> residue(2/sin(x), x, 0)
    2

    This function is essential for the Residue Theorem [1].

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Residue_theorem
    """
    # The current implementation uses series expansion to
    # calculate it. A more general implementation is explained in
    # the section 5.6 of the Bronstein's book {M. Bronstein:
    # Symbolic Integration I, Springer Verlag (2005)}. For purely
    # rational functions, the algorithm is much easier. See
    # sections 2.4, 2.5, and 2.7 (this section actually gives an
    # algorithm for computing any Laurent series coefficient for
    # a rational function). The theory in section 2.4 will help to
    # understand why the resultant works in the general algorithm.
    # For the definition of a resultant, see section 1.4 (and any
    # previous sections for more review).

    from sympy.series.order import Order
    from sympy.simplify.radsimp import collect
    expr = sympify(expr)
    if x0 != 0:
        expr = expr.subs(x, x + x0)
    for n in (0, 1, 2, 4, 8, 16, 32):
        s = expr.nseries(x, n=n)
        if not s.has(Order) or s.getn() >= 0:
            break
    s = collect(s.removeO(), x)
    if s.is_Add:
        args = s.args
    else:
        args = [s]
    res = S.Zero
    for arg in args:
        c, m = arg.as_coeff_mul(x)
        m = Mul(*m)
        if not (m in (S.One, x) or (m.is_Pow and m.exp.is_Integer)):
            raise NotImplementedError('term of unexpected form: %s' % m)
        if m == 1/x:
            res += c
    return res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_parse_dates.py ===
"""
Tests date parsing functionality for all of the
parsers defined in parsers.py
"""

from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
from io import StringIO

from dateutil.parser import parse as du_parse
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import parsing

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at

from pandas.io.parsers import read_csv

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@xfail_pyarrow
def test_read_csv_with_custom_date_parser(all_parsers):
    # GH36111
    def __custom_date_parser(time):
        time = time.astype(np.float64)
        time = time.astype(int)  # convert float seconds to int type
        return pd.to_timedelta(time, unit="s")

    testdata = StringIO(
        """time e n h
        41047.00 -98573.7297 871458.0640 389.0089
        41048.00 -98573.7299 871458.0640 389.0089
        41049.00 -98573.7300 871458.0642 389.0088
        41050.00 -98573.7299 871458.0643 389.0088
        41051.00 -98573.7302 871458.0640 389.0086
        """
    )
    result = all_parsers.read_csv_check_warnings(
        FutureWarning,
        "Please use 'date_format' instead",
        testdata,
        delim_whitespace=True,
        parse_dates=True,
        date_parser=__custom_date_parser,
        index_col="time",
    )
    time = [41047, 41048, 41049, 41050, 41051]
    time = pd.TimedeltaIndex([pd.to_timedelta(i, unit="s") for i in time], name="time")
    expected = DataFrame(
        {
            "e": [-98573.7297, -98573.7299, -98573.7300, -98573.7299, -98573.7302],
            "n": [871458.0640, 871458.0640, 871458.0642, 871458.0643, 871458.0640],
            "h": [389.0089, 389.0089, 389.0088, 389.0088, 389.0086],
        },
        index=time,
    )

    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_read_csv_with_custom_date_parser_parse_dates_false(all_parsers):
    # GH44366
    def __custom_date_parser(time):
        time = time.astype(np.float64)
        time = time.astype(int)  # convert float seconds to int type
        return pd.to_timedelta(time, unit="s")

    testdata = StringIO(
        """time e
        41047.00 -93.77
        41048.00 -95.79
        41049.00 -98.73
        41050.00 -93.99
        41051.00 -97.72
        """
    )
    result = all_parsers.read_csv_check_warnings(
        FutureWarning,
        "Please use 'date_format' instead",
        testdata,
        delim_whitespace=True,
        parse_dates=False,
        date_parser=__custom_date_parser,
        index_col="time",
    )
    time = Series([41047.00, 41048.00, 41049.00, 41050.00, 41051.00], name="time")
    expected = DataFrame(
        {"e": [-93.77, -95.79, -98.73, -93.99, -97.72]},
        index=time,
    )

    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_separator_date_conflict(all_parsers):
    # Regression test for gh-4678
    #
    # Make sure thousands separator and
    # date parsing do not conflict.
    parser = all_parsers
    data = "06-02-2013;13:00;1-000.215"
    expected = DataFrame(
        [[datetime(2013, 6, 2, 13, 0, 0), 1000.215]], columns=["Date", 2]
    )

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        df = parser.read_csv(
            StringIO(data),
            sep=";",
            thousands="-",
            parse_dates={"Date": [0, 1]},
            header=None,
        )
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("keep_date_col", [True, False])
def test_multiple_date_col_custom(all_parsers, keep_date_col, request):
    data = """\
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    parser = all_parsers

    if keep_date_col and parser.engine == "pyarrow":
        # For this to pass, we need to disable auto-inference on the date columns
        # in parse_dates. We have no way of doing this though
        mark = pytest.mark.xfail(
            reason="pyarrow doesn't support disabling auto-inference on column numbers."
        )
        request.applymarker(mark)

    def date_parser(*date_cols):
        """
        Test date parser.

        Parameters
        ----------
        date_cols : args
            The list of data columns to parse.

        Returns
        -------
        parsed : Series
        """
        return parsing.try_parse_dates(
            parsing.concat_date_cols(date_cols), parser=du_parse
        )

    kwds = {
        "header": None,
        "date_parser": date_parser,
        "parse_dates": {"actual": [1, 2], "nominal": [1, 3]},
        "keep_date_col": keep_date_col,
        "names": ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"],
    }
    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        **kwds,
        raise_on_extra_warnings=False,
    )

    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                datetime(1999, 1, 27, 18, 56),
                "KORD",
                "19990127",
                " 19:00:00",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                datetime(1999, 1, 27, 19, 56),
                "KORD",
                "19990127",
                " 20:00:00",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 20, 56),
                "KORD",
                "19990127",
                " 21:00:00",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 21, 18),
                "KORD",
                "19990127",
                " 21:00:00",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                datetime(1999, 1, 27, 21, 56),
                "KORD",
                "19990127",
                " 22:00:00",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                datetime(1999, 1, 27, 22, 56),
                "KORD",
                "19990127",
                " 23:00:00",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "actual",
            "nominal",
            "X0",
            "X1",
            "X2",
            "X3",
            "X4",
            "X5",
            "X6",
            "X7",
            "X8",
        ],
    )

    if not keep_date_col:
        expected = expected.drop(["X1", "X2", "X3"], axis=1)

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("container", [list, tuple, Index, Series])
@pytest.mark.parametrize("dim", [1, 2])
def test_concat_date_col_fail(container, dim):
    msg = "not all elements from date_cols are numpy arrays"
    value = "19990127"

    date_cols = tuple(container([value]) for _ in range(dim))

    with pytest.raises(ValueError, match=msg):
        parsing.concat_date_cols(date_cols)


@pytest.mark.parametrize("keep_date_col", [True, False])
def test_multiple_date_col(all_parsers, keep_date_col, request):
    data = """\
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    parser = all_parsers

    if keep_date_col and parser.engine == "pyarrow":
        # For this to pass, we need to disable auto-inference on the date columns
        # in parse_dates. We have no way of doing this though
        mark = pytest.mark.xfail(
            reason="pyarrow doesn't support disabling auto-inference on column numbers."
        )
        request.applymarker(mark)

    depr_msg = "The 'keep_date_col' keyword in pd.read_csv is deprecated"

    kwds = {
        "header": None,
        "parse_dates": [[1, 2], [1, 3]],
        "keep_date_col": keep_date_col,
        "names": ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"],
    }
    with tm.assert_produces_warning(
        (DeprecationWarning, FutureWarning), match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(StringIO(data), **kwds)

    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                datetime(1999, 1, 27, 18, 56),
                "KORD",
                "19990127",
                " 19:00:00",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                datetime(1999, 1, 27, 19, 56),
                "KORD",
                "19990127",
                " 20:00:00",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 20, 56),
                "KORD",
                "19990127",
                " 21:00:00",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 21, 18),
                "KORD",
                "19990127",
                " 21:00:00",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                datetime(1999, 1, 27, 21, 56),
                "KORD",
                "19990127",
                " 22:00:00",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                datetime(1999, 1, 27, 22, 56),
                "KORD",
                "19990127",
                " 23:00:00",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "X1_X2",
            "X1_X3",
            "X0",
            "X1",
            "X2",
            "X3",
            "X4",
            "X5",
            "X6",
            "X7",
            "X8",
        ],
    )

    if not keep_date_col:
        expected = expected.drop(["X1", "X2", "X3"], axis=1)

    tm.assert_frame_equal(result, expected)


def test_date_col_as_index_col(all_parsers):
    data = """\
KORD,19990127 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
"""
    parser = all_parsers
    kwds = {
        "header": None,
        "parse_dates": [1],
        "index_col": 1,
        "names": ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"],
    }
    result = parser.read_csv(StringIO(data), **kwds)

    index = Index(
        [
            datetime(1999, 1, 27, 19, 0),
            datetime(1999, 1, 27, 20, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 21, 0),
            datetime(1999, 1, 27, 22, 0),
        ],
        name="X1",
    )
    expected = DataFrame(
        [
            ["KORD", " 18:56:00", 0.81, 2.81, 7.2, 0.0, 280.0],
            ["KORD", " 19:56:00", 0.01, 2.21, 7.2, 0.0, 260.0],
            ["KORD", " 20:56:00", -0.59, 2.21, 5.7, 0.0, 280.0],
            ["KORD", " 21:18:00", -0.99, 2.01, 3.6, 0.0, 270.0],
            ["KORD", " 21:56:00", -0.59, 1.71, 5.1, 0.0, 290.0],
        ],
        columns=["X0", "X2", "X3", "X4", "X5", "X6", "X7"],
        index=index,
    )
    if parser.engine == "pyarrow":
        # https://github.com/pandas-dev/pandas/issues/44231
        # pyarrow 6.0 starts to infer time type
        expected["X2"] = pd.to_datetime("1970-01-01" + expected["X2"]).dt.time

    tm.assert_frame_equal(result, expected)


def test_multiple_date_cols_int_cast(all_parsers):
    data = (
        "KORD,19990127, 19:00:00, 18:56:00, 0.8100\n"
        "KORD,19990127, 20:00:00, 19:56:00, 0.0100\n"
        "KORD,19990127, 21:00:00, 20:56:00, -0.5900\n"
        "KORD,19990127, 21:00:00, 21:18:00, -0.9900\n"
        "KORD,19990127, 22:00:00, 21:56:00, -0.5900\n"
        "KORD,19990127, 23:00:00, 22:56:00, -0.5900"
    )
    parse_dates = {"actual": [1, 2], "nominal": [1, 3]}
    parser = all_parsers

    kwds = {
        "header": None,
        "parse_dates": parse_dates,
        "date_parser": pd.to_datetime,
    }
    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        **kwds,
        raise_on_extra_warnings=False,
    )

    expected = DataFrame(
        [
            [datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 18, 56), "KORD", 0.81],
            [datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 19, 56), "KORD", 0.01],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 20, 56),
                "KORD",
                -0.59,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                datetime(1999, 1, 27, 21, 18),
                "KORD",
                -0.99,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                datetime(1999, 1, 27, 21, 56),
                "KORD",
                -0.59,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                datetime(1999, 1, 27, 22, 56),
                "KORD",
                -0.59,
            ],
        ],
        columns=["actual", "nominal", 0, 4],
    )

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


def test_multiple_date_col_timestamp_parse(all_parsers):
    parser = all_parsers
    data = """05/31/2012,15:30:00.029,1306.25,1,E,0,,1306.25
05/31/2012,15:30:00.029,1306.25,8,E,0,,1306.25"""

    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        parse_dates=[[0, 1]],
        header=None,
        date_parser=Timestamp,
        raise_on_extra_warnings=False,
    )
    expected = DataFrame(
        [
            [
                Timestamp("05/31/2012, 15:30:00.029"),
                1306.25,
                1,
                "E",
                0,
                np.nan,
                1306.25,
            ],
            [
                Timestamp("05/31/2012, 15:30:00.029"),
                1306.25,
                8,
                "E",
                0,
                np.nan,
                1306.25,
            ],
        ],
        columns=["0_1", 2, 3, 4, 5, 6, 7],
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_multiple_date_cols_with_header(all_parsers):
    parser = all_parsers
    data = """\
ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000"""

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(StringIO(data), parse_dates={"nominal": [1, 2]})
    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                "KORD",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                "KORD",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                "KORD",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                "KORD",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "nominal",
            "ID",
            "ActualTime",
            "TDew",
            "TAir",
            "Windspeed",
            "Precip",
            "WindDir",
        ],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,parse_dates,msg",
    [
        (
            """\
date_NominalTime,date,NominalTime
KORD1,19990127, 19:00:00
KORD2,19990127, 20:00:00""",
            [[1, 2]],
            ("New date column already in dict date_NominalTime"),
        ),
        (
            """\
ID,date,nominalTime
KORD,19990127, 19:00:00
KORD,19990127, 20:00:00""",
            {"ID": [1, 2]},
            "Date column ID already in dict",
        ),
    ],
)
def test_multiple_date_col_name_collision(all_parsers, data, parse_dates, msg):
    parser = all_parsers

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(
            (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
        ):
            parser.read_csv(StringIO(data), parse_dates=parse_dates)


def test_date_parser_int_bug(all_parsers):
    # see gh-3071
    parser = all_parsers
    data = (
        "posix_timestamp,elapsed,sys,user,queries,query_time,rows,"
        "accountid,userid,contactid,level,silo,method\n"
        "1343103150,0.062353,0,4,6,0.01690,3,"
        "12345,1,-1,3,invoice_InvoiceResource,search\n"
    )

    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        index_col=0,
        parse_dates=[0],
        # Note: we must pass tz and then drop the tz attribute
        # (if we don't CI will flake out depending on the runner's local time)
        date_parser=lambda x: datetime.fromtimestamp(int(x), tz=timezone.utc).replace(
            tzinfo=None
        ),
        raise_on_extra_warnings=False,
    )
    expected = DataFrame(
        [
            [
                0.062353,
                0,
                4,
                6,
                0.01690,
                3,
                12345,
                1,
                -1,
                3,
                "invoice_InvoiceResource",
                "search",
            ]
        ],
        columns=[
            "elapsed",
            "sys",
            "user",
            "queries",
            "query_time",
            "rows",
            "accountid",
            "userid",
            "contactid",
            "level",
            "silo",
            "method",
        ],
        index=Index([Timestamp("2012-07-24 04:12:30")], name="posix_timestamp"),
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_nat_parse(all_parsers):
    # see gh-3062
    parser = all_parsers
    df = DataFrame(
        {
            "A": np.arange(10, dtype="float64"),
            "B": Timestamp("20010101").as_unit("ns"),
        }
    )
    df.iloc[3:6, :] = np.nan

    with tm.ensure_clean("__nat_parse_.csv") as path:
        df.to_csv(path)

        result = parser.read_csv(path, index_col=0, parse_dates=["B"])
        tm.assert_frame_equal(result, df)


@skip_pyarrow
def test_csv_custom_parser(all_parsers):
    data = """A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        date_parser=lambda x: datetime.strptime(x, "%Y%m%d"),
    )
    expected = parser.read_csv(StringIO(data), parse_dates=True)
    tm.assert_frame_equal(result, expected)
    result = parser.read_csv(StringIO(data), date_format="%Y%m%d")
    tm.assert_frame_equal(result, expected)


@skip_pyarrow
def test_parse_dates_implicit_first_col(all_parsers):
    data = """A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), parse_dates=True)

    expected = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_parse_dates_string(all_parsers):
    data = """date,A,B,C
20090101,a,1,2
20090102,b,3,4
20090103,c,4,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col="date", parse_dates=["date"])
    # freq doesn't round-trip
    index = date_range("1/1/2009", periods=3, name="date")._with_freq(None)

    expected = DataFrame(
        {"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]}, index=index
    )
    tm.assert_frame_equal(result, expected)


# Bug in https://github.com/dateutil/dateutil/issues/217
# has been addressed, but we just don't pass in the `yearfirst`
@pytest.mark.xfail(reason="yearfirst is not surfaced in read_*")
@pytest.mark.parametrize("parse_dates", [[["date", "time"]], [[0, 1]]])
def test_yy_format_with_year_first(all_parsers, parse_dates):
    data = """date,time,B,C
090131,0010,1,2
090228,1020,3,4
090331,0830,5,6
"""
    parser = all_parsers
    result = parser.read_csv_check_warnings(
        UserWarning,
        "Could not infer format",
        StringIO(data),
        index_col=0,
        parse_dates=parse_dates,
    )
    index = DatetimeIndex(
        [
            datetime(2009, 1, 31, 0, 10, 0),
            datetime(2009, 2, 28, 10, 20, 0),
            datetime(2009, 3, 31, 8, 30, 0),
        ],
        dtype=object,
        name="date_time",
    )
    expected = DataFrame({"B": [1, 3, 5], "C": [2, 4, 6]}, index=index)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize("parse_dates", [[0, 2], ["a", "c"]])
def test_parse_dates_column_list(all_parsers, parse_dates):
    data = "a,b,c\n01/01/2010,1,15/02/2010"
    parser = all_parsers

    expected = DataFrame(
        {"a": [datetime(2010, 1, 1)], "b": [1], "c": [datetime(2010, 2, 15)]}
    )
    expected = expected.set_index(["a", "b"])

    result = parser.read_csv(
        StringIO(data), index_col=[0, 1], parse_dates=parse_dates, dayfirst=True
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize("index_col", [[0, 1], [1, 0]])
def test_multi_index_parse_dates(all_parsers, index_col):
    data = """index1,index2,A,B,C
20090101,one,a,1,2
20090101,two,b,3,4
20090101,three,c,4,5
20090102,one,a,1,2
20090102,two,b,3,4
20090102,three,c,4,5
20090103,one,a,1,2
20090103,two,b,3,4
20090103,three,c,4,5
"""
    parser = all_parsers
    index = MultiIndex.from_product(
        [
            (datetime(2009, 1, 1), datetime(2009, 1, 2), datetime(2009, 1, 3)),
            ("one", "two", "three"),
        ],
        names=["index1", "index2"],
    )

    # Out of order.
    if index_col == [1, 0]:
        index = index.swaplevel(0, 1)

    expected = DataFrame(
        [
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
            ["a", 1, 2],
            ["b", 3, 4],
            ["c", 4, 5],
        ],
        columns=["A", "B", "C"],
        index=index,
    )
    result = parser.read_csv_check_warnings(
        UserWarning,
        "Could not infer format",
        StringIO(data),
        index_col=index_col,
        parse_dates=True,
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
@pytest.mark.parametrize("kwargs", [{"dayfirst": True}, {"day_first": True}])
def test_parse_dates_custom_euro_format(all_parsers, kwargs):
    parser = all_parsers
    data = """foo,bar,baz
31/01/2010,1,2
01/02/2010,1,NA
02/02/2010,1,2
"""
    if "dayfirst" in kwargs:
        df = parser.read_csv_check_warnings(
            FutureWarning,
            "use 'date_format' instead",
            StringIO(data),
            names=["time", "Q", "NTU"],
            date_parser=lambda d: du_parse(d, **kwargs),
            header=0,
            index_col=0,
            parse_dates=True,
            na_values=["NA"],
        )
        exp_index = Index(
            [datetime(2010, 1, 31), datetime(2010, 2, 1), datetime(2010, 2, 2)],
            name="time",
        )
        expected = DataFrame(
            {"Q": [1, 1, 1], "NTU": [2, np.nan, 2]},
            index=exp_index,
            columns=["Q", "NTU"],
        )
        tm.assert_frame_equal(df, expected)
    else:
        msg = "got an unexpected keyword argument 'day_first'"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv_check_warnings(
                FutureWarning,
                "use 'date_format' instead",
                StringIO(data),
                names=["time", "Q", "NTU"],
                date_parser=lambda d: du_parse(d, **kwargs),
                skiprows=[0],
                index_col=0,
                parse_dates=True,
                na_values=["NA"],
            )


def test_parse_tz_aware(all_parsers):
    # See gh-1693
    parser = all_parsers
    data = "Date,x\n2012-06-13T01:39:00Z,0.5"

    result = parser.read_csv(StringIO(data), index_col=0, parse_dates=True)
    # TODO: make unit check more specific
    if parser.engine == "pyarrow":
        result.index = result.index.as_unit("ns")
    expected = DataFrame(
        {"x": [0.5]}, index=Index([Timestamp("2012-06-13 01:39:00+00:00")], name="Date")
    )
    if parser.engine == "pyarrow":
        expected_tz = pytz.utc
    else:
        expected_tz = timezone.utc
    tm.assert_frame_equal(result, expected)
    assert result.index.tz is expected_tz


@xfail_pyarrow
@pytest.mark.parametrize(
    "parse_dates,index_col",
    [({"nominal": [1, 2]}, "nominal"), ({"nominal": [1, 2]}, 0), ([[1, 2]], 0)],
)
def test_multiple_date_cols_index(all_parsers, parse_dates, index_col):
    parser = all_parsers
    data = """
ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir
KORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                "KORD1",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                "KORD2",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD3",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD4",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                "KORD5",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                "KORD6",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=[
            "nominal",
            "ID",
            "ActualTime",
            "TDew",
            "TAir",
            "Windspeed",
            "Precip",
            "WindDir",
        ],
    )
    expected = expected.set_index("nominal")

    if not isinstance(parse_dates, dict):
        expected.index.name = "date_NominalTime"

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), parse_dates=parse_dates, index_col=index_col
        )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_multiple_date_cols_chunked(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""

    expected = DataFrame(
        [
            [
                datetime(1999, 1, 27, 19, 0),
                "KORD",
                " 18:56:00",
                0.81,
                2.81,
                7.2,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 20, 0),
                "KORD",
                " 19:56:00",
                0.01,
                2.21,
                7.2,
                0.0,
                260.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 20:56:00",
                -0.59,
                2.21,
                5.7,
                0.0,
                280.0,
            ],
            [
                datetime(1999, 1, 27, 21, 0),
                "KORD",
                " 21:18:00",
                -0.99,
                2.01,
                3.6,
                0.0,
                270.0,
            ],
            [
                datetime(1999, 1, 27, 22, 0),
                "KORD",
                " 21:56:00",
                -0.59,
                1.71,
                5.1,
                0.0,
                290.0,
            ],
            [
                datetime(1999, 1, 27, 23, 0),
                "KORD",
                " 22:56:00",
                -0.59,
                1.71,
                4.6,
                0.0,
                280.0,
            ],
        ],
        columns=["nominal", "ID", "actualTime", "A", "B", "C", "D", "E"],
    )
    expected = expected.set_index("nominal")

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        with parser.read_csv(
            StringIO(data),
            parse_dates={"nominal": [1, 2]},
            index_col="nominal",
            chunksize=2,
        ) as reader:
            chunks = list(reader)

    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


def test_multiple_date_col_named_index_compat(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
    ):
        with_indices = parser.read_csv(
            StringIO(data), parse_dates={"nominal": [1, 2]}, index_col="nominal"
        )

    with tm.assert_produces_warning(
        (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
    ):
        with_names = parser.read_csv(
            StringIO(data),
            index_col="nominal",
            parse_dates={"nominal": ["date", "nominalTime"]},
        )
    tm.assert_frame_equal(with_indices, with_names)


def test_multiple_date_col_multiple_index_compat(all_parsers):
    parser = all_parsers
    data = """\
ID,date,nominalTime,actualTime,A,B,C,D,E
KORD,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000
"""
    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), index_col=["nominal", "ID"], parse_dates={"nominal": [1, 2]}
        )
    with tm.assert_produces_warning(
        (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
    ):
        expected = parser.read_csv(StringIO(data), parse_dates={"nominal": [1, 2]})

    expected = expected.set_index(["nominal", "ID"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{}, {"index_col": "C"}])
def test_read_with_parse_dates_scalar_non_bool(all_parsers, kwargs):
    # see gh-5636
    parser = all_parsers
    msg = (
        "Only booleans, lists, and dictionaries "
        "are accepted for the 'parse_dates' parameter"
    )
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates="C", **kwargs)


@pytest.mark.parametrize("parse_dates", [(1,), np.array([4, 5]), {1, 3}])
def test_read_with_parse_dates_invalid_type(all_parsers, parse_dates):
    parser = all_parsers
    msg = (
        "Only booleans, lists, and dictionaries "
        "are accepted for the 'parse_dates' parameter"
    )
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates=(1,))


@pytest.mark.parametrize("cache_dates", [True, False])
@pytest.mark.parametrize("value", ["nan", ""])
def test_bad_date_parse(all_parsers, cache_dates, value):
    # if we have an invalid date make sure that we handle this with
    # and w/o the cache properly
    parser = all_parsers
    s = StringIO((f"{value},\n") * (start_caching_at + 1))

    parser.read_csv(
        s,
        header=None,
        names=["foo", "bar"],
        parse_dates=["foo"],
        cache_dates=cache_dates,
    )


@pytest.mark.parametrize("cache_dates", [True, False])
@pytest.mark.parametrize("value", ["0"])
def test_bad_date_parse_with_warning(all_parsers, cache_dates, value):
    # if we have an invalid date make sure that we handle this with
    # and w/o the cache properly.
    parser = all_parsers
    s = StringIO((f"{value},\n") * 50000)

    if parser.engine == "pyarrow":
        # pyarrow reads "0" as 0 (of type int64), and so
        # pandas doesn't try to guess the datetime format
        # TODO: parse dates directly in pyarrow, see
        # https://github.com/pandas-dev/pandas/issues/48017
        warn = None
    elif cache_dates:
        # Note: warning is not raised if 'cache_dates', because here there is only a
        # single unique date and hence no risk of inconsistent parsing.
        warn = None
    else:
        warn = UserWarning
    parser.read_csv_check_warnings(
        warn,
        "Could not infer format",
        s,
        header=None,
        names=["foo", "bar"],
        parse_dates=["foo"],
        cache_dates=cache_dates,
        raise_on_extra_warnings=False,
    )


@xfail_pyarrow
def test_parse_dates_empty_string(all_parsers):
    # see gh-2263
    parser = all_parsers
    data = "Date,test\n2012-01-01,1\n,2"
    result = parser.read_csv(StringIO(data), parse_dates=["Date"], na_filter=False)

    expected = DataFrame(
        [[datetime(2012, 1, 1), 1], [pd.NaT, 2]], columns=["Date", "test"]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "reader", ["read_csv_check_warnings", "read_table_check_warnings"]
)
def test_parse_dates_infer_datetime_format_warning(all_parsers, reader):
    # GH 49024, 51017
    parser = all_parsers
    data = "Date,test\n2012-01-01,1\n,2"

    getattr(parser, reader)(
        FutureWarning,
        "The argument 'infer_datetime_format' is deprecated",
        StringIO(data),
        parse_dates=["Date"],
        infer_datetime_format=True,
        sep=",",
        raise_on_extra_warnings=False,
    )


@pytest.mark.parametrize(
    "reader", ["read_csv_check_warnings", "read_table_check_warnings"]
)
def test_parse_dates_date_parser_and_date_format(all_parsers, reader):
    # GH 50601
    parser = all_parsers
    data = "Date,test\n2012-01-01,1\n,2"
    msg = "Cannot use both 'date_parser' and 'date_format'"
    with pytest.raises(TypeError, match=msg):
        getattr(parser, reader)(
            FutureWarning,
            "use 'date_format' instead",
            StringIO(data),
            parse_dates=["Date"],
            date_parser=pd.to_datetime,
            date_format="ISO8601",
            sep=",",
        )


@xfail_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            "a\n04.15.2016",
            {"parse_dates": ["a"]},
            DataFrame([datetime(2016, 4, 15)], columns=["a"]),
        ),
        (
            "a\n04.15.2016",
            {"parse_dates": True, "index_col": 0},
            DataFrame(index=DatetimeIndex(["2016-04-15"], name="a"), columns=[]),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",
            {"parse_dates": ["a", "b"]},
            DataFrame(
                [[datetime(2016, 4, 15), datetime(2013, 9, 16)]], columns=["a", "b"]
            ),
        ),
        (
            "a,b\n04.15.2016,09.16.2013",
            {"parse_dates": True, "index_col": [0, 1]},
            DataFrame(
                index=MultiIndex.from_tuples(
                    [(datetime(2016, 4, 15), datetime(2013, 9, 16))], names=["a", "b"]
                ),
                columns=[],
            ),
        ),
    ],
)
def test_parse_dates_no_convert_thousands(all_parsers, data, kwargs, expected):
    # see gh-14066
    parser = all_parsers

    result = parser.read_csv(StringIO(data), thousands=".", **kwargs)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_parse_date_time_multi_level_column_name(all_parsers):
    data = """\
D,T,A,B
date, time,a,b
2001-01-05, 09:00:00, 0.0, 10.
2001-01-06, 00:00:00, 1.0, 11.
"""
    parser = all_parsers
    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        header=[0, 1],
        parse_dates={"date_time": [0, 1]},
        date_parser=pd.to_datetime,
    )

    expected_data = [
        [datetime(2001, 1, 5, 9, 0, 0), 0.0, 10.0],
        [datetime(2001, 1, 6, 0, 0, 0), 1.0, 11.0],
    ]
    expected = DataFrame(expected_data, columns=["date_time", ("A", "a"), ("B", "b")])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            """\
date,time,a,b
2001-01-05, 10:00:00, 0.0, 10.
2001-01-05, 00:00:00, 1., 11.
""",
            {"header": 0, "parse_dates": {"date_time": [0, 1]}},
            DataFrame(
                [
                    [datetime(2001, 1, 5, 10, 0, 0), 0.0, 10],
                    [datetime(2001, 1, 5, 0, 0, 0), 1.0, 11.0],
                ],
                columns=["date_time", "a", "b"],
            ),
        ),
        (
            (
                "KORD,19990127, 19:00:00, 18:56:00, 0.8100\n"
                "KORD,19990127, 20:00:00, 19:56:00, 0.0100\n"
                "KORD,19990127, 21:00:00, 20:56:00, -0.5900\n"
                "KORD,19990127, 21:00:00, 21:18:00, -0.9900\n"
                "KORD,19990127, 22:00:00, 21:56:00, -0.5900\n"
                "KORD,19990127, 23:00:00, 22:56:00, -0.5900"
            ),
            {"header": None, "parse_dates": {"actual": [1, 2], "nominal": [1, 3]}},
            DataFrame(
                [
                    [
                        datetime(1999, 1, 27, 19, 0),
                        datetime(1999, 1, 27, 18, 56),
                        "KORD",
                        0.81,
                    ],
                    [
                        datetime(1999, 1, 27, 20, 0),
                        datetime(1999, 1, 27, 19, 56),
                        "KORD",
                        0.01,
                    ],
                    [
                        datetime(1999, 1, 27, 21, 0),
                        datetime(1999, 1, 27, 20, 56),
                        "KORD",
                        -0.59,
                    ],
                    [
                        datetime(1999, 1, 27, 21, 0),
                        datetime(1999, 1, 27, 21, 18),
                        "KORD",
                        -0.99,
                    ],
                    [
                        datetime(1999, 1, 27, 22, 0),
                        datetime(1999, 1, 27, 21, 56),
                        "KORD",
                        -0.59,
                    ],
                    [
                        datetime(1999, 1, 27, 23, 0),
                        datetime(1999, 1, 27, 22, 56),
                        "KORD",
                        -0.59,
                    ],
                ],
                columns=["actual", "nominal", 0, 4],
            ),
        ),
    ],
)
def test_parse_date_time(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        date_parser=pd.to_datetime,
        **kwargs,
        raise_on_extra_warnings=False,
    )

    # Python can sometimes be flaky about how
    # the aggregated columns are entered, so
    # this standardizes the order.
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)


def test_parse_date_fields(all_parsers):
    parser = all_parsers
    data = "year,month,day,a\n2001,01,10,10.\n2001,02,1,11."
    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        header=0,
        parse_dates={"ymd": [0, 1, 2]},
        date_parser=lambda x: x,
        raise_on_extra_warnings=False,
    )

    expected = DataFrame(
        [[datetime(2001, 1, 10), 10.0], [datetime(2001, 2, 1), 11.0]],
        columns=["ymd", "a"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("key", "value", "warn"),
    [
        (
            "date_parser",
            lambda x: pd.to_datetime(x, format="%Y %m %d %H %M %S"),
            FutureWarning,
        ),
        ("date_format", "%Y %m %d %H %M %S", None),
    ],
)
def test_parse_date_all_fields(all_parsers, key, value, warn):
    parser = all_parsers
    data = """\
year,month,day,hour,minute,second,a,b
2001,01,05,10,00,0,0.0,10.
2001,01,5,10,0,00,1.,11.
"""
    result = parser.read_csv_check_warnings(
        warn,
        "use 'date_format' instead",
        StringIO(data),
        header=0,
        parse_dates={"ymdHMS": [0, 1, 2, 3, 4, 5]},
        **{key: value},
        raise_on_extra_warnings=False,
    )
    expected = DataFrame(
        [
            [datetime(2001, 1, 5, 10, 0, 0), 0.0, 10.0],
            [datetime(2001, 1, 5, 10, 0, 0), 1.0, 11.0],
        ],
        columns=["ymdHMS", "a", "b"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("key", "value", "warn"),
    [
        (
            "date_parser",
            lambda x: pd.to_datetime(x, format="%Y %m %d %H %M %S.%f"),
            FutureWarning,
        ),
        ("date_format", "%Y %m %d %H %M %S.%f", None),
    ],
)
def test_datetime_fractional_seconds(all_parsers, key, value, warn):
    parser = all_parsers
    data = """\
year,month,day,hour,minute,second,a,b
2001,01,05,10,00,0.123456,0.0,10.
2001,01,5,10,0,0.500000,1.,11.
"""
    result = parser.read_csv_check_warnings(
        warn,
        "use 'date_format' instead",
        StringIO(data),
        header=0,
        parse_dates={"ymdHMS": [0, 1, 2, 3, 4, 5]},
        **{key: value},
        raise_on_extra_warnings=False,
    )
    expected = DataFrame(
        [
            [datetime(2001, 1, 5, 10, 0, 0, microsecond=123456), 0.0, 10.0],
            [datetime(2001, 1, 5, 10, 0, 0, microsecond=500000), 1.0, 11.0],
        ],
        columns=["ymdHMS", "a", "b"],
    )
    tm.assert_frame_equal(result, expected)


def test_generic(all_parsers):
    parser = all_parsers
    data = "year,month,day,a\n2001,01,10,10.\n2001,02,1,11."

    def parse_function(yy, mm):
        return [date(year=int(y), month=int(m), day=1) for y, m in zip(yy, mm)]

    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        header=0,
        parse_dates={"ym": [0, 1]},
        date_parser=parse_function,
        raise_on_extra_warnings=False,
    )
    expected = DataFrame(
        [[date(2001, 1, 1), 10, 10.0], [date(2001, 2, 1), 1, 11.0]],
        columns=["ym", "day", "a"],
    )
    expected["ym"] = expected["ym"].astype("datetime64[ns]")
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow
def test_date_parser_resolution_if_not_ns(all_parsers):
    # see gh-10245
    parser = all_parsers
    data = """\
date,time,prn,rxstatus
2013-11-03,19:00:00,126,00E80000
2013-11-03,19:00:00,23,00E80000
2013-11-03,19:00:00,13,00E80000
"""

    def date_parser(dt, time):
        try:
            arr = dt + "T" + time
        except TypeError:
            # dt & time are date/time objects
            arr = [datetime.combine(d, t) for d, t in zip(dt, time)]
        return np.array(arr, dtype="datetime64[s]")

    result = parser.read_csv_check_warnings(
        FutureWarning,
        "use 'date_format' instead",
        StringIO(data),
        date_parser=date_parser,
        parse_dates={"datetime": ["date", "time"]},
        index_col=["datetime", "prn"],
    )

    datetimes = np.array(["2013-11-03T19:00:00"] * 3, dtype="datetime64[s]")
    expected = DataFrame(
        data={"rxstatus": ["00E80000"] * 3},
        index=MultiIndex.from_arrays(
            [datetimes, [126, 23, 13]],
            names=["datetime", "prn"],
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_parse_date_column_with_empty_string(all_parsers):
    # see gh-6428
    parser = all_parsers
    data = "case,opdate\n7,10/18/2006\n7,10/18/2008\n621, "
    result = parser.read_csv(StringIO(data), parse_dates=["opdate"])

    expected_data = [[7, "10/18/2006"], [7, "10/18/2008"], [621, " "]]
    expected = DataFrame(expected_data, columns=["case", "opdate"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            "a\n135217135789158401\n1352171357E+5",
            DataFrame({"a": [135217135789158401, 135217135700000]}, dtype="float64"),
        ),
        (
            "a\n99999999999\n123456789012345\n1234E+0",
            DataFrame({"a": [99999999999, 123456789012345, 1234]}, dtype="float64"),
        ),
    ],
)
@pytest.mark.parametrize("parse_dates", [True, False])
def test_parse_date_float(all_parsers, data, expected, parse_dates):
    # see gh-2697
    #
    # Date parsing should fail, so we leave the data untouched
    # (i.e. float precision should remain unchanged).
    parser = all_parsers

    result = parser.read_csv(StringIO(data), parse_dates=parse_dates)
    tm.assert_frame_equal(result, expected)


def test_parse_timezone(all_parsers):
    # see gh-22256
    parser = all_parsers
    data = """dt,val
              2018-01-04 09:01:00+09:00,23350
              2018-01-04 09:02:00+09:00,23400
              2018-01-04 09:03:00+09:00,23400
              2018-01-04 09:04:00+09:00,23400
              2018-01-04 09:05:00+09:00,23400"""
    result = parser.read_csv(StringIO(data), parse_dates=["dt"])

    dti = date_range(
        start="2018-01-04 09:01:00",
        end="2018-01-04 09:05:00",
        freq="1min",
        tz=timezone(timedelta(minutes=540)),
    )._with_freq(None)
    expected_data = {"dt": dti, "val": [23350, 23400, 23400, 23400, 23400]}

    expected = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # pandas.errors.ParserError: CSV parse error
@pytest.mark.parametrize(
    "date_string",
    ["32/32/2019", "02/30/2019", "13/13/2019", "13/2019", "a3/11/2018", "10/11/2o17"],
)
def test_invalid_parse_delimited_date(all_parsers, date_string):
    parser = all_parsers
    expected = DataFrame({0: [date_string]}, dtype="str")
    result = parser.read_csv(
        StringIO(date_string),
        header=None,
        parse_dates=[0],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "date_string,dayfirst,expected",
    [
        # %d/%m/%Y; month > 12 thus replacement
        ("13/02/2019", True, datetime(2019, 2, 13)),
        # %m/%d/%Y; day > 12 thus there will be no replacement
        ("02/13/2019", False, datetime(2019, 2, 13)),
        # %d/%m/%Y; dayfirst==True thus replacement
        ("04/02/2019", True, datetime(2019, 2, 4)),
    ],
)
def test_parse_delimited_date_swap_no_warning(
    all_parsers, date_string, dayfirst, expected, request
):
    parser = all_parsers
    expected = DataFrame({0: [expected]}, dtype="datetime64[ns]")
    if parser.engine == "pyarrow":
        if not dayfirst:
            # "CSV parse error: Empty CSV file or block"
            pytest.skip(reason="https://github.com/apache/arrow/issues/38676")
        msg = "The 'dayfirst' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0]
            )
        return

    result = parser.read_csv(
        StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0]
    )
    tm.assert_frame_equal(result, expected)


# ArrowInvalid: CSV parse error: Empty CSV file or block: cannot infer number of columns
@skip_pyarrow
@pytest.mark.parametrize(
    "date_string,dayfirst,expected",
    [
        # %d/%m/%Y; month > 12
        ("13/02/2019", False, datetime(2019, 2, 13)),
        # %m/%d/%Y; day > 12
        ("02/13/2019", True, datetime(2019, 2, 13)),
    ],
)
def test_parse_delimited_date_swap_with_warning(
    all_parsers, date_string, dayfirst, expected
):
    parser = all_parsers
    expected = DataFrame({0: [expected]}, dtype="datetime64[ns]")
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )
    result = parser.read_csv_check_warnings(
        UserWarning,
        warning_msg,
        StringIO(date_string),
        header=None,
        dayfirst=dayfirst,
        parse_dates=[0],
    )
    tm.assert_frame_equal(result, expected)


def test_parse_multiple_delimited_dates_with_swap_warnings():
    # GH46210
    with pytest.raises(
        ValueError,
        match=(
            r'^time data "31/05/2000" doesn\'t match format "%m/%d/%Y", '
            r"at position 1. You might want to try:"
        ),
    ):
        pd.to_datetime(["01/01/2000", "31/05/2000", "31/05/2001", "01/02/2000"])


# ArrowKeyError: Column 'fdate1' in include_columns does not exist in CSV file
@skip_pyarrow
@pytest.mark.parametrize(
    "names, usecols, parse_dates, missing_cols",
    [
        (None, ["val"], ["date", "time"], "date, time"),
        (None, ["val"], [0, "time"], "time"),
        (None, ["val"], [["date", "time"]], "date, time"),
        (None, ["val"], [[0, "time"]], "time"),
        (None, ["val"], {"date": [0, "time"]}, "time"),
        (None, ["val"], {"date": ["date", "time"]}, "date, time"),
        (None, ["val"], [["date", "time"], "date"], "date, time"),
        (["date1", "time1", "temperature"], None, ["date", "time"], "date, time"),
        (
            ["date1", "time1", "temperature"],
            ["date1", "temperature"],
            ["date1", "time"],
            "time",
        ),
    ],
)
def test_missing_parse_dates_column_raises(
    all_parsers, names, usecols, parse_dates, missing_cols
):
    # gh-31251 column names provided in parse_dates could be missing.
    parser = all_parsers
    content = StringIO("date,time,val\n2020-01-31,04:20:32,32\n")
    msg = f"Missing column provided to 'parse_dates': '{missing_cols}'"

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    warn = FutureWarning
    if isinstance(parse_dates, list) and all(
        isinstance(x, (int, str)) for x in parse_dates
    ):
        warn = None

    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(warn, match=depr_msg, check_stacklevel=False):
            parser.read_csv(
                content, sep=",", names=names, usecols=usecols, parse_dates=parse_dates
            )


@xfail_pyarrow  # mismatched shape
def test_date_parser_and_names(all_parsers):
    # GH#33699
    parser = all_parsers
    data = StringIO("""x,y\n1,2""")
    warn = UserWarning
    if parser.engine == "pyarrow":
        # DeprecationWarning for passing a Manager object
        warn = (UserWarning, DeprecationWarning)
    result = parser.read_csv_check_warnings(
        warn,
        "Could not infer format",
        data,
        parse_dates=["B"],
        names=["B"],
    )
    expected = DataFrame({"B": ["y", "2"]}, index=["x", "1"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_date_parser_multiindex_columns(all_parsers):
    parser = all_parsers
    data = """a,b
1,2
2019-12-31,6"""
    result = parser.read_csv(StringIO(data), parse_dates=[("a", "1")], header=[0, 1])
    expected = DataFrame(
        {("a", "1"): Timestamp("2019-12-31").as_unit("ns"), ("b", "2"): [6]}
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
@pytest.mark.parametrize(
    "parse_spec, col_name",
    [
        ([[("a", "1"), ("b", "2")]], ("a_b", "1_2")),
        ({("foo", "1"): [("a", "1"), ("b", "2")]}, ("foo", "1")),
    ],
)
def test_date_parser_multiindex_columns_combine_cols(all_parsers, parse_spec, col_name):
    parser = all_parsers
    data = """a,b,c
1,2,3
2019-12,-31,6"""

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data),
            parse_dates=parse_spec,
            header=[0, 1],
        )
    expected = DataFrame(
        {col_name: Timestamp("2019-12-31").as_unit("ns"), ("c", "3"): [6]}
    )
    tm.assert_frame_equal(result, expected)


def test_date_parser_usecols_thousands(all_parsers):
    # GH#39365
    data = """A,B,C
    1,3,20-09-01-01
    2,4,20-09-01-01
    """

    parser = all_parsers

    if parser.engine == "pyarrow":
        # DeprecationWarning for passing a Manager object
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                parse_dates=[1],
                usecols=[1, 2],
                thousands="-",
            )
        return

    result = parser.read_csv_check_warnings(
        UserWarning,
        "Could not infer format",
        StringIO(data),
        parse_dates=[1],
        usecols=[1, 2],
        thousands="-",
    )
    expected = DataFrame({"B": [3, 4], "C": [Timestamp("20-09-2001 01:00:00")] * 2})
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # mismatched shape
def test_parse_dates_and_keep_original_column(all_parsers):
    # GH#13378
    parser = all_parsers
    data = """A
20150908
20150909
"""
    depr_msg = "The 'keep_date_col' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(
        FutureWarning, match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), parse_dates={"date": ["A"]}, keep_date_col=True
        )
    expected_data = [Timestamp("2015-09-08"), Timestamp("2015-09-09")]
    expected = DataFrame({"date": expected_data, "A": expected_data})
    tm.assert_frame_equal(result, expected)


def test_dayfirst_warnings():
    # GH 12585

    # CASE 1: valid input
    input = "date\n31/12/2014\n10/03/2011"
    expected = DatetimeIndex(
        ["2014-12-31", "2011-03-10"], dtype="datetime64[ns]", freq=None, name="date"
    )
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )

    # A. dayfirst arg correct, no warning
    res1 = read_csv(
        StringIO(input), parse_dates=["date"], dayfirst=True, index_col="date"
    ).index
    tm.assert_index_equal(expected, res1)

    # B. dayfirst arg incorrect, warning
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res2 = read_csv(
            StringIO(input), parse_dates=["date"], dayfirst=False, index_col="date"
        ).index
    tm.assert_index_equal(expected, res2)

    # CASE 2: invalid input
    # cannot consistently process with single format
    # return to user unaltered

    # first in DD/MM/YYYY, second in MM/DD/YYYY
    input = "date\n31/12/2014\n03/30/2011"
    expected = Index(["31/12/2014", "03/30/2011"], dtype="str", name="date")

    # A. use dayfirst=True
    res5 = read_csv(
        StringIO(input), parse_dates=["date"], dayfirst=True, index_col="date"
    ).index
    tm.assert_index_equal(expected, res5)

    # B. use dayfirst=False
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res6 = read_csv(
            StringIO(input), parse_dates=["date"], dayfirst=False, index_col="date"
        ).index
    tm.assert_index_equal(expected, res6)


@pytest.mark.parametrize(
    "date_string, dayfirst",
    [
        pytest.param(
            "31/1/2014",
            False,
            id="second date is single-digit",
        ),
        pytest.param(
            "1/31/2014",
            True,
            id="first date is single-digit",
        ),
    ],
)
def test_dayfirst_warnings_no_leading_zero(date_string, dayfirst):
    # GH47880
    initial_value = f"date\n{date_string}"
    expected = DatetimeIndex(
        ["2014-01-31"], dtype="datetime64[ns]", freq=None, name="date"
    )
    warning_msg = (
        "Parsing dates in .* format when dayfirst=.* was specified. "
        "Pass `dayfirst=.*` or specify a format to silence this warning."
    )
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res = read_csv(
            StringIO(initial_value),
            parse_dates=["date"],
            index_col="date",
            dayfirst=dayfirst,
        ).index
    tm.assert_index_equal(expected, res)


@skip_pyarrow  # CSV parse error: Expected 3 columns, got 4
def test_infer_first_column_as_index(all_parsers):
    # GH#11019
    parser = all_parsers
    data = "a,b,c\n1970-01-01,2,3,4"
    result = parser.read_csv(
        StringIO(data),
        parse_dates=["a"],
    )
    expected = DataFrame({"a": "2", "b": 3, "c": 4}, index=["1970-01-01"])
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # pyarrow engine doesn't support passing a dict for na_values
@pytest.mark.parametrize(
    ("key", "value", "warn"),
    [
        ("date_parser", lambda x: pd.to_datetime(x, format="%Y-%m-%d"), FutureWarning),
        ("date_format", "%Y-%m-%d", None),
    ],
)
def test_replace_nans_before_parsing_dates(all_parsers, key, value, warn):
    # GH#26203
    parser = all_parsers
    data = """Test
2012-10-01
0
2015-05-15
#
2017-09-09
"""
    result = parser.read_csv_check_warnings(
        warn,
        "use 'date_format' instead",
        StringIO(data),
        na_values={"Test": ["#", "0"]},
        parse_dates=["Test"],
        **{key: value},
    )
    expected = DataFrame(
        {
            "Test": [
                Timestamp("2012-10-01"),
                pd.NaT,
                Timestamp("2015-05-15"),
                pd.NaT,
                Timestamp("2017-09-09"),
            ]
        }
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # string[python] instead of dt64[ns]
def test_parse_dates_and_string_dtype(all_parsers):
    # GH#34066
    parser = all_parsers
    data = """a,b
1,2019-12-31
"""
    result = parser.read_csv(StringIO(data), dtype="string", parse_dates=["b"])
    expected = DataFrame({"a": ["1"], "b": [Timestamp("2019-12-31")]})
    expected["a"] = expected["a"].astype("string")
    tm.assert_frame_equal(result, expected)


def test_parse_dot_separated_dates(all_parsers):
    # https://github.com/pandas-dev/pandas/issues/2586
    parser = all_parsers
    data = """a,b
27.03.2003 14:55:00.000,1
03.08.2003 15:20:00.000,2"""
    if parser.engine == "pyarrow":
        expected_index = Index(
            ["27.03.2003 14:55:00.000", "03.08.2003 15:20:00.000"],
            dtype="str",
            name="a",
        )
        warn = None
    else:
        expected_index = DatetimeIndex(
            ["2003-03-27 14:55:00", "2003-08-03 15:20:00"],
            dtype="datetime64[ns]",
            name="a",
        )
        warn = UserWarning
    msg = r"when dayfirst=False \(the default\) was specified"
    result = parser.read_csv_check_warnings(
        warn,
        msg,
        StringIO(data),
        parse_dates=True,
        index_col=0,
        raise_on_extra_warnings=False,
    )
    expected = DataFrame({"b": [1, 2]}, index=expected_index)
    tm.assert_frame_equal(result, expected)


def test_parse_dates_dict_format(all_parsers):
    # GH#51240
    parser = all_parsers
    data = """a,b
2019-12-31,31-12-2019
2020-12-31,31-12-2020"""

    result = parser.read_csv(
        StringIO(data),
        date_format={"a": "%Y-%m-%d", "b": "%d-%m-%Y"},
        parse_dates=["a", "b"],
    )
    expected = DataFrame(
        {
            "a": [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
            "b": [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "key, parse_dates", [("a_b", [[0, 1]]), ("foo", {"foo": [0, 1]})]
)
def test_parse_dates_dict_format_two_columns(all_parsers, key, parse_dates):
    # GH#51240
    parser = all_parsers
    data = """a,b
31-,12-2019
31-,12-2020"""

    depr_msg = (
        "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    )
    with tm.assert_produces_warning(
        (FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), date_format={key: "%d- %m-%Y"}, parse_dates=parse_dates
        )
    expected = DataFrame(
        {
            key: [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
        }
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # object dtype index
def test_parse_dates_dict_format_index(all_parsers):
    # GH#51240
    parser = all_parsers
    data = """a,b
2019-12-31,31-12-2019
2020-12-31,31-12-2020"""

    result = parser.read_csv(
        StringIO(data), date_format={"a": "%Y-%m-%d"}, parse_dates=True, index_col=0
    )
    expected = DataFrame(
        {
            "b": ["31-12-2019", "31-12-2020"],
        },
        index=Index([Timestamp("2019-12-31"), Timestamp("2020-12-31")], name="a"),
    )
    tm.assert_frame_equal(result, expected)


def test_parse_dates_arrow_engine(all_parsers):
    # GH#53295
    parser = all_parsers
    data = """a,b
2000-01-01 00:00:00,1
2000-01-01 00:00:01,1"""

    result = parser.read_csv(StringIO(data), parse_dates=["a"])
    # TODO: make unit check more specific
    if parser.engine == "pyarrow":
        result["a"] = result["a"].dt.as_unit("ns")
    expected = DataFrame(
        {
            "a": [
                Timestamp("2000-01-01 00:00:00"),
                Timestamp("2000-01-01 00:00:01"),
            ],
            "b": 1,
        }
    )
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # object dtype index
def test_from_csv_with_mixed_offsets(all_parsers):
    parser = all_parsers
    data = "a\n2020-01-01T00:00:00+01:00\n2020-01-01T00:00:00+00:00"
    result = parser.read_csv(StringIO(data), parse_dates=["a"])["a"]
    expected = Series(
        [
            Timestamp("2020-01-01 00:00:00+01:00"),
            Timestamp("2020-01-01 00:00:00+00:00"),
        ],
        name="a",
        index=[0, 1],
    )
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\filters.py ===
# SPDX-License-Identifier: MIT

"""
Commonly useful filters for `attrs.asdict` and `attrs.astuple`.
"""

from ._make import Attribute


def _split_what(what):
    """
    Returns a tuple of `frozenset`s of classes and attributes.
    """
    return (
        frozenset(cls for cls in what if isinstance(cls, type)),
        frozenset(cls for cls in what if isinstance(cls, str)),
        frozenset(cls for cls in what if isinstance(cls, Attribute)),
    )


def include(*what):
    """
    Create a filter that only allows *what*.

    Args:
        what (list[type, str, attrs.Attribute]):
            What to include. Can be a type, a name, or an attribute.

    Returns:
        Callable:
            A callable that can be passed to `attrs.asdict`'s and
            `attrs.astuple`'s *filter* argument.

    .. versionchanged:: 23.1.0 Accept strings with field names.
    """
    cls, names, attrs = _split_what(what)

    def include_(attribute, value):
        return (
            value.__class__ in cls
            or attribute.name in names
            or attribute in attrs
        )

    return include_


def exclude(*what):
    """
    Create a filter that does **not** allow *what*.

    Args:
        what (list[type, str, attrs.Attribute]):
            What to exclude. Can be a type, a name, or an attribute.

    Returns:
        Callable:
            A callable that can be passed to `attrs.asdict`'s and
            `attrs.astuple`'s *filter* argument.

    .. versionchanged:: 23.3.0 Accept field name string as input argument
    """
    cls, names, attrs = _split_what(what)

    def exclude_(attribute, value):
        return not (
            value.__class__ in cls
            or attribute.name in names
            or attribute in attrs
        )

    return exclude_

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\binary_op.py ===
import onnx
from onnx import onnx_pb as onnx_proto  # noqa: F401

from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType, attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase


class QLinearBinaryOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        (
            data_found,
            output_scale_name,
            output_zp_name,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])
        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0, 1])
        if not data_found or quantized_input_names is None:
            return super().quantize()

        qlinear_binary_math_output = node.output[0] + TENSOR_NAME_QUANT_SUFFIX
        qlinear_binary_math_name = node.name + "_quant" if node.name else ""

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlinear_binary_math_inputs = []
        # Input 0
        qlinear_binary_math_inputs.append(quantized_input_names[0])
        qlinear_binary_math_inputs.append(scale_names[0])
        qlinear_binary_math_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_binary_math_inputs.append(quantized_input_names[1])
        qlinear_binary_math_inputs.append(scale_names[1])
        qlinear_binary_math_inputs.append(zero_point_names[1])

        # Output
        qlinear_binary_math_inputs.append(output_scale_name)
        qlinear_binary_math_inputs.append(output_zp_name)

        qlinear_binary_math_node = onnx.helper.make_node(
            "QLinear" + node.op_type,
            qlinear_binary_math_inputs,
            [qlinear_binary_math_output],
            qlinear_binary_math_name,
            **kwargs,
        )
        nodes.append(qlinear_binary_math_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(
            node.output[0],
            qlinear_binary_math_output,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\DeprecatedSubGraphSessionState.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

# deprecated: no longer using kernel def hashes
class DeprecatedSubGraphSessionState(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DeprecatedSubGraphSessionState()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDeprecatedSubGraphSessionState(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DeprecatedSubGraphSessionStateBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # DeprecatedSubGraphSessionState
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DeprecatedSubGraphSessionState
    def GraphId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # DeprecatedSubGraphSessionState
    def SessionState(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.DeprecatedSessionState import DeprecatedSessionState
            obj = DeprecatedSessionState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def DeprecatedSubGraphSessionStateStart(builder):
    builder.StartObject(2)

def Start(builder):
    DeprecatedSubGraphSessionStateStart(builder)

def DeprecatedSubGraphSessionStateAddGraphId(builder, graphId):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(graphId), 0)

def AddGraphId(builder, graphId):
    DeprecatedSubGraphSessionStateAddGraphId(builder, graphId)

def DeprecatedSubGraphSessionStateAddSessionState(builder, sessionState):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(sessionState), 0)

def AddSessionState(builder, sessionState):
    DeprecatedSubGraphSessionStateAddSessionState(builder, sessionState)

def DeprecatedSubGraphSessionStateEnd(builder):
    return builder.EndObject()

def End(builder):
    return DeprecatedSubGraphSessionStateEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\spss.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.inference import is_list_like

from pandas.io.common import stringify_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pandas._typing import DtypeBackend

    from pandas import DataFrame


def read_spss(
    path: str | Path,
    usecols: Sequence[str] | None = None,
    convert_categoricals: bool = True,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame:
    """
    Load an SPSS file from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str or Path
        File path.
    usecols : list-like, optional
        Return a subset of the columns. If None, return all columns.
    convert_categoricals : bool, default is True
        Convert categorical columns into pd.Categorical.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> df = pd.read_spss("spss_data.sav")  # doctest: +SKIP
    """
    pyreadstat = import_optional_dependency("pyreadstat")
    check_dtype_backend(dtype_backend)

    if usecols is not None:
        if not is_list_like(usecols):
            raise TypeError("usecols must be list-like.")
        usecols = list(usecols)  # pyreadstat requires a list

    df, metadata = pyreadstat.read_sav(
        stringify_path(path), usecols=usecols, apply_value_formats=convert_categoricals
    )
    df.attrs = metadata.__dict__
    if dtype_backend is not lib.no_default:
        df = df.convert_dtypes(dtype_backend=dtype_backend)
    return df

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\plugin.py ===
"""
    pygments.plugin
    ~~~~~~~~~~~~~~~

    Pygments plugin interface.

    lexer plugins::

        [pygments.lexers]
        yourlexer = yourmodule:YourLexer

    formatter plugins::

        [pygments.formatters]
        yourformatter = yourformatter:YourFormatter
        /.ext = yourformatter:YourFormatter

    As you can see, you can define extensions for the formatter
    with a leading slash.

    syntax plugins::

        [pygments.styles]
        yourstyle = yourstyle:YourStyle

    filter plugin::

        [pygments.filter]
        yourfilter = yourfilter:YourFilter


    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
from importlib.metadata import entry_points

LEXER_ENTRY_POINT = 'pygments.lexers'
FORMATTER_ENTRY_POINT = 'pygments.formatters'
STYLE_ENTRY_POINT = 'pygments.styles'
FILTER_ENTRY_POINT = 'pygments.filters'


def iter_entry_points(group_name):
    groups = entry_points()
    if hasattr(groups, 'select'):
        # New interface in Python 3.10 and newer versions of the
        # importlib_metadata backport.
        return groups.select(group=group_name)
    else:
        # Older interface, deprecated in Python 3.10 and recent
        # importlib_metadata, but we need it in Python 3.8 and 3.9.
        return groups.get(group_name, [])


def find_plugin_lexers():
    for entrypoint in iter_entry_points(LEXER_ENTRY_POINT):
        yield entrypoint.load()


def find_plugin_formatters():
    for entrypoint in iter_entry_points(FORMATTER_ENTRY_POINT):
        yield entrypoint.name, entrypoint.load()


def find_plugin_styles():
    for entrypoint in iter_entry_points(STYLE_ENTRY_POINT):
        yield entrypoint.name, entrypoint.load()


def find_plugin_filters():
    for entrypoint in iter_entry_points(FILTER_ENTRY_POINT):
        yield entrypoint.name, entrypoint.load()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_numbers.py ===
import numbers as nums
import decimal
from sympy.concrete.summations import Sum
from sympy.core import (EulerGamma, Catalan, TribonacciConstant,
    GoldenRatio)
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import (mpf_norm, seterr,
    Integer, I, pi, comp, Rational, E, nan,
    oo, AlgebraicNumber, Number, Float, zoo, equal_valued,
    int_valued, all_close)
from sympy.core.intfunc import (igcd, igcdex, igcd2, igcd_lehmer,
    ilcm, integer_nthroot, isqrt, integer_log, mod_inverse)
from sympy.core.power import Pow
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.integers import floor
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.polys.domains.realfield import RealField
from sympy.printing.latex import latex
from sympy.printing.repr import srepr
from sympy.simplify import simplify
from sympy.polys.domains.groundtypes import PythonRational
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import (XFAIL, raises, _both_exp_pow,
                                  warns_deprecated_sympy)
from sympy import Add

from mpmath import mpf
import mpmath
from sympy.core import numbers
t = Symbol('t', real=False)

_ninf = float(-oo)
_inf = float(oo)


def same_and_same_prec(a, b):
    # stricter matching for Floats
    return a == b and a._prec == b._prec


def test_seterr():
    seterr(divide=True)
    raises(ValueError, lambda: S.Zero/S.Zero)
    seterr(divide=False)
    assert S.Zero / S.Zero is S.NaN


def test_mod():
    x = S.Half
    y = Rational(3, 4)
    z = Rational(5, 18043)

    assert x % x == 0
    assert x % y == S.Half
    assert x % z == Rational(3, 36086)
    assert y % x == Rational(1, 4)
    assert y % y == 0
    assert y % z == Rational(9, 72172)
    assert z % x == Rational(5, 18043)
    assert z % y == Rational(5, 18043)
    assert z % z == 0

    a = Float(2.6)

    assert (a % .2) == 0.0
    assert (a % 2).round(15) == 0.6
    assert (a % 0.5).round(15) == 0.1

    p = Symbol('p', infinite=True)

    assert oo % oo is nan
    assert zoo % oo is nan
    assert 5 % oo is nan
    assert p % 5 is nan

    # In these two tests, if the precision of m does
    # not match the precision of the ans, then it is
    # likely that the change made now gives an answer
    # with degraded accuracy.
    r = Rational(500, 41)
    f = Float('.36', 3)
    m = r % f
    ans = Float(r % Rational(f), 3)
    assert m == ans and m._prec == ans._prec
    f = Float('8.36', 3)
    m = f % r
    ans = Float(Rational(f) % r, 3)
    assert m == ans and m._prec == ans._prec

    s = S.Zero

    assert s % float(1) == 0.0

    # No rounding required since these numbers can be represented
    # exactly.
    assert Rational(3, 4) % Float(1.1) == 0.75
    assert Float(1.5) % Rational(5, 4) == 0.25
    assert Rational(5, 4).__rmod__(Float('1.5')) == 0.25
    assert Float('1.5').__rmod__(Float('2.75')) == Float('1.25')
    assert 2.75 % Float('1.5') == Float('1.25')

    a = Integer(7)
    b = Integer(4)

    assert type(a % b) == Integer
    assert a % b == Integer(3)
    assert Integer(1) % Rational(2, 3) == Rational(1, 3)
    assert Rational(7, 5) % Integer(1) == Rational(2, 5)
    assert Integer(2) % 1.5 == 0.5

    assert Integer(3).__rmod__(Integer(10)) == Integer(1)
    assert Integer(10) % 4 == Integer(2)
    assert 15 % Integer(4) == Integer(3)


def test_divmod():
    x = Symbol("x")
    assert divmod(S(12), S(8)) == Tuple(1, 4)
    assert divmod(-S(12), S(8)) == Tuple(-2, 4)
    assert divmod(S.Zero, S.One) == Tuple(0, 0)
    raises(ZeroDivisionError, lambda: divmod(S.Zero, S.Zero))
    raises(ZeroDivisionError, lambda: divmod(S.One, S.Zero))
    assert divmod(S(12), 8) == Tuple(1, 4)
    assert divmod(12, S(8)) == Tuple(1, 4)
    assert S(1024)//x == 1024//x == floor(1024/x)

    assert divmod(S("2"), S("3/2")) == Tuple(S("1"), S("1/2"))
    assert divmod(S("3/2"), S("2")) == Tuple(S("0"), S("3/2"))
    assert divmod(S("2"), S("3.5")) == Tuple(S("0"), S("2."))
    assert divmod(S("3.5"), S("2")) == Tuple(S("1"), S("1.5"))
    assert divmod(S("2"), S("1/3")) == Tuple(S("6"), S("0"))
    assert divmod(S("1/3"), S("2")) == Tuple(S("0"), S("1/3"))
    assert divmod(S("2"), S("1/10")) == Tuple(S("20"), S("0"))
    assert divmod(S("2"), S(".1"))[0] == 19
    assert divmod(S("0.1"), S("2")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("2"), 2) == Tuple(S("1"), S("0"))
    assert divmod(2, S("2")) == Tuple(S("1"), S("0"))
    assert divmod(S("2"), 1.5) == Tuple(S("1"), S("0.5"))
    assert divmod(1.5, S("2")) == Tuple(S("0"), S("1.5"))
    assert divmod(0.3, S("2")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("3/2"), S("3.5")) == Tuple(S("0"), S(3/2))
    assert divmod(S("3.5"), S("3/2")) == Tuple(S("2"), S("0.5"))
    assert divmod(S("3/2"), S("1/3")) == Tuple(S("4"), S("1/6"))
    assert divmod(S("1/3"), S("3/2")) == Tuple(S("0"), S("1/3"))
    assert divmod(S("3/2"), S("0.1"))[0] == 14
    assert divmod(S("0.1"), S("3/2")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("3/2"), 2) == Tuple(S("0"), S("3/2"))
    assert divmod(2, S("3/2")) == Tuple(S("1"), S("1/2"))
    assert divmod(S("3/2"), 1.5) == Tuple(S("1"), S("0."))
    assert divmod(1.5, S("3/2")) == Tuple(S("1"), S("0."))
    assert divmod(S("3/2"), 0.3) == Tuple(S("5"), S("0."))
    assert divmod(0.3, S("3/2")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("1/3"), S("3.5")) == (0, 1/3)
    assert divmod(S("3.5"), S("0.1")) == Tuple(S("35"), S("0."))
    assert divmod(S("0.1"), S("3.5")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("3.5"), 2) == Tuple(S("1"), S("1.5"))
    assert divmod(2, S("3.5")) == Tuple(S("0"), S("2."))
    assert divmod(S("3.5"), 1.5) == Tuple(S("2"), S("0.5"))
    assert divmod(1.5, S("3.5")) == Tuple(S("0"), S("1.5"))
    assert divmod(0.3, S("3.5")) == Tuple(S("0"), S("0.3"))
    assert divmod(S("0.1"), S("1/3")) == Tuple(S("0"), S("0.1"))
    assert divmod(S("1/3"), 2) == Tuple(S("0"), S("1/3"))
    assert divmod(2, S("1/3")) == Tuple(S("6"), S("0"))
    assert divmod(S("1/3"), 1.5) == (0, 1/3)
    assert divmod(0.3, S("1/3")) == (0, 0.3)
    assert divmod(S("0.1"), 2) == (0, 0.1)
    assert divmod(2, S("0.1"))[0] == 19
    assert divmod(S("0.1"), 1.5) == (0, 0.1)
    assert divmod(1.5, S("0.1")) == Tuple(S("15"), S("0."))
    assert divmod(S("0.1"), 0.3) == Tuple(S("0"), S("0.1"))

    assert str(divmod(S("2"), 0.3)) == '(6, 0.2)'
    assert str(divmod(S("3.5"), S("1/3"))) == '(10, 0.166666666666667)'
    assert str(divmod(S("3.5"), 0.3)) == '(11, 0.2)'
    assert str(divmod(S("1/3"), S("0.1"))) == '(3, 0.0333333333333333)'
    assert str(divmod(1.5, S("1/3"))) == '(4, 0.166666666666667)'
    assert str(divmod(S("1/3"), 0.3)) == '(1, 0.0333333333333333)'
    assert str(divmod(0.3, S("0.1"))) == '(2, 0.1)'

    assert divmod(-3, S(2)) == (-2, 1)
    assert divmod(S(-3), S(2)) == (-2, 1)
    assert divmod(S(-3), 2) == (-2, 1)

    assert divmod(oo, 1) == (S.NaN, S.NaN)
    assert divmod(S.NaN, 1) == (S.NaN, S.NaN)
    assert divmod(1, S.NaN) == (S.NaN, S.NaN)
    ans = [(-1, oo), (-1, oo), (0, 0), (0, 1), (0, 2)]
    OO = float('inf')
    ANS = [tuple(map(float, i)) for i in ans]
    assert [divmod(i, oo) for i in range(-2, 3)] == ans
    ans = [(0, -2), (0, -1), (0, 0), (-1, -oo), (-1, -oo)]
    ANS = [tuple(map(float, i)) for i in ans]
    assert [divmod(i, -oo) for i in range(-2, 3)] == ans
    assert [divmod(i, -OO) for i in range(-2, 3)] == ANS

    # sympy's divmod gives an Integer for the quotient rather than a float
    dmod = lambda a, b: tuple([j if i else int(j) for i, j in enumerate(divmod(a, b))])
    for a in (4, 4., 4.25, 0, 0., -4, -4. -4.25):
        for b in (2, 2., 2.5, -2, -2., -2.5):
            assert divmod(S(a), S(b)) == dmod(a, b)


def test_igcd():
    assert igcd(0, 0) == 0
    assert igcd(0, 1) == 1
    assert igcd(1, 0) == 1
    assert igcd(0, 7) == 7
    assert igcd(7, 0) == 7
    assert igcd(7, 1) == 1
    assert igcd(1, 7) == 1
    assert igcd(-1, 0) == 1
    assert igcd(0, -1) == 1
    assert igcd(-1, -1) == 1
    assert igcd(-1, 7) == 1
    assert igcd(7, -1) == 1
    assert igcd(8, 2) == 2
    assert igcd(4, 8) == 4
    assert igcd(8, 16) == 8
    assert igcd(7, -3) == 1
    assert igcd(-7, 3) == 1
    assert igcd(-7, -3) == 1
    assert igcd(*[10, 20, 30]) == 10
    raises(TypeError, lambda: igcd())
    raises(TypeError, lambda: igcd(2))
    raises(ValueError, lambda: igcd(0, None))
    raises(ValueError, lambda: igcd(1, 2.2))
    for args in permutations((45.1, 1, 30)):
        raises(ValueError, lambda: igcd(*args))
    for args in permutations((1, 2, None)):
        raises(ValueError, lambda: igcd(*args))


def test_igcd_lehmer():
    a, b = fibonacci(10001), fibonacci(10000)
    # len(str(a)) == 2090
    # small divisors, long Euclidean sequence
    assert igcd_lehmer(a, b) == 1
    c = fibonacci(100)
    assert igcd_lehmer(a*c, b*c) == c
    # big divisor
    assert igcd_lehmer(a, 10**1000) == 1
    # swapping argument
    assert igcd_lehmer(1, 2) == igcd_lehmer(2, 1)


def test_igcd2():
    # short loop
    assert igcd2(2**100 - 1, 2**99 - 1) == 1
    # Lehmer's algorithm
    a, b = int(fibonacci(10001)), int(fibonacci(10000))
    assert igcd2(a, b) == 1


def test_ilcm():
    assert ilcm(0, 0) == 0
    assert ilcm(1, 0) == 0
    assert ilcm(0, 1) == 0
    assert ilcm(1, 1) == 1
    assert ilcm(2, 1) == 2
    assert ilcm(8, 2) == 8
    assert ilcm(8, 6) == 24
    assert ilcm(8, 7) == 56
    assert ilcm(*[10, 20, 30]) == 60
    raises(ValueError, lambda: ilcm(8.1, 7))
    raises(ValueError, lambda: ilcm(8, 7.1))
    raises(TypeError, lambda: ilcm(8))


def test_igcdex():
    assert igcdex(2, 3) == (-1, 1, 1)
    assert igcdex(10, 12) == (-1, 1, 2)
    assert igcdex(100, 2004) == (-20, 1, 4)
    assert igcdex(0, 0) == (0, 0, 0)
    assert igcdex(1, 0) == (1, 0, 1)


def _strictly_equal(a, b):
    return (a.p, a.q, type(a.p), type(a.q)) == \
           (b.p, b.q, type(b.p), type(b.q))


def _test_rational_new(cls):
    """
    Tests that are common between Integer and Rational.
    """
    assert cls(0) is S.Zero
    assert cls(1) is S.One
    assert cls(-1) is S.NegativeOne
    # These look odd, but are similar to int():
    assert cls('1') is S.One
    assert cls('-1') is S.NegativeOne

    i = Integer(10)
    assert _strictly_equal(i, cls('10'))
    assert _strictly_equal(i, cls('10'))
    assert _strictly_equal(i, cls(int(10)))
    assert _strictly_equal(i, cls(i))

    raises(TypeError, lambda: cls(Symbol('x')))


def test_Integer_new():
    """
    Test for Integer constructor
    """
    _test_rational_new(Integer)

    assert _strictly_equal(Integer(0.9), S.Zero)
    assert _strictly_equal(Integer(10.5), Integer(10))
    raises(ValueError, lambda: Integer("10.5"))
    assert Integer(Rational('1.' + '9'*20)) == 1


def test_Rational_new():
    """"
    Test for Rational constructor
    """
    _test_rational_new(Rational)

    n1 = S.Half
    assert n1 == Rational(Integer(1), 2)
    assert n1 == Rational(Integer(1), Integer(2))
    assert n1 == Rational(1, Integer(2))
    assert n1 == Rational(S.Half)
    assert 1 == Rational(n1, n1)
    assert Rational(3, 2) == Rational(S.Half, Rational(1, 3))
    assert Rational(3, 1) == Rational(1, Rational(1, 3))
    n3_4 = Rational(3, 4)
    assert Rational('3/4') == n3_4
    assert -Rational('-3/4') == n3_4
    assert Rational('.76').limit_denominator(4) == n3_4
    assert Rational(19, 25).limit_denominator(4) == n3_4
    assert Rational('19/25').limit_denominator(4) == n3_4
    assert Rational(1.0, 3) == Rational(1, 3)
    assert Rational(1, 3.0) == Rational(1, 3)
    assert Rational(Float(0.5)) == S.Half
    assert Rational('1e2/1e-2') == Rational(10000)
    assert Rational('1 234') == Rational(1234)
    assert Rational('1/1 234') == Rational(1, 1234)
    assert Rational(-1, 0) is S.ComplexInfinity
    assert Rational(1, 0) is S.ComplexInfinity
    # Make sure Rational doesn't lose precision on Floats
    assert Rational(pi.evalf(100)).evalf(100) == pi.evalf(100)
    raises(TypeError, lambda: Rational('3**3'))
    raises(TypeError, lambda: Rational('1/2 + 2/3'))

    # handle fractions.Fraction instances
    try:
        import fractions
        assert Rational(fractions.Fraction(1, 2)) == S.Half
    except ImportError:
        pass

    assert Rational(PythonRational(2, 6)) == Rational(1, 3)

    with warns_deprecated_sympy():
        assert Rational(2, 4, gcd=1).q == 4
    with warns_deprecated_sympy():
        n = Rational(2, -4, gcd=1)
    assert n.q == 4
    assert n.p == -2

    assert Rational.from_coprime_ints(3, 5) == Rational(3, 5)


def test_issue_24543():
    for p in ('1.5', 1.5, 2):
        for q in ('1.5', 1.5, 2):
            assert Rational(p, q).as_numer_denom() == Rational('%s/%s'%(p,q)).as_numer_denom()

    assert Rational('0.5', '100') == Rational(1, 200)


def test_Number_new():
    """"
    Test for Number constructor
    """
    # Expected behavior on numbers and strings
    assert Number(1) is S.One
    assert Number(2).__class__ is Integer
    assert Number(-622).__class__ is Integer
    assert Number(5, 3).__class__ is Rational
    assert Number(5.3).__class__ is Float
    assert Number('1') is S.One
    assert Number('2').__class__ is Integer
    assert Number('-622').__class__ is Integer
    assert Number('5/3').__class__ is Rational
    assert Number('5.3').__class__ is Float
    raises(ValueError, lambda: Number('cos'))
    raises(TypeError, lambda: Number(cos))
    a = Rational(3, 5)
    assert Number(a) is a  # Check idempotence on Numbers
    u = ['inf', '-inf', 'nan', 'iNF', '+inf']
    v = [oo, -oo, nan, oo, oo]
    for i, a in zip(u, v):
        assert Number(i) is a, (i, Number(i), a)


def test_Number_cmp():
    n1 = Number(1)
    n2 = Number(2)
    n3 = Number(-3)

    assert n1 < n2
    assert n1 <= n2
    assert n3 < n1
    assert n2 > n3
    assert n2 >= n3

    raises(TypeError, lambda: n1 < S.NaN)
    raises(TypeError, lambda: n1 <= S.NaN)
    raises(TypeError, lambda: n1 > S.NaN)
    raises(TypeError, lambda: n1 >= S.NaN)


def test_Rational_cmp():
    n1 = Rational(1, 4)
    n2 = Rational(1, 3)
    n3 = Rational(2, 4)
    n4 = Rational(2, -4)
    n5 = Rational(0)
    n6 = Rational(1)
    n7 = Rational(3)
    n8 = Rational(-3)

    assert n8 < n5
    assert n5 < n6
    assert n6 < n7
    assert n8 < n7
    assert n7 > n8
    assert (n1 + 1)**n2 < 2
    assert ((n1 + n6)/n7) < 1

    assert n4 < n3
    assert n2 < n3
    assert n1 < n2
    assert n3 > n1
    assert not n3 < n1
    assert not (Rational(-1) > 0)
    assert Rational(-1) < 0

    raises(TypeError, lambda: n1 < S.NaN)
    raises(TypeError, lambda: n1 <= S.NaN)
    raises(TypeError, lambda: n1 > S.NaN)
    raises(TypeError, lambda: n1 >= S.NaN)


def test_Float():
    def eq(a, b):
        t = Float("1.0E-15")
        return (-t < a - b < t)

    equal_pairs = [
        (0, 0.0), # This is just how Python works...
        (0, S.Zero),
        (0.0, Float(0)),
    ]
    unequal_pairs = [
        (0.0, S.Zero),
        (0, Float(0)),
        (S.Zero, Float(0)),
    ]
    for p1, p2 in equal_pairs:
        assert (p1 == p2) is True
        assert (p1 != p2) is False
        assert (p2 == p1) is True
        assert (p2 != p1) is False
    for p1, p2 in unequal_pairs:
        assert (p1 == p2) is False
        assert (p1 != p2) is True
        assert (p2 == p1) is False
        assert (p2 != p1) is True

    assert S.Zero.is_zero

    a = Float(2) ** Float(3)
    assert eq(a.evalf(), Float(8))
    assert eq((pi ** -1).evalf(), Float("0.31830988618379067"))
    a = Float(2) ** Float(4)
    assert eq(a.evalf(), Float(16))
    assert (S(.3) == S(.5)) is False

    mpf = (0, 5404319552844595, -52, 53)
    x_str = Float((0, '13333333333333', -52, 53))
    x_0xstr = Float((0, '0x13333333333333', -52, 53))
    x2_str = Float((0, '26666666666666', -53, 54))
    x_hex = Float((0, int(0x13333333333333), -52, 53))
    x_dec = Float(mpf)
    assert x_str == x_0xstr == x_hex == x_dec == Float(1.2)
    # x2_str was entered slightly malformed in that the mantissa
    # was even -- it should be odd and the even part should be
    # included with the exponent, but this is resolved by normalization
    # ONLY IF REQUIREMENTS of mpf_norm are met: the bitcount must
    # be exact: double the mantissa ==> increase bc by 1
    assert Float(1.2)._mpf_ == mpf
    assert x2_str._mpf_ == mpf

    assert Float((0, int(0), -123, -1)) is S.NaN
    assert Float((0, int(0), -456, -2)) is S.Infinity
    assert Float((1, int(0), -789, -3)) is S.NegativeInfinity
    # if you don't give the full signature, it's not special
    assert Float((0, int(0), -123)) == Float(0)
    assert Float((0, int(0), -456)) == Float(0)
    assert Float((1, int(0), -789)) == Float(0)

    raises(ValueError, lambda: Float((0, 7, 1, 3), ''))

    assert Float('0.0').is_finite is True
    assert Float('0.0').is_negative is False
    assert Float('0.0').is_positive is False
    assert Float('0.0').is_infinite is False
    assert Float('0.0').is_zero is True

    # rationality properties
    # if the integer test fails then the use of intlike
    # should be removed from gamma_functions.py
    assert Float(1).is_integer is None
    assert Float(1).is_rational is None
    assert Float(1).is_irrational is None
    assert sqrt(2).n(15).is_rational is None
    assert sqrt(2).n(15).is_irrational is None

    # do not automatically evalf
    def teq(a):
        assert (a.evalf() == a) is False
        assert (a.evalf() != a) is True
        assert (a == a.evalf()) is False
        assert (a != a.evalf()) is True

    teq(pi)
    teq(2*pi)
    teq(cos(0.1, evaluate=False))

    # long integer
    i = 12345678901234567890
    assert same_and_same_prec(Float(12, ''), Float('12', ''))
    assert same_and_same_prec(Float(Integer(i), ''), Float(i, ''))
    assert same_and_same_prec(Float(i, ''), Float(str(i), 20))
    assert same_and_same_prec(Float(str(i)), Float(i, ''))
    assert same_and_same_prec(Float(i), Float(i, ''))

    # inexact floats (repeating binary = denom not multiple of 2)
    # cannot have precision greater than 15
    assert Float(.125, 22)._prec == 76
    assert Float(2.0, 22)._prec == 76
    # only default prec is equal, even for exactly representable float
    assert Float(.125, 22) != .125
    #assert Float(2.0, 22) == 2
    assert float(Float('.12500000000000001', '')) == .125
    raises(ValueError, lambda: Float(.12500000000000001, ''))

    # allow spaces
    assert Float('123 456.123 456') == Float('123456.123456')
    assert Integer('123 456') == Integer('123456')
    assert Rational('123 456.123 456') == Rational('123456.123456')
    assert Float(' .3e2') == Float('0.3e2')
    # but treat them as strictly ass underscore between digits: only 1
    raises(ValueError, lambda: Float('1  2'))

    # allow underscore between digits
    assert Float('1_23.4_56') == Float('123.456')
    # assert Float('1_23.4_5_6', 12) == Float('123.456', 12)
    # ...but not in all cases (per Py 3.6)
    raises(ValueError, lambda: Float('1_'))
    raises(ValueError, lambda: Float('1__2'))
    raises(ValueError, lambda: Float('_1'))
    raises(ValueError, lambda: Float('_inf'))

    # allow auto precision detection
    assert Float('.1', '') == Float(.1, 1)
    assert Float('.125', '') == Float(.125, 3)
    assert Float('.100', '') == Float(.1, 3)
    assert Float('2.0', '') == Float('2', 2)

    raises(ValueError, lambda: Float("12.3d-4", ""))
    raises(ValueError, lambda: Float(12.3, ""))
    raises(ValueError, lambda: Float('.'))
    raises(ValueError, lambda: Float('-.'))

    zero = Float('0.0')
    assert Float('-0') == zero
    assert Float('.0') == zero
    assert Float('-.0') == zero
    assert Float('-0.0') == zero
    assert Float(0.0) == zero
    assert Float(0) == zero
    assert Float(0, '') == Float('0', '')
    assert Float(1) == Float(1.0)
    assert Float(S.Zero) == zero
    assert Float(S.One) == Float(1.0)

    assert Float(decimal.Decimal('0.1'), 3) == Float('.1', 3)
    assert Float(decimal.Decimal('nan')) is S.NaN
    assert Float(decimal.Decimal('Infinity')) is S.Infinity
    assert Float(decimal.Decimal('-Infinity')) is S.NegativeInfinity

    assert '{:.3f}'.format(Float(4.236622)) == '4.237'
    assert '{:.35f}'.format(Float(pi.n(40), 40)) == \
        '3.14159265358979323846264338327950288'

    # unicode
    assert Float('0.73908513321516064100000000') == \
        Float('0.73908513321516064100000000')
    assert Float('0.73908513321516064100000000', 28) == \
        Float('0.73908513321516064100000000', 28)

    # binary precision
    # Decimal value 0.1 cannot be expressed precisely as a base 2 fraction
    a = Float(S.One/10, dps=15)
    b = Float(S.One/10, dps=16)
    p = Float(S.One/10, precision=53)
    q = Float(S.One/10, precision=54)
    assert a._mpf_ == p._mpf_
    assert not a._mpf_ == q._mpf_
    assert not b._mpf_ == q._mpf_

    # Precision specifying errors
    raises(ValueError, lambda: Float("1.23", dps=3, precision=10))
    raises(ValueError, lambda: Float("1.23", dps="", precision=10))
    raises(ValueError, lambda: Float("1.23", dps=3, precision=""))
    raises(ValueError, lambda: Float("1.23", dps="", precision=""))

    # from NumberSymbol
    assert same_and_same_prec(Float(pi, 32), pi.evalf(32))
    assert same_and_same_prec(Float(Catalan), Catalan.evalf())

    # oo and nan
    u = ['inf', '-inf', 'nan', 'iNF', '+inf']
    v = [oo, -oo, nan, oo, oo]
    for i, a in zip(u, v):
        assert Float(i) is a


def test_zero_not_false():
    # https://github.com/sympy/sympy/issues/20796
    assert (S(0.0) == S.false) is False
    assert (S.false == S(0.0)) is False
    assert (S(0) == S.false) is False
    assert (S.false == S(0)) is False


@conserve_mpmath_dps
def test_float_mpf():
    import mpmath
    mpmath.mp.dps = 100
    mp_pi = mpmath.pi()

    assert Float(mp_pi, 100) == Float(mp_pi._mpf_, 100) == pi.evalf(100)

    mpmath.mp.dps = 15

    assert Float(mp_pi, 100) == Float(mp_pi._mpf_, 100) == pi.evalf(100)


def test_Float_RealElement():
    repi = RealField(dps=100)(pi.evalf(100))
    # We still have to pass the precision because Float doesn't know what
    # RealElement is, but make sure it keeps full precision from the result.
    assert Float(repi, 100) == pi.evalf(100)


def test_Float_default_to_highprec_from_str():
    s = str(pi.evalf(128))
    assert same_and_same_prec(Float(s), Float(s, ''))


def test_Float_eval():
    a = Float(3.2)
    assert (a**2).is_Float


def test_Float_issue_2107():
    a = Float(0.1, 10)
    b = Float("0.1", 10)

    assert a - a == 0
    assert a + (-a) == 0
    assert S.Zero + a - a == 0
    assert S.Zero + a + (-a) == 0

    assert b - b == 0
    assert b + (-b) == 0
    assert S.Zero + b - b == 0
    assert S.Zero + b + (-b) == 0


def test_issue_14289():
    from sympy.polys.numberfields import to_number_field

    a = 1 - sqrt(2)
    b = to_number_field(a)
    assert b.as_expr() == a
    assert b.minpoly(a).expand() == 0


def test_Float_from_tuple():
    a = Float((0, '1L', 0, 1))
    b = Float((0, '1', 0, 1))
    assert a == b


def test_Infinity():
    assert oo != 1
    assert 1*oo is oo
    assert 1 != oo
    assert oo != -oo
    assert oo != Symbol("x")**3
    assert oo + 1 is oo
    assert 2 + oo is oo
    assert 3*oo + 2 is oo
    assert S.Half**oo == 0
    assert S.Half**(-oo) is oo
    assert -oo*3 is -oo
    assert oo + oo is oo
    assert -oo + oo*(-5) is -oo
    assert 1/oo == 0
    assert 1/(-oo) == 0
    assert 8/oo == 0
    assert oo % 2 is nan
    assert 2 % oo is nan
    assert oo/oo is nan
    assert oo/-oo is nan
    assert -oo/oo is nan
    assert -oo/-oo is nan
    assert oo - oo is nan
    assert oo - -oo is oo
    assert -oo - oo is -oo
    assert -oo - -oo is nan
    assert oo + -oo is nan
    assert -oo + oo is nan
    assert oo + oo is oo
    assert -oo + oo is nan
    assert oo + -oo is nan
    assert -oo + -oo is -oo
    assert oo*oo is oo
    assert -oo*oo is -oo
    assert oo*-oo is -oo
    assert -oo*-oo is oo
    assert oo/0 is oo
    assert -oo/0 is -oo
    assert 0/oo == 0
    assert 0/-oo == 0
    assert oo*0 is nan
    assert -oo*0 is nan
    assert 0*oo is nan
    assert 0*-oo is nan
    assert oo + 0 is oo
    assert -oo + 0 is -oo
    assert 0 + oo is oo
    assert 0 + -oo is -oo
    assert oo - 0 is oo
    assert -oo - 0 is -oo
    assert 0 - oo is -oo
    assert 0 - -oo is oo
    assert oo/2 is oo
    assert -oo/2 is -oo
    assert oo/-2 is -oo
    assert -oo/-2 is oo
    assert oo*2 is oo
    assert -oo*2 is -oo
    assert oo*-2 is -oo
    assert 2/oo == 0
    assert 2/-oo == 0
    assert -2/oo == 0
    assert -2/-oo == 0
    assert 2*oo is oo
    assert 2*-oo is -oo
    assert -2*oo is -oo
    assert -2*-oo is oo
    assert 2 + oo is oo
    assert 2 - oo is -oo
    assert -2 + oo is oo
    assert -2 - oo is -oo
    assert 2 + -oo is -oo
    assert 2 - -oo is oo
    assert -2 + -oo is -oo
    assert -2 - -oo is oo
    assert S(2) + oo is oo
    assert S(2) - oo is -oo
    assert oo/I == -oo*I
    assert -oo/I == oo*I
    assert oo*float(1) == _inf and (oo*float(1)) is oo
    assert -oo*float(1) == _ninf and (-oo*float(1)) is -oo
    assert oo/float(1) == _inf and (oo/float(1)) is oo
    assert -oo/float(1) == _ninf and (-oo/float(1)) is -oo
    assert oo*float(-1) == _ninf and (oo*float(-1)) is -oo
    assert -oo*float(-1) == _inf and (-oo*float(-1)) is oo
    assert oo/float(-1) == _ninf and (oo/float(-1)) is -oo
    assert -oo/float(-1) == _inf and (-oo/float(-1)) is oo
    assert oo + float(1) == _inf and (oo + float(1)) is oo
    assert -oo + float(1) == _ninf and (-oo + float(1)) is -oo
    assert oo - float(1) == _inf and (oo - float(1)) is oo
    assert -oo - float(1) == _ninf and (-oo - float(1)) is -oo
    assert float(1)*oo == _inf and (float(1)*oo) is oo
    assert float(1)*-oo == _ninf and (float(1)*-oo) is -oo
    assert float(1)/oo == 0
    assert float(1)/-oo == 0
    assert float(-1)*oo == _ninf and (float(-1)*oo) is -oo
    assert float(-1)*-oo == _inf and (float(-1)*-oo) is oo
    assert float(-1)/oo == 0
    assert float(-1)/-oo == 0
    assert float(1) + oo is oo
    assert float(1) + -oo is -oo
    assert float(1) - oo is -oo
    assert float(1) - -oo is oo
    assert oo == float(oo)
    assert (oo != float(oo)) is False
    assert type(float(oo)) is float
    assert -oo == float(-oo)
    assert (-oo != float(-oo)) is False
    assert type(float(-oo)) is float

    assert Float('nan') is nan
    assert nan*1.0 is nan
    assert -1.0*nan is nan
    assert nan*oo is nan
    assert nan*-oo is nan
    assert nan/oo is nan
    assert nan/-oo is nan
    assert nan + oo is nan
    assert nan + -oo is nan
    assert nan - oo is nan
    assert nan - -oo is nan
    assert -oo * S.Zero is nan

    assert oo*nan is nan
    assert -oo*nan is nan
    assert oo/nan is nan
    assert -oo/nan is nan
    assert oo + nan is nan
    assert -oo + nan is nan
    assert oo - nan is nan
    assert -oo - nan is nan
    assert S.Zero * oo is nan
    assert oo.is_Rational is False
    assert isinstance(oo, Rational) is False

    assert S.One/oo == 0
    assert -S.One/oo == 0
    assert S.One/-oo == 0
    assert -S.One/-oo == 0
    assert S.One*oo is oo
    assert -S.One*oo is -oo
    assert S.One*-oo is -oo
    assert -S.One*-oo is oo
    assert S.One/nan is nan
    assert S.One - -oo is oo
    assert S.One + nan is nan
    assert S.One - nan is nan
    assert nan - S.One is nan
    assert nan/S.One is nan
    assert -oo - S.One is -oo


def test_Infinity_2():
    x = Symbol('x')
    assert oo*x != oo
    assert oo*(pi - 1) is oo
    assert oo*(1 - pi) is -oo

    assert (-oo)*x != -oo
    assert (-oo)*(pi - 1) is -oo
    assert (-oo)*(1 - pi) is oo

    assert (-1)**S.NaN is S.NaN
    assert oo - _inf is S.NaN
    assert oo + _ninf is S.NaN
    assert oo*0 is S.NaN
    assert oo/_inf is S.NaN
    assert oo/_ninf is S.NaN
    assert oo**S.NaN is S.NaN
    assert -oo + _inf is S.NaN
    assert -oo - _ninf is S.NaN
    assert -oo*S.NaN is S.NaN
    assert -oo*0 is S.NaN
    assert -oo/_inf is S.NaN
    assert -oo/_ninf is S.NaN
    assert -oo/S.NaN is S.NaN
    assert abs(-oo) is oo
    assert all((-oo)**i is S.NaN for i in (oo, -oo, S.NaN))
    assert (-oo)**3 is -oo
    assert (-oo)**2 is oo
    assert abs(S.ComplexInfinity) is oo


def test_Mul_Infinity_Zero():
    assert Float(0)*_inf is nan
    assert Float(0)*_ninf is nan
    assert Float(0)*_inf is nan
    assert Float(0)*_ninf is nan
    assert _inf*Float(0) is nan
    assert _ninf*Float(0) is nan
    assert _inf*Float(0) is nan
    assert _ninf*Float(0) is nan


def test_Div_By_Zero():
    assert 1/S.Zero is zoo
    assert 1/Float(0) is zoo
    assert 0/S.Zero is nan
    assert 0/Float(0) is nan
    assert S.Zero/0 is nan
    assert Float(0)/0 is nan
    assert -1/S.Zero is zoo
    assert -1/Float(0) is zoo


@_both_exp_pow
def test_Infinity_inequations():
    assert oo > pi
    assert not (oo < pi)
    assert exp(-3) < oo

    assert _inf > pi
    assert not (_inf < pi)
    assert exp(-3) < _inf

    raises(TypeError, lambda: oo < I)
    raises(TypeError, lambda: oo <= I)
    raises(TypeError, lambda: oo > I)
    raises(TypeError, lambda: oo >= I)
    raises(TypeError, lambda: -oo < I)
    raises(TypeError, lambda: -oo <= I)
    raises(TypeError, lambda: -oo > I)
    raises(TypeError, lambda: -oo >= I)

    raises(TypeError, lambda: I < oo)
    raises(TypeError, lambda: I <= oo)
    raises(TypeError, lambda: I > oo)
    raises(TypeError, lambda: I >= oo)
    raises(TypeError, lambda: I < -oo)
    raises(TypeError, lambda: I <= -oo)
    raises(TypeError, lambda: I > -oo)
    raises(TypeError, lambda: I >= -oo)

    assert oo > -oo and oo >= -oo
    assert (oo < -oo) == False and (oo <= -oo) == False
    assert -oo < oo and -oo <= oo
    assert (-oo > oo) == False and (-oo >= oo) == False

    assert (oo < oo) == False  # issue 7775
    assert (oo > oo) == False
    assert (-oo > -oo) == False and (-oo < -oo) == False
    assert oo >= oo and oo <= oo and -oo >= -oo and -oo <= -oo
    assert (-oo < -_inf) ==  False
    assert (oo > _inf) == False
    assert -oo >= -_inf
    assert oo <= _inf

    x = Symbol('x')
    b = Symbol('b', finite=True, real=True)
    assert (x < oo) == Lt(x, oo)  # issue 7775
    assert b < oo and b > -oo and b <= oo and b >= -oo
    assert oo > b and oo >= b and (oo < b) == False and (oo <= b) == False
    assert (-oo > b) == False and (-oo >= b) == False and -oo < b and -oo <= b
    assert (oo < x) == Lt(oo, x) and (oo > x) == Gt(oo, x)
    assert (oo <= x) == Le(oo, x) and (oo >= x) == Ge(oo, x)
    assert (-oo < x) == Lt(-oo, x) and (-oo > x) == Gt(-oo, x)
    assert (-oo <= x) == Le(-oo, x) and (-oo >= x) == Ge(-oo, x)


def test_NaN():
    assert nan is nan
    assert nan != 1
    assert 1*nan is nan
    assert 1 != nan
    assert -nan is nan
    assert oo != Symbol("x")**3
    assert 2 + nan is nan
    assert 3*nan + 2 is nan
    assert -nan*3 is nan
    assert nan + nan is nan
    assert -nan + nan*(-5) is nan
    assert 8/nan is nan
    raises(TypeError, lambda: nan > 0)
    raises(TypeError, lambda: nan < 0)
    raises(TypeError, lambda: nan >= 0)
    raises(TypeError, lambda: nan <= 0)
    raises(TypeError, lambda: 0 < nan)
    raises(TypeError, lambda: 0 > nan)
    raises(TypeError, lambda: 0 <= nan)
    raises(TypeError, lambda: 0 >= nan)
    assert nan**0 == 1  # as per IEEE 754
    assert 1**nan is nan # IEEE 754 is not the best choice for symbolic work
    # test Pow._eval_power's handling of NaN
    assert Pow(nan, 0, evaluate=False)**2 == 1
    for n in (1, 1., S.One, S.NegativeOne, Float(1)):
        assert n + nan is nan
        assert n - nan is nan
        assert nan + n is nan
        assert nan - n is nan
        assert n/nan is nan
        assert nan/n is nan


def test_special_numbers():
    assert isinstance(S.NaN, Number) is True
    assert isinstance(S.Infinity, Number) is True
    assert isinstance(S.NegativeInfinity, Number) is True

    assert S.NaN.is_number is True
    assert S.Infinity.is_number is True
    assert S.NegativeInfinity.is_number is True
    assert S.ComplexInfinity.is_number is True

    assert isinstance(S.NaN, Rational) is False
    assert isinstance(S.Infinity, Rational) is False
    assert isinstance(S.NegativeInfinity, Rational) is False

    assert S.NaN.is_rational is not True
    assert S.Infinity.is_rational is not True
    assert S.NegativeInfinity.is_rational is not True


def test_powers():
    assert integer_nthroot(1, 2) == (1, True)
    assert integer_nthroot(1, 5) == (1, True)
    assert integer_nthroot(2, 1) == (2, True)
    assert integer_nthroot(2, 2) == (1, False)
    assert integer_nthroot(2, 5) == (1, False)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(123**25, 25) == (123, True)
    assert integer_nthroot(123**25 + 1, 25) == (123, False)
    assert integer_nthroot(123**25 - 1, 25) == (122, False)
    assert integer_nthroot(1, 1) == (1, True)
    assert integer_nthroot(0, 1) == (0, True)
    assert integer_nthroot(0, 3) == (0, True)
    assert integer_nthroot(10000, 1) == (10000, True)
    assert integer_nthroot(4, 2) == (2, True)
    assert integer_nthroot(16, 2) == (4, True)
    assert integer_nthroot(26, 2) == (5, False)
    assert integer_nthroot(1234567**7, 7) == (1234567, True)
    assert integer_nthroot(1234567**7 + 1, 7) == (1234567, False)
    assert integer_nthroot(1234567**7 - 1, 7) == (1234566, False)
    b = 25**1000
    assert integer_nthroot(b, 1000) == (25, True)
    assert integer_nthroot(b + 1, 1000) == (25, False)
    assert integer_nthroot(b - 1, 1000) == (24, False)
    c = 10**400
    c2 = c**2
    assert integer_nthroot(c2, 2) == (c, True)
    assert integer_nthroot(c2 + 1, 2) == (c, False)
    assert integer_nthroot(c2 - 1, 2) == (c - 1, False)
    assert integer_nthroot(2, 10**10) == (1, False)

    p, r = integer_nthroot(int(factorial(10000)), 100)
    assert p % (10**10) == 5322420655
    assert not r

    # Test that this is fast
    assert integer_nthroot(2, 10**10) == (1, False)

    # output should be int if possible
    assert type(integer_nthroot(2**61, 2)[0]) is int


def test_integer_nthroot_overflow():
    assert integer_nthroot(10**(50*50), 50) == (10**50, True)
    assert integer_nthroot(10**100000, 10000) == (10**10, True)


def test_integer_log():
    raises(ValueError, lambda: integer_log(2, 1))
    raises(ValueError, lambda: integer_log(0, 2))
    raises(ValueError, lambda: integer_log(1.1, 2))
    raises(ValueError, lambda: integer_log(1, 2.2))

    assert integer_log(1, 2) == (0, True)
    assert integer_log(1, 3) == (0, True)
    assert integer_log(2, 3) == (0, False)
    assert integer_log(3, 3) == (1, True)
    assert integer_log(3*2, 3) == (1, False)
    assert integer_log(3**2, 3) == (2, True)
    assert integer_log(3*4, 3) == (2, False)
    assert integer_log(3**3, 3) == (3, True)
    assert integer_log(27, 5) == (2, False)
    assert integer_log(2, 3) == (0, False)
    assert integer_log(-4, 2) == (2, False)
    assert integer_log(-16, 4) == (0, False)
    assert integer_log(-4, -2) == (2, False)
    assert integer_log(4, -2) == (2, True)
    assert integer_log(-8, -2) == (3, True)
    assert integer_log(8, -2) == (3, False)
    assert integer_log(-9, 3) == (0, False)
    assert integer_log(-9, -3) == (2, False)
    assert integer_log(9, -3) == (2, True)
    assert integer_log(-27, -3) == (3, True)
    assert integer_log(27, -3) == (3, False)


def test_isqrt():
    from math import sqrt as _sqrt
    limit = 4503599761588223
    assert int(_sqrt(limit)) == integer_nthroot(limit, 2)[0]
    assert int(_sqrt(limit + 1)) != integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + 1) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + S.Half) == integer_nthroot(limit, 2)[0]
    assert isqrt(limit + 1 + S.Half) == integer_nthroot(limit + 1, 2)[0]
    assert isqrt(limit + 2 + S.Half) == integer_nthroot(limit + 2, 2)[0]

    # Regression tests for https://github.com/sympy/sympy/issues/17034
    assert isqrt(4503599761588224) == 67108864
    assert isqrt(9999999999999999) == 99999999

    # Other corner cases, especially involving non-integers.
    raises(ValueError, lambda: isqrt(-1))
    raises(ValueError, lambda: isqrt(-10**1000))
    raises(ValueError, lambda: isqrt(Rational(-1, 2)))

    tiny = Rational(1, 10**1000)
    raises(ValueError, lambda: isqrt(-tiny))
    assert isqrt(1-tiny) == 0
    assert isqrt(4503599761588224-tiny) == 67108864
    assert isqrt(10**100 - tiny) == 10**50 - 1


def test_powers_Integer():
    """Test Integer._eval_power"""
    # check infinity
    assert S.One ** S.Infinity is S.NaN
    assert S.NegativeOne** S.Infinity is S.NaN
    assert S(2) ** S.Infinity is S.Infinity
    assert S(-2)** S.Infinity == zoo
    assert S(0) ** S.Infinity is S.Zero

    # check Nan
    assert S.One ** S.NaN is S.NaN
    assert S.NegativeOne ** S.NaN is S.NaN

    # check for exact roots
    assert S.NegativeOne ** Rational(6, 5) == - (-1)**(S.One/5)
    assert sqrt(S(4)) == 2
    assert sqrt(S(-4)) == I * 2
    assert S(16) ** Rational(1, 4) == 2
    assert S(-16) ** Rational(1, 4) == 2 * (-1)**Rational(1, 4)
    assert S(9) ** Rational(3, 2) == 27
    assert S(-9) ** Rational(3, 2) == -27*I
    assert S(27) ** Rational(2, 3) == 9
    assert S(-27) ** Rational(2, 3) == 9 * (S.NegativeOne ** Rational(2, 3))
    assert (-2) ** Rational(-2, 1) == Rational(1, 4)

    # not exact roots
    assert sqrt(-3) == I*sqrt(3)
    assert (3) ** (Rational(3, 2)) == 3 * sqrt(3)
    assert (-3) ** (Rational(3, 2)) == - 3 * sqrt(-3)
    assert (-3) ** (Rational(5, 2)) == 9 * I * sqrt(3)
    assert (-3) ** (Rational(7, 2)) == - I * 27 * sqrt(3)
    assert (2) ** (Rational(3, 2)) == 2 * sqrt(2)
    assert (2) ** (Rational(-3, 2)) == sqrt(2) / 4
    assert (81) ** (Rational(2, 3)) == 9 * (S(3) ** (Rational(2, 3)))
    assert (-81) ** (Rational(2, 3)) == 9 * (S(-3) ** (Rational(2, 3)))
    assert (-3) ** Rational(-7, 3) == \
        -(-1)**Rational(2, 3)*3**Rational(2, 3)/27
    assert (-3) ** Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*3**Rational(1, 3)/3

    # join roots
    assert sqrt(6) + sqrt(24) == 3*sqrt(6)
    assert sqrt(2) * sqrt(3) == sqrt(6)

    # separate symbols & constansts
    x = Symbol("x")
    assert sqrt(49 * x) == 7 * sqrt(x)
    assert sqrt((3 - sqrt(pi)) ** 2) == 3 - sqrt(pi)

    # check that it is fast for big numbers
    assert (2**64 + 1) ** Rational(4, 3)
    assert (2**64 + 1) ** Rational(17, 25)

    # negative rational power and negative base
    assert (-3) ** Rational(-7, 3) == \
        -(-1)**Rational(2, 3)*3**Rational(2, 3)/27
    assert (-3) ** Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*3**Rational(1, 3)/3
    assert (-2) ** Rational(-10, 3) == \
        (-1)**Rational(2, 3)*2**Rational(2, 3)/16
    assert abs(Pow(-2, Rational(-10, 3)).n() -
        Pow(-2, Rational(-10, 3), evaluate=False).n()) < 1e-16

    # negative base and rational power with some simplification
    assert (-8) ** Rational(2, 5) == \
        2*(-1)**Rational(2, 5)*2**Rational(1, 5)
    assert (-4) ** Rational(9, 5) == \
        -8*(-1)**Rational(4, 5)*2**Rational(3, 5)

    assert S(1234).factors() == {617: 1, 2: 1}
    assert Rational(2*3, 3*5*7).factors() == {2: 1, 5: -1, 7: -1}

    # test that eval_power factors numbers bigger than
    # the current limit in factor_trial_division (2**15)
    from sympy.ntheory.generate import nextprime
    n = nextprime(2**15)
    assert sqrt(n**2) == n
    assert sqrt(n**3) == n*sqrt(n)
    assert sqrt(4*n) == 2*sqrt(n)

    # check that factors of base with powers sharing gcd with power are removed
    assert (2**4*3)**Rational(1, 6) == 2**Rational(2, 3)*3**Rational(1, 6)
    assert (2**4*3)**Rational(5, 6) == 8*2**Rational(1, 3)*3**Rational(5, 6)

    # check that bases sharing a gcd are exptracted
    assert 2**Rational(1, 3)*3**Rational(1, 4)*6**Rational(1, 5) == \
        2**Rational(8, 15)*3**Rational(9, 20)
    assert sqrt(8)*24**Rational(1, 3)*6**Rational(1, 5) == \
        4*2**Rational(7, 10)*3**Rational(8, 15)
    assert sqrt(8)*(-24)**Rational(1, 3)*(-6)**Rational(1, 5) == \
        4*(-3)**Rational(8, 15)*2**Rational(7, 10)
    assert 2**Rational(1, 3)*2**Rational(8, 9) == 2*2**Rational(2, 9)
    assert 2**Rational(2, 3)*6**Rational(1, 3) == 2*3**Rational(1, 3)
    assert 2**Rational(2, 3)*6**Rational(8, 9) == \
        2*2**Rational(5, 9)*3**Rational(8, 9)
    assert (-2)**Rational(2, S(3))*(-4)**Rational(1, S(3)) == -2*2**Rational(1, 3)
    assert 3*Pow(3, 2, evaluate=False) == 3**3
    assert 3*Pow(3, Rational(-1, 3), evaluate=False) == 3**Rational(2, 3)
    assert (-2)**Rational(1, 3)*(-3)**Rational(1, 4)*(-5)**Rational(5, 6) == \
        -(-1)**Rational(5, 12)*2**Rational(1, 3)*3**Rational(1, 4) * \
        5**Rational(5, 6)

    assert Integer(-2)**Symbol('', even=True) == \
        Integer(2)**Symbol('', even=True)
    assert (-1)**Float(.5) == 1.0*I


def test_powers_Rational():
    """Test Rational._eval_power"""
    # check infinity
    assert S.Half ** S.Infinity == 0
    assert Rational(3, 2) ** S.Infinity is S.Infinity
    assert Rational(-1, 2) ** S.Infinity == 0
    assert Rational(-3, 2) ** S.Infinity == zoo

    # check Nan
    assert Rational(3, 4) ** S.NaN is S.NaN
    assert Rational(-2, 3) ** S.NaN is S.NaN

    # exact roots on numerator
    assert sqrt(Rational(4, 3)) == 2 * sqrt(3) / 3
    assert Rational(4, 3) ** Rational(3, 2) == 8 * sqrt(3) / 9
    assert sqrt(Rational(-4, 3)) == I * 2 * sqrt(3) / 3
    assert Rational(-4, 3) ** Rational(3, 2) == - I * 8 * sqrt(3) / 9
    assert Rational(27, 2) ** Rational(1, 3) == 3 * (2 ** Rational(2, 3)) / 2
    assert Rational(5**3, 8**3) ** Rational(4, 3) == Rational(5**4, 8**4)

    # exact root on denominator
    assert sqrt(Rational(1, 4)) == S.Half
    assert sqrt(Rational(1, -4)) == I * S.Half
    assert sqrt(Rational(3, 4)) == sqrt(3) / 2
    assert sqrt(Rational(3, -4)) == I * sqrt(3) / 2
    assert Rational(5, 27) ** Rational(1, 3) == (5 ** Rational(1, 3)) / 3

    # not exact roots
    assert sqrt(S.Half) == sqrt(2) / 2
    assert sqrt(Rational(-4, 7)) == I * sqrt(Rational(4, 7))
    assert Rational(-3, 2)**Rational(-7, 3) == \
        -4*(-1)**Rational(2, 3)*2**Rational(1, 3)*3**Rational(2, 3)/27
    assert Rational(-3, 2)**Rational(-2, 3) == \
        -(-1)**Rational(1, 3)*2**Rational(2, 3)*3**Rational(1, 3)/3
    assert Rational(-3, 2)**Rational(-10, 3) == \
        8*(-1)**Rational(2, 3)*2**Rational(1, 3)*3**Rational(2, 3)/81
    assert abs(Pow(Rational(-2, 3), Rational(-7, 4)).n() -
        Pow(Rational(-2, 3), Rational(-7, 4), evaluate=False).n()) < 1e-16

    # negative integer power and negative rational base
    assert Rational(-2, 3) ** Rational(-2, 1) == Rational(9, 4)

    a = Rational(1, 10)
    assert a**Float(a, 2) == Float(a, 2)**Float(a, 2)
    assert Rational(-2, 3)**Symbol('', even=True) == \
        Rational(2, 3)**Symbol('', even=True)


def test_powers_Float():
    assert str((S('-1/10')**S('3/10')).n()) == str(Float(-.1)**(.3))


def test_lshift_Integer():
    assert Integer(0) << Integer(2) == Integer(0)
    assert Integer(0) << 2 == Integer(0)
    assert 0 << Integer(2) == Integer(0)

    assert Integer(0b11) << Integer(0) == Integer(0b11)
    assert Integer(0b11) << 0 == Integer(0b11)
    assert 0b11 << Integer(0) == Integer(0b11)

    assert Integer(0b11) << Integer(2) == Integer(0b11 << 2)
    assert Integer(0b11) << 2 == Integer(0b11 << 2)
    assert 0b11 << Integer(2) == Integer(0b11 << 2)

    assert Integer(-0b11) << Integer(2) == Integer(-0b11 << 2)
    assert Integer(-0b11) << 2 == Integer(-0b11 << 2)
    assert -0b11 << Integer(2) == Integer(-0b11 << 2)

    raises(TypeError, lambda: Integer(2) << 0.0)
    raises(TypeError, lambda: 0.0 << Integer(2))
    raises(ValueError, lambda: Integer(1) << Integer(-1))


def test_rshift_Integer():
    assert Integer(0) >> Integer(2) == Integer(0)
    assert Integer(0) >> 2 == Integer(0)
    assert 0 >> Integer(2) == Integer(0)

    assert Integer(0b11) >> Integer(0) == Integer(0b11)
    assert Integer(0b11) >> 0 == Integer(0b11)
    assert 0b11 >> Integer(0) == Integer(0b11)

    assert Integer(0b11) >> Integer(2) == Integer(0)
    assert Integer(0b11) >> 2 == Integer(0)
    assert 0b11 >> Integer(2) == Integer(0)

    assert Integer(-0b11) >> Integer(2) == Integer(-1)
    assert Integer(-0b11) >> 2 == Integer(-1)
    assert -0b11 >> Integer(2) == Integer(-1)

    assert Integer(0b1100) >> Integer(2) == Integer(0b1100 >> 2)
    assert Integer(0b1100) >> 2 == Integer(0b1100 >> 2)
    assert 0b1100 >> Integer(2) == Integer(0b1100 >> 2)

    assert Integer(-0b1100) >> Integer(2) == Integer(-0b1100 >> 2)
    assert Integer(-0b1100) >> 2 == Integer(-0b1100 >> 2)
    assert -0b1100 >> Integer(2) == Integer(-0b1100 >> 2)

    raises(TypeError, lambda: Integer(0b10) >> 0.0)
    raises(TypeError, lambda: 0.0 >> Integer(2))
    raises(ValueError, lambda: Integer(1) >> Integer(-1))


def test_and_Integer():
    assert Integer(0b01010101) & Integer(0b10101010) == Integer(0)
    assert Integer(0b01010101) & 0b10101010 == Integer(0)
    assert 0b01010101 & Integer(0b10101010) == Integer(0)

    assert Integer(0b01010101) & Integer(0b11011011) == Integer(0b01010001)
    assert Integer(0b01010101) & 0b11011011 == Integer(0b01010001)
    assert 0b01010101 & Integer(0b11011011) == Integer(0b01010001)

    assert -Integer(0b01010101) & Integer(0b11011011) == Integer(-0b01010101 & 0b11011011)
    assert Integer(-0b01010101) & 0b11011011 == Integer(-0b01010101 & 0b11011011)
    assert -0b01010101 & Integer(0b11011011) == Integer(-0b01010101 & 0b11011011)

    assert Integer(0b01010101) & -Integer(0b11011011) == Integer(0b01010101 & -0b11011011)
    assert Integer(0b01010101) & -0b11011011 == Integer(0b01010101 & -0b11011011)
    assert 0b01010101 & Integer(-0b11011011) == Integer(0b01010101 & -0b11011011)

    raises(TypeError, lambda: Integer(2) & 0.0)
    raises(TypeError, lambda: 0.0 & Integer(2))


def test_xor_Integer():
    assert Integer(0b01010101) ^ Integer(0b11111111) == Integer(0b10101010)
    assert Integer(0b01010101) ^ 0b11111111 == Integer(0b10101010)
    assert 0b01010101 ^ Integer(0b11111111) == Integer(0b10101010)

    assert Integer(0b01010101) ^ Integer(0b11011011) == Integer(0b10001110)
    assert Integer(0b01010101) ^ 0b11011011 == Integer(0b10001110)
    assert 0b01010101 ^ Integer(0b11011011) == Integer(0b10001110)

    assert -Integer(0b01010101) ^ Integer(0b11011011) == Integer(-0b01010101 ^ 0b11011011)
    assert Integer(-0b01010101) ^ 0b11011011 == Integer(-0b01010101 ^ 0b11011011)
    assert -0b01010101 ^ Integer(0b11011011) == Integer(-0b01010101 ^ 0b11011011)

    assert Integer(0b01010101) ^ -Integer(0b11011011) == Integer(0b01010101 ^ -0b11011011)
    assert Integer(0b01010101) ^ -0b11011011 == Integer(0b01010101 ^ -0b11011011)
    assert 0b01010101 ^ Integer(-0b11011011) == Integer(0b01010101 ^ -0b11011011)

    raises(TypeError, lambda: Integer(2) ^ 0.0)
    raises(TypeError, lambda: 0.0 ^ Integer(2))


def test_or_Integer():
    assert Integer(0b01010101) | Integer(0b10101010) == Integer(0b11111111)
    assert Integer(0b01010101) | 0b10101010 == Integer(0b11111111)
    assert 0b01010101 | Integer(0b10101010) == Integer(0b11111111)

    assert Integer(0b01010101) | Integer(0b11011011) == Integer(0b11011111)
    assert Integer(0b01010101) | 0b11011011 == Integer(0b11011111)
    assert 0b01010101 | Integer(0b11011011) == Integer(0b11011111)

    assert -Integer(0b01010101) | Integer(0b11011011) == Integer(-0b01010101 | 0b11011011)
    assert Integer(-0b01010101) | 0b11011011 == Integer(-0b01010101 | 0b11011011)
    assert -0b01010101 | Integer(0b11011011) == Integer(-0b01010101 | 0b11011011)

    assert Integer(0b01010101) | -Integer(0b11011011) == Integer(0b01010101 | -0b11011011)
    assert Integer(0b01010101) | -0b11011011 == Integer(0b01010101 | -0b11011011)
    assert 0b01010101 | Integer(-0b11011011) == Integer(0b01010101 | -0b11011011)

    raises(TypeError, lambda: Integer(2) | 0.0)
    raises(TypeError, lambda: 0.0 | Integer(2))


def test_invert_Integer():
    assert ~Integer(0b01010101) == Integer(-0b01010110)
    assert ~Integer(0b01010101) == Integer(~0b01010101)
    assert ~(~Integer(0b01010101)) == Integer(0b01010101)


def test_abs1():
    assert Rational(1, 6) != Rational(-1, 6)
    assert abs(Rational(1, 6)) == abs(Rational(-1, 6))


def test_accept_int():
    assert not Float(4) == 4
    assert Float(4) != 4
    assert Float(4) == 4.0


def test_dont_accept_str():
    assert Float("0.2") != "0.2"
    assert not (Float("0.2") == "0.2")


def test_int():
    a = Rational(5)
    assert int(a) == 5
    a = Rational(9, 10)
    assert int(a) == int(-a) == 0
    assert 1/(-1)**Rational(2, 3) == -(-1)**Rational(1, 3)
    # issue 10368
    a = Rational(32442016954, 78058255275)
    assert type(int(a)) is type(int(-a)) is int


def test_int_NumberSymbols():
    assert int(Catalan) == 0
    assert int(EulerGamma) == 0
    assert int(pi) == 3
    assert int(E) == 2
    assert int(GoldenRatio) == 1
    assert int(TribonacciConstant) == 1
    for i in [Catalan, E, EulerGamma, GoldenRatio, TribonacciConstant, pi]:
        a, b = i.approximation_interval(Integer)
        ia = int(i)
        assert ia == a
        assert isinstance(ia, int)
        assert b == a + 1
        assert a.is_Integer and b.is_Integer


def test_real_bug():
    x = Symbol("x")
    assert str(2.0*x*x) in ["(2.0*x)*x", "2.0*x**2", "2.00000000000000*x**2"]
    assert str(2.1*x*x) != "(2.0*x)*x"


def test_bug_sqrt():
    assert ((sqrt(Rational(2)) + 1)*(sqrt(Rational(2)) - 1)).expand() == 1


def test_pi_Pi():
    "Test that pi (instance) is imported, but Pi (class) is not"
    from sympy import pi  # noqa
    with raises(ImportError):
        from sympy import Pi  # noqa


def test_no_len():
    # there should be no len for numbers
    raises(TypeError, lambda: len(Rational(2)))
    raises(TypeError, lambda: len(Rational(2, 3)))
    raises(TypeError, lambda: len(Integer(2)))


def test_issue_3321():
    assert sqrt(Rational(1, 5)) == Rational(1, 5)**S.Half
    assert 5 * sqrt(Rational(1, 5)) == sqrt(5)


def test_issue_3692():
    assert ((-1)**Rational(1, 6)).expand(complex=True) == I/2 + sqrt(3)/2
    assert ((-5)**Rational(1, 6)).expand(complex=True) == \
        5**Rational(1, 6)*I/2 + 5**Rational(1, 6)*sqrt(3)/2
    assert ((-64)**Rational(1, 6)).expand(complex=True) == I + sqrt(3)


def test_issue_3423():
    x = Symbol("x")
    assert sqrt(x - 1).as_base_exp() == (x - 1, S.Half)
    assert sqrt(x - 1) != I*sqrt(1 - x)


def test_issue_3449():
    x = Symbol("x")
    assert sqrt(x - 1).subs(x, 5) == 2


def test_issue_13890():
    x = Symbol("x")
    e = (-x/4 - S.One/12)**x - 1
    f = simplify(e)
    a = Rational(9, 5)
    assert abs(e.subs(x,a).evalf() - f.subs(x,a).evalf()) < 1e-15


def test_Integer_factors():
    def F(i):
        return Integer(i).factors()

    assert F(1) == {}
    assert F(2) == {2: 1}
    assert F(3) == {3: 1}
    assert F(4) == {2: 2}
    assert F(5) == {5: 1}
    assert F(6) == {2: 1, 3: 1}
    assert F(7) == {7: 1}
    assert F(8) == {2: 3}
    assert F(9) == {3: 2}
    assert F(10) == {2: 1, 5: 1}
    assert F(11) == {11: 1}
    assert F(12) == {2: 2, 3: 1}
    assert F(13) == {13: 1}
    assert F(14) == {2: 1, 7: 1}
    assert F(15) == {3: 1, 5: 1}
    assert F(16) == {2: 4}
    assert F(17) == {17: 1}
    assert F(18) == {2: 1, 3: 2}
    assert F(19) == {19: 1}
    assert F(20) == {2: 2, 5: 1}
    assert F(21) == {3: 1, 7: 1}
    assert F(22) == {2: 1, 11: 1}
    assert F(23) == {23: 1}
    assert F(24) == {2: 3, 3: 1}
    assert F(25) == {5: 2}
    assert F(26) == {2: 1, 13: 1}
    assert F(27) == {3: 3}
    assert F(28) == {2: 2, 7: 1}
    assert F(29) == {29: 1}
    assert F(30) == {2: 1, 3: 1, 5: 1}
    assert F(31) == {31: 1}
    assert F(32) == {2: 5}
    assert F(33) == {3: 1, 11: 1}
    assert F(34) == {2: 1, 17: 1}
    assert F(35) == {5: 1, 7: 1}
    assert F(36) == {2: 2, 3: 2}
    assert F(37) == {37: 1}
    assert F(38) == {2: 1, 19: 1}
    assert F(39) == {3: 1, 13: 1}
    assert F(40) == {2: 3, 5: 1}
    assert F(41) == {41: 1}
    assert F(42) == {2: 1, 3: 1, 7: 1}
    assert F(43) == {43: 1}
    assert F(44) == {2: 2, 11: 1}
    assert F(45) == {3: 2, 5: 1}
    assert F(46) == {2: 1, 23: 1}
    assert F(47) == {47: 1}
    assert F(48) == {2: 4, 3: 1}
    assert F(49) == {7: 2}
    assert F(50) == {2: 1, 5: 2}
    assert F(51) == {3: 1, 17: 1}


def test_Rational_factors():
    def F(p, q, visual=None):
        return Rational(p, q).factors(visual=visual)

    assert F(2, 3) == {2: 1, 3: -1}
    assert F(2, 9) == {2: 1, 3: -2}
    assert F(2, 15) == {2: 1, 3: -1, 5: -1}
    assert F(6, 10) == {3: 1, 5: -1}


def test_issue_4107():
    assert pi*(E + 10) + pi*(-E - 10) != 0
    assert pi*(E + 10**10) + pi*(-E - 10**10) != 0
    assert pi*(E + 10**20) + pi*(-E - 10**20) != 0
    assert pi*(E + 10**80) + pi*(-E - 10**80) != 0

    assert (pi*(E + 10) + pi*(-E - 10)).expand() == 0
    assert (pi*(E + 10**10) + pi*(-E - 10**10)).expand() == 0
    assert (pi*(E + 10**20) + pi*(-E - 10**20)).expand() == 0
    assert (pi*(E + 10**80) + pi*(-E - 10**80)).expand() == 0


def test_IntegerInteger():
    a = Integer(4)
    b = Integer(a)

    assert a == b


def test_Rational_gcd_lcm_cofactors():
    assert Integer(4).gcd(2) == Integer(2)
    assert Integer(4).lcm(2) == Integer(4)
    assert Integer(4).gcd(Integer(2)) == Integer(2)
    assert Integer(4).lcm(Integer(2)) == Integer(4)
    a, b = 720**99911, 480**12342
    assert Integer(a).lcm(b) == a*b/Integer(a).gcd(b)

    assert Integer(4).gcd(3) == Integer(1)
    assert Integer(4).lcm(3) == Integer(12)
    assert Integer(4).gcd(Integer(3)) == Integer(1)
    assert Integer(4).lcm(Integer(3)) == Integer(12)

    assert Rational(4, 3).gcd(2) == Rational(2, 3)
    assert Rational(4, 3).lcm(2) == Integer(4)
    assert Rational(4, 3).gcd(Integer(2)) == Rational(2, 3)
    assert Rational(4, 3).lcm(Integer(2)) == Integer(4)

    assert Integer(4).gcd(Rational(2, 9)) == Rational(2, 9)
    assert Integer(4).lcm(Rational(2, 9)) == Integer(4)

    assert Rational(4, 3).gcd(Rational(2, 9)) == Rational(2, 9)
    assert Rational(4, 3).lcm(Rational(2, 9)) == Rational(4, 3)
    assert Rational(4, 5).gcd(Rational(2, 9)) == Rational(2, 45)
    assert Rational(4, 5).lcm(Rational(2, 9)) == Integer(4)
    assert Rational(5, 9).lcm(Rational(3, 7)) == Rational(Integer(5).lcm(3),Integer(9).gcd(7))

    assert Integer(4).cofactors(2) == (Integer(2), Integer(2), Integer(1))
    assert Integer(4).cofactors(Integer(2)) == \
        (Integer(2), Integer(2), Integer(1))

    assert Integer(4).gcd(Float(2.0)) == Float(1.0)
    assert Integer(4).lcm(Float(2.0)) == Float(8.0)
    assert Integer(4).cofactors(Float(2.0)) == (Float(1.0), Float(4.0), Float(2.0))

    assert S.Half.gcd(Float(2.0)) == Float(1.0)
    assert S.Half.lcm(Float(2.0)) == Float(1.0)
    assert S.Half.cofactors(Float(2.0)) == \
        (Float(1.0), Float(0.5), Float(2.0))


def test_Float_gcd_lcm_cofactors():
    assert Float(2.0).gcd(Integer(4)) == Float(1.0)
    assert Float(2.0).lcm(Integer(4)) == Float(8.0)
    assert Float(2.0).cofactors(Integer(4)) == (Float(1.0), Float(2.0), Float(4.0))

    assert Float(2.0).gcd(S.Half) == Float(1.0)
    assert Float(2.0).lcm(S.Half) == Float(1.0)
    assert Float(2.0).cofactors(S.Half) == \
        (Float(1.0), Float(2.0), Float(0.5))


def test_issue_4611():
    assert abs(pi._evalf(50) - 3.14159265358979) < 1e-10
    assert abs(E._evalf(50) - 2.71828182845905) < 1e-10
    assert abs(Catalan._evalf(50) - 0.915965594177219) < 1e-10
    assert abs(EulerGamma._evalf(50) - 0.577215664901533) < 1e-10
    assert abs(GoldenRatio._evalf(50) - 1.61803398874989) < 1e-10
    assert abs(TribonacciConstant._evalf(50) - 1.83928675521416) < 1e-10

    x = Symbol("x")
    assert (pi + x).evalf() == pi.evalf() + x
    assert (E + x).evalf() == E.evalf() + x
    assert (Catalan + x).evalf() == Catalan.evalf() + x
    assert (EulerGamma + x).evalf() == EulerGamma.evalf() + x
    assert (GoldenRatio + x).evalf() == GoldenRatio.evalf() + x
    assert (TribonacciConstant + x).evalf() == TribonacciConstant.evalf() + x


@conserve_mpmath_dps
def test_conversion_to_mpmath():
    assert mpmath.mpmathify(Integer(1)) == mpmath.mpf(1)
    assert mpmath.mpmathify(S.Half) == mpmath.mpf(0.5)
    assert mpmath.mpmathify(Float('1.23', 15)) == mpmath.mpf('1.23')

    assert mpmath.mpmathify(I) == mpmath.mpc(1j)

    assert mpmath.mpmathify(1 + 2*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1 + 2.0*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2.0*I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(S.Half + S.Half*I) == mpmath.mpc(0.5 + 0.5j)

    assert mpmath.mpmathify(2*I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(2.0*I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(S.Half*I) == mpmath.mpc(0.5j)

    mpmath.mp.dps = 100
    assert mpmath.mpmathify(pi.evalf(100) + pi.evalf(100)*I) == mpmath.pi + mpmath.pi*mpmath.j
    assert mpmath.mpmathify(pi.evalf(100)*I) == mpmath.pi*mpmath.j


def test_relational():
    # real
    x = S(.1)
    assert (x != cos) is True
    assert (x == cos) is False

    # rational
    x = Rational(1, 3)
    assert (x != cos) is True
    assert (x == cos) is False

    # integer defers to rational so these tests are omitted

    # number symbol
    x = pi
    assert (x != cos) is True
    assert (x == cos) is False


def test_Integer_as_index():
    assert 'hello'[Integer(2):] == 'llo'


def test_Rational_int():
    assert int( Rational(7, 5)) == 1
    assert int( S.Half) == 0
    assert int(Rational(-1, 2)) == 0
    assert int(-Rational(7, 5)) == -1


def test_zoo():
    b = Symbol('b', finite=True)
    nz = Symbol('nz', nonzero=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    im = Symbol('i', imaginary=True)
    c = Symbol('c', complex=True)
    pb = Symbol('pb', positive=True)
    nb = Symbol('nb', negative=True)
    imb = Symbol('ib', imaginary=True, finite=True)
    for i in [I, S.Infinity, S.NegativeInfinity, S.Zero, S.One, S.Pi, S.Half, S(3), log(3),
              b, nz, p, n, im, pb, nb, imb, c]:
        if i.is_finite and (i.is_real or i.is_imaginary):
            assert i + zoo is zoo
            assert i - zoo is zoo
            assert zoo + i is zoo
            assert zoo - i is zoo
        elif i.is_finite is not False:
            assert (i + zoo).is_Add
            assert (i - zoo).is_Add
            assert (zoo + i).is_Add
            assert (zoo - i).is_Add
        else:
            assert (i + zoo) is S.NaN
            assert (i - zoo) is S.NaN
            assert (zoo + i) is S.NaN
            assert (zoo - i) is S.NaN

        if fuzzy_not(i.is_zero) and (i.is_extended_real or i.is_imaginary):
            assert i*zoo is zoo
            assert zoo*i is zoo
        elif i.is_zero:
            assert i*zoo is S.NaN
            assert zoo*i is S.NaN
        else:
            assert (i*zoo).is_Mul
            assert (zoo*i).is_Mul

        if fuzzy_not((1/i).is_zero) and (i.is_real or i.is_imaginary):
            assert zoo/i is zoo
        elif (1/i).is_zero:
            assert zoo/i is S.NaN
        elif i.is_zero:
            assert zoo/i is zoo
        else:
            assert (zoo/i).is_Mul

    assert (I*oo).is_Mul  # allow directed infinity
    assert zoo + zoo is S.NaN
    assert zoo * zoo is zoo
    assert zoo - zoo is S.NaN
    assert zoo/zoo is S.NaN
    assert zoo**zoo is S.NaN
    assert zoo**0 is S.One
    assert zoo**2 is zoo
    assert 1/zoo is S.Zero

    assert Mul.flatten([S.NegativeOne, oo, S(0)]) == ([S.NaN], [], None)


def test_issue_4122():
    x = Symbol('x', nonpositive=True)
    assert oo + x is oo
    x = Symbol('x', extended_nonpositive=True)
    assert (oo + x).is_Add
    x = Symbol('x', finite=True)
    assert (oo + x).is_Add  # x could be imaginary
    x = Symbol('x', nonnegative=True)
    assert oo + x is oo
    x = Symbol('x', extended_nonnegative=True)
    assert oo + x is oo
    x = Symbol('x', finite=True, real=True)
    assert oo + x is oo

    # similarly for negative infinity
    x = Symbol('x', nonnegative=True)
    assert -oo + x is -oo
    x = Symbol('x', extended_nonnegative=True)
    assert (-oo + x).is_Add
    x = Symbol('x', finite=True)
    assert (-oo + x).is_Add
    x = Symbol('x', nonpositive=True)
    assert -oo + x is -oo
    x = Symbol('x', extended_nonpositive=True)
    assert -oo + x is -oo
    x = Symbol('x', finite=True, real=True)
    assert -oo + x is -oo


def test_GoldenRatio_expand():
    assert GoldenRatio.expand(func=True) == S.Half + sqrt(5)/2


def test_TribonacciConstant_expand():
        assert TribonacciConstant.expand(func=True) == \
          (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3


def test_as_content_primitive():
    assert S.Zero.as_content_primitive() == (1, 0)
    assert S.Half.as_content_primitive() == (S.Half, 1)
    assert (Rational(-1, 2)).as_content_primitive() == (S.Half, -1)
    assert S(3).as_content_primitive() == (3, 1)
    assert S(3.1).as_content_primitive() == (1, 3.1)


def test_hashing_sympy_integers():
    # Test for issue 5072
    assert {Integer(3)} == {int(3)}
    assert hash(Integer(4)) == hash(int(4))


def test_rounding_issue_4172():
    assert int((E**100).round()) == \
        26881171418161354484126255515800135873611119
    assert int((pi**100).round()) == \
        51878483143196131920862615246303013562686760680406
    assert int((Rational(1)/EulerGamma**100).round()) == \
        734833795660954410469466


@XFAIL
def test_mpmath_issues():
    from mpmath.libmp.libmpf import _normalize
    import mpmath.libmp as mlib
    rnd = mlib.round_nearest
    mpf = (0, int(0), -123, -1, 53, rnd)  # nan
    assert _normalize(mpf, 53) != (0, int(0), 0, 0)
    mpf = (0, int(0), -456, -2, 53, rnd)  # +inf
    assert _normalize(mpf, 53) != (0, int(0), 0, 0)
    mpf = (1, int(0), -789, -3, 53, rnd)  # -inf
    assert _normalize(mpf, 53) != (0, int(0), 0, 0)

    from mpmath.libmp.libmpf import fnan
    assert mlib.mpf_eq(fnan, fnan)


def test_Catalan_EulerGamma_prec():
    n = GoldenRatio
    f = Float(n.n(), 5)
    assert f._mpf_ == (0, int(212079), -17, 18)
    assert f._prec == 20
    assert n._as_mpf_val(20) == f._mpf_

    n = EulerGamma
    f = Float(n.n(), 5)
    assert f._mpf_ == (0, int(302627), -19, 19)
    assert f._prec == 20
    assert n._as_mpf_val(20) == f._mpf_


def test_Catalan_rewrite():
    k = Dummy('k', integer=True, nonnegative=True)
    assert Catalan.rewrite(Sum).dummy_eq(
            Sum((-1)**k/(2*k + 1)**2, (k, 0, oo)))
    assert Catalan.rewrite() == Catalan


def test_bool_eq():
    assert 0 == False
    assert S(0) == False
    assert S(0) != S.false
    assert 1 == True
    assert S.One == True
    assert S.One != S.true


def test_Float_eq():
    # Floats with different precision should not compare equal
    assert Float(.5, 10) != Float(.5, 11) != Float(.5, 1)
    # but floats that aren't exact in base-2 still
    # don't compare the same because they have different
    # underlying mpf values
    assert Float(.12, 3) != Float(.12, 4)
    assert Float(.12, 3) != .12
    assert 0.12 != Float(.12, 3)
    assert Float('.12', 22) != .12
    # issue 11707
    # but Float/Rational -- except for 0 --
    # are exact so Rational(x) = Float(y) only if
    # Rational(x) == Rational(Float(y))
    assert Float('1.1') != Rational(11, 10)
    assert Rational(11, 10) != Float('1.1')
    # coverage
    assert not Float(3) == 2
    assert not Float(3) == Float(2)
    assert not Float(3) == 3
    assert not Float(2**2) == S.Half
    assert Float(2**2) == 4.0
    assert not Float(2**-2) == 1
    assert Float(2**-1) == 0.5
    assert not Float(2*3) == 3
    assert not Float(2*3) == 0.5
    assert Float(2*3) == 6.0
    assert not Float(2*3) == 6
    assert not Float(2*3) == 8
    assert not Float(.75) == Rational(3, 4)
    assert Float(.75) == 0.75
    assert Float(5/18) == 5/18
    # 4473
    assert Float(2.) != 3
    assert not Float((0,1,-3)) == S.One/8
    assert Float((0,1,-3)) == 1/8
    assert Float((0,1,-3)) != S.One/9
    # 16196
    assert not 2 == Float(2)  # unlike Python
    assert t**2 != t**2.0


def test_issue_6640():
    from mpmath.libmp.libmpf import finf, fninf
    # fnan is not included because Float no longer returns fnan,
    # but otherwise, the same sort of test could apply
    assert Float(finf).is_zero is False
    assert Float(fninf).is_zero is False
    assert bool(Float(0)) is False


def test_issue_6349():
    assert Float('23.e3', '')._prec == 10
    assert Float('23e3', '')._prec == 20
    assert Float('23000', '')._prec == 20
    assert Float('-23000', '')._prec == 20


def test_mpf_norm():
    assert mpf_norm((1, 0, 1, 0), 10) == mpf('0')._mpf_
    assert Float._new((1, 0, 1, 0), 10)._mpf_ == mpf('0')._mpf_


def test_latex():
    assert latex(pi) == r"\pi"
    assert latex(E) == r"e"
    assert latex(GoldenRatio) == r"\phi"
    assert latex(TribonacciConstant) == r"\text{TribonacciConstant}"
    assert latex(EulerGamma) == r"\gamma"
    assert latex(oo) == r"\infty"
    assert latex(-oo) == r"-\infty"
    assert latex(zoo) == r"\tilde{\infty}"
    assert latex(nan) == r"\text{NaN}"
    assert latex(I) == r"i"


def test_issue_7742():
    assert -oo % 1 is nan


def test_simplify_AlgebraicNumber():
    A = AlgebraicNumber
    e = 3**(S.One/6)*(3 + (135 + 78*sqrt(3))**Rational(2, 3))/(45 + 26*sqrt(3))**(S.One/3)
    assert simplify(A(e)) == A(12)  # wester test_C20

    e = (41 + 29*sqrt(2))**(S.One/5)
    assert simplify(A(e)) == A(1 + sqrt(2))  # wester test_C21

    e = (3 + 4*I)**Rational(3, 2)
    assert simplify(A(e)) == A(2 + 11*I)  # issue 4401


def test_Float_idempotence():
    x = Float('1.23', '')
    y = Float(x)
    z = Float(x, 15)
    assert same_and_same_prec(y, x)
    assert not same_and_same_prec(z, x)
    x = Float(10**20)
    y = Float(x)
    z = Float(x, 15)
    assert same_and_same_prec(y, x)
    assert not same_and_same_prec(z, x)


def test_comp1():
    # sqrt(2) = 1.414213 5623730950...
    a = sqrt(2).n(7)
    assert comp(a, 1.4142129) is False
    assert comp(a, 1.4142130)
    #                  ...
    assert comp(a, 1.4142141)
    assert comp(a, 1.4142142) is False
    assert comp(sqrt(2).n(2), '1.4')
    assert comp(sqrt(2).n(2), Float(1.4, 2), '')
    assert comp(sqrt(2).n(2), 1.4, '')
    assert comp(sqrt(2).n(2), Float(1.4, 3), '') is False
    assert comp(sqrt(2) + sqrt(3)*I, 1.4 + 1.7*I, .1)
    assert not comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*0.89, .1)
    assert comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*0.90, .1)
    assert comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*1.07, .1)
    assert not comp(sqrt(2) + sqrt(3)*I, (1.5 + 1.7*I)*1.08, .1)
    assert [(i, j)
            for i in range(130, 150)
            for j in range(170, 180)
            if comp((sqrt(2)+ I*sqrt(3)).n(3), i/100. + I*j/100.)] == [
        (141, 173), (142, 173)]
    raises(ValueError, lambda: comp(t, '1'))
    raises(ValueError, lambda: comp(t, 1))
    assert comp(0, 0.0)
    assert comp(.5, S.Half)
    assert comp(2 + sqrt(2), 2.0 + sqrt(2))
    assert not comp(0, 1)
    assert not comp(2, sqrt(2))
    assert not comp(2 + I, 2.0 + sqrt(2))
    assert not comp(2.0 + sqrt(2), 2 + I)
    assert not comp(2.0 + sqrt(2), sqrt(3))
    assert comp(1/pi.n(4), 0.3183, 1e-5)
    assert not comp(1/pi.n(4), 0.3183, 8e-6)


def test_issue_9491():
    assert oo**zoo is nan


def test_issue_10063():
    assert 2**Float(3) == Float(8)


def test_issue_10020():
    assert oo**I is S.NaN
    assert oo**(1 + I) is S.ComplexInfinity
    assert oo**(-1 + I) is S.Zero
    assert (-oo)**I is S.NaN
    assert (-oo)**(-1 + I) is S.Zero
    assert oo**t == Pow(oo, t, evaluate=False)
    assert (-oo)**t == Pow(-oo, t, evaluate=False)


def test_invert_numbers():
    assert S(2).invert(5) == 3
    assert S(2).invert(Rational(5, 2)) == S.Half
    assert S(2).invert(5.) == S.Half
    assert S(2).invert(S(5)) == 3
    assert S(2.).invert(5) == 0.5
    assert S(sqrt(2)).invert(5) == 1/sqrt(2)
    assert S(sqrt(2)).invert(sqrt(3)) == 1/sqrt(2)


def test_mod_inverse():
    assert mod_inverse(3, 11) == 4
    assert mod_inverse(5, 11) == 9
    assert mod_inverse(21124921, 521512) == 7713
    assert mod_inverse(124215421, 5125) == 2981
    assert mod_inverse(214, 12515) == 1579
    assert mod_inverse(5823991, 3299) == 1442
    assert mod_inverse(123, 44) == 39
    assert mod_inverse(2, 5) == 3
    assert mod_inverse(-2, 5) == 2
    assert mod_inverse(2, -5) == -2
    assert mod_inverse(-2, -5) == -3
    assert mod_inverse(-3, -7) == -5
    x = Symbol('x')
    assert S(2).invert(x) == S.Half
    raises(TypeError, lambda: mod_inverse(2, x))
    raises(ValueError, lambda: mod_inverse(2, S.Half))
    raises(ValueError, lambda: mod_inverse(2, cos(1)**2 + sin(1)**2))


def test_golden_ratio_rewrite_as_sqrt():
    assert GoldenRatio.rewrite(sqrt) == S.Half + sqrt(5)*S.Half


def test_tribonacci_constant_rewrite_as_sqrt():
    assert TribonacciConstant.rewrite(sqrt) == \
      (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3


def test_comparisons_with_unknown_type():
    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.

        """

    ni, nf, nr = Integer(3), Float(1.0), Rational(1, 3)
    foo = Foo()

    for n in ni, nf, nr, oo, -oo, zoo, nan:
        assert n != foo
        assert foo != n
        assert not n == foo
        assert not foo == n
        raises(TypeError, lambda: n < foo)
        raises(TypeError, lambda: foo > n)
        raises(TypeError, lambda: n > foo)
        raises(TypeError, lambda: foo < n)
        raises(TypeError, lambda: n <= foo)
        raises(TypeError, lambda: foo >= n)
        raises(TypeError, lambda: n >= foo)
        raises(TypeError, lambda: foo <= n)

    class Bar:
        """
        Class that considers itself equal to any instance of Number except
        infinities and nans, and relies on SymPy types returning the
        NotImplemented singleton for symmetric equality relations.

        """
        def __eq__(self, other):
            if other in (oo, -oo, zoo, nan):
                return False
            if isinstance(other, Number):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other

    bar = Bar()

    for n in ni, nf, nr:
        assert n == bar
        assert bar == n
        assert not n != bar
        assert not bar != n

    for n in oo, -oo, zoo, nan:
        assert n != bar
        assert bar != n
        assert not n == bar
        assert not bar == n

    for n in ni, nf, nr, oo, -oo, zoo, nan:
        raises(TypeError, lambda: n < bar)
        raises(TypeError, lambda: bar > n)
        raises(TypeError, lambda: n > bar)
        raises(TypeError, lambda: bar < n)
        raises(TypeError, lambda: n <= bar)
        raises(TypeError, lambda: bar >= n)
        raises(TypeError, lambda: n >= bar)
        raises(TypeError, lambda: bar <= n)


def test_NumberSymbol_comparison():
    from sympy.core.tests.test_relational import rel_check
    rpi = Rational('905502432259640373/288230376151711744')
    fpi = Float(float(pi))
    assert rel_check(rpi, fpi)


def test_Integer_precision():
    # Make sure Integer inputs for keyword args work
    assert Float('1.0', dps=Integer(15))._prec == 53
    assert Float('1.0', precision=Integer(15))._prec == 15
    assert type(Float('1.0', precision=Integer(15))._prec) == int
    assert sympify(srepr(Float('1.0', precision=15))) == Float('1.0', precision=15)


def test_numpy_to_float():
    from sympy.testing.pytest import skip
    from sympy.external import import_module
    np = import_module('numpy')
    if not np:
        skip('numpy not installed. Abort numpy tests.')

    def check_prec_and_relerr(npval, ratval):
        prec = np.finfo(npval).nmant + 1
        x = Float(npval)
        assert x._prec == prec
        y = Float(ratval, precision=prec)
        assert abs((x - y)/y) < 2**(-(prec + 1))

    check_prec_and_relerr(np.float16(2.0/3), Rational(2, 3))
    check_prec_and_relerr(np.float32(2.0/3), Rational(2, 3))
    check_prec_and_relerr(np.float64(2.0/3), Rational(2, 3))
    # extended precision, on some arch/compilers:
    x = np.longdouble(2)/3
    check_prec_and_relerr(x, Rational(2, 3))
    y = Float(x, precision=10)
    assert same_and_same_prec(y, Float(Rational(2, 3), precision=10))

    raises(TypeError, lambda: Float(np.complex64(1+2j)))
    raises(TypeError, lambda: Float(np.complex128(1+2j)))


def test_Integer_ceiling_floor():
    a = Integer(4)

    assert a.floor() == a
    assert a.ceiling() == a


def test_ComplexInfinity():
    assert zoo.floor() is zoo
    assert zoo.ceiling() is zoo
    assert zoo**zoo is S.NaN


def test_Infinity_floor_ceiling_power():
    assert oo.floor() is oo
    assert oo.ceiling() is oo
    assert oo**S.NaN is S.NaN
    assert oo**zoo is S.NaN


def test_One_power():
    assert S.One**12 is S.One
    assert S.NegativeOne**S.NaN is S.NaN


def test_NegativeInfinity():
    assert (-oo).floor() is -oo
    assert (-oo).ceiling() is -oo
    assert (-oo)**11 is -oo
    assert (-oo)**12 is oo


def test_issue_6133():
    raises(TypeError, lambda: (-oo < None))
    raises(TypeError, lambda: (S(-2) < None))
    raises(TypeError, lambda: (oo < None))
    raises(TypeError, lambda: (oo > None))
    raises(TypeError, lambda: (S(2) < None))


def test_abc():
    x = numbers.Float(5)
    assert(isinstance(x, nums.Number))
    assert(isinstance(x, numbers.Number))
    assert(isinstance(x, nums.Real))
    y = numbers.Rational(1, 3)
    assert(isinstance(y, nums.Number))
    assert(y.numerator == 1)
    assert(y.denominator == 3)
    assert(isinstance(y, nums.Rational))
    z = numbers.Integer(3)
    assert(isinstance(z, nums.Number))
    assert(isinstance(z, numbers.Number))
    assert(isinstance(z, nums.Rational))
    assert(isinstance(z, numbers.Rational))
    assert(isinstance(z, nums.Integral))


def test_floordiv():
    assert S(2)//S.Half == 4


def test_negation():
    assert -S.Zero is S.Zero
    assert -Float(0) is not S.Zero and -Float(0) == 0.0


def test_exponentiation_of_0():
    x = Symbol('x')
    assert 0**-x == zoo**x
    assert unchanged(Pow, 0, x)
    x = Symbol('x', zero=True)
    assert 0**-x == S.One
    assert 0**x == S.One


def test_int_valued():
    x = Symbol('x')
    assert int_valued(x) == False
    assert int_valued(S.Half) == False
    assert int_valued(S.One) == True
    assert int_valued(Float(1)) == True
    assert int_valued(Float(1.1)) == False
    assert int_valued(pi) == False


def test_equal_valued():
    x = Symbol('x')

    equal_values = [
        [1, 1.0, S(1), S(1.0), S(1).n(5)],
        [2, 2.0, S(2), S(2.0), S(2).n(5)],
        [-1, -1.0, -S(1), -S(1.0), -S(1).n(5)],
        [0.5, S(0.5), S(1)/2],
        [-0.5, -S(0.5), -S(1)/2],
        [0, 0.0, S(0), S(0.0), S(0).n()],
        [pi], [pi.n()],           # <-- not equal
        [S(1)/10], [0.1, S(0.1)], # <-- not equal
        [S(0.1).n(5)],
        [oo],
        [cos(x/2)], [cos(0.5*x)], # <-- no recursion
    ]

    for m, values_m in enumerate(equal_values):
        for value_i in values_m:

            # All values in same list equal
            for value_j in values_m:
                assert equal_valued(value_i, value_j) is True

            # Not equal to anything in any other list:
            for n, values_n in enumerate(equal_values):
                if n == m:
                    continue
                for value_j in values_n:
                    assert equal_valued(value_i, value_j) is False


def test_all_close():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert all_close(2, 2) is True
    assert all_close(2, 2.0000) is True
    assert all_close(2, 2.0001) is False
    assert all_close(1/3, 1/3.0001) is False
    assert all_close(1/3, 1/3.0001, 1e-3, 1e-3) is True
    assert all_close(1/3, Rational(1, 3)) is True
    assert all_close(0.1*exp(0.2*x), exp(x/5)/10) is True
    # The expressions should be structurally the same modulo identity:
    assert all_close(1.4142135623730951, sqrt(2)) is False
    assert all_close(1.4142135623730951, sqrt(2).evalf()) is True
    assert all_close(x + 1e-20, x) is True
    # We should be able to match terms of an Add/Mul in any order
    assert all_close(Add(1, 2, evaluate=False), Add(2, 1, evaluate=False))
    # coverage
    assert not all_close(2*x, 3*x)
    assert all_close(2*x, 3*x, 1)
    assert not all_close(2*x, 3*x, 0, 0.5)
    assert all_close(2*x, 3*x, 0, 1)
    assert not all_close(y*x, z*x)
    assert all_close(2*x*exp(1.0*x), 2.0*x*exp(x))
    assert not all_close(2*x*exp(1.0*x), 2.0*x*exp(2.*x))
    assert all_close(x + 2.*y, 1.*x + 2*y)
    assert all_close(x + exp(2.*x)*y, 1.*x + exp(2*x)*y)
    assert not all_close(x + exp(2.*x)*y, 1.*x + 2*exp(2*x)*y)
    assert not all_close(x + exp(2.*x)*y, 1.*x + exp(3*x)*y)
    assert not all_close(x + 2.*y, 1.*x + 3*y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\utilities.py ===
from contextlib import contextmanager
from threading import local

from sympy.core.function import expand_mul


class DotProdSimpState(local):
    def __init__(self):
        self.state = None

_dotprodsimp_state = DotProdSimpState()

@contextmanager
def dotprodsimp(x):
    old = _dotprodsimp_state.state

    try:
        _dotprodsimp_state.state = x
        yield
    finally:
        _dotprodsimp_state.state = old


def _dotprodsimp(expr, withsimp=False):
    """Wrapper for simplify.dotprodsimp to avoid circular imports."""
    from sympy.simplify.simplify import dotprodsimp as dps
    return dps(expr, withsimp=withsimp)


def _get_intermediate_simp(deffunc=lambda x: x, offfunc=lambda x: x,
        onfunc=_dotprodsimp, dotprodsimp=None):
    """Support function for controlling intermediate simplification. Returns a
    simplification function according to the global setting of dotprodsimp
    operation.

    ``deffunc``     - Function to be used by default.
    ``offfunc``     - Function to be used if dotprodsimp has been turned off.
    ``onfunc``      - Function to be used if dotprodsimp has been turned on.
    ``dotprodsimp`` - True, False or None. Will be overridden by global
                      _dotprodsimp_state.state if that is not None.
    """

    if dotprodsimp is False or _dotprodsimp_state.state is False:
        return offfunc
    if dotprodsimp is True or _dotprodsimp_state.state is True:
        return onfunc

    return deffunc # None, None


def _get_intermediate_simp_bool(default=False, dotprodsimp=None):
    """Same as ``_get_intermediate_simp`` but returns bools instead of functions
    by default."""

    return _get_intermediate_simp(default, False, True, dotprodsimp)


def _iszero(x):
    """Returns True if x is zero."""
    return getattr(x, 'is_zero', None)


def _is_zero_after_expand_mul(x):
    """Tests by expand_mul only, suitable for polynomials and rational
    functions."""
    return expand_mul(x) == 0


def _simplify(expr):
    """ Wrapper to avoid circular imports. """
    from sympy.simplify.simplify import simplify
    return simplify(expr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\__init__.py ===
"""A module that handles matrices.

Includes functions for fast creating matrices like zero, one/eye, random
matrix, etc.
"""
from .exceptions import ShapeError, NonSquareMatrixError
from .kind import MatrixKind
from .dense import (
    GramSchmidt, casoratian, diag, eye, hessian, jordan_cell,
    list2numpy, matrix2numpy, matrix_multiply_elementwise, ones,
    randMatrix, rot_axis1, rot_axis2, rot_axis3, rot_ccw_axis1,
    rot_ccw_axis2, rot_ccw_axis3, rot_givens,
    symarray, wronskian, zeros)
from .dense import MutableDenseMatrix
from .matrixbase import DeferredVector, MatrixBase

MutableMatrix = MutableDenseMatrix
Matrix = MutableMatrix

from .sparse import MutableSparseMatrix
from .sparsetools import banded
from .immutable import ImmutableDenseMatrix, ImmutableSparseMatrix

ImmutableMatrix = ImmutableDenseMatrix
SparseMatrix = MutableSparseMatrix

from .expressions import (
    MatrixSlice, BlockDiagMatrix, BlockMatrix, FunctionMatrix, Identity,
    Inverse, MatAdd, MatMul, MatPow, MatrixExpr, MatrixSymbol, Trace,
    Transpose, ZeroMatrix, OneMatrix, blockcut, block_collapse, matrix_symbols, Adjoint,
    hadamard_product, HadamardProduct, HadamardPower, Determinant, det,
    diagonalize_vector, DiagMatrix, DiagonalMatrix, DiagonalOf, trace,
    DotProduct, kronecker_product, KroneckerProduct,
    PermutationMatrix, MatrixPermute, MatrixSet, Permanent, per)

from .utilities import dotprodsimp

__all__ = [
    'ShapeError', 'NonSquareMatrixError', 'MatrixKind',

    'GramSchmidt', 'casoratian', 'diag', 'eye', 'hessian', 'jordan_cell',
    'list2numpy', 'matrix2numpy', 'matrix_multiply_elementwise', 'ones',
    'randMatrix', 'rot_axis1', 'rot_axis2', 'rot_axis3', 'symarray',
    'wronskian', 'zeros', 'rot_ccw_axis1', 'rot_ccw_axis2', 'rot_ccw_axis3',
    'rot_givens',

    'MutableDenseMatrix',

    'DeferredVector', 'MatrixBase',

    'Matrix', 'MutableMatrix',

    'MutableSparseMatrix',

    'banded',

    'ImmutableDenseMatrix', 'ImmutableSparseMatrix',

    'ImmutableMatrix', 'SparseMatrix',

    'MatrixSlice', 'BlockDiagMatrix', 'BlockMatrix', 'FunctionMatrix',
    'Identity', 'Inverse', 'MatAdd', 'MatMul', 'MatPow', 'MatrixExpr',
    'MatrixSymbol', 'Trace', 'Transpose', 'ZeroMatrix', 'OneMatrix',
    'blockcut', 'block_collapse', 'matrix_symbols', 'Adjoint',
    'hadamard_product', 'HadamardProduct', 'HadamardPower', 'Determinant',
    'det', 'diagonalize_vector', 'DiagMatrix', 'DiagonalMatrix',
    'DiagonalOf', 'trace', 'DotProduct', 'kronecker_product',
    'KroneckerProduct', 'PermutationMatrix', 'MatrixPermute', 'MatrixSet',
    'Permanent', 'per',

    'dotprodsimp',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\piab.py ===
"""1D quantum particle in a box."""

from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.sets.sets import Interval

from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.constants import hbar
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import L2

m = Symbol('m')
L = Symbol('L')


__all__ = [
    'PIABHamiltonian',
    'PIABKet',
    'PIABBra'
]


class PIABHamiltonian(HermitianOperator):
    """Particle in a box Hamiltonian operator."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PIABKet(self, ket, **options):
        n = ket.label[0]
        return (n**2*pi**2*hbar**2)/(2*m*L**2)*ket


class PIABKet(Ket):
    """Particle in a box eigenket."""

    @classmethod
    def _eval_hilbert_space(cls, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    @classmethod
    def dual_class(self):
        return PIABBra

    def _represent_default_basis(self, **options):
        return self._represent_XOp(None, **options)

    def _represent_XOp(self, basis, **options):
        x = Symbol('x')
        n = Symbol('n')
        subs_info = options.get('subs', {})
        return sqrt(2/L)*sin(n*pi*x/L).subs(subs_info)

    def _eval_innerproduct_PIABBra(self, bra):
        return KroneckerDelta(bra.label[0], self.label[0])


class PIABBra(Bra):
    """Particle in a box eigenbra."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    @classmethod
    def dual_class(self):
        return PIABKet

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sandbox\indexed_integrals.py ===
from sympy.tensor import Indexed
from sympy.core.containers import Tuple
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.integrals.integrals import Integral


class IndexedIntegral(Integral):
    """
    Experimental class to test integration by indexed variables.

    Usage is analogue to ``Integral``, it simply adds awareness of
    integration over indices.

    Contraction of non-identical index symbols referring to the same
    ``IndexedBase`` is not yet supported.

    Examples
    ========

    >>> from sympy.sandbox.indexed_integrals import IndexedIntegral
    >>> from sympy import IndexedBase, symbols
    >>> A = IndexedBase('A')
    >>> i, j = symbols('i j', integer=True)
    >>> ii = IndexedIntegral(A[i], A[i])
    >>> ii
    Integral(_A[i], _A[i])
    >>> ii.doit()
    A[i]**2/2

    If the indices are different, indexed objects are considered to be
    different variables:

    >>> i2 = IndexedIntegral(A[j], A[i])
    >>> i2
    Integral(A[j], _A[i])
    >>> i2.doit()
    A[i]*A[j]
    """

    def __new__(cls, function, *limits, **assumptions):
        repl, limits = IndexedIntegral._indexed_process_limits(limits)
        function = sympify(function)
        function = function.xreplace(repl)
        obj = Integral.__new__(cls, function, *limits, **assumptions)
        obj._indexed_repl = repl
        obj._indexed_reverse_repl = {val: key for key, val in repl.items()}
        return obj

    def doit(self):
        res = super().doit()
        return res.xreplace(self._indexed_reverse_repl)

    @staticmethod
    def _indexed_process_limits(limits):
        repl = {}
        newlimits = []
        for i in limits:
            if isinstance(i, (tuple, list, Tuple)):
                v = i[0]
                vrest = i[1:]
            else:
                v = i
                vrest = ()
            if isinstance(v, Indexed):
                if v not in repl:
                    r = Dummy(str(v))
                    repl[v] = r
                newlimits.append((r,)+vrest)
            else:
                newlimits.append(i)
        return repl, newlimits

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\scalar.py ===
from sympy.core import AtomicExpr, Symbol, S
from sympy.core.sympify import _sympify
from sympy.printing.pretty.stringpict import prettyForm
from sympy.printing.precedence import PRECEDENCE
from sympy.core.kind import NumberKind


class BaseScalar(AtomicExpr):
    """
    A coordinate symbol/base scalar.

    Ideally, users should not instantiate this class.

    """

    kind = NumberKind

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        from sympy.vector.coordsysrect import CoordSys3D
        if pretty_str is None:
            pretty_str = "x{}".format(index)
        elif isinstance(pretty_str, Symbol):
            pretty_str = pretty_str.name
        if latex_str is None:
            latex_str = "x_{}".format(index)
        elif isinstance(latex_str, Symbol):
            latex_str = latex_str.name

        index = _sympify(index)
        system = _sympify(system)
        obj = super().__new__(cls, index, system)
        if not isinstance(system, CoordSys3D):
            raise TypeError("system should be a CoordSys3D")
        if index not in range(0, 3):
            raise ValueError("Invalid index specified.")
        # The _id is used for equating purposes, and for hashing
        obj._id = (index, system)
        obj._name = obj.name = system._name + '.' + system._variable_names[index]
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system

        return obj

    is_commutative = True
    is_symbol = True

    @property
    def free_symbols(self):
        return {self}

    _diff_wrt = True

    def _eval_derivative(self, s):
        if self == s:
            return S.One
        return S.Zero

    def _latex(self, printer=None):
        return self._latex_form

    def _pretty(self, printer=None):
        return prettyForm(self._pretty_form)

    precedence = PRECEDENCE['Atom']

    @property
    def system(self):
        return self._system

    def _sympystr(self, printer):
        return self._name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_abc.py ===
from __future__ import annotations

import attrs
import pytest

from .. import abc as tabc
from ..lowlevel import Task


def test_instrument_implements_hook_methods() -> None:
    attrs = {
        "before_run": (),
        "after_run": (),
        "task_spawned": (Task,),
        "task_scheduled": (Task,),
        "before_task_step": (Task,),
        "after_task_step": (Task,),
        "task_exited": (Task,),
        "before_io_wait": (3.3,),
        "after_io_wait": (3.3,),
    }

    mayonnaise = tabc.Instrument()

    for method_name, args in attrs.items():
        assert hasattr(mayonnaise, method_name)
        method = getattr(mayonnaise, method_name)
        assert callable(method)
        method(*args)


async def test_AsyncResource_defaults() -> None:
    @attrs.define(slots=False)
    class MyAR(tabc.AsyncResource):
        record: list[str] = attrs.Factory(list)

        async def aclose(self) -> None:
            self.record.append("ac")

    async with MyAR() as myar:
        assert isinstance(myar, MyAR)
        assert myar.record == []

    assert myar.record == ["ac"]


def test_abc_generics() -> None:
    # Pythons below 3.5.2 had a typing.Generic that would throw
    # errors when instantiating or subclassing a parameterized
    # version of a class with any __slots__. This is why RunVar
    # (which has slots) is not generic. This tests that
    # the generic ABCs are fine, because while they are slotted
    # they don't actually define any slots.

    class SlottedChannel(tabc.SendChannel[tabc.Stream]):
        __slots__ = ("x",)

        def send_nowait(self, value: object) -> None:
            raise RuntimeError

        async def send(self, value: object) -> None:
            raise RuntimeError  # pragma: no cover

        def clone(self) -> None:
            raise RuntimeError  # pragma: no cover

        async def aclose(self) -> None:
            pass  # pragma: no cover

    channel = SlottedChannel()
    with pytest.raises(RuntimeError):
        channel.send_nowait(None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\i18n.py ===
import os


def messages_path():
    """
    Determine the path to the 'messages' directory as best possible.
    """
    module_path = os.path.abspath(__file__)
    locale_path = os.path.join(os.path.dirname(module_path), "locale")
    if not os.path.exists(locale_path):  # pragma: no cover
        locale_path = "/usr/share/locale"
    return locale_path


def get_builtin_gnu_translations(languages=None):
    """
    Get a gettext.GNUTranslations object pointing at the
    included translation files.

    :param languages:
        A list of languages to try, in order. If omitted or None, then
        gettext will try to use locale information from the environment.
    """
    import gettext

    return gettext.translation("wtforms", messages_path(), languages)


def get_translations(languages=None, getter=get_builtin_gnu_translations):
    """
    Get a WTForms translation object which wraps a low-level translations object.

    :param languages:
        A sequence of languages to try, in order.
    :param getter:
        A single-argument callable which returns a low-level translations object.
    """
    return getter(languages)


class DefaultTranslations:
    """
    A WTForms translations object to wrap translations objects which use
    ugettext/ungettext.
    """

    def __init__(self, translations):
        self.translations = translations

    def gettext(self, string):
        return self.translations.ugettext(string)

    def ngettext(self, singular, plural, n):
        return self.translations.ungettext(singular, plural, n)


class DummyTranslations:
    """
    A translations object which simply returns unmodified strings.

    This is typically used when translations are disabled or if no valid
    translations provider can be found.
    """

    def gettext(self, string):
        return string

    def ngettext(self, singular, plural, n):
        if n == 1:
            return singular

        return plural

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_expr.py ===
from sympy.assumptions.refine import refine
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import (ExprBuilder, unchanged, Expr,
    UnevaluatedExpr)
from sympy.core.function import (Function, expand, WildFunction,
    AppliedUndef, Derivative, diff, Subs)
from sympy.core.mul import Mul, _unevaluated_Mul
from sympy.core.numbers import (NumberSymbol, E, zoo, oo, Float, I,
    Rational, nan, Integer, Number, pi, _illegal)
from sympy.core.power import Pow
from sympy.core.relational import Ge, Lt, Gt, Le
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols, Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import sinh, tanh
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import tan, sin, cos
from sympy.functions.special.delta_functions import (Heaviside,
    DiracDelta)
from sympy.functions.special.error_functions import Si
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate, Integral
from sympy.physics.secondquant import FockState
from sympy.polys.partfrac import apart
from sympy.polys.polytools import factor, cancel, Poly
from sympy.polys.rationaltools import together
from sympy.series.order import O
from sympy.sets.sets import FiniteSet
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import collect, radsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify, nsimplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import Indexed
from sympy.physics.units import meter

from sympy.testing.pytest import raises, XFAIL

from sympy.abc import a, b, c, n, t, u, x, y, z


f, g, h = symbols('f,g,h', cls=Function)


class DummyNumber:
    """
    Minimal implementation of a number that works with SymPy.

    If one has a Number class (e.g. Sage Integer, or some other custom class)
    that one wants to work well with SymPy, one has to implement at least the
    methods of this class DummyNumber, resp. its subclasses I5 and F1_1.

    Basically, one just needs to implement either __int__() or __float__() and
    then one needs to make sure that the class works with Python integers and
    with itself.
    """

    def __radd__(self, a):
        if isinstance(a, (int, float)):
            return a + self.number
        return NotImplemented

    def __add__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number + a
        return NotImplemented

    def __rsub__(self, a):
        if isinstance(a, (int, float)):
            return a - self.number
        return NotImplemented

    def __sub__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number - a
        return NotImplemented

    def __rmul__(self, a):
        if isinstance(a, (int, float)):
            return a * self.number
        return NotImplemented

    def __mul__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number * a
        return NotImplemented

    def __rtruediv__(self, a):
        if isinstance(a, (int, float)):
            return a / self.number
        return NotImplemented

    def __truediv__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number / a
        return NotImplemented

    def __rpow__(self, a):
        if isinstance(a, (int, float)):
            return a ** self.number
        return NotImplemented

    def __pow__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number ** a
        return NotImplemented

    def __pos__(self):
        return self.number

    def __neg__(self):
        return - self.number


class I5(DummyNumber):
    number = 5

    def __int__(self):
        return self.number


class F1_1(DummyNumber):
    number = 1.1

    def __float__(self):
        return self.number

i5 = I5()
f1_1 = F1_1()

# basic SymPy objects
basic_objs = [
    Rational(2),
    Float("1.3"),
    x,
    y,
    pow(x, y)*y,
]

# all supported objects
all_objs = basic_objs + [
    5,
    5.5,
    i5,
    f1_1
]


def dotest(s):
    for xo in all_objs:
        for yo in all_objs:
            s(xo, yo)
    return True


def test_basic():
    def j(a, b):
        x = a
        x = +a
        x = -a
        x = a + b
        x = a - b
        x = a*b
        x = a/b
        x = a**b
        del x
    assert dotest(j)


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


class NonBasic:
    '''This class represents an object that knows how to implement binary
    operations like +, -, etc with Expr but is not a subclass of Basic itself.
    The NonExpr subclass below does subclass Basic but not Expr.

    For both NonBasic and NonExpr it should be possible for them to override
    Expr.__add__ etc because Expr.__add__ should be returning NotImplemented
    for non Expr classes. Otherwise Expr.__add__ would create meaningless
    objects like Add(Integer(1), FiniteSet(2)) and it wouldn't be possible for
    other classes to override these operations when interacting with Expr.
    '''
    def __add__(self, other):
        return SpecialOp('+', self, other)

    def __radd__(self, other):
        return SpecialOp('+', other, self)

    def __sub__(self, other):
        return SpecialOp('-', self, other)

    def __rsub__(self, other):
        return SpecialOp('-', other, self)

    def __mul__(self, other):
        return SpecialOp('*', self, other)

    def __rmul__(self, other):
        return SpecialOp('*', other, self)

    def __truediv__(self, other):
        return SpecialOp('/', self, other)

    def __rtruediv__(self, other):
        return SpecialOp('/', other, self)

    def __floordiv__(self, other):
        return SpecialOp('//', self, other)

    def __rfloordiv__(self, other):
        return SpecialOp('//', other, self)

    def __mod__(self, other):
        return SpecialOp('%', self, other)

    def __rmod__(self, other):
        return SpecialOp('%', other, self)

    def __divmod__(self, other):
        return SpecialOp('divmod', self, other)

    def __rdivmod__(self, other):
        return SpecialOp('divmod', other, self)

    def __pow__(self, other):
        return SpecialOp('**', self, other)

    def __rpow__(self, other):
        return SpecialOp('**', other, self)

    def __lt__(self, other):
        return SpecialOp('<', self, other)

    def __gt__(self, other):
        return SpecialOp('>', self, other)

    def __le__(self, other):
        return SpecialOp('<=', self, other)

    def __ge__(self, other):
        return SpecialOp('>=', self, other)


class NonExpr(Basic, NonBasic):
    '''Like NonBasic above except this is a subclass of Basic but not Expr'''
    pass


class SpecialOp():
    '''Represents the results of operations with NonBasic and NonExpr'''
    def __new__(cls, op, arg1, arg2):
        obj = object.__new__(cls)
        obj.args = (op, arg1, arg2)
        return obj


class NonArithmetic(Basic):
    '''Represents a Basic subclass that does not support arithmetic operations'''
    pass


def test_cooperative_operations():
    '''Tests that Expr uses binary operations cooperatively.

    In particular it should be possible for non-Expr classes to override
    binary operators like +, - etc when used with Expr instances. This should
    work for non-Expr classes whether they are Basic subclasses or not. Also
    non-Expr classes that do not define binary operators with Expr should give
    TypeError.
    '''
    # A bunch of instances of Expr subclasses
    exprs = [
        Expr(),
        S.Zero,
        S.One,
        S.Infinity,
        S.NegativeInfinity,
        S.ComplexInfinity,
        S.Half,
        Float(0.5),
        Integer(2),
        Symbol('x'),
        Mul(2, Symbol('x')),
        Add(2, Symbol('x')),
        Pow(2, Symbol('x')),
    ]

    for e in exprs:
        # Test that these classes can override arithmetic operations in
        # combination with various Expr types.
        for ne in [NonBasic(), NonExpr()]:

            results = [
                (ne + e, ('+', ne, e)),
                (e + ne, ('+', e, ne)),
                (ne - e, ('-', ne, e)),
                (e - ne, ('-', e, ne)),
                (ne * e, ('*', ne, e)),
                (e * ne, ('*', e, ne)),
                (ne / e, ('/', ne, e)),
                (e / ne, ('/', e, ne)),
                (ne // e, ('//', ne, e)),
                (e // ne, ('//', e, ne)),
                (ne % e, ('%', ne, e)),
                (e % ne, ('%', e, ne)),
                (divmod(ne, e), ('divmod', ne, e)),
                (divmod(e, ne), ('divmod', e, ne)),
                (ne ** e, ('**', ne, e)),
                (e ** ne, ('**', e, ne)),
                (e < ne, ('>', ne, e)),
                (ne < e, ('<', ne, e)),
                (e > ne, ('<', ne, e)),
                (ne > e, ('>', ne, e)),
                (e <= ne, ('>=', ne, e)),
                (ne <= e, ('<=', ne, e)),
                (e >= ne, ('<=', ne, e)),
                (ne >= e, ('>=', ne, e)),
            ]

            for res, args in results:
                assert type(res) is SpecialOp and res.args == args

        # These classes do not support binary operators with Expr. Every
        # operation should raise in combination with any of the Expr types.
        for na in [NonArithmetic(), object()]:

            raises(TypeError, lambda : e + na)
            raises(TypeError, lambda : na + e)
            raises(TypeError, lambda : e - na)
            raises(TypeError, lambda : na - e)
            raises(TypeError, lambda : e * na)
            raises(TypeError, lambda : na * e)
            raises(TypeError, lambda : e / na)
            raises(TypeError, lambda : na / e)
            raises(TypeError, lambda : e // na)
            raises(TypeError, lambda : na // e)
            raises(TypeError, lambda : e % na)
            raises(TypeError, lambda : na % e)
            raises(TypeError, lambda : divmod(e, na))
            raises(TypeError, lambda : divmod(na, e))
            raises(TypeError, lambda : e ** na)
            raises(TypeError, lambda : na ** e)
            raises(TypeError, lambda : e > na)
            raises(TypeError, lambda : na > e)
            raises(TypeError, lambda : e < na)
            raises(TypeError, lambda : na < e)
            raises(TypeError, lambda : e >= na)
            raises(TypeError, lambda : na >= e)
            raises(TypeError, lambda : e <= na)
            raises(TypeError, lambda : na <= e)


def test_relational():
    from sympy.core.relational import Lt
    assert (pi < 3) is S.false
    assert (pi <= 3) is S.false
    assert (pi > 3) is S.true
    assert (pi >= 3) is S.true
    assert (-pi < 3) is S.true
    assert (-pi <= 3) is S.true
    assert (-pi > 3) is S.false
    assert (-pi >= 3) is S.false
    r = Symbol('r', real=True)
    assert (r - 2 < r - 3) is S.false
    assert Lt(x + I, x + I + 2).func == Lt  # issue 8288


def test_relational_assumptions():
    m1 = Symbol("m1", nonnegative=False)
    m2 = Symbol("m2", positive=False)
    m3 = Symbol("m3", nonpositive=False)
    m4 = Symbol("m4", negative=False)
    assert (m1 < 0) == Lt(m1, 0)
    assert (m2 <= 0) == Le(m2, 0)
    assert (m3 > 0) == Gt(m3, 0)
    assert (m4 >= 0) == Ge(m4, 0)
    m1 = Symbol("m1", nonnegative=False, real=True)
    m2 = Symbol("m2", positive=False, real=True)
    m3 = Symbol("m3", nonpositive=False, real=True)
    m4 = Symbol("m4", negative=False, real=True)
    assert (m1 < 0) is S.true
    assert (m2 <= 0) is S.true
    assert (m3 > 0) is S.true
    assert (m4 >= 0) is S.true
    m1 = Symbol("m1", negative=True)
    m2 = Symbol("m2", nonpositive=True)
    m3 = Symbol("m3", positive=True)
    m4 = Symbol("m4", nonnegative=True)
    assert (m1 < 0) is S.true
    assert (m2 <= 0) is S.true
    assert (m3 > 0) is S.true
    assert (m4 >= 0) is S.true
    m1 = Symbol("m1", negative=False, real=True)
    m2 = Symbol("m2", nonpositive=False, real=True)
    m3 = Symbol("m3", positive=False, real=True)
    m4 = Symbol("m4", nonnegative=False, real=True)
    assert (m1 < 0) is S.false
    assert (m2 <= 0) is S.false
    assert (m3 > 0) is S.false
    assert (m4 >= 0) is S.false


# See https://github.com/sympy/sympy/issues/17708
#def test_relational_noncommutative():
#    from sympy import Lt, Gt, Le, Ge
#    A, B = symbols('A,B', commutative=False)
#    assert (A < B) == Lt(A, B)
#    assert (A <= B) == Le(A, B)
#    assert (A > B) == Gt(A, B)
#    assert (A >= B) == Ge(A, B)


def test_basic_nostr():
    for obj in basic_objs:
        raises(TypeError, lambda: obj + '1')
        raises(TypeError, lambda: obj - '1')
        if obj == 2:
            assert obj * '1' == '11'
        else:
            raises(TypeError, lambda: obj * '1')
        raises(TypeError, lambda: obj / '1')
        raises(TypeError, lambda: obj ** '1')


def test_series_expansion_for_uniform_order():
    assert (1/x + y + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + y + x).series(x, 0, 1) == 1/x + y + O(x)
    assert (1/x + 1 + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + 1 + x).series(x, 0, 1) == 1/x + 1 + O(x)
    assert (1/x + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + y + y*x + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + y + y*x + x).series(x, 0, 1) == 1/x + y + O(x)


def test_leadterm():
    assert (3 + 2*x**(log(3)/log(2) - 1)).leadterm(x) == (3, 0)

    assert (1/x**2 + 1 + x + x**2).leadterm(x)[1] == -2
    assert (1/x + 1 + x + x**2).leadterm(x)[1] == -1
    assert (x**2 + 1/x).leadterm(x)[1] == -1
    assert (1 + x**2).leadterm(x)[1] == 0
    assert (x + 1).leadterm(x)[1] == 0
    assert (x + x**2).leadterm(x)[1] == 1
    assert (x**2).leadterm(x)[1] == 2


def test_as_leading_term():
    assert (3 + 2*x**(log(3)/log(2) - 1)).as_leading_term(x) == 3
    assert (1/x**2 + 1 + x + x**2).as_leading_term(x) == 1/x**2
    assert (1/x + 1 + x + x**2).as_leading_term(x) == 1/x
    assert (x**2 + 1/x).as_leading_term(x) == 1/x
    assert (1 + x**2).as_leading_term(x) == 1
    assert (x + 1).as_leading_term(x) == 1
    assert (x + x**2).as_leading_term(x) == x
    assert (x**2).as_leading_term(x) == x**2
    assert (x + oo).as_leading_term(x) is oo

    raises(ValueError, lambda: (x + 1).as_leading_term(1))

    # https://github.com/sympy/sympy/issues/21177
    e = -3*x + (x + Rational(3, 2) - sqrt(3)*S.ImaginaryUnit/2)**2\
        - Rational(3, 2) + 3*sqrt(3)*S.ImaginaryUnit/2
    assert e.as_leading_term(x) == -sqrt(3)*I*x

    # https://github.com/sympy/sympy/issues/21245
    e = 1 - x - x**2
    d = (1 + sqrt(5))/2
    assert e.subs(x, y + 1/d).as_leading_term(y) == \
        (-40*y - 16*sqrt(5)*y)/(16 + 8*sqrt(5))

    # https://github.com/sympy/sympy/issues/26991
    assert sinh(tanh(3/(100*x))).as_leading_term(x, cdir = 1) == sinh(1)


def test_leadterm2():
    assert (x*cos(1)*cos(1 + sin(1)) + sin(1 + sin(1))).leadterm(x) == \
           (sin(1 + sin(1)), 0)


def test_leadterm3():
    assert (y + z + x).leadterm(x) == (y + z, 0)


def test_as_leading_term2():
    assert (x*cos(1)*cos(1 + sin(1)) + sin(1 + sin(1))).as_leading_term(x) == \
        sin(1 + sin(1))


def test_as_leading_term3():
    assert (2 + pi + x).as_leading_term(x) == 2 + pi
    assert (2*x + pi*x + x**2).as_leading_term(x) == 2*x + pi*x


def test_as_leading_term4():
    # see issue 6843
    n = Symbol('n', integer=True, positive=True)
    r = -n**3/(2*n**2 + 4*n + 2) - n**2/(n**2 + 2*n + 1) + \
        n**2/(n + 1) - n/(2*n**2 + 4*n + 2) + n/(n*x + x) + 2*n/(n + 1) - \
        1 + 1/(n*x + x) + 1/(n + 1) - 1/x
    assert r.as_leading_term(x).cancel() == n/2


def test_as_leading_term_stub():
    class foo(Function):
        pass
    assert foo(1/x).as_leading_term(x) == foo(1/x)
    assert foo(1).as_leading_term(x) == foo(1)
    raises(NotImplementedError, lambda: foo(x).as_leading_term(x))


def test_as_leading_term_deriv_integral():
    # related to issue 11313
    assert Derivative(x ** 3, x).as_leading_term(x) == 3*x**2
    assert Derivative(x ** 3, y).as_leading_term(x) == 0

    assert Integral(x ** 3, x).as_leading_term(x) == x**4/4
    assert Integral(x ** 3, y).as_leading_term(x) == y*x**3

    assert Derivative(exp(x), x).as_leading_term(x) == 1
    assert Derivative(log(x), x).as_leading_term(x) == (1/x).as_leading_term(x)


def test_atoms():
    assert x.atoms() == {x}
    assert (1 + x).atoms() == {x, S.One}

    assert (1 + 2*cos(x)).atoms(Symbol) == {x}
    assert (1 + 2*cos(x)).atoms(Symbol, Number) == {S.One, S(2), x}

    assert (2*(x**(y**x))).atoms() == {S(2), x, y}

    assert S.Half.atoms() == {S.Half}
    assert S.Half.atoms(Symbol) == set()

    assert sin(oo).atoms(oo) == set()

    assert Poly(0, x).atoms() == {S.Zero, x}
    assert Poly(1, x).atoms() == {S.One, x}

    assert Poly(x, x).atoms() == {x}
    assert Poly(x, x, y).atoms() == {x, y}
    assert Poly(x + y, x, y).atoms() == {x, y}
    assert Poly(x + y, x, y, z).atoms() == {x, y, z}
    assert Poly(x + y*t, x, y, z).atoms() == {t, x, y, z}

    assert (I*pi).atoms(NumberSymbol) == {pi}
    assert (I*pi).atoms(NumberSymbol, I) == \
        (I*pi).atoms(I, NumberSymbol) == {pi, I}

    assert exp(exp(x)).atoms(exp) == {exp(exp(x)), exp(x)}
    assert (1 + x*(2 + y) + exp(3 + z)).atoms(Add) == \
        {1 + x*(2 + y) + exp(3 + z), 2 + y, 3 + z}

    # issue 6132
    e = (f(x) + sin(x) + 2)
    assert e.atoms(AppliedUndef) == \
        {f(x)}
    assert e.atoms(AppliedUndef, Function) == \
        {f(x), sin(x)}
    assert e.atoms(Function) == \
        {f(x), sin(x)}
    assert e.atoms(AppliedUndef, Number) == \
        {f(x), S(2)}
    assert e.atoms(Function, Number) == \
        {S(2), sin(x), f(x)}


def test_is_polynomial():
    k = Symbol('k', nonnegative=True, integer=True)

    assert Rational(2).is_polynomial(x, y, z) is True
    assert (S.Pi).is_polynomial(x, y, z) is True

    assert x.is_polynomial(x) is True
    assert x.is_polynomial(y) is True

    assert (x**2).is_polynomial(x) is True
    assert (x**2).is_polynomial(y) is True

    assert (x**(-2)).is_polynomial(x) is False
    assert (x**(-2)).is_polynomial(y) is True

    assert (2**x).is_polynomial(x) is False
    assert (2**x).is_polynomial(y) is True

    assert (x**k).is_polynomial(x) is False
    assert (x**k).is_polynomial(k) is False
    assert (x**x).is_polynomial(x) is False
    assert (k**k).is_polynomial(k) is False
    assert (k**x).is_polynomial(k) is False

    assert (x**(-k)).is_polynomial(x) is False
    assert ((2*x)**k).is_polynomial(x) is False

    assert (x**2 + 3*x - 8).is_polynomial(x) is True
    assert (x**2 + 3*x - 8).is_polynomial(y) is True

    assert (x**2 + 3*x - 8).is_polynomial() is True

    assert sqrt(x).is_polynomial(x) is False
    assert (sqrt(x)**3).is_polynomial(x) is False

    assert (x**2 + 3*x*sqrt(y) - 8).is_polynomial(x) is True
    assert (x**2 + 3*x*sqrt(y) - 8).is_polynomial(y) is False

    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(2)).is_polynomial() is True
    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(x)).is_polynomial() is False

    assert (
        (x**2)*(y**2) + x*(y**2) + y*x + exp(2)).is_polynomial(x, y) is True
    assert (
        (x**2)*(y**2) + x*(y**2) + y*x + exp(x)).is_polynomial(x, y) is False

    assert (1/f(x) + 1).is_polynomial(f(x)) is False


def test_is_rational_function():
    assert Integer(1).is_rational_function() is True
    assert Integer(1).is_rational_function(x) is True

    assert Rational(17, 54).is_rational_function() is True
    assert Rational(17, 54).is_rational_function(x) is True

    assert (12/x).is_rational_function() is True
    assert (12/x).is_rational_function(x) is True

    assert (x/y).is_rational_function() is True
    assert (x/y).is_rational_function(x) is True
    assert (x/y).is_rational_function(x, y) is True

    assert (x**2 + 1/x/y).is_rational_function() is True
    assert (x**2 + 1/x/y).is_rational_function(x) is True
    assert (x**2 + 1/x/y).is_rational_function(x, y) is True

    assert (sin(y)/x).is_rational_function() is False
    assert (sin(y)/x).is_rational_function(y) is False
    assert (sin(y)/x).is_rational_function(x) is True
    assert (sin(y)/x).is_rational_function(x, y) is False

    for i in _illegal:
        assert not i.is_rational_function()
        for d in (1, x):
            assert not (i/d).is_rational_function()


def test_is_meromorphic():
    f = a/x**2 + b + x + c*x**2
    assert f.is_meromorphic(x, 0) is True
    assert f.is_meromorphic(x, 1) is True
    assert f.is_meromorphic(x, zoo) is True

    g = 3 + 2*x**(log(3)/log(2) - 1)
    assert g.is_meromorphic(x, 0) is False
    assert g.is_meromorphic(x, 1) is True
    assert g.is_meromorphic(x, zoo) is False

    n = Symbol('n', integer=True)
    e = sin(1/x)**n*x
    assert e.is_meromorphic(x, 0) is False
    assert e.is_meromorphic(x, 1) is True
    assert e.is_meromorphic(x, zoo) is False

    e = log(x)**pi
    assert e.is_meromorphic(x, 0) is False
    assert e.is_meromorphic(x, 1) is False
    assert e.is_meromorphic(x, 2) is True
    assert e.is_meromorphic(x, zoo) is False

    assert (log(x)**a).is_meromorphic(x, 0) is False
    assert (log(x)**a).is_meromorphic(x, 1) is False
    assert (a**log(x)).is_meromorphic(x, 0) is None
    assert (3**log(x)).is_meromorphic(x, 0) is False
    assert (3**log(x)).is_meromorphic(x, 1) is True

def test_is_algebraic_expr():
    assert sqrt(3).is_algebraic_expr(x) is True
    assert sqrt(3).is_algebraic_expr() is True

    eq = ((1 + x**2)/(1 - y**2))**(S.One/3)
    assert eq.is_algebraic_expr(x) is True
    assert eq.is_algebraic_expr(y) is True

    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr(x) is True
    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr(y) is True
    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr() is True

    assert (cos(y)/sqrt(x)).is_algebraic_expr() is False
    assert (cos(y)/sqrt(x)).is_algebraic_expr(x) is True
    assert (cos(y)/sqrt(x)).is_algebraic_expr(y) is False
    assert (cos(y)/sqrt(x)).is_algebraic_expr(x, y) is False


def test_SAGE1():
    #see https://github.com/sympy/sympy/issues/3346
    class MyInt:
        def _sympy_(self):
            return Integer(5)
    m = MyInt()
    e = Rational(2)*m
    assert e == 10

    raises(TypeError, lambda: Rational(2)*MyInt)


def test_SAGE2():
    class MyInt:
        def __int__(self):
            return 5
    assert sympify(MyInt()) == 5
    e = Rational(2)*MyInt()
    assert e == 10

    raises(TypeError, lambda: Rational(2)*MyInt)


def test_SAGE3():
    class MySymbol:
        def __rmul__(self, other):
            return ('mys', other, self)

    o = MySymbol()
    e = x*o

    assert e == ('mys', x, o)


def test_len():
    e = x*y
    assert len(e.args) == 2
    e = x + y + z
    assert len(e.args) == 3


def test_doit():
    a = Integral(x**2, x)

    assert isinstance(a.doit(), Integral) is False

    assert isinstance(a.doit(integrals=True), Integral) is False
    assert isinstance(a.doit(integrals=False), Integral) is True

    assert (2*Integral(x, x)).doit() == x**2


def test_attribute_error():
    raises(AttributeError, lambda: x.cos())
    raises(AttributeError, lambda: x.sin())
    raises(AttributeError, lambda: x.exp())


def test_args():
    assert (x*y).args in ((x, y), (y, x))
    assert (x + y).args in ((x, y), (y, x))
    assert (x*y + 1).args in ((x*y, 1), (1, x*y))
    assert sin(x*y).args == (x*y,)
    assert sin(x*y).args[0] == x*y
    assert (x**y).args == (x, y)
    assert (x**y).args[0] == x
    assert (x**y).args[1] == y


def test_noncommutative_expand_issue_3757():
    A, B, C = symbols('A,B,C', commutative=False)
    assert A*B - B*A != 0
    assert (A*(A + B)*B).expand() == A**2*B + A*B**2
    assert (A*(A + B + C)*B).expand() == A**2*B + A*B**2 + A*C*B


def test_as_numer_denom():
    a, b, c = symbols('a, b, c')

    assert nan.as_numer_denom() == (nan, 1)
    assert oo.as_numer_denom() == (oo, 1)
    assert (-oo).as_numer_denom() == (-oo, 1)
    assert zoo.as_numer_denom() == (zoo, 1)
    assert (-zoo).as_numer_denom() == (zoo, 1)

    assert x.as_numer_denom() == (x, 1)
    assert (1/x).as_numer_denom() == (1, x)
    assert (x/y).as_numer_denom() == (x, y)
    assert (x/2).as_numer_denom() == (x, 2)
    assert (x*y/z).as_numer_denom() == (x*y, z)
    assert (x/(y*z)).as_numer_denom() == (x, y*z)
    assert S.Half.as_numer_denom() == (1, 2)
    assert (1/y**2).as_numer_denom() == (1, y**2)
    assert (x/y**2).as_numer_denom() == (x, y**2)
    assert ((x**2 + 1)/y).as_numer_denom() == (x**2 + 1, y)
    assert (x*(y + 1)/y**7).as_numer_denom() == (x*(y + 1), y**7)
    assert (x**-2).as_numer_denom() == (1, x**2)
    assert (a/x + b/2/x + c/3/x).as_numer_denom() == \
        (6*a + 3*b + 2*c, 6*x)
    assert (a/x + b/2/x + c/3/y).as_numer_denom() == \
        (2*c*x + y*(6*a + 3*b), 6*x*y)
    assert (a/x + b/2/x + c/.5/x).as_numer_denom() == \
        (2*a + b + 4.0*c, 2*x)
    # this should take no more than a few seconds
    assert int(log(Add(*[Dummy()/i/x for i in range(1, 705)]
                       ).as_numer_denom()[1]/x).n(4)) == 705
    for i in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
        assert (i + x/3).as_numer_denom() == \
            (x + i, 3)
    assert (S.Infinity + x/3 + y/4).as_numer_denom() == \
        (4*x + 3*y + S.Infinity, 12)
    assert (oo*x + zoo*y).as_numer_denom() == \
        (zoo*y + oo*x, 1)

    A, B, C = symbols('A,B,C', commutative=False)

    assert (A*B*C**-1).as_numer_denom() == (A*B*C**-1, 1)
    assert (A*B*C**-1/x).as_numer_denom() == (A*B*C**-1, x)
    assert (C**-1*A*B).as_numer_denom() == (C**-1*A*B, 1)
    assert (C**-1*A*B/x).as_numer_denom() == (C**-1*A*B, x)
    assert ((A*B*C)**-1).as_numer_denom() == ((A*B*C)**-1, 1)
    assert ((A*B*C)**-1/x).as_numer_denom() == ((A*B*C)**-1, x)

    # the following morphs from Add to Mul during processing
    assert Add(0, (x + y)/z/-2, evaluate=False).as_numer_denom(
        ) == (-x - y, 2*z)


def test_trunc():
    import math
    x, y = symbols('x y')
    assert math.trunc(2) == 2
    assert math.trunc(4.57) == 4
    assert math.trunc(-5.79) == -5
    assert math.trunc(pi) == 3
    assert math.trunc(log(7)) == 1
    assert math.trunc(exp(5)) == 148
    assert math.trunc(cos(pi)) == -1
    assert math.trunc(sin(5)) == 0

    raises(TypeError, lambda: math.trunc(x))
    raises(TypeError, lambda: math.trunc(x + y**2))
    raises(TypeError, lambda: math.trunc(oo))


def test_as_independent():
    assert S.Zero.as_independent(x, as_Add=True) == (0, 0)
    assert S.Zero.as_independent(x, as_Add=False) == (0, 0)
    assert (2*x*sin(x) + y + x).as_independent(x) == (y, x + 2*x*sin(x))
    assert (2*x*sin(x) + y + x).as_independent(y) == (x + 2*x*sin(x), y)

    assert (2*x*sin(x) + y + x).as_independent(x, y) == (0, y + x + 2*x*sin(x))

    assert (x*sin(x)*cos(y)).as_independent(x) == (cos(y), x*sin(x))
    assert (x*sin(x)*cos(y)).as_independent(y) == (x*sin(x), cos(y))

    assert (x*sin(x)*cos(y)).as_independent(x, y) == (1, x*sin(x)*cos(y))

    assert (sin(x)).as_independent(x) == (1, sin(x))
    assert (sin(x)).as_independent(y) == (sin(x), 1)

    assert (2*sin(x)).as_independent(x) == (2, sin(x))
    assert (2*sin(x)).as_independent(y) == (2*sin(x), 1)

    # issue 4903 = 1766b
    n1, n2, n3 = symbols('n1 n2 n3', commutative=False)
    assert (n1 + n1*n2).as_independent(n2) == (n1, n1*n2)
    assert (n2*n1 + n1*n2).as_independent(n2) == (0, n1*n2 + n2*n1)
    assert (n1*n2*n1).as_independent(n2) == (n1, n2*n1)
    assert (n1*n2*n1).as_independent(n1) == (1, n1*n2*n1)

    assert (3*x).as_independent(x, as_Add=True) == (0, 3*x)
    assert (3*x).as_independent(x, as_Add=False) == (3, x)
    assert (3 + x).as_independent(x, as_Add=True) == (3, x)
    assert (3 + x).as_independent(x, as_Add=False) == (1, 3 + x)

    # issue 5479
    assert (3*x).as_independent(Symbol) == (3, x)

    # issue 5648
    assert (n1*x*y).as_independent(x) == (n1*y, x)
    assert ((x + n1)*(x - y)).as_independent(x) == (1, (x + n1)*(x - y))
    assert ((x + n1)*(x - y)).as_independent(y) == (x + n1, x - y)
    assert (DiracDelta(x - n1)*DiracDelta(x - y)).as_independent(x) \
        == (1, DiracDelta(x - n1)*DiracDelta(x - y))
    assert (x*y*n1*n2*n3).as_independent(n2) == (x*y*n1, n2*n3)
    assert (x*y*n1*n2*n3).as_independent(n1) == (x*y, n1*n2*n3)
    assert (x*y*n1*n2*n3).as_independent(n3) == (x*y*n1*n2, n3)
    assert (DiracDelta(x - n1)*DiracDelta(y - n1)*DiracDelta(x - n2)).as_independent(y) == \
           (DiracDelta(x - n1)*DiracDelta(x - n2), DiracDelta(y - n1))

    # issue 5784
    assert (x + Integral(x, (x, 1, 2))).as_independent(x, strict=True) == \
           (Integral(x, (x, 1, 2)), x)

    eq = Add(x, -x, 2, -3, evaluate=False)
    assert eq.as_independent(x) == (-1, Add(x, -x, evaluate=False))
    eq = Mul(x, 1/x, 2, -3, evaluate=False)
    assert eq.as_independent(x) == (-6, Mul(x, 1/x, evaluate=False))

    assert (x*y).as_independent(z, as_Add=True) == (x*y, 0)

@XFAIL
def test_call_2():
    # TODO UndefinedFunction does not subclass Expr
    assert (2*f)(x) == 2*f(x)


def test_replace():
    e = log(sin(x)) + tan(sin(x**2))

    assert e.replace(sin, cos) == log(cos(x)) + tan(cos(x**2))
    assert e.replace(
        sin, lambda a: sin(2*a)) == log(sin(2*x)) + tan(sin(2*x**2))

    a = Wild('a')
    b = Wild('b')

    assert e.replace(sin(a), cos(a)) == log(cos(x)) + tan(cos(x**2))
    assert e.replace(
        sin(a), lambda a: sin(2*a)) == log(sin(2*x)) + tan(sin(2*x**2))
    # test exact
    assert (2*x).replace(a*x + b, b - a, exact=True) == 2*x
    assert (2*x).replace(a*x + b, b - a) == 2*x
    assert (2*x).replace(a*x + b, b - a, exact=False) == 2/x
    assert (2*x).replace(a*x + b, lambda a, b: b - a, exact=True) == 2*x
    assert (2*x).replace(a*x + b, lambda a, b: b - a) == 2*x
    assert (2*x).replace(a*x + b, lambda a, b: b - a, exact=False) == 2/x

    g = 2*sin(x**3)

    assert g.replace(
        lambda expr: expr.is_Number, lambda expr: expr**2) == 4*sin(x**9)

    assert cos(x).replace(cos, sin, map=True) == (sin(x), {cos(x): sin(x)})
    assert sin(x).replace(cos, sin) == sin(x)

    cond, func = lambda x: x.is_Mul, lambda x: 2*x
    assert (x*y).replace(cond, func, map=True) == (2*x*y, {x*y: 2*x*y})
    assert (x*(1 + x*y)).replace(cond, func, map=True) == \
        (2*x*(2*x*y + 1), {x*(2*x*y + 1): 2*x*(2*x*y + 1), x*y: 2*x*y})
    assert (y*sin(x)).replace(sin, lambda expr: sin(expr)/y, map=True) == \
        (sin(x), {sin(x): sin(x)/y})
    # if not simultaneous then y*sin(x) -> y*sin(x)/y = sin(x) -> sin(x)/y
    assert (y*sin(x)).replace(sin, lambda expr: sin(expr)/y,
        simultaneous=False) == sin(x)/y
    assert (x**2 + O(x**3)).replace(Pow, lambda b, e: b**e/e
        ) == x**2/2 + O(x**3)
    assert (x**2 + O(x**3)).replace(Pow, lambda b, e: b**e/e,
        simultaneous=False) == x**2/2 + O(x**3)
    assert (x*(x*y + 3)).replace(lambda x: x.is_Mul, lambda x: 2 + x) == \
        x*(x*y + 5) + 2
    e = (x*y + 1)*(2*x*y + 1) + 1
    assert e.replace(cond, func, map=True) == (
        2*((2*x*y + 1)*(4*x*y + 1)) + 1,
        {2*x*y: 4*x*y, x*y: 2*x*y, (2*x*y + 1)*(4*x*y + 1):
        2*((2*x*y + 1)*(4*x*y + 1))})
    assert x.replace(x, y) == y
    assert (x + 1).replace(1, 2) == x + 2

    # https://groups.google.com/forum/#!topic/sympy/8wCgeC95tz0
    n1, n2, n3 = symbols('n1:4', commutative=False)
    assert (n1*f(n2)).replace(f, lambda x: x) == n1*n2
    assert (n3*f(n2)).replace(f, lambda x: x) == n3*n2

    # issue 16725
    assert S.Zero.replace(Wild('x'), 1) == 1
    # let the user override the default decision of False
    assert S.Zero.replace(Wild('x'), 1, exact=True) == 0


def test_replace_integral():
    # https://github.com/sympy/sympy/issues/27142
    q, p, s, t = symbols('q p s t', cls=Wild)
    a, b, c, d = symbols('a b c d')
    i = Integral(a + b, (b, c, d))
    pattern = Integral(q, (p, s, t))
    assert i.replace(pattern, q) == a + b


def test_find():
    expr = (x + y + 2 + sin(3*x))

    assert expr.find(lambda u: u.is_Integer) == {S(2), S(3)}
    assert expr.find(lambda u: u.is_Symbol) == {x, y}

    assert expr.find(lambda u: u.is_Integer, group=True) == {S(2): 1, S(3): 1}
    assert expr.find(lambda u: u.is_Symbol, group=True) == {x: 2, y: 1}

    assert expr.find(Integer) == {S(2), S(3)}
    assert expr.find(Symbol) == {x, y}

    assert expr.find(Integer, group=True) == {S(2): 1, S(3): 1}
    assert expr.find(Symbol, group=True) == {x: 2, y: 1}

    a = Wild('a')

    expr = sin(sin(x)) + sin(x) + cos(x) + x

    assert expr.find(lambda u: type(u) is sin) == {sin(x), sin(sin(x))}
    assert expr.find(
        lambda u: type(u) is sin, group=True) == {sin(x): 2, sin(sin(x)): 1}

    assert expr.find(sin(a)) == {sin(x), sin(sin(x))}
    assert expr.find(sin(a), group=True) == {sin(x): 2, sin(sin(x)): 1}

    assert expr.find(sin) == {sin(x), sin(sin(x))}
    assert expr.find(sin, group=True) == {sin(x): 2, sin(sin(x)): 1}


def test_count():
    expr = (x + y + 2 + sin(3*x))

    assert expr.count(lambda u: u.is_Integer) == 2
    assert expr.count(lambda u: u.is_Symbol) == 3

    assert expr.count(Integer) == 2
    assert expr.count(Symbol) == 3
    assert expr.count(2) == 1

    a = Wild('a')

    assert expr.count(sin) == 1
    assert expr.count(sin(a)) == 1
    assert expr.count(lambda u: type(u) is sin) == 1

    assert f(x).count(f(x)) == 1
    assert f(x).diff(x).count(f(x)) == 1
    assert f(x).diff(x).count(x) == 2


def test_has_basics():
    p = Wild('p')

    assert sin(x).has(x)
    assert sin(x).has(sin)
    assert not sin(x).has(y)
    assert not sin(x).has(cos)
    assert f(x).has(x)
    assert f(x).has(f)
    assert not f(x).has(y)
    assert not f(x).has(g)

    assert f(x).diff(x).has(x)
    assert f(x).diff(x).has(f)
    assert f(x).diff(x).has(Derivative)
    assert not f(x).diff(x).has(y)
    assert not f(x).diff(x).has(g)
    assert not f(x).diff(x).has(sin)

    assert (x**2).has(Symbol)
    assert not (x**2).has(Wild)
    assert (2*p).has(Wild)

    assert not x.has()

    # see issue at https://github.com/sympy/sympy/issues/5190
    assert not S(1).has(Wild)
    assert not x.has(Wild)


def test_has_multiple():
    f = x**2*y + sin(2**t + log(z))

    assert f.has(x)
    assert f.has(y)
    assert f.has(z)
    assert f.has(t)

    assert not f.has(u)

    assert f.has(x, y, z, t)
    assert f.has(x, y, z, t, u)

    i = Integer(4400)

    assert not i.has(x)

    assert (i*x**i).has(x)
    assert not (i*y**i).has(x)
    assert (i*y**i).has(x, y)
    assert not (i*y**i).has(x, z)


def test_has_piecewise():
    f = (x*y + 3/y)**(3 + 2)
    p = Piecewise((g(x), x < -1), (1, x <= 1), (f, True))

    assert p.has(x)
    assert p.has(y)
    assert not p.has(z)
    assert p.has(1)
    assert p.has(3)
    assert not p.has(4)
    assert p.has(f)
    assert p.has(g)
    assert not p.has(h)


def test_has_iterative():
    A, B, C = symbols('A,B,C', commutative=False)
    f = x*gamma(x)*sin(x)*exp(x*y)*A*B*C*cos(x*A*B)

    assert f.has(x)
    assert f.has(x*y)
    assert f.has(x*sin(x))
    assert not f.has(x*sin(y))
    assert f.has(x*A)
    assert f.has(x*A*B)
    assert not f.has(x*A*C)
    assert f.has(x*A*B*C)
    assert not f.has(x*A*C*B)
    assert f.has(x*sin(x)*A*B*C)
    assert not f.has(x*sin(x)*A*C*B)
    assert not f.has(x*sin(y)*A*B*C)
    assert f.has(x*gamma(x))
    assert not f.has(x + sin(x))

    assert (x & y & z).has(x & z)


def test_has_integrals():
    f = Integral(x**2 + sin(x*y*z), (x, 0, x + y + z))

    assert f.has(x + y)
    assert f.has(x + z)
    assert f.has(y + z)

    assert f.has(x*y)
    assert f.has(x*z)
    assert f.has(y*z)

    assert not f.has(2*x + y)
    assert not f.has(2*x*y)


def test_has_tuple():
    assert Tuple(x, y).has(x)
    assert not Tuple(x, y).has(z)
    assert Tuple(f(x), g(x)).has(x)
    assert not Tuple(f(x), g(x)).has(y)
    assert Tuple(f(x), g(x)).has(f)
    assert Tuple(f(x), g(x)).has(f(x))
    # XXX to be deprecated
    #assert not Tuple(f, g).has(x)
    #assert Tuple(f, g).has(f)
    #assert not Tuple(f, g).has(h)
    assert Tuple(True).has(True)
    assert Tuple(True).has(S.true)
    assert not Tuple(True).has(1)


def test_has_units():
    from sympy.physics.units import m, s

    assert (x*m/s).has(x)
    assert (x*m/s).has(y, z) is False


def test_has_polys():
    poly = Poly(x**2 + x*y*sin(z), x, y, t)

    assert poly.has(x)
    assert poly.has(x, y, z)
    assert poly.has(x, y, z, t)


def test_has_physics():
    assert FockState((x, y)).has(x)


def test_as_poly_as_expr():
    f = x**2 + 2*x*y

    assert f.as_poly().as_expr() == f
    assert f.as_poly(x, y).as_expr() == f

    assert (f + sin(x)).as_poly(x, y) is None

    p = Poly(f, x, y)

    assert p.as_poly() == p

    # https://github.com/sympy/sympy/issues/20610
    assert S(2).as_poly() is None
    assert sqrt(2).as_poly(extension=True) is None

    raises(AttributeError, lambda: Tuple(x, x).as_poly(x))
    raises(AttributeError, lambda: Tuple(x ** 2, x, y).as_poly(x))


def test_nonzero():
    assert bool(S.Zero) is False
    assert bool(S.One) is True
    assert bool(x) is True
    assert bool(x + y) is True
    assert bool(x - x) is False
    assert bool(x*y) is True
    assert bool(x*1) is True
    assert bool(x*0) is False


def test_is_number():
    assert Float(3.14).is_number is True
    assert Integer(737).is_number is True
    assert Rational(3, 2).is_number is True
    assert Rational(8).is_number is True
    assert x.is_number is False
    assert (2*x).is_number is False
    assert (x + y).is_number is False
    assert log(2).is_number is True
    assert log(x).is_number is False
    assert (2 + log(2)).is_number is True
    assert (8 + log(2)).is_number is True
    assert (2 + log(x)).is_number is False
    assert (8 + log(2) + x).is_number is False
    assert (1 + x**2/x - x).is_number is True
    assert Tuple(Integer(1)).is_number is False
    assert Add(2, x).is_number is False
    assert Mul(3, 4).is_number is True
    assert Pow(log(2), 2).is_number is True
    assert oo.is_number is True
    g = WildFunction('g')
    assert g.is_number is False
    assert (2*g).is_number is False
    assert (x**2).subs(x, 3).is_number is True

    # test extensibility of .is_number
    # on subinstances of Basic
    class A(Basic):
        pass
    a = A()
    assert a.is_number is False


def test_as_coeff_add():
    assert S(2).as_coeff_add() == (2, ())
    assert S(3.0).as_coeff_add() == (0, (S(3.0),))
    assert S(-3.0).as_coeff_add() == (0, (S(-3.0),))
    assert x.as_coeff_add() == (0, (x,))
    assert (x - 1).as_coeff_add() == (-1, (x,))
    assert (x + 1).as_coeff_add() == (1, (x,))
    assert (x + 2).as_coeff_add() == (2, (x,))
    assert (x + y).as_coeff_add(y) == (x, (y,))
    assert (3*x).as_coeff_add(y) == (3*x, ())
    # don't do expansion
    e = (x + y)**2
    assert e.as_coeff_add(y) == (0, (e,))


def test_as_coeff_mul():
    assert S(2).as_coeff_mul() == (2, ())
    assert S(3.0).as_coeff_mul() == (1, (S(3.0),))
    assert S(-3.0).as_coeff_mul() == (-1, (S(3.0),))
    assert S(-3.0).as_coeff_mul(rational=False) == (-S(3.0), ())
    assert x.as_coeff_mul() == (1, (x,))
    assert (-x).as_coeff_mul() == (-1, (x,))
    assert (2*x).as_coeff_mul() == (2, (x,))
    assert (x*y).as_coeff_mul(y) == (x, (y,))
    assert (3 + x).as_coeff_mul() == (1, (3 + x,))
    assert (3 + x).as_coeff_mul(y) == (3 + x, ())
    # don't do expansion
    e = exp(x + y)
    assert e.as_coeff_mul(y) == (1, (e,))
    e = 2**(x + y)
    assert e.as_coeff_mul(y) == (1, (e,))
    assert (1.1*x).as_coeff_mul(rational=False) == (1.1, (x,))
    assert (1.1*x).as_coeff_mul() == (1, (1.1, x))
    assert (-oo*x).as_coeff_mul(rational=True) == (-1, (oo, x))


def test_as_coeff_exponent():
    assert (3*x**4).as_coeff_exponent(x) == (3, 4)
    assert (2*x**3).as_coeff_exponent(x) == (2, 3)
    assert (4*x**2).as_coeff_exponent(x) == (4, 2)
    assert (6*x**1).as_coeff_exponent(x) == (6, 1)
    assert (3*x**0).as_coeff_exponent(x) == (3, 0)
    assert (2*x**0).as_coeff_exponent(x) == (2, 0)
    assert (1*x**0).as_coeff_exponent(x) == (1, 0)
    assert (0*x**0).as_coeff_exponent(x) == (0, 0)
    assert (-1*x**0).as_coeff_exponent(x) == (-1, 0)
    assert (-2*x**0).as_coeff_exponent(x) == (-2, 0)
    assert (2*x**3 + pi*x**3).as_coeff_exponent(x) == (2 + pi, 3)
    assert (x*log(2)/(2*x + pi*x)).as_coeff_exponent(x) == \
        (log(2)/(2 + pi), 0)
    # issue 4784
    D = Derivative
    fx = D(f(x), x)
    assert fx.as_coeff_exponent(f(x)) == (fx, 0)


def test_extractions():
    for base in (2, S.Exp1):
        assert Pow(base**x, 3, evaluate=False
            ).extract_multiplicatively(base**x) == base**(2*x)
        assert (base**(5*x)).extract_multiplicatively(
            base**(3*x)) == base**(2*x)
    assert ((x*y)**3).extract_multiplicatively(x**2 * y) == x*y**2
    assert ((x*y)**3).extract_multiplicatively(x**4 * y) is None
    assert (2*x).extract_multiplicatively(2) == x
    assert (2*x).extract_multiplicatively(3) is None
    assert (2*x).extract_multiplicatively(-1) is None
    assert (S.Half*x).extract_multiplicatively(3) == x/6
    assert (sqrt(x)).extract_multiplicatively(x) is None
    assert (sqrt(x)).extract_multiplicatively(1/x) is None
    assert x.extract_multiplicatively(-x) is None
    assert (-2 - 4*I).extract_multiplicatively(-2) == 1 + 2*I
    assert (-2 - 4*I).extract_multiplicatively(3) is None
    assert (-2*x - 4*y - 8).extract_multiplicatively(-2) == x + 2*y + 4
    assert (-2*x*y - 4*x**2*y).extract_multiplicatively(-2*y) == 2*x**2 + x
    assert (2*x*y + 4*x**2*y).extract_multiplicatively(2*y) == 2*x**2 + x
    assert (-4*y**2*x).extract_multiplicatively(-3*y) is None
    assert (2*x).extract_multiplicatively(1) == 2*x
    assert (-oo).extract_multiplicatively(5) is -oo
    assert (oo).extract_multiplicatively(5) is oo

    assert ((x*y)**3).extract_additively(1) is None
    assert (x + 1).extract_additively(x) == 1
    assert (x + 1).extract_additively(2*x) is None
    assert (x + 1).extract_additively(-x) is None
    assert (-x + 1).extract_additively(2*x) is None
    assert (2*x + 3).extract_additively(x) == x + 3
    assert (2*x + 3).extract_additively(2) == 2*x + 1
    assert (2*x + 3).extract_additively(3) == 2*x
    assert (2*x + 3).extract_additively(-2) is None
    assert (2*x + 3).extract_additively(3*x) is None
    assert (2*x + 3).extract_additively(2*x) == 3
    assert x.extract_additively(0) == x
    assert S(2).extract_additively(x) is None
    assert S(2.).extract_additively(2.) is S.Zero
    assert S(2.).extract_additively(2) is S.Zero
    assert S(2*x + 3).extract_additively(x + 1) == x + 2
    assert S(2*x + 3).extract_additively(y + 1) is None
    assert S(2*x - 3).extract_additively(x + 1) is None
    assert S(2*x - 3).extract_additively(y + z) is None
    assert ((a + 1)*x*4 + y).extract_additively(x).expand() == \
        4*a*x + 3*x + y
    assert ((a + 1)*x*4 + 3*y).extract_additively(x + 2*y).expand() == \
        4*a*x + 3*x + y
    assert (y*(x + 1)).extract_additively(x + 1) is None
    assert ((y + 1)*(x + 1) + 3).extract_additively(x + 1) == \
        y*(x + 1) + 3
    assert ((x + y)*(x + 1) + x + y + 3).extract_additively(x + y) == \
        x*(x + y) + 3
    assert (x + y + 2*((x + y)*(x + 1)) + 3).extract_additively((x + y)*(x + 1)) == \
        x + y + (x + 1)*(x + y) + 3
    assert ((y + 1)*(x + 2*y + 1) + 3).extract_additively(y + 1) == \
        (x + 2*y)*(y + 1) + 3
    assert (-x - x*I).extract_additively(-x) == -I*x
    # extraction does not leave artificats, now
    assert (4*x*(y + 1) + y).extract_additively(x) == x*(4*y + 3) + y

    n = Symbol("n", integer=True)
    assert (Integer(-3)).could_extract_minus_sign() is True
    assert (-n*x + x).could_extract_minus_sign() != \
        (n*x - x).could_extract_minus_sign()
    assert (x - y).could_extract_minus_sign() != \
        (-x + y).could_extract_minus_sign()
    assert (1 - x - y).could_extract_minus_sign() is True
    assert (1 - x + y).could_extract_minus_sign() is False
    assert ((-x - x*y)/y).could_extract_minus_sign() is False
    assert ((x + x*y)/(-y)).could_extract_minus_sign() is True
    assert ((x + x*y)/y).could_extract_minus_sign() is False
    assert ((-x - y)/(x + y)).could_extract_minus_sign() is False

    class sign_invariant(Function, Expr):
        nargs = 1
        def __neg__(self):
            return self
    foo = sign_invariant(x)
    assert foo == -foo
    assert foo.could_extract_minus_sign() is False
    assert (x - y).could_extract_minus_sign() is False
    assert (-x + y).could_extract_minus_sign() is True
    assert (x - 1).could_extract_minus_sign() is False
    assert (1 - x).could_extract_minus_sign() is True
    assert (sqrt(2) - 1).could_extract_minus_sign() is True
    assert (1 - sqrt(2)).could_extract_minus_sign() is False
    # check that result is canonical
    eq = (3*x + 15*y).extract_multiplicatively(3)
    assert eq.args == eq.func(*eq.args).args


def test_nan_extractions():
    for r in (1, 0, I, nan):
        assert nan.extract_additively(r) is None
        assert nan.extract_multiplicatively(r) is None


def test_coeff():
    assert (x + 1).coeff(x + 1) == 1
    assert (3*x).coeff(0) == 0
    assert (z*(1 + x)*x**2).coeff(1 + x) == z*x**2
    assert (1 + 2*x*x**(1 + x)).coeff(x*x**(1 + x)) == 2
    assert (1 + 2*x**(y + z)).coeff(x**(y + z)) == 2
    assert (3 + 2*x + 4*x**2).coeff(1) == 0
    assert (3 + 2*x + 4*x**2).coeff(-1) == 0
    assert (3 + 2*x + 4*x**2).coeff(x) == 2
    assert (3 + 2*x + 4*x**2).coeff(x**2) == 4
    assert (3 + 2*x + 4*x**2).coeff(x**3) == 0

    assert (-x/8 + x*y).coeff(x) == Rational(-1, 8) + y
    assert (-x/8 + x*y).coeff(-x) == S.One/8
    assert (4*x).coeff(2*x) == 0
    assert (2*x).coeff(2*x) == 1
    assert (-oo*x).coeff(x*oo) == -1
    assert (10*x).coeff(x, 0) == 0
    assert (10*x).coeff(10*x, 0) == 0

    n1, n2 = symbols('n1 n2', commutative=False)
    assert (n1*n2).coeff(n1) == 1
    assert (n1*n2).coeff(n2) == n1
    assert (n1*n2 + x*n1).coeff(n1) == 1  # 1*n1*(n2+x)
    assert (n2*n1 + x*n1).coeff(n1) == n2 + x
    assert (n2*n1 + x*n1**2).coeff(n1) == n2
    assert (n1**x).coeff(n1) == 0
    assert (n1*n2 + n2*n1).coeff(n1) == 0
    assert (2*(n1 + n2)*n2).coeff(n1 + n2, right=1) == n2
    assert (2*(n1 + n2)*n2).coeff(n1 + n2, right=0) == 2

    assert (2*f(x) + 3*f(x).diff(x)).coeff(f(x)) == 2

    expr = z*(x + y)**2
    expr2 = z*(x + y)**2 + z*(2*x + 2*y)**2
    assert expr.coeff(z) == (x + y)**2
    assert expr.coeff(x + y) == 0
    assert expr2.coeff(z) == (x + y)**2 + (2*x + 2*y)**2

    assert (x + y + 3*z).coeff(1) == x + y
    assert (-x + 2*y).coeff(-1) == x
    assert (x - 2*y).coeff(-1) == 2*y
    assert (3 + 2*x + 4*x**2).coeff(1) == 0
    assert (-x - 2*y).coeff(2) == -y
    assert (x + sqrt(2)*x).coeff(sqrt(2)) == x
    assert (3 + 2*x + 4*x**2).coeff(x) == 2
    assert (3 + 2*x + 4*x**2).coeff(x**2) == 4
    assert (3 + 2*x + 4*x**2).coeff(x**3) == 0
    assert (z*(x + y)**2).coeff((x + y)**2) == z
    assert (z*(x + y)**2).coeff(x + y) == 0
    assert (2 + 2*x + (x + 1)*y).coeff(x + 1) == y

    assert (x + 2*y + 3).coeff(1) == x
    assert (x + 2*y + 3).coeff(x, 0) == 2*y + 3
    assert (x**2 + 2*y + 3*x).coeff(x**2, 0) == 2*y + 3*x
    assert x.coeff(0, 0) == 0
    assert x.coeff(x, 0) == 0

    n, m, o, l = symbols('n m o l', commutative=False)
    assert n.coeff(n) == 1
    assert y.coeff(n) == 0
    assert (3*n).coeff(n) == 3
    assert (2 + n).coeff(x*m) == 0
    assert (2*x*n*m).coeff(x) == 2*n*m
    assert (2 + n).coeff(x*m*n + y) == 0
    assert (2*x*n*m).coeff(3*n) == 0
    assert (n*m + m*n*m).coeff(n) == 1 + m
    assert (n*m + m*n*m).coeff(n, right=True) == m  # = (1 + m)*n*m
    assert (n*m + m*n).coeff(n) == 0
    assert (n*m + o*m*n).coeff(m*n) == o
    assert (n*m + o*m*n).coeff(m*n, right=True) == 1
    assert (n*m + n*m*n).coeff(n*m, right=True) == 1 + n  # = n*m*(n + 1)

    assert (x*y).coeff(z, 0) == x*y

    assert (x*n + y*n + z*m).coeff(n) == x + y
    assert (n*m + n*o + o*l).coeff(n, right=True) == m + o
    assert (x*n*m*n + y*n*m*o + z*l).coeff(m, right=True) == x*n + y*o
    assert (x*n*m*n + x*n*m*o + z*l).coeff(m, right=True) == n + o
    assert (x*n*m*n + x*n*m*o + z*l).coeff(m) == x*n


def test_coeff2():
    r, kappa = symbols('r, kappa')
    psi = Function("psi")
    g = 1/r**2 * (2*r*psi(r).diff(r, 1) + r**2 * psi(r).diff(r, 2))
    g = g.expand()
    assert g.coeff(psi(r).diff(r)) == 2/r


def test_coeff2_0():
    r, kappa = symbols('r, kappa')
    psi = Function("psi")
    g = 1/r**2 * (2*r*psi(r).diff(r, 1) + r**2 * psi(r).diff(r, 2))
    g = g.expand()

    assert g.coeff(psi(r).diff(r, 2)) == 1


def test_coeff_expand():
    expr = z*(x + y)**2
    expr2 = z*(x + y)**2 + z*(2*x + 2*y)**2
    assert expr.coeff(z) == (x + y)**2
    assert expr2.coeff(z) == (x + y)**2 + (2*x + 2*y)**2


def test_integrate():
    assert x.integrate(x) == x**2/2
    assert x.integrate((x, 0, 1)) == S.Half


def test_as_base_exp():
    assert x.as_base_exp() == (x, S.One)
    assert (x*y*z).as_base_exp() == (x*y*z, S.One)
    assert (x + y + z).as_base_exp() == (x + y + z, S.One)
    assert ((x + y)**z).as_base_exp() == (x + y, z)
    assert (x**2*y**2).as_base_exp() == (x*y, 2)
    assert (x**z*y**z).as_base_exp() == (x**z*y**z, S.One)


def test_issue_4963():
    assert hasattr(Mul(x, y), "is_commutative")
    assert hasattr(Mul(x, y, evaluate=False), "is_commutative")
    assert hasattr(Pow(x, y), "is_commutative")
    assert hasattr(Pow(x, y, evaluate=False), "is_commutative")
    expr = Mul(Pow(2, 2, evaluate=False), 3, evaluate=False) + 1
    assert hasattr(expr, "is_commutative")


def test_action_verbs():
    assert nsimplify(1/(exp(3*pi*x/5) + 1)) == \
        (1/(exp(3*pi*x/5) + 1)).nsimplify()
    assert ratsimp(1/x + 1/y) == (1/x + 1/y).ratsimp()
    assert trigsimp(log(x), deep=True) == (log(x)).trigsimp(deep=True)
    assert radsimp(1/(2 + sqrt(2))) == (1/(2 + sqrt(2))).radsimp()
    assert radsimp(1/(a + b*sqrt(c)), symbolic=False) == \
        (1/(a + b*sqrt(c))).radsimp(symbolic=False)
    assert powsimp(x**y*x**z*y**z, combine='all') == \
        (x**y*x**z*y**z).powsimp(combine='all')
    assert (x**t*y**t).powsimp(force=True) == (x*y)**t
    assert simplify(x**y*x**z*y**z) == (x**y*x**z*y**z).simplify()
    assert together(1/x + 1/y) == (1/x + 1/y).together()
    assert collect(a*x**2 + b*x**2 + a*x - b*x + c, x) == \
        (a*x**2 + b*x**2 + a*x - b*x + c).collect(x)
    assert apart(y/(y + 2)/(y + 1), y) == (y/(y + 2)/(y + 1)).apart(y)
    assert combsimp(y/(x + 2)/(x + 1)) == (y/(x + 2)/(x + 1)).combsimp()
    assert gammasimp(gamma(x)/gamma(x-5)) == (gamma(x)/gamma(x-5)).gammasimp()
    assert factor(x**2 + 5*x + 6) == (x**2 + 5*x + 6).factor()
    assert refine(sqrt(x**2)) == sqrt(x**2).refine()
    assert cancel((x**2 + 5*x + 6)/(x + 2)) == ((x**2 + 5*x + 6)/(x + 2)).cancel()


def test_as_powers_dict():
    assert x.as_powers_dict() == {x: 1}
    assert (x**y*z).as_powers_dict() == {x: y, z: 1}
    assert Mul(2, 2, evaluate=False).as_powers_dict() == {S(2): S(2)}
    assert (x*y).as_powers_dict()[z] == 0
    assert (x + y).as_powers_dict()[z] == 0


def test_as_coefficients_dict():
    check = [S.One, x, y, x*y, 1]
    assert [Add(3*x, 2*x, y, 3).as_coefficients_dict()[i] for i in check] == \
        [3, 5, 1, 0, 3]
    assert [Add(3*x, 2*x, y, 3, evaluate=False).as_coefficients_dict()[i]
            for i in check] == [3, 5, 1, 0, 3]
    assert [(3*x*y).as_coefficients_dict()[i] for i in check] == \
        [0, 0, 0, 3, 0]
    assert [(3.0*x*y).as_coefficients_dict()[i] for i in check] == \
        [0, 0, 0, 3.0, 0]
    assert (3.0*x*y).as_coefficients_dict()[3.0*x*y] == 0
    eq = x*(x + 1)*a + x*b + c/x
    assert eq.as_coefficients_dict(x) == {x: b, 1/x: c,
        x*(x + 1): a}
    assert eq.expand().as_coefficients_dict(x) == {x**2: a, x: a + b, 1/x: c}
    assert x.as_coefficients_dict() == {x: S.One}


def test_args_cnc():
    A = symbols('A', commutative=False)
    assert (x + A).args_cnc() == \
        [[], [x + A]]
    assert (x + a).args_cnc() == \
        [[a + x], []]
    assert (x*a).args_cnc() == \
        [[a, x], []]
    assert (x*y*A*(A + 1)).args_cnc(cset=True) == \
        [{x, y}, [A, 1 + A]]
    assert Mul(x, x, evaluate=False).args_cnc(cset=True, warn=False) == \
        [{x}, []]
    assert Mul(x, x**2, evaluate=False).args_cnc(cset=True, warn=False) == \
        [{x, x**2}, []]
    raises(ValueError, lambda: Mul(x, x, evaluate=False).args_cnc(cset=True))
    assert Mul(x, y, x, evaluate=False).args_cnc() == \
        [[x, y, x], []]
    # always split -1 from leading number
    assert (-1.*x).args_cnc() == [[-1, 1.0, x], []]


def test_new_rawargs():
    n = Symbol('n', commutative=False)
    a = x + n
    assert a.is_commutative is False
    assert a._new_rawargs(x).is_commutative
    assert a._new_rawargs(x, y).is_commutative
    assert a._new_rawargs(x, n).is_commutative is False
    assert a._new_rawargs(x, y, n).is_commutative is False
    m = x*n
    assert m.is_commutative is False
    assert m._new_rawargs(x).is_commutative
    assert m._new_rawargs(n).is_commutative is False
    assert m._new_rawargs(x, y).is_commutative
    assert m._new_rawargs(x, n).is_commutative is False
    assert m._new_rawargs(x, y, n).is_commutative is False

    assert m._new_rawargs(x, n, reeval=False).is_commutative is False
    assert m._new_rawargs(S.One) is S.One


def test_issue_5226():
    assert Add(evaluate=False) == 0
    assert Mul(evaluate=False) == 1
    assert Mul(x + y, evaluate=False).is_Add


def test_free_symbols():
    # free_symbols should return the free symbols of an object
    assert S.One.free_symbols == set()
    assert x.free_symbols == {x}
    assert Integral(x, (x, 1, y)).free_symbols == {y}
    assert (-Integral(x, (x, 1, y))).free_symbols == {y}
    assert meter.free_symbols == set()
    assert (meter**x).free_symbols == {x}


def test_has_free():
    assert x.has_free(x)
    assert not x.has_free(y)
    assert (x + y).has_free(x)
    assert (x + y).has_free(*(x, z))
    assert f(x).has_free(x)
    assert f(x).has_free(f(x))
    assert Integral(f(x), (f(x), 1, y)).has_free(y)
    assert not Integral(f(x), (f(x), 1, y)).has_free(x)
    assert not Integral(f(x), (f(x), 1, y)).has_free(f(x))
    # simple extraction
    assert (x + 1 + y).has_free(x + 1)
    assert not (x + 2 + y).has_free(x + 1)
    assert (2 + 3*x*y).has_free(3*x)
    raises(TypeError, lambda: x.has_free({x, y}))
    s = FiniteSet(1, 2)
    assert Piecewise((s, x > 3), (4, True)).has_free(s)
    assert not Piecewise((1, x > 3), (4, True)).has_free(s)
    # can't make set of these, but fallback will handle
    raises(TypeError, lambda: x.has_free(y, []))


def test_has_xfree():
    assert (x + 1).has_xfree({x})
    assert ((x + 1)**2).has_xfree({x + 1})
    assert not (x + y + 1).has_xfree({x + 1})
    raises(TypeError, lambda: x.has_xfree(x))
    raises(TypeError, lambda: x.has_xfree([x]))


def test_issue_5300():
    x = Symbol('x', commutative=False)
    assert x*sqrt(2)/sqrt(6) == x*sqrt(3)/3


def test_floordiv():
    from sympy.functions.elementary.integers import floor
    assert x // y == floor(x / y)


def test_as_coeff_Mul():
    assert Integer(3).as_coeff_Mul() == (Integer(3), Integer(1))
    assert Rational(3, 4).as_coeff_Mul() == (Rational(3, 4), Integer(1))
    assert Float(5.0).as_coeff_Mul() == (Float(5.0), Integer(1))
    assert Float(0.0).as_coeff_Mul() == (Float(0.0), Integer(1))

    assert (Integer(3)*x).as_coeff_Mul() == (Integer(3), x)
    assert (Rational(3, 4)*x).as_coeff_Mul() == (Rational(3, 4), x)
    assert (Float(5.0)*x).as_coeff_Mul() == (Float(5.0), x)

    assert (Integer(3)*x*y).as_coeff_Mul() == (Integer(3), x*y)
    assert (Rational(3, 4)*x*y).as_coeff_Mul() == (Rational(3, 4), x*y)
    assert (Float(5.0)*x*y).as_coeff_Mul() == (Float(5.0), x*y)

    assert (x).as_coeff_Mul() == (S.One, x)
    assert (x*y).as_coeff_Mul() == (S.One, x*y)
    assert (-oo*x).as_coeff_Mul(rational=True) == (-1, oo*x)


def test_as_coeff_Add():
    assert Integer(3).as_coeff_Add() == (Integer(3), Integer(0))
    assert Rational(3, 4).as_coeff_Add() == (Rational(3, 4), Integer(0))
    assert Float(5.0).as_coeff_Add() == (Float(5.0), Integer(0))

    assert (Integer(3) + x).as_coeff_Add() == (Integer(3), x)
    assert (Rational(3, 4) + x).as_coeff_Add() == (Rational(3, 4), x)
    assert (Float(5.0) + x).as_coeff_Add() == (Float(5.0), x)
    assert (Float(5.0) + x).as_coeff_Add(rational=True) == (0, Float(5.0) + x)

    assert (Integer(3) + x + y).as_coeff_Add() == (Integer(3), x + y)
    assert (Rational(3, 4) + x + y).as_coeff_Add() == (Rational(3, 4), x + y)
    assert (Float(5.0) + x + y).as_coeff_Add() == (Float(5.0), x + y)

    assert (x).as_coeff_Add() == (S.Zero, x)
    assert (x*y).as_coeff_Add() == (S.Zero, x*y)


def test_expr_sorting():

    exprs = [1/x**2, 1/x, sqrt(sqrt(x)), sqrt(x), x, sqrt(x)**3, x**2]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [x, 2*x, 2*x**2, 2*x**3, x**n, 2*x**n, sin(x), sin(x)**n,
             sin(x**2), cos(x), cos(x**2), tan(x)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [x + 1, x**2 + x + 1, x**3 + x**2 + x + 1]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [S(4), x - 3*I/2, x + 3*I/2, x - 4*I + 1, x + 4*I + 1]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [f(x), g(x), exp(x), sin(x), cos(x), factorial(x)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [Tuple(x, y), Tuple(x, z), Tuple(x, y, z)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [[3], [1, 2]]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [[1, 2], [2, 3]]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [[1, 2], [1, 2, 3]]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [{x: -y}, {x: y}]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [{1}, {1, 2}]
    assert sorted(exprs, key=default_sort_key) == exprs

    a, b = exprs = [Dummy('x'), Dummy('x')]
    assert sorted([b, a], key=default_sort_key) == exprs


def test_as_ordered_factors():

    assert x.as_ordered_factors() == [x]
    assert (2*x*x**n*sin(x)*cos(x)).as_ordered_factors() \
        == [Integer(2), x, x**n, sin(x), cos(x)]

    args = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    expr = Mul(*args)

    assert expr.as_ordered_factors() == args

    A, B = symbols('A,B', commutative=False)

    assert (A*B).as_ordered_factors() == [A, B]
    assert (B*A).as_ordered_factors() == [B, A]


def test_as_ordered_terms():

    assert x.as_ordered_terms() == [x]
    assert (sin(x)**2*cos(x) + sin(x)*cos(x)**2 + 1).as_ordered_terms() \
        == [sin(x)**2*cos(x), sin(x)*cos(x)**2, 1]

    args = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    expr = Add(*args)

    assert expr.as_ordered_terms() == args

    assert (1 + 4*sqrt(3)*pi*x).as_ordered_terms() == [4*pi*x*sqrt(3), 1]

    assert ( 2 + 3*I).as_ordered_terms() == [2, 3*I]
    assert (-2 + 3*I).as_ordered_terms() == [-2, 3*I]
    assert ( 2 - 3*I).as_ordered_terms() == [2, -3*I]
    assert (-2 - 3*I).as_ordered_terms() == [-2, -3*I]

    assert ( 4 + 3*I).as_ordered_terms() == [4, 3*I]
    assert (-4 + 3*I).as_ordered_terms() == [-4, 3*I]
    assert ( 4 - 3*I).as_ordered_terms() == [4, -3*I]
    assert (-4 - 3*I).as_ordered_terms() == [-4, -3*I]

    e = x**2*y**2 + x*y**4 + y + 2

    assert e.as_ordered_terms(order="lex") == [x**2*y**2, x*y**4, y, 2]
    assert e.as_ordered_terms(order="grlex") == [x*y**4, x**2*y**2, y, 2]
    assert e.as_ordered_terms(order="rev-lex") == [2, y, x*y**4, x**2*y**2]
    assert e.as_ordered_terms(order="rev-grlex") == [2, y, x**2*y**2, x*y**4]

    k = symbols('k')
    assert k.as_ordered_terms(data=True) == ([(k, ((1.0, 0.0), (1,), ()))], [k])


def test_sort_key_atomic_expr():
    from sympy.physics.units import m, s
    assert sorted([-m, s], key=lambda arg: arg.sort_key()) == [-m, s]


def test_eval_interval():
    assert exp(x)._eval_interval(*Tuple(x, 0, 1)) == exp(1) - exp(0)

    # issue 4199
    a = x/y
    raises(NotImplementedError, lambda: a._eval_interval(x, S.Zero, oo)._eval_interval(y, oo, S.Zero))
    raises(NotImplementedError, lambda: a._eval_interval(x, S.Zero, oo)._eval_interval(y, S.Zero, oo))
    a = x - y
    raises(NotImplementedError, lambda: a._eval_interval(x, S.One, oo)._eval_interval(y, oo, S.One))
    raises(ValueError, lambda: x._eval_interval(x, None, None))
    a = -y*Heaviside(x - y)
    assert a._eval_interval(x, -oo, oo) == -y
    assert a._eval_interval(x, oo, -oo) == y


def test_eval_interval_zoo():
    # Test that limit is used when zoo is returned
    assert Si(1/x)._eval_interval(x, S.Zero, S.One) == -pi/2 + Si(1)


def test_primitive():
    assert (3*(x + 1)**2).primitive() == (3, (x + 1)**2)
    assert (6*x + 2).primitive() == (2, 3*x + 1)
    assert (x/2 + 3).primitive() == (S.Half, x + 6)
    eq = (6*x + 2)*(x/2 + 3)
    assert eq.primitive()[0] == 1
    eq = (2 + 2*x)**2
    assert eq.primitive()[0] == 1
    assert (4.0*x).primitive() == (1, 4.0*x)
    assert (4.0*x + y/2).primitive() == (S.Half, 8.0*x + y)
    assert (-2*x).primitive() == (2, -x)
    assert Add(5*z/7, 0.5*x, 3*y/2, evaluate=False).primitive() == \
        (S.One/14, 7.0*x + 21*y + 10*z)
    for i in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
        assert (i + x/3).primitive() == \
            (S.One/3, i + x)
    assert (S.Infinity + 2*x/3 + 4*y/7).primitive() == \
        (S.One/21, 14*x + 12*y + oo)
    assert S.Zero.primitive() == (S.One, S.Zero)


def test_issue_5843():
    a = 1 + x
    assert (2*a).extract_multiplicatively(a) == 2
    assert (4*a).extract_multiplicatively(2*a) == 2
    assert ((3*a)*(2*a)).extract_multiplicatively(a) == 6*a


def test_is_constant():
    from sympy.solvers.solvers import checksol
    assert Sum(x, (x, 1, 10)).is_constant() is True
    assert Sum(x, (x, 1, n)).is_constant() is False
    assert Sum(x, (x, 1, n)).is_constant(y) is True
    assert Sum(x, (x, 1, n)).is_constant(n) is False
    assert Sum(x, (x, 1, n)).is_constant(x) is True
    eq = a*cos(x)**2 + a*sin(x)**2 - a
    assert eq.is_constant() is True
    assert eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
    assert x.is_constant() is False
    assert x.is_constant(y) is True
    assert log(x/y).is_constant() is False

    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    assert f(1).is_constant
    assert checksol(x, x, f(x)) is False

    assert Pow(x, S.Zero, evaluate=False).is_constant() is True  # == 1
    assert Pow(S.Zero, x, evaluate=False).is_constant() is False  # == 0 or 1
    assert (2**x).is_constant() is False
    assert Pow(S(2), S(3), evaluate=False).is_constant() is True

    z1, z2 = symbols('z1 z2', zero=True)
    assert (z1 + 2*z2).is_constant() is True

    assert meter.is_constant() is True
    assert (3*meter).is_constant() is True
    assert (x*meter).is_constant() is False


def test_equals():
    assert (-3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2).equals(0)
    assert (x**2 - 1).equals((x + 1)*(x - 1))
    assert (cos(x)**2 + sin(x)**2).equals(1)
    assert (a*cos(x)**2 + a*sin(x)**2).equals(a)
    r = sqrt(2)
    assert (-1/(r + r*x) + 1/r/(1 + x)).equals(0)
    assert factorial(x + 1).equals((x + 1)*factorial(x))
    assert sqrt(3).equals(2*sqrt(3)) is False
    assert (sqrt(5)*sqrt(3)).equals(sqrt(3)) is False
    assert (sqrt(5) + sqrt(3)).equals(0) is False
    assert (sqrt(5) + pi).equals(0) is False
    assert meter.equals(0) is False
    assert (3*meter**2).equals(0) is False
    eq = -(-1)**(S(3)/4)*6**(S.One/4) + (-6)**(S.One/4)*I
    if eq != 0:  # if canonicalization makes this zero, skip the test
        assert eq.equals(0)
    assert sqrt(x).equals(0) is False

    # from integrate(x*sqrt(1 + 2*x), x);
    # diff is zero only when assumptions allow
    i = 2*sqrt(2)*x**(S(5)/2)*(1 + 1/(2*x))**(S(5)/2)/5 + \
        2*sqrt(2)*x**(S(3)/2)*(1 + 1/(2*x))**(S(5)/2)/(-6 - 3/x)
    ans = sqrt(2*x + 1)*(6*x**2 + x - 1)/15
    diff = i - ans
    assert diff.equals(0) is None  # should be False, but previously this was False due to wrong intermediate result
    assert diff.subs(x, Rational(-1, 2)/2) == 7*sqrt(2)/120
    # there are regions for x for which the expression is True, for
    # example, when x < -1/2 or x > 0 the expression is zero
    p = Symbol('p', positive=True)
    assert diff.subs(x, p).equals(0) is True
    assert diff.subs(x, -1).equals(0) is True

    # prove via minimal_polynomial or self-consistency
    eq = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    assert eq.equals(0)
    q = 3**Rational(1, 3) + 3
    p = expand(q**3)**Rational(1, 3)
    assert (p - q).equals(0)

    # issue 6829
    # eq = q*x + q/4 + x**4 + x**3 + 2*x**2 - S.One/3
    # z = eq.subs(x, solve(eq, x)[0])
    q = symbols('q')
    z = (q*(-sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12)/2 - sqrt((2*q - S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/6)/2 - S.One/4) + q/4 + (-sqrt(-2*(-(q
    - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q
    - S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**4 + (-sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q -
    S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**3 + 2*(-sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q -
    S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**2 - Rational(1, 3))
    assert z.equals(0)


def test_random():
    from sympy.functions.combinatorial.numbers import lucas
    from sympy.simplify.simplify import posify
    assert posify(x)[0]._random() is not None
    assert lucas(n)._random(2, -2, 0, -1, 1) is None

    # issue 8662
    assert Piecewise((Max(x, y), z))._random() is None


def test_round():
    assert str(Float('0.1249999').round(2)) == '0.12'
    d20 = 12345678901234567890
    ans = S(d20).round(2)
    assert ans.is_Integer and ans == d20
    ans = S(d20).round(-2)
    assert ans.is_Integer and ans == 12345678901234567900
    assert str(S('1/7').round(4)) == '0.1429'
    assert str(S('.[12345]').round(4)) == '0.1235'
    assert str(S('.1349').round(2)) == '0.13'
    n = S(12345)
    ans = n.round()
    assert ans.is_Integer
    assert ans == n
    ans = n.round(1)
    assert ans.is_Integer
    assert ans == n
    ans = n.round(4)
    assert ans.is_Integer
    assert ans == n
    assert n.round(-1) == 12340

    r = Float(str(n)).round(-4)
    assert r == 10000.0

    assert n.round(-5) == 0

    assert str((pi + sqrt(2)).round(2)) == '4.56'
    assert (10*(pi + sqrt(2))).round(-1) == 50.0
    raises(TypeError, lambda: round(x + 2, 2))
    assert str(S(2.3).round(1)) == '2.3'
    # rounding in SymPy (as in Decimal) should be
    # exact for the given precision; we check here
    # that when a 5 follows the last digit that
    # the rounded digit will be even.
    for i in range(-99, 100):
        # construct a decimal that ends in 5, e.g. 123 -> 0.1235
        s = str(abs(i))
        p = len(s)  # we are going to round to the last digit of i
        n = '0.%s5' % s  # put a 5 after i's digits
        j = p + 2  # 2 for '0.'
        if i < 0:  # 1 for '-'
            j += 1
            n = '-' + n
        v = str(Float(n).round(p))[:j]  # pertinent digits
        if v.endswith('.'):
            continue  # it ends with 0 which is even
        L = int(v[-1])  # last digit
        assert L % 2 == 0, (n, '->', v)

    assert (Float(.3, 3) + 2*pi).round() == 7
    assert (Float(.3, 3) + 2*pi*100).round() == 629
    assert (pi + 2*E*I).round() == 3 + 5*I
    # don't let request for extra precision give more than
    # what is known (in this case, only 3 digits)
    assert str((Float(.03, 3) + 2*pi/100).round(5)) == '0.0928'
    assert str((Float(.03, 3) + 2*pi/100).round(4)) == '0.0928'

    assert S.Zero.round() == 0

    a = (Add(1, Float('1.' + '9'*27, ''), evaluate=False))
    assert a.round(10) == Float('3.000000000000000000000000000', '')
    assert a.round(25) == Float('3.000000000000000000000000000', '')
    assert a.round(26) == Float('3.000000000000000000000000000', '')
    assert a.round(27) == Float('2.999999999999999999999999999', '')
    assert a.round(30) == Float('2.999999999999999999999999999', '')
    #assert a.round(10) == Float('3.0000000000', '')
    #assert a.round(25) == Float('3.0000000000000000000000000', '')
    #assert a.round(26) == Float('3.00000000000000000000000000', '')
    #assert a.round(27) == Float('2.999999999999999999999999999', '')
    #assert a.round(30) == Float('2.999999999999999999999999999', '')

    # XXX: Should round set the precision of the result?
    #      The previous version of the tests above is this but they only pass
    #      because Floats with unequal precision compare equal:
    #
    # assert a.round(10) == Float('3.0000000000', '')
    # assert a.round(25) == Float('3.0000000000000000000000000', '')
    # assert a.round(26) == Float('3.00000000000000000000000000', '')
    # assert a.round(27) == Float('2.999999999999999999999999999', '')
    # assert a.round(30) == Float('2.999999999999999999999999999', '')

    raises(TypeError, lambda: x.round())
    raises(TypeError, lambda: f(1).round())

    # exact magnitude of 10
    assert str(S.One.round()) == '1'
    assert str(S(100).round()) == '100'

    # applied to real and imaginary portions
    assert (2*pi + E*I).round() == 6 + 3*I
    assert (2*pi + I/10).round() == 6
    assert (pi/10 + 2*I).round() == 2*I
    # the lhs re and im parts are Float with dps of 2
    # and those on the right have dps of 15 so they won't compare
    # equal unless we use string or compare components (which will
    # then coerce the floats to the same precision) or re-create
    # the floats
    assert str((pi/10 + E*I).round(2)) == '0.31 + 2.72*I'
    assert str((pi/10 + E*I).round(2).as_real_imag()) == '(0.31, 2.72)'
    assert str((pi/10 + E*I).round(2)) == '0.31 + 2.72*I'

    # issue 6914
    assert (I**(I + 3)).round(3) == Float('-0.208', '')*I

    # issue 8720
    assert S(-123.6).round() == -124
    assert S(-1.5).round() == -2
    assert S(-100.5).round() == -100
    assert S(-1.5 - 10.5*I).round() == -2 - 10*I

    # issue 7961
    assert str(S(0.006).round(2)) == '0.01'
    assert str(S(0.00106).round(4)) == '0.0011'

    # issue 8147
    assert S.NaN.round() is S.NaN
    assert S.Infinity.round() is S.Infinity
    assert S.NegativeInfinity.round() is S.NegativeInfinity
    assert S.ComplexInfinity.round() is S.ComplexInfinity

    # check that types match
    for i in range(2):
        fi = float(i)
        # 2 args
        assert all(type(round(i, p)) is int for p in (-1, 0, 1))
        assert all(S(i).round(p).is_Integer for p in (-1, 0, 1))
        assert all(type(round(fi, p)) is float for p in (-1, 0, 1))
        assert all(S(fi).round(p).is_Float for p in (-1, 0, 1))
        # 1 arg (p is None)
        assert type(round(i)) is int
        assert S(i).round().is_Integer
        assert type(round(fi)) is int
        assert S(fi).round().is_Integer

        # issue 25698
        n = 6000002
        assert int(n*(log(n) + log(log(n)))) == 110130079
        one = cos(2)**2 + sin(2)**2
        eq = exp(one*I*pi)
        qr, qi = eq.as_real_imag()
        assert qi.round(2) == 0.0
        assert eq.round(2) == -1.0
        eq = one - 1/S(10**120)
        assert S.true not in (eq > 1, eq < 1)
        assert int(eq) == int(.9) == 0
        assert int(-eq) == int(-.9) == 0


def test_held_expression_UnevaluatedExpr():
    x = symbols("x")
    he = UnevaluatedExpr(1/x)
    e1 = x*he

    assert isinstance(e1, Mul)
    assert e1.args == (x, he)
    assert e1.doit() == 1
    assert UnevaluatedExpr(Derivative(x, x)).doit(deep=False
        ) == Derivative(x, x)
    assert UnevaluatedExpr(Derivative(x, x)).doit() == 1

    xx = Mul(x, x, evaluate=False)
    assert xx != x**2

    ue2 = UnevaluatedExpr(xx)
    assert isinstance(ue2, UnevaluatedExpr)
    assert ue2.args == (xx,)
    assert ue2.doit() == x**2
    assert ue2.doit(deep=False) == xx

    x2 = UnevaluatedExpr(2)*2
    assert type(x2) is Mul
    assert x2.args == (2, UnevaluatedExpr(2))

def test_round_exception_nostr():
    # Don't use the string form of the expression in the round exception, as
    # it's too slow
    s = Symbol('bad')
    try:
        s.round()
    except TypeError as e:
        assert 'bad' not in str(e)
    else:
        # Did not raise
        raise AssertionError("Did not raise")


def test_extract_branch_factor():
    assert exp_polar(2.0*I*pi).extract_branch_factor() == (1, 1)


def test_identity_removal():
    assert Add.make_args(x + 0) == (x,)
    assert Mul.make_args(x*1) == (x,)


def test_float_0():
    assert Float(0.0) + 1 == Float(1.0)


@XFAIL
def test_float_0_fail():
    assert Float(0.0)*x == Float(0.0)
    assert (x + Float(0.0)).is_Add


def test_issue_6325():
    ans = (b**2 + z**2 - (b*(a + b*t) + z*(c + t*z))**2/(
        (a + b*t)**2 + (c + t*z)**2))/sqrt((a + b*t)**2 + (c + t*z)**2)
    e = sqrt((a + b*t)**2 + (c + z*t)**2)
    assert diff(e, t, 2) == ans
    assert e.diff(t, 2) == ans
    assert diff(e, t, 2, simplify=False) != ans


def test_issue_7426():
    f1 = a % c
    f2 = x % z
    assert f1.equals(f2) is None


def test_issue_11122():
    x = Symbol('x', extended_positive=False)
    assert unchanged(Gt, x, 0)  # (x > 0)
    # (x > 0) should remain unevaluated after PR #16956

    x = Symbol('x', positive=False, real=True)
    assert (x > 0) is S.false


def test_issue_10651():
    x = Symbol('x', real=True)
    e1 = (-1 + x)/(1 - x)
    e3 = (4*x**2 - 4)/((1 - x)*(1 + x))
    e4 = 1/(cos(x)**2) - (tan(x))**2
    x = Symbol('x', positive=True)
    e5 = (1 + x)/x
    assert e1.is_constant() is None
    assert e3.is_constant() is None
    assert e4.is_constant() is None
    assert e5.is_constant() is False


def test_issue_10161():
    x = symbols('x', real=True)
    assert x*abs(x)*abs(x) == x**3


def test_issue_10755():
    x = symbols('x')
    raises(TypeError, lambda: int(log(x)))
    raises(TypeError, lambda: log(x).round(2))


def test_issue_11877():
    x = symbols('x')
    assert integrate(log(S.Half - x), (x, 0, S.Half)) == Rational(-1, 2) -log(2)/2


def test_normal():
    x = symbols('x')
    e = Mul(S.Half, 1 + x, evaluate=False)
    assert e.normal() == e


def test_expr():
    x = symbols('x')
    raises(TypeError, lambda: tan(x).series(x, 2, oo, "+"))


def test_ExprBuilder():
    eb = ExprBuilder(Mul)
    eb.args.extend([x, x])
    assert eb.build() == x**2


def test_issue_22020():
    from sympy.parsing.sympy_parser import parse_expr
    x = parse_expr("log((2*V/3-V)/C)/-(R+r)*C")
    y = parse_expr("log((2*V/3-V)/C)/-(R+r)*2")
    assert x.equals(y) is False


def test_non_string_equality():
    # Expressions should not compare equal to strings
    x = symbols('x')
    one = sympify(1)
    assert (x == 'x') is False
    assert (x != 'x') is True
    assert (one == '1') is False
    assert (one != '1') is True
    assert (x + 1 == 'x + 1') is False
    assert (x + 1 != 'x + 1') is True

    # Make sure == doesn't try to convert the resulting expression to a string
    # (e.g., by calling sympify() instead of _sympify())

    class BadRepr:
        def __repr__(self):
            raise RuntimeError

    assert (x == BadRepr()) is False
    assert (x != BadRepr()) is True


def test_21494():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert x.expr_free_symbols == {x}

    with warns_deprecated_sympy():
        assert Basic().expr_free_symbols == set()

    with warns_deprecated_sympy():
        assert S(2).expr_free_symbols == {S(2)}

    with warns_deprecated_sympy():
        assert Indexed("A", x).expr_free_symbols == {Indexed("A", x)}

    with warns_deprecated_sympy():
        assert Subs(x, x, 0).expr_free_symbols == set()


def test_Expr__eq__iterable_handling():
    assert x != range(3)


def test_format():
    assert '{:1.2f}'.format(S.Zero) == '0.00'
    assert '{:+3.0f}'.format(S(3)) == ' +3'
    assert '{:23.20f}'.format(pi) == ' 3.14159265358979323846'
    assert '{:50.48f}'.format(exp(sin(1))) == '2.319776824715853173956590377503266813254904772376'


def test_issue_24045():
    assert powsimp(exp(a)/((c*a - c*b)*(Float(1.0)*c*a - Float(1.0)*c*b)))  # doesn't raise


def test__unevaluated_Mul():
    A, B = symbols('A B', commutative=False)
    assert _unevaluated_Mul(x, A, B, S(2), A).args == (2, x, A, B, A)
    assert _unevaluated_Mul(-x*A*B, S(2), A).args == (-2, x, A, B, A)


def test_Float_zero_division_error():
    # issue 27165
    assert Float('1.7567e-1417').round(15) == Float(0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\utils.py ===
# -*- coding: utf-8 -*-
"""
This module offers general convenience and utility functions for dealing with
datetimes.

.. versionadded:: 2.7.0
"""
from __future__ import unicode_literals

from datetime import datetime, time


def today(tzinfo=None):
    """
    Returns a :py:class:`datetime` representing the current day at midnight

    :param tzinfo:
        The time zone to attach (also used to determine the current day).

    :return:
        A :py:class:`datetime.datetime` object representing the current day
        at midnight.
    """

    dt = datetime.now(tzinfo)
    return datetime.combine(dt.date(), time(0, tzinfo=tzinfo))


def default_tzinfo(dt, tzinfo):
    """
    Sets the ``tzinfo`` parameter on naive datetimes only

    This is useful for example when you are provided a datetime that may have
    either an implicit or explicit time zone, such as when parsing a time zone
    string.

    .. doctest::

        >>> from dateutil.tz import tzoffset
        >>> from dateutil.parser import parse
        >>> from dateutil.utils import default_tzinfo
        >>> dflt_tz = tzoffset("EST", -18000)
        >>> print(default_tzinfo(parse('2014-01-01 12:30 UTC'), dflt_tz))
        2014-01-01 12:30:00+00:00
        >>> print(default_tzinfo(parse('2014-01-01 12:30'), dflt_tz))
        2014-01-01 12:30:00-05:00

    :param dt:
        The datetime on which to replace the time zone

    :param tzinfo:
        The :py:class:`datetime.tzinfo` subclass instance to assign to
        ``dt`` if (and only if) it is naive.

    :return:
        Returns an aware :py:class:`datetime.datetime`.
    """
    if dt.tzinfo is not None:
        return dt
    else:
        return dt.replace(tzinfo=tzinfo)


def within_delta(dt1, dt2, delta):
    """
    Useful for comparing two datetimes that may have a negligible difference
    to be considered equal.
    """
    delta = abs(delta)
    difference = dt1 - dt2
    return -delta <= difference <= delta

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\greenlet\__init__.py ===
# -*- coding: utf-8 -*-
"""
The root of the greenlet package.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    '__version__',
    '_C_API',

    'GreenletExit',
    'error',

    'getcurrent',
    'greenlet',

    'gettrace',
    'settrace',
]

# pylint:disable=no-name-in-module

###
# Metadata
###
__version__ = '3.2.3'
from ._greenlet import _C_API # pylint:disable=no-name-in-module

###
# Exceptions
###
from ._greenlet import GreenletExit
from ._greenlet import error

###
# greenlets
###
from ._greenlet import getcurrent
from ._greenlet import greenlet

###
# tracing
###
try:
    from ._greenlet import gettrace
    from ._greenlet import settrace
except ImportError:
    # Tracing wasn't supported.
    # XXX: The option to disable it was removed in 1.0,
    # so this branch should be dead code.
    pass

###
# Constants
# These constants aren't documented and aren't recommended.
# In 1.0, USE_GC and USE_TRACING are always true, and USE_CONTEXT_VARS
# is the same as ``sys.version_info[:2] >= 3.7``
###
from ._greenlet import GREENLET_USE_CONTEXT_VARS # pylint:disable=unused-import
from ._greenlet import GREENLET_USE_GC # pylint:disable=unused-import
from ._greenlet import GREENLET_USE_TRACING # pylint:disable=unused-import

# Controlling the use of the gc module. Provisional API for this greenlet
# implementation in 2.0.
from ._greenlet import CLOCKS_PER_SEC # pylint:disable=unused-import
from ._greenlet import enable_optional_cleanup # pylint:disable=unused-import
from ._greenlet import get_clocks_used_doing_optional_cleanup # pylint:disable=unused-import

# Other APIS in the _greenlet module are for test support.

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\Dimension.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Dimension(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Dimension()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDimension(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DimensionBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Dimension
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Dimension
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.DimensionValue import DimensionValue
            obj = DimensionValue()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Dimension
    def Denotation(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def DimensionStart(builder):
    builder.StartObject(2)

def Start(builder):
    DimensionStart(builder)

def DimensionAddValue(builder, value):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(value), 0)

def AddValue(builder, value):
    DimensionAddValue(builder, value)

def DimensionAddDenotation(builder, denotation):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(denotation), 0)

def AddDenotation(builder, denotation):
    DimensionAddDenotation(builder, denotation)

def DimensionEnd(builder):
    return builder.EndObject()

def End(builder):
    return DimensionEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\MapType.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class MapType(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MapType()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMapType(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def MapTypeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # MapType
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MapType
    def KeyType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # MapType
    def ValueType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.TypeInfo import TypeInfo
            obj = TypeInfo()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def MapTypeStart(builder):
    builder.StartObject(2)

def Start(builder):
    MapTypeStart(builder)

def MapTypeAddKeyType(builder, keyType):
    builder.PrependInt32Slot(0, keyType, 0)

def AddKeyType(builder, keyType):
    MapTypeAddKeyType(builder, keyType)

def MapTypeAddValueType(builder, valueType):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(valueType), 0)

def AddValueType(builder, valueType):
    MapTypeAddValueType(builder, valueType)

def MapTypeEnd(builder):
    return builder.EndObject()

def End(builder):
    return MapTypeEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\TensorTypeAndShape.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TensorTypeAndShape(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TensorTypeAndShape()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensorTypeAndShape(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def TensorTypeAndShapeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # TensorTypeAndShape
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TensorTypeAndShape
    def ElemType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # TensorTypeAndShape
    def Shape(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.Shape import Shape
            obj = Shape()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TensorTypeAndShapeStart(builder):
    builder.StartObject(2)

def Start(builder):
    TensorTypeAndShapeStart(builder)

def TensorTypeAndShapeAddElemType(builder, elemType):
    builder.PrependInt32Slot(0, elemType, 0)

def AddElemType(builder, elemType):
    TensorTypeAndShapeAddElemType(builder, elemType)

def TensorTypeAndShapeAddShape(builder, shape):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(shape), 0)

def AddShape(builder, shape):
    TensorTypeAndShapeAddShape(builder, shape)

def TensorTypeAndShapeEnd(builder):
    return builder.EndObject()

def End(builder):
    return TensorTypeAndShapeEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\reader\drawings.py ===

# Copyright (c) 2010-2024 openpyxl


from io import BytesIO
from warnings import warn

from openpyxl.xml.functions import fromstring
from openpyxl.xml.constants import IMAGE_NS
from openpyxl.packaging.relationship import (
    get_rel,
    get_rels_path,
    get_dependents,
)
from openpyxl.drawing.spreadsheet_drawing import SpreadsheetDrawing
from openpyxl.drawing.image import Image, PILImage
from openpyxl.chart.chartspace import ChartSpace
from openpyxl.chart.reader import read_chart


def find_images(archive, path):
    """
    Given the path to a drawing file extract charts and images

    Ignore errors due to unsupported parts of DrawingML
    """

    src = archive.read(path)
    tree = fromstring(src)
    try:
        drawing = SpreadsheetDrawing.from_tree(tree)
    except TypeError:
        warn("DrawingML support is incomplete and limited to charts and images only. Shapes and drawings will be lost.")
        return [], []

    rels_path = get_rels_path(path)
    deps = []
    if rels_path in archive.namelist():
        deps = get_dependents(archive, rels_path)

    charts = []
    for rel in drawing._chart_rels:
        try:
            cs = get_rel(archive, deps, rel.id, ChartSpace)
        except TypeError as e:
            warn(f"Unable to read chart {rel.id} from {path} {e}")
            continue
        chart = read_chart(cs)
        chart.anchor = rel.anchor
        charts.append(chart)

    images = []
    if not PILImage: # Pillow not installed, drop images
        return charts, images

    for rel in drawing._blip_rels:
        dep = deps.get(rel.embed)
        if dep.Type == IMAGE_NS:
            try:
                image = Image(BytesIO(archive.read(dep.target)))
            except OSError:
                msg = "The image {0} will be removed because it cannot be read".format(dep.target)
                warn(msg)
                continue
            if image.format.upper() == "WMF": # cannot save
                msg = "{0} image format is not supported so the image is being dropped".format(image.format)
                warn(msg)
                continue
            image.anchor = rel.anchor
            images.append(image)
    return charts, images

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_windows.py ===
import sys
from dataclasses import dataclass


@dataclass
class WindowsConsoleFeatures:
    """Windows features available."""

    vt: bool = False
    """The console supports VT codes."""
    truecolor: bool = False
    """The console supports truecolor."""


try:
    import ctypes
    from ctypes import LibraryLoader

    if sys.platform == "win32":
        windll = LibraryLoader(ctypes.WinDLL)
    else:
        windll = None
        raise ImportError("Not windows")

    from pip._vendor.rich._win32_console import (
        ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        GetConsoleMode,
        GetStdHandle,
        LegacyWindowsError,
    )

except (AttributeError, ImportError, ValueError):
    # Fallback if we can't load the Windows DLL
    def get_windows_console_features() -> WindowsConsoleFeatures:
        features = WindowsConsoleFeatures()
        return features

else:

    def get_windows_console_features() -> WindowsConsoleFeatures:
        """Get windows console features.

        Returns:
            WindowsConsoleFeatures: An instance of WindowsConsoleFeatures.
        """
        handle = GetStdHandle()
        try:
            console_mode = GetConsoleMode(handle)
            success = True
        except LegacyWindowsError:
            console_mode = 0
            success = False
        vt = bool(success and console_mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        truecolor = False
        if vt:
            win_version = sys.getwindowsversion()
            truecolor = win_version.major > 10 or (
                win_version.major == 10 and win_version.build >= 15063
            )
        features = WindowsConsoleFeatures(vt=vt, truecolor=truecolor)
        return features


if __name__ == "__main__":
    import platform

    features = get_windows_console_features()
    from pip._vendor.rich import print

    print(f'platform="{platform.system()}"')
    print(repr(features))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\schema.py ===
# schema.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Compatibility namespace for sqlalchemy.sql.schema and related.

"""

from __future__ import annotations

from .sql.base import SchemaVisitor as SchemaVisitor
from .sql.ddl import _CreateDropBase as _CreateDropBase
from .sql.ddl import _DropView as _DropView
from .sql.ddl import AddConstraint as AddConstraint
from .sql.ddl import BaseDDLElement as BaseDDLElement
from .sql.ddl import CreateColumn as CreateColumn
from .sql.ddl import CreateIndex as CreateIndex
from .sql.ddl import CreateSchema as CreateSchema
from .sql.ddl import CreateSequence as CreateSequence
from .sql.ddl import CreateTable as CreateTable
from .sql.ddl import DDL as DDL
from .sql.ddl import DDLElement as DDLElement
from .sql.ddl import DropColumnComment as DropColumnComment
from .sql.ddl import DropConstraint as DropConstraint
from .sql.ddl import DropConstraintComment as DropConstraintComment
from .sql.ddl import DropIndex as DropIndex
from .sql.ddl import DropSchema as DropSchema
from .sql.ddl import DropSequence as DropSequence
from .sql.ddl import DropTable as DropTable
from .sql.ddl import DropTableComment as DropTableComment
from .sql.ddl import ExecutableDDLElement as ExecutableDDLElement
from .sql.ddl import InvokeDDLBase as InvokeDDLBase
from .sql.ddl import SetColumnComment as SetColumnComment
from .sql.ddl import SetConstraintComment as SetConstraintComment
from .sql.ddl import SetTableComment as SetTableComment
from .sql.ddl import sort_tables as sort_tables
from .sql.ddl import (
    sort_tables_and_constraints as sort_tables_and_constraints,
)
from .sql.naming import conv as conv
from .sql.schema import _get_table_key as _get_table_key
from .sql.schema import BLANK_SCHEMA as BLANK_SCHEMA
from .sql.schema import CheckConstraint as CheckConstraint
from .sql.schema import Column as Column
from .sql.schema import (
    ColumnCollectionConstraint as ColumnCollectionConstraint,
)
from .sql.schema import ColumnCollectionMixin as ColumnCollectionMixin
from .sql.schema import ColumnDefault as ColumnDefault
from .sql.schema import Computed as Computed
from .sql.schema import Constraint as Constraint
from .sql.schema import DefaultClause as DefaultClause
from .sql.schema import DefaultGenerator as DefaultGenerator
from .sql.schema import FetchedValue as FetchedValue
from .sql.schema import ForeignKey as ForeignKey
from .sql.schema import ForeignKeyConstraint as ForeignKeyConstraint
from .sql.schema import HasConditionalDDL as HasConditionalDDL
from .sql.schema import Identity as Identity
from .sql.schema import Index as Index
from .sql.schema import insert_sentinel as insert_sentinel
from .sql.schema import MetaData as MetaData
from .sql.schema import PrimaryKeyConstraint as PrimaryKeyConstraint
from .sql.schema import SchemaConst as SchemaConst
from .sql.schema import SchemaItem as SchemaItem
from .sql.schema import SchemaVisitable as SchemaVisitable
from .sql.schema import Sequence as Sequence
from .sql.schema import Table as Table
from .sql.schema import UniqueConstraint as UniqueConstraint

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\matrix_nodes.py ===
"""
Additional AST nodes for operations on matrices. The nodes in this module
are meant to represent optimization of matrix expressions within codegen's
target languages that cannot be represented by SymPy expressions.

As an example, we can use :meth:`sympy.codegen.rewriting.optimize` and the
``matin_opt`` optimization provided in :mod:`sympy.codegen.rewriting` to
transform matrix multiplication under certain assumptions:

    >>> from sympy import symbols, MatrixSymbol
    >>> n = symbols('n', integer=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> x = MatrixSymbol('x', n, 1)
    >>> expr = A**(-1) * x
    >>> from sympy import assuming, Q
    >>> from sympy.codegen.rewriting import matinv_opt, optimize
    >>> with assuming(Q.fullrank(A)):
    ...     optimize(expr, [matinv_opt])
    MatrixSolve(A, vector=x)
"""

from .ast import Token
from sympy.matrices import MatrixExpr
from sympy.core.sympify import sympify


class MatrixSolve(Token, MatrixExpr):
    """Represents an operation to solve a linear matrix equation.

    Parameters
    ==========

    matrix : MatrixSymbol

      Matrix representing the coefficients of variables in the linear
      equation. This matrix must be square and full-rank (i.e. all columns must
      be linearly independent) for the solving operation to be valid.

    vector : MatrixSymbol

      One-column matrix representing the solutions to the equations
      represented in ``matrix``.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol
    >>> from sympy.codegen.matrix_nodes import MatrixSolve
    >>> n = symbols('n', integer=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> x = MatrixSymbol('x', n, 1)
    >>> from sympy.printing.numpy import NumPyPrinter
    >>> NumPyPrinter().doprint(MatrixSolve(A, x))
    'numpy.linalg.solve(A, x)'
    >>> from sympy import octave_code
    >>> octave_code(MatrixSolve(A, x))
    'A \\\\ x'

    """
    __slots__ = _fields = ('matrix', 'vector')

    _construct_matrix = staticmethod(sympify)
    _construct_vector = staticmethod(sympify)

    @property
    def shape(self):
        return self.vector.shape

    def _eval_derivative(self, x):
        A, b = self.matrix, self.vector
        return MatrixSolve(A, b.diff(x) - A.diff(x) * MatrixSolve(A, b))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\maxima.py ===
import re
from sympy.concrete.products import product
from sympy.concrete.summations import Sum
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)


class MaximaHelpers:
    def maxima_expand(expr):
        return expr.expand()

    def maxima_float(expr):
        return expr.evalf()

    def maxima_trigexpand(expr):
        return expr.expand(trig=True)

    def maxima_sum(a1, a2, a3, a4):
        return Sum(a1, (a2, a3, a4)).doit()

    def maxima_product(a1, a2, a3, a4):
        return product(a1, (a2, a3, a4))

    def maxima_csc(expr):
        return 1/sin(expr)

    def maxima_sec(expr):
        return 1/cos(expr)

sub_dict = {
    'pi': re.compile(r'%pi'),
    'E': re.compile(r'%e'),
    'I': re.compile(r'%i'),
    '**': re.compile(r'\^'),
    'oo': re.compile(r'\binf\b'),
    '-oo': re.compile(r'\bminf\b'),
    "'-'": re.compile(r'\bminus\b'),
    'maxima_expand': re.compile(r'\bexpand\b'),
    'maxima_float': re.compile(r'\bfloat\b'),
    'maxima_trigexpand': re.compile(r'\btrigexpand'),
    'maxima_sum': re.compile(r'\bsum\b'),
    'maxima_product': re.compile(r'\bproduct\b'),
    'cancel': re.compile(r'\bratsimp\b'),
    'maxima_csc': re.compile(r'\bcsc\b'),
    'maxima_sec': re.compile(r'\bsec\b')
}

var_name = re.compile(r'^\s*(\w+)\s*:')


def parse_maxima(str, globals=None, name_dict={}):
    str = str.strip()
    str = str.rstrip('; ')

    for k, v in sub_dict.items():
        str = v.sub(k, str)

    assign_var = None
    var_match = var_name.search(str)
    if var_match:
        assign_var = var_match.group(1)
        str = str[var_match.end():].strip()

    dct = MaximaHelpers.__dict__.copy()
    dct.update(name_dict)
    obj = sympify(str, locals=dct)

    if assign_var and globals:
        globals[assign_var] = obj

    return obj

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_tracing.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

import trio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


async def coro1(event: trio.Event) -> None:
    event.set()
    await trio.sleep_forever()


async def coro2(event: trio.Event) -> None:
    await coro1(event)


async def coro3(event: trio.Event) -> None:
    await coro2(event)


async def coro2_async_gen(event: trio.Event) -> AsyncGenerator[None, None]:
    # mypy does not like `yield await trio.lowlevel.checkpoint()` - but that
    # should be equivalent to splitting the statement
    await trio.lowlevel.checkpoint()
    yield
    await coro1(event)
    yield  # pragma: no cover
    await trio.lowlevel.checkpoint()  # pragma: no cover
    yield  # pragma: no cover


async def coro3_async_gen(event: trio.Event) -> None:
    async for _ in coro2_async_gen(event):
        pass


async def test_task_iter_await_frames() -> None:
    async with trio.open_nursery() as nursery:
        event = trio.Event()
        nursery.start_soon(coro3, event)
        await event.wait()

        (task,) = nursery.child_tasks

        assert [frame.f_code.co_name for frame, _ in task.iter_await_frames()][:3] == [
            "coro3",
            "coro2",
            "coro1",
        ]

        nursery.cancel_scope.cancel()


async def test_task_iter_await_frames_async_gen() -> None:
    async with trio.open_nursery() as nursery:
        event = trio.Event()
        nursery.start_soon(coro3_async_gen, event)
        await event.wait()

        (task,) = nursery.child_tasks

        assert [frame.f_code.co_name for frame, _ in task.iter_await_frames()][:3] == [
            "coro3_async_gen",
            "coro2_async_gen",
            "coro1",
        ]

        nursery.cancel_scope.cancel()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\fields\__init__.py ===
from wtforms.fields.choices import RadioField
from wtforms.fields.choices import SelectField
from wtforms.fields.choices import SelectFieldBase
from wtforms.fields.choices import SelectMultipleField
from wtforms.fields.core import Field
from wtforms.fields.core import Flags
from wtforms.fields.core import Label
from wtforms.fields.datetime import DateField
from wtforms.fields.datetime import DateTimeField
from wtforms.fields.datetime import DateTimeLocalField
from wtforms.fields.datetime import MonthField
from wtforms.fields.datetime import TimeField
from wtforms.fields.datetime import WeekField
from wtforms.fields.form import FormField
from wtforms.fields.list import FieldList
from wtforms.fields.numeric import DecimalField
from wtforms.fields.numeric import DecimalRangeField
from wtforms.fields.numeric import FloatField
from wtforms.fields.numeric import IntegerField
from wtforms.fields.numeric import IntegerRangeField
from wtforms.fields.simple import BooleanField
from wtforms.fields.simple import ColorField
from wtforms.fields.simple import EmailField
from wtforms.fields.simple import FileField
from wtforms.fields.simple import HiddenField
from wtforms.fields.simple import MultipleFileField
from wtforms.fields.simple import PasswordField
from wtforms.fields.simple import SearchField
from wtforms.fields.simple import StringField
from wtforms.fields.simple import SubmitField
from wtforms.fields.simple import TelField
from wtforms.fields.simple import TextAreaField
from wtforms.fields.simple import URLField
from wtforms.utils import unset_value as _unset_value

__all__ = [
    "Field",
    "Flags",
    "Label",
    "SelectField",
    "SelectMultipleField",
    "SelectFieldBase",
    "RadioField",
    "DateTimeField",
    "DateField",
    "TimeField",
    "MonthField",
    "DateTimeLocalField",
    "WeekField",
    "FormField",
    "IntegerField",
    "DecimalField",
    "FloatField",
    "IntegerRangeField",
    "DecimalRangeField",
    "BooleanField",
    "TextAreaField",
    "PasswordField",
    "FileField",
    "MultipleFileField",
    "HiddenField",
    "SearchField",
    "SubmitField",
    "StringField",
    "TelField",
    "URLField",
    "EmailField",
    "ColorField",
    "FieldList",
    "_unset_value",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_constructors.py ===
from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
    datetime,
    timedelta,
)

from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest

from pandas._libs import (
    iNaT,
    lib,
)
from pandas.compat import HAS_PYARROW
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td

from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    IntegerArray,
    IntervalArray,
    period_array,
)
from pandas.core.internals.blocks import NumpyBlock


class TestSeriesConstructors:
    def test_from_ints_with_non_nano_dt64_dtype(self, index_or_series):
        values = np.arange(10)

        res = index_or_series(values, dtype="M8[s]")
        expected = index_or_series(values.astype("M8[s]"))
        tm.assert_equal(res, expected)

        res = index_or_series(list(values), dtype="M8[s]")
        tm.assert_equal(res, expected)

    def test_from_na_value_and_interval_of_datetime_dtype(self):
        # GH#41805
        ser = Series([None], dtype="interval[datetime64[ns]]")
        assert ser.isna().all()
        assert ser.dtype == "interval[datetime64[ns], right]"

    def test_infer_with_date_and_datetime(self):
        # GH#49341 pre-2.0 we inferred datetime-and-date to datetime64, which
        #  was inconsistent with Index behavior
        ts = Timestamp(2016, 1, 1)
        vals = [ts.to_pydatetime(), ts.date()]

        ser = Series(vals)
        expected = Series(vals, dtype=object)
        tm.assert_series_equal(ser, expected)

        idx = Index(vals)
        expected = Index(vals, dtype=object)
        tm.assert_index_equal(idx, expected)

    def test_unparsable_strings_with_dt64_dtype(self):
        # pre-2.0 these would be silently ignored and come back with object dtype
        vals = ["aa"]
        msg = "^Unknown datetime string format, unable to parse: aa, at position 0$"
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series(np.array(vals, dtype=object), dtype="datetime64[ns]")

    @pytest.mark.parametrize(
        "constructor",
        [
            # NOTE: some overlap with test_constructor_empty but that test does not
            # test for None or an empty generator.
            # test_constructor_pass_none tests None but only with the index also
            # passed.
            (lambda idx: Series(index=idx)),
            (lambda idx: Series(None, index=idx)),
            (lambda idx: Series({}, index=idx)),
            (lambda idx: Series((), index=idx)),
            (lambda idx: Series([], index=idx)),
            (lambda idx: Series((_ for _ in []), index=idx)),
            (lambda idx: Series(data=None, index=idx)),
            (lambda idx: Series(data={}, index=idx)),
            (lambda idx: Series(data=(), index=idx)),
            (lambda idx: Series(data=[], index=idx)),
            (lambda idx: Series(data=(_ for _ in []), index=idx)),
        ],
    )
    @pytest.mark.parametrize("empty_index", [None, []])
    def test_empty_constructor(self, constructor, empty_index):
        # GH 49573 (addition of empty_index parameter)
        expected = Series(index=empty_index)
        result = constructor(empty_index)

        assert result.dtype == object
        assert len(result.index) == 0
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_invalid_dtype(self):
        # GH15520
        msg = "not understood"
        invalid_list = [Timestamp, "Timestamp", list]
        for dtype in invalid_list:
            with pytest.raises(TypeError, match=msg):
                Series([], name="time", dtype=dtype)

    def test_invalid_compound_dtype(self):
        # GH#13296
        c_dtype = np.dtype([("a", "i8"), ("b", "f4")])
        cdt_arr = np.array([(1, 0.4), (256, -13)], dtype=c_dtype)

        with pytest.raises(ValueError, match="Use DataFrame instead"):
            Series(cdt_arr, index=["A", "B"])

    def test_scalar_conversion(self):
        # Pass in scalar is disabled
        scalar = Series(0.5)
        assert not isinstance(scalar, float)

    def test_scalar_extension_dtype(self, ea_scalar_and_dtype):
        # GH 28401

        ea_scalar, ea_dtype = ea_scalar_and_dtype

        ser = Series(ea_scalar, index=range(3))
        expected = Series([ea_scalar] * 3, dtype=ea_dtype)

        assert ser.dtype == ea_dtype
        tm.assert_series_equal(ser, expected)

    def test_constructor(self, datetime_series, using_infer_string):
        empty_series = Series()
        assert datetime_series.index._is_all_dates

        # Pass in Series
        derived = Series(datetime_series)
        assert derived.index._is_all_dates

        tm.assert_index_equal(derived.index, datetime_series.index)
        # Ensure new index is not created
        assert id(datetime_series.index) == id(derived.index)

        # Mixed type Series
        mixed = Series(["hello", np.nan], index=[0, 1])
        assert mixed.dtype == np.object_ if not using_infer_string else "str"
        assert np.isnan(mixed[1])

        assert not empty_series.index._is_all_dates
        assert not Series().index._is_all_dates

        # exception raised is of type ValueError GH35744
        with pytest.raises(
            ValueError,
            match=r"Data must be 1-dimensional, got ndarray of shape \(3, 3\) instead",
        ):
            Series(np.random.default_rng(2).standard_normal((3, 3)), index=np.arange(3))

        mixed.name = "Series"
        rs = Series(mixed).name
        xp = "Series"
        assert rs == xp

        # raise on MultiIndex GH4187
        m = MultiIndex.from_arrays([[1, 2], [3, 4]])
        msg = "initializing a Series from a MultiIndex is not supported"
        with pytest.raises(NotImplementedError, match=msg):
            Series(m)

    def test_constructor_index_ndim_gt_1_raises(self):
        # GH#18579
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=[3, 6, 9])
        with pytest.raises(ValueError, match="Index data must be 1-dimensional"):
            Series([1, 3, 2], index=df)

    @pytest.mark.parametrize("input_class", [list, dict, OrderedDict])
    def test_constructor_empty(self, input_class, using_infer_string):
        empty = Series()
        empty2 = Series(input_class())

        # these are Index() and RangeIndex() which don't compare type equal
        # but are just .equals
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # With explicit dtype:
        empty = Series(dtype="float64")
        empty2 = Series(input_class(), dtype="float64")
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # GH 18515 : with dtype=category:
        empty = Series(dtype="category")
        empty2 = Series(input_class(), dtype="category")
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        if input_class is not list:
            # With index:
            empty = Series(index=range(10))
            empty2 = Series(input_class(), index=range(10))
            tm.assert_series_equal(empty, empty2)

            # With index and dtype float64:
            empty = Series(np.nan, index=range(10))
            empty2 = Series(input_class(), index=range(10), dtype="float64")
            tm.assert_series_equal(empty, empty2)

            # GH 19853 : with empty string, index and dtype str
            empty = Series("", dtype=str, index=range(3))
            if using_infer_string:
                empty2 = Series("", index=range(3), dtype="str")
            else:
                empty2 = Series("", index=range(3))
            tm.assert_series_equal(empty, empty2)

    @pytest.mark.parametrize("input_arg", [np.nan, float("nan")])
    def test_constructor_nan(self, input_arg):
        empty = Series(dtype="float64", index=range(10))
        empty2 = Series(input_arg, index=range(10))

        tm.assert_series_equal(empty, empty2, check_index_type=False)

    @pytest.mark.parametrize(
        "dtype",
        ["f8", "i8", "M8[ns]", "m8[ns]", "category", "object", "datetime64[ns, UTC]"],
    )
    @pytest.mark.parametrize("index", [None, Index([])])
    def test_constructor_dtype_only(self, dtype, index):
        # GH-20865
        result = Series(dtype=dtype, index=index)
        assert result.dtype == dtype
        assert len(result) == 0

    def test_constructor_no_data_index_order(self):
        result = Series(index=["b", "a", "c"])
        assert result.index.tolist() == ["b", "a", "c"]

    def test_constructor_no_data_string_type(self):
        # GH 22477
        result = Series(index=[1], dtype=str)
        assert np.isnan(result.iloc[0])

    @pytest.mark.parametrize("item", ["entry", "ѐ", 13])
    def test_constructor_string_element_string_type(self, item):
        # GH 22477
        result = Series(item, index=[1], dtype=str)
        assert result.iloc[0] == str(item)

    def test_constructor_dtype_str_na_values(self, string_dtype):
        # https://github.com/pandas-dev/pandas/issues/21083
        ser = Series(["x", None], dtype=string_dtype)
        result = ser.isna()
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)
        assert ser.iloc[1] is None

        ser = Series(["x", np.nan], dtype=string_dtype)
        assert np.isnan(ser.iloc[1])

    def test_constructor_series(self):
        index1 = ["d", "b", "a", "c"]
        index2 = sorted(index1)
        s1 = Series([4, 7, -5, 3], index=index1)
        s2 = Series(s1, index=index2)

        tm.assert_series_equal(s2, s1.sort_index())

    def test_constructor_iterable(self):
        # GH 21987
        class Iter:
            def __iter__(self) -> Iterator:
                yield from range(10)

        expected = Series(list(range(10)), dtype="int64")
        result = Series(Iter(), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_sequence(self):
        # GH 21987
        expected = Series(list(range(10)), dtype="int64")
        result = Series(range(10), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_single_str(self):
        # GH 21987
        expected = Series(["abc"])
        result = Series("abc")
        tm.assert_series_equal(result, expected)

    def test_constructor_list_like(self):
        # make sure that we are coercing different
        # list-likes to standard dtypes and not
        # platform specific
        expected = Series([1, 2, 3], dtype="int64")
        for obj in [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3], dtype="int64")]:
            result = Series(obj, index=[0, 1, 2])
            tm.assert_series_equal(result, expected)

    def test_constructor_boolean_index(self):
        # GH#18579
        s1 = Series([1, 2, 3], index=[4, 5, 6])

        index = s1 == 2
        result = Series([1, 3, 2], index=index)
        expected = Series([1, 3, 2], index=[False, True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float64"])
    def test_constructor_index_dtype(self, dtype):
        # GH 17088

        s = Series(Index([0, 2, 4]), dtype=dtype)
        assert s.dtype == dtype

    @pytest.mark.parametrize(
        "input_vals",
        [
            ([1, 2]),
            (["1", "2"]),
            (list(date_range("1/1/2011", periods=2, freq="h"))),
            (list(date_range("1/1/2011", periods=2, freq="h", tz="US/Eastern"))),
            ([Interval(left=0, right=5)]),
        ],
    )
    def test_constructor_list_str(self, input_vals, string_dtype):
        # GH 16605
        # Ensure that data elements from a list are converted to strings
        # when dtype is str, 'str', or 'U'
        result = Series(input_vals, dtype=string_dtype)
        expected = Series(input_vals).astype(string_dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype):
        result = Series([1.0, 2.0, np.nan], dtype=string_dtype)
        expected = Series(["1.0", "2.0", np.nan], dtype=object)
        tm.assert_series_equal(result, expected)
        assert np.isnan(result[2])

    def test_constructor_generator(self):
        gen = (i for i in range(10))

        result = Series(gen)
        exp = Series(range(10))
        tm.assert_series_equal(result, exp)

        # same but with non-default index
        gen = (i for i in range(10))
        result = Series(gen, index=range(10, 20))
        exp.index = range(10, 20)
        tm.assert_series_equal(result, exp)

    def test_constructor_map(self):
        # GH8909
        m = (x for x in range(10))

        result = Series(m)
        exp = Series(range(10))
        tm.assert_series_equal(result, exp)

        # same but with non-default index
        m = (x for x in range(10))
        result = Series(m, index=range(10, 20))
        exp.index = range(10, 20)
        tm.assert_series_equal(result, exp)

    def test_constructor_categorical(self):
        cat = Categorical([0, 1, 2, 0, 1, 2], ["a", "b", "c"])
        res = Series(cat)
        tm.assert_categorical_equal(res.values, cat)

        # can cast to a new dtype
        result = Series(Categorical([1, 2, 3]), dtype="int64")
        expected = Series([1, 2, 3], dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_construct_from_categorical_with_dtype(self):
        # GH12574
        ser = Series(Categorical([1, 2, 3]), dtype="category")
        assert isinstance(ser.dtype, CategoricalDtype)

    def test_construct_intlist_values_category_dtype(self):
        ser = Series([1, 2, 3], dtype="category")
        assert isinstance(ser.dtype, CategoricalDtype)

    def test_constructor_categorical_with_coercion(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"])
        # test basic creation / coercion of categoricals
        s = Series(factor, name="A")
        assert s.dtype == "category"
        assert len(s) == len(factor)

        # in a frame
        df = DataFrame({"A": factor})
        result = df["A"]
        tm.assert_series_equal(result, s)
        result = df.iloc[:, 0]
        tm.assert_series_equal(result, s)
        assert len(df) == len(factor)

        df = DataFrame({"A": s})
        result = df["A"]
        tm.assert_series_equal(result, s)
        assert len(df) == len(factor)

        # multiples
        df = DataFrame({"A": s, "B": s, "C": 1})
        result1 = df["A"]
        result2 = df["B"]
        tm.assert_series_equal(result1, s)
        tm.assert_series_equal(result2, s, check_names=False)
        assert result2.name == "B"
        assert len(df) == len(factor)

    def test_constructor_categorical_with_coercion2(self):
        # GH8623
        x = DataFrame(
            [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
            columns=["person_id", "person_name"],
        )
        x["person_name"] = Categorical(x.person_name)  # doing this breaks transform

        expected = x.iloc[0].person_name
        result = x.person_name.iloc[0]
        assert result == expected

        result = x.person_name[0]
        assert result == expected

        result = x.person_name.loc[0]
        assert result == expected

    def test_constructor_series_to_categorical(self):
        # see GH#16524: test conversion of Series to Categorical
        series = Series(["a", "b", "c"])

        result = Series(series, dtype="category")
        expected = Series(["a", "b", "c"], dtype="category")

        tm.assert_series_equal(result, expected)

    def test_constructor_categorical_dtype(self):
        result = Series(
            ["a", "b"], dtype=CategoricalDtype(["a", "b", "c"], ordered=True)
        )
        assert isinstance(result.dtype, CategoricalDtype)
        tm.assert_index_equal(result.cat.categories, Index(["a", "b", "c"]))
        assert result.cat.ordered

        result = Series(["a", "b"], dtype=CategoricalDtype(["b", "a"]))
        assert isinstance(result.dtype, CategoricalDtype)
        tm.assert_index_equal(result.cat.categories, Index(["b", "a"]))
        assert result.cat.ordered is False

        # GH 19565 - Check broadcasting of scalar with Categorical dtype
        result = Series(
            "a", index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        expected = Series(
            ["a", "a"], index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_categorical_string(self):
        # GH 26336: the string 'category' maintains existing CategoricalDtype
        cdt = CategoricalDtype(categories=list("dabc"), ordered=True)
        expected = Series(list("abcabc"), dtype=cdt)

        # Series(Categorical, dtype='category') keeps existing dtype
        cat = Categorical(list("abcabc"), dtype=cdt)
        result = Series(cat, dtype="category")
        tm.assert_series_equal(result, expected)

        # Series(Series[Categorical], dtype='category') keeps existing dtype
        result = Series(result, dtype="category")
        tm.assert_series_equal(result, expected)

    def test_categorical_sideeffects_free(self):
        # Passing a categorical to a Series and then changing values in either
        # the series or the categorical should not change the values in the
        # other one, IF you specify copy!
        cat = Categorical(["a", "b", "c", "a"])
        s = Series(cat, copy=True)
        assert s.cat is not cat
        s = s.cat.rename_categories([1, 2, 3])
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        exp_cat = np.array(["a", "b", "c", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(s.__array__(), exp_s)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # setting
        s[0] = 2
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # however, copy is False by default
        # so this WILL change values
        cat = Categorical(["a", "b", "c", "a"])
        s = Series(cat, copy=False)
        assert s.values is cat
        s = s.cat.rename_categories([1, 2, 3])
        assert s.values is not cat
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s)

        s[0] = 2
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)

    def test_unordered_compare_equal(self):
        left = Series(["a", "b", "c"], dtype=CategoricalDtype(["a", "b"]))
        right = Series(Categorical(["a", "b", np.nan], categories=["a", "b"]))
        tm.assert_series_equal(left, right)

    def test_constructor_maskedarray(self):
        data = ma.masked_all((3,), dtype=float)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan])
        tm.assert_series_equal(result, expected)

        data[0] = 0.0
        data[2] = 2.0
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([0.0, np.nan, 2.0], index=index)
        tm.assert_series_equal(result, expected)

        data[1] = 1.0
        result = Series(data, index=index)
        expected = Series([0.0, 1.0, 2.0], index=index)
        tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype=int)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan], dtype=float)
        tm.assert_series_equal(result, expected)

        data[0] = 0
        data[2] = 2
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([0, np.nan, 2], index=index, dtype=float)
        tm.assert_series_equal(result, expected)

        data[1] = 1
        result = Series(data, index=index)
        expected = Series([0, 1, 2], index=index, dtype=int)
        with pytest.raises(AssertionError, match="Series classes are different"):
            # TODO should this be raising at all?
            # https://github.com/pandas-dev/pandas/issues/56131
            tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype=bool)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan], dtype=object)
        tm.assert_series_equal(result, expected)

        data[0] = True
        data[2] = False
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([True, np.nan, False], index=index, dtype=object)
        tm.assert_series_equal(result, expected)

        data[1] = True
        result = Series(data, index=index)
        expected = Series([True, True, False], index=index, dtype=bool)
        with pytest.raises(AssertionError, match="Series classes are different"):
            # TODO should this be raising at all?
            # https://github.com/pandas-dev/pandas/issues/56131
            tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype="M8[ns]")
        result = Series(data)
        expected = Series([iNaT, iNaT, iNaT], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)

        data[0] = datetime(2001, 1, 1)
        data[2] = datetime(2001, 1, 3)
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series(
            [datetime(2001, 1, 1), iNaT, datetime(2001, 1, 3)],
            index=index,
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

        data[1] = datetime(2001, 1, 2)
        result = Series(data, index=index)
        expected = Series(
            [datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 3)],
            index=index,
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_maskedarray_hardened(self):
        # Check numpy masked arrays with hard masks -- from GH24574
        data = ma.masked_all((3,), dtype=float).harden_mask()
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan])
        tm.assert_series_equal(result, expected)

    def test_series_ctor_plus_datetimeindex(self, using_copy_on_write):
        rng = date_range("20090415", "20090519", freq="B")
        data = {k: 1 for k in rng}

        result = Series(data, index=rng)
        if using_copy_on_write:
            assert result.index.is_(rng)
        else:
            assert result.index is rng

    def test_constructor_default_index(self):
        s = Series([0, 1, 2])
        tm.assert_index_equal(s.index, Index(range(3)), exact=True)

    @pytest.mark.parametrize(
        "input",
        [
            [1, 2, 3],
            (1, 2, 3),
            list(range(3)),
            Categorical(["a", "b", "a"]),
            (i for i in range(3)),
            (x for x in range(3)),
        ],
    )
    def test_constructor_index_mismatch(self, input):
        # GH 19342
        # test that construction of a Series with an index of different length
        # raises an error
        msg = r"Length of values \(3\) does not match length of index \(4\)"
        with pytest.raises(ValueError, match=msg):
            Series(input, index=np.arange(4))

    def test_constructor_numpy_scalar(self):
        # GH 19342
        # construction with a numpy scalar
        # should not raise
        result = Series(np.array(100), index=np.arange(4), dtype="int64")
        expected = Series(100, index=np.arange(4), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_broadcast_list(self):
        # GH 19342
        # construction with single-element container and index
        # should raise
        msg = r"Length of values \(1\) does not match length of index \(3\)"
        with pytest.raises(ValueError, match=msg):
            Series(["foo"], index=["a", "b", "c"])

    def test_constructor_corner(self):
        df = DataFrame(range(5), index=date_range("2020-01-01", periods=5))
        objs = [df, df]
        s = Series(objs, index=[0, 1])
        assert isinstance(s, Series)

    def test_constructor_sanitize(self):
        s = Series(np.array([1.0, 1.0, 8.0]), dtype="i8")
        assert s.dtype == np.dtype("i8")

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(np.array([1.0, 1.0, np.nan]), copy=True, dtype="i8")

    def test_constructor_copy(self):
        # GH15125
        # test dtype parameter has no side effects on copy=True
        for data in [[1.0], np.array([1.0])]:
            x = Series(data)
            y = Series(x, copy=True, dtype=float)

            # copy=True maintains original data in Series
            tm.assert_series_equal(x, y)

            # changes to origin of copy does not affect the copy
            x[0] = 2.0
            assert not x.equals(y)
            assert x[0] == 2.0
            assert y[0] == 1.0

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite test
    @pytest.mark.parametrize(
        "index",
        [
            date_range("20170101", periods=3, tz="US/Eastern"),
            date_range("20170101", periods=3),
            timedelta_range("1 day", periods=3),
            period_range("2012Q1", periods=3, freq="Q"),
            Index(list("abc")),
            Index([1, 2, 3]),
            RangeIndex(0, 3),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_limit_copies(self, index):
        # GH 17449
        # limit copies of input
        s = Series(index)

        # we make 1 copy; this is just a smoke test here
        assert s._mgr.blocks[0].values is not index

    def test_constructor_shallow_copy(self):
        # constructing a Series from Series with copy=False should still
        # give a "shallow" copy (share data, not attributes)
        # https://github.com/pandas-dev/pandas/issues/49523
        s = Series([1, 2, 3])
        s_orig = s.copy()
        s2 = Series(s)
        assert s2._mgr is not s._mgr
        # Overwriting index of s2 doesn't change s
        s2.index = ["a", "b", "c"]
        tm.assert_series_equal(s, s_orig)

    def test_constructor_pass_none(self):
        s = Series(None, index=range(5))
        assert s.dtype == np.float64

        s = Series(None, index=range(5), dtype=object)
        assert s.dtype == np.object_

        # GH 7431
        # inference on the index
        s = Series(index=np.array([None]))
        expected = Series(index=Index([None]))
        tm.assert_series_equal(s, expected)

    def test_constructor_pass_nan_nat(self):
        # GH 13467
        exp = Series([np.nan, np.nan], dtype=np.float64)
        assert exp.dtype == np.float64
        tm.assert_series_equal(Series([np.nan, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, np.nan])), exp)

        exp = Series([NaT, NaT])
        assert exp.dtype == "datetime64[ns]"
        tm.assert_series_equal(Series([NaT, NaT]), exp)
        tm.assert_series_equal(Series(np.array([NaT, NaT])), exp)

        tm.assert_series_equal(Series([NaT, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([NaT, np.nan])), exp)

        tm.assert_series_equal(Series([np.nan, NaT]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, NaT])), exp)

    def test_constructor_cast(self):
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            Series(["a", "b", "c"], dtype=float)

    def test_constructor_signed_int_overflow_raises(self):
        # GH#41734 disallow silent overflow, enforced in 2.0
        if np_version_gt2:
            msg = "The elements provided in the data cannot all be casted to the dtype"
            err = OverflowError
        else:
            msg = "Values are too large to be losslessly converted"
            err = ValueError
        with pytest.raises(err, match=msg):
            Series([1, 200, 923442], dtype="int8")

        with pytest.raises(err, match=msg):
            Series([1, 200, 923442], dtype="uint8")

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1], dtype=np.uint16),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
            [np.uint16(1)],
            [np.uint32(1)],
            [np.uint64(1)],
        ],
    )
    def test_constructor_numpy_uints(self, values):
        # GH#47294
        value = values[0]
        result = Series(values)

        assert result[0].dtype == value.dtype
        assert result[0] == value

    def test_constructor_unsigned_dtype_overflow(self, any_unsigned_int_numpy_dtype):
        # see gh-15832
        if np_version_gt2:
            msg = (
                f"The elements provided in the data cannot "
                f"all be casted to the dtype {any_unsigned_int_numpy_dtype}"
            )
        else:
            msg = "Trying to coerce negative values to unsigned integers"
        with pytest.raises(OverflowError, match=msg):
            Series([-1], dtype=any_unsigned_int_numpy_dtype)

    def test_constructor_floating_data_int_dtype(self, frame_or_series):
        # GH#40110
        arr = np.random.default_rng(2).standard_normal(2)

        # Long-standing behavior (for Series, new in 2.0 for DataFrame)
        #  has been to ignore the dtype on these;
        #  not clear if this is what we want long-term
        # expected = frame_or_series(arr)

        # GH#49599 as of 2.0 we raise instead of silently retaining float dtype
        msg = "Trying to coerce float values to integer"
        with pytest.raises(ValueError, match=msg):
            frame_or_series(arr, dtype="i8")

        with pytest.raises(ValueError, match=msg):
            frame_or_series(list(arr), dtype="i8")

        # pre-2.0, when we had NaNs, we silently ignored the integer dtype
        arr[0] = np.nan
        # expected = frame_or_series(arr)

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            frame_or_series(arr, dtype="i8")

        exc = IntCastingNaNError
        if frame_or_series is Series:
            # TODO: try to align these
            exc = ValueError
            msg = "cannot convert float NaN to integer"
        with pytest.raises(exc, match=msg):
            # same behavior if we pass list instead of the ndarray
            frame_or_series(list(arr), dtype="i8")

        # float array that can be losslessly cast to integers
        arr = np.array([1.0, 2.0], dtype="float64")
        expected = frame_or_series(arr.astype("i8"))

        obj = frame_or_series(arr, dtype="i8")
        tm.assert_equal(obj, expected)

        obj = frame_or_series(list(arr), dtype="i8")
        tm.assert_equal(obj, expected)

    def test_constructor_coerce_float_fail(self, any_int_numpy_dtype):
        # see gh-15832
        # Updated: make sure we treat this list the same as we would treat
        #  the equivalent ndarray
        # GH#49599 pre-2.0 we silently retained float dtype, in 2.0 we raise
        vals = [1, 2, 3.5]

        msg = "Trying to coerce float values to integer"
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype=any_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            Series(np.array(vals), dtype=any_int_numpy_dtype)

    def test_constructor_coerce_float_valid(self, float_numpy_dtype):
        s = Series([1, 2, 3.5], dtype=float_numpy_dtype)
        expected = Series([1, 2, 3.5]).astype(float_numpy_dtype)
        tm.assert_series_equal(s, expected)

    def test_constructor_invalid_coerce_ints_with_float_nan(self, any_int_numpy_dtype):
        # GH 22585
        # Updated: make sure we treat this list the same as we would treat the
        # equivalent ndarray
        vals = [1, 2, np.nan]
        # pre-2.0 this would return with a float dtype, in 2.0 we raise

        msg = "cannot convert float NaN to integer"
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype=any_int_numpy_dtype)
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(np.array(vals), dtype=any_int_numpy_dtype)

    def test_constructor_dtype_no_cast(self, using_copy_on_write, warn_copy_on_write):
        # see gh-1572
        s = Series([1, 2, 3])
        s2 = Series(s, dtype=np.int64)

        warn = FutureWarning if warn_copy_on_write else None
        with tm.assert_produces_warning(warn):
            s2[1] = 5
        if using_copy_on_write:
            assert s[1] == 2
        else:
            assert s[1] == 5

    def test_constructor_datelike_coercion(self):
        # GH 9477
        # incorrectly inferring on dateimelike looking when object dtype is
        # specified
        s = Series([Timestamp("20130101"), "NOV"], dtype=object)
        assert s.iloc[0] == Timestamp("20130101")
        assert s.iloc[1] == "NOV"
        assert s.dtype == object

    def test_constructor_datelike_coercion2(self):
        # the dtype was being reset on the slicing and re-inferred to datetime
        # even thought the blocks are mixed
        belly = "216 3T19".split()
        wing1 = "2T15 4H19".split()
        wing2 = "416 4T20".split()
        mat = pd.to_datetime("2016-01-22 2019-09-07".split())
        df = DataFrame({"wing1": wing1, "wing2": wing2, "mat": mat}, index=belly)

        result = df.loc["3T19"]
        assert result.dtype == object
        result = df.loc["216"]
        assert result.dtype == object

    def test_constructor_mixed_int_and_timestamp(self, frame_or_series):
        # specifically Timestamp with nanos, not datetimes
        objs = [Timestamp(9), 10, NaT._value]
        result = frame_or_series(objs, dtype="M8[ns]")

        expected = frame_or_series([Timestamp(9), Timestamp(10), NaT])
        tm.assert_equal(result, expected)

    def test_constructor_datetimes_with_nulls(self):
        # gh-15869
        for arr in [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
        ]:
            result = Series(arr)
            assert result.dtype == "M8[ns]"

    def test_constructor_dtype_datetime64(self):
        s = Series(iNaT, dtype="M8[ns]", index=range(5))
        assert isna(s).all()

        # in theory this should be all nulls, but since
        # we are not specifying a dtype is ambiguous
        s = Series(iNaT, index=range(5))
        assert not isna(s).all()

        s = Series(np.nan, dtype="M8[ns]", index=range(5))
        assert isna(s).all()

        s = Series([datetime(2001, 1, 2, 0, 0), iNaT], dtype="M8[ns]")
        assert isna(s[1])
        assert s.dtype == "M8[ns]"

        s = Series([datetime(2001, 1, 2, 0, 0), np.nan], dtype="M8[ns]")
        assert isna(s[1])
        assert s.dtype == "M8[ns]"

    def test_constructor_dtype_datetime64_10(self):
        # GH3416
        pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
        dates = [np.datetime64(x) for x in pydates]

        ser = Series(dates)
        assert ser.dtype == "M8[ns]"

        ser.iloc[0] = np.nan
        assert ser.dtype == "M8[ns]"

        # GH3414 related
        expected = Series(pydates, dtype="datetime64[ms]")

        result = Series(Series(dates).astype(np.int64) / 1000000, dtype="M8[ms]")
        tm.assert_series_equal(result, expected)

        result = Series(dates, dtype="datetime64[ms]")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [NaT, datetime(2013, 1, 2), datetime(2013, 1, 3)], dtype="datetime64[ns]"
        )
        result = Series([np.nan] + dates[1:], dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

    def test_constructor_dtype_datetime64_11(self):
        pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
        dates = [np.datetime64(x) for x in pydates]

        dts = Series(dates, dtype="datetime64[ns]")

        # valid astype
        dts.astype("int64")

        # invalid casting
        msg = r"Converting from datetime64\[ns\] to int32 is not supported"
        with pytest.raises(TypeError, match=msg):
            dts.astype("int32")

        # ints are ok
        # we test with np.int64 to get similar results on
        # windows / 32-bit platforms
        result = Series(dts, dtype=np.int64)
        expected = Series(dts.astype(np.int64))
        tm.assert_series_equal(result, expected)

    def test_constructor_dtype_datetime64_9(self):
        # invalid dates can be help as object
        result = Series([datetime(2, 1, 1)])
        assert result[0] == datetime(2, 1, 1, 0, 0)

        result = Series([datetime(3000, 1, 1)])
        assert result[0] == datetime(3000, 1, 1, 0, 0)

    def test_constructor_dtype_datetime64_8(self):
        # don't mix types
        result = Series([Timestamp("20130101"), 1], index=["a", "b"])
        assert result["a"] == Timestamp("20130101")
        assert result["b"] == 1

    def test_constructor_dtype_datetime64_7(self):
        # GH6529
        # coerce datetime64 non-ns properly
        dates = date_range("01-Jan-2015", "01-Dec-2015", freq="ME")
        values2 = dates.view(np.ndarray).astype("datetime64[ns]")
        expected = Series(values2, index=dates)

        for unit in ["s", "D", "ms", "us", "ns"]:
            dtype = np.dtype(f"M8[{unit}]")
            values1 = dates.view(np.ndarray).astype(dtype)
            result = Series(values1, dates)
            if unit == "D":
                # for unit="D" we cast to nearest-supported reso, i.e. "s"
                dtype = np.dtype("M8[s]")
            assert result.dtype == dtype
            tm.assert_series_equal(result, expected.astype(dtype))

        # GH 13876
        # coerce to non-ns to object properly
        expected = Series(values2, index=dates, dtype=object)
        for dtype in ["s", "D", "ms", "us", "ns"]:
            values1 = dates.view(np.ndarray).astype(f"M8[{dtype}]")
            result = Series(values1, index=dates, dtype=object)
            tm.assert_series_equal(result, expected)

        # leave datetime.date alone
        dates2 = np.array([d.date() for d in dates.to_pydatetime()], dtype=object)
        series1 = Series(dates2, dates)
        tm.assert_numpy_array_equal(series1.values, dates2)
        assert series1.dtype == object

    def test_constructor_dtype_datetime64_6(self):
        # as of 2.0, these no longer infer datetime64 based on the strings,
        #  matching the Index behavior

        ser = Series([None, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([np.nan, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([NaT, None, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([NaT, np.nan, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

    def test_constructor_dtype_datetime64_5(self):
        # tz-aware (UTC and other tz's)
        # GH 8411
        dr = date_range("20130101", periods=3)
        assert Series(dr).iloc[0].tz is None
        dr = date_range("20130101", periods=3, tz="UTC")
        assert str(Series(dr).iloc[0].tz) == "UTC"
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        assert str(Series(dr).iloc[0].tz) == "US/Eastern"

    def test_constructor_dtype_datetime64_4(self):
        # non-convertible
        ser = Series([1479596223000, -1479590, NaT])
        assert ser.dtype == "object"
        assert ser[2] is NaT
        assert "NaT" in str(ser)

    def test_constructor_dtype_datetime64_3(self):
        # if we passed a NaT it remains
        ser = Series([datetime(2010, 1, 1), datetime(2, 1, 1), NaT])
        assert ser.dtype == "object"
        assert ser[2] is NaT
        assert "NaT" in str(ser)

    def test_constructor_dtype_datetime64_2(self):
        # if we passed a nan it remains
        ser = Series([datetime(2010, 1, 1), datetime(2, 1, 1), np.nan])
        assert ser.dtype == "object"
        assert ser[2] is np.nan
        assert "NaN" in str(ser)

    def test_constructor_with_datetime_tz(self):
        # 8260
        # support datetime64 with tz

        dr = date_range("20130101", periods=3, tz="US/Eastern")
        s = Series(dr)
        assert s.dtype.name == "datetime64[ns, US/Eastern]"
        assert s.dtype == "datetime64[ns, US/Eastern]"
        assert isinstance(s.dtype, DatetimeTZDtype)
        assert "datetime64[ns, US/Eastern]" in str(s)

        # export
        result = s.values
        assert isinstance(result, np.ndarray)
        assert result.dtype == "datetime64[ns]"

        exp = DatetimeIndex(result)
        exp = exp.tz_localize("UTC").tz_convert(tz=s.dt.tz)
        tm.assert_index_equal(dr, exp)

        # indexing
        result = s.iloc[0]
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")
        result = s[0]
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")

        result = s[Series([True, True, False], index=s.index)]
        tm.assert_series_equal(result, s[0:2])

        result = s.iloc[0:1]
        tm.assert_series_equal(result, Series(dr[0:1]))

        # concat
        result = pd.concat([s.iloc[0:1], s.iloc[1:]])
        tm.assert_series_equal(result, s)

        # short str
        assert "datetime64[ns, US/Eastern]" in str(s)

        # formatting with NaT
        result = s.shift()
        assert "datetime64[ns, US/Eastern]" in str(result)
        assert "NaT" in str(result)

        result = DatetimeIndex(s, freq="infer")
        tm.assert_index_equal(result, dr)

    def test_constructor_with_datetime_tz5(self):
        # long str
        ser = Series(date_range("20130101", periods=1000, tz="US/Eastern"))
        assert "datetime64[ns, US/Eastern]" in str(ser)

    def test_constructor_with_datetime_tz4(self):
        # inference
        ser = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Pacific"),
            ]
        )
        assert ser.dtype == "datetime64[ns, US/Pacific]"
        assert lib.infer_dtype(ser, skipna=True) == "datetime64"

    def test_constructor_with_datetime_tz3(self):
        ser = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Eastern"),
            ]
        )
        assert ser.dtype == "object"
        assert lib.infer_dtype(ser, skipna=True) == "datetime"

    def test_constructor_with_datetime_tz2(self):
        # with all NaT
        ser = Series(NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        dti = DatetimeIndex(["NaT", "NaT"], tz="US/Eastern").as_unit("ns")
        expected = Series(dti)
        tm.assert_series_equal(ser, expected)

    def test_constructor_no_partial_datetime_casting(self):
        # GH#40111
        vals = [
            "nan",
            Timestamp("1990-01-01"),
            "2015-03-14T16:15:14.123-08:00",
            "2019-03-04T21:56:32.620-07:00",
            None,
        ]
        ser = Series(vals)
        assert all(ser[i] is vals[i] for i in range(len(vals)))

    @pytest.mark.parametrize("arr_dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("kind", ["M", "m"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_construction_to_datetimelike_unit(self, arr_dtype, kind, unit):
        # tests all units
        # gh-19223
        # TODO: GH#19223 was about .astype, doesn't belong here
        dtype = f"{kind}8[{unit}]"
        arr = np.array([1, 2, 3], dtype=arr_dtype)
        ser = Series(arr)
        result = ser.astype(dtype)

        expected = Series(arr.astype(dtype))

        if unit in ["ns", "us", "ms", "s"]:
            assert result.dtype == dtype
            assert expected.dtype == dtype
        else:
            # Otherwise we cast to nearest-supported unit, i.e. seconds
            assert result.dtype == f"{kind}8[s]"
            assert expected.dtype == f"{kind}8[s]"

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("arg", ["2013-01-01 00:00:00", NaT, np.nan, None])
    def test_constructor_with_naive_string_and_datetimetz_dtype(self, arg):
        # GH 17415: With naive string
        result = Series([arg], dtype="datetime64[ns, CET]")
        expected = Series(Timestamp(arg)).dt.tz_localize("CET")
        tm.assert_series_equal(result, expected)

    def test_constructor_datetime64_bigendian(self):
        # GH#30976
        ms = np.datetime64(1, "ms")
        arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")

        result = Series(arr)
        expected = Series([Timestamp(ms)]).astype("M8[ms]")
        assert expected.dtype == "M8[ms]"
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("interval_constructor", [IntervalIndex, IntervalArray])
    def test_construction_interval(self, interval_constructor):
        # construction from interval & array of intervals
        intervals = interval_constructor.from_breaks(np.arange(3), closed="right")
        result = Series(intervals)
        assert result.dtype == "interval[int64, right]"
        tm.assert_index_equal(Index(result.values), Index(intervals))

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_infer_interval(self, data_constructor):
        # GH 23563: consistent closed results in interval dtype
        data = [Interval(0, 1), Interval(0, 2), None]
        result = Series(data_constructor(data))
        expected = Series(IntervalArray(data))
        assert result.dtype == "interval[float64, right]"
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_interval_mixed_closed(self, data_constructor):
        # GH 23563: mixed closed results in object dtype (not interval dtype)
        data = [Interval(0, 1, closed="both"), Interval(0, 2, closed="neither")]
        result = Series(data_constructor(data))
        assert result.dtype == object
        assert result.tolist() == data

    def test_construction_consistency(self):
        # make sure that we are not re-localizing upon construction
        # GH 14928
        ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))

        result = Series(ser, dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        result = Series(ser.dt.tz_convert("UTC"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        # Pre-2.0 dt64 values were treated as utc, which was inconsistent
        #  with DatetimeIndex, which treats them as wall times, see GH#33401
        result = Series(ser.values, dtype=ser.dtype)
        expected = Series(ser.values).dt.tz_localize(ser.dtype.tz)
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(None):
            # one suggested alternative to the deprecated (changed in 2.0) usage
            middle = Series(ser.values).dt.tz_localize("UTC")
            result = middle.dt.tz_convert(ser.dtype.tz)
        tm.assert_series_equal(result, ser)

        with tm.assert_produces_warning(None):
            # the other suggested alternative to the deprecated usage
            result = Series(ser.values.view("int64"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_infer_period(self, data_constructor):
        data = [Period("2000", "D"), Period("2001", "D"), None]
        result = Series(data_constructor(data))
        expected = Series(period_array(data))
        tm.assert_series_equal(result, expected)
        assert result.dtype == "Period[D]"

    @pytest.mark.xfail(reason="PeriodDtype Series not supported yet")
    def test_construct_from_ints_including_iNaT_scalar_period_dtype(self):
        series = Series([0, 1000, 2000, pd._libs.iNaT], dtype="period[D]")

        val = series[3]
        assert isna(val)

        series[2] = val
        assert isna(series[2])

    def test_constructor_period_incompatible_frequency(self):
        data = [Period("2000", "D"), Period("2001", "Y")]
        result = Series(data)
        assert result.dtype == object
        assert result.tolist() == data

    def test_constructor_periodindex(self):
        # GH7932
        # converting a PeriodIndex when put in a Series

        pi = period_range("20130101", periods=5, freq="D")
        s = Series(pi)
        assert s.dtype == "Period[D]"
        with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
            expected = Series(pi.astype(object))
        tm.assert_series_equal(s, expected)

    def test_constructor_dict(self):
        d = {"a": 0.0, "b": 1.0, "c": 2.0}

        result = Series(d)
        expected = Series(d, index=sorted(d.keys()))
        tm.assert_series_equal(result, expected)

        result = Series(d, index=["b", "c", "d", "a"])
        expected = Series([1, 2, np.nan, 0], index=["b", "c", "d", "a"])
        tm.assert_series_equal(result, expected)

        pidx = period_range("2020-01-01", periods=10, freq="D")
        d = {pidx[0]: 0, pidx[1]: 1}
        result = Series(d, index=pidx)
        expected = Series(np.nan, pidx, dtype=np.float64)
        expected.iloc[0] = 0
        expected.iloc[1] = 1
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_list_value_explicit_dtype(self):
        # GH 18625
        d = {"a": [[2], [3], [4]]}
        result = Series(d, index=["a"], dtype="object")
        expected = Series(d, index=["a"])
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_order(self):
        # GH19018
        # initialization ordering: by insertion order
        d = {"b": 1, "a": 0, "c": 2}
        result = Series(d)
        expected = Series([1, 0, 2], index=list("bac"))
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_extension(self, ea_scalar_and_dtype, request):
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        if isinstance(ea_scalar, Timestamp):
            mark = pytest.mark.xfail(
                reason="Construction from dict goes through "
                "maybe_convert_objects which casts to nano"
            )
            request.applymarker(mark)
        d = {"a": ea_scalar}
        result = Series(d, index=["a"])
        expected = Series(ea_scalar, index=["a"], dtype=ea_dtype)

        assert result.dtype == ea_dtype

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("value", [2, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # GH 18480
        d = {1: "a", value: "b", float("nan"): "c", 4: "d"}
        result = Series(d).sort_values()
        expected = Series(["a", "b", "c", "d"], index=[1, value, np.nan, 4])
        tm.assert_series_equal(result, expected)

        # MultiIndex:
        d = {(1, 1): "a", (2, np.nan): "b", (3, value): "c"}
        result = Series(d).sort_values()
        expected = Series(
            ["a", "b", "c"], index=Index([(1, 1), (2, np.nan), (3, value)])
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_datetime64_index(self):
        # GH 9456

        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]
        values = [42544017.198965244, 1234565, 40512335.181958228, -1]

        def create_data(constructor):
            return dict(zip((constructor(x) for x in dates_as_str), values))

        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp = create_data(Timestamp)

        expected = Series(values, (Timestamp(x) for x in dates_as_str))

        result_datetime64 = Series(data_datetime64)
        result_datetime = Series(data_datetime)
        result_Timestamp = Series(data_Timestamp)

        tm.assert_series_equal(result_datetime64, expected)
        tm.assert_series_equal(result_datetime, expected)
        tm.assert_series_equal(result_Timestamp, expected)

    def test_constructor_dict_tuple_indexer(self):
        # GH 12948
        data = {(1, 1, None): -1.0}
        result = Series(data)
        expected = Series(
            -1.0, index=MultiIndex(levels=[[1], [1], [np.nan]], codes=[[0], [0], [-1]])
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_mapping(self, non_dict_mapping_subclass):
        # GH 29788
        ndm = non_dict_mapping_subclass({3: "three"})
        result = Series(ndm)
        expected = Series(["three"], index=[3])

        tm.assert_series_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        data = [(1, 1), (2, 2), (2, 3)]
        s = Series(data)
        assert list(s) == data

    def test_constructor_tuple_of_tuples(self):
        data = ((1, 1), (2, 2), (2, 3))
        s = Series(data)
        assert tuple(s) == data

    def test_constructor_dict_of_tuples(self):
        data = {(1, 2): 3, (None, 5): 6}
        result = Series(data).sort_values()
        expected = Series([3, 6], index=MultiIndex.from_tuples([(1, 2), (None, 5)]))
        tm.assert_series_equal(result, expected)

    # https://github.com/pandas-dev/pandas/issues/22698
    @pytest.mark.filterwarnings("ignore:elementwise comparison:FutureWarning")
    def test_fromDict(self, using_infer_string):
        data = {"a": 0, "b": 1, "c": 2, "d": 3}

        series = Series(data)
        tm.assert_is_sorted(series.index)

        data = {"a": 0, "b": "1", "c": "2", "d": datetime.now()}
        series = Series(data)
        assert series.dtype == np.object_

        data = {"a": 0, "b": "1", "c": "2", "d": "3"}
        series = Series(data)
        assert series.dtype == np.object_ if not using_infer_string else "str"

        data = {"a": "0", "b": "1"}
        series = Series(data, dtype=float)
        assert series.dtype == np.float64

    def test_fromValue(self, datetime_series, using_infer_string):
        nans = Series(np.nan, index=datetime_series.index, dtype=np.float64)
        assert nans.dtype == np.float64
        assert len(nans) == len(datetime_series)

        strings = Series("foo", index=datetime_series.index)
        assert strings.dtype == np.object_ if not using_infer_string else "str"
        assert len(strings) == len(datetime_series)

        d = datetime.now()
        dates = Series(d, index=datetime_series.index)
        assert dates.dtype == "M8[us]"
        assert len(dates) == len(datetime_series)

        # GH12336
        # Test construction of categorical series from value
        categorical = Series(0, index=datetime_series.index, dtype="category")
        expected = Series(0, index=datetime_series.index).astype("category")
        assert categorical.dtype == "category"
        assert len(categorical) == len(datetime_series)
        tm.assert_series_equal(categorical, expected)

    def test_constructor_dtype_timedelta64(self):
        # basic
        td = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == "timedelta64[ns]"

        td = Series([timedelta(days=1)])
        assert td.dtype == "timedelta64[ns]"

        td = Series([timedelta(days=1), timedelta(days=2), np.timedelta64(1, "s")])

        assert td.dtype == "timedelta64[ns]"

        # mixed with NaT
        td = Series([timedelta(days=1), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        td = Series([timedelta(days=1), np.nan], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        td = Series([np.timedelta64(300000000), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        # improved inference
        # GH5689
        td = Series([np.timedelta64(300000000), NaT])
        assert td.dtype == "timedelta64[ns]"

        # because iNaT is int, not coerced to timedelta
        td = Series([np.timedelta64(300000000), iNaT])
        assert td.dtype == "object"

        td = Series([np.timedelta64(300000000), np.nan])
        assert td.dtype == "timedelta64[ns]"

        td = Series([NaT, np.timedelta64(300000000)])
        assert td.dtype == "timedelta64[ns]"

        td = Series([np.timedelta64(1, "s")])
        assert td.dtype == "timedelta64[ns]"

        # valid astype
        td.astype("int64")

        # invalid casting
        msg = r"Converting from timedelta64\[ns\] to int32 is not supported"
        with pytest.raises(TypeError, match=msg):
            td.astype("int32")

        # this is an invalid casting
        msg = "|".join(
            [
                "Could not convert object to NumPy timedelta",
                "Could not convert 'foo' to NumPy timedelta",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            Series([timedelta(days=1), "foo"], dtype="m8[ns]")

        # leave as object here
        td = Series([timedelta(days=i) for i in range(3)] + ["foo"])
        assert td.dtype == "object"

        # as of 2.0, these no longer infer timedelta64 based on the strings,
        #  matching Index behavior
        ser = Series([None, NaT, "1 Day"])
        assert ser.dtype == object

        ser = Series([np.nan, NaT, "1 Day"])
        assert ser.dtype == object

        ser = Series([NaT, None, "1 Day"])
        assert ser.dtype == object

        ser = Series([NaT, np.nan, "1 Day"])
        assert ser.dtype == object

    # GH 16406
    def test_constructor_mixed_tz(self):
        s = Series([Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")])
        expected = Series(
            [Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")],
            dtype="object",
        )
        tm.assert_series_equal(s, expected)

    def test_NaT_scalar(self):
        series = Series([0, 1000, 2000, iNaT], dtype="M8[ns]")

        val = series[3]
        assert isna(val)

        series[2] = val
        assert isna(series[2])

    def test_NaT_cast(self):
        # GH10747
        result = Series([np.nan]).astype("M8[ns]")
        expected = Series([NaT], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)

    def test_constructor_name_hashable(self):
        for n in [777, 777.0, "name", datetime(2001, 11, 11), (1,), "\u05D0"]:
            for data in [[1, 2, 3], np.ones(3), {"a": 0, "b": 1}]:
                s = Series(data, name=n)
                assert s.name == n

    def test_constructor_name_unhashable(self):
        msg = r"Series\.name must be a hashable type"
        for n in [["name_list"], np.ones(2), {1: 2}]:
            for data in [["name_list"], np.ones(2), {1: 2}]:
                with pytest.raises(TypeError, match=msg):
                    Series(data, name=n)

    def test_auto_conversion(self):
        series = Series(list(date_range("1/1/2000", periods=10)))
        assert series.dtype == "M8[ns]"

    def test_convert_non_ns(self):
        # convert from a numpy array of non-ns timedelta64
        arr = np.array([1, 2, 3], dtype="timedelta64[s]")
        ser = Series(arr)
        assert ser.dtype == arr.dtype

        tdi = timedelta_range("00:00:01", periods=3, freq="s").as_unit("s")
        expected = Series(tdi)
        assert expected.dtype == arr.dtype
        tm.assert_series_equal(ser, expected)

        # convert from a numpy array of non-ns datetime64
        arr = np.array(
            ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
        )
        ser = Series(arr)
        expected = Series(date_range("20130101", periods=3, freq="D"), dtype="M8[s]")
        assert expected.dtype == "M8[s]"
        tm.assert_series_equal(ser, expected)

        arr = np.array(
            ["2013-01-01 00:00:01", "2013-01-01 00:00:02", "2013-01-01 00:00:03"],
            dtype="datetime64[s]",
        )
        ser = Series(arr)
        expected = Series(
            date_range("20130101 00:00:01", periods=3, freq="s"), dtype="M8[s]"
        )
        assert expected.dtype == "M8[s]"
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),
            timedelta_range("1 day", periods=10),
            period_range("2000-Q1", periods=10, freq="Q"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_cant_cast_datetimelike(self, index):
        # floats are not ok
        # strip Index to convert PeriodIndex -> Period
        # We don't care whether the error message says
        # PeriodIndex or PeriodArray
        msg = f"Cannot cast {type(index).__name__.rstrip('Index')}.*? to "

        with pytest.raises(TypeError, match=msg):
            Series(index, dtype=float)

        # ints are ok
        # we test with np.int64 to get similar results on
        # windows / 32-bit platforms
        result = Series(index, dtype=np.int64)
        expected = Series(index.astype(np.int64))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),
            timedelta_range("1 day", periods=10),
            period_range("2000-Q1", periods=10, freq="Q"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_cast_object(self, index):
        s = Series(index, dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

        s = Series(Index(index, dtype=object), dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

        s = Series(index.astype(object), dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    def test_constructor_generic_timestamp_no_frequency(self, dtype, request):
        # see gh-15524, gh-15987
        msg = "dtype has no unit. Please pass in"

        if np.dtype(dtype).name not in ["timedelta64", "datetime64"]:
            mark = pytest.mark.xfail(reason="GH#33890 Is assigned ns unit")
            request.applymarker(mark)

        with pytest.raises(ValueError, match=msg):
            Series([], dtype=dtype)

    @pytest.mark.parametrize("unit", ["ps", "as", "fs", "Y", "M", "W", "D", "h", "m"])
    @pytest.mark.parametrize("kind", ["m", "M"])
    def test_constructor_generic_timestamp_bad_frequency(self, kind, unit):
        # see gh-15524, gh-15987
        # as of 2.0 we raise on any non-supported unit rather than silently
        #  cast to nanos; previously we only raised for frequencies higher
        #  than ns
        dtype = f"{kind}8[{unit}]"

        msg = "dtype=.* is not supported. Supported resolutions are"
        with pytest.raises(TypeError, match=msg):
            Series([], dtype=dtype)

        with pytest.raises(TypeError, match=msg):
            # pre-2.0 the DataFrame cast raised but the Series case did not
            DataFrame([[0]], dtype=dtype)

    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    def test_constructor_range_dtype(self, dtype):
        # GH 16804
        expected = Series([0, 1, 2, 3, 4], dtype=dtype or "int64")
        result = Series(range(5), dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_range_overflows(self):
        # GH#30173 range objects that overflow int64
        rng = range(2**63, 2**63 + 4)
        ser = Series(rng)
        expected = Series(list(rng))
        tm.assert_series_equal(ser, expected)
        assert list(ser) == list(rng)
        assert ser.dtype == np.uint64

        rng2 = range(2**63 + 4, 2**63, -1)
        ser2 = Series(rng2)
        expected2 = Series(list(rng2))
        tm.assert_series_equal(ser2, expected2)
        assert list(ser2) == list(rng2)
        assert ser2.dtype == np.uint64

        rng3 = range(-(2**63), -(2**63) - 4, -1)
        ser3 = Series(rng3)
        expected3 = Series(list(rng3))
        tm.assert_series_equal(ser3, expected3)
        assert list(ser3) == list(rng3)
        assert ser3.dtype == object

        rng4 = range(2**73, 2**73 + 4)
        ser4 = Series(rng4)
        expected4 = Series(list(rng4))
        tm.assert_series_equal(ser4, expected4)
        assert list(ser4) == list(rng4)
        assert ser4.dtype == object

    def test_constructor_tz_mixed_data(self):
        # GH 13051
        dt_list = [
            Timestamp("2016-05-01 02:03:37"),
            Timestamp("2016-04-30 19:03:37-0700", tz="US/Pacific"),
        ]
        result = Series(dt_list)
        expected = Series(dt_list, dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        # GH#25843, GH#41555, GH#33401
        tz = tz_aware_fixture
        ts = Timestamp("2019", tz=tz)
        if pydt:
            ts = ts.to_pydatetime()

        msg = (
            "Cannot convert timezone-aware data to timezone-naive dtype. "
            r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
        )
        with pytest.raises(ValueError, match=msg):
            Series([ts], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series(np.array([ts], dtype=object), dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series({0: ts}, dtype="datetime64[ns]")

        msg = "Cannot unbox tzaware Timestamp to tznaive dtype"
        with pytest.raises(TypeError, match=msg):
            Series(ts, index=[0], dtype="datetime64[ns]")

    def test_constructor_datetime64(self):
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates = np.asarray(rng)

        series = Series(dates)
        assert np.issubdtype(series.dtype, np.dtype("M8[ns]"))

    def test_constructor_datetimelike_scalar_to_string_dtype(
        self, nullable_string_dtype
    ):
        # https://github.com/pandas-dev/pandas/pull/33846
        result = Series("M", index=[1, 2, 3], dtype=nullable_string_dtype)
        expected = Series(["M", "M", "M"], index=[1, 2, 3], dtype=nullable_string_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            [np.datetime64("2012-01-01"), np.datetime64("2013-01-01")],
            ["2012-01-01", "2013-01-01"],
        ],
    )
    def test_constructor_sparse_datetime64(self, values):
        # https://github.com/pandas-dev/pandas/issues/35762
        dtype = pd.SparseDtype("datetime64[ns]")
        result = Series(values, dtype=dtype)
        arr = pd.arrays.SparseArray(values, dtype=dtype)
        expected = Series(arr)
        tm.assert_series_equal(result, expected)

    def test_construction_from_ordered_collection(self):
        # https://github.com/pandas-dev/pandas/issues/36044
        result = Series({"a": 1, "b": 2}.keys())
        expected = Series(["a", "b"])
        tm.assert_series_equal(result, expected)

        result = Series({"a": 1, "b": 2}.values())
        expected = Series([1, 2])
        tm.assert_series_equal(result, expected)

    def test_construction_from_large_int_scalar_no_overflow(self):
        # https://github.com/pandas-dev/pandas/issues/36291
        n = 1_000_000_000_000_000_000_000
        result = Series(n, index=[0])
        expected = Series(n)
        tm.assert_series_equal(result, expected)

    def test_constructor_list_of_periods_infers_period_dtype(self):
        series = Series(list(period_range("2000-01-01", periods=10, freq="D")))
        assert series.dtype == "Period[D]"

        series = Series(
            [Period("2011-01-01", freq="D"), Period("2011-02-01", freq="D")]
        )
        assert series.dtype == "Period[D]"

    def test_constructor_subclass_dict(self, dict_subclass):
        data = dict_subclass((x, 10.0 * x) for x in range(10))
        series = Series(data)
        expected = Series(dict(data.items()))
        tm.assert_series_equal(series, expected)

    def test_constructor_ordereddict(self):
        # GH3283
        data = OrderedDict(
            (f"col{i}", np.random.default_rng(2).random()) for i in range(12)
        )

        series = Series(data)
        expected = Series(list(data.values()), list(data.keys()))
        tm.assert_series_equal(series, expected)

        # Test with subclass
        class A(OrderedDict):
            pass

        series = Series(A(data))
        tm.assert_series_equal(series, expected)

    def test_constructor_dict_multiindex(self):
        d = {("a", "a"): 0.0, ("b", "a"): 1.0, ("b", "c"): 2.0}
        _d = sorted(d.items())
        result = Series(d)
        expected = Series(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        )
        tm.assert_series_equal(result, expected)

        d["z"] = 111.0
        _d.insert(0, ("z", d["z"]))
        result = Series(d)
        expected = Series(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        )
        result = result.reindex(index=expected.index)
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_multiindex_reindex_flat(self):
        # construction involves reindexing with a MultiIndex corner case
        data = {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2, "j": np.nan}
        expected = Series(data)

        result = Series(expected[:-1].to_dict(), index=expected.index)
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_timedelta_index(self):
        # GH #12169 : Resample category data with timedelta index
        # construct Series from dict as data and TimedeltaIndex as index
        # will result NaN in result Series data
        expected = Series(
            data=["A", "B", "C"], index=pd.to_timedelta([0, 10, 20], unit="s")
        )

        result = Series(
            data={
                pd.to_timedelta(0, unit="s"): "A",
                pd.to_timedelta(10, unit="s"): "B",
                pd.to_timedelta(20, unit="s"): "C",
            },
            index=pd.to_timedelta([0, 10, 20], unit="s"),
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_infer_index_tz(self):
        values = [188.5, 328.25]
        tzinfo = tzoffset(None, 7200)
        index = [
            datetime(2012, 5, 11, 11, tzinfo=tzinfo),
            datetime(2012, 5, 11, 12, tzinfo=tzinfo),
        ]
        series = Series(data=values, index=index)

        assert series.index.tz == tzinfo

        # it works! GH#2443
        repr(series.index[0])

    def test_constructor_with_pandas_dtype(self):
        # going through 2D->1D path
        vals = [(1,), (2,), (3,)]
        ser = Series(vals)
        dtype = ser.array.dtype  # NumpyEADtype
        ser2 = Series(vals, dtype=dtype)
        tm.assert_series_equal(ser, ser2)

    def test_constructor_int_dtype_missing_values(self):
        # GH#43017
        result = Series(index=[0], dtype="int64")
        expected = Series(np.nan, index=[0], dtype="float64")
        tm.assert_series_equal(result, expected)

    def test_constructor_bool_dtype_missing_values(self):
        # GH#43018
        result = Series(index=[0], dtype="bool")
        expected = Series(True, index=[0], dtype="bool")
        tm.assert_series_equal(result, expected)

    def test_constructor_int64_dtype(self, any_int_dtype):
        # GH#44923
        result = Series(["0", "1", "2"], dtype=any_int_dtype)
        expected = Series([0, 1, 2], dtype=any_int_dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_raise_on_lossy_conversion_of_strings(self):
        # GH#44923
        if not np_version_gt2:
            raises = pytest.raises(
                ValueError, match="string values cannot be losslessly cast to int8"
            )
        else:
            raises = pytest.raises(
                OverflowError, match="The elements provided in the data"
            )
        with raises:
            Series(["128"], dtype="int8")

    def test_constructor_dtype_timedelta_alternative_construct(self):
        # GH#35465
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[ns]")
        expected = Series(pd.to_timedelta([1000000, 200000, 3000000], unit="ns"))
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(
        reason="Not clear what the correct expected behavior should be with "
        "integers now that we support non-nano. ATM (2022-10-08) we treat ints "
        "as nanoseconds, then cast to the requested dtype. xref #48312"
    )
    def test_constructor_dtype_timedelta_ns_s(self):
        # GH#35465
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[ns]")
        expected = Series([1000000, 200000, 3000000], dtype="timedelta64[s]")
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(
        reason="Not clear what the correct expected behavior should be with "
        "integers now that we support non-nano. ATM (2022-10-08) we treat ints "
        "as nanoseconds, then cast to the requested dtype. xref #48312"
    )
    def test_constructor_dtype_timedelta_ns_s_astype_int64(self):
        # GH#35465
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[ns]").astype(
            "int64"
        )
        expected = Series([1000000, 200000, 3000000], dtype="timedelta64[s]").astype(
            "int64"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning"
    )
    @pytest.mark.parametrize("func", [Series, DataFrame, Index, pd.array])
    def test_constructor_mismatched_null_nullable_dtype(
        self, func, any_numeric_ea_dtype
    ):
        # GH#44514
        msg = "|".join(
            [
                "cannot safely cast non-equivalent object",
                r"int\(\) argument must be a string, a bytes-like object "
                "or a (real )?number",
                r"Cannot cast array data from dtype\('O'\) to dtype\('float64'\) "
                "according to the rule 'safe'",
                "object cannot be converted to a FloatingDtype",
                "'values' contains non-numeric NA",
            ]
        )

        for null in tm.NP_NAT_OBJECTS + [NaT]:
            with pytest.raises(TypeError, match=msg):
                func([null, 1.0, 3.0], dtype=any_numeric_ea_dtype)

    def test_series_constructor_ea_int_from_bool(self):
        # GH#42137
        result = Series([True, False, True, pd.NA], dtype="Int64")
        expected = Series([1, 0, 1, pd.NA], dtype="Int64")
        tm.assert_series_equal(result, expected)

        result = Series([True, False, True], dtype="Int64")
        expected = Series([1, 0, 1], dtype="Int64")
        tm.assert_series_equal(result, expected)

    def test_series_constructor_ea_int_from_string_bool(self):
        # GH#42137
        with pytest.raises(ValueError, match="invalid literal"):
            Series(["True", "False", "True", pd.NA], dtype="Int64")

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_series_constructor_overflow_uint_ea(self, val):
        # GH#38798
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, val], dtype="UInt64")
        expected = Series(np.array([max_val, 1], dtype="uint64"), dtype="UInt64")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_series_constructor_overflow_uint_ea_with_na(self, val):
        # GH#38798
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, val, pd.NA], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([max_val, 1, 0], dtype="uint64"),
                np.array([0, 0, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_constructor_overflow_uint_with_nan(self):
        # GH#38798
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, np.nan], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([max_val, 1], dtype="uint64"),
                np.array([0, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_constructor_ea_all_na(self):
        # GH#38798
        result = Series([np.nan, np.nan], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([1, 1], dtype="uint64"),
                np.array([1, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_from_index_dtype_equal_does_not_copy(self):
        # GH#52008
        idx = Index([1, 2, 3])
        expected = idx.copy(deep=True)
        ser = Series(idx, dtype="int64")
        ser.iloc[0] = 100
        tm.assert_index_equal(idx, expected)

    def test_series_string_inference(self):
        # GH#54430
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", "b"])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series(["a", "b"], dtype=dtype)
        tm.assert_series_equal(ser, expected)

        expected = Series(["a", 1], dtype="object")
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", 1])
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("na_value", [None, np.nan, pd.NA])
    def test_series_string_with_na_inference(self, na_value):
        # GH#54430
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", na_value])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series(["a", None], dtype=dtype)
        tm.assert_series_equal(ser, expected)

    def test_series_string_inference_scalar(self):
        # GH#54430
        with pd.option_context("future.infer_string", True):
            ser = Series("a", index=[1])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series("a", index=[1], dtype=dtype)
        tm.assert_series_equal(ser, expected)

    def test_series_string_inference_array_string_dtype(self):
        # GH#54496
        with pd.option_context("future.infer_string", True):
            ser = Series(np.array(["a", "b"]))
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series(["a", "b"], dtype=dtype)
        tm.assert_series_equal(ser, expected)

    def test_series_string_inference_storage_definition(self):
        # https://github.com/pandas-dev/pandas/issues/54793
        # but after PDEP-14 (string dtype), it was decided to keep dtype="string"
        # returning the NA string dtype, so expected is changed from
        # "string[pyarrow_numpy]" to "string[python]"
        expected = Series(["a", "b"], dtype="string[python]")
        with pd.option_context("future.infer_string", True):
            result = Series(["a", "b"], dtype="string")
        tm.assert_series_equal(result, expected)

        expected = Series(["a", "b"], dtype=pd.StringDtype(na_value=np.nan))
        with pd.option_context("future.infer_string", True):
            result = Series(["a", "b"], dtype="str")
        tm.assert_series_equal(result, expected)

    def test_series_constructor_infer_string_scalar(self):
        # GH#55537
        with pd.option_context("future.infer_string", True):
            ser = Series("a", index=[1, 2], dtype="string[python]")
        expected = Series(["a", "a"], index=[1, 2], dtype="string[python]")
        tm.assert_series_equal(ser, expected)
        assert ser.dtype.storage == "python"

    def test_series_string_inference_na_first(self):
        # GH#55655
        with pd.option_context("future.infer_string", True):
            result = Series([pd.NA, "b"])
        dtype = pd.StringDtype("pyarrow" if HAS_PYARROW else "python", na_value=np.nan)
        expected = Series([None, "b"], dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_inference_on_pandas_objects(self):
        # GH#56012
        ser = Series([Timestamp("2019-12-31")], dtype=object)
        with tm.assert_produces_warning(None):
            # This doesn't do inference
            result = Series(ser)
        assert result.dtype == np.object_

        idx = Index([Timestamp("2019-12-31")], dtype=object)

        with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
            result = Series(idx)
        assert result.dtype != np.object_


class TestSeriesConstructorIndexCoercion:
    def test_series_constructor_datetimelike_index_coercion(self):
        idx = date_range("2020-01-01", periods=5)
        ser = Series(
            np.random.default_rng(2).standard_normal(len(idx)), idx.astype(object)
        )
        # as of 2.0, we no longer silently cast the object-dtype index
        #  to DatetimeIndex GH#39307, GH#23598
        assert not isinstance(ser.index, DatetimeIndex)

    @pytest.mark.parametrize("container", [None, np.array, Series, Index])
    @pytest.mark.parametrize("data", [1.0, range(4)])
    def test_series_constructor_infer_multiindex(self, container, data):
        indexes = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]
        if container is not None:
            indexes = [container(ind) for ind in indexes]

        multi = Series(data, index=indexes)
        assert isinstance(multi.index, MultiIndex)

    # TODO: make this not cast to object in pandas 3.0
    @pytest.mark.skipif(
        not np_version_gt2, reason="StringDType only available in numpy 2 and above"
    )
    @pytest.mark.parametrize(
        "data",
        [
            ["a", "b", "c"],
            ["a", "b", np.nan],
        ],
    )
    def test_np_string_array_object_cast(self, data):
        from numpy.dtypes import StringDType

        arr = np.array(data, dtype=StringDType())
        res = Series(arr)
        assert res.dtype == np.object_
        assert (res == data).all()


class TestSeriesConstructorInternals:
    def test_constructor_no_pandas_array(self, using_array_manager):
        ser = Series([1, 2, 3])
        result = Series(ser.array)
        tm.assert_series_equal(ser, result)
        if not using_array_manager:
            assert isinstance(result._mgr.blocks[0], NumpyBlock)
            assert result._mgr.blocks[0].is_numeric

    @td.skip_array_manager_invalid_test
    def test_from_array(self):
        result = Series(pd.array(["1h", "2h"], dtype="timedelta64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

        result = Series(pd.array(["2015"], dtype="datetime64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

    @td.skip_array_manager_invalid_test
    def test_from_list_dtype(self):
        result = Series(["1h", "2h"], dtype="timedelta64[ns]")
        assert result._mgr.blocks[0].is_extension is False

        result = Series(["2015"], dtype="datetime64[ns]")
        assert result._mgr.blocks[0].is_extension is False


def test_constructor(rand_series_with_duplicate_datetimeindex):
    dups = rand_series_with_duplicate_datetimeindex
    assert isinstance(dups, Series)
    assert isinstance(dups.index, DatetimeIndex)


@pytest.mark.parametrize(
    "input_dict,expected",
    [
        ({0: 0}, np.array([[0]], dtype=np.int64)),
        ({"a": "a"}, np.array([["a"]], dtype=object)),
        ({1: 1}, np.array([[1]], dtype=np.int64)),
    ],
)
def test_numpy_array(input_dict, expected):
    result = np.array([Series(input_dict)])
    tm.assert_numpy_array_equal(result, expected)


def test_index_ordered_dict_keys():
    # GH 22077

    param_index = OrderedDict(
        [
            ((("a", "b"), ("c", "d")), 1),
            ((("a", None), ("c", "d")), 2),
        ]
    )
    series = Series([1, 2], index=param_index.keys())
    expected = Series(
        [1, 2],
        index=MultiIndex.from_tuples(
            [(("a", "b"), ("c", "d")), (("a", None), ("c", "d"))]
        ),
    )
    tm.assert_series_equal(series, expected)


@pytest.mark.parametrize(
    "input_list",
    [
        [1, complex("nan"), 2],
        [1 + 1j, complex("nan"), 2 + 2j],
    ],
)
def test_series_with_complex_nan(input_list):
    # GH#53627
    ser = Series(input_list)
    result = Series(ser.array)
    assert ser.dtype == "complex128"
    tm.assert_series_equal(ser, result)