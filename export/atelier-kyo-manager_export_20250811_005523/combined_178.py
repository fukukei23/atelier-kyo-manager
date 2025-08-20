
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\__init__.py ===
# A highish-level implementation of the HTTP/1.1 wire protocol (RFC 7230),
# containing no networking code at all, loosely modelled on hyper-h2's generic
# implementation of HTTP/2 (and in particular the h2.connection.H2Connection
# class). There's still a bunch of subtle details you need to get right if you
# want to make this actually useful, because it doesn't implement all the
# semantics to check that what you're asking to write to the wire is sensible,
# but at least it gets you out of dealing with the wire itself.

from h11._connection import Connection, NEED_DATA, PAUSED
from h11._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from h11._state import (
    CLIENT,
    CLOSED,
    DONE,
    ERROR,
    IDLE,
    MIGHT_SWITCH_PROTOCOL,
    MUST_CLOSE,
    SEND_BODY,
    SEND_RESPONSE,
    SERVER,
    SWITCHED_PROTOCOL,
)
from h11._util import LocalProtocolError, ProtocolError, RemoteProtocolError
from h11._version import __version__

PRODUCT_ID = "python-h11/" + __version__


__all__ = (
    "Connection",
    "NEED_DATA",
    "PAUSED",
    "ConnectionClosed",
    "Data",
    "EndOfMessage",
    "Event",
    "InformationalResponse",
    "Request",
    "Response",
    "CLIENT",
    "CLOSED",
    "DONE",
    "ERROR",
    "IDLE",
    "MUST_CLOSE",
    "SEND_BODY",
    "SEND_RESPONSE",
    "SERVER",
    "SWITCHED_PROTOCOL",
    "ProtocolError",
    "LocalProtocolError",
    "RemoteProtocolError",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\_isocbind.py ===
"""
ISO_C_BINDING maps for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# These map to keys in c2py_map, via forced casting for now, see gh-25229
iso_c_binding_map = {
    'integer': {
        'c_int': 'int',
        'c_short': 'short',  # 'short' <=> 'int' for now
        'c_long': 'long',  # 'long' <=> 'int' for now
        'c_long_long': 'long_long',
        'c_signed_char': 'signed_char',
        'c_size_t': 'unsigned',  # size_t <=> 'unsigned' for now
        'c_int8_t': 'signed_char',  # int8_t <=> 'signed_char' for now
        'c_int16_t': 'short',  # int16_t <=> 'short' for now
        'c_int32_t': 'int',  # int32_t <=> 'int' for now
        'c_int64_t': 'long_long',
        'c_int_least8_t': 'signed_char',  # int_least8_t <=> 'signed_char' for now
        'c_int_least16_t': 'short',  # int_least16_t <=> 'short' for now
        'c_int_least32_t': 'int',  # int_least32_t <=> 'int' for now
        'c_int_least64_t': 'long_long',
        'c_int_fast8_t': 'signed_char',  # int_fast8_t <=> 'signed_char' for now
        'c_int_fast16_t': 'short',  # int_fast16_t <=> 'short' for now
        'c_int_fast32_t': 'int',  # int_fast32_t <=> 'int' for now
        'c_int_fast64_t': 'long_long',
        'c_intmax_t': 'long_long',  # intmax_t <=> 'long_long' for now
        'c_intptr_t': 'long',  # intptr_t <=> 'long' for now
        'c_ptrdiff_t': 'long',  # ptrdiff_t <=> 'long' for now
    },
    'real': {
        'c_float': 'float',
        'c_double': 'double',
        'c_long_double': 'long_double'
    },
    'complex': {
        'c_float_complex': 'complex_float',
        'c_double_complex': 'complex_double',
        'c_long_double_complex': 'complex_long_double'
    },
    'logical': {
        'c_bool': 'unsigned_char'  # _Bool <=> 'unsigned_char' for now
    },
    'character': {
        'c_char': 'char'
    }
}

# TODO: See gh-25229
isoc_c2pycode_map = {}
iso_c2py_map = {}

isoc_kindmap = {}
for fortran_type, c_type_dict in iso_c_binding_map.items():
    for c_type in c_type_dict.keys():
        isoc_kindmap[c_type] = fortran_type

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\concat.py ===
import onnx

from ..quant_utils import (  # noqa: F401
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    attribute_to_kwarg,
    ms_domain,
)
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase  # noqa: F401


class QLinearConcat(QuantOperatorBase):
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
            q_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [*range(len(node.input))])
        if not data_found or q_input_names is None:
            return super().quantize()

        # Create an entry for output quantized value
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        quantized_output_value = QuantizedValue(
            node.output[0],
            node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
            output_scale_name,
            output_zp_name,
            quantized_input_value.value_type,
        )
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qnode_name = node.name + "_quant" if node.name else ""

        qlconcat_inputs = [output_scale_name, output_zp_name]
        for i in range(len(q_input_names)):
            qlconcat_inputs.extend([q_input_names[i], scale_names[i], zero_point_names[i]])
        qlconcat_node = onnx.helper.make_node(
            "QLinearConcat", qlconcat_inputs, [quantized_output_value.q_name], qnode_name, **kwargs
        )

        self.quantizer.new_nodes += nodes
        self.quantizer.new_nodes += [qlconcat_node]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\gavgpool.py ===
import onnx

from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType, attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase


class QGlobalAveragePool(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "GlobalAveragePool"

        # If input to this node is not quantized then keep this node.
        if node.input[0] not in self.quantizer.quantized_value_map:
            return super().quantize()

        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]

        # Create an entry for output quantized value.
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        (
            data_found,
            output_scale_name_from_parameter,
            output_zp_name_from_parameter,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])
        # Just use input scale and zp if parameters for output is not specified.
        output_scale_name = output_scale_name_from_parameter if data_found else quantized_input_value.scale_name
        output_zp_name = output_zp_name_from_parameter if data_found else quantized_input_value.zp_name
        quantized_output_value = QuantizedValue(
            node.output[0],
            node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        kwargs["channels_last"] = 0
        qnode_name = node.name + "_quant" if node.name else ""

        qnode = onnx.helper.make_node(
            "QLinear" + node.op_type,
            [
                quantized_input_value.q_name,
                quantized_input_value.scale_name,
                quantized_input_value.zp_name,
                output_scale_name,
                output_zp_name,
            ],
            [quantized_output_value.q_name],
            qnode_name,
            **kwargs,
        )
        self.quantizer.new_nodes += [qnode]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\utils.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import typing

from ..logger import get_logger
from .operator_type_usage_processors import OperatorTypeUsageManager
from .ort_model_processor import OrtFormatModelProcessor

log = get_logger("ort_format_model.utils")


def _extract_ops_and_types_from_ort_models(model_files: typing.Iterable[pathlib.Path], enable_type_reduction: bool):
    required_ops = {}
    op_type_usage_manager = OperatorTypeUsageManager() if enable_type_reduction else None

    for model_file in model_files:
        if not model_file.is_file():
            raise ValueError(f"Path is not a file: '{model_file}'")
        model_processor = OrtFormatModelProcessor(str(model_file), required_ops, op_type_usage_manager)
        model_processor.process()  # this updates required_ops and op_type_processors

    return required_ops, op_type_usage_manager


def create_config_from_models(
    model_files: typing.Iterable[pathlib.Path], output_file: pathlib.Path, enable_type_reduction: bool
):
    """
    Create a configuration file with required operators and optionally required types.
    :param model_files: Model files to use to generate the configuration file.
    :param output_file: File to write configuration to.
    :param enable_type_reduction: Include required type information for individual operators in the configuration.
    """

    required_ops, op_type_processors = _extract_ops_and_types_from_ort_models(model_files, enable_type_reduction)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as out:
        out.write("# Generated from model/s:\n")
        for model_file in sorted(model_files):
            out.write(f"# - {model_file}\n")

        for domain in sorted(required_ops.keys()):
            for opset in sorted(required_ops[domain].keys()):
                ops = required_ops[domain][opset]
                if ops:
                    out.write(f"{domain};{opset};")
                    if enable_type_reduction:
                        # type string is empty if op hasn't been seen
                        entries = [
                            "{}{}".format(op, op_type_processors.get_config_entry(domain, op) or "")
                            for op in sorted(ops)
                        ]
                    else:
                        entries = sorted(ops)

                    out.write("{}\n".format(",".join(entries)))

    log.info("Created config in %s", output_file)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\error_bar.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Float,
    Set,
    Alias
)

from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.nested import (
    NestedNoneSet,
    NestedSet,
    NestedBool,
    NestedFloat,
)

from .data_source import NumDataSource
from .shapes import GraphicalProperties


class ErrorBars(Serialisable):

    tagname = "errBars"

    errDir = NestedNoneSet(values=(['x', 'y']))
    direction = Alias("errDir")
    errBarType = NestedSet(values=(['both', 'minus', 'plus']))
    style = Alias("errBarType")
    errValType = NestedSet(values=(['cust', 'fixedVal', 'percentage', 'stdDev', 'stdErr']))
    size = Alias("errValType")
    noEndCap = NestedBool(nested=True, allow_none=True)
    plus = Typed(expected_type=NumDataSource, allow_none=True)
    minus = Typed(expected_type=NumDataSource, allow_none=True)
    val = NestedFloat(allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias("spPr")
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('errDir','errBarType', 'errValType', 'noEndCap','minus', 'plus', 'val', 'spPr')


    def __init__(self,
                 errDir=None,
                 errBarType="both",
                 errValType="fixedVal",
                 noEndCap=None,
                 plus=None,
                 minus=None,
                 val=None,
                 spPr=None,
                 extLst=None,
                ):
        self.errDir = errDir
        self.errBarType = errBarType
        self.errValType = errValType
        self.noEndCap = noEndCap
        self.plus = plus
        self.minus = minus
        self.val = val
        self.spPr = spPr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\comments\comments.py ===
# Copyright (c) 2010-2024 openpyxl


class Comment:

    _parent = None

    def __init__(self, text, author, height=79, width=144):
        self.content = text
        self.author = author
        self.height = height
        self.width = width


    @property
    def parent(self):
        return self._parent


    def __eq__(self, other):
        return (
            self.content == other.content
            and self.author == other.author
        )

    def __repr__(self):
        return "Comment: {0} by {1}".format(self.content, self.author)


    def __copy__(self):
        """Create a detached copy of this comment."""
        clone = self.__class__(self.content, self.author, self.height, self.width)
        return clone


    def bind(self, cell):
        """
        Bind comment to a particular cell
        """
        if cell is not None and self._parent is not None and self._parent != cell:
            fmt = "Comment already assigned to {0} in worksheet {1}. Cannot assign a comment to more than one cell"
            raise AttributeError(fmt.format(cell.coordinate, cell.parent.title))
        self._parent = cell


    def unbind(self):
        """
        Unbind a comment from a cell
        """
        self._parent = None


    @property
    def text(self):
        """
        Any comment text stripped of all formatting.
        """
        return self.content

    @text.setter
    def text(self, value):
        self.content = value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\alignment.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.compat import safe_string

from openpyxl.descriptors import Bool, MinMax, Min, Alias, NoneSet
from openpyxl.descriptors.serialisable import Serialisable


horizontal_alignments = (
    "general", "left", "center", "right", "fill", "justify", "centerContinuous",
    "distributed", )
vertical_aligments = (
    "top", "center", "bottom", "justify", "distributed",
)

class Alignment(Serialisable):
    """Alignment options for use in styles."""

    tagname = "alignment"

    horizontal = NoneSet(values=horizontal_alignments)
    vertical = NoneSet(values=vertical_aligments)
    textRotation = NoneSet(values=range(181))
    textRotation.values.add(255)
    text_rotation = Alias('textRotation')
    wrapText = Bool(allow_none=True)
    wrap_text = Alias('wrapText')
    shrinkToFit = Bool(allow_none=True)
    shrink_to_fit = Alias('shrinkToFit')
    indent = MinMax(min=0, max=255)
    relativeIndent = MinMax(min=-255, max=255)
    justifyLastLine = Bool(allow_none=True)
    readingOrder = Min(min=0)

    def __init__(self, horizontal=None, vertical=None,
                 textRotation=0, wrapText=None, shrinkToFit=None, indent=0, relativeIndent=0,
                 justifyLastLine=None, readingOrder=0, text_rotation=None,
                 wrap_text=None, shrink_to_fit=None, mergeCell=None):
        self.horizontal = horizontal
        self.vertical = vertical
        self.indent = indent
        self.relativeIndent = relativeIndent
        self.justifyLastLine = justifyLastLine
        self.readingOrder = readingOrder
        if text_rotation is not None:
            textRotation = text_rotation
        if textRotation is not None:
            self.textRotation = int(textRotation)
        if wrap_text is not None:
            wrapText = wrap_text
        self.wrapText = wrapText
        if shrink_to_fit is not None:
            shrinkToFit = shrink_to_fit
        self.shrinkToFit = shrinkToFit
        # mergeCell is vestigial


    def __iter__(self):
        for attr in self.__attrs__:
            value = getattr(self, attr)
            if value is not None and value != 0:
                yield attr, safe_string(value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\proxy.py ===
# Copyright (c) 2010-2024 openpyxl

from copy import copy

from openpyxl.compat import deprecated


class StyleProxy:
    """
    Proxy formatting objects so that they cannot be altered
    """

    __slots__ = ('__target')

    def __init__(self, target):
        self.__target = target


    def __repr__(self):
        return repr(self.__target)


    def __getattr__(self, attr):
        return getattr(self.__target, attr)


    def __setattr__(self, attr, value):
        if attr != "_StyleProxy__target":
            raise AttributeError("Style objects are immutable and cannot be changed."
                                 "Reassign the style with a copy")
        super().__setattr__(attr, value)


    def __copy__(self):
        """
        Return a copy of the proxied object.
        """
        return copy(self.__target)


    def __add__(self, other):
        """
        Add proxied object to another instance and return the combined object
        """
        return self.__target + other


    @deprecated("Use copy(obj) or cell.obj = cell.obj + other")
    def copy(self, **kw):
        """Return a copy of the proxied object. Keyword args will be passed through"""
        cp = copy(self.__target)
        for k, v in kw.items():
            setattr(cp, k, v)
        return cp


    def __eq__(self, other):
        return self.__target == other


    def __ne__(self, other):
        return not self == other

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\roperator.py ===
"""
Reversed Operations not available in the stdlib operator module.
Defining these instead of using lambdas allows us to reference them by name.
"""
from __future__ import annotations

import operator


def radd(left, right):
    return right + left


def rsub(left, right):
    return right - left


def rmul(left, right):
    return right * left


def rdiv(left, right):
    return right / left


def rtruediv(left, right):
    return right / left


def rfloordiv(left, right):
    return right // left


def rmod(left, right):
    # check if right is a string as % is the string
    # formatting operation; this is a TypeError
    # otherwise perform the op
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f"{typ} cannot perform the operation mod")

    return right % left


def rdivmod(left, right):
    return divmod(right, left)


def rpow(left, right):
    return right**left


def rand_(left, right):
    return operator.and_(right, left)


def ror_(left, right):
    return operator.or_(right, left)


def rxor(left, right):
    return operator.xor(right, left)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\ops\invalid.py ===
"""
Templates for invalid operations.
"""
from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt


def invalid_comparison(left, right, op) -> npt.NDArray[np.bool_]:
    """
    If a comparison has mismatched types and is not necessarily meaningful,
    follow python3 conventions by:

        - returning all-False for equality
        - returning all-True for inequality
        - raising TypeError otherwise

    Parameters
    ----------
    left : array-like
    right : scalar, array-like
    op : operator.{eq, ne, lt, le, gt}

    Raises
    ------
    TypeError : on inequality comparisons
    """
    if op is operator.eq:
        res_values = np.zeros(left.shape, dtype=bool)
    elif op is operator.ne:
        res_values = np.ones(left.shape, dtype=bool)
    else:
        typ = type(right).__name__
        raise TypeError(f"Invalid comparison between dtype={left.dtype} and {typ}")
    return res_values


def make_invalid_op(name: str):
    """
    Return a binary method that always raises a TypeError.

    Parameters
    ----------
    name : str

    Returns
    -------
    invalid_op : function
    """

    def invalid_op(self, other=None):
        typ = type(self).__name__
        raise TypeError(f"cannot perform {name} with this index type: {typ}")

    invalid_op.__name__ = name
    return invalid_op

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_config\display.py ===
"""
Unopinionated display configuration.
"""

from __future__ import annotations

import locale
import sys

from pandas._config import config as cf

# -----------------------------------------------------------------------------
# Global formatting options
_initial_defencoding: str | None = None


def detect_console_encoding() -> str:
    """
    Try to find the most capable encoding supported by the console.
    slightly modified from the way IPython handles the same issue.
    """
    global _initial_defencoding

    encoding = None
    try:
        encoding = sys.stdout.encoding or sys.stdin.encoding
    except (AttributeError, OSError):
        pass

    # try again for something better
    if not encoding or "ascii" in encoding.lower():
        try:
            encoding = locale.getpreferredencoding()
        except locale.Error:
            # can be raised by locale.setlocale(), which is
            #  called by getpreferredencoding
            #  (on some systems, see stdlib locale docs)
            pass

    # when all else fails. this will usually be "ascii"
    if not encoding or "ascii" in encoding.lower():
        encoding = sys.getdefaultencoding()

    # GH#3360, save the reported defencoding at import time
    # MPL backends may change it. Make available for debugging.
    if not _initial_defencoding:
        _initial_defencoding = sys.getdefaultencoding()

    return encoding


pc_encoding_doc = """
: str/unicode
    Defaults to the detected encoding of the console.
    Specifies the encoding to be used for strings returned by to_string,
    these are generally strings meant to be displayed on the console.
"""

with cf.config_prefix("display"):
    cf.register_option(
        "encoding", detect_console_encoding(), pc_encoding_doc, validator=cf.is_text
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\dependency_groups\_pip_wrapper.py ===
from __future__ import annotations

import argparse
import subprocess
import sys

from ._implementation import DependencyGroupResolver
from ._toml_compat import tomllib


def _invoke_pip(deps: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])


def main(*, argv: list[str] | None = None) -> None:
    if tomllib is None:
        print(
            "Usage error: dependency-groups CLI requires tomli or Python 3.11+",
            file=sys.stderr,
        )
        raise SystemExit(2)

    parser = argparse.ArgumentParser(description="Install Dependency Groups.")
    parser.add_argument(
        "DEPENDENCY_GROUP", nargs="+", help="The dependency groups to install."
    )
    parser.add_argument(
        "-f",
        "--pyproject-file",
        default="pyproject.toml",
        help="The pyproject.toml file. Defaults to trying in the current directory.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    with open(args.pyproject_file, "rb") as fp:
        pyproject = tomllib.load(fp)
    dependency_groups_raw = pyproject.get("dependency-groups", {})

    errors: list[str] = []
    resolved: list[str] = []
    try:
        resolver = DependencyGroupResolver(dependency_groups_raw)
    except (ValueError, TypeError) as e:
        errors.append(f"{type(e).__name__}: {e}")
    else:
        for groupname in args.DEPENDENCY_GROUP:
            try:
                resolved.extend(str(r) for r in resolver.resolve(groupname))
            except (LookupError, ValueError, TypeError) as e:
                errors.append(f"{type(e).__name__}: {e}")

    if errors:
        print("errors encountered while examining dependency groups:")
        for msg in errors:
            print(f"  {msg}")
        sys.exit(1)

    _invoke_pip(resolved)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\__init__.py ===
# engine/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""SQL connections, SQL execution and high-level DB-API interface.

The engine package defines the basic components used to interface
DB-API modules with higher-level statement construction,
connection-management, execution and result contexts.  The primary
"entry point" class into this package is the Engine and its public
constructor ``create_engine()``.

"""

from . import events as events
from . import util as util
from .base import Connection as Connection
from .base import Engine as Engine
from .base import NestedTransaction as NestedTransaction
from .base import RootTransaction as RootTransaction
from .base import Transaction as Transaction
from .base import TwoPhaseTransaction as TwoPhaseTransaction
from .create import create_engine as create_engine
from .create import create_pool_from_url as create_pool_from_url
from .create import engine_from_config as engine_from_config
from .cursor import CursorResult as CursorResult
from .cursor import ResultProxy as ResultProxy
from .interfaces import AdaptedConnection as AdaptedConnection
from .interfaces import BindTyping as BindTyping
from .interfaces import Compiled as Compiled
from .interfaces import Connectable as Connectable
from .interfaces import ConnectArgsType as ConnectArgsType
from .interfaces import ConnectionEventsTarget as ConnectionEventsTarget
from .interfaces import CreateEnginePlugin as CreateEnginePlugin
from .interfaces import Dialect as Dialect
from .interfaces import ExceptionContext as ExceptionContext
from .interfaces import ExecutionContext as ExecutionContext
from .interfaces import TypeCompiler as TypeCompiler
from .mock import create_mock_engine as create_mock_engine
from .reflection import Inspector as Inspector
from .reflection import ObjectKind as ObjectKind
from .reflection import ObjectScope as ObjectScope
from .result import ChunkedIteratorResult as ChunkedIteratorResult
from .result import FilterResult as FilterResult
from .result import FrozenResult as FrozenResult
from .result import IteratorResult as IteratorResult
from .result import MappingResult as MappingResult
from .result import MergedResult as MergedResult
from .result import Result as Result
from .result import result_tuple as result_tuple
from .result import ScalarResult as ScalarResult
from .result import TupleResult as TupleResult
from .row import BaseRow as BaseRow
from .row import Row as Row
from .row import RowMapping as RowMapping
from .url import make_url as make_url
from .url import URL as URL
from .util import connection_memoize as connection_memoize
from ..sql import ddl as ddl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\factorizations.py ===
from sympy.matrices.expressions import MatrixExpr
from sympy.assumptions.ask import Q

class Factorization(MatrixExpr):
    arg = property(lambda self: self.args[0])
    shape = property(lambda self: self.arg.shape)  # type: ignore

class LofLU(Factorization):
    @property
    def predicates(self):
        return (Q.lower_triangular,)
class UofLU(Factorization):
    @property
    def predicates(self):
        return (Q.upper_triangular,)

class LofCholesky(LofLU): pass
class UofCholesky(UofLU): pass

class QofQR(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)
class RofQR(Factorization):
    @property
    def predicates(self):
        return (Q.upper_triangular,)

class EigenVectors(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)
class EigenValues(Factorization):
    @property
    def predicates(self):
        return (Q.diagonal,)

class UofSVD(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)
class SofSVD(Factorization):
    @property
    def predicates(self):
        return (Q.diagonal,)
class VofSVD(Factorization):
    @property
    def predicates(self):
        return (Q.orthogonal,)


def lu(expr):
    return LofLU(expr), UofLU(expr)

def qr(expr):
    return QofQR(expr), RofQR(expr)

def eig(expr):
    return EigenValues(expr), EigenVectors(expr)

def svd(expr):
    return UofSVD(expr), SofSVD(expr), VofSVD(expr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\__init__.py ===
""" A module which handles Matrix Expressions """

from .slice import MatrixSlice
from .blockmatrix import BlockMatrix, BlockDiagMatrix, block_collapse, blockcut
from .companion import CompanionMatrix
from .funcmatrix import FunctionMatrix
from .inverse import Inverse
from .matadd import MatAdd
from .matexpr import MatrixExpr, MatrixSymbol, matrix_symbols
from .matmul import MatMul
from .matpow import MatPow
from .trace import Trace, trace
from .determinant import Determinant, det, Permanent, per
from .transpose import Transpose
from .adjoint import Adjoint
from .hadamard import hadamard_product, HadamardProduct, hadamard_power, HadamardPower
from .diagonal import DiagonalMatrix, DiagonalOf, DiagMatrix, diagonalize_vector
from .dotproduct import DotProduct
from .kronecker import kronecker_product, KroneckerProduct, combine_kronecker
from .permutation import PermutationMatrix, MatrixPermute
from .sets import MatrixSet
from .special import ZeroMatrix, Identity, OneMatrix

__all__ = [
    'MatrixSlice',

    'BlockMatrix', 'BlockDiagMatrix', 'block_collapse', 'blockcut',
    'FunctionMatrix',

    'CompanionMatrix',

    'Inverse',

    'MatAdd',

    'Identity', 'MatrixExpr', 'MatrixSymbol', 'ZeroMatrix', 'OneMatrix',
    'matrix_symbols', 'MatrixSet',

    'MatMul',

    'MatPow',

    'Trace', 'trace',

    'Determinant', 'det',

    'Transpose',

    'Adjoint',

    'hadamard_product', 'HadamardProduct', 'hadamard_power', 'HadamardPower',

    'DiagonalMatrix', 'DiagonalOf', 'DiagMatrix', 'diagonalize_vector',

    'DotProduct',

    'kronecker_product', 'KroneckerProduct', 'combine_kronecker',

    'PermutationMatrix', 'MatrixPermute',

    'Permanent', 'per'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\computation\test_eval.py ===
from __future__ import annotations

from functools import reduce
from itertools import product
import operator

import numpy as np
import pytest

from pandas.compat import PY312
from pandas.errors import (
    NumExprClobberingError,
    PerformanceWarning,
    UndefinedVariableError,
)
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import (
    is_bool,
    is_float,
    is_list_like,
    is_scalar,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.computation import (
    expr,
    pytables,
)
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
    BaseExprVisitor,
    PandasExprVisitor,
    PythonExprVisitor,
)
from pandas.core.computation.expressions import (
    NUMEXPR_INSTALLED,
    USE_NUMEXPR,
)
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    SPECIAL_CASE_ARITH_OPS_SYMS,
    _binary_math_ops,
    _binary_ops_dict,
    _unary_math_ops,
)
from pandas.core.computation.scope import DEFAULT_GLOBALS


@pytest.fixture(
    params=(
        pytest.param(
            engine,
            marks=[
                pytest.mark.skipif(
                    engine == "numexpr" and not USE_NUMEXPR,
                    reason=f"numexpr enabled->{USE_NUMEXPR}, "
                    f"installed->{NUMEXPR_INSTALLED}",
                ),
                td.skip_if_no("numexpr"),
            ],
        )
        for engine in ENGINES
    )
)
def engine(request):
    return request.param


@pytest.fixture(params=expr.PARSERS)
def parser(request):
    return request.param


def _eval_single_bin(lhs, cmp1, rhs, engine):
    c = _binary_ops_dict[cmp1]
    if ENGINES[engine].has_neg_frac:
        try:
            return c(lhs, rhs)
        except ValueError as e:
            if str(e).startswith(
                "negative number cannot be raised to a fractional power"
            ):
                return np.nan
            raise
    return c(lhs, rhs)


# TODO: using range(5) here is a kludge
@pytest.fixture(
    params=list(range(5)),
    ids=["DataFrame", "Series", "SeriesNaN", "DataFrameNaN", "float"],
)
def lhs(request):
    nan_df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    nan_df1[nan_df1 > 0.5] = np.nan

    opts = (
        DataFrame(np.random.default_rng(2).standard_normal((10, 5))),
        Series(np.random.default_rng(2).standard_normal(5)),
        Series([1, 2, np.nan, np.nan, 5]),
        nan_df1,
        np.random.default_rng(2).standard_normal(),
    )
    return opts[request.param]


rhs = lhs
midhs = lhs


@pytest.fixture
def idx_func_dict():
    return {
        "i": lambda n: Index(np.arange(n), dtype=np.int64),
        "f": lambda n: Index(np.arange(n), dtype=np.float64),
        "s": lambda n: Index([f"{i}_{chr(i)}" for i in range(97, 97 + n)]),
        "dt": lambda n: date_range("2020-01-01", periods=n),
        "td": lambda n: timedelta_range("1 day", periods=n),
        "p": lambda n: period_range("2020-01-01", periods=n, freq="D"),
    }


class TestEval:
    @pytest.mark.parametrize(
        "cmp1",
        ["!=", "==", "<=", ">=", "<", ">"],
        ids=["ne", "eq", "le", "ge", "lt", "gt"],
    )
    @pytest.mark.parametrize("cmp2", [">", "<"], ids=["gt", "lt"])
    @pytest.mark.parametrize("binop", expr.BOOL_OPS_SYMS)
    def test_complex_cmp_ops(self, cmp1, cmp2, binop, lhs, rhs, engine, parser):
        if parser == "python" and binop in ["and", "or"]:
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                ex = f"(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)"
                pd.eval(ex, engine=engine, parser=parser)
            return

        lhs_new = _eval_single_bin(lhs, cmp1, rhs, engine)
        rhs_new = _eval_single_bin(lhs, cmp2, rhs, engine)
        expected = _eval_single_bin(lhs_new, binop, rhs_new, engine)

        ex = f"(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)"
        result = pd.eval(ex, engine=engine, parser=parser)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("cmp_op", expr.CMP_OPS_SYMS)
    def test_simple_cmp_ops(self, cmp_op, lhs, rhs, engine, parser):
        lhs = lhs < 0
        rhs = rhs < 0

        if parser == "python" and cmp_op in ["in", "not in"]:
            msg = "'(In|NotIn)' nodes are not implemented"

            with pytest.raises(NotImplementedError, match=msg):
                ex = f"lhs {cmp_op} rhs"
                pd.eval(ex, engine=engine, parser=parser)
            return

        ex = f"lhs {cmp_op} rhs"
        msg = "|".join(
            [
                r"only list-like( or dict-like)? objects are allowed to be "
                r"passed to (DataFrame\.)?isin\(\), you passed a "
                r"(`|')bool(`|')",
                "argument of type 'bool' is not iterable",
            ]
        )
        if cmp_op in ("in", "not in") and not is_list_like(rhs):
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    engine=engine,
                    parser=parser,
                    local_dict={"lhs": lhs, "rhs": rhs},
                )
        else:
            expected = _eval_single_bin(lhs, cmp_op, rhs, engine)
            result = pd.eval(ex, engine=engine, parser=parser)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize("op", expr.CMP_OPS_SYMS)
    def test_compound_invert_op(self, op, lhs, rhs, request, engine, parser):
        if parser == "python" and op in ["in", "not in"]:
            msg = "'(In|NotIn)' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                ex = f"~(lhs {op} rhs)"
                pd.eval(ex, engine=engine, parser=parser)
            return

        if (
            is_float(lhs)
            and not is_float(rhs)
            and op in ["in", "not in"]
            and engine == "python"
            and parser == "pandas"
        ):
            mark = pytest.mark.xfail(
                reason="Looks like expected is negative, unclear whether "
                "expected is incorrect or result is incorrect"
            )
            request.applymarker(mark)
        skip_these = ["in", "not in"]
        ex = f"~(lhs {op} rhs)"

        msg = "|".join(
            [
                r"only list-like( or dict-like)? objects are allowed to be "
                r"passed to (DataFrame\.)?isin\(\), you passed a "
                r"(`|')float(`|')",
                "argument of type 'float' is not iterable",
            ]
        )
        if is_scalar(rhs) and op in skip_these:
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    engine=engine,
                    parser=parser,
                    local_dict={"lhs": lhs, "rhs": rhs},
                )
        else:
            # compound
            if is_scalar(lhs) and is_scalar(rhs):
                lhs, rhs = (np.array([x]) for x in (lhs, rhs))
            expected = _eval_single_bin(lhs, op, rhs, engine)
            if is_scalar(expected):
                expected = not expected
            else:
                expected = ~expected
            result = pd.eval(ex, engine=engine, parser=parser)
            tm.assert_almost_equal(expected, result)

    @pytest.mark.parametrize("cmp1", ["<", ">"])
    @pytest.mark.parametrize("cmp2", ["<", ">"])
    def test_chained_cmp_op(self, cmp1, cmp2, lhs, midhs, rhs, engine, parser):
        mid = midhs
        if parser == "python":
            ex1 = f"lhs {cmp1} mid {cmp2} rhs"
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex1, engine=engine, parser=parser)
            return

        lhs_new = _eval_single_bin(lhs, cmp1, mid, engine)
        rhs_new = _eval_single_bin(mid, cmp2, rhs, engine)

        if lhs_new is not None and rhs_new is not None:
            ex1 = f"lhs {cmp1} mid {cmp2} rhs"
            ex2 = f"lhs {cmp1} mid and mid {cmp2} rhs"
            ex3 = f"(lhs {cmp1} mid) & (mid {cmp2} rhs)"
            expected = _eval_single_bin(lhs_new, "&", rhs_new, engine)

            for ex in (ex1, ex2, ex3):
                result = pd.eval(ex, engine=engine, parser=parser)

                tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "arith1", sorted(set(ARITH_OPS_SYMS).difference(SPECIAL_CASE_ARITH_OPS_SYMS))
    )
    def test_binary_arith_ops(self, arith1, lhs, rhs, engine, parser):
        ex = f"lhs {arith1} rhs"
        result = pd.eval(ex, engine=engine, parser=parser)
        expected = _eval_single_bin(lhs, arith1, rhs, engine)

        tm.assert_almost_equal(result, expected)
        ex = f"lhs {arith1} rhs {arith1} rhs"
        result = pd.eval(ex, engine=engine, parser=parser)
        nlhs = _eval_single_bin(lhs, arith1, rhs, engine)
        try:
            nlhs, ghs = nlhs.align(rhs)
        except (ValueError, TypeError, AttributeError):
            # ValueError: series frame or frame series align
            # TypeError, AttributeError: series or frame with scalar align
            return
        else:
            if engine == "numexpr":
                import numexpr as ne

                # direct numpy comparison
                expected = ne.evaluate(f"nlhs {arith1} ghs")
                # Update assert statement due to unreliable numerical
                # precision component (GH37328)
                # TODO: update testing code so that assert_almost_equal statement
                #  can be replaced again by the assert_numpy_array_equal statement
                tm.assert_almost_equal(result.values, expected)
            else:
                expected = eval(f"nlhs {arith1} ghs")
                tm.assert_almost_equal(result, expected)

    # modulus, pow, and floor division require special casing

    def test_modulus(self, lhs, rhs, engine, parser):
        ex = r"lhs % rhs"
        result = pd.eval(ex, engine=engine, parser=parser)
        expected = lhs % rhs
        tm.assert_almost_equal(result, expected)

        if engine == "numexpr":
            import numexpr as ne

            expected = ne.evaluate(r"expected % rhs")
            if isinstance(result, (DataFrame, Series)):
                tm.assert_almost_equal(result.values, expected)
            else:
                tm.assert_almost_equal(result, expected.item())
        else:
            expected = _eval_single_bin(expected, "%", rhs, engine)
            tm.assert_almost_equal(result, expected)

    def test_floor_division(self, lhs, rhs, engine, parser):
        ex = "lhs // rhs"

        if engine == "python":
            res = pd.eval(ex, engine=engine, parser=parser)
            expected = lhs // rhs
            tm.assert_equal(res, expected)
        else:
            msg = (
                r"unsupported operand type\(s\) for //: 'VariableNode' and "
                "'VariableNode'"
            )
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    local_dict={"lhs": lhs, "rhs": rhs},
                    engine=engine,
                    parser=parser,
                )

    @td.skip_if_windows
    def test_pow(self, lhs, rhs, engine, parser):
        # odd failure on win32 platform, so skip
        ex = "lhs ** rhs"
        expected = _eval_single_bin(lhs, "**", rhs, engine)
        result = pd.eval(ex, engine=engine, parser=parser)

        if (
            is_scalar(lhs)
            and is_scalar(rhs)
            and isinstance(expected, (complex, np.complexfloating))
            and np.isnan(result)
        ):
            msg = "(DataFrame.columns|numpy array) are different"
            with pytest.raises(AssertionError, match=msg):
                tm.assert_numpy_array_equal(result, expected)
        else:
            tm.assert_almost_equal(result, expected)

            ex = "(lhs ** rhs) ** rhs"
            result = pd.eval(ex, engine=engine, parser=parser)

            middle = _eval_single_bin(lhs, "**", rhs, engine)
            expected = _eval_single_bin(middle, "**", rhs, engine)
            tm.assert_almost_equal(result, expected)

    def test_check_single_invert_op(self, lhs, engine, parser):
        # simple
        try:
            elb = lhs.astype(bool)
        except AttributeError:
            elb = np.array([bool(lhs)])
        expected = ~elb
        result = pd.eval("~elb", engine=engine, parser=parser)
        tm.assert_almost_equal(expected, result)

    def test_frame_invert(self, engine, parser):
        expr = "~lhs"

        # ~ ##
        # frame
        # float always raises
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)))
        if engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert_dd'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "ufunc 'invert' not supported for the input types"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)

        # int raises on numexpr
        lhs = DataFrame(np.random.default_rng(2).integers(5, size=(5, 2)))
        if engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = ~lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_frame_equal(expect, result)

        # bool always works
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5)
        expect = ~lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)

        # object raises
        lhs = DataFrame(
            {"b": ["a", 1, 2.0], "c": np.random.default_rng(2).standard_normal(3) > 0.5}
        )
        if engine == "numexpr":
            with pytest.raises(ValueError, match="unknown type object"):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "bad operand type for unary ~: 'str'"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)

    def test_series_invert(self, engine, parser):
        # ~ ####
        expr = "~lhs"

        # series
        # float raises
        lhs = Series(np.random.default_rng(2).standard_normal(5))
        if engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert_dd'"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "ufunc 'invert' not supported for the input types"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)

        # int raises on numexpr
        lhs = Series(np.random.default_rng(2).integers(5, size=5))
        if engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = ~lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_series_equal(expect, result)

        # bool
        lhs = Series(np.random.default_rng(2).standard_normal(5) > 0.5)
        expect = ~lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)

        # float
        # int
        # bool

        # object
        lhs = Series(["a", 1, 2.0])
        if engine == "numexpr":
            with pytest.raises(ValueError, match="unknown type object"):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            msg = "bad operand type for unary ~: 'str'"
            with pytest.raises(TypeError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)

    def test_frame_negate(self, engine, parser):
        expr = "-lhs"

        # float
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)

        # int
        lhs = DataFrame(np.random.default_rng(2).integers(5, size=(5, 2)))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)

        # bool doesn't work with numexpr but works elsewhere
        lhs = DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5)
        if engine == "numexpr":
            msg = "couldn't find matching opcode for 'neg_bb'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = -lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_frame_equal(expect, result)

    def test_series_negate(self, engine, parser):
        expr = "-lhs"

        # float
        lhs = Series(np.random.default_rng(2).standard_normal(5))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)

        # int
        lhs = Series(np.random.default_rng(2).integers(5, size=5))
        expect = -lhs
        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)

        # bool doesn't work with numexpr but works elsewhere
        lhs = Series(np.random.default_rng(2).standard_normal(5) > 0.5)
        if engine == "numexpr":
            msg = "couldn't find matching opcode for 'neg_bb'"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(expr, engine=engine, parser=parser)
        else:
            expect = -lhs
            result = pd.eval(expr, engine=engine, parser=parser)
            tm.assert_series_equal(expect, result)

    @pytest.mark.parametrize(
        "lhs",
        [
            # Float
            DataFrame(np.random.default_rng(2).standard_normal((5, 2))),
            # Int
            DataFrame(np.random.default_rng(2).integers(5, size=(5, 2))),
            # bool doesn't work with numexpr but works elsewhere
            DataFrame(np.random.default_rng(2).standard_normal((5, 2)) > 0.5),
        ],
    )
    def test_frame_pos(self, lhs, engine, parser):
        expr = "+lhs"
        expect = lhs

        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_frame_equal(expect, result)

    @pytest.mark.parametrize(
        "lhs",
        [
            # Float
            Series(np.random.default_rng(2).standard_normal(5)),
            # Int
            Series(np.random.default_rng(2).integers(5, size=5)),
            # bool doesn't work with numexpr but works elsewhere
            Series(np.random.default_rng(2).standard_normal(5) > 0.5),
        ],
    )
    def test_series_pos(self, lhs, engine, parser):
        expr = "+lhs"
        expect = lhs

        result = pd.eval(expr, engine=engine, parser=parser)
        tm.assert_series_equal(expect, result)

    def test_scalar_unary(self, engine, parser):
        msg = "bad operand type for unary ~: 'float'"
        warn = None
        if PY312 and not (engine == "numexpr" and parser == "pandas"):
            warn = DeprecationWarning
        with pytest.raises(TypeError, match=msg):
            pd.eval("~1.0", engine=engine, parser=parser)

        assert pd.eval("-1.0", parser=parser, engine=engine) == -1.0
        assert pd.eval("+1.0", parser=parser, engine=engine) == +1.0
        assert pd.eval("~1", parser=parser, engine=engine) == ~1
        assert pd.eval("-1", parser=parser, engine=engine) == -1
        assert pd.eval("+1", parser=parser, engine=engine) == +1
        with tm.assert_produces_warning(
            warn, match="Bitwise inversion", check_stacklevel=False
        ):
            assert pd.eval("~True", parser=parser, engine=engine) == ~True
        with tm.assert_produces_warning(
            warn, match="Bitwise inversion", check_stacklevel=False
        ):
            assert pd.eval("~False", parser=parser, engine=engine) == ~False
        assert pd.eval("-True", parser=parser, engine=engine) == -True
        assert pd.eval("-False", parser=parser, engine=engine) == -False
        assert pd.eval("+True", parser=parser, engine=engine) == +True
        assert pd.eval("+False", parser=parser, engine=engine) == +False

    def test_unary_in_array(self):
        # GH 11235
        # TODO: 2022-01-29: result return list with numexpr 2.7.3 in CI
        # but cannot reproduce locally
        result = np.array(
            pd.eval("[-True, True, +True, -False, False, +False, -37, 37, ~37, +37]"),
            dtype=np.object_,
        )
        expected = np.array(
            [
                -True,
                True,
                +True,
                -False,
                False,
                +False,
                -37,
                37,
                ~37,
                +37,
            ],
            dtype=np.object_,
        )
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("expr", ["x < -0.1", "-5 > x"])
    def test_float_comparison_bin_op(self, float_numpy_dtype, expr):
        # GH 16363
        df = DataFrame({"x": np.array([0], dtype=float_numpy_dtype)})
        res = df.eval(expr)
        assert res.values == np.array([False])

    def test_unary_in_function(self):
        # GH 46471
        df = DataFrame({"x": [0, 1, np.nan]})

        result = df.eval("x.fillna(-1)")
        expected = df.x.fillna(-1)
        # column name becomes None if using numexpr
        # only check names when the engine is not numexpr
        tm.assert_series_equal(result, expected, check_names=not USE_NUMEXPR)

        result = df.eval("x.shift(1, fill_value=-1)")
        expected = df.x.shift(1, fill_value=-1)
        tm.assert_series_equal(result, expected, check_names=not USE_NUMEXPR)

    @pytest.mark.parametrize(
        "ex",
        (
            "1 or 2",
            "1 and 2",
            "a and b",
            "a or b",
            "1 or 2 and (3 + 2) > 3",
            "2 * x > 2 or 1 and 2",
            "2 * df > 3 and 1 or a",
        ),
    )
    def test_disallow_scalar_bool_ops(self, ex, engine, parser):
        x, a, b = np.random.default_rng(2).standard_normal(3), 1, 2  # noqa: F841
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))  # noqa: F841

        msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)

    def test_identical(self, engine, parser):
        # see gh-10546
        x = 1
        result = pd.eval("x", engine=engine, parser=parser)
        assert result == 1
        assert is_scalar(result)

        x = 1.5
        result = pd.eval("x", engine=engine, parser=parser)
        assert result == 1.5
        assert is_scalar(result)

        x = False
        result = pd.eval("x", engine=engine, parser=parser)
        assert not result
        assert is_bool(result)
        assert is_scalar(result)

        x = np.array([1])
        result = pd.eval("x", engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([1]))
        assert result.shape == (1,)

        x = np.array([1.5])
        result = pd.eval("x", engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([1.5]))
        assert result.shape == (1,)

        x = np.array([False])  # noqa: F841
        result = pd.eval("x", engine=engine, parser=parser)
        tm.assert_numpy_array_equal(result, np.array([False]))
        assert result.shape == (1,)

    def test_line_continuation(self, engine, parser):
        # GH 11149
        exp = """1 + 2 * \
        5 - 1 + 2 """
        result = pd.eval(exp, engine=engine, parser=parser)
        assert result == 12

    def test_float_truncation(self, engine, parser):
        # GH 14241
        exp = "1000000000.006"
        result = pd.eval(exp, engine=engine, parser=parser)
        expected = np.float64(exp)
        assert result == expected

        df = DataFrame({"A": [1000000000.0009, 1000000000.0011, 1000000000.0015]})
        cutoff = 1000000000.0006
        result = df.query(f"A < {cutoff:.4f}")
        assert result.empty

        cutoff = 1000000000.0010
        result = df.query(f"A > {cutoff:.4f}")
        expected = df.loc[[1, 2], :]
        tm.assert_frame_equal(expected, result)

        exact = 1000000000.0011
        result = df.query(f"A == {exact:.4f}")
        expected = df.loc[[1], :]
        tm.assert_frame_equal(expected, result)

    def test_disallow_python_keywords(self):
        # GH 18221
        df = DataFrame([[0, 0, 0]], columns=["foo", "bar", "class"])
        msg = "Python keyword not valid identifier in numexpr query"
        with pytest.raises(SyntaxError, match=msg):
            df.query("class == 0")

        df = DataFrame()
        df.index.name = "lambda"
        with pytest.raises(SyntaxError, match=msg):
            df.query("lambda == 0")

    def test_true_false_logic(self):
        # GH 25823
        # This behavior is deprecated in Python 3.12
        with tm.maybe_produces_warning(
            DeprecationWarning, PY312, check_stacklevel=False
        ):
            assert pd.eval("not True") == -2
            assert pd.eval("not False") == -1
            assert pd.eval("True and not True") == 0

    def test_and_logic_string_match(self):
        # GH 25823
        event = Series({"a": "hello"})
        assert pd.eval(f"{event.str.match('hello').a}")
        assert pd.eval(f"{event.str.match('hello').a and event.str.match('hello').a}")


# -------------------------------------
# gh-12388: Typecasting rules consistency with python


class TestTypeCasting:
    @pytest.mark.parametrize("op", ["+", "-", "*", "**", "/"])
    # maybe someday... numexpr has too many upcasting rules now
    # chain(*(np.core.sctypes[x] for x in ['uint', 'int', 'float']))
    @pytest.mark.parametrize("left_right", [("df", "3"), ("3", "df")])
    def test_binop_typecasting(
        self, engine, parser, op, complex_or_float_dtype, left_right, request
    ):
        # GH#21374
        dtype = complex_or_float_dtype
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), dtype=dtype)
        left, right = left_right
        s = f"{left} {op} {right}"
        res = pd.eval(s, engine=engine, parser=parser)
        if dtype == "complex64" and engine == "numexpr":
            mark = pytest.mark.xfail(
                reason="numexpr issue with complex that are upcast "
                "to complex 128 "
                "https://github.com/pydata/numexpr/issues/492"
            )
            request.applymarker(mark)
        assert df.values.dtype == dtype
        assert res.values.dtype == dtype
        tm.assert_frame_equal(res, eval(s), check_exact=False)


# -------------------------------------
# Basic and complex alignment


def should_warn(*args):
    not_mono = not any(map(operator.attrgetter("is_monotonic_increasing"), args))
    only_one_dt = reduce(
        operator.xor, (issubclass(x.dtype.type, np.datetime64) for x in args)
    )
    return not_mono and only_one_dt


class TestAlignment:
    index_types = ["i", "s", "dt"]
    lhs_index_types = index_types + ["s"]  # 'p'

    def test_align_nested_unary_op(self, engine, parser):
        s = "df * ~2"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        res = pd.eval(s, engine=engine, parser=parser)
        tm.assert_frame_equal(res, df * ~2)

    @pytest.mark.filterwarnings("always::RuntimeWarning")
    @pytest.mark.parametrize("lr_idx_type", lhs_index_types)
    @pytest.mark.parametrize("rr_idx_type", index_types)
    @pytest.mark.parametrize("c_idx_type", index_types)
    def test_basic_frame_alignment(
        self, engine, parser, lr_idx_type, rr_idx_type, c_idx_type, idx_func_dict
    ):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[lr_idx_type](10),
            columns=idx_func_dict[c_idx_type](10),
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((20, 10)),
            index=idx_func_dict[rr_idx_type](20),
            columns=idx_func_dict[c_idx_type](10),
        )
        # only warns if not monotonic and not sortable
        if should_warn(df.index, df2.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("df + df2", engine=engine, parser=parser)
        else:
            res = pd.eval("df + df2", engine=engine, parser=parser)
        tm.assert_frame_equal(res, df + df2)

    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    @pytest.mark.parametrize("c_idx_type", lhs_index_types)
    def test_frame_comparison(
        self, engine, parser, r_idx_type, c_idx_type, idx_func_dict
    ):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](10),
        )
        res = pd.eval("df < 2", engine=engine, parser=parser)
        tm.assert_frame_equal(res, df < 2)

        df3 = DataFrame(
            np.random.default_rng(2).standard_normal(df.shape),
            index=df.index,
            columns=df.columns,
        )
        res = pd.eval("df < df3", engine=engine, parser=parser)
        tm.assert_frame_equal(res, df < df3)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("r1", lhs_index_types)
    @pytest.mark.parametrize("c1", index_types)
    @pytest.mark.parametrize("r2", index_types)
    @pytest.mark.parametrize("c2", index_types)
    def test_medium_complex_frame_alignment(
        self, engine, parser, r1, c1, r2, c2, idx_func_dict
    ):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 2)),
            index=idx_func_dict[r1](3),
            columns=idx_func_dict[c1](2),
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            index=idx_func_dict[r2](4),
            columns=idx_func_dict[c2](2),
        )
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=idx_func_dict[r2](5),
            columns=idx_func_dict[c2](2),
        )
        if should_warn(df.index, df2.index, df3.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("df + df2 + df3", engine=engine, parser=parser)
        else:
            res = pd.eval("df + df2 + df3", engine=engine, parser=parser)
        tm.assert_frame_equal(res, df + df2 + df3)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("index_name", ["index", "columns"])
    @pytest.mark.parametrize("c_idx_type", index_types)
    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    def test_basic_frame_series_alignment(
        self, engine, parser, index_name, r_idx_type, c_idx_type, idx_func_dict
    ):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](10),
        )
        index = getattr(df, index_name)
        s = Series(np.random.default_rng(2).standard_normal(5), index[:5])

        if should_warn(df.index, s.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("df + s", engine=engine, parser=parser)
        else:
            res = pd.eval("df + s", engine=engine, parser=parser)

        if r_idx_type == "dt" or c_idx_type == "dt":
            expected = df.add(s) if engine == "numexpr" else df + s
        else:
            expected = df + s
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize("index_name", ["index", "columns"])
    @pytest.mark.parametrize(
        "r_idx_type, c_idx_type",
        list(product(["i", "s"], ["i", "s"])) + [("dt", "dt")],
    )
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_basic_series_frame_alignment(
        self, request, engine, parser, index_name, r_idx_type, c_idx_type, idx_func_dict
    ):
        if (
            engine == "numexpr"
            and parser in ("pandas", "python")
            and index_name == "index"
            and r_idx_type == "i"
            and c_idx_type == "s"
        ):
            reason = (
                f"Flaky column ordering when engine={engine}, "
                f"parser={parser}, index_name={index_name}, "
                f"r_idx_type={r_idx_type}, c_idx_type={c_idx_type}"
            )
            request.applymarker(pytest.mark.xfail(reason=reason, strict=False))
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 7)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](7),
        )
        index = getattr(df, index_name)
        s = Series(np.random.default_rng(2).standard_normal(5), index[:5])
        if should_warn(s.index, df.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("s + df", engine=engine, parser=parser)
        else:
            res = pd.eval("s + df", engine=engine, parser=parser)

        if r_idx_type == "dt" or c_idx_type == "dt":
            expected = df.add(s) if engine == "numexpr" else s + df
        else:
            expected = s + df
        tm.assert_frame_equal(res, expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("c_idx_type", index_types)
    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    @pytest.mark.parametrize("index_name", ["index", "columns"])
    @pytest.mark.parametrize("op", ["+", "*"])
    def test_series_frame_commutativity(
        self, engine, parser, index_name, op, r_idx_type, c_idx_type, idx_func_dict
    ):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 10)),
            index=idx_func_dict[r_idx_type](10),
            columns=idx_func_dict[c_idx_type](10),
        )
        index = getattr(df, index_name)
        s = Series(np.random.default_rng(2).standard_normal(5), index[:5])

        lhs = f"s {op} df"
        rhs = f"df {op} s"
        if should_warn(df.index, s.index):
            with tm.assert_produces_warning(RuntimeWarning):
                a = pd.eval(lhs, engine=engine, parser=parser)
            with tm.assert_produces_warning(RuntimeWarning):
                b = pd.eval(rhs, engine=engine, parser=parser)
        else:
            a = pd.eval(lhs, engine=engine, parser=parser)
            b = pd.eval(rhs, engine=engine, parser=parser)

        if r_idx_type != "dt" and c_idx_type != "dt":
            if engine == "numexpr":
                tm.assert_frame_equal(a, b)

    @pytest.mark.filterwarnings("always::RuntimeWarning")
    @pytest.mark.parametrize("r1", lhs_index_types)
    @pytest.mark.parametrize("c1", index_types)
    @pytest.mark.parametrize("r2", index_types)
    @pytest.mark.parametrize("c2", index_types)
    def test_complex_series_frame_alignment(
        self, engine, parser, r1, c1, r2, c2, idx_func_dict
    ):
        n = 3
        m1 = 5
        m2 = 2 * m1
        df = DataFrame(
            np.random.default_rng(2).standard_normal((m1, n)),
            index=idx_func_dict[r1](m1),
            columns=idx_func_dict[c1](n),
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((m2, n)),
            index=idx_func_dict[r2](m2),
            columns=idx_func_dict[c2](n),
        )
        index = df2.columns
        ser = Series(np.random.default_rng(2).standard_normal(n), index[:n])

        if r2 == "dt" or c2 == "dt":
            if engine == "numexpr":
                expected2 = df2.add(ser)
            else:
                expected2 = df2 + ser
        else:
            expected2 = df2 + ser

        if r1 == "dt" or c1 == "dt":
            if engine == "numexpr":
                expected = expected2.add(df)
            else:
                expected = expected2 + df
        else:
            expected = expected2 + df

        if should_warn(df2.index, ser.index, df.index):
            with tm.assert_produces_warning(RuntimeWarning):
                res = pd.eval("df2 + ser + df", engine=engine, parser=parser)
        else:
            res = pd.eval("df2 + ser + df", engine=engine, parser=parser)
        assert res.shape == expected.shape
        tm.assert_frame_equal(res, expected)

    def test_performance_warning_for_poor_alignment(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((1000, 10)))
        s = Series(np.random.default_rng(2).standard_normal(10000))
        if engine == "numexpr":
            seen = PerformanceWarning
        else:
            seen = False

        with tm.assert_produces_warning(seen):
            pd.eval("df + s", engine=engine, parser=parser)

        s = Series(np.random.default_rng(2).standard_normal(1000))
        with tm.assert_produces_warning(False):
            pd.eval("df + s", engine=engine, parser=parser)

        df = DataFrame(np.random.default_rng(2).standard_normal((10, 10000)))
        s = Series(np.random.default_rng(2).standard_normal(10000))
        with tm.assert_produces_warning(False):
            pd.eval("df + s", engine=engine, parser=parser)

        df = DataFrame(np.random.default_rng(2).standard_normal((10, 10)))
        s = Series(np.random.default_rng(2).standard_normal(10000))

        is_python_engine = engine == "python"

        if not is_python_engine:
            wrn = PerformanceWarning
        else:
            wrn = False

        with tm.assert_produces_warning(wrn) as w:
            pd.eval("df + s", engine=engine, parser=parser)

            if not is_python_engine:
                assert len(w) == 1
                msg = str(w[0].message)
                logged = np.log10(s.size - df.shape[1])
                expected = (
                    f"Alignment difference on axis 1 is larger "
                    f"than an order of magnitude on term 'df', "
                    f"by more than {logged:.4g}; performance may suffer."
                )
                assert msg == expected


# ------------------------------------
# Slightly more complex ops


class TestOperations:
    def eval(self, *args, **kwargs):
        kwargs["level"] = kwargs.pop("level", 0) + 1
        return pd.eval(*args, **kwargs)

    def test_simple_arith_ops(self, engine, parser):
        exclude_arith = []
        if parser == "python":
            exclude_arith = ["in", "not in"]

        arith_ops = [
            op
            for op in expr.ARITH_OPS_SYMS + expr.CMP_OPS_SYMS
            if op not in exclude_arith
        ]

        ops = (op for op in arith_ops if op != "//")

        for op in ops:
            ex = f"1 {op} 1"
            ex2 = f"x {op} 1"
            ex3 = f"1 {op} (x + 1)"

            if op in ("in", "not in"):
                msg = "argument of type 'int' is not iterable"
                with pytest.raises(TypeError, match=msg):
                    pd.eval(ex, engine=engine, parser=parser)
            else:
                expec = _eval_single_bin(1, op, 1, engine)
                x = self.eval(ex, engine=engine, parser=parser)
                assert x == expec

                expec = _eval_single_bin(x, op, 1, engine)
                y = self.eval(ex2, local_dict={"x": x}, engine=engine, parser=parser)
                assert y == expec

                expec = _eval_single_bin(1, op, x + 1, engine)
                y = self.eval(ex3, local_dict={"x": x}, engine=engine, parser=parser)
                assert y == expec

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_simple_bool_ops(self, rhs, lhs, op):
        ex = f"{lhs} {op} {rhs}"

        if parser == "python" and op in ["and", "or"]:
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                self.eval(ex)
            return

        res = self.eval(ex)
        exp = eval(ex)
        assert res == exp

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_bool_ops_with_constants(self, rhs, lhs, op):
        ex = f"{lhs} {op} {rhs}"

        if parser == "python" and op in ["and", "or"]:
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                self.eval(ex)
            return

        res = self.eval(ex)
        exp = eval(ex)
        assert res == exp

    def test_4d_ndarray_fails(self):
        x = np.random.default_rng(2).standard_normal((3, 4, 5, 6))
        y = Series(np.random.default_rng(2).standard_normal(10))
        msg = "N-dimensional objects, where N > 2, are not supported with eval"
        with pytest.raises(NotImplementedError, match=msg):
            self.eval("x + y", local_dict={"x": x, "y": y})

    def test_constant(self):
        x = self.eval("1")
        assert x == 1

    def test_single_variable(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df2 = self.eval("df", local_dict={"df": df})
        tm.assert_frame_equal(df, df2)

    def test_failing_subscript_with_name_error(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))  # noqa: F841
        with pytest.raises(NameError, match="name 'x' is not defined"):
            self.eval("df[x > 2] > 2")

    def test_lhs_expression_subscript(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        result = self.eval("(df + 1)[df > 2]", local_dict={"df": df})
        expected = (df + 1)[df > 2]
        tm.assert_frame_equal(result, expected)

    def test_attr_expression(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=list("abc")
        )
        expr1 = "df.a < df.b"
        expec1 = df.a < df.b
        expr2 = "df.a + df.b + df.c"
        expec2 = df.a + df.b + df.c
        expr3 = "df.a + df.b + df.c[df.b < 0]"
        expec3 = df.a + df.b + df.c[df.b < 0]
        exprs = expr1, expr2, expr3
        expecs = expec1, expec2, expec3
        for e, expec in zip(exprs, expecs):
            tm.assert_series_equal(expec, self.eval(e, local_dict={"df": df}))

    def test_assignment_fails(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=list("abc")
        )
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        expr1 = "df = df2"
        msg = "cannot assign without a target object"
        with pytest.raises(ValueError, match=msg):
            self.eval(expr1, local_dict={"df": df, "df2": df2})

    def test_assignment_column_multiple_raise(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # multiple assignees
        with pytest.raises(SyntaxError, match="invalid syntax"):
            df.eval("d c = a + b")

    def test_assignment_column_invalid_assign(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # invalid assignees
        msg = "left hand side of an assignment must be a single name"
        with pytest.raises(SyntaxError, match=msg):
            df.eval("d,c = a + b")

    def test_assignment_column_invalid_assign_function_call(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        msg = "cannot assign to function call"
        with pytest.raises(SyntaxError, match=msg):
            df.eval('Timestamp("20131001") = a + b')

    def test_assignment_single_assign_existing(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # single assignment - existing variable
        expected = df.copy()
        expected["a"] = expected["a"] + expected["b"]
        df.eval("a = a + b", inplace=True)
        tm.assert_frame_equal(df, expected)

    def test_assignment_single_assign_new(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # single assignment - new variable
        expected = df.copy()
        expected["c"] = expected["a"] + expected["b"]
        df.eval("c = a + b", inplace=True)
        tm.assert_frame_equal(df, expected)

    def test_assignment_single_assign_local_overlap(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        df = df.copy()
        a = 1  # noqa: F841
        df.eval("a = 1 + b", inplace=True)

        expected = df.copy()
        expected["a"] = 1 + expected["b"]
        tm.assert_frame_equal(df, expected)

    def test_assignment_single_assign_name(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )

        a = 1  # noqa: F841
        old_a = df.a.copy()
        df.eval("a = a + b", inplace=True)
        result = old_a + df.b
        tm.assert_series_equal(result, df.a, check_names=False)
        assert result.name is None

    def test_assignment_multiple_raises(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # multiple assignment
        df.eval("c = a + b", inplace=True)
        msg = "can only assign a single expression"
        with pytest.raises(SyntaxError, match=msg):
            df.eval("c = a = b")

    def test_assignment_explicit(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )
        # explicit targets
        self.eval("c = df.a + df.b", local_dict={"df": df}, target=df, inplace=True)
        expected = df.copy()
        expected["c"] = expected["a"] + expected["b"]
        tm.assert_frame_equal(df, expected)

    def test_column_in(self):
        # GH 11235
        df = DataFrame({"a": [11], "b": [-32]})
        result = df.eval("a in [11, -32]")
        expected = Series([True])
        # TODO: 2022-01-29: Name check failed with numexpr 2.7.3 in CI
        # but cannot reproduce locally
        tm.assert_series_equal(result, expected, check_names=False)

    @pytest.mark.xfail(reason="Unknown: Omitted test_ in name prior.")
    def test_assignment_not_inplace(self):
        # see gh-9297
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("ab")
        )

        actual = df.eval("c = a + b", inplace=False)
        assert actual is not None

        expected = df.copy()
        expected["c"] = expected["a"] + expected["b"]
        tm.assert_frame_equal(df, expected)

    def test_multi_line_expression(self, warn_copy_on_write):
        # GH 11149
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()

        expected["c"] = expected["a"] + expected["b"]
        expected["d"] = expected["c"] + expected["b"]
        answer = df.eval(
            """
        c = a + b
        d = c + b""",
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

        expected["a"] = expected["a"] - 1
        expected["e"] = expected["a"] + 2
        answer = df.eval(
            """
        a = a - 1
        e = a + 2""",
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

        # multi-line not valid if not all assignments
        msg = "Multi-line expressions are only valid if all expressions contain"
        with pytest.raises(ValueError, match=msg):
            df.eval(
                """
            a = b + 2
            b - 2""",
                inplace=False,
            )

    def test_multi_line_expression_not_inplace(self):
        # GH 11149
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()

        expected["c"] = expected["a"] + expected["b"]
        expected["d"] = expected["c"] + expected["b"]
        df = df.eval(
            """
        c = a + b
        d = c + b""",
            inplace=False,
        )
        tm.assert_frame_equal(expected, df)

        expected["a"] = expected["a"] - 1
        expected["e"] = expected["a"] + 2
        df = df.eval(
            """
        a = a - 1
        e = a + 2""",
            inplace=False,
        )
        tm.assert_frame_equal(expected, df)

    def test_multi_line_expression_local_variable(self):
        # GH 15342
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()

        local_var = 7
        expected["c"] = expected["a"] * local_var
        expected["d"] = expected["c"] + local_var
        answer = df.eval(
            """
        c = a * @local_var
        d = c + @local_var
        """,
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

    def test_multi_line_expression_callable_local_variable(self):
        # 26426
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        def local_func(a, b):
            return b

        expected = df.copy()
        expected["c"] = expected["a"] * local_func(1, 7)
        expected["d"] = expected["c"] + local_func(1, 7)
        answer = df.eval(
            """
        c = a * @local_func(1, 7)
        d = c + @local_func(1, 7)
        """,
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

    def test_multi_line_expression_callable_local_variable_with_kwargs(self):
        # 26426
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        def local_func(a, b):
            return b

        expected = df.copy()
        expected["c"] = expected["a"] * local_func(b=7, a=1)
        expected["d"] = expected["c"] + local_func(b=7, a=1)
        answer = df.eval(
            """
        c = a * @local_func(b=7, a=1)
        d = c + @local_func(b=7, a=1)
        """,
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

    def test_assignment_in_query(self):
        # GH 8664
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()
        msg = "cannot assign without a target object"
        with pytest.raises(ValueError, match=msg):
            df.query("a = 1")
        tm.assert_frame_equal(df, df_orig)

    def test_query_inplace(self):
        # see gh-11149
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()
        expected = expected[expected["a"] == 2]
        df.query("a == 2", inplace=True)
        tm.assert_frame_equal(expected, df)

        df = {}
        expected = {"a": 3}

        self.eval("a = 1 + 2", target=df, inplace=True)
        tm.assert_dict_equal(df, expected)

    @pytest.mark.parametrize("invalid_target", [1, "cat", [1, 2], np.array([]), (1, 3)])
    def test_cannot_item_assign(self, invalid_target):
        msg = "Cannot assign expression output to target"
        expression = "a = 1 + 2"

        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=invalid_target, inplace=True)

        if hasattr(invalid_target, "copy"):
            with pytest.raises(ValueError, match=msg):
                self.eval(expression, target=invalid_target, inplace=False)

    @pytest.mark.parametrize("invalid_target", [1, "cat", (1, 3)])
    def test_cannot_copy_item(self, invalid_target):
        msg = "Cannot return a copy of the target"
        expression = "a = 1 + 2"

        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=invalid_target, inplace=False)

    @pytest.mark.parametrize("target", [1, "cat", [1, 2], np.array([]), (1, 3), {1: 2}])
    def test_inplace_no_assignment(self, target):
        expression = "1 + 2"

        assert self.eval(expression, target=target, inplace=False) == 3

        msg = "Cannot operate inplace if there is no assignment"
        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=target, inplace=True)

    def test_basic_period_index_boolean_expression(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)),
            columns=period_range("2020-01-01", freq="D", periods=2),
        )
        e = df < 2
        r = self.eval("df < 2", local_dict={"df": df})
        x = df < 2

        tm.assert_frame_equal(r, e)
        tm.assert_frame_equal(x, e)

    def test_basic_period_index_subscript_expression(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)),
            columns=period_range("2020-01-01", freq="D", periods=2),
        )
        r = self.eval("df[df < 2 + 3]", local_dict={"df": df})
        e = df[df < 2 + 3]
        tm.assert_frame_equal(r, e)

    def test_nested_period_index_subscript_expression(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)),
            columns=period_range("2020-01-01", freq="D", periods=2),
        )
        r = self.eval("df[df[df < 2] < 2] + df * 2", local_dict={"df": df})
        e = df[df[df < 2] < 2] + df * 2
        tm.assert_frame_equal(r, e)

    def test_date_boolean(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        res = self.eval(
            "df.dates1 < 20130101",
            local_dict={"df": df},
            engine=engine,
            parser=parser,
        )
        expec = df.dates1 < "20130101"
        tm.assert_series_equal(res, expec, check_names=False)

    def test_simple_in_ops(self, engine, parser):
        if parser != "python":
            res = pd.eval("1 in [1, 2]", engine=engine, parser=parser)
            assert res

            res = pd.eval("2 in (1, 2)", engine=engine, parser=parser)
            assert res

            res = pd.eval("3 in (1, 2)", engine=engine, parser=parser)
            assert not res

            res = pd.eval("3 not in (1, 2)", engine=engine, parser=parser)
            assert res

            res = pd.eval("[3] not in (1, 2)", engine=engine, parser=parser)
            assert res

            res = pd.eval("[3] in ([3], 2)", engine=engine, parser=parser)
            assert res

            res = pd.eval("[[3]] in [[[3]], 2]", engine=engine, parser=parser)
            assert res

            res = pd.eval("(3,) in [(3,), 2]", engine=engine, parser=parser)
            assert res

            res = pd.eval("(3,) not in [(3,), 2]", engine=engine, parser=parser)
            assert not res

            res = pd.eval("[(3,)] in [[(3,)], 2]", engine=engine, parser=parser)
            assert res
        else:
            msg = "'In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("1 in [1, 2]", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("2 in (1, 2)", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("3 in (1, 2)", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("[(3,)] in (1, 2, [(3,)])", engine=engine, parser=parser)
            msg = "'NotIn' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("3 not in (1, 2)", engine=engine, parser=parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("[3] not in (1, 2, [[3]])", engine=engine, parser=parser)

    def test_check_many_exprs(self, engine, parser):
        a = 1  # noqa: F841
        expr = " * ".join("a" * 33)
        expected = 1
        res = pd.eval(expr, engine=engine, parser=parser)
        assert res == expected

    @pytest.mark.parametrize(
        "expr",
        [
            "df > 2 and df > 3",
            "df > 2 or df > 3",
            "not df > 2",
        ],
    )
    def test_fails_and_or_not(self, expr, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        if parser == "python":
            msg = "'BoolOp' nodes are not implemented"
            if "not" in expr:
                msg = "'Not' nodes are not implemented"

            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(
                    expr,
                    local_dict={"df": df},
                    parser=parser,
                    engine=engine,
                )
        else:
            # smoke-test, should not raise
            pd.eval(
                expr,
                local_dict={"df": df},
                parser=parser,
                engine=engine,
            )

    @pytest.mark.parametrize("char", ["|", "&"])
    def test_fails_ampersand_pipe(self, char, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))  # noqa: F841
        ex = f"(df + 2)[df > 1] > 0 {char} (df > 0)"
        if parser == "python":
            msg = "cannot evaluate scalar only bool ops"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, parser=parser, engine=engine)
        else:
            # smoke-test, should not raise
            pd.eval(ex, parser=parser, engine=engine)


class TestMath:
    def eval(self, *args, **kwargs):
        kwargs["level"] = kwargs.pop("level", 0) + 1
        return pd.eval(*args, **kwargs)

    @pytest.mark.skipif(
        not NUMEXPR_INSTALLED, reason="Unary ops only implemented for numexpr"
    )
    @pytest.mark.parametrize("fn", _unary_math_ops)
    def test_unary_functions(self, fn):
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
        a = df.a

        expr = f"{fn}(a)"
        got = self.eval(expr)
        with np.errstate(all="ignore"):
            expect = getattr(np, fn)(a)
        tm.assert_series_equal(got, expect, check_names=False)

    @pytest.mark.parametrize("fn", _binary_math_ops)
    def test_binary_functions(self, fn):
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
            }
        )
        a = df.a
        b = df.b

        expr = f"{fn}(a, b)"
        got = self.eval(expr)
        with np.errstate(all="ignore"):
            expect = getattr(np, fn)(a, b)
        tm.assert_almost_equal(got, expect, check_names=False)

    def test_df_use_case(self, engine, parser):
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
            }
        )
        df.eval(
            "e = arctan2(sin(a), b)",
            engine=engine,
            parser=parser,
            inplace=True,
        )
        got = df.e
        expect = np.arctan2(np.sin(df.a), df.b)
        tm.assert_series_equal(got, expect, check_names=False)

    def test_df_arithmetic_subexpression(self, engine, parser):
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
            }
        )
        df.eval("e = sin(a + b)", engine=engine, parser=parser, inplace=True)
        got = df.e
        expect = np.sin(df.a + df.b)
        tm.assert_series_equal(got, expect, check_names=False)

    @pytest.mark.parametrize(
        "dtype, expect_dtype",
        [
            (np.int32, np.float64),
            (np.int64, np.float64),
            (np.float32, np.float32),
            (np.float64, np.float64),
            pytest.param(np.complex128, np.complex128, marks=td.skip_if_windows),
        ],
    )
    def test_result_types(self, dtype, expect_dtype, engine, parser):
        # xref https://github.com/pandas-dev/pandas/issues/12293
        #  this fails on Windows, apparently a floating point precision issue

        # Did not test complex64 because DataFrame is converting it to
        # complex128. Due to https://github.com/pandas-dev/pandas/issues/10952
        df = DataFrame(
            {"a": np.random.default_rng(2).standard_normal(10).astype(dtype)}
        )
        assert df.a.dtype == dtype
        df.eval("b = sin(a)", engine=engine, parser=parser, inplace=True)
        got = df.b
        expect = np.sin(df.a)
        assert expect.dtype == got.dtype
        assert expect_dtype == got.dtype
        tm.assert_series_equal(got, expect, check_names=False)

    def test_undefined_func(self, engine, parser):
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
        msg = '"mysin" is not a supported function'

        with pytest.raises(ValueError, match=msg):
            df.eval("mysin(a)", engine=engine, parser=parser)

    def test_keyword_arg(self, engine, parser):
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
        msg = 'Function "sin" does not support keyword arguments'

        with pytest.raises(TypeError, match=msg):
            df.eval("sin(x=a)", engine=engine, parser=parser)


_var_s = np.random.default_rng(2).standard_normal(10)


class TestScope:
    def test_global_scope(self, engine, parser):
        e = "_var_s * 2"
        tm.assert_numpy_array_equal(
            _var_s * 2, pd.eval(e, engine=engine, parser=parser)
        )

    def test_no_new_locals(self, engine, parser):
        x = 1
        lcls = locals().copy()
        pd.eval("x + 1", local_dict=lcls, engine=engine, parser=parser)
        lcls2 = locals().copy()
        lcls2.pop("lcls")
        assert lcls == lcls2

    def test_no_new_globals(self, engine, parser):
        x = 1  # noqa: F841
        gbls = globals().copy()
        pd.eval("x + 1", engine=engine, parser=parser)
        gbls2 = globals().copy()
        assert gbls == gbls2

    def test_empty_locals(self, engine, parser):
        # GH 47084
        x = 1  # noqa: F841
        msg = "name 'x' is not defined"
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval("x + 1", engine=engine, parser=parser, local_dict={})

    def test_empty_globals(self, engine, parser):
        # GH 47084
        msg = "name '_var_s' is not defined"
        e = "_var_s * 2"
        with pytest.raises(UndefinedVariableError, match=msg):
            pd.eval(e, engine=engine, parser=parser, global_dict={})


@td.skip_if_no("numexpr")
def test_invalid_engine():
    msg = "Invalid engine 'asdf' passed"
    with pytest.raises(KeyError, match=msg):
        pd.eval("x + y", local_dict={"x": 1, "y": 2}, engine="asdf")


@td.skip_if_no("numexpr")
@pytest.mark.parametrize(
    ("use_numexpr", "expected"),
    (
        (True, "numexpr"),
        (False, "python"),
    ),
)
def test_numexpr_option_respected(use_numexpr, expected):
    # GH 32556
    from pandas.core.computation.eval import _check_engine

    with pd.option_context("compute.use_numexpr", use_numexpr):
        result = _check_engine(None)
        assert result == expected


@td.skip_if_no("numexpr")
def test_numexpr_option_incompatible_op():
    # GH 32556
    with pd.option_context("compute.use_numexpr", False):
        df = DataFrame(
            {"A": [True, False, True, False, None, None], "B": [1, 2, 3, 4, 5, 6]}
        )
        result = df.query("A.isnull()")
        expected = DataFrame({"A": [None, None], "B": [5, 6]}, index=[4, 5])
        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numexpr")
def test_invalid_parser():
    msg = "Invalid parser 'asdf' passed"
    with pytest.raises(KeyError, match=msg):
        pd.eval("x + y", local_dict={"x": 1, "y": 2}, parser="asdf")


_parsers: dict[str, type[BaseExprVisitor]] = {
    "python": PythonExprVisitor,
    "pytables": pytables.PyTablesExprVisitor,
    "pandas": PandasExprVisitor,
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parser", _parsers)
def test_disallowed_nodes(engine, parser):
    VisitorClass = _parsers[parser]
    inst = VisitorClass("x + 1", engine, parser)

    for ops in VisitorClass.unsupported_nodes:
        msg = "nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            getattr(inst, ops)()


def test_syntax_error_exprs(engine, parser):
    e = "s +"
    with pytest.raises(SyntaxError, match="invalid syntax"):
        pd.eval(e, engine=engine, parser=parser)


def test_name_error_exprs(engine, parser):
    e = "s + t"
    msg = "name 's' is not defined"
    with pytest.raises(NameError, match=msg):
        pd.eval(e, engine=engine, parser=parser)


@pytest.mark.parametrize("express", ["a + @b", "@a + b", "@a + @b"])
def test_invalid_local_variable_reference(engine, parser, express):
    a, b = 1, 2  # noqa: F841

    if parser != "pandas":
        with pytest.raises(SyntaxError, match="The '@' prefix is only"):
            pd.eval(express, engine=engine, parser=parser)
    else:
        with pytest.raises(SyntaxError, match="The '@' prefix is not"):
            pd.eval(express, engine=engine, parser=parser)


def test_numexpr_builtin_raises(engine, parser):
    sin, dotted_line = 1, 2
    if engine == "numexpr":
        msg = "Variables in expression .+"
        with pytest.raises(NumExprClobberingError, match=msg):
            pd.eval("sin + dotted_line", engine=engine, parser=parser)
    else:
        res = pd.eval("sin + dotted_line", engine=engine, parser=parser)
        assert res == sin + dotted_line


def test_bad_resolver_raises(engine, parser):
    cannot_resolve = 42, 3.0
    with pytest.raises(TypeError, match="Resolver of type .+"):
        pd.eval("1 + 2", resolvers=cannot_resolve, engine=engine, parser=parser)


def test_empty_string_raises(engine, parser):
    # GH 13139
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        pd.eval("", engine=engine, parser=parser)


def test_more_than_one_expression_raises(engine, parser):
    with pytest.raises(SyntaxError, match="only a single expression is allowed"):
        pd.eval("1 + 1; 2 + 2", engine=engine, parser=parser)


@pytest.mark.parametrize("cmp", ("and", "or"))
@pytest.mark.parametrize("lhs", (int, float))
@pytest.mark.parametrize("rhs", (int, float))
def test_bool_ops_fails_on_scalars(lhs, cmp, rhs, engine, parser):
    gen = {
        int: lambda: np.random.default_rng(2).integers(10),
        float: np.random.default_rng(2).standard_normal,
    }

    mid = gen[lhs]()  # noqa: F841
    lhs = gen[lhs]()
    rhs = gen[rhs]()

    ex1 = f"lhs {cmp} mid {cmp} rhs"
    ex2 = f"lhs {cmp} mid and mid {cmp} rhs"
    ex3 = f"(lhs {cmp} mid) & (mid {cmp} rhs)"
    for ex in (ex1, ex2, ex3):
        msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)


@pytest.mark.parametrize(
    "other",
    [
        "'x'",
        "...",
    ],
)
def test_equals_various(other):
    df = DataFrame({"A": ["a", "b", "c"]}, dtype=object)
    result = df.eval(f"A == {other}")
    expected = Series([False, False, False], name="A")
    if USE_NUMEXPR:
        # https://github.com/pandas-dev/pandas/issues/10239
        # lose name with numexpr engine. Remove when that's fixed.
        expected.name = None
    tm.assert_series_equal(result, expected)


def test_inf(engine, parser):
    s = "inf + 1"
    expected = np.inf
    result = pd.eval(s, engine=engine, parser=parser)
    assert result == expected


@pytest.mark.parametrize("column", ["Temp(°C)", "Capacitance(μF)"])
def test_query_token(engine, column):
    # See: https://github.com/pandas-dev/pandas/pull/42826
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 2)), columns=[column, "b"]
    )
    expected = df[df[column] > 5]
    query_string = f"`{column}` > 5"
    result = df.query(query_string, engine=engine)
    tm.assert_frame_equal(result, expected)


def test_negate_lt_eq_le(engine, parser):
    df = DataFrame([[0, 10], [1, 20]], columns=["cat", "count"])
    expected = df[~(df.cat > 0)]

    result = df.query("~(cat > 0)", engine=engine, parser=parser)
    tm.assert_frame_equal(result, expected)

    if parser == "python":
        msg = "'Not' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query("not (cat > 0)", engine=engine, parser=parser)
    else:
        result = df.query("not (cat > 0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "column",
    DEFAULT_GLOBALS.keys(),
)
def test_eval_no_support_column_name(request, column):
    # GH 44603
    if column in ["True", "False", "inf", "Inf"]:
        request.applymarker(
            pytest.mark.xfail(
                raises=KeyError,
                reason=f"GH 47859 DataFrame eval not supported with {column}",
            )
        )

    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, size=(10, 2)),
        columns=[column, "col1"],
    )
    expected = df[df[column] > 6]
    result = df.query(f"{column}>6")

    tm.assert_frame_equal(result, expected)


def test_set_inplace(using_copy_on_write, warn_copy_on_write):
    # https://github.com/pandas-dev/pandas/issues/47449
    # Ensure we don't only update the DataFrame inplace, but also the actual
    # column values, such that references to this column also get updated
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    result_view = df[:]
    ser = df["A"]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.eval("A = B + C", inplace=True)
    expected = DataFrame({"A": [11, 13, 15], "B": [4, 5, 6], "C": [7, 8, 9]})
    tm.assert_frame_equal(df, expected)
    if not using_copy_on_write:
        tm.assert_series_equal(ser, expected["A"])
        tm.assert_series_equal(result_view["A"], expected["A"])
    else:
        expected = Series([1, 2, 3], name="A")
        tm.assert_series_equal(ser, expected)
        tm.assert_series_equal(result_view["A"], expected)


class TestValidate:
    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        msg = 'For argument "inplace" expected type bool, received type'
        with pytest.raises(ValueError, match=msg):
            pd.eval("2+2", inplace=value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\test_extras.py ===
"""Tests suite for MaskedArray.
Adapted from the original test_ma by Pierre Gerard-Marchant

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu

"""
import itertools
import warnings

import pytest

import numpy as np
from numpy._core.numeric import normalize_axis_tuple
from numpy.ma.core import (
    MaskedArray,
    arange,
    array,
    count,
    getmaskarray,
    masked,
    masked_array,
    nomask,
    ones,
    shape,
    zeros,
)
from numpy.ma.extras import (
    _covhelper,
    apply_along_axis,
    apply_over_axes,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    average,
    clump_masked,
    clump_unmasked,
    compress_nd,
    compress_rowcols,
    corrcoef,
    cov,
    diagflat,
    dot,
    ediff1d,
    flatnotmasked_contiguous,
    in1d,
    intersect1d,
    isin,
    mask_rowcols,
    masked_all,
    masked_all_like,
    median,
    mr_,
    ndenumerate,
    notmasked_contiguous,
    notmasked_edges,
    polyfit,
    setdiff1d,
    setxor1d,
    stack,
    union1d,
    unique,
    vstack,
)
from numpy.ma.testutils import (
    assert_,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
from numpy.testing import assert_warns, suppress_warnings


class TestGeneric:
    #
    def test_masked_all(self):
        # Tests masked_all
        # Standard dtype
        test = masked_all((2,), dtype=float)
        control = array([1, 1], mask=[1, 1], dtype=float)
        assert_equal(test, control)
        # Flexible dtype
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
        test = masked_all((2,), dtype=dt)
        control = array([(0, 0), (0, 0)], mask=[(1, 1), (1, 1)], dtype=dt)
        assert_equal(test, control)
        test = masked_all((2, 2), dtype=dt)
        control = array([[(0, 0), (0, 0)], [(0, 0), (0, 0)]],
                        mask=[[(1, 1), (1, 1)], [(1, 1), (1, 1)]],
                        dtype=dt)
        assert_equal(test, control)
        # Nested dtype
        dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
        test = masked_all((2,), dtype=dt)
        control = array([(1, (1, 1)), (1, (1, 1))],
                        mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
        assert_equal(test, control)
        test = masked_all((2,), dtype=dt)
        control = array([(1, (1, 1)), (1, (1, 1))],
                        mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
        assert_equal(test, control)
        test = masked_all((1, 1), dtype=dt)
        control = array([[(1, (1, 1))]], mask=[[(1, (1, 1))]], dtype=dt)
        assert_equal(test, control)

    def test_masked_all_with_object_nested(self):
        # Test masked_all works with nested array with dtype of an 'object'
        # refers to issue #15895
        my_dtype = np.dtype([('b', ([('c', object)], (1,)))])
        masked_arr = np.ma.masked_all((1,), my_dtype)

        assert_equal(type(masked_arr['b']), np.ma.core.MaskedArray)
        assert_equal(type(masked_arr['b']['c']), np.ma.core.MaskedArray)
        assert_equal(len(masked_arr['b']['c']), 1)
        assert_equal(masked_arr['b']['c'].shape, (1, 1))
        assert_equal(masked_arr['b']['c']._fill_value.shape, ())

    def test_masked_all_with_object(self):
        # same as above except that the array is not nested
        my_dtype = np.dtype([('b', (object, (1,)))])
        masked_arr = np.ma.masked_all((1,), my_dtype)

        assert_equal(type(masked_arr['b']), np.ma.core.MaskedArray)
        assert_equal(len(masked_arr['b']), 1)
        assert_equal(masked_arr['b'].shape, (1, 1))
        assert_equal(masked_arr['b']._fill_value.shape, ())

    def test_masked_all_like(self):
        # Tests masked_all
        # Standard dtype
        base = array([1, 2], dtype=float)
        test = masked_all_like(base)
        control = array([1, 1], mask=[1, 1], dtype=float)
        assert_equal(test, control)
        # Flexible dtype
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
        base = array([(0, 0), (0, 0)], mask=[(1, 1), (1, 1)], dtype=dt)
        test = masked_all_like(base)
        control = array([(10, 10), (10, 10)], mask=[(1, 1), (1, 1)], dtype=dt)
        assert_equal(test, control)
        # Nested dtype
        dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
        control = array([(1, (1, 1)), (1, (1, 1))],
                        mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
        test = masked_all_like(control)
        assert_equal(test, control)

    def check_clump(self, f):
        for i in range(1, 7):
            for j in range(2**i):
                k = np.arange(i, dtype=int)
                ja = np.full(i, j, dtype=int)
                a = masked_array(2**k)
                a.mask = (ja & (2**k)) != 0
                s = 0
                for sl in f(a):
                    s += a.data[sl].sum()
                if f == clump_unmasked:
                    assert_equal(a.compressed().sum(), s)
                else:
                    a.mask = ~a.mask
                    assert_equal(a.compressed().sum(), s)

    def test_clump_masked(self):
        # Test clump_masked
        a = masked_array(np.arange(10))
        a[[0, 1, 2, 6, 8, 9]] = masked
        #
        test = clump_masked(a)
        control = [slice(0, 3), slice(6, 7), slice(8, 10)]
        assert_equal(test, control)

        self.check_clump(clump_masked)

    def test_clump_unmasked(self):
        # Test clump_unmasked
        a = masked_array(np.arange(10))
        a[[0, 1, 2, 6, 8, 9]] = masked
        test = clump_unmasked(a)
        control = [slice(3, 6), slice(7, 8), ]
        assert_equal(test, control)

        self.check_clump(clump_unmasked)

    def test_flatnotmasked_contiguous(self):
        # Test flatnotmasked_contiguous
        a = arange(10)
        # No mask
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [slice(0, a.size)])
        # mask of all false
        a.mask = np.zeros(10, dtype=bool)
        assert_equal(test, [slice(0, a.size)])
        # Some mask
        a[(a < 3) | (a > 8) | (a == 5)] = masked
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [slice(3, 5), slice(6, 9)])
        #
        a[:] = masked
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [])


class TestAverage:
    # Several tests of average. Why so many ? Good point...
    def test_testAverage1(self):
        # Test of average.
        ott = array([0., 1., 2., 3.], mask=[True, False, False, False])
        assert_equal(2.0, average(ott, axis=0))
        assert_equal(2.0, average(ott, weights=[1., 1., 2., 1.]))
        result, wts = average(ott, weights=[1., 1., 2., 1.], returned=True)
        assert_equal(2.0, result)
        assert_(wts == 4.0)
        ott[:] = masked
        assert_equal(average(ott, axis=0).mask, [True])
        ott = array([0., 1., 2., 3.], mask=[True, False, False, False])
        ott = ott.reshape(2, 2)
        ott[:, 1] = masked
        assert_equal(average(ott, axis=0), [2.0, 0.0])
        assert_equal(average(ott, axis=1).mask[0], [True])
        assert_equal([2., 0.], average(ott, axis=0))
        result, wts = average(ott, axis=0, returned=True)
        assert_equal(wts, [1., 0.])

    def test_testAverage2(self):
        # More tests of average.
        w1 = [0, 1, 1, 1, 1, 0]
        w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
        x = arange(6, dtype=np.float64)
        assert_equal(average(x, axis=0), 2.5)
        assert_equal(average(x, axis=0, weights=w1), 2.5)
        y = array([arange(6, dtype=np.float64), 2.0 * arange(6)])
        assert_equal(average(y, None), np.add.reduce(np.arange(6)) * 3. / 12.)
        assert_equal(average(y, axis=0), np.arange(6) * 3. / 2.)
        assert_equal(average(y, axis=1),
                     [average(x, axis=0), average(x, axis=0) * 2.0])
        assert_equal(average(y, None, weights=w2), 20. / 6.)
        assert_equal(average(y, axis=0, weights=w2),
                     [0., 1., 2., 3., 4., 10.])
        assert_equal(average(y, axis=1),
                     [average(x, axis=0), average(x, axis=0) * 2.0])
        m1 = zeros(6)
        m2 = [0, 0, 1, 1, 0, 0]
        m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
        m4 = ones(6)
        m5 = [0, 1, 1, 1, 1, 1]
        assert_equal(average(masked_array(x, m1), axis=0), 2.5)
        assert_equal(average(masked_array(x, m2), axis=0), 2.5)
        assert_equal(average(masked_array(x, m4), axis=0).mask, [True])
        assert_equal(average(masked_array(x, m5), axis=0), 0.0)
        assert_equal(count(average(masked_array(x, m4), axis=0)), 0)
        z = masked_array(y, m3)
        assert_equal(average(z, None), 20. / 6.)
        assert_equal(average(z, axis=0), [0., 1., 99., 99., 4.0, 7.5])
        assert_equal(average(z, axis=1), [2.5, 5.0])
        assert_equal(average(z, axis=0, weights=w2),
                     [0., 1., 99., 99., 4.0, 10.0])

    def test_testAverage3(self):
        # Yet more tests of average!
        a = arange(6)
        b = arange(6) * 3
        r1, w1 = average([[a, b], [b, a]], axis=1, returned=True)
        assert_equal(shape(r1), shape(w1))
        assert_equal(r1.shape, w1.shape)
        r2, w2 = average(ones((2, 2, 3)), axis=0, weights=[3, 1], returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), weights=ones((2, 2, 3)), returned=True)
        assert_equal(shape(w2), shape(r2))
        a2d = array([[1, 2], [0, 4]], float)
        a2dm = masked_array(a2d, [[False, False], [True, False]])
        a2da = average(a2d, axis=0)
        assert_equal(a2da, [0.5, 3.0])
        a2dma = average(a2dm, axis=0)
        assert_equal(a2dma, [1.0, 3.0])
        a2dma = average(a2dm, axis=None)
        assert_equal(a2dma, 7. / 3.)
        a2dma = average(a2dm, axis=1)
        assert_equal(a2dma, [1.5, 4.0])

    def test_testAverage4(self):
        # Test that `keepdims` works with average
        x = np.array([2, 3, 4]).reshape(3, 1)
        b = np.ma.array(x, mask=[[False], [False], [True]])
        w = np.array([4, 5, 6]).reshape(3, 1)
        actual = average(b, weights=w, axis=1, keepdims=True)
        desired = masked_array([[2.], [3.], [4.]], [[False], [False], [True]])
        assert_equal(actual, desired)

    def test_weight_and_input_dims_different(self):
        # this test mirrors a test for np.average()
        # in lib/test/test_function_base.py
        y = np.arange(12).reshape(2, 2, 3)
        w = np.array([0., 0., 1., .5, .5, 0., 0., .5, .5, 1., 0., 0.])\
            .reshape(2, 2, 3)

        m = np.full((2, 2, 3), False)
        yma = np.ma.array(y, mask=m)
        subw0 = w[:, :, 0]

        actual = average(yma, axis=(0, 1), weights=subw0)
        desired = masked_array([7., 8., 9.], mask=[False, False, False])
        assert_almost_equal(actual, desired)

        m = np.full((2, 2, 3), False)
        m[:, :, 0] = True
        m[0, 0, 1] = True
        yma = np.ma.array(y, mask=m)
        actual = average(yma, axis=(0, 1), weights=subw0)
        desired = masked_array(
            [np.nan, 8., 9.],
            mask=[True, False, False])
        assert_almost_equal(actual, desired)

        m = np.full((2, 2, 3), False)
        yma = np.ma.array(y, mask=m)

        subw1 = w[1, :, :]
        actual = average(yma, axis=(1, 2), weights=subw1)
        desired = masked_array([2.25, 8.25], mask=[False, False])
        assert_almost_equal(actual, desired)

        # here the weights have the wrong shape for the specified axes
        with pytest.raises(
                ValueError,
                match="Shape of weights must be consistent with "
                      "shape of a along specified axis"):
            average(yma, axis=(0, 1, 2), weights=subw0)

        with pytest.raises(
                ValueError,
                match="Shape of weights must be consistent with "
                      "shape of a along specified axis"):
            average(yma, axis=(0, 1), weights=subw1)

        # swapping the axes should be same as transposing weights
        actual = average(yma, axis=(1, 0), weights=subw0)
        desired = average(yma, axis=(0, 1), weights=subw0.T)
        assert_almost_equal(actual, desired)

    def test_onintegers_with_mask(self):
        # Test average on integers with mask
        a = average(array([1, 2]))
        assert_equal(a, 1.5)
        a = average(array([1, 2, 3, 4], mask=[False, False, True, True]))
        assert_equal(a, 1.5)

    def test_complex(self):
        # Test with complex data.
        # (Regression test for https://github.com/numpy/numpy/issues/2684)
        mask = np.array([[0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0]], dtype=bool)
        a = masked_array([[0, 1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
                          [9j, 0 + 1j, 2 + 3j, 4 + 5j, 7 + 7j]],
                         mask=mask)

        av = average(a)
        expected = np.average(a.compressed())
        assert_almost_equal(av.real, expected.real)
        assert_almost_equal(av.imag, expected.imag)

        av0 = average(a, axis=0)
        expected0 = average(a.real, axis=0) + average(a.imag, axis=0) * 1j
        assert_almost_equal(av0.real, expected0.real)
        assert_almost_equal(av0.imag, expected0.imag)

        av1 = average(a, axis=1)
        expected1 = average(a.real, axis=1) + average(a.imag, axis=1) * 1j
        assert_almost_equal(av1.real, expected1.real)
        assert_almost_equal(av1.imag, expected1.imag)

        # Test with the 'weights' argument.
        wts = np.array([[0.5, 1.0, 2.0, 1.0, 0.5],
                        [1.0, 1.0, 1.0, 1.0, 1.0]])
        wav = average(a, weights=wts)
        expected = np.average(a.compressed(), weights=wts[~mask])
        assert_almost_equal(wav.real, expected.real)
        assert_almost_equal(wav.imag, expected.imag)

        wav0 = average(a, weights=wts, axis=0)
        expected0 = (average(a.real, weights=wts, axis=0) +
                     average(a.imag, weights=wts, axis=0) * 1j)
        assert_almost_equal(wav0.real, expected0.real)
        assert_almost_equal(wav0.imag, expected0.imag)

        wav1 = average(a, weights=wts, axis=1)
        expected1 = (average(a.real, weights=wts, axis=1) +
                     average(a.imag, weights=wts, axis=1) * 1j)
        assert_almost_equal(wav1.real, expected1.real)
        assert_almost_equal(wav1.imag, expected1.imag)

    @pytest.mark.parametrize(
        'x, axis, expected_avg, weights, expected_wavg, expected_wsum',
        [([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]),
         ([[1, 2, 5], [1, 6, 11]], 0, [[1.0, 4.0, 8.0]],
          [1, 3], [[1.0, 5.0, 9.5]], [[4, 4, 4]])],
    )
    def test_basic_keepdims(self, x, axis, expected_avg,
                            weights, expected_wavg, expected_wsum):
        avg = np.ma.average(x, axis=axis, keepdims=True)
        assert avg.shape == np.shape(expected_avg)
        assert_array_equal(avg, expected_avg)

        wavg = np.ma.average(x, axis=axis, weights=weights, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)

        wavg, wsum = np.ma.average(x, axis=axis, weights=weights,
                                   returned=True, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        assert wsum.shape == np.shape(expected_wsum)
        assert_array_equal(wsum, expected_wsum)

    def test_masked_weights(self):
        # Test with masked weights.
        # (Regression test for https://github.com/numpy/numpy/issues/10438)
        a = np.ma.array(np.arange(9).reshape(3, 3),
                        mask=[[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        weights_unmasked = masked_array([5, 28, 31], mask=False)
        weights_masked = masked_array([5, 28, 31], mask=[1, 0, 0])

        avg_unmasked = average(a, axis=0,
                               weights=weights_unmasked, returned=False)
        expected_unmasked = np.array([6.0, 5.21875, 6.21875])
        assert_almost_equal(avg_unmasked, expected_unmasked)

        avg_masked = average(a, axis=0, weights=weights_masked, returned=False)
        expected_masked = np.array([6.0, 5.576271186440678, 6.576271186440678])
        assert_almost_equal(avg_masked, expected_masked)

        # weights should be masked if needed
        # depending on the array mask. This is to avoid summing
        # masked nan or other values that are not cancelled by a zero
        a = np.ma.array([1.0,   2.0,   3.0,  4.0],
                   mask=[False, False, True, True])
        avg_unmasked = average(a, weights=[1, 1, 1, np.nan])

        assert_almost_equal(avg_unmasked, 1.5)

        a = np.ma.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 1.0, 2.0, 3.0],
        ], mask=[
            [False, True, True, False],
            [True, False, True, True],
            [True, False, True, False],
        ])

        avg_masked = np.ma.average(a, weights=[1, np.nan, 1], axis=0)
        avg_expected = np.ma.array([1.0, np.nan, np.nan, 3.5],
                              mask=[False, True, True, False])

        assert_almost_equal(avg_masked, avg_expected)
        assert_equal(avg_masked.mask, avg_expected.mask)


class TestConcatenator:
    # Tests for mr_, the equivalent of r_ for masked arrays.

    def test_1d(self):
        # Tests mr_ on 1D arrays.
        assert_array_equal(mr_[1, 2, 3, 4, 5, 6], array([1, 2, 3, 4, 5, 6]))
        b = ones(5)
        m = [1, 0, 0, 0, 0]
        d = masked_array(b, mask=m)
        c = mr_[d, 0, 0, d]
        assert_(isinstance(c, MaskedArray))
        assert_array_equal(c, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        assert_array_equal(c.mask, mr_[m, 0, 0, m])

    def test_2d(self):
        # Tests mr_ on 2D arrays.
        a_1 = np.random.rand(5, 5)
        a_2 = np.random.rand(5, 5)
        m_1 = np.round(np.random.rand(5, 5), 0)
        m_2 = np.round(np.random.rand(5, 5), 0)
        b_1 = masked_array(a_1, mask=m_1)
        b_2 = masked_array(a_2, mask=m_2)
        # append columns
        d = mr_['1', b_1, b_2]
        assert_(d.shape == (5, 10))
        assert_array_equal(d[:, :5], b_1)
        assert_array_equal(d[:, 5:], b_2)
        assert_array_equal(d.mask, np.r_['1', m_1, m_2])
        d = mr_[b_1, b_2]
        assert_(d.shape == (10, 5))
        assert_array_equal(d[:5, :], b_1)
        assert_array_equal(d[5:, :], b_2)
        assert_array_equal(d.mask, np.r_[m_1, m_2])

    def test_masked_constant(self):
        actual = mr_[np.ma.masked, 1]
        assert_equal(actual.mask, [True, False])
        assert_equal(actual.data[1], 1)

        actual = mr_[[1, 2], np.ma.masked]
        assert_equal(actual.mask, [False, False, True])
        assert_equal(actual.data[:2], [1, 2])


class TestNotMasked:
    # Tests notmasked_edges and notmasked_contiguous.

    def test_edges(self):
        # Tests unmasked_edges
        data = masked_array(np.arange(25).reshape(5, 5),
                            mask=[[0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 1],
                                  [1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [1, 1, 1, 0, 0]],)
        test = notmasked_edges(data, None)
        assert_equal(test, [0, 24])
        test = notmasked_edges(data, 0)
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(3, 3, 3, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data, 1)
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 2, 0, 3)])
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 2, 4, 4, 4)])
        #
        test = notmasked_edges(data.data, None)
        assert_equal(test, [0, 24])
        test = notmasked_edges(data.data, 0)
        assert_equal(test[0], [(0, 0, 0, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(4, 4, 4, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data.data, -1)
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 0, 0, 0)])
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 4, 4, 4, 4)])
        #
        data[-2] = masked
        test = notmasked_edges(data, 0)
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(1, 1, 2, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data, -1)
        assert_equal(test[0], [(0, 1, 2, 4), (0, 0, 2, 3)])
        assert_equal(test[1], [(0, 1, 2, 4), (4, 2, 4, 4)])

    def test_contiguous(self):
        # Tests notmasked_contiguous
        a = masked_array(np.arange(24).reshape(3, 8),
                         mask=[[0, 0, 0, 0, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0, 0, 1, 0]])
        tmp = notmasked_contiguous(a, None)
        assert_equal(tmp, [
            slice(0, 4, None),
            slice(16, 22, None),
            slice(23, 24, None)
        ])

        tmp = notmasked_contiguous(a, 0)
        assert_equal(tmp, [
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(2, 3, None)],
            [slice(2, 3, None)],
            [],
            [slice(2, 3, None)]
        ])
        #
        tmp = notmasked_contiguous(a, 1)
        assert_equal(tmp, [
            [slice(0, 4, None)],
            [],
            [slice(0, 6, None), slice(7, 8, None)]
        ])


class TestCompressFunctions:

    def test_compress_nd(self):
        # Tests compress_nd
        x = np.array(list(range(3 * 4 * 5))).reshape(3, 4, 5)
        m = np.zeros((3, 4, 5)).astype(bool)
        m[1, 1, 1] = True
        x = array(x, mask=m)

        # axis=None
        a = compress_nd(x)
        assert_equal(a, [[[ 0,  2,  3,  4],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[40, 42, 43, 44],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        # axis=0
        a = compress_nd(x, 0)
        assert_equal(a, [[[ 0,  1,  2,  3,  4],
                          [ 5,  6,  7,  8,  9],
                          [10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19]],
                         [[40, 41, 42, 43, 44],
                          [45, 46, 47, 48, 49],
                          [50, 51, 52, 53, 54],
                          [55, 56, 57, 58, 59]]])

        # axis=1
        a = compress_nd(x, 1)
        assert_equal(a, [[[ 0,  1,  2,  3,  4],
                          [10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19]],
                         [[20, 21, 22, 23, 24],
                          [30, 31, 32, 33, 34],
                          [35, 36, 37, 38, 39]],
                         [[40, 41, 42, 43, 44],
                          [50, 51, 52, 53, 54],
                          [55, 56, 57, 58, 59]]])

        a2 = compress_nd(x, (1,))
        a3 = compress_nd(x, -2)
        a4 = compress_nd(x, (-2,))
        assert_equal(a, a2)
        assert_equal(a, a3)
        assert_equal(a, a4)

        # axis=2
        a = compress_nd(x, 2)
        assert_equal(a, [[[ 0, 2,  3,  4],
                          [ 5, 7,  8,  9],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[20, 22, 23, 24],
                          [25, 27, 28, 29],
                          [30, 32, 33, 34],
                          [35, 37, 38, 39]],
                         [[40, 42, 43, 44],
                          [45, 47, 48, 49],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        a2 = compress_nd(x, (2,))
        a3 = compress_nd(x, -1)
        a4 = compress_nd(x, (-1,))
        assert_equal(a, a2)
        assert_equal(a, a3)
        assert_equal(a, a4)

        # axis=(0, 1)
        a = compress_nd(x, (0, 1))
        assert_equal(a, [[[ 0,  1,  2,  3,  4],
                          [10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19]],
                         [[40, 41, 42, 43, 44],
                          [50, 51, 52, 53, 54],
                          [55, 56, 57, 58, 59]]])
        a2 = compress_nd(x, (0, -2))
        assert_equal(a, a2)

        # axis=(1, 2)
        a = compress_nd(x, (1, 2))
        assert_equal(a, [[[ 0,  2,  3,  4],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[20, 22, 23, 24],
                          [30, 32, 33, 34],
                          [35, 37, 38, 39]],
                         [[40, 42, 43, 44],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        a2 = compress_nd(x, (-2, 2))
        a3 = compress_nd(x, (1, -1))
        a4 = compress_nd(x, (-2, -1))
        assert_equal(a, a2)
        assert_equal(a, a3)
        assert_equal(a, a4)

        # axis=(0, 2)
        a = compress_nd(x, (0, 2))
        assert_equal(a, [[[ 0,  2,  3,  4],
                          [ 5,  7,  8,  9],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[40, 42, 43, 44],
                          [45, 47, 48, 49],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        a2 = compress_nd(x, (0, -1))
        assert_equal(a, a2)

    def test_compress_rowcols(self):
        # Tests compress_rowcols
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[4, 5], [7, 8]])
        assert_equal(compress_rowcols(x, 0), [[3, 4, 5], [6, 7, 8]])
        assert_equal(compress_rowcols(x, 1), [[1, 2], [4, 5], [7, 8]])
        x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[0, 2], [6, 8]])
        assert_equal(compress_rowcols(x, 0), [[0, 1, 2], [6, 7, 8]])
        assert_equal(compress_rowcols(x, 1), [[0, 2], [3, 5], [6, 8]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[8]])
        assert_equal(compress_rowcols(x, 0), [[6, 7, 8]])
        assert_equal(compress_rowcols(x, 1,), [[2], [5], [8]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert_equal(compress_rowcols(x).size, 0)
        assert_equal(compress_rowcols(x, 0).size, 0)
        assert_equal(compress_rowcols(x, 1).size, 0)

    def test_mask_rowcols(self):
        # Tests mask_rowcols.
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1).mask,
                     [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1).mask,
                     [[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[1, 1, 1], [1, 1, 1], [1, 1, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[1, 1, 1], [1, 1, 1], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1,).mask,
                     [[1, 1, 0], [1, 1, 0], [1, 1, 0]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert_(mask_rowcols(x).all() is masked)
        assert_(mask_rowcols(x, 0).all() is masked)
        assert_(mask_rowcols(x, 1).all() is masked)
        assert_(mask_rowcols(x).mask.all())
        assert_(mask_rowcols(x, 0).mask.all())
        assert_(mask_rowcols(x, 1).mask.all())

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize(["func", "rowcols_axis"],
                             [(np.ma.mask_rows, 0), (np.ma.mask_cols, 1)])
    def test_mask_row_cols_axis_deprecation(self, axis, func, rowcols_axis):
        # Test deprecation of the axis argument to `mask_rows` and `mask_cols`
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        with assert_warns(DeprecationWarning):
            res = func(x, axis=axis)
            assert_equal(res, mask_rowcols(x, rowcols_axis))

    def test_dot(self):
        # Tests dot product
        n = np.arange(1, 7)
        #
        m = [1, 0, 0, 0, 0, 0]
        a = masked_array(n, mask=m).reshape(2, 3)
        b = masked_array(n, mask=m).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[1, 1], [1, 0]])
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        m = [0, 0, 0, 0, 0, 1]
        a = masked_array(n, mask=m).reshape(2, 3)
        b = masked_array(n, mask=m).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[0, 1], [1, 1]])
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        assert_equal(c, dot(a, b))
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        m = [0, 0, 0, 0, 0, 0]
        a = masked_array(n, mask=m).reshape(2, 3)
        b = masked_array(n, mask=m).reshape(3, 2)
        c = dot(a, b)
        assert_equal(c.mask, nomask)
        c = dot(b, a)
        assert_equal(c.mask, nomask)
        #
        a = masked_array(n, mask=[1, 0, 0, 0, 0, 0]).reshape(2, 3)
        b = masked_array(n, mask=[0, 0, 0, 0, 0, 0]).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[1, 1], [0, 0]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        a = masked_array(n, mask=[0, 0, 0, 0, 0, 1]).reshape(2, 3)
        b = masked_array(n, mask=[0, 0, 0, 0, 0, 0]).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[0, 0], [1, 1]])
        c = dot(a, b)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        a = masked_array(n, mask=[0, 0, 0, 0, 0, 1]).reshape(2, 3)
        b = masked_array(n, mask=[0, 0, 1, 0, 0, 0]).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[1, 0], [1, 1]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[0, 0, 1], [1, 1, 1], [0, 0, 1]])
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        a = masked_array(np.arange(8).reshape(2, 2, 2),
                         mask=[[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
        b = masked_array(np.arange(8).reshape(2, 2, 2),
                         mask=[[[0, 0], [0, 0]], [[0, 0], [0, 1]]])
        c = dot(a, b, strict=True)
        assert_equal(c.mask,
                     [[[[1, 1], [1, 1]], [[0, 0], [0, 1]]],
                      [[[0, 0], [0, 1]], [[0, 0], [0, 1]]]])
        c = dot(a, b, strict=False)
        assert_equal(c.mask,
                     [[[[0, 0], [0, 1]], [[0, 0], [0, 0]]],
                      [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]])
        c = dot(b, a, strict=True)
        assert_equal(c.mask,
                     [[[[1, 0], [0, 0]], [[1, 0], [0, 0]]],
                      [[[1, 0], [0, 0]], [[1, 1], [1, 1]]]])
        c = dot(b, a, strict=False)
        assert_equal(c.mask,
                     [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                      [[[0, 0], [0, 0]], [[1, 0], [0, 0]]]])
        #
        a = masked_array(np.arange(8).reshape(2, 2, 2),
                         mask=[[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
        b = 5.
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
        c = dot(a, b, strict=False)
        assert_equal(c.mask, [[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
        c = dot(b, a, strict=False)
        assert_equal(c.mask, [[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
        #
        a = masked_array(np.arange(8).reshape(2, 2, 2),
                         mask=[[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
        b = masked_array(np.arange(2), mask=[0, 1])
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[1, 1], [1, 1]])
        c = dot(a, b, strict=False)
        assert_equal(c.mask, [[1, 0], [0, 0]])

    def test_dot_returns_maskedarray(self):
        # See gh-6611
        a = np.eye(3)
        b = array(a)
        assert_(type(dot(a, a)) is MaskedArray)
        assert_(type(dot(a, b)) is MaskedArray)
        assert_(type(dot(b, a)) is MaskedArray)
        assert_(type(dot(b, b)) is MaskedArray)

    def test_dot_out(self):
        a = array(np.eye(3))
        out = array(np.zeros((3, 3)))
        res = dot(a, a, out=out)
        assert_(res is out)
        assert_equal(a, res)


class TestApplyAlongAxis:
    # Tests 2D functions
    def test_3d(self):
        a = arange(12.).reshape(2, 2, 3)

        def myfunc(b):
            return b[1]

        xa = apply_along_axis(myfunc, 2, a)
        assert_equal(xa, [[1, 4], [7, 10]])

    # Tests kwargs functions
    def test_3d_kwargs(self):
        a = arange(12).reshape(2, 2, 3)

        def myfunc(b, offset=0):
            return b[1 + offset]

        xa = apply_along_axis(myfunc, 2, a, offset=1)
        assert_equal(xa, [[2, 5], [8, 11]])


class TestApplyOverAxes:
    # Tests apply_over_axes
    def test_basic(self):
        a = arange(24).reshape(2, 3, 4)
        test = apply_over_axes(np.sum, a, [0, 2])
        ctrl = np.array([[[60], [92], [124]]])
        assert_equal(test, ctrl)
        a[(a % 2).astype(bool)] = masked
        test = apply_over_axes(np.sum, a, [0, 2])
        ctrl = np.array([[[28], [44], [60]]])
        assert_equal(test, ctrl)


class TestMedian:
    def test_pytype(self):
        r = np.ma.median([[np.inf, np.inf], [np.inf, np.inf]], axis=-1)
        assert_equal(r, np.inf)

    def test_inf(self):
        # test that even which computes handles inf / x = masked
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]]), axis=-1)
        assert_equal(r, np.inf)
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]]), axis=None)
        assert_equal(r, np.inf)
        # all masked
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]], mask=True),
                         axis=-1)
        assert_equal(r.mask, True)
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]], mask=True),
                         axis=None)
        assert_equal(r.mask, True)

    def test_non_masked(self):
        x = np.arange(9)
        assert_equal(np.ma.median(x), 4.)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = range(8)
        assert_equal(np.ma.median(x), 3.5)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = 5
        assert_equal(np.ma.median(x), 5.)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # integer
        x = np.arange(9 * 8).reshape(9, 8)
        assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
        assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
        assert_(np.ma.median(x, axis=1) is not MaskedArray)
        # float
        x = np.arange(9 * 8.).reshape(9, 8)
        assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
        assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
        assert_(np.ma.median(x, axis=1) is not MaskedArray)

    def test_docstring_examples(self):
        "test the examples given in the docstring of ma.median"
        x = array(np.arange(8), mask=[0] * 4 + [1] * 4)
        assert_equal(np.ma.median(x), 1.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = array(np.arange(10).reshape(2, 5), mask=[0] * 6 + [1] * 4)
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        ma_x = np.ma.median(x, axis=-1, overwrite_input=True)
        assert_equal(ma_x, [2., 5.])
        assert_equal(ma_x.shape, (2,), "shape mismatch")
        assert_(type(ma_x) is MaskedArray)

    def test_axis_argument_errors(self):
        msg = "mask = %s, ndim = %s, axis = %s, overwrite_input = %s"
        for ndmin in range(5):
            for mask in [False, True]:
                x = array(1, ndmin=ndmin, mask=mask)

                # Valid axis values should not raise exception
                args = itertools.product(range(-ndmin, ndmin), [False, True])
                for axis, over in args:
                    try:
                        np.ma.median(x, axis=axis, overwrite_input=over)
                    except Exception:
                        raise AssertionError(msg % (mask, ndmin, axis, over))

                # Invalid axis values should raise exception
                args = itertools.product([-(ndmin + 1), ndmin], [False, True])
                for axis, over in args:
                    try:
                        np.ma.median(x, axis=axis, overwrite_input=over)
                    except np.exceptions.AxisError:
                        pass
                    else:
                        raise AssertionError(msg % (mask, ndmin, axis, over))

    def test_masked_0d(self):
        # Check values
        x = array(1, mask=False)
        assert_equal(np.ma.median(x), 1)
        x = array(1, mask=True)
        assert_equal(np.ma.median(x), np.ma.masked)

    def test_masked_1d(self):
        x = array(np.arange(5), mask=True)
        assert_equal(np.ma.median(x), np.ma.masked)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is np.ma.core.MaskedConstant)
        x = array(np.arange(5), mask=False)
        assert_equal(np.ma.median(x), 2.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = array(np.arange(5), mask=[0, 1, 0, 0, 0])
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = array(np.arange(5), mask=[0, 1, 1, 1, 1])
        assert_equal(np.ma.median(x), 0.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # integer
        x = array(np.arange(5), mask=[0, 1, 1, 0, 0])
        assert_equal(np.ma.median(x), 3.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # float
        x = array(np.arange(5.), mask=[0, 1, 1, 0, 0])
        assert_equal(np.ma.median(x), 3.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # integer
        x = array(np.arange(6), mask=[0, 1, 1, 1, 1, 0])
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # float
        x = array(np.arange(6.), mask=[0, 1, 1, 1, 1, 0])
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)

    def test_1d_shape_consistency(self):
        assert_equal(np.ma.median(array([1, 2, 3], mask=[0, 0, 0])).shape,
                     np.ma.median(array([1, 2, 3], mask=[0, 1, 0])).shape)

    def test_2d(self):
        # Tests median w/ 2D
        (n, p) = (101, 30)
        x = masked_array(np.linspace(-1., 1., n),)
        x[:10] = x[-10:] = masked
        z = masked_array(np.empty((n, p), dtype=float))
        z[:, 0] = x[:]
        idx = np.arange(len(x))
        for i in range(1, p):
            np.random.shuffle(idx)
            z[:, i] = x[idx]
        assert_equal(median(z[:, 0]), 0)
        assert_equal(median(z), 0)
        assert_equal(median(z, axis=0), np.zeros(p))
        assert_equal(median(z.T, axis=1), np.zeros(p))

    def test_2d_waxis(self):
        # Tests median w/ 2D arrays and different axis.
        x = masked_array(np.arange(30).reshape(10, 3))
        x[:3] = x[-3:] = masked
        assert_equal(median(x), 14.5)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        assert_equal(median(x, axis=0), [13.5, 14.5, 15.5])
        assert_(type(np.ma.median(x, axis=0)) is MaskedArray)
        assert_equal(median(x, axis=1), [0, 0, 0, 10, 13, 16, 19, 0, 0, 0])
        assert_(type(np.ma.median(x, axis=1)) is MaskedArray)
        assert_equal(median(x, axis=1).mask, [1, 1, 1, 0, 0, 0, 0, 1, 1, 1])

    def test_3d(self):
        # Tests median w/ 3D
        x = np.ma.arange(24).reshape(3, 4, 2)
        x[x % 3 == 0] = masked
        assert_equal(median(x, 0), [[12, 9], [6, 15], [12, 9], [18, 15]])
        x.shape = (4, 3, 2)
        assert_equal(median(x, 0), [[99, 10], [11, 99], [13, 14]])
        x = np.ma.arange(24).reshape(4, 3, 2)
        x[x % 5 == 0] = masked
        assert_equal(median(x, 0), [[12, 10], [8, 9], [16, 17]])

    def test_neg_axis(self):
        x = masked_array(np.arange(30).reshape(10, 3))
        x[:3] = x[-3:] = masked
        assert_equal(median(x, axis=-1), median(x, axis=1))

    def test_out_1d(self):
        # integer float even odd
        for v in (30, 30., 31, 31.):
            x = masked_array(np.arange(v))
            x[:3] = x[-3:] = masked
            out = masked_array(np.ones(()))
            r = median(x, out=out)
            if v == 30:
                assert_equal(out, 14.5)
            else:
                assert_equal(out, 15.)
            assert_(r is out)
            assert_(type(r) is MaskedArray)

    def test_out(self):
        # integer float even odd
        for v in (40, 40., 30, 30.):
            x = masked_array(np.arange(v).reshape(10, -1))
            x[:3] = x[-3:] = masked
            out = masked_array(np.ones(10))
            r = median(x, axis=1, out=out)
            if v == 30:
                e = masked_array([0.] * 3 + [10, 13, 16, 19] + [0.] * 3,
                                 mask=[True] * 3 + [False] * 4 + [True] * 3)
            else:
                e = masked_array([0.] * 3 + [13.5, 17.5, 21.5, 25.5] + [0.] * 3,
                                 mask=[True] * 3 + [False] * 4 + [True] * 3)
            assert_equal(r, e)
            assert_(r is out)
            assert_(type(r) is MaskedArray)

    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1, ),
            (0, 1),
            (-3, -1),
        ]
    )
    def test_keepdims_out(self, axis):
        mask = np.zeros((3, 5, 7, 11), dtype=bool)
        # Randomly set some elements to True:
        w = np.random.random((4, 200)) * np.array(mask.shape)[:, None]
        w = w.astype(np.intp)
        mask[tuple(w)] = np.nan
        d = masked_array(np.ones(mask.shape), mask=mask)
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        out = masked_array(np.empty(shape_out))
        result = median(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_single_non_masked_value_on_axis(self):
        data = [[1., 0.],
                [0., 3.],
                [0., 0.]]
        masked_arr = np.ma.masked_equal(data, 0)
        expected = [1., 3.]
        assert_array_equal(np.ma.median(masked_arr, axis=0),
                           expected)

    def test_nan(self):
        for mask in (False, np.zeros(6, dtype=bool)):
            dm = np.ma.array([[1, np.nan, 3], [1, 2, 3]])
            dm.mask = mask

            # scalar result
            r = np.ma.median(dm, axis=None)
            assert_(np.isscalar(r))
            assert_array_equal(r, np.nan)
            r = np.ma.median(dm.ravel(), axis=0)
            assert_(np.isscalar(r))
            assert_array_equal(r, np.nan)

            r = np.ma.median(dm, axis=0)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [1, np.nan, 3])
            r = np.ma.median(dm, axis=1)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [np.nan, 2])
            r = np.ma.median(dm, axis=-1)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [np.nan, 2])

        dm = np.ma.array([[1, np.nan, 3], [1, 2, 3]])
        dm[:, 2] = np.ma.masked
        assert_array_equal(np.ma.median(dm, axis=None), np.nan)
        assert_array_equal(np.ma.median(dm, axis=0), [1, np.nan, 3])
        assert_array_equal(np.ma.median(dm, axis=1), [np.nan, 1.5])

    def test_out_nan(self):
        o = np.ma.masked_array(np.zeros((4,)))
        d = np.ma.masked_array(np.ones((3, 4)))
        d[2, 1] = np.nan
        d[2, 2] = np.ma.masked
        assert_equal(np.ma.median(d, 0, out=o), o)
        o = np.ma.masked_array(np.zeros((3,)))
        assert_equal(np.ma.median(d, 1, out=o), o)
        o = np.ma.masked_array(np.zeros(()))
        assert_equal(np.ma.median(d, out=o), o)

    def test_nan_behavior(self):
        a = np.ma.masked_array(np.arange(24, dtype=float))
        a[::3] = np.ma.masked
        a[2] = np.nan
        assert_array_equal(np.ma.median(a), np.nan)
        assert_array_equal(np.ma.median(a, axis=0), np.nan)

        a = np.ma.masked_array(np.arange(24, dtype=float).reshape(2, 3, 4))
        a.mask = np.arange(a.size) % 2 == 1
        aorig = a.copy()
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # no axis
        assert_array_equal(np.ma.median(a), np.nan)
        assert_(np.isscalar(np.ma.median(a)))

        # axis0
        b = np.ma.median(aorig, axis=0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.ma.median(a, 0), b)

        # axis1
        b = np.ma.median(aorig, axis=1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.ma.median(a, 1), b)

        # axis02
        b = np.ma.median(aorig, axis=(0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.ma.median(a, (0, 2)), b)

    def test_ambigous_fill(self):
        # 255 is max value, used as filler for sort
        a = np.array([[3, 3, 255], [3, 3, 255]], dtype=np.uint8)
        a = np.ma.masked_array(a, mask=a == 3)
        assert_array_equal(np.ma.median(a, axis=1), 255)
        assert_array_equal(np.ma.median(a, axis=1).mask, False)
        assert_array_equal(np.ma.median(a, axis=0), a[0])
        assert_array_equal(np.ma.median(a), 255)

    def test_special(self):
        for inf in [np.inf, -np.inf]:
            a = np.array([[inf, np.nan], [np.nan, np.nan]])
            a = np.ma.masked_array(a, mask=np.isnan(a))
            assert_equal(np.ma.median(a, axis=0), [inf, np.nan])
            assert_equal(np.ma.median(a, axis=1), [inf, np.nan])
            assert_equal(np.ma.median(a), inf)

            a = np.array([[np.nan, np.nan, inf], [np.nan, np.nan, inf]])
            a = np.ma.masked_array(a, mask=np.isnan(a))
            assert_array_equal(np.ma.median(a, axis=1), inf)
            assert_array_equal(np.ma.median(a, axis=1).mask, False)
            assert_array_equal(np.ma.median(a, axis=0), a[0])
            assert_array_equal(np.ma.median(a), inf)

            # no mask
            a = np.array([[inf, inf], [inf, inf]])
            assert_equal(np.ma.median(a), inf)
            assert_equal(np.ma.median(a, axis=0), inf)
            assert_equal(np.ma.median(a, axis=1), inf)

            a = np.array([[inf, 7, -inf, -9],
                          [-10, np.nan, np.nan, 5],
                          [4, np.nan, np.nan, inf]],
                          dtype=np.float32)
            a = np.ma.masked_array(a, mask=np.isnan(a))
            if inf > 0:
                assert_equal(np.ma.median(a, axis=0), [4., 7., -inf, 5.])
                assert_equal(np.ma.median(a), 4.5)
            else:
                assert_equal(np.ma.median(a, axis=0), [-10., 7., -inf, -9.])
                assert_equal(np.ma.median(a), -2.5)
            assert_equal(np.ma.median(a, axis=1), [-1., -2.5, inf])

            for i in range(10):
                for j in range(1, 10):
                    a = np.array([([np.nan] * i) + ([inf] * j)] * 2)
                    a = np.ma.masked_array(a, mask=np.isnan(a))
                    assert_equal(np.ma.median(a), inf)
                    assert_equal(np.ma.median(a, axis=1), inf)
                    assert_equal(np.ma.median(a, axis=0),
                                 ([np.nan] * i) + [inf] * j)

    def test_empty(self):
        # empty arrays
        a = np.ma.masked_array(np.array([], dtype=float))
        with suppress_warnings() as w:
            w.record(RuntimeWarning)
            assert_array_equal(np.ma.median(a), np.nan)
            assert_(w.log[0].category is RuntimeWarning)

        # multiple dimensions
        a = np.ma.masked_array(np.array([], dtype=float, ndmin=3))
        # no axis
        with suppress_warnings() as w:
            w.record(RuntimeWarning)
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_array_equal(np.ma.median(a), np.nan)
            assert_(w.log[0].category is RuntimeWarning)

        # axis 0 and 1
        b = np.ma.masked_array(np.array([], dtype=float, ndmin=2))
        assert_equal(np.ma.median(a, axis=0), b)
        assert_equal(np.ma.median(a, axis=1), b)

        # axis 2
        b = np.ma.masked_array(np.array(np.nan, dtype=float, ndmin=2))
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.ma.median(a, axis=2), b)
            assert_(w[0].category is RuntimeWarning)

    def test_object(self):
        o = np.ma.masked_array(np.arange(7.))
        assert_(type(np.ma.median(o.astype(object))), float)
        o[2] = np.nan
        assert_(type(np.ma.median(o.astype(object))), float)


class TestCov:

    def setup_method(self):
        self.data = array(np.random.rand(12))

    def test_covhelper(self):
        x = self.data
        # Test not mask output type is a float.
        assert_(_covhelper(x, rowvar=True)[1].dtype, np.float32)
        assert_(_covhelper(x, y=x, rowvar=False)[1].dtype, np.float32)
        # Test not mask output is equal after casting to float.
        mask = x > 0.5
        assert_array_equal(
            _covhelper(
                np.ma.masked_array(x, mask), rowvar=True
            )[1].astype(bool),
            ~mask.reshape(1, -1),
        )
        assert_array_equal(
            _covhelper(
                np.ma.masked_array(x, mask), y=x, rowvar=False
            )[1].astype(bool),
            np.vstack((~mask, ~mask)),
        )

    def test_1d_without_missing(self):
        # Test cov on 1D variable w/o missing values
        x = self.data
        assert_almost_equal(np.cov(x), cov(x))
        assert_almost_equal(np.cov(x, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(x, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))

    def test_2d_without_missing(self):
        # Test cov on 1 2D variable w/o missing values
        x = self.data.reshape(3, 4)
        assert_almost_equal(np.cov(x), cov(x))
        assert_almost_equal(np.cov(x, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(x, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))

    def test_1d_with_missing(self):
        # Test cov 1 1D variable w/missing values
        x = self.data
        x[-1] = masked
        x -= x.mean()
        nx = x.compressed()
        assert_almost_equal(np.cov(nx), cov(x))
        assert_almost_equal(np.cov(nx, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(nx, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))
        #
        try:
            cov(x, allow_masked=False)
        except ValueError:
            pass
        #
        # 2 1D variables w/ missing values
        nx = x[1:-1]
        assert_almost_equal(np.cov(nx, nx[::-1]), cov(x, x[::-1]))
        assert_almost_equal(np.cov(nx, nx[::-1], rowvar=False),
                            cov(x, x[::-1], rowvar=False))
        assert_almost_equal(np.cov(nx, nx[::-1], rowvar=False, bias=True),
                            cov(x, x[::-1], rowvar=False, bias=True))

    def test_2d_with_missing(self):
        # Test cov on 2D variable w/ missing value
        x = self.data
        x[-1] = masked
        x = x.reshape(3, 4)
        valid = np.logical_not(getmaskarray(x)).astype(int)
        frac = np.dot(valid, valid.T)
        xf = (x - x.mean(1)[:, None]).filled(0)
        assert_almost_equal(cov(x),
                            np.cov(xf) * (x.shape[1] - 1) / (frac - 1.))
        assert_almost_equal(cov(x, bias=True),
                            np.cov(xf, bias=True) * x.shape[1] / frac)
        frac = np.dot(valid.T, valid)
        xf = (x - x.mean(0)).filled(0)
        assert_almost_equal(cov(x, rowvar=False),
                            (np.cov(xf, rowvar=False) *
                             (x.shape[0] - 1) / (frac - 1.)))
        assert_almost_equal(cov(x, rowvar=False, bias=True),
                            (np.cov(xf, rowvar=False, bias=True) *
                             x.shape[0] / frac))


class TestCorrcoef:

    def setup_method(self):
        self.data = array(np.random.rand(12))
        self.data2 = array(np.random.rand(12))

    def test_ddof(self):
        # ddof raises DeprecationWarning
        x, y = self.data, self.data2
        expected = np.corrcoef(x)
        expected2 = np.corrcoef(x, y)
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, x, ddof=-1)
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof has no or negligible effect on the function
            assert_almost_equal(np.corrcoef(x, ddof=0), corrcoef(x, ddof=0))
            assert_almost_equal(corrcoef(x, ddof=-1), expected)
            assert_almost_equal(corrcoef(x, y, ddof=-1), expected2)
            assert_almost_equal(corrcoef(x, ddof=3), expected)
            assert_almost_equal(corrcoef(x, y, ddof=3), expected2)

    def test_bias(self):
        x, y = self.data, self.data2
        expected = np.corrcoef(x)
        # bias raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, x, y, True, False)
            assert_warns(DeprecationWarning, corrcoef, x, y, True, True)
            assert_warns(DeprecationWarning, corrcoef, x, bias=False)
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # bias has no or negligible effect on the function
            assert_almost_equal(corrcoef(x, bias=1), expected)

    def test_1d_without_missing(self):
        # Test cov on 1D variable w/o missing values
        x = self.data
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))

    def test_2d_without_missing(self):
        # Test corrcoef on 1 2D variable w/o missing values
        x = self.data.reshape(3, 4)
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))

    def test_1d_with_missing(self):
        # Test corrcoef 1 1D variable w/missing values
        x = self.data
        x[-1] = masked
        x -= x.mean()
        nx = x.compressed()
        assert_almost_equal(np.corrcoef(nx), corrcoef(x))
        assert_almost_equal(np.corrcoef(nx, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(nx, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))
        try:
            corrcoef(x, allow_masked=False)
        except ValueError:
            pass
        # 2 1D variables w/ missing values
        nx = x[1:-1]
        assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1]))
        assert_almost_equal(np.corrcoef(nx, nx[::-1], rowvar=False),
                            corrcoef(x, x[::-1], rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof and bias have no or negligible effect on the function
            assert_almost_equal(np.corrcoef(nx, nx[::-1]),
                                corrcoef(x, x[::-1], bias=1))
            assert_almost_equal(np.corrcoef(nx, nx[::-1]),
                                corrcoef(x, x[::-1], ddof=2))

    def test_2d_with_missing(self):
        # Test corrcoef on 2D variable w/ missing value
        x = self.data
        x[-1] = masked
        x = x.reshape(3, 4)

        test = corrcoef(x)
        control = np.corrcoef(x)
        assert_almost_equal(test[:-1, :-1], control[:-1, :-1])
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof and bias have no or negligible effect on the function
            assert_almost_equal(corrcoef(x, ddof=-2)[:-1, :-1],
                                control[:-1, :-1])
            assert_almost_equal(corrcoef(x, ddof=3)[:-1, :-1],
                                control[:-1, :-1])
            assert_almost_equal(corrcoef(x, bias=1)[:-1, :-1],
                                control[:-1, :-1])


class TestPolynomial:
    #
    def test_polyfit(self):
        # Tests polyfit
        # On ndarrays
        x = np.random.rand(10)
        y = np.random.rand(20).reshape(-1, 2)
        assert_almost_equal(polyfit(x, y, 3), np.polyfit(x, y, 3))
        # ON 1D maskedarrays
        x = x.view(MaskedArray)
        x[0] = masked
        y = y.view(MaskedArray)
        y[0, 0] = y[-1, -1] = masked
        #
        (C, R, K, S, D) = polyfit(x, y[:, 0], 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:], y[1:, 0].compressed(), 3,
                                     full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        #
        (C, R, K, S, D) = polyfit(x, y[:, -1], 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1, -1], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        #
        (C, R, K, S, D) = polyfit(x, y, 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1, :], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        #
        w = np.random.rand(10) + 1
        wo = w.copy()
        xs = x[1:-1]
        ys = y[1:-1]
        ws = w[1:-1]
        (C, R, K, S, D) = polyfit(x, y, 3, full=True, w=w)
        (c, r, k, s, d) = np.polyfit(xs, ys, 3, full=True, w=ws)
        assert_equal(w, wo)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)

    def test_polyfit_with_masked_NaNs(self):
        x = np.random.rand(10)
        y = np.random.rand(20).reshape(-1, 2)

        x[0] = np.nan
        y[-1, -1] = np.nan
        x = x.view(MaskedArray)
        y = y.view(MaskedArray)
        x[0] = masked
        y[-1, -1] = masked

        (C, R, K, S, D) = polyfit(x, y, 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1, :], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)


class TestArraySetOps:

    def test_unique_onlist(self):
        # Test unique on list
        data = [1, 1, 1, 2, 2, 3]
        test = unique(data, return_index=True, return_inverse=True)
        assert_(isinstance(test[0], MaskedArray))
        assert_equal(test[0], masked_array([1, 2, 3], mask=[0, 0, 0]))
        assert_equal(test[1], [0, 3, 5])
        assert_equal(test[2], [0, 0, 0, 1, 1, 2])

    def test_unique_onmaskedarray(self):
        # Test unique on masked data w/use_mask=True
        data = masked_array([1, 1, 1, 2, 2, 3], mask=[0, 0, 1, 0, 1, 0])
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])
        #
        data.fill_value = 3
        data = masked_array(data=[1, 1, 1, 2, 2, 3],
                            mask=[0, 0, 1, 0, 1, 0], fill_value=3)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])

    def test_unique_allmasked(self):
        # Test all masked
        data = masked_array([1, 1, 1], mask=True)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, ], mask=[True]))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0, 0, 0])
        #
        # Test masked
        data = masked
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array(masked))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0])

    def test_ediff1d(self):
        # Tests mediff1d
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        control = array([1, 1, 1, 4], mask=[1, 0, 0, 1])
        test = ediff1d(x)
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_tobegin(self):
        # Test ediff1d w/ to_begin
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_begin=masked)
        control = array([0, 1, 1, 1, 4], mask=[1, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_begin=[1, 2, 3])
        control = array([1, 2, 3, 1, 1, 1, 4], mask=[0, 0, 0, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_toend(self):
        # Test ediff1d w/ to_end
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_end=masked)
        control = array([1, 1, 1, 4, 0], mask=[1, 0, 0, 1, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=[1, 2, 3])
        control = array([1, 1, 1, 4, 1, 2, 3], mask=[1, 0, 0, 1, 0, 0, 0])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_tobegin_toend(self):
        # Test ediff1d w/ to_begin and to_end
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 4, 0], mask=[1, 1, 0, 0, 1, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=[1, 2, 3], to_begin=masked)
        control = array([0, 1, 1, 1, 4, 1, 2, 3],
                        mask=[1, 1, 0, 0, 1, 0, 0, 0])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_ndarray(self):
        # Test ediff1d w/ a ndarray
        x = np.arange(5)
        test = ediff1d(x)
        control = array([1, 1, 1, 1], mask=[0, 0, 0, 0])
        assert_equal(test, control)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 1, 0], mask=[1, 0, 0, 0, 0, 1])
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_intersect1d(self):
        # Test intersect1d
        x = array([1, 3, 3, 3], mask=[0, 0, 0, 1])
        y = array([3, 1, 1, 1], mask=[0, 0, 0, 1])
        test = intersect1d(x, y)
        control = array([1, 3, -1], mask=[0, 0, 1])
        assert_equal(test, control)

    def test_setxor1d(self):
        # Test setxor1d
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7]))
        #
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = [1, 2, 3, 4, 5]
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7, -1], mask=[0, 0, 0, 1]))
        #
        a = array([1, 2, 3])
        b = array([6, 5, 4])
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        a = array([1, 8, 2, 3], mask=[0, 1, 0, 0])
        b = array([6, 5, 4, 8], mask=[0, 0, 0, 1])
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        assert_array_equal([], setxor1d([], []))

    def test_setxor1d_unique(self):
        # Test setxor1d with assume_unique=True
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = [1, 2, 3, 4, 5]
        test = setxor1d(a, b, assume_unique=True)
        assert_equal(test, array([3, 4, 7, -1], mask=[0, 0, 0, 1]))
        #
        a = array([1, 8, 2, 3], mask=[0, 1, 0, 0])
        b = array([6, 5, 4, 8], mask=[0, 0, 0, 1])
        test = setxor1d(a, b, assume_unique=True)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        a = array([[1], [8], [2], [3]])
        b = array([[6, 5], [4, 8]])
        test = setxor1d(a, b, assume_unique=True)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])

    def test_isin(self):
        # the tests for in1d cover most of isin's behavior
        # if in1d is removed, would need to change those tests to test
        # isin instead.
        a = np.arange(24).reshape([2, 3, 4])
        mask = np.zeros([2, 3, 4])
        mask[1, 2, 0] = 1
        a = array(a, mask=mask)
        b = array(data=[0, 10, 20, 30,  1,  3, 11, 22, 33],
                  mask=[0,  1,  0,  1,  0,  1,  0,  1,  0])
        ec = zeros((2, 3, 4), dtype=bool)
        ec[0, 0, 0] = True
        ec[0, 0, 1] = True
        ec[0, 2, 3] = True
        c = isin(a, b)
        assert_(isinstance(c, MaskedArray))
        assert_array_equal(c, ec)
        # compare results of np.isin to ma.isin
        d = np.isin(a, b[~b.mask]) & ~a.mask
        assert_array_equal(c, d)

    def test_in1d(self):
        # Test in1d
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = in1d(a, b)
        assert_equal(test, [True, True, True, False, True])
        #
        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        test = in1d(a, b)
        assert_equal(test, [True, True, False, True, True])
        #
        assert_array_equal([], in1d([], []))

    def test_in1d_invert(self):
        # Test in1d's invert parameter
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))

        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))

        assert_array_equal([], in1d([], [], invert=True))

    def test_union1d(self):
        # Test union1d
        a = array([1, 2, 5, 7, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = union1d(a, b)
        control = array([1, 2, 3, 4, 5, 7, -1], mask=[0, 0, 0, 0, 0, 0, 1])
        assert_equal(test, control)

        # Tests gh-10340, arguments to union1d should be
        # flattened if they are not already 1D
        x = array([[0, 1, 2], [3, 4, 5]], mask=[[0, 0, 0], [0, 0, 1]])
        y = array([0, 1, 2, 3, 4], mask=[0, 0, 0, 0, 1])
        ez = array([0, 1, 2, 3, 4, 5], mask=[0, 0, 0, 0, 0, 1])
        z = union1d(x, y)
        assert_equal(z, ez)
        #
        assert_array_equal([], union1d([], []))

    def test_setdiff1d(self):
        # Test setdiff1d
        a = array([6, 5, 4, 7, 7, 1, 2, 1], mask=[0, 0, 0, 0, 0, 0, 0, 1])
        b = array([2, 4, 3, 3, 2, 1, 5])
        test = setdiff1d(a, b)
        assert_equal(test, array([6, 7, -1], mask=[0, 0, 1]))
        #
        a = arange(10)
        b = arange(8)
        assert_equal(setdiff1d(a, b), array([8, 9]))
        a = array([], np.uint32, mask=[])
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    def test_setdiff1d_char_array(self):
        # Test setdiff1d_charray
        a = np.array(['a', 'b', 'c'])
        b = np.array(['a', 'b', 's'])
        assert_array_equal(setdiff1d(a, b), np.array(['c']))


class TestShapeBase:

    def test_atleast_2d(self):
        # Test atleast_2d
        a = masked_array([0, 1, 2], mask=[0, 1, 0])
        b = atleast_2d(a)
        assert_equal(b.shape, (1, 3))
        assert_equal(b.mask.shape, b.data.shape)
        assert_equal(a.shape, (3,))
        assert_equal(a.mask.shape, a.data.shape)
        assert_equal(b.mask.shape, b.data.shape)

    def test_shape_scalar(self):
        # the atleast and diagflat function should work with scalars
        # GitHub issue #3367
        # Additionally, the atleast functions should accept multiple scalars
        # correctly
        b = atleast_1d(1.0)
        assert_equal(b.shape, (1,))
        assert_equal(b.mask.shape, b.shape)
        assert_equal(b.data.shape, b.shape)

        b = atleast_1d(1.0, 2.0)
        for a in b:
            assert_equal(a.shape, (1,))
            assert_equal(a.mask.shape, a.shape)
            assert_equal(a.data.shape, a.shape)

        b = atleast_2d(1.0)
        assert_equal(b.shape, (1, 1))
        assert_equal(b.mask.shape, b.shape)
        assert_equal(b.data.shape, b.shape)

        b = atleast_2d(1.0, 2.0)
        for a in b:
            assert_equal(a.shape, (1, 1))
            assert_equal(a.mask.shape, a.shape)
            assert_equal(a.data.shape, a.shape)

        b = atleast_3d(1.0)
        assert_equal(b.shape, (1, 1, 1))
        assert_equal(b.mask.shape, b.shape)
        assert_equal(b.data.shape, b.shape)

        b = atleast_3d(1.0, 2.0)
        for a in b:
            assert_equal(a.shape, (1, 1, 1))
            assert_equal(a.mask.shape, a.shape)
            assert_equal(a.data.shape, a.shape)

        b = diagflat(1.0)
        assert_equal(b.shape, (1, 1))
        assert_equal(b.mask.shape, b.data.shape)


class TestNDEnumerate:

    def test_ndenumerate_nomasked(self):
        ordinary = np.arange(6.).reshape((1, 3, 2))
        empty_mask = np.zeros_like(ordinary, dtype=bool)
        with_mask = masked_array(ordinary, mask=empty_mask)
        assert_equal(list(np.ndenumerate(ordinary)),
                     list(ndenumerate(ordinary)))
        assert_equal(list(ndenumerate(ordinary)),
                     list(ndenumerate(with_mask)))
        assert_equal(list(ndenumerate(with_mask)),
                     list(ndenumerate(with_mask, compressed=False)))

    def test_ndenumerate_allmasked(self):
        a = masked_all(())
        b = masked_all((100,))
        c = masked_all((2, 3, 4))
        assert_equal(list(ndenumerate(a)), [])
        assert_equal(list(ndenumerate(b)), [])
        assert_equal(list(ndenumerate(b, compressed=False)),
                     list(zip(np.ndindex((100,)), 100 * [masked])))
        assert_equal(list(ndenumerate(c)), [])
        assert_equal(list(ndenumerate(c, compressed=False)),
                     list(zip(np.ndindex((2, 3, 4)), 2 * 3 * 4 * [masked])))

    def test_ndenumerate_mixedmasked(self):
        a = masked_array(np.arange(12).reshape((3, 4)),
                         mask=[[1, 1, 1, 1],
                               [1, 1, 0, 1],
                               [0, 0, 0, 0]])
        items = [((1, 2), 6),
                 ((2, 0), 8), ((2, 1), 9), ((2, 2), 10), ((2, 3), 11)]
        assert_equal(list(ndenumerate(a)), items)
        assert_equal(len(list(ndenumerate(a, compressed=False))), a.size)
        for coordinate, value in ndenumerate(a, compressed=False):
            assert_equal(a[coordinate], value)


class TestStack:

    def test_stack_1d(self):
        a = masked_array([0, 1, 2], mask=[0, 1, 0])
        b = masked_array([9, 8, 7], mask=[1, 0, 0])

        c = stack([a, b], axis=0)
        assert_equal(c.shape, (2, 3))
        assert_array_equal(a.mask, c[0].mask)
        assert_array_equal(b.mask, c[1].mask)

        d = vstack([a, b])
        assert_array_equal(c.data, d.data)
        assert_array_equal(c.mask, d.mask)

        c = stack([a, b], axis=1)
        assert_equal(c.shape, (3, 2))
        assert_array_equal(a.mask, c[:, 0].mask)
        assert_array_equal(b.mask, c[:, 1].mask)

    def test_stack_masks(self):
        a = masked_array([0, 1, 2], mask=True)
        b = masked_array([9, 8, 7], mask=False)

        c = stack([a, b], axis=0)
        assert_equal(c.shape, (2, 3))
        assert_array_equal(a.mask, c[0].mask)
        assert_array_equal(b.mask, c[1].mask)

        d = vstack([a, b])
        assert_array_equal(c.data, d.data)
        assert_array_equal(c.mask, d.mask)

        c = stack([a, b], axis=1)
        assert_equal(c.shape, (3, 2))
        assert_array_equal(a.mask, c[:, 0].mask)
        assert_array_equal(b.mask, c[:, 1].mask)

    def test_stack_nd(self):
        # 2D
        shp = (3, 2)
        d1 = np.random.randint(0, 10, shp)
        d2 = np.random.randint(0, 10, shp)
        m1 = np.random.randint(0, 2, shp).astype(bool)
        m2 = np.random.randint(0, 2, shp).astype(bool)
        a1 = masked_array(d1, mask=m1)
        a2 = masked_array(d2, mask=m2)

        c = stack([a1, a2], axis=0)
        c_shp = (2,) + shp
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[0].mask)
        assert_array_equal(a2.mask, c[1].mask)

        c = stack([a1, a2], axis=-1)
        c_shp = shp + (2,)
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[..., 0].mask)
        assert_array_equal(a2.mask, c[..., 1].mask)

        # 4D
        shp = (3, 2, 4, 5,)
        d1 = np.random.randint(0, 10, shp)
        d2 = np.random.randint(0, 10, shp)
        m1 = np.random.randint(0, 2, shp).astype(bool)
        m2 = np.random.randint(0, 2, shp).astype(bool)
        a1 = masked_array(d1, mask=m1)
        a2 = masked_array(d2, mask=m2)

        c = stack([a1, a2], axis=0)
        c_shp = (2,) + shp
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[0].mask)
        assert_array_equal(a2.mask, c[1].mask)

        c = stack([a1, a2], axis=-1)
        c_shp = shp + (2,)
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[..., 0].mask)
        assert_array_equal(a2.mask, c[..., 1].mask)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_dtype.py ===
import ctypes
import gc
import operator
import pickle
import random
import sys
import types
from itertools import permutations
from typing import Any

import hypothesis
import pytest
from hypothesis.extra import numpy as hynp
from numpy._core._multiarray_tests import create_custom_field_dtype
from numpy._core._rational_tests import rational

import numpy as np
import numpy.dtypes
from numpy.testing import (
    HAS_REFCOUNT,
    IS_PYSTON,
    IS_WASM,
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
)


def assert_dtype_equal(a, b):
    assert_equal(a, b)
    assert_equal(hash(a), hash(b),
                 "two equivalent types do not hash to the same value !")

def assert_dtype_not_equal(a, b):
    assert_(a != b)
    assert_(hash(a) != hash(b),
            "two different types hash to the same value !")

class TestBuiltin:
    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object])
    def test_run(self, t):
        """Only test hash runs at all."""
        dt = np.dtype(t)
        hash(dt)

    @pytest.mark.parametrize('t', [int, float])
    def test_dtype(self, t):
        # Make sure equivalent byte order char hash the same (e.g. < and = on
        # little endian)
        dt = np.dtype(t)
        dt2 = dt.newbyteorder("<")
        dt3 = dt.newbyteorder(">")
        if dt == dt2:
            assert_(dt.byteorder != dt2.byteorder, "bogus test")
            assert_dtype_equal(dt, dt2)
        else:
            assert_(dt.byteorder != dt3.byteorder, "bogus test")
            assert_dtype_equal(dt, dt3)

    def test_equivalent_dtype_hashing(self):
        # Make sure equivalent dtypes with different type num hash equal
        uintp = np.dtype(np.uintp)
        if uintp.itemsize == 4:
            left = uintp
            right = np.dtype(np.uint32)
        else:
            left = uintp
            right = np.dtype(np.ulonglong)
        assert_(left == right)
        assert_(hash(left) == hash(right))

    def test_invalid_types(self):
        # Make sure invalid type strings raise an error

        assert_raises(TypeError, np.dtype, 'O3')
        assert_raises(TypeError, np.dtype, 'O5')
        assert_raises(TypeError, np.dtype, 'O7')
        assert_raises(TypeError, np.dtype, 'b3')
        assert_raises(TypeError, np.dtype, 'h4')
        assert_raises(TypeError, np.dtype, 'I5')
        assert_raises(TypeError, np.dtype, 'e3')
        assert_raises(TypeError, np.dtype, 'f5')

        if np.dtype('g').itemsize == 8 or np.dtype('g').itemsize == 16:
            assert_raises(TypeError, np.dtype, 'g12')
        elif np.dtype('g').itemsize == 12:
            assert_raises(TypeError, np.dtype, 'g16')

        if np.dtype('l').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'l4')
            assert_raises(TypeError, np.dtype, 'L4')
        else:
            assert_raises(TypeError, np.dtype, 'l8')
            assert_raises(TypeError, np.dtype, 'L8')

        if np.dtype('q').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'q4')
            assert_raises(TypeError, np.dtype, 'Q4')
        else:
            assert_raises(TypeError, np.dtype, 'q8')
            assert_raises(TypeError, np.dtype, 'Q8')

        # Make sure negative-sized dtype raises an error
        assert_raises(TypeError, np.dtype, 'S-1')
        assert_raises(TypeError, np.dtype, 'U-1')
        assert_raises(TypeError, np.dtype, 'V-1')

    def test_richcompare_invalid_dtype_equality(self):
        # Make sure objects that cannot be converted to valid
        # dtypes results in False/True when compared to valid dtypes.
        # Here 7 cannot be converted to dtype. No exceptions should be raised

        assert not np.dtype(np.int32) == 7, "dtype richcompare failed for =="
        assert np.dtype(np.int32) != 7, "dtype richcompare failed for !="

    @pytest.mark.parametrize(
        'operation',
        [operator.le, operator.lt, operator.ge, operator.gt])
    def test_richcompare_invalid_dtype_comparison(self, operation):
        # Make sure TypeError is raised for comparison operators
        # for invalid dtypes. Here 7 is an invalid dtype.

        with pytest.raises(TypeError):
            operation(np.dtype(np.int32), 7)

    @pytest.mark.parametrize("dtype",
             ['Bool', 'Bytes0', 'Complex32', 'Complex64',
              'Datetime64', 'Float16', 'Float32', 'Float64',
              'Int8', 'Int16', 'Int32', 'Int64',
              'Object0', 'Str0', 'Timedelta64',
              'UInt8', 'UInt16', 'Uint32', 'UInt32',
              'Uint64', 'UInt64', 'Void0',
              "Float128", "Complex128"])
    def test_numeric_style_types_are_invalid(self, dtype):
        with assert_raises(TypeError):
            np.dtype(dtype)

    def test_expired_dtypes_with_bad_bytesize(self):
        match: str = r".*removed in NumPy 2.0.*"
        with pytest.raises(TypeError, match=match):
            np.dtype("int0")
        with pytest.raises(TypeError, match=match):
            np.dtype("uint0")
        with pytest.raises(TypeError, match=match):
            np.dtype("bool8")
        with pytest.raises(TypeError, match=match):
            np.dtype("bytes0")
        with pytest.raises(TypeError, match=match):
            np.dtype("str0")
        with pytest.raises(TypeError, match=match):
            np.dtype("object0")
        with pytest.raises(TypeError, match=match):
            np.dtype("void0")

    @pytest.mark.parametrize(
        'value',
        ['m8', 'M8', 'datetime64', 'timedelta64',
         'i4, (2,3)f8, f4', 'S3, 3u8, (3,4)S10',
         '>f', '<f', '=f', '|f',
        ])
    def test_dtype_bytes_str_equivalence(self, value):
        bytes_value = value.encode('ascii')
        from_bytes = np.dtype(bytes_value)
        from_str = np.dtype(value)
        assert_dtype_equal(from_bytes, from_str)

    def test_dtype_from_bytes(self):
        # Empty bytes object
        assert_raises(TypeError, np.dtype, b'')
        # Byte order indicator, but no type
        assert_raises(TypeError, np.dtype, b'|')

        # Single character with ordinal < NPY_NTYPES_LEGACY returns
        # type by index into _builtin_descrs
        assert_dtype_equal(np.dtype(bytes([0])), np.dtype('bool'))
        assert_dtype_equal(np.dtype(bytes([17])), np.dtype(object))

        # Single character where value is a valid type code
        assert_dtype_equal(np.dtype(b'f'), np.dtype('float32'))

        # Bytes with non-ascii values raise errors
        assert_raises(TypeError, np.dtype, b'\xff')
        assert_raises(TypeError, np.dtype, b's\xff')

    def test_bad_param(self):
        # Can't give a size that's too small
        assert_raises(ValueError, np.dtype,
                        {'names': ['f0', 'f1'],
                         'formats': ['i4', 'i1'],
                         'offsets': [0, 4],
                         'itemsize': 4})
        # If alignment is enabled, the alignment (4) must divide the itemsize
        assert_raises(ValueError, np.dtype,
                        {'names': ['f0', 'f1'],
                         'formats': ['i4', 'i1'],
                         'offsets': [0, 4],
                         'itemsize': 9}, align=True)
        # If alignment is enabled, the individual fields must be aligned
        assert_raises(ValueError, np.dtype,
                        {'names': ['f0', 'f1'],
                         'formats': ['i1', 'f4'],
                         'offsets': [0, 2]}, align=True)

    def test_field_order_equality(self):
        x = np.dtype({'names': ['A', 'B'],
                      'formats': ['i4', 'f4'],
                      'offsets': [0, 4]})
        y = np.dtype({'names': ['B', 'A'],
                      'formats': ['i4', 'f4'],
                      'offsets': [4, 0]})
        assert_equal(x == y, False)
        # This is an safe cast (not equiv) due to the different names:
        assert np.can_cast(x, y, casting="safe")

    @pytest.mark.parametrize(
        ["type_char", "char_size", "scalar_type"],
        [["U", 4, np.str_],
         ["S", 1, np.bytes_]])
    def test_create_string_dtypes_directly(
            self, type_char, char_size, scalar_type):
        dtype_class = type(np.dtype(type_char))

        dtype = dtype_class(8)
        assert dtype.type is scalar_type
        assert dtype.itemsize == 8 * char_size

    def test_create_invalid_string_errors(self):
        one_too_big = np.iinfo(np.intc).max + 1
        with pytest.raises(TypeError):
            type(np.dtype("U"))(one_too_big // 4)

        with pytest.raises(TypeError):
            # Code coverage for very large numbers:
            type(np.dtype("U"))(np.iinfo(np.intp).max // 4 + 1)

        if one_too_big < sys.maxsize:
            with pytest.raises(TypeError):
                type(np.dtype("S"))(one_too_big)

        with pytest.raises(ValueError):
            type(np.dtype("U"))(-1)

        # OverflowError on 32 bit
        with pytest.raises((TypeError, OverflowError)):
            # see gh-26556
            type(np.dtype("S"))(2**61)

        with pytest.raises(TypeError):
            np.dtype("S1234hello")

    def test_leading_zero_parsing(self):
        dt1 = np.dtype('S010')
        dt2 = np.dtype('S10')

        assert dt1 == dt2
        assert repr(dt1) == "dtype('S10')"
        assert dt1.itemsize == 10


class TestRecord:
    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        assert_dtype_equal(a, b)

    def test_different_names(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype([('yo', int)])
        b = np.dtype([('ye', int)])
        assert_dtype_not_equal(a, b)

    def test_different_titles(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['RRed pixel', 'Blue pixel']})
        assert_dtype_not_equal(a, b)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount_dictionary_setting(self):
        names = ["name1"]
        formats = ["f8"]
        titles = ["t1"]
        offsets = [0]
        d = {"names": names, "formats": formats, "titles": titles, "offsets": offsets}
        refcounts = {k: sys.getrefcount(i) for k, i in d.items()}
        np.dtype(d)
        refcounts_new = {k: sys.getrefcount(i) for k, i in d.items()}
        assert refcounts == refcounts_new

    def test_mutate(self):
        # Mutating a dtype should reset the cached hash value.
        # NOTE: Mutating should be deprecated, but new API added to replace it.
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        c = np.dtype([('ye', int)])
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)
        a.names = ['ye']
        assert_dtype_equal(a, c)
        assert_dtype_not_equal(a, b)
        state = b.__reduce__()[2]
        a.__setstate__(state)
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)

    def test_init_simple_structured(self):
        dt1 = np.dtype("i, i")
        assert dt1.names == ("f0", "f1")

        dt2 = np.dtype("i,")
        assert dt2.names == ("f0",)

    def test_mutate_error(self):
        # NOTE: Mutating should be deprecated, but new API added to replace it.
        a = np.dtype("i,i")

        with pytest.raises(ValueError, match="must replace all names at once"):
            a.names = ["f0"]

        with pytest.raises(ValueError, match=".*and not string"):
            a.names = ["f0", b"not a unicode name"]

    def test_not_lists(self):
        """Test if an appropriate exception is raised when passing bad values to
        the dtype constructor.
        """
        assert_raises(TypeError, np.dtype,
                      {"names": {'A', 'B'}, "formats": ['f8', 'i4']})
        assert_raises(TypeError, np.dtype,
                      {"names": ['A', 'B'], "formats": {'f8', 'i4'}})

    def test_aligned_size(self):
        # Check that structured dtypes get padded to an aligned size
        dt = np.dtype('i4, i1', align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype([('f0', 'i4'), ('f1', 'i1')], align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'names': ['f0', 'f1'],
                       'formats': ['i4', 'u1'],
                       'offsets': [0, 4]}, align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'f0': ('i4', 0), 'f1': ('u1', 4)}, align=True)
        assert_equal(dt.itemsize, 8)
        # Nesting should preserve that alignment
        dt1 = np.dtype([('f0', 'i4'),
                       ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                       ('f2', 'i1')], align=True)
        assert_equal(dt1.itemsize, 20)
        dt2 = np.dtype({'names': ['f0', 'f1', 'f2'],
                       'formats': ['i4',
                                  [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                  'i1'],
                       'offsets': [0, 4, 16]}, align=True)
        assert_equal(dt2.itemsize, 20)
        dt3 = np.dtype({'f0': ('i4', 0),
                       'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                       'f2': ('i1', 16)}, align=True)
        assert_equal(dt3.itemsize, 20)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        # Nesting should preserve packing
        dt1 = np.dtype([('f0', 'i4'),
                       ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                       ('f2', 'i1')], align=False)
        assert_equal(dt1.itemsize, 11)
        dt2 = np.dtype({'names': ['f0', 'f1', 'f2'],
                       'formats': ['i4',
                                  [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                  'i1'],
                       'offsets': [0, 4, 10]}, align=False)
        assert_equal(dt2.itemsize, 11)
        dt3 = np.dtype({'f0': ('i4', 0),
                       'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                       'f2': ('i1', 10)}, align=False)
        assert_equal(dt3.itemsize, 11)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        # Array of subtype should preserve alignment
        dt1 = np.dtype([('a', '|i1'),
                        ('b', [('f0', '<i2'),
                        ('f1', '<f4')], 2)], align=True)
        assert_equal(dt1.descr, [('a', '|i1'), ('', '|V3'),
                                 ('b', [('f0', '<i2'), ('', '|V2'),
                                 ('f1', '<f4')], (2,))])

    def test_empty_struct_alignment(self):
        # Empty dtypes should have an alignment of 1
        dt = np.dtype([], align=True)
        assert_equal(dt.alignment, 1)
        dt = np.dtype([('f0', [])], align=True)
        assert_equal(dt.alignment, 1)
        dt = np.dtype({'names': [],
                       'formats': [],
                       'offsets': []}, align=True)
        assert_equal(dt.alignment, 1)
        dt = np.dtype({'names': ['f0'],
                       'formats': [[]],
                       'offsets': [0]}, align=True)
        assert_equal(dt.alignment, 1)

    def test_union_struct(self):
        # Should be able to create union dtypes
        dt = np.dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['<u4', '<u2', '<u2'],
                        'offsets': [0, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 4)
        a = np.array([3], dtype='<u4').view(dt)
        a['f1'] = 10
        a['f2'] = 36
        assert_equal(a['f0'], 10 + 36 * 256 * 256)
        # Should be able to specify fields out of order
        dt = np.dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['<u4', '<u2', '<u2'],
                        'offsets': [4, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 8)
        # field name should not matter: assignment is by position
        dt2 = np.dtype({'names': ['f2', 'f0', 'f1'],
                        'formats': ['<u4', '<u2', '<u2'],
                        'offsets': [4, 0, 2]}, align=True)
        vals = [(0, 1, 2), (3, 2**15 - 1, 4)]
        vals2 = [(0, 1, 2), (3, 2**15 - 1, 4)]
        a = np.array(vals, dt)
        b = np.array(vals2, dt2)
        assert_equal(a.astype(dt2), b)
        assert_equal(b.astype(dt), a)
        assert_equal(a.view(dt2), b)
        assert_equal(b.view(dt), a)
        # Should not be able to overlap objects with other types
        assert_raises(TypeError, np.dtype,
                {'names': ['f0', 'f1'],
                 'formats': ['O', 'i1'],
                 'offsets': [0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names': ['f0', 'f1'],
                 'formats': ['i4', 'O'],
                 'offsets': [0, 3]})
        assert_raises(TypeError, np.dtype,
                {'names': ['f0', 'f1'],
                 'formats': [[('a', 'O')], 'i1'],
                 'offsets': [0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names': ['f0', 'f1'],
                 'formats': ['i4', [('a', 'O')]],
                 'offsets': [0, 3]})
        # Out of order should still be ok, however
        dt = np.dtype({'names': ['f0', 'f1'],
                       'formats': ['i1', 'O'],
                       'offsets': [np.dtype('intp').itemsize, 0]})

    @pytest.mark.parametrize(["obj", "dtype", "expected"],
        [([], ("2f4"), np.empty((0, 2), dtype="f4")),
         (3, "(3,)f4", [3, 3, 3]),
         (np.float64(2), "(2,)f4", [2, 2]),
         ([((0, 1), (1, 2)), ((2,),)], '(2,2)f4', None),
         (["1", "2"], "2i", None)])
    def test_subarray_list(self, obj, dtype, expected):
        dtype = np.dtype(dtype)
        res = np.array(obj, dtype=dtype)

        if expected is None:
            # iterate the 1-d list to fill the array
            expected = np.empty(len(obj), dtype=dtype)
            for i in range(len(expected)):
                expected[i] = obj[i]

        assert_array_equal(res, expected)

    def test_parenthesized_single_number(self):
        with pytest.raises(TypeError, match="not understood"):
            np.dtype("(2)f4")

        # Deprecation also tested in
        # test_deprecations.py::TestDeprecatedDTypeParenthesizedRepeatCount
        # Left here to allow easy conversion to an exception check.
        with pytest.warns(DeprecationWarning,
                          match="parenthesized single number"):
            np.dtype("(2)f4,")

    def test_comma_datetime(self):
        dt = np.dtype('M8[D],datetime64[Y],i8')
        assert_equal(dt, np.dtype([('f0', 'M8[D]'),
                                   ('f1', 'datetime64[Y]'),
                                   ('f2', 'i8')]))

    def test_from_dictproxy(self):
        # Tests for PR #5920
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'f4']})
        assert_dtype_equal(dt, np.dtype(dt.fields))
        dt2 = np.dtype((np.void, dt.fields))
        assert_equal(dt2.fields, dt.fields)

    def test_from_dict_with_zero_width_field(self):
        # Regression test for #6430 / #2196
        dt = np.dtype([('val1', np.float32, (0,)), ('val2', int)])
        dt2 = np.dtype({'names': ['val1', 'val2'],
                        'formats': [(np.float32, (0,)), int]})

        assert_dtype_equal(dt, dt2)
        assert_equal(dt.fields['val1'][0].itemsize, 0)
        assert_equal(dt.itemsize, dt.fields['val2'][0].itemsize)

    def test_bool_commastring(self):
        d = np.dtype('?,?,?')  # raises?
        assert_equal(len(d.names), 3)
        for n in d.names:
            assert_equal(d.fields[n][0], np.dtype('?'))

    def test_nonint_offsets(self):
        # gh-8059
        def make_dtype(off):
            return np.dtype({'names': ['A'], 'formats': ['i4'],
                             'offsets': [off]})

        assert_raises(TypeError, make_dtype, 'ASD')
        assert_raises(OverflowError, make_dtype, 2**70)
        assert_raises(TypeError, make_dtype, 2.3)
        assert_raises(ValueError, make_dtype, -10)

        # no errors here:
        dt = make_dtype(np.uint32(0))
        np.zeros(1, dtype=dt)[0].item()

    def test_fields_by_index(self):
        dt = np.dtype([('a', np.int8), ('b', np.float32, 3)])
        assert_dtype_equal(dt[0], np.dtype(np.int8))
        assert_dtype_equal(dt[1], np.dtype((np.float32, 3)))
        assert_dtype_equal(dt[-1], dt[1])
        assert_dtype_equal(dt[-2], dt[0])
        assert_raises(IndexError, lambda: dt[-3])

        assert_raises(TypeError, operator.getitem, dt, 3.0)

        assert_equal(dt[1], dt[np.int8(1)])

    @pytest.mark.parametrize('align_flag', [False, True])
    def test_multifield_index(self, align_flag):
        # indexing with a list produces subfields
        # the align flag should be preserved
        dt = np.dtype([
            (('title', 'col1'), '<U20'), ('A', '<f8'), ('B', '<f8')
        ], align=align_flag)

        dt_sub = dt[['B', 'col1']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B', 'col1'],
                'formats': ['<f8', '<U20'],
                'offsets': [88, 0],
                'titles': [None, 'title'],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        dt_sub = dt[['B']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B'],
                'formats': ['<f8'],
                'offsets': [88],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        dt_sub = dt[[]]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': [],
                'formats': [],
                'offsets': [],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        assert_raises(TypeError, operator.getitem, dt, ())
        assert_raises(TypeError, operator.getitem, dt, [1, 2, 3])
        assert_raises(TypeError, operator.getitem, dt, ['col1', 2])
        assert_raises(KeyError, operator.getitem, dt, ['fake'])
        assert_raises(KeyError, operator.getitem, dt, ['title'])
        assert_raises(ValueError, operator.getitem, dt, ['col1', 'col1'])

    def test_partial_dict(self):
        # 'names' is missing
        assert_raises(ValueError, np.dtype,
                {'formats': ['i4', 'i4'], 'f0': ('i4', 0), 'f1': ('i4', 4)})

    def test_fieldless_views(self):
        a = np.zeros(2, dtype={'names': [], 'formats': [], 'offsets': [],
                               'itemsize': 8})
        assert_raises(ValueError, a.view, np.dtype([]))

        d = np.dtype((np.dtype([]), 10))
        assert_equal(d.shape, (10,))
        assert_equal(d.itemsize, 0)
        assert_equal(d.base, np.dtype([]))

        arr = np.fromiter((() for i in range(10)), [])
        assert_equal(arr.dtype, np.dtype([]))
        assert_raises(ValueError, np.frombuffer, b'', dtype=[])
        assert_equal(np.frombuffer(b'', dtype=[], count=2),
                     np.empty(2, dtype=[]))

        assert_raises(ValueError, np.dtype, ([], 'f8'))
        assert_raises(ValueError, np.zeros(1, dtype='i4').view, [])

        assert_equal(np.zeros(2, dtype=[]) == np.zeros(2, dtype=[]),
                     np.ones(2, dtype=bool))

        assert_equal(np.zeros((1, 2), dtype=[]) == a,
                     np.ones((1, 2), dtype=bool))

    def test_nonstructured_with_object(self):
        # See gh-23277, the dtype here thinks it contain objects, if the
        # assert about that fails, the test becomes meaningless (which is OK)
        arr = np.recarray((0,), dtype="O")
        assert arr.dtype.names is None  # no fields
        assert arr.dtype.hasobject  # but claims to contain objects
        del arr  # the deletion failed previously.


class TestSubarray:
    def test_single_subarray(self):
        a = np.dtype((int, (2)))
        b = np.dtype((int, (2,)))
        assert_dtype_equal(a, b)

        assert_equal(type(a.subdtype[1]), tuple)
        assert_equal(type(b.subdtype[1]), tuple)

    def test_equivalent_record(self):
        """Test whether equivalent subarray dtypes hash the same."""
        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (2, 3)))
        assert_dtype_equal(a, b)

    def test_nonequivalent_record(self):
        """Test whether different subarray dtypes hash differently."""
        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (3, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (2, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((int, (1, 2, 3)))
        b = np.dtype((int, (1, 2)))
        assert_dtype_not_equal(a, b)

    def test_shape_equal(self):
        """Test some data types that are equal"""
        assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', ())))
        assert_dtype_equal(np.dtype('(1,)f8'), np.dtype(('f8', 1)))
        assert np.dtype(('f8', 1)).shape == (1,)
        assert_dtype_equal(np.dtype((int, 2)), np.dtype((int, (2,))))
        assert_dtype_equal(np.dtype(('<f4', (3, 2))), np.dtype(('<f4', (3, 2))))
        d = ([('a', 'f4', (1, 2)), ('b', 'f8', (3, 1))], (3, 2))
        assert_dtype_equal(np.dtype(d), np.dtype(d))

    def test_shape_simple(self):
        """Test some simple cases that shouldn't be equal"""
        assert_dtype_not_equal(np.dtype('f8'), np.dtype(('f8', (1,))))
        assert_dtype_not_equal(np.dtype(('f8', (1,))), np.dtype(('f8', (1, 1))))
        assert_dtype_not_equal(np.dtype(('f4', (3, 2))), np.dtype(('f4', (2, 3))))

    def test_shape_monster(self):
        """Test some more complicated cases that shouldn't be equal"""
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (1, 2)), ('b', 'f8', (1, 3))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'i8', (1, 3))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('e', 'f8', (1, 3)), ('d', 'f4', (2, 1))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', [('a', 'i4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', [('a', 'u4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))))

    def test_shape_sequence(self):
        # Any sequence of integers should work as shape, but the result
        # should be a tuple (immutable) of base type integers.
        a = np.array([1, 2, 3], dtype=np.int16)
        l = [1, 2, 3]
        # Array gets converted
        dt = np.dtype([('a', 'f4', a)])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))
        # List gets converted
        dt = np.dtype([('a', 'f4', l)])
        assert_(isinstance(dt['a'].shape, tuple))
        #

        class IntLike:
            def __index__(self):
                return 3

            def __int__(self):
                # (a PyNumber_Check fails without __int__)
                return 3

        dt = np.dtype([('a', 'f4', IntLike())])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))
        dt = np.dtype([('a', 'f4', (IntLike(),))])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))

    def test_shape_matches_ndim(self):
        dt = np.dtype([('a', 'f4', ())])
        assert_equal(dt['a'].shape, ())
        assert_equal(dt['a'].ndim, 0)

        dt = np.dtype([('a', 'f4')])
        assert_equal(dt['a'].shape, ())
        assert_equal(dt['a'].ndim, 0)

        dt = np.dtype([('a', 'f4', 4)])
        assert_equal(dt['a'].shape, (4,))
        assert_equal(dt['a'].ndim, 1)

        dt = np.dtype([('a', 'f4', (1, 2, 3))])
        assert_equal(dt['a'].shape, (1, 2, 3))
        assert_equal(dt['a'].ndim, 3)

    def test_shape_invalid(self):
        # Check that the shape is valid.
        max_int = np.iinfo(np.intc).max
        max_intp = np.iinfo(np.intp).max
        # Too large values (the datatype is part of this)
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int // 4 + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (max_int, 2))])
        # Takes a different code path (fails earlier:
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_intp + 1)])
        # Negative values
        assert_raises(ValueError, np.dtype, [('a', 'f4', -1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (-1, -1))])

    def test_alignment(self):
        # Check that subarrays are aligned
        t1 = np.dtype('(1,)i4', align=True)
        t2 = np.dtype('2i4', align=True)
        assert_equal(t1.alignment, t2.alignment)

    def test_aligned_empty(self):
        # Mainly regression test for gh-19696: construction failed completely
        dt = np.dtype([], align=True)
        assert dt == np.dtype([])
        dt = np.dtype({"names": [], "formats": [], "itemsize": 0}, align=True)
        assert dt == np.dtype([])

    def test_subarray_base_item(self):
        arr = np.ones(3, dtype=[("f", "i", 3)])
        # Extracting the field "absorbs" the subarray into a view:
        assert arr["f"].base is arr
        # Extract the structured item, and then check the tuple component:
        item = arr.item(0)
        assert type(item) is tuple and len(item) == 1
        assert item[0].base is arr

    def test_subarray_cast_copies(self):
        # Older versions of NumPy did NOT copy, but they got the ownership
        # wrong (not actually knowing the correct base!).  Versions since 1.21
        # (I think) crashed fairly reliable.  This defines the correct behavior
        # as a copy.  Keeping the ownership would be possible (but harder)
        arr = np.ones(3, dtype=[("f", "i", 3)])
        cast = arr.astype(object)
        for fields in cast:
            assert type(fields) == tuple and len(fields) == 1
            subarr = fields[0]
            assert subarr.base is None
            assert subarr.flags.owndata


def iter_struct_object_dtypes():
    """
    Iterates over a few complex dtypes and object pattern which
    fill the array with a given object (defaults to a singleton).

    Yields
    ------
    dtype : dtype
    pattern : tuple
        Structured tuple for use with `np.array`.
    count : int
        Number of objects stored in the dtype.
    singleton : object
        A singleton object. The returned pattern is constructed so that
        all objects inside the datatype are set to the singleton.
    """
    obj = object()

    dt = np.dtype([('b', 'O', (2, 3))])
    p = ([[obj] * 3] * 2,)
    yield pytest.param(dt, p, 6, obj, id="<subarray>")

    dt = np.dtype([('a', 'i4'), ('b', 'O', (2, 3))])
    p = (0, [[obj] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<subarray in field>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'i1')], (2, 3))])
    p = (0, [[(obj, 0)] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<structured subarray 1>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'O')], (2, 3))])
    p = (0, [[(obj, obj)] * 3] * 2)
    yield pytest.param(dt, p, 12, obj, id="<structured subarray 2>")


@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="Python 3.12 has immortal refcounts, this test will no longer "
           "work. See gh-23986"
)
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
class TestStructuredObjectRefcounting:
    """These tests cover various uses of complicated structured types which
    include objects and thus require reference counting.
    """
    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(["creation_func", "creation_obj"], [
        pytest.param(np.empty, None,
             # None is probably used for too many things
             marks=pytest.mark.skip("unreliable due to python's behaviour")),
        (np.ones, 1),
        (np.zeros, 0)])
    def test_structured_object_create_delete(self, dt, pat, count, singleton,
                                             creation_func, creation_obj):
        """Structured object reference counting in creation and deletion"""
        # The test assumes that 0, 1, and None are singletons.
        gc.collect()
        before = sys.getrefcount(creation_obj)
        arr = creation_func(3, dt)

        now = sys.getrefcount(creation_obj)
        assert now - before == count * 3
        del arr
        now = sys.getrefcount(creation_obj)
        assert now == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    def test_structured_object_item_setting(self, dt, pat, count, singleton):
        """Structured object reference counting for simple item setting"""
        one = 1

        gc.collect()
        before = sys.getrefcount(singleton)
        arr = np.array([pat] * 3, dt)
        assert sys.getrefcount(singleton) - before == count * 3
        # Fill with `1` and check that it was replaced correctly:
        before2 = sys.getrefcount(one)
        arr[...] = one
        after2 = sys.getrefcount(one)
        assert after2 - before2 == count * 3
        del arr
        gc.collect()
        assert sys.getrefcount(one) == before2
        assert sys.getrefcount(singleton) == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(
        ['shape', 'index', 'items_changed'],
        [((3,), ([0, 2],), 2),
         ((3, 2), ([0, 2], slice(None)), 4),
         ((3, 2), ([0, 2], [1]), 2),
         ((3,), ([True, False, True]), 2)])
    def test_structured_object_indexing(self, shape, index, items_changed,
                                        dt, pat, count, singleton):
        """Structured object reference counting for advanced indexing."""
        # Use two small negative values (should be singletons, but less likely
        # to run into race-conditions).  This failed in some threaded envs
        # When using 0 and 1.  If it fails again, should remove all explicit
        # checks, and rely on `pytest-leaks` reference count checker only.
        val0 = -4
        val1 = -5

        arr = np.full(shape, val0, dt)

        gc.collect()
        before_val0 = sys.getrefcount(val0)
        before_val1 = sys.getrefcount(val1)
        # Test item getting:
        part = arr[index]
        after_val0 = sys.getrefcount(val0)
        assert after_val0 - before_val0 == count * items_changed
        del part
        # Test item setting:
        arr[index] = val1
        gc.collect()
        after_val0 = sys.getrefcount(val0)
        after_val1 = sys.getrefcount(val1)
        assert before_val0 - after_val0 == count * items_changed
        assert after_val1 - before_val1 == count * items_changed

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    def test_structured_object_take_and_repeat(self, dt, pat, count, singleton):
        """Structured object reference counting for specialized functions.
        The older functions such as take and repeat use different code paths
        then item setting (when writing this).
        """
        indices = [0, 1]

        arr = np.array([pat] * 3, dt)
        gc.collect()
        before = sys.getrefcount(singleton)
        res = arr.take(indices)
        after = sys.getrefcount(singleton)
        assert after - before == count * 2
        new = res.repeat(10)
        gc.collect()
        after_repeat = sys.getrefcount(singleton)
        assert after_repeat - after == count * 2 * 10


class TestStructuredDtypeSparseFields:
    """Tests subarray fields which contain sparse dtypes so that
    not all memory is used by the dtype work. Such dtype's should
    leave the underlying memory unchanged.
    """
    dtype = np.dtype([('a', {'names': ['aa', 'ab'], 'formats': ['f', 'f'],
                             'offsets': [0, 4]}, (2, 3))])
    sparse_dtype = np.dtype([('a', {'names': ['ab'], 'formats': ['f'],
                                    'offsets': [4]}, (2, 3))])

    def test_sparse_field_assignment(self):
        arr = np.zeros(3, self.dtype)
        sparse_arr = arr.view(self.sparse_dtype)

        sparse_arr[...] = np.finfo(np.float32).max
        # dtype is reduced when accessing the field, so shape is (3, 2, 3):
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))

    def test_sparse_field_assignment_fancy(self):
        # Fancy assignment goes to the copyswap function for complex types:
        arr = np.zeros(3, self.dtype)
        sparse_arr = arr.view(self.sparse_dtype)

        sparse_arr[[0, 1, 2]] = np.finfo(np.float32).max
        # dtype is reduced when accessing the field, so shape is (3, 2, 3):
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))


class TestMonsterType:
    """Test deeply nested subtypes."""

    def test1(self):
        simple1 = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        a = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((int, (3, 2))))])
        b = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((int, (3, 2))))])
        assert_dtype_equal(a, b)

        c = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        d = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        assert_dtype_equal(c, d)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    @pytest.mark.skipif(IS_WASM, reason="Pyodide/WASM has limited stack size")
    def test_list_recursion(self):
        l = []
        l.append(('f', l))
        with pytest.raises(RecursionError):
            np.dtype(l)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    @pytest.mark.skipif(IS_WASM, reason="Pyodide/WASM has limited stack size")
    def test_tuple_recursion(self):
        d = np.int32
        for i in range(100000):
            d = (d, (1,))
        with pytest.raises(RecursionError):
            np.dtype(d)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    @pytest.mark.skipif(IS_WASM, reason="Pyodide/WASM has limited stack size")
    def test_dict_recursion(self):
        d = {"names": ['self'], "formats": [None], "offsets": [0]}
        d['formats'][0] = d
        with pytest.raises(RecursionError):
            np.dtype(d)


class TestMetadata:
    def test_no_metadata(self):
        d = np.dtype(int)
        assert_(d.metadata is None)

    def test_metadata_takes_dict(self):
        d = np.dtype(int, metadata={'datum': 1})
        assert_(d.metadata == {'datum': 1})

    def test_metadata_rejects_nondict(self):
        assert_raises(TypeError, np.dtype, int, metadata='datum')
        assert_raises(TypeError, np.dtype, int, metadata=1)
        assert_raises(TypeError, np.dtype, int, metadata=None)

    def test_nested_metadata(self):
        d = np.dtype([('a', np.dtype(int, metadata={'datum': 1}))])
        assert_(d['a'].metadata == {'datum': 1})

    def test_base_metadata_copied(self):
        d = np.dtype((np.void, np.dtype('i4,i4', metadata={'datum': 1})))
        assert_(d.metadata == {'datum': 1})

class TestString:
    def test_complex_dtype_str(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(str(dt),
                     "[('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])]")

        # If the sticky aligned flag is set to True, it makes the
        # str() function use a dict representation with an 'aligned' flag
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))],
                                (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])],
                       align=True)
        assert_equal(str(dt),
                    "{'names': ['top', 'bottom'],"
                    " 'formats': [([('tiles', ('>f4', (64, 64)), (1,)), "
                                   "('rtile', '>f4', (64, 36))], (3,)), "
                                  "[('bleft', ('>f4', (8, 64)), (1,)), "
                                   "('bright', '>f4', (8, 36))]],"
                    " 'offsets': [0, 76800],"
                    " 'itemsize': 80000,"
                    " 'aligned': True}")
        with np.printoptions(legacy='1.21'):
            assert_equal(str(dt),
                        "{'names':['top','bottom'], "
                         "'formats':[([('tiles', ('>f4', (64, 64)), (1,)), "
                                      "('rtile', '>f4', (64, 36))], (3,)),"
                                     "[('bleft', ('>f4', (8, 64)), (1,)), "
                                      "('bright', '>f4', (8, 36))]], "
                         "'offsets':[0,76800], "
                         "'itemsize':80000, "
                         "'aligned':True}")
        assert_equal(np.dtype(eval(str(dt))), dt)

        dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "[(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')]")

        dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'],
                       'formats': ['<u4', 'u1', 'u1', 'u1'],
                       'offsets': [0, 0, 1, 2],
                       'titles': ['Color', 'Red pixel',
                                  'Green pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "{'names': ['rgba', 'r', 'g', 'b'],"
                    " 'formats': ['<u4', 'u1', 'u1', 'u1'],"
                    " 'offsets': [0, 0, 1, 2],"
                    " 'titles': ['Color', 'Red pixel', "
                               "'Green pixel', 'Blue pixel'],"
                    " 'itemsize': 4}")

        dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "{'names': ['r', 'b'],"
                    " 'formats': ['u1', 'u1'],"
                    " 'offsets': [0, 2],"
                    " 'titles': ['Red pixel', 'Blue pixel'],"
                    " 'itemsize': 3}")

        dt = np.dtype([('a', '<m8[D]'), ('b', '<M8[us]')])
        assert_equal(str(dt),
                    "[('a', '<m8[D]'), ('b', '<M8[us]')]")

    def test_repr_structured(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(repr(dt),
                     "dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])])")

        dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']},
                        align=True)
        assert_equal(repr(dt),
                    "dtype([(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')], align=True)")

    def test_repr_structured_not_packed(self):
        dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'],
                       'formats': ['<u4', 'u1', 'u1', 'u1'],
                       'offsets': [0, 0, 1, 2],
                       'titles': ['Color', 'Red pixel',
                                  'Green pixel', 'Blue pixel']}, align=True)
        assert_equal(repr(dt),
                    "dtype({'names': ['rgba', 'r', 'g', 'b'],"
                    " 'formats': ['<u4', 'u1', 'u1', 'u1'],"
                    " 'offsets': [0, 0, 1, 2],"
                    " 'titles': ['Color', 'Red pixel', "
                                "'Green pixel', 'Blue pixel'],"
                    " 'itemsize': 4}, align=True)")

        dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel'],
                        'itemsize': 4})
        assert_equal(repr(dt),
                    "dtype({'names': ['r', 'b'], "
                    "'formats': ['u1', 'u1'], "
                    "'offsets': [0, 2], "
                    "'titles': ['Red pixel', 'Blue pixel'], "
                    "'itemsize': 4})")

    def test_repr_structured_datetime(self):
        dt = np.dtype([('a', '<M8[D]'), ('b', '<m8[us]')])
        assert_equal(repr(dt),
                    "dtype([('a', '<M8[D]'), ('b', '<m8[us]')])")

    def test_repr_str_subarray(self):
        dt = np.dtype(('<i2', (1,)))
        assert_equal(repr(dt), "dtype(('<i2', (1,)))")
        assert_equal(str(dt), "('<i2', (1,))")

    def test_base_dtype_with_object_type(self):
        # Issue gh-2798, should not error.
        np.array(['a'], dtype="O").astype(("O", [("name", "O")]))

    def test_empty_string_to_object(self):
        # Pull request #4722
        np.array(["", ""]).astype(object)

    def test_void_subclass_unsized(self):
        dt = np.dtype(np.record)
        assert_equal(repr(dt), "dtype('V')")
        assert_equal(str(dt), '|V0')
        assert_equal(dt.name, 'record')

    def test_void_subclass_sized(self):
        dt = np.dtype((np.record, 2))
        assert_equal(repr(dt), "dtype('V2')")
        assert_equal(str(dt), '|V2')
        assert_equal(dt.name, 'record16')

    def test_void_subclass_fields(self):
        dt = np.dtype((np.record, [('a', '<u2')]))
        assert_equal(repr(dt), "dtype((numpy.record, [('a', '<u2')]))")
        assert_equal(str(dt), "(numpy.record, [('a', '<u2')])")
        assert_equal(dt.name, 'record16')

    def test_custom_dtype_str(self):
        dt = np.dtypes.StringDType()
        assert_equal(dt.str, "StringDType()")


class TestDtypeAttributeDeletion:

    def test_dtype_non_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["subdtype", "descr", "str", "name", "base", "shape",
                "isbuiltin", "isnative", "isalignedstruct", "fields",
                "metadata", "hasobject"]

        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)

    def test_dtype_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["names"]
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)


class TestDtypeAttributes:
    def test_descr_has_trailing_void(self):
        # see gh-6359
        dtype = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        new_dtype = np.dtype(dtype.descr)
        assert_equal(new_dtype.itemsize, 16)

    def test_name_dtype_subclass(self):
        # Ticket #4357
        class user_def_subcls(np.void):
            pass
        assert_equal(np.dtype(user_def_subcls).name, 'user_def_subcls')

    def test_zero_stride(self):
        arr = np.ones(1, dtype="i8")
        arr = np.broadcast_to(arr, 10)
        assert arr.strides == (0,)
        with pytest.raises(ValueError):
            arr.dtype = "i1"

class TestDTypeMakeCanonical:
    def check_canonical(self, dtype, canonical):
        """
        Check most properties relevant to "canonical" versions of a dtype,
        which is mainly native byte order for datatypes supporting this.

        The main work is checking structured dtypes with fields, where we
        reproduce most the actual logic used in the C-code.
        """
        assert type(dtype) is type(canonical)

        # a canonical DType should always have equivalent casting (both ways)
        assert np.can_cast(dtype, canonical, casting="equiv")
        assert np.can_cast(canonical, dtype, casting="equiv")
        # a canonical dtype (and its fields) is always native (checks fields):
        assert canonical.isnative

        # Check that canonical of canonical is the same (no casting):
        assert np.result_type(canonical) == canonical

        if not dtype.names:
            # The flags currently never change for unstructured dtypes
            assert dtype.flags == canonical.flags
            return

        # Must have all the needs API flag set:
        assert dtype.flags & 0b10000

        # Check that the fields are identical (including titles):
        assert dtype.fields.keys() == canonical.fields.keys()

        def aligned_offset(offset, alignment):
            # round up offset:
            return - (-offset // alignment) * alignment

        totalsize = 0
        max_alignment = 1
        for name in dtype.names:
            # each field is also canonical:
            new_field_descr = canonical.fields[name][0]
            self.check_canonical(dtype.fields[name][0], new_field_descr)

            # Must have the "inherited" object related flags:
            expected = 0b11011 & new_field_descr.flags
            assert (canonical.flags & expected) == expected

            if canonical.isalignedstruct:
                totalsize = aligned_offset(totalsize, new_field_descr.alignment)
                max_alignment = max(new_field_descr.alignment, max_alignment)

            assert canonical.fields[name][1] == totalsize
            # if a title exists, they must match (otherwise empty tuple):
            assert dtype.fields[name][2:] == canonical.fields[name][2:]

            totalsize += new_field_descr.itemsize

        if canonical.isalignedstruct:
            totalsize = aligned_offset(totalsize, max_alignment)
        assert canonical.itemsize == totalsize
        assert canonical.alignment == max_alignment

    def test_simple(self):
        dt = np.dtype(">i4")
        assert np.result_type(dt).isnative
        assert np.result_type(dt).num == dt.num

        # dtype with empty space:
        struct_dt = np.dtype(">i4,<i1,i8,V3")[["f0", "f2"]]
        canonical = np.result_type(struct_dt)
        assert canonical.itemsize == 4 + 8
        assert canonical.isnative

        # aligned struct dtype with empty space:
        struct_dt = np.dtype(">i1,<i4,i8,V3", align=True)[["f0", "f2"]]
        canonical = np.result_type(struct_dt)
        assert canonical.isalignedstruct
        assert canonical.itemsize == np.dtype("i8").alignment + 8
        assert canonical.isnative

    def test_object_flag_not_inherited(self):
        # The following dtype still indicates "object", because its included
        # in the unaccessible space (maybe this could change at some point):
        arr = np.ones(3, "i,O,i")[["f0", "f2"]]
        assert arr.dtype.hasobject
        canonical_dt = np.result_type(arr.dtype)
        assert not canonical_dt.hasobject

    @pytest.mark.slow
    @hypothesis.given(dtype=hynp.nested_dtypes())
    def test_make_canonical_hypothesis(self, dtype):
        canonical = np.result_type(dtype)
        self.check_canonical(dtype, canonical)
        # result_type with two arguments should always give identical results:
        two_arg_result = np.result_type(dtype, dtype)
        assert np.can_cast(two_arg_result, canonical, casting="no")

    @pytest.mark.slow
    @hypothesis.given(
            dtype=hypothesis.extra.numpy.array_dtypes(
                subtype_strategy=hypothesis.extra.numpy.array_dtypes(),
                min_size=5, max_size=10, allow_subarrays=True))
    def test_structured(self, dtype):
        # Pick 4 of the fields at random.  This will leave empty space in the
        # dtype (since we do not canonicalize it here).
        field_subset = random.sample(dtype.names, k=4)
        dtype_with_empty_space = dtype[field_subset]
        assert dtype_with_empty_space.itemsize == dtype.itemsize
        canonicalized = np.result_type(dtype_with_empty_space)
        self.check_canonical(dtype_with_empty_space, canonicalized)
        # promotion with two arguments should always give identical results:
        two_arg_result = np.promote_types(
                dtype_with_empty_space, dtype_with_empty_space)
        assert np.can_cast(two_arg_result, canonicalized, casting="no")

        # Ensure that we also check aligned struct (check the opposite, in
        # case hypothesis grows support for `align`.  Then repeat the test:
        dtype_aligned = np.dtype(dtype.descr, align=not dtype.isalignedstruct)
        dtype_with_empty_space = dtype_aligned[field_subset]
        assert dtype_with_empty_space.itemsize == dtype_aligned.itemsize
        canonicalized = np.result_type(dtype_with_empty_space)
        self.check_canonical(dtype_with_empty_space, canonicalized)
        # promotion with two arguments should always give identical results:
        two_arg_result = np.promote_types(
            dtype_with_empty_space, dtype_with_empty_space)
        assert np.can_cast(two_arg_result, canonicalized, casting="no")


class TestPickling:

    def check_pickling(self, dtype):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            buf = pickle.dumps(dtype, proto)
            # The dtype pickling itself pickles `np.dtype` if it is pickled
            # as a singleton `dtype` should be stored in the buffer:
            assert b"_DType_reconstruct" not in buf
            assert b"dtype" in buf
            pickled = pickle.loads(buf)
            assert_equal(pickled, dtype)
            assert_equal(pickled.descr, dtype.descr)
            if dtype.metadata is not None:
                assert_equal(pickled.metadata, dtype.metadata)
            # Check the reconstructed dtype is functional
            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object,
                                   bool])
    def test_builtin(self, t):
        self.check_pickling(np.dtype(t))

    def test_structured(self):
        dt = np.dtype(([('a', '>f4', (2, 1)), ('b', '<f8', (1, 3))], (2, 2)))
        self.check_pickling(dt)

    def test_structured_aligned(self):
        dt = np.dtype('i4, i1', align=True)
        self.check_pickling(dt)

    def test_structured_unaligned(self):
        dt = np.dtype('i4, i1', align=False)
        self.check_pickling(dt)

    def test_structured_padded(self):
        dt = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        self.check_pickling(dt)

    def test_structured_titles(self):
        dt = np.dtype({'names': ['r', 'b'],
                       'formats': ['u1', 'u1'],
                       'titles': ['Red pixel', 'Blue pixel']})
        self.check_pickling(dt)

    @pytest.mark.parametrize('base', ['m8', 'M8'])
    @pytest.mark.parametrize('unit', ['', 'Y', 'M', 'W', 'D', 'h', 'm', 's',
                                      'ms', 'us', 'ns', 'ps', 'fs', 'as'])
    def test_datetime(self, base, unit):
        dt = np.dtype(f'{base}[{unit}]' if unit else base)
        self.check_pickling(dt)
        if unit:
            dt = np.dtype(f'{base}[7{unit}]')
            self.check_pickling(dt)

    def test_metadata(self):
        dt = np.dtype(int, metadata={'datum': 1})
        self.check_pickling(dt)

    @pytest.mark.parametrize("DType",
        [type(np.dtype(t)) for t in np.typecodes['All']] +
        [type(np.dtype(rational)), np.dtype])
    def test_pickle_dtype_class(self, DType):
        # Check that DTypes (the classes/types) roundtrip when pickling
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
            assert roundtrip_DType is DType

    @pytest.mark.parametrize("dt",
        [np.dtype(t) for t in np.typecodes['All']] +
        [np.dtype(rational)])
    def test_pickle_dtype(self, dt):
        # Check that dtype instances roundtrip when pickling and that pickling
        # doesn't change the hash value
        pre_pickle_hash = hash(dt)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_dt = pickle.loads(pickle.dumps(dt, proto))
            assert roundtrip_dt == dt
            assert hash(dt) == pre_pickle_hash


class TestPromotion:
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """
    @pytest.mark.parametrize(["other", "expected"],
            [(2**16 - 1, np.complex64),
             (2**32 - 1, np.complex64),
             (np.float16(2), np.complex64),
             (np.float32(2), np.complex64),
             (np.longdouble(2), np.clongdouble),
             # Base of the double value to sidestep any rounding issues:
             (np.longdouble(np.nextafter(1.7e308, 0.)), np.clongdouble),
             # Additionally use "nextafter" so the cast can't round down:
             (np.longdouble(np.nextafter(1.7e308, np.inf)), np.clongdouble),
             # repeat for complex scalars:
             (np.complex64(2), np.complex64),
             (np.clongdouble(2), np.clongdouble),
             # Base of the double value to sidestep any rounding issues:
             (np.clongdouble(np.nextafter(1.7e308, 0.) * 1j), np.clongdouble),
             # Additionally use "nextafter" so the cast can't round down:
             (np.clongdouble(np.nextafter(1.7e308, np.inf)), np.clongdouble),
             ])
    def test_complex_other_value_based(self, other, expected):
        # This would change if we modify the value based promotion
        min_complex = np.dtype(np.complex64)

        res = np.result_type(other, min_complex)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        assert res == expected

    @pytest.mark.parametrize(["other", "expected"],
                 [(np.bool, np.complex128),
                  (np.int64, np.complex128),
                  (np.float16, np.complex64),
                  (np.float32, np.complex64),
                  (np.float64, np.complex128),
                  (np.longdouble, np.clongdouble),
                  (np.complex64, np.complex64),
                  (np.complex128, np.complex128),
                  (np.clongdouble, np.clongdouble),
                  ])
    def test_complex_scalar_value_based(self, other, expected):
        # This would change if we modify the value based promotion
        complex_scalar = 1j

        res = np.result_type(other, complex_scalar)
        assert res == expected
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        assert res == expected

    def test_complex_pyscalar_promote_rational(self):
        with pytest.raises(TypeError,
                match=r".* no common DType exists for the given inputs"):
            np.result_type(1j, rational)

        with pytest.raises(TypeError,
                match=r".* no common DType exists for the given inputs"):
            np.result_type(1j, rational(1, 2))

    @pytest.mark.parametrize("val", [2, 2**32, 2**63, 2**64, 2 * 100])
    def test_python_integer_promotion(self, val):
        # If we only pass scalars (mainly python ones!), NEP 50 means
        # that we get the default integer
        expected_dtype = np.dtype(int)  # the default integer
        assert np.result_type(val, 0) == expected_dtype
        # With NEP 50, the NumPy scalar wins though:
        assert np.result_type(val, np.int8(0)) == np.int8

    @pytest.mark.parametrize(["other", "expected"],
            [(1, rational), (1., np.float64)])
    def test_float_int_pyscalar_promote_rational(self, other, expected):
        # Note that rationals are a bit awkward as they promote with float64
        # or default ints, but not float16 or uint8/int8 (which looks
        # inconsistent here).  The new promotion fixed this (partially?)
        assert np.result_type(other, rational) == expected
        assert np.result_type(other, rational(1, 2)) == expected

    @pytest.mark.parametrize(["dtypes", "expected"], [
             # These promotions are not associative/commutative:
             ([np.uint16, np.int16, np.float16], np.float32),
             ([np.uint16, np.int8, np.float16], np.float32),
             ([np.uint8, np.int16, np.float16], np.float32),
             # The following promotions are not ambiguous, but cover code
             # paths of abstract promotion (no particular logic being tested)
             ([1, 1, np.float64], np.float64),
             ([1, 1., np.complex128], np.complex128),
             ([1, 1j, np.float64], np.complex128),
             ([1., 1., np.int64], np.float64),
             ([1., 1j, np.float64], np.complex128),
             ([1j, 1j, np.float64], np.complex128),
             ([1, True, np.bool], np.int_),
            ])
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        # Tests that most permutations do not influence the result.  In the
        # above some uint and int combinations promote to a larger integer
        # type, which would then promote to a larger than necessary float.
        for perm in permutations(dtypes):
            assert np.result_type(*perm) == expected


def test_rational_dtype():
    # test for bug gh-5719
    a = np.array([1111], dtype=rational).astype
    assert_raises(OverflowError, a, 'int8')

    # test that dtype detection finds user-defined types
    x = rational(1)
    assert_equal(np.array([x, x]).dtype, np.dtype(rational))


def test_dtypes_are_true():
    # test for gh-6294
    assert bool(np.dtype('f8'))
    assert bool(np.dtype('i8'))
    assert bool(np.dtype([('a', 'i8'), ('b', 'f4')]))


def test_invalid_dtype_string():
    # test for gh-10440
    assert_raises(TypeError, np.dtype, 'f8,i8,[f8,i8]')
    assert_raises(TypeError, np.dtype, 'Fl\xfcgel')


def test_keyword_argument():
    # test for https://github.com/numpy/numpy/pull/16574#issuecomment-642660971
    assert np.dtype(dtype=np.float64) == np.dtype(np.float64)


class TestFromDTypeAttribute:
    def test_simple(self):
        class dt:
            dtype = np.dtype("f8")

        assert np.dtype(dt) == np.float64
        assert np.dtype(dt()) == np.float64

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    @pytest.mark.skipif(IS_WASM, reason="Pyodide/WASM has limited stack size")
    def test_recursion(self):
        class dt:
            pass

        dt.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)

    def test_void_subtype(self):
        class dt(np.void):
            # This code path is fully untested before, so it is unclear
            # what this should be useful for. Note that if np.void is used
            # numpy will think we are deallocating a base type [1.17, 2019-02].
            dtype = np.dtype("f,f")

        np.dtype(dt)
        np.dtype(dt(1))

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    @pytest.mark.skipif(IS_WASM, reason="Pyodide/WASM has limited stack size")
    def test_void_subtype_recursion(self):
        class vdt(np.void):
            pass

        vdt.dtype = vdt

        with pytest.raises(RecursionError):
            np.dtype(vdt)

        with pytest.raises(RecursionError):
            np.dtype(vdt(1))


class TestDTypeClasses:
    @pytest.mark.parametrize("dtype", list(np.typecodes['All']) + [rational])
    def test_basic_dtypes_subclass_properties(self, dtype):
        # Note: Except for the isinstance and type checks, these attributes
        #       are considered currently private and may change.
        dtype = np.dtype(dtype)
        assert isinstance(dtype, np.dtype)
        assert type(dtype) is not np.dtype
        if dtype.type.__name__ != "rational":
            dt_name = type(dtype).__name__.lower().removesuffix("dtype")
            if dt_name in {"uint", "int"}:
                # The scalar names has a `c` attached because "int" is Python
                # int and that is long...
                dt_name += "c"
            sc_name = dtype.type.__name__
            assert dt_name == sc_name.strip("_")
            assert type(dtype).__module__ == "numpy.dtypes"

            assert getattr(numpy.dtypes, type(dtype).__name__) is type(dtype)
        else:
            assert type(dtype).__name__ == "dtype[rational]"
            assert type(dtype).__module__ == "numpy"

        assert not type(dtype)._abstract

        # the flexible dtypes and datetime/timedelta have additional parameters
        # which are more than just storage information, these would need to be
        # given when creating a dtype:
        parametric = (np.void, np.str_, np.bytes_, np.datetime64, np.timedelta64)
        if dtype.type not in parametric:
            assert not type(dtype)._parametric
            assert type(dtype)() is dtype
        else:
            assert type(dtype)._parametric
            with assert_raises(TypeError):
                type(dtype)()

    def test_dtype_superclass(self):
        assert type(np.dtype) is not type
        assert isinstance(np.dtype, type)

        assert type(np.dtype).__name__ == "_DTypeMeta"
        assert type(np.dtype).__module__ == "numpy"
        assert np.dtype._abstract

    def test_is_numeric(self):
        all_codes = set(np.typecodes['All'])
        numeric_codes = set(np.typecodes['AllInteger'] +
                            np.typecodes['AllFloat'] + '?')
        non_numeric_codes = all_codes - numeric_codes

        for code in numeric_codes:
            assert type(np.dtype(code))._is_numeric

        for code in non_numeric_codes:
            assert not type(np.dtype(code))._is_numeric

    @pytest.mark.parametrize("int_", ["UInt", "Int"])
    @pytest.mark.parametrize("size", [8, 16, 32, 64])
    def test_integer_alias_names(self, int_, size):
        DType = getattr(numpy.dtypes, f"{int_}{size}DType")
        sctype = getattr(numpy, f"{int_.lower()}{size}")
        assert DType.type is sctype
        assert DType.__name__.lower().removesuffix("dtype") == sctype.__name__

    @pytest.mark.parametrize("name",
            ["Half", "Float", "Double", "CFloat", "CDouble"])
    def test_float_alias_names(self, name):
        with pytest.raises(AttributeError):
            getattr(numpy.dtypes, name + "DType") is numpy.dtypes.Float16DType

    def test_scalar_helper_all_dtypes(self):
        for dtype in np.dtypes.__all__:
            dt_class = getattr(np.dtypes, dtype)
            dt = np.dtype(dt_class)
            if dt.char not in 'OTVM':
                assert np._core.multiarray.scalar(dt) == dt.type()
            elif dt.char == 'V':
                assert np._core.multiarray.scalar(dt) == dt.type(b'\x00')
            elif dt.char == 'M':
                # can't do anything with this without generating ValueError
                # because 'M' has no units
                _ = np._core.multiarray.scalar(dt)
            else:
                with pytest.raises(TypeError):
                    np._core.multiarray.scalar(dt)


class TestFromCTypes:

    @staticmethod
    def check(ctype, dtype):
        dtype = np.dtype(dtype)
        assert np.dtype(ctype) == dtype
        assert np.dtype(ctype()) == dtype
        assert ctypes.sizeof(ctype) == dtype.itemsize

    def test_array(self):
        c8 = ctypes.c_uint8
        self.check(     3 * c8,  (np.uint8, (3,)))
        self.check(     1 * c8,  (np.uint8, (1,)))
        self.check(     0 * c8,  (np.uint8, (0,)))
        self.check(1 * (3 * c8), ((np.uint8, (3,)), (1,)))
        self.check(3 * (1 * c8), ((np.uint8, (1,)), (3,)))

    def test_padded_structure(self):
        class PaddedStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_bit_fields(self):
        class BitfieldStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8, 7),
                ('b', ctypes.c_uint8, 1)
            ]
        assert_raises(TypeError, np.dtype, BitfieldStruct)
        assert_raises(TypeError, np.dtype, BitfieldStruct())

    def test_pointer(self):
        p_uint8 = ctypes.POINTER(ctypes.c_uint8)
        assert_raises(TypeError, np.dtype, p_uint8)

    def test_size_t(self):
        assert np.dtype(np.uintp) is np.dtype("N")
        self.check(ctypes.c_size_t, np.uintp)

    def test_void_pointer(self):
        self.check(ctypes.c_void_p, "P")

    def test_union(self):
        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
            ]
        expected = np.dtype({
            "names": ['a', 'b'],
            "formats": [np.uint8, np.uint16],
            "offsets": [0, 0],
            "itemsize": 2
        })
        self.check(Union, expected)

    def test_union_with_struct_packed(self):
        class Struct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]

        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]
        expected = np.dtype({
            "names": ['a', 'b', 'c', 'd'],
            "formats": ['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            "offsets": [0, 0, 0, 0],
            "itemsize": ctypes.sizeof(Union)
        })
        self.check(Union, expected)

    def test_union_packed(self):
        class Struct(ctypes.Structure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1

        class Union(ctypes.Union):
            _pack_ = 1
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]
        expected = np.dtype({
            "names": ['a', 'b', 'c', 'd'],
            "formats": ['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            "offsets": [0, 0, 0, 0],
            "itemsize": ctypes.sizeof(Union)
        })
        self.check(Union, expected)

    def test_packed_structure(self):
        class PackedStructure(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ])
        self.check(PackedStructure, expected)

    def test_large_packed_structure(self):
        class PackedStructure(ctypes.Structure):
            _pack_ = 2
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint8),
                ('d', ctypes.c_uint16),
                ('e', ctypes.c_uint32),
                ('f', ctypes.c_uint32),
                ('g', ctypes.c_uint8)
                ]
        expected = np.dtype({
            "formats": [np.uint8, np.uint16, np.uint8, np.uint16, np.uint32, np.uint32, np.uint8],
            "offsets": [0, 2, 4, 6, 8, 12, 16],
            "names": ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            "itemsize": 18})
        self.check(PackedStructure, expected)

    def test_big_endian_structure_packed(self):
        class BigEndStruct(ctypes.BigEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '>u4')])
        self.check(BigEndStruct, expected)

    def test_little_endian_structure_packed(self):
        class LittleEndStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '<u4')])
        self.check(LittleEndStruct, expected)

    def test_little_endian_structure(self):
        class PaddedStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', '<B'),
            ('b', '<H')
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_big_endian_structure(self):
        class PaddedStruct(ctypes.BigEndianStructure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', '>B'),
            ('b', '>H')
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_simple_endian_types(self):
        self.check(ctypes.c_uint16.__ctype_le__, np.dtype('<u2'))
        self.check(ctypes.c_uint16.__ctype_be__, np.dtype('>u2'))
        self.check(ctypes.c_uint8.__ctype_le__, np.dtype('u1'))
        self.check(ctypes.c_uint8.__ctype_be__, np.dtype('u1'))

    all_types = set(np.typecodes['All'])
    all_pairs = permutations(all_types, 2)

    @pytest.mark.parametrize("pair", all_pairs)
    def test_pairs(self, pair):
        """
        Check that np.dtype('x,y') matches [np.dtype('x'), np.dtype('y')]
        Example: np.dtype('d,I') -> dtype([('f0', '<f8'), ('f1', '<u4')])
        """
        # gh-5645: check that np.dtype('i,L') can be used
        pair_type = np.dtype('{},{}'.format(*pair))
        expected = np.dtype([('f0', pair[0]), ('f1', pair[1])])
        assert_equal(pair_type, expected)


class TestUserDType:
    @pytest.mark.leaks_references(reason="dynamically creates custom dtype.")
    def test_custom_structured_dtype(self):
        class mytype:
            pass

        blueprint = np.dtype([("field", object)])
        dt = create_custom_field_dtype(blueprint, mytype, 0)
        assert dt.type == mytype
        # We cannot (currently) *create* this dtype with `np.dtype` because
        # mytype does not inherit from `np.generic`.  This seems like an
        # unnecessary restriction, but one that has been around forever:
        assert np.dtype(mytype) == np.dtype("O")

        if HAS_REFCOUNT:
            # Create an array and test that memory gets cleaned up (gh-25949)
            o = object()
            startcount = sys.getrefcount(o)
            a = np.array([o], dtype=dt)
            del a
            assert sys.getrefcount(o) == startcount

    def test_custom_structured_dtype_errors(self):
        class mytype:
            pass

        blueprint = np.dtype([("field", object)])

        with pytest.raises(ValueError):
            # Tests what happens if fields are unset during creation
            # which is currently rejected due to the containing object
            # (see PyArray_RegisterDataType).
            create_custom_field_dtype(blueprint, mytype, 1)

        with pytest.raises(RuntimeError):
            # Tests that a dtype must have its type field set up to np.dtype
            # or in this case a builtin instance.
            create_custom_field_dtype(blueprint, mytype, 2)


class TestClassGetItem:
    def test_dtype(self) -> None:
        alias = np.dtype[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.dtype

    @pytest.mark.parametrize("code", np.typecodes["All"])
    def test_dtype_subclass(self, code: str) -> None:
        cls = type(np.dtype(code))
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.dtype[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.dtype[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.dtype[Any]


def test_result_type_integers_and_unitless_timedelta64():
    # Regression test for gh-20077.  The following call of `result_type`
    # would cause a seg. fault.
    td = np.timedelta64(4)
    result = np.result_type(0, td)
    assert_dtype_equal(result, td.dtype)


def test_creating_dtype_with_dtype_class_errors():
    # Regression test for #25031, calling `np.dtype` with itself segfaulted.
    with pytest.raises(TypeError, match="Cannot convert np.dtype into a"):
        np.array(np.ones(10), dtype=np.dtype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\parser\__init__.py ===
# -*- coding: utf-8 -*-
from ._parser import parse, parser, parserinfo, ParserError
from ._parser import DEFAULTPARSER, DEFAULTTZPARSER
from ._parser import UnknownTimezoneWarning

from ._parser import __doc__

from .isoparser import isoparser, isoparse

__all__ = ['parse', 'parser', 'parserinfo',
           'isoparse', 'isoparser',
           'ParserError',
           'UnknownTimezoneWarning']


###
# Deprecate portions of the private interface so that downstream code that
# is improperly relying on it is given *some* notice.


def __deprecated_private_func(f):
    from functools import wraps
    import warnings

    msg = ('{name} is a private function and may break without warning, '
           'it will be moved and or renamed in future versions.')
    msg = msg.format(name=f.__name__)

    @wraps(f)
    def deprecated_func(*args, **kwargs):
        warnings.warn(msg, DeprecationWarning)
        return f(*args, **kwargs)

    return deprecated_func

def __deprecate_private_class(c):
    import warnings

    msg = ('{name} is a private class and may break without warning, '
           'it will be moved and or renamed in future versions.')
    msg = msg.format(name=c.__name__)

    class private_class(c):
        __doc__ = c.__doc__

        def __init__(self, *args, **kwargs):
            warnings.warn(msg, DeprecationWarning)
            super(private_class, self).__init__(*args, **kwargs)

    private_class.__name__ = c.__name__

    return private_class


from ._parser import _timelex, _resultbase
from ._parser import _tzparser, _parsetz

_timelex = __deprecate_private_class(_timelex)
_tzparser = __deprecate_private_class(_tzparser)
_resultbase = __deprecate_private_class(_resultbase)
_parsetz = __deprecated_private_func(_parsetz)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chartsheet\custom.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.worksheet.header_footer import HeaderFooter

from openpyxl.descriptors import (
    Bool,
    Integer,
    Set,
    Typed,
    Sequence
)
from openpyxl.descriptors.excel import Guid
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.worksheet.page import (
    PageMargins,
    PrintPageSetup
)


class CustomChartsheetView(Serialisable):
    tagname = "customSheetView"

    guid = Guid()
    scale = Integer()
    state = Set(values=(['visible', 'hidden', 'veryHidden']))
    zoomToFit = Bool(allow_none=True)
    pageMargins = Typed(expected_type=PageMargins, allow_none=True)
    pageSetup = Typed(expected_type=PrintPageSetup, allow_none=True)
    headerFooter = Typed(expected_type=HeaderFooter, allow_none=True)

    __elements__ = ('pageMargins', 'pageSetup', 'headerFooter')

    def __init__(self,
                 guid=None,
                 scale=None,
                 state='visible',
                 zoomToFit=None,
                 pageMargins=None,
                 pageSetup=None,
                 headerFooter=None,
                 ):
        self.guid = guid
        self.scale = scale
        self.state = state
        self.zoomToFit = zoomToFit
        self.pageMargins = pageMargins
        self.pageSetup = pageSetup
        self.headerFooter = headerFooter


class CustomChartsheetViews(Serialisable):
    tagname = "customSheetViews"

    customSheetView = Sequence(expected_type=CustomChartsheetView, allow_none=True)

    __elements__ = ('customSheetView',)

    def __init__(self,
                 customSheetView=None,
                 ):
        self.customSheetView = customSheetView

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\_structures.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


class InfinityType:
    def __repr__(self) -> str:
        return "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self: object) -> "NegativeInfinityType":
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self) -> str:
        return "-Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self: object) -> InfinityType:
        return Infinity


NegativeInfinity = NegativeInfinityType()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\network\xmlrpc.py ===
"""xmlrpclib.Transport implementation"""

import logging
import urllib.parse
import xmlrpc.client
from typing import TYPE_CHECKING, Tuple

from pip._internal.exceptions import NetworkConnectionError
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status

if TYPE_CHECKING:
    from xmlrpc.client import _HostType, _Marshallable

    from _typeshed import SizedBuffer

logger = logging.getLogger(__name__)


class PipXmlrpcTransport(xmlrpc.client.Transport):
    """Provide a `xmlrpclib.Transport` implementation via a `PipSession`
    object.
    """

    def __init__(
        self, index_url: str, session: PipSession, use_datetime: bool = False
    ) -> None:
        super().__init__(use_datetime)
        index_parts = urllib.parse.urlparse(index_url)
        self._scheme = index_parts.scheme
        self._session = session

    def request(
        self,
        host: "_HostType",
        handler: str,
        request_body: "SizedBuffer",
        verbose: bool = False,
    ) -> Tuple["_Marshallable", ...]:
        assert isinstance(host, str)
        parts = (self._scheme, host, handler, None, None, None)
        url = urllib.parse.urlunparse(parts)
        try:
            headers = {"Content-Type": "text/xml"}
            response = self._session.post(
                url,
                data=request_body,
                headers=headers,
                stream=True,
            )
            raise_for_status(response)
            self.verbose = verbose
            return self.parse_response(response.raw)
        except NetworkConnectionError as exc:
            assert exc.response
            logger.critical(
                "HTTP error %s while getting %s",
                exc.response.status_code,
                url,
            )
            raise

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\_structures.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


class InfinityType:
    def __repr__(self) -> str:
        return "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self: object) -> "NegativeInfinityType":
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self) -> str:
        return "-Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self: object) -> InfinityType:
        return Infinity


NegativeInfinity = NegativeInfinityType()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\styles\__init__.py ===
"""
    pygments.styles
    ~~~~~~~~~~~~~~~

    Contains built-in styles.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pip._vendor.pygments.plugin import find_plugin_styles
from pip._vendor.pygments.util import ClassNotFound
from pip._vendor.pygments.styles._mapping import STYLES

#: A dictionary of built-in styles, mapping style names to
#: ``'submodule::classname'`` strings.
#: This list is deprecated. Use `pygments.styles.STYLES` instead
STYLE_MAP = {v[1]: v[0].split('.')[-1] + '::' + k for k, v in STYLES.items()}

#: Internal reverse mapping to make `get_style_by_name` more efficient
_STYLE_NAME_TO_MODULE_MAP = {v[1]: (v[0], k) for k, v in STYLES.items()}


def get_style_by_name(name):
    """
    Return a style class by its short name. The names of the builtin styles
    are listed in :data:`pygments.styles.STYLE_MAP`.

    Will raise :exc:`pygments.util.ClassNotFound` if no style of that name is
    found.
    """
    if name in _STYLE_NAME_TO_MODULE_MAP:
        mod, cls = _STYLE_NAME_TO_MODULE_MAP[name]
        builtin = "yes"
    else:
        for found_name, style in find_plugin_styles():
            if name == found_name:
                return style
        # perhaps it got dropped into our styles package
        builtin = ""
        mod = 'pygments.styles.' + name
        cls = name.title() + "Style"

    try:
        mod = __import__(mod, None, None, [cls])
    except ImportError:
        raise ClassNotFound(f"Could not find style module {mod!r}" +
                            (builtin and ", though it should be builtin")
                            + ".")
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ClassNotFound(f"Could not find style class {cls!r} in style module.")


def get_all_styles():
    """Return a generator for all styles by name, both builtin and plugin."""
    for v in STYLES.values():
        yield v[1]
    for name, _ in find_plugin_styles():
        yield name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\__init__.py ===
# dialects/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Callable
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING

from .. import util

if TYPE_CHECKING:
    from ..engine.interfaces import Dialect

__all__ = ("mssql", "mysql", "oracle", "postgresql", "sqlite")


def _auto_fn(name: str) -> Optional[Callable[[], Type[Dialect]]]:
    """default dialect importer.

    plugs into the :class:`.PluginLoader`
    as a first-hit system.

    """
    if "." in name:
        dialect, driver = name.split(".")
    else:
        dialect = name
        driver = "base"

    try:
        if dialect == "mariadb":
            # it's "OK" for us to hardcode here since _auto_fn is already
            # hardcoded.   if mysql / mariadb etc were third party dialects
            # they would just publish all the entrypoints, which would actually
            # look much nicer.
            module = __import__(
                "sqlalchemy.dialects.mysql.mariadb"
            ).dialects.mysql.mariadb
            return module.loader(driver)  # type: ignore
        else:
            module = __import__("sqlalchemy.dialects.%s" % (dialect,)).dialects
            module = getattr(module, dialect)
    except ImportError:
        return None

    if hasattr(module, driver):
        module = getattr(module, driver)
        return lambda: module.dialect
    else:
        return None


registry = util.PluginLoader("sqlalchemy.dialects", auto_fn=_auto_fn)

plugins = util.PluginLoader("sqlalchemy.plugins")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\psycopg2cffi.py ===
# dialects/postgresql/psycopg2cffi.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r"""
.. dialect:: postgresql+psycopg2cffi
    :name: psycopg2cffi
    :dbapi: psycopg2cffi
    :connectstring: postgresql+psycopg2cffi://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://pypi.org/project/psycopg2cffi/

``psycopg2cffi`` is an adaptation of ``psycopg2``, using CFFI for the C
layer. This makes it suitable for use in e.g. PyPy. Documentation
is as per ``psycopg2``.

.. seealso::

    :mod:`sqlalchemy.dialects.postgresql.psycopg2`

"""  # noqa
from .psycopg2 import PGDialect_psycopg2
from ... import util


class PGDialect_psycopg2cffi(PGDialect_psycopg2):
    driver = "psycopg2cffi"
    supports_unicode_statements = True
    supports_statement_cache = True

    # psycopg2cffi's first release is 2.5.0, but reports
    # __version__ as 2.4.4.  Subsequent releases seem to have
    # fixed this.

    FEATURE_VERSION_MAP = dict(
        native_json=(2, 4, 4),
        native_jsonb=(2, 7, 1),
        sane_multi_rowcount=(2, 4, 4),
        array_oid=(2, 4, 4),
        hstore_adapter=(2, 4, 4),
    )

    @classmethod
    def import_dbapi(cls):
        return __import__("psycopg2cffi")

    @util.memoized_property
    def _psycopg2_extensions(cls):
        root = __import__("psycopg2cffi", fromlist=["extensions"])
        return root.extensions

    @util.memoized_property
    def _psycopg2_extras(cls):
        root = __import__("psycopg2cffi", fromlist=["extras"])
        return root.extras


dialect = PGDialect_psycopg2cffi

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\processors.py ===
# engine/processors.py
# Copyright (C) 2010-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
# Copyright (C) 2010 Gaetan de Menten gdementen@gmail.com
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""defines generic type conversion functions, as used in bind and result
processors.

They all share one common characteristic: None is passed through unchanged.

"""
from __future__ import annotations

import typing

from ._py_processors import str_to_datetime_processor_factory  # noqa
from ..util._has_cy import HAS_CYEXTENSION

if typing.TYPE_CHECKING or not HAS_CYEXTENSION:
    from ._py_processors import int_to_boolean as int_to_boolean
    from ._py_processors import str_to_date as str_to_date
    from ._py_processors import str_to_datetime as str_to_datetime
    from ._py_processors import str_to_time as str_to_time
    from ._py_processors import (
        to_decimal_processor_factory as to_decimal_processor_factory,
    )
    from ._py_processors import to_float as to_float
    from ._py_processors import to_str as to_str
else:
    from sqlalchemy.cyextension.processors import (
        DecimalResultProcessor,
    )
    from sqlalchemy.cyextension.processors import (  # noqa: F401
        int_to_boolean as int_to_boolean,
    )
    from sqlalchemy.cyextension.processors import (  # noqa: F401,E501
        str_to_date as str_to_date,
    )
    from sqlalchemy.cyextension.processors import (  # noqa: F401
        str_to_datetime as str_to_datetime,
    )
    from sqlalchemy.cyextension.processors import (  # noqa: F401,E501
        str_to_time as str_to_time,
    )
    from sqlalchemy.cyextension.processors import (  # noqa: F401,E501
        to_float as to_float,
    )
    from sqlalchemy.cyextension.processors import (  # noqa: F401,E501
        to_str as to_str,
    )

    def to_decimal_processor_factory(target_class, scale):
        # Note that the scale argument is not taken into account for integer
        # values in the C implementation while it is in the Python one.
        # For example, the Python implementation might return
        # Decimal('5.00000') whereas the C implementation will
        # return Decimal('5'). These are equivalent of course.
        return DecimalResultProcessor(target_class, "%%.%df" % scale).process

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\group_constructs.py ===
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
from sympy.utilities.iterables import uniq

_af_new = Permutation._af_new


def DirectProduct(*groups):
    """
    Returns the direct product of several groups as a permutation group.

    Explanation
    ===========

    This is implemented much like the __mul__ procedure for taking the direct
    product of two permutation groups, but the idea of shifting the
    generators is realized in the case of an arbitrary number of groups.
    A call to DirectProduct(G1, G2, ..., Gn) is generally expected to be faster
    than a call to G1*G2*...*Gn (and thus the need for this algorithm).

    Examples
    ========

    >>> from sympy.combinatorics.group_constructs import DirectProduct
    >>> from sympy.combinatorics.named_groups import CyclicGroup
    >>> C = CyclicGroup(4)
    >>> G = DirectProduct(C, C, C)
    >>> G.order()
    64

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.__mul__

    """
    degrees = []
    gens_count = []
    total_degree = 0
    total_gens = 0
    for group in groups:
        current_deg = group.degree
        current_num_gens = len(group.generators)
        degrees.append(current_deg)
        total_degree += current_deg
        gens_count.append(current_num_gens)
        total_gens += current_num_gens
    array_gens = []
    for i in range(total_gens):
        array_gens.append(list(range(total_degree)))
    current_gen = 0
    current_deg = 0
    for i in range(len(gens_count)):
        for j in range(current_gen, current_gen + gens_count[i]):
            gen = ((groups[i].generators)[j - current_gen]).array_form
            array_gens[j][current_deg:current_deg + degrees[i]] = \
                [x + current_deg for x in gen]
        current_gen += gens_count[i]
        current_deg += degrees[i]
    perm_gens = list(uniq([_af_new(list(a)) for a in array_gens]))
    return PermutationGroup(perm_gens, dups=False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\window\test_rolling.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_arm,
    is_platform_power,
)

from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer

from pandas.tseries.offsets import BusinessDay


def test_doc_string():
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df
    df.rolling(2).sum()
    df.rolling(2, min_periods=1).sum()


def test_constructor(frame_or_series):
    # GH 12669

    c = frame_or_series(range(5)).rolling

    # valid
    c(0)
    c(window=2)
    c(window=2, min_periods=1)
    c(window=2, min_periods=1, center=True)
    c(window=2, min_periods=1, center=False)

    # GH 13383

    msg = "window must be an integer 0 or greater"

    with pytest.raises(ValueError, match=msg):
        c(-1)


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_invalid_constructor(frame_or_series, w):
    # not valid

    c = frame_or_series(range(5)).rolling

    msg = "|".join(
        [
            "window must be an integer",
            "passed window foo is not compatible with a datetimelike index",
        ]
    )
    with pytest.raises(ValueError, match=msg):
        c(window=w)

    msg = "min_periods must be an integer"
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=w)

    msg = "center must be a boolean"
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=1, center=w)


@pytest.mark.parametrize(
    "window",
    [
        timedelta(days=3),
        Timedelta(days=3),
        "3D",
        VariableOffsetWindowIndexer(
            index=date_range("2015-12-25", periods=5), offset=BusinessDay(1)
        ),
    ],
)
def test_freq_window_not_implemented(window):
    # GH 15354
    df = DataFrame(
        np.arange(10),
        index=date_range("2015-12-24", periods=10, freq="D"),
    )
    with pytest.raises(
        NotImplementedError, match="^step (not implemented|is not supported)"
    ):
        df.rolling(window, step=3).sum()


@pytest.mark.parametrize("agg", ["cov", "corr"])
def test_step_not_implemented_for_cov_corr(agg):
    # GH 15354
    roll = DataFrame(range(2)).rolling(1, step=2)
    with pytest.raises(NotImplementedError, match="step not implemented"):
        getattr(roll, agg)()


@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window):
    # GH 15440
    n = 10
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )
    expected_data = np.append([0.0, 1.0], np.arange(3.0, 27.0, 3))

    result = df.rolling(window=window).sum()
    expected = DataFrame(
        {"value": expected_data},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )
    tm.assert_frame_equal(result, expected)
    expected = df.rolling("3D").sum()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3), "3D"])
def test_constructor_timedelta_window_and_minperiods(window, raw):
    # GH 15305
    n = 10
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    expected = DataFrame(
        {"value": np.append([np.nan, 1.0], np.arange(3.0, 27.0, 3))},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    result_roll_sum = df.rolling(window=window, min_periods=2).sum()
    result_roll_generic = df.rolling(window=window, min_periods=2).apply(sum, raw=raw)
    tm.assert_frame_equal(result_roll_sum, expected)
    tm.assert_frame_equal(result_roll_generic, expected)


def test_closed_fixed(closed, arithmetic_win_operators):
    # GH 34315
    func_name = arithmetic_win_operators
    df_fixed = DataFrame({"A": [0, 1, 2, 3, 4]})
    df_time = DataFrame({"A": [0, 1, 2, 3, 4]}, index=date_range("2020", periods=5))

    result = getattr(
        df_fixed.rolling(2, closed=closed, min_periods=1),
        func_name,
    )()
    expected = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1),
        func_name,
    )().reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "closed, window_selections",
    [
        (
            "both",
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [False, False, False, True, True],
            ],
        ),
        (
            "left",
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
            ],
        ),
        (
            "right",
            [
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
                [False, False, False, False, True],
            ],
        ),
        (
            "neither",
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
        ),
    ],
)
def test_datetimelike_centered_selections(
    closed, window_selections, arithmetic_win_operators
):
    # GH 34315
    func_name = arithmetic_win_operators
    df_time = DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=date_range("2020", periods=5)
    )

    expected = DataFrame(
        {"A": [getattr(df_time["A"].iloc[s], func_name)() for s in window_selections]},
        index=date_range("2020", periods=5),
    )

    if func_name == "sem":
        kwargs = {"ddof": 0}
    else:
        kwargs = {}

    result = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1, center=True),
        func_name,
    )(**kwargs)

    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("3s", "right", [3.0, 3.0, 3.0]),
        ("3s", "both", [3.0, 3.0, 3.0]),
        ("3s", "left", [3.0, 3.0, 3.0]),
        ("3s", "neither", [3.0, 3.0, 3.0]),
        ("2s", "right", [3.0, 2.0, 2.0]),
        ("2s", "both", [3.0, 3.0, 3.0]),
        ("2s", "left", [1.0, 3.0, 3.0]),
        ("2s", "neither", [1.0, 2.0, 2.0]),
    ],
)
def test_datetimelike_centered_offset_covers_all(
    window, closed, expected, frame_or_series
):
    # GH 42753

    index = [
        Timestamp("20130101 09:00:01"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:02"),
    ]
    df = frame_or_series([1, 1, 1], index=index)

    result = df.rolling(window, closed=closed, center=True).sum()
    expected = frame_or_series(expected, index=index)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("2D", "right", [4, 4, 4, 4, 4, 4, 2, 2]),
        ("2D", "left", [2, 2, 4, 4, 4, 4, 4, 4]),
        ("2D", "both", [4, 4, 6, 6, 6, 6, 4, 4]),
        ("2D", "neither", [2, 2, 2, 2, 2, 2, 2, 2]),
    ],
)
def test_datetimelike_nonunique_index_centering(
    window, closed, expected, frame_or_series
):
    index = DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
            "2020-01-03",
            "2020-01-04",
            "2020-01-04",
        ]
    )

    df = frame_or_series([1] * 8, index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)

    result = df.rolling(window, center=True, closed=closed).sum()

    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("left", [np.nan, np.nan, 1, 1, 1, 10, 14, 14, 18, 21]),
        ("neither", [np.nan, np.nan, 1, 1, 1, 9, 5, 5, 13, 8]),
        ("right", [0, 1, 3, 6, 10, 14, 11, 18, 21, 17]),
        ("both", [0, 1, 3, 6, 10, 15, 20, 27, 26, 30]),
    ],
)
def test_variable_window_nonunique(closed, expected, frame_or_series):
    # GH 20712
    index = DatetimeIndex(
        [
            "2011-01-01",
            "2011-01-01",
            "2011-01-02",
            "2011-01-02",
            "2011-01-02",
            "2011-01-03",
            "2011-01-04",
            "2011-01-04",
            "2011-01-05",
            "2011-01-06",
        ]
    )

    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)

    result = df.rolling("2D", closed=closed).sum()

    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("left", [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 18, 21]),
        ("neither", [np.nan, np.nan, 1, 1, 1, 10, 15, 15, 13, 8]),
        ("right", [0, 1, 3, 6, 10, 15, 21, 28, 21, 17]),
        ("both", [0, 1, 3, 6, 10, 15, 21, 28, 26, 30]),
    ],
)
def test_variable_offset_window_nonunique(closed, expected, frame_or_series):
    # GH 20712
    index = DatetimeIndex(
        [
            "2011-01-01",
            "2011-01-01",
            "2011-01-02",
            "2011-01-02",
            "2011-01-02",
            "2011-01-03",
            "2011-01-04",
            "2011-01-04",
            "2011-01-05",
            "2011-01-06",
        ]
    )

    df = frame_or_series(range(10), index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)

    offset = BusinessDay(2)
    indexer = VariableOffsetWindowIndexer(index=index, offset=offset)
    result = df.rolling(indexer, closed=closed, min_periods=1).sum()

    tm.assert_equal(result, expected)


def test_even_number_window_alignment():
    # see discussion in GH 38780
    s = Series(range(3), index=date_range(start="2020-01-01", freq="D", periods=3))

    # behavior of index- and datetime-based windows differs here!
    # s.rolling(window=2, min_periods=1, center=True).mean()

    result = s.rolling(window="2D", min_periods=1, center=True).mean()

    expected = Series([0.5, 1.5, 2], index=s.index)

    tm.assert_series_equal(result, expected)


def test_closed_fixed_binary_col(center, step):
    # GH 34315
    data = [0, 1, 1, 0, 0, 1, 0, 1]
    df = DataFrame(
        {"binary_col": data},
        index=date_range(start="2020-01-01", freq="min", periods=len(data)),
    )

    if center:
        expected_data = [2 / 3, 0.5, 0.4, 0.5, 0.428571, 0.5, 0.571429, 0.5]
    else:
        expected_data = [np.nan, 0, 0.5, 2 / 3, 0.5, 0.4, 0.5, 0.428571]

    expected = DataFrame(
        expected_data,
        columns=["binary_col"],
        index=date_range(start="2020-01-01", freq="min", periods=len(expected_data)),
    )[::step]

    rolling = df.rolling(
        window=len(df), closed="left", min_periods=1, center=center, step=step
    )
    result = rolling.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("closed", ["neither", "left"])
def test_closed_empty(closed, arithmetic_win_operators):
    # GH 26005
    func_name = arithmetic_win_operators
    ser = Series(data=np.arange(5), index=date_range("2000", periods=5, freq="2D"))
    roll = ser.rolling("1D", closed=closed)

    result = getattr(roll, func_name)()
    expected = Series([np.nan] * 5, index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry(func):
    # GH24718
    ser = Series(data=[2], index=date_range("2000", periods=1))
    result = getattr(ser.rolling("10D", closed="left"), func)()
    tm.assert_series_equal(result, Series([np.nan], index=ser.index))


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry_groupby(func):
    # GH24718
    ser = DataFrame(
        data={"A": [1, 1, 2], "B": [3, 2, 1]},
        index=date_range("2000", periods=3),
    )
    result = getattr(
        ser.groupby("A", sort=False)["B"].rolling("10D", closed="left"), func
    )()
    exp_idx = MultiIndex.from_arrays(arrays=[[1, 1, 2], ser.index], names=("A", None))
    expected = Series(data=[np.nan, 3, np.nan], index=exp_idx, name="B")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("input_dtype", ["int", "float"])
@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "both", [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("min", "neither", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "left", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", "right", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "both", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "neither", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("max", "left", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_closed_min_max_datetime(input_dtype, func, closed, expected):
    # see gh-21704
    ser = Series(
        data=np.arange(10).astype(input_dtype),
        index=date_range("2000", periods=10),
    )

    result = getattr(ser.rolling("3D", closed=closed), func)()
    expected = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected)


def test_closed_uneven():
    # see gh-21704
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))

    # uneven
    ser = ser.drop(index=ser.index[[1, 5]])
    result = ser.rolling("3D", closed="left").min()
    expected = Series([np.nan, 0, 0, 2, 3, 4, 6, 6], index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "both", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("min", "neither", [np.nan, np.nan, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "left", [np.nan, np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("max", "right", [np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan, np.nan]),
        ("max", "both", [np.nan, 1, 2, 3, 4, 5, 6, 6, 6, np.nan]),
        ("max", "neither", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, np.nan, np.nan]),
        ("max", "left", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan]),
    ],
)
def test_closed_min_max_minp(func, closed, expected):
    # see gh-21704
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))
    # Explicit cast to float to avoid implicit cast when setting nan
    ser = ser.astype("float")
    ser[ser.index[-3:]] = np.nan
    result = getattr(ser.rolling("3D", min_periods=2, closed=closed), func)()
    expected = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("right", [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("both", [0, 0.5, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("neither", [np.nan, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("left", [np.nan, 0, 0.5, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_closed_median_quantile(closed, expected):
    # GH 26005
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))
    roll = ser.rolling("3D", closed=closed)
    expected = Series(expected, index=ser.index)

    result = roll.median()
    tm.assert_series_equal(result, expected)

    result = roll.quantile(0.5)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("roller", ["1s", 1])
def tests_empty_df_rolling(roller):
    # GH 15819 Verifies that datetime and integer rolling windows can be
    # applied to empty DataFrames
    expected = DataFrame()
    result = DataFrame().rolling(roller).sum()
    tm.assert_frame_equal(result, expected)

    # Verifies that datetime and integer rolling windows can be applied to
    # empty DataFrames with datetime index
    expected = DataFrame(index=DatetimeIndex([]))
    result = DataFrame(index=DatetimeIndex([])).rolling(roller).sum()
    tm.assert_frame_equal(result, expected)


def test_empty_window_median_quantile():
    # GH 26005
    expected = Series([np.nan, np.nan, np.nan])
    roll = Series(np.arange(3)).rolling(0)

    result = roll.median()
    tm.assert_series_equal(result, expected)

    result = roll.quantile(0.1)
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero():
    # https://github.com/pandas-dev/pandas/pull/18921
    # minp=0
    x = Series([np.nan])
    result = x.rolling(1, min_periods=0).sum()
    expected = Series([0.0])
    tm.assert_series_equal(result, expected)

    # minp=1
    result = x.rolling(1, min_periods=1).sum()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero_variable():
    # https://github.com/pandas-dev/pandas/pull/18921
    x = Series(
        [np.nan] * 4,
        index=DatetimeIndex(["2017-01-01", "2017-01-04", "2017-01-06", "2017-01-07"]),
    )
    result = x.rolling(Timedelta("2d"), min_periods=0).sum()
    expected = Series(0.0, index=x.index)
    tm.assert_series_equal(result, expected)


def test_multi_index_names():
    # GH 16789, 16825
    cols = MultiIndex.from_product([["A", "B"], ["C", "D", "E"]], names=["1", "2"])
    df = DataFrame(np.ones((10, 6)), columns=cols)
    result = df.rolling(3).cov()

    tm.assert_index_equal(result.columns, df.columns)
    assert result.index.names == [None, "1", "2"]


def test_rolling_axis_sum(axis_frame):
    # see gh-23372.
    df = DataFrame(np.ones((10, 20)))
    axis = df._get_axis_number(axis_frame)

    if axis == 0:
        msg = "The 'axis' keyword in DataFrame.rolling"
        expected = DataFrame({i: [np.nan] * 2 + [3.0] * 8 for i in range(20)})
    else:
        # axis == 1
        msg = "Support for axis=1 in DataFrame.rolling is deprecated"
        expected = DataFrame([[np.nan] * 2 + [3.0] * 18] * 10)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(3, axis=axis_frame).sum()
    tm.assert_frame_equal(result, expected)


def test_rolling_axis_count(axis_frame):
    # see gh-26055
    df = DataFrame({"x": range(3), "y": range(3)})

    axis = df._get_axis_number(axis_frame)

    if axis in [0, "index"]:
        msg = "The 'axis' keyword in DataFrame.rolling"
        expected = DataFrame({"x": [1.0, 2.0, 2.0], "y": [1.0, 2.0, 2.0]})
    else:
        msg = "Support for axis=1 in DataFrame.rolling is deprecated"
        expected = DataFrame({"x": [1.0, 1.0, 1.0], "y": [2.0, 2.0, 2.0]})

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(2, axis=axis_frame, min_periods=0).count()
    tm.assert_frame_equal(result, expected)


def test_readonly_array():
    # GH-27766
    arr = np.array([1, 3, np.nan, 3, 5])
    arr.setflags(write=False)
    result = Series(arr).rolling(2).mean()
    expected = Series([np.nan, 2, np.nan, np.nan, 4])
    tm.assert_series_equal(result, expected)


def test_rolling_datetime(axis_frame, tz_naive_fixture):
    # GH-28192
    tz = tz_naive_fixture
    df = DataFrame(
        {i: [1] * 2 for i in date_range("2019-8-01", "2019-08-03", freq="D", tz=tz)}
    )

    if axis_frame in [0, "index"]:
        msg = "The 'axis' keyword in DataFrame.rolling"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.T.rolling("2D", axis=axis_frame).sum().T
    else:
        msg = "Support for axis=1 in DataFrame.rolling"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.rolling("2D", axis=axis_frame).sum()
    expected = DataFrame(
        {
            **{
                i: [1.0] * 2
                for i in date_range("2019-8-01", periods=1, freq="D", tz=tz)
            },
            **{
                i: [2.0] * 2
                for i in date_range("2019-8-02", "2019-8-03", freq="D", tz=tz)
            },
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("center", [True, False])
def test_rolling_window_as_string(center):
    # see gh-22590
    date_today = datetime.now()
    days = date_range(date_today, date_today + timedelta(365), freq="D")

    data = np.ones(len(days))
    df = DataFrame({"DateCol": days, "metric": data})

    df.set_index("DateCol", inplace=True)
    result = df.rolling(window="21D", min_periods=2, closed="left", center=center)[
        "metric"
    ].agg("max")

    index = days.rename("DateCol")
    index = index._with_freq(None)
    expected_data = np.ones(len(days), dtype=np.float64)
    if not center:
        expected_data[:2] = np.nan
    expected = Series(expected_data, index=index, name="metric")
    tm.assert_series_equal(result, expected)


def test_min_periods1():
    # GH#6795
    df = DataFrame([0, 1, 2, 1, 0], columns=["a"])
    result = df["a"].rolling(3, center=True, min_periods=1).max()
    expected = Series([1.0, 2.0, 2.0, 2.0, 1.0], name="a")
    tm.assert_series_equal(result, expected)


def test_rolling_count_with_min_periods(frame_or_series):
    # GH 26996
    result = frame_or_series(range(5)).rolling(3, min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 3.0, 3.0])
    tm.assert_equal(result, expected)


def test_rolling_count_default_min_periods_with_null_values(frame_or_series):
    # GH 26996
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0]

    # GH 31302
    result = frame_or_series(values).rolling(3, min_periods=0).count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "df,expected,window,min_periods",
    [
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            3,
            None,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            2,
            1,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            2,
            2,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            1,
            1,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            1,
            0,
        ),
        (DataFrame({"A": [1], "B": [4]}), [], 2, None),
        (DataFrame({"A": [1], "B": [4]}), [], 2, 1),
        (DataFrame(), [({}, [])], 2, None),
        (
            DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}),
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            3,
            2,
        ),
    ],
)
def test_iter_rolling_dataframe(df, expected, window, min_periods):
    # GH 11704
    expected = [DataFrame(values, index=index) for (values, index) in expected]

    for expected, actual in zip(expected, df.rolling(window, min_periods=min_periods)):
        tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "expected,window",
    [
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            "2D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            "3D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            "1D",
        ),
    ],
)
def test_iter_rolling_on_dataframe(expected, window):
    # GH 11704, 40373
    df = DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [4, 5, 6, 7, 8],
            "C": date_range(start="2016-01-01", periods=5, freq="D"),
        }
    )

    expected = [
        DataFrame(values, index=df.loc[index, "C"]) for (values, index) in expected
    ]
    for expected, actual in zip(expected, df.rolling(window, on="C")):
        tm.assert_frame_equal(actual, expected)


def test_iter_rolling_on_dataframe_unordered():
    # GH 43386
    df = DataFrame({"a": ["x", "y", "x"], "b": [0, 1, 2]})
    results = list(df.groupby("a").rolling(2))
    expecteds = [df.iloc[idx, [1]] for idx in [[0], [0, 2], [1]]]
    for result, expected in zip(results, expecteds):
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ser,expected,window, min_periods",
    [
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            None,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            2,
        ),
        (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 0),
        (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 1),
        (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2, 0),
        (Series([], dtype="int64"), [], 2, 1),
    ],
)
def test_iter_rolling_series(ser, expected, window, min_periods):
    # GH 11704
    expected = [Series(values, index=index) for (values, index) in expected]

    for expected, actual in zip(expected, ser.rolling(window, min_periods=min_periods)):
        tm.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "expected,expected_index,window",
    [
        (
            [[0], [1], [2], [3], [4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-02", periods=1, freq="D"),
                date_range("2020-01-03", periods=1, freq="D"),
                date_range("2020-01-04", periods=1, freq="D"),
                date_range("2020-01-05", periods=1, freq="D"),
            ],
            "1D",
        ),
        (
            [[0], [0, 1], [1, 2], [2, 3], [3, 4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-02", periods=2, freq="D"),
                date_range("2020-01-03", periods=2, freq="D"),
                date_range("2020-01-04", periods=2, freq="D"),
            ],
            "2D",
        ),
        (
            [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-01", periods=3, freq="D"),
                date_range("2020-01-02", periods=3, freq="D"),
                date_range("2020-01-03", periods=3, freq="D"),
            ],
            "3D",
        ),
    ],
)
def test_iter_rolling_datetime(expected, expected_index, window):
    # GH 11704
    ser = Series(range(5), index=date_range(start="2020-01-01", periods=5, freq="D"))

    expected = [
        Series(values, index=idx) for (values, idx) in zip(expected, expected_index)
    ]

    for expected, actual in zip(expected, ser.rolling(window)):
        tm.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "grouping,_index",
    [
        (
            {"level": 0},
            MultiIndex.from_tuples(
                [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)], names=[None, None]
            ),
        ),
        (
            {"by": "X"},
            MultiIndex.from_tuples(
                [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], names=["X", None]
            ),
        ),
    ],
)
def test_rolling_positional_argument(grouping, _index, raw):
    # GH 34605

    def scaled_sum(*args):
        if len(args) < 2:
            raise ValueError("The function needs two arguments")
        array, scale = args
        return array.sum() / scale

    df = DataFrame(data={"X": range(5)}, index=[0, 0, 1, 1, 1])

    expected = DataFrame(data={"X": [0.0, 0.5, 1.0, 1.5, 2.0]}, index=_index)
    # GH 40341
    if "by" in grouping:
        expected = expected.drop(columns="X", errors="ignore")
    result = df.groupby(**grouping).rolling(1).apply(scaled_sum, raw=raw, args=(2,))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("add", [0.0, 2.0])
def test_rolling_numerical_accuracy_kahan_mean(add, unit):
    # GH: 36031 implementing kahan summation
    dti = DatetimeIndex(
        [
            Timestamp("19700101 09:00:00"),
            Timestamp("19700101 09:00:03"),
            Timestamp("19700101 09:00:06"),
        ]
    ).as_unit(unit)
    df = DataFrame(
        {"A": [3002399751580331.0 + add, -0.0, -0.0]},
        index=dti,
    )
    result = (
        df.resample("1s").ffill().rolling("3s", closed="left", min_periods=3).mean()
    )
    dates = date_range("19700101 09:00:00", periods=7, freq="s", unit=unit)
    expected = DataFrame(
        {
            "A": [
                np.nan,
                np.nan,
                np.nan,
                3002399751580330.5,
                2001599834386887.25,
                1000799917193443.625,
                0.0,
            ]
        },
        index=dates,
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_numerical_accuracy_kahan_sum():
    # GH: 13254
    df = DataFrame([2.186, -1.647, 0.0, 0.0, 0.0, 0.0], columns=["x"])
    result = df["x"].rolling(3).sum()
    expected = Series([np.nan, np.nan, 0.539, -1.647, 0.0, 0.0], name="x")
    tm.assert_series_equal(result, expected)


def test_rolling_numerical_accuracy_jump():
    # GH: 32761
    index = date_range(start="2020-01-01", end="2020-01-02", freq="60s").append(
        DatetimeIndex(["2020-01-03"])
    )
    data = np.random.default_rng(2).random(len(index))

    df = DataFrame({"data": data}, index=index)
    result = df.rolling("60s").mean()
    tm.assert_frame_equal(result, df[["data"]])


def test_rolling_numerical_accuracy_small_values():
    # GH: 10319
    s = Series(
        data=[0.00012456, 0.0003, -0.0, -0.0],
        index=date_range("1999-02-03", "1999-02-06"),
    )
    result = s.rolling(1).mean()
    tm.assert_series_equal(result, s)


def test_rolling_numerical_too_large_numbers():
    # GH: 11645
    dates = date_range("2015-01-01", periods=10, freq="D")
    ds = Series(data=range(10), index=dates, dtype=np.float64)
    ds.iloc[2] = -9e33
    result = ds.rolling(5).mean()
    expected = Series(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            -1.8e33,
            -1.8e33,
            -1.8e33,
            5.0,
            6.0,
            7.0,
        ],
        index=dates,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "value"),
    [("sum", 2.0), ("max", 1.0), ("min", 1.0), ("mean", 1.0), ("median", 1.0)],
)
def test_rolling_mixed_dtypes_axis_1(func, value):
    # GH: 20649
    df = DataFrame(1, index=[1, 2], columns=["a", "b", "c"])
    df["c"] = 1.0
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        roll = df.rolling(window=2, min_periods=1, axis=1)
    result = getattr(roll, func)()
    expected = DataFrame(
        {"a": [1.0, 1.0], "b": [value, value], "c": [value, value]},
        index=[1, 2],
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_axis_one_with_nan():
    # GH: 35596
    df = DataFrame(
        [
            [0, 1, 2, 4, np.nan, np.nan, np.nan],
            [0, 1, 2, np.nan, np.nan, np.nan, np.nan],
            [0, 2, 2, np.nan, 2, np.nan, 1],
        ]
    )
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=7, min_periods=1, axis="columns").sum()
    expected = DataFrame(
        [
            [0.0, 1.0, 3.0, 7.0, 7.0, 7.0, 7.0],
            [0.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [0.0, 2.0, 4.0, 4.0, 6.0, 6.0, 7.0],
        ]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    ["test", to_datetime("2019-12-31"), to_timedelta("1 days 06:05:01.00003")],
)
def test_rolling_axis_1_non_numeric_dtypes(value):
    # GH: 20649
    df = DataFrame({"a": [1, 2]})
    df["b"] = value
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=2, min_periods=1, axis=1).sum()
    expected = DataFrame({"a": [1.0, 2.0]})
    tm.assert_frame_equal(result, expected)


def test_rolling_on_df_transposed():
    # GH: 32724
    df = DataFrame({"A": [1, None], "B": [4, 5], "C": [7, 8]})
    expected = DataFrame({"A": [1.0, np.nan], "B": [5.0, 5.0], "C": [11.0, 13.0]})
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(min_periods=1, window=2, axis=1).sum()
    tm.assert_frame_equal(result, expected)

    result = df.T.rolling(min_periods=1, window=2).sum().T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("index", "window"),
    [
        (
            period_range(start="2020-01-01 08:00", end="2020-01-01 08:08", freq="min"),
            "2min",
        ),
        (
            period_range(
                start="2020-01-01 08:00", end="2020-01-01 12:00", freq="30min"
            ),
            "1h",
        ),
    ],
)
@pytest.mark.parametrize(
    ("func", "values"),
    [
        ("min", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("sum", [np.nan, 0, 1, 3, 5, 7, 9, 11, 13]),
    ],
)
def test_rolling_period_index(index, window, func, values):
    # GH: 34225
    ds = Series([0, 1, 2, 3, 4, 5, 6, 7, 8], index=index)
    result = getattr(ds.rolling(window, closed="left"), func)()
    expected = Series(values, index=index)
    tm.assert_series_equal(result, expected)


def test_rolling_sem(frame_or_series):
    # GH: 26476
    obj = frame_or_series([0, 1, 2])
    result = obj.rolling(2, min_periods=1).sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan] + [0.7071067811865476] * 2)
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    is_platform_arm() or is_platform_power(),
    reason="GH 38921",
)
@pytest.mark.parametrize(
    ("func", "third_value", "values"),
    [
        ("var", 1, [5e33, 0, 0.5, 0.5, 2, 0]),
        ("std", 1, [7.071068e16, 0, 0.7071068, 0.7071068, 1.414214, 0]),
        ("var", 2, [5e33, 0.5, 0, 0.5, 2, 0]),
        ("std", 2, [7.071068e16, 0.7071068, 0, 0.7071068, 1.414214, 0]),
    ],
)
def test_rolling_var_numerical_issues(func, third_value, values):
    # GH: 37051
    ds = Series([99999999999999999, 1, third_value, 2, 3, 1, 1])
    result = getattr(ds.rolling(2), func)()
    expected = Series([np.nan] + values)
    tm.assert_series_equal(result, expected)
    # GH 42064
    # new `roll_var` will output 0.0 correctly
    tm.assert_series_equal(result == 0, expected == 0)


def test_timeoffset_as_window_parameter_for_corr(unit):
    # GH: 28266
    dti = DatetimeIndex(
        [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130102 09:00:02"),
            Timestamp("20130103 09:00:03"),
            Timestamp("20130105 09:00:05"),
            Timestamp("20130106 09:00:06"),
        ]
    ).as_unit(unit)
    mi = MultiIndex.from_product([dti, ["B", "A"]])

    exp = DataFrame(
        {
            "B": [
                np.nan,
                np.nan,
                0.9999999999999998,
                -1.0,
                1.0,
                -0.3273268353539892,
                0.9999999999999998,
                1.0,
                0.9999999999999998,
                1.0,
            ],
            "A": [
                np.nan,
                np.nan,
                -1.0,
                1.0000000000000002,
                -0.3273268353539892,
                0.9999999999999966,
                1.0,
                1.0000000000000002,
                1.0,
                1.0000000000000002,
            ],
        },
        index=mi,
    )

    df = DataFrame(
        {"B": [0, 1, 2, 4, 3], "A": [7, 4, 6, 9, 3]},
        index=dti,
    )

    res = df.rolling(window="3d").corr()

    tm.assert_frame_equal(exp, res)


@pytest.mark.parametrize("method", ["var", "sum", "mean", "skew", "kurt", "min", "max"])
def test_rolling_decreasing_indices(method):
    """
    Make sure that decreasing indices give the same results as increasing indices.

    GH 36933
    """
    df = DataFrame({"values": np.arange(-15, 10) ** 2})
    df_reverse = DataFrame({"values": df["values"][::-1]}, index=df.index[::-1])

    increasing = getattr(df.rolling(window=5), method)()
    decreasing = getattr(df_reverse.rolling(window=5), method)()

    assert np.abs(decreasing.values[::-1][:-4] - increasing.values[4:]).max() < 1e-12


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("2s", "right", [1.0, 3.0, 5.0, 3.0]),
        ("2s", "left", [0.0, 1.0, 3.0, 5.0]),
        ("2s", "both", [1.0, 3.0, 6.0, 5.0]),
        ("2s", "neither", [0.0, 1.0, 2.0, 3.0]),
        ("3s", "right", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "left", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "both", [1.0, 3.0, 6.0, 5.0]),
        ("3s", "neither", [1.0, 3.0, 6.0, 5.0]),
    ],
)
def test_rolling_decreasing_indices_centered(window, closed, expected, frame_or_series):
    """
    Ensure that a symmetrical inverted index return same result as non-inverted.
    """
    #  GH 43927

    index = date_range("2020", periods=4, freq="1s")
    df_inc = frame_or_series(range(4), index=index)
    df_dec = frame_or_series(range(4), index=index[::-1])

    expected_inc = frame_or_series(expected, index=index)
    expected_dec = frame_or_series(expected, index=index[::-1])

    result_inc = df_inc.rolling(window, closed=closed, center=True).sum()
    result_dec = df_dec.rolling(window, closed=closed, center=True).sum()

    tm.assert_equal(result_inc, expected_inc)
    tm.assert_equal(result_dec, expected_dec)


@pytest.mark.parametrize(
    "window,expected",
    [
        ("1ns", [1.0, 1.0, 1.0, 1.0]),
        ("3ns", [2.0, 3.0, 3.0, 2.0]),
    ],
)
def test_rolling_center_nanosecond_resolution(
    window, closed, expected, frame_or_series
):
    index = date_range("2020", periods=4, freq="1ns")
    df = frame_or_series([1, 1, 1, 1], index=index, dtype=float)
    expected = frame_or_series(expected, index=index, dtype=float)
    result = df.rolling(window, closed=closed, center=True).sum()
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            "var",
            [
                float("nan"),
                43.0,
                float("nan"),
                136.333333,
                43.5,
                94.966667,
                182.0,
                318.0,
            ],
        ),
        (
            "mean",
            [float("nan"), 7.5, float("nan"), 21.5, 6.0, 9.166667, 13.0, 17.5],
        ),
        (
            "sum",
            [float("nan"), 30.0, float("nan"), 86.0, 30.0, 55.0, 91.0, 140.0],
        ),
        (
            "skew",
            [
                float("nan"),
                0.709296,
                float("nan"),
                0.407073,
                0.984656,
                0.919184,
                0.874674,
                0.842418,
            ],
        ),
        (
            "kurt",
            [
                float("nan"),
                -0.5916711736073559,
                float("nan"),
                -1.0028993131317954,
                -0.06103844629409494,
                -0.254143227116194,
                -0.37362637362637585,
                -0.45439658241367054,
            ],
        ),
    ],
)
def test_rolling_non_monotonic(method, expected):
    """
    Make sure the (rare) branch of non-monotonic indices is covered by a test.

    output from 1.1.3 is assumed to be the expected output. Output of sum/mean has
    manually been verified.

    GH 36933.
    """
    # Based on an example found in computation.rst
    use_expanding = [True, False, True, False, True, True, True, True]
    df = DataFrame({"values": np.arange(len(use_expanding)) ** 2})

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=4, use_expanding=use_expanding)

    result = getattr(df.rolling(indexer), method)()
    expected = DataFrame({"values": expected})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("index", "window"),
    [
        ([0, 1, 2, 3, 4], 2),
        (date_range("2001-01-01", freq="D", periods=5), "2D"),
    ],
)
def test_rolling_corr_timedelta_index(index, window):
    # GH: 31286
    x = Series([1, 2, 3, 4, 5], index=index)
    y = x.copy()
    x.iloc[0:2] = 0.0
    result = x.rolling(window).corr(y)
    expected = Series([np.nan, np.nan, 1, 1, 1], index=index)
    tm.assert_almost_equal(result, expected)


def test_groupby_rolling_nan_included():
    # GH 35542
    data = {"group": ["g1", np.nan, "g1", "g2", np.nan], "B": [0, 1, 2, 3, 4]}
    df = DataFrame(data)
    result = df.groupby("group", dropna=False).rolling(1, min_periods=1).mean()
    expected = DataFrame(
        {"B": [0.0, 2.0, 3.0, 1.0, 4.0]},
        # GH-38057 from_tuples puts the NaNs in the codes, result expects them
        # to be in the levels, at the moment
        # index=MultiIndex.from_tuples(
        #     [("g1", 0), ("g1", 2), ("g2", 3), (np.nan, 1), (np.nan, 4)],
        #     names=["group", None],
        # ),
        index=MultiIndex(
            [["g1", "g2", np.nan], [0, 1, 2, 3, 4]],
            [[0, 0, 1, 2, 2], [0, 2, 3, 1, 4]],
            names=["group", None],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_rolling_skew_kurt_numerical_stability(method):
    # GH#6929
    ser = Series(np.random.default_rng(2).random(10))
    ser_copy = ser.copy()
    expected = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(ser, ser_copy)
    ser = ser + 50000
    result = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("method", "values"),
    [
        ("skew", [2.0, 0.854563, 0.0, 1.999984]),
        ("kurt", [4.0, -1.289256, -1.2, 3.999946]),
    ],
)
def test_rolling_skew_kurt_large_value_range(method, values):
    # GH: 37557
    s = Series([3000000, 1, 1, 2, 3, 4, 999])
    result = getattr(s.rolling(4), method)()
    expected = Series([np.nan] * 3 + values)
    tm.assert_series_equal(result, expected)


def test_invalid_method():
    with pytest.raises(ValueError, match="method must be 'table' or 'single"):
        Series(range(1)).rolling(1, method="foo")


@pytest.mark.parametrize("window", [1, "1d"])
def test_rolling_descending_date_order_with_offset(window, frame_or_series):
    # GH#40002
    idx = date_range(start="2020-01-01", end="2020-01-03", freq="1d")
    obj = frame_or_series(range(1, 4), index=idx)
    result = obj.rolling("1d", closed="left").sum()
    expected = frame_or_series([np.nan, 1, 2], index=idx)
    tm.assert_equal(result, expected)

    result = obj.iloc[::-1].rolling("1d", closed="left").sum()
    idx = date_range(start="2020-01-03", end="2020-01-01", freq="-1d")
    expected = frame_or_series([np.nan, 3, 2], index=idx)
    tm.assert_equal(result, expected)


def test_rolling_var_floating_artifact_precision():
    # GH 37051
    s = Series([7, 5, 5, 5])
    result = s.rolling(3).var()
    expected = Series([np.nan, np.nan, 4 / 3, 0])
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)
    # GH 42064
    # new `roll_var` will output 0.0 correctly
    tm.assert_series_equal(result == 0, expected == 0)


def test_rolling_std_small_values():
    # GH 37051
    s = Series(
        [
            0.00000054,
            0.00000053,
            0.00000054,
        ]
    )
    result = s.rolling(2).std()
    expected = Series([np.nan, 7.071068e-9, 7.071068e-9])
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)


@pytest.mark.parametrize(
    "start, exp_values",
    [
        (1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]),
        (2, [0.001, 0.001, 0.0015, 0.00366666]),
    ],
)
def test_rolling_mean_all_nan_window_floating_artifacts(start, exp_values):
    # GH#41053
    df = DataFrame(
        [
            0.03,
            0.03,
            0.001,
            np.nan,
            0.002,
            0.008,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.005,
            0.2,
        ]
    )

    values = exp_values + [
        0.00366666,
        0.005,
        0.005,
        0.008,
        np.nan,
        np.nan,
        0.005,
        0.102500,
    ]
    expected = DataFrame(
        values,
        index=list(range(start, len(values) + start)),
    )
    result = df.iloc[start:].rolling(5, min_periods=0).mean()
    tm.assert_frame_equal(result, expected)


def test_rolling_sum_all_nan_window_floating_artifacts():
    # GH#41053
    df = DataFrame([0.002, 0.008, 0.005, np.nan, np.nan, np.nan])
    result = df.rolling(3, min_periods=0).sum()
    expected = DataFrame([0.002, 0.010, 0.015, 0.013, 0.005, 0.0])
    tm.assert_frame_equal(result, expected)


def test_rolling_zero_window():
    # GH 22719
    s = Series(range(1))
    result = s.rolling(0).min()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_rolling_float_dtype(float_numpy_dtype):
    # GH#42452
    df = DataFrame({"A": range(5), "B": range(10, 15)}, dtype=float_numpy_dtype)
    expected = DataFrame(
        {"A": [np.nan] * 5, "B": range(10, 20, 2)},
        dtype=float_numpy_dtype,
    )
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(2, axis=1).sum()
    tm.assert_frame_equal(result, expected, check_dtype=False)


def test_rolling_numeric_dtypes():
    # GH#41779
    df = DataFrame(np.arange(40).reshape(4, 10), columns=list("abcdefghij")).astype(
        {
            "a": "float16",
            "b": "float32",
            "c": "float64",
            "d": "int8",
            "e": "int16",
            "f": "int32",
            "g": "uint8",
            "h": "uint16",
            "i": "uint32",
            "j": "uint64",
        }
    )
    msg = "Support for axis=1 in DataFrame.rolling is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.rolling(window=2, min_periods=1, axis=1).min()
    expected = DataFrame(
        {
            "a": range(0, 40, 10),
            "b": range(0, 40, 10),
            "c": range(1, 40, 10),
            "d": range(2, 40, 10),
            "e": range(3, 40, 10),
            "f": range(4, 40, 10),
            "g": range(5, 40, 10),
            "h": range(6, 40, 10),
            "i": range(7, 40, 10),
            "j": range(8, 40, 10),
        },
        dtype="float64",
    )
    tm.assert_frame_equal(result, expected)


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

    expected = ser.rolling(window).apply(
        lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1]
    )
    result = ser.rolling(window).rank(method=method, pct=pct, ascending=ascending)

    tm.assert_series_equal(result, expected)


def test_rolling_quantile_np_percentile():
    # #9413: Tests that rolling window's quantile default behavior
    # is analogous to Numpy's percentile
    row = 10
    col = 5
    idx = date_range("20100101", periods=row, freq="B")
    df = DataFrame(
        np.random.default_rng(2).random(row * col).reshape((row, -1)), index=idx
    )

    df_quantile = df.quantile([0.25, 0.5, 0.75], axis=0)
    np_percentile = np.percentile(df, [25, 50, 75], axis=0)

    tm.assert_almost_equal(df_quantile.values, np.array(np_percentile))


@pytest.mark.parametrize("quantile", [0.0, 0.1, 0.45, 0.5, 1])
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [8.0, 1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0],
        [0.0, np.nan, 0.2, np.nan, 0.4],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0.1, np.nan, 0.3, 0.4, 0.5],
        [0.5],
        [np.nan, 0.7, 0.6],
    ],
)
def test_rolling_quantile_interpolation_options(quantile, interpolation, data):
    # Tests that rolling window's quantile behavior is analogous to
    # Series' quantile for each interpolation option
    s = Series(data)

    q1 = s.quantile(quantile, interpolation)
    q2 = s.expanding(min_periods=1).quantile(quantile, interpolation).iloc[-1]

    if np.isnan(q1):
        assert np.isnan(q2)
    else:
        if not IS64:
            # Less precision on 32-bit
            assert np.allclose([q1], [q2], rtol=1e-07, atol=0)
        else:
            assert q1 == q2


def test_invalid_quantile_value():
    data = np.arange(5)
    s = Series(data)

    msg = "Interpolation 'invalid' is not supported"
    with pytest.raises(ValueError, match=msg):
        s.rolling(len(data), min_periods=1).quantile(0.5, interpolation="invalid")


def test_rolling_quantile_param():
    ser = Series([0.0, 0.1, 0.5, 0.9, 1.0])
    msg = "quantile value -0.1 not in \\[0, 1\\]"
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(-0.1)

    msg = "quantile value 10.0 not in \\[0, 1\\]"
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(10.0)

    msg = "must be real number, not str"
    with pytest.raises(TypeError, match=msg):
        ser.rolling(3).quantile("foo")


def test_rolling_std_1obs():
    vals = Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = vals.rolling(1, min_periods=1).std()
    expected = Series([np.nan] * 5)
    tm.assert_series_equal(result, expected)

    result = vals.rolling(1, min_periods=1).std(ddof=0)
    expected = Series([0.0] * 5)
    tm.assert_series_equal(result, expected)

    result = Series([np.nan, np.nan, 3, 4, 5]).rolling(3, min_periods=2).std()
    assert np.isnan(result[2])


def test_rolling_std_neg_sqrt():
    # unit test from Bottleneck

    # Test move_nanstd for neg sqrt.

    a = Series(
        [
            0.0011448196318903589,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
        ]
    )
    b = a.rolling(window=3).std()
    assert np.isfinite(b[2:]).all()

    b = a.ewm(span=3).std()
    assert np.isfinite(b[2:]).all()


def test_step_not_integer_raises():
    with pytest.raises(ValueError, match="step must be an integer"):
        DataFrame(range(2)).rolling(1, step="foo")


def test_step_not_positive_raises():
    with pytest.raises(ValueError, match="step must be >= 0"):
        DataFrame(range(2)).rolling(1, step=-1)


@pytest.mark.parametrize(
    ["values", "window", "min_periods", "expected"],
    [
        [
            [20, 10, 10, np.inf, 1, 1, 2, 3],
            3,
            1,
            [np.nan, 50, 100 / 3, 0, 40.5, 0, 1 / 3, 1],
        ],
        [
            [20, 10, 10, np.nan, 10, 1, 2, 3],
            3,
            1,
            [np.nan, 50, 100 / 3, 0, 0, 40.5, 73 / 3, 1],
        ],
        [
            [np.nan, 5, 6, 7, 5, 5, 5],
            3,
            3,
            [np.nan] * 3 + [1, 1, 4 / 3, 0],
        ],
        [
            [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
            3,
            3,
            [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [1 / 3, 0],
        ],
        [
            [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
            3,
            3,
            [np.nan] * 2 + [4 / 3, 0] + [np.nan] * 4 + [16 / 3, 0],
        ],
        [
            [5, 7] * 4,
            3,
            3,
            [np.nan] * 2 + [4 / 3] * 6,
        ],
        [
            [5, 7, 5, np.nan, 7, 5, 7],
            3,
            2,
            [np.nan, 2, 4 / 3] + [2] * 3 + [4 / 3],
        ],
    ],
)
def test_rolling_var_same_value_count_logic(values, window, min_periods, expected):
    # GH 42064.

    expected = Series(expected)
    sr = Series(values)

    # With new algo implemented, result will be set to .0 in rolling var
    # if sufficient amount of consecutively same values are found.
    result_var = sr.rolling(window, min_periods=min_periods).var()

    # use `assert_series_equal` twice to check for equality,
    # because `check_exact=True` will fail in 32-bit tests due to
    # precision loss.

    # 1. result should be close to correct value
    # non-zero values can still differ slightly from "truth"
    # as the result of online algorithm
    tm.assert_series_equal(result_var, expected)
    # 2. zeros should be exactly the same since the new algo takes effect here
    tm.assert_series_equal(expected == 0, result_var == 0)

    # std should also pass as it's just a sqrt of var
    result_std = sr.rolling(window, min_periods=min_periods).std()
    tm.assert_series_equal(result_std, np.sqrt(expected))
    tm.assert_series_equal(expected == 0, result_std == 0)


def test_rolling_mean_sum_floating_artifacts():
    # GH 42064.

    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(3)
    result = r.mean()
    assert (result[-3:] == 0).all()
    result = r.sum()
    assert (result[-3:] == 0).all()


def test_rolling_skew_kurt_floating_artifacts():
    # GH 42064 46431

    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(4)
    result = r.skew()
    assert (result[-2:] == 0).all()
    result = r.kurt()
    assert (result[-2:] == -3).all()


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # GH#46560
    kernel = arithmetic_win_operators
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    rolling = df.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
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
    rolling = df.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    result = op(*arg, numeric_only=numeric_only)

    # Compare result to op using float dtypes, dropping c when numeric_only is True
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    rolling2 = df2.rolling(2, min_periods=1)
    op2 = getattr(rolling2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # GH#46560
    kernel = arithmetic_win_operators
    ser = Series([1], dtype=dtype)
    rolling = ser.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    if numeric_only and dtype is object:
        msg = f"Rolling.{kernel} does not implement numeric_only"
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
    rolling = ser.rolling(2, min_periods=1)
    op = getattr(rolling, kernel)
    if numeric_only and dtype is object:
        msg = f"Rolling.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        result = op(*arg, numeric_only=numeric_only)

        ser2 = ser.astype(float)
        arg2 = (ser2,) if use_arg else ()
        rolling2 = ser2.rolling(2, min_periods=1)
        op2 = getattr(rolling2, kernel)
        expected = op2(*arg2, numeric_only=numeric_only)
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Prague"])
def test_rolling_timedelta_window_non_nanoseconds(unit, tz):
    # Test Sum, GH#55106
    df_time = DataFrame(
        {"A": range(5)}, index=date_range("2013-01-01", freq="1s", periods=5, tz=tz)
    )
    sum_in_nanosecs = df_time.rolling("1s").sum()
    # microseconds / milliseconds should not break the correct rolling
    df_time.index = df_time.index.as_unit(unit)
    sum_in_microsecs = df_time.rolling("1s").sum()
    sum_in_microsecs.index = sum_in_microsecs.index.as_unit("ns")
    tm.assert_frame_equal(sum_in_nanosecs, sum_in_microsecs)

    # Test max, GH#55026
    ref_dates = date_range("2023-01-01", "2023-01-10", unit="ns", tz=tz)
    ref_series = Series(0, index=ref_dates)
    ref_series.iloc[0] = 1
    ref_max_series = ref_series.rolling(Timedelta(days=4)).max()

    dates = date_range("2023-01-01", "2023-01-10", unit=unit, tz=tz)
    series = Series(0, index=dates)
    series.iloc[0] = 1
    max_series = series.rolling(Timedelta(days=4)).max()

    ref_df = DataFrame(ref_max_series)
    df = DataFrame(max_series)
    df.index = df.index.as_unit("ns")

    tm.assert_frame_equal(ref_df, df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\umath.py ===
"""
Create the numpy._core.umath namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.

"""

import numpy

from . import _multiarray_umath
from ._multiarray_umath import *

# These imports are needed for backward compatibility,
# do not change them. issue gh-11862
# _ones_like is semi-public, on purpose not added to __all__
# These imports are needed for the strip & replace implementations
from ._multiarray_umath import (
    _UFUNC_API,
    _add_newdoc_ufunc,
    _center,
    _expandtabs,
    _expandtabs_length,
    _extobj_contextvar,
    _get_extobj_dict,
    _ljust,
    _lstrip_chars,
    _lstrip_whitespace,
    _make_extobj,
    _ones_like,
    _partition,
    _partition_index,
    _replace,
    _rjust,
    _rpartition,
    _rpartition_index,
    _rstrip_chars,
    _rstrip_whitespace,
    _slice,
    _strip_chars,
    _strip_whitespace,
    _zfill,
)

__all__ = [
    'absolute', 'add',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj',
    'conjugate', 'copysign', 'cos', 'cosh', 'bitwise_count', 'deg2rad',
    'degrees', 'divide', 'divmod', 'e', 'equal', 'euler_gamma', 'exp', 'exp2',
    'expm1', 'fabs', 'floor', 'floor_divide', 'float_power', 'fmax', 'fmin',
    'fmod', 'frexp', 'frompyfunc', 'gcd', 'greater', 'greater_equal',
    'heaviside', 'hypot', 'invert', 'isfinite', 'isinf', 'isnan', 'isnat',
    'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'log', 'log10',
    'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not',
    'logical_or', 'logical_xor', 'matvec', 'maximum', 'minimum', 'mod', 'modf',
    'multiply', 'negative', 'nextafter', 'not_equal', 'pi', 'positive',
    'power', 'rad2deg', 'radians', 'reciprocal', 'remainder', 'right_shift',
    'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square',
    'subtract', 'tan', 'tanh', 'true_divide', 'trunc', 'vecdot', 'vecmat']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\inference.py ===
# Copyright (c) 2010-2024 openpyxl

"""
Type inference functions
"""
import datetime
import re

from openpyxl.styles import numbers

PERCENT_REGEX = re.compile(r'^(?P<number>\-?[0-9]*\.?[0-9]*\s?)\%$')
TIME_REGEX = re.compile(r"""
^(?: # HH:MM and HH:MM:SS
(?P<hour>[0-1]{0,1}[0-9]{2}):
(?P<minute>[0-5][0-9]):?
(?P<second>[0-5][0-9])?$)
|
^(?: # MM:SS.
([0-5][0-9]):
([0-5][0-9])?\.
(?P<microsecond>\d{1,6}))
""", re.VERBOSE)
NUMBER_REGEX = re.compile(r'^-?([\d]|[\d]+\.[\d]*|\.[\d]+|[1-9][\d]+\.?[\d]*)((E|e)[-+]?[\d]+)?$')


def cast_numeric(value):
    """Explicitly convert a string to a numeric value"""
    if NUMBER_REGEX.match(value):
        try:
            return int(value)
        except ValueError:
            return float(value)


def cast_percentage(value):
    """Explicitly convert a string to numeric value and format as a
    percentage"""
    match = PERCENT_REGEX.match(value)
    if match:
        return float(match.group('number')) / 100



def cast_time(value):
    """Explicitly convert a string to a number and format as datetime or
    time"""
    match = TIME_REGEX.match(value)
    if match:
        if match.group("microsecond") is not None:
            value = value[:12]
            pattern = "%M:%S.%f"
            #fmt = numbers.FORMAT_DATE_TIME5
        elif match.group('second') is None:
            #fmt = numbers.FORMAT_DATE_TIME3
            pattern = "%H:%M"
        else:
            pattern = "%H:%M:%S"
            #fmt = numbers.FORMAT_DATE_TIME6
        value = datetime.datetime.strptime(value, pattern)
        return value.time()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\chromium\remote_connection.py ===
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
from typing import Optional

from selenium.webdriver.remote.client_config import ClientConfig
from selenium.webdriver.remote.remote_connection import RemoteConnection


class ChromiumRemoteConnection(RemoteConnection):
    def __init__(
        self,
        remote_server_addr: str,
        vendor_prefix: str,
        browser_name: str,
        keep_alive: bool = True,
        ignore_proxy: Optional[bool] = False,
        client_config: Optional[ClientConfig] = None,
    ) -> None:
        client_config = client_config or ClientConfig(
            remote_server_addr=remote_server_addr, keep_alive=keep_alive, timeout=120
        )
        super().__init__(
            ignore_proxy=ignore_proxy,
            client_config=client_config,
        )
        self.browser_name = browser_name
        commands = self._remote_commands(vendor_prefix)
        for key, value in commands.items():
            self._commands[key] = value

    def _remote_commands(self, vendor_prefix):
        remote_commands = {
            "launchApp": ("POST", "/session/$sessionId/chromium/launch_app"),
            "setPermissions": ("POST", "/session/$sessionId/permissions"),
            "setNetworkConditions": ("POST", "/session/$sessionId/chromium/network_conditions"),
            "getNetworkConditions": ("GET", "/session/$sessionId/chromium/network_conditions"),
            "deleteNetworkConditions": ("DELETE", "/session/$sessionId/chromium/network_conditions"),
            "executeCdpCommand": ("POST", f"/session/$sessionId/{vendor_prefix}/cdp/execute"),
            "getSinks": ("GET", f"/session/$sessionId/{vendor_prefix}/cast/get_sinks"),
            "getIssueMessage": ("GET", f"/session/$sessionId/{vendor_prefix}/cast/get_issue_message"),
            "setSinkToUse": ("POST", f"/session/$sessionId/{vendor_prefix}/cast/set_sink_to_use"),
            "startDesktopMirroring": ("POST", f"/session/$sessionId/{vendor_prefix}/cast/start_desktop_mirroring"),
            "startTabMirroring": ("POST", f"/session/$sessionId/{vendor_prefix}/cast/start_tab_mirroring"),
            "stopCasting": ("POST", f"/session/$sessionId/{vendor_prefix}/cast/stop_casting"),
        }
        return remote_commands

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\rewritingsystem_fsm.py ===
class State:
    '''
    A representation of a state managed by a ``StateMachine``.

    Attributes:
        name (instance of FreeGroupElement or string) -- State name which is also assigned to the Machine.
        transisitons (OrderedDict) -- Represents all the transitions of the state object.
        state_type (string) -- Denotes the type (accept/start/dead) of the state.
        rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.
        state_machine (instance of StateMachine object) -- The finite state machine that the state belongs to.
    '''

    def __init__(self, name, state_machine, state_type=None, rh_rule=None):
        self.name = name
        self.transitions = {}
        self.state_machine = state_machine
        self.state_type = state_type[0]
        self.rh_rule = rh_rule

    def add_transition(self, letter, state):
        '''
        Add a transition from the current state to a new state.

        Keyword Arguments:
            letter -- The alphabet element the current state reads to make the state transition.
            state -- This will be an instance of the State object which represents a new state after in the transition after the alphabet is read.

        '''
        self.transitions[letter] = state

class StateMachine:
    '''
    Representation of a finite state machine the manages the states and the transitions of the automaton.

    Attributes:
        states (dictionary) -- Collection of all registered `State` objects.
        name (str) -- Name of the state machine.
    '''

    def __init__(self, name, automaton_alphabet):
        self.name = name
        self.automaton_alphabet = automaton_alphabet
        self.states = {} # Contains all the states in the machine.
        self.add_state('start', state_type='s')

    def add_state(self, state_name, state_type=None, rh_rule=None):
        '''
        Instantiate a state object and stores it in the 'states' dictionary.

        Arguments:
            state_name (instance of FreeGroupElement or string) -- name of the new states.
            state_type (string) -- Denotes the type (accept/start/dead) of the state added.
            rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.

        '''
        new_state = State(state_name, self, state_type, rh_rule)
        self.states[state_name] = new_state

    def __repr__(self):
        return "%s" % (self.name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\adjoint.py ===
from sympy.core import Basic
from sympy.functions import adjoint, conjugate
from sympy.matrices.expressions.matexpr import MatrixExpr


class Adjoint(MatrixExpr):
    """
    The Hermitian adjoint of a matrix expression.

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the adjoint, use the ``adjoint()``
    function.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Adjoint, adjoint
    >>> A = MatrixSymbol('A', 3, 5)
    >>> B = MatrixSymbol('B', 5, 3)
    >>> Adjoint(A*B)
    Adjoint(A*B)
    >>> adjoint(A*B)
    Adjoint(B)*Adjoint(A)
    >>> adjoint(A*B) == Adjoint(A*B)
    False
    >>> adjoint(A*B) == Adjoint(A*B).doit()
    True
    """
    is_Adjoint = True

    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True) and isinstance(arg, Basic):
            return adjoint(arg.doit(**hints))
        else:
            return adjoint(self.arg)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape[::-1]

    def _entry(self, i, j, **kwargs):
        return conjugate(self.arg._entry(j, i, **kwargs))

    def _eval_adjoint(self):
        return self.arg

    def _eval_transpose(self):
        return self.arg.conjugate()

    def _eval_conjugate(self):
        return self.arg.transpose()

    def _eval_trace(self):
        from sympy.matrices.expressions.trace import Trace
        return conjugate(Trace(self.arg))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\__init__.py ===
"""The module helps converting SymPy expressions into shorter forms of them.

for example:
the expression E**(pi*I) will be converted into -1
the expression (x+x)**2 will be converted into 4*x**2
"""
from .simplify import (simplify, hypersimp, hypersimilar,
    logcombine, separatevars, posify, besselsimp, kroneckersimp,
    signsimp, nsimplify)

from .fu import FU, fu

from .sqrtdenest import sqrtdenest

from .cse_main import cse

from .epathtools import epath, EPath

from .hyperexpand import hyperexpand

from .radsimp import collect, rcollect, radsimp, collect_const, fraction, numer, denom

from .trigsimp import trigsimp, exptrigsimp

from .powsimp import powsimp, powdenest

from .combsimp import combsimp

from .gammasimp import gammasimp

from .ratsimp import ratsimp, ratsimpmodprime

__all__ = [
    'simplify', 'hypersimp', 'hypersimilar', 'logcombine', 'separatevars',
    'posify', 'besselsimp', 'kroneckersimp', 'signsimp',
    'nsimplify',

    'FU', 'fu',

    'sqrtdenest',

    'cse',

    'epath', 'EPath',

    'hyperexpand',

    'collect', 'rcollect', 'radsimp', 'collect_const', 'fraction', 'numer',
    'denom',

    'trigsimp', 'exptrigsimp',

    'powsimp', 'powdenest',

    'combsimp',

    'gammasimp',

    'ratsimp', 'ratsimpmodprime',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\hash.py ===
import hashlib
import logging
import sys
from optparse import Values
from typing import List

from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.utils.hashes import FAVORITE_HASH, STRONG_HASHES
from pip._internal.utils.misc import read_chunks, write_output

logger = logging.getLogger(__name__)


class HashCommand(Command):
    """
    Compute a hash of a local package archive.

    These can be used with --hash in a requirements file to do repeatable
    installs.
    """

    usage = "%prog [options] <file> ..."
    ignore_require_venv = True

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            "-a",
            "--algorithm",
            dest="algorithm",
            choices=STRONG_HASHES,
            action="store",
            default=FAVORITE_HASH,
            help="The hash algorithm to use: one of {}".format(
                ", ".join(STRONG_HASHES)
            ),
        )
        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        if not args:
            self.parser.print_usage(sys.stderr)
            return ERROR

        algorithm = options.algorithm
        for path in args:
            write_output(
                "%s:\n--hash=%s:%s", path, algorithm, _hash_of_file(path, algorithm)
            )
        return SUCCESS


def _hash_of_file(path: str, algorithm: str) -> str:
    """Return the hash digest of a file."""
    with open(path, "rb") as archive:
        hash = hashlib.new(algorithm)
        for chunk in read_chunks(archive):
            hash.update(chunk)
    return hash.hexdigest()