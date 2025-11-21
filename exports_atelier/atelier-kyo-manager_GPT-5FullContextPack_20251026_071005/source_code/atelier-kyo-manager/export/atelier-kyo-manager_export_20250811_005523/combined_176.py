
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\globals.py ===
from __future__ import annotations

import typing as t
from threading import local

if t.TYPE_CHECKING:
    from .core import Context

_local = local()


@t.overload
def get_current_context(silent: t.Literal[False] = False) -> Context: ...


@t.overload
def get_current_context(silent: bool = ...) -> Context | None: ...


def get_current_context(silent: bool = False) -> Context | None:
    """Returns the current click context.  This can be used as a way to
    access the current context object from anywhere.  This is a more implicit
    alternative to the :func:`pass_context` decorator.  This function is
    primarily useful for helpers such as :func:`echo` which might be
    interested in changing its behavior based on the current context.

    To push the current context, :meth:`Context.scope` can be used.

    .. versionadded:: 5.0

    :param silent: if set to `True` the return value is `None` if no context
                   is available.  The default behavior is to raise a
                   :exc:`RuntimeError`.
    """
    try:
        return t.cast("Context", _local.stack[-1])
    except (AttributeError, IndexError) as e:
        if not silent:
            raise RuntimeError("There is no active click context.") from e

    return None


def push_context(ctx: Context) -> None:
    """Pushes a new context to the current stack."""
    _local.__dict__.setdefault("stack", []).append(ctx)


def pop_context() -> None:
    """Removes the top level from the stack."""
    _local.stack.pop()


def resolve_color_default(color: bool | None = None) -> bool | None:
    """Internal helper to get the default value of the color flag.  If a
    value is passed it's returned unchanged, otherwise it's looked up from
    the current context.
    """
    if color is not None:
        return color

    ctx = get_current_context(silent=True)

    if ctx is not None:
        return ctx.color

    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\_examples\numba\extending_distributions.py ===
r"""
Building the required library in this example requires a source distribution
of NumPy or clone of the NumPy git repository since distributions.c is not
included in binary distributions.

On *nix, execute in numpy/random/src/distributions

export ${PYTHON_VERSION}=3.8 # Python version
export PYTHON_INCLUDE=#path to Python's include folder, usually \
    ${PYTHON_HOME}/include/python${PYTHON_VERSION}m
export NUMPY_INCLUDE=#path to numpy's include folder, usually \
    ${PYTHON_HOME}/lib/python${PYTHON_VERSION}/site-packages/numpy/_core/include
gcc -shared -o libdistributions.so -fPIC distributions.c \
    -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE}
mv libdistributions.so ../../_examples/numba/

On Windows

rem PYTHON_HOME and PYTHON_VERSION are setup dependent, this is an example
set PYTHON_HOME=c:\Anaconda
set PYTHON_VERSION=38
cl.exe /LD .\distributions.c -DDLL_EXPORT \
    -I%PYTHON_HOME%\lib\site-packages\numpy\_core\include \
    -I%PYTHON_HOME%\include %PYTHON_HOME%\libs\python%PYTHON_VERSION%.lib
move distributions.dll ../../_examples/numba/
"""
import os

import numba as nb
from cffi import FFI

import numpy as np
from numpy.random import PCG64

ffi = FFI()
if os.path.exists('./distributions.dll'):
    lib = ffi.dlopen('./distributions.dll')
elif os.path.exists('./libdistributions.so'):
    lib = ffi.dlopen('./libdistributions.so')
else:
    raise RuntimeError('Required DLL/so file was not found.')

ffi.cdef("""
double random_standard_normal(void *bitgen_state);
""")
x = PCG64()
xffi = x.cffi
bit_generator = xffi.bit_generator

random_standard_normal = lib.random_standard_normal


def normals(n, bit_generator):
    out = np.empty(n)
    for i in range(n):
        out[i] = random_standard_normal(bit_generator)
    return out


normalsj = nb.jit(normals, nopython=True)

# Numba requires a memory address for void *
# Can also get address from x.ctypes.bit_generator.value
bit_generator_address = int(ffi.cast('uintptr_t', bit_generator))

norm = normalsj(1000, bit_generator_address)
print(norm[:12])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\pooling.py ===
import onnx

from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType, attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase


class QLinearPool(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        # only try to quantize when given quantization parameters for it
        (
            data_found,
            output_scale_name,
            output_zp_name,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])

        # get quantized input tensor names, quantize input if needed
        (
            quantized_input_names,
            input_zero_point_names,
            input_scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0])

        if not data_found or quantized_input_names is None:
            return super().quantize()

        # Create an entry for output quantized value.
        qlinear_output_name = node.output[0] + TENSOR_NAME_QUANT_SUFFIX
        quantized_output_value = QuantizedValue(
            node.output[0],
            qlinear_output_name,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        # Create qlinear pool node for given type (AveragePool, etc)
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qlinear_node_name = node.name + "_quant" if node.name else ""
        qnode = onnx.helper.make_node(
            "QLinear" + node.op_type,
            [
                quantized_input_names[0],
                input_scale_names[0],
                input_zero_point_names[0],
                output_scale_name,
                output_zp_name,
            ],
            [qlinear_output_name],
            qlinear_node_name,
            **kwargs,
        )

        # add all newly created nodes
        nodes.append(qnode)
        self.quantizer.new_nodes += nodes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\ArgTypeAndIndex.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ArgTypeAndIndex(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ArgTypeAndIndex()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsArgTypeAndIndex(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ArgTypeAndIndexBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # ArgTypeAndIndex
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ArgTypeAndIndex
    def ArgType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # ArgTypeAndIndex
    def Index(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def ArgTypeAndIndexStart(builder):
    builder.StartObject(2)

def Start(builder):
    ArgTypeAndIndexStart(builder)

def ArgTypeAndIndexAddArgType(builder, argType):
    builder.PrependInt8Slot(0, argType, 0)

def AddArgType(builder, argType):
    ArgTypeAndIndexAddArgType(builder, argType)

def ArgTypeAndIndexAddIndex(builder, index):
    builder.PrependUint32Slot(1, index, 0)

def AddIndex(builder, index):
    ArgTypeAndIndexAddIndex(builder, index)

def ArgTypeAndIndexEnd(builder):
    return builder.EndObject()

def End(builder):
    return ArgTypeAndIndexEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\FloatProperty.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FloatProperty(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FloatProperty()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFloatProperty(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def FloatPropertyBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x44\x54\x43", size_prefixed=size_prefixed)

    # FloatProperty
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FloatProperty
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # FloatProperty
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def FloatPropertyStart(builder):
    builder.StartObject(2)

def Start(builder):
    FloatPropertyStart(builder)

def FloatPropertyAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    FloatPropertyAddName(builder, name)

def FloatPropertyAddValue(builder, value):
    builder.PrependFloat32Slot(1, value, 0.0)

def AddValue(builder, value):
    FloatPropertyAddValue(builder, value)

def FloatPropertyEnd(builder):
    return builder.EndObject()

def End(builder):
    return FloatPropertyEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\IntProperty.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class IntProperty(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = IntProperty()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsIntProperty(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def IntPropertyBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x44\x54\x43", size_prefixed=size_prefixed)

    # IntProperty
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # IntProperty
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # IntProperty
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

def IntPropertyStart(builder):
    builder.StartObject(2)

def Start(builder):
    IntPropertyStart(builder)

def IntPropertyAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    IntPropertyAddName(builder, name)

def IntPropertyAddValue(builder, value):
    builder.PrependInt64Slot(1, value, 0)

def AddValue(builder, value):
    IntPropertyAddValue(builder, value)

def IntPropertyEnd(builder):
    return builder.EndObject()

def End(builder):
    return IntPropertyEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\OperatorSetId.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class OperatorSetId(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OperatorSetId()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsOperatorSetId(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def OperatorSetIdBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # OperatorSetId
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OperatorSetId
    def Domain(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # OperatorSetId
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

def OperatorSetIdStart(builder):
    builder.StartObject(2)

def Start(builder):
    OperatorSetIdStart(builder)

def OperatorSetIdAddDomain(builder, domain):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(domain), 0)

def AddDomain(builder, domain):
    OperatorSetIdAddDomain(builder, domain)

def OperatorSetIdAddVersion(builder, version):
    builder.PrependInt64Slot(1, version, 0)

def AddVersion(builder, version):
    OperatorSetIdAddVersion(builder, version)

def OperatorSetIdEnd(builder):
    return builder.EndObject()

def End(builder):
    return OperatorSetIdEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\StringProperty.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class StringProperty(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = StringProperty()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsStringProperty(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def StringPropertyBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x44\x54\x43", size_prefixed=size_prefixed)

    # StringProperty
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # StringProperty
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # StringProperty
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def StringPropertyStart(builder):
    builder.StartObject(2)

def Start(builder):
    StringPropertyStart(builder)

def StringPropertyAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    StringPropertyAddName(builder, name)

def StringPropertyAddValue(builder, value):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(value), 0)

def AddValue(builder, value):
    StringPropertyAddValue(builder, value)

def StringPropertyEnd(builder):
    return builder.EndObject()

def End(builder):
    return StringPropertyEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\StringStringEntry.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class StringStringEntry(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = StringStringEntry()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsStringStringEntry(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def StringStringEntryBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # StringStringEntry
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # StringStringEntry
    def Key(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # StringStringEntry
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def StringStringEntryStart(builder):
    builder.StartObject(2)

def Start(builder):
    StringStringEntryStart(builder)

def StringStringEntryAddKey(builder, key):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(key), 0)

def AddKey(builder, key):
    StringStringEntryAddKey(builder, key)

def StringStringEntryAddValue(builder, value):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(value), 0)

def AddValue(builder, value):
    StringStringEntryAddValue(builder, value)

def StringStringEntryEnd(builder):
    return builder.EndObject()

def End(builder):
    return StringStringEntryEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\bubble_chart.py ===
#Autogenerated schema
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Set,
    MinMax,
    Bool,
    Integer,
    Alias,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.nested import (
    NestedNoneSet,
    NestedMinMax,
    NestedBool,
)

from ._chart import ChartBase
from .axis import TextAxis, NumericAxis
from .series import XYSeries
from .label import DataLabelList


class BubbleChart(ChartBase):

    tagname = "bubbleChart"

    varyColors = NestedBool(allow_none=True)
    ser = Sequence(expected_type=XYSeries, allow_none=True)
    dLbls = Typed(expected_type=DataLabelList, allow_none=True)
    dataLabels = Alias("dLbls")
    bubble3D = NestedBool(allow_none=True)
    bubbleScale = NestedMinMax(min=0, max=300, allow_none=True)
    showNegBubbles = NestedBool(allow_none=True)
    sizeRepresents = NestedNoneSet(values=(['area', 'w']))
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    x_axis = Typed(expected_type=NumericAxis)
    y_axis = Typed(expected_type=NumericAxis)

    _series_type = "bubble"

    __elements__ = ('varyColors', 'ser', 'dLbls', 'bubble3D', 'bubbleScale',
                    'showNegBubbles', 'sizeRepresents', 'axId')

    def __init__(self,
                 varyColors=None,
                 ser=(),
                 dLbls=None,
                 bubble3D=None,
                 bubbleScale=None,
                 showNegBubbles=None,
                 sizeRepresents=None,
                 extLst=None,
                 **kw
                ):
        self.varyColors = varyColors
        self.ser = ser
        self.dLbls = dLbls
        self.bubble3D = bubble3D
        self.bubbleScale = bubbleScale
        self.showNegBubbles = showNegBubbles
        self.sizeRepresents = sizeRepresents
        self.x_axis = NumericAxis(axId=10, crossAx=20)
        self.y_axis = NumericAxis(axId=20, crossAx=10)
        super().__init__(**kw)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\datetimelike_accumulations.py ===
"""
datetimelke_accumulations.py is for accumulations of datetimelike extension arrays
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from pandas._libs import iNaT

from pandas.core.dtypes.missing import isna


def _cum_func(
    func: Callable,
    values: np.ndarray,
    *,
    skipna: bool = True,
):
    """
    Accumulations for 1D datetimelike arrays.

    Parameters
    ----------
    func : np.cumsum, np.maximum.accumulate, np.minimum.accumulate
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation). Values is changed is modified inplace.
    skipna : bool, default True
        Whether to skip NA.
    """
    try:
        fill_value = {
            np.maximum.accumulate: np.iinfo(np.int64).min,
            np.cumsum: 0,
            np.minimum.accumulate: np.iinfo(np.int64).max,
        }[func]
    except KeyError:
        raise ValueError(f"No accumulation for {func} implemented on BaseMaskedArray")

    mask = isna(values)
    y = values.view("i8")
    y[mask] = fill_value

    if not skipna:
        mask = np.maximum.accumulate(mask)

    result = func(y)
    result[mask] = iNaT

    if values.dtype.kind in "mM":
        return result.view(values.dtype.base)
    return result


def cumsum(values: np.ndarray, *, skipna: bool = True) -> np.ndarray:
    return _cum_func(np.cumsum, values, skipna=skipna)


def cummin(values: np.ndarray, *, skipna: bool = True):
    return _cum_func(np.minimum.accumulate, values, skipna=skipna)


def cummax(values: np.ndarray, *, skipna: bool = True):
    return _cum_func(np.maximum.accumulate, values, skipna=skipna)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\check.py ===
import logging
from optparse import Values
from typing import List

from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.metadata import get_default_environment
from pip._internal.operations.check import (
    check_package_set,
    check_unsupported,
    create_package_set_from_installed,
)
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.misc import write_output

logger = logging.getLogger(__name__)


class CheckCommand(Command):
    """Verify installed packages have compatible dependencies."""

    ignore_require_venv = True
    usage = """
      %prog [options]"""

    def run(self, options: Values, args: List[str]) -> int:
        package_set, parsing_probs = create_package_set_from_installed()
        missing, conflicting = check_package_set(package_set)
        unsupported = list(
            check_unsupported(
                get_default_environment().iter_installed_distributions(),
                get_supported(),
            )
        )

        for project_name in missing:
            version = package_set[project_name].version
            for dependency in missing[project_name]:
                write_output(
                    "%s %s requires %s, which is not installed.",
                    project_name,
                    version,
                    dependency[0],
                )

        for project_name in conflicting:
            version = package_set[project_name].version
            for dep_name, dep_version, req in conflicting[project_name]:
                write_output(
                    "%s %s has requirement %s, but you have %s %s.",
                    project_name,
                    version,
                    req,
                    dep_name,
                    dep_version,
                )
        for package in unsupported:
            write_output(
                "%s %s is not supported on this platform",
                package.raw_name,
                package.version,
            )
        if missing or conflicting or parsing_probs or unsupported:
            return ERROR
        else:
            write_output("No broken requirements found.")
            return SUCCESS

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\chromium\service.py ===
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
from io import IOBase
from typing import List, Mapping, Optional

from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import service


class ChromiumService(service.Service):
    """A Service class that is responsible for the starting and stopping the
    WebDriver instance of the ChromiumDriver.

    :param executable_path: install path of the executable.
    :param port: Port for the service to run on, defaults to 0 where the operating system will decide.
    :param service_args: (Optional) List of args to be passed to the subprocess when launching the executable.
    :param log_output: (Optional) int representation of STDOUT/DEVNULL, any IO instance or String path to file.
    :param env: (Optional) Mapping of environment variables for the new process, defaults to `os.environ`.
    :param driver_path_env_key: (Optional) Environment variable to use to get the path to the driver executable.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        port: int = 0,
        service_args: Optional[List[str]] = None,
        log_output: Optional[SubprocessStdAlias] = None,
        env: Optional[Mapping[str, str]] = None,
        driver_path_env_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.service_args = service_args or []
        driver_path_env_key = driver_path_env_key or "SE_CHROMEDRIVER"

        if isinstance(log_output, str):
            self.service_args.append(f"--log-path={log_output}")
            self.log_output: Optional[IOBase] = None
        elif isinstance(log_output, IOBase):
            self.log_output = log_output
        else:
            self.log_output = log_output

        super().__init__(
            executable_path=executable_path,
            port=port,
            env=env,
            log_output=self.log_output,
            driver_path_env_key=driver_path_env_key,
            **kwargs,
        )

    def command_line_args(self) -> List[str]:
        return [f"--port={self.port}"] + self.service_args

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\webkitgtk\webdriver.py ===
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

import http.client as http_client
from typing import Optional

from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver

from .options import Options
from .service import Service


class WebDriver(RemoteWebDriver):
    """Controls the WebKitGTKDriver and allows you to drive the browser."""

    def __init__(
        self,
        options=None,
        service: Optional[Service] = None,
    ):
        """Creates a new instance of the WebKitGTK driver.

        Starts the service and then creates new instance of WebKitGTK Driver.

        :Args:
         - options : an instance of WebKitGTKOptions
         - service : Service object for handling the browser driver if you need to pass extra details
        """

        options = options if options else Options()
        self.service = service if service else Service()
        self.service.path = DriverFinder(self.service, options).get_driver_path()
        self.service.start()

        super().__init__(command_executor=self.service.service_url, options=options)
        self._is_remote = False

    def quit(self):
        """Closes the browser and shuts down the WebKitGTKDriver executable
        that is started when starting the WebKitGTKDriver."""
        try:
            super().quit()
        except http_client.BadStatusLine:
            pass
        finally:
            self.service.stop()

    def download_file(self, *args, **kwargs):
        raise NotImplementedError

    def get_downloadable_files(self, *args, **kwargs):
        raise NotImplementedError

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\wpewebkit\webdriver.py ===
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

import http.client as http_client
from typing import Optional

from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver

from .options import Options
from .service import Service


class WebDriver(RemoteWebDriver):
    """Controls the WPEWebKitDriver and allows you to drive the browser."""

    def __init__(
        self,
        options=None,
        service: Optional[Service] = None,
    ):
        """Creates a new instance of the WPEWebKit driver.

        Starts the service and then creates new instance of WPEWebKit Driver.

        :Args:
         - options : an instance of ``WPEWebKitOptions``
         - service : Service object for handling the browser driver if you need to pass extra details
        """

        options = options if options else Options()
        self.service = service if service else Service()
        self.service.path = DriverFinder(self.service, options).get_driver_path()
        self.service.start()

        super().__init__(command_executor=self.service.service_url, options=options)
        self._is_remote = False

    def quit(self):
        """Closes the browser and shuts down the WPEWebKitDriver executable
        that is started when starting the WPEWebKitDriver."""
        try:
            super().quit()
        except http_client.BadStatusLine:
            pass
        finally:
            self.service.stop()

    def download_file(self, *args, **kwargs):
        raise NotImplementedError

    def get_downloadable_files(self, *args, **kwargs):
        raise NotImplementedError

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\mariadb.py ===
# dialects/mysql/mariadb.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from .base import MariaDBIdentifierPreparer
from .base import MySQLDialect
from .base import MySQLTypeCompiler
from ...sql import sqltypes


class INET4(sqltypes.TypeEngine[str]):
    """INET4 column type for MariaDB

    .. versionadded:: 2.0.37
    """

    __visit_name__ = "INET4"


class INET6(sqltypes.TypeEngine[str]):
    """INET6 column type for MariaDB

    .. versionadded:: 2.0.37
    """

    __visit_name__ = "INET6"


class MariaDBTypeCompiler(MySQLTypeCompiler):
    def visit_INET4(self, type_, **kwargs) -> str:
        return "INET4"

    def visit_INET6(self, type_, **kwargs) -> str:
        return "INET6"


class MariaDBDialect(MySQLDialect):
    is_mariadb = True
    supports_statement_cache = True
    name = "mariadb"
    preparer = MariaDBIdentifierPreparer
    type_compiler_cls = MariaDBTypeCompiler


def loader(driver):
    dialect_mod = __import__(
        "sqlalchemy.dialects.mysql.%s" % driver
    ).dialects.mysql

    driver_mod = getattr(dialect_mod, driver)
    if hasattr(driver_mod, "mariadb_dialect"):
        driver_cls = driver_mod.mariadb_dialect
        return driver_cls
    else:
        driver_cls = driver_mod.dialect

        return type(
            "MariaDBDialect_%s" % driver,
            (
                MariaDBDialect,
                driver_cls,
            ),
            {"supports_statement_cache": True},
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\__init__.py ===
"""
Number theory module (primes, etc)
"""

from .generate import nextprime, prevprime, prime, primepi, primerange, \
    randprime, Sieve, sieve, primorial, cycle_length, composite, compositepi
from .primetest import isprime, is_gaussian_prime, is_mersenne_prime
from .factor_ import divisors, proper_divisors, factorint, multiplicity, \
    multiplicity_in_factorial, perfect_power, factor_cache, pollard_pm1, \
    pollard_rho, primefactors, totient, \
    divisor_count, proper_divisor_count, divisor_sigma, factorrat, \
    reduced_totient, primenu, primeomega, mersenne_prime_exponent, \
    is_perfect, is_abundant, is_deficient, is_amicable, is_carmichael, \
    abundance, dra, drm

from .partitions_ import npartitions
from .residue_ntheory import is_primitive_root, is_quad_residue, \
    legendre_symbol, jacobi_symbol, n_order, sqrt_mod, quadratic_residues, \
    primitive_root, nthroot_mod, is_nthpow_residue, sqrt_mod_iter, mobius, \
    discrete_log, quadratic_congruence, polynomial_congruence
from .multinomial import binomial_coefficients, binomial_coefficients_list, \
    multinomial_coefficients
from .continued_fraction import continued_fraction_periodic, \
    continued_fraction_iterator, continued_fraction_reduce, \
    continued_fraction_convergents, continued_fraction
from .digits import count_digits, digits, is_palindromic
from .egyptian_fraction import egyptian_fraction
from .ecm import ecm
from .qs import qs, qs_factor
__all__ = [
    'nextprime', 'prevprime', 'prime', 'primepi', 'primerange', 'randprime',
    'Sieve', 'sieve', 'primorial', 'cycle_length', 'composite', 'compositepi',

    'isprime', 'is_gaussian_prime', 'is_mersenne_prime',


    'divisors', 'proper_divisors', 'factorint', 'multiplicity', 'perfect_power',
    'pollard_pm1', 'factor_cache', 'pollard_rho', 'primefactors', 'totient',
    'divisor_count', 'proper_divisor_count', 'divisor_sigma', 'factorrat',
    'reduced_totient', 'primenu', 'primeomega', 'mersenne_prime_exponent',
    'is_perfect', 'is_abundant', 'is_deficient', 'is_amicable',
    'is_carmichael', 'abundance', 'dra', 'drm', 'multiplicity_in_factorial',

    'npartitions',

    'is_primitive_root', 'is_quad_residue', 'legendre_symbol',
    'jacobi_symbol', 'n_order', 'sqrt_mod', 'quadratic_residues',
    'primitive_root', 'nthroot_mod', 'is_nthpow_residue', 'sqrt_mod_iter',
    'mobius', 'discrete_log', 'quadratic_congruence', 'polynomial_congruence',

    'binomial_coefficients', 'binomial_coefficients_list',
    'multinomial_coefficients',

    'continued_fraction_periodic', 'continued_fraction_iterator',
    'continued_fraction_reduce', 'continued_fraction_convergents',
    'continued_fraction',

    'digits',
    'count_digits',
    'is_palindromic',

    'egyptian_fraction',

    'ecm',

    'qs', 'qs_factor',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\polyconfig.py ===
"""Configuration utilities for polynomial manipulation algorithms. """


from contextlib import contextmanager

_default_config = {
    'USE_COLLINS_RESULTANT':      False,
    'USE_SIMPLIFY_GCD':           True,
    'USE_HEU_GCD':                True,

    'USE_IRREDUCIBLE_IN_FACTOR':  False,
    'USE_CYCLOTOMIC_FACTOR':      True,

    'EEZ_RESTART_IF_NEEDED':      True,
    'EEZ_NUMBER_OF_CONFIGS':      3,
    'EEZ_NUMBER_OF_TRIES':        5,
    'EEZ_MODULUS_STEP':           2,

    'GF_IRRED_METHOD':            'rabin',
    'GF_FACTOR_METHOD':           'zassenhaus',

    'GROEBNER':                   'buchberger',
}

_current_config = {}

@contextmanager
def using(**kwargs):
    for k, v in kwargs.items():
        setup(k, v)

    yield

    for k in kwargs.keys():
        setup(k)

def setup(key, value=None):
    """Assign a value to (or reset) a configuration item. """
    key = key.upper()

    if value is not None:
        _current_config[key] = value
    else:
        _current_config[key] = _default_config[key]


def query(key):
    """Ask for a value of the given configuration item. """
    return _current_config.get(key.upper(), None)


def configure():
    """Initialized configuration of polys module. """
    from os import getenv

    for key, default in _default_config.items():
        value = getenv('SYMPY_' + key)

        if value is not None:
            try:
                _current_config[key] = eval(value)
            except NameError:
                _current_config[key] = value
        else:
            _current_config[key] = default

configure()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\exceptions.py ===
"""

Module to define exceptions to be used in sympy.polys.matrices modules and
classes.

Ideally all exceptions raised in these modules would be defined and documented
here and not e.g. imported from matrices. Also ideally generic exceptions like
ValueError/TypeError would not be raised anywhere.

"""


class DMError(Exception):
    """Base class for errors raised by DomainMatrix"""
    pass


class DMBadInputError(DMError):
    """list of lists is inconsistent with shape"""
    pass


class DMDomainError(DMError):
    """domains do not match"""
    pass


class DMNotAField(DMDomainError):
    """domain is not a field"""
    pass


class DMFormatError(DMError):
    """mixed dense/sparse not supported"""
    pass


class DMNonInvertibleMatrixError(DMError):
    """The matrix in not invertible"""
    pass


class DMRankError(DMError):
    """matrix does not have expected rank"""
    pass


class DMShapeError(DMError):
    """shapes are inconsistent"""
    pass


class DMNonSquareMatrixError(DMShapeError):
    """The matrix is not square"""
    pass


class DMValueError(DMError):
    """The value passed is invalid"""
    pass


__all__ = [
    'DMError', 'DMBadInputError', 'DMDomainError', 'DMFormatError',
    'DMRankError', 'DMShapeError', 'DMNotAField',
    'DMNonInvertibleMatrixError', 'DMNonSquareMatrixError', 'DMValueError'
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\kind.py ===
#sympy.vector.kind

from sympy.core.kind import Kind, _NumberKind, NumberKind
from sympy.core.mul import Mul

class VectorKind(Kind):
    """
    Kind for all vector objects in SymPy.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is
        :class:`sympy.core.kind.NumberKind`,
        which means that the vector contains only numbers.

    Examples
    ========

    Any instance of Vector class has kind ``VectorKind``:

    >>> from sympy.vector.coordsysrect import CoordSys3D
    >>> Sys = CoordSys3D('Sys')
    >>> Sys.i.kind
    VectorKind(NumberKind)

    Operations between instances of Vector keep also have the kind ``VectorKind``:

    >>> from sympy.core.add import Add
    >>> v1 = Sys.i * 2 + Sys.j * 3 + Sys.k * 4
    >>> v2 = Sys.i * Sys.x + Sys.j * Sys.y + Sys.k * Sys.z
    >>> v1.kind
    VectorKind(NumberKind)
    >>> v2.kind
    VectorKind(NumberKind)
    >>> Add(v1, v2).kind
    VectorKind(NumberKind)

    Subclasses of Vector also have the kind ``VectorKind``, such as
    Cross, VectorAdd, VectorMul or VectorZero.

    See Also
    ========

    sympy.core.kind.Kind
    sympy.matrices.kind.MatrixKind

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "VectorKind(%s)" % self.element_kind

@Mul._kind_dispatcher.register(_NumberKind, VectorKind)
def num_vec_mul(k1, k2):
    """
    The result of a multiplication between a number and a Vector should be of VectorKind.
    The element kind is selected by recursive dispatching.
    """
    if not isinstance(k2, VectorKind):
        k1, k2 = k2, k1
    elemk = Mul._kind_dispatcher(k1, k2.element_kind)
    return VectorKind(elemk)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_wait_for_object.py ===
from __future__ import annotations

import math

import trio

from ._core._windows_cffi import (
    CData,
    ErrorCodes,
    _handle,
    ffi,
    handle_array,
    kernel32,
    raise_winerror,
)


async def WaitForSingleObject(obj: int | CData) -> None:
    """Async and cancellable variant of WaitForSingleObject. Windows only.

    Args:
      handle: A Win32 handle, as a Python integer.

    Raises:
      OSError: If the handle is invalid, e.g. when it is already closed.

    """
    # Allow ints or whatever we can convert to a win handle
    handle = _handle(obj)

    # Quick check; we might not even need to spawn a thread. The zero
    # means a zero timeout; this call never blocks. We also exit here
    # if the handle is already closed for some reason.
    retcode = kernel32.WaitForSingleObject(handle, 0)
    if retcode == ErrorCodes.WAIT_FAILED:
        raise_winerror()
    elif retcode != ErrorCodes.WAIT_TIMEOUT:
        return

    # Wait for a thread that waits for two handles: the handle plus a handle
    # that we can use to cancel the thread.
    cancel_handle = kernel32.CreateEventA(ffi.NULL, True, False, ffi.NULL)
    try:
        await trio.to_thread.run_sync(
            WaitForMultipleObjects_sync,
            handle,
            cancel_handle,
            abandon_on_cancel=True,
            limiter=trio.CapacityLimiter(math.inf),
        )
    finally:
        # Clean up our cancel handle. In case we get here because this task was
        # cancelled, we also want to set the cancel_handle to stop the thread.
        kernel32.SetEvent(cancel_handle)
        kernel32.CloseHandle(cancel_handle)


def WaitForMultipleObjects_sync(*handles: int | CData) -> None:
    """Wait for any of the given Windows handles to be signaled."""
    n = len(handles)
    handle_arr = handle_array(n)
    for i in range(n):
        handle_arr[i] = handles[i]
    timeout = 0xFFFFFFFF  # INFINITE
    retcode = kernel32.WaitForMultipleObjects(n, handle_arr, False, timeout)  # blocking
    if retcode == ErrorCodes.WAIT_FAILED:
        raise_winerror()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_arithmetic.py ===
from collections import deque
from datetime import (
    datetime,
    timezone,
)
from enum import Enum
import functools
import operator
import re

import numpy as np
import pytest

from pandas.compat import HAS_PYARROW
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
    _check_mixed_float,
    _check_mixed_int,
)


@pytest.fixture
def simple_frame():
    """
    Fixture for simple 3x3 DataFrame

    Columns are ['one', 'two', 'three'], index is ['a', 'b', 'c'].

       one  two  three
    a  1.0  2.0    3.0
    b  4.0  5.0    6.0
    c  7.0  8.0    9.0
    """
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    return DataFrame(arr, columns=["one", "two", "three"], index=["a", "b", "c"])


@pytest.fixture(autouse=True, params=[0, 100], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield request.param


class DummyElement:
    def __init__(self, value, dtype) -> None:
        self.value = value
        self.dtype = np.dtype(dtype)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.value, dtype=self.dtype)

    def __str__(self) -> str:
        return f"DummyElement({self.value}, {self.dtype})"

    def __repr__(self) -> str:
        return str(self)

    def astype(self, dtype, copy=False):
        self.dtype = dtype
        return self

    def view(self, dtype):
        return type(self)(self.value.view(dtype), dtype)

    def any(self, axis=None):
        return bool(self.value)


# -------------------------------------------------------------------
# Comparisons


class TestFrameComparisons:
    # Specifically _not_ flex-comparisons

    def test_comparison_with_categorical_dtype(self):
        # GH#12564

        df = DataFrame({"A": ["foo", "bar", "baz"]})
        exp = DataFrame({"A": [True, False, False]})

        res = df == "foo"
        tm.assert_frame_equal(res, exp)

        # casting to categorical shouldn't affect the result
        df["A"] = df["A"].astype("category")

        res = df == "foo"
        tm.assert_frame_equal(res, exp)

    def test_frame_in_list(self):
        # GH#12689 this should raise at the DataFrame level, not blocks
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)), columns=list("ABCD")
        )
        msg = "The truth value of a DataFrame is ambiguous"
        with pytest.raises(ValueError, match=msg):
            df in [None]

    @pytest.mark.parametrize(
        "arg, arg2",
        [
            [
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": pd.date_range("20010101", periods=10),
                },
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": np.random.default_rng(2).integers(10, size=10),
                },
            ],
            [
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": np.random.default_rng(2).integers(10, size=10),
                },
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": pd.date_range("20010101", periods=10),
                },
            ],
            [
                {
                    "a": pd.date_range("20010101", periods=10),
                    "b": pd.date_range("20010101", periods=10),
                },
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": np.random.default_rng(2).integers(10, size=10),
                },
            ],
            [
                {
                    "a": np.random.default_rng(2).integers(10, size=10),
                    "b": pd.date_range("20010101", periods=10),
                },
                {
                    "a": pd.date_range("20010101", periods=10),
                    "b": pd.date_range("20010101", periods=10),
                },
            ],
        ],
    )
    def test_comparison_invalid(self, arg, arg2):
        # GH4968
        # invalid date/int comparisons
        x = DataFrame(arg)
        y = DataFrame(arg2)
        # we expect the result to match Series comparisons for
        # == and !=, inequalities should raise
        result = x == y
        expected = DataFrame(
            {col: x[col] == y[col] for col in x.columns},
            index=x.index,
            columns=x.columns,
        )
        tm.assert_frame_equal(result, expected)

        result = x != y
        expected = DataFrame(
            {col: x[col] != y[col] for col in x.columns},
            index=x.index,
            columns=x.columns,
        )
        tm.assert_frame_equal(result, expected)

        msgs = [
            r"Invalid comparison between dtype=datetime64\[ns\] and ndarray",
            "invalid type promotion",
            (
                # npdev 1.20.0
                r"The DTypes <class 'numpy.dtype\[.*\]'> and "
                r"<class 'numpy.dtype\[.*\]'> do not have a common DType."
            ),
        ]
        msg = "|".join(msgs)
        with pytest.raises(TypeError, match=msg):
            x >= y
        with pytest.raises(TypeError, match=msg):
            x > y
        with pytest.raises(TypeError, match=msg):
            x < y
        with pytest.raises(TypeError, match=msg):
            x <= y

    @pytest.mark.parametrize(
        "left, right",
        [
            ("gt", "lt"),
            ("lt", "gt"),
            ("ge", "le"),
            ("le", "ge"),
            ("eq", "eq"),
            ("ne", "ne"),
        ],
    )
    def test_timestamp_compare(self, left, right):
        # make sure we can compare Timestamps on the right AND left hand side
        # GH#4982
        df = DataFrame(
            {
                "dates1": pd.date_range("20010101", periods=10),
                "dates2": pd.date_range("20010102", periods=10),
                "intcol": np.random.default_rng(2).integers(1000000000, size=10),
                "floatcol": np.random.default_rng(2).standard_normal(10),
                "stringcol": [chr(100 + i) for i in range(10)],
            }
        )
        df.loc[np.random.default_rng(2).random(len(df)) > 0.5, "dates2"] = pd.NaT
        left_f = getattr(operator, left)
        right_f = getattr(operator, right)

        # no nats
        if left in ["eq", "ne"]:
            expected = left_f(df, pd.Timestamp("20010109"))
            result = right_f(pd.Timestamp("20010109"), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = (
                "'(<|>)=?' not supported between "
                "instances of 'numpy.ndarray' and 'Timestamp'"
            )
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp("20010109"))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp("20010109"), df)
        # nats
        if left in ["eq", "ne"]:
            expected = left_f(df, pd.Timestamp("nat"))
            result = right_f(pd.Timestamp("nat"), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = (
                "'(<|>)=?' not supported between "
                "instances of 'numpy.ndarray' and 'NaTType'"
            )
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp("nat"))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp("nat"), df)

    def test_mixed_comparison(self):
        # GH#13128, GH#22163 != datetime64 vs non-dt64 should be False,
        # not raise TypeError
        # (this appears to be fixed before GH#22163, not sure when)
        df = DataFrame([["1989-08-01", 1], ["1989-08-01", 2]])
        other = DataFrame([["a", "b"], ["c", "d"]])

        result = df == other
        assert not result.any().any()

        result = df != other
        assert result.all().all()

    def test_df_boolean_comparison_error(self):
        # GH#4576, GH#22880
        # comparing DataFrame against list/tuple with len(obj) matching
        #  len(df.columns) is supported as of GH#22800
        df = DataFrame(np.arange(6).reshape((3, 2)))

        expected = DataFrame([[False, False], [True, False], [False, False]])

        result = df == (2, 2)
        tm.assert_frame_equal(result, expected)

        result = df == [2, 2]
        tm.assert_frame_equal(result, expected)

    def test_df_float_none_comparison(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            index=range(8),
            columns=["A", "B", "C"],
        )

        result = df.__eq__(None)
        assert not result.any().any()

    def test_df_string_comparison(self):
        df = DataFrame([{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}])
        mask_a = df.a > 1
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])

        mask_b = df.b == "foo"
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])


class TestFrameFlexComparisons:
    # TODO: test_bool_flex_frame needs a better name
    @pytest.mark.parametrize("op", ["eq", "ne", "gt", "lt", "ge", "le"])
    def test_bool_flex_frame(self, op):
        data = np.random.default_rng(2).standard_normal((5, 3))
        other_data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        other = DataFrame(other_data)
        ndim_5 = np.ones(df.shape + (1, 3))

        # DataFrame
        assert df.eq(df).values.all()
        assert not df.ne(df).values.any()
        f = getattr(df, op)
        o = getattr(operator, op)
        # No NAs
        tm.assert_frame_equal(f(other), o(df, other))
        # Unaligned
        part_o = other.loc[3:, 1:].copy()
        rs = f(part_o)
        xp = o(df, part_o.reindex(index=df.index, columns=df.columns))
        tm.assert_frame_equal(rs, xp)
        # ndarray
        tm.assert_frame_equal(f(other.values), o(df, other.values))
        # scalar
        tm.assert_frame_equal(f(0), o(df, 0))
        # NAs
        msg = "Unable to coerce to Series/DataFrame"
        tm.assert_frame_equal(f(np.nan), o(df, np.nan))
        with pytest.raises(ValueError, match=msg):
            f(ndim_5)

    @pytest.mark.parametrize("box", [np.array, Series])
    def test_bool_flex_series(self, box):
        # Series
        # list/tuple
        data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        idx_ser = box(np.random.default_rng(2).standard_normal(5))
        col_ser = box(np.random.default_rng(2).standard_normal(3))

        idx_eq = df.eq(idx_ser, axis=0)
        col_eq = df.eq(col_ser)
        idx_ne = df.ne(idx_ser, axis=0)
        col_ne = df.ne(col_ser)
        tm.assert_frame_equal(col_eq, df == Series(col_ser))
        tm.assert_frame_equal(col_eq, -col_ne)
        tm.assert_frame_equal(idx_eq, -idx_ne)
        tm.assert_frame_equal(idx_eq, df.T.eq(idx_ser).T)
        tm.assert_frame_equal(col_eq, df.eq(list(col_ser)))
        tm.assert_frame_equal(idx_eq, df.eq(Series(idx_ser), axis=0))
        tm.assert_frame_equal(idx_eq, df.eq(list(idx_ser), axis=0))

        idx_gt = df.gt(idx_ser, axis=0)
        col_gt = df.gt(col_ser)
        idx_le = df.le(idx_ser, axis=0)
        col_le = df.le(col_ser)

        tm.assert_frame_equal(col_gt, df > Series(col_ser))
        tm.assert_frame_equal(col_gt, -col_le)
        tm.assert_frame_equal(idx_gt, -idx_le)
        tm.assert_frame_equal(idx_gt, df.T.gt(idx_ser).T)

        idx_ge = df.ge(idx_ser, axis=0)
        col_ge = df.ge(col_ser)
        idx_lt = df.lt(idx_ser, axis=0)
        col_lt = df.lt(col_ser)
        tm.assert_frame_equal(col_ge, df >= Series(col_ser))
        tm.assert_frame_equal(col_ge, -col_lt)
        tm.assert_frame_equal(idx_ge, -idx_lt)
        tm.assert_frame_equal(idx_ge, df.T.ge(idx_ser).T)

        idx_ser = Series(np.random.default_rng(2).standard_normal(5))
        col_ser = Series(np.random.default_rng(2).standard_normal(3))

    def test_bool_flex_frame_na(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # NA
        df.loc[0, 0] = np.nan
        rs = df.eq(df)
        assert not rs.loc[0, 0]
        rs = df.ne(df)
        assert rs.loc[0, 0]
        rs = df.gt(df)
        assert not rs.loc[0, 0]
        rs = df.lt(df)
        assert not rs.loc[0, 0]
        rs = df.ge(df)
        assert not rs.loc[0, 0]
        rs = df.le(df)
        assert not rs.loc[0, 0]

    def test_bool_flex_frame_complex_dtype(self):
        # complex
        arr = np.array([np.nan, 1, 6, np.nan])
        arr2 = np.array([2j, np.nan, 7, None])
        df = DataFrame({"a": arr})
        df2 = DataFrame({"a": arr2})

        msg = "|".join(
            [
                "'>' not supported between instances of '.*' and 'complex'",
                r"unorderable types: .*complex\(\)",  # PY35
            ]
        )
        with pytest.raises(TypeError, match=msg):
            # inequalities are not well-defined for complex numbers
            df.gt(df2)
        with pytest.raises(TypeError, match=msg):
            # regression test that we get the same behavior for Series
            df["a"].gt(df2["a"])
        with pytest.raises(TypeError, match=msg):
            # Check that we match numpy behavior here
            df.values > df2.values

        rs = df.ne(df2)
        assert rs.values.all()

        arr3 = np.array([2j, np.nan, None])
        df3 = DataFrame({"a": arr3})

        with pytest.raises(TypeError, match=msg):
            # inequalities are not well-defined for complex numbers
            df3.gt(2j)
        with pytest.raises(TypeError, match=msg):
            # regression test that we get the same behavior for Series
            df3["a"].gt(2j)
        with pytest.raises(TypeError, match=msg):
            # Check that we match numpy behavior here
            df3.values > 2j

    def test_bool_flex_frame_object_dtype(self):
        # corner, dtype=object
        df1 = DataFrame({"col": ["foo", np.nan, "bar"]}, dtype=object)
        df2 = DataFrame({"col": ["foo", datetime.now(), "bar"]}, dtype=object)
        result = df1.ne(df2)
        exp = DataFrame({"col": [False, True, False]})
        tm.assert_frame_equal(result, exp)

    def test_flex_comparison_nat(self):
        # GH 15697, GH 22163 df.eq(pd.NaT) should behave like df == pd.NaT,
        # and _definitely_ not be NaN
        df = DataFrame([pd.NaT])

        result = df == pd.NaT
        # result.iloc[0, 0] is a np.bool_ object
        assert result.iloc[0, 0].item() is False

        result = df.eq(pd.NaT)
        assert result.iloc[0, 0].item() is False

        result = df != pd.NaT
        assert result.iloc[0, 0].item() is True

        result = df.ne(pd.NaT)
        assert result.iloc[0, 0].item() is True

    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    def test_df_flex_cmp_constant_return_types(self, opname):
        # GH 15077, non-empty DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
        const = 2

        result = getattr(df, opname)(const).dtypes.value_counts()
        tm.assert_series_equal(
            result, Series([2], index=[np.dtype(bool)], name="count")
        )

    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    def test_df_flex_cmp_constant_return_types_empty(self, opname):
        # GH 15077 empty DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
        const = 2

        empty = df.iloc[:0]
        result = getattr(empty, opname)(const).dtypes.value_counts()
        tm.assert_series_equal(
            result, Series([2], index=[np.dtype(bool)], name="count")
        )

    def test_df_flex_cmp_ea_dtype_with_ndarray_series(self):
        ii = pd.IntervalIndex.from_breaks([1, 2, 3])
        df = DataFrame({"A": ii, "B": ii})

        ser = Series([0, 0])
        res = df.eq(ser, axis=0)

        expected = DataFrame({"A": [False, False], "B": [False, False]})
        tm.assert_frame_equal(res, expected)

        ser2 = Series([1, 2], index=["A", "B"])
        res2 = df.eq(ser2, axis=1)
        tm.assert_frame_equal(res2, expected)


# -------------------------------------------------------------------
# Arithmetic


class TestFrameFlexArithmetic:
    def test_floordiv_axis0(self):
        # make sure we df.floordiv(ser, axis=0) matches column-wise result
        arr = np.arange(3)
        ser = Series(arr)
        df = DataFrame({"A": ser, "B": ser})

        result = df.floordiv(ser, axis=0)

        expected = DataFrame({col: df[col] // ser for col in df.columns})

        tm.assert_frame_equal(result, expected)

        result2 = df.floordiv(ser.values, axis=0)
        tm.assert_frame_equal(result2, expected)

    def test_df_add_td64_columnwise(self):
        # GH 22534 Check that column-wise addition broadcasts correctly
        dti = pd.date_range("2016-01-01", periods=10)
        tdi = pd.timedelta_range("1", periods=10)
        tser = Series(tdi)
        df = DataFrame({0: dti, 1: tdi})

        result = df.add(tser, axis=0)
        expected = DataFrame({0: dti + tdi, 1: tdi + tdi})
        tm.assert_frame_equal(result, expected)

    def test_df_add_flex_filled_mixed_dtypes(self):
        # GH 19611
        dti = pd.date_range("2016-01-01", periods=3)
        ser = Series(["1 Day", "NaT", "2 Days"], dtype="timedelta64[ns]")
        df = DataFrame({"A": dti, "B": ser})
        other = DataFrame({"A": ser, "B": ser})
        fill = pd.Timedelta(days=1).to_timedelta64()
        result = df.add(other, fill_value=fill)

        expected = DataFrame(
            {
                "A": Series(
                    ["2016-01-02", "2016-01-03", "2016-01-05"], dtype="datetime64[ns]"
                ),
                "B": ser * 2,
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_frame(
        self, all_arithmetic_operators, float_frame, mixed_float_frame
    ):
        # one instance of parametrized fixture
        op = all_arithmetic_operators

        def f(x, y):
            # r-versions not in operator-stdlib; get op without "r" and invert
            if op.startswith("__r"):
                return getattr(operator, op.replace("__r", "__"))(y, x)
            return getattr(operator, op)(x, y)

        result = getattr(float_frame, op)(2 * float_frame)
        expected = f(float_frame, 2 * float_frame)
        tm.assert_frame_equal(result, expected)

        # vs mix float
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={"C": None})

    @pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__"])
    def test_arith_flex_frame_mixed(
        self,
        op,
        int_frame,
        mixed_int_frame,
        mixed_float_frame,
        switch_numexpr_min_elements,
    ):
        f = getattr(operator, op)

        # vs mix int
        result = getattr(mixed_int_frame, op)(2 + mixed_int_frame)
        expected = f(mixed_int_frame, 2 + mixed_int_frame)

        # no overflow in the uint
        dtype = None
        if op in ["__sub__"]:
            dtype = {"B": "uint64", "C": None}
        elif op in ["__add__", "__mul__"]:
            dtype = {"C": None}
        if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
            # when using numexpr, the casting rules are slightly different:
            # in the `2 + mixed_int_frame` operation, int32 column becomes
            # and int64 column (not preserving dtype in operation with Python
            # scalar), and then the int32/int64 combo results in int64 result
            dtype["A"] = (2 + mixed_int_frame)["A"].dtype
        tm.assert_frame_equal(result, expected)
        _check_mixed_int(result, dtype=dtype)

        # vs mix float
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={"C": None})

        # vs plain int
        result = getattr(int_frame, op)(2 * int_frame)
        expected = f(int_frame, 2 * int_frame)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dim", range(3, 6))
    def test_arith_flex_frame_raise(self, all_arithmetic_operators, float_frame, dim):
        # one instance of parametrized fixture
        op = all_arithmetic_operators

        # Check that arrays with dim >= 3 raise
        arr = np.ones((1,) * dim)
        msg = "Unable to coerce to Series/DataFrame"
        with pytest.raises(ValueError, match=msg):
            getattr(float_frame, op)(arr)

    def test_arith_flex_frame_corner(self, float_frame):
        const_add = float_frame.add(1)
        tm.assert_frame_equal(const_add, float_frame + 1)

        # corner cases
        result = float_frame.add(float_frame[:0])
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)

        result = float_frame[:0].add(float_frame)
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)

        with pytest.raises(NotImplementedError, match="fill_value"):
            float_frame.add(float_frame.iloc[0], fill_value=3)

        with pytest.raises(NotImplementedError, match="fill_value"):
            float_frame.add(float_frame.iloc[0], axis="index", fill_value=3)

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "mod"])
    def test_arith_flex_series_ops(self, simple_frame, op):
        # after arithmetic refactor, add truediv here
        df = simple_frame

        row = df.xs("a")
        col = df["two"]
        f = getattr(df, op)
        op = getattr(operator, op)
        tm.assert_frame_equal(f(row), op(df, row))
        tm.assert_frame_equal(f(col, axis=0), op(df.T, col).T)

    def test_arith_flex_series(self, simple_frame):
        df = simple_frame

        row = df.xs("a")
        col = df["two"]
        # special case for some reason
        tm.assert_frame_equal(df.add(row, axis=None), df + row)

        # cases which will be refactored after big arithmetic refactor
        tm.assert_frame_equal(df.div(row), df / row)
        tm.assert_frame_equal(df.div(col, axis=0), (df.T / col).T)

    @pytest.mark.parametrize("dtype", ["int64", "float64"])
    def test_arith_flex_series_broadcasting(self, dtype):
        # broadcasting issue in GH 7325
        df = DataFrame(np.arange(3 * 2).reshape((3, 2)), dtype=dtype)
        expected = DataFrame([[np.nan, np.inf], [1.0, 1.5], [1.0, 1.25]])
        result = df.div(df[0], axis="index")
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_zero_len_raises(self):
        # GH 19522 passing fill_value to frame flex arith methods should
        # raise even in the zero-length special cases
        ser_len0 = Series([], dtype=object)
        df_len0 = DataFrame(columns=["A", "B"])
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])

        with pytest.raises(NotImplementedError, match="fill_value"):
            df.add(ser_len0, fill_value="E")

        with pytest.raises(NotImplementedError, match="fill_value"):
            df_len0.sub(df["A"], axis=None, fill_value=3)

    def test_flex_add_scalar_fill_value(self):
        # GH#12723
        dat = np.array([0, 1, np.nan, 3, 4, 5], dtype="float")
        df = DataFrame({"foo": dat}, index=range(6))

        exp = df.fillna(0).add(2)
        res = df.add(2, fill_value=0)
        tm.assert_frame_equal(res, exp)

    def test_sub_alignment_with_duplicate_index(self):
        # GH#5185 dup aligning operations should work
        df1 = DataFrame([1, 2, 3, 4, 5], index=[1, 2, 1, 2, 3])
        df2 = DataFrame([1, 2, 3], index=[1, 2, 3])
        expected = DataFrame([0, 2, 0, 2, 2], index=[1, 1, 2, 2, 3])
        result = df1.sub(df2)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op", ["__add__", "__mul__", "__sub__", "__truediv__"])
    def test_arithmetic_with_duplicate_columns(self, op):
        # operations
        df = DataFrame({"A": np.arange(10), "B": np.random.default_rng(2).random(10)})
        expected = getattr(df, op)(df)
        expected.columns = ["A", "A"]
        df.columns = ["A", "A"]
        result = getattr(df, op)(df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("level", [0, None])
    def test_broadcast_multiindex(self, level):
        # GH34388
        df1 = DataFrame({"A": [0, 1, 2], "B": [1, 2, 3]})
        df1.columns = df1.columns.set_names("L1")

        df2 = DataFrame({("A", "C"): [0, 0, 0], ("A", "D"): [0, 0, 0]})
        df2.columns = df2.columns.set_names(["L1", "L2"])

        result = df1.add(df2, level=level)
        expected = DataFrame({("A", "C"): [0, 1, 2], ("A", "D"): [0, 1, 2]})
        expected.columns = expected.columns.set_names(["L1", "L2"])

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations(self):
        # GH 43321
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        series = Series(
            [0.4],
            index=MultiIndex.from_product([["b"], ["a"]], names=["mod", "scen"]),
        )

        expected = DataFrame(
            {2010: [1.4, 2.4, 3.4], 2020: [3.4, 4.4, 5.4]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_series_index_to_frame_index(self):
        # GH 43321
        df = DataFrame(
            {2010: [1], 2020: [3]},
            index=MultiIndex.from_product([["a"], ["b"]], names=["scen", "mod"]),
        )

        series = Series(
            [10.0, 20.0, 30.0],
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        expected = DataFrame(
            {2010: [11.0, 21, 31.0], 2020: [13.0, 23.0, 33.0]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_no_align(self):
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        series = Series(
            [0.4],
            index=MultiIndex.from_product([["c"], ["a"]], names=["mod", "scen"]),
        )

        expected = DataFrame(
            {2010: np.nan, 2020: np.nan},
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "b", 2),
                    ("a", "c", np.nan),
                ],
                names=["scen", "mod", "id"],
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_part_align(self):
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                names=["scen", "mod", "id"],
            ),
        )

        series = Series(
            [0.4],
            index=MultiIndex.from_product([["b"], ["a"]], names=["mod", "scen"]),
        )

        expected = DataFrame(
            {2010: [1.4, 2.4, np.nan], 2020: [3.4, 4.4, np.nan]},
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                names=["scen", "mod", "id"],
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)


class TestFrameArithmetic:
    def test_td64_op_nat_casting(self):
        # Make sure we don't accidentally treat timedelta64(NaT) as datetime64
        #  when calling dispatch_to_series in DataFrame arithmetic
        ser = Series(["NaT", "NaT"], dtype="timedelta64[ns]")
        df = DataFrame([[1, 2], [3, 4]])

        result = df * ser
        expected = DataFrame({0: ser, 1: ser})
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_rowlike_broadcasts(self):
        # GH#23000
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        rowlike = arr[[1], :]  # shape --> (1, ncols)
        assert rowlike.shape == (1, df.shape[1])

        expected = DataFrame(
            [[2, 4], [4, 6], [6, 8]],
            columns=df.columns,
            index=df.index,
            # specify dtype explicitly to avoid failing
            # on 32bit builds
            dtype=arr.dtype,
        )
        result = df + rowlike
        tm.assert_frame_equal(result, expected)
        result = rowlike + df
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_collike_broadcasts(self):
        # GH#23000
        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        collike = arr[:, [1]]  # shape --> (nrows, 1)
        assert collike.shape == (df.shape[0], 1)

        expected = DataFrame(
            [[1, 2], [5, 6], [9, 10]],
            columns=df.columns,
            index=df.index,
            # specify dtype explicitly to avoid failing
            # on 32bit builds
            dtype=arr.dtype,
        )
        result = df + collike
        tm.assert_frame_equal(result, expected)
        result = collike + df
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_rowlike_broadcasts(
        self, request, all_arithmetic_operators, using_array_manager
    ):
        # GH#23000
        opname = all_arithmetic_operators

        if using_array_manager and opname in ("__rmod__", "__rfloordiv__"):
            # TODO(ArrayManager) decide on dtypes
            td.mark_array_manager_not_yet_implemented(request)

        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        rowlike = arr[[1], :]  # shape --> (1, ncols)
        assert rowlike.shape == (1, df.shape[1])

        exvals = [
            getattr(df.loc["A"], opname)(rowlike.squeeze()),
            getattr(df.loc["B"], opname)(rowlike.squeeze()),
            getattr(df.loc["C"], opname)(rowlike.squeeze()),
        ]

        expected = DataFrame(exvals, columns=df.columns, index=df.index)

        result = getattr(df, opname)(rowlike)
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_collike_broadcasts(
        self, request, all_arithmetic_operators, using_array_manager
    ):
        # GH#23000
        opname = all_arithmetic_operators

        if using_array_manager and opname in ("__rmod__", "__rfloordiv__"):
            # TODO(ArrayManager) decide on dtypes
            td.mark_array_manager_not_yet_implemented(request)

        arr = np.arange(6).reshape(3, 2)
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        collike = arr[:, [1]]  # shape --> (nrows, 1)
        assert collike.shape == (df.shape[0], 1)

        exvals = {
            True: getattr(df[True], opname)(collike.squeeze()),
            False: getattr(df[False], opname)(collike.squeeze()),
        }

        dtype = None
        if opname in ["__rmod__", "__rfloordiv__"]:
            # Series ops may return mixed int/float dtypes in cases where
            #   DataFrame op will return all-float.  So we upcast `expected`
            dtype = np.common_type(*(x.values for x in exvals.values()))

        expected = DataFrame(exvals, columns=df.columns, index=df.index, dtype=dtype)

        result = getattr(df, opname)(collike)
        tm.assert_frame_equal(result, expected)

    def test_df_bool_mul_int(self):
        # GH 22047, GH 22163 multiplication by 1 should result in int dtype,
        # not object dtype
        df = DataFrame([[False, True], [False, False]])
        result = df * 1

        # On appveyor this comes back as np.int32 instead of np.int64,
        # so we check dtype.kind instead of just dtype
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == "i").all()

        result = 1 * df
        kinds = result.dtypes.apply(lambda x: x.kind)
        assert (kinds == "i").all()

    def test_arith_mixed(self):
        left = DataFrame({"A": ["a", "b", "c"], "B": [1, 2, 3]})

        result = left + left
        expected = DataFrame({"A": ["aa", "bb", "cc"], "B": [2, 4, 6]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("col", ["A", "B"])
    def test_arith_getitem_commute(self, all_arithmetic_functions, col):
        df = DataFrame({"A": [1.1, 3.3], "B": [2.5, -3.9]})
        result = all_arithmetic_functions(df, 1)[col]
        expected = all_arithmetic_functions(df[col], 1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values", [[1, 2], (1, 2), np.array([1, 2]), range(1, 3), deque([1, 2])]
    )
    def test_arith_alignment_non_pandas_object(self, values):
        # GH#17901
        df = DataFrame({"A": [1, 1], "B": [1, 1]})
        expected = DataFrame({"A": [2, 2], "B": [3, 3]})
        result = df + values
        tm.assert_frame_equal(result, expected)

    def test_arith_non_pandas_object(self):
        df = DataFrame(
            np.arange(1, 10, dtype="f8").reshape(3, 3),
            columns=["one", "two", "three"],
            index=["a", "b", "c"],
        )

        val1 = df.xs("a").values
        added = DataFrame(df.values + val1, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val1, added)

        added = DataFrame((df.values.T + val1).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val1, axis=0), added)

        val2 = list(df["two"])

        added = DataFrame(df.values + val2, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df + val2, added)

        added = DataFrame((df.values.T + val2).T, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val2, axis="index"), added)

        val3 = np.random.default_rng(2).random(df.shape)
        added = DataFrame(df.values + val3, index=df.index, columns=df.columns)
        tm.assert_frame_equal(df.add(val3), added)

    def test_operations_with_interval_categories_index(self, all_arithmetic_operators):
        # GH#27415
        op = all_arithmetic_operators
        ind = pd.CategoricalIndex(pd.interval_range(start=0.0, end=2.0))
        data = [1, 2]
        df = DataFrame([data], columns=ind)
        num = 10
        result = getattr(df, op)(num)
        expected = DataFrame([[getattr(n, op)(num) for n in data]], columns=ind)
        tm.assert_frame_equal(result, expected)

    def test_frame_with_frame_reindex(self):
        # GH#31623
        df = DataFrame(
            {
                "foo": [pd.Timestamp("2019"), pd.Timestamp("2020")],
                "bar": [pd.Timestamp("2018"), pd.Timestamp("2021")],
            },
            columns=["foo", "bar"],
            dtype="M8[ns]",
        )
        df2 = df[["foo"]]

        result = df - df2

        expected = DataFrame(
            {"foo": [pd.Timedelta(0), pd.Timedelta(0)], "bar": [np.nan, np.nan]},
            columns=["bar", "foo"],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "value, dtype",
        [
            (1, "i8"),
            (1.0, "f8"),
            (2**63, "f8"),
            (1j, "complex128"),
            (2**63, "complex128"),
            (True, "bool"),
            (np.timedelta64(20, "ns"), "<m8[ns]"),
            (np.datetime64(20, "ns"), "<M8[ns]"),
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.mod,
            operator.pow,
        ],
        ids=lambda x: x.__name__,
    )
    def test_binop_other(self, op, value, dtype, switch_numexpr_min_elements):
        skip = {
            (operator.truediv, "bool"),
            (operator.pow, "bool"),
            (operator.add, "bool"),
            (operator.mul, "bool"),
        }

        elem = DummyElement(value, dtype)
        df = DataFrame({"A": [elem.value, elem.value]}, dtype=elem.dtype)

        invalid = {
            (operator.pow, "<M8[ns]"),
            (operator.mod, "<M8[ns]"),
            (operator.truediv, "<M8[ns]"),
            (operator.mul, "<M8[ns]"),
            (operator.add, "<M8[ns]"),
            (operator.pow, "<m8[ns]"),
            (operator.mul, "<m8[ns]"),
            (operator.sub, "bool"),
            (operator.mod, "complex128"),
        }

        if (op, dtype) in invalid:
            warn = None
            if (dtype == "<M8[ns]" and op == operator.add) or (
                dtype == "<m8[ns]" and op == operator.mul
            ):
                msg = None
            elif dtype == "complex128":
                msg = "ufunc 'remainder' not supported for the input types"
            elif op is operator.sub:
                msg = "numpy boolean subtract, the `-` operator, is "
                if (
                    dtype == "bool"
                    and expr.USE_NUMEXPR
                    and switch_numexpr_min_elements == 0
                ):
                    warn = UserWarning  # "evaluating in Python space because ..."
            else:
                msg = (
                    f"cannot perform __{op.__name__}__ with this "
                    "index type: (DatetimeArray|TimedeltaArray)"
                )

            with pytest.raises(TypeError, match=msg):
                with tm.assert_produces_warning(warn):
                    op(df, elem.value)

        elif (op, dtype) in skip:
            if op in [operator.add, operator.mul]:
                if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                    # "evaluating in Python space because ..."
                    warn = UserWarning
                else:
                    warn = None
                with tm.assert_produces_warning(warn):
                    op(df, elem.value)

            else:
                msg = "operator '.*' not implemented for .* dtypes"
                with pytest.raises(NotImplementedError, match=msg):
                    op(df, elem.value)

        else:
            with tm.assert_produces_warning(None):
                result = op(df, elem.value).dtypes
                expected = op(df, value).dtypes
            tm.assert_series_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes(self):
        # GH#49769
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2 = MultiIndex.from_arrays([Series([1, 2], dtype="Int8"), Series([3, 4])])
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result = left - right
        expected = DataFrame([[0, 0], [0, 0]], columns=midx)
        tm.assert_frame_equal(result, expected)

    def test_arithmetic_midx_cols_different_dtypes_different_order(self):
        # GH#49769
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        midx2 = MultiIndex.from_arrays([Series([2, 1], dtype="Int8"), Series([4, 3])])
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        result = left - right
        expected = DataFrame([[-1, 1], [-1, 1]], columns=midx)
        tm.assert_frame_equal(result, expected)


def test_frame_with_zero_len_series_corner_cases():
    # GH#28600
    # easy all-float case
    df = DataFrame(
        np.random.default_rng(2).standard_normal(6).reshape(3, 2), columns=["A", "B"]
    )
    ser = Series(dtype=np.float64)

    result = df + ser
    expected = DataFrame(df.values * np.nan, columns=df.columns)
    tm.assert_frame_equal(result, expected)

    with pytest.raises(ValueError, match="not aligned"):
        # Automatic alignment for comparisons deprecated GH#36795, enforced 2.0
        df == ser

    # non-float case should not raise TypeError on comparison
    df2 = DataFrame(df.values.view("M8[ns]"), columns=df.columns)
    with pytest.raises(ValueError, match="not aligned"):
        # Automatic alignment for comparisons deprecated
        df2 == ser


def test_zero_len_frame_with_series_corner_cases():
    # GH#28600
    df = DataFrame(columns=["A", "B"], dtype=np.float64)
    ser = Series([1, 2], index=["A", "B"])

    result = df + ser
    expected = df
    tm.assert_frame_equal(result, expected)


def test_frame_single_columns_object_sum_axis_1():
    # GH 13758
    data = {
        "One": Series(["A", 1.2, np.nan]),
    }
    df = DataFrame(data)
    result = df.sum(axis=1)
    expected = Series(["A", 1.2, 0])
    tm.assert_series_equal(result, expected)


# -------------------------------------------------------------------
# Unsorted
#  These arithmetic tests were previously in other files, eventually
#  should be parametrized and put into tests.arithmetic


class TestFrameArithmeticUnsorted:
    def test_frame_add_tz_mismatch_converts_to_utc(self):
        rng = pd.date_range("1/1/2011", periods=10, freq="h", tz="US/Eastern")
        df = DataFrame(
            np.random.default_rng(2).standard_normal(len(rng)), index=rng, columns=["a"]
        )

        df_moscow = df.tz_convert("Europe/Moscow")
        result = df + df_moscow
        assert result.index.tz is timezone.utc

        result = df_moscow + df
        assert result.index.tz is timezone.utc

    def test_align_frame(self):
        rng = pd.period_range("1/1/2000", "1/1/2010", freq="Y")
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )

        result = ts + ts[::2]
        expected = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_frame_equal(result, expected)

        half = ts[::2]
        result = ts + half.take(np.random.default_rng(2).permutation(len(half)))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "op", [operator.add, operator.sub, operator.mul, operator.truediv]
    )
    def test_operators_none_as_na(self, op):
        df = DataFrame(
            {"col1": [2, 5.0, 123, None], "col2": [1, 2, 3, 4]}, dtype=object
        )

        # since filling converts dtypes from object, changed expected to be
        # object
        msg = "Downcasting object dtype arrays"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            filled = df.fillna(np.nan)
        result = op(df, 3)
        expected = op(filled, 3).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)

        result = op(df, df)
        expected = op(filled, filled).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)

        msg = "Downcasting object dtype arrays"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = op(df, df.fillna(7))
        tm.assert_frame_equal(result, expected)

        msg = "Downcasting object dtype arrays"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = op(df.fillna(7), df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op,res", [("__eq__", False), ("__ne__", True)])
    # TODO: not sure what's correct here.
    @pytest.mark.filterwarnings("ignore:elementwise:FutureWarning")
    def test_logical_typeerror_with_non_valid(self, op, res, float_frame):
        # we are comparing floats vs a string
        result = getattr(float_frame, op)("foo")
        assert bool(result.all().all()) is res

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "truediv"])
    def test_binary_ops_align(self, op):
        # test aligning binary ops

        # GH 6681
        index = MultiIndex.from_product(
            [list("abc"), ["one", "two", "three"], [1, 2, 3]],
            names=["first", "second", "third"],
        )

        df = DataFrame(
            np.arange(27 * 3).reshape(27, 3),
            index=index,
            columns=["value1", "value2", "value3"],
        ).sort_index()

        idx = pd.IndexSlice
        opa = getattr(operator, op, None)
        if opa is None:
            return

        x = Series([1.0, 10.0, 100.0], [1, 2, 3])
        result = getattr(df, op)(x, level="third", axis=0)

        expected = pd.concat(
            [opa(df.loc[idx[:, :, i], :], v) for i, v in x.items()]
        ).sort_index()
        tm.assert_frame_equal(result, expected)

        x = Series([1.0, 10.0], ["two", "three"])
        result = getattr(df, op)(x, level="second", axis=0)

        expected = (
            pd.concat([opa(df.loc[idx[:, i], :], v) for i, v in x.items()])
            .reindex_like(df)
            .sort_index()
        )
        tm.assert_frame_equal(result, expected)

    def test_binary_ops_align_series_dataframe(self):
        # GH9463 (alignment level of dataframe with series)

        midx = MultiIndex.from_product([["A", "B"], ["a", "b"]])
        df = DataFrame(np.ones((2, 4), dtype="int64"), columns=midx)
        s = Series({"a": 1, "b": 2})

        df2 = df.copy()
        df2.columns.names = ["lvl0", "lvl1"]
        s2 = s.copy()
        s2.index.name = "lvl1"

        # different cases of integer/string level names:
        res1 = df.mul(s, axis=1, level=1)
        res2 = df.mul(s2, axis=1, level=1)
        res3 = df2.mul(s, axis=1, level=1)
        res4 = df2.mul(s2, axis=1, level=1)
        res5 = df2.mul(s, axis=1, level="lvl1")
        res6 = df2.mul(s2, axis=1, level="lvl1")

        exp = DataFrame(
            np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype="int64"), columns=midx
        )

        for res in [res1, res2]:
            tm.assert_frame_equal(res, exp)

        exp.columns.names = ["lvl0", "lvl1"]
        for res in [res3, res4, res5, res6]:
            tm.assert_frame_equal(res, exp)

    def test_add_with_dti_mismatched_tzs(self):
        base = pd.DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz="UTC")
        idx1 = base.tz_convert("Asia/Tokyo")[:2]
        idx2 = base.tz_convert("US/Eastern")[1:]

        df1 = DataFrame({"A": [1, 2]}, index=idx1)
        df2 = DataFrame({"A": [1, 1]}, index=idx2)
        exp = DataFrame({"A": [np.nan, 3, np.nan]}, index=base)
        tm.assert_frame_equal(df1 + df2, exp)

    def test_combineFrame(self, float_frame, mixed_float_frame, mixed_int_frame):
        frame_copy = float_frame.reindex(float_frame.index[::2])

        del frame_copy["D"]
        # adding NAs to first 5 values of column "C"
        frame_copy.loc[: frame_copy.index[4], "C"] = np.nan

        added = float_frame + frame_copy

        indexer = added["A"].dropna().index
        exp = (float_frame["A"] * 2).copy()

        tm.assert_series_equal(added["A"].dropna(), exp.loc[indexer])

        exp.loc[~exp.index.isin(indexer)] = np.nan
        tm.assert_series_equal(added["A"], exp.loc[added["A"].index])

        assert np.isnan(added["C"].reindex(frame_copy.index)[:5]).all()

        # assert(False)

        assert np.isnan(added["D"]).all()

        self_added = float_frame + float_frame
        tm.assert_index_equal(self_added.index, float_frame.index)

        added_rev = frame_copy + float_frame
        assert np.isnan(added["D"]).all()
        assert np.isnan(added_rev["D"]).all()

        # corner cases

        # empty
        plus_empty = float_frame + DataFrame()
        assert np.isnan(plus_empty.values).all()

        empty_plus = DataFrame() + float_frame
        assert np.isnan(empty_plus.values).all()

        empty_empty = DataFrame() + DataFrame()
        assert empty_empty.empty

        # out of order
        reverse = float_frame.reindex(columns=float_frame.columns[::-1])

        tm.assert_frame_equal(reverse + float_frame, float_frame * 2)

        # mix vs float64, upcast
        added = float_frame + mixed_float_frame
        _check_mixed_float(added, dtype="float64")
        added = mixed_float_frame + float_frame
        _check_mixed_float(added, dtype="float64")

        # mix vs mix
        added = mixed_float_frame + mixed_float_frame
        _check_mixed_float(added, dtype={"C": None})

        # with int
        added = float_frame + mixed_int_frame
        _check_mixed_float(added, dtype="float64")

    def test_combine_series(self, float_frame, mixed_float_frame, mixed_int_frame):
        # Series
        series = float_frame.xs(float_frame.index[0])

        added = float_frame + series

        for key, s in added.items():
            tm.assert_series_equal(s, float_frame[key] + series[key])

        larger_series = series.to_dict()
        larger_series["E"] = 1
        larger_series = Series(larger_series)
        larger_added = float_frame + larger_series

        for key, s in float_frame.items():
            tm.assert_series_equal(larger_added[key], s + series[key])
        assert "E" in larger_added
        assert np.isnan(larger_added["E"]).all()

        # no upcast needed
        added = mixed_float_frame + series
        assert np.all(added.dtypes == series.dtype)

        # vs mix (upcast) as needed
        added = mixed_float_frame + series.astype("float32")
        _check_mixed_float(added, dtype={"C": None})
        added = mixed_float_frame + series.astype("float16")
        _check_mixed_float(added, dtype={"C": None})

        # these used to raise with numexpr as we are adding an int64 to an
        #  uint64....weird vs int
        added = mixed_int_frame + (100 * series).astype("int64")
        _check_mixed_int(
            added, dtype={"A": "int64", "B": "float64", "C": "int64", "D": "int64"}
        )
        added = mixed_int_frame + (100 * series).astype("int32")
        _check_mixed_int(
            added, dtype={"A": "int32", "B": "float64", "C": "int32", "D": "int64"}
        )

    def test_combine_timeseries(self, datetime_frame):
        # TimeSeries
        ts = datetime_frame["A"]

        # 10890
        # we no longer allow auto timeseries broadcasting
        # and require explicit broadcasting
        added = datetime_frame.add(ts, axis="index")

        for key, col in datetime_frame.items():
            result = col + ts
            tm.assert_series_equal(added[key], result, check_names=False)
            assert added[key].name == key
            if col.name == ts.name:
                assert result.name == "A"
            else:
                assert result.name is None

        smaller_frame = datetime_frame[:-5]
        smaller_added = smaller_frame.add(ts, axis="index")

        tm.assert_index_equal(smaller_added.index, datetime_frame.index)

        smaller_ts = ts[:-5]
        smaller_added2 = datetime_frame.add(smaller_ts, axis="index")
        tm.assert_frame_equal(smaller_added, smaller_added2)

        # length 0, result is all-nan
        result = datetime_frame.add(ts[:0], axis="index")
        expected = DataFrame(
            np.nan, index=datetime_frame.index, columns=datetime_frame.columns
        )
        tm.assert_frame_equal(result, expected)

        # Frame is all-nan
        result = datetime_frame[:0].add(ts, axis="index")
        expected = DataFrame(
            np.nan, index=datetime_frame.index, columns=datetime_frame.columns
        )
        tm.assert_frame_equal(result, expected)

        # empty but with non-empty index
        frame = datetime_frame[:1].reindex(columns=[])
        result = frame.mul(ts, axis="index")
        assert len(result) == len(ts)

    def test_combineFunc(self, float_frame, mixed_float_frame):
        result = float_frame * 2
        tm.assert_numpy_array_equal(result.values, float_frame.values * 2)

        # vs mix
        result = mixed_float_frame * 2
        for c, s in result.items():
            tm.assert_numpy_array_equal(s.values, mixed_float_frame[c].values * 2)
        _check_mixed_float(result, dtype={"C": None})

        result = DataFrame() * 2
        assert result.index.equals(DataFrame().index)
        assert len(result.columns) == 0

    @pytest.mark.parametrize(
        "func",
        [operator.eq, operator.ne, operator.lt, operator.gt, operator.ge, operator.le],
    )
    def test_comparisons(self, simple_frame, float_frame, func):
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=pd.date_range("2000-01-01", periods=30, freq="B"),
        )
        df2 = df1.copy()

        row = simple_frame.xs("a")
        ndim_5 = np.ones(df1.shape + (1, 1, 1))

        result = func(df1, df2)
        tm.assert_numpy_array_equal(result.values, func(df1.values, df2.values))

        msg = (
            "Unable to coerce to Series/DataFrame, "
            "dimension must be <= 2: (30, 4, 1, 1, 1)"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            func(df1, ndim_5)

        result2 = func(simple_frame, row)
        tm.assert_numpy_array_equal(
            result2.values, func(simple_frame.values, row.values)
        )

        result3 = func(float_frame, 0)
        tm.assert_numpy_array_equal(result3.values, func(float_frame.values, 0))

        msg = (
            r"Can only compare identically-labeled \(both index and columns\) "
            "DataFrame objects"
        )
        with pytest.raises(ValueError, match=msg):
            func(simple_frame, simple_frame[:2])

    def test_strings_to_numbers_comparisons_raises(self, compare_operators_no_eq_ne):
        # GH 11565
        df = DataFrame(
            {x: {"x": "foo", "y": "bar", "z": "baz"} for x in ["a", "b", "c"]}
        )

        f = getattr(operator, compare_operators_no_eq_ne)
        msg = "|".join(
            [
                "'[<>]=?' not supported between instances of 'str' and 'int'",
                "Invalid comparison between dtype=str and int",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            f(df, 0)

    def test_comparison_protected_from_errstate(self):
        missing_df = DataFrame(
            np.ones((10, 4), dtype=np.float64),
            columns=Index(list("ABCD"), dtype=object),
        )
        missing_df.loc[missing_df.index[0], "A"] = np.nan
        with np.errstate(invalid="ignore"):
            expected = missing_df.values < 0
        with np.errstate(invalid="raise"):
            result = (missing_df < 0).values
        tm.assert_numpy_array_equal(result, expected)

    def test_boolean_comparison(self):
        # GH 4576
        # boolean comparisons with a tuple/list give unexpected results
        df = DataFrame(np.arange(6).reshape((3, 2)))
        b = np.array([2, 2])
        b_r = np.atleast_2d([2, 2])
        b_c = b_r.T
        lst = [2, 2, 2]
        tup = tuple(lst)

        # gt
        expected = DataFrame([[False, False], [False, True], [True, True]])
        result = df > b
        tm.assert_frame_equal(result, expected)

        result = df.values > b
        tm.assert_numpy_array_equal(result, expected.values)

        msg1d = "Unable to coerce to Series, length must be 2: given 3"
        msg2d = "Unable to coerce to DataFrame, shape must be"
        msg2db = "operands could not be broadcast together with shapes"
        with pytest.raises(ValueError, match=msg1d):
            # wrong shape
            df > lst

        with pytest.raises(ValueError, match=msg1d):
            # wrong shape
            df > tup

        # broadcasts like ndarray (GH#23000)
        result = df > b_r
        tm.assert_frame_equal(result, expected)

        result = df.values > b_r
        tm.assert_numpy_array_equal(result, expected.values)

        with pytest.raises(ValueError, match=msg2d):
            df > b_c

        with pytest.raises(ValueError, match=msg2db):
            df.values > b_c

        # ==
        expected = DataFrame([[False, False], [True, False], [False, False]])
        result = df == b
        tm.assert_frame_equal(result, expected)

        with pytest.raises(ValueError, match=msg1d):
            df == lst

        with pytest.raises(ValueError, match=msg1d):
            df == tup

        # broadcasts like ndarray (GH#23000)
        result = df == b_r
        tm.assert_frame_equal(result, expected)

        result = df.values == b_r
        tm.assert_numpy_array_equal(result, expected.values)

        with pytest.raises(ValueError, match=msg2d):
            df == b_c

        assert df.values.shape != b_c.shape

        # with alignment
        df = DataFrame(
            np.arange(6).reshape((3, 2)), columns=list("AB"), index=list("abc")
        )
        expected.index = df.index
        expected.columns = df.columns

        with pytest.raises(ValueError, match=msg1d):
            df == lst

        with pytest.raises(ValueError, match=msg1d):
            df == tup

    def test_inplace_ops_alignment(self):
        # inplace ops / ops alignment
        # GH 8511

        columns = list("abcdefg")
        X_orig = DataFrame(
            np.arange(10 * len(columns)).reshape(-1, len(columns)),
            columns=columns,
            index=range(10),
        )
        Z = 100 * X_orig.iloc[:, 1:-1].copy()
        block1 = list("bedcf")
        subs = list("bcdef")

        # add
        X = X_orig.copy()
        result1 = (X[block1] + Z).reindex(columns=subs)

        X[block1] += Z
        result2 = X.reindex(columns=subs)

        X = X_orig.copy()
        result3 = (X[block1] + Z[block1]).reindex(columns=subs)

        X[block1] += Z[block1]
        result4 = X.reindex(columns=subs)

        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        tm.assert_frame_equal(result1, result4)

        # sub
        X = X_orig.copy()
        result1 = (X[block1] - Z).reindex(columns=subs)

        X[block1] -= Z
        result2 = X.reindex(columns=subs)

        X = X_orig.copy()
        result3 = (X[block1] - Z[block1]).reindex(columns=subs)

        X[block1] -= Z[block1]
        result4 = X.reindex(columns=subs)

        tm.assert_frame_equal(result1, result2)
        tm.assert_frame_equal(result1, result3)
        tm.assert_frame_equal(result1, result4)

    def test_inplace_ops_identity(self):
        # GH 5104
        # make sure that we are actually changing the object
        s_orig = Series([1, 2, 3])
        df_orig = DataFrame(
            np.random.default_rng(2).integers(0, 5, size=10).reshape(-1, 5)
        )

        # no dtype change
        s = s_orig.copy()
        s2 = s
        s += 1
        tm.assert_series_equal(s, s2)
        tm.assert_series_equal(s_orig + 1, s)
        assert s is s2
        assert s._mgr is s2._mgr

        df = df_orig.copy()
        df2 = df
        df += 1
        tm.assert_frame_equal(df, df2)
        tm.assert_frame_equal(df_orig + 1, df)
        assert df is df2
        assert df._mgr is df2._mgr

        # dtype change
        s = s_orig.copy()
        s2 = s
        s += 1.5
        tm.assert_series_equal(s, s2)
        tm.assert_series_equal(s_orig + 1.5, s)

        df = df_orig.copy()
        df2 = df
        df += 1.5
        tm.assert_frame_equal(df, df2)
        tm.assert_frame_equal(df_orig + 1.5, df)
        assert df is df2
        assert df._mgr is df2._mgr

        # mixed dtype
        arr = np.random.default_rng(2).integers(0, 10, size=5)
        df_orig = DataFrame({"A": arr.copy(), "B": "foo"})
        df = df_orig.copy()
        df2 = df
        df["A"] += 1
        expected = DataFrame({"A": arr.copy() + 1, "B": "foo"})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr

        df = df_orig.copy()
        df2 = df
        df["A"] += 1.5
        expected = DataFrame({"A": arr.copy() + 1.5, "B": "foo"})
        tm.assert_frame_equal(df, expected)
        tm.assert_frame_equal(df2, expected)
        assert df._mgr is df2._mgr

    @pytest.mark.parametrize(
        "op",
        [
            "add",
            "and",
            pytest.param(
                "div",
                marks=pytest.mark.xfail(
                    raises=AttributeError, reason="__idiv__ not implemented"
                ),
            ),
            "floordiv",
            "mod",
            "mul",
            "or",
            "pow",
            "sub",
            "truediv",
            "xor",
        ],
    )
    def test_inplace_ops_identity2(self, op):
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [1, 2, 3]})

        operand = 2
        if op in ("and", "or", "xor"):
            # cannot use floats for boolean ops
            df["a"] = [True, False, True]

        df_copy = df.copy()
        iop = f"__i{op}__"
        op = f"__{op}__"

        # no id change and value is correct
        getattr(df, iop)(operand)
        expected = getattr(df_copy, op)(operand)
        tm.assert_frame_equal(df, expected)
        expected = id(df)
        assert id(df) == expected

    @pytest.mark.parametrize(
        "val",
        [
            [1, 2, 3],
            (1, 2, 3),
            np.array([1, 2, 3], dtype=np.int64),
            range(1, 4),
        ],
    )
    def test_alignment_non_pandas(self, val):
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        align = DataFrame._align_for_op

        expected = DataFrame({"X": val, "Y": val, "Z": val}, index=df.index)
        tm.assert_frame_equal(align(df, val, axis=0)[1], expected)

        expected = DataFrame(
            {"X": [1, 1, 1], "Y": [2, 2, 2], "Z": [3, 3, 3]}, index=df.index
        )
        tm.assert_frame_equal(align(df, val, axis=1)[1], expected)

    @pytest.mark.parametrize("val", [[1, 2], (1, 2), np.array([1, 2]), range(1, 3)])
    def test_alignment_non_pandas_length_mismatch(self, val):
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        align = DataFrame._align_for_op
        # length mismatch
        msg = "Unable to coerce to Series, length must be 3: given 2"
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)

        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    def test_alignment_non_pandas_index_columns(self):
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        align = DataFrame._align_for_op
        val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tm.assert_frame_equal(
            align(df, val, axis=0)[1],
            DataFrame(val, index=df.index, columns=df.columns),
        )
        tm.assert_frame_equal(
            align(df, val, axis=1)[1],
            DataFrame(val, index=df.index, columns=df.columns),
        )

        # shape mismatch
        msg = "Unable to coerce to DataFrame, shape must be"
        val = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)

        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

        val = np.zeros((3, 3, 3))
        msg = re.escape(
            "Unable to coerce to Series/DataFrame, dimension must be <= 2: (3, 3, 3)"
        )
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    def test_no_warning(self, all_arithmetic_operators):
        df = DataFrame({"A": [0.0, 0.0], "B": [0.0, None]})
        b = df["B"]
        with tm.assert_produces_warning(None):
            getattr(df, all_arithmetic_operators)(b)

    def test_dunder_methods_binary(self, all_arithmetic_operators):
        # GH#??? frame.__foo__ should only accept one argument
        df = DataFrame({"A": [0.0, 0.0], "B": [0.0, None]})
        b = df["B"]
        with pytest.raises(TypeError, match="takes 2 positional arguments"):
            getattr(df, all_arithmetic_operators)(b, 0)

    def test_align_int_fill_bug(self):
        # GH#910
        X = np.arange(10 * 10, dtype="float64").reshape(10, 10)
        Y = np.ones((10, 1), dtype=int)

        df1 = DataFrame(X)
        df1["0.X"] = Y.squeeze()

        df2 = df1.astype(float)

        result = df1 - df1.mean()
        expected = df2 - df2.mean()
        tm.assert_frame_equal(result, expected)


def test_pow_with_realignment():
    # GH#32685 pow has special semantics for operating with null values
    left = DataFrame({"A": [0, 1, 2]})
    right = DataFrame(index=[0, 1, 2])

    result = left**right
    expected = DataFrame({"A": [np.nan, 1.0, np.nan]})
    tm.assert_frame_equal(result, expected)


def test_dataframe_series_extension_dtypes():
    # https://github.com/pandas-dev/pandas/issues/34311
    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, (10, 3)), columns=["a", "b", "c"]
    )
    ser = Series([1, 2, 3], index=["a", "b", "c"])

    expected = df.to_numpy("int64") + ser.to_numpy("int64").reshape(-1, 3)
    expected = DataFrame(expected, columns=df.columns, dtype="Int64")

    df_ea = df.astype("Int64")
    result = df_ea + ser
    tm.assert_frame_equal(result, expected)
    result = df_ea + ser.astype("Int64")
    tm.assert_frame_equal(result, expected)


def test_dataframe_blockwise_slicelike():
    # GH#34367
    arr = np.random.default_rng(2).integers(0, 1000, (100, 10))
    df1 = DataFrame(arr)
    # Explicit cast to float to avoid implicit cast when setting nan
    df2 = df1.copy().astype({1: "float", 3: "float", 7: "float"})
    df2.iloc[0, [1, 3, 7]] = np.nan

    # Explicit cast to float to avoid implicit cast when setting nan
    df3 = df1.copy().astype({5: "float"})
    df3.iloc[0, [5]] = np.nan

    # Explicit cast to float to avoid implicit cast when setting nan
    df4 = df1.copy().astype({2: "float", 3: "float", 4: "float"})
    df4.iloc[0, np.arange(2, 5)] = np.nan
    # Explicit cast to float to avoid implicit cast when setting nan
    df5 = df1.copy().astype({4: "float", 5: "float", 6: "float"})
    df5.iloc[0, np.arange(4, 7)] = np.nan

    for left, right in [(df1, df2), (df2, df3), (df4, df5)]:
        res = left + right

        expected = DataFrame({i: left[i] + right[i] for i in left.columns})
        tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "df, col_dtype",
    [
        (DataFrame([[1.0, 2.0], [4.0, 5.0]], columns=list("ab")), "float64"),
        (
            DataFrame([[1.0, "b"], [4.0, "b"]], columns=list("ab")).astype(
                {"b": object}
            ),
            "object",
        ),
    ],
)
def test_dataframe_operation_with_non_numeric_types(df, col_dtype):
    # GH #22663
    expected = DataFrame([[0.0, np.nan], [3.0, np.nan]], columns=list("ab"))
    expected = expected.astype({"b": col_dtype})
    result = df + Series([-1.0], index=list("a"))
    tm.assert_frame_equal(result, expected)


def test_arith_reindex_with_duplicates():
    # https://github.com/pandas-dev/pandas/issues/35194
    df1 = DataFrame(data=[[0]], columns=["second"])
    df2 = DataFrame(data=[[0, 0, 0]], columns=["first", "second", "second"])
    result = df1 + df2
    expected = DataFrame([[np.nan, 0, 0]], columns=["first", "second", "second"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("to_add", [[Series([1, 1])], [Series([1, 1]), Series([1, 1])]])
def test_arith_list_of_arraylike_raise(to_add):
    # GH 36702. Raise when trying to add list of array-like to DataFrame
    df = DataFrame({"x": [1, 2], "y": [1, 2]})

    msg = f"Unable to coerce list of {type(to_add[0])} to Series/DataFrame"
    with pytest.raises(ValueError, match=msg):
        df + to_add
    with pytest.raises(ValueError, match=msg):
        to_add + df


def test_inplace_arithmetic_series_update(using_copy_on_write, warn_copy_on_write):
    # https://github.com/pandas-dev/pandas/issues/36373
    df = DataFrame({"A": [1, 2, 3]})
    df_orig = df.copy()
    series = df["A"]
    vals = series._values

    with tm.assert_cow_warning(warn_copy_on_write):
        series += 1
    if using_copy_on_write:
        assert series._values is not vals
        tm.assert_frame_equal(df, df_orig)
    else:
        assert series._values is vals

        expected = DataFrame({"A": [2, 3, 4]})
        tm.assert_frame_equal(df, expected)


def test_arithmetic_multiindex_align():
    """
    Regression test for: https://github.com/pandas-dev/pandas/issues/33765
    """
    df1 = DataFrame(
        [[1]],
        index=["a"],
        columns=MultiIndex.from_product([[0], [1]], names=["a", "b"]),
    )
    df2 = DataFrame([[1]], index=["a"], columns=Index([0], name="a"))
    expected = DataFrame(
        [[0]],
        index=["a"],
        columns=MultiIndex.from_product([[0], [1]], names=["a", "b"]),
    )
    result = df1 - df2
    tm.assert_frame_equal(result, expected)


def test_bool_frame_mult_float():
    # GH 18549
    df = DataFrame(True, list("ab"), list("cd"))
    result = df * 1.0
    expected = DataFrame(np.ones((2, 2)), list("ab"), list("cd"))
    tm.assert_frame_equal(result, expected)


def test_frame_sub_nullable_int(any_int_ea_dtype):
    # GH 32822
    series1 = Series([1, 2, None], dtype=any_int_ea_dtype)
    series2 = Series([1, 2, 3], dtype=any_int_ea_dtype)
    expected = DataFrame([0, 0, None], dtype=any_int_ea_dtype)
    result = series1.to_frame() - series2.to_frame()
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)
def test_frame_op_subclass_nonclass_constructor():
    # GH#43201 subclass._constructor is a function, not the subclass itself

    class SubclassedSeries(Series):
        @property
        def _constructor(self):
            return SubclassedSeries

        @property
        def _constructor_expanddim(self):
            return SubclassedDataFrame

    class SubclassedDataFrame(DataFrame):
        _metadata = ["my_extra_data"]

        def __init__(self, my_extra_data, *args, **kwargs) -> None:
            self.my_extra_data = my_extra_data
            super().__init__(*args, **kwargs)

        @property
        def _constructor(self):
            return functools.partial(type(self), self.my_extra_data)

        @property
        def _constructor_sliced(self):
            return SubclassedSeries

    sdf = SubclassedDataFrame("some_data", {"A": [1, 2, 3], "B": [4, 5, 6]})
    result = sdf * 2
    expected = SubclassedDataFrame("some_data", {"A": [2, 4, 6], "B": [8, 10, 12]})
    tm.assert_frame_equal(result, expected)

    result = sdf + sdf
    tm.assert_frame_equal(result, expected)


def test_enum_column_equality():
    Cols = Enum("Cols", "col1 col2")

    q1 = DataFrame({Cols.col1: [1, 2, 3]})
    q2 = DataFrame({Cols.col1: [1, 2, 3]})

    result = q1[Cols.col1] == q2[Cols.col1]
    expected = Series([True, True, True], name=Cols.col1)

    tm.assert_series_equal(result, expected)


def test_mixed_col_index_dtype(using_infer_string):
    # GH 47382
    df1 = DataFrame(columns=list("abc"), data=1.0, index=[0])
    df2 = DataFrame(columns=list("abc"), data=0.0, index=[0])
    df1.columns = df2.columns.astype("string")
    result = df1 + df2
    expected = DataFrame(columns=list("abc"), data=1.0, index=[0])
    if using_infer_string:
        # df2.columns.dtype will be "str" instead of object,
        #  so the aligned result will be "string", not object
        if HAS_PYARROW:
            dtype = "string[pyarrow]"
        else:
            dtype = "string"
        expected.columns = expected.columns.astype(dtype)
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_biasgelu.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionBiasGelu(Fusion):
    def __init__(self, model: OnnxModel, is_fastgelu):
        if is_fastgelu:
            super().__init__(model, "FastGelu", "FastGelu", "add bias")
        else:
            super().__init__(model, "BiasGelu", "Gelu")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        gelu_op_type = node.op_type
        fuse_op_type = "BiasGelu" if gelu_op_type == "Gelu" else "FastGelu"

        if len(node.input) != 1:
            return

        nodes = self.model.match_parent_path(node, ["Add", "MatMul"], [0, None])
        if nodes is None:
            return
        (add, matmul) = nodes

        bias_weight = None
        # bias should be one dimension
        bias_index = -1
        for i, input in enumerate(add.input):
            initializer = self.model.get_initializer(input)
            if initializer is None:
                continue
            bias_index = i
            bias_weight = NumpyHelper.to_array(initializer)
            break
        if bias_weight is None:
            return
        if len(bias_weight.shape) != 1:
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes, [node.output[0]], input_name_to_nodes, output_name_to_node
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        fused_node = helper.make_node(
            fuse_op_type,
            inputs=[matmul.output[0], add.input[bias_index]],
            outputs=node.output,
            name=self.model.create_node_name(fuse_op_type, gelu_op_type + "_AddBias_"),
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\truststore\_openssl.py ===
import contextlib
import os
import re
import ssl
import typing

# candidates based on https://github.com/tiran/certifi-system-store by Christian Heimes
_CA_FILE_CANDIDATES = [
    # Alpine, Arch, Fedora 34+, OpenWRT, RHEL 9+, BSD
    "/etc/ssl/cert.pem",
    # Fedora <= 34, RHEL <= 9, CentOS <= 9
    "/etc/pki/tls/cert.pem",
    # Debian, Ubuntu (requires ca-certificates)
    "/etc/ssl/certs/ca-certificates.crt",
    # SUSE
    "/etc/ssl/ca-bundle.pem",
]

_HASHED_CERT_FILENAME_RE = re.compile(r"^[0-9a-fA-F]{8}\.[0-9]$")


@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    # First, check whether the default locations from OpenSSL
    # seem like they will give us a usable set of CA certs.
    # ssl.get_default_verify_paths already takes care of:
    # - getting cafile from either the SSL_CERT_FILE env var
    #   or the path configured when OpenSSL was compiled,
    #   and verifying that that path exists
    # - getting capath from either the SSL_CERT_DIR env var
    #   or the path configured when OpenSSL was compiled,
    #   and verifying that that path exists
    # In addition we'll check whether capath appears to contain certs.
    defaults = ssl.get_default_verify_paths()
    if defaults.cafile or (defaults.capath and _capath_contains_certs(defaults.capath)):
        ctx.set_default_verify_paths()
    else:
        # cafile from OpenSSL doesn't exist
        # and capath from OpenSSL doesn't contain certs.
        # Let's search other common locations instead.
        for cafile in _CA_FILE_CANDIDATES:
            if os.path.isfile(cafile):
                ctx.load_verify_locations(cafile=cafile)
                break

    yield


def _capath_contains_certs(capath: str) -> bool:
    """Check whether capath exists and contains certs in the expected format."""
    if not os.path.isdir(capath):
        return False
    for name in os.listdir(capath):
        if _HASHED_CERT_FILENAME_RE.match(name):
            return True
    return False


def _verify_peercerts_impl(
    ssl_context: ssl.SSLContext,
    cert_chain: list[bytes],
    server_hostname: str | None = None,
) -> None:
    # This is a no-op because we've enabled SSLContext's built-in
    # verification via verify_mode=CERT_REQUIRED, and don't need to repeat it.
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\rules.py ===
"""
Replacement rules.
"""

class Transform:
    """
    Immutable mapping that can be used as a generic transformation rule.

    Parameters
    ==========

    transform : callable
        Computes the value corresponding to any key.

    filter : callable, optional
        If supplied, specifies which objects are in the mapping.

    Examples
    ========

    >>> from sympy.core.rules import Transform
    >>> from sympy.abc import x

    This Transform will return, as a value, one more than the key:

    >>> add1 = Transform(lambda x: x + 1)
    >>> add1[1]
    2
    >>> add1[x]
    x + 1

    By default, all values are considered to be in the dictionary. If a filter
    is supplied, only the objects for which it returns True are considered as
    being in the dictionary:

    >>> add1_odd = Transform(lambda x: x + 1, lambda x: x%2 == 1)
    >>> 2 in add1_odd
    False
    >>> add1_odd.get(2, 0)
    0
    >>> 3 in add1_odd
    True
    >>> add1_odd[3]
    4
    >>> add1_odd.get(3, 0)
    4
    """

    def __init__(self, transform, filter=lambda x: True):
        self._transform = transform
        self._filter = filter

    def __contains__(self, item):
        return self._filter(item)

    def __getitem__(self, key):
        if self._filter(key):
            return self._transform(key)
        else:
            raise KeyError(key)

    def get(self, item, default=None):
        if item in self:
            return self[item]
        else:
            return default

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\benchmarks\bench_galoispolys.py ===
"""Benchmarks for polynomials over Galois fields. """


from sympy.polys.galoistools import gf_from_dict, gf_factor_sqf
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime


def gathen_poly(n, p, K):
    return gf_from_dict({n: K.one, 1: K.one, 0: K.one}, p, K)


def shoup_poly(n, p, K):
    f = [K.one] * (n + 1)
    for i in range(1, n + 1):
        f[i] = (f[i - 1]**2 + K.one) % p
    return f


def genprime(n, K):
    return K(nextprime(int((2**n * pi).evalf())))

p_10 = genprime(10, ZZ)
f_10 = gathen_poly(10, p_10, ZZ)

p_20 = genprime(20, ZZ)
f_20 = gathen_poly(20, p_20, ZZ)


def timeit_gathen_poly_f10_zassenhaus():
    gf_factor_sqf(f_10, p_10, ZZ, method='zassenhaus')


def timeit_gathen_poly_f10_shoup():
    gf_factor_sqf(f_10, p_10, ZZ, method='shoup')


def timeit_gathen_poly_f20_zassenhaus():
    gf_factor_sqf(f_20, p_20, ZZ, method='zassenhaus')


def timeit_gathen_poly_f20_shoup():
    gf_factor_sqf(f_20, p_20, ZZ, method='shoup')

P_08 = genprime(8, ZZ)
F_10 = shoup_poly(10, P_08, ZZ)

P_18 = genprime(18, ZZ)
F_20 = shoup_poly(20, P_18, ZZ)


def timeit_shoup_poly_F10_zassenhaus():
    gf_factor_sqf(F_10, P_08, ZZ, method='zassenhaus')


def timeit_shoup_poly_F10_shoup():
    gf_factor_sqf(F_10, P_08, ZZ, method='shoup')


def timeit_shoup_poly_F20_zassenhaus():
    gf_factor_sqf(F_20, P_18, ZZ, method='zassenhaus')


def timeit_shoup_poly_F20_shoup():
    gf_factor_sqf(F_20, P_18, ZZ, method='shoup')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\stochastic_process.py ===
from sympy.core.basic import Basic
from sympy.stats.joint_rv import ProductPSpace
from sympy.stats.rv import ProductDomain, _symbol_converter, Distribution


class StochasticPSpace(ProductPSpace):
    """
    Represents probability space of stochastic processes
    and their random variables. Contains mechanics to do
    computations for queries of stochastic processes.

    Explanation
    ===========

    Initialized by symbol, the specific process and
    distribution(optional) if the random indexed symbols
    of the process follows any specific distribution, like,
    in Bernoulli Process, each random indexed symbol follows
    Bernoulli distribution. For processes with memory, this
    parameter should not be passed.
    """

    def __new__(cls, sym, process, distribution=None):
        sym = _symbol_converter(sym)
        from sympy.stats.stochastic_process_types import StochasticProcess
        if not isinstance(process, StochasticProcess):
            raise TypeError("`process` must be an instance of StochasticProcess.")
        if distribution is None:
            distribution = Distribution()
        return Basic.__new__(cls, sym, process, distribution)

    @property
    def process(self):
        """
        The associated stochastic process.
        """
        return self.args[1]

    @property
    def domain(self):
        return ProductDomain(self.process.index_set,
                             self.process.state_space)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[2]

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        """
        return self.process.probability(condition, given_condition, evaluate, **kwargs)

    def compute_expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        """
        return self.process.expectation(expr, condition, evaluate, **kwargs)

# === atelier-kyo-manager/tools\check_csv.py ===
import pandas as pd

# CSVファイルをcp932で読み込む
df = pd.read_csv('tools/buyma_products.csv', encoding='cp932')

def check_constraints(product):
    errors = []
    # 必要に応じて型変換
    try:
        price = float(product.get("価格", 0))
        weight = float(product.get("重量", 0))
        battery = str(product.get("電池有無", "")).strip()
        size_sum = float(product.get("3辺和", 0))
        insurance = float(product.get("保険金額", 0))
        region = str(product.get("地域", "")).strip()
        anonymity = str(product.get("匿名希望", "")).strip() == "あり"
        shipping = str(product.get("発送方法", "")).strip()
    except Exception as e:
        errors.append(f"データ型エラー: {e}")
        return errors

    if price >= 500000 and shipping == "DHL EXPRESS":
        errors.append("DHLは50万円以上の商品不可")
    if weight > 10 and shipping == "DHL EXPRESS":
        errors.append("DHLは10kg超不可")
    if battery == "あり" and shipping == "Speedpost":
        errors.append("Speedpostは電池類不可")
    if size_sum > 200 and shipping == "BUYMA YAMATO":
        errors.append("BUYMA YAMATOは3辺和200cm超不可")
    if insurance > 500 and shipping == "USPS特別割引":
        errors.append("USPS特別割引は$500超不可")
    return errors

def select_shipping(product):
    price = float(product.get("価格", 0))
    weight = float(product.get("重量", 0))
    battery = str(product.get("電池有無", "")).strip()
    region = str(product.get("地域", "")).strip()
    anonymity = str(product.get("匿名希望", "")).strip() == "あり"

    if price >= 500000 or weight > 10:
        return "BUYMA YAMATO" if region == "北米" else "個別見積もり"
    if battery == "あり":
        return "DHL EXPRESS"
    if region == "北米" and anonymity:
        return "BUYMA YAMATO"
    if price <= 500 and region == "アメリカ":
        return "USPS特別割引"
    if region == "香港" and battery != "あり":
        return "Speedpost"
    return "標準発送"

# 商品ごとに判定＆チェック
for idx, row in df.iterrows():
    product = row.to_dict()
    product["発送方法"] = select_shipping(product)
    errors = check_constraints(product)
    if errors:
        print(f"エラー: {product.get('商品管理番号', idx+1)} - {', '.join(errors)}")
    else:
        print(f"OK: {product.get('商品管理番号', idx+1)} - {product['発送方法']}")

# 必要なら発送方法をファイルに保存
df["発送方法自動判定"] = df.apply(select_shipping, axis=1)
df.to_csv('tools/buyma_products_checked.csv', encoding='cp932', index=False)
print("\n判定結果を 'tools/buyma_products_checked.csv' に保存しました。")

# === atelier-kyo-manager/tools\scripts\shipping_checker.py ===
import pandas as pd

# CSVファイルをcp932で読み込む（正しいパスに修正）
df = pd.read_csv('../buyma_products.csv', encoding='cp932')

def check_constraints(product):
    errors = []
    # 必要に応じて型変換
    try:
        price = float(product.get("価格", 0))
        weight = float(product.get("重量", 0))
        battery = str(product.get("電池有無", "")).strip()
        size_sum = float(product.get("3辺和", 0))
        insurance = float(product.get("保険金額", 0))
        region = str(product.get("地域", "")).strip()
        anonymity = str(product.get("匿名希望", "")).strip() == "あり"
        shipping = str(product.get("発送方法", "")).strip()
    except Exception as e:
        errors.append(f"データ型エラー: {e}")
        return errors

    if price >= 500000 and shipping == "DHL EXPRESS":
        errors.append("DHLは50万円以上の商品不可")
    if weight > 10 and shipping == "DHL EXPRESS":
        errors.append("DHLは10kg超不可")
    if battery == "あり" and shipping == "Speedpost":
        errors.append("Speedpostは電池類不可")
    if size_sum > 200 and shipping == "BUYMA YAMATO":
        errors.append("BUYMA YAMATOは3辺和200cm超不可")
    if insurance > 500 and shipping == "USPS特別割引":
        errors.append("USPS特別割引は$500超不可")
    return errors

def select_shipping(product):
    price = float(product.get("価格", 0))
    weight = float(product.get("重量", 0))
    battery = str(product.get("電池有無", "")).strip()
    region = str(product.get("地域", "")).strip()
    anonymity = str(product.get("匿名希望", "")).strip() == "あり"

    if price >= 500000 or weight > 10:
        return "BUYMA YAMATO" if region == "北米" else "個別見積もり"
    if battery == "あり":
        return "DHL EXPRESS"
    if region == "北米" and anonymity:
        return "BUYMA YAMATO"
    if price <= 500 and region == "アメリカ":
        return "USPS特別割引"
    if region == "香港" and battery != "あり":
        return "Speedpost"
    return "標準発送"

# 商品ごとに判定＆チェック
for idx, row in df.iterrows():
    product = row.to_dict()
    product["発送方法自動判定"] = select_shipping(product)
    errors = check_constraints(product)
    if errors:
        print(f"エラー: {product.get('商品管理番号', idx+1)} - {', '.join(errors)}")
    else:
        print(f"OK: {product.get('商品管理番号', idx+1)} - {product['発送方法自動判定']}")

# 発送方法自動判定をCSVに保存
df["発送方法自動判定"] = df.apply(select_shipping, axis=1)
df.to_csv('../buyma_products_checked.csv', encoding='cp932', index=False)
print("\n判定結果を '../buyma_products_checked.csv' に保存しました。")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_randomstate.py ===
import hashlib
import pickle
import sys
import warnings

import pytest

import numpy as np
from numpy import random
from numpy.random import MT19937, PCG64
from numpy.testing import (
    IS_WASM,
    assert_,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_no_warnings,
    assert_raises,
    assert_warns,
    suppress_warnings,
)

INT_FUNCS = {'binomial': (100.0, 0.6),
             'geometric': (.5,),
             'hypergeometric': (20, 20, 10),
             'logseries': (.5,),
             'multinomial': (20, np.ones(6) / 6.0),
             'negative_binomial': (100, .5),
             'poisson': (10.0,),
             'zipf': (2,),
             }

if np.iinfo(np.long).max < 2**32:
    # Windows and some 32-bit platforms, e.g., ARM
    INT_FUNC_HASHES = {'binomial':          '2fbead005fc63942decb5326d36a1f32fe2c9d32c904ee61e46866b88447c263',  # noqa: E501
                       'logseries':         '23ead5dcde35d4cfd4ef2c105e4c3d43304b45dc1b1444b7823b9ee4fa144ebb',  # noqa: E501
                       'geometric':         '0d764db64f5c3bad48c8c33551c13b4d07a1e7b470f77629bef6c985cac76fcf',  # noqa: E501
                       'hypergeometric':    '7b59bf2f1691626c5815cdcd9a49e1dd68697251d4521575219e4d2a1b8b2c67',  # noqa: E501
                       'multinomial':       'd754fa5b92943a38ec07630de92362dd2e02c43577fc147417dc5b9db94ccdd3',  # noqa: E501
                       'negative_binomial': '8eb216f7cb2a63cf55605422845caaff002fddc64a7dc8b2d45acd477a49e824',  # noqa: E501
                       'poisson':           '70c891d76104013ebd6f6bcf30d403a9074b886ff62e4e6b8eb605bf1a4673b7',  # noqa: E501
                       'zipf':              '01f074f97517cd5d21747148ac6ca4074dde7fcb7acbaec0a936606fecacd93f',  # noqa: E501
                       }
else:
    INT_FUNC_HASHES = {'binomial':          '8626dd9d052cb608e93d8868de0a7b347258b199493871a1dc56e2a26cacb112',  # noqa: E501
                       'geometric':         '8edd53d272e49c4fc8fbbe6c7d08d563d62e482921f3131d0a0e068af30f0db9',  # noqa: E501
                       'hypergeometric':    '83496cc4281c77b786c9b7ad88b74d42e01603a55c60577ebab81c3ba8d45657',  # noqa: E501
                       'logseries':         '65878a38747c176bc00e930ebafebb69d4e1e16cd3a704e264ea8f5e24f548db',  # noqa: E501
                       'multinomial':       '7a984ae6dca26fd25374479e118b22f55db0aedccd5a0f2584ceada33db98605',  # noqa: E501
                       'negative_binomial': 'd636d968e6a24ae92ab52fe11c46ac45b0897e98714426764e820a7d77602a61',  # noqa: E501
                       'poisson':           '956552176f77e7c9cb20d0118fc9cf690be488d790ed4b4c4747b965e61b0bb4',  # noqa: E501
                       'zipf':              'f84ba7feffda41e606e20b28dfc0f1ea9964a74574513d4a4cbc98433a8bfa45',  # noqa: E501
                       }


@pytest.fixture(scope='module', params=INT_FUNCS)
def int_func(request):
    return (request.param, INT_FUNCS[request.param],
            INT_FUNC_HASHES[request.param])


@pytest.fixture
def restore_singleton_bitgen():
    """Ensures that the singleton bitgen is restored after a test"""
    orig_bitgen = np.random.get_bit_generator()
    yield
    np.random.set_bit_generator(orig_bitgen)


def assert_mt19937_state_equal(a, b):
    assert_equal(a['bit_generator'], b['bit_generator'])
    assert_array_equal(a['state']['key'], b['state']['key'])
    assert_array_equal(a['state']['pos'], b['state']['pos'])
    assert_equal(a['has_gauss'], b['has_gauss'])
    assert_equal(a['gauss'], b['gauss'])


class TestSeed:
    def test_scalar(self):
        s = random.RandomState(0)
        assert_equal(s.randint(1000), 684)
        s = random.RandomState(4294967295)
        assert_equal(s.randint(1000), 419)

    def test_array(self):
        s = random.RandomState(range(10))
        assert_equal(s.randint(1000), 468)
        s = random.RandomState(np.arange(10))
        assert_equal(s.randint(1000), 468)
        s = random.RandomState([0])
        assert_equal(s.randint(1000), 973)
        s = random.RandomState([4294967295])
        assert_equal(s.randint(1000), 265)

    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, random.RandomState, -0.5)
        assert_raises(ValueError, random.RandomState, -1)

    def test_invalid_array(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, random.RandomState, [-0.5])
        assert_raises(ValueError, random.RandomState, [-1])
        assert_raises(ValueError, random.RandomState, [4294967296])
        assert_raises(ValueError, random.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, random.RandomState, [1, -2, 4294967296])

    def test_invalid_array_shape(self):
        # gh-9832
        assert_raises(ValueError, random.RandomState, np.array([],
                                                               dtype=np.int64))
        assert_raises(ValueError, random.RandomState, [[1, 2, 3]])
        assert_raises(ValueError, random.RandomState, [[1, 2, 3],
                                                       [4, 5, 6]])

    def test_cannot_seed(self):
        rs = random.RandomState(PCG64(0))
        with assert_raises(TypeError):
            rs.seed(1234)

    def test_invalid_initialization(self):
        assert_raises(ValueError, random.RandomState, MT19937)


class TestBinomial:
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        # This test addresses issue #3480.
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            assert_(random.binomial(0, p) == 0)
            assert_array_equal(random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        # Issue #4571.
        assert_raises(ValueError, random.binomial, 1, np.nan)


class TestMultinomial:
    def test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert_(-5 <= random.randint(-5, -1) < -1)
        x = random.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def test_size(self):
        # gh-3173
        p = [0.5, 0.5]
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, random.multinomial, 1, p,
                      float(1))

    def test_invalid_prob(self):
        assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
        assert_raises(ValueError, random.multinomial, 100, [-.1, 0.9])

    def test_invalid_n(self):
        assert_raises(ValueError, random.multinomial, -1, [0.8, 0.2])

    def test_p_non_contiguous(self):
        p = np.arange(15.)
        p /= np.sum(p[1::3])
        pvals = p[1::3]
        random.seed(1432985819)
        non_contig = random.multinomial(100, pvals=pvals)
        random.seed(1432985819)
        contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
        assert_array_equal(non_contig, contig)

    def test_multinomial_pvals_float32(self):
        x = np.array([9.9e-01, 9.9e-01, 1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09,
                      1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09], dtype=np.float32)
        pvals = x / x.sum()
        match = r"[\w\s]*pvals array is cast to 64-bit floating"
        with pytest.raises(ValueError, match=match):
            random.multinomial(1, pvals)

    def test_multinomial_n_float(self):
        # Non-index integer types should gracefully truncate floats
        random.multinomial(100.5, [0.2, 0.8])

class TestSetState:
    def setup_method(self):
        self.seed = 1234567890
        self.random_state = random.RandomState(self.seed)
        self.state = self.random_state.get_state()

    def test_basic(self):
        old = self.random_state.tomaxint(16)
        self.random_state.set_state(self.state)
        new = self.random_state.tomaxint(16)
        assert_(np.all(old == new))

    def test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.random_state.standard_normal(size=3)
        self.random_state.set_state(self.state)
        new = self.random_state.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.

        self.random_state.standard_normal()
        state = self.random_state.get_state()
        old = self.random_state.standard_normal(size=3)
        self.random_state.set_state(state)
        new = self.random_state.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_backwards_compatibility(self):
        # Make sure we can accept old state tuples that do not have the
        # cached Gaussian value.
        old_state = self.state[:-2]
        x1 = self.random_state.standard_normal(size=16)
        self.random_state.set_state(old_state)
        x2 = self.random_state.standard_normal(size=16)
        self.random_state.set_state(self.state)
        x3 = self.random_state.standard_normal(size=16)
        assert_(np.all(x1 == x2))
        assert_(np.all(x1 == x3))

    def test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.random_state.negative_binomial(0.5, 0.5)

    def test_get_state_warning(self):
        rs = random.RandomState(PCG64())
        with suppress_warnings() as sup:
            w = sup.record(RuntimeWarning)
            state = rs.get_state()
            assert_(len(w) == 1)
            assert isinstance(state, dict)
            assert state['bit_generator'] == 'PCG64'

    def test_invalid_legacy_state_setting(self):
        state = self.random_state.get_state()
        new_state = ('Unknown', ) + state[1:]
        assert_raises(ValueError, self.random_state.set_state, new_state)
        assert_raises(TypeError, self.random_state.set_state,
                      np.array(new_state, dtype=object))
        state = self.random_state.get_state(legacy=False)
        del state['bit_generator']
        assert_raises(ValueError, self.random_state.set_state, state)

    def test_pickle(self):
        self.random_state.seed(0)
        self.random_state.random_sample(100)
        self.random_state.standard_normal()
        pickled = self.random_state.get_state(legacy=False)
        assert_equal(pickled['has_gauss'], 1)
        rs_unpick = pickle.loads(pickle.dumps(self.random_state))
        unpickled = rs_unpick.get_state(legacy=False)
        assert_mt19937_state_equal(pickled, unpickled)

    def test_state_setting(self):
        attr_state = self.random_state.__getstate__()
        self.random_state.standard_normal()
        self.random_state.__setstate__(attr_state)
        state = self.random_state.get_state(legacy=False)
        assert_mt19937_state_equal(attr_state, state)

    def test_repr(self):
        assert repr(self.random_state).startswith('RandomState(MT19937)')


class TestRandint:

    rfunc = random.randint

    # valid integer/boolean types
    itype = [np.bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self):
        assert_raises(TypeError, self.rfunc, 1, dtype=float)

    def test_bounds_checking(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)

    def test_rng_zero_and_extremes(self):
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    def test_full_range(self):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            try:
                self.rfunc(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_in_bounds_fuzz(self):
        # Don't use fixed seed
        random.seed()

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd, size=2**16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = self.rfunc(0, 2, size=2**16, dtype=np.bool)

        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_repeatability(self):
        # We use a sha256 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool':   '509aea74d792fb931784c4b0135392c65aec64beee12b0cc167548a2c3d31e71',  # noqa: E501
               'int16':  '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',  # noqa: E501
               'int32':  'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',  # noqa: E501
               'int64':  '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',  # noqa: E501
               'int8':   '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404',  # noqa: E501
               'uint16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4',  # noqa: E501
               'uint32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f',  # noqa: E501
               'uint64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e',  # noqa: E501
               'uint8':  '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404'}  # noqa: E501

        for dt in self.itype[1:]:
            random.seed(1234)

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()

            res = hashlib.sha256(val.view(np.int8)).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianness
        random.seed(1234)
        val = self.rfunc(0, 2, size=1000, dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    @pytest.mark.skipif(np.iinfo('l').max < 2**32,
                        reason='Cannot test with 32-bit C long')
    def test_repeatability_32bit_boundary_broadcasting(self):
        desired = np.array([[[3992670689, 2438360420, 2557845020],
                             [4107320065, 4142558326, 3216529513],
                             [1605979228, 2807061240,  665605495]],
                            [[3211410639, 4128781000,  457175120],
                             [1712592594, 1282922662, 3081439808],
                             [3997822960, 2008322436, 1563495165]],
                            [[1398375547, 4269260146,  115316740],
                             [3414372578, 3437564012, 2112038651],
                             [3572980305, 2260248732, 3908238631]],
                            [[2561372503,  223155946, 3127879445],
                             [ 441282060, 3514786552, 2148440361],
                             [1629275283, 3479737011, 3003195987]],
                            [[ 412181688,  940383289, 3047321305],
                             [2978368172,  764731833, 2282559898],
                             [ 105711276,  720447391, 3596512484]]])
        for size in [None, (5, 3, 3)]:
            random.seed(12345)
            x = self.rfunc([[-1], [0], [1]], [2**32 - 1, 2**32, 2**32 + 1],
                           size=size)
            assert_array_equal(x, desired if size is not None else desired[0])

    def test_int64_uint64_corner_case(self):
        # When stored in Numpy arrays, `lbnd` is casted
        # as np.int64, and `ubnd` is casted as np.uint64.
        # Checking whether `lbnd` >= `ubnd` used to be
        # done solely via direct comparison, which is incorrect
        # because when Numpy tries to compare both numbers,
        # it casts both to np.float64 because there is
        # no integer superset of np.int64 and np.uint64. However,
        # `ubnd` is too large to be represented in np.float64,
        # causing it be round down to np.iinfo(np.int64).max,
        # leading to a ValueError because `lbnd` now equals
        # the new `ubnd`.

        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)

        # None of these function calls should
        # generate a ValueError now.
        actual = random.randint(lbnd, ubnd, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_equal(sample.dtype, np.dtype(dt))

        for dt in (bool, int):
            # The legacy random generation forces the use of "long" on this
            # branch even when the input is `int` and the default dtype
            # for int changed (dtype=int is also the functions default)
            op_dtype = "long" if dt is int else "bool"
            lbnd = 0 if dt is bool else np.iinfo(op_dtype).min
            ubnd = 2 if dt is bool else np.iinfo(op_dtype).max + 1

            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_(not hasattr(sample, 'dtype'))
            assert_equal(type(sample), dt)


class TestRandomDist:
    # Make sure the random distribution returns the correct value for a
    # given seed

    def setup_method(self):
        self.seed = 1234567890

    def test_rand(self):
        random.seed(self.seed)
        actual = random.rand(3, 2)
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rand_singleton(self):
        random.seed(self.seed)
        actual = random.rand()
        desired = 0.61879477158567997
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn(self):
        random.seed(self.seed)
        actual = random.randn(3, 2)
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                           [1.498988344300628, -0.2286433324536169],
                           [2.031033998682787, 2.17032494605655257]])
        assert_array_almost_equal(actual, desired, decimal=15)

        random.seed(self.seed)
        actual = random.randn()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_randint(self):
        random.seed(self.seed)
        actual = random.randint(-99, 99, size=(3, 2))
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers(self):
        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(-99, 99, size=(3, 2))
            assert_(len(w) == 1)
        desired = np.array([[31, 3],
                            [-52, 41],
                            [-48, -66]])
        assert_array_equal(actual, desired)

        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(198, size=(3, 2))
            assert_(len(w) == 1)
        assert_array_equal(actual, desired + 100)

    def test_tomaxint(self):
        random.seed(self.seed)
        rs = random.RandomState(self.seed)
        actual = rs.tomaxint(size=(3, 2))
        if np.iinfo(np.long).max == 2147483647:
            desired = np.array([[1328851649,  731237375],
                                [1270502067,  320041495],
                                [1908433478,  499156889]], dtype=np.int64)
        else:
            desired = np.array([[5707374374421908479, 5456764827585442327],
                                [8196659375100692377, 8224063923314595285],
                                [4220315081820346526, 7177518203184491332]],
                               dtype=np.int64)

        assert_equal(actual, desired)

        rs.seed(self.seed)
        actual = rs.tomaxint()
        assert_equal(actual, desired[0, 0])

    def test_random_integers_max_int(self):
        # Tests whether random_integers can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(np.iinfo('l').max,
                                            np.iinfo('l').max)
            assert_(len(w) == 1)

        desired = np.iinfo('l').max
        assert_equal(actual, desired)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            typer = np.dtype('l').type
            actual = random.random_integers(typer(np.iinfo('l').max),
                                            typer(np.iinfo('l').max))
            assert_(len(w) == 1)
        assert_equal(actual, desired)

    def test_random_integers_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # DeprecationWarning raised with high == None
            assert_raises(DeprecationWarning,
                          random.random_integers,
                          np.iinfo('l').max)

            # DeprecationWarning raised with high != None
            assert_raises(DeprecationWarning,
                          random.random_integers,
                          np.iinfo('l').max, np.iinfo('l').max)

    def test_random_sample(self):
        random.seed(self.seed)
        actual = random.random_sample((3, 2))
        desired = np.array([[0.61879477158567997, 0.59162362775974664],
                            [0.88868358904449662, 0.89165480011560816],
                            [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

        random.seed(self.seed)
        actual = random.random_sample()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_choice_uniform_replace(self):
        random.seed(self.seed)
        actual = random.choice(4, 4)
        desired = np.array([2, 3, 2, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_replace(self):
        random.seed(self.seed)
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([1, 1, 2, 2])
        assert_array_equal(actual, desired)

    def test_choice_uniform_noreplace(self):
        random.seed(self.seed)
        actual = random.choice(4, 3, replace=False)
        desired = np.array([0, 1, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_noreplace(self):
        random.seed(self.seed)
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([2, 3, 1])
        assert_array_equal(actual, desired)

    def test_choice_noninteger(self):
        random.seed(self.seed)
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['c', 'd', 'c', 'd'])
        assert_array_equal(actual, desired)

    def test_choice_exceptions(self):
        sample = random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # gh-13087
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])

    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        assert_(np.isscalar(random.choice(2, replace=True)))
        assert_(np.isscalar(random.choice(2, replace=False)))
        assert_(np.isscalar(random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(random.choice([1, 2], replace=True)))
        assert_(random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)

        # Check 0-d array
        s = ()
        assert_(not np.isscalar(random.choice(2, s, replace=True)))
        assert_(not np.isscalar(random.choice(2, s, replace=False)))
        assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))
        assert_(random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)
        assert_equal(random.choice(6, s, replace=False).shape, s)
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)

        # Check zero-size
        assert_equal(random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.randint(0, -10, size=0).shape, (0,))
        assert_equal(random.randint(10, 10, size=0).shape, (0,))
        assert_equal(random.choice(0, size=0).shape, (0,))
        assert_equal(random.choice([], size=(0,)).shape, (0,))
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)

    def test_choice_p_non_contiguous(self):
        p = np.ones(10) / 5
        p[1::2] = 3.0
        random.seed(self.seed)
        non_contig = random.choice(5, 3, p=p[::2])
        random.seed(self.seed)
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)

    def test_bytes(self):
        random.seed(self.seed)
        actual = random.bytes(10)
        desired = b'\x82Ui\x9e\xff\x97+Wf\xa5'
        assert_equal(actual, desired)

    def test_shuffle(self):
        # Test lists, arrays (of various dtypes), and multidimensional versions
        # of both, c-contiguous or not:
        for conv in [lambda x: np.array([]),
                     lambda x: x,
                     lambda x: np.asarray(x).astype(np.int8),
                     lambda x: np.asarray(x).astype(np.float32),
                     lambda x: np.asarray(x).astype(np.complex64),
                     lambda x: np.asarray(x).astype(object),
                     lambda x: [(i, i) for i in x],
                     lambda x: np.asarray([[i, i] for i in x]),
                     lambda x: np.vstack([x, x]).T,
                     # gh-11442
                     lambda x: (np.asarray([(i, i) for i in x],
                                           [("a", int), ("b", int)])
                                .view(np.recarray)),
                     # gh-4270
                     lambda x: np.asarray([(i, i) for i in x],
                                          [("a", object, (1,)),
                                           ("b", np.int32, (1,))])]:
            random.seed(self.seed)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            random.shuffle(alist)
            actual = alist
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
            assert_array_equal(actual, desired)

    def test_shuffle_masked(self):
        # gh-3263
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            random.shuffle(a)
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            random.shuffle(b)
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

        def test_shuffle_invalid_objects(self):
            x = np.array(3)
            assert_raises(TypeError, random.shuffle, x)

    def test_permutation(self):
        random.seed(self.seed)
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actual = random.permutation(alist)
        desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
        assert_array_equal(actual, desired)

        random.seed(self.seed)
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        actual = random.permutation(arr_2d)
        assert_array_equal(actual, np.atleast_2d(desired).T)

        random.seed(self.seed)
        bad_x_str = "abcd"
        assert_raises(IndexError, random.permutation, bad_x_str)

        random.seed(self.seed)
        bad_x_float = 1.2
        assert_raises(IndexError, random.permutation, bad_x_float)

        integer_val = 10
        desired = [9, 0, 8, 5, 1, 3, 4, 7, 6, 2]

        random.seed(self.seed)
        actual = random.permutation(integer_val)
        assert_array_equal(actual, desired)

    def test_beta(self):
        random.seed(self.seed)
        actual = random.beta(.1, .9, size=(3, 2))
        desired = np.array(
                [[1.45341850513746058e-02, 5.31297615662868145e-04],
                 [1.85366619058432324e-06, 4.19214516800110563e-03],
                 [1.58405155108498093e-04, 1.26252891949397652e-04]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        random.seed(self.seed)
        actual = random.binomial(100.123, .456, size=(3, 2))
        desired = np.array([[37, 43],
                            [42, 48],
                            [46, 45]])
        assert_array_equal(actual, desired)

        random.seed(self.seed)
        actual = random.binomial(100.123, .456)
        desired = 37
        assert_array_equal(actual, desired)

    def test_chisquare(self):
        random.seed(self.seed)
        actual = random.chisquare(50, size=(3, 2))
        desired = np.array([[63.87858175501090585, 68.68407748911370447],
                            [65.77116116901505904, 47.09686762438974483],
                            [72.3828403199695174, 74.18408615260374006]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        random.seed(self.seed)
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.54539444573611562, 0.45460555426388438],
                             [0.62345816822039413, 0.37654183177960598]],
                            [[0.55206000085785778, 0.44793999914214233],
                             [0.58964023305154301, 0.41035976694845688]],
                            [[0.59266909280647828, 0.40733090719352177],
                             [0.56974431743975207, 0.43025568256024799]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        bad_alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)

        random.seed(self.seed)
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha)
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_dirichlet_size(self):
        # gh-3173
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        # gh-2089
        alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, alpha)

    def test_dirichlet_alpha_non_contiguous(self):
        a = np.array([51.72840233779265162, -1.0, 39.74494232180943953])
        alpha = a[::2]
        random.seed(self.seed)
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random.seed(self.seed)
        contig = random.dirichlet(np.ascontiguousarray(alpha),
                                  size=(3, 2))
        assert_array_almost_equal(non_contig, contig)

    def test_exponential(self):
        random.seed(self.seed)
        actual = random.exponential(1.1234, size=(3, 2))
        desired = np.array([[1.08342649775011624, 1.00607889924557314],
                            [2.46628830085216721, 2.49668106809923884],
                            [0.68717433461363442, 1.69175666993575979]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        assert_equal(random.exponential(scale=0), 0)
        assert_raises(ValueError, random.exponential, scale=-0.)

    def test_f(self):
        random.seed(self.seed)
        actual = random.f(12, 77, size=(3, 2))
        desired = np.array([[1.21975394418575878, 1.75135759791559775],
                            [1.44803115017146489, 1.22108959480396262],
                            [1.02176975757740629, 1.34431827623300415]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        random.seed(self.seed)
        actual = random.gamma(5, 3, size=(3, 2))
        desired = np.array([[24.60509188649287182, 28.54993563207210627],
                            [26.13476110204064184, 12.56988482927716078],
                            [31.71863275789960568, 33.30143302795922011]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        assert_equal(random.gamma(shape=0, scale=0), 0)
        assert_raises(ValueError, random.gamma, shape=-0., scale=-0.)

    def test_geometric(self):
        random.seed(self.seed)
        actual = random.geometric(.123456789, size=(3, 2))
        desired = np.array([[8, 7],
                            [17, 17],
                            [5, 12]])
        assert_array_equal(actual, desired)

    def test_geometric_exceptions(self):
        assert_raises(ValueError, random.geometric, 1.1)
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        assert_raises(ValueError, random.geometric, -0.1)
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.geometric, np.nan)
            assert_raises(ValueError, random.geometric, [np.nan] * 10)

    def test_gumbel(self):
        random.seed(self.seed)
        actual = random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.19591898743416816, 0.34405539668096674],
                            [-1.4492522252274278, -1.47374816298446865],
                            [1.10651090478803416, -0.69535848626236174]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        assert_equal(random.gumbel(scale=0), 0)
        assert_raises(ValueError, random.gumbel, scale=-0.)

    def test_hypergeometric(self):
        random.seed(self.seed)
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[10, 10],
                            [10, 10],
                            [9, 9]])
        assert_array_equal(actual, desired)

        # Test nbad = 0
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # Test ngood = 0
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def test_laplace(self):
        random.seed(self.seed)
        actual = random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.66599721112760157, 0.52829452552221945],
                            [3.12791959514407125, 3.18202813572992005],
                            [-0.05391065675859356, 1.74901336242837324]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        assert_equal(random.laplace(scale=0), 0)
        assert_raises(ValueError, random.laplace, scale=-0.)

    def test_logistic(self):
        random.seed(self.seed)
        actual = random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[1.09232835305011444, 0.8648196662399954],
                            [4.27818590694950185, 4.33897006346929714],
                            [-0.21682183359214885, 2.63373365386060332]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        random.seed(self.seed)
        actual = random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[16.50698631688883822, 36.54846706092654784],
                            [22.67886599981281748, 0.71617561058995771],
                            [65.72798501792723869, 86.84341601437161273]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_lognormal_0(self):
        assert_equal(random.lognormal(sigma=0), 1)
        assert_raises(ValueError, random.lognormal, sigma=-0.)

    def test_logseries(self):
        random.seed(self.seed)
        actual = random.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[2, 2],
                            [6, 17],
                            [3, 6]])
        assert_array_equal(actual, desired)

    def test_logseries_zero(self):
        assert random.logseries(0) == 1

    @pytest.mark.parametrize("value", [np.nextafter(0., -1), 1., np.nan, 5.])
    def test_logseries_exceptions(self, value):
        with np.errstate(invalid="ignore"):
            with pytest.raises(ValueError):
                random.logseries(value)
            with pytest.raises(ValueError):
                # contiguous path:
                random.logseries(np.array([value] * 10))
            with pytest.raises(ValueError):
                # non-contiguous path:
                random.logseries(np.array([value] * 10)[::2])

    def test_multinomial(self):
        random.seed(self.seed)
        actual = random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[4, 3, 5, 4, 2, 2],
                             [5, 2, 8, 2, 2, 1]],
                            [[3, 4, 3, 6, 0, 4],
                             [2, 1, 4, 3, 6, 4]],
                            [[4, 4, 2, 5, 2, 3],
                             [4, 3, 4, 2, 3, 4]]])
        assert_array_equal(actual, desired)

    def test_multivariate_normal(self):
        random.seed(self.seed)
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = random.multivariate_normal(mean, cov, size)
        desired = np.array([[[1.463620246718631, 11.73759122771936],
                             [1.622445133300628, 9.771356667546383]],
                            [[2.154490787682787, 12.170324946056553],
                             [1.719909438201865, 9.230548443648306]],
                            [[0.689515026297799, 9.880729819607714],
                             [-0.023054015651998, 9.201096623542879]]])

        assert_array_almost_equal(actual, desired, decimal=15)

        # Check for default size, was raising deprecation warning
        actual = random.multivariate_normal(mean, cov)
        desired = np.array([0.895289569463708, 9.17180864067987])
        assert_array_almost_equal(actual, desired, decimal=15)

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')

        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0

        mu = np.zeros(2)
        cov = np.eye(2)
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='other')
        assert_raises(ValueError, random.multivariate_normal,
                      np.zeros((2, 1, 1)), cov)
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.empty((3, 2)))
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.eye(3))

    def test_negative_binomial(self):
        random.seed(self.seed)
        actual = random.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = np.array([[848, 841],
                            [892, 611],
                            [779, 647]])
        assert_array_equal(actual, desired)

    def test_negative_binomial_exceptions(self):
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.negative_binomial, 100, np.nan)
            assert_raises(ValueError, random.negative_binomial, 100,
                          [np.nan] * 10)

    def test_noncentral_chisquare(self):
        random.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[23.91905354498517511, 13.35324692733826346],
                            [31.22452661329736401, 16.60047399466177254],
                            [5.03461598262724586, 17.94973089023519464]])
        assert_array_almost_equal(actual, desired, decimal=14)

        actual = random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = np.array([[1.47145377828516666,  0.15052899268012659],
                            [0.00943803056963588,  1.02647251615666169],
                            [0.332334982684171,  0.15451287602753125]])
        assert_array_almost_equal(actual, desired, decimal=14)

        random.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[9.597154162763948, 11.725484450296079],
                            [10.413711048138335, 3.694475922923986],
                            [13.484222138963087, 14.377255424602957]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        random.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                     size=(3, 2))
        desired = np.array([[1.40598099674926669, 0.34207973179285761],
                            [3.57715069265772545, 7.92632662577829805],
                            [0.43741599463544162, 1.1774208752428319]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f_nan(self):
        random.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        assert np.isnan(actual)

    def test_normal(self):
        random.seed(self.seed)
        actual = random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[2.80378370443726244, 3.59863924443872163],
                            [3.121433477601256, -0.33382987590723379],
                            [4.18552478636557357, 4.46410668111310471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_normal_0(self):
        assert_equal(random.normal(scale=0), 0)
        assert_raises(ValueError, random.normal, scale=-0.)

    def test_pareto(self):
        random.seed(self.seed)
        actual = random.pareto(a=.123456789, size=(3, 2))
        desired = np.array(
                [[2.46852460439034849e+03, 1.41286880810518346e+03],
                 [5.28287797029485181e+07, 6.57720981047328785e+07],
                 [1.40840323350391515e+02, 1.98390255135251704e+05]])
        # For some reason on 32-bit x86 Ubuntu 12.10 the [1, 0] entry in this
        # matrix differs by 24 nulps. Discussion:
        #   https://mail.python.org/pipermail/numpy-discussion/2012-September/063801.html
        # Consensus is that this is probably some gcc quirk that affects
        # rounding but not in any important way, so we just use a looser
        # tolerance on this test:
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        random.seed(self.seed)
        actual = random.poisson(lam=.123456789, size=(3, 2))
        desired = np.array([[0, 0],
                            [1, 0],
                            [0, 0]])
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        lambig = np.iinfo('l').max
        lamneg = -1
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)

    def test_power(self):
        random.seed(self.seed)
        actual = random.power(a=.123456789, size=(3, 2))
        desired = np.array([[0.02048932883240791, 0.01424192241128213],
                            [0.38446073748535298, 0.39499689943484395],
                            [0.00177699707563439, 0.13115505880863756]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        random.seed(self.seed)
        actual = random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[13.8882496494248393, 13.383318339044731],
                            [20.95413364294492098, 21.08285015800712614],
                            [11.06066537006854311, 17.35468505778271009]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        assert_equal(random.rayleigh(scale=0), 0)
        assert_raises(ValueError, random.rayleigh, scale=-0.)

    def test_standard_cauchy(self):
        random.seed(self.seed)
        actual = random.standard_cauchy(size=(3, 2))
        desired = np.array([[0.77127660196445336, -6.55601161955910605],
                            [0.93582023391158309, -2.07479293013759447],
                            [-4.74601644297011926, 0.18338989290760804]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        random.seed(self.seed)
        actual = random.standard_exponential(size=(3, 2))
        desired = np.array([[0.96441739162374596, 0.89556604882105506],
                            [2.1953785836319808, 2.22243285392490542],
                            [0.6116915921431676, 1.50592546727413201]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_gamma(self):
        random.seed(self.seed)
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[5.50841531318455058, 6.62953470301903103],
                            [5.93988484943779227, 2.31044849402133989],
                            [7.54838614231317084, 8.012756093271868]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_gamma_0(self):
        assert_equal(random.standard_gamma(shape=0), 0)
        assert_raises(ValueError, random.standard_gamma, shape=-0.)

    def test_standard_normal(self):
        random.seed(self.seed)
        actual = random.standard_normal(size=(3, 2))
        desired = np.array([[1.34016345771863121, 1.73759122771936081],
                            [1.498988344300628, -0.2286433324536169],
                            [2.031033998682787, 2.17032494605655257]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn_singleton(self):
        random.seed(self.seed)
        actual = random.randn()
        desired = np.array(1.34016345771863121)
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_t(self):
        random.seed(self.seed)
        actual = random.standard_t(df=10, size=(3, 2))
        desired = np.array([[0.97140611862659965, -0.08830486548450577],
                            [1.36311143689505321, -0.55317463909867071],
                            [-0.18473749069684214, 0.61181537341755321]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        random.seed(self.seed)
        actual = random.triangular(left=5.12, mode=10.23, right=20.34,
                                   size=(3, 2))
        desired = np.array([[12.68117178949215784, 12.4129206149193152],
                            [16.20131377335158263, 16.25692138747600524],
                            [11.20400690911820263, 14.4978144835829923]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        random.seed(self.seed)
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[6.99097932346268003, 6.73801597444323974],
                            [9.50364421400426274, 9.53130618907631089],
                            [5.48995325769805476, 8.47493103280052118]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_uniform_range_bounds(self):
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        func = random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # (fmax / 1e17) - fmin is within range, so this should not throw
        # account for i386 extended precision DBL_MAX / 1e17 + DBL_MAX >
        # DBL_MAX by increasing fmin a bit
        random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    def test_scalar_exception_propagation(self):
        # Tests that exceptions are correctly propagated in distributions
        # when called with objects that throw exceptions when converted to
        # scalars.
        #
        # Regression test for gh: 8865

        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, random.uniform, throwing_float,
                      throwing_float)

        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        random.seed(self.seed)
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[2.28567572673902042, 2.89163838442285037],
                            [0.38198375564286025, 2.57638023113890746],
                            [1.19153771588353052, 1.83509849681825354]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        # check infinite loop, gh-4720
        random.seed(self.seed)
        r = random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        assert_(np.isfinite(r).all())

    def test_vonmises_large(self):
        # guard against changes in RandomState when Generator is fixed
        random.seed(self.seed)
        actual = random.vonmises(mu=0., kappa=1e7, size=3)
        desired = np.array([4.634253748521111e-04,
                            3.558873596114509e-04,
                            -2.337119622577433e-04])
        assert_array_almost_equal(actual, desired, decimal=8)

    def test_vonmises_nan(self):
        random.seed(self.seed)
        r = random.vonmises(mu=0., kappa=np.nan)
        assert_(np.isnan(r))

    def test_wald(self):
        random.seed(self.seed)
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[3.82935265715889983, 5.13125249184285526],
                            [0.35045403618358717, 1.50832396872003538],
                            [0.24124319895843183, 0.22031101461955038]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        random.seed(self.seed)
        actual = random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[0.97097342648766727, 0.91422896443565516],
                            [1.89517770034962929, 1.91414357960479564],
                            [0.67057783752390987, 1.39494046635066793]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_weibull_0(self):
        random.seed(self.seed)
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        assert_raises(ValueError, random.weibull, a=-0.)

    def test_zipf(self):
        random.seed(self.seed)
        actual = random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[66, 29],
                            [1, 1],
                            [3, 13]])
        assert_array_equal(actual, desired)


class TestBroadcast:
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments
    def setup_method(self):
        self.seed = 123456789

    def set_seed(self):
        random.seed(self.seed)

    def test_uniform(self):
        low = [0]
        high = [1]
        uniform = random.uniform
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.set_seed()
        actual = uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        self.set_seed()
        actual = uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        normal = random.normal
        desired = np.array([2.2129019979039612,
                            2.1283977976520019,
                            1.8417114045748335])

        self.set_seed()
        actual = normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc * 3, bad_scale)

        self.set_seed()
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        beta = random.beta
        desired = np.array([0.19843558305989056,
                            0.075230336409423643,
                            0.24976865978980844])

        self.set_seed()
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        self.set_seed()
        actual = beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a, b * 3)
        assert_raises(ValueError, beta, a, bad_b * 3)

    def test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        exponential = random.exponential
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.set_seed()
        actual = exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, exponential, bad_scale * 3)

    def test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        std_gamma = random.standard_gamma
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.set_seed()
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        gamma = random.gamma
        desired = np.array([1.5221370731769048,
                            1.5277256455738331,
                            1.4248762625178359])

        self.set_seed()
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        self.set_seed()
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        f = random.f
        desired = np.array([0.80038951638264799,
                            0.86768719635363512,
                            2.7251095168386801])

        self.set_seed()
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        self.set_seed()
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)

    def test_noncentral_f(self):
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        nonc_f = random.noncentral_f
        desired = np.array([9.1393943263705211,
                            13.025456344595602,
                            8.8018098359100545])

        self.set_seed()
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))

        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        self.set_seed()
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        self.set_seed()
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def test_noncentral_f_small_df(self):
        self.set_seed()
        desired = np.array([6.869638627492048, 0.785880199263955])
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_chisquare(self):
        df = [1]
        bad_df = [-1]
        chisquare = random.chisquare
        desired = np.array([0.57022801133088286,
                            0.51947702108840776,
                            0.1320969254923558])

        self.set_seed()
        actual = chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        nonc_chi = random.noncentral_chisquare
        desired = np.array([9.0015599467913763,
                            4.5804135049718742,
                            6.0872302432834564])

        self.set_seed()
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        self.set_seed()
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def test_standard_t(self):
        df = [1]
        bad_df = [-1]
        t = random.standard_t
        desired = np.array([3.0702872575217643,
                            5.8560725167361607,
                            1.0274791436474273])

        self.set_seed()
        actual = t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, t, bad_df * 3)
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        vonmises = random.vonmises
        desired = np.array([2.9883443664201312,
                            -2.7064099483995943,
                            -1.8672476700665914])

        self.set_seed()
        actual = vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu * 3, bad_kappa)

        self.set_seed()
        actual = vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        a = [1]
        bad_a = [-1]
        pareto = random.pareto
        desired = np.array([1.1405622680198362,
                            1.1465519762044529,
                            1.0389564467453547])

        self.set_seed()
        actual = pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, pareto, bad_a * 3)
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def test_weibull(self):
        a = [1]
        bad_a = [-1]
        weibull = random.weibull
        desired = np.array([0.76106853658845242,
                            0.76386282278691653,
                            0.71243813125891797])

        self.set_seed()
        actual = weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, weibull, bad_a * 3)
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def test_power(self):
        a = [1]
        bad_a = [-1]
        power = random.power
        desired = np.array([0.53283302478975902,
                            0.53413660089041659,
                            0.50955303552646702])

        self.set_seed()
        actual = power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, power, bad_a * 3)
        assert_raises(ValueError, random.power, bad_a * 3)

    def test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        laplace = random.laplace
        desired = np.array([0.067921356028507157,
                            0.070715642226971326,
                            0.019290950698972624])

        self.set_seed()
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        self.set_seed()
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        gumbel = random.gumbel
        desired = np.array([0.2730318639556768,
                            0.26936705726291116,
                            0.33906220393037939])

        self.set_seed()
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        self.set_seed()
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        logistic = random.logistic
        desired = np.array([0.13152135837586171,
                            0.13675915696285773,
                            0.038216792802833396])

        self.set_seed()
        actual = logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc * 3, bad_scale)

        self.set_seed()
        actual = logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc, bad_scale * 3)
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        lognormal = random.lognormal
        desired = np.array([9.1422086044848427,
                            8.4013952870126261,
                            6.3073234116578671])

        self.set_seed()
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)
        assert_raises(ValueError, random.lognormal, mean * 3, bad_sigma)

        self.set_seed()
        actual = lognormal(mean, sigma * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        rayleigh = random.rayleigh
        desired = np.array([1.2337491937897689,
                            1.2360119924878694,
                            1.1936818095781789])

        self.set_seed()
        actual = rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, rayleigh, bad_scale * 3)

    def test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        wald = random.wald
        desired = np.array([0.11873681120271318,
                            0.12450084820795027,
                            0.9096122728408238])

        self.set_seed()
        actual = wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        assert_raises(ValueError, wald, mean * 3, bad_scale)
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)

        self.set_seed()
        actual = wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        assert_raises(ValueError, wald, 0.0, 1)
        assert_raises(ValueError, wald, 0.5, 0.0)

    def test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        triangular = random.triangular
        desired = np.array([2.03339048710429,
                            2.0347400359389356,
                            2.0095991069536208])

        self.set_seed()
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        self.set_seed()
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        self.set_seed()
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)

        assert_raises(ValueError, triangular, 10., 0., 20.)
        assert_raises(ValueError, triangular, 10., 25., 20.)
        assert_raises(ValueError, triangular, 10., 10., 10.)

    def test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        binom = random.binomial
        desired = np.array([1, 1, 1])

        self.set_seed()
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        self.set_seed()
        actual = binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)

    def test_negative_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        neg_binom = random.negative_binomial
        desired = np.array([1, 0, 1])

        self.set_seed()
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        self.set_seed()
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def test_poisson(self):
        max_lam = random.RandomState()._poisson_lam_max

        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        desired = np.array([1, 1, 0])

        self.set_seed()
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def test_zipf(self):
        a = [2]
        bad_a = [0]
        zipf = random.zipf
        desired = np.array([2, 2, 1])

        self.set_seed()
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    def test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        geom = random.geometric
        desired = np.array([2, 2, 2])

        self.set_seed()
        actual = geom(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geom, bad_p_one * 3)
        assert_raises(ValueError, geom, bad_p_two * 3)

    def test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [0]
        bad_nsample_two = [4]
        hypergeom = random.hypergeometric
        desired = np.array([1, 1, 1])

        self.set_seed()
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)

        self.set_seed()
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)

        self.set_seed()
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, 0)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

    def test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        logseries = random.logseries
        desired = np.array([1, 1, 1])

        self.set_seed()
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
class TestThread:
    # make sure each state produces the same sequence even in threads
    def setup_method(self):
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # threaded generation
        t = [Thread(target=function, args=(random.RandomState(s), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(random.RandomState(s), o)

        # these platforms change x87 fpu precision mode in threads
        if np.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)

        self.check_function(gen_random, sz=(10000,))

    def test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))

        self.check_function(gen_random, sz=(100, 1000))

    def test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1 / 6.] * 6, size=10000)

        self.check_function(gen_random, sz=(10000, 6))


# See Issue #4263
class TestSingleEltArrayInput:
    def setup_method(self):
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        funcs = (random.exponential, random.standard_gamma,
                 random.chisquare, random.standard_t,
                 random.pareto, random.weibull,
                 random.power, random.rayleigh,
                 random.poisson, random.zipf,
                 random.geometric, random.logseries)

        probfuncs = (random.geometric, random.logseries)

        for func in funcs:
            if func in probfuncs:  # p < 1.0
                out = func(np.array([0.5]))

            else:
                out = func(self.argOne)

            assert_equal(out.shape, self.tgtShape)

    def test_two_arg_funcs(self):
        funcs = (random.uniform, random.normal,
                 random.beta, random.gamma,
                 random.f, random.noncentral_chisquare,
                 random.vonmises, random.laplace,
                 random.gumbel, random.logistic,
                 random.lognormal, random.wald,
                 random.binomial, random.negative_binomial)

        probfuncs = (random.binomial, random.negative_binomial)

        for func in funcs:
            if func in probfuncs:  # p <= 1
                argTwo = np.array([0.5])

            else:
                argTwo = self.argTwo

            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def test_three_arg_funcs(self):
        funcs = [random.noncentral_f, random.triangular,
                 random.hypergeometric]

        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)


# Ensure returned array dtype is correct for platform
def test_integer_dtype(int_func):
    random.seed(123456789)
    fname, args, sha256 = int_func
    f = getattr(random, fname)
    actual = f(*args, size=2)
    assert_(actual.dtype == np.dtype('l'))


def test_integer_repeat(int_func):
    random.seed(123456789)
    fname, args, sha256 = int_func
    f = getattr(random, fname)
    val = f(*args, size=1000000)
    if sys.byteorder != 'little':
        val = val.byteswap()
    res = hashlib.sha256(val.view(np.int8)).hexdigest()
    assert_(res == sha256)


def test_broadcast_size_error():
    # GH-16833
    with pytest.raises(ValueError):
        random.binomial(1, [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], 0.3, size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], [0.3, 0.7], size=(2, 1))


def test_randomstate_ctor_old_style_pickle():
    rs = np.random.RandomState(MT19937(0))
    rs.standard_normal(1)
    # Directly call reduce which is used in pickling
    ctor, args, state_a = rs.__reduce__()
    # Simulate unpickling an old pickle that only has the name
    assert args[0].__class__.__name__ == "MT19937"
    b = ctor(*("MT19937",))
    b.set_state(state_a)
    state_b = b.get_state(legacy=False)

    assert_equal(state_a['bit_generator'], state_b['bit_generator'])
    assert_array_equal(state_a['state']['key'], state_b['state']['key'])
    assert_array_equal(state_a['state']['pos'], state_b['state']['pos'])
    assert_equal(state_a['has_gauss'], state_b['has_gauss'])
    assert_equal(state_a['gauss'], state_b['gauss'])


def test_hot_swap(restore_singleton_bitgen):
    # GH 21808
    def_bg = np.random.default_rng(0)
    bg = def_bg.bit_generator
    np.random.set_bit_generator(bg)
    assert isinstance(np.random.mtrand._rand._bit_generator, type(bg))

    second_bg = np.random.get_bit_generator()
    assert bg is second_bg


def test_seed_alt_bit_gen(restore_singleton_bitgen):
    # GH 21808
    bg = PCG64(0)
    np.random.set_bit_generator(bg)
    state = np.random.get_state(legacy=False)
    np.random.seed(1)
    new_state = np.random.get_state(legacy=False)
    print(state)
    print(new_state)
    assert state["bit_generator"] == "PCG64"
    assert state["state"]["state"] != new_state["state"]["state"]
    assert state["state"]["inc"] != new_state["state"]["inc"]


def test_state_error_alt_bit_gen(restore_singleton_bitgen):
    # GH 21808
    state = np.random.get_state()
    bg = PCG64(0)
    np.random.set_bit_generator(bg)
    with pytest.raises(ValueError, match="state must be for a PCG64"):
        np.random.set_state(state)


def test_swap_worked(restore_singleton_bitgen):
    # GH 21808
    np.random.seed(98765)
    vals = np.random.randint(0, 2 ** 30, 10)
    bg = PCG64(0)
    state = bg.state
    np.random.set_bit_generator(bg)
    state_direct = np.random.get_state(legacy=False)
    for field in state:
        assert state[field] == state_direct[field]
    np.random.seed(98765)
    pcg_vals = np.random.randint(0, 2 ** 30, 10)
    assert not np.all(vals == pcg_vals)
    new_state = bg.state
    assert new_state["state"]["state"] != state["state"]["state"]
    assert new_state["state"]["inc"] == new_state["state"]["inc"]


def test_swapped_singleton_against_direct(restore_singleton_bitgen):
    np.random.set_bit_generator(PCG64(98765))
    singleton_vals = np.random.randint(0, 2 ** 30, 10)
    rg = np.random.RandomState(PCG64(98765))
    non_singleton_vals = rg.randint(0, 2 ** 30, 10)
    assert_equal(non_singleton_vals, singleton_vals)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_reductions.py ===
from datetime import timedelta
from decimal import Decimal
import re

from dateutil.tz import tzlocal
import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    DatetimeIndex,
    Index,
    PeriodIndex,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.core import (
    algorithms,
    nanops,
)

is_windows_np2_or_is32 = (is_platform_windows() and not np_version_gt2) or not IS64
is_windows_or_is32 = is_platform_windows() or not IS64


def make_skipna_wrapper(alternative, skipna_alternative=None):
    """
    Create a function for calling on an array.

    Parameters
    ----------
    alternative : function
        The function to be called on the array with no NaNs.
        Only used when 'skipna_alternative' is None.
    skipna_alternative : function
        The function to be called on the original array

    Returns
    -------
    function
    """
    if skipna_alternative:

        def skipna_wrapper(x):
            return skipna_alternative(x.values)

    else:

        def skipna_wrapper(x):
            nona = x.dropna()
            if len(nona) == 0:
                return np.nan
            return alternative(nona)

    return skipna_wrapper


def assert_stat_op_calc(
    opname,
    alternative,
    frame,
    has_skipna=True,
    check_dtype=True,
    check_dates=False,
    rtol=1e-5,
    atol=1e-8,
    skipna_alternative=None,
):
    """
    Check that operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    check_dtype : bool, default True
        Whether the dtypes of the result of "frame.opname()" and
        "alternative(frame)" should be checked.
    check_dates : bool, default false
        Whether opname should be tested on a Datetime Series
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    skipna_alternative : function, default None
        NaN-safe version of alternative
    """
    f = getattr(frame, opname)

    if check_dates:
        df = DataFrame({"b": date_range("1/1/2001", periods=2)})
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)

        df["a"] = range(len(df))
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)
        assert len(result)

    if has_skipna:

        def wrapper(x):
            return alternative(x.values)

        skipna_wrapper = make_skipna_wrapper(alternative, skipna_alternative)
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(
            result0, frame.apply(wrapper), check_dtype=check_dtype, rtol=rtol, atol=atol
        )
        tm.assert_series_equal(
            result1,
            frame.apply(wrapper, axis=1),
            rtol=rtol,
            atol=atol,
        )
    else:
        skipna_wrapper = alternative

    result0 = f(axis=0)
    result1 = f(axis=1)
    tm.assert_series_equal(
        result0,
        frame.apply(skipna_wrapper),
        check_dtype=check_dtype,
        rtol=rtol,
        atol=atol,
    )

    if opname in ["sum", "prod"]:
        expected = frame.apply(skipna_wrapper, axis=1)
        tm.assert_series_equal(
            result1, expected, check_dtype=False, rtol=rtol, atol=atol
        )

    # check dtypes
    if check_dtype:
        lcd_dtype = frame.values.dtype
        assert lcd_dtype == result0.dtype
        assert lcd_dtype == result1.dtype

    # bad axis
    with pytest.raises(ValueError, match="No axis named 2"):
        f(axis=2)

    # all NA case
    if has_skipna:
        all_na = frame * np.nan
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname in ["sum", "prod"]:
            unit = 1 if opname == "prod" else 0  # result for empty sum/prod
            expected = Series(unit, index=r0.index, dtype=r0.dtype)
            tm.assert_series_equal(r0, expected)
            expected = Series(unit, index=r1.index, dtype=r1.dtype)
            tm.assert_series_equal(r1, expected)


@pytest.fixture
def bool_frame_with_na():
    """
    Fixture for DataFrame of booleans with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.concatenate(
            [np.ones((15, 4), dtype=bool), np.zeros((15, 4), dtype=bool)], axis=0
        ),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
        dtype=object,
    )
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


@pytest.fixture
def float_frame_with_na():
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),
        columns=Index(list("ABCD"), dtype=object),
    )
    # set some NAs
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


class TestDataFrameAnalytics:
    # ---------------------------------------------------------------------
    # Reductions
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "nunique",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    def test_stat_op_api_float_string_frame(self, float_string_frame, axis, opname):
        if (opname in ("sum", "min", "max") and axis == 0) or opname in (
            "count",
            "nunique",
        ):
            getattr(float_string_frame, opname)(axis=axis)
        else:
            if opname in ["var", "std", "sem", "skew", "kurt"]:
                msg = "could not convert string to float: 'bar'"
            elif opname == "product":
                if axis == 1:
                    msg = "can't multiply sequence by non-int of type 'float'"
                else:
                    msg = "can't multiply sequence by non-int of type 'str'"
            elif opname == "sum":
                msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname == "mean":
                if axis == 0:
                    # different message on different builds
                    msg = "|".join(
                        [
                            r"Could not convert \['.*'\] to numeric",
                            "Could not convert string '(bar){30}' to numeric",
                        ]
                    )
                else:
                    msg = r"unsupported operand type\(s\) for \+: 'float' and 'str'"
            elif opname in ["min", "max"]:
                msg = "'[><]=' not supported between instances of 'float' and 'str'"
            elif opname == "median":
                msg = re.compile(
                    r"Cannot convert \[.*\] to numeric|does not support|Cannot perform",
                    flags=re.S,
                )
            if not isinstance(msg, re.Pattern):
                msg = msg + "|does not support|Cannot perform reduction"
            with pytest.raises(TypeError, match=msg):
                getattr(float_string_frame, opname)(axis=axis)
        if opname != "nunique":
            getattr(float_string_frame, opname)(axis=axis, numeric_only=True)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        "opname",
        [
            "count",
            "sum",
            "mean",
            "product",
            "median",
            "min",
            "max",
            "var",
            "std",
            "sem",
            pytest.param("skew", marks=td.skip_if_no("scipy")),
            pytest.param("kurt", marks=td.skip_if_no("scipy")),
        ],
    )
    def test_stat_op_api_float_frame(self, float_frame, axis, opname):
        getattr(float_frame, opname)(axis=axis, numeric_only=False)

    def test_stat_op_calc(self, float_frame_with_na, mixed_float_frame):
        def count(s):
            return notna(s).sum()

        def nunique(s):
            return len(algorithms.unique1d(s.dropna()))

        def var(x):
            return np.var(x, ddof=1)

        def std(x):
            return np.std(x, ddof=1)

        def sem(x):
            return np.std(x, ddof=1) / np.sqrt(len(x))

        assert_stat_op_calc(
            "nunique",
            nunique,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

        # GH#32571: rol needed for flaky CI builds
        # mixed types (with upcasting happening)
        assert_stat_op_calc(
            "sum",
            np.sum,
            mixed_float_frame.astype("float32"),
            check_dtype=False,
            rtol=1e-3,
        )

        assert_stat_op_calc(
            "sum", np.sum, float_frame_with_na, skipna_alternative=np.nansum
        )
        assert_stat_op_calc("mean", np.mean, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "product", np.prod, float_frame_with_na, skipna_alternative=np.nanprod
        )

        assert_stat_op_calc("var", var, float_frame_with_na)
        assert_stat_op_calc("std", std, float_frame_with_na)
        assert_stat_op_calc("sem", sem, float_frame_with_na)

        assert_stat_op_calc(
            "count",
            count,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

    def test_stat_op_calc_skew_kurtosis(self, float_frame_with_na):
        sp_stats = pytest.importorskip("scipy.stats")

        def skewness(x):
            if len(x) < 3:
                return np.nan
            return sp_stats.skew(x, bias=False)

        def kurt(x):
            if len(x) < 4:
                return np.nan
            return sp_stats.kurtosis(x, bias=False)

        assert_stat_op_calc("skew", skewness, float_frame_with_na)
        assert_stat_op_calc("kurt", kurt, float_frame_with_na)

    def test_median(self, float_frame_with_na, int_frame):
        def wrapper(x):
            if isna(x).any():
                return np.nan
            return np.median(x)

        assert_stat_op_calc("median", wrapper, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "median", wrapper, int_frame, check_dtype=False, check_dates=True
        )

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "prod", "var", "std", "skew", "min", "max"]
    )
    @pytest.mark.parametrize(
        "df",
        [
            DataFrame(
                {
                    "a": [
                        -0.00049987540199591344,
                        -0.0016467257772919831,
                        0.00067695870775883013,
                    ],
                    "b": [-0, -0, 0.0],
                    "c": [
                        0.00031111847529610595,
                        0.0014902627951905339,
                        -0.00094099200035979691,
                    ],
                },
                index=["foo", "bar", "baz"],
                dtype="O",
            ),
            DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object),
        ],
    )
    @pytest.mark.filterwarnings("ignore:Mismatched null-like values:FutureWarning")
    def test_stat_operators_attempt_obj_array(self, method, df, axis):
        # GH#676
        assert df.values.dtype == np.object_
        result = getattr(df, method)(axis=axis)
        expected = getattr(df.astype("f8"), method)(axis=axis).astype(object)
        if axis in [1, "columns"] and method in ["min", "max"]:
            expected[expected.isna()] = None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("op", ["mean", "std", "var", "skew", "kurt", "sem"])
    def test_mixed_ops(self, op):
        # GH#16116
        df = DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.0, 2.0, 3.0, 4.0],
                "str": ["a", "b", "c", "d"],
            }
        )
        msg = "|".join(
            [
                "Could not convert",
                "could not convert",
                "can't multiply sequence by non-int",
                "does not support",
                "Cannot perform",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            getattr(df, op)()

        with pd.option_context("use_bottleneck", False):
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)()

    def test_reduce_mixed_frame(self):
        # GH 6806
        df = DataFrame(
            {
                "bool_data": [True, True, False, False, False],
                "int_data": [10, 20, 30, 40, 50],
                "string_data": ["a", "b", "c", "d", "e"],
            }
        )
        df.reindex(columns=["bool_data", "int_data", "string_data"])
        test = df.sum(axis=0)
        tm.assert_numpy_array_equal(
            test.values, np.array([2, 150, "abcde"], dtype=object)
        )
        alt = df.T.sum(axis=1)
        tm.assert_series_equal(test, alt)

    def test_nunique(self):
        df = DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": [1, np.nan, 3]})
        tm.assert_series_equal(df.nunique(), Series({"A": 1, "B": 3, "C": 2}))
        tm.assert_series_equal(
            df.nunique(dropna=False), Series({"A": 1, "B": 3, "C": 3})
        )
        tm.assert_series_equal(df.nunique(axis=1), Series({0: 1, 1: 2, 2: 2}))
        tm.assert_series_equal(
            df.nunique(axis=1, dropna=False), Series({0: 1, 1: 3, 2: 2})
        )

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_mixed_datetime_numeric(self, tz):
        # https://github.com/pandas-dev/pandas/issues/24752
        df = DataFrame({"A": [1, 1], "B": [Timestamp("2000", tz=tz)] * 2})
        result = df.mean()
        expected = Series([1.0, Timestamp("2000", tz=tz)], index=["A", "B"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_includes_datetimes(self, tz):
        # https://github.com/pandas-dev/pandas/issues/24752
        # Behavior in 0.24.0rc1 was buggy.
        # As of 2.0 with numeric_only=None we do *not* drop datetime columns
        df = DataFrame({"A": [Timestamp("2000", tz=tz)] * 2})
        result = df.mean()

        expected = Series([Timestamp("2000", tz=tz)], index=["A"])
        tm.assert_series_equal(result, expected)

    def test_mean_mixed_string_decimal(self):
        # GH 11670
        # possible bug when calculating mean of DataFrame?

        d = [
            {"A": 2, "B": None, "C": Decimal("628.00")},
            {"A": 1, "B": None, "C": Decimal("383.00")},
            {"A": 3, "B": None, "C": Decimal("651.00")},
            {"A": 2, "B": None, "C": Decimal("575.00")},
            {"A": 4, "B": None, "C": Decimal("1114.00")},
            {"A": 1, "B": "TEST", "C": Decimal("241.00")},
            {"A": 2, "B": None, "C": Decimal("572.00")},
            {"A": 4, "B": None, "C": Decimal("609.00")},
            {"A": 3, "B": None, "C": Decimal("820.00")},
            {"A": 5, "B": None, "C": Decimal("1223.00")},
        ]

        df = DataFrame(d)

        with pytest.raises(
            TypeError, match="unsupported operand type|does not support|Cannot perform"
        ):
            df.mean()
        result = df[["A", "C"]].mean()
        expected = Series([2.7, 681.6], index=["A", "C"], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_var_std(self, datetime_frame):
        result = datetime_frame.std(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4))
        tm.assert_almost_equal(result, expected)

        result = datetime_frame.var(ddof=4)
        expected = datetime_frame.apply(lambda x: x.var(ddof=4))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nanvar(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context("use_bottleneck", False):
            result = nanops.nanvar(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize("meth", ["sem", "var", "std"])
    def test_numeric_only_flag(self, meth):
        # GH 9201
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            columns=["foo", "bar", "baz"],
        )
        # Cast to object to avoid implicit cast when setting entry to "100" below
        df1 = df1.astype({"foo": object})
        # set one entry to a number in str format
        df1.loc[0, "foo"] = "100"

        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            columns=["foo", "bar", "baz"],
        )
        # Cast to object to avoid implicit cast when setting entry to "a" below
        df2 = df2.astype({"foo": object})
        # set one entry to a non-number str
        df2.loc[0, "foo"] = "a"

        result = getattr(df1, meth)(axis=1, numeric_only=True)
        expected = getattr(df1[["bar", "baz"]], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        result = getattr(df2, meth)(axis=1, numeric_only=True)
        expected = getattr(df2[["bar", "baz"]], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        # df1 has all numbers, df2 has a letter inside
        msg = r"unsupported operand type\(s\) for -: 'float' and 'str'"
        with pytest.raises(TypeError, match=msg):
            getattr(df1, meth)(axis=1, numeric_only=False)
        msg = "could not convert string to float: 'a'"
        with pytest.raises(TypeError, match=msg):
            getattr(df2, meth)(axis=1, numeric_only=False)

    def test_sem(self, datetime_frame):
        result = datetime_frame.sem(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4) / np.sqrt(len(x)))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nansem(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context("use_bottleneck", False):
            result = nanops.nansem(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize(
        "dropna, expected",
        [
            (
                True,
                {
                    "A": [12],
                    "B": [10.0],
                    "C": [1.0],
                    "D": ["a"],
                    "E": Categorical(["a"], categories=["a"]),
                    "F": DatetimeIndex(["2000-01-02"], dtype="M8[ns]"),
                    "G": to_timedelta(["1 days"]),
                },
            ),
            (
                False,
                {
                    "A": [12],
                    "B": [10.0],
                    "C": [np.nan],
                    "D": Series([np.nan], dtype="str"),
                    "E": Categorical([np.nan], categories=["a"]),
                    "F": DatetimeIndex([pd.NaT], dtype="M8[ns]"),
                    "G": to_timedelta([pd.NaT]),
                },
            ),
            (
                True,
                {
                    "H": [8, 9, np.nan, np.nan],
                    "I": [8, 9, np.nan, np.nan],
                    "J": [1, np.nan, np.nan, np.nan],
                    "K": Categorical(["a", np.nan, np.nan, np.nan], categories=["a"]),
                    "L": DatetimeIndex(
                        ["2000-01-02", "NaT", "NaT", "NaT"], dtype="M8[ns]"
                    ),
                    "M": to_timedelta(["1 days", "nan", "nan", "nan"]),
                    "N": [0, 1, 2, 3],
                },
            ),
            (
                False,
                {
                    "H": [8, 9, np.nan, np.nan],
                    "I": [8, 9, np.nan, np.nan],
                    "J": [1, np.nan, np.nan, np.nan],
                    "K": Categorical([np.nan, "a", np.nan, np.nan], categories=["a"]),
                    "L": DatetimeIndex(
                        ["NaT", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"
                    ),
                    "M": to_timedelta(["nan", "1 days", "nan", "nan"]),
                    "N": [0, 1, 2, 3],
                },
            ),
        ],
    )
    def test_mode_dropna(self, dropna, expected):
        df = DataFrame(
            {
                "A": [12, 12, 19, 11],
                "B": [10, 10, np.nan, 3],
                "C": [1, np.nan, np.nan, np.nan],
                "D": Series([np.nan, np.nan, "a", np.nan], dtype="str"),
                "E": Categorical([np.nan, np.nan, "a", np.nan]),
                "F": DatetimeIndex(["NaT", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"),
                "G": to_timedelta(["1 days", "nan", "nan", "nan"]),
                "H": [8, 8, 9, 9],
                "I": [9, 9, 8, 8],
                "J": [1, 1, np.nan, np.nan],
                "K": Categorical(["a", np.nan, "a", np.nan]),
                "L": DatetimeIndex(
                    ["2000-01-02", "2000-01-02", "NaT", "NaT"], dtype="M8[ns]"
                ),
                "M": to_timedelta(["1 days", "nan", "1 days", "nan"]),
                "N": np.arange(4, dtype="int64"),
            }
        )

        result = df[sorted(expected.keys())].mode(dropna=dropna)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    def test_mode_sort_with_na(self, using_infer_string):
        df = DataFrame({"A": [np.nan, np.nan, "a", "a"]})
        expected = DataFrame({"A": ["a", np.nan]})
        result = df.mode(dropna=False)
        tm.assert_frame_equal(result, expected)

    def test_mode_empty_df(self):
        df = DataFrame([], columns=["a", "b"])
        result = df.mode()
        expected = DataFrame([], columns=["a", "b"], index=Index([], dtype=np.int64))
        tm.assert_frame_equal(result, expected)

    def test_operators_timedelta64(self):
        df = DataFrame(
            {
                "A": date_range("2012-1-1", periods=3, freq="D"),
                "B": date_range("2012-1-2", periods=3, freq="D"),
                "C": Timestamp("20120101") - timedelta(minutes=5, seconds=5),
            }
        )

        diffs = DataFrame({"A": df["A"] - df["C"], "B": df["A"] - df["B"]})

        # min
        result = diffs.min()
        assert result.iloc[0] == diffs.loc[0, "A"]
        assert result.iloc[1] == diffs.loc[0, "B"]

        result = diffs.min(axis=1)
        assert (result == diffs.loc[0, "B"]).all()

        # max
        result = diffs.max()
        assert result.iloc[0] == diffs.loc[2, "A"]
        assert result.iloc[1] == diffs.loc[2, "B"]

        result = diffs.max(axis=1)
        assert (result == diffs["A"]).all()

        # abs
        result = diffs.abs()
        result2 = abs(diffs)
        expected = DataFrame({"A": df["A"] - df["C"], "B": df["B"] - df["A"]})
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # mixed frame
        mixed = diffs.copy()
        mixed["C"] = "foo"
        mixed["D"] = 1
        mixed["E"] = 1.0
        mixed["F"] = Timestamp("20130101")

        # results in an object array
        result = mixed.min()
        expected = Series(
            [
                pd.Timedelta(timedelta(seconds=5 * 60 + 5)),
                pd.Timedelta(timedelta(days=-1)),
                "foo",
                1,
                1.0,
                Timestamp("20130101"),
            ],
            index=mixed.columns,
        )
        tm.assert_series_equal(result, expected)

        # excludes non-numeric
        result = mixed.min(axis=1, numeric_only=True)
        expected = Series([1, 1, 1.0], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        # works when only those columns are selected
        result = mixed[["A", "B"]].min(1)
        expected = Series([timedelta(days=-1)] * 3)
        tm.assert_series_equal(result, expected)

        result = mixed[["A", "B"]].min()
        expected = Series(
            [timedelta(seconds=5 * 60 + 5), timedelta(days=-1)], index=["A", "B"]
        )
        tm.assert_series_equal(result, expected)

        # GH 3106
        df = DataFrame(
            {
                "time": date_range("20130102", periods=5),
                "time2": date_range("20130105", periods=5),
            }
        )
        df["off1"] = df["time2"] - df["time"]
        assert df["off1"].dtype == "timedelta64[ns]"

        df["off2"] = df["time"] - df["time2"]
        df._consolidate_inplace()
        assert df["off1"].dtype == "timedelta64[ns]"
        assert df["off2"].dtype == "timedelta64[ns]"

    def test_std_timedelta64_skipna_false(self):
        # GH#37392
        tdi = pd.timedelta_range("1 Day", periods=10)
        df = DataFrame({"A": tdi, "B": tdi}, copy=True)
        df.iloc[-2, -1] = pd.NaT

        result = df.std(skipna=False)
        expected = Series(
            [df["A"].std(), pd.NaT], index=["A", "B"], dtype="timedelta64[ns]"
        )
        tm.assert_series_equal(result, expected)

        result = df.std(axis=1, skipna=False)
        expected = Series([pd.Timedelta(0)] * 8 + [pd.NaT, pd.Timedelta(0)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values", [["2022-01-01", "2022-01-02", pd.NaT, "2022-01-03"], 4 * [pd.NaT]]
    )
    def test_std_datetime64_with_nat(
        self, values, skipna, using_array_manager, request, unit
    ):
        # GH#51335
        if using_array_manager and (
            not skipna or all(value is pd.NaT for value in values)
        ):
            mark = pytest.mark.xfail(
                reason="GH#51446: Incorrect type inference on NaT in reduction result"
            )
            request.applymarker(mark)
        dti = to_datetime(values).as_unit(unit)
        df = DataFrame({"a": dti})
        result = df.std(skipna=skipna)
        if not skipna or all(value is pd.NaT for value in values):
            expected = Series({"a": pd.NaT}, dtype=f"timedelta64[{unit}]")
        else:
            # 86400000000000ns == 1 day
            expected = Series({"a": 86400000000000}, dtype=f"timedelta64[{unit}]")
        tm.assert_series_equal(result, expected)

    def test_sum_corner(self):
        empty_frame = DataFrame()

        axis0 = empty_frame.sum(0)
        axis1 = empty_frame.sum(1)
        assert isinstance(axis0, Series)
        assert isinstance(axis1, Series)
        assert len(axis0) == 0
        assert len(axis1) == 0

    @pytest.mark.parametrize(
        "index",
        [
            RangeIndex(0),
            DatetimeIndex([]),
            Index([], dtype=np.int64),
            Index([], dtype=np.float64),
            DatetimeIndex([], freq="ME"),
            PeriodIndex([], freq="D"),
        ],
    )
    def test_axis_1_empty(self, all_reductions, index):
        df = DataFrame(columns=["a"], index=index)
        result = getattr(df, all_reductions)(axis=1)
        if all_reductions in ("any", "all"):
            expected_dtype = "bool"
        elif all_reductions == "count":
            expected_dtype = "int64"
        else:
            expected_dtype = "object"
        expected = Series([], index=index, dtype=expected_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method, unit", [("sum", 0), ("prod", 1)])
    @pytest.mark.parametrize("numeric_only", [None, True, False])
    def test_sum_prod_nanops(self, method, unit, numeric_only):
        idx = ["a", "b", "c"]
        df = DataFrame({"a": [unit, unit], "b": [unit, np.nan], "c": [np.nan, np.nan]})
        # The default
        result = getattr(df, method)(numeric_only=numeric_only)
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = getattr(df, method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, unit, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = getattr(df, method)(numeric_only=numeric_only, min_count=0)
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        tm.assert_series_equal(result, expected)

        result = getattr(df.iloc[1:], method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count > 1
        df = DataFrame({"A": [unit] * 10, "B": [unit] * 5 + [np.nan] * 5})
        result = getattr(df, method)(numeric_only=numeric_only, min_count=5)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

        result = getattr(df, method)(numeric_only=numeric_only, min_count=6)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_timedelta(self):
        # prod isn't defined on timedeltas
        idx = ["a", "b", "c"]
        df = DataFrame({"a": [0, 0], "b": [0, np.nan], "c": [np.nan, np.nan]})

        df2 = df.apply(to_timedelta)

        # 0 by default
        result = df2.sum()
        expected = Series([0, 0, 0], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = df2.sum(min_count=0)
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = df2.sum(min_count=1)
        expected = Series([0, 0, np.nan], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_min_count(self):
        # https://github.com/pandas-dev/pandas/issues/39738
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = df.sum(min_count=10)
        expected = Series([np.nan, np.nan], index=["x", "y"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("float_type", ["float16", "float32", "float64"])
    @pytest.mark.parametrize(
        "kwargs, expected_result",
        [
            ({"axis": 1, "min_count": 2}, [3.2, 5.3, np.nan]),
            ({"axis": 1, "min_count": 3}, [np.nan, np.nan, np.nan]),
            ({"axis": 1, "skipna": False}, [3.2, 5.3, np.nan]),
        ],
    )
    def test_sum_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        # GH#46947
        df = DataFrame({"a": [1.0, 2.3, 4.4], "b": [2.2, 3, np.nan]}, dtype=float_type)
        result = df.sum(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("float_type", ["float16", "float32", "float64"])
    @pytest.mark.parametrize(
        "kwargs, expected_result",
        [
            ({"axis": 1, "min_count": 2}, [2.0, 4.0, np.nan]),
            ({"axis": 1, "min_count": 3}, [np.nan, np.nan, np.nan]),
            ({"axis": 1, "skipna": False}, [2.0, 4.0, np.nan]),
        ],
    )
    def test_prod_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
        # GH#46947
        df = DataFrame(
            {"a": [1.0, 2.0, 4.4], "b": [2.0, 2.0, np.nan]}, dtype=float_type
        )
        result = df.prod(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    def test_sum_object(self, float_frame):
        values = float_frame.values.astype(int)
        frame = DataFrame(values, index=float_frame.index, columns=float_frame.columns)
        deltas = frame * timedelta(1)
        deltas.sum()

    def test_sum_bool(self, float_frame):
        # ensure this works, bug report
        bools = np.isnan(float_frame)
        bools.sum(1)
        bools.sum(0)

    def test_sum_mixed_datetime(self):
        # GH#30886
        df = DataFrame({"A": date_range("2000", periods=4), "B": [1, 2, 3, 4]}).reindex(
            [2, 3, 4]
        )
        with pytest.raises(TypeError, match="does not support reduction 'sum'"):
            df.sum()

    def test_mean_corner(self, float_frame, float_string_frame):
        # unit test when have object data
        msg = "Could not convert|does not support|Cannot perform"
        with pytest.raises(TypeError, match=msg):
            float_string_frame.mean(axis=0)

        # xs sum mixed type, just want to know it works...
        with pytest.raises(TypeError, match="unsupported operand type"):
            float_string_frame.mean(axis=1)

        # take mean of boolean column
        float_frame["bool"] = float_frame["A"] > 0
        means = float_frame.mean(0)
        assert means["bool"] == float_frame["bool"].values.mean()

    def test_mean_datetimelike(self):
        # GH#24757 check that datetimelike are excluded by default, handled
        #  correctly with numeric_only=True
        #  As of 2.0, datetimelike are *not* excluded with numeric_only=None

        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
                "D": pd.period_range("2016", periods=3, freq="Y"),
            }
        )
        result = df.mean(numeric_only=True)
        expected = Series({"A": 1.0})
        tm.assert_series_equal(result, expected)

        with pytest.raises(TypeError, match="mean is not implemented for PeriodArray"):
            df.mean()

    def test_mean_datetimelike_numeric_only_false(self):
        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
            }
        )

        # datetime(tz) and timedelta work
        result = df.mean(numeric_only=False)
        expected = Series({"A": 1, "B": df.loc[1, "B"], "C": df.loc[1, "C"]})
        tm.assert_series_equal(result, expected)

        # mean of period is not allowed
        df["D"] = pd.period_range("2016", periods=3, freq="Y")

        with pytest.raises(TypeError, match="mean is not implemented for Period"):
            df.mean(numeric_only=False)

    def test_mean_extensionarray_numeric_only_true(self):
        # https://github.com/pandas-dev/pandas/issues/33256
        arr = np.random.default_rng(2).integers(1000, size=(10, 5))
        df = DataFrame(arr, dtype="Int64")
        result = df.mean(numeric_only=True)
        expected = DataFrame(arr).mean().astype("Float64")
        tm.assert_series_equal(result, expected)

    def test_stats_mixed_type(self, float_string_frame):
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.std(1)
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.var(1)
        with pytest.raises(TypeError, match="unsupported operand type"):
            float_string_frame.mean(1)
        with pytest.raises(TypeError, match="could not convert"):
            float_string_frame.skew(1)

    def test_sum_bools(self):
        df = DataFrame(index=range(1), columns=range(10))
        bools = isna(df)
        assert bools.sum(axis=1)[0] == 10

    # ----------------------------------------------------------------------
    # Index of max / min

    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_idxmin(self, float_frame, int_frame, skipna, axis):
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            warn = None
            if skipna is False or axis == 1:
                warn = None if df is int_frame else FutureWarning
            msg = "The behavior of DataFrame.idxmin with all-NA values"
            with tm.assert_produces_warning(warn, match=msg):
                result = df.idxmin(axis=axis, skipna=skipna)

            msg2 = "The behavior of Series.idxmin"
            with tm.assert_produces_warning(warn, match=msg2):
                expected = df.apply(Series.idxmin, axis=axis, skipna=skipna)
            expected = expected.astype(df.index.dtype)
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_idxmin_empty(self, index, skipna, axis):
        # GH53265
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)

        result = frame.idxmin(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_idxmin_numeric_only(self, numeric_only):
        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1], "c": list("xyx")})
        result = df.idxmin(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([2, 1], index=["a", "b"])
        else:
            expected = Series([2, 1, 0], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

    def test_idxmin_axis_2(self, float_frame):
        frame = float_frame
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.idxmin(axis=2)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_idxmax(self, float_frame, int_frame, skipna, axis):
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            warn = None
            if skipna is False or axis == 1:
                warn = None if df is int_frame else FutureWarning
            msg = "The behavior of DataFrame.idxmax with all-NA values"
            with tm.assert_produces_warning(warn, match=msg):
                result = df.idxmax(axis=axis, skipna=skipna)

            msg2 = "The behavior of Series.idxmax"
            with tm.assert_produces_warning(warn, match=msg2):
                expected = df.apply(Series.idxmax, axis=axis, skipna=skipna)
            expected = expected.astype(df.index.dtype)
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_idxmax_empty(self, index, skipna, axis):
        # GH53265
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)

        result = frame.idxmax(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_idxmax_numeric_only(self, numeric_only):
        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1], "c": list("xyx")})
        result = df.idxmax(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([1, 0], index=["a", "b"])
        else:
            expected = Series([1, 0, 1], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

    def test_idxmax_arrow_types(self):
        # GH#55368
        pytest.importorskip("pyarrow")

        df = DataFrame({"a": [2, 3, 1], "b": [2, 1, 1]}, dtype="int64[pyarrow]")
        result = df.idxmax()
        expected = Series([1, 0], index=["a", "b"])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([2, 1], index=["a", "b"])
        tm.assert_series_equal(result, expected)

        df = DataFrame({"a": ["b", "c", "a"]}, dtype="string[pyarrow]")
        result = df.idxmax(numeric_only=False)
        expected = Series([1], index=["a"])
        tm.assert_series_equal(result, expected)

        result = df.idxmin(numeric_only=False)
        expected = Series([2], index=["a"])
        tm.assert_series_equal(result, expected)

    def test_idxmax_axis_2(self, float_frame):
        frame = float_frame
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.idxmax(axis=2)

    def test_idxmax_mixed_dtype(self):
        # don't cast to object, which would raise in nanops
        dti = date_range("2016-01-01", periods=3)

        # Copying dti is needed for ArrayManager otherwise when we set
        #  df.loc[0, 3] = pd.NaT below it edits dti
        df = DataFrame({1: [0, 2, 1], 2: range(3)[::-1], 3: dti.copy(deep=True)})

        result = df.idxmax()
        expected = Series([1, 0, 2], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 0], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        # with NaTs
        df.loc[0, 3] = pd.NaT
        result = df.idxmax()
        expected = Series([1, 0, 2], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 1], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        # with multi-column dt64 block
        df[4] = dti[::-1]
        df._consolidate_inplace()

        result = df.idxmax()
        expected = Series([1, 0, 2, 0], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 1, 2], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op, expected_value",
        [("idxmax", [0, 4]), ("idxmin", [0, 5])],
    )
    def test_idxmax_idxmin_convert_dtypes(self, op, expected_value):
        # GH 40346
        df = DataFrame(
            {
                "ID": [100, 100, 100, 200, 200, 200],
                "value": [0, 0, 0, 1, 2, 0],
            },
            dtype="Int64",
        )
        df = df.groupby("ID")

        result = getattr(df, op)()
        expected = DataFrame(
            {"value": expected_value},
            index=Index([100, 200], name="ID", dtype="Int64"),
        )
        tm.assert_frame_equal(result, expected)

    def test_idxmax_dt64_multicolumn_axis1(self):
        dti = date_range("2016-01-01", periods=3)
        df = DataFrame({3: dti, 4: dti[::-1]}, copy=True)
        df.iloc[0, 0] = pd.NaT

        df._consolidate_inplace()

        result = df.idxmax(axis=1)
        expected = Series([4, 3, 3])
        tm.assert_series_equal(result, expected)

        result = df.idxmin(axis=1)
        expected = Series([4, 3, 4])
        tm.assert_series_equal(result, expected)

    # ----------------------------------------------------------------------
    # Logical reductions

    @pytest.mark.parametrize("opname", ["any", "all"])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("bool_only", [False, True])
    def test_any_all_mixed_float(self, opname, axis, bool_only, float_string_frame):
        # make sure op works on mixed-type frame
        mixed = float_string_frame
        mixed["_bool_"] = np.random.default_rng(2).standard_normal(len(mixed)) > 0.5

        getattr(mixed, opname)(axis=axis, bool_only=bool_only)

    @pytest.mark.parametrize("opname", ["any", "all"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_any_all_bool_with_na(self, opname, axis, bool_frame_with_na):
        getattr(bool_frame_with_na, opname)(axis=axis, bool_only=False)

    @pytest.mark.filterwarnings("ignore:Downcasting object dtype arrays:FutureWarning")
    @pytest.mark.parametrize("opname", ["any", "all"])
    def test_any_all_bool_frame(self, opname, bool_frame_with_na):
        # GH#12863: numpy gives back non-boolean data for object type
        # so fill NaNs to compare with pandas behavior
        frame = bool_frame_with_na.fillna(True)
        alternative = getattr(np, opname)
        f = getattr(frame, opname)

        def skipna_wrapper(x):
            nona = x.dropna().values
            return alternative(nona)

        def wrapper(x):
            return alternative(x.values)

        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)

        tm.assert_series_equal(result0, frame.apply(wrapper))
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1))

        result0 = f(axis=0)
        result1 = f(axis=1)

        tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
        tm.assert_series_equal(
            result1, frame.apply(skipna_wrapper, axis=1), check_dtype=False
        )

        # bad axis
        with pytest.raises(ValueError, match="No axis named 2"):
            f(axis=2)

        # all NA case
        all_na = frame * np.nan
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname == "any":
            assert not r0.any()
            assert not r1.any()
        else:
            assert r0.all()
            assert r1.all()

    def test_any_all_extra(self):
        df = DataFrame(
            {
                "A": [True, False, False],
                "B": [True, True, False],
                "C": [True, True, True],
            },
            index=["a", "b", "c"],
        )
        result = df[["A", "B"]].any(axis=1)
        expected = Series([True, True, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        result = df[["A", "B"]].any(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)

        result = df.all(1)
        expected = Series([True, False, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        result = df.all(1, bool_only=True)
        tm.assert_series_equal(result, expected)

        # Axis is None
        result = df.all(axis=None).item()
        assert result is False

        result = df.any(axis=None).item()
        assert result is True

        result = df[["C"]].all(axis=None).item()
        assert result is True

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("bool_agg_func", ["any", "all"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_any_all_object_dtype(self, axis, bool_agg_func, skipna):
        # GH#35450
        df = DataFrame(
            data=[
                [1, np.nan, np.nan, True],
                [np.nan, 2, np.nan, True],
                [np.nan, np.nan, np.nan, True],
                [np.nan, np.nan, "5", np.nan],
            ]
        )
        result = getattr(df, bool_agg_func)(axis=axis, skipna=skipna)
        expected = Series([True, True, True, True])
        tm.assert_series_equal(result, expected)

    # GH#50947 deprecates this but it is not emitting a warning in some builds.
    @pytest.mark.filterwarnings(
        "ignore:'any' with datetime64 dtypes is deprecated.*:FutureWarning"
    )
    def test_any_datetime(self):
        # GH 23070
        float_data = [1, np.nan, 3, np.nan]
        datetime_data = [
            Timestamp("1960-02-15"),
            Timestamp("1960-02-16"),
            pd.NaT,
            pd.NaT,
        ]
        df = DataFrame({"A": float_data, "B": datetime_data})

        result = df.any(axis=1)

        expected = Series([True, True, True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_bool_only(self):
        # GH 25101
        df = DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [None, None, None]},
            columns=Index(["col1", "col2", "col3"], dtype=object),
        )

        result = df.all(bool_only=True)
        expected = Series(dtype=np.bool_, index=[])
        tm.assert_series_equal(result, expected)

        df = DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [None, None, None],
                "col4": [False, False, True],
            }
        )

        result = df.all(bool_only=True)
        expected = Series({"col4": False})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func, data, expected",
        [
            (np.any, {}, False),
            (np.all, {}, True),
            (np.any, {"A": []}, False),
            (np.all, {"A": []}, True),
            (np.any, {"A": [False, False]}, False),
            (np.all, {"A": [False, False]}, False),
            (np.any, {"A": [True, False]}, True),
            (np.all, {"A": [True, False]}, False),
            (np.any, {"A": [True, True]}, True),
            (np.all, {"A": [True, True]}, True),
            (np.any, {"A": [False], "B": [False]}, False),
            (np.all, {"A": [False], "B": [False]}, False),
            (np.any, {"A": [False, False], "B": [False, True]}, True),
            (np.all, {"A": [False, False], "B": [False, True]}, False),
            # other types
            (np.all, {"A": Series([0.0, 1.0], dtype="float")}, False),
            (np.any, {"A": Series([0.0, 1.0], dtype="float")}, True),
            (np.all, {"A": Series([0, 1], dtype=int)}, False),
            (np.any, {"A": Series([0, 1], dtype=int)}, True),
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns]")}, False),
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, False),
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns]")}, True),
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.all, {"A": Series([0, 1], dtype="m8[ns]")}, False),
            pytest.param(np.any, {"A": Series([0, 1], dtype="m8[ns]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            # np.all on Categorical raises, so the reduction drops the
            #  column, so all is being done on an empty Series, so is True
            (np.all, {"A": Series([0, 1], dtype="category")}, True),
            (np.any, {"A": Series([0, 1], dtype="category")}, False),
            (np.all, {"A": Series([1, 2], dtype="category")}, True),
            (np.any, {"A": Series([1, 2], dtype="category")}, False),
            # Mix GH#21484
            pytest.param(
                np.all,
                {
                    "A": Series([10, 20], dtype="M8[ns]"),
                    "B": Series([10, 20], dtype="m8[ns]"),
                },
                True,
            ),
        ],
    )
    def test_any_all_np_func(self, func, data, expected):
        # GH 19976
        data = DataFrame(data)

        if any(isinstance(x, CategoricalDtype) for x in data.dtypes):
            with pytest.raises(
                TypeError, match="dtype category does not support reduction"
            ):
                func(data)

            # method version
            with pytest.raises(
                TypeError, match="dtype category does not support reduction"
            ):
                getattr(DataFrame(data), func.__name__)(axis=None)
        else:
            msg = "'(any|all)' with datetime64 dtypes is deprecated"
            if data.dtypes.apply(lambda x: x.kind == "M").any():
                warn = FutureWarning
            else:
                warn = None

            with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
                # GH#34479
                result = func(data)
            assert isinstance(result, np.bool_)
            assert result.item() is expected

            # method version
            with tm.assert_produces_warning(warn, match=msg):
                # GH#34479
                result = getattr(DataFrame(data), func.__name__)(axis=None)
            assert isinstance(result, np.bool_)
            assert result.item() is expected

    def test_any_all_object(self):
        # GH 19976
        result = np.all(DataFrame(columns=["a", "b"])).item()
        assert result is True

        result = np.any(DataFrame(columns=["a", "b"])).item()
        assert result is False

    def test_any_all_object_bool_only(self):
        df = DataFrame({"A": ["foo", 2], "B": [True, False]}).astype(object)
        df._consolidate_inplace()
        df["C"] = Series([True, True])

        # Categorical of bools is _not_ considered booly
        df["D"] = df["C"].astype("category")

        # The underlying bug is in DataFrame._get_bool_data, so we check
        #  that while we're here
        res = df._get_bool_data()
        expected = df[["C"]]
        tm.assert_frame_equal(res, expected)

        res = df.all(bool_only=True, axis=0)
        expected = Series([True], index=["C"])
        tm.assert_series_equal(res, expected)

        # operating on a subset of columns should not produce a _larger_ Series
        res = df[["B", "C"]].all(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)

        assert df.all(bool_only=True, axis=None)

        res = df.any(bool_only=True, axis=0)
        expected = Series([True], index=["C"])
        tm.assert_series_equal(res, expected)

        # operating on a subset of columns should not produce a _larger_ Series
        res = df[["C"]].any(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)

        assert df.any(bool_only=True, axis=None)

    # ---------------------------------------------------------------------
    # Unsorted

    def test_series_broadcasting(self):
        # smoke test for numpy warnings
        # GH 16378, GH 16306
        df = DataFrame([1.0, 1.0, 1.0])
        df_nan = DataFrame({"A": [np.nan, 2.0, np.nan]})
        s = Series([1, 1, 1])
        s_nan = Series([np.nan, np.nan, 1])

        with tm.assert_produces_warning(None):
            df_nan.clip(lower=s, axis=0)
            for op in ["lt", "le", "gt", "ge", "eq", "ne"]:
                getattr(df, op)(s_nan, axis=0)


class TestDataFrameReductions:
    def test_min_max_dt64_with_NaT(self):
        # Both NaT and Timestamp are in DataFrame.
        df = DataFrame({"foo": [pd.NaT, pd.NaT, Timestamp("2012-05-01")]})

        res = df.min()
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)

        res = df.max()
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)

        # GH12941, only NaTs are in DataFrame.
        df = DataFrame({"foo": [pd.NaT, pd.NaT]})

        res = df.min()
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)

        res = df.max()
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)

    def test_min_max_dt64_with_NaT_skipna_false(self, request, tz_naive_fixture):
        # GH#36907
        tz = tz_naive_fixture
        if isinstance(tz, tzlocal) and is_platform_windows():
            pytest.skip(
                "GH#37659 OSError raised within tzlocal bc Windows "
                "chokes in times before 1970-01-01"
            )

        df = DataFrame(
            {
                "a": [
                    Timestamp("2020-01-01 08:00:00", tz=tz),
                    Timestamp("1920-02-01 09:00:00", tz=tz),
                ],
                "b": [Timestamp("2020-02-01 08:00:00", tz=tz), pd.NaT],
            }
        )
        res = df.min(axis=1, skipna=False)
        expected = Series([df.loc[0, "a"], pd.NaT])
        assert expected.dtype == df["a"].dtype

        tm.assert_series_equal(res, expected)

        res = df.max(axis=1, skipna=False)
        expected = Series([df.loc[0, "b"], pd.NaT])
        assert expected.dtype == df["a"].dtype

        tm.assert_series_equal(res, expected)

    def test_min_max_dt64_api_consistency_with_NaT(self):
        # Calling the following sum functions returned an error for dataframes but
        # returned NaT for series. These tests check that the API is consistent in
        # min/max calls on empty Series/DataFrames. See GH:33704 for more
        # information
        df = DataFrame({"x": to_datetime([])})
        expected_dt_series = Series(to_datetime([]))
        # check axis 0
        assert (df.min(axis=0).x is pd.NaT) == (expected_dt_series.min() is pd.NaT)
        assert (df.max(axis=0).x is pd.NaT) == (expected_dt_series.max() is pd.NaT)

        # check axis 1
        tm.assert_series_equal(df.min(axis=1), expected_dt_series)
        tm.assert_series_equal(df.max(axis=1), expected_dt_series)

    def test_min_max_dt64_api_consistency_empty_df(self):
        # check DataFrame/Series api consistency when calling min/max on an empty
        # DataFrame/Series.
        df = DataFrame({"x": []})
        expected_float_series = Series([], dtype=float)
        # check axis 0
        assert np.isnan(df.min(axis=0).x) == np.isnan(expected_float_series.min())
        assert np.isnan(df.max(axis=0).x) == np.isnan(expected_float_series.max())
        # check axis 1
        tm.assert_series_equal(df.min(axis=1), expected_float_series)
        tm.assert_series_equal(df.min(axis=1), expected_float_series)

    @pytest.mark.parametrize(
        "initial",
        ["2018-10-08 13:36:45+00:00", "2018-10-08 13:36:45+03:00"],  # Non-UTC timezone
    )
    @pytest.mark.parametrize("method", ["min", "max"])
    def test_preserve_timezone(self, initial: str, method):
        # GH 28552
        initial_dt = to_datetime(initial)
        expected = Series([initial_dt])
        df = DataFrame([expected])
        result = getattr(df, method)(axis=1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_minmax_tzaware_skipna_axis_1(self, method, skipna):
        # GH#51242
        val = to_datetime("1900-01-01", utc=True)
        df = DataFrame(
            {"a": Series([pd.NaT, pd.NaT, val]), "b": Series([pd.NaT, val, val])}
        )
        op = getattr(df, method)
        result = op(axis=1, skipna=skipna)
        if skipna:
            expected = Series([pd.NaT, val, val])
        else:
            expected = Series([pd.NaT, pd.NaT, val])
        tm.assert_series_equal(result, expected)

    def test_frame_any_with_timedelta(self):
        # GH#17667
        df = DataFrame(
            {
                "a": Series([0, 0]),
                "t": Series([to_timedelta(0, "s"), to_timedelta(1, "ms")]),
            }
        )

        result = df.any(axis=0)
        expected = Series(data=[False, True], index=["a", "t"])
        tm.assert_series_equal(result, expected)

        result = df.any(axis=1)
        expected = Series(data=[False, True])
        tm.assert_series_equal(result, expected)

    def test_reductions_skipna_none_raises(
        self, request, frame_or_series, all_reductions
    ):
        if all_reductions == "count":
            request.applymarker(
                pytest.mark.xfail(reason="Count does not accept skipna")
            )
        obj = frame_or_series([1, 2, 3])
        msg = 'For argument "skipna" expected type bool, received type NoneType.'
        with pytest.raises(ValueError, match=msg):
            getattr(obj, all_reductions)(skipna=None)

    @td.skip_array_manager_invalid_test
    def test_reduction_timestamp_smallest_unit(self):
        # GH#52524
        df = DataFrame(
            {
                "a": Series([Timestamp("2019-12-31")], dtype="datetime64[s]"),
                "b": Series(
                    [Timestamp("2019-12-31 00:00:00.123")], dtype="datetime64[ms]"
                ),
            }
        )
        result = df.max()
        expected = Series(
            [Timestamp("2019-12-31"), Timestamp("2019-12-31 00:00:00.123")],
            dtype="datetime64[ms]",
            index=["a", "b"],
        )
        tm.assert_series_equal(result, expected)

    @td.skip_array_manager_not_yet_implemented
    def test_reduction_timedelta_smallest_unit(self):
        # GH#52524
        df = DataFrame(
            {
                "a": Series([pd.Timedelta("1 days")], dtype="timedelta64[s]"),
                "b": Series([pd.Timedelta("1 days")], dtype="timedelta64[ms]"),
            }
        )
        result = df.max()
        expected = Series(
            [pd.Timedelta("1 days"), pd.Timedelta("1 days")],
            dtype="timedelta64[ms]",
            index=["a", "b"],
        )
        tm.assert_series_equal(result, expected)


class TestNuisanceColumns:
    @pytest.mark.parametrize("method", ["any", "all"])
    def test_any_all_categorical_dtype_nuisance_column(self, method):
        # GH#36076 DataFrame should match Series behavior
        ser = Series([0, 1], dtype="category", name="A")
        df = ser.to_frame()

        # Double-check the Series behavior is to raise
        with pytest.raises(TypeError, match="does not support reduction"):
            getattr(ser, method)()

        with pytest.raises(TypeError, match="does not support reduction"):
            getattr(np, method)(ser)

        with pytest.raises(TypeError, match="does not support reduction"):
            getattr(df, method)(bool_only=False)

        with pytest.raises(TypeError, match="does not support reduction"):
            getattr(df, method)(bool_only=None)

        with pytest.raises(TypeError, match="does not support reduction"):
            getattr(np, method)(df, axis=0)

    def test_median_categorical_dtype_nuisance_column(self):
        # GH#21020 DataFrame.median should match Series.median
        df = DataFrame({"A": Categorical([1, 2, 2, 2, 3])})
        ser = df["A"]

        # Double-check the Series behavior is to raise
        with pytest.raises(TypeError, match="does not support reduction"):
            ser.median()

        with pytest.raises(TypeError, match="does not support reduction"):
            df.median(numeric_only=False)

        with pytest.raises(TypeError, match="does not support reduction"):
            df.median()

        # same thing, but with an additional non-categorical column
        df["B"] = df["A"].astype(int)

        with pytest.raises(TypeError, match="does not support reduction"):
            df.median(numeric_only=False)

        with pytest.raises(TypeError, match="does not support reduction"):
            df.median()

        # TODO: np.median(df, axis=0) gives np.array([2.0, 2.0]) instead
        #  of expected.values

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_min_max_categorical_dtype_non_ordered_nuisance_column(self, method):
        # GH#28949 DataFrame.min should behave like Series.min
        cat = Categorical(["a", "b", "c", "b"], ordered=False)
        ser = Series(cat)
        df = ser.to_frame("A")

        # Double-check the Series behavior
        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(ser, method)()

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(np, method)(ser)

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(df, method)(numeric_only=False)

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(df, method)()

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(np, method)(df, axis=0)

        # same thing, but with an additional non-categorical column
        df["B"] = df["A"].astype(object)
        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(df, method)()

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(np, method)(df, axis=0)


class TestEmptyDataFrameReductions:
    @pytest.mark.parametrize(
        "opname, dtype, exp_value, exp_dtype",
        [
            ("sum", np.int8, 0, np.int64),
            ("prod", np.int8, 1, np.int_),
            ("sum", np.int64, 0, np.int64),
            ("prod", np.int64, 1, np.int64),
            ("sum", np.uint8, 0, np.uint64),
            ("prod", np.uint8, 1, np.uint),
            ("sum", np.uint64, 0, np.uint64),
            ("prod", np.uint64, 1, np.uint64),
            ("sum", np.float32, 0, np.float32),
            ("prod", np.float32, 1, np.float32),
            ("sum", np.float64, 0, np.float64),
        ],
    )
    def test_df_empty_min_count_0(self, opname, dtype, exp_value, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)

        expected = Series([exp_value, exp_value], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "opname, dtype, exp_dtype",
        [
            ("sum", np.int8, np.float64),
            ("prod", np.int8, np.float64),
            ("sum", np.int64, np.float64),
            ("prod", np.int64, np.float64),
            ("sum", np.uint8, np.float64),
            ("prod", np.uint8, np.float64),
            ("sum", np.uint64, np.float64),
            ("prod", np.uint64, np.float64),
            ("sum", np.float32, np.float32),
            ("prod", np.float32, np.float32),
            ("sum", np.float64, np.float64),
        ],
    )
    def test_df_empty_min_count_1(self, opname, dtype, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)

        expected = Series([np.nan, np.nan], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "opname, dtype, exp_value, exp_dtype",
        [
            ("sum", "Int8", 0, ("Int32" if is_windows_np2_or_is32 else "Int64")),
            ("prod", "Int8", 1, ("Int32" if is_windows_np2_or_is32 else "Int64")),
            ("prod", "Int8", 1, ("Int32" if is_windows_np2_or_is32 else "Int64")),
            ("sum", "Int64", 0, "Int64"),
            ("prod", "Int64", 1, "Int64"),
            ("sum", "UInt8", 0, ("UInt32" if is_windows_np2_or_is32 else "UInt64")),
            ("prod", "UInt8", 1, ("UInt32" if is_windows_np2_or_is32 else "UInt64")),
            ("sum", "UInt64", 0, "UInt64"),
            ("prod", "UInt64", 1, "UInt64"),
            ("sum", "Float32", 0, "Float32"),
            ("prod", "Float32", 1, "Float32"),
            ("sum", "Float64", 0, "Float64"),
        ],
    )
    def test_df_empty_nullable_min_count_0(self, opname, dtype, exp_value, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)

        expected = Series([exp_value, exp_value], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

    # TODO: why does min_count=1 impact the resulting Windows dtype
    # differently than min_count=0?
    @pytest.mark.parametrize(
        "opname, dtype, exp_dtype",
        [
            ("sum", "Int8", ("Int32" if is_windows_or_is32 else "Int64")),
            ("prod", "Int8", ("Int32" if is_windows_or_is32 else "Int64")),
            ("sum", "Int64", "Int64"),
            ("prod", "Int64", "Int64"),
            ("sum", "UInt8", ("UInt32" if is_windows_or_is32 else "UInt64")),
            ("prod", "UInt8", ("UInt32" if is_windows_or_is32 else "UInt64")),
            ("sum", "UInt64", "UInt64"),
            ("prod", "UInt64", "UInt64"),
            ("sum", "Float32", "Float32"),
            ("prod", "Float32", "Float32"),
            ("sum", "Float64", "Float64"),
        ],
    )
    def test_df_empty_nullable_min_count_1(self, opname, dtype, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)

        expected = Series([pd.NA, pd.NA], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)


def test_sum_timedelta64_skipna_false(using_array_manager, request):
    # GH#17235
    if using_array_manager:
        mark = pytest.mark.xfail(
            reason="Incorrect type inference on NaT in reduction result"
        )
        request.applymarker(mark)

    arr = np.arange(8).astype(np.int64).view("m8[s]").reshape(4, 2)
    arr[-1, -1] = "Nat"

    df = DataFrame(arr)
    assert (df.dtypes == arr.dtype).all()

    result = df.sum(skipna=False)
    expected = Series([pd.Timedelta(seconds=12), pd.NaT], dtype="m8[s]")
    tm.assert_series_equal(result, expected)

    result = df.sum(axis=0, skipna=False)
    tm.assert_series_equal(result, expected)

    result = df.sum(axis=1, skipna=False)
    expected = Series(
        [
            pd.Timedelta(seconds=1),
            pd.Timedelta(seconds=5),
            pd.Timedelta(seconds=9),
            pd.NaT,
        ],
        dtype="m8[s]",
    )
    tm.assert_series_equal(result, expected)


def test_mixed_frame_with_integer_sum():
    # https://github.com/pandas-dev/pandas/issues/34520
    df = DataFrame([["a", 1]], columns=list("ab"))
    df = df.astype({"b": "Int64"})
    result = df.sum()
    expected = Series(["a", 1], index=["a", "b"])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False, None])
@pytest.mark.parametrize("method", ["min", "max"])
def test_minmax_extensionarray(method, numeric_only):
    # https://github.com/pandas-dev/pandas/issues/32651
    int64_info = np.iinfo("int64")
    ser = Series([int64_info.max, None, int64_info.min], dtype=pd.Int64Dtype())
    df = DataFrame({"Int64": ser})
    result = getattr(df, method)(numeric_only=numeric_only)
    expected = Series(
        [getattr(int64_info, method)],
        dtype="Int64",
        index=Index(["Int64"]),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ts_value", [Timestamp("2000-01-01"), pd.NaT])
def test_frame_mixed_numeric_object_with_timestamp(ts_value):
    # GH 13912
    df = DataFrame({"a": [1], "b": [1.1], "c": ["foo"], "d": [ts_value]})
    with pytest.raises(
        TypeError, match="does not support (operation|reduction)|Cannot perform"
    ):
        df.sum()


def test_prod_sum_min_count_mixed_object():
    # https://github.com/pandas-dev/pandas/issues/41074
    df = DataFrame([1, "a", True])

    result = df.prod(axis=0, min_count=1, numeric_only=False)
    expected = Series(["a"], dtype=object)
    tm.assert_series_equal(result, expected)

    msg = re.escape("unsupported operand type(s) for +: 'int' and 'str'")
    with pytest.raises(TypeError, match=msg):
        df.sum(axis=0, min_count=1, numeric_only=False)


@pytest.mark.parametrize("method", ["min", "max", "mean", "median", "skew", "kurt"])
@pytest.mark.parametrize("numeric_only", [True, False])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_reduction_axis_none_returns_scalar(method, numeric_only, dtype):
    # GH#21597 As of 2.0, axis=None reduces over all axes.

    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), dtype=dtype)

    result = getattr(df, method)(axis=None, numeric_only=numeric_only)
    np_arr = df.to_numpy(dtype=np.float64)
    if method in {"skew", "kurt"}:
        comp_mod = pytest.importorskip("scipy.stats")
        if method == "kurt":
            method = "kurtosis"
        expected = getattr(comp_mod, method)(np_arr, bias=False, axis=None)
        tm.assert_almost_equal(result, expected)
    else:
        expected = getattr(np, method)(np_arr, axis=None)
        assert result == expected


@pytest.mark.parametrize(
    "kernel",
    [
        "corr",
        "corrwith",
        "cov",
        "idxmax",
        "idxmin",
        "kurt",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "quantile",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
    ],
)
def test_fails_on_non_numeric(kernel):
    # GH#46852
    df = DataFrame({"a": [1, 2, 3], "b": object})
    args = (df,) if kernel == "corrwith" else ()
    msg = "|".join(
        [
            "not allowed for this dtype",
            "argument must be a string or a number",
            "not supported between instances of",
            "unsupported operand type",
            "argument must be a string or a real number",
        ]
    )
    if kernel == "median":
        # slightly different message on different builds
        msg1 = (
            r"Cannot convert \[\[<class 'object'> <class 'object'> "
            r"<class 'object'>\]\] to numeric"
        )
        msg2 = (
            r"Cannot convert \[<class 'object'> <class 'object'> "
            r"<class 'object'>\] to numeric"
        )
        msg = "|".join([msg1, msg2])
    with pytest.raises(TypeError, match=msg):
        getattr(df, kernel)(*args)


@pytest.mark.parametrize(
    "method",
    [
        "all",
        "any",
        "count",
        "idxmax",
        "idxmin",
        "kurt",
        "kurtosis",
        "max",
        "mean",
        "median",
        "min",
        "nunique",
        "prod",
        "product",
        "sem",
        "skew",
        "std",
        "sum",
        "var",
    ],
)
@pytest.mark.parametrize("min_count", [0, 2])
def test_numeric_ea_axis_1(method, skipna, min_count, any_numeric_ea_dtype):
    # GH 54341
    df = DataFrame(
        {
            "a": Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype),
            "b": Series([0, 1, pd.NA, 3], dtype=any_numeric_ea_dtype),
        },
    )
    expected_df = DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [0.0, 1.0, np.nan, 3.0],
        },
    )
    if method in ("count", "nunique"):
        expected_dtype = "int64"
    elif method in ("all", "any"):
        expected_dtype = "boolean"
    elif method in (
        "kurt",
        "kurtosis",
        "mean",
        "median",
        "sem",
        "skew",
        "std",
        "var",
    ) and not any_numeric_ea_dtype.startswith("Float"):
        expected_dtype = "Float64"
    else:
        expected_dtype = any_numeric_ea_dtype

    kwargs = {}
    if method not in ("count", "nunique", "quantile"):
        kwargs["skipna"] = skipna
    if method in ("prod", "product", "sum"):
        kwargs["min_count"] = min_count

    warn = None
    msg = None
    if not skipna and method in ("idxmax", "idxmin"):
        warn = FutureWarning
        msg = f"The behavior of DataFrame.{method} with all-NA values"
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(df, method)(axis=1, **kwargs)
    with tm.assert_produces_warning(warn, match=msg):
        expected = getattr(expected_df, method)(axis=1, **kwargs)
    if method not in ("idxmax", "idxmin"):
        expected = expected.astype(expected_dtype)
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\pivot.py ===

# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Alias,
    Typed,
)
from openpyxl.descriptors.nested import NestedInteger, NestedText
from openpyxl.descriptors.excel import ExtensionList

from .label import DataLabel
from .marker import Marker
from .shapes import GraphicalProperties
from .text import RichText


class PivotSource(Serialisable):

    tagname = "pivotSource"

    name = NestedText(expected_type=str)
    fmtId = NestedInteger(expected_type=int)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('name', 'fmtId')

    def __init__(self,
                 name=None,
                 fmtId=None,
                 extLst=None,
                ):
        self.name = name
        self.fmtId = fmtId


class PivotFormat(Serialisable):

    tagname = "pivotFmt"

    idx = NestedInteger(nested=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias("spPr")
    txPr = Typed(expected_type=RichText, allow_none=True)
    TextBody = Alias("txPr")
    marker = Typed(expected_type=Marker, allow_none=True)
    dLbl = Typed(expected_type=DataLabel, allow_none=True)
    DataLabel = Alias("dLbl")
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('idx', 'spPr', 'txPr', 'marker', 'dLbl')

    def __init__(self,
                 idx=0,
                 spPr=None,
                 txPr=None,
                 marker=None,
                 dLbl=None,
                 extLst=None,
                ):
        self.idx = idx
        self.spPr = spPr
        self.txPr = txPr
        self.marker = marker
        self.dLbl = dLbl

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\image.py ===
# Copyright (c) 2010-2024 openpyxl

from io import BytesIO

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = False


def _import_image(img):
    if not PILImage:
        raise ImportError('You must install Pillow to fetch image objects')

    if not isinstance(img, PILImage.Image):
        img = PILImage.open(img)

    return img


class Image:
    """Image in a spreadsheet"""

    _id = 1
    _path = "/xl/media/image{0}.{1}"
    anchor = "A1"

    def __init__(self, img):

        self.ref = img
        mark_to_close = isinstance(img, str)
        image = _import_image(img)
        self.width, self.height = image.size

        try:
            self.format = image.format.lower()
        except AttributeError:
            self.format = "png"
        if mark_to_close:
            # PIL instances created for metadata should be closed.
            image.close()


    def _data(self):
        """
        Return image data, convert to supported types if necessary
        """
        img = _import_image(self.ref)
        # don't convert these file formats
        if self.format in ['gif', 'jpeg', 'png']:
            img.fp.seek(0)
            fp = img.fp
        else:
            fp = BytesIO()
            img.save(fp, format="png")
            fp.seek(0)

        data = fp.read()
        fp.close()
        return data


    @property
    def path(self):
        return self._path.format(self._id, self.format)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\api.py ===
"""
Data IO api
"""

from pandas.io.clipboards import read_clipboard
from pandas.io.excel import (
    ExcelFile,
    ExcelWriter,
    read_excel,
)
from pandas.io.feather_format import read_feather
from pandas.io.gbq import read_gbq
from pandas.io.html import read_html
from pandas.io.json import read_json
from pandas.io.orc import read_orc
from pandas.io.parquet import read_parquet
from pandas.io.parsers import (
    read_csv,
    read_fwf,
    read_table,
)
from pandas.io.pickle import (
    read_pickle,
    to_pickle,
)
from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)
from pandas.io.sas import read_sas
from pandas.io.spss import read_spss
from pandas.io.sql import (
    read_sql,
    read_sql_query,
    read_sql_table,
)
from pandas.io.stata import read_stata
from pandas.io.xml import read_xml

__all__ = [
    "ExcelFile",
    "ExcelWriter",
    "HDFStore",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "to_pickle",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\dependency_groups\__main__.py ===
import argparse
import sys

from ._implementation import resolve
from ._toml_compat import tomllib


def main() -> None:
    if tomllib is None:
        print(
            "Usage error: dependency-groups CLI requires tomli or Python 3.11+",
            file=sys.stderr,
        )
        raise SystemExit(2)

    parser = argparse.ArgumentParser(
        description=(
            "A dependency-groups CLI. Prints out a resolved group, newline-delimited."
        )
    )
    parser.add_argument(
        "GROUP_NAME", nargs="*", help="The dependency group(s) to resolve."
    )
    parser.add_argument(
        "-f",
        "--pyproject-file",
        default="pyproject.toml",
        help="The pyproject.toml file. Defaults to trying in the current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="An output file. Defaults to stdout.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List the available dependency groups",
    )
    args = parser.parse_args()

    with open(args.pyproject_file, "rb") as fp:
        pyproject = tomllib.load(fp)

    dependency_groups_raw = pyproject.get("dependency-groups", {})

    if args.list:
        print(*dependency_groups_raw.keys())
        return
    if not args.GROUP_NAME:
        print("A GROUP_NAME is required", file=sys.stderr)
        raise SystemExit(3)

    content = "\n".join(resolve(dependency_groups_raw, *args.GROUP_NAME))

    if args.output is None or args.output == "-":
        print(content)
    else:
        with open(args.output, "w", encoding="utf-8") as fp:
            print(content, file=fp)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\ie\service.py ===
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
from typing import List, Optional

from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import service


class Service(service.Service):
    """Object that manages the starting and stopping of the IEDriver."""

    def __init__(
        self,
        executable_path: Optional[str] = None,
        port: int = 0,
        host: Optional[str] = None,
        service_args: Optional[List[str]] = None,
        log_level: Optional[str] = None,
        log_output: Optional[SubprocessStdAlias] = None,
        driver_path_env_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Creates a new instance of the Service.

        :Args:
         - executable_path : Path to the IEDriver
         - port : Port the service is running on
         - host : IP address the service port is bound
         - log_level : Level of logging of service, may be "FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE".
           Default is "FATAL".
         - log_output: (Optional) int representation of STDOUT/DEVNULL, any IO instance or String path to file.
           Default is "stdout".
        """
        self.service_args = service_args or []
        driver_path_env_key = driver_path_env_key or "SE_IEDRIVER"

        if host:
            self.service_args.append(f"--host={host}")
        if log_level:
            self.service_args.append(f"--log-level={log_level}")

        super().__init__(
            executable_path=executable_path,
            port=port,
            log_output=log_output,
            driver_path_env_key=driver_path_env_key,
            **kwargs,
        )

    def command_line_args(self) -> List[str]:
        return [f"--port={self.port}"] + self.service_args

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\wpewebkit\options.py ===
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

from typing import Dict

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions


class Options(ArgOptions):
    KEY = "wpe:browserOptions"

    def __init__(self) -> None:
        super().__init__()
        self._binary_location = ""

    @property
    def binary_location(self) -> str:
        """Returns the location of the browser binary otherwise an empty
        string."""
        return self._binary_location

    @binary_location.setter
    def binary_location(self, value: str) -> None:
        """Allows you to set the browser binary to launch.

        :Args:
         - value : path to the browser binary
        """
        if not isinstance(value, str):
            raise TypeError(self.BINARY_LOCATION_ERROR)
        self._binary_location = value

    def to_capabilities(self):
        """Creates a capabilities with all the options that have been set and
        returns a dictionary with everything."""
        caps = self._caps

        browser_options = {}
        if self.binary_location:
            browser_options["binary"] = self.binary_location
        if self.arguments:
            browser_options["args"] = self.arguments

        caps[Options.KEY] = browser_options

        return caps

    @property
    def default_capabilities(self) -> Dict[str, str]:
        return DesiredCapabilities.WPEWEBKIT.copy()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\declarative\__init__.py ===
# ext/declarative/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from .extensions import AbstractConcreteBase
from .extensions import ConcreteBase
from .extensions import DeferredReflection
from ... import util
from ...orm.decl_api import as_declarative as _as_declarative
from ...orm.decl_api import declarative_base as _declarative_base
from ...orm.decl_api import DeclarativeMeta
from ...orm.decl_api import declared_attr
from ...orm.decl_api import has_inherited_table as _has_inherited_table
from ...orm.decl_api import synonym_for as _synonym_for


@util.moved_20(
    "The ``declarative_base()`` function is now available as "
    ":func:`sqlalchemy.orm.declarative_base`."
)
def declarative_base(*arg, **kw):
    return _declarative_base(*arg, **kw)


@util.moved_20(
    "The ``as_declarative()`` function is now available as "
    ":func:`sqlalchemy.orm.as_declarative`"
)
def as_declarative(*arg, **kw):
    return _as_declarative(*arg, **kw)


@util.moved_20(
    "The ``has_inherited_table()`` function is now available as "
    ":func:`sqlalchemy.orm.has_inherited_table`."
)
def has_inherited_table(*arg, **kw):
    return _has_inherited_table(*arg, **kw)


@util.moved_20(
    "The ``synonym_for()`` function is now available as "
    ":func:`sqlalchemy.orm.synonym_for`"
)
def synonym_for(*arg, **kw):
    return _synonym_for(*arg, **kw)


__all__ = [
    "declarative_base",
    "synonym_for",
    "has_inherited_table",
    "instrument_declarative",
    "declared_attr",
    "as_declarative",
    "ConcreteBase",
    "AbstractConcreteBase",
    "DeclarativeMeta",
    "DeferredReflection",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\_print_helpers.py ===
"""
Base class to provide str and repr hooks that `init_printing` can overwrite.

This is exposed publicly in the `printing.defaults` module,
but cannot be defined there without causing circular imports.
"""

class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way. Natively Python uses ``repr()``
    even if ``str()`` was explicitly requested. Mix in this trait into
    a class to get proper default printing.

    This also adds support for LaTeX printing in jupyter notebooks.
    """

    # Since this class is used as a mixin we set empty slots. That means that
    # instances of any subclasses that use slots will not need to have a
    # __dict__.
    __slots__ = ()

    # Note, we always use the default ordering (lex) in __str__ and __repr__,
    # regardless of the global setting. See issue 5487.
    def __str__(self):
        from sympy.printing.str import sstr
        return sstr(self, order=None)

    __repr__ = __str__

    def _repr_disabled(self):
        """
        No-op repr function used to disable jupyter display hooks.

        When :func:`sympy.init_printing` is used to disable certain display
        formats, this function is copied into the appropriate ``_repr_*_``
        attributes.

        While we could just set the attributes to `None``, doing it this way
        allows derived classes to call `super()`.
        """
        return None

    # We don't implement _repr_png_ here because it would add a large amount of
    # data to any notebook containing SymPy expressions, without adding
    # anything useful to the notebook. It can still enabled manually, e.g.,
    # for the qtconsole, with init_printing().
    _repr_png_ = _repr_disabled

    _repr_svg_ = _repr_disabled

    def _repr_latex_(self):
        """
        IPython/Jupyter LaTeX printing

        To change the behavior of this (e.g., pass in some settings to LaTeX),
        use init_printing(). init_printing() will also enable LaTeX printing
        for built in numeric types like ints and container types that contain
        SymPy objects, like lists and dictionaries of expressions.
        """
        from sympy.printing.latex import latex
        s = latex(self, mode='plain')
        return "$\\displaystyle %s$" % s

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\__init__.py ===
# Names exposed by 'from sympy.physics.quantum import *'

__all__ = [
    'AntiCommutator',

    'qapply',

    'Commutator',

    'Dagger',

    'HilbertSpaceError', 'HilbertSpace', 'TensorProductHilbertSpace',
    'TensorPowerHilbertSpace', 'DirectSumHilbertSpace', 'ComplexSpace', 'L2',
    'FockSpace',

    'InnerProduct',

    'Operator', 'HermitianOperator', 'UnitaryOperator', 'IdentityOperator',
    'OuterProduct', 'DifferentialOperator',

    'represent', 'rep_innerproduct', 'rep_expectation', 'integrate_result',
    'get_basis', 'enumerate_states',

    'KetBase', 'BraBase', 'StateBase', 'State', 'Ket', 'Bra', 'TimeDepState',
    'TimeDepBra', 'TimeDepKet', 'OrthogonalKet', 'OrthogonalBra',
    'OrthogonalState', 'Wavefunction',

    'TensorProduct', 'tensor_product_simp',

    'hbar', 'HBar',

    '_postprocess_state_mul', '_postprocess_state_pow'
]

from .anticommutator import AntiCommutator

from .qapply import qapply

from .commutator import Commutator

from .dagger import Dagger

from .hilbert import (HilbertSpaceError, HilbertSpace,
        TensorProductHilbertSpace, TensorPowerHilbertSpace,
        DirectSumHilbertSpace, ComplexSpace, L2, FockSpace)

from .innerproduct import InnerProduct

from .operator import (Operator, HermitianOperator, UnitaryOperator,
        IdentityOperator, OuterProduct, DifferentialOperator)

from .represent import (represent, rep_innerproduct, rep_expectation,
        integrate_result, get_basis, enumerate_states)

from .state import (KetBase, BraBase, StateBase, State, Ket, Bra,
        TimeDepState, TimeDepBra, TimeDepKet, OrthogonalKet,
        OrthogonalBra, OrthogonalState, Wavefunction)

from .tensorproduct import TensorProduct, tensor_product_simp

from .constants import hbar, HBar

# These are private, but need to be imported so they are registered
# as postprocessing transformers with Mul and Pow.
from .transforms import _postprocess_state_mul, _postprocess_state_pow

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_highlevel_open_unix_stream.py ===
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, TypeVar

import trio
from trio.socket import SOCK_STREAM, socket

if TYPE_CHECKING:
    from collections.abc import Generator


class Closable(Protocol):
    def close(self) -> None: ...


CloseT = TypeVar("CloseT", bound=Closable)


try:
    from trio.socket import AF_UNIX

    has_unix = True
except ImportError:
    has_unix = False


@contextmanager
def close_on_error(obj: CloseT) -> Generator[CloseT, None, None]:
    try:
        yield obj
    except:
        obj.close()
        raise


async def open_unix_socket(
    filename: str | bytes | os.PathLike[str] | os.PathLike[bytes],
) -> trio.SocketStream:
    """Opens a connection to the specified
    `Unix domain socket <https://en.wikipedia.org/wiki/Unix_domain_socket>`__.

    You must have read/write permission on the specified file to connect.

    Args:
      filename (str or bytes): The filename to open the connection to.

    Returns:
      SocketStream: a :class:`~trio.abc.Stream` connected to the given file.

    Raises:
      OSError: If the socket file could not be connected to.
      RuntimeError: If AF_UNIX sockets are not supported.
    """
    if not has_unix:
        raise RuntimeError("Unix sockets are not supported on this platform")

    # much more simplified logic vs tcp sockets - one socket type and only one
    # possible location to connect to
    sock = socket(AF_UNIX, SOCK_STREAM)
    with close_on_error(sock):
        await sock.connect(os.fspath(filename))

    return trio.SocketStream(sock)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\blinker\_utilities.py ===
from __future__ import annotations

import collections.abc as c
import inspect
import typing as t
from weakref import ref
from weakref import WeakMethod

T = t.TypeVar("T")


class Symbol:
    """A constant symbol, nicer than ``object()``. Repeated calls return the
    same instance.

    >>> Symbol('foo') is Symbol('foo')
    True
    >>> Symbol('foo')
    foo
    """

    symbols: t.ClassVar[dict[str, Symbol]] = {}

    def __new__(cls, name: str) -> Symbol:
        if name in cls.symbols:
            return cls.symbols[name]

        obj = super().__new__(cls)
        cls.symbols[name] = obj
        return obj

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __getnewargs__(self) -> tuple[t.Any, ...]:
        return (self.name,)


def make_id(obj: object) -> c.Hashable:
    """Get a stable identifier for a receiver or sender, to be used as a dict
    key or in a set.
    """
    if inspect.ismethod(obj):
        # The id of a bound method is not stable, but the id of the unbound
        # function and instance are.
        return id(obj.__func__), id(obj.__self__)

    if isinstance(obj, (str, int)):
        # Instances with the same value always compare equal and have the same
        # hash, even if the id may change.
        return obj

    # Assume other types are not hashable but will always be the same instance.
    return id(obj)


def make_ref(obj: T, callback: c.Callable[[ref[T]], None] | None = None) -> ref[T]:
    if inspect.ismethod(obj):
        return WeakMethod(obj, callback)  # type: ignore[arg-type, return-value]

    return ref(obj, callback)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\legacy.py ===
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

from .api import from_bytes
from .constant import CHARDET_CORRESPONDENCE

# TODO: remove this check when dropping Python 3.7 support
if TYPE_CHECKING:
    from typing_extensions import TypedDict

    class ResultDict(TypedDict):
        encoding: str | None
        language: str
        confidence: float | None


def detect(
    byte_str: bytes, should_rename_legacy: bool = False, **kwargs: Any
) -> ResultDict:
    """
    chardet legacy method
    Detect the encoding of the given byte string. It should be mostly backward-compatible.
    Encoding name will match Chardet own writing whenever possible. (Not on encoding name unsupported by it)
    This function is deprecated and should be used to migrate your project easily, consult the documentation for
    further information. Not planned for removal.

    :param byte_str:     The byte sequence to examine.
    :param should_rename_legacy:  Should we rename legacy encodings
                                  to their more modern equivalents?
    """
    if len(kwargs):
        warn(
            f"charset-normalizer disregard arguments '{','.join(list(kwargs.keys()))}' in legacy function detect()"
        )

    if not isinstance(byte_str, (bytearray, bytes)):
        raise TypeError(  # pragma: nocover
            f"Expected object of type bytes or bytearray, got: {type(byte_str)}"
        )

    if isinstance(byte_str, bytearray):
        byte_str = bytes(byte_str)

    r = from_bytes(byte_str).best()

    encoding = r.encoding if r is not None else None
    language = r.language if r is not None and r.language != "Unknown" else ""
    confidence = 1.0 - r.chaos if r is not None else None

    # Note: CharsetNormalizer does not return 'UTF-8-SIG' as the sig get stripped in the detection/normalization process
    # but chardet does return 'utf-8-sig' and it is a valid codec name.
    if r is not None and encoding == "utf_8" and r.bom:
        encoding += "_sig"

    if should_rename_legacy is False and encoding in CHARDET_CORRESPONDENCE:
        encoding = CHARDET_CORRESPONDENCE[encoding]

    return {
        "encoding": encoding,
        "language": language,
        "confidence": confidence,
    }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\gather.py ===
from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase

"""
    Quantize Gather
"""


class GatherQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def should_quantize(self):
        if not self.quantizer.should_quantize_node(self.node):
            return False

        return self.quantizer.is_valid_quantize_weight(self.node.input[0])

    def quantize(self):
        node = self.node
        assert node.op_type == "Gather"

        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0])
        if quantized_input_names is None:
            return super().quantize()

        gather_new_output = node.output[0] + TENSOR_NAME_QUANT_SUFFIX

        # Create an entry for this quantized value
        q_output = QuantizedValue(
            node.output[0],
            gather_new_output,
            scale_names[0],
            zero_point_names[0],
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        node.output[0] = gather_new_output
        node.input[0] = quantized_input_names[0]
        nodes.append(node)

        self.quantizer.new_nodes += nodes


class QDQGather(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Gather" or node.op_type == "GatherElements"

        if self.quantizer.is_valid_quantize_weight(node.input[0]) or self.quantizer.force_quantize_no_input_check:
            self.quantizer.quantize_activation_tensor(node.input[0])
            self.quantizer.quantize_output_same_as_input(node.output[0], node.input[0], node.name)
        elif self.quantizer.is_tensor_quantized(node.input[0]):
            self.quantizer.quantize_output_same_as_input(node.output[0], node.input[0], node.name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\xml\test_xml.py ===
from __future__ import annotations

from io import (
    BytesIO,
    StringIO,
)
from lzma import LZMAError
import os
from tarfile import ReadError
from urllib.error import HTTPError
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import numpy as np
import pytest

from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    EmptyDataError,
    ParserError,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    NA,
    DataFrame,
    Series,
)
import pandas._testing as tm

from pandas.io.common import get_handle
from pandas.io.xml import read_xml

# CHECK LIST

# [x] - ValueError: "Values for parser can only be lxml or etree."

# etree
# [X] - ImportError: "lxml not found, please install or use the etree parser."
# [X] - TypeError: "expected str, bytes or os.PathLike object, not NoneType"
# [X] - ValueError: "Either element or attributes can be parsed not both."
# [X] - ValueError: "xpath does not return any nodes..."
# [X] - SyntaxError: "You have used an incorrect or unsupported XPath"
# [X] - ValueError: "names does not match length of child elements in xpath."
# [X] - TypeError: "...is not a valid type for names"
# [X] - ValueError: "To use stylesheet, you need lxml installed..."
# []  - URLError: (GENERAL ERROR WITH HTTPError AS SUBCLASS)
# [X] - HTTPError: "HTTP Error 404: Not Found"
# []  - OSError: (GENERAL ERROR WITH FileNotFoundError AS SUBCLASS)
# [X] - FileNotFoundError: "No such file or directory"
# []  - ParseError    (FAILSAFE CATCH ALL FOR VERY COMPLEX XML)
# [X] - UnicodeDecodeError: "'utf-8' codec can't decode byte 0xe9..."
# [X] - UnicodeError: "UTF-16 stream does not start with BOM"
# [X] - BadZipFile: "File is not a zip file"
# [X] - OSError: "Invalid data stream"
# [X] - LZMAError: "Input format not supported by decoder"
# [X] - ValueError: "Unrecognized compression type"
# [X] - PermissionError: "Forbidden"

# lxml
# [X] - ValueError: "Either element or attributes can be parsed not both."
# [X] - AttributeError: "__enter__"
# [X] - XSLTApplyError: "Cannot resolve URI"
# [X] - XSLTParseError: "document is not a stylesheet"
# [X] - ValueError: "xpath does not return any nodes."
# [X] - XPathEvalError: "Invalid expression"
# []  - XPathSyntaxError: (OLD VERSION IN lxml FOR XPATH ERRORS)
# [X] - TypeError: "empty namespace prefix is not supported in XPath"
# [X] - ValueError: "names does not match length of child elements in xpath."
# [X] - TypeError: "...is not a valid type for names"
# [X] - LookupError: "unknown encoding"
# []  - URLError: (USUALLY DUE TO NETWORKING)
# [X  - HTTPError: "HTTP Error 404: Not Found"
# [X] - OSError: "failed to load external entity"
# [X] - XMLSyntaxError: "Start tag expected, '<' not found"
# []  - ParserError: (FAILSAFE CATCH ALL FOR VERY COMPLEX XML
# [X] - ValueError: "Values for parser can only be lxml or etree."
# [X] - UnicodeDecodeError: "'utf-8' codec can't decode byte 0xe9..."
# [X] - UnicodeError: "UTF-16 stream does not start with BOM"
# [X] - BadZipFile: "File is not a zip file"
# [X] - OSError: "Invalid data stream"
# [X] - LZMAError: "Input format not supported by decoder"
# [X] - ValueError: "Unrecognized compression type"
# [X] - PermissionError: "Forbidden"

geom_df = DataFrame(
    {
        "shape": ["square", "circle", "triangle"],
        "degrees": [360, 360, 180],
        "sides": [4, np.nan, 3],
    }
)

xml_default_nmsp = """\
<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
  <row>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4</sides>
  </row>
  <row>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3</sides>
  </row>
</data>"""

xml_prefix_nmsp = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://example.com">
  <doc:row>
    <doc:shape>square</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides>4.0</doc:sides>
  </doc:row>
  <doc:row>
    <doc:shape>circle</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides/>
  </doc:row>
  <doc:row>
    <doc:shape>triangle</doc:shape>
    <doc:degrees>180</doc:degrees>
    <doc:sides>3.0</doc:sides>
  </doc:row>
</doc:data>"""


df_kml = DataFrame(
    {
        "id": {
            0: "ID_00001",
            1: "ID_00002",
            2: "ID_00003",
            3: "ID_00004",
            4: "ID_00005",
        },
        "name": {
            0: "Blue Line (Forest Park)",
            1: "Red, Purple Line",
            2: "Red, Purple Line",
            3: "Red, Purple Line",
            4: "Red, Purple Line",
        },
        "styleUrl": {
            0: "#LineStyle01",
            1: "#LineStyle01",
            2: "#LineStyle01",
            3: "#LineStyle01",
            4: "#LineStyle01",
        },
        "extrude": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "altitudeMode": {
            0: "clampedToGround",
            1: "clampedToGround",
            2: "clampedToGround",
            3: "clampedToGround",
            4: "clampedToGround",
        },
        "coordinates": {
            0: (
                "-87.77678526964958,41.8708863930319,0 "
                "-87.77826234150609,41.87097820122218,0 "
                "-87.78251583439344,41.87130129991005,0 "
                "-87.78418294588424,41.87145055520308,0 "
                "-87.7872369165933,41.8717239119163,0 "
                "-87.79160214925886,41.87210797280065,0"
            ),
            1: (
                "-87.65758750947528,41.96427269188822,0 "
                "-87.65802133507393,41.96581929055245,0 "
                "-87.65819033925305,41.96621846093642,0 "
                "-87.6583189819129,41.96650362897086,0 "
                "-87.65835858701473,41.96669002089185,0 "
                "-87.65838428411853,41.96688150295095,0 "
                "-87.65842208882658,41.96745896091846,0 "
                "-87.65846556843937,41.9683761425439,0 "
                "-87.65849296214573,41.96913893870342,0"
            ),
            2: (
                "-87.65492939166126,41.95377494531437,0 "
                "-87.65557043199591,41.95376544118533,0 "
                "-87.65606302030132,41.95376391658746,0 "
                "-87.65623502146268,41.95377379126367,0 "
                "-87.65634748981634,41.95380103566435,0 "
                "-87.65646537904269,41.95387703994676,0 "
                "-87.65656532461145,41.95396622645799,0 "
                "-87.65664760856414,41.95404201996044,0 "
                "-87.65671750555913,41.95416647054043,0 "
                "-87.65673983607117,41.95429949810849,0 "
                "-87.65673866475777,41.95441024240925,0 "
                "-87.6567690255541,41.95490657227902,0 "
                "-87.65683672482363,41.95692259283837,0 "
                "-87.6568900886376,41.95861070983142,0 "
                "-87.65699865558875,41.96181418669004,0 "
                "-87.65756347177603,41.96397045777844,0 "
                "-87.65758750947528,41.96427269188822,0"
            ),
            3: (
                "-87.65362593118043,41.94742799535678,0 "
                "-87.65363554415794,41.94819886386848,0 "
                "-87.6536456393239,41.95059994675451,0 "
                "-87.65365831235026,41.95108288489359,0 "
                "-87.6536604873874,41.9519954657554,0 "
                "-87.65362592053201,41.95245597302328,0 "
                "-87.65367158496069,41.95311153649393,0 "
                "-87.65368468595476,41.9533202828916,0 "
                "-87.65369271253692,41.95343095587119,0 "
                "-87.65373335834569,41.95351536301472,0 "
                "-87.65378605844126,41.95358212680591,0 "
                "-87.65385067928185,41.95364452823767,0 "
                "-87.6539390793817,41.95370263886964,0 "
                "-87.6540786298351,41.95373403675265,0 "
                "-87.65430648647626,41.9537535411832,0 "
                "-87.65492939166126,41.95377494531437,0"
            ),
            4: (
                "-87.65345391792157,41.94217681262115,0 "
                "-87.65342448305786,41.94237224420864,0 "
                "-87.65339745703922,41.94268217746244,0 "
                "-87.65337753982941,41.94288140770284,0 "
                "-87.65336256753105,41.94317369618263,0 "
                "-87.65338799707138,41.94357253961736,0 "
                "-87.65340240886648,41.94389158188269,0 "
                "-87.65341837392448,41.94406444407721,0 "
                "-87.65342275247338,41.94421065714904,0 "
                "-87.65347469646018,41.94434829382345,0 "
                "-87.65351486483024,41.94447699917548,0 "
                "-87.65353483605053,41.9453896864472,0 "
                "-87.65361975532807,41.94689193720703,0 "
                "-87.65362593118043,41.94742799535678,0"
            ),
        },
    }
)


def test_literal_xml_deprecation():
    # GH 53809
    pytest.importorskip("lxml")
    msg = (
        "Passing literal xml to 'read_xml' is deprecated and "
        "will be removed in a future version. To read from a "
        "literal string, wrap it in a 'StringIO' object."
    )

    with tm.assert_produces_warning(FutureWarning, match=msg):
        read_xml(xml_default_nmsp)


@pytest.fixture(params=["rb", "r"])
def mode(request):
    return request.param


@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    return request.param


def read_xml_iterparse(data, **kwargs):
    with tm.ensure_clean() as path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        return read_xml(path, **kwargs)


def read_xml_iterparse_comp(comp_path, compression_only, **kwargs):
    with get_handle(comp_path, "r", compression=compression_only) as handles:
        with tm.ensure_clean() as path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(handles.handle.read())
            return read_xml(path, **kwargs)


# FILE / URL


def test_parser_consistency_file(xml_books):
    pytest.importorskip("lxml")
    df_file_lxml = read_xml(xml_books, parser="lxml")
    df_file_etree = read_xml(xml_books, parser="etree")

    df_iter_lxml = read_xml(
        xml_books,
        parser="lxml",
        iterparse={"book": ["category", "title", "year", "author", "price"]},
    )
    df_iter_etree = read_xml(
        xml_books,
        parser="etree",
        iterparse={"book": ["category", "title", "year", "author", "price"]},
    )

    tm.assert_frame_equal(df_file_lxml, df_file_etree)
    tm.assert_frame_equal(df_file_lxml, df_iter_lxml)
    tm.assert_frame_equal(df_iter_lxml, df_iter_etree)


@pytest.mark.network
@pytest.mark.single_cpu
def test_parser_consistency_url(parser, httpserver):
    httpserver.serve_content(content=xml_default_nmsp)

    df_xpath = read_xml(StringIO(xml_default_nmsp), parser=parser)
    df_iter = read_xml(
        BytesIO(xml_default_nmsp.encode()),
        parser=parser,
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    tm.assert_frame_equal(df_xpath, df_iter)


def test_file_like(xml_books, parser, mode):
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        df_file = read_xml(f, parser=parser)

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)


def test_file_io(xml_books, parser, mode):
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        xml_obj = f.read()

    df_io = read_xml(
        (BytesIO(xml_obj) if isinstance(xml_obj, bytes) else StringIO(xml_obj)),
        parser=parser,
    )

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_io, df_expected)


def test_file_buffered_reader_string(xml_books, parser, mode):
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        xml_obj = f.read()

    if mode == "rb":
        xml_obj = StringIO(xml_obj.decode())
    elif mode == "r":
        xml_obj = StringIO(xml_obj)

    df_str = read_xml(xml_obj, parser=parser)

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_str, df_expected)


def test_file_buffered_reader_no_xml_declaration(xml_books, parser, mode):
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        next(f)
        xml_obj = f.read()

    if mode == "rb":
        xml_obj = StringIO(xml_obj.decode())
    elif mode == "r":
        xml_obj = StringIO(xml_obj)

    df_str = read_xml(xml_obj, parser=parser)

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_str, df_expected)


def test_string_charset(parser):
    txt = "<中文標籤><row><c1>1</c1><c2>2</c2></row></中文標籤>"

    df_str = read_xml(StringIO(txt), parser=parser)

    df_expected = DataFrame({"c1": 1, "c2": 2}, index=[0])

    tm.assert_frame_equal(df_str, df_expected)


def test_file_charset(xml_doc_ch_utf, parser):
    df_file = read_xml(xml_doc_ch_utf, parser=parser)

    df_expected = DataFrame(
        {
            "問": [
                "問  若箇是邪而言破邪 何者是正而道(Sorry, this is Big5 only)申正",
                "問 既破有得申無得 亦應但破性執申假名以不",
                "問 既破性申假 亦應但破有申無 若有無兩洗 亦應性假雙破耶",
            ],
            "答": [
                "".join(
                    [
                        "答  邪既無量 正亦多途  大略為言不出二種 謂",
                        "有得與無得 有得是邪須破 無得是正須申\n\t\t故",
                    ]
                ),
                None,
                "答  不例  有無皆是性 所以須雙破 既分性假異 故有破不破",
            ],
            "a": [
                None,
                "答 性執是有得 假名是無得  今破有得申無得 即是破性執申假名也",
                None,
            ],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)


def test_file_handle_close(xml_books, parser):
    with open(xml_books, "rb") as f:
        read_xml(BytesIO(f.read()), parser=parser)

        assert not f.closed


@pytest.mark.parametrize("val", ["", b""])
def test_empty_string_lxml(val):
    lxml_etree = pytest.importorskip("lxml.etree")

    msg = "|".join(
        [
            "Document is empty",
            # Seen on Mac with lxml 4.91
            r"None \(line 0\)",
        ]
    )
    with pytest.raises(lxml_etree.XMLSyntaxError, match=msg):
        if isinstance(val, str):
            read_xml(StringIO(val), parser="lxml")
        else:
            read_xml(BytesIO(val), parser="lxml")


@pytest.mark.parametrize("val", ["", b""])
def test_empty_string_etree(val):
    with pytest.raises(ParseError, match="no element found"):
        if isinstance(val, str):
            read_xml(StringIO(val), parser="etree")
        else:
            read_xml(BytesIO(val), parser="etree")


def test_wrong_file_path(parser):
    msg = (
        "Passing literal xml to 'read_xml' is deprecated and "
        "will be removed in a future version. To read from a "
        "literal string, wrap it in a 'StringIO' object."
    )
    filename = os.path.join("data", "html", "books.xml")

    with pytest.raises(
        FutureWarning,
        match=msg,
    ):
        read_xml(filename, parser=parser)


@pytest.mark.network
@pytest.mark.single_cpu
def test_url(httpserver, xml_file):
    pytest.importorskip("lxml")
    with open(xml_file, encoding="utf-8") as f:
        httpserver.serve_content(content=f.read())
        df_url = read_xml(httpserver.url, xpath=".//book[count(*)=4]")

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_url, df_expected)


@pytest.mark.network
@pytest.mark.single_cpu
def test_wrong_url(parser, httpserver):
    httpserver.serve_content("NOT FOUND", code=404)
    with pytest.raises(HTTPError, match=("HTTP Error 404: NOT FOUND")):
        read_xml(httpserver.url, xpath=".//book[count(*)=4]", parser=parser)


# CONTENT


def test_whitespace(parser):
    xml = """
      <data>
        <row sides=" 4 ">
          <shape>
              square
          </shape>
          <degrees>&#009;360&#009;</degrees>
        </row>
        <row sides=" 0 ">
          <shape>
              circle
          </shape>
          <degrees>&#009;360&#009;</degrees>
        </row>
        <row sides=" 3 ">
          <shape>
              triangle
          </shape>
          <degrees>&#009;180&#009;</degrees>
        </row>
      </data>"""

    df_xpath = read_xml(StringIO(xml), parser=parser, dtype="string")

    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        iterparse={"row": ["sides", "shape", "degrees"]},
        dtype="string",
    )

    df_expected = DataFrame(
        {
            "sides": [" 4 ", " 0 ", " 3 "],
            "shape": [
                "\n              square\n          ",
                "\n              circle\n          ",
                "\n              triangle\n          ",
            ],
            "degrees": ["\t360\t", "\t360\t", "\t180\t"],
        },
        dtype="string",
    )

    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


# XPATH


def test_empty_xpath_lxml(xml_books):
    pytest.importorskip("lxml")
    with pytest.raises(ValueError, match=("xpath does not return any nodes")):
        read_xml(xml_books, xpath=".//python", parser="lxml")


def test_bad_xpath_etree(xml_books):
    with pytest.raises(
        SyntaxError, match=("You have used an incorrect or unsupported XPath")
    ):
        read_xml(xml_books, xpath=".//[book]", parser="etree")


def test_bad_xpath_lxml(xml_books):
    lxml_etree = pytest.importorskip("lxml.etree")

    with pytest.raises(lxml_etree.XPathEvalError, match=("Invalid expression")):
        read_xml(xml_books, xpath=".//[book]", parser="lxml")


# NAMESPACE


def test_default_namespace(parser):
    df_nmsp = read_xml(
        StringIO(xml_default_nmsp),
        xpath=".//ns:row",
        namespaces={"ns": "http://example.com"},
        parser=parser,
    )

    df_iter = read_xml_iterparse(
        xml_default_nmsp,
        parser=parser,
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_nmsp, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_prefix_namespace(parser):
    df_nmsp = read_xml(
        StringIO(xml_prefix_nmsp),
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser=parser,
    )
    df_iter = read_xml_iterparse(
        xml_prefix_nmsp, parser=parser, iterparse={"row": ["shape", "degrees", "sides"]}
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_nmsp, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_consistency_default_namespace():
    pytest.importorskip("lxml")
    df_lxml = read_xml(
        StringIO(xml_default_nmsp),
        xpath=".//ns:row",
        namespaces={"ns": "http://example.com"},
        parser="lxml",
    )

    df_etree = read_xml(
        StringIO(xml_default_nmsp),
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser="etree",
    )

    tm.assert_frame_equal(df_lxml, df_etree)


def test_consistency_prefix_namespace():
    pytest.importorskip("lxml")
    df_lxml = read_xml(
        StringIO(xml_prefix_nmsp),
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser="lxml",
    )

    df_etree = read_xml(
        StringIO(xml_prefix_nmsp),
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser="etree",
    )

    tm.assert_frame_equal(df_lxml, df_etree)


# PREFIX


def test_missing_prefix_with_default_namespace(xml_books, parser):
    with pytest.raises(ValueError, match=("xpath does not return any nodes")):
        read_xml(xml_books, xpath=".//Placemark", parser=parser)


def test_missing_prefix_definition_etree(kml_cta_rail_lines):
    with pytest.raises(SyntaxError, match=("you used an undeclared namespace prefix")):
        read_xml(kml_cta_rail_lines, xpath=".//kml:Placemark", parser="etree")


def test_missing_prefix_definition_lxml(kml_cta_rail_lines):
    lxml_etree = pytest.importorskip("lxml.etree")

    with pytest.raises(lxml_etree.XPathEvalError, match=("Undefined namespace prefix")):
        read_xml(kml_cta_rail_lines, xpath=".//kml:Placemark", parser="lxml")


@pytest.mark.parametrize("key", ["", None])
def test_none_namespace_prefix(key):
    pytest.importorskip("lxml")
    with pytest.raises(
        TypeError, match=("empty namespace prefix is not supported in XPath")
    ):
        read_xml(
            StringIO(xml_default_nmsp),
            xpath=".//kml:Placemark",
            namespaces={key: "http://www.opengis.net/kml/2.2"},
            parser="lxml",
        )


# ELEMS AND ATTRS


def test_file_elems_and_attrs(xml_books, parser):
    df_file = read_xml(xml_books, parser=parser)
    df_iter = read_xml(
        xml_books,
        parser=parser,
        iterparse={"book": ["category", "title", "author", "year", "price"]},
    )
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_file_only_attrs(xml_books, parser):
    df_file = read_xml(xml_books, attrs_only=True, parser=parser)
    df_iter = read_xml(xml_books, parser=parser, iterparse={"book": ["category"]})
    df_expected = DataFrame({"category": ["cooking", "children", "web"]})

    tm.assert_frame_equal(df_file, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_file_only_elems(xml_books, parser):
    df_file = read_xml(xml_books, elems_only=True, parser=parser)
    df_iter = read_xml(
        xml_books,
        parser=parser,
        iterparse={"book": ["title", "author", "year", "price"]},
    )
    df_expected = DataFrame(
        {
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_elem_and_attrs_only(kml_cta_rail_lines, parser):
    with pytest.raises(
        ValueError,
        match=("Either element or attributes can be parsed not both"),
    ):
        read_xml(kml_cta_rail_lines, elems_only=True, attrs_only=True, parser=parser)


def test_empty_attrs_only(parser):
    xml = """
      <data>
        <row>
          <shape sides="4">square</shape>
          <degrees>360</degrees>
        </row>
        <row>
          <shape sides="0">circle</shape>
          <degrees>360</degrees>
        </row>
        <row>
          <shape sides="3">triangle</shape>
          <degrees>180</degrees>
        </row>
      </data>"""

    with pytest.raises(
        ValueError,
        match=("xpath does not return any nodes or attributes"),
    ):
        read_xml(StringIO(xml), xpath="./row", attrs_only=True, parser=parser)


def test_empty_elems_only(parser):
    xml = """
      <data>
        <row sides="4" shape="square" degrees="360"/>
        <row sides="0" shape="circle" degrees="360"/>
        <row sides="3" shape="triangle" degrees="180"/>
      </data>"""

    with pytest.raises(
        ValueError,
        match=("xpath does not return any nodes or attributes"),
    ):
        read_xml(StringIO(xml), xpath="./row", elems_only=True, parser=parser)


def test_attribute_centric_xml():
    pytest.importorskip("lxml")
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<TrainSchedule>
      <Stations>
         <station Name="Manhattan" coords="31,460,195,498"/>
         <station Name="Laraway Road" coords="63,409,194,455"/>
         <station Name="179th St (Orland Park)" coords="0,364,110,395"/>
         <station Name="153rd St (Orland Park)" coords="7,333,113,362"/>
         <station Name="143rd St (Orland Park)" coords="17,297,115,330"/>
         <station Name="Palos Park" coords="128,281,239,303"/>
         <station Name="Palos Heights" coords="148,257,283,279"/>
         <station Name="Worth" coords="170,230,248,255"/>
         <station Name="Chicago Ridge" coords="70,187,208,214"/>
         <station Name="Oak Lawn" coords="166,159,266,185"/>
         <station Name="Ashburn" coords="197,133,336,157"/>
         <station Name="Wrightwood" coords="219,106,340,133"/>
         <station Name="Chicago Union Sta" coords="220,0,360,43"/>
      </Stations>
</TrainSchedule>"""

    df_lxml = read_xml(StringIO(xml), xpath=".//station")
    df_etree = read_xml(StringIO(xml), xpath=".//station", parser="etree")

    df_iter_lx = read_xml_iterparse(xml, iterparse={"station": ["Name", "coords"]})
    df_iter_et = read_xml_iterparse(
        xml, parser="etree", iterparse={"station": ["Name", "coords"]}
    )

    tm.assert_frame_equal(df_lxml, df_etree)
    tm.assert_frame_equal(df_iter_lx, df_iter_et)


# NAMES


def test_names_option_output(xml_books, parser):
    df_file = read_xml(
        xml_books, names=["Col1", "Col2", "Col3", "Col4", "Col5"], parser=parser
    )
    df_iter = read_xml(
        xml_books,
        parser=parser,
        names=["Col1", "Col2", "Col3", "Col4", "Col5"],
        iterparse={"book": ["category", "title", "author", "year", "price"]},
    )

    df_expected = DataFrame(
        {
            "Col1": ["cooking", "children", "web"],
            "Col2": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "Col3": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "Col4": [2005, 2005, 2003],
            "Col5": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_repeat_names(parser):
    xml = """\
<shapes>
  <shape type="2D">
    <name>circle</name>
    <type>curved</type>
  </shape>
  <shape type="3D">
    <name>sphere</name>
    <type>curved</type>
  </shape>
</shapes>"""
    df_xpath = read_xml(
        StringIO(xml),
        xpath=".//shape",
        parser=parser,
        names=["type_dim", "shape", "type_edge"],
    )

    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        iterparse={"shape": ["type", "name", "type"]},
        names=["type_dim", "shape", "type_edge"],
    )

    df_expected = DataFrame(
        {
            "type_dim": ["2D", "3D"],
            "shape": ["circle", "sphere"],
            "type_edge": ["curved", "curved"],
        }
    )

    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_repeat_values_new_names(parser):
    xml = """\
<shapes>
  <shape>
    <name>rectangle</name>
    <family>rectangle</family>
  </shape>
  <shape>
    <name>square</name>
    <family>rectangle</family>
  </shape>
  <shape>
    <name>ellipse</name>
    <family>ellipse</family>
  </shape>
  <shape>
    <name>circle</name>
    <family>ellipse</family>
  </shape>
</shapes>"""
    df_xpath = read_xml(
        StringIO(xml), xpath=".//shape", parser=parser, names=["name", "group"]
    )

    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        iterparse={"shape": ["name", "family"]},
        names=["name", "group"],
    )

    df_expected = DataFrame(
        {
            "name": ["rectangle", "square", "ellipse", "circle"],
            "group": ["rectangle", "rectangle", "ellipse", "ellipse"],
        }
    )

    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_repeat_elements(parser):
    xml = """\
<shapes>
  <shape>
    <value item="name">circle</value>
    <value item="family">ellipse</value>
    <value item="degrees">360</value>
    <value item="sides">0</value>
  </shape>
  <shape>
    <value item="name">triangle</value>
    <value item="family">polygon</value>
    <value item="degrees">180</value>
    <value item="sides">3</value>
  </shape>
  <shape>
    <value item="name">square</value>
    <value item="family">polygon</value>
    <value item="degrees">360</value>
    <value item="sides">4</value>
  </shape>
</shapes>"""
    df_xpath = read_xml(
        StringIO(xml),
        xpath=".//shape",
        parser=parser,
        names=["name", "family", "degrees", "sides"],
    )

    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        iterparse={"shape": ["value", "value", "value", "value"]},
        names=["name", "family", "degrees", "sides"],
    )

    df_expected = DataFrame(
        {
            "name": ["circle", "triangle", "square"],
            "family": ["ellipse", "polygon", "polygon"],
            "degrees": [360, 180, 360],
            "sides": [0, 3, 4],
        }
    )

    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_names_option_wrong_length(xml_books, parser):
    with pytest.raises(ValueError, match=("names does not match length")):
        read_xml(xml_books, names=["Col1", "Col2", "Col3"], parser=parser)


def test_names_option_wrong_type(xml_books, parser):
    with pytest.raises(TypeError, match=("is not a valid type for names")):
        read_xml(xml_books, names="Col1, Col2, Col3", parser=parser)


# ENCODING


def test_wrong_encoding(xml_baby_names, parser):
    with pytest.raises(UnicodeDecodeError, match=("'utf-8' codec can't decode")):
        read_xml(xml_baby_names, parser=parser)


def test_utf16_encoding(xml_baby_names, parser):
    with pytest.raises(
        UnicodeError,
        match=(
            "UTF-16 stream does not start with BOM|"
            "'utf-16(-le)?' codec can't decode byte"
        ),
    ):
        read_xml(xml_baby_names, encoding="UTF-16", parser=parser)


def test_unknown_encoding(xml_baby_names, parser):
    with pytest.raises(LookupError, match=("unknown encoding: UFT-8")):
        read_xml(xml_baby_names, encoding="UFT-8", parser=parser)


def test_ascii_encoding(xml_baby_names, parser):
    with pytest.raises(UnicodeDecodeError, match=("'ascii' codec can't decode byte")):
        read_xml(xml_baby_names, encoding="ascii", parser=parser)


def test_parser_consistency_with_encoding(xml_baby_names):
    pytest.importorskip("lxml")
    df_xpath_lxml = read_xml(xml_baby_names, parser="lxml", encoding="ISO-8859-1")
    df_xpath_etree = read_xml(xml_baby_names, parser="etree", encoding="iso-8859-1")

    df_iter_lxml = read_xml(
        xml_baby_names,
        parser="lxml",
        encoding="ISO-8859-1",
        iterparse={"row": ["rank", "malename", "femalename"]},
    )
    df_iter_etree = read_xml(
        xml_baby_names,
        parser="etree",
        encoding="ISO-8859-1",
        iterparse={"row": ["rank", "malename", "femalename"]},
    )

    tm.assert_frame_equal(df_xpath_lxml, df_xpath_etree)
    tm.assert_frame_equal(df_xpath_etree, df_iter_etree)
    tm.assert_frame_equal(df_iter_lxml, df_iter_etree)


def test_wrong_encoding_for_lxml():
    pytest.importorskip("lxml")
    # GH#45133
    data = """<data>
  <row>
    <a>c</a>
  </row>
</data>
"""
    with pytest.raises(TypeError, match="encoding None"):
        read_xml(StringIO(data), parser="lxml", encoding=None)


def test_none_encoding_etree():
    # GH#45133
    data = """<data>
  <row>
    <a>c</a>
  </row>
</data>
"""
    result = read_xml(StringIO(data), parser="etree", encoding=None)
    expected = DataFrame({"a": ["c"]})
    tm.assert_frame_equal(result, expected)


# PARSER


@td.skip_if_installed("lxml")
def test_default_parser_no_lxml(xml_books):
    with pytest.raises(
        ImportError, match=("lxml not found, please install or use the etree parser.")
    ):
        read_xml(xml_books)


def test_wrong_parser(xml_books):
    with pytest.raises(
        ValueError, match=("Values for parser can only be lxml or etree.")
    ):
        read_xml(xml_books, parser="bs4")


# STYLESHEET


def test_stylesheet_file(kml_cta_rail_lines, xsl_flatten_doc):
    pytest.importorskip("lxml")
    df_style = read_xml(
        kml_cta_rail_lines,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl_flatten_doc,
    )

    df_iter = read_xml(
        kml_cta_rail_lines,
        iterparse={
            "Placemark": [
                "id",
                "name",
                "styleUrl",
                "extrude",
                "altitudeMode",
                "coordinates",
            ]
        },
    )

    tm.assert_frame_equal(df_kml, df_style)
    tm.assert_frame_equal(df_kml, df_iter)


def test_stylesheet_file_like(kml_cta_rail_lines, xsl_flatten_doc, mode):
    pytest.importorskip("lxml")
    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        df_style = read_xml(
            kml_cta_rail_lines,
            xpath=".//k:Placemark",
            namespaces={"k": "http://www.opengis.net/kml/2.2"},
            stylesheet=f,
        )

    tm.assert_frame_equal(df_kml, df_style)


def test_stylesheet_io(kml_cta_rail_lines, xsl_flatten_doc, mode):
    # note: By default the bodies of untyped functions are not checked,
    # consider using --check-untyped-defs
    pytest.importorskip("lxml")
    xsl_obj: BytesIO | StringIO  # type: ignore[annotation-unchecked]

    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        if mode == "rb":
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())

    df_style = read_xml(
        kml_cta_rail_lines,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl_obj,
    )

    tm.assert_frame_equal(df_kml, df_style)


def test_stylesheet_buffered_reader(kml_cta_rail_lines, xsl_flatten_doc, mode):
    pytest.importorskip("lxml")
    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        xsl_obj = f.read()

    df_style = read_xml(
        kml_cta_rail_lines,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl_obj,
    )

    tm.assert_frame_equal(df_kml, df_style)


def test_style_charset():
    pytest.importorskip("lxml")
    xml = "<中文標籤><row><c1>1</c1><c2>2</c2></row></中文標籤>"

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
 <xsl:output omit-xml-declaration="yes" indent="yes"/>
 <xsl:strip-space elements="*"/>

 <xsl:template match="node()|@*">
     <xsl:copy>
       <xsl:apply-templates select="node()|@*"/>
     </xsl:copy>
 </xsl:template>

 <xsl:template match="中文標籤">
     <根>
       <xsl:apply-templates />
     </根>
 </xsl:template>

</xsl:stylesheet>"""

    df_orig = read_xml(StringIO(xml))
    df_style = read_xml(StringIO(xml), stylesheet=xsl)

    tm.assert_frame_equal(df_orig, df_style)


def test_not_stylesheet(kml_cta_rail_lines, xml_books):
    lxml_etree = pytest.importorskip("lxml.etree")

    with pytest.raises(
        lxml_etree.XSLTParseError, match=("document is not a stylesheet")
    ):
        read_xml(kml_cta_rail_lines, stylesheet=xml_books)


def test_incorrect_xsl_syntax(kml_cta_rail_lines):
    lxml_etree = pytest.importorskip("lxml.etree")

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                              xmlns:k="http://www.opengis.net/kml/2.2"/>
    <xsl:output method="xml" omit-xml-declaration="yes"
                cdata-section-elements="k:description" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <xsl:template match="node()|@*">
     <xsl:copy>
       <xsl:apply-templates select="node()|@*"/>
     </xsl:copy>
    </xsl:template>

    <xsl:template match="k:MultiGeometry|k:LineString">
        <xsl:apply-templates select='*'/>
    </xsl:template>

    <xsl:template match="k:description|k:Snippet|k:Style"/>
</xsl:stylesheet>"""

    with pytest.raises(
        lxml_etree.XMLSyntaxError, match=("Extra content at the end of the document")
    ):
        read_xml(kml_cta_rail_lines, stylesheet=xsl)


def test_incorrect_xsl_eval(kml_cta_rail_lines):
    lxml_etree = pytest.importorskip("lxml.etree")

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                              xmlns:k="http://www.opengis.net/kml/2.2">
    <xsl:output method="xml" omit-xml-declaration="yes"
                cdata-section-elements="k:description" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <xsl:template match="node(*)|@*">
     <xsl:copy>
       <xsl:apply-templates select="node()|@*"/>
     </xsl:copy>
    </xsl:template>

    <xsl:template match="k:MultiGeometry|k:LineString">
        <xsl:apply-templates select='*'/>
    </xsl:template>

    <xsl:template match="k:description|k:Snippet|k:Style"/>
</xsl:stylesheet>"""

    with pytest.raises(lxml_etree.XSLTParseError, match=("failed to compile")):
        read_xml(kml_cta_rail_lines, stylesheet=xsl)


def test_incorrect_xsl_apply(kml_cta_rail_lines):
    lxml_etree = pytest.importorskip("lxml.etree")

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:copy-of select="document('non_existent.xml')/*"/>
        </xsl:copy>
    </xsl:template>
</xsl:stylesheet>"""

    with pytest.raises(lxml_etree.XSLTApplyError, match=("Cannot resolve URI")):
        read_xml(kml_cta_rail_lines, stylesheet=xsl)


def test_wrong_stylesheet(kml_cta_rail_lines, xml_data_path):
    xml_etree = pytest.importorskip("lxml.etree")

    xsl = xml_data_path / "flatten.xsl"

    with pytest.raises(
        xml_etree.XMLSyntaxError,
        match=("Start tag expected, '<' not found"),
    ):
        read_xml(kml_cta_rail_lines, stylesheet=xsl)


def test_stylesheet_file_close(kml_cta_rail_lines, xsl_flatten_doc, mode):
    # note: By default the bodies of untyped functions are not checked,
    # consider using --check-untyped-defs
    pytest.importorskip("lxml")
    xsl_obj: BytesIO | StringIO  # type: ignore[annotation-unchecked]

    with open(xsl_flatten_doc, mode, encoding="utf-8" if mode == "r" else None) as f:
        if mode == "rb":
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())

        read_xml(kml_cta_rail_lines, stylesheet=xsl_obj)

        assert not f.closed


def test_stylesheet_with_etree(kml_cta_rail_lines, xsl_flatten_doc):
    pytest.importorskip("lxml")
    with pytest.raises(
        ValueError, match=("To use stylesheet, you need lxml installed")
    ):
        read_xml(kml_cta_rail_lines, parser="etree", stylesheet=xsl_flatten_doc)


@pytest.mark.parametrize("val", ["", b""])
def test_empty_stylesheet(val):
    pytest.importorskip("lxml")
    msg = (
        "Passing literal xml to 'read_xml' is deprecated and "
        "will be removed in a future version. To read from a "
        "literal string, wrap it in a 'StringIO' object."
    )
    kml = os.path.join("data", "xml", "cta_rail_lines.kml")

    with pytest.raises(FutureWarning, match=msg):
        read_xml(kml, stylesheet=val)


# ITERPARSE
def test_file_like_iterparse(xml_books, parser, mode):
    with open(xml_books, mode, encoding="utf-8" if mode == "r" else None) as f:
        if mode == "r" and parser == "lxml":
            with pytest.raises(
                TypeError, match=("reading file objects must return bytes objects")
            ):
                read_xml(
                    f,
                    parser=parser,
                    iterparse={
                        "book": ["category", "title", "year", "author", "price"]
                    },
                )
            return None
        else:
            df_filelike = read_xml(
                f,
                parser=parser,
                iterparse={"book": ["category", "title", "year", "author", "price"]},
            )

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_filelike, df_expected)


def test_file_io_iterparse(xml_books, parser, mode):
    funcIO = StringIO if mode == "r" else BytesIO
    with open(
        xml_books,
        mode,
        encoding="utf-8" if mode == "r" else None,
    ) as f:
        with funcIO(f.read()) as b:
            if mode == "r" and parser == "lxml":
                with pytest.raises(
                    TypeError, match=("reading file objects must return bytes objects")
                ):
                    read_xml(
                        b,
                        parser=parser,
                        iterparse={
                            "book": ["category", "title", "year", "author", "price"]
                        },
                    )
                return None
            else:
                df_fileio = read_xml(
                    b,
                    parser=parser,
                    iterparse={
                        "book": ["category", "title", "year", "author", "price"]
                    },
                )

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_fileio, df_expected)


@pytest.mark.network
@pytest.mark.single_cpu
def test_url_path_error(parser, httpserver, xml_file):
    with open(xml_file, encoding="utf-8") as f:
        httpserver.serve_content(content=f.read())
        with pytest.raises(
            ParserError, match=("iterparse is designed for large XML files")
        ):
            read_xml(
                httpserver.url,
                parser=parser,
                iterparse={"row": ["shape", "degrees", "sides", "date"]},
            )


def test_compression_error(parser, compression_only):
    with tm.ensure_clean(filename="geom_xml.zip") as path:
        geom_df.to_xml(path, parser=parser, compression=compression_only)

        with pytest.raises(
            ParserError, match=("iterparse is designed for large XML files")
        ):
            read_xml(
                path,
                parser=parser,
                iterparse={"row": ["shape", "degrees", "sides", "date"]},
                compression=compression_only,
            )


def test_wrong_dict_type(xml_books, parser):
    with pytest.raises(TypeError, match="list is not a valid type for iterparse"):
        read_xml(
            xml_books,
            parser=parser,
            iterparse=["category", "title", "year", "author", "price"],
        )


def test_wrong_dict_value(xml_books, parser):
    with pytest.raises(
        TypeError, match="<class 'str'> is not a valid type for value in iterparse"
    ):
        read_xml(xml_books, parser=parser, iterparse={"book": "category"})


def test_bad_xml(parser):
    bad_xml = """\
<?xml version='1.0' encoding='utf-8'?>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
    <date>2020-01-01</date>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
    <date>2021-01-01</date>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
    <date>2022-01-01</date>
  </row>
"""
    with tm.ensure_clean(filename="bad.xml") as path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(bad_xml)

        with pytest.raises(
            SyntaxError,
            match=(
                "Extra content at the end of the document|"
                "junk after document element"
            ),
        ):
            read_xml(
                path,
                parser=parser,
                parse_dates=["date"],
                iterparse={"row": ["shape", "degrees", "sides", "date"]},
            )


def test_comment(parser):
    xml = """\
<!-- comment before root -->
<shapes>
  <!-- comment within root -->
  <shape>
    <name>circle</name>
    <type>2D</type>
  </shape>
  <shape>
    <name>sphere</name>
    <type>3D</type>
    <!-- comment within child -->
  </shape>
  <!-- comment within root -->
</shapes>
<!-- comment after root -->"""

    df_xpath = read_xml(StringIO(xml), xpath=".//shape", parser=parser)

    df_iter = read_xml_iterparse(
        xml, parser=parser, iterparse={"shape": ["name", "type"]}
    )

    df_expected = DataFrame(
        {
            "name": ["circle", "sphere"],
            "type": ["2D", "3D"],
        }
    )

    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtd(parser):
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE non-profits [
    <!ELEMENT shapes (shape*) >
    <!ELEMENT shape ( name, type )>
    <!ELEMENT name (#PCDATA)>
]>
<shapes>
  <shape>
    <name>circle</name>
    <type>2D</type>
  </shape>
  <shape>
    <name>sphere</name>
    <type>3D</type>
  </shape>
</shapes>"""

    df_xpath = read_xml(StringIO(xml), xpath=".//shape", parser=parser)

    df_iter = read_xml_iterparse(
        xml, parser=parser, iterparse={"shape": ["name", "type"]}
    )

    df_expected = DataFrame(
        {
            "name": ["circle", "sphere"],
            "type": ["2D", "3D"],
        }
    )

    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_processing_instruction(parser):
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="style.xsl"?>
<?display table-view?>
<?sort alpha-ascending?>
<?textinfo whitespace is allowed ?>
<?elementnames <shape>, <name>, <type> ?>
<shapes>
  <shape>
    <name>circle</name>
    <type>2D</type>
  </shape>
  <shape>
    <name>sphere</name>
    <type>3D</type>
  </shape>
</shapes>"""

    df_xpath = read_xml(StringIO(xml), xpath=".//shape", parser=parser)

    df_iter = read_xml_iterparse(
        xml, parser=parser, iterparse={"shape": ["name", "type"]}
    )

    df_expected = DataFrame(
        {
            "name": ["circle", "sphere"],
            "type": ["2D", "3D"],
        }
    )

    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_no_result(xml_books, parser):
    with pytest.raises(
        ParserError, match="No result from selected items in iterparse."
    ):
        read_xml(
            xml_books,
            parser=parser,
            iterparse={"node": ["attr1", "elem1", "elem2", "elem3"]},
        )


def test_empty_data(xml_books, parser):
    with pytest.raises(EmptyDataError, match="No columns to parse from file"):
        read_xml(
            xml_books,
            parser=parser,
            iterparse={"book": ["attr1", "elem1", "elem2", "elem3"]},
        )


def test_online_stylesheet():
    pytest.importorskip("lxml")
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<catalog>
  <cd>
    <title>Empire Burlesque</title>
    <artist>Bob Dylan</artist>
    <country>USA</country>
    <company>Columbia</company>
    <price>10.90</price>
    <year>1985</year>
  </cd>
  <cd>
    <title>Hide your heart</title>
    <artist>Bonnie Tyler</artist>
    <country>UK</country>
    <company>CBS Records</company>
    <price>9.90</price>
    <year>1988</year>
  </cd>
  <cd>
    <title>Greatest Hits</title>
    <artist>Dolly Parton</artist>
    <country>USA</country>
    <company>RCA</company>
    <price>9.90</price>
    <year>1982</year>
  </cd>
  <cd>
    <title>Still got the blues</title>
    <artist>Gary Moore</artist>
    <country>UK</country>
    <company>Virgin records</company>
    <price>10.20</price>
    <year>1990</year>
  </cd>
  <cd>
    <title>Eros</title>
    <artist>Eros Ramazzotti</artist>
    <country>EU</country>
    <company>BMG</company>
    <price>9.90</price>
    <year>1997</year>
  </cd>
  <cd>
    <title>One night only</title>
    <artist>Bee Gees</artist>
    <country>UK</country>
    <company>Polydor</company>
    <price>10.90</price>
    <year>1998</year>
  </cd>
  <cd>
    <title>Sylvias Mother</title>
    <artist>Dr.Hook</artist>
    <country>UK</country>
    <company>CBS</company>
    <price>8.10</price>
    <year>1973</year>
  </cd>
  <cd>
    <title>Maggie May</title>
    <artist>Rod Stewart</artist>
    <country>UK</country>
    <company>Pickwick</company>
    <price>8.50</price>
    <year>1990</year>
  </cd>
  <cd>
    <title>Romanza</title>
    <artist>Andrea Bocelli</artist>
    <country>EU</country>
    <company>Polydor</company>
    <price>10.80</price>
    <year>1996</year>
  </cd>
  <cd>
    <title>When a man loves a woman</title>
    <artist>Percy Sledge</artist>
    <country>USA</country>
    <company>Atlantic</company>
    <price>8.70</price>
    <year>1987</year>
  </cd>
  <cd>
    <title>Black angel</title>
    <artist>Savage Rose</artist>
    <country>EU</country>
    <company>Mega</company>
    <price>10.90</price>
    <year>1995</year>
  </cd>
  <cd>
    <title>1999 Grammy Nominees</title>
    <artist>Many</artist>
    <country>USA</country>
    <company>Grammy</company>
    <price>10.20</price>
    <year>1999</year>
  </cd>
  <cd>
    <title>For the good times</title>
    <artist>Kenny Rogers</artist>
    <country>UK</country>
    <company>Mucik Master</company>
    <price>8.70</price>
    <year>1995</year>
  </cd>
  <cd>
    <title>Big Willie style</title>
    <artist>Will Smith</artist>
    <country>USA</country>
    <company>Columbia</company>
    <price>9.90</price>
    <year>1997</year>
  </cd>
  <cd>
    <title>Tupelo Honey</title>
    <artist>Van Morrison</artist>
    <country>UK</country>
    <company>Polydor</company>
    <price>8.20</price>
    <year>1971</year>
  </cd>
  <cd>
    <title>Soulsville</title>
    <artist>Jorn Hoel</artist>
    <country>Norway</country>
    <company>WEA</company>
    <price>7.90</price>
    <year>1996</year>
  </cd>
  <cd>
    <title>The very best of</title>
    <artist>Cat Stevens</artist>
    <country>UK</country>
    <company>Island</company>
    <price>8.90</price>
    <year>1990</year>
  </cd>
  <cd>
    <title>Stop</title>
    <artist>Sam Brown</artist>
    <country>UK</country>
    <company>A and M</company>
    <price>8.90</price>
    <year>1988</year>
  </cd>
  <cd>
    <title>Bridge of Spies</title>
    <artist>T`Pau</artist>
    <country>UK</country>
    <company>Siren</company>
    <price>7.90</price>
    <year>1987</year>
  </cd>
  <cd>
    <title>Private Dancer</title>
    <artist>Tina Turner</artist>
    <country>UK</country>
    <company>Capitol</company>
    <price>8.90</price>
    <year>1983</year>
  </cd>
  <cd>
    <title>Midt om natten</title>
    <artist>Kim Larsen</artist>
    <country>EU</country>
    <company>Medley</company>
    <price>7.80</price>
    <year>1983</year>
  </cd>
  <cd>
    <title>Pavarotti Gala Concert</title>
    <artist>Luciano Pavarotti</artist>
    <country>UK</country>
    <company>DECCA</company>
    <price>9.90</price>
    <year>1991</year>
  </cd>
  <cd>
    <title>The dock of the bay</title>
    <artist>Otis Redding</artist>
    <country>USA</country>
    <COMPANY>Stax Records</COMPANY>
    <PRICE>7.90</PRICE>
    <YEAR>1968</YEAR>
  </cd>
  <cd>
    <title>Picture book</title>
    <artist>Simply Red</artist>
    <country>EU</country>
    <company>Elektra</company>
    <price>7.20</price>
    <year>1985</year>
  </cd>
  <cd>
    <title>Red</title>
    <artist>The Communards</artist>
    <country>UK</country>
    <company>London</company>
    <price>7.80</price>
    <year>1987</year>
  </cd>
  <cd>
    <title>Unchain my heart</title>
    <artist>Joe Cocker</artist>
    <country>USA</country>
    <company>EMI</company>
    <price>8.20</price>
    <year>1987</year>
  </cd>
</catalog>
"""
    xsl = """\
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:template match="/">
<html>
<body>
  <h2>My CD Collection</h2>
  <table border="1">
    <tr bgcolor="#9acd32">
      <th style="text-align:left">Title</th>
      <th style="text-align:left">Artist</th>
    </tr>
    <xsl:for-each select="catalog/cd">
    <tr>
      <td><xsl:value-of select="title"/></td>
      <td><xsl:value-of select="artist"/></td>
    </tr>
    </xsl:for-each>
  </table>
</body>
</html>
</xsl:template>
</xsl:stylesheet>
"""

    df_xsl = read_xml(
        StringIO(xml),
        xpath=".//tr[td and position() <= 6]",
        names=["title", "artist"],
        stylesheet=xsl,
    )

    df_expected = DataFrame(
        {
            "title": {
                0: "Empire Burlesque",
                1: "Hide your heart",
                2: "Greatest Hits",
                3: "Still got the blues",
                4: "Eros",
            },
            "artist": {
                0: "Bob Dylan",
                1: "Bonnie Tyler",
                2: "Dolly Parton",
                3: "Gary Moore",
                4: "Eros Ramazzotti",
            },
        }
    )

    tm.assert_frame_equal(df_expected, df_xsl)


# COMPRESSION


def test_compression_read(parser, compression_only):
    with tm.ensure_clean() as comp_path:
        geom_df.to_xml(
            comp_path, index=False, parser=parser, compression=compression_only
        )

        df_xpath = read_xml(comp_path, parser=parser, compression=compression_only)

        df_iter = read_xml_iterparse_comp(
            comp_path,
            compression_only,
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides"]},
            compression=compression_only,
        )

    tm.assert_frame_equal(df_xpath, geom_df)
    tm.assert_frame_equal(df_iter, geom_df)


def test_wrong_compression(parser, compression, compression_only):
    actual_compression = compression
    attempted_compression = compression_only

    if actual_compression == attempted_compression:
        pytest.skip(f"{actual_compression} == {attempted_compression}")

    errors = {
        "bz2": (OSError, "Invalid data stream"),
        "gzip": (OSError, "Not a gzipped file"),
        "zip": (BadZipFile, "File is not a zip file"),
        "tar": (ReadError, "file could not be opened successfully"),
    }
    zstd = import_optional_dependency("zstandard", errors="ignore")
    if zstd is not None:
        errors["zstd"] = (zstd.ZstdError, "Unknown frame descriptor")
    lzma = import_optional_dependency("lzma", errors="ignore")
    if lzma is not None:
        errors["xz"] = (LZMAError, "Input format not supported by decoder")
    error_cls, error_str = errors[attempted_compression]

    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=actual_compression)

        with pytest.raises(error_cls, match=error_str):
            read_xml(path, parser=parser, compression=attempted_compression)


def test_unsuported_compression(parser):
    with pytest.raises(ValueError, match="Unrecognized compression type"):
        with tm.ensure_clean() as path:
            read_xml(path, parser=parser, compression="7z")


# STORAGE OPTIONS


@pytest.mark.network
@pytest.mark.single_cpu
def test_s3_parser_consistency(s3_public_bucket_with_data, s3so):
    pytest.importorskip("s3fs")
    pytest.importorskip("lxml")
    s3 = f"s3://{s3_public_bucket_with_data.name}/books.xml"

    df_lxml = read_xml(s3, parser="lxml", storage_options=s3so)

    df_etree = read_xml(s3, parser="etree", storage_options=s3so)

    tm.assert_frame_equal(df_lxml, df_etree)


def test_read_xml_nullable_dtypes(
    parser, string_storage, dtype_backend, using_infer_string
):
    # GH#50500
    data = """<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
<row>
  <a>x</a>
  <b>1</b>
  <c>4.0</c>
  <d>x</d>
  <e>2</e>
  <f>4.0</f>
  <g></g>
  <h>True</h>
  <i>False</i>
</row>
<row>
  <a>y</a>
  <b>2</b>
  <c>5.0</c>
  <d></d>
  <e></e>
  <f></f>
  <g></g>
  <h>False</h>
  <i></i>
</row>
</data>"""

    with pd.option_context("mode.string_storage", string_storage):
        result = read_xml(StringIO(data), parser=parser, dtype_backend=dtype_backend)

    if dtype_backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        string_dtype = pd.ArrowDtype(pa.string())
    else:
        string_dtype = pd.StringDtype(string_storage)

    expected = DataFrame(
        {
            "a": Series(["x", "y"], dtype=string_dtype),
            "b": Series([1, 2], dtype="Int64"),
            "c": Series([4.0, 5.0], dtype="Float64"),
            "d": Series(["x", None], dtype=string_dtype),
            "e": Series([2, NA], dtype="Int64"),
            "f": Series([4.0, NA], dtype="Float64"),
            "g": Series([NA, NA], dtype="Int64"),
            "h": Series([True, False], dtype="boolean"),
            "i": Series([False, NA], dtype="boolean"),
        }
    )

    if dtype_backend == "pyarrow":
        pa = pytest.importorskip("pyarrow")
        from pandas.arrays import ArrowExtensionArray

        expected = DataFrame(
            {
                col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                for col in expected.columns
            }
        )
        expected["g"] = ArrowExtensionArray(pa.array([None, None]))

    # the storage of the str columns' Index is also affected by the
    # string_storage setting -> ignore that for checking the result
    tm.assert_frame_equal(result, expected, check_column_type=False)


def test_invalid_dtype_backend():
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    with pytest.raises(ValueError, match=msg):
        read_xml("test", dtype_backend="numpy")