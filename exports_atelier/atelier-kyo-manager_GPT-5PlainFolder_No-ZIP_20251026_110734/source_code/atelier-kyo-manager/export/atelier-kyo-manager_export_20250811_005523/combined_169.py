
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\easter.py ===
# -*- coding: utf-8 -*-
"""
This module offers a generic Easter computing method for any given year, using
Western, Orthodox or Julian algorithms.
"""

import datetime

__all__ = ["easter", "EASTER_JULIAN", "EASTER_ORTHODOX", "EASTER_WESTERN"]

EASTER_JULIAN = 1
EASTER_ORTHODOX = 2
EASTER_WESTERN = 3


def easter(year, method=EASTER_WESTERN):
    """
    This method was ported from the work done by GM Arts,
    on top of the algorithm by Claus Tondering, which was
    based in part on the algorithm of Ouding (1940), as
    quoted in "Explanatory Supplement to the Astronomical
    Almanac", P.  Kenneth Seidelmann, editor.

    This algorithm implements three different Easter
    calculation methods:

    1. Original calculation in Julian calendar, valid in
       dates after 326 AD
    2. Original method, with date converted to Gregorian
       calendar, valid in years 1583 to 4099
    3. Revised method, in Gregorian calendar, valid in
       years 1583 to 4099 as well

    These methods are represented by the constants:

    * ``EASTER_JULIAN   = 1``
    * ``EASTER_ORTHODOX = 2``
    * ``EASTER_WESTERN  = 3``

    The default method is method 3.

    More about the algorithm may be found at:

    `GM Arts: Easter Algorithms <http://www.gmarts.org/index.php?go=415>`_

    and

    `The Calendar FAQ: Easter <https://www.tondering.dk/claus/cal/easter.php>`_

    """

    if not (1 <= method <= 3):
        raise ValueError("invalid method")

    # g - Golden year - 1
    # c - Century
    # h - (23 - Epact) mod 30
    # i - Number of days from March 21 to Paschal Full Moon
    # j - Weekday for PFM (0=Sunday, etc)
    # p - Number of days from March 21 to Sunday on or before PFM
    #     (-6 to 28 methods 1 & 3, to 56 for method 2)
    # e - Extra days to add for method 2 (converting Julian
    #     date to Gregorian date)

    y = year
    g = y % 19
    e = 0
    if method < 3:
        # Old method
        i = (19*g + 15) % 30
        j = (y + y//4 + i) % 7
        if method == 2:
            # Extra dates to convert Julian to Gregorian date
            e = 10
            if y > 1600:
                e = e + y//100 - 16 - (y//100 - 16)//4
    else:
        # New method
        c = y//100
        h = (c - c//4 - (8*c + 13)//25 + 19*g + 15) % 30
        i = h - (h//28)*(1 - (h//28)*(29//(h + 1))*((21 - g)//11))
        j = (y + y//4 + i + 2 - c + c//4) % 7

    # p can be from -6 to 56 corresponding to dates 22 March to 23 May
    # (later dates apply to method 2, although 23 May never actually occurs)
    p = i - j + e
    d = 1 + (p + 27 + (p + 6)//40) % 31
    m = 3 + (p + 26)//30
    return datetime.date(int(y), int(m), int(d))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\shapes.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Alias
)
from openpyxl.descriptors.nested import (
    EmptyTag
)
from openpyxl.drawing.colors import ColorChoiceDescriptor
from openpyxl.drawing.fill import *
from openpyxl.drawing.line import LineProperties
from openpyxl.drawing.geometry import (
    Shape3D,
    Scene3D,
    Transform2D,
    CustomGeometry2D,
    PresetGeometry2D,
)


class GraphicalProperties(Serialisable):

    """
    Somewhat vaguely 21.2.2.197 says this:

    This element specifies the formatting for the parent chart element. The
    custGeom, prstGeom, scene3d, and xfrm elements are not supported. The
    bwMode attribute is not supported.

    This doesn't leave much. And the element is used in different places.
    """

    tagname = "spPr"

    bwMode = NoneSet(values=(['clr', 'auto', 'gray', 'ltGray', 'invGray',
                          'grayWhite', 'blackGray', 'blackWhite', 'black', 'white', 'hidden']
                         )
                 )

    xfrm = Typed(expected_type=Transform2D, allow_none=True)
    transform = Alias('xfrm')
    custGeom = Typed(expected_type=CustomGeometry2D, allow_none=True) # either or
    prstGeom = Typed(expected_type=PresetGeometry2D, allow_none=True)

    # fills one of
    noFill = EmptyTag(namespace=DRAWING_NS)
    solidFill = ColorChoiceDescriptor()
    gradFill = Typed(expected_type=GradientFillProperties, allow_none=True)
    pattFill = Typed(expected_type=PatternFillProperties, allow_none=True)

    ln = Typed(expected_type=LineProperties, allow_none=True)
    line = Alias('ln')
    scene3d = Typed(expected_type=Scene3D, allow_none=True)
    sp3d = Typed(expected_type=Shape3D, allow_none=True)
    shape3D = Alias('sp3d')
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    __elements__ = ('xfrm', 'prstGeom', 'noFill', 'solidFill', 'gradFill', 'pattFill',
                    'ln', 'scene3d', 'sp3d')

    def __init__(self,
                 bwMode=None,
                 xfrm=None,
                 noFill=None,
                 solidFill=None,
                 gradFill=None,
                 pattFill=None,
                 ln=None,
                 scene3d=None,
                 custGeom=None,
                 prstGeom=None,
                 sp3d=None,
                 extLst=None,
                ):
        self.bwMode = bwMode
        self.xfrm = xfrm
        self.noFill = noFill
        self.solidFill = solidFill
        self.gradFill = gradFill
        self.pattFill = pattFill
        if ln is None:
            ln = LineProperties()
        self.ln = ln
        self.custGeom = custGeom
        self.prstGeom = prstGeom
        self.scene3d = scene3d
        self.sp3d = sp3d

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\__init__.py ===
"""
This namespace represents the core functionality that has to be built-in
and deal with private internal data structures. Things in this namespace
are publicly available in either trio, trio.lowlevel, or trio.testing.
"""

import sys

from ._entry_queue import TrioToken
from ._exceptions import (
    BrokenResourceError,
    BusyResourceError,
    Cancelled,
    ClosedResourceError,
    EndOfChannel,
    RunFinishedError,
    TrioInternalError,
    WouldBlock,
)
from ._ki import currently_ki_protected, disable_ki_protection, enable_ki_protection
from ._local import RunVar, RunVarToken
from ._mock_clock import MockClock
from ._parking_lot import (
    ParkingLot,
    ParkingLotStatistics,
    add_parking_lot_breaker,
    remove_parking_lot_breaker,
)

# Imports that always exist
from ._run import (
    TASK_STATUS_IGNORED,
    CancelScope,
    Nursery,
    RunStatistics,
    Task,
    TaskStatus,
    add_instrument,
    checkpoint,
    checkpoint_if_cancelled,
    current_clock,
    current_effective_deadline,
    current_root_task,
    current_statistics,
    current_task,
    current_time,
    current_trio_token,
    in_trio_run,
    in_trio_task,
    notify_closing,
    open_nursery,
    remove_instrument,
    reschedule,
    run,
    spawn_system_task,
    start_guest_run,
    wait_all_tasks_blocked,
    wait_readable,
    wait_writable,
)
from ._thread_cache import start_thread_soon

# Has to come after _run to resolve a circular import
from ._traps import (
    Abort,
    RaiseCancelT,
    cancel_shielded_checkpoint,
    permanently_detach_coroutine_object,
    reattach_detached_coroutine_object,
    temporarily_detach_coroutine_object,
    wait_task_rescheduled,
)
from ._unbounded_queue import UnboundedQueue, UnboundedQueueStatistics

# Windows imports
if sys.platform == "win32":
    from ._run import (
        current_iocp,
        monitor_completion_key,
        readinto_overlapped,
        register_with_iocp,
        wait_overlapped,
        write_overlapped,
    )
# Kqueue imports
elif sys.platform != "linux" and sys.platform != "win32":
    from ._run import current_kqueue, monitor_kevent, wait_kevent

del sys  # It would be better to import sys as _sys, but mypy does not understand it

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\filepost.py ===
from __future__ import annotations

import binascii
import codecs
import os
import typing
from io import BytesIO

from .fields import _TYPE_FIELD_VALUE_TUPLE, RequestField

writer = codecs.lookup("utf-8")[3]

_TYPE_FIELDS_SEQUENCE = typing.Sequence[
    typing.Union[tuple[str, _TYPE_FIELD_VALUE_TUPLE], RequestField]
]
_TYPE_FIELDS = typing.Union[
    _TYPE_FIELDS_SEQUENCE,
    typing.Mapping[str, _TYPE_FIELD_VALUE_TUPLE],
]


def choose_boundary() -> str:
    """
    Our embarrassingly-simple replacement for mimetools.choose_boundary.
    """
    return binascii.hexlify(os.urandom(16)).decode()


def iter_field_objects(fields: _TYPE_FIELDS) -> typing.Iterable[RequestField]:
    """
    Iterate over fields.

    Supports list of (k, v) tuples and dicts, and lists of
    :class:`~urllib3.fields.RequestField`.

    """
    iterable: typing.Iterable[RequestField | tuple[str, _TYPE_FIELD_VALUE_TUPLE]]

    if isinstance(fields, typing.Mapping):
        iterable = fields.items()
    else:
        iterable = fields

    for field in iterable:
        if isinstance(field, RequestField):
            yield field
        else:
            yield RequestField.from_tuples(*field)


def encode_multipart_formdata(
    fields: _TYPE_FIELDS, boundary: str | None = None
) -> tuple[bytes, str]:
    """
    Encode a dictionary of ``fields`` using the multipart/form-data MIME format.

    :param fields:
        Dictionary of fields or list of (key, :class:`~urllib3.fields.RequestField`).
        Values are processed by :func:`urllib3.fields.RequestField.from_tuples`.

    :param boundary:
        If not specified, then a random boundary will be generated using
        :func:`urllib3.filepost.choose_boundary`.
    """
    body = BytesIO()
    if boundary is None:
        boundary = choose_boundary()

    for field in iter_field_objects(fields):
        body.write(f"--{boundary}\r\n".encode("latin-1"))

        writer(body).write(field.render_headers())
        data = field.data

        if isinstance(data, int):
            data = str(data)  # Backwards compatibility

        if isinstance(data, str):
            writer(body).write(data)
        else:
            body.write(data)

        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode("latin-1"))

    content_type = f"multipart/form-data; boundary={boundary}"

    return body.getvalue(), content_type

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_sqlalchemy\track_modifications.py ===
from __future__ import annotations

import typing as t

import sqlalchemy as sa
import sqlalchemy.event as sa_event
import sqlalchemy.orm as sa_orm
from flask import current_app
from flask import has_app_context
from flask.signals import Namespace  # type: ignore[attr-defined]

if t.TYPE_CHECKING:
    from .session import Session

_signals = Namespace()

models_committed = _signals.signal("models-committed")
"""This Blinker signal is sent after the session is committed if there were changed
models in the session.

The sender is the application that emitted the changes. The receiver is passed the
``changes`` argument with a list of tuples in the form ``(instance, operation)``.
The operations are ``"insert"``, ``"update"``, and ``"delete"``.
"""

before_models_committed = _signals.signal("before-models-committed")
"""This signal works exactly like :data:`models_committed` but is emitted before the
commit takes place.
"""


def _listen(session: sa_orm.scoped_session[Session]) -> None:
    sa_event.listen(session, "before_flush", _record_ops, named=True)
    sa_event.listen(session, "before_commit", _record_ops, named=True)
    sa_event.listen(session, "before_commit", _before_commit)
    sa_event.listen(session, "after_commit", _after_commit)
    sa_event.listen(session, "after_rollback", _after_rollback)


def _record_ops(session: Session, **kwargs: t.Any) -> None:
    if not has_app_context():
        return

    if not current_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]:
        return

    for targets, operation in (
        (session.new, "insert"),
        (session.dirty, "update"),
        (session.deleted, "delete"),
    ):
        for target in targets:
            state = sa.inspect(target)
            key = state.identity_key if state.has_identity else id(target)
            session._model_changes[key] = (target, operation)


def _before_commit(session: Session) -> None:
    if not has_app_context():
        return

    app = current_app._get_current_object()  # type: ignore[attr-defined]

    if not app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]:
        return

    if session._model_changes:
        changes = list(session._model_changes.values())
        before_models_committed.send(app, changes=changes)


def _after_commit(session: Session) -> None:
    if not has_app_context():
        return

    app = current_app._get_current_object()  # type: ignore[attr-defined]

    if not app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]:
        return

    if session._model_changes:
        changes = list(session._model_changes.values())
        models_committed.send(app, changes=changes)
        session._model_changes.clear()


def _after_rollback(session: Session) -> None:
    session._model_changes.clear()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\_pickle.py ===
from ._generator import Generator
from ._mt19937 import MT19937
from ._pcg64 import PCG64, PCG64DXSM
from ._philox import Philox
from ._sfc64 import SFC64
from .bit_generator import BitGenerator
from .mtrand import RandomState

BitGenerators = {'MT19937': MT19937,
                 'PCG64': PCG64,
                 'PCG64DXSM': PCG64DXSM,
                 'Philox': Philox,
                 'SFC64': SFC64,
                 }


def __bit_generator_ctor(bit_generator: str | type[BitGenerator] = 'MT19937'):
    """
    Pickling helper function that returns a bit generator object

    Parameters
    ----------
    bit_generator : type[BitGenerator] or str
        BitGenerator class or string containing the name of the BitGenerator

    Returns
    -------
    BitGenerator
        BitGenerator instance
    """
    if isinstance(bit_generator, type):
        bit_gen_class = bit_generator
    elif bit_generator in BitGenerators:
        bit_gen_class = BitGenerators[bit_generator]
    else:
        raise ValueError(
            str(bit_generator) + ' is not a known BitGenerator module.'
        )

    return bit_gen_class()


def __generator_ctor(bit_generator_name="MT19937",
                     bit_generator_ctor=__bit_generator_ctor):
    """
    Pickling helper function that returns a Generator object

    Parameters
    ----------
    bit_generator_name : str or BitGenerator
        String containing the core BitGenerator's name or a
        BitGenerator instance
    bit_generator_ctor : callable, optional
        Callable function that takes bit_generator_name as its only argument
        and returns an instantized bit generator.

    Returns
    -------
    rg : Generator
        Generator using the named core BitGenerator
    """
    if isinstance(bit_generator_name, BitGenerator):
        return Generator(bit_generator_name)
    # Legacy path that uses a bit generator name and ctor
    return Generator(bit_generator_ctor(bit_generator_name))


def __randomstate_ctor(bit_generator_name="MT19937",
                       bit_generator_ctor=__bit_generator_ctor):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    bit_generator_name : str
        String containing the core BitGenerator's name
    bit_generator_ctor : callable, optional
        Callable function that takes bit_generator_name as its only argument
        and returns an instantized bit generator.

    Returns
    -------
    rs : RandomState
        Legacy RandomState using the named core BitGenerator
    """
    if isinstance(bit_generator_name, BitGenerator):
        return RandomState(bit_generator_name)
    return RandomState(bit_generator_ctor(bit_generator_name))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\InferenceSession.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class InferenceSession(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = InferenceSession()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsInferenceSession(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def InferenceSessionBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # InferenceSession
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # InferenceSession
    def OrtVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # InferenceSession
    def Model(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.Model import Model
            obj = Model()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # InferenceSession
    def KernelTypeStrResolver(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.KernelTypeStrResolver import KernelTypeStrResolver
            obj = KernelTypeStrResolver()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def InferenceSessionStart(builder):
    builder.StartObject(4)

def Start(builder):
    InferenceSessionStart(builder)

def InferenceSessionAddOrtVersion(builder, ortVersion):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(ortVersion), 0)

def AddOrtVersion(builder, ortVersion):
    InferenceSessionAddOrtVersion(builder, ortVersion)

def InferenceSessionAddModel(builder, model):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(model), 0)

def AddModel(builder, model):
    InferenceSessionAddModel(builder, model)

def InferenceSessionAddKernelTypeStrResolver(builder, kernelTypeStrResolver):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(kernelTypeStrResolver), 0)

def AddKernelTypeStrResolver(builder, kernelTypeStrResolver):
    InferenceSessionAddKernelTypeStrResolver(builder, kernelTypeStrResolver)

def InferenceSessionEnd(builder):
    return builder.EndObject()

def End(builder):
    return InferenceSessionEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\filesize.py ===
"""Functions for reporting filesizes. Borrowed from https://github.com/PyFilesystem/pyfilesystem2

The functions declared in this module should cover the different
use cases needed to generate a string representation of a file size
using several different units. Since there are many standards regarding
file size units, three different functions have been implemented.

See Also:
    * `Wikipedia: Binary prefix <https://en.wikipedia.org/wiki/Binary_prefix>`_

"""

__all__ = ["decimal"]

from typing import Iterable, List, Optional, Tuple


def _to_str(
    size: int,
    suffixes: Iterable[str],
    base: int,
    *,
    precision: Optional[int] = 1,
    separator: Optional[str] = " ",
) -> str:
    if size == 1:
        return "1 byte"
    elif size < base:
        return f"{size:,} bytes"

    for i, suffix in enumerate(suffixes, 2):  # noqa: B007
        unit = base**i
        if size < unit:
            break
    return "{:,.{precision}f}{separator}{}".format(
        (base * size / unit),
        suffix,
        precision=precision,
        separator=separator,
    )


def pick_unit_and_suffix(size: int, suffixes: List[str], base: int) -> Tuple[int, str]:
    """Pick a suffix and base for the given size."""
    for i, suffix in enumerate(suffixes):
        unit = base**i
        if size < unit * base:
            break
    return unit, suffix


def decimal(
    size: int,
    *,
    precision: Optional[int] = 1,
    separator: Optional[str] = " ",
) -> str:
    """Convert a filesize in to a string (powers of 1000, SI prefixes).

    In this convention, ``1000 B = 1 kB``.

    This is typically the format used to advertise the storage
    capacity of USB flash drives and the like (*256 MB* meaning
    actually a storage capacity of more than *256 000 000 B*),
    or used by **Mac OS X** since v10.6 to report file sizes.

    Arguments:
        int (size): A file size.
        int (precision): The number of decimal places to include (default = 1).
        str (separator): The string to separate the value from the units (default = " ").

    Returns:
        `str`: A string containing a abbreviated file size and units.

    Example:
        >>> filesize.decimal(30000)
        '30.0 kB'
        >>> filesize.decimal(30000, precision=2, separator="")
        '30.00kB'

    """
    return _to_str(
        size,
        ("kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"),
        1000,
        precision=precision,
        separator=separator,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mssql\__init__.py ===
# dialects/mssql/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from . import aioodbc  # noqa
from . import base  # noqa
from . import pymssql  # noqa
from . import pyodbc  # noqa
from .base import BIGINT
from .base import BINARY
from .base import BIT
from .base import CHAR
from .base import DATE
from .base import DATETIME
from .base import DATETIME2
from .base import DATETIMEOFFSET
from .base import DECIMAL
from .base import DOUBLE_PRECISION
from .base import FLOAT
from .base import IMAGE
from .base import INTEGER
from .base import JSON
from .base import MONEY
from .base import NCHAR
from .base import NTEXT
from .base import NUMERIC
from .base import NVARCHAR
from .base import REAL
from .base import ROWVERSION
from .base import SMALLDATETIME
from .base import SMALLINT
from .base import SMALLMONEY
from .base import SQL_VARIANT
from .base import TEXT
from .base import TIME
from .base import TIMESTAMP
from .base import TINYINT
from .base import UNIQUEIDENTIFIER
from .base import VARBINARY
from .base import VARCHAR
from .base import XML
from ...sql import try_cast


base.dialect = dialect = pyodbc.dialect


__all__ = (
    "JSON",
    "INTEGER",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "VARCHAR",
    "NVARCHAR",
    "CHAR",
    "NCHAR",
    "TEXT",
    "NTEXT",
    "DECIMAL",
    "NUMERIC",
    "FLOAT",
    "DATETIME",
    "DATETIME2",
    "DATETIMEOFFSET",
    "DATE",
    "DOUBLE_PRECISION",
    "TIME",
    "SMALLDATETIME",
    "BINARY",
    "VARBINARY",
    "BIT",
    "REAL",
    "IMAGE",
    "TIMESTAMP",
    "ROWVERSION",
    "MONEY",
    "SMALLMONEY",
    "UNIQUEIDENTIFIER",
    "SQL_VARIANT",
    "XML",
    "dialect",
    "try_cast",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\qho_1d.py ===
from sympy.core import S, pi, Rational
from sympy.functions import hermite, sqrt, exp, factorial, Abs
from sympy.physics.quantum.constants import hbar


def psi_n(n, x, m, omega):
    """
    Returns the wavefunction psi_{n} for the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        the "nodal" quantum number.  Corresponds to the number of nodes in the
        wavefunction.  ``n >= 0``
    x :
        x coordinate.
    m :
        Mass of the particle.
    omega :
        Angular frequency of the oscillator.

    Examples
    ========

    >>> from sympy.physics.qho_1d import psi_n
    >>> from sympy.abc import m, x, omega
    >>> psi_n(0, x, m, omega)
    (m*omega)**(1/4)*exp(-m*omega*x**2/(2*hbar))/(hbar**(1/4)*pi**(1/4))

    """

    # sympify arguments
    n, x, m, omega = map(S, [n, x, m, omega])
    nu = m * omega / hbar
    # normalization coefficient
    C = (nu/pi)**Rational(1, 4) * sqrt(1/(2**n*factorial(n)))

    return C * exp(-nu* x**2 /2) * hermite(n, sqrt(nu)*x)


def E_n(n, omega):
    """
    Returns the Energy of the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    omega :
        The harmonic oscillator angular frequency.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_n = hbar * omega*(n + 1/2)

    Examples
    ========

    >>> from sympy.physics.qho_1d import E_n
    >>> from sympy.abc import x, omega
    >>> E_n(x, omega)
    hbar*omega*(x + 1/2)
    """

    return hbar * omega * (n + S.Half)


def coherent_state(n, alpha):
    """
    Returns <n|alpha> for the coherent states of 1D harmonic oscillator.
    See https://en.wikipedia.org/wiki/Coherent_states

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    alpha :
        The eigen value of annihilation operator.
    """

    return exp(- Abs(alpha)**2/2)*(alpha**n)/sqrt(factorial(n))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\conventions.py ===
"""
A few practical conventions common to all printers.
"""

import re

from collections.abc import Iterable
from sympy.core.function import Derivative

_name_with_digits_p = re.compile(r'^([^\W\d_]+)(\d+)$', re.UNICODE)


def split_super_sub(text):
    """Split a symbol name into a name, superscripts and subscripts

    The first part of the symbol name is considered to be its actual
    'name', followed by super- and subscripts. Each superscript is
    preceded with a "^" character or by "__". Each subscript is preceded
    by a "_" character.  The three return values are the actual name, a
    list with superscripts and a list with subscripts.

    Examples
    ========

    >>> from sympy.printing.conventions import split_super_sub
    >>> split_super_sub('a_x^1')
    ('a', ['1'], ['x'])
    >>> split_super_sub('var_sub1__sup_sub2')
    ('var', ['sup'], ['sub1', 'sub2'])

    """
    if not text:
        return text, [], []

    pos = 0
    name = None
    supers = []
    subs = []
    while pos < len(text):
        start = pos + 1
        if text[pos:pos + 2] == "__":
            start += 1
        pos_hat = text.find("^", start)
        if pos_hat < 0:
            pos_hat = len(text)
        pos_usc = text.find("_", start)
        if pos_usc < 0:
            pos_usc = len(text)
        pos_next = min(pos_hat, pos_usc)
        part = text[pos:pos_next]
        pos = pos_next
        if name is None:
            name = part
        elif part.startswith("^"):
            supers.append(part[1:])
        elif part.startswith("__"):
            supers.append(part[2:])
        elif part.startswith("_"):
            subs.append(part[1:])
        else:
            raise RuntimeError("This should never happen.")

    # Make a little exception when a name ends with digits, i.e. treat them
    # as a subscript too.
    m = _name_with_digits_p.match(name)
    if m:
        name, sub = m.groups()
        subs.insert(0, sub)

    return name, supers, subs


def requires_partial(expr):
    """Return whether a partial derivative symbol is required for printing

    This requires checking how many free variables there are,
    filtering out the ones that are integers. Some expressions do not have
    free variables. In that case, check its variable list explicitly to
    get the context of the expression.
    """

    if isinstance(expr, Derivative):
        return requires_partial(expr.expr)

    if not isinstance(expr.free_symbols, Iterable):
        return len(set(expr.variables)) > 1

    return sum(not s.is_integer for s in expr.free_symbols) > 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_repl.py ===
from __future__ import annotations

import ast
import contextlib
import inspect
import sys
import types
import warnings
from code import InteractiveConsole

import outcome

import trio
import trio.lowlevel
from trio._util import final


@final
class TrioInteractiveConsole(InteractiveConsole):
    def __init__(self, repl_locals: dict[str, object] | None = None) -> None:
        super().__init__(locals=repl_locals)
        self.compile.compiler.flags |= ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

    def runcode(self, code: types.CodeType) -> None:
        # https://github.com/python/typeshed/issues/13768
        func = types.FunctionType(code, self.locals)  # type: ignore[arg-type]
        if inspect.iscoroutinefunction(func):
            result = trio.from_thread.run(outcome.acapture, func)
        else:
            result = trio.from_thread.run_sync(outcome.capture, func)
        if isinstance(result, outcome.Error):
            # If it is SystemExit, quit the repl. Otherwise, print the traceback.
            # If there is a SystemExit inside a BaseExceptionGroup, it probably isn't
            # the user trying to quit the repl, but rather an error in the code. So, we
            # don't try to inspect groups for SystemExit. Instead, we just print and
            # return to the REPL.
            if isinstance(result.error, SystemExit):
                raise result.error
            else:
                # Inline our own version of self.showtraceback that can use
                # outcome.Error.error directly to print clean tracebacks.
                # This also means overriding self.showtraceback does nothing.
                sys.last_type, sys.last_value = type(result.error), result.error
                sys.last_traceback = result.error.__traceback__
                # see https://docs.python.org/3/library/sys.html#sys.last_exc
                if sys.version_info >= (3, 12):
                    sys.last_exc = result.error

                # We always use sys.excepthook, unlike other implementations.
                # This means that overriding self.write also does nothing to tbs.
                sys.excepthook(sys.last_type, sys.last_value, sys.last_traceback)


async def run_repl(console: TrioInteractiveConsole) -> None:
    banner = (
        f"trio REPL {sys.version} on {sys.platform}\n"
        f'Use "await" directly instead of "trio.run()".\n'
        f'Type "help", "copyright", "credits" or "license" '
        f"for more information.\n"
        f'{getattr(sys, "ps1", ">>> ")}import trio'
    )
    try:
        await trio.to_thread.run_sync(console.interact, banner)
    finally:
        warnings.filterwarnings(
            "ignore",
            message=r"^coroutine .* was never awaited$",
            category=RuntimeWarning,
        )


def main(original_locals: dict[str, object]) -> None:
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401

    repl_locals: dict[str, object] = {"trio": trio}
    for key in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__file__",
    }:
        repl_locals[key] = original_locals[key]

    console = TrioInteractiveConsole(repl_locals)
    trio.run(run_repl, console)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wsproto\utilities.py ===
"""
wsproto/utilities
~~~~~~~~~~~~~~~~~

Utility functions that do not belong in a separate module.
"""
import base64
import hashlib
import os
from typing import Dict, List, Optional, Union

from h11._headers import Headers as H11Headers

from .events import Event
from .typing import Headers

# RFC6455, Section 1.3 - Opening Handshake
ACCEPT_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class ProtocolError(Exception):
    pass


class LocalProtocolError(ProtocolError):
    """Indicates an error due to local/programming errors.

    This is raised when the connection is asked to do something that
    is either incompatible with the state or the websocket standard.

    """

    pass  # noqa


class RemoteProtocolError(ProtocolError):
    """Indicates an error due to the remote's actions.

    This is raised when processing the bytes from the remote if the
    remote has sent data that is incompatible with the websocket
    standard.

    .. attribute:: event_hint

       This is a suggested wsproto Event to send to the client based
       on the error. It could be None if no hint is available.

    """

    def __init__(self, message: str, event_hint: Optional[Event] = None) -> None:
        self.event_hint = event_hint
        super().__init__(message)


# Some convenience utilities for working with HTTP headers
def normed_header_dict(h11_headers: Union[Headers, H11Headers]) -> Dict[bytes, bytes]:
    # This mangles Set-Cookie headers. But it happens that we don't care about
    # any of those, so it's OK. For every other HTTP header, if there are
    # multiple instances then you're allowed to join them together with
    # commas.
    name_to_values: Dict[bytes, List[bytes]] = {}
    for name, value in h11_headers:
        name_to_values.setdefault(name, []).append(value)
    name_to_normed_value = {}
    for name, values in name_to_values.items():
        name_to_normed_value[name] = b", ".join(values)
    return name_to_normed_value


# We use this for parsing the proposed protocol list, and for parsing the
# proposed and accepted extension lists. For the proposed protocol list it's
# fine, because the ABNF is just 1#token. But for the extension lists, it's
# wrong, because those can contain quoted strings, which can in turn contain
# commas. XX FIXME
def split_comma_header(value: bytes) -> List[str]:
    return [piece.decode("ascii").strip() for piece in value.split(b",")]


def generate_nonce() -> bytes:
    # os.urandom may be overkill for this use case, but I don't think this
    # is a bottleneck, and better safe than sorry...
    return base64.b64encode(os.urandom(16))


def generate_accept_token(token: bytes) -> bytes:
    accept_token = token + ACCEPT_GUID
    accept_token = hashlib.sha1(accept_token).digest()
    return base64.b64encode(accept_token)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\where.py ===
import onnx

from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType, attribute_to_kwarg, ms_domain
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase


class QLinearWhere(QuantOperatorBase):
    def should_quantize(self):
        return True

    def quantize(self):
        node = self.node
        assert node.op_type == "Where"
        if not self.quantizer.force_quantize_no_input_check:
            self.quantizer.new_nodes += [node]
            return
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
        ) = self.quantizer.quantize_activation(node, [1, 2])
        if not data_found or q_input_names is None:
            return super().quantize()
        qlinear_output = node.output[0] + TENSOR_NAME_QUANT_SUFFIX
        qlinear_output_name = node.name + "_quant" if node.name else ""

        q_output = QuantizedValue(
            node.output[0],
            qlinear_output,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain

        qlwhere_inputs = [
            node.input[0],
            q_input_names[0],
            scale_names[0],
            zero_point_names[0],
            q_input_names[1],
            scale_names[1],
            zero_point_names[1],
            output_scale_name,
            output_zp_name,
        ]
        qlwhere_node = onnx.helper.make_node(
            "QLinearWhere", qlwhere_inputs, [qlinear_output], qlinear_output_name, **kwargs
        )

        self.quantizer.new_nodes += nodes
        self.quantizer.new_nodes += [qlwhere_node]


class QDQWhere(QDQOperatorBase):
    def quantize(self):
        node = self.node
        assert node.op_type == "Where"
        if self.quantizer.force_quantize_no_input_check:
            if not self.quantizer.is_tensor_quantized(node.input[1]):
                self.quantizer.quantize_activation_tensor(node.input[1])
            if not self.quantizer.is_tensor_quantized(node.input[2]):
                self.quantizer.quantize_activation_tensor(node.input[2])
            if not self.disable_qdq_for_node_output:
                for output in node.output:
                    self.quantizer.quantize_activation_tensor(output)
        elif (
            self.quantizer.is_tensor_quantized(node.input[1])
            and self.quantizer.is_tensor_quantized(node.input[2])
            and not self.disable_qdq_for_node_output
        ):
            for output in node.output:
                self.quantizer.quantize_activation_tensor(output)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\dataframe.py ===
# Copyright (c) 2010-2024 openpyxl

from itertools import accumulate
import operator
import numpy
from openpyxl.compat.product import prod


def dataframe_to_rows(df, index=True, header=True):
    """
    Convert a Pandas dataframe into something suitable for passing into a worksheet.
    If index is True then the index will be included, starting one row below the header.
    If header is True then column headers will be included starting one column to the right.
    Formatting should be done by client code.
    """
    from pandas import Timestamp

    if header:
        if df.columns.nlevels > 1:
            rows = expand_index(df.columns, header)
        else:
            rows = [list(df.columns.values)]
        for row in rows:
            n = []
            for v in row:
                if isinstance(v, numpy.datetime64):
                    v = Timestamp(v)
                n.append(v)
            row = n
            if index:
                row = [None]*df.index.nlevels + row
            yield row

    if index:
        yield df.index.names

    expanded = ([v] for v in df.index)
    if df.index.nlevels > 1:
        expanded = expand_index(df.index)

    # Using the expanded index is preferable to df.itertuples(index=True) so that we have 'None' inserted where applicable
    for (df_index, row) in zip(expanded, df.itertuples(index=False)):
        row = list(row)
        if index:
            row = df_index + row
        yield row


def expand_index(index, header=False):
    """
    Expand axis or column Multiindex
    For columns use header = True
    For axes use header = False (default)
    """

    # For each element of the index, zip the members with the previous row
    # If the 2 elements of the zipped list do not match, we can insert the new value into the row
    # or if an earlier member was different, all later members should be added to the row
    values = list(index.values)
    previous_value = [None] * len(values[0])
    result = []

    for value in values:
        row = [None] * len(value)

        # Once there's a difference in member of an index with the prior index, we need to store all subsequent members in the row
        prior_change = False
        for idx, (current_index_member, previous_index_member) in enumerate(zip(value, previous_value)):

            if current_index_member != previous_index_member or prior_change:
                row[idx] = current_index_member
                prior_change = True

        previous_value = value

        # If this is for a row index, we're already returning a row so just yield
        if not header:
            yield row
        else:
            result.append(row)

    # If it's for a header, we need to transpose to get it in row order
    # Example: result = [['A', 'A'], [None, 'B']] -> [['A', None], ['A', 'B']]
    if header:
        result = numpy.array(result).transpose().tolist()
        for row in result:
            yield row

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\xml\functions.py ===
# Copyright (c) 2010-2024 openpyxl

"""
XML compatibility functions
"""

# Python stdlib imports
import re
from functools import partial

from openpyxl import DEFUSEDXML, LXML

if LXML is True:
    from lxml.etree import (
    Element,
    SubElement,
    register_namespace,
    QName,
    xmlfile,
    XMLParser,
    )
    from lxml.etree import fromstring, tostring
    # do not resolve entities
    safe_parser = XMLParser(resolve_entities=False)
    fromstring = partial(fromstring, parser=safe_parser)

else:
    from xml.etree.ElementTree import (
    Element,
    SubElement,
    fromstring,
    tostring,
    QName,
    register_namespace
    )
    from et_xmlfile import xmlfile
    if DEFUSEDXML is True:
        from defusedxml.ElementTree import fromstring

from xml.etree.ElementTree import iterparse
if DEFUSEDXML is True:
    from defusedxml.ElementTree import iterparse

from openpyxl.xml.constants import (
    CHART_NS,
    DRAWING_NS,
    SHEET_DRAWING_NS,
    CHART_DRAWING_NS,
    SHEET_MAIN_NS,
    REL_NS,
    VTYPES_NS,
    COREPROPS_NS,
    CUSTPROPS_NS,
    DCTERMS_NS,
    DCTERMS_PREFIX,
    XML_NS
)

register_namespace(DCTERMS_PREFIX, DCTERMS_NS)
register_namespace('dcmitype', 'http://purl.org/dc/dcmitype/')
register_namespace('cp', COREPROPS_NS)
register_namespace('c', CHART_NS)
register_namespace('a', DRAWING_NS)
register_namespace('s', SHEET_MAIN_NS)
register_namespace('r', REL_NS)
register_namespace('vt', VTYPES_NS)
register_namespace('xdr', SHEET_DRAWING_NS)
register_namespace('cdr', CHART_DRAWING_NS)
register_namespace('xml', XML_NS)
register_namespace('cust', CUSTPROPS_NS)


tostring = partial(tostring, encoding="utf-8")

NS_REGEX = re.compile("({(?P<namespace>.*)})?(?P<localname>.*)")

def localname(node):
    if callable(node.tag):
        return "comment"
    m = NS_REGEX.match(node.tag)
    return m.group('localname')


def whitespace(node):
    stripped = node.text.strip()
    if stripped and node.text != stripped:
        node.set("{%s}space" % XML_NS, "preserve")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\groupby\categorical.py ===
from __future__ import annotations

import numpy as np

from pandas.core.algorithms import unique1d
from pandas.core.arrays.categorical import (
    Categorical,
    CategoricalDtype,
    recode_for_categories,
)


def recode_for_groupby(
    c: Categorical, sort: bool, observed: bool
) -> tuple[Categorical, Categorical | None]:
    """
    Code the categories to ensure we can groupby for categoricals.

    If observed=True, we return a new Categorical with the observed
    categories only.

    If sort=False, return a copy of self, coded with categories as
    returned by .unique(), followed by any categories not appearing in
    the data. If sort=True, return self.

    This method is needed solely to ensure the categorical index of the
    GroupBy result has categories in the order of appearance in the data
    (GH-8868).

    Parameters
    ----------
    c : Categorical
    sort : bool
        The value of the sort parameter groupby was called with.
    observed : bool
        Account only for the observed values

    Returns
    -------
    Categorical
        If sort=False, the new categories are set to the order of
        appearance in codes (unless ordered=True, in which case the
        original order is preserved), followed by any unrepresented
        categories in the original order.
    Categorical or None
        If we are observed, return the original categorical, otherwise None
    """
    # we only care about observed values
    if observed:
        # In cases with c.ordered, this is equivalent to
        #  return c.remove_unused_categories(), c

        unique_codes = unique1d(c.codes)

        take_codes = unique_codes[unique_codes != -1]
        if sort:
            take_codes = np.sort(take_codes)

        # we recode according to the uniques
        categories = c.categories.take(take_codes)
        codes = recode_for_categories(c.codes, c.categories, categories)

        # return a new categorical that maps our new codes
        # and categories
        dtype = CategoricalDtype(categories, ordered=c.ordered)
        return Categorical._simple_new(codes, dtype=dtype), c

    # Already sorted according to c.categories; all is fine
    if sort:
        return c, None

    # sort=False should order groups in as-encountered order (GH-8868)

    # xref GH:46909: Re-ordering codes faster than using (set|add|reorder)_categories
    all_codes = np.arange(c.categories.nunique())
    # GH 38140: exclude nan from indexer for categories
    unique_notnan_codes = unique1d(c.codes[c.codes != -1])
    if sort:
        unique_notnan_codes = np.sort(unique_notnan_codes)
    if len(all_codes) > len(unique_notnan_codes):
        # GH 13179: All categories need to be present, even if missing from the data
        missing_codes = np.setdiff1d(all_codes, unique_notnan_codes, assume_unique=True)
        take_codes = np.concatenate((unique_notnan_codes, missing_codes))
    else:
        take_codes = unique_notnan_codes

    return Categorical(c, c.unique().categories.take(take_codes)), None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_libs\tslibs\__init__.py ===
__all__ = [
    "dtypes",
    "localize_pydatetime",
    "NaT",
    "NaTType",
    "iNaT",
    "nat_strings",
    "OutOfBoundsDatetime",
    "OutOfBoundsTimedelta",
    "IncompatibleFrequency",
    "Period",
    "Resolution",
    "Timedelta",
    "normalize_i8_timestamps",
    "is_date_array_normalized",
    "dt64arr_to_periodarr",
    "delta_to_nanoseconds",
    "ints_to_pydatetime",
    "ints_to_pytimedelta",
    "get_resolution",
    "Timestamp",
    "tz_convert_from_utc_single",
    "tz_convert_from_utc",
    "to_offset",
    "Tick",
    "BaseOffset",
    "tz_compare",
    "is_unitless",
    "astype_overflowsafe",
    "get_unit_from_dtype",
    "periods_per_day",
    "periods_per_second",
    "guess_datetime_format",
    "add_overflowsafe",
    "get_supported_dtype",
    "is_supported_dtype",
]

from pandas._libs.tslibs import dtypes  # pylint: disable=import-self
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.dtypes import (
    Resolution,
    periods_per_day,
    periods_per_second,
)
from pandas._libs.tslibs.nattype import (
    NaT,
    NaTType,
    iNaT,
    nat_strings,
)
from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    add_overflowsafe,
    astype_overflowsafe,
    get_supported_dtype,
    is_supported_dtype,
    is_unitless,
    py_get_unit_from_dtype as get_unit_from_dtype,
)
from pandas._libs.tslibs.offsets import (
    BaseOffset,
    Tick,
    to_offset,
)
from pandas._libs.tslibs.parsing import guess_datetime_format
from pandas._libs.tslibs.period import (
    IncompatibleFrequency,
    Period,
)
from pandas._libs.tslibs.timedeltas import (
    Timedelta,
    delta_to_nanoseconds,
    ints_to_pytimedelta,
)
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._libs.tslibs.timezones import tz_compare
from pandas._libs.tslibs.tzconversion import tz_convert_from_utc_single
from pandas._libs.tslibs.vectorized import (
    dt64arr_to_periodarr,
    get_resolution,
    ints_to_pydatetime,
    is_date_array_normalized,
    normalize_i8_timestamps,
    tz_convert_from_utc,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\direct_url_helpers.py ===
from typing import Optional

from pip._internal.models.direct_url import ArchiveInfo, DirectUrl, DirInfo, VcsInfo
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import vcs


def direct_url_as_pep440_direct_reference(direct_url: DirectUrl, name: str) -> str:
    """Convert a DirectUrl to a pip requirement string."""
    direct_url.validate()  # if invalid, this is a pip bug
    requirement = name + " @ "
    fragments = []
    if isinstance(direct_url.info, VcsInfo):
        requirement += (
            f"{direct_url.info.vcs}+{direct_url.url}@{direct_url.info.commit_id}"
        )
    elif isinstance(direct_url.info, ArchiveInfo):
        requirement += direct_url.url
        if direct_url.info.hash:
            fragments.append(direct_url.info.hash)
    else:
        assert isinstance(direct_url.info, DirInfo)
        requirement += direct_url.url
    if direct_url.subdirectory:
        fragments.append("subdirectory=" + direct_url.subdirectory)
    if fragments:
        requirement += "#" + "&".join(fragments)
    return requirement


def direct_url_for_editable(source_dir: str) -> DirectUrl:
    return DirectUrl(
        url=path_to_url(source_dir),
        info=DirInfo(editable=True),
    )


def direct_url_from_link(
    link: Link, source_dir: Optional[str] = None, link_is_in_wheel_cache: bool = False
) -> DirectUrl:
    if link.is_vcs:
        vcs_backend = vcs.get_backend_for_scheme(link.scheme)
        assert vcs_backend
        url, requested_revision, _ = vcs_backend.get_url_rev_and_auth(
            link.url_without_fragment
        )
        # For VCS links, we need to find out and add commit_id.
        if link_is_in_wheel_cache:
            # If the requested VCS link corresponds to a cached
            # wheel, it means the requested revision was an
            # immutable commit hash, otherwise it would not have
            # been cached. In that case we don't have a source_dir
            # with the VCS checkout.
            assert requested_revision
            commit_id = requested_revision
        else:
            # If the wheel was not in cache, it means we have
            # had to checkout from VCS to build and we have a source_dir
            # which we can inspect to find out the commit id.
            assert source_dir
            commit_id = vcs_backend.get_revision(source_dir)
        return DirectUrl(
            url=url,
            info=VcsInfo(
                vcs=vcs_backend.name,
                commit_id=commit_id,
                requested_revision=requested_revision,
            ),
            subdirectory=link.subdirectory_fragment,
        )
    elif link.is_existing_dir():
        return DirectUrl(
            url=link.url_without_fragment,
            info=DirInfo(),
            subdirectory=link.subdirectory_fragment,
        )
    else:
        hash = None
        hash_name = link.hash_name
        if hash_name:
            hash = f"{hash_name}={link.hash}"
        return DirectUrl(
            url=link.url_without_fragment,
            info=ArchiveInfo(hash=hash),
            subdirectory=link.subdirectory_fragment,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\entrypoints.py ===
import itertools
import os
import shutil
import sys
from typing import List, Optional

from pip._internal.cli.main import main
from pip._internal.utils.compat import WINDOWS

_EXECUTABLE_NAMES = [
    "pip",
    f"pip{sys.version_info.major}",
    f"pip{sys.version_info.major}.{sys.version_info.minor}",
]
if WINDOWS:
    _allowed_extensions = {"", ".exe"}
    _EXECUTABLE_NAMES = [
        "".join(parts)
        for parts in itertools.product(_EXECUTABLE_NAMES, _allowed_extensions)
    ]


def _wrapper(args: Optional[List[str]] = None) -> int:
    """Central wrapper for all old entrypoints.

    Historically pip has had several entrypoints defined. Because of issues
    arising from PATH, sys.path, multiple Pythons, their interactions, and most
    of them having a pip installed, users suffer every time an entrypoint gets
    moved.

    To alleviate this pain, and provide a mechanism for warning users and
    directing them to an appropriate place for help, we now define all of
    our old entrypoints as wrappers for the current one.
    """
    sys.stderr.write(
        "WARNING: pip is being invoked by an old script wrapper. This will "
        "fail in a future version of pip.\n"
        "Please see https://github.com/pypa/pip/issues/5599 for advice on "
        "fixing the underlying issue.\n"
        "To avoid this problem you can invoke Python with '-m pip' instead of "
        "running pip directly.\n"
    )
    return main(args)


def get_best_invocation_for_this_pip() -> str:
    """Try to figure out the best way to invoke pip in the current environment."""
    binary_directory = "Scripts" if WINDOWS else "bin"
    binary_prefix = os.path.join(sys.prefix, binary_directory)

    # Try to use pip[X[.Y]] names, if those executables for this environment are
    # the first on PATH with that name.
    path_parts = os.path.normcase(os.environ.get("PATH", "")).split(os.pathsep)
    exe_are_in_PATH = os.path.normcase(binary_prefix) in path_parts
    if exe_are_in_PATH:
        for exe_name in _EXECUTABLE_NAMES:
            found_executable = shutil.which(exe_name)
            binary_executable = os.path.join(binary_prefix, exe_name)
            if (
                found_executable
                and os.path.exists(binary_executable)
                and os.path.samefile(
                    found_executable,
                    binary_executable,
                )
            ):
                return exe_name

    # Use the `-m` invocation, if there's no "nice" invocation.
    return f"{get_best_invocation_for_this_python()} -m pip"


def get_best_invocation_for_this_python() -> str:
    """Try to figure out the best way to invoke the current Python."""
    exe = sys.executable
    exe_name = os.path.basename(exe)

    # Try to use the basename, if it's the first executable.
    found_executable = shutil.which(exe_name)
    # Virtual environments often symlink to their parent Python binaries, but we don't
    # want to treat the Python binaries as equivalent when the environment's Python is
    # not on PATH (not activated). Thus, we don't follow symlinks.
    if found_executable and os.path.samestat(os.lstat(found_executable), os.lstat(exe)):
        return exe_name

    # Use the full executable name, because we couldn't find something simpler.
    return exe

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\from_matrix_to_array.py ===
from sympy import KroneckerProduct
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct)
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.tensor.array.expressions.array_expressions import \
    ArrayElementwiseApplyFunc, _array_tensor_product, _array_contraction, \
    _array_diagonal, _array_add, _permute_dims, Reshape


def convert_matrix_to_array(expr: Basic) -> Basic:
    if isinstance(expr, MatMul):
        args_nonmat = []
        args = []
        for arg in expr.args:
            if isinstance(arg, MatrixExpr):
                args.append(arg)
            else:
                args_nonmat.append(convert_matrix_to_array(arg))
        contractions = [(2*i+1, 2*i+2) for i in range(len(args)-1)]
        scalar = _array_tensor_product(*args_nonmat) if args_nonmat else S.One
        if scalar == 1:
            tprod = _array_tensor_product(
                *[convert_matrix_to_array(arg) for arg in args])
        else:
            tprod = _array_tensor_product(
                scalar,
                *[convert_matrix_to_array(arg) for arg in args])
        return _array_contraction(
                tprod,
                *contractions
        )
    elif isinstance(expr, MatAdd):
        return _array_add(
                *[convert_matrix_to_array(arg) for arg in expr.args]
        )
    elif isinstance(expr, Transpose):
        return _permute_dims(
                convert_matrix_to_array(expr.args[0]), [1, 0]
        )
    elif isinstance(expr, Trace):
        inner_expr: MatrixExpr = convert_matrix_to_array(expr.arg) # type: ignore
        return _array_contraction(inner_expr, (0, len(inner_expr.shape) - 1))
    elif isinstance(expr, Mul):
        return _array_tensor_product(*[convert_matrix_to_array(i) for i in expr.args])
    elif isinstance(expr, Pow):
        base = convert_matrix_to_array(expr.base)
        if (expr.exp > 0) == True:
            return _array_tensor_product(*[base for i in range(expr.exp)])
        else:
            return expr
    elif isinstance(expr, MatPow):
        base = convert_matrix_to_array(expr.base)
        if expr.exp.is_Integer != True:
            b = symbols("b", cls=Dummy)
            return ArrayElementwiseApplyFunc(Lambda(b, b**expr.exp), convert_matrix_to_array(base))
        elif (expr.exp > 0) == True:
            return convert_matrix_to_array(MatMul.fromiter(base for i in range(expr.exp)))
        else:
            return expr
    elif isinstance(expr, HadamardProduct):
        tp = _array_tensor_product(*[convert_matrix_to_array(arg) for arg in expr.args])
        diag = [[2*i for i in range(len(expr.args))], [2*i+1 for i in range(len(expr.args))]]
        return _array_diagonal(tp, *diag)
    elif isinstance(expr, HadamardPower):
        base, exp = expr.args
        if isinstance(exp, Integer) and exp > 0:
            return convert_matrix_to_array(HadamardProduct.fromiter(base for i in range(exp)))
        else:
            d = Dummy("d")
            return ArrayElementwiseApplyFunc(Lambda(d, d**exp), base)
    elif isinstance(expr, KroneckerProduct):
        kp_args = [convert_matrix_to_array(arg) for arg in expr.args]
        permutation = [2*i for i in range(len(kp_args))] + [2*i + 1 for i in range(len(kp_args))]
        return Reshape(_permute_dims(_array_tensor_product(*kp_args), permutation), expr.shape)
    else:
        return expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\lowlevel.py ===
"""
This namespace represents low-level functionality not intended for daily use,
but useful for extending Trio's functionality.
"""

# imports are renamed with leading underscores to indicate they are not part of the public API

import select as _select

# static checkers don't understand if importing this as _sys, so it's deleted later
import sys
import typing as _t

# Generally available symbols
from ._core import (
    Abort as Abort,
    ParkingLot as ParkingLot,
    ParkingLotStatistics as ParkingLotStatistics,
    RaiseCancelT as RaiseCancelT,
    RunStatistics as RunStatistics,
    RunVar as RunVar,
    RunVarToken as RunVarToken,
    Task as Task,
    TrioToken as TrioToken,
    UnboundedQueue as UnboundedQueue,
    UnboundedQueueStatistics as UnboundedQueueStatistics,
    add_instrument as add_instrument,
    add_parking_lot_breaker as add_parking_lot_breaker,
    cancel_shielded_checkpoint as cancel_shielded_checkpoint,
    checkpoint as checkpoint,
    checkpoint_if_cancelled as checkpoint_if_cancelled,
    current_clock as current_clock,
    current_root_task as current_root_task,
    current_statistics as current_statistics,
    current_task as current_task,
    current_trio_token as current_trio_token,
    currently_ki_protected as currently_ki_protected,
    disable_ki_protection as disable_ki_protection,
    enable_ki_protection as enable_ki_protection,
    in_trio_run as in_trio_run,
    in_trio_task as in_trio_task,
    notify_closing as notify_closing,
    permanently_detach_coroutine_object as permanently_detach_coroutine_object,
    reattach_detached_coroutine_object as reattach_detached_coroutine_object,
    remove_instrument as remove_instrument,
    remove_parking_lot_breaker as remove_parking_lot_breaker,
    reschedule as reschedule,
    spawn_system_task as spawn_system_task,
    start_guest_run as start_guest_run,
    start_thread_soon as start_thread_soon,
    temporarily_detach_coroutine_object as temporarily_detach_coroutine_object,
    wait_readable as wait_readable,
    wait_task_rescheduled as wait_task_rescheduled,
    wait_writable as wait_writable,
)
from ._subprocess import open_process as open_process

# This is the union of a subset of trio/_core/ and some things from trio/*.py.
# See comments in trio/__init__.py for details.

# Uses `from x import y as y` for compatibility with `pyright --verifytypes` (#2625)


if sys.platform == "win32":
    # Windows symbols
    from ._core import (
        current_iocp as current_iocp,
        monitor_completion_key as monitor_completion_key,
        readinto_overlapped as readinto_overlapped,
        register_with_iocp as register_with_iocp,
        wait_overlapped as wait_overlapped,
        write_overlapped as write_overlapped,
    )
    from ._wait_for_object import WaitForSingleObject as WaitForSingleObject
else:
    # Unix symbols
    from ._unix_pipes import FdStream as FdStream

    # Kqueue-specific symbols
    if sys.platform != "linux" and (_t.TYPE_CHECKING or not hasattr(_select, "epoll")):
        from ._core import (
            current_kqueue as current_kqueue,
            monitor_kevent as monitor_kevent,
            wait_kevent as wait_kevent,
        )

del sys

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\testing\_sequencer.py ===
from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import attrs

from .. import Event, _core, _util

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@_util.final
@attrs.define(eq=False, slots=False)
class Sequencer:
    """A convenience class for forcing code in different tasks to run in an
    explicit linear order.

    Instances of this class implement a ``__call__`` method which returns an
    async context manager. The idea is that you pass a sequence number to
    ``__call__`` to say where this block of code should go in the linear
    sequence. Block 0 starts immediately, and then block N doesn't start until
    block N-1 has finished.

    Example:
      An extremely elaborate way to print the numbers 0-5, in order::

         async def worker1(seq):
             async with seq(0):
                 print(0)
             async with seq(4):
                 print(4)

         async def worker2(seq):
             async with seq(2):
                 print(2)
             async with seq(5):
                 print(5)

         async def worker3(seq):
             async with seq(1):
                 print(1)
             async with seq(3):
                 print(3)

         async def main():
            seq = trio.testing.Sequencer()
            async with trio.open_nursery() as nursery:
                nursery.start_soon(worker1, seq)
                nursery.start_soon(worker2, seq)
                nursery.start_soon(worker3, seq)

    """

    _sequence_points: defaultdict[int, Event] = attrs.field(
        factory=lambda: defaultdict(Event),
        init=False,
    )
    _claimed: set[int] = attrs.field(factory=set, init=False)
    _broken: bool = attrs.field(default=False, init=False)

    @asynccontextmanager
    async def __call__(self, position: int) -> AsyncIterator[None]:
        if position in self._claimed:
            raise RuntimeError(f"Attempted to reuse sequence point {position}")
        if self._broken:
            raise RuntimeError("sequence broken!")
        self._claimed.add(position)
        if position != 0:
            try:
                await self._sequence_points[position].wait()
            except _core.Cancelled:
                self._broken = True
                for event in self._sequence_points.values():
                    event.set()
                raise RuntimeError(
                    "Sequencer wait cancelled -- sequence broken",
                ) from None
            else:
                if self._broken:
                    raise RuntimeError("sequence broken!")
        try:
            yield
        finally:
            self._sequence_points[position + 1].set()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\http2\probe.py ===
from __future__ import annotations

import threading


class _HTTP2ProbeCache:
    __slots__ = (
        "_lock",
        "_cache_locks",
        "_cache_values",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache_locks: dict[tuple[str, int], threading.RLock] = {}
        self._cache_values: dict[tuple[str, int], bool | None] = {}

    def acquire_and_get(self, host: str, port: int) -> bool | None:
        # By the end of this block we know that
        # _cache_[values,locks] is available.
        value = None
        with self._lock:
            key = (host, port)
            try:
                value = self._cache_values[key]
                # If it's a known value we return right away.
                if value is not None:
                    return value
            except KeyError:
                self._cache_locks[key] = threading.RLock()
                self._cache_values[key] = None

        # If the value is unknown, we acquire the lock to signal
        # to the requesting thread that the probe is in progress
        # or that the current thread needs to return their findings.
        key_lock = self._cache_locks[key]
        key_lock.acquire()
        try:
            # If the by the time we get the lock the value has been
            # updated we want to return the updated value.
            value = self._cache_values[key]

        # In case an exception like KeyboardInterrupt is raised here.
        except BaseException as e:  # Defensive:
            assert not isinstance(e, KeyError)  # KeyError shouldn't be possible.
            key_lock.release()
            raise

        return value

    def set_and_release(
        self, host: str, port: int, supports_http2: bool | None
    ) -> None:
        key = (host, port)
        key_lock = self._cache_locks[key]
        with key_lock:  # Uses an RLock, so can be locked again from same thread.
            if supports_http2 is None and self._cache_values[key] is not None:
                raise ValueError(
                    "Cannot reset HTTP/2 support for origin after value has been set."
                )  # Defensive: not expected in normal usage

        self._cache_values[key] = supports_http2
        key_lock.release()

    def _values(self) -> dict[tuple[str, int], bool | None]:
        """This function is for testing purposes only. Gets the current state of the probe cache"""
        with self._lock:
            return {k: v for k, v in self._cache_values.items()}

    def _reset(self) -> None:
        """This function is for testing purposes only. Reset the cache values"""
        with self._lock:
            self._cache_locks = {}
            self._cache_values = {}


_HTTP2_PROBE_CACHE = _HTTP2ProbeCache()

set_and_release = _HTTP2_PROBE_CACHE.set_and_release
acquire_and_get = _HTTP2_PROBE_CACHE.acquire_and_get
_values = _HTTP2_PROBE_CACHE._values
_reset = _HTTP2_PROBE_CACHE._reset

__all__ = [
    "set_and_release",
    "acquire_and_get",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\tests\test_generator_mt19937.py ===
import hashlib
import os.path
import sys

import pytest

import numpy as np
from numpy.exceptions import AxisError
from numpy.linalg import LinAlgError
from numpy.random import MT19937, Generator, RandomState, SeedSequence
from numpy.testing import (
    IS_WASM,
    assert_,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_no_warnings,
    assert_raises,
    assert_warns,
    suppress_warnings,
)

random = Generator(MT19937())

JUMP_TEST_DATA = [
    {
        "seed": 0,
        "steps": 10,
        "initial": {"key_sha256": "bb1636883c2707b51c5b7fc26c6927af4430f2e0785a8c7bc886337f919f9edf", "pos": 9},    # noqa: E501
        "jumped":  {"key_sha256": "ff682ac12bb140f2d72fba8d3506cf4e46817a0db27aae1683867629031d8d55", "pos": 598},  # noqa: E501
    },
    {
        "seed": 384908324,
        "steps": 312,
        "initial": {"key_sha256": "16b791a1e04886ccbbb4d448d6ff791267dc458ae599475d08d5cced29d11614", "pos": 311},  # noqa: E501
        "jumped":  {"key_sha256": "a0110a2cf23b56be0feaed8f787a7fc84bef0cb5623003d75b26bdfa1c18002c", "pos": 276},  # noqa: E501
    },
    {
        "seed": [839438204, 980239840, 859048019, 821],
        "steps": 511,
        "initial": {"key_sha256": "d306cf01314d51bd37892d874308200951a35265ede54d200f1e065004c3e9ea", "pos": 510},  # noqa: E501
        "jumped":  {"key_sha256": "0e00ab449f01a5195a83b4aee0dfbc2ce8d46466a640b92e33977d2e42f777f8", "pos": 475},  # noqa: E501
    },
]


@pytest.fixture(scope='module', params=[True, False])
def endpoint(request):
    return request.param


class TestSeed:
    def test_scalar(self):
        s = Generator(MT19937(0))
        assert_equal(s.integers(1000), 479)
        s = Generator(MT19937(4294967295))
        assert_equal(s.integers(1000), 324)

    def test_array(self):
        s = Generator(MT19937(range(10)))
        assert_equal(s.integers(1000), 465)
        s = Generator(MT19937(np.arange(10)))
        assert_equal(s.integers(1000), 465)
        s = Generator(MT19937([0]))
        assert_equal(s.integers(1000), 479)
        s = Generator(MT19937([4294967295]))
        assert_equal(s.integers(1000), 324)

    def test_seedsequence(self):
        s = MT19937(SeedSequence(0))
        assert_equal(s.random_raw(1), 2058676884)

    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, MT19937, -0.5)
        assert_raises(ValueError, MT19937, -1)

    def test_invalid_array(self):
        # seed must be an unsigned integer
        assert_raises(TypeError, MT19937, [-0.5])
        assert_raises(ValueError, MT19937, [-1])
        assert_raises(ValueError, MT19937, [1, -2, 4294967296])

    def test_noninstantized_bitgen(self):
        assert_raises(ValueError, Generator, MT19937)


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
        assert_(-5 <= random.integers(-5, -1) < -1)
        x = random.integers(-5, -1, 5)
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
        assert_raises(ValueError, random.multinomial, [-1] * 10, [0.8, 0.2])

    def test_p_non_contiguous(self):
        p = np.arange(15.)
        p /= np.sum(p[1::3])
        pvals = p[1::3]
        random = Generator(MT19937(1432985819))
        non_contig = random.multinomial(100, pvals=pvals)
        random = Generator(MT19937(1432985819))
        contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
        assert_array_equal(non_contig, contig)

    def test_multinomial_pvals_float32(self):
        x = np.array([9.9e-01, 9.9e-01, 1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09,
                      1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09], dtype=np.float32)
        pvals = x / x.sum()
        random = Generator(MT19937(1432985819))
        match = r"[\w\s]*pvals array is cast to 64-bit floating"
        with pytest.raises(ValueError, match=match):
            random.multinomial(1, pvals)


class TestMultivariateHypergeometric:

    def setup_method(self):
        self.seed = 8675309

    def test_argument_validation(self):
        # Error cases...

        # `colors` must be a 1-d sequence
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      10, 4)

        # Negative nsample
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], -1)

        # Negative color
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [-1, 2, 3], 2)

        # nsample exceeds sum(colors)
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], 10)

        # nsample exceeds sum(colors) (edge case of empty colors)
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [], 1)

        # Validation errors associated with very large values in colors.
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [999999999, 101], 5, 1, 'marginals')

        int64_info = np.iinfo(np.int64)
        max_int64 = int64_info.max
        max_int64_index = max_int64 // int64_info.dtype.itemsize
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [max_int64_index - 100, 101], 5, 1, 'count')

    @pytest.mark.parametrize('method', ['count', 'marginals'])
    def test_edge_cases(self, method):
        # Set the seed, but in fact, all the results in this test are
        # deterministic, so we don't really need this.
        random = Generator(MT19937(self.seed))

        x = random.multivariate_hypergeometric([0, 0, 0], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        x = random.multivariate_hypergeometric([], 0, method=method)
        assert_array_equal(x, [])

        x = random.multivariate_hypergeometric([], 0, size=1, method=method)
        assert_array_equal(x, np.empty((1, 0), dtype=np.int64))

        x = random.multivariate_hypergeometric([1, 2, 3], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        x = random.multivariate_hypergeometric([9, 0, 0], 3, method=method)
        assert_array_equal(x, [3, 0, 0])

        colors = [1, 1, 0, 1, 1]
        x = random.multivariate_hypergeometric(colors, sum(colors),
                                               method=method)
        assert_array_equal(x, colors)

        x = random.multivariate_hypergeometric([3, 4, 5], 12, size=3,
                                               method=method)
        assert_array_equal(x, [[3, 4, 5]] * 3)

    # Cases for nsample:
    #     nsample < 10
    #     10 <= nsample < colors.sum()/2
    #     colors.sum()/2 < nsample < colors.sum() - 10
    #     colors.sum() - 10 < nsample < colors.sum()
    @pytest.mark.parametrize('nsample', [8, 25, 45, 55])
    @pytest.mark.parametrize('method', ['count', 'marginals'])
    @pytest.mark.parametrize('size', [5, (2, 3), 150000])
    def test_typical_cases(self, nsample, method, size):
        random = Generator(MT19937(self.seed))

        colors = np.array([10, 5, 20, 25])
        sample = random.multivariate_hypergeometric(colors, nsample, size,
                                                    method=method)
        if isinstance(size, int):
            expected_shape = (size,) + colors.shape
        else:
            expected_shape = size + colors.shape
        assert_equal(sample.shape, expected_shape)
        assert_((sample >= 0).all())
        assert_((sample <= colors).all())
        assert_array_equal(sample.sum(axis=-1),
                           np.full(size, fill_value=nsample, dtype=int))
        if isinstance(size, int) and size >= 100000:
            # This sample is large enough to compare its mean to
            # the expected values.
            assert_allclose(sample.mean(axis=0),
                            nsample * colors / colors.sum(),
                            rtol=1e-3, atol=0.005)

    def test_repeatability1(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([3, 4, 5], 5, size=5,
                                                    method='count')
        expected = np.array([[2, 1, 2],
                             [2, 1, 2],
                             [1, 1, 3],
                             [2, 0, 3],
                             [2, 1, 2]])
        assert_array_equal(sample, expected)

    def test_repeatability2(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 50,
                                                    size=5,
                                                    method='marginals')
        expected = np.array([[ 9, 17, 24],
                             [ 7, 13, 30],
                             [ 9, 15, 26],
                             [ 9, 17, 24],
                             [12, 14, 24]])
        assert_array_equal(sample, expected)

    def test_repeatability3(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 12,
                                                    size=5,
                                                    method='marginals')
        expected = np.array([[2, 3, 7],
                             [5, 3, 4],
                             [2, 5, 5],
                             [5, 3, 4],
                             [1, 5, 6]])
        assert_array_equal(sample, expected)


class TestSetState:
    def setup_method(self):
        self.seed = 1234567890
        self.rg = Generator(MT19937(self.seed))
        self.bit_generator = self.rg.bit_generator
        self.state = self.bit_generator.state
        self.legacy_state = (self.state['bit_generator'],
                             self.state['state']['key'],
                             self.state['state']['pos'])

    def test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = self.state
        new = self.rg.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.

        self.rg.standard_normal()
        state = self.bit_generator.state
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = state
        new = self.rg.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.rg.negative_binomial(0.5, 0.5)


class TestIntegers:
    rfunc = random.integers

    # valid integer/boolean types
    itype = [bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self, endpoint):
        assert_raises(TypeError, self.rfunc, 1, endpoint=endpoint, dtype=float)

    def test_bounds_checking(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, endpoint=endpoint,
                          dtype=dt)

            assert_raises(ValueError, self.rfunc, [lbnd - 1], ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd], [ubnd + 1],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd], [lbnd],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, [0],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd + 1], [ubnd],
                          endpoint=endpoint, dtype=dt)

    def test_bounds_checking_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + (not endpoint)

            assert_raises(ValueError, self.rfunc, [lbnd - 1] * 2, [ubnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd] * 2,
                          [ubnd + 1] * 2, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, [lbnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [1] * 2, 0,
                          endpoint=endpoint, dtype=dt)

    def test_rng_zero_and_extremes(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            is_open = not endpoint

            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc(tgt, [tgt + is_open], size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], [tgt + is_open],
                                    size=1000, endpoint=endpoint, dtype=dt),
                         tgt)

    def test_rng_zero_and_extremes_array(self, endpoint):
        size = 1000
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            tgt = ubnd - 1
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

    def test_full_range(self, endpoint):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_full_range_array(self, endpoint):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc([lbnd] * 2, [ubnd], endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_in_bounds_fuzz(self, endpoint):
        # Don't use fixed seed
        random = Generator(MT19937())

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd - endpoint, size=2 ** 16,
                                  endpoint=endpoint, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = self.rfunc(0, 2 - endpoint, size=2 ** 16, endpoint=endpoint,
                          dtype=bool)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_scalar_array_equiv(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            size = 1000
            random = Generator(MT19937(1234))
            scalar = random.integers(lbnd, ubnd, size=size, endpoint=endpoint,
                                dtype=dt)

            random = Generator(MT19937(1234))
            scalar_array = random.integers([lbnd], [ubnd], size=size,
                                      endpoint=endpoint, dtype=dt)

            random = Generator(MT19937(1234))
            array = random.integers([lbnd] * size, [ubnd] *
                               size, size=size, endpoint=endpoint, dtype=dt)
            assert_array_equal(scalar, scalar_array)
            assert_array_equal(scalar, array)

    def test_repeatability(self, endpoint):
        # We use a sha256 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool':   '053594a9b82d656f967c54869bc6970aa0358cf94ad469c81478459c6a90eee3',  # noqa: E501
               'int16':  '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4',  # noqa: E501
               'int32':  'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b',  # noqa: E501
               'int64':  '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1',  # noqa: E501
               'int8':   '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1',  # noqa: E501
               'uint16': '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4',  # noqa: E501
               'uint32': 'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b',  # noqa: E501
               'uint64': '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1',  # noqa: E501
               'uint8':  '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1'}  # noqa: E501

        for dt in self.itype[1:]:
            random = Generator(MT19937(1234))

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                 dtype=dt)
            else:
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                 dtype=dt).byteswap()

            res = hashlib.sha256(val).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianness
        random = Generator(MT19937(1234))
        val = random.integers(0, 2 - endpoint, size=1000, endpoint=endpoint,
                         dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    def test_repeatability_broadcasting(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt in (bool, np.bool) else np.iinfo(dt).min
            ubnd = 2 if dt in (bool, np.bool) else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # view as little endian for hash
            random = Generator(MT19937(1234))
            val = random.integers(lbnd, ubnd, size=1000, endpoint=endpoint,
                             dtype=dt)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, ubnd, endpoint=endpoint,
                                dtype=dt)

            assert_array_equal(val, val_bc)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, [ubnd] * 1000,
                                endpoint=endpoint, dtype=dt)

            assert_array_equal(val, val_bc)

    @pytest.mark.parametrize(
        'bound, expected',
        [(2**32 - 1, np.array([517043486, 1364798665, 1733884389, 1353720612,
                               3769704066, 1170797179, 4108474671])),
         (2**32, np.array([517043487, 1364798666, 1733884390, 1353720613,
                           3769704067, 1170797180, 4108474672])),
         (2**32 + 1, np.array([517043487, 1733884390, 3769704068, 4108474673,
                               1831631863, 1215661561, 3869512430]))]
    )
    def test_repeatability_32bit_boundary(self, bound, expected):
        for size in [None, len(expected)]:
            random = Generator(MT19937(1234))
            x = random.integers(bound, size=size)
            assert_equal(x, expected if size is not None else expected[0])

    def test_repeatability_32bit_boundary_broadcasting(self):
        desired = np.array([[[1622936284, 3620788691, 1659384060],
                             [1417365545,  760222891, 1909653332],
                             [3788118662,  660249498, 4092002593]],
                            [[3625610153, 2979601262, 3844162757],
                             [ 685800658,  120261497, 2694012896],
                             [1207779440, 1586594375, 3854335050]],
                            [[3004074748, 2310761796, 3012642217],
                             [2067714190, 2786677879, 1363865881],
                             [ 791663441, 1867303284, 2169727960]],
                            [[1939603804, 1250951100,  298950036],
                             [1040128489, 3791912209, 3317053765],
                             [3155528714,   61360675, 2305155588]],
                            [[ 817688762, 1335621943, 3288952434],
                             [1770890872, 1102951817, 1957607470],
                             [3099996017,  798043451,   48334215]]])
        for size in [None, (5, 3, 3)]:
            random = Generator(MT19937(12345))
            x = random.integers([[-1], [0], [1]],
                                [2**32 - 1, 2**32, 2**32 + 1],
                                size=size)
            assert_array_equal(x, desired if size is not None else desired[0])

    def test_int64_uint64_broadcast_exceptions(self, endpoint):
        configs = {np.uint64: ((0, 2**65), (-1, 2**62), (10, 9), (0, 0)),
                   np.int64: ((0, 2**64), (-(2**64), 2**62), (10, 9), (0, 0),
                              (-2**63 - 1, -2**63 - 1))}
        for dtype in configs:
            for config in configs[dtype]:
                low, high = config
                high = high - endpoint
                low_a = np.array([[low] * 10])
                high_a = np.array([high] * 10)
                assert_raises(ValueError, random.integers, low, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_a,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high_a,
                              endpoint=endpoint, dtype=dtype)

                low_o = np.array([[low] * 10], dtype=object)
                high_o = np.array([high] * 10, dtype=object)
                assert_raises(ValueError, random.integers, low_o, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_o,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_o, high_o,
                              endpoint=endpoint, dtype=dtype)

    def test_int64_uint64_corner_case(self, endpoint):
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
        ubnd = np.uint64(np.iinfo(np.int64).max + 1 - endpoint)

        # None of these function calls should
        # generate a ValueError now.
        actual = random.integers(lbnd, ubnd, endpoint=endpoint, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool if dt is bool else dt

            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)

        for dt in (bool, int):
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # gh-7284: Ensure that we get Python data types
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert not hasattr(sample, 'dtype')
            assert_equal(type(sample), dt)

    def test_respect_dtype_array(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool if dt is bool else dt

            sample = self.rfunc([lbnd], [ubnd], endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)
            sample = self.rfunc([lbnd] * 2, [ubnd] * 2, endpoint=endpoint,
                                dtype=dt)
            assert_equal(sample.dtype, dt)

    def test_zero_size(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            sample = self.rfunc(0, 0, (3, 0, 4), endpoint=endpoint, dtype=dt)
            assert sample.shape == (3, 0, 4)
            assert sample.dtype == dt
            assert self.rfunc(0, -10, 0, endpoint=endpoint,
                              dtype=dt).shape == (0,)
            assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape,
                         (3, 0, 4))
            assert_equal(random.integers(0, -10, size=0).shape, (0,))
            assert_equal(random.integers(10, 10, size=0).shape, (0,))

    def test_error_byteorder(self):
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        with pytest.raises(ValueError):
            random.integers(0, 200, size=10, dtype=other_byteord_dt)

    # chi2max is the maximum acceptable chi-squared value.
    @pytest.mark.slow
    @pytest.mark.parametrize('sample_size,high,dtype,chi2max',
        [(5000000, 5, np.int8, 125.0),          # p-value ~4.6e-25
         (5000000, 7, np.uint8, 150.0),         # p-value ~7.7e-30
         (10000000, 2500, np.int16, 3300.0),    # p-value ~3.0e-25
         (50000000, 5000, np.uint16, 6500.0),   # p-value ~3.5e-25
        ])
    def test_integers_small_dtype_chisquared(self, sample_size, high,
                                             dtype, chi2max):
        # Regression test for gh-14774.
        samples = random.integers(high, size=sample_size, dtype=dtype)

        values, counts = np.unique(samples, return_counts=True)
        expected = sample_size / high
        chi2 = ((counts - expected)**2 / expected).sum()
        assert chi2 < chi2max


class TestRandomDist:
    # Make sure the random distribution returns the correct value for a
    # given seed

    def setup_method(self):
        self.seed = 1234567890

    def test_integers(self):
        random = Generator(MT19937(self.seed))
        actual = random.integers(-99, 99, size=(3, 2))
        desired = np.array([[-80, -56], [41, 37], [-83, -16]])
        assert_array_equal(actual, desired)

    def test_integers_masked(self):
        # Test masked rejection sampling algorithm to generate array of
        # uint32 in an interval.
        random = Generator(MT19937(self.seed))
        actual = random.integers(0, 99, size=(3, 2), dtype=np.uint32)
        desired = np.array([[9, 21], [70, 68], [8, 41]], dtype=np.uint32)
        assert_array_equal(actual, desired)

    def test_integers_closed(self):
        random = Generator(MT19937(self.seed))
        actual = random.integers(-99, 99, size=(3, 2), endpoint=True)
        desired = np.array([[-80, -56], [41, 38], [-83, -15]])
        assert_array_equal(actual, desired)

    def test_integers_max_int(self):
        # Tests whether integers with closed=True can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        actual = random.integers(np.iinfo('l').max, np.iinfo('l').max,
                                 endpoint=True)

        desired = np.iinfo('l').max
        assert_equal(actual, desired)

    def test_random(self):
        random = Generator(MT19937(self.seed))
        actual = random.random((3, 2))
        desired = np.array([[0.096999199829214, 0.707517457682192],
                            [0.084364834598269, 0.767731206553125],
                            [0.665069021359413, 0.715487190596693]])
        assert_array_almost_equal(actual, desired, decimal=15)

        random = Generator(MT19937(self.seed))
        actual = random.random()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_random_float(self):
        random = Generator(MT19937(self.seed))
        actual = random.random((3, 2))
        desired = np.array([[0.0969992 , 0.70751746],  # noqa: E203
                            [0.08436483, 0.76773121],
                            [0.66506902, 0.71548719]])
        assert_array_almost_equal(actual, desired, decimal=7)

    def test_random_float_scalar(self):
        random = Generator(MT19937(self.seed))
        actual = random.random(dtype=np.float32)
        desired = 0.0969992
        assert_array_almost_equal(actual, desired, decimal=7)

    @pytest.mark.parametrize('dtype, uint_view_type',
                             [(np.float32, np.uint32),
                              (np.float64, np.uint64)])
    def test_random_distribution_of_lsb(self, dtype, uint_view_type):
        random = Generator(MT19937(self.seed))
        sample = random.random(100000, dtype=dtype)
        num_ones_in_lsb = np.count_nonzero(sample.view(uint_view_type) & 1)
        # The probability of a 1 in the least significant bit is 0.25.
        # With a sample size of 100000, the probability that num_ones_in_lsb
        # is outside the following range is less than 5e-11.
        assert 24100 < num_ones_in_lsb < 25900

    def test_random_unsupported_type(self):
        assert_raises(TypeError, random.random, dtype='int32')

    def test_choice_uniform_replace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 4)
        desired = np.array([0, 0, 2, 2], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_replace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([0, 1, 0, 1], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_uniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 3, replace=False)
        desired = np.array([2, 0, 3], dtype=np.int64)
        assert_array_equal(actual, desired)
        actual = random.choice(4, 4, replace=False, shuffle=False)
        desired = np.arange(4, dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([0, 2, 3], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_noninteger(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['a', 'a', 'c', 'c'])
        assert_array_equal(actual, desired)

    def test_choice_multidimensional_default_axis(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 3)
        desired = np.array([[0, 1], [0, 1], [4, 5]])
        assert_array_equal(actual, desired)

    def test_choice_multidimensional_custom_axis(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 1, axis=1)
        desired = np.array([[0], [2], [4], [6]])
        assert_array_equal(actual, desired)

    def test_choice_exceptions(self):
        sample = random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
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
        assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.integers(0, -10, size=0).shape, (0,))
        assert_equal(random.integers(10, 10, size=0).shape, (0,))
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
        random = Generator(MT19937(self.seed))
        non_contig = random.choice(5, 3, p=p[::2])
        random = Generator(MT19937(self.seed))
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)

    def test_choice_return_type(self):
        # gh 9867
        p = np.ones(4) / 4.
        actual = random.choice(4, 2)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, replace=False)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, p=p)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, p=p, replace=False)
        assert actual.dtype == np.int64

    def test_choice_large_sample(self):
        choice_hash = '4266599d12bfcfb815213303432341c06b4349f5455890446578877bb322e222'
        random = Generator(MT19937(self.seed))
        actual = random.choice(10000, 5000, replace=False)
        if sys.byteorder != 'little':
            actual = actual.byteswap()
        res = hashlib.sha256(actual.view(np.int8)).hexdigest()
        assert_(choice_hash == res)

    def test_choice_array_size_empty_tuple(self):
        random = Generator(MT19937(self.seed))
        assert_array_equal(random.choice([1, 2, 3], size=()), np.array(1),
                           strict=True)
        assert_array_equal(random.choice([[1, 2, 3]], size=()), [1, 2, 3])
        assert_array_equal(random.choice([[1]], size=()), [1], strict=True)
        assert_array_equal(random.choice([[1]], size=(), axis=1), [1],
                           strict=True)

    def test_bytes(self):
        random = Generator(MT19937(self.seed))
        actual = random.bytes(10)
        desired = b'\x86\xf0\xd4\x18\xe1\x81\t8%\xdd'
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
            random = Generator(MT19937(self.seed))
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            random.shuffle(alist)
            actual = alist
            desired = conv([4, 1, 9, 8, 0, 5, 3, 6, 2, 7])
            assert_array_equal(actual, desired)

    def test_shuffle_custom_axis(self):
        random = Generator(MT19937(self.seed))
        actual = np.arange(16).reshape((4, 4))
        random.shuffle(actual, axis=1)
        desired = np.array([[ 0,  3,  1,  2],
                            [ 4,  7,  5,  6],
                            [ 8, 11,  9, 10],
                            [12, 15, 13, 14]])
        assert_array_equal(actual, desired)
        random = Generator(MT19937(self.seed))
        actual = np.arange(16).reshape((4, 4))
        random.shuffle(actual, axis=-1)
        assert_array_equal(actual, desired)

    def test_shuffle_custom_axis_empty(self):
        random = Generator(MT19937(self.seed))
        desired = np.array([]).reshape((0, 6))
        for axis in (0, 1):
            actual = np.array([]).reshape((0, 6))
            random.shuffle(actual, axis=axis)
            assert_array_equal(actual, desired)

    def test_shuffle_axis_nonsquare(self):
        y1 = np.arange(20).reshape(2, 10)
        y2 = y1.copy()
        random = Generator(MT19937(self.seed))
        random.shuffle(y1, axis=1)
        random = Generator(MT19937(self.seed))
        random.shuffle(y2.T)
        assert_array_equal(y1, y2)

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

    def test_shuffle_exceptions(self):
        random = Generator(MT19937(self.seed))
        arr = np.arange(10)
        assert_raises(AxisError, random.shuffle, arr, 1)
        arr = np.arange(9).reshape((3, 3))
        assert_raises(AxisError, random.shuffle, arr, 3)
        assert_raises(TypeError, random.shuffle, arr, slice(1, 2, None))
        arr = [[1, 2, 3], [4, 5, 6]]
        assert_raises(NotImplementedError, random.shuffle, arr, 1)

        arr = np.array(3)
        assert_raises(TypeError, random.shuffle, arr)
        arr = np.ones((3, 2))
        assert_raises(AxisError, random.shuffle, arr, 2)

    def test_shuffle_not_writeable(self):
        random = Generator(MT19937(self.seed))
        a = np.zeros(5)
        a.flags.writeable = False
        with pytest.raises(ValueError, match='read-only'):
            random.shuffle(a)

    def test_permutation(self):
        random = Generator(MT19937(self.seed))
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actual = random.permutation(alist)
        desired = [4, 1, 9, 8, 0, 5, 3, 6, 2, 7]
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        actual = random.permutation(arr_2d)
        assert_array_equal(actual, np.atleast_2d(desired).T)

        bad_x_str = "abcd"
        assert_raises(AxisError, random.permutation, bad_x_str)

        bad_x_float = 1.2
        assert_raises(AxisError, random.permutation, bad_x_float)

        random = Generator(MT19937(self.seed))
        integer_val = 10
        desired = [3, 0, 8, 7, 9, 4, 2, 5, 1, 6]

        actual = random.permutation(integer_val)
        assert_array_equal(actual, desired)

    def test_permutation_custom_axis(self):
        a = np.arange(16).reshape((4, 4))
        desired = np.array([[ 0,  3,  1,  2],
                            [ 4,  7,  5,  6],
                            [ 8, 11,  9, 10],
                            [12, 15, 13, 14]])
        random = Generator(MT19937(self.seed))
        actual = random.permutation(a, axis=1)
        assert_array_equal(actual, desired)
        random = Generator(MT19937(self.seed))
        actual = random.permutation(a, axis=-1)
        assert_array_equal(actual, desired)

    def test_permutation_exceptions(self):
        random = Generator(MT19937(self.seed))
        arr = np.arange(10)
        assert_raises(AxisError, random.permutation, arr, 1)
        arr = np.arange(9).reshape((3, 3))
        assert_raises(AxisError, random.permutation, arr, 3)
        assert_raises(TypeError, random.permutation, arr, slice(1, 2, None))

    @pytest.mark.parametrize("dtype", [int, object])
    @pytest.mark.parametrize("axis, expected",
                             [(None, np.array([[3, 7, 0, 9, 10, 11],
                                               [8, 4, 2, 5,  1,  6]])),
                              (0, np.array([[6, 1, 2, 9, 10, 11],
                                            [0, 7, 8, 3,  4,  5]])),
                              (1, np.array([[ 5, 3,  4, 0, 2, 1],
                                            [11, 9, 10, 6, 8, 7]]))])
    def test_permuted(self, dtype, axis, expected):
        random = Generator(MT19937(self.seed))
        x = np.arange(12).reshape(2, 6).astype(dtype)
        random.permuted(x, axis=axis, out=x)
        assert_array_equal(x, expected)

        random = Generator(MT19937(self.seed))
        x = np.arange(12).reshape(2, 6).astype(dtype)
        y = random.permuted(x, axis=axis)
        assert y.dtype == dtype
        assert_array_equal(y, expected)

    def test_permuted_with_strides(self):
        random = Generator(MT19937(self.seed))
        x0 = np.arange(22).reshape(2, 11)
        x1 = x0.copy()
        x = x0[:, ::3]
        y = random.permuted(x, axis=1, out=x)
        expected = np.array([[0, 9, 3, 6],
                             [14, 20, 11, 17]])
        assert_array_equal(y, expected)
        x1[:, ::3] = expected
        # Verify that the original x0 was modified in-place as expected.
        assert_array_equal(x1, x0)

    def test_permuted_empty(self):
        y = random.permuted([])
        assert_array_equal(y, [])

    @pytest.mark.parametrize('outshape', [(2, 3), 5])
    def test_permuted_out_with_wrong_shape(self, outshape):
        a = np.array([1, 2, 3])
        out = np.zeros(outshape, dtype=a.dtype)
        with pytest.raises(ValueError, match='same shape'):
            random.permuted(a, out=out)

    def test_permuted_out_with_wrong_type(self):
        out = np.zeros((3, 5), dtype=np.int32)
        x = np.ones((3, 5))
        with pytest.raises(TypeError, match='Cannot cast'):
            random.permuted(x, axis=1, out=out)

    def test_permuted_not_writeable(self):
        x = np.zeros((2, 5))
        x.flags.writeable = False
        with pytest.raises(ValueError, match='read-only'):
            random.permuted(x, axis=1, out=x)

    def test_beta(self):
        random = Generator(MT19937(self.seed))
        actual = random.beta(.1, .9, size=(3, 2))
        desired = np.array(
            [[1.083029353267698e-10, 2.449965303168024e-11],
             [2.397085162969853e-02, 3.590779671820755e-08],
             [2.830254190078299e-04, 1.744709918330393e-01]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.binomial(100.123, .456, size=(3, 2))
        desired = np.array([[42, 41],
                            [42, 48],
                            [44, 50]])
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.binomial(100.123, .456)
        desired = 42
        assert_array_equal(actual, desired)

    def test_chisquare(self):
        random = Generator(MT19937(self.seed))
        actual = random.chisquare(50, size=(3, 2))
        desired = np.array([[32.9850547060149, 39.0219480493301],
                            [56.2006134779419, 57.3474165711485],
                            [55.4243733880198, 55.4209797925213]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        random = Generator(MT19937(self.seed))
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.5439892869558927,  0.45601071304410745],
                             [0.5588917345860708,  0.4411082654139292 ]],  # noqa: E202
                            [[0.5632074165063435,  0.43679258349365657],
                             [0.54862581112627,    0.45137418887373015]],
                            [[0.49961831357047226, 0.5003816864295278 ],  # noqa: E202
                             [0.52374806183482,    0.47625193816517997]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        bad_alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)

        random = Generator(MT19937(self.seed))
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

        # gh-15876
        assert_raises(ValueError, random.dirichlet, [[5, 1]])
        assert_raises(ValueError, random.dirichlet, [[5], [1]])
        assert_raises(ValueError, random.dirichlet, [[[5], [1]], [[1], [5]]])
        assert_raises(ValueError, random.dirichlet, np.array([[5, 1], [1, 5]]))

    def test_dirichlet_alpha_non_contiguous(self):
        a = np.array([51.72840233779265162, -1.0, 39.74494232180943953])
        alpha = a[::2]
        random = Generator(MT19937(self.seed))
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random = Generator(MT19937(self.seed))
        contig = random.dirichlet(np.ascontiguousarray(alpha),
                                  size=(3, 2))
        assert_array_almost_equal(non_contig, contig)

    def test_dirichlet_small_alpha(self):
        eps = 1.0e-9  # 1.0e-10 -> runtime x 10; 1e-11 -> runtime x 200, etc.
        alpha = eps * np.array([1., 1.0e-3])
        random = Generator(MT19937(self.seed))
        actual = random.dirichlet(alpha, size=(3, 2))
        expected = np.array([
            [[1., 0.],
             [1., 0.]],
            [[1., 0.],
             [1., 0.]],
            [[1., 0.],
             [1., 0.]]
        ])
        assert_array_almost_equal(actual, expected, decimal=15)

    @pytest.mark.slow
    def test_dirichlet_moderately_small_alpha(self):
        # Use alpha.max() < 0.1 to trigger stick breaking code path
        alpha = np.array([0.02, 0.04, 0.03])
        exact_mean = alpha / alpha.sum()
        random = Generator(MT19937(self.seed))
        sample = random.dirichlet(alpha, size=20000000)
        sample_mean = sample.mean(axis=0)
        assert_allclose(sample_mean, exact_mean, rtol=1e-3)

    # This set of parameters includes inputs with alpha.max() >= 0.1 and
    # alpha.max() < 0.1 to exercise both generation methods within the
    # dirichlet code.
    @pytest.mark.parametrize(
        'alpha',
        [[5, 9, 0, 8],
         [0.5, 0, 0, 0],
         [1, 5, 0, 0, 1.5, 0, 0, 0],
         [0.01, 0.03, 0, 0.005],
         [1e-5, 0, 0, 0],
         [0.002, 0.015, 0, 0, 0.04, 0, 0, 0],
         [0.0],
         [0, 0, 0]],
    )
    def test_dirichlet_multiple_zeros_in_alpha(self, alpha):
        alpha = np.array(alpha)
        y = random.dirichlet(alpha)
        assert_equal(y[alpha == 0], 0.0)

    def test_exponential(self):
        random = Generator(MT19937(self.seed))
        actual = random.exponential(1.1234, size=(3, 2))
        desired = np.array([[0.098845481066258, 1.560752510746964],
                            [0.075730916041636, 1.769098974710777],
                            [1.488602544592235, 2.49684815275751 ]])  # noqa: E202
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        assert_equal(random.exponential(scale=0), 0)
        assert_raises(ValueError, random.exponential, scale=-0.)

    def test_f(self):
        random = Generator(MT19937(self.seed))
        actual = random.f(12, 77, size=(3, 2))
        desired = np.array([[0.461720027077085, 1.100441958872451],
                            [1.100337455217484, 0.91421736740018 ],  # noqa: E202
                            [0.500811891303113, 0.826802454552058]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        random = Generator(MT19937(self.seed))
        actual = random.gamma(5, 3, size=(3, 2))
        desired = np.array([[ 5.03850858902096,  7.9228656732049 ],  # noqa: E202
                            [18.73983605132985, 19.57961681699238],
                            [18.17897755150825, 18.17653912505234]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        assert_equal(random.gamma(shape=0, scale=0), 0)
        assert_raises(ValueError, random.gamma, shape=-0., scale=-0.)

    def test_geometric(self):
        random = Generator(MT19937(self.seed))
        actual = random.geometric(.123456789, size=(3, 2))
        desired = np.array([[1, 11],
                            [1, 12],
                            [11, 17]])
        assert_array_equal(actual, desired)

    def test_geometric_exceptions(self):
        assert_raises(ValueError, random.geometric, 1.1)
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        assert_raises(ValueError, random.geometric, -0.1)
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.geometric, np.nan)
            assert_raises(ValueError, random.geometric, [np.nan] * 10)

    def test_gumbel(self):
        random = Generator(MT19937(self.seed))
        actual = random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[ 4.688397515056245, -0.289514845417841],
                            [ 4.981176042584683, -0.633224272589149],
                            [-0.055915275687488, -0.333962478257953]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        assert_equal(random.gumbel(scale=0), 0)
        assert_raises(ValueError, random.gumbel, scale=-0.)

    def test_hypergeometric(self):
        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[ 9, 9],
                            [ 9, 9],
                            [10, 9]])
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
        random = Generator(MT19937(self.seed))
        actual = random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-3.156353949272393,  1.195863024830054],
                            [-3.435458081645966,  1.656882398925444],
                            [ 0.924824032467446,  1.251116432209336]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        assert_equal(random.laplace(scale=0), 0)
        assert_raises(ValueError, random.laplace, scale=-0.)

    def test_logistic(self):
        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-4.338584631510999,  1.890171436749954],
                            [-4.64547787337966 ,  2.514545562919217],  # noqa: E203
                            [ 1.495389489198666,  1.967827627577474]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        random = Generator(MT19937(self.seed))
        actual = random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[ 0.0268252166335, 13.9534486483053],
                            [ 0.1204014788936,  2.2422077497792],
                            [ 4.2484199496128, 12.0093343977523]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_lognormal_0(self):
        assert_equal(random.lognormal(sigma=0), 1)
        assert_raises(ValueError, random.lognormal, sigma=-0.)

    def test_logseries(self):
        random = Generator(MT19937(self.seed))
        actual = random.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[14, 17],
                            [3, 18],
                            [5, 1]])
        assert_array_equal(actual, desired)

    def test_logseries_zero(self):
        random = Generator(MT19937(self.seed))
        assert random.logseries(0) == 1

    @pytest.mark.parametrize("value", [np.nextafter(0., -1), 1., np.nan, 5.])
    def test_logseries_exceptions(self, value):
        random = Generator(MT19937(self.seed))
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
        random = Generator(MT19937(self.seed))
        actual = random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[1, 5, 1, 6, 4, 3],
                             [4, 2, 6, 2, 4, 2]],
                            [[5, 3, 2, 6, 3, 1],
                             [4, 4, 0, 2, 3, 7]],
                            [[6, 3, 1, 5, 3, 2],
                             [5, 5, 3, 1, 2, 4]]])
        assert_array_equal(actual, desired)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
    def test_multivariate_normal(self, method):
        random = Generator(MT19937(self.seed))
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = random.multivariate_normal(mean, cov, size, method=method)
        desired = np.array([[[-1.747478062846581,  11.25613495182354 ],  # noqa: E202
                             [-0.9967333370066214, 10.342002097029821]],
                            [[ 0.7850019631242964, 11.181113712443013],
                             [ 0.8901349653255224,  8.873825399642492]],
                            [[ 0.7130260107430003,  9.551628690083056],
                             [ 0.7127098726541128, 11.991709234143173]]])

        assert_array_almost_equal(actual, desired, decimal=15)

        # Check for default size, was raising deprecation warning
        actual = random.multivariate_normal(mean, cov, method=method)
        desired = np.array([0.233278563284287, 9.424140804347195])
        assert_array_almost_equal(actual, desired, decimal=15)
        # Check that non symmetric covariance input raises exception when
        # check_valid='raises' if using default svd method.
        mean = [0, 0]
        cov = [[1, 2], [1, 2]]
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov,
                     method='eigh')
        assert_raises(LinAlgError, random.multivariate_normal, mean, cov,
                      method='cholesky')

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise', method='eigh')

        # check degenerate samples from singular covariance matrix
        cov = [[1, 1], [1, 1]]
        if method in ('svd', 'eigh'):
            samples = random.multivariate_normal(mean, cov, size=(3, 2),
                                                 method=method)
            assert_array_almost_equal(samples[..., 0], samples[..., 1],
                                      decimal=6)
        else:
            assert_raises(LinAlgError, random.multivariate_normal, mean, cov,
                          method='cholesky')

        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov, method=method)
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

    @pytest.mark.parametrize('mean, cov', [([0], [[1 + 1j]]), ([0j], [[1]])])
    def test_multivariate_normal_disallow_complex(self, mean, cov):
        random = Generator(MT19937(self.seed))
        with pytest.raises(TypeError, match="must not be complex"):
            random.multivariate_normal(mean, cov)

    @pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
    def test_multivariate_normal_basic_stats(self, method):
        random = Generator(MT19937(self.seed))
        n_s = 1000
        mean = np.array([1, 2])
        cov = np.array([[2, 1], [1, 2]])
        s = random.multivariate_normal(mean, cov, size=(n_s,), method=method)
        s_center = s - mean
        cov_emp = (s_center.T @ s_center) / (n_s - 1)
        # these are pretty loose and are only designed to detect major errors
        assert np.all(np.abs(s_center.mean(-2)) < 0.1)
        assert np.all(np.abs(cov_emp - cov) < 0.2)

    def test_negative_binomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = np.array([[543, 727],
                            [775, 760],
                            [600, 674]])
        assert_array_equal(actual, desired)

    def test_negative_binomial_exceptions(self):
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.negative_binomial, 100, np.nan)
            assert_raises(ValueError, random.negative_binomial, 100,
                          [np.nan] * 10)

    def test_negative_binomial_p0_exception(self):
        # Verify that p=0 raises an exception.
        with assert_raises(ValueError):
            x = random.negative_binomial(1, 0)

    def test_negative_binomial_invalid_p_n_combination(self):
        # Verify that values of p and n that would result in an overflow
        # or infinite loop raise an exception.
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.negative_binomial, 2**62, 0.1)
            assert_raises(ValueError, random.negative_binomial, [2**62], [0.1])

    def test_noncentral_chisquare(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[ 1.70561552362133, 15.97378184942111],
                            [13.71483425173724, 20.17859633310629],
                            [11.3615477156643 ,  3.67891108738029]])  # noqa: E203
        assert_array_almost_equal(actual, desired, decimal=14)

        actual = random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = np.array([[9.41427665607629e-04, 1.70473157518850e-04],
                            [1.14554372041263e+00, 1.38187755933435e-03],
                            [1.90659181905387e+00, 1.21772577941822e+00]])
        assert_array_almost_equal(actual, desired, decimal=14)

        random = Generator(MT19937(self.seed))
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[0.82947954590419, 1.80139670767078],
                            [6.58720057417794, 7.00491463609814],
                            [6.31101879073157, 6.30982307753005]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                     size=(3, 2))
        desired = np.array([[0.060310671139  , 0.23866058175939],  # noqa: E203
                            [0.86860246709073, 0.2668510459738 ],  # noqa: E202
                            [0.23375780078364, 1.88922102885943]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f_nan(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        assert np.isnan(actual)

    def test_normal(self):
        random = Generator(MT19937(self.seed))
        actual = random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-3.618412914693162,  2.635726692647081],
                            [-2.116923463013243,  0.807460983059643],
                            [ 1.446547137248593,  2.485684213886024]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_normal_0(self):
        assert_equal(random.normal(scale=0), 0)
        assert_raises(ValueError, random.normal, scale=-0.)

    def test_pareto(self):
        random = Generator(MT19937(self.seed))
        actual = random.pareto(a=.123456789, size=(3, 2))
        desired = np.array([[1.0394926776069018e+00, 7.7142534343505773e+04],
                            [7.2640150889064703e-01, 3.4650454783825594e+05],
                            [4.5852344481994740e+04, 6.5851383009539105e+07]])
        # For some reason on 32-bit x86 Ubuntu 12.10 the [1, 0] entry in this
        # matrix differs by 24 nulps. Discussion:
        #   https://mail.python.org/pipermail/numpy-discussion/2012-September/063801.html
        # Consensus is that this is probably some gcc quirk that affects
        # rounding but not in any important way, so we just use a looser
        # tolerance on this test:
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        random = Generator(MT19937(self.seed))
        actual = random.poisson(lam=.123456789, size=(3, 2))
        desired = np.array([[0, 0],
                            [0, 0],
                            [0, 0]])
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        lambig = np.iinfo('int64').max
        lamneg = -1
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)

    def test_power(self):
        random = Generator(MT19937(self.seed))
        actual = random.power(a=.123456789, size=(3, 2))
        desired = np.array([[1.977857368842754e-09, 9.806792196620341e-02],
                            [2.482442984543471e-10, 1.527108843266079e-01],
                            [8.188283434244285e-02, 3.950547209346948e-01]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        random = Generator(MT19937(self.seed))
        actual = random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[4.19494429102666, 16.66920198906598],
                            [3.67184544902662, 17.74695521962917],
                            [16.27935397855501, 21.08355560691792]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        assert_equal(random.rayleigh(scale=0), 0)
        assert_raises(ValueError, random.rayleigh, scale=-0.)

    def test_standard_cauchy(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_cauchy(size=(3, 2))
        desired = np.array([[-1.489437778266206, -3.275389641569784],
                            [ 0.560102864910406, -0.680780916282552],
                            [-1.314912905226277,  0.295852965660225]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_exponential(size=(3, 2), method='inv')
        desired = np.array([[0.102031839440643, 1.229350298474972],
                            [0.088137284693098, 1.459859985522667],
                            [1.093830802293668, 1.256977002164613]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_expoential_type_error(self):
        assert_raises(TypeError, random.standard_exponential, dtype=np.int32)

    def test_standard_gamma(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[0.62970724056362, 1.22379851271008],
                            [3.899412530884  , 4.12479964250139],  # noqa: E203
                            [3.74994102464584, 3.74929307690815]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_gammma_scalar_float(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_gamma(3, dtype=np.float32)
        desired = 2.9242148399353027
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_standard_gamma_float(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[0.62971, 1.2238],
                            [3.89941, 4.1248],
                            [3.74994, 3.74929]])
        assert_array_almost_equal(actual, desired, decimal=5)

    def test_standard_gammma_float_out(self):
        actual = np.zeros((3, 2), dtype=np.float32)
        random = Generator(MT19937(self.seed))
        random.standard_gamma(10.0, out=actual, dtype=np.float32)
        desired = np.array([[10.14987,  7.87012],
                             [ 9.46284, 12.56832],
                             [13.82495,  7.81533]], dtype=np.float32)
        assert_array_almost_equal(actual, desired, decimal=5)

        random = Generator(MT19937(self.seed))
        random.standard_gamma(10.0, out=actual, size=(3, 2), dtype=np.float32)
        assert_array_almost_equal(actual, desired, decimal=5)

    def test_standard_gamma_unknown_type(self):
        assert_raises(TypeError, random.standard_gamma, 1.,
                      dtype='int32')

    def test_out_size_mismatch(self):
        out = np.zeros(10)
        assert_raises(ValueError, random.standard_gamma, 10.0, size=20,
                      out=out)
        assert_raises(ValueError, random.standard_gamma, 10.0, size=(10, 1),
                      out=out)

    def test_standard_gamma_0(self):
        assert_equal(random.standard_gamma(shape=0), 0)
        assert_raises(ValueError, random.standard_gamma, shape=-0.)

    def test_standard_normal(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_normal(size=(3, 2))
        desired = np.array([[-1.870934851846581,  1.25613495182354 ],  # noqa: E202
                            [-1.120190126006621,  0.342002097029821],
                            [ 0.661545174124296,  1.181113712443012]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_normal_unsupported_type(self):
        assert_raises(TypeError, random.standard_normal, dtype=np.int32)

    def test_standard_t(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_t(df=10, size=(3, 2))
        desired = np.array([[-1.484666193042647,  0.30597891831161],
                            [ 1.056684299648085, -0.407312602088507],
                            [ 0.130704414281157, -2.038053410490321]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        random = Generator(MT19937(self.seed))
        actual = random.triangular(left=5.12, mode=10.23, right=20.34,
                                   size=(3, 2))
        desired = np.array([[ 7.86664070590917, 13.6313848513185 ],  # noqa: E202
                            [ 7.68152445215983, 14.36169131136546],
                            [13.16105603911429, 13.72341621856971]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        random = Generator(MT19937(self.seed))
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[2.13306255040998 , 7.816987531021207],  # noqa: E203
                            [2.015436610109887, 8.377577533009589],
                            [7.421792588856135, 7.891185744455209]])
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

    def test_uniform_zero_range(self):
        func = random.uniform
        result = func(1.5, 1.5)
        assert_allclose(result, 1.5)
        result = func([0.0, np.pi], [0.0, np.pi])
        assert_allclose(result, [0.0, np.pi])
        result = func([[2145.12], [2145.12]], [2145.12, 2145.12])
        assert_allclose(result, 2145.12 + np.zeros((2, 2)))

    def test_uniform_neg_range(self):
        func = random.uniform
        assert_raises(ValueError, func, 2, 1)
        assert_raises(ValueError, func,  [1, 2], [1, 1])
        assert_raises(ValueError, func,  [[0, 1], [2, 3]], 2)

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
        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[ 1.107972248690106,  2.841536476232361],
                            [ 1.832602376042457,  1.945511926976032],
                            [-0.260147475776542,  2.058047492231698]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        # check infinite loop, gh-4720
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        assert_(np.isfinite(r).all())

    def test_vonmises_nan(self):
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu=0., kappa=np.nan)
        assert_(np.isnan(r))

    @pytest.mark.parametrize("kappa", [1e4, 1e15])
    def test_vonmises_large_kappa(self, kappa):
        random = Generator(MT19937(self.seed))
        rs = RandomState(random.bit_generator)
        state = random.bit_generator.state

        random_state_vals = rs.vonmises(0, kappa, size=10)
        random.bit_generator.state = state
        gen_vals = random.vonmises(0, kappa, size=10)
        if kappa < 1e6:
            assert_allclose(random_state_vals, gen_vals)
        else:
            assert np.all(random_state_vals != gen_vals)

    @pytest.mark.parametrize("mu", [-7., -np.pi, -3.1, np.pi, 3.2])
    @pytest.mark.parametrize("kappa", [1e-9, 1e-6, 1, 1e3, 1e15])
    def test_vonmises_large_kappa_range(self, mu, kappa):
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu, kappa, 50)
        assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_wald(self):
        random = Generator(MT19937(self.seed))
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[0.26871721804551, 3.2233942732115 ],  # noqa: E202
                            [2.20328374987066, 2.40958405189353],
                            [2.07093587449261, 0.73073890064369]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        random = Generator(MT19937(self.seed))
        actual = random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[0.138613914769468, 1.306463419753191],
                            [0.111623365934763, 1.446570494646721],
                            [1.257145775276011, 1.914247725027957]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_weibull_0(self):
        random = Generator(MT19937(self.seed))
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        assert_raises(ValueError, random.weibull, a=-0.)

    def test_zipf(self):
        random = Generator(MT19937(self.seed))
        actual = random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[  1,   1],
                            [ 10, 867],
                            [354,   2]])
        assert_array_equal(actual, desired)


class TestBroadcast:
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments
    def setup_method(self):
        self.seed = 123456789

    def test_uniform(self):
        random = Generator(MT19937(self.seed))
        low = [0]
        high = [1]
        uniform = random.uniform
        desired = np.array([0.16693771389729, 0.19635129550675, 0.75563050964095])

        random = Generator(MT19937(self.seed))
        actual = random.uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        random = Generator(MT19937(self.seed))
        actual = random.uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        random = Generator(MT19937(self.seed))
        desired = np.array([-0.38736406738527, 0.79594375042255, 0.0197076236097])

        random = Generator(MT19937(self.seed))
        actual = random.normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.normal, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        normal = random.normal
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        desired = np.array([0.18719338682602, 0.73234824491364, 0.17928615186455])

        random = Generator(MT19937(self.seed))
        beta = random.beta
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        random = Generator(MT19937(self.seed))
        actual = random.beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        actual = random.exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.exponential, bad_scale * 3)

    def test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        std_gamma = random.standard_gamma
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        desired = np.array([1.34491986425611, 0.42760990636187, 1.4355697857258])

        random = Generator(MT19937(self.seed))
        gamma = random.gamma
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        gamma = random.gamma
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        desired = np.array([0.07765056244107, 7.72951397913186, 0.05786093891763])

        random = Generator(MT19937(self.seed))
        f = random.f
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        random = Generator(MT19937(self.seed))
        f = random.f
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
        desired = np.array([2.02434240411421, 12.91838601070124, 1.24395160354629])

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))

        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def test_noncentral_f_small_df(self):
        random = Generator(MT19937(self.seed))
        desired = np.array([0.04714867120827, 0.1239390327694])
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_chisquare(self):
        df = [1]
        bad_df = [-1]
        desired = np.array([0.05573640064251, 1.47220224353539, 2.9469379318589])

        random = Generator(MT19937(self.seed))
        actual = random.chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        desired = np.array([0.07710766249436, 5.27829115110304, 0.630732147399])

        random = Generator(MT19937(self.seed))
        nonc_chi = random.noncentral_chisquare
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_chi = random.noncentral_chisquare
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def test_standard_t(self):
        df = [1]
        bad_df = [-1]
        desired = np.array([-1.39498829447098, -1.23058658835223, 0.17207021065983])

        random = Generator(MT19937(self.seed))
        actual = random.standard_t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        desired = np.array([2.25935584988528, 2.23326261461399, -2.84152146503326])

        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.vonmises, mu * 3, bad_kappa)

        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        a = [1]
        bad_a = [-1]
        desired = np.array([0.95905052946317, 0.2383810889437, 1.04988745750013])

        random = Generator(MT19937(self.seed))
        actual = random.pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def test_weibull(self):
        a = [1]
        bad_a = [-1]
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        actual = random.weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def test_power(self):
        a = [1]
        bad_a = [-1]
        desired = np.array([0.48954864361052, 0.19249412888486, 0.51216834058807])

        random = Generator(MT19937(self.seed))
        actual = random.power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.power, bad_a * 3)

    def test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = np.array([-1.09698732625119, -0.93470271947368, 0.71592671378202])

        random = Generator(MT19937(self.seed))
        laplace = random.laplace
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        laplace = random.laplace
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = np.array([1.70020068231762, 1.52054354273631, -0.34293267607081])

        random = Generator(MT19937(self.seed))
        gumbel = random.gumbel
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        gumbel = random.gumbel
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = np.array([-1.607487640433, -1.40925686003678, 1.12887112820397])

        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.logistic, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.logistic, loc, bad_scale * 3)
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        desired = np.array([0.67884390500697, 2.21653186290321, 1.01990310084276])

        random = Generator(MT19937(self.seed))
        lognormal = random.lognormal
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)

        random = Generator(MT19937(self.seed))
        actual = random.lognormal(mean, sigma * 3)
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        desired = np.array(
            [1.1597068009872629,
             0.6539188836253857,
             1.1981526554349398]
        )

        random = Generator(MT19937(self.seed))
        actual = random.rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.rayleigh, bad_scale * 3)

    def test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        desired = np.array([0.38052407392905, 0.50701641508592, 0.484935249864])

        random = Generator(MT19937(self.seed))
        actual = random.wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        actual = random.wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.wald, bad_mean, scale * 3)
        assert_raises(ValueError, random.wald, mean, bad_scale * 3)

    def test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        desired = np.array([1.57781954604754, 1.62665986867957, 2.30090130831326])

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
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
        desired = np.array([0, 0, 1])

        random = Generator(MT19937(self.seed))
        binom = random.binomial
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        random = Generator(MT19937(self.seed))
        actual = random.binomial(n, p * 3)
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
        desired = np.array([0, 2, 1], dtype=np.int64)

        random = Generator(MT19937(self.seed))
        neg_binom = random.negative_binomial
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        random = Generator(MT19937(self.seed))
        neg_binom = random.negative_binomial
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def test_poisson(self):

        lam = [1]
        bad_lam_one = [-1]
        desired = np.array([0, 0, 3])

        random = Generator(MT19937(self.seed))
        max_lam = random._poisson_lam_max
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def test_zipf(self):
        a = [2]
        bad_a = [0]
        desired = np.array([1, 8, 1])

        random = Generator(MT19937(self.seed))
        zipf = random.zipf
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
        desired = np.array([1, 1, 3])

        random = Generator(MT19937(self.seed))
        geometric = random.geometric
        actual = geometric(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geometric, bad_p_one * 3)
        assert_raises(ValueError, geometric, bad_p_two * 3)

    def test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [-1]
        bad_nsample_two = [4]
        desired = np.array([0, 0, 1])

        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, random.hypergeometric, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, nbad, bad_nsample_one)  # noqa: E501
        assert_raises(ValueError, random.hypergeometric, ngood * 3, nbad, bad_nsample_two)  # noqa: E501

        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, random.hypergeometric, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood, nbad * 3, bad_nsample_one)  # noqa: E501
        assert_raises(ValueError, random.hypergeometric, ngood, nbad * 3, bad_nsample_two)  # noqa: E501

        random = Generator(MT19937(self.seed))
        hypergeom = random.hypergeometric
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, -1)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

        # ValueError for arguments that are too big.
        assert_raises(ValueError, hypergeom, 2**30, 10, 20)
        assert_raises(ValueError, hypergeom, 999, 2**31, 50)
        assert_raises(ValueError, hypergeom, 999, [2**29, 2**30], 1000)

    def test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        desired = np.array([1, 1, 1])

        random = Generator(MT19937(self.seed))
        logseries = random.logseries
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)

    def test_multinomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[0, 0, 2, 1, 2, 0],
                             [2, 3, 6, 4, 2, 3]],
                            [[1, 0, 1, 0, 2, 1],
                             [7, 2, 2, 1, 4, 4]],
                            [[0, 2, 0, 1, 2, 0],
                             [3, 2, 3, 3, 4, 5]]], dtype=np.int64)
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [1 / 6.] * 6)
        desired = np.array([[0, 0, 2, 1, 2, 0],
                            [2, 3, 6, 4, 2, 3]], dtype=np.int64)
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [[1 / 6.] * 6] * 2)
        desired = np.array([[0, 0, 2, 1, 2, 0],
                            [2, 3, 6, 4, 2, 3]], dtype=np.int64)
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.multinomial([[5], [20]], [[1 / 6.] * 6] * 2)
        desired = np.array([[[0, 0, 2, 1, 2, 0],
                             [0, 0, 2, 1, 1, 1]],
                            [[4, 2, 3, 3, 5, 3],
                             [7, 2, 2, 1, 4, 4]]], dtype=np.int64)
        assert_array_equal(actual, desired)

    @pytest.mark.parametrize("n", [10,
                                   np.array([10, 10]),
                                   np.array([[[10]], [[10]]])
                                   ]
                             )
    def test_multinomial_pval_broadcast(self, n):
        random = Generator(MT19937(self.seed))
        pvals = np.array([1 / 4] * 4)
        actual = random.multinomial(n, pvals)
        n_shape = () if isinstance(n, int) else n.shape
        expected_shape = n_shape + (4,)
        assert actual.shape == expected_shape
        pvals = np.vstack([pvals, pvals])
        actual = random.multinomial(n, pvals)
        expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1]) + (4,)
        assert actual.shape == expected_shape

        pvals = np.vstack([[pvals], [pvals]])
        actual = random.multinomial(n, pvals)
        expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1])
        assert actual.shape == expected_shape + (4,)
        actual = random.multinomial(n, pvals, size=(3, 2) + expected_shape)
        assert actual.shape == (3, 2) + expected_shape + (4,)

        with pytest.raises(ValueError):
            # Ensure that size is not broadcast
            actual = random.multinomial(n, pvals, size=(1,) * 6)

    def test_invalid_pvals_broadcast(self):
        random = Generator(MT19937(self.seed))
        pvals = [[1 / 6] * 6, [1 / 4] * 6]
        assert_raises(ValueError, random.multinomial, 1, pvals)
        assert_raises(ValueError, random.multinomial, 6, 0.5)

    def test_empty_outputs(self):
        random = Generator(MT19937(self.seed))
        actual = random.multinomial(np.empty((10, 0, 6), "i8"), [1 / 6] * 6)
        assert actual.shape == (10, 0, 6, 6)
        actual = random.multinomial(12, np.empty((10, 0, 10)))
        assert actual.shape == (10, 0, 10)
        actual = random.multinomial(np.empty((3, 0, 7), "i8"),
                                    np.empty((3, 0, 7, 4)))
        assert actual.shape == (3, 0, 7, 4)


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
        t = [Thread(target=function, args=(Generator(MT19937(s)), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(Generator(MT19937(s)), o)

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

    def test_integers(self, endpoint):
        itype = [np.bool, np.int8, np.uint8, np.int16, np.uint16,
                 np.int32, np.uint32, np.int64, np.uint64]
        func = random.integers
        high = np.array([1])
        low = np.array([0])

        for dt in itype:
            out = func(low, high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low[0], high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low, high[0], endpoint=endpoint, dtype=dt)
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


@pytest.mark.parametrize("config", JUMP_TEST_DATA)
def test_jumped(config):
    # Each config contains the initial seed, a number of raw steps
    # the sha256 hashes of the initial and the final states' keys and
    # the position of the initial and the final state.
    # These were produced using the original C implementation.
    seed = config["seed"]
    steps = config["steps"]

    mt19937 = MT19937(seed)
    # Burn step
    mt19937.random_raw(steps)
    key = mt19937.state["state"]["key"]
    if sys.byteorder == 'big':
        key = key.byteswap()
    sha256 = hashlib.sha256(key)
    assert mt19937.state["state"]["pos"] == config["initial"]["pos"]
    assert sha256.hexdigest() == config["initial"]["key_sha256"]

    jumped = mt19937.jumped()
    key = jumped.state["state"]["key"]
    if sys.byteorder == 'big':
        key = key.byteswap()
    sha256 = hashlib.sha256(key)
    assert jumped.state["state"]["pos"] == config["jumped"]["pos"]
    assert sha256.hexdigest() == config["jumped"]["key_sha256"]


def test_broadcast_size_error():
    mu = np.ones(3)
    sigma = np.ones((4, 3))
    size = (10, 4, 2)
    assert random.normal(mu, sigma, size=(5, 4, 3)).shape == (5, 4, 3)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=size)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=(1, 3))
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=(4, 1, 1))
    # 1 arg
    shape = np.ones((4, 3))
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=size)
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=(3,))
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=3)
    # Check out
    out = np.empty(size)
    with pytest.raises(ValueError):
        random.standard_gamma(shape, out=out)

    # 2 arg
    with pytest.raises(ValueError):
        random.binomial(1, [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], 0.3, size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.multinomial([2, 2], [.3, .7], size=(2, 1))

    # 3 arg
    a = random.chisquare(5, size=3)
    b = random.chisquare(5, size=(4, 3))
    c = random.chisquare(5, size=(5, 4, 3))
    assert random.noncentral_f(a, b, c).shape == (5, 4, 3)
    with pytest.raises(ValueError, match=r"Output size \(6, 5, 1, 1\) is"):
        random.noncentral_f(a, b, c, size=(6, 5, 1, 1))


def test_broadcast_size_scalar():
    mu = np.ones(3)
    sigma = np.ones(3)
    random.normal(mu, sigma, size=3)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=2)


def test_ragged_shuffle():
    # GH 18142
    seq = [[], [], 1]
    gen = Generator(MT19937(0))
    assert_no_warnings(gen.shuffle, seq)
    assert seq == [1, [], []]


@pytest.mark.parametrize("high", [-2, [-2]])
@pytest.mark.parametrize("endpoint", [True, False])
def test_single_arg_integer_exception(high, endpoint):
    # GH 14333
    gen = Generator(MT19937(0))
    msg = 'high < 0' if endpoint else 'high <= 0'
    with pytest.raises(ValueError, match=msg):
        gen.integers(high, endpoint=endpoint)
    msg = 'low > high' if endpoint else 'low >= high'
    with pytest.raises(ValueError, match=msg):
        gen.integers(-1, high, endpoint=endpoint)
    with pytest.raises(ValueError, match=msg):
        gen.integers([-1], high, endpoint=endpoint)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_c_contig_req_out(dtype):
    # GH 18704
    out = np.empty((2, 3), order="F", dtype=dtype)
    shape = [1, 2, 3]
    with pytest.raises(ValueError, match="Supplied output array"):
        random.standard_gamma(shape, out=out, dtype=dtype)
    with pytest.raises(ValueError, match="Supplied output array"):
        random.standard_gamma(shape, out=out, size=out.shape, dtype=dtype)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dist", [random.standard_normal, random.random])
def test_contig_req_out(dist, order, dtype):
    # GH 18704
    out = np.empty((2, 3), dtype=dtype, order=order)
    variates = dist(out=out, dtype=dtype)
    assert variates is out
    variates = dist(out=out, dtype=dtype, size=out.shape)
    assert variates is out


def test_generator_ctor_old_style_pickle():
    rg = np.random.Generator(np.random.PCG64DXSM(0))
    rg.standard_normal(1)
    # Directly call reduce which is used in pickling
    ctor, (bit_gen, ), _ = rg.__reduce__()
    # Simulate unpickling an old pickle that only has the name
    assert bit_gen.__class__.__name__ == "PCG64DXSM"
    print(ctor)
    b = ctor(*("PCG64DXSM",))
    print(b)
    b.bit_generator.state = bit_gen.state
    state_b = b.bit_generator.state
    assert bit_gen.state == state_b


def test_pickle_preserves_seed_sequence():
    # GH 26234
    # Add explicit test that bit generators preserve seed sequences
    import pickle

    rg = np.random.Generator(np.random.PCG64DXSM(20240411))
    ss = rg.bit_generator.seed_seq
    rg_plk = pickle.loads(pickle.dumps(rg))
    ss_plk = rg_plk.bit_generator.seed_seq
    assert_equal(ss.state, ss_plk.state)
    assert_equal(ss.pool, ss_plk.pool)

    rg.bit_generator.seed_seq.spawn(10)
    rg_plk = pickle.loads(pickle.dumps(rg))
    ss_plk = rg_plk.bit_generator.seed_seq
    assert_equal(ss.state, ss_plk.state)


@pytest.mark.parametrize("version", [121, 126])
def test_legacy_pickle(version):
    # Pickling format was changes in 1.22.x and in 2.0.x
    import gzip
    import pickle

    base_path = os.path.split(os.path.abspath(__file__))[0]
    pkl_file = os.path.join(
        base_path, "data", f"generator_pcg64_np{version}.pkl.gz"
    )
    with gzip.open(pkl_file) as gz:
        rg = pickle.load(gz)
    state = rg.bit_generator.state['state']

    assert isinstance(rg, Generator)
    assert isinstance(rg.bit_generator, np.random.PCG64)
    assert state['state'] == 35399562948360463058890781895381311971
    assert state['inc'] == 87136372517582989555478159403783844777

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\_version_info.py ===
# SPDX-License-Identifier: MIT


from functools import total_ordering

from ._funcs import astuple
from ._make import attrib, attrs


@total_ordering
@attrs(eq=False, order=False, slots=True, frozen=True)
class VersionInfo:
    """
    A version object that can be compared to tuple of length 1--4:

    >>> attr.VersionInfo(19, 1, 0, "final")  <= (19, 2)
    True
    >>> attr.VersionInfo(19, 1, 0, "final") < (19, 1, 1)
    True
    >>> vi = attr.VersionInfo(19, 2, 0, "final")
    >>> vi < (19, 1, 1)
    False
    >>> vi < (19,)
    False
    >>> vi == (19, 2,)
    True
    >>> vi == (19, 2, 1)
    False

    .. versionadded:: 19.2
    """

    year = attrib(type=int)
    minor = attrib(type=int)
    micro = attrib(type=int)
    releaselevel = attrib(type=str)

    @classmethod
    def _from_version_string(cls, s):
        """
        Parse *s* and return a _VersionInfo.
        """
        v = s.split(".")
        if len(v) == 3:
            v.append("final")

        return cls(
            year=int(v[0]), minor=int(v[1]), micro=int(v[2]), releaselevel=v[3]
        )

    def _ensure_tuple(self, other):
        """
        Ensure *other* is a tuple of a valid length.

        Returns a possibly transformed *other* and ourselves as a tuple of
        the same length as *other*.
        """

        if self.__class__ is other.__class__:
            other = astuple(other)

        if not isinstance(other, tuple):
            raise NotImplementedError

        if not (1 <= len(other) <= 4):
            raise NotImplementedError

        return astuple(self)[: len(other)], other

    def __eq__(self, other):
        try:
            us, them = self._ensure_tuple(other)
        except NotImplementedError:
            return NotImplemented

        return us == them

    def __lt__(self, other):
        try:
            us, them = self._ensure_tuple(other)
        except NotImplementedError:
            return NotImplemented

        # Since alphabetically "dev0" < "final" < "post1" < "post2", we don't
        # have to do anything special with releaselevel for now.
        return us < them

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dotenv\variables.py ===
import re
from abc import ABCMeta, abstractmethod
from typing import Iterator, Mapping, Optional, Pattern

_posix_variable: Pattern[str] = re.compile(
    r"""
    \$\{
        (?P<name>[^\}:]*)
        (?::-
            (?P<default>[^\}]*)
        )?
    \}
    """,
    re.VERBOSE,
)


class Atom(metaclass=ABCMeta):
    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @abstractmethod
    def resolve(self, env: Mapping[str, Optional[str]]) -> str: ...


class Literal(Atom):
    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Literal(value={self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((self.__class__, self.value))

    def resolve(self, env: Mapping[str, Optional[str]]) -> str:
        return self.value


class Variable(Atom):
    def __init__(self, name: str, default: Optional[str]) -> None:
        self.name = name
        self.default = default

    def __repr__(self) -> str:
        return f"Variable(name={self.name}, default={self.default})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.name, self.default) == (other.name, other.default)

    def __hash__(self) -> int:
        return hash((self.__class__, self.name, self.default))

    def resolve(self, env: Mapping[str, Optional[str]]) -> str:
        default = self.default if self.default is not None else ""
        result = env.get(self.name, default)
        return result if result is not None else ""


def parse_variables(value: str) -> Iterator[Atom]:
    cursor = 0

    for match in _posix_variable.finditer(value):
        (start, end) = match.span()
        name = match["name"]
        default = match["default"]

        if start > cursor:
            yield Literal(value=value[cursor:start])

        yield Variable(name=name, default=default)
        cursor = end

    length = len(value)
    if cursor < length:
        yield Literal(value=value[cursor:length])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\compat.py ===
# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" A tiny version of `six` to help with backwards compability. Also includes
 compatibility helpers for numpy. """

import sys

PY2 = sys.version_info[0] == 2
PY26 = sys.version_info[0:2] == (2, 6)
PY27 = sys.version_info[0:2] == (2, 7)
PY275 = sys.version_info[0:3] >= (2, 7, 5)
PY3 = sys.version_info[0] == 3
PY34 = sys.version_info[0:2] >= (3, 4)

if PY3:
    import importlib.machinery
    string_types = (str,)
    binary_types = (bytes,bytearray)
    range_func = range
    memoryview_type = memoryview
    struct_bool_decl = "?"
else:
    import imp
    string_types = (unicode,)
    if PY26 or PY27:
        binary_types = (str,bytearray)
    else:
        binary_types = (str,)
    range_func = xrange
    if PY26 or (PY27 and not PY275):
        memoryview_type = buffer
        struct_bool_decl = "<b"
    else:
        memoryview_type = memoryview
        struct_bool_decl = "?"

# Helper functions to facilitate making numpy optional instead of required

def import_numpy():
    """
    Returns the numpy module if it exists on the system,
    otherwise returns None.
    """
    if PY3:
        numpy_exists = (
            importlib.machinery.PathFinder.find_spec('numpy') is not None)
    else:
        try:
            imp.find_module('numpy')
            numpy_exists = True
        except ImportError:
            numpy_exists = False

    if numpy_exists:
        # We do this outside of try/except block in case numpy exists
        # but is not installed correctly. We do not want to catch an
        # incorrect installation which would manifest as an
        # ImportError.
        import numpy as np
    else:
        np = None

    return np


class NumpyRequiredForThisFeature(RuntimeError):
    """
    Error raised when user tries to use a feature that
    requires numpy without having numpy installed.
    """
    pass


# NOTE: Future Jython support may require code here (look at `six`).

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\__init__.py ===
"""Fortran to Python Interface Generator.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the terms
of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__all__ = ['run_main', 'get_include']

import os
import subprocess
import sys
import warnings

from numpy.exceptions import VisibleDeprecationWarning

from . import diagnose, f2py2e

run_main = f2py2e.run_main
main = f2py2e.main


def get_include():
    """
    Return the directory that contains the ``fortranobject.c`` and ``.h`` files.

    .. note::

        This function is not needed when building an extension with
        `numpy.distutils` directly from ``.f`` and/or ``.pyf`` files
        in one go.

    Python extension modules built with f2py-generated code need to use
    ``fortranobject.c`` as a source file, and include the ``fortranobject.h``
    header. This function can be used to obtain the directory containing
    both of these files.

    Returns
    -------
    include_path : str
        Absolute path to the directory containing ``fortranobject.c`` and
        ``fortranobject.h``.

    Notes
    -----
    .. versionadded:: 1.21.1

    Unless the build system you are using has specific support for f2py,
    building a Python extension using a ``.pyf`` signature file is a two-step
    process. For a module ``mymod``:

    * Step 1: run ``python -m numpy.f2py mymod.pyf --quiet``. This
      generates ``mymodmodule.c`` and (if needed)
      ``mymod-f2pywrappers.f`` files next to ``mymod.pyf``.
    * Step 2: build your Python extension module. This requires the
      following source files:

      * ``mymodmodule.c``
      * ``mymod-f2pywrappers.f`` (if it was generated in Step 1)
      * ``fortranobject.c``

    See Also
    --------
    numpy.get_include : function that returns the numpy include directory

    """
    return os.path.join(os.path.dirname(__file__), 'src')


def __getattr__(attr):

    # Avoid importing things that aren't needed for building
    # which might import the main numpy module
    if attr == "test":
        from numpy._pytesttester import PytestTester
        test = PytestTester(__name__)
        return test

    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def __dir__():
    return list(globals().keys() | {"test"})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\_examples\numba\extending.py ===
from timeit import timeit

import numba as nb

import numpy as np
from numpy.random import PCG64

bit_gen = PCG64()
next_d = bit_gen.cffi.next_double
state_addr = bit_gen.cffi.state_address

def normals(n, state):
    out = np.empty(n)
    for i in range((n + 1) // 2):
        x1 = 2.0 * next_d(state) - 1.0
        x2 = 2.0 * next_d(state) - 1.0
        r2 = x1 * x1 + x2 * x2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * next_d(state) - 1.0
            x2 = 2.0 * next_d(state) - 1.0
            r2 = x1 * x1 + x2 * x2
        f = np.sqrt(-2.0 * np.log(r2) / r2)
        out[2 * i] = f * x1
        if 2 * i + 1 < n:
            out[2 * i + 1] = f * x2
    return out


# Compile using Numba
normalsj = nb.jit(normals, nopython=True)
# Must use state address not state with numba
n = 10000

def numbacall():
    return normalsj(n, state_addr)


rg = np.random.Generator(PCG64())

def numpycall():
    return rg.normal(size=n)


# Check that the functions work
r1 = numbacall()
r2 = numpycall()
assert r1.shape == (n,)
assert r1.shape == r2.shape

t1 = timeit(numbacall, number=1000)
print(f'{t1:.2f} secs for {n} PCG64 (Numba/PCG64) gaussian randoms')
t2 = timeit(numpycall, number=1000)
print(f'{t2:.2f} secs for {n} PCG64 (NumPy/PCG64) gaussian randoms')

# example 2

next_u32 = bit_gen.ctypes.next_uint32
ctypes_state = bit_gen.ctypes.state

@nb.jit(nopython=True)
def bounded_uint(lb, ub, state):
    mask = delta = ub - lb
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    val = next_u32(state) & mask
    while val > delta:
        val = next_u32(state) & mask

    return lb + val


print(bounded_uint(323, 2394691, ctypes_state.value))


@nb.jit(nopython=True)
def bounded_uints(lb, ub, n, state):
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        out[i] = bounded_uint(lb, ub, state)


bounded_uints(323, 2394691, 10000000, ctypes_state.value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_model_processor.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ort_flatbuffers_py.fbs as fbs

from .operator_type_usage_processors import OperatorTypeUsageManager


class OrtFormatModelProcessor:
    "Class to process an ORT format model and determine required operators and types."

    def __init__(self, model_path: str, required_ops: dict, processors: OperatorTypeUsageManager):
        """
        Initialize ORT format model processor
        :param model_path: Path to model to load
        :param required_ops: Dictionary required operator information will be added to.
        :param processors: Operator type usage processors which will be called for each matching Node.
        """
        self._required_ops = required_ops  # dictionary of {domain: {opset:[operators]}}
        self._file = open(model_path, "rb").read()  # noqa: SIM115
        self._buffer = bytearray(self._file)
        if not fbs.InferenceSession.InferenceSession.InferenceSessionBufferHasIdentifier(self._buffer, 0):
            raise RuntimeError(f"File does not appear to be a valid ORT format model: '{model_path}'")
        self._model = fbs.InferenceSession.InferenceSession.GetRootAsInferenceSession(self._buffer, 0).Model()
        self._op_type_processors = processors

    @staticmethod
    def _setup_type_info(graph: fbs.Graph, outer_scope_value_typeinfo={}):  # noqa: B006
        """
        Setup the node args for this level of Graph.
        We copy the current list which represents the outer scope values, and add the local node args to that
        to create the valid list of values for the current Graph.
        :param graph: Graph to create NodeArg list for
        :param outer_scope_value_typeinfo: TypeInfo for outer scope values. Empty for the top-level graph in a model.
        :return: Dictionary of NodeArg name to TypeInfo
        """
        value_name_to_typeinfo = outer_scope_value_typeinfo.copy()
        for j in range(graph.NodeArgsLength()):
            n = graph.NodeArgs(j)
            value_name_to_typeinfo[n.Name()] = n.Type()  # TypeInfo for this NodeArg's name

        return value_name_to_typeinfo

    def _add_required_op(self, domain: str, opset: int, op_type: str):
        if domain not in self._required_ops:
            self._required_ops[domain] = {opset: {op_type}}
        elif opset not in self._required_ops[domain]:
            self._required_ops[domain][opset] = {op_type}
        else:
            self._required_ops[domain][opset].add(op_type)

    def _process_graph(self, graph: fbs.Graph, outer_scope_value_typeinfo: dict):
        """
        Process one level of the Graph, descending into any subgraphs when they are found
        :param outer_scope_value_typeinfo: Outer scope NodeArg dictionary from ancestor graphs
        """
        # Merge the TypeInfo for all values in this level of the graph with the outer scope value TypeInfo.
        value_name_to_typeinfo = OrtFormatModelProcessor._setup_type_info(graph, outer_scope_value_typeinfo)

        for i in range(graph.NodesLength()):
            node = graph.Nodes(i)

            optype = node.OpType().decode()
            domain = node.Domain().decode() or "ai.onnx"  # empty domain defaults to ai.onnx

            self._add_required_op(domain, node.SinceVersion(), optype)

            if self._op_type_processors:
                self._op_type_processors.process_node(node, value_name_to_typeinfo)

            # Read all the attributes
            for j in range(node.AttributesLength()):
                attr = node.Attributes(j)
                attr_type = attr.Type()
                if attr_type == fbs.AttributeType.AttributeType.GRAPH:
                    self._process_graph(attr.G(), value_name_to_typeinfo)
                elif attr_type == fbs.AttributeType.AttributeType.GRAPHS:
                    # the ONNX spec doesn't currently define any operators that have multiple graphs in an attribute
                    # so entering this 'elif' isn't currently possible
                    for k in range(attr.GraphsLength()):
                        self._process_graph(attr.Graphs(k), value_name_to_typeinfo)

    def process(self):
        graph = self._model.Graph()
        outer_scope_value_typeinfo = {}  # no outer scope values for the main graph
        self._process_graph(graph, outer_scope_value_typeinfo)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\metadata\_json.py ===
# Extracted from https://github.com/pfmoore/pkg_metadata

from email.header import Header, decode_header, make_header
from email.message import Message
from typing import Any, Dict, List, Union, cast

METADATA_FIELDS = [
    # Name, Multiple-Use
    ("Metadata-Version", False),
    ("Name", False),
    ("Version", False),
    ("Dynamic", True),
    ("Platform", True),
    ("Supported-Platform", True),
    ("Summary", False),
    ("Description", False),
    ("Description-Content-Type", False),
    ("Keywords", False),
    ("Home-page", False),
    ("Download-URL", False),
    ("Author", False),
    ("Author-email", False),
    ("Maintainer", False),
    ("Maintainer-email", False),
    ("License", False),
    ("License-Expression", False),
    ("License-File", True),
    ("Classifier", True),
    ("Requires-Dist", True),
    ("Requires-Python", False),
    ("Requires-External", True),
    ("Project-URL", True),
    ("Provides-Extra", True),
    ("Provides-Dist", True),
    ("Obsoletes-Dist", True),
]


def json_name(field: str) -> str:
    return field.lower().replace("-", "_")


def msg_to_json(msg: Message) -> Dict[str, Any]:
    """Convert a Message object into a JSON-compatible dictionary."""

    def sanitise_header(h: Union[Header, str]) -> str:
        if isinstance(h, Header):
            chunks = []
            for bytes, encoding in decode_header(h):
                if encoding == "unknown-8bit":
                    try:
                        # See if UTF-8 works
                        bytes.decode("utf-8")
                        encoding = "utf-8"
                    except UnicodeDecodeError:
                        # If not, latin1 at least won't fail
                        encoding = "latin1"
                chunks.append((bytes, encoding))
            return str(make_header(chunks))
        return str(h)

    result = {}
    for field, multi in METADATA_FIELDS:
        if field not in msg:
            continue
        key = json_name(field)
        if multi:
            value: Union[str, List[str]] = [
                sanitise_header(v) for v in msg.get_all(field)  # type: ignore
            ]
        else:
            value = sanitise_header(msg.get(field))  # type: ignore
            if key == "keywords":
                # Accept both comma-separated and space-separated
                # forms, for better compatibility with old data.
                if "," in value:
                    value = [v.strip() for v in value.split(",")]
                else:
                    value = value.split()
        result[key] = value

    payload = cast(str, msg.get_payload())
    if payload:
        result["description"] = payload

    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\scope.py ===
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .highlighter import ReprHighlighter
from .panel import Panel
from .pretty import Pretty
from .table import Table
from .text import Text, TextType

if TYPE_CHECKING:
    from .console import ConsoleRenderable


def render_scope(
    scope: "Mapping[str, Any]",
    *,
    title: Optional[TextType] = None,
    sort_keys: bool = True,
    indent_guides: bool = False,
    max_length: Optional[int] = None,
    max_string: Optional[int] = None,
) -> "ConsoleRenderable":
    """Render python variables in a given scope.

    Args:
        scope (Mapping): A mapping containing variable names and values.
        title (str, optional): Optional title. Defaults to None.
        sort_keys (bool, optional): Enable sorting of items. Defaults to True.
        indent_guides (bool, optional): Enable indentation guides. Defaults to False.
        max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to None.
        max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to None.

    Returns:
        ConsoleRenderable: A renderable object.
    """
    highlighter = ReprHighlighter()
    items_table = Table.grid(padding=(0, 1), expand=False)
    items_table.add_column(justify="right")

    def sort_items(item: Tuple[str, Any]) -> Tuple[bool, str]:
        """Sort special variables first, then alphabetically."""
        key, _ = item
        return (not key.startswith("__"), key.lower())

    items = sorted(scope.items(), key=sort_items) if sort_keys else scope.items()
    for key, value in items:
        key_text = Text.assemble(
            (key, "scope.key.special" if key.startswith("__") else "scope.key"),
            (" =", "scope.equals"),
        )
        items_table.add_row(
            key_text,
            Pretty(
                value,
                highlighter=highlighter,
                indent_guides=indent_guides,
                max_length=max_length,
                max_string=max_string,
            ),
        )
    return Panel.fit(
        items_table,
        title=title,
        border_style="scope.border",
        padding=(0, 1),
    )


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich import print

    print()

    def test(foo: float, bar: float) -> None:
        list_of_things = [1, 2, 3, None, 4, True, False, "Hello World"]
        dict_of_things = {
            "version": "1.1",
            "method": "confirmFruitPurchase",
            "params": [["apple", "orange", "mangoes", "pomelo"], 1.123],
            "id": "194521489",
        }
        print(render_scope(locals(), title="[i]locals", sort_keys=False))

    test(20.3423, 3.1427)
    print()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\_build_autolev_antlr.py ===
import os
import subprocess
import glob

from sympy.utilities.misc import debug

here = os.path.dirname(__file__)
grammar_file = os.path.abspath(os.path.join(here, "Autolev.g4"))
dir_autolev_antlr = os.path.join(here, "_antlr")

header = '''\
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
'''


def check_antlr_version():
    debug("Checking antlr4 version...")

    try:
        debug(subprocess.check_output(["antlr4"])
              .decode('utf-8').split("\n")[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        debug("The 'antlr4' command line tool is not installed, "
              "or not on your PATH.\n"
              "> Please refer to the README.md file for more information.")
        return False


def build_parser(output_dir=dir_autolev_antlr):
    check_antlr_version()

    debug("Updating ANTLR-generated code in {}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "__init__.py"), "w+") as fp:
        fp.write(header)

    args = [
        "antlr4",
        grammar_file,
        "-o", output_dir,
        "-no-visitor",
    ]

    debug("Running code generation...\n\t$ {}".format(" ".join(args)))
    subprocess.check_output(args, cwd=output_dir)

    debug("Applying headers, removing unnecessary files and renaming...")
    # Handle case insensitive file systems. If the files are already
    # generated, they will be written to autolev* but Autolev*.* won't match them.
    for path in (glob.glob(os.path.join(output_dir, "Autolev*.*")) or
        glob.glob(os.path.join(output_dir, "autolev*.*"))):

        # Remove files ending in .interp or .tokens as they are not needed.
        if not path.endswith(".py"):
            os.unlink(path)
            continue

        new_path = os.path.join(output_dir, os.path.basename(path).lower())
        with open(path, 'r') as f:
            lines = [line.rstrip().replace('AutolevParser import', 'autolevparser import') +'\n'
                     for line in f]

        os.unlink(path)

        with open(new_path, "w") as out_file:
            offset = 0
            while lines[offset].startswith('#'):
                offset += 1
            out_file.write(header)
            out_file.writelines(lines[offset:])

        debug("\t{}".format(new_path))

    return True


if __name__ == "__main__":
    build_parser()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_highlevel_open_unix_stream.py ===
import os
import socket
import sys
import tempfile
from typing import TYPE_CHECKING

import pytest

from trio import Path, open_unix_socket
from trio._highlevel_open_unix_stream import close_on_error

assert not TYPE_CHECKING or sys.platform != "win32"

skip_if_not_unix = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Needs unix socket support",
)


@skip_if_not_unix
def test_close_on_error() -> None:
    class CloseMe:
        closed = False

        def close(self) -> None:
            self.closed = True

    with close_on_error(CloseMe()) as c:
        pass
    assert not c.closed

    with pytest.raises(RuntimeError):
        with close_on_error(CloseMe()) as c:
            raise RuntimeError
    assert c.closed


@skip_if_not_unix
@pytest.mark.parametrize("filename", [4, 4.5])
async def test_open_with_bad_filename_type(filename: float) -> None:
    with pytest.raises(TypeError):
        await open_unix_socket(filename)  # type: ignore[arg-type]


@skip_if_not_unix
async def test_open_bad_socket() -> None:
    # mktemp is marked as insecure, but that's okay, we don't want the file to
    # exist
    name = tempfile.mktemp()
    with pytest.raises(FileNotFoundError):
        await open_unix_socket(name)


@skip_if_not_unix
async def test_open_unix_socket() -> None:
    for name_type in [Path, str]:
        name = tempfile.mktemp()
        serv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with serv_sock:
            serv_sock.bind(name)
            try:
                serv_sock.listen(1)

                # The actual function we're testing
                unix_socket = await open_unix_socket(name_type(name))

                async with unix_socket:
                    client, _ = serv_sock.accept()
                    with client:
                        await unix_socket.send_all(b"test")
                        assert client.recv(2048) == b"test"

                        client.sendall(b"response")
                        received = await unix_socket.receive_some(2048)
                        assert received == b"response"
            finally:
                os.unlink(name)


@pytest.mark.skipif(hasattr(socket, "AF_UNIX"), reason="Test for non-unix platforms")
async def test_error_on_no_unix() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"^Unix sockets are not supported on this platform$",
    ):
        await open_unix_socket("")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\onnx_randomizer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# An offline standalone script to declassify an ONNX model by randomizing the tensor data in initializers.
# The ORT Performance may change especially on generative models.

import argparse
from pathlib import Path

import numpy as np
from onnx import load_model, numpy_helper, onnx_pb, save_model

# An experimental small value for differentiating shape data and weights.
# The tensor data with larger size can't be shape data.
# User may adjust this value as needed.
SIZE_THRESHOLD = 10


def graph_iterator(model, func):
    graph_queue = [model.graph]
    while graph_queue:
        graph = graph_queue.pop(0)
        func(graph)
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx_pb.AttributeProto.AttributeType.GRAPH:
                    assert isinstance(attr.g, onnx_pb.GraphProto)
                    graph_queue.append(attr.g)
                if attr.type == onnx_pb.AttributeProto.AttributeType.GRAPHS:
                    for g in attr.graphs:
                        assert isinstance(g, onnx_pb.GraphProto)
                        graph_queue.append(g)


def randomize_graph_initializer(graph):
    for i_tensor in graph.initializer:
        array = numpy_helper.to_array(i_tensor)
        # TODO: need to find a better way to differentiate shape data and weights.
        if array.size > SIZE_THRESHOLD:
            random_array = np.random.uniform(array.min(), array.max(), size=array.shape).astype(array.dtype)
            o_tensor = numpy_helper.from_array(random_array, i_tensor.name)
            i_tensor.CopyFrom(o_tensor)


def main():
    parser = argparse.ArgumentParser(description="Randomize the weights of an ONNX model")
    parser.add_argument("-m", type=str, required=True, help="input onnx model path")
    parser.add_argument("-o", type=str, required=True, help="output onnx model path")
    parser.add_argument(
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="Store or Save in external data format",
    )
    parser.add_argument(
        "--all_tensors_to_one_file",
        required=False,
        action="store_true",
        help="Save all tensors to one file",
    )
    args = parser.parse_args()

    data_path = None
    if args.use_external_data_format:
        if Path(args.m).parent == Path(args.o).parent:
            raise RuntimeError("Please specify output directory with different parent path to input directory.")
        if args.all_tensors_to_one_file:
            data_path = Path(args.o).name + ".data"

    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    onnx_model = load_model(args.m, load_external_data=args.use_external_data_format)
    graph_iterator(onnx_model, randomize_graph_initializer)
    save_model(
        onnx_model,
        args.o,
        save_as_external_data=args.use_external_data_format,
        all_tensors_to_one_file=args.all_tensors_to_one_file,
        location=data_path,
    )


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\types.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ort_flatbuffers_py.fbs as fbs


class FbsTypeInfo:
    "Class to provide conversion between ORT flatbuffers schema values and C++ types"

    tensordatatype_to_string = {  # noqa: RUF012
        fbs.TensorDataType.TensorDataType.FLOAT: "float",
        fbs.TensorDataType.TensorDataType.UINT8: "uint8_t",
        fbs.TensorDataType.TensorDataType.INT8: "int8_t",
        fbs.TensorDataType.TensorDataType.UINT16: "uint16_t",
        fbs.TensorDataType.TensorDataType.INT16: "int16_t",
        fbs.TensorDataType.TensorDataType.INT32: "int32_t",
        fbs.TensorDataType.TensorDataType.INT64: "int64_t",
        fbs.TensorDataType.TensorDataType.STRING: "std::string",
        fbs.TensorDataType.TensorDataType.BOOL: "bool",
        fbs.TensorDataType.TensorDataType.FLOAT16: "MLFloat16",
        fbs.TensorDataType.TensorDataType.DOUBLE: "double",
        fbs.TensorDataType.TensorDataType.UINT32: "uint32_t",
        fbs.TensorDataType.TensorDataType.UINT64: "uint64_t",
        # fbs.TensorDataType.TensorDataType.COMPLEX64: 'complex64 is not supported',
        # fbs.TensorDataType.TensorDataType.COMPLEX128: 'complex128 is not supported',
        fbs.TensorDataType.TensorDataType.BFLOAT16: "BFloat16",
        fbs.TensorDataType.TensorDataType.FLOAT8E4M3FN: "Float8E4M3FN",
        fbs.TensorDataType.TensorDataType.FLOAT8E4M3FNUZ: "Float8E4M3FNUZ",
        fbs.TensorDataType.TensorDataType.FLOAT8E5M2: "Float8E5M2",
        fbs.TensorDataType.TensorDataType.FLOAT8E5M2FNUZ: "Float8E5M2FNUZ",
    }

    @staticmethod
    def typeinfo_to_str(type: fbs.TypeInfo):
        value_type = type.ValueType()
        value = type.Value()
        type_str = "unknown"

        if value_type == fbs.TypeInfoValue.TypeInfoValue.tensor_type:
            tensor_type_and_shape = fbs.TensorTypeAndShape.TensorTypeAndShape()
            tensor_type_and_shape.Init(value.Bytes, value.Pos)
            elem_type = tensor_type_and_shape.ElemType()
            type_str = FbsTypeInfo.tensordatatype_to_string[elem_type]

        elif value_type == fbs.TypeInfoValue.TypeInfoValue.map_type:
            map_type = fbs.MapType.MapType()
            map_type.init(value.Bytes, value.Pos)
            key_type = map_type.KeyType()  # TensorDataType
            key_type_str = FbsTypeInfo.tensordatatype_to_string[key_type]
            value_type = map_type.ValueType()  # TypeInfo
            value_type_str = FbsTypeInfo.typeinfo_to_str(value_type)
            type_str = f"std::map<{key_type_str},{value_type_str}>"

        elif value_type == fbs.TypeInfoValue.TypeInfoValue.sequence_type:
            sequence_type = fbs.SequenceType.SequenceType()
            sequence_type.Init(value.Bytes, value.Pos)
            elem_type = sequence_type.ElemType()  # TypeInfo
            elem_type_str = FbsTypeInfo.typeinfo_to_str(elem_type)
            # TODO: Decide if we need to wrap the type in a std::vector. Issue is that the element type is internal
            # to the onnxruntime::Tensor class so we're really returning the type inside the Tensor not vector<Tensor>.
            # For now, return the element type (which will be the Tensor element type, or a map<A,B>) as
            # an operator input or output will either be a sequence or a not, so we don't need to disambiguate
            # between the two (i.e. we know if the returned value refers to the contents of a sequence, and can
            # handle whether it's the element type of a Tensor in the sequence, or the map type in a sequence of maps
            # due to this).
            type_str = elem_type_str
        else:
            raise ValueError(f"Unknown or missing value type of {value_type}")

        return type_str


def get_typeinfo(name: str, value_name_to_typeinfo: dict) -> fbs.TypeInfo:
    "Lookup a name in a dictionary mapping value name to TypeInfo."
    if name not in value_name_to_typeinfo:
        raise RuntimeError("Missing TypeInfo entry for " + name)

    return value_name_to_typeinfo[name]  # TypeInfo object


def value_name_to_typestr(name: str, value_name_to_typeinfo: dict):
    "Lookup TypeInfo for value name and convert to a string representing the C++ type."
    type = get_typeinfo(name, value_name_to_typeinfo)
    type_str = FbsTypeInfo.typeinfo_to_str(type)
    return type_str

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\_musllinux.py ===
"""PEP 656 support.

This module implements logic to detect if the currently running Python is
linked against musl, and what musl version is used.
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys
from typing import Iterator, NamedTuple, Sequence

from ._elffile import ELFFile


class _MuslVersion(NamedTuple):
    major: int
    minor: int


def _parse_musl_version(output: str) -> _MuslVersion | None:
    lines = [n for n in (n.strip() for n in output.splitlines()) if n]
    if len(lines) < 2 or lines[0][:4] != "musl":
        return None
    m = re.match(r"Version (\d+)\.(\d+)", lines[1])
    if not m:
        return None
    return _MuslVersion(major=int(m.group(1)), minor=int(m.group(2)))


@functools.lru_cache
def _get_musl_version(executable: str) -> _MuslVersion | None:
    """Detect currently-running musl runtime version.

    This is done by checking the specified executable's dynamic linking
    information, and invoking the loader to parse its output for a version
    string. If the loader is musl, the output would be something like::

        musl libc (x86_64)
        Version 1.2.2
        Dynamic Program Loader
    """
    try:
        with open(executable, "rb") as f:
            ld = ELFFile(f).interpreter
    except (OSError, TypeError, ValueError):
        return None
    if ld is None or "musl" not in ld:
        return None
    proc = subprocess.run([ld], stderr=subprocess.PIPE, text=True)
    return _parse_musl_version(proc.stderr)


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate musllinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be musllinux-compatible.

    :returns: An iterator of compatible musllinux tags.
    """
    sys_musl = _get_musl_version(sys.executable)
    if sys_musl is None:  # Python not dynamically linked against musl.
        return
    for arch in archs:
        for minor in range(sys_musl.minor, -1, -1):
            yield f"musllinux_{sys_musl.major}_{minor}_{arch}"


if __name__ == "__main__":  # pragma: no cover
    import sysconfig

    plat = sysconfig.get_platform()
    assert plat.startswith("linux-"), "not linux"

    print("plat:", plat)
    print("musl:", _get_musl_version(sys.executable))
    print("tags:", end=" ")
    for t in platform_tags(re.sub(r"[.-]", "_", plat.split("-", 1)[-1])):
        print(t, end="\n      ")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\api.py ===
from pandas.core.dtypes.common import (
    is_any_real_numeric_dtype,
    is_array_like,
    is_bool,
    is_bool_dtype,
    is_categorical_dtype,
    is_complex,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_dict_like,
    is_dtype_equal,
    is_extension_array_dtype,
    is_file_like,
    is_float,
    is_float_dtype,
    is_hashable,
    is_int64_dtype,
    is_integer,
    is_integer_dtype,
    is_interval,
    is_interval_dtype,
    is_iterator,
    is_list_like,
    is_named_tuple,
    is_number,
    is_numeric_dtype,
    is_object_dtype,
    is_period_dtype,
    is_re,
    is_re_compilable,
    is_scalar,
    is_signed_integer_dtype,
    is_sparse,
    is_string_dtype,
    is_timedelta64_dtype,
    is_timedelta64_ns_dtype,
    is_unsigned_integer_dtype,
    pandas_dtype,
)

__all__ = [
    "is_any_real_numeric_dtype",
    "is_array_like",
    "is_bool",
    "is_bool_dtype",
    "is_categorical_dtype",
    "is_complex",
    "is_complex_dtype",
    "is_datetime64_any_dtype",
    "is_datetime64_dtype",
    "is_datetime64_ns_dtype",
    "is_datetime64tz_dtype",
    "is_dict_like",
    "is_dtype_equal",
    "is_extension_array_dtype",
    "is_file_like",
    "is_float",
    "is_float_dtype",
    "is_hashable",
    "is_int64_dtype",
    "is_integer",
    "is_integer_dtype",
    "is_interval",
    "is_interval_dtype",
    "is_iterator",
    "is_list_like",
    "is_named_tuple",
    "is_number",
    "is_numeric_dtype",
    "is_object_dtype",
    "is_period_dtype",
    "is_re",
    "is_re_compilable",
    "is_scalar",
    "is_signed_integer_dtype",
    "is_sparse",
    "is_string_dtype",
    "is_timedelta64_dtype",
    "is_timedelta64_ns_dtype",
    "is_unsigned_integer_dtype",
    "pandas_dtype",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\internals\__init__.py ===
from pandas.core.internals.api import make_block  # 2023-09-18 pyarrow uses this
from pandas.core.internals.array_manager import (
    ArrayManager,
    SingleArrayManager,
)
from pandas.core.internals.base import (
    DataManager,
    SingleDataManager,
)
from pandas.core.internals.concat import concatenate_managers
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
)

__all__ = [
    "Block",  # pylint: disable=undefined-all-variable
    "DatetimeTZBlock",  # pylint: disable=undefined-all-variable
    "ExtensionBlock",  # pylint: disable=undefined-all-variable
    "make_block",
    "DataManager",
    "ArrayManager",
    "BlockManager",
    "SingleDataManager",
    "SingleBlockManager",
    "SingleArrayManager",
    "concatenate_managers",
]


def __getattr__(name: str):
    # GH#55139
    import warnings

    if name == "create_block_manager_from_blocks":
        # GH#33892
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            DeprecationWarning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )
        from pandas.core.internals.managers import create_block_manager_from_blocks

        return create_block_manager_from_blocks

    if name in [
        "NumericBlock",
        "ObjectBlock",
        "Block",
        "ExtensionBlock",
        "DatetimeTZBlock",
    ]:
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            DeprecationWarning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )
        if name == "NumericBlock":
            from pandas.core.internals.blocks import NumericBlock

            return NumericBlock
        elif name == "DatetimeTZBlock":
            from pandas.core.internals.blocks import DatetimeTZBlock

            return DatetimeTZBlock
        elif name == "ExtensionBlock":
            from pandas.core.internals.blocks import ExtensionBlock

            return ExtensionBlock
        elif name == "Block":
            from pandas.core.internals.blocks import Block

            return Block
        else:
            from pandas.core.internals.blocks import ObjectBlock

            return ObjectBlock

    raise AttributeError(f"module 'pandas.core.internals' has no attribute '{name}'")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\reshape\util.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.dtypes.common import is_list_like

if TYPE_CHECKING:
    from pandas._typing import NumpyIndexT


def cartesian_product(X) -> list[np.ndarray]:
    """
    Numpy version of itertools.product.
    Sometimes faster (for large inputs)...

    Parameters
    ----------
    X : list-like of list-likes

    Returns
    -------
    product : list of ndarrays

    Examples
    --------
    >>> cartesian_product([list('ABC'), [1, 2]])
    [array(['A', 'A', 'B', 'B', 'C', 'C'], dtype='<U1'), array([1, 2, 1, 2, 1, 2])]

    See Also
    --------
    itertools.product : Cartesian product of input iterables.  Equivalent to
        nested for-loops.
    """
    msg = "Input must be a list-like of list-likes"
    if not is_list_like(X):
        raise TypeError(msg)
    for x in X:
        if not is_list_like(x):
            raise TypeError(msg)

    if len(X) == 0:
        return []

    lenX = np.fromiter((len(x) for x in X), dtype=np.intp)
    cumprodX = np.cumprod(lenX)

    if np.any(cumprodX < 0):
        raise ValueError("Product space too large to allocate arrays!")

    a = np.roll(cumprodX, 1)
    a[0] = 1

    if cumprodX[-1] != 0:
        b = cumprodX[-1] / cumprodX
    else:
        # if any factor is empty, the cartesian product is empty
        b = np.zeros_like(cumprodX)

    # error: Argument of type "int_" cannot be assigned to parameter "num" of
    # type "int" in function "tile_compat"
    return [
        tile_compat(
            np.repeat(x, b[i]),
            np.prod(a[i]),
        )
        for i, x in enumerate(X)
    ]


def tile_compat(arr: NumpyIndexT, num: int) -> NumpyIndexT:
    """
    Index compat for np.tile.

    Notes
    -----
    Does not support multi-dimensional `num`.
    """
    if isinstance(arr, np.ndarray):
        return np.tile(arr, num)

    # Otherwise we have an Index
    taker = np.tile(np.arange(len(arr)), num)
    return arr.take(taker)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\metadata\importlib\_compat.py ===
import importlib.metadata
import os
from typing import Any, Optional, Protocol, Tuple, cast

from pip._vendor.packaging.utils import NormalizedName, canonicalize_name


class BadMetadata(ValueError):
    def __init__(self, dist: importlib.metadata.Distribution, *, reason: str) -> None:
        self.dist = dist
        self.reason = reason

    def __str__(self) -> str:
        return f"Bad metadata in {self.dist} ({self.reason})"


class BasePath(Protocol):
    """A protocol that various path objects conform.

    This exists because importlib.metadata uses both ``pathlib.Path`` and
    ``zipfile.Path``, and we need a common base for type hints (Union does not
    work well since ``zipfile.Path`` is too new for our linter setup).

    This does not mean to be exhaustive, but only contains things that present
    in both classes *that we need*.
    """

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def parent(self) -> "BasePath":
        raise NotImplementedError()


def get_info_location(d: importlib.metadata.Distribution) -> Optional[BasePath]:
    """Find the path to the distribution's metadata directory.

    HACK: This relies on importlib.metadata's private ``_path`` attribute. Not
    all distributions exist on disk, so importlib.metadata is correct to not
    expose the attribute as public. But pip's code base is old and not as clean,
    so we do this to avoid having to rewrite too many things. Hopefully we can
    eliminate this some day.
    """
    return getattr(d, "_path", None)


def parse_name_and_version_from_info_directory(
    dist: importlib.metadata.Distribution,
) -> Tuple[Optional[str], Optional[str]]:
    """Get a name and version from the metadata directory name.

    This is much faster than reading distribution metadata.
    """
    info_location = get_info_location(dist)
    if info_location is None:
        return None, None

    stem, suffix = os.path.splitext(info_location.name)
    if suffix == ".dist-info":
        name, sep, version = stem.partition("-")
        if sep:
            return name, version

    if suffix == ".egg-info":
        name = stem.split("-", 1)[0]
        return name, None

    return None, None


def get_dist_canonical_name(dist: importlib.metadata.Distribution) -> NormalizedName:
    """Get the distribution's normalized name.

    The ``name`` attribute is only available in Python 3.10 or later. We are
    targeting exactly that, but Mypy does not know this.
    """
    if name := parse_name_and_version_from_info_directory(dist)[0]:
        return canonicalize_name(name)

    name = cast(Any, dist).name
    if not isinstance(name, str):
        raise BadMetadata(dist, reason="invalid metadata entry 'name'")
    return canonicalize_name(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\_musllinux.py ===
"""PEP 656 support.

This module implements logic to detect if the currently running Python is
linked against musl, and what musl version is used.
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys
from typing import Iterator, NamedTuple, Sequence

from ._elffile import ELFFile


class _MuslVersion(NamedTuple):
    major: int
    minor: int


def _parse_musl_version(output: str) -> _MuslVersion | None:
    lines = [n for n in (n.strip() for n in output.splitlines()) if n]
    if len(lines) < 2 or lines[0][:4] != "musl":
        return None
    m = re.match(r"Version (\d+)\.(\d+)", lines[1])
    if not m:
        return None
    return _MuslVersion(major=int(m.group(1)), minor=int(m.group(2)))


@functools.lru_cache
def _get_musl_version(executable: str) -> _MuslVersion | None:
    """Detect currently-running musl runtime version.

    This is done by checking the specified executable's dynamic linking
    information, and invoking the loader to parse its output for a version
    string. If the loader is musl, the output would be something like::

        musl libc (x86_64)
        Version 1.2.2
        Dynamic Program Loader
    """
    try:
        with open(executable, "rb") as f:
            ld = ELFFile(f).interpreter
    except (OSError, TypeError, ValueError):
        return None
    if ld is None or "musl" not in ld:
        return None
    proc = subprocess.run([ld], stderr=subprocess.PIPE, text=True)
    return _parse_musl_version(proc.stderr)


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate musllinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be musllinux-compatible.

    :returns: An iterator of compatible musllinux tags.
    """
    sys_musl = _get_musl_version(sys.executable)
    if sys_musl is None:  # Python not dynamically linked against musl.
        return
    for arch in archs:
        for minor in range(sys_musl.minor, -1, -1):
            yield f"musllinux_{sys_musl.major}_{minor}_{arch}"


if __name__ == "__main__":  # pragma: no cover
    import sysconfig

    plat = sysconfig.get_platform()
    assert plat.startswith("linux-"), "not linux"

    print("plat:", plat)
    print("musl:", _get_musl_version(sys.executable))
    print("tags:", end=" ")
    for t in platform_tags(re.sub(r"[.-]", "_", plat.split("-", 1)[-1])):
        print(t, end="\n      ")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\rationaltools.py ===
"""Tools for manipulation of rational expressions. """


from sympy.core import Basic, Add, sympify
from sympy.core.exprtools import gcd_terms
from sympy.utilities import public
from sympy.utilities.iterables import iterable


@public
def together(expr, deep=False, fraction=True):
    """
    Denest and combine rational expressions using symbolic methods.

    This function takes an expression or a container of expressions
    and puts it (them) together by denesting and combining rational
    subexpressions. No heroic measures are taken to minimize degree
    of the resulting numerator and denominator. To obtain completely
    reduced expression use :func:`~.cancel`. However, :func:`~.together`
    can preserve as much as possible of the structure of the input
    expression in the output (no expansion is performed).

    A wide variety of objects can be put together including lists,
    tuples, sets, relational objects, integrals and others. It is
    also possible to transform interior of function applications,
    by setting ``deep`` flag to ``True``.

    By definition, :func:`~.together` is a complement to :func:`~.apart`,
    so ``apart(together(expr))`` should return expr unchanged. Note
    however, that :func:`~.together` uses only symbolic methods, so
    it might be necessary to use :func:`~.cancel` to perform algebraic
    simplification and minimize degree of the numerator and denominator.

    Examples
    ========

    >>> from sympy import together, exp
    >>> from sympy.abc import x, y, z

    >>> together(1/x + 1/y)
    (x + y)/(x*y)
    >>> together(1/x + 1/y + 1/z)
    (x*y + x*z + y*z)/(x*y*z)

    >>> together(1/(x*y) + 1/y**2)
    (x + y)/(x*y**2)

    >>> together(1/(1 + 1/x) + 1/(1 + 1/y))
    (x*(y + 1) + y*(x + 1))/((x + 1)*(y + 1))

    >>> together(exp(1/x + 1/y))
    exp(1/y + 1/x)
    >>> together(exp(1/x + 1/y), deep=True)
    exp((x + y)/(x*y))

    >>> together(1/exp(x) + 1/(x*exp(x)))
    (x + 1)*exp(-x)/x

    >>> together(1/exp(2*x) + 1/(x*exp(3*x)))
    (x*exp(x) + 1)*exp(-3*x)/x

    """
    def _together(expr):
        if isinstance(expr, Basic):
            if expr.is_Atom or (expr.is_Function and not deep):
                return expr
            elif expr.is_Add:
                return gcd_terms(list(map(_together, Add.make_args(expr))), fraction=fraction)
            elif expr.is_Pow:
                base = _together(expr.base)

                if deep:
                    exp = _together(expr.exp)
                else:
                    exp = expr.exp

                return expr.func(base, exp)
            else:
                return expr.func(*[ _together(arg) for arg in expr.args ])
        elif iterable(expr):
            return expr.__class__([ _together(ex) for ex in expr ])

        return expr

    return _together(sympify(expr))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_pivot.py ===
from datetime import (
    date,
    datetime,
    timedelta,
)
from itertools import product
import re

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table


@pytest.fixture(params=[True, False])
def dropna(request):
    return request.param


@pytest.fixture(params=[([0] * 4, [1] * 4), (range(3), range(1, 4))])
def interval_values(request, closed):
    left, right = request.param
    return Categorical(pd.IntervalIndex.from_arrays(left, right, closed))


class TestPivotTable:
    @pytest.fixture
    def data(self):
        return DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "foo",
                    "foo",
                    "foo",
                ],
                "B": [
                    "one",
                    "one",
                    "one",
                    "two",
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "two",
                    "one",
                ],
                "C": [
                    "dull",
                    "dull",
                    "shiny",
                    "dull",
                    "dull",
                    "shiny",
                    "shiny",
                    "dull",
                    "shiny",
                    "shiny",
                    "shiny",
                ],
                "D": np.random.default_rng(2).standard_normal(11),
                "E": np.random.default_rng(2).standard_normal(11),
                "F": np.random.default_rng(2).standard_normal(11),
            }
        )

    def test_pivot_table(self, observed, data):
        index = ["A", "B"]
        columns = "C"
        table = pivot_table(
            data, values="D", index=index, columns=columns, observed=observed
        )

        table2 = data.pivot_table(
            values="D", index=index, columns=columns, observed=observed
        )
        tm.assert_frame_equal(table, table2)

        # this works
        pivot_table(data, values="D", index=index, observed=observed)

        if len(index) > 1:
            assert table.index.names == tuple(index)
        else:
            assert table.index.name == index[0]

        if len(columns) > 1:
            assert table.columns.names == columns
        else:
            assert table.columns.name == columns[0]

        expected = data.groupby(index + [columns])["D"].agg("mean").unstack()
        tm.assert_frame_equal(table, expected)

    def test_pivot_table_categorical_observed_equal(self, observed):
        # issue #24923
        df = DataFrame(
            {"col1": list("abcde"), "col2": list("fghij"), "col3": [1, 2, 3, 4, 5]}
        )

        expected = df.pivot_table(
            index="col1", values="col3", columns="col2", aggfunc="sum", fill_value=0
        )

        expected.index = expected.index.astype("category")
        expected.columns = expected.columns.astype("category")

        df.col1 = df.col1.astype("category")
        df.col2 = df.col2.astype("category")

        result = df.pivot_table(
            index="col1",
            values="col3",
            columns="col2",
            aggfunc="sum",
            fill_value=0,
            observed=observed,
        )

        tm.assert_frame_equal(result, expected)

    def test_pivot_table_nocols(self):
        df = DataFrame(
            {"rows": ["a", "b", "c"], "cols": ["x", "y", "z"], "values": [1, 2, 3]}
        )
        rs = df.pivot_table(columns="cols", aggfunc="sum")
        xp = df.pivot_table(index="cols", aggfunc="sum").T
        tm.assert_frame_equal(rs, xp)

        rs = df.pivot_table(columns="cols", aggfunc={"values": "mean"})
        xp = df.pivot_table(index="cols", aggfunc={"values": "mean"}).T
        tm.assert_frame_equal(rs, xp)

    def test_pivot_table_dropna(self):
        df = DataFrame(
            {
                "amount": {0: 60000, 1: 100000, 2: 50000, 3: 30000},
                "customer": {0: "A", 1: "A", 2: "B", 3: "C"},
                "month": {0: 201307, 1: 201309, 2: 201308, 3: 201310},
                "product": {0: "a", 1: "b", 2: "c", 3: "d"},
                "quantity": {0: 2000000, 1: 500000, 2: 1000000, 3: 1000000},
            }
        )
        pv_col = df.pivot_table(
            "quantity", "month", ["customer", "product"], dropna=False
        )
        pv_ind = df.pivot_table(
            "quantity", ["customer", "product"], "month", dropna=False
        )

        m = MultiIndex.from_tuples(
            [
                ("A", "a"),
                ("A", "b"),
                ("A", "c"),
                ("A", "d"),
                ("B", "a"),
                ("B", "b"),
                ("B", "c"),
                ("B", "d"),
                ("C", "a"),
                ("C", "b"),
                ("C", "c"),
                ("C", "d"),
            ],
            names=["customer", "product"],
        )
        tm.assert_index_equal(pv_col.columns, m)
        tm.assert_index_equal(pv_ind.index, m)

    def test_pivot_table_categorical(self):
        cat1 = Categorical(
            ["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True
        )
        cat2 = Categorical(
            ["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True
        )
        df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = pivot_table(df, values="values", index=["A", "B"], dropna=True)

        exp_index = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
        expected = DataFrame({"values": [1.0, 2.0, 3.0, 4.0]}, index=exp_index)
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_dropna_categoricals(self, dropna):
        # GH 15193
        categories = ["a", "b", "c", "d"]

        df = DataFrame(
            {
                "A": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "B": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "C": range(9),
            }
        )

        df["A"] = df["A"].astype(CategoricalDtype(categories, ordered=False))
        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.pivot_table(index="B", columns="A", values="C", dropna=dropna)
        expected_columns = Series(["a", "b", "c"], name="A")
        expected_columns = expected_columns.astype(
            CategoricalDtype(categories, ordered=False)
        )
        expected_index = Series([1, 2, 3], name="B")
        expected = DataFrame(
            [[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]],
            index=expected_index,
            columns=expected_columns,
        )
        if not dropna:
            # add back the non observed to compare
            expected = expected.reindex(columns=Categorical(categories)).astype("float")

        tm.assert_frame_equal(result, expected)

    def test_pivot_with_non_observable_dropna(self, dropna):
        # gh-21133
        df = DataFrame(
            {
                "A": Categorical(
                    [np.nan, "low", "high", "low", "high"],
                    categories=["low", "high"],
                    ordered=True,
                ),
                "B": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        )

        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.pivot_table(index="A", values="B", dropna=dropna)
        if dropna:
            values = [2.0, 3.0]
            codes = [0, 1]
        else:
            # GH: 10772
            values = [2.0, 3.0, 0.0]
            codes = [0, 1, -1]
        expected = DataFrame(
            {"B": values},
            index=Index(
                Categorical.from_codes(
                    codes, categories=["low", "high"], ordered=dropna
                ),
                name="A",
            ),
        )

        tm.assert_frame_equal(result, expected)

    def test_pivot_with_non_observable_dropna_multi_cat(self, dropna):
        # gh-21378
        df = DataFrame(
            {
                "A": Categorical(
                    ["left", "low", "high", "low", "high"],
                    categories=["low", "high", "left"],
                    ordered=True,
                ),
                "B": range(5),
            }
        )

        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.pivot_table(index="A", values="B", dropna=dropna)
        expected = DataFrame(
            {"B": [2.0, 3.0, 0.0]},
            index=Index(
                Categorical.from_codes(
                    [0, 1, 2], categories=["low", "high", "left"], ordered=True
                ),
                name="A",
            ),
        )
        if not dropna:
            expected["B"] = expected["B"].astype(float)

        tm.assert_frame_equal(result, expected)

    def test_pivot_with_interval_index(self, interval_values, dropna):
        # GH 25814
        df = DataFrame({"A": interval_values, "B": 1})

        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.pivot_table(index="A", values="B", dropna=dropna)
        expected = DataFrame(
            {"B": 1.0}, index=Index(interval_values.unique(), name="A")
        )
        if not dropna:
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_interval_index_margins(self):
        # GH 25815
        ordered_cat = pd.IntervalIndex.from_arrays([0, 0, 1, 1], [1, 1, 2, 2])
        df = DataFrame(
            {
                "A": np.arange(4, 0, -1, dtype=np.intp),
                "B": ["a", "b", "a", "b"],
                "C": Categorical(ordered_cat, ordered=True).sort_values(
                    ascending=False
                ),
            }
        )

        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            pivot_tab = pivot_table(
                df, index="C", columns="B", values="A", aggfunc="sum", margins=True
            )

        result = pivot_tab["All"]
        expected = Series(
            [3, 7, 10],
            index=Index([pd.Interval(0, 1), pd.Interval(1, 2), "All"], name="C"),
            name="All",
            dtype=np.intp,
        )
        tm.assert_series_equal(result, expected)

    def test_pass_array(self, data):
        result = data.pivot_table("D", index=data.A, columns=data.C)
        expected = data.pivot_table("D", index="A", columns="C")
        tm.assert_frame_equal(result, expected)

    def test_pass_function(self, data):
        result = data.pivot_table("D", index=lambda x: x // 5, columns=data.C)
        expected = data.pivot_table("D", index=data.index // 5, columns="C")
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_multiple(self, data):
        index = ["A", "B"]
        columns = "C"
        table = pivot_table(data, index=index, columns=columns)
        expected = data.groupby(index + [columns]).agg("mean").unstack()
        tm.assert_frame_equal(table, expected)

    def test_pivot_dtypes(self):
        # can convert dtypes
        f = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1, 2, 3, 4],
                "i": ["a", "b", "a", "b"],
            }
        )
        assert f.dtypes["v"] == "int64"

        z = pivot_table(
            f, values="v", index=["a"], columns=["i"], fill_value=0, aggfunc="sum"
        )
        result = z.dtypes
        expected = Series([np.dtype("int64")] * 2, index=Index(list("ab"), name="i"))
        tm.assert_series_equal(result, expected)

        # cannot convert dtypes
        f = DataFrame(
            {
                "a": ["cat", "bat", "cat", "bat"],
                "v": [1.5, 2.5, 3.5, 4.5],
                "i": ["a", "b", "a", "b"],
            }
        )
        assert f.dtypes["v"] == "float64"

        z = pivot_table(
            f, values="v", index=["a"], columns=["i"], fill_value=0, aggfunc="mean"
        )
        result = z.dtypes
        expected = Series([np.dtype("float64")] * 2, index=Index(list("ab"), name="i"))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "columns,values",
        [
            ("bool1", ["float1", "float2"]),
            ("bool1", ["float1", "float2", "bool1"]),
            ("bool2", ["float1", "float2", "bool1"]),
        ],
    )
    def test_pivot_preserve_dtypes(self, columns, values):
        # GH 7142 regression test
        v = np.arange(5, dtype=np.float64)
        df = DataFrame(
            {"float1": v, "float2": v + 2.0, "bool1": v <= 2, "bool2": v <= 3}
        )

        df_res = df.reset_index().pivot_table(
            index="index", columns=columns, values=values
        )

        result = dict(df_res.dtypes)
        expected = {col: np.dtype("float64") for col in df_res}
        assert result == expected

    def test_pivot_no_values(self):
        # GH 14380
        idx = pd.DatetimeIndex(
            ["2011-01-01", "2011-02-01", "2011-01-02", "2011-01-01", "2011-01-02"]
        )
        df = DataFrame({"A": [1, 2, 3, 4, 5]}, index=idx)
        res = df.pivot_table(index=df.index.month, columns=df.index.day)

        exp_columns = MultiIndex.from_tuples([("A", 1), ("A", 2)])
        exp_columns = exp_columns.set_levels(
            exp_columns.levels[1].astype(np.int32), level=1
        )
        exp = DataFrame(
            [[2.5, 4.0], [2.0, np.nan]],
            index=Index([1, 2], dtype=np.int32),
            columns=exp_columns,
        )
        tm.assert_frame_equal(res, exp)

        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "dt": date_range("2011-01-01", freq="D", periods=5),
            },
            index=idx,
        )
        res = df.pivot_table(index=df.index.month, columns=Grouper(key="dt", freq="ME"))
        exp_columns = MultiIndex.from_arrays(
            [["A"], pd.DatetimeIndex(["2011-01-31"], dtype="M8[ns]")],
            names=[None, "dt"],
        )
        exp = DataFrame(
            [3.25, 2.0], index=Index([1, 2], dtype=np.int32), columns=exp_columns
        )
        tm.assert_frame_equal(res, exp)

        res = df.pivot_table(
            index=Grouper(freq="YE"), columns=Grouper(key="dt", freq="ME")
        )
        exp = DataFrame(
            [3.0],
            index=pd.DatetimeIndex(["2011-12-31"], freq="YE"),
            columns=exp_columns,
        )
        tm.assert_frame_equal(res, exp)

    def test_pivot_multi_values(self, data):
        result = pivot_table(
            data, values=["D", "E"], index="A", columns=["B", "C"], fill_value=0
        )
        expected = pivot_table(
            data.drop(["F"], axis=1), index="A", columns=["B", "C"], fill_value=0
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_multi_functions(self, data):
        f = lambda func: pivot_table(
            data, values=["D", "E"], index=["A", "B"], columns="C", aggfunc=func
        )
        result = f(["mean", "std"])
        means = f("mean")
        stds = f("std")
        expected = concat([means, stds], keys=["mean", "std"], axis=1)
        tm.assert_frame_equal(result, expected)

        # margins not supported??
        f = lambda func: pivot_table(
            data,
            values=["D", "E"],
            index=["A", "B"],
            columns="C",
            aggfunc=func,
            margins=True,
        )
        result = f(["mean", "std"])
        means = f("mean")
        stds = f("std")
        expected = concat([means, stds], keys=["mean", "std"], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_index_with_nan(self, method):
        # GH 3588
        nan = np.nan
        df = DataFrame(
            {
                "a": ["R1", "R2", nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, 17, 20],
            }
        )
        if method:
            result = df.pivot(index="a", columns="b", values="c")
        else:
            result = pd.pivot(df, index="a", columns="b", values="c")
        expected = DataFrame(
            [
                [nan, nan, 17, nan],
                [10, nan, nan, nan],
                [nan, 15, nan, nan],
                [nan, nan, nan, 20],
            ],
            index=Index([nan, "R1", "R2", "R4"], name="a"),
            columns=Index(["C1", "C2", "C3", "C4"], name="b"),
        )
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df.pivot(index="b", columns="a", values="c"), expected.T)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_index_with_nan_dates(self, method):
        # GH9491
        df = DataFrame(
            {
                "a": date_range("2014-02-01", periods=6, freq="D"),
                "c": 100 + np.arange(6),
            }
        )
        df["b"] = df["a"] - pd.Timestamp("2014-02-02")
        df.loc[1, "a"] = df.loc[3, "a"] = np.nan
        df.loc[1, "b"] = df.loc[4, "b"] = np.nan

        if method:
            pv = df.pivot(index="a", columns="b", values="c")
        else:
            pv = pd.pivot(df, index="a", columns="b", values="c")
        assert pv.notna().values.sum() == len(df)

        for _, row in df.iterrows():
            assert pv.loc[row["a"], row["b"]] == row["c"]

        if method:
            result = df.pivot(index="b", columns="a", values="c")
        else:
            result = pd.pivot(df, index="b", columns="a", values="c")
        tm.assert_frame_equal(result, pv.T)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_tz(self, method, unit):
        # GH 5878
        df = DataFrame(
            {
                "dt1": pd.DatetimeIndex(
                    [
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                        datetime(2013, 1, 1, 9, 0),
                        datetime(2013, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, US/Pacific]",
                ),
                "dt2": pd.DatetimeIndex(
                    [
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 1, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                        datetime(2014, 1, 2, 9, 0),
                    ],
                    dtype=f"M8[{unit}, Asia/Tokyo]",
                ),
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )

        exp_col1 = Index(["data1", "data1", "data2", "data2"])
        exp_col2 = pd.DatetimeIndex(
            ["2014/01/01 09:00", "2014/01/02 09:00"] * 2,
            name="dt2",
            dtype=f"M8[{unit}, Asia/Tokyo]",
        )
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        exp_idx = pd.DatetimeIndex(
            ["2013/01/01 09:00", "2013/01/02 09:00"],
            name="dt1",
            dtype=f"M8[{unit}, US/Pacific]",
        )
        expected = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=exp_idx,
            columns=exp_col,
        )

        if method:
            pv = df.pivot(index="dt1", columns="dt2")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2")
        tm.assert_frame_equal(pv, expected)

        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=exp_idx,
            columns=exp_col2[:2],
        )

        if method:
            pv = df.pivot(index="dt1", columns="dt2", values="data1")
        else:
            pv = pd.pivot(df, index="dt1", columns="dt2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_tz_in_values(self):
        # GH 14948
        df = DataFrame(
            [
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 13:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-12 14:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific"),
                },
                {
                    "uid": "aa",
                    "ts": pd.Timestamp("2016-08-25 13:00:00-0700", tz="US/Pacific"),
                },
            ]
        )

        df = df.set_index("ts").reset_index()
        mins = df.ts.map(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))

        result = pivot_table(
            df.set_index("ts").reset_index(),
            values="ts",
            index=["uid"],
            columns=[mins],
            aggfunc="min",
        )
        expected = DataFrame(
            [
                [
                    pd.Timestamp("2016-08-12 08:00:00-0700", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 11:00:00-0700", tz="US/Pacific"),
                ]
            ],
            index=Index(["aa"], name="uid"),
            columns=pd.DatetimeIndex(
                [
                    pd.Timestamp("2016-08-12 00:00:00", tz="US/Pacific"),
                    pd.Timestamp("2016-08-25 00:00:00", tz="US/Pacific"),
                ],
                name="ts",
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_periods(self, method):
        df = DataFrame(
            {
                "p1": [
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                    pd.Period("2013-01-01", "D"),
                    pd.Period("2013-01-02", "D"),
                ],
                "p2": [
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-01", "M"),
                    pd.Period("2013-02", "M"),
                    pd.Period("2013-02", "M"),
                ],
                "data1": np.arange(4, dtype="int64"),
                "data2": np.arange(4, dtype="int64"),
            }
        )

        exp_col1 = Index(["data1", "data1", "data2", "data2"])
        exp_col2 = pd.PeriodIndex(["2013-01", "2013-02"] * 2, name="p2", freq="M")
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        expected = DataFrame(
            [[0, 2, 0, 2], [1, 3, 1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=exp_col,
        )
        if method:
            pv = df.pivot(index="p1", columns="p2")
        else:
            pv = pd.pivot(df, index="p1", columns="p2")
        tm.assert_frame_equal(pv, expected)

        expected = DataFrame(
            [[0, 2], [1, 3]],
            index=pd.PeriodIndex(["2013-01-01", "2013-01-02"], name="p1", freq="D"),
            columns=pd.PeriodIndex(["2013-01", "2013-02"], name="p2", freq="M"),
        )
        if method:
            pv = df.pivot(index="p1", columns="p2", values="data1")
        else:
            pv = pd.pivot(df, index="p1", columns="p2", values="data1")
        tm.assert_frame_equal(pv, expected)

    def test_pivot_periods_with_margins(self):
        # GH 28323
        df = DataFrame(
            {
                "a": [1, 1, 2, 2],
                "b": [
                    pd.Period("2019Q1"),
                    pd.Period("2019Q2"),
                    pd.Period("2019Q1"),
                    pd.Period("2019Q2"),
                ],
                "x": 1.0,
            }
        )

        expected = DataFrame(
            data=1.0,
            index=Index([1, 2, "All"], name="a"),
            columns=Index([pd.Period("2019Q1"), pd.Period("2019Q2"), "All"], name="b"),
        )

        result = df.pivot_table(index="a", columns="b", values="x", margins=True)
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "values",
        [
            ["baz", "zoo"],
            np.array(["baz", "zoo"]),
            Series(["baz", "zoo"]),
            Index(["baz", "zoo"]),
        ],
    )
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_list_like_values(self, values, method):
        # issue #17160
        df = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )

        if method:
            result = df.pivot(index="foo", columns="bar", values=values)
        else:
            result = pd.pivot(df, index="foo", columns="bar", values=values)

        data = [[1, 2, 3, "x", "y", "z"], [4, 5, 6, "q", "w", "t"]]
        index = Index(data=["one", "two"], name="foo")
        columns = MultiIndex(
            levels=[["baz", "zoo"], ["A", "B", "C"]],
            codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
            names=[None, "bar"],
        )
        expected = DataFrame(data=data, index=index, columns=columns)
        expected["baz"] = expected["baz"].astype(object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            ["bar", "baz"],
            np.array(["bar", "baz"]),
            Series(["bar", "baz"]),
            Index(["bar", "baz"]),
        ],
    )
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_list_like_values_nans(self, values, method):
        # issue #17160
        df = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )

        if method:
            result = df.pivot(index="zoo", columns="foo", values=values)
        else:
            result = pd.pivot(df, index="zoo", columns="foo", values=values)

        data = [
            [np.nan, "A", np.nan, 4],
            [np.nan, "C", np.nan, 6],
            [np.nan, "B", np.nan, 5],
            ["A", np.nan, 1, np.nan],
            ["B", np.nan, 2, np.nan],
            ["C", np.nan, 3, np.nan],
        ]
        index = Index(data=["q", "t", "w", "x", "y", "z"], name="zoo")
        columns = MultiIndex(
            levels=[["bar", "baz"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=[None, "foo"],
        )
        expected = DataFrame(data=data, index=index, columns=columns)
        expected["baz"] = expected["baz"].astype(object)
        tm.assert_frame_equal(result, expected)

    def test_pivot_columns_none_raise_error(self):
        # GH 30924
        df = DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3], "col3": [1, 2, 3]})
        msg = r"pivot\(\) missing 1 required keyword-only argument: 'columns'"
        with pytest.raises(TypeError, match=msg):
            df.pivot(index="col1", values="col3")  # pylint: disable=missing-kwoa

    @pytest.mark.xfail(
        reason="MultiIndexed unstack with tuple names fails with KeyError GH#19966"
    )
    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_multiindex(self, method):
        # issue #17160
        index = Index(data=[0, 1, 2, 3, 4, 5])
        data = [
            ["one", "A", 1, "x"],
            ["one", "B", 2, "y"],
            ["one", "C", 3, "z"],
            ["two", "A", 4, "q"],
            ["two", "B", 5, "w"],
            ["two", "C", 6, "t"],
        ]
        columns = MultiIndex(
            levels=[["bar", "baz"], ["first", "second"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        )
        df = DataFrame(data=data, index=index, columns=columns, dtype="object")
        if method:
            result = df.pivot(
                index=("bar", "first"),
                columns=("bar", "second"),
                values=("baz", "first"),
            )
        else:
            result = pd.pivot(
                df,
                index=("bar", "first"),
                columns=("bar", "second"),
                values=("baz", "first"),
            )

        data = {
            "A": Series([1, 4], index=["one", "two"]),
            "B": Series([2, 5], index=["one", "two"]),
            "C": Series([3, 6], index=["one", "two"]),
        }
        expected = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", [True, False])
    def test_pivot_with_tuple_of_values(self, method):
        # issue #17160
        df = DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        with pytest.raises(KeyError, match=r"^\('bar', 'baz'\)$"):
            # tuple is seen as a single column name
            if method:
                df.pivot(index="zoo", columns="foo", values=("bar", "baz"))
            else:
                pd.pivot(df, index="zoo", columns="foo", values=("bar", "baz"))

    def _check_output(
        self,
        result,
        values_col,
        data,
        index=["A", "B"],
        columns=["C"],
        margins_col="All",
    ):
        col_margins = result.loc[result.index[:-1], margins_col]
        expected_col_margins = data.groupby(index)[values_col].mean()
        tm.assert_series_equal(col_margins, expected_col_margins, check_names=False)
        assert col_margins.name == margins_col

        result = result.sort_index()
        index_margins = result.loc[(margins_col, "")].iloc[:-1]

        expected_ix_margins = data.groupby(columns)[values_col].mean()
        tm.assert_series_equal(index_margins, expected_ix_margins, check_names=False)
        assert index_margins.name == (margins_col, "")

        grand_total_margins = result.loc[(margins_col, ""), margins_col]
        expected_total_margins = data[values_col].mean()
        assert grand_total_margins == expected_total_margins

    def test_margins(self, data):
        # column specified
        result = data.pivot_table(
            values="D", index=["A", "B"], columns="C", margins=True, aggfunc="mean"
        )
        self._check_output(result, "D", data)

        # Set a different margins_name (not 'All')
        result = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="mean",
            margins_name="Totals",
        )
        self._check_output(result, "D", data, margins_col="Totals")

        # no column specified
        table = data.pivot_table(
            index=["A", "B"], columns="C", margins=True, aggfunc="mean"
        )
        for value_col in table.columns.levels[0]:
            self._check_output(table[value_col], value_col, data)

    def test_no_col(self, data, using_infer_string):
        # no col

        # to help with a buglet
        data.columns = [k * 2 for k in data.columns]
        msg = re.escape("agg function failed [how->mean,dtype->")
        if using_infer_string:
            msg = "dtype 'str' does not support operation 'mean'"
        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=["AA", "BB"], margins=True, aggfunc="mean")
        table = data.drop(columns="CC").pivot_table(
            index=["AA", "BB"], margins=True, aggfunc="mean"
        )
        for value_col in table.columns:
            totals = table.loc[("All", ""), value_col]
            assert totals == data[value_col].mean()

        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=["AA", "BB"], margins=True, aggfunc="mean")
        table = data.drop(columns="CC").pivot_table(
            index=["AA", "BB"], margins=True, aggfunc="mean"
        )
        for item in ["DD", "EE", "FF"]:
            totals = table.loc[("All", ""), item]
            assert totals == data[item].mean()

    @pytest.mark.parametrize(
        "columns, aggfunc, values, expected_columns",
        [
            (
                "A",
                "mean",
                [[5.5, 5.5, 2.2, 2.2], [8.0, 8.0, 4.4, 4.4]],
                Index(["bar", "All", "foo", "All"], name="A"),
            ),
            (
                ["A", "B"],
                "sum",
                [
                    [9, 13, 22, 5, 6, 11],
                    [14, 18, 32, 11, 11, 22],
                ],
                MultiIndex.from_tuples(
                    [
                        ("bar", "one"),
                        ("bar", "two"),
                        ("bar", "All"),
                        ("foo", "one"),
                        ("foo", "two"),
                        ("foo", "All"),
                    ],
                    names=["A", "B"],
                ),
            ),
        ],
    )
    def test_margin_with_only_columns_defined(
        self, columns, aggfunc, values, expected_columns, using_infer_string
    ):
        # GH 31016
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        if aggfunc != "sum":
            msg = re.escape("agg function failed [how->mean,dtype->")
            if using_infer_string:
                msg = "dtype 'str' does not support operation 'mean'"
            with pytest.raises(TypeError, match=msg):
                df.pivot_table(columns=columns, margins=True, aggfunc=aggfunc)
        if "B" not in columns:
            df = df.drop(columns="B")
        result = df.drop(columns="C").pivot_table(
            columns=columns, margins=True, aggfunc=aggfunc
        )
        expected = DataFrame(values, index=Index(["D", "E"]), columns=expected_columns)

        tm.assert_frame_equal(result, expected)

    def test_margins_dtype(self, data):
        # GH 17013

        df = data.copy()
        df[["D", "E", "F"]] = np.arange(len(df) * 3).reshape(len(df), 3).astype("i8")

        mi_val = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        mi = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        expected = DataFrame(
            {"dull": [12, 21, 3, 9, 45], "shiny": [33, 0, 36, 51, 120]}, index=mi
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]

        result = df.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc="sum",
            fill_value=0,
        )

        tm.assert_frame_equal(expected, result)

    def test_margins_dtype_len(self, data):
        mi_val = list(product(["bar", "foo"], ["one", "two"])) + [("All", "")]
        mi = MultiIndex.from_tuples(mi_val, names=("A", "B"))
        expected = DataFrame(
            {"dull": [1, 1, 2, 1, 5], "shiny": [2, 0, 2, 2, 6]}, index=mi
        ).rename_axis("C", axis=1)
        expected["All"] = expected["dull"] + expected["shiny"]

        result = data.pivot_table(
            values="D",
            index=["A", "B"],
            columns="C",
            margins=True,
            aggfunc=len,
            fill_value=0,
        )

        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("cols", [(1, 2), ("a", "b"), (1, "b"), ("a", 1)])
    def test_pivot_table_multiindex_only(self, cols):
        # GH 17038
        df2 = DataFrame({cols[0]: [1, 2, 3], cols[1]: [1, 2, 3], "v": [4, 5, 6]})

        result = df2.pivot_table(values="v", columns=cols)
        expected = DataFrame(
            [[4.0, 5.0, 6.0]],
            columns=MultiIndex.from_tuples([(1, 1), (2, 2), (3, 3)], names=cols),
            index=Index(["v"], dtype="str" if cols == ("a", "b") else "object"),
        )

        tm.assert_frame_equal(result, expected)

    def test_pivot_table_retains_tz(self):
        dti = date_range("2016-01-01", periods=3, tz="Europe/Amsterdam")
        df = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(3),
                "B": np.random.default_rng(2).standard_normal(3),
                "C": dti,
            }
        )
        result = df.pivot_table(index=["B", "C"], dropna=False)

        # check tz retention
        assert result.index.levels[1].equals(dti)

    def test_pivot_integer_columns(self):
        # caused by upstream bug in unstack

        d = date.min
        data = list(
            product(
                ["foo", "bar"],
                ["A", "B", "C"],
                ["x1", "x2"],
                [d + timedelta(i) for i in range(20)],
                [1.0],
            )
        )
        df = DataFrame(data)
        table = df.pivot_table(values=4, index=[0, 1, 3], columns=[2])

        df2 = df.rename(columns=str)
        table2 = df2.pivot_table(values="4", index=["0", "1", "3"], columns=["2"])

        tm.assert_frame_equal(table, table2, check_names=False)

    def test_pivot_no_level_overlap(self):
        # GH #1181

        data = DataFrame(
            {
                "a": ["a", "a", "a", "a", "b", "b", "b", "b"] * 2,
                "b": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
                "c": (["foo"] * 4 + ["bar"] * 4) * 2,
                "value": np.random.default_rng(2).standard_normal(16),
            }
        )

        table = data.pivot_table("value", index="a", columns=["b", "c"])

        grouped = data.groupby(["a", "b", "c"])["value"].mean()
        expected = grouped.unstack("b").unstack("c").dropna(axis=1, how="all")
        tm.assert_frame_equal(table, expected)

    def test_pivot_columns_lexsorted(self):
        n = 10000

        dtype = np.dtype(
            [
                ("Index", object),
                ("Symbol", object),
                ("Year", int),
                ("Month", int),
                ("Day", int),
                ("Quantity", int),
                ("Price", float),
            ]
        )

        products = np.array(
            [
                ("SP500", "ADBE"),
                ("SP500", "NVDA"),
                ("SP500", "ORCL"),
                ("NDQ100", "AAPL"),
                ("NDQ100", "MSFT"),
                ("NDQ100", "GOOG"),
                ("FTSE", "DGE.L"),
                ("FTSE", "TSCO.L"),
                ("FTSE", "GSK.L"),
            ],
            dtype=[("Index", object), ("Symbol", object)],
        )
        items = np.empty(n, dtype=dtype)
        iproduct = np.random.default_rng(2).integers(0, len(products), n)
        items["Index"] = products["Index"][iproduct]
        items["Symbol"] = products["Symbol"][iproduct]
        dr = date_range(date(2000, 1, 1), date(2010, 12, 31))
        dates = dr[np.random.default_rng(2).integers(0, len(dr), n)]
        items["Year"] = dates.year
        items["Month"] = dates.month
        items["Day"] = dates.day
        items["Price"] = np.random.default_rng(2).lognormal(4.0, 2.0, n)

        df = DataFrame(items)

        pivoted = df.pivot_table(
            "Price",
            index=["Month", "Day"],
            columns=["Index", "Symbol", "Year"],
            aggfunc="mean",
        )

        assert pivoted.columns.is_monotonic_increasing

    def test_pivot_complex_aggfunc(self, data):
        f = {"D": ["std"], "E": ["sum"]}
        expected = data.groupby(["A", "B"]).agg(f).unstack("B")
        result = data.pivot_table(index="A", columns="B", aggfunc=f)

        tm.assert_frame_equal(result, expected)

    def test_margins_no_values_no_cols(self, data):
        # Regression test on pivot table: no values or cols passed.
        result = data[["A", "B"]].pivot_table(
            index=["A", "B"], aggfunc=len, margins=True
        )
        result_list = result.tolist()
        assert sum(result_list[:-1]) == result_list[-1]

    def test_margins_no_values_two_rows(self, data):
        # Regression test on pivot table: no values passed but rows are a
        # multi-index
        result = data[["A", "B", "C"]].pivot_table(
            index=["A", "B"], columns="C", aggfunc=len, margins=True
        )
        assert result.All.tolist() == [3.0, 1.0, 4.0, 3.0, 11.0]

    def test_margins_no_values_one_row_one_col(self, data):
        # Regression test on pivot table: no values passed but row and col
        # defined
        result = data[["A", "B"]].pivot_table(
            index="A", columns="B", aggfunc=len, margins=True
        )
        assert result.All.tolist() == [4.0, 7.0, 11.0]

    def test_margins_no_values_two_row_two_cols(self, data):
        # Regression test on pivot table: no values passed but rows and cols
        # are multi-indexed
        data["D"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        result = data[["A", "B", "C", "D"]].pivot_table(
            index=["A", "B"], columns=["C", "D"], aggfunc=len, margins=True
        )
        assert result.All.tolist() == [3.0, 1.0, 4.0, 3.0, 11.0]

    @pytest.mark.parametrize("margin_name", ["foo", "one", 666, None, ["a", "b"]])
    def test_pivot_table_with_margins_set_margin_name(self, margin_name, data):
        # see gh-3335
        msg = (
            f'Conflicting name "{margin_name}" in margins|'
            "margins_name argument must be a string"
        )
        with pytest.raises(ValueError, match=msg):
            # multi-index index
            pivot_table(
                data,
                values="D",
                index=["A", "B"],
                columns=["C"],
                margins=True,
                margins_name=margin_name,
            )
        with pytest.raises(ValueError, match=msg):
            # multi-index column
            pivot_table(
                data,
                values="D",
                index=["C"],
                columns=["A", "B"],
                margins=True,
                margins_name=margin_name,
            )
        with pytest.raises(ValueError, match=msg):
            # non-multi-index index/column
            pivot_table(
                data,
                values="D",
                index=["A"],
                columns=["B"],
                margins=True,
                margins_name=margin_name,
            )

    def test_pivot_timegrouper(self, using_array_manager):
        df = DataFrame(
            {
                "Branch": "A A A A A A A B".split(),
                "Buyer": "Carl Mark Carl Carl Joe Joe Joe Carl".split(),
                "Quantity": [1, 3, 5, 1, 8, 1, 9, 3],
                "Date": [
                    datetime(2013, 1, 1),
                    datetime(2013, 1, 1),
                    datetime(2013, 10, 1),
                    datetime(2013, 10, 2),
                    datetime(2013, 10, 1),
                    datetime(2013, 10, 2),
                    datetime(2013, 12, 2),
                    datetime(2013, 12, 2),
                ],
            }
        ).set_index("Date")

        expected = DataFrame(
            np.array([10, 18, 3], dtype="int64").reshape(1, 3),
            index=pd.DatetimeIndex([datetime(2013, 12, 31)], freq="YE"),
            columns="Carl Joe Mark".split(),
        )
        expected.index.name = "Date"
        expected.columns.name = "Buyer"

        result = pivot_table(
            df,
            index=Grouper(freq="YE"),
            columns="Buyer",
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index="Buyer",
            columns=Grouper(freq="YE"),
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected.T)

        expected = DataFrame(
            np.array([1, np.nan, 3, 9, 18, np.nan]).reshape(2, 3),
            index=pd.DatetimeIndex(
                [datetime(2013, 1, 1), datetime(2013, 7, 1)], freq="6MS"
            ),
            columns="Carl Joe Mark".split(),
        )
        expected.index.name = "Date"
        expected.columns.name = "Buyer"
        if using_array_manager:
            # INFO(ArrayManager) column without NaNs can preserve int dtype
            expected["Carl"] = expected["Carl"].astype("int64")

        result = pivot_table(
            df,
            index=Grouper(freq="6MS"),
            columns="Buyer",
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index="Buyer",
            columns=Grouper(freq="6MS"),
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected.T)

        # passing the name
        df = df.reset_index()
        result = pivot_table(
            df,
            index=Grouper(freq="6MS", key="Date"),
            columns="Buyer",
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index="Buyer",
            columns=Grouper(freq="6MS", key="Date"),
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected.T)

        msg = "'The grouper name foo is not found'"
        with pytest.raises(KeyError, match=msg):
            pivot_table(
                df,
                index=Grouper(freq="6MS", key="foo"),
                columns="Buyer",
                values="Quantity",
                aggfunc="sum",
            )
        with pytest.raises(KeyError, match=msg):
            pivot_table(
                df,
                index="Buyer",
                columns=Grouper(freq="6MS", key="foo"),
                values="Quantity",
                aggfunc="sum",
            )

        # passing the level
        df = df.set_index("Date")
        result = pivot_table(
            df,
            index=Grouper(freq="6MS", level="Date"),
            columns="Buyer",
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index="Buyer",
            columns=Grouper(freq="6MS", level="Date"),
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected.T)

        msg = "The level foo is not valid"
        with pytest.raises(ValueError, match=msg):
            pivot_table(
                df,
                index=Grouper(freq="6MS", level="foo"),
                columns="Buyer",
                values="Quantity",
                aggfunc="sum",
            )
        with pytest.raises(ValueError, match=msg):
            pivot_table(
                df,
                index="Buyer",
                columns=Grouper(freq="6MS", level="foo"),
                values="Quantity",
                aggfunc="sum",
            )

    def test_pivot_timegrouper_double(self):
        # double grouper
        df = DataFrame(
            {
                "Branch": "A A A A A A A B".split(),
                "Buyer": "Carl Mark Carl Carl Joe Joe Joe Carl".split(),
                "Quantity": [1, 3, 5, 1, 8, 1, 9, 3],
                "Date": [
                    datetime(2013, 11, 1, 13, 0),
                    datetime(2013, 9, 1, 13, 5),
                    datetime(2013, 10, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 11, 1, 20, 0),
                    datetime(2013, 10, 2, 10, 0),
                    datetime(2013, 10, 2, 12, 0),
                    datetime(2013, 12, 5, 14, 0),
                ],
                "PayDay": [
                    datetime(2013, 10, 4, 0, 0),
                    datetime(2013, 10, 15, 13, 5),
                    datetime(2013, 9, 5, 20, 0),
                    datetime(2013, 11, 2, 10, 0),
                    datetime(2013, 10, 7, 20, 0),
                    datetime(2013, 9, 5, 10, 0),
                    datetime(2013, 12, 30, 12, 0),
                    datetime(2013, 11, 20, 14, 0),
                ],
            }
        )

        result = pivot_table(
            df,
            index=Grouper(freq="ME", key="Date"),
            columns=Grouper(freq="ME", key="PayDay"),
            values="Quantity",
            aggfunc="sum",
        )
        expected = DataFrame(
            np.array(
                [
                    np.nan,
                    3,
                    np.nan,
                    np.nan,
                    6,
                    np.nan,
                    1,
                    9,
                    np.nan,
                    9,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    3,
                    np.nan,
                ]
            ).reshape(4, 4),
            index=pd.DatetimeIndex(
                [
                    datetime(2013, 9, 30),
                    datetime(2013, 10, 31),
                    datetime(2013, 11, 30),
                    datetime(2013, 12, 31),
                ],
                freq="ME",
            ),
            columns=pd.DatetimeIndex(
                [
                    datetime(2013, 9, 30),
                    datetime(2013, 10, 31),
                    datetime(2013, 11, 30),
                    datetime(2013, 12, 31),
                ],
                freq="ME",
            ),
        )
        expected.index.name = "Date"
        expected.columns.name = "PayDay"

        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index=Grouper(freq="ME", key="PayDay"),
            columns=Grouper(freq="ME", key="Date"),
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected.T)

        tuples = [
            (datetime(2013, 9, 30), datetime(2013, 10, 31)),
            (datetime(2013, 10, 31), datetime(2013, 9, 30)),
            (datetime(2013, 10, 31), datetime(2013, 11, 30)),
            (datetime(2013, 10, 31), datetime(2013, 12, 31)),
            (datetime(2013, 11, 30), datetime(2013, 10, 31)),
            (datetime(2013, 12, 31), datetime(2013, 11, 30)),
        ]
        idx = MultiIndex.from_tuples(tuples, names=["Date", "PayDay"])
        expected = DataFrame(
            np.array(
                [3, np.nan, 6, np.nan, 1, np.nan, 9, np.nan, 9, np.nan, np.nan, 3]
            ).reshape(6, 2),
            index=idx,
            columns=["A", "B"],
        )
        expected.columns.name = "Branch"

        result = pivot_table(
            df,
            index=[Grouper(freq="ME", key="Date"), Grouper(freq="ME", key="PayDay")],
            columns=["Branch"],
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index=["Branch"],
            columns=[Grouper(freq="ME", key="Date"), Grouper(freq="ME", key="PayDay")],
            values="Quantity",
            aggfunc="sum",
        )
        tm.assert_frame_equal(result, expected.T)

    def test_pivot_datetime_tz(self):
        dates1 = pd.DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ],
            dtype="M8[ns, US/Pacific]",
            name="dt1",
        )
        dates2 = pd.DatetimeIndex(
            [
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
            ],
            dtype="M8[ns, Asia/Tokyo]",
        )
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "dt1": dates1,
                "dt2": dates2,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )

        exp_idx = dates1[:3]
        exp_col1 = Index(["value1", "value1"])
        exp_col2 = Index(["a", "b"], name="label")
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        expected = DataFrame(
            [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]], index=exp_idx, columns=exp_col
        )
        result = pivot_table(df, index=["dt1"], columns=["label"], values=["value1"])
        tm.assert_frame_equal(result, expected)

        exp_col1 = Index(["sum", "sum", "sum", "sum", "mean", "mean", "mean", "mean"])
        exp_col2 = Index(["value1", "value1", "value2", "value2"] * 2)
        exp_col3 = pd.DatetimeIndex(
            ["2013-01-01 15:00:00", "2013-02-01 15:00:00"] * 4,
            dtype="M8[ns, Asia/Tokyo]",
            name="dt2",
        )
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2, exp_col3])
        expected1 = DataFrame(
            np.array(
                [
                    [
                        0,
                        3,
                        1,
                        2,
                    ],
                    [1, 4, 2, 1],
                    [2, 5, 1, 2],
                ],
                dtype="int64",
            ),
            index=exp_idx,
            columns=exp_col[:4],
        )
        expected2 = DataFrame(
            np.array(
                [
                    [0.0, 3.0, 1.0, 2.0],
                    [1.0, 4.0, 2.0, 1.0],
                    [2.0, 5.0, 1.0, 2.0],
                ],
            ),
            index=exp_idx,
            columns=exp_col[4:],
        )
        expected = concat([expected1, expected2], axis=1)

        result = pivot_table(
            df,
            index=["dt1"],
            columns=["dt2"],
            values=["value1", "value2"],
            aggfunc=["sum", "mean"],
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_dtaccessor(self):
        # GH 8103
        dates1 = pd.DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ]
        )
        dates2 = pd.DatetimeIndex(
            [
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-01-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
                "2013-02-01 15:00:00",
            ]
        )
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "dt1": dates1,
                "dt2": dates2,
                "value1": np.arange(6, dtype="int64"),
                "value2": [1, 2] * 3,
            }
        )

        result = pivot_table(
            df, index="label", columns=df["dt1"].dt.hour, values="value1"
        )

        exp_idx = Index(["a", "b"], name="label")
        expected = DataFrame(
            {7: [0.0, 3.0], 8: [1.0, 4.0], 9: [2.0, 5.0]},
            index=exp_idx,
            columns=Index([7, 8, 9], dtype=np.int32, name="dt1"),
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df, index=df["dt2"].dt.month, columns=df["dt1"].dt.hour, values="value1"
        )

        expected = DataFrame(
            {7: [0.0, 3.0], 8: [1.0, 4.0], 9: [2.0, 5.0]},
            index=Index([1, 2], dtype=np.int32, name="dt2"),
            columns=Index([7, 8, 9], dtype=np.int32, name="dt1"),
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index=df["dt2"].dt.year.values,
            columns=[df["dt1"].dt.hour, df["dt2"].dt.month],
            values="value1",
        )

        exp_col = MultiIndex.from_arrays(
            [
                np.array([7, 7, 8, 8, 9, 9], dtype=np.int32),
                np.array([1, 2] * 3, dtype=np.int32),
            ],
            names=["dt1", "dt2"],
        )
        expected = DataFrame(
            np.array([[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]]),
            index=Index([2013], dtype=np.int32),
            columns=exp_col,
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            df,
            index=np.array(["X", "X", "X", "X", "Y", "Y"]),
            columns=[df["dt1"].dt.hour, df["dt2"].dt.month],
            values="value1",
        )
        expected = DataFrame(
            np.array(
                [[0, 3, 1, np.nan, 2, np.nan], [np.nan, np.nan, np.nan, 4, np.nan, 5]]
            ),
            index=["X", "Y"],
            columns=exp_col,
        )
        tm.assert_frame_equal(result, expected)

    def test_daily(self):
        rng = date_range("1/1/2000", "12/31/2004", freq="D")
        ts = Series(np.arange(len(rng)), index=rng)

        result = pivot_table(
            DataFrame(ts), index=ts.index.year, columns=ts.index.dayofyear
        )
        result.columns = result.columns.droplevel(0)

        doy = np.asarray(ts.index.dayofyear)

        expected = {}
        for y in ts.index.year.unique().values:
            mask = ts.index.year == y
            expected[y] = Series(ts.values[mask], index=doy[mask])
        expected = DataFrame(expected, dtype=float).T
        tm.assert_frame_equal(result, expected)

    def test_monthly(self):
        rng = date_range("1/1/2000", "12/31/2004", freq="ME")
        ts = Series(np.arange(len(rng)), index=rng)

        result = pivot_table(DataFrame(ts), index=ts.index.year, columns=ts.index.month)
        result.columns = result.columns.droplevel(0)

        month = np.asarray(ts.index.month)
        expected = {}
        for y in ts.index.year.unique().values:
            mask = ts.index.year == y
            expected[y] = Series(ts.values[mask], index=month[mask])
        expected = DataFrame(expected, dtype=float).T
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_with_iterator_values(self, data):
        # GH 12017
        aggs = {"D": "sum", "E": "mean"}

        pivot_values_list = pivot_table(
            data, index=["A"], values=list(aggs.keys()), aggfunc=aggs
        )

        pivot_values_keys = pivot_table(
            data, index=["A"], values=aggs.keys(), aggfunc=aggs
        )
        tm.assert_frame_equal(pivot_values_keys, pivot_values_list)

        agg_values_gen = (value for value in aggs)
        pivot_values_gen = pivot_table(
            data, index=["A"], values=agg_values_gen, aggfunc=aggs
        )
        tm.assert_frame_equal(pivot_values_gen, pivot_values_list)

    def test_pivot_table_margins_name_with_aggfunc_list(self):
        # GH 13354
        margins_name = "Weekly"
        costs = DataFrame(
            {
                "item": ["bacon", "cheese", "bacon", "cheese"],
                "cost": [2.5, 4.5, 3.2, 3.3],
                "day": ["ME", "ME", "T", "T"],
            }
        )
        table = costs.pivot_table(
            index="item",
            columns="day",
            margins=True,
            margins_name=margins_name,
            aggfunc=["mean", "max"],
        )
        ix = Index(["bacon", "cheese", margins_name], name="item")
        tups = [
            ("mean", "cost", "ME"),
            ("mean", "cost", "T"),
            ("mean", "cost", margins_name),
            ("max", "cost", "ME"),
            ("max", "cost", "T"),
            ("max", "cost", margins_name),
        ]
        cols = MultiIndex.from_tuples(tups, names=[None, None, "day"])
        expected = DataFrame(table.values, index=ix, columns=cols)
        tm.assert_frame_equal(table, expected)

    def test_categorical_margins(self, observed):
        # GH 10989
        df = DataFrame(
            {"x": np.arange(8), "y": np.arange(8) // 4, "z": np.arange(8) % 2}
        )

        expected = DataFrame([[1.0, 2.0, 1.5], [5, 6, 5.5], [3, 4, 3.5]])
        expected.index = Index([0, 1, "All"], name="y")
        expected.columns = Index([0, 1, "All"], name="z")

        table = df.pivot_table("x", "y", "z", dropna=observed, margins=True)
        tm.assert_frame_equal(table, expected)

    def test_categorical_margins_category(self, observed):
        df = DataFrame(
            {"x": np.arange(8), "y": np.arange(8) // 4, "z": np.arange(8) % 2}
        )

        expected = DataFrame([[1.0, 2.0, 1.5], [5, 6, 5.5], [3, 4, 3.5]])
        expected.index = Index([0, 1, "All"], name="y")
        expected.columns = Index([0, 1, "All"], name="z")

        df.y = df.y.astype("category")
        df.z = df.z.astype("category")
        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            table = df.pivot_table("x", "y", "z", dropna=observed, margins=True)
        tm.assert_frame_equal(table, expected)

    def test_margins_casted_to_float(self):
        # GH 24893
        df = DataFrame(
            {
                "A": [2, 4, 6, 8],
                "B": [1, 4, 5, 8],
                "C": [1, 3, 4, 6],
                "D": ["X", "X", "Y", "Y"],
            }
        )

        result = pivot_table(df, index="D", margins=True)
        expected = DataFrame(
            {"A": [3.0, 7.0, 5], "B": [2.5, 6.5, 4.5], "C": [2.0, 5.0, 3.5]},
            index=Index(["X", "Y", "All"], name="D"),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_categorical(self, observed, ordered):
        # gh-21370
        idx = [np.nan, "low", "high", "low", np.nan]
        col = [np.nan, "A", "B", np.nan, "A"]
        df = DataFrame(
            {
                "In": Categorical(idx, categories=["low", "high"], ordered=ordered),
                "Col": Categorical(col, categories=["A", "B"], ordered=ordered),
                "Val": range(1, 6),
            }
        )
        # case with index/columns/value
        result = df.pivot_table(
            index="In", columns="Col", values="Val", observed=observed
        )

        expected_cols = pd.CategoricalIndex(["A", "B"], ordered=ordered, name="Col")

        expected = DataFrame(data=[[2.0, np.nan], [np.nan, 3.0]], columns=expected_cols)
        expected.index = Index(
            Categorical(["low", "high"], categories=["low", "high"], ordered=ordered),
            name="In",
        )

        tm.assert_frame_equal(result, expected)

        # case with columns/value
        result = df.pivot_table(columns="Col", values="Val", observed=observed)

        expected = DataFrame(
            data=[[3.5, 3.0]], columns=expected_cols, index=Index(["Val"])
        )

        tm.assert_frame_equal(result, expected)

    def test_categorical_aggfunc(self, observed):
        # GH 9534
        df = DataFrame(
            {"C1": ["A", "B", "C", "C"], "C2": ["a", "a", "b", "b"], "V": [1, 2, 3, 4]}
        )
        df["C1"] = df["C1"].astype("category")
        msg = "The default value of observed=False is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.pivot_table(
                "V", index="C1", columns="C2", dropna=observed, aggfunc="count"
            )

        expected_index = pd.CategoricalIndex(
            ["A", "B", "C"], categories=["A", "B", "C"], ordered=False, name="C1"
        )
        expected_columns = Index(["a", "b"], name="C2")
        expected_data = np.array([[1, 0], [1, 0], [0, 2]], dtype=np.int64)
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        tm.assert_frame_equal(result, expected)

    def test_categorical_pivot_index_ordering(self, observed):
        # GH 8731
        df = DataFrame(
            {
                "Sales": [100, 120, 220],
                "Month": ["January", "January", "January"],
                "Year": [2013, 2014, 2013],
            }
        )
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        df["Month"] = df["Month"].astype("category").cat.set_categories(months)
        result = df.pivot_table(
            values="Sales",
            index="Month",
            columns="Year",
            observed=observed,
            aggfunc="sum",
        )
        expected_columns = Index([2013, 2014], name="Year", dtype="int64")
        expected_index = pd.CategoricalIndex(
            months, categories=months, ordered=False, name="Month"
        )
        expected_data = [[320, 120]] + [[0, 0]] * 11
        expected = DataFrame(
            expected_data, index=expected_index, columns=expected_columns
        )
        if observed:
            expected = expected.loc[["January"]]

        tm.assert_frame_equal(result, expected)

    def test_pivot_table_not_series(self):
        # GH 4386
        # pivot_table always returns a DataFrame
        # when values is not list like and columns is None
        # and aggfunc is not instance of list
        df = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"], "col3": [1, 3, 9]})

        result = df.pivot_table("col1", index=["col3", "col2"], aggfunc="sum")
        m = MultiIndex.from_arrays([[1, 3, 9], ["C", "D", "E"]], names=["col3", "col2"])
        expected = DataFrame([3, 4, 5], index=m, columns=["col1"])

        tm.assert_frame_equal(result, expected)

        result = df.pivot_table("col1", index="col3", columns="col2", aggfunc="sum")
        expected = DataFrame(
            [[3, np.nan, np.nan], [np.nan, 4, np.nan], [np.nan, np.nan, 5]],
            index=Index([1, 3, 9], name="col3"),
            columns=Index(["C", "D", "E"], name="col2"),
        )

        tm.assert_frame_equal(result, expected)

        result = df.pivot_table("col1", index="col3", aggfunc=["sum"])
        m = MultiIndex.from_arrays([["sum"], ["col1"]])
        expected = DataFrame([3, 4, 5], index=Index([1, 3, 9], name="col3"), columns=m)

        tm.assert_frame_equal(result, expected)

    def test_pivot_margins_name_unicode(self):
        # issue #13292
        greek = "\u0394\u03bf\u03ba\u03b9\u03bc\u03ae"
        frame = DataFrame({"foo": [1, 2, 3]}, columns=Index(["foo"], dtype=object))
        table = pivot_table(
            frame, index=["foo"], aggfunc=len, margins=True, margins_name=greek
        )
        index = Index([1, 2, 3, greek], dtype="object", name="foo")
        expected = DataFrame(index=index, columns=[])
        tm.assert_frame_equal(table, expected)

    def test_pivot_string_as_func(self):
        # GH #18713
        # for correctness purposes
        data = DataFrame(
            {
                "A": [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                    "foo",
                    "foo",
                    "foo",
                ],
                "B": [
                    "one",
                    "one",
                    "one",
                    "two",
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "two",
                    "one",
                ],
                "C": range(11),
            }
        )

        result = pivot_table(data, index="A", columns="B", aggfunc="sum")
        mi = MultiIndex(
            levels=[["C"], ["one", "two"]], codes=[[0, 0], [0, 1]], names=[None, "B"]
        )
        expected = DataFrame(
            {("C", "one"): {"bar": 15, "foo": 13}, ("C", "two"): {"bar": 7, "foo": 20}},
            columns=mi,
        ).rename_axis("A")
        tm.assert_frame_equal(result, expected)

        result = pivot_table(data, index="A", columns="B", aggfunc=["sum", "mean"])
        mi = MultiIndex(
            levels=[["sum", "mean"], ["C"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
            names=[None, None, "B"],
        )
        expected = DataFrame(
            {
                ("mean", "C", "one"): {"bar": 5.0, "foo": 3.25},
                ("mean", "C", "two"): {"bar": 7.0, "foo": 6.666666666666667},
                ("sum", "C", "one"): {"bar": 15, "foo": 13},
                ("sum", "C", "two"): {"bar": 7, "foo": 20},
            },
            columns=mi,
        ).rename_axis("A")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "f, f_numpy",
        [
            ("sum", np.sum),
            ("mean", np.mean),
            ("std", np.std),
            (["sum", "mean"], [np.sum, np.mean]),
            (["sum", "std"], [np.sum, np.std]),
            (["std", "mean"], [np.std, np.mean]),
        ],
    )
    def test_pivot_string_func_vs_func(self, f, f_numpy, data):
        # GH #18713
        # for consistency purposes
        data = data.drop(columns="C")
        result = pivot_table(data, index="A", columns="B", aggfunc=f)
        ops = "|".join(f) if isinstance(f, list) else f
        msg = f"using DataFrameGroupBy.[{ops}]"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = pivot_table(data, index="A", columns="B", aggfunc=f_numpy)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.slow
    def test_pivot_number_of_levels_larger_than_int32(self, monkeypatch):
        # GH 20601
        # GH 26314: Change ValueError to PerformanceWarning
        class MockUnstacker(reshape_lib._Unstacker):
            def __init__(self, *args, **kwargs) -> None:
                # __init__ will raise the warning
                super().__init__(*args, **kwargs)
                raise Exception("Don't compute final result.")

        with monkeypatch.context() as m:
            m.setattr(reshape_lib, "_Unstacker", MockUnstacker)
            df = DataFrame(
                {"ind1": np.arange(2**16), "ind2": np.arange(2**16), "count": 0}
            )

            msg = "The following operation may generate"
            with tm.assert_produces_warning(PerformanceWarning, match=msg):
                with pytest.raises(Exception, match="Don't compute final result."):
                    df.pivot_table(
                        index="ind1", columns="ind2", values="count", aggfunc="count"
                    )

    def test_pivot_table_aggfunc_dropna(self, dropna):
        # GH 22159
        df = DataFrame(
            {
                "fruit": ["apple", "peach", "apple"],
                "size": [1, 1, 2],
                "taste": [7, 6, 6],
            }
        )

        def ret_one(x):
            return 1

        def ret_sum(x):
            return sum(x)

        def ret_none(x):
            return np.nan

        result = pivot_table(
            df, columns="fruit", aggfunc=[ret_sum, ret_none, ret_one], dropna=dropna
        )

        data = [[3, 1, np.nan, np.nan, 1, 1], [13, 6, np.nan, np.nan, 1, 1]]
        col = MultiIndex.from_product(
            [["ret_sum", "ret_none", "ret_one"], ["apple", "peach"]],
            names=[None, "fruit"],
        )
        expected = DataFrame(data, index=["size", "taste"], columns=col)

        if dropna:
            expected = expected.dropna(axis="columns")

        tm.assert_frame_equal(result, expected)

    def test_pivot_table_aggfunc_scalar_dropna(self, dropna):
        # GH 22159
        df = DataFrame(
            {"A": ["one", "two", "one"], "x": [3, np.nan, 2], "y": [1, np.nan, np.nan]}
        )

        result = pivot_table(df, columns="A", aggfunc="mean", dropna=dropna)

        data = [[2.5, np.nan], [1, np.nan]]
        col = Index(["one", "two"], name="A")
        expected = DataFrame(data, index=["x", "y"], columns=col)

        if dropna:
            expected = expected.dropna(axis="columns")

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("margins", [True, False])
    def test_pivot_table_empty_aggfunc(self, margins):
        # GH 9186 & GH 13483 & GH 49240
        df = DataFrame(
            {
                "A": [2, 2, 3, 3, 2],
                "id": [5, 6, 7, 8, 9],
                "C": ["p", "q", "q", "p", "q"],
                "D": [None, None, None, None, None],
            }
        )
        result = df.pivot_table(
            index="A", columns="D", values="id", aggfunc=np.size, margins=margins
        )
        exp_cols = Index([], name="D")
        expected = DataFrame(index=Index([], dtype="int64", name="A"), columns=exp_cols)
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_no_column_raises(self):
        # GH 10326
        def agg(arr):
            return np.mean(arr)

        df = DataFrame({"X": [0, 0, 1, 1], "Y": [0, 1, 0, 1], "Z": [10, 20, 30, 40]})
        with pytest.raises(KeyError, match="notpresent"):
            df.pivot_table("notpresent", "X", "Y", aggfunc=agg)

    def test_pivot_table_multiindex_columns_doctest_case(self):
        # The relevant characteristic is that the call
        #  to maybe_downcast_to_dtype(agged[v], data[v].dtype) in
        #  __internal_pivot_table has `agged[v]` a DataFrame instead of Series,
        #  In this case this is because agged.columns is a MultiIndex and 'v'
        #  is only indexing on its first level.
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )

        table = pivot_table(
            df,
            values=["D", "E"],
            index=["A", "C"],
            aggfunc={"D": "mean", "E": ["min", "max", "mean"]},
        )
        cols = MultiIndex.from_tuples(
            [("D", "mean"), ("E", "max"), ("E", "mean"), ("E", "min")]
        )
        index = MultiIndex.from_tuples(
            [("bar", "large"), ("bar", "small"), ("foo", "large"), ("foo", "small")],
            names=["A", "C"],
        )
        vals = np.array(
            [
                [5.5, 9.0, 7.5, 6.0],
                [5.5, 9.0, 8.5, 8.0],
                [2.0, 5.0, 4.5, 4.0],
                [2.33333333, 6.0, 4.33333333, 2.0],
            ]
        )
        expected = DataFrame(vals, columns=cols, index=index)
        expected[("E", "min")] = expected[("E", "min")].astype(np.int64)
        expected[("E", "max")] = expected[("E", "max")].astype(np.int64)
        tm.assert_frame_equal(table, expected)

    def test_pivot_table_sort_false(self):
        # GH#39143
        df = DataFrame(
            {
                "a": ["d1", "d4", "d3"],
                "col": ["a", "b", "c"],
                "num": [23, 21, 34],
                "year": ["2018", "2018", "2019"],
            }
        )
        result = df.pivot_table(
            index=["a", "col"], columns="year", values="num", aggfunc="sum", sort=False
        )
        expected = DataFrame(
            [[23, np.nan], [21, np.nan], [np.nan, 34]],
            columns=Index(["2018", "2019"], name="year"),
            index=MultiIndex.from_arrays(
                [["d1", "d4", "d3"], ["a", "b", "c"]], names=["a", "col"]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_nullable_margins(self):
        # GH#48681
        df = DataFrame(
            {"a": "A", "b": [1, 2], "sales": Series([10, 11], dtype="Int64")}
        )

        result = df.pivot_table(index="b", columns="a", margins=True, aggfunc="sum")
        expected = DataFrame(
            [[10, 10], [11, 11], [21, 21]],
            index=Index([1, 2, "All"], name="b"),
            columns=MultiIndex.from_tuples(
                [("sales", "A"), ("sales", "All")], names=[None, "a"]
            ),
            dtype="Int64",
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_sort_false_with_multiple_values(self):
        df = DataFrame(
            {
                "firstname": ["John", "Michael"],
                "lastname": ["Foo", "Bar"],
                "height": [173, 182],
                "age": [47, 33],
            }
        )
        result = df.pivot_table(
            index=["lastname", "firstname"], values=["height", "age"], sort=False
        )
        expected = DataFrame(
            [[173.0, 47.0], [182.0, 33.0]],
            columns=["height", "age"],
            index=MultiIndex.from_tuples(
                [("Foo", "John"), ("Bar", "Michael")],
                names=["lastname", "firstname"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_with_margins_and_numeric_columns(self):
        # GH 26568
        df = DataFrame([["a", "x", 1], ["a", "y", 2], ["b", "y", 3], ["b", "z", 4]])
        df.columns = [10, 20, 30]

        result = df.pivot_table(
            index=10, columns=20, values=30, aggfunc="sum", fill_value=0, margins=True
        )

        expected = DataFrame([[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]])
        expected.columns = ["x", "y", "z", "All"]
        expected.index = ["a", "b", "All"]
        expected.columns.name = 20
        expected.index.name = 10

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dropna", [True, False])
    def test_pivot_ea_dtype_dropna(self, dropna):
        # GH#47477
        df = DataFrame({"x": "a", "y": "b", "age": Series([20, 40], dtype="Int64")})
        result = df.pivot_table(
            index="x", columns="y", values="age", aggfunc="mean", dropna=dropna
        )
        expected = DataFrame(
            [[30]],
            index=Index(["a"], name="x"),
            columns=Index(["b"], name="y"),
            dtype="Float64",
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_datetime_warning(self):
        # GH#48683
        df = DataFrame(
            {
                "a": "A",
                "b": [1, 2],
                "date": pd.Timestamp("2019-12-31"),
                "sales": [10.0, 11],
            }
        )
        with tm.assert_produces_warning(None):
            result = df.pivot_table(
                index=["b", "date"], columns="a", margins=True, aggfunc="sum"
            )
        expected = DataFrame(
            [[10.0, 10.0], [11.0, 11.0], [21.0, 21.0]],
            index=MultiIndex.from_arrays(
                [
                    Index([1, 2, "All"], name="b"),
                    Index(
                        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31"), ""],
                        dtype=object,
                        name="date",
                    ),
                ]
            ),
            columns=MultiIndex.from_tuples(
                [("sales", "A"), ("sales", "All")], names=[None, "a"]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_with_mixed_nested_tuples(self, using_array_manager):
        # GH 50342
        df = DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
                ("col5",): [
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "foo",
                    "bar",
                    "bar",
                    "bar",
                    "bar",
                ],
                ("col6", 6): [
                    "one",
                    "one",
                    "one",
                    "two",
                    "two",
                    "one",
                    "one",
                    "two",
                    "two",
                ],
                (7, "seven"): [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
            }
        )
        result = pivot_table(
            df, values="D", index=["A", "B"], columns=[(7, "seven")], aggfunc="sum"
        )
        expected = DataFrame(
            [[4.0, 5.0], [7.0, 6.0], [4.0, 1.0], [np.nan, 6.0]],
            columns=Index(["large", "small"], name=(7, "seven")),
            index=MultiIndex.from_arrays(
                [["bar", "bar", "foo", "foo"], ["one", "two"] * 2], names=["A", "B"]
            ),
        )
        if using_array_manager:
            # INFO(ArrayManager) column without NaNs can preserve int dtype
            expected["small"] = expected["small"].astype("int64")
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_aggfunc_nunique_with_different_values(self):
        test = DataFrame(
            {
                "a": range(10),
                "b": range(10),
                "c": range(10),
                "d": range(10),
            }
        )

        columnval = MultiIndex.from_arrays(
            [
                ["nunique" for i in range(10)],
                ["c" for i in range(10)],
                range(10),
            ],
            names=(None, None, "b"),
        )
        nparr = np.full((10, 10), np.nan)
        np.fill_diagonal(nparr, 1.0)

        expected = DataFrame(nparr, index=Index(range(10), name="a"), columns=columnval)
        result = test.pivot_table(
            index=[
                "a",
            ],
            columns=[
                "b",
            ],
            values=[
                "c",
            ],
            aggfunc=["nunique"],
        )

        tm.assert_frame_equal(result, expected)


class TestPivot:
    def test_pivot(self):
        data = {
            "index": ["A", "B", "C", "C", "B", "A"],
            "columns": ["One", "One", "One", "Two", "Two", "Two"],
            "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }

        frame = DataFrame(data)
        pivoted = frame.pivot(index="index", columns="columns", values="values")

        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        expected.index.name, expected.columns.name = "index", "columns"
        tm.assert_frame_equal(pivoted, expected)

        # name tracking
        assert pivoted.index.name == "index"
        assert pivoted.columns.name == "columns"

        # don't specify values
        pivoted = frame.pivot(index="index", columns="columns")
        assert pivoted.index.name == "index"
        assert pivoted.columns.names == (None, "columns")

    def test_pivot_duplicates(self):
        data = DataFrame(
            {
                "a": ["bar", "bar", "foo", "foo", "foo"],
                "b": ["one", "two", "one", "one", "two"],
                "c": [1.0, 2.0, 3.0, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError, match="duplicate entries"):
            data.pivot(index="a", columns="b", values="c")

    def test_pivot_empty(self):
        df = DataFrame(columns=["a", "b", "c"])
        result = df.pivot(index="a", columns="b", values="c")
        expected = DataFrame(index=[], columns=[])
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_pivot_integer_bug(self, any_string_dtype):
        df = DataFrame(
            data=[("A", "1", "A1"), ("B", "2", "B2")], dtype=any_string_dtype
        )

        result = df.pivot(index=1, columns=0, values=2)
        expected_columns = Index(["A", "B"], name=0, dtype=any_string_dtype)
        if any_string_dtype == "object":
            expected_columns = expected_columns.astype("str")
        tm.assert_index_equal(result.columns, expected_columns)

    def test_pivot_index_none(self):
        # GH#3962
        data = {
            "index": ["A", "B", "C", "C", "B", "A"],
            "columns": ["One", "One", "One", "Two", "Two", "Two"],
            "values": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }

        frame = DataFrame(data).set_index("index")
        result = frame.pivot(columns="columns", values="values")
        expected = DataFrame(
            {
                "One": {"A": 1.0, "B": 2.0, "C": 3.0},
                "Two": {"A": 1.0, "B": 2.0, "C": 3.0},
            }
        )

        expected.index.name, expected.columns.name = "index", "columns"
        tm.assert_frame_equal(result, expected)

        # omit values
        result = frame.pivot(columns="columns")

        expected.columns = MultiIndex.from_tuples(
            [("values", "One"), ("values", "Two")], names=[None, "columns"]
        )
        expected.index.name = "index"
        tm.assert_frame_equal(result, expected, check_names=False)
        assert result.index.name == "index"
        assert result.columns.names == (None, "columns")
        expected.columns = expected.columns.droplevel(0)
        result = frame.pivot(columns="columns", values="values")

        expected.columns.name = "columns"
        tm.assert_frame_equal(result, expected)

    def test_pivot_index_list_values_none_immutable_args(self):
        # GH37635
        df = DataFrame(
            {
                "lev1": [1, 1, 1, 2, 2, 2],
                "lev2": [1, 1, 2, 1, 1, 2],
                "lev3": [1, 2, 1, 2, 1, 2],
                "lev4": [1, 2, 3, 4, 5, 6],
                "values": [0, 1, 2, 3, 4, 5],
            }
        )
        index = ["lev1", "lev2"]
        columns = ["lev3"]
        result = df.pivot(index=index, columns=columns)

        expected = DataFrame(
            np.array(
                [
                    [1.0, 2.0, 0.0, 1.0],
                    [3.0, np.nan, 2.0, np.nan],
                    [5.0, 4.0, 4.0, 3.0],
                    [np.nan, 6.0, np.nan, 5.0],
                ]
            ),
            index=MultiIndex.from_arrays(
                [(1, 1, 2, 2), (1, 2, 1, 2)], names=["lev1", "lev2"]
            ),
            columns=MultiIndex.from_arrays(
                [("lev4", "lev4", "values", "values"), (1, 2, 1, 2)],
                names=[None, "lev3"],
            ),
        )

        tm.assert_frame_equal(result, expected)

        assert index == ["lev1", "lev2"]
        assert columns == ["lev3"]

    def test_pivot_columns_not_given(self):
        # GH#48293
        df = DataFrame({"a": [1], "b": 1})
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
            df.pivot()  # pylint: disable=missing-kwoa

    # this still fails because columns=None gets passed down to unstack as level=None
    # while at that point None was converted to NaN
    @pytest.mark.xfail(
        using_string_dtype(), reason="TODO(infer_string) None is cast to NaN"
    )
    def test_pivot_columns_is_none(self):
        # GH#48293
        df = DataFrame({None: [1], "b": 2, "c": 3})
        result = df.pivot(columns=None)
        expected = DataFrame({("b", 1): [2], ("c", 1): 3})
        tm.assert_frame_equal(result, expected)

        result = df.pivot(columns=None, index="b")
        expected = DataFrame({("c", 1): 3}, index=Index([2], name="b"))
        tm.assert_frame_equal(result, expected)

        result = df.pivot(columns=None, index="b", values="c")
        expected = DataFrame({1: 3}, index=Index([2], name="b"))
        tm.assert_frame_equal(result, expected)

    def test_pivot_index_is_none(self, using_infer_string):
        # GH#48293
        df = DataFrame({None: [1], "b": 2, "c": 3})

        result = df.pivot(columns="b", index=None)
        expected = DataFrame({("c", 2): 3}, index=[1])
        expected.columns.names = [None, "b"]
        tm.assert_frame_equal(result, expected)

        result = df.pivot(columns="b", index=None, values="c")
        expected = DataFrame(3, index=[1], columns=Index([2], name="b"))
        if using_infer_string:
            expected.index.name = np.nan
        tm.assert_frame_equal(result, expected)

    def test_pivot_values_is_none(self):
        # GH#48293
        df = DataFrame({None: [1], "b": 2, "c": 3})

        result = df.pivot(columns="b", index="c", values=None)
        expected = DataFrame(
            1, index=Index([3], name="c"), columns=Index([2], name="b")
        )
        tm.assert_frame_equal(result, expected)

        result = df.pivot(columns="b", values=None)
        expected = DataFrame(1, index=[0], columns=Index([2], name="b"))
        tm.assert_frame_equal(result, expected)

    def test_pivot_not_changing_index_name(self):
        # GH#52692
        df = DataFrame({"one": ["a"], "two": 0, "three": 1})
        expected = df.copy(deep=True)
        df.pivot(index="one", columns="two", values="three")
        tm.assert_frame_equal(df, expected)

    def test_pivot_table_empty_dataframe_correct_index(self):
        # GH 21932
        df = DataFrame([], columns=["a", "b", "value"])
        pivot = df.pivot_table(index="a", columns="b", values="value", aggfunc="count")

        expected = Index([], dtype="object", name="b")
        tm.assert_index_equal(pivot.columns, expected)

    def test_pivot_table_handles_explicit_datetime_types(self):
        # GH#43574
        df = DataFrame(
            [
                {"a": "x", "date_str": "2023-01-01", "amount": 1},
                {"a": "y", "date_str": "2023-01-02", "amount": 2},
                {"a": "z", "date_str": "2023-01-03", "amount": 3},
            ]
        )
        df["date"] = pd.to_datetime(df["date_str"])

        with tm.assert_produces_warning(False):
            pivot = df.pivot_table(
                index=["a", "date"], values=["amount"], aggfunc="sum", margins=True
            )

        expected = MultiIndex.from_tuples(
            [
                ("x", datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")),
                ("y", datetime.strptime("2023-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")),
                ("z", datetime.strptime("2023-01-03 00:00:00", "%Y-%m-%d %H:%M:%S")),
                ("All", ""),
            ],
            names=["a", "date"],
        )
        tm.assert_index_equal(pivot.index, expected)

    def test_pivot_table_with_margins_and_numeric_column_names(self):
        # GH#26568
        df = DataFrame([["a", "x", 1], ["a", "y", 2], ["b", "y", 3], ["b", "z", 4]])

        result = df.pivot_table(
            index=0, columns=1, values=2, aggfunc="sum", fill_value=0, margins=True
        )

        expected = DataFrame(
            [[1, 2, 0, 3], [0, 3, 4, 7], [1, 5, 4, 10]],
            columns=Index(["x", "y", "z", "All"], name=1),
            index=Index(["a", "b", "All"], name=0),
        )
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\tests\test_solvers.py ===
from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import MatrixSymbol, SparseMatrix
from sympy.polys.polytools import Poly, groebner
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)

from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
    solve_undetermined_coeffs
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
    det_quick, det_perm, det_minor, _simple_dens, denoms

from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf

from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn

from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


def test_swap_back():
    f, g = map(Function, 'fg')
    fx, gx = f(x), g(x)
    assert solve([fx + y - 2, fx - gx - 5], fx, y, gx) == \
        {fx: gx + 5, y: -gx - 3}
    assert solve(fx + gx*x - 2, [fx, gx], dict=True) == [{fx: 2, gx: 0}]
    assert solve(fx + gx**2*x - y, [fx, gx], dict=True) == [{fx: y, gx: 0}]
    assert solve([f(1) - 2, x + 2], dict=True) == [{x: -2, f(1): 2}]


def guess_solve_strategy(eq, symbol):
    try:
        solve(eq, symbol)
        return True
    except (TypeError, NotImplementedError):
        return False


def test_guess_poly():
    # polynomial equations
    assert guess_solve_strategy( S(4), x )  # == GS_POLY
    assert guess_solve_strategy( x, x )  # == GS_POLY
    assert guess_solve_strategy( x + a, x )  # == GS_POLY
    assert guess_solve_strategy( 2*x, x )  # == GS_POLY
    assert guess_solve_strategy( x + sqrt(2), x)  # == GS_POLY
    assert guess_solve_strategy( x + 2**Rational(1, 4), x)  # == GS_POLY
    assert guess_solve_strategy( x**2 + 1, x )  # == GS_POLY
    assert guess_solve_strategy( x**2 - 1, x )  # == GS_POLY
    assert guess_solve_strategy( x*y + y, x )  # == GS_POLY
    assert guess_solve_strategy( x*exp(y) + y, x)  # == GS_POLY
    assert guess_solve_strategy(
        (x - y**3)/(y**2*sqrt(1 - y**2)), x)  # == GS_POLY


def test_guess_poly_cv():
    # polynomial equations via a change of variable
    assert guess_solve_strategy( sqrt(x) + 1, x )  # == GS_POLY_CV_1
    assert guess_solve_strategy(
        x**Rational(1, 3) + sqrt(x) + 1, x )  # == GS_POLY_CV_1
    assert guess_solve_strategy( 4*x*(1 - sqrt(x)), x )  # == GS_POLY_CV_1

    # polynomial equation multiplying both sides by x**n
    assert guess_solve_strategy( x + 1/x + y, x )  # == GS_POLY_CV_2


def test_guess_rational_cv():
    # rational functions
    assert guess_solve_strategy( (x + 1)/(x**2 + 2), x)  # == GS_RATIONAL
    assert guess_solve_strategy(
        (x - y**3)/(y**2*sqrt(1 - y**2)), y)  # == GS_RATIONAL_CV_1

    # rational functions via the change of variable y -> x**n
    assert guess_solve_strategy( (sqrt(x) + 1)/(x**Rational(1, 3) + sqrt(x) + 1), x ) \
        #== GS_RATIONAL_CV_1


def test_guess_transcendental():
    #transcendental functions
    assert guess_solve_strategy( exp(x) + 1, x )  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy( 2*cos(x) - y, x )  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy(
        exp(x) + exp(-x) - y, x )  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy(3**x - 10, x)  # == GS_TRANSCENDENTAL
    assert guess_solve_strategy(-3**x + 10, x)  # == GS_TRANSCENDENTAL

    assert guess_solve_strategy(a*x**b - y, x)  # == GS_TRANSCENDENTAL


def test_solve_args():
    # equation container, issue 5113
    ans = {x: -3, y: 1}
    eqs = (x + 5*y - 2, -3*x + 6*y - 15)
    assert all(solve(container(eqs), x, y) == ans for container in
        (tuple, list, set, frozenset))
    assert solve(Tuple(*eqs), x, y) == ans
    # implicit symbol to solve for
    assert set(solve(x**2 - 4)) == {S(2), -S(2)}
    assert solve([x + y - 3, x - y - 5]) == {x: 4, y: -1}
    assert solve(x - exp(x), x, implicit=True) == [exp(x)]
    # no symbol to solve for
    assert solve(42) == solve(42, x) == []
    assert solve([1, 2]) == []
    assert solve([sqrt(2)],[x]) == []
    # duplicate symbols raises
    raises(ValueError, lambda: solve((x - 3, y + 2), x, y, x))
    raises(ValueError, lambda: solve(x, x, x))
    # no error in exclude
    assert solve(x, x, exclude=[y, y]) == [0]
    # duplicate symbols raises
    raises(ValueError, lambda: solve((x - 3, y + 2), x, y, x))
    raises(ValueError, lambda: solve(x, x, x))
    # no error in exclude
    assert solve(x, x, exclude=[y, y]) == [0]
    # unordered symbols
    # only 1
    assert solve(y - 3, {y}) == [3]
    # more than 1
    assert solve(y - 3, {x, y}) == [{y: 3}]
    # multiple symbols: take the first linear solution+
    # - return as tuple with values for all requested symbols
    assert solve(x + y - 3, [x, y]) == [(3 - y, y)]
    # - unless dict is True
    assert solve(x + y - 3, [x, y], dict=True) == [{x: 3 - y}]
    # - or no symbols are given
    assert solve(x + y - 3) == [{x: 3 - y}]
    # multiple symbols might represent an undetermined coefficients system
    assert solve(a + b*x - 2, [a, b]) == {a: 2, b: 0}
    assert solve((a + b)*x + b - c, [a, b]) == {a: -c, b: c}
    eq = a*x**2 + b*x + c - ((x - h)**2 + 4*p*k)/4/p
    # - check that flags are obeyed
    sol = solve(eq, [h, p, k], exclude=[a, b, c])
    assert sol == {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}
    assert solve(eq, [h, p, k], dict=True) == [sol]
    assert solve(eq, [h, p, k], set=True) == \
        ([h, p, k], {(-b/(2*a), 1/(4*a), (4*a*c - b**2)/(4*a))})
    # issue 23889 - polysys not simplified
    assert solve(eq, [h, p, k], exclude=[a, b, c], simplify=False) == \
        {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}
    # but this only happens when system has a single solution
    args = (a + b)*x - b**2 + 2, a, b
    assert solve(*args) == [((b**2 - b*x - 2)/x, b)]
    # and if the system has a solution; the following doesn't so
    # an algebraic solution is returned
    assert solve(a*x + b**2/(x + 4) - 3*x - 4/x, a, b, dict=True) == \
        [{a: (-b**2*x + 3*x**3 + 12*x**2 + 4*x + 16)/(x**2*(x + 4))}]
    # failed single equation
    assert solve(1/(1/x - y + exp(y))) == []
    raises(
        NotImplementedError, lambda: solve(exp(x) + sin(x) + exp(y) + sin(y)))
    # failed system
    # --  when no symbols given, 1 fails
    assert solve([y, exp(x) + x]) == [{x: -LambertW(1), y: 0}]
    #     both fail
    assert solve(
        (exp(x) - x, exp(y) - y)) == [{x: -LambertW(-1), y: -LambertW(-1)}]
    # --  when symbols given
    assert solve([y, exp(x) + x], x, y) == [(-LambertW(1), 0)]
    # symbol is a number
    assert solve(x**2 - pi, pi) == [x**2]
    # no equations
    assert solve([], [x]) == []
    # nonlinear system
    assert solve((x**2 - 4, y - 2), x, y) == [(-2, 2), (2, 2)]
    assert solve((x**2 - 4, y - 2), y, x) == [(2, -2), (2, 2)]
    assert solve((x**2 - 4 + z, y - 2 - z), a, z, y, x, set=True
        ) == ([a, z, y, x], {
        (a, z, z + 2, -sqrt(4 - z)),
        (a, z, z + 2, sqrt(4 - z))})
    # overdetermined system
    # - nonlinear
    assert solve([(x + y)**2 - 4, x + y - 2]) == [{x: -y + 2}]
    # - linear
    assert solve((x + y - 2, 2*x + 2*y - 4)) == {x: -y + 2}
    # When one or more args are Boolean
    assert solve(Eq(x**2, 0.0)) == [0.0]  # issue 19048
    assert solve([True, Eq(x, 0)], [x], dict=True) == [{x: 0}]
    assert solve([Eq(x, x), Eq(x, 0), Eq(x, x+1)], [x], dict=True) == []
    assert not solve([Eq(x, x+1), x < 2], x)
    assert solve([Eq(x, 0), x+1<2]) == Eq(x, 0)
    assert solve([Eq(x, x), Eq(x, x+1)], x) == []
    assert solve(True, x) == []
    assert solve([x - 1, False], [x], set=True) == ([], set())
    assert solve([-y*(x + y - 1)/2, (y - 1)/x/y + 1/y],
        set=True, check=False) == ([x, y], {(1 - y, y), (x, 0)})
    # ordering should be canonical, fastest to order by keys instead
    # of by size
    assert list(solve((y - 1, x - sqrt(3)*z)).keys()) == [x, y]
    # as set always returns as symbols, set even if no solution
    assert solve([x - 1, x], (y, x), set=True) == ([y, x], set())
    assert solve([x - 1, x], {y, x}, set=True) == ([x, y], set())


def test_solve_polynomial1():
    assert solve(3*x - 2, x) == [Rational(2, 3)]
    assert solve(Eq(3*x, 2), x) == [Rational(2, 3)]

    assert set(solve(x**2 - 1, x)) == {-S.One, S.One}
    assert set(solve(Eq(x**2, 1), x)) == {-S.One, S.One}

    assert solve(x - y**3, x) == [y**3]
    rx = root(x, 3)
    assert solve(x - y**3, y) == [
        rx, -rx/2 - sqrt(3)*I*rx/2, -rx/2 +  sqrt(3)*I*rx/2]
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    assert solve([a11*x + a12*y - b1, a21*x + a22*y - b2], x, y) == \
        {
            x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21),
            y: (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
        }

    solution = {x: S.Zero, y: S.Zero}

    assert solve((x - y, x + y), x, y ) == solution
    assert solve((x - y, x + y), (x, y)) == solution
    assert solve((x - y, x + y), [x, y]) == solution

    assert set(solve(x**3 - 15*x - 4, x)) == {
        -2 + 3**S.Half,
        S(4),
        -2 - 3**S.Half
    }

    assert set(solve((x**2 - 1)**2 - a, x)) == \
        {sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
             sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))}


def test_solve_polynomial2():
    assert solve(4, x) == []


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to a polynomial equation
    using the change of variable y -> x**Rational(p, q)
    """
    assert solve( sqrt(x) - 1, x) == [1]
    assert solve( sqrt(x) - 2, x) == [4]
    assert solve( x**Rational(1, 4) - 2, x) == [16]
    assert solve( x**Rational(1, 3) - 3, x) == [27]
    assert solve(sqrt(x) + x**Rational(1, 3) + x**Rational(1, 4), x) == [0]


def test_solve_polynomial_cv_1b():
    assert set(solve(4*x*(1 - a*sqrt(x)), x)) == {S.Zero, 1/a**2}
    assert set(solve(x*(root(x, 3) - 3), x)) == {S.Zero, S(27)}


def test_solve_polynomial_cv_2():
    """
    Test for solving on equations that can be converted to a polynomial equation
    multiplying both sides of the equation by x**m
    """
    assert solve(x + 1/x - 1, x) in \
        [[ S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2],
         [ S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]]


def test_quintics_1():
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    s = solve(f, check=False)
    for r in s:
        res = f.subs(x, r.n()).n()
        assert tn(res, 0)

    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = solve(f)
    for r in s:
        assert r.func == CRootOf

    # if one uses solve to get the roots of a polynomial that has a CRootOf
    # solution, make sure that the use of nfloat during the solve process
    # doesn't fail. Note: if you want numerical solutions to a polynomial
    # it is *much* faster to use nroots to get them than to solve the
    # equation only to get RootOf solutions which are then numerically
    # evaluated. So for eq = x**5 + 3*x + 7 do Poly(eq).nroots() rather
    # than [i.n() for i in solve(eq)] to get the numerical roots of eq.
    assert nfloat(solve(x**5 + 3*x**3 + 7)[0], exponent=False) == \
        CRootOf(x**5 + 3*x**3 + 7, 0).n()


def test_quintics_2():
    f = x**5 + 15*x + 12
    s = solve(f, check=False)
    for r in s:
        res = f.subs(x, r.n()).n()
        assert tn(res, 0)

    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = solve(f)
    for r in s:
        assert r.func == CRootOf

    assert solve(x**5 - 6*x**3 - 6*x**2 + x - 6) == [
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 0),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 1),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 2),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 3),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 4)]

def test_quintics_3():
    y = x**5 + x**3 - 2**Rational(1, 3)
    assert solve(y) == solve(-y) == []


def test_highorder_poly():
    # just testing that the uniq generator is unpacked
    sol = solve(x**6 - 2*x + 2)
    assert all(isinstance(i, CRootOf) for i in sol) and len(sol) == 6


def test_solve_rational():
    """Test solve for rational functions"""
    assert solve( ( x - y**3 )/( (y**2)*sqrt(1 - y**2) ), x) == [y**3]


def test_solve_conjugate():
    """Test solve for simple conjugate functions"""
    assert solve(conjugate(x) -3 + I) == [3 + I]


def test_solve_nonlinear():
    assert solve(x**2 - y**2, x, y, dict=True) == [{x: -y}, {x: y}]
    assert solve(x**2 - y**2/exp(x), y, x, dict=True) == [{y: -x*sqrt(exp(x))},
                                                          {y: x*sqrt(exp(x))}]


def test_issue_8666():
    x = symbols('x')
    assert solve(Eq(x**2 - 1/(x**2 - 4), 4 - 1/(x**2 - 4)), x) == []
    assert solve(Eq(x + 1/x, 1/x), x) == []


def test_issue_7228():
    assert solve(4**(2*(x**2) + 2*x) - 8, x) == [Rational(-3, 2), S.Half]


def test_issue_7190():
    assert solve(log(x-3) + log(x+3), x) == [sqrt(10)]


def test_issue_21004():
    x = symbols('x')
    f = x/sqrt(x**2+1)
    f_diff = f.diff(x)
    assert solve(f_diff, x) == []


def test_issue_24650():
    x = symbols('x')
    r = solve(Eq(Piecewise((x, Eq(x, 0) | (x > 1))), 0))
    assert r == [0]
    r = checksol(Eq(Piecewise((x, Eq(x, 0) | (x > 1))), 0), x, sol=0)
    assert r is True


def test_linear_system():
    x, y, z, t, n = symbols('x, y, z, t, n')

    assert solve([x - 1, x - y, x - 2*y, y - 1], [x, y]) == []

    assert solve([x - 1, x - y, x - 2*y, x - 1], [x, y]) == []
    assert solve([x - 1, x - 1, x - y, x - 2*y], [x, y]) == []

    assert solve([x + 5*y - 2, -3*x + 6*y - 15], x, y) == {x: -3, y: 1}

    M = Matrix([[0, 0, n*(n + 1), (n + 1)**2, 0],
                [n + 1, n + 1, -2*n - 1, -(n + 1), 0],
                [-1, 0, 1, 0, 0]])

    assert solve_linear_system(M, x, y, z, t) == \
        {x: t*(-n-1)/n, y: 0, z: t*(-n-1)/n}

    assert solve([x + y + z + t, -z - t], x, y, z, t) == {x: -y, z: -t}


@XFAIL
def test_linear_system_xfail():
    # https://github.com/sympy/sympy/issues/6420
    M = Matrix([[0,    15.0, 10.0, 700.0],
                [1,    1,    1,    100.0],
                [0,    10.0, 5.0,  200.0],
                [-5.0, 0,    0,    0    ]])

    assert solve_linear_system(M, x, y, z) == {x: 0, y: -60.0, z: 160.0}


def test_linear_system_function():
    a = Function('a')
    assert solve([a(0, 0) + a(0, 1) + a(1, 0) + a(1, 1), -a(1, 0) - a(1, 1)],
        a(0, 0), a(0, 1), a(1, 0), a(1, 1)) == {a(1, 0): -a(1, 1), a(0, 0): -a(0, 1)}


def test_linear_system_symbols_doesnt_hang_1():

    def _mk_eqs(wy):
        # Equations for fitting a wy*2 - 1 degree polynomial between two points,
        # at end points derivatives are known up to order: wy - 1
        order = 2*wy - 1
        x, x0, x1 = symbols('x, x0, x1', real=True)
        y0s = symbols('y0_:{}'.format(wy), real=True)
        y1s = symbols('y1_:{}'.format(wy), real=True)
        c = symbols('c_:{}'.format(order+1), real=True)

        expr = sum(coeff*x**o for o, coeff in enumerate(c))
        eqs = []
        for i in range(wy):
            eqs.append(expr.diff(x, i).subs({x: x0}) - y0s[i])
            eqs.append(expr.diff(x, i).subs({x: x1}) - y1s[i])
        return eqs, c

    #
    # The purpose of this test is just to see that these calls don't hang. The
    # expressions returned are complicated so are not included here. Testing
    # their correctness takes longer than solving the system.
    #

    for n in range(1, 7+1):
        eqs, c = _mk_eqs(n)
        solve(eqs, c)


def test_linear_system_symbols_doesnt_hang_2():

    M = Matrix([
        [66, 24, 39, 50, 88, 40, 37, 96, 16, 65, 31, 11, 37, 72, 16, 19, 55, 37, 28, 76],
        [10, 93, 34, 98, 59, 44, 67, 74, 74, 94, 71, 61, 60, 23,  6,  2, 57,  8, 29, 78],
        [19, 91, 57, 13, 64, 65, 24, 53, 77, 34, 85, 58, 87, 39, 39,  7, 36, 67, 91,  3],
        [74, 70, 15, 53, 68, 43, 86, 83, 81, 72, 25, 46, 67, 17, 59, 25, 78, 39, 63,  6],
        [69, 40, 67, 21, 67, 40, 17, 13, 93, 44, 46, 89, 62, 31, 30, 38, 18, 20, 12, 81],
        [50, 22, 74, 76, 34, 45, 19, 76, 28, 28, 11, 99, 97, 82,  8, 46, 99, 57, 68, 35],
        [58, 18, 45, 88, 10, 64,  9, 34, 90, 82, 17, 41, 43, 81, 45, 83, 22, 88, 24, 39],
        [42, 21, 70, 68,  6, 33, 64, 81, 83, 15, 86, 75, 86, 17, 77, 34, 62, 72, 20, 24],
        [ 7,  8,  2, 72, 71, 52, 96,  5, 32, 51, 31, 36, 79, 88, 25, 77, 29, 26, 33, 13],
        [19, 31, 30, 85, 81, 39, 63, 28, 19, 12, 16, 49, 37, 66, 38, 13,  3, 71, 61, 51],
        [29, 82, 80, 49, 26, 85,  1, 37,  2, 74, 54, 82, 26, 47, 54,  9, 35,  0, 99, 40],
        [15, 49, 82, 91, 93, 57, 45, 25, 45, 97, 15, 98, 48, 52, 66, 24, 62, 54, 97, 37],
        [62, 23, 73, 53, 52, 86, 28, 38,  0, 74, 92, 38, 97, 70, 71, 29, 26, 90, 67, 45],
        [ 2, 32, 23, 24, 71, 37, 25, 71,  5, 41, 97, 65, 93, 13, 65, 45, 25, 88, 69, 50],
        [40, 56,  1, 29, 79, 98, 79, 62, 37, 28, 45, 47,  3,  1, 32, 74, 98, 35, 84, 32],
        [33, 15, 87, 79, 65,  9, 14, 63, 24, 19, 46, 28, 74, 20, 29, 96, 84, 91, 93,  1],
        [97, 18, 12, 52,  1,  2, 50, 14, 52, 76, 19, 82, 41, 73, 51, 79, 13,  3, 82, 96],
        [40, 28, 52, 10, 10, 71, 56, 78, 82,  5, 29, 48,  1, 26, 16, 18, 50, 76, 86, 52],
        [38, 89, 83, 43, 29, 52, 90, 77, 57,  0, 67, 20, 81, 88, 48, 96, 88, 58, 14,  3]])

    syms = x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18 = symbols('x:19')

    sol = {
        x0:  -S(1967374186044955317099186851240896179)/3166636564687820453598895768302256588,
        x1:  -S(84268280268757263347292368432053826)/791659141171955113399723942075564147,
        x2:  -S(229962957341664730974463872411844965)/1583318282343910226799447884151128294,
        x3:   S(990156781744251750886760432229180537)/6333273129375640907197791536604513176,
        x4:  -S(2169830351210066092046760299593096265)/18999819388126922721593374609813539528,
        x5:   S(4680868883477577389628494526618745355)/9499909694063461360796687304906769764,
        x6:  -S(1590820774344371990683178396480879213)/3166636564687820453598895768302256588,
        x7:  -S(54104723404825537735226491634383072)/339282489073695048599881689460956063,
        x8:   S(3182076494196560075964847771774733847)/6333273129375640907197791536604513176,
        x9:  -S(10870817431029210431989147852497539675)/18999819388126922721593374609813539528,
        x10: -S(13118019242576506476316318268573312603)/18999819388126922721593374609813539528,
        x11: -S(5173852969886775824855781403820641259)/4749954847031730680398343652453384882,
        x12:  S(4261112042731942783763341580651820563)/4749954847031730680398343652453384882,
        x13: -S(821833082694661608993818117038209051)/6333273129375640907197791536604513176,
        x14:  S(906881575107250690508618713632090559)/904753304196520129599684505229216168,
        x15: -S(732162528717458388995329317371283987)/6333273129375640907197791536604513176,
        x16:  S(4524215476705983545537087360959896817)/9499909694063461360796687304906769764,
        x17: -S(3898571347562055611881270844646055217)/6333273129375640907197791536604513176,
        x18:  S(7513502486176995632751685137907442269)/18999819388126922721593374609813539528
    }

    eqs = list(M * Matrix(syms + (1,)))
    assert solve(eqs, syms) == sol

    y = Symbol('y')
    eqs = list(y * M * Matrix(syms + (1,)))
    assert solve(eqs, syms) == sol


def test_linear_systemLU():
    n = Symbol('n')

    M = Matrix([[1, 2, 0, 1], [1, 3, 2*n, 1], [4, -1, n**2, 1]])

    assert solve_linear_system_LU(M, [x, y, z]) == {z: -3/(n**2 + 18*n),
                                                  x: 1 - 12*n/(n**2 + 18*n),
                                                  y: 6*n/(n**2 + 18*n)}

# Note: multiple solutions exist for some of these equations, so the tests
# should be expected to break if the implementation of the solver changes
# in such a way that a different branch is chosen

@slow
def test_solve_transcendental():
    from sympy.abc import a, b

    assert solve(exp(x) - 3, x) == [log(3)]
    assert set(solve((a*x + b)*(exp(x) - 3), x)) == {-b/a, log(3)}
    assert solve(cos(x) - y, x) == [-acos(y) + 2*pi, acos(y)]
    assert solve(2*cos(x) - y, x) == [-acos(y/2) + 2*pi, acos(y/2)]
    assert solve(Eq(cos(x), sin(x)), x) == [pi/4]

    assert set(solve(exp(x) + exp(-x) - y, x)) in [{
        log(y/2 - sqrt(y**2 - 4)/2),
        log(y/2 + sqrt(y**2 - 4)/2),
    }, {
        log(y - sqrt(y**2 - 4)) - log(2),
        log(y + sqrt(y**2 - 4)) - log(2)},
    {
        log(y/2 - sqrt((y - 2)*(y + 2))/2),
        log(y/2 + sqrt((y - 2)*(y + 2))/2)}]
    assert solve(exp(x) - 3, x) == [log(3)]
    assert solve(Eq(exp(x), 3), x) == [log(3)]
    assert solve(log(x) - 3, x) == [exp(3)]
    assert solve(sqrt(3*x) - 4, x) == [Rational(16, 3)]
    assert solve(3**(x + 2), x) == []
    assert solve(3**(2 - x), x) == []
    assert solve(x + 2**x, x) == [-LambertW(log(2))/log(2)]
    assert solve(2*x + 5 + log(3*x - 2), x) == \
        [Rational(2, 3) + LambertW(2*exp(Rational(-19, 3))/3)/2]
    assert solve(3*x + log(4*x), x) == [LambertW(Rational(3, 4))/3]
    assert set(solve((2*x + 8)*(8 + exp(x)), x)) == {S(-4), log(8) + pi*I}
    eq = 2*exp(3*x + 4) - 3
    ans = solve(eq, x)  # this generated a failure in flatten
    assert len(ans) == 3 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)
    assert solve(2*log(3*x + 4) - 3, x) == [(exp(Rational(3, 2)) - 4)/3]
    assert solve(exp(x) + 1, x) == [pi*I]

    eq = 2*(3*x + 4)**5 - 6*7**(3*x + 9)
    result = solve(eq, x)
    x0 = -log(2401)
    x1 = 3**Rational(1, 5)
    x2 = log(7**(7*x1/20))
    x3 = sqrt(2)
    x4 = sqrt(5)
    x5 = x3*sqrt(x4 - 5)
    x6 = x4 + 1
    x7 = 1/(3*log(7))
    x8 = -x4
    x9 = x3*sqrt(x8 - 5)
    x10 = x8 + 1
    ans = [x7*(x0 - 5*LambertW(x2*(-x5 + x6))),
           x7*(x0 - 5*LambertW(x2*(x5 + x6))),
           x7*(x0 - 5*LambertW(x2*(x10 - x9))),
           x7*(x0 - 5*LambertW(x2*(x10 + x9))),
           x7*(x0 - 5*LambertW(-log(7**(7*x1/5))))]
    assert result == ans, result
    # it works if expanded, too
    assert solve(eq.expand(), x) == result

    assert solve(z*cos(x) - y, x) == [-acos(y/z) + 2*pi, acos(y/z)]
    assert solve(z*cos(2*x) - y, x) == [-acos(y/z)/2 + pi, acos(y/z)/2]
    assert solve(z*cos(sin(x)) - y, x) == [
        pi - asin(acos(y/z)), asin(acos(y/z) - 2*pi) + pi,
        -asin(acos(y/z) - 2*pi), asin(acos(y/z))]

    assert solve(z*cos(x), x) == [pi/2, pi*Rational(3, 2)]

    # issue 4508
    assert solve(y - b*x/(a + x), x) in [[-a*y/(y - b)], [a*y/(b - y)]]
    assert solve(y - b*exp(a/x), x) == [a/log(y/b)]
    # issue 4507
    assert solve(y - b/(1 + a*x), x) in [[(b - y)/(a*y)], [-((y - b)/(a*y))]]
    # issue 4506
    assert solve(y - a*x**b, x) == [(y/a)**(1/b)]
    # issue 4505
    assert solve(z**x - y, x) == [log(y)/log(z)]
    # issue 4504
    assert solve(2**x - 10, x) == [1 + log(5)/log(2)]
    # issue 6744
    assert solve(x*y) == [{x: 0}, {y: 0}]
    assert solve([x*y]) == [{x: 0}, {y: 0}]
    assert solve(x**y - 1) == [{x: 1}, {y: 0}]
    assert solve([x**y - 1]) == [{x: 1}, {y: 0}]
    assert solve(x*y*(x**2 - y**2)) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    assert solve([x*y*(x**2 - y**2)]) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    # issue 4739
    assert solve(exp(log(5)*x) - 2**x, x) == [0]
    # issue 14791
    assert solve(exp(log(5)*x) - exp(log(2)*x), x) == [0]
    f = Function('f')
    assert solve(y*f(log(5)*x) - y*f(log(2)*x), x) == [0]
    assert solve(f(x) - f(0), x) == [0]
    assert solve(f(x) - f(2 - x), x) == [1]
    raises(NotImplementedError, lambda: solve(f(x, y) - f(1, 2), x))
    raises(NotImplementedError, lambda: solve(f(x, y) - f(2 - x, 2), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1 - x), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1), x))

    # misc
    # make sure that the right variables is picked up in tsolve
    # shouldn't generate a GeneratorsNeeded error in _tsolve when the NaN is generated
    # for eq_down. Actual answers, as determined numerically are approx. +/- 0.83
    raises(NotImplementedError, lambda:
        solve(sinh(x)*sinh(sinh(x)) + cosh(x)*cosh(sinh(x)) - 3))

    # watch out for recursive loop in tsolve
    raises(NotImplementedError, lambda: solve((x + 2)**y*x - 3, x))

    # issue 7245
    assert solve(sin(sqrt(x))) == [0, pi**2]

    # issue 7602
    a, b = symbols('a, b', real=True, negative=False)
    assert str(solve(Eq(a, 0.5 - cos(pi*b)/2), b)) == \
        '[2.0 - 0.318309886183791*acos(1.0 - 2.0*a), 0.318309886183791*acos(1.0 - 2.0*a)]'

    # issue 15325
    assert solve(y**(1/x) - z, x) == [log(y)/log(z)]

    # issue 25685 (basic trig identities should give simple solutions)
    for yi in [cos(2*x),sin(2*x),cos(x - pi/3)]:
        sol = solve([cos(x) - S(3)/5, yi - y])
        assert (sol[0][y] + sol[1][y]).is_Rational, (yi,sol)
    # don't allow massive expansion
    assert solve(cos(1000*x) - S.Half) == [pi/3000, pi/600]
    assert solve(cos(x - 1000*y) - 1, x) == [1000*y, 1000*y + 2*pi]
    assert solve(cos(x + y + z) - 1, x) == [-y - z, -y - z + 2*pi]

    # issue 26008
    assert solve(sin(x + pi/6)) == [-pi/6, 5*pi/6]


def test_solve_for_functions_derivatives():
    t = Symbol('t')
    x = Function('x')(t)
    y = Function('y')(t)
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    soln = solve([a11*x + a12*y - b1, a21*x + a22*y - b2], x, y)
    assert soln == {
        x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21),
        y: (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
    }

    assert solve(x - 1, x) == [1]
    assert solve(3*x - 2, x) == [Rational(2, 3)]

    soln = solve([a11*x.diff(t) + a12*y.diff(t) - b1, a21*x.diff(t) +
            a22*y.diff(t) - b2], x.diff(t), y.diff(t))
    assert soln == { y.diff(t): (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
            x.diff(t): (a22*b1 - a12*b2)/(a11*a22 - a12*a21) }

    assert solve(x.diff(t) - 1, x.diff(t)) == [1]
    assert solve(3*x.diff(t) - 2, x.diff(t)) == [Rational(2, 3)]

    eqns = {3*x - 1, 2*y - 4}
    assert solve(eqns, {x, y}) == { x: Rational(1, 3), y: 2 }
    x = Symbol('x')
    f = Function('f')
    F = x**2 + f(x)**2 - 4*x - 1
    assert solve(F.diff(x), diff(f(x), x)) == [(-x + 2)/f(x)]

    # Mixed cased with a Symbol and a Function
    x = Symbol('x')
    y = Function('y')(t)

    soln = solve([a11*x + a12*y.diff(t) - b1, a21*x +
            a22*y.diff(t) - b2], x, y.diff(t))
    assert soln == { y.diff(t): (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
            x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21) }

    # issue 13263
    x = Symbol('x')
    f = Function('f')
    soln = solve([f(x).diff(x) + f(x).diff(x, 2) - 1, f(x).diff(x) - f(x).diff(x, 2)],
            f(x).diff(x), f(x).diff(x, 2))
    assert soln == { f(x).diff(x, 2): S(1)/2, f(x).diff(x): S(1)/2 }

    soln = solve([f(x).diff(x, 2) + f(x).diff(x, 3) - 1, 1 - f(x).diff(x, 2) -
            f(x).diff(x, 3), 1 - f(x).diff(x,3)], f(x).diff(x, 2), f(x).diff(x, 3))
    assert soln == { f(x).diff(x, 2): 0, f(x).diff(x, 3): 1 }


def test_issue_3725():
    f = Function('f')
    F = x**2 + f(x)**2 - 4*x - 1
    e = F.diff(x)
    assert solve(e, f(x).diff(x)) in [[(2 - x)/f(x)], [-((x - 2)/f(x))]]


def test_solve_Matrix():
    # https://github.com/sympy/sympy/issues/3870
    a, b, c, d = symbols('a b c d')
    A = Matrix(2, 2, [a, b, c, d])
    B = Matrix(2, 2, [0, 2, -3, 0])
    C = Matrix(2, 2, [1, 2, 3, 4])

    assert solve(A*B - C, [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}
    assert solve([A*B - C], [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}
    assert solve(Eq(A*B, C), [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}

    assert solve([A*B - B*A], [a, b, c, d]) == {a: d, b: Rational(-2, 3)*c}
    assert solve([A*C - C*A], [a, b, c, d]) == {a: d - c, b: Rational(2, 3)*c}
    assert solve([A*B - B*A, A*C - C*A], [a, b, c, d]) == {a: d, b: 0, c: 0}

    assert solve([Eq(A*B, B*A)], [a, b, c, d]) == {a: d, b: Rational(-2, 3)*c}
    assert solve([Eq(A*C, C*A)], [a, b, c, d]) == {a: d - c, b: Rational(2, 3)*c}
    assert solve([Eq(A*B, B*A), Eq(A*C, C*A)], [a, b, c, d]) == {a: d, b: 0, c: 0}

    # https://github.com/sympy/sympy/issues/27854
    m, n = symbols("m n")
    A = MatrixSymbol("A", m, n)
    x = MatrixSymbol("x", n, 1)
    b = MatrixSymbol('b', m, 1)
    r = A * x - b
    f = r.T * r
    grad_f = f.diff(x)
    raises(ValueError, lambda: solve(grad_f, x))


def test_solve_linear():
    w = Wild('w')
    assert solve_linear(x, x) == (0, 1)
    assert solve_linear(x, exclude=[x]) == (0, 1)
    assert solve_linear(x, symbols=[w]) == (0, 1)
    assert solve_linear(x, y - 2*x) in [(x, y/3), (y, 3*x)]
    assert solve_linear(x, y - 2*x, exclude=[x]) == (y, 3*x)
    assert solve_linear(3*x - y, 0) in [(x, y/3), (y, 3*x)]
    assert solve_linear(3*x - y, 0, [x]) == (x, y/3)
    assert solve_linear(3*x - y, 0, [y]) == (y, 3*x)
    assert solve_linear(x**2/y, 1) == (y, x**2)
    assert solve_linear(w, x) in [(w, x), (x, w)]
    assert solve_linear(cos(x)**2 + sin(x)**2 + 2 + y) == \
        (y, -2 - cos(x)**2 - sin(x)**2)
    assert solve_linear(cos(x)**2 + sin(x)**2 + 2 + y, symbols=[x]) == (0, 1)
    assert solve_linear(Eq(x, 3)) == (x, 3)
    assert solve_linear(1/(1/x - 2)) == (0, 0)
    assert solve_linear((x + 1)*exp(-x), symbols=[x]) == (x, -1)
    assert solve_linear((x + 1)*exp(x), symbols=[x]) == ((x + 1)*exp(x), 1)
    assert solve_linear(x*exp(-x**2), symbols=[x]) == (x, 0)
    assert solve_linear(0**x - 1) == (0**x - 1, 1)
    assert solve_linear(1 + 1/(x - 1)) == (x, 0)
    eq = y*cos(x)**2 + y*sin(x)**2 - y  # = y*(1 - 1) = 0
    assert solve_linear(eq) == (0, 1)
    eq = cos(x)**2 + sin(x)**2  # = 1
    assert solve_linear(eq) == (0, 1)
    raises(ValueError, lambda: solve_linear(Eq(x, 3), 3))


def test_solve_undetermined_coeffs():
    assert solve_undetermined_coeffs(
        a*x**2 + b*x**2 + b*x + 2*c*x + c + 1, [a, b, c], x
        ) == {a: -2, b: 2, c: -1}
    # Test that rational functions work
    assert solve_undetermined_coeffs(a/x + b/(x + 1)
        - (2*x + 1)/(x**2 + x), [a, b], x) == {a: 1, b: 1}
    # Test cancellation in rational functions
    assert solve_undetermined_coeffs(
        ((c + 1)*a*x**2 + (c + 1)*b*x**2 +
        (c + 1)*b*x + (c + 1)*2*c*x + (c + 1)**2)/(c + 1),
        [a, b, c], x) == \
        {a: -2, b: 2, c: -1}
    # multivariate
    X, Y, Z = y, x**y, y*x**y
    eq = a*X + b*Y + c*Z - X - 2*Y - 3*Z
    coeffs = a, b, c
    syms = x, y
    assert solve_undetermined_coeffs(eq, coeffs) == {
        a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, syms) == {
        a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, *syms) == {
        a: 1, b: 2, c: 3}
    # check output format
    assert solve_undetermined_coeffs(a*x + a - 2, [a]) == []
    assert solve_undetermined_coeffs(a**2*x - 4*x, [a]) == [
        {a: -2}, {a: 2}]
    assert solve_undetermined_coeffs(0, [a]) == []
    assert solve_undetermined_coeffs(0, [a], dict=True) == []
    assert solve_undetermined_coeffs(0, [a], set=True) == ([], {})
    assert solve_undetermined_coeffs(1, [a]) == []
    abeq = a*x - 2*x + b - 3
    s = {b, a}
    assert solve_undetermined_coeffs(abeq, s, x) == {a: 2, b: 3}
    assert solve_undetermined_coeffs(abeq, s, x, set=True) == ([a, b], {(2, 3)})
    assert solve_undetermined_coeffs(sin(a*x) - sin(2*x), (a,)) is None
    assert solve_undetermined_coeffs(a*x + b*x - 2*x, (a, b)) == {a: 2 - b}


def test_solve_inequalities():
    x = Symbol('x')
    sol = And(S.Zero < x, x < oo)
    assert solve(x + 1 > 1) == sol
    assert solve([x + 1 > 1]) == sol
    assert solve([x + 1 > 1], x) == sol
    assert solve([x + 1 > 1], [x]) == sol

    system = [Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]
    assert solve(system) == \
        And(Or(And(Lt(-sqrt(2), x), Lt(x, -1)),
               And(Lt(1, x), Lt(x, sqrt(2)))), Eq(0, 0))

    x = Symbol('x', real=True)
    system = [Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]
    assert solve(system) == \
        Or(And(Lt(-sqrt(2), x), Lt(x, -1)), And(Lt(1, x), Lt(x, sqrt(2))))

    # issues 6627, 3448
    assert solve((x - 3)/(x - 2) < 0, x) == And(Lt(2, x), Lt(x, 3))
    assert solve(x/(x + 1) > 1, x) == And(Lt(-oo, x), Lt(x, -1))

    assert solve(sin(x) > S.Half) == And(pi/6 < x, x < pi*Rational(5, 6))

    assert solve(Eq(False, x < 1)) == (S.One <= x) & (x < oo)
    assert solve(Eq(True, x < 1)) == (-oo < x) & (x < 1)
    assert solve(Eq(x < 1, False)) == (S.One <= x) & (x < oo)
    assert solve(Eq(x < 1, True)) == (-oo < x) & (x < 1)

    assert solve(Eq(False, x)) == False
    assert solve(Eq(0, x)) == [0]
    assert solve(Eq(True, x)) == True
    assert solve(Eq(1, x)) == [1]
    assert solve(Eq(False, ~x)) == True
    assert solve(Eq(True, ~x)) == False
    assert solve(Ne(True, x)) == False
    assert solve(Ne(1, x)) == (x > -oo) & (x < oo) & Ne(x, 1)


def test_issue_4793():
    assert solve(1/x) == []
    assert solve(x*(1 - 5/x)) == [5]
    assert solve(x + sqrt(x) - 2) == [1]
    assert solve(-(1 + x)/(2 + x)**2 + 1/(2 + x)) == []
    assert solve(-x**2 - 2*x + (x + 1)**2 - 1) == []
    assert solve((x/(x + 1) + 3)**(-2)) == []
    assert solve(x/sqrt(x**2 + 1), x) == [0]
    assert solve(exp(x) - y, x) == [log(y)]
    assert solve(exp(x)) == []
    assert solve(x**2 + x + sin(y)**2 + cos(y)**2 - 1, x) in [[0, -1], [-1, 0]]
    eq = 4*3**(5*x + 2) - 7
    ans = solve(eq, x)
    assert len(ans) == 5 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)
    assert solve(log(x**2) - y**2/exp(x), x, y, set=True) == (
        [x, y],
        {(x, sqrt(exp(x) * log(x ** 2))), (x, -sqrt(exp(x) * log(x ** 2)))})
    assert solve(x**2*z**2 - z**2*y**2) == [{x: -y}, {x: y}, {z: 0}]
    assert solve((x - 1)/(1 + 1/(x - 1))) == []
    assert solve(x**(y*z) - x, x) == [1]
    raises(NotImplementedError, lambda: solve(log(x) - exp(x), x))
    raises(NotImplementedError, lambda: solve(2**x - exp(x) - 3))


def test_PR1964():
    # issue 5171
    assert solve(sqrt(x)) == solve(sqrt(x**3)) == [0]
    assert solve(sqrt(x - 1)) == [1]
    # issue 4462
    a = Symbol('a')
    assert solve(-3*a/sqrt(x), x) == []
    # issue 4486
    assert solve(2*x/(x + 2) - 1, x) == [2]
    # issue 4496
    assert set(solve((x**2/(7 - x)).diff(x))) == {S.Zero, S(14)}
    # issue 4695
    f = Function('f')
    assert solve((3 - 5*x/f(x))*f(x), f(x)) == [x*Rational(5, 3)]
    # issue 4497
    assert solve(1/root(5 + x, 5) - 9, x) == [Rational(-295244, 59049)]

    assert solve(sqrt(x) + sqrt(sqrt(x)) - 4) == [(Rational(-1, 2) + sqrt(17)/2)**4]
    assert set(solve(Poly(sqrt(exp(x)) + sqrt(exp(-x)) - 4))) in \
        [
            {log((-sqrt(3) + 2)**2), log((sqrt(3) + 2)**2)},
            {2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2)},
            {log(-4*sqrt(3) + 7), log(4*sqrt(3) + 7)},
        ]
    assert set(solve(Poly(exp(x) + exp(-x) - 4))) == \
        {log(-sqrt(3) + 2), log(sqrt(3) + 2)}
    assert set(solve(x**y + x**(2*y) - 1, x)) == \
        {(Rational(-1, 2) + sqrt(5)/2)**(1/y), (Rational(-1, 2) - sqrt(5)/2)**(1/y)}

    assert solve(exp(x/y)*exp(-z/y) - 2, y) == [(x - z)/log(2)]
    assert solve(
        x**z*y**z - 2, z) in [[log(2)/(log(x) + log(y))], [log(2)/(log(x*y))]]
    # if you do inversion too soon then multiple roots (as for the following)
    # will be missed, e.g. if exp(3*x) = exp(3) -> 3*x = 3
    E = S.Exp1
    assert solve(exp(3*x) - exp(3), x) in [
        [1, log(E*(Rational(-1, 2) - sqrt(3)*I/2)), log(E*(Rational(-1, 2) + sqrt(3)*I/2))],
        [1, log(-E/2 - sqrt(3)*E*I/2), log(-E/2 + sqrt(3)*E*I/2)],
        ]

    # coverage test
    p = Symbol('p', positive=True)
    assert solve((1/p + 1)**(p + 1)) == []


def test_issue_5197():
    x = Symbol('x', real=True)
    assert solve(x**2 + 1, x) == []
    n = Symbol('n', integer=True, positive=True)
    assert solve((n - 1)*(n + 2)*(2*n - 1), n) == [1]
    x = Symbol('x', positive=True)
    y = Symbol('y')
    assert solve([x + 5*y - 2, -3*x + 6*y - 15], x, y) == []
                 # not {x: -3, y: 1} b/c x is positive
    # The solution following should not contain (-sqrt(2), sqrt(2))
    assert solve([(x + y), 2 - y**2], x, y) == [(sqrt(2), -sqrt(2))]
    y = Symbol('y', positive=True)
    # The solution following should not contain {y: -x*exp(x/2)}
    assert solve(x**2 - y**2/exp(x), y, x, dict=True) == [{y: x*exp(x/2)}]
    x, y, z = symbols('x y z', positive=True)
    assert solve(z**2*x**2 - z**2*y**2/exp(x), y, x, z, dict=True) == [{y: x*exp(x/2)}]


def test_checking():
    assert set(
        solve(x*(x - y/x), x, check=False)) == {sqrt(y), S.Zero, -sqrt(y)}
    assert set(solve(x*(x - y/x), x, check=True)) == {sqrt(y), -sqrt(y)}
    # {x: 0, y: 4} sets denominator to 0 in the following so system should return None
    assert solve((1/(1/x + 2), 1/(y - 3) - 1)) == []
    # 0 sets denominator of 1/x to zero so None is returned
    assert solve(1/(1/x + 2)) == []


def test_issue_4671_4463_4467():
    assert solve(sqrt(x**2 - 1) - 2) in ([sqrt(5), -sqrt(5)],
                                           [-sqrt(5), sqrt(5)])
    assert solve((2**exp(y**2/x) + 2)/(x**2 + 15), y) == [
        -sqrt(x*log(1 + I*pi/log(2))), sqrt(x*log(1 + I*pi/log(2)))]

    C1, C2 = symbols('C1 C2')
    f = Function('f')
    assert solve(C1 + C2/x**2 - exp(-f(x)), f(x)) == [log(x**2/(C1*x**2 + C2))]
    a = Symbol('a')
    E = S.Exp1
    assert solve(1 - log(a + 4*x**2), x) in (
        [-sqrt(-a + E)/2, sqrt(-a + E)/2],
        [sqrt(-a + E)/2, -sqrt(-a + E)/2]
    )
    assert solve(log(a**(-3) - x**2)/a, x) in (
        [-sqrt(-1 + a**(-3)), sqrt(-1 + a**(-3))],
        [sqrt(-1 + a**(-3)), -sqrt(-1 + a**(-3))],)
    assert solve(1 - log(a + 4*x**2), x) in (
        [-sqrt(-a + E)/2, sqrt(-a + E)/2],
        [sqrt(-a + E)/2, -sqrt(-a + E)/2],)
    assert solve((a**2 + 1)*(sin(a*x) + cos(a*x)), x) == [-pi/(4*a)]
    assert solve(3 - (sinh(a*x) + cosh(a*x)), x) == [log(3)/a]
    assert set(solve(3 - (sinh(a*x) + cosh(a*x)**2), x)) == \
        {log(-2 + sqrt(5))/a, log(-sqrt(2) + 1)/a,
        log(-sqrt(5) - 2)/a, log(1 + sqrt(2))/a}
    assert solve(atan(x) - 1) == [tan(1)]


def test_issue_5132():
    r, t = symbols('r,t')
    assert set(solve([r - x**2 - y**2, tan(t) - y/x], [x, y])) == \
        {(
            -sqrt(r*cos(t)**2), -1*sqrt(r*cos(t)**2)*tan(t)),
            (sqrt(r*cos(t)**2), sqrt(r*cos(t)**2)*tan(t))}
    assert solve([exp(x) - sin(y), 1/y - 3], [x, y]) == \
        [(log(sin(Rational(1, 3))), Rational(1, 3))]
    assert solve([exp(x) - sin(y), 1/exp(y) - 3], [x, y]) == \
        [(log(-sin(log(3))), -log(3))]
    assert set(solve([exp(x) - sin(y), y**2 - 4], [x, y])) == \
        {(log(-sin(2)), -S(2)), (log(sin(2)), S(2))}
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    assert solve(eqs, set=True) == \
        ([y, z], {
        (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
        (-log(3), -sqrt(-exp(2*x) - sin(log(3))))})
    assert solve(eqs, x, z, set=True) == (
        [x, z],
        {(x, sqrt(-exp(2*x) + sin(y))), (x, -sqrt(-exp(2*x) + sin(y)))})
    assert set(solve(eqs, x, y)) == \
        {
            (log(-sqrt(-z**2 - sin(log(3)))), -log(3)),
        (log(-z**2 - sin(log(3)))/2, -log(3))}
    assert set(solve(eqs, y, z)) == \
        {
            (-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
        (-log(3), sqrt(-exp(2*x) - sin(log(3))))}
    eqs = [exp(x)**2 - sin(y) + z, 1/exp(y) - 3]
    assert solve(eqs, set=True) == ([y, z], {
        (-log(3), -exp(2*x) - sin(log(3)))})
    assert solve(eqs, x, z, set=True) == (
        [x, z], {(x, -exp(2*x) + sin(y))})
    assert set(solve(eqs, x, y)) == {
            (log(-sqrt(-z - sin(log(3)))), -log(3)),
            (log(-z - sin(log(3)))/2, -log(3))}
    assert solve(eqs, z, y) == \
        [(-exp(2*x) - sin(log(3)), -log(3))]
    assert solve((sqrt(x**2 + y**2) - sqrt(10), x + y - 4), set=True) == (
        [x, y], {(S.One, S(3)), (S(3), S.One)})
    assert set(solve((sqrt(x**2 + y**2) - sqrt(10), x + y - 4), x, y)) == \
        {(S.One, S(3)), (S(3), S.One)}


def test_issue_5335():
    lam, a0, conc = symbols('lam a0 conc')
    a = 0.005
    b = 0.743436700916726
    eqs = [lam + 2*y - a0*(1 - x/2)*x - a*x/2*x,
           a0*(1 - x/2)*x - 1*y - b*y,
           x + y - conc]
    sym = [x, y, a0]
    # there are 4 solutions obtained manually but only two are valid
    assert len(solve(eqs, sym, manual=True, minimal=True)) == 2
    assert len(solve(eqs, sym)) == 2  # cf below with rational=False


@SKIP("Hangs")
def _test_issue_5335_float():
    # gives ZeroDivisionError: polynomial division
    lam, a0, conc = symbols('lam a0 conc')
    a = 0.005
    b = 0.743436700916726
    eqs = [lam + 2*y - a0*(1 - x/2)*x - a*x/2*x,
           a0*(1 - x/2)*x - 1*y - b*y,
           x + y - conc]
    sym = [x, y, a0]
    assert len(solve(eqs, sym, rational=False)) == 2


def test_issue_5767():
    assert set(solve([x**2 + y + 4], [x])) == \
        {(-sqrt(-y - 4),), (sqrt(-y - 4),)}


def _make_example_24609():
    D, R, H, B_g, V, D_c = symbols("D, R, H, B_g, V, D_c", real=True, positive=True)
    Sigma_f, Sigma_a, nu = symbols("Sigma_f, Sigma_a, nu", real=True, positive=True)
    x = symbols("x", real=True, positive=True)
    eq = (
        2**(S(2)/3)*pi**(S(2)/3)*D_c*(S(231361)/10000 + pi**2/x**2)
        /(6*V**(S(2)/3)*x**(S(1)/3))
      - 2**(S(2)/3)*pi**(S(8)/3)*D_c/(2*V**(S(2)/3)*x**(S(7)/3))
    )
    expected = 100*sqrt(2)*pi/481
    return eq, expected, x


def test_issue_24609():
    # https://github.com/sympy/sympy/issues/24609
    eq, expected, x = _make_example_24609()
    assert solve(eq, x, simplify=True) == [expected]
    [solapprox] = solve(eq.n(), x)
    assert abs(solapprox - expected.n()) < 1e-14


@XFAIL
def test_issue_24609_xfail():
    #
    # This returns 5 solutions when it should be 1 (with x positive).
    # Simplification reveals all solutions to be equivalent. It is expected
    # that solve without simplify=True returns duplicate solutions in some
    # cases but the core of this equation is a simple quadratic that can easily
    # be solved without introducing any redundant solutions:
    #
    #     >>> print(factor_terms(eq.as_numer_denom()[0]))
    #     2**(2/3)*pi**(2/3)*D_c*V**(2/3)*x**(7/3)*(231361*x**2 - 20000*pi**2)
    #
    eq, expected, x = _make_example_24609()
    assert len(solve(eq, x)) == [expected]
    #
    # We do not want to pass this test just by using simplify so if the above
    # passes then uncomment the additional test below:
    #
    # assert len(solve(eq, x, simplify=False)) == 1


def test_polysys():
    assert set(solve([x**2 + 2/y - 2, x + y - 3], [x, y])) == \
        {(S.One, S(2)), (1 + sqrt(5), 2 - sqrt(5)),
        (1 - sqrt(5), 2 + sqrt(5))}
    assert solve([x**2 + y - 2, x**2 + y]) == []
    # the ordering should be whatever the user requested
    assert solve([x**2 + y - 3, x - y - 4], (x, y)) != solve([x**2 +
                 y - 3, x - y - 4], (y, x))


@slow
def test_unrad1():
    raises(NotImplementedError, lambda:
        unrad(sqrt(x) + sqrt(x + 1) + sqrt(1 - sqrt(x)) + 3))
    raises(NotImplementedError, lambda:
        unrad(sqrt(x) + (x + 1)**Rational(1, 3) + 2*sqrt(y)))

    s = symbols('s', cls=Dummy)

    # checkers to deal with possibility of answer coming
    # back with a sign change (cf issue 5203)
    def check(rv, ans):
        assert bool(rv[1]) == bool(ans[1])
        if ans[1]:
            return s_check(rv, ans)
        e = rv[0].expand()
        a = ans[0].expand()
        return e in [a, -a] and rv[1] == ans[1]

    def s_check(rv, ans):
        # get the dummy
        rv = list(rv)
        d = rv[0].atoms(Dummy)
        reps = list(zip(d, [s]*len(d)))
        # replace s with this dummy
        rv = (rv[0].subs(reps).expand(), [rv[1][0].subs(reps), rv[1][1].subs(reps)])
        ans = (ans[0].subs(reps).expand(), [ans[1][0].subs(reps), ans[1][1].subs(reps)])
        return str(rv[0]) in [str(ans[0]), str(-ans[0])] and \
            str(rv[1]) == str(ans[1])

    assert unrad(1) is None
    assert check(unrad(sqrt(x)),
        (x, []))
    assert check(unrad(sqrt(x) + 1),
        (x - 1, []))
    assert check(unrad(sqrt(x) + root(x, 3) + 2),
        (s**3 + s**2 + 2, [s, s**6 - x]))
    assert check(unrad(sqrt(x)*root(x, 3) + 2),
        (x**5 - 64, []))
    assert check(unrad(sqrt(x) + (x + 1)**Rational(1, 3)),
        (x**3 - (x + 1)**2, []))
    assert check(unrad(sqrt(x) + sqrt(x + 1) + sqrt(2*x)),
        (-2*sqrt(2)*x - 2*x + 1, []))
    assert check(unrad(sqrt(x) + sqrt(x + 1) + 2),
        (16*x - 9, []))
    assert check(unrad(sqrt(x) + sqrt(x + 1) + sqrt(1 - x)),
        (5*x**2 - 4*x, []))
    assert check(unrad(a*sqrt(x) + b*sqrt(x) + c*sqrt(y) + d*sqrt(y)),
        ((a*sqrt(x) + b*sqrt(x))**2 - (c*sqrt(y) + d*sqrt(y))**2, []))
    assert check(unrad(sqrt(x) + sqrt(1 - x)),
        (2*x - 1, []))
    assert check(unrad(sqrt(x) + sqrt(1 - x) - 3),
        (x**2 - x + 16, []))
    assert check(unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x)),
        (5*x**2 - 2*x + 1, []))
    assert unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - 3) in [
        (25*x**4 + 376*x**3 + 1256*x**2 - 2272*x + 784, []),
        (25*x**8 - 476*x**6 + 2534*x**4 - 1468*x**2 + 169, [])]
    assert unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - sqrt(1 - 2*x)) == \
        (41*x**4 + 40*x**3 + 232*x**2 - 160*x + 16, [])  # orig root at 0.487
    assert check(unrad(sqrt(x) + sqrt(x + 1)), (S.One, []))

    eq = sqrt(x) + sqrt(x + 1) + sqrt(1 - sqrt(x))
    assert check(unrad(eq),
        (16*x**2 - 9*x, []))
    assert set(solve(eq, check=False)) == {S.Zero, Rational(9, 16)}
    assert solve(eq) == []
    # but this one really does have those solutions
    assert set(solve(sqrt(x) - sqrt(x + 1) + sqrt(1 - sqrt(x)))) == \
        {S.Zero, Rational(9, 16)}

    assert check(unrad(sqrt(x) + root(x + 1, 3) + 2*sqrt(y), y),
        (S('2*sqrt(x)*(x + 1)**(1/3) + x - 4*y + (x + 1)**(2/3)'), []))
    assert check(unrad(sqrt(x/(1 - x)) + (x + 1)**Rational(1, 3)),
        (x**5 - x**4 - x**3 + 2*x**2 + x - 1, []))
    assert check(unrad(sqrt(x/(1 - x)) + 2*sqrt(y), y),
        (4*x*y + x - 4*y, []))
    assert check(unrad(sqrt(x)*sqrt(1 - x) + 2, x),
        (x**2 - x + 4, []))

    # http://tutorial.math.lamar.edu/
    #        Classes/Alg/SolveRadicalEqns.aspx#Solve_Rad_Ex2_a
    assert solve(Eq(x, sqrt(x + 6))) == [3]
    assert solve(Eq(x + sqrt(x - 4), 4)) == [4]
    assert solve(Eq(1, x + sqrt(2*x - 3))) == []
    assert set(solve(Eq(sqrt(5*x + 6) - 2, x))) == {-S.One, S(2)}
    assert set(solve(Eq(sqrt(2*x - 1) - sqrt(x - 4), 2))) == {S(5), S(13)}
    assert solve(Eq(sqrt(x + 7) + 2, sqrt(3 - x))) == [-6]
    # http://www.purplemath.com/modules/solverad.htm
    assert solve((2*x - 5)**Rational(1, 3) - 3) == [16]
    assert set(solve(x + 1 - root(x**4 + 4*x**3 - x, 4))) == \
        {Rational(-1, 2), Rational(-1, 3)}
    assert set(solve(sqrt(2*x**2 - 7) - (3 - x))) == {-S(8), S(2)}
    assert solve(sqrt(2*x + 9) - sqrt(x + 1) - sqrt(x + 4)) == [0]
    assert solve(sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)) == [5]
    assert solve(sqrt(x)*sqrt(x - 7) - 12) == [16]
    assert solve(sqrt(x - 3) + sqrt(x) - 3) == [4]
    assert solve(sqrt(9*x**2 + 4) - (3*x + 2)) == [0]
    assert solve(sqrt(x) - 2 - 5) == [49]
    assert solve(sqrt(x - 3) - sqrt(x) - 3) == []
    assert solve(sqrt(x - 1) - x + 7) == [10]
    assert solve(sqrt(x - 2) - 5) == [27]
    assert solve(sqrt(17*x - sqrt(x**2 - 5)) - 7) == [3]
    assert solve(sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))) == []

    # don't posify the expression in unrad and do use _mexpand
    z = sqrt(2*x + 1)/sqrt(x) - sqrt(2 + 1/x)
    p = posify(z)[0]
    assert solve(p) == []
    assert solve(z) == []
    assert solve(z + 6*I) == [Rational(-1, 11)]
    assert solve(p + 6*I) == []
    # issue 8622
    assert unrad(root(x + 1, 5) - root(x, 3)) == (
        -(x**5 - x**3 - 3*x**2 - 3*x - 1), [])
    # issue #8679
    assert check(unrad(x + root(x, 3) + root(x, 3)**2 + sqrt(y), x),
        (s**3 + s**2 + s + sqrt(y), [s, s**3 - x]))

    # for coverage
    assert check(unrad(sqrt(x) + root(x, 3) + y),
        (s**3 + s**2 + y, [s, s**6 - x]))
    assert solve(sqrt(x) + root(x, 3) - 2) == [1]
    raises(NotImplementedError, lambda:
        solve(sqrt(x) + root(x, 3) + root(x + 1, 5) - 2))
    # fails through a different code path
    raises(NotImplementedError, lambda: solve(-sqrt(2) + cosh(x)/x))
    # unrad some
    assert solve(sqrt(x + root(x, 3))+root(x - y, 5), y) == [
        x + (x**Rational(1, 3) + x)**Rational(5, 2)]
    assert check(unrad(sqrt(x) - root(x + 1, 3)*sqrt(x + 2) + 2),
        (s**10 + 8*s**8 + 24*s**6 - 12*s**5 - 22*s**4 - 160*s**3 - 212*s**2 -
        192*s - 56, [s, s**2 - x]))
    e = root(x + 1, 3) + root(x, 3)
    assert unrad(e) == (2*x + 1, [])
    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    assert check(unrad(eq),
        (15625*x**4 + 173000*x**3 + 355600*x**2 - 817920*x + 331776, []))
    assert check(unrad(root(x, 4) + root(x, 4)**3 - 1),
        (s**3 + s - 1, [s, s**4 - x]))
    assert check(unrad(root(x, 2) + root(x, 2)**3 - 1),
        (x**3 + 2*x**2 + x - 1, []))
    assert unrad(x**0.5) is None
    assert check(unrad(t + root(x + y, 5) + root(x + y, 5)**3),
        (s**3 + s + t, [s, s**5 - x - y]))
    assert check(unrad(x + root(x + y, 5) + root(x + y, 5)**3, y),
        (s**3 + s + x, [s, s**5 - x - y]))
    assert check(unrad(x + root(x + y, 5) + root(x + y, 5)**3, x),
        (s**5 + s**3 + s - y, [s, s**5 - x - y]))
    assert check(unrad(root(x - 1, 3) + root(x + 1, 5) + root(2, 5)),
        (s**5 + 5*2**Rational(1, 5)*s**4 + s**3 + 10*2**Rational(2, 5)*s**3 +
        10*2**Rational(3, 5)*s**2 + 5*2**Rational(4, 5)*s + 4, [s, s**3 - x + 1]))
    raises(NotImplementedError, lambda:
        unrad((root(x, 2) + root(x, 3) + root(x, 4)).subs(x, x**5 - x + 1)))

    # the simplify flag should be reset to False for unrad results;
    # if it's not then this next test will take a long time
    assert solve(root(x, 3) + root(x, 5) - 2) == [1]
    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    assert check(unrad(eq),
        ((5*x - 4)*(3125*x**3 + 37100*x**2 + 100800*x - 82944), []))
    ans = S('''
        [4/5, -1484/375 + 172564/(140625*(114*sqrt(12657)/78125 +
        12459439/52734375)**(1/3)) +
        4*(114*sqrt(12657)/78125 + 12459439/52734375)**(1/3)]''')
    assert solve(eq) == ans
    # duplicate radical handling
    assert check(unrad(sqrt(x + root(x + 1, 3)) - root(x + 1, 3) - 2),
        (s**3 - s**2 - 3*s - 5, [s, s**3 - x - 1]))
    # cov post-processing
    e = root(x**2 + 1, 3) - root(x**2 - 1, 5) - 2
    assert check(unrad(e),
        (s**5 - 10*s**4 + 39*s**3 - 80*s**2 + 80*s - 30,
        [s, s**3 - x**2 - 1]))

    e = sqrt(x + root(x + 1, 2)) - root(x + 1, 3) - 2
    assert check(unrad(e),
        (s**6 - 2*s**5 - 7*s**4 - 3*s**3 + 26*s**2 + 40*s + 25,
        [s, s**3 - x - 1]))
    assert check(unrad(e, _reverse=True),
        (s**6 - 14*s**5 + 73*s**4 - 187*s**3 + 276*s**2 - 228*s + 89,
        [s, s**2 - x - sqrt(x + 1)]))
    # this one needs r0, r1 reversal to work
    assert check(unrad(sqrt(x + sqrt(root(x, 3) - 1)) - root(x, 6) - 2),
        (s**12 - 2*s**8 - 8*s**7 - 8*s**6 + s**4 + 8*s**3 + 23*s**2 +
        32*s + 17, [s, s**6 - x]))

    # why does this pass
    assert unrad(root(cosh(x), 3)/x*root(x + 1, 5) - 1) == (
        -(x**15 - x**3*cosh(x)**5 - 3*x**2*cosh(x)**5 - 3*x*cosh(x)**5
        - cosh(x)**5), [])
    # and this fail?
    #assert unrad(sqrt(cosh(x)/x) + root(x + 1, 3)*sqrt(x) - 1) == (
    #    -s**6 + 6*s**5 - 15*s**4 + 20*s**3 - 15*s**2 + 6*s + x**5 +
    #    2*x**4 + x**3 - 1, [s, s**2 - cosh(x)/x])

    # watch for symbols in exponents
    assert unrad(S('(x+y)**(2*y/3) + (x+y)**(1/3) + 1')) is None
    assert check(unrad(S('(x+y)**(2*y/3) + (x+y)**(1/3) + 1'), x),
        (s**(2*y) + s + 1, [s, s**3 - x - y]))
    # should _Q be so lenient?
    assert unrad(x**(S.Half/y) + y, x) == (x**(1/y) - y**2, [])

    # This tests two things: that if full unrad is attempted and fails
    # the solution should still be found; also it tests that the use of
    # composite
    assert len(solve(sqrt(y)*x + x**3 - 1, x)) == 3
    assert len(solve(-512*y**3 + 1344*(x + 2)**Rational(1, 3)*y**2 -
        1176*(x + 2)**Rational(2, 3)*y - 169*x + 686, y, _unrad=False)) == 3

    # watch out for when the cov doesn't involve the symbol of interest
    eq = S('-x + (7*y/8 - (27*x/2 + 27*sqrt(x**2)/2)**(1/3)/3)**3 - 1')
    assert solve(eq, y) == [
        2**(S(2)/3)*(27*x + 27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 +
        S(512)/343)**(S(1)/3)*(-S(1)/2 - sqrt(3)*I/2), 2**(S(2)/3)*(27*x +
        27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 +
        S(512)/343)**(S(1)/3)*(-S(1)/2 + sqrt(3)*I/2), 2**(S(2)/3)*(27*x +
        27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 + S(512)/343)**(S(1)/3)]

    eq = root(x + 1, 3) - (root(x, 3) + root(x, 5))
    assert check(unrad(eq),
        (3*s**13 + 3*s**11 + s**9 - 1, [s, s**15 - x]))
    assert check(unrad(eq - 2),
        (3*s**13 + 3*s**11 + 6*s**10 + s**9 + 12*s**8 + 6*s**6 + 12*s**5 +
        12*s**3 + 7, [s, s**15 - x]))
    assert check(unrad(root(x, 3) - root(x + 1, 4)/2 + root(x + 2, 3)),
        (s*(4096*s**9 + 960*s**8 + 48*s**7 - s**6 - 1728),
        [s, s**4 - x - 1]))  # orig expr has two real roots: -1, -.389
    assert check(unrad(root(x, 3) + root(x + 1, 4) - root(x + 2, 3)/2),
        (343*s**13 + 2904*s**12 + 1344*s**11 + 512*s**10 - 1323*s**9 -
        3024*s**8 - 1728*s**7 + 1701*s**5 + 216*s**4 - 729*s, [s, s**4 - x -
        1]))  # orig expr has one real root: -0.048
    assert check(unrad(root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3)),
        (729*s**13 - 216*s**12 + 1728*s**11 - 512*s**10 + 1701*s**9 -
        3024*s**8 + 1344*s**7 + 1323*s**5 - 2904*s**4 + 343*s, [s, s**4 - x -
        1]))  # orig expr has 2 real roots: -0.91, -0.15
    assert check(unrad(root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3) - 2),
        (729*s**13 + 1242*s**12 + 18496*s**10 + 129701*s**9 + 388602*s**8 +
        453312*s**7 - 612864*s**6 - 3337173*s**5 - 6332418*s**4 - 7134912*s**3
        - 5064768*s**2 - 2111913*s - 398034, [s, s**4 - x - 1]))
        # orig expr has 1 real root: 19.53

    ans = solve(sqrt(x) + sqrt(x + 1) -
                sqrt(1 - x) - sqrt(2 + x))
    assert len(ans) == 1 and NS(ans[0])[:4] == '0.73'
    # the fence optimization problem
    # https://github.com/sympy/sympy/issues/4793#issuecomment-36994519
    F = Symbol('F')
    eq = F - (2*x + 2*y + sqrt(x**2 + y**2))
    ans = F*Rational(2, 7) - sqrt(2)*F/14
    X = solve(eq, x, check=False)
    for xi in reversed(X):  # reverse since currently, ans is the 2nd one
        Y = solve((x*y).subs(x, xi).diff(y), y, simplify=False, check=False)
        if any((a - ans).expand().is_zero for a in Y):
            break
    else:
        assert None  # no answer was found
    assert solve(sqrt(x + 1) + root(x, 3) - 2) == S('''
        [(-11/(9*(47/54 + sqrt(93)/6)**(1/3)) + 1/3 + (47/54 +
        sqrt(93)/6)**(1/3))**3]''')
    assert solve(sqrt(sqrt(x + 1)) + x**Rational(1, 3) - 2) == S('''
        [(-sqrt(-2*(-1/16 + sqrt(6913)/16)**(1/3) + 6/(-1/16 +
        sqrt(6913)/16)**(1/3) + 17/2 + 121/(4*sqrt(-6/(-1/16 +
        sqrt(6913)/16)**(1/3) + 2*(-1/16 + sqrt(6913)/16)**(1/3) + 17/4)))/2 +
        sqrt(-6/(-1/16 + sqrt(6913)/16)**(1/3) + 2*(-1/16 +
        sqrt(6913)/16)**(1/3) + 17/4)/2 + 9/4)**3]''')
    assert solve(sqrt(x) + root(sqrt(x) + 1, 3) - 2) == S('''
        [(-(81/2 + 3*sqrt(741)/2)**(1/3)/3 + (81/2 + 3*sqrt(741)/2)**(-1/3) +
        2)**2]''')
    eq = S('''
        -x + (1/2 - sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3
        + x*(3*x**2 - 34) + 90)**2/4 - 39304/27) - 45)**(1/3) + 34/(3*(1/2 -
        sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3 + x*(3*x**2
        - 34) + 90)**2/4 - 39304/27) - 45)**(1/3))''')
    assert check(unrad(eq),
        (s*-(-s**6 + sqrt(3)*s**6*I - 153*2**Rational(2, 3)*3**Rational(1, 3)*s**4 +
        51*12**Rational(1, 3)*s**4 - 102*2**Rational(2, 3)*3**Rational(5, 6)*s**4*I - 1620*s**3 +
        1620*sqrt(3)*s**3*I + 13872*18**Rational(1, 3)*s**2 - 471648 +
        471648*sqrt(3)*I), [s, s**3 - 306*x - sqrt(3)*sqrt(31212*x**2 -
        165240*x + 61484) + 810]))

    assert solve(eq) == [] # not other code errors
    eq = root(x, 3) - root(y, 3) + root(x, 5)
    assert check(unrad(eq),
           (s**15 + 3*s**13 + 3*s**11 + s**9 - y, [s, s**15 - x]))
    eq = root(x, 3) + root(y, 3) + root(x*y, 4)
    assert check(unrad(eq),
                 (s*y*(-s**12 - 3*s**11*y - 3*s**10*y**2 - s**9*y**3 -
                       3*s**8*y**2 + 21*s**7*y**3 - 3*s**6*y**4 - 3*s**4*y**4 -
                       3*s**3*y**5 - y**6), [s, s**4 - x*y]))
    raises(NotImplementedError,
           lambda: unrad(root(x, 3) + root(y, 3) + root(x*y, 5)))

    # Test unrad with an Equality
    eq = Eq(-x**(S(1)/5) + x**(S(1)/3), -3**(S(1)/3) - (-1)**(S(3)/5)*3**(S(1)/5))
    assert check(unrad(eq),
        (-s**5 + s**3 - 3**(S(1)/3) - (-1)**(S(3)/5)*3**(S(1)/5), [s, s**15 - x]))

    # make sure buried radicals are exposed
    s = sqrt(x) - 1
    assert unrad(s**2 - s**3) == (x**3 - 6*x**2 + 9*x - 4, [])
    # make sure numerators which are already polynomial are rejected
    assert unrad((x/(x + 1) + 3)**(-2), x) is None

    # https://github.com/sympy/sympy/issues/23707
    eq = sqrt(x - y)*exp(t*sqrt(x - y)) - exp(t*sqrt(x - y))
    assert solve(eq, y) == [x - 1]
    assert unrad(eq) is None


@slow
def test_unrad_slow():
    # this has roots with multiplicity > 1; there should be no
    # repeats in roots obtained, however
    eq = (sqrt(1 + sqrt(1 - 4*x**2)) - x*(1 + sqrt(1 + 2*sqrt(1 - 4*x**2))))
    assert solve(eq) == [S.Half]


@XFAIL
def test_unrad_fail():
    # this only works if we check real_root(eq.subs(x, Rational(1, 3)))
    # but checksol doesn't work like that
    assert solve(root(x**3 - 3*x**2, 3) + 1 - x) == [Rational(1, 3)]
    assert solve(root(x + 1, 3) + root(x**2 - 2, 5) + 1) == [
        -1, -1 + CRootOf(x**5 + x**4 + 5*x**3 + 8*x**2 + 10*x + 5, 0)**3]


def test_checksol():
    x, y, r, t = symbols('x, y, r, t')
    eq = r - x**2 - y**2
    dict_var_soln = {y: - sqrt(r) / sqrt(tan(t)**2 + 1),
        x: -sqrt(r)*tan(t)/sqrt(tan(t)**2 + 1)}
    assert checksol(eq, dict_var_soln) == True
    assert checksol(Eq(x, False), {x: False}) is True
    assert checksol(Ne(x, False), {x: False}) is False
    assert checksol(Eq(x < 1, True), {x: 0}) is True
    assert checksol(Eq(x < 1, True), {x: 1}) is False
    assert checksol(Eq(x < 1, False), {x: 1}) is True
    assert checksol(Eq(x < 1, False), {x: 0}) is False
    assert checksol(Eq(x + 1, x**2 + 1), {x: 1}) is True
    assert checksol([x - 1, x**2 - 1], x, 1) is True
    assert checksol([x - 1, x**2 - 2], x, 1) is False
    assert checksol(Poly(x**2 - 1), x, 1) is True
    assert checksol(0, {}) is True
    assert checksol([1e-10, x - 2], x, 2) is False
    assert checksol([0.5, 0, x], x, 0) is False
    assert checksol(y, x, 2) is False
    assert checksol(x+1e-10, x, 0, numerical=True) is True
    assert checksol(x+1e-10, x, 0, numerical=False) is False
    assert checksol(exp(92*x), {x: log(sqrt(2)/2)}) is False
    assert checksol(exp(92*x), {x: log(sqrt(2)/2) + I*pi}) is False
    assert checksol(1/x**5, x, 1000) is False
    raises(ValueError, lambda: checksol(x, 1))
    raises(ValueError, lambda: checksol([], x, 1))


def test__invert():
    assert _invert(x - 2) == (2, x)
    assert _invert(2) == (2, 0)
    assert _invert(exp(1/x) - 3, x) == (1/log(3), x)
    assert _invert(exp(1/x + a/x) - 3, x) == ((a + 1)/log(3), x)
    assert _invert(a, x) == (a, 0)


def test_issue_4463():
    assert solve(-a*x + 2*x*log(x), x) == [exp(a/2)]
    assert solve(x**x) == []
    assert solve(x**x - 2) == [exp(LambertW(log(2)))]
    assert solve(((x - 3)*(x - 2))**((x - 3)*(x - 4))) == [2]

@slow
def test_issue_5114_solvers():
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = symbols('a:r')

    # there is no 'a' in the equation set but this is how the
    # problem was originally posed
    syms = a, b, c, f, h, k, n
    eqs = [b + r/d - c/d,
    c*(1/d + 1/e + 1/g) - f/g - r/d,
        f*(1/g + 1/i + 1/j) - c/g - h/i,
        h*(1/i + 1/l + 1/m) - f/i - k/m,
        k*(1/m + 1/o + 1/p) - h/m - n/p,
        n*(1/p + 1/q) - k/p]
    assert len(solve(eqs, syms, manual=True, check=False, simplify=False)) == 1


def test_issue_5849():
    #
    # XXX: This system does not have a solution for most values of the
    # parameters. Generally solve returns the empty set for systems that are
    # generically inconsistent.
    #
    I1, I2, I3, I4, I5, I6 = symbols('I1:7')
    dI1, dI4, dQ2, dQ4, Q2, Q4 = symbols('dI1,dI4,dQ2,dQ4,Q2,Q4')

    e = (
        I1 - I2 - I3,
        I3 - I4 - I5,
        I4 + I5 - I6,
        -I1 + I2 + I6,
        -2*I1 - 2*I3 - 2*I5 - 3*I6 - dI1/2 + 12,
        -I4 + dQ4,
        -I2 + dQ2,
        2*I3 + 2*I5 + 3*I6 - Q2,
        I4 - 2*I5 + 2*Q4 + dI4
    )

    ans = [{
    I1: I2 + I3,
    dI1: -4*I2 - 8*I3 - 4*I5 - 6*I6 + 24,
    I4: I3 - I5,
    dQ4: I3 - I5,
    Q4: -I3/2 + 3*I5/2 - dI4/2,
    dQ2: I2,
    Q2: 2*I3 + 2*I5 + 3*I6}]

    v = I1, I4, Q2, Q4, dI1, dI4, dQ2, dQ4
    assert solve(e, *v, manual=True, check=False, dict=True) == ans
    assert solve(e, *v, manual=True, check=False) == [
        tuple([a.get(i, i) for i in v]) for a in ans]
    assert solve(e, *v, manual=True) == []
    assert solve(e, *v) == []

    # the matrix solver (tested below) doesn't like this because it produces
    # a zero row in the matrix. Is this related to issue 4551?
    assert [ei.subs(
        ans[0]) for ei in e] == [0, 0, I3 - I6, -I3 + I6, 0, 0, 0, 0, 0]


def test_issue_5849_matrix():
    '''Same as test_issue_5849 but solved with the matrix solver.

    A solution only exists if I3 == I6 which is not generically true,
    but `solve` does not return conditions under which the solution is
    valid, only a solution that is canonical and consistent with the input.
    '''
    # a simple example with the same issue
    # assert solve([x+y+z, x+y], [x, y]) == {x: y}
    # the longer example
    I1, I2, I3, I4, I5, I6 = symbols('I1:7')
    dI1, dI4, dQ2, dQ4, Q2, Q4 = symbols('dI1,dI4,dQ2,dQ4,Q2,Q4')

    e = (
        I1 - I2 - I3,
        I3 - I4 - I5,
        I4 + I5 - I6,
        -I1 + I2 + I6,
        -2*I1 - 2*I3 - 2*I5 - 3*I6 - dI1/2 + 12,
        -I4 + dQ4,
        -I2 + dQ2,
        2*I3 + 2*I5 + 3*I6 - Q2,
        I4 - 2*I5 + 2*Q4 + dI4
    )
    assert solve(e, I1, I4, Q2, Q4, dI1, dI4, dQ2, dQ4) == []


def test_issue_21882():

    a, b, c, d, f, g, k = unknowns = symbols('a, b, c, d, f, g, k')

    equations = [
        -k*a + b + 5*f/6 + 2*c/9 + 5*d/6 + 4*a/3,
        -k*f + 4*f/3 + d/2,
        -k*d + f/6 + d,
        13*b/18 + 13*c/18 + 13*a/18,
        -k*c + b/2 + 20*c/9 + a,
        -k*b + b + c/18 + a/6,
        5*b/3 + c/3 + a,
        2*b/3 + 2*c + 4*a/3,
        -g,
    ]

    answer = [
        {a: 0, f: 0, b: 0, d: 0, c: 0, g: 0},
        {a: 0, f: -d, b: 0, k: S(5)/6, c: 0, g: 0},
        {a: -2*c, f: 0, b: c, d: 0, k: S(13)/18, g: 0}]
    # but not {a: 0, f: 0, b: 0, k: S(3)/2, c: 0, d: 0, g: 0}
    # since this is already covered by the first solution
    got = solve(equations, unknowns, dict=True)
    assert got == answer, (got,answer)


def test_issue_5901():
    f, g, h = map(Function, 'fgh')
    a = Symbol('a')
    D = Derivative(f(x), x)
    G = Derivative(g(a), a)
    assert solve(f(x) + f(x).diff(x), f(x)) == \
        [-D]
    assert solve(f(x) - 3, f(x)) == \
        [3]
    assert solve(f(x) - 3*f(x).diff(x), f(x)) == \
        [3*D]
    assert solve([f(x) - 3*f(x).diff(x)], f(x)) == \
        {f(x): 3*D}
    assert solve([f(x) - 3*f(x).diff(x), f(x)**2 - y + 4], f(x), y) == \
        [(3*D, 9*D**2 + 4)]
    assert solve(-f(a)**2*g(a)**2 + f(a)**2*h(a)**2 + g(a).diff(a),
                h(a), g(a), set=True) == \
        ([h(a), g(a)], {
        (-sqrt(f(a)**2*g(a)**2 - G)/f(a), g(a)),
        (sqrt(f(a)**2*g(a)**2 - G)/f(a), g(a))}), solve(-f(a)**2*g(a)**2 + f(a)**2*h(a)**2 + g(a).diff(a),
                h(a), g(a), set=True)
    args = [[f(x).diff(x, 2)*(f(x) + g(x)), 2 - g(x)**2], f(x), g(x)]
    assert solve(*args, set=True)[1] == \
        {(-sqrt(2), sqrt(2)), (sqrt(2), -sqrt(2))}
    eqs = [f(x)**2 + g(x) - 2*f(x).diff(x), g(x)**2 - 4]
    assert solve(eqs, f(x), g(x), set=True) == \
        ([f(x), g(x)], {
        (-sqrt(2*D - 2), S(2)),
        (sqrt(2*D - 2), S(2)),
        (-sqrt(2*D + 2), -S(2)),
        (sqrt(2*D + 2), -S(2))})

    # the underlying problem was in solve_linear that was not masking off
    # anything but a Mul or Add; it now raises an error if it gets anything
    # but a symbol and solve handles the substitutions necessary so solve_linear
    # won't make this error
    raises(
        ValueError, lambda: solve_linear(f(x) + f(x).diff(x), symbols=[f(x)]))
    assert solve_linear(f(x) + f(x).diff(x), symbols=[x]) == \
        (f(x) + Derivative(f(x), x), 1)
    assert solve_linear(f(x) + Integral(x, (x, y)), symbols=[x]) == \
        (f(x) + Integral(x, (x, y)), 1)
    assert solve_linear(f(x) + Integral(x, (x, y)) + x, symbols=[x]) == \
        (x + f(x) + Integral(x, (x, y)), 1)
    assert solve_linear(f(y) + Integral(x, (x, y)) + x, symbols=[x]) == \
        (x, -f(y) - Integral(x, (x, y)))
    assert solve_linear(x - f(x)/a + (f(x) - 1)/a, symbols=[x]) == \
        (x, 1/a)
    assert solve_linear(x + Derivative(2*x, x)) == \
        (x, -2)
    assert solve_linear(x + Integral(x, y), symbols=[x]) == \
        (x, 0)
    assert solve_linear(x + Integral(x, y) - 2, symbols=[x]) == \
        (x, 2/(y + 1))

    assert set(solve(x + exp(x)**2, exp(x))) == \
        {-sqrt(-x), sqrt(-x)}
    assert solve(x + exp(x), x, implicit=True) == \
        [-exp(x)]
    assert solve(cos(x) - sin(x), x, implicit=True) == []
    assert solve(x - sin(x), x, implicit=True) == \
        [sin(x)]
    assert solve(x**2 + x - 3, x, implicit=True) == \
        [-x**2 + 3]
    assert solve(x**2 + x - 3, x**2, implicit=True) == \
        [-x + 3]


def test_issue_5912():
    assert set(solve(x**2 - x - 0.1, rational=True)) == \
        {S.Half + sqrt(35)/10, -sqrt(35)/10 + S.Half}
    ans = solve(x**2 - x - 0.1, rational=False)
    assert len(ans) == 2 and all(a.is_Number for a in ans)
    ans = solve(x**2 - x - 0.1)
    assert len(ans) == 2 and all(a.is_Number for a in ans)


def test_float_handling():
    def test(e1, e2):
        return len(e1.atoms(Float)) == len(e2.atoms(Float))
    assert solve(x - 0.5, rational=True)[0].is_Rational
    assert solve(x - 0.5, rational=False)[0].is_Float
    assert solve(x - S.Half, rational=False)[0].is_Rational
    assert solve(x - 0.5, rational=None)[0].is_Float
    assert solve(x - S.Half, rational=None)[0].is_Rational
    assert test(nfloat(1 + 2*x), 1.0 + 2.0*x)
    for contain in [list, tuple, set]:
        ans = nfloat(contain([1 + 2*x]))
        assert type(ans) is contain and test(list(ans)[0], 1.0 + 2.0*x)
    k, v = list(nfloat({2*x: [1 + 2*x]}).items())[0]
    assert test(k, 2*x) and test(v[0], 1.0 + 2.0*x)
    assert test(nfloat(cos(2*x)), cos(2.0*x))
    assert test(nfloat(3*x**2), 3.0*x**2)
    assert test(nfloat(3*x**2, exponent=True), 3.0*x**2.0)
    assert test(nfloat(exp(2*x)), exp(2.0*x))
    assert test(nfloat(x/3), x/3.0)
    assert test(nfloat(x**4 + 2*x + cos(Rational(1, 3)) + 1),
            x**4 + 2.0*x + 1.94495694631474)
    # don't call nfloat if there is no solution
    tot = 100 + c + z + t
    assert solve(((.7 + c)/tot - .6, (.2 + z)/tot - .3, t/tot - .1)) == []


def test_check_assumptions():
    x = symbols('x', positive=True)
    assert solve(x**2 - 1) == [1]


def test_issue_6056():
    assert solve(tanh(x + 3)*tanh(x - 3) - 1) == []
    assert solve(tanh(x - 1)*tanh(x + 1) + 1) == \
            [I*pi*Rational(-3, 4), -I*pi/4, I*pi/4, I*pi*Rational(3, 4)]
    assert solve((tanh(x + 3)*tanh(x - 3) + 1)**2) == \
            [I*pi*Rational(-3, 4), -I*pi/4, I*pi/4, I*pi*Rational(3, 4)]


def test_issue_5673():
    eq = -x + exp(exp(LambertW(log(x)))*LambertW(log(x)))
    assert checksol(eq, x, 2) is True
    assert checksol(eq, x, 2, numerical=False) is None


def test_exclude():
    R, C, Ri, Vout, V1, Vminus, Vplus, s = \
        symbols('R, C, Ri, Vout, V1, Vminus, Vplus, s')
    Rf = symbols('Rf', positive=True)  # to eliminate Rf = 0 soln
    eqs = [C*V1*s + Vplus*(-2*C*s - 1/R),
           Vminus*(-1/Ri - 1/Rf) + Vout/Rf,
           C*Vplus*s + V1*(-C*s - 1/R) + Vout/R,
           -Vminus + Vplus]
    assert solve(eqs, exclude=s*C*R) == [
        {
            Rf: Ri*(C*R*s + 1)**2/(C*R*s),
            Vminus: Vplus,
            V1: 2*Vplus + Vplus/(C*R*s),
            Vout: C*R*Vplus*s + 3*Vplus + Vplus/(C*R*s)},
        {
            Vplus: 0,
            Vminus: 0,
            V1: 0,
            Vout: 0},
    ]

    # TODO: Investigate why currently solution [0] is preferred over [1].
    assert solve(eqs, exclude=[Vplus, s, C]) in [[{
        Vminus: Vplus,
        V1: Vout/2 + Vplus/2 + sqrt((Vout - 5*Vplus)*(Vout - Vplus))/2,
        R: (Vout - 3*Vplus - sqrt(Vout**2 - 6*Vout*Vplus + 5*Vplus**2))/(2*C*Vplus*s),
        Rf: Ri*(Vout - Vplus)/Vplus,
    }, {
        Vminus: Vplus,
        V1: Vout/2 + Vplus/2 - sqrt((Vout - 5*Vplus)*(Vout - Vplus))/2,
        R: (Vout - 3*Vplus + sqrt(Vout**2 - 6*Vout*Vplus + 5*Vplus**2))/(2*C*Vplus*s),
        Rf: Ri*(Vout - Vplus)/Vplus,
    }], [{
        Vminus: Vplus,
        Vout: (V1**2 - V1*Vplus - Vplus**2)/(V1 - 2*Vplus),
        Rf: Ri*(V1 - Vplus)**2/(Vplus*(V1 - 2*Vplus)),
        R: Vplus/(C*s*(V1 - 2*Vplus)),
    }]]


def test_high_order_roots():
    s = x**5 + 4*x**3 + 3*x**2 + Rational(7, 4)
    assert set(solve(s)) == set(Poly(s*4, domain='ZZ').all_roots())


def test_minsolve_linear_system():
    pqt = {"quick": True, "particular": True}
    pqf = {"quick": False, "particular": True}
    assert solve([x + y - 5, 2*x - y - 1], **pqt) == {x: 2, y: 3}
    assert solve([x + y - 5, 2*x - y - 1], **pqf) == {x: 2, y: 3}
    def count(dic):
        return len([x for x in dic.values() if x == 0])
    assert count(solve([x + y + z, y + z + a + t], **pqt)) == 3
    assert count(solve([x + y + z, y + z + a + t], **pqf)) == 3
    assert count(solve([x + y + z, y + z + a], **pqt)) == 1
    assert count(solve([x + y + z, y + z + a], **pqf)) == 2
    # issue 22718
    A = Matrix([
        [ 1,  1,  1,  0,  1,  1,  0,  1,  0,  0,  1,  1,  1,  0],
        [ 1,  1,  0,  1,  1,  0,  1,  0,  1,  0, -1, -1,  0,  0],
        [-1, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1,  1,  0,  1],
        [ 1,  0,  1,  1,  0,  1,  1,  0,  0,  1, -1,  0, -1,  0],
        [-1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  1,  0,  1,  1],
        [-1,  0,  0, -1,  0,  0, -1,  0,  0,  0, -1,  0,  0, -1],
        [ 0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0, -1, -1,  0],
        [ 0, -1, -1,  0,  0,  0,  0, -1,  0,  0,  0,  1,  1,  1],
        [ 0, -1,  0, -1,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1],
        [ 0,  0, -1, -1,  0,  0,  0,  0,  0, -1,  0,  0, -1, -1],
        [ 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -1, -1,  0, -1,  0,  0,  0,  0,  0,  0]])
    v = Matrix(symbols("v:14", integer=True))
    B = Matrix([[2], [-2], [0], [0], [0], [0], [0], [0], [0],
        [0], [0], [0]])
    eqs = A@v-B
    assert solve(eqs) == []
    assert solve(eqs, particular=True) == []  # assumption violated
    assert all(v for v in solve([x + y + z, y + z + a]).values())
    for _q in (True, False):
        assert not all(v for v in solve(
            [x + y + z, y + z + a], quick=_q,
            particular=True).values())
        # raise error if quick used w/o particular=True
        raises(ValueError, lambda: solve([x + 1], quick=_q))
        raises(ValueError, lambda: solve([x + 1], quick=_q, particular=False))
    # and give a good error message if someone tries to use
    # particular with a single equation
    raises(ValueError, lambda: solve(x + 1, particular=True))


def test_real_roots():
    # cf. issue 6650
    x = Symbol('x', real=True)
    assert len(solve(x**5 + x**3 + 1)) == 1


def test_issue_6528():
    eqs = [
        327600995*x**2 - 37869137*x + 1809975124*y**2 - 9998905626,
        895613949*x**2 - 273830224*x*y + 530506983*y**2 - 10000000000]
    # two expressions encountered are > 1400 ops long so if this hangs
    # it is likely because simplification is being done
    assert len(solve(eqs, y, x, check=False)) == 4


def test_overdetermined():
    x = symbols('x', real=True)
    eqs = [Abs(4*x - 7) - 5, Abs(3 - 8*x) - 1]
    assert solve(eqs, x) == [(S.Half,)]
    assert solve(eqs, x, manual=True) == [(S.Half,)]
    assert solve(eqs, x, manual=True, check=False) == [(S.Half,), (S(3),)]


def test_issue_6605():
    x = symbols('x')
    assert solve(4**(x/2) - 2**(x/3)) == [0, 3*I*pi/log(2)]
    # while the first one passed, this one failed
    x = symbols('x', real=True)
    assert solve(5**(x/2) - 2**(x/3)) == [0]
    b = sqrt(6)*sqrt(log(2))/sqrt(log(5))
    assert solve(5**(x/2) - 2**(3/x)) == [-b, b]


def test__ispow():
    assert _ispow(x**2)
    assert not _ispow(x)
    assert not _ispow(True)


def test_issue_6644():
    eq = -sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) + sqrt((-m**2/2 - sqrt(
    4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m - sqrt(
    4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2)
    sol = solve(eq, q, simplify=False, check=False)
    assert len(sol) == 5


def test_issue_6752():
    assert solve([a**2 + a, a - b], [a, b]) == [(-1, -1), (0, 0)]
    assert solve([a**2 + a*c, a - b], [a, b]) == [(0, 0), (-c, -c)]


def test_issue_6792():
    assert solve(x*(x - 1)**2*(x + 1)*(x**6 - x + 1)) == [
        -1, 0, 1, CRootOf(x**6 - x + 1, 0), CRootOf(x**6 - x + 1, 1),
         CRootOf(x**6 - x + 1, 2), CRootOf(x**6 - x + 1, 3),
         CRootOf(x**6 - x + 1, 4), CRootOf(x**6 - x + 1, 5)]


def test_issues_6819_6820_6821_6248_8692_25777_25779():
    # issue 6821
    x, y = symbols('x y', real=True)
    assert solve(abs(x + 3) - 2*abs(x - 3)) == [1, 9]
    assert solve([abs(x) - 2, arg(x) - pi], x) == [(-2,)]
    assert set(solve(abs(x - 7) - 8)) == {-S.One, S(15)}

    # issue 8692
    assert solve(Eq(Abs(x + 1) + Abs(x**2 - 7), 9), x) == [
        Rational(-1, 2) + sqrt(61)/2, -sqrt(69)/2 + S.Half]

    # issue 7145
    assert solve(2*abs(x) - abs(x - 1)) == [-1, Rational(1, 3)]

    # 25777
    assert solve(abs(x**3 + x + 2)/(x + 1)) == []

    # 25779
    assert solve(abs(x)) == [0]
    assert solve(Eq(abs(x**2 - 2*x), 4), x) == [
        1 - sqrt(5), 1 + sqrt(5)]
    nn = symbols('nn', nonnegative=True)
    assert solve(abs(sqrt(nn))) == [0]
    nz = symbols('nz', nonzero=True)
    assert solve(Eq(Abs(4 + 1 / (4*nz)), 0)) == [-Rational(1, 16)]

    x = symbols('x')
    assert solve([re(x) - 1, im(x) - 2], x) == [
        {x: 1 + 2*I, re(x): 1, im(x): 2}]

    # check for 'dict' handling of solution
    eq = sqrt(re(x)**2 + im(x)**2) - 3
    assert solve(eq) == solve(eq, x)

    i = symbols('i', imaginary=True)
    assert solve(abs(i) - 3) == [-3*I, 3*I]
    raises(NotImplementedError, lambda: solve(abs(x) - 3))

    w = symbols('w', integer=True)
    assert solve(2*x**w - 4*y**w, w) == solve((x/y)**w - 2, w)

    x, y = symbols('x y', real=True)
    assert solve(x + y*I + 3) == {y: 0, x: -3}
    # issue 2642
    assert solve(x*(1 + I)) == [0]

    x, y = symbols('x y', imaginary=True)
    assert solve(x + y*I + 3 + 2*I) == {x: -2*I, y: 3*I}

    x = symbols('x', real=True)
    assert solve(x + y + 3 + 2*I) == {x: -3, y: -2*I}

    # issue 6248
    f = Function('f')
    assert solve(f(x + 1) - f(2*x - 1)) == [2]
    assert solve(log(x + 1) - log(2*x - 1)) == [2]

    x = symbols('x')
    assert solve(2**x + 4**x) == [I*pi/log(2)]

def test_issue_17638():

    assert solve(((2-exp(2*x))*exp(x))/(exp(2*x)+2)**2 > 0, x) == (-oo < x) & (x < log(2)/2)
    assert solve(((2-exp(2*x)+2)*exp(x+2))/(exp(x)+2)**2 > 0, x) == (-oo < x) & (x < log(4)/2)
    assert solve((exp(x)+2+x**2)*exp(2*x+2)/(exp(x)+2)**2 > 0, x) == (-oo < x) & (x < oo)



def test_issue_14607():
    # issue 14607
    s, tau_c, tau_1, tau_2, phi, K = symbols(
        's, tau_c, tau_1, tau_2, phi, K')

    target = (s**2*tau_1*tau_2 + s*tau_1 + s*tau_2 + 1)/(K*s*(-phi + tau_c))

    K_C, tau_I, tau_D = symbols('K_C, tau_I, tau_D',
                                positive=True, nonzero=True)
    PID = K_C*(1 + 1/(tau_I*s) + tau_D*s)

    eq = (target - PID).together()
    eq *= denom(eq).simplify()
    eq = Poly(eq, s)
    c = eq.coeffs()

    vars = [K_C, tau_I, tau_D]
    s = solve(c, vars, dict=True)

    assert len(s) == 1

    knownsolution = {K_C: -(tau_1 + tau_2)/(K*(phi - tau_c)),
                     tau_I: tau_1 + tau_2,
                     tau_D: tau_1*tau_2/(tau_1 + tau_2)}

    for var in vars:
        assert s[0][var].simplify() == knownsolution[var].simplify()


def test_lambert_multivariate():
    from sympy.abc import x, y
    assert _filtered_gens(Poly(x + 1/x + exp(x) + y), x) == {x, exp(x)}
    assert _lambert(x, x) == []
    assert solve((x**2 - 2*x + 1).subs(x, log(x) + 3*x)) == [LambertW(3*S.Exp1)/3]
    assert solve((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1)) == \
          [LambertW(3*exp(-sqrt(2)))/3, LambertW(3*exp(sqrt(2)))/3]
    assert solve((x**2 - 2*x - 2).subs(x, log(x) + 3*x)) == \
          [LambertW(3*exp(1 - sqrt(3)))/3, LambertW(3*exp(1 + sqrt(3)))/3]
    eq = (x*exp(x) - 3).subs(x, x*exp(x))
    assert solve(eq) == [LambertW(3*exp(-LambertW(3)))]
    # coverage test
    raises(NotImplementedError, lambda: solve(x - sin(x)*log(y - x), x))
    ans = [3, -3*LambertW(-log(3)/3)/log(3)]  # 3 and 2.478...
    assert solve(x**3 - 3**x, x) == ans
    assert set(solve(3*log(x) - x*log(3))) == set(ans)
    assert solve(LambertW(2*x) - y, x) == [y*exp(y)/2]


@XFAIL
def test_other_lambert():
    assert solve(3*sin(x) - x*sin(3), x) == [3]
    assert set(solve(x**a - a**x), x) == {
        a, -a*LambertW(-log(a)/a)/log(a)}


@slow
def test_lambert_bivariate():
    # tests passing current implementation
    assert solve((x**2 + x)*exp(x**2 + x) - 1) == [
        Rational(-1, 2) + sqrt(1 + 4*LambertW(1))/2,
        Rational(-1, 2) - sqrt(1 + 4*LambertW(1))/2]
    assert solve((x**2 + x)*exp((x**2 + x)*2) - 1) == [
        Rational(-1, 2) + sqrt(1 + 2*LambertW(2))/2,
        Rational(-1, 2) - sqrt(1 + 2*LambertW(2))/2]
    assert solve(a/x + exp(x/2), x) == [2*LambertW(-a/2)]
    assert solve((a/x + exp(x/2)).diff(x), x) == \
            [4*LambertW(-sqrt(2)*sqrt(a)/4), 4*LambertW(sqrt(2)*sqrt(a)/4)]
    assert solve((1/x + exp(x/2)).diff(x), x) == \
        [4*LambertW(-sqrt(2)/4),
        4*LambertW(sqrt(2)/4),  # nsimplifies as 2*2**(141/299)*3**(206/299)*5**(205/299)*7**(37/299)/21
        4*LambertW(-sqrt(2)/4, -1)]
    assert solve(x*log(x) + 3*x + 1, x) == \
            [exp(-3 + LambertW(-exp(3)))]
    assert solve(-x**2 + 2**x, x) == [2, 4, -2*LambertW(log(2)/2)/log(2)]
    assert solve(x**2 - 2**x, x) == [2, 4, -2*LambertW(log(2)/2)/log(2)]
    ans = solve(3*x + 5 + 2**(-5*x + 3), x)
    assert len(ans) == 1 and ans[0].expand() == \
        Rational(-5, 3) + LambertW(-10240*root(2, 3)*log(2)/3)/(5*log(2))
    assert solve(5*x - 1 + 3*exp(2 - 7*x), x) == \
        [Rational(1, 5) + LambertW(-21*exp(Rational(3, 5))/5)/7]
    assert solve((log(x) + x).subs(x, x**2 + 1)) == [
        -I*sqrt(-LambertW(1) + 1), sqrt(-1 + LambertW(1))]
    # check collection
    ax = a**(3*x + 5)
    ans = solve(3*log(ax) + b*log(ax) + ax, x)
    x0 = 1/log(a)
    x1 = sqrt(3)*I
    x2 = b + 3
    x3 = x2*LambertW(1/x2)/a**5
    x4 = x3**Rational(1, 3)/2
    assert ans == [
        x0*log(x4*(-x1 - 1)),
        x0*log(x4*(x1 - 1)),
        x0*log(x3)/3]
    x1 = LambertW(Rational(1, 3))
    x2 = a**(-5)
    x3 = -3**Rational(1, 3)
    x4 = 3**Rational(5, 6)*I
    x5 = x1**Rational(1, 3)*x2**Rational(1, 3)/2
    ans = solve(3*log(ax) + ax, x)
    assert ans == [
        x0*log(3*x1*x2)/3,
        x0*log(x5*(x3 - x4)),
        x0*log(x5*(x3 + x4))]
    # coverage
    p = symbols('p', positive=True)
    eq = 4*2**(2*p + 3) - 2*p - 3
    assert _solve_lambert(eq, p, _filtered_gens(Poly(eq), p)) == [
        Rational(-3, 2) - LambertW(-4*log(2))/(2*log(2))]
    assert set(solve(3**cos(x) - cos(x)**3)) == {
        acos(3), acos(-3*LambertW(-log(3)/3)/log(3))}
    # should give only one solution after using `uniq`
    assert solve(2*log(x) - 2*log(z) + log(z + log(x) + log(z)), x) == [
        exp(-z + LambertW(2*z**4*exp(2*z))/2)/z]
    # cases when p != S.One
    # issue 4271
    ans = solve((a/x + exp(x/2)).diff(x, 2), x)
    x0 = (-a)**Rational(1, 3)
    x1 = sqrt(3)*I
    x2 = x0/6
    assert ans == [
        6*LambertW(x0/3),
        6*LambertW(x2*(-x1 - 1)),
        6*LambertW(x2*(x1 - 1))]
    assert solve((1/x + exp(x/2)).diff(x, 2), x) == \
                [6*LambertW(Rational(-1, 3)), 6*LambertW(Rational(1, 6) - sqrt(3)*I/6), \
                6*LambertW(Rational(1, 6) + sqrt(3)*I/6), 6*LambertW(Rational(-1, 3), -1)]
    assert solve(x**2 - y**2/exp(x), x, y, dict=True) == \
                [{x: 2*LambertW(-y/2)}, {x: 2*LambertW(y/2)}]
    # this is slow but not exceedingly slow
    assert solve((x**3)**(x/2) + pi/2, x) == [
        exp(LambertW(-2*log(2)/3 + 2*log(pi)/3 + I*pi*Rational(2, 3)))]

    # issue 23253
    assert solve((1/log(sqrt(x) + 2)**2 - 1/x)) == [
        (LambertW(-exp(-2), -1) + 2)**2]
    assert solve((1/log(1/sqrt(x) + 2)**2 - x)) == [
        (LambertW(-exp(-2), -1) + 2)**-2]
    assert solve((1/log(x**2 + 2)**2 - x**-4)) == [
        -I*sqrt(2 - LambertW(exp(2))),
        -I*sqrt(LambertW(-exp(-2)) + 2),
        sqrt(-2 - LambertW(-exp(-2))),
        sqrt(-2 + LambertW(exp(2))),
        -sqrt(-2 - LambertW(-exp(-2), -1)),
        sqrt(-2 - LambertW(-exp(-2), -1))]


def test_rewrite_trig():
    assert solve(sin(x) + tan(x)) == [0, -pi, pi, 2*pi]
    assert solve(sin(x) + sec(x)) == [
        -2*atan(Rational(-1, 2) + sqrt(2)*sqrt(1 - sqrt(3)*I)/2 + sqrt(3)*I/2),
        2*atan(S.Half - sqrt(2)*sqrt(1 + sqrt(3)*I)/2 + sqrt(3)*I/2), 2*atan(S.Half
        + sqrt(2)*sqrt(1 + sqrt(3)*I)/2 + sqrt(3)*I/2), 2*atan(S.Half -
        sqrt(3)*I/2 + sqrt(2)*sqrt(1 - sqrt(3)*I)/2)]
    assert solve(sinh(x) + tanh(x)) == [0, I*pi]

    # issue 6157
    assert solve(2*sin(x) - cos(x), x) == [atan(S.Half)]


@XFAIL
def test_rewrite_trigh():
    # if this import passes then the test below should also pass
    from sympy.functions.elementary.hyperbolic import sech
    assert solve(sinh(x) + sech(x)) == [
        2*atanh(Rational(-1, 2) + sqrt(5)/2 - sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(Rational(-1, 2) + sqrt(5)/2 + sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(-sqrt(5)/2 - S.Half + sqrt(2 + 2*sqrt(5))/2),
        2*atanh(-sqrt(2 + 2*sqrt(5))/2 - sqrt(5)/2 - S.Half)]


def test_uselogcombine():
    eq = z - log(x) + log(y/(x*(-1 + y**2/x**2)))
    assert solve(eq, x, force=True) == [-sqrt(y*(y - exp(z))), sqrt(y*(y - exp(z)))]
    assert solve(log(x + 3) + log(1 + 3/x) - 3) in [
        [-3 + sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 + exp(3)/2,
        -sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 - 3 + exp(3)/2],
        [-3 + sqrt(-36 + (-exp(3) + 6)**2)/2 + exp(3)/2,
        -3 - sqrt(-36 + (-exp(3) + 6)**2)/2 + exp(3)/2],
        ]
    assert solve(log(exp(2*x) + 1) + log(-tanh(x) + 1) - log(2)) == []


def test_atan2():
    assert solve(atan2(x, 2) - pi/3, x) == [2*sqrt(3)]


def test_errorinverses():
    assert solve(erf(x) - y, x) == [erfinv(y)]
    assert solve(erfinv(x) - y, x) == [erf(y)]
    assert solve(erfc(x) - y, x) == [erfcinv(y)]
    assert solve(erfcinv(x) - y, x) == [erfc(y)]


def test_issue_2725():
    R = Symbol('R')
    eq = sqrt(2)*R*sqrt(1/(R + 1)) + (R + 1)*(sqrt(2)*sqrt(1/(R + 1)) - 1)
    sol = solve(eq, R, set=True)[1]
    assert sol == {(Rational(5, 3) + (Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3) + 40/(9*((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3))),), (Rational(5, 3) + 40/(9*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3)) + (Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3),)}


def test_issue_5114_6611():
    # See that it doesn't hang; this solves in about 2 seconds.
    # Also check that the solution is relatively small.
    # Note: the system in issue 6611 solves in about 5 seconds and has
    # an op-count of 138336 (with simplify=False).
    b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = symbols('b:r')
    eqs = Matrix([
        [b - c/d + r/d], [c*(1/g + 1/e + 1/d) - f/g - r/d],
        [-c/g + f*(1/j + 1/i + 1/g) - h/i], [-f/i + h*(1/m + 1/l + 1/i) - k/m],
        [-h/m + k*(1/p + 1/o + 1/m) - n/p], [-k/p + n*(1/q + 1/p)]])
    v = Matrix([f, h, k, n, b, c])
    ans = solve(list(eqs), list(v), simplify=False)
    # If time is taken to simplify then then 2617 below becomes
    # 1168 and the time is about 50 seconds instead of 2.
    assert sum(s.count_ops() for s in ans.values()) <= 3270


def test_det_quick():
    m = Matrix(3, 3, symbols('a:9'))
    assert m.det() == det_quick(m)  # calls det_perm
    m[0, 0] = 1
    assert m.det() == det_quick(m)  # calls det_minor
    m = Matrix(3, 3, list(range(9)))
    assert m.det() == det_quick(m)  # defaults to .det()
    # make sure they work with Sparse
    s = SparseMatrix(2, 2, (1, 2, 1, 4))
    assert det_perm(s) == det_minor(s) == s.det()


def test_real_imag_splitting():
    a, b = symbols('a b', real=True)
    assert solve(sqrt(a**2 + b**2) - 3, a) == \
        [-sqrt(-b**2 + 9), sqrt(-b**2 + 9)]
    a, b = symbols('a b', imaginary=True)
    assert solve(sqrt(a**2 + b**2) - 3, a) == []


def test_issue_7110():
    y = -2*x**3 + 4*x**2 - 2*x + 5
    assert any(ask(Q.real(i)) for i in solve(y))


def test_units():
    assert solve(1/x - 1/(2*cm)) == [2*cm]


def test_issue_7547():
    A, B, V = symbols('A,B,V')
    eq1 = Eq(630.26*(V - 39.0)*V*(V + 39) - A + B, 0)
    eq2 = Eq(B, 1.36*10**8*(V - 39))
    eq3 = Eq(A, 5.75*10**5*V*(V + 39.0))
    sol = Matrix(nsolve(Tuple(eq1, eq2, eq3), [A, B, V], (0, 0, 0)))
    assert str(sol) == str(Matrix(
        [['4442890172.68209'],
         ['4289299466.1432'],
         ['70.5389666628177']]))


def test_issue_7895():
    r = symbols('r', real=True)
    assert solve(sqrt(r) - 2) == [4]


def test_issue_2777():
    # the equations represent two circles
    x, y = symbols('x y', real=True)
    e1, e2 = sqrt(x**2 + y**2) - 10, sqrt(y**2 + (-x + 10)**2) - 3
    a, b = Rational(191, 20), 3*sqrt(391)/20
    ans = [(a, -b), (a, b)]
    assert solve((e1, e2), (x, y)) == ans
    assert solve((e1, e2/(x - a)), (x, y)) == []
    # make the 2nd circle's radius be -3
    e2 += 6
    assert solve((e1, e2), (x, y)) == []
    assert solve((e1, e2), (x, y), check=False) == ans


def test_issue_7322():
    number = 5.62527e-35
    assert solve(x - number, x)[0] == number


def test_nsolve():
    raises(ValueError, lambda: nsolve(x, (-1, 1), method='bisect'))
    raises(TypeError, lambda: nsolve((x - y + 3,x + y,z - y),(x,y,z),(-50,50)))
    raises(TypeError, lambda: nsolve((x + y, x - y), (0, 1)))
    raises(TypeError, lambda: nsolve(x < 0.5, x, 1))


@slow
def test_high_order_multivariate():
    assert len(solve(a*x**3 - x + 1, x)) == 3
    assert len(solve(a*x**4 - x + 1, x)) == 4
    assert solve(a*x**5 - x + 1, x) == []  # incomplete solution allowed
    raises(NotImplementedError, lambda:
        solve(a*x**5 - x + 1, x, incomplete=False))

    # result checking must always consider the denominator and CRootOf
    # must be checked, too
    d = x**5 - x + 1
    assert solve(d*(1 + 1/d)) == [CRootOf(d + 1, i) for i in range(5)]
    d = x - 1
    assert solve(d*(2 + 1/d)) == [S.Half]


def test_base_0_exp_0():
    assert solve(0**x - 1) == [0]
    assert solve(0**(x - 2) - 1) == [2]
    assert solve(S('x*(1/x**0 - x)', evaluate=False)) == \
        [0, 1]


def test__simple_dens():
    assert _simple_dens(1/x**0, [x]) == set()
    assert _simple_dens(1/x**y, [x]) == {x**y}
    assert _simple_dens(1/root(x, 3), [x]) == {x}


def test_issue_8755():
    # This tests two things: that if full unrad is attempted and fails
    # the solution should still be found; also it tests the use of
    # keyword `composite`.
    assert len(solve(sqrt(y)*x + x**3 - 1, x)) == 3
    assert len(solve(-512*y**3 + 1344*(x + 2)**Rational(1, 3)*y**2 -
        1176*(x + 2)**Rational(2, 3)*y - 169*x + 686, y, _unrad=False)) == 3


@slow
def test_issue_8828():
    x1 = 0
    y1 = -620
    r1 = 920
    x2 = 126
    y2 = 276
    x3 = 51
    y3 = 205
    r3 = 104
    v = x, y, z

    f1 = (x - x1)**2 + (y - y1)**2 - (r1 - z)**2
    f2 = (x - x2)**2 + (y - y2)**2 - z**2
    f3 = (x - x3)**2 + (y - y3)**2 - (r3 - z)**2
    F = f1,f2,f3

    g1 = sqrt((x - x1)**2 + (y - y1)**2) + z - r1
    g2 = f2
    g3 = sqrt((x - x3)**2 + (y - y3)**2) + z - r3
    G = g1,g2,g3

    A = solve(F, v)
    B = solve(G, v)
    C = solve(G, v, manual=True)

    p, q, r = [{tuple(i.evalf(2) for i in j) for j in R} for R in [A, B, C]]
    assert p == q == r


def test_issue_2840_8155():
    # with parameter-free solutions (i.e. no `n`), we want to avoid
    # excessive periodic solutions
    assert solve(sin(3*x) + sin(6*x)) == [0, -2*pi/9, 2*pi/9]
    assert solve(sin(300*x) + sin(600*x)) == [0, -pi/450, pi/450]
    assert solve(2*sin(x) - 2*sin(2*x)) == [0, -pi/3, pi/3]


def test_issue_9567():
    assert solve(1 + 1/(x - 1)) == [0]


def test_issue_11538():
    assert solve(x + E) == [-E]
    assert solve(x**2 + E) == [-I*sqrt(E), I*sqrt(E)]
    assert solve(x**3 + 2*E) == [
        -cbrt(2 * E),
        cbrt(2)*cbrt(E)/2 - cbrt(2)*sqrt(3)*I*cbrt(E)/2,
        cbrt(2)*cbrt(E)/2 + cbrt(2)*sqrt(3)*I*cbrt(E)/2]
    assert solve([x + 4, y + E], x, y) == {x: -4, y: -E}
    assert solve([x**2 + 4, y + E], x, y) == [
        (-2*I, -E), (2*I, -E)]

    e1 = x - y**3 + 4
    e2 = x + y + 4 + 4 * E
    assert len(solve([e1, e2], x, y)) == 3


@slow
def test_issue_12114():
    a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g')
    terms = [1 + a*b + d*e, 1 + a*c + d*f, 1 + b*c + e*f,
             g - a**2 - d**2, g - b**2 - e**2, g - c**2 - f**2]
    sol = solve(terms, [a, b, c, d, e, f, g], dict=True)
    s = sqrt(-f**2 - 1)
    s2 = sqrt(2 - f**2)
    s3 = sqrt(6 - 3*f**2)
    s4 = sqrt(3)*f
    s5 = sqrt(3)*s2
    assert sol == [
        {a: -s, b: -s, c: -s, d: f, e: f, g: -1},
        {a: s, b: s, c: s, d: f, e: f, g: -1},
        {a: -s4/2 - s2/2, b: s4/2 - s2/2, c: s2,
            d: -f/2 + s3/2, e: -f/2 - s5/2, g: 2},
        {a: -s4/2 + s2/2, b: s4/2 + s2/2, c: -s2,
            d: -f/2 - s3/2, e: -f/2 + s5/2, g: 2},
        {a: s4/2 - s2/2, b: -s4/2 - s2/2, c: s2,
            d: -f/2 - s3/2, e: -f/2 + s5/2, g: 2},
        {a: s4/2 + s2/2, b: -s4/2 + s2/2, c: -s2,
            d: -f/2 + s3/2, e: -f/2 - s5/2, g: 2}]


def test_inf():
    assert solve(1 - oo*x) == []
    assert solve(oo*x, x) == []
    assert solve(oo*x - oo, x) == []


def test_issue_12448():
    f = Function('f')
    fun = [f(i) for i in range(15)]
    sym = symbols('x:15')
    reps = dict(zip(fun, sym))

    (x, y, z), c = sym[:3], sym[3:]
    ssym = solve([c[4*i]*x + c[4*i + 1]*y + c[4*i + 2]*z + c[4*i + 3]
        for i in range(3)], (x, y, z))

    (x, y, z), c = fun[:3], fun[3:]
    sfun = solve([c[4*i]*x + c[4*i + 1]*y + c[4*i + 2]*z + c[4*i + 3]
        for i in range(3)], (x, y, z))

    assert sfun[fun[0]].xreplace(reps).count_ops() == \
        ssym[sym[0]].count_ops()


def test_denoms():
    assert denoms(x/2 + 1/y) == {2, y}
    assert denoms(x/2 + 1/y, y) == {y}
    assert denoms(x/2 + 1/y, [y]) == {y}
    assert denoms(1/x + 1/y + 1/z, [x, y]) == {x, y}
    assert denoms(1/x + 1/y + 1/z, x, y) == {x, y}
    assert denoms(1/x + 1/y + 1/z, {x, y}) == {x, y}


def test_issue_12476():
    x0, x1, x2, x3, x4, x5 = symbols('x0 x1 x2 x3 x4 x5')
    eqns = [x0**2 - x0, x0*x1 - x1, x0*x2 - x2, x0*x3 - x3, x0*x4 - x4, x0*x5 - x5,
            x0*x1 - x1, -x0/3 + x1**2 - 2*x2/3, x1*x2 - x1/3 - x2/3 - x3/3,
            x1*x3 - x2/3 - x3/3 - x4/3, x1*x4 - 2*x3/3 - x5/3, x1*x5 - x4, x0*x2 - x2,
            x1*x2 - x1/3 - x2/3 - x3/3, -x0/6 - x1/6 + x2**2 - x2/6 - x3/3 - x4/6,
            -x1/6 + x2*x3 - x2/3 - x3/6 - x4/6 - x5/6, x2*x4 - x2/3 - x3/3 - x4/3,
            x2*x5 - x3, x0*x3 - x3, x1*x3 - x2/3 - x3/3 - x4/3,
            -x1/6 + x2*x3 - x2/3 - x3/6 - x4/6 - x5/6,
            -x0/6 - x1/6 - x2/6 + x3**2 - x3/3 - x4/6, -x1/3 - x2/3 + x3*x4 - x3/3,
            -x2 + x3*x5, x0*x4 - x4, x1*x4 - 2*x3/3 - x5/3, x2*x4 - x2/3 - x3/3 - x4/3,
            -x1/3 - x2/3 + x3*x4 - x3/3, -x0/3 - 2*x2/3 + x4**2, -x1 + x4*x5, x0*x5 - x5,
            x1*x5 - x4, x2*x5 - x3, -x2 + x3*x5, -x1 + x4*x5, -x0 + x5**2, x0 - 1]
    sols = [{x0: 1, x3: Rational(1, 6), x2: Rational(1, 6), x4: Rational(-2, 3), x1: Rational(-2, 3), x5: 1},
            {x0: 1, x3: S.Half, x2: Rational(-1, 2), x4: 0, x1: 0, x5: -1},
            {x0: 1, x3: Rational(-1, 3), x2: Rational(-1, 3), x4: Rational(1, 3), x1: Rational(1, 3), x5: 1},
            {x0: 1, x3: 1, x2: 1, x4: 1, x1: 1, x5: 1},
            {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: sqrt(5)/3, x1: -sqrt(5)/3, x5: -1},
            {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: -sqrt(5)/3, x1: sqrt(5)/3, x5: -1}]

    assert solve(eqns) == sols


def test_issue_13849():
    t = symbols('t')
    assert solve((t*(sqrt(5) + sqrt(2)) - sqrt(2), t), t) == []


def test_issue_14860():
    from sympy.physics.units import newton, kilo
    assert solve(8*kilo*newton + x + y, x) == [-8000*newton - y]


def test_issue_14721():
    k, h, a, b = symbols(':4')
    assert solve([
        -1 + (-k + 1)**2/b**2 + (-h - 1)**2/a**2,
        -1 + (-k + 1)**2/b**2 + (-h + 1)**2/a**2,
        h, k + 2], h, k, a, b) == [
        (0, -2, -b*sqrt(1/(b**2 - 9)), b),
        (0, -2, b*sqrt(1/(b**2 - 9)), b)]
    assert solve([
        h, h/a + 1/b**2 - 2, -h/2 + 1/b**2 - 2], a, h, b) == [
        (a, 0, -sqrt(2)/2), (a, 0, sqrt(2)/2)]
    assert solve((a + b**2 - 1, a + b**2 - 2)) == []


def test_issue_14779():
    x = symbols('x', real=True)
    assert solve(sqrt(x**4 - 130*x**2 + 1089) + sqrt(x**4 - 130*x**2
                 + 3969) - 96*Abs(x)/x,x) == [sqrt(130)]


def test_issue_15307():
    assert solve((y - 2, Mul(x + 3,x - 2, evaluate=False))) == \
        [{x: -3, y: 2}, {x: 2, y: 2}]
    assert solve((y - 2, Mul(3, x - 2, evaluate=False))) == \
        {x: 2, y: 2}
    assert solve((y - 2, Add(x + 4, x - 2, evaluate=False))) == \
        {x: -1, y: 2}
    eq1 = Eq(12513*x + 2*y - 219093, -5726*x - y)
    eq2 = Eq(-2*x + 8, 2*x - 40)
    assert solve([eq1, eq2]) == {x:12, y:75}


def test_issue_15415():
    assert solve(x - 3, x) == [3]
    assert solve([x - 3], x) == {x:3}
    assert solve(Eq(y + 3*x**2/2, y + 3*x), y) == []
    assert solve([Eq(y + 3*x**2/2, y + 3*x)], y) == []
    assert solve([Eq(y + 3*x**2/2, y + 3*x), Eq(x, 1)], y) == []


@slow
def test_issue_15731():
    # f(x)**g(x)=c
    assert solve(Eq((x**2 - 7*x + 11)**(x**2 - 13*x + 42), 1)) == [2, 3, 4, 5, 6, 7]
    assert solve((x)**(x + 4) - 4) == [-2]
    assert solve((-x)**(-x + 4) - 4) == [2]
    assert solve((x**2 - 6)**(x**2 - 2) - 4) == [-2, 2]
    assert solve((x**2 - 2*x - 1)**(x**2 - 3) - 1/(1 - 2*sqrt(2))) == [sqrt(2)]
    assert solve(x**(x + S.Half) - 4*sqrt(2)) == [S(2)]
    assert solve((x**2 + 1)**x - 25) == [2]
    assert solve(x**(2/x) - 2) == [2, 4]
    assert solve((x/2)**(2/x) - sqrt(2)) == [4, 8]
    assert solve(x**(x + S.Half) - Rational(9, 4)) == [Rational(3, 2)]
    # a**g(x)=c
    assert solve((-sqrt(sqrt(2)))**x - 2) == [4, log(2)/(log(2**Rational(1, 4)) + I*pi)]
    assert solve((sqrt(2))**x - sqrt(sqrt(2))) == [S.Half]
    assert solve((-sqrt(2))**x + 2*(sqrt(2))) == [3,
            (3*log(2)**2 + 4*pi**2 - 4*I*pi*log(2))/(log(2)**2 + 4*pi**2)]
    assert solve((sqrt(2))**x - 2*(sqrt(2))) == [3]
    assert solve(I**x + 1) == [2]
    assert solve((1 + I)**x - 2*I) == [2]
    assert solve((sqrt(2) + sqrt(3))**x - (2*sqrt(6) + 5)**Rational(1, 3)) == [Rational(2, 3)]
    # bases of both sides are equal
    b = Symbol('b')
    assert solve(b**x - b**2, x) == [2]
    assert solve(b**x - 1/b, x) == [-1]
    assert solve(b**x - b, x) == [1]
    b = Symbol('b', positive=True)
    assert solve(b**x - b**2, x) == [2]
    assert solve(b**x - 1/b, x) == [-1]


def test_issue_10933():
    assert solve(x**4 + y*(x + 0.1), x)  # doesn't fail
    assert solve(I*x**4 + x**3 + x**2 + 1.)  # doesn't fail


def test_Abs_handling():
    x = symbols('x', real=True)
    assert solve(abs(x/y), x) == [0]


def test_issue_7982():
    x = Symbol('x')
    # Test that no exception happens
    assert solve([2*x**2 + 5*x + 20 <= 0, x >= 1.5], x) is S.false
    # From #8040
    assert solve([x**3 - 8.08*x**2 - 56.48*x/5 - 106 >= 0, x - 1 <= 0], [x]) is S.false


def test_issue_14645():
    x, y = symbols('x y')
    assert solve([x*y - x - y, x*y - x - y], [x, y]) == [(y/(y - 1), y)]


def test_issue_12024():
    x, y = symbols('x y')
    assert solve(Piecewise((0.0, x < 0.1), (x, x >= 0.1)) - y) == \
        [{y: Piecewise((0.0, x < 0.1), (x, True))}]


def test_issue_17452():
    assert solve((7**x)**x + pi, x) == [-sqrt(log(pi) + I*pi)/sqrt(log(7)),
                                        sqrt(log(pi) + I*pi)/sqrt(log(7))]
    assert solve(x**(x/11) + pi/11, x) == [exp(LambertW(-11*log(11) + 11*log(pi) + 11*I*pi))]


def test_issue_17799():
    assert solve(-erf(x**(S(1)/3))**pi + I, x) == []


def test_issue_17650():
    x = Symbol('x', real=True)
    assert solve(abs(abs(x**2 - 1) - x) - x) == [1, -1 + sqrt(2), 1 + sqrt(2)]


def test_issue_17882():
    eq = -8*x**2/(9*(x**2 - 1)**(S(4)/3)) + 4/(3*(x**2 - 1)**(S(1)/3))
    assert unrad(eq) is None


def test_issue_17949():
    assert solve(exp(+x+x**2), x) == []
    assert solve(exp(-x+x**2), x) == []
    assert solve(exp(+x-x**2), x) == []
    assert solve(exp(-x-x**2), x) == []


def test_issue_10993():
    assert solve(Eq(binomial(x, 2), 3)) == [-2, 3]
    assert solve(Eq(pow(x, 2) + binomial(x, 3), x)) == [-4, 0, 1]
    assert solve(Eq(binomial(x, 2), 0)) == [0, 1]
    assert solve(a+binomial(x, 3), a) == [-binomial(x, 3)]
    assert solve(x-binomial(a, 3) + binomial(y, 2) + sin(a), x) == [-sin(a) + binomial(a, 3) - binomial(y, 2)]
    assert solve((x+1)-binomial(x+1, 3), x) == [-2, -1, 3]


def test_issue_11553():
    eq1 = x + y + 1
    eq2 = x + GoldenRatio
    assert solve([eq1, eq2], x, y) == {x: -GoldenRatio, y: -1 + GoldenRatio}
    eq3 = x + 2 + TribonacciConstant
    assert solve([eq1, eq3], x, y) == {x: -2 - TribonacciConstant, y: 1 + TribonacciConstant}


def test_issue_19113_19102():
    t = S(1)/3
    solve(cos(x)**5-sin(x)**5)
    assert solve(4*cos(x)**3 - 2*sin(x)**3) == [
        atan(2**(t)), -atan(2**(t)*(1 - sqrt(3)*I)/2),
        -atan(2**(t)*(1 + sqrt(3)*I)/2)]
    h = S.Half
    assert solve(cos(x)**2 + sin(x)) == [
        2*atan(-h + sqrt(5)/2 + sqrt(2)*sqrt(1 - sqrt(5))/2),
        -2*atan(h + sqrt(5)/2 + sqrt(2)*sqrt(1 + sqrt(5))/2),
        -2*atan(-sqrt(5)/2 + h + sqrt(2)*sqrt(1 - sqrt(5))/2),
        -2*atan(-sqrt(2)*sqrt(1 + sqrt(5))/2 + h + sqrt(5)/2)]
    assert solve(3*cos(x) - sin(x)) == [atan(3)]


def test_issue_19509():
    a = S(3)/4
    b = S(5)/8
    c = sqrt(5)/8
    d = sqrt(5)/4
    assert solve(1/(x -1)**5 - 1) == [2,
        -d + a - sqrt(-b + c),
        -d + a + sqrt(-b + c),
        d + a - sqrt(-b - c),
        d + a + sqrt(-b - c)]

def test_issue_20747():
    THT, HT, DBH, dib, c0, c1, c2, c3, c4  = symbols('THT HT DBH dib c0 c1 c2 c3 c4')
    f = DBH*c3 + THT*c4 + c2
    rhs = 1 - ((HT - 1)/(THT - 1))**c1*(1 - exp(c0/f))
    eq = dib - DBH*(c0 - f*log(rhs))
    term = ((1 - exp((DBH*c0 - dib)/(DBH*(DBH*c3 + THT*c4 + c2))))
            / (1 - exp(c0/(DBH*c3 + THT*c4 + c2))))
    sol = [THT*term**(1/c1) - term**(1/c1) + 1]
    assert solve(eq, HT) == sol


def test_issue_27001():
    assert solve((x, x**2), (x, y, z), dict=True) == [{x: 0}]
    s = a1, a2, a3, a4, a5 = symbols('a1:6')
    eqs = [8*a1**4*a2 + 4*a1**2*a2**3 - 8*a1**2*a2*a4 + a2**5/2 - 2*a2**3*a4 +
        8*a2*a3**2 + 2*a2*a4**2 + 8*a2*a5, 12*a1**4 + 6*a1**2*a2**2 -
        8*a1**2*a4 + 3*a2**4/4 - 2*a2**2*a4 + 4*a3**2 + a4**2 + 4*a5, 16*a1**3
        + 4*a1*a2**2 - 8*a1*a4, -8*a1**2*a2 - 2*a2**3 + 4*a2*a4]
    sol = [{a4: 2*a1**2 + a2**2/2, a5: -a3**2}, {a1: 0, a2: 0, a5: -a3**2 - a4**2/4}]
    assert solve(eqs, s, dict=True) == sol
    assert (g:=solve(groebner(eqs, s), dict=True)) == sol, g


def test_issue_20902():
    f = (t / ((1 + t) ** 2))
    assert solve(f.subs({t: 3 * x + 2}).diff(x) > 0, x) == (S(-1) < x) & (x < S(-1)/3)
    assert solve(f.subs({t: 3 * x + 3}).diff(x) > 0, x) == (S(-4)/3 < x) & (x < S(-2)/3)
    assert solve(f.subs({t: 3 * x + 4}).diff(x) > 0, x) == (S(-5)/3 < x) & (x < S(-1))
    assert solve(f.subs({t: 3 * x + 2}).diff(x) > 0, x) == (S(-1) < x) & (x < S(-1)/3)


def test_issue_21034():
    a = symbols('a', real=True)
    system = [x - cosh(cos(4)), y - sinh(cos(a)), z - tanh(x)]
    # constants inside hyperbolic functions should not be rewritten in terms of exp
    assert solve(system, x, y, z) == [(cosh(cos(4)), sinh(cos(a)), tanh(cosh(cos(4))))]
    # but if the variable of interest is present in a hyperbolic function,
    # then it should be rewritten in terms of exp and solved further
    newsystem = [(exp(x) - exp(-x)) - tanh(x)*(exp(x) + exp(-x)) + x - 5]
    assert solve(newsystem, x) == {x: 5}


def test_issue_4886():
    z = a*sqrt(R**2*a**2 + R**2*b**2 - c**2)/(a**2 + b**2)
    t = b*c/(a**2 + b**2)
    sol = [((b*(t - z) - c)/(-a), t - z), ((b*(t + z) - c)/(-a), t + z)]
    assert solve([x**2 + y**2 - R**2, a*x + b*y - c], x, y) == sol


def test_issue_6819():
    a, b, c, d = symbols('a b c d', positive=True)
    assert solve(a*b**x - c*d**x, x) == [log(c/a)/log(b/d)]


def test_issue_17454():
    x = Symbol('x')
    assert solve((1 - x - I)**4, x) == [1 - I]


def test_issue_21852():
    solution = [21 - 21*sqrt(2)/2]
    assert solve(2*x + sqrt(2*x**2) - 21) == solution


def test_issue_21942():
    eq = -d + (a*c**(1 - e) + b**(1 - e)*(1 - a))**(1/(1 - e))
    sol = solve(eq, c, simplify=False, check=False)
    assert sol == [((a*b**(1 - e) - b**(1 - e) +
        d**(1 - e))/a)**(1/(1 - e))]


def test_solver_flags():
    root = solve(x**5 + x**2 - x - 1, cubics=False)
    rad = solve(x**5 + x**2 - x - 1, cubics=True)
    assert root != rad


def test_issue_22768():
    eq = 2*x**3 - 16*(y - 1)**6*z**3
    assert solve(eq.expand(), x, simplify=False
        ) == [2*z*(y - 1)**2, z*(-1 + sqrt(3)*I)*(y - 1)**2,
        -z*(1 + sqrt(3)*I)*(y - 1)**2]


def test_issue_22717():
    assert solve((-y**2 + log(y**2/x) + 2, -2*x*y + 2*x/y)) == [
        {y: -1, x: E}, {y: 1, x: E}]


def test_issue_25176():
    eq = (x - 5)**-8 - 3
    sol = solve(eq)
    assert not any(eq.subs(x, i) for i in sol)


def test_issue_10169():
    eq = S(-8*a - x**5*(a + b + c + e) - x**4*(4*a - 2**Rational(3,4)*c + 4*c +
        d + 2**Rational(3,4)*e + 4*e + k) - x**3*(-4*2**Rational(3,4)*c + sqrt(2)*c -
        2**Rational(3,4)*d + 4*d + sqrt(2)*e + 4*2**Rational(3,4)*e + 2**Rational(3,4)*k + 4*k) -
        x**2*(4*sqrt(2)*c - 4*2**Rational(3,4)*d + sqrt(2)*d + 4*sqrt(2)*e +
        sqrt(2)*k + 4*2**Rational(3,4)*k) - x*(2*a + 2*b + 4*sqrt(2)*d +
        4*sqrt(2)*k) + 5)
    assert solve_undetermined_coeffs(eq, [a, b, c, d, e, k], x) == {
        a: Rational(5,8),
        b: Rational(-5,1032),
        c: Rational(-40,129) - 5*2**Rational(3,4)/129 + 5*2**Rational(1,4)/1032,
        d: -20*2**Rational(3,4)/129 - 10*sqrt(2)/129 - 5*2**Rational(1,4)/258,
        e: Rational(-40,129) - 5*2**Rational(1,4)/1032 + 5*2**Rational(3,4)/129,
        k: -10*sqrt(2)/129 + 5*2**Rational(1,4)/258 + 20*2**Rational(3,4)/129
    }


def test_solve_undetermined_coeffs_issue_23927():
    A, B, r, phi = symbols('A, B, r, phi')
    e = Eq(A*sin(t) + B*cos(t), r*sin(t - phi))
    eq = (e.lhs - e.rhs).expand(trig=True)
    soln = solve_undetermined_coeffs(eq, (r, phi), t)
    assert soln == [{
        phi: 2*atan((A - sqrt(A**2 + B**2))/B),
        r: (-A**2 + A*sqrt(A**2 + B**2) - B**2)/(A - sqrt(A**2 + B**2))
        }, {
        phi: 2*atan((A + sqrt(A**2 + B**2))/B),
        r: (A**2 + A*sqrt(A**2 + B**2) + B**2)/(A + sqrt(A**2 + B**2))/-1
        }]

def test_issue_24368():
    # Ideally these would produce a solution, but for now just check that they
    # don't fail with a RuntimeError
    raises(NotImplementedError, lambda: solve(Mod(x**2, 49), x))
    s2 = Symbol('s2', integer=True, positive=True)
    f = floor(s2/2 - S(1)/2)
    raises(NotImplementedError, lambda: solve((Mod(f**2/(f + 1) + 2*f/(f + 1) + 1/(f + 1), 1))*f + Mod(f**2/(f + 1) + 2*f/(f + 1) + 1/(f + 1), 1), s2))


def test_solve_Piecewise():
    assert [S(10)/3] == solve(3*Piecewise(
        (S.NaN, x <= 0),
        (20*x - 3*(x - 6)**2/2 - 176, (x >= 0) & (x >= 2) & (x>= 4) & (x >= 6) & (x < 10)),
        (100 - 26*x, (x >= 0) & (x >= 2) & (x >= 4) & (x < 10)),
        (16*x - 3*(x - 6)**2/2 - 176, (x >= 2) & (x >= 4) & (x >= 6) & (x < 10)),
        (100 - 30*x, (x >= 2) & (x >= 4) & (x < 10)),
        (30*x - 3*(x - 6)**2/2 - 196, (x>= 0) & (x >= 4) & (x >= 6) & (x < 10)),
        (80 - 16*x, (x >= 0) & (x >= 4) & (x < 10)),
        (26*x - 3*(x - 6)**2/2 - 196, (x >= 4) & (x >= 6) & (x < 10)),
        (80 - 20*x, (x >= 4) & (x < 10)),
        (40*x - 3*(x - 6)**2/2 - 256, (x >= 0) & (x >= 2) & (x >= 6) & (x < 10)),
        (20 - 6*x, (x >= 0) & (x >= 2) & (x < 10)),
        (36*x - 3*(x - 6)**2/2 - 256, (x >= 2) & (x >= 6) & (x < 10)),
        (20 - 10*x, (x >= 2) & (x < 10)),
        (50*x - 3*(x - 6)**2/2 - 276, (x >= 0) & (x >= 6) & (x < 10)),
        (4*x, (x >= 0) & (x < 10)),
        (46*x - 3*(x - 6)**2/2 - 276, (x >= 6) & (x < 10)),
        (0, x < 10),  # this will simplify away
        (S.NaN,True)))